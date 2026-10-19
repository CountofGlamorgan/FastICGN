#!/usr/bin/env bash
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# FastICGN tiered verification: one command that answers "did I break something".
#
#   scripts/verify.sh --tier static|quick|gpu|lab [--json FILE] [--build-dir DIR]
#                     [--preset NAME] [--python PY] [--jobs N] [--gpu-device N]
#                     [--ctest-label REGEX]
#
#   static  no build: docs sync, licence headers, --describe == the contract,
#           pipeline + MCP unit tests, the smoke unit tests that need no binary
#   quick   static + cpu-only build + ctest + smoke/selfcheck.sh (CPU only)
#   gpu     static + CUDA build (preset cuda-fftw, or --preset cuda-tests)
#           + ctest + smoke/selfcheck.sh (needs a CUDA device)
#   lab     production anchors (smoke/run_production_anchors.sh) on a CUDA build
#           (preset cuda-tests); inputs from ANCHOR_REF, ANCHOR_TAR_LIST_60 and
#           ANCHOR_TAR_LIST_800 (ANCHOR_FFTW_BIN, ANCHOR_GPU_DEVICE, ANCHOR_KEEP_RAW
#           optional); without them the tier reports SKIP, not FAIL
#
# --ctest-label REGEX (quick and gpu tiers) narrows the ctest step to the tests
# whose label matches, as `ctest -L` does: numeric, integrity, unit, cpu, gpu
# (CMakeLists.txt). Without it every registered test runs, which is the gate;
# with it the report records the filter, and a label that matches no test fails.
#
# Builds go into the run directory (<run>/build/<preset>), never into the
# checkout's build/ trees, which the pipeline would otherwise pick up as its
# binary; --build-dir DIR builds there instead. So every run configures and
# builds from scratch (no incremental build). Do not point --build-dir at the
# checkout's build/<preset> to get incremental builds back: a pipeline or MCP
# run without --gpu-bin would then silently use that binary.
# --python picks the interpreter for every Python step (it needs
# pipeline/requirements.txt; fastmcp for the MCP tests). Each step prints its
# command, exit code and duration and keeps its output in a log under a fresh
# run directory (mktemp -d, left in place, not deleted: about 40-60 MB with a
# CPU build, more with CUDA). HOME, XDG_CONFIG_HOME and TMPDIR point into that
# directory and every FASTICGN_* variable is cleared, so no user configuration
# leaks in; PYTHONUSERBASE keeps pointing at the caller's user site, so packages
# installed with pip --user stay importable.
# Exit status: 0 every step passed, 1 a step failed, 77 nothing failed but a
# step was skipped, 130 interrupted, 2 usage error. Each step runs in its own
# process group; on INT or TERM the running step's group gets TERM (KILL after
# 10 s), the step is recorded as INTERRUPTED and later steps do not run. The run
# directory always gets verify-report.json, also when interrupted; --json FILE
# writes a copy (commands, exit codes, durations, logs, git describe --always
# --dirty, and for each tested binary its sha256 and the git_hash, version and
# git_dirty it reports itself).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() { awk 'NR >= 8 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"; }
die_usage() { echo "verify: $*" >&2; echo "usage: scripts/verify.sh --tier static|quick|gpu|lab [options]" >&2; exit 2; }
absolute() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s/%s' "${PWD}" "$1" ;; esac; }

TIER=""; JSON_OUT=""; BUILD_DIR=""; PRESET=""; PY="python3"; JOBS=""; GPU_DEVICE="0"
CTEST_LABEL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier|--json|--build-dir|--preset|--python|--jobs|--gpu-device|--ctest-label)
      [[ $# -ge 2 && -n "$2" ]] || die_usage "$1 needs a value"
      case "$1" in
        --tier) TIER="$2" ;; --json) JSON_OUT="$(absolute "$2")" ;;
        --build-dir) BUILD_DIR="$(absolute "$2")" ;; --preset) PRESET="$2" ;;
        --python) PY="$2" ;; --jobs) JOBS="$2" ;; --gpu-device) GPU_DEVICE="$2" ;;
        --ctest-label) CTEST_LABEL="$2" ;;
      esac
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die_usage "unknown argument: $1" ;;
  esac
done

case "${TIER}" in
  static) PRESET=""; BUILD_DIR="" ;;
  quick) PRESET="${PRESET:-cpu-only}"
         [[ "${PRESET}" == cpu-only ]] || die_usage "the quick tier builds the cpu-only preset" ;;
  gpu)   PRESET="${PRESET:-cuda-fftw}"
         [[ "${PRESET}" == cuda-fftw || "${PRESET}" == cuda-tests ]] \
           || die_usage "the gpu tier builds cuda-fftw or cuda-tests" ;;
  lab)   PRESET="${PRESET:-cuda-tests}"
         [[ "${PRESET}" == cuda-tests || "${PRESET}" == cuda-fftw || "${PRESET}" == cuda-release ]] \
           || die_usage "the lab tier builds cuda-tests, cuda-fftw or cuda-release" ;;
  "") die_usage "--tier is required" ;;
  *) die_usage "unknown tier: ${TIER}" ;;
esac
[[ "${GPU_DEVICE}" =~ ^[0-9]+$ ]] || die_usage "--gpu-device must be a non-negative integer"
if [[ -n "${CTEST_LABEL}" && "${TIER}" != quick && "${TIER}" != gpu ]]; then
  die_usage "--ctest-label applies to the quick and gpu tiers, which run ctest"
fi
if [[ -z "${JOBS}" ]]; then JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"; fi
[[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || die_usage "--jobs must be a positive integer"
PY_ABS="$(command -v -- "${PY}")" || die_usage "python interpreter not found: ${PY}"
PY_ABS="$(absolute "${PY_ABS}")"

# ── a fresh, private environment ─────────────────────────────────────────────
ORIGINAL_TMPDIR="${TMPDIR:-/tmp}"
RUN_DIR="$(mktemp -d "${ORIGINAL_TMPDIR%/}/fasticgn-verify-${TIER}.XXXXXX")" \
  || { echo "verify: cannot create a run directory under ${ORIGINAL_TMPDIR}" >&2; exit 2; }
mkdir -p "${RUN_DIR}/home" "${RUN_DIR}/xdg-config" "${RUN_DIR}/tmp" "${RUN_DIR}/logs"
if [[ "${TIER}" != static && -z "${BUILD_DIR}" ]]; then BUILD_DIR="${RUN_DIR}/build/${PRESET}"; fi
# The user site follows HOME: keep the caller's, so pip --user packages still import.
if [[ -z "${PYTHONUSERBASE:-}" ]]; then
  PYTHONUSERBASE="$("${PY_ABS}" -c 'import site; print(site.getuserbase())' 2>/dev/null || true)"
  [[ -n "${PYTHONUSERBASE}" ]] && export PYTHONUSERBASE
fi
export HOME="${RUN_DIR}/home" XDG_CONFIG_HOME="${RUN_DIR}/xdg-config" TMPDIR="${RUN_DIR}/tmp"
CLEARED=""
while IFS= read -r name; do
  [[ -n "${name}" ]] || continue
  CLEARED+="${name} "
  unset "${name}"
done < <(compgen -e | grep '^FASTICGN_' || true)
cd "${REPO}" || exit 2

STEPS="${RUN_DIR}/steps.jsonl"
: > "${STEPS}"
STEP_INDEX=0
WORST="PASS"
# The step in progress, for the INT/TERM trap (review 2026-10-04 recheck N2).
CURRENT_PID=""
CURRENT_STEP=""
CURRENT_LOG=""
CURRENT_START=""
CURRENT_COMMAND=()
# Runs a step's command as the leader of a new session and process group, so
# the trap can stop the whole step (make, compilers, test subprocesses).
SPAWN_GROUP='import os, sys
try:
    os.setsid()
except OSError:
    pass
os.execvp(sys.argv[1], sys.argv[1:])'

now() {  # seconds since the epoch; EPOCHREALTIME follows the locale's decimal mark
  local stamp="${EPOCHREALTIME:-$(date +%s.%N)}"
  printf '%s' "${stamp/,/.}"
}
T0="$(now)"
seconds_between() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.3f", b - a }'; }

record() {  # record NAME STATUS RC SECONDS LOG NOTE [COMMAND...]
  local name="$1" status="$2" rc="$3" seconds="$4" log="$5" note="$6"
  shift 6
  "${PY_ABS}" - "${STEPS}" "${name}" "${status}" "${rc}" "${seconds}" "${log}" "${note}" "$@" <<'PY'
import json, shlex, sys
path, name, status, rc, seconds, log, note, *command = sys.argv[1:]
entry = {"name": name, "status": status, "rc": int(rc) if rc else None,
         "duration_s": round(float(seconds), 3), "log": log or None, "note": note or None,
         "command": shlex.join(command) if command else None}
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
PY
  case "${status}" in
    FAIL) WORST="FAIL" ;;
    SKIP) [[ "${WORST}" == "PASS" ]] && WORST="SKIP" ;;
  esac
  printf '[verify] %-4s %-30s rc=%-3s %8.1fs  %s\n' "${status}" "${name}" "${rc:--}" "${seconds}" \
    "${note:-${log}}"
}

run_step() {  # run_step NAME COMMAND... -> its exit code; output goes to the step log
  local name="$1"
  shift
  STEP_INDEX=$((STEP_INDEX + 1))
  local log start rc seconds
  log="$(printf '%s/logs/%02d-%s.log' "${RUN_DIR}" "${STEP_INDEX}" "${name}")"
  printf '[verify] run  %-30s %s\n' "${name}" \
    "$("${PY_ABS}" -c 'import shlex, sys; print(shlex.join(sys.argv[1:]))' "$@")"
  start="$(now)"
  # In the background and waited for: bash runs a trap only after a foreground
  # command ends, so a TERM used to wait for the whole step (a CUDA build) and
  # the step was missing from the report (review 2026-10-04 recheck N2).
  "${PY_ABS}" -c "${SPAWN_GROUP}" "$@" > "${log}" 2>&1 < /dev/null &
  CURRENT_PID=$!
  CURRENT_STEP="${name}"
  CURRENT_LOG="${log}"
  CURRENT_START="${start}"
  CURRENT_COMMAND=("$@")
  wait "${CURRENT_PID}"
  rc=$?
  CURRENT_PID=""
  seconds="$(seconds_between "${start}" "$(now)")"
  if [[ ${rc} -eq 0 ]]; then
    record "${name}" PASS "${rc}" "${seconds}" "${log}" "" "$@"
  else
    record "${name}" FAIL "${rc}" "${seconds}" "${log}" "" "$@"
    tail -n 20 "${log}" | sed 's/^/[verify]   | /'
  fi
  return "${rc}"
}

skip_step() {  # skip_step NAME REASON [COMMAND...]
  local name="$1" reason="$2"
  shift 2
  record "${name}" SKIP "" 0 "" "${reason}" "$@"
}

have_gpu() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q '^GPU '; }

echo "[verify] tier ${TIER}, repository ${REPO}"
if [[ -n "${CTEST_LABEL}" ]]; then
  echo "[verify] ctest narrowed to label '${CTEST_LABEL}': a partial run, not the tier's gate"
fi
echo "[verify] python ${PY_ABS}; run directory ${RUN_DIR}"
echo "[verify] HOME, XDG_CONFIG_HOME, TMPDIR -> ${RUN_DIR}; cleared: ${CLEARED:-no FASTICGN_* variables}"

# ── report ───────────────────────────────────────────────────────────────────
write_report() {  # write_report EXIT: verify-report.json (+ --json copy) and the closing lines
EXIT="$1"
SECONDS_TOTAL="$(seconds_between "${T0}" "$(now)")"
GIT_DESCRIBE="$(git -C "${REPO}" describe --always --dirty 2>/dev/null || echo unknown)"
GIT_HEAD="$(git -C "${REPO}" rev-parse HEAD 2>/dev/null || echo unknown)"
REPORT="${RUN_DIR}/verify-report.json"
VERIFY_BINARIES=""
CANDIDATES=()
if [[ -n "${BUILD_DIR}" ]]; then
  CANDIDATES+=("${BUILD_DIR}/opencorr_cli_cuda_mvp" "${BUILD_DIR}/opencorr_cli_cpu")
fi
if [[ "${TIER}" == lab && -n "${ANCHOR_FFTW_BIN:-}" && "${ANCHOR_FFTW_BIN}" != "${BUILD_DIR}/opencorr_cli_cuda_mvp" ]]; then
  CANDIDATES+=("${ANCHOR_FFTW_BIN}")
fi
for candidate in "${CANDIDATES[@]+"${CANDIDATES[@]}"}"; do
  [[ -f "${candidate}" ]] && VERIFY_BINARIES+="${candidate}"$'\n'
done
export VERIFY_BINARIES
"${PY_ABS}" - "${STEPS}" "${REPORT}" "${TIER}" "${WORST}" "${EXIT}" "${SECONDS_TOTAL}" "${REPO}" \
  "${GIT_DESCRIBE}" "${GIT_HEAD}" "${PRESET}" "${BUILD_DIR}" "${PY_ABS}" "${RUN_DIR}" "${CLEARED}" \
  "${CTEST_LABEL}" <<'PY'
import datetime, hashlib, json, os, subprocess, sys
(steps_path, report_path, tier, status, exit_code, seconds, repo, describe, head, preset, build_dir,
 python, run_dir, cleared, ctest_label) = sys.argv[1:]
steps = [json.loads(line) for line in open(steps_path, encoding="utf-8") if line.strip()]
binaries = []
for path in filter(None, os.environ.get("VERIFY_BINARIES", "").splitlines()):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    try:
        version = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30).stdout
        identity = dict(line.split("=", 1) for line in version.splitlines() if "=" in line)
    except (OSError, subprocess.SubprocessError):
        identity = {}
    # version and git_dirty are null for binaries that predate those --version lines.
    binaries.append({"path": path, "sha256": digest.hexdigest(), "git_hash": identity.get("git_hash"),
                     "version": identity.get("version"), "git_dirty": identity.get("git_dirty")})
report = {
    "schema": "fasticgn.verify/1",
    "tier": tier,
    "status": status,
    "exit_code": int(exit_code),
    "duration_s": round(float(seconds), 3),
    "finished_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "repository": repo,
    "git_describe": describe,
    "git_head": head,
    "preset": preset or None,
    "build_dir": build_dir or None,
    "ctest_label": ctest_label or None,
    "binaries": binaries,
    "python": python,
    "python_version": sys.version.split()[0],
    "run_dir": run_dir,
    "environment": {"HOME": os.environ.get("HOME"), "XDG_CONFIG_HOME": os.environ.get("XDG_CONFIG_HOME"),
                    "TMPDIR": os.environ.get("TMPDIR"), "cleared": cleared.split()},
    "steps": steps,
}
with open(report_path, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
for binary in binaries:
    print(f"[verify] binary {binary['path']} sha256 {binary['sha256']} git_hash {binary['git_hash']}"
          f" git_dirty {binary['git_dirty']} version {binary['version']}")
PY
if [[ -n "${JSON_OUT}" ]]; then
  mkdir -p "$(dirname "${JSON_OUT}")" && cp "${REPORT}" "${JSON_OUT}"
fi
echo "[verify] tier ${TIER}: ${WORST} (exit ${EXIT}) in ${SECONDS_TOTAL}s; git ${GIT_DESCRIBE}"
echo "[verify] report ${REPORT}${JSON_OUT:+ (copy: ${JSON_OUT})}; logs ${RUN_DIR}/logs"
}


step_running() {  # step_running PID: the process exists and is not a zombie
  local state
  state="$(ps -o stat= -p "$1" 2>/dev/null | tr -d ' ')"
  [[ -n "${state}" && "${state}" != Z* ]]
}

interrupted() {  # Ctrl-C or TERM: stop the running step, record it and still write the report
  trap - INT TERM
  if [[ -n "${CURRENT_PID}" ]]; then
    local pid="${CURRENT_PID}" tries=0 rc
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null
    while [[ ${tries} -lt 100 ]] && step_running "${pid}"; do
      sleep 0.1
      tries=$((tries + 1))
    done
    kill -KILL -- "-${pid}" 2>/dev/null
    wait "${pid}" 2>/dev/null
    rc=$?
    CURRENT_PID=""
    record "${CURRENT_STEP}" INTERRUPTED "${rc}" "$(seconds_between "${CURRENT_START}" "$(now)")" \
      "${CURRENT_LOG}" "stopped by a signal: its process group got TERM (KILL after 10 s)" \
      "${CURRENT_COMMAND[@]+"${CURRENT_COMMAND[@]}"}"
  fi
  record interrupted FAIL 130 0 "" "interrupted by a signal; later steps did not run"
  write_report 130
  exit 130
}
trap interrupted INT TERM

# ── tiers ────────────────────────────────────────────────────────────────────
static_steps() {
  run_step docs_sync "${PY_ABS}" pipeline/sync_agent_docs.py --check
  if git -C "${REPO}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    run_step licence_headers "${PY_ABS}" smoke/check_license_headers.py
  else
    skip_step licence_headers "not a git checkout: the check reads git ls-files" \
      "${PY_ABS}" smoke/check_license_headers.py
  fi
  run_step describe_matches_contract bash -c \
    '"$0" pipeline/dic_pipeline.py --describe > "$1" && "$0" -c "import json, sys; json.load(open(sys.argv[1]))" "$1" && cmp "$1" pipeline/interface_contract.json' \
    "${PY_ABS}" "${RUN_DIR}/describe.json"
  run_step pipeline_mcp_unittests "${PY_ABS}" -m unittest discover -s pipeline
  local test_file
  for test_file in smoke/test_*.py; do
    [[ -e "${test_file}" ]] || continue
    if grep -Eq "add_argument\(['\"]--binary['\"], *required=True" "${test_file}"; then
      echo "[verify] note ${test_file} needs a built binary: ctest runs it in the quick and gpu tiers"
      continue
    fi
    run_step "smoke_$(basename "${test_file}" .py)" \
      "${PY_ABS}" -m unittest discover -s smoke -p "$(basename "${test_file}")"
  done
}

# The preset's cache variables with an explicit binary directory: -B and
# --test-dir (ctest --preset would still create Testing/ in build/<preset>).
build_steps() {  # configure + build PRESET into BUILD_DIR; returns non-zero when the build is unusable
  if ! run_step "configure_${PRESET}" cmake --preset "${PRESET}" -B "${BUILD_DIR}"; then
    skip_step "build_${PRESET}" "configure failed"
    return 1
  fi
  run_step "build_${PRESET}" cmake --build "${BUILD_DIR}" -j "${JOBS}"
}

test_steps() {  # ctest + selfcheck on BUILD_DIR
  local label_args=()
  if [[ -n "${CTEST_LABEL}" ]]; then label_args=(-L "${CTEST_LABEL}" --no-tests=error); fi
  run_step "ctest_${PRESET}" ctest --test-dir "${BUILD_DIR}" --output-on-failure \
    --output-junit "${RUN_DIR}/logs/ctest-${PRESET}.junit.xml" \
    "${label_args[@]+"${label_args[@]}"}"
  run_step selfcheck bash smoke/selfcheck.sh --build-dir "${BUILD_DIR}" \
    --work-dir "${RUN_DIR}/selfcheck" --python "${PY_ABS}" --gpu-device "${GPU_DEVICE}"
}

case "${TIER}" in
  static)
    static_steps ;;
  quick)
    static_steps
    if build_steps; then test_steps; else skip_step "ctest_${PRESET}" "build failed"; skip_step selfcheck "build failed"; fi ;;
  gpu)
    static_steps
    if ! have_gpu; then
      skip_step "gpu_build_and_tests" "no CUDA device visible (nvidia-smi -L)"
    elif build_steps; then
      test_steps
    else
      skip_step "ctest_${PRESET}" "build failed"
      skip_step selfcheck "build failed"
    fi ;;
  lab)
    missing=""
    for var in ANCHOR_REF ANCHOR_TAR_LIST_60 ANCHOR_TAR_LIST_800; do
      if [[ -z "${!var:-}" ]]; then missing+="${var} unset; "
      elif [[ ! -f "${!var}" ]]; then missing+="${var}=${!var} missing; "; fi
    done
    if [[ -n "${missing}" ]]; then
      skip_step production_anchors "lab data not available: ${missing}see the header of smoke/run_production_anchors.sh"
    elif ! have_gpu; then
      skip_step production_anchors "no CUDA device visible (nvidia-smi -L)"
    elif build_steps; then
      if [[ -z "${ANCHOR_FFTW_BIN:-}" && "${PRESET}" == cuda-fftw ]]; then
        export ANCHOR_FFTW_BIN="${BUILD_DIR}/opencorr_cli_cuda_mvp"
      fi
      run_step production_anchors env ANCHOR_BIN="${BUILD_DIR}/opencorr_cli_cuda_mvp" \
        ANCHOR_OUT_ROOT="${ANCHOR_OUT_ROOT:-${RUN_DIR}/anchors}" \
        ANCHOR_GPU_DEVICE="${ANCHOR_GPU_DEVICE:-${GPU_DEVICE}}" \
        bash smoke/run_production_anchors.sh
      if grep -q '^skipped a800_fftcc_cpu' "$(printf '%s/logs/%02d-production_anchors.log' "${RUN_DIR}" "${STEP_INDEX}")"; then
        skip_step anchor_a800_fftcc_cpu "ANCHOR_FFTW_BIN not set: the --fftcc cpu anchor needs a CUDA+FFTW build"
      fi
    else
      skip_step production_anchors "build failed"
    fi ;;
esac

case "${WORST}" in PASS) FINAL=0 ;; SKIP) FINAL=77 ;; *) FINAL=1 ;; esac
trap - INT TERM
write_report "${FINAL}"
exit "${FINAL}"
