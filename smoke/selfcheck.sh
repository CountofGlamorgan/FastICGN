#!/usr/bin/env bash
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# FastICGN self-check: one command, synthetic data only.
#
#   bash smoke/selfcheck.sh [--build-dir DIR] [--work-dir DIR] [--gpu-device N]
#                           [--python INTERPRETER]
#
# It generates its own inputs, runs the CLI end to end, and reports PASS/FAIL
# per stage plus an evidence directory. It reads no dataset and no deployment
# path: binaries come from --build-dir, $FASTICGN_GPU_BIN/$FASTICGN_CPU_BIN, or
# the repository's own build/ trees. On a CPU-only build the GPU-specific
# stages are skipped with an explicit note instead of failing, and so is any
# stage whose Python dependencies (pipeline/requirements.txt) are missing.
# A skipped stage was not checked, and the report says how many there were.
# The --subset-size stage and the sub-pixel accuracy stage need only the
# standard library and are never skipped.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_SUBDIRS=(build/cuda-release build/cuda-tests build/cuda-fftw build/cpu-only build)

BUILD_DIR=""
WORK_DIR=""
GPU_DEVICE="${FASTICGN_SELFCHECK_GPU_DEVICE:-0}"
FRAMES=3
WIDTH=160
HEIGHT=128
SUBSET_SIZE=15
# Step 1 on purpose: the CPU reference grid starts at margin radius+1 and the
# FastICGN grid at radius+2, so only step 1 gives the two a shared coordinate
# set to join on.
GRID_SPACE=1
THREADS="${FASTICGN_SELFCHECK_THREADS:-4}"
PYTHON="${FASTICGN_PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --gpu-device) GPU_DEVICE="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    -h|--help) sed -n '8,21p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; exit 2 ;;
  esac
done

find_binary() {  # $1 = binary name, $2 = env override value
  local name="$1" override="${2:-}" candidate
  if [[ -n "${override}" ]]; then printf '%s' "${override}"; return 0; fi
  if [[ -n "${BUILD_DIR}" && -x "${BUILD_DIR}/${name}" ]]; then
    printf '%s' "${BUILD_DIR}/${name}"; return 0
  fi
  for sub in "${BUILD_SUBDIRS[@]}"; do
    candidate="${REPO_ROOT}/${sub}/${name}"
    if [[ -x "${candidate}" ]]; then printf '%s' "${candidate}"; return 0; fi
  done
  return 1
}

GPU_BIN="$(find_binary opencorr_cli_cuda_mvp "${FASTICGN_GPU_BIN:-}")" || {
  echo "error: opencorr_cli_cuda_mvp not found." >&2
  echo "       Build it (cmake --preset cuda-release, or cpu-only), then pass" >&2
  echo "       --build-dir <dir> or set FASTICGN_GPU_BIN." >&2
  exit 1
}
CPU_BIN="$(find_binary opencorr_cli_cpu "${FASTICGN_CPU_BIN:-}")" || CPU_BIN=""

if [[ -z "${WORK_DIR}" ]]; then
  WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fasticgn-selfcheck-XXXXXX")"
else
  mkdir -p "${WORK_DIR}"
fi
mkdir -p "${WORK_DIR}/logs"

# A CUDA build links libcudart; a CPU-only build does not. A CUDA build still
# needs a visible device before the GPU stages mean anything.
# The match is done by the shell, with no pipe. `... | grep -q` exits at the
# first match, and under `set -o pipefail` the writer's SIGPIPE then counts as
# "no match"; feeding grep from printf instead of from ldd still lost that race
# in about 1 % of runs on a busy host (24 of 3000, 2026-10-05), which reported
# a CUDA build as CPU-only and turned its GPU stages into SKIP.
CUDA_BUILD=0
CUDA_LINKAGE="$(ldd "${GPU_BIN}" 2>/dev/null || true)"
case "${CUDA_LINKAGE}" in *libcudart*) CUDA_BUILD=1 ;; esac
HAVE_GPU=0
if [[ "${CUDA_BUILD}" -eq 1 ]] && command -v nvidia-smi >/dev/null 2>&1 \
   && nvidia-smi -L >/dev/null 2>&1; then HAVE_GPU=1; fi

declare -a RESULT_NAMES=() RESULT_STATES=() RESULT_NOTES=()
record() { RESULT_NAMES+=("$1"); RESULT_STATES+=("$2"); RESULT_NOTES+=("${3:-}"); }

echo "FastICGN self-check"
echo "  repository : ${REPO_ROOT}"
echo "  gpu cli    : ${GPU_BIN}"
echo "  cpu cli    : ${CPU_BIN:-<not built>}"
echo "  python     : ${PYTHON}"
echo "  build type : $([[ ${CUDA_BUILD} -eq 1 ]] && echo CUDA || echo CPU-only)"
echo "  gpu device : $([[ ${HAVE_GPU} -eq 1 ]] && echo "${GPU_DEVICE}" || echo none)"
echo "  evidence   : ${WORK_DIR}"
echo

# ── 1. version probe ────────────────────────────────────────────────────────
if "${GPU_BIN}" --version > "${WORK_DIR}/logs/version.log" 2>&1; then
  record "version probe" PASS "$(grep -m1 '^git_hash=' "${WORK_DIR}/logs/version.log" || true)"
else
  record "version probe" FAIL "see logs/version.log"
fi

HAVE_NUMPY=0
if "${PYTHON}" -c 'import numpy' >/dev/null 2>&1; then HAVE_NUMPY=1; fi

# ── 2. synthetic inputs ─────────────────────────────────────────────────────
if "${PYTHON}" "${SCRIPT_DIR}/generate_synthetic_dic.py" --out-dir "${WORK_DIR}/data" \
     --width "${WIDTH}" --height "${HEIGHT}" --frames "${FRAMES}" \
     > "${WORK_DIR}/logs/generate.log" 2>&1; then
  record "synthetic inputs" PASS "${WORK_DIR}/data"
else
  record "synthetic inputs" FAIL "see logs/generate.log"
  printf '\nSELF-CHECK FAIL: cannot generate inputs\n'
  exit 1
fi

# ── 3. three-frame sequence, raw output, read back through fasticgn_io ──────
SEQ_DIR="${WORK_DIR}/sequence-raw"
mkdir -p "${SEQ_DIR}"
: > "${WORK_DIR}/data/tar_list_selfcheck.txt"
for ((frame = 0; frame < FRAMES; ++frame)); do
  printf '%s/data/frame_%06d.pgm frame_%06d.csv\n' "${WORK_DIR}" "${frame}" "${frame}" \
    >> "${WORK_DIR}/data/tar_list_selfcheck.txt"
done
seq_args=(--multi-frame --ref "${WORK_DIR}/data/ref.pgm"
          --tar-list "${WORK_DIR}/data/tar_list_selfcheck.txt"
          --output-dir "${SEQ_DIR}" --subset-size "${SUBSET_SIZE}" "${GRID_SPACE}"
          --threads "${THREADS}" --conv-threshold 0.001 --output-format raw)
[[ "${HAVE_GPU}" -eq 1 ]] && seq_args+=(--gpu-device "${GPU_DEVICE}")
if "${GPU_BIN}" "${seq_args[@]}" > "${WORK_DIR}/logs/sequence.log" 2>&1; then
  record "3-frame raw sequence" PASS "${SEQ_DIR}"
else
  record "3-frame raw sequence" FAIL "see logs/sequence.log"
fi

# --subset-size takes a SIZE: the sequence above must have run with radius
# (size - 1) / 2. The CLI prints the radius it used and records it in the raw
# manifest. A CLI that took the size as a radius used to pass this script,
# because every later stage only compares coordinates both sides share
# (review 2026-10-03, section 2.4).
EXPECTED_RADIUS=$(( (SUBSET_SIZE - 1) / 2 ))
SUBSET_STAGE="--subset-size ${SUBSET_SIZE} is radius ${EXPECTED_RADIUS}"
if grep -q "^Subset radius: ${EXPECTED_RADIUS}, " "${WORK_DIR}/logs/sequence.log" 2>/dev/null \
   && grep -Eq "\"subset_radius\": *${EXPECTED_RADIUS}," \
        "${SEQ_DIR}/manifest.opencorr-raw.json" 2>/dev/null; then
  record "${SUBSET_STAGE}" PASS "CLI log and raw manifest"
else
  record "${SUBSET_STAGE}" FAIL \
    "CLI says '$(grep -m1 '^Subset radius:' "${WORK_DIR}/logs/sequence.log" 2>/dev/null)'; see logs/sequence.log"
fi

# The reader check is its own file so pipeline/fasticgn_io.py is exercised the
# same way a downstream user would import it.
if [[ "${HAVE_NUMPY}" -eq 0 ]]; then
  record "raw reader + manifest engine block" SKIP \
    "numpy missing: ${PYTHON} -m pip install -r pipeline/requirements.txt"
elif PYTHONPATH="${REPO_ROOT}/pipeline${PYTHONPATH:+:${PYTHONPATH}}" \
   "${PYTHON}" "${SCRIPT_DIR}/selfcheck_read_raw.py" --output-dir "${SEQ_DIR}" \
     --expect-frames "${FRAMES}" > "${WORK_DIR}/logs/raw_reader.log" 2>&1; then
  record "raw reader + manifest engine block" PASS "$(tail -n1 "${WORK_DIR}/logs/raw_reader.log")"
else
  record "raw reader + manifest engine block" FAIL "see logs/raw_reader.log"
fi

# ── 4. cache regression (known-truth sequences); GPU builds only ────────────
if [[ "${HAVE_GPU}" -eq 1 ]]; then
  if "${PYTHON}" "${SCRIPT_DIR}/run_cache_regression.py" --binary "${GPU_BIN}" \
       --gpu-device "${GPU_DEVICE}" --threads "${THREADS}" \
       --work-dir "${WORK_DIR}/cache-regression" \
       > "${WORK_DIR}/logs/cache_regression.log" 2>&1; then
    record "cache regression" PASS "${WORK_DIR}/cache-regression"
  else
    record "cache regression" FAIL "see logs/cache_regression.log"
  fi
else
  record "cache regression" SKIP "CPU-only build or no CUDA device"
fi

# ── 5. CPU reference and CPU/GPU parity ─────────────────────────────────────
if [[ -z "${CPU_BIN}" ]]; then
  record "CPU/GPU parity" SKIP "opencorr_cli_cpu not built (use the cpu-only or cuda-fftw preset)"
else
  mkdir -p "${WORK_DIR}/cpu" "${WORK_DIR}/gpu"
  cpu_ok=1
  for ((frame = 0; frame < FRAMES; ++frame)); do
    name="$(printf 'frame_%06d' "${frame}")"
    if ! "${CPU_BIN}" "${WORK_DIR}/data/ref.pgm" "${WORK_DIR}/data/${name}.pgm" \
         "$(( (SUBSET_SIZE - 1) / 2 ))" "${GRID_SPACE}" "${WORK_DIR}/cpu/${name}.csv" \
         --threads "${THREADS}" > "${WORK_DIR}/logs/cpu_${name}.log" 2>&1; then
      cpu_ok=0
      break
    fi
  done
  gpu_args=(--multi-frame --ref "${WORK_DIR}/data/ref.pgm"
            --tar-list "${WORK_DIR}/data/tar_list_selfcheck.txt"
            --output-dir "${WORK_DIR}/gpu" --subset-size "${SUBSET_SIZE}" "${GRID_SPACE}"
            --threads "${THREADS}" --max-iter 10 --conv-threshold 0.001 --no-skip-fftcc)
  [[ "${HAVE_GPU}" -eq 1 ]] && gpu_args+=(--gpu-device "${GPU_DEVICE}")
  if [[ "${cpu_ok}" -eq 1 ]] \
     && "${GPU_BIN}" "${gpu_args[@]}" > "${WORK_DIR}/logs/gpu_parity.log" 2>&1 \
     && "${PYTHON}" "${SCRIPT_DIR}/check_cpu_gpu_parity.py" \
          --cpu-dir "${WORK_DIR}/cpu" --gpu-dir "${WORK_DIR}/gpu" \
          --frames "${FRAMES}" --median-threshold 1e-3 \
          > "${WORK_DIR}/logs/parity.log" 2>&1; then
    label="CPU/GPU parity"
    [[ "${HAVE_GPU}" -eq 1 ]] || label="CPU/CPU parity (CPU-only build)"
    record "${label}" PASS "$(tail -n1 "${WORK_DIR}/logs/parity.log")"
  else
    record "CPU/GPU parity" FAIL "see logs/parity.log, logs/gpu_parity.log, logs/cpu_*.log"
  fi
fi

# ── out-of-bounds rejection rule ─────────────────────────────────────────────
if "${PYTHON}" "${SCRIPT_DIR}/run_oob_rejection_gate.py" \
    --binary "${GPU_BIN}" --gpu-device "${GPU_DEVICE}" \
    --work-dir "${WORK_DIR}/oob-rejection" >"${WORK_DIR}/logs/oob_rejection.log" 2>&1; then
  record "out-of-bounds rejection" PASS "$(grep -m1 '^binary:' "${WORK_DIR}/logs/oob_rejection.log" || true)"
else
  record "out-of-bounds rejection" FAIL "$(tail -n2 "${WORK_DIR}/logs/oob_rejection.log" | tr '\n' ' ')"
fi

# ── sub-pixel accuracy on noisy speckle, and the CLI's numeric defaults ──────
# Known sub-pixel truth with sensor noise, so ICGN iterates for real; also the
# default --conv-threshold / --max-iter and --subset-size against the same
# values spelled out. Standard-library Python: this stage is never skipped.
SUBPIXEL_DIR="${WORK_DIR}/subpixel-accuracy"
if [[ -e "${SUBPIXEL_DIR}" ]]; then
  SUBPIXEL_DIR="$(mktemp -d "${WORK_DIR}/subpixel-accuracy-XXXXXX")"
fi
if "${PYTHON}" "${SCRIPT_DIR}/run_subpixel_accuracy_gate.py" \
    --binary "${GPU_BIN}" --gpu-device "${GPU_DEVICE}" --threads "${THREADS}" \
    --work-dir "${SUBPIXEL_DIR}" > "${WORK_DIR}/logs/subpixel_accuracy.log" 2>&1; then
  record "sub-pixel accuracy + CLI defaults" PASS \
    "$(grep -c '^case .*: PASS' "${WORK_DIR}/logs/subpixel_accuracy.log") cases, ${SUBPIXEL_DIR}"
else
  record "sub-pixel accuracy + CLI defaults" FAIL \
    "$(grep -m2 '^FAIL' "${WORK_DIR}/logs/subpixel_accuracy.log" | tr '\n' ' '); see logs/subpixel_accuracy.log"
fi

# ── report ──────────────────────────────────────────────────────────────────
echo
failed=0
skipped=0
for index in "${!RESULT_NAMES[@]}"; do
  printf '  %-6s %-38s %s\n' "${RESULT_STATES[$index]}" "${RESULT_NAMES[$index]}" \
    "${RESULT_NOTES[$index]}"
  [[ "${RESULT_STATES[$index]}" == "FAIL" ]] && failed=1
  [[ "${RESULT_STATES[$index]}" == "SKIP" ]] && skipped=$((skipped + 1))
done
echo
echo "Evidence: ${WORK_DIR}"
if [[ "${skipped}" -gt 0 ]]; then
  echo "Skipped: ${skipped} stage(s) did not run and were not checked (reasons above)."
fi
if [[ "${failed}" -eq 0 ]]; then
  echo "SELF-CHECK PASS"
  exit 0
fi
echo "SELF-CHECK FAIL"
exit 1
