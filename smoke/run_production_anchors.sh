#!/usr/bin/env bash
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Replay the four production raw-output anchors and compare their SHA-256 tree
# hashes against the recorded values. Unlike the synthetic smoke checks this one
# needs a real dataset, so every input is an environment variable and the script
# carries no site-specific default path.
#
#   ANCHOR_BIN            CUDA CLI built without FFTW (default FFTCC backend)
#   ANCHOR_FFTW_BIN       optional CUDA+FFTW CLI, used for the --fftcc cpu anchor
#   ANCHOR_REF            reference image
#   ANCHOR_TAR_LIST_60    60-frame target list
#   ANCHOR_TAR_LIST_800   800-frame target list
#   ANCHOR_OUT_ROOT       scratch output root (default: a mktemp -d directory)
#   ANCHOR_LOG_DIR        CLI logs           (default: $ANCHOR_OUT_ROOT/logs)
#   ANCHOR_GPU_DEVICE     --gpu-device       (default: 0)
#   ANCHOR_THREADS        --threads          (default: 10)
#   ANCHOR_KEEP_RAW=1     keep frame_*.bin instead of deleting them after hashing
#
# Protocol (frozen): subset radius 7, grid 1, --conv-threshold 0.001,
# --skip-fftcc, --output-format raw; tree hash = sha256 of `sha256sum frame_*.bin`.

set -uo pipefail

BIN="${ANCHOR_BIN:-}"
FFTW_BIN="${ANCHOR_FFTW_BIN:-}"
REF="${ANCHOR_REF:-}"
TL60="${ANCHOR_TAR_LIST_60:-}"
TL800="${ANCHOR_TAR_LIST_800:-}"
GPU_DEVICE="${ANCHOR_GPU_DEVICE:-0}"
THREADS="${ANCHOR_THREADS:-10}"
KEEP_RAW="${ANCHOR_KEEP_RAW:-0}"

# Recorded anchors. Override only with new evidence.
#
# Current values date from 2026-10-19: the CUDA precompute kernel now rejects a
# zero-texture reference subset (every sampled value equal) before iterating
# with zncc = -3, through the same exit as a singular Hessian (open-items N-1;
# docs/output-format.md, "Reference-side rejection"). On the reference sequence
# that changes exactly 12 records per frame, all in the camera's black
# information bar at the top of the image, (x,y) = (847,16) (168,31) (327,31)
# (688,40) (176,47) (831,47) (831,48) (319,63) (176,64) (319,64) (672,64)
# (879,64): before, those POIs left the solver as "converged" with zncc exactly
# 1.0 (10 of them, u = v = 0) or within 1e-7 of 0.5 (2 of them); now all 12 are
# -3. Every other record of the 60-frame and 800-frame default trees is
# byte-identical between the two builds (compared record by record on
# 2026-10-19), so every anchor moved and no other output did.
#
# The values each generation replaced are kept as history; a tree that
# reproduces one of those is running older numerics, not a passing gate.
#
#   history (2026-09-17 stage 2E, partially-out-of-bounds rejection, up to 84a832e)
#     60f default    26ddb93ee552919a8f4e01c547d7463b7517e84757fb8dabad3b3b07b1adf4d0
#     60f no-fallback 9b22c4bc39f07e882e0008c727b68447bce6410261c2ec1984e7359d62e0df12
#     800f default   d09619465548c6250775a79c9fe67e5dbb78663e74740dd698a5f761cbc836eb
#     800f fftcc cpu f90740896e0da3aa08fb1fd312d83b005dc6b964c6379eb87524ccba651dab46
#     60f legacy     1d6ab28c78f9edb742a2a4999b332f558c9200a663cc4b6b32c9652e59bfa9c3
#     800f legacy    9b59c132666c301da52863aa2831b00cdba1f06b22f65eb4b663686543aba4a0
#   history (151dcc0 and earlier)
#     60f default    9c79044b0e7eae74093bd409dc99e70d6d5961f6fc02887b53a3c03916a702a3
#     60f no-fallback f12e0cb991692e4c31962a501c5774323d1b31d11ff4101244f48c0b85febf87
#     800f default   2412d5df4c827110a2f3b207b5c962c24d89c676d4eead489acea01be1b18910
#     800f fftcc cpu ff4b2e57d9dbdf93c7d44b7c020a4498555dfac30d0f8463e5fc5846fb38d45d
SHA_60_DEFAULT="${ANCHOR_SHA_60_DEFAULT:-9de34b035eabf21162580100f3f3abf7ffca3a3b384975144275b6edecc28183}"
SHA_60_NOFB="${ANCHOR_SHA_60_NOFB:-c40cbd8b9b66be0a137a6ac3a2cac4e05f061253cbe8f15cb91eaf8e38dec3e2}"
SHA_800_DEFAULT="${ANCHOR_SHA_800_DEFAULT:-3cfea4add5d99363be0bcb4ca6383f776852401a6323e74d28f8b51395b2db12}"
SHA_800_FFTCC_CPU="${ANCHOR_SHA_800_FFTCC_CPU:-86258f57d932028f5d7339839af7436513cdc00709690c80f77d792215543bd5}"
# The legacy block-per-POI kernel is a separate numerics baseline (see
# docs/output-format.md), so it carries its own anchors.
SHA_60_LEGACY="${ANCHOR_SHA_60_LEGACY:-061248cd3a4557f4619f5654e644ec3b472b9e0392a83ff8dfb627c78e067763}"
SHA_800_LEGACY="${ANCHOR_SHA_800_LEGACY:-196b4cd2b282432f4f4bf9618376d4a12a1731321bf7f49d24e79a8c39e9f03f}"

missing=0
for var in ANCHOR_BIN ANCHOR_REF ANCHOR_TAR_LIST_60 ANCHOR_TAR_LIST_800; do
  if [[ -z "${!var:-}" ]]; then
    echo "error: ${var} is not set" >&2
    missing=1
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  echo "error: production anchors need a real dataset; see the header of $0" >&2
  exit 2
fi

OUT_ROOT="${ANCHOR_OUT_ROOT:-$(mktemp -d -t opencorr-anchors-XXXXXX)}"
LOG_DIR="${ANCHOR_LOG_DIR:-${OUT_ROOT}/logs}"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}" || exit 2

ARGS=(7 1 --gpu-device "${GPU_DEVICE}" --threads "${THREADS}"
      --conv-threshold 0.001 --skip-fftcc --output-format raw)

failures=0
printf '%-16s %-4s %-64s %s\n' "anchor" "rc" "sha256" "verdict"

run_anchor() {
  local name="$1" bin="$2" list="$3" expected="$4"
  shift 4
  local out="${OUT_ROOT}/${name}"
  rm -rf "${out}"
  mkdir -p "${out}"
  "${bin}" --multi-frame --ref "${REF}" --tar-list "${list}" \
    --output-dir "${out}" "${ARGS[@]}" "$@" \
    >"${LOG_DIR}/anchor-${name}.log" 2>&1
  local rc=$?
  local hash
  hash="$( cd "${out}" && sha256sum frame_*.bin 2>/dev/null | sha256sum | cut -d' ' -f1 )"
  local verdict="MATCH"
  if [[ "${rc}" -ne 0 || "${hash}" != "${expected}" ]]; then
    verdict="MISMATCH (expected ${expected})"
    failures=$((failures + 1))
  fi
  printf '%-16s %-4s %-64s %s\n' "${name}" "${rc}" "${hash}" "${verdict}"
  if [[ "${KEEP_RAW}" != "1" ]]; then
    find "${out}" -name 'frame_*.bin' -delete
  fi
}

run_anchor a60_default "${BIN}" "${TL60}" "${SHA_60_DEFAULT}"
run_anchor a60_nofb "${BIN}" "${TL60}" "${SHA_60_NOFB}" --no-poi-fallback
run_anchor a800_default "${BIN}" "${TL800}" "${SHA_800_DEFAULT}"
run_anchor a60_legacy "${BIN}" "${TL60}" "${SHA_60_LEGACY}" --kernel legacy
run_anchor a800_legacy "${BIN}" "${TL800}" "${SHA_800_LEGACY}" --kernel legacy
if [[ -n "${FFTW_BIN}" ]]; then
  run_anchor a800_fftcc_cpu "${FFTW_BIN}" "${TL800}" "${SHA_800_FFTCC_CPU}" --fftcc cpu
else
  echo "skipped a800_fftcc_cpu: ANCHOR_FFTW_BIN is not set (needs a CUDA+FFTW build)"
fi

echo "logs: ${LOG_DIR}"
if [[ "${failures}" -ne 0 ]]; then
  echo "ANCHORS FAILED: ${failures}"
  exit 1
fi
echo "ANCHORS OK"
