#!/usr/bin/env bash
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-${TMPDIR:-/tmp}/fasticgn-smoke-c5}"
# shellcheck source=smoke/find_build_binary.sh
source "${SCRIPT_DIR}/find_build_binary.sh"
GPU_BIN="$(fasticgn_require_binary opencorr_cli_cuda_mvp "${GPU_BIN:-}" GPU_BIN)"
# CPU reference binary: OPENCORR_CPU_BIN wins, then the legacy CPU_BIN
# override, then the opencorr_cli_cpu sitting next to GPU_BIN (built by any
# OPENCORR_GPU_MVP_WITH_FFTW=ON configuration, including every CPU-only one).
CPU_BIN="${OPENCORR_CPU_BIN:-${CPU_BIN:-}}"
if [[ -z "${CPU_BIN}" ]]; then
  CPU_BIN="$(dirname "${GPU_BIN}")/opencorr_cli_cpu"
fi
if [[ ! -x "${CPU_BIN}" ]] && CPU_BIN_DIR="$(fasticgn_find_build_dir opencorr_cli_cpu -x)"; then
  CPU_BIN="${CPU_BIN_DIR}/opencorr_cli_cpu"
fi
if [[ ! -x "${CPU_BIN}" ]]; then
  echo "error: CPU reference CLI not found at '${CPU_BIN}'." >&2
  echo "       Build target opencorr_cli_cpu (cmake --preset cuda-fftw, or cpu-only)," >&2
  echo "       or set OPENCORR_CPU_BIN to an existing OpenCorr CLI." >&2
  exit 1
fi
THREADS="${THREADS:-4}"
WIDTH=160
HEIGHT=128
SUBSET_RADIUS=15
GRID_SPACE=1
DX=10

rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}/out" "${WORK_DIR}/logs"

python3 "${SCRIPT_DIR}/generate_synthetic_dic.py" \
  --out-dir "${WORK_DIR}/data" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --frames 10

"${CPU_BIN}" \
  "${WORK_DIR}/data/ref.pgm" \
  "${WORK_DIR}/data/tar_oob_dx10.pgm" \
  "${SUBSET_RADIUS}" \
  "${GRID_SPACE}" \
  "${WORK_DIR}/out/cpu_oob.csv" \
  --threads "${THREADS}" \
  >"${WORK_DIR}/logs/cpu_oob.log" 2>&1

"${GPU_BIN}" \
  "${WORK_DIR}/data/ref.pgm" \
  "${WORK_DIR}/data/tar_oob_dx10.pgm" \
  "${SUBSET_RADIUS}" \
  "${GRID_SPACE}" \
  "${WORK_DIR}/out/gpu_oob.csv" \
  --threads "${THREADS}" \
  --max-iter 10 \
  --conv-threshold 0.001 \
  >"${WORK_DIR}/logs/gpu_oob.log" 2>&1

python3 "${SCRIPT_DIR}/check_oob_boundary.py" \
  --cpu-csv "${WORK_DIR}/out/cpu_oob.csv" \
  --gpu-csv "${WORK_DIR}/out/gpu_oob.csv" \
  --subset-radius "${SUBSET_RADIUS}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --dx "${DX}" \
  --dy 0

echo "C5 OOB boundary smoke completed; both backends reject a partially out-of-bounds target subset with zncc=-3"
