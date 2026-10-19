#!/usr/bin/env bash
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-${TMPDIR:-/tmp}/fasticgn-smoke-v9}"
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
SUBSET_RADIUS="${SUBSET_RADIUS:-15}"
GRID_SPACE="${GRID_SPACE:-1}"
FRAMES="${FRAMES:-10}"

rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}/cpu" "${WORK_DIR}/gpu" "${WORK_DIR}/logs"

python3 "${SCRIPT_DIR}/generate_synthetic_dic.py" \
  --out-dir "${WORK_DIR}/data" \
  --width 160 \
  --height 128 \
  --frames "${FRAMES}"

for ((frame = 0; frame < FRAMES; ++frame)); do
  name="$(printf 'frame_%06d' "${frame}")"
  "${CPU_BIN}" \
    "${WORK_DIR}/data/ref.pgm" \
    "${WORK_DIR}/data/${name}.pgm" \
    "${SUBSET_RADIUS}" \
    "${GRID_SPACE}" \
    "${WORK_DIR}/cpu/${name}.csv" \
    --threads "${THREADS}" \
    >"${WORK_DIR}/logs/cpu_${name}.log" 2>&1
done

"${GPU_BIN}" \
  --multi-frame \
  --ref "${WORK_DIR}/data/ref.pgm" \
  --tar-list "${WORK_DIR}/data/tar_list_10.txt" \
  --output-dir "${WORK_DIR}/gpu" \
  "${SUBSET_RADIUS}" \
  "${GRID_SPACE}" \
  --threads "${THREADS}" \
  --max-iter 10 \
  --conv-threshold 0.001 \
  --no-skip-fftcc \
  >"${WORK_DIR}/logs/gpu_v9.log" 2>&1

python3 "${SCRIPT_DIR}/check_cpu_gpu_parity.py" \
  --cpu-dir "${WORK_DIR}/cpu" \
  --gpu-dir "${WORK_DIR}/gpu" \
  --frames "${FRAMES}" \
  --median-threshold 1e-3

echo "V9 parity smoke passed; logs: ${WORK_DIR}/logs"
