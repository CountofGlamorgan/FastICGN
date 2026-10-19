#!/usr/bin/env bash
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="${WORK_DIR:-${TMPDIR:-/tmp}/fasticgn-smoke-c6}"
# shellcheck source=smoke/find_build_binary.sh
source "${SCRIPT_DIR}/find_build_binary.sh"
# The probe links libopencorr_gpu_mvp_core.a out of this tree.
BUILD_DIR="$(fasticgn_require_build_dir libopencorr_gpu_mvp_core.a "${BUILD_DIR:-}" BUILD_DIR)"
CMAKE_CUDA_COMPILER="${CMAKE_CUDA_COMPILER:-/usr/local/cuda-13.2/bin/nvcc}"
COMPUTE_SANITIZER="${COMPUTE_SANITIZER:-/usr/local/cuda-13.2/bin/compute-sanitizer}"

rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"

python3 "${SCRIPT_DIR}/generate_synthetic_dic.py" \
  --out-dir "${WORK_DIR}/data" \
  --width 512 \
  --height 512 \
  --frames 10

cmake -S "${SCRIPT_DIR}/c6_subset_capacity_probe" \
  -B "${WORK_DIR}/build" \
  -DCMAKE_CUDA_COMPILER="${CMAKE_CUDA_COMPILER}" \
  -DOPENCORR_GPU_MVP_SOURCE_DIR="${SRC_DIR}" \
  -DOPENCORR_GPU_MVP_BUILD_DIR="${BUILD_DIR}"
cmake --build "${WORK_DIR}/build" -j"$(nproc)"

if [[ -x "${COMPUTE_SANITIZER}" ]]; then
  "${COMPUTE_SANITIZER}" --error-exitcode 99 \
    "${WORK_DIR}/build/c6_subset_capacity_probe" \
    "${WORK_DIR}/data/ref.pgm" \
    "${WORK_DIR}/data/tar0.pgm"
else
  "${WORK_DIR}/build/c6_subset_capacity_probe" \
    "${WORK_DIR}/data/ref.pgm" \
    "${WORK_DIR}/data/tar0.pgm"
fi

echo "C6 subset capacity smoke passed"
