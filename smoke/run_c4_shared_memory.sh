#!/usr/bin/env bash
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-${TMPDIR:-/tmp}/fasticgn-smoke-c4}"
# shellcheck source=smoke/find_build_binary.sh
source "${SCRIPT_DIR}/find_build_binary.sh"
GPU_BIN="$(fasticgn_require_binary opencorr_cli_cuda_mvp "${GPU_BIN:-}" GPU_BIN)"
THREADS="${THREADS:-4}"

rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}/out" "${WORK_DIR}/logs"

python3 "${SCRIPT_DIR}/generate_synthetic_dic.py" \
  --out-dir "${WORK_DIR}/data" \
  --width 512 \
  --height 512 \
  --frames 10

for radius in 20 25 31; do
  "${GPU_BIN}" \
    "${WORK_DIR}/data/ref.pgm" \
    "${WORK_DIR}/data/tar0.pgm" \
    "${radius}" \
    64 \
    "${WORK_DIR}/out/s$((2 * radius + 1)).csv" \
    --threads "${THREADS}" \
    >"${WORK_DIR}/logs/c4_s$((2 * radius + 1)).log" 2>&1
done

grep "CUDA ICGN launch config" "${WORK_DIR}"/logs/c4_*.log
if grep -q "dynamic shared-memory opt-in applied" "${WORK_DIR}"/logs/c4_*.log; then
  echo "C4 opt-in branch exercised"
else
  echo "C4 opt-in branch not exercised on this GPU; inspect logs for selected block sizes and limits"
fi
