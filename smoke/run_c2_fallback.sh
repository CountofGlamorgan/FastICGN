#!/usr/bin/env bash
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-${TMPDIR:-/tmp}/fasticgn-smoke-c2}"
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

"${GPU_BIN}" \
  --multi-frame \
  --ref "${WORK_DIR}/data/ref.pgm" \
  --tar-list "${WORK_DIR}/data/tar_list_fallback.txt" \
  --output-dir "${WORK_DIR}/out" \
  15 \
  16 \
  --threads "${THREADS}" \
  --skip-fftcc \
  >"${WORK_DIR}/logs/c2_fallback.log" 2>&1

grep -q "Frames with fallback FFTCC: 1" "${WORK_DIR}/logs/c2_fallback.log"
grep -q "\\[FALLBACK\\]" "${WORK_DIR}/logs/c2_fallback.log"
cat "${WORK_DIR}/logs/c2_fallback.log"
echo "C2 fallback smoke passed"
