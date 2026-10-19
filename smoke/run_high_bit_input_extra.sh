#!/usr/bin/env bash
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Extra high-bit input regression (2026-07-10):
#   1) spaced-path tar-list end-to-end (space- and tab-separated fields)
#   2) --input-depth auto inference anchors, incl. documented misjudgment cases
#   3) --input-normalize to-8 / to-16 / unit output ranges
# Complements run_high_bit_input.sh. Anchors documented behavior — some anchored
# outcomes are known limitations (see README input-depth caveat); they are
# asserted here so any future behavior change must be made deliberately.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-${TMPDIR:-/tmp}/fasticgn-smoke-high-bit-extra}"
# shellcheck source=smoke/find_build_binary.sh
source "${SCRIPT_DIR}/find_build_binary.sh"
BUILD_DIR="$(fasticgn_require_build_dir image_input_depth_probe "${BUILD_DIR:-}" BUILD_DIR)"
GPU_BIN="${GPU_BIN:-${BUILD_DIR}/opencorr_cli_cuda_mvp}"
PROBE_BIN="${PROBE_BIN:-${BUILD_DIR}/image_input_depth_probe}"

[[ -x "${PROBE_BIN}" ]] || { echo "missing image_input_depth_probe: ${PROBE_BIN}" >&2; exit 1; }
[[ -x "${GPU_BIN}" ]] || { echo "missing opencorr_cli_cuda_mvp: ${GPU_BIN}" >&2; exit 1; }

rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}/logs" "${WORK_DIR}/out-space" "${WORK_DIR}/dir with space"

python3 - "${WORK_DIR}" <<'PY'
from pathlib import Path
import random
import struct
import sys

work = Path(sys.argv[1])
spaced = work / "dir with space"

def write_pgm16(path, width, height, values):
    with Path(path).open("wb") as handle:
        handle.write(f"P5\n{width} {height}\n65535\n".encode("ascii"))
        for value in values:
            handle.write(struct.pack(">H", int(value) & 0xFFFF))

values12 = [0, 1, 255, 256, 4095]
dark12 = [0, 1, 15, 128, 255]
write_pgm16(work / "values12_right.pgm", len(values12), 1, values12)
write_pgm16(work / "bright_left.pgm", len(values12), 1, [v << 4 for v in values12])
write_pgm16(work / "dark_left.pgm", len(dark12), 1, [v << 4 for v in dark12])
write_pgm16(work / "values16.pgm", 5, 1, [0, 1, 4095, 32768, 65535])

# 96x96 12-bit left-shifted speckle pair in a spaced dir, with spaced filenames
width, height = 96, 96
rng = random.Random(20260710)
base12 = [(x * 37 + y * 53 + rng.randrange(0, 512)) % 4096
          for y in range(height) for x in range(width)]
left12 = [v << 4 for v in base12]
write_pgm16(spaced / "speckle ref.pgm", width, height, left12)
write_pgm16(spaced / "speckle tar 01.pgm", width, height, left12)
write_pgm16(spaced / "speckle tar 02.pgm", width, height, left12)

# frame 1: space-separated fields; frame 2: tab-separated fields
with (work / "tar_list_space.txt").open("w", encoding="utf-8") as handle:
    handle.write(f"{spaced / 'speckle tar 01.pgm'} frame_000001.csv\n")
    handle.write(f"{spaced / 'speckle tar 02.pgm'}\tframe_000002.csv\n")
PY

probe_field() {  # probe_field <field> <probe-args...>
  local field="$1"; shift
  "${PROBE_BIN}" "$@" | tee -a "${WORK_DIR}/logs/probe.log" \
    | awk -F= -v f="${field}" '$1==f{print $2}'
}

expect() {  # expect <label> <got> <want>
  [[ "$2" == "$3" ]] || { echo "FAIL $1: got '$2', want '$3'" >&2; exit 1; }
  echo "ok $1: $2"
}

# --- 2) --input-depth auto anchors ------------------------------------------
# Bright left-shifted (stored max > 4095) -> inferred 16, values pass through raw.
expect "auto/bright-left depth"  "$(probe_field effective_depth --input-depth auto "${WORK_DIR}/bright_left.pgm")" "16"
expect "auto/bright-left values" "$(probe_field values          --input-depth auto "${WORK_DIR}/bright_left.pgm")" "0,16,4080,4096,65520"
# True 12-bit right-aligned (max = 4095) -> inferred 12, align auto -> right.
expect "auto/right12 depth"  "$(probe_field effective_depth --input-depth auto "${WORK_DIR}/values12_right.pgm")" "12"
expect "auto/right12 align"  "$(probe_field effective_align --input-depth auto "${WORK_DIR}/values12_right.pgm")" "right"
expect "auto/right12 values" "$(probe_field values          --input-depth auto "${WORK_DIR}/values12_right.pgm")" "0,1,255,256,4095"
# DOCUMENTED LIMITATION anchor: dark left-shifted data (stored <= 4095) is kept
# right-aligned, values NOT divided by 16 (true values were 0,1,15,128,255).
expect "auto/dark-left depth (documented misjudgment)"  "$(probe_field effective_depth --input-depth auto "${WORK_DIR}/dark_left.pgm")" "12"
expect "auto/dark-left align (documented misjudgment)"  "$(probe_field effective_align --input-depth auto "${WORK_DIR}/dark_left.pgm")" "right"
expect "auto/dark-left values (documented misjudgment)" "$(probe_field values          --input-depth auto "${WORK_DIR}/dark_left.pgm")" "0,16,240,2048,4080"
# Same misjudgment through explicit 12 + auto alignment.
expect "12+auto-align/dark-left values (documented misjudgment)" "$(probe_field values --input-depth 12 --input-align auto "${WORK_DIR}/dark_left.pgm")" "0,16,240,2048,4080"

# --- 3) --input-normalize -----------------------------------------------------
expect "to-8/16-bit range"  "$(probe_field output_range --input-depth 16 --input-normalize to-8  "${WORK_DIR}/values16.pgm")" "0,255"
expect "to-16/16-bit range" "$(probe_field output_range --input-depth 16 --input-normalize to-16 "${WORK_DIR}/values16.pgm")" "0,65535"
expect "unit/16-bit range"  "$(probe_field output_range --input-depth 16 --input-normalize unit  "${WORK_DIR}/values16.pgm")" "0,1"
expect "to-8/12-left range" "$(probe_field output_range --input-depth 12 --input-align left --input-normalize to-8 "${WORK_DIR}/bright_left.pgm")" "0,255"

# --- 1) spaced-path tar-list end-to-end ---------------------------------------
"${GPU_BIN}" \
  --multi-frame \
  --ref "${WORK_DIR}/dir with space/speckle ref.pgm" \
  --tar-list "${WORK_DIR}/tar_list_space.txt" \
  --output-dir "${WORK_DIR}/out-space" \
  7 16 \
  --threads 2 --max-iter 5 --conv-threshold 0.001 --no-skip-fftcc \
  --output-format csv \
  --input-depth 12 --input-align left \
  >"${WORK_DIR}/logs/cli_space.log" 2>&1

grep -q "Input image mode: depth=12, align=left" "${WORK_DIR}/logs/cli_space.log"
for f in frame_000001.csv frame_000002.csv; do
  csv="${WORK_DIR}/out-space/${f}"
  [[ -s "${csv}" ]] || { echo "FAIL spaced-path: missing/empty ${csv}" >&2; exit 1; }
  rows=$(wc -l < "${csv}")
  (( rows >= 25 )) || { echo "FAIL spaced-path: ${f} only ${rows} rows" >&2; exit 1; }
  echo "ok spaced-path ${f}: ${rows} rows"
done

echo "High-bit extra smoke passed; logs: ${WORK_DIR}/logs"
