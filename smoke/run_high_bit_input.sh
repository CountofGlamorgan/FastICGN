#!/usr/bin/env bash
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-${TMPDIR:-/tmp}/fasticgn-smoke-high-bit-input}"
# shellcheck source=smoke/find_build_binary.sh
source "${SCRIPT_DIR}/find_build_binary.sh"
BUILD_DIR="$(fasticgn_require_build_dir image_input_depth_probe "${BUILD_DIR:-}" BUILD_DIR)"
GPU_BIN="${GPU_BIN:-${BUILD_DIR}/opencorr_cli_cuda_mvp}"
PROBE_BIN="${PROBE_BIN:-${BUILD_DIR}/image_input_depth_probe}"

if [[ ! -x "${PROBE_BIN}" ]]; then
  echo "missing image_input_depth_probe: ${PROBE_BIN}" >&2
  exit 1
fi
if [[ ! -x "${GPU_BIN}" ]]; then
  echo "missing opencorr_cli_cuda_mvp: ${GPU_BIN}" >&2
  exit 1
fi

rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}/data" "${WORK_DIR}/logs" "${WORK_DIR}/out-preload" "${WORK_DIR}/out-sync"

python3 - "${WORK_DIR}/data" <<'PY'
from pathlib import Path
import random
import struct
import sys

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

def write_pgm16(path, width, height, values):
    with Path(path).open("wb") as handle:
        handle.write(f"P5\n{width} {height}\n65535\n".encode("ascii"))
        for value in values:
            handle.write(struct.pack(">H", int(value) & 0xFFFF))

def write_tar_list(path, entries):
    with Path(path).open("w", encoding="utf-8") as handle:
        for image, name in entries:
            handle.write(f"{image} {name}\n")

values16 = [0, 1, 255, 256, 4095, 32768, 65535]
values12 = [0, 1, 255, 256, 4095]
write_pgm16(out / "values16.pgm", len(values16), 1, values16)
write_pgm16(out / "values12_right.pgm", len(values12), 1, values12)
write_pgm16(out / "values12_left.pgm", len(values12), 1, [v << 4 for v in values12])

width, height = 96, 96
rng = random.Random(20260708)
base12 = []
for y in range(height):
    for x in range(width):
        value = (x * 37 + y * 53 + rng.randrange(0, 512)) % 4096
        base12.append(value)

base16 = [(value * 16 + (idx % 16)) & 0xFFFF for idx, value in enumerate(base12)]
write_pgm16(out / "ref16.pgm", width, height, base16)
write_pgm16(out / "tar16.pgm", width, height, base16)
write_tar_list(out / "tar_list16.txt", [(out / "tar16.pgm", "frame_000000.csv")])

left12 = [value << 4 for value in base12]
write_pgm16(out / "ref12_left.pgm", width, height, left12)
write_pgm16(out / "tar12_left.pgm", width, height, left12)
write_tar_list(out / "tar_list12_left.txt", [(out / "tar12_left.pgm", "frame_000000.csv")])
PY

probe_values() {
  "${PROBE_BIN}" "$@" | tee -a "${WORK_DIR}/logs/probe.log" | awk -F= '/^values=/{print $2}'
}

legacy_values="$(probe_values --legacy-load "${WORK_DIR}/data/values16.pgm")"
[[ "${legacy_values}" == "0,0,0,1,15,128,255" ]] \
  || { echo "legacy 16-bit downcast mismatch: ${legacy_values}" >&2; exit 1; }

compat_values="$(probe_values --input-depth 8 "${WORK_DIR}/data/values16.pgm")"
[[ "${compat_values}" == "${legacy_values}" ]] \
  || { echo "default 8-bit compatibility mismatch: ${compat_values}" >&2; exit 1; }

depth16_values="$(probe_values --input-depth 16 "${WORK_DIR}/data/values16.pgm")"
[[ "${depth16_values}" == "0,1,255,256,4095,32768,65535" ]] \
  || { echo "16-bit values mismatch: ${depth16_values}" >&2; exit 1; }

right12_values="$(probe_values --input-depth 12 --input-align right "${WORK_DIR}/data/values12_right.pgm")"
[[ "${right12_values}" == "0,1,255,256,4095" ]] \
  || { echo "12-bit right values mismatch: ${right12_values}" >&2; exit 1; }

left12_values="$(probe_values --input-depth 12 --input-align left "${WORK_DIR}/data/values12_left.pgm")"
[[ "${left12_values}" == "0,1,255,256,4095" ]] \
  || { echo "12-bit left values mismatch: ${left12_values}" >&2; exit 1; }

auto12_values="$(probe_values --input-depth 12 --input-align auto "${WORK_DIR}/data/values12_left.pgm")"
[[ "${auto12_values}" == "0,1,255,256,4095" ]] \
  || { echo "12-bit auto-align values mismatch: ${auto12_values}" >&2; exit 1; }

"${GPU_BIN}" \
  --multi-frame \
  --ref "${WORK_DIR}/data/ref16.pgm" \
  --tar-list "${WORK_DIR}/data/tar_list16.txt" \
  --output-dir "${WORK_DIR}/out-preload" \
  7 \
  16 \
  --threads 2 \
  --max-iter 5 \
  --conv-threshold 0.001 \
  --no-skip-fftcc \
  --output-format raw \
  --input-depth 16 \
  >"${WORK_DIR}/logs/cli_preload_16.log" 2>&1

python3 - "${WORK_DIR}/out-preload/manifest.opencorr-raw.json" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
input_image = manifest["input_image"]
assert input_image["depth"] == "16", input_image
assert input_image["normalize"] == "none", input_image
assert input_image["imread_flags"] == "IMREAD_ANYDEPTH|IMREAD_GRAYSCALE", input_image
assert len(manifest["frames"]) == 1, manifest["frames"]
print("preload manifest input_image:", input_image)
PY

"${GPU_BIN}" \
  --multi-frame \
  --ref "${WORK_DIR}/data/ref12_left.pgm" \
  --tar-list "${WORK_DIR}/data/tar_list12_left.txt" \
  --output-dir "${WORK_DIR}/out-sync" \
  7 \
  16 \
  --threads 2 \
  --max-iter 5 \
  --conv-threshold 0.001 \
  --no-skip-fftcc \
  --output-format raw \
  --input-depth 12 \
  --input-align left \
  --no-preload \
  >"${WORK_DIR}/logs/cli_sync_12_left.log" 2>&1

python3 - "${WORK_DIR}/out-sync/manifest.opencorr-raw.json" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
input_image = manifest["input_image"]
assert input_image["depth"] == "12", input_image
assert input_image["align"] == "left", input_image
assert input_image["imread_flags"] == "IMREAD_ANYDEPTH|IMREAD_GRAYSCALE", input_image
assert len(manifest["frames"]) == 1, manifest["frames"]
print("sync manifest input_image:", input_image)
PY

grep -q "Input image mode: depth=16" "${WORK_DIR}/logs/cli_preload_16.log"
grep -q "Input image mode: depth=12, align=left" "${WORK_DIR}/logs/cli_sync_12_left.log"
grep -q "Image preload: disabled" "${WORK_DIR}/logs/cli_sync_12_left.log"

echo "High-bit input smoke passed; logs: ${WORK_DIR}/logs"
