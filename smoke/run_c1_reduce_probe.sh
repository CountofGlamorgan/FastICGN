#!/usr/bin/env bash
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-${TMPDIR:-/tmp}/fasticgn-smoke-c1}"
NVCC="${NVCC:-/usr/local/cuda-13.2/bin/nvcc}"

mkdir -p "${WORK_DIR}"
"${NVCC}" -std=c++17 "${SCRIPT_DIR}/c1_reduce_probe.cu" -o "${WORK_DIR}/c1_reduce_probe"
"${WORK_DIR}/c1_reduce_probe"
