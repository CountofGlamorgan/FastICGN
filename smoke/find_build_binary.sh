#!/usr/bin/env bash
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Shared build-tree discovery for the targeted smoke scripts. Source it; it is
# not meant to be executed on its own.
#
#   value from the environment  ->  a build tree inside this repository  ->  error
#
# The candidate build trees are the ones CMakePresets.json creates, in the same
# order smoke/selfcheck.sh and pipeline/dic_pipeline.py use, so no smoke script
# needs a site-specific default path.

FASTICGN_BUILD_SUBDIRS=(build/cuda-release build/cuda-tests build/cuda-fftw build/cpu-only build)
FASTICGN_SMOKE_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# fasticgn_find_build_dir <file> [test-flag] -> prints the first repository
# build tree holding <file>, or returns 1.
fasticgn_find_build_dir() {
  local wanted="$1" flag="${2:--e}" sub candidate
  for sub in "${FASTICGN_BUILD_SUBDIRS[@]}"; do
    candidate="${FASTICGN_SMOKE_REPO_ROOT}/${sub}"
    # `[` (not `[[`) so the test flag can come from a variable.
    if [ "${flag}" "${candidate}/${wanted}" ]; then printf '%s' "${candidate}"; return 0; fi
  done
  return 1
}

fasticgn_smoke_not_found() {  # <what> <variable-name>
  echo "error: no repository build tree with '$1' was found." >&2
  echo "       Looked under ${FASTICGN_SMOKE_REPO_ROOT}/ in: ${FASTICGN_BUILD_SUBDIRS[*]}" >&2
  echo "       Set $2, or build one from the repository root:" >&2
  echo "         cmake --preset cuda-fftw && cmake --build --preset cuda-fftw -j \"\$(nproc)\"" >&2
  echo "       (cmake --preset cpu-only is enough for the CPU-only checks)." >&2
}

# fasticgn_require_build_dir <file> <override> <variable-name>
fasticgn_require_build_dir() {
  local wanted="$1" override="${2:-}" varname="${3:-BUILD_DIR}" found
  if [[ -n "${override}" ]]; then printf '%s' "${override}"; return 0; fi
  if found="$(fasticgn_find_build_dir "${wanted}")"; then printf '%s' "${found}"; return 0; fi
  fasticgn_smoke_not_found "${wanted}" "${varname}"
  exit 1
}

# fasticgn_require_binary <name> <override> <variable-name>
fasticgn_require_binary() {
  local name="$1" override="${2:-}" varname="${3:-GPU_BIN}" found
  if [[ -n "${override}" ]]; then printf '%s' "${override}"; return 0; fi
  if found="$(fasticgn_find_build_dir "${name}" -x)"; then
    printf '%s' "${found}/${name}"
    return 0
  fi
  fasticgn_smoke_not_found "${name}" "${varname}"
  exit 1
}
