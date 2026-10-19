#!/bin/bash
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# FastICGN pipeline wrapper.
#
# Resolution order for dic_pipeline.py, highest first:
#   1. FASTICGN_PIPELINE                explicit dic_pipeline.py path
#   2. FASTICGN_HOME                    deployment root: <root>/dic_pipeline.py
#   3. [paths].home in the config file  $FASTICGN_CONFIG, else
#                                       ~/.config/fasticgn/config.toml
#   4. this script's own directory      a plain clone needs nothing else
#   5. the deprecated lab default       used only if the file is really there,
#                                       and always with a warning on stderr
#
# Interpreter, highest first: FASTICGN_PYTHON (skips venv activation),
# FASTICGN_VENV, <FASTICGN_HOME>/.venv, a .venv beside the resolved
# dic_pipeline.py, then plain python3.
#
# The script's own directory is taken from ${BASH_SOURCE[0]} without resolving
# symlinks, so a wrapper copied or linked into a deployment directory keeps
# using that deployment and a clone keeps using the clone.
#
# Binary resolution is dic_pipeline.py's job (FASTICGN_GPU_BIN, the optional
# config file, the repository build/ trees, then the deprecated lab path).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEGACY_LAB_HOME="/home/chonglang/opencorr-gpu-mvp-build"

FASTICGN_HOME="${FASTICGN_HOME:-}"
HOME_SOURCE=""
if [[ -n "${FASTICGN_HOME}" ]]; then
  HOME_SOURCE="environment:FASTICGN_HOME"
else
  # Same [paths].home key dic_pipeline.py reads; silent when python3, tomllib
  # or the file are missing, because every later candidate still applies.
  CONFIG_FILE="${FASTICGN_CONFIG:-${HOME:-}/.config/fasticgn/config.toml}"
  if [[ -n "${CONFIG_FILE}" && -f "${CONFIG_FILE}" ]]; then
    CONFIG_HOME="$("${FASTICGN_PYTHON:-python3}" - "${CONFIG_FILE}" <<'PY' 2>/dev/null || true
import sys
try:
    import tomllib
    with open(sys.argv[1], "rb") as handle:
        value = tomllib.load(handle).get("paths", {}).get("home", "")
    if isinstance(value, str):
        print(value)
except Exception:
    pass
PY
)"
    if [[ -n "${CONFIG_HOME}" ]]; then
      FASTICGN_HOME="${CONFIG_HOME}"
      HOME_SOURCE="config:${CONFIG_FILE}"
    fi
  fi
fi

PIPELINE="${FASTICGN_PIPELINE:-}"
PIPELINE_SOURCE="environment:FASTICGN_PIPELINE"
if [[ -z "${PIPELINE}" ]]; then
  PIPELINE_SOURCE=""
  for candidate in "${FASTICGN_HOME:+${FASTICGN_HOME}/dic_pipeline.py}" \
                   "${SCRIPT_DIR}/dic_pipeline.py" \
                   "${LEGACY_LAB_HOME}/dic_pipeline.py"; do
    if [[ -n "${candidate}" && -f "${candidate}" ]]; then
      PIPELINE="${candidate}"
      case "${candidate}" in
        "${FASTICGN_HOME}/dic_pipeline.py") PIPELINE_SOURCE="${HOME_SOURCE}" ;;
        "${SCRIPT_DIR}/dic_pipeline.py") PIPELINE_SOURCE="script_directory" ;;
        *) PIPELINE_SOURCE="legacy_lab_path" ;;
      esac
      break
    fi
  done
fi
if [[ -z "${PIPELINE}" ]]; then
  echo "error: dic_pipeline.py not found next to ${SCRIPT_DIR}; set FASTICGN_PIPELINE or FASTICGN_HOME." >&2
  exit 1
fi
if [[ "${PIPELINE_SOURCE}" == "legacy_lab_path" ]]; then
  echo "warning: deprecated_lab_default_path: dic_pipeline.py resolved to ${PIPELINE} from the" \
       "hard-coded lab path; set FASTICGN_PIPELINE or FASTICGN_HOME instead" >&2
fi

PYTHON="${FASTICGN_PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  for venv in "${FASTICGN_VENV:-}" \
              "${FASTICGN_HOME:+${FASTICGN_HOME}/.venv}" \
              "$(dirname "${PIPELINE}")/.venv"; do
    if [[ -n "${venv}" && -f "${venv}/bin/activate" ]]; then
      # shellcheck disable=SC1091
      source "${venv}/bin/activate"
      break
    fi
  done
  PYTHON="python3"
fi

exec "${PYTHON}" "${PIPELINE}" "$@"
