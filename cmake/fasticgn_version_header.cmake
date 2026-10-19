# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Writes opencorr_gpu_mvp_version.h from src/opencorr_gpu_mvp_version.h.in.
#
# CMakeLists.txt includes this file at configure time and the
# opencorr_gpu_mvp_version target runs it again (cmake -P) on every build, so
# git_hash and git_dirty describe the tree that was compiled, not the tree that
# was configured. configure_file() rewrites the header only when its content
# changes, so a build of an unchanged tree recompiles nothing.
#
# Inputs (variables of the including scope, or -D on the cmake -P command line):
#   FASTICGN_SOURCE_DIR          the source tree
#   FASTICGN_VERSION_TEMPLATE    src/opencorr_gpu_mvp_version.h.in
#   FASTICGN_VERSION_HEADER      the header to write
#   OPENCORR_GPU_MVP_VERSION     semantic version, e.g. 1.2.0-dev
#   OPENCORR_GPU_MVP_BUILD_DATE  UTC time the build tree was configured
# Sets OPENCORR_GPU_MVP_GIT_HASH (12 hex digits, or "unknown" outside a git
# checkout) and OPENCORR_GPU_MVP_GIT_DIRTY ("0", "1", or "unknown").

set(OPENCORR_GPU_MVP_GIT_HASH "unknown")
set(OPENCORR_GPU_MVP_GIT_DIRTY "unknown")

# Only a checkout whose top level is this source tree counts: a source archive
# unpacked inside some other repository must not report that repository's HEAD.
execute_process(
  COMMAND git rev-parse --show-toplevel
  WORKING_DIRECTORY "${FASTICGN_SOURCE_DIR}"
  OUTPUT_VARIABLE fasticgn_git_toplevel
  RESULT_VARIABLE fasticgn_git_result
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(fasticgn_git_result EQUAL 0 AND fasticgn_git_toplevel)
  get_filename_component(fasticgn_git_toplevel "${fasticgn_git_toplevel}" REALPATH)
  get_filename_component(fasticgn_source_real "${FASTICGN_SOURCE_DIR}" REALPATH)
  if(fasticgn_git_toplevel STREQUAL fasticgn_source_real)
    execute_process(
      COMMAND git rev-parse --short=12 HEAD
      WORKING_DIRECTORY "${FASTICGN_SOURCE_DIR}"
      OUTPUT_VARIABLE fasticgn_git_hash
      RESULT_VARIABLE fasticgn_git_result
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(fasticgn_git_result EQUAL 0 AND fasticgn_git_hash)
      set(OPENCORR_GPU_MVP_GIT_HASH "${fasticgn_git_hash}")
      # Tracked files only, like `git describe --dirty`. --no-optional-locks
      # keeps the probe read-only: it never rewrites the index.
      execute_process(
        COMMAND git --no-optional-locks status --porcelain --untracked-files=no
        WORKING_DIRECTORY "${FASTICGN_SOURCE_DIR}"
        OUTPUT_VARIABLE fasticgn_git_status
        RESULT_VARIABLE fasticgn_git_result
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
      if(fasticgn_git_result EQUAL 0)
        if(fasticgn_git_status STREQUAL "")
          set(OPENCORR_GPU_MVP_GIT_DIRTY "0")
        else()
          set(OPENCORR_GPU_MVP_GIT_DIRTY "1")
        endif()
      endif()
    endif()
  endif()
endif()

configure_file("${FASTICGN_VERSION_TEMPLATE}" "${FASTICGN_VERSION_HEADER}" @ONLY)
