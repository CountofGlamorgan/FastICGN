# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# 3.18 is the real floor: CUDAToolkit imported targets need 3.17 and
# CMAKE_CUDA_ARCHITECTURES needs 3.18. Driving the build through
# CMakePresets.json additionally requires a CMake client of 3.21 or newer.
cmake_minimum_required(VERSION 3.18)
# The project/target/binary names stay opencorr_gpu_mvp / opencorr_cli_cuda_mvp:
# deployment paths and scripts depend on them. FastICGN is the public name.
project(opencorr_gpu_mvp
  VERSION 1.2.0
  DESCRIPTION "FastICGN: GPU-accelerated 2D digital image correlation built on OpenCorr"
  LANGUAGES CXX)

include(GNUInstallDirs)

option(OPENCORR_GPU_MVP_ENABLE_CUDA
  "Build the CUDA MVP refinement path; OFF builds the CPU-only backend" ON)

# FFTW (GPL-2+) backs only the legacy CPU FFTCC path. CUDA builds default to
# the batched-cuFFT FFTCC (src/fftcc2d_cuda.*) and neither look for nor link
# FFTW; CPU-only builds keep the FFTW path. Turn this ON explicitly to get a
# dual-backend CUDA build for --fftcc cpu|gpu A/B runs.
if(OPENCORR_GPU_MVP_ENABLE_CUDA)
  set(OPENCORR_GPU_MVP_WITH_FFTW_DEFAULT OFF)
else()
  set(OPENCORR_GPU_MVP_WITH_FFTW_DEFAULT ON)
endif()
option(OPENCORR_GPU_MVP_WITH_FFTW
  "Build the FFTW-based CPU FFTCC path (fftw3f, GPL-2.0-or-later; see NOTICE.md)"
  ${OPENCORR_GPU_MVP_WITH_FFTW_DEFAULT})

option(OPENCORR_GPU_MVP_BUILD_TESTS
  "Register the ctest regression suite for the selected backend" OFF)

if(NOT OPENCORR_GPU_MVP_ENABLE_CUDA AND NOT OPENCORR_GPU_MVP_WITH_FFTW)
  message(FATAL_ERROR "CPU-only CLI requires FFTW: enable OPENCORR_GPU_MVP_WITH_FFTW")
endif()

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    Debug Release RelWithDebInfo MinSizeRel)
  message(STATUS
    "CMAKE_BUILD_TYPE was not set; defaulting to Release "
    "(the validated production configuration)")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Semantic version printed by `opencorr_cli_cuda_mvp --version` and recorded in
# manifests: "<PROJECT_VERSION><suffix>". The suffix stays "-dev" between
# releases and is emptied in the commit that gets tagged v<PROJECT_VERSION>.
# CITATION.cff carries the same string with a leading "v", and pyproject.toml
# its PEP 440 spelling; smoke/test_version_metadata.py binds the three.
set(OPENCORR_GPU_MVP_VERSION_SUFFIX "")
set(OPENCORR_GPU_MVP_VERSION "${PROJECT_VERSION}${OPENCORR_GPU_MVP_VERSION_SUFFIX}")

# Build identity. build_date is the time this build tree was configured; the
# git hash and the dirty flag are sampled now and again on every build (see
# cmake/fasticgn_version_header.cmake), so editing a tracked file or moving
# HEAD after configuring cannot leave a stale identity in the binary.
string(TIMESTAMP OPENCORR_GPU_MVP_BUILD_DATE "%Y-%m-%dT%H:%M:%SZ" UTC)
set(FASTICGN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FASTICGN_VERSION_TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/src/opencorr_gpu_mvp_version.h.in)
set(FASTICGN_VERSION_HEADER ${CMAKE_CURRENT_BINARY_DIR}/opencorr_gpu_mvp_version.h)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/fasticgn_version_header.cmake)
add_custom_target(opencorr_gpu_mvp_version ALL
  COMMAND ${CMAKE_COMMAND}
    -DFASTICGN_SOURCE_DIR=${FASTICGN_SOURCE_DIR}
    -DFASTICGN_VERSION_TEMPLATE=${FASTICGN_VERSION_TEMPLATE}
    -DFASTICGN_VERSION_HEADER=${FASTICGN_VERSION_HEADER}
    -DOPENCORR_GPU_MVP_VERSION=${OPENCORR_GPU_MVP_VERSION}
    -DOPENCORR_GPU_MVP_BUILD_DATE=${OPENCORR_GPU_MVP_BUILD_DATE}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/fasticgn_version_header.cmake
  BYPRODUCTS ${FASTICGN_VERSION_HEADER}
  COMMENT "Refreshing the embedded build identity (git hash, dirty flag)"
  VERBATIM
)

if(OPENCORR_GPU_MVP_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures")
    message(STATUS
      "CMAKE_CUDA_ARCHITECTURES was not set; defaulting to 89 (Ada, the "
      "validated production target). Pass -DCMAKE_CUDA_ARCHITECTURES=native "
      "or an explicit list for other GPUs.")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FLAGS)
  separate_arguments(OPENCORR_OPENMP_CXX_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
endif()

# Eigen is header-only and is resolved without hard-coded system paths:
#   1. -DOPENCORR_GPU_MVP_EIGEN_INCLUDE_DIR=<dir containing Eigen/Core> wins;
#   2. otherwise find_package(Eigen3 3.3 CONFIG) and the Eigen3::Eigen dir;
#   3. otherwise find_path() for Eigen/Core under the usual prefixes.
# The vendored headers use upstream OpenCorr's non-standard `#include <Eigen>`
# spelling, so both the Eigen root and its Eigen/ subdirectory go on the
# include path; the vendored sources stay unmodified.
set(OPENCORR_GPU_MVP_EIGEN_INCLUDE_DIR "" CACHE PATH
  "Directory containing Eigen/Core; overrides automatic Eigen discovery")

set(OPENCORR_EIGEN_INCLUDE_DIR "")
set(OPENCORR_EIGEN_ORIGIN "")
if(OPENCORR_GPU_MVP_EIGEN_INCLUDE_DIR)
  set(OPENCORR_EIGEN_INCLUDE_DIR "${OPENCORR_GPU_MVP_EIGEN_INCLUDE_DIR}")
  set(OPENCORR_EIGEN_ORIGIN "OPENCORR_GPU_MVP_EIGEN_INCLUDE_DIR")
else()
  find_package(Eigen3 3.3 CONFIG QUIET)
  if(TARGET Eigen3::Eigen)
    get_target_property(OPENCORR_EIGEN_TARGET_DIRS
      Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    foreach(candidate IN LISTS OPENCORR_EIGEN_TARGET_DIRS)
      if(NOT "${candidate}" MATCHES "\\$<" AND EXISTS "${candidate}/Eigen/Core")
        set(OPENCORR_EIGEN_INCLUDE_DIR "${candidate}")
        break()
      endif()
    endforeach()
    if(NOT OPENCORR_EIGEN_INCLUDE_DIR AND EIGEN3_INCLUDE_DIR
        AND EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Core")
      set(OPENCORR_EIGEN_INCLUDE_DIR "${EIGEN3_INCLUDE_DIR}")
    endif()
    if(OPENCORR_EIGEN_INCLUDE_DIR)
      set(OPENCORR_EIGEN_ORIGIN "Eigen3::Eigen ${Eigen3_VERSION} (find_package CONFIG)")
    endif()
  endif()
  if(NOT OPENCORR_EIGEN_INCLUDE_DIR)
    find_path(OPENCORR_GPU_MVP_EIGEN_FOUND_DIR
      NAMES Eigen/Core
      PATH_SUFFIXES eigen3
      DOC "Directory containing Eigen/Core, located by find_path")
    if(OPENCORR_GPU_MVP_EIGEN_FOUND_DIR)
      set(OPENCORR_EIGEN_INCLUDE_DIR "${OPENCORR_GPU_MVP_EIGEN_FOUND_DIR}")
      set(OPENCORR_EIGEN_ORIGIN "find_path")
    endif()
  endif()
endif()

if(NOT OPENCORR_EIGEN_INCLUDE_DIR OR NOT EXISTS "${OPENCORR_EIGEN_INCLUDE_DIR}/Eigen/Core")
  message(FATAL_ERROR
    "Eigen 3 headers not found: no directory containing Eigen/Core.\n"
    "Install the development package (Debian/Ubuntu: libeigen3-dev, "
    "Fedora: eigen3-devel, conda: eigen, Homebrew: eigen), or point CMake at "
    "an existing copy with -DEigen3_DIR=<prefix>/share/eigen3/cmake or "
    "-DOPENCORR_GPU_MVP_EIGEN_INCLUDE_DIR=<dir containing Eigen/Core>.")
endif()
message(STATUS "Eigen include dir: ${OPENCORR_EIGEN_INCLUDE_DIR} (via ${OPENCORR_EIGEN_ORIGIN})")

if(OPENCORR_GPU_MVP_WITH_FFTW)
  find_library(FFTW3_LIB fftw3)
  find_library(FFTW3F_LIB fftw3f)
  find_library(FFTW3L_LIB fftw3l)

  if(NOT FFTW3F_LIB)
    message(FATAL_ERROR "fftw3f library not found")
  endif()
endif()

set(OPENCORR_GPU_MVP_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(opencorr_gpu_mvp_core
  src/oc_dic.cpp
  src/oc_image.cpp
  src/oc_deformation.cpp
  src/oc_subset.cpp
  src/oc_gradient.cpp
  src/oc_cubic_bspline.cpp
  src/oc_icgn.cpp
  src/icgn2d1_cuda_mvp.cpp
)

target_include_directories(opencorr_gpu_mvp_core
  PUBLIC
    ${OPENCORR_GPU_MVP_SRC_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${OPENCORR_EIGEN_INCLUDE_DIR}
    ${OPENCORR_EIGEN_INCLUDE_DIR}/Eigen
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(opencorr_gpu_mvp_core
  PUBLIC
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
)

if(OPENCORR_GPU_MVP_WITH_FFTW)
  target_sources(opencorr_gpu_mvp_core PUBLIC src/oc_fftcc.cpp)
  target_compile_definitions(opencorr_gpu_mvp_core PUBLIC OPENCORR_HAS_FFTW=1)
  target_link_libraries(opencorr_gpu_mvp_core
    PUBLIC
      ${FFTW3_LIB}
      ${FFTW3F_LIB}
      ${FFTW3L_LIB}
  )
endif()

if(OPENCORR_GPU_MVP_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)

  target_sources(opencorr_gpu_mvp_core PRIVATE
    src/icgn2d1_cuda_mvp.cu
    src/fftcc2d_cuda.cpp
    src/fftcc2d_cuda.cu
  )
  target_compile_definitions(opencorr_gpu_mvp_core PUBLIC OPENCORR_GPU_MVP_USE_CUDA=1)
  foreach(openmp_flag IN LISTS OPENCORR_OPENMP_CXX_FLAGS)
    target_compile_options(opencorr_gpu_mvp_core PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${openmp_flag}>
    )
  endforeach()
  set_target_properties(opencorr_gpu_mvp_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )
  target_link_libraries(opencorr_gpu_mvp_core PUBLIC CUDA::cudart CUDA::cufft)
endif()

add_library(opencorr_cli_io STATIC
  src/cli_output_integrity.cpp
  src/cli_image_probe.cpp)
target_include_directories(opencorr_cli_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
add_executable(opencorr_cli_cuda_mvp opencorr_cli_cuda_mvp.cpp)
target_link_libraries(opencorr_cli_cuda_mvp PRIVATE opencorr_gpu_mvp_core opencorr_cli_io)
# The CLI is the only translation unit that includes the generated version
# header; refresh it before the CLI is compiled.
add_dependencies(opencorr_cli_cuda_mvp opencorr_gpu_mvp_version)

# Reference CPU CLI (upstream OpenCorr argument and CSV contract). It is the
# CPU side of smoke/run_v9_cpu_gpu_parity.sh and smoke/run_c5_oob_boundary.sh,
# so those checks no longer need an external OpenCorr checkout. It uses the
# vendored CPU FFTCC, hence the FFTW requirement.
if(OPENCORR_GPU_MVP_WITH_FFTW)
  add_executable(opencorr_cli_cpu opencorr_cli.cpp)
  target_link_libraries(opencorr_cli_cpu PRIVATE opencorr_gpu_mvp_core)
else()
  message(STATUS
    "opencorr_cli_cpu is not built: the CPU reference CLI needs the FFTW "
    "FFTCC backend. Re-configure with -DOPENCORR_GPU_MVP_WITH_FFTW=ON "
    "(GPL-2.0-or-later, see NOTICE.md) or use the cpu-only preset, where "
    "FFTW is enabled by default.")
endif()

add_executable(image_input_depth_probe smoke/image_input_depth_probe.cpp)
target_link_libraries(image_input_depth_probe PRIVATE opencorr_gpu_mvp_core)

install(TARGETS opencorr_cli_cuda_mvp
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(TARGET opencorr_cli_cpu)
  install(TARGETS opencorr_cli_cpu
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(TARGETS opencorr_gpu_mvp_core
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
# Headers for C++ embedding. docs/library-contracts.md names the supported
# public API (FFTCC2DCuda, the vendored FFTCC2D, ICGN2D1CudaMvp); this list is
# the transitive include closure of those three headers and nothing else, so
# every installed header compiles on its own against what is installed.
# Deliberately not installed: src/cli_output_integrity.h (needs
# third_party/nlohmann and the internal opencorr_cli_io target), src/opencorr.h
# and the vendored headers only it pulls in (oc_feature, oc_sift,
# oc_nearest_neighbor + nanoflann.hpp, oc_io, oc_strain, oc_stereovision,
# oc_calibration, oc_epipolar_search, oc_feature_affine, oc_iclm, oc_nr).
# smoke/check_installed_headers.sh compiles the installed set one header at a
# time and cross-checks it against docs/library-contracts.md.
set(OPENCORR_GPU_MVP_PUBLIC_HEADERS
  ${OPENCORR_GPU_MVP_SRC_DIR}/icgn2d1_cuda_mvp.h
  ${OPENCORR_GPU_MVP_SRC_DIR}/oc_array.h
  ${OPENCORR_GPU_MVP_SRC_DIR}/oc_cubic_bspline.h
  ${OPENCORR_GPU_MVP_SRC_DIR}/oc_deformation.h
  ${OPENCORR_GPU_MVP_SRC_DIR}/oc_dic.h
  ${OPENCORR_GPU_MVP_SRC_DIR}/oc_gradient.h
  ${OPENCORR_GPU_MVP_SRC_DIR}/oc_icgn.h
  ${OPENCORR_GPU_MVP_SRC_DIR}/oc_image.h
  ${OPENCORR_GPU_MVP_SRC_DIR}/oc_interpolation.h
  ${OPENCORR_GPU_MVP_SRC_DIR}/oc_poi.h
  ${OPENCORR_GPU_MVP_SRC_DIR}/oc_point.h
  ${OPENCORR_GPU_MVP_SRC_DIR}/oc_subset.h)
if(OPENCORR_GPU_MVP_ENABLE_CUDA)
  # FFTCC2DCuda only exists in a CUDA build.
  list(APPEND OPENCORR_GPU_MVP_PUBLIC_HEADERS
    ${OPENCORR_GPU_MVP_SRC_DIR}/fftcc2d_cuda.h)
endif()
if(OPENCORR_GPU_MVP_WITH_FFTW)
  # FFTCC2D is compiled into the core library, and its header includes
  # <fftw3.h>, only when the FFTW backend is enabled.
  list(APPEND OPENCORR_GPU_MVP_PUBLIC_HEADERS
    ${OPENCORR_GPU_MVP_SRC_DIR}/oc_fftcc.h)
endif()
install(FILES ${OPENCORR_GPU_MVP_PUBLIC_HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/opencorr-gpu)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/opencorr_gpu_mvp_version.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/opencorr-gpu)
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE
    ${CMAKE_CURRENT_SOURCE_DIR}/NOTICE.md
    ${CMAKE_CURRENT_SOURCE_DIR}/README.md
    ${CMAKE_CURRENT_SOURCE_DIR}/CITATION.cff
  DESTINATION ${CMAKE_INSTALL_DOCDIR})
# docs/lab/ (one laboratory's deployment manual: host names, dataset paths)
# stays in the repository but is not installed.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs/
  DESTINATION ${CMAKE_INSTALL_DOCDIR}/docs
  FILES_MATCHING PATTERN "*.md"
  PATTERN "lab" EXCLUDE)

if(OPENCORR_GPU_MVP_BUILD_TESTS)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  enable_testing()
  add_executable(library_boundary_probe smoke/library_boundary_probe.cpp)
  target_link_libraries(library_boundary_probe PRIVATE opencorr_gpu_mvp_core)
  add_test(NAME library_boundaries COMMAND library_boundary_probe)
  add_library(io_faults SHARED smoke/io_faults.cpp)
  target_link_libraries(io_faults PRIVATE ${CMAKE_DL_LIBS})
  add_test(NAME output_integrity
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/smoke/test_output_integrity.py
      --binary $<TARGET_FILE:opencorr_cli_cuda_mvp> --fault-lib $<TARGET_FILE:io_faults>)
  # The CLI's own interface: --help, --version, the stdout status line, --quiet,
  # bit-depth warnings, the resume contract and --verify-only.
  if(OPENCORR_GPU_MVP_ENABLE_CUDA)
    set(OPENCORR_GPU_MVP_TEST_BACKEND cuda)
  else()
    set(OPENCORR_GPU_MVP_TEST_BACKEND cpu)
  endif()
  if(OPENCORR_GPU_MVP_WITH_FFTW)
    set(OPENCORR_GPU_MVP_TEST_FFTW 1)
  else()
    set(OPENCORR_GPU_MVP_TEST_FFTW 0)
  endif()
  add_test(NAME cli_interface
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/smoke/test_cli_interface.py
      --binary $<TARGET_FILE:opencorr_cli_cuda_mvp> --fault-lib $<TARGET_FILE:io_faults>
      --expect-version ${OPENCORR_GPU_MVP_VERSION}
      --expect-backend ${OPENCORR_GPU_MVP_TEST_BACKEND}
      --expect-fftw ${OPENCORR_GPU_MVP_TEST_FFTW})
  add_test(NAME parity_unit
    COMMAND ${Python3_EXECUTABLE} -m unittest discover
      -s ${CMAKE_CURRENT_SOURCE_DIR}/smoke -p test_check_cpu_gpu_parity.py)
  add_test(NAME version_metadata
    COMMAND ${Python3_EXECUTABLE} -m unittest discover
      -s ${CMAKE_CURRENT_SOURCE_DIR}/smoke -p test_version_metadata.py)
  # The partially-out-of-bounds rejection rule is backend-independent, so this
  # asserts it against whichever ICGN backend this configuration built.
  add_test(NAME oob_rejection
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/smoke/run_oob_rejection_gate.py
      --binary $<TARGET_FILE:opencorr_cli_cuda_mvp>)
  # The whole-frame fallback merge (review-2026-10-03 S1-1) lives in the CLI, so
  # this synthetic gate runs against whichever ICGN backend this configuration built.
  add_test(NAME frame_fallback
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/smoke/run_frame_fallback_gate.py
      --binary $<TARGET_FILE:opencorr_cli_cuda_mvp>)
  if(OPENCORR_GPU_MVP_ENABLE_CUDA)
    add_executable(cache_lifecycle_probe smoke/cache_lifecycle_probe.cpp)
    target_link_libraries(cache_lifecycle_probe PRIVATE opencorr_gpu_mvp_core)
    add_test(NAME cache_lifecycle
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/smoke/run_cache_regression.py
        --binary $<TARGET_FILE:opencorr_cli_cuda_mvp>
        --probe $<TARGET_FILE:cache_lifecycle_probe>)
    # Solvers owned by different host threads share only the process-wide CUDA
    # buffer registries (review-2026-10-03 S1-7); device 0, like cache_lifecycle.
    find_package(Threads REQUIRED)
    add_executable(registry_thread_probe smoke/registry_thread_probe.cpp)
    target_link_libraries(registry_thread_probe PRIVATE opencorr_gpu_mvp_core Threads::Threads)
    add_test(NAME registry_threads COMMAND registry_thread_probe 0)
  else()
    add_executable(cpu_lifecycle_probe smoke/cpu_lifecycle_probe.cpp)
    target_link_libraries(cpu_lifecycle_probe PRIVATE opencorr_gpu_mvp_core)
    add_test(NAME cpu_sequence_t1
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/smoke/run_cache_regression.py
        --binary $<TARGET_FILE:opencorr_cli_cuda_mvp> --threads 1
        --probe $<TARGET_FILE:cpu_lifecycle_probe>)
    add_test(NAME cpu_sequence_t4
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/smoke/run_cache_regression.py
        --binary $<TARGET_FILE:opencorr_cli_cuda_mvp> --threads 4
        --probe $<TARGET_FILE:cpu_lifecycle_probe>)
  endif()

  # ── review-2026-10-03 batch 4 (#26, #27). Keep this block last in the tests
  # section: its closing loop labels whatever was registered above it. ────────
  # Sub-pixel accuracy on noisy synthetic speckle with exact truth, plus the
  # CLI's numeric defaults (--conv-threshold, --max-iter, --subset-size). The
  # portable guard for the regressions only the lab anchors used to see
  # (finalconv, conv default, subset size/radius, s63); it runs against
  # whichever ICGN backend this configuration built.
  add_test(NAME subpixel_accuracy
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/smoke/run_subpixel_accuracy_gate.py
      --binary $<TARGET_FILE:opencorr_cli_cuda_mvp>)
  # Zero-texture reference subsets (open-items N-1, 2026-10-19): a constant
  # reference subset is rejected before iterating, -3 by the CUDA precompute
  # kernel and -5 by the CPU backend's 0/0 cost; the gate asserts whichever
  # backend this configuration built and that the POIs outside the constant
  # blocks are untouched.
  add_test(NAME zero_texture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/smoke/run_zero_texture_gate.py
      --binary $<TARGET_FILE:opencorr_cli_cuda_mvp>)

  # Labels for `ctest -L` / `-LE`, and timeouts.
  #   cpu | gpu   runs without a CUDA device | launches a CUDA binary, needs one
  #   numeric     compares computed displacements with a known truth
  #   integrity   output publication, resume and repair; the CLI's own interface
  #   unit        runs no CLI: Python unit tests, the library's host-thread probe
  # Every test has exactly one of cpu/gpu. No test needs the lab dataset: the
  # production anchors stay outside ctest (scripts/verify.sh --tier lab).
  # TIMEOUT is roughly 30 times the duration measured on the lab host on
  # 2026-10-05 (output_integrity 30 s with CUDA, frame_fallback 10 s CPU-only,
  # every other test under 5 s), so a hung CLI fails in minutes instead of
  # after ctest's 1500 s default.
  if(OPENCORR_GPU_MVP_ENABLE_CUDA)
    set(OPENCORR_TEST_DEVICE_LABEL gpu)
  else()
    set(OPENCORR_TEST_DEVICE_LABEL cpu)
  endif()
  set_tests_properties(parity_unit PROPERTIES LABELS "cpu;unit" TIMEOUT 120)
  set_tests_properties(output_integrity PROPERTIES
    LABELS "${OPENCORR_TEST_DEVICE_LABEL};integrity" TIMEOUT 900)
  set_tests_properties(library_boundaries oob_rejection frame_fallback subpixel_accuracy
    zero_texture
    PROPERTIES LABELS "${OPENCORR_TEST_DEVICE_LABEL};numeric" TIMEOUT 300)
  # cli_interface launches the CLI this configuration built. version_metadata
  # reads source files and runs no binary, so it is cpu in every build and
  # `ctest -LE gpu` (the CI's CUDA compile job) reaches it. Their timeout, and
  # that of registry_threads, comes from the closing loop.
  set_tests_properties(cli_interface PROPERTIES
    LABELS "${OPENCORR_TEST_DEVICE_LABEL};integrity")
  set_tests_properties(version_metadata PROPERTIES LABELS "cpu;unit")
  if(OPENCORR_GPU_MVP_ENABLE_CUDA)
    set_tests_properties(cache_lifecycle PROPERTIES LABELS "gpu;numeric" TIMEOUT 300)
    set_tests_properties(registry_threads PROPERTIES LABELS "gpu;unit")
  else()
    set_tests_properties(cpu_sequence_t1 cpu_sequence_t4
      PROPERTIES LABELS "cpu;numeric" TIMEOUT 300)
  endif()
  # A test registered above without properties still gets the device label and
  # a timeout, so `ctest -L gpu` and `-LE gpu` always partition the suite.
  get_property(OPENCORR_REGISTERED_TESTS DIRECTORY PROPERTY TESTS)
  foreach(registered_test IN LISTS OPENCORR_REGISTERED_TESTS)
    get_test_property(${registered_test} LABELS registered_labels)
    if(NOT registered_labels)
      set_tests_properties(${registered_test} PROPERTIES
        LABELS ${OPENCORR_TEST_DEVICE_LABEL})
    endif()
    get_test_property(${registered_test} TIMEOUT registered_timeout)
    if(NOT registered_timeout)
      set_tests_properties(${registered_test} PROPERTIES TIMEOUT 300)
    endif()
  endforeach()
endif()
