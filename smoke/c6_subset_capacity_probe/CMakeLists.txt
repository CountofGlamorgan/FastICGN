# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.18)
project(c6_subset_capacity_probe LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures" FORCE)

set(OPENCORR_GPU_MVP_SOURCE_DIR "" CACHE PATH "OpenCorr GPU MVP source directory")
set(OPENCORR_GPU_MVP_BUILD_DIR "" CACHE PATH "OpenCorr GPU MVP build directory")

if(NOT EXISTS "${OPENCORR_GPU_MVP_SOURCE_DIR}/src/icgn2d1_cuda_mvp.h")
  message(FATAL_ERROR "Set OPENCORR_GPU_MVP_SOURCE_DIR to the FastICGN source tree")
endif()
if(NOT EXISTS "${OPENCORR_GPU_MVP_BUILD_DIR}/libopencorr_gpu_mvp_core.a")
  message(FATAL_ERROR "Set OPENCORR_GPU_MVP_BUILD_DIR to a built FastICGN CMake tree")
endif()

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# The core library pulls in the FFTW libraries only when it was built with
# OPENCORR_GPU_MVP_WITH_FFTW=ON (default OFF for CUDA builds since the
# batched-cuFFT FFTCC landed), so probe them without REQUIRED; cuFFT is
# always needed by the CUDA build of the core library.
find_library(FFTW3_LIB fftw3)
find_library(FFTW3F_LIB fftw3f)
find_library(FFTW3L_LIB fftw3l)

set(FFTW_PROBE_LIBS "")
foreach(fftw_lib IN LISTS FFTW3_LIB FFTW3F_LIB FFTW3L_LIB)
  if(fftw_lib)
    list(APPEND FFTW_PROBE_LIBS ${fftw_lib})
  endif()
endforeach()

add_executable(c6_subset_capacity_probe
  c6_subset_capacity_probe.cpp
  cuda_device_link_stub.cu
)
target_include_directories(c6_subset_capacity_probe
  PRIVATE
    "${OPENCORR_GPU_MVP_SOURCE_DIR}/src"
    "${OPENCORR_GPU_MVP_BUILD_DIR}"
    /usr/include/eigen3
    /usr/include/eigen3/Eigen
    ${OpenCV_INCLUDE_DIRS}
)
target_compile_definitions(c6_subset_capacity_probe PRIVATE OPENCORR_GPU_MVP_USE_CUDA=1)
target_link_libraries(c6_subset_capacity_probe
  PRIVATE
    "${OPENCORR_GPU_MVP_BUILD_DIR}/libopencorr_gpu_mvp_core.a"
    ${OpenCV_LIBS}
    ${FFTW_PROBE_LIBS}
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cufft
)
set_target_properties(c6_subset_capacity_probe PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_ARCHITECTURES 89
)
