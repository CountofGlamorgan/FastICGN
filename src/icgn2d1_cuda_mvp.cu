/*
 * Copyright (c) 2026 Chonglang Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "icgn2d1_cuda_mvp.h"

#ifdef OPENCORR_GPU_MVP_USE_CUDA

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace opencorr
{
namespace
{

constexpr int kDefaultThreadsPerBlock = 128;
constexpr int kMaxStreams = 8;
constexpr int kParamCount = 6;
constexpr int kHessianCount = 36;
// Warp-per-POI iteration kernel launch shape (review-2026-09-02 §2.3): a
// fixed 128-thread block carries four POIs, one per warp.  The block size is
// fixed rather than occupancy-tuned because the kernel needs no dynamic
// shared memory and its reductions are warp-local shuffles, so the block
// size can change neither the shared footprint nor any per-POI result.
constexpr int kWarpKernelBlockSize = 128;
constexpr int kWarpsPerIterationBlock = kWarpKernelBlockSize / 32;
// Per-lane register cache capacity for sampled target values; subsets with
// subset_size <= 32*kWarpCacheSlots (e.g. production 15x15 = 225) sample
// once per iteration, larger subsets re-sample per pass (bit-identical, see
// the kernel comment).
constexpr int kWarpCacheSlots = 8;
// Partially-out-of-bounds rejection sentinel (upstream OpenCorr baa3de6,
// "Fix silent corruption from partially out-of-bounds target subsets in
// ICGN").  CPU rule, in src/oc_icgn.cpp: on every iteration, right after the
// warped target subset is reconstructed and before its zero-mean norm is
// taken, the subset is scanned for the interpolator's out-of-range sentinel
// (BicubicBspline::compute() returns exactly -1.f when the sample coordinate
// satisfies x < 1 || y < 1 || x >= width - 2 || y >= height - 2 || isnan);
// any hit rejects the POI with zncc = -3 and returns at once, leaving
// poi->deformation at the incoming guess and poi->result.iteration /
// .convergence untouched (0 for a fresh POI).
//
// GPU equivalent.  There is no -1.f sentinel to scan for here: this kernel's
// own guard in sampleTargetValue() tests the identical coordinate predicate
// and substitutes 0.0f, so the faithful port is to reject on that predicate
// firing rather than on a negative sample value.  The two differ only where
// the CPU's detector has a false positive -- a genuinely in-range B-spline
// sample that undershoots below zero -- which is an artefact of upstream's
// detection mechanism, not part of the rule.  NaN coordinates keep this
// kernel's pre-existing -5 treatment and are tracked on a separate flag.
//
// Output record: p[] reset to the incoming guess, iteration and convergence
// zeroed, zncc and status set to this sentinel -- byte-for-byte the record
// the reference-side precompute gate already writes for its own -3 rejection,
// and the same fields the CPU leaves behind.
constexpr float kOobRejectZncc = -3.0f;
// #define ICGN_DEBUG_POI 1  // disabled — re-enable to instrument iter trace for POI (518,20)

struct GpuPoiInput
{
  float x;
  float y;
  float p[kParamCount];
};

struct GpuPoiOutput
{
  float p[kParamCount];
  float zncc;
  float iteration;
  float convergence;
  float status;
};

struct LayoutPoint
{
  std::uint32_t x_bits = 0;
  std::uint32_t y_bits = 0;
};

struct CudaBuffers
{
  int ref_width = 0;
  int ref_height = 0;
  int tar_width = 0;
  int tar_height = 0;
  int subset_width = 0;
  int subset_height = 0;
  int subset_size = 0;
  bool ref_prepared = false;
  bool tar_prepared = false;
  bool tar_borrowed = false;
  const ICGN2D1CudaMvp::Impl* tar_donor = nullptr;
  bool poi_prepared = false;

  float* d_ref = nullptr;
  float* d_tar = nullptr;
  float* d_ref_grad_x = nullptr;
  float* d_ref_grad_y = nullptr;
  float* d_tar_b = nullptr;
  size_t tar_capacity_pixels = 0;

  GpuPoiInput* d_poi_in = nullptr;
  GpuPoiOutput* d_poi_out = nullptr;
  GpuPoiOutput* d_poi_precompute = nullptr;
  GpuPoiInput* h_poi_in = nullptr;
  GpuPoiOutput* h_poi_out = nullptr;
  // per-POI reference subset mean only; the zero-mean subset is recomputed
  // in the iteration kernel preamble (review-2026-09-02 K2)
  float* d_ref_mean = nullptr;
  float* d_ref_norm = nullptr;
  float* d_inv_hessian = nullptr;
  int* d_gather_indices = nullptr;
  int gather_index_capacity = 0;
  int poi_capacity = 0;
  int poi_capacity_subset_size = 0;
  int prepared_poi_count = 0;
  std::vector<LayoutPoint> prepared_layout;
  std::vector<LayoutPoint> staged_layout;
  std::array<cudaStream_t, kMaxStreams> streams = {};
  int stream_count = 0;
  int precompute_block_size = kDefaultThreadsPerBlock;
  int iteration_block_size = kDefaultThreadsPerBlock;
};

struct DeviceWork
{
  int device_id = 0;
  int start = 0;
  int count = 0;
  int stream_count = 0;
  bool d2h_enqueued = false;
  CudaBuffers* buffers = nullptr;
};

// Read-only kernel parameters are marked __restrict__ so the compiler can use
// the read-only (non-coherent) load path.  This is an aliasing promise only:
// every referenced buffer is a separate cudaMalloc allocation, and no floating
// operation or evaluation order changes, so results are bit-identical.
__global__ void precomputePoiKernel(const float* __restrict__ ref, const float* __restrict__ grad_x,
                                    const float* __restrict__ grad_y,
                                    int width, int height, const GpuPoiInput* __restrict__ poi_in,
                                    int subset_radius_x, int subset_radius_y, int subset_size,
                                    float* __restrict__ ref_mean, float* __restrict__ ref_norm,
                                    float* __restrict__ inv_hessian,
                                    GpuPoiOutput* __restrict__ poi_out);

// review-2026-09-02 §3.2: gather one POI's precomputed ref-side data from the
// donor instance's buffers (see definition).
__global__ void gatherPrecomputedPoiKernel(const int* __restrict__ indices, int subset_size,
                                           const float* __restrict__ src_ref_mean,
                                           const float* __restrict__ src_ref_norm,
                                           const float* __restrict__ src_inv_hessian,
                                           const GpuPoiOutput* __restrict__ src_poi_precompute,
                                           float* __restrict__ dst_ref_mean,
                                           float* __restrict__ dst_ref_norm,
                                           float* __restrict__ dst_inv_hessian,
                                           GpuPoiOutput* __restrict__ dst_poi_precompute);

__global__ void icgnIterationKernel(const float* __restrict__ tar_b, int tar_width, int tar_height,
                                    const float* __restrict__ ref_grad_x, const float* __restrict__ ref_grad_y,
                                    int ref_width, const GpuPoiInput* __restrict__ poi_in, int subset_radius_x,
                                    int subset_radius_y, int subset_size, float conv_criterion,
                                    int stop_condition, const float* __restrict__ ref,
                                    const float* __restrict__ ref_mean,
                                    const float* __restrict__ ref_norm, const float* __restrict__ inv_hessian,
                                    const GpuPoiOutput* __restrict__ poi_precompute,
                                    GpuPoiOutput* __restrict__ poi_out);

// review-2026-09-02 §2.3: warp-per-POI variant of icgnIterationKernel (see
// the definition for the design and determinism notes).  Same arguments as
// the legacy kernel plus poi_count (the launch grid is ceil(poi_count / 4),
// so whole warps whose POI index is out of range exit immediately).
__global__ void icgnIterationKernelWarp(const float* __restrict__ tar_b, int tar_width, int tar_height,
                                        const float* __restrict__ ref_grad_x, const float* __restrict__ ref_grad_y,
                                        int ref_width, const GpuPoiInput* __restrict__ poi_in, int subset_radius_x,
                                        int subset_radius_y, int subset_size, float conv_criterion,
                                        int stop_condition, const float* __restrict__ ref,
                                        const float* __restrict__ ref_mean,
                                        const float* __restrict__ ref_norm, const float* __restrict__ inv_hessian,
                                        const GpuPoiOutput* __restrict__ poi_precompute,
                                        GpuPoiOutput* __restrict__ poi_out, int poi_count);

struct RegistryEntry
{
  ICGN2D1CudaMvp::Impl* impl = nullptr;
  int device_id = 0;
  CudaBuffers* buffers = nullptr;
};

// review-2026-10-03 S1-7: the registry is process-wide state shared by every
// ICGN2D1CudaMvp instance, so each lookup, insertion, removal and walk over it
// holds registryMutex().  The same mutex guards CudaBuffers::tar_borrowed and
// tar_donor, the cross-instance link the borrower walks read on every
// registered entry.  It is never held across a CUDA call, a kernel launch or
// per-POI work: callers take the CudaBuffers pointer out under the lock and
// work on it unlocked, which is safe because one instance, and a tar donor
// together with its borrowers, is used from one host thread at a time
// (docs/library-contracts.md, "Thread ownership").
std::mutex& registryMutex()
{
  static std::mutex mutex;
  return mutex;
}

// Caller holds registryMutex().
std::vector<RegistryEntry>& cudaBufferRegistry()
{
  static std::vector<RegistryEntry> registry;
  return registry;
}

// Returns the pointer by value, never a reference into the registry vector: an
// insertion by another instance may reallocate the vector at any time, while
// the heap-allocated CudaBuffers itself stays put.
CudaBuffers* cudaBuffers(ICGN2D1CudaMvp::Impl& impl, int device_id)
{
  std::lock_guard<std::mutex> lock(registryMutex());
  auto& registry = cudaBufferRegistry();
  for (auto& entry : registry)
  {
    if (entry.impl == &impl && entry.device_id == device_id)
    {
      return entry.buffers;
    }
  }
  registry.push_back({&impl, device_id, new CudaBuffers()});
  return registry.back().buffers;
}

// Caller holds registryMutex().
CudaBuffers* findCudaBuffersLocked(const ICGN2D1CudaMvp::Impl& impl, int device_id)
{
  auto& registry = cudaBufferRegistry();
  for (const auto& entry : registry)
  {
    if (entry.impl == &impl && entry.device_id == device_id)
    {
      return entry.buffers;
    }
  }
  return nullptr;
}

CudaBuffers* findCudaBuffers(const ICGN2D1CudaMvp::Impl& impl, int device_id)
{
  std::lock_guard<std::mutex> lock(registryMutex());
  return findCudaBuffersLocked(impl, device_id);
}

void checkCuda(cudaError_t code, const char* context)
{
  if (code != cudaSuccess)
  {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(code));
  }
}

double wallSeconds()
{
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

size_t precomputeSharedBytes(int block_size)
{
  return sizeof(double) * static_cast<size_t>(block_size) * kHessianCount
       + sizeof(float)  * static_cast<size_t>(block_size) * 3;
}

size_t iterationSharedBytes(int block_size, int subset_size)
{
  return sizeof(double) * static_cast<size_t>(block_size) * kParamCount
       + sizeof(float)  * (static_cast<size_t>(subset_size) * 4
                         + static_cast<size_t>(block_size) * 2
                         + kParamCount + kHessianCount);
}

int clampBlockSize(int block_size)
{
  if (block_size >= 128)
  {
    return 128;
  }
  if (block_size >= 64)
  {
    return 64;
  }
  return 32;
}

struct PrecomputeSharedMemSizer
{
  size_t operator()(int candidate_block_size) const
  {
    return precomputeSharedBytes(candidate_block_size);
  }
};

struct IterationSharedMemSizer
{
  int subset_size = 0;

  size_t operator()(int candidate_block_size) const
  {
    // Keep launch tuning numerically stable with the historical three-buffer
    // layout.  The kernel now aliases sum/error scratch at launch time, but a
    // tuner-selected block-size change would alter the floating reduction tree.
    return iterationSharedBytes(candidate_block_size, subset_size)
         + sizeof(float) * static_cast<size_t>(candidate_block_size);
  }
};

void tuneLaunchConfig(CudaBuffers& buffers)
{
  int min_grid = 0;
  int block_size = kDefaultThreadsPerBlock;
  cudaError_t status = cudaOccupancyMaxPotentialBlockSizeVariableSMem(
      &min_grid, &block_size, precomputePoiKernel, PrecomputeSharedMemSizer(), 0);
  if (status == cudaSuccess)
  {
    buffers.precompute_block_size = clampBlockSize(block_size);
  }
  else
  {
    cudaGetLastError();
    buffers.precompute_block_size = kDefaultThreadsPerBlock;
  }

  status = cudaOccupancyMaxPotentialBlockSizeVariableSMem(
      &min_grid, &block_size, icgnIterationKernel,
      IterationSharedMemSizer{buffers.subset_size}, 0);
  if (status == cudaSuccess)
  {
    buffers.iteration_block_size = clampBlockSize(block_size);
  }
  else
  {
    cudaGetLastError();
    buffers.iteration_block_size = kDefaultThreadsPerBlock;
  }
}

void configureDynamicSharedMemory(CudaBuffers& buffers, int device_id)
{
  const size_t precompute_shared = precomputeSharedBytes(buffers.precompute_block_size);
  const size_t iteration_shared = iterationSharedBytes(buffers.iteration_block_size, buffers.subset_size);

  int default_limit = 0;
  checkCuda(cudaDeviceGetAttribute(&default_limit, cudaDevAttrMaxSharedMemoryPerBlock, device_id),
            "cudaDeviceGetAttribute MaxSharedMemoryPerBlock");

  int optin_limit = default_limit;
  cudaError_t status = cudaDeviceGetAttribute(&optin_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);
  if (status != cudaSuccess)
  {
    cudaGetLastError();
    optin_limit = default_limit;
  }

  auto requireLimit = [&](size_t shared_bytes, const char* kernel_name) {
    if (shared_bytes > static_cast<size_t>(optin_limit))
    {
      throw std::runtime_error(std::string(kernel_name) + " needs "
          + std::to_string(shared_bytes) + " dynamic shared-memory bytes, exceeding device opt-in limit "
          + std::to_string(optin_limit));
    }
  };

  requireLimit(precompute_shared, "precomputePoiKernel");
  requireLimit(iteration_shared, "icgnIterationKernel");

  if (precompute_shared > static_cast<size_t>(default_limit))
  {
    checkCuda(cudaFuncSetAttribute(precomputePoiKernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(precompute_shared)),
              "set precomputePoiKernel dynamic shared memory limit");
  }
  if (iteration_shared > static_cast<size_t>(default_limit))
  {
    checkCuda(cudaFuncSetAttribute(icgnIterationKernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(iteration_shared)),
              "set icgnIterationKernel dynamic shared memory limit");
  }

  std::cerr << "CUDA ICGN launch config: device=" << device_id
            << ", precompute_block=" << buffers.precompute_block_size
            << ", precompute_shared=" << precompute_shared << " bytes"
            << ", iteration_block=" << buffers.iteration_block_size
            << ", iteration_shared=" << iteration_shared << " bytes"
            << ", default_shared_limit=" << default_limit << " bytes"
            << ", optin_shared_limit=" << optin_limit << " bytes";
  if (precompute_shared > static_cast<size_t>(default_limit)
      || iteration_shared > static_cast<size_t>(default_limit))
  {
    std::cerr << " (dynamic shared-memory opt-in applied)";
  }
  std::cerr << std::endl;
}

std::vector<int> activeDevices(const ICGN2D1CudaMvp::Impl& impl)
{
  if (!impl.dual_gpu)
  {
    return {impl.device_id};
  }

  int device_count = 0;
  checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count < 2)
  {
    throw std::runtime_error("--dual-gpu requested but fewer than 2 CUDA devices are available");
  }
  return {0, 1};
}

LayoutPoint makeLayoutPoint(float x, float y)
{
  LayoutPoint point;
  std::memcpy(&point.x_bits, &x, sizeof(float));
  std::memcpy(&point.y_bits, &y, sizeof(float));
  return point;
}

bool stagedLayoutMatchesPrepared(const CudaBuffers& buffers, int count)
{
  if (!buffers.poi_prepared
      || buffers.prepared_poi_count != count
      || static_cast<int>(buffers.prepared_layout.size()) != count)
  {
    return false;
  }
  if (count == 0)
  {
    return true;
  }
  return std::memcmp(buffers.prepared_layout.data(), buffers.staged_layout.data(),
                     sizeof(LayoutPoint) * static_cast<size_t>(count)) == 0;
}

void commitPreparedLayout(CudaBuffers& buffers, int count)
{
  buffers.prepared_layout.resize(count);
  if (count > 0)
  {
    std::memcpy(buffers.prepared_layout.data(), buffers.staged_layout.data(),
                sizeof(LayoutPoint) * static_cast<size_t>(count));
  }
  buffers.poi_prepared = true;
  buffers.prepared_poi_count = count;
}

__device__ bool invert6x6(const double* src, float* dst)
{
  double aug[6][12];
  for (int r = 0; r < 6; ++r)
  {
    for (int c = 0; c < 6; ++c)
    {
      aug[r][c] = src[r * 6 + c];
      aug[r][6 + c] = (r == c) ? 1.0 : 0.0;
    }
  }

  for (int col = 0; col < 6; ++col)
  {
    int pivot = col;
    double pivot_abs = fabs(aug[col][col]);
    for (int row = col + 1; row < 6; ++row)
    {
      double candidate = fabs(aug[row][col]);
      if (candidate > pivot_abs)
      {
        pivot = row;
        pivot_abs = candidate;
      }
    }
    if (pivot_abs < 1.0e-12)
    {
      return false;
    }
    if (pivot != col)
    {
      for (int c = 0; c < 12; ++c)
      {
        double tmp = aug[col][c];
        aug[col][c] = aug[pivot][c];
        aug[pivot][c] = tmp;
      }
    }

    double denom = aug[col][col];
    for (int c = 0; c < 12; ++c)
    {
      aug[col][c] /= denom;
    }

    for (int row = 0; row < 6; ++row)
    {
      if (row == col)
      {
        continue;
      }
      double factor = aug[row][col];
      for (int c = 0; c < 12; ++c)
      {
        aug[row][c] -= factor * aug[col][c];
      }
    }
  }

  for (int r = 0; r < 6; ++r)
  {
    for (int c = 0; c < 6; ++c)
    {
      dst[r * 6 + c] = static_cast<float>(aug[r][6 + c]);
    }
  }
  return true;
}

template <typename T>
__device__ void reduceAdd(T* buf)
{
  const int tid = threadIdx.x;
  if (blockDim.x != 32 && blockDim.x != 64 && blockDim.x != 128)
  {
    int active = blockDim.x;
    while (active > 1)
    {
      int lower_count = active / 2;
      int upper_offset = active - lower_count;
      __syncthreads();
      if (tid < lower_count)
      {
        buf[tid] += buf[tid + upper_offset];
      }
      active = upper_offset;
    }
    __syncthreads();
    return;
  }

  if (blockDim.x >= 128)
  {
    __syncthreads();
    if (tid < 64)
    {
      buf[tid] += buf[tid + 64];
    }
  }
  if (blockDim.x >= 64)
  {
    __syncthreads();
    if (tid < 32)
    {
      buf[tid] += buf[tid + 32];
    }
  }

  // Launch tuning clamps the block size to 32, 64, or 128 threads.  Once the
  // reduction reaches one warp, warp barriers preserve the original binary
  // tree while avoiding block-wide barriers for the final five stages.
  __syncthreads();
  if (tid < 32)
  {
    constexpr unsigned kFullWarp = 0xffffffffu;
    if (tid < 16)
    {
      buf[tid] += buf[tid + 16];
    }
    __syncwarp(kFullWarp);
    if (tid < 8)
    {
      buf[tid] += buf[tid + 8];
    }
    __syncwarp(kFullWarp);
    if (tid < 4)
    {
      buf[tid] += buf[tid + 4];
    }
    __syncwarp(kFullWarp);
    if (tid < 2)
    {
      buf[tid] += buf[tid + 2];
    }
    __syncwarp(kFullWarp);
    if (tid == 0)
    {
      buf[0] += buf[1];
    }
  }
  __syncthreads();
}

template <typename T>
__device__ void reduceAddVector(T* buf, int values_per_thread)
{
  const int tid = threadIdx.x;
  if (blockDim.x != 32 && blockDim.x != 64 && blockDim.x != 128)
  {
    int active = blockDim.x;
    while (active > 1)
    {
      int lower_count = active / 2;
      int upper_offset = active - lower_count;
      __syncthreads();
      if (tid < lower_count)
      {
        int dst = tid * values_per_thread;
        int src = (tid + upper_offset) * values_per_thread;
        for (int i = 0; i < values_per_thread; ++i)
        {
          buf[dst + i] += buf[src + i];
        }
      }
      active = upper_offset;
    }
    __syncthreads();
    return;
  }

  if (blockDim.x >= 128)
  {
    __syncthreads();
    if (tid < 64)
    {
      int dst = tid * values_per_thread;
      int src = (tid + 64) * values_per_thread;
      for (int i = 0; i < values_per_thread; ++i)
      {
        buf[dst + i] += buf[src + i];
      }
    }
  }
  if (blockDim.x >= 64)
  {
    __syncthreads();
    if (tid < 32)
    {
      int dst = tid * values_per_thread;
      int src = (tid + 32) * values_per_thread;
      for (int i = 0; i < values_per_thread; ++i)
      {
        buf[dst + i] += buf[src + i];
      }
    }
  }

  __syncthreads();
  if (tid < 32)
  {
    constexpr unsigned kFullWarp = 0xffffffffu;
    for (int offset = 16; offset > 1; offset /= 2)
    {
      if (tid < offset)
      {
        int dst = tid * values_per_thread;
        int src = (tid + offset) * values_per_thread;
        for (int i = 0; i < values_per_thread; ++i)
        {
          buf[dst + i] += buf[src + i];
        }
      }
      __syncwarp(kFullWarp);
    }
    if (tid == 0)
    {
      for (int i = 0; i < values_per_thread; ++i)
      {
        buf[i] += buf[values_per_thread + i];
      }
    }
  }
  __syncthreads();
}

__device__ void setWarpMatrix(const float* p, float* warp)
{
  warp[0] = 1.0f + p[1];
  warp[1] = p[2];
  warp[2] = p[0];
  warp[3] = p[4];
  warp[4] = 1.0f + p[5];
  warp[5] = p[3];
  warp[6] = 0.0f;
  warp[7] = 0.0f;
  warp[8] = 1.0f;
}

__device__ bool invertAffine3x3(const float* mat, float* inv)
{
  float det = mat[0] * mat[4] - mat[1] * mat[3];
  if (fabsf(det) < 1e-8f)
  {
    return false;
  }

  float inv_det = 1.0f / det;
  inv[0] = mat[4] * inv_det;
  inv[1] = -mat[1] * inv_det;
  inv[3] = -mat[3] * inv_det;
  inv[4] = mat[0] * inv_det;
  inv[2] = -(inv[0] * mat[2] + inv[1] * mat[5]);
  inv[5] = -(inv[3] * mat[2] + inv[4] * mat[5]);
  inv[6] = 0.0f;
  inv[7] = 0.0f;
  inv[8] = 1.0f;
  return true;
}

__device__ void mul3x3(const float* a, const float* b, float* out)
{
  for (int r = 0; r < 3; ++r)
  {
    for (int c = 0; c < 3; ++c)
    {
      out[r * 3 + c] = 0.0f;
      for (int k = 0; k < 3; ++k)
      {
        out[r * 3 + c] += a[r * 3 + k] * b[k * 3 + c];
      }
    }
  }
}

__device__ void extractParams(const float* warp, float* p)
{
  p[0] = warp[2];
  p[1] = warp[0] - 1.0f;
  p[2] = warp[1];
  p[3] = warp[5];
  p[4] = warp[3];
  p[5] = warp[4] - 1.0f;
}

__device__ float interpolateBicubic(const float* coeff, float dx, float dy)
{
  // coeff points at 16 consecutive B-spline coefficients for one texel.  The
  // base is a cudaMalloc allocation and the texel stride is 64 bytes, so every
  // row of four coefficients is 16-byte aligned and float4 loads are safe.
  // The multiply-add chain keeps the original left-to-right evaluation order,
  // so results are bit-identical to the scalar version.
  float vx[4] = {1.0f, dx, dx * dx, dx * dx * dx};
  float vy[4] = {1.0f, dy, dy * dy, dy * dy * dy};
  float value = 0.0f;
  for (int r = 0; r < 4; ++r)
  {
    const float4 c4 = __ldg(reinterpret_cast<const float4*>(coeff + r * 4));
    float row = 0.0f;
    row += c4.x * vx[0];
    row += c4.y * vx[1];
    row += c4.z * vx[2];
    row += c4.w * vx[3];
    value += vy[r] * row;
  }
  return value;
}

// The sampling body of icgnIterationKernel (warp offsets -> warped target
// coordinate -> NaN/out-of-range guard -> bicubic), lifted into a helper so
// icgnIterationKernelWarp can evaluate the identical expression sequence in
// each of its passes.  Returns the sampled value, or 0.0f when the sample is
// NaN or out of range (sampled=false, matching the legacy kernel's
// tar_values[idx] = 0.0f stores); NaN coordinates additionally set
// nan_seen, which the caller turns into the POI failure flag, and
// out-of-range coordinates set oob_seen, which the caller turns into the
// partially-out-of-bounds rejection (see kOobRejectZncc).
__device__ float sampleTargetValue(const float* __restrict__ tar_b, int tar_width, int tar_height,
                                   float poi_x, float poi_y, const float* params,
                                   int lx, int ly, bool& nan_seen, bool& oob_seen, bool& sampled)
{
  float x = (1.0f + params[1]) * lx + params[2] * ly + params[0];
  float y = params[4] * lx + (1.0f + params[5]) * ly + params[3];
  float gx = poi_x + x;
  float gy = poi_y + y;
  if (isnan(gx) || isnan(gy))
  {
    nan_seen = true;
    sampled = false;
    return 0.0f;
  }
  if (gx < 1.0f || gy < 1.0f || gx >= tar_width - 2.0f || gy >= tar_height - 2.0f)
  {
    oob_seen = true;
    sampled = false;
    return 0.0f;
  }

  int ix = static_cast<int>(floorf(gx));
  int iy = static_cast<int>(floorf(gy));
  float dx = gx - ix;
  float dy = gy - iy;
  // Texel index stays 32-bit (iy * tar_width + ix is valid up to 2^31 pixels);
  // only the x16 element scaling is widened, which is where the ~134 Mpx int32
  // overflow lived.  A 64-bit multiply per sample costs ~8% of the warp
  // kernel on Ada, so the widening is confined to the final shift.  For
  // in-range images the computed address is identical (bit-identical).
  const float* coeff = tar_b + static_cast<size_t>(iy * tar_width + ix) * 16;
  sampled = true;
  return interpolateBicubic(coeff, dx, dy);
}

__global__ void precomputeBMatrixKernel(const float* __restrict__ tar, int width, int height,
                                        float* __restrict__ out_b)
{
  constexpr float bc[4][4] = {
      {-144.0f / 336.0f, 384.0f / 336.0f, -384.0f / 336.0f, 144.0f / 336.0f},
      {342.0f / 336.0f, -702.0f / 336.0f, 450.0f / 336.0f, -90.0f / 336.0f},
      {-198.0f / 336.0f, -18.0f / 336.0f, 270.0f / 336.0f, -54.0f / 336.0f},
      {0.0f, 1.0f, 0.0f, 0.0f}};

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
  {
    return;
  }

  // Same convention as the sampling path: 32-bit texel index, 64-bit x16.
  float* dst = out_b + static_cast<size_t>(y * width + x) * 16;
  for (int i = 0; i < 16; ++i)
  {
    dst[i] = 0.0f;
  }

  if (x < 1 || y < 1 || x >= width - 2 || y >= height - 2)
  {
    return;
  }

  float q[4][4];
  for (int r = 0; r < 4; ++r)
  {
    for (int c = 0; c < 4; ++c)
    {
      q[r][c] = tar[(y - 1 + r) * width + (x - 1 + c)];
    }
  }

  float p[4][4] = {};
  for (int k = 0; k < 4; ++k)
  {
    for (int l = 0; l < 4; ++l)
    {
      for (int m = 0; m < 4; ++m)
      {
        for (int n = 0; n < 4; ++n)
        {
          p[k][l] += bc[l][m] * bc[k][n] * q[n][m];
        }
      }
    }
  }

  for (int k = 0; k < 4; ++k)
  {
    for (int l = 0; l < 4; ++l)
    {
      dst[(3 - k) * 4 + (3 - l)] = p[k][l];
    }
  }
}

__global__ void gradientKernel(const float* __restrict__ ref, int width, int height,
                               float* __restrict__ grad_x, float* __restrict__ grad_y)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
  {
    return;
  }

  float gx = 0.0f;
  float gy = 0.0f;
  if (x >= 2 && x < width - 2)
  {
    gx = (1.0f / 12.0f) * ref[y * width + (x - 2)]
       - (2.0f / 3.0f) * ref[y * width + (x - 1)]
       + (2.0f / 3.0f) * ref[y * width + (x + 1)]
       - (1.0f / 12.0f) * ref[y * width + (x + 2)];
  }
  if (y >= 2 && y < height - 2)
  {
    gy = (1.0f / 12.0f) * ref[(y - 2) * width + x]
       - (2.0f / 3.0f) * ref[(y - 1) * width + x]
       + (2.0f / 3.0f) * ref[(y + 1) * width + x]
       - (1.0f / 12.0f) * ref[(y + 2) * width + x];
  }

  grad_x[y * width + x] = gx;
  grad_y[y * width + x] = gy;
}

__global__ void precomputePoiKernel(const float* __restrict__ ref, const float* __restrict__ grad_x,
                                    const float* __restrict__ grad_y,
                                    int width, int height, const GpuPoiInput* __restrict__ poi_in,
                                    int subset_radius_x, int subset_radius_y, int subset_size,
                                    float* __restrict__ ref_mean, float* __restrict__ ref_norm,
                                    float* __restrict__ inv_hessian,
                                    GpuPoiOutput* __restrict__ poi_out)
{
  int poi_idx = blockIdx.x;
  int tid = threadIdx.x;
  const GpuPoiInput poi = poi_in[poi_idx];

  extern __shared__ double shared_d[];
  double* hessian = shared_d;
  float* raw_sum = reinterpret_cast<float*>(hessian + blockDim.x * kHessianCount);
  float* sq_sum = raw_sum + blockDim.x;
  // Per-thread count of reference samples that differ from the subset's first
  // sample (the third float array precomputeSharedBytes() already reserves).
  float* mismatch = sq_sum + blockDim.x;

  raw_sum[tid] = 0.0f;
  sq_sum[tid] = 0.0f;
  mismatch[tid] = 0.0f;
  for (int i = tid; i < blockDim.x * kHessianCount; i += blockDim.x)
  {
    hessian[i] = 0.0;
  }
  __syncthreads();

  int center_x = static_cast<int>(poi.x);
  int center_y = static_cast<int>(poi.y);

  if (center_x - subset_radius_x < 2 || center_y - subset_radius_y < 2 ||
      center_x + subset_radius_x >= width - 2 || center_y + subset_radius_y >= height - 2)
  {
    if (tid == 0)
    {
      for (int i = 0; i < kParamCount; ++i)
      {
        poi_out[poi_idx].p[i] = poi.p[i];
      }
      poi_out[poi_idx].zncc = -3.0f;
      poi_out[poi_idx].iteration = 0.0f;
      poi_out[poi_idx].convergence = 0.0f;
      poi_out[poi_idx].status = -3.0f;
    }
    return;
  }

  // Zero-texture reference subsets (2026-10-19).  A reference subset whose
  // samples are all equal has nothing to correlate: its zero-mean norm is 0,
  // and the 1e-12 clamps below and in the iteration kernels then turn the
  // ZNSSD cost into a constant (1 for a textured target subset, 0 for a
  // constant one), so the POI left the solver as a "converged" measurement with
  // zncc = 0.5 or 1.0 whenever the 4-tap gradient stencil reaching 2 px past
  // the subset edge made the Hessian merely ill-conditioned instead of
  // singular.  Such a subset is rejected here, before iterating, through the
  // same exit as a singular Hessian: p[] keeps the incoming guess, iteration
  // and convergence are 0, zncc and status are -3.  The predicate is exact
  // equality of every sampled value with the subset's first sample (max ==
  // min of the samples), so it does not depend on --input-normalize or on a
  // norm threshold.  The CPU backend (src/oc_icgn.cpp, unchanged) reaches the
  // same subsets through 0/0 = NaN and reports -5; see docs/output-format.md.
  const float first_sample = ref[(center_y - subset_radius_y) * width + (center_x - subset_radius_x)];

  for (int idx = tid; idx < subset_size; idx += blockDim.x)
  {
    int lx = idx % (2 * subset_radius_x + 1) - subset_radius_x;
    int ly = idx / (2 * subset_radius_x + 1) - subset_radius_y;
    int gx = center_x + lx;
    int gy = center_y + ly;
    float value = ref[gy * width + gx];
    raw_sum[tid] += value;
    mismatch[tid] += (value != first_sample) ? 1.0f : 0.0f;

    double gxd = static_cast<double>(grad_x[gy * width + gx]);
    double gyd = static_cast<double>(grad_y[gy * width + gx]);
    double lxd = static_cast<double>(lx);
    double lyd = static_cast<double>(ly);
    double sd[kParamCount] = {gxd, gxd * lxd, gxd * lyd, gyd, gyd * lxd, gyd * lyd};

    for (int r = 0; r < kParamCount; ++r)
    {
      for (int c = 0; c < kParamCount; ++c)
      {
        hessian[tid * kHessianCount + r * 6 + c] += sd[r] * sd[c];
      }
    }
  }

  reduceAdd(raw_sum);
  reduceAdd(mismatch);
  reduceAddVector(hessian, kHessianCount);

  float mean = raw_sum[0] / static_cast<float>(subset_size);
  if (tid == 0)
  {
    ref_mean[poi_idx] = mean;
  }

  for (int idx = tid; idx < subset_size; idx += blockDim.x)
  {
    int lx = idx % (2 * subset_radius_x + 1) - subset_radius_x;
    int ly = idx / (2 * subset_radius_x + 1) - subset_radius_y;
    int gx = center_x + lx;
    int gy = center_y + ly;
    float zero_mean = ref[gy * width + gx] - mean;
    sq_sum[tid] += zero_mean * zero_mean;
  }
  reduceAdd(sq_sum);

  if (tid == 0)
  {
    ref_norm[poi_idx] = sqrtf(fmaxf(sq_sum[0], 1.0e-12f));
    // A constant subset is rejected without inverting; every other subset
    // takes exactly the path it took before this rule existed.
    const bool zero_texture = mismatch[0] == 0.0f;
    bool ok = !zero_texture
           && invert6x6(hessian, inv_hessian + static_cast<long long>(poi_idx) * kHessianCount);
    for (int i = 0; i < kParamCount; ++i)
    {
      poi_out[poi_idx].p[i] = poi.p[i];
    }
    poi_out[poi_idx].zncc = ok ? 0.0f : -3.0f;
    poi_out[poi_idx].iteration = 0.0f;
    poi_out[poi_idx].convergence = 0.0f;
    poi_out[poi_idx].status = ok ? 0.0f : -3.0f;
  }
}

// review-2026-09-02 §3.2: copy one POI's precomputed ref-side data (subset
// mean (K2), norm, inverse Hessian, precompute record) from the donor
// instance's slot indices[poi_idx] into this instance's slot poi_idx.  This is
// a pure device-to-device gather with no float math, so the copied bytes are
// exactly the donor's bytes.  Donor and recipient buffers must be distinct
// allocations (self-gather could overlap); the caller enforces this.
__global__ void gatherPrecomputedPoiKernel(const int* __restrict__ indices, int subset_size,
                                           const float* __restrict__ src_ref_mean,
                                           const float* __restrict__ src_ref_norm,
                                           const float* __restrict__ src_inv_hessian,
                                           const GpuPoiOutput* __restrict__ src_poi_precompute,
                                           float* __restrict__ dst_ref_mean,
                                           float* __restrict__ dst_ref_norm,
                                           float* __restrict__ dst_inv_hessian,
                                           GpuPoiOutput* __restrict__ dst_poi_precompute)
{
  int poi_idx = blockIdx.x;
  int src_idx = indices[poi_idx];
  // 64-bit POI index math: poi * kHessianCount overflows int32 above ~59.7M
  // POIs; addresses are identical for in-range queues, so bytes are unchanged.
  for (int i = threadIdx.x; i < kHessianCount; i += blockDim.x)
  {
    dst_inv_hessian[static_cast<long long>(poi_idx) * kHessianCount + i] =
        src_inv_hessian[static_cast<long long>(src_idx) * kHessianCount + i];
  }
  if (threadIdx.x == 0)
  {
    dst_ref_mean[poi_idx] = src_ref_mean[src_idx];
    dst_ref_norm[poi_idx] = src_ref_norm[src_idx];
    dst_poi_precompute[poi_idx] = src_poi_precompute[src_idx];
  }
  (void)subset_size;  // subset body no longer copied (K2)
}

__global__ void icgnIterationKernel(const float* __restrict__ tar_b, int tar_width, int tar_height,
                                    const float* __restrict__ ref_grad_x, const float* __restrict__ ref_grad_y,
                                    int ref_width, const GpuPoiInput* __restrict__ poi_in, int subset_radius_x,
                                    int subset_radius_y, int subset_size, float conv_criterion,
                                    int stop_condition, const float* __restrict__ ref,
                                    const float* __restrict__ ref_mean,
                                    const float* __restrict__ ref_norm, const float* __restrict__ inv_hessian,
                                    const GpuPoiOutput* __restrict__ poi_precompute,
                                    GpuPoiOutput* __restrict__ poi_out)
{
  __shared__ int failed_flag;
  // Set by any thread whose warped sample left the target image's
  // interpolatable range; read after a barrier, so the break below is
  // block-uniform (see kOobRejectZncc).
  __shared__ int oob_flag;
  __shared__ float s_final_conv;
  __shared__ float s_final_znssd;
  __shared__ float s_dp[kParamCount];
  int poi_idx = blockIdx.x;
  int tid = threadIdx.x;
  const GpuPoiInput* poi = poi_in + poi_idx;
  const float poi_x = poi->x;
  const float poi_y = poi->y;
  const float precompute_status = poi_precompute[poi_idx].status;
  if (precompute_status < 0.0f)
  {
    if (tid == 0)
    {
      for (int i = 0; i < kParamCount; ++i)
      {
        poi_out[poi_idx].p[i] = poi->p[i];
      }
      poi_out[poi_idx].zncc = poi_precompute[poi_idx].zncc;
      poi_out[poi_idx].iteration = poi_precompute[poi_idx].iteration;
      poi_out[poi_idx].convergence = poi_precompute[poi_idx].convergence;
      poi_out[poi_idx].status = precompute_status;
    }
    return;
  }

  extern __shared__ double shared_d[];
  float* num_buf = reinterpret_cast<float*>(shared_d);
  float* tar_values = num_buf + blockDim.x * kParamCount;
  // The sum and error reductions have disjoint lifetimes.  The intervening
  // norm reduction retains a block barrier, so every thread has consumed the
  // sum before this storage is reused for the error reduction.
  float* sum_err_buf = tar_values + subset_size;
  float* norm_buf = sum_err_buf + blockDim.x;
  float* params = norm_buf + blockDim.x;
  // Cache the per-POI reference subset invariants once; every iteration reuses them unchanged.
  float* s_ref_zero_mean = params + kParamCount;
  float* s_grad_x = s_ref_zero_mean + subset_size;
  float* s_grad_y = s_grad_x + subset_size;
  float* s_inv_hessian = s_grad_y + subset_size;

  if (tid < kParamCount)
  {
    params[tid] = poi->p[tid];
  }
  if (tid == 0)
  {
    failed_flag = 0;
    oob_flag = 0;
    s_final_conv = 0.0f;
    s_final_znssd = 4.0f;
  }
  __syncthreads();

  const float poi_ref_mean = ref_mean[poi_idx];
  const float* poi_inv_hessian = inv_hessian + static_cast<long long>(poi_idx) * kHessianCount;
  const float poi_ref_norm = ref_norm[poi_idx];
  const int subset_width = 2 * subset_radius_x + 1;
  const int center_x = static_cast<int>(poi_x);
  const int center_y = static_cast<int>(poi_y);

  for (int idx = tid; idx < subset_size; idx += blockDim.x)
  {
    int lx = idx % subset_width - subset_radius_x;
    int ly = idx / subset_width - subset_radius_y;
    int gx = center_x + lx;
    int gy = center_y + ly;
    // K2: recompute the zero-mean reference value; identical float expression
    // and operands to the precompute kernel's former cached store.
    s_ref_zero_mean[idx] = ref[gy * ref_width + gx] - poi_ref_mean;
    s_grad_x[idx] = ref_grad_x[gy * ref_width + gx];
    s_grad_y[idx] = ref_grad_y[gy * ref_width + gx];
  }
  for (int i = tid; i < kHessianCount; i += blockDim.x)
  {
    s_inv_hessian[i] = poi_inv_hessian[i];
  }
  __syncthreads();

  float final_znssd = 4.0f;
  float final_conv = 0.0f;
  int iteration = 0;
  bool oob_rejected = false;

  // Per-thread iteration invariants.  The sd factors used by the numerator
  // accumulation depend only on the thread's (fixed) strided subset indices
  // and the iteration-invariant gradient caches, so the first two strides
  // are computed once here and held in registers.  The doubles produced are
  // exactly the values the in-loop construction computes on every iteration,
  // and the num accumulation order is unchanged, so results are bit-identical.
  // Threads with more than two strides (larger subsets) fall through to the
  // in-loop construction for the remaining strides.
  constexpr int kSdCacheSlots = 2;
  float sd_cache[kSdCacheSlots][kParamCount];
#pragma unroll
  for (int s = 0; s < kSdCacheSlots; ++s)
  {
    const int idx = tid + s * blockDim.x;
    if (idx < subset_size)
    {
      const int lx = idx % subset_width - subset_radius_x;
      const int ly = idx / subset_width - subset_radius_y;
      const float gxd = s_grad_x[idx];
      const float gyd = s_grad_y[idx];
      const float lxd = static_cast<float>(lx);
      const float lyd = static_cast<float>(ly);
      sd_cache[s][0] = gxd;
      sd_cache[s][1] = gxd * lxd;
      sd_cache[s][2] = gxd * lyd;
      sd_cache[s][3] = gyd;
      sd_cache[s][4] = gyd * lxd;
      sd_cache[s][5] = gyd * lyd;
    }
  }

#ifdef ICGN_DEBUG_POI
  bool dbg = (tid == 0 && fabsf(poi_x - 518.0f) < 0.5f && fabsf(poi_y - 20.0f) < 0.5f);
  if (dbg)
  {
    printf("[GPU POI %d at (%.1f,%.1f)] init params: u=%.6f ux=%.6f uy=%.6f v=%.6f vx=%.6f vy=%.6f, ref_norm=%.4f\n",
           poi_idx, poi_x, poi_y, params[0], params[1], params[2], params[3], params[4], params[5], poi_ref_norm);
    printf("[GPU POI %d] inv_H row0: %.6e %.6e %.6e %.6e %.6e %.6e\n",
           poi_idx, s_inv_hessian[0], s_inv_hessian[1], s_inv_hessian[2], s_inv_hessian[3], s_inv_hessian[4], s_inv_hessian[5]);
    printf("[GPU POI %d] inv_H diag: %.6e %.6e %.6e %.6e %.6e %.6e\n",
           poi_idx, s_inv_hessian[0], s_inv_hessian[7], s_inv_hessian[14], s_inv_hessian[21], s_inv_hessian[28], s_inv_hessian[35]);
  }
#endif
  while (iteration < stop_condition)
  {
    ++iteration;
    float sum_local = 0.0f;
    float norm_local = 0.0f;
    float err_local = 0.0f;
    float num_local[kParamCount] = {};

#ifdef ICGN_DEBUG_POI
    if (dbg)
    {
      printf("[GPU POI %d iter %d ENTER] params tid0: u=%.6f ux=%.6f uy=%.6f v=%.6f vx=%.6f vy=%.6f\n",
             poi_idx, iteration, params[0], params[1], params[2], params[3], params[4], params[5]);
    }
    if (tid == 1 && fabsf(poi_x - 518.0f) < 0.5f && fabsf(poi_y - 20.0f) < 0.5f)
    {
      printf("[GPU POI %d iter %d ENTER] params tid1: u=%.6f ux=%.6f uy=%.6f v=%.6f vx=%.6f vy=%.6f\n",
             poi_idx, iteration, params[0], params[1], params[2], params[3], params[4], params[5]);
    }
#endif

    for (int idx = tid; idx < subset_size; idx += blockDim.x)
    {
      int lx = idx % subset_width - subset_radius_x;
      int ly = idx / subset_width - subset_radius_y;

      float x = (1.0f + params[1]) * lx + params[2] * ly + params[0];
      float y = params[4] * lx + (1.0f + params[5]) * ly + params[3];
      float gx = poi_x + x;
      float gy = poi_y + y;
#ifdef ICGN_DEBUG_POI
      if (idx == subset_size / 2 && fabsf(poi_x - 518.0f) < 0.5f && fabsf(poi_y - 20.0f) < 0.5f)
      {
        printf("[GPU POI %d iter %d CTR_SAMPLE tid=%d] lx=%d ly=%d p012=(%.6f,%.6f,%.6f) gx=%.6f gy=%.6f\n",
               poi_idx, iteration, tid, lx, ly, params[0], params[1], params[2], gx, gy);
      }
#endif

      if (isnan(gx) || isnan(gy))
      {
        atomicExch(&failed_flag, 1);
        tar_values[idx] = 0.0f;
        continue;
      }
      if (gx < 1.0f || gy < 1.0f || gx >= tar_width - 2.0f || gy >= tar_height - 2.0f)
      {
        atomicExch(&oob_flag, 1);
        tar_values[idx] = 0.0f;
        continue;
      }

      int ix = static_cast<int>(floorf(gx));
      int iy = static_cast<int>(floorf(gy));
      float dx = gx - ix;
      float dy = gy - iy;
      // Same widening as sampleTargetValue(): the texel index stays 32-bit and
      // only the x16 element scaling is done in size_t (the int32 product
      // overflowed past ~134 Mpx).  In-range addresses are unchanged.
      const float* coeff = tar_b + static_cast<size_t>(iy * tar_width + ix) * 16;
      float value = interpolateBicubic(coeff, dx, dy);
      tar_values[idx] = value;
      sum_local += value;
    }

    // Partially-out-of-bounds rejection (kOobRejectZncc).  The barrier makes
    // oob_flag block-uniform, so every thread leaves the loop together; the
    // decision is taken at the same point as the CPU's -- after the warped
    // subset is reconstructed, before its mean and norm are formed -- so no
    // fabricated sample ever reaches the ZNSSD cost or the Newton step.
    __syncthreads();
    if (oob_flag != 0)
    {
      oob_rejected = true;
      break;
    }

    sum_err_buf[tid] = sum_local;
    reduceAdd(sum_err_buf);
    float tar_mean = sum_err_buf[0] / static_cast<float>(subset_size);

#ifdef ICGN_DEBUG_POI
    if (dbg)
    {
      printf("[GPU POI %d iter %d POSTSAMPLE] tar_mean=%.4f tv[0]=%.4f tv[1]=%.4f tv[100]=%.4f tv[144]=%.4f tv[200]=%.4f tv[288]=%.4f\n",
             poi_idx, iteration, tar_mean, tar_values[0], tar_values[1], tar_values[100], tar_values[144], tar_values[200], tar_values[288]);
    }
#endif

    for (int idx = tid; idx < subset_size; idx += blockDim.x)
    {
      float centered = tar_values[idx] - tar_mean;
      tar_values[idx] = centered;
      norm_local += centered * centered;
    }
    norm_buf[tid] = norm_local;
    reduceAdd(norm_buf);
    float tar_norm = sqrtf(fmaxf(norm_buf[0], 1.0e-12f));

    int num_idx = tid;
#pragma unroll
    for (int s = 0; s < kSdCacheSlots; ++s)
    {
      if (num_idx < subset_size)
      {
        const float error = tar_values[num_idx] * (poi_ref_norm / tar_norm) - s_ref_zero_mean[num_idx];
        err_local += error * error;
        const float err_d = error;
#pragma unroll
        for (int i = 0; i < kParamCount; ++i)
        {
          num_local[i] += sd_cache[s][i] * err_d;
        }
        num_idx += blockDim.x;
      }
    }
    for (; num_idx < subset_size; num_idx += blockDim.x)
    {
      const int lx = num_idx % subset_width - subset_radius_x;
      const int ly = num_idx / subset_width - subset_radius_y;

      const float error = tar_values[num_idx] * (poi_ref_norm / tar_norm) - s_ref_zero_mean[num_idx];
      err_local += error * error;

      const float gxd = s_grad_x[num_idx];
      const float gyd = s_grad_y[num_idx];
      const float lxd = static_cast<float>(lx);
      const float lyd = static_cast<float>(ly);
      const float err_d = error;
      const float sd[kParamCount] = {gxd, gxd * lxd, gxd * lyd, gyd, gyd * lxd, gyd * lyd};
#pragma unroll
      for (int i = 0; i < kParamCount; ++i)
      {
        num_local[i] += sd[i] * err_d;
      }
    }

    sum_err_buf[tid] = err_local;
    for (int i = 0; i < kParamCount; ++i)
    {
      num_buf[tid * kParamCount + i] = num_local[i];
    }
    reduceAdd(sum_err_buf);
    reduceAddVector(num_buf, kParamCount);

    // review-2026-09-02 K1: the six rows of dp = invH * num are independent;
    // thread r accumulates row r in the original c order (bit-identical), then
    // thread 0 composes the warp update from the shared dp.
    if (tid < kParamCount)
    {
      double dp_d = 0.0;
      for (int c = 0; c < kParamCount; ++c)
      {
        dp_d += static_cast<double>(s_inv_hessian[tid * 6 + c]) * num_buf[c];
      }
      s_dp[tid] = static_cast<float>(dp_d);
    }
    __syncthreads();

    if (tid == 0)
    {
      float dp[kParamCount];
      for (int r = 0; r < kParamCount; ++r)
      {
        dp[r] = s_dp[r];
      }

      float dp_abs_max = fmaxf(fmaxf(fabsf(dp[0]), fabsf(dp[3])),
                               fmaxf(fmaxf(fabsf(dp[1]), fabsf(dp[2])),
                                     fmaxf(fabsf(dp[4]), fabsf(dp[5]))));
      if (dp_abs_max > 100.0f || isnan(dp_abs_max))
      {
        failed_flag = 1;
      }

      float current_warp[9];
      float inc_warp[9];
      float inc_inv[9];
      float next_warp[9];

      setWarpMatrix(params, current_warp);
      setWarpMatrix(dp, inc_warp);
      if (!invertAffine3x3(inc_warp, inc_inv))
      {
        failed_flag = 1;
      }
      else
      {
        mul3x3(current_warp, inc_inv, next_warp);
        extractParams(next_warp, params);
      }

      float subset_rx2 = static_cast<float>(subset_radius_x * subset_radius_x);
      float subset_ry2 = static_cast<float>(subset_radius_y * subset_radius_y);
      final_conv = sqrtf(dp[0] * dp[0]
                       + dp[1] * dp[1] * subset_rx2
                       + dp[2] * dp[2] * subset_ry2
                       + dp[3] * dp[3]
                       + dp[4] * dp[4] * subset_rx2
                       + dp[5] * dp[5] * subset_ry2);
      final_znssd = sum_err_buf[0] / fmaxf(poi_ref_norm * poi_ref_norm, 1.0e-12f);
      s_final_conv = final_conv;
      s_final_znssd = final_znssd;
#ifdef ICGN_DEBUG_POI
      if (dbg)
      {
        printf("[GPU POI %d iter %d] tar_mean=%.4f tar_norm=%.4f znssd=%.6f conv=%.6f\n",
               poi_idx, iteration, tar_mean, tar_norm, final_znssd, final_conv);
        printf("[GPU POI %d iter %d] num: %.4e %.4e %.4e %.4e %.4e %.4e\n",
               poi_idx, iteration, num_buf[0], num_buf[1], num_buf[2], num_buf[3], num_buf[4], num_buf[5]);
        printf("[GPU POI %d iter %d] dp:  %.6e %.6e %.6e %.6e %.6e %.6e\n",
               poi_idx, iteration, dp[0], dp[1], dp[2], dp[3], dp[4], dp[5]);
        printf("[GPU POI %d iter %d] new params: u=%.6f ux=%.6f uy=%.6f v=%.6f vx=%.6f vy=%.6f\n",
               poi_idx, iteration, params[0], params[1], params[2], params[3], params[4], params[5]);
      }
#endif
    }
    __syncthreads();

    if (failed_flag != 0 || s_final_conv < conv_criterion)
    {
      break;
    }
  }

  if (tid == 0)
  {
    if (oob_rejected)
    {
      for (int i = 0; i < kParamCount; ++i)
      {
        poi_out[poi_idx].p[i] = poi->p[i];
      }
      poi_out[poi_idx].zncc = kOobRejectZncc;
      poi_out[poi_idx].status = kOobRejectZncc;
      poi_out[poi_idx].iteration = 0.0f;
      poi_out[poi_idx].convergence = 0.0f;
      return;
    }

    for (int i = 0; i < kParamCount; ++i)
    {
      poi_out[poi_idx].p[i] = params[i];
    }
    poi_out[poi_idx].iteration = static_cast<float>(iteration);
    poi_out[poi_idx].convergence = final_conv;

    if (failed_flag != 0 || isnan(final_znssd))
    {
      for (int i = 0; i < kParamCount; ++i)
      {
        poi_out[poi_idx].p[i] = poi->p[i];
      }
      poi_out[poi_idx].zncc = -5.0f;
      poi_out[poi_idx].status = -5.0f;
    }
    else
    {
      poi_out[poi_idx].zncc = 0.5f * (2.0f - final_znssd);
      poi_out[poi_idx].status = (final_conv >= conv_criterion && iteration >= stop_condition) ? -4.0f : 1.0f;
      if (poi_out[poi_idx].status < 0.0f)
      {
        poi_out[poi_idx].zncc = -4.0f;
      }
    }
  }
}

// review-2026-09-02 §2.3: warp-per-POI variant of icgnIterationKernel.  One
// warp (32 lanes) owns one POI; four warps share a fixed 128-thread block,
// so each block processes four POIs and the launch grid is
// ceil(poi_count / kWarpsPerIterationBlock).  Motivation: steady-state POIs
// iterate only ~1.84 times, so the legacy block-per-POI kernel's fixed
// costs (shared staging of 3*subset_size floats, four block-wide tree
// reductions with barriers per iteration, a serial tail with 122 idle
// threads) are comparable to the iteration work itself.
//
// Lane mapping: lane l owns subset pixel indices l, l+32, l+64, ... and
// reads the reference image, its gradients and the target B-spline
// coefficient table straight from global memory (the reference planes are
// L2-resident; the coefficient table is the dominant streaming read either
// way).  No per-pixel data is staged in shared memory.
//
// Reductions: the target sum, target norm, error sum and the six numerator
// components each run through a fixed __shfl_down_sync(0xffffffff, ...)
// butterfly (offsets 16, 8, 4, 2, 1) followed by a __shfl_sync broadcast
// from lane 0 where the whole warp needs the value.  Lanes 0-5 then solve
// one row each of dp = invH * num in double (float numerator promoted, the
// same split as the legacy K1), lane 0 gathers the six rows with shuffles
// and runs the unchanged divergence guard, inverse-compositional warp
// update and convergence computation.  The updated parameters are published
// through the warp's shared slot (s_params) with __syncwarp; the
// convergence value and failure flag travel by shuffle.
//
// Sample caching: when subset_size <= 32*kWarpCacheSlots (production 15x15
// = 225 <= 256) each lane keeps its sampled values in the fixed register
// array tar_cache and samples once per iteration; larger subsets (up to
// 63x63 = 3969 in smoke c4) re-sample per pass.  Both paths evaluate the
// identical float expressions in the identical per-lane order, so they are
// bit-identical to each other.
//
// Determinism: this kernel contains no atomics (the legacy kernel's
// atomicExch failure flag becomes a warp vote) and every reduction follows
// the fixed shuffle tree above, so a POI's result is a pure function of its
// own inputs — independent of blockIdx assignment, stream chunking and
// sibling POIs.  Same build + same input => byte-identical output.  The
// per-lane accumulation grouping (stride 32) and the 32-lane shuffle tree
// differ from the legacy kernel's blockDim-wide shared-memory tree, so warp
// mode is a NEW numerics baseline expected to differ from legacy at the
// ~1e-7 px level; everything else (error formula, numerator, double dp
// solve, divergence guard, warp compose, convergence criterion, iteration
// counting, output records, precompute-status early out) is deliberately
// unchanged.
//
// Shared memory footprint: 96 bytes per block (static s_params, 4 warps x 6
// floats), zero dynamic shared memory — the launch needs no
// cudaFuncSetAttribute opt-in and no occupancy auto-tune.
__global__ void icgnIterationKernelWarp(const float* __restrict__ tar_b, int tar_width, int tar_height,
                                        const float* __restrict__ ref_grad_x, const float* __restrict__ ref_grad_y,
                                        int ref_width, const GpuPoiInput* __restrict__ poi_in, int subset_radius_x,
                                        int subset_radius_y, int subset_size, float conv_criterion,
                                        int stop_condition, const float* __restrict__ ref,
                                        const float* __restrict__ ref_mean,
                                        const float* __restrict__ ref_norm, const float* __restrict__ inv_hessian,
                                        const GpuPoiOutput* __restrict__ poi_precompute,
                                        GpuPoiOutput* __restrict__ poi_out, int poi_count)
{
  constexpr unsigned kFullWarp = 0xffffffffu;
  // Per-warp broadcast slot for the deformation parameters, mirroring the
  // legacy kernel's shared params[].
  __shared__ float s_params[kWarpsPerIterationBlock][kParamCount];
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  // poi_idx is warp-uniform; out-of-range warps (the count % 4 tail) exit
  // together.
  const int poi_idx = blockIdx.x * kWarpsPerIterationBlock + warp_id;
  if (poi_idx >= poi_count)
  {
    return;
  }

  const GpuPoiInput* poi = poi_in + poi_idx;
  const float poi_x = poi->x;
  const float poi_y = poi->y;
  const float precompute_status = poi_precompute[poi_idx].status;
  if (precompute_status < 0.0f)
  {
    if (lane == 0)
    {
      for (int i = 0; i < kParamCount; ++i)
      {
        poi_out[poi_idx].p[i] = poi->p[i];
      }
      poi_out[poi_idx].zncc = poi_precompute[poi_idx].zncc;
      poi_out[poi_idx].iteration = poi_precompute[poi_idx].iteration;
      poi_out[poi_idx].convergence = poi_precompute[poi_idx].convergence;
      poi_out[poi_idx].status = precompute_status;
    }
    return;
  }

  const float poi_ref_mean = ref_mean[poi_idx];
  const float* poi_inv_hessian = inv_hessian + static_cast<long long>(poi_idx) * kHessianCount;
  const float poi_ref_norm = ref_norm[poi_idx];
  const int subset_width = 2 * subset_radius_x + 1;
  const int center_x = static_cast<int>(poi_x);
  const int center_y = static_cast<int>(poi_y);

  if (lane < kParamCount)
  {
    s_params[warp_id][lane] = poi->p[lane];
  }
  __syncwarp(kFullWarp);
  float* params = s_params[warp_id];
  // Row lane of the iteration-invariant inverse Hessian (lanes 0-5 only).
  float invh_row[kParamCount];
  if (lane < kParamCount)
  {
#pragma unroll
    for (int c = 0; c < kParamCount; ++c)
    {
      invh_row[c] = poi_inv_hessian[lane * kParamCount + c];
    }
  }

  float final_znssd = 4.0f;
  float final_conv = 0.0f;
  int iteration = 0;
  bool failed = false;
  bool oob_rejected = false;

  const bool cache_samples = subset_size <= 32 * kWarpCacheSlots;
  float tar_cache[kWarpCacheSlots] = {};

  while (iteration < stop_condition)
  {
    ++iteration;
    bool nan_seen = false;
    bool oob_seen = false;

    // Pass 1: sample the warped target subset and reduce the target sum.
    // Only in-range samples enter the sum, exactly as in the legacy kernel.
    float sum_local = 0.0f;
    if (cache_samples)
    {
#pragma unroll
      for (int s = 0; s < kWarpCacheSlots; ++s)
      {
        const int idx = lane + s * 32;
        if (idx < subset_size)
        {
          const int lx = idx % subset_width - subset_radius_x;
          const int ly = idx / subset_width - subset_radius_y;
          bool sampled = false;
          const float value = sampleTargetValue(tar_b, tar_width, tar_height, poi_x, poi_y,
                                                params, lx, ly, nan_seen, oob_seen, sampled);
          tar_cache[s] = value;
          if (sampled)
          {
            sum_local += value;
          }
        }
      }
    }
    else
    {
      for (int idx = lane; idx < subset_size; idx += 32)
      {
        const int lx = idx % subset_width - subset_radius_x;
        const int ly = idx / subset_width - subset_radius_y;
        bool sampled = false;
        const float value = sampleTargetValue(tar_b, tar_width, tar_height, poi_x, poi_y,
                                              params, lx, ly, nan_seen, oob_seen, sampled);
        if (sampled)
        {
          sum_local += value;
        }
      }
    }
    // One ballot covers both sampling faults, so the common path (neither
    // fault) costs exactly the one warp vote this kernel already paid for the
    // NaN flag; the second ballot runs only when some lane raised a flag.
    //
    // Out of range wins over NaN, matching the CPU: there the interpolator
    // returns its -1.f sentinel for both, and upstream baa3de6's check sees
    // that sentinel before anything else runs.  The decision is taken at the
    // same point as the CPU's -- after the warped subset is sampled, before
    // its mean and norm are formed -- so no fabricated sample ever reaches the
    // ZNSSD cost or the Newton step.  The vote is warp-uniform, so every lane
    // leaves the loop together.
    //
    // A NaN-only iteration keeps this kernel's pre-existing -5 treatment: as
    // in the legacy kernel the flag is merely recorded here and acted on at
    // the end of the iteration, which is why setting it before the sum
    // reduction rather than after changes nothing.
    if (__any_sync(kFullWarp, nan_seen || oob_seen))
    {
      if (__any_sync(kFullWarp, oob_seen))
      {
        oob_rejected = true;
        break;
      }
      failed = true;
    }

    for (int offset = 16; offset > 0; offset /= 2)
    {
      sum_local += __shfl_down_sync(kFullWarp, sum_local, offset);
    }
    const float tar_mean = __shfl_sync(kFullWarp, sum_local, 0) / static_cast<float>(subset_size);

    // Pass 2: center the samples and reduce the target norm.
    float norm_local = 0.0f;
    if (cache_samples)
    {
#pragma unroll
      for (int s = 0; s < kWarpCacheSlots; ++s)
      {
        if (lane + s * 32 < subset_size)
        {
          tar_cache[s] -= tar_mean;
          norm_local += tar_cache[s] * tar_cache[s];
        }
      }
    }
    else
    {
      for (int idx = lane; idx < subset_size; idx += 32)
      {
        const int lx = idx % subset_width - subset_radius_x;
        const int ly = idx / subset_width - subset_radius_y;
        bool sampled = false;
        const float value = sampleTargetValue(tar_b, tar_width, tar_height, poi_x, poi_y,
                                              params, lx, ly, nan_seen, oob_seen, sampled);
        const float centered = value - tar_mean;
        norm_local += centered * centered;
      }
    }
    for (int offset = 16; offset > 0; offset /= 2)
    {
      norm_local += __shfl_down_sync(kFullWarp, norm_local, offset);
    }
    const float tar_norm = sqrtf(fmaxf(__shfl_sync(kFullWarp, norm_local, 0), 1.0e-12f));

    // Pass 3: ZNSSD error sum and the six steepest-descent numerator
    // components (FP32 accumulation, same as the legacy kernel).  The
    // per-pixel quotient poi_ref_norm / tar_norm is hoisted to a per-lane
    // invariant; the division is deterministic, so the value is identical.
    const float norm_scale = poi_ref_norm / tar_norm;
    float err_local = 0.0f;
    float num_local[kParamCount] = {};
    if (cache_samples)
    {
#pragma unroll
      for (int s = 0; s < kWarpCacheSlots; ++s)
      {
        const int idx = lane + s * 32;
        if (idx < subset_size)
        {
          const int lx = idx % subset_width - subset_radius_x;
          const int ly = idx / subset_width - subset_radius_y;
          const int gx0 = center_x + lx;
          const int gy0 = center_y + ly;
          // K2: same zero-mean reference expression as the legacy preamble.
          const float ref_zero_mean = ref[gy0 * ref_width + gx0] - poi_ref_mean;
          const float error = tar_cache[s] * norm_scale - ref_zero_mean;
          err_local += error * error;

          const float gxd = ref_grad_x[gy0 * ref_width + gx0];
          const float gyd = ref_grad_y[gy0 * ref_width + gx0];
          const float lxd = static_cast<float>(lx);
          const float lyd = static_cast<float>(ly);
          const float err_d = error;
          const float sd[kParamCount] = {gxd, gxd * lxd, gxd * lyd, gyd, gyd * lxd, gyd * lyd};
#pragma unroll
          for (int i = 0; i < kParamCount; ++i)
          {
            num_local[i] += sd[i] * err_d;
          }
        }
      }
    }
    else
    {
      for (int idx = lane; idx < subset_size; idx += 32)
      {
        const int lx = idx % subset_width - subset_radius_x;
        const int ly = idx / subset_width - subset_radius_y;
        bool sampled = false;
        const float value = sampleTargetValue(tar_b, tar_width, tar_height, poi_x, poi_y,
                                              params, lx, ly, nan_seen, oob_seen, sampled);
        const float centered = value - tar_mean;
        const int gx0 = center_x + lx;
        const int gy0 = center_y + ly;
        const float ref_zero_mean = ref[gy0 * ref_width + gx0] - poi_ref_mean;
        const float error = centered * norm_scale - ref_zero_mean;
        err_local += error * error;

        const float gxd = ref_grad_x[gy0 * ref_width + gx0];
        const float gyd = ref_grad_y[gy0 * ref_width + gx0];
        const float lxd = static_cast<float>(lx);
        const float lyd = static_cast<float>(ly);
        const float err_d = error;
        const float sd[kParamCount] = {gxd, gxd * lxd, gxd * lyd, gyd, gyd * lxd, gyd * lyd};
#pragma unroll
        for (int i = 0; i < kParamCount; ++i)
        {
          num_local[i] += sd[i] * err_d;
        }
      }
    }

    // One fused butterfly reduces the error sum and the six numerator
    // components; lane 0 then holds every total, which is broadcast back so
    // lanes 0-5 can each solve one row of dp = invH * num in double (float
    // numerator promoted, the same split as the legacy K1).
    for (int offset = 16; offset > 0; offset /= 2)
    {
      err_local += __shfl_down_sync(kFullWarp, err_local, offset);
#pragma unroll
      for (int i = 0; i < kParamCount; ++i)
      {
        num_local[i] += __shfl_down_sync(kFullWarp, num_local[i], offset);
      }
    }
#pragma unroll
    for (int i = 0; i < kParamCount; ++i)
    {
      num_local[i] = __shfl_sync(kFullWarp, num_local[i], 0);
    }

    double dp_d = 0.0;
    if (lane < kParamCount)
    {
#pragma unroll
      for (int c = 0; c < kParamCount; ++c)
      {
        dp_d += static_cast<double>(invh_row[c]) * num_local[c];
      }
    }
    const float dp_row = static_cast<float>(dp_d);
    float dp[kParamCount];
#pragma unroll
    for (int r = 0; r < kParamCount; ++r)
    {
      dp[r] = __shfl_sync(kFullWarp, dp_row, r);
    }

    // Lane 0 runs the unchanged divergence guard, inverse-compositional
    // warp update and convergence computation.  As in the legacy kernel the
    // compose still runs when the divergence guard fires; the updated
    // parameters are discarded only when the output record is written.
    bool solve_failed = false;
    if (lane == 0)
    {
      const float dp_abs_max = fmaxf(fmaxf(fabsf(dp[0]), fabsf(dp[3])),
                                     fmaxf(fmaxf(fabsf(dp[1]), fabsf(dp[2])),
                                           fmaxf(fabsf(dp[4]), fabsf(dp[5]))));
      if (dp_abs_max > 100.0f || isnan(dp_abs_max))
      {
        solve_failed = true;
      }

      float current_warp[9];
      float inc_warp[9];
      float inc_inv[9];
      float next_warp[9];
      setWarpMatrix(params, current_warp);
      setWarpMatrix(dp, inc_warp);
      if (!invertAffine3x3(inc_warp, inc_inv))
      {
        solve_failed = true;
      }
      else
      {
        mul3x3(current_warp, inc_inv, next_warp);
        extractParams(next_warp, params);
      }

      const float subset_rx2 = static_cast<float>(subset_radius_x * subset_radius_x);
      const float subset_ry2 = static_cast<float>(subset_radius_y * subset_radius_y);
      final_conv = sqrtf(dp[0] * dp[0]
                       + dp[1] * dp[1] * subset_rx2
                       + dp[2] * dp[2] * subset_ry2
                       + dp[3] * dp[3]
                       + dp[4] * dp[4] * subset_rx2
                       + dp[5] * dp[5] * subset_ry2);
      final_znssd = err_local / fmaxf(poi_ref_norm * poi_ref_norm, 1.0e-12f);
      failed = failed || solve_failed;
    }

    // Broadcast lane 0's update: the convergence value and failure flag by
    // shuffle, the parameters through the shared slot below (the
    // __syncwarp publishes them before the next iteration reads them), so
    // the break decision and every lane's state stay warp-uniform.
    final_conv = __shfl_sync(kFullWarp, final_conv, 0);
    int failed_flag = failed ? 1 : 0;
    failed_flag = __shfl_sync(kFullWarp, failed_flag, 0);
    failed = failed_flag != 0;
    __syncwarp(kFullWarp);

    if (failed || final_conv < conv_criterion)
    {
      break;
    }
  }

  if (lane == 0)
  {
    if (oob_rejected)
    {
      for (int i = 0; i < kParamCount; ++i)
      {
        poi_out[poi_idx].p[i] = poi->p[i];
      }
      poi_out[poi_idx].zncc = kOobRejectZncc;
      poi_out[poi_idx].status = kOobRejectZncc;
      poi_out[poi_idx].iteration = 0.0f;
      poi_out[poi_idx].convergence = 0.0f;
      return;
    }

    for (int i = 0; i < kParamCount; ++i)
    {
      poi_out[poi_idx].p[i] = params[i];
    }
    poi_out[poi_idx].iteration = static_cast<float>(iteration);
    poi_out[poi_idx].convergence = final_conv;

    if (failed || isnan(final_znssd))
    {
      for (int i = 0; i < kParamCount; ++i)
      {
        poi_out[poi_idx].p[i] = poi->p[i];
      }
      poi_out[poi_idx].zncc = -5.0f;
      poi_out[poi_idx].status = -5.0f;
    }
    else
    {
      poi_out[poi_idx].zncc = 0.5f * (2.0f - final_znssd);
      poi_out[poi_idx].status = (final_conv >= conv_criterion && iteration >= stop_condition) ? -4.0f : 1.0f;
      if (poi_out[poi_idx].status < 0.0f)
      {
        poi_out[poi_idx].zncc = -4.0f;
      }
    }
  }
}

void freeBuffer(float*& ptr)
{
  if (ptr != nullptr)
  {
    cudaFree(ptr);
    ptr = nullptr;
  }
}

template <typename T>
void freeBuffer(T*& ptr)
{
  if (ptr != nullptr)
  {
    cudaFree(ptr);
    ptr = nullptr;
  }
}

template <typename T>
void freeHostBuffer(T*& ptr)
{
  if (ptr != nullptr)
  {
    cudaFreeHost(ptr);
    ptr = nullptr;
  }
}

void clearTarStorageState(CudaBuffers& buffers)
{
  buffers.tar_width = 0;
  buffers.tar_height = 0;
  buffers.d_tar = nullptr;
  buffers.d_tar_b = nullptr;
  buffers.tar_capacity_pixels = 0;
  buffers.tar_prepared = false;
}

// Caller holds registryMutex(): tar_borrowed / tar_donor are read by the
// borrower walks of other instances.
void detachTarViewLocked(CudaBuffers& buffers)
{
  clearTarStorageState(buffers);
  buffers.tar_borrowed = false;
  buffers.tar_donor = nullptr;
}

void detachTarView(CudaBuffers& buffers)
{
  std::lock_guard<std::mutex> lock(registryMutex());
  detachTarViewLocked(buffers);
}

void releaseTarView(CudaBuffers& buffers)
{
  if (!buffers.tar_borrowed)
  {
    freeBuffer(buffers.d_tar);
    freeBuffer(buffers.d_tar_b);
  }
  detachTarView(buffers);
}

bool detachBorrowedTarView(CudaBuffers& buffers)
{
  if (!buffers.tar_borrowed)
  {
    return false;
  }
  detachTarView(buffers);
  return true;
}

void suspendTarBorrowers(const ICGN2D1CudaMvp::Impl& donor)
{
  std::lock_guard<std::mutex> lock(registryMutex());
  for (auto& entry : cudaBufferRegistry())
  {
    CudaBuffers* buffers = entry.buffers;
    if (buffers != nullptr && buffers->tar_borrowed && buffers->tar_donor == &donor)
    {
      clearTarStorageState(*buffers);
      entry.impl->tar_ready = false;
    }
  }
}

void refreshTarBorrowers(const ICGN2D1CudaMvp::Impl& donor)
{
  std::lock_guard<std::mutex> lock(registryMutex());
  std::vector<ICGN2D1CudaMvp::Impl*> refreshed_impls;
  for (auto& entry : cudaBufferRegistry())
  {
    CudaBuffers* recipient = entry.buffers;
    if (recipient == nullptr || !recipient->tar_borrowed || recipient->tar_donor != &donor)
    {
      continue;
    }

    CudaBuffers* source = findCudaBuffersLocked(donor, entry.device_id);
    if (source == nullptr || !source->tar_prepared
        || source->d_tar == nullptr || source->d_tar_b == nullptr)
    {
      throw std::runtime_error("ICGN2D1CudaMvp tar donor refresh failed");
    }
    recipient->tar_width = source->tar_width;
    recipient->tar_height = source->tar_height;
    recipient->d_tar = source->d_tar;
    recipient->d_tar_b = source->d_tar_b;
    recipient->tar_capacity_pixels = source->tar_capacity_pixels;
    recipient->tar_prepared = true;
    refreshed_impls.push_back(entry.impl);
  }

  for (ICGN2D1CudaMvp::Impl* recipient_impl : refreshed_impls)
  {
    recipient_impl->tar_img = donor.tar_img;
    recipient_impl->tar_ready = true;
  }
}

void detachTarBorrowers(const ICGN2D1CudaMvp::Impl& donor)
{
  std::lock_guard<std::mutex> lock(registryMutex());
  for (auto& entry : cudaBufferRegistry())
  {
    CudaBuffers* buffers = entry.buffers;
    if (buffers != nullptr && buffers->tar_borrowed && buffers->tar_donor == &donor)
    {
      detachTarViewLocked(*buffers);
      entry.impl->tar_img = nullptr;
      entry.impl->tar_ready = false;
    }
  }
}

void destroyPersistentStreams(CudaBuffers& buffers)
{
  for (int i = 0; i < buffers.stream_count; ++i)
  {
    if (buffers.streams[i] != nullptr)
    {
      cudaStreamDestroy(buffers.streams[i]);
      buffers.streams[i] = nullptr;
    }
  }
  buffers.stream_count = 0;
}

void ensureStreams(CudaBuffers& buffers, int stream_count)
{
  if (stream_count <= buffers.stream_count)
  {
    return;
  }
  for (int i = buffers.stream_count; i < stream_count; ++i)
  {
    checkCuda(cudaStreamCreate(&buffers.streams[i]), "cudaStreamCreate");
  }
  buffers.stream_count = stream_count;
}

void ensurePoiCapacity(CudaBuffers& buffers, int poi_count, int subset_size)
{
  if (poi_count <= buffers.poi_capacity && subset_size <= buffers.poi_capacity_subset_size)
  {
    return;
  }

  int new_poi_capacity = std::max(poi_count, buffers.poi_capacity);
  int new_subset_capacity = std::max(subset_size, buffers.poi_capacity_subset_size);

  freeBuffer(buffers.d_poi_in);
  freeBuffer(buffers.d_poi_out);
  freeBuffer(buffers.d_poi_precompute);
  freeHostBuffer(buffers.h_poi_in);
  freeHostBuffer(buffers.h_poi_out);
  freeBuffer(buffers.d_ref_mean);
  freeBuffer(buffers.d_ref_norm);
  freeBuffer(buffers.d_inv_hessian);

  checkCuda(cudaMalloc(&buffers.d_poi_in, sizeof(GpuPoiInput) * new_poi_capacity), "cudaMalloc d_poi_in");
  checkCuda(cudaMalloc(&buffers.d_poi_out, sizeof(GpuPoiOutput) * new_poi_capacity), "cudaMalloc d_poi_out");
  checkCuda(cudaMalloc(&buffers.d_poi_precompute, sizeof(GpuPoiOutput) * new_poi_capacity), "cudaMalloc d_poi_precompute");
  checkCuda(cudaHostAlloc(&buffers.h_poi_in, sizeof(GpuPoiInput) * new_poi_capacity,
                          cudaHostAllocDefault), "cudaHostAlloc h_poi_in");
  checkCuda(cudaHostAlloc(&buffers.h_poi_out, sizeof(GpuPoiOutput) * new_poi_capacity,
                          cudaHostAllocDefault), "cudaHostAlloc h_poi_out");
  checkCuda(cudaMalloc(&buffers.d_ref_mean, sizeof(float) * new_poi_capacity),
            "cudaMalloc d_ref_mean");
  checkCuda(cudaMalloc(&buffers.d_ref_norm, sizeof(float) * new_poi_capacity), "cudaMalloc d_ref_norm");
  checkCuda(cudaMalloc(&buffers.d_inv_hessian,
                       sizeof(float) * static_cast<size_t>(new_poi_capacity) * kHessianCount),
            "cudaMalloc d_inv_hessian");
  buffers.poi_capacity = new_poi_capacity;
  buffers.poi_capacity_subset_size = new_subset_capacity;
  buffers.poi_prepared = false;
  buffers.prepared_poi_count = 0;
  buffers.prepared_layout.clear();
  buffers.staged_layout.resize(new_poi_capacity);
}

void ensureGatherIndexCapacity(CudaBuffers& buffers, int count)
{
  if (count <= buffers.gather_index_capacity)
  {
    return;
  }

  int new_capacity = std::max(count, buffers.gather_index_capacity);
  freeBuffer(buffers.d_gather_indices);
  checkCuda(cudaMalloc(&buffers.d_gather_indices, sizeof(int) * new_capacity),
            "cudaMalloc d_gather_indices");
  buffers.gather_index_capacity = new_capacity;
}

void releaseBuffers(CudaBuffers& buffers)
{
  destroyPersistentStreams(buffers);
  freeBuffer(buffers.d_ref);
  releaseTarView(buffers);
  freeBuffer(buffers.d_ref_grad_x);
  freeBuffer(buffers.d_ref_grad_y);
  freeBuffer(buffers.d_poi_in);
  freeBuffer(buffers.d_poi_out);
  freeBuffer(buffers.d_poi_precompute);
  freeHostBuffer(buffers.h_poi_in);
  freeHostBuffer(buffers.h_poi_out);
  freeBuffer(buffers.d_ref_mean);
  freeBuffer(buffers.d_ref_norm);
  freeBuffer(buffers.d_inv_hessian);
  freeBuffer(buffers.d_gather_indices);
  buffers = CudaBuffers();
}

void stagePoiInputAndLayout(CudaBuffers& buffers, const std::vector<POI2D>& poi_queue,
                            int start, int count)
{
  if (count <= 0)
  {
    return;
  }

#pragma omp parallel for if(count >= 4096) schedule(static)
  for (int i = 0; i < count; ++i)
  {
    const POI2D& poi = poi_queue[start + i];
    GpuPoiInput& input = buffers.h_poi_in[i];
    input.x = poi.x;
    input.y = poi.y;
    input.p[0] = poi.deformation.u;
    input.p[1] = poi.deformation.ux;
    input.p[2] = poi.deformation.uy;
    input.p[3] = poi.deformation.v;
    input.p[4] = poi.deformation.vx;
    input.p[5] = poi.deformation.vy;
    buffers.staged_layout[i] = makeLayoutPoint(poi.x, poi.y);
  }
}

void uploadPoiInput(CudaBuffers& buffers, int count)
{
  checkCuda(cudaMemcpy(buffers.d_poi_in, buffers.h_poi_in,
                       sizeof(GpuPoiInput) * static_cast<size_t>(count),
                       cudaMemcpyHostToDevice), "copy poi input");
}

void runPoiPrecomputeFromStagedInput(ICGN2D1CudaMvp::Impl& impl, CudaBuffers& buffers,
                                     int count)
{
  if (count <= 0)
  {
    return;
  }
  if (!buffers.ref_prepared)
  {
    throw std::runtime_error("ICGN2D1CudaMvp::prepareRef must run before prepareTar/compute()");
  }

  ensurePoiCapacity(buffers, count, buffers.subset_size);

  size_t precompute_shared = precomputeSharedBytes(buffers.precompute_block_size);
  precomputePoiKernel<<<count, buffers.precompute_block_size, precompute_shared>>>(
      buffers.d_ref, buffers.d_ref_grad_x, buffers.d_ref_grad_y, buffers.ref_width, buffers.ref_height,
      buffers.d_poi_in, impl.subset_radius_x, impl.subset_radius_y, buffers.subset_size,
      buffers.d_ref_mean, buffers.d_ref_norm, buffers.d_inv_hessian, buffers.d_poi_precompute);
  checkCuda(cudaGetLastError(), "launch precomputePoiKernel");
  checkCuda(cudaDeviceSynchronize(), "sync precomputePoiKernel");
  commitPreparedLayout(buffers, count);
}

void ensurePoiPrecomputed(ICGN2D1CudaMvp::Impl& impl, CudaBuffers& buffers,
                          const std::vector<POI2D>& poi_queue, int start, int count)
{
  if (count <= 0)
  {
    return;
  }
  ensurePoiCapacity(buffers, count, buffers.subset_size);
  stagePoiInputAndLayout(buffers, poi_queue, start, count);
  if (stagedLayoutMatchesPrepared(buffers, count))
  {
    return;
  }
  uploadPoiInput(buffers, count);
  runPoiPrecomputeFromStagedInput(impl, buffers, count);
}

void prepareRefBuffers(ICGN2D1CudaMvp::Impl& impl, CudaBuffers& buffers, int device_id)
{
  buffers.ref_width = impl.ref_img->width;
  buffers.ref_height = impl.ref_img->height;
  buffers.subset_width = 2 * impl.subset_radius_x + 1;
  buffers.subset_height = 2 * impl.subset_radius_y + 1;
  buffers.subset_size = buffers.subset_width * buffers.subset_height;

  size_t ref_bytes = sizeof(float) * buffers.ref_width * buffers.ref_height;

  freeBuffer(buffers.d_ref);
  freeBuffer(buffers.d_ref_grad_x);
  freeBuffer(buffers.d_ref_grad_y);

  checkCuda(cudaMalloc(&buffers.d_ref, ref_bytes), "cudaMalloc d_ref");
  checkCuda(cudaMalloc(&buffers.d_ref_grad_x, ref_bytes), "cudaMalloc d_ref_grad_x");
  checkCuda(cudaMalloc(&buffers.d_ref_grad_y, ref_bytes), "cudaMalloc d_ref_grad_y");

  std::vector<float> ref_host(static_cast<size_t>(buffers.ref_width) * buffers.ref_height);
  for (int y = 0; y < buffers.ref_height; ++y)
  {
    for (int x = 0; x < buffers.ref_width; ++x)
    {
      ref_host[y * buffers.ref_width + x] = impl.ref_img->eg_mat(y, x);
    }
  }

  checkCuda(cudaMemcpy(buffers.d_ref, ref_host.data(), ref_bytes, cudaMemcpyHostToDevice), "copy ref image");

  dim3 block2d(16, 16);
  dim3 ref_grid((buffers.ref_width + block2d.x - 1) / block2d.x,
                (buffers.ref_height + block2d.y - 1) / block2d.y);

  gradientKernel<<<ref_grid, block2d>>>(buffers.d_ref, buffers.ref_width, buffers.ref_height,
                                        buffers.d_ref_grad_x, buffers.d_ref_grad_y);
  checkCuda(cudaGetLastError(), "launch prepareRef kernel");
  checkCuda(cudaDeviceSynchronize(), "sync prepareRef kernel");
  tuneLaunchConfig(buffers);
  configureDynamicSharedMemory(buffers, device_id);
  buffers.ref_prepared = true;
  buffers.poi_prepared = false;
  buffers.prepared_poi_count = 0;
  buffers.prepared_layout.clear();
}

void prepareTarBuffers(ICGN2D1CudaMvp::Impl& impl, CudaBuffers& buffers)
{
  double total_start = wallSeconds();
  double free_sec = 0.0;
  double malloc_sec = 0.0;
  double host_staging_sec = 0.0;
  double h2d_sec = 0.0;
  double kernel_sec = 0.0;

  detachBorrowedTarView(buffers);

  buffers.tar_width = impl.tar_img->width;
  buffers.tar_height = impl.tar_img->height;

  size_t tar_pixels = static_cast<size_t>(buffers.tar_width) * buffers.tar_height;
  size_t tar_bytes = sizeof(float) * tar_pixels;
  size_t b_bytes = sizeof(float) * tar_pixels * 16;

  bool need_alloc = buffers.d_tar == nullptr || buffers.d_tar_b == nullptr
                    || tar_pixels > buffers.tar_capacity_pixels;
  if (need_alloc)
  {
    double free_start = wallSeconds();
    freeBuffer(buffers.d_tar);
    freeBuffer(buffers.d_tar_b);
    buffers.tar_capacity_pixels = 0;
    free_sec = wallSeconds() - free_start;

    double malloc_start = wallSeconds();
    checkCuda(cudaMalloc(&buffers.d_tar, tar_bytes), "cudaMalloc d_tar");
    checkCuda(cudaMalloc(&buffers.d_tar_b, b_bytes), "cudaMalloc d_tar_b");
    buffers.tar_capacity_pixels = tar_pixels;
    malloc_sec = wallSeconds() - malloc_start;
  }

  double host_start = wallSeconds();
  std::vector<float> tar_host(tar_pixels);
  for (int y = 0; y < buffers.tar_height; ++y)
  {
    for (int x = 0; x < buffers.tar_width; ++x)
    {
      tar_host[y * buffers.tar_width + x] = impl.tar_img->eg_mat(y, x);
    }
  }
  host_staging_sec = wallSeconds() - host_start;

  double h2d_start = wallSeconds();
  checkCuda(cudaMemcpy(buffers.d_tar, tar_host.data(), tar_bytes, cudaMemcpyHostToDevice), "copy tar image");
  h2d_sec = wallSeconds() - h2d_start;

  dim3 block2d(16, 16);
  dim3 tar_grid((buffers.tar_width + block2d.x - 1) / block2d.x,
                (buffers.tar_height + block2d.y - 1) / block2d.y);

  double kernel_start = wallSeconds();
  precomputeBMatrixKernel<<<tar_grid, block2d>>>(buffers.d_tar, buffers.tar_width, buffers.tar_height,
                                                 buffers.d_tar_b);
  checkCuda(cudaGetLastError(), "launch prepareTar kernel");
  checkCuda(cudaDeviceSynchronize(), "sync prepareTar kernel");
  kernel_sec = wallSeconds() - kernel_start;

  impl.profile_stats.prepare_tar_calls += 1;
  impl.profile_stats.prepare_tar_free_sec += free_sec;
  impl.profile_stats.prepare_tar_malloc_sec += malloc_sec;
  impl.profile_stats.prepare_tar_host_staging_sec += host_staging_sec;
  impl.profile_stats.prepare_tar_h2d_sec += h2d_sec;
  impl.profile_stats.prepare_tar_kernel_sec += kernel_sec;
  impl.profile_stats.prepare_tar_total_sec += wallSeconds() - total_start;
  buffers.tar_prepared = true;
}

DeviceWork enqueueDeviceWork(ICGN2D1CudaMvp::Impl& impl, std::vector<POI2D>& poi_queue,
                             int start, int count, int device_id)
{
  DeviceWork work;
  work.device_id = device_id;
  work.start = start;
  work.count = count;
  if (count <= 0)
  {
    return work;
  }

  CudaBuffers* buffers = cudaBuffers(impl, device_id);
  if (buffers == nullptr || !buffers->ref_prepared || !buffers->tar_prepared)
  {
    throw std::runtime_error("ICGN2D1CudaMvp::prepareRef/prepareTar must run before compute()");
  }

  checkCuda(cudaSetDevice(device_id), "cudaSetDevice");
  ensurePoiCapacity(*buffers, count, buffers->subset_size);

  work.buffers = buffers;

  double staging_start = wallSeconds();
  stagePoiInputAndLayout(*buffers, poi_queue, start, count);
  impl.profile_stats.compute_poi_in_staging_sec += wallSeconds() - staging_start;

  double guard_start = wallSeconds();
  bool layout_matches = stagedLayoutMatchesPrepared(*buffers, count);
  impl.profile_stats.compute_layout_precompute_guard_sec += wallSeconds() - guard_start;

  double h2d_start = wallSeconds();
  if (!layout_matches)
  {
    // Layout changed (first frame of an event): keep the legacy whole-array
    // synchronous upload so the precompute kernel sees the complete input
    // before the iteration kernels launch.
    uploadPoiInput(*buffers, count);
    impl.profile_stats.compute_h2d_sec += wallSeconds() - h2d_start;

    double precompute_start = wallSeconds();
    runPoiPrecomputeFromStagedInput(impl, *buffers, count);
    impl.profile_stats.compute_layout_precompute_guard_sec += wallSeconds() - precompute_start;
    impl.profile_stats.compute_layout_cache_misses += 1;
  }
  else
  {
    impl.profile_stats.compute_layout_cache_hits += 1;
  }

  // Output initialization moved into icgnIterationKernel from the resident precompute metadata.
  impl.profile_stats.compute_poi_out_init_sec += 0.0;

  double stream_start = wallSeconds();
  int stream_count = std::min(kMaxStreams, count);
  ensureStreams(*buffers, stream_count);
  work.stream_count = stream_count;
  impl.profile_stats.compute_stream_management_sec += wallSeconds() - stream_start;

  int batch_size = (count + stream_count - 1) / stream_count;
  size_t iter_shared = iterationSharedBytes(buffers->iteration_block_size, buffers->subset_size);

  // On the steady-state layout-hit path each stream gets its own
  // H2D upload -> kernel -> D2H retrieval chain for exactly the same POI
  // chunk the legacy whole-array path used.  Stream ordering guarantees
  // copy-kernel-copy execution, and per-POI results do not depend on
  // scheduling, so outputs are bit-identical while the chunk copies of
  // sibling streams overlap kernel execution.
  double enqueue_start = wallSeconds();
  for (int stream_index = 0; stream_index < stream_count; ++stream_index)
  {
    int batch_start = stream_index * batch_size;
    int batch_count = std::min(batch_size, count - batch_start);
    if (batch_count <= 0)
    {
      break;
    }

    cudaStream_t stream = buffers->streams[stream_index];
    if (layout_matches)
    {
      checkCuda(cudaMemcpyAsync(buffers->d_poi_in + batch_start,
                                buffers->h_poi_in + batch_start,
                                sizeof(GpuPoiInput) * static_cast<size_t>(batch_count),
                                cudaMemcpyHostToDevice, stream),
                "async copy poi input chunk");
    }
    if (impl.iteration_kernel_mode == ICGN2D1IterationKernelMode::Warp)
    {
      // Warp-per-POI launch (review-2026-09-02 §2.3): ceil(batch_count / 4)
      // blocks of 128 threads, zero dynamic shared memory; the kernel
      // early-outs whole warps whose POI index is out of range.  The same
      // per-stream chunking as the legacy path applies unchanged.
      const int warp_grid = (batch_count + kWarpsPerIterationBlock - 1) / kWarpsPerIterationBlock;
      icgnIterationKernelWarp<<<warp_grid, kWarpKernelBlockSize, 0, stream>>>(
          buffers->d_tar_b, buffers->tar_width, buffers->tar_height, buffers->d_ref_grad_x, buffers->d_ref_grad_y,
          buffers->ref_width, buffers->d_poi_in + batch_start, impl.subset_radius_x, impl.subset_radius_y,
          buffers->subset_size, impl.conv_criterion, impl.stop_condition,
          buffers->d_ref,
          buffers->d_ref_mean + batch_start,
          buffers->d_ref_norm + batch_start,
          buffers->d_inv_hessian + static_cast<size_t>(batch_start) * kHessianCount,
          buffers->d_poi_precompute + batch_start,
          buffers->d_poi_out + batch_start, batch_count);
    }
    else
    {
      icgnIterationKernel<<<batch_count, buffers->iteration_block_size, iter_shared, stream>>>(
          buffers->d_tar_b, buffers->tar_width, buffers->tar_height, buffers->d_ref_grad_x, buffers->d_ref_grad_y,
          buffers->ref_width, buffers->d_poi_in + batch_start, impl.subset_radius_x, impl.subset_radius_y,
          buffers->subset_size, impl.conv_criterion, impl.stop_condition,
          buffers->d_ref,
          buffers->d_ref_mean + batch_start,
          buffers->d_ref_norm + batch_start,
          buffers->d_inv_hessian + static_cast<size_t>(batch_start) * kHessianCount,
          buffers->d_poi_precompute + batch_start,
          buffers->d_poi_out + batch_start);
    }
    if (layout_matches)
    {
      checkCuda(cudaMemcpyAsync(buffers->h_poi_out + batch_start,
                                buffers->d_poi_out + batch_start,
                                sizeof(GpuPoiOutput) * static_cast<size_t>(batch_count),
                                cudaMemcpyDeviceToHost, stream),
                "async copy poi output chunk");
    }
  }
  work.d2h_enqueued = layout_matches;

  checkCuda(cudaGetLastError(), "launch compute kernels");
  impl.profile_stats.compute_kernel_enqueue_sec += wallSeconds() - enqueue_start;
  impl.profile_stats.compute_device_work_items += 1;
  return work;
}

void finalizeDeviceWork(ICGN2D1CudaMvp::Impl& impl, DeviceWork& work, std::vector<POI2D>& poi_queue)
{
  if (work.count <= 0)
  {
    return;
  }

  checkCuda(cudaSetDevice(work.device_id), "cudaSetDevice");

  auto scatter_range = [&](int begin, int end)
  {
#pragma omp parallel for if(end - begin >= 4096) schedule(static)
    for (int i = begin; i < end; ++i)
    {
      POI2D& poi = poi_queue[work.start + i];
      const GpuPoiInput& poi_in = work.buffers->h_poi_in[i];
      const GpuPoiOutput& poi_out = work.buffers->h_poi_out[i];
      poi.deformation.u = poi_out.p[0];
      poi.deformation.ux = poi_out.p[1];
      poi.deformation.uy = poi_out.p[2];
      poi.deformation.v = poi_out.p[3];
      poi.deformation.vx = poi_out.p[4];
      poi.deformation.vy = poi_out.p[5];
      poi.result.u0 = poi_in.p[0];
      poi.result.v0 = poi_in.p[3];
      poi.result.zncc = poi_out.zncc;
      poi.result.iteration = poi_out.iteration;
      poi.result.convergence = poi_out.convergence;
      poi.subset_radius.x = static_cast<float>(impl.subset_radius_x);
      poi.subset_radius.y = static_cast<float>(impl.subset_radius_y);
    }
  };

  if (!work.d2h_enqueued)
  {
    // Layout-miss frames keep the legacy order: sync every stream, one
    // synchronous whole-array copy, then scatter.
    double kernel_sync_start = wallSeconds();
    for (int i = 0; i < work.stream_count; ++i)
    {
      checkCuda(cudaStreamSynchronize(work.buffers->streams[i]), "cudaStreamSynchronize");
    }
    impl.profile_stats.compute_kernel_sync_sec += wallSeconds() - kernel_sync_start;

    double d2h_start = wallSeconds();
    checkCuda(cudaMemcpy(work.buffers->h_poi_out, work.buffers->d_poi_out,
                         sizeof(GpuPoiOutput) * static_cast<size_t>(work.count),
                         cudaMemcpyDeviceToHost), "copy poi output");
    impl.profile_stats.compute_d2h_sec += wallSeconds() - d2h_start;

    double scatter_start = wallSeconds();
    scatter_range(0, work.count);
    impl.profile_stats.compute_host_scatter_sec += wallSeconds() - scatter_start;
    return;
  }

  // review-2026-09-02 K3: steady-state frames sync each stream and scatter its
  // chunk immediately, overlapping host scatter with the remaining streams'
  // kernels/D2H.  Chunks are disjoint and per-POI writes are independent, so
  // outputs are bit-identical.  Chunk bounds mirror the enqueue batching.
  const int batch_size = (work.count + work.stream_count - 1) / work.stream_count;
  for (int i = 0; i < work.stream_count; ++i)
  {
    const int chunk_begin = i * batch_size;
    if (chunk_begin >= work.count)
    {
      break;
    }
    double kernel_sync_start = wallSeconds();
    checkCuda(cudaStreamSynchronize(work.buffers->streams[i]), "cudaStreamSynchronize");
    impl.profile_stats.compute_kernel_sync_sec += wallSeconds() - kernel_sync_start;

    double scatter_start = wallSeconds();
    scatter_range(chunk_begin, std::min(chunk_begin + batch_size, work.count));
    impl.profile_stats.compute_host_scatter_sec += wallSeconds() - scatter_start;
  }
}

// review-2026-09-02 §3.2 fast path: fill the recipient's ref-side per-POI
// buffers by gathering the donor's resident precompute (indices upload +
// gatherPrecomputedPoiKernel) instead of uploadPoiInput +
// runPoiPrecomputeFromStagedInput.  The gathered bytes are exactly what this
// instance's precompute kernel would have produced (see
// icgn2d1CudaMvpComputeGathered), while the per-frame seeds staged in
// h_poi_in must still go through the device, so the same per-stream
// H2D -> kernel -> D2H pipeline as the layout-hit path runs afterwards.
// Per-POI results do not depend on copy chunking, so outputs are
// bit-identical to the legacy path.
DeviceWork enqueueGatheredDeviceWork(ICGN2D1CudaMvp::Impl& impl, CudaBuffers& buffers,
                                     const CudaBuffers& donor_buffers,
                                     const std::vector<int>& donor_indices,
                                     std::vector<POI2D>& poi_queue, int device_id)
{
  const int count = static_cast<int>(poi_queue.size());
  DeviceWork work;
  work.device_id = device_id;
  work.start = 0;
  work.count = count;
  work.buffers = &buffers;

  checkCuda(cudaSetDevice(device_id), "cudaSetDevice");
  ensurePoiCapacity(buffers, count, buffers.subset_size);

  double staging_start = wallSeconds();
  stagePoiInputAndLayout(buffers, poi_queue, 0, count);
  impl.profile_stats.compute_poi_in_staging_sec += wallSeconds() - staging_start;

  double h2d_start = wallSeconds();
  ensureGatherIndexCapacity(buffers, count);
  checkCuda(cudaMemcpy(buffers.d_gather_indices, donor_indices.data(),
                       sizeof(int) * static_cast<size_t>(count),
                       cudaMemcpyHostToDevice), "copy gather indices");
  impl.profile_stats.compute_h2d_sec += wallSeconds() - h2d_start;

  // review-2026-10-03 S1-6: the gather overwrites this instance's per-POI
  // precompute slots [0, count) with the donor's records, so a layout
  // committed by an earlier prepareRef(queue) or compute() no longer describes
  // them.  Drop that claim before the first slot is written, so that a failed
  // launch cannot leave it standing either: the next compute() or
  // prepareRef(queue) re-runs the precompute instead of hitting a stale cache,
  // and hasPreparedReference() is false until it has.
  buffers.poi_prepared = false;
  buffers.prepared_poi_count = 0;
  buffers.prepared_layout.clear();

  double gather_start = wallSeconds();
  gatherPrecomputedPoiKernel<<<count, kDefaultThreadsPerBlock>>>(
      buffers.d_gather_indices, buffers.subset_size,
      donor_buffers.d_ref_mean, donor_buffers.d_ref_norm,
      donor_buffers.d_inv_hessian, donor_buffers.d_poi_precompute,
      buffers.d_ref_mean, buffers.d_ref_norm,
      buffers.d_inv_hessian, buffers.d_poi_precompute);
  checkCuda(cudaGetLastError(), "launch gatherPrecomputedPoiKernel");
  checkCuda(cudaDeviceSynchronize(), "sync gatherPrecomputedPoiKernel");
  impl.profile_stats.compute_layout_precompute_guard_sec += wallSeconds() - gather_start;
  impl.profile_stats.compute_layout_cache_hits += 1;

  // Output initialization moved into icgnIterationKernel from the resident precompute metadata.
  impl.profile_stats.compute_poi_out_init_sec += 0.0;

  double stream_start = wallSeconds();
  int stream_count = std::min(kMaxStreams, count);
  ensureStreams(buffers, stream_count);
  work.stream_count = stream_count;
  impl.profile_stats.compute_stream_management_sec += wallSeconds() - stream_start;

  int batch_size = (count + stream_count - 1) / stream_count;
  size_t iter_shared = iterationSharedBytes(buffers.iteration_block_size, buffers.subset_size);

  double enqueue_start = wallSeconds();
  for (int stream_index = 0; stream_index < stream_count; ++stream_index)
  {
    int batch_start = stream_index * batch_size;
    int batch_count = std::min(batch_size, count - batch_start);
    if (batch_count <= 0)
    {
      break;
    }

    cudaStream_t stream = buffers.streams[stream_index];
    checkCuda(cudaMemcpyAsync(buffers.d_poi_in + batch_start,
                              buffers.h_poi_in + batch_start,
                              sizeof(GpuPoiInput) * static_cast<size_t>(batch_count),
                              cudaMemcpyHostToDevice, stream),
              "async copy poi input chunk");
    if (impl.iteration_kernel_mode == ICGN2D1IterationKernelMode::Warp)
    {
      // Warp-per-POI launch (review-2026-09-02 §2.3); same chunking as the
      // gathered legacy path.
      const int warp_grid = (batch_count + kWarpsPerIterationBlock - 1) / kWarpsPerIterationBlock;
      icgnIterationKernelWarp<<<warp_grid, kWarpKernelBlockSize, 0, stream>>>(
          buffers.d_tar_b, buffers.tar_width, buffers.tar_height, buffers.d_ref_grad_x, buffers.d_ref_grad_y,
          buffers.ref_width, buffers.d_poi_in + batch_start, impl.subset_radius_x, impl.subset_radius_y,
          buffers.subset_size, impl.conv_criterion, impl.stop_condition,
          buffers.d_ref,
          buffers.d_ref_mean + batch_start,
          buffers.d_ref_norm + batch_start,
          buffers.d_inv_hessian + static_cast<size_t>(batch_start) * kHessianCount,
          buffers.d_poi_precompute + batch_start,
          buffers.d_poi_out + batch_start, batch_count);
    }
    else
    {
      icgnIterationKernel<<<batch_count, buffers.iteration_block_size, iter_shared, stream>>>(
          buffers.d_tar_b, buffers.tar_width, buffers.tar_height, buffers.d_ref_grad_x, buffers.d_ref_grad_y,
          buffers.ref_width, buffers.d_poi_in + batch_start, impl.subset_radius_x, impl.subset_radius_y,
          buffers.subset_size, impl.conv_criterion, impl.stop_condition,
          buffers.d_ref,
          buffers.d_ref_mean + batch_start,
          buffers.d_ref_norm + batch_start,
          buffers.d_inv_hessian + static_cast<size_t>(batch_start) * kHessianCount,
          buffers.d_poi_precompute + batch_start,
          buffers.d_poi_out + batch_start);
    }
    checkCuda(cudaMemcpyAsync(buffers.h_poi_out + batch_start,
                              buffers.d_poi_out + batch_start,
                              sizeof(GpuPoiOutput) * static_cast<size_t>(batch_count),
                              cudaMemcpyDeviceToHost, stream),
              "async copy poi output chunk");
  }
  work.d2h_enqueued = true;

  checkCuda(cudaGetLastError(), "launch compute kernels");
  impl.profile_stats.compute_kernel_enqueue_sec += wallSeconds() - enqueue_start;
  impl.profile_stats.compute_device_work_items += 1;
  return work;
}

} // namespace

// Read-only host query used by ICGN2D1CudaMvp::hasPreparedReference: is the
// per-POI precompute layout from prepareRef(poi_queue) resident on every
// device that call populated?  Touches only the host-side buffer registry --
// no CUDA API call, no allocation, no state change -- so it is safe to mark
// noexcept and to call from a const accessor.
bool icgn2d1CudaMvpHasPreparedPoiLayout(const ICGN2D1CudaMvp::Impl& impl) noexcept
{
  // Mirrors activeDevices() without calling cudaGetDeviceCount(): dual-GPU
  // work always lands on devices 0 and 1.
  const int devices[2] = {impl.dual_gpu ? 0 : impl.device_id, 1};
  const int device_count = impl.dual_gpu ? 2 : 1;
  int prepared_devices = 0;
  long long layout_pois = 0;
  for (int i = 0; i < device_count; ++i)
  {
    const CudaBuffers* buffers = findCudaBuffers(impl, devices[i]);
    if (buffers == nullptr || !buffers->ref_prepared || !buffers->poi_prepared)
    {
      continue;
    }
    ++prepared_devices;
    layout_pois += buffers->prepared_poi_count;
  }
  if (prepared_devices == device_count)
  {
    return true;
  }
  // icgn2d1CudaMvpPrepareRef() only splits a queue holding more than one POI,
  // so a single-POI dual-GPU layout legitimately lives on one device.
  return impl.dual_gpu && prepared_devices == 1 && layout_pois <= 1;
}

void icgn2d1CudaMvpRelease(ICGN2D1CudaMvp::Impl& impl)
{
  detachTarBorrowers(impl);
  // Unregister under the lock, free outside it: once an entry has left the
  // registry no other thread's lookup or walk can reach its buffers.
  for (;;)
  {
    RegistryEntry released;
    {
      std::lock_guard<std::mutex> lock(registryMutex());
      auto& registry = cudaBufferRegistry();
      auto it = std::find_if(registry.begin(), registry.end(),
                             [&impl](const RegistryEntry& entry) { return entry.impl == &impl; });
      if (it == registry.end())
      {
        return;
      }
      released = *it;
      registry.erase(it);
    }
    cudaSetDevice(released.device_id);
    if (released.buffers != nullptr)
    {
      releaseBuffers(*released.buffers);
      delete released.buffers;
    }
  }
}

void icgn2d1CudaMvpPrepareRef(ICGN2D1CudaMvp::Impl& impl)
{
  for (int device_id : activeDevices(impl))
  {
    CudaBuffers* buffers = cudaBuffers(impl, device_id);
    checkCuda(cudaSetDevice(device_id), "cudaSetDevice");
    if (buffers == nullptr)
    {
      throw std::runtime_error("Failed to allocate CUDA buffer registry");
    }
    prepareRefBuffers(impl, *buffers, device_id);
  }
  impl.ref_ready = true;
}

void icgn2d1CudaMvpPrepareRef(ICGN2D1CudaMvp::Impl& impl, const std::vector<POI2D>& poi_queue)
{
  if (!impl.ref_ready)
  {
    icgn2d1CudaMvpPrepareRef(impl);
  }
  if (poi_queue.empty())
  {
    return;
  }

  if (impl.dual_gpu && poi_queue.size() > 1)
  {
    int first_count = static_cast<int>(poi_queue.size() / 2);
    int second_count = static_cast<int>(poi_queue.size()) - first_count;

    checkCuda(cudaSetDevice(0), "cudaSetDevice");
    ensurePoiPrecomputed(impl, *cudaBuffers(impl, 0), poi_queue, 0, first_count);
    checkCuda(cudaSetDevice(1), "cudaSetDevice");
    ensurePoiPrecomputed(impl, *cudaBuffers(impl, 1), poi_queue, first_count, second_count);
  }
  else
  {
    checkCuda(cudaSetDevice(impl.device_id), "cudaSetDevice");
    ensurePoiPrecomputed(impl, *cudaBuffers(impl, impl.device_id), poi_queue,
                         0, static_cast<int>(poi_queue.size()));
  }
}

void icgn2d1CudaMvpPrepareTar(ICGN2D1CudaMvp::Impl& impl)
{
  const std::vector<int> devices = activeDevices(impl);
  suspendTarBorrowers(impl);
  try
  {
    for (int device_id : devices)
    {
      CudaBuffers* buffers = cudaBuffers(impl, device_id);
      checkCuda(cudaSetDevice(device_id), "cudaSetDevice");
      if (buffers == nullptr)
      {
        throw std::runtime_error("Failed to allocate CUDA buffer registry");
      }
      prepareTarBuffers(impl, *buffers);
    }
  }
  catch (...)
  {
    impl.tar_ready = false;
    throw;
  }
  impl.tar_ready = true;
  refreshTarBorrowers(impl);
}

void icgn2d1CudaMvpAdoptPreparedTar(ICGN2D1CudaMvp::Impl& impl,
                                    const ICGN2D1CudaMvp::Impl& donor)
{
  const std::vector<int> devices = activeDevices(impl);
  if (devices != activeDevices(donor))
  {
    throw std::runtime_error("ICGN2D1CudaMvp tar donor must use the same CUDA devices");
  }
  if (!donor.tar_ready)
  {
    throw std::runtime_error("ICGN2D1CudaMvp tar donor has not prepared its target image");
  }

  std::vector<CudaBuffers*> donor_buffers;
  donor_buffers.reserve(devices.size());
  for (int device_id : devices)
  {
    CudaBuffers* buffers = findCudaBuffers(donor, device_id);
    if (buffers == nullptr || !buffers->tar_prepared
        || buffers->d_tar == nullptr || buffers->d_tar_b == nullptr)
    {
      throw std::runtime_error("ICGN2D1CudaMvp tar donor buffers are not prepared");
    }
    if (buffers->tar_borrowed)
    {
      throw std::runtime_error("ICGN2D1CudaMvp does not support chained tar borrowing");
    }
    donor_buffers.push_back(buffers);
  }

  std::vector<CudaBuffers*> recipient_buffers;
  recipient_buffers.reserve(devices.size());
  for (int device_id : devices)
  {
    recipient_buffers.push_back(cudaBuffers(impl, device_id));
  }

  // Validate every device context before changing any recipient ownership.
  for (int device_id : devices)
  {
    checkCuda(cudaSetDevice(device_id), "cudaSetDevice");
  }
  // A recipient that becomes a borrower can no longer own views lent to
  // downstream instances; invalidate those views before releasing its tar.
  detachTarBorrowers(impl);
  impl.tar_ready = false;
  for (size_t i = 0; i < devices.size(); ++i)
  {
    checkCuda(cudaSetDevice(devices[i]), "cudaSetDevice");
    CudaBuffers& recipient = *recipient_buffers[i];
    const CudaBuffers& source = *donor_buffers[i];
    releaseTarView(recipient);
    recipient.tar_width = source.tar_width;
    recipient.tar_height = source.tar_height;
    recipient.d_tar = source.d_tar;
    recipient.d_tar_b = source.d_tar_b;
    recipient.tar_capacity_pixels = source.tar_capacity_pixels;
    recipient.tar_prepared = true;
    {
      std::lock_guard<std::mutex> lock(registryMutex());
      recipient.tar_borrowed = true;
      recipient.tar_donor = &donor;
    }
  }
}

bool icgn2d1CudaMvpDetachBorrowedTar(ICGN2D1CudaMvp::Impl& impl)
{
  suspendTarBorrowers(impl);
  bool detached = false;
  for (int device_id : activeDevices(impl))
  {
    CudaBuffers* buffers = findCudaBuffers(impl, device_id);
    if (buffers != nullptr)
    {
      detached = detachBorrowedTarView(*buffers) || detached;
    }
  }
  return detached;
}

void icgn2d1CudaMvpCompute(ICGN2D1CudaMvp::Impl& impl, std::vector<POI2D>& poi_queue)
{
  if (poi_queue.empty())
  {
    return;
  }

  double compute_start = wallSeconds();
  impl.profile_stats.compute_calls += 1;

  std::vector<DeviceWork> work_items;
  if (impl.dual_gpu && poi_queue.size() > 1)
  {
    int first_count = static_cast<int>(poi_queue.size() / 2);
    int second_count = static_cast<int>(poi_queue.size()) - first_count;
    // Split the POI queue evenly so both devices execute the same algorithm on disjoint work.
    work_items.push_back(enqueueDeviceWork(impl, poi_queue, 0, first_count, 0));
    work_items.push_back(enqueueDeviceWork(impl, poi_queue, first_count, second_count, 1));
  }
  else
  {
    work_items.push_back(enqueueDeviceWork(impl, poi_queue, 0, static_cast<int>(poi_queue.size()), impl.device_id));
  }

  for (auto& work : work_items)
  {
    finalizeDeviceWork(impl, work, poi_queue);
  }
  impl.profile_stats.compute_total_sec += wallSeconds() - compute_start;
}

bool icgn2d1CudaMvpComputeGathered(ICGN2D1CudaMvp::Impl& impl,
                                   const ICGN2D1CudaMvp::Impl& donor,
                                   const std::vector<int>& donor_indices,
                                   std::vector<POI2D>& poi_queue)
{
  if (poi_queue.empty())
  {
    return true;
  }
  if (donor_indices.size() != poi_queue.size())
  {
    return false;
  }

  // review-2026-09-02 §3.2 fast-path preconditions; any miss makes the caller
  // fall back to the legacy compute(), which recomputes the per-POI
  // precompute on this instance.
  //
  // Byte-identity argument: precomputePoiKernel output for a POI is a
  // deterministic function of (reference image, its gradients, POI
  // coordinates, subset radii, reduction block size).  Same physical device,
  // the same reference image object, equal radii and equal tuned precompute
  // block size make the donor's resident bytes exactly what this instance's
  // precompute kernel would produce for the same POI, and the gather kernel
  // copies those bytes verbatim.  icgnIterationKernel reads only
  // status/zncc/iteration/convergence from the precompute record (never its
  // p[] seed copy), so the donor's stale seeds inside the record cannot leak
  // into the output; everything else the iteration kernel consumes (tar
  // B-spline coefficients, per-frame poi_in seeds, this instance's own ref
  // gradients) is staged exactly as on the legacy path.
  if (&impl == &donor)
  {
    // Self-gather could overlap source and destination slots.
    return false;
  }
  if (impl.dual_gpu || donor.dual_gpu || impl.device_id != donor.device_id)
  {
    return false;
  }
  if (impl.subset_radius_x != donor.subset_radius_x
      || impl.subset_radius_y != donor.subset_radius_y)
  {
    return false;
  }
  if (impl.ref_img == nullptr || impl.ref_img != donor.ref_img)
  {
    return false;
  }
  if (!impl.ref_ready || !impl.tar_ready || !donor.ref_ready)
  {
    return false;
  }

  const int device_id = impl.device_id;
  CudaBuffers* buffers = findCudaBuffers(impl, device_id);
  CudaBuffers* donor_buffers = findCudaBuffers(donor, device_id);
  if (buffers == nullptr || donor_buffers == nullptr)
  {
    return false;
  }
  if (!buffers->ref_prepared || !buffers->tar_prepared
      || !donor_buffers->ref_prepared || !donor_buffers->poi_prepared)
  {
    return false;
  }
  if (buffers->subset_size != donor_buffers->subset_size
      || buffers->ref_width != donor_buffers->ref_width
      || buffers->ref_height != donor_buffers->ref_height
      || buffers->precompute_block_size != donor_buffers->precompute_block_size)
  {
    return false;
  }
  if (donor_buffers->d_ref_mean == nullptr || donor_buffers->d_ref_norm == nullptr
      || donor_buffers->d_inv_hessian == nullptr || donor_buffers->d_poi_precompute == nullptr)
  {
    return false;
  }

  // The donor's prepared layout must hold the exact same coordinates at every
  // referenced slot; otherwise the gathered bytes would describe a different
  // POI than the queued one.
  const int count = static_cast<int>(poi_queue.size());
  double guard_start = wallSeconds();
  bool layout_usable = true;
  for (int i = 0; i < count; ++i)
  {
    const int src_index = donor_indices[i];
    if (src_index < 0
        || static_cast<size_t>(src_index) >= donor_buffers->prepared_layout.size())
    {
      layout_usable = false;
      break;
    }
    const LayoutPoint& prepared = donor_buffers->prepared_layout[src_index];
    const LayoutPoint staged = makeLayoutPoint(poi_queue[i].x, poi_queue[i].y);
    if (prepared.x_bits != staged.x_bits || prepared.y_bits != staged.y_bits)
    {
      layout_usable = false;
      break;
    }
  }
  double guard_sec = wallSeconds() - guard_start;
  if (!layout_usable)
  {
    return false;
  }

  double compute_start = wallSeconds();
  impl.profile_stats.compute_calls += 1;
  impl.profile_stats.compute_layout_precompute_guard_sec += guard_sec;

  DeviceWork work = enqueueGatheredDeviceWork(impl, *buffers, *donor_buffers,
                                              donor_indices, poi_queue, device_id);
  finalizeDeviceWork(impl, work, poi_queue);
  impl.profile_stats.compute_total_sec += wallSeconds() - compute_start;
  return true;
}

} // namespace opencorr

#endif
