/*
 * Copyright (c) 2026 Chonglang Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "fftcc2d_cuda.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>

// review-2026-09-02 §4: batched-cuFFT FFTCC2D.  See fftcc2d_cuda.h for the
// semantic contract with the FFTW path.  Determinism: every reduction here is
// a serial per-POI loop or a plain elementwise kernel — no atomics — and each
// batched cuFFT execution is deterministic for a fixed plan on a fixed GPU,
// so one binary on one GPU reproduces outputs byte for byte.
//
// Scratch is processed in chunks of chunk_capacity POIs so queue length is
// bounded by host memory, not device memory.  cuFFT plans are cached per
// batch size; queues smaller than kSmallBatchQuantumThreshold round their
// batch up to a multiple of kSmallBatchQuantum so the small, ever-changing
// POI-fallback queue sizes reuse a bounded set of plans (padded slots carry
// stale scratch and their outputs are never downloaded — batched transforms
// are independent across batch elements, so real outputs are unaffected).
//
// Coordinate convention (mirrors src/oc_fftcc.cpp exactly for integral POI
// coordinates, which is what the CLI generates): the reference subset is
// centered at (int(x), int(y)), the target subset at (int(x+u0), int(y+v0)),
// both truncated toward zero.  The upstream compute(POI2D*) per-pixel access
// int(((x + c) - rx) + u0) coincides with int(x+u0) + c - rx whenever x and
// u0 are integer-valued; for fractional inputs the two can differ by one
// pixel near sign flips, which is documented but not exercised by the CLI.

namespace opencorr
{
namespace
{

constexpr std::size_t kMaxChunkScratchBytes = std::size_t{1} << 30; // 1 GiB
constexpr int kMinChunkPois = 4096;
constexpr int kMaxChunkPois = 1 << 20;
constexpr int kSmallBatchQuantumThreshold = 4096;
constexpr int kSmallBatchQuantum = 256;
constexpr std::size_t kMaxCachedPlans = 16;
constexpr int kThreadsPerBlock = 256;

struct FftccPoiGeom
{
  float x;
  float y;
  float u0;
  float v0;
};

struct FftccPoiOut
{
  float u;
  float v;
  float u0;
  float v0;
  float zncc;
  int computed;
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
  int freq_len = 0;

  float* d_ref = nullptr;
  std::size_t ref_capacity_pixels = 0;
  float* d_tar = nullptr;
  std::size_t tar_capacity_pixels = 0;

  // Prepared-reference cache (persistent between prepareRef calls).
  cufftComplex* d_ref_freq = nullptr;
  float* d_ref_norm = nullptr;
  int* d_ref_valid = nullptr;
  std::size_t prepared_capacity = 0; // slots; count padded so a padded final chunk fits
  std::size_t prepared_count = 0;

  // Chunk scratch.
  int chunk_capacity = 0;
  FftccPoiGeom* d_geom = nullptr;
  int* d_slots = nullptr;
  float* d_subset_a = nullptr;
  float* d_subset_b = nullptr;
  cufftComplex* d_freq_a = nullptr;
  cufftComplex* d_freq_b = nullptr;
  float* d_norm_a = nullptr;
  float* d_norm_b = nullptr;
  int* d_valid_a = nullptr;
  int* d_valid_b = nullptr;
  FftccPoiOut* d_out = nullptr;
  FftccPoiGeom* h_geom = nullptr;
  int* h_slots = nullptr;
  FftccPoiOut* h_out = nullptr;

  std::map<int, cufftHandle> r2c_plans;
  std::map<int, cufftHandle> c2r_plans;
};

struct RegistryEntry
{
  FFTCC2DCuda::Impl* impl = nullptr;
  CudaBuffers* buffers = nullptr;
};

// review-2026-10-03 S1-7: the registry is process-wide state shared by every
// FFTCC2DCuda instance, so each lookup, insertion and removal holds
// registryMutex().  The mutex is never held across a CUDA or cuFFT call:
// callers take the CudaBuffers pointer out under the lock and work on it
// unlocked, which is safe because one instance is used from one host thread at
// a time (docs/library-contracts.md, "Thread ownership").
std::mutex& registryMutex()
{
  static std::mutex mutex;
  return mutex;
}

// Caller holds registryMutex().
std::vector<RegistryEntry>& bufferRegistry()
{
  static std::vector<RegistryEntry> registry;
  return registry;
}

// Returns the pointer by value, never a reference into the registry vector: an
// insertion by another instance may reallocate the vector at any time, while
// the heap-allocated CudaBuffers itself stays put.
CudaBuffers* cudaBuffers(FFTCC2DCuda::Impl& impl)
{
  std::lock_guard<std::mutex> lock(registryMutex());
  auto& registry = bufferRegistry();
  for (auto& entry : registry)
  {
    if (entry.impl == &impl)
    {
      return entry.buffers;
    }
  }
  registry.push_back({&impl, new CudaBuffers()});
  return registry.back().buffers;
}

void checkCuda(cudaError_t code, const char* context)
{
  if (code != cudaSuccess)
  {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(code));
  }
}

void checkCufft(cufftResult code, const char* context)
{
  if (code != CUFFT_SUCCESS)
  {
    throw std::runtime_error(std::string(context) + ": cuFFT error " + std::to_string(static_cast<int>(code)));
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

// Extract zero-mean subsets for one batch of POIs (one thread per POI,
// serial accumulation in the CPU path's r-then-c order).  Out-of-bounds POIs
// write a zeroed subset with norm 0 and valid 0; their downstream outputs are
// marked not-computed and never written back to the host queue.
__global__ void fftccExtractSubsetKernel(const float* __restrict__ img, int img_width,
                                         int check_width, int check_height,
                                         const FftccPoiGeom* __restrict__ geom, int count,
                                         int rx, int ry, int sw, int sh,
                                         int use_initial_guess,
                                         float* __restrict__ subsets,
                                         float* __restrict__ norms,
                                         int* __restrict__ valid)
{
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= count)
  {
    return;
  }

  const FftccPoiGeom poi = geom[b];
  const int cx = use_initial_guess != 0 ? static_cast<int>(poi.x + poi.u0) : static_cast<int>(poi.x);
  const int cy = use_initial_guess != 0 ? static_cast<int>(poi.y + poi.v0) : static_cast<int>(poi.y);
  const int subset_size = sw * sh;
  float* subset = subsets + static_cast<std::size_t>(b) * subset_size;

  if (cx < rx || cx >= check_width - rx || cy < ry || cy >= check_height - ry)
  {
    for (int i = 0; i < subset_size; ++i)
    {
      subset[i] = 0.f;
    }
    norms[b] = 0.f;
    valid[b] = 0;
    return;
  }

  float mean = 0.f;
  for (int r = 0; r < sh; ++r)
  {
    const float* row = img + static_cast<std::size_t>(cy + r - ry) * img_width + (cx - rx);
    for (int c = 0; c < sw; ++c)
    {
      const float value = row[c];
      subset[r * sw + c] = value;
      mean += value;
    }
  }
  mean /= subset_size;

  float norm = 0.f;
  for (int i = 0; i < subset_size; ++i)
  {
    subset[i] -= mean;
    norm += subset[i] * subset[i];
  }
  norms[b] = norm;
  valid[b] = 1;
}

// zncc_freq = conj(ref_freq) * tar_freq, written back over tar_freq.  The ref
// slot of POI b is slots[b] when an index mapping is given, otherwise
// slot_base + b (identity mapping over the prepared cache, or the per-chunk
// scratch spectra of the uncached path with slot_base = 0).  The persistent
// prepared cache is only ever read here.
__global__ void fftccConjMultiplyKernel(const cufftComplex* __restrict__ ref_freq,
                                        const int* __restrict__ slots, int slot_base,
                                        int freq_len, cufftComplex* __restrict__ tar_freq)
{
  const int b = blockIdx.x;
  const int slot = slots != nullptr ? slots[b] : slot_base + b;
  const cufftComplex* ref = ref_freq + static_cast<std::size_t>(slot) * freq_len;
  cufftComplex* tar = tar_freq + static_cast<std::size_t>(b) * freq_len;
  for (int k = threadIdx.x; k < freq_len; k += blockDim.x)
  {
    const cufftComplex r = ref[k];
    const cufftComplex t = tar[k];
    cufftComplex z;
    z.x = r.x * t.x + r.y * t.y;
    z.y = r.x * t.y - r.y * t.x;
    tar[k] = z;
  }
}

// Argmax with the CPU path's exact tie semantics: scan from index 0, strict
// '>', so the lowest index wins ties; initial sentinel -2 matches the CPU
// class.  One thread per POI, serial scan — deterministic by construction.
__global__ void fftccArgmaxKernel(const float* __restrict__ zncc_map,
                                  const FftccPoiGeom* __restrict__ geom, int count,
                                  int rx, int ry, int sw, int sh,
                                  const float* __restrict__ ref_norms,
                                  const int* __restrict__ ref_valids,
                                  const int* __restrict__ slots, int slot_base,
                                  const float* __restrict__ tar_norms,
                                  const int* __restrict__ tar_valids,
                                  FftccPoiOut* __restrict__ out)
{
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= count)
  {
    return;
  }

  const int slot = slots != nullptr ? slots[b] : slot_base + b;
  const int subset_size = sw * sh;
  FftccPoiOut result;
  result.u = 0.f;
  result.v = 0.f;
  result.u0 = 0.f;
  result.v0 = 0.f;
  result.zncc = 0.f;
  result.computed = 0;

  if (ref_valids[slot] != 0 && tar_valids[b] != 0)
  {
    const float* zncc = zncc_map + static_cast<std::size_t>(b) * subset_size;
    float max_zncc = -2.f;
    int max_zncc_index = 0;
    for (int i = 0; i < subset_size; ++i)
    {
      if (zncc[i] > max_zncc)
      {
        max_zncc = zncc[i];
        max_zncc_index = i;
      }
    }
    int local_u = max_zncc_index % sw;
    int local_v = max_zncc_index / sw;
    if (local_u > rx)
    {
      local_u -= sw;
    }
    if (local_v > ry)
    {
      local_v -= sh;
    }

    const FftccPoiGeom poi = geom[b];
    result.u = static_cast<float>(local_u) + poi.u0;
    result.v = static_cast<float>(local_v) + poi.v0;
    result.u0 = poi.u0;
    result.v0 = poi.v0;
    result.zncc = max_zncc / (sqrtf(ref_norms[slot] * tar_norms[b]) * subset_size);
    result.computed = 1;
  }
  out[b] = result;
}

int fftBatchFor(int count)
{
  if (count <= kSmallBatchQuantumThreshold)
  {
    return ((count + kSmallBatchQuantum - 1) / kSmallBatchQuantum) * kSmallBatchQuantum;
  }
  return count;
}

cufftHandle getPlan(CudaBuffers& buffers, int batch, bool r2c)
{
  std::map<int, cufftHandle>& plans = r2c ? buffers.r2c_plans : buffers.c2r_plans;
  auto it = plans.find(batch);
  if (it != plans.end())
  {
    return it->second;
  }
  if (plans.size() >= kMaxCachedPlans)
  {
    for (auto& entry : plans)
    {
      cufftDestroy(entry.second);
    }
    plans.clear();
  }

  cufftHandle plan;
  // Row-major H x W subsets: n[0] is the slowest-varying (row) dimension.
  const int n[2] = {buffers.subset_height, buffers.subset_width};
  checkCufft(cufftPlanMany(&plan, 2, const_cast<int*>(n), nullptr, 1, 0, nullptr, 1, 0,
                           r2c ? CUFFT_R2C : CUFFT_C2R, batch),
             r2c ? "cufftPlanMany R2C" : "cufftPlanMany C2R");
  plans.emplace(batch, plan);
  return plan;
}

void uploadImage(const Image2D& img, float*& d_img, std::size_t& capacity_pixels)
{
  const std::size_t pixels = static_cast<std::size_t>(img.width) * img.height;
  if (pixels > capacity_pixels)
  {
    freeBuffer(d_img);
    checkCuda(cudaMalloc(&d_img, sizeof(float) * pixels), "cudaMalloc fftcc image");
    capacity_pixels = pixels;
  }

  // Eigen storage is column-major; stage into a row-major host buffer, the
  // same staging pattern as the ICGN MVP path.
  std::vector<float> host(pixels);
  for (int y = 0; y < img.height; ++y)
  {
    for (int x = 0; x < img.width; ++x)
    {
      host[static_cast<std::size_t>(y) * img.width + x] = img.eg_mat(y, x);
    }
  }
  checkCuda(cudaMemcpy(d_img, host.data(), sizeof(float) * pixels, cudaMemcpyHostToDevice),
            "copy fftcc image");
}

void ensureChunkScratch(CudaBuffers& buffers, int subset_width, int subset_height)
{
  if (buffers.chunk_capacity > 0 && buffers.subset_width == subset_width
      && buffers.subset_height == subset_height)
  {
    return;
  }

  buffers.subset_width = subset_width;
  buffers.subset_height = subset_height;
  buffers.subset_size = subset_width * subset_height;
  buffers.freq_len = subset_width * (subset_height / 2 + 1);

  const std::size_t per_poi = 2 * sizeof(float) * buffers.subset_size
                              + 2 * sizeof(cufftComplex) * buffers.freq_len
                              + sizeof(FftccPoiGeom) + sizeof(FftccPoiOut) + 3 * sizeof(int)
                              + 2 * sizeof(float);
  std::size_t capacity = kMaxChunkScratchBytes / per_poi;
  capacity = std::max<std::size_t>(capacity, kMinChunkPois);
  capacity = std::min<std::size_t>(capacity, kMaxChunkPois);
  buffers.chunk_capacity = static_cast<int>(capacity);

  const std::size_t c = static_cast<std::size_t>(buffers.chunk_capacity);
  checkCuda(cudaMalloc(&buffers.d_geom, sizeof(FftccPoiGeom) * c), "cudaMalloc fftcc geom");
  checkCuda(cudaMalloc(&buffers.d_slots, sizeof(int) * c), "cudaMalloc fftcc slots");
  checkCuda(cudaMalloc(&buffers.d_subset_a, sizeof(float) * c * buffers.subset_size),
            "cudaMalloc fftcc subset_a");
  checkCuda(cudaMalloc(&buffers.d_subset_b, sizeof(float) * c * buffers.subset_size),
            "cudaMalloc fftcc subset_b");
  checkCuda(cudaMalloc(&buffers.d_freq_a, sizeof(cufftComplex) * c * buffers.freq_len),
            "cudaMalloc fftcc freq_a");
  checkCuda(cudaMalloc(&buffers.d_freq_b, sizeof(cufftComplex) * c * buffers.freq_len),
            "cudaMalloc fftcc freq_b");
  checkCuda(cudaMalloc(&buffers.d_norm_a, sizeof(float) * c), "cudaMalloc fftcc norm_a");
  checkCuda(cudaMalloc(&buffers.d_norm_b, sizeof(float) * c), "cudaMalloc fftcc norm_b");
  checkCuda(cudaMalloc(&buffers.d_valid_a, sizeof(int) * c), "cudaMalloc fftcc valid_a");
  checkCuda(cudaMalloc(&buffers.d_valid_b, sizeof(int) * c), "cudaMalloc fftcc valid_b");
  checkCuda(cudaMalloc(&buffers.d_out, sizeof(FftccPoiOut) * c), "cudaMalloc fftcc out");
  checkCuda(cudaMallocHost(&buffers.h_geom, sizeof(FftccPoiGeom) * c), "cudaMallocHost fftcc geom");
  checkCuda(cudaMallocHost(&buffers.h_slots, sizeof(int) * c), "cudaMallocHost fftcc slots");
  checkCuda(cudaMallocHost(&buffers.h_out, sizeof(FftccPoiOut) * c), "cudaMallocHost fftcc out");
}

void releaseBuffers(CudaBuffers& buffers)
{
  for (auto& entry : buffers.r2c_plans)
  {
    cufftDestroy(entry.second);
  }
  for (auto& entry : buffers.c2r_plans)
  {
    cufftDestroy(entry.second);
  }
  buffers.r2c_plans.clear();
  buffers.c2r_plans.clear();

  freeBuffer(buffers.d_ref);
  freeBuffer(buffers.d_tar);
  freeBuffer(buffers.d_ref_freq);
  freeBuffer(buffers.d_ref_norm);
  freeBuffer(buffers.d_ref_valid);
  freeBuffer(buffers.d_geom);
  freeBuffer(buffers.d_slots);
  freeBuffer(buffers.d_subset_a);
  freeBuffer(buffers.d_subset_b);
  freeBuffer(buffers.d_freq_a);
  freeBuffer(buffers.d_freq_b);
  freeBuffer(buffers.d_norm_a);
  freeBuffer(buffers.d_norm_b);
  freeBuffer(buffers.d_valid_a);
  freeBuffer(buffers.d_valid_b);
  freeBuffer(buffers.d_out);
  if (buffers.h_geom != nullptr)
  {
    cudaFreeHost(buffers.h_geom);
    buffers.h_geom = nullptr;
  }
  if (buffers.h_slots != nullptr)
  {
    cudaFreeHost(buffers.h_slots);
    buffers.h_slots = nullptr;
  }
  if (buffers.h_out != nullptr)
  {
    cudaFreeHost(buffers.h_out);
    buffers.h_out = nullptr;
  }
  buffers.ref_capacity_pixels = 0;
  buffers.tar_capacity_pixels = 0;
  buffers.prepared_capacity = 0;
  buffers.prepared_count = 0;
  buffers.chunk_capacity = 0;
}

// One chunk of the compute pipeline shared by the cached and uncached paths.
// slot semantics: slots_dev != nullptr -> absolute cache slot per POI;
// otherwise slot = slot_base + b (identity over the prepared cache, or 0-based
// over the chunk-local scratch spectra of the uncached path).
void runComputeChunk(CudaBuffers& buffers, int count, bool use_cache,
                     const int* slots_dev, int slot_base)
{
  const int batch = fftBatchFor(count);
  // Kernels cover exactly the real POIs; only the batched FFTs run on the
  // padded batch.  Padded slots hold stale scratch that never leaves the
  // device (batched transforms are independent across batch elements).
  const int poi_blocks = (count + kThreadsPerBlock - 1) / kThreadsPerBlock;

  if (!use_cache)
  {
    fftccExtractSubsetKernel<<<poi_blocks, kThreadsPerBlock>>>(
        buffers.d_ref, buffers.ref_width, buffers.ref_width, buffers.ref_height,
        buffers.d_geom, count, buffers.subset_width / 2, buffers.subset_height / 2,
        buffers.subset_width, buffers.subset_height, 0, buffers.d_subset_a,
        buffers.d_norm_a, buffers.d_valid_a);
    checkCuda(cudaGetLastError(), "launch fftcc ref extract");
    checkCufft(cufftExecR2C(getPlan(buffers, batch, true), buffers.d_subset_a, buffers.d_freq_a),
               "cufftExecR2C ref");
  }

  fftccExtractSubsetKernel<<<poi_blocks, kThreadsPerBlock>>>(
      buffers.d_tar, buffers.tar_width, buffers.ref_width, buffers.ref_height,
      buffers.d_geom, count, buffers.subset_width / 2, buffers.subset_height / 2,
      buffers.subset_width, buffers.subset_height, 1, buffers.d_subset_b,
      buffers.d_norm_b, buffers.d_valid_b);
  checkCuda(cudaGetLastError(), "launch fftcc tar extract");
  checkCufft(cufftExecR2C(getPlan(buffers, batch, true), buffers.d_subset_b, buffers.d_freq_b),
             "cufftExecR2C tar");

  const cufftComplex* ref_freq = use_cache ? buffers.d_ref_freq : buffers.d_freq_a;
  fftccConjMultiplyKernel<<<count, kThreadsPerBlock>>>(ref_freq, slots_dev, slot_base,
                                                       buffers.freq_len, buffers.d_freq_b);
  checkCuda(cudaGetLastError(), "launch fftcc conj multiply");

  // d_subset_a is free scratch at this point (its FFT was already taken, or
  // the cached path never fills it); reuse it for the C2R output.
  checkCufft(cufftExecC2R(getPlan(buffers, batch, false), buffers.d_freq_b, buffers.d_subset_a),
             "cufftExecC2R zncc");

  fftccArgmaxKernel<<<poi_blocks, kThreadsPerBlock>>>(
      buffers.d_subset_a, buffers.d_geom, count, buffers.subset_width / 2,
      buffers.subset_height / 2, buffers.subset_width, buffers.subset_height,
      use_cache ? buffers.d_ref_norm : buffers.d_norm_a,
      use_cache ? buffers.d_ref_valid : buffers.d_valid_a, slots_dev, slot_base,
      buffers.d_norm_b, buffers.d_valid_b, buffers.d_out);
  checkCuda(cudaGetLastError(), "launch fftcc argmax");

}

} // namespace

void fftcc2dCudaRelease(FFTCC2DCuda::Impl& impl)
{
  // Unregister under the lock, free outside it: once an entry has left the
  // registry no other thread can reach its buffers.
  for (;;)
  {
    CudaBuffers* buffers = nullptr;
    {
      std::lock_guard<std::mutex> lock(registryMutex());
      auto& registry = bufferRegistry();
      auto it = std::find_if(registry.begin(), registry.end(),
                             [&impl](const RegistryEntry& entry) { return entry.impl == &impl; });
      if (it == registry.end())
      {
        return;
      }
      buffers = it->buffers;
      registry.erase(it);
    }
    cudaSetDevice(impl.device_id);
    if (buffers != nullptr)
    {
      releaseBuffers(*buffers);
      delete buffers;
    }
  }
}

void fftcc2dCudaEnsureRefImage(FFTCC2DCuda::Impl& impl)
{
  if (impl.ref_img == nullptr)
  {
    throw std::runtime_error("FFTCC2DCuda: reference image is not set");
  }
  checkCuda(cudaSetDevice(impl.device_id), "cudaSetDevice");
  CudaBuffers* buffers = cudaBuffers(impl);
  uploadImage(*impl.ref_img, buffers->d_ref, buffers->ref_capacity_pixels);
  buffers->ref_width = impl.ref_img->width;
  buffers->ref_height = impl.ref_img->height;
  impl.ref_ready = true;
}

void fftcc2dCudaPrepareTar(FFTCC2DCuda::Impl& impl)
{
  if (impl.tar_img == nullptr)
  {
    throw std::runtime_error("FFTCC2DCuda: target image is not set");
  }
  checkCuda(cudaSetDevice(impl.device_id), "cudaSetDevice");
  CudaBuffers* buffers = cudaBuffers(impl);
  uploadImage(*impl.tar_img, buffers->d_tar, buffers->tar_capacity_pixels);
  buffers->tar_width = impl.tar_img->width;
  buffers->tar_height = impl.tar_img->height;
  impl.tar_ready = true;
}

void fftcc2dCudaPrepareRef(FFTCC2DCuda::Impl& impl, const std::vector<POI2D>& poi_queue)
{
  checkCuda(cudaSetDevice(impl.device_id), "cudaSetDevice");
  if (!impl.ref_ready)
  {
    fftcc2dCudaEnsureRefImage(impl);
  }
  CudaBuffers* buffers = cudaBuffers(impl);

  const int subset_width = 2 * impl.subset_radius_x;
  const int subset_height = 2 * impl.subset_radius_y;
  ensureChunkScratch(*buffers, subset_width, subset_height);

  const std::size_t count = poi_queue.size();
  buffers->prepared_count = count;
  if (count == 0)
  {
    return;
  }

  // The final chunk may round its batch up to the small-queue threshold;
  // pad the cache allocation so those padded FFT outputs stay in bounds.
  const std::size_t capacity = count + kSmallBatchQuantumThreshold;
  if (capacity > buffers->prepared_capacity)
  {
    freeBuffer(buffers->d_ref_freq);
    freeBuffer(buffers->d_ref_norm);
    freeBuffer(buffers->d_ref_valid);
    checkCuda(cudaMalloc(&buffers->d_ref_freq, sizeof(cufftComplex) * capacity * buffers->freq_len),
              "cudaMalloc fftcc ref_freq cache");
    checkCuda(cudaMalloc(&buffers->d_ref_norm, sizeof(float) * capacity),
              "cudaMalloc fftcc ref_norm cache");
    checkCuda(cudaMalloc(&buffers->d_ref_valid, sizeof(int) * capacity),
              "cudaMalloc fftcc ref_valid cache");
    buffers->prepared_capacity = capacity;
  }

  for (std::size_t chunk_start = 0; chunk_start < count; chunk_start += buffers->chunk_capacity)
  {
    const int chunk_count = static_cast<int>(std::min<std::size_t>(
        buffers->chunk_capacity, count - chunk_start));
    const int batch = fftBatchFor(chunk_count);

    for (int i = 0; i < chunk_count; ++i)
    {
      const POI2D& poi = poi_queue[chunk_start + i];
      buffers->h_geom[i] = FftccPoiGeom{poi.x, poi.y, poi.deformation.u, poi.deformation.v};
    }
    checkCuda(cudaMemcpy(buffers->d_geom, buffers->h_geom, sizeof(FftccPoiGeom) * chunk_count,
                         cudaMemcpyHostToDevice),
              "copy fftcc prepareRef geom");

    // Cover exactly the real POIs; padded slots keep stale scratch and their
    // spectra (written into the padded cache tail) are never consumed.
    const int poi_blocks = (chunk_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    fftccExtractSubsetKernel<<<poi_blocks, kThreadsPerBlock>>>(
        buffers->d_ref, buffers->ref_width, buffers->ref_width, buffers->ref_height,
        buffers->d_geom, chunk_count, impl.subset_radius_x, impl.subset_radius_y,
        subset_width, subset_height, 0, buffers->d_subset_a, buffers->d_norm_a,
        buffers->d_valid_a);
    checkCuda(cudaGetLastError(), "launch fftcc prepareRef extract");

    cufftComplex* cache_out = buffers->d_ref_freq
                              + chunk_start * static_cast<std::size_t>(buffers->freq_len);
    checkCufft(cufftExecR2C(getPlan(*buffers, batch, true), buffers->d_subset_a, cache_out),
               "cufftExecR2C prepareRef");
    checkCuda(cudaMemcpy(buffers->d_ref_norm + chunk_start, buffers->d_norm_a,
                         sizeof(float) * chunk_count, cudaMemcpyDeviceToDevice),
              "copy fftcc ref norms");
    checkCuda(cudaMemcpy(buffers->d_ref_valid + chunk_start, buffers->d_valid_a,
                         sizeof(int) * chunk_count, cudaMemcpyDeviceToDevice),
              "copy fftcc ref valids");
  }
  checkCuda(cudaDeviceSynchronize(), "sync fftcc prepareRef");
}

void fftcc2dCudaCompute(FFTCC2DCuda::Impl& impl, std::vector<POI2D>& poi_queue, bool use_cache,
                        const size_t* cache_slots)
{
  checkCuda(cudaSetDevice(impl.device_id), "cudaSetDevice");
  CudaBuffers* buffers = cudaBuffers(impl);
  if (!impl.ref_ready || !impl.tar_ready)
  {
    throw std::runtime_error("FFTCC2DCuda: images must be uploaded before compute()");
  }
  if (use_cache && buffers->prepared_count == 0)
  {
    throw std::runtime_error("FFTCC2DCuda: cached compute requires a prepared reference");
  }
  // review-2026-10-03 (fftcc F3): the target extraction below is bounds-checked
  // against the reference size and strides by the target width, so a target of
  // another size was read out of bounds.  Every compute overload ends up here,
  // and these are the sizes actually uploaded to the device.
  if (buffers->tar_width != buffers->ref_width || buffers->tar_height != buffers->ref_height)
  {
    throw std::invalid_argument(
        "FFTCC2DCuda::compute: reference image is " + std::to_string(buffers->ref_width) + "x"
        + std::to_string(buffers->ref_height) + " but target image is "
        + std::to_string(buffers->tar_width) + "x" + std::to_string(buffers->tar_height)
        + "; reference and target sizes must match");
  }

  const int subset_width = 2 * impl.subset_radius_x;
  const int subset_height = 2 * impl.subset_radius_y;
  ensureChunkScratch(*buffers, subset_width, subset_height);

  const std::size_t count = poi_queue.size();
  for (std::size_t chunk_start = 0; chunk_start < count; chunk_start += buffers->chunk_capacity)
  {
    const int chunk_count = static_cast<int>(std::min<std::size_t>(
        buffers->chunk_capacity, count - chunk_start));

    for (int i = 0; i < chunk_count; ++i)
    {
      const POI2D& poi = poi_queue[chunk_start + i];
      buffers->h_geom[i] = FftccPoiGeom{poi.x, poi.y, poi.deformation.u, poi.deformation.v};
    }
    checkCuda(cudaMemcpy(buffers->d_geom, buffers->h_geom, sizeof(FftccPoiGeom) * chunk_count,
                         cudaMemcpyHostToDevice),
              "copy fftcc geom");

    const int* slots_dev = nullptr;
    int slot_base = 0;
    if (use_cache)
    {
      if (cache_slots != nullptr)
      {
        for (int i = 0; i < chunk_count; ++i)
        {
          buffers->h_slots[i] = static_cast<int>(cache_slots[chunk_start + i]);
        }
        checkCuda(cudaMemcpy(buffers->d_slots, buffers->h_slots, sizeof(int) * chunk_count,
                             cudaMemcpyHostToDevice),
                  "copy fftcc cache slots");
        slots_dev = buffers->d_slots;
      }
      else
      {
        slot_base = static_cast<int>(chunk_start);
      }
    }

    runComputeChunk(*buffers, chunk_count, use_cache, slots_dev, slot_base);

    checkCuda(cudaMemcpy(buffers->h_out, buffers->d_out, sizeof(FftccPoiOut) * chunk_count,
                         cudaMemcpyDeviceToHost),
              "copy fftcc results");

    for (int i = 0; i < chunk_count; ++i)
    {
      const FftccPoiOut& result = buffers->h_out[i];
      if (result.computed == 0)
      {
        continue;
      }
      POI2D& poi = poi_queue[chunk_start + i];
      poi.deformation.u = result.u;
      poi.deformation.v = result.v;
      poi.result.u0 = result.u0;
      poi.result.v0 = result.v0;
      poi.result.zncc = result.zncc;
    }
  }
  checkCuda(cudaDeviceSynchronize(), "sync fftcc compute");
}

} // namespace opencorr
