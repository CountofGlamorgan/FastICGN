/*
 * Copyright (c) 2026 Chonglang Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include <cmath>
#include <cstdio>

#include <cuda_runtime.h>

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

__device__ void oldReduceAdd(float* buf)
{
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
  {
    __syncthreads();
    if (threadIdx.x < stride)
    {
      buf[threadIdx.x] += buf[threadIdx.x + stride];
    }
  }
  __syncthreads();
}

template <typename T>
__device__ void newReduceAdd(T* buf)
{
  int active = blockDim.x;
  while (active > 1)
  {
    int lower_count = active / 2;
    int upper_offset = active - lower_count;
    __syncthreads();
    if (threadIdx.x < lower_count)
    {
      buf[threadIdx.x] += buf[threadIdx.x + upper_offset];
    }
    active = upper_offset;
  }
  __syncthreads();
}

__global__ void oldKernel(float* out)
{
  __shared__ float buf[128];
  buf[threadIdx.x] = static_cast<float>(threadIdx.x + 1);
  oldReduceAdd(buf);
  if (threadIdx.x == 0)
  {
    out[0] = buf[0];
  }
}

__global__ void newKernel(float* out)
{
  __shared__ float buf[128];
  buf[threadIdx.x] = static_cast<float>(threadIdx.x + 1);
  newReduceAdd(buf);
  if (threadIdx.x == 0)
  {
    out[0] = buf[0];
  }
}

int main()
{
  float* d_out = nullptr;
  float out[3] = {};
  cudaMalloc(&d_out, sizeof(out));

  oldKernel<<<1, 96>>>(d_out);
  cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  newKernel<<<1, 96>>>(d_out + 1);
  cudaMemcpy(out + 1, d_out + 1, sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  newKernel<<<1, 128>>>(d_out + 2);
  cudaMemcpy(out + 2, d_out + 2, sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaFree(d_out);

  const float expected96 = 96.0f * 97.0f / 2.0f;
  const float expected128 = 128.0f * 129.0f / 2.0f;
  const int clamped96 = clampBlockSize(96);

  std::printf("old96=%.0f expected96=%.0f\n", out[0], expected96);
  std::printf("new96=%.0f expected96=%.0f\n", out[1], expected96);
  std::printf("new128=%.0f expected128=%.0f\n", out[2], expected128);
  std::printf("clampBlockSize(96)=%d\n", clamped96);

  if (std::fabs(out[0] - expected96) < 0.5f)
  {
    std::fprintf(stderr, "old 96-thread reduction unexpectedly matched expected sum\n");
    return 1;
  }
  if (std::fabs(out[1] - expected96) > 0.5f || std::fabs(out[2] - expected128) > 0.5f)
  {
    std::fprintf(stderr, "new reduction did not match expected sums\n");
    return 1;
  }
  if (clamped96 != 64)
  {
    std::fprintf(stderr, "clampBlockSize(96) did not return 64\n");
    return 1;
  }
  return 0;
}
