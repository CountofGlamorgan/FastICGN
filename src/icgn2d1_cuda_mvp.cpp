/*
 * Copyright (c) 2026 Chonglang Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "icgn2d1_cuda_mvp.h"

#include <stdexcept>
#include <utility>
#ifndef OPENCORR_GPU_MVP_USE_CUDA
#include <omp.h>
#include <atomic>
#endif

#include "oc_icgn.h"

namespace opencorr
{

#ifdef OPENCORR_GPU_MVP_USE_CUDA
void icgn2d1CudaMvpRelease(ICGN2D1CudaMvp::Impl& impl);
void icgn2d1CudaMvpPrepareRef(ICGN2D1CudaMvp::Impl& impl);
void icgn2d1CudaMvpPrepareRef(ICGN2D1CudaMvp::Impl& impl, const std::vector<POI2D>& poi_queue);
void icgn2d1CudaMvpPrepareTar(ICGN2D1CudaMvp::Impl& impl);
void icgn2d1CudaMvpAdoptPreparedTar(ICGN2D1CudaMvp::Impl& impl,
                                    const ICGN2D1CudaMvp::Impl& donor);
bool icgn2d1CudaMvpDetachBorrowedTar(ICGN2D1CudaMvp::Impl& impl);
void icgn2d1CudaMvpCompute(ICGN2D1CudaMvp::Impl& impl, std::vector<POI2D>& poi_queue);
bool icgn2d1CudaMvpComputeGathered(ICGN2D1CudaMvp::Impl& impl,
                                   const ICGN2D1CudaMvp::Impl& donor,
                                   const std::vector<int>& donor_indices,
                                   std::vector<POI2D>& poi_queue);
// Read-only query over the CUDA buffer registry; performs no CUDA API call
// and changes no state.
bool icgn2d1CudaMvpHasPreparedPoiLayout(const ICGN2D1CudaMvp::Impl& impl) noexcept;
#endif

ICGN2D1CudaMvp::ICGN2D1CudaMvp(int subset_radius_x, int subset_radius_y,
                               float conv_criterion, int stop_condition, int device_id,
                               bool dual_gpu)
    : impl_(std::make_unique<Impl>(subset_radius_x, subset_radius_y, conv_criterion,
                                   stop_condition, device_id, dual_gpu))
{
}

ICGN2D1CudaMvp::~ICGN2D1CudaMvp()
{
#ifdef OPENCORR_GPU_MVP_USE_CUDA
  if (impl_)
  {
    icgn2d1CudaMvpRelease(*impl_);
  }
#endif
}

void ICGN2D1CudaMvp::setImages(const Image2D& ref_img, const Image2D& tar_img)
{
  setRefImage(ref_img);
  setTarImage(tar_img);
}

#ifndef OPENCORR_GPU_MVP_USE_CUDA
namespace { std::atomic<unsigned long long> cpu_tar_generation{0}; }

void ICGN2D1CudaMvp::ensureCpuThreadCapacity()
{
  const int threads = omp_get_max_threads();
  if (!impl_->cpu_fallback || impl_->cpu_fallback->thread_number != threads)
  {
    auto replacement = std::make_unique<ICGN2D1>(impl_->subset_radius_x,
        impl_->subset_radius_y, impl_->conv_criterion, impl_->stop_condition, threads);
    impl_->cpu_fallback = std::move(replacement);
    impl_->ref_ready = false;
    impl_->tar_ready = false;
  }
  // The CPU API stores non-const pointers but preparation only reads the images.
  // Bind each side independently: prepareRef must work before any target exists.
  impl_->cpu_fallback->ref_img = const_cast<Image2D*>(impl_->ref_img);
  impl_->cpu_fallback->tar_img = const_cast<Image2D*>(impl_->tar_img);
}
#endif

void ICGN2D1CudaMvp::setRefImage(const Image2D& ref_img)
{
  // Treat the setter as notification of new contents, including in-place loads.
  // Fixed-reference sequences should set the reference once, then setTarImage.
  impl_->ref_ready = false;
  impl_->ref_img = &ref_img;
}

void ICGN2D1CudaMvp::setTarImage(const Image2D& tar_img)
{
#ifdef OPENCORR_GPU_MVP_USE_CUDA
  icgn2d1CudaMvpDetachBorrowedTar(*impl_);
#endif
  impl_->tar_ready = false;
  impl_->tar_img = &tar_img;
}

bool ICGN2D1CudaMvp::hasPreparedReference(const Image2D& ref) const noexcept
{
  if (impl_->ref_img != &ref || !impl_->ref_ready)
  {
    return false;
  }
#ifdef OPENCORR_GPU_MVP_USE_CUDA
  // review-2026-09-16 P2-A: ref_ready alone only means the reference image
  // was uploaded, so prepareRef() without a POI queue used to satisfy the
  // caller-facing "prepared reference" contract.  Require the per-POI
  // precompute layout as well.
  return icgn2d1CudaMvpHasPreparedPoiLayout(*impl_);
#else
  return true;
#endif
}

void ICGN2D1CudaMvp::prepare()
{
  if (impl_->ref_img == nullptr || impl_->tar_img == nullptr)
  {
    throw std::runtime_error("ICGN2D1CudaMvp::setImages must be called before prepare()");
  }

  prepareRef();
  prepareTar();
}

void ICGN2D1CudaMvp::prepareRef()
{
  if (impl_->ref_img == nullptr)
  {
    throw std::runtime_error("ICGN2D1CudaMvp::setRefImage must be called before prepareRef()");
  }

#ifdef OPENCORR_GPU_MVP_USE_CUDA
  icgn2d1CudaMvpPrepareRef(*impl_);
#else
  ensureCpuThreadCapacity();
  impl_->ref_ready = false;
  impl_->cpu_fallback->prepareRef();
  impl_->ref_ready = true;
#endif
}

void ICGN2D1CudaMvp::prepareRef(const std::vector<POI2D>& poi_queue)
{
  if (impl_->ref_img == nullptr)
  {
    throw std::runtime_error("ICGN2D1CudaMvp::setRefImage must be called before prepareRef()");
  }

#ifdef OPENCORR_GPU_MVP_USE_CUDA
  icgn2d1CudaMvpPrepareRef(*impl_, poi_queue);
#else
  (void)poi_queue;
  prepareRef();
#endif
}

void ICGN2D1CudaMvp::prepareTar()
{
  if (impl_->tar_img == nullptr)
  {
    throw std::runtime_error("ICGN2D1CudaMvp::setTarImage must be called before prepareTar()");
  }

#ifdef OPENCORR_GPU_MVP_USE_CUDA
  icgn2d1CudaMvpPrepareTar(*impl_);
#else
  ensureCpuThreadCapacity();
  impl_->tar_ready = false;
  impl_->cpu_fallback->prepareTar();
  impl_->cpu_tar_generation = ++cpu_tar_generation;
  impl_->cpu_adopted_generation = 0;
  ++impl_->profile_stats.prepare_tar_calls;
  impl_->tar_ready = true;
#endif
}

void ICGN2D1CudaMvp::adoptPreparedTar(const ICGN2D1CudaMvp& donor)
{
  if (this == &donor)
  {
    throw std::runtime_error("ICGN2D1CudaMvp cannot adopt prepared tar buffers from itself");
  }
  if (donor.impl_ == nullptr || donor.impl_->tar_img == nullptr)
  {
    throw std::runtime_error("ICGN2D1CudaMvp donor must have a target image before adoptPreparedTar()");
  }

#ifdef OPENCORR_GPU_MVP_USE_CUDA
  icgn2d1CudaMvpAdoptPreparedTar(*impl_, *donor.impl_);
  impl_->tar_img = donor.impl_->tar_img;
  impl_->tar_ready = true;
#else
  if (!donor.impl_->tar_ready)
    throw std::runtime_error("ICGN2D1CudaMvp tar donor has not prepared its target image");
  ensureCpuThreadCapacity();
  // A successful preparation has a process-unique generation, not an image or
  // donor address. This also handles donor replacement at a recycled address.
  if (impl_->tar_ready && impl_->cpu_adopted_generation == donor.impl_->cpu_tar_generation)
    return;
  setTarImage(*donor.impl_->tar_img);
  prepareTar();
  impl_->cpu_adopted_generation = donor.impl_->cpu_tar_generation;
#endif
}

void ICGN2D1CudaMvp::setIterationKernelMode(ICGN2D1IterationKernelMode mode)
{
  impl_->iteration_kernel_mode = mode;
}

void ICGN2D1CudaMvp::compute(std::vector<POI2D>& poi_queue)
{
  if (impl_->ref_img == nullptr || impl_->tar_img == nullptr)
  {
    throw std::runtime_error("ICGN2D1CudaMvp::setImages must be called before compute()");
  }

#ifndef OPENCORR_GPU_MVP_USE_CUDA
  ensureCpuThreadCapacity();
#endif
  if (!impl_->ref_ready)
  {
    prepareRef();
  }
  if (!impl_->tar_ready)
  {
    prepareTar();
  }
#ifdef OPENCORR_GPU_MVP_USE_CUDA
  icgn2d1CudaMvpCompute(*impl_, poi_queue);
#else
  impl_->cpu_fallback->compute(poi_queue);
#endif
}

bool ICGN2D1CudaMvp::computeGathered(const ICGN2D1CudaMvp& donor,
                                     const std::vector<int>& donor_indices,
                                     std::vector<POI2D>& poi_queue)
{
  if (impl_->ref_img == nullptr || impl_->tar_img == nullptr)
  {
    throw std::runtime_error("ICGN2D1CudaMvp::setImages must be called before computeGathered()");
  }
  if (donor_indices.size() != poi_queue.size())
  {
    throw std::runtime_error("ICGN2D1CudaMvp::computeGathered requires one donor index per POI");
  }

#ifdef OPENCORR_GPU_MVP_USE_CUDA
  if (poi_queue.empty())
  {
    return true;
  }
  if (!impl_->ref_ready)
  {
    prepareRef();
  }
  if (!impl_->tar_ready)
  {
    prepareTar();
  }
  return icgn2d1CudaMvpComputeGathered(*impl_, *donor.impl_, donor_indices, poi_queue);
#else
  (void)donor;
  (void)donor_indices;
  (void)poi_queue;
  return false;
#endif
}

ICGN2D1CudaMvpProfileStats ICGN2D1CudaMvp::profileStats() const
{
  return impl_->profile_stats;
}

void ICGN2D1CudaMvp::resetProfileStats()
{
  impl_->profile_stats = ICGN2D1CudaMvpProfileStats{};
}

} // namespace opencorr
