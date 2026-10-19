/*
 * Copyright (c) 2026 Chonglang Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "fftcc2d_cuda.h"

#include <stdexcept>
#include <cmath>

namespace opencorr
{

// Device-side entry points (src/fftcc2d_cuda.cu).
void fftcc2dCudaRelease(FFTCC2DCuda::Impl& impl);
void fftcc2dCudaEnsureRefImage(FFTCC2DCuda::Impl& impl);
void fftcc2dCudaPrepareTar(FFTCC2DCuda::Impl& impl);
void fftcc2dCudaPrepareRef(FFTCC2DCuda::Impl& impl, const std::vector<POI2D>& poi_queue);
void fftcc2dCudaCompute(FFTCC2DCuda::Impl& impl, std::vector<POI2D>& poi_queue,
                        bool use_cache, const size_t* cache_slots);

FFTCC2DCuda::FFTCC2DCuda(int subset_radius_x, int subset_radius_y, int device_id)
    : impl_(std::make_unique<Impl>(subset_radius_x, subset_radius_y, device_id))
{
  if (subset_radius_x <= 0 || subset_radius_x != subset_radius_y || subset_radius_x > 23170)
    throw std::invalid_argument("FFTCC2DCuda supports positive square radii only (1..23170); rectangular FFTs are unsupported");
}

FFTCC2DCuda::~FFTCC2DCuda()
{
  if (impl_)
  {
    fftcc2dCudaRelease(*impl_);
  }
}

void FFTCC2DCuda::setImages(const Image2D& ref_img, const Image2D& tar_img)
{
  setRefImage(ref_img);
  setTarImage(tar_img);
}

void FFTCC2DCuda::setRefImage(const Image2D& ref_img)
{
  // Image2D is mutable; an unchanged address does not identify its contents.
  // Explicitly setting the reference invalidates both the upload and spectra.
  impl_->ref_ready = false;
  impl_->prepared_ready = false;
  impl_->ref_img = &ref_img;
}

void FFTCC2DCuda::setTarImage(const Image2D& tar_img)
{
  // Sequence callers commonly reload the same Image2D for every frame.
  impl_->tar_ready = false;
  impl_->tar_img = &tar_img;
}

void FFTCC2DCuda::prepareTar()
{
  if (impl_->tar_img == nullptr)
  {
    throw std::runtime_error("FFTCC2DCuda::setTarImage must be called before prepareTar()");
  }
  if (!impl_->tar_ready)
  {
    fftcc2dCudaPrepareTar(*impl_);
  }
}

void FFTCC2DCuda::prepareRef(const std::vector<POI2D>& poi_queue)
{
  if (impl_->ref_img == nullptr)
  {
    throw std::runtime_error("FFTCC2DCuda::setRefImage must be called before prepareRef()");
  }

  impl_->prepared_ready = false;
  impl_->prepared_x.resize(poi_queue.size());
  impl_->prepared_y.resize(poi_queue.size());
  for (size_t i = 0; i < poi_queue.size(); ++i)
  {
    impl_->prepared_x[i] = static_cast<int>(poi_queue[i].x);
    impl_->prepared_y[i] = static_cast<int>(poi_queue[i].y);
  }
  fftcc2dCudaPrepareRef(*impl_, poi_queue);
  impl_->prepared_ready = true;
}

void FFTCC2DCuda::ensureReferenceCache()
{
  if (impl_->prepared_ready) return;
  // Reference extraction uses integer POI coordinates, so the retained layout
  // is sufficient to rebuild exactly the previous slots for the new reference.
  std::vector<POI2D> layout;
  layout.reserve(impl_->prepared_x.size());
  for (size_t i = 0; i < impl_->prepared_x.size(); ++i)
    layout.emplace_back(static_cast<float>(impl_->prepared_x[i]),
                        static_cast<float>(impl_->prepared_y[i]));
  prepareRef(layout);
}

namespace
{

bool preparedMatchesQueue(const FFTCC2DCuda::Impl& impl, const std::vector<POI2D>& poi_queue)
{
  if (impl.prepared_x.size() != poi_queue.size())
  {
    return false;
  }
  for (size_t i = 0; i < poi_queue.size(); ++i)
  {
    if (impl.prepared_x[i] != static_cast<int>(poi_queue[i].x)
        || impl.prepared_y[i] != static_cast<int>(poi_queue[i].y))
    {
      return false;
    }
  }
  return true;
}

} // namespace

void FFTCC2DCuda::compute(std::vector<POI2D>& poi_queue)
{
  if (impl_->ref_img == nullptr || impl_->tar_img == nullptr)
  {
    throw std::runtime_error("FFTCC2DCuda::setImages must be called before compute()");
  }
  if (poi_queue.empty())
  {
    return;
  }
  const bool matching_layout = preparedMatchesQueue(*impl_, poi_queue);
  if (matching_layout) ensureReferenceCache();
  if (!impl_->ref_ready)
  {
    fftcc2dCudaEnsureRefImage(*impl_);
  }
  if (!impl_->tar_ready)
  {
    fftcc2dCudaPrepareTar(*impl_);
  }

  if (matching_layout)
  {
    // Identity mapping: queue position i reuses prepared slot i.
    fftcc2dCudaCompute(*impl_, poi_queue, true, nullptr);
    return;
  }
  // Cache miss: whole queue recomputes the reference FFT per POI, mirroring
  // the CPU class's uncached fallback.
  fftcc2dCudaCompute(*impl_, poi_queue, false, nullptr);
}

void FFTCC2DCuda::compute(std::vector<POI2D>& poi_queue, const std::vector<size_t>& cache_indices)
{
  if (impl_->ref_img == nullptr || impl_->tar_img == nullptr)
  {
    throw std::runtime_error("FFTCC2DCuda::setImages must be called before compute()");
  }
  if (cache_indices.size() != poi_queue.size())
  {
    throw std::invalid_argument("FFTCC2DCuda::compute requires one cache index per POI");
  }
  if (poi_queue.empty())
  {
    return;
  }
  if (impl_->prepared_x.empty())
  {
    throw std::logic_error("FFTCC2DCuda indexed compute requires a prepared reference layout");
  }
  for (size_t i = 0; i < poi_queue.size(); ++i)
  {
    const size_t index = cache_indices[i];
    if (index >= impl_->prepared_x.size())
      throw std::out_of_range("FFTCC2DCuda reference cache index out of range");
    if (!std::isfinite(poi_queue[i].x) || !std::isfinite(poi_queue[i].y)
        || impl_->prepared_x[index] != static_cast<int>(poi_queue[i].x)
        || impl_->prepared_y[index] != static_cast<int>(poi_queue[i].y))
      throw std::invalid_argument("FFTCC2DCuda reference cache index does not match POI coordinates");
  }
  ensureReferenceCache();
  if (!impl_->tar_ready) fftcc2dCudaPrepareTar(*impl_);

  fftcc2dCudaCompute(*impl_, poi_queue, true, cache_indices.data());
}

} // namespace opencorr
