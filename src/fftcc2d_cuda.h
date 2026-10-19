/*
 * Copyright (c) 2026 Chonglang Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifndef OPENCORR_FFTCC2D_CUDA_H_
#define OPENCORR_FFTCC2D_CUDA_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "oc_image.h"
#include "oc_poi.h"

namespace opencorr
{

// review-2026-09-02 §4: GPU FFTCC2D — batched-cuFFT reimplementation of the
// FFTW-based FFTCC2D initial-guess search (src/oc_fftcc.cpp), dropping the
// GPL-licensed FFTW dependency from the production CUDA binary.  The class
// replicates the CPU path's semantics exactly:
//  - zero-mean subsets, unnormalized R2C/C2R round trip (cuFFT and FFTW share
//    the same normalization convention), zncc = max / (sqrt(ref_norm*tar_norm)
//    * subset_size);
//  - argmax scans from index 0 with a strict '>' comparison, so ties resolve
//    to the lowest linear index (no atomics; results are run-to-run
//    deterministic on a fixed GPU and binary);
//  - local displacement wraps to negative when it exceeds the subset radius;
//  - a POI that fails the boundary check is returned untouched (no writes to
//    deformation or result), and the prepared-reference cache semantics of
//    prepareRef/compute/compute(queue, cache_indices) mirror the CPU class,
//    with lazy cache rebuilds and checked indexed calls.
// Only positive square radii are supported. Rectangles are rejected by the
// host constructor before CUDA allocation, and the packed FFT geometry behind
// that rejection is square-only in two independent ways:
//  - the cuFFT plan is built with n = {height, width}, the correct row-major
//    H x W mapping, while upstream FFTW calls fftwf_plan_dft_r2c_2d(width,
//    height, ...) and so reinterprets the packed subset with swapped
//    dimensions.  The two coincide for square subsets (the only production
//    configuration), but the NON-square behavior of this class has NOT been
//    cross-validated against the FFTW path;
//  - ensureChunkScratch (src/fftcc2d_cuda.cu) sizes the R2C spectra as
//    freq_len = subset_width * (subset_height / 2 + 1), while the plan above
//    needs subset_height * (subset_width / 2 + 1).  The two agree only when
//    width == height; a rectangular subset would under-allocate d_freq_a /
//    d_freq_b / d_ref_freq and let cuFFT write past the end of a device
//    buffer.  The constructor guard makes that unreachable today.
// Opening a rectangular path therefore means correcting freq_len together with
// the d_ref_freq stride and the per-slot offsets that index it, not merely
// relaxing the constructor guard. See docs/library-contracts.md.
class FFTCC2DCuda
{
public:
  struct Impl
  {
    Impl(int subset_radius_x_in, int subset_radius_y_in, int device_id_in)
        : subset_radius_x(subset_radius_x_in),
          subset_radius_y(subset_radius_y_in),
          device_id(device_id_in)
    {
    }

    int subset_radius_x;
    int subset_radius_y;
    int device_id;
    const Image2D* ref_img = nullptr;
    const Image2D* tar_img = nullptr;
    bool ref_ready = false;  // device-side reference image upload is current
    bool tar_ready = false;  // device-side target image upload is current

    // Host-side mirror of the CPU class's prepared_ref_ metadata: cache
    // validity is decided here (coordinates), while the cached spectra and
    // norms live on the device.
    std::vector<int> prepared_x;
    std::vector<int> prepared_y;
    bool prepared_ready = false;
  };

  FFTCC2DCuda(int subset_radius_x, int subset_radius_y, int device_id = 0);
  ~FFTCC2DCuda();

  // Setters notify changed contents even when Image2D addresses are reused.
  // setImages invalidates both sides; setRefImage also invalidates cached FFTs.
  // For a fixed reference, set/prepare it once and call setTarImage per frame.
  void setImages(const Image2D& ref_img, const Image2D& tar_img);
  void setRefImage(const Image2D& ref_img);
  void setTarImage(const Image2D& tar_img);
  const Image2D* refImage() const noexcept { return impl_->ref_img; }
  bool hasPreparedReference(const Image2D& ref) const noexcept
  { return impl_->ref_img == &ref && impl_->ref_ready && impl_->prepared_ready; }
  // Uploads once after each setTarImage/setImages notification. A second call
  // without a setter reuses the same prepared image; compute also prepares lazily.
  void prepareTar();
  void prepareRef(const std::vector<POI2D>& poi_queue);
  // Reference and target must have the same width and height. Both compute
  // overloads throw std::invalid_argument naming the two sizes otherwise and
  // leave the queue untouched: the target extraction is bounds-checked against
  // the reference size, so a smaller target would be read out of bounds.
  void compute(std::vector<POI2D>& poi_queue);
  // Indexed calls require a layout established by prepareRef(queue).
  // Reference changes lazily rebuild that layout. Invalid indices or
  // mismatched integer coordinates throw; indices are never silently ignored.
  void compute(std::vector<POI2D>& poi_queue, const std::vector<size_t>& cache_indices);

private:
  void ensureReferenceCache();
  std::unique_ptr<Impl> impl_;
};

} // namespace opencorr

#endif
