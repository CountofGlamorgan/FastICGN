/*
 * Copyright (c) 2026 Chonglang Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifndef OPENCORR_ICGN2D1_CUDA_MVP_H_
#define OPENCORR_ICGN2D1_CUDA_MVP_H_

#include <memory>
#include <vector>

#include "oc_icgn.h"
#include "oc_image.h"
#include "oc_poi.h"

namespace opencorr
{

// review-2026-09-02 §2.3: selects the ICGN iteration kernel.  Warp is the
// default warp-per-POI variant with shuffle reductions (production default
// since 2026-09-02).  Legacy is the pre-2026-09 block-per-POI production
// baseline, kept as the byte-identical A/B reference.  The warp reduction
// order differs from legacy, so it defines a new numerics baseline gated
// separately.
enum class ICGN2D1IterationKernelMode
{
  Legacy,
  Warp
};

struct ICGN2D1CudaMvpProfileStats
{
  long long prepare_tar_calls = 0;
  double prepare_tar_total_sec = 0.0;
  double prepare_tar_free_sec = 0.0;
  double prepare_tar_malloc_sec = 0.0;
  double prepare_tar_host_staging_sec = 0.0;
  double prepare_tar_h2d_sec = 0.0;
  double prepare_tar_kernel_sec = 0.0;

  long long compute_calls = 0;
  long long compute_device_work_items = 0;
  long long compute_layout_cache_hits = 0;
  long long compute_layout_cache_misses = 0;
  double compute_total_sec = 0.0;
  double compute_stream_management_sec = 0.0;
  double compute_poi_in_staging_sec = 0.0;
  double compute_layout_precompute_guard_sec = 0.0;
  double compute_poi_out_init_sec = 0.0;
  double compute_h2d_sec = 0.0;
  double compute_kernel_enqueue_sec = 0.0;
  double compute_kernel_sync_sec = 0.0;
  double compute_d2h_sec = 0.0;
  double compute_host_scatter_sec = 0.0;
};

class ICGN2D1CudaMvp
{
public:
  struct Impl
  {
    Impl(int subset_radius_x_in, int subset_radius_y_in, float conv_criterion_in,
         int stop_condition_in, int device_id_in, bool dual_gpu_in)
        : subset_radius_x(subset_radius_x_in),
          subset_radius_y(subset_radius_y_in),
          conv_criterion(conv_criterion_in),
          stop_condition(stop_condition_in),
          device_id(device_id_in),
          dual_gpu(dual_gpu_in)
    {
    }

    int subset_radius_x;
    int subset_radius_y;
    float conv_criterion;
    int stop_condition;
    int device_id;
    bool dual_gpu;
    ICGN2D1IterationKernelMode iteration_kernel_mode = ICGN2D1IterationKernelMode::Warp;
    const Image2D* ref_img = nullptr;
    const Image2D* tar_img = nullptr;
    bool ref_ready = false;
    bool tar_ready = false;
    ICGN2D1CudaMvpProfileStats profile_stats;

#ifndef OPENCORR_GPU_MVP_USE_CUDA
    std::unique_ptr<ICGN2D1> cpu_fallback;
    unsigned long long cpu_tar_generation = 0;
    unsigned long long cpu_adopted_generation = 0;
#endif
  };

  ICGN2D1CudaMvp(int subset_radius_x, int subset_radius_y,
                 float conv_criterion, int stop_condition, int device_id = 0,
                 bool dual_gpu = false);
  ~ICGN2D1CudaMvp();

  // A setter notifies changed image contents, even at the same object address.
  // Both backends prepare invalidated state lazily in compute. Fixed-reference
  // sequences should set/prepare the reference once, then setTarImage per frame.
  void setImages(const Image2D& ref_img, const Image2D& tar_img);
  void setRefImage(const Image2D& ref_img);
  void setTarImage(const Image2D& tar_img);
  // True only when `ref` is the reference this instance holds, the reference
  // upload is current, and the per-POI precompute layout from
  // prepareRef(poi_queue) is resident on every device that call populated
  // (both devices under dual-GPU).  Defined out of line because the layout
  // state lives in the CUDA translation unit. CPU mode instead requires the
  // reference gradient preparation; its Hessians are computed per POI at run time.
  bool hasPreparedReference(const Image2D& ref) const noexcept;
  void prepare();
  void prepareRef();
  void prepareRef(const std::vector<POI2D>& poi_queue);
  void prepareTar();
  void adoptPreparedTar(const ICGN2D1CudaMvp& donor);
  void setIterationKernelMode(ICGN2D1IterationKernelMode mode);
  void compute(std::vector<POI2D>& poi_queue);
  // review-2026-09-02 §3.2: compute a queue whose entries are copies of the
  // donor's prepared-queue POIs (keyed by donor_indices), gathering the
  // donor's resident per-POI precompute (zero-mean reference subset, norm,
  // inverse Hessian) instead of re-running the precompute kernel.  Returns
  // false when the fast-path preconditions do not hold; the caller must then
  // fall back to compute().  A call that returns true has overwritten this
  // instance's per-POI precompute slots with the donor's records, so a layout
  // from an earlier prepareRef(poi_queue) or compute() is no longer resident:
  // hasPreparedReference() is false until the next compute() or
  // prepareRef(poi_queue) has rebuilt one (review-2026-10-03 S1-6).
  bool computeGathered(const ICGN2D1CudaMvp& donor, const std::vector<int>& donor_indices,
                       std::vector<POI2D>& poi_queue);
  ICGN2D1CudaMvpProfileStats profileStats() const;
  void resetProfileStats();

private:
#ifndef OPENCORR_GPU_MVP_USE_CUDA
  void ensureCpuThreadCapacity();
#endif
  std::unique_ptr<Impl> impl_;
};

} // namespace opencorr

#endif
