/*
 * Copyright (c) 2026 Chonglang Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

// Regression probes for mutable Image2D objects reused at the same address.
// Compare each reused solver with a fresh solver on identical current images.
#include "fftcc2d_cuda.h"
#include "icgn2d1_cuda_mvp.h"
#ifdef OPENCORR_HAS_FFTW
#include "oc_fftcc.h"
#endif

#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <stdexcept>
#include <omp.h>

using namespace opencorr;

namespace {
int failures = 0;

void check(const std::string& name, bool passed)
{
  failures += !passed;
  std::cout << (passed ? "PASS " : "FAIL ") << name << '\n';
}

template <class Error, class Operation>
void rejects(const std::string& name, Operation operation)
{
  bool rejected = false;
  try { operation(); }
  catch (const Error&) { rejected = true; }
  check(name, rejected);
}

std::vector<POI2D> grid(float seed = 0.f)
{
  std::vector<POI2D> result;
  for (int y = 32; y < 96; y += 8)
    for (int x = 32; x < 160; x += 8)
    {
      result.emplace_back(static_cast<float>(x), static_cast<float>(y));
      result.back().deformation.u = seed;
    }
  return result;
}

bool equalFloat(float a, float b)
{
  return a == b || (std::isnan(a) && std::isnan(b));
}

void compare(const std::string& name, const std::vector<POI2D>& actual,
             const std::vector<POI2D>& expected, bool icgn)
{
  int mismatches = 0, valid = 0;
  for (size_t i = 0; i < expected.size(); ++i)
  {
    const auto& a = actual[i];
    const auto& b = expected[i];
    valid += std::isfinite(b.result.zncc) && b.result.zncc >= 0.f;
    bool same = equalFloat(a.deformation.u, b.deformation.u)
                && equalFloat(a.deformation.v, b.deformation.v)
                && equalFloat(a.result.zncc, b.result.zncc);
    if (icgn)
      same = same && equalFloat(a.deformation.ux, b.deformation.ux)
                  && equalFloat(a.deformation.uy, b.deformation.uy)
                  && equalFloat(a.deformation.vx, b.deformation.vx)
                  && equalFloat(a.deformation.vy, b.deformation.vy);
    mismatches += !same;
  }
  const bool pass = mismatches == 0 && valid >= static_cast<int>(expected.size() * .95);
  failures += !pass;
  std::cout << (pass ? "PASS " : "FAIL ") << name << " mismatches=" << mismatches
            << " fresh_valid=" << valid << "/" << expected.size() << '\n';
}

void fftTarget(const std::string& dir, int device, const std::vector<int>& shifts, bool indexed)
{
  Image2D ref(dir + "/ref.pgm"), tar(192, 128);
  FFTCC2DCuda solver(7, 7, device);
  auto points = grid();
  std::vector<size_t> indices(points.size());
  std::iota(indices.begin(), indices.end(), 0);
  solver.setRefImage(ref);
  solver.prepareRef(points);
  int frame = 0;
  for (int shift : shifts)
  {
    tar.load(dir + "/tar" + std::to_string(shift) + ".pgm");
    solver.setTarImage(tar); // No explicit prepareTar: compute must refresh it.
    auto actual = points, expected = points;
    if (indexed) solver.compute(actual, indices); else solver.compute(actual);
    FFTCC2DCuda fresh(7, 7, device);
    fresh.setImages(ref, tar);
    fresh.prepareRef(points);
    if (indexed) fresh.compute(expected, indices); else fresh.compute(expected);
    compare("fft-target-" + std::to_string(shifts.front()) + "-indexed-"
            + std::to_string(indexed) + "-frame-" + std::to_string(frame++), actual, expected, false);
  }
}

void icgnTarget(const std::string& dir, int device, const std::vector<int>& shifts, bool both)
{
  Image2D ref(dir + "/ref.pgm"), tar(192, 128);
  ICGN2D1CudaMvp solver(7, 7, .001f, 10, device);
  solver.setRefImage(ref);
  solver.prepareRef(grid());
  int frame = 0;
  for (int shift : shifts)
  {
    tar.load(dir + "/tar" + std::to_string(shift) + ".pgm");
    if (both) solver.setImages(ref, tar); else solver.setTarImage(tar);
    auto actual = grid(static_cast<float>(shift)), expected = actual;
    solver.compute(actual);
    ICGN2D1CudaMvp fresh(7, 7, .001f, 10, device);
    fresh.setImages(ref, tar);
    fresh.compute(expected);
    compare("icgn-target-" + std::to_string(shifts.front()) + "-setImages-"
            + std::to_string(both) + "-frame-" + std::to_string(frame++), actual, expected, true);
  }
}

void reference(const std::string& dir, int device, bool indexed, bool replace_object)
{
  Image2D ref(dir + "/ref.pgm"), alternate(dir + "/tar5.pgm"), tar(dir + "/ref.pgm");
  FFTCC2DCuda fft(7, 7, device);
  ICGN2D1CudaMvp icgn(7, 7, .001f, 10, device);
  auto points = grid();
  std::vector<size_t> indices(points.size());
  std::iota(indices.begin(), indices.end(), 0);
  fft.setRefImage(ref); fft.prepareRef(points);
  icgn.setRefImage(ref); icgn.prepareRef(points);
  int frame = 0;
  for (int shift : {0, 5, 0})
  {
    // Test both mutation at the same address and a different reference object.
    // Keep the target fixed, so the correct displacement changes 0 -> -5 -> 0.
    if (!replace_object) ref.load(dir + "/tar" + std::to_string(shift) + ".pgm");
    const Image2D& current_ref = replace_object && shift == 5 ? alternate : ref;
    fft.setRefImage(current_ref); fft.setTarImage(tar);
    icgn.setRefImage(current_ref); icgn.setTarImage(tar);
    auto actual_fft = points, expected_fft = points;
    auto actual_icgn = grid(static_cast<float>(-shift)), expected_icgn = actual_icgn;
    if (indexed) fft.compute(actual_fft, indices); else fft.compute(actual_fft);
    icgn.compute(actual_icgn);
    FFTCC2DCuda fresh_fft(7, 7, device);
    fresh_fft.setImages(current_ref, tar); fresh_fft.prepareRef(points);
    if (indexed) fresh_fft.compute(expected_fft, indices); else fresh_fft.compute(expected_fft);
    ICGN2D1CudaMvp fresh_icgn(7, 7, .001f, 10, device);
    fresh_icgn.setImages(current_ref, tar); fresh_icgn.compute(expected_icgn);
    const std::string suffix = "-new-object-" + std::to_string(replace_object) + "-indexed-"
                               + std::to_string(indexed) + "-frame-" + std::to_string(frame++);
    compare("fft-reference" + suffix, actual_fft, expected_fft, false);
    compare("icgn-reference" + suffix, actual_icgn, expected_icgn, true);
  }
}

template <class Factory>
void cacheContract(const std::string& dir, const std::string& backend, Factory factory)
{
  Image2D ref(dir + "/ref.pgm"), tar(dir + "/ref.pgm");
  auto solver = factory();
  const auto points = grid();
  solver->setImages(ref, tar);
  auto missing = points;
  std::vector<size_t> identity(points.size());
  std::iota(identity.begin(), identity.end(), 0);
  rejects<std::logic_error>(backend + "-indexed-needs-layout", [&] { solver->compute(missing, identity); });
  solver->compute(missing); // preserve uncached API when no layout was prepared
  check(backend + "-unprepared-compute-stays-uncached", !solver->hasPreparedReference(ref));
  solver->prepareRef(points);
  check(backend + "-explicit-prepare-establishes-layout", solver->hasPreparedReference(ref));
  const std::vector<size_t> slots{points.size() - 1, 3, 12, points.size() - 1};
  for (int shift : {0, 5, 0})
  {
    ref.load(dir + "/tar" + std::to_string(shift) + ".pgm");
    solver->setRefImage(ref);
    check(backend + "-setter-invalidates-cache", !solver->hasPreparedReference(ref));
    std::vector<POI2D> actual;
    for (size_t slot : slots) actual.push_back(points[slot]);
    auto expected = actual;
    solver->compute(actual, slots);
    check(backend + "-indexed-rebuilds-cache", solver->hasPreparedReference(ref));
    auto fresh = factory();
    fresh->setImages(ref, tar); fresh->prepareRef(points);
    fresh->compute(expected, slots);
    compare(backend + "-reordered-duplicate-slots-" + std::to_string(shift), actual, expected, false);
  }
  auto selected = std::vector<POI2D>{points.back()};
  rejects<std::invalid_argument>(backend + "-index-count-mismatch", [&] { solver->compute(selected, {}); });
  rejects<std::out_of_range>(backend + "-out-of-range-index", [&] { solver->compute(selected, {points.size()}); });
  rejects<std::invalid_argument>(backend + "-wrong-coordinate-index", [&] { solver->compute(selected, {0}); });
  // Test validation while the cache is dirty as well as while it is ready.
  solver->setRefImage(ref);
  rejects<std::out_of_range>(backend + "-dirty-out-of-range-index", [&] { solver->compute(selected, {points.size()}); });
  check(backend + "-bad-index-did-not-rebuild", !solver->hasPreparedReference(ref));
}

// review-2026-10-03 S1-6: computeGathered() fills the recipient's per-POI
// precompute slots with the donor's records for another queue. The layout the
// recipient had prepared no longer describes those slots, so a later compute()
// on that layout must rebuild them instead of hitting the cache, and the
// recipient must not hand them on as a donor either.
void icgnGatherThenCompute(const std::string& dir, int device, ICGN2D1IterationKernelMode mode)
{
  const std::string tag = std::string("icgn-gather-")
                          + (mode == ICGN2D1IterationKernelMode::Legacy ? "legacy" : "warp");
  Image2D ref(dir + "/ref.pgm"), tar(dir + "/tar5.pgm");
  const auto layout = grid(5.f);
  auto make = [&] {
    auto solver = std::make_unique<ICGN2D1CudaMvp>(7, 7, .001f, 10, device);
    solver->setIterationKernelMode(mode);
    solver->setImages(ref, tar);
    return solver;
  };
  auto donor = make(), recipient = make();
  donor->prepareRef(layout);
  recipient->prepareRef(layout);
  check(tag + "-recipient-layout-prepared", recipient->hasPreparedReference(ref));

  // Slot i of the recipient receives the donor's record of POI n-1-i.
  const int count = 32;
  std::vector<int> donor_indices;
  std::vector<POI2D> subset;
  for (int i = 0; i < count; ++i)
  {
    donor_indices.push_back(static_cast<int>(layout.size()) - 1 - i);
    subset.push_back(layout[donor_indices.back()]);
  }
  auto gathered = subset, gathered_expected = subset;
  check(tag + "-fast-path-taken", recipient->computeGathered(*donor, donor_indices, gathered));
  make()->compute(gathered_expected);
  compare(tag + "-gathered-queue", gathered, gathered_expected, true);
  check(tag + "-drops-overwritten-layout", !recipient->hasPreparedReference(ref));

  // Slots 0..count-1 of the recipient no longer hold POIs 0..count-1.
  std::vector<int> head_indices(count);
  std::iota(head_indices.begin(), head_indices.end(), 0);
  std::vector<POI2D> head(layout.begin(), layout.begin() + count);
  auto head_expected = head;
  auto downstream = make();
  if (!downstream->computeGathered(*recipient, head_indices, head)) downstream->compute(head);
  make()->compute(head_expected);
  compare(tag + "-overwritten-recipient-as-donor", head, head_expected, true);

  auto actual = layout, expected = layout;
  recipient->compute(actual);
  make()->compute(expected);
  compare(tag + "-compute-after-gather", actual, expected, true);
  check(tag + "-compute-rebuilds-layout", recipient->hasPreparedReference(ref));
}
} // namespace

int main(int argc, char** argv)
{
  if (argc != 3) { std::cerr << "usage: cache_lifecycle_probe IMAGE_DIR DEVICE\n"; return 2; }
  try
  {
    omp_set_num_threads(2);
    const std::string dir = argv[1];
    const int device = std::stoi(argv[2]);
    for (const auto& shifts : {std::vector<int>{0, 5, 0}, std::vector<int>{5, 0, 5}})
      for (bool variant : {false, true})
      {
        fftTarget(dir, device, shifts, variant);
        icgnTarget(dir, device, shifts, variant);
      }
    for (bool indexed : {false, true})
      for (bool replace_object : {false, true})
        reference(dir, device, indexed, replace_object);
    cacheContract(dir, "gpu-fft", [device] { return std::make_unique<FFTCC2DCuda>(7, 7, device); });
    for (auto mode : {ICGN2D1IterationKernelMode::Warp, ICGN2D1IterationKernelMode::Legacy})
      icgnGatherThenCompute(dir, device, mode);
#ifdef OPENCORR_HAS_FFTW
    cacheContract(dir, "cpu-fft", [] { return std::make_unique<FFTCC2D>(7, 7, 2); });
    // A caller using the upstream DIC interface must still invalidate FFTCC.
    Image2D ref(dir + "/ref.pgm"), tar(dir + "/ref.pgm");
    FFTCC2D cpu(7, 7, 2);
    cpu.setImages(ref, tar); cpu.prepareRef(grid());
    DIC& base = cpu;
    base.setImages(ref, tar);
    check("cpu-base-setImages-invalidates", !cpu.hasPreparedReference(ref));
#endif
    std::cout << "cache lifecycle failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
  }
  catch (const std::exception& ex) { std::cerr << ex.what() << '\n'; return 2; }
  catch (const std::string& ex) { std::cerr << ex << '\n'; return 2; }
}
