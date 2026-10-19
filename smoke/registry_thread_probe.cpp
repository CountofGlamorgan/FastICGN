/*
 * Copyright (c) 2026 Chonglang Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

// review-2026-10-03 S1-7: FFTCC2DCuda and ICGN2D1CudaMvp keep their device
// buffers in two process-wide registries. Solvers owned by different host
// threads used to corrupt them (double free on release, dangling references
// when the registry vector grew). Every worker thread here owns its solvers
// outright: only the registries and the read-only images are shared. Each
// round constructs, uses and destroys short-lived solvers next to long-lived
// ones, including a borrowed target and a gathered precompute, and every
// result must equal the single-threaded one bit for bit. Needs no data files.
#include "fftcc2d_cuda.h"
#include "icgn2d1_cuda_mvp.h"

#include <atomic>
#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <omp.h>

using namespace opencorr;

namespace {

const int kRadius = 7;
const int kRepairCount = 12;

std::vector<POI2D> grid()
{
  std::vector<POI2D> result;
  for (int y = 32; y < 96; y += 8)
    for (int x = 32; x < 160; x += 8)
      result.emplace_back(static_cast<float>(x), static_cast<float>(y));
  return result;
}

struct Round
{
  std::vector<POI2D> seeded, refined, repaired, kept_seeded, kept_refined;
};

bool sameBits(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }

bool sameQueue(const std::vector<POI2D>& a, const std::vector<POI2D>& b)
{
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
  {
    const auto& p = a[i].deformation;
    const auto& q = b[i].deformation;
    if (!sameBits(p.u, q.u) || !sameBits(p.ux, q.ux) || !sameBits(p.uy, q.uy)
        || !sameBits(p.v, q.v) || !sameBits(p.vx, q.vx) || !sameBits(p.vy, q.vy)
        || !sameBits(a[i].result.zncc, b[i].result.zncc))
      return false;
  }
  return true;
}

bool sameRound(const Round& a, const Round& b)
{
  return sameQueue(a.seeded, b.seeded) && sameQueue(a.refined, b.refined)
         && sameQueue(a.repaired, b.repaired) && sameQueue(a.kept_seeded, b.kept_seeded)
         && sameQueue(a.kept_refined, b.kept_refined);
}

// The solvers one thread keeps for all of its rounds.
struct Kept
{
  Kept(const Image2D& ref, int device)
      : fft(kRadius, kRadius, device), icgn(kRadius, kRadius, .001f, 10, device)
  {
    const auto layout = grid();
    fft.setRefImage(ref);
    fft.prepareRef(layout);
    icgn.setRefImage(ref);
    icgn.prepareRef(layout);
  }
  FFTCC2DCuda fft;
  ICGN2D1CudaMvp icgn;
};

Round oneRound(const Image2D& ref, const Image2D& tar, int device, Kept& kept)
{
  const auto layout = grid();
  Round round;

  // Short-lived solvers: each round inserts into and erases from both registries.
  FFTCC2DCuda fft(kRadius, kRadius, device);
  ICGN2D1CudaMvp icgn(kRadius, kRadius, .001f, 10, device);
  ICGN2D1CudaMvp repair(kRadius, kRadius, .001f, 10, device);
  fft.setImages(ref, tar);
  fft.prepareRef(layout);
  round.seeded = layout;
  fft.compute(round.seeded);
  icgn.setImages(ref, tar);
  icgn.prepareRef(layout);
  round.refined = round.seeded;
  icgn.compute(round.refined);

  // Borrowed target and gathered precompute: the walks over the ICGN registry.
  std::vector<int> donor_indices;
  for (int i = 0; i < kRepairCount; ++i)
  {
    donor_indices.push_back(static_cast<int>(layout.size()) - 1 - 3 * i);
    round.repaired.push_back(round.seeded[donor_indices.back()]);
  }
  repair.setRefImage(ref);
  repair.prepareRef();
  repair.adoptPreparedTar(icgn);
  if (!repair.computeGathered(icgn, donor_indices, round.repaired)) repair.compute(round.repaired);
  icgn.setTarImage(tar); // suspends the borrower, prepareTar refreshes it
  icgn.prepareTar();

  // Long-lived solvers: their registry entries stay put while others come and go.
  kept.fft.setTarImage(tar);
  round.kept_seeded = layout;
  kept.fft.compute(round.kept_seeded);
  kept.icgn.setTarImage(tar);
  round.kept_refined = round.kept_seeded;
  kept.icgn.compute(round.kept_refined);
  return round;
}

} // namespace

int main(int argc, char** argv)
{
  if (argc < 2 || argc > 4)
  {
    std::cerr << "usage: registry_thread_probe DEVICE [THREADS=4] [ROUNDS=30]\n";
    return 2;
  }
  try
  {
    omp_set_num_threads(2);
    const int device = std::stoi(argv[1]);
    const int threads = argc > 2 ? std::stoi(argv[2]) : 4;
    const int rounds = argc > 3 ? std::stoi(argv[3]) : 30;
    if (threads < 1 || rounds < 1) throw std::invalid_argument("THREADS and ROUNDS must be positive");

    Image2D ref(192, 128), tar(192, 128);
    unsigned state = 12345;
    for (int y = 0; y < 128; ++y)
      for (int x = 0; x < 192; ++x)
      {
        state = 1664525 * state + 1013904223;
        ref.eg_mat(y, x) = static_cast<float>(state >> 24);
      }
    for (int y = 0; y < 128; ++y)
      for (int x = 0; x < 192; ++x)
        tar.eg_mat(y, x) = (x >= 2 && y + 1 < 128) ? ref.eg_mat(y + 1, x - 2) : 127.f;

    Round expected;
    int truth = 0;
    {
      Kept kept(ref, device);
      expected = oneRound(ref, tar, device, kept);
      const Round repeat = oneRound(ref, tar, device, kept);
      for (const auto& poi : expected.refined)
        truth += std::fabs(poi.deformation.u - 2.f) < .01f && std::fabs(poi.deformation.v + 1.f) < .01f
                 && poi.result.zncc > .9f;
      const bool deterministic = sameRound(expected, repeat);
      const bool correct = truth == static_cast<int>(expected.refined.size());
      // Flushed: the threaded phase below used to end in a crash, not a FAIL line.
      std::cout << (deterministic ? "PASS " : "FAIL ") << "single-thread-rounds-repeat\n"
                << (correct ? "PASS " : "FAIL ") << "single-thread-translation-truth " << truth
                << "/" << expected.refined.size() << std::endl;
      if (!deterministic || !correct) return 1;
    }

    std::atomic<int> mismatches{0}, completed{0};
    std::mutex error_mutex;
    std::vector<std::string> errors;
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t)
      workers.emplace_back([&] {
        try
        {
          omp_set_num_threads(2);
          Kept kept(ref, device);
          for (int r = 0; r < rounds; ++r)
          {
            mismatches += !sameRound(oneRound(ref, tar, device, kept), expected);
            ++completed;
          }
        }
        catch (const std::exception& ex)
        {
          std::lock_guard<std::mutex> lock(error_mutex);
          errors.push_back(ex.what());
        }
        catch (const std::string& ex)
        {
          std::lock_guard<std::mutex> lock(error_mutex);
          errors.push_back(ex);
        }
      });
    for (auto& worker : workers) worker.join();

    for (const auto& error : errors) std::cout << "worker exception: " << error << '\n';
    const bool pass = errors.empty() && mismatches == 0 && completed == threads * rounds;
    std::cout << (pass ? "PASS " : "FAIL ") << "threads-own-their-solvers threads=" << threads
              << " rounds=" << completed << "/" << threads * rounds
              << " mismatched_rounds=" << mismatches << " exceptions=" << errors.size() << '\n';
    return pass ? 0 : 1;
  }
  catch (const std::exception& ex) { std::cerr << ex.what() << '\n'; return 2; }
  catch (const std::string& ex) { std::cerr << ex << '\n'; return 2; }
}
