/*
 * Copyright (c) 2026 Chonglang Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

// Compare the CPU-only wrapper with the native CPU solver on current images.
#include "icgn2d1_cuda_mvp.h"
#include "oc_icgn.h"
#include <omp.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace opencorr;

namespace {
int failures = 0;
void check(const std::string& name, bool ok)
{
  failures += !ok;
  std::cout << (ok ? "PASS " : "FAIL ") << name << '\n';
}

std::vector<POI2D> points(float seed)
{
  std::vector<POI2D> result;
  for (int y = 32; y < 96; y += 8)
    for (int x = 32; x < 160; x += 8)
    {
      result.emplace_back(x, y);
      result.back().deformation.u = seed;
    }
  return result;
}

bool equalFloat(float a, float b)
{
  return a == b || (std::isnan(a) && std::isnan(b));
}

void compare(const std::string& name, const std::vector<POI2D>& actual,
             Image2D& ref, Image2D& tar, float seed)
{
  ICGN2D1 native(7, 7, .001f, 10, omp_get_max_threads());
  native.setImages(ref, tar);
  native.prepare();
  auto expected = points(seed);
  native.compute(expected);
  int different = 0, valid = 0;
  for (size_t i = 0; i < expected.size(); ++i)
  {
    bool same = true;
    const auto& a = actual[i].deformation;
    const auto& b = expected[i].deformation;
    same &= equalFloat(a.u, b.u) && equalFloat(a.v, b.v)
            && equalFloat(a.ux, b.ux) && equalFloat(a.uy, b.uy)
            && equalFloat(a.vx, b.vx) && equalFloat(a.vy, b.vy);
    same &= equalFloat(actual[i].result.zncc, expected[i].result.zncc);
    different += !same;
    valid += std::isfinite(actual[i].result.zncc) && actual[i].result.zncc >= 0;
  }
  check(name + " differing=" + std::to_string(different), different == 0 && valid >= 122);
}

void run(const std::string& dir)
{
  omp_set_num_threads(1);
  Image2D ref(dir + "/ref.pgm"), tar(dir + "/tar0.pgm");
  ICGN2D1CudaMvp solver(7, 7, .001f, 10);
  solver.setRefImage(ref);
  solver.prepareRef(points(0)); // must prepare gradients before a target is set
  check("reference-ready-before-target", solver.hasPreparedReference(ref));
  for (int threads : {1, 4, 2})
  {
    omp_set_num_threads(threads); // resize an existing instance's worker pool
    for (int shift : {0, 5, 0})
    {
      tar.load(dir + "/tar" + std::to_string(shift) + ".pgm");
      solver.setTarImage(tar);
      check("target-setter-preserves-reference", solver.hasPreparedReference(ref));
      auto queue = points(shift);
      solver.compute(queue);
      compare("target-threads-" + std::to_string(threads) + "-shift-" + std::to_string(shift),
              queue, ref, tar, shift);
      check("reference-ready-after-compute", solver.hasPreparedReference(ref));
    }
  }
  for (int shift : {5, 0})
  {
    ref.load(dir + "/tar" + std::to_string(shift) + ".pgm");
    tar.load(dir + "/tar0.pgm");
    solver.setImages(ref, tar);
    check("reference-change-invalidates", !solver.hasPreparedReference(ref));
    auto queue = points(-shift);
    solver.compute(queue); // lazy preparation of both sides
    compare("reference-shift-" + std::to_string(shift), queue, ref, tar, -shift);
    check("reference-change-rebuilt", solver.hasPreparedReference(ref));
  }
  solver.setImages(ref, tar);
  solver.prepare();
  check("prepare-marks-reference-ready", solver.hasPreparedReference(ref));

  ICGN2D1CudaMvp repair(7, 7, .001f, 10);
  repair.setRefImage(ref); repair.prepareRef(points(0));
  for (int shift : {0, 5})
  {
    tar.load(dir + "/tar" + std::to_string(shift) + ".pgm");
    solver.setTarImage(tar); solver.prepareTar();
    repair.adoptPreparedTar(solver);
    const auto preparations = repair.profileStats().prepare_tar_calls;
    repair.adoptPreparedTar(solver);
    check("repeat-adopt-does-not-prepare",repair.profileStats().prepare_tar_calls == preparations);
    solver.prepareTar(); // same Image2D address, explicitly new preparation
    repair.adoptPreparedTar(solver);
    check("new-donor-generation-prepares",repair.profileStats().prepare_tar_calls == preparations + 1);
    auto queue = points(shift);
    std::vector<int> indices(queue.size());
    check("CPU-gather-declines-device-fast-path", !repair.computeGathered(solver, indices, queue));
    repair.compute(queue);
    compare("repair-target-" + std::to_string(shift), queue, ref, tar, shift);
  }
  const auto count = repair.profileStats().prepare_tar_calls;
  omp_set_num_threads(4);
  repair.adoptPreparedTar(solver);
  check("adopt-after-thread-pool-rebuild",repair.profileStats().prepare_tar_calls == count + 1);
  auto repaired = points(5);repair.compute(repaired);compare("adopt-resized-pool",repaired,ref,tar,5);
  solver.setTarImage(tar); // donor is now dirty
  bool rejected = false;
  try { repair.adoptPreparedTar(solver); }
  catch (const std::runtime_error&) { rejected = true; }
  check("reject-unprepared-donor", rejected);
}
}

int main(int argc, char** argv)
{
  if (argc < 2 || argc > 3) return 2;
  try { run(argv[1]); }
  catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
  catch (const std::string& e) { std::cerr << e << '\n'; return 2; }
  std::cout << "CPU lifecycle failures=" << failures << '\n';
  return failures ? 1 : 0;
}
