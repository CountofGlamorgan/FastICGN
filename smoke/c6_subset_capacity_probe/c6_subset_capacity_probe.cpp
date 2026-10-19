/*
 * Copyright (c) 2026 Chonglang Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "icgn2d1_cuda_mvp.h"
#include "oc_image.h"
#include "oc_poi.h"
#include "oc_point.h"

using namespace opencorr;

std::vector<POI2D> buildPoiQueue(const Image2D& image, int subset_radius, int grid_space)
{
  std::vector<POI2D> poi_queue;
  const int margin = subset_radius + 2;
  for (int y = margin; y < image.height - margin; y += grid_space)
  {
    for (int x = margin; x < image.width - margin; x += grid_space)
    {
      poi_queue.emplace_back(Point2D(static_cast<float>(x), static_cast<float>(y)));
    }
  }
  return poi_queue;
}

void runSubset(const Image2D& ref, const Image2D& tar, int subset_radius, int grid_space)
{
  std::vector<POI2D> poi_queue = buildPoiQueue(ref, subset_radius, grid_space);
  if (poi_queue.empty())
  {
    throw std::runtime_error("no POIs generated");
  }

  ICGN2D1CudaMvp icgn(subset_radius, subset_radius, 0.05f, 10, 0, false);
  icgn.setImages(ref, tar);
  icgn.prepareRef(poi_queue);
  icgn.prepareTar();
  icgn.compute(poi_queue);

  int valid = 0;
  for (const auto& poi : poi_queue)
  {
    if (poi.result.zncc >= 0.0f)
    {
      ++valid;
    }
  }
  std::cout << "subset_side=" << (2 * subset_radius + 1)
            << " poi_count=" << poi_queue.size()
            << " valid=" << valid << std::endl;
}

int main(int argc, char* argv[])
{
  if (argc != 3)
  {
    std::cerr << "Usage: c6_subset_capacity_probe <ref.pgm> <tar.pgm>" << std::endl;
    return 1;
  }

  Image2D ref(argv[1]);
  Image2D tar(argv[2]);

  runSubset(ref, tar, 7, 64);
  runSubset(ref, tar, 25, 64);
  return 0;
}
