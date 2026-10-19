/*
 * Copyright (c) 2026 Chonglang Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "oc_image.h"

using namespace opencorr;
using namespace std;

namespace
{

bool parseDepth(const string& text, ImageInputDepthMode& depth)
{
  if (text == "8")
  {
    depth = ImageInputDepthMode::Compat8;
    return true;
  }
  if (text == "12")
  {
    depth = ImageInputDepthMode::Depth12;
    return true;
  }
  if (text == "16")
  {
    depth = ImageInputDepthMode::Depth16;
    return true;
  }
  if (text == "auto")
  {
    depth = ImageInputDepthMode::Auto;
    return true;
  }
  return false;
}

bool parseAlignment(const string& text, ImageInputAlignment& alignment)
{
  if (text == "right")
  {
    alignment = ImageInputAlignment::Right;
    return true;
  }
  if (text == "left")
  {
    alignment = ImageInputAlignment::Left;
    return true;
  }
  if (text == "auto")
  {
    alignment = ImageInputAlignment::Auto;
    return true;
  }
  return false;
}

bool parseNormalize(const string& text, ImageInputNormalize& normalize)
{
  if (text == "none")
  {
    normalize = ImageInputNormalize::None;
    return true;
  }
  if (text == "to-8")
  {
    normalize = ImageInputNormalize::To8;
    return true;
  }
  if (text == "to-16")
  {
    normalize = ImageInputNormalize::To16;
    return true;
  }
  if (text == "unit")
  {
    normalize = ImageInputNormalize::Unit;
    return true;
  }
  return false;
}

void printUsage()
{
  cerr << "Usage: image_input_depth_probe [--legacy-load] "
       << "[--input-depth 8|12|16|auto] [--input-align right|left|auto] "
       << "[--input-normalize none|to-8|to-16|unit] <image>" << endl;
}

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    ImageLoadOptions options;
    bool legacy_load = false;
    string image_path;

    for (int i = 1; i < argc; ++i)
    {
      string arg = argv[i];
      if (arg == "--legacy-load")
      {
        legacy_load = true;
      }
      else if (arg == "--input-depth" && i + 1 < argc)
      {
        if (!parseDepth(argv[++i], options.depth))
        {
          cerr << "invalid --input-depth" << endl;
          return 1;
        }
      }
      else if (arg == "--input-align" && i + 1 < argc)
      {
        if (!parseAlignment(argv[++i], options.alignment))
        {
          cerr << "invalid --input-align" << endl;
          return 1;
        }
      }
      else if (arg == "--input-normalize" && i + 1 < argc)
      {
        if (!parseNormalize(argv[++i], options.normalize))
        {
          cerr << "invalid --input-normalize" << endl;
          return 1;
        }
      }
      else if (!arg.empty() && arg[0] == '-')
      {
        cerr << "unknown option: " << arg << endl;
        printUsage();
        return 1;
      }
      else
      {
        image_path = arg;
      }
    }

    if (image_path.empty())
    {
      printUsage();
      return 1;
    }

    Image2D image(1, 1);
    if (legacy_load)
    {
      image.load(image_path);
    }
    else
    {
      image.loadWithOptions(image_path, options);
    }

    cout << "requested_depth=" << imageInputDepthModeName(image.load_info.requested_depth) << "\n";
    cout << "effective_depth=" << imageInputDepthModeName(image.load_info.effective_depth) << "\n";
    cout << "requested_align=" << imageInputAlignmentName(image.load_info.requested_alignment) << "\n";
    cout << "effective_align=" << imageInputAlignmentName(image.load_info.effective_alignment) << "\n";
    cout << "normalize=" << imageInputNormalizeName(image.load_info.normalize) << "\n";
    cout << "source_depth=" << imageCvDepthName(image.load_info.source_depth) << "\n";
    cout << "output_depth=" << imageCvDepthName(image.load_info.output_depth) << "\n";
    cout << "source_range=" << image.load_info.source_min << "," << image.load_info.source_max << "\n";
    cout << "output_range=" << image.load_info.output_min << "," << image.load_info.output_max << "\n";
    cout << "values=";
    for (int x = 0; x < image.width; ++x)
    {
      if (x != 0)
      {
        cout << ",";
      }
      float value = image.eg_mat(0, x);
      if (fabs(value - round(value)) < 1e-6f)
      {
        cout << static_cast<long long>(llround(value));
      }
      else
      {
        cout << value;
      }
    }
    cout << "\n";
    return 0;
  }
  catch (const exception& ex)
  {
    cerr << "Error: " << ex.what() << endl;
    return 1;
  }
  catch (const string& ex)
  {
    cerr << "Error: " << ex << endl;
    return 1;
  }
}
