/*
 * Copyright (c) 2026 Chonglang Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <filesystem>

namespace opencorr_cli_io {

// Bits per sample an image file declares in its own header, read without
// decoding a single pixel (a few small reads, so it is cheap enough to run on
// every target frame). The CLI uses it to notice a frame whose stored depth
// differs from the reference when the pixels themselves are decoded as 8-bit
// and therefore no longer say how deep the file was.
struct StoredSampleDepth
{
  // 0: not determined (a container this probe does not parse, or a header that
  // is truncated, malformed or unreadable).
  int bits = 0;
  // "tiff", "png", "jpeg", "pnm", "bmp" or "unknown".
  const char* container = "unknown";

  bool known() const { return bits > 0; }
  // What matters to an 8-bit read: OpenCV hands back 8-bit samples for
  // anything up to 8 bits and reduces everything deeper.
  bool deeperThan8() const { return bits > 8; }
};

// Never throws: every failure is reported as bits == 0.
StoredSampleDepth probeStoredSampleDepth(const std::filesystem::path& path);

} // namespace opencorr_cli_io
