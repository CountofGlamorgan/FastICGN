/*
 * Copyright (c) 2026 Chonglang Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "cli_image_probe.h"

#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>

namespace opencorr_cli_io {
namespace {

class HeaderReader
{
public:
  explicit HeaderReader(const std::filesystem::path& path) : in_(path, std::ios::binary) {}

  // Bytes actually read from `offset`, at most `count`.
  size_t readAt(uint64_t offset, unsigned char* out, size_t count)
  {
    if (!in_.is_open() || offset > static_cast<uint64_t>(INT64_MAX)) return 0;
    in_.clear();
    in_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!in_) return 0;
    in_.read(reinterpret_cast<char*>(out), static_cast<std::streamsize>(count));
    return static_cast<size_t>(in_.gcount());
  }

  bool readExactly(uint64_t offset, unsigned char* out, size_t count)
  {
    return readAt(offset, out, count) == count;
  }

private:
  std::ifstream in_;
};

uint64_t unsignedAt(const unsigned char* bytes, int width, bool little_endian)
{
  uint64_t value = 0;
  for (int i = 0; i < width; ++i)
  {
    const int index = little_endian ? width - 1 - i : i;
    value = (value << 8) | bytes[index];
  }
  return value;
}

// Classic TIFF (magic 42) and BigTIFF (magic 43): BitsPerSample (tag 258) of
// the first image file directory. The tag defaults to 1 when it is absent.
int tiffBits(HeaderReader& reader, const unsigned char* head, size_t head_size)
{
  const bool little = head[0] == 'I';
  if (head_size < 8) return 0;
  const uint64_t magic = unsignedAt(head + 2, 2, little);
  uint64_t directory = 0;
  int count_width = 2;
  int entry_bytes = 12;
  int value_offset = 8;
  int value_width = 4;
  if (magic == 42)
  {
    directory = unsignedAt(head + 4, 4, little);
  }
  else if (magic == 43)
  {
    if (head_size < 16 || unsignedAt(head + 4, 2, little) != 8 || unsignedAt(head + 6, 2, little) != 0)
      return 0;
    directory = unsignedAt(head + 8, 8, little);
    count_width = 8;
    entry_bytes = 20;
    value_offset = 12;
    value_width = 8;
  }
  else
  {
    return 0;
  }

  unsigned char raw[20];
  if (!reader.readExactly(directory, raw, static_cast<size_t>(count_width))) return 0;
  const uint64_t entries = unsignedAt(raw, count_width, little);
  if (entries == 0 || entries > 4096) return 0;
  for (uint64_t i = 0; i < entries; ++i)
  {
    const uint64_t offset = directory + static_cast<uint64_t>(count_width) + i * static_cast<uint64_t>(entry_bytes);
    if (!reader.readExactly(offset, raw, static_cast<size_t>(entry_bytes))) return 0;
    // Entries should be sorted by tag, but files that are not exist: scan them all.
    if (unsignedAt(raw, 2, little) != 258) continue;
    const uint64_t type = unsignedAt(raw + 2, 2, little);
    const uint64_t values = unsignedAt(raw + 4, value_width, little);
    // BitsPerSample is a SHORT (type 3). Some writers store a LONG (type 4),
    // which libtiff, and so OpenCV, reads as well.
    const int unit = type == 3 ? 2 : type == 4 ? 4 : 0;
    if (unit == 0 || values == 0) return 0;
    uint64_t bits = 0;
    if (values <= static_cast<uint64_t>(value_width / unit))
    {
      bits = unsignedAt(raw + value_offset, unit, little); // stored in the entry itself
    }
    else
    {
      unsigned char first[4];
      if (!reader.readExactly(unsignedAt(raw + value_offset, value_width, little), first,
                              static_cast<size_t>(unit)))
        return 0;
      bits = unsignedAt(first, unit, little);
    }
    return bits >= 1 && bits <= 64 ? static_cast<int>(bits) : 0;
  }
  return 1;
}

// JPEG: sample precision of the first start-of-frame segment.
int jpegBits(HeaderReader& reader)
{
  uint64_t offset = 2;
  unsigned char raw[4];
  for (int segment = 0; segment < 4096; ++segment)
  {
    if (!reader.readExactly(offset, raw, 2) || raw[0] != 0xFF) return 0;
    const unsigned char marker = raw[1];
    if (marker == 0xFF) { offset += 1; continue; }                    // fill byte
    if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD8)) { offset += 2; continue; }
    if (marker == 0xD9 || marker == 0xDA) return 0;                   // image data before any frame header
    if (!reader.readExactly(offset + 2, raw, 3)) return 0;
    const uint64_t length = unsignedAt(raw, 2, false);
    if (length < 2) return 0;
    const bool start_of_frame = marker >= 0xC0 && marker <= 0xCF
                                && marker != 0xC4 && marker != 0xC8 && marker != 0xCC;
    if (start_of_frame) return raw[2] >= 1 && raw[2] <= 16 ? raw[2] : 0;
    offset += 2 + length;
  }
  return 0;
}

// Netpbm P1..P6: the header is text; maxval above 255 means two bytes per sample.
int pnmBits(const unsigned char* head, size_t head_size)
{
  const char kind = static_cast<char>(head[1]);
  if (kind == '1' || kind == '4') return 1;
  size_t position = 2;
  long fields[3] = {0, 0, 0};
  for (int field = 0; field < 3; ++field)
  {
    while (position < head_size)
    {
      if (head[position] == '#')
      {
        while (position < head_size && head[position] != '\n') ++position;
      }
      else if (std::isspace(head[position]))
      {
        ++position;
      }
      else
      {
        break;
      }
    }
    if (position >= head_size || !std::isdigit(head[position])) return 0;
    long value = 0;
    while (position < head_size && std::isdigit(head[position]))
    {
      value = value * 10 + (head[position] - '0');
      if (value > 100000000L) return 0;
      ++position;
    }
    if (position >= head_size) return 0; // the number may continue past the bytes we hold
    fields[field] = value;
  }
  if (fields[2] < 1 || fields[2] > 65535) return 0;
  return fields[2] <= 255 ? 8 : 16;
}

} // namespace

StoredSampleDepth probeStoredSampleDepth(const std::filesystem::path& path)
{
  StoredSampleDepth result;
  try
  {
    HeaderReader reader(path);
    unsigned char head[512];
    const size_t head_size = reader.readAt(0, head, sizeof(head));
    if (head_size < 4) return result;

    static const unsigned char png_signature[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    if (head_size >= 26 && std::memcmp(head, png_signature, 8) == 0
        && std::memcmp(head + 12, "IHDR", 4) == 0)
    {
      result.container = "png";
      const int bits = head[24];
      if (bits == 1 || bits == 2 || bits == 4 || bits == 8 || bits == 16) result.bits = bits;
      return result;
    }
    if ((head[0] == 'I' && head[1] == 'I') || (head[0] == 'M' && head[1] == 'M'))
    {
      const int bits = tiffBits(reader, head, head_size);
      if (bits > 0)
      {
        result.container = "tiff";
        result.bits = bits;
      }
      return result;
    }
    if (head[0] == 0xFF && head[1] == 0xD8)
    {
      result.container = "jpeg";
      result.bits = jpegBits(reader);
      return result;
    }
    if (head[0] == 'P' && head[1] >= '1' && head[1] <= '6' && std::isspace(head[2]))
    {
      result.container = "pnm";
      result.bits = pnmBits(head, head_size);
      return result;
    }
    if (head[0] == 'B' && head[1] == 'M')
    {
      // OpenCV decodes every BMP flavour to 8-bit samples.
      result.container = "bmp";
      result.bits = 8;
      return result;
    }
  }
  catch (...)
  {
    result = StoredSampleDepth();
  }
  return result;
}

} // namespace opencorr_cli_io
