/*
 * Copyright (c) 2026 Chonglang Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdarg>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <omp.h>
#include <sstream>
#include <set>
#include <type_traits>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>

#include "icgn2d1_cuda_mvp.h"
#include "cli_image_probe.h"
#include "cli_output_integrity.h"
#ifdef OPENCORR_HAS_FFTW
#include "oc_fftcc.h"
#endif
#ifdef OPENCORR_GPU_MVP_USE_CUDA
#include "fftcc2d_cuda.h"
#include <cuda_runtime.h>
#endif
#include "oc_image.h"
#include "oc_poi.h"
#include "oc_point.h"
#include "opencorr_gpu_mvp_version.h"

using namespace opencorr;
using namespace std;
namespace fs = std::filesystem;
using opencorr_cli_io::Json;
// Insertion-ordered JSON for the lines this program prints on stdout, so the
// fields a reader looks for first (status, exit_code) come first.
using OrderedJson = nlohmann::ordered_json;
using opencorr_cli_io::Durability;
using opencorr_cli_io::OutputSession;

namespace
{

enum class OutputFormat
{
  Csv,
  RawBinary
};

// review-2026-09-02 section 4: selects the FFTCC initial-guess backend.  Gpu
// is the batched-cuFFT path (default in CUDA builds, no FFTW/GPL linkage);
// Cpu is the legacy FFTW path, available only in builds with
// OPENCORR_GPU_MVP_WITH_FFTW=ON.
enum class FftccBackend
{
  Cpu,
  Gpu
};

// Optional output-layer crop (write-side only; compute and QA always run on
// the full POI field).  A POI passes when its integer output coordinates fall
// inside the inclusive ROI rectangle and, with step > 1, both coordinates are
// multiples of step.  The default (no ROI, step 1) keeps every POI, which
// reproduces the legacy output byte for byte.
struct OutputFilter
{
  bool roi_enabled = false;
  int roi_x1 = 0;
  int roi_y1 = 0;
  int roi_x2 = 0;
  int roi_y2 = 0;
  int step = 1;

  bool active() const
  {
    return roi_enabled || step > 1;
  }

  bool keep(int x, int y) const
  {
    if (roi_enabled && (x < roi_x1 || x > roi_x2 || y < roi_y1 || y > roi_y2))
    {
      return false;
    }
    if (step > 1 && (x % step != 0 || y % step != 0))
    {
      return false;
    }
    return true;
  }
};

struct CliOptions
{
  int gpu_device = 0;
  // 0 = unset. --subset-size carries the odd subset SIZE; the solver still
  // works in radius, and radius = (size - 1) / 2 exactly as the positional
  // subset_radius argument. Only the spelling differs, never the numbers.
  int subset_size = 0;
  int max_iteration = 10;
  float conv_threshold = 0.001f;
  int num_threads = 1;
  bool dual_gpu = false;
  ICGN2D1IterationKernelMode kernel_mode = ICGN2D1IterationKernelMode::Warp;
#ifdef OPENCORR_GPU_MVP_USE_CUDA
  FftccBackend fftcc_backend = FftccBackend::Gpu;
#else
  FftccBackend fftcc_backend = FftccBackend::Cpu;
#endif
  bool skip_fftcc = true;
  bool poi_fallback = true;
  bool temporal_extrapolation = false;
  OutputFormat output_format = OutputFormat::Csv;
  bool diagnostics_enabled = false;
  bool preload_enabled = true;
  int decode_threads = 1;
  OutputFilter output_filter;
  ImageLoadOptions input_options;
  string diagnostics_dir;
  vector<int> line_ys;
  // Publication policy only: neither field changes numerical results, and
  // neither enters the resume contract.
  Durability durability = Durability::Rename;
  bool repair = false;
  // Presentation only: drops the per-frame progress lines from stderr.
  bool quiet = false;
};

// What the closing status line on stdout reports (docs/output-format.md,
// "Process output"). It is filled in while the run progresses, so a failure at
// any point still ends in one truthful line. Writers and the image preloader
// report from their own threads, hence the mutex.
class RunReport
{
public:
  void setMode(const char* mode)
  {
    lock_guard<mutex> lock(mutex_);
    mode_ = mode;
  }

  void setOutput(const fs::path& directory, const fs::path& file, const char* format)
  {
    lock_guard<mutex> lock(mutex_);
    std::error_code ignored;
    const fs::path absolute = fs::weakly_canonical(fs::absolute(directory.empty() ? fs::path(".") : directory, ignored), ignored);
    output_dir_ = absolute.empty() ? directory.string() : absolute.string();
    if (!file.empty()) output_path_ = (fs::path(output_dir_) / file.filename()).string();
    output_format_ = format;
  }

  void setFrames(size_t total, size_t processed, size_t skipped)
  {
    lock_guard<mutex> lock(mutex_);
    frames_total_ = total;
    frames_processed_ = processed;
    frames_skipped_ = skipped;
  }

  void frameProcessed()
  {
    lock_guard<mutex> lock(mutex_);
    ++frames_processed_;
  }

  void setTotals(OrderedJson totals)
  {
    lock_guard<mutex> lock(mutex_);
    totals_ = std::move(totals);
  }

  void addWarning(const string& text)
  {
    lock_guard<mutex> lock(mutex_);
    ++warning_count_;
    if (warnings_.size() < kStoredWarnings) warnings_.push_back(text);
  }

  // The first error is the cause; whatever follows it is fallout.
  void setError(const string& text)
  {
    lock_guard<mutex> lock(mutex_);
    if (!has_error_)
    {
      has_error_ = true;
      error_ = text;
    }
  }

  OrderedJson snapshot(int exit_code, double elapsed_sec) const
  {
    lock_guard<mutex> lock(mutex_);
    OrderedJson line = OrderedJson::object();
    line["schema"] = "fasticgn.run-status/1";
    line["status"] = exit_code == 0 ? "ok" : "error";
    line["exit_code"] = exit_code;
    line["error"] = exit_code == 0 ? OrderedJson(nullptr)
                    : OrderedJson(has_error_ ? error_ : string("failed without an error message; see stderr"));
    line["mode"] = mode_.empty() ? OrderedJson(nullptr) : OrderedJson(mode_);
    line["frames_total"] = frames_total_;
    line["frames_processed"] = frames_processed_;
    line["frames_skipped"] = frames_skipped_;
    line["output_dir"] = output_dir_.empty() ? OrderedJson(nullptr) : OrderedJson(output_dir_);
    if (!output_path_.empty()) line["output_path"] = output_path_;
    line["output_format"] = output_format_.empty() ? OrderedJson(nullptr) : OrderedJson(output_format_);
    line["elapsed_sec"] = std::round(elapsed_sec * 1000.0) / 1000.0;
    line["tool"] = "opencorr_cli_cuda_mvp";
    line["version"] = OPENCORR_GPU_MVP_VERSION;
    line["git_hash"] = OPENCORR_GPU_MVP_GIT_HASH;
    line["git_dirty"] = string(OPENCORR_GPU_MVP_GIT_DIRTY) == "1" ? OrderedJson(true)
                        : string(OPENCORR_GPU_MVP_GIT_DIRTY) == "0" ? OrderedJson(false) : OrderedJson(nullptr);
    line["build_date"] = OPENCORR_GPU_MVP_BUILD_DATE;
#ifdef OPENCORR_GPU_MVP_USE_CUDA
    line["backend"] = "cuda";
#else
    line["backend"] = "cpu";
#endif
    if (!totals_.is_null()) line["totals"] = totals_;
    line["warning_count"] = warning_count_;
    line["warnings"] = warnings_;
    return line;
  }

private:
  static constexpr size_t kStoredWarnings = 20;
  mutable mutex mutex_;
  string mode_;
  string output_dir_;
  string output_path_;
  string output_format_;
  size_t frames_total_ = 0;
  size_t frames_processed_ = 0;
  size_t frames_skipped_ = 0;
  size_t warning_count_ = 0;
  vector<string> warnings_;
  bool has_error_ = false;
  string error_;
  OrderedJson totals_;
};

RunReport g_report;

// `ErrorLine() << a << b;` prints exactly what `cerr << "Error: " << a << b << endl;`
// printed, and keeps the text for the status line. WarningLine does the same
// for "Warning: ". Both write when the statement ends.
class DiagnosticLine
{
public:
  DiagnosticLine(const DiagnosticLine&) = delete;
  DiagnosticLine& operator=(const DiagnosticLine&) = delete;

  template <typename T>
  DiagnosticLine& operator<<(const T& value)
  {
    text_ << value;
    return *this;
  }

protected:
  explicit DiagnosticLine(bool error) : error_(error) {}

  ~DiagnosticLine()
  {
    const string text = text_.str();
    cerr << (error_ ? "Error: " : "Warning: ") << text << endl;
    if (error_) g_report.setError(text);
    else g_report.addWarning(text);
  }

private:
  bool error_;
  ostringstream text_;
};

struct ErrorLine : DiagnosticLine
{
  ErrorLine() : DiagnosticLine(true) {}
};

struct WarningLine : DiagnosticLine
{
  WarningLine() : DiagnosticLine(false) {}
};

struct MultiFrameEntry
{
  string tar_path;
  string output_name;
};

struct PreloadedImage
{
  string tar_path;
  unique_ptr<Image2D> image;
  double load_time = 0.0;
  bool success = false;
  string error;
};

struct SerializedCsv
{
  vector<char> buffer;
  size_t length = 0;
};

using PoiBuffer = shared_ptr<vector<POI2D>>;
using PublishedPoiBuffer = shared_ptr<const vector<POI2D>>;

class PoiBufferPool
{
public:
  PoiBuffer acquire()
  {
    shared_ptr<State> state = state_;
    unique_ptr<vector<POI2D>> buffer;
    {
      lock_guard<mutex> lock(state->mutex_);
      if (!state->free_.empty())
      {
        buffer = std::move(state->free_.back());
        state->free_.pop_back();
      }
    }
    if (!buffer)
    {
      buffer = make_unique<vector<POI2D>>();
    }

    return PoiBuffer(buffer.release(), [state](vector<POI2D>* returned) noexcept
    {
      unique_ptr<vector<POI2D>> buffer_to_return(returned);
      try
      {
        lock_guard<mutex> lock(state->mutex_);
        state->free_.push_back(std::move(buffer_to_return));
      }
      catch (...)
      {
      }
    });
  }

private:
  struct State
  {
    mutex mutex_;
    vector<unique_ptr<vector<POI2D>>> free_;
  };

  shared_ptr<State> state_ = make_shared<State>();
};

struct RawBinaryWriteStats
{
  size_t rows = 0;
  size_t bytes = 0;
  double format_time = 0.0;
  double write_time = 0.0;
};

struct FrameQaRecord
{
  bool has_bad_ratio = false;
  float bad_ratio = 0.0f;
  bool has_baseline_bad_ratio = false;
  float baseline_bad_ratio = 0.0f;
  bool has_fallback_threshold = false;
  float fallback_threshold = 0.0f;
  bool has_fallback_used = false;
  bool fallback_used = false;
  bool has_skip_fftcc_used = false;
  bool skip_fftcc_used = false;
  bool has_valid_ratio = false;
  float valid_ratio = 0.0f;
  bool has_iteration_mean = false;
  float iteration_mean = 0.0f;
  bool has_iteration_max = false;
  int iteration_max = 0;
  bool has_poi_fallback_count = false;
  int poi_fallback_count = 0;
  bool has_poi_fallback_recovered = false;
  int poi_fallback_recovered = 0;
  bool has_temporal_seed_used = false;
  bool temporal_seed_used = false;
  bool has_temporal_seed_count = false;
  int temporal_seed_count = 0;
};

struct OutputFrameRecord
{
  string tar_path;
  string output_name;
  string file_name;
  size_t rows = 0;
  size_t bytes = 0;
  double format_time = 0.0;
  double write_time = 0.0;
  FrameQaRecord qa;
};

struct DiagnosticsFrameRecord
{
  string tar_path;
  string output_name;
  string file_name;
  size_t rows = 0;
};

struct PoiFallbackTimingStats
{
  long long set_images_calls = 0;
  long long prepare_tar_calls = 0;
  long long adopt_prepared_tar_calls = 0;
  long long compute_calls = 0;
  double set_images_sec = 0.0;
  double prepare_tar_sec = 0.0;
  double adopt_prepared_tar_sec = 0.0;
  double compute_sec = 0.0;
};

struct FrameProcessStats
{
  double seed_sec = 0.0;
  double scan_sec = 0.0;
  bool frame_fallback_used = false;
  bool poi_fallback_used = false;
  int poi_fallback_count = 0;
  int poi_fallback_recovered = 0;
  bool temporal_seed_used = false;
  int temporal_seed_count = 0;
  float bad_ratio = 0.0f;
  float pre_repair_bad_ratio = 0.0f;
  float frame_fallback_trigger_bad_ratio = 0.0f;
  PoiFallbackTimingStats poi_fallback_timing;
};

bool writeCsv(const string& output_path, const vector<POI2D>& poi_queue,
              const OutputFilter& filter);
bool writeCsv(const string& output_path, const SerializedCsv& csv_data, OutputSession* session = nullptr);
bool writeDiagnosticsCsv(const fs::path& path, const vector<POI2D>& queue, OutputSession* session = nullptr);
bool serializeCsv(const vector<POI2D>& poi_queue, SerializedCsv& csv_data,
                  const OutputFilter& filter);
// Serialization only; the caller owns and reuses the destination buffer. Fills
// stats.rows/bytes/format_time, leaves stats.write_time to the publisher.
void formatRawBinary(const vector<POI2D>& poi_queue, const OutputFilter& filter,
                     vector<char>& buffer, RawBinaryWriteStats& stats);

constexpr size_t kCsvHeaderLength = sizeof("x,y,u,v,zncc,uy,vy,ux,vx\n") - 1;
constexpr size_t kCsvBytesPerRow = 176;
constexpr size_t kRawBinaryRecordBytes = sizeof(int32_t) * 2 + sizeof(float) * 7;
constexpr float kSkipFftccZnccThreshold = 0.8f;
constexpr float kSkipFftccBadRatioMargin = 0.03f;
constexpr float kTemporalSeedMinStepPx = 0.05f;

bool samePoiLocation(const POI2D& a, const POI2D& b)
{
  return a.x == b.x && a.y == b.y;
}

bool hasFiniteUv(const POI2D& poi)
{
  return isfinite(poi.deformation.u) && isfinite(poi.deformation.v);
}

bool isBadPoiForFallback(const POI2D& poi, float zncc_threshold)
{
  return poi.result.zncc != -3.f
         && (!isfinite(poi.result.zncc) || poi.result.zncc < zncc_threshold);
}

bool isEligibleTemporalSeedHistory(const POI2D& poi)
{
  return hasFiniteUv(poi)
         && isfinite(poi.result.zncc)
         && poi.result.zncc >= kSkipFftccZnccThreshold;
}

bool shouldReplacePoiWithFallback(const POI2D& current, const POI2D& fallback)
{
  if (!isfinite(fallback.result.zncc) || fallback.result.zncc == -3.f)
  {
    return false;
  }
  if (!isfinite(current.result.zncc))
  {
    return true;
  }
  return fallback.result.zncc >= current.result.zncc;
}

// Whole-frame fallback merge (manifest frame_fallback_policy "per-poi-merge-v1",
// review-2026-10-03 S1-1): true when the from-scratch result replaces the skip
// attempt's. A good skip result is never replaced; a bad one only under the
// POI-level repair rule; a -3 one only by a good from-scratch result. That last
// case differs from the POI-level path, where -3 is terminal: after a jump
// between frames the seed is far from the truth and the iterate leaves the
// image although the subset is inside it at the true displacement. No case adds
// a bad POI, so the merged frame has at most the skip attempt's bad count.
bool shouldTakeFrameFallbackResult(const POI2D& skip, const POI2D& scratch)
{
  if (skip.result.zncc == -3.f)
  {
    return isfinite(scratch.result.zncc) && scratch.result.zncc != -3.f
           && scratch.result.zncc >= kSkipFftccZnccThreshold;
  }
  if (isBadPoiForFallback(skip, kSkipFftccZnccThreshold))
  {
    return shouldReplacePoiWithFallback(skip, scratch);
  }
  return false;
}

bool isPoiLevelFallbackCandidate(const POI2D& current, const POI2D* previous,
                                 bool current_is_bad)
{
  if (!current_is_bad)
  {
    return false;
  }
  if (!isfinite(current.result.zncc) || current.result.zncc < 0.0f)
  {
    return true;
  }
  if (previous == nullptr || !samePoiLocation(current, *previous)
      || !isfinite(previous->result.zncc))
  {
    return true;
  }
  return previous->result.zncc >= kSkipFftccZnccThreshold;
}

void seedPoiQueueFromPrevious(const vector<POI2D>& poi_template,
                              const vector<POI2D>& prev_results,
                              const vector<POI2D>* prev_prev_results,
                              bool enable_temporal_extrapolation,
                              float max_temporal_delta,
                              vector<POI2D>& poi_queue,
                              int* temporal_seed_count)
{
  const bool template_already_sized = poi_queue.size() == poi_template.size();
  if (!template_already_sized)
  {
    poi_queue = poi_template;
  }
  int extrapolated_count = 0;
  size_t copy_count = min(poi_queue.size(), prev_results.size());
  bool has_second_history = prev_prev_results != nullptr
                            && prev_prev_results->size() == prev_results.size();
#pragma omp parallel for reduction(+ : extrapolated_count) schedule(static)
  for (size_t i = 0; i < poi_queue.size(); ++i)
  {
    if (template_already_sized)
    {
      poi_queue[i] = poi_template[i];
    }
    if (i >= copy_count)
    {
      continue;
    }
    if (!isfinite(prev_results[i].result.zncc) || prev_results[i].result.zncc < 0.f)
    {
      continue;
    }

    for (size_t p = 0; p < 12; ++p)
    {
      poi_queue[i].deformation.p[p] = prev_results[i].deformation.p[p];
    }
    if (enable_temporal_extrapolation && has_second_history
        && samePoiLocation(prev_results[i], (*prev_prev_results)[i])
        && samePoiLocation(prev_results[i], poi_template[i])
        && isEligibleTemporalSeedHistory(prev_results[i])
        && isEligibleTemporalSeedHistory((*prev_prev_results)[i]))
    {
      float du = prev_results[i].deformation.u - (*prev_prev_results)[i].deformation.u;
      float dv = prev_results[i].deformation.v - (*prev_prev_results)[i].deformation.v;
      float predicted_u = prev_results[i].deformation.u + du;
      float predicted_v = prev_results[i].deformation.v + dv;
      float step_sq = du * du + dv * dv;
      if (isfinite(predicted_u) && isfinite(predicted_v)
          && step_sq >= kTemporalSeedMinStepPx * kTemporalSeedMinStepPx
          && fabs(du) <= max_temporal_delta && fabs(dv) <= max_temporal_delta)
      {
        poi_queue[i].deformation.u = predicted_u;
        poi_queue[i].deformation.v = predicted_v;
        ++extrapolated_count;
      }
    }
    fill(begin(poi_queue[i].result.r), end(poi_queue[i].result.r), 0.f);
    fill(begin(poi_queue[i].strain.e), end(poi_queue[i].strain.e), 0.f);
  }
  if (temporal_seed_count != nullptr)
  {
    *temporal_seed_count = extrapolated_count;
  }
}

float computeBadZnccRatioFromCount(size_t bad_count, const vector<POI2D>& poi_queue)
{
  if (poi_queue.empty())
  {
    return 0.f;
  }

  return static_cast<float>(bad_count) / static_cast<float>(poi_queue.size());
}

float computeBadZnccRatio(const vector<POI2D>& poi_queue, float zncc_threshold)
{
  size_t bad_count = 0;
  if (poi_queue.empty())
  {
    return 0.f;
  }

  for (const auto& poi : poi_queue)
  {
    if (isBadPoiForFallback(poi, zncc_threshold))
    {
      ++bad_count;
    }
  }
  return computeBadZnccRatioFromCount(bad_count, poi_queue);
}

FrameQaRecord computeFrameQa(const vector<POI2D>& poi_queue, float bad_ratio,
                             bool baseline_bad_ratio_ready, float baseline_bad_ratio,
                             bool fallback_threshold_ready, float fallback_threshold,
                             bool fallback_used, bool skip_fftcc_used,
                             const FrameProcessStats& process_stats)
{
  FrameQaRecord qa;
  qa.has_bad_ratio = true;
  qa.bad_ratio = bad_ratio;
  qa.has_baseline_bad_ratio = baseline_bad_ratio_ready;
  qa.baseline_bad_ratio = baseline_bad_ratio;
  qa.has_fallback_threshold = fallback_threshold_ready;
  qa.fallback_threshold = fallback_threshold;
  qa.has_fallback_used = true;
  qa.fallback_used = fallback_used;
  qa.has_skip_fftcc_used = true;
  qa.skip_fftcc_used = skip_fftcc_used;
  qa.has_poi_fallback_count = true;
  qa.poi_fallback_count = process_stats.poi_fallback_count;
  qa.has_poi_fallback_recovered = true;
  qa.poi_fallback_recovered = process_stats.poi_fallback_recovered;
  qa.has_temporal_seed_used = true;
  qa.temporal_seed_used = process_stats.temporal_seed_used;
  qa.has_temporal_seed_count = true;
  qa.temporal_seed_count = process_stats.temporal_seed_count;

  if (poi_queue.empty())
  {
    qa.has_valid_ratio = true;
    qa.valid_ratio = 0.0f;
    qa.has_iteration_mean = true;
    qa.iteration_mean = 0.0f;
    qa.has_iteration_max = true;
    qa.iteration_max = 0;
    return qa;
  }

  long long valid_count = 0;
  long long finite_iteration_count = 0;
  long long iteration_sum = 0;
  int iteration_max = 0;
  const long long qa_n = static_cast<long long>(poi_queue.size());
#pragma omp parallel for reduction(+ : valid_count, finite_iteration_count, iteration_sum) reduction(max : iteration_max) schedule(static)
  for (long long qi = 0; qi < qa_n; ++qi)
  {
    const POI2D& poi = poi_queue[static_cast<size_t>(qi)];
    if (isfinite(poi.result.zncc) && poi.result.zncc >= 0.0f)
    {
      ++valid_count;
    }
    if (isfinite(poi.result.iteration))
    {
      ++finite_iteration_count;
      iteration_sum += static_cast<long long>(poi.result.iteration);
      iteration_max = max(iteration_max,
                          static_cast<int>(poi.result.iteration + 0.5f));
    }
  }

  qa.has_valid_ratio = true;
  qa.valid_ratio = static_cast<float>(valid_count) / static_cast<float>(poi_queue.size());
  qa.has_iteration_mean = true;
  qa.iteration_mean = finite_iteration_count == 0
                        ? 0.0f
                        : static_cast<float>(iteration_sum
                                             / static_cast<double>(finite_iteration_count));
  qa.has_iteration_max = true;
  qa.iteration_max = iteration_max;
  return qa;
}

float diagnosticStatusFromZncc(float zncc)
{
  if (!isfinite(zncc))
  {
    return -5.0f;
  }
  return zncc >= 0.0f ? 1.0f : zncc;
}

size_t estimateCsvBytes(size_t poi_count)
{
  return kCsvHeaderLength + poi_count * kCsvBytesPerRow;
}

const char* outputFormatName(OutputFormat format)
{
  return format == OutputFormat::Csv ? "csv" : "raw";
}

const char* iterationKernelModeName(ICGN2D1IterationKernelMode mode)
{
#ifdef OPENCORR_GPU_MVP_USE_CUDA
  return mode == ICGN2D1IterationKernelMode::Warp ? "warp" : "legacy";
#else
  (void)mode;
  return "cpu (ICGN2D1)";
#endif
}

const char* icgnExecutionLabel()
{
#ifdef OPENCORR_GPU_MVP_USE_CUDA
  return "GPU ICGN";
#else
  return "CPU ICGN";
#endif
}

const char* fftccBackendName(FftccBackend backend)
{
  return backend == FftccBackend::Gpu ? "gpu" : "cpu";
}

fs::path makeOutputPath(const string& output_dir, const string& output_name, OutputFormat format)
{
  fs::path output_path = fs::path(output_dir) / output_name;
  if (format == OutputFormat::RawBinary && output_path.extension() == ".csv")
  {
    output_path.replace_extension(".bin");
  }
  return output_path;
}

fs::path makeSingleOutputPath(const string& output_path, OutputFormat format)
{
  fs::path path(output_path);
  if (format == OutputFormat::RawBinary && path.extension() == ".csv")
  {
    path.replace_extension(".bin");
  }
  return path;
}

void requireUtf8(const char* role, const string& value)
{
  // nlohmann's strict dump() rejects invalid UTF-8. Fail at startup with a
  // readable message instead of deep inside contract/manifest serialization.
  try
  {
    (void)Json(value).dump();
  }
  catch (const exception&)
  {
    throw runtime_error(string("non-UTF-8 ") + role
                        + " cannot be recorded in JSON metadata; use UTF-8 paths: " + value);
  }
}

string jsonEscape(const string& input)
{
  requireUtf8("text", input);
  const string encoded = Json(input).dump();
  return encoded.substr(1, encoded.size() - 2);
}

void writeJsonNullableFloat(ostream& out, bool has_value, float value)
{
  if (has_value)
  {
    out << value;
  }
  else
  {
    out << "null";
  }
}

void writeJsonNullableInt(ostream& out, bool has_value, int value)
{
  if (has_value)
  {
    out << value;
  }
  else
  {
    out << "null";
  }
}

void writeJsonNullableBool(ostream& out, bool has_value, bool value)
{
  if (has_value)
  {
    out << (value ? "true" : "false");
  }
  else
  {
    out << "null";
  }
}

template <typename T>
void readOptionalQa(const Json& item, const char* key, bool& present, T& value)
{
  present = item.contains(key) && !item.at(key).is_null();
  if (!present) return;
  const auto& field = item.at(key);
  if constexpr (std::is_same_v<T, bool>)
  {
    if (!field.is_boolean()) throw runtime_error(string("invalid QA boolean: ") + key);
    value = field.get<bool>();
  }
  else
  {
    if (!field.is_number() || (std::is_integral_v<T> && !field.is_number_integer()))
      throw runtime_error(string("invalid QA number: ") + key);
    const double number = field.get<double>();
    if (!std::isfinite(number) || number < std::numeric_limits<T>::lowest()
        || number > std::numeric_limits<T>::max())
      throw runtime_error(string("out-of-range QA number: ") + key);
    value = static_cast<T>(number);
  }
}

bool readRawManifestFrames(const fs::path& path, vector<OutputFrameRecord>& frames, string& error)
{
  frames.clear();
  if (!fs::exists(fs::symlink_status(path))) return true;
  try
  {
    const Json document = opencorr_cli_io::readJson(path);
    if (!document.is_object() || document.at("format") != "opencorr-gpu-raw-v1"
        || document.at("record_bytes") != kRawBinaryRecordBytes
        || document.at("endianness") != "little" || !document.at("frames").is_array())
      throw runtime_error("invalid raw manifest schema");
    const std::array<const char*, 9> field_names{{"x", "y", "u", "v", "ux", "uy", "vx", "vy", "zncc"}};
    const auto& fields = document.at("fields");
    if (!fields.is_array() || fields.size() != field_names.size()) throw runtime_error("invalid raw field layout");
    for (size_t i = 0; i < field_names.size(); ++i)
      if (fields[i].at("name") != field_names[i] || fields[i].at("offset") != i * 4
          || fields[i].at("dtype") != (i < 2 ? "int32" : "float32"))
        throw runtime_error("invalid raw field layout");
    if (document.at("csv_field_order") != Json({"x","y","u","v","zncc","uy","vy","ux","vx"}))
      throw runtime_error("invalid CSV field order");
    std::set<string> names;
    for (const auto& item : document.at("frames"))
    {
      OutputFrameRecord frame;
      frame.tar_path = item.at("tar_path").get<string>();
      frame.output_name = item.at("output_name").get<string>();
      frame.file_name = item.at("file").get<string>();
      if (!names.insert(frame.file_name).second) throw runtime_error("duplicate manifest frame file");
      if (!item.at("rows").is_number_unsigned() || !item.at("bytes").is_number_unsigned())
        throw runtime_error("invalid manifest frame size");
      frame.rows = item.at("rows").get<size_t>();
      frame.bytes = item.at("bytes").get<size_t>();
      if (frame.rows > std::numeric_limits<size_t>::max() / kRawBinaryRecordBytes
          || frame.bytes != frame.rows * kRawBinaryRecordBytes)
        throw runtime_error("raw manifest row/byte count mismatch");
      frame.format_time = item.value("format_sec", 0.0);
      frame.write_time = item.value("write_sec", 0.0);
      if (!isfinite(frame.format_time) || !isfinite(frame.write_time)
          || frame.format_time < 0 || frame.write_time < 0)
        throw runtime_error("invalid raw manifest timing");
#define QA_FIELD(name) readOptionalQa(item, #name, frame.qa.has_##name, frame.qa.name)
      QA_FIELD(bad_ratio); QA_FIELD(baseline_bad_ratio); QA_FIELD(fallback_threshold);
      QA_FIELD(fallback_used); QA_FIELD(skip_fftcc_used); QA_FIELD(valid_ratio);
      QA_FIELD(iteration_mean); QA_FIELD(iteration_max); QA_FIELD(poi_fallback_count);
      QA_FIELD(poi_fallback_recovered); QA_FIELD(temporal_seed_used); QA_FIELD(temporal_seed_count);
#undef QA_FIELD
      frames.push_back(std::move(frame));
    }
    return true;
  }
  catch (const exception& ex) { error = ex.what(); return false; }
}

bool publishBytes(const fs::path& path, const char* bytes, size_t count, OutputSession* session = nullptr)
{
  try
  {
    if (session) session->publish(path, bytes, count);
    else opencorr_cli_io::atomicWrite(path, bytes, count);
    return true;
  }
  catch (const exception& ex) { ErrorLine() << "output publication failed: " << ex.what(); return false; }
}

bool publishMetadata(const fs::path& path, const string& bytes)
{
  try {
    const auto validated = opencorr_cli_io::parseJson(bytes, path.string());
    if (!validated.is_object()) throw runtime_error("metadata root must be an object");
  }
  catch (const exception& ex) { ErrorLine() << "invalid generated metadata: " << ex.what(); return false; }
  return publishBytes(path, bytes.data(), bytes.size());
}

void printPrepareTarStats(const ICGN2D1CudaMvpProfileStats& stats)
{
  cerr << "prepareTar calls: " << stats.prepare_tar_calls << endl;
  cerr << "Accumulated prepareTar total: " << stats.prepare_tar_total_sec << " sec" << endl;
  cerr << "Accumulated prepareTar cudaFree: " << stats.prepare_tar_free_sec << " sec" << endl;
  cerr << "Accumulated prepareTar cudaMalloc: " << stats.prepare_tar_malloc_sec << " sec" << endl;
  cerr << "Accumulated prepareTar host row-major staging: "
       << stats.prepare_tar_host_staging_sec << " sec" << endl;
  cerr << "Accumulated prepareTar H2D copy: " << stats.prepare_tar_h2d_sec << " sec" << endl;
  cerr << "Accumulated prepareTar B-spline kernel: " << stats.prepare_tar_kernel_sec << " sec" << endl;
  cerr << "ICGN compute calls: " << stats.compute_calls << endl;
  cerr << "ICGN compute device work items: " << stats.compute_device_work_items << endl;
  cerr << "ICGN layout cache hits: " << stats.compute_layout_cache_hits << endl;
  cerr << "ICGN layout cache misses: " << stats.compute_layout_cache_misses << endl;
  cerr << "Accumulated ICGN compute profile total: " << stats.compute_total_sec << " sec" << endl;
  cerr << "Accumulated ICGN stream management: "
       << stats.compute_stream_management_sec << " sec" << endl;
  cerr << "Accumulated ICGN layout/precompute guard: "
       << stats.compute_layout_precompute_guard_sec << " sec" << endl;
  cerr << "Accumulated ICGN POI input staging: "
       << stats.compute_poi_in_staging_sec << " sec" << endl;
  cerr << "Accumulated ICGN POI output init: "
       << stats.compute_poi_out_init_sec << " sec" << endl;
  cerr << "Accumulated ICGN H2D copy: " << stats.compute_h2d_sec << " sec" << endl;
  cerr << "Accumulated ICGN kernel enqueue: "
       << stats.compute_kernel_enqueue_sec << " sec" << endl;
  cerr << "Accumulated ICGN kernel sync: "
       << stats.compute_kernel_sync_sec << " sec" << endl;
  cerr << "Accumulated ICGN D2H copy: " << stats.compute_d2h_sec << " sec" << endl;
  cerr << "Accumulated ICGN host scatter: "
       << stats.compute_host_scatter_sec << " sec" << endl;
}

void accumulatePoiFallbackTiming(PoiFallbackTimingStats& total,
                                 const PoiFallbackTimingStats& frame)
{
  total.set_images_calls += frame.set_images_calls;
  total.prepare_tar_calls += frame.prepare_tar_calls;
  total.adopt_prepared_tar_calls += frame.adopt_prepared_tar_calls;
  total.compute_calls += frame.compute_calls;
  total.set_images_sec += frame.set_images_sec;
  total.prepare_tar_sec += frame.prepare_tar_sec;
  total.adopt_prepared_tar_sec += frame.adopt_prepared_tar_sec;
  total.compute_sec += frame.compute_sec;
}

void printPoiFallbackTiming(const PoiFallbackTimingStats& stats)
{
  cerr << "POI fallback timing: setImages_calls=" << stats.set_images_calls
       << ", setImages=" << stats.set_images_sec << " sec"
       << ", prepareTar_calls=" << stats.prepare_tar_calls
       << ", prepareTar=" << stats.prepare_tar_sec << " sec"
       << ", adoptPreparedTar_calls=" << stats.adopt_prepared_tar_calls
       << ", adoptPreparedTar=" << stats.adopt_prepared_tar_sec << " sec"
       << ", compute_calls=" << stats.compute_calls
       << ", compute=" << stats.compute_sec << " sec" << endl;
}

bool appendFormat(vector<char>& buffer, size_t& length, const char* format, ...)
{
  while (true)
  {
    if (length >= buffer.size())
    {
      size_t next_size = buffer.empty() ? 1024 : buffer.size() * 2;
      buffer.resize(next_size);
    }

    va_list args;
    va_start(args, format);
    int written = vsnprintf(buffer.data() + length, buffer.size() - length, format, args);
    va_end(args);

    if (written < 0)
    {
      return false;
    }

    size_t required = static_cast<size_t>(written);
    if (length + required < buffer.size())
    {
      length += required;
      return true;
    }

    buffer.resize((length + required + 1) * 2);
  }
}

void ensureCsvCapacity(vector<char>& buffer, size_t length, size_t additional)
{
  if (length + additional < buffer.size())
  {
    return;
  }

  size_t next_size = buffer.empty() ? 1024 : buffer.size();
  while (length + additional >= next_size)
  {
    next_size *= 2;
  }
  buffer.resize(next_size);
}

bool appendInt(vector<char>& buffer, size_t& length, int value)
{
  while (true)
  {
    ensureCsvCapacity(buffer, length, 16);
    auto result = to_chars(buffer.data() + length, buffer.data() + buffer.size(), value);
    if (result.ec == errc())
    {
      length = static_cast<size_t>(result.ptr - buffer.data());
      return true;
    }
    if (result.ec != errc::value_too_large)
    {
      return false;
    }
    buffer.resize(buffer.size() * 2);
  }
}

bool appendFloat(vector<char>& buffer, size_t& length, float value)
{
  while (true)
  {
    ensureCsvCapacity(buffer, length, 32);
    auto result = to_chars(buffer.data() + length, buffer.data() + buffer.size(), value,
                           chars_format::general, 6);
    if (result.ec == errc())
    {
      length = static_cast<size_t>(result.ptr - buffer.data());
      return true;
    }
    if (result.ec != errc::value_too_large)
    {
      return appendFormat(buffer, length, "%.6g", value);
    }
    buffer.resize(buffer.size() * 2);
  }
}

void appendChar(vector<char>& buffer, size_t& length, char c)
{
  ensureCsvCapacity(buffer, length, 1);
  buffer[length++] = c;
}

bool serializeCsv(const vector<POI2D>& poi_queue, SerializedCsv& csv_data,
                  const OutputFilter& filter)
{
  if (csv_data.buffer.size() < estimateCsvBytes(poi_queue.size()))
  {
    csv_data.buffer.resize(estimateCsvBytes(poi_queue.size()));
  }

  csv_data.length = 0;
  if (!appendFormat(csv_data.buffer, csv_data.length, "x,y,u,v,zncc,uy,vy,ux,vx\n"))
  {
    return false;
  }

  for (const auto& poi : poi_queue)
  {
    if (!filter.keep(static_cast<int>(poi.x), static_cast<int>(poi.y)))
    {
      continue;
    }
    if (!appendInt(csv_data.buffer, csv_data.length, static_cast<int>(poi.x)))
    {
      return false;
    }
    appendChar(csv_data.buffer, csv_data.length, ',');
    if (!appendInt(csv_data.buffer, csv_data.length, static_cast<int>(poi.y)))
    {
      return false;
    }
    appendChar(csv_data.buffer, csv_data.length, ',');
    if (!appendFloat(csv_data.buffer, csv_data.length, poi.deformation.u))
    {
      return false;
    }
    appendChar(csv_data.buffer, csv_data.length, ',');
    if (!appendFloat(csv_data.buffer, csv_data.length, poi.deformation.v))
    {
      return false;
    }
    appendChar(csv_data.buffer, csv_data.length, ',');
    if (!appendFloat(csv_data.buffer, csv_data.length, poi.result.zncc))
    {
      return false;
    }
    appendChar(csv_data.buffer, csv_data.length, ',');
    if (!appendFloat(csv_data.buffer, csv_data.length, poi.deformation.uy))
    {
      return false;
    }
    appendChar(csv_data.buffer, csv_data.length, ',');
    if (!appendFloat(csv_data.buffer, csv_data.length, poi.deformation.vy))
    {
      return false;
    }
    appendChar(csv_data.buffer, csv_data.length, ',');
    if (!appendFloat(csv_data.buffer, csv_data.length, poi.deformation.ux))
    {
      return false;
    }
    appendChar(csv_data.buffer, csv_data.length, ',');
    if (!appendFloat(csv_data.buffer, csv_data.length, poi.deformation.vx))
    {
      return false;
    }
    appendChar(csv_data.buffer, csv_data.length, '\n');
  }
  return true;
}

string inputImreadFlagsName(const ImageLoadOptions& options)
{
  return options.depth == ImageInputDepthMode::Compat8
           ? "IMREAD_GRAYSCALE"
           : "IMREAD_ANYDEPTH|IMREAD_GRAYSCALE";
}

Image2D loadConfiguredImage(const string& file_path, const ImageLoadOptions& options)
{
  Image2D image(1, 1);
  image.loadWithOptions(file_path, options);
  return image;
}

void printInputModeSummary(const CliOptions& options)
{
  cerr << "Input image mode: depth="
       << imageInputDepthModeName(options.input_options.depth)
       << ", align=" << imageInputAlignmentName(options.input_options.alignment)
       << ", normalize=" << imageInputNormalizeName(options.input_options.normalize)
       << ", flags=" << inputImreadFlagsName(options.input_options)
       << " (OpenCV grayscale conversion)" << endl;
}

void printLoadedImageSummary(const char* label, const ImageLoadInfo& info)
{
  cerr << label << ": source_depth=" << imageCvDepthName(info.source_depth)
       << ", source_range=[" << info.source_min << "," << info.source_max << "]"
       << ", effective_depth=" << imageInputDepthModeName(info.effective_depth)
       << ", effective_align=" << imageInputAlignmentName(info.effective_alignment)
       << ", output_depth=" << imageCvDepthName(info.output_depth)
       << ", output_range=[" << info.output_min << "," << info.output_max << "]"
       << ", scale=" << info.output_scale << endl;
}

// review-2026-10-03 #20. What the input files are, as opposed to how they were
// asked to be read. With --input-depth 8 OpenCV hands back 8-bit samples for a
// file of any depth, so the loader's own source_depth cannot see a deeper file
// and nothing downstream can tell. The reference is therefore read once more
// with IMREAD_ANYDEPTH (one extra decode per run), and every target is compared
// with the reference through its file header (a few small reads, no second
// decode per frame). None of this changes what the solver is given.
struct InputDepthAudit
{
  bool compat8 = false;        // --input-depth 8: samples arrive as 8-bit whatever the file holds
  bool reference_known = false;
  int reference_cv_depth = -1; // OpenCV depth of the reference file as stored
  double reference_min = 0.0;
  double reference_max = 0.0;
  opencorr_cli_io::StoredSampleDepth reference_header;
  size_t target_mismatches = 0; // targets whose stored depth differs from the reference
  string first_mismatch;
};

string storedDepthAdvice(const cv::Mat& stored, double stored_max)
{
  if (stored.depth() != CV_16U)
  {
    return "pass --input-depth 16 (or auto)";
  }
  if (stored_max <= 4095.0)
  {
    return "pass --input-depth 12 --input-align right (its values fit 12 bits, which looks like right-aligned 12-bit data)";
  }
  cv::Mat low_bits;
  cv::bitwise_and(stored, cv::Scalar(255), low_bits);
  if (cv::countNonZero(low_bits) == 0)
  {
    return "pass --input-depth 16 (every value is a multiple of 256, which looks like 8-bit data in a 16-bit file: this image itself loses nothing in the reduction)";
  }
  cv::bitwise_and(stored, cv::Scalar(15), low_bits);
  if (cv::countNonZero(low_bits) == 0)
  {
    return "pass --input-depth 12 --input-align left (every value is a multiple of 16, which looks like 12-bit data stored shifted left by 4)";
  }
  return "pass --input-depth 16";
}

InputDepthAudit auditReferenceDepth(const string& ref_path, const Image2D& ref_img,
                                    const CliOptions& options)
{
  InputDepthAudit audit;
  audit.compat8 = options.input_options.depth == ImageInputDepthMode::Compat8;
  audit.reference_header = opencorr_cli_io::probeStoredSampleDepth(ref_path);
  if (!audit.compat8)
  {
    // Decoded with IMREAD_ANYDEPTH already: the loader's source depth is the file's.
    audit.reference_known = ref_img.load_info.source_depth >= 0;
    audit.reference_cv_depth = ref_img.load_info.source_depth;
    audit.reference_min = ref_img.load_info.source_min;
    audit.reference_max = ref_img.load_info.source_max;
    return audit;
  }
  cv::Mat stored;
  try
  {
    stored = cv::imread(ref_path, cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE);
  }
  catch (const exception&)
  {
    // Advisory only: the load the run depends on has already succeeded.
  }
  if (!stored.data)
  {
    return audit;
  }
  audit.reference_known = true;
  audit.reference_cv_depth = stored.depth();
  cv::minMaxLoc(stored, &audit.reference_min, &audit.reference_max);
  if (stored.depth() != CV_8U && stored.depth() != CV_8S)
  {
    WarningLine() << "the reference image is stored as " << imageCvDepthName(stored.depth())
                  << " (values " << audit.reference_min << ".." << audit.reference_max
                  << ") but is read with --input-depth 8 (the default): OpenCV reduces every sample"
                     " to 8 bits before correlation (a 16-bit sample loses its low 8 bits, so right-aligned"
                     " 12-bit data keeps 4 bits). The run still exits 0 and looks valid; precision is lost"
                     " silently. To read the full depth, "
                  << storedDepthAdvice(stored, audit.reference_max)
                  << ", and use a new output directory. Reference: " << ref_path;
  }
  return audit;
}

// Warns about the first target whose stored depth differs from the reference
// and counts the rest; reportTargetDepthTotal() gives the total.
void auditTargetDepth(InputDepthAudit& audit, const string& label, const string& tar_path,
                      const ImageLoadInfo& tar_info)
{
  string reference_depth;
  string target_depth;
  if (audit.compat8)
  {
    if (!audit.reference_header.known())
    {
      return; // a container the header probe does not parse: nothing to compare
    }
    const auto header = opencorr_cli_io::probeStoredSampleDepth(tar_path);
    if (!header.known())
    {
      return; // a header the probe cannot read is not compared
    }
    // OpenCV expands every depth up to 8 bits to full-range 8-bit samples, so
    // those are one grey scale; only above 8 bits does the exact depth matter.
    const bool same_scale = header.deeperThan8()
                              ? header.bits == audit.reference_header.bits
                              : !audit.reference_header.deeperThan8();
    if (same_scale)
    {
      return;
    }
    reference_depth = to_string(audit.reference_header.bits) + "-bit";
    target_depth = to_string(header.bits) + "-bit";
  }
  else
  {
    if (!audit.reference_known || tar_info.source_depth == audit.reference_cv_depth)
    {
      return;
    }
    reference_depth = imageCvDepthName(audit.reference_cv_depth);
    target_depth = imageCvDepthName(tar_info.source_depth);
  }
  ++audit.target_mismatches;
  if (audit.target_mismatches > 1)
  {
    return;
  }
  audit.first_mismatch = label;
  WarningLine() << "target " << label << " is stored as " << target_depth
                << " samples but the reference as " << reference_depth
                << ": frames of different bit depth are not on one grey scale"
                << (audit.compat8 ? " (with --input-depth 8 each is reduced to 8 bits in its own way)" : "")
                << ", so their correlation is unreliable. Check how the sequence was exported. Target: "
                << tar_path;
}

void reportTargetDepthTotal(const InputDepthAudit& audit, size_t frames)
{
  if (audit.target_mismatches > 1)
  {
    WarningLine() << audit.target_mismatches << " of " << frames
                  << " target frames differ from the reference in stored bit depth (first: "
                  << audit.first_mismatch << ")";
  }
}

class ImagePreloader
{
public:
  explicit ImagePreloader(const ImageLoadOptions& input_options, int worker_count = 1)
      : input_options_(input_options)
  {
    const int workers = worker_count < 1 ? 1 : worker_count;
    workers_.reserve(static_cast<size_t>(workers));
    for (int i = 0; i < workers; ++i)
    {
      workers_.emplace_back(&ImagePreloader::workerMain, this);
    }
  }

  ~ImagePreloader()
  {
    stop();
  }

  // Queue one more frame for background decoding.  Requests are decoded
  // independently (possibly concurrently when more than one worker is
  // configured) but are delivered strictly in request order: the N-th
  // waitAndConsume returns the N-th request's image, so the decoded bytes a
  // frame receives do not depend on the worker count.
  void request(const string& tar_path)
  {
    {
      lock_guard<mutex> lock(mutex_);
      pending_.emplace_back(next_seq_, tar_path);
      ++next_seq_;
    }
    cv_request_.notify_one();
  }

  PreloadedImage waitAndConsume(const string& tar_path)
  {
    unique_lock<mutex> lock(mutex_);
    const uint64_t seq = next_consume_seq_;
    cv_ready_.wait(lock, [&]
    {
      return stop_ || results_.find(seq) != results_.end();
    });

    PreloadedImage result;
    auto it = results_.find(seq);
    if (it != results_.end())
    {
      result = std::move(it->second);
      results_.erase(it);
      if (result.tar_path != tar_path)
      {
        // Defensive: the request/consume interleave in runMultiFrame is FIFO,
        // so a mismatch means a caller-side ordering bug, not a decode race.
        result.success = false;
        result.image.reset();
        result.error = "preloader delivered out-of-order frame";
      }
    }
    ++next_consume_seq_;
    return result;
  }

  void stop()
  {
    {
      lock_guard<mutex> lock(mutex_);
      if (stop_)
      {
        return;
      }
      stop_ = true;
      cv_request_.notify_all();
      cv_ready_.notify_all();
    }
    for (thread& worker : workers_)
    {
      if (worker.joinable())
      {
        worker.join();
      }
    }
  }

private:
  void workerMain()
  {
    while (true)
    {
      uint64_t seq = 0;
      string tar_path;
      {
        unique_lock<mutex> lock(mutex_);
        cv_request_.wait(lock, [&]
        {
          return stop_ || !pending_.empty();
        });
        if (stop_)
        {
          return;
        }

        seq = pending_.front().first;
        tar_path = pending_.front().second;
        pending_.pop_front();
      }

      PreloadedImage next_image;
      next_image.tar_path = tar_path;
      double t0 = omp_get_wtime();
      try
      {
        next_image.image = make_unique<Image2D>(loadConfiguredImage(tar_path, input_options_));
        next_image.success = true;
      }
      catch (const exception& ex)
      {
        next_image.error = ex.what();
      }
      catch (const string& ex)
      {
        next_image.error = ex;
      }
      next_image.load_time = omp_get_wtime() - t0;

      {
        lock_guard<mutex> lock(mutex_);
        results_.emplace(seq, std::move(next_image));
      }
      cv_ready_.notify_all();
    }
  }

  vector<thread> workers_;
  mutex mutex_;
  condition_variable cv_request_;
  condition_variable cv_ready_;
  bool stop_ = false;
  uint64_t next_seq_ = 0;
  uint64_t next_consume_seq_ = 0;
  deque<pair<uint64_t, string>> pending_;
  map<uint64_t, PreloadedImage> results_;
  ImageLoadOptions input_options_;
};

class AsyncCsvWriter
{
public:
  explicit AsyncCsvWriter(OutputSession* session = nullptr, bool diagnostics = false)
      : session_(session), diagnostics_(diagnostics)
  {
    for (int i = 0; i < kNumSlots; ++i)
    {
      free_slots_.push_back(i);
    }
    worker_ = thread(&AsyncCsvWriter::workerMain, this);
  }

  ~AsyncCsvWriter()
  {
    finish();
  }

  bool submit(string output_path, PublishedPoiBuffer poi_queue,
              const OutputFilter& filter, double& wait_time, double& enqueue_time)
  {
    int slot_index = acquireSlot(wait_time);
    if (slot_index < 0)
    {
      return false;
    }

    double enqueue_start = omp_get_wtime();
    Slot& slot = slots_[slot_index];
    slot.output_path = std::move(output_path);
    slot.poi_queue = std::move(poi_queue);
    slot.filter = filter;
    enqueue_time = omp_get_wtime() - enqueue_start;

    {
      lock_guard<mutex> lock(mutex_);
      ready_slots_.push_back(slot_index);
    }
    cv_task_.notify_one();
    return true;
  }

  bool finish()
  {
    unique_lock<mutex> lock(mutex_);
    if (stopped_)
    {
      return last_success_;
    }

    cv_drain_.wait(lock, [&]
    {
      return ready_slots_.empty() && in_flight_ == 0;
    });

    bool success = last_success_;
    string error_message = last_error_;
    stop_ = true;
    stopped_ = true;
    cv_task_.notify_one();
    lock.unlock();

    if (worker_.joinable())
    {
      worker_.join();
    }

    if (!success && !error_message.empty())
    {
      cerr << error_message << endl;
    }
    return success;
  }

  double totalWriteTime() const
  {
    return total_write_time_;
  }

  double totalFormatTime() const
  {
    return total_format_time_;
  }

private:
  static constexpr int kNumSlots = 3;

  struct Slot
  {
    string output_path;
    PublishedPoiBuffer poi_queue;
    SerializedCsv csv_data;
    OutputFilter filter;
  };

  int acquireSlot(double& wait_time)
  {
    double wait_start = omp_get_wtime();
    unique_lock<mutex> lock(mutex_);
    cv_free_.wait(lock, [&]
    {
      return stop_ || !free_slots_.empty();
    });
    wait_time = omp_get_wtime() - wait_start;

    if (!last_success_)
    {
      cerr << last_error_ << endl;
      return -1;
    }

    if (stop_)
    {
      return -1;
    }

    int slot_index = free_slots_.front();
    free_slots_.erase(free_slots_.begin());
    ++in_flight_;
    return slot_index;
  }

  void workerMain()
  {
    while (true)
    {
      int slot_index = -1;
      {
        unique_lock<mutex> lock(mutex_);
        cv_task_.wait(lock, [&]
        {
          return stop_ || !ready_slots_.empty();
        });
        if (stop_ && ready_slots_.empty())
        {
          return;
        }

        slot_index = ready_slots_.front();
        ready_slots_.erase(ready_slots_.begin());
      }

      Slot& slot = slots_[slot_index];
      double format_start = omp_get_wtime();
      bool success = false;
      double format_time = 0.0;
      double write_time = 0.0;
      try
      {
        if (diagnostics_)
        {
          success = writeDiagnosticsCsv(slot.output_path, *slot.poi_queue, session_);
          write_time = omp_get_wtime() - format_start;
        }
        else
        {
          success = serializeCsv(*slot.poi_queue, slot.csv_data, slot.filter);
          format_time = omp_get_wtime() - format_start;
          if (success)
          {
            double write_start = omp_get_wtime();
            success = writeCsv(slot.output_path, slot.csv_data, session_);
            write_time = omp_get_wtime() - write_start;
          }
        }
      }
      catch (const exception& ex)
      {
        ErrorLine() << "CSV writer: " << ex.what();
        success = false;
      }

	      {
	        lock_guard<mutex> lock(mutex_);
	        if (!success)
	        {
	          last_success_ = false;
	          last_error_ = "Error: asynchronous CSV serialization/write failed for " + slot.output_path;
	        }
	        slot.output_path.clear();
        slot.poi_queue.reset();
        --in_flight_;
        free_slots_.push_back(slot_index);
        total_format_time_ += format_time;
        total_write_time_ += write_time;
      }
      cv_free_.notify_one();
      cv_drain_.notify_all();
    }
  }

  OutputSession* session_ = nullptr;
  bool diagnostics_ = false;
  array<Slot, kNumSlots> slots_;
  thread worker_;
  mutable mutex mutex_;
  condition_variable cv_task_;
  condition_variable cv_free_;
  condition_variable cv_drain_;
  bool stop_ = false;
  bool stopped_ = false;
  bool last_success_ = true;
  string last_error_;
  vector<int> free_slots_;
  vector<int> ready_slots_;
  int in_flight_ = 0;
  double total_format_time_ = 0.0;
  double total_write_time_ = 0.0;
};

// Two stages behind the main loop: a format thread turns a published POI buffer
// into the raw payload bytes, a publish thread hands those bytes to the output
// session (digest, ledger record, atomic file publication). Splitting them means
// the sustained writer rate is max(format, publish) rather than their sum, and
// the formatted payloads - 36 B/POI against 100 B/POI for a POI buffer - are the
// cheap place to hold the queue that absorbs writer jitter.
// Both stages are single-threaded and both queues are FIFO, so completed frames
// keep submission order for the manifest.
class AsyncRawBinaryWriter
{
public:
  explicit AsyncRawBinaryWriter(OutputSession* session = nullptr) : session_(session)
  {
    for (int i = 0; i < kNumSlots; ++i)
    {
      free_slots_.push_back(i);
    }
    for (int i = 0; i < kNumPayloads; ++i)
    {
      free_payloads_.push_back(i);
    }
    format_worker_ = thread(&AsyncRawBinaryWriter::formatMain, this);
    publish_worker_ = thread(&AsyncRawBinaryWriter::publishMain, this);
  }

  ~AsyncRawBinaryWriter()
  {
    finish();
  }

  bool submit(string output_path, OutputFrameRecord frame, PublishedPoiBuffer poi_queue,
              const OutputFilter& filter, double& wait_time, double& enqueue_time)
  {
    int slot_index = acquireSlot(wait_time);
    if (slot_index < 0)
    {
      return false;
    }

    double enqueue_start = omp_get_wtime();
    Slot& slot = slots_[slot_index];
    slot.output_path = std::move(output_path);
    slot.frame = std::move(frame);
    slot.poi_queue = std::move(poi_queue);
    slot.filter = filter;
    enqueue_time = omp_get_wtime() - enqueue_start;

    {
      lock_guard<mutex> lock(mutex_);
      ready_slots_.push_back(slot_index);
    }
    cv_format_.notify_one();
    return true;
  }

  bool finish()
  {
    unique_lock<mutex> lock(mutex_);
    if (stopped_)
    {
      return last_success_;
    }

    cv_drain_.wait(lock, [&]
    {
      return ready_slots_.empty() && in_flight_ == 0
             && ready_payloads_.empty() && publish_in_flight_ == 0;
    });

    bool success = last_success_;
    string error_message = last_error_;
    stop_ = true;
    stopped_ = true;
    cv_format_.notify_all();
    cv_publish_.notify_all();
    lock.unlock();

    if (format_worker_.joinable())
    {
      format_worker_.join();
    }
    if (publish_worker_.joinable())
    {
      publish_worker_.join();
    }

    if (!success && !error_message.empty())
    {
      cerr << error_message << endl;
    }
    return success;
  }

  vector<OutputFrameRecord> completedFrames() const
  {
    lock_guard<mutex> lock(mutex_);
    return completed_frames_;
  }

  double totalWriteTime() const
  {
    lock_guard<mutex> lock(mutex_);
    return total_write_time_;
  }

  double totalFormatTime() const
  {
    lock_guard<mutex> lock(mutex_);
    return total_format_time_;
  }

private:
  static constexpr int kNumSlots = 3;
  // Formatted payloads in flight between the two stages. 800f raw is
  // 171024 POI x 36 B = 5.87 MiB each, so this queue costs at most 47 MiB
  // resident; the buffers are pooled, so after the first frame the serializer
  // neither allocates nor zero-fills them.
  static constexpr int kNumPayloads = 8;

  struct Slot
  {
    string output_path;
    OutputFrameRecord frame;
    PublishedPoiBuffer poi_queue;
    OutputFilter filter;
  };

  struct Payload
  {
    vector<char> buffer;
    size_t used = 0;
    string output_path;
    OutputFrameRecord frame;
  };

  int acquireSlot(double& wait_time)
  {
    double wait_start = omp_get_wtime();
    unique_lock<mutex> lock(mutex_);
    cv_free_.wait(lock, [&]
    {
      return stop_ || !free_slots_.empty();
    });
    wait_time = omp_get_wtime() - wait_start;

    if (!last_success_)
    {
      cerr << last_error_ << endl;
      return -1;
    }

    if (stop_)
    {
      return -1;
    }

    int slot_index = free_slots_.front();
    free_slots_.erase(free_slots_.begin());
    ++in_flight_;
    return slot_index;
  }

  void formatMain()
  {
    while (true)
    {
      int slot_index = -1;
      int payload_index = -1;
      {
        unique_lock<mutex> lock(mutex_);
        cv_format_.wait(lock, [&]
        {
          return (stop_ && ready_slots_.empty())
                 || (!ready_slots_.empty() && !free_payloads_.empty());
        });
        if (ready_slots_.empty())
        {
          return; // implies stop_
        }

        slot_index = ready_slots_.front();
        ready_slots_.erase(ready_slots_.begin());
        payload_index = free_payloads_.front();
        free_payloads_.erase(free_payloads_.begin());
      }

      Slot& slot = slots_[slot_index];
      Payload& payload = payloads_[payload_index];
      RawBinaryWriteStats stats;
      bool success = false;
      try
      {
        formatRawBinary(*slot.poi_queue, slot.filter, payload.buffer, stats);
        success = true;
      }
      catch (const exception& ex) { ErrorLine() << "raw serializer: " << ex.what(); }
      if (success)
      {
        payload.used = stats.bytes;
        payload.output_path = slot.output_path;
        payload.frame = slot.frame;
        payload.frame.rows = stats.rows;
        payload.frame.bytes = stats.bytes;
        payload.frame.format_time = stats.format_time;
      }

      {
        lock_guard<mutex> lock(mutex_);
        if (!success)
        {
          // Nothing is handed to the publisher: a frame that failed to serialize
          // must never reach the output directory, not even as an empty file.
          last_success_ = false;
          last_error_ = "Error: asynchronous raw binary serialization failed for " + slot.output_path;
          free_payloads_.push_back(payload_index);
        }
        else
        {
          ready_payloads_.push_back(payload_index);
          ++publish_in_flight_;
        }
        slot.output_path.clear();
        slot.poi_queue.reset();
        slot.frame = OutputFrameRecord();
        --in_flight_;
        free_slots_.push_back(slot_index);
        total_format_time_ += stats.format_time;
      }
      cv_free_.notify_one();
      cv_publish_.notify_one();
      cv_drain_.notify_all();
    }
  }

  void publishMain()
  {
    while (true)
    {
      int payload_index = -1;
      {
        unique_lock<mutex> lock(mutex_);
        cv_publish_.wait(lock, [&]
        {
          return stop_ || !ready_payloads_.empty();
        });
        if (ready_payloads_.empty())
        {
          return; // implies stop_
        }

        payload_index = ready_payloads_.front();
        ready_payloads_.erase(ready_payloads_.begin());
      }

      Payload& payload = payloads_[payload_index];
      bool success = false;
      double write_start = omp_get_wtime();
      try
      {
        success = publishBytes(payload.output_path, payload.buffer.data(), payload.used, session_);
      }
      catch (const exception& ex) { ErrorLine() << "raw writer: " << ex.what(); }
      double write_time = omp_get_wtime() - write_start;
      OutputFrameRecord completed = std::move(payload.frame);
      completed.write_time = write_time;

      {
        lock_guard<mutex> lock(mutex_);
        if (!success)
        {
          last_success_ = false;
          last_error_ = "Error: asynchronous raw binary serialization/write failed for "
                        + payload.output_path;
        }
        else
        {
          completed_frames_.push_back(std::move(completed));
        }
        payload.output_path.clear();
        payload.frame = OutputFrameRecord();
        payload.used = 0;
        --publish_in_flight_;
        free_payloads_.push_back(payload_index);
        total_write_time_ += write_time;
      }
      cv_format_.notify_one();
      cv_free_.notify_one(); // a failed publish must release a blocked producer
      cv_drain_.notify_all();
    }
  }

  OutputSession* session_ = nullptr;
  array<Slot, kNumSlots> slots_;
  array<Payload, kNumPayloads> payloads_;
  thread format_worker_;
  thread publish_worker_;
  mutable mutex mutex_;
  condition_variable cv_format_;
  condition_variable cv_publish_;
  condition_variable cv_free_;
  condition_variable cv_drain_;
  bool stop_ = false;
  bool stopped_ = false;
  bool last_success_ = true;
  string last_error_;
  vector<int> free_slots_;
  vector<int> ready_slots_;
  vector<int> free_payloads_;
  vector<int> ready_payloads_;
  vector<OutputFrameRecord> completed_frames_;
  int in_flight_ = 0;
  int publish_in_flight_ = 0;
  double total_format_time_ = 0.0;
  double total_write_time_ = 0.0;
};

CliOptions defaultOptions();

// Usage errors print this to stderr and exit 1; --help prints it to stdout,
// followed by every option with its default, and exits 0.
void printUsage(ostream& out, bool with_defaults)
{
  out << "FastICGN (opencorr_cli_cuda_mvp)" << endl;
  out << "Usage: opencorr_cli_cuda_mvp <ref_image> <tar_image> <subset_radius> <grid_space> <output_path>" << endl;
  out << "       [--subset-size N] [--gpu-device N] [--max-iter M] [--conv-threshold T] [--line y1,y2] [--threads N] [--dual-gpu]" << endl;
  out << "       [--output-format csv|raw] [--diagnostics|--diagnostics-dir DIR] [--kernel legacy|warp] [--fftcc cpu|gpu]" << endl;
  out << "       [--input-depth 8|12|16|auto] [--input-align right|left|auto] [--input-normalize none|to-8|to-16|unit]" << endl;
  out << "       [--output-roi x1,y1,x2,y2] [--output-step N] [--durability rename|fsync] [--quiet]" << endl;
  out << "   or: opencorr_cli_cuda_mvp --multi-frame --ref <ref_image> --tar-list <tar_list.txt> --output-dir <output_dir>" << endl;
  out << "       <subset_radius> <grid_space> [--subset-size N] [--gpu-device N] [--max-iter M] [--conv-threshold T] [--line y1,y2]" << endl;
  out << "       [--threads N] [--dual-gpu] [--skip-fftcc|--no-skip-fftcc] [--output-format csv|raw]" << endl;
  out << "       [--poi-fallback|--no-poi-fallback] [--temporal-extrapolation] [--preload|--no-preload] [--decode-threads N]" << endl;
  out << "       [--diagnostics|--diagnostics-dir DIR] [--kernel legacy|warp] [--fftcc cpu|gpu]" << endl;
  out << "       [--output-roi x1,y1,x2,y2] [--output-step N]" << endl;
  out << "       [--input-depth 8|12|16|auto] [--input-align right|left|auto] [--input-normalize none|to-8|to-16|unit]" << endl;
  out << "       [--durability rename|fsync] [--repair] [--quiet]" << endl;
  out << "   or: opencorr_cli_cuda_mvp --verify-only <output_dir>" << endl;
  out << "   or: opencorr_cli_cuda_mvp --version" << endl;
  out << "   or: opencorr_cli_cuda_mvp --help" << endl;
  out << "<subset_radius> is the subset RADIUS in pixels. --subset-size N gives the same subset as the odd full SIZE N," << endl;
  out << "  with radius = (N - 1) / 2, and replaces the positional radius: --subset-size 15 == radius 7, and 7 != 15." << endl;
  out << "  Supplying both the positional radius and --subset-size is an error; the computed output is identical either way." << endl;
  out << "--durability selects how output is made permanent. rename (default) publishes every file through a private temporary plus atomic rename/link with zero fsync per frame," << endl;
  out << "  and takes one constant-cost barrier over the ledger and manifests when a multi-frame job completes; a killed process still never leaves a partial file." << endl;
  out << "  fsync additionally forces every payload, ledger record and directory to the device before reporting success (power-loss durability, about 4 fsync per frame)." << endl;
  out << "--repair moves damaged tracked frames (truncated, same-size corruption, missing intent) to .opencorr-state/quarantine/<timestamp>/ and regenerates them by replay;" << endl;
  out << "  crash temporaries are archived on every resume, with or without it." << endl;
  out << "  Changed inputs/parameters/binary, reordered or shortened target lists, unknown files and output symlinks stay rejected with or without --repair;" << endl;
  out << "  the rejection names the contract field that changed with its recorded and requested value." << endl;
  out << "--verify-only audits a multi-frame output directory without computing or writing anything: every payload against its ledger record, the completion record," << endl;
  out << "  the manifests, leftovers of interrupted runs and the input fingerprints. It prints one JSON object on stdout and exits 0 only when the directory is complete." << endl;
  out << "--kernel selects the ICGN iteration kernel: warp-per-POI (default) or legacy block-per-POI (pre-2026-09 production baseline)." << endl;
  out << "--fftcc selects the FFTCC initial-guess backend: gpu = batched cuFFT (default in CUDA builds), cpu = legacy FFTW (requires an FFTW-enabled build)." << endl;
  out << "--decode-threads sets the multi-frame image decode worker count (default 1); results are delivered in frame order and are identical for any count." << endl;
  out << "--threads, --preload, --decode-threads and --gpu-device only schedule the work: they never change an output byte and are not part of the resume contract." << endl;
  out << "--output-roi/--output-step crop only the written output (inclusive pixel rectangle; step N keeps POIs whose x and y are both multiples of N," << endl;
  out << "  which is not every N-th grid point); compute and QA still cover the full field." << endl;
  out << "Input defaults to --input-depth 8 --input-normalize none, matching legacy OpenCorr IMREAD_GRAYSCALE: a 16-bit file is then reduced to 8 bits (a warning says so)." << endl;
  out << "High-bit modes use IMREAD_ANYDEPTH|IMREAD_GRAYSCALE. For --input-depth 12, right keeps 0..4095;" << endl;
  out << "left divides uint16 storage by 16; auto chooses left only when the decoded range exceeds 4095." << endl;
  out << "--quiet drops the per-frame progress lines from stderr; the run header, warnings, errors and the closing summary stay." << endl;
  out << "stdout: every computing run, successful or not, ends with one JSON status line (schema fasticgn.run-status/1: status, exit_code, frames, output_dir," << endl;
  out << "  elapsed_sec, version, git_hash, warnings). --version, --help and --verify-only print their own answer instead. Everything else goes to stderr." << endl;
  if (!with_defaults)
  {
    out << "Run opencorr_cli_cuda_mvp --help for every option with its default." << endl;
    return;
  }

  const CliOptions defaults = defaultOptions();
  const auto onOff = [](bool value) { return value ? "on" : "off"; };
  out << endl;
  out << "Options (default in brackets):" << endl;
  out << "  --subset-size N               odd subset SIZE in pixels, instead of the positional radius [unset]" << endl;
  out << "  --gpu-device N                CUDA device for FFTCC and ICGN [" << defaults.gpu_device << "]" << endl;
  out << "  --dual-gpu                    split the ICGN POIs across CUDA devices 0 and 1 [" << onOff(defaults.dual_gpu) << "]" << endl;
  out << "  --max-iter M                  maximum ICGN iterations per POI [" << defaults.max_iteration << "]" << endl;
  out << "  --conv-threshold T            ICGN convergence threshold [" << defaults.conv_threshold << "]" << endl;
#ifdef OPENCORR_GPU_MVP_USE_CUDA
  out << "  --kernel legacy|warp          ICGN iteration kernel [" << iterationKernelModeName(defaults.kernel_mode) << "]" << endl;
#else
  out << "  --kernel legacy|warp          CUDA ICGN iteration kernel [warp; no effect in this CPU-only build]" << endl;
#endif
  out << "  --fftcc cpu|gpu               FFTCC initial-guess backend [" << fftccBackendName(defaults.fftcc_backend) << " in this build]" << endl;
  out << "  --threads N                   host OpenMP threads [logical CPUs - 1, at least 1: " << defaults.num_threads << " here]" << endl;
  out << "  --line y1,y2,...              compute only these image rows; repeatable [all rows]" << endl;
  out << "  --output-format csv|raw       payload encoding; binary is accepted for raw [" << outputFormatName(defaults.output_format) << "]" << endl;
  out << "  --binary-output               same as --output-format raw [off]" << endl;
  out << "  --output-roi x1,y1,x2,y2      write only POIs inside this inclusive rectangle [off]" << endl;
  out << "  --output-step N               write only POIs whose x and y are multiples of N [" << defaults.output_filter.step << "]" << endl;
  out << "  --diagnostics                 also write per-POI iteration/convergence CSV files [" << onOff(defaults.diagnostics_enabled) << "]" << endl;
  out << "  --diagnostics-dir DIR         where they go; implies --diagnostics [<output_dir>/diagnostics, single frame: <output stem>.diagnostics]" << endl;
  out << "  --input-depth 8|12|16|auto    how input samples are read [" << imageInputDepthModeName(defaults.input_options.depth) << "]" << endl;
  out << "  --input-align right|left|auto placement of 12-bit samples in a 16-bit file; only with --input-depth 12 ["
      << imageInputAlignmentName(defaults.input_options.alignment) << "]" << endl;
  out << "  --input-normalize none|to-8|to-16|unit  linear grey scaling before correlation ["
      << imageInputNormalizeName(defaults.input_options.normalize) << "]" << endl;
  out << "  --durability rename|fsync     when output reaches the device [" << opencorr_cli_io::durabilityName(defaults.durability) << "]" << endl;
  out << "  --quiet                       no per-frame progress lines on stderr [" << onOff(defaults.quiet) << "]" << endl;
  out << "Multi-frame only (accepted and ignored in single-frame mode):" << endl;
  out << "  --skip-fftcc | --no-skip-fftcc    seed each frame after the first from the previous one [" << onOff(defaults.skip_fftcc) << "]" << endl;
  out << "  --poi-fallback | --no-poi-fallback  re-seed failed POIs one by one [" << onOff(defaults.poi_fallback) << "]" << endl;
  out << "  --temporal-extrapolation | --no-temporal-extrapolation  two-frame displacement extrapolation seed ["
      << onOff(defaults.temporal_extrapolation) << "]" << endl;
  out << "  --preload | --no-preload      decode the next target image in the background [" << onOff(defaults.preload_enabled) << "]" << endl;
  out << "  --decode-threads N            image decode workers [" << defaults.decode_threads << "]" << endl;
  out << "  --repair | --no-repair        on resume, quarantine damaged tracked files and regenerate them [" << onOff(defaults.repair) << "]" << endl;
  out << "Exit status: 0 success (--verify-only: the directory is complete and verified); 1 any usage, input, compute or output error" << endl;
  out << "  (--verify-only: the directory is not complete, or could not be read)." << endl;
}

void printVersion()
{
  // The first four lines are a fixed interface (the pipeline and the release
  // checks read them); new facts are appended below them.
  cout << "FastICGN (opencorr_cli_cuda_mvp)" << endl;
  cout << "name=opencorr_cli_cuda_mvp" << endl;
  cout << "git_hash=" << OPENCORR_GPU_MVP_GIT_HASH << endl;
  cout << "build_date=" << OPENCORR_GPU_MVP_BUILD_DATE << endl;
  cout << "version=" << OPENCORR_GPU_MVP_VERSION << endl;
  cout << "git_dirty=" << OPENCORR_GPU_MVP_GIT_DIRTY << endl;
#ifdef OPENCORR_GPU_MVP_USE_CUDA
  cout << "backend=cuda" << endl;
#else
  cout << "backend=cpu" << endl;
#endif
#ifdef OPENCORR_HAS_FFTW
  cout << "fftw=1" << endl;
#else
  cout << "fftw=0" << endl;
#endif
}

CliOptions defaultOptions()
{
  CliOptions options;
  options.num_threads = omp_get_num_procs() - 1;
  if (options.num_threads < 1)
  {
    options.num_threads = 1;
  }
  return options;
}

bool parseIntStrict(const string& text, const string& name, int min_value, int max_value,
                    int& output, string& error)
{
  if (text.empty())
  {
    error = name + " must not be empty";
    return false;
  }

  int value = 0;
  const char* begin_ptr = text.data();
  const char* end_ptr = text.data() + text.size();
  auto result = from_chars(begin_ptr, end_ptr, value);
  if (result.ec != errc() || result.ptr != end_ptr)
  {
    error = name + " must be an integer, got '" + text + "'";
    return false;
  }
  if (value < min_value || value > max_value)
  {
    error = name + " out of range: " + text;
    return false;
  }

  output = value;
  return true;
}

bool parsePositiveFloatStrict(const string& text, const string& name, float& output, string& error)
{
  if (text.empty())
  {
    error = name + " must not be empty";
    return false;
  }

  float value = 0.0f;
  const char* begin_ptr = text.data();
  const char* end_ptr = text.data() + text.size();
  auto result = from_chars(begin_ptr, end_ptr, value);
  if (result.ec != errc() || result.ptr != end_ptr || !isfinite(value) || value <= 0.0f)
  {
    error = name + " must be a finite positive number, got '" + text + "'";
    return false;
  }

  output = value;
  return true;
}

bool parseLineYsStrict(const string& text, vector<int>& line_ys, string& error)
{
  if (text.empty())
  {
    error = "--line must not be empty";
    return false;
  }

  vector<int> parsed;
  stringstream ss(text);
  string token;
  while (getline(ss, token, ','))
  {
    int y = 0;
    if (!parseIntStrict(token, "--line", 0, numeric_limits<int>::max(), y, error))
    {
      return false;
    }
    parsed.push_back(y);
  }

  if (parsed.empty())
  {
    error = "--line must include at least one y coordinate";
    return false;
  }

  line_ys.insert(line_ys.end(), parsed.begin(), parsed.end());
  return true;
}

bool parseOutputFormatStrict(const string& text, OutputFormat& output_format, string& error)
{
  if (text == "csv")
  {
    output_format = OutputFormat::Csv;
    return true;
  }
  if (text == "raw" || text == "binary")
  {
    output_format = OutputFormat::RawBinary;
    return true;
  }

  error = "--output-format must be csv or raw, got '" + text + "'";
  return false;
}

bool parseOutputRoiStrict(const string& text, OutputFilter& filter, string& error)
{
  if (text.empty())
  {
    error = "--output-roi must not be empty";
    return false;
  }

  vector<int> parsed;
  stringstream ss(text);
  string token;
  while (getline(ss, token, ','))
  {
    int value = 0;
    if (!parseIntStrict(token, "--output-roi", numeric_limits<int>::min(),
                        numeric_limits<int>::max(), value, error))
    {
      return false;
    }
    parsed.push_back(value);
  }

  if (parsed.size() != 4)
  {
    error = "--output-roi must be x1,y1,x2,y2 (four integers), got '" + text + "'";
    return false;
  }
  if (parsed[0] > parsed[2] || parsed[1] > parsed[3])
  {
    error = "--output-roi requires x1<=x2 and y1<=y2, got '" + text + "'";
    return false;
  }

  filter.roi_enabled = true;
  filter.roi_x1 = parsed[0];
  filter.roi_y1 = parsed[1];
  filter.roi_x2 = parsed[2];
  filter.roi_y2 = parsed[3];
  return true;
}

bool parseIterationKernelModeStrict(const string& text, ICGN2D1IterationKernelMode& mode, string& error)
{
  if (text == "legacy")
  {
    mode = ICGN2D1IterationKernelMode::Legacy;
    return true;
  }
  if (text == "warp")
  {
    mode = ICGN2D1IterationKernelMode::Warp;
    return true;
  }

  error = "--kernel must be legacy or warp, got '" + text + "'";
  return false;
}

bool parseFftccBackendStrict(const string& text, FftccBackend& backend, string& error)
{
  if (text == "cpu")
  {
    backend = FftccBackend::Cpu;
    return true;
  }
  if (text == "gpu")
  {
    backend = FftccBackend::Gpu;
    return true;
  }

  error = "--fftcc must be cpu or gpu, got '" + text + "'";
  return false;
}

bool fftccBackendAvailable(FftccBackend backend)
{
  if (backend == FftccBackend::Cpu)
  {
#ifdef OPENCORR_HAS_FFTW
    return true;
#else
    return false;
#endif
  }
#ifdef OPENCORR_GPU_MVP_USE_CUDA
  return true;
#else
  return false;
#endif
}

bool parseInputDepthStrict(const string& text, ImageInputDepthMode& depth, string& error)
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

  error = "--input-depth must be 8, 12, 16, or auto, got '" + text + "'";
  return false;
}

bool parseInputAlignmentStrict(const string& text, ImageInputAlignment& alignment, string& error)
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

  error = "--input-align must be right, left, or auto, got '" + text + "'";
  return false;
}

bool parseInputNormalizeStrict(const string& text, ImageInputNormalize& normalize, string& error)
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

  error = "--input-normalize must be none, to-8, to-16, or unit, got '" + text + "'";
  return false;
}

bool validateInputOptions(const CliOptions& options, string& error)
{
#ifndef OPENCORR_GPU_MVP_USE_CUDA
  if (options.dual_gpu)
  {
    error = "--dual-gpu requires a CUDA-enabled build";
    return false;
  }
#endif
  if (options.input_options.depth != ImageInputDepthMode::Depth12
      && options.input_options.alignment != ImageInputAlignment::Auto)
  {
    error = "--input-align is only meaningful with --input-depth 12; use --input-depth 12 for left/right 12-bit storage";
    return false;
  }
  if (!fftccBackendAvailable(options.fftcc_backend))
  {
    error = string("--fftcc ") + fftccBackendName(options.fftcc_backend)
            + " is not available in this build ("
            + (options.fftcc_backend == FftccBackend::Cpu
                   ? "built without FFTW: configure with OPENCORR_GPU_MVP_WITH_FFTW=ON"
                   : "built without CUDA support")
            + ")";
    return false;
  }
  return true;
}

vector<POI2D> buildPoiQueue(const Image2D& ref_img, int subset_radius, int grid_space,
                            const vector<int>& line_ys)
{
  vector<POI2D> poi_queue;
  int margin = subset_radius + 2;

  if (!line_ys.empty())
  {
    for (int y : line_ys)
    {
      if (y < margin || y >= ref_img.height - margin)
      {
        WarningLine() << "line y=" << y << " too close to edge, skipping";
        continue;
      }
      for (int x = margin; x < ref_img.width - margin; x += grid_space)
      {
        poi_queue.emplace_back(Point2D(static_cast<float>(x), static_cast<float>(y)));
      }
    }
  }
  else
  {
    for (int y = margin; y < ref_img.height - margin; y += grid_space)
    {
      for (int x = margin; x < ref_img.width - margin; x += grid_space)
      {
        poi_queue.emplace_back(Point2D(static_cast<float>(x), static_cast<float>(y)));
      }
    }
  }

  return poi_queue;
}

bool writeCsv(const string& output_path, const vector<POI2D>& poi_queue,
              const OutputFilter& filter)
{
  SerializedCsv csv_data;
  if (!serializeCsv(poi_queue, csv_data, filter))
  {
    ErrorLine() << "failed to serialize CSV data for " << output_path;
    return false;
  }
  return writeCsv(output_path, csv_data);
}

bool writeCsv(const string& output_path, const SerializedCsv& csv_data, OutputSession* session)
{
  return publishBytes(output_path, csv_data.buffer.data(), csv_data.length, session);
}

template <typename T>
void writePod(char*& cursor, const T& value)
{
  memcpy(cursor, &value, sizeof(T));
  cursor += sizeof(T);
}

void formatRawBinary(const vector<POI2D>& poi_queue, const OutputFilter& filter,
                     vector<char>& buffer, RawBinaryWriteStats& stats)
{
  double format_start = omp_get_wtime();
  // A pooled buffer that is already large enough is reused as is: resize() only
  // grows it, so a steady frame size costs neither an allocation nor a zero fill
  // after the first frame. Stale bytes past `used` are never published.
  const size_t capacity = poi_queue.size() * kRawBinaryRecordBytes;
  if (buffer.size() < capacity)
  {
    buffer.resize(capacity);
  }
  char* cursor = buffer.data();
  for (const auto& poi : poi_queue)
  {
    int32_t x = static_cast<int32_t>(poi.x);
    int32_t y = static_cast<int32_t>(poi.y);
    if (!filter.keep(x, y))
    {
      continue;
    }
    writePod(cursor, x);
    writePod(cursor, y);
    writePod(cursor, poi.deformation.u);
    writePod(cursor, poi.deformation.v);
    writePod(cursor, poi.deformation.ux);
    writePod(cursor, poi.deformation.uy);
    writePod(cursor, poi.deformation.vx);
    writePod(cursor, poi.deformation.vy);
    writePod(cursor, poi.result.zncc);
  }
  // With an active filter fewer rows are emitted than poi_queue.size(); the
  // buffer is sized for the full queue, so only the filled prefix is written.
  // With the filter off the prefix is the whole buffer, as before.
  const size_t used = static_cast<size_t>(cursor - buffer.data());
  stats.format_time = omp_get_wtime() - format_start;
  stats.rows = used / kRawBinaryRecordBytes;
  stats.bytes = used;
}

// review-2026-09-16 section 4: a manifest that records only a few parameters
// cannot be replayed or audited. This block names the engine that actually ran:
// kernel mode, FFT backend, the skip/fallback/temporal policies, durability,
// build identity, device and thread counts. It is additive metadata; frame
// payload bytes are untouched, and the ledger digests the manifest it finds on
// disk, so the completion record stays consistent by construction.
void writeEngineMetadata(ostringstream& out, const CliOptions& options,
                         const char* indent)
{
  out << indent << "\"engine\": {\n";
  out << indent << "  \"implementation\": \"FastICGN\",\n";
  out << indent << "  \"git_hash\": \"" << jsonEscape(OPENCORR_GPU_MVP_GIT_HASH) << "\",\n";
  out << indent << "  \"build_date\": \"" << jsonEscape(OPENCORR_GPU_MVP_BUILD_DATE) << "\",\n";
  // Semantic version, and whether tracked files differed from git_hash when
  // this binary was built (null: not built from a git checkout).
  out << indent << "  \"version\": \"" << jsonEscape(OPENCORR_GPU_MVP_VERSION) << "\",\n";
  out << indent << "  \"git_dirty\": "
      << (string(OPENCORR_GPU_MVP_GIT_DIRTY) == "1" ? "true"
          : string(OPENCORR_GPU_MVP_GIT_DIRTY) == "0" ? "false" : "null") << ",\n";
#ifdef OPENCORR_GPU_MVP_USE_CUDA
  out << indent << "  \"icgn_backend\": \"cuda\",\n";
#else
  out << indent << "  \"icgn_backend\": \"cpu\",\n";
#endif
  out << indent << "  \"icgn_kernel\": \"" << iterationKernelModeName(options.kernel_mode) << "\",\n";
  // Numerical-semantics marker, so a raw tree says which out-of-bounds rule
  // produced it without anyone having to date it against a commit. Bump the
  // version whenever that rule changes. "geometric-v1": a warped target subset
  // with any sample outside the interpolatable range is rejected with
  // zncc = -3; see docs/output-format.md.
  out << indent << "  \"oob_policy\": \"geometric-v1\",\n";
  out << indent << "  \"fftcc_backend\": \"" << fftccBackendName(options.fftcc_backend) << "\",\n";
  out << indent << "  \"skip_fftcc\": " << (options.skip_fftcc ? "true" : "false") << ",\n";
  out << indent << "  \"poi_fallback\": " << (options.poi_fallback ? "true" : "false") << ",\n";
  out << indent << "  \"temporal_extrapolation\": "
      << (options.temporal_extrapolation ? "true" : "false") << ",\n";
  // Numerical-semantics marker for the skip-FFTCC path's whole-frame fallback,
  // versioned like oob_policy. "per-poi-merge-v1": the from-scratch pass and
  // the skip attempt are merged per POI by shouldTakeFrameFallbackResult();
  // see docs/output-format.md.
  out << indent << "  \"frame_fallback_policy\": \"per-poi-merge-v1\",\n";
  out << indent << "  \"preload\": " << (options.preload_enabled ? "true" : "false") << ",\n";
  out << indent << "  \"durability\": \""
      << opencorr_cli_io::durabilityName(opencorr_cli_io::defaultDurability()) << "\",\n";
  out << indent << "  \"threads\": " << options.num_threads << ",\n";
  out << indent << "  \"decode_threads\": " << options.decode_threads << ",\n";
  out << indent << "  \"dual_gpu\": " << (options.dual_gpu ? "true" : "false") << ",\n";
  out << indent << "  \"gpu_device\": " << options.gpu_device << ",\n";
  out << indent << "  \"cuda\": ";
#ifdef OPENCORR_GPU_MVP_USE_CUDA
  {
    cudaDeviceProp properties{};
    const bool have_properties =
        cudaGetDeviceProperties(&properties, options.gpu_device) == cudaSuccess;
    int driver_version = 0;
    int runtime_version = 0;
    const bool have_driver = cudaDriverGetVersion(&driver_version) == cudaSuccess;
    const bool have_runtime = cudaRuntimeGetVersion(&runtime_version) == cudaSuccess;
    out << "{\n";
    out << indent << "    \"device_name\": ";
    if (have_properties) out << "\"" << jsonEscape(properties.name) << "\"";
    else out << "null";
    out << ",\n" << indent << "    \"compute_capability\": ";
    if (have_properties) out << "\"" << properties.major << "." << properties.minor << "\"";
    else out << "null";
    out << ",\n" << indent << "    \"driver_version\": ";
    if (have_driver) out << driver_version; else out << "null";
    out << ",\n" << indent << "    \"runtime_version\": ";
    if (have_runtime) out << runtime_version; else out << "null";
    out << "\n" << indent << "  }";
  }
#else
  out << "null";
#endif
  out << "\n" << indent << "},\n";
}

bool writeRawManifest(const fs::path& manifest_path, const string& ref_path,
                      const string& tar_list_path, const string& output_dir,
                      int subset_radius, int grid_space, const CliOptions& options,
                      const vector<OutputFrameRecord>& frames, const InputDepthAudit& input_audit)
{
  ostringstream out;

  out << "{\n";
  out << "  \"format\": \"opencorr-gpu-raw-v1\",\n";
  out << "  \"endianness\": \"little\",\n";
  out << "  \"record_bytes\": " << kRawBinaryRecordBytes << ",\n";
  out << "  \"ref_path\": \"" << jsonEscape(ref_path) << "\",\n";
  out << "  \"tar_list_path\": \"" << jsonEscape(tar_list_path) << "\",\n";
  out << "  \"output_dir\": \"" << jsonEscape(output_dir) << "\",\n";
  out << "  \"subset_radius\": " << subset_radius << ",\n";
  out << "  \"grid_space\": " << grid_space << ",\n";
  out << "  \"conv_threshold\": " << options.conv_threshold << ",\n";
  out << "  \"max_iteration\": " << options.max_iteration << ",\n";
  out << "  \"durability\": \""
      << opencorr_cli_io::durabilityName(opencorr_cli_io::defaultDurability()) << "\",\n";
  writeEngineMetadata(out, options, "  ");
  if (options.output_filter.active())
  {
    out << "  \"output_filter\": {\n";
    out << "    \"roi_enabled\": " << (options.output_filter.roi_enabled ? "true" : "false");
    if (options.output_filter.roi_enabled)
    {
      out << ",\n    \"roi\": [" << options.output_filter.roi_x1 << ", "
          << options.output_filter.roi_y1 << ", " << options.output_filter.roi_x2
          << ", " << options.output_filter.roi_y2 << "]";
    }
    out << ",\n    \"step\": " << options.output_filter.step << "\n";
    out << "  },\n";
  }
  out << "  \"input_image\": {\n";
  out << "    \"depth\": \"" << imageInputDepthModeName(options.input_options.depth) << "\",\n";
  out << "    \"align\": \"" << imageInputAlignmentName(options.input_options.alignment) << "\",\n";
  out << "    \"normalize\": \"" << imageInputNormalizeName(options.input_options.normalize) << "\",\n";
  out << "    \"imread_flags\": \"" << inputImreadFlagsName(options.input_options) << "\",\n";
  // What the files are, as opposed to how they were asked to be read (the three
  // fields above). Information only: none of this enters the resume contract.
  out << "    \"reference_source_depth\": ";
  if (input_audit.reference_known) out << "\"" << imageCvDepthName(input_audit.reference_cv_depth) << "\"";
  else out << "null";
  out << ",\n    \"reference_source_range\": ";
  if (input_audit.reference_known && isfinite(input_audit.reference_min) && isfinite(input_audit.reference_max))
    out << "[" << input_audit.reference_min << ", " << input_audit.reference_max << "]";
  else out << "null";
  out << ",\n    \"target_source_depth_mismatches\": " << input_audit.target_mismatches << ",\n";
  out << "    \"gray_conversion\": \"OpenCV grayscale imread flag; color inputs are converted to one-channel grayscale by OpenCV\"\n";
  out << "  },\n";
  out << "  \"fields\": [\n";
  out << "    {\"name\":\"x\",\"dtype\":\"int32\",\"offset\":0},\n";
  out << "    {\"name\":\"y\",\"dtype\":\"int32\",\"offset\":4},\n";
  out << "    {\"name\":\"u\",\"dtype\":\"float32\",\"offset\":8},\n";
  out << "    {\"name\":\"v\",\"dtype\":\"float32\",\"offset\":12},\n";
  out << "    {\"name\":\"ux\",\"dtype\":\"float32\",\"offset\":16},\n";
  out << "    {\"name\":\"uy\",\"dtype\":\"float32\",\"offset\":20},\n";
  out << "    {\"name\":\"vx\",\"dtype\":\"float32\",\"offset\":24},\n";
  out << "    {\"name\":\"vy\",\"dtype\":\"float32\",\"offset\":28},\n";
  out << "    {\"name\":\"zncc\",\"dtype\":\"float32\",\"offset\":32}\n";
  out << "  ],\n";
  out << "  \"csv_field_order\": [\"x\",\"y\",\"u\",\"v\",\"zncc\",\"uy\",\"vy\",\"ux\",\"vx\"],\n";
  out << "  \"frames\": [\n";
  for (size_t i = 0; i < frames.size(); ++i)
  {
    const auto& frame = frames[i];
    out << "    {\"tar_path\":\"" << jsonEscape(frame.tar_path)
        << "\",\"output_name\":\"" << jsonEscape(frame.output_name)
        << "\",\"file\":\"" << jsonEscape(frame.file_name)
        << "\",\"rows\":" << frame.rows
        << ",\"bytes\":" << frame.bytes
        << ",\"format_sec\":" << frame.format_time
        << ",\"write_sec\":" << frame.write_time
        << ",\"bad_ratio\":";
    writeJsonNullableFloat(out, frame.qa.has_bad_ratio, frame.qa.bad_ratio);
    out << ",\"baseline_bad_ratio\":";
    writeJsonNullableFloat(out, frame.qa.has_baseline_bad_ratio,
                           frame.qa.baseline_bad_ratio);
    out << ",\"fallback_threshold\":";
    writeJsonNullableFloat(out, frame.qa.has_fallback_threshold,
                           frame.qa.fallback_threshold);
    out << ",\"fallback_used\":";
    writeJsonNullableBool(out, frame.qa.has_fallback_used, frame.qa.fallback_used);
    out << ",\"skip_fftcc_used\":";
    writeJsonNullableBool(out, frame.qa.has_skip_fftcc_used,
                          frame.qa.skip_fftcc_used);
    out << ",\"valid_ratio\":";
    writeJsonNullableFloat(out, frame.qa.has_valid_ratio, frame.qa.valid_ratio);
    out << ",\"iteration_mean\":";
    writeJsonNullableFloat(out, frame.qa.has_iteration_mean,
                           frame.qa.iteration_mean);
    out << ",\"iteration_max\":";
    writeJsonNullableInt(out, frame.qa.has_iteration_max, frame.qa.iteration_max);
    out << ",\"poi_fallback_count\":";
    writeJsonNullableInt(out, frame.qa.has_poi_fallback_count,
                         frame.qa.poi_fallback_count);
    out << ",\"poi_fallback_recovered\":";
    writeJsonNullableInt(out, frame.qa.has_poi_fallback_recovered,
                         frame.qa.poi_fallback_recovered);
    out << ",\"temporal_seed_used\":";
    writeJsonNullableBool(out, frame.qa.has_temporal_seed_used,
                          frame.qa.temporal_seed_used);
    out << ",\"temporal_seed_count\":";
    writeJsonNullableInt(out, frame.qa.has_temporal_seed_count,
                         frame.qa.temporal_seed_count);
    out << "}";
    if (i + 1 < frames.size())
    {
      out << ",";
    }
    out << "\n";
  }
  out << "  ]\n";
  out << "}\n";
  if (!out.good()) return false;
  return publishMetadata(manifest_path, out.str());
}

bool writeDiagnosticsCsv(const fs::path& diagnostics_path, const vector<POI2D>& poi_queue, OutputSession* session)
{
  ostringstream out;

  out << setprecision(9);
  out << "x,y,iteration,convergence,status,zncc\n";
  for (const auto& poi : poi_queue)
  {
    out << static_cast<int>(poi.x) << ","
        << static_cast<int>(poi.y) << ","
        << poi.result.iteration << ","
        << poi.result.convergence << ","
        << diagnosticStatusFromZncc(poi.result.zncc) << ","
        << poi.result.zncc << "\n";
  }

  if (!out.good())
  {
    ErrorLine() << "failed to write diagnostics file " << diagnostics_path.string();
    return false;
  }
  const auto bytes = out.str();
  return publishBytes(diagnostics_path, bytes.data(), bytes.size(), session);
}

bool writeDiagnosticsManifest(const fs::path& manifest_path, const string& ref_path,
                              const string& tar_list_path, const CliOptions& options,
                              const vector<DiagnosticsFrameRecord>& frames)
{
  ostringstream out;

  out << "{\n";
  out << "  \"format\": \"opencorr-gpu-diagnostics-csv-v1\",\n";
  out << "  \"ref_path\": \"" << jsonEscape(ref_path) << "\",\n";
  out << "  \"tar_list_path\": \"" << jsonEscape(tar_list_path) << "\",\n";
  out << "  \"status_semantics\": \"1 means zncc>=0 success; negative values mirror failure/status codes carried by zncc; non-finite zncc maps to -5\",\n";
  out << "  \"durability\": \""
      << opencorr_cli_io::durabilityName(opencorr_cli_io::defaultDurability()) << "\",\n";
  writeEngineMetadata(out, options, "  ");
  out << "  \"columns\": [\"x\",\"y\",\"iteration\",\"convergence\",\"status\",\"zncc\"],\n";
  out << "  \"frames\": [\n";
  for (size_t i = 0; i < frames.size(); ++i)
  {
    const auto& frame = frames[i];
    out << "    {\"tar_path\":\"" << jsonEscape(frame.tar_path)
        << "\",\"output_name\":\"" << jsonEscape(frame.output_name)
        << "\",\"file\":\"" << jsonEscape(frame.file_name)
        << "\",\"rows\":" << frame.rows << "}";
    if (i + 1 < frames.size())
    {
      out << ",";
    }
    out << "\n";
  }
  out << "  ]\n";
  out << "}\n";
  if (!out.good()) return false;
  return publishMetadata(manifest_path, out.str());
}

fs::path defaultDiagnosticsDirForSingleFrame(const fs::path& output_path)
{
  fs::path dir_name = output_path.stem();
  dir_name += ".diagnostics";
  fs::path parent = output_path.parent_path();
  return parent.empty() ? dir_name : parent / dir_name;
}

fs::path diagnosticsPathForFrame(const fs::path& diagnostics_dir, const fs::path& output_path)
{
  fs::path file_name = output_path.stem();
  file_name += ".diagnostics.csv";
  return diagnostics_dir / file_name;
}

// review-2026-09-02 section 4: thin owning wrapper that dispatches the FFTCC
// initial-guess stage to the legacy FFTW CPU class or the batched-cuFFT GPU
// class, keeping processFrame and the POI-fallback path backend-agnostic.
// Call signatures mirror FFTCC2D so both backends share call sites.
class FftccEngine
{
public:
  FftccEngine(FftccBackend backend, int subset_radius, int thread_number, int device_id)
  {
#ifdef OPENCORR_HAS_FFTW
    if (backend == FftccBackend::Cpu)
    {
      cpu_ = make_unique<FFTCC2D>(subset_radius, subset_radius, thread_number);
      return;
    }
#endif
#ifdef OPENCORR_GPU_MVP_USE_CUDA
    if (backend == FftccBackend::Gpu)
    {
      gpu_ = make_unique<FFTCC2DCuda>(subset_radius, subset_radius, device_id);
      return;
    }
#endif
    throw runtime_error(string("FFTCC backend '") + fftccBackendName(backend)
                        + "' is not available in this build");
  }

  void setImages(Image2D& ref_img, Image2D& tar_img)
  {
#ifdef OPENCORR_HAS_FFTW
    if (cpu_)
    {
      cpu_->setImages(ref_img, tar_img);
      return;
    }
#endif
#ifdef OPENCORR_GPU_MVP_USE_CUDA
    gpu_->setImages(ref_img, tar_img);
#endif
  }

  void prepareTar()
  {
#ifdef OPENCORR_HAS_FFTW
    if (cpu_)
    {
      cpu_->prepareTar();
      return;
    }
#endif
#ifdef OPENCORR_GPU_MVP_USE_CUDA
    gpu_->prepareTar();
#endif
  }

  void setTarImage(Image2D& tar_img)
  {
    if (refImage() == nullptr)
      throw logic_error("FftccEngine::setTarImage requires a reference image");
#ifdef OPENCORR_HAS_FFTW
    if (cpu_)
    {
      cpu_->setTarImage(tar_img);
      return;
    }
#endif
#ifdef OPENCORR_GPU_MVP_USE_CUDA
    gpu_->setTarImage(tar_img);
#endif
  }

  const Image2D* refImage() const
  {
#ifdef OPENCORR_HAS_FFTW
    if (cpu_) return cpu_->ref_img;
#endif
#ifdef OPENCORR_GPU_MVP_USE_CUDA
    return gpu_->refImage();
#else
    return nullptr;
#endif
  }

  bool hasPreparedReference(const Image2D& ref) const
  {
#ifdef OPENCORR_HAS_FFTW
    if (cpu_) return cpu_->hasPreparedReference(ref);
#endif
#ifdef OPENCORR_GPU_MVP_USE_CUDA
    return gpu_->hasPreparedReference(ref);
#else
    return false;
#endif
  }

  void prepareRef(const vector<POI2D>& poi_queue)
  {
#ifdef OPENCORR_HAS_FFTW
    if (cpu_)
    {
      cpu_->prepareRef(poi_queue);
      return;
    }
#endif
#ifdef OPENCORR_GPU_MVP_USE_CUDA
    gpu_->prepareRef(poi_queue);
#endif
  }

  void compute(vector<POI2D>& poi_queue)
  {
#ifdef OPENCORR_HAS_FFTW
    if (cpu_)
    {
      cpu_->compute(poi_queue);
      return;
    }
#endif
#ifdef OPENCORR_GPU_MVP_USE_CUDA
    gpu_->compute(poi_queue);
#endif
  }

  void compute(vector<POI2D>& poi_queue, const vector<size_t>& cache_indices)
  {
#ifdef OPENCORR_HAS_FFTW
    if (cpu_)
    {
      cpu_->compute(poi_queue, cache_indices);
      return;
    }
#endif
#ifdef OPENCORR_GPU_MVP_USE_CUDA
    gpu_->compute(poi_queue, cache_indices);
#endif
  }

private:
#ifdef OPENCORR_HAS_FFTW
  unique_ptr<FFTCC2D> cpu_;
#endif
#ifdef OPENCORR_GPU_MVP_USE_CUDA
  unique_ptr<FFTCC2DCuda> gpu_;
#endif
};

size_t runPoiLevelFallback(const vector<POI2D>& poi_template,
                           const vector<size_t>& fallback_indices,
                           FftccEngine& fftcc, ICGN2D1CudaMvp& main_icgn,
                           ICGN2D1CudaMvp& icgn,
                           vector<POI2D>& poi_queue,
                           double& fftcc_time, double& icgn_time,
                           FrameProcessStats& stats)
{
  if (fallback_indices.empty())
  {
    return size_t{0};
  }

  vector<POI2D> fallback_queue;
  fallback_queue.reserve(fallback_indices.size());
  for (size_t index : fallback_indices)
  {
    fallback_queue.push_back(poi_template[index]);
  }

  double fftcc_start = omp_get_wtime();
  // review-2026-09-02 §3.2: reuse the reference FFTs cached by
  // prepareRef(poi_template), keyed by template index; the overload falls back
  // to the uncached path when the cache does not cover these POIs.
  fftcc.compute(fallback_queue, fallback_indices);
  fftcc_time += omp_get_wtime() - fftcc_start;

  double icgn_start = omp_get_wtime();
  // review-2026-09-02 §3.2: gather the main instance's resident per-POI
  // precompute instead of re-running the precompute kernel on the small queue;
  // falls back to the legacy compute() when the fast-path preconditions fail
  // (e.g. dual GPU or the main instance doubling as the repair instance).
  vector<int> donor_indices;
  donor_indices.reserve(fallback_indices.size());
  for (size_t index : fallback_indices)
  {
    donor_indices.push_back(static_cast<int>(index));
  }
  if (!icgn.computeGathered(main_icgn, donor_indices, fallback_queue))
  {
    icgn.compute(fallback_queue);
  }
  double icgn_elapsed = omp_get_wtime() - icgn_start;
  icgn_time += icgn_elapsed;
  stats.poi_fallback_timing.compute_calls += 1;
  stats.poi_fallback_timing.compute_sec += icgn_elapsed;

  size_t recovered = 0;
  for (size_t i = 0; i < fallback_indices.size(); ++i)
  {
    size_t index = fallback_indices[i];
    if (shouldReplacePoiWithFallback(poi_queue[index], fallback_queue[i]))
    {
      bool recovered_this_poi =
          !isBadPoiForFallback(fallback_queue[i], kSkipFftccZnccThreshold);
      poi_queue[index] = fallback_queue[i];
      if (recovered_this_poi)
      {
        ++recovered;
      }
    }
  }

  stats.poi_fallback_used = true;
  stats.poi_fallback_count = static_cast<int>(fallback_indices.size());
  stats.poi_fallback_recovered = static_cast<int>(recovered);
  return recovered;
}

bool processFrame(Image2D& ref_img, Image2D& tar_img, const vector<POI2D>& poi_template,
                  FftccEngine& fftcc, ICGN2D1CudaMvp& icgn, vector<POI2D>& poi_queue,
                  double& fftcc_time, double& icgn_time,
                  const vector<POI2D>* prev_results = nullptr,
                  const vector<POI2D>* prev_prev_results = nullptr,
                  ICGN2D1CudaMvp* poi_fallback_icgn = nullptr,
                  bool allow_skip_fftcc = false,
                  bool enable_poi_fallback = true,
                  bool enable_temporal_extrapolation = false,
                  float temporal_seed_max_delta = numeric_limits<float>::infinity(),
                  float fallback_bad_ratio_threshold = numeric_limits<float>::infinity(),
                  FrameProcessStats* stats = nullptr)
{
  FrameProcessStats local_stats;
  if (!fftcc.hasPreparedReference(ref_img) || !icgn.hasPreparedReference(ref_img))
    throw logic_error("processFrame requires both engines to hold the same prepared reference image");
  if (ref_img.width != tar_img.width || ref_img.height != tar_img.height)
  {
    ErrorLine() << "reference and target image sizes must match";
    return false;
  }

  // Reference state was prepared once by the caller. Only the target changes
  // here; setImages would intentionally invalidate the reference caches too.
  fftcc.setTarImage(tar_img);
  fftcc.prepareTar();
  icgn.setTarImage(tar_img);
  icgn.prepareTar();

  bool use_skip_fftcc = allow_skip_fftcc && prev_results != nullptr
                        && prev_results->size() == poi_template.size();
  bool did_fallback = false;
  fftcc_time = 0.0;
  icgn_time = 0.0;

  if (use_skip_fftcc)
  {
    double seed_start = omp_get_wtime();
    seedPoiQueueFromPrevious(poi_template, *prev_results, prev_prev_results,
                             enable_temporal_extrapolation,
                             temporal_seed_max_delta, poi_queue,
                             &local_stats.temporal_seed_count);
    local_stats.seed_sec += omp_get_wtime() - seed_start;
    local_stats.temporal_seed_used = local_stats.temporal_seed_count > 0;

    double icgn_start = omp_get_wtime();
    icgn.compute(poi_queue);
    icgn_time += omp_get_wtime() - icgn_start;

    size_t bad_count = 0;
    vector<size_t> fallback_indices;
    if (enable_poi_fallback)
    {
      fallback_indices.reserve(poi_queue.size());
    }
    double scan_start = omp_get_wtime();
    const long long scan_n = static_cast<long long>(poi_queue.size());
    vector<vector<size_t>> tls_fallback(omp_get_max_threads());
#pragma omp parallel for reduction(+ : bad_count) schedule(static)
    for (long long i = 0; i < scan_n; ++i)
    {
      const size_t idx = static_cast<size_t>(i);
      const bool current_is_bad =
          isBadPoiForFallback(poi_queue[idx], kSkipFftccZnccThreshold);
      if (current_is_bad)
      {
        ++bad_count;
      }
      if (enable_poi_fallback)
      {
        const POI2D* previous = prev_results != nullptr && idx < prev_results->size()
                                  ? &(*prev_results)[idx]
                                  : nullptr;
        if (isPoiLevelFallbackCandidate(poi_queue[idx], previous, current_is_bad))
        {
          tls_fallback[omp_get_thread_num()].push_back(idx);
        }
      }
    }
    for (const auto& tls : tls_fallback)
    {
      fallback_indices.insert(fallback_indices.end(), tls.begin(), tls.end());
    }
    // ascending under schedule(static); sort removes any schedule dependence
    sort(fallback_indices.begin(), fallback_indices.end());
    local_stats.scan_sec += omp_get_wtime() - scan_start;
    float current_bad_ratio = computeBadZnccRatioFromCount(bad_count, poi_queue);
    local_stats.bad_ratio = current_bad_ratio;
    local_stats.pre_repair_bad_ratio = current_bad_ratio;
    local_stats.frame_fallback_trigger_bad_ratio = current_bad_ratio;

    if (current_bad_ratio > fallback_bad_ratio_threshold)
    {
      did_fallback = true;
      local_stats.frame_fallback_used = true;
      // review-2026-10-03 S1-1: the from-scratch pass used to overwrite the
      // whole skip attempt. Near the FFTCC capture limit that result is far
      // worse than the attempt it replaces, and it seeds the next frame, so the
      // fallback fired again on every later frame. Keep the attempt and merge.
      const vector<POI2D> skip_results = poi_queue;
      poi_queue = poi_template;

      double fftcc_start = omp_get_wtime();
      fftcc.compute(poi_queue);
      double fftcc_end = omp_get_wtime();

      double fallback_icgn_start = omp_get_wtime();
      icgn.compute(poi_queue);
      double fallback_icgn_end = omp_get_wtime();

      fftcc_time += fftcc_end - fftcc_start;
      icgn_time += fallback_icgn_end - fallback_icgn_start;

      // Per POI the skip attempt stays unless shouldTakeFrameFallbackResult()
      // prefers the from-scratch result. Each index is written once from its
      // own inputs, so the result does not depend on the thread count.
      const long long merge_n = static_cast<long long>(poi_queue.size());
#pragma omp parallel for schedule(static)
      for (long long i = 0; i < merge_n; ++i)
      {
        const size_t idx = static_cast<size_t>(i);
        if (!shouldTakeFrameFallbackResult(skip_results[idx], poi_queue[idx]))
        {
          poi_queue[idx] = skip_results[idx];
        }
      }

      local_stats.bad_ratio = computeBadZnccRatio(poi_queue, kSkipFftccZnccThreshold);
    }
    // review-2026-09-16 F1: with no candidates runPoiLevelFallback returns 0
    // without touching poi_queue, so bad_ratio would be reassigned the value it
    // already holds. Skipping the branch leaves CSV/raw output identical while
    // dropping one adoptPreparedTar per candidate-free frame; in CPU-only builds
    // that call rebuilds the repair instance's whole B-spline table.
    else if (enable_poi_fallback && !fallback_indices.empty())
    {
      ICGN2D1CudaMvp* repair_icgn = poi_fallback_icgn != nullptr ? poi_fallback_icgn : &icgn;
      if (poi_fallback_icgn != nullptr)
      {
        double adopt_prepared_tar_start = omp_get_wtime();
        poi_fallback_icgn->adoptPreparedTar(icgn);
        local_stats.poi_fallback_timing.adopt_prepared_tar_calls += 1;
        local_stats.poi_fallback_timing.adopt_prepared_tar_sec +=
            omp_get_wtime() - adopt_prepared_tar_start;
      }
      const size_t recovered = runPoiLevelFallback(poi_template, fallback_indices, fftcc,
                                                   icgn, *repair_icgn, poi_queue, fftcc_time,
                                                   icgn_time, local_stats);
      if (recovered > bad_count)
      {
        ErrorLine() << "POI fallback recovered count exceeds initial bad count";
        return false;
      }
      // Candidates are a subset of the initially bad POIs, repair mutates only
      // candidate indices, and recovered counts exactly replacements that are no
      // longer bad. Therefore the remaining bad count is exact in integer space.
      local_stats.bad_ratio =
          computeBadZnccRatioFromCount(bad_count - recovered, poi_queue);
    }
  }
  else
  {
    poi_queue = poi_template;

    double fftcc_start = omp_get_wtime();
    fftcc.compute(poi_queue);
    double fftcc_end = omp_get_wtime();

    double icgn_start = omp_get_wtime();
    icgn.compute(poi_queue);
    double icgn_end = omp_get_wtime();

    fftcc_time = fftcc_end - fftcc_start;
    icgn_time = icgn_end - icgn_start;

    local_stats.bad_ratio = computeBadZnccRatio(poi_queue, kSkipFftccZnccThreshold);
    local_stats.pre_repair_bad_ratio = local_stats.bad_ratio;
    local_stats.frame_fallback_trigger_bad_ratio = local_stats.bad_ratio;
  }

  if (stats != nullptr)
  {
    local_stats.frame_fallback_used = did_fallback;
    *stats = local_stats;
  }
  return true;
}

bool parseCommonOption(const vector<string>& args, size_t& i, CliOptions& options, string& error)
{
  const string& arg = args[i];
  auto requireValue = [&](const char* option_name) -> const string* {
    if (i + 1 >= args.size())
    {
      error = string("missing value for ") + option_name;
      return nullptr;
    }
    return &args[++i];
  };

  if (arg == "--gpu-device")
  {
    const string* value = requireValue("--gpu-device");
    if (value == nullptr)
    {
      return false;
    }
    return parseIntStrict(*value, "--gpu-device", 0, numeric_limits<int>::max(),
                          options.gpu_device, error);
  }
  if (arg == "--max-iter")
  {
    const string* value = requireValue("--max-iter");
    if (value == nullptr)
    {
      return false;
    }
    return parseIntStrict(*value, "--max-iter", 1, numeric_limits<int>::max(),
                          options.max_iteration, error);
  }
  if (arg == "--subset-size")
  {
    const string* value = requireValue("--subset-size");
    if (value == nullptr)
    {
      return false;
    }
    int size = 0;
    if (!parseIntStrict(*value, "--subset-size", 3, numeric_limits<int>::max(), size, error))
    {
      return false;
    }
    if (size % 2 == 0)
    {
      error = "--subset-size must be an odd subset SIZE in pixels; radius = (size - 1) / 2";
      return false;
    }
    options.subset_size = size;
    return true;
  }
  if (arg == "--conv-threshold")
  {
    const string* value = requireValue("--conv-threshold");
    if (value == nullptr)
    {
      return false;
    }
    return parsePositiveFloatStrict(*value, "--conv-threshold", options.conv_threshold, error);
  }
  if (arg == "--kernel")
  {
    const string* value = requireValue("--kernel");
    if (value == nullptr)
    {
      return false;
    }
    return parseIterationKernelModeStrict(*value, options.kernel_mode, error);
  }
  if (arg == "--fftcc")
  {
    const string* value = requireValue("--fftcc");
    if (value == nullptr)
    {
      return false;
    }
    return parseFftccBackendStrict(*value, options.fftcc_backend, error);
  }
  if (arg == "--threads")
  {
    const string* value = requireValue("--threads");
    if (value == nullptr)
    {
      return false;
    }
    return parseIntStrict(*value, "--threads", 1, numeric_limits<int>::max(),
                          options.num_threads, error);
  }
  if (arg == "--line")
  {
    const string* value = requireValue("--line");
    if (value == nullptr)
    {
      return false;
    }
    return parseLineYsStrict(*value, options.line_ys, error);
  }
  if (arg == "--durability")
  {
    const string* value = requireValue("--durability");
    if (value == nullptr)
    {
      return false;
    }
    if (!opencorr_cli_io::parseDurability(*value, options.durability))
    {
      error = "--durability must be rename or fsync, got '" + *value + "'";
      return false;
    }
    return true;
  }
  if (arg == "--repair")
  {
    options.repair = true;
    return true;
  }
  if (arg == "--no-repair")
  {
    options.repair = false;
    return true;
  }
  if (arg == "--quiet")
  {
    options.quiet = true;
    return true;
  }
  if (arg == "--dual-gpu")
  {
    options.dual_gpu = true;
    return true;
  }
  if (arg == "--skip-fftcc")
  {
    options.skip_fftcc = true;
    return true;
  }
  if (arg == "--no-skip-fftcc")
  {
    options.skip_fftcc = false;
    return true;
  }
  if (arg == "--poi-fallback")
  {
    options.poi_fallback = true;
    return true;
  }
  if (arg == "--no-poi-fallback")
  {
    options.poi_fallback = false;
    return true;
  }
  if (arg == "--temporal-extrapolation")
  {
    options.temporal_extrapolation = true;
    return true;
  }
  if (arg == "--no-temporal-extrapolation")
  {
    options.temporal_extrapolation = false;
    return true;
  }
  if (arg == "--output-format")
  {
    const string* value = requireValue("--output-format");
    if (value == nullptr)
    {
      return false;
    }
    return parseOutputFormatStrict(*value, options.output_format, error);
  }
  if (arg == "--input-depth")
  {
    const string* value = requireValue("--input-depth");
    if (value == nullptr)
    {
      return false;
    }
    return parseInputDepthStrict(*value, options.input_options.depth, error);
  }
  if (arg == "--input-align")
  {
    const string* value = requireValue("--input-align");
    if (value == nullptr)
    {
      return false;
    }
    return parseInputAlignmentStrict(*value, options.input_options.alignment, error);
  }
  if (arg == "--input-normalize")
  {
    const string* value = requireValue("--input-normalize");
    if (value == nullptr)
    {
      return false;
    }
    return parseInputNormalizeStrict(*value, options.input_options.normalize, error);
  }
  if (arg == "--diagnostics")
  {
    options.diagnostics_enabled = true;
    return true;
  }
  if (arg == "--diagnostics-dir")
  {
    const string* value = requireValue("--diagnostics-dir");
    if (value == nullptr)
    {
      return false;
    }
    if (value->empty())
    {
      error = "--diagnostics-dir must not be empty";
      return false;
    }
    options.diagnostics_enabled = true;
    options.diagnostics_dir = *value;
    return true;
  }
  if (arg == "--binary-output")
  {
    options.output_format = OutputFormat::RawBinary;
    return true;
  }
  if (arg == "--preload")
  {
    options.preload_enabled = true;
    return true;
  }
  if (arg == "--no-preload")
  {
    options.preload_enabled = false;
    return true;
  }
  if (arg == "--decode-threads")
  {
    const string* value = requireValue("--decode-threads");
    if (value == nullptr)
    {
      return false;
    }
    return parseIntStrict(*value, "--decode-threads", 1, numeric_limits<int>::max(),
                          options.decode_threads, error);
  }
  if (arg == "--output-roi")
  {
    const string* value = requireValue("--output-roi");
    if (value == nullptr)
    {
      return false;
    }
    return parseOutputRoiStrict(*value, options.output_filter, error);
  }
  if (arg == "--output-step")
  {
    const string* value = requireValue("--output-step");
    if (value == nullptr)
    {
      return false;
    }
    return parseIntStrict(*value, "--output-step", 1, numeric_limits<int>::max(),
                          options.output_filter.step, error);
  }
  if (!arg.empty() && arg[0] == '-')
  {
    error = "unknown option: " + arg;
  }
  return false;
}

bool loadTarList(const string& tar_list_path, vector<MultiFrameEntry>& entries)
{
  ifstream tar_list(tar_list_path);
  if (!tar_list.is_open())
  {
    ErrorLine() << "cannot open tar list " << tar_list_path;
    return false;
  }

  string line;
  while (getline(tar_list, line))
  {
    // Strip a trailing CR (CRLF files) and any trailing whitespace.
    while (!line.empty()
           && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
    {
      line.pop_back();
    }
    if (line.empty())
    {
      continue;
    }

    // Format is "<tar_path> <output_name>". Split on the LAST whitespace run so
    // that tar_path may itself contain spaces (camera exports often do); a tab,
    // if present, wins as the delimiter. output_name never contains whitespace.
    size_t sep = line.rfind('\t');
    if (sep == string::npos)
    {
      sep = line.find_last_of(" \t");
    }
    if (sep == string::npos)
    {
      ErrorLine() << "invalid tar list line (need '<tar_path> <output_name>'): "
           << line;
      return false;
    }

    MultiFrameEntry entry;
    entry.tar_path = line.substr(0, sep);
    entry.output_name = line.substr(sep + 1);
    while (!entry.tar_path.empty()
           && (entry.tar_path.back() == ' ' || entry.tar_path.back() == '\t'))
    {
      entry.tar_path.pop_back();
    }
    size_t name_start = entry.output_name.find_first_not_of(" \t");
    entry.output_name = name_start == string::npos
                          ? string()
                          : entry.output_name.substr(name_start);
    if (entry.tar_path.empty() || entry.output_name.empty())
    {
      ErrorLine() << "invalid tar list line (need '<tar_path> <output_name>'): "
           << line;
      return false;
    }
    entries.push_back(entry);
  }
  return true;
}

int runSingleFrame(const vector<string>& args)
{
  g_report.setMode("single-frame");
  // Options and positionals are collected in one pass so that --subset-size can
  // replace the positional subset_radius. With no --subset-size the accepted
  // argument order is byte-for-byte the historical one.
  CliOptions options = defaultOptions();
  vector<string> positionals;
  string parse_error;
  for (size_t i = 0; i < args.size(); ++i)
  {
    parse_error.clear();
    if (parseCommonOption(args, i, options, parse_error))
    {
      continue;
    }
    if (!parse_error.empty())
    {
      ErrorLine() << parse_error;
      printUsage(cerr, false);
      return 1;
    }
    if (!args[i].empty() && args[i][0] == '-')
    {
      ErrorLine() << "unexpected argument: " << args[i];
      printUsage(cerr, false);
      return 1;
    }
    positionals.push_back(args[i]);
  }

  if (options.subset_size > 0 && positionals.size() == 5)
  {
    ErrorLine() << "--subset-size conflicts with the positional subset_radius; pass exactly one";
    printUsage(cerr, false);
    return 1;
  }
  if (positionals.size() != (options.subset_size > 0 ? 4u : 5u))
  {
    ErrorLine() << "single-frame mode takes "
                << (options.subset_size > 0
                        ? "<ref_image> <tar_image> <grid_space> <output_path> with --subset-size"
                        : "<ref_image> <tar_image> <subset_radius> <grid_space> <output_path>")
                << ", got " << positionals.size() << " positional argument(s)";
    printUsage(cerr, false);
    return 1;
  }

  string ref_path = positionals[0];
  string tar_path = positionals[1];
  int subset_radius = 0;
  int grid_space = 0;
  size_t positional = 2;
  bool parsed = true;
  if (options.subset_size > 0)
  {
    subset_radius = (options.subset_size - 1) / 2;
  }
  else
  {
    parsed = parseIntStrict(positionals[positional++], "subset_radius", 1,
                            numeric_limits<int>::max(), subset_radius, parse_error);
  }
  if (parsed)
  {
    parsed = parseIntStrict(positionals[positional++], "grid_space", 1,
                            numeric_limits<int>::max(), grid_space, parse_error);
  }
  if (!parsed)
  {
    ErrorLine() << parse_error;
    printUsage(cerr, false);
    return 1;
  }
  string output_path = positionals[positional];

  parse_error.clear();
  if (!validateInputOptions(options, parse_error))
  {
    ErrorLine() << parse_error;
    printUsage(cerr, false);
    return 1;
  }

  opencorr_cli_io::setDefaultDurability(options.durability);
  {
    const fs::path reported_output = makeSingleOutputPath(output_path, options.output_format);
    g_report.setOutput(reported_output.parent_path(), reported_output,
                       outputFormatName(options.output_format));
    g_report.setFrames(1, 0, 0);
  }
  Image2D ref_img = loadConfiguredImage(ref_path, options.input_options);
  Image2D tar_img = loadConfiguredImage(tar_path, options.input_options);
  InputDepthAudit input_audit = auditReferenceDepth(ref_path, ref_img, options);
  auditTargetDepth(input_audit, fs::path(tar_path).filename().string(), tar_path, tar_img.load_info);
  vector<POI2D> poi_template = buildPoiQueue(ref_img, subset_radius, grid_space, options.line_ys);
  if (poi_template.empty())
  {
    ErrorLine() << "no POIs generated";
    return 1;
  }

  omp_set_num_threads(options.num_threads);
  cerr << "Image size: " << ref_img.width << " x " << ref_img.height << endl;
  cerr << "Subset radius: " << subset_radius << ", grid space: " << grid_space << endl;
  cerr << "Threads: " << options.num_threads << ", GPU device: " << options.gpu_device;
  if (options.dual_gpu)
  {
    cerr << " (dual GPU: enabled on devices 0 and 1)";
  }
  cerr << endl;
  cerr << "Output format: " << outputFormatName(options.output_format) << endl;
  if (options.output_filter.active())
  {
    cerr << "Output filter (write-side crop):";
    if (options.output_filter.roi_enabled)
    {
      cerr << " roi=[" << options.output_filter.roi_x1 << "," << options.output_filter.roi_y1
           << ".." << options.output_filter.roi_x2 << "," << options.output_filter.roi_y2 << "]";
    }
    cerr << " step=" << options.output_filter.step << endl;
  }
  cerr << "ICGN iteration kernel: " << iterationKernelModeName(options.kernel_mode) << endl;
  cerr << "FFTCC backend: " << fftccBackendName(options.fftcc_backend) << endl;
  printInputModeSummary(options);
  printLoadedImageSummary("Reference input", ref_img.load_info);
  printLoadedImageSummary("Target input", tar_img.load_info);
  cerr << "Total POIs: " << poi_template.size() << endl;

  double total_start = omp_get_wtime();

  FftccEngine fftcc(options.fftcc_backend, subset_radius, options.num_threads,
                    options.gpu_device);
  fftcc.setImages(ref_img, tar_img);
  fftcc.prepareRef(poi_template);

  ICGN2D1CudaMvp icgn(subset_radius, subset_radius, options.conv_threshold,
                      options.max_iteration, options.gpu_device, options.dual_gpu);
  icgn.setIterationKernelMode(options.kernel_mode);
  icgn.setImages(ref_img, tar_img);
  icgn.prepareRef(poi_template);

	  double fftcc_time = 0.0;
	  double icgn_time = 0.0;
	  vector<POI2D> poi_queue;
	  FrameProcessStats process_stats;
	  if (!processFrame(ref_img, tar_img, poi_template, fftcc, icgn, poi_queue,
	                    fftcc_time, icgn_time, nullptr, nullptr, nullptr, false,
	                    options.poi_fallback, options.temporal_extrapolation,
	                    static_cast<float>(subset_radius),
	                    numeric_limits<float>::infinity(), &process_stats))
	  {
	    return 1;
  }
	  fs::path final_output_path = makeSingleOutputPath(output_path, options.output_format);
	  FrameQaRecord qa = computeFrameQa(poi_queue, process_stats.bad_ratio, false, 0.0f,
	                                    false, 0.0f, false, false, process_stats);
	  vector<DiagnosticsFrameRecord> diagnostics_frames;
	  vector<OutputFrameRecord> raw_frames;
	  if (options.output_format == OutputFormat::Csv)
	  {
	    if (!writeCsv(final_output_path.string(), poi_queue, options.output_filter))
	    {
      return 1;
    }
	  }
	  else
	  {
	    AsyncRawBinaryWriter raw_writer;
	    OutputFrameRecord frame;
	    frame.tar_path = tar_path;
	    frame.output_name = fs::path(output_path).filename().string();
	    frame.file_name = final_output_path.filename().string();
	    frame.qa = qa;
	    double raw_wait_time = 0.0;
	    double raw_enqueue_time = 0.0;
	    if (!raw_writer.submit(final_output_path.string(), frame,
	                           make_shared<vector<POI2D>>(poi_queue), options.output_filter,
	                           raw_wait_time, raw_enqueue_time))
	    {
	      return 1;
	    }
	    if (!raw_writer.finish())
	    {
	      return 1;
	    }
	    raw_frames = raw_writer.completedFrames();
	    fs::path manifest_path = final_output_path;
	    manifest_path += ".manifest.json";
	    if (!writeRawManifest(manifest_path, ref_path, "", final_output_path.parent_path().string(),
	                          subset_radius, grid_space, options, raw_frames, input_audit))
	    {
	      return 1;
	    }
	  }
	  if (options.diagnostics_enabled)
	  {
	    fs::path diagnostics_dir = options.diagnostics_dir.empty()
	                                 ? defaultDiagnosticsDirForSingleFrame(final_output_path)
	                                 : fs::path(options.diagnostics_dir);
	    fs::create_directories(diagnostics_dir);
	    fs::path diagnostics_path = diagnosticsPathForFrame(diagnostics_dir, final_output_path);
	    if (!writeDiagnosticsCsv(diagnostics_path, poi_queue))
	    {
	      return 1;
	    }
	    diagnostics_frames.push_back({tar_path, fs::path(output_path).filename().string(),
	                                  diagnostics_path.filename().string(), poi_queue.size()});
	    if (!writeDiagnosticsManifest(diagnostics_dir / "manifest.opencorr-diagnostics.json",
	                                  ref_path, "", options, diagnostics_frames))
	    {
	      return 1;
	    }
	  }
	  double total_end = omp_get_wtime();

  cerr << "FFTCC: " << fftcc_time << " sec" << endl;
  cerr << icgnExecutionLabel() << " MVP: " << icgn_time << " sec" << endl;
  printPrepareTarStats(icgn.profileStats());
  cerr << "Results saved to " << final_output_path.string() << endl;
  cerr << "Total time: " << (total_end - total_start) << " sec" << endl;
  g_report.setFrames(1, 1, 0);
  return 0;
}

Json inputIdentity(const string& name)
{
  const auto path = fs::canonical(name);
  requireUtf8("input path", path.string());
  return {{"path", path.string()}, {"content", opencorr_cli_io::fingerprint(path)}};
}

void validateOutputName(const string& name)
{
  const fs::path path(name);
  if (name.empty() || path.filename() != path || name == "." || name == ".."
      || name.rfind(".opencorr-", 0) == 0
      || std::any_of(name.begin(), name.end(), [](unsigned char c) { return c < 32 || c == 127; }))
    throw runtime_error("output names must be unique plain filenames: " + name);
  requireUtf8("output name", name);
}

Json runContract(const fs::path& executable, const string& ref_path,
                 const vector<MultiFrameEntry>& entries, int radius, int step,
                 const CliOptions& options)
{
  if (!options.diagnostics_dir.empty())
    requireUtf8("diagnostics directory",
                fs::weakly_canonical(fs::absolute(options.diagnostics_dir)).string());
  Json config{{"schema_version", 1}, {"git_hash", OPENCORR_GPU_MVP_GIT_HASH},
              {"build_date", OPENCORR_GPU_MVP_BUILD_DATE},
              {"binary", opencorr_cli_io::executableIdentity(executable)},
              {"reference", inputIdentity(ref_path)}};
  // Only what decides the output bytes. --gpu-device, --threads, --preload and
  // --decode-threads schedule the same computation (2026-10-05: each of them was
  // varied against the 60-frame production anchor and the output tree stayed
  // byte-identical), so, like --durability, they are recorded in the manifest's
  // engine block but are not part of the contract: changing one of them is not
  // a different run. opencorr_cli_io::kUncomparedLegacyOptions keeps contracts
  // that still list them resumable.
  config["options"] = {{"radius", radius}, {"step", step}, {"dual_gpu", options.dual_gpu},
      {"kernel", iterationKernelModeName(options.kernel_mode)},
      {"fftcc", fftccBackendName(options.fftcc_backend)}, {"max_iteration", options.max_iteration},
      {"convergence", options.conv_threshold}, {"skip_fftcc", options.skip_fftcc},
      {"poi_fallback", options.poi_fallback}, {"temporal_extrapolation", options.temporal_extrapolation},
      {"format", outputFormatName(options.output_format)}, {"line_ys", options.line_ys},
      {"input_depth", imageInputDepthModeName(options.input_options.depth)},
      {"input_align", imageInputAlignmentName(options.input_options.alignment)},
      {"input_normalize", imageInputNormalizeName(options.input_options.normalize)},
      {"output_roi_enabled", options.output_filter.roi_enabled},
      {"output_roi", {options.output_filter.roi_x1, options.output_filter.roi_y1,
                       options.output_filter.roi_x2, options.output_filter.roi_y2}},
      {"output_step", options.output_filter.step}, {"diagnostics", options.diagnostics_enabled},
      {"diagnostics_dir", options.diagnostics_dir.empty() ? "default"
          : fs::weakly_canonical(fs::absolute(options.diagnostics_dir)).string()}};
  config["targets"] = Json::array();
  std::map<string, Json> fingerprints;
  std::set<string> names;
  for (const auto& entry : entries)
  {
    validateOutputName(entry.output_name);
    const string output = makeOutputPath("", entry.output_name, options.output_format).string();
    if (!names.insert(output).second) throw runtime_error("duplicate output filename: " + output);
    const string path = fs::canonical(entry.tar_path).string();
    auto found = fingerprints.find(path);
    if (found == fingerprints.end()) found = fingerprints.emplace(path, inputIdentity(path)).first;
    config["targets"].push_back({{"input", found->second}, {"output_name", entry.output_name}, {"file", output}});
  }
  return config;
}

int runMultiFrame(const vector<string>& args, const fs::path& executable)
{
  g_report.setMode("multi-frame");
  CliOptions options = defaultOptions();
  string ref_path;
  string tar_list_path;
  string output_dir;
  vector<string> positionals;

  for (size_t i = 1; i < args.size(); ++i)
  {
    const string& arg = args[i];
    if (arg == "--ref" && i + 1 < args.size())
    {
      ref_path = args[++i];
    }
    else if (arg == "--tar-list" && i + 1 < args.size())
    {
      tar_list_path = args[++i];
    }
    else if (arg == "--output-dir" && i + 1 < args.size())
    {
      output_dir = args[++i];
    }
    else if (arg == "--ref" || arg == "--tar-list" || arg == "--output-dir")
    {
      ErrorLine() << "missing value for " << arg;
      printUsage(cerr, false);
      return 1;
    }
    else
    {
      string parse_error;
      if (parseCommonOption(args, i, options, parse_error))
      {
      }
      else if (!parse_error.empty())
      {
        ErrorLine() << parse_error;
        printUsage(cerr, false);
        return 1;
      }
      else
      {
        positionals.push_back(arg);
      }
    }
  }

  if (options.subset_size > 0 && positionals.size() == 2)
  {
    ErrorLine() << "--subset-size conflicts with the positional subset_radius; pass exactly one";
    printUsage(cerr, false);
    return 1;
  }
  if (ref_path.empty() || tar_list_path.empty() || output_dir.empty()
      || positionals.size() != (options.subset_size > 0 ? 1u : 2u))
  {
    ErrorLine() << "multi-frame mode needs --ref, --tar-list and --output-dir, each with a value, and "
                << (options.subset_size > 0 ? "<grid_space> with --subset-size"
                                            : "<subset_radius> <grid_space>")
                << " (missing:" << (ref_path.empty() ? " --ref" : "")
                << (tar_list_path.empty() ? " --tar-list" : "")
                << (output_dir.empty() ? " --output-dir" : "")
                << "; positional arguments given: " << positionals.size() << ")";
    printUsage(cerr, false);
    return 1;
  }

  string parse_error;
  if (!validateInputOptions(options, parse_error))
  {
    ErrorLine() << parse_error;
    printUsage(cerr, false);
    return 1;
  }

  opencorr_cli_io::setDefaultDurability(options.durability);
  g_report.setOutput(output_dir, fs::path(), outputFormatName(options.output_format));
  requireUtf8("reference path", ref_path);
  requireUtf8("target list path", tar_list_path);
  requireUtf8("output directory", fs::weakly_canonical(fs::absolute(output_dir)).string());

  int subset_radius = 0;
  int grid_space = 0;
  size_t positional = 0;
  bool parsed = true;
  if (options.subset_size > 0)
  {
    subset_radius = (options.subset_size - 1) / 2;
  }
  else
  {
    parsed = parseIntStrict(positionals[positional++], "subset_radius", 1,
                            numeric_limits<int>::max(), subset_radius, parse_error);
  }
  if (parsed)
  {
    parsed = parseIntStrict(positionals[positional], "grid_space", 1,
                            numeric_limits<int>::max(), grid_space, parse_error);
  }
  if (!parsed)
  {
    ErrorLine() << parse_error;
    printUsage(cerr, false);
    return 1;
  }
  vector<MultiFrameEntry> entries;
  if (!loadTarList(tar_list_path, entries))
  {
    return 1;
  }
  if (entries.empty())
  {
    ErrorLine() << "tar list is empty";
    return 1;
  }
  g_report.setFrames(entries.size(), 0, 0);

  fs::create_directories(output_dir);

  Image2D ref_img = loadConfiguredImage(ref_path, options.input_options);
  InputDepthAudit input_audit = auditReferenceDepth(ref_path, ref_img, options);
  vector<POI2D> poi_template = buildPoiQueue(ref_img, subset_radius, grid_space, options.line_ys);
  if (poi_template.empty())
  {
    ErrorLine() << "no POIs generated";
    return 1;
  }

  double total_start = omp_get_wtime();
  fs::path diagnostics_dir;
  if (options.diagnostics_enabled)
    diagnostics_dir = options.diagnostics_dir.empty() ? fs::path(output_dir) / "diagnostics" : fs::path(options.diagnostics_dir);
  const size_t output_rows = static_cast<size_t>(std::count_if(poi_template.begin(), poi_template.end(),
      [&](const auto& poi) { return options.output_filter.keep(static_cast<int>(poi.x), static_cast<int>(poi.y)); }));
  vector<opencorr_cli_io::OutputSpec> output_specs;
  for (const auto& entry : entries)
  {
    const auto path = makeOutputPath(output_dir, entry.output_name, options.output_format);
    output_specs.push_back({"frame/" + path.filename().string(), path, output_rows,
                           options.output_format == OutputFormat::RawBinary ? kRawBinaryRecordBytes : 0});
    if (options.diagnostics_enabled)
    {
      const auto diagnostic = diagnosticsPathForFrame(diagnostics_dir, path);
      output_specs.push_back({"diagnostics/" + diagnostic.filename().string(), diagnostic, poi_template.size(), 0});
    }
  }
  vector<fs::path> metadata_paths;
  vector<OutputFrameRecord> prior_frames;
  if (options.output_format == OutputFormat::RawBinary)
  {
    metadata_paths.push_back(fs::path(output_dir) / "manifest.opencorr-raw.json");

  }
  if (options.diagnostics_enabled) metadata_paths.push_back(diagnostics_dir / "manifest.opencorr-diagnostics.json");
  OutputSession output_session(output_dir, runContract(executable, ref_path, entries, subset_radius, grid_space, options),
                               std::move(output_specs), metadata_paths, tar_list_path,
                               options.durability, options.repair);
  if (output_session.repairedFiles())
  {
    cerr << "Repair: quarantined " << output_session.repairedFiles() << " damaged file(s) in "
         << output_session.quarantineDirectory().string()
         << "; they are regenerated by replay" << endl;
  }
  if (options.output_format == OutputFormat::RawBinary)
  {
    string manifest_error;
    if (!readRawManifestFrames(fs::path(output_dir) / "manifest.opencorr-raw.json", prior_frames, manifest_error))
      throw runtime_error("invalid existing raw manifest: " + manifest_error);
    if (prior_frames.size() > entries.size()) throw runtime_error("existing manifest is not a prefix of this run");
    for (size_t i = 0; i < prior_frames.size(); ++i)
    {
      const auto& frame = prior_frames[i];
      if (frame.output_name != entries[i].output_name
          || frame.file_name != makeOutputPath("", entries[i].output_name, options.output_format).string()
          || fs::canonical(frame.tar_path) != fs::canonical(entries[i].tar_path)
          || frame.rows != output_rows)
        throw runtime_error("existing raw manifest does not describe the requested frames");
    }
  }
  if (options.diagnostics_enabled && fs::exists(diagnostics_dir / "manifest.opencorr-diagnostics.json"))
  {
    const auto document = opencorr_cli_io::readJson(diagnostics_dir / "manifest.opencorr-diagnostics.json");
    if (!document.is_object() || document.at("format") != "opencorr-gpu-diagnostics-csv-v1"
        || document.at("columns") != Json({"x","y","iteration","convergence","status","zncc"})
        || !document.at("frames").is_array() || document.at("frames").size() > entries.size())
      throw runtime_error("invalid diagnostics manifest schema");
    for (size_t i = 0; i < document.at("frames").size(); ++i)
    {
      const auto& frame = document.at("frames")[i];
      const auto path = makeOutputPath(output_dir, entries[i].output_name, options.output_format);
      if (frame.at("output_name") != entries[i].output_name
          || frame.at("file") != diagnosticsPathForFrame(diagnostics_dir, path).filename().string()
          || !frame.at("rows").is_number_unsigned() || frame.at("rows").get<size_t>() != poi_template.size())
        throw runtime_error("diagnostics manifest does not describe the requested frames");
    }
  }
  output_session.cleanupResidue();
  if (output_session.complete())
  {
    if (options.output_format == OutputFormat::RawBinary && prior_frames.size() != entries.size())
      throw runtime_error("completed manifest has missing frames");
    printInputModeSummary(options);
    printLoadedImageSummary("Reference input", ref_img.load_info);
    ostringstream verified_size;
    verified_size.setf(std::ios::fixed, std::ios::floatfield);
    verified_size << setprecision(1)
                  << (static_cast<double>(output_session.verifiedBytes()) / (1024.0 * 1024.0));
    cerr << "Verified complete run; no computation or output rewrite required ("
         << output_session.verifiedFiles() << " output files re-read and checked, "
         << verified_size.str() << " MiB, plus a full fingerprint of every input)" << endl;
    cerr << "Processed frames: 0, skipped: " << entries.size() << endl;
    g_report.setFrames(entries.size(), 0, entries.size());
    return 0;
  }
  output_session.begin();
  if (output_session.existingFiles())
    cerr << "Resume: " << output_session.existingFiles()
         << " verified files; replay from frame 1 to restore temporal state (no overwrite)" << endl;
  if (options.diagnostics_enabled) fs::create_directories(diagnostics_dir);

  omp_set_num_threads(options.num_threads);
  cerr << "Multi-frame mode: " << entries.size() << " target frames" << endl;
  cerr << "Durability: " << opencorr_cli_io::durabilityName(options.durability)
       << (options.repair ? ", repair: enabled" : "") << endl;
  cerr << "Image size: " << ref_img.width << " x " << ref_img.height << endl;
  cerr << "Subset radius: " << subset_radius << ", grid space: " << grid_space << endl;
  cerr << "Threads: " << options.num_threads << ", GPU device: " << options.gpu_device;
  if (options.dual_gpu)
  {
    cerr << " (dual GPU: enabled on devices 0 and 1)";
  }
  cerr << endl;
  cerr << "Output format: " << outputFormatName(options.output_format) << endl;
  if (options.output_filter.active())
  {
    cerr << "Output filter (write-side crop):";
    if (options.output_filter.roi_enabled)
    {
      cerr << " roi=[" << options.output_filter.roi_x1 << "," << options.output_filter.roi_y1
           << ".." << options.output_filter.roi_x2 << "," << options.output_filter.roi_y2 << "]";
    }
    cerr << " step=" << options.output_filter.step << endl;
  }
  cerr << "ICGN iteration kernel: " << iterationKernelModeName(options.kernel_mode) << endl;
  cerr << "FFTCC backend: " << fftccBackendName(options.fftcc_backend) << endl;
  printInputModeSummary(options);
  printLoadedImageSummary("Reference input", ref_img.load_info);
  cerr << "Image preload: " << (options.preload_enabled ? "enabled" : "disabled") << endl;
  cerr << "Skip FFTCC after first frame: " << (options.skip_fftcc ? "enabled" : "disabled")
       << " (fallback zncc<" << kSkipFftccZnccThreshold
       << ", bad_ratio>baseline+" << kSkipFftccBadRatioMargin << ")" << endl;
  cerr << "POI-level fallback: " << (options.poi_fallback ? "enabled" : "disabled")
       << " (runs only below the frame fallback threshold)" << endl;
  cerr << "Temporal u/v extrapolation seed: "
       << (options.temporal_extrapolation ? "enabled" : "disabled")
       << " (opt-in; invalid history falls back to previous-frame seed)" << endl;
  cerr << "Total POIs: " << poi_template.size() << endl;


  FftccEngine fftcc(options.fftcc_backend, subset_radius, options.num_threads,
                    options.gpu_device);
  fftcc.setImages(ref_img, ref_img);
  fftcc.prepareRef(poi_template);

  ICGN2D1CudaMvp icgn(subset_radius, subset_radius, options.conv_threshold,
                      options.max_iteration, options.gpu_device, options.dual_gpu);
  icgn.setIterationKernelMode(options.kernel_mode);
  icgn.setRefImage(ref_img);
  icgn.prepareRef(poi_template);

  unique_ptr<ICGN2D1CudaMvp> poi_fallback_icgn;
  if (options.poi_fallback)
  {
    poi_fallback_icgn = make_unique<ICGN2D1CudaMvp>(subset_radius, subset_radius,
                                                    options.conv_threshold,
                                                    options.max_iteration,
                                                    options.gpu_device,
                                                    options.dual_gpu);
    // The POI-fallback instance iterates with the same kernel as the main
    // instance so repaired POIs share the frame's numerics baseline.
    poi_fallback_icgn->setIterationKernelMode(options.kernel_mode);
    poi_fallback_icgn->setRefImage(ref_img);
    poi_fallback_icgn->prepareRef(poi_template);
  }
  PoiFallbackTimingStats total_poi_fallback_timing;

  const vector<MultiFrameEntry>& pending_entries = entries;
  const int skipped = 0;

  double total_fftcc = 0.0;
  double total_icgn = 0.0;
  double total_image_wait = 0.0;
  double total_seed = 0.0;
  double total_scan = 0.0;
  double total_qa = 0.0;
  double total_tar_move = 0.0;
	  double total_csv_wait = 0.0;
	  double total_csv_enqueue = 0.0;
	  double total_raw_wait = 0.0;
	  double total_raw_enqueue = 0.0;
	  int processed = 0;
	  int skip_fftcc_frames = 0;
	  int fallback_frames = 0;
	  int poi_fallback_frames = 0;
	  int poi_fallback_pois = 0;
	  int poi_fallback_recovered = 0;
	  int temporal_seed_frames = 0;
	  int temporal_seed_pois = 0;
	  // A buffer is writable only between acquire() and submit(). After submit it is
	  // published and remains immutable until writer/history references are released.
	  PoiBufferPool poi_buffer_pool;
	  PoiBuffer poi_queue = poi_buffer_pool.acquire();
	  PoiBuffer prev_results;
	  PoiBuffer prev_prev_results;
	  vector<OutputFrameRecord> output_frames;
	  vector<DiagnosticsFrameRecord> diagnostics_frames;
	  bool baseline_bad_ratio_ready = false;
	  float baseline_bad_ratio = 0.0f;

	  unique_ptr<ImagePreloader> preloader;
	  AsyncCsvWriter csv_writer(&output_session);
      unique_ptr<AsyncCsvWriter> diagnostics_writer;
      if (options.diagnostics_enabled) diagnostics_writer = make_unique<AsyncCsvWriter>(&output_session, true);
	  AsyncRawBinaryWriter raw_writer(&output_session);
	  if (options.preload_enabled)
	  {
	    preloader = make_unique<ImagePreloader>(options.input_options, options.decode_threads);
	    // Keep up to decode_threads frames in flight; with the default 1 this
	    // is exactly the legacy request-next-after-consume interleave.
	    const size_t warmup = std::min(static_cast<size_t>(options.decode_threads),
	                                   pending_entries.size());
	    for (size_t k = 0; k < warmup; ++k)
	    {
	      preloader->request(pending_entries[k].tar_path);
	    }
	  }

  for (size_t i = 0; i < pending_entries.size(); ++i)
  {
    const MultiFrameEntry& entry = pending_entries[i];
    fs::path output_path = makeOutputPath(output_dir, entry.output_name, options.output_format);

    double image_wait_start = omp_get_wtime();
    Image2D tar_img(1, 1);
    bool used_preload = false;
    double preload_io_time = 0.0;

    if (preloader)
    {
      try
      {
        PreloadedImage preloaded = preloader->waitAndConsume(entry.tar_path);
        preload_io_time = preloaded.load_time;
        if (preloaded.success && preloaded.image)
        {
          double tar_move_start = omp_get_wtime();
          tar_img = std::move(*preloaded.image);
          total_tar_move += omp_get_wtime() - tar_move_start;
          used_preload = true;
        }
        else if (!preloaded.error.empty())
        {
          WarningLine() << "preload failed for " << entry.tar_path
               << ", fallback to synchronous load: " << preloaded.error;
        }
      }
      catch (const exception& ex)
      {
        WarningLine() << "preload failed for " << entry.tar_path
             << ", fallback to synchronous load: " << ex.what();
      }
      catch (const string& ex)
      {
        WarningLine() << "preload failed for " << entry.tar_path
             << ", fallback to synchronous load: " << ex;
      }
    }

    if (!used_preload)
    {
      try
      {
        tar_img.loadWithOptions(entry.tar_path, options.input_options);
      }
      catch (const exception& ex)
      {
        ErrorLine() << "failed to load target image " << entry.tar_path
             << " after preload fallback: " << ex.what();
        return 1;
      }
      catch (const string& ex)
      {
        ErrorLine() << "failed to load target image " << entry.tar_path
             << " after preload fallback: " << ex;
        return 1;
      }
    }
    double image_wait_time = omp_get_wtime() - image_wait_start;
    auditTargetDepth(input_audit, entry.output_name, entry.tar_path, tar_img.load_info);

    if (preloader)
    {
      const size_t next = i + static_cast<size_t>(options.decode_threads);
      if (next < pending_entries.size())
      {
        preloader->request(pending_entries[next].tar_path);
      }
    }

    double fftcc_time = 0.0;
    double icgn_time = 0.0;
	    bool attempted_skip_fftcc = options.skip_fftcc
	                                 && baseline_bad_ratio_ready
	                                 && prev_results && !prev_results->empty();
	    FrameProcessStats process_stats;
	    float fallback_threshold = baseline_bad_ratio_ready
	                               ? baseline_bad_ratio + kSkipFftccBadRatioMargin
	                               : numeric_limits<float>::infinity();
	    if (!processFrame(ref_img, tar_img, poi_template, fftcc, icgn, *poi_queue,
	                      fftcc_time, icgn_time,
	                      attempted_skip_fftcc ? prev_results.get() : nullptr,
	                      (attempted_skip_fftcc && prev_prev_results)
	                        ? prev_prev_results.get() : nullptr,
	                      poi_fallback_icgn.get(),
	                      attempted_skip_fftcc, options.poi_fallback,
	                      options.temporal_extrapolation,
	                      static_cast<float>(subset_radius),
	                      fallback_threshold, &process_stats))
	    {
	      return 1;
	    }
	    accumulatePoiFallbackTiming(total_poi_fallback_timing,
	                                process_stats.poi_fallback_timing);
	    total_seed += process_stats.seed_sec;
	    total_scan += process_stats.scan_sec;
	    bool fallback_used = process_stats.frame_fallback_used;
	    float bad_ratio = process_stats.bad_ratio;
	    float fallback_trigger_bad_ratio = process_stats.frame_fallback_trigger_bad_ratio;

	    if (options.skip_fftcc && !baseline_bad_ratio_ready && !attempted_skip_fftcc)
    {
      baseline_bad_ratio = bad_ratio;
      baseline_bad_ratio_ready = true;
      fallback_threshold = baseline_bad_ratio + kSkipFftccBadRatioMargin;
      if (!options.quiet)
      {
        cerr << "  [BASELINE] skip-FFTCC guard baseline_bad_ratio=" << baseline_bad_ratio
             << ", margin=" << kSkipFftccBadRatioMargin
             << ", fallback_threshold=" << fallback_threshold
             << ", source=" << entry.output_name << endl;
      }
    }
	    bool fallback_threshold_ready = options.skip_fftcc && baseline_bad_ratio_ready;
	    double qa_start = omp_get_wtime();
	    FrameQaRecord qa = computeFrameQa(*poi_queue, bad_ratio, baseline_bad_ratio_ready,
	                                      baseline_bad_ratio, fallback_threshold_ready,
	                                      fallback_threshold, fallback_used,
	                                      attempted_skip_fftcc, process_stats);
	    total_qa += omp_get_wtime() - qa_start;

    if (attempted_skip_fftcc)
    {
      ++skip_fftcc_frames;
    }
	    if (fallback_used)
	    {
	      ++fallback_frames;
	      if (!options.quiet) cerr << "  [FALLBACK] " << entry.output_name
           << " reason=bad_ratio > baseline + margin"
           << " bad_ratio=" << fallback_trigger_bad_ratio
           << ", baseline_bad_ratio=" << baseline_bad_ratio
           << ", margin=" << kSkipFftccBadRatioMargin
           << ", fallback_threshold=" << fallback_threshold
	           << ", post_fallback_bad_ratio=" << bad_ratio
	           << ", re-ran FFTCC + ICGN" << endl;
	    }
	    if (process_stats.poi_fallback_used)
	    {
	      ++poi_fallback_frames;
	      poi_fallback_pois += process_stats.poi_fallback_count;
	      poi_fallback_recovered += process_stats.poi_fallback_recovered;
	      if (!options.quiet) cerr << "  [POI-FALLBACK] " << entry.output_name
	           << " count=" << process_stats.poi_fallback_count
	           << ", recovered=" << process_stats.poi_fallback_recovered
	           << ", pre_bad_ratio=" << process_stats.pre_repair_bad_ratio
	           << ", post_bad_ratio=" << bad_ratio
	           << ", frame_threshold=" << fallback_threshold << endl;
	    }
	    if (process_stats.temporal_seed_used)
	    {
	      ++temporal_seed_frames;
	      temporal_seed_pois += process_stats.temporal_seed_count;
	    }

    double csv_wait_time = 0.0;
    double csv_enqueue_time = 0.0;
    double raw_wait_time = 0.0;
    double raw_enqueue_time = 0.0;
    if (options.output_format == OutputFormat::Csv)
    {
      if (!csv_writer.submit(output_path.string(), poi_queue, options.output_filter,
                             csv_wait_time, csv_enqueue_time))
      {
        return 1;
      }
    }
    else
    {
      OutputFrameRecord frame;
      frame.tar_path = entry.tar_path;
      frame.output_name = entry.output_name;
      frame.file_name = output_path.filename().string();
      frame.qa = qa;
      if (!raw_writer.submit(output_path.string(), frame, poi_queue, options.output_filter,
                             raw_wait_time, raw_enqueue_time))
      {
        return 1;
      }
    }
    if (options.diagnostics_enabled)
    {
      fs::path diagnostics_path = diagnosticsPathForFrame(diagnostics_dir, output_path);
      double diagnostics_wait = 0.0, diagnostics_enqueue = 0.0;
      if (!diagnostics_writer->submit(diagnostics_path.string(), poi_queue, OutputFilter{}, diagnostics_wait, diagnostics_enqueue))
      {
        return 1;
      }
      diagnostics_frames.push_back({entry.tar_path, entry.output_name,
                                    diagnostics_path.filename().string(),
                                    poi_queue->size()});
    }

    total_fftcc += fftcc_time;
    total_icgn += icgn_time;
    total_image_wait += image_wait_time;
    total_csv_wait += csv_wait_time;
    total_csv_enqueue += csv_enqueue_time;
    total_raw_wait += raw_wait_time;
	    total_raw_enqueue += raw_enqueue_time;
	    ++processed;
	    prev_prev_results = std::move(prev_results);
	    prev_results = std::move(poi_queue);
	    poi_queue = poi_buffer_pool.acquire();
	    g_report.frameProcessed();
	    // --quiet: the per-frame facts below stay available as the closing totals,
	    // the status line and, for raw output, the manifest's per-frame QA.
	    if (!options.quiet)
	    {
	    cerr << "[" << (processed + skipped) << "/" << entries.size() << "] "
         << "load-visible " << image_wait_time << " sec"
         << " (preload I/O " << preload_io_time << " sec), "
         << "FFTCC " << fftcc_time << " sec, " << icgnExecutionLabel() << " " << icgn_time
         << " sec, bad_ratio=" << bad_ratio;
    if (options.skip_fftcc && baseline_bad_ratio_ready)
    {
      cerr << ", baseline_bad_ratio=" << baseline_bad_ratio
           << ", fallback_threshold=" << fallback_threshold;
    }
    else if (options.skip_fftcc)
    {
      cerr << ", baseline_bad_ratio=pending, fallback_threshold=pending";
    }
    cerr
         << ", valid_ratio=" << qa.valid_ratio
	         << ", iteration_mean=" << qa.iteration_mean
	         << ", iteration_max=" << qa.iteration_max
	         << ", poi_fallback_count=" << process_stats.poi_fallback_count
	         << ", poi_fallback_recovered=" << process_stats.poi_fallback_recovered
	         << ", temporal_seed_count=" << process_stats.temporal_seed_count
	         << ", CSV-wait " << csv_wait_time
         << " sec, CSV-enqueue " << csv_enqueue_time
         << " sec, raw-wait " << raw_wait_time
         << " sec, raw-enqueue " << raw_enqueue_time
         << " sec, skip-fftcc=" << (attempted_skip_fftcc ? "yes" : "no")
         << ", fallback " << (fallback_used ? "yes" : "no")
         << " -> " << output_path.string() << endl;
	    }
	  }
  reportTargetDepthTotal(input_audit, entries.size());

  if (preloader)
  {
    preloader->stop();
  }
  if (diagnostics_writer && !diagnostics_writer->finish()) return 1;
  if (options.output_format == OutputFormat::Csv && !csv_writer.finish())
  {
    return 1;
  }
  if (options.output_format == OutputFormat::RawBinary)
  {
    if (!raw_writer.finish())
    {
      return 1;
    }
    output_frames = raw_writer.completedFrames();
  }
  if (options.diagnostics_enabled)
  {
    if (!writeDiagnosticsManifest(diagnostics_dir / "manifest.opencorr-diagnostics.json",
                                  ref_path, tar_list_path, options, diagnostics_frames))
    {
      return 1;
    }
  }
  if (options.output_format == OutputFormat::RawBinary)
  {
    fs::path manifest_path = fs::path(output_dir) / "manifest.opencorr-raw.json";
    if (!writeRawManifest(manifest_path, ref_path, tar_list_path, output_dir,
                          subset_radius, grid_space, options, output_frames, input_audit))
    {
      return 1;
    }
  }

  output_session.finish();
  try
  {
    // The completion record is already durable here, so this second sweep is
    // pure housekeeping: it must never turn a finished run into a failure.
    output_session.cleanupResidue();
  }
  catch (const std::exception& ex)
  {
    WarningLine() << "residue cleanup after completion failed: " << ex.what();
  }
  double total_end = omp_get_wtime();
  cerr << "Processed frames: " << processed << ", skipped: " << skipped << endl;
  cerr << "Frames with skipped FFTCC: " << skip_fftcc_frames << endl;
  cerr << "Frames with fallback FFTCC: " << fallback_frames << endl;
  cerr << "Frames with POI-level fallback: " << poi_fallback_frames << endl;
  cerr << "POIs submitted to POI-level fallback: " << poi_fallback_pois << endl;
  cerr << "POIs recovered by POI-level fallback: " << poi_fallback_recovered << endl;
  cerr << "Frames with temporal extrapolation seed: " << temporal_seed_frames << endl;
  cerr << "POIs seeded by temporal extrapolation: " << temporal_seed_pois << endl;
  cerr << "Accumulated FFTCC: " << total_fftcc << " sec" << endl;
  cerr << "Accumulated " << icgnExecutionLabel() << ": " << total_icgn << " sec" << endl;
  cerr << "Accumulated serial seed: " << total_seed << " sec" << endl;
  cerr << "Accumulated serial bad-scan: " << total_scan << " sec" << endl;
  cerr << "Accumulated serial frame QA: " << total_qa << " sec" << endl;
  cerr << "Accumulated tar Image2D move: " << total_tar_move << " sec" << endl;
  cerr << "Accumulated visible image wait: " << total_image_wait << " sec" << endl;
  cerr << "Accumulated visible CSV wait: " << total_csv_wait << " sec" << endl;
  cerr << "Accumulated visible CSV enqueue: " << total_csv_enqueue << " sec" << endl;
  cerr << "Accumulated visible raw wait: " << total_raw_wait << " sec" << endl;
  cerr << "Accumulated visible raw enqueue: " << total_raw_enqueue << " sec" << endl;
  cerr << "Accumulated background CSV format: " << csv_writer.totalFormatTime() << " sec" << endl;
  cerr << "Accumulated async CSV write: " << csv_writer.totalWriteTime() << " sec" << endl;
  cerr << "Accumulated raw binary format: " << raw_writer.totalFormatTime() << " sec" << endl;
  cerr << "Accumulated raw binary write: " << raw_writer.totalWriteTime() << " sec" << endl;
  {
    const auto publish_profile = opencorr_cli_io::publishProfile();
    cerr << "Accumulated publish resolve: " << publish_profile.resolve << " sec" << endl;
    cerr << "Accumulated publish digest: " << publish_profile.digest << " sec" << endl;
    cerr << "Accumulated publish intent: " << publish_profile.intent << " sec" << endl;
    cerr << "Accumulated publish payload create: " << publish_profile.payload_open << " sec" << endl;
    cerr << "Accumulated publish payload write: " << publish_profile.payload_write << " sec" << endl;
    cerr << "Accumulated publish payload link: " << publish_profile.payload_link << " sec" << endl;
    cerr << "Published payloads: " << publish_profile.frames << endl;
  }
  printPoiFallbackTiming(total_poi_fallback_timing);
  printPrepareTarStats(icgn.profileStats());
  cerr << "Total wall time: " << (total_end - total_start) << " sec" << endl;
  g_report.setFrames(entries.size(), static_cast<size_t>(processed), static_cast<size_t>(skipped));
  g_report.setTotals({{"skip_fftcc_frames", skip_fftcc_frames},
                      {"fallback_frames", fallback_frames},
                      {"poi_fallback_frames", poi_fallback_frames},
                      {"poi_fallback_pois", poi_fallback_pois},
                      {"poi_fallback_recovered", poi_fallback_recovered},
                      {"temporal_seed_frames", temporal_seed_frames},
                      {"temporal_seed_pois", temporal_seed_pois},
                      {"resumed_from_existing_files", output_session.existingFiles()}});
  return 0;
}

// How long the closing line waits for a non-blocking stdout that has no room
// (a full pipe whose reader is busy). A reader that never makes room must not
// keep a finished process alive, so the wait is bounded.
constexpr int kStdoutLineWaitMs = 5000;

// One line on stdout, written with write(2) so that a reader which has gone
// away cannot turn a finished run into a SIGPIPE death or a failure.
void writeStdoutLine(const OrderedJson& document)
{
  cout.flush();
  const string line = document.dump(-1, ' ', false, OrderedJson::error_handler_t::replace) + "\n";
  std::signal(SIGPIPE, SIG_IGN);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kStdoutLineWaitMs);
  bool reported_ready = false;
  size_t offset = 0;
  while (offset < line.size())
  {
    const ssize_t count = ::write(STDOUT_FILENO, line.data() + offset, line.size() - offset);
    if (count > 0)
    {
      offset += static_cast<size_t>(count);
      reported_ready = false;
      continue;
    }
    if (count < 0 && errno == EINTR)
    {
      continue;
    }
    if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
    {
      // Non-blocking and full: wait for room instead of dropping the line.
      const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
          deadline - std::chrono::steady_clock::now()).count();
      if (remaining <= 0)
      {
        return; // the reader made no room in time: the line is cut short or missing
      }
      if (reported_ready)
      {
        // poll() saw room that write() could not use: do not spin on it.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      struct pollfd waiting = {STDOUT_FILENO, POLLOUT, 0};
      const int ready = ::poll(&waiting, 1, static_cast<int>(std::min<long long>(remaining, 100)));
      if (ready < 0 && errno != EINTR)
      {
        return;
      }
      reported_ready = ready > 0;
      continue; // room, a hang-up (the next write reports it), or another 100 ms slice
    }
    return; // EPIPE and the like: nobody is left to tell
  }
}

void recordLibraryWarning(const string& text)
{
  g_report.addWarning(text);
}

// The files a run contract promises, by the same naming rules the run used.
opencorr_cli_io::AuditLayout auditLayoutFromContract(const Json& contract, const fs::path& root)
{
  opencorr_cli_io::AuditLayout layout;
  const Json& options = contract.at("options");
  const string format = options.at("format").get<string>();
  if (format != "csv" && format != "raw")
  {
    throw runtime_error("unknown output format in contract: " + format);
  }
  const OutputFormat output_format = format == "raw" ? OutputFormat::RawBinary : OutputFormat::Csv;
  const bool diagnostics = options.value("diagnostics", false);
  fs::path diagnostics_dir;
  if (diagnostics)
  {
    const string recorded = options.value("diagnostics_dir", string("default"));
    diagnostics_dir = recorded == "default" ? root / "diagnostics" : fs::path(recorded);
  }
  for (const auto& target : contract.at("targets"))
  {
    const string file = target.at("file").get<string>();
    validateOutputName(file);
    const fs::path path = root / file;
    layout.outputs.push_back({"frame/" + file, path,
                              output_format == OutputFormat::RawBinary ? kRawBinaryRecordBytes : 0, true});
    if (diagnostics)
    {
      const fs::path diagnostic = diagnosticsPathForFrame(diagnostics_dir, path);
      layout.outputs.push_back({"diagnostics/" + diagnostic.filename().string(), diagnostic, 0, false});
    }
  }
  if (output_format == OutputFormat::RawBinary)
  {
    layout.metadata.push_back(root / "manifest.opencorr-raw.json");
  }
  if (diagnostics)
  {
    layout.metadata.push_back(diagnostics_dir / "manifest.opencorr-diagnostics.json");
  }
  return layout;
}

// --verify-only <output_dir>: is this multi-frame output directory complete and
// does every file still match the ledger? Read-only: nothing is computed,
// created, archived or rewritten, and no GPU is touched. One JSON object on
// stdout; exit 0 only for status "complete".
int runVerifyOnly(const vector<string>& args)
{
  Json report = Json::object();
  Json contract;
  const bool usable_arguments = args.size() == 2 && !args[1].empty();
  if (!usable_arguments)
  {
    // Still one JSON verdict on stdout: a caller that parses stdout gets an
    // answer for every invocation of this mode.
    const string message = "--verify-only takes exactly one argument, the multi-frame output directory (got "
                           + to_string(args.size() - 1) + (args.size() == 2 ? ", empty)" : ")");
    cerr << "Error: " << message << endl;
    report["status"] = "error";
    report["complete"] = false;
    report["output_dir"] = nullptr;
    report["problems"] = Json::array({message});
  }
  else try
  {
    report = opencorr_cli_io::auditRunDirectory(
        args[1], [&](const Json& found, const fs::path& root)
        {
          // The first call is for the recorded contract; a later one, if any,
          // for a pending append. The manifest check below is about the first.
          if (contract.is_null()) contract = found;
          return auditLayoutFromContract(found, root);
        });
    // The raw manifest must describe exactly the frames the contract lists: a
    // reader takes the frame list from the manifest, not from the ledger.
    const string status = report.at("status").get<string>();
    if ((status == "complete" || status == "inputs_changed")
        && contract.at("options").at("format") == "raw")
    {
      const fs::path root(report.at("output_dir").get<string>());
      vector<OutputFrameRecord> frames;
      string manifest_error;
      string problem;
      if (!readRawManifestFrames(root / "manifest.opencorr-raw.json", frames, manifest_error))
      {
        problem = "raw manifest layout is invalid: " + manifest_error;
      }
      else if (frames.size() != contract.at("targets").size())
      {
        problem = "raw manifest lists " + to_string(frames.size()) + " frames, the contract "
                  + to_string(contract.at("targets").size());
      }
      else
      {
        for (size_t i = 0; i < frames.size() && problem.empty(); ++i)
        {
          const auto& target = contract.at("targets")[i];
          std::error_code size_error;
          const auto size = fs::file_size(root / frames[i].file_name, size_error);
          if (frames[i].file_name != target.at("file").get<string>()
              || frames[i].output_name != target.at("output_name").get<string>()
              || size_error || size != frames[i].bytes)
          {
            problem = "raw manifest entry " + to_string(i + 1) + " does not describe frame file "
                      + target.at("file").get<string>();
          }
        }
      }
      if (!problem.empty())
      {
        report["problems"].push_back(problem);
        report["status"] = "damaged";
        report["complete"] = false;
      }
    }
  }
  catch (const std::system_error& ex)
  {
    report["status"] = "error";
    report["complete"] = false;
    report["problems"] = Json::array({string(ex.what()) + " (errno=" + to_string(ex.code().value()) + ")"});
  }
  catch (const exception& ex)
  {
    report["status"] = "error";
    report["complete"] = false;
    report["problems"] = Json::array({string(ex.what())});
  }

  const bool complete = report.value("complete", false);
  OrderedJson document = OrderedJson::object();
  document["schema"] = "fasticgn.verify-only/1";
  // The verdict first, then the counts, then the detail the audit collected.
  for (const char* key : {"status", "complete"})
  {
    if (report.contains(key)) document[key] = OrderedJson(report[key]);
  }
  document["exit_code"] = complete ? 0 : 1;
  // Only an audit that threw before it could report leaves output_dir out, and it had args[1].
  document["output_dir"] = report.contains("output_dir") ? OrderedJson(report["output_dir"]) : OrderedJson(args[1]);
  for (const char* key : {"frames_expected", "frames_verified", "files_expected", "files_verified",
                          "bytes_verified", "rows_per_frame", "problems", "warnings", "completion",
                          "manifests", "inputs", "contract"})
  {
    if (report.contains(key)) document[key] = OrderedJson(report[key]);
  }
  for (auto item = report.begin(); item != report.end(); ++item)
  {
    if (!document.contains(item.key())) document[item.key()] = OrderedJson(item.value());
  }
  // The binary that did the checking; the one that wrote the directory is in "contract".
  document["tool"] = "opencorr_cli_cuda_mvp";
  document["version"] = OPENCORR_GPU_MVP_VERSION;
  document["git_hash"] = OPENCORR_GPU_MVP_GIT_HASH;
  const string status = document.value("status", string("error"));
  cerr << "Verify-only: " << status;
  if (document.contains("frames_expected"))
  {
    cerr << ", " << document.value("frames_verified", 0) << "/" << document.value("frames_expected", 0)
         << " frames match the ledger";
  }
  cerr << "; nothing was computed or written" << endl;
  if (document.contains("problems"))
  {
    for (const auto& problem : document["problems"])
    {
      cerr << "  problem: " << problem.get<string>() << endl;
    }
  }
  if (document.contains("warnings"))
  {
    for (const auto& warning : document["warnings"])
    {
      cerr << "  note: " << warning.get<string>() << endl;
    }
  }
  writeStdoutLine(document);
  return complete ? 0 : 1;
}

// Options that are followed by a value. "--help" or "-h" right after one of
// them is that value (a directory may be called -h), not a request for help.
// smoke/test_cli_interface.py binds this list to the options that read a value.
constexpr const char* kOptionsWithValue[] = {
    "--ref", "--tar-list", "--output-dir",
    "--gpu-device", "--max-iter", "--subset-size", "--conv-threshold", "--kernel", "--fftcc",
    "--threads", "--line", "--durability", "--output-format", "--input-depth", "--input-align",
    "--input-normalize", "--diagnostics-dir", "--decode-threads", "--output-roi", "--output-step"};

bool helpRequested(const vector<string>& args)
{
  for (size_t i = 0; i < args.size(); ++i)
  {
    if (args[i] == "--help" || args[i] == "-h")
    {
      return true;
    }
    if (std::any_of(std::begin(kOptionsWithValue), std::end(kOptionsWithValue),
                    [&](const char* option) { return args[i] == option; }))
    {
      ++i; // its value, whatever it looks like
    }
  }
  return false;
}

int runComputation(const vector<string>& args, const char* executable)
{
  try
  {
    if (args.empty())
    {
      ErrorLine() << "no arguments given";
      printUsage(cerr, false);
      return 1;
    }
    if (args[0] == "--multi-frame")
    {
      return runMultiFrame(args, executable);
    }
    if (!args[0].empty() && args[0][0] == '-')
    {
      ErrorLine() << "unknown option: " << args[0];
      printUsage(cerr, false);
      return 1;
    }
    return runSingleFrame(args);
  }
  catch (const std::system_error& ex)
  {
    // An I/O fault, never a damage verdict: name the path and the errno.
    ErrorLine() << ex.what() << " (errno=" << ex.code().value() << ")";
    return 1;
  }
  catch (const exception& ex)
  {
    ErrorLine() << ex.what();
    return 1;
  }
  catch (const string& ex)
  {
    ErrorLine() << ex;
    return 1;
  }
}

} // namespace

int main(int argc, char* argv[])
{
  const auto started = std::chrono::steady_clock::now();
  // A process started with stdout closed must not write its status line into
  // whatever file later reuses descriptor 1.
  const bool stdout_open = ::fcntl(STDOUT_FILENO, F_GETFD) != -1;
  vector<string> args;
  args.reserve(argc > 1 ? static_cast<size_t>(argc - 1) : 0);
  for (int i = 1; i < argc; ++i)
  {
    args.emplace_back(argv[i]);
  }

  // The three informational modes answer on stdout themselves and print no
  // status line. --verify-only comes first: its one argument is always a
  // directory, also when that directory is called --help or -h.
  if (!args.empty() && args[0] == "--verify-only")
  {
    return runVerifyOnly(args);
  }
  if (helpRequested(args))
  {
    printUsage(cout, true);
    cout.flush();
    return cout.good() ? 0 : 1;
  }
  if (!args.empty() && args[0] == "--version")
  {
    if (args.size() != 1)
    {
      cerr << "Error: --version does not accept additional arguments" << endl;
      return 1;
    }
    printVersion();
    return 0;
  }

  opencorr_cli_io::setWarningSink(recordLibraryWarning);
  const int exit_code = runComputation(args, argc > 0 ? argv[0] : "opencorr_cli_cuda_mvp");
  if (stdout_open)
  {
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    writeStdoutLine(g_report.snapshot(exit_code, elapsed));
  }
  return exit_code;
}
