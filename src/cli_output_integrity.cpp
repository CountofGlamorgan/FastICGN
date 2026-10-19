/*
 * Copyright (c) 2026 Chonglang Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "cli_output_integrity.h"
#define XXH_INLINE_ALL
#include "xxhash/xxhash.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <system_error>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace opencorr_cli_io {
namespace {
constexpr const char* algorithm = "xxh3-128";
constexpr const char* repair_hint = " (use --repair to quarantine and regenerate it)";
std::atomic<unsigned long long> sequence{0};
std::atomic<Durability> process_durability{Durability::Fsync};

struct Fd {
  int value = -1;
  ~Fd() { if (value >= 0) ::close(value); }
};

// Lock-free accumulation of the publication cost. Every counter is nanoseconds
// on the steady clock; the CLI converts to seconds when it reports them.
using Clock = std::chrono::steady_clock;
struct Counter {
  std::atomic<uint64_t> ns{0};
  void add(Clock::time_point start) {
    ns.fetch_add(static_cast<uint64_t>(
                     std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count()),
                 std::memory_order_relaxed);
  }
  double seconds() const { return static_cast<double>(ns.load(std::memory_order_relaxed)) * 1e-9; }
};
struct Profile {
  Counter resolve, digest, intent, payload_open, payload_write, payload_link;
  std::atomic<uint64_t> frames{0};
};
Profile profile;

// Per-call breakdown of publishFile, recorded only for frame payloads.
struct FileTiming {
  Counter* open = nullptr;
  Counter* write = nullptr;
  Counter* link = nullptr;
};

void fail(const std::string& operation, const fs::path& path) {
  const int code = errno;
  throw std::system_error(code, std::generic_category(), operation + ": " + path.string());
}

fs::path canonicalPath(const fs::path& path) {
  return fs::weakly_canonical(fs::absolute(path));
}

std::string hex(XXH128_hash_t value) {
  XXH128_canonical_t bytes;
  XXH128_canonicalFromHash(&bytes, value);
  constexpr char alphabet[] = "0123456789abcdef";
  std::string result;
  for (unsigned char b : bytes.digest) { result += alphabet[b >> 4]; result += alphabet[b & 15]; }
  return result;
}

void syncDirectory(const fs::path& dir) {
  Fd fd;
  fd.value = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fd.value < 0) fail("open output directory", dir);
  if (::fsync(fd.value) != 0) fail("sync output directory", dir);
}

// Durability barrier over an already published file, used once per finished run
// by the rename policy instead of once per frame.
void syncFile(const fs::path& path) {
  Fd fd;
  fd.value = ::open(path.c_str(), O_WRONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd.value < 0) fail("open published output", path);
  if (::fsync(fd.value) != 0) fail("sync published output", path);
}

bool isRegularNonSymlink(const fs::path& path) {
  return !fs::is_symlink(fs::symlink_status(path)) && fs::is_regular_file(path);
}

void requireRegular(const fs::path& path) {
  if (!isRegularNonSymlink(path))
    throw std::runtime_error("expected a regular, non-symlink output file: " + path.string());
}

// One verdict for the reserved private namespace, shared by the resume scan and
// the post-run sweep, so both see the same names under the same rule. --repair
// only widens what counts as repairable damage; it never widens this.
constexpr const char* residue_prefix = ".opencorr-tmp-";
enum class Residue {
  Foreign,      // outside the reserved namespace: housekeeping never touches it
  Unrecognized, // reserved, but not a name this binary ever publishes under
  NotRegular,   // a well-formed reserved name that is not a regular file
  Reclaimable   // reserved, well formed, and safe to archive
};

Residue classifyResidue(const fs::path& path) {
  const std::string name = path.filename().string();
  if (name.rfind(residue_prefix, 0) != 0) return Residue::Foreign;
  const auto suffix = name.substr(std::strlen(residue_prefix));
  const auto separator = suffix.find('-');
  if (separator == std::string::npos || separator == 0 || separator + 1 == suffix.size()
      || suffix.find('-', separator + 1) != std::string::npos
      || suffix.substr(0, separator).find_first_not_of("0123456789") != std::string::npos
      || suffix.substr(separator + 1).find_first_not_of("0123456789") != std::string::npos)
    return Residue::Unrecognized;
  return isRegularNonSymlink(path) ? Residue::Reclaimable : Residue::NotRegular;
}

std::atomic<void (*)(const std::string&)> warning_sink{nullptr};

void emitWarning(const std::string& text) {
  std::cerr << "Warning: " << text << std::endl;
  if (const auto sink = warning_sink.load(std::memory_order_relaxed)) sink(text);
}

// Housekeeping never fails a run: a reserved entry that is not reclaimable is
// reported and left exactly where it is, with or without --repair.
void reportResidueSkip(Residue verdict, const fs::path& path) {
  if (verdict == Residue::NotRegular)
    emitWarning("leaving reserved name in place, not a regular file: " + path.string());
  else if (verdict == Residue::Unrecognized)
    emitWarning("leaving unrecognized reserved name in place: " + path.string());
}

// Compact, never-throwing rendering of one JSON leaf for a message. The
// contract's floating-point leaves were single precision before they were
// stored, so seven digits show the value that was typed (0.001), not its
// double expansion.
std::string renderLeaf(const Json& value) {
  if (value.is_number_float()) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.7g", value.get<double>());
    return text;
  }
  return value.dump(-1, ' ', false, Json::error_handler_t::replace);
}

// "a.b" -> rendered leaf for every non-object value below `value`.
void flattenJson(const Json& value, const std::string& prefix, std::map<std::string, std::string>& leaves) {
  if (value.is_object() && !value.empty()) {
    for (auto item = value.begin(); item != value.end(); ++item)
      flattenJson(item.value(), prefix.empty() ? item.key() : prefix + "." + item.key(), leaves);
  } else {
    leaves[prefix] = renderLeaf(value);
  }
}

// The contract as resume compares it: everything but the target list (compared
// separately, because appending is allowed) and the scheduling options that no
// longer belong to it.
Json comparableContract(Json contract) {
  if (!contract.is_object()) return contract;
  contract.erase("targets");
  const auto options = contract.find("options");
  if (options != contract.end() && options->is_object())
    for (const char* key : kUncomparedLegacyOptions) options->erase(key);
  return contract;
}

using Leaves = std::map<std::string, std::string>;

// Keys whose rendered leaf differs, or that only one side has, in key order.
std::vector<std::string> differingKeys(const Leaves& before, const Leaves& after) {
  std::set<std::string> keys;
  for (const auto& leaf : before) keys.insert(leaf.first);
  for (const auto& leaf : after) keys.insert(leaf.first);
  std::vector<std::string> changed;
  for (const auto& key : keys) {
    const auto old_leaf = before.find(key), new_leaf = after.find(key);
    if (old_leaf == before.end() || new_leaf == after.end() || old_leaf->second != new_leaf->second)
      changed.push_back(key);
  }
  return changed;
}

// Says which contract fields differ, and which kind of change that is, so that
// a rejected resume can be acted on without diffing the ledger by hand.
std::string describeContractChange(const Json& recorded, const Json& requested) {
  Leaves before, after;
  flattenJson(recorded, "", before);
  flattenJson(requested, "", after);
  bool binary = false, reference = false, options = false;
  for (const auto& key : differingKeys(before, after)) {
    if (key.rfind("binary.", 0) == 0 || key == "git_hash" || key == "build_date") binary = true;
    else if (key.rfind("reference.", 0) == 0) reference = true;
    else if (key.rfind("options.", 0) == 0) options = true;
  }
  std::string hint;
  if (options) hint += "; an option that decides the output differs from the recorded run";
  if (reference) hint += "; the reference image path or content differs";
  if (binary) hint += "; another build of the executable wrote this directory (resume needs the byte-identical binary)";
  return " (recorded -> requested): " + describeJsonDifferences(recorded, requested) + hint
         + "; use a new output directory for a different run";
}

// Where two target lists stop agreeing, for the same reason.
std::string describeTargetChange(const Json& recorded, const Json& requested) {
  if (!recorded.is_array() || !requested.is_array()) return std::string();
  const size_t common = std::min(recorded.size(), requested.size());
  for (size_t i = 0; i < common; ++i)
    if (recorded[i] != requested[i])
      return ": target " + std::to_string(i + 1) + " of the recorded " + std::to_string(recorded.size())
             + " differs (recorded -> requested): " + describeJsonDifferences(recorded[i], requested[i], 4);
  if (requested.size() < recorded.size())
    return ": the recorded run lists " + std::to_string(recorded.size()) + " targets, this command only "
           + std::to_string(requested.size()) + " (a shorter list is not a resume)";
  return std::string();
}

void requireRecord(const Json& record) {
  if (!record.is_object() || record.value("algorithm", "") != algorithm
      || !record.contains("bytes") || !record["bytes"].is_number_unsigned()
      || !record.contains("digest") || !record["digest"].is_string())
    throw std::runtime_error("invalid integrity record schema");
  const auto value = record["digest"].get<std::string>();
  if (value.size() != 32 || value.find_first_not_of("0123456789abcdef") != std::string::npos)
    throw std::runtime_error("invalid integrity digest");
}

template <typename Source>
Json parseStrict(Source&& source, const std::string& label) {
  std::vector<std::set<std::string>> keys;
  const auto callback = [&](int depth, Json::parse_event_t event, Json& parsed) {
    if (depth > 64) throw std::runtime_error("JSON metadata nesting exceeds 64: " + label);
    if (event == Json::parse_event_t::object_start) keys.emplace_back();
    if (event == Json::parse_event_t::key && !keys.back().insert(parsed.get<std::string>()).second)
      throw std::runtime_error("duplicate JSON object key: " + parsed.get<std::string>());
    if (event == Json::parse_event_t::object_end) keys.pop_back();
    return true;
  };
  return Json::parse(std::forward<Source>(source), callback); // strict grammar, UTF-8 and Unicode escapes
}

// The single publication primitive. sync == false keeps the atomic temporary +
// rename/link publication and drops only the durability barriers.
void publishFile(const fs::path& path, const char* data, size_t size, bool replace, bool sync,
                 FileTiming timing = {}) {
  if (fs::is_symlink(fs::symlink_status(path))) throw std::runtime_error("refusing symlink output: " + path.string());
  const fs::path parent = path.parent_path().empty() ? fs::path(".") : path.parent_path();
  Fd fd;
  fs::path temporary;
  auto mark = Clock::now();
  for (int attempt = 0; attempt < 1024; ++attempt) {
    temporary = parent / (".opencorr-tmp-" + std::to_string(::getpid()) + "-" + std::to_string(sequence++));
    fd.value = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
    if (fd.value >= 0) break;
    if (errno != EEXIST) fail("create temporary output", temporary);
  }
  if (fd.value < 0) fail("allocate temporary output", path);
  if (timing.open) { timing.open->add(mark); mark = Clock::now(); }
  try {
    size_t offset = 0;
    while (offset < size) {
      const ssize_t count = ::write(fd.value, data + offset, std::min(size - offset, size_t{1} << 30));
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) { if (count == 0) errno = EIO; fail("write temporary output", temporary); }
      offset += static_cast<size_t>(count);
    }
    if (sync && ::fsync(fd.value) != 0) fail("sync temporary output", temporary);
    const int handle = fd.value;
    fd.value = -1;
    if (::close(handle) != 0) fail("close temporary output", temporary);
    if (timing.write) { timing.write->add(mark); mark = Clock::now(); }
    if (replace) fs::rename(temporary, path);
    else {
      // link publishes the fully written inode atomically without replacing a
      // destination that appeared after validation (including non-cooperating writers).
      if (::link(temporary.c_str(), path.c_str()) != 0) fail("publish new output", path);
      fs::remove(temporary);
    }
    if (sync) syncDirectory(parent);
    if (timing.link) timing.link->add(mark);
  } catch (...) {
    // Only this call's private temporary file may be removed. Never remove the
    // destination; a rename followed by a failed directory sync is recoverable.
    std::error_code ignored;
    fs::remove(temporary, ignored);
    throw;
  }
}

void publishJson(const fs::path& path, const Json& value, bool sync) {
  const auto bytes = value.dump(2) + "\n";
  publishFile(path, bytes.data(), bytes.size(), true, sync);
}
} // namespace

const char* const kUncomparedLegacyOptions[4] = {"gpu_device", "threads", "preload", "decode_threads"};

void setWarningSink(void (*sink)(const std::string& text)) {
  warning_sink.store(sink, std::memory_order_relaxed);
}

std::string describeJsonDifferences(const Json& recorded, const Json& requested, size_t limit) {
  Leaves before, after;
  flattenJson(recorded, "", before);
  flattenJson(requested, "", after);
  const auto changed = differingKeys(before, after);
  if (changed.empty()) return "no difference";
  const auto render = [](const Leaves& leaves, const std::string& key) {
    const auto leaf = leaves.find(key);
    return leaf == leaves.end() ? std::string("<absent>") : leaf->second;
  };
  std::string text;
  for (size_t i = 0; i < changed.size() && i < limit; ++i) {
    if (i) text += "; ";
    text += changed[i] + ": " + render(before, changed[i]) + " -> " + render(after, changed[i]);
  }
  if (changed.size() > limit) text += "; and " + std::to_string(changed.size() - limit) + " more";
  return text;
}

bool parseDurability(const std::string& text, Durability& mode) {
  if (text == "rename") { mode = Durability::Rename; return true; }
  if (text == "fsync") { mode = Durability::Fsync; return true; }
  return false;
}
const char* durabilityName(Durability mode) { return mode == Durability::Fsync ? "fsync" : "rename"; }

PublishProfile publishProfile() {
  PublishProfile result;
  result.resolve = profile.resolve.seconds();
  result.digest = profile.digest.seconds();
  result.intent = profile.intent.seconds();
  result.payload_open = profile.payload_open.seconds();
  result.payload_write = profile.payload_write.seconds();
  result.payload_link = profile.payload_link.seconds();
  result.frames = profile.frames.load(std::memory_order_relaxed);
  return result;
}

void setDefaultDurability(Durability mode) { process_durability.store(mode, std::memory_order_relaxed); }
Durability defaultDurability() { return process_durability.load(std::memory_order_relaxed); }

std::string digest(const char* data, size_t size) { return hex(XXH3_128bits(data, size)); }

Json fingerprint(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read fingerprint input: " + path.string());
  std::unique_ptr<XXH3_state_t, decltype(&XXH3_freeState)> state(XXH3_createState(), &XXH3_freeState);
  if (!state || XXH3_128bits_reset(state.get()) != XXH_OK) throw std::runtime_error("cannot initialize integrity digest");
  std::array<char, 64 * 1024> buffer;
  uint64_t bytes = 0;
  while (in) {
    in.read(buffer.data(), buffer.size());
    const auto count = in.gcount();
    if (count > 0) {
      XXH3_128bits_update(state.get(), buffer.data(), static_cast<size_t>(count));
      bytes += static_cast<uint64_t>(count);
    }
  }
  if (!in.eof()) throw std::runtime_error("fingerprint read failed: " + path.string());
  return {{"algorithm", algorithm}, {"bytes", bytes}, {"digest", hex(XXH3_128bits_digest(state.get()))}};
}

Json executableIdentity(const fs::path& invoked_path) {
#ifdef __linux__
  (void)invoked_path;
  return fingerprint("/proc/self/exe");
#else
  // POSIX fallback for an explicitly located executable.
  return fingerprint(fs::canonical(invoked_path));
#endif
}

// Reaching the bytes is I/O, and describing them is content. A file that is
// missing, cannot be opened, or faults while being read raises std::system_error
// carrying its errno: it propagates, is never isolated, and exits 1 naming the
// path and the errno. Only a file that was read and then says the wrong thing
// stays a runtime_error, which is what --repair may quarantine.
Json readJson(const fs::path& path) {
  if (!fs::exists(fs::symlink_status(path))) {
    errno = ENOENT;
    fail("open JSON metadata", path);
  }
  requireRegular(path); // a directory or symlink here stays a hard rejection
  if (fs::file_size(path) > 256ULL * 1024 * 1024)
    throw std::runtime_error("JSON metadata exceeds 256 MiB: " + path.string());
  Fd probe; // a real syscall, so an unopenable file reports its own errno
  probe.value = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (probe.value < 0) fail("open JSON metadata", path);
  std::ifstream in(path);
  if (!in) fail("open JSON metadata", path);
  auto result = parseStrict(in, path.string());
  if (in.bad()) {
    if (errno == 0) errno = EIO;
    fail("read JSON metadata", path);
  }
  return result;
}

Json parseJson(const std::string& text, const std::string& label) {
  if (text.size() > 256ULL * 1024 * 1024)
    throw std::runtime_error("JSON metadata exceeds 256 MiB: " + label);
  return parseStrict(text, label);
}

void atomicWrite(const fs::path& path, const char* data, size_t size, bool replace) {
  publishFile(path, data, size, replace, defaultDurability() == Durability::Fsync);
}

void atomicWriteJson(const fs::path& path, const Json& value) {
  publishJson(path, value, defaultDurability() == Durability::Fsync);
}

struct OutputSession::Impl {
  Fd lock;
  std::vector<std::unique_ptr<Fd>> extra_locks;
  fs::path root, state;
  std::vector<fs::path> extra_directories;
  Json contract;
  std::string contract_digest;
  std::map<fs::path, OutputSpec> outputs;
  std::map<std::string, Json> intents;
  std::set<std::string> committed;
  std::vector<fs::path> metadata;
  std::mutex mutex;
  size_t existing = 0;
  size_t verified_files = 0;
  uint64_t verified_bytes = 0;
  size_t repaired = 0;
  bool done = false;
  bool begun = false;
  bool config_dirty = false;
  bool has_prior = false;
  bool pending_contract = false;
  bool sync_each = true;
  bool repair = false;
  fs::path quarantine;
  fs::path intentPath(const std::string& key) const {
    return state / (digest(key.data(), key.size()) + ".json");
  }
  fs::path contractPath() const { return state / "contract.json"; }
  // An appended contract lives here until the extended run actually publishes a
  // new payload, so an append that dies early can still resume the old list.
  fs::path pendingContractPath() const { return state / "contract.pending.json"; }
  fs::path quarantineDir() {
    if (!quarantine.empty()) return quarantine;
    const std::time_t now = std::time(nullptr);
    struct tm parts {};
    ::gmtime_r(&now, &parts);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%dT%H%M%SZ", &parts);
    quarantine = state / "quarantine" / (std::string(stamp) + "-" + std::to_string(::getpid()));
    fs::create_directories(quarantine);
    return quarantine;
  }
  // Damaged files are moved aside, never deleted; replay regenerates them.
  void quarantineEntry(const fs::path& path) {
    const auto dir = quarantineDir();
    fs::path destination = dir / path.filename();
    for (int n = 1; fs::exists(fs::symlink_status(destination)); ++n)
      destination = dir / (path.filename().string() + "." + std::to_string(n));
    std::error_code code;
    fs::rename(path, destination, code);
    if (code) { // a diagnostics directory may live on another filesystem
      fs::copy_file(path, destination, fs::copy_options::overwrite_existing);
      fs::remove(path);
    }
    ++repaired;
  }
  void promoteContract() {
    if (!pending_contract) return;
    publishJson(contractPath(), contract, sync_each);
    std::error_code ignored;
    fs::remove(pendingContractPath(), ignored);
    pending_contract = false;
  }
};

OutputSession::OutputSession(const fs::path& root, Json contract,
                             std::vector<OutputSpec> outputs, std::vector<fs::path> metadata,
                             const fs::path& tar_list, Durability durability, bool repair)
    : impl_(std::make_unique<Impl>()) {
  auto& p = *impl_;
  p.sync_each = durability == Durability::Fsync;
  p.repair = repair;
  p.root = canonicalPath(root);
  p.state = p.root / ".opencorr-state";
  p.contract = std::move(contract);
  const auto serialized_contract = p.contract.dump();
  p.contract_digest = digest(serialized_contract.data(), serialized_contract.size());
  p.lock.value = ::open(p.root.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (p.lock.value < 0) fail("open run directory", p.root);
  if (::flock(p.lock.value, LOCK_EX | LOCK_NB) != 0) fail("run directory already locked", p.root);
  if (fs::is_symlink(fs::symlink_status(p.state)))
    throw std::runtime_error("run state must not be a symlink");
  std::set<fs::path> allowed{p.state, canonicalPath(tar_list)};
  std::set<fs::path> extra_directories;
  auto allowPath = [&](const fs::path& path) {
    allowed.insert(path);
    const auto relative = path.lexically_relative(p.root);
    if (!relative.empty() && *relative.begin() != "..") allowed.insert(p.root / *relative.begin());
    if (path.parent_path() != p.root) extra_directories.insert(path.parent_path());
  };
  for (auto& spec : outputs) {
    // Preserve the final component so symlink outputs are rejected, not followed.
    spec.path = canonicalPath(spec.path.parent_path()) / spec.path.filename();
    if (!p.outputs.emplace(spec.path, spec).second)
      throw std::runtime_error("duplicate output destination: " + spec.path.string());
    if (spec.path == p.state || spec.path.filename().string().rfind(".opencorr-", 0) == 0)
      throw std::runtime_error("reserved output filename: " + spec.path.string());
    allowPath(spec.path);
  }
  for (auto& path : metadata) {
    path = canonicalPath(path.parent_path()) / path.filename();
    if (p.outputs.count(path)) throw std::runtime_error("data output collides with manifest: " + path.string());
    allowPath(path);
    p.metadata.push_back(path);
  }
  const auto config_path = p.contractPath();
  const bool prior = fs::exists(config_path);
  Json prior_contract;
  if (prior) {
    prior_contract = readJson(config_path);
    const auto before = comparableContract(prior_contract), after = comparableContract(p.contract);
    if (before != after)
      throw std::runtime_error("resume rejected: run configuration/reference/binary changed"
                               + describeContractChange(before, after));
    const auto& old_targets = prior_contract.at("targets");
    const auto& new_targets = p.contract.at("targets");
    if (!old_targets.is_array() || !new_targets.is_array() || old_targets.size() > new_targets.size()
        || !std::equal(old_targets.begin(), old_targets.end(), new_targets.begin()))
      throw std::runtime_error("resume rejected: targets changed/reordered; only an identical prefix plus append is allowed"
                               + describeTargetChange(old_targets, new_targets));
  }
  // Quarantine only ever applies to files this contract already describes.
  const bool repairing = p.repair && prior;
  for (const auto& dir : extra_directories) {
    fs::create_directories(dir);
    auto lock = std::make_unique<Fd>();
    lock->value = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (lock->value < 0) fail("open diagnostics directory", dir);
    if (::flock(lock->value, LOCK_EX | LOCK_NB) != 0) fail("diagnostics directory already locked", dir);
    p.extra_locks.push_back(std::move(lock));
    p.extra_directories.push_back(dir);
  }
  // A dedicated output directory prevents foreign/stale frames entering later
  // aggregation. Reserved private names are recognized here but never archived
  // here: cleanupResidue() owns that in every mode, under the same classifier.
  auto verify_directory = [&](const fs::path& dir) {
    for (const auto& item : fs::directory_iterator(dir)) {
      if (classifyResidue(item.path()) != Residue::Foreign) continue;
      if (!allowed.count(item.path()))
        throw std::runtime_error("untracked entry in output directory: " + item.path().string());
    }
  };
  verify_directory(p.root);
  for (const auto& dir : extra_directories) verify_directory(dir);
  if (!prior) {
    for (const auto& pair : p.outputs)
      if (fs::exists(fs::symlink_status(pair.first)))
        throw std::runtime_error("existing output has no verified run record: " + pair.first.string());
    for (const auto& path : p.metadata)
      if (fs::exists(fs::symlink_status(path)))
        throw std::runtime_error("existing manifest has no verified run record: " + path.string());
    if (fs::exists(p.state))
      for (const auto& item : fs::directory_iterator(p.state))
        if (classifyResidue(item.path()) == Residue::Foreign)
          throw std::runtime_error("run state has no valid contract: " + p.state.string());
  }
  for (const auto& pair : p.outputs) {
    const auto& spec = pair.second;
    const auto intent_path = p.intentPath(spec.key);
    bool has_intent = fs::exists(intent_path);
    if (has_intent) {
      Json intent;
      try {
        intent = readJson(intent_path);
        requireRecord(intent);
        if (intent.value("schema_version", 0) != 1 || intent.value("key", "") != spec.key
            || !intent.contains("rows") || !intent["rows"].is_number_unsigned()
            || intent["rows"].get<uint64_t>() != spec.rows)
          throw std::runtime_error("invalid output intent: " + intent_path.string());
        if (spec.record_bytes && intent["bytes"].get<uint64_t>() != spec.rows * spec.record_bytes)
          throw std::runtime_error("raw record size mismatch in intent: " + spec.path.string());
      } catch (const std::system_error&) {
        throw; // fs::filesystem_error included: an I/O fault is not a damaged record
      } catch (const std::exception& ex) {
        if (!repairing) throw std::runtime_error(std::string(ex.what()) + repair_hint);
        p.quarantineEntry(intent_path);
        has_intent = false;
      }
      if (has_intent) p.intents.emplace(spec.key, intent);
    }
    if (fs::exists(fs::symlink_status(spec.path))) {
      requireRegular(spec.path); // symlinked outputs stay a hard rejection, also under --repair
      Json observed;
      bool intact = has_intent;
      if (intact) {
        observed = fingerprint(spec.path);
        const auto& expected = p.intents.at(spec.key);
        intact = observed["bytes"] == expected["bytes"] && observed["digest"] == expected["digest"];
      }
      if (!intact) {
        if (!repairing)
          throw std::runtime_error(std::string(has_intent ? "corrupt/truncated output; resume refused: "
                                                          : "existing output has no commit intent: ")
                                   + spec.path.string() + repair_hint);
        p.quarantineEntry(spec.path);
      } else {
        p.committed.insert(spec.key);
        p.verified_bytes += observed["bytes"].get<uint64_t>();
        ++p.verified_files;
        ++p.existing;
      }
    }
  }
  // Validate all pre-existing JSON even if a fully completed run can return early.
  for (const auto& path : p.metadata) {
    if (!fs::exists(path)) continue;
    requireRegular(path);
    try {
      (void)readJson(path);
      p.verified_bytes += fs::file_size(path);
      ++p.verified_files;
    } catch (const std::system_error&) {
      throw; // fs::filesystem_error included: an I/O fault is not a damaged record
    } catch (const std::exception&) {
      if (!repairing) throw;
      p.quarantineEntry(path);
    }
  }
  if (prior && fs::exists(p.state / "complete.json")) {
    const auto completion_path = p.state / "complete.json";
    requireRegular(completion_path); // --repair never follows a symlink
    // Only a parse or schema/digest verdict counts as a damaged record. Every
    // system_error (fs::filesystem_error included) is rethrown so a transient
    // I/O fault is reported instead of being silently repaired into a rebuild.
    bool damaged = false;
    std::string reason;
    fs::path damaged_manifest; // set only while a verdict can name one manifest
    try {
      const auto completion = readJson(completion_path);
      if (!completion.is_object() || completion.value("schema_version", 0) != 1
          || !completion.contains("contract_digest") || !completion["contract_digest"].is_string()
          || !completion.contains("metadata") || !completion["metadata"].is_object())
        throw std::runtime_error("invalid run completion record");
      const auto hash = completion["contract_digest"].get<std::string>();
      if (hash.size() != 32 || hash.find_first_not_of("0123456789abcdef") != std::string::npos)
        throw std::runtime_error("invalid completion contract digest");
      if (completion["contract_digest"] == p.contract_digest) {
        if (completion["metadata"].size() != p.metadata.size())
          throw std::runtime_error("invalid metadata completion record");
        for (const auto& path : p.metadata) {
          const auto key = path.filename().string();
          if (!completion["metadata"].contains(key) || !completion["metadata"][key].is_string())
            throw std::runtime_error("missing metadata completion digest");
          if (fs::exists(path)) {
            damaged_manifest = path; // a verdict raised below is about this file
            const auto normalized = readJson(path).dump();
            if (completion["metadata"][key] != digest(normalized.data(), normalized.size()))
              throw std::runtime_error("completed manifest changed: " + path.string());
            damaged_manifest.clear();
          }
        }
      }
      p.done = completion["contract_digest"] == p.contract_digest
               && p.committed.size() == p.outputs.size()
               && std::all_of(p.metadata.begin(), p.metadata.end(), [](const auto& f) { return fs::is_regular_file(f); });
    } catch (const std::system_error&) {
      throw; // filesystem_error included: an I/O fault is not a damaged record
    } catch (const Json::exception& ex) {
      damaged = true;
      reason = ex.what();
    } catch (const std::runtime_error& ex) {
      damaged = true;
      reason = ex.what();
    }
    if (damaged) {
      if (!repairing) throw std::runtime_error(reason + repair_hint);
      // Inputs/configuration and every payload were checked above. Preserve both
      // sides of a completion/manifest disagreement, then reconstruct by replay:
      // the completion record, plus the one manifest the verdict named, if any.
      // Manifests that verified stay in place and are rewritten by the replay.
      std::cerr << "Repair: run completion record rejected (" << reason << ")" << std::endl;
      p.quarantineEntry(completion_path);
      if (!damaged_manifest.empty() && fs::exists(damaged_manifest)) p.quarantineEntry(damaged_manifest);
      p.done = false;
    }
  }
  p.has_prior = prior;
  p.config_dirty = !prior || prior_contract != p.contract;
}

OutputSession::~OutputSession() = default;
void OutputSession::begin() {
  auto& p = *impl_;
  fs::create_directories(p.state);
  if (p.sync_each) syncDirectory(p.root);
  if (p.config_dirty && p.has_prior) {
    // Appending: the shorter contract on disk stays authoritative until the
    // first appended payload is published (OutputSession::publish promotes it).
    publishJson(p.pendingContractPath(), p.contract, p.sync_each);
    p.pending_contract = true;
  } else if (p.config_dirty) {
    publishJson(p.contractPath(), p.contract, p.sync_each);
  } else {
    std::error_code ignored;
    fs::remove(p.pendingContractPath(), ignored); // abandoned append
  }
  p.begun = true;
}
void OutputSession::cleanupResidue() {
  auto& p = *impl_;
  if (!p.has_prior && !p.begun) return;
  std::vector<fs::path> directories{p.root, p.state};
  directories.insert(directories.end(), p.extra_directories.begin(), p.extra_directories.end());
  std::vector<fs::path> residue;
  for (const auto& dir : directories) {
    for (const auto& item : fs::directory_iterator(dir)) {
      // The same classifier the resume scan used, so --repair and a plain resume
      // archive and report exactly the same entries.
      const auto verdict = classifyResidue(item.path());
      if (verdict == Residue::Reclaimable) residue.push_back(item.path());
      else reportResidueSkip(verdict, item.path());
    }
  }
  if (!p.pending_contract && fs::exists(fs::symlink_status(p.pendingContractPath()))) {
    if (isRegularNonSymlink(p.pendingContractPath())) residue.push_back(p.pendingContractPath());
    else reportResidueSkip(Residue::NotRegular, p.pendingContractPath());
  }
  for (const auto& path : residue) p.quarantineEntry(path);
  if (!residue.empty())
    std::cerr << "Archived " << residue.size() << " abandoned private file(s) to "
              << p.quarantine.string() << "; nothing was deleted" << std::endl;
  if (!residue.empty() && p.sync_each) {
    syncDirectory(p.quarantine);
    for (const auto& dir : directories) syncDirectory(dir);
  }
}
bool OutputSession::complete() const { return impl_->done; }
size_t OutputSession::existingFiles() const { return impl_->existing; }
size_t OutputSession::verifiedFiles() const { return impl_->verified_files; }
uint64_t OutputSession::verifiedBytes() const { return impl_->verified_bytes; }
size_t OutputSession::repairedFiles() const { return impl_->repaired; }
fs::path OutputSession::quarantineDirectory() const { return impl_->quarantine; }

void OutputSession::publish(const fs::path& path, const char* data, size_t size) {
  auto& p = *impl_;
  if (!p.begun) throw std::logic_error("output session has not begun");
  auto mark = Clock::now();
  const auto full_path = canonicalPath(path.parent_path()) / path.filename();
  const auto& spec = p.outputs.at(full_path);
  if (spec.record_bytes) {
    if (size != spec.rows * spec.record_bytes) throw std::runtime_error("serialized raw row count mismatch");
  } else if (static_cast<size_t>(std::count(data, data + size, '\n')) != spec.rows + 1) {
    throw std::runtime_error("serialized CSV row count mismatch");
  }
  profile.resolve.add(mark);
  mark = Clock::now();
  const auto payload_digest = digest(data, size);
  profile.digest.add(mark);
  Json intent{{"schema_version", 1}, {"key", spec.key}, {"rows", spec.rows},
              {"algorithm", algorithm}, {"bytes", size}, {"digest", payload_digest}};
  const auto prior = p.intents.find(spec.key); // immutable after construction
  if (prior != p.intents.end()) {
    if (prior->second != intent) throw std::runtime_error("replayed bytes differ from recorded output: " + path.string());
    std::lock_guard<std::mutex> guard(p.mutex);
    if (p.committed.count(spec.key)) return; // keep existing file byte-for-byte and inode unchanged
  }
  mark = Clock::now();
  if (fs::exists(fs::symlink_status(full_path)))
    throw std::runtime_error("output appeared while run was active: " + full_path.string());
  // Intent first makes a crash between the two publications recoverable.
  publishJson(p.intentPath(spec.key), intent, p.sync_each);
  profile.intent.add(mark);
  publishFile(full_path, data, size, false, p.sync_each,
              FileTiming{&profile.payload_open, &profile.payload_write, &profile.payload_link});
  profile.frames.fetch_add(1, std::memory_order_relaxed);
  std::lock_guard<std::mutex> guard(p.mutex);
  p.promoteContract(); // the old target list stopped being a valid resume point
  p.committed.insert(spec.key);
}

void OutputSession::finish() {
  auto& p = *impl_;
  std::lock_guard<std::mutex> guard(p.mutex);
  if (p.committed.size() != p.outputs.size()) throw std::runtime_error("run has uncommitted output files");
  p.promoteContract();
  Json metadata = Json::object();
  for (const auto& path : p.metadata) {
    const auto normalized = readJson(path).dump();
    metadata[path.filename().string()] = digest(normalized.data(), normalized.size());
  }
  if (!p.sync_each) {
    // One constant-cost barrier for the whole job: ledger and manifests become
    // durable, frame payloads stay page-cache resident and are digest-verified.
    for (const auto& path : p.metadata) syncFile(path);
    syncFile(p.contractPath());
    syncDirectory(p.root);
    for (const auto& dir : p.extra_directories) syncDirectory(dir);
  }
  // The completion record is always durable, in both policies.
  publishJson(p.state / "complete.json",
              {{"schema_version", 1}, {"contract_digest", p.contract_digest},
               {"durability", durabilityName(p.sync_each ? Durability::Fsync : Durability::Rename)},
               {"metadata", metadata}}, true);
  p.done = true;
}

namespace {
// A capped list of names for the audit report: the count is exact, the names
// are the first few, so a directory with thousands of bad frames still yields
// one short line.
struct AuditNames {
  size_t count = 0;
  Json names = Json::array();
  void add(const std::string& name) {
    if (count++ < 20) names.push_back(name);
  }
  void store(Json& report, const char* key) const {
    report[key] = names;
    report[std::string(key) + "_count"] = count;
  }
};

std::string auditName(const fs::path& path, const fs::path& root) {
  const auto relative = path.lexically_relative(root);
  return (!relative.empty() && *relative.begin() != "..") ? relative.string() : path.string();
}
} // namespace

Json auditRunDirectory(const fs::path& requested_root, const AuditLayoutFn& layout_fn) {
  Json report = Json::object();
  Json problems = Json::array();
  Json warnings = Json::array();
  const fs::path root = canonicalPath(requested_root);
  report["output_dir"] = root.string();
  const auto conclude = [&](const char* status) {
    report["status"] = status;
    report["complete"] = std::string(status) == "complete";
    report["problems"] = problems;
    report["warnings"] = warnings;
    return report;
  };
  if (!fs::exists(root)) {
    problems.push_back("the output directory does not exist");
    return conclude("not_found");
  }
  if (!fs::is_directory(root)) {
    problems.push_back("the path is not a directory");
    return conclude("not_a_run");
  }
  Fd lock;
  lock.value = ::open(root.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (lock.value < 0) fail("open run directory", root);
  // Shared: several audits may run at once, and a writer (exclusive lock) is
  // either seen here or refused while the audit reads.
  if (::flock(lock.value, LOCK_SH | LOCK_NB) != 0) {
    if (errno != EWOULDBLOCK) fail("lock run directory", root);
    problems.push_back("a run holds this directory's lock right now; verify again when it has finished");
    return conclude("locked");
  }
  const fs::path state = root / ".opencorr-state";
  const fs::path contract_path = state / "contract.json";
  if (fs::is_symlink(fs::symlink_status(state))) {
    problems.push_back("the run state .opencorr-state is a symlink");
    return conclude("damaged");
  }
  if (!fs::exists(fs::symlink_status(contract_path))) {
    problems.push_back("no .opencorr-state/contract.json: not a ledgered multi-frame output directory "
                       "(single-frame output and binaries older than the output ledger write none)");
    return conclude("not_a_run");
  }

  Json contract;
  AuditLayout layout;
  try {
    contract = readJson(contract_path);
    if (!contract.is_object() || !contract.contains("targets") || !contract["targets"].is_array()
        || !contract.contains("options") || !contract["options"].is_object())
      throw std::runtime_error("unexpected contract layout");
    layout = layout_fn(contract, root);
  } catch (const std::system_error&) {
    throw;
  } catch (const std::exception& ex) {
    problems.push_back(std::string("contract.json is not a usable run contract: ") + ex.what());
    return conclude("damaged");
  }
  const auto serialized_contract = contract.dump();
  const auto contract_digest = digest(serialized_contract.data(), serialized_contract.size());
  {
    Json summary = {{"digest", contract_digest}, {"options", contract["options"]}};
    for (const char* key : {"schema_version", "git_hash", "build_date"})
      if (contract.contains(key)) summary[key] = contract[key];
    if (contract.contains("binary") && contract["binary"].is_object() && contract["binary"].contains("digest"))
      summary["binary_digest"] = contract["binary"]["digest"];
    if (contract.contains("reference") && contract["reference"].is_object() && contract["reference"].contains("path"))
      summary["reference"] = contract["reference"]["path"];
    Json legacy = Json::array();
    for (const char* key : kUncomparedLegacyOptions)
      if (contract["options"].contains(key)) legacy.push_back(key);
    // Recorded by a binary from before 2026-10-05; informational only.
    summary["legacy_scheduling_options"] = legacy;
    report["contract"] = summary;
  }

  bool damaged = false;
  bool incomplete = false;

  // Payloads against their ledger records.
  AuditNames missing, mismatched, no_intent, invalid_intent;
  size_t frames_expected = 0, frames_verified = 0, files_verified = 0;
  uint64_t bytes_verified = 0;
  std::set<uint64_t> frame_rows;
  std::set<fs::path> tracked{state};
  std::set<fs::path> directories{root};
  const auto track = [&](const fs::path& path) {
    tracked.insert(path);
    const auto relative = path.lexically_relative(root);
    if (!relative.empty() && *relative.begin() != "..") tracked.insert(root / *relative.begin());
    if (path.parent_path() != root) directories.insert(path.parent_path());
  };
  // One promised output against its ledger record, in the order a resume
  // checks it: the record itself, then the file, then its bytes.
  enum class Check { Verified, Missing, NotRegular, NoIntent, InvalidIntent, Mismatched };
  struct Checked {
    Check verdict;
    uint64_t bytes = 0;
    uint64_t rows = 0;
  };
  const auto checkOutput = [&](const AuditOutput& output) -> Checked {
    const fs::path intent_path = state / (digest(output.key.data(), output.key.size()) + ".json");
    Json intent;
    const bool has_intent = fs::exists(fs::symlink_status(intent_path));
    if (has_intent) {
      try {
        intent = readJson(intent_path);
        requireRecord(intent);
        if (intent.value("schema_version", 0) != 1 || intent.value("key", "") != output.key
            || !intent.contains("rows") || !intent["rows"].is_number_unsigned())
          throw std::runtime_error("invalid output intent");
        if (output.record_bytes
            && intent["bytes"].get<uint64_t>() != intent["rows"].get<uint64_t>() * output.record_bytes)
          throw std::runtime_error("raw record size mismatch in intent");
      } catch (const std::system_error&) {
        throw;
      } catch (const std::exception&) {
        return {Check::InvalidIntent};
      }
    }
    if (!fs::exists(fs::symlink_status(output.path))) return {Check::Missing};
    if (!isRegularNonSymlink(output.path)) return {Check::NotRegular};
    if (!has_intent) return {Check::NoIntent};
    const auto observed = fingerprint(output.path);
    if (observed["bytes"] != intent["bytes"] || observed["digest"] != intent["digest"]) return {Check::Mismatched};
    return {Check::Verified, observed["bytes"].get<uint64_t>(), intent["rows"].get<uint64_t>()};
  };
  // Verdicts that contradict the ledger, filed under the list that names them.
  const auto fileDamage = [&](Check verdict, const std::string& name) {
    if (verdict == Check::InvalidIntent) invalid_intent.add(name);
    else if (verdict == Check::NoIntent) no_intent.add(name);
    else mismatched.add(name); // NotRegular, Mismatched
    damaged = true;
  };

  // Per directory, the extensions of the outputs the run publishes there.
  std::map<fs::path, std::set<std::string>> output_suffixes;
  std::set<fs::path> promised;
  for (const auto& output : layout.outputs) {
    track(output.path);
    promised.insert(output.path);
    output_suffixes[output.path.parent_path()].insert(output.path.extension().string());
    if (output.frame) ++frames_expected;
    const std::string name = auditName(output.path, root);
    const auto checked = checkOutput(output);
    if (checked.verdict == Check::Missing) {
      missing.add(name); // with or without an intent, the next run publishes it by replay
      incomplete = true;
    } else if (checked.verdict != Check::Verified) {
      fileDamage(checked.verdict, name);
    } else {
      ++files_verified;
      bytes_verified += checked.bytes;
      if (output.frame) {
        ++frames_verified;
        frame_rows.insert(checked.rows);
      }
    }
  }

  // An append in progress. The longer contract waits in contract.pending.json
  // and replaces contract.json only once the first appended output has been
  // published. A process that dies between that publication and the promotion
  // leaves an output on disk that the recorded contract does not list yet.
  // That is an unfinished append, not a foreign file, exactly when a resume
  // with the longer list would accept it: the pending contract extends the
  // recorded one (same binary, reference and options, the recorded targets as
  // a prefix), promises this file, and the file matches its own ledger record
  // in size and digest. Anything short of that keeps its damage verdict.
  const fs::path pending_path = state / "contract.pending.json";
  const bool pending_append = fs::exists(fs::symlink_status(pending_path));
  const char* pending_state = "none";
  AuditNames appended;
  bool appended_damage = false;
  if (pending_append) {
    pending_state = "unreadable";
    try {
      const Json pending = readJson(pending_path);
      if (!pending.is_object() || !pending.contains("targets") || !pending["targets"].is_array()
          || !pending.contains("options") || !pending["options"].is_object())
        throw std::runtime_error("unexpected contract layout");
      const auto& recorded_targets = contract["targets"];
      const auto& pending_targets = pending["targets"];
      if (comparableContract(pending) != comparableContract(contract)
          || recorded_targets.size() > pending_targets.size()
          || !std::equal(recorded_targets.begin(), recorded_targets.end(), pending_targets.begin())) {
        pending_state = "unrelated";
      } else if (recorded_targets.size() == pending_targets.size()) {
        pending_state = "promoted"; // the promotion happened; only its source file is left
      } else {
        const AuditLayout longer = layout_fn(pending, root);
        pending_state = "append";
        for (const auto& output : longer.outputs) {
          if (promised.count(output.path)) continue;
          const auto checked = checkOutput(output);
          if (checked.verdict == Check::Missing) continue; // not computed yet
          const std::string name = auditName(output.path, root);
          track(output.path);
          if (checked.verdict != Check::Verified) {
            fileDamage(checked.verdict, name);
            appended_damage = true;
          } else {
            appended.add(name);
            if (output.frame) frame_rows.insert(checked.rows);
          }
        }
      }
    } catch (const std::system_error&) {
      throw;
    } catch (const std::exception&) {
      pending_state = "unreadable";
    }
  }
  if (appended.count) {
    problems.push_back("an append was interrupted after it published its first new output and before its "
                       "longer target list replaced the recorded one: rerun the append command (the longer "
                       "--tar-list) unchanged to continue, no --repair needed; the shorter list is no longer "
                       "accepted, because it does not account for the appended output");
    incomplete = true;
  }
  if (appended_damage)
    problems.push_back("an interrupted append left an output that contradicts its ledger record; rerunning "
                       "the append command (the longer --tar-list) with --repair regenerates it");
  if (frame_rows.size() > 1) {
    problems.push_back("frames record different row counts; one run writes the same rows for every frame");
    damaged = true;
  }

  // Completion record: does it describe this contract, and do the manifests
  // still say what it digested?
  Json completion = {{"present", false}, {"state", "missing"}};
  Json completion_record;
  const fs::path completion_path = state / "complete.json";
  bool completion_current = false;
  if (fs::exists(fs::symlink_status(completion_path))) {
    completion["present"] = true;
    try {
      requireRegular(completion_path);
      completion_record = readJson(completion_path);
      if (!completion_record.is_object() || completion_record.value("schema_version", 0) != 1
          || !completion_record.contains("contract_digest") || !completion_record["contract_digest"].is_string()
          || !completion_record.contains("metadata") || !completion_record["metadata"].is_object())
        throw std::runtime_error("invalid run completion record");
      completion_current = completion_record["contract_digest"] == contract_digest;
      completion["state"] = completion_current ? "current" : "other_contract";
      if (completion_record.contains("durability")) completion["durability"] = completion_record["durability"];
      if (!completion_current) {
        problems.push_back("complete.json belongs to another target list or contract (an append or resume "
                           "was started and did not finish)");
        incomplete = true;
      }
    } catch (const std::system_error&) {
      throw;
    } catch (const std::exception& ex) {
      completion["state"] = "invalid";
      problems.push_back(std::string("complete.json is damaged: ") + ex.what());
      damaged = true;
    }
  } else {
    problems.push_back("no complete.json: the run has not finished");
    incomplete = true;
  }
  report["completion"] = completion;

  Json manifests = Json::array();
  for (const auto& path : layout.metadata) {
    track(path);
    Json entry = {{"file", auditName(path, root)}, {"state", "ok"}};
    if (!fs::exists(fs::symlink_status(path))) {
      entry["state"] = "missing";
      incomplete = true;
      if (completion_current) problems.push_back("manifest missing: " + auditName(path, root));
    } else {
      try {
        requireRegular(path);
        const auto normalized = readJson(path).dump();
        ++files_verified;
        bytes_verified += fs::file_size(path);
        if (completion_current) {
          const auto key = path.filename().string();
          const auto& recorded = completion_record["metadata"];
          if (!recorded.contains(key) || !recorded[key].is_string()
              || recorded[key] != digest(normalized.data(), normalized.size())) {
            entry["state"] = "changed";
            problems.push_back("manifest differs from the one the completion record digested: "
                               + auditName(path, root));
            damaged = true;
          }
        }
      } catch (const std::system_error&) {
        throw;
      } catch (const std::exception& ex) {
        entry["state"] = "invalid";
        problems.push_back("manifest does not parse: " + auditName(path, root) + ": " + ex.what());
        damaged = true;
      }
    }
    manifests.push_back(entry);
  }
  report["manifests"] = manifests;

  // Everything else in the directories the run owns.
  AuditNames untracked, untracked_outputs, residue;
  for (const auto& dir : directories) {
    if (!fs::is_directory(dir)) continue;
    for (const auto& item : fs::directory_iterator(dir)) {
      const auto verdict = classifyResidue(item.path());
      if (verdict != Residue::Foreign) {
        residue.add(auditName(item.path(), root));
        continue;
      }
      if (tracked.count(item.path())) continue;
      // The run's own --tar-list file may live here (the pipeline keeps it
      // there) and the ledger does not name it, so an unknown name alone is
      // only listed. An entry with the extension of this directory's outputs
      // is different: a reader that globs frames would take it for one, and
      // a glob does not care whether the name is a file, a symlink or a
      // directory, so neither does this test.
      const auto suffixes = output_suffixes.find(dir);
      if (suffixes != output_suffixes.end() && suffixes->second.count(item.path().extension().string()))
        untracked_outputs.add(auditName(item.path(), root));
      else
        untracked.add(auditName(item.path(), root));
    }
  }
  for (const auto& item : fs::directory_iterator(state))
    if (classifyResidue(item.path()) != Residue::Foreign) residue.add(auditName(item.path(), root));
  if (untracked_outputs.count) {
    problems.push_back("entries named like this run's outputs (same directory, same extension) that the "
                       "ledger does not list");
    damaged = true;
  }
  if (residue.count)
    warnings.push_back("temporary files of an interrupted run; the next run archives them under "
                       ".opencorr-state/quarantine/");
  const std::string pending_text(pending_state);
  if (pending_text == "append" && !appended.count && !appended_damage)
    warnings.push_back("contract.pending.json: an append was started and stopped before any appended output "
                       "was published; the recorded target list is still the valid one");
  else if (pending_text == "promoted")
    warnings.push_back("contract.pending.json repeats the recorded contract: an append was promoted and only "
                       "this file is left over; the next run archives it");
  else if (pending_text == "unrelated")
    warnings.push_back("contract.pending.json does not extend the recorded contract (other binary, reference, "
                       "options or target prefix), so it says nothing about this directory; the next run "
                       "archives it");
  else if (pending_text == "unreadable")
    warnings.push_back("contract.pending.json cannot be read as a contract, so it says nothing about an "
                       "interrupted append; the next run archives it");
  if (missing.count) problems.push_back("frames or diagnostics files the contract promises are missing");
  if (mismatched.count)
    problems.push_back("files whose bytes do not match their ledger record (or that are not regular files)");
  if (no_intent.count) problems.push_back("files with no ledger record");
  if (invalid_intent.count) problems.push_back("ledger records that do not parse or do not describe their file");

  // Inputs: still the bytes the contract fingerprinted?
  AuditNames inputs_changed, inputs_missing;
  size_t inputs_checked = 0;
  {
    std::map<std::string, Json> inputs;
    const auto collect = [&](const Json& identity) {
      if (identity.is_object() && identity.contains("path") && identity["path"].is_string()
          && identity.contains("content") && identity["content"].is_object())
        inputs.emplace(identity["path"].get<std::string>(), identity["content"]);
    };
    if (contract.contains("reference")) collect(contract["reference"]);
    for (const auto& target : contract["targets"])
      if (target.is_object() && target.contains("input")) collect(target["input"]);
    for (const auto& input : inputs) {
      const fs::path path(input.first);
      std::error_code code;
      if (!fs::is_regular_file(path, code)) {
        inputs_missing.add(input.first);
        continue;
      }
      ++inputs_checked;
      const auto observed = fingerprint(path);
      if (!input.second.contains("bytes") || !input.second.contains("digest")
          || observed["bytes"] != input.second["bytes"] || observed["digest"] != input.second["digest"])
        inputs_changed.add(input.first);
    }
  }
  Json inputs_report = {{"checked", inputs_checked},
                        {"status", inputs_changed.count ? "changed" : inputs_missing.count ? "missing" : "match"}};
  inputs_changed.store(inputs_report, "changed");
  inputs_missing.store(inputs_report, "missing");
  report["inputs"] = inputs_report;
  if (inputs_changed.count)
    problems.push_back("input images no longer match the fingerprints in the contract; the outputs still "
                       "match their ledger, but they describe images that are no longer at those paths");
  if (inputs_missing.count)
    warnings.push_back("input images are not at the recorded paths, so they were not compared; rerunning "
                       "the command needs them there");

  report["frames_expected"] = frames_expected;
  report["frames_verified"] = frames_verified;
  report["files_expected"] = layout.outputs.size() + layout.metadata.size();
  report["files_verified"] = files_verified;
  report["bytes_verified"] = bytes_verified;
  if (frame_rows.size() == 1) report["rows_per_frame"] = *frame_rows.begin();
  missing.store(report, "missing");
  mismatched.store(report, "mismatched");
  no_intent.store(report, "no_intent");
  invalid_intent.store(report, "invalid_intent");
  untracked_outputs.store(report, "untracked_outputs");
  untracked.store(report, "untracked");
  residue.store(report, "residue");
  report["pending_append"] = pending_append;
  report["pending_append_state"] = pending_state;
  appended.store(report, "appended_published");
  if (damaged) return conclude("damaged");
  if (incomplete) return conclude("incomplete");
  if (inputs_changed.count) return conclude("inputs_changed");
  return conclude("complete");
}
} // namespace opencorr_cli_io
