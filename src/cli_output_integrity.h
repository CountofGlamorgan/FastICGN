/*
 * Copyright (c) 2026 Chonglang Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"

namespace opencorr_cli_io {
namespace fs = std::filesystem;
using Json = nlohmann::json;

// Publication durability policy.
//   Rename: temporary file + atomic rename/link only. Crash-consistent for any
//           process death (the page cache is shared), zero fsync per frame.
//           A finished job takes one constant-cost barrier over its ledger and
//           manifests; frame payloads stay in the page cache and are re-verified
//           by digest on the next run.
//   Fsync:  additionally fsync every payload, every ledger record and both
//           directories before reporting success (power-loss durability).
enum class Durability { Rename, Fsync };
bool parseDurability(const std::string& text, Durability& mode);
const char* durabilityName(Durability mode);
// Process-wide default used by atomicWrite/atomicWriteJson. Set once from the
// command line before any output is published.
void setDefaultDurability(Durability mode);
Durability defaultDurability();

// Accumulated cost of the publication path, in seconds, for the whole process.
// Reported at the end of a multi-frame run so the writer thread stays
// measurable without an external profiler.
struct PublishProfile {
  double resolve = 0.0;       // destination canonicalisation, spec lookup, symlink checks
  double digest = 0.0;        // XXH3-128 over the payload
  double intent = 0.0;        // ledger record publication (whole atomic write)
  double payload_open = 0.0;  // temporary payload file creation
  double payload_write = 0.0; // payload write() + close()
  double payload_link = 0.0;  // link/rename publication of the payload
  uint64_t frames = 0;        // payloads published through a session
};
PublishProfile publishProfile();

// Warnings this library prints on stderr ("Warning: <text>") are also handed to
// this callback, so the CLI can repeat them in its closing status line. Set it
// once, before any session exists; nullptr (the default) only prints.
void setWarningSink(void (*sink)(const std::string& text));

// Scheduling options that used to be part of the run contract. They only decide
// how fast the same bytes are produced (shown on 2026-10-05 against the 60-frame
// production anchor: every value gave the identical output tree), so contracts
// written since then omit them and a contract that still lists them is compared
// as if it did not: changing one of them never rejects a resume.
extern const char* const kUncomparedLegacyOptions[4]; // gpu_device, threads, preload, decode_threads

// "a.b: old -> new; ..." for every leaf that differs between two JSON objects,
// at most `limit` entries plus a count of the rest. Used to say which contract
// field rejected a resume.
std::string describeJsonDifferences(const Json& recorded, const Json& requested, size_t limit = 8);

Json readJson(const fs::path& path);
// Same strict grammar, duplicate-key and depth checks as readJson, for text
// this process generated itself.
Json parseJson(const std::string& text, const std::string& label);
Json fingerprint(const fs::path& path);
std::string digest(const char* data, size_t size);
Json executableIdentity(const fs::path& invoked_path);
void atomicWrite(const fs::path& path, const char* data, size_t size, bool replace = true);
void atomicWriteJson(const fs::path& path, const Json& value);

struct OutputSpec {
  std::string key; // stable logical name, independent of output-root relocation
  fs::path path;
  size_t rows;
  size_t record_bytes; // zero means newline-delimited CSV with one header
};

// One multi-frame job. All inputs must remain immutable during a run.
// A durable intent is published BEFORE its data file. A matching intent+file
// proves completion; an intent without a file is safely replayable.
class OutputSession {
public:
  OutputSession(const fs::path& root, Json contract, std::vector<OutputSpec> outputs,
                std::vector<fs::path> metadata, const fs::path& tar_list,
                Durability durability = Durability::Fsync, bool repair = false);
  ~OutputSession();
  OutputSession(const OutputSession&) = delete;
  OutputSession& operator=(const OutputSession&) = delete;
  bool complete() const;
  void begin(); // call after application-level manifest validation, while locks are held
  void cleanupResidue(); // archive abandoned private files after validation, under directory locks
  size_t existingFiles() const;
  size_t verifiedFiles() const;   // payloads plus metadata re-read at startup
  uint64_t verifiedBytes() const;
  size_t repairedFiles() const;   // files moved aside by --repair
  fs::path quarantineDirectory() const; // empty unless something was quarantined
  void publish(const fs::path& path, const char* data, size_t size);
  void finish();
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// --verify-only: a read-only audit of one multi-frame output directory. Nothing
// is created, moved, repaired or rewritten (no quarantine, no residue archiving,
// no directory creation); the directory is held under a shared lock while it is
// read, so a run in progress is reported as "locked" instead of being judged.
struct AuditOutput
{
  std::string key;     // ledger key, for example "frame/frame_000001.bin"
  fs::path path;
  size_t record_bytes; // zero means newline-delimited CSV
  bool frame;          // counted in frames_expected / frames_verified
};
struct AuditLayout
{
  std::vector<AuditOutput> outputs;
  std::vector<fs::path> metadata; // manifests the completion record digests
};
// Derives the files a contract promises. The CLI owns the naming rules, so it
// supplies this; an exception marks the contract as unusable. It is called
// first for the recorded contract and, when contract.pending.json extends that
// contract, once more for the pending one.
using AuditLayoutFn = std::function<AuditLayout(const Json& contract, const fs::path& root)>;
// Returns the verdict as a JSON object; "status" is one of complete, incomplete,
// damaged, inputs_changed, locked, not_a_run, not_found. I/O faults while
// reading (std::system_error) propagate to the caller.
Json auditRunDirectory(const fs::path& root, const AuditLayoutFn& layout);
} // namespace opencorr_cli_io
