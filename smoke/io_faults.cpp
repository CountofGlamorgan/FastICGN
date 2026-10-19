/*
 * Copyright (c) 2026 Chonglang Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

// Test-only syscall faults. Never linked into the CLI; loaded only by regression subprocesses.
#include <dlfcn.h>
#include <unistd.h>
#include <sys/stat.h>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
// Durability-syscall counter. OPENCORR_TEST_IO_COUNT names an append-only report
// file; OPENCORR_TEST_IO_COUNT_ROOT optionally restricts counting to descriptors
// under one directory, isolating the CLI's own publication barriers.
std::atomic<unsigned long long> sync_calls{0};
bool counted(int fd) {
  if (!std::getenv("OPENCORR_TEST_IO_COUNT")) return false;
  const char* root = std::getenv("OPENCORR_TEST_IO_COUNT_ROOT");
  if (!root) return true;
  char name[64], path[4096];
  std::snprintf(name, sizeof(name), "/proc/self/fd/%d", fd);
  const ssize_t n = ::readlink(name, path, sizeof(path) - 1);
  if (n < 0) return false;
  path[n] = 0;
  return std::strncmp(path, root, std::strlen(root)) == 0;
}
__attribute__((destructor)) void report_sync_calls() {
  const char* target = std::getenv("OPENCORR_TEST_IO_COUNT");
  if (!target) return;
  FILE* file = std::fopen(target, "a");
  if (!file) return;
  std::fprintf(file, "%llu\n", sync_calls.load());
  std::fclose(file);
}
bool mode(const char* expected) {
  const char* value = std::getenv("OPENCORR_TEST_IO_FAULT");
  return value && std::strcmp(value, expected) == 0;
}
bool selected(const char* path) {
  const char* target = std::getenv("OPENCORR_TEST_IO_TARGET");
  const char* leaf = std::strrchr(path, '/');
  return target && std::strcmp(leaf ? leaf + 1 : path, target) == 0;
}
bool payload(int fd) {
  char name[64], path[4096];
  std::snprintf(name, sizeof(name), "/proc/self/fd/%d", fd);
  const ssize_t n = ::readlink(name, path, sizeof(path) - 1);
  if (n < 0) return false;
  path[n] = 0;
  struct stat info;
  return std::strstr(path, ".opencorr-tmp-") && ::fstat(fd, &info) == 0
         && S_ISREG(info.st_mode) && info.st_size > 16384;
}
}
extern "C" int link(const char* from, const char* to) {
  static auto real = reinterpret_cast<int(*)(const char*, const char*)>(dlsym(RTLD_NEXT, "link"));
  if (selected(to) && mode("crash-before-publish")) _exit(86);
  if (selected(to) && mode("publish-error")) { errno = ENOSPC; return -1; }
  const int result = real(from, to);
  if (result == 0 && selected(to) && mode("crash-after-publish")) _exit(86);
  return result;
}
extern "C" int rename(const char* from, const char* to) {
  static auto real = reinterpret_cast<int(*)(const char*, const char*)>(dlsym(RTLD_NEXT, "rename"));
  if (selected(to) && mode("metadata-error")) { errno = EIO; return -1; }
  return real(from, to);
}
extern "C" ssize_t write(int fd, const void* bytes, size_t count) {
  static auto real = reinterpret_cast<ssize_t(*)(int,const void*,size_t)>(dlsym(RTLD_NEXT, "write"));
  static thread_local int failed_fd = -1;
  if (mode("short-write")) {
    if (failed_fd == fd) { errno = ENOSPC; return -1; }
    if (count > 16384) { failed_fd = fd; return real(fd, bytes, count / 2); }
  }
  return real(fd, bytes, count);
}
extern "C" int fsync(int fd) {
  static auto real = reinterpret_cast<int(*)(int)>(dlsym(RTLD_NEXT, "fsync"));
  if (counted(fd)) ++sync_calls;
  if (mode("fsync-error") && payload(fd)) { errno = EIO; return -1; }
  return real(fd);
}
extern "C" int fdatasync(int fd) {
  static auto real = reinterpret_cast<int(*)(int)>(dlsym(RTLD_NEXT, "fdatasync"));
  if (counted(fd)) ++sync_calls;
  return real(fd);
}
extern "C" int close(int fd) {
  static auto real = reinterpret_cast<int(*)(int)>(dlsym(RTLD_NEXT, "close"));
  const bool inject = mode("close-error") && payload(fd);
  const int result = real(fd);
  if (inject) { errno = EIO; return -1; }
  return result;
}
