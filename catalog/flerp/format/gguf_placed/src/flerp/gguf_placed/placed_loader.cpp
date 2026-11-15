// SPDX-License-Identifier: MIT
// See placed_loader.hpp.
#include "flerp/gguf_placed/placed_loader.hpp"

#include <hip/hip_runtime.h>

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <thread>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern char** environ;

#include "src/core/hip/weight_upload.hpp"

namespace flerp::gguf_placed {
namespace {

using Clock = std::chrono::steady_clock;

double Since(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

// RssFile of this process in bytes (Linux); 0 where /proc has no such line.
std::uint64_t RssFileBytes() {
  std::FILE* f = std::fopen("/proc/self/status", "r");
  if (f == nullptr) return 0;
  char line[256];
  std::uint64_t kb = 0;
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    if (std::strncmp(line, "RssFile:", 8) == 0) {
      unsigned long long value = 0;
      if (std::sscanf(line + 8, "%llu", &value) == 1) kb = value;
      break;
    }
  }
  std::fclose(f);
  return kb * 1024;
}

// One thread, one descriptor, one pinned buffer. Returns an error text, empty on success.
std::string FillSingleReader(const gufo::core::GgufMappedRegion& file, const RankPlan& rp, std::uint8_t* base,
                             FillPolicy policy, LoadReport& rep) {
  constexpr std::size_t kPiece = 16ULL << 20;
  constexpr std::size_t kPage = 4096;
  const std::string path = policy.data_path.empty() ? "/proc/self/fd/" + std::to_string(file.file_descriptor) : policy.data_path;
  if (!policy.data_path.empty()) {
    struct stat a{}, b{};
    if (::fstat(file.file_descriptor, &a) != 0 || ::stat(path.c_str(), &b) != 0) return "cannot stat the data file " + path;
    if (a.st_size != b.st_size) {
      return "data file " + path + " has " + std::to_string(b.st_size) + " bytes, the GGUF " + std::to_string(a.st_size);
    }
  }
  rep.data_path = policy.data_path;
  const bool punch = policy.punch_behind && !policy.data_path.empty();
  const int punch_fd = punch ? ::open(path.c_str(), O_RDWR | O_CLOEXEC) : -1;
  if (punch && punch_fd < 0) return "cannot open " + path + " to free it behind the reads";
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
  bool direct = fd >= 0;
  if (!direct) fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return "cannot reopen the GGUF file";
  void* staging = nullptr;
  if (hipHostMalloc(&staging, kPiece + kPage) != hipSuccess) {
    ::close(fd);
    return "staging allocation failed";
  }
  std::string failure;
  rep.direct_io = direct;
  const auto nap = [&](std::uint32_t ms) {
    if (ms == 0) return;
    const auto start = Clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    rep.pause_s += Since(start);
  };
  for (const auto& entry : rp.entries) {
    for (std::uint64_t done = 0; done < entry.bytes && failure.empty();) {
      const std::uint64_t offset = entry.file_offset + done;
      const std::size_t want = static_cast<std::size_t>(std::min<std::uint64_t>(kPiece, entry.bytes - done));
      const std::uint64_t begin = direct ? offset & ~static_cast<std::uint64_t>(kPage - 1) : offset;
      const std::size_t skip = static_cast<std::size_t>(offset - begin);
      const std::size_t length = direct ? (skip + want + kPage - 1) & ~(kPage - 1) : want;
      std::size_t got = 0;
      while (got < skip + want) {
        const ssize_t n = ::pread(fd, static_cast<char*>(staging) + got, length - got, static_cast<off_t>(begin + got));
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && direct && errno == EINVAL && got == 0) {
          // the filesystem took the open and refuses the read: buffered from here on, pages dropped behind
          ::close(fd);
          fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
          direct = false;
          rep.direct_io = false;
          if (fd < 0) failure = "cannot reopen the GGUF file";
          break;
        }
        if (n <= 0) {
          failure = "read of " + entry.name + " failed at " + std::to_string(begin + got);
          break;
        }
        got += static_cast<std::size_t>(n);
      }
      if (!failure.empty()) break;
      if (got < skip + want) continue;   // switched to buffered: read this piece again
      ++rep.read_calls;
      if (!direct) (void)::posix_fadvise(fd, static_cast<off_t>(begin), static_cast<off_t>(length), POSIX_FADV_DONTNEED);
      if (const auto status = hipMemcpy(base + entry.device_offset + done, static_cast<const char*>(staging) + skip,
                                        want, hipMemcpyHostToDevice);
          status != hipSuccess) {
        failure = "copy of " + entry.name + " to the device failed: " + hipGetErrorString(status);
        break;
      }
      if (punch) {
        // whole pages inside this piece only: a page shared with the next tensor stays until its own piece
        const std::uint64_t a = (offset + kPage - 1) & ~static_cast<std::uint64_t>(kPage - 1);
        const std::uint64_t b = (offset + want) & ~static_cast<std::uint64_t>(kPage - 1);
        if (b > a) {
          if (::fallocate(punch_fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, static_cast<off_t>(a), static_cast<off_t>(b - a)) != 0) {
            failure = "freeing " + path + " behind the read failed: " + std::strerror(errno);
            break;
          }
          rep.punched_bytes += b - a;
        }
      }
      done += want;
      nap(policy.chunk_pause_ms);
    }
    if (!failure.empty()) break;
    nap(policy.pause_ms);
  }
  (void)hipHostFree(staging);
  if (fd >= 0) ::close(fd);
  if (punch_fd >= 0) ::close(punch_fd);
  return failure;
}

// --- loom-swarm source --------------------------------------------------------------------------------------------
bool ReadFull(int fd, void* dst, std::size_t n) {
  auto* p = static_cast<char*>(dst);
  while (n > 0) {
    const ssize_t got = ::read(fd, p, n);
    if (got < 0 && errno == EINTR) continue;
    if (got <= 0) return false;
    p += got;
    n -= static_cast<std::size_t>(got);
  }
  return true;
}

// One helper run for the entries `todo` (indices into rp.entries, which are in file order). Adds the bytes it
// delivered to covered[]. Returns an error text, empty when the helper ended its stream cleanly and exited 0.
std::string SwarmRun(const RankPlan& rp, const std::vector<std::size_t>& todo, std::uint8_t* base, const SwarmSource& sw,
                     void* staging, std::size_t staging_bytes, std::vector<std::uint64_t>& covered, LoadReport& rep) {
  int in[2], out[2], err[2];
  if (::pipe2(in, O_CLOEXEC) != 0) return "pipe failed";
  if (::pipe2(out, O_CLOEXEC) != 0 || ::pipe2(err, O_CLOEXEC) != 0) return "pipe failed";
  (void)::fcntl(out[0], F_SETPIPE_SZ, 1 << 20);
  std::vector<std::string> args = {sw.helper, "stream", "--manifest", sw.manifest, "--ranges", "-", "--conns", std::to_string(sw.conns)};
  for (const auto& s : sw.sources) {
    args.push_back("--from");
    args.push_back(s);
  }
  std::vector<char*> argv;
  for (auto& a : args) argv.push_back(a.data());
  argv.push_back(nullptr);
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_adddup2(&fa, in[0], 0);
  posix_spawn_file_actions_adddup2(&fa, out[1], 1);
  posix_spawn_file_actions_adddup2(&fa, err[1], 2);
  pid_t pid = -1;
  const int rc = ::posix_spawn(&pid, sw.helper.c_str(), &fa, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&fa);
  ::close(in[0]);
  ::close(out[1]);
  ::close(err[1]);
  if (rc != 0) {
    ::close(in[1]);
    ::close(out[0]);
    ::close(err[0]);
    return "cannot start " + sw.helper + ": " + std::strerror(rc);
  }
  ++rep.swarm_runs;
  // the ranges are small (a few KiB): write them all, then EOF
  std::string ranges;
  for (const auto i : todo) ranges += std::to_string(rp.entries[i].file_offset) + " " + std::to_string(rp.entries[i].bytes) + "\n";
  const bool wrote = ::write(in[1], ranges.data(), ranges.size()) == static_cast<ssize_t>(ranges.size());
  ::close(in[1]);
  // stderr drained on a thread so the helper never blocks on it
  std::string err_text;
  std::thread drain([&] {
    char b[4096];
    for (ssize_t n; (n = ::read(err[0], b, sizeof(b))) != 0;) {
      if (n < 0 && errno == EINTR) continue;
      if (n < 0) break;
      err_text.append(b, static_cast<std::size_t>(n));
      if (err_text.size() > (1 << 16)) err_text.erase(0, err_text.size() - (1 << 15));
    }
  });
  std::string failure = wrote ? "" : "cannot hand the ranges to the helper";
  bool ended = false;
  while (failure.empty()) {
    std::uint64_t head[2];
    if (!ReadFull(out[0], head, sizeof(head))) {
      failure = "the helper's stream ended without its end frame";
      break;
    }
    if (head[0] == ~0ULL && head[1] == 0) {
      ended = true;
      break;
    }
    const std::uint64_t off = head[0], len = head[1];
    if (len == 0 || len > staging_bytes) {
      failure = "a frame of " + std::to_string(len) + " bytes";
      break;
    }
    if (!ReadFull(out[0], staging, len)) {
      failure = "the helper's stream broke inside a frame";
      break;
    }
    rep.swarm_bytes += len;
    // the entries this segment touches: the first whose end is past `off`, then on while they start before its end
    auto it = std::partition_point(todo.begin(), todo.end(), [&](std::size_t i) {
      return rp.entries[i].file_offset + rp.entries[i].bytes <= off;
    });
    for (; it != todo.end() && rp.entries[*it].file_offset < off + len; ++it) {
      const auto& e = rp.entries[*it];
      const std::uint64_t a = std::max(off, e.file_offset), b = std::min(off + len, e.file_offset + e.bytes);
      if (b <= a) continue;
      if (const auto st = hipMemcpy(base + e.device_offset + (a - e.file_offset), static_cast<const char*>(staging) + (a - off), b - a,
                                    hipMemcpyHostToDevice);
          st != hipSuccess) {
        failure = "copy of " + e.name + " to the device failed: " + hipGetErrorString(st);
        break;
      }
      covered[*it] += b - a;
    }
  }
  if (!failure.empty()) ::kill(pid, SIGTERM);
  ::close(out[0]);
  int status = 0;
  while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
  }
  drain.join();
  ::close(err[0]);
  const auto nl = err_text.rfind("{\"event\":\"stream\"");
  if (nl != std::string::npos) rep.swarm_stats = err_text.substr(nl, err_text.find('\n', nl) - nl);
  if (failure.empty() && (!ended || !WIFEXITED(status) || WEXITSTATUS(status) != 0)) failure = "the helper failed";
  if (!failure.empty()) {
    const std::string tail = err_text.size() > 400 ? err_text.substr(err_text.size() - 400) : err_text;
    return failure + " (exit status " + std::to_string(status) + "): " + tail;
  }
  return {};
}

std::string FillSwarm(const RankPlan& rp, std::uint8_t* base, const SwarmSource& sw, LoadReport& rep) {
  constexpr std::size_t kPiece = 16ULL << 20;   // loom-swarm's piece; a frame is at most one piece
  void* staging = nullptr;
  if (hipHostMalloc(&staging, kPiece) != hipSuccess) return "staging allocation failed";
  rep.swarm = true;
  std::vector<std::size_t> todo(rp.entries.size());
  for (std::size_t i = 0; i < todo.size(); ++i) todo[i] = i;
  for (std::size_t i = 1; i < todo.size(); ++i) {
    if (rp.entries[i].file_offset < rp.entries[i - 1].file_offset + rp.entries[i - 1].bytes) {
      (void)hipHostFree(staging);
      return "plan entries are not in file order";
    }
  }
  std::vector<std::uint64_t> covered(rp.entries.size(), 0);
  std::string failure;
  for (std::uint32_t attempt = 0; attempt < std::max<std::uint32_t>(sw.attempts, 1) && !todo.empty(); ++attempt) {
    failure = SwarmRun(rp, todo, base, sw, staging, kPiece, covered, rep);
    std::vector<std::size_t> left;
    for (const auto i : todo) {
      if (covered[i] > rp.entries[i].bytes) {
        (void)hipHostFree(staging);
        return "entry " + rp.entries[i].name + " was delivered twice";
      }
      if (covered[i] != rp.entries[i].bytes) {
        covered[i] = 0;   // asked again whole: the bytes it gets are the same
        left.push_back(i);
      }
    }
    todo = std::move(left);
    if (failure.empty() && !todo.empty()) failure = std::to_string(todo.size()) + " entries incomplete after a clean stream";
  }
  (void)hipHostFree(staging);
  if (!todo.empty()) return "loom-swarm: " + failure;
  return {};
}

}  // namespace

PlacedWeights::~PlacedWeights() {
  if (base_ != nullptr) (void)hipFree(base_);
}

void* PlacedWeights::Release() noexcept {
  void* out = base_;
  base_ = nullptr;
  return out;
}

std::unique_ptr<PlacedWeights> LoadRank(const gufo::core::GgufReader& reader, const Plan& plan, std::uint32_t rank,
                                        LoadReport* report, std::string* error, FillPolicy policy) {
  const auto fail = [&](std::string message) -> std::unique_ptr<PlacedWeights> {
    if (error != nullptr) *error = std::move(message);
    return nullptr;
  };
  if (rank >= plan.ranks.size()) return fail("rank " + std::to_string(rank) + " is outside the plan's world");
  const RankPlan& rp = plan.ranks[rank];
  LoadReport local;
  LoadReport& rep = report != nullptr ? *report : local;
  rep.placed = true;
  rep.plan_sha256 = plan.Sha256();
  rep.world = plan.world;
  rep.rank = rank;
  rep.blocks = plan.blocks;
  rep.block_begin = rp.block_begin;
  rep.block_end = rp.block_end;
  rep.plan_device_bytes = rp.device_bytes;
  rep.tensor_bytes = rp.tensor_bytes;
  rep.tensors = rp.entries.size();
  rep.unloaded_tensors = plan.unloaded_tensors;
  rep.unloaded_bytes = plan.unloaded_bytes;

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) return fail("hipMemGetInfo failed");
  rep.hip_free_before = free_bytes;
  rep.hip_total = total_bytes;
  constexpr std::uint64_t kReserve = 1ULL << 30;
  if (rp.device_bytes + kReserve > free_bytes) {
    return fail("the placement needs " + std::to_string(rp.device_bytes) + " bytes, the device has " +
                std::to_string(free_bytes) + " free");
  }

  auto placed = std::make_unique<PlacedWeights>();
  auto start = Clock::now();
  if (const auto status = hipMalloc(&placed->base_, rp.device_bytes); status != hipSuccess) {
    placed->base_ = nullptr;
    return fail("hipMalloc of " + std::to_string(rp.device_bytes) + " bytes failed: " + hipGetErrorString(status));
  }
  placed->bytes_ = rp.device_bytes;
  rep.allocated_bytes = rp.device_bytes;
  rep.alloc_s = Since(start);

  rep.rss_file_before = RssFileBytes();
  start = Clock::now();
  std::string upload_error;
  auto* base = static_cast<std::uint8_t*>(placed->base_);
  placed->regions_.reserve(rp.entries.size());
  rep.single_reader = policy.single_reader || policy.swarm.enabled();
  if (policy.single_reader || policy.swarm.enabled()) {
    for (const auto& entry : rp.entries) {
      if (entry.device_offset + entry.bytes > rp.device_bytes) return fail("plan entry " + entry.name + " lies outside the allocation");
      placed->regions_.push_back({entry.host, base + entry.device_offset, entry.bytes});
    }
    const std::string failure = policy.swarm.enabled() ? FillSwarm(rp, base, policy.swarm, rep)
                                                       : FillSingleReader(reader.GetMappedRegions()[0], rp, base, policy, rep);
    if (!failure.empty()) return fail("placed fill: " + failure);
    rep.fill_s = Since(start);
    rep.rss_file_after = RssFileBytes();
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) rep.hip_free_after = free_bytes;
    return placed;
  }
  auto uploader = gufo::hip::WeightUpload::Create(reader.GetMappedRegions(), &upload_error);
  if (!uploader) return fail("weight upload: " + upload_error);
  for (const auto& entry : rp.entries) {
    if (entry.device_offset + entry.bytes > rp.device_bytes) {
      (void)uploader->Finish(nullptr);
      return fail("plan entry " + entry.name + " lies outside the allocation");
    }
    placed->regions_.push_back({entry.host, base + entry.device_offset, entry.bytes});
    if (!uploader->CopyMapped(entry.host, entry.bytes, base + entry.device_offset, &upload_error)) break;
  }
  // Finish waits for every queued copy, also after a failed one.
  if (!uploader->Finish(&upload_error) || !upload_error.empty()) {
    uploader.reset();
    return fail("weight upload: " + upload_error);
  }
  uploader.reset();
  rep.fill_s = Since(start);
  rep.rss_file_after = RssFileBytes();
  if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) rep.hip_free_after = free_bytes;
  return placed;
}

}  // namespace flerp::gguf_placed
