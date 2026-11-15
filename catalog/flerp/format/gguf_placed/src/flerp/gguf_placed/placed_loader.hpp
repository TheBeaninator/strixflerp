// SPDX-License-Identifier: MIT
// format.flerp.gguf_placed, the device side: one device allocation of a rank plan's byte total, and every held
// tensor filled into its region by byte-range reads of the file.
//
// The reads are gufo's WeightUpload (src/core/hip/weight_upload.hpp): CopyMapped resolves a tensor's address in the
// reader's mapping to a file offset and reads the range through its own unbuffered descriptor into pinned staging
// (16 MiB pieces), so the mapping's data pages are never touched and host residency is the staging buffers. Where
// the filesystem refuses unbuffered reads it reads through the buffered descriptor and drops the pages behind it.
//
// FillPolicy{single_reader = true} is the quiet way, for a host whose disk and page cache others depend on: one
// thread reads each range through one O_DIRECT descriptor in 16 MiB pieces into one pinned buffer and copies it to
// the device, sleeping pause_ms between tensors (and chunk_pause_ms between pieces). Where O_DIRECT is refused it
// reads buffered and drops each piece from the page cache behind it (POSIX_FADV_DONTNEED).
//
// FillPolicy{data_path} (single reader only) reads the bytes from another file at the same offsets: loom-swarm's
// sparse full-size copy of a rank's layout (swarm.sh `path`, on the RAM disk /srv/swarm or under /cache), whose size
// must equal the GGUF's. The header still comes from the reader's own file. punch_behind then frees each piece of
// that file once it is on the device (fallocate PUNCH_HOLE | KEEP_SIZE): on a tmpfs the RAM is given back as the
// load goes, so a stage's RAM-disk copy and its device copy (one 128 GB pool on Strix Halo) never coexist in full.
// It destroys the copy: never set it on a file someone else keeps.
//
// FillPolicy{swarm} reads nothing from a file: the rank's bytes come over the network from loom-swarm (the
// origin on control-node's rail, 198.51.100.31:7150, and any peers), straight into the device. A `loom-swarm stream`
// helper process (loom-swarm's own client: its GETs, its BLAKE3 check of every 16 MiB piece against the manifest,
// its retries of a failed piece on the next source, `conns` requests in flight) is given the plan's byte ranges on
// stdin and writes each verified piece to a pipe as frames {u64 file offset, u64 length, bytes}; this side reads the
// frames into its pinned staging buffer and copies every part that lies in a plan entry to that entry's device
// offset. Every entry must be covered exactly once; a helper that fails is run again, up to swarm.attempts, for the
// entries it left incomplete. The weights never touch a disk or a tmpfs. The header still comes from the reader's
// file (only its first pages are read).
//
// The caller (an executor's loader) re-points its tensor references onto the regions by host address, exactly as
// it re-points them into a whole-file region.
#ifndef FLERP_GGUF_PLACED_PLACED_LOADER_HPP_
#define FLERP_GGUF_PLACED_PLACED_LOADER_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "flerp/gguf_placed/placement.hpp"

namespace flerp::gguf_placed {

struct PlacedRegion {
  const void* host{nullptr};  // the tensor's address in the reader's mapping
  void* device{nullptr};
  std::size_t size{0};        // the tensor's encoded bytes
};

struct SwarmSource {
  std::string helper;                 // the loom-swarm binary (with the `stream` verb)
  std::string manifest;               // the file's (or a rank layout's) .swarm.json
  std::vector<std::string> sources;   // addr:port, the origin first
  std::uint32_t conns{4};
  std::uint32_t attempts{3};          // helper runs at most
  [[nodiscard]] bool enabled() const noexcept { return !helper.empty() && !manifest.empty() && !sources.empty(); }
};

struct FillPolicy {
  bool single_reader{false};
  std::uint32_t pause_ms{0};        // between tensors
  std::uint32_t chunk_pause_ms{0};  // between 16 MiB pieces
  std::string data_path;            // single reader: read the bytes from this file (same offsets, same size)
  bool punch_behind{false};         // single reader + data_path: free each piece of data_path after its copy
  SwarmSource swarm;                // set: the bytes come from loom-swarm, not from any file (implies one reader)
};

/// What a load did. The gate compares allocated_bytes with the plan (computed again, elsewhere, from the header).
struct LoadReport {
  bool placed{false};
  std::string plan_sha256;
  std::uint32_t world{1};
  std::uint32_t rank{0};
  std::uint32_t blocks{0};
  std::uint32_t block_begin{0};
  std::uint32_t block_end{0};
  std::uint64_t plan_device_bytes{0};
  std::uint64_t allocated_bytes{0};    // the size given to the one hipMalloc
  std::uint64_t tensor_bytes{0};       // bytes read from the file into the device
  std::size_t tensors{0};
  std::size_t unloaded_tensors{0};
  std::uint64_t unloaded_bytes{0};
  double plan_s{0.0};
  double alloc_s{0.0};
  double fill_s{0.0};
  bool single_reader{false};
  bool direct_io{false};               // single reader: every piece came through O_DIRECT
  std::uint64_t read_calls{0};         // single reader: pieces read
  double pause_s{0.0};                 // single reader: seconds slept on purpose, inside fill_s
  std::string data_path;               // single reader: the file the bytes came from (empty: the GGUF)
  std::uint64_t punched_bytes{0};      // bytes of data_path freed behind the reads
  bool swarm{false};                   // the bytes came from loom-swarm
  std::uint64_t swarm_bytes{0};        // bytes received in frames (whole pieces: more than tensor_bytes at the edges)
  std::uint32_t swarm_runs{0};         // helper processes run
  std::string swarm_stats;             // the last helper's stderr JSON line
  std::uint64_t hip_free_before{0};
  std::uint64_t hip_total{0};
  std::uint64_t hip_free_after{0};
  std::uint64_t rss_file_before{0};    // RssFile of the process around the fill: pages of the mapping it faulted in
  std::uint64_t rss_file_after{0};
};

/// Owns the allocation until Release().
class PlacedWeights {
public:
  PlacedWeights() = default;
  ~PlacedWeights();
  PlacedWeights(const PlacedWeights&) = delete;
  PlacedWeights& operator=(const PlacedWeights&) = delete;

  [[nodiscard]] void* base() const noexcept { return base_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] const std::vector<PlacedRegion>& regions() const noexcept { return regions_; }
  /// The caller takes the allocation (hipFree(base) is then its job).
  void* Release() noexcept;

private:
  friend std::unique_ptr<PlacedWeights> LoadRank(const gufo::core::GgufReader&, const Plan&, std::uint32_t,
                                                 LoadReport*, std::string*, FillPolicy);
  void* base_{nullptr};
  std::size_t bytes_{0};
  std::vector<PlacedRegion> regions_;
};

/// Allocates plan.ranks[rank].device_bytes on the current device and fills every entry. Null with *error on failure
/// (nothing stays allocated). Refuses a plan larger than the device's free memory less 1 GiB.
[[nodiscard]] std::unique_ptr<PlacedWeights> LoadRank(const gufo::core::GgufReader& reader, const Plan& plan,
                                                      std::uint32_t rank, LoadReport* report, std::string* error,
                                                      FillPolicy policy = {});

}  // namespace flerp::gguf_placed

#endif  // FLERP_GGUF_PLACED_PLACED_LOADER_HPP_
