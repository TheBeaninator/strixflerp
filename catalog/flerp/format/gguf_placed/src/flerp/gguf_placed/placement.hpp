// SPDX-License-Identifier: MIT
// format.flerp.gguf_placed: which tensors of a GGUF file a rank holds, and where in that rank's one device
// allocation each sits (docs/PHASE5-FOURNODE-DESIGN.md sections 3.1 and 3.2).
//
// A plan is a pure function of (GGUF header, world, mode): every rank computes the same plan from its own mapping
// of the file and the ranks compare its hash. Nothing here touches a device or reads a tensor's bytes.
//
// Mode `stage`: rank r holds the blocks [cuts[r], cuts[r+1]) with all their tensors; token_embd is on rank 0,
// output_norm and output on the last rank (with a tied head, token_embd there too); blocks past the stack
// (<arch>.nextn_predict_layers, the MTP head) and tensors that belong to nothing are not loaded. The cuts minimise
// the largest rank by bytes, then the sum of the squared rank sizes, then come as early as possible - integers
// only, so gate/gguf_census.py (best_cuts, stage_plan, plan_text) computes the same plan and the same hash.
// World 1 with a block range is the plan of one GPU holding a part of the model: the single-GPU gates of a range.
//
// In a rank's allocation the held tensors lie in file order, each at a multiple of kPlanAlign, followed by
// kPlanTail bytes.
#ifndef FLERP_GGUF_PLACED_PLACEMENT_HPP_
#define FLERP_GGUF_PLACED_PLACEMENT_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace flerp::gguf_placed {

inline constexpr std::uint64_t kPlanAlign = 64;
inline constexpr std::uint64_t kPlanTail = 4096;

struct PlanEntry {
  std::string name;
  std::uint32_t type_id{0};        // the ggml type id of the header
  std::uint64_t bytes{0};          // encoded bytes of the tensor
  std::uint64_t file_offset{0};    // absolute, in the file
  std::uint64_t device_offset{0};  // in the rank's allocation
  const void* host{nullptr};       // the tensor's address in the reader's mapping; never dereferenced here
};

struct RankPlan {
  std::uint32_t rank{0};
  std::uint32_t block_begin{0};
  std::uint32_t block_end{0};
  std::uint64_t tensor_bytes{0};   // sum of the entries' bytes
  std::uint64_t device_bytes{0};   // bytes of the one allocation: padded tensors plus kPlanTail
  std::vector<PlanEntry> entries;  // file order
};

struct BlockRange {
  std::uint32_t begin{0};
  std::uint32_t end{0};
};

struct Plan {
  std::uint32_t world{1};
  std::uint32_t blocks{0};           // blocks the executor runs (block_count - nextn_predict_layers)
  std::vector<std::uint32_t> cuts;   // world + 1 values
  std::vector<RankPlan> ranks;
  std::uint64_t file_bytes{0};
  bool tied_head{false};
  std::size_t unloaded_tensors{0};
  std::uint64_t unloaded_bytes{0};

  /// The canonical text of the plan: what every rank hashes (plan_text of gate/gguf_census.py).
  [[nodiscard]] std::string Text() const;
  [[nodiscard]] std::string Sha256() const;
};

/// Cuts of block_cost.size() blocks into `world` contiguous non-empty stages; empty when world does not fit.
[[nodiscard]] std::vector<std::uint32_t> BestCuts(std::span<const std::uint64_t> block_cost,
                                                  std::uint64_t first_extra, std::uint64_t last_extra,
                                                  std::uint32_t world);

/// The stage plan of a single-file GGUF. `range` only with world 1.
[[nodiscard]] std::optional<Plan> StagePlan(const gufo::core::GgufReader& reader, std::uint32_t world,
                                            std::optional<BlockRange> range, std::string* error);

}  // namespace flerp::gguf_placed

#endif  // FLERP_GGUF_PLACED_PLACEMENT_HPP_
