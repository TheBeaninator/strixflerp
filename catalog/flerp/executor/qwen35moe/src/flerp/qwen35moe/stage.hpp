// SPDX-License-Identifier: MIT
// executor.flerp.qwen35moe as one stage of a model (docs/PHASE5-FOURNODE-DESIGN.md sections 3.1, 3.2, 3.5): how it
// is told to load through format.flerp.gguf_placed and to run a block range, and how a stage's hidden rows get in
// and out. The origin's executor API (src/models/qwen/hip/executor.hpp) is unchanged, so these are settings of the
// process, made before the model is created and before the first forward.
//
//   LoadOptions   placed = false: the origin's loader, the whole file in one device region (the default).
//                 placed = true:  the stage plan of (header, world, rank[, block range]); one allocation of the
//                 plan's bytes, tensors filled by byte-range reads, tensors outside the plan dropped. The loaded
//                 model runs blocks [block_begin, block_end) only (QwenModelWeights::BlockBegin/BlockEnd).
//   StageIo       device buffers, constant for the life of an executor (a captured decode graph bakes them):
//                 d_input   [rows, hidden] FP32 read in place of the embedding lookup. Required when the range does
//                           not start at block 0; with a range that does, it replaces the token embeddings.
//                 d_output  [rows, hidden] FP32 written with the hidden rows after the last block of the range.
//                 A decode step uses row 0; a prompt chunk uses one row a token.
//   A range that does not end at the last block has no head: a forward asked for logits writes token 0.
//
// The decode graph's workload identity folds the block range and the stage buffers (hip/decode.cpp, adapted), so a
// graph captured for one range or one set of buffers is never replayed for another.
//
// Only ForwardPromptBatch / ForwardToken honour a range (prefill_chunk.cpp, decode_step.cpp). The batched decode,
// the verification paths and the CPU reference run the whole stack and must not be called on a partial range.
#ifndef FLERP_QWEN35MOE_STAGE_HPP_
#define FLERP_QWEN35MOE_STAGE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "flerp/gguf_placed/placed_loader.hpp"

namespace flerp::qwen35moe {

struct LoadOptions {
  bool placed{false};
  std::uint32_t world{1};
  std::uint32_t rank{0};
  bool has_range{false};          // world 1 only: blocks [block_begin, block_end)
  std::uint32_t block_begin{0};
  std::uint32_t block_end{0};
  flerp::gguf_placed::FillPolicy fill;   // how the placed loader reads (placed_loader.hpp)
};
void SetLoadOptions(const LoadOptions& options) noexcept;
[[nodiscard]] const LoadOptions& GetLoadOptions() noexcept;

/// What the most recent QwenGpuModel::CreateFromGguf did (placed == false for the origin's loader).
[[nodiscard]] const flerp::gguf_placed::LoadReport& LastLoadReport() noexcept;
void SetLastLoadReport(const flerp::gguf_placed::LoadReport& report);

struct StageIo {
  const float* d_input{nullptr};
  float* d_output{nullptr};
};
void SetStageIo(const StageIo& io) noexcept;
[[nodiscard]] const StageIo& GetStageIo() noexcept;

// Stage P2: several stages in one process (the loopback), each an executor over its own QwenModelWeights (a view of
// one loaded model with its own block range). BindStageIo ties buffers to one weights object; the decode step and
// the prompt chunk ask GetStageIo(&weights), which gives the bound buffers, else the process-wide ones above.
void BindStageIo(const void* weights, const StageIo& io);
void UnbindStageIo(const void* weights) noexcept;
[[nodiscard]] StageIo GetStageIo(const void* weights) noexcept;

// Decode graph switch (hip/decode.cpp, adapted): false runs every decode step eagerly. Read at each step; a
// captured graph stays captured and is used again once the switch is back on. Default on (the origin's behaviour).
void SetDecodeGraph(bool enabled) noexcept;
[[nodiscard]] bool DecodeGraphEnabled() noexcept;

// ORNITH-O3 step two (gate/ornith-o3-spec.toml): the recurrent ring of a stage that verifies drafts as back-to-back
// single-row decode steps. SetRecurrentRing(executor, slots) with slots >= 2 runs every decode step of that
// QwenGpuExecutor on a ring of `slots` recurrent states (GDN conv window + delta state of every recurrent layer):
//   slot(p) = (p - p0) mod slots, p0 = the first decode position after the last ResetRecurrentRing;
//   the step at position p reads slot(p) and writes slot(p + 1) (the kernels' diverging pass: no copy);
//   slot 0 is the arena's own state, which the prompt chunks write in place; slots 1.. are allocated by the first
//   step that needs them (hip/decode.cpp) and kept until ReleaseRecurrentRing.
// So a round of k + 1 steps at p .. p + k that keeps n + 1 of them needs no keep call: the next step is at p + n + 1
// and reads the slot step n wrote. With slots = k + 1 a round never overwrites a slot a later step reads (it
// overwrites slot(p), which step 0 read and nothing reads again). The KV cache needs nothing either: attention reads
// positions <= its own and every kept position is written by the step that keeps it. Each (read slot, write slot)
// pair has its own captured decode graph (the graph bakes the state pointers). slots 0 or 1: off, the origin's
// in-place state and its one graph, unchanged.
struct RecurrentRing {
  std::uint32_t slots{0};
  bool started{false};
  std::uint32_t base{0};            // p0
  std::vector<float*> conv;         // slots 1..: device buffers
  std::vector<void*> delta;
  std::size_t bytes{0};             // device bytes of those
  std::shared_ptr<void> graphs;     // the pairs' decode graphs (hip/decode.cpp owns the type)
  std::uint64_t captures{0};        // decode graphs captured on the ring
  std::uint64_t steps{0};           // decode steps run on the ring
};
void SetRecurrentRing(const void* executor, std::uint32_t slots);
/// The next decode step is the ring's position 0 (call it with the executor's Reset).
void ResetRecurrentRing(const void* executor) noexcept;
/// Frees the slots and the graphs; call it before the executor is destroyed.
void ReleaseRecurrentRing(const void* executor) noexcept;
/// nullptr when the executor has no ring entry.
[[nodiscard]] RecurrentRing* FindRecurrentRing(const void* executor) noexcept;

}  // namespace flerp::qwen35moe

#endif  // FLERP_QWEN35MOE_STAGE_HPP_
