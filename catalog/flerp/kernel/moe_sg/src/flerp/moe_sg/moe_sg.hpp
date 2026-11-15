// SPDX-License-Identifier: MIT
// Stream-grid decode experts for one token (ORNITH-CRUCIBLE item 1, gate/ornith-crucible.toml): the four
// IndexedExperts launches of a qwen35moe block's decode step (routed gate/up + SwiGLU, routed down, shared gate/up,
// shared down; flerp/qwen35moe/moe_ffn.cpp) as ONE launch, the recipe of Crucible's kernel.crucible.moe_decode
// (catalog/crucible/kernel/moe_decode: engine/layers/moe_sg.h, moe_shared_first.h) re-expressed for row-major GGUF
// Q4_K / Q6_K stacks:
//
//   persistent blocks of 8 waves; every wave claims one unit at a time from a global counter, a unit being 16 output
//   rows x the whole K of one matrix of one job (gate/up units: 16 rows of gate AND up, the SwiGLU product written;
//   down units: 16 rows of down); no block barrier after the start (the block copies x into LDS once);
//   jobs: the shared expert first (moe_shared_first.h), then the routed slots in slot order; all gate/up units are
//   claimed before any down unit, and a down unit waits (spin, s_sleep) until its job's gate/up units are counted, so
//   the wait cannot deadlock (every unit it waits for is already claimed by a running wave, and gate/up units never
//   wait); the last wave out zeroes the control words for the next launch (graph-capturable, no host decision).
//
// Arithmetic: every output row is computed exactly as kernel.flerp.moe_kquant's wide reader computes it
// (moe_kquant.hip RowDotWide<kQ4K / kQ6K>: lane L owns sub-blocks L, L + 32, ... in that order, the same decode,
// sumf += scale * dot - offset * xsum, the same xor reduction; SwiGLU (a / (1 + expf(-a))) * b), so the outputs are
// bit-identical to the four IndexedExperts launches (fixed lane order; checked by gate/ornith_crucible_bench.hip).
// A wave runs G rows of a unit side by side (each row keeps its own accumulator), which changes when bytes arrive,
// never the order of any row's sums.
//
// Output layout = the old launches': act[slot * ff + m], shared_act[m], expert_out[slot * hidden + m], shared_out[m].
#ifndef FLERP_MOE_SG_HPP_
#define FLERP_MOE_SG_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace flerp::moe_sg {

enum class WType : int { kQ4K = 0, kQ6K = 1 };

struct Stack {
  const void* data{nullptr};  // [experts][rows][k] row-major GGUF blocks (a stack of one for the shared expert)
  WType type{WType::kQ4K};
};

struct Layer {
  Stack gate, up, down;       // routed stacks; gate and up of one type
  Stack sgate, sup, sdown;    // shared expert (sgate.data null: none); sgate and sup of one type
  int hidden{0}, ff{0}, shared_ff{0}, used{0}, experts{0};
};

/// The shapes this kernel takes: hidden and ff multiples of 512 with hidden / 512 and ff / 512 in {1, 2, 4, 8},
/// shared_ff == ff when there is a shared expert, hidden <= 8192, used <= 64, every stack 16-byte aligned (the
/// conditions of moe_kquant's wide route), x 16-byte aligned. Anything else: false (use IndexedExperts).
[[nodiscard]] bool Supported(const Layer& layer, const void* x) noexcept;

/// Device bytes of the control words (zeroed once by the caller; every launch leaves them zeroed).
[[nodiscard]] std::size_t CtlBytes(const Layer& layer) noexcept;

/// Variants (rows a wave runs side by side gate/up : down, sub-block rounds unrolled): 0 = 1:4 x1, 1 = 1:4 x2,
/// 2 = 2:8 x1, 3 = 1:8 x1. Same bits.
constexpr int kVariants = 4;

/// One token: ids [used] (an id outside [0, experts) reads the last expert, as IndexedExperts), x F32 [hidden].
void Experts(const Layer& layer, const std::int32_t* ids, const float* x, float* act, float* shared_act,
             float* expert_out, float* shared_out, unsigned* ctl, hipStream_t stream, int variant = 0);

/// The persistent grid of a variant (resident blocks of that kernel instance on this device).
[[nodiscard]] int Grid(const Layer& layer, int variant);

}  // namespace flerp::moe_sg

#endif  // defined(ENGINE_ENABLE_HIP)
#endif  // FLERP_MOE_SG_HPP_
