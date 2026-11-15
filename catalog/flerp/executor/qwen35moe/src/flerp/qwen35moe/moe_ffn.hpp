// SPDX-License-Identifier: MIT
// The feed forward of a qwen35moe block on the device, as llama.cpp computes it
// (src/models/qwen35moe.cpp:494-548 build_layer_ffn, src/llama-graph.cpp:2002-2310 build_moe_ffn, at 86a24a1):
//
//   logits  = ffn_gate_inp . x                                   F32 router, one logit an expert
//   probs   = softmax(logits) over all experts                   LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX
//   ids     = the expert_used largest probs
//   w       = probs[ids] / max(sum(probs[ids]), 6.103515625e-5)  norm_w = true; no expert_weights_scale
//   routed  = sum_s w[s] * down_exps[ids[s]] . (silu(gate_exps[ids[s]] . x) * (up_exps[ids[s]] . x))
//   shared  = down_shexp . (silu(gate_shexp . x) * (up_shexp . x))
//   out     = routed + sigmoid(ffn_gate_inp_shexp . x) * shared
//
// Seven launches a block for any number of tokens, none of them decided on the host:
//   flerp::moe::GateLogits              router logits and the shared gate logit      (kernel.flerp.moe_kquant)
//   gufo qwen38_flash_next RouterTopK   softmax, top-k, renormalised weights          (kernel.gufo.qfn_kernels)
//   flerp::moe::IndexedExperts x4       gate/up + SwiGLU and down, routed and shared  (kernel.flerp.moe_kquant)
//   gufo qwen38_flash_next MoeEpilogue  weighted sum, sigmoid gate, shared expert     (kernel.gufo.qfn_kernels)
//
// The intermediates live in the arena's three per-token FFN scratch buffers, which weights.cpp sizes for them
// (QwenMoeDims::ScratchFloats floats a token each). With T tokens, E experts, k used, hidden H, widths F and S:
//   a (ffn.gate)        expert_out [T, k, H]
//   b (ffn.up)          logits [T, E + 1] | ids [T, k] int32 | weights [T, k] | shared_out [T, H]
//   c (ffn.activation)  act [T, k, F] | shared act [T, S]
#ifndef FLERP_QWEN35MOE_MOE_FFN_HPP_
#define FLERP_QWEN35MOE_MOE_FFN_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/models/qwen/state.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace flerp::qwen35moe {

struct MoeScratch {
  float* a{nullptr};
  float* b{nullptr};
  float* c{nullptr};
  std::size_t floats_per_token{0};   // capacity of each buffer, per token
};

/// What the router chose in one block of one traced forward.
struct MoeTraceLayer {
  std::uint32_t layer{0};
  std::size_t tokens{0};
  std::uint32_t used{0};
  std::vector<std::int32_t> ids;        // [tokens, used]
  std::vector<float> weights;           // [tokens, used]
  std::vector<float> shared_gate_logit; // [tokens]; empty without a shared expert
};
struct MoeTrace {
  std::vector<MoeTraceLayer> layers;
};
/// A trace to fill, or null (the default). A traced launch waits for the stream and copies to the host, so only
/// launches that pass traceable = true are traced: prompt chunks, never the graph-captured decode step.
void SetMoeTrace(MoeTrace* trace) noexcept;

/// A dump of whole MoE blocks as the device computed them, for gate/moe_block_check.py: for each launch of a block
/// in `layers` and each row in `rows` that the launch has, one record in the format of the CPU reference's
/// FLERP_MOE_DUMP (forward.cpp): int32 layer, position (position_base + row), hidden, used, experts; x[hidden];
/// ids[used] int32; weights[used]; the shared gate logit; out[hidden]. A dumping launch waits for the stream and
/// copies to the host, so the caller arms it only around forwards that run eagerly (a prompt chunk, or a decode
/// step asked for no logits, which the origin never captures in a graph).
struct MoeDump {
  std::vector<std::uint32_t> layers;
  std::vector<std::uint32_t> rows;
  std::uint32_t position_base{0};
  bool armed{false};
  std::vector<char> bytes;      // the records, appended
  std::size_t records{0};
};
void SetMoeDump(MoeDump* dump) noexcept;

/// PREFILL-ORNITH: the grouped prompt-chunk experts (kernel.flerp.moe_kgrouped) off (0), F16 (1) or exact (2:
/// exact weight codes, hi/lo F16 activations) for this process, or -1 (the default) to follow FLERP_MOE_GROUPED. Lets one load run both arms of a comparison.
void SetMoeGrouped(int mode) noexcept;

/// PREFILL-ORNITH: the prompt chunk's dense projections on Q4_K / Q6_K weights through kgrouped::Linear: off (0), F16
/// (1) or exact (2), or -1 (default) to follow FLERP_DENSE_GROUPED.
void SetDenseGrouped(int mode) noexcept;
int DenseGroupedMode() noexcept;
/// y [rows, n] = x [rows, k] . W^T on the grouped kernel (x FP32, converted here); false (nothing launched) when the
/// type is not Q4_K / Q6_K or k is not whole blocks.
bool LaunchDenseGrouped(const void* w, gufo::core::GgmlType type, const float* x, float* y, std::size_t rows,
                        std::size_t n, std::size_t k, bool exact, hipStream_t stream);
unsigned* DenseOverflowFlag() noexcept;

/// PREFILL-ORNITH attention: prompt-chunk attention through kernel.flerp.attn_wmma_gqa (1) or the origin's chain (0),
/// or -1 (default) to follow FLERP_ATTN_GQA.
void SetAttnGqa(int mode) noexcept;
int AttnGqaMode() noexcept;

/// out[T, H] = the block's feed forward of x[T, H]. Aborts with a message when the scratch is too small.
void LaunchMoeFfn(const gufo::models::QwenLayerWeights& layer, const float* x, float* out, std::size_t tokens,
                  std::size_t hidden, MoeScratch scratch, hipStream_t stream, std::uint32_t layer_index,
                  bool traceable);

}  // namespace flerp::qwen35moe

#endif  // defined(ENGINE_ENABLE_HIP)
#endif  // FLERP_QWEN35MOE_MOE_FFN_HPP_
