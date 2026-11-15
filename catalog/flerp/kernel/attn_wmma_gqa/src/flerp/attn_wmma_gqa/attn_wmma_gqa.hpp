// SPDX-License-Identifier: MIT
// kernel.flerp.attn_wmma_gqa: causal prefill attention (tiled online softmax on v_wmma_f32_16x16x16_f16) for GQA
// shapes other than the 27B's, over the canonical FP16 KV cache plane [layer][position][kv_heads x head_dim] that the
// fused QK-norm/RoPE/KV write already filled for this chunk. See attn_wmma_gqa.hip.
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace flerp::attn {

bool Supported(std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim) noexcept;

// out_context [batch, num_heads x head_dim] = softmax(q k^T / 16, causal) v, times sigmoid(gate) when gate is given
// (qwen35 gated attention). q [batch, num_heads x head_dim] FP32 after norm and RoPE. packed_*_ws: optional idle
// scratch for the head-major copy large chunks take (the origin's FFN workspace); null skips it. false: not launched.
bool LaunchGqaWmmaAttention(const float* q, const float* gate, const void* k_cache_f16, const void* v_cache_f16,
                            float* out_context, std::uint32_t layer_idx, std::uint32_t start_pos,
                            std::size_t batch_size, std::uint32_t max_context, std::uint32_t num_heads,
                            std::uint32_t num_kv_heads, std::uint32_t head_dim, hipStream_t stream,
                            void* packed_k_ws, std::size_t packed_k_bytes, void* packed_v_ws,
                            std::size_t packed_v_bytes);

}  // namespace flerp::attn
