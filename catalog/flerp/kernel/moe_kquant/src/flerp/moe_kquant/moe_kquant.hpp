// SPDX-License-Identifier: MIT
// Mixture-of-experts kernels on GGUF tensors as llama.cpp lays them out for qwen35moe and its relatives:
// a router matrix in F32 and expert stacks whose last GGUF dimension is the expert, so expert e of a stack is one
// contiguous row-major [rows, row_elements] matrix at byte offset e * expert_bytes.
//
// Every size is an argument. Nothing here knows a model: the caller reads the sizes from the GGUF header.
// All launches are asynchronous on `stream`, read only device memory and take no decision on the host, so a
// sequence of them can be captured in a HIP graph.
//
// The weight dot products are gufo's: QuantWarpBlockDot (src/families/qwen/hip_common/quant_ops.hpp), the row
// reader of its decode GEMV, one wave a row. So an expert projection here has the summation order of gufo's
// LaunchQ8KBlockGEMV on the same row, at any number of tokens.
#ifndef FLERP_MOE_KQUANT_HPP_
#define FLERP_MOE_KQUANT_HPP_

#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace flerp::moe {

/// Weight types IndexedExperts reads: Q8_0, Q8_K, the K-quants and IQ formats gufo decodes sixteen values at a
/// time (Q3_K Q4_K Q5_K Q6_K IQ4_NL IQ4_XS IQ3_S), F32, BF16, F16.
[[nodiscard]] bool SupportsExpertType(gufo::core::GgmlType type) noexcept;

/// Bytes of one expert of a stack: `rows` rows of `row_elements` values in `type`. 0 when the type is not
/// supported or a quantized row is not a whole number of blocks.
[[nodiscard]] std::size_t ExpertBytes(gufo::core::GgmlType type, std::size_t rows,
                                      std::size_t row_elements) noexcept;

/// Router logits, and the shared expert's gate logit beside them.
///   logits[t * stride + e] = w_router[e] . x[t]           e < n_experts
///   logits[t * stride + n_experts] = w_shared_gate . x[t] when w_shared_gate is not null (stride > n_experts)
/// w_router is F32 [n_experts, hidden], w_shared_gate F32 [hidden], x F32 [tokens, hidden]. One wave a dot,
/// lanes take every 32nd element, FP32.
void GateLogits(const float* w_router, const float* w_shared_gate, const float* x, float* logits,
                std::size_t tokens, std::size_t n_experts, std::size_t hidden, std::size_t stride,
                hipStream_t stream);

/// One projection, or one gate/up pair with SwiGLU, of the experts each token selected.
///   slot s of token t uses expert e = ids[t * slots + s] (ids null: expert 0, a stack of one - the shared expert)
///   input row       x[(t * slots + s) / x_share]      x_share = slots: the slots of a token read the token's row
///                                                      x_share = 1:     every slot has its own row
///   w_b null        y[(t * slots + s) * rows + m] = w_a[e][m] . x_row
///   w_b not null    y[...] = silu(w_a[e][m] . x_row) * (w_b[e][m] . x_row)        (w_a gate, w_b up)
/// An id outside [0, n_experts) is read as n_experts - 1 (a router fed NaN must not read outside the stack).
void IndexedExperts(const void* w_a, gufo::core::GgmlType type_a, const void* w_b, gufo::core::GgmlType type_b,
                    const std::int32_t* ids, const float* x, float* y, std::size_t tokens, std::size_t slots,
                    std::size_t x_share, std::size_t rows, std::size_t row_elements, std::size_t n_experts,
                    hipStream_t stream);

}  // namespace flerp::moe

#endif  // defined(ENGINE_ENABLE_HIP)
#endif  // FLERP_MOE_KQUANT_HPP_
