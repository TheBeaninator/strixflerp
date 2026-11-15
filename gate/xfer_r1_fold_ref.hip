// SHAPE-TRANSFERS R1 fold: the reference. kernel.flerp.moe_kquant's moe_kquant.hip exactly as at commit 504ca20,
// except the namespace (flerp_ref::moe) and no include of moe_kquant.hpp, so it links beside the new source.
// SPDX-License-Identifier: MIT
// See moe_kquant.hpp. Wave32 kernels, four waves a block, one wave an output element.
#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <gufo/ext/experimental/ggml_dequant.hpp>
#include <iostream>

#include "src/core/gguf_reader.hpp"
#include "src/families/qwen/hip_common/quant_ops.hpp"

namespace flerp_ref::moe {
namespace {

using gufo::core::GgmlType;

constexpr std::uint32_t kWaves = 4;
constexpr std::uint32_t kThreads = kWaves * 32;
// A grid dimension of the HIP runtime is at most 65535 blocks: token x slot rows go out in chunks of this many.
constexpr std::size_t kMaxGridRows = 32768;

__device__ inline float WaveSum(float v) {
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor(v, off);
  }
  return v;
}

// row . x for one row of one expert, by the 32 lanes of a wave. Every lane returns the full sum for the plain
// float formats; for the block formats gufo's reader returns it in lane 0 and 0 elsewhere. Callers use lane 0.
__device__ inline float RowDot(GgmlType type, const void* __restrict__ base, std::size_t row,
                               const float* __restrict__ x, std::size_t k, std::size_t lane) {
  switch (type) {
    case GgmlType::kF32: {
      const float* w = static_cast<const float*>(base) + row * k;
      float dot = 0.0F;
      for (std::size_t i = lane; i < k; i += 32) {
        dot += w[i] * x[i];
      }
      return WaveSum(dot);
    }
    case GgmlType::kBF16: {
      const hip_bfloat16* w = static_cast<const hip_bfloat16*>(base) + row * k;
      float dot = 0.0F;
      for (std::size_t i = lane; i < k; i += 32) {
        dot += static_cast<float>(w[i]) * x[i];
      }
      return WaveSum(dot);
    }
    case GgmlType::kF16: {
      const __half* w = static_cast<const __half*>(base) + row * k;
      float dot = 0.0F;
      for (std::size_t i = lane; i < k; i += 32) {
        dot += __half2float(w[i]) * x[i];
      }
      return WaveSum(dot);
    }
    default:
      return gufo::hip::QuantWarpBlockDot<float>(type, base, row, x, k, lane, 0, 1);
  }
}

__launch_bounds__(kThreads) __global__
    void GateLogitsKernel(const float* __restrict__ w_router, const float* __restrict__ w_shared_gate,
                          const float* __restrict__ x, float* __restrict__ logits, std::size_t n_experts,
                          std::size_t hidden, std::size_t stride) {
  const std::size_t lane = threadIdx.x & 31U;
  const std::size_t r = (static_cast<std::size_t>(blockIdx.x) * kWaves) + (threadIdx.x >> 5U);
  const std::size_t t = blockIdx.y;
  const float* w = nullptr;
  if (r < n_experts) {
    w = w_router + r * hidden;
  } else if (r == n_experts && w_shared_gate != nullptr) {
    w = w_shared_gate;
  } else {
    return;
  }
  const float* xr = x + t * hidden;
  float dot = 0.0F;
  for (std::size_t i = lane; i < hidden; i += 32) {
    dot += w[i] * xr[i];
  }
  dot = WaveSum(dot);
  if (lane == 0) {
    logits[t * stride + r] = dot;
  }
}

__launch_bounds__(kThreads) __global__
    void IndexedExpertsKernel(const void* __restrict__ w_a, GgmlType type_a, std::size_t bytes_a,
                              const void* __restrict__ w_b, GgmlType type_b, std::size_t bytes_b,
                              const std::int32_t* __restrict__ ids, const float* __restrict__ x,
                              float* __restrict__ y, std::size_t slots, std::size_t x_share, std::size_t rows,
                              std::size_t row_elements, std::size_t n_experts, std::size_t first_row) {
  const std::size_t lane = threadIdx.x & 31U;
  const std::size_t m = (static_cast<std::size_t>(blockIdx.x) * kWaves) + (threadIdx.x >> 5U);
  if (m >= rows) {
    return;
  }
  const std::size_t ts = first_row + blockIdx.y;         // token * slots + slot
  std::size_t e = 0;
  if (ids != nullptr) {
    const std::int32_t id = ids[ts];
    e = (id < 0 || static_cast<std::size_t>(id) >= n_experts) ? n_experts - 1 : static_cast<std::size_t>(id);
  }
  (void)slots;
  const float* xr = x + (ts / x_share) * row_elements;
  const float a = RowDot(type_a, static_cast<const char*>(w_a) + e * bytes_a, m, xr, row_elements, lane);
  float out = a;
  if (w_b != nullptr) {
    const float b = RowDot(type_b, static_cast<const char*>(w_b) + e * bytes_b, m, xr, row_elements, lane);
    out = (a / (1.0F + expf(-a))) * b;
  }
  if (lane == 0) {
    y[ts * rows + m] = out;
  }
}

void Check(const char* what) {
  const hipError_t status = hipGetLastError();
  if (status != hipSuccess) {
    std::cerr << "flerp::moe: " << what << ": " << hipGetErrorString(status) << "\n";
    std::abort();
  }
}

}  // namespace

bool SupportsExpertType(GgmlType type) noexcept {
  switch (type) {
    case GgmlType::kF32:
    case GgmlType::kBF16:
    case GgmlType::kF16:
    case GgmlType::kQ8_0:
    case GgmlType::kQ8_K:
    case GgmlType::kQ3_K:
    case GgmlType::kQ4_K:
    case GgmlType::kQ5_K:
    case GgmlType::kQ6_K:
    case GgmlType::kIQ4_NL:
    case GgmlType::kIQ4_XS:
    case GgmlType::kIQ3_S:
      return true;
    default:
      return false;
  }
}

std::size_t ExpertBytes(GgmlType type, std::size_t rows, std::size_t row_elements) noexcept {
  if (!SupportsExpertType(type)) {
    return 0;
  }
  if (type == GgmlType::kF32) {
    return rows * row_elements * sizeof(float);
  }
  if (type == GgmlType::kBF16 || type == GgmlType::kF16) {
    return rows * row_elements * sizeof(std::uint16_t);
  }
  return rows * gufo::quant::QuantizedRowBytes(type, row_elements);
}

void GateLogits(const float* w_router, const float* w_shared_gate, const float* x, float* logits, std::size_t tokens,
                std::size_t n_experts, std::size_t hidden, std::size_t stride, hipStream_t stream) {
  if (tokens == 0 || n_experts == 0) {
    return;
  }
  const std::size_t outputs = n_experts + (w_shared_gate != nullptr ? 1 : 0);
  if (stride < outputs) {
    std::cerr << "flerp::moe::GateLogits: stride " << stride << " is below " << outputs << " outputs\n";
    std::abort();
  }
  const auto grid_x = static_cast<std::uint32_t>((outputs + kWaves - 1) / kWaves);
  for (std::size_t first = 0; first < tokens; first += kMaxGridRows) {
    const std::size_t n = std::min(kMaxGridRows, tokens - first);
    GateLogitsKernel<<<dim3(grid_x, static_cast<std::uint32_t>(n)), dim3(kThreads), 0, stream>>>(
        w_router, w_shared_gate, x + first * hidden, logits + first * stride, n_experts, hidden, stride);
  }
  Check("GateLogits");
}

void IndexedExperts(const void* w_a, GgmlType type_a, const void* w_b, GgmlType type_b, const std::int32_t* ids,
                    const float* x, float* y, std::size_t tokens, std::size_t slots, std::size_t x_share,
                    std::size_t rows, std::size_t row_elements, std::size_t n_experts, hipStream_t stream) {
  if (tokens == 0 || slots == 0 || rows == 0) {
    return;
  }
  const std::size_t bytes_a = ExpertBytes(type_a, rows, row_elements);
  const std::size_t bytes_b = w_b != nullptr ? ExpertBytes(type_b, rows, row_elements) : 0;
  if (bytes_a == 0 || (w_b != nullptr && bytes_b == 0) || x_share == 0 || n_experts == 0) {
    std::cerr << "flerp::moe::IndexedExperts: unsupported weight type or shape (types "
              << gufo::core::ToString(type_a) << ", " << gufo::core::ToString(type_b) << ", rows " << rows
              << ", row elements " << row_elements << ")\n";
    std::abort();
  }
  const auto grid_x = static_cast<std::uint32_t>((rows + kWaves - 1) / kWaves);
  const std::size_t total = tokens * slots;
  for (std::size_t first = 0; first < total; first += kMaxGridRows) {
    const std::size_t n = std::min(kMaxGridRows, total - first);
    IndexedExpertsKernel<<<dim3(grid_x, static_cast<std::uint32_t>(n)), dim3(kThreads), 0, stream>>>(
        w_a, type_a, bytes_a, w_b, type_b, bytes_b, ids, x, y, slots, x_share, rows, row_elements, n_experts, first);
  }
  Check("IndexedExperts");
}

}  // namespace flerp_ref::moe
#endif  // defined(ENGINE_ENABLE_HIP)
