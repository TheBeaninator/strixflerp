// SPDX-License-Identifier: MIT
// See moe_kquant.hpp. Wave32 kernels, four waves a block, one wave an output element.
#if defined(ENGINE_ENABLE_HIP)
#include "flerp/moe_kquant/moe_kquant.hpp"

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

namespace flerp::moe {
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

// Q4_K and Q6_K rows with wide loads (SHAPE-TRANSFERS R1 arm a, gate/xfer-r1-fold.toml). The lane mapping and the
// arithmetic are QuantWarpBlockDot's sub16 route (DecodeQuantSub16, then sumf += scale*dot - offset*xsum, then the
// xor reduction), so every output has the same summation order and the same bits; what changes is how the bytes
// arrive: one 16-byte load for the Q4_K codes (blocks are 144 = 9 x 16 bytes, so aligned when the stack is),
// two 16-byte loads (ql, qh) for Q6_K, and four 16-byte loads for the sixteen activations. The host takes these
// routes only when the stack, and x, are 16-byte aligned (Q6_K blocks are 210 bytes: its loads are 2-byte aligned,
// which the hardware serves; the compiler is told the alignment so it does not split them into bytes).
enum class Route : int { kGeneric = 0, kQ4K = 1, kQ6K = 2 };

__device__ inline void LoadX16(const float* __restrict__ xb, float xv[16]) {
  const float4* x4 = reinterpret_cast<const float4*>(xb);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float4 v = x4[i];
    xv[4 * i] = v.x;
    xv[4 * i + 1] = v.y;
    xv[4 * i + 2] = v.z;
    xv[4 * i + 3] = v.w;
  }
}

// Sixteen code bytes at a 2-byte-aligned address as four words.
__device__ inline uint4 Load16Align2(const std::uint8_t* p) {
  uint4 v;
  __builtin_memcpy(&v, __builtin_assume_aligned(p, 2), 16);
  return v;
}

template <Route R>
__device__ inline float RowDotWide(const std::uint8_t* __restrict__ row, const float* __restrict__ x, std::size_t k,
                                   std::size_t lane) {
  float sumf = 0.0F;
  const std::size_t nsub = k / 16;
  for (std::size_t sb = lane; sb < nsub; sb += 32) {
    gufo::hip::QuantSub16 sub;
    const std::size_t sb32 = (sb / 2) % 8;
    if constexpr (R == Route::kQ4K) {
      const std::uint8_t* blk = row + (sb / 16) * sizeof(gufo::hip::Q4KBlock);
      const std::size_t base = ((sb32 / 2) * 32) + ((sb % 2) * 16);
      const unsigned shift = 4U * static_cast<unsigned>(sb32 & 1U);
      const uint4 p = *reinterpret_cast<const uint4*>(blk + offsetof(gufo::hip::Q4KBlock, qs) + base);
      sub.w[0] = (p.x >> shift) & 0x0F0F0F0FU;
      sub.w[1] = (p.y >> shift) & 0x0F0F0F0FU;
      sub.w[2] = (p.z >> shift) & 0x0F0F0F0FU;
      sub.w[3] = (p.w >> shift) & 0x0F0F0F0FU;
      std::uint8_t sc = 0;
      std::uint8_t m = 0;
      gufo::hip::GetQKScaleMin(sb32, blk + offsetof(gufo::hip::Q4KBlock, scales), sc, m);
      const __half* dh = reinterpret_cast<const __half*>(blk);
      sub.scale = __half2float(dh[0]) * static_cast<float>(sc);
      sub.offset = __half2float(dh[1]) * static_cast<float>(m);
    } else {
      const std::uint8_t* blk = row + (sb / 16) * sizeof(gufo::hip::Q6KBlock);
      const std::size_t half = sb32 / 4;
      const std::size_t segment = sb32 % 4;
      const std::size_t lane0 = (sb % 2) * 16;
      const std::size_t ql_base = (half * 64) + (((segment & 1U) != 0U) ? 32 : 0);
      const unsigned ql_shift = (segment >= 2) ? 4U : 0U;
      const unsigned qh_shift = 2U * static_cast<unsigned>(segment);
      const uint4 lo = Load16Align2(blk + offsetof(gufo::hip::Q6KBlock, ql) + ql_base + lane0);
      const uint4 hi = Load16Align2(blk + offsetof(gufo::hip::Q6KBlock, qh) + (half * 32) + lane0);
      const std::uint32_t low_bits[4] = {lo.x, lo.y, lo.z, lo.w};
      const std::uint32_t high_bits[4] = {hi.x, hi.y, hi.z, hi.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const std::uint32_t low = (low_bits[i] >> ql_shift) & 0x0F0F0F0FU;
        const std::uint32_t high = (high_bits[i] >> qh_shift) & 0x03030303U;
        sub.w[i] = gufo::hip::SubBytes(low | (high << 4U), 0x20202020U);
      }
      const std::size_t scale_index = (half * 8) + (segment * 2) + (sb % 2);
      const auto* scales = reinterpret_cast<const std::int8_t*>(blk + offsetof(gufo::hip::Q6KBlock, scales));
      const __half* dh = reinterpret_cast<const __half*>(blk + offsetof(gufo::hip::Q6KBlock, d));
      sub.scale = __half2float(*dh) * static_cast<float>(scales[scale_index]);
      sub.offset = 0.0F;
    }
    float xv[16];
    LoadX16(x + (sb * 16), xv);
    float dot = 0.0F;
    float xsum = 0.0F;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      dot += static_cast<float>(sub.q[j]) * xv[j];
      xsum += xv[j];
    }
    sumf += (sub.scale * dot) - (sub.offset * xsum);
  }
  return WaveSum(sumf);
}

template <Route R>
__device__ inline float RowDotRoute(GgmlType type, const void* __restrict__ base, std::size_t row,
                                    const float* __restrict__ x, std::size_t k, std::size_t lane) {
  if constexpr (R == Route::kGeneric) {
    return RowDot(type, base, row, x, k, lane);
  } else {
    constexpr std::size_t kBlockBytes =
        R == Route::kQ4K ? sizeof(gufo::hip::Q4KBlock) : sizeof(gufo::hip::Q6KBlock);
    return RowDotWide<R>(static_cast<const std::uint8_t*>(base) + row * (k / 256 * kBlockBytes), x, k, lane);
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

// GateLogitsKernel's arithmetic with wide loads (DECODE-KERN3, gate/decode-kern3.toml). Lane L still sums the
// elements L, L+32, L+64, ... of its row in that order with the same `dot += w * x`, then the same xor reduction, so
// the bits are the same; what changes is how the bytes arrive. The block first copies its token's x row into LDS with
// 16-byte loads. Each wave then issues its row as 16-byte loads, kRouterGroup float4 a lane (1,024 floats) in flight
// at a time (the old kernel had one 4-byte load a lane in flight and ran at ~50 GB/s cache-cold), puts them through a
// per-wave LDS slice so that lane L reads back exactly its old elements, and sums. Four waves a block: 165 GB/s
// cache-cold on the 35B router against 157 at two waves, and 2,048 floats in flight measured no better
// (gate/results/decode-kern3/sweep.jsonl). The host takes this route when hidden is a multiple of kRouterGroup * 128,
// hidden <= kRouterMaxHidden, and the weights and x are 16-byte aligned.
constexpr std::uint32_t kRouterWaves = 4;
constexpr std::uint32_t kRouterThreads = kRouterWaves * 32;
constexpr std::size_t kRouterGroup = 8;  // float4 a lane in flight = 1,024 floats a wave
constexpr std::size_t kRouterMaxHidden = 8192;  // LDS: (8192 + 4 x 1024) floats = 48 KiB

__launch_bounds__(kRouterThreads) __global__
    void GateLogitsWideKernel(const float* __restrict__ w_router, const float* __restrict__ w_shared_gate,
                              const float* __restrict__ x, float* __restrict__ logits, std::size_t n_experts,
                              std::size_t hidden, std::size_t stride) {
  extern __shared__ float4 router_lds[];
  float* xs = reinterpret_cast<float*>(router_lds);
  const std::size_t lane = threadIdx.x & 31U;
  const std::size_t wave = threadIdx.x >> 5U;
  const std::size_t r = (static_cast<std::size_t>(blockIdx.x) * kRouterWaves) + wave;
  const std::size_t t = blockIdx.y;
  const float4* x4 = reinterpret_cast<const float4*>(x + t * hidden);
  for (std::size_t i = threadIdx.x; i < hidden / 4; i += kRouterThreads) {
    router_lds[i] = x4[i];
  }
  __syncthreads();
  const float* w = nullptr;
  if (r < n_experts) {
    w = w_router + r * hidden;
  } else if (r == n_experts && w_shared_gate != nullptr) {
    w = w_shared_gate;
  } else {
    return;
  }
  float* stage = xs + hidden + (wave * kRouterGroup * 128);
  float4* stage4 = reinterpret_cast<float4*>(stage);
  const float4* w4 = reinterpret_cast<const float4*>(w);
  float dot = 0.0F;
  for (std::size_t c0 = 0; c0 < hidden / 128; c0 += kRouterGroup) {
    float4 v[kRouterGroup];
#pragma unroll
    for (std::size_t g = 0; g < kRouterGroup; ++g) {
      v[g] = w4[((c0 + g) * 32) + lane];  // floats (c0+g)*128 + 4*lane .. +3
    }
#pragma unroll
    for (std::size_t g = 0; g < kRouterGroup; ++g) {
      stage4[(g * 32) + lane] = v[g];
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#pragma unroll
    for (std::size_t g = 0; g < kRouterGroup; ++g) {
#pragma unroll
      for (std::size_t q = 0; q < 4; ++q) {
        const std::size_t i = ((c0 + g) * 128) + (q * 32) + lane;  // the old loop's i, in the old order
        dot += stage[(g * 128) + (q * 32) + lane] * xs[i];
      }
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
  }
  dot = WaveSum(dot);
  if (lane == 0) {
    logits[t * stride + r] = dot;
  }
}

template <Route R>
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
  const float a = RowDotRoute<R>(type_a, static_cast<const char*>(w_a) + e * bytes_a, m, xr, row_elements, lane);
  float out = a;
  if (w_b != nullptr) {
    const float b = RowDotRoute<R>(type_b, static_cast<const char*>(w_b) + e * bytes_b, m, xr, row_elements, lane);
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
  const auto aligned16 = [](const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15U) == 0; };
  if (hidden % (kRouterGroup * 128) == 0 && hidden <= kRouterMaxHidden && aligned16(w_router) &&
      (w_shared_gate == nullptr || aligned16(w_shared_gate)) && aligned16(x)) {
    const auto wide_x = static_cast<std::uint32_t>((outputs + kRouterWaves - 1) / kRouterWaves);
    const std::size_t lds = (hidden + (kRouterWaves * kRouterGroup * 128)) * sizeof(float);
    for (std::size_t first = 0; first < tokens; first += kMaxGridRows) {
      const std::size_t n = std::min(kMaxGridRows, tokens - first);
      GateLogitsWideKernel<<<dim3(wide_x, static_cast<std::uint32_t>(n)), dim3(kRouterThreads), lds, stream>>>(
          w_router, w_shared_gate, x + first * hidden, logits + first * stride, n_experts, hidden, stride);
    }
    Check("GateLogits");
    return;
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
  // The wide routes: both stacks of the one type, a whole number of 256-value blocks a row (always, for these types),
  // the stacks and x 16-byte aligned (then every expert, row and x row is: 144-byte Q4_K rows and 256-float x rows
  // are multiples of 16; Q6_K rows only of 2, which its loads allow). Anything else takes the generic reader.
  const auto aligned16 = [](const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15U) == 0; };
  const bool same_type = w_b == nullptr || type_b == type_a;
  const bool wide_ok = same_type && aligned16(w_a) && (w_b == nullptr || aligned16(w_b)) && aligned16(x) &&
                       row_elements % 256 == 0;
  Route route = Route::kGeneric;
  if (wide_ok && type_a == GgmlType::kQ4_K) {
    route = Route::kQ4K;
  } else if (wide_ok && type_a == GgmlType::kQ6_K) {
    route = Route::kQ6K;
  }
  const auto grid_x = static_cast<std::uint32_t>((rows + kWaves - 1) / kWaves);
  const std::size_t total = tokens * slots;
  for (std::size_t first = 0; first < total; first += kMaxGridRows) {
    const std::size_t n = std::min(kMaxGridRows, total - first);
    const dim3 grid(grid_x, static_cast<std::uint32_t>(n));
    switch (route) {
      case Route::kQ4K:
        IndexedExpertsKernel<Route::kQ4K><<<grid, dim3(kThreads), 0, stream>>>(
            w_a, type_a, bytes_a, w_b, type_b, bytes_b, ids, x, y, slots, x_share, rows, row_elements, n_experts, first);
        break;
      case Route::kQ6K:
        IndexedExpertsKernel<Route::kQ6K><<<grid, dim3(kThreads), 0, stream>>>(
            w_a, type_a, bytes_a, w_b, type_b, bytes_b, ids, x, y, slots, x_share, rows, row_elements, n_experts, first);
        break;
      default:
        IndexedExpertsKernel<Route::kGeneric><<<grid, dim3(kThreads), 0, stream>>>(
            w_a, type_a, bytes_a, w_b, type_b, bytes_b, ids, x, y, slots, x_share, rows, row_elements, n_experts, first);
        break;
    }
  }
  Check("IndexedExperts");
}

}  // namespace flerp::moe
#endif  // defined(ENGINE_ENABLE_HIP)
