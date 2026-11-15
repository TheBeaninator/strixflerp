// SPDX-License-Identifier: MIT
// DECODE-KERN3 (gate/decode-kern3.toml): the reference. Verbatim copies, as at commit e5af237, of
//   kernel.flerp.moe_kquant          moe_kquant.hip   WaveSum, GateLogitsKernel, GateLogits (+ the constants)
//   kernel.gufo.qwen_linear_decode   gemv_quant.hip   Q8KBlockGEMVKernel, Q8KBlockGEMVKernel_2Rows, LaunchQ8KBlockGEMV
// in namespace kern3_ref (Check reduced to the same abort), so they link beside the new sources. Not edited otherwise.
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "src/core/gguf_reader.hpp"
#include "src/families/qwen/hip_common/quant_ops.hpp"

namespace kern3_ref::moe {
namespace {
using gufo::core::GgmlType;
constexpr std::uint32_t kWaves = 4;
constexpr std::uint32_t kThreads = kWaves * 32;
constexpr std::size_t kMaxGridRows = 32768;
__device__ inline float WaveSum(float v) {
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor(v, off);
  }
  return v;
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

void Check(const char* what) {
  const hipError_t status = hipGetLastError();
  if (status != hipSuccess) {
    std::cerr << "kern3_ref::moe: " << what << ": " << hipGetErrorString(status) << "\n";
    std::abort();
  }
}
}  // namespace

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
}  // namespace kern3_ref::moe

namespace kern3_ref::hip {
using namespace gufo::hip;
namespace core = gufo::core;
template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Q8KBlockGEMVKernel(const void* __restrict__ A, core::GgmlType type,
                            const float* __restrict__ x, float* __restrict__ y,
                            std::size_t M, std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  if (m >= M) {
    return;
  }

  const float sumf = QuantWarpBlockDot(type, A, m, x, K, lane_id, 0, 1);
  if (lane_id == 0) {
    y[m] = sumf;
  }
}

template<std::uint32_t WavesPerBlock,
         core::GgmlType StaticType = core::GgmlType::kF32>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Q8KBlockGEMVKernel_2Rows(const void* __restrict__ A,
                                  core::GgmlType type,
                                  const float* __restrict__ x,
                                  float* __restrict__ y, std::size_t M,
                                  std::size_t K) {
  // F32 denotes runtime format dispatch here; the common packed formats
  // specialize this same arithmetic body to remove per-block type branches.
  if constexpr (StaticType != core::GgmlType::kF32) {
    type = StaticType;
  }
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t pair_idx = (blockIdx.x * WavesPerBlock) + warp_id;
  const std::size_t m0 = pair_idx * 2;
  const std::size_t m1 = m0 + 1;
  if (m0 >= M) {
    return;
  }

  if (type == core::GgmlType::kQ8_0) {
    const std::size_t num_blocks = K / kQ8_0BlockSize;
    const auto* row0_blocks =
        static_cast<const Q8_0Block*>(A) + (m0 * num_blocks);
    const auto* row1_blocks =
        (m1 < M) ? (static_cast<const Q8_0Block*>(A) + (m1 * num_blocks))
                 : nullptr;
    float sum0 = 0.0F;
    float sum1 = 0.0F;

    for (std::size_t b = lane_id; b < num_blocks; b += 32) {
      const float* xb = x + (b * kQ8_0BlockSize);
      const Q8_0Block& wblk0 = row0_blocks[b];
      const float d0 = __half2float(wblk0.d);
      float blk_dot0 = 0.0F;
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        blk_dot0 += static_cast<float>(wblk0.qs[i]) * xb[i];
      }
      sum0 += d0 * blk_dot0;

      if (row1_blocks != nullptr) {
        const Q8_0Block& wblk1 = row1_blocks[b];
        const float d1 = __half2float(wblk1.d);
        float blk_dot1 = 0.0F;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
          blk_dot1 += static_cast<float>(wblk1.qs[i]) * xb[i];
        }
        sum1 += d1 * blk_dot1;
      }
    }

    for (int off = 16; off > 0; off >>= 1) {
      sum0 += __shfl_xor(sum0, off);
      sum1 += __shfl_xor(sum1, off);
    }

    if (lane_id == 0) {
      y[m0] = sum0;
      if (m1 < M) {
        y[m1] = sum1;
      }
    }
    return;
  }

  if (type == core::GgmlType::kQ8_K) {
    const std::size_t num_blocks = K / kQ8KBlockSize;
    const auto* row0_blocks =
        static_cast<const Q8KBlock*>(A) + (m0 * num_blocks);
    const auto* row1_blocks =
        (m1 < M) ? (static_cast<const Q8KBlock*>(A) + (m1 * num_blocks))
                 : nullptr;
    float sum0 = 0.0F;
    float sum1 = 0.0F;

    for (std::size_t b = 0; b < num_blocks; ++b) {
      const Q8KBlock& wblk0 = row0_blocks[b];
      const float d_w0 = wblk0.d;
      const Q8KBlock* wblk1 =
          (row1_blocks != nullptr) ? &row1_blocks[b] : nullptr;
      const float d_w1 = (wblk1 != nullptr) ? wblk1->d : 0.0F;
      const float* xb = x + (b * kQ8KBlockSize);
      float dot0 = 0.0F;
      float dot1 = 0.0F;
#pragma unroll
      for (std::size_t k = 0; k < 8; ++k) {
        const std::size_t idx = lane_id + (32 * k);
        const float xv = xb[idx];
        dot0 += static_cast<float>(wblk0.qs[idx]) * xv;
        if (wblk1 != nullptr) {
          dot1 += static_cast<float>(wblk1->qs[idx]) * xv;
        }
      }
      sum0 += d_w0 * dot0;
      sum1 += d_w1 * dot1;
    }

    for (int off = 16; off > 0; off >>= 1) {
      sum0 += __shfl_xor(sum0, off);
      sum1 += __shfl_xor(sum1, off);
    }

    if (lane_id == 0) {
      y[m0] = sum0;
      if (m1 < M) {
        y[m1] = sum1;
      }
    }
    return;
  }

  const float s0 = QuantWarpBlockDot(type, A, m0, x, K, lane_id, 0, 1);
  const float s1 =
      (m1 < M) ? QuantWarpBlockDot(type, A, m1, x, K, lane_id, 0, 1) : 0.0F;
  if (lane_id == 0) {
    y[m0] = s0;
    if (m1 < M) {
      y[m1] = s1;
    }
  }
}

// opt-c1xx-q8k-gemv: launches the Q8_K / Q8_0 block GEMV.
void LaunchQ8KBlockGEMV(const void* A, core::GgmlType type, const float* x,
                        float* y, std::size_t M, std::size_t K,
                        hipStream_t stream) {
  constexpr std::uint32_t kWavesPerBlock = 4;
  constexpr std::uint32_t kBlockThreads = kWavesPerBlock * 32;

  if (M >= 2 && (M % 2 == 0)) {
    const std::size_t num_pairs = M / 2;
    const std::size_t grid = (num_pairs + kWavesPerBlock - 1U) / kWavesPerBlock;
    const auto launch = [&]<core::GgmlType StaticType>() {
      Q8KBlockGEMVKernel_2Rows<kWavesPerBlock, StaticType>
          <<<grid, kBlockThreads, 0, stream>>>(A, type, x, y, M, K);
    };
    switch (type) {
      case core::GgmlType::kQ4_K:
        launch.template operator()<core::GgmlType::kQ4_K>();
        break;
      case core::GgmlType::kQ5_K:
        launch.template operator()<core::GgmlType::kQ5_K>();
        break;
      case core::GgmlType::kIQ4_XS:
        launch.template operator()<core::GgmlType::kIQ4_XS>();
        break;
      default:
        launch.template operator()<core::GgmlType::kF32>();
        break;
    }
    return;
  }

  const std::size_t grid = (M + kWavesPerBlock - 1U) / kWavesPerBlock;
  Q8KBlockGEMVKernel<kWavesPerBlock>
      <<<grid, kBlockThreads, 0, stream>>>(A, type, x, y, M, K);
}

}  // namespace kern3_ref::hip
#endif  // defined(ENGINE_ENABLE_HIP)
