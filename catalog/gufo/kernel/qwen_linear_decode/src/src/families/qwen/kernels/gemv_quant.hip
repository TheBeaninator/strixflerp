#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"
#include "src/families/qwen/kernels/ops/gemm.hpp"
#include "src/families/qwen/hip_common/quant_ops.hpp"

namespace gufo::hip {

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

// flerp DECODE-KERN3 (strixflerp gate/decode-kern3.toml): Q8KBlockGEMVKernel_2Rows for Q4_K with wide loads.
// Each row's arithmetic is QuantWarpBlockDot's sub16 route unchanged - lane L owns sub-blocks L, L+32, ... in that
// order, decodes them as DecodeQuantSub16 does, sumf += scale * dot - offset * xsum, then the same xor reduction,
// lane 0 writes - so every output has the same bits. What changes is how the bytes arrive: one 16-byte load for a
// sub-block's codes, issued for both rows before either is decoded (the scale bytes and d are read at decode, as
// DecodeQuantSub16 reads them, from the same cache lines), four 16-byte loads for the sixteen activations, which
// both rows share (the old kernel read x twice), and eight waves (sixteen rows) a block: 184 GB/s cache-cold on the
// 35B out projections against 111 for the old kernel, 162 at four waves and 172 at two; loading groups of
// sub-blocks ahead was slower (92-151) (gate/results/decode-kern3/sweep.jsonl). The same decode as
// kernel.flerp.moe_kquant's wide expert reader (SHAPE-TRANSFERS R1 fold). Taken when M is even, K a multiple of 256,
// and A and x 16-byte aligned (then every 144-byte-block row is).
constexpr std::uint32_t kWideWaves = 8;

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Q8KBlockGEMVKernel_2RowsWideQ4K(const std::uint8_t* __restrict__ A,
                                         const float* __restrict__ x,
                                         float* __restrict__ y, std::size_t M,
                                         std::size_t K) {
  const std::size_t lane_id = threadIdx.x & 31u;
  const std::size_t pair_idx = (blockIdx.x * WavesPerBlock) + (threadIdx.x >> 5u);
  const std::size_t m0 = pair_idx * 2;
  if (m0 >= M) {
    return;
  }
  const std::size_t row_bytes = (K / 256) * sizeof(Q4KBlock);
  const std::uint8_t* row0 = A + (m0 * row_bytes);
  const std::uint8_t* row1 = row0 + row_bytes;
  const std::size_t nsub = K / 16;
  float sum0 = 0.0F;
  float sum1 = 0.0F;
  for (std::size_t sb = lane_id; sb < nsub; sb += 32) {  // the old loop's sub-blocks, in the old order
    const std::size_t sb32 = (sb / 2) % 8;
    const std::size_t code_off = (sb / 16) * sizeof(Q4KBlock) + offsetof(Q4KBlock, qs) + ((sb32 / 2) * 32) + ((sb % 2) * 16);
    const uint4 p0 = *reinterpret_cast<const uint4*>(row0 + code_off);
    const uint4 p1 = *reinterpret_cast<const uint4*>(row1 + code_off);
    const unsigned shift = 4U * static_cast<unsigned>(sb32 & 1U);
    QuantSub16 a;
    QuantSub16 b;
    a.w[0] = (p0.x >> shift) & 0x0F0F0F0FU;
    a.w[1] = (p0.y >> shift) & 0x0F0F0F0FU;
    a.w[2] = (p0.z >> shift) & 0x0F0F0F0FU;
    a.w[3] = (p0.w >> shift) & 0x0F0F0F0FU;
    b.w[0] = (p1.x >> shift) & 0x0F0F0F0FU;
    b.w[1] = (p1.y >> shift) & 0x0F0F0F0FU;
    b.w[2] = (p1.z >> shift) & 0x0F0F0F0FU;
    b.w[3] = (p1.w >> shift) & 0x0F0F0F0FU;
    const std::uint8_t* blk0 = row0 + (sb / 16) * sizeof(Q4KBlock);
    const std::uint8_t* blk1 = row1 + (sb / 16) * sizeof(Q4KBlock);
    std::uint8_t sc = 0;
    std::uint8_t m = 0;
    GetQKScaleMin(sb32, blk0 + offsetof(Q4KBlock, scales), sc, m);
    a.scale = __half2float(reinterpret_cast<const __half*>(blk0)[0]) * static_cast<float>(sc);
    a.offset = __half2float(reinterpret_cast<const __half*>(blk0)[1]) * static_cast<float>(m);
    GetQKScaleMin(sb32, blk1 + offsetof(Q4KBlock, scales), sc, m);
    b.scale = __half2float(reinterpret_cast<const __half*>(blk1)[0]) * static_cast<float>(sc);
    b.offset = __half2float(reinterpret_cast<const __half*>(blk1)[1]) * static_cast<float>(m);
    float xv[16];
    const float4* x4 = reinterpret_cast<const float4*>(x + (sb * 16));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float4 v = x4[i];
      xv[4 * i] = v.x;
      xv[4 * i + 1] = v.y;
      xv[4 * i + 2] = v.z;
      xv[4 * i + 3] = v.w;
    }
    float dot0 = 0.0F;
    float xsum0 = 0.0F;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      dot0 += static_cast<float>(a.q[j]) * xv[j];
      xsum0 += xv[j];
    }
    sum0 += (a.scale * dot0) - (a.offset * xsum0);
    float dot1 = 0.0F;
    float xsum1 = 0.0F;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      dot1 += static_cast<float>(b.q[j]) * xv[j];
      xsum1 += xv[j];
    }
    sum1 += (b.scale * dot1) - (b.offset * xsum1);
  }
  for (int off = 16; off > 0; off >>= 1) {
    sum0 += __shfl_xor(sum0, off);
    sum1 += __shfl_xor(sum1, off);
  }
  if (lane_id == 0) {
    y[m0] = sum0;
    y[m0 + 1] = sum1;
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
    // flerp DECODE-KERN3: the wide-load route (same bits) for Q4_K. Not for Q6_K: on the 35B lm head (Q6_K,
    // 248320 x 2048) the same recipe measured no faster cache-cold (215 vs 218 GB/s), so it failed the gate's
    // continue rule and Q6_K keeps the kernel below.
    const auto aligned16 = [](const void* p) {
      return (reinterpret_cast<std::uintptr_t>(p) & 15U) == 0;
    };
    if (type == core::GgmlType::kQ4_K && K % 256 == 0 && aligned16(A) && aligned16(x)) {
      const std::size_t wide_grid = (num_pairs + kWideWaves - 1U) / kWideWaves;
      Q8KBlockGEMVKernel_2RowsWideQ4K<kWideWaves><<<wide_grid, kWideWaves * 32, 0, stream>>>(
          static_cast<const std::uint8_t*>(A), x, y, M, K);
      return;
    }
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

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
