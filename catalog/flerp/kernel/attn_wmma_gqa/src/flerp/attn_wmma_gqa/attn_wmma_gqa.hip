// SPDX-License-Identifier: MIT
// kernel.flerp.attn_wmma_gqa: gufo's masked prefill attention on the WMMA matrix cores (kernels/attention_wmma.hip,
// WmmaCausalAttention and PackAttentionHeads, gufo f070a17, MIT; catalog/gufo/kernel/qwen_attention holds it verbatim)
// with the query and KV head counts as template parameters instead of the 27B's 24 / 4, so qwen35moe's attention
// blocks (35B: 16 query heads, Ornith-1.5 397B: 32; both 2 KV heads of 256) take the tiled online-softmax WMMA pass
// instead of the scalar online fallback. The tile, the arithmetic and the order are the origin's; the changes are
// marked `flerp (kernel.flerp.attn_wmma_gqa)`. Only the case where the FP16 cache plane already holds this chunk's K
// and V (the fused QK-norm/RoPE/KV write) is taken: the origin's pack/sync kernels are 27B-shaped.
#include "flerp/attn_wmma_gqa/attn_wmma_gqa.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace flerp::attn {
namespace {

constexpr std::uint32_t kHeadDim = 256;

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

constexpr std::uint32_t kWmmaHeads = 2;  // query heads per block, must divide GQA
constexpr std::uint32_t kWmmaQueryRows = 32;
constexpr std::uint32_t kWmmaKeys = 16;
constexpr std::uint32_t kWmmaKSteps = kHeadDim / 16;
constexpr std::uint32_t kWmmaKStride = kHeadDim + 8;

}  // namespace

// External linkage on purpose: an anonymous-namespace kernel is dispatched with
// an empty name, and rocprofv3 then merges it into an unattributed "other"
// bucket -- 7.3% of a depth-8192 profile in this kernel's case.
//
// kQueryRows queries x kWmmaHeads heads form (kQueryRows / 16) * kWmmaHeads
// 16-row blocks, and kKeys keys form kKeys / 16 key blocks. Their product is
// pinned to 4 so the eight waves split as four S tiles x two halves of the head
// dimension, which is what keeps Q in registers at 64 VGPRs. Growing kQueryRows
// and shrinking kKeys in step therefore trades nothing in the tiling but halves
// the K/V bytes each block pulls per query row -- the dominant cost.
template<std::uint32_t kQueryHeads, std::uint32_t kKvHeads, std::uint32_t kQueryRows, std::uint32_t kKeys,
         bool kHeadMajor = false>
// LLVM 24 otherwise uses 241 VGPRs for packed V, crossing the six-wave
// allocation boundary. Keep six waves resident without changing the tiling.
__launch_bounds__(256, (__clang_major__ >= 24 && kHeadMajor) ? 6 : 2) __global__
    void GqaWmmaCausalAttention(const float* __restrict__ q,
                             const float* __restrict__ gate,
                             const __half* __restrict__ k_cache_f16,
                             const __half* __restrict__ v_cache_f16,
                             float* __restrict__ out_context,
                             std::uint32_t start_pos, std::size_t batch_size,
                             float* __restrict__ lse_out,
                             std::uint32_t key_begin,
                             std::uint32_t first_query) {
  constexpr std::uint32_t kGqa = kQueryHeads / kKvHeads;
  constexpr std::uint32_t kAttnWidth = kQueryHeads * kHeadDim;
  constexpr std::uint32_t kKvWidth = kKvHeads * kHeadDim;
  static_assert(kGqa % kWmmaHeads == 0, "head pairs within a KV group");
  constexpr std::uint32_t kRowBlocks = (kQueryRows / 16) * kWmmaHeads;
  constexpr std::uint32_t kKeyBlocks = kKeys / 16;
  constexpr std::uint32_t kSTiles = kRowBlocks * kKeyBlocks;
  constexpr std::uint32_t kKStepsPerWave = kWmmaKSteps / (8 / kSTiles);
  constexpr std::uint32_t kRows = kRowBlocks * 16;
  constexpr std::uint32_t kOTilesPerWave = (kRowBlocks * kWmmaKSteps) / 8;
  constexpr std::uint32_t kSoftmaxLanes = 256 / kRows;
  constexpr std::uint32_t kVtStride = kKeys + 8;
  static_assert(kSTiles == 4, "eight waves cover four S tiles in two halves");
  static_assert(kOTilesPerWave == 2 * kRowBlocks, "O tiles per wave");
  static_assert(kKeys % 16 == 0 && kQueryRows % 16 == 0, "16-row WMMA tiles");
  static_assert(kKStepsPerWave * 8 / kSTiles == kWmmaKSteps, "k split");
  static_assert(!kHeadMajor || kKeys == 16, "packed value tile width");
  static_assert(kWmmaKStride % 8 == 0 && kVtStride % 8 == 0,
                "fragment rows must start on a 16-byte boundary");
  static_assert(kSoftmaxLanes * (kKeys / kSoftmaxLanes) == kKeys, "softmax");

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;

  const std::uint32_t query_start = first_query + blockIdx.x * kQueryRows;
  const std::uint32_t head_pair = blockIdx.y;
  const std::uint32_t kv_head = head_pair / (kGqa / kWmmaHeads);
  const std::uint32_t pair_in_group = head_pair % (kGqa / kWmmaHeads);
  const std::uint32_t first_query_head =
      (kv_head * kGqa) + (pair_in_group * kWmmaHeads);
  constexpr float attention_scale = 1.0F / 16.0F;

  // Row block -> (query head, 16-query sub-block of this block's rows).
  const auto row_block_head = [&](std::uint32_t rb) {
    return first_query_head + (rb % kWmmaHeads);
  };
  const auto row_block_offset = [&](std::uint32_t rb) {
    return (rb / kWmmaHeads) * 16;
  };

  constexpr std::uint32_t kKvLdsHalves =
      kHeadMajor || kKeys * kWmmaKStride > kHeadDim * kVtStride
          ? kKeys * kWmmaKStride
          : kHeadDim * kVtStride;
  __shared__ __half kv_lds[kKvLdsHalves];
  // Padding avoids repeated banks in softmax reads and probability fragments.
  __shared__ float s_lds[2][kSTiles][16][17];
  __shared__ __half p_lds[kRows][kKeys + 8];
  __shared__ float row_max[kRows];
  __shared__ float row_sum[kRows];
  __shared__ float row_scale[kRows];

  const std::uint32_t s_tile = wave % kSTiles;
  const std::uint32_t s_kh = wave / kSTiles;
  const std::uint32_t s_rb = s_tile % kRowBlocks;
  const std::uint32_t s_kb = s_tile / kRowBlocks;

  v16h q_frag[kKStepsPerWave];
  {
    const std::uint32_t local_query =
        query_start + row_block_offset(s_rb) + sub;
    const bool live = local_query < batch_size;
    const float* q_row =
        q + (static_cast<std::size_t>(local_query) * kAttnWidth) +
        (static_cast<std::size_t>(row_block_head(s_rb)) * kHeadDim);
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
      const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
      const auto* qp = reinterpret_cast<const float4*>(q_row + d0);
#pragma unroll
      for (std::uint32_t v = 0; v < 4; ++v) {
        const float4 f = live ? qp[v] : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        q_frag[ks][(v * 4) + 0] = static_cast<_Float16>(f.x * attention_scale);
        q_frag[ks][(v * 4) + 1] = static_cast<_Float16>(f.y * attention_scale);
        q_frag[ks][(v * 4) + 2] = static_cast<_Float16>(f.z * attention_scale);
        q_frag[ks][(v * 4) + 3] = static_cast<_Float16>(f.w * attention_scale);
      }
    }
  }

  // This wave owns dim tiles `wave` and `wave + 8` for every row block, so both
  // halves of the head dimension stay evenly loaded.
  v8f o_acc[kRowBlocks][2] = {};
  // Each softmax row keeps the same four owner lanes across the key sweep.
  // Only the final statistics and each tile's rescale factor cross waves.
  float running_max = -INFINITY;
  float running_sum = 0.0F;

  const std::uint32_t context_end =
      start_pos + static_cast<std::uint32_t>(batch_size);
  const std::uint32_t max_visible =
      min(context_end, start_pos + query_start + kQueryRows);

  // One key per lane for V, a 16-dim slice per thread. This costs a strided
  // global read but makes the transpose's LDS writes conflict-free, which is
  // worth 1.8x; a coalesced remapping was measured and is slower at depth.
  constexpr std::uint32_t kVRegs = (kKeys * kHeadDim) / (256 * 8);
  constexpr std::uint32_t kKRegs = (kKeys * (kHeadDim / 8)) / 256;
  static_assert(kKRegs * 256 == kKeys * (kHeadDim / 8), "K stages evenly");
  const std::uint32_t v_key = lane % kKeys;
  const std::uint32_t v_slice = (tid / kKeys) * (kVRegs * 8);
  constexpr std::size_t kv_stride = kHeadMajor ? kHeadDim : kKvWidth;
  const std::size_t head_offset = static_cast<std::size_t>(kv_head) * kHeadDim *
                                  (kHeadMajor ? context_end : 1);
  const auto* v_base = v_cache_f16 + head_offset + v_slice;
  const auto* k_base = k_cache_f16 + head_offset;

  using ValueFragment = std::conditional_t<kHeadMajor, v16h, uint4>;
  constexpr std::uint32_t kValueFragments = kHeadMajor ? 2 : kVRegs;
  const auto load_v = [&](std::uint32_t key_start, ValueFragment* dst) {
    if constexpr (kHeadMajor) {
      // Each wave consumes two dimension tiles. The temporary value layout
      // supplies their WMMA fragments directly, avoiding a repeated LDS
      // transpose and barrier for every 16-key tile.
      const std::size_t padded = (context_end + 15u) & ~std::size_t{15};
#pragma unroll
      for (std::uint32_t t = 0; t < 2; ++t) {
        const std::uint32_t dim = (wave + t * 8u) * 16u + sub;
        dst[t] = LoadFrag(
            v_cache_f16 +
            static_cast<std::size_t>(kv_head) * padded * kHeadDim +
            static_cast<std::size_t>(key_start / 16u) * kHeadDim * 16u +
            dim * 16u);
      }
    } else {
      const std::uint32_t key_position = key_start + v_key;
      const auto* src =
          v_base + (static_cast<std::size_t>(key_position) * kv_stride);
      const bool live = key_position < context_end;
#pragma unroll
      for (std::uint32_t j = 0; j < kVRegs; ++j) {
        dst[j] = live ? *reinterpret_cast<const uint4*>(src + (j * 8))
                      : make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };
  // Coalesced: a wave reads one key row's 512 contiguous bytes.
  const auto load_k = [&](std::uint32_t key_start, uint4* dst) {
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_position = key_start + (idx / (kHeadDim / 8));
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      dst[n] =
          (key_position < context_end)
              ? *reinterpret_cast<const uint4*>(
                    k_base +
                    (static_cast<std::size_t>(key_position) * kv_stride) + d8)
              : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  uint4 k_cur[kKRegs];
  ValueFragment v_cur[kValueFragments];
  uint4 k_pre[kKRegs] = {};
  ValueFragment v_pre[kValueFragments] = {};
  load_k(key_begin, k_cur);
  load_v(key_begin, v_cur);

  for (std::uint32_t key_start = key_begin; key_start < max_visible;
       key_start += kKeys) {
    // --- stage K from the registers the previous iteration prefetched
    __syncthreads();
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_row = idx / (kHeadDim / 8);
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      *reinterpret_cast<uint4*>(&kv_lds[(key_row * kWmmaKStride) + d8]) =
          k_cur[n];
    }
    __syncthreads();

    // --- prefetch the next key tile. Everything below covers its latency.
    const std::uint32_t next_start = key_start + kKeys;
    if (next_start < max_visible) {
      load_k(next_start, k_pre);
      if constexpr (!kHeadMajor)
        load_v(next_start, v_pre);
    }

    // --- S = Q K^T ---
    {
      v8f s_acc = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
        const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
        const v16h k_frag =
            LoadFrag(&kv_lds[(((s_kb * 16) + sub) * kWmmaKStride) + d0]);
        s_acc = Wmma(q_frag[ks], k_frag, s_acc);
      }
      // Each half writes its own slot, so the reader sums them instead of a
      // read-modify-write that would need a barrier between the halves.
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[s_kh][s_tile][(2 * i) + half_id][sub] = s_acc[i];
      }
    }
    __syncthreads();

    // --- online softmax: kSoftmaxLanes threads per query row ---
    {
      const std::uint32_t rg = tid / kSoftmaxLanes;
      const std::uint32_t seg = tid % kSoftmaxLanes;
      constexpr std::uint32_t kPerLane = kKeys / kSoftmaxLanes;
      const std::uint32_t rb = rg / 16;
      const std::uint32_t row = rg % 16;
      const std::uint32_t local_query =
          query_start + row_block_offset(rb) + row;
      const std::uint32_t absolute_query = start_pos + local_query;
      float part_max = -INFINITY;
      float vals[kPerLane];
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const std::uint32_t col = (seg * kPerLane) + m;
        const std::uint32_t key_position = key_start + col;
        const bool valid = local_query < batch_size &&
                           key_position <= absolute_query &&
                           key_position < context_end;
        const std::uint32_t tile = ((col / 16) * kRowBlocks) + rb;
        vals[m] = valid ? (s_lds[0][tile][row][col % 16] +
                           s_lds[1][tile][row][col % 16])
                        : -INFINITY;
        part_max = fmaxf(part_max, vals[m]);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_max = fmaxf(part_max, __shfl_xor(part_max, off));
      }
      const float prev_max = running_max;
      const float next_max = fmaxf(prev_max, part_max);
      const float prior_scale =
          isfinite(prev_max) ? __expf(prev_max - next_max) : 0.0F;
      float part_sum = 0.0F;
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const float w = isfinite(vals[m]) ? __expf(vals[m] - next_max) : 0.0F;
        part_sum += w;
        p_lds[rg][(seg * kPerLane) + m] = static_cast<__half>(w);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_sum += __shfl_xor(part_sum, off);
      }
      running_max = next_max;
      running_sum = (running_sum * prior_scale) + part_sum;
      if (seg == 0)
        row_scale[rg] = prior_scale;
    }
    __syncthreads();

    // The unpacked path shares each scale across both dimension tiles. Packed
    // V instead rescales immediately before PV, shortening register lifetimes
    // enough to retain six waves per SIMD with the causal-tail correction.
    if constexpr (!kHeadMajor) {
#pragma unroll
      for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
        float scale[8];
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          scale[i] = row_scale[(rb * 16) + (2 * i) + half_id];
        }
#pragma unroll
        for (std::uint32_t t = 0; t < 2; ++t) {
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i) {
            o_acc[rb][t][i] *= scale[i];
          }
        }
      }
    }

    // --- stage V transposed. No barrier is needed here: the barrier after the
    // softmax already separates every read of K in the S phase from these
    // writes to the same buffer.
    //
    // Lane L owns key L % kKeys and each thread a dim slice, so consecutive
    // lanes write consecutive *columns* of V^T: two 16-bit writes share a dword
    // and the wave covers 16 banks. The mapping the coalesced global read would
    // prefer -- each lane taking 8 contiguous dims of one key -- puts all 32
    // lanes on rows kVtStride halves apart, which is one bank, and costs a
    // 32-way conflict on every write.
    if constexpr (!kHeadMajor) {
#pragma unroll
      for (std::uint32_t j = 0; j < kVRegs; ++j) {
        const auto* packed = reinterpret_cast<const __half*>(&v_cur[j]);
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          kv_lds[((v_slice + (j * 8) + i) * kVtStride) + v_key] = packed[i];
        }
      }
      __syncthreads();
    }

    // --- O += P V ---
    // Packed V is in registers: visit each row block's two dimension tiles
    // together, sharing its scales. Unpacked V comes from LDS: visit dimension
    // tiles first so the compiler can share their fragments across row blocks.
    // The order is compile-time constant and both paths keep the same
    // arithmetic.
#pragma unroll
    for (std::uint32_t outer = 0; outer < (kHeadMajor ? kRowBlocks : 2);
         ++outer) {
      float scale[8];
      if constexpr (kHeadMajor) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i)
          scale[i] = row_scale[outer * 16 + 2 * i + half_id];
      }
#pragma unroll
      for (std::uint32_t inner = 0; inner < (kHeadMajor ? 2 : kRowBlocks);
           ++inner) {
        const std::uint32_t rb = kHeadMajor ? outer : inner;
        const std::uint32_t t = kHeadMajor ? inner : outer;
        const std::uint32_t dim_tile = wave + t * 8;
        v16h v_frag[kKeyBlocks];
#pragma unroll
        for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
          if constexpr (kHeadMajor)
            v_frag[kb] = v_cur[t];
          else
            v_frag[kb] =
                LoadFrag(&kv_lds[(dim_tile * 16 + sub) * kVtStride + kb * 16]);
        }
        if constexpr (kHeadMajor) {
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i)
            o_acc[rb][t][i] *= scale[i];
        }
#pragma unroll
        for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
          const v16h p_frag = LoadFrag(&p_lds[(rb * 16) + sub][kb * 16]);
          v8f next = Wmma(p_frag, v_frag[kb], o_acc[rb][t]);
          const auto first_key = key_start + kb * 16;
          // Like Flash-Next, sum a query's final partial tile over visible
          // keys only. WMMA can round differently when a zero probability
          // multiplies a populated future value instead of chunk padding.
          // Complete tiles keep the matrix-core path and its original order.
          if (first_key + 15 > start_pos + query_start) {
            v8f tail = o_acc[rb][t];
            // Keep the position difference uniform across the wave; only the
            // even/odd row owner varies between its two half-waves.
            const int relative_key = static_cast<int>(first_key) -
                                     static_cast<int>(start_pos + query_start +
                                                      row_block_offset(rb));
#pragma unroll
            for (std::uint32_t key = 0; key < 16; ++key) {
              const float value = static_cast<float>(v_frag[kb][key]);
#pragma unroll
              for (std::uint32_t i = 0; i < 8; ++i) {
                if (static_cast<int>(half_id) <
                        relative_key + 15 - static_cast<int>(2 * i) &&
                    static_cast<int>(half_id) >= relative_key +
                                                     static_cast<int>(key) -
                                                     static_cast<int>(2 * i)) {
                  tail[i] =
                      fmaf(__half2float(
                               p_lds[rb * 16 + 2 * i + half_id][kb * 16 + key]),
                           value, tail[i]);
                }
              }
            }
#pragma unroll
            for (std::uint32_t i = 0; i < 8; ++i)
              if (static_cast<int>(half_id) <
                  relative_key + 15 - static_cast<int>(2 * i))
                next[i] = tail[i];
          }
          o_acc[rb][t] = next;
#if __clang_major__ >= 24
          // Bound overlapping LDS fragments in the unpacked tail. LLVM 24
          // otherwise spills them across the unrolled row/dimension tiles.
          if constexpr (!kHeadMajor)
            __builtin_amdgcn_sched_barrier(0);
#endif
        }
      }
    }

#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      k_cur[n] = k_pre[n];
    }
    if constexpr (kHeadMajor) {
      // Packed V is already in WMMA order. Load it after the current PV work;
      // the next QK/softmax hides its latency without retaining a second set
      // of fragments across the causal-tail calculation.
      if (next_start < max_visible)
        load_v(next_start, v_cur);
    } else {
#pragma unroll
      for (std::uint32_t n = 0; n < kValueFragments; ++n)
        v_cur[n] = v_pre[n];
    }
  }
  if (tid % kSoftmaxLanes == 0) {
    row_max[tid / kSoftmaxLanes] = running_max;
    row_sum[tid / kSoftmaxLanes] = running_sum;
  }
  __syncthreads();

  // --- epilogue ---
#pragma unroll
  for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
    const std::uint32_t query_head = row_block_head(rb);
    const std::uint32_t row_offset = row_block_offset(rb);
#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
      const std::uint32_t dim_tile = wave + (t * 8);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t row = (2 * i) + half_id;
        const std::uint32_t local_query = query_start + row_offset + row;
        if (local_query >= batch_size) {
          continue;
        }
        const float denominator = row_sum[(rb * 16) + row];
        const std::size_t offset =
            (static_cast<std::size_t>(local_query) * kAttnWidth) +
            (static_cast<std::size_t>(query_head) * kHeadDim) +
            (dim_tile * 16) + sub;
        float value =
            (denominator > 0.0F) ? (o_acc[rb][t][i] / denominator) : 0.0F;
        if (gate != nullptr && lse_out == nullptr) {
          value *= 1.0F / (1.0F + __expf(-gate[offset]));
        }
        out_context[offset] = value;
      }
    }
  }
  if (lse_out != nullptr && wave == 0) {
    for (std::uint32_t rg = lane; rg < kRows; rg += 32) {
      const std::uint32_t rb = rg / 16;
      const std::uint32_t local_query =
          query_start + row_block_offset(rb) + (rg % 16);
      if (local_query >= batch_size) {
        continue;
      }
      const float denominator = row_sum[rg];
      lse_out[(static_cast<std::size_t>(row_block_head(rb)) * batch_size) +
              local_query] =
          (denominator > 0.0F) ? (row_max[rg] + logf(denominator)) : -INFINITY;
    }
  }
}

// FFN scratch is idle during attention. Keep K head-major and transpose each
// 16-key V tile into the layout consumed by WMMA. Persistent KV is unchanged.
template<std::uint32_t kKvHeads>
__global__ void GqaPackAttentionHeads(const __half* keys, const __half* values,
                                   __half* packed_keys, __half* packed_values,
                                   std::uint32_t length) {
  __shared__ __half tile[16][kHeadDim + 8];
  // Adjacent blocks read adjacent heads of the same source token tile.
  const std::uint32_t head = blockIdx.x, base = blockIdx.y * 16;
  for (std::uint32_t i = threadIdx.x; i < 16 * kHeadDim / 8; i += 256) {
    const std::uint32_t key = i / (kHeadDim / 8);
    const std::uint32_t dim = (i % (kHeadDim / 8)) * 8;
    uint4 value = make_uint4(0, 0, 0, 0);
    if (base + key < length) {
      const auto src =
          (static_cast<std::size_t>(base + key) * kKvHeads + head) * kHeadDim +
          dim;
      const auto dst =
          (static_cast<std::size_t>(head) * length + base + key) * kHeadDim +
          dim;
      *reinterpret_cast<uint4*>(packed_keys + dst) =
          *reinterpret_cast<const uint4*>(keys + src);
      value = *reinterpret_cast<const uint4*>(values + src);
    }
    *reinterpret_cast<uint4*>(&tile[key][dim]) = value;
  }
  __syncthreads();
  const std::size_t padded = (length + 15u) & ~std::size_t{15};
  for (std::uint32_t i = threadIdx.x; i < 16 * kHeadDim / 8; i += 256) {
    const std::uint32_t dim = i / 2, key = (i % 2) * 8;
    union {
      uint4 packed;
      __half halves[8];
    } value;
#pragma unroll
    for (std::uint32_t j = 0; j < 8; ++j)
      value.halves[j] = tile[key + j][dim];
    *reinterpret_cast<uint4*>(
        packed_values + static_cast<std::size_t>(head) * padded * kHeadDim +
        static_cast<std::size_t>(base) * kHeadDim + dim * 16 + key) =
        value.packed;
  }
}

namespace {
template <std::uint32_t QH, std::uint32_t KVH>
bool LaunchShape(const float* q, const float* gate, const void* layer_k, const void* layer_v, float* out,
                 std::uint32_t start_pos, std::size_t batch_size, hipStream_t stream, void* packed_k_ws,
                 std::size_t packed_k_bytes, void* packed_v_ws, std::size_t packed_v_bytes) {
  constexpr std::size_t block_size = 256;
  constexpr std::uint32_t kGqa = QH / KVH;
  const dim3 grid(static_cast<unsigned int>((batch_size + kWmmaQueryRows - 1) / kWmmaQueryRows), KVH * (kGqa / kWmmaHeads));
  const auto length = start_pos + static_cast<std::uint32_t>(batch_size);
  const std::size_t key_head_bytes = static_cast<std::size_t>(length) * kHeadDim * sizeof(__half);
  const std::size_t value_head_bytes = ((static_cast<std::size_t>(length) + 15u) & ~std::size_t{15}) * kHeadDim * sizeof(__half);
  const auto heads_per_group = static_cast<std::uint32_t>(
      std::min({std::size_t{KVH}, packed_k_ws != nullptr ? packed_k_bytes / key_head_bytes : 0,
                packed_v_ws != nullptr ? packed_v_bytes / value_head_bytes : 0}));
  const auto* lk = static_cast<const __half*>(layer_k);
  const auto* lv = static_cast<const __half*>(layer_v);
  // As the origin: large chunks copy the prefix head-major into the idle FFN scratch first.
  if (batch_size >= 1024 && heads_per_group != 0) {
    auto* pk = static_cast<__half*>(packed_k_ws);
    auto* pv = static_cast<__half*>(packed_v_ws);
    for (std::uint32_t first = 0; first < KVH; first += heads_per_group) {
      const auto heads = std::min(heads_per_group, KVH - first);
      const std::size_t query_offset = static_cast<std::size_t>(first) * kGqa * kHeadDim;
      GqaPackAttentionHeads<KVH><<<dim3(heads, (length + 15u) / 16u), block_size, 0, stream>>>(
          lk + first * kHeadDim, lv + first * kHeadDim, pk, pv, length);
      const dim3 group_grid(grid.x, heads * (kGqa / kWmmaHeads));
      GqaWmmaCausalAttention<QH, KVH, kWmmaQueryRows, kWmmaKeys, true><<<group_grid, block_size, 0, stream>>>(
          q + query_offset, gate ? gate + query_offset : nullptr, pk, pv, out + query_offset, start_pos, batch_size,
          nullptr, 0, 0);
    }
  } else {
    GqaWmmaCausalAttention<QH, KVH, kWmmaQueryRows, kWmmaKeys><<<grid, block_size, 0, stream>>>(
        q, gate, lk, lv, out, start_pos, batch_size, nullptr, 0, 0);
  }
  return hipGetLastError() == hipSuccess;
}
}  // namespace

bool Supported(std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim) noexcept {
  return head_dim == kHeadDim && ((num_kv_heads == 2 && (num_heads == 16 || num_heads == 32 || num_heads == 8)) ||
                                  (num_kv_heads == 4 && (num_heads == 16 || num_heads == 32)));
}

bool LaunchGqaWmmaAttention(const float* q, const float* gate, const void* k_cache_f16, const void* v_cache_f16,
                            float* out_context, std::uint32_t layer_idx, std::uint32_t start_pos,
                            std::size_t batch_size, std::uint32_t max_context, std::uint32_t num_heads,
                            std::uint32_t num_kv_heads, std::uint32_t head_dim, hipStream_t stream,
                            void* packed_k_ws, std::size_t packed_k_bytes, void* packed_v_ws,
                            std::size_t packed_v_bytes) {
  if (!Supported(num_heads, num_kv_heads, head_dim) || batch_size == 0 || k_cache_f16 == nullptr ||
      v_cache_f16 == nullptr || static_cast<std::size_t>(start_pos) + batch_size > max_context)
    return false;
  const std::size_t layer_off = static_cast<std::size_t>(layer_idx) * max_context * num_kv_heads * head_dim;
  const void* lk = static_cast<const __half*>(k_cache_f16) + layer_off;
  const void* lv = static_cast<const __half*>(v_cache_f16) + layer_off;
#define FLERP_GQA(QH, KVH) \
  if (num_heads == QH && num_kv_heads == KVH) \
    return LaunchShape<QH, KVH>(q, gate, lk, lv, out_context, start_pos, batch_size, stream, packed_k_ws, \
                                packed_k_bytes, packed_v_ws, packed_v_bytes);
  FLERP_GQA(8, 2)
  FLERP_GQA(16, 2)
  FLERP_GQA(32, 2)
  FLERP_GQA(16, 4)
  FLERP_GQA(32, 4)
#undef FLERP_GQA
  return false;
}

}  // namespace flerp::attn
