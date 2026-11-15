// SPDX-License-Identifier: MIT
// See moe_kgrouped.hpp. Wave32, eight waves a block. Recipes re-expressed for GGUF K-quants (docs/PREFILL-ORNITH.md):
// strixite moe_grouped / linear_wmma (F16 weight tile decoded into LDS once a K step, x tile staged beside it,
// register prefetch of the next step), Crucible's gate/up pair with the SwiGLU epilogue in registers, a counting-sort
// grouping and a device tile list.
#include "flerp/moe_kgrouped/moe_kgrouped.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace flerp::kgrouped {
namespace {

constexpr int kThreads = 256;  // 8 waves of 32
constexpr int kBK = 64;        // K per step: a quarter of a 256-value block
constexpr int kLd = kBK + 8;   // LDS row stride in F16 (144 bytes: 16-byte aligned rows, banks staggered)
constexpr int kWRows = 128;    // weight rows a block: 64 gate + 64 up (GateUp) or 128 outputs (Down, Linear)
constexpr std::size_t kQ4KBytes = 144;
constexpr std::size_t kQ6KBytes = 210;
constexpr float kF16Max = 65504.0F;

typedef _Float16 f16x16 __attribute__((ext_vector_type(16)));
typedef _Float16 f16x2 __attribute__((ext_vector_type(2)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x8 __attribute__((ext_vector_type(8)));

__device__ inline f16x16 LoadFrag(const uint16_t* p) {  // row (lane % 16) of a 16-byte aligned LDS tile, 16 K values
  const uint4* r = reinterpret_cast<const uint4*>(p + (threadIdx.x % 16) * kLd);
  const uint4 lo = r[0], hi = r[1];
  const u32x8 v = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
  return __builtin_bit_cast(f16x16, v);
}

__device__ inline uint32_t PackF16(float a, float b) {  // round to nearest even, a in the low half
  const f16x2 h = {static_cast<_Float16>(a), static_cast<_Float16>(b)};
  return __builtin_bit_cast(uint32_t, h);
}

__device__ inline uint4 Load16Align2(const uint8_t* p) {
  uint4 v;
  __builtin_memcpy(&v, __builtin_assume_aligned(p, 2), 16);
  return v;
}

// The raw bytes one thread needs for its part of a weight tile step, fetched a step ahead.
struct WRaw {
  uint4 a, b, c;
  uint32_t s;
};

// Thread part u (0, 1) of weight row `row` at step kb (K values [64 kb, 64 kb + 64)): the 16 values 16u..16u+15 of
// both 32-value halves of the step.
template <WType WT>
__device__ inline WRaw FetchW(const uint8_t* row, int kb, int u) {
  WRaw r{};
  const int s = kb & 3;
  if constexpr (WT == WType::kQ4K) {
    const uint8_t* blk = row + static_cast<std::size_t>(kb >> 2) * kQ4KBytes;
    r.a = *reinterpret_cast<const uint4*>(blk);                       // d, dmin, scales[12]
    r.b = *reinterpret_cast<const uint4*>(blk + 16 + 32 * s + 16 * u);  // qs: sub-blocks 2s (low), 2s+1 (high)
  } else {
    const uint8_t* blk = row + static_cast<std::size_t>(kb >> 2) * kQ6KBytes;
    const int h = s >> 1;
    r.a = Load16Align2(blk + 64 * h + 16 * u);        // ql, segment 2(s&1)
    r.b = Load16Align2(blk + 64 * h + 32 + 16 * u);   // ql, segment 2(s&1)+1
    r.c = Load16Align2(blk + 128 + 32 * h + 16 * u);  // qh
    const int seg0 = (s & 1) * 2;
    const auto* sc = reinterpret_cast<const int8_t*>(blk + 192);
    const uint32_t s0 = static_cast<uint8_t>(sc[8 * h + u + 2 * seg0]);
    const uint32_t s1 = static_cast<uint8_t>(sc[8 * h + u + 2 * seg0 + 2]);
    const uint32_t d = *reinterpret_cast<const uint16_t*>(__builtin_assume_aligned(blk + 208, 2));
    r.s = s0 | (s1 << 8) | (d << 16);
  }
  return r;
}

// Decode a fetched part to F16 into LDS: run 0 (K 16u..) and run 1 (K 32 + 16u..) of row `dst`.
template <WType WT>
__device__ inline void DecodeW(const WRaw& r, int kb, uint16_t* dst, int u) {
  const int s = kb & 3;
  float ds0, dm0, ds1, dm1;
  uint32_t c0[4], c1[4];  // codes, one per byte
  if constexpr (WT == WType::kQ4K) {
    const float d = static_cast<float>(__builtin_bit_cast(_Float16, static_cast<uint16_t>(r.a.x & 0xFFFFU)));
    const float dmin = static_cast<float>(__builtin_bit_cast(_Float16, static_cast<uint16_t>(r.a.x >> 16)));
    // get_scale_min_k4 for sub-blocks j = 2s, 2s+1 from the 12 scale bytes (words a.y, a.z, a.w).
    const unsigned sh = 8U * static_cast<unsigned>((2 * s) & 3);
    const uint32_t lo_s = r.a.y >> sh, lo_m = r.a.z >> sh, hi = r.a.w >> sh;
    uint32_t sc0, sc1, m0, m1;
    if (s < 2) {
      sc0 = lo_s & 63U;
      sc1 = (lo_s >> 8) & 63U;
      m0 = lo_m & 63U;
      m1 = (lo_m >> 8) & 63U;
    } else {
      sc0 = (hi & 15U) | (((lo_s >> 6) & 3U) << 4);
      sc1 = ((hi >> 8) & 15U) | (((lo_s >> 14) & 3U) << 4);
      m0 = ((hi >> 4) & 15U) | (((lo_m >> 6) & 3U) << 4);
      m1 = ((hi >> 12) & 15U) | (((lo_m >> 14) & 3U) << 4);
    }
    ds0 = d * static_cast<float>(sc0);
    dm0 = dmin * static_cast<float>(m0);
    ds1 = d * static_cast<float>(sc1);
    dm1 = dmin * static_cast<float>(m1);
    const uint32_t w[4] = {r.b.x, r.b.y, r.b.z, r.b.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      c0[i] = w[i] & 0x0F0F0F0FU;
      c1[i] = (w[i] >> 4) & 0x0F0F0F0FU;
    }
  } else {
    const float d = static_cast<float>(__builtin_bit_cast(_Float16, static_cast<uint16_t>(r.s >> 16)));
    ds0 = d * static_cast<float>(static_cast<int8_t>(r.s & 0xFFU));
    ds1 = d * static_cast<float>(static_cast<int8_t>((r.s >> 8) & 0xFFU));
    dm0 = 32.0F * ds0;  // value = ds * (code - 32) = ds * code - 32 ds (32 ds exact)
    dm1 = 32.0F * ds1;
    const unsigned nsh = (s & 1) ? 4U : 0U;
    const unsigned qsh = 4U * static_cast<unsigned>(s & 1);
    const uint32_t la[4] = {r.a.x, r.a.y, r.a.z, r.a.w};
    const uint32_t lb[4] = {r.b.x, r.b.y, r.b.z, r.b.w};
    const uint32_t qh[4] = {r.c.x, r.c.y, r.c.z, r.c.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      c0[i] = ((la[i] >> nsh) & 0x0F0F0F0FU) | (((qh[i] >> qsh) & 0x03030303U) << 4);
      c1[i] = ((lb[i] >> nsh) & 0x0F0F0F0FU) | (((qh[i] >> (qsh + 2U)) & 0x03030303U) << 4);
    }
  }
  uint32_t o0[8], o1[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      const float a0 = static_cast<float>((c0[i] >> (16 * p)) & 0xFFU);
      const float b0 = static_cast<float>((c0[i] >> (16 * p + 8)) & 0xFFU);
      const float a1 = static_cast<float>((c1[i] >> (16 * p)) & 0xFFU);
      const float b1 = static_cast<float>((c1[i] >> (16 * p + 8)) & 0xFFU);
      o0[2 * i + p] = PackF16(__builtin_fmaf(ds0, a0, -dm0), __builtin_fmaf(ds0, b0, -dm0));
      o1[2 * i + p] = PackF16(__builtin_fmaf(ds1, a1, -dm1), __builtin_fmaf(ds1, b1, -dm1));
    }
  }
  uint4* d0 = reinterpret_cast<uint4*>(dst + 16 * u);
  uint4* d1 = reinterpret_cast<uint4*>(dst + 32 + 16 * u);
  d0[0] = make_uint4(o0[0], o0[1], o0[2], o0[3]);
  d0[1] = make_uint4(o0[4], o0[5], o0[6], o0[7]);
  d1[0] = make_uint4(o1[0], o1[1], o1[2], o1[3]);
  d1[1] = make_uint4(o1[4], o1[5], o1[6], o1[7]);
}

// One work item: BM rows of one expert's group (tile (e, r0) of the list) x kWRows weight rows.
//   SWIGLU (GateUp): weight rows 0..63 = gate columns c0..c0+63, 64..127 = up columns c0..; x row = token of the
//                    pair; h F16 at the pair's sorted position, columns [c0, c0 + 64), c0 = 64 blockIdx.y.
//   !SWIGLU (Down, Linear): weight rows = outputs [c0, c0 + 128), c0 = 128 blockIdx.y; A row = the sorted position
//                    (a_by_pos, Down) or the pair's token (Linear); FP32 out at the pair's row.
// Waves: wm = wave & 1 takes row tiles wm, wm + 2, ..; wn = wave >> 1 takes B tiles (SWIGLU: gate 16wn and up
// 64 + 16wn - the same 16 columns, so h forms in registers; else 32wn and 32wn + 16).
template <int BM, WType WT, bool SWIGLU>
__global__ void __launch_bounds__(kThreads)
    GroupedKernel(const uint8_t* __restrict__ w0, const uint8_t* __restrict__ w1, std::size_t expert_bytes,
                  std::size_t row_bytes, const uint16_t* __restrict__ a_in, int a_by_pos, int x_share,
                  const int* __restrict__ sorted, const int* __restrict__ offsets, const int* __restrict__ counts,
                  const int2* __restrict__ tiles, const int* __restrict__ n_tiles, void* __restrict__ out,
                  int n, int k, unsigned* __restrict__ overflow, int col_fast) {
  constexpr int TM = BM / 32;
  static_assert(BM % 32 == 0 && TM >= 1, "BM a multiple of 32");
  // col_fast: consecutive blocks are the column blocks of one tile (its x / h rows stay in L2 across them);
  // otherwise the tiles of one column block (its weight panel stays in L2 across an expert's tiles).
  const int ti = col_fast ? static_cast<int>(blockIdx.y) : static_cast<int>(blockIdx.x);
  const int cb = col_fast ? static_cast<int>(blockIdx.x) : static_cast<int>(blockIdx.y);
  if (ti >= *n_tiles) return;
  __shared__ __attribute__((aligned(16))) uint16_t xs[BM * kLd];
  __shared__ __attribute__((aligned(16))) uint16_t ws[kWRows * kLd];
  __shared__ int64_t aoff[BM], ooff[BM];

  const int tid = static_cast<int>(threadIdx.x);
  const int2 tile = tiles[ti];
  const int e = tile.x, r0 = tile.y;
  const int rows = min(BM, counts[e] - r0);
  const int base = offsets[e] + r0;
  for (int r = tid; r < BM; r += kThreads) {
    if (r < rows) {
      const int pos = base + r;
      const int pair = sorted[pos];
      aoff[r] = static_cast<int64_t>(a_by_pos ? pos : pair / x_share) * k;
      ooff[r] = static_cast<int64_t>(SWIGLU ? pos : pair) * n;
    } else {
      aoff[r] = -1;
      ooff[r] = -1;
    }
  }
  // This thread's weight row (tid / 2) and part (tid % 2).
  const int wr = tid >> 1, wu = tid & 1;
  const uint8_t* wrow = nullptr;
  {
    const int ncols = SWIGLU ? 64 : kWRows;
    const int c0 = cb * ncols;
    const int col = c0 + (SWIGLU ? (wr & 63) : wr);
    const uint8_t* stack = (SWIGLU && wr >= 64) ? w1 : w0;
    if (col < n) wrow = stack + static_cast<std::size_t>(e) * expert_bytes + static_cast<std::size_t>(col) * row_bytes;
  }
  __syncthreads();

  constexpr int kXLoads = BM * (kBK / 8) / kThreads;  // uint4 of x a thread a step
  static_assert(kXLoads >= 1, "whole x passes");
  const int nk = k / kBK;
  uint4 xr[kXLoads];
  WRaw wraw{};
  const auto fetch = [&](int kb) {
#pragma unroll
    for (int i = 0; i < kXLoads; ++i) {
      const int idx = tid + kThreads * i, row = idx >> 3, seg = idx & 7;
      const int64_t off = aoff[row];
      xr[i] = off >= 0 ? *reinterpret_cast<const uint4*>(a_in + off + kb * kBK + seg * 8) : make_uint4(0, 0, 0, 0);
    }
    if (wrow != nullptr) wraw = FetchW<WT>(wrow, kb, wu);
  };

  const int wave = tid >> 5, wm = wave & 1, wn = wave >> 1;
  const int b0 = SWIGLU ? 16 * wn : 32 * wn;
  const int b1 = SWIGLU ? 64 + 16 * wn : 32 * wn + 16;
  f32x8 acc[TM][2];
#pragma unroll
  for (int i = 0; i < TM; ++i) acc[i][0] = acc[i][1] = f32x8{};

  fetch(0);
  for (int kb = 0; kb < nk; ++kb) {
    __syncthreads();  // the previous step's fragments are read
#pragma unroll
    for (int i = 0; i < kXLoads; ++i) {
      const int idx = tid + kThreads * i, row = idx >> 3, seg = idx & 7;
      *reinterpret_cast<uint4*>(xs + row * kLd + seg * 8) = xr[i];
    }
    if (wrow != nullptr) {
      DecodeW<WT>(wraw, kb, ws + wr * kLd, wu);
    } else {
      uint4* z = reinterpret_cast<uint4*>(ws + wr * kLd);
      z[2 * wu] = z[2 * wu + 1] = z[4 + 2 * wu] = z[5 + 2 * wu] = make_uint4(0, 0, 0, 0);
    }
    __syncthreads();
    if (kb + 1 < nk) fetch(kb + 1);  // in flight during this step's matrix work
#pragma unroll
    for (int kk = 0; kk < kBK; kk += 16) {
      f16x16 a[TM];
#pragma unroll
      for (int i = 0; i < TM; ++i) a[i] = LoadFrag(xs + (16 * (2 * i + wm)) * kLd + kk);
      const f16x16 fb0 = LoadFrag(ws + b0 * kLd + kk);
      const f16x16 fb1 = LoadFrag(ws + b1 * kLd + kk);
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        acc[i][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], fb0, acc[i][0]);
        acc[i][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], fb1, acc[i][1]);
      }
    }
  }

  const int lane = tid & 31;
  if constexpr (SWIGLU) {
    uint16_t* h = static_cast<uint16_t*>(out);
    const int c = cb * 64 + 16 * wn + (lane & 15);
    bool over = false;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        const int r = 16 * (2 * i + wm) + 2 * v + (lane >> 4);
        if (r < rows && c < n) {
          const float g = acc[i][0][v], up = acc[i][1][v];
          float hv = (g / (1.0F + expf(-g))) * up;
          if (!(fabsf(hv) <= kF16Max)) {
            over = true;
            hv = hv != hv ? 0.0F : copysignf(kF16Max, hv);
          }
          h[ooff[r] + c] = __builtin_bit_cast(uint16_t, static_cast<_Float16>(hv));
        }
      }
    }
    if (over) atomicOr(overflow, 1U);
  } else {
    float* y = static_cast<float*>(out);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int c = cb * kWRows + 32 * wn + 16 * j + (lane & 15);
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int v = 0; v < 8; ++v) {
          const int r = 16 * (2 * i + wm) + 2 * v + (lane >> 4);
          if (r < rows && c < n) y[ooff[r] + c] = acc[i][j][v];
        }
      }
    }
  }
}

// ---- Exact variant (docs/PREFILL-ORNITH.md section 7; owner 2026-11-14): the weight codes enter the matrix units as
// exact F16 integers (Q4_K q in 0..15, Q6_K q - 32 in -32..31), each 16-K group's scale d*sc (and Q4_K's min dmin*m
// times the row's 32-value x sum) goes to the FP32 accumulator; x and h enter as a hi/lo F16 pair (hi = f16(v),
// lo = f16(v - hi): about 22 significant bits), two WMMAs a fragment. No weight rounding; activations near FP32.

// The codes of a fetched part as exact F16 into LDS (runs as DecodeW), and the four 16-K groups' scales of this row
// (thread part u writes groups u and 2 + u) and, Q4_K, the two sub-blocks' mins.
template <WType WT>
__device__ inline void DecodeCodes(const WRaw& r, int kb, uint16_t* dst, int u, float* scl, float* mnl) {
  const int s = kb & 3;
  float ds0, ds1, dm0 = 0.0F, dm1 = 0.0F, bias;
  uint32_t c0[4], c1[4];
  if constexpr (WT == WType::kQ4K) {
    const float d = static_cast<float>(__builtin_bit_cast(_Float16, static_cast<uint16_t>(r.a.x & 0xFFFFU)));
    const float dmin = static_cast<float>(__builtin_bit_cast(_Float16, static_cast<uint16_t>(r.a.x >> 16)));
    const unsigned sh = 8U * static_cast<unsigned>((2 * s) & 3);
    const uint32_t lo_s = r.a.y >> sh, lo_m = r.a.z >> sh, hi = r.a.w >> sh;
    uint32_t sc0, sc1, m0, m1;
    if (s < 2) {
      sc0 = lo_s & 63U;
      sc1 = (lo_s >> 8) & 63U;
      m0 = lo_m & 63U;
      m1 = (lo_m >> 8) & 63U;
    } else {
      sc0 = (hi & 15U) | (((lo_s >> 6) & 3U) << 4);
      sc1 = ((hi >> 8) & 15U) | (((lo_s >> 14) & 3U) << 4);
      m0 = ((hi >> 4) & 15U) | (((lo_m >> 6) & 3U) << 4);
      m1 = ((hi >> 12) & 15U) | (((lo_m >> 14) & 3U) << 4);
    }
    ds0 = d * static_cast<float>(sc0);
    dm0 = dmin * static_cast<float>(m0);
    ds1 = d * static_cast<float>(sc1);
    dm1 = dmin * static_cast<float>(m1);
    bias = 0.0F;
    const uint32_t w[4] = {r.b.x, r.b.y, r.b.z, r.b.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      c0[i] = w[i] & 0x0F0F0F0FU;
      c1[i] = (w[i] >> 4) & 0x0F0F0F0FU;
    }
  } else {
    const float d = static_cast<float>(__builtin_bit_cast(_Float16, static_cast<uint16_t>(r.s >> 16)));
    ds0 = d * static_cast<float>(static_cast<int8_t>(r.s & 0xFFU));
    ds1 = d * static_cast<float>(static_cast<int8_t>((r.s >> 8) & 0xFFU));
    bias = 32.0F;
    const unsigned nsh = (s & 1) ? 4U : 0U;
    const unsigned qsh = 4U * static_cast<unsigned>(s & 1);
    const uint32_t la[4] = {r.a.x, r.a.y, r.a.z, r.a.w};
    const uint32_t lb[4] = {r.b.x, r.b.y, r.b.z, r.b.w};
    const uint32_t qh[4] = {r.c.x, r.c.y, r.c.z, r.c.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      c0[i] = ((la[i] >> nsh) & 0x0F0F0F0FU) | (((qh[i] >> qsh) & 0x03030303U) << 4);
      c1[i] = ((lb[i] >> nsh) & 0x0F0F0F0FU) | (((qh[i] >> (qsh + 2U)) & 0x03030303U) << 4);
    }
  }
  uint32_t o0[8], o1[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      o0[2 * i + p] = PackF16(static_cast<float>((c0[i] >> (16 * p)) & 0xFFU) - bias,
                              static_cast<float>((c0[i] >> (16 * p + 8)) & 0xFFU) - bias);
      o1[2 * i + p] = PackF16(static_cast<float>((c1[i] >> (16 * p)) & 0xFFU) - bias,
                              static_cast<float>((c1[i] >> (16 * p + 8)) & 0xFFU) - bias);
    }
  }
  uint4* d0 = reinterpret_cast<uint4*>(dst + 16 * u);
  uint4* d1 = reinterpret_cast<uint4*>(dst + 32 + 16 * u);
  d0[0] = make_uint4(o0[0], o0[1], o0[2], o0[3]);
  d0[1] = make_uint4(o0[4], o0[5], o0[6], o0[7]);
  d1[0] = make_uint4(o1[0], o1[1], o1[2], o1[3]);
  d1[1] = make_uint4(o1[4], o1[5], o1[6], o1[7]);
  scl[u] = ds0;
  scl[2 + u] = ds1;
  if (u == 0) {
    mnl[0] = dm0;
    mnl[1] = dm1;
  }
}

template <int BM, WType WT, bool SWIGLU>
__global__ void __launch_bounds__(kThreads)
    GroupedExactKernel(const uint8_t* __restrict__ w0, const uint8_t* __restrict__ w1, std::size_t expert_bytes,
                       std::size_t row_bytes, const uint16_t* __restrict__ a_hi, const uint16_t* __restrict__ a_lo,
                       int a_by_pos, int x_share, const int* __restrict__ sorted, const int* __restrict__ offsets,
                       const int* __restrict__ counts, const int2* __restrict__ tiles, const int* __restrict__ n_tiles,
                       void* __restrict__ out, uint16_t* __restrict__ out_lo, int n, int k,
                       unsigned* __restrict__ overflow, int col_fast) {
  constexpr int TM = BM / 32;
  constexpr bool kMins = WT == WType::kQ4K;
  const int ti = col_fast ? static_cast<int>(blockIdx.y) : static_cast<int>(blockIdx.x);
  const int cb = col_fast ? static_cast<int>(blockIdx.x) : static_cast<int>(blockIdx.y);
  if (ti >= *n_tiles) return;
  __shared__ __attribute__((aligned(16))) uint16_t xsh[BM * kLd];
  __shared__ __attribute__((aligned(16))) uint16_t xsl[BM * kLd];
  __shared__ __attribute__((aligned(16))) uint16_t ws[kWRows * kLd];
  __shared__ float scl[kWRows * 4], mnl[kWRows * 2], xsum[BM * 2];
  __shared__ int64_t aoff[BM], ooff[BM];

  const int tid = static_cast<int>(threadIdx.x);
  const int2 tile = tiles[ti];
  const int e = tile.x, r0 = tile.y;
  const int rows = min(BM, counts[e] - r0);
  const int base = offsets[e] + r0;
  for (int r = tid; r < BM; r += kThreads) {
    if (r < rows) {
      const int pos = base + r;
      const int pair = sorted[pos];
      aoff[r] = static_cast<int64_t>(a_by_pos ? pos : pair / x_share) * k;
      ooff[r] = static_cast<int64_t>(SWIGLU ? pos : pair) * n;
    } else {
      aoff[r] = -1;
      ooff[r] = -1;
    }
  }
  const int wr = tid >> 1, wu = tid & 1;
  const uint8_t* wrow = nullptr;
  {
    const int ncols = SWIGLU ? 64 : kWRows;
    const int col = cb * ncols + (SWIGLU ? (wr & 63) : wr);
    const uint8_t* stack = (SWIGLU && wr >= 64) ? w1 : w0;
    if (col < n) wrow = stack + static_cast<std::size_t>(e) * expert_bytes + static_cast<std::size_t>(col) * row_bytes;
  }
  __syncthreads();

  constexpr int kXLoads = BM * (kBK / 8) / kThreads;
  static_assert(kXLoads >= 1, "whole x passes");
  const int nk = k / kBK;
  uint4 xh[kXLoads], xl[kXLoads];
  WRaw wraw{};
  const auto fetch = [&](int kb) {
#pragma unroll
    for (int i = 0; i < kXLoads; ++i) {
      const int idx = tid + kThreads * i, row = idx >> 3, seg = idx & 7;
      const int64_t off = aoff[row];
      const std::size_t o = static_cast<std::size_t>(off) + kb * kBK + seg * 8;
      xh[i] = off >= 0 ? *reinterpret_cast<const uint4*>(a_hi + o) : make_uint4(0, 0, 0, 0);
      xl[i] = off >= 0 ? *reinterpret_cast<const uint4*>(a_lo + o) : make_uint4(0, 0, 0, 0);
    }
    if (wrow != nullptr) wraw = FetchW<WT>(wrow, kb, wu);
  };

  const int wave = tid >> 5, wm = wave & 1, wn = wave >> 1, lane = tid & 31;
  const int b0 = SWIGLU ? 16 * wn : 32 * wn;
  const int b1 = SWIGLU ? 64 + 16 * wn : 32 * wn + 16;
  const int bcol[2] = {b0 + (lane & 15), b1 + (lane & 15)};
  f32x8 acc[TM][2];
#pragma unroll
  for (int i = 0; i < TM; ++i) acc[i][0] = acc[i][1] = f32x8{};

  fetch(0);
  for (int kb = 0; kb < nk; ++kb) {
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kXLoads; ++i) {
      const int idx = tid + kThreads * i, row = idx >> 3, seg = idx & 7;
      *reinterpret_cast<uint4*>(xsh + row * kLd + seg * 8) = xh[i];
      *reinterpret_cast<uint4*>(xsl + row * kLd + seg * 8) = xl[i];
    }
    if (wrow != nullptr) {
      DecodeCodes<WT>(wraw, kb, ws + wr * kLd, wu, scl + wr * 4, mnl + wr * 2);
    } else {
      uint4* z = reinterpret_cast<uint4*>(ws + wr * kLd);
      z[2 * wu] = z[2 * wu + 1] = z[4 + 2 * wu] = z[5 + 2 * wu] = make_uint4(0, 0, 0, 0);
      scl[wr * 4 + wu] = scl[wr * 4 + 2 + wu] = 0.0F;
      if (wu == 0) mnl[wr * 2] = mnl[wr * 2 + 1] = 0.0F;
    }
    __syncthreads();
    if constexpr (kMins) {  // the rows' 32-value sums of x (hi + lo), one (row, sub-block) a thread
      for (int q = tid; q < BM * 2; q += kThreads) {
        const int row = q >> 1, sb = q & 1;
        const uint16_t* ph = xsh + row * kLd + 32 * sb;
        const uint16_t* pl = xsl + row * kLd + 32 * sb;
        float sum = 0.0F;
#pragma unroll 8
        for (int j = 0; j < 32; ++j)
          sum += static_cast<float>(__builtin_bit_cast(_Float16, ph[j])) + static_cast<float>(__builtin_bit_cast(_Float16, pl[j]));
        xsum[q] = sum;
      }
      __syncthreads();
    }
    if (kb + 1 < nk) fetch(kb + 1);
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      const int kk = 16 * g;
      f16x16 ah[TM], al[TM];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        ah[i] = LoadFrag(xsh + (16 * (2 * i + wm)) * kLd + kk);
        al[i] = LoadFrag(xsl + (16 * (2 * i + wm)) * kLd + kk);
      }
      const f16x16 fb[2] = {LoadFrag(ws + b0 * kLd + kk), LoadFrag(ws + b1 * kLd + kk)};
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const float sc = scl[bcol[j] * 4 + g];
#pragma unroll
        for (int i = 0; i < TM; ++i) {
          f32x8 t = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah[i], fb[j], f32x8{});
          t = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al[i], fb[j], t);
#pragma unroll
          for (int v = 0; v < 8; ++v) acc[i][j][v] = __builtin_fmaf(sc, t[v], acc[i][j][v]);
        }
      }
    }
    if constexpr (kMins) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const float m0 = mnl[bcol[j] * 2], m1 = mnl[bcol[j] * 2 + 1];
#pragma unroll
        for (int i = 0; i < TM; ++i) {
#pragma unroll
          for (int v = 0; v < 8; ++v) {
            const int r = 16 * (2 * i + wm) + 2 * v + (lane >> 4);
            acc[i][j][v] = __builtin_fmaf(-m0, xsum[r * 2], __builtin_fmaf(-m1, xsum[r * 2 + 1], acc[i][j][v]));
          }
        }
      }
    }
  }

  if constexpr (SWIGLU) {
    uint16_t* h = static_cast<uint16_t*>(out);
    const int c = cb * 64 + 16 * wn + (lane & 15);
    bool over = false;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        const int r = 16 * (2 * i + wm) + 2 * v + (lane >> 4);
        if (r < rows && c < n) {
          const float g = acc[i][0][v], up = acc[i][1][v];
          float hv = (g / (1.0F + expf(-g))) * up;
          if (!(fabsf(hv) <= kF16Max)) {
            over = true;
            hv = hv != hv ? 0.0F : copysignf(kF16Max, hv);
          }
          const _Float16 hh = static_cast<_Float16>(hv);
          const _Float16 hl = static_cast<_Float16>(hv - static_cast<float>(hh));
          h[ooff[r] + c] = __builtin_bit_cast(uint16_t, hh);
          out_lo[ooff[r] + c] = __builtin_bit_cast(uint16_t, hl);
        }
      }
    }
    if (over) atomicOr(overflow, 1U);
  } else {
    float* y = static_cast<float*>(out);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int c = cb * kWRows + 32 * wn + 16 * j + (lane & 15);
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int v = 0; v < 8; ++v) {
          const int r = 16 * (2 * i + wm) + 2 * v + (lane >> 4);
          if (r < rows && c < n) y[ooff[r] + c] = acc[i][j][v];
        }
      }
    }
  }
}

__global__ void ToF16SplitKernel(const float* __restrict__ x, uint16_t* __restrict__ hi, uint16_t* __restrict__ lo,
                                 std::size_t n, unsigned* __restrict__ overflow) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i];
  if (!(fabsf(v) <= kF16Max)) {
    if (overflow != nullptr) atomicOr(overflow, 1U);
    v = v != v ? 0.0F : copysignf(kF16Max, v);
  }
  const _Float16 h = static_cast<_Float16>(v);
  hi[i] = __builtin_bit_cast(uint16_t, h);
  lo[i] = __builtin_bit_cast(uint16_t, static_cast<_Float16>(v - static_cast<float>(h)));
}

__global__ void CountKernel(const int32_t* __restrict__ ids, std::size_t pairs, int n_experts, int* __restrict__ counts) {
  const std::size_t p = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= pairs) return;
  const int id = ids[p];
  atomicAdd(&counts[(id < 0 || id >= n_experts) ? n_experts - 1 : id], 1);
}

// One block of 1,024 threads, thread e = expert e: offsets (exclusive scan of counts), fill = 0, the tile list.
__global__ void ScanKernel(const int* __restrict__ counts, int n_experts, int bm, int* __restrict__ offsets,
                           int* __restrict__ fill, int2* __restrict__ tiles, int* __restrict__ n_tiles) {
  __shared__ int sc[1024], st[1024];
  const int e = static_cast<int>(threadIdx.x);
  const int c = e < n_experts ? counts[e] : 0;
  const int t = (c + bm - 1) / bm;
  sc[e] = c;
  st[e] = t;
  __syncthreads();
  for (int off = 1; off < 1024; off <<= 1) {  // inclusive Hillis-Steele scan
    const int vc = e >= off ? sc[e - off] : 0;
    const int vt = e >= off ? st[e - off] : 0;
    __syncthreads();
    sc[e] += vc;
    st[e] += vt;
    __syncthreads();
  }
  if (e < n_experts) {
    offsets[e] = sc[e] - c;
    fill[e] = 0;
    const int t0 = st[e] - t;
    for (int j = 0; j < t; ++j) tiles[t0 + j] = make_int2(e, j * bm);
  }
  if (e == 1023) *n_tiles = st[1023];
}

__global__ void PlaceKernel(const int32_t* __restrict__ ids, std::size_t pairs, int n_experts,
                            const int* __restrict__ offsets, int* __restrict__ fill, int* __restrict__ sorted) {
  const std::size_t p = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= pairs) return;
  const int id = ids[p];
  const int e = (id < 0 || id >= n_experts) ? n_experts - 1 : id;
  sorted[offsets[e] + atomicAdd(&fill[e], 1)] = static_cast<int>(p);
}

__global__ void IdentityKernel(std::size_t pairs, int bm, int* __restrict__ counts, int* __restrict__ offsets,
                               int* __restrict__ sorted, int2* __restrict__ tiles, int* __restrict__ n_tiles) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t nt = (pairs + bm - 1) / bm;
  if (i < pairs) sorted[i] = static_cast<int>(i);
  if (i < nt) tiles[i] = make_int2(0, static_cast<int>(i) * bm);
  if (i == 0) {
    counts[0] = static_cast<int>(pairs);
    offsets[0] = 0;
    *n_tiles = static_cast<int>(nt);
  }
}

__global__ void ToF16Kernel(const float* __restrict__ x, uint16_t* __restrict__ xh, std::size_t n,
                            unsigned* __restrict__ overflow) {
  const std::size_t i = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  if (i >= n) return;
  bool over = false;
  float v[4];
  if (i + 4 <= n && (reinterpret_cast<std::uintptr_t>(x) & 15U) == 0) {
    const float4 q = *reinterpret_cast<const float4*>(x + i);
    v[0] = q.x, v[1] = q.y, v[2] = q.z, v[3] = q.w;
  } else {
    for (int j = 0; j < 4; ++j) v[j] = i + j < n ? x[i + j] : 0.0F;
  }
  for (int j = 0; j < 4; ++j) {
    if (!(fabsf(v[j]) <= kF16Max)) {
      over = true;
      v[j] = v[j] != v[j] ? 0.0F : copysignf(kF16Max, v[j]);
    }
    if (i + j < n) xh[i + j] = __builtin_bit_cast(uint16_t, static_cast<_Float16>(v[j]));
  }
  if (over && overflow != nullptr) atomicOr(overflow, 1U);
}

std::size_t Align(std::size_t v) { return (v + 255) & ~static_cast<std::size_t>(255); }

void Check(const char* what) {
  const hipError_t status = hipGetLastError();
  if (status != hipSuccess) {
    std::fprintf(stderr, "flerp::kgrouped: %s: %s\n", what, hipGetErrorString(status));
    std::abort();
  }
}

template <WType WT, bool SWIGLU>
void Launch(const void* w0, const void* w1, std::size_t expert_rows, const uint16_t* a_in, int a_by_pos,
            std::size_t x_share, const Plan& plan, void* out, std::size_t n, std::size_t k, unsigned* overflow,
            hipStream_t stream) {
  if (k % 256 != 0 || plan.max_tiles == 0) {
    std::fprintf(stderr, "flerp::kgrouped: K %zu is not a whole number of 256-value blocks\n", k);
    std::abort();
  }
  const std::size_t rb = RowBytes(WT, k);
  const std::size_t eb = expert_rows * rb;
  const int ncols = SWIGLU ? 64 : kWRows;
  const unsigned ncb = static_cast<unsigned>((n + ncols - 1) / ncols);
  static const int order = [] {  // FLERP_KG_ORDER: 1 = column blocks fastest (default), 0 = tiles fastest
    const char* v = std::getenv("FLERP_KG_ORDER");
    return v != nullptr ? std::atoi(v) : 1;
  }();
  const int col_fast = (order != 0 && plan.max_tiles <= 65535) ? 1 : 0;
  const dim3 grid = col_fast ? dim3(ncb, static_cast<unsigned>(plan.max_tiles))
                             : dim3(static_cast<unsigned>(plan.max_tiles), ncb);
  const auto* p0 = static_cast<const uint8_t*>(w0);
  const auto* p1 = static_cast<const uint8_t*>(w1);
#define FLERP_KG_LAUNCH(BMV)                                                                                         \
  GroupedKernel<BMV, WT, SWIGLU><<<grid, dim3(kThreads), 0, stream>>>(                                               \
      p0, p1, eb, rb, a_in, a_by_pos, static_cast<int>(x_share), plan.sorted, plan.offsets, plan.counts, plan.tiles, \
      plan.n_tiles, out, static_cast<int>(n), static_cast<int>(k), overflow, col_fast)
  switch (plan.bm) {
    case 32: FLERP_KG_LAUNCH(32); break;
    case 64: FLERP_KG_LAUNCH(64); break;
    case 128: FLERP_KG_LAUNCH(128); break;
    default:
      std::fprintf(stderr, "flerp::kgrouped: BM %d is not 32, 64 or 128\n", plan.bm);
      std::abort();
  }
#undef FLERP_KG_LAUNCH
  Check(SWIGLU ? "GateUp" : "Down/Linear");
}

template <WType WT, bool SWIGLU>
void LaunchExact(const void* w0, const void* w1, std::size_t expert_rows, const uint16_t* a_hi, const uint16_t* a_lo,
                 int a_by_pos, std::size_t x_share, const Plan& plan, void* out, uint16_t* out_lo, std::size_t n,
                 std::size_t k, unsigned* overflow, hipStream_t stream) {
  if (k % 256 != 0 || plan.max_tiles == 0 || plan.max_tiles > 65535) {
    std::fprintf(stderr, "flerp::kgrouped: exact: K %zu or %zu tiles out of range\n", k, plan.max_tiles);
    std::abort();
  }
  const std::size_t rb = RowBytes(WT, k);
  const std::size_t eb = expert_rows * rb;
  const int ncols = SWIGLU ? 64 : kWRows;
  const dim3 grid(static_cast<unsigned>((n + ncols - 1) / ncols), static_cast<unsigned>(plan.max_tiles));
  const auto* p0 = static_cast<const uint8_t*>(w0);
  const auto* p1 = static_cast<const uint8_t*>(w1);
#define FLERP_KGX_LAUNCH(BMV)                                                                                         \
  GroupedExactKernel<BMV, WT, SWIGLU><<<grid, dim3(kThreads), 0, stream>>>(                                           \
      p0, p1, eb, rb, a_hi, a_lo, a_by_pos, static_cast<int>(x_share), plan.sorted, plan.offsets, plan.counts,        \
      plan.tiles, plan.n_tiles, out, out_lo, static_cast<int>(n), static_cast<int>(k), overflow, 1)
  switch (plan.bm) {
    case 32: FLERP_KGX_LAUNCH(32); break;
    case 64: FLERP_KGX_LAUNCH(64); break;
    case 128: FLERP_KGX_LAUNCH(128); break;
    default:
      std::fprintf(stderr, "flerp::kgrouped: BM %d is not 32, 64 or 128\n", plan.bm);
      std::abort();
  }
#undef FLERP_KGX_LAUNCH
  Check(SWIGLU ? "GateUpExact" : "DownExact/LinearExact");
}

}  // namespace

std::size_t RowBytes(WType type, std::size_t k) noexcept {
  return (k / 256) * (type == WType::kQ4K ? kQ4KBytes : kQ6KBytes);
}

std::size_t MaxTiles(std::size_t pairs, std::size_t n_experts, int bm) noexcept {
  return (pairs + bm - 1) / bm + std::min(n_experts, pairs);
}

std::size_t PlanBytes(std::size_t pairs, std::size_t n_experts, int bm) noexcept {
  return 3 * Align(n_experts * sizeof(int)) + Align(pairs * sizeof(int)) +
         Align(MaxTiles(pairs, n_experts, bm) * sizeof(int2)) + Align(sizeof(int));
}

Plan MakePlan(void* scratch, std::size_t pairs, std::size_t n_experts, int bm) noexcept {
  Plan p;
  auto* b = static_cast<char*>(scratch);
  p.counts = reinterpret_cast<int*>(b);
  b += Align(n_experts * sizeof(int));
  p.offsets = reinterpret_cast<int*>(b);
  b += Align(n_experts * sizeof(int));
  p.fill = reinterpret_cast<int*>(b);
  b += Align(n_experts * sizeof(int));
  p.sorted = reinterpret_cast<int*>(b);
  b += Align(pairs * sizeof(int));
  p.tiles = reinterpret_cast<int2*>(b);
  b += Align(MaxTiles(pairs, n_experts, bm) * sizeof(int2));
  p.n_tiles = reinterpret_cast<int*>(b);
  p.pairs = pairs;
  p.n_experts = n_experts;
  p.bm = bm;
  p.max_tiles = MaxTiles(pairs, n_experts, bm);
  return p;
}

int ChooseBm(std::size_t pairs, std::size_t n_experts) noexcept {
  const std::size_t active = std::min(pairs, n_experts);
  const std::size_t per = active == 0 ? 0 : pairs / active;
  // gate/results/prefill-ornith/speed2-s2.jsonl (K, column-fast order): BM 32 best at 20 rows an expert (1,024 rows),
  // BM 128 at 80 and 320 (4,096, 16,384); BM 64 was never the fastest.
  return per <= 40 ? 32 : 128;
}

void Group(const std::int32_t* ids, Plan& plan, hipStream_t stream) {
  if (plan.n_experts == 0 || plan.n_experts > 1024 || plan.pairs == 0) {
    std::fprintf(stderr, "flerp::kgrouped::Group: %zu experts, %zu pairs\n", plan.n_experts, plan.pairs);
    std::abort();
  }
  const unsigned blocks = static_cast<unsigned>((plan.pairs + 255) / 256);
  if (ids == nullptr) {
    const std::size_t nt = (plan.pairs + plan.bm - 1) / plan.bm;
    const unsigned ib = static_cast<unsigned>((std::max(plan.pairs, nt) + 255) / 256);
    IdentityKernel<<<ib, 256, 0, stream>>>(plan.pairs, plan.bm, plan.counts, plan.offsets, plan.sorted, plan.tiles,
                                          plan.n_tiles);
    Check("Group identity");
    return;
  }
  const int ne = static_cast<int>(plan.n_experts);
  (void)hipMemsetAsync(plan.counts, 0, plan.n_experts * sizeof(int), stream);
  CountKernel<<<blocks, 256, 0, stream>>>(ids, plan.pairs, ne, plan.counts);
  ScanKernel<<<1, 1024, 0, stream>>>(plan.counts, ne, plan.bm, plan.offsets, plan.fill, plan.tiles, plan.n_tiles);
  PlaceKernel<<<blocks, 256, 0, stream>>>(ids, plan.pairs, ne, plan.offsets, plan.fill, plan.sorted);
  Check("Group");
}

void ToF16(const float* x, std::uint16_t* xh, std::size_t n, unsigned* overflow, hipStream_t stream) {
  if (n == 0) return;
  ToF16Kernel<<<static_cast<unsigned>((n / 4 + 255) / 256 + 1), 256, 0, stream>>>(x, xh, n, overflow);
  Check("ToF16");
}

void GateUp(WType type, const void* w_gate, const void* w_up, const std::uint16_t* xh, std::size_t x_share,
            const Plan& plan, std::uint16_t* h, std::size_t n, std::size_t k, unsigned* overflow, hipStream_t stream) {
  if (type == WType::kQ4K)
    Launch<WType::kQ4K, true>(w_gate, w_up, n, xh, 0, x_share, plan, h, n, k, overflow, stream);
  else
    Launch<WType::kQ6K, true>(w_gate, w_up, n, xh, 0, x_share, plan, h, n, k, overflow, stream);
}

void Down(WType type, const void* w_down, const std::uint16_t* h, const Plan& plan, float* y, std::size_t n,
          std::size_t k, hipStream_t stream) {
  if (type == WType::kQ4K)
    Launch<WType::kQ4K, false>(w_down, nullptr, n, h, 1, 1, plan, y, n, k, nullptr, stream);
  else
    Launch<WType::kQ6K, false>(w_down, nullptr, n, h, 1, 1, plan, y, n, k, nullptr, stream);
}

void Linear(WType type, const void* w, const std::uint16_t* xh, const Plan& plan, float* y, std::size_t n,
            std::size_t k, hipStream_t stream) {
  if (type == WType::kQ4K)
    Launch<WType::kQ4K, false>(w, nullptr, n, xh, 0, 1, plan, y, n, k, nullptr, stream);
  else
    Launch<WType::kQ6K, false>(w, nullptr, n, xh, 0, 1, plan, y, n, k, nullptr, stream);
}

int ChooseBmExact(std::size_t pairs, std::size_t n_experts) noexcept {
  const std::size_t active = std::min(pairs, n_experts);
  const std::size_t per = active == 0 ? 0 : pairs / active;
  return per <= 40 ? 32 : 64;
}

void ToF16Split(const float* x, std::uint16_t* hi, std::uint16_t* lo, std::size_t n, unsigned* overflow,
                hipStream_t stream) {
  if (n == 0) return;
  ToF16SplitKernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(x, hi, lo, n, overflow);
  Check("ToF16Split");
}

void GateUpExact(WType type, const void* w_gate, const void* w_up, const std::uint16_t* x_hi,
                 const std::uint16_t* x_lo, std::size_t x_share, const Plan& plan, std::uint16_t* h_hi,
                 std::uint16_t* h_lo, std::size_t n, std::size_t k, unsigned* overflow, hipStream_t stream) {
  if (type == WType::kQ4K)
    LaunchExact<WType::kQ4K, true>(w_gate, w_up, n, x_hi, x_lo, 0, x_share, plan, h_hi, h_lo, n, k, overflow, stream);
  else
    LaunchExact<WType::kQ6K, true>(w_gate, w_up, n, x_hi, x_lo, 0, x_share, plan, h_hi, h_lo, n, k, overflow, stream);
}

void DownExact(WType type, const void* w_down, const std::uint16_t* h_hi, const std::uint16_t* h_lo, const Plan& plan,
               float* y, std::size_t n, std::size_t k, hipStream_t stream) {
  if (type == WType::kQ4K)
    LaunchExact<WType::kQ4K, false>(w_down, nullptr, n, h_hi, h_lo, 1, 1, plan, y, nullptr, n, k, nullptr, stream);
  else
    LaunchExact<WType::kQ6K, false>(w_down, nullptr, n, h_hi, h_lo, 1, 1, plan, y, nullptr, n, k, nullptr, stream);
}

void LinearExact(WType type, const void* w, const std::uint16_t* x_hi, const std::uint16_t* x_lo, const Plan& plan,
                 float* y, std::size_t n, std::size_t k, hipStream_t stream) {
  if (type == WType::kQ4K)
    LaunchExact<WType::kQ4K, false>(w, nullptr, n, x_hi, x_lo, 0, 1, plan, y, nullptr, n, k, nullptr, stream);
  else
    LaunchExact<WType::kQ6K, false>(w, nullptr, n, x_hi, x_lo, 0, 1, plan, y, nullptr, n, k, nullptr, stream);
}

}  // namespace flerp::kgrouped
