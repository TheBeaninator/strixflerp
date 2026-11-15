// SPDX-License-Identifier: MIT
// PREFILL-ORNITH kernel bench (gate/prefill-ornith.toml): kernel.flerp.moe_kgrouped against the current prefill path
// (kernel.flerp.moe_kquant IndexedExperts, its wide Q4_K / Q6_K routes copied below unchanged in arithmetic) at
// Ornith-1.5 397B shapes, synthetic codes in the real GGUF block layouts. One binary, no model, no gufo tree:
//
//   prefill-ornith-bench --check  [--tokens 192] [--down q6k|q4k]     correctness against a float64 reference
//   prefill-ornith-bench --speed  --rows N [--down q6k|q4k] [--skew] [--bm 0|32|64|128] [--reps R] [--old 0|1]
//   prefill-ornith-bench --dense  --rows N --n N --k K [--type q4k|q6k] [--reps R]
//
// One JSON object a line on stdout. Build: hipcc -O3 --offload-arch=gfx1151 -std=c++17
//   -I ../catalog/flerp/kernel/moe_kgrouped/src prefill_ornith_bench.hip
//   ../catalog/flerp/kernel/moe_kgrouped/src/flerp/moe_kgrouped/moe_kgrouped.hip -o prefill-ornith-bench -lpthread
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "flerp/moe_kgrouped/moe_kgrouped.hpp"

#define CK(x)                                                                               \
  do {                                                                                      \
    hipError_t e_ = (x);                                                                    \
    if (e_ != hipSuccess) {                                                                 \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
      std::exit(2);                                                                         \
    }                                                                                       \
  } while (0)

namespace kg = flerp::kgrouped;

// ---------------------------------------------------------------------------------------------------------------
// The current path: moe_kquant.hip IndexedExpertsKernel with RowDotWide<kQ4K / kQ6K> (lines 72-171, 271-300 at
// 0ac108f), and gufo's helpers it calls (quant_ops.hpp Q4KBlock, Q6KBlock, GetQKScaleMin, SubBytes), copied with
// the same lane mapping, summation order and expressions.
namespace old {
struct Q4KBlock {
  __half d;
  __half dmin;
  uint8_t scales[12];
  uint8_t qs[128];
};
struct Q6KBlock {
  uint8_t ql[128];
  uint8_t qh[64];
  int8_t scales[16];
  __half d;
};
struct QuantSub16 {
  union {
    int8_t q[16];
    uint32_t w[4];
  };
  float scale;
  float offset;
};
__device__ inline void GetQKScaleMin(std::size_t j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
  if (j < 4) {
    sc = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}
__device__ inline uint32_t SubBytes(uint32_t v, uint32_t bias) { return ((v | 0x80808080U) - bias) ^ 0x80808080U; }
__device__ inline float WaveSum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off);
  return v;
}
__device__ inline uint4 Load16Align2(const uint8_t* p) {
  uint4 v;
  __builtin_memcpy(&v, __builtin_assume_aligned(p, 2), 16);
  return v;
}
__device__ inline void LoadX16(const float* __restrict__ xb, float xv[16]) {
  const float4* x4 = reinterpret_cast<const float4*>(xb);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float4 v = x4[i];
    xv[4 * i] = v.x, xv[4 * i + 1] = v.y, xv[4 * i + 2] = v.z, xv[4 * i + 3] = v.w;
  }
}
template <bool Q4>
__device__ inline float RowDotWide(const uint8_t* __restrict__ row, const float* __restrict__ x, std::size_t k,
                                   std::size_t lane) {
  float sumf = 0.0F;
  const std::size_t nsub = k / 16;
  for (std::size_t sb = lane; sb < nsub; sb += 32) {
    QuantSub16 sub;
    const std::size_t sb32 = (sb / 2) % 8;
    if constexpr (Q4) {
      const uint8_t* blk = row + (sb / 16) * sizeof(Q4KBlock);
      const std::size_t base = ((sb32 / 2) * 32) + ((sb % 2) * 16);
      const unsigned shift = 4U * static_cast<unsigned>(sb32 & 1U);
      const uint4 p = *reinterpret_cast<const uint4*>(blk + offsetof(Q4KBlock, qs) + base);
      sub.w[0] = (p.x >> shift) & 0x0F0F0F0FU;
      sub.w[1] = (p.y >> shift) & 0x0F0F0F0FU;
      sub.w[2] = (p.z >> shift) & 0x0F0F0F0FU;
      sub.w[3] = (p.w >> shift) & 0x0F0F0F0FU;
      uint8_t sc = 0, m = 0;
      GetQKScaleMin(sb32, blk + offsetof(Q4KBlock, scales), sc, m);
      const __half* dh = reinterpret_cast<const __half*>(blk);
      sub.scale = __half2float(dh[0]) * static_cast<float>(sc);
      sub.offset = __half2float(dh[1]) * static_cast<float>(m);
    } else {
      const uint8_t* blk = row + (sb / 16) * sizeof(Q6KBlock);
      const std::size_t half = sb32 / 4, segment = sb32 % 4, lane0 = (sb % 2) * 16;
      const std::size_t ql_base = (half * 64) + (((segment & 1U) != 0U) ? 32 : 0);
      const unsigned ql_shift = (segment >= 2) ? 4U : 0U;
      const unsigned qh_shift = 2U * static_cast<unsigned>(segment);
      const uint4 lo = Load16Align2(blk + offsetof(Q6KBlock, ql) + ql_base + lane0);
      const uint4 hi = Load16Align2(blk + offsetof(Q6KBlock, qh) + (half * 32) + lane0);
      const uint32_t low_bits[4] = {lo.x, lo.y, lo.z, lo.w};
      const uint32_t high_bits[4] = {hi.x, hi.y, hi.z, hi.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t low = (low_bits[i] >> ql_shift) & 0x0F0F0F0FU;
        const uint32_t high = (high_bits[i] >> qh_shift) & 0x03030303U;
        sub.w[i] = SubBytes(low | (high << 4U), 0x20202020U);
      }
      const std::size_t scale_index = (half * 8) + (segment * 2) + (sb % 2);
      const auto* scales = reinterpret_cast<const int8_t*>(blk + offsetof(Q6KBlock, scales));
      const __half* dh = reinterpret_cast<const __half*>(blk + offsetof(Q6KBlock, d));
      sub.scale = __half2float(*dh) * static_cast<float>(scales[scale_index]);
      sub.offset = 0.0F;
    }
    float xv[16];
    LoadX16(x + (sb * 16), xv);
    float dot = 0.0F, xsum = 0.0F;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      dot += static_cast<float>(sub.q[j]) * xv[j];
      xsum += xv[j];
    }
    sumf += (sub.scale * dot) - (sub.offset * xsum);
  }
  return WaveSum(sumf);
}
constexpr std::size_t kMaxGridRows = 32768;
template <bool Q4>
__launch_bounds__(128) __global__
    void IndexedExpertsKernel(const void* __restrict__ w_a, std::size_t bytes_a, const void* __restrict__ w_b,
                              std::size_t bytes_b, const int32_t* __restrict__ ids, const float* __restrict__ x,
                              float* __restrict__ y, std::size_t x_share, std::size_t rows, std::size_t row_elements,
                              std::size_t n_experts, std::size_t first_row) {
  const std::size_t lane = threadIdx.x & 31U;
  const std::size_t m = (static_cast<std::size_t>(blockIdx.x) * 4) + (threadIdx.x >> 5U);
  if (m >= rows) return;
  const std::size_t ts = first_row + blockIdx.y;
  std::size_t e = 0;
  if (ids != nullptr) {
    const int32_t id = ids[ts];
    e = (id < 0 || static_cast<std::size_t>(id) >= n_experts) ? n_experts - 1 : static_cast<std::size_t>(id);
  }
  const float* xr = x + (ts / x_share) * row_elements;
  constexpr std::size_t kBlk = Q4 ? 144 : 210;
  const std::size_t rb = row_elements / 256 * kBlk;
  const float a = RowDotWide<Q4>(static_cast<const uint8_t*>(w_a) + e * bytes_a + m * rb, xr, row_elements, lane);
  float out = a;
  if (w_b != nullptr) {
    const float b = RowDotWide<Q4>(static_cast<const uint8_t*>(w_b) + e * bytes_b + m * rb, xr, row_elements, lane);
    out = (a / (1.0F + expf(-a))) * b;
  }
  if (lane == 0) y[ts * rows + m] = out;
}
void IndexedExperts(bool q4, const void* w_a, const void* w_b, const int32_t* ids, const float* x, float* y,
                    std::size_t tokens, std::size_t slots, std::size_t x_share, std::size_t rows,
                    std::size_t row_elements, std::size_t n_experts, hipStream_t stream) {
  const std::size_t bytes = rows * (row_elements / 256) * (q4 ? 144 : 210);
  const auto grid_x = static_cast<uint32_t>((rows + 3) / 4);
  const std::size_t total = tokens * slots;
  for (std::size_t first = 0; first < total; first += kMaxGridRows) {
    const std::size_t n = std::min(kMaxGridRows, total - first);
    const dim3 grid(grid_x, static_cast<uint32_t>(n));
    if (q4)
      IndexedExpertsKernel<true><<<grid, 128, 0, stream>>>(w_a, bytes, w_b, bytes, ids, x, y, x_share, rows,
                                                           row_elements, n_experts, first);
    else
      IndexedExpertsKernel<false><<<grid, 128, 0, stream>>>(w_a, bytes, w_b, bytes, ids, x, y, x_share, rows,
                                                            row_elements, n_experts, first);
  }
  CK(hipGetLastError());
}
}  // namespace old

// ---------------------------------------------------------------------------------------------------------------
// Synthetic blocks on the device: random codes, scales and mins; d, dmin set so values look like trained weights
// (|w| up to ~0.06, sub-blocks roughly centred).
__device__ inline uint32_t Hash(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return static_cast<uint32_t>(x);
}
__device__ inline float U01(uint64_t k) { return (Hash(k) >> 8) * (1.0F / 16777216.0F); }
__global__ void FillBlocks(uint8_t* w, std::size_t nblocks, int q4, uint64_t seed) {
  const std::size_t b = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= nblocks) return;
  const std::size_t bytes = q4 ? 144 : 210;
  uint8_t* p = w + b * bytes;
  const uint64_t s = seed * 0x9E3779B97F4A7C15ULL + b * 1000003ULL;
  for (std::size_t i = 0; i < bytes; i += 4) {
    const uint32_t r = Hash(s + i);
    for (std::size_t j = 0; j < 4 && i + j < bytes; ++j) p[i + j] = static_cast<uint8_t>(r >> (8 * j));
  }
  if (q4) {
    const float d = 2.0e-4F * (0.5F + U01(s + 7777));
    const float dmin = d * 7.5F * (0.8F + 0.4F * U01(s + 8888));
    const _Float16 hd = static_cast<_Float16>(d), hm = static_cast<_Float16>(dmin);
    __builtin_memcpy(p, &hd, 2);
    __builtin_memcpy(p + 2, &hm, 2);
  } else {
    for (int i = 0; i < 16; ++i) p[192 + i] = static_cast<uint8_t>(1 + (Hash(s + 50 + i) % 127));
    const _Float16 hd = static_cast<_Float16>(1.5e-5F * (0.5F + U01(s + 9999)));
    __builtin_memcpy(p + 208, &hd, 2);
  }
}
__global__ void FillNormal(float* x, std::size_t n, uint64_t seed) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float u1 = fmaxf(U01(seed * 31 + 2 * i), 1e-7F), u2 = U01(seed * 31 + 2 * i + 1);
  x[i] = sqrtf(-2.0F * logf(u1)) * cosf(6.2831853F * u2);
}

struct Stack {
  uint8_t* d = nullptr;
  std::size_t bytes = 0;
  bool q4 = true;
};
Stack MakeStack(bool q4, std::size_t experts, std::size_t rows, std::size_t k, uint64_t seed) {
  Stack s;
  s.q4 = q4;
  const std::size_t nblocks = experts * rows * (k / 256);
  s.bytes = nblocks * (q4 ? 144 : 210);
  CK(hipMalloc(&s.d, s.bytes + 64));
  FillBlocks<<<static_cast<unsigned>((nblocks + 255) / 256), 256>>>(s.d, nblocks, q4 ? 1 : 0, seed);
  CK(hipGetLastError());
  return s;
}

// ---------------------------------------------------------------------------------------------------------------
// float64 reference: llama.cpp dequantize_row_q4_K / q6_K (ggml-quants.c) written out again, double arithmetic.
double H2D(uint16_t h) {
  const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
  double v;
  if (e == 0) v = std::ldexp(static_cast<double>(m), -24);
  else if (e == 31) v = m ? NAN : INFINITY;
  else v = std::ldexp(static_cast<double>(m | 1024), static_cast<int>(e) - 25);
  return s ? -v : v;
}
void DequantRow(bool q4, const uint8_t* row, std::size_t k, double* out) {
  for (std::size_t b = 0; b < k / 256; ++b) {
    double* y = out + 256 * b;
    if (q4) {
      const uint8_t* x = row + 144 * b;
      uint16_t dh, mh;
      std::memcpy(&dh, x, 2);
      std::memcpy(&mh, x + 2, 2);
      const double d = H2D(dh), dmin = H2D(mh);
      const uint8_t* sc = x + 4;
      const uint8_t* q = x + 16;
      int is = 0;
      for (int j = 0; j < 256; j += 64) {
        uint8_t s1, m1, s2, m2;
        auto gsm = [&](int jj, uint8_t& s, uint8_t& m) {
          if (jj < 4) {
            s = sc[jj] & 63;
            m = sc[jj + 4] & 63;
          } else {
            s = (sc[jj + 4] & 0xF) | ((sc[jj - 4] >> 6) << 4);
            m = (sc[jj + 4] >> 4) | ((sc[jj] >> 6) << 4);
          }
        };
        gsm(is, s1, m1);
        gsm(is + 1, s2, m2);
        for (int l = 0; l < 32; ++l) *y++ = d * s1 * (q[l] & 0xF) - dmin * m1;
        for (int l = 0; l < 32; ++l) *y++ = d * s2 * (q[l] >> 4) - dmin * m2;
        q += 32;
        is += 2;
      }
    } else {
      const uint8_t* x = row + 210 * b;
      const uint8_t* ql = x;
      const uint8_t* qh = x + 128;
      const int8_t* sc = reinterpret_cast<const int8_t*>(x + 192);
      uint16_t dh;
      std::memcpy(&dh, x + 208, 2);
      const double d = H2D(dh);
      for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
          const int is = l / 16;
          const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
          const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
          const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
          const int q4v = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
          y[l] = d * sc[is] * q1;
          y[l + 32] = d * sc[is + 2] * q2;
          y[l + 64] = d * sc[is + 4] * q3;
          y[l + 96] = d * sc[is + 6] * q4v;
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
      }
    }
  }
}
// One expert's matrix [rows, k] dequantized from device (cached by expert id).
std::vector<double> DequantExpert(const Stack& s, std::size_t e, std::size_t rows, std::size_t k) {
  const std::size_t rb = (k / 256) * (s.q4 ? 144 : 210);
  std::vector<uint8_t> raw(rows * rb);
  CK(hipMemcpy(raw.data(), s.d + e * rows * rb, raw.size(), hipMemcpyDeviceToHost));
  std::vector<double> w(rows * k);
  for (std::size_t r = 0; r < rows; ++r) DequantRow(s.q4, raw.data() + r * rb, k, w.data() + r * k);
  return w;
}

struct Err {
  double rel_rms = 0, max_abs_over_rms = 0, ref_rms = 0;
  std::size_t n = 0, nonfinite = 0;
};
Err Compare(const std::vector<double>& ref, const std::vector<float>& got) {
  Err e;
  double ss = 0, se = 0, mx = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (!std::isfinite(got[i])) ++e.nonfinite;
    const double d = static_cast<double>(got[i]) - ref[i];
    ss += ref[i] * ref[i];
    se += d * d;
    mx = std::max(mx, std::fabs(d));
  }
  e.n = ref.size();
  e.ref_rms = std::sqrt(ss / std::max<std::size_t>(1, ref.size()));
  e.rel_rms = std::sqrt(se / std::max(ss, 1e-300));
  e.max_abs_over_rms = mx / std::max(e.ref_rms, 1e-300);
  return e;
}
std::string ErrJson(const char* name, const Err& e) {
  char b[256];
  std::snprintf(b, sizeof b, "\"%s\":{\"rel_rms\":%.4g,\"max_abs_over_rms\":%.4g,\"ref_rms\":%.4g,\"n\":%zu,\"nonfinite\":%zu}",
                name, e.rel_rms, e.max_abs_over_rms, e.ref_rms, e.n, e.nonfinite);
  return b;
}

// ---------------------------------------------------------------------------------------------------------------
constexpr std::size_t kHidden = 4096, kFf = 1024, kExperts = 512, kUsed = 10, kSharedFf = 1024;

struct Moe {
  Stack gate, up, down, sgate, sup, sdown;
};
Moe MakeMoe(bool down_q6) {
  Moe m;
  m.gate = MakeStack(true, kExperts, kFf, kHidden, 11);
  m.up = MakeStack(true, kExperts, kFf, kHidden, 12);
  m.down = MakeStack(!down_q6, kExperts, kHidden, kFf, 13);
  m.sgate = MakeStack(true, 1, kSharedFf, kHidden, 21);
  m.sup = MakeStack(true, 1, kSharedFf, kHidden, 22);
  m.sdown = MakeStack(!down_q6, 1, kHidden, kSharedFf, 23);
  CK(hipDeviceSynchronize());
  return m;
}

// Routing: uniform (10 distinct of 512), skewed (a quarter of the picks from 32 hot experts), hot (4 experts every
// token: groups larger than every BM, for the tile boundaries).
std::vector<int32_t> Route(std::size_t tokens, int mode, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<int32_t> ids(tokens * kUsed);
  for (std::size_t t = 0; t < tokens; ++t) {
    std::vector<int32_t> pick;
    if (mode == 2) pick = {3, 77, 200, 511};
    while (pick.size() < kUsed) {
      int32_t e;
      if (mode == 1 && (rng() % 4) == 0) e = static_cast<int32_t>(rng() % 32) * 16;
      else e = static_cast<int32_t>(rng() % kExperts);
      if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
    }
    std::shuffle(pick.begin(), pick.end(), rng);
    std::copy(pick.begin(), pick.end(), ids.begin() + t * kUsed);
  }
  return ids;
}

struct NewBufs {
  void* plan_mem = nullptr;
  void* splan_mem = nullptr;
  uint16_t* xh = nullptr;
  uint16_t* h = nullptr;
  uint16_t* sh = nullptr;
  uint16_t* xl = nullptr;   // exact variant: the lo halves
  uint16_t* hl = nullptr;
  uint16_t* shl = nullptr;
  unsigned* overflow = nullptr;
};
bool g_exact = false;       // --exact: the exact-code, hi/lo F16 variant
NewBufs AllocNew(std::size_t tokens) {
  NewBufs b;
  CK(hipMalloc(&b.plan_mem, kg::PlanBytes(tokens * kUsed, kExperts, 32)));
  CK(hipMalloc(&b.splan_mem, kg::PlanBytes(tokens, 1, 32)));
  CK(hipMalloc(&b.xh, tokens * kHidden * 2));
  CK(hipMalloc(&b.h, tokens * kUsed * kFf * 2));
  CK(hipMalloc(&b.sh, tokens * kSharedFf * 2));
  CK(hipMalloc(&b.xl, tokens * kHidden * 2));
  CK(hipMalloc(&b.hl, tokens * kUsed * kFf * 2));
  CK(hipMalloc(&b.shl, tokens * kSharedFf * 2));
  CK(hipMalloc(&b.overflow, 4));
  CK(hipMemset(b.overflow, 0, 4));
  return b;
}
void FreeNew(NewBufs& b) {
  (void)hipFree(b.plan_mem), (void)hipFree(b.splan_mem), (void)hipFree(b.xh), (void)hipFree(b.h), (void)hipFree(b.sh),
      (void)hipFree(b.overflow), (void)hipFree(b.xl), (void)hipFree(b.hl), (void)hipFree(b.shl);
}

// The new routed pass: group, x to F16, gate/up + SwiGLU, down into y [token * used + slot, hidden].
void NewRouted(const Moe& m, const int32_t* ids, const float* x, float* y, std::size_t tokens, int bm, NewBufs& b,
               hipStream_t s, hipEvent_t* ev = nullptr) {
  kg::Plan plan = kg::MakePlan(b.plan_mem, tokens * kUsed, kExperts, bm);
  if (ev) CK(hipEventRecord(ev[0], s));
  kg::Group(ids, plan, s);
  const kg::WType dt = m.down.q4 ? kg::WType::kQ4K : kg::WType::kQ6K;
  if (g_exact) {
    kg::ToF16Split(x, b.xh, b.xl, tokens * kHidden, b.overflow, s);
    if (ev) CK(hipEventRecord(ev[1], s));
    kg::GateUpExact(kg::WType::kQ4K, m.gate.d, m.up.d, b.xh, b.xl, kUsed, plan, b.h, b.hl, kFf, kHidden, b.overflow, s);
    if (ev) CK(hipEventRecord(ev[2], s));
    kg::DownExact(dt, m.down.d, b.h, b.hl, plan, y, kHidden, kFf, s);
    if (ev) CK(hipEventRecord(ev[3], s));
    return;
  }
  kg::ToF16(x, b.xh, tokens * kHidden, b.overflow, s);
  if (ev) CK(hipEventRecord(ev[1], s));
  kg::GateUp(kg::WType::kQ4K, m.gate.d, m.up.d, b.xh, kUsed, plan, b.h, kFf, kHidden, b.overflow, s);
  if (ev) CK(hipEventRecord(ev[2], s));
  kg::Down(dt, m.down.d, b.h, plan, y, kHidden, kFf, s);
  if (ev) CK(hipEventRecord(ev[3], s));
}
// The new shared expert (x already F16 in b.xh): identity group over the tokens.
void NewShared(const Moe& m, float* y, std::size_t tokens, int bm, NewBufs& b, hipStream_t s) {
  kg::Plan plan = kg::MakePlan(b.splan_mem, tokens, 1, bm);
  kg::Group(nullptr, plan, s);
  const kg::WType dt = m.sdown.q4 ? kg::WType::kQ4K : kg::WType::kQ6K;
  if (g_exact) {
    kg::GateUpExact(kg::WType::kQ4K, m.sgate.d, m.sup.d, b.xh, b.xl, 1, plan, b.sh, b.shl, kSharedFf, kHidden, b.overflow, s);
    kg::DownExact(dt, m.sdown.d, b.sh, b.shl, plan, y, kHidden, kSharedFf, s);
    return;
  }
  kg::GateUp(kg::WType::kQ4K, m.sgate.d, m.sup.d, b.xh, 1, plan, b.sh, kSharedFf, kHidden, b.overflow, s);
  kg::Down(dt, m.sdown.d, b.sh, plan, y, kHidden, kSharedFf, s);
}
void OldRouted(const Moe& m, const int32_t* ids, const float* x, float* act, float* y, std::size_t tokens,
               hipStream_t s) {
  old::IndexedExperts(true, m.gate.d, m.up.d, ids, x, act, tokens, kUsed, kUsed, kFf, kHidden, kExperts, s);
  old::IndexedExperts(m.down.q4, m.down.d, nullptr, ids, act, y, tokens, kUsed, 1, kHidden, kFf, kExperts, s);
}
void OldShared(const Moe& m, const float* x, float* act, float* y, std::size_t tokens, hipStream_t s) {
  old::IndexedExperts(true, m.sgate.d, m.sup.d, nullptr, x, act, tokens, 1, 1, kSharedFf, kHidden, 1, s);
  old::IndexedExperts(m.sdown.q4, m.sdown.d, nullptr, nullptr, act, y, tokens, 1, 1, kHidden, kSharedFf, 1, s);
}

std::vector<float> D2H(const float* d, std::size_t n) {
  std::vector<float> h(n);
  CK(hipMemcpy(h.data(), d, n * 4, hipMemcpyDeviceToHost));
  return h;
}

template <class F>
void ParallelFor(std::size_t n, F f) {
  const unsigned nt = std::max(1U, std::min(16U, std::thread::hardware_concurrency()));
  std::vector<std::thread> th;
  for (unsigned t = 0; t < nt; ++t)
    th.emplace_back([&, t] {
      for (std::size_t i = t; i < n; i += nt) f(i);
    });
  for (auto& x : th) x.join();
}

int Check(std::size_t tokens, bool down_q6) {
  Moe m = MakeMoe(down_q6);
  hipStream_t s = nullptr;
  float *x, *y_old, *y_new, *act, *ys_old, *ys_new;
  int32_t* ids_d;
  CK(hipMalloc(&x, tokens * kHidden * 4));
  CK(hipMalloc(&y_old, tokens * kUsed * kHidden * 4));
  CK(hipMalloc(&y_new, tokens * kUsed * kHidden * 4));
  CK(hipMalloc(&act, tokens * kUsed * kFf * 4));
  CK(hipMalloc(&ys_old, tokens * kHidden * 4));
  CK(hipMalloc(&ys_new, tokens * kHidden * 4));
  CK(hipMalloc(&ids_d, tokens * kUsed * 4));
  FillNormal<<<static_cast<unsigned>((tokens * kHidden + 255) / 256), 256>>>(x, tokens * kHidden, 5);
  const auto ids = Route(tokens, 2, 99);
  CK(hipMemcpy(ids_d, ids.data(), ids.size() * 4, hipMemcpyHostToDevice));
  NewBufs b = AllocNew(tokens);

  OldRouted(m, ids_d, x, act, y_old, tokens, s);
  OldShared(m, x, act, ys_old, tokens, s);
  NewRouted(m, ids_d, x, y_new, tokens, 32, b, s);
  NewShared(m, ys_new, tokens, 32, b, s);
  CK(hipDeviceSynchronize());
  const auto h_old = D2H(y_old, tokens * kUsed * kHidden), h_new = D2H(y_new, tokens * kUsed * kHidden);
  const auto hs_old = D2H(ys_old, tokens * kHidden), hs_new = D2H(ys_new, tokens * kHidden);
  unsigned over = 0;
  CK(hipMemcpy(&over, b.overflow, 4, hipMemcpyDeviceToHost));

  // K2: bit equality across BM 64 / 128 and across a chunk cut (the tokens as two chunks).
  std::size_t diff_bm = 0, diff_cut = 0;
  for (int bm : {64, 128}) {
    if (g_exact && bm == 128) continue;
    CK(hipMemset(y_new, 0, tokens * kUsed * kHidden * 4));
    NewRouted(m, ids_d, x, y_new, tokens, bm, b, s);
    CK(hipDeviceSynchronize());
    const auto h2 = D2H(y_new, tokens * kUsed * kHidden);
    for (std::size_t i = 0; i < h2.size(); ++i) diff_bm += std::memcmp(&h2[i], &h_new[i], 4) != 0;
  }
  {
    const std::size_t t1 = tokens / 3 + 5;  // an uneven cut
    CK(hipMemset(y_new, 0, tokens * kUsed * kHidden * 4));
    NewBufs b1 = AllocNew(tokens);
    NewRouted(m, ids_d, x, y_new, t1, 32, b1, s);
    NewRouted(m, ids_d + t1 * kUsed, x + t1 * kHidden, y_new + t1 * kUsed * kHidden, tokens - t1, g_exact ? 64 : 128, b1, s);
    CK(hipDeviceSynchronize());
    FreeNew(b1);
    const auto h2 = D2H(y_new, tokens * kUsed * kHidden);
    for (std::size_t i = 0; i < h2.size(); ++i) diff_cut += std::memcmp(&h2[i], &h_new[i], 4) != 0;
  }

  // float64 reference on every third token (rows are independent), all slots, and the shared expert.
  std::vector<std::size_t> check_tokens;
  for (std::size_t t = 0; t < tokens; t += 3) check_tokens.push_back(t);
  const auto xh = D2H(x, tokens * kHidden);
  std::vector<int> need(kExperts, 0);
  for (std::size_t t : check_tokens)
    for (std::size_t sl = 0; sl < kUsed; ++sl) need[ids[t * kUsed + sl]] = 1;
  std::vector<std::vector<double>> G(kExperts), U(kExperts), Dn(kExperts);
  for (std::size_t e = 0; e < kExperts; ++e)
    if (need[e]) {
      G[e] = DequantExpert(m.gate, e, kFf, kHidden);
      U[e] = DequantExpert(m.up, e, kFf, kHidden);
      Dn[e] = DequantExpert(m.down, e, kHidden, kFf);
    }
  const auto SG = DequantExpert(m.sgate, 0, kSharedFf, kHidden), SU = DequantExpert(m.sup, 0, kSharedFf, kHidden),
             SD = DequantExpert(m.sdown, 0, kHidden, kSharedFf);
  const std::size_t nct = check_tokens.size();
  std::vector<double> ref(nct * kUsed * kHidden), sref(nct * kHidden);
  auto ffn = [&](const double* xr, const std::vector<double>& g, const std::vector<double>& u,
                 const std::vector<double>& dn, std::size_t ff, double* out) {
    std::vector<double> hh(ff);
    for (std::size_t j = 0; j < ff; ++j) {
      double a = 0, c = 0;
      const double* gr = g.data() + j * kHidden;
      const double* ur = u.data() + j * kHidden;
      for (std::size_t i = 0; i < kHidden; ++i) a += gr[i] * xr[i], c += ur[i] * xr[i];
      hh[j] = a / (1.0 + std::exp(-a)) * c;
    }
    for (std::size_t o = 0; o < kHidden; ++o) {
      double v = 0;
      const double* dr = dn.data() + o * ff;
      for (std::size_t j = 0; j < ff; ++j) v += dr[j] * hh[j];
      out[o] = v;
    }
  };
  ParallelFor(nct * (kUsed + 1), [&](std::size_t job) {
    const std::size_t ci = job / (kUsed + 1), sl = job % (kUsed + 1), t = check_tokens[ci];
    std::vector<double> xr(kHidden);
    for (std::size_t i = 0; i < kHidden; ++i) xr[i] = xh[t * kHidden + i];
    if (sl == kUsed) {
      ffn(xr.data(), SG, SU, SD, kSharedFf, sref.data() + ci * kHidden);
    } else {
      const int e = ids[t * kUsed + sl];
      ffn(xr.data(), G[e], U[e], Dn[e], kFf, ref.data() + (ci * kUsed + sl) * kHidden);
    }
  });
  auto pick = [&](const std::vector<float>& all, std::size_t per) {
    std::vector<float> o(nct * per);
    for (std::size_t ci = 0; ci < nct; ++ci)
      std::copy_n(all.begin() + check_tokens[ci] * per, per, o.begin() + ci * per);
    return o;
  };
  const Err eo = Compare(ref, pick(h_old, kUsed * kHidden)), en = Compare(ref, pick(h_new, kUsed * kHidden));
  const Err so = Compare(sref, pick(hs_old, kHidden)), sn = Compare(sref, pick(hs_new, kHidden));
  std::printf("{\"event\":\"check\",\"variant\":\"%s\",\"tokens\":%zu,\"down\":\"%s\",\"checked_tokens\":%zu,%s,%s,%s,%s,"
              "\"differing_floats_across_bm\":%zu,\"differing_floats_across_cut\":%zu,\"overflow\":%u}\n",
              g_exact ? "exact" : "f16", tokens, down_q6 ? "q6k" : "q4k", nct, ErrJson("routed_old", eo).c_str(), ErrJson("routed_new", en).c_str(),
              ErrJson("shared_old", so).c_str(), ErrJson("shared_new", sn).c_str(), diff_bm, diff_cut, over);
  std::fflush(stdout);
  return 0;
}

// Dense projection: Linear (identity group) against float64 on a sample of rows.
int CheckDense(std::size_t rows, std::size_t n, std::size_t k, bool q4) {
  Stack w = MakeStack(q4, 1, n, k, 41);
  float *x, *y;
  uint16_t* xh;
  void* plan_mem;
  CK(hipMalloc(&x, rows * k * 4));
  CK(hipMalloc(&xh, rows * k * 2));
  CK(hipMalloc(&y, rows * n * 4));
  CK(hipMalloc(&plan_mem, kg::PlanBytes(rows, 1, 64)));
  FillNormal<<<static_cast<unsigned>((rows * k + 255) / 256), 256>>>(x, rows * k, 6);
  kg::Plan plan = kg::MakePlan(plan_mem, rows, 1, 64);
  kg::Group(nullptr, plan, nullptr);
  uint16_t* xl;
  CK(hipMalloc(&xl, rows * k * 2));
  if (g_exact) {
    kg::ToF16Split(x, xh, xl, rows * k, nullptr, nullptr);
    kg::LinearExact(q4 ? kg::WType::kQ4K : kg::WType::kQ6K, w.d, xh, xl, plan, y, n, k, nullptr);
  } else {
    kg::ToF16(x, xh, rows * k, nullptr, nullptr);
    kg::Linear(q4 ? kg::WType::kQ4K : kg::WType::kQ6K, w.d, xh, plan, y, n, k, nullptr);
  }
  CK(hipDeviceSynchronize());
  const auto hy = D2H(y, rows * n), hx = D2H(x, rows * k);
  const auto W = DequantExpert(w, 0, n, k);
  std::vector<std::size_t> rs;
  for (std::size_t r = 0; r < rows; r += 7) rs.push_back(r);
  std::vector<double> ref(rs.size() * n);
  std::vector<float> got(rs.size() * n);
  ParallelFor(rs.size(), [&](std::size_t i) {
    const std::size_t r = rs[i];
    for (std::size_t o = 0; o < n; ++o) {
      double v = 0;
      for (std::size_t j = 0; j < k; ++j) v += W[o * k + j] * hx[r * k + j];
      ref[i * n + o] = v;
      got[i * n + o] = hy[r * n + o];
    }
  });
  std::printf("{\"event\":\"check_dense\",\"variant\":\"%s\",\"rows\":%zu,\"n\":%zu,\"k\":%zu,\"type\":\"%s\",%s}\n", g_exact ? "exact" : "f16", rows, n, k,
              q4 ? "q4k" : "q6k", ErrJson("linear_new", Compare(ref, got)).c_str());
  std::fflush(stdout);
  return 0;
}

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int Speed(std::size_t tokens, bool down_q6, int route_mode, int bm_arg, int reps, bool run_old) {
  Moe m = MakeMoe(down_q6);
  hipStream_t s = nullptr;
  float *x, *y, *act, *ys;
  int32_t* ids_d;
  CK(hipMalloc(&x, tokens * kHidden * 4));
  CK(hipMalloc(&y, tokens * kUsed * kHidden * 4));
  CK(hipMalloc(&act, tokens * kUsed * kFf * 4));
  CK(hipMalloc(&ys, tokens * kHidden * 4));
  CK(hipMalloc(&ids_d, tokens * kUsed * 4));
  FillNormal<<<static_cast<unsigned>((tokens * kHidden + 255) / 256), 256>>>(x, tokens * kHidden, 5);
  const auto ids = Route(tokens, route_mode, 1234);
  CK(hipMemcpy(ids_d, ids.data(), ids.size() * 4, hipMemcpyHostToDevice));
  std::vector<int> cnt(kExperts, 0);
  for (int32_t e : ids) ++cnt[e];
  std::size_t active = 0;
  for (int c : cnt) active += c > 0;
  NewBufs b = AllocNew(tokens);
  hipEvent_t ev[5];
  for (auto& e : ev) CK(hipEventCreate(&e));
  const double pairs = static_cast<double>(tokens * kUsed);
  const double flop_gu = pairs * 2.0 * 2.0 * kHidden * kFf, flop_dn = pairs * 2.0 * kHidden * kFf;
  const double flop_sh = static_cast<double>(tokens) * 2.0 * 3.0 * kHidden * kSharedFf;
  const double wb_gu = 2.0 * active * kFf * (kHidden / 256) * 144.0;
  const double wb_dn = static_cast<double>(active) * kHidden * (kFf / 256) * (down_q6 ? 210.0 : 144.0);
  const int bms[3] = {32, 64, 128};
  for (int bm : bms) {
    if (bm_arg != 0 && bm != bm_arg) continue;
    std::vector<double> t_grp, t_gu, t_dn, t_all, t_sh;
    for (int r = 0; r < reps + 2; ++r) {
      NewRouted(m, ids_d, x, y, tokens, bm, b, s, ev);
      CK(hipEventRecord(ev[4], s));
      NewShared(m, ys, tokens, bm, b, s);
      hipEvent_t e5;
      CK(hipEventCreate(&e5));
      CK(hipEventRecord(e5, s));
      CK(hipEventSynchronize(e5));
      float a, bb, c, d;
      CK(hipEventElapsedTime(&a, ev[0], ev[1]));
      CK(hipEventElapsedTime(&bb, ev[1], ev[2]));
      CK(hipEventElapsedTime(&c, ev[2], ev[3]));
      CK(hipEventElapsedTime(&d, ev[4], e5));
      (void)hipEventDestroy(e5);
      if (r < 2) continue;
      t_grp.push_back(a), t_gu.push_back(bb), t_dn.push_back(c), t_all.push_back(a + bb + c), t_sh.push_back(d);
    }
    unsigned over = 0;
    CK(hipMemcpy(&over, b.overflow, 4, hipMemcpyDeviceToHost));
    const double ga = Median(t_grp), gu = Median(t_gu), dn = Median(t_dn), all = Median(t_all), sh = Median(t_sh);
    std::printf("{\"event\":\"speed_new\",\"variant\":\"%s\",\"rows\":%zu,\"down\":\"%s\",\"routing\":%d,\"bm\":%d,\"active_experts\":%zu,"
                "\"ms\":{\"group_tof16\":%.3f,\"gate_up\":%.3f,\"down\":%.3f,\"routed\":%.3f,\"shared\":%.3f},"
                "\"tflops\":{\"gate_up\":%.2f,\"down\":%.2f,\"routed\":%.2f,\"shared\":%.2f},"
                "\"weight_gbps\":{\"gate_up\":%.1f,\"down\":%.1f},\"reps\":%d,\"overflow\":%u}\n",
                g_exact ? "exact" : "f16", tokens, down_q6 ? "q6k" : "q4k", route_mode, bm, active, ga, gu, dn, all, sh, flop_gu / gu / 1e9,
                flop_dn / dn / 1e9, (flop_gu + flop_dn) / all / 1e9, flop_sh / sh / 1e9, wb_gu / gu / 1e6,
                wb_dn / dn / 1e6, reps, over);
    std::fflush(stdout);
  }
  if (run_old) {
    std::vector<double> t_gu, t_dn, t_sh;
    const int oreps = tokens >= 8192 ? 2 : std::max(2, reps / 2);
    for (int r = 0; r < oreps + 1; ++r) {
      CK(hipEventRecord(ev[0], s));
      old::IndexedExperts(true, m.gate.d, m.up.d, ids_d, x, act, tokens, kUsed, kUsed, kFf, kHidden, kExperts, s);
      CK(hipEventRecord(ev[1], s));
      old::IndexedExperts(m.down.q4, m.down.d, nullptr, ids_d, act, y, tokens, kUsed, 1, kHidden, kFf, kExperts, s);
      CK(hipEventRecord(ev[2], s));
      OldShared(m, x, act, ys, tokens, s);
      CK(hipEventRecord(ev[3], s));
      CK(hipEventSynchronize(ev[3]));
      float a, bb, c;
      CK(hipEventElapsedTime(&a, ev[0], ev[1]));
      CK(hipEventElapsedTime(&bb, ev[1], ev[2]));
      CK(hipEventElapsedTime(&c, ev[2], ev[3]));
      if (r < 1) continue;
      t_gu.push_back(a), t_dn.push_back(bb), t_sh.push_back(c);
    }
    const double gu = Median(t_gu), dn = Median(t_dn), sh = Median(t_sh);
    const double req_gu = pairs * 2.0 * kFf * (kHidden / 256) * 144.0;
    const double req_dn = pairs * kHidden * (kFf / 256) * (down_q6 ? 210.0 : 144.0);
    std::printf("{\"event\":\"speed_old\",\"rows\":%zu,\"down\":\"%s\",\"routing\":%d,"
                "\"ms\":{\"gate_up\":%.3f,\"down\":%.3f,\"routed\":%.3f,\"shared\":%.3f},"
                "\"tflops\":{\"gate_up\":%.3f,\"down\":%.3f,\"routed\":%.3f,\"shared\":%.3f},"
                "\"requested_gbps\":{\"gate_up\":%.1f,\"down\":%.1f},\"reps\":%d}\n",
                tokens, down_q6 ? "q6k" : "q4k", route_mode, gu, dn, gu + dn, sh, flop_gu / gu / 1e9, flop_dn / dn / 1e9,
                (flop_gu + flop_dn) / (gu + dn) / 1e9, flop_sh / sh / 1e9, req_gu / gu / 1e6, req_dn / dn / 1e6, oreps);
    std::fflush(stdout);
  }
  return 0;
}

int Dense(std::size_t rows, std::size_t n, std::size_t k, bool q4, int reps) {
  Stack w = MakeStack(q4, 1, n, k, 41);
  float *x, *y;
  uint16_t* xh;
  void* plan_mem;
  CK(hipMalloc(&x, rows * k * 4));
  CK(hipMalloc(&xh, rows * k * 2));
  CK(hipMalloc(&y, rows * n * 4));
  CK(hipMalloc(&plan_mem, kg::PlanBytes(rows, 1, 128)));
  FillNormal<<<static_cast<unsigned>((rows * k + 255) / 256), 256>>>(x, rows * k, 6);
  kg::ToF16(x, xh, rows * k, nullptr, nullptr);
  hipEvent_t e0, e1;
  CK(hipEventCreate(&e0));
  CK(hipEventCreate(&e1));
  for (int bm : {64, 128}) {
    kg::Plan plan = kg::MakePlan(plan_mem, rows, 1, bm);
    kg::Group(nullptr, plan, nullptr);
    std::vector<double> t;
    for (int r = 0; r < reps + 2; ++r) {
      CK(hipEventRecord(e0, nullptr));
      if (g_exact)
        kg::LinearExact(q4 ? kg::WType::kQ4K : kg::WType::kQ6K, w.d, xh, xh, plan, y, n, k, nullptr);
      else
        kg::Linear(q4 ? kg::WType::kQ4K : kg::WType::kQ6K, w.d, xh, plan, y, n, k, nullptr);
      CK(hipEventRecord(e1, nullptr));
      CK(hipEventSynchronize(e1));
      float ms;
      CK(hipEventElapsedTime(&ms, e0, e1));
      if (r >= 2) t.push_back(ms);
    }
    const double ms = Median(t);
    std::printf("{\"event\":\"speed_dense\",\"variant\":\"%s\",\"rows\":%zu,\"n\":%zu,\"k\":%zu,\"type\":\"%s\",\"bm\":%d,\"ms\":%.3f,"
                "\"tflops\":%.2f,\"reps\":%d}\n",
                g_exact ? "exact" : "f16", rows, n, k, q4 ? "q4k" : "q6k", bm, ms, 2.0 * rows * n * k / ms / 1e9, reps);
    std::fflush(stdout);
  }
  return 0;
}

int main(int argc, char** argv) {
  std::string mode;
  std::size_t tokens = 192, rows = 1024, n = 4096, k = 4096;
  bool down_q6 = true, q4 = true, run_old = true;
  int route = 0, bm = 0, reps = 5;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto v = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (a == "--check" || a == "--speed" || a == "--dense" || a == "--check-dense") mode = a;
    else if (a == "--tokens") tokens = std::stoul(v());
    else if (a == "--rows") rows = std::stoul(v());
    else if (a == "--n") n = std::stoul(v());
    else if (a == "--k") k = std::stoul(v());
    else if (a == "--down") down_q6 = v() == "q6k";
    else if (a == "--type") q4 = v() == "q4k";
    else if (a == "--skew") route = 1;
    else if (a == "--bm") bm = std::stoi(v());
    else if (a == "--reps") reps = std::stoi(v());
    else if (a == "--old") run_old = v() != "0";
    else if (a == "--case") (void)v();
    else if (a == "--exact") g_exact = true;
    else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }
  hipDeviceProp_t prop;
  CK(hipGetDeviceProperties(&prop, 0));
  std::fprintf(stderr, "device %s %s, %d CUs\n", prop.name, prop.gcnArchName, prop.multiProcessorCount);
  if (mode == "--check") return Check(tokens, down_q6);
  if (mode == "--check-dense") return CheckDense(rows, n, k, q4);
  if (mode == "--speed") return Speed(rows, down_q6, route, bm, reps, run_old);
  if (mode == "--dense") return Dense(rows, n, k, q4, reps);
  std::fprintf(stderr, "usage: see the header\n");
  return 2;
}
