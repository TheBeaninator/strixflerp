// SPDX-License-Identifier: MIT
// See draft_coarse.hpp.
#if defined(ENGINE_ENABLE_HIP)
#include "flerp/draft_coarse/draft_coarse.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "src/families/qwen/hip_common/quant_ops.hpp"

namespace flerp::draft_coarse {
namespace {

using gufo::core::GgmlType;

constexpr int kGroup = 64;  // values a scale/minimum pair covers

__device__ inline float WaveSum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off);
  return v;
}

// Element i of a GGUF row (llama.cpp dequantize_row_q6_K / q4_K), FP32. Only used to build the low-bit copy and its
// error; the exact logits come from QuantWarpBlockDot.
__device__ inline float Weight(bool q6, const std::uint8_t* row, int i) {
  const int b = i >> 8, e = i & 255;
  if (q6) {
    const auto* blk = row + static_cast<std::size_t>(b) * 210;
    const std::uint8_t* ql = blk;
    const std::uint8_t* qh = blk + 128;
    const auto* sc = reinterpret_cast<const std::int8_t*>(blk + 192);
    const float d = __half2float(*reinterpret_cast<const __half*>(blk + 208));
    const int n = e >> 7, r = e & 127, quarter = r >> 5, l = r & 31, is = l >> 4;
    const int qlb = n * 64, qhb = n * 32, scb = n * 8;
    int q = 0;
    switch (quarter) {
      case 0: q = (ql[qlb + l] & 0xF) | (((qh[qhb + l] >> 0) & 3) << 4); break;
      case 1: q = (ql[qlb + l + 32] & 0xF) | (((qh[qhb + l] >> 2) & 3) << 4); break;
      case 2: q = (ql[qlb + l] >> 4) | (((qh[qhb + l] >> 4) & 3) << 4); break;
      default: q = (ql[qlb + l + 32] >> 4) | (((qh[qhb + l] >> 6) & 3) << 4); break;
    }
    return d * static_cast<float>(sc[scb + is + 2 * quarter]) * static_cast<float>(q - 32);
  }
  const auto* blk = row + static_cast<std::size_t>(b) * 144;
  const float d = __half2float(*reinterpret_cast<const __half*>(blk));
  const float dmin = __half2float(*reinterpret_cast<const __half*>(blk + 2));
  const std::uint8_t* scales = blk + 4;
  const std::uint8_t* qs = blk + 16;
  const int j = e >> 6, w = e & 63, l = w & 31, hi = w >> 5;
  const int q = hi ? (qs[32 * j + l] >> 4) : (qs[32 * j + l] & 0xF);
  const int is = 2 * j + hi;
  int sc, m;
  if (is < 4) {
    sc = scales[is] & 63;
    m = scales[is + 4] & 63;
  } else {
    sc = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
    m = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4);
  }
  return d * static_cast<float>(sc) * static_cast<float>(q) - dmin * static_cast<float>(m);
}

// Bit position of value j (0..63) of a group: q2 word j / 16, byte t % 4, field t / 4 (t = j % 16); q4 word j / 8,
// byte t % 4, nibble t / 4 (t = j % 8). So a mask of 0x03030303 (0x0F0F0F0F) after one shift gives 4 values.
__device__ inline void Place(int bits, int j, unsigned code, std::uint32_t* words) {
  if (bits == 2) {
    const int w = j >> 4, t = j & 15;
    words[w] |= code << (8 * (t & 3) + 2 * (t >> 2));
  } else {
    const int w = j >> 3, t = j & 7;
    words[w] |= code << (8 * (t & 3) + 4 * (t >> 2));
  }
}

__global__ void __launch_bounds__(128) BuildKernel(const std::uint8_t* __restrict__ head, int q6, int vocab, int hidden,
                                                   int bits, std::uint32_t* __restrict__ codes, __half* __restrict__ scale,
                                                   __half* __restrict__ minv, unsigned* __restrict__ err_bits) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (row >= vocab) return;
  const std::size_t rb = static_cast<std::size_t>(hidden / 256) * (q6 ? 210 : 144);
  const std::uint8_t* rp = head + static_cast<std::size_t>(row) * rb;
  const int ng = hidden / kGroup, wpg = 2 * bits;  // words a group
  const float levels = static_cast<float>((1 << bits) - 1);
  float err = 0.0F;
  for (int g = lane; g < ng; g += 32) {
    float w[kGroup];
    float lo = INFINITY, hi = -INFINITY;
#pragma unroll
    for (int j = 0; j < kGroup; ++j) {
      w[j] = Weight(q6 != 0, rp, g * kGroup + j);
      lo = fminf(lo, w[j]);
      hi = fmaxf(hi, w[j]);
    }
    const __half dh = __float2half((hi - lo) / levels);
    const __half mh = __float2half(lo);
    const float df = __half2float(dh), mf = __half2float(mh);
    std::uint32_t words[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int j = 0; j < kGroup; ++j) {
      float c = df > 0.0F ? rintf((w[j] - mf) / df) : 0.0F;
      c = fminf(fmaxf(c, 0.0F), levels);
      Place(bits, j, static_cast<unsigned>(c), words);
      const float r = w[j] - (df * c + mf);
      err += r * r;
    }
    std::uint32_t* dst = codes + (static_cast<std::size_t>(row) * ng + g) * wpg;
    for (int i = 0; i < wpg; ++i) dst[i] = words[i];
    scale[static_cast<std::size_t>(row) * ng + g] = dh;
    minv[static_cast<std::size_t>(row) * ng + g] = mh;
  }
  err = WaveSum(err);
  if (lane == 0) atomicMax(err_bits, __float_as_uint(sqrtf(err / static_cast<float>(hidden))));
}

// Coarse logits: one block of 8 waves per 128 tokens, 16 rows a wave, two rows at a time. x sits in LDS transposed
// (T[(i * 64 + j) * 32 + L] = x[64 (L + 32 i) + j]) so lane L reads its groups' values conflict-free. NGI = hidden /
// 2048 groups a lane.
template <int BITS, int NGI>
__global__ void __launch_bounds__(256) CoarseKernel(const uint4* __restrict__ codes, const __half* __restrict__ scale,
                                                    const __half* __restrict__ minv, const float* __restrict__ x, int vocab,
                                                    float* __restrict__ logits, float* __restrict__ maxima) {
  constexpr int H = NGI * 2048, NG = H / kGroup, U4 = BITS / 2;  // uint4 a group
  __shared__ float t[H];
  __shared__ float wmax[8];
  for (int idx = threadIdx.x; idx < H; idx += 256) {
    const int L = idx & 31, j = (idx >> 5) & 63, i = idx >> 11;
    t[idx] = x[64 * (L + 32 * i) + j];
  }
  __syncthreads();
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  float xs[NGI];
#pragma unroll
  for (int i = 0; i < NGI; ++i) {
    float s = 0.0F;
    for (int j = 0; j < 64; ++j) s += t[(i * 64 + j) * 32 + lane];
    xs[i] = s;
  }
  float best = -INFINITY;
  const int row0 = blockIdx.x * kBlockTokens + wave * 16;
#pragma unroll 1
  for (int r = 0; r < 16; r += 2) {
    const int ra = min(row0 + r, vocab - 1), rb = min(row0 + r + 1, vocab - 1);
    float acc0 = 0.0F, acc1 = 0.0F;
#pragma unroll
    for (int i = 0; i < NGI; ++i) {
      const int g = lane + 32 * i;
      uint4 c0[U4], c1[U4];
#pragma unroll
      for (int u = 0; u < U4; ++u) {
        c0[u] = codes[(static_cast<std::size_t>(ra) * NG + g) * U4 + u];
        c1[u] = codes[(static_cast<std::size_t>(rb) * NG + g) * U4 + u];
      }
      const float d0 = __half2float(scale[static_cast<std::size_t>(ra) * NG + g]);
      const float m0 = __half2float(minv[static_cast<std::size_t>(ra) * NG + g]);
      const float d1 = __half2float(scale[static_cast<std::size_t>(rb) * NG + g]);
      const float m1 = __half2float(minv[static_cast<std::size_t>(rb) * NG + g]);
      float q0 = 0.0F, q1 = 0.0F;
#pragma unroll
      for (int wi = 0; wi < 4 * U4; ++wi) {
        const std::uint32_t w0 = (&c0[wi >> 2].x)[wi & 3];
        const std::uint32_t w1 = (&c1[wi >> 2].x)[wi & 3];
        constexpr int kFields = BITS == 2 ? 4 : 2, kPer = BITS == 2 ? 16 : 8;
        constexpr std::uint32_t kMask = BITS == 2 ? 0x03030303U : 0x0F0F0F0FU;
#pragma unroll
        for (int f = 0; f < kFields; ++f) {
          const std::uint32_t v0 = (w0 >> (BITS * f)) & kMask, v1 = (w1 >> (BITS * f)) & kMask;
#pragma unroll
          for (int b = 0; b < 4; ++b) {
            const int j = wi * kPer + 4 * f + b;
            const float xv = t[(i * 64 + j) * 32 + lane];
            q0 += static_cast<float>((v0 >> (8 * b)) & 0xFFU) * xv;
            q1 += static_cast<float>((v1 >> (8 * b)) & 0xFFU) * xv;
          }
        }
      }
      acc0 += d0 * q0 + m0 * xs[i];
      acc1 += d1 * q1 + m1 * xs[i];
    }
    const float v0 = WaveSum(acc0), v1 = WaveSum(acc1);
    if (row0 + r < vocab) {
      if (lane == 0) logits[row0 + r] = v0;
      best = fmaxf(best, isnan(v0) ? INFINITY : v0);
    }
    if (row0 + r + 1 < vocab) {
      if (lane == 0) logits[row0 + r + 1] = v1;
      best = fmaxf(best, isnan(v1) ? INFINITY : v1);
    }
  }
  if (lane == 0) wmax[wave] = best;
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = wmax[0];
    for (int w = 1; w < 8; ++w) m = fmaxf(m, wmax[w]);
    maxima[blockIdx.x] = m;
  }
}

struct Cand {
  float v;
  int i;
};
__device__ inline bool Better(const Cand& a, const Cand& b) { return a.v > b.v || (a.v == b.v && a.i < b.i); }

// keep + 1 rounds of a block-wide best over the block maxima: thread t holds blocks t, t + 256, ... (at most 8), a
// round takes the best untaken one of every thread (wave shuffles, then the 8 wave results), the owner marks it taken.
__global__ void __launch_bounds__(256) SelectKernel(const float* __restrict__ maxima, int nb, int keep,
                                                    int* __restrict__ chosen, float* __restrict__ outside) {
  __shared__ Cand wbest[2][8];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  float mv[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int b = tid + 256 * k;
    const float v = b < nb ? maxima[b] : -INFINITY;
    mv[k] = isnan(v) ? INFINITY : v;
  }
  unsigned taken = 0;
  for (int round = 0; round <= keep; ++round) {
    Cand c{-INFINITY, INT_MAX};
    int ck = -1;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const Cand o{mv[k], tid + 256 * k};
      if (o.i < nb && ((taken >> k) & 1U) == 0 && Better(o, c)) {
        c = o;
        ck = k;
      }
    }
    for (int off = 16; off > 0; off >>= 1) {
      Cand o{__shfl_xor(c.v, off), __shfl_xor(c.i, off)};
      if (Better(o, c)) c = o;
    }
    if (lane == 0) wbest[round & 1][wave] = c;
    __syncthreads();
    Cand w = wbest[round & 1][0];
    for (int k = 1; k < 8; ++k) {
      if (Better(wbest[round & 1][k], w)) w = wbest[round & 1][k];
    }
    if (w.i != INT_MAX && ck >= 0 && w.i == tid + 256 * ck) taken |= 1U << ck;
    if (tid == 0) {
      if (round < keep) chosen[round] = w.i == INT_MAX ? -1 : w.i;
      else *outside = w.i == INT_MAX ? -INFINITY : w.v;
    }
  }
}

// The kept blocks' rows, exact: gufo's QuantWarpBlockDot (lane 0 holds the row dot), as Q8KBlockGEMVKernel_2Rows.
__global__ void __launch_bounds__(128) RescoreKernel(const void* __restrict__ head, GgmlType type, const int* __restrict__ chosen,
                                                     const float* __restrict__ x, int hidden, int vocab,
                                                     float* __restrict__ logits, float* __restrict__ cand) {
  const std::size_t lane = threadIdx.x & 31U;
  const int ri = blockIdx.x * 4 + static_cast<int>(threadIdx.x >> 5);
  const int slot = ri / kBlockTokens, r = ri % kBlockTokens;
  const int blk = chosen[slot];
  const int row = blk * kBlockTokens + r;
  if (blk < 0 || row >= vocab) {
    if (lane == 0) cand[ri] = -INFINITY;
    return;
  }
  const float v = gufo::hip::QuantWarpBlockDot(type, head, static_cast<std::size_t>(row), x,
                                               static_cast<std::size_t>(hidden), lane, 0, 1);
  if (lane == 0) {
    logits[row] = v;
    cand[ri] = v;
  }
}

// Best and second exact logit of the kept rows (finite only; lower id on a tie, ArgmaxKernel's rule), ||x||, tau and
// the fallback flag.
__global__ void __launch_bounds__(256) ChooseKernel(const float* __restrict__ cand, const int* __restrict__ chosen, int keep,
                                                    const float* __restrict__ outside, const float* __restrict__ x, int hidden,
                                                    float k_err, Result* __restrict__ out) {
  __shared__ float sb[256], ss[256], sx[256];
  __shared__ unsigned si[256];
  const int tid = threadIdx.x;
  float b = -INFINITY, s = -INFINITY, xx = 0.0F;
  unsigned id = 0xFFFFFFFFU;
  for (int i = tid; i < keep * kBlockTokens; i += 256) {
    const float v = cand[i];
    if (!isfinite(v)) continue;
    const unsigned tok = static_cast<unsigned>(chosen[i / kBlockTokens] * kBlockTokens + i % kBlockTokens);
    if (v > b || (v == b && tok < id)) {
      s = b;
      b = v;
      id = tok;
    } else {
      s = fmaxf(s, v);
    }
  }
  for (int i = tid; i < hidden; i += 256) xx += x[i] * x[i];
  sb[tid] = b;
  ss[tid] = s;
  si[tid] = id;
  sx[tid] = xx;
  __syncthreads();
  for (int st = 128; st > 0; st >>= 1) {
    if (tid < st) {
      const float ob = sb[tid + st], os = ss[tid + st];
      const unsigned oi = si[tid + st];
      if (ob > sb[tid] || (ob == sb[tid] && oi < si[tid])) {
        ss[tid] = fmaxf(sb[tid], os);
        sb[tid] = ob;
        si[tid] = oi;
      } else {
        ss[tid] = fmaxf(ss[tid], ob);
      }
      sx[tid] += sx[tid + st];
    }
    __syncthreads();
  }
  if (tid == 0) {
    Result r;
    r.id = si[0];
    r.best = sb[0];
    r.second = ss[0];
    r.outside = *outside;
    r.xnorm = sqrtf(sx[0]);
    r.tau = k_err * r.xnorm;
    const bool ok = si[0] != 0xFFFFFFFFU && isfinite(sb[0]) && !isnan(r.outside) && isfinite(r.tau) &&
                    (r.outside == -INFINITY || (isfinite(ss[0]) && ss[0] - r.outside >= r.tau));
    r.fallback = ok ? 0U : 1U;
    r.pad = 0;
    *out = r;
  }
}

template <int BITS>
void LaunchCoarse(int ngi, const uint4* codes, const __half* sc, const __half* mn, const float* x, int vocab, float* logits,
                  float* maxima, int blocks, hipStream_t s) {
  switch (ngi) {
    case 1: CoarseKernel<BITS, 1><<<blocks, 256, 0, s>>>(codes, sc, mn, x, vocab, logits, maxima); break;
    case 2: CoarseKernel<BITS, 2><<<blocks, 256, 0, s>>>(codes, sc, mn, x, vocab, logits, maxima); break;
    case 3: CoarseKernel<BITS, 3><<<blocks, 256, 0, s>>>(codes, sc, mn, x, vocab, logits, maxima); break;
    default: CoarseKernel<BITS, 4><<<blocks, 256, 0, s>>>(codes, sc, mn, x, vocab, logits, maxima); break;
  }
}

}  // namespace

Coarse::~Coarse() {
  for (void* p : {static_cast<void*>(codes_), static_cast<void*>(scale_), static_cast<void*>(minv_), static_cast<void*>(maxima_),
                  static_cast<void*>(chosen_), static_cast<void*>(outside_), static_cast<void*>(cand_)}) {
    if (p != nullptr) (void)hipFree(p);
  }
}

std::unique_ptr<Coarse> Coarse::Build(const void* head, GgmlType type, int vocab, int hidden, int bits, int keep,
                                      float k_tau, hipStream_t stream, std::string* error) {
  const auto fail = [&](const std::string& why) -> std::unique_ptr<Coarse> {
    if (error != nullptr) *error = "draft_coarse: " + why;
    return nullptr;
  };
  if (type != GgmlType::kQ6_K && type != GgmlType::kQ4_K) return fail("head type is not Q6_K or Q4_K");
  if (hidden % 2048 != 0 || hidden > 8192) return fail("hidden must be a multiple of 2048, at most 8192");
  if (bits != 2 && bits != 4) return fail("bits must be 2 or 4");
  if (keep < 1 || keep > kMaxKeep) return fail("keep must be 1..32");
  if (vocab < 1 || (vocab + kBlockTokens - 1) / kBlockTokens > 2048) return fail("vocab out of range");
  if ((reinterpret_cast<std::uintptr_t>(head) & 1U) != 0) return fail("head not 2-byte aligned");
  std::unique_ptr<Coarse> c(new Coarse());
  c->head_ = head;
  c->type_ = type;
  c->vocab_ = vocab;
  c->hidden_ = hidden;
  c->bits_ = bits;
  c->keep_ = keep;
  c->k_tau_ = k_tau;
  c->blocks_ = (vocab + kBlockTokens - 1) / kBlockTokens;
  const std::size_t ng = static_cast<std::size_t>(hidden / kGroup);
  const std::size_t code_bytes = static_cast<std::size_t>(vocab) * ng * 2 * bits * 4;
  const std::size_t par_bytes = static_cast<std::size_t>(vocab) * ng * 2;
  unsigned* err_bits = nullptr;
  if (hipMalloc(&c->codes_, code_bytes) != hipSuccess || hipMalloc(&c->scale_, par_bytes) != hipSuccess ||
      hipMalloc(&c->minv_, par_bytes) != hipSuccess || hipMalloc(&c->maxima_, c->blocks_ * sizeof(float)) != hipSuccess ||
      hipMalloc(&c->chosen_, kMaxKeep * sizeof(int)) != hipSuccess || hipMalloc(&c->outside_, sizeof(float)) != hipSuccess ||
      hipMalloc(&c->cand_, kMaxKeep * kBlockTokens * sizeof(float)) != hipSuccess ||
      hipMalloc(&err_bits, sizeof(unsigned)) != hipSuccess) {
    if (err_bits != nullptr) (void)hipFree(err_bits);
    return fail("device allocation");
  }
  c->bytes_ = code_bytes + 2 * par_bytes;
  (void)hipMemsetAsync(err_bits, 0, sizeof(unsigned), stream);
  BuildKernel<<<(vocab + 3) / 4, 128, 0, stream>>>(static_cast<const std::uint8_t*>(head), type == GgmlType::kQ6_K ? 1 : 0,
                                                   vocab, hidden, bits, c->codes_, c->scale_, c->minv_, err_bits);
  unsigned eb = 0;
  const hipError_t st = hipGetLastError();
  if (st != hipSuccess || hipMemcpyAsync(&eb, err_bits, sizeof(unsigned), hipMemcpyDeviceToHost, stream) != hipSuccess ||
      hipStreamSynchronize(stream) != hipSuccess) {
    (void)hipFree(err_bits);
    return fail(std::string("build: ") + hipGetErrorString(st));
  }
  (void)hipFree(err_bits);
  std::memcpy(&c->err_rms_, &eb, sizeof(float));
  return c;
}

void Coarse::CoarsePass(const float* x, float* logits, hipStream_t s) const {
  const auto* w = reinterpret_cast<const uint4*>(codes_);
  if (bits_ == 2) LaunchCoarse<2>(hidden_ / 2048, w, scale_, minv_, x, vocab_, logits, maxima_, blocks_, s);
  else LaunchCoarse<4>(hidden_ / 2048, w, scale_, minv_, x, vocab_, logits, maxima_, blocks_, s);
}

void Coarse::SelectPass(hipStream_t s) const { SelectKernel<<<1, 256, 0, s>>>(maxima_, blocks_, keep_, chosen_, outside_); }

void Coarse::RescorePass(const float* x, float* logits, hipStream_t s) const {
  RescoreKernel<<<keep_ * kBlockTokens / 4, 128, 0, s>>>(head_, type_, chosen_, x, hidden_, vocab_, logits, cand_);
}

void Coarse::ChoosePass(const float* x, Result* result, hipStream_t s) const {
  ChooseKernel<<<1, 256, 0, s>>>(cand_, chosen_, keep_, outside_, x, hidden_, k_tau_ * err_rms_, result);
}

void Coarse::Score(const float* x, float* logits, Result* result, hipStream_t s) const {
  CoarsePass(x, logits, s);
  SelectPass(s);
  RescorePass(x, logits, s);
  ChoosePass(x, result, s);
}

}  // namespace flerp::draft_coarse
#endif  // defined(ENGINE_ENABLE_HIP)
