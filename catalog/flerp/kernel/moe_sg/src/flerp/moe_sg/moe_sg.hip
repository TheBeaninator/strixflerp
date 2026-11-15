// SPDX-License-Identifier: MIT
// See moe_sg.hpp. Wave32, 8 waves a block, persistent grid.
#if defined(ENGINE_ENABLE_HIP)
#include "flerp/moe_sg/moe_sg.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "src/families/qwen/hip_common/quant_ops.hpp"

namespace flerp::moe_sg {
namespace {

constexpr int kWaves = 8;
constexpr int kThreads = kWaves * 32;
constexpr int kUnitRows = 16;  // output rows of a unit (whole K)

__device__ __forceinline__ float WaveSum(float v) {
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor(v, off);
  }
  return v;
}

__device__ __forceinline__ void LoadX16(const float* __restrict__ xb, float xv[16]) {
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

__device__ __forceinline__ uint4 Load16Align2(const std::uint8_t* p) {
  uint4 v;
  __builtin_memcpy(&v, __builtin_assume_aligned(p, 2), 16);
  return v;
}

// Sub-block sb of a row: moe_kquant.hip RowDotWide's decode, unchanged (the same lane mapping and expressions).
template <WType T>
__device__ __forceinline__ void Decode(const std::uint8_t* __restrict__ row, std::size_t sb, gufo::hip::QuantSub16& sub) {
  const std::size_t sb32 = (sb / 2) % 8;
  if constexpr (T == WType::kQ4K) {
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
}

template <WType T>
constexpr std::size_t RowBytes(std::size_t k) {
  return (k / 256) * (T == WType::kQ4K ? sizeof(gufo::hip::Q4KBlock) : sizeof(gufo::hip::Q6KBlock));
}

// N rows . x side by side; each row's sum is RowDotWide's (sub-blocks lane, lane + 32, ... in order, then WaveSum).
// NI = k / 512 sub-block rounds a lane, UR of them unrolled (more bytes in flight against more registers).
template <WType T, int NI, int N, int UR>
__device__ __forceinline__ void Dots(const std::uint8_t* const (&rows)[N], const float* __restrict__ x, std::size_t lane,
                            float (&out)[N]) {
  float sumf[N];
#pragma unroll
  for (int g = 0; g < N; ++g) sumf[g] = 0.0F;
#pragma unroll UR
  for (int i = 0; i < NI; ++i) {
    const std::size_t sb = lane + (32 * static_cast<std::size_t>(i));
    gufo::hip::QuantSub16 sub[N];
#pragma unroll
    for (int g = 0; g < N; ++g) Decode<T>(rows[g], sb, sub[g]);
    float xv[16];
    LoadX16(x + (sb * 16), xv);
#pragma unroll
    for (int g = 0; g < N; ++g) {
      float dot = 0.0F;
      float xsum = 0.0F;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        dot += static_cast<float>(sub[g].q[j]) * xv[j];
        xsum += xv[j];
      }
      sumf[g] += (sub[g].scale * dot) - (sub[g].offset * xsum);
    }
  }
#pragma unroll
  for (int g = 0; g < N; ++g) out[g] = WaveSum(sumf[g]);
}

// A gate/up unit: rows m0 .. m0 + 15 of gate and up, y[m] = silu(gate . x) * (up . x) (IndexedExpertsKernel's).
template <WType T, int NI, int G, int UR>
__device__ __forceinline__ void GateUpUnit(const std::uint8_t* g, const std::uint8_t* u, const float* x, float* y, int m0,
                                  std::size_t lane) {
  constexpr std::size_t rb = RowBytes<T>(static_cast<std::size_t>(NI) * 512);
#pragma unroll 1
  for (int r = 0; r < kUnitRows; r += G) {
    const std::uint8_t* rows[2 * G];
#pragma unroll
    for (int i = 0; i < G; ++i) {
      rows[i] = g + static_cast<std::size_t>(m0 + r + i) * rb;
      rows[G + i] = u + static_cast<std::size_t>(m0 + r + i) * rb;
    }
    float v[2 * G];
    Dots<T, NI, 2 * G, UR>(rows, x, lane, v);
#pragma unroll
    for (int i = 0; i < G; ++i) {
      const float a = v[i];
      const float b = v[G + i];
      const float out = (a / (1.0F + expf(-a))) * b;
      if (lane == 0) y[m0 + r + i] = out;
    }
  }
}

// A down unit: rows m0 .. m0 + 15 of down, y[m] = down . x.
template <WType T, int NI, int G, int UR>
__device__ __forceinline__ void DownUnit(const std::uint8_t* d, const float* x, float* y, int m0, std::size_t lane) {
  constexpr std::size_t rb = RowBytes<T>(static_cast<std::size_t>(NI) * 512);
#pragma unroll 1
  for (int r = 0; r < kUnitRows; r += G) {
    const std::uint8_t* rows[G];
#pragma unroll
    for (int i = 0; i < G; ++i) rows[i] = d + static_cast<std::size_t>(m0 + r + i) * rb;
    float v[G];
    Dots<T, NI, G, UR>(rows, x, lane, v);
#pragma unroll
    for (int i = 0; i < G; ++i) {
      if (lane == 0) y[m0 + r + i] = v[i];
    }
  }
}

struct Args {
  const std::uint8_t* gate;
  const std::uint8_t* up;
  const std::uint8_t* down;
  const std::uint8_t* sgate;
  const std::uint8_t* sup;
  const std::uint8_t* sdown;
  WType tg, td, tsg, tsd;
  std::size_t gu_ebytes, dn_ebytes;  // one routed expert of gate (= up) and of down
  int hidden, ff, used, experts, shared;
  const std::int32_t* ids;
  const float* x;
  float* act;
  float* sact;
  float* eout;
  float* sout;
  unsigned* ctl;  // [0] claims, [1] exits, [2 + job] gate/up units done
};

__device__ __forceinline__ unsigned Claim(unsigned* ctl, int lane) {
  unsigned c = 0;
  if (lane == 0) c = atomicAdd(ctl, 1U);
  return __builtin_amdgcn_readfirstlane(__shfl(c, 0));
}

// NIH = hidden / 512, NIF = ff / 512; GG, GD rows side by side in gate/up and down units; UR sub-block rounds
// unrolled.
template <int NIH, int NIF, int GG, int GD, int UR>
__global__ void __launch_bounds__(kThreads) SgKernel(Args a) {
  extern __shared__ float4 sg_xl4[];  // the token's x, hidden floats
  const float4* xg4 = reinterpret_cast<const float4*>(a.x);
  for (int i = static_cast<int>(threadIdx.x); i < a.hidden / 4; i += kThreads) sg_xl4[i] = xg4[i];
  __syncthreads();
  const float* xl = reinterpret_cast<const float*>(sg_xl4);
  const int lane = static_cast<int>(threadIdx.x & 31U);
  const std::size_t ulane = static_cast<std::size_t>(lane);
  const int gu_per = a.ff / kUnitRows;
  const int dn_per = a.hidden / kUnitRows;
  const int jobs = a.used + a.shared;
  const unsigned n_gu = static_cast<unsigned>(jobs * gu_per);
  const unsigned total = n_gu + static_cast<unsigned>(jobs * dn_per);
  unsigned* done = a.ctl + 2;
  unsigned c = Claim(a.ctl, lane);
  while (c < total) {
    const unsigned cn = Claim(a.ctl, lane);  // the next claim first, so it returns before this unit's loads
    const bool is_gu = c < n_gu;
    const unsigned cc = is_gu ? c : c - n_gu;
    const int per = is_gu ? gu_per : dn_per;
    const int job = static_cast<int>(cc / static_cast<unsigned>(per));
    const int k = static_cast<int>(cc - static_cast<unsigned>(job * per));
    const bool sh = job < a.shared;
    const int slot = job - a.shared;
    std::size_t e = 0;
    if (!sh) {
      const std::int32_t id = a.ids[slot];
      e = (id < 0 || id >= a.experts) ? static_cast<std::size_t>(a.experts - 1) : static_cast<std::size_t>(id);
    }
    const int m0 = k * kUnitRows;
    if (is_gu) {
      const WType t = sh ? a.tsg : a.tg;
      const std::uint8_t* g = sh ? a.sgate : a.gate + e * a.gu_ebytes;
      const std::uint8_t* u = sh ? a.sup : a.up + e * a.gu_ebytes;
      float* y = sh ? a.sact : a.act + static_cast<std::size_t>(slot) * a.ff;
      if (t == WType::kQ4K) GateUpUnit<WType::kQ4K, NIH, GG, UR>(g, u, xl, y, m0, ulane);
      else GateUpUnit<WType::kQ6K, NIH, GG, UR>(g, u, xl, y, m0, ulane);
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
      if (lane == 0) __hip_atomic_fetch_add(done + job, 1U, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } else {
      if (lane == 0) {
        while (__hip_atomic_load(done + job, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < static_cast<unsigned>(gu_per))
          __builtin_amdgcn_s_sleep(1);
      }
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
      const WType t = sh ? a.tsd : a.td;
      const std::uint8_t* d = sh ? a.sdown : a.down + e * a.dn_ebytes;
      const float* in = sh ? a.sact : a.act + static_cast<std::size_t>(slot) * a.ff;
      float* y = sh ? a.sout : a.eout + static_cast<std::size_t>(slot) * a.hidden;
      if (t == WType::kQ4K) DownUnit<WType::kQ4K, NIF, GD, UR>(d, in, y, m0, ulane);
      else DownUnit<WType::kQ6K, NIF, GD, UR>(d, in, y, m0, ulane);
    }
    c = cn;
  }
  unsigned last = 0;
  if (lane == 0) last = atomicAdd(a.ctl + 1, 1U) == gridDim.x * kWaves - 1 ? 1U : 0U;
  if (__builtin_amdgcn_readfirstlane(__shfl(last, 0)) == 1U) {
    // Last wave out: every claim and counter of this launch is done; zero them for the next launch.
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    for (int i = lane; i < 2 + jobs; i += 32) a.ctl[i] = 0U;
  }
}

using KernelFn = void (*)(Args);

template <int NIH, int NIF>
KernelFn Pick(int variant) {
  switch (variant) {
    case 1: return SgKernel<NIH, NIF, 1, 4, 2>;
    case 2: return SgKernel<NIH, NIF, 2, 8, 1>;
    case 3: return SgKernel<NIH, NIF, 1, 8, 1>;
    default: return SgKernel<NIH, NIF, 1, 4, 1>;
  }
}

// The shapes compiled: Ornith-1.5 397B (hidden 4096, ff 1024) and the 35B proxy (2048, 512).
KernelFn Kernel(const Layer& l, int variant) {
  const int nih = l.hidden / 512, nif = l.ff / 512;
  if (nih == 8 && nif == 2) return Pick<8, 2>(variant);
  if (nih == 4 && nif == 1) return Pick<4, 1>(variant);
  return nullptr;
}

bool Aligned16(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15U) == 0; }

std::size_t Bytes(WType t, std::size_t rows, std::size_t k) {
  return rows * (k / 256) * (t == WType::kQ4K ? 144 : 210);
}

}  // namespace

bool Supported(const Layer& l, const void* x) noexcept {
  if (l.hidden % 512 != 0 || l.ff % 512 != 0 || l.hidden > 8192 || l.used < 1 || l.used > 64 || l.experts < 1) return false;
  if (Kernel(l, 0) == nullptr) return false;
  if (l.gate.data == nullptr || l.up.data == nullptr || l.down.data == nullptr || l.gate.type != l.up.type) return false;
  if (!Aligned16(l.gate.data) || !Aligned16(l.up.data) || !Aligned16(l.down.data) || !Aligned16(x)) return false;
  if (l.sgate.data != nullptr) {
    if (l.shared_ff != l.ff || l.sup.data == nullptr || l.sdown.data == nullptr || l.sgate.type != l.sup.type) return false;
    if (!Aligned16(l.sgate.data) || !Aligned16(l.sup.data) || !Aligned16(l.sdown.data)) return false;
  }
  return true;
}

std::size_t CtlBytes(const Layer& l) noexcept { return static_cast<std::size_t>(2 + l.used + 1) * sizeof(unsigned); }

int Grid(const Layer& l, int variant) {
  const KernelFn k = Kernel(l, variant);
  if (k == nullptr) return 0;
  struct Entry {
    KernelFn k;
    int hidden;
    int grid;
  };
  static Entry cache[16] = {};
  for (auto& c : cache) {
    if (c.k == k && c.hidden == l.hidden) return c.grid;
  }
  int dev = 0, per = 0;
  (void)hipGetDevice(&dev);
  hipDeviceProp_t p{};
  (void)hipGetDeviceProperties(&p, dev);
  const std::size_t lds = static_cast<std::size_t>(l.hidden) * sizeof(float);
  if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&per, reinterpret_cast<const void*>(k), kThreads, lds) != hipSuccess ||
      per < 1) {
    std::cerr << "flerp::moe_sg: no resident block for this kernel instance\n";
    std::abort();
  }
  const int grid = per * p.multiProcessorCount;
  for (auto& c : cache) {
    if (c.k == nullptr) {
      c = {k, l.hidden, grid};
      break;
    }
  }
  return grid;
}

void Experts(const Layer& l, const std::int32_t* ids, const float* x, float* act, float* shared_act, float* expert_out,
             float* shared_out, unsigned* ctl, hipStream_t stream, int variant) {
  const KernelFn k = Kernel(l, variant);
  if (k == nullptr || !Supported(l, x)) {
    std::cerr << "flerp::moe_sg::Experts: unsupported shape (hidden " << l.hidden << ", ff " << l.ff << ")\n";
    std::abort();
  }
  Args a{};
  a.gate = static_cast<const std::uint8_t*>(l.gate.data);
  a.up = static_cast<const std::uint8_t*>(l.up.data);
  a.down = static_cast<const std::uint8_t*>(l.down.data);
  a.shared = l.sgate.data != nullptr ? 1 : 0;
  a.sgate = static_cast<const std::uint8_t*>(l.sgate.data);
  a.sup = static_cast<const std::uint8_t*>(l.sup.data);
  a.sdown = static_cast<const std::uint8_t*>(l.sdown.data);
  a.tg = l.gate.type;
  a.td = l.down.type;
  a.tsg = a.shared ? l.sgate.type : WType::kQ4K;
  a.tsd = a.shared ? l.sdown.type : WType::kQ4K;
  a.gu_ebytes = Bytes(l.gate.type, static_cast<std::size_t>(l.ff), static_cast<std::size_t>(l.hidden));
  a.dn_ebytes = Bytes(l.down.type, static_cast<std::size_t>(l.hidden), static_cast<std::size_t>(l.ff));
  a.hidden = l.hidden;
  a.ff = l.ff;
  a.used = l.used;
  a.experts = l.experts;
  a.ids = ids;
  a.x = x;
  a.act = act;
  a.sact = shared_act;
  a.eout = expert_out;
  a.sout = shared_out;
  a.ctl = ctl;
  const std::size_t lds = static_cast<std::size_t>(l.hidden) * sizeof(float);
  hipLaunchKernelGGL(k, dim3(static_cast<unsigned>(Grid(l, variant))), dim3(kThreads), lds, stream, a);
  const hipError_t st = hipGetLastError();
  if (st != hipSuccess) {
    std::cerr << "flerp::moe_sg::Experts: " << hipGetErrorString(st) << "\n";
    std::abort();
  }
}

}  // namespace flerp::moe_sg
#endif  // defined(ENGINE_ENABLE_HIP)
