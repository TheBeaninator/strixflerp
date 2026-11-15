// SPDX-License-Identifier: MIT
// SHAPE-TRANSFERS R1 first experiment (criteria gate/xfer-r1-splitk.toml): GGUF Q4_K expert decode GEMV,
// kernel.flerp.moe_kquant (reference, its source #included and compiled here with its build flags) against
//   arm a: the reference's lane mapping and summation order, 16-byte vector loads of codes and x;
//   arm b: strixite linear_q4's shape - a lane owns a 32-byte code run (both nibbles, two 32-weight groups),
//          s*sum(x*q) - m*sum(x) per group, KS waves split K, LDS combine across the waves.
//   xfer-r1-splitk --case NAME --rows N --k K --experts E --slots S --tokens T --xshare 1|S [--pair]
//                  [--gguf FILE --off-a BYTES [--off-b BYTES]] [--iters 200] [--warm 20]
// One JSON line a case on stdout.
#include "flerp/moe_kquant/moe_kquant.hip"

#include <fcntl.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace xr1 {
using gufo::core::GgmlType;
constexpr int kBlk = 144;

#define HC(x)                                                                         \
  do {                                                                                \
    hipError_t e_ = (x);                                                              \
    if (e_ != hipSuccess) {                                                           \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
      std::exit(1);                                                                   \
    }                                                                                 \
  } while (0)

__device__ inline float Wsum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off);
  return v;
}

// arm a: QuantWarpBlockDot's Q4_K route (sub16 per lane, every 32nd), same arithmetic, vector loads.
__device__ inline float DotA(const std::uint8_t* __restrict__ row, const float* __restrict__ x, std::size_t k,
                             std::size_t lane) {
  float sumf = 0.0F;
  const std::size_t nsub = k / 16;
  for (std::size_t sb = lane; sb < nsub; sb += 32) {
    const std::uint8_t* blk = row + (sb / 16) * kBlk;
    const std::size_t sb32 = (sb / 2) % 8;
    const std::size_t base = ((sb32 / 2) * 32) + ((sb % 2) * 16);
    const unsigned shift = 4U * static_cast<unsigned>(sb32 & 1U);
    const uint4 p = *reinterpret_cast<const uint4*>(blk + 16 + base);
    gufo::hip::QuantSub16 sub;
    sub.w[0] = (p.x >> shift) & 0x0F0F0F0FU;
    sub.w[1] = (p.y >> shift) & 0x0F0F0F0FU;
    sub.w[2] = (p.z >> shift) & 0x0F0F0F0FU;
    sub.w[3] = (p.w >> shift) & 0x0F0F0F0FU;
    std::uint8_t sc = 0, m = 0;
    gufo::hip::GetQKScaleMin(sb32, blk + 4, sc, m);
    const __half* dh = reinterpret_cast<const __half*>(blk);
    sub.scale = __half2float(dh[0]) * static_cast<float>(sc);
    sub.offset = __half2float(dh[1]) * static_cast<float>(m);
    const float4* x4 = reinterpret_cast<const float4*>(x + sb * 16);
    float xv[16];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float4 v = x4[i];
      xv[4 * i] = v.x, xv[4 * i + 1] = v.y, xv[4 * i + 2] = v.z, xv[4 * i + 3] = v.w;
    }
    float dot = 0.0F, xsum = 0.0F;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      dot += static_cast<float>(sub.q[j]) * xv[j];
      xsum += xv[j];
    }
    sumf += (sub.scale * dot) - (sub.offset * xsum);
  }
  return Wsum(sumf);
}

// arm b: strixite shape. Chunk c = 64 weights: block c/4, code bytes [32*(c%4), +32): low nibbles are group 2*(c%4)
// (x[64j .. 64j+31]), high nibbles group 2*(c%4)+1 (x[64j+32 .. 64j+63]).
__device__ inline float DotB(const std::uint8_t* __restrict__ row, const float* __restrict__ x, std::size_t k, int lane,
                             int ks, int kss) {
  float acc = 0.0F;
  const std::size_t nch = k / 64;
  for (std::size_t c = static_cast<std::size_t>(ks) * 32 + lane; c < nch; c += static_cast<std::size_t>(kss) * 32) {
    const std::uint8_t* blk = row + (c / 4) * kBlk;
    const std::size_t j = c % 4;
    const uint4* q4 = reinterpret_cast<const uint4*>(blk + 16 + 32 * j);
    const uint4 p0 = q4[0], p1 = q4[1];
    const std::uint32_t w[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
    std::uint8_t sc0, m0, sc1, m1;
    gufo::hip::GetQKScaleMin(2 * j, blk + 4, sc0, m0);
    gufo::hip::GetQKScaleMin(2 * j + 1, blk + 4, sc1, m1);
    const __half* dh = reinterpret_cast<const __half*>(blk);
    const float d = __half2float(dh[0]), dm = __half2float(dh[1]);
    const float4* x4 = reinterpret_cast<const float4*>(x + (c / 4) * 256 + 64 * j);
    float dot0 = 0.F, xs0 = 0.F, dot1 = 0.F, xs1 = 0.F;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float4 lo = x4[i], hi = x4[8 + i];
      const float l[4] = {lo.x, lo.y, lo.z, lo.w}, h[4] = {hi.x, hi.y, hi.z, hi.w};
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const std::uint32_t byte = (w[i] >> (8 * b)) & 0xFFU;
        dot0 = fmaf(l[b], static_cast<float>(byte & 0xFU), dot0);
        dot1 = fmaf(h[b], static_cast<float>(byte >> 4), dot1);
        xs0 += l[b];
        xs1 += h[b];
      }
    }
    acc = fmaf(d * static_cast<float>(sc0), dot0, fmaf(-dm * static_cast<float>(m0), xs0, acc));
    acc = fmaf(d * static_cast<float>(sc1), dot1, fmaf(-dm * static_cast<float>(m1), xs1, acc));
  }
  return Wsum(acc);
}

struct Args {
  const std::uint8_t* wa;
  const std::uint8_t* wb;
  std::size_t ebytes, row_bytes;
  const std::int32_t* ids;
  const float* x;
  float* y;
  std::size_t x_share, rows, k, n_experts;
};

__device__ inline std::size_t Expert(const Args& a, std::size_t ts) {
  const std::int32_t id = a.ids[ts];
  return (id < 0 || static_cast<std::size_t>(id) >= a.n_experts) ? a.n_experts - 1 : static_cast<std::size_t>(id);
}

__launch_bounds__(128) __global__ void KernelA(Args a) {
  const std::size_t lane = threadIdx.x & 31U;
  const std::size_t m = blockIdx.x * 4 + (threadIdx.x >> 5U);
  if (m >= a.rows) return;
  const std::size_t ts = blockIdx.y, e = Expert(a, ts);
  const float* xr = a.x + (ts / a.x_share) * a.k;
  const float va = DotA(a.wa + e * a.ebytes + m * a.row_bytes, xr, a.k, lane);
  float out = va;
  if (a.wb != nullptr) {
    const float vb = DotA(a.wb + e * a.ebytes + m * a.row_bytes, xr, a.k, lane);
    out = (va / (1.0F + expf(-va))) * vb;
  }
  if (lane == 0) a.y[ts * a.rows + m] = out;
}

template <int KS>
__launch_bounds__(128) __global__ void KernelB(Args a) {
  __shared__ float pa[4], pb[4];
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, ks = wave % KS;
  const std::size_t m = blockIdx.x * (4 / KS) + wave / KS;
  const std::size_t ts = blockIdx.y;
  float va = 0.F, vb = 0.F;
  if (m < a.rows) {
    const std::size_t e = Expert(a, ts);
    const float* xr = a.x + (ts / a.x_share) * a.k;
    va = DotB(a.wa + e * a.ebytes + m * a.row_bytes, xr, a.k, lane, ks, KS);
    if (a.wb != nullptr) vb = DotB(a.wb + e * a.ebytes + m * a.row_bytes, xr, a.k, lane, ks, KS);
  }
  if constexpr (KS > 1) {
    if (lane == 0) pa[wave] = va, pb[wave] = vb;
    __syncthreads();
    if (ks != 0) return;
    va = 0.F, vb = 0.F;
    for (int i = 0; i < KS; ++i) va += pa[wave + i], vb += pb[wave + i];
  }
  if (lane == 0 && m < a.rows) a.y[ts * a.rows + m] = a.wb != nullptr ? (va / (1.0F + expf(-va))) * vb : va;
}

// host -------------------------------------------------------------------------------------------------------
float H2F(std::uint16_t h) {
  const std::uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, f = h & 1023;
  float v;
  if (e == 0) v = std::ldexp(static_cast<float>(f), -24);
  else if (e == 31) v = f ? NAN : INFINITY;
  else v = std::ldexp(static_cast<float>(f | 1024), static_cast<int>(e) - 25);
  return s ? -v : v;
}
std::uint16_t F2H(float v) {  // normal range only, round to nearest
  int ex;
  const float mnt = std::frexp(std::fabs(v), &ex);  // v = mnt * 2^ex, mnt in [0.5,1)
  const std::uint32_t f = static_cast<std::uint32_t>(std::lround((mnt * 2.0F - 1.0F) * 1024.0F));
  return static_cast<std::uint16_t>(((v < 0) << 15) | ((ex - 1 + 15) << 10) | (f & 1023));
}
void ScaleMin(int j, const std::uint8_t* q, int& d, int& m) {
  if (j < 4) d = q[j] & 63, m = q[j + 4] & 63;
  else d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
}
double Dot64(const std::uint8_t* row, const float* x, std::size_t k) {
  double s = 0;
  for (std::size_t b = 0; b < k / 256; ++b) {
    const std::uint8_t* blk = row + b * kBlk;
    std::uint16_t dh, mh;
    std::memcpy(&dh, blk, 2), std::memcpy(&mh, blk + 2, 2);
    const double d = H2F(dh), dm = H2F(mh);
    for (int j = 0; j < 4; ++j) {
      int sc0, m0, sc1, m1;
      ScaleMin(2 * j, blk + 4, sc0, m0), ScaleMin(2 * j + 1, blk + 4, sc1, m1);
      for (int l = 0; l < 32; ++l) {
        const int q = blk[16 + 32 * j + l];
        s += (d * sc0 * (q & 15) - dm * m0) * x[b * 256 + 64 * j + l];
        s += (d * sc1 * (q >> 4) - dm * m1) * x[b * 256 + 64 * j + 32 + l];
      }
    }
  }
  return s;
}

}  // namespace xr1

int main(int argc, char** argv) {
  using namespace xr1;
  std::string name = "case", gguf;
  std::size_t rows = 512, k = 2048, n_exp = 256, slots = 8, tokens = 1, xshare = 0, off_a = 0, off_b = 0;
  int iters = 200, warm = 20;
  bool pair = false;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto nx = [&]() { return std::string(argv[++i]); };
    if (s == "--case") name = nx();
    else if (s == "--rows") rows = std::stoull(nx());
    else if (s == "--k") k = std::stoull(nx());
    else if (s == "--experts") n_exp = std::stoull(nx());
    else if (s == "--slots") slots = std::stoull(nx());
    else if (s == "--tokens") tokens = std::stoull(nx());
    else if (s == "--xshare") xshare = std::stoull(nx());
    else if (s == "--pair") pair = true;
    else if (s == "--gguf") gguf = nx();
    else if (s == "--off-a") off_a = std::stoull(nx());
    else if (s == "--off-b") off_b = std::stoull(nx());
    else if (s == "--iters") iters = std::stoi(nx());
    else if (s == "--warm") warm = std::stoi(nx());
  }
  if (xshare == 0) xshare = slots;
  const std::size_t row_bytes = k / 256 * kBlk, ebytes = rows * row_bytes, tbytes = ebytes * n_exp;
  const int nmat = pair ? 2 : 1;
  std::vector<std::vector<std::uint8_t>> host(nmat, std::vector<std::uint8_t>(tbytes));
  std::mt19937_64 rng(12345);
  for (int w = 0; w < nmat; ++w) {
    if (!gguf.empty()) {
      const int fd = open(gguf.c_str(), O_RDONLY);
      if (fd < 0) return std::perror("open"), 1;
      std::size_t done = 0;
      const std::size_t off = w == 0 ? off_a : off_b;
      while (done < tbytes) {
        const ssize_t r = pread(fd, host[w].data() + done, tbytes - done, static_cast<off_t>(off + done));
        if (r <= 0) return std::perror("pread"), 1;
        done += static_cast<std::size_t>(r);
      }
      close(fd);
    } else {  // synthetic codes and scales in the real block layout; d, dmin normal halves
      std::uniform_real_distribution<float> ud(0.002F, 0.02F);
      for (std::size_t i = 0; i < tbytes; i += 8) {
        const std::uint64_t v = rng();
        std::memcpy(host[w].data() + i, &v, std::min<std::size_t>(8, tbytes - i));
      }
      for (std::size_t b = 0; b < tbytes / kBlk; ++b) {
        const std::uint16_t d = F2H(ud(rng)), dm = F2H(ud(rng));
        std::memcpy(host[w].data() + b * kBlk, &d, 2), std::memcpy(host[w].data() + b * kBlk + 2, &dm, 2);
      }
    }
  }
  std::uint8_t* dw[2] = {nullptr, nullptr};
  for (int w = 0; w < nmat; ++w) {
    HC(hipMalloc(&dw[w], tbytes));
    HC(hipMemcpy(dw[w], host[w].data(), tbytes, hipMemcpyHostToDevice));
  }
  const std::size_t ts_n = tokens * slots, xrows = ts_n / xshare;
  std::vector<float> xh(xrows * k);
  std::uniform_real_distribution<float> ux(-1.F, 1.F);
  for (auto& v : xh) v = ux(rng);
  std::vector<std::int32_t> idh(static_cast<std::size_t>(warm + iters) * ts_n);
  double distinct_sum = 0;
  for (int it = 0; it < warm + iters; ++it)
    for (std::size_t t = 0; t < tokens; ++t) {
      std::set<int> used;
      for (std::size_t s = 0; s < slots; ++s) {
        int e;
        do e = static_cast<int>(rng() % n_exp); while (used.count(e));
        used.insert(e);
        idh[it * ts_n + t * slots + s] = e;
      }
    }
  for (int it = warm; it < warm + iters; ++it)
    distinct_sum += std::set<int>(idh.begin() + it * ts_n, idh.begin() + (it + 1) * ts_n).size();
  const double distinct = distinct_sum / iters;
  float *dx, *dy[4];
  std::int32_t* did;
  HC(hipMalloc(&dx, xh.size() * 4));
  HC(hipMemcpy(dx, xh.data(), xh.size() * 4, hipMemcpyHostToDevice));
  HC(hipMalloc(&did, idh.size() * 4));
  HC(hipMemcpy(did, idh.data(), idh.size() * 4, hipMemcpyHostToDevice));
  for (auto& p : dy) HC(hipMalloc(&p, ts_n * rows * 4));
  hipStream_t st;
  HC(hipStreamCreate(&st));
  Args base{dw[0], dw[1], ebytes, row_bytes, did, dx, nullptr, xshare, rows, k, n_exp};
  const char* arm_name[4] = {"ref", "a", "b_ks1", "b_ks2"};
  auto launch = [&](int arm, int it, float* y) {
    Args a = base;
    a.ids = did + static_cast<std::size_t>(it) * ts_n;
    a.y = y;
    const auto ts = static_cast<std::uint32_t>(ts_n);
    switch (arm) {
      case 0:
        flerp::moe::IndexedExperts(dw[0], GgmlType::kQ4_K, dw[1], GgmlType::kQ4_K, a.ids, dx, y, tokens, slots, xshare,
                                   rows, k, n_exp, st);
        break;
      case 1: KernelA<<<dim3((rows + 3) / 4, ts), 128, 0, st>>>(a); break;
      case 2: KernelB<1><<<dim3((rows + 3) / 4, ts), 128, 0, st>>>(a); break;
      case 3: KernelB<2><<<dim3((rows + 1) / 2, ts), 128, 0, st>>>(a); break;
    }
  };
  // correctness on iteration 0's ids
  std::vector<std::vector<float>> out(4, std::vector<float>(ts_n * rows));
  for (int arm = 0; arm < 4; ++arm) {
    launch(arm, 0, dy[arm]);
    HC(hipStreamSynchronize(st));
    HC(hipMemcpy(out[arm].data(), dy[arm], out[arm].size() * 4, hipMemcpyDeviceToHost));
  }
  // fp64 reference: token 0, all slots, all rows
  double rms = 0;
  std::vector<double> r64(slots * rows);
  for (std::size_t s = 0; s < slots; ++s) {
    const int e = idh[s];
    const float* xr = xh.data() + (s / xshare) * k;
    for (std::size_t m = 0; m < rows; ++m) {
      const double g = Dot64(host[0].data() + e * ebytes + m * row_bytes, xr, k);
      double v = g;
      if (pair) v = g / (1 + std::exp(-g)) * Dot64(host[1].data() + e * ebytes + m * row_bytes, xr, k);
      r64[s * rows + m] = v, rms += v * v;
    }
  }
  rms = std::sqrt(rms / r64.size());
  std::string corr;
  for (int arm = 0; arm < 4; ++arm) {
    double e64 = 0, eref = 0, rr = 0;
    std::size_t ndiff = 0;
    for (std::size_t i = 0; i < out[arm].size(); ++i) {
      eref = std::max(eref, static_cast<double>(std::fabs(out[arm][i] - out[0][i])));
      rr += static_cast<double>(out[0][i]) * out[0][i];
      ndiff += std::memcmp(&out[arm][i], &out[0][i], 4) != 0;
    }
    for (std::size_t i = 0; i < r64.size(); ++i) e64 = std::max(e64, std::fabs(out[arm][i] - r64[i]));
    rr = std::sqrt(rr / out[arm].size());
    char b[256];
    std::snprintf(b, sizeof b, "%s\"%s\":{\"outputs_differing_from_ref\":%zu,\"max_abs_vs_ref_over_rms\":%.3e,\"max_abs_vs_fp64_over_rms\":%.3e}",
                  arm ? "," : "", arm_name[arm], ndiff, eref / rr, e64 / rms);
    corr += b;
  }
  // timing
  std::string tim;
  hipEvent_t e0, e1;
  HC(hipEventCreate(&e0));
  HC(hipEventCreate(&e1));
  const double bytes = distinct * static_cast<double>(ebytes) * nmat;
  for (int arm = 0; arm < 4; ++arm) {
    for (int it = 0; it < warm; ++it) launch(arm, it, dy[arm]);
    HC(hipEventRecord(e0, st));
    for (int it = warm; it < warm + iters; ++it) launch(arm, it, dy[arm]);
    HC(hipEventRecord(e1, st));
    HC(hipEventSynchronize(e1));
    float ms = 0;
    HC(hipEventElapsedTime(&ms, e0, e1));
    const double us = 1e3 * ms / iters;
    char b[200];
    std::snprintf(b, sizeof b, "%s\"%s\":{\"us\":%.2f,\"GBps\":%.1f}", arm ? "," : "", arm_name[arm], us, bytes / (us * 1e3));
    tim += b;
  }
  std::printf("{\"case\":\"%s\",\"rows\":%zu,\"k\":%zu,\"experts\":%zu,\"slots\":%zu,\"tokens\":%zu,\"x_share\":%zu,\"pair\":%s,"
              "\"real_weights\":%s,\"distinct_experts_per_launch\":%.2f,\"bytes_per_launch\":%.0f,\"timing\":{%s},\"correctness\":{%s}}\n",
              name.c_str(), rows, k, n_exp, slots, tokens, xshare, pair ? "true" : "false", gguf.empty() ? "false" : "true",
              distinct, bytes, tim.c_str(), corr.c_str());
  return 0;
}
