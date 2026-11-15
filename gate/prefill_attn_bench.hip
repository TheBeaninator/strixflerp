// SPDX-License-Identifier: MIT
// PREFILL-ORNITH attention bench (gate/prefill-attn.toml): kernel.flerp.attn_wmma_gqa at qwen35moe shapes, synthetic
// q, gate, and an FP16 KV cache plane [position][kv_heads x 256] as the executor's fused write leaves it.
//   prefill-attn-bench --check --heads H --kv K --depth D --rows N [--gate]     float64 reference on sampled rows
//   prefill-attn-bench --speed --heads H --kv K --depth D --rows N [--reps R]
// Build: hipcc -O3 --offload-arch=gfx1151 -std=c++17 -I src prefill_attn_bench.hip src/flerp/attn_wmma_gqa/attn_wmma_gqa.hip
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "flerp/attn_wmma_gqa/attn_wmma_gqa.hpp"

#define CK(x)                                                                               \
  do {                                                                                      \
    hipError_t e_ = (x);                                                                    \
    if (e_ != hipSuccess) {                                                                 \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
      std::exit(2);                                                                         \
    }                                                                                       \
  } while (0)

constexpr int kDim = 256;

__device__ inline float Hn(uint64_t k) {
  uint64_t x = k * 0x9E3779B97F4A7C15ULL;
  x ^= x >> 31;
  x *= 0xBF58476D1CE4E5B9ULL;
  x ^= x >> 27;
  const float u1 = fmaxf(((x >> 40) & 0xFFFFFF) / 16777216.0F, 1e-7F);
  const float u2 = ((x >> 16) & 0xFFFFFF) / 16777216.0F;
  return sqrtf(-2.0F * logf(u1)) * cosf(6.2831853F * u2);
}
__global__ void FillF32(float* p, std::size_t n, uint64_t seed, float scale) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) p[i] = scale * Hn(seed * 1000003ULL + i);
}
__global__ void FillF16(__half* p, std::size_t n, uint64_t seed, float scale) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) p[i] = __float2half(scale * Hn(seed * 1000003ULL + i));
}

struct Case {
  int heads, kv, depth, rows;
  bool gate;
  float *q = nullptr, *g = nullptr, *out = nullptr;
  __half *k = nullptr, *v = nullptr;
  void *pk = nullptr, *pv = nullptr;
  std::size_t pbytes = 0;
};

void Setup(Case& c) {
  const std::size_t len = static_cast<std::size_t>(c.depth) + c.rows;
  const std::size_t qn = static_cast<std::size_t>(c.rows) * c.heads * kDim, kn = len * c.kv * kDim;
  CK(hipMalloc(&c.q, qn * 4));
  CK(hipMalloc(&c.out, qn * 4));
  CK(hipMalloc(&c.k, kn * 2));
  CK(hipMalloc(&c.v, kn * 2));
  FillF32<<<(qn + 255) / 256, 256>>>(c.q, qn, 1, 1.0F);
  FillF16<<<(kn + 255) / 256, 256>>>(c.k, kn, 2, 1.0F);
  FillF16<<<(kn + 255) / 256, 256>>>(c.v, kn, 3, 1.0F);
  if (c.gate) {
    CK(hipMalloc(&c.g, qn * 4));
    FillF32<<<(qn + 255) / 256, 256>>>(c.g, qn, 4, 1.0F);
  }
  c.pbytes = static_cast<std::size_t>(c.kv) * ((len + 15) & ~std::size_t{15}) * kDim * 2;
  CK(hipMalloc(&c.pk, c.pbytes));
  CK(hipMalloc(&c.pv, c.pbytes));
  CK(hipDeviceSynchronize());
}

bool Run(Case& c, hipStream_t s) {
  return flerp::attn::LaunchGqaWmmaAttention(c.q, c.g, c.k, c.v, c.out, 0, c.depth, c.rows, c.depth + c.rows, c.heads,
                                             c.kv, kDim, s, c.pk, c.pbytes, c.pv, c.pbytes);
}

double H2D(__half h) { return static_cast<double>(__half2float(h)); }

int Check(Case c) {
  Setup(c);
  if (!Run(c, nullptr)) {
    std::printf("{\"event\":\"check\",\"error\":\"not launched\"}\n");
    return 1;
  }
  CK(hipDeviceSynchronize());
  const std::size_t len = static_cast<std::size_t>(c.depth) + c.rows;
  const std::size_t qn = static_cast<std::size_t>(c.rows) * c.heads * kDim, kn = len * c.kv * kDim;
  std::vector<float> q(qn), out(qn), g(c.gate ? qn : 0);
  std::vector<__half> k(kn), v(kn);
  CK(hipMemcpy(q.data(), c.q, qn * 4, hipMemcpyDeviceToHost));
  CK(hipMemcpy(out.data(), c.out, qn * 4, hipMemcpyDeviceToHost));
  CK(hipMemcpy(k.data(), c.k, kn * 2, hipMemcpyDeviceToHost));
  CK(hipMemcpy(v.data(), c.v, kn * 2, hipMemcpyDeviceToHost));
  if (c.gate) CK(hipMemcpy(g.data(), c.g, qn * 4, hipMemcpyDeviceToHost));
  std::mt19937 rng(7);
  std::vector<int> rows;
  for (int r = 0; r < c.rows; r += std::max(1, c.rows / 24)) rows.push_back(r);
  rows.push_back(c.rows - 1);
  double ss = 0, se = 0, mx = 0;
  std::size_t n = 0, nonfinite = 0;
  for (int r : rows) {
    for (int h = 0; h < c.heads; h += std::max(1, c.heads / 8)) {
      const int kvh = h / (c.heads / c.kv);
      const std::size_t visible = static_cast<std::size_t>(c.depth) + r + 1;
      std::vector<double> sc(visible);
      double m = -1e300;
      const float* qr = q.data() + (static_cast<std::size_t>(r) * c.heads + h) * kDim;
      for (std::size_t j = 0; j < visible; ++j) {
        const __half* kr = k.data() + (j * c.kv + kvh) * kDim;
        double d = 0;
        for (int i = 0; i < kDim; ++i) d += static_cast<double>(qr[i]) * H2D(kr[i]);
        sc[j] = d / 16.0;
        m = std::max(m, sc[j]);
      }
      double sum = 0;
      for (auto& x : sc) sum += (x = std::exp(x - m));
      for (int i = 0; i < kDim; ++i) {
        double o = 0;
        for (std::size_t j = 0; j < visible; ++j) o += sc[j] * H2D(v[(j * c.kv + kvh) * kDim + i]);
        o /= sum;
        const std::size_t idx = (static_cast<std::size_t>(r) * c.heads + h) * kDim + i;
        if (c.gate) o *= 1.0 / (1.0 + std::exp(-static_cast<double>(g[idx])));
        const double got = out[idx];
        if (!std::isfinite(got)) ++nonfinite;
        ss += o * o;
        se += (got - o) * (got - o);
        mx = std::max(mx, std::fabs(got - o));
        ++n;
      }
    }
  }
  const double rms = std::sqrt(ss / n);
  std::printf("{\"event\":\"check\",\"heads\":%d,\"kv\":%d,\"depth\":%d,\"rows\":%d,\"gate\":%s,\"n\":%zu,\"rel_rms\":%.4g,"
              "\"max_abs_over_rms\":%.4g,\"ref_rms\":%.4g,\"nonfinite\":%zu}\n",
              c.heads, c.kv, c.depth, c.rows, c.gate ? "true" : "false", n, std::sqrt(se / ss), mx / rms, rms, nonfinite);
  std::fflush(stdout);
  return 0;
}

int Speed(Case c, int reps) {
  Setup(c);
  hipEvent_t e0, e1;
  CK(hipEventCreate(&e0));
  CK(hipEventCreate(&e1));
  std::vector<double> t;
  for (int r = 0; r < reps + 1; ++r) {
    CK(hipEventRecord(e0, nullptr));
    if (!Run(c, nullptr)) {
      std::printf("{\"event\":\"speed\",\"error\":\"not launched\"}\n");
      return 1;
    }
    CK(hipEventRecord(e1, nullptr));
    CK(hipEventSynchronize(e1));
    float ms;
    CK(hipEventElapsedTime(&ms, e0, e1));
    if (r > 0) t.push_back(ms);
  }
  std::sort(t.begin(), t.end());
  const double ms = t[t.size() / 2];
  const double keys = static_cast<double>(c.rows) * c.depth + static_cast<double>(c.rows) * (c.rows + 1) / 2.0;
  const double flop = 4.0 * c.heads * kDim * keys;
  std::printf("{\"event\":\"speed\",\"heads\":%d,\"kv\":%d,\"depth\":%d,\"rows\":%d,\"ms\":%.3f,\"tflops\":%.2f,\"reps\":%d}\n",
              c.heads, c.kv, c.depth, c.rows, ms, flop / ms / 1e9, reps);
  std::fflush(stdout);
  return 0;
}

// G4: the same queries as one launch of `rows` rows at depth 0 against launches of `chunk` rows at depths 0, chunk, ..
// over the same cache: every output float compared bit for bit.
int ChunkCheck(Case c, int chunk) {
  Setup(c);
  const std::size_t width = static_cast<std::size_t>(c.heads) * kDim;
  if (!Run(c, nullptr)) return 1;
  CK(hipDeviceSynchronize());
  std::vector<float> whole(static_cast<std::size_t>(c.rows) * width), parts(whole.size());
  CK(hipMemcpy(whole.data(), c.out, whole.size() * 4, hipMemcpyDeviceToHost));
  CK(hipMemset(c.out, 0, whole.size() * 4));
  for (int start = 0; start < c.rows; start += chunk) {
    const int n = std::min(chunk, c.rows - start);
    if (!flerp::attn::LaunchGqaWmmaAttention(c.q + start * width, c.g ? c.g + start * width : nullptr, c.k, c.v,
                                             c.out + start * width, 0, start, n, c.depth + c.rows, c.heads, c.kv, kDim,
                                             nullptr, c.pk, c.pbytes, c.pv, c.pbytes))
      return 1;
  }
  CK(hipDeviceSynchronize());
  CK(hipMemcpy(parts.data(), c.out, parts.size() * 4, hipMemcpyDeviceToHost));
  std::size_t diff = 0;
  for (std::size_t i = 0; i < whole.size(); ++i) diff += std::memcmp(&whole[i], &parts[i], 4) != 0;
  std::printf("{\"event\":\"chunkcheck\",\"heads\":%d,\"kv\":%d,\"rows\":%d,\"chunk\":%d,\"gate\":%s,\"floats\":%zu,\"differing\":%zu}\n",
              c.heads, c.kv, c.rows, chunk, c.gate ? "true" : "false", whole.size(), diff);
  std::fflush(stdout);
  return 0;
}

int main(int argc, char** argv) {
  std::string mode;
  Case c{32, 2, 4096, 64, false};
  int reps = 5, chunk = 4096;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto v = [&]() { return std::atoi(argv[++i]); };
    if (a == "--check" || a == "--speed" || a == "--chunkcheck") mode = a;
    else if (a == "--chunk") chunk = v();
    else if (a == "--heads") c.heads = v();
    else if (a == "--kv") c.kv = v();
    else if (a == "--depth") c.depth = v();
    else if (a == "--rows") c.rows = v();
    else if (a == "--reps") reps = v();
    else if (a == "--gate") c.gate = true;
    else return 2;
  }
  if (mode == "--chunkcheck") return ChunkCheck(c, chunk);
  return mode == "--check" ? Check(c) : Speed(c, reps);
}
