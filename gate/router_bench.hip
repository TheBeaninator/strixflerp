// ROUTER-GEMM (c6473d6063) kernel bench, gate/router-gemm.toml R1 (bitwise logits) and R2 (speed): flerp::moe::GateLogits
// with the tiled prompt path off (GateLogitsWideKernel) and on (GateLogitsTiledKernel) on the same random FP32 router,
// shared gate and x. One JSON line a case.
//   ./router-bench --hidden 2048 --experts 256 --tokens 4096 [--reps 7]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "flerp/moe_kquant/moe_kquant.hpp"

#define CK(x)                                                                 \
  do {                                                                        \
    hipError_t e_ = (x);                                                      \
    if (e_ != hipSuccess) {                                                   \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
      std::exit(2);                                                           \
    }                                                                         \
  } while (0)

int main(int argc, char** argv) {
  std::size_t hidden = 2048, experts = 256, tokens = 4096;
  int reps = 7;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    const std::size_t v = std::strtoull(argv[i + 1], nullptr, 10);
    if (a == "--hidden") hidden = v;
    else if (a == "--experts") experts = v;
    else if (a == "--tokens") tokens = v;
    else if (a == "--reps") reps = static_cast<int>(v);
  }
  const std::size_t stride = experts + 1;
  std::mt19937 rng(1234);
  std::normal_distribution<float> nd(0.0F, 1.0F);
  std::vector<float> hw(experts * hidden), hs(hidden), hx(tokens * hidden);
  for (auto& v : hw) v = 0.02F * nd(rng);
  for (auto& v : hs) v = 0.02F * nd(rng);
  for (auto& v : hx) v = nd(rng);
  float *w, *s, *x, *l0, *l1;
  CK(hipMalloc(&w, hw.size() * 4));
  CK(hipMalloc(&s, hs.size() * 4));
  CK(hipMalloc(&x, hx.size() * 4));
  CK(hipMalloc(&l0, tokens * stride * 4));
  CK(hipMalloc(&l1, tokens * stride * 4));
  CK(hipMemcpy(w, hw.data(), hw.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(s, hs.data(), hs.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemset(l0, 0xFF, tokens * stride * 4));
  CK(hipMemset(l1, 0xFF, tokens * stride * 4));
  hipEvent_t a, b;
  CK(hipEventCreate(&a));
  CK(hipEventCreate(&b));
  auto run = [&](int tiled, float* out) {
    flerp::moe::SetRouterTiled(tiled);
    CK(hipEventRecord(a, nullptr));
    flerp::moe::GateLogits(w, s, x, out, tokens, experts, hidden, stride, nullptr);
    CK(hipEventRecord(b, nullptr));
    CK(hipEventSynchronize(b));
    float ms = 0;
    CK(hipEventElapsedTime(&ms, a, b));
    return ms;
  };
  run(0, l0);
  run(1, l1);  // warm-up, and the identity sample
  std::vector<std::uint32_t> r0(tokens * stride), r1(tokens * stride);
  CK(hipMemcpy(r0.data(), l0, r0.size() * 4, hipMemcpyDeviceToHost));
  CK(hipMemcpy(r1.data(), l1, r1.size() * 4, hipMemcpyDeviceToHost));
  std::size_t diff = 0, first = static_cast<std::size_t>(-1);
  for (std::size_t i = 0; i < r0.size(); ++i) {
    if (r0[i] != r1[i]) {
      if (first == static_cast<std::size_t>(-1)) first = i;
      ++diff;
    }
  }
  std::vector<float> t0, t1;
  for (int r = 0; r < reps; ++r) {
    t0.push_back(run(0, l0));
    t1.push_back(run(1, l1));
  }
  auto med = [](std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
  const double flop = 2.0 * static_cast<double>(tokens) * static_cast<double>(stride) * static_cast<double>(hidden);
  const float m0 = med(t0), m1 = med(t1);
  std::printf("{\"event\":\"router\",\"hidden\":%zu,\"experts\":%zu,\"tokens\":%zu,\"words\":%zu,\"differ\":%zu,"
              "\"first_differ\":%lld,\"wide_ms\":%.4f,\"tiled_ms\":%.4f,\"ratio\":%.4f,\"tiled_tflops\":%.2f,\"reps\":%d}\n",
              hidden, experts, tokens, r0.size(), diff, diff ? static_cast<long long>(first) : -1LL, m0, m1, m1 / m0,
              flop / (m1 * 1e-3) / 1e12, reps);
  return diff == 0 ? 0 : 1;
}
