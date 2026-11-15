// MOE-COMBINE (2dc55e67f0) cause bench, gate/moe-combine.toml C1: why MoeEpilogueKernel moves 335 MB a block in 7 ms
// inside the 35B prompt chunk. Built with the qfn kernels' flags (-O3 -ffast-math -fno-finite-math-only). Kernels:
//   orig   MoeEpilogueKernel, copied verbatim from catalog/gufo/kernel/qfn_kernels (kernels.hip.cpp), its launch shape
//   vec4o  the same per-element expression (Materialize(acc + w * v) slot by slot, then acc + SigmoidF(g) * sh), four
//          adjacent elements a thread (the fix candidate; its words are compared with orig's)
//   probe  a float4 read of the same bytes (slot rows + shared) and a write of out: the memory rate of this pass
// Each runs isolated (expert_out written long before) and right after a writer that fills expert_out the way the
// grouped Down does (each pair's FP32 row from 128-column blocks, pairs in expert-sorted order), whose time is kept
// separate. One JSON line a case.
//   ./combine-bench --tokens 4096 --slots 8 --dim 2048 [--reps 7]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define CK(x)                                                                              \
  do {                                                                                     \
    hipError_t e_ = (x);                                                                   \
    if (e_ != hipSuccess) {                                                                \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_));       \
      std::exit(2);                                                                        \
    }                                                                                      \
  } while (0)

constexpr unsigned kThreads = 256;
__device__ __forceinline__ float SigmoidF(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float Materialize(float v) {
  asm volatile("" : "+v"(v));
  return v;
}

// --- verbatim from kernels.hip.cpp (MoeEpilogueKernel) ---
__global__ void MoeEpilogueKernel(const float* expert_out, const float* weights,
                                  const float* shared_out, const float* gate,
                                  std::uint32_t gate_stride, float* out,
                                  std::uint32_t k, std::uint32_t dim) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i >= dim) {
    return;
  }
  constexpr std::uint32_t kMaxSlots = 16;
  if (k <= kMaxSlots) {
    const float* rows = expert_out + static_cast<std::size_t>(t) * k * dim + i;
    float w[kMaxSlots];
    float v[kMaxSlots];
#pragma unroll
    for (std::uint32_t s = 0; s < kMaxSlots; ++s) {
      w[s] = s < k ? weights[t * k + s] : 0.0f;
      v[s] = s < k ? rows[static_cast<std::size_t>(s) * dim] : 0.0f;
    }
    const std::size_t idx = static_cast<std::size_t>(t) * dim + i;
    const float sh = shared_out[idx];
    const float g = gate[static_cast<std::size_t>(t) * gate_stride];
    float acc = Materialize(0.0f);
#pragma unroll
    for (std::uint32_t s = 0; s < kMaxSlots; ++s) {
      if (s < k) {
        acc = Materialize(acc + w[s] * v[s]);
      }
    }
    out[idx] = acc + SigmoidF(g) * sh;
    return;
  }
  float acc = 0.0f;
  for (std::uint32_t s = 0; s < k; ++s) {
    acc += weights[t * k + s] *
           expert_out[(static_cast<std::size_t>(t) * k + s) * dim + i];
  }
  const std::size_t idx = static_cast<std::size_t>(t) * dim + i;
  out[idx] = acc + SigmoidF(gate[static_cast<std::size_t>(t) * gate_stride]) *
                       shared_out[idx];
}
// --- end verbatim ---

// The same expression per element as MoeEpilogueKernel's k <= 16 branch, four adjacent elements a thread.
__global__ void MoeEpilogueVec4OrigKernel(const float* expert_out, const float* weights, const float* shared_out,
                                          const float* gate, std::uint32_t gate_stride, float* out, std::uint32_t k,
                                          std::uint32_t dim) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
  if (i >= dim) return;
  constexpr std::uint32_t kMaxSlots = 16;
  const float* rows = expert_out + static_cast<std::size_t>(t) * k * dim + i;
  float w[kMaxSlots];
  float4 v[kMaxSlots];
#pragma unroll
  for (std::uint32_t s = 0; s < kMaxSlots; ++s) {
    w[s] = s < k ? weights[t * k + s] : 0.0f;
    v[s] = s < k ? *reinterpret_cast<const float4*>(rows + static_cast<std::size_t>(s) * dim) : float4{0, 0, 0, 0};
  }
  const std::size_t idx = static_cast<std::size_t>(t) * dim + i;
  const float4 sh = *reinterpret_cast<const float4*>(shared_out + idx);
  const float g = gate[static_cast<std::size_t>(t) * gate_stride];
  float a0 = Materialize(0.0f), a1 = Materialize(0.0f), a2 = Materialize(0.0f), a3 = Materialize(0.0f);
#pragma unroll
  for (std::uint32_t s = 0; s < kMaxSlots; ++s) {
    if (s < k) {
      a0 = Materialize(a0 + w[s] * v[s].x);
      a1 = Materialize(a1 + w[s] * v[s].y);
      a2 = Materialize(a2 + w[s] * v[s].z);
      a3 = Materialize(a3 + w[s] * v[s].w);
    }
  }
  float4 o;
  o.x = a0 + SigmoidF(g) * sh.x;
  o.y = a1 + SigmoidF(g) * sh.y;
  o.z = a2 + SigmoidF(g) * sh.z;
  o.w = a3 + SigmoidF(g) * sh.w;
  *reinterpret_cast<float4*>(out + idx) = o;
}

__global__ void ProbeKernel(const float* expert_out, const float* shared_out, float* out, std::uint32_t k,
                            std::uint32_t dim) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
  if (i >= dim) return;
  const float* rows = expert_out + static_cast<std::size_t>(t) * k * dim + i;
  float4 a = *reinterpret_cast<const float4*>(shared_out + static_cast<std::size_t>(t) * dim + i);
  for (std::uint32_t s = 0; s < k; ++s) {
    const float4 v = *reinterpret_cast<const float4*>(rows + static_cast<std::size_t>(s) * dim);
    a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
  }
  *reinterpret_cast<float4*>(out + static_cast<std::size_t>(t) * dim + i) = a;
}

// Writes expert_out like the grouped Down: block (tile of 128 sorted pairs, 128-column block); each pair's row at its
// own (token * k + slot) position, 16-column pieces per lane group.
__global__ void WriterKernel(const int* __restrict__ sorted, float* expert_out, std::uint32_t pairs, std::uint32_t dim) {
  const std::uint32_t p0 = blockIdx.y * 128, c0 = blockIdx.x * 128;
  for (std::uint32_t r = threadIdx.x / 16; r < 128; r += kThreads / 16) {
    const std::uint32_t pos = p0 + r;
    if (pos >= pairs) break;
    const int pair = sorted[pos];
    float* row = expert_out + static_cast<std::size_t>(pair) * dim + c0;
    for (std::uint32_t c = (threadIdx.x & 15); c < 128; c += 16) row[c] = static_cast<float>(pos + c);
  }
}

int main(int argc, char** argv) {
  std::uint32_t tokens = 4096, k = 8, dim = 2048;
  int reps = 7;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    const unsigned v = static_cast<unsigned>(std::strtoul(argv[i + 1], nullptr, 10));
    if (a == "--tokens") tokens = v;
    else if (a == "--slots") k = v;
    else if (a == "--dim") dim = v;
    else if (a == "--reps") reps = static_cast<int>(v);
  }
  const std::size_t pairs = static_cast<std::size_t>(tokens) * k;
  const std::size_t stride = 257;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0F, 1.0F);
  std::vector<float> h_e(pairs * dim), h_w(pairs), h_s(static_cast<std::size_t>(tokens) * dim), h_g(tokens * stride);
  for (auto& v : h_e) v = nd(rng);
  for (auto& v : h_w) v = std::abs(nd(rng)) * 0.2F;
  for (auto& v : h_s) v = nd(rng);
  for (auto& v : h_g) v = nd(rng);
  // expert-sorted pair order (random experts), as the Down kernel walks them
  std::vector<int> expert(pairs), order(pairs);
  for (auto& e : expert) e = static_cast<int>(rng() % 256);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return expert[a] < expert[b]; });
  float *e, *w, *s, *g, *o0, *o1, *o2, *junk;
  int* sorted;
  CK(hipMalloc(&e, h_e.size() * 4));
  CK(hipMalloc(&w, h_w.size() * 4));
  CK(hipMalloc(&s, h_s.size() * 4));
  CK(hipMalloc(&g, h_g.size() * 4));
  CK(hipMalloc(&o0, h_s.size() * 4));
  CK(hipMalloc(&o1, h_s.size() * 4));
  CK(hipMalloc(&o2, h_s.size() * 4));
  CK(hipMalloc(&junk, h_e.size() * 4));
  CK(hipMalloc(&sorted, pairs * 4));
  CK(hipMemcpy(w, h_w.data(), h_w.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(s, h_s.data(), h_s.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(g, h_g.data(), h_g.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(sorted, order.data(), pairs * 4, hipMemcpyHostToDevice));
  hipEvent_t a, b, c;
  CK(hipEventCreate(&a));
  CK(hipEventCreate(&b));
  CK(hipEventCreate(&c));
  const float* gate = g + 256;  // the shared-gate logit column, stride 257
  auto fill = [&]() { CK(hipMemcpy(e, h_e.data(), h_e.size() * 4, hipMemcpyHostToDevice)); };
  auto flush = [&]() { CK(hipMemset(junk, 1, h_e.size() * 4)); CK(hipDeviceSynchronize()); };  // evict caches
  auto launch = [&](int which) {
    if (which == 0)
      MoeEpilogueKernel<<<dim3(tokens, (dim + kThreads - 1) / kThreads), kThreads>>>(e, w, s, gate, stride, o0, k, dim);
    else if (which == 1)
      MoeEpilogueVec4OrigKernel<<<dim3(tokens, (dim / 4 + kThreads - 1) / kThreads), kThreads>>>(e, w, s, gate, stride, o1, k, dim);
    else
      ProbeKernel<<<dim3(tokens, (dim / 4 + kThreads - 1) / kThreads), kThreads>>>(e, s, o2, k, dim);
  };
  const char* names[3] = {"orig", "vec4o", "probe"};
  const double bytes = (static_cast<double>(pairs) * dim + 2.0 * tokens * dim) * 4.0;
  fill();
  for (int wch = 0; wch < 3; ++wch) launch(wch);  // warm-up
  CK(hipDeviceSynchronize());
  for (int mode = 0; mode < 2; ++mode) {  // 0 isolated (caches flushed), 1 right after the writer
    std::vector<float> t[3], tw;
    for (int r = 0; r < reps; ++r) {
      for (int wch = 0; wch < 3; ++wch) {
        flush();
        if (mode == 1) {
          CK(hipEventRecord(c, nullptr));
          WriterKernel<<<dim3(dim / 128, static_cast<unsigned>((pairs + 127) / 128)), kThreads>>>(sorted, e, static_cast<unsigned>(pairs), dim);
        }
        CK(hipEventRecord(a, nullptr));
        launch(wch);
        CK(hipEventRecord(b, nullptr));
        CK(hipEventSynchronize(b));
        float ms = 0;
        CK(hipEventElapsedTime(&ms, a, b));
        t[wch].push_back(ms);
        if (mode == 1) {
          float mw = 0;
          CK(hipEventElapsedTime(&mw, c, a));
          tw.push_back(mw);
        }
      }
    }
    for (int wch = 0; wch < 3; ++wch) {
      std::sort(t[wch].begin(), t[wch].end());
      const float m = t[wch][t[wch].size() / 2];
      std::printf("{\"event\":\"combine\",\"kernel\":\"%s\",\"mode\":\"%s\",\"tokens\":%u,\"slots\":%u,\"dim\":%u,\"ms\":%.4f,"
                  "\"gbps\":%.1f,\"reps\":%d}\n", names[wch], mode ? "after_writer" : "isolated", tokens, k, dim, m,
                  bytes / (m * 1e-3) / 1e9, reps);
    }
    if (mode == 1) {
      std::sort(tw.begin(), tw.end());
      std::printf("{\"event\":\"writer\",\"tokens\":%u,\"slots\":%u,\"dim\":%u,\"ms\":%.4f}\n", tokens, k, dim, tw[tw.size() / 2]);
    }
  }
  // identity: orig vs vec4o on the same random inputs
  fill();
  launch(0);
  launch(1);
  CK(hipDeviceSynchronize());
  std::vector<std::uint32_t> r0(h_s.size()), r1(h_s.size());
  CK(hipMemcpy(r0.data(), o0, r0.size() * 4, hipMemcpyDeviceToHost));
  CK(hipMemcpy(r1.data(), o1, r1.size() * 4, hipMemcpyDeviceToHost));
  std::size_t diff = 0;
  for (std::size_t i = 0; i < r0.size(); ++i) diff += r0[i] != r1[i];
  std::printf("{\"event\":\"identity\",\"orig_vs_vec4o_differ\":%zu,\"words\":%zu}\n", diff, r0.size());
  return 0;
}
