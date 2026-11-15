// SPDX-License-Identifier: MIT
// ORNITH-CRUCIBLE kernel bench (gate/ornith-crucible.toml), one binary, synthetic codes in the real GGUF block
// layouts at the real dims (no model):
//
//   ornith-crucible-bench --experts [--hidden 4096 --ff 1024 --used 10 --n-experts 512] [--down q6k|q4k] [--sdown ..]
//                         [--variants 0,1,2,3] [--check-sets 16] [--reps 200 --warm 20 --rounds 7] [--no-timing]
//       item 1: kernel.flerp.moe_sg Experts (one launch) against the current decode path, the four
//       kernel.flerp.moe_kquant IndexedExperts launches of moe_ffn.cpp (the catalogue's moe_kquant.hip itself,
//       #included below): byte equality of act, expert_out, shared_act, shared_out for every check set, x kind and
//       variant (memcmp), control words zero after every launch; then per-launch time cache-cold (each launch reads
//       its own routed experts from a 512-expert stack, and a shared expert from 64 copies), rounds alternating
//       old/new order, median round.
//
//   ornith-crucible-bench --head [--vocab 248320 --hidden 4096] [--bits 2,4] [--keep 8] [--k 8] [--reps 50 ...]
//       item 2: kernel.flerp.draft_coarse Score against the exact head (Q8KBlockGEMVKernel_2Rows' Q6_K path and
//       gufo's ArgmaxKernel, copied below): the kept rows' bits against the exact logits, the choice against the exact
//       argmax, and the time of a head step (coarse + select + rescore + choose against GEMV + argmax) and of each pass.
//
// One JSON object a line on stdout.
#include "flerp/moe_kquant/moe_kquant.hip"
#include "flerp/moe_sg/moe_sg.hip"
#include "flerp/draft_coarse/draft_coarse.hip"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// moe_kquant's ExpertBytes calls gufo::quant::QuantizedRowBytes (gufo_core); the bench links no gufo library, so the
// two types it uses are answered here (llama.cpp block sizes).
namespace gufo::quant {
std::size_t QuantizedRowBytes(core::GgmlType type, std::size_t elements) noexcept {
  if (elements % 256 != 0) return 0;
  if (type == core::GgmlType::kQ4_K) return elements / 256 * 144;
  if (type == core::GgmlType::kQ6_K) return elements / 256 * 210;
  return 0;
}
}  // namespace gufo::quant

namespace ocb {
using gufo::core::GgmlType;

#define HC(x)                                                                         \
  do {                                                                                \
    hipError_t e_ = (x);                                                              \
    if (e_ != hipSuccess) {                                                           \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
      std::exit(1);                                                                   \
    }                                                                                 \
  } while (0)

__device__ inline std::uint64_t Mix(std::uint64_t z) {
  z += 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// Random bytes, then every block's fp16 d (and dmin) set to uniform(0.002, 0.02), as decode_kern3's synthetic rows.
__global__ void FillKernel(std::uint8_t* p, std::size_t n, std::uint64_t seed) {
  const std::size_t i = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
  if (i >= n) return;
  const std::uint64_t v = Mix(seed * 0x100000001B3ULL + i);
  for (int b = 0; b < 8 && i + b < n; ++b) p[i + b] = static_cast<std::uint8_t>(v >> (8 * b));
}
__global__ void FixKernel(std::uint8_t* p, std::size_t blocks, int q6, std::uint64_t seed) {
  const std::size_t b = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= blocks) return;
  const std::uint64_t v = Mix(seed ^ (b * 0x2545F4914F6CDD1DULL));
  const float u0 = static_cast<float>(v & 0xFFFFFF) / 16777216.0F, u1 = static_cast<float>((v >> 24) & 0xFFFFFF) / 16777216.0F;
  const __half d = __float2half(0.002F + 0.018F * u0), dm = __float2half(0.002F + 0.018F * u1);
  if (q6) {
    std::memcpy(p + b * 210 + 208, &d, 2);
  } else {
    std::memcpy(p + b * 144, &d, 2);
    std::memcpy(p + b * 144 + 2, &dm, 2);
  }
}

std::size_t RowBytes(bool q6, std::size_t k) { return k / 256 * (q6 ? 210 : 144); }

std::uint8_t* Synth(bool q6, std::size_t rows, std::size_t k, std::uint64_t seed) {
  const std::size_t n = rows * RowBytes(q6, k);
  std::uint8_t* p = nullptr;
  HC(hipMalloc(&p, n));
  FillKernel<<<(n / 8 + 255) / 256 + 1, 256>>>(p, n, seed);
  const std::size_t blocks = n / (q6 ? 210 : 144);
  FixKernel<<<(blocks + 255) / 256, 256>>>(p, blocks, q6 ? 1 : 0, seed + 77);
  HC(hipGetLastError());
  return p;
}

std::vector<int> List(const std::string& s) {
  std::vector<int> v;
  std::stringstream ss(s);
  std::string it;
  while (std::getline(ss, it, ',')) {
    if (!it.empty()) v.push_back(std::stoi(it));
  }
  return v;
}

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : v[v.size() / 2];
}

template <class T>
std::vector<T> D2H(const T* p, std::size_t n) {
  std::vector<T> h(n);
  HC(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost));
  return h;
}

std::size_t DiffWords(const void* a, const void* b, std::size_t n_floats) {
  const auto* x = static_cast<const std::uint32_t*>(a);
  const auto* y = static_cast<const std::uint32_t*>(b);
  std::size_t d = 0;
  for (std::size_t i = 0; i < n_floats; ++i) d += x[i] != y[i];
  return d;
}

// ------------------------------------------------------------------------------------------------- item 1
int Experts(int argc, char** argv) {
  int H = 4096, F = 1024, U = 10, E = 512, copies = 64, id_sets = 512, check_sets = 16, reps = 200, warm = 20, rounds = 7;
  bool down_q6 = true, sdown_q6 = true, sdown_set = false, timing = true;
  std::string vlist = "0,1,2,3", name = "experts";
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto nx = [&]() { return std::string(argv[++i]); };
    if (s == "--hidden") H = std::stoi(nx());
    else if (s == "--ff") F = std::stoi(nx());
    else if (s == "--used") U = std::stoi(nx());
    else if (s == "--n-experts") E = std::stoi(nx());
    else if (s == "--down") down_q6 = nx() == "q6k";
    else if (s == "--sdown") sdown_q6 = nx() == "q6k", sdown_set = true;
    else if (s == "--variants") vlist = nx();
    else if (s == "--check-sets") check_sets = std::stoi(nx());
    else if (s == "--reps") reps = std::stoi(nx());
    else if (s == "--warm") warm = std::stoi(nx());
    else if (s == "--rounds") rounds = std::stoi(nx());
    else if (s == "--copies") copies = std::stoi(nx());
    else if (s == "--no-timing") timing = false;
    else if (s == "--case") name = nx();
  }
  if (!sdown_set) sdown_q6 = down_q6;
  const std::vector<int> variants = List(vlist);
  const GgmlType tq4 = GgmlType::kQ4_K, td = down_q6 ? GgmlType::kQ6_K : GgmlType::kQ4_K,
                 tsd = sdown_q6 ? GgmlType::kQ6_K : GgmlType::kQ4_K;
  // Weights: routed stacks [E][rows][k]; shared copies.
  std::uint8_t* gate = Synth(false, static_cast<std::size_t>(E) * F, H, 11);
  std::uint8_t* up = Synth(false, static_cast<std::size_t>(E) * F, H, 12);
  std::uint8_t* down = Synth(down_q6, static_cast<std::size_t>(E) * H, F, 13);
  std::vector<std::uint8_t*> sg(copies), su(copies), sd(copies);
  for (int c = 0; c < copies; ++c) {
    sg[c] = Synth(false, F, H, 100 + 3 * c);
    su[c] = Synth(false, F, H, 101 + 3 * c);
    sd[c] = Synth(sdown_q6, H, F, 102 + 3 * c);
  }
  HC(hipDeviceSynchronize());
  // Routing: id_sets sets of U distinct experts; set 0 carries an id past the stack and a negative id (both read the
  // last expert, as IndexedExperts does).
  std::mt19937_64 rng(4242);
  std::vector<std::int32_t> ids(static_cast<std::size_t>(id_sets) * U);
  for (int s = 0; s < id_sets; ++s) {
    std::vector<int> perm(E);
    for (int e = 0; e < E; ++e) perm[e] = e;
    for (int k = 0; k < U; ++k) {
      std::uniform_int_distribution<int> d(k, E - 1);
      std::swap(perm[k], perm[d(rng)]);
      ids[static_cast<std::size_t>(s) * U + k] = perm[k];
    }
  }
  ids[0] = E + 88;
  if (U > 1) ids[1] = -1;
  std::int32_t* d_ids = nullptr;
  HC(hipMalloc(&d_ids, ids.size() * 4));
  HC(hipMemcpy(d_ids, ids.data(), ids.size() * 4, hipMemcpyHostToDevice));
  float* d_x[2];
  for (int k = 0; k < 2; ++k) {
    std::vector<float> xh(H);
    std::mt19937_64 xr(1000 + k);
    std::uniform_real_distribution<float> ux(-1.F, 1.F);
    std::normal_distribution<float> nxd(0.F, 0.05F);
    for (auto& v : xh) v = k == 0 ? ux(xr) : nxd(xr);
    HC(hipMalloc(&d_x[k], H * 4));
    HC(hipMemcpy(d_x[k], xh.data(), H * 4, hipMemcpyHostToDevice));
  }
  struct Out {
    float *act, *eout, *sact, *sout;
  } o[2];
  for (auto& b : o) {
    HC(hipMalloc(&b.act, static_cast<std::size_t>(U) * F * 4));
    HC(hipMalloc(&b.eout, static_cast<std::size_t>(U) * H * 4));
    HC(hipMalloc(&b.sact, F * 4));
    HC(hipMalloc(&b.sout, H * 4));
  }
  auto layer_of = [&](int c) {
    flerp::moe_sg::Layer l;
    l.gate = {gate, flerp::moe_sg::WType::kQ4K};
    l.up = {up, flerp::moe_sg::WType::kQ4K};
    l.down = {down, down_q6 ? flerp::moe_sg::WType::kQ6K : flerp::moe_sg::WType::kQ4K};
    l.sgate = {sg[c], flerp::moe_sg::WType::kQ4K};
    l.sup = {su[c], flerp::moe_sg::WType::kQ4K};
    l.sdown = {sd[c], sdown_q6 ? flerp::moe_sg::WType::kQ6K : flerp::moe_sg::WType::kQ4K};
    l.hidden = H, l.ff = F, l.shared_ff = F, l.used = U, l.experts = E;
    return l;
  };
  if (!flerp::moe_sg::Supported(layer_of(0), d_x[0])) {
    std::printf("{\"event\":\"error\",\"case\":\"%s\",\"why\":\"moe_sg does not take this shape\"}\n", name.c_str());
    return 2;
  }
  unsigned* ctl = nullptr;
  const std::size_t ctl_bytes = flerp::moe_sg::CtlBytes(layer_of(0));
  HC(hipMalloc(&ctl, ctl_bytes));
  HC(hipMemset(ctl, 0, ctl_bytes));
  hipStream_t st;
  HC(hipStreamCreate(&st));
  auto old_launch = [&](int set, int c, const float* x, const Out& b) {
    const std::int32_t* id = d_ids + static_cast<std::size_t>(set) * U;
    flerp::moe::IndexedExperts(gate, tq4, up, tq4, id, x, b.act, 1, U, U, F, H, E, st);
    flerp::moe::IndexedExperts(down, td, nullptr, td, id, b.act, b.eout, 1, U, 1, H, F, E, st);
    flerp::moe::IndexedExperts(sg[c], tq4, su[c], tq4, nullptr, x, b.sact, 1, 1, 1, F, H, 1, st);
    flerp::moe::IndexedExperts(sd[c], tsd, nullptr, tsd, nullptr, b.sact, b.sout, 1, 1, 1, H, F, 1, st);
  };
  auto new_launch = [&](int set, int c, const float* x, const Out& b, int v) {
    flerp::moe_sg::Experts(layer_of(c), d_ids + static_cast<std::size_t>(set) * U, x, b.act, b.sact, b.eout, b.sout, ctl, st, v);
  };
  std::vector<int> grids;
  for (int v : variants) grids.push_back(flerp::moe_sg::Grid(layer_of(0), v));
  // Correctness.
  std::vector<std::size_t> vdiff(variants.size(), 0);
  std::size_t ctl_dirty = 0, checks = 0;
  for (int s = 0; s < check_sets; ++s) {
    for (int xk = 0; xk < 2; ++xk) {
      const int c = s % copies;
      HC(hipMemsetAsync(o[0].act, 0xA5, static_cast<std::size_t>(U) * F * 4, st));
      HC(hipMemsetAsync(o[0].eout, 0xA5, static_cast<std::size_t>(U) * H * 4, st));
      old_launch(s, c, d_x[xk], o[0]);
      HC(hipStreamSynchronize(st));
      const auto a0 = D2H(o[0].act, static_cast<std::size_t>(U) * F), e0 = D2H(o[0].eout, static_cast<std::size_t>(U) * H);
      const auto s0 = D2H(o[0].sact, F), t0 = D2H(o[0].sout, H);
      std::ostringstream js;
      js << "{\"event\":\"check\",\"case\":\"" << name << "\",\"set\":" << s << ",\"x\":" << xk << ",\"diff\":{";
      for (std::size_t vi = 0; vi < variants.size(); ++vi) {
        HC(hipMemsetAsync(o[1].act, 0x5A, static_cast<std::size_t>(U) * F * 4, st));
        HC(hipMemsetAsync(o[1].eout, 0x5A, static_cast<std::size_t>(U) * H * 4, st));
        HC(hipMemsetAsync(o[1].sact, 0x5A, F * 4, st));
        HC(hipMemsetAsync(o[1].sout, 0x5A, H * 4, st));
        new_launch(s, c, d_x[xk], o[1], variants[vi]);
        HC(hipStreamSynchronize(st));
        const auto a1 = D2H(o[1].act, static_cast<std::size_t>(U) * F), e1 = D2H(o[1].eout, static_cast<std::size_t>(U) * H);
        const auto s1 = D2H(o[1].sact, F), t1 = D2H(o[1].sout, H);
        const std::size_t d = DiffWords(a0.data(), a1.data(), a0.size()) + DiffWords(e0.data(), e1.data(), e0.size()) +
                              DiffWords(s0.data(), s1.data(), s0.size()) + DiffWords(t0.data(), t1.data(), t0.size());
        const auto cw = D2H(ctl, ctl_bytes / 4);
        for (unsigned w : cw) ctl_dirty += w != 0;
        vdiff[vi] += d;
        js << (vi ? "," : "") << "\"v" << variants[vi] << "\":" << d;
      }
      bool finite = true;
      for (float f : e0) finite = finite && std::isfinite(f);
      js << "},\"old_finite\":" << (finite ? "true" : "false") << "}";
      std::printf("%s\n", js.str().c_str());
      ++checks;
    }
  }
  {
    std::ostringstream js;
    js << "{\"event\":\"check_total\",\"case\":\"" << name << "\",\"checks\":" << checks << ",\"ctl_dirty\":" << ctl_dirty
       << ",\"diff\":{";
    for (std::size_t vi = 0; vi < variants.size(); ++vi) js << (vi ? "," : "") << "\"v" << variants[vi] << "\":" << vdiff[vi];
    js << "}}";
    std::printf("%s\n", js.str().c_str());
    std::fflush(stdout);
  }
  if (!timing) return 0;
  // Timing: launch i reads id set i % id_sets and shared copy i % copies.
  const double bytes = static_cast<double>(U) * (2.0 * F * RowBytes(false, H) + static_cast<double>(H) * RowBytes(down_q6, F)) +
                       2.0 * F * RowBytes(false, H) + static_cast<double>(H) * RowBytes(sdown_q6, F);
  hipEvent_t ev0, ev1;
  HC(hipEventCreate(&ev0));
  HC(hipEventCreate(&ev1));
  const int arms = 1 + static_cast<int>(variants.size());
  std::vector<std::vector<double>> us(arms);
  int counter = 0;
  auto run_arm = [&](int arm) {
    auto one = [&]() {
      const int set = counter % id_sets, c = counter % copies;
      ++counter;
      if (arm == 0) old_launch(set, c, d_x[0], o[0]);
      else new_launch(set, c, d_x[0], o[1], variants[arm - 1]);
    };
    for (int i = 0; i < warm; ++i) one();
    HC(hipEventRecord(ev0, st));
    for (int i = 0; i < reps; ++i) one();
    HC(hipEventRecord(ev1, st));
    HC(hipEventSynchronize(ev1));
    float ms = 0;
    HC(hipEventElapsedTime(&ms, ev0, ev1));
    us[arm].push_back(1000.0 * ms / reps);
  };
  for (int r = 0; r < rounds; ++r) {
    if (r % 2 == 0) {
      for (int a = 0; a < arms; ++a) run_arm(a);
    } else {
      for (int a = arms - 1; a >= 0; --a) run_arm(a);
    }
  }
  std::ostringstream js;
  auto arm_json = [&](int a) {
    const auto mm = std::minmax_element(us[a].begin(), us[a].end());
    const double m = Median(us[a]);
    std::ostringstream q;
    q << "\"us\":" << m << ",\"us_min\":" << *mm.first << ",\"us_max\":" << *mm.second << ",\"gbs\":" << bytes / (m * 1e3);
    return q.str();
  };
  js << "{\"event\":\"speed\",\"case\":\"" << name << "\",\"hidden\":" << H << ",\"ff\":" << F << ",\"used\":" << U
     << ",\"experts\":" << E << ",\"down\":\"" << (down_q6 ? "q6k" : "q4k") << "\",\"sdown\":\"" << (sdown_q6 ? "q6k" : "q4k")
     << "\",\"bytes\":" << bytes << ",\"reps\":" << reps << ",\"rounds\":" << rounds << ",\"old\":{" << arm_json(0) << "},\"new\":[";
  for (std::size_t vi = 0; vi < variants.size(); ++vi) {
    js << (vi ? "," : "") << "{\"variant\":" << variants[vi] << ",\"grid\":" << grids[vi] << "," << arm_json(1 + static_cast<int>(vi)) << "}";
  }
  js << "]}";
  std::printf("%s\n", js.str().c_str());
  return 0;
}

// ------------------------------------------------------------------------------------------------- item 2
// The exact head: Q8KBlockGEMVKernel_2Rows<4, kF32> (gufo gemv_quant.hip) on its QuantWarpBlockDot path (the type is
// not Q8_0 / Q8_K), copied; and gufo's ArgmaxKernel (sample.hip), copied.
__launch_bounds__(128, 4) __global__ void ExactHeadKernel(const void* __restrict__ A, GgmlType type, const float* __restrict__ x,
                                                          float* __restrict__ y, std::size_t M, std::size_t K) {
  const std::size_t lane_id = threadIdx.x & 31u;
  const std::size_t pair_idx = (blockIdx.x * 4) + (threadIdx.x >> 5u);
  const std::size_t m0 = pair_idx * 2;
  const std::size_t m1 = m0 + 1;
  if (m0 >= M) return;
  const float s0 = gufo::hip::QuantWarpBlockDot(type, A, m0, x, K, lane_id, 0, 1);
  const float s1 = (m1 < M) ? gufo::hip::QuantWarpBlockDot(type, A, m1, x, K, lane_id, 0, 1) : 0.0F;
  if (lane_id == 0) {
    y[m0] = s0;
    if (m1 < M) y[m1] = s1;
  }
}
__global__ void ArgmaxKernel(const float* __restrict__ logits, std::uint32_t* __restrict__ out_token, std::size_t vocab_size) {
  const std::size_t tid = threadIdx.x;
  const std::size_t block_size = blockDim.x;
  float max_val = -INFINITY;
  std::uint32_t max_idx = 0xFFFFFFFFU;
  for (std::size_t i = tid; i < vocab_size; i += block_size) {
    const float val = logits[i];
    if (isfinite(val) && val > max_val) {
      max_val = val;
      max_idx = static_cast<std::uint32_t>(i);
    }
  }
  __shared__ float s_max[256];
  __shared__ std::uint32_t s_idx[256];
  s_max[tid] = max_val;
  s_idx[tid] = max_idx;
  __syncthreads();
  for (std::size_t stride = block_size / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      if (s_max[tid + stride] > s_max[tid] || (s_max[tid + stride] == s_max[tid] && s_idx[tid + stride] < s_idx[tid])) {
        s_max[tid] = s_max[tid + stride];
        s_idx[tid] = s_idx[tid + stride];
      }
    }
    __syncthreads();
  }
  if (tid == 0) *out_token = s_idx[0];
}

int Head(int argc, char** argv) {
  int V = 248320, H = 4096, keep = 8, reps = 50, warm = 5, rounds = 7, xsets = 8;
  float k = 8.0F;
  bool q6 = true;
  std::string blist = "2,4", name = "head";
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto nx = [&]() { return std::string(argv[++i]); };
    if (s == "--vocab") V = std::stoi(nx());
    else if (s == "--hidden") H = std::stoi(nx());
    else if (s == "--keep") keep = std::stoi(nx());
    else if (s == "--k") k = std::stof(nx());
    else if (s == "--bits") blist = nx();
    else if (s == "--type") q6 = nx() == "q6k";
    else if (s == "--reps") reps = std::stoi(nx());
    else if (s == "--warm") warm = std::stoi(nx());
    else if (s == "--rounds") rounds = std::stoi(nx());
    else if (s == "--xsets") xsets = std::stoi(nx());
    else if (s == "--case") name = nx();
  }
  const GgmlType type = q6 ? GgmlType::kQ6_K : GgmlType::kQ4_K;
  std::uint8_t* head = Synth(q6, V, H, 31);
  HC(hipDeviceSynchronize());
  hipStream_t st;
  HC(hipStreamCreate(&st));
  float *d_exact = nullptr, *d_logits = nullptr, *d_x = nullptr;
  std::uint32_t* d_tok = nullptr;
  flerp::draft_coarse::Result* d_res = nullptr;
  HC(hipMalloc(&d_exact, static_cast<std::size_t>(V) * 4));
  HC(hipMalloc(&d_logits, static_cast<std::size_t>(V) * 4));
  HC(hipMalloc(&d_x, static_cast<std::size_t>(H) * 4 * xsets));
  HC(hipMalloc(&d_tok, 4));
  HC(hipMalloc(&d_res, sizeof(flerp::draft_coarse::Result)));
  std::vector<float> xh(static_cast<std::size_t>(H) * xsets);
  std::mt19937_64 xr(2024);
  std::normal_distribution<float> nd(0.F, 1.F);
  for (auto& v : xh) v = nd(xr);
  HC(hipMemcpy(d_x, xh.data(), xh.size() * 4, hipMemcpyHostToDevice));
  auto exact = [&](const float* x) {
    ExactHeadKernel<<<(V / 2 + 3) / 4, 128, 0, st>>>(head, type, x, d_exact, V, H);
    ArgmaxKernel<<<1, 256, 0, st>>>(d_exact, d_tok, V);
  };
  hipEvent_t ev0, ev1;
  HC(hipEventCreate(&ev0));
  HC(hipEventCreate(&ev1));
  const double head_bytes = static_cast<double>(V) * RowBytes(q6, H);
  for (const int bits : List(blist)) {
    std::string why;
    const auto t0 = std::chrono::steady_clock::now();
    auto c = flerp::draft_coarse::Coarse::Build(head, type, V, H, bits, keep, k, st, &why);
    const double build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!c) {
      std::printf("{\"event\":\"error\",\"case\":\"%s\",\"why\":\"%s\"}\n", name.c_str(), why.c_str());
      return 2;
    }
    // Correctness on xsets inputs: kept rows' bits, the choice against the exact argmax.
    std::size_t mism = 0, kept_rows = 0, wrong = 0, fallbacks = 0, miss1 = 0;
    for (int s = 0; s < xsets; ++s) {
      const float* x = d_x + static_cast<std::size_t>(s) * H;
      exact(x);
      c->Score(x, d_logits, d_res, st);
      HC(hipStreamSynchronize(st));
      const auto ex = D2H(d_exact, V), lg = D2H(d_logits, V);
      const auto kept = D2H(c->chosen(), keep);
      const auto res = D2H(d_res, 1)[0];
      const auto tok = D2H(d_tok, 1)[0];
      bool in1 = false;
      for (int b : kept) {
        for (int t = 0; b >= 0 && t < 128; ++t) {
          const std::size_t i = static_cast<std::size_t>(b) * 128 + t;
          if (i >= static_cast<std::size_t>(V)) break;
          ++kept_rows;
          mism += std::memcmp(&ex[i], &lg[i], 4) != 0;
          in1 = in1 || i == tok;
        }
      }
      miss1 += !in1;
      fallbacks += res.fallback;
      wrong += res.fallback == 0 && res.id != tok;
    }
    std::printf("{\"event\":\"head_check\",\"case\":\"%s\",\"vocab\":%d,\"hidden\":%d,\"bits\":%d,\"keep\":%d,\"k\":%g,"
                "\"err_rms\":%g,\"coarse_bytes\":%zu,\"build_s\":%.4f,\"xsets\":%d,\"kept_rows\":%zu,\"rescore_mismatch\":%zu,"
                "\"wrong_ids_without_fallback\":%zu,\"fallbacks\":%zu,\"miss_top1\":%zu,\"note\":\"synthetic random head: misses and fallbacks here say nothing about a real head\"}\n",
                name.c_str(), V, H, bits, keep, k, c->err_rms(), c->bytes(), build_s, xsets, kept_rows, mism, wrong, fallbacks, miss1);
    std::fflush(stdout);
    // Timing: a step = old (exact GEMV + argmax) or new (Score); each pass of Score alone too. Rounds alternate.
    std::vector<double> t_old, t_new, t_c, t_s, t_r, t_ch;
    auto time_it = [&](auto&& body, std::vector<double>& out) {
      for (int i = 0; i < warm; ++i) body(i);
      HC(hipEventRecord(ev0, st));
      for (int i = 0; i < reps; ++i) body(i);
      HC(hipEventRecord(ev1, st));
      HC(hipEventSynchronize(ev1));
      float ms = 0;
      HC(hipEventElapsedTime(&ms, ev0, ev1));
      out.push_back(1000.0 * ms / reps);
    };
    auto xs = [&](int i) { return d_x + static_cast<std::size_t>(i % xsets) * H; };
    auto old_step = [&](int i) { exact(xs(i)); };
    auto new_step = [&](int i) { c->Score(xs(i), d_logits, d_res, st); };
    for (int r = 0; r < rounds; ++r) {
      if (r % 2 == 0) {
        time_it(old_step, t_old);
        time_it(new_step, t_new);
      } else {
        time_it(new_step, t_new);
        time_it(old_step, t_old);
      }
      time_it([&](int i) { c->CoarsePass(xs(i), d_logits, st); }, t_c);
      time_it([&](int) { c->SelectPass(st); }, t_s);
      time_it([&](int i) { c->RescorePass(xs(i), d_logits, st); }, t_r);
      time_it([&](int i) { c->ChoosePass(xs(i), d_res, st); }, t_ch);
    }
    const double mo = Median(t_old), mn = Median(t_new), mc = Median(t_c);
    std::printf("{\"event\":\"head_speed\",\"case\":\"%s\",\"vocab\":%d,\"hidden\":%d,\"bits\":%d,\"keep\":%d,\"reps\":%d,\"rounds\":%d,"
                "\"old_us\":%.2f,\"old_us_min\":%.2f,\"old_us_max\":%.2f,\"old_gbs\":%.1f,\"new_us\":%.2f,\"new_us_min\":%.2f,\"new_us_max\":%.2f,"
                "\"coarse_us\":%.2f,\"coarse_gbs\":%.1f,\"select_us\":%.2f,\"rescore_us\":%.2f,\"choose_us\":%.2f,\"saved_us\":%.2f}\n",
                name.c_str(), V, H, bits, keep, reps, rounds, mo, *std::min_element(t_old.begin(), t_old.end()),
                *std::max_element(t_old.begin(), t_old.end()), head_bytes / (mo * 1e3), mn,
                *std::min_element(t_new.begin(), t_new.end()), *std::max_element(t_new.begin(), t_new.end()), mc,
                static_cast<double>(c->bytes()) / (mc * 1e3), Median(t_s), Median(t_r), Median(t_ch), mo - mn);
    std::fflush(stdout);
  }
  return 0;
}
}  // namespace ocb

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--experts") return ocb::Experts(argc, argv);
    if (std::string(argv[i]) == "--head") return ocb::Head(argc, argv);
  }
  std::fprintf(stderr, "usage: ornith-crucible-bench --experts ... | --head ...\n");
  return 2;
}
