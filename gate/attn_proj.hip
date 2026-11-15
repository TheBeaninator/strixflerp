// SPDX-License-Identifier: MIT
// ATTN-PROJ (criteria gate/attn-proj.toml [kernel]): the catalogue's new gufo::hip::LaunchFusedQKVProjections
// (kernel.gufo.qwen_linear_decode qkv.hip) and LaunchFusedSSMInputProjections (kernel.gufo.qwen_gdn ssm.hip), compiled
// here, against the old ones (gate/attn_proj_ref.hip, namespace attn_proj_ref) and the verify path
// (gufo::hip::LaunchQ8KBlockGEMV a matrix a row): same weights, same x, byte equality and cache-cold GB/s.
//   attn-proj --case NAME --kind attn --k K --q ROWS --qt T --kv ROWS --kt T --vt T
//   attn-proj --case NAME --kind gdn  --k K --qkv ROWS --qkvt T --gate ROWS --gt T --rank ROWS --rt T
//   real weights: --gguf F --off <segment>=<offset> (segments q,k,v or qkv,gate,alpha,beta), types as in the file
//   common: [--tokens 1,2,3,4,8,16] [--iters 200] [--warm 20] [--rounds 7] [--no-timing] [--cold-mib 0]
// T is q4k or q6k. Byte equality for two x (uniform(-1, 1), normal * 0.05; fixed seeds) at every --tokens count: T
// rows of x, every arm launched once a row. Timing (1 token, uniform x): rounds alternating old,new / new,old,
// cache-cold with --cold-mib (launches cycle over copies of all the launch's weights). One JSON line a (x, tokens).
#include "src/families/qwen/kernels/gemv_quant.hip"
#include "src/families/qwen/kernels/qkv.hip"
#include "src/families/qwen/kernels/ssm.hip"
#include "attn_proj_ref.hip"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace ap {
using gufo::core::GgmlType;

#define HC(x)                                                                         \
  do {                                                                                \
    hipError_t e_ = (x);                                                              \
    if (e_ != hipSuccess) {                                                           \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
      std::exit(1);                                                                   \
    }                                                                                 \
  } while (0)

std::uint16_t F2H(float v) {  // normal range only
  int ex;
  const float mnt = std::frexp(std::fabs(v), &ex);
  const auto f = static_cast<std::uint32_t>(std::lround((mnt * 2.0F - 1.0F) * 1024.0F));
  return static_cast<std::uint16_t>(((v < 0) << 15) | ((ex - 1 + 15) << 10) | (f & 1023));
}

std::vector<std::size_t> List(const std::string& s) {
  std::vector<std::size_t> v;
  std::stringstream ss(s);
  std::string it;
  while (std::getline(ss, it, ',')) {
    if (!it.empty()) v.push_back(std::stoull(it));
  }
  return v;
}

void ReadAt(const std::string& path, std::size_t off, std::uint8_t* dst, std::size_t n) {
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) std::perror("open"), std::exit(1);
  std::size_t done = 0;
  while (done < n) {
    const ssize_t r = pread(fd, dst + done, n - done, static_cast<off_t>(off + done));
    if (r <= 0) std::perror("pread"), std::exit(1);
    done += static_cast<std::size_t>(r);
  }
  close(fd);
}

struct Seg {
  std::string name;
  GgmlType type = GgmlType::kQ4_K;
  std::size_t rows = 0;
  std::size_t bytes = 0;
  std::size_t off = 0;
  bool have_off = false;
};

GgmlType Type(const std::string& s) {
  if (s == "q4k") return GgmlType::kQ4_K;
  if (s == "q6k") return GgmlType::kQ6_K;
  std::fprintf(stderr, "type %s: q4k or q6k\n", s.c_str());
  std::exit(2);
}
const char* TypeName(GgmlType t) { return t == GgmlType::kQ6_K ? "q6k" : "q4k"; }
std::size_t BlockBytes(GgmlType t) { return t == GgmlType::kQ6_K ? 210 : 144; }
constexpr std::size_t kGuard = 64;  // floats after every output, must stay 0xA5A5A5A5
}  // namespace ap

int main(int argc, char** argv) {
  using namespace ap;
  std::string name = "case", kind = "attn", gguf, toks = "1,2,3,4,8,16";
  std::size_t k = 4096;
  int iters = 200, warm = 20, rounds = 7;
  bool timing = true;
  std::size_t cold_mib = 0;
  std::map<std::string, std::string> a;
  std::map<std::string, std::size_t> offs;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto nx = [&]() { return std::string(argv[++i]); };
    if (s == "--case") name = nx();
    else if (s == "--kind") kind = nx();
    else if (s == "--k") k = std::stoull(nx());
    else if (s == "--gguf") gguf = nx();
    else if (s == "--tokens") toks = nx();
    else if (s == "--iters") iters = std::stoi(nx());
    else if (s == "--warm") warm = std::stoi(nx());
    else if (s == "--rounds") rounds = std::stoi(nx());
    else if (s == "--no-timing") timing = false;
    else if (s == "--cold-mib") cold_mib = std::stoull(nx());
    else if (s == "--off") {
      const std::string v = nx();
      const auto eq = v.find('=');
      offs[v.substr(0, eq)] = std::stoull(v.substr(eq + 1));
    } else if (s.rfind("--", 0) == 0 && i + 1 < argc) a[s.substr(2)] = nx();
  }
  auto num = [&](const char* key) { return a.count(key) ? std::stoull(a[key]) : 0ULL; };
  auto typ = [&](const char* key) { return Type(a.count(key) ? a[key] : "q4k"); };
  const bool attn = kind == "attn";
  std::vector<Seg> seg;
  if (attn) {
    seg = {{"q", typ("qt"), num("q")}, {"k", typ("kt"), num("kv")}, {"v", typ("vt"), num("kv")}};
  } else {
    seg = {{"qkv", typ("qkvt"), num("qkv")}, {"gate", typ("gt"), num("gate")},
           {"alpha", typ("rt"), num("rank")}, {"beta", typ("rt"), num("rank")}};
  }
  std::size_t wbytes = 0;
  for (auto& s : seg) {
    s.bytes = s.rows * (k / 256) * BlockBytes(s.type);
    wbytes += s.bytes;
    if (offs.count(s.name)) s.off = offs[s.name], s.have_off = true;
  }
  // Weights: real rows from the GGUF, or random codes in the real block layout with fp16 d (and dmin) normal.
  std::mt19937_64 rng(4243);
  std::vector<std::uint8_t*> dw(seg.size(), nullptr);
  for (std::size_t i = 0; i < seg.size(); ++i) {
    const Seg& s = seg[i];
    if (s.rows == 0) continue;
    std::vector<std::uint8_t> h(s.bytes);
    if (!gguf.empty()) {
      if (!s.have_off) std::fprintf(stderr, "segment %s: no --off\n", s.name.c_str()), std::exit(2);
      ReadAt(gguf, s.off, h.data(), s.bytes);
    } else {
      const std::size_t blk = BlockBytes(s.type);
      std::uniform_real_distribution<float> ud(0.002F, 0.02F);
      for (std::size_t j = 0; j < s.bytes; j += 8) {
        const std::uint64_t v = rng();
        std::memcpy(h.data() + j, &v, std::min<std::size_t>(8, s.bytes - j));
      }
      for (std::size_t b = 0; b < s.bytes / blk; ++b) {
        const std::uint16_t d = F2H(ud(rng)), dm = F2H(ud(rng));
        if (s.type == GgmlType::kQ6_K) std::memcpy(h.data() + b * blk + 208, &d, 2);
        else std::memcpy(h.data() + b * blk, &d, 2), std::memcpy(h.data() + b * blk + 2, &dm, 2);
      }
    }
    HC(hipMalloc(&dw[i], s.bytes));
    HC(hipMemcpy(dw[i], h.data(), s.bytes, hipMemcpyHostToDevice));
  }
  // Copies for cold timing (copy 0 is dw itself): copies of the whole launch's weights totalling >= cold_mib.
  const std::size_t ncopy = std::max<std::size_t>(1, ((cold_mib << 20) + wbytes - 1) / wbytes);
  std::vector<std::vector<std::uint8_t*>> cw(ncopy, dw);
  for (std::size_t c = 1; c < ncopy; ++c)
    for (std::size_t i = 0; i < seg.size(); ++i) {
      if (seg[i].rows == 0) continue;
      HC(hipMalloc(&cw[c][i], seg[i].bytes));
      HC(hipMemcpy(cw[c][i], dw[i], seg[i].bytes, hipMemcpyDeviceToDevice));
    }
  hipStream_t st;
  HC(hipStreamCreate(&st));
  hipEvent_t e0, e1;
  HC(hipEventCreate(&e0));
  HC(hipEventCreate(&e1));
  std::size_t wide_segments = 0;  // the launcher's own predicate, for the record
  for (const auto& s : seg)
    wide_segments += s.rows != 0 && s.type == GgmlType::kQ4_K && k % 256 == 0;
  std::size_t rows_total = 0;
  for (const auto& s : seg) rows_total += s.rows;
  for (int xkind = 0; xkind < 2; ++xkind) {
    for (const std::size_t tokens : List(toks)) {
      std::vector<float> xh(tokens * k);
      std::mt19937_64 xr(2000 + xkind);
      std::uniform_real_distribution<float> ux(-1.F, 1.F);
      std::normal_distribution<float> nx(0.F, 0.05F);
      for (auto& v : xh) v = xkind == 0 ? ux(xr) : nx(xr);
      float* dx;
      HC(hipMalloc(&dx, xh.size() * 4));
      HC(hipMemcpy(dx, xh.data(), xh.size() * 4, hipMemcpyHostToDevice));
      // arm 0 old, 1 new, 2 verify path; one output buffer a segment, tokens * rows + guard.
      float* dy[3][4] = {};
      for (int arm = 0; arm < 3; ++arm)
        for (std::size_t i = 0; i < seg.size(); ++i) {
          const std::size_t n = tokens * seg[i].rows + kGuard;
          HC(hipMalloc(&dy[arm][i], n * 4));
          HC(hipMemset(dy[arm][i], 0xA5, n * 4));
        }
      auto launch = [&](int arm, std::size_t t, std::size_t c) {
        const float* x = dx + t * k;
        const auto& w = cw[c];
        float* o[4] = {};
        for (std::size_t i = 0; i < seg.size(); ++i) o[i] = dy[arm][i] + t * seg[i].rows;
        if (arm == 2) {
          for (std::size_t i = 0; i < seg.size(); ++i)
            if (seg[i].rows) gufo::hip::LaunchQ8KBlockGEMV(w[i], seg[i].type, x, o[i], seg[i].rows, k, st);
        } else if (attn) {
          auto f = arm == 0 ? attn_proj_ref::LaunchFusedQKVProjections : gufo::hip::LaunchFusedQKVProjections;
          f(w[0], seg[0].type, w[1], seg[1].type, w[2], seg[2].type, x, o[0], o[1], o[2], seg[0].rows, seg[1].rows, k, st);
        } else {
          auto f = arm == 0 ? attn_proj_ref::LaunchFusedSSMInputProjections : gufo::hip::LaunchFusedSSMInputProjections;
          f(w[0], seg[0].type, w[1], seg[1].type, w[2], seg[2].type, w[3], seg[3].type, x, o[0], o[1], o[2], o[3], k,
            seg[0].rows, seg[1].rows, seg[2].rows, st);
        }
      };
      for (int arm = 0; arm < 3; ++arm) {
        for (std::size_t t = 0; t < tokens; ++t) launch(arm, t, 0);
        HC(hipStreamSynchronize(st));
        HC(hipGetLastError());
      }
      std::size_t nd_new = 0, nd_verify = 0, guard = 0, nonfinite = 0, checked = 0;
      double rms = 0;
      for (std::size_t i = 0; i < seg.size(); ++i) {
        const std::size_t n = tokens * seg[i].rows;
        std::vector<std::uint32_t> h[3];
        for (int arm = 0; arm < 3; ++arm) {
          h[arm].resize(n + kGuard);
          HC(hipMemcpy(h[arm].data(), dy[arm][i], (n + kGuard) * 4, hipMemcpyDeviceToHost));
        }
        for (std::size_t j = 0; j < n; ++j) {
          nd_new += h[1][j] != h[0][j];
          nd_verify += h[2][j] != h[0][j];
          float f;
          std::memcpy(&f, &h[0][j], 4);
          nonfinite += !std::isfinite(f);
          rms += static_cast<double>(f) * f;
          ++checked;
        }
        for (int arm = 0; arm < 3; ++arm)
          for (std::size_t j = n; j < n + kGuard; ++j) guard += h[arm][j] != 0xA5A5A5A5U;
      }
      rms = std::sqrt(rms / static_cast<double>(std::max<std::size_t>(1, checked)));
      std::string tim = "null";
      if (timing && tokens == 1 && xkind == 0) {
        const double bytes = static_cast<double>(wbytes) + static_cast<double>(k) * 4.0;
        std::vector<double> us[2];
        for (int r = 0; r < rounds; ++r) {
          for (int j = 0; j < 2; ++j) {
            const int arm = (r % 2 == 0) ? j : 1 - j;
            for (int it = 0; it < warm; ++it) launch(arm, 0, static_cast<std::size_t>(it) % ncopy);
            HC(hipEventRecord(e0, st));
            for (int it = 0; it < iters; ++it) launch(arm, 0, static_cast<std::size_t>(warm + it) % ncopy);
            HC(hipEventRecord(e1, st));
            HC(hipEventSynchronize(e1));
            float ms = 0;
            HC(hipEventElapsedTime(&ms, e0, e1));
            us[arm].push_back(1e3 * ms / iters);
          }
        }
        double med[2], lo[2], hi[2];
        for (int arm = 0; arm < 2; ++arm) {
          std::vector<double> v = us[arm];
          std::sort(v.begin(), v.end());
          med[arm] = v[v.size() / 2], lo[arm] = v.front(), hi[arm] = v.back();
        }
        char b[512];
        std::snprintf(b, sizeof b,
                      "{\"rounds\":%d,\"ref\":{\"us\":%.2f,\"GBps\":%.1f,\"GBps_min\":%.1f,\"GBps_max\":%.1f},"
                      "\"new\":{\"us\":%.2f,\"GBps\":%.1f,\"GBps_min\":%.1f,\"GBps_max\":%.1f},\"speedup\":%.3f}",
                      rounds, med[0], bytes / (med[0] * 1e3), bytes / (hi[0] * 1e3), bytes / (lo[0] * 1e3), med[1],
                      bytes / (med[1] * 1e3), bytes / (hi[1] * 1e3), bytes / (lo[1] * 1e3), med[0] / med[1]);
        tim = b;
      }
      std::string segs;
      for (const auto& s : seg) {
        if (!segs.empty()) segs += ",";
        segs += "\"" + s.name + "\":\"" + std::to_string(s.rows) + " " + TypeName(s.type) + "\"";
      }
      std::printf("{\"case\":\"%s\",\"kind\":\"%s\",\"k\":%zu,\"segments\":{%s},\"rows\":%zu,\"weight_bytes\":%zu,"
                  "\"wide_segments\":%zu,\"real_weights\":%s,\"x\":\"%s\",\"tokens\":%zu,\"outputs\":%zu,"
                  "\"outputs_differing\":%zu,\"verify_differing\":%zu,\"guard_touched\":%zu,\"ref_nonfinite\":%zu,"
                  "\"ref_rms\":%.4e,\"weight_copies\":%zu,\"timing\":%s}\n",
                  name.c_str(), kind.c_str(), k, segs.c_str(), rows_total, wbytes, wide_segments,
                  gguf.empty() ? "false" : "true", xkind == 0 ? "uniform" : "normal0.05", tokens, checked, nd_new,
                  nd_verify, guard, nonfinite, rms, ncopy, tim.c_str());
      std::fflush(stdout);
      HC(hipFree(dx));
      for (auto& arm : dy)
        for (auto* p : arm)
          if (p) HC(hipFree(p));
    }
  }
  return 0;
}
