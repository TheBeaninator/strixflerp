// SPDX-License-Identifier: MIT
// SHAPE-TRANSFERS R1 fold (criteria gate/xfer-r1-fold.toml [kernel]): flerp::moe::IndexedExperts of the catalogue's
// kernel.flerp.moe_kquant (the new source, compiled here) against flerp_ref::moe::IndexedExperts (the source as at
// 504ca20, gate/xfer_r1_fold_ref.hip), same weights, same ids, same x.
//   xfer-r1-fold --case NAME --type q4k|q6k --rows N --k K --experts E --slots S [--xshare 1|S] [--pair] [--shared]
//                [--gguf FILE --off-a BYTES [--off-b BYTES]] [--tokens-equal 1,2,..] [--tokens-timing 1,2,..]
//                [--iters 200] [--warm 20] [--rounds 1]
// --shared: the shared expert's call (ids null, one expert, one slot). One JSON line per tokens value on stdout.
#include "flerp/moe_kquant/moe_kquant.hip"
#include "xfer_r1_fold_ref.hip"

#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace xf {
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
}  // namespace xf

int main(int argc, char** argv) {
  using namespace xf;
  std::string name = "case", gguf, type_s = "q4k", teq = "1,2,3,4,8,21,128,1024", ttim = "1,2,4,8";
  std::size_t rows = 512, k = 2048, n_exp = 256, slots = 8, xshare = 0, off_a = 0, off_b = 0;
  int iters = 200, warm = 20, rounds = 1;
  bool pair = false, shared = false;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto nx = [&]() { return std::string(argv[++i]); };
    if (s == "--case") name = nx();
    else if (s == "--type") type_s = nx();
    else if (s == "--rows") rows = std::stoull(nx());
    else if (s == "--k") k = std::stoull(nx());
    else if (s == "--experts") n_exp = std::stoull(nx());
    else if (s == "--slots") slots = std::stoull(nx());
    else if (s == "--xshare") xshare = std::stoull(nx());
    else if (s == "--pair") pair = true;
    else if (s == "--shared") shared = true;
    else if (s == "--gguf") gguf = nx();
    else if (s == "--off-a") off_a = std::stoull(nx());
    else if (s == "--off-b") off_b = std::stoull(nx());
    else if (s == "--tokens-equal") teq = nx();
    else if (s == "--tokens-timing") ttim = nx();
    else if (s == "--iters") iters = std::stoi(nx());
    else if (s == "--warm") warm = std::stoi(nx());
    else if (s == "--rounds") rounds = std::stoi(nx());
  }
  if (shared) n_exp = 1, slots = 1, xshare = 1;
  if (xshare == 0) xshare = slots;
  const GgmlType type = type_s == "q6k" ? GgmlType::kQ6_K : GgmlType::kQ4_K;
  const std::size_t blk = type == GgmlType::kQ6_K ? 210 : 144;
  const std::size_t row_bytes = k / 256 * blk, ebytes = rows * row_bytes, tbytes = ebytes * n_exp;
  if (ebytes != flerp::moe::ExpertBytes(type, rows, k)) return std::fprintf(stderr, "expert bytes\n"), 1;
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
    } else {  // synthetic: random bytes in the real block layout, fp16 d (and dmin) normal
      std::uniform_real_distribution<float> ud(0.002F, 0.02F);
      for (std::size_t i = 0; i < tbytes; i += 8) {
        const std::uint64_t v = rng();
        std::memcpy(host[w].data() + i, &v, std::min<std::size_t>(8, tbytes - i));
      }
      for (std::size_t b = 0; b < tbytes / blk; ++b) {
        const std::uint16_t d = F2H(ud(rng)), dm = F2H(ud(rng));
        if (type == GgmlType::kQ6_K) {
          std::memcpy(host[w].data() + b * blk + 208, &d, 2);
        } else {
          std::memcpy(host[w].data() + b * blk, &d, 2), std::memcpy(host[w].data() + b * blk + 2, &dm, 2);
        }
      }
    }
  }
  std::uint8_t* dw[2] = {nullptr, nullptr};
  for (int w = 0; w < nmat; ++w) {
    HC(hipMalloc(&dw[w], tbytes));
    HC(hipMemcpy(dw[w], host[w].data(), tbytes, hipMemcpyHostToDevice));
  }
  hipStream_t st;
  HC(hipStreamCreate(&st));
  hipEvent_t e0, e1;
  HC(hipEventCreate(&e0));
  HC(hipEventCreate(&e1));
  const std::vector<std::size_t> teq_v = List(teq), ttim_v = List(ttim);
  std::set<std::size_t> all(teq_v.begin(), teq_v.end());
  all.insert(ttim_v.begin(), ttim_v.end());
  for (const std::size_t tokens : all) {
    const bool timed = std::find(ttim_v.begin(), ttim_v.end(), tokens) != ttim_v.end();
    const int nit = timed ? warm + iters : 1;
    const std::size_t ts_n = tokens * slots, xrows = ts_n / xshare;
    std::vector<float> xh(xrows * k);
    std::uniform_real_distribution<float> ux(-1.F, 1.F);
    for (auto& v : xh) v = ux(rng);
    std::vector<std::int32_t> idh(static_cast<std::size_t>(nit) * ts_n, 0);
    double distinct_sum = 0;
    for (int it = 0; it < nit; ++it) {
      std::set<int> launch_used;
      for (std::size_t t = 0; t < tokens; ++t) {
        std::set<int> used;
        for (std::size_t s = 0; s < slots; ++s) {
          int e = 0;
          if (!shared) {
            do e = static_cast<int>(rng() % n_exp); while (used.count(e));
          }
          used.insert(e), launch_used.insert(e);
          idh[it * ts_n + t * slots + s] = e;
        }
      }
      if (it >= nit - iters || !timed) distinct_sum += static_cast<double>(launch_used.size());
    }
    const double distinct = distinct_sum / (timed ? iters : 1);
    float *dx, *dy[2];
    std::int32_t* did;
    HC(hipMalloc(&dx, xh.size() * 4));
    HC(hipMemcpy(dx, xh.data(), xh.size() * 4, hipMemcpyHostToDevice));
    HC(hipMalloc(&did, idh.size() * 4));
    HC(hipMemcpy(did, idh.data(), idh.size() * 4, hipMemcpyHostToDevice));
    for (auto& p : dy) {
      HC(hipMalloc(&p, ts_n * rows * 4));
      HC(hipMemset(p, 0xA5, ts_n * rows * 4));
    }
    auto launch = [&](int arm, int it, float* y) {
      const std::int32_t* ids = shared ? nullptr : did + static_cast<std::size_t>(it) * ts_n;
      if (arm == 0)
        flerp_ref::moe::IndexedExperts(dw[0], type, dw[1], type, ids, dx, y, tokens, slots, xshare, rows, k, n_exp, st);
      else
        flerp::moe::IndexedExperts(dw[0], type, dw[1], type, ids, dx, y, tokens, slots, xshare, rows, k, n_exp, st);
    };
    std::vector<std::vector<float>> out(2, std::vector<float>(ts_n * rows));
    for (int arm = 0; arm < 2; ++arm) {
      launch(arm, 0, dy[arm]);
      HC(hipStreamSynchronize(st));
      HC(hipMemcpy(out[arm].data(), dy[arm], out[arm].size() * 4, hipMemcpyDeviceToHost));
    }
    std::size_t ndiff = 0, nonfinite = 0;
    double rms = 0;
    for (std::size_t i = 0; i < out[0].size(); ++i) {
      ndiff += std::memcmp(&out[0][i], &out[1][i], 4) != 0;
      nonfinite += !std::isfinite(out[0][i]);
      rms += static_cast<double>(out[0][i]) * out[0][i];
    }
    rms = std::sqrt(rms / out[0].size());
    std::string tim = "null";
    if (timed) {
      const double bytes = distinct * static_cast<double>(ebytes) * nmat;
      // `rounds` rounds, the arms' order alternating (ref first, then new first, ...); each a warm-up and a timed
      // loop of `iters` launches. The median round of each arm is the number; min and max are the spread.
      std::vector<double> us[2];
      for (int r = 0; r < rounds; ++r) {
        for (int j = 0; j < 2; ++j) {
          const int arm = (r % 2 == 0) ? j : 1 - j;
          for (int it = 0; it < warm; ++it) launch(arm, it, dy[arm]);
          HC(hipEventRecord(e0, st));
          for (int it = warm; it < warm + iters; ++it) launch(arm, it, dy[arm]);
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
    std::printf("{\"case\":\"%s\",\"type\":\"%s\",\"rows\":%zu,\"k\":%zu,\"experts\":%zu,\"slots\":%zu,\"x_share\":%zu,"
                "\"pair\":%s,\"shared\":%s,\"real_weights\":%s,\"tokens\":%zu,\"outputs\":%zu,\"outputs_differing\":%zu,"
                "\"ref_nonfinite\":%zu,\"ref_rms\":%.4e,\"distinct_experts_per_launch\":%.2f,\"timing\":%s}\n",
                name.c_str(), type_s.c_str(), rows, k, n_exp, slots, xshare, pair ? "true" : "false",
                shared ? "true" : "false", gguf.empty() ? "false" : "true", tokens, out[0].size(), ndiff, nonfinite, rms,
                distinct, tim.c_str());
    std::fflush(stdout);
    HC(hipFree(dx));
    HC(hipFree(did));
    for (auto& p : dy) HC(hipFree(p));
  }
  return 0;
}
