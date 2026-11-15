// SPDX-License-Identifier: MIT
// DECODE-KERN3 (criteria gate/decode-kern3.toml [kernel]): the catalogue's new flerp::moe::GateLogits
// (kernel.flerp.moe_kquant) and gufo::hip::LaunchQ8KBlockGEMV (kernel.gufo.qwen_linear_decode), compiled here, against
// the old sources (gate/decode_kern3_ref.hip, namespace kern3_ref): same weights, same x, byte equality and GB/s.
//   decode-kern3 --case NAME --kind router --hidden H --experts E [--no-shared] [--gguf F --off-a W --off-b SHEXP]
//                [--tokens-equal 1,2,..]
//   decode-kern3 --case NAME --kind gemv --type q4k|q6k --rows M --k K [--gguf F --off-a W]
//   common: [--iters 200] [--warm 20] [--rounds 7] [--no-timing] [--cold-mib 0]
// --cold-mib N (amendment [kernel_cold]): timing launches cycle over copies of the weights totalling >= N MiB, so
// each launch reads its weights from DRAM, not from the L2 / 32 MB MALL (the in-model condition).
// Byte equality for two x: uniform(-1, 1) and normal * 0.05 (fixed seeds); the router at every --tokens-equal
// count. Timing (router 1 token, gemv one vector): rounds alternating ref,new / new,ref, median round a arm.
// One JSON line a (x, tokens) on stdout.
#include "flerp/moe_kquant/moe_kquant.hip"
#include "src/families/qwen/kernels/gemv_quant.hip"
#include "decode_kern3_ref.hip"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace k3 {
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
}  // namespace k3

int main(int argc, char** argv) {
  using namespace k3;
  std::string name = "case", kind = "gemv", gguf, type_s = "q4k", teq = "1";
  std::size_t rows = 2048, k = 4096, hidden = 2048, n_exp = 256, off_a = 0, off_b = 0;
  int iters = 200, warm = 20, rounds = 7;
  bool shared = true, timing = true;
  std::size_t cold_mib = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto nx = [&]() { return std::string(argv[++i]); };
    if (s == "--case") name = nx();
    else if (s == "--kind") kind = nx();
    else if (s == "--type") type_s = nx();
    else if (s == "--rows") rows = std::stoull(nx());
    else if (s == "--k") k = std::stoull(nx());
    else if (s == "--hidden") hidden = std::stoull(nx());
    else if (s == "--experts") n_exp = std::stoull(nx());
    else if (s == "--no-shared") shared = false;
    else if (s == "--gguf") gguf = nx();
    else if (s == "--off-a") off_a = std::stoull(nx());
    else if (s == "--off-b") off_b = std::stoull(nx());
    else if (s == "--tokens-equal") teq = nx();
    else if (s == "--iters") iters = std::stoi(nx());
    else if (s == "--warm") warm = std::stoi(nx());
    else if (s == "--rounds") rounds = std::stoi(nx());
    else if (s == "--no-timing") timing = false;
    else if (s == "--cold-mib") cold_mib = std::stoull(nx());
  }
  const bool router = kind == "router";
  std::mt19937_64 rng(4242);
  // Weights.
  std::size_t wbytes = 0, sbytes = 0, in = 0, outs = 0, stride = 0;
  GgmlType type = GgmlType::kF32;
  if (router) {
    wbytes = n_exp * hidden * 4, sbytes = shared ? hidden * 4 : 0, in = hidden;
    outs = n_exp + (shared ? 1 : 0), stride = n_exp + 1;
  } else {
    type = type_s == "q6k" ? GgmlType::kQ6_K : GgmlType::kQ4_K;
    const std::size_t blk = type == GgmlType::kQ6_K ? 210 : 144;
    wbytes = rows * (k / 256) * blk, in = k, outs = rows, stride = rows;
  }
  std::vector<std::uint8_t> hw(wbytes), hs(sbytes);
  if (!gguf.empty()) {
    ReadAt(gguf, off_a, hw.data(), wbytes);
    if (sbytes) ReadAt(gguf, off_b, hs.data(), sbytes);
  } else if (router) {
    std::normal_distribution<float> nd(0.F, 0.02F);
    for (std::size_t i = 0; i < wbytes / 4; ++i) { const float v = nd(rng); std::memcpy(hw.data() + 4 * i, &v, 4); }
    for (std::size_t i = 0; i < sbytes / 4; ++i) { const float v = nd(rng); std::memcpy(hs.data() + 4 * i, &v, 4); }
  } else {  // random codes in the real block layout, fp16 d (and dmin) normal
    const std::size_t blk = type == GgmlType::kQ6_K ? 210 : 144;
    std::uniform_real_distribution<float> ud(0.002F, 0.02F);
    for (std::size_t i = 0; i < wbytes; i += 8) {
      const std::uint64_t v = rng();
      std::memcpy(hw.data() + i, &v, std::min<std::size_t>(8, wbytes - i));
    }
    for (std::size_t b = 0; b < wbytes / blk; ++b) {
      const std::uint16_t d = F2H(ud(rng)), dm = F2H(ud(rng));
      if (type == GgmlType::kQ6_K) std::memcpy(hw.data() + b * blk + 208, &d, 2);
      else std::memcpy(hw.data() + b * blk, &d, 2), std::memcpy(hw.data() + b * blk + 2, &dm, 2);
    }
  }
  std::uint8_t *dw = nullptr, *ds = nullptr;
  HC(hipMalloc(&dw, wbytes));
  HC(hipMemcpy(dw, hw.data(), wbytes, hipMemcpyHostToDevice));
  if (sbytes) {
    HC(hipMalloc(&ds, sbytes));
    HC(hipMemcpy(ds, hs.data(), sbytes, hipMemcpyHostToDevice));
  }
  // Copies for cold timing (copy 0 is dw/ds itself).
  const std::size_t ncopy = std::max<std::size_t>(1, (cold_mib << 20) / (wbytes + sbytes) + ((cold_mib << 20) % (wbytes + sbytes) != 0));
  std::vector<std::uint8_t*> cw(ncopy, dw), cs(ncopy, ds);
  for (std::size_t c = 1; c < ncopy; ++c) {
    HC(hipMalloc(&cw[c], wbytes));
    HC(hipMemcpy(cw[c], dw, wbytes, hipMemcpyDeviceToDevice));
    if (sbytes) {
      HC(hipMalloc(&cs[c], sbytes));
      HC(hipMemcpy(cs[c], ds, sbytes, hipMemcpyDeviceToDevice));
    }
  }
  hipStream_t st;
  HC(hipStreamCreate(&st));
  hipEvent_t e0, e1;
  HC(hipEventCreate(&e0));
  HC(hipEventCreate(&e1));
  const std::vector<std::size_t> tlist = router ? List(teq) : std::vector<std::size_t>{1};
  for (int xkind = 0; xkind < 2; ++xkind) {
    for (const std::size_t tokens : tlist) {
      std::vector<float> xh(tokens * in);
      std::mt19937_64 xr(1000 + xkind);
      std::uniform_real_distribution<float> ux(-1.F, 1.F);
      std::normal_distribution<float> nx(0.F, 0.05F);
      for (auto& v : xh) v = xkind == 0 ? ux(xr) : nx(xr);
      float* dx;
      float* dy[2];
      HC(hipMalloc(&dx, xh.size() * 4));
      HC(hipMemcpy(dx, xh.data(), xh.size() * 4, hipMemcpyHostToDevice));
      const std::size_t ny = tokens * stride;
      for (auto& p : dy) {
        HC(hipMalloc(&p, ny * 4));
        HC(hipMemset(p, 0xA5, ny * 4));
      }
      auto launch = [&](int arm, float* y, std::size_t c = 0) {
        std::uint8_t* const dw = cw[c];
        if (router) {
          const float* w = reinterpret_cast<const float*>(dw);
          const float* s = reinterpret_cast<const float*>(cs[c]);
          if (arm == 0) kern3_ref::moe::GateLogits(w, s, dx, y, tokens, n_exp, hidden, stride, st);
          else flerp::moe::GateLogits(w, s, dx, y, tokens, n_exp, hidden, stride, st);
        } else {
          if (arm == 0) kern3_ref::hip::LaunchQ8KBlockGEMV(dw, type, dx, y, rows, k, st);
          else gufo::hip::LaunchQ8KBlockGEMV(dw, type, dx, y, rows, k, st);
        }
      };
      std::vector<std::vector<float>> out(2, std::vector<float>(ny));
      for (int arm = 0; arm < 2; ++arm) {
        launch(arm, dy[arm]);
        HC(hipStreamSynchronize(st));
        HC(hipGetLastError());
        HC(hipMemcpy(out[arm].data(), dy[arm], ny * 4, hipMemcpyDeviceToHost));
      }
      std::size_t ndiff = 0, nonfinite = 0, checked = 0;
      double rms = 0;
      for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t o = 0; o < outs; ++o) {
          const std::size_t i = t * stride + o;
          ndiff += std::memcmp(&out[0][i], &out[1][i], 4) != 0;
          nonfinite += !std::isfinite(out[0][i]);
          rms += static_cast<double>(out[0][i]) * out[0][i];
          ++checked;
        }
      }
      // Bytes between outputs (the stride gap) must be left as they were by both arms.
      std::size_t gap_touched = 0;
      for (std::size_t t = 0; t < tokens; ++t)
        for (std::size_t o = outs; o < stride; ++o) {
          std::uint32_t b0, b1;
          std::memcpy(&b0, &out[0][t * stride + o], 4), std::memcpy(&b1, &out[1][t * stride + o], 4);
          gap_touched += (b0 != 0xA5A5A5A5U) + (b1 != 0xA5A5A5A5U);
        }
      rms = std::sqrt(rms / static_cast<double>(checked));
      std::string tim = "null";
      if (timing && tokens == 1 && xkind == 0) {
        const double bytes = static_cast<double>(wbytes + sbytes) + static_cast<double>(in) * 4.0;
        std::vector<double> us[2];
        for (int r = 0; r < rounds; ++r) {
          for (int j = 0; j < 2; ++j) {
            const int arm = (r % 2 == 0) ? j : 1 - j;
            for (int it = 0; it < warm; ++it) launch(arm, dy[arm], static_cast<std::size_t>(it) % ncopy);
            HC(hipEventRecord(e0, st));
            for (int it = 0; it < iters; ++it) launch(arm, dy[arm], static_cast<std::size_t>(warm + it) % ncopy);
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
      std::printf("{\"case\":\"%s\",\"kind\":\"%s\",\"type\":\"%s\",\"rows\":%zu,\"k\":%zu,\"real_weights\":%s,"
                  "\"x\":\"%s\",\"tokens\":%zu,\"outputs\":%zu,\"outputs_differing\":%zu,\"gap_touched\":%zu,"
                  "\"ref_nonfinite\":%zu,\"ref_rms\":%.4e,\"weight_copies\":%zu,\"timing\":%s}\n",
                  name.c_str(), kind.c_str(), router ? "f32" : type_s.c_str(), router ? outs : rows, in,
                  gguf.empty() ? "false" : "true", xkind == 0 ? "uniform" : "normal0.05", tokens, checked, ndiff,
                  gap_touched, nonfinite, rms, ncopy, tim.c_str());
      std::fflush(stdout);
      HC(hipFree(dx));
      for (auto& p : dy) HC(hipFree(p));
    }
  }
  return 0;
}
