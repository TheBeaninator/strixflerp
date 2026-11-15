// tool.flerp.strixw_panel_kernel_bench: the kernel gate of the strixw panel layout (gate/strixw-panel-kernels.toml).
//
// For each tensor class named on the command line (class=tensor name of the fleet strixw), it reads that tensor's
// q / scale / min components from the file by byte range (pread; the index is the only other thing read), uploads
// them row-major as the fleet has them and, after format.flerp.q4panel's reference relayout in memory, as panels -
// one copy of the same dequantized weights in each layout - and runs every kernel path both ways through the
// names strixite's executor calls (the adapters of kernel.flerp.linear_q4p / linear_q8p / wmma_q4p / moe_grouped_p
// send the row-major view to strixite's kernel and the panel-tagged view to the panel kernel). Outputs are compared
// byte for byte; each kernel is timed with hip events over --reps launches, origin and panel alternating, the GPU's
// caches flushed (a 256 MB device memset) before every timed launch.
//
// A class is run at its stored width (real codes) and at the other width (codes drawn from a fixed seed, the real
// scales and mins), so every class shape is covered at Q4 and Q8. Output: one JSON object a line in --out.

#include "common/hip_check.hpp"
#include "flerp/q4panel/q4panel.hpp"
#include "formats/strixw.hpp"
#include "kernels/linear_q4.hpp"
#include "kernels/linear_q8.hpp"
#include "kernels/linear_wmma.hpp"
#include "kernels/moe_grouped.hpp"
#include "runtime/panel_device.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <tuple>
#include <random>
#include <string>
#include <vector>

using namespace strix;
using kernels::Act;

namespace {

uint64_t g_bytes_read = 0;

[[noreturn]] void die(const std::string &m) {
    std::fprintf(stderr, "strixw_panel_kernel_bench: %s\n", m.c_str());
    std::exit(2);
}

std::vector<uint8_t> pread_range(int fd, uint64_t off, uint64_t n) {
    std::vector<uint8_t> b(n);
    uint64_t got = 0;
    while (got < n) {
        const ssize_t r = ::pread(fd, b.data() + got, n - got, (off_t)(off + got));
        if (r <= 0) die("pread failed at " + std::to_string(off + got));
        got += (uint64_t)r;
    }
    g_bytes_read += n;
    return b;
}

struct Dev {
    void *p = nullptr;
    size_t n = 0;
    explicit Dev(size_t bytes) : n(bytes) { STRIX_HIP_CHECK(hipMalloc(&p, bytes ? bytes : 16), "bench buffer ", bytes); }
    ~Dev() { (void)hipFree(p); }
    Dev(const Dev &) = delete;
    template <typename T> T *as() const { return static_cast<T *>(p); }
    void up(const void *h, size_t bytes) { STRIX_HIP_CHECK(hipMemcpy(p, h, bytes, hipMemcpyHostToDevice), "upload"); }
    std::vector<uint8_t> down(size_t bytes) const {
        std::vector<uint8_t> h(bytes);
        STRIX_HIP_CHECK(hipMemcpy(h.data(), p, bytes, hipMemcpyDeviceToHost), "download");
        return h;
    }
};

// One weight in both layouts.
struct Weight {
    int bits = 0;
    int64_t N = 0, K = 0, G = 0, E = 0;
    bool real = true;
    std::unique_ptr<Dev> q, s, m, panels;
    uint64_t bytes = 0;  // q + scale + min
    Q4DeviceView rm4() const { return {q->as<uint8_t>(), s->as<uint16_t>(), m->as<uint16_t>(), N, K, G}; }
    Q8DeviceView rm8() const { return {q->as<uint8_t>(), s->as<uint16_t>(), m->as<uint16_t>(), N, K, G}; }
    Q4DeviceView p4() const { return flerp::q4_panel_view(panels->as<uint8_t>(), N, K, G); }
    Q8DeviceView p8() const { return flerp::q8_panel_view(panels->as<uint8_t>(), N, K, G); }
};

Weight make_weight(int bits, int64_t N, int64_t K, int64_t G, int64_t E, const std::vector<uint8_t> &codes,
                   const std::vector<uint8_t> &scale, const std::vector<uint8_t> &minv, bool real) {
    Weight w;
    w.bits = bits, w.N = N, w.K = K, w.G = G, w.E = E, w.real = real;
    const flerp::Panel16Geom g = flerp::panel16_geom(N, K, G, bits, "bench weight");
    std::vector<uint8_t> pan(g.total_bytes);
    flerp::panel16_pack(g, codes.data(), scale.data(), minv.data(), pan.data());
    w.q = std::make_unique<Dev>(codes.size());
    w.q->up(codes.data(), codes.size());
    w.s = std::make_unique<Dev>(scale.size());
    w.s->up(scale.data(), scale.size());
    w.m = std::make_unique<Dev>(minv.size());
    w.m->up(minv.data(), minv.size());
    w.panels = std::make_unique<Dev>(pan.size());
    w.panels->up(pan.data(), pan.size());
    w.bytes = codes.size() + scale.size() + minv.size();
    return w;
}

struct Timer {
    hipEvent_t a, b;
    Timer() {
        STRIX_HIP_CHECK(hipEventCreate(&a), "event");
        STRIX_HIP_CHECK(hipEventCreate(&b), "event");
    }
};

struct Bench {
    FILE *out;
    int reps, warm;
    Dev flush{256u << 20};
    Timer t;
    hipStream_t st = nullptr;
    int cases = 0, unequal = 0;

    template <typename F> float time_one(F &&f) {
        STRIX_HIP_CHECK(hipMemsetAsync(flush.p, cases & 0xff, flush.n, st), "flush");
        STRIX_HIP_CHECK(hipEventRecord(t.a, st), "event");
        f();
        STRIX_HIP_CHECK(hipEventRecord(t.b, st), "event");
        STRIX_HIP_CHECK(hipEventSynchronize(t.b), "sync");
        float ms = 0;
        STRIX_HIP_CHECK(hipEventElapsedTime(&ms, t.a, t.b), "elapsed");
        return ms;
    }
    static void stats(std::vector<float> v, double &med, double &lo, double &hi, double &sd) {
        std::sort(v.begin(), v.end());
        med = v[v.size() / 2], lo = v[v.size() / 4], hi = v[(3 * v.size()) / 4];
        double mean = 0;
        for (float x : v) mean += x;
        mean /= v.size();
        double s2 = 0;
        for (float x : v) s2 += (x - mean) * (x - mean);
        sd = std::sqrt(s2 / (v.size() > 1 ? v.size() - 1 : 1));
    }
    // origin(), panel(): launch into their own output buffers; outs: (origin, panel, bytes) to compare after the
    // first (untimed) run of each. extra: JSON fields naming the case.
    template <typename FO, typename FP>
    void run(const std::string &extra, uint64_t wbytes, FO &&origin, FP &&panel,
             const std::vector<std::tuple<const Dev *, const Dev *, size_t>> &outs) {
        ++cases;
        for (auto &[a, b, n] : outs) {  // outputs poisoned differently; error words (<= 16 bytes) zeroed
            STRIX_HIP_CHECK(hipMemsetAsync(a->p, n <= 16 ? 0 : 0x5a, n, st), "poison");
            STRIX_HIP_CHECK(hipMemsetAsync(b->p, n <= 16 ? 0 : 0xa5, n, st), "poison");
        }
        origin();
        panel();
        STRIX_HIP_CHECK(hipStreamSynchronize(st), "sync");
        bool equal = true;
        size_t first = (size_t)-1, ndiff = 0;
        for (auto &[a, b, n] : outs) {
            const auto ha = a->down(n), hb = b->down(n);
            for (size_t i = 0; i < n; ++i)
                if (ha[i] != hb[i]) {
                    if (equal) first = i;
                    equal = false;
                    ++ndiff;
                }
        }
        if (!equal) ++unequal;
        for (int i = 0; i < warm; ++i) origin(), panel();
        std::vector<float> to, tp;
        for (int i = 0; i < reps; ++i) {
            if (i % 2 == 0) {
                to.push_back(time_one(origin));
                tp.push_back(time_one(panel));
            } else {
                tp.push_back(time_one(panel));
                to.push_back(time_one(origin));
            }
        }
        double om, ol, oh, os, pm, pl, ph, ps;
        stats(to, om, ol, oh, os);
        stats(tp, pm, pl, ph, ps);
        std::fprintf(out,
                     "{%s, \"equal\": %s, \"diff_bytes\": %zu, \"first_diff\": %lld, \"weight_bytes\": %llu, "
                     "\"origin_ms\": {\"median\": %.5f, \"q1\": %.5f, \"q3\": %.5f, \"sd\": %.5f}, "
                     "\"panel_ms\": {\"median\": %.5f, \"q1\": %.5f, \"q3\": %.5f, \"sd\": %.5f}, \"reps\": %d, "
                     "\"origin_gbps\": %.2f, \"panel_gbps\": %.2f, \"panel_pct_of_origin_time\": %.2f}\n",
                     extra.c_str(), equal ? "true" : "false", ndiff, equal ? -1ll : (long long)first,
                     (unsigned long long)wbytes, om, ol, oh, os, pm, pl, ph, ps, reps, wbytes / om / 1e6, wbytes / pm / 1e6,
                     100.0 * pm / om);
        std::fflush(out);
        std::fprintf(stderr, "%s equal=%d origin %.4f ms panel %.4f ms (%.1f%%)\n", extra.c_str(), (int)equal, om, pm,
                     100.0 * pm / om);
    }
};

std::vector<uint16_t> rand_bf16(size_t n, uint64_t seed, float scale = 1.0f) {
    std::mt19937_64 r(seed);
    std::normal_distribution<float> d(0.f, scale);
    std::vector<uint16_t> v(n);
    for (auto &x : v) {
        const float f = d(r);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        u += 0x7fffu + ((u >> 16) & 1u);
        x = (uint16_t)(u >> 16);
    }
    return v;
}

// Routing: per token A distinct experts drawn from a Zipf(1.1) popularity over a fixed random permutation of E.
std::vector<int32_t> route(int64_t M, int64_t A, int64_t E, uint64_t seed) {
    std::mt19937_64 r(seed);
    std::vector<int32_t> perm(E);
    for (int64_t i = 0; i < E; ++i) perm[i] = (int32_t)i;
    std::shuffle(perm.begin(), perm.end(), r);
    std::vector<double> w(E);
    for (int64_t i = 0; i < E; ++i) w[i] = 1.0 / std::pow((double)(i + 1), 1.1);
    std::discrete_distribution<int64_t> d(w.begin(), w.end());
    std::vector<int32_t> ids(M * A);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t a = 0; a < A; ++a) {
            int32_t e;
            bool dup;
            do {
                e = perm[d(r)];
                dup = false;
                for (int64_t b = 0; b < a; ++b) dup = dup || ids[m * A + b] == e;
            } while (dup);
            ids[m * A + a] = e;
        }
    return ids;
}

uint64_t distinct_expert_bytes(const std::vector<int32_t> &ids, int64_t E, uint64_t per_expert) {
    std::vector<char> seen(E + 1, 0);
    uint64_t n = 0;
    for (int32_t i : ids)
        if (i >= 0 && i < E && !seen[i]) seen[i] = 1, ++n;
    return n * per_expert;
}

std::string jcase(const std::string &cls, const Weight &w, const std::string &path, int64_t M, const std::string &more) {
    char b[512];
    std::snprintf(b, sizeof b,
                  "\"class\": \"%s\", \"bits\": %d, \"codes\": \"%s\", \"N\": %lld, \"K\": %lld, \"G\": %lld, "
                  "\"path\": \"%s\", \"M\": %lld%s",
                  cls.c_str(), w.bits, w.real ? "real" : "synthetic", (long long)w.N, (long long)w.K, (long long)w.G,
                  path.c_str(), (long long)M, more.c_str());
    return b;
}

void dense_paths(Bench &B, const std::string &cls, const Weight &w, const std::vector<int64_t> &dec_m,
                 const std::vector<int64_t> &pre_m) {
    const int64_t maxM = std::max(*std::max_element(dec_m.begin(), dec_m.end()),
                                  pre_m.empty() ? 1 : *std::max_element(pre_m.begin(), pre_m.end()));
    const auto xh = rand_bf16((size_t)(maxM * w.K), 1000 + w.K + w.N, 1.0f);
    Dev x(xh.size() * 2);
    x.up(xh.data(), xh.size() * 2);
    for (Act oa : {Act::BF16, Act::F32}) {
        const size_t oes = oa == Act::F32 ? 4 : 2;
        const std::string os = oa == Act::F32 ? ", \"out\": \"f32\"" : ", \"out\": \"bf16\"";
        auto one = [&](int64_t M, bool prefill) {
            Dev yo((size_t)(M * w.N) * oes), yp((size_t)(M * w.N) * oes);
            auto o = [&] {
                if (prefill) {
                    if (w.bits == 4) kernels::linear_q4_wmma(x.p, w.rm4(), yo.p, M, Act::BF16, oa, B.st);
                    else kernels::linear_q8_wmma(x.p, w.rm8(), yo.p, M, Act::BF16, oa, B.st);
                } else {
                    if (w.bits == 4) kernels::linear_q4(x.p, w.rm4(), yo.p, M, Act::BF16, oa, B.st);
                    else kernels::linear_q8(x.p, w.rm8(), yo.p, M, Act::BF16, oa, B.st);
                }
            };
            auto p = [&] {
                if (prefill) {
                    if (w.bits == 4) kernels::linear_q4_wmma(x.p, w.p4(), yp.p, M, Act::BF16, oa, B.st);
                    else kernels::linear_q8_wmma(x.p, w.p8(), yp.p, M, Act::BF16, oa, B.st);
                } else {
                    if (w.bits == 4) kernels::linear_q4(x.p, w.p4(), yp.p, M, Act::BF16, oa, B.st);
                    else kernels::linear_q8(x.p, w.p8(), yp.p, M, Act::BF16, oa, B.st);
                }
            };
            B.run(jcase(cls, w, prefill ? "dense-prefill-wmma" : "dense-decode", M, os), w.bytes, o, p,
                  {{&yo, &yp, (size_t)(M * w.N) * oes}});
        };
        for (int64_t M : dec_m) one(M, false);
        if (w.K % 64 == 0)
            for (int64_t M : pre_m) {
                // two outputs of M x N must fit beside the weights: the LM head (N = 248320) runs at the smallest M only
                // (the executor runs it on a chunk's last rows, never on 8192)
                if ((uint64_t)M * w.N * oes * 2 > (3ull << 30)) {
                    std::fprintf(B.out, "{%s, \"skipped\": \"outputs %.1f GB\"}\n",
                                 jcase(cls, w, "dense-prefill-wmma", M, os).c_str(), M * w.N * oes * 2 / 1e9);
                    continue;
                }
                one(M, true);
            }
    }
}

// Expert decode (Q4): gather + SwiGLU and gather on gate_up, combine on down; ids per token; one case with a bad id.
void expert_decode(Bench &B, const Weight &gu, const Weight &dn, int64_t A, const std::vector<int64_t> &dec_m) {
    const int64_t E = gu.E, Ngu = gu.N / E, Ndn = dn.N / E, I = Ngu / 2, d = gu.K;
    for (int64_t M : dec_m) {
        for (int bad = 0; bad < (M == 4 ? 2 : 1); ++bad) {
            auto ids_h = route(M, A, E, 77 + M);
            if (bad) ids_h[(M - 1) * A + 3] = (int32_t)E;  // one out-of-range id: NaN outputs and the error word
            Dev ids(ids_h.size() * 4);
            ids.up(ids_h.data(), ids_h.size() * 4);
            const auto xh = rand_bf16((size_t)(M * d), 5 + M), hh = rand_bf16((size_t)(M * A * I), 6 + M, 0.5f);
            Dev x(xh.size() * 2), h(hh.size() * 2);
            x.up(xh.data(), xh.size() * 2);
            h.up(hh.data(), hh.size() * 2);
            std::vector<float> coef_h(M * A);
            std::mt19937 r(9 + M);
            for (auto &c : coef_h) c = std::uniform_real_distribution<float>(0.01f, 0.3f)(r);
            Dev coef(coef_h.size() * 4);
            coef.up(coef_h.data(), coef_h.size() * 4);
            Dev eo(16), ep(16);
            const std::string more = std::string(", \"A\": ") + std::to_string(A) + ", \"bad_id\": " + (bad ? "true" : "false");
            const uint64_t gub = distinct_expert_bytes(ids_h, E, gu.bytes / E), dnb = distinct_expert_bytes(ids_h, E, dn.bytes / E);
            {
                Dev yo((size_t)(M * A * I) * 2), yp((size_t)(M * A * I) * 2);
                STRIX_HIP_CHECK(hipMemset(eo.p, 0, 16), "err");
                STRIX_HIP_CHECK(hipMemset(ep.p, 0, 16), "err");
                B.run(jcase("exp_gu", gu, "expert-decode-gather-swiglu", M, more), gub,
                      [&] { kernels::linear_q4_experts_gather_swiglu(x.p, gu.rm4(), E, ids.as<int32_t>(), M, A, yo.p, Act::BF16, eo.as<uint32_t>(), B.st); },
                      [&] { kernels::linear_q4_experts_gather_swiglu(x.p, gu.p4(), E, ids.as<int32_t>(), M, A, yp.p, Act::BF16, ep.as<uint32_t>(), B.st); },
                      {{&yo, &yp, (size_t)(M * A * I) * 2}, {&eo, &ep, 12}});
            }
            {
                Dev yo((size_t)(M * A * Ngu) * 2), yp((size_t)(M * A * Ngu) * 2);
                STRIX_HIP_CHECK(hipMemset(eo.p, 0, 16), "err");
                STRIX_HIP_CHECK(hipMemset(ep.p, 0, 16), "err");
                B.run(jcase("exp_gu", gu, "expert-decode-gather", M, more), gub,
                      [&] { kernels::linear_q4_experts_gather(x.p, gu.rm4(), E, ids.as<int32_t>(), M, A, yo.p, Act::BF16, eo.as<uint32_t>(), B.st); },
                      [&] { kernels::linear_q4_experts_gather(x.p, gu.p4(), E, ids.as<int32_t>(), M, A, yp.p, Act::BF16, ep.as<uint32_t>(), B.st); },
                      {{&yo, &yp, (size_t)(M * A * Ngu) * 2}, {&eo, &ep, 12}});
            }
            {
                Dev yo((size_t)(M * Ndn) * 2), yp((size_t)(M * Ndn) * 2);
                STRIX_HIP_CHECK(hipMemset(eo.p, 0, 16), "err");
                STRIX_HIP_CHECK(hipMemset(ep.p, 0, 16), "err");
                B.run(jcase("exp_down", dn, "expert-decode-combine", M, more), dnb,
                      [&] { kernels::linear_q4_experts_combine(h.p, dn.rm4(), E, ids.as<int32_t>(), coef.as<float>(), M, A, yo.p, Act::BF16, eo.as<uint32_t>(), B.st); },
                      [&] { kernels::linear_q4_experts_combine(h.p, dn.p4(), E, ids.as<int32_t>(), coef.as<float>(), M, A, yp.p, Act::BF16, ep.as<uint32_t>(), B.st); },
                      {{&yo, &yp, (size_t)(M * Ndn) * 2}, {&eo, &ep, 12}});
            }
        }
    }
}

// Expert prefill: strixite's routing once, then each F16 kind (Narrow / Wide / Wide256) through the gather + SwiGLU
// (BF16 h and F16 h), the plain gather, and the combine (BF16 h, F16 h, F16 h + shared add).
void expert_prefill(Bench &B, const Weight &gu, const Weight &dn, int64_t A, const std::vector<int64_t> &pre_m) {
    using kernels::MoeMath;
    const int64_t E = gu.E, Ngu = gu.N / E, Ndn = dn.N / E, I = Ngu / 2, d = gu.K;
    for (int64_t M : pre_m) {
        for (int bad = 0; bad < (M == pre_m.front() ? 2 : 1); ++bad) {
            auto ids_h = route(M, A, E, 1234 + M);
            if (bad) ids_h[(M / 2) * A + 1] = (int32_t)E;
            Dev ids(ids_h.size() * 4), err(16);
            ids.up(ids_h.data(), ids_h.size() * 4);
            STRIX_HIP_CHECK(hipMemset(err.p, 0, 16), "err");
            const size_t wsb = kernels::moe_route_workspace_bytes(M, A, E);
            Dev ws(wsb);
            kernels::moe_group_routes(ids.as<int32_t>(), M, A, E, ws.p, wsb, err.as<uint32_t>(), B.st);
            const auto xh = rand_bf16((size_t)(M * d), 11 + M);
            Dev x(xh.size() * 2);
            x.up(xh.data(), xh.size() * 2);
            std::vector<float> coef_h(M * A);
            std::mt19937 r(13 + M);
            for (auto &c : coef_h) c = std::uniform_real_distribution<float>(0.01f, 0.3f)(r);
            Dev coef(coef_h.size() * 4);
            coef.up(coef_h.data(), coef_h.size() * 4);
            const auto sh = rand_bf16((size_t)(M * Ndn), 17 + M, 0.5f);
            Dev shared_s(sh.size() * 2);
            shared_s.up(sh.data(), sh.size() * 2);
            std::vector<float> gl(M);
            for (auto &g : gl) g = std::uniform_real_distribution<float>(-3.f, 3.f)(r);
            Dev gate(gl.size() * 4), serr_o(16), serr_p(16);
            gate.up(gl.data(), gl.size() * 4);
            const uint64_t gub = distinct_expert_bytes(ids_h, E, gu.bytes / E), dnb = distinct_expert_bytes(ids_h, E, dn.bytes / E);
            // the down's inputs: the origin's own SwiGLU outputs (BF16 h and F16 h)
            Dev hb((size_t)(M * A * I) * 2), hf((size_t)(M * A * I) * 2);
            kernels::linear_q4_experts_gather_grouped_swiglu(x.p, gu.rm4(), E, ws.p, wsb, M, A, hb.p, Act::BF16, MoeMath::WmmaF16, B.st, false);
            kernels::linear_q4_experts_gather_grouped_swiglu(x.p, gu.rm4(), E, ws.p, wsb, M, A, hf.p, Act::BF16, MoeMath::WmmaF16, B.st, true);
            const std::pair<MoeMath, const char *> kinds[] = {{MoeMath::WmmaF16Narrow, "narrow"},
                                                              {MoeMath::WmmaF16Wide, "wide"},
                                                              {MoeMath::WmmaF16Wide256, "wide256"}};
            for (auto [math, mname] : kinds) {
                const std::string more = std::string(", \"A\": ") + std::to_string(A) + ", \"math\": \"" + mname +
                                         "\", \"bad_id\": " + (bad ? "true" : "false");
                for (int xh16 = 0; xh16 < 2; ++xh16) {
                    Dev yo((size_t)(M * A * I) * 2), yp((size_t)(M * A * I) * 2);
                    B.run(jcase("exp_gu", gu, xh16 ? "expert-prefill-swiglu-f16h" : "expert-prefill-swiglu", M, more), gub,
                          [&] { kernels::linear_q4_experts_gather_grouped_swiglu(x.p, gu.rm4(), E, ws.p, wsb, M, A, yo.p, Act::BF16, math, B.st, xh16); },
                          [&] { kernels::linear_q4_experts_gather_grouped_swiglu(x.p, gu.p4(), E, ws.p, wsb, M, A, yp.p, Act::BF16, math, B.st, xh16); },
                          {{&yo, &yp, (size_t)(M * A * I) * 2}});
                }
                {
                    Dev yo((size_t)(M * A * Ngu) * 2), yp((size_t)(M * A * Ngu) * 2);
                    B.run(jcase("exp_gu", gu, "expert-prefill-gather", M, more), gub,
                          [&] { kernels::linear_q4_experts_gather_grouped(x.p, gu.rm4(), E, ws.p, wsb, M, A, yo.p, Act::BF16, math, B.st); },
                          [&] { kernels::linear_q4_experts_gather_grouped(x.p, gu.p4(), E, ws.p, wsb, M, A, yp.p, Act::BF16, math, B.st); },
                          {{&yo, &yp, (size_t)(M * A * Ngu) * 2}});
                }
                for (int variant = 0; variant < 3; ++variant) {  // BF16 h; F16 h; F16 h + shared add
                    const bool f16h = variant >= 1;
                    Dev yo((size_t)(M * Ndn) * 2), yp((size_t)(M * Ndn) * 2);
                    Dev po((size_t)(M * A * Ndn) * 4), pp((size_t)(M * A * Ndn) * 4);
                    kernels::MoeSharedAdd so, sp;
                    so.s = sp.s = shared_s.p;
                    so.gate_logit = sp.gate_logit = gate.as<float>();
                    so.gate_ld = sp.gate_ld = 1;
                    so.gate_index = sp.gate_index = 7;
                    so.err = serr_o.as<uint32_t>(), sp.err = serr_p.as<uint32_t>();
                    STRIX_HIP_CHECK(hipMemset(serr_o.p, 0, 16), "err");
                    STRIX_HIP_CHECK(hipMemset(serr_p.p, 0, 16), "err");
                    const kernels::MoeSharedAdd *sho = variant == 2 ? &so : nullptr, *shp = variant == 2 ? &sp : nullptr;
                    const void *hin = f16h ? hf.p : hb.p;
                    const char *pname = variant == 0 ? "expert-prefill-combine" : variant == 1 ? "expert-prefill-combine-f16h"
                                                                                               : "expert-prefill-combine-f16h-shared";
                    B.run(jcase("exp_down", dn, pname, M, more), dnb,
                          [&] { kernels::linear_q4_experts_combine_grouped(hin, dn.rm4(), E, ids.as<int32_t>(), coef.as<float>(), ws.p, wsb, po.as<float>(), M, A, yo.p, Act::BF16, math, B.st, sho, f16h); },
                          [&] { kernels::linear_q4_experts_combine_grouped(hin, dn.p4(), E, ids.as<int32_t>(), coef.as<float>(), ws.p, wsb, pp.as<float>(), M, A, yp.p, Act::BF16, math, B.st, shp, f16h); },
                          {{&yo, &yp, (size_t)(M * Ndn) * 2}, {&serr_o, &serr_p, 12}});
                }
            }
        }
    }
}

std::vector<int64_t> parse_list(const std::string &s) {
    std::vector<int64_t> v;
    size_t a = 0;
    while (a < s.size()) {
        size_t b = s.find(',', a);
        if (b == std::string::npos) b = s.size();
        if (b > a) v.push_back(std::stoll(s.substr(a, b - a)));
        a = b + 1;
    }
    return v;
}

}  // namespace

int main(int argc, char **argv) {
    std::string weights, out, only;
    std::vector<std::pair<std::string, std::string>> classes;  // class -> tensor name
    std::vector<int64_t> dec_m = {1, 2, 3, 4, 5, 6, 7, 8}, pre_m = {512, 2048, 8192};
    int reps = 20, warm = 3;
    int64_t A = 10;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) die("missing value for " + a);
            return argv[++i];
        };
        if (a == "--weights") weights = val();
        else if (a == "--out") out = val();
        else if (a == "--class") {
            const std::string v = val();
            const size_t eq = v.find('=');
            if (eq == std::string::npos) die("--class wants class=tensor");
            classes.emplace_back(v.substr(0, eq), v.substr(eq + 1));
        } else if (a == "--decode-m") dec_m = parse_list(val());
        else if (a == "--prefill-m") pre_m = parse_list(val());
        else if (a == "--reps") reps = std::stoi(val());
        else if (a == "--warm") warm = std::stoi(val());
        else if (a == "--active") A = std::stoll(val());
        else if (a == "--only") only = val();  // dense | experts
        else die("unknown option " + a);
    }
    if (weights.empty() || out.empty() || classes.empty()) die("usage: --weights F --out FILE --class c=name ...");
    const StrixwFile f(weights);  // header and index; data is read below by byte range only
    const int fd = ::open(weights.c_str(), O_RDONLY);
    if (fd < 0) die("cannot open " + weights);
    FILE *o = std::fopen(out.c_str(), "w");
    if (!o) die("cannot write " + out);
    Bench B{o, reps, warm};
    STRIX_HIP_CHECK(hipStreamCreate(&B.st), "stream");

    std::map<std::string, std::unique_ptr<Weight>> real;  // experts kept for the expert paths
    for (auto &[cls, name] : classes) {
        const StrixwTensor *t = f.find(name);
        if (!t) die("no tensor " + name);
        const int bits = t->encoding == StrixwEncoding::Q4RowMajor ? 4 : t->encoding == StrixwEncoding::Q8RowMajor ? 8 : 0;
        if (!bits) die(name + ": not q4 / q8 row-major (" + strixw_encoding_name(t->encoding) + ")");
        std::vector<uint8_t> q, s, m;
        for (const StrixwComponent &c : t->components) {
            auto b = pread_range(fd, c.offset, c.bytes);
            if (c.role == StrixwRole::Q) q = std::move(b);
            else if (c.role == StrixwRole::Scale) s = std::move(b);
            else if (c.role == StrixwRole::Min) m = std::move(b);
        }
        const int64_t N = t->shape[0], K = t->shape[1], G = t->group_size, E = t->experts;
        std::fprintf(stderr, "%s: %s [%lld, %lld] G%lld E%lld q%d, %.1f MB read\n", cls.c_str(), name.c_str(),
                     (long long)N, (long long)K, (long long)G, (long long)E, bits, (q.size() + s.size() + m.size()) / 1e6);
        const bool experts = E > 0;
        if (experts) {
            real[cls] = std::make_unique<Weight>(make_weight(bits, N, K, G, E, q, s, m, true));
            continue;
        }
        if (only == "experts") continue;
        for (int width : {bits, bits == 4 ? 8 : 4}) {
            std::vector<uint8_t> codes = q;
            if (width != bits) {  // the other width: codes from a fixed seed, the real scales and mins
                codes.assign((size_t)(N * K * width / 8), 0);
                std::mt19937_64 r(31337 + N + K);
                for (auto &c : codes) c = (uint8_t)r();
            }
            const Weight w = make_weight(width, N, K, G, 0, codes, s, m, width == bits);
            dense_paths(B, cls, w, dec_m, pre_m);
        }
    }
    if (only != "dense" && real.count("exp_gu") && real.count("exp_down")) {
        expert_decode(B, *real["exp_gu"], *real["exp_down"], A, dec_m);
        expert_prefill(B, *real["exp_gu"], *real["exp_down"], A, pre_m);
    }
    std::fprintf(o, "{\"summary\": true, \"cases\": %d, \"unequal\": %d, \"bytes_read\": %llu}\n", B.cases, B.unequal,
                 (unsigned long long)g_bytes_read);
    std::fclose(o);
    std::fprintf(stderr, "cases %d, unequal %d, %.3f GB read from %s\n", B.cases, B.unequal, g_bytes_read / 1e9,
                 weights.c_str());
    ::close(fd);
    return B.unequal == 0 ? 0 : 1;
}
