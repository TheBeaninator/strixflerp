// SPDX-License-Identifier: MIT
// Interface gate driver for the decode GEMV on GGUF Q8_0 and Q4_K blocks, through gufo's launch API:
//   gufo::hip::LaunchQ8KBlockGEMV(A, type, x, y, M, K, stream)
#include "flerp_gate/gate_common.hpp"
#include "src/families/qwen/kernels/ops/gemm.hpp"
#include "gguf_gemv.hpp"

#include <functional>

using namespace gate;

static void put_half(unsigned char *p, float v) {
    const uint16_t h = f32_to_half(v);
    std::memcpy(p, &h, 2);
}

// Random blocks with the statistics of a quantized projection: scales around 1e-3, codes over their whole range.
static std::vector<unsigned char> make_weights(bool q4k, int64_t M, int64_t K, uint64_t seed) {
    Rng r(seed);
    const int64_t qk = q4k ? 256 : 32, bytes = q4k ? 144 : 34, blocks = M * (K / qk);
    std::vector<unsigned char> w((size_t)(blocks * bytes));
    for (int64_t b = 0; b < blocks; ++b) {
        unsigned char *blk = w.data() + b * bytes;
        if (!q4k) {
            put_half(blk, 0.0005f + (float)r.uniform() * 0.002f);
            for (int i = 0; i < 32; ++i) blk[2 + i] = (unsigned char)(r.next() & 0xff);
        } else {
            put_half(blk, 0.0002f + (float)r.uniform() * 0.0008f);
            put_half(blk + 2, 0.0004f + (float)r.uniform() * 0.0015f);
            for (int i = 0; i < 12 + 128; ++i) blk[4 + i] = (unsigned char)(r.next() & 0xff);
        }
    }
    return w;
}

int main(int argc, char **argv) {
    return run_driver(argc, argv, "gguf_gemv", [](const Case &c, Obj &o, hipStream_t stream, const std::string &dumps) {
        const bool q4k = c.str("type") == "q4_k";
        const auto type = q4k ? gufo::core::GgmlType::kQ4_K : gufo::core::GgmlType::kQ8_0;
        const int64_t M = c.num("M"), K = c.num("K"), qk = q4k ? 256 : 32, bytes = q4k ? 144 : 34;
        const std::vector<unsigned char> w = make_weights(q4k, M, K, (uint64_t)c.num("seed", 1));
        std::vector<float> x((size_t)K);
        Rng r((uint64_t)c.num("seed", 1) + 7);
        const std::string pattern = c.str("pattern", "gauss");
        for (auto &v : x) v = pattern == "wide-range" ? r.gauss(1.0f) * std::pow(10.0f, (float)(r.uniform() * 10.0 - 5.0)) : r.gauss(1.0f);
        Dev<unsigned char> dw(w.size());
        dw.put(w);
        Dev<float> dx(x.size()), dy((size_t)M);
        dx.put(x);
        auto launch = [&] { gufo::hip::LaunchQ8KBlockGEMV(dw.p, type, dx.p, dy.p, (std::size_t)M, (std::size_t)K, stream); };
        o.s("type", c.str("type")).i("M", M).i("K", K);
        if (c.str("kind") == "parity") {
            dy.fill_bytes(0x5a);
            launch();
            hip_ok(hipStreamSynchronize(stream), "sync");
            const std::vector<float> got = dy.get();
            std::vector<float> want((size_t)M);
            double worst = 0, max_abs_exact = 0;
            // Order emulations, each a claim about one kernel's source; the first that reproduces every emulated row is reported.
            struct Emu {
                const char *name;
                std::function<float(const unsigned char *, const float *, int64_t)> fn;
                long long equal;
            };
            std::vector<Emu> emus;
            if (q4k) {
                emus.push_back({"hipfire-lanes (fma)", [](auto r, auto xv, auto k) { return reference::q4k_hipfire(r, xv, k, true); }, 0});
                emus.push_back({"hipfire-lanes (mul, add)", [](auto r, auto xv, auto k) { return reference::q4k_hipfire(r, xv, k, false); }, 0});
                for (int form = 0; form < 3; ++form)
                    for (int fd = 1; fd >= 0; --fd) {
                        static const char *names[6] = {"gufo-sub16 (fma dot, fma scale)", "gufo-sub16 (mul-add dot, fma scale)",
                                                       "gufo-sub16 (fma dot, fma offset)", "gufo-sub16 (mul-add dot, fma offset)",
                                                       "gufo-sub16 (fma dot, no fusion)", "gufo-sub16 (mul-add dot, no fusion)"};
                        emus.push_back({names[form * 2 + (1 - fd)], [=](auto r, auto xv, auto k) { return reference::q4k_gufo_sub16(r, xv, k, fd != 0, form); }, 0});
                    }
            } else {
                emus.push_back({"hipfire-lanes (fma)", [](auto r, auto xv, auto k) { return reference::q8_0_hipfire(r, xv, k, true); }, 0});
                emus.push_back({"hipfire-lanes (mul, add)", [](auto r, auto xv, auto k) { return reference::q8_0_hipfire(r, xv, k, false); }, 0});
                emus.push_back({"gufo-block-sums (fma, fma)", [](auto r, auto xv, auto k) { return reference::q8_0_gufo_rows(r, xv, k, true, true); }, 0});
                emus.push_back({"gufo-block-sums (mul-add, fma)", [](auto r, auto xv, auto k) { return reference::q8_0_gufo_rows(r, xv, k, false, true); }, 0});
                emus.push_back({"gufo-block-sums (fma, mul-add)", [](auto r, auto xv, auto k) { return reference::q8_0_gufo_rows(r, xv, k, true, false); }, 0});
                emus.push_back({"gufo-block-sums (mul-add, mul-add)", [](auto r, auto xv, auto k) { return reference::q8_0_gufo_rows(r, xv, k, false, false); }, 0});
                emus.push_back({"gufo-warp-blocks (fma)", [](auto r, auto xv, auto k) { return reference::q8_0_gufo_warp(r, xv, k, true); }, 0});
                emus.push_back({"gufo-warp-blocks (mul, add)", [](auto r, auto xv, auto k) { return reference::q8_0_gufo_warp(r, xv, k, false); }, 0});
            }
            const int64_t emu_rows = std::min<int64_t>(M, c.num("emulate_rows", 64));      // the emulations are slow; a prefix of the rows
            for (int64_t m = 0; m < M; ++m) {
                const unsigned char *row = w.data() + m * (K / qk) * bytes;
                const reference::GgExact e = q4k ? reference::q4k_exact(row, x.data(), K) : reference::q8_0_exact(row, x.data(), K);
                want[(size_t)m] = (float)e.sum;
                const double err = std::fabs((double)got[(size_t)m] - e.sum);
                max_abs_exact = std::max(max_abs_exact, err);
                worst = std::max(worst, err / order_bound(e.sum_abs, q4k ? 2 * K : K, q4k ? 4 : 2));
                if (m >= emu_rows) continue;
                const uint32_t g = f32_bits(got[(size_t)m]);
                for (auto &e : emus) e.equal += g == f32_bits(e.fn(row, x.data(), K));
            }
            put_diff(o, "", diff_f32(got.data(), want.data(), got.size()));
            o.d("err_over_bound", worst).d("max_abs_vs_exact", max_abs_exact).i("rows_emulated", emu_rows);
            std::string emu = "none";
            for (const auto &e : emus)
                if (e.equal == emu_rows) { emu = e.name; break; }
            o.s("order_emulation", emu);
            dump(dumps, c.name, got.data(), got.size() * 4);
        } else {
            const Timing t = time_launches(launch, stream, (int)c.num("iters"), (int)c.num("repeats"));
            put_timing(o, t, (double)w.size() + 4.0 * (double)K + 4.0 * (double)M, 1);
        }
    });
}
