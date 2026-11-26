// SPDX-License-Identifier: MIT
// Interface gate driver for the strixw decode GEMV, through strixite's launch API:
//   strix::kernels::linear_q4 / linear_q5 / linear_q6 / linear_q8(x, view, y, M, act, out_act, stream)
// y[M, N] = x[M, K] * dequant(W)^T on synthetic weights (codes over their whole range, BF16 scales and mins of a
// quantized projection's size). A case with layout=panel hands the same codes, scales and mins as a Q4Panel16 /
// Q8Panel16 tensor (format.flerp.q4panel's packer, format.flerp.panel_device's tagged view): only a provider whose
// linear_q4 / linear_q8 is the panel adapter (kernel.flerp.linear_q4p / linear_q8p) takes those, so the config gives
// layout=panel to those providers only (provider case_args).
#include "flerp_gate/gate_common.hpp"
#include "kernels/linear_q4.hpp"
#include "kernels/linear_q5.hpp"
#include "kernels/linear_q6.hpp"
#include "kernels/linear_q8.hpp"
#include "flerp/q4panel/q4panel.hpp"
#include "runtime/panel_device.hpp"
#include "strixw_decode.hpp"

using namespace gate;
using strix::kernels::Act;

struct Weights {
    int bits = 4;
    int64_t N = 0, K = 0, G = 0;
    std::vector<uint8_t> q;
    std::vector<uint16_t> scale, minv;
};

// Random codes over the format's whole range; per group a BF16 scale around 1e-3 (for 4 bits; narrower codes get
// proportionally larger steps) and a negative BF16 min around half the range, as a quantizer produces them.
static Weights make_weights(int bits, int64_t N, int64_t K, int64_t G, uint64_t seed) {
    Rng r(seed);
    Weights w;
    w.bits = bits, w.N = N, w.K = K, w.G = G;
    const int64_t rb = reference::sw_row_bytes(bits, K), groups = K / G;
    const int maxc = reference::sw_max_code(bits);
    w.q.assign((size_t)(N * rb), 0);
    w.scale.resize((size_t)(N * groups));
    w.minv.resize((size_t)(N * groups));
    for (int64_t i = 0; i < N * groups; ++i) {
        const float s = (0.0005f + (float)r.uniform() * 0.002f) * 15.0f / (float)maxc;
        w.scale[(size_t)i] = f32_to_bf16(s);
        w.minv[(size_t)i] = f32_to_bf16(-0.5f * (float)maxc * bf16_to_f32(w.scale[(size_t)i]) * (0.5f + (float)r.uniform()));
    }
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k)
            reference::sw_put_code(bits, w.q.data() + n * rb, K, k, (int)(r.next() % (uint64_t)(maxc + 1)));
    return w;
}

int main(int argc, char **argv) {
    return run_driver(argc, argv, "strixw_decode", [](const Case &c, Obj &o, hipStream_t stream, const std::string &dumps) {
        const int bits = (int)c.num("bits");
        const int64_t N = c.num("N"), K = c.num("K"), G = c.num("G"), M = c.num("M");
        const bool x_bf16 = c.str("act") == "bf16", y_bf16 = c.str("out") == "bf16";
        const bool panel = c.str("layout", "row") == "panel";
        const Act act = x_bf16 ? Act::BF16 : Act::F32, out_act = y_bf16 ? Act::BF16 : Act::F32;
        const uint64_t seed = (uint64_t)c.num("seed", 1);
        const Weights w = make_weights(bits, N, K, G, seed);

        // activations: x_f holds the values the kernel sees (BF16-rounded for a BF16 run)
        Rng r(seed + 7);
        const std::string pattern = c.str("pattern", "gauss");
        std::vector<float> x_f((size_t)(M * K));
        for (auto &v : x_f) {
            v = pattern == "wide-range" ? r.gauss(1.0f) * std::pow(10.0f, (float)(r.uniform() * 8.0 - 4.0)) : r.gauss(1.0f);
            if (x_bf16) v = bf16_to_f32(f32_to_bf16(v));
        }
        std::vector<uint16_t> x_b((size_t)(M * K));
        for (size_t i = 0; i < x_b.size(); ++i) x_b[i] = f32_to_bf16(x_f[i]);
        Dev<unsigned char> dx((size_t)(M * K) * (x_bf16 ? 2 : 4));
        if (x_bf16) dx.put(reinterpret_cast<const unsigned char *>(x_b.data()), x_b.size() * 2);
        else dx.put(reinterpret_cast<const unsigned char *>(x_f.data()), x_f.size() * 4);
        Dev<unsigned char> dy((size_t)(M * N) * (y_bf16 ? 2 : 4));

        // weights, row-major planes or one panel buffer
        Dev<uint8_t> dq(w.q.size());
        Dev<uint16_t> ds(w.scale.size()), dm(w.minv.size());
        std::vector<uint8_t> packed;
        if (panel) {
            if (bits != 4 && bits != 8) throw std::runtime_error("layout=panel is for 4- and 8-bit weights only");
            const flerp::Panel16Geom g = flerp::panel16_geom(N, K, G, bits, c.name.c_str());
            packed.resize(g.total_bytes);
            flerp::panel16_pack(g, w.q.data(), reinterpret_cast<const uint8_t *>(w.scale.data()),
                                reinterpret_cast<const uint8_t *>(w.minv.data()), packed.data());
        }
        Dev<uint8_t> dp(packed.size());
        if (panel) dp.put(packed);
        else {
            dq.put(w.q);
            ds.put(w.scale);
            dm.put(w.minv);
        }
        auto launch = [&] {
            if (bits == 4) {
                const strix::Q4DeviceView v = panel ? flerp::q4_panel_view(dp.p, N, K, G)
                                                    : strix::Q4DeviceView{dq.p, ds.p, dm.p, N, K, G};
                strix::kernels::linear_q4(dx.p, v, dy.p, M, act, out_act, stream);
            } else if (bits == 8) {
                const strix::Q8DeviceView v = panel ? flerp::q8_panel_view(dp.p, N, K, G)
                                                    : strix::Q8DeviceView{dq.p, ds.p, dm.p, N, K, G};
                strix::kernels::linear_q8(dx.p, v, dy.p, M, act, out_act, stream);
            } else if (bits == 5) {
                strix::kernels::linear_q5(dx.p, strix::Q5DeviceView{dq.p, ds.p, dm.p, N, K, G}, dy.p, M, act, out_act, stream);
            } else if (bits == 6) {
                strix::kernels::linear_q6(dx.p, strix::Q6DeviceView{dq.p, ds.p, dm.p, N, K, G}, dy.p, M, act, out_act, stream);
            } else {
                throw std::runtime_error("bits must be 4, 5, 6 or 8");
            }
        };
        o.i("bits", bits).i("N", N).i("K", K).i("G", G).i("M", M).s("act", c.str("act")).s("out", c.str("out"));
        o.s("layout", panel ? "panel" : "row");
        const double weight_bytes = (double)(w.q.size() + 2 * w.scale.size() + 2 * w.minv.size());
        if (c.str("kind") == "parity") {
            dy.fill_bytes(0x5a);
            launch();
            hip_ok(hipStreamSynchronize(stream), "sync");
            const std::vector<unsigned char> raw = dy.get();
            std::vector<float> got((size_t)(M * N)), want((size_t)(M * N));
            for (size_t i = 0; i < got.size(); ++i) {
                if (y_bf16) {
                    uint16_t b;
                    std::memcpy(&b, raw.data() + 2 * i, 2);
                    got[i] = bf16_to_f32(b);
                } else {
                    std::memcpy(&got[i], raw.data() + 4 * i, 4);
                }
            }
            const int64_t rb = reference::sw_row_bytes(bits, K), groups = K / G;
            double worst = 0, max_abs_exact = 0;
            for (int64_t m = 0; m < M; ++m)
                for (int64_t n = 0; n < N; ++n) {
                    const reference::SwExact e = reference::sw_exact(bits, w.q.data() + n * rb, w.scale.data() + n * groups,
                                                                     w.minv.data() + n * groups, x_f.data() + m * K, K, G);
                    const size_t j = (size_t)(m * N + n);
                    want[j] = y_bf16 ? bf16_to_f32(f32_to_bf16((float)e.sum)) : (float)e.sum;
                    double bound = order_bound(e.sum_abs, 2 * K, 4);
                    if (y_bf16) bound += 0.001953125 * (std::fabs(e.sum) + bound);    // one BF16 rounding on store: 2^-9
                    const double err = std::fabs((double)got[j] - e.sum);
                    max_abs_exact = std::max(max_abs_exact, err);
                    worst = std::max(worst, got[j] != got[j] ? INFINITY : err / bound);
                }
            put_diff(o, "", diff_f32(got.data(), want.data(), got.size()));
            o.d("err_over_bound", worst).d("max_abs_vs_exact", max_abs_exact);
            dump(dumps, c.name, got.data(), got.size() * 4);
        } else {
            const Timing t = time_launches(launch, stream, (int)c.num("iters"), (int)c.num("repeats"));
            put_timing(o, t, weight_bytes + (double)(M * K) * (x_bf16 ? 2 : 4) + (double)(M * N) * (y_bf16 ? 2 : 4), 1);
        }
    });
}
