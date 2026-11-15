// SPDX-License-Identifier: MIT
// Interface gate driver for the BF16-weight dense GEMV, BF16 in and FP32 out, through strixite's launch API:
//   strix::kernels::linear_bf16w(x, w, y, M, N, K, Act::BF16, Act::F32, stream)
#include "flerp_gate/gate_common.hpp"
#include "kernels/linear.hpp"
#include "dense_bf16.hpp"

using namespace gate;
using strix::kernels::Act;

static void fill(std::vector<uint16_t> &x, std::vector<uint16_t> &w, uint64_t seed, const std::string &pattern) {
    Rng r(seed);
    for (auto &v : x) v = f32_to_bf16(r.gauss(1.0f));                  // a normalized activation row
    for (auto &v : w) v = f32_to_bf16(r.gauss(0.02f));                 // router weights
    if (pattern == "cancel") {      // big terms of both signs that almost cancel: the order matters most here
        for (size_t i = 0; i < x.size(); ++i) x[i] = f32_to_bf16((i & 1 ? -1.0f : 1.0f) * (60.0f + r.gauss(1.0f)));
        for (auto &v : w) v = f32_to_bf16(1.0f + r.gauss(0.01f));
    }
    if (pattern == "wide-range")    // magnitudes over 12 decades in one row
        for (auto &v : x) v = f32_to_bf16(r.gauss(1.0f) * std::pow(10.0f, (float)(r.uniform() * 12.0 - 6.0)));
}

int main(int argc, char **argv) {
    return run_driver(argc, argv, "dense_bf16", [](const Case &c, Obj &o, hipStream_t stream, const std::string &dumps) {
        const int64_t M = c.num("M"), N = c.num("N"), K = c.num("K");
        const int64_t w_shift = c.num("w_shift", 0);      // elements: 1 makes w 2 bytes off a 16-byte boundary
        std::vector<uint16_t> x((size_t)(M * K)), w((size_t)(N * K));
        fill(x, w, (uint64_t)c.num("seed", 1), c.str("pattern", "gauss"));
        Dev<uint16_t> dx(x.size()), dw(w.size() + 8);
        Dev<float> dy((size_t)(M * N));
        dx.put(x);
        hip_ok(hipMemcpy(dw.p + w_shift, w.data(), w.size() * 2, hipMemcpyHostToDevice), "upload w");
        auto launch = [&] { strix::kernels::linear_bf16w(dx.p, dw.p + w_shift, dy.p, M, N, K, Act::BF16, Act::F32, stream); };
        o.i("M", M).i("N", N).i("K", K);
        if (c.str("kind") == "parity") {
            dy.fill_bytes(0x5a);
            launch();
            hip_ok(hipStreamSynchronize(stream), "sync");
            const std::vector<float> got = dy.get();
            std::vector<float> want(got.size());
            long long seq = 0, sf = 0, sx = 0;      // outputs whose bits equal each order emulation
            double worst = 0, max_abs_exact = 0;
            for (int64_t m = 0; m < M; ++m)
                for (int64_t n = 0; n < N; ++n) {
                    const uint16_t *xr = x.data() + m * K, *wr = w.data() + n * K;
                    const reference::Exact e = reference::dense_exact(xr, wr, K);
                    const size_t j = (size_t)(m * N + n);
                    want[j] = (float)e.sum;
                    const double err = std::fabs((double)got[j] - e.sum);
                    max_abs_exact = std::max(max_abs_exact, err);
                    worst = std::max(worst, err / order_bound(e.sum_abs, K, 0));
                    seq += f32_bits(got[j]) == f32_bits(reference::dense_sequential(xr, wr, K));
                    sf += f32_bits(got[j]) == f32_bits(reference::dense_strixfold(xr, wr, K));
                    sx += f32_bits(got[j]) == f32_bits(reference::dense_strixite_rows(xr, wr, K));
                }
            put_diff(o, "", diff_f32(got.data(), want.data(), got.size()));
            const long long all = (long long)got.size();
            o.d("err_over_bound", worst).d("max_abs_vs_exact", max_abs_exact);
            o.i("equal_to_sequential_order", seq).i("equal_to_strixfold_order", sf).i("equal_to_strixite_rows_order", sx);
            o.i("outputs", all);
            o.s("order_emulation", sx == all && sf == all ? "strixite-rows = strixfold-lanes (the same order at this K)"
                                   : sx == all ? "strixite-rows" : sf == all ? "strixfold-lanes"
                                   : seq == all ? "sequential" : "none");
            dump(dumps, c.name, got.data(), got.size() * 4);
        } else {
            const Timing t = time_launches(launch, stream, (int)c.num("iters"), (int)c.num("repeats"));
            put_timing(o, t, 2.0 * (double)(N * K) + 2.0 * (double)(M * K) + 4.0 * (double)(M * N), 1);
        }
    });
}
