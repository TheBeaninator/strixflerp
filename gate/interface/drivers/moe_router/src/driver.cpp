// SPDX-License-Identifier: MIT
// Interface gate driver for moe.router (FP32 logits, no shared slot: how the pod's Qwen3.8-Flash-Next weights are
// routed), through strixite's launch API:
//   strix::kernels::moe_router(logits, ld, M, E, k, nullptr, 0, 0, ids, coef, Act::F32, err, stream)
#include "flerp_gate/gate_common.hpp"
#include "kernels/moe_router.hpp"
#include "moe_router.hpp"

using namespace gate;
using strix::kernels::Act;

static void fill(std::vector<float> &lg, int64_t M, int64_t ld, int64_t E, uint64_t seed, const std::string &pattern) {
    Rng r(seed);
    for (int64_t m = 0; m < M; ++m) {
        float *row = lg.data() + m * ld;
        for (int64_t e = 0; e < ld; ++e) {
            float v = r.gauss(2.0f);
            if (pattern == "exact-ties") v = std::round(v * 4.0f) / 4.0f;      // steps of 0.25: the top k is full of ties
            if (e >= E) v = 1.0e9f;                                            // the gate logit's column: never an expert
            row[e] = v;
        }
        if (pattern == "near-ties") {      // the top logits one ulp apart, the larger one at the HIGHER index
            for (int j = 0; j < 12; ++j) row[(size_t)(40 * j + 7)] = bits_f32(f32_bits(9.0f) + (uint32_t)j);
        }
        if (pattern == "boundary-tie") {   // 16 equal top logits across threads: which 10 (or 8) win is the tie rule
            const int idx[16] = {5, 260, 300, 44, 511, 256, 31, 287, 100, 356, 2, 258, 129, 385, 70, 326};
            for (int j : idx) row[j] = 11.0f;
        }
    }
    if (pattern == "nan-token") lg[(size_t)((M / 2) * ld + 77)] = NAN;
    if (pattern == "inf-token") lg[(size_t)((M / 2) * ld + 300)] = INFINITY;
}

int main(int argc, char **argv) {
    return run_driver(argc, argv, "moe_router", [](const Case &c, Obj &o, hipStream_t stream, const std::string &dumps) {
        const int64_t M = c.num("M"), E = c.num("E", 512), k = c.num("k"), ld = c.num("ld", E + 1);
        std::vector<float> lg((size_t)(M * ld));
        fill(lg, M, ld, E, (uint64_t)c.num("seed", 1), c.str("pattern", "gauss"));
        Dev<float> dl(lg.size());
        dl.put(lg);
        Dev<int32_t> ids((size_t)(M * k));
        Dev<float> coef((size_t)(M * k));
        Dev<uint32_t> err(4);
        auto launch = [&] {
            strix::kernels::moe_router(dl.p, ld, M, E, k, nullptr, 0, 0, ids.p, coef.p, Act::F32, err.p, stream);
        };
        o.i("M", M).i("E", E).i("k", k).i("ld", ld);
        if (c.str("kind") == "parity") {
            ids.fill_bytes(0x5a), coef.fill_bytes(0x5a), err.fill_bytes(0);
            launch();
            hip_ok(hipStreamSynchronize(stream), "sync");
            const std::vector<int32_t> gi = ids.get();
            const std::vector<float> gc = coef.get();
            const std::vector<uint32_t> ge = err.get();
            const reference::Routed want = reference::moe_router(lg.data(), ld, M, E, k);
            const double tol = (double)c.num("coef_tol_ulps", 32) * 5.9604644775390625e-08;
            long long bad_ids = 0, bad_sets = 0, bad_nan = 0, exact = 0, compared = 0;
            double max_rel = 0;
            std::string first;
            for (int64_t m = 0; m < M; ++m) {
                std::vector<int32_t> a(gi.begin() + m * k, gi.begin() + (m + 1) * k), b(want.ids.begin() + m * k, want.ids.begin() + (m + 1) * k);
                bool row_same = a == b;
                if (!row_same && first.empty()) {
                    first = "token " + std::to_string(m) + ": got";
                    for (auto v : a) first += " " + std::to_string(v);
                    first += ", reference";
                    for (auto v : b) first += " " + std::to_string(v);
                }
                bad_ids += !row_same;
                std::sort(a.begin(), a.end()), std::sort(b.begin(), b.end());
                bad_sets += a != b;
                for (int64_t j = 0; j < k; ++j) {
                    const size_t q = (size_t)(m * k + j);
                    const float g = gc[q];
                    if (std::isnan(want.coef[q])) { bad_nan += !(g != g); continue; }
                    if (!row_same) continue;      // weights of a different selection are not comparable slot by slot
                    ++compared;
                    exact += f32_bits(g) == f32_bits((float)want.coef[q]);
                    max_rel = std::max(max_rel, std::fabs((double)g - want.coef[q]) / want.coef[q]);
                }
            }
            const bool err_same = ge[0] == want.err[0] && (!want.err[0] || (ge[1] == want.err[1] && ge[2] == want.err[2]));
            o.i("tokens", M).i("tokens_ids_differ", bad_ids).i("tokens_set_differs", bad_sets).i("nan_coef_missing", bad_nan);
            o.b("err_word_equal", err_same).i("coef_compared", compared).i("coef_correctly_rounded", exact);
            o.d("max_rel", max_rel).d("max_abs", 0).d("err_over_bound", max_rel / tol);
            o.b("structure_ok", bad_ids == 0 && bad_nan == 0 && err_same);
            o.b("bit_identical", bad_ids == 0 && bad_nan == 0 && err_same && exact == compared).i("mismatched", bad_ids + (compared - exact));
            if (!first.empty()) o.s("detail", first);
            std::vector<unsigned char> raw(gi.size() * 4 + gc.size() * 4 + 12);
            std::memcpy(raw.data(), gi.data(), gi.size() * 4);
            std::memcpy(raw.data() + gi.size() * 4, gc.data(), gc.size() * 4);
            std::memcpy(raw.data() + gi.size() * 8, ge.data(), 12);
            dump(dumps, c.name, raw.data(), raw.size());
        } else {
            err.fill_bytes(0);
            const Timing t = time_launches(launch, stream, (int)c.num("iters"), (int)c.num("repeats"));
            put_timing(o, t, 4.0 * (double)(M * E) + 8.0 * (double)(M * k), 0);
        }
    });
}
