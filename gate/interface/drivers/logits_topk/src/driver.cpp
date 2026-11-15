// SPDX-License-Identifier: MIT
// Interface gate driver for logits.topk/-/-, through strixite's launch API:
//   strix::kernels::logits_topk(logits, rows, ld, n_valid, out, nan, workspace, workspace_bytes, stream, mask, mask_words)
#include "flerp_gate/gate_common.hpp"
#include "kernels/logits_topk.hpp"
#include "logits_topk.hpp"

using namespace gate;
using strix::kernels::kLogitCands;
using strix::kernels::LogitCand;

static void fill(std::vector<float> &lg, int64_t rows, int64_t ld, int64_t n_valid, uint64_t seed, const std::string &pattern) {
    Rng r(seed);
    for (int64_t row = 0; row < rows; ++row) {
        float *p = lg.data() + row * ld;
        for (int64_t i = 0; i < ld; ++i) {
            float v = r.gauss(4.0f);
            if (pattern == "ties") v = std::round(v * 2.0f) / 2.0f;                       // 0.5 steps: the top is all ties
            if (pattern == "signed-zero") v = v > 1.0f ? (r.next() & 1 ? 0.0f : -0.0f) : -std::fabs(v) - 1.0f;   // top = zeros of both signs
            if (pattern == "neg-inf-masked") v = r.next() % 400 == 0 ? v : -INFINITY;     // ~600 finite logits a row
            if (pattern == "few-finite") v = r.next() % 40000 == 0 ? v : -INFINITY;       // ~6 finite logits a row
            if (i >= n_valid) v = 1.0e9f + (float)i;                                      // the padding must never win
            p[i] = v;
        }
        if (pattern == "nan") p[(size_t)(r.next() % (uint64_t)n_valid)] = NAN;
        if (pattern == "nan-few-finite") {
            for (int64_t i = 0; i < n_valid; ++i) p[i] = r.next() % 40000 == 0 ? p[i] : -INFINITY;
            p[3] = NAN;      // a low id: where gufo's -inf substitute would be listed
        }
        if (pattern == "pos-inf") p[(size_t)(r.next() % (uint64_t)n_valid)] = INFINITY;
    }
}

int main(int argc, char **argv) {
    return run_driver(argc, argv, "logits_topk", [](const Case &c, Obj &o, hipStream_t stream, const std::string &dumps) {
        const int64_t rows = c.num("rows"), n_valid = c.num("n_valid"), ld = c.num("ld", n_valid);
        const std::string pattern = c.str("pattern", "gauss");
        const bool masked = c.num("mask", 0) != 0;
        const int64_t mask_words = masked ? (n_valid + 31) / 32 : 0;
        std::vector<float> lg((size_t)(rows * ld));
        fill(lg, rows, ld, n_valid, (uint64_t)c.num("seed", 1), pattern);
        std::vector<uint32_t> mask;
        if (masked) {      // about one id in three allowed
            Rng r((uint64_t)c.num("seed", 1) + 99);
            mask.resize((size_t)(rows * mask_words));
            for (auto &w : mask) w = (uint32_t)r.next() & (uint32_t)r.next() & (uint32_t)(r.next() | r.next());
        }
        Dev<float> dl(lg.size());
        dl.put(lg);
        Dev<uint32_t> dm(mask.size());
        if (masked) dm.put(mask);
        const size_t ws_bytes = strix::kernels::logits_topk_workspace_bytes(rows, n_valid);
        Dev<unsigned char> ws(ws_bytes);
        Dev<LogitCand> out((size_t)(rows * kLogitCands));
        Dev<uint32_t> nan((size_t)rows);
        auto launch = [&] {
            strix::kernels::logits_topk(dl.p, rows, ld, n_valid, out.p, nan.p, ws.p, ws_bytes, stream,
                                        masked ? dm.p : nullptr, mask_words);
        };
        o.i("rows", rows).i("n_valid", n_valid).i("ld", ld).i("workspace_bytes", (long long)ws_bytes);
        if (c.str("kind") == "parity") {
            out.fill_bytes(0x5a), nan.fill_bytes(0x5a);
            launch();
            hip_ok(hipStreamSynchronize(stream), "sync");
            const std::vector<LogitCand> got = out.get();
            const std::vector<uint32_t> got_nan = nan.get();
            std::vector<reference::Cand> want;
            std::vector<uint32_t> want_nan;
            reference::logits_topk(lg.data(), rows, ld, n_valid, masked ? mask.data() : nullptr, mask_words, kLogitCands, want, want_nan);
            long long bad = 0, bad_ids = 0, bad_flags = 0;
            std::string first;
            for (size_t j = 0; j < got.size(); ++j) {
                const bool same = f32_bits(got[j].v) == f32_bits(want[j].v) && got[j].id == want[j].id;
                if (!same && first.empty()) {
                    char buf[160];
                    std::snprintf(buf, sizeof buf, "row %zu slot %zu: got (%g, %d), reference (%g, %d)", j / kLogitCands,
                                  j % kLogitCands, (double)got[j].v, got[j].id, (double)want[j].v, want[j].id);
                    first = buf;
                }
                bad += !same;
                bad_ids += got[j].id != want[j].id;
            }
            for (size_t r = 0; r < got_nan.size(); ++r) bad_flags += got_nan[r] != want_nan[r];
            o.b("bit_identical", bad == 0 && bad_flags == 0).i("mismatched", bad).i("ids_differing", bad_ids);
            o.i("nan_flags_differing", bad_flags).b("structure_ok", bad_flags == 0);
            if (!first.empty()) o.s("detail", first);
            std::vector<unsigned char> raw(got.size() * sizeof(LogitCand) + got_nan.size() * 4);
            std::memcpy(raw.data(), got.data(), got.size() * sizeof(LogitCand));
            std::memcpy(raw.data() + got.size() * sizeof(LogitCand), got_nan.data(), got_nan.size() * 4);
            dump(dumps, c.name, raw.data(), raw.size());
        } else {
            const Timing t = time_launches(launch, stream, (int)c.num("iters"), (int)c.num("repeats"));
            put_timing(o, t, 4.0 * (double)(rows * n_valid), (int)c.num("launches", 0));
        }
    });
}
