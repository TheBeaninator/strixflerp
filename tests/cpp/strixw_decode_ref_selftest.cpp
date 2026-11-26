// SPDX-License-Identifier: MIT
// Host self-test of the strixw-decode gate's CPU reference (gate/interface/reference/strixw_decode.hpp) against
// strixite's own format code (catalog/strixite/format/q4, q5, q6, q8: quantize_qN / dequantize_qN):
//   1. the reference's code extraction times scale plus min (q*s then +m, FP32, no contraction) gives every
//      dequantized weight of strixite's dequantize_qN bit for bit;
//   2. sw_put_code then sw_code returns every code (the driver's generator writes what the reference reads);
//   3. sw_exact's sum agrees with a double sum over strixite's dequantized weights to 1e-12 relative.
// Built and run by tests/test_strixw_decode_ref.py (g++ -ffp-contract=off). Prints one line per format; exit 0 = pass.
#include "formats/q4.hpp"
#include "formats/q5.hpp"
#include "formats/q6.hpp"
#include "formats/q8.hpp"
#include "strixw_decode.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

template <typename W>
static int check(int bits, const W &q, const std::vector<float> &deq) {
    const int64_t N = q.N, K = q.K, G = q.G, rb = reference::sw_row_bytes(bits, K);
    int bad = 0;
    if ((int64_t)q.q.size() != N * rb) { std::printf("q%d: row bytes %lld disagree with the format\n", bits, (long long)rb); return 1; }
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            const int c = reference::sw_code(bits, q.q.data() + n * rb, K, k);
            volatile float p = (float)c * reference::sw_bf(q.scale[(size_t)(n * (K / G) + k / G)]);
            const float v = p + reference::sw_bf(q.minv[(size_t)(n * (K / G) + k / G)]);
            uint32_t a, b;
            std::memcpy(&a, &v, 4);
            std::memcpy(&b, &deq[(size_t)(n * K + k)], 4);
            if (a != b && bad++ < 3) std::printf("q%d: weight [%lld,%lld] %.9g vs strixite %.9g\n", bits, (long long)n, (long long)k, v, deq[(size_t)(n * K + k)]);
        }
    std::vector<uint8_t> row((size_t)rb, 0);
    for (int64_t k = 0; k < K; ++k) reference::sw_put_code(bits, row.data(), K, k, reference::sw_code(bits, q.q.data(), K, k));
    if (std::memcmp(row.data(), q.q.data(), (size_t)rb) != 0) { std::printf("q%d: put/get round trip differs\n", bits); ++bad; }
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t)K);
    for (auto &v : x) v = nd(rng);
    for (int64_t n = 0; n < N; ++n) {
        double ref = 0;
        for (int64_t k = 0; k < K; ++k) ref += (double)x[(size_t)k] * (double)deq[(size_t)(n * K + k)];
        const reference::SwExact e = reference::sw_exact(bits, q.q.data() + n * rb, q.scale.data() + n * (K / G),
                                                         q.minv.data() + n * (K / G), x.data(), K, G);
        if (std::fabs(e.sum - ref) > 1e-12 * e.sum_abs && bad++ < 6) std::printf("q%d: row %lld sum %.17g vs %.17g\n", bits, (long long)n, e.sum, ref);
    }
    std::printf("q%d: N %lld K %lld G %lld: %s\n", bits, (long long)N, (long long)K, (long long)G, bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}

int main() {
    int fails = 0;
    for (int64_t G : {32, 64, 128}) {
        const int64_t N = 16, K = 512;
        std::mt19937 rng((unsigned)G);
        std::normal_distribution<float> nd(0.f, 0.02f);
        std::vector<float> w((size_t)(N * K));
        for (auto &v : w) v = nd(rng);
        fails += check(4, strix::quantize_q4(w.data(), N, K, G), strix::dequantize_q4(strix::quantize_q4(w.data(), N, K, G)));
        fails += check(5, strix::quantize_q5(w.data(), N, K, G), strix::dequantize_q5(strix::quantize_q5(w.data(), N, K, G)));
        fails += check(6, strix::quantize_q6(w.data(), N, K, G), strix::dequantize_q6(strix::quantize_q6(w.data(), N, K, G)));
        fails += check(8, strix::quantize_q8(w.data(), N, K, G), strix::dequantize_q8(strix::quantize_q8(w.data(), N, K, G)));
    }
    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
