// SPDX-License-Identifier: MIT
// CPU reference for moe.router (strixite kernels/moe_router.hpp), without the shared slot:
//   per token, the k experts with the largest logits, largest first, an equal logit going to the lower index;
//   coef_j = exp(l_j - l_max) / sum over the k of exp(l_i - l_max).
//   A token with a non-finite logit: ids -1, coef NaN, and err = {1, token, lowest non-finite expert index}
//   (the reference records the lowest such token; the gate's cases have at most one).
// The weights are computed in double: they are the mathematically intended values to within 1e-15, and a
// provider's fp32 weights are judged by their relative distance from them.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace reference {

struct Routed {
    std::vector<int32_t> ids;       // [M, k]
    std::vector<double> coef;       // [M, k], NaN for a failed token
    uint32_t err[3] = {0, 0, 0};
};

inline Routed moe_router(const float *logits, int64_t ld, int64_t M, int64_t E, int64_t k) {
    Routed r;
    r.ids.assign((size_t)(M * k), -1);
    r.coef.assign((size_t)(M * k), std::nan(""));
    for (int64_t m = 0; m < M; ++m) {
        const float *row = logits + m * ld;
        int64_t bad = -1;
        for (int64_t e = 0; e < E && bad < 0; ++e)
            if (!std::isfinite(row[e])) bad = e;
        if (bad >= 0) {
            if (!r.err[0]) r.err[0] = 1, r.err[1] = (uint32_t)m, r.err[2] = (uint32_t)bad;
            continue;
        }
        std::vector<int32_t> order((size_t)E);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return row[a] > row[b]; });
        double sum = 0;
        for (int64_t j = 0; j < k; ++j) sum += std::exp((double)row[order[(size_t)j]] - (double)row[order[0]]);
        for (int64_t j = 0; j < k; ++j) {
            r.ids[(size_t)(m * k + j)] = order[(size_t)j];
            r.coef[(size_t)(m * k + j)] = std::exp((double)row[order[(size_t)j]] - (double)row[order[0]]) / sum;
        }
    }
    return r;
}

}  // namespace reference
