// SPDX-License-Identifier: MIT
// CPU reference for logits.topk/-/-: the contract of strixite's kernels/logits_topk.hpp, written the slow way.
//   candidates of a row = its ids in [0, n_valid) that the mask allows and whose logit is not NaN,
//                         ordered by value descending, the lower id first when two values compare equal
//                         (-0 and +0 compare equal), the first 20 of them;
//                         fewer than 20 such ids: the rest is (-inf, INT32_MAX);
//   nan flag of a row   = 1 if any of its first n_valid logits is NaN, allowed by the mask or not.
// A logit of -inf or +inf is a value like any other.
#pragma once
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <vector>

namespace reference {

struct Cand {
    float v;
    int32_t id;
};

inline void logits_topk(const float *logits, int64_t rows, int64_t ld, int64_t n_valid, const uint32_t *mask,
                        int64_t mask_words, int keep, std::vector<Cand> &out, std::vector<uint32_t> &nan) {
    out.assign((size_t)(rows * keep), Cand{-INFINITY, INT32_MAX});
    nan.assign((size_t)rows, 0u);
    for (int64_t r = 0; r < rows; ++r) {
        const float *row = logits + r * ld;
        std::vector<Cand> all;
        for (int64_t id = 0; id < n_valid; ++id) {
            if (row[id] != row[id]) {
                nan[(size_t)r] = 1;
                continue;
            }
            if (mask && !((mask[r * mask_words + id / 32] >> (id % 32)) & 1u)) continue;
            all.push_back(Cand{row[id], (int32_t)id});
        }
        std::stable_sort(all.begin(), all.end(), [](const Cand &a, const Cand &b) { return a.v > b.v; });   // ids ascend within equal values
        for (int j = 0; j < keep && j < (int)all.size(); ++j) out[(size_t)(r * keep + j)] = all[(size_t)j];
    }
}

}  // namespace reference
