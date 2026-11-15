// SPDX-License-Identifier: MIT
// CPU reference for the BF16-weight dense GEMV with BF16 input and FP32 output (the MoE router logits):
//   y[m][n] = sum over k of x[m][k] * w[n][k],  x [M, K] and w [N, K] BF16, y [M, N] FP32.
//
// exact()      the sum in double. A BF16 value has 8 significant bits, so every product is exact in FP32 and in
//              double, and a double sum of a few thousand of them is far inside one FP32 rounding step. Its
//              rounding to FP32 is the reference value; `sum_abs` is the sum of |terms| the error bound needs.
// The three orders below are FP32 emulations of what each kernel's source says it does, element for element.
// They are not the reference: they exist to answer "is the summation order the only difference?" - if a
// provider's bits equal an emulation's on every output, then yes, and that order is the one it runs.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace reference {

inline float bf(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct Exact {
    double sum, sum_abs;
};

inline Exact dense_exact(const uint16_t *x, const uint16_t *w, int64_t K) {
    Exact e{0.0, 0.0};
    for (int64_t k = 0; k < K; ++k) {
        const double t = (double)bf(x[k]) * (double)bf(w[k]);
        e.sum += t;
        e.sum_abs += std::fabs(t);
    }
    return e;
}

// 32 lane partials -> one value, as `v += shfl_xor(v, 16), 8, 4, 2, 1` does on every lane at once.
inline float xor_tree(float v[32]) {
    for (int off = 16; off > 0; off >>= 1) {
        float n[32];
        for (int i = 0; i < 32; ++i) {
            volatile float s = v[i] + v[i ^ off];
            n[i] = s;
        }
        std::memcpy(v, n, sizeof n);
    }
    return v[0];
}

// One element at a time, k ascending, multiply then add, no fma: the textbook loop.
inline float dense_sequential(const uint16_t *x, const uint16_t *w, int64_t K) {
    float acc = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        volatile float p = bf(x[k]) * bf(w[k]);
        volatile float s = acc + p;
        acc = s;
    }
    return acc;
}

// StrixFold zig/kernels/hip/dsv41/dense.hip: tiles of 32 * E elements (E = 8 when K % 256 == 0, else 4); lane l
// folds elements l * E .. l * E + E - 1 of every tile, in order, with one fmaf each; then the xor tree.
inline float dense_strixfold(const uint16_t *x, const uint16_t *w, int64_t K) {
    const int E = K % 256 == 0 ? 8 : 4, tile = 32 * E;
    float acc[32];
    for (int l = 0; l < 32; ++l) {
        float a = 0.0f;
        for (int64_t t = 0; t < K; t += tile)
            for (int j = 0; j < E; ++j) a = std::fmaf(bf(w[t + l * E + j]), bf(x[t + l * E + j]), a);
        acc[l] = a;
    }
    return xor_tree(acc);
}

// strixite kernels/linear.hip, rows path (K % 8 == 0, aligned): KS waves per output (KS doubles while
// K > 5 * 256 * KS, at most 8); lane l of wave ks folds the 8 elements at ((ks * 32 + l) + i * KS * 32) * 8,
// i = 0, 1, ..., with one fmaf each; xor tree per wave; then the KS wave sums are added in order, from 0.
inline int strixite_split_k(int64_t K) {
    int ks = 1;
    while (ks < 8 && K > 5 * 256 * (int64_t)ks) ks *= 2;
    return ks;
}
inline float dense_strixite_rows(const uint16_t *x, const uint16_t *w, int64_t K) {
    const int KS = strixite_split_k(K);
    float v = 0.0f;
    for (int ks = 0; ks < KS; ++ks) {
        float acc[32];
        for (int l = 0; l < 32; ++l) {
            float a = 0.0f;
            for (int64_t k = ((int64_t)ks * 32 + l) * 8; k < K; k += (int64_t)KS * 32 * 8)
                for (int j = 0; j < 8; ++j) a = std::fmaf(bf(x[k + j]), bf(w[k + j]), a);
            acc[l] = a;
        }
        volatile float s = v + xor_tree(acc);
        v = s;
    }
    return v;
}

}  // namespace reference
