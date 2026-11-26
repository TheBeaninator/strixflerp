// SPDX-License-Identifier: MIT
// CPU reference for the strixw decode GEMV (linear.decode/strixw-q4|q5|q6|q8, strixite's formats/q4.hpp, q5.hpp,
// q6.hpp and q8.hpp):  y[m][n] = sum over k of x[m][k] * w[n][k],  w[n][k] = q[n][k] * scale[n][k/G] + min[n][k/G],
// scale and min BF16, q an unsigned 4/5/6/8-bit code, x F32 or BF16 (widened exactly), y F32 or BF16.
//
// exact()   the sum in double over the real-valued weights. q * s is exact in double (an 8-bit code times an
//           8-significant-bit BF16), so is + m for the scale/min ranges the gate draws, and each x * w term carries
//           at most one double rounding; a double sum of a few thousand of them is far inside one FP32 rounding.
//           Its rounding to FP32 (or BF16) is the reference value.
// sum_abs   the sum of |x q s| + |x m|: the kernels compute s * sum(x q) + m * sum(x) per chunk (formats' comments,
//           kernels/linear_q4.hpp), so these are the terms an any-order bound is taken over (gate_common.hpp
//           order_bound with 2K terms and 4 roundings inside a term: x*q, the chunk sums, s*. and m*.).
// The code extraction below follows the formats' own comments, byte for byte; it is the part a format change breaks.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace reference {

inline float sw_bf(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// Bytes of one row's codes and the largest code, per format bit width (4, 5, 6, 8).
inline int64_t sw_row_bytes(int bits, int64_t K) {
    return bits == 4 ? K / 2 : bits == 5 ? 5 * K / 8 : bits == 6 ? 3 * K / 4 : K;
}
inline int sw_max_code(int bits) { return (1 << bits) - 1; }

// The code of weight k in a row's code bytes.
//   Q4  low nibble = even k (byte k/2)
//   Q5  low plane [K/2] as Q4, then high plane [K/8]: bit 4 of code k at bit k%8 of byte k/8
//   Q6  low plane [K/2] as Q4, then high plane [K/4]: bits 4..5 of code k at bits 2*(k%4) of byte k/4
//   Q8  one byte per code
inline int sw_code(int bits, const uint8_t *row, int64_t K, int64_t k) {
    if (bits == 8) return row[k];
    const int lo = (row[k >> 1] >> ((k & 1) * 4)) & 15;
    if (bits == 4) return lo;
    const uint8_t *hi = row + K / 2;
    if (bits == 5) return lo | (((hi[k >> 3] >> (k & 7)) & 1) << 4);
    return lo | (((hi[k >> 2] >> (2 * (k & 3))) & 3) << 4);
}

// The inverse, for the driver's generator: writes code c of weight k into a zeroed row.
inline void sw_put_code(int bits, uint8_t *row, int64_t K, int64_t k, int c) {
    if (bits == 8) { row[k] = (uint8_t)c; return; }
    row[k >> 1] |= (uint8_t)((c & 15) << ((k & 1) * 4));
    if (bits == 4) return;
    uint8_t *hi = row + K / 2;
    if (bits == 5) hi[k >> 3] |= (uint8_t)(((c >> 4) & 1) << (k & 7));
    else hi[k >> 2] |= (uint8_t)(((c >> 4) & 3) << (2 * (k & 3)));
}

struct SwExact {
    double sum, sum_abs;
};

// One output: row n of W (codes row, its K/G scales and mins) against one activation row x (F32 values).
inline SwExact sw_exact(int bits, const uint8_t *row, const uint16_t *scale, const uint16_t *minv, const float *x,
                        int64_t K, int64_t G) {
    double s = 0, a = 0;
    for (int64_t k = 0; k < K; ++k) {
        const double sc = sw_bf(scale[k / G]), mn = sw_bf(minv[k / G]), xv = x[k];
        const double q = sw_code(bits, row, K, k);
        s += xv * (q * sc + mn);
        a += std::fabs(xv * q * sc) + std::fabs(xv * mn);
    }
    return {s, a};
}

}  // namespace reference
