// SPDX-License-Identifier: MIT
// CPU reference for the decode GEMV on packed GGUF blocks, y[m] = sum over k of W[m][k] * x[k], FP32 x and y.
//
// Weights, as ggml defines them (ggml-quants.c, dequantize_row_q8_0 / dequantize_row_q4_K):
//   Q8_0  block = { f16 d; int8 qs[32] }                         w[i] = d * qs[i]
//   Q4_K  block = { f16 d; f16 dmin; u8 scales[12]; u8 qs[128] } 256 weights in 8 sub-blocks of 32; sub-block j has
//         a 6-bit scale sc_j and a 6-bit min m_j packed in scales[] (get_scale_min_k4);
//         w = (d * sc_j) * q - (dmin * m_j), q the 4-bit code: low nibbles of qs[32 * (j / 2) ..] for even j,
//         high nibbles for odd j.
// exact(): the sum in double with the weights taken as real numbers (d * q, and d * sc * q - dmin * m, are exact
// in double). sum_abs is the sum of the magnitudes that any formulation adds up: |w x| for Q8_0, and
// |d sc q x| + |dmin m x| for Q4_K (a kernel may keep the scale part and the min part in separate sums).
// The order emulations follow the kernels' sources; see dense_bf16.hpp for what they are for.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

namespace reference {

inline float gg_half(const unsigned char *p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    const uint32_t sign = (uint32_t)(h >> 15) << 31, exp = (h >> 10) & 0x1f, frac = h & 0x3ff;
    uint32_t u;
    if (exp == 0) {
        const float v = std::ldexp((float)frac, -24);
        return sign ? -v : v;
    }
    u = exp == 31 ? (sign | 0x7f800000u | (frac << 13)) : (sign | ((exp + 112) << 23) | (frac << 13));
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct GgExact {
    double sum, sum_abs;
};

inline GgExact q8_0_exact(const unsigned char *row, const float *x, int64_t K) {
    GgExact e{0, 0};
    for (int64_t b = 0; b < K / 32; ++b) {
        const unsigned char *blk = row + b * 34;
        const double d = gg_half(blk);
        for (int i = 0; i < 32; ++i) {
            const double t = d * (double)(int8_t)blk[2 + i] * (double)x[b * 32 + i];
            e.sum += t, e.sum_abs += std::fabs(t);
        }
    }
    return e;
}

inline void q4k_scale_min(int j, const unsigned char *sc, int &scale, int &mn) {      // ggml get_scale_min_k4
    if (j < 4) {
        scale = sc[j] & 63, mn = sc[j + 4] & 63;
    } else {
        scale = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4), mn = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
    }
}

inline GgExact q4k_exact(const unsigned char *row, const float *x, int64_t K) {
    GgExact e{0, 0};
    for (int64_t b = 0; b < K / 256; ++b) {
        const unsigned char *blk = row + b * 144;
        const double d = gg_half(blk), dmin = gg_half(blk + 2);
        for (int j = 0; j < 8; ++j) {
            int sc, mn;
            q4k_scale_min(j, blk + 4, sc, mn);
            for (int l = 0; l < 32; ++l) {
                const int q = j & 1 ? blk[16 + 32 * (j / 2) + l] >> 4 : blk[16 + 32 * (j / 2) + l] & 0xF;
                const double xv = x[b * 256 + j * 32 + l];
                const double a = d * sc * q * xv, m = dmin * mn * xv;
                e.sum += a - m, e.sum_abs += std::fabs(a) + std::fabs(m);
            }
        }
    }
    return e;
}

inline float lane_tree(float v[32]) {      // lane 0 of `sum += shfl(sum, 16), 8, 4, 2, 1` (down or xor: the same for lane 0)
    for (int off = 16; off > 0; off >>= 1)
        for (int i = 0; i < off; ++i) {
            volatile float s = v[i] + v[i + off];
            v[i] = s;
        }
    return v[0];
}

inline float madd(float a, float b, float acc, bool fma) {
    if (fma) return std::fmaf(a, b, acc);
    volatile float p = a * b;
    volatile float s = acc + p;
    return s;
}

// hipfire kernels/src/gemv_q8_0.hip: thread t adds (d * q[t]) * x[t] over the row's blocks in order.
inline float q8_0_hipfire(const unsigned char *row, const float *x, int64_t K, bool fma) {
    float lane[32];
    for (int t = 0; t < 32; ++t) {
        float sum = 0.0f;
        for (int64_t b = 0; b < K / 32; ++b) {
            const unsigned char *blk = row + b * 34;
            volatile float w = gg_half(blk) * (float)(int8_t)blk[2 + t];
            sum = madd(w, x[b * 32 + t], sum, fma);
        }
        lane[t] = sum;
    }
    return lane_tree(lane);
}

// gufo gemv_quant.hip, the two-row kernel's Q8_0 branch: lane l takes blocks l, l + 32, ..; a block's 32
// products q * x are summed in order, the block sum is scaled by d and added to the lane's sum.
inline float q8_0_gufo_rows(const unsigned char *row, const float *x, int64_t K, bool fma_in, bool fma_out) {
    float lane[32];
    for (int l = 0; l < 32; ++l) {
        float sum = 0.0f;
        for (int64_t b = l; b < K / 32; b += 32) {
            const unsigned char *blk = row + b * 34;
            float dot = 0.0f;
            for (int i = 0; i < 32; ++i) dot = madd((float)(int8_t)blk[2 + i], x[b * 32 + i], dot, fma_in);
            sum = madd(gg_half(blk), dot, sum, fma_out);
        }
        lane[l] = sum;
    }
    return lane_tree(lane);
}

// gufo quant_ops.hpp, QuantWarpBlockDot's Q8_0 branch (the one-row kernel, and every odd row count): the whole
// warp walks the row's blocks in order; a block's 32 products q * x, one per lane, meet in a xor tree; the block
// sum is scaled by d and added to the running sum.
inline float q8_0_gufo_warp(const unsigned char *row, const float *x, int64_t K, bool fma) {
    float sum = 0.0f;
    for (int64_t b = 0; b < K / 32; ++b) {
        const unsigned char *blk = row + b * 34;
        float v[32];
        for (int l = 0; l < 32; ++l) {
            volatile float p = (float)(int8_t)blk[2 + l] * x[b * 32 + l];
            v[l] = p;
        }
        sum = madd(gg_half(blk), lane_tree(v), sum, fma);
    }
    return sum;
}

// gufo quant_ops.hpp, QuantWarpBlockDot's sub-16 route for Q4_K: lane l owns the 16-element sub-blocks l, l + 32,
// ..; a sub-block contributes scale * (sum of q * x) - offset * (sum of x), scale = d * sc and offset = dmin * m
// in fp32; the 32 lane sums meet in a xor tree. `form` is where the compiler may have fused:
//   0  t = fma(scale, dot, -(offset * xsum))   1  t = fma(-offset, xsum, scale * dot)   2  no fusion
inline float q4k_gufo_sub16(const unsigned char *row, const float *x, int64_t K, bool fma_dot, int form) {
    float lane[32];
    const int64_t subs = K / 16;
    for (int l = 0; l < 32; ++l) {
        float sum = 0.0f;
        for (int64_t sb = l; sb < subs; sb += 32) {
            const unsigned char *blk = row + (sb / 16) * 144;
            const int j = (int)((sb / 2) % 8), half = (int)(sb % 2);      // 32-wide sub-block j, its first or second 16
            int sc, mn;
            q4k_scale_min(j, blk + 4, sc, mn);
            volatile float scale = gg_half(blk) * (float)sc;
            volatile float offset = gg_half(blk + 2) * (float)mn;
            float dot = 0.0f, xsum = 0.0f;
            for (int i = 0; i < 16; ++i) {
                const unsigned char byte = blk[16 + 32 * (j / 2) + half * 16 + i];
                const int q = j & 1 ? byte >> 4 : byte & 0xF;
                const float xv = x[sb * 16 + i];
                dot = madd((float)q, xv, dot, fma_dot);
                volatile float xs = xsum + xv;
                xsum = xs;
            }
            float t;
            if (form == 0) {
                volatile float o = offset * xsum;
                t = std::fmaf(scale, dot, -o);
            } else if (form == 1) {
                volatile float a = scale * dot;
                t = std::fmaf(-offset, xsum, a);
            } else {
                volatile float a = scale * dot;
                volatile float o = offset * xsum;
                volatile float d = a - o;
                t = d;
            }
            volatile float s2 = sum + t;
            sum = s2;
        }
        lane[l] = sum;
    }
    return lane_tree(lane);
}

// hipfire kernels/src/gemv_q4k.hip: thread t takes, per block, sub-block j's element t for j = 0..7 in order:
// sum += ((d * sc_j) * q - (dmin * m_j)) * x.
inline float q4k_hipfire(const unsigned char *row, const float *x, int64_t K, bool fma) {
    float lane[32];
    for (int t = 0; t < 32; ++t) {
        float sum = 0.0f;
        for (int64_t b = 0; b < K / 256; ++b) {
            const unsigned char *blk = row + b * 144;
            const float d = gg_half(blk), dmin = gg_half(blk + 2);
            for (int j = 0; j < 8; ++j) {
                int sc, mn;
                q4k_scale_min(j, blk + 4, sc, mn);
                const int q = j & 1 ? blk[16 + 32 * (j / 2) + t] >> 4 : blk[16 + 32 * (j / 2) + t] & 0xF;
                volatile float s = d * (float)sc;
                volatile float m = dmin * (float)mn;
                volatile float w = std::fmaf(s, (float)q, -m);      // s * q is exact in fp32: fused or not, one rounding
                sum = madd(w, x[b * 256 + j * 32 + t], sum, fma);
            }
        }
        lane[t] = sum;
    }
    return lane_tree(lane);
}

}  // namespace reference
