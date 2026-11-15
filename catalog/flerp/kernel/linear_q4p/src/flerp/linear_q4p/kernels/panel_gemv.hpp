#pragma once

// kernel.flerp.linear_q4p: the decode GEMV of kernels/linear_q4.hip (Q4) and kernels/linear_q8.hip (Q8) on panels
// (runtime/panel_device.hpp), bit-equal to them. Included by linear_q4p.hip and kernel.flerp.linear_q8p's .hip only.
//
// The origin: a wave owns one output row (per split-K part ks of KS); lane o (0..31) takes the 16-byte chunks
// c = 32 ks + o + 32 KS j (j = 0, 1, ..) of that row in that order, accumulating acc = fmaf(s, dot, fmaf(m, xsum,
// acc)) per chunk, then a butterfly v += shfl_xor(v, off), off = 16, 8, 4, 2, 1, then the KS waves' totals summed in
// LDS from 0 in ks order. In a panel, one row's consecutive chunks are 256 bytes apart (16 rows' chunk c, then
// chunk c + 1), so that lane order would touch 32 cache lines a load.
//
// Here a row group of RW rows (8: half a panel, 16: a whole panel) has H = 32 / RW lanes per row: lane L = RW h + rr
// takes row rr and, in step t (t < RW) of each 32-chunk block, chunk 32 ks + 32 KS j + H t + h - origin lane
// o = H t + h. The RW lanes of one h read chunk c of RW consecutive rows (RW 8: one 128-byte line; RW 16: 256 bytes,
// and the two h's of a load make one 512-byte run, the origin's pattern). Each accumulator receives exactly the chunks
// one origin lane received, in the same order, through the same expression; the butterfly is replayed on o (PMap,
// p_tree, p_reduce below). Float addition commutes, so every value the origin forms is formed here with the same
// operands: the outputs are bit-equal, not merely close. The split-K factor and the LDS sum are the origin's.

#include "common/hip_check.hpp"
#include "kernels/device_info.hpp"
#include "kernels/norm.hpp"  // Act
#include "runtime/panel_device.hpp"

#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace strix::kernels {
namespace {

constexpr int kPThreads = 256;

// A wave map: RW rows a row group (8 = half a panel, a chunk column is one 128-byte line; 16 = a whole panel, two
// chunk columns are one 512-byte run), H = 32 / RW chunk columns a load, S = RW steps per 32-chunk block (origin lane
// o = H t + h, t < S), TPW of the steps a wave, NT = S / TPW waves per row group and split-K part, BT block threads.
template <int RW_, int TPW_, int BT_>
struct PMap {
    static constexpr int RW = RW_, H = 32 / RW_, S = RW_, TPW = TPW_, NT = RW_ / TPW_, BT = BT_;
};

__device__ inline float p_bf16_bits_to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ inline uint16_t p_f32_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
__device__ inline void p_store(float *p, int64_t i, float v) { p[i] = v; }
__device__ inline void p_store(uint16_t *p, int64_t i, float v) { p[i] = p_f32_to_bf16_bits(v); }

typedef __bf16 p_bf16x2 __attribute__((ext_vector_type(2)));
__device__ inline float p_dot2(uint32_t a, uint32_t b, float c) {
    return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(p_bf16x2, a), __builtin_bit_cast(p_bf16x2, b), c, false);
}
constexpr uint32_t kPOnePair = 0x3F803F80u;  // BF16 (1, 1)

// Codes 2p, 2p+1 of a chunk as a packed BF16 pair: linear_q4.hip / linear_q8.hip code_pair.
template <int BITS>
__device__ inline uint32_t p_code_pair(const uint32_t (&words)[4], int p) {
    if constexpr (BITS == 4) {
        const uint32_t b = (words[p / 4] >> (8 * (p % 4))) & 0xFFu;
        const float lo = (float)(b & 0xFu), hi = (float)(b >> 4);
        return (__float_as_uint(lo) >> 16) | (__float_as_uint(hi) & 0xffff0000u);
    } else {
        const uint32_t w = words[p / 2], sh = 16 * (p % 2);
        const float lo = (float)((w >> sh) & 0xFFu), hi = (float)((w >> (sh + 8)) & 0xFFu);
        return (__float_as_uint(lo) >> 16) | (__float_as_uint(hi) & 0xffff0000u);
    }
}
template <int BITS>
__device__ inline float p_code(const uint32_t (&words)[4], int i) {
    if constexpr (BITS == 4) return (float)((words[i / 8] >> (4 * (i % 8))) & 0xFu);
    else return (float)((words[i / 4] >> (8 * (i % 4))) & 0xFFu);
}

// FP32 values of x [off, off + n) (n = 32 or 16), as the origins' load_x32 / load_x16.
template <int NC>
__device__ inline void p_load_x(const float *x, int64_t off, float (&v)[NC]) {
    const float4 *p = reinterpret_cast<const float4 *>(x + off);
#pragma unroll
    for (int i = 0; i < NC / 4; ++i) {
        const float4 f = p[i];
        v[4 * i] = f.x, v[4 * i + 1] = f.y, v[4 * i + 2] = f.z, v[4 * i + 3] = f.w;
    }
}
template <int NC>
__device__ inline void p_load_x(const uint16_t *x, int64_t off, float (&v)[NC]) {
    const uint4 *p = reinterpret_cast<const uint4 *>(x + off);
#pragma unroll
    for (int i = 0; i < NC / 8; ++i) {
        const uint4 u = p[i];
        const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[8 * i + 2 * j] = __uint_as_float(w4[j] << 16);
            v[8 * i + 2 * j + 1] = __uint_as_float(w4[j] & 0xffff0000u);
        }
    }
}

// The origin's butterfly v += shfl_xor(v, off), off = 16 .. 1, on lane o = H t + h: offsets >= H flip bits of t, top
// bit first; offsets < H flip bits of h (here lane L ^ off RW). p_tree does the t-steps over an array of n values
// whose index bits are the next log2 n bits of t from the top: nv[i] = a[i] + a[i ^ f], f = n/2 .. 1.
template <int NV>
__device__ inline float p_tree(float (&a)[NV]) {
#pragma unroll
    for (int f = NV / 2; f >= 1; f >>= 1) {
        float nv[NV];
#pragma unroll
        for (int i = 0; i < NV; ++i) nv[i] = a[i] + a[i ^ f];
#pragma unroll
        for (int i = 0; i < NV; ++i) a[i] = nv[i];
    }
    return a[0];
}

// The wave map (decode2, gate/strixw-panel-decode2.toml). A row group's S steps are split over NT waves: wave tg's
// accumulator u is step t = tg + NT u, so a wave's accumulators differ in the top bits of t - the bits the butterfly
// sums first. With TPW = 1 a lane holds one accumulator per x row, the origin's register count, and the grid has the
// origin's N x KS waves. Every load instruction reads whole lines (RW 8: 4 lines of 128 B; RW 16: one 512-byte run,
// the origin's pattern). Reduction in the butterfly's order: each wave's top-bit steps in registers (p_tree), the
// next steps over the NT waves' values gathered in LDS by the tg = 0 wave, then the h-steps as shuffles, then the KS
// parts in ks order. Same operands, same order: bit-equal. Waves of a block are numbered (group gi, ks, tg), tg
// fastest; a block is max(8, NT KS) waves.
template <class MP, int MR>
struct PRed {
    static constexpr int W = MP::BT / 32;
    float red[MP::NT > 1 ? W / MP::NT * (MP::NT - 1) * MR * 32 : 1];  // the tg >= 1 waves' values
    float part[W * MR * MP::RW];
};

template <class MP>
struct PRole {
    int lane, wv, tg, ks, rr, h;
    int64_t group;
    __device__ explicit PRole(int KS) {
        lane = threadIdx.x % 32, wv = threadIdx.x / 32;
        const int WG = MP::NT * KS, gi = wv / WG;
        tg = wv % MP::NT, ks = (wv / MP::NT) % KS;
        rr = lane % MP::RW, h = lane / MP::RW;
        group = (int64_t)blockIdx.x * (MP::BT / 32 / WG) + gi;
    }
    // origin lane step of accumulator u
    __device__ int step(int u) const { return tg + MP::NT * u; }
    __device__ bool writer() const { return h == 0 && tg == 0 && ks == 0; }
};

// acc[r][u] -> tot[r], valid where R.writer(). Every thread of the block must call it (barriers).
template <class MP, int MR>
__device__ inline void p_reduce(float (&acc)[MR][MP::TPW], PRed<MP, MR> &sm, const PRole<MP> &R, int KS,
                                float (&tot)[MR]) {
    constexpr int NT = MP::NT;
    float d[MR];
#pragma unroll
    for (int r = 0; r < MR; ++r) d[r] = p_tree<MP::TPW>(acc[r]);  // steps on the top bits of t
    if constexpr (NT > 1) {
        const int base = R.wv / NT * (NT - 1);  // slot of this (gi, ks)'s tg = 1 wave
        if (R.tg > 0)
#pragma unroll
            for (int r = 0; r < MR; ++r) sm.red[((base + R.tg - 1) * MR + r) * 32 + R.lane] = d[r];
        __syncthreads();
        if (R.tg == 0) {
#pragma unroll
            for (int r = 0; r < MR; ++r) {
                float a[NT];
                a[0] = d[r];
#pragma unroll
                for (int t = 1; t < NT; ++t) a[t] = sm.red[((base + t - 1) * MR + r) * 32 + R.lane];
                d[r] = p_tree<NT>(a);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        float v = d[r];
#pragma unroll
        for (int off = MP::H / 2; off > 0; off >>= 1) v += __shfl_xor(v, off * MP::RW, 32);
        tot[r] = v;
    }
    if (KS == 1) return;
    if (R.tg == 0 && R.h == 0)
#pragma unroll
        for (int r = 0; r < MR; ++r) sm.part[(R.wv * MR + r) * MP::RW + R.rr] = tot[r];
    __syncthreads();
    if (R.writer())
#pragma unroll
        for (int r = 0; r < MR; ++r) {
            float v = 0.f;
            for (int i = 0; i < KS; ++i) v += sm.part[((R.wv + i * NT) * MR + r) * MP::RW + R.rr];
            tot[r] = v;
        }
}

// Dense GEMV, BITS 4 or 8 (chunk = 32 or 16 codes): y[M, N] = x[M, K] * deq(W)^T, W panels.
template <typename TX, typename TY, int BITS, int G, int MR, class MP>
__global__ void __launch_bounds__(MP::BT)
    panel_gemv_kernel(const TX *x, const uint8_t *w, TY *y, int64_t M, int64_t N, int64_t K, int KS) {
    constexpr int CH = BITS == 4 ? 32 : 16;  // codes per 16-byte chunk
    constexpr int TPW = MP::TPW;
    __shared__ PRed<MP, MR> sm;
    const PRole<MP> R(KS);
    const int64_t n = R.group * MP::RW + R.rr;
    const int64_t m0 = (int64_t)blockIdx.y * MR;
    const int rows = (int)((M - m0) < MR ? (M - m0) : MR);
    const int64_t P = (BITS == 4 ? 8 : 16) * K + 64 * K / G, codes = (BITS == 4 ? 8 : 16) * K;
    const int64_t chunks = K / CH;

    float acc[MR][TPW];
#pragma unroll
    for (int r = 0; r < MR; ++r)
#pragma unroll
        for (int u = 0; u < TPW; ++u) acc[r][u] = 0.f;

    if (n < N) {  // no early return: every wave reaches the barriers below
        for (int64_t cb = (int64_t)R.ks * 32; cb < chunks; cb += (int64_t)KS * 32) {
#pragma unroll
            for (int u = 0; u < TPW; ++u) {
                const int64_t c = cb + MP::H * R.step(u) + R.h;
                if (c >= chunks) continue;
                const int64_t k = c * CH;
                const uint4 packed = *reinterpret_cast<const uint4 *>(flerp::panel_chunk(w, P, n, c));
                const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
                const uint32_t smv = flerp::panel_sm(w, P, codes, n, k / G);
                const float s = p_bf16_bits_to_f32((uint16_t)(smv & 0xffffu)), mn = p_bf16_bits_to_f32((uint16_t)(smv >> 16));
                if constexpr (std::is_same_v<TX, uint16_t>) {
                    uint32_t cp[CH / 2];
#pragma unroll
                    for (int p = 0; p < CH / 2; ++p) cp[p] = p_code_pair<BITS>(words, p);
#pragma unroll
                    for (int r = 0; r < MR; ++r) {
                        if (r < rows) {
                            const uint4 *xp = reinterpret_cast<const uint4 *>(x + (m0 + r) * K + k);
                            float dot = 0.f, xsum = 0.f;
                            if constexpr (BITS == 4) {
#pragma unroll
                                for (int i = 0; i < CH / 8; ++i) {
                                    const uint4 uu = xp[i];
                                    const uint32_t xw[4] = {uu.x, uu.y, uu.z, uu.w};
#pragma unroll
                                    for (int j = 0; j < 4; ++j) {
                                        dot = p_dot2(xw[j], cp[4 * i + j], dot);
                                        xsum = p_dot2(xw[j], kPOnePair, xsum);
                                    }
                                }
                            } else {
                                const uint4 u0 = xp[0], u1 = xp[1];
                                const uint32_t xw[CH / 2] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
#pragma unroll
                                for (int p = 0; p < CH / 2; ++p) {
                                    dot = p_dot2(xw[p], cp[p], dot);
                                    xsum = p_dot2(xw[p], kPOnePair, xsum);
                                }
                            }
                            acc[r][u] = fmaf(s, dot, fmaf(mn, xsum, acc[r][u]));
                        }
                    }
                    continue;
                }
#pragma unroll
                for (int r = 0; r < MR; ++r) {
                    if (r < rows) {
                        float xv[CH];
                        p_load_x<CH>(x, (m0 + r) * K + k, xv);
                        float dot = 0.f, xsum = 0.f;
#pragma unroll
                        for (int i = 0; i < CH; ++i) {
                            dot = fmaf(xv[i], p_code<BITS>(words, i), dot);
                            xsum += xv[i];
                        }
                        acc[r][u] = fmaf(s, dot, fmaf(mn, xsum, acc[r][u]));
                    }
                }
            }
        }
    }
    float tot[MR];
    p_reduce<MP, MR>(acc, sm, R, KS, tot);
    if (R.writer() && n < N)
        for (int r = 0; r < rows; ++r) p_store(y, (m0 + r) * N + n, tot[r]);
}

// The map's block: max(8, NT KS) waves, at most 1024 threads.
inline int p_block_threads(int RW, int TPW, int KS) {
    const int wg = (RW / TPW) * KS;
    return 32 * (wg > 8 ? wg : 8);
}
inline bool p_map_ok(int RW, int TPW, int KS) {
    // the instantiated maps (p_with_map)
    return (RW == 8 || RW == 16) && TPW >= 1 && TPW <= (RW == 8 ? 8 : 4) && (TPW & (TPW - 1)) == 0 && KS >= 1 &&
           KS <= 8 && (KS & (KS - 1)) == 0 && (RW / TPW) * KS <= 32;
}

struct PChoice {
    int rw, tpw;
};
// FLERP_PANEL_RW=8|16 and FLERP_PANEL_TPW override the table below (the decode2 sweep); unset or invalid: the table.
inline int p_env(const char *name) {
    const char *e = std::getenv(name);
    return e ? std::atoi(e) : 0;
}
// The chosen map (gate/results/strixw-panel-decode2/sweep.md, the sweep over RW 8 / 16 and TPW 1 / 2 / 4 on every
// fleet class shape): whole panels (RW 16), whose loads are the origin's 512-byte runs; one step a wave (TPW 1: the
// origin's waves and registers) for Q4 with K >= 2048, the expert kernels and Q8 with few rows (N <= 2048: sh_gu at
// TPW 2 ran 105-113% at M 4-8, too few waves), two steps a wave for the other Q8, four for short-K Q4 (fewer, longer waves
// measured 1-4% faster there). Where the block would pass 1024 threads (NT KS > 32) TPW doubles.
inline PChoice p_map(int KS, int MR, int bits, int64_t K, int64_t N, bool experts = false) {
    if (MR > 8) return {8, 8};  // the M > 8 pass: not decode; the committed map
    static const int erw = p_env("FLERP_PANEL_RW"), etpw = p_env("FLERP_PANEL_TPW");
    PChoice c;
    if (erw || etpw) c = {erw ? erw : 16, etpw ? etpw : 1};
    else if (experts) c = {16, 1};
    else if (bits == 4) c = {16, K >= 2048 ? 1 : 4};
    else c = {16, N <= 2048 ? 1 : 2};
    while (!p_map_ok(c.rw, c.tpw, KS) && c.tpw < 8) c.tpw *= 2;
    if (!p_map_ok(c.rw, c.tpw, KS)) c = {8, 8};
    return c;
}

// Calls f(PMap<RW, TPW, BT>{}) for a valid map.
template <typename F>
void p_with_map(PChoice c, int KS, F &&f) {
    const int bt = p_block_threads(c.rw, c.tpw, KS);
#define FLERP_PMAP(RW, TPW)                                                       \
    if (c.rw == RW && c.tpw == TPW) {                                             \
        if (bt == 256) return f(PMap<RW, TPW, 256>{});                            \
        if constexpr (RW / TPW >= 2) if (bt == 512) return f(PMap<RW, TPW, 512>{}); \
        if constexpr (RW / TPW >= 4) return f(PMap<RW, TPW, 1024>{});            \
    }
    FLERP_PMAP(8, 1) FLERP_PMAP(8, 2) FLERP_PMAP(8, 4) FLERP_PMAP(8, 8) FLERP_PMAP(16, 1) FLERP_PMAP(16, 2) FLERP_PMAP(16, 4)
#undef FLERP_PMAP
    return f(PMap<8, 8, 256>{});
}

template <typename TX, typename TY, int BITS, int MR, class MP>
void panel_gemv_go(const TX *x, const flerp::PanelView &w, TY *y, int64_t M, int KS, hipStream_t stream) {
    const int64_t groups_per_block = MP::BT / 32 / (MP::NT * KS), groups = (w.N + MP::RW - 1) / MP::RW;
    const dim3 grid((unsigned)((groups + groups_per_block - 1) / groups_per_block), (unsigned)((M + MR - 1) / MR)),
        block(MP::BT);
    switch (w.G) {
        case 32: panel_gemv_kernel<TX, TY, BITS, 32, MR, MP><<<grid, block, 0, stream>>>(x, w.p, y, M, w.N, w.K, KS); break;
        case 64: panel_gemv_kernel<TX, TY, BITS, 64, MR, MP><<<grid, block, 0, stream>>>(x, w.p, y, M, w.N, w.K, KS); break;
        case 128: panel_gemv_kernel<TX, TY, BITS, 128, MR, MP><<<grid, block, 0, stream>>>(x, w.p, y, M, w.N, w.K, KS); break;
        default: STRIX_FAIL("panel GEMV: group size ", w.G, " unsupported (32, 64 or 128)");
    }
}

template <typename TX, typename TY, int BITS, int MR>
void panel_gemv_launch_mr(const TX *x, const flerp::PanelView &w, TY *y, int64_t M, int KS, hipStream_t stream) {
    STRIX_CHECK(KS >= 1 && KS <= 8 && (KS & (KS - 1)) == 0, "panel GEMV: split-K factor ", KS, ", expected 1, 2, 4 or 8");
    if constexpr (MR > 8) {
        panel_gemv_go<TX, TY, BITS, MR, PMap<8, 8, 256>>(x, w, y, M, KS, stream);
    } else {
        p_with_map(p_map(KS, MR, BITS, w.K, w.N), KS, [&](auto mp) { panel_gemv_go<TX, TY, BITS, MR, decltype(mp)>(x, w, y, M, KS, stream); });
    }
}

// The origins' M -> rows-per-pass choice (linear_q4.hip / linear_q8.hip launch).
template <typename TX, typename TY, int BITS>
void panel_gemv_launch(const TX *x, const flerp::PanelView &w, TY *y, int64_t M, int KS, hipStream_t stream) {
    if (M == 1) panel_gemv_launch_mr<TX, TY, BITS, 1>(x, w, y, M, KS, stream);
    else if (M == 2) panel_gemv_launch_mr<TX, TY, BITS, 2>(x, w, y, M, KS, stream);
    else if (M <= 4) panel_gemv_launch_mr<TX, TY, BITS, 4>(x, w, y, M, KS, stream);
    else if (M <= 8) panel_gemv_launch_mr<TX, TY, BITS, 8>(x, w, y, M, KS, stream);
    else panel_gemv_launch_mr<TX, TY, BITS, 16>(x, w, y, M, KS, stream);
}

// The checks of the origins' linear_q4 / linear_q8 entry, on a panel view.
inline void panel_check(const char *fn, const void *x, const flerp::PanelView &w, const void *y, int64_t M, Act act,
                        Act out_act, int64_t max_m) {
    STRIX_CHECK(x != nullptr && y != nullptr && w.p != nullptr, fn, ": null pointer (x=", x, ", y=", y, ", panels=",
                (const void *)w.p, ")");
    STRIX_CHECK(w.G == 32 || w.G == 64 || w.G == 128, fn, ": group size ", w.G, " unsupported (32, 64 or 128)");
    STRIX_CHECK(M >= 1 && M <= max_m, fn, ": M (rows of x) = ", M, ", expected 1..", max_m);
    STRIX_CHECK(w.N >= 1 && w.N <= 0x7fffffff, fn, ": N = ", w.N, ", expected 1..2^31-1");
    STRIX_CHECK(w.K >= w.G && w.K <= (1 << 24) && w.K % w.G == 0 && w.K % 32 == 0, fn, ": K = ", w.K,
                " must be a positive multiple of 32 and of group size ", w.G, ", <= 2^24");
    STRIX_CHECK(reinterpret_cast<uintptr_t>(w.p) % 16 == 0, fn, ": panels ", (const void *)w.p, " must be 16-byte aligned");
    STRIX_CHECK(reinterpret_cast<uintptr_t>(x) % 16 == 0, fn, ": x ", x, " must be 16-byte aligned for vector loads");
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, fn, ": unsupported activation dtype ", (int)act);
    STRIX_CHECK(out_act == act || (act == Act::BF16 && out_act == Act::F32), fn, ": output dtype ", act_name(out_act),
                " with input ", act_name(act), " unsupported (same dtype, or BF16 -> F32)");
    STRIX_CHECK(device_warp_size(kPThreads) == 32, fn, ": the panel kernels are written for wave32 (gfx11)");
    const size_t es = act == Act::F32 ? 4 : 2, oes = out_act == Act::F32 ? 4 : 2;
    const auto xb = reinterpret_cast<uintptr_t>(x), yb = reinterpret_cast<uintptr_t>(y);
    STRIX_CHECK(yb + (uintptr_t)(M * w.N) * oes <= xb || xb + (uintptr_t)(M * w.K) * es <= yb, fn, ": output y [", y,
                ", +", M * w.N * (int64_t)oes, " B) overlaps input x [", x, ", +", M * w.K * (int64_t)es, " B)");
}

template <int BITS>
void panel_gemv(const char *fn, const void *x, const flerp::PanelView &w, void *y, int64_t M, Act act, Act out_act,
                int KS, hipStream_t stream) {
    STRIX_CHECK(w.bits == BITS, fn, ": a ", w.bits, "-bit panel view, expected ", BITS);
    panel_check(fn, x, w, y, M, act, out_act, 65535ll * 8);
    if (act == Act::F32)
        panel_gemv_launch<float, float, BITS>(static_cast<const float *>(x), w, static_cast<float *>(y), M, KS, stream);
    else if (out_act == Act::F32)
        panel_gemv_launch<uint16_t, float, BITS>(static_cast<const uint16_t *>(x), w, static_cast<float *>(y), M, KS, stream);
    else
        panel_gemv_launch<uint16_t, uint16_t, BITS>(static_cast<const uint16_t *>(x), w, static_cast<uint16_t *>(y), M, KS,
                                                   stream);
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), "->", act_name(out_act), " M=", M, " N=", w.N, " K=", w.K, " G=", w.G);
}

}  // namespace
}  // namespace strix::kernels
