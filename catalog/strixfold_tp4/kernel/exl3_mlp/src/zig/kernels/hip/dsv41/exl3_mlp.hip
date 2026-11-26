/* exl3.glinear_mlp: exl3.hip's decode glinear with a compile-time prefetch depth. Same arguments, same grid, same
 * kernel specialisation (k2, warps). exl3.hip is not included and is not modified: every symbol here is new.
 *
 * ACCUMULATE ORDER, kept verbatim from exl3.hip (the lines marked below name the source). A per-output sum is:
 *   strip_tiles_1row dot2 chain, exl3.hip 310-319, then the half-wave add 321-325;
 *   strip_rows_part dot2 chain, exl3.hip 371-387, then the half-wave add 392-407;
 *   glinear_body wave add 655-676 (wave 0, then 1 .. WK-1) and split add 687-695 (split 0, then 1 .. SK-1);
 *   kt0, exl3.hip 616: split * (per_warp * WK) + wave * per_warp, per_warp = (K/16) / SK / WK.
 * The dot2 chain is one copy. The k loop is not unrolled across depth, so a deeper ring does not duplicate it.
 * Depth only changes when loads issue and where a tile waits. lds_fence still waits lgkmcnt, not vmcnt. Loads go
 * through gl() (address space 1). Trellis loads stay nontemporal. Activation loads stay cached, as in exl3.hip.
 * The compiler may still drain vmcnt before a use; the dots then match exl3.hip.
 *
 * Template arguments of glinear_mlp_body / the strip functions:
 *   DEPTH    k tiles of trellis words in flight, 2..4
 *   ACT      activation tiles in flight: 1, or DEPTH
 *   STAGING  0 register ring (each in-flight tile stays in VGPRs across the dot of the older one)
 *            1 LDS ring (landed tiles sit in LDS; the copy into a slot is a reused dword, not a DEPTH-wide ring).
 *            gfx1151 has no global_load_lds, so this is not an async copy. It is the low-register schedule.
 * The k loop steps one tile. Slot j % DEPTH holds tile j. Order of tiles, dots, half-wave, wave add and split add
 * does not depend on DEPTH, ACT or STAGING.
 *
 * Kit symbols, one per (k2, wk), picked from the TheRock 10.2 scan (0 scratch, and occupancy >= 6 where a
 * register schedule can do it). Register family tf_ds_exl3_glinear_mlp_k*_w*:
 *   k4 w4/w8     depth 4, one activation tile
 *   k6/k8 w4/w8  depth 2, two activation tiles
 *   k10/k12      depth 2, one activation tile
 *   k16 w4       depth 2, one activation tile, amdgpu_waves_per_eu(6) (unpinned is 241 VGPRs, occupancy 5)
 *   k16 w8       depth 2, two activation tiles (250 VGPRs, occupancy 5: the pin spills, so it stays unpinned)
 * LDS family tf_ds_exl3_glinear_mlp_lds_k*_w*, one activation tile, trellis words parked in LDS:
 *   k4/k6/k8 and k10 w4   depth 4
 *   k10 w8, k12, k16      depth 2
 * TF_DS_MLP_SCAN (never defined by the kit build, not in registry.json) also instantiates
 * tf_ds_exl3_glinear_mlp_scan_k{K}_w{W}_{d2,d3,d4,d4a1,d2a1e6,lds2,lds4,lds2e6}. lds4 is omitted when the ring
 * plus the 8-row reduction would pass 64KB (k10/k12/k16 with 8 warps). d2a1e6 and lds2e6 pin amdgpu_waves_per_eu(6).
 *
 * Host: TF_DS_TP4_FILL=exact launches the register symbols, =exact_lds the LDS symbols, for a 2D node's wq_b,
 * wo_a and wo_b at 1..16 rows. The plan's sk and wk are exl3.hip's. Default off. M >= 64 stays on exl3.hip's
 * glinear (glinear_wide is a different schedule).
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include "abi.h"

namespace {

typedef __attribute__((__vector_size__(16 * sizeof(__fp16)))) __fp16 f16x16;
typedef __attribute__((__vector_size__(8 * sizeof(float)))) float f32x8;
typedef _Float16 h2 __attribute__((ext_vector_type(2)));

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)
enum DType : int { F16 = 0, BF16 = 1, F32 = 2 };

// ---- conversions, loads and stores ---------------------------------------------------------------------------------

// clang 23 / HIP 7.15 compiles fp16(a * b) of fp32 a, b as v_fma_mixlo_f16 / v_fma_mixhi_f16: ONE rounding of the exact
// product to fp16 where the source (and the CUDA build) rounds to fp32 first, then to fp16; the two differ on fp16 ties
// (measured: rot_out 1 ulp off on 1 of ~8k values). An fp32 value made opaque first keeps both roundings.
__device__ __forceinline__ float f32_value(float x) {
    asm volatile("" : "+v"(x));
    return x;
}
__device__ __forceinline__ __half f2h(float f) { return __float2half_rn(f32_value(f)); }
__device__ __forceinline__ float bf16_bits_to_f(uint32_t b) { return __uint_as_float(b << 16); }
__device__ __forceinline__ uint32_t f_to_bf16_bits(float f) { return __bfloat16_as_ushort(__float2bfloat16(f32_value(f))); }
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16(f32_value(x))); }
__device__ __forceinline__ uint32_t f_to_h_bits(float f) { return __half_as_ushort(f2h(f)); }

__device__ __forceinline__ void load4(const void* p, int dtype, size_t i, float (&v)[4]) {
    if (dtype == F32) {
        const float4 u = *reinterpret_cast<const float4*>(static_cast<const float*>(p) + i);
        v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
    } else {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const uint16_t*>(p) + i);
        if (dtype == BF16) {
            v[0] = bf16_bits_to_f(u.x & 0xffffu); v[1] = bf16_bits_to_f(u.x >> 16);
            v[2] = bf16_bits_to_f(u.y & 0xffffu); v[3] = bf16_bits_to_f(u.y >> 16);
        } else {
            v[0] = __half2float(__ushort_as_half(u.x & 0xffffu)); v[1] = __half2float(__ushort_as_half(u.x >> 16));
            v[2] = __half2float(__ushort_as_half(u.y & 0xffffu)); v[3] = __half2float(__ushort_as_half(u.y >> 16));
        }
    }
}

__device__ __forceinline__ void store4(void* p, int dtype, size_t i, const float (&v)[4]) {
    if (dtype == F32) {
        *reinterpret_cast<float4*>(static_cast<float*>(p) + i) = make_float4(v[0], v[1], v[2], v[3]);
    } else if (dtype == BF16) {
        *reinterpret_cast<uint2*>(static_cast<uint16_t*>(p) + i) =
            make_uint2(f_to_bf16_bits(v[0]) | f_to_bf16_bits(v[1]) << 16, f_to_bf16_bits(v[2]) | f_to_bf16_bits(v[3]) << 16);
    } else {
        *reinterpret_cast<uint2*>(static_cast<uint16_t*>(p) + i) =
            make_uint2(f_to_h_bits(v[0]) | f_to_h_bits(v[1]) << 16, f_to_h_bits(v[2]) | f_to_h_bits(v[3]) << 16);
    }
}

// A device address as a global (address space 1) pointer. Pointers made from the ABI's u64 fields or from the experts'
// pointer tables are generic, and generic loads compile to flat_load, which counts in lgkmcnt as well as vmcnt: every
// s_waitcnt lgkmcnt(0) before an LDS read would then also wait for the next step's weight prefetch (found by w-pf).
template <typename T>
__device__ __forceinline__ const __attribute__((address_space(1))) T* gl(const T* p) {
    return (const __attribute__((address_space(1))) T*)p;
}

// a coherent (L2) load and the acquire / release pair of the decode experts' readiness flags
__device__ __forceinline__ float ld_cg(const float* p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
__device__ __forceinline__ int ld_acquire(const int* p) { return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT); }
__device__ __forceinline__ void st_release(int* p, int v) { __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT); }

// ---- 128-point Walsh-Hadamard, the input rotation and the output epilogue ------------------------------------------

// Unscaled Walsh-Hadamard transform of 128 fp32 values, 4 a lane (elements 4 lane + j), in decode.cuh's fixed butterfly
// order (multiply by 1 / sqrt(128) after).
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    const float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor(v[j], m, 32);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

// One row's 128 rotated inputs ((x * suh) @ H) / sqrt(128) in fp32 (linear_common.cuh rot128_in: the first butterfly
// takes the products as fma(x0, s0, +-(x1 s1)) and fma(x2, s2, +-(x3 s3))).
__device__ __forceinline__ void rot128_in(float (&v)[4], const float (&s)[4], int lane) {
    const float p1 = v[1] * s[1], p3 = v[3] * s[3];
    const float a = fmaf(v[0], s[0], p1), b = fmaf(v[0], s[0], -p1);
    const float c = fmaf(v[2], s[2], p3), d = fmaf(v[2], s[2], -p3);
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor(v[j], m, 32);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = v[j] * HAD_SCALE;
}

// The finished outputs of one row's 128 columns from their fp32 sums (linear_common.cuh finish): H / sqrt(128), * svh,
// + bias (the last product and the bias as one fma).
__device__ __forceinline__ void finish(float (&v)[4], int lane, const __half* svh, const __half* bias, int col) {
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float t = v[j] * HAD_SCALE, sv = __half2float(svh[col + j]);
        v[j] = bias ? fmaf(t, sv, __half2float(bias[col + j])) : t * sv;
    }
}

// ---- the tile decode -------------------------------------------------------------------------------------------------

// E(p): one past the last stream bit of value p's 16-bit window (format.py stream_ends; decode.cuh stream_end).
template <int K2>
__host__ __device__ constexpr int stream_end(int p) {
    return (K2 & 1) ? ((p + 1) * K2 - ((p + 1) & 1)) / 2 : (p + 1) * (K2 / 2);
}

// A lane (column c = lane & 15, k half h = lane >> 4) owns the value pairs p0 = 32 (c % 8) + 8 q + 4 (c / 8) + 2 h,
// p0 + 1 (q = 0..3): rows 8 h + 2 q and 8 h + 2 q + 1 of column c. The pair's two windows lie in the 64 bits of words
// hi, lo (lo = the word after hi, wrapping), starting o bits into hi; the second window starts D bits after the first.
// hi, lo, sh: the 32 stream bits from the first window's start are alignbit(w[hi], w[lo], sh) (lo = hi and sh = 0
// when the window starts a word: v_alignbit_b32 takes its shift mod 32).
template <int K2>
struct LaneWin {
    static constexpr int TW = 4 * K2;
    static constexpr int D = stream_end<K2>(1) - stream_end<K2>(0);
    int hi[4], lo[4], sh[4];
    __device__ __forceinline__ explicit LaneWin(int lane) {
        const int c = lane & 15, h = lane >> 4;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int p0 = 32 * (c & 7) + 8 * q + 4 * (c >> 3) + 2 * h;
            const int s = (stream_end<K2>(p0) - 16 + 128 * K2) % (128 * K2);
            const int o = s & 31;
            hi[q] = s >> 5;
            lo[q] = o ? (hi[q] + 1) % TW : hi[q];
            sh[q] = (32 - o) & 31;
        }
    }
};

// x * 0x83DCD12D mod 2^32 of a 16-bit state. EXL3_MUL24=1: two 24-bit multiplies and a shift-add instead of
// v_mul_lo_u32 (measured: +1-5% at 1 row, -2-10% at 6 rows from its extra registers; off by default).
__device__ __forceinline__ uint32_t mul1_hash(uint32_t w) {
#if defined(EXL3_MUL24) && EXL3_MUL24
    // the barrier keeps the compiler from folding the shifted product back into a v_mul_lo_u32
    uint32_t b = __umul24(w, 0x83DCu);
    asm("" : "+v"(b));
    return __umul24(w, 0xD12Du) + (b << 16);
#else
    return w * 0x83DCD12Du;
#endif
}

// Two consecutive values from the 32 stream bits u at the first window's start -> packed fp16 pair (low half the
// first): MUL1's exact decode, fp16(0x6400 + bytesum(w * 0x83DCD12D)) * fp16(0x1EEE) + fp16(0xC931) in one fused fp16
// multiply-add.
template <int D>
__device__ __forceinline__ uint32_t decode_pair(uint32_t u) {
    const uint32_t w0 = u >> 16;
    const uint32_t w1 = __builtin_amdgcn_ubfe(u, 16 - D, 16);
    const uint32_t s0 = __builtin_amdgcn_sad_u8(mul1_hash(w0), 0u, 0x64006400u);   // low half: 0x6400 + bytesum
    const uint32_t s = __builtin_amdgcn_sad_hi_u8(mul1_hash(w1), 0u, s0);           // high half: the second's
    h2 hv = __builtin_bit_cast(h2, s);  // bitcast-ok: a whole u32 to f16x2, no lane extract
    hv = __builtin_elementwise_fma(hv, __builtin_bit_cast(h2, 0x1eee1eeeu), __builtin_bit_cast(h2, 0xc931c931u));  // bitcast-ok: whole-vector constants
    return __builtin_bit_cast(uint32_t, hv);  // bitcast-ok: a whole f16x2 to u32, no lane extract
}

// This lane's 8 values of the tile at `tw` (words in LDS or global memory): k rows 8 h .. 8 h + 7 of column c, as four
// packed pairs (rows 8 h + 2 q, 8 h + 2 q + 1).
template <int K2>
__device__ __forceinline__ void decode_own(const uint32_t* tw, const LaneWin<K2>& w, uint32_t (&own)[4]) {
#pragma unroll
    for (int q = 0; q < 4; ++q) own[q] = decode_pair<LaneWin<K2>::D>(__builtin_amdgcn_alignbit(tw[w.hi[q]], tw[w.lo[q]], w.sh[q]));
}

// The tile's B fragment for column lane & 15: this lane's eight k (8 h .. 8 h + 7) from its own decode, then its
// partner's (lane ^ 16, by v_permlanex16). Each half-wave computes its own output rows from its own copies of A and
// B, so a half's k order is free as long as A follows it: lanes 16-31 hold k 8-15 first, and their A fragments are
// loaded with the two 8-k halves swapped (load16h_k, a_swap = 1), which spares the half selects.
__device__ __forceinline__ f16x16 bfrag_from_own(const uint32_t (&own)[4], int lane) {
    (void)lane;
    uint32_t bw[8];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        bw[q] = own[q];
        bw[4 + q] = __builtin_amdgcn_permlanex16(own[q], own[q], 0x76543210u, 0xfedcba98u, false, false);
    }
    f16x16 b;
    __builtin_memcpy(&b, bw, 32);
    return b;
}

__device__ __forceinline__ f16x16 frag8(const uint4& lo, const uint4& hi) {
    const uint32_t w[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    f16x16 a;
    __builtin_memcpy(&a, w, 32);
    return a;
}

// 16 contiguous halves (32-byte aligned) as two 16-byte loads; zero for a null row
__device__ __forceinline__ void load16h(const _Float16* p, uint4& lo, uint4& hi) {
    if (p) {
        const auto q = gl(reinterpret_cast<const uint4*>(p));
        lo = q[0];
        hi = q[1];
    } else {
        lo = hi = make_uint4(0u, 0u, 0u, 0u);
    }
}

// The same for an A fragment against bfrag_from_own's B: lanes 16-31 (sw = 1) take k 8-15 first
__device__ __forceinline__ void load16h_k(const _Float16* p, int sw, uint4& lo, uint4& hi) {
    if (p) {
        const auto q = gl(reinterpret_cast<const uint4*>(p));
        lo = q[sw];
        hi = q[sw ^ 1];
    } else {
        lo = hi = make_uint4(0u, 0u, 0u, 0u);
    }
}

__device__ __forceinline__ f32x8 wmma(const f16x16& a, const f16x16& b, const f32x8& c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ void lds_fence() {
    __builtin_amdgcn_wave_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}

// One wave's k tiles [kt0, kt0 + n) of a 128-column strip (eight 16 x 16 tiles a k step, contiguous; steps `step` words
// apart) against a single row (xrow: 16 halves a k tile), without the matrix unit. A k step's words are loaded
// coalesced one step ahead into registers, staged in LDS (st: the wave's, 8 * 4 K2 words), and each tile's windows read
// back from there. A lane's four decoded pairs (k rows 8 h + 2 q, + 1 of column c) meet the row's matching input
// pairs in v_dot2_f32_f16; the two half-waves' partial sums are added at the end. acc[t][0] of lanes 0-15 then holds
// output column 16 t + c (row 0 of WMMA's C layout); the other elements are left as they were (zero).
__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c) {
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(h2, a), __builtin_bit_cast(h2, b), c, false);  // bitcast-ok: whole u32 to f16x2
}


// Register ring (STAGING 0) or LDS ring (STAGING 1). DEPTH tiles of trellis words, ACT activation tiles.
// The dot2 chain below is one copy of exl3.hip's; the k loop is not unrolled across DEPTH.
enum { TF_DS_MLP_REG = 0, TF_DS_MLP_LDS = 1 };

template <int RL, int PW>
__device__ __forceinline__ void mlp_lds_fill(uint32_t* st, const uint32_t* p, int lane) {
#pragma unroll 1
    for (int i = 0; i < RL; ++i) {
        const int wi = i * 32 + lane;
        if (wi < PW) st[wi] = __builtin_nontemporal_load(gl(p) + wi);
    }
}

__device__ __forceinline__ uint4 mlp_act_1row(const _Float16* xrow, int kt, int h) {
    return *gl(reinterpret_cast<const uint4*>(xrow + 16 * kt + 8 * h));
}

__device__ __forceinline__ uint4 mlp_act_row(const uint4* xl, bool xlive, int kt) {
    uint4 a = make_uint4(0, 0, 0, 0);
    if (xlive) a = xl[2 * kt];
    return a;
}

// One wave's k tiles. wb is this wave's LDS: one tile (register staging) or DEPTH tiles (LDS staging).
template <int K2, int DEPTH, int ACT, int STAGING>
__device__ __forceinline__ void strip_tiles_1row(const uint32_t* tiles, int64_t step, int kt0, int n, const _Float16* xrow,
                                                 uint32_t* wb, int lane, f32x8 (&acc)[8]) {
    static_assert(DEPTH >= 2 && DEPTH <= 4, "depth 2..4");
    static_assert(ACT == 1 || ACT == DEPTH, "act is 1 or DEPTH");
    static_assert(STAGING == TF_DS_MLP_REG || STAGING == TF_DS_MLP_LDS, "staging");
    static_assert(STAGING == TF_DS_MLP_REG || ACT == 1, "an LDS ring keeps one activation tile");
    constexpr int TW = 4 * K2, LW = 8 * TW / 32, TILE = 8 * TW;
    static_assert(TILE == LW * 32, "1-row tile");
    if (n <= 0) return;
    const LaneWin<K2> win(lane);
    const int h = lane >> 4;
    float s[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) s[t] = 0.f;
    uint4 act[ACT];
#pragma unroll
    for (int d = 0; d < ACT; ++d) act[d] = make_uint4(0, 0, 0, 0);

    if constexpr (STAGING == TF_DS_MLP_REG) {
        uint32_t pf[DEPTH][LW];
#pragma unroll
        for (int d = 0; d < DEPTH; ++d) {
            if (d < n) {
                const uint32_t* p = tiles + (int64_t)(kt0 + d) * step;
#pragma unroll
                for (int i = 0; i < LW; ++i) pf[d][i] = __builtin_nontemporal_load(gl(p) + i * 32 + lane);
                if constexpr (ACT == DEPTH) act[d] = mlp_act_1row(xrow, kt0 + d, h);
            }
        }
        if constexpr (ACT == 1) {
            if (n > 0) act[0] = mlp_act_1row(xrow, kt0, h);
        }
#define TF_DS_MLP_REG_SLOT(S_)                                                                                          \
        do {                                                                                                            \
            _Pragma("unroll") for (int i_ = 0; i_ < LW; ++i_) st[i_ * 32 + lane] = pf[S_][i_];                         \
            if constexpr (ACT == DEPTH) x = act[S_];                                                                    \
            if (j + DEPTH < n) {                                                                                        \
                const uint32_t* p_ = tiles + (int64_t)(kt0 + j + DEPTH) * step;                                         \
                _Pragma("unroll") for (int i_ = 0; i_ < LW; ++i_)                                                       \
                    pf[S_][i_] = __builtin_nontemporal_load(gl(p_) + i_ * 32 + lane);                                   \
                if constexpr (ACT == DEPTH) act[S_] = mlp_act_1row(xrow, kt0 + j + DEPTH, h);                           \
            }                                                                                                           \
        } while (0)
#pragma unroll 1
        for (int j = 0; j < n; ++j) {
            const int slot = j - (j / DEPTH) * DEPTH;
            uint32_t* st = wb;
            uint4 x = make_uint4(0, 0, 0, 0);
            if (slot == 0) {
                TF_DS_MLP_REG_SLOT(0);
            } else if constexpr (DEPTH > 1) {
                if (slot == 1) TF_DS_MLP_REG_SLOT(1);
                else if constexpr (DEPTH > 2) {
                    if (slot == 2) TF_DS_MLP_REG_SLOT(2);
                    else if constexpr (DEPTH > 3) {
                        if (slot == 3) TF_DS_MLP_REG_SLOT(3);
                    }
                }
            }
            if constexpr (ACT == 1) {
                x = act[0];
                if (j + 1 < n) act[0] = mlp_act_1row(xrow, kt0 + j + 1, h);
            }
            lds_fence();
            // ACCUMULATE ORDER verbatim from exl3.hip strip_tiles_1row lines 310-319. One copy for every depth.
            // Tile j is k tile kt0 + j; j runs 0 .. n-1. The half-wave add (lines 321-325) stays after this k loop.
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                uint32_t own[4];
                decode_own<K2>(st + t * TW, win, own);
                float a = dot2(own[0], x.x, s[t]);
                a = dot2(own[1], x.y, a);
                a = dot2(own[2], x.z, a);
                s[t] = dot2(own[3], x.w, a);
            }
            __builtin_amdgcn_wave_barrier();
        }
#undef TF_DS_MLP_REG_SLOT
    } else {
#pragma unroll 1
        for (int d = 0; d < DEPTH; ++d)
            if (d < n) mlp_lds_fill<LW, TILE>(wb + d * TILE, tiles + (int64_t)(kt0 + d) * step, lane);
        if (n > 0) act[0] = mlp_act_1row(xrow, kt0, h);
#pragma unroll 1
        for (int j = 0; j < n; ++j) {
            const int slot = j - (j / DEPTH) * DEPTH;
            uint32_t* st = wb + slot * TILE;
            uint4 x = act[0];
            if (j + 1 < n) act[0] = mlp_act_1row(xrow, kt0 + j + 1, h);
            lds_fence();
            // ACCUMULATE ORDER verbatim from exl3.hip strip_tiles_1row lines 310-319. One copy for every depth.
            // Tile j is k tile kt0 + j; j runs 0 .. n-1. The half-wave add (lines 321-325) stays after this k loop.
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                uint32_t own[4];
                decode_own<K2>(st + t * TW, win, own);
                float a = dot2(own[0], x.x, s[t]);
                a = dot2(own[1], x.y, a);
                a = dot2(own[2], x.z, a);
                s[t] = dot2(own[3], x.w, a);
            }
            __builtin_amdgcn_wave_barrier();
            if (j + DEPTH < n) mlp_lds_fill<LW, TILE>(st, tiles + (int64_t)(kt0 + j + DEPTH) * step, lane);
        }
    }
    // ACCUMULATE ORDER verbatim from exl3.hip strip_tiles_1row lines 321-325 (half-wave add, after the k loop).
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const float o = __uint_as_float(__builtin_amdgcn_permlanex16(__float_as_uint(s[t]), __float_as_uint(s[t]), 0x76543210u, 0xfedcba98u, false, false));
        acc[t][0] += s[t] + o;
    }
}

template <int K2, int RM, int TT, int t0, int DEPTH, int ACT, int STAGING>
__device__ __forceinline__ void strip_rows_part(const uint32_t* tiles, int64_t step, int kt0, int n, const _Float16* xlane,
                                                int R, uint32_t* wb, uint4* xst, int lane, f32x8 (&acc)[8]) {
    static_assert(DEPTH >= 2 && DEPTH <= 4, "depth 2..4");
    static_assert(ACT == 1 || ACT == DEPTH, "act is 1 or DEPTH");
    static_assert(STAGING == TF_DS_MLP_REG || ACT == 1, "an LDS ring keeps one activation tile");
    constexpr int TW = 4 * K2, PW = TT * TW, RL = (PW + 31) / 32, TILE = 8 * TW;
    const LaneWin<K2> win(lane);
    const int h = lane >> 4;
    const bool xlive = (lane >> 1) < R;
    const uint4* xl = gl(reinterpret_cast<const uint4*>(xlane + 8 * (lane & 1)));
    const uint32_t* tp = tiles + t0 * TW;
    float s[RM][TT];
#pragma unroll
    for (int r = 0; r < RM; ++r)
#pragma unroll
        for (int t = 0; t < TT; ++t) s[r][t] = 0.f;
    uint4 act[ACT];
#pragma unroll
    for (int d = 0; d < ACT; ++d) act[d] = make_uint4(0, 0, 0, 0);

    if constexpr (STAGING == TF_DS_MLP_REG) {
        uint32_t pf[DEPTH][RL];
#pragma unroll
        for (int d = 0; d < DEPTH; ++d) {
            if (d < n) {
                const uint32_t* p = tp + (int64_t)(kt0 + d) * step;
#pragma unroll
                for (int i = 0; i < RL; ++i) {
                    const int wi = i * 32 + lane;
                    pf[d][i] = wi < PW ? __builtin_nontemporal_load(gl(p) + wi) : 0u;
                }
                if constexpr (ACT == DEPTH) act[d] = mlp_act_row(xl, xlive, kt0 + d);
            }
        }
        if constexpr (ACT == 1) {
            if (n > 0) act[0] = mlp_act_row(xl, xlive, kt0);
        }
#define TF_DS_MLP_REG_SLOT_R(S_)                                                                                        \
        do {                                                                                                            \
            _Pragma("unroll") for (int i_ = 0; i_ < RL; ++i_) {                                                        \
                const int wi_ = i_ * 32 + lane;                                                                         \
                if (wi_ < PW) st[wi_] = pf[S_][i_];                                                                     \
            }                                                                                                           \
            if constexpr (ACT == DEPTH) lane_x = act[S_];                                                               \
            if (j + DEPTH < n) {                                                                                        \
                const uint32_t* p_ = tp + (int64_t)(kt0 + j + DEPTH) * step;                                            \
                _Pragma("unroll") for (int i_ = 0; i_ < RL; ++i_) {                                                    \
                    const int wi_ = i_ * 32 + lane;                                                                     \
                    pf[S_][i_] = wi_ < PW ? __builtin_nontemporal_load(gl(p_) + wi_) : 0u;                              \
                }                                                                                                       \
                if constexpr (ACT == DEPTH) act[S_] = mlp_act_row(xl, xlive, kt0 + j + DEPTH);                          \
            }                                                                                                           \
        } while (0)
#pragma unroll 1
        for (int j = 0; j < n; ++j) {
            const int slot = j - (j / DEPTH) * DEPTH;
            uint32_t* st = wb;
            uint4 lane_x = make_uint4(0, 0, 0, 0);
            if (slot == 0) {
                TF_DS_MLP_REG_SLOT_R(0);
            } else if constexpr (DEPTH > 1) {
                if (slot == 1) TF_DS_MLP_REG_SLOT_R(1);
                else if constexpr (DEPTH > 2) {
                    if (slot == 2) TF_DS_MLP_REG_SLOT_R(2);
                    else if constexpr (DEPTH > 3) {
                        if (slot == 3) TF_DS_MLP_REG_SLOT_R(3);
                    }
                }
            }
            if constexpr (ACT == 1) {
                lane_x = act[0];
                if (j + 1 < n) act[0] = mlp_act_row(xl, xlive, kt0 + j + 1);
            }
            xst[lane] = lane_x;
            lds_fence();
            // ACCUMULATE ORDER verbatim from exl3.hip strip_rows_part lines 371-387. Tiles j = 0 .. n-1.
            // The half-wave add (lines 392-407) stays after this k loop.
            uint32_t own[TT][4];
#pragma unroll
            for (int t = 0; t < TT; ++t) decode_own<K2>(st + t * TW, win, own[t]);
#pragma unroll
            for (int r = 0; r < RM; ++r) {
                if (r < R) {
                    const uint4 x = xst[2 * r + h];
#pragma unroll
                    for (int t = 0; t < TT; ++t) {
                        float a = dot2(own[t][0], x.x, s[r][t]);
                        a = dot2(own[t][1], x.y, a);
                        a = dot2(own[t][2], x.z, a);
                        s[r][t] = dot2(own[t][3], x.w, a);
                    }
                }
            }
            __builtin_amdgcn_wave_barrier();
        }
#undef TF_DS_MLP_REG_SLOT_R
    } else {
#pragma unroll 1
        for (int d = 0; d < DEPTH; ++d)
            if (d < n) mlp_lds_fill<RL, PW>(wb + d * TILE, tp + (int64_t)(kt0 + d) * step, lane);
        if (n > 0) act[0] = mlp_act_row(xl, xlive, kt0);
#pragma unroll 1
        for (int j = 0; j < n; ++j) {
            const int slot = j - (j / DEPTH) * DEPTH;
            uint32_t* st = wb + slot * TILE;
            uint4 lane_x = act[0];
            if (j + 1 < n) act[0] = mlp_act_row(xl, xlive, kt0 + j + 1);
            xst[lane] = lane_x;
            lds_fence();
            // ACCUMULATE ORDER verbatim from exl3.hip strip_rows_part lines 371-387. Tiles j = 0 .. n-1.
            // The half-wave add (lines 392-407) stays after this k loop.
            uint32_t own[TT][4];
#pragma unroll
            for (int t = 0; t < TT; ++t) decode_own<K2>(st + t * TW, win, own[t]);
#pragma unroll
            for (int r = 0; r < RM; ++r) {
                if (r < R) {
                    const uint4 x = xst[2 * r + h];
#pragma unroll
                    for (int t = 0; t < TT; ++t) {
                        float a = dot2(own[t][0], x.x, s[r][t]);
                        a = dot2(own[t][1], x.y, a);
                        a = dot2(own[t][2], x.z, a);
                        s[r][t] = dot2(own[t][3], x.w, a);
                    }
                }
            }
            __builtin_amdgcn_wave_barrier();
            if (j + DEPTH < n) mlp_lds_fill<RL, PW>(st, tp + (int64_t)(kt0 + j + DEPTH) * step, lane);
        }
    }
    // ACCUMULATE ORDER verbatim from exl3.hip strip_rows_part lines 392-407 (half-wave add, after the k loop).
#pragma unroll
    for (int t = 0; t < TT; ++t) {
        float v[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (2 * i + 1 < RM) {
                const float s0 = s[2 * i][t], s1 = s[2 * i + 1][t];
                const float o0 = __uint_as_float(__builtin_amdgcn_permlanex16(__float_as_uint(s0), __float_as_uint(s0), 0x76543210u, 0xfedcba98u, false, false));
                const float o1 = __uint_as_float(__builtin_amdgcn_permlanex16(__float_as_uint(s1), __float_as_uint(s1), 0x76543210u, 0xfedcba98u, false, false));
                v[i] = h ? s1 + o1 : s0 + o0;
            } else {
                v[i] = 0.f;
            }
        }
        acc[t0 + t] += (f32x8){v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]};
    }
}

template <int K2, int DEPTH, int ACT, int STAGING>
__device__ __forceinline__ void strip_tiles_rows(const uint32_t* tiles, int64_t step, int kt0, int n, const _Float16* xlane,
                                                 int R, uint32_t* wb, uint4* xst, int lane, f32x8 (&acc)[8]) {
    if (n <= 0) return;
    if (R <= 8) {
        strip_rows_part<K2, 8, 8, 0, DEPTH, ACT, STAGING>(tiles, step, kt0, n, xlane, R, wb, xst, lane, acc);
    } else {
        strip_rows_part<K2, 16, 4, 0, DEPTH, ACT, STAGING>(tiles, step, kt0, n, xlane, R, wb, xst, lane, acc);
        __builtin_amdgcn_wave_barrier();   // the second part restages st and xst
        strip_rows_part<K2, 16, 4, 4, DEPTH, ACT, STAGING>(tiles, step, kt0, n, xlane, R, wb, xst, lane, acc);
    }
}

__device__ __forceinline__ void rot_out(const tf_ds_exl3_glayer& L, const float (&v)[4], int y_dtype, int lane, size_t row, int col0) {
    float u[4], s[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
        u[j] = y_dtype == BF16 ? bf16r(v[j]) : y_dtype == F16 ? __half2float(f2h(v[j])) : v[j];
    const int c = L.roff + col0 + 4 * lane;
    load4(reinterpret_cast<const void*>(L.rsuh), F16, c, s);
    rot128_in(u, s, lane);
    store4(reinterpret_cast<void*>(L.rxh), F16, row * (size_t)L.ldr + c, u);
}

// rope_heads (forward) of one row's 128 stored columns when they hold a head's rotary part: v rounded as stored (bf16),
// then re = fma(xe, cs, -(xo sn)), im = fma(xe, sn, xo cs) on each adjacent pair
__device__ __forceinline__ void rope_out(const tf_ds_exl3_glayer& L, float (&v)[4], int lane, size_t row, int col0) {
    const int d0 = (col0 % L.rhd) + 4 * lane, lo = L.rhd - L.rrd;
    if (d0 < lo) return;
    const int64_t p = reinterpret_cast<const int64_t*>(L.rpos)[row];
    const float* rcos = reinterpret_cast<const float*>(L.rcos);
    const float* rsin = reinterpret_cast<const float*>(L.rsin);
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int j = (d0 + 2 * q - lo) >> 1;
        const float cs = rcos[p * (L.rrd >> 1) + j], sn = rsin[p * (L.rrd >> 1) + j];
        const float xe = bf16r(v[2 * q]), xo = bf16r(v[2 * q + 1]);
        v[2 * q] = fmaf(xe, cs, -(xo * sn));
        v[2 * q + 1] = fmaf(xe, sn, xo * cs);
    }
}

__device__ __forceinline__ void finish_store(const tf_ds_exl3_glayer& L, float (&v)[4], int lane, size_t row, int col0) {
    finish(v, lane, reinterpret_cast<const __half*>(L.svh), reinterpret_cast<const __half*>(L.bias), col0 + 4 * lane);
    if (L.rcos) rope_out(L, v, lane, row, col0);
    store4(reinterpret_cast<void*>(L.y), L.y_dtype, row * (size_t)L.ldy + col0 + 4 * lane, v);
    if (L.rsuh) rot_out(L, v, L.y_dtype, lane, row, col0);
}

template <int K2, int WK, int DEPTH, int ACT, int STAGING>
__device__ __forceinline__ void glinear_mlp_body(const tf_ds_exl3_gargs& args) {
    constexpr int TW = 4 * K2;
    constexpr int TILE = 8 * TW;
    constexpr int SLOTS = STAGING == TF_DS_MLP_LDS ? DEPTH : 1;
    // static ring + the launcher's reduction (WK * min(M, 8) * 128 floats). 64KB is the workgroup limit.
    static_assert(SLOTS * WK * TILE * 4 + WK * 8 * 128 * 4 <= 65536, "LDS ring plus the 8-row reduction exceeds 64KB");
    extern __shared__ __attribute__((aligned(16))) float red[];   // WK * min(M, 8) * 128 floats (the launcher's)
    __shared__ __attribute__((aligned(16))) uint32_t stage[WK * SLOTS * TILE];
    const int M = args.M, RH = M < 8 ? M : 8;
    int li = 0;
#pragma unroll
    for (int q = 1; q < TF_DS_EXL3_GMAX; ++q)
        if (q < args.n && (int)blockIdx.x >= args.l[q].first) li = q;
    const tf_ds_exl3_glayer& L = args.l[li];
    const _Float16* xh = reinterpret_cast<const _Float16*>(L.xh);
    float* Z = reinterpret_cast<float*>(L.Z);
    int* counters = reinterpret_cast<int*>(L.counters);
    const int K = L.K, N = L.N, SK = L.SK;
    const int NB = N >> 7;
    const int b = (int)blockIdx.x - L.first;
    const int nb = b % NB, split = b / NB;
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    // ACCUMULATE ORDER verbatim from exl3.hip glinear_body line 616: kt0 = split * (per_warp * WK) + wave * per_warp.
    const int per_warp = (K >> 4) / SK / WK;
    const int kt0 = split * (per_warp * WK) + wave * per_warp;
    const uint32_t* tiles = reinterpret_cast<const uint32_t*>(L.T) + nb * L.stride_nb;
    const int col0 = nb * 128;
    uint32_t* wb = stage + wave * (SLOTS * TILE);
    // M >= 64 stays on exl3.hip's glinear (glinear_wide). The host launches this kernel only for 1..16 rows.
    for (int m0 = 0, pass = 0; m0 < M; m0 += 16, ++pass) {
        const int R = min(16, M - m0);
        f32x8 acc[8];
#pragma unroll
        for (int t = 0; t < 8; ++t) acc[t] = (f32x8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        // every row in the 1-row order (row invariance: a row's outputs do not depend on M); the rows' staging is the
        // wave's first 128 floats of red, free until the pass's reduction (its barrier)
        if (R == 1) {
            strip_tiles_1row<K2, DEPTH, ACT, STAGING>(tiles, L.stride_k, kt0, per_warp, xh + (size_t)m0 * L.ldx, wb, lane, acc);
        } else {
            const int ar = m0 + min(lane >> 1, R - 1);
            strip_tiles_rows<K2, DEPTH, ACT, STAGING>(tiles, L.stride_k, kt0, per_warp, xh + (size_t)ar * L.ldx, R, wb,
                                                      reinterpret_cast<uint4*>(red + wave * 128), lane, acc);
        }

#pragma unroll
        for (int hp = 0; hp < 2; ++hp) {
            const int rlo = 8 * hp;
            if (rlo >= R) break;
            const int rn = min(R - rlo, 8);
            __syncthreads();                                    // red is reused by every half and pass
#pragma unroll
            for (int t = 0; t < 8; ++t)
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int rr = 2 * i + (lane >> 4);
                    if (rr < RH) red[(wave * RH + rr) * 128 + t * 16 + (lane & 15)] = acc[t][4 * hp + i];
                }
            __syncthreads();
            // ACCUMULATE ORDER verbatim from exl3.hip glinear_body: wave 0's partial, then waves 1 .. WK-1 added on.
            if (SK == 1) {
                for (int r = wave; r < rn; r += WK) {
                    float v[4];
                    const float4 u = *reinterpret_cast<const float4*>(red + r * 128 + 4 * lane);
                    v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + 4 * lane);
                        v[0] += q.x; v[1] += q.y; v[2] += q.z; v[3] += q.w;
                    }
                    finish_store(L, v, lane, (size_t)(m0 + rlo + r), col0);
                }
            } else {
                for (int idx = threadIdx.x; idx < rn * 32; idx += WK * 32) {
                    const int r = idx >> 5, c = 4 * (idx & 31);
                    float4 s = *reinterpret_cast<const float4*>(red + r * 128 + c);
                    // ACCUMULATE ORDER verbatim from exl3.hip glinear_body: the same wave order into Z.
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + c);
                        s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                    }
                    *reinterpret_cast<float4*>(Z + ((size_t)split * M + m0 + rlo + r) * N + col0 + c) = s;
                }
            }
        }
        if (SK > 1) {
            __threadfence();
            __syncthreads();
            // stage is dead after the k loop (the next pass refills it). One word stands in for last_sh so a
            // depth-2 LDS ring on the widest variant still fits beside the reduction in 64KB. The volatile
            // accesses keep this word off the tile values a lane still holds in registers.
            if (threadIdx.x == 0)
                *static_cast<volatile uint32_t*>(stage) = atomicAdd(counters + pass * NB + nb, 1) == SK - 1 ? 1u : 0u;
            __syncthreads();
            const uint32_t split_done = *static_cast<volatile uint32_t*>(stage);
            if (split_done) {
                __threadfence();
                for (int r = wave; r < R; r += WK) {
                    const size_t at = ((size_t)m0 + r) * N + col0 + 4 * lane;
                    float v[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) v[j] = ld_cg(Z + at + j);
                    // ACCUMULATE ORDER verbatim from exl3.hip glinear_body: split 0, then splits 1 .. SK-1 added on.
                    for (int q = 1; q < SK; ++q)
#pragma unroll
                        for (int j = 0; j < 4; ++j) v[j] += ld_cg(Z + (size_t)q * M * N + at + j);
                    finish_store(L, v, lane, (size_t)m0 + r, col0);
                }
                if (threadIdx.x == 0) counters[pass * NB + nb] = 0;   // every program of the block has arrived
            }
        }
        __syncthreads();                                        // red is reused in the next pass
    }
}

}  // namespace

// Register family. DEPTH, ACT from the scan: deepest 0-scratch schedule with occupancy >= 6, except k16_w8.
#define TF_DS_EXL3_GLINEAR_MLP(K2_, WK_, D_, A_)                                                                        \
    extern "C" __global__ void __launch_bounds__(WK_ * 32)                                                              \
        tf_ds_exl3_glinear_mlp_k##K2_##_w##WK_(tf_ds_exl3_gargs args) {                                                 \
        glinear_mlp_body<K2_, WK_, D_, A_, TF_DS_MLP_REG>(args);                                                        \
    }
#define TF_DS_EXL3_GLINEAR_MLP_EU(K2_, WK_, D_, A_)                                                                     \
    extern "C" __global__ __launch_bounds__(WK_ * 32) __attribute__((amdgpu_waves_per_eu(6)))                           \
    void tf_ds_exl3_glinear_mlp_k##K2_##_w##WK_(tf_ds_exl3_gargs args) {                                                \
        glinear_mlp_body<K2_, WK_, D_, A_, TF_DS_MLP_REG>(args);                                                        \
    }
TF_DS_EXL3_GLINEAR_MLP(4, 4, 4, 1)
TF_DS_EXL3_GLINEAR_MLP(4, 8, 4, 1)
TF_DS_EXL3_GLINEAR_MLP(6, 4, 2, 2)
TF_DS_EXL3_GLINEAR_MLP(6, 8, 2, 2)
TF_DS_EXL3_GLINEAR_MLP(8, 4, 2, 2)
TF_DS_EXL3_GLINEAR_MLP(8, 8, 2, 2)
TF_DS_EXL3_GLINEAR_MLP(10, 4, 2, 1)
TF_DS_EXL3_GLINEAR_MLP(10, 8, 2, 1)
TF_DS_EXL3_GLINEAR_MLP(12, 4, 2, 1)
TF_DS_EXL3_GLINEAR_MLP(12, 8, 2, 1)
TF_DS_EXL3_GLINEAR_MLP_EU(16, 4, 2, 1)
TF_DS_EXL3_GLINEAR_MLP(16, 8, 2, 2)
#undef TF_DS_EXL3_GLINEAR_MLP_EU
#undef TF_DS_EXL3_GLINEAR_MLP

// LDS family. Deepest 0-scratch schedule with occupancy >= 6. Every (k2, wk) has one.
#define TF_DS_EXL3_GLINEAR_MLP_LDS(K2_, WK_, D_)                                                                        \
    extern "C" __global__ void __launch_bounds__(WK_ * 32)                                                              \
        tf_ds_exl3_glinear_mlp_lds_k##K2_##_w##WK_(tf_ds_exl3_gargs args) {                                             \
        glinear_mlp_body<K2_, WK_, D_, 1, TF_DS_MLP_LDS>(args);                                                         \
    }
TF_DS_EXL3_GLINEAR_MLP_LDS(4, 4, 4)
TF_DS_EXL3_GLINEAR_MLP_LDS(4, 8, 4)
TF_DS_EXL3_GLINEAR_MLP_LDS(6, 4, 4)
TF_DS_EXL3_GLINEAR_MLP_LDS(6, 8, 4)
TF_DS_EXL3_GLINEAR_MLP_LDS(8, 4, 4)
TF_DS_EXL3_GLINEAR_MLP_LDS(8, 8, 4)
TF_DS_EXL3_GLINEAR_MLP_LDS(10, 4, 4)
TF_DS_EXL3_GLINEAR_MLP_LDS(10, 8, 2)
TF_DS_EXL3_GLINEAR_MLP_LDS(12, 4, 2)
TF_DS_EXL3_GLINEAR_MLP_LDS(12, 8, 2)
TF_DS_EXL3_GLINEAR_MLP_LDS(16, 4, 2)
TF_DS_EXL3_GLINEAR_MLP_LDS(16, 8, 2)
#undef TF_DS_EXL3_GLINEAR_MLP_LDS

// Not the kit build. hipcc -DTF_DS_MLP_SCAN reports one resource line per symbol; nothing here is in registry.json.
#ifdef TF_DS_MLP_SCAN
#define TF_DS_MLP_SCAN_ONE(K2_, WK_, D_, A_, S_, SUF_)                                                                  \
    extern "C" __global__ void __launch_bounds__(WK_ * 32)                                                              \
        tf_ds_exl3_glinear_mlp_scan_k##K2_##_w##WK_##SUF_(tf_ds_exl3_gargs args) {                                     \
        glinear_mlp_body<K2_, WK_, D_, A_, S_>(args);                                                                   \
    }
#define TF_DS_MLP_SCAN_EU(K2_, WK_, D_, A_, S_, SUF_)                                                                   \
    extern "C" __global__ __launch_bounds__(WK_ * 32) __attribute__((amdgpu_waves_per_eu(6)))                           \
    void tf_ds_exl3_glinear_mlp_scan_k##K2_##_w##WK_##SUF_(tf_ds_exl3_gargs args) {                                     \
        glinear_mlp_body<K2_, WK_, D_, A_, S_>(args);                                                                   \
    }
#define TF_DS_MLP_SCAN_ALL(MACRO_, D_, A_, S_, SUF_)                                                                    \
    MACRO_(4, 4, D_, A_, S_, SUF_)                                                                                      \
    MACRO_(4, 8, D_, A_, S_, SUF_)                                                                                      \
    MACRO_(6, 4, D_, A_, S_, SUF_)                                                                                      \
    MACRO_(6, 8, D_, A_, S_, SUF_)                                                                                      \
    MACRO_(8, 4, D_, A_, S_, SUF_)                                                                                      \
    MACRO_(8, 8, D_, A_, S_, SUF_)                                                                                      \
    MACRO_(10, 4, D_, A_, S_, SUF_)                                                                                     \
    MACRO_(10, 8, D_, A_, S_, SUF_)                                                                                     \
    MACRO_(12, 4, D_, A_, S_, SUF_)                                                                                     \
    MACRO_(12, 8, D_, A_, S_, SUF_)                                                                                     \
    MACRO_(16, 4, D_, A_, S_, SUF_)                                                                                     \
    MACRO_(16, 8, D_, A_, S_, SUF_)
TF_DS_MLP_SCAN_ALL(TF_DS_MLP_SCAN_ONE, 2, 2, TF_DS_MLP_REG, _d2)
TF_DS_MLP_SCAN_ALL(TF_DS_MLP_SCAN_ONE, 3, 3, TF_DS_MLP_REG, _d3)
TF_DS_MLP_SCAN_ALL(TF_DS_MLP_SCAN_ONE, 4, 4, TF_DS_MLP_REG, _d4)
TF_DS_MLP_SCAN_ALL(TF_DS_MLP_SCAN_ONE, 4, 1, TF_DS_MLP_REG, _d4a1)
TF_DS_MLP_SCAN_ALL(TF_DS_MLP_SCAN_EU, 2, 1, TF_DS_MLP_REG, _d2a1e6)
TF_DS_MLP_SCAN_ALL(TF_DS_MLP_SCAN_ONE, 2, 1, TF_DS_MLP_LDS, _lds2)
TF_DS_MLP_SCAN_ALL(TF_DS_MLP_SCAN_EU, 2, 1, TF_DS_MLP_LDS, _lds2e6)
// lds4 at 8 warps and k2 10, 12 or 16 is past 64KB (ring + the 8-row reduction). Those three are not instantiated.
#define TF_DS_MLP_SCAN_LDS4(K2_, WK_)                                                                                   \
    TF_DS_MLP_SCAN_ONE(K2_, WK_, 4, 1, TF_DS_MLP_LDS, _lds4)
TF_DS_MLP_SCAN_LDS4(4, 4)
TF_DS_MLP_SCAN_LDS4(4, 8)
TF_DS_MLP_SCAN_LDS4(6, 4)
TF_DS_MLP_SCAN_LDS4(6, 8)
TF_DS_MLP_SCAN_LDS4(8, 4)
TF_DS_MLP_SCAN_LDS4(8, 8)
TF_DS_MLP_SCAN_LDS4(10, 4)
TF_DS_MLP_SCAN_LDS4(12, 4)
TF_DS_MLP_SCAN_LDS4(16, 4)
#undef TF_DS_MLP_SCAN_LDS4
#undef TF_DS_MLP_SCAN_ALL
#undef TF_DS_MLP_SCAN_EU
#undef TF_DS_MLP_SCAN_ONE
#endif
