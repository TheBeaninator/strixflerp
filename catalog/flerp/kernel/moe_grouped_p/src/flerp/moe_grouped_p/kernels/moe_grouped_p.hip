// kernel.flerp.moe_grouped_p: kernels/moe_grouped.hip of strixite 20937dc (v0.7.0-pod), its three F16 grouped
// kernels (MoeMath::WmmaF16Narrow / Wide / Wide256, with their SWI, XH and COMBINE forms), the combine reduce and the
// launch, reading Q4Panel16 experts (runtime/panel_device.hpp). Changes are marked `kernel.flerp.moe_grouped_p`;
// the routing (moe_group_routes) and every row-major path stay strixite's (kernel.strixite.moe_grouped, renamed where
// this adapter takes a name over). The FP32 and WmmaBf16 kernels have no panel form and refuse panels.
//
// Bit-equality: each thread gets the same code bytes and scale / min the origin's thread got (the same staged rows,
// the same window slots), so the F16 tiles, the WMMA sequence and every store are the origin's. Loads: Narrow straight
// to registers (a wave owns its 16 rows: the B fragment, wmma.hpp:6-8, wants lanes l and l + 16 on the same row, which
// a panel's 512-byte run of chunks c, c + 1 serves with one lane-pair swap - no LDS code tile, one barrier fewer of
// LDS traffic); Wide and Wide256 keep the LDS tile (two waves share each weight slab, decoded once) with whole-line
// loads (kernels/wmma_panel.hpp).
#include "kernels/moe_grouped_p.hpp"
#include "flerp/adapter_rt.hpp"
#include "kernels/wmma_panel.hpp"

#include "common/hip_check.hpp"
#include "kernels/device_info.hpp"
#include "kernels/wmma_gemm.hpp"

#include <cstdint>
#include <type_traits>

namespace strix::kernels {

// strixite's entry points, renamed in an assembly holding this adapter (SPEC.md section 5).
void linear_q4_experts_gather_grouped_flerp_origin(const void *x, const Q4DeviceView &w, int64_t E, const void *ws,
                                                   size_t ws_bytes, int64_t M, int64_t A, void *y, Act act,
                                                   MoeMath math, hipStream_t stream);
void linear_q4_experts_gather_grouped_swiglu_flerp_origin(const void *x, const Q4DeviceView &w, int64_t E,
                                                          const void *ws, size_t ws_bytes, int64_t M, int64_t A,
                                                          void *h, Act act, MoeMath math, hipStream_t stream,
                                                          bool h_f16);
void linear_q4_experts_combine_grouped_flerp_origin(const void *h, const Q4DeviceView &w, int64_t E,
                                                    const int32_t *ids, const float *coef, const void *ws,
                                                    size_t ws_bytes, float *partial, int64_t M, int64_t A, void *y,
                                                    Act act, MoeMath math, hipStream_t stream,
                                                    const MoeSharedAdd *shared, bool h_f16);

namespace {

constexpr int kThreads = 256;
constexpr int kRouteThreads = 1024;  // one per bucket: E experts + the invalid-id bucket
constexpr int kChunk = 32;           // Q4 codes per 16-byte load
constexpr int R = kMoeTileRows;

__device__ inline float bf16_bits_to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
__device__ inline void store(float *p, int64_t i, float v) { p[i] = v; }
__device__ inline void store(uint16_t *p, int64_t i, float v) { p[i] = f32_to_bf16_bits(v); }
// The F16 kernels' x staging: a BF16 pair as F16 bits, rounded toward zero - exact in F16's normal range.
__device__ inline uint32_t bf16x2_to_f16x2(uint32_t b) {
    return __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_pkrtz(__uint_as_float(b << 16), __uint_as_float(b & 0xffff0000u)));
}
// The SwiGLU gather's h: BF16 bits, or (XH) the down's staged F16 of them - BF16 rounding, then bf16x2_to_f16x2's
// conversion - so a down kernel that stages F16 h as it is reads what it would have converted from BF16 h.
template <bool XH>
__device__ inline uint16_t h_bits(float v) {
    const uint16_t b = f32_to_bf16_bits(v);
    return XH ? (uint16_t)bf16x2_to_f16x2(b) : b;
}
template <bool XH>
__device__ inline void store_h(uint16_t *p, int64_t i, float v) { p[i] = h_bits<XH>(v); }
template <bool XH>
__device__ inline void store_h(float *p, int64_t i, float v) {
    static_assert(!XH, "F16 h: BF16 activations only");
    p[i] = v;
}
// The combine's partials (one per slot, summed by combine_reduce_kernel): FP32, or - the F16 kernel, MoeMath::WmmaF16,
// the served path for Q4 experts - rounded to BF16 (to nearest even) into the same buffer read as uint16_t, half its
// bytes. BF16 partials: the down kernel 88 -> 76 us a token at 8192, the same 400-needle long-context score as FP32
// partials (365 vs 366; F16 partials, as fast, scored 301), NLL inside the noise floor.
__device__ inline void put_partial(float *p, int64_t i, float v, bool bf16) {
    if (bf16) reinterpret_cast<uint16_t *>(p)[i] = f32_to_bf16_bits(v);
    else p[i] = v;
}
__device__ inline float get_partial(const float *p, int64_t i, bool bf16) {
    return bf16 ? bf16_bits_to_f32(reinterpret_cast<const uint16_t *>(p)[i]) : p[i];
}

// A workgroup barrier that orders LDS only (as attention.hip's lds_barrier): __syncthreads' workgroup fence also waits
// for every global load in flight (s_waitcnt vmcnt(0)), which drains the K loops' loads issued ahead.
__device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

__device__ inline void load_x32(const float *x, int64_t off, float (&v)[kChunk]) {
    const float4 *p = reinterpret_cast<const float4 *>(x + off);
#pragma unroll
    for (int i = 0; i < kChunk / 4; ++i) {
        const float4 f = p[i];
        v[4 * i] = f.x, v[4 * i + 1] = f.y, v[4 * i + 2] = f.z, v[4 * i + 3] = f.w;
    }
}
__device__ inline void load_x32(const uint16_t *x, int64_t off, float (&v)[kChunk]) {
    const uint4 *p = reinterpret_cast<const uint4 *>(x + off);
#pragma unroll
    for (int i = 0; i < kChunk / 8; ++i) {
        const uint4 u = p[i];
        const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[8 * i + 2 * j] = __uint_as_float(w4[j] << 16);
            v[8 * i + 2 * j + 1] = __uint_as_float(w4[j] & 0xffff0000u);
        }
    }
}

// Workspace layout (int32): header [n_tiles, S, E, n_wtiles], offsets [E + 2] (bucket b's slots at slots[offsets[b]
// .. offsets[b+1]); bucket E = invalid ids), slots [S], tile_bucket [Tmax], tile_start [Tmax] (R-slot tiles, the
// FP32 kernels), wtile_bucket [WTmax], wtile_start [WTmax] (kMoeWmmaTileRows-slot tiles, the WMMA kernels),
// xtile_bucket [XTmax], xtile_start [XTmax] (kMoeWmmaWideRows-slot tiles, MoeMath::WmmaF16Wide), then n_xtiles.
constexpr int RW = kMoeWmmaTileRows, RX = kMoeWmmaWideRows;
struct RouteLayout {
    int64_t S, E, Tmax, WTmax, XTmax;
    int64_t offsets() const { return 4; }
    int64_t slots() const { return offsets() + E + 2; }
    int64_t tile_bucket() const { return slots() + S; }
    int64_t tile_start() const { return tile_bucket() + Tmax; }
    int64_t wtile_bucket() const { return tile_start() + Tmax; }
    int64_t wtile_start() const { return wtile_bucket() + WTmax; }
    int64_t xtile_bucket() const { return wtile_start() + WTmax; }
    int64_t xtile_start() const { return xtile_bucket() + XTmax; }
    int64_t n_xtiles() const { return xtile_start() + XTmax; }
    int64_t ints() const { return n_xtiles() + 1; }
};
RouteLayout route_layout(int64_t M, int64_t A, int64_t E) {
    const int64_t S = M * A;
    // sum over buckets of ceil(count / R) <= S / R + (E + 1), likewise for RW and RX
    return {S, E, S / R + E + 2, S / RW + E + 2, S / RX + E + 2};
}

constexpr int kWmmaBN = 128;  // moe_grouped.hip:281 (with the WmmaBf16 kernel this file does not copy)

// F16 on the matrix units (MoeMath::WmmaF16; Q4 experts): the same tiles and grid
// as grouped_wmma_kernel, but no per-group work in the K loop. x goes to LDS as F16 (exact from BF16 at these ranges);
// the weight codes go to LDS still packed (coalesced loads by the block), each wave decodes its own 16 rows to F16
// in registers (code * scale + min, one F16 rounding), and the matrix units accumulate the whole K extent in FP32.
// (Loading each wave's rows straight from memory instead was 1.65x slower: scattered code and 2-byte scale loads,
// what costs is cache lines touched, not bytes.)
using f16x2 = wmma_f16x2;  // kernels/wmma_gemm.hpp's F16 helpers
__device__ inline float silu_f32(float v) { return v / (1.0f + expf(-v)); }  // the form in kernels/swiglu.hip
__device__ inline void q4_word_to_f16(uint32_t w, f16x2 s, f16x2 m, uint32_t *out) { wmma_q4_word_to_f16(w, s, m, out); }
__device__ inline f16x2 bf16_bits_to_f16x2(uint16_t b) { return wmma_bf16_bits_to_f16x2(b); }

// SWI (gather only): the block's waves 0..3 compute gate columns [j0, j0 + 64) and waves 4..7 the matching up columns
// [I + j0, ...) of the stacked gate_up rows (N = 2I); the epilogue meets them in LDS and stores h = silu(g) u [slot, I]
// directly - SwiGLU from the FP32 accumulators, rounded once, instead of storing gu and running kernels/swiglu.
template <typename T, int G, bool COMBINE, int TR, bool SWI = false, bool XH = false>
__global__ void __launch_bounds__(kWmmaThreads)
    grouped_wmma_f16_kernel(const T *x, const uint8_t *q, const uint16_t *scale, const uint16_t *minv,
                            const int32_t *ws, int64_t A, int64_t N, int64_t K, T *y, float *partial, bool pbf16) {
    static_assert(TR == RW, "the route kernel's WMMA tile size");
    static_assert(!(SWI && COMBINE), "SwiGLU fuses into the gather only");
    static_assert(!XH || ((COMBINE || SWI) && sizeof(T) == 2), "XH: the down's x / the SwiGLU gather's h, BF16 path");
    constexpr int kHalf = kWmmaBN / 2;  // SWI: h columns a block
    static_assert(G == 32 || G == 64 || G == 128, "group sizes: a 32-code half-step lies in one group");
    constexpr int TM = TR / 16;
    // LDS: x as F16 [TR][kWmmaLd]; the step's codes packed [kWmmaBN rows][2 x 16 bytes + 16 pad] (coalesced global
    // loads, then each wave reads its own rows); scale / min windows of kWin groups per row, refreshed every kWin groups.
    constexpr int kWin = 8, kCodeLd = 3;  // uint4 per code row (2 used)
    __shared__ __attribute__((aligned(16))) uint16_t xs[TR * kWmmaLd];
    __shared__ uint16_t sw[kWmmaBN * kWin], mw[kWmmaBN * kWin];
    __shared__ int64_t xoff[TR], ooff[TR];
    __shared__ float ub[SWI ? TR * (kHalf + 1) : 1];  // SWI: the up waves' accumulators for the gate waves
    const int32_t S = ws[1], E = ws[2], n_tiles = ws[3];
    const int64_t tile = blockIdx.y;
    if (tile >= n_tiles) return;  // uniform per block, before any barrier
    const int32_t *offsets = ws + 4, *slots = offsets + E + 2;
    const int64_t Tmax = (int64_t)S / R + E + 2, WTmax = (int64_t)S / RW + E + 2;
    const int32_t *tile_bucket = slots + S + 2 * Tmax, *tile_start = tile_bucket + WTmax;
    const int32_t bucket = tile_bucket[tile], st = tile_start[tile];
    const int rows = min(TR, offsets[bucket + 1] - st), tm = (rows + 15) / 16;
    if ((int)threadIdx.x < rows) {
        const int64_t slot = slots[st + threadIdx.x];
        xoff[threadIdx.x] = (COMBINE ? slot : slot / A) * K;
        ooff[threadIdx.x] = slot * (SWI ? N / 2 : N);
    }
    __syncthreads();
    // SWI: n0 is the block's first h column j0 and rows_w its h columns; otherwise its first and count of output columns.
    constexpr int kCols = SWI ? kHalf : kWmmaBN;
    const int64_t n0 = (int64_t)blockIdx.x * kCols, Nh = SWI ? N / 2 : N;
    const int rows_w = (int)(Nh - n0 < kCols ? Nh - n0 : kCols), wave = threadIdx.x / 32;
    // The expert's weight row (from its first) behind staged row r of kWmmaBN, clamped to a valid row: SWI rows
    // [0, 64) are gate rows n0 + r, rows [64, 128) the up rows N / 2 + n0 + r - 64.
    auto wrow = [&](int r) -> int64_t {
        if constexpr (SWI) {
            const int rr = r % kHalf;
            return (r < kHalf ? 0 : N / 2) + n0 + (rr < rows_w ? rr : 0);
        } else {
            return n0 + (r < rows_w ? r : 0);
        }
    };
    if (bucket == E) {  // invalid ids: NaN outputs (err was set by moe_group_routes)
        for (int i = threadIdx.x; i < rows * rows_w; i += kWmmaThreads) {
            const int r = i / rows_w, c = i % rows_w;
            if (COMBINE) put_partial(partial, ooff[r] + n0 + c, __int_as_float(0x7fc00000), pbf16);
            else store_h<XH>(y, ooff[r] + n0 + c, __int_as_float(0x7fc00000));
        }
        return;
    }
    const int64_t row0 = (int64_t)bucket * N, n_groups = K / G;
    // Global fetch (prefetched a step ahead): thread t, code row t / 2 (clamped to a valid row), 16-byte half t % 2;
    // at a window start, kWin (scale, min) of kWmmaBN * kWin / kWmmaThreads rows - consecutive threads read
    // consecutive groups of a row (a 16-byte run), not one scattered 2-byte value per lane.
    static_assert(kWmmaBN * 2 == kWmmaThreads, "one 16-byte code load per thread and step");
    constexpr int kWinLoads = kWmmaBN * kWin / kWmmaThreads;
    // kernel.flerp.moe_grouped_p: straight to registers. Lane l of wave w loads chunk k0 / 32 + l / 16 of its own
    // weight row (staged row w * 16 + l % 16, the row its B fragment holds): the wave reads its panel's 512 contiguous
    // bytes, and lanes l, l ^ 16 swap halves (one shuffle per word) - every lane then holds what the origin's lane
    // read back from the LDS code tile. The window: entry e at (panel row, slot) WmmaWCodesP's order, a line a wave.
    const int64_t P = 8 * K + 64 * K / G, pcodes = 8 * K;
    const int lhalf = ((int)threadIdx.x % 32) / 16;
    const int64_t crow_g = row0 + wrow(wave * 16 + (int)threadIdx.x % 16);
    uint4 cq;
    uint16_t fs[kWinLoads], fm[kWinLoads];
    auto wfetch = [&](int64_t k0) {
        cq = *reinterpret_cast<const uint4 *>(flerp::panel_chunk(q, P, crow_g, k0 / 32 + lhalf));
        const int64_t g0 = k0 / G;
        if (k0 % G == 0 && g0 % kWin == 0)
#pragma unroll
            for (int i = 0; i < kWinLoads; ++i) {
                const int e = (int)threadIdx.x + i * kWmmaThreads;
                const int64_t g = g0 + WmmaWCodesP<4, G, kWmmaBN>::win_slot(e),
                              row = row0 + wrow(WmmaWCodesP<4, G, kWmmaBN>::win_row(e));
                const uint32_t sm = g < n_groups ? flerp::panel_sm(q, P, pcodes, row, g) : 0u;
                fs[i] = (uint16_t)(sm & 0xffffu), fm[i] = (uint16_t)(sm >> 16);
            }
    };
    WmmaXTile<T, TR, G, false> xt;
    constexpr int kSegs = kWmmaKC / 8, kPasses = (TR * kSegs + kWmmaThreads - 1) / kWmmaThreads;
    const int wr = wave * 16 + (int)threadIdx.x % 16;  // this lane's weight row (B column); lanes l, l + 16 the same
    // The K loop for TMA active 16-row tiles, chosen once per block: a per-tile `if` inside the loop made the compiler
    // copy all the accumulators at every join (913 register moves, 241 VGPRs in the ISA).
    auto k_loop = [&](auto tma_c) {
        constexpr int TMA = decltype(tma_c)::value;
        wmma_f32x8 acc[TMA] = {};
        xt.fetch(x, xoff, rows, 0);
        wfetch(0);
        for (int64_t k0 = 0; k0 < K; k0 += kWmmaKC) {
            // stage: x as F16 (BF16 -> F16 exact in F16's normal range), the packed codes, a new scale window
#pragma unroll
            for (int p = 0; p < kPasses; ++p) {
                const int seg = (int)threadIdx.x + p * kWmmaThreads;
                if (seg >= TR * kSegs) break;  // wave-uniform
                uint32_t h[4];
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    h[i] = XH && COMBINE ? xt.b[p][i] : bf16x2_to_f16x2(xt.b[p][i]);  // XH: x is F16 already
                *reinterpret_cast<uint4 *>(xs + (seg / kSegs) * kWmmaLd + (seg % kSegs) * 8) =
                    make_uint4(h[0], h[1], h[2], h[3]);
            }
            // kernel.flerp.moe_grouped_p: this step's two code chunks of row wr from the lane pair (l, l ^ 16)
            const uint4 co = make_uint4(__shfl_xor(cq.x, 16, 32), __shfl_xor(cq.y, 16, 32), __shfl_xor(cq.z, 16, 32),
                                        __shfl_xor(cq.w, 16, 32));
            const uint4 c0 = lhalf ? co : cq, c1 = lhalf ? cq : co;
            if (k0 % G == 0 && (k0 / G) % kWin == 0)
#pragma unroll
                for (int i = 0; i < kWinLoads; ++i) {
                    const int e = (int)threadIdx.x + i * kWmmaThreads;
                    const int at = WmmaWCodesP<4, G, kWmmaBN>::win_row(e) * kWin + WmmaWCodesP<4, G, kWmmaBN>::win_slot(e);
                    sw[at] = fs[i], mw[at] = fm[i];  // kernel.flerp.moe_grouped_p: the origin's [r][slot]
                }
            lds_barrier();
            if (k0 + kWmmaKC < K) {  // the next step's loads in flight during this one's math
                xt.fetch(x, xoff, rows, k0 + kWmmaKC);
                wfetch(k0 + kWmmaKC);
            }
            const uint32_t wv[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
            const int ga = (int)((k0 / G) % kWin), gb = (int)(((k0 + 32) / G) % kWin);
            const f16x2 s0 = bf16_bits_to_f16x2(sw[wr * kWin + ga]), m0 = bf16_bits_to_f16x2(mw[wr * kWin + ga]),
                        s1 = bf16_bits_to_f16x2(sw[wr * kWin + gb]), m1 = bf16_bits_to_f16x2(mw[wr * kWin + gb]);
#pragma unroll
            for (int kk = 0; kk < kWmmaKC / 16; ++kk) {
                uint32_t b[8];
                const f16x2 s = kk < 2 ? s0 : s1, m = kk < 2 ? m0 : m1;
                q4_word_to_f16(wv[2 * kk], s, m, b);
                q4_word_to_f16(wv[2 * kk + 1], s, m, b + 4);
                const wmma_u32x8 bv = {b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]};
                const wmma_f16x16 bf = __builtin_bit_cast(wmma_f16x16, bv);
#pragma unroll
                for (int i = 0; i < TMA; ++i)
                    acc[i] = wmma_f16(wmma_load_rows_a16_f16(xs + 16 * i * kWmmaLd + kk * 16, kWmmaLd), bf, acc[i]);
            }
            lds_barrier();  // the next step's loads stay in flight
        }
        if constexpr (SWI) {
            const int c = (wave % 4) * 16 + (int)threadIdx.x % 16;
            if (wave >= 4)
#pragma unroll
                for (int i = 0; i < TMA; ++i)
#pragma unroll
                    for (int v = 0; v < 8; ++v) ub[(16 * i + wmma_c_row(v)) * (kHalf + 1) + c] = acc[i][v];
            __syncthreads();
            if (wave < 4)
#pragma unroll
                for (int i = 0; i < TMA; ++i)
#pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        const int r = 16 * i + wmma_c_row(v);
                        if (r < rows && c < rows_w)
                            store_h<XH>(y, ooff[r] + n0 + c, silu_f32(acc[i][v]) * ub[r * (kHalf + 1) + c]);
                    }
        } else {
#pragma unroll
            for (int i = 0; i < TMA; ++i)
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    const int r = 16 * i + wmma_c_row(v), c = wave * 16 + (int)threadIdx.x % 16;
                    if (r < rows && c < rows_w) {
                        if (COMBINE) put_partial(partial, ooff[r] + n0 + c, acc[i][v], pbf16);
                        else store(y, ooff[r] + n0 + c, acc[i][v]);
                    }
                }
        }
    };
    static_assert(TM == 4, "the dispatch below covers 1..4 active tiles");
    switch (tm) {  // uniform per block
        case 1: k_loop(std::integral_constant<int, 1>{}); break;
        case 2: k_loop(std::integral_constant<int, 2>{}); break;
        case 3: k_loop(std::integral_constant<int, 3>{}); break;
        default: k_loop(std::integral_constant<int, 4>{}); break;
    }
}

// MoeMath::WmmaF16Wide: grouped_wmma_f16_kernel's math on linear_wmma_f16_kernel's structure. A block takes up to
// RX = 128 slots of one expert x 128 weight rows; each K step dequantizes the weight tile once into LDS (every code
// once, by all 256 threads) where the F16 kernel's waves decode their own 16 rows in registers (lanes l and l + 16 the
// same row: twice) for at most 4 row tiles. Waves are 2 (row tiles, interleaved: wave row wm takes tiles wm, wm + 2,
// ..) x 4 (32 weight rows each, 2 column tiles), so a block's active tiles split evenly and the per-wave count is
// block-uniform (one specialized K loop per count, no branch around a WMMA). The F16 weights (wmma_stage_codes_f16 - the same
// wmma_q4_word_to_f16 decode), the F16 x and each output tile's WMMA sequence over K are the F16 kernel's, so the
// outputs are bit-identical to it. SWI: staged rows [0, 64) are gate rows, [64, 128) the matching up rows; waves of
// column group wn < 2 hold gate columns, wn >= 2 the up columns, met in LDS (the tiles' bytes, free after the loop).
template <typename T, int G, bool COMBINE, bool SWI, bool XH = false>
__global__ void __launch_bounds__(kWmmaThreads)
    grouped_wmma_f16_wide_kernel(const T *x, const uint8_t *q, const uint16_t *scale, const uint16_t *minv,
                                 const int32_t *ws, int64_t A, int64_t N, int64_t K, T *y, float *partial, bool pbf16) {
    static_assert(!(SWI && COMBINE), "SwiGLU fuses into the gather only");
    static_assert(!XH || ((COMBINE || SWI) && sizeof(T) == 2), "XH: the down's x / the SwiGLU gather's h, BF16 path");
    static_assert(G == 32 || G == 64 || G == 128, "group sizes: a 16-byte code load lies in one group");
    using WT = WmmaWCodesP<4, G, kWmmaBN>;  // kernel.flerp.moe_grouped_p: fetch_mapped from the panels
    constexpr int TR = RX, kHalf = kWmmaBN / 2, kTileElems = (TR + kWmmaBN) * kWmmaLd;
    static_assert(!SWI || TR * (kHalf + 1) * 4 <= kTileElems * 2, "SWI: the up accumulators fit in the tiles' LDS");
    constexpr int kOutLd = kWmmaBN + 8;  // COMBINE: the BF16 output tile's LDS row stride (16-byte aligned rows)
    static_assert(TR * kOutLd <= kTileElems && kOutLd * 2 % 16 == 0, "COMBINE: the output tile fits in the tiles' LDS");
    __shared__ __attribute__((aligned(16))) uint16_t tiles_lds[kTileElems];
    uint16_t *xs = tiles_lds, *wsm = tiles_lds + TR * kWmmaLd;
    float *ub = reinterpret_cast<float *>(tiles_lds);  // SWI, after the K loop
    __shared__ uint16_t sw[kWmmaBN * WT::kWin], mw[kWmmaBN * WT::kWin];
    __shared__ int64_t xoff[TR], ooff[TR];
    const int32_t S = ws[1], E = ws[2];
    const int64_t Tmax = (int64_t)S / R + E + 2, WTmax = (int64_t)S / RW + E + 2, XTmax = (int64_t)S / RX + E + 2;
    const int32_t *offsets = ws + 4, *slots = offsets + E + 2;
    const int32_t *xtile_bucket = slots + S + 2 * Tmax + 2 * WTmax, *xtile_start = xtile_bucket + XTmax;
    const int64_t tile = blockIdx.y;
    if (tile >= xtile_start[XTmax]) return;  // n_xtiles; uniform per block, before any barrier
    const int32_t bucket = xtile_bucket[tile], st = xtile_start[tile];
    const int rows = min(TR, offsets[bucket + 1] - st), nt = (rows + 15) / 16;
    constexpr int kCols = SWI ? kHalf : kWmmaBN;
    const int64_t n0 = (int64_t)blockIdx.x * kCols, Nh = SWI ? N / 2 : N;
    const int rows_w = (int)(Nh - n0 < kCols ? Nh - n0 : kCols);
    if ((int)threadIdx.x < rows) {
        const int64_t slot = slots[st + threadIdx.x];
        xoff[threadIdx.x] = (COMBINE ? slot : slot / A) * K;
        ooff[threadIdx.x] = slot * (SWI ? N / 2 : N);
    }
    __syncthreads();
    if (bucket == E) {  // invalid ids: NaN outputs (err was set by moe_group_routes)
        for (int i = threadIdx.x; i < rows * rows_w; i += kWmmaThreads) {
            const int r = i / rows_w, c = i % rows_w;
            if (COMBINE) put_partial(partial, ooff[r] + n0 + c, __int_as_float(0x7fc00000), pbf16);
            else store_h<XH>(y, ooff[r] + n0 + c, __int_as_float(0x7fc00000));
        }
        return;
    }
    const int wave = (int)threadIdx.x / 32, wm = wave / 4, wn = wave % 4;
    const int64_t wbase = (int64_t)bucket * N + n0;
    // 16-byte partial stores: every slot row's columns start 16-byte aligned and the block's columns are whole 8-column
    // chunks (uniform per launch)
    const bool vec_out = N % 8 == 0 && reinterpret_cast<uintptr_t>(partial) % 16 == 0;
    // Staged weight row r -> its global row in the stacked [E * N, K] codes (< 0: past the block's columns). Arithmetic,
    // not an LDS table: the table cost ~30 VGPRs (250 -> occupancy 4; this form 217 -> 6, gfx1151).
    const auto row_of = [&](int r) -> int64_t {
        const int c = SWI ? r % kHalf : r;
        return c < rows_w ? wbase + (SWI && r >= kHalf ? N / 2 - kHalf : 0) + r : -1;
    };
    WmmaXTile<T, TR, G, false> xt;
    WT wt;
    auto k_loop = [&](auto tma_c) {
        constexpr int TMA = decltype(tma_c)::value;  // this wave's row tiles: wm, wm + 2, .., 2 (TMA - 1) + wm
        wmma_f32x8 acc[TMA][2] = {};
        xt.fetch(x, xoff, rows, 0);
        wt.fetch_mapped(q, scale, minv, row_of, K, 0);
        for (int64_t k0 = 0; k0 < K; k0 += kWmmaKC) {
            if (wt.win) {  // uniform: depends on k0 only
                wt.stage_window(sw, mw);
                lds_barrier();
            }
            if constexpr (XH && COMBINE) xt.stage(xs, nullptr);  // x is F16 already: staged as it is
            else xt.stage_f16(xs);
            wt.stage_f16(wsm, sw, mw, k0);
            lds_barrier();
            if (k0 + kWmmaKC < K) {  // the next step's loads in flight during this one's math
                xt.fetch(x, xoff, rows, k0 + kWmmaKC);
                wt.fetch_mapped(q, scale, minv, row_of, K, k0 + kWmmaKC);
            }
#pragma unroll
            for (int kk = 0; kk < kWmmaKC; kk += 16) {
                wmma_f16x16 a[TMA], b[2];
#pragma unroll
                for (int i = 0; i < TMA; ++i) a[i] = wmma_load_rows_a16_f16(xs + 16 * (2 * i + wm) * kWmmaLd + kk, kWmmaLd);
#pragma unroll
                for (int j = 0; j < 2; ++j) b[j] = wmma_load_rows_a16_f16(wsm + (wn * 32 + 16 * j) * kWmmaLd + kk, kWmmaLd);
#pragma unroll
                for (int i = 0; i < TMA; ++i)
#pragma unroll
                    for (int j = 0; j < 2; ++j) acc[i][j] = wmma_f16(a[i], b[j], acc[i][j]);
            }
            lds_barrier();  // the next step's loads stay in flight
        }
        if constexpr (SWI) {
            if (wn >= 2)  // up columns: hand them to the gate waves
#pragma unroll
                for (int i = 0; i < TMA; ++i)
#pragma unroll
                    for (int j = 0; j < 2; ++j)
#pragma unroll
                        for (int v = 0; v < 8; ++v) {
                            const int r = 16 * (2 * i + wm) + wmma_c_row(v), c = (wn - 2) * 32 + 16 * j + (int)threadIdx.x % 16;
                            ub[r * (kHalf + 1) + c] = acc[i][j][v];
                        }
            __syncthreads();
            if (wn < 2)
#pragma unroll
                for (int i = 0; i < TMA; ++i)
#pragma unroll
                    for (int j = 0; j < 2; ++j)
#pragma unroll
                        for (int v = 0; v < 8; ++v) {
                            const int r = 16 * (2 * i + wm) + wmma_c_row(v), c = wn * 32 + 16 * j + (int)threadIdx.x % 16;
                            if (r < rows && c < rows_w)
                                store_h<XH>(y, ooff[r] + n0 + c, silu_f32(acc[i][j][v]) * ub[r * (kHalf + 1) + c]);
                        }
        } else if (COMBINE && pbf16 && vec_out) {
            // BF16 partials through LDS (the tiles' bytes, free after the loop's last barrier): each value rounded as
            // put_partial does, then written a 16-byte chunk (8 columns of one slot's row) a store - where 2-byte
            // stores, a half-wave's 32 bytes each, were the down's epilogue.
#pragma unroll
            for (int i = 0; i < TMA; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j)
#pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        const int r = 16 * (2 * i + wm) + wmma_c_row(v), c = wn * 32 + 16 * j + (int)threadIdx.x % 16;
                        tiles_lds[r * kOutLd + c] = f32_to_bf16_bits(acc[i][j][v]);
                    }
            __syncthreads();
            uint16_t *pb = reinterpret_cast<uint16_t *>(partial);
            for (int e = (int)threadIdx.x; e < rows * (kWmmaBN / 8); e += kWmmaThreads) {
                const int r = e / (kWmmaBN / 8), c = e % (kWmmaBN / 8) * 8;
                if (c >= rows_w) continue;  // rows_w % 8 == 0 (N % 8 == 0, n0 % kWmmaBN == 0): whole chunks only
                *reinterpret_cast<uint4 *>(pb + ooff[r] + n0 + c) =
                    *reinterpret_cast<const uint4 *>(tiles_lds + r * kOutLd + c);
            }
        } else {
#pragma unroll
            for (int i = 0; i < TMA; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j)
#pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        const int r = 16 * (2 * i + wm) + wmma_c_row(v), c = wn * 32 + 16 * j + (int)threadIdx.x % 16;
                        if (r < rows && c < rows_w) {
                            if (COMBINE) put_partial(partial, ooff[r] + n0 + c, acc[i][j][v], pbf16);
                            else store(y, ooff[r] + n0 + c, acc[i][j][v]);
                        }
                    }
        }
    };
    switch ((nt + 1) / 2) {  // uniform per block: the busier wave row's tiles
        case 1: k_loop(std::integral_constant<int, 1>{}); break;
        case 2: k_loop(std::integral_constant<int, 2>{}); break;
        case 3: k_loop(std::integral_constant<int, 3>{}); break;
        default: k_loop(std::integral_constant<int, 4>{}); break;
    }
}

// MoeMath::WmmaF16Wide256: the wide kernel's math on 128-slot x 256-weight-row blocks, K steps of 32, each wave 4 x 4
// output tiles where the 128 x 128 form held 4 x 2. Per WMMA that halves the fragment loads from LDS and the x tile's
// conversion and loads. Thread t stages weight row t: its 32 codes a step (one 16-byte load, one group) and that
// group's scale / min, loaded by the thread when a step opens a group - no scale-window LDS or its barrier. LDS rows
// are 40 F16 (80 bytes: 8 lanes' 16-byte reads hit distinct banks). SWI: wave column group wn stages gate rows
// [32 wn, 32 wn + 32) of the block's h columns as its tiles 0-1 and the matching up rows as tiles 2-3, so
// h = silu(g) u forms in registers. The staged F16 values (wmma_stage_codes_f16, the x conversion) and each output
// tile's WMMA sequence over K are the wide kernel's, so the outputs are bit-identical to it.
template <typename T, int G, bool COMBINE, bool SWI, bool XH = false>
__global__ void __launch_bounds__(kWmmaThreads)
    grouped_wmma_f16_wide256_kernel(const T *x, const uint8_t *q, const uint16_t *scale, const uint16_t *minv,
                                    const int32_t *ws, int64_t A, int64_t N, int64_t K, T *y, float *partial, bool pbf16) {
    static_assert(!(SWI && COMBINE), "SwiGLU fuses into the gather only");
    static_assert(!XH || ((COMBINE || SWI) && sizeof(T) == 2), "XH: the down's x / the SwiGLU gather's h, BF16 path");
    static_assert(G == 32 || G == 64 || G == 128, "group sizes: a step's 32 codes lie in one group");
    constexpr int TR = RX, BW = kWmmaThreads, KC = 32, LD = KC + 8, kSegs = KC / 8;
    constexpr int kXPasses = TR * kSegs / kWmmaThreads;
    static_assert(TR * kSegs % kWmmaThreads == 0 && LD * 2 % 16 == 0, "whole x passes, 16-byte aligned LDS rows");
    __shared__ __attribute__((aligned(16))) uint16_t xs[TR * LD], wsm[BW * LD];
    __shared__ int64_t xoff[TR], ooff[TR];
    const int32_t S = ws[1], E = ws[2];
    const int64_t Tmax = (int64_t)S / R + E + 2, WTmax = (int64_t)S / RW + E + 2, XTmax = (int64_t)S / RX + E + 2;
    const int32_t *offsets = ws + 4, *slots = offsets + E + 2;
    const int32_t *xtile_bucket = slots + S + 2 * Tmax + 2 * WTmax, *xtile_start = xtile_bucket + XTmax;
    const int64_t tile = blockIdx.y;
    if (tile >= xtile_start[XTmax]) return;  // n_xtiles; uniform per block, before any barrier
    const int32_t bucket = xtile_bucket[tile], st = xtile_start[tile];
    const int rows = min(TR, offsets[bucket + 1] - st), nt = (rows + 15) / 16;
    constexpr int kCols = SWI ? BW / 2 : BW;
    const int64_t n0 = (int64_t)blockIdx.x * kCols, Nh = SWI ? N / 2 : N;
    const int rows_w = (int)(Nh - n0 < kCols ? Nh - n0 : kCols);
    if ((int)threadIdx.x < rows) {
        const int64_t slot = slots[st + threadIdx.x];
        xoff[threadIdx.x] = (COMBINE ? slot : slot / A) * K;
        ooff[threadIdx.x] = slot * (SWI ? N / 2 : N);
    }
    __syncthreads();
    if (bucket == E) {  // invalid ids: NaN outputs (err was set by moe_group_routes)
        for (int i = threadIdx.x; i < rows * rows_w; i += kWmmaThreads) {
            const int r = i / rows_w, c = i % rows_w;
            if (COMBINE) put_partial(partial, ooff[r] + n0 + c, __int_as_float(0x7fc00000), pbf16);
            else store_h<XH>(y, ooff[r] + n0 + c, __int_as_float(0x7fc00000));
        }
        return;
    }
    const int wave = (int)threadIdx.x / 32, wm = wave / 4, wn = wave % 4;
    // SWI: 16-byte h stores - every slot row's columns start 16-byte aligned and the block's columns are whole 8-column
    // chunks (uniform per launch)
    const bool vec_out = (N / 2) % 8 == 0 && reinterpret_cast<uintptr_t>(y) % 16 == 0;
    // This thread's staged weight row (< 0: past the block's columns, staged as zeros).
    const int64_t wrow = [&]() -> int64_t {
        const int r = (int)threadIdx.x;
        if constexpr (SWI) {
            const int c = (r / 64) * 32 + r % 32;
            return c < rows_w ? (int64_t)bucket * N + (r % 64 >= 32 ? N / 2 : 0) + n0 + c : -1;
        } else {
            return r < rows_w ? (int64_t)bucket * N + n0 + r : -1;
        }
    }();
    const int64_t groups = K / G;
    uint4 qc = make_uint4(0, 0, 0, 0);  // this step's codes of row wrow
    uint16_t sb = 0, mb = 0;            // their group's scale / min (BF16 bits)
    uint32_t xb[kXPasses][4];           // 8 BF16 x values a pass
    // kernel.flerp.moe_grouped_p: row wrow's chunk and (scale, min) from its panel (a wave's 32 rows: two panels'
    // 256-byte runs of this chunk). The row's byte offsets are formed once, as 32-bit values (the launcher checks the
    // panel buffer is under 4 GiB): flerp::panel_chunk / panel_sm with the panel stride and the row's panel and lane
    // hoisted and shifts for the 16-row panel, 32-code chunk and power-of-two G - chunk c at qoff + 256 c, group g's
    // (scale, min) at smoff + 64 g. No run-time division or 64-bit address pair live across the K loop
    // (gate/results/isa-spills: those spilled 24-36 B in this kernel).
    constexpr int kLogG = G == 32 ? 5 : G == 64 ? 6 : 7;
    uint32_t qoff = 0, smoff = 0;
    if (wrow >= 0) {
        const uint32_t P = (uint32_t)(8 * K + ((64 * K) >> kLogG)), pan = (uint32_t)(wrow >> 4) * P;
        qoff = pan + (uint32_t)(wrow & 15) * 16u;
        smoff = pan + (uint32_t)(8 * K) + (uint32_t)(wrow & 15) * 4u;
    }
    const auto fetch = [&](int64_t k0) {
        if (wrow >= 0) {
            const uint32_t kk = (uint32_t)k0;
            qc = *reinterpret_cast<const uint4 *>(q + (qoff + ((kk >> 5) << 8)));
            if ((kk & (G - 1)) == 0) {
                const uint32_t sm = *reinterpret_cast<const uint32_t *>(q + (smoff + ((kk >> kLogG) << 6)));
                sb = (uint16_t)(sm & 0xffffu), mb = (uint16_t)(sm >> 16);
            }
        }
#pragma unroll
        for (int p = 0; p < kXPasses; ++p) {
            const int seg = (int)threadIdx.x + p * kWmmaThreads, r = seg / kSegs;
            xb[p][0] = xb[p][1] = xb[p][2] = xb[p][3] = 0;
            if (r >= rows) continue;
            const T *src = x + xoff[r] + k0 + (seg % kSegs) * 8;
            if constexpr (sizeof(T) == 2) {
                const uint4 u = *reinterpret_cast<const uint4 *>(src);
                xb[p][0] = u.x, xb[p][1] = u.y, xb[p][2] = u.z, xb[p][3] = u.w;
            } else {
                const float4 f0 = reinterpret_cast<const float4 *>(src)[0], f1 = reinterpret_cast<const float4 *>(src)[1];
                const float v[8] = {f0.x, f0.y, f0.z, f0.w, f1.x, f1.y, f1.z, f1.w};
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    xb[p][i] = (uint32_t)wmma_f32_to_bf16(v[2 * i]) | ((uint32_t)wmma_f32_to_bf16(v[2 * i + 1]) << 16);
            }
        }
    };
    const auto stage = [&]() {  // x as F16 (WmmaXTile::stage_f16's conversion), row wrow's codes dequantized to F16
#pragma unroll
        for (int p = 0; p < kXPasses; ++p) {
            const int seg = (int)threadIdx.x + p * kWmmaThreads;
            uint32_t h[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
                h[i] = XH && COMBINE ? xb[p][i] : bf16x2_to_f16x2(xb[p][i]);  // XH: x is F16 already
            *reinterpret_cast<uint4 *>(xs + (seg / kSegs) * LD + (seg % kSegs) * 8) = make_uint4(h[0], h[1], h[2], h[3]);
        }
        wmma_stage_codes_f16<4>(qc, wmma_bf16_bits_to_f16x2(sb), wmma_bf16_bits_to_f16x2(mb), wsm + threadIdx.x * LD);
    };
    auto k_loop = [&](auto tma_c) {
        constexpr int TMA = decltype(tma_c)::value;  // this wave's row tiles: wm, wm + 2, .., 2 (TMA - 1) + wm
        wmma_f32x8 acc[TMA][4] = {};
        fetch(0);
        for (int64_t k0 = 0; k0 < K; k0 += KC) {
            stage();
            lds_barrier();
            if (k0 + KC < K) fetch(k0 + KC);  // the next step's loads in flight during this one's math
#pragma unroll
            for (int kk = 0; kk < KC; kk += 16) {
                wmma_f16x16 b[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) b[j] = wmma_load_rows_a16_f16(wsm + (wn * 64 + 16 * j) * LD + kk, LD);
#pragma unroll
                for (int i = 0; i < TMA; ++i) {
                    const wmma_f16x16 a = wmma_load_rows_a16_f16(xs + 16 * (2 * i + wm) * LD + kk, LD);
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[i][j] = wmma_f16(a, b[j], acc[i][j]);
                }
            }
            lds_barrier();  // the next step's loads stay in flight
        }
        if constexpr (SWI && sizeof(T) == 2) {
            if (vec_out) {
                // h through LDS (the weight tile's bytes, free after the loop's last barrier) in two 64-row halves,
                // each value rounded as store() does, then written a 16-byte chunk (8 columns of one slot's row) a
                // store, where 2-byte stores, a half-wave's 32 bytes each, were the epilogue.
                constexpr int kOutLd = kCols + 8, kChunks = kCols / 8;
                static_assert(TR / 2 * kOutLd <= BW * LD && kOutLd * 2 % 16 == 0, "SWI: half the h tile fits in LDS");
                uint16_t *yb = reinterpret_cast<uint16_t *>(y);
#pragma unroll
                for (int half = 0; half < 2; ++half) {
#pragma unroll
                    for (int i = 0; i < TMA; ++i) {
                        if (i / 2 != half) continue;  // row tile 2 i + wm lies in rows [64 half, 64 half + 64)
#pragma unroll
                        for (int j = 0; j < 2; ++j)
#pragma unroll
                            for (int v = 0; v < 8; ++v) {
                                const int r = 16 * (2 * i + wm) + wmma_c_row(v) - 64 * half;
                                const int c = wn * 32 + 16 * j + (int)threadIdx.x % 16;
                                wsm[r * kOutLd + c] = h_bits<XH>(silu_f32(acc[i][j][v]) * acc[i][j + 2][v]);
                            }
                    }
                    __syncthreads();
                    for (int e = (int)threadIdx.x; e < TR / 2 * kChunks; e += kWmmaThreads) {
                        const int rl = e / kChunks, r = 64 * half + rl, c = e % kChunks * 8;
                        if (r >= rows || c >= rows_w) continue;  // rows_w % 8 == 0: whole chunks only
                        *reinterpret_cast<uint4 *>(yb + ooff[r] + n0 + c) =
                            *reinterpret_cast<const uint4 *>(wsm + rl * kOutLd + c);
                    }
                    __syncthreads();  // the next half overwrites the staged rows
                }
                return;
            }
        }
#pragma unroll
        for (int i = 0; i < TMA; ++i)
#pragma unroll
            for (int j = 0; j < (SWI ? 2 : 4); ++j)
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    const int r = 16 * (2 * i + wm) + wmma_c_row(v);
                    const int c = wn * (SWI ? 32 : 64) + 16 * j + (int)threadIdx.x % 16;
                    if (r < rows && c < rows_w) {
                        if constexpr (SWI) store_h<XH>(y, ooff[r] + n0 + c, silu_f32(acc[i][j][v]) * acc[i][j + 2][v]);
                        else if (COMBINE) put_partial(partial, ooff[r] + n0 + c, acc[i][j][v], pbf16);
                        else store(y, ooff[r] + n0 + c, acc[i][j][v]);
                    }
                }
    };
    switch ((nt + 1) / 2) {  // uniform per block: the busier wave row's tiles
        case 1: k_loop(std::integral_constant<int, 1>{}); break;
        case 2: k_loop(std::integral_constant<int, 2>{}); break;
        case 3: k_loop(std::integral_constant<int, 3>{}); break;
        default: k_loop(std::integral_constant<int, 4>{}); break;
    }
}

// y[m, n] = sum_a coef[m, a] * partial[m * A + a, n], a in order, rounded once; NaN if any of token m's ids is
// invalid (checked here too, so the row is NaN whatever that slot's partial holds). V consecutive columns a thread
// (V = 8: one 16-byte load of BF16 partials a slot, N % 8 == 0); each column's sum is the V = 1 form's, so the
// outputs are bit-identical. One BF16 column a thread (2-byte loads) moved ~171 GB/s at the served shape.
template <int V> struct PartialLoad;
template <> struct PartialLoad<1> {
    __device__ static void run(const float *p, int64_t i, bool bf16, float (&v)[1]) { v[0] = get_partial(p, i, bf16); }
};
template <> struct PartialLoad<8> {
    __device__ static void run(const float *p, int64_t i, bool bf16, float (&v)[8]) {
        if (bf16) {
            const uint4 u = *reinterpret_cast<const uint4 *>(reinterpret_cast<const uint16_t *>(p) + i);
            const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) v[2 * j] = __uint_as_float(w[j] << 16), v[2 * j + 1] = __uint_as_float(w[j] & 0xffff0000u);
        } else {
            const float4 a = reinterpret_cast<const float4 *>(p + i)[0], b = reinterpret_cast<const float4 *>(p + i)[1];
            v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
        }
    }
};
template <typename T, int V> __device__ inline void store_v(T *y, int64_t i, const float (&v)[V]) {
    if constexpr (V == 1) {
        store(y, i, v[0]);
    } else if constexpr (sizeof(T) == 2) {
        uint32_t w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) w[j] = (uint32_t)f32_to_bf16_bits(v[2 * j]) | ((uint32_t)f32_to_bf16_bits(v[2 * j + 1]) << 16);
        *reinterpret_cast<uint4 *>(y + i) = make_uint4(w[0], w[1], w[2], w[3]);
    } else {
        reinterpret_cast<float4 *>(y + i)[0] = make_float4(v[0], v[1], v[2], v[3]);
        reinterpret_cast<float4 *>(y + i)[1] = make_float4(v[4], v[5], v[6], v[7]);
    }
}
template <typename T, int V> __device__ inline void load_v(const T *p, int64_t i, float (&v)[V]) {
    if constexpr (V == 1) {
        v[0] = sizeof(T) == 2 ? bf16_bits_to_f32((uint16_t)p[i]) : (float)p[i];
    } else if constexpr (sizeof(T) == 2) {
        const uint4 u = *reinterpret_cast<const uint4 *>(p + i);
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) v[2 * j] = __uint_as_float(w[j] << 16), v[2 * j + 1] = __uint_as_float(w[j] & 0xffff0000u);
    } else {
        const float4 a = reinterpret_cast<const float4 *>(p + i)[0], b = reinterpret_cast<const float4 *>(p + i)[1];
        v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
    }
}
// The value v reads back as once stored in T (BF16: rounded once).
template <typename T> __device__ inline float as_stored(float v) {
    if constexpr (sizeof(T) == 2) return bf16_bits_to_f32(f32_to_bf16_bits(v));
    else return v;
}
// SH: the shared expert's gated add folded in (MoeSharedAdd): y = stored(sum) + g s, moe_shared_add's arithmetic.
template <typename T, int V, bool SH>
__global__ void __launch_bounds__(kThreads)
    combine_reduce_kernel(const float *partial, const int32_t *ids, const float *coef, int64_t M, int64_t A,
                          int64_t E, int64_t N, T *y, bool pbf16, const T *s, const float *gate_logit, int64_t gate_ld,
                          uint32_t gate_index, uint32_t *err) {
    const int64_t i = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * V;
    if (i >= M * N) return;
    const int64_t m = i / N, n = i % N;  // N % V == 0: the V columns lie in one row
    bool ok = true;
    float v[V] = {};
    for (int64_t a = 0; a < A; ++a) {
        const int32_t id = ids[m * A + a];
        ok = ok && id >= 0 && id < E;
        const float c = coef[m * A + a];
        float p[V];
        PartialLoad<V>::run(partial, (m * A + a) * N + n, pbf16, p);
#pragma unroll
        for (int j = 0; j < V; ++j) v[j] = fmaf(c, p[j], v[j]);
    }
    if (!ok)
#pragma unroll
        for (int j = 0; j < V; ++j) v[j] = __int_as_float(0x7fc00000);
    if constexpr (SH) {
        const float gl = gate_logit[m * gate_ld];
        const bool gok = isfinite(gl);
        if (!gok && n == 0 && atomicCAS(&err[0], 0u, 1u) == 0u) {
            err[1] = (uint32_t)m;
            err[2] = gate_index;
        }
        const float g = gok ? 1.f / (1.f + expf(-gl)) : __builtin_nanf("");
        float sv[V];
        load_v<T, V>(s, i, sv);
#pragma unroll
        for (int j = 0; j < V; ++j) v[j] = as_stored<T>(v[j]) + g * sv[j];
    }
    store_v<T, V>(y, i, v);
}

void check_common(const char *fn, const void *x, const Q4DeviceView &w, int64_t E, int64_t M, int64_t A,
                  const void *ws, size_t ws_bytes, const void *y, Act act) {
    STRIX_CHECK(x != nullptr && y != nullptr && ws != nullptr, fn, ": null pointer (x=", x, ", y=", y, ", ws=", ws, ")");
    STRIX_CHECK(w.q != nullptr, fn, ": null panel pointer");  // kernel.flerp.moe_grouped_p: no scale / min planes
    STRIX_CHECK((w.N / (E >= 1 ? E : 1)) % 16 == 0, fn, ": rows per expert must be whole 16-row panels");
    STRIX_CHECK(w.G == 32 || w.G == 64 || w.G == 128, fn, ": group size ", w.G, " unsupported (32, 64 or 128)");
    STRIX_CHECK(w.K >= w.G && w.K % w.G == 0 && w.K <= (1 << 24), fn, ": K = ", w.K,
                " must be a positive multiple of group size ", w.G, " and <= 2^24");
    STRIX_CHECK(E >= 1 && E <= kMoeMaxExperts && w.N % E == 0, fn, ": weight has ", w.N, " rows for E = ", E,
                " experts (E must be 1..", kMoeMaxExperts, " and divide the rows)");
    STRIX_CHECK(M >= 1 && A >= 1 && A <= 64 && A <= E && M * A <= kMoeMaxSlots, fn, ": M = ", M, " tokens x A = ", A,
                " slots (A 1..min(64, E = ", E, "), M*A <= ", kMoeMaxSlots, ")");
    STRIX_CHECK(ws_bytes >= moe_route_workspace_bytes(M, A, E), fn, ": workspace of ", ws_bytes, " bytes, needs ",
                moe_route_workspace_bytes(M, A, E), " for M = ", M, ", A = ", A, ", E = ", E);
    STRIX_CHECK(reinterpret_cast<uintptr_t>(w.q) % 16 == 0 && reinterpret_cast<uintptr_t>(x) % 16 == 0, fn,
                ": code array ", (const void *)w.q, " and input ", x, " must be 16-byte aligned");
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, fn, ": unsupported activation dtype ", (int)act);
}

void check_no_overlap(const char *fn, const char *a_name, const void *a, int64_t a_bytes, const char *b_name,
                      const void *b, int64_t b_bytes) {
    const auto ab = reinterpret_cast<uintptr_t>(a), bb = reinterpret_cast<uintptr_t>(b);
    STRIX_CHECK(ab + (uintptr_t)a_bytes <= bb || bb + (uintptr_t)b_bytes <= ab, fn, ": ", a_name, " [", a, ", +",
                a_bytes, " B) overlaps ", b_name, " [", b, ", +", b_bytes, " B)");
}

bool is_f16(MoeMath m) {
    return m == MoeMath::WmmaF16 || m == MoeMath::WmmaF16Narrow || m == MoeMath::WmmaF16Wide ||
           m == MoeMath::WmmaF16Wide256;
}
const char *math_label(MoeMath m) {
    switch (m) {
        case MoeMath::F32: return "";
        case MoeMath::WmmaBf16: return " wmma";
        case MoeMath::WmmaF16: return " wmma-f16";
        case MoeMath::WmmaF16Narrow: return " wmma-f16-narrow";
        case MoeMath::WmmaF16Wide: return " wmma-f16-wide";
        case MoeMath::WmmaF16Wide256: return " wmma-f16-wide256";
    }
    return " ?";
}

// MoeMath::WmmaF16's choice: from these token counts on, the gather runs WmmaF16Wide256 and the combine's down
// WmmaF16Wide (its BF16 partials through LDS, 16-byte stores); below them both run WmmaF16Narrow. Measured on real routing (gfx1151, G128; the router ids of a 64k
// repository prompt replayed through the kernels), ms at M = 2048 / 4096 / 8192 / 16384: the SwiGLU gather narrow
// 6.06 / 9.79 / 17.33 / 32.76, wide 6.59 / 9.91 / 16.41 / 29.59, wide256 6.60 / 9.43 / 15.33 / 27.56; the down
// narrow 4.07 / 6.71 / 11.94 / 22.38, wide 4.50 / 7.07 / 12.01 / 21.77, wide256 4.28 / 6.88 / 12.00 / 22.26. Below
// these counts too many experts hold a fraction of a 128-slot tile (51% of 64-slot tile rows carry a slot at 2048, 90%
// at 16384); the down's K = 640 is 20 steps of 32 a block, too few for wide256's longer blocks. The down's epilogue
// through LDS then took wide to 6.65 / 11.26 / 20.41 at 4096 / 8192 / 16384 (4.27 at 2048), below narrow from 4096.
constexpr int64_t kWideMinMGather = 4096, kWideMinMCombine = 4096;

// XH: the F16 kernels' x (COMBINE) is F16 bits, or their SwiGLU h (SWI) is stored as F16 bits (h_bits).
template <typename T, bool COMBINE, int HB, bool SWI = false, bool XH = false>
void launch_grouped(const T *x, const Q4DeviceView &w, int64_t E, const void *ws, int64_t M, int64_t A, T *y,
                    float *partial, MoeMath math, hipStream_t stream) {
    const int64_t N = w.N / E;
    if (math == MoeMath::WmmaF16)
        math = COMBINE ? (M >= kWideMinMCombine ? MoeMath::WmmaF16Wide : MoeMath::WmmaF16Narrow)
                       : (M >= kWideMinMGather ? MoeMath::WmmaF16Wide256 : MoeMath::WmmaF16Narrow);
    STRIX_CHECK(!SWI || (is_f16(math) && N % 2 == 0), "moe grouped (SwiGLU): needs a MoeMath::WmmaF16 kind and an ",
                "even gate_up width, got math ", (int)math, ", N = ", N);
    STRIX_CHECK(!XH || is_f16(math), "moe grouped (F16 h): needs a MoeMath::WmmaF16 kind, got math ", (int)math);
    const auto *wsi = static_cast<const int32_t *>(ws);
    if (math == MoeMath::WmmaF16Wide) {
        STRIX_CHECK(HB == 0, "moe grouped (WMMA F16 wide): Q4 experts only (Q5 runs on MoeMath::WmmaBf16)");
        STRIX_CHECK(w.K % kWmmaKC == 0, "moe grouped (WMMA F16 wide): K = ", w.K, ", expected a multiple of ", kWmmaKC);
        STRIX_CHECK(device_warp_size(kWmmaThreads) == 32, "moe grouped (WMMA F16 wide): needs wave32 (gfx11 WMMA)");
        const int64_t tiles = route_layout(M, A, E).XTmax;
        STRIX_CHECK(tiles <= 65535, "moe grouped (WMMA F16 wide): ", tiles, " tile slots exceed the grid");
        const int64_t cols = SWI ? N / 2 : N, per_block = SWI ? kWmmaBN / 2 : kWmmaBN;  // SWI: h columns
        dim3 grid((unsigned)((cols + per_block - 1) / per_block), (unsigned)tiles), block(kWmmaThreads);
        switch (w.G) {
            case 32: grouped_wmma_f16_wide_kernel<T, 32, COMBINE, SWI, XH><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, y, partial, COMBINE); break;
            case 64: grouped_wmma_f16_wide_kernel<T, 64, COMBINE, SWI, XH><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, y, partial, COMBINE); break;
            default: grouped_wmma_f16_wide_kernel<T, 128, COMBINE, SWI, XH><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, y, partial, COMBINE); break;
        }
        return;
    }
    if (math == MoeMath::WmmaF16Wide256) {
        STRIX_CHECK(HB == 0, "moe grouped (WMMA F16 wide256): Q4 experts only (Q5 runs on MoeMath::WmmaBf16)");
        STRIX_CHECK((w.N / 16) * (8 * w.K + 64 * w.K / w.G) < (int64_t(1) << 32),
                    "moe grouped (WMMA F16 wide256): panel buffer over 4 GiB (the kernel's 32-bit row offsets)");
        STRIX_CHECK(w.K % 32 == 0, "moe grouped (WMMA F16 wide256): K = ", w.K, ", expected a multiple of 32 (its K step)");
        STRIX_CHECK(device_warp_size(kWmmaThreads) == 32, "moe grouped (WMMA F16 wide256): needs wave32 (gfx11 WMMA)");
        const int64_t tiles = route_layout(M, A, E).XTmax;
        STRIX_CHECK(tiles <= 65535, "moe grouped (WMMA F16 wide256): ", tiles, " tile slots exceed the grid");
        // a block: 256 staged weight rows - output columns, or SWI 128 h columns' gate and up rows
        const int64_t cols = SWI ? N / 2 : N, per_block = SWI ? kWmmaThreads / 2 : kWmmaThreads;
        dim3 grid((unsigned)((cols + per_block - 1) / per_block), (unsigned)tiles), block(kWmmaThreads);
        switch (w.G) {
            case 32: grouped_wmma_f16_wide256_kernel<T, 32, COMBINE, SWI, XH><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, y, partial, COMBINE); break;
            case 64: grouped_wmma_f16_wide256_kernel<T, 64, COMBINE, SWI, XH><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, y, partial, COMBINE); break;
            default: grouped_wmma_f16_wide256_kernel<T, 128, COMBINE, SWI, XH><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, y, partial, COMBINE); break;
        }
        return;
    }
    if (math == MoeMath::WmmaF16Narrow) {
        STRIX_CHECK(HB == 0, "moe grouped (WMMA F16): Q4 experts only (Q5 runs on MoeMath::WmmaBf16)");
        STRIX_CHECK(w.K % kWmmaKC == 0, "moe grouped (WMMA F16): K = ", w.K, ", expected a multiple of ", kWmmaKC);
        STRIX_CHECK(device_warp_size(kWmmaThreads) == 32, "moe grouped (WMMA F16): needs wave32 (gfx11 WMMA)");
        const int64_t tiles = route_layout(M, A, E).WTmax;
        STRIX_CHECK(tiles <= 65535, "moe grouped (WMMA F16): ", tiles, " tile slots exceed the grid");
        const int64_t cols = SWI ? N / 2 : N, per_block = SWI ? kWmmaBN / 2 : kWmmaBN;  // SWI: h columns
        dim3 grid((unsigned)((cols + per_block - 1) / per_block), (unsigned)tiles), block(kWmmaThreads);
        switch (w.G) {
            case 32: grouped_wmma_f16_kernel<T, 32, COMBINE, RW, SWI, XH><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, y, partial, COMBINE); break;
            case 64: grouped_wmma_f16_kernel<T, 64, COMBINE, RW, SWI, XH><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, y, partial, COMBINE); break;
            default: grouped_wmma_f16_kernel<T, 128, COMBINE, RW, SWI, XH><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, wsi, A, N, w.K, y, partial, COMBINE); break;
        }
        return;
    }
    // kernel.flerp.moe_grouped_p: MoeMath::WmmaBf16 and F32 read row-major planes only (no panel form yet)
    STRIX_FAIL("moe grouped (panels): math ", (int)math, " has no panel kernel; panel experts run the MoeMath::WmmaF16 ",
               "kinds (Narrow / Wide / Wide256)");
}



template <int HB>
void experts_gather_grouped(const char *fn, const void *x, const Q4DeviceView &w, int64_t E, const void *ws,
                            size_t ws_bytes, int64_t M, int64_t A, void *y, Act act, MoeMath math, hipStream_t stream) {
    STRIX_CHECK(HB == 0 || w.K % 128 == 0, fn, ": Q5 K = ", w.K, " is not a multiple of 128 (the layout's)");
    check_common(fn, x, w, E, M, A, ws, ws_bytes, y, act);
    const int64_t es = act == Act::F32 ? 4 : 2, N = w.N / E;
    check_no_overlap(fn, "y", y, M * A * N * es, "x", x, M * w.K * es);
    if (act == Act::F32)
        launch_grouped<float, false, HB>(static_cast<const float *>(x), w, E, ws, M, A, static_cast<float *>(y), nullptr,
                                     math, stream);
    else
        launch_grouped<uint16_t, false, HB>(static_cast<const uint16_t *>(x), w, E, ws, M, A, static_cast<uint16_t *>(y),
                                        nullptr, math, stream);
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), math_label(math), " M=", M, " A=", A, " E=",
                           E, " N=", N, " K=", w.K, " G=", w.G);
}

void experts_gather_grouped_swiglu(const char *fn, const void *x, const Q4DeviceView &w, int64_t E, const void *ws,
                                   size_t ws_bytes, int64_t M, int64_t A, void *h, Act act, MoeMath math,
                                   hipStream_t stream, bool h_f16) {
    check_common(fn, x, w, E, M, A, ws, ws_bytes, h, act);
    STRIX_CHECK(is_f16(math), fn, ": the fused SwiGLU runs on MoeMath::WmmaF16 (or its Narrow / Wide) only, got ",
                (int)math);
    const int64_t es = act == Act::F32 ? 4 : 2, N = w.N / E;
    STRIX_CHECK(N % 2 == 0, fn, ": gate_up has ", N, " rows an expert, expected an even count (gate rows then up rows)");
    check_no_overlap(fn, "h", h, M * A * (N / 2) * es, "x", x, M * w.K * es);
    STRIX_CHECK(!h_f16 || act == Act::BF16, fn, ": F16 h needs BF16 activations, got ", act_name(act));
    if (act == Act::F32)
        launch_grouped<float, false, 0, true>(static_cast<const float *>(x), w, E, ws, M, A, static_cast<float *>(h),
                                              nullptr, math, stream);
    else if (h_f16)
        launch_grouped<uint16_t, false, 0, true, true>(static_cast<const uint16_t *>(x), w, E, ws, M, A,
                                                       static_cast<uint16_t *>(h), nullptr, math, stream);
    else
        launch_grouped<uint16_t, false, 0, true>(static_cast<const uint16_t *>(x), w, E, ws, M, A,
                                                 static_cast<uint16_t *>(h), nullptr, math, stream);
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), " wmma-f16", h_f16 ? " (F16 h)" : "", " M=", M, " A=", A, " E=", E,
                           " N=", N, " K=", w.K, " G=", w.G);
}

template <int HB>
void experts_combine_grouped(const char *fn, const void *h, const Q4DeviceView &w, int64_t E, const int32_t *ids,
                             const float *coef, const void *ws, size_t ws_bytes, float *partial, int64_t M, int64_t A,
                             void *y, Act act, MoeMath math, hipStream_t stream, const MoeSharedAdd *shared,
                             bool h_f16) {
    STRIX_CHECK(HB == 0 || w.K % 128 == 0, fn, ": Q5 K = ", w.K, " is not a multiple of 128 (the layout's)");
    STRIX_CHECK(!h_f16 || (HB == 0 && act == Act::BF16 && is_f16(math)), fn, ": F16 h needs Q4 experts, BF16 ",
                "activations and a MoeMath::WmmaF16 kind, got ", HB == 0 ? "Q4" : "Q5", " ", act_name(act), " math ",
                (int)math);
    check_common(fn, h, w, E, M, A, ws, ws_bytes, y, act);
    STRIX_CHECK(ids != nullptr && coef != nullptr && partial != nullptr, fn, ": null pointer (ids=",
                (const void *)ids, ", coef=", (const void *)coef, ", partial=", (const void *)partial, ")");
    const int64_t es = act == Act::F32 ? 4 : 2, N = w.N / E;
    check_no_overlap(fn, "y", y, M * N * es, "h", h, M * A * w.K * es);
    check_no_overlap(fn, "y", y, M * N * es, "partial", partial, M * A * N * 4);
    check_no_overlap(fn, "partial", partial, M * A * N * 4, "h", h, M * A * w.K * es);
    if (shared != nullptr) {
        STRIX_CHECK(shared->s != nullptr && shared->gate_logit != nullptr && shared->err != nullptr, fn,
                    ": shared add with a null pointer (s=", shared->s, ", gate_logit=", (const void *)shared->gate_logit,
                    ", err=", (const void *)shared->err, ")");
        STRIX_CHECK(shared->gate_ld >= 1, fn, ": shared add gate_ld = ", shared->gate_ld, ", expected >= 1");
        check_no_overlap(fn, "y", y, M * N * es, "shared s", shared->s, M * N * es);
        check_no_overlap(fn, "partial", partial, M * A * N * 4, "shared s", shared->s, M * N * es);
    }
    const bool pbf16 = is_f16(math);  // the F16 kernels write BF16 partials (put_partial), the others FP32
    // 8 columns a thread when they are whole 16-byte runs (partial, y and s rows: N % 8 == 0, 16-byte aligned bases).
    const bool vec = N % 8 == 0 && reinterpret_cast<uintptr_t>(partial) % 16 == 0 && reinterpret_cast<uintptr_t>(y) % 16 == 0 &&
                     (shared == nullptr || reinterpret_cast<uintptr_t>(shared->s) % 16 == 0);
    const unsigned blocks = (unsigned)((M * N / (vec ? 8 : 1) + kThreads - 1) / kThreads);
    auto reduce = [&](auto *yt) {
        using T = std::remove_pointer_t<decltype(yt)>;
        const T *s = shared ? static_cast<const T *>(shared->s) : nullptr;
        const float *gl = shared ? shared->gate_logit : nullptr;
        const int64_t gld = shared ? shared->gate_ld : 0;
        const uint32_t gi = shared ? shared->gate_index : 0;
        uint32_t *err = shared ? shared->err : nullptr;
        auto go = [&](auto v_c, auto sh_c) {
            combine_reduce_kernel<T, decltype(v_c)::value, decltype(sh_c)::value><<<blocks, kThreads, 0, stream>>>(
                partial, ids, coef, M, A, E, N, yt, pbf16, s, gl, gld, gi, err);
        };
        using V1 = std::integral_constant<int, 1>;
        using V8 = std::integral_constant<int, 8>;
        if (vec && shared) go(V8{}, std::true_type{});
        else if (vec) go(V8{}, std::false_type{});
        else if (shared) go(V1{}, std::true_type{});
        else go(V1{}, std::false_type{});
    };
    if (act == Act::F32) {
        launch_grouped<float, true, HB>(static_cast<const float *>(h), w, E, ws, M, A, nullptr, partial, math, stream);
        reduce(static_cast<float *>(y));
    } else {
        if constexpr (HB == 0)
            if (h_f16)
                launch_grouped<uint16_t, true, HB, false, true>(static_cast<const uint16_t *>(h), w, E, ws, M, A, nullptr,
                                                                partial, math, stream);
        if (!h_f16)
            launch_grouped<uint16_t, true, HB>(static_cast<const uint16_t *>(h), w, E, ws, M, A, nullptr, partial, math,
                                               stream);
        reduce(static_cast<uint16_t *>(y));
    }
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), math_label(math), shared ? " +shared" : "", h_f16 ? " (F16 h)" : "",
                           " M=", M, " A=", A, " E=", E, " N=", N, " K=", w.K, " G=", w.G);
}


flerp::Tally &tally() {
    static flerp::Tally t("kernel.flerp.moe_grouped_p", "panel kernels (Q4Panel16 experts)");
    return t;
}

// A panel view as the launch above takes it: q = the panel buffer, no scale / min planes, the plain group size.
Q4DeviceView launch_view(const flerp::PanelView &w, const char *fn) {
    STRIX_CHECK(w.bits == 4, fn, ": a ", w.bits, "-bit panel view, expected 4 (Q4 experts)");
    return {w.p, nullptr, nullptr, w.N, w.K, w.G};
}

}  // namespace

void linear_q4p_experts_gather_grouped(const void *x, const flerp::PanelView &w, int64_t E, const void *ws,
                                       size_t ws_bytes, int64_t M, int64_t A, void *y, Act act, MoeMath math,
                                       hipStream_t stream) {
    const char *fn = "linear_q4p_experts_gather_grouped";
    experts_gather_grouped<0>(fn, x, launch_view(w, fn), E, ws, ws_bytes, M, A, y, act, math, stream);
}
void linear_q4p_experts_gather_grouped_swiglu(const void *x, const flerp::PanelView &w, int64_t E, const void *ws,
                                              size_t ws_bytes, int64_t M, int64_t A, void *h, Act act, MoeMath math,
                                              hipStream_t stream, bool h_f16) {
    const char *fn = "linear_q4p_experts_gather_grouped_swiglu";
    experts_gather_grouped_swiglu(fn, x, launch_view(w, fn), E, ws, ws_bytes, M, A, h, act, math, stream, h_f16);
}
void linear_q4p_experts_combine_grouped(const void *h, const flerp::PanelView &w, int64_t E, const int32_t *ids,
                                        const float *coef, const void *ws, size_t ws_bytes, float *partial, int64_t M,
                                        int64_t A, void *y, Act act, MoeMath math, hipStream_t stream,
                                        const MoeSharedAdd *shared, bool h_f16) {
    const char *fn = "linear_q4p_experts_combine_grouped";
    experts_combine_grouped<0>(fn, h, launch_view(w, fn), E, ids, coef, ws, ws_bytes, partial, M, A, y, act, math,
                               stream, shared, h_f16);
}

// ---- the adapter: the names strixite's callers use ----
void linear_q4_experts_gather_grouped(const void *x, const Q4DeviceView &w, int64_t E, const void *ws,
                                      size_t ws_bytes, int64_t M, int64_t A, void *y, Act act, MoeMath math,
                                      hipStream_t stream) {
    if (flerp::is_panel(w)) {
        tally().foreign();
        linear_q4p_experts_gather_grouped(x, flerp::panel_of(w), E, ws, ws_bytes, M, A, y, act, math, stream);
    } else {
        tally().origin();
        linear_q4_experts_gather_grouped_flerp_origin(x, w, E, ws, ws_bytes, M, A, y, act, math, stream);
    }
}
void linear_q4_experts_gather_grouped_swiglu(const void *x, const Q4DeviceView &w, int64_t E, const void *ws,
                                              size_t ws_bytes, int64_t M, int64_t A, void *h, Act act, MoeMath math,
                                              hipStream_t stream, bool h_f16) {
    if (flerp::is_panel(w)) {
        tally().foreign();
        linear_q4p_experts_gather_grouped_swiglu(x, flerp::panel_of(w), E, ws, ws_bytes, M, A, h, act, math, stream,
                                                 h_f16);
    } else {
        tally().origin();
        linear_q4_experts_gather_grouped_swiglu_flerp_origin(x, w, E, ws, ws_bytes, M, A, h, act, math, stream, h_f16);
    }
}
void linear_q4_experts_combine_grouped(const void *h, const Q4DeviceView &w, int64_t E, const int32_t *ids,
                                       const float *coef, const void *ws, size_t ws_bytes, float *partial, int64_t M,
                                       int64_t A, void *y, Act act, MoeMath math, hipStream_t stream,
                                       const MoeSharedAdd *shared, bool h_f16) {
    if (flerp::is_panel(w)) {
        tally().foreign();
        linear_q4p_experts_combine_grouped(h, flerp::panel_of(w), E, ids, coef, ws, ws_bytes, partial, M, A, y, act,
                                           math, stream, shared, h_f16);
    } else {
        tally().origin();
        linear_q4_experts_combine_grouped_flerp_origin(h, w, E, ids, coef, ws, ws_bytes, partial, M, A, y, act, math,
                                                       stream, shared, h_f16);
    }
}

}  // namespace strix::kernels
