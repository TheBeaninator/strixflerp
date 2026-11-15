// kernel.flerp.linear_q4p: Q4 panel decode kernels and the adapter over kernel.strixite.linear_q4. The expert kernels
// are linear_q4.hip's (gather, gather + SwiGLU, combine) with its q4_row_partial / reduce_row replayed on 8-row panel
// halves exactly as kernels/panel_gemv.hpp does for the dense GEMV: same chunks per origin lane, same order, same
// expressions (the expert partials are the scalar fmaf form for both dtypes, as the origin's), same butterfly, same
// LDS sum.

#include "kernels/linear_q4p.hpp"

#include "flerp/adapter_rt.hpp"
#include "kernels/panel_gemv.hpp"

namespace strix::kernels {

// strixite's kernels, renamed in this assembly (SPEC.md section 5, adapters).
void linear_q4_flerp_origin(const void *x, const Q4DeviceView &w, void *y, int64_t M, Act act, Act out_act,
                            hipStream_t stream);
void linear_q4_experts_gather_flerp_origin(const void *x, const Q4DeviceView &w, int64_t E, const int32_t *ids,
                                           int64_t M, int64_t A, void *y, Act act, uint32_t *err, hipStream_t stream);
void linear_q4_experts_gather_swiglu_flerp_origin(const void *x, const Q4DeviceView &w, int64_t E, const int32_t *ids,
                                                  int64_t M, int64_t A, void *h, Act act, uint32_t *err,
                                                  hipStream_t stream);
void linear_q4_experts_combine_flerp_origin(const void *h, const Q4DeviceView &w, int64_t E, const int32_t *ids,
                                            const float *coef, int64_t M, int64_t A, void *y, Act act, uint32_t *err,
                                            hipStream_t stream);

namespace {

flerp::Tally &tally() {
    static flerp::Tally t("kernel.flerp.linear_q4p", "panel kernels (Q4Panel16)");
    return t;
}

// linear_q4.hip split_k: waves per output row (~2048 codes a wave).
int q4_split_k(int64_t K) {
    int ks = 1;
    while (ks < 8 && K / (2048 * (ks * 2)) >= 1) ks *= 2;
    return ks;
}

__device__ inline bool p_expert_id_ok(int32_t id, int64_t E, uint32_t *err, int64_t slot) {
    if (id >= 0 && id < E) return true;
    if (atomicCAS(err, 0u, 1u) == 0u) {
        err[1] = (uint32_t)id;
        err[2] = (uint32_t)slot;
    }
    return false;
}

// Lane partials of row `row` over its K chunks, as q4_row_partial's lanes: acc[u] = origin lane H (TPW tg + u) + h
// (the wave map of kernels/panel_gemv.hpp).
template <typename T, int G, class MP>
__device__ inline void q4p_row_partials(const uint8_t *w, int64_t P, int64_t codes, int64_t row, const T *xrow,
                                        int64_t chunks, const PRole<MP> &R, int KS, float (&acc)[MP::TPW]) {
    constexpr int TPW = MP::TPW;
    const int ks = R.ks;
    for (int64_t cb = (int64_t)ks * 32; cb < chunks; cb += (int64_t)KS * 32) {
#pragma unroll
        for (int u = 0; u < TPW; ++u) {
            const int64_t c = cb + MP::H * R.step(u) + R.h;
            if (c >= chunks) continue;
            const uint4 packed = *reinterpret_cast<const uint4 *>(flerp::panel_chunk(w, P, row, c));
            const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
            const uint32_t sm = flerp::panel_sm(w, P, codes, row, c * 32 / G);
            const float s = p_bf16_bits_to_f32((uint16_t)(sm & 0xffffu)), mn = p_bf16_bits_to_f32((uint16_t)(sm >> 16));
            float xv[32];
            p_load_x<32>(xrow, c * 32, xv);
            float dot = 0.f, xsum = 0.f;
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                dot = fmaf(xv[i], p_code<4>(words, i), dot);
                xsum += xv[i];
            }
            acc[u] = fmaf(s, dot, fmaf(mn, xsum, acc[u]));
        }
    }
}

template <typename T>
__device__ inline float p_as_stored(float v) {
    if constexpr (std::is_same_v<T, float>) return v;
    else return p_bf16_bits_to_f32(p_f32_to_bf16_bits(v));
}
__device__ inline float p_silu(float v) { return v / (1.0f + expf(-v)); }  // kernels/swiglu.hip's form

// Gather (SWI false) and gather + SwiGLU (SWI true; gate row j and up row j reduced as two independent rows, r = 0, 1):
// grid (row groups, A, M).
template <typename T, int G, bool SWI, class MP>
__global__ void __launch_bounds__(MP::BT)
    q4p_experts_gather_kernel(const T *x, const uint8_t *w, const int32_t *ids, int64_t E, int64_t A, T *y, int64_t N,
                              int64_t K, int KS, uint32_t *err) {
    constexpr int NR = SWI ? 2 : 1;
    constexpr int TPW = MP::TPW;
    __shared__ PRed<MP, NR> sm;
    const PRole<MP> R(KS);
    const int64_t Nout = SWI ? N / 2 : N;
    const int64_t n = R.group * MP::RW + R.rr;
    const int64_t slot = (int64_t)blockIdx.z * A + blockIdx.y;
    const int32_t id = ids[slot];
    const bool ok = id >= 0 && id < E;
    const int64_t P = 8 * K + 64 * K / G, codes = 8 * K;
    float acc[NR][TPW];
#pragma unroll
    for (int r = 0; r < NR; ++r)
#pragma unroll
        for (int u = 0; u < TPW; ++u) acc[r][u] = 0.f;
    if (ok && n < Nout) {  // no early return: every wave reaches the barriers
        const T *xr = x + (int64_t)blockIdx.z * K;
        q4p_row_partials<T, G, MP>(w, P, codes, (int64_t)id * N + n, xr, K / 32, R, KS, acc[0]);
        if constexpr (SWI) q4p_row_partials<T, G, MP>(w, P, codes, (int64_t)id * N + n + N / 2, xr, K / 32, R, KS, acc[NR - 1]);
    }
    float tot[NR];
    p_reduce<MP, NR>(acc, sm, R, KS, tot);
    if (R.writer() && n < Nout) {
        if (!ok && n == 0) p_expert_id_ok(id, E, err, slot);
        if constexpr (SWI) {
            const float g = p_as_stored<T>(tot[0]), u = p_as_stored<T>(tot[NR - 1]);
            const float hv = p_as_stored<T>(p_silu(g)) * u;  // as linear_q4.hip's swiglu epilogue
            p_store(y, slot * Nout + n, ok ? hv : __int_as_float(0x7fc00000));
        } else {
            p_store(y, slot * N + n, ok ? tot[0] : __int_as_float(0x7fc00000));
        }
    }
}

// Combine: y[m, n] = sum_a coef[m, a] * h[m, a, :] * W[ids[m, a]]^T, the chunks of the A*K reduction flattened across
// experts as linear_q4_experts_combine_kernel's lanes walk them. Grid (row groups, M).
template <typename T, int G, class MP>
__global__ void __launch_bounds__(MP::BT)
    q4p_experts_combine_kernel(const T *hin, const uint8_t *w, const int32_t *ids, const float *coef, int64_t E,
                               int64_t A, T *y, int64_t N, int64_t K, int KS, uint32_t *err) {
    constexpr int TPW = MP::TPW;
    __shared__ PRed<MP, 1> sm;
    const PRole<MP> R(KS);
    const int64_t n = R.group * MP::RW + R.rr;
    const int64_t m = blockIdx.y;
    const int64_t P = 8 * K + 64 * K / G, codes = 8 * K;
    const uint32_t kc = (uint32_t)(K / 32), total = (uint32_t)A * kc;
    bool ok = true;
    for (int64_t a = 0; a < A; ++a) ok = ok && ids[m * A + a] >= 0 && ids[m * A + a] < E;
    float acc[1][TPW];
#pragma unroll
    for (int u = 0; u < TPW; ++u) acc[0][u] = 0.f;
    if (ok && n < N) {
        for (uint32_t cb = (uint32_t)(R.ks * 32); cb < total; cb += (uint32_t)(KS * 32)) {
#pragma unroll
            for (int u = 0; u < TPW; ++u) {
                const uint32_t c = cb + (uint32_t)(MP::H * R.step(u) + R.h);
                if (c >= total) continue;
                const uint32_t a = c / kc, cc = c % kc;
                const int64_t row = (int64_t)ids[m * A + a] * N + n;
                const uint4 packed = *reinterpret_cast<const uint4 *>(flerp::panel_chunk(w, P, row, cc));
                const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
                const uint32_t smv = flerp::panel_sm(w, P, codes, row, (int64_t)cc * 32 / G);
                const float s = p_bf16_bits_to_f32((uint16_t)(smv & 0xffffu)),
                            mn = p_bf16_bits_to_f32((uint16_t)(smv >> 16));
                float xv[32];
                p_load_x<32>(hin, (m * A + a) * K + (int64_t)cc * 32, xv);
                float dot = 0.f, xsum = 0.f;
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    dot = fmaf(xv[i], p_code<4>(words, i), dot);
                    xsum += xv[i];
                }
                acc[0][u] = fmaf(coef[m * A + a], fmaf(s, dot, mn * xsum), acc[0][u]);
            }
        }
    }
    float tot[1];
    p_reduce<MP, 1>(acc, sm, R, KS, tot);
    if (R.writer() && n < N) {
        if (!ok && n == 0)
            for (int64_t a = 0; a < A; ++a)
                if (!p_expert_id_ok(ids[m * A + a], E, err, m * A + a)) break;
        p_store(y, m * N + n, ok ? tot[0] : __int_as_float(0x7fc00000));
    }
}

// linear_q4.hip linear_q4_experts' checks and launch, on a panel view.
void q4p_experts(const char *fn, const void *x, const flerp::PanelView &w, int64_t E, const int32_t *ids,
                 const float *coef, int64_t M, int64_t A, void *y, Act act, uint32_t *err, hipStream_t stream,
                 bool swiglu) {
    STRIX_CHECK(x != nullptr && y != nullptr && ids != nullptr && err != nullptr && w.p != nullptr, fn,
                ": null pointer (x=", x, ", y=", y, ", ids=", (const void *)ids, ", err=", (const void *)err,
                ", panels=", (const void *)w.p, ")");
    STRIX_CHECK(w.bits == 4, fn, ": a ", w.bits, "-bit panel view, expected 4");
    STRIX_CHECK(w.G == 32 || w.G == 64 || w.G == 128, fn, ": group size ", w.G, " unsupported (32, 64 or 128)");
    STRIX_CHECK(w.K >= w.G && w.K <= (1 << 24) && w.K % w.G == 0, fn, ": K = ", w.K, " must be a positive multiple of ",
                w.G, " and <= 2^24");
    STRIX_CHECK(reinterpret_cast<uintptr_t>(w.p) % 16 == 0, fn, ": panels must be 16-byte aligned");
    STRIX_CHECK(E >= 1 && w.N % E == 0, fn, ": weight has ", w.N, " rows, not a multiple of E = ", E, " experts");
    const int64_t N = w.N / E;
    STRIX_CHECK(N % 16 == 0, fn, ": ", N, " rows per expert, not whole 16-row panels");
    STRIX_CHECK(M >= 1 && M <= 65535, fn, ": M (tokens) = ", M, ", expected 1..65535");
    STRIX_CHECK(A >= 1 && A <= 64 && A <= E, fn, ": A (active experts per token) = ", A, ", expected 1..min(64, E=", E,
                ")");
    STRIX_CHECK(reinterpret_cast<uintptr_t>(x) % 16 == 0, fn, ": input ", x, " must be 16-byte aligned");
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, fn, ": unsupported activation dtype ", (int)act);
    STRIX_CHECK(device_warp_size(kPThreads) == 32, fn, ": the panel kernels are written for wave32 (gfx11)");
    const bool combine = coef != nullptr;
    STRIX_CHECK(!(swiglu && combine), fn, ": the SwiGLU epilogue is for gathers only");
    STRIX_CHECK(!combine || A * (w.K / 32) <= (int64_t)UINT32_MAX, fn, ": A x K / 32 must be < 2^32");
    const int64_t es = act == Act::F32 ? 4 : 2;
    {
        const auto ib = reinterpret_cast<uintptr_t>(x), ob = reinterpret_cast<uintptr_t>(y);
        const uintptr_t in_b = (uintptr_t)((combine ? M * A : M) * w.K * es),
                        out_b = (uintptr_t)((combine ? M : M * A) * (swiglu ? N / 2 : N) * es);
        STRIX_CHECK(ob + out_b <= ib || ib + in_b <= ob, fn, ": output overlaps input");
    }
    const int KS = q4_split_k(w.K);
    const PChoice map = p_map(KS, 1, 4, w.K, N, true);
    const int64_t out_rows = swiglu ? N / 2 : N;
    auto go = [&](auto tc) {
        using T = decltype(tc);
        const T *xt = static_cast<const T *>(x);
        T *yt = static_cast<T *>(y);
        p_with_map(map, KS, [&](auto mp) {
            using MP = decltype(mp);
            const int64_t groups_per_block = MP::BT / 32 / (MP::NT * KS);
            const unsigned bx = (unsigned)(((out_rows + MP::RW - 1) / MP::RW + groups_per_block - 1) / groups_per_block);
            const dim3 block(MP::BT);
#define FLERP_Q4P_EXPERTS(G)                                                                                         \
    if (swiglu)                                                                                                      \
        q4p_experts_gather_kernel<T, G, true, MP><<<dim3(bx, (unsigned)A, (unsigned)M), block, 0, stream>>>(    \
            xt, w.p, ids, E, A, yt, N, w.K, KS, err);                                                                \
    else if (combine)                                                                                                \
        q4p_experts_combine_kernel<T, G, MP><<<dim3(bx, (unsigned)M), block, 0, stream>>>(xt, w.p, ids, coef, E, \
                                                                                             A, yt, N, w.K, KS, err); \
    else                                                                                                             \
        q4p_experts_gather_kernel<T, G, false, MP><<<dim3(bx, (unsigned)A, (unsigned)M), block, 0, stream>>>(   \
            xt, w.p, ids, E, A, yt, N, w.K, KS, err);
            switch (w.G) {
                case 32: FLERP_Q4P_EXPERTS(32) break;
                case 64: FLERP_Q4P_EXPERTS(64) break;
                default: FLERP_Q4P_EXPERTS(128) break;
            }
#undef FLERP_Q4P_EXPERTS
        });
    };
    if (act == Act::F32) go(float{});
    else go(uint16_t{});
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), " M=", M, " A=", A, " E=", E, " N=", N, " K=", w.K, " G=", w.G,
                           swiglu ? " swiglu" : "");
}

}  // namespace

void linear_q4p(const void *x, const flerp::PanelView &w, void *y, int64_t M, Act act, Act out_act, hipStream_t stream) {
    panel_gemv<4>("linear_q4p", x, w, y, M, act, out_act, q4_split_k(w.K), stream);
}
void linear_q4p_experts_gather(const void *x, const flerp::PanelView &w, int64_t E, const int32_t *ids, int64_t M,
                               int64_t A, void *y, Act act, uint32_t *err, hipStream_t stream) {
    q4p_experts("linear_q4p_experts_gather", x, w, E, ids, nullptr, M, A, y, act, err, stream, false);
}
void linear_q4p_experts_gather_swiglu(const void *x, const flerp::PanelView &w, int64_t E, const int32_t *ids,
                                      int64_t M, int64_t A, void *h, Act act, uint32_t *err, hipStream_t stream) {
    q4p_experts("linear_q4p_experts_gather_swiglu", x, w, E, ids, nullptr, M, A, h, act, err, stream, true);
}
void linear_q4p_experts_combine(const void *h, const flerp::PanelView &w, int64_t E, const int32_t *ids,
                                const float *coef, int64_t M, int64_t A, void *y, Act act, uint32_t *err,
                                hipStream_t stream) {
    STRIX_CHECK(coef != nullptr, "linear_q4p_experts_combine: null coef");
    q4p_experts("linear_q4p_experts_combine", h, w, E, ids, coef, M, A, y, act, err, stream, false);
}

// ---- the adapter: the names strixite's callers use ----
void linear_q4(const void *x, const Q4DeviceView &w, void *y, int64_t M, Act act, hipStream_t stream) {
    linear_q4(x, w, y, M, act, act, stream);
}
void linear_q4(const void *x, const Q4DeviceView &w, void *y, int64_t M, Act act, Act out_act, hipStream_t stream) {
    if (flerp::is_panel(w)) {
        tally().foreign();
        linear_q4p(x, flerp::panel_of(w), y, M, act, out_act, stream);
    } else {
        tally().origin();
        linear_q4_flerp_origin(x, w, y, M, act, out_act, stream);
    }
}
void linear_q4_experts_gather(const void *x, const Q4DeviceView &w, int64_t E, const int32_t *ids, int64_t M,
                              int64_t A, void *y, Act act, uint32_t *err, hipStream_t stream) {
    if (flerp::is_panel(w)) {
        tally().foreign();
        linear_q4p_experts_gather(x, flerp::panel_of(w), E, ids, M, A, y, act, err, stream);
    } else {
        tally().origin();
        linear_q4_experts_gather_flerp_origin(x, w, E, ids, M, A, y, act, err, stream);
    }
}
void linear_q4_experts_gather_swiglu(const void *x, const Q4DeviceView &w, int64_t E, const int32_t *ids, int64_t M,
                                     int64_t A, void *h, Act act, uint32_t *err, hipStream_t stream) {
    if (flerp::is_panel(w)) {
        tally().foreign();
        linear_q4p_experts_gather_swiglu(x, flerp::panel_of(w), E, ids, M, A, h, act, err, stream);
    } else {
        tally().origin();
        linear_q4_experts_gather_swiglu_flerp_origin(x, w, E, ids, M, A, h, act, err, stream);
    }
}
void linear_q4_experts_combine(const void *h, const Q4DeviceView &w, int64_t E, const int32_t *ids,
                               const float *coef, int64_t M, int64_t A, void *y, Act act, uint32_t *err,
                               hipStream_t stream) {
    if (flerp::is_panel(w)) {
        tally().foreign();
        linear_q4p_experts_combine(h, flerp::panel_of(w), E, ids, coef, M, A, y, act, err, stream);
    } else {
        tally().origin();
        linear_q4_experts_combine_flerp_origin(h, w, E, ids, coef, M, A, y, act, err, stream);
    }
}

}  // namespace strix::kernels
