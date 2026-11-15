// kernel.flerp.wmma_q4p: linear_wmma.hip's linear_wmma_f16_kernel and its launch (strixite 20937dc), the weight
// tile fetched by WmmaWCodesP (kernels/wmma_panel.hpp) in place of WmmaWCodes. Changes from the origin are marked
// `kernel.flerp.wmma_q4p`.
//
// Fragment layout and why the LDS staging stays (wmma.hpp:6-8 of kernel.strixite.wmma, wmma_gemm.hpp:39-55): a B
// fragment is lane l < 16 holding row l's 16 K values and lanes 16..31 holding the same - each lane needs its row's
// whole 16-value run, and both half-waves need it. Here 8 waves tile 2 (x rows) x 4 (weight rows), so every 32-row
// weight slab is used by two waves; decoding into registers would dequantize each code four times (two waves, two
// half-waves) where the LDS tile dequantizes it once, and this kernel is VALU-bound (linear_wmma.hip:16-18). So the
// panel layout changes only the global loads (whole 128-byte lines instead of 32 bytes of each), and the LDS tile and
// every instruction after it are the origin's.

#include "kernels/wmma_q4p.hpp"

#include "common/hip_check.hpp"
#include "flerp/adapter_rt.hpp"
#include "kernels/device_info.hpp"
#include "kernels/wmma_panel.hpp"

#include <cstdint>
#include <type_traits>

namespace strix::kernels {

void linear_q4_wmma_flerp_origin(const void *x, const Q4DeviceView &w, void *y, int64_t M, Act act, Act out_act,
                                 hipStream_t stream);
void linear_q8_wmma_flerp_origin(const void *x, const Q8DeviceView &w, void *y, int64_t M, Act act, Act out_act,
                                 hipStream_t stream);

namespace {

constexpr int BN = 128;
constexpr int64_t kMaxM = 65535ll * 64;

__device__ inline void store(float *p, int64_t i, float v) { p[i] = v; }
__device__ inline void store(uint16_t *p, int64_t i, float v) { p[i] = wmma_f32_to_bf16(v); }

flerp::Tally &tally() {
    static flerp::Tally t("kernel.flerp.wmma_q4p", "panel kernels (Q4Panel16 / Q8Panel16)");
    return t;
}

// linear_wmma.hip linear_wmma_f16_kernel; kernel.flerp.wmma_q4p: q is the panel buffer, the tile WmmaWCodesP.
template <typename TX, typename TY, int BITS, int G, int BMT>
__global__ void __launch_bounds__(kWmmaThreads)
    linear_wmma_f16_p_kernel(const TX *x, const uint8_t *q, TY *y, int64_t M, int64_t N, int64_t K) {
    static_assert(BITS == 4 || BITS == 8, "F16 path: Q4 / Q8 weights");
    constexpr int TM = BMT / 32;
    using WT = WmmaWCodesP<BITS, G, BN>;  // kernel.flerp.wmma_q4p
    __shared__ __attribute__((aligned(16))) uint16_t xs[BMT * kWmmaLd], ws[BN * kWmmaLd];
    __shared__ uint16_t sw[BN * WT::kWin], mw[BN * WT::kWin];
    __shared__ int64_t xoff[BMT];
    const int64_t m0 = (int64_t)blockIdx.y * BMT, n0 = (int64_t)blockIdx.x * BN;
    const int rows_x = (int)(M - m0 < BMT ? M - m0 : BMT), rows_w = (int)(N - n0 < BN ? N - n0 : BN);
    if (threadIdx.x < BMT) xoff[threadIdx.x] = (m0 + threadIdx.x) * K;
    const int wave = threadIdx.x / 32, wm = wave / 4, wn = wave % 4;
    wmma_f32x8 acc[TM][2] = {};
    __syncthreads();
    WmmaXTile<TX, BMT, G, false> xt;
    WT wt;
    xt.fetch(x, xoff, rows_x, 0);
    wt.fetch(q, nullptr, nullptr, n0, rows_w, K, 0);
    for (int64_t k0 = 0; k0 < K; k0 += kWmmaKC) {
        if (wt.win) {  // uniform: depends on k0 only
            wt.stage_window(sw, mw);
            __syncthreads();
        }
        xt.stage_f16(xs);
        wt.stage_f16(ws, sw, mw, k0);
        __syncthreads();
        if (k0 + kWmmaKC < K) {
            xt.fetch(x, xoff, rows_x, k0 + kWmmaKC);
            wt.fetch(q, nullptr, nullptr, n0, rows_w, K, k0 + kWmmaKC);
        }
        wmma_step_f16<TM, 2>(xs, ws, wm * (BMT / 2), wn * 32, acc);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int64_t m = m0 + wm * (BMT / 2) + i * 16 + wmma_c_row(v), n = n0 + wn * 32 + j * 16 + threadIdx.x % 16;
                if (m < M && n < N) store(y, m * N + n, acc[i][j][v]);
            }
}

int device_mps() {
    static const int n = [] {
        int dev = 0, v = 0;
        STRIX_HIP_CHECK(hipGetDevice(&dev), "wmma_q4p: current device");
        STRIX_HIP_CHECK(hipDeviceGetAttribute(&v, hipDeviceAttributeMultiprocessorCount, dev), "wmma_q4p: MP count");
        STRIX_CHECK(v >= 1, "wmma_q4p: device reports ", v, " multiprocessors");
        return v;
    }();
    return n;
}

// linear_wmma.hip launch: the 128-row x tile when it still gives >= 8 blocks per WGP, else 64.
template <typename TX, typename TY, int BITS>
void launch(const TX *x, const flerp::PanelView &w, TY *y, int64_t M, hipStream_t stream) {
    const int64_t col_blocks = (w.N + BN - 1) / BN;
    auto go = [&](auto bmt) {
        constexpr int BMT = decltype(bmt)::value;
        dim3 grid((unsigned)col_blocks, (unsigned)((M + BMT - 1) / BMT)), block(kWmmaThreads);
        switch (w.G) {
            case 32: linear_wmma_f16_p_kernel<TX, TY, BITS, 32, BMT><<<grid, block, 0, stream>>>(x, w.p, y, M, w.N, w.K); break;
            case 64: linear_wmma_f16_p_kernel<TX, TY, BITS, 64, BMT><<<grid, block, 0, stream>>>(x, w.p, y, M, w.N, w.K); break;
            default: linear_wmma_f16_p_kernel<TX, TY, BITS, 128, BMT><<<grid, block, 0, stream>>>(x, w.p, y, M, w.N, w.K); break;
        }
    };
    if (col_blocks * ((M + 127) / 128) >= 8 * (int64_t)device_mps()) go(std::integral_constant<int, 128>{});
    else go(std::integral_constant<int, 64>{});
}

}  // namespace

void linear_qp_wmma(const void *x, const flerp::PanelView &w, void *y, int64_t M, Act act, Act out_act,
                    hipStream_t stream) {
    const char *fn = w.bits == 8 ? "linear_q8p_wmma" : "linear_q4p_wmma";
    STRIX_CHECK(w.bits == 4 || w.bits == 8, fn, ": a ", w.bits, "-bit panel view, expected 4 or 8");
    STRIX_CHECK(x != nullptr && y != nullptr && w.p != nullptr, fn, ": null pointer (x=", x, ", y=", y, ", panels=",
                (const void *)w.p, ")");
    STRIX_CHECK(w.G == 32 || w.G == 64 || w.G == 128, fn, ": group size ", w.G, " unsupported (32, 64 or 128)");
    STRIX_CHECK(M >= 1 && M <= kMaxM, fn, ": M (rows of x) = ", M, ", expected 1..", kMaxM);
    STRIX_CHECK(w.N >= 1 && w.N <= 0x7fffffff, fn, ": N = ", w.N, ", expected 1..2^31-1");
    STRIX_CHECK(w.K >= kWmmaKC && w.K % kWmmaKC == 0 && w.K % w.G == 0 && w.K <= (1 << 24), fn, ": K = ", w.K,
                ", expected a multiple of ", kWmmaKC, " (the K step) and of the group size ", w.G, ", <= 2^24");
    STRIX_CHECK(reinterpret_cast<uintptr_t>(w.p) % 16 == 0 && reinterpret_cast<uintptr_t>(x) % 16 == 0, fn,
                ": panels ", (const void *)w.p, " and x ", x, " must be 16-byte aligned");
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, fn, ": unsupported activation dtype ", (int)act);
    STRIX_CHECK(out_act == act || (act == Act::BF16 && out_act == Act::F32), fn, ": output dtype ", act_name(out_act),
                " with input ", act_name(act), " unsupported (same dtype, or BF16 -> F32)");
    STRIX_CHECK(device_warp_size(kWmmaThreads) == 32, fn, ": needs wave32 (gfx11 WMMA)");
    const int64_t es = act == Act::F32 ? 4 : 2, oes = out_act == Act::F32 ? 4 : 2;
    const auto xb = reinterpret_cast<uintptr_t>(x), yb = reinterpret_cast<uintptr_t>(y);
    STRIX_CHECK(yb + (uintptr_t)(M * w.N * oes) <= xb || xb + (uintptr_t)(M * w.K * es) <= yb, fn, ": output y [", y,
                ", +", M * w.N * oes, " B) overlaps input x [", x, ", +", M * w.K * es, " B)");
    auto go = [&](auto bits_c) {
        constexpr int BITS = decltype(bits_c)::value;
        if (act == Act::F32) launch<float, float, BITS>(static_cast<const float *>(x), w, static_cast<float *>(y), M, stream);
        else if (out_act == Act::F32)
            launch<uint16_t, float, BITS>(static_cast<const uint16_t *>(x), w, static_cast<float *>(y), M, stream);
        else launch<uint16_t, uint16_t, BITS>(static_cast<const uint16_t *>(x), w, static_cast<uint16_t *>(y), M, stream);
    };
    if (w.bits == 4) go(std::integral_constant<int, 4>{});
    else go(std::integral_constant<int, 8>{});
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), "->", act_name(out_act), " M=", M, " N=", w.N, " K=", w.K, " G=", w.G);
}

void linear_q4_wmma(const void *x, const Q4DeviceView &w, void *y, int64_t M, Act act, Act out_act, hipStream_t stream) {
    if (flerp::is_panel(w)) {
        tally().foreign();
        linear_qp_wmma(x, flerp::panel_of(w), y, M, act, out_act, stream);
    } else {
        tally().origin();
        linear_q4_wmma_flerp_origin(x, w, y, M, act, out_act, stream);
    }
}
void linear_q8_wmma(const void *x, const Q8DeviceView &w, void *y, int64_t M, Act act, Act out_act, hipStream_t stream) {
    if (flerp::is_panel(w)) {
        tally().foreign();
        linear_qp_wmma(x, flerp::panel_of(w), y, M, act, out_act, stream);
    } else {
        tally().origin();
        linear_q8_wmma_flerp_origin(x, w, y, M, act, out_act, stream);
    }
}

}  // namespace strix::kernels
