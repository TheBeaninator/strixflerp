// SPDX-License-Identifier: MIT
// strixite's linear_bf16w(x, w, y, M, N, K, act, out_act, stream) on StrixFold's tf_ds_dense_gemv code object -
// the adapter docs/INTERFACES.md calls `launch-bf16-gemv`. In a strixite recipe this is the MoE router's logits:
// x BF16 [M, 2560], w BF16 [513, 2560], y FP32 [M, 513].
//
// What matches without help: w is raw BF16 [N, K] row-major on both sides, x is BF16 rows, y is the unrounded
// FP32 accumulator, and neither kernel allocates. StrixFold's kernel is not linked: the assembly compiles
// zig/kernels/hip/dsv41/dense.hip with StrixFold's own hipcc command into a raw gfx1151 code object and embeds
// it; this file loads it and launches it as StrixFold's dense.zig does (block 256, grid ceil(N / 32)), with
//   tf_ds_dense_gemv(w, x, y, k = K, n = N, rows = M, ldx = K, ldy = N, accumulate = 0).
// What differs:
//   numeric contract   both sum exact BF16 x BF16 products in FP32 with one fmaf per element, in different
//                      orders: StrixFold keeps 32 lane partials over tiles of 256 (128 when K % 256 != 0) and
//                      adds them by xor shuffles; strixite splits a row over KS waves when K > 1280 (KS = 2
//                      at K = 2560) and adds the waves' sums last. Same bits when strixite does not split
//                      (K <= 1280, K % 256 == 0), else the last bits differ. Not the same full interface name.
//   preconditions      StrixFold: K a multiple of 128, w and x rows 16-byte aligned, a decode / verify window
//                      (its manifest says 1..128 rows). Outside that, and for every dtype pair other than
//                      BF16 -> FP32, the call goes to strixite's kernel.
#include "kernels/linear.hpp"

#include "flerp/adapter_rt.hpp"

#include <climits>

extern "C" const unsigned char flerp_co_strixfold_dense[], flerp_co_strixfold_dense_end[];

namespace strix::kernels {

void linear_bf16w_flerp_origin(const void *x, const uint16_t *w, void *y, int64_t M, int64_t N, int64_t K, Act act,
                               Act out_act, hipStream_t stream);

namespace {
constexpr int64_t kMaxRows = 128;      // kernel.strixfold.dense_bf16's stated regime
}

void linear_bf16w(const void *x, const uint16_t *w, void *y, int64_t M, int64_t N, int64_t K, Act act, Act out_act,
                  hipStream_t stream) {
    static flerp::Tally tally("kernel.flerp.dense_bf16_strixfold", "kernel.strixfold.dense_bf16:tf_ds_dense_gemv");
    static flerp::CodeObject co(flerp_co_strixfold_dense, flerp_co_strixfold_dense_end, "tf_ds_dense_gemv");
    const bool fits = act == Act::BF16 && out_act == Act::F32 && x != nullptr && w != nullptr && y != nullptr &&
                      M >= 1 && M <= kMaxRows && N >= 1 && N <= INT_MAX - 64 && K >= 128 && K <= INT_MAX &&
                      K % 128 == 0 && reinterpret_cast<uintptr_t>(x) % 16 == 0 &&
                      reinterpret_cast<uintptr_t>(w) % 16 == 0 && reinterpret_cast<uintptr_t>(y) % 4 == 0;
    if (!fits) {
        tally.origin();
        linear_bf16w_flerp_origin(x, w, y, M, N, K, act, out_act, stream);
        return;
    }
    tally.foreign();
    const uint16_t *xp = static_cast<const uint16_t *>(x);
    float *yp = static_cast<float *>(y);
    int32_t k = (int32_t)K, n = (int32_t)N, rows = (int32_t)M, accumulate = 0;
    int64_t ldx = K, ldy = N;
    void *args[] = {&w, &xp, &yp, &k, &n, &rows, &ldx, &ldy, &accumulate};
    co.launch(dim3((unsigned)((N + 31) / 32)), dim3(256), stream, args);
}

void linear_bf16w(const void *x, const uint16_t *w, void *y, int64_t M, int64_t N, int64_t K, Act act,
                  hipStream_t stream) {
    linear_bf16w(x, w, y, M, N, K, act, act, stream);
}

}  // namespace strix::kernels
