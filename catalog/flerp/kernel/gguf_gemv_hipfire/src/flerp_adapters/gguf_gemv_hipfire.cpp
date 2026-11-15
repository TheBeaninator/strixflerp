// SPDX-License-Identifier: MIT
// gufo's LaunchQ8KBlockGEMV(A, type, x, y, M, K, stream) - the decode GEMV on packed GGUF blocks - on hipfire's
// gemv_q8_0 and gemv_q4k code objects: the adapter docs/INTERFACES.md calls `launch-gguf-gemv`.
//
// What matches without help: the weight bytes (GGUF block_q8_0: f16 scale + 32 int8, 34 bytes; block_q4_K: 144
// bytes per 256 weights, rows of whole blocks), FP32 x [K] and y [M], shape arguments (A, x, y, M, K). hipfire's
// kernels are launched as hipfire launches them: one 32-thread block per row, grid M.
// What differs:
//   formats            hipfire's two kernels cover Q8_0 and Q4_K; every other GGUF type goes to gufo's kernel.
//   shape              M and K are `int` in hipfire's kernels; K must be a whole number of blocks (both sides).
//   numeric contract   same weights, different order: hipfire adds (scale * q) * x element by element in a lane
//                      and reduces 32 lanes with shfl_down; gufo's Q8_0 kernel sums q * x over a block first,
//                      scales the block sum, and walks blocks lane by lane. Tolerance parity, not bit parity.
// Scope: no recipe of this repository consumes a GGUF GEMV yet, so this adapter is gated at module level only.
// The rename covers gufo's own translation units too: LaunchGEMV inside kernel.gufo.qwen_linear_decode keeps
// calling gufo's kernel. A gufo recipe that wants hipfire behind LaunchGEMV would take over that symbol instead.
#include "src/families/qwen/kernels/ops/gemm.hpp"

#include "flerp/adapter_rt.hpp"

#include <climits>

extern "C" const unsigned char flerp_co_hipfire_gemv_q8_0[], flerp_co_hipfire_gemv_q8_0_end[];
extern "C" const unsigned char flerp_co_hipfire_gemv_q4k[], flerp_co_hipfire_gemv_q4k_end[];

namespace gufo::hip {

void LaunchQ8KBlockGEMV_flerp_origin(const void *A, core::GgmlType type, const float *x, float *y, std::size_t M,
                                     std::size_t K, hipStream_t stream);

void LaunchQ8KBlockGEMV(const void *A, core::GgmlType type, const float *x, float *y, std::size_t M, std::size_t K,
                        hipStream_t stream) {
    static flerp::Tally tally("kernel.flerp.gguf_gemv_hipfire", "kernel.hipfire.gemv_ggml:gemv_q8_0,gemv_q4k");
    static flerp::CodeObject q8(flerp_co_hipfire_gemv_q8_0, flerp_co_hipfire_gemv_q8_0_end, "gemv_q8_0");
    static flerp::CodeObject q4k(flerp_co_hipfire_gemv_q4k, flerp_co_hipfire_gemv_q4k_end, "gemv_q4k");
    const std::size_t qk = type == core::GgmlType::kQ8_0 ? 32 : type == core::GgmlType::kQ4_K ? 256 : 0;
    if (qk == 0 || A == nullptr || x == nullptr || y == nullptr || M < 1 || M > INT_MAX || K < qk || K > INT_MAX ||
        K % qk != 0) {
        tally.origin();
        LaunchQ8KBlockGEMV_flerp_origin(A, type, x, y, M, K, stream);
        return;
    }
    tally.foreign();
    const unsigned char *a = static_cast<const unsigned char *>(A);
    int m = (int)M, k = (int)K;
    void *args[] = {&a, &x, &y, &m, &k};
    (type == core::GgmlType::kQ8_0 ? q8 : q4k).launch(dim3((unsigned)M), dim3(32), stream, args);
}

}  // namespace gufo::hip
