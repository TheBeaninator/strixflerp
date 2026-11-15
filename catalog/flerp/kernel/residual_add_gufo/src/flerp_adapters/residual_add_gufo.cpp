// SPDX-License-Identifier: MIT
// strixite's residual_add(x, y, n, act, stream) on gufo's LaunchResidualAdd(a, b, out, dim, stream).
//
// What the adapter does: argument order, and out == a (strixite adds in place, gufo writes a third buffer).
// What it cannot do: gufo has no BF16 form, so a BF16 call goes back to strixite's own kernel.
// Contract notes, checked by the module gate (gate/interface/residual-add.toml):
//  - one IEEE fp32 add per element on both sides, so the result must be bit-identical;
//  - gufo's kernel declares a, b and out __restrict__; the in-place call passes out == a. Each element is one
//    load of a[i], one of b[i] and one store to out[i], so the aliasing cannot reorder anything that matters,
//    but it is outside what the qualifier promises;
//  - y == x (strixite allows it: x doubles) is passed through as a == b == out.
#include "kernels/residual.hpp"

#include "flerp/adapter_rt.hpp"
#include "src/families/qwen/kernels/ops/norm_residual.hpp"

#include <cstddef>

namespace strix::kernels {

void residual_add_flerp_origin(void *x, const void *y, int64_t n, Act act, hipStream_t stream);

void residual_add(void *x, const void *y, int64_t n, Act act, hipStream_t stream) {
    static flerp::Tally tally("kernel.flerp.residual_add_gufo", "kernel.gufo.qwen_norm_rope:LaunchResidualAdd");
    // gufo's grid is (dim + 255) / 256 blocks in an unsigned: stay well inside it.
    if (act != Act::F32 || x == nullptr || y == nullptr || n < 1 || n > (int64_t{1} << 38)) {
        tally.origin();
        residual_add_flerp_origin(x, y, n, act, stream);   // BF16, or an argument error strixite reports itself
        return;
    }
    tally.foreign();
    gufo::hip::LaunchResidualAdd(static_cast<const float *>(x), static_cast<const float *>(y),
                                 static_cast<float *>(x), static_cast<std::size_t>(n), stream);
}

}  // namespace strix::kernels
