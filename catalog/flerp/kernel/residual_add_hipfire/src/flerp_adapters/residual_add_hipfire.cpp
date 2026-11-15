// SPDX-License-Identifier: MIT
// strixite's residual_add(x, y, n, act, stream) on hipfire's add_inplace_f32(a, b, n) code object.
//
// What the adapter does: the launch. hipfire has no host launcher to link: its kernels are HIP source text that the
// runtime compiles with `hipcc --genco` and launches through the module API. The assembly compiles the unit
// `add_inplace` with hipfire's own command and embeds the bundle; this file loads it and launches it the way
// hipfire does (block 256, grid ceil(n / 256): crates/rdna-compute/src/norm.rs).
// What it cannot do: n is an `int` in the kernel, and there is no BF16 form: those calls go back to strixite.
// Contract: one IEEE fp32 add per element, in place: bit-identical by construction; the module gate checks it.
#include "kernels/residual.hpp"

#include "flerp/adapter_rt.hpp"

#include <climits>

extern "C" const unsigned char flerp_co_hipfire_add_inplace[], flerp_co_hipfire_add_inplace_end[];

namespace strix::kernels {

void residual_add_flerp_origin(void *x, const void *y, int64_t n, Act act, hipStream_t stream);

void residual_add(void *x, const void *y, int64_t n, Act act, hipStream_t stream) {
    static flerp::Tally tally("kernel.flerp.residual_add_hipfire", "kernel.hipfire.tensor_ops:add_inplace_f32");
    static flerp::CodeObject co(flerp_co_hipfire_add_inplace, flerp_co_hipfire_add_inplace_end, "add_inplace_f32");
    if (act != Act::F32 || x == nullptr || y == nullptr || n < 1 || n > INT_MAX - 256) {
        tally.origin();
        residual_add_flerp_origin(x, y, n, act, stream);
        return;
    }
    tally.foreign();
    float *a = static_cast<float *>(x);
    const float *b = static_cast<const float *>(y);
    int count = (int)n;
    void *args[] = {&a, &b, &count};
    co.launch(dim3((unsigned)((n + 255) / 256)), dim3(256), stream, args);
}

}  // namespace strix::kernels
