// SPDX-License-Identifier: MIT
// strixite's router contract for a non-finite logit, for adapters whose foreign router has no such contract:
// a token with a non-finite logit among its E gets ids -1 and coef NaN in all its A slots, and the error word
// gets err[0] = 1, err[1] = the token, err[2] = the lowest non-finite expert index; the first token to flag
// wins (kernels/moe_router.hpp). Launched after the foreign router, on the same stream, so it overwrites
// whatever that kernel made of the row. Include from a .hip file; the kernel is local to the including file.
#pragma once

#include <hip/hip_runtime.h>

#include <climits>
#include <cstdint>

namespace flerp::router {
namespace {

__global__ void __launch_bounds__(32)
    nonfinite_fixup(const float *logits, int64_t ld, int E, int32_t *ids, float *coef, int A, uint32_t *err) {
    const int64_t m = blockIdx.x;
    const int lane = threadIdx.x;
    const float *row = logits + m * ld;
    int bad = INT_MAX;
    for (int e = lane; e < E; e += 32)
        if ((__float_as_uint(row[e]) & 0x7f800000u) == 0x7f800000u && e < bad) bad = e;      // inf or NaN, by bits
    for (int off = 16; off > 0; off >>= 1) bad = min(bad, __shfl_xor(bad, off, 32));
    if (bad == INT_MAX) return;
    if (lane == 0 && atomicCAS(&err[0], 0u, 1u) == 0u) {
        err[1] = (uint32_t)m;
        err[2] = (uint32_t)bad;
    }
    for (int a = lane; a < A; a += 32) {
        ids[m * A + a] = -1;
        coef[m * A + a] = __builtin_nanf("");
    }
}

}  // namespace
}  // namespace flerp::router
