#pragma once

// Device-side RoPE arithmetic shared by kernels/rope.hip and the fused
// q/k norm + RoPE in kernels/attention_gqa.hip, so both rotate identically
// (transformers' arithmetic: kernels/rope.hpp).

#include "common/hip_runtime.hpp"
#include "kernels/math_utils.hpp"

#include <cstdint>

namespace strix::kernels::rope_math {

using strix::kernels::bf16_bits_to_f32;
using strix::kernels::f32_to_bf16_bits;
using strix::kernels::rb;

// cos/sin of fp32(pos * inv_freq), each times scale in FP32 (YaRN's attention factor - transformers multiplies its
// cos/sin tables by it before the cast; 1 for plain RoPE, which leaves every value bit-identical), then rounded to
// BF16 when kBF16.
template <bool kBF16>
__device__ inline void cos_sin(int64_t pos, float inv_freq, float scale, float &c, float &s) {
    sincosf(__fmul_rn((float)pos, inv_freq), &s, &c);
    c = __fmul_rn(c, scale), s = __fmul_rn(s, scale);
    if (kBF16) c = rb(c), s = rb(s);
}

// (x0, x1) = dims (i, i + rot_dim/2) -> rotated pair; products and sum rounded separately.
template <bool kBF16>
__device__ inline void rotate(float x0, float x1, float c, float s, float &y0, float &y1) {
    if (kBF16) {
        y0 = __fsub_rn(rb(__fmul_rn(x0, c)), rb(__fmul_rn(x1, s)));
        y1 = __fadd_rn(rb(__fmul_rn(x1, c)), rb(__fmul_rn(x0, s)));
    } else {
        y0 = __fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s));
        y1 = __fadd_rn(__fmul_rn(x1, c), __fmul_rn(x0, s));
    }
}

}  // namespace strix::kernels::rope_math
