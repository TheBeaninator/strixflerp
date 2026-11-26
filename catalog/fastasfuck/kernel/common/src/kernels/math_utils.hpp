#pragma once

// Common device-side mathematical and type conversion primitives for AMD GPU kernels.

#include "common/hip_runtime.hpp"
#include "kernels/types.hpp"

#include <cmath>
#include <cstdint>

namespace strix::kernels {

__device__ inline float bf16_bits_to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }

__device__ inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

// Round FP32 float to nearest-even BF16 representation in float domain.
__device__ inline float rb(float v) { return bf16_bits_to_f32(f32_to_bf16_bits(v)); }

__device__ inline float h2f(uint32_t bits16) { return (float)__builtin_bit_cast(_Float16, (uint16_t)bits16); }
__device__ inline uint16_t f2h(float f) { return __builtin_bit_cast(uint16_t, (_Float16)f); }  // RNE

}  // namespace strix::kernels
