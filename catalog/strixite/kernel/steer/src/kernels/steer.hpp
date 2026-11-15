#pragma once

// Activation steering: x[r, k] += scale * v[k] for rows r of x, FP32 add and one rounding to the activation dtype.

#include "kernels/norm.hpp"  // Act

#include "common/hip_runtime.hpp"

#include <cstdint>

namespace strix::kernels {

// x: rows * cols activations (row-major); v: cols FP32 values on the device.
void steer_add_rows(void *x, const float *v, int64_t rows, int64_t cols, float scale, Act act, hipStream_t stream);

}  // namespace strix::kernels
