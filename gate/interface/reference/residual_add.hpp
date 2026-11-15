// SPDX-License-Identifier: MIT
// CPU reference for residual.add/-/f32: x[i] = x[i] + y[i], one IEEE-754 binary32 addition per element, round to
// nearest even. Nothing else. (Compiled without -ffast-math; an fp32 add cannot be contracted or reordered.)
#pragma once
#include <cstddef>

namespace reference {
inline void residual_add_f32(float *x, const float *y, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        volatile float s = x[i] + y[i];      // volatile: one add, stored as binary32, whatever the optimizer thinks
        x[i] = s;
    }
}
}  // namespace reference
