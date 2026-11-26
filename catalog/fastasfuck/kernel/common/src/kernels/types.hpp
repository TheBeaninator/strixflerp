#pragma once

// Common kernel types and scalar declarations across all strix compute primitives.

#include "common/hip_runtime.hpp"

#include <cstdint>

namespace strix::kernels {

enum class Act { F32, BF16 };

inline const char *act_name(Act a) {
    switch (a) {
        case Act::F32: return "F32";
        case Act::BF16: return "BF16";
    }
    return "unknown";
}

}  // namespace strix::kernels
