#include "kernels/steer.hpp"

#include "common/hip_check.hpp"

#include <cstdint>

namespace strix::kernels {
namespace {

constexpr int kThreads = 256;

__device__ inline float bf16_bits_to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

__global__ void __launch_bounds__(kThreads) add_rows_f32(float *x, const float *v, int64_t n, int64_t cols, float s) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
        x[i] += s * v[i % cols];
}

__global__ void __launch_bounds__(kThreads) add_rows_bf16(uint16_t *x, const float *v, int64_t n, int64_t cols, float s) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
        x[i] = f32_to_bf16_bits(bf16_bits_to_f32(x[i]) + s * v[i % cols]);
}

}  // namespace

void steer_add_rows(void *x, const float *v, int64_t rows, int64_t cols, float scale, Act act, hipStream_t stream) {
    STRIX_CHECK(x != nullptr && v != nullptr, "steer_add_rows: null pointer (x=", x, ", v=", v, ")");
    STRIX_CHECK(rows >= 1 && cols >= 1 && rows <= (1ll << 24) && cols <= (1ll << 20), "steer_add_rows: ", rows, " x ", cols);
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, "steer_add_rows: unsupported activation dtype ", (int)act);
    const int64_t n = rows * cols, blocks = (n + kThreads - 1) / kThreads;
    dim3 grid((unsigned)(blocks < 65535 * 16 ? blocks : 65535 * 16)), block(kThreads);
    if (act == Act::F32)
        add_rows_f32<<<grid, block, 0, stream>>>(static_cast<float *>(x), v, n, cols, scale);
    else
        add_rows_bf16<<<grid, block, 0, stream>>>(static_cast<uint16_t *>(x), v, n, cols, scale);
    STRIX_HIP_CHECK_LAUNCH("steer_add_rows ", act_name(act), " ", rows, "x", cols);
}

}  // namespace strix::kernels
