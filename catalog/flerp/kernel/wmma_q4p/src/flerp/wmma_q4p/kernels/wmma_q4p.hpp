#pragma once

// kernel.flerp.wmma_q4p: kernels/linear_wmma.hpp's Q4 / Q8 prefill GEMM (strixite's linear_wmma_f16_kernel: weights
// dequantized to F16 into LDS once a K step, x as F16, FP32 accumulation) on Q4Panel16 / Q8Panel16 tensors, bit-equal
// to it (kernels/wmma_panel.hpp). As an adapter of kernel.strixite.linear_wmma it defines linear_q4_wmma /
// linear_q8_wmma: a panel-tagged view (runtime/panel_device.hpp) goes to linear_qp_wmma, any other to strixite's
// (renamed <name>_flerp_origin).

#include "kernels/linear_wmma.hpp"
#include "runtime/panel_device.hpp"

namespace strix::kernels {

// y[M, N] = x[M, K] * deq(W)^T, W a panel view (bits 4 or 8); linear_q4_wmma's requirements (K % 64 == 0, wave32).
void linear_qp_wmma(const void *x, const flerp::PanelView &w, void *y, int64_t M, Act act, Act out_act,
                    hipStream_t stream);

}  // namespace strix::kernels
