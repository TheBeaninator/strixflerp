#pragma once

// kernel.flerp.linear_q4p: the Q4 decode kernels of kernels/linear_q4.hpp (strixite) reading Q4Panel16 tensors
// (runtime/panel_device.hpp), bit-equal to them on the same dequantized weights (kernels/panel_gemv.hpp says why).
// As an adapter of kernel.strixite.linear_q4 it also defines linear_q4 / linear_q4_experts_gather / _gather_swiglu /
// _combine: a panel-tagged view goes to the functions below, any other view to strixite's kernel (renamed
// <name>_flerp_origin in this assembly), so a call site never chooses and a row-major kernel never sees panels.

#include "kernels/linear_q4.hpp"
#include "runtime/panel_device.hpp"

namespace strix::kernels {

// y[M, N] = x[M, K] * deq(W)^T, W a Q4 panel view (bits 4); requirements as linear_q4's, plus wave32.
void linear_q4p(const void *x, const flerp::PanelView &w, void *y, int64_t M, Act act, Act out_act, hipStream_t stream);
// Experts stacked [E*N, K] as panels (N per expert % 16 == 0): linear_q4_experts_gather / _gather_swiglu / _combine's
// semantics, error word and NaN rule.
void linear_q4p_experts_gather(const void *x, const flerp::PanelView &w, int64_t E, const int32_t *ids, int64_t M,
                               int64_t A, void *y, Act act, uint32_t *err, hipStream_t stream);
void linear_q4p_experts_gather_swiglu(const void *x, const flerp::PanelView &w, int64_t E, const int32_t *ids,
                                      int64_t M, int64_t A, void *h, Act act, uint32_t *err, hipStream_t stream);
void linear_q4p_experts_combine(const void *h, const flerp::PanelView &w, int64_t E, const int32_t *ids,
                                const float *coef, int64_t M, int64_t A, void *y, Act act, uint32_t *err,
                                hipStream_t stream);

}  // namespace strix::kernels
