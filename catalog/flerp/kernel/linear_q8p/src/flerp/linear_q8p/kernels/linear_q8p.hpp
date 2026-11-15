#pragma once

// kernel.flerp.linear_q8p: kernels/linear_q8.hpp's decode GEMV (strixite) on Q8Panel16 tensors
// (runtime/panel_device.hpp), bit-equal to it (kernels/panel_gemv.hpp of kernel.flerp.linear_q4p says why). As an
// adapter of kernel.strixite.linear_q8 it defines linear_q8: a panel-tagged view goes to linear_q8p, any other to
// strixite's kernel (renamed linear_q8_flerp_origin).

#include "kernels/linear_q8.hpp"
#include "runtime/panel_device.hpp"

namespace strix::kernels {

void linear_q8p(const void *x, const flerp::PanelView &w, void *y, int64_t M, Act act, Act out_act, hipStream_t stream);

}  // namespace strix::kernels
