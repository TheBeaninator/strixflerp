#pragma once

// kernel.flerp.moe_grouped_p: kernels/moe_grouped.hpp's grouped expert GEMMs (strixite) on Q4Panel16 experts, the
// MoeMath::WmmaF16 kinds (Narrow, Wide, Wide256; SwiGLU gather, F16 h, combine with the shared-expert add), bit-equal
// to them on the same dequantized weights. Routing is strixite's moe_group_routes. As an adapter of
// kernel.strixite.moe_grouped it defines linear_q4_experts_gather_grouped / _gather_grouped_swiglu /
// _combine_grouped: a panel-tagged view (runtime/panel_device.hpp) goes to the functions below, any other view to
// strixite's (renamed <name>_flerp_origin). MoeMath::F32 / WmmaBf16 on panels are refused.

#include "kernels/moe_grouped.hpp"
#include "runtime/panel_device.hpp"

namespace strix::kernels {

void linear_q4p_experts_gather_grouped(const void *x, const flerp::PanelView &w, int64_t E, const void *ws,
                                       size_t ws_bytes, int64_t M, int64_t A, void *y, Act act, MoeMath math,
                                       hipStream_t stream);
void linear_q4p_experts_gather_grouped_swiglu(const void *x, const flerp::PanelView &w, int64_t E, const void *ws,
                                              size_t ws_bytes, int64_t M, int64_t A, void *h, Act act, MoeMath math,
                                              hipStream_t stream, bool h_f16 = false);
void linear_q4p_experts_combine_grouped(const void *h, const flerp::PanelView &w, int64_t E, const int32_t *ids,
                                        const float *coef, const void *ws, size_t ws_bytes, float *partial, int64_t M,
                                        int64_t A, void *y, Act act, MoeMath math, hipStream_t stream,
                                        const MoeSharedAdd *shared = nullptr, bool h_f16 = false);

}  // namespace strix::kernels
