// SPDX-License-Identifier: MIT
// strixite's moe_router (top-k experts per token + softmax weights renormalized over the k) on gufo's RouterTopK
// for Qwen3.8-Flash-Next - the adapter docs/INTERFACES.md calls `router-slots`.
//
// What matches without help: FP32 logits rows `stride` apart (strixite passes ld = E + 1: the gate logit rides
// in the same row), ids [M, k] int32 and weights [M, k] FP32, neither side allocates.
//   RouterTopK(logits, stride = ld, ids, weights = coef, n_tokens = M, n_experts = E, k, stream)
// What the adapter adds: strixite's non-finite contract (flerp/router_nonfinite.hpp), one more launch.
// What it does not do:
//   shared slot       with a stacked shared expert strixite's router also writes slot k (id shared_id, coef
//                     sigmoid(gate logit)), so the rows are k + 1 wide; gufo's rows are k wide and its shared
//                     expert lives in its MoE epilogue. Those calls go to strixite's kernel. (The pod's weights
//                     keep the shared expert separate, so the served model does not take that path.)
//   BF16 logits       gufo reads FP32 only: strixite's kernel.
// Where the numeric contracts differ (same job, not the same bits - measured by gate/interface/moe-router.toml):
//   selection         strixite picks on the logits; gufo picks on probabilities exp(l - max) / sum over all E,
//                     computed with -ffast-math. Two logits one ulp apart can collapse to one probability, and
//                     then gufo's tie rule (lower index) decides instead of the logit order.
//   weights           strixite: exp(l_j - l_max) / sum over the k. gufo: p_j / max(sum of the k p's, 2^-14),
//                     p from the full softmax: two more divisions, so the last bits differ.
//   the unit          RouterTopK is one of 83 kernels in gufo's 9,352-line kernels.hip.cpp; the whole unit is
//                     compiled (gufo's flags, -ffast-math) and the linker keeps what the call reaches.
#include "kernels/moe_router.hpp"

#include "common/hip_check.hpp"
#include "flerp/adapter_rt.hpp"
#include "flerp/router_nonfinite.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace strix::kernels {

void moe_router_flerp_origin(const void *logits, int64_t ld, int64_t M, int64_t E, int64_t k, const void *shared_logit,
                             int64_t shared_ld, int32_t shared_id, int32_t *ids, float *coef, Act act, uint32_t *err,
                             hipStream_t stream);

void moe_router(const void *logits, int64_t ld, int64_t M, int64_t E, int64_t k, const void *shared_logit,
                int64_t shared_ld, int32_t shared_id, int32_t *ids, float *coef, Act act, uint32_t *err,
                hipStream_t stream) {
    static flerp::Tally tally("kernel.flerp.moe_router_gufo", "kernel.gufo.qfn_kernels:RouterTopK");
    const bool fits = act == Act::F32 && shared_logit == nullptr && logits != nullptr && ids != nullptr &&
                      coef != nullptr && err != nullptr && (const void *)ids != (const void *)coef && M >= 1 &&
                      M <= 0x7fffffff && E >= 1 && E <= 1024 && k >= 1 && k <= 32 && k <= E && ld >= E &&
                      ld <= 0xffffffffll;
    if (!fits) {
        tally.origin();
        moe_router_flerp_origin(logits, ld, M, E, k, shared_logit, shared_ld, shared_id, ids, coef, act, err, stream);
        return;
    }
    tally.foreign();
    const float *lg = static_cast<const float *>(logits);
    gufo::models::qwen38_flash_next::rocm::RouterTopK(lg, (std::uint32_t)ld, ids, coef, (std::uint32_t)M,
                                                      (std::uint32_t)E, (std::uint32_t)k, stream);
    flerp::router::nonfinite_fixup<<<dim3((unsigned)M), dim3(32), 0, stream>>>(lg, ld, (int)E, ids, coef, (int)k, err);
    STRIX_HIP_CHECK_LAUNCH("moe_router (gufo adapter) M=", M, " E=", E, " k=", k);
}

}  // namespace strix::kernels
