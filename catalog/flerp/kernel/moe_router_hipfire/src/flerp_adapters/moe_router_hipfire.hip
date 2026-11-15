// SPDX-License-Identifier: MIT
// strixite's moe_router on hipfire's moe_softmax_topk_renorm_k8 code object (unit moe_softmax_topk_k8, compiled
// with hipfire's own hipcc --genco command and embedded by the assembly).
//
// What matches: FP32 logits, 8 int32 ids and 8 FP32 weights per token, softmax then top-8 then renormalize
// (norm_topk = 1), nothing allocated.
// What the adapter has to work around:
//   one token per launch   the kernel routes one row. hipfire's batched twin takes rows exactly n_exp apart;
//                          strixite's rows are E + 1 apart (the gate logit rides along), so the batched kernel
//                          cannot read them. The adapter launches the single-token kernel M times.
//   k is compiled in       8. Any other k goes to strixite's kernel.
//   non-finite logits      hipfire clamps a bad winner to expert 0 with weight 0; strixite fails the token and
//                          reports it. The adapter adds strixite's rule (flerp/router_nonfinite.hpp).
//   shared slot, BF16      as for the gufo adapter: strixite's kernel.
// Where the numeric contracts differ (measured by gate/interface/moe-router.toml):
//   tie rule               strixite: equal logits -> lower expert index. hipfire's arg-max keeps the lower
//                          THREAD on a tie (thread t holds experts t, t + 256, ..), so expert 260 beats expert 5
//                          when both are their threads' bests and equal. Not strixite's rule.
//   selection, weights     on probabilities, with 1 / sum multiplications: last bits differ, and logits one ulp
//                          apart can collapse into a tie.
#include "kernels/moe_router.hpp"

#include "common/hip_check.hpp"
#include "flerp/adapter_rt.hpp"
#include "flerp/router_nonfinite.hpp"

extern "C" const unsigned char flerp_co_hipfire_moe_topk_k8[], flerp_co_hipfire_moe_topk_k8_end[];

namespace strix::kernels {

void moe_router_flerp_origin(const void *logits, int64_t ld, int64_t M, int64_t E, int64_t k, const void *shared_logit,
                             int64_t shared_ld, int32_t shared_id, int32_t *ids, float *coef, Act act, uint32_t *err,
                             hipStream_t stream);

namespace {
constexpr int64_t kMaxTokens = 4096;      // one launch per token: past this, strixite's one-launch kernel
}

void moe_router(const void *logits, int64_t ld, int64_t M, int64_t E, int64_t k, const void *shared_logit,
                int64_t shared_ld, int32_t shared_id, int32_t *ids, float *coef, Act act, uint32_t *err,
                hipStream_t stream) {
    static flerp::Tally tally("kernel.flerp.moe_router_hipfire", "kernel.hipfire.moe:moe_softmax_topk_renorm_k8");
    static flerp::CodeObject co(flerp_co_hipfire_moe_topk_k8, flerp_co_hipfire_moe_topk_k8_end,
                                "moe_softmax_topk_renorm_k8");
    const bool fits = act == Act::F32 && shared_logit == nullptr && logits != nullptr && ids != nullptr &&
                      coef != nullptr && err != nullptr && (const void *)ids != (const void *)coef && M >= 1 &&
                      M <= kMaxTokens && E >= 8 && E <= 1024 && k == 8 && ld >= E;
    if (!fits) {
        tally.origin();
        moe_router_flerp_origin(logits, ld, M, E, k, shared_logit, shared_ld, shared_id, ids, coef, act, err, stream);
        return;
    }
    tally.foreign();
    const float *lg = static_cast<const float *>(logits);
    int n_exp = (int)E, norm_topk = 1;
    for (int64_t m = 0; m < M; ++m) {
        const float *row = lg + m * ld;
        int *idx = ids + m * 8;
        float *w = coef + m * 8;
        void *args[] = {&row, &idx, &w, &n_exp, &norm_topk};
        co.launch(dim3(1), dim3(256), stream, args);
    }
    flerp::router::nonfinite_fixup<<<dim3((unsigned)M), dim3(32), 0, stream>>>(lg, ld, (int)E, ids, coef, 8, err);
    STRIX_HIP_CHECK_LAUNCH("moe_router (hipfire adapter) M=", M, " E=", E);
}

}  // namespace strix::kernels
