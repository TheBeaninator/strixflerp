/* include/flerp/sampler.h for a recipe whose executor never hands logits to the host (StrixFold draws on the
 * device): there is nothing to sample from, so every call says so. The front end asks for a host sampler only when
 * the executor offers FLERP_OUT_FULL or FLERP_OUT_CANDIDATES. */
#include "flerp/sampler.h"

static const char *const k_none = "no host sampler in this recipe: the executor draws its tokens (flerp_seq_args.sampling)";

const char *flerp_sampler_describe(void) { return "none (the executor draws)"; }
const char *flerp_sampler_last_error(void) { return k_none; }
flerp_status flerp_sampler_new(const flerp_sampling *params, int64_t n_valid, flerp_sampler **out) {
    (void)params, (void)n_valid, (void)out;
    return FLERP_E_UNSUPPORTED;
}
void flerp_sampler_free(flerp_sampler *s) { (void)s; }
int flerp_sampler_takes_candidates(const flerp_sampler *s, int32_t candidates) {
    (void)s, (void)candidates;
    return 0;
}
flerp_status flerp_sampler_pick(flerp_sampler *s, uint32_t delivery, const flerp_rows *rows, int64_t row, int32_t candidates,
                                int64_t logits_row, flerp_token *out) {
    (void)s, (void)delivery, (void)rows, (void)row, (void)candidates, (void)logits_row, (void)out;
    return FLERP_E_UNSUPPORTED;
}
