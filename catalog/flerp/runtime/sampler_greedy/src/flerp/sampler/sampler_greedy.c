/* include/flerp/sampler.h, greedy only: the token of a row is its largest logit over ids [0, n_valid), the lower id
 * first on a tie (the order of executor.flerp.qwen35moe's device argmax and of the conformance driver's greedy()).
 * A request with temperature above 0 is the client's error (FLERP_E_ARGUMENT, so the front end answers HTTP 400):
 * this recipe draws nothing at random. With temperature 0 the other fields are unread, as flerp_sampling says. */
#include "flerp/sampler.h"

#include <stdlib.h>

struct flerp_sampler {
    int64_t n_valid;
};

static const char *g_error = "";

const char *flerp_sampler_describe(void) { return "greedy (runtime.flerp.sampler_greedy: argmax of each row, temperature 0 only)"; }
const char *flerp_sampler_last_error(void) { return g_error; }

flerp_status flerp_sampler_new(const flerp_sampling *params, int64_t n_valid, flerp_sampler **out) {
    if (out == NULL || n_valid < 1) {
        g_error = "flerp_sampler_new: null out or no valid ids";
        return FLERP_E_ARGUMENT;
    }
    if (params != NULL && params->temperature > 0) {
        g_error = "this recipe samples greedily only: send temperature 0 (or leave it out where the server's default is 0)";
        return FLERP_E_ARGUMENT;
    }
    flerp_sampler *s = (flerp_sampler *)malloc(sizeof(*s));
    if (s == NULL) {
        g_error = "out of memory";
        return FLERP_E_FAILED;
    }
    s->n_valid = n_valid;
    *out = s;
    return FLERP_OK;
}

void flerp_sampler_free(flerp_sampler *s) { free(s); }

/* candidates are value descending with the lower id first on a tie, so candidate 0 is the greedy token */
int flerp_sampler_takes_candidates(const flerp_sampler *s, int32_t candidates) {
    (void)s;
    return candidates >= 1;
}

flerp_status flerp_sampler_pick(flerp_sampler *s, uint32_t delivery, const flerp_rows *rows, int64_t row, int32_t candidates,
                                int64_t logits_row, flerp_token *out) {
    if (s == NULL || rows == NULL || out == NULL || row < 0 || row >= rows->rows) {
        g_error = "flerp_sampler_pick: bad arguments";
        return FLERP_E_ARGUMENT;
    }
    if (delivery == FLERP_OUT_CANDIDATES) {
        if (rows->cand_id == NULL || candidates < 1) {
            g_error = "flerp_sampler_pick: no candidates";
            return FLERP_E_ARGUMENT;
        }
        *out = rows->cand_id[row * candidates];
        return FLERP_OK;
    }
    if (delivery != FLERP_OUT_FULL || rows->full == NULL || logits_row < 1) {
        g_error = "flerp_sampler_pick: the rows hold neither candidates nor full rows";
        return FLERP_E_ARGUMENT;
    }
    const float *x = rows->full + row * logits_row;
    const int64_t n = s->n_valid < logits_row ? s->n_valid : logits_row;
    int64_t best = 0;
    for (int64_t i = 1; i < n; ++i)
        if (x[i] > x[best]) best = i;
    *out = (flerp_token)best;
    return FLERP_OK;
}
