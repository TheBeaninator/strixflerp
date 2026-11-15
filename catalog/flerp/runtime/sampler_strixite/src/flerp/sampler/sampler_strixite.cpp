// include/flerp/sampler.h over strixite's Sampler (serve/sampler.hpp): the origin's host sampler, so a front end that
// picks rows in the engine's order makes the engine's draws (temperature, top-k, top-p, one mt19937_64 seeded with
// the request's seed). min_p is not a parameter of this sampler and is refused.
#include "flerp/sampler.h"

#include "common/check.hpp"
#include "serve/sampler.hpp"

#include <memory>
#include <string>

using namespace strix;

struct flerp_sampler {
    std::unique_ptr<Sampler> s;
};

namespace {
thread_local std::string g_error;
}

extern "C" {

const char *flerp_sampler_describe(void) { return "strixite Sampler (host; temperature, top_k, top_p, seed)"; }
const char *flerp_sampler_last_error(void) { return g_error.c_str(); }

flerp_status flerp_sampler_new(const flerp_sampling *params, int64_t n_valid, flerp_sampler **out) {
    if (out == nullptr) return FLERP_E_ARGUMENT;
    SamplingParams p;
    p.temperature = 0, p.top_k = 0, p.top_p = 1, p.seed = 0;  // null params: greedy
    if (params != nullptr) {
        if (params->min_p > 0) {
            g_error = "min_p is not supported by this sampler (temperature, top_k, top_p, seed)";
            return FLERP_E_ARGUMENT;
        }
        p.temperature = params->temperature, p.top_k = params->top_k, p.top_p = params->top_p, p.seed = params->seed;
    }
    try {
        auto s = std::make_unique<flerp_sampler>();
        s->s = std::make_unique<Sampler>(p, n_valid);
        *out = s.release();
    } catch (const std::exception &e) {
        g_error = e.what();
        return FLERP_E_ARGUMENT;
    }
    return FLERP_OK;
}

void flerp_sampler_free(flerp_sampler *s) { delete s; }

int flerp_sampler_takes_candidates(const flerp_sampler *s, int32_t candidates) {
    return candidates >= Sampler::kCandidates && s->s->takes_candidates() ? 1 : 0;
}

flerp_status flerp_sampler_pick(flerp_sampler *s, uint32_t delivery, const flerp_rows *rows, int64_t row, int32_t candidates,
                                int64_t logits_row, flerp_token *out) {
    try {
        if (delivery == FLERP_OUT_CANDIDATES) {
            const bool nan = rows->nan != nullptr && rows->nan[row] != 0;
            *out = s->s->sample_candidates(rows->cand_logit + row * candidates, rows->cand_id + row * candidates, candidates, nan);
        } else if (delivery == FLERP_OUT_FULL) {
            *out = s->s->sample(rows->full + row * logits_row, (size_t)logits_row);
        } else {
            g_error = "flerp_sampler_pick: rows of a delivery without logits";
            return FLERP_E_ARGUMENT;
        }
    } catch (const std::exception &e) {
        g_error = e.what();
        return FLERP_E_FAILED;
    }
    return FLERP_OK;
}

}  // extern "C"
