/* strixflerp host sampler ABI, version 1: next-token sampling on the host from rows an executor hands back.
 *
 * Only an executor that offers FLERP_OUT_FULL or FLERP_OUT_CANDIDATES needs one (strixite); an executor that draws
 * on the device (StrixFold) is sampled by flerp_seq_args.sampling. A front end reaches the sampler through this
 * header and the recipe binds the module (sample.host/flerp-v1/-):
 *   runtime.flerp.sampler_strixite  strixite's Sampler (serve/sampler.hpp): the origin's draws, bit for bit
 *   runtime.flerp.sampler_none      no host sampler: flerp_sampler_new is FLERP_E_UNSUPPORTED (a recipe whose
 *                                   executor never hands logits)
 */
#ifndef FLERP_SAMPLER_H
#define FLERP_SAMPLER_H

#include "flerp/executor.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FLERP_SAMPLER_ABI 1

typedef struct flerp_sampler flerp_sampler; /* opaque: one request's sampler (its random state) */

/* What the module is, for /health and logs; never null. */
const char *flerp_sampler_describe(void);
/* `params` as flerp_seq_args.sampling (temperature 0 = greedy); tokens come from ids [0, n_valid).
 * FLERP_E_ARGUMENT: a parameter the sampler does not take or out of range (the client's error);
 * FLERP_E_UNSUPPORTED: this module samples nothing. flerp_sampler_last_error says which. */
flerp_status flerp_sampler_new(const flerp_sampling *params, int64_t n_valid, flerp_sampler **out);
void flerp_sampler_free(flerp_sampler *s);
const char *flerp_sampler_last_error(void);
/* 1 when the sampler's draw needs only each row's top `candidates` (FLERP_OUT_CANDIDATES), 0 when it needs full
 * rows. */
int flerp_sampler_takes_candidates(const flerp_sampler *s, int32_t candidates);
/* The token of row `row` of `rows`, which hold `delivery` (FLERP_OUT_CANDIDATES with `candidates` a row, or
 * FLERP_OUT_FULL with `logits_row` floats a row). One call a row, in sequence order: each draws from the sampler's
 * random state. */
flerp_status flerp_sampler_pick(flerp_sampler *s, uint32_t delivery, const flerp_rows *rows, int64_t row,
                                int32_t candidates, int64_t logits_row, flerp_token *out);

#ifdef __cplusplus
}
#endif
#endif /* FLERP_SAMPLER_H */
