/* The decode loop over include/flerp/executor.h that every caller of an executor shares: the front end
 * (serve.flerp.http) and the conformance driver (executor.flerp.conformance) run this one, not copies of it.
 *
 * It is the decode loop of strixite's serving engine (Engine::run, serve/engine.cpp 1256-1691 at 20937dc) written
 * over the ABI, with the origin's acceptance rule: a token is sampled from the trunk's row; the executor's draft
 * head proposes up to `drafts` tokens after it; one verify runs the token and its drafts; drafts are accepted while
 * the token picked from each verified row is the draft; the accepted prefix is kept (flerp_keep / flerp_keep_prefix)
 * and the token picked at the first rejected row is carried into the next verify instead of a forward of its own.
 * The two draft forms of the ABI differ only in who decides the chain:
 *   FLERP_CAP_DRAFT_STEP  (strixite) the chain grows while the head's top-2 margin reaches `margin` and the draft is
 *                         not an end id - the engine's mtp-margin rule;
 *   FLERP_CAP_DRAFT_HELD  (StrixFold) the executor holds `drafts` tokens for the verify, which returns them.
 * With drafts = 0, or an executor without a draft head, every token is one flerp_forward: the plain loop.
 *
 * What a caller decides from the reply's text stays with the caller, as hooks: which token a row gives (`pick`: a
 * sampler's draw, or the token the executor drew), whether the reply ends at a token (`emit`: stop strings), and ids
 * the caller feeds instead of sampling (`fed`: strixite's thinking-budget stop text). Rows are sampled in the order
 * the engine samples them and only when the engine would, so a seeded sampler behind `pick` makes the same draws.
 */
#ifndef FLERP_DECODE_LOOP_H
#define FLERP_DECODE_LOOP_H

#include "flerp/executor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --- prompt forwards ----------------------------------------------------------------------------------------- */
typedef struct flerp_feed {
    uint32_t size;
    flerp_executor *ex;
    int32_t seq;
    int64_t max_forward;      /* caps.max_forward */
    const int64_t *cuts;      /* a forward ends at each of these positions inside it (strixite's logits depend on */
    size_t n_cuts;            /* where a prompt is cut: docs/EXECUTOR-ABI.md, "Forward shape") */
    int lookahead;            /* pass the next forward's ids as flerp_forward's hint, as strixite's engine does */
    int64_t snapshot_at;      /* -1, or: save a snapshot when a forward ends at this position ... */
    flerp_snapshot *snapshot; /* ... into here (the engine's turn snapshot) */
} flerp_feed;

/* ids[from, n) into the seq in forwards of at most max_forward ids that end at the cuts. `last` (may be null): what
 * the final forward asks for; its rows go to `out`. */
flerp_status flerp_feed_ids(const flerp_feed *f, const flerp_token *ids, size_t from, size_t n,
                            const flerp_want *last, flerp_rows *out);

/* --- the loop ------------------------------------------------------------------------------------------------ */
typedef struct flerp_loop_verify { /* one verify, for a trace */
    int64_t index, at;        /* verifies before it; tokens in the sequence after the prompt before it */
    flerp_token token;        /* row 0's input */
    int drafts, accepted;
    int redone, cut, ended;
    const flerp_rows *rows;
} flerp_loop_verify;

typedef struct flerp_loop_hooks {
    void *ctx;
    /* Required. The token of row `row` of `rows`, which hold the loop's delivery. Called once a row, in sequence
     * order, only for rows whose token the reply may take. */
    flerp_token (*pick)(void *ctx, const flerp_rows *rows, int64_t row);
    /* A token joined the reply: picked and not an end id (fed = 0), or fed by the caller (fed = 1). Nonzero from a
     * picked token ends the reply with finish "stop". */
    int (*emit)(void *ctx, flerp_token token, int fed);
    /* Ids the caller feeds instead of sampling once `at` tokens have entered the sequence after the prompt: their
     * count (0: none) and *ids. A pure function of `at` within one loop. */
    int (*fed)(void *ctx, int64_t at, const flerp_token **ids);
    /* Called with a verify's rows before they are resolved (again = 0). Nonzero: the loop drops the verify
     * (flerp_drop), drafts and verifies again, and calls back with again = 1 (the return is then unread). */
    int (*redo)(void *ctx, int64_t verify, const flerp_rows *rows, int64_t n_rows, int again);
    void (*step)(void *ctx, int drafted, int accepted, int emitted); /* strixite's capture record of a step */
    void (*entered)(void *ctx, const flerp_token *ids, size_t n);    /* ids entered the sequence (were forwarded) */
    void (*verified)(void *ctx, const flerp_loop_verify *v);
    int (*cancelled)(void *ctx);
    /* Optional, for the model's reply policy (serve.flerp.http binds them to include/flerp/reply.h): */
    /* Ids fed ahead of a picked token that is not a draft (strixite's think close before a tool call): their count
     * (0: none) and *ids. The token is picked and counted; the ids are counted and emitted (fed = 1), then the token
     * is emitted, and both enter in one forward, with no drafts from the token. No room for the ids under
     * max_tokens: the reply ends "length". */
    int (*before)(void *ctx, flerp_token token, const flerp_token **ids);
    /* Whether a stepped draft `d` may join the chain at position k (0: the first draft after the picked token);
     * the chain ends at the first one refused. Null: every draft the margin passes. */
    int (*draft_ok)(void *ctx, int k, flerp_token d);
    /* Whether ids `fed` would give at `at` stop the drafts and the acceptance of a verify there (the engine's
     * stop_due, which a nudge does not count in). Null: any ids `fed` gives do. */
    int (*fed_stops_drafts)(void *ctx, int64_t at);
    /* Masks for a forward or verify of the reply: row r follows the reply so far plus pending[0, r) (n_pending + 1
     * rows; pending: a verify's drafts). Null return, or a null hook: the loop's `masks` ride on it. */
    const flerp_masks *(*masks)(void *ctx, const flerp_token *pending, size_t n_pending);
} flerp_loop_hooks;

typedef struct flerp_loop {
    uint32_t size;
    flerp_executor *ex;
    int32_t seq;              /* its prompt has been forwarded and `rows` hold the prompt's last row */
    const flerp_caps *caps;
    uint32_t delivery;        /* the delivery of `rows` and of every forward and verify of the loop */
    int64_t n_valid;
    const flerp_masks *masks; /* null, or a logit bias (masks->rows = 0) that rides on every forward and verify */
    flerp_rows *rows;         /* the caller's buffers for caps.max_verify_rows rows of `delivery` */
    int64_t max_tokens;       /* the engine's completion tokens: picked (the end id too) and fed */
    int32_t drafts;           /* most drafts a verify (0: none); bounded by caps.max_draft and caps.max_verify_rows - 1 */
    double margin;            /* FLERP_CAP_DRAFT_STEP: the top-2 logit margin a draft needs */
    const flerp_token *end_ids;
    size_t n_end;
} flerp_loop;

enum { FLERP_FED_STEP = 255 }; /* `drafted` of a step that fed ids (serve/capture.hpp kThinkingStopStep) */

typedef struct flerp_loop_result {
    uint32_t size;
    const char *finish;       /* "stop", "length", "cancelled"; null when the loop failed */
    const char *error;        /* the loop's own reason for a failure, or null: flerp_last_error has the executor's */
    int64_t completion;       /* completion tokens */
    int64_t entered;          /* tokens that entered the sequence after the prompt */
    int64_t forwards, verifies, redone, drafted, accepted;
} flerp_loop_result;

flerp_status flerp_decode_loop(const flerp_loop *loop, const flerp_loop_hooks *hooks, flerp_loop_result *result);

#ifdef __cplusplus
}
#endif
#endif /* FLERP_DECODE_LOOP_H */
