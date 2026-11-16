/* include/flerp/decode_loop.h: the decode loop every caller of an executor shares. See the header. */
#include "flerp/decode_loop.h"

#include <string.h>

#define MAX_RUN 64 /* ids of one verify: the token and its drafts */

static size_t take_of(const flerp_feed *f, size_t from, size_t n) {
    size_t take = n - from;
    if (f->max_forward > 0 && (int64_t)take > f->max_forward) take = (size_t)f->max_forward;
    for (size_t c = 0; c < f->n_cuts; ++c) /* a forward ends at a cut inside it */
        if (f->cuts[c] > (int64_t)from && f->cuts[c] < (int64_t)(from + take)) take = (size_t)f->cuts[c] - from;
    return take;
}

flerp_status flerp_feed_ids(const flerp_feed *f, const flerp_token *ids, size_t from, size_t n, const flerp_want *last,
                            flerp_rows *out) {
    flerp_want none;
    memset(&none, 0, sizeof(none));
    none.size = sizeof(none), none.delivery = FLERP_OUT_NONE;
    while (from < n) {
        const size_t take = take_of(f, from, n);
        const int final = last != NULL && from + take == n;
        const size_t next = f->lookahead && from + take < n ? take_of(f, from + take, n) : 0;
        const flerp_status st = flerp_forward(f->ex, f->seq, ids + from, take, final ? last : &none, final ? out : NULL,
                                              next ? ids + from + take : NULL, next);
        if (st != FLERP_OK) return st;
        from += take;
        if (f->snapshot != NULL && f->snapshot_at >= 0 && (int64_t)from == f->snapshot_at) {
            const flerp_status ss = flerp_snapshot_save(f->ex, f->seq, f->snapshot);
            if (ss != FLERP_OK) return ss;
        }
    }
    return FLERP_OK;
}

static int is_end(const flerp_loop *L, flerp_token t) {
    for (size_t i = 0; i < L->n_end; ++i)
        if (L->end_ids[i] == t) return 1;
    return 0;
}

static int fed_at(const flerp_loop_hooks *H, int64_t at, const flerp_token **ids) {
    const flerp_token *none = NULL;
    if (ids == NULL) ids = &none;
    return H->fed ? H->fed(H->ctx, at, ids) : 0;
}

/* fed ids due at `at` that stop drafting there */
static int stops_drafts(const flerp_loop_hooks *H, int64_t at) {
    return H->fed_stops_drafts ? H->fed_stops_drafts(H->ctx, at) : fed_at(H, at, NULL) > 0;
}

static const flerp_masks *masks_of(const flerp_loop *L, const flerp_loop_hooks *H, const flerp_token *pending, size_t n) {
    const flerp_masks *m = H->masks ? H->masks(H->ctx, pending, n) : NULL;
    return m ? m : L->masks;
}

static flerp_want want_of(const flerp_loop *L) {
    flerp_want w;
    memset(&w, 0, sizeof(w));
    w.size = sizeof(w), w.delivery = L->delivery, w.rows = 1, w.n_valid = L->n_valid, w.masks = L->masks;
    return w;
}

/* One forward of ids whose last row is the loop's next row. */
static flerp_status forward(const flerp_loop *L, const flerp_loop_hooks *H, flerp_loop_result *R, const flerp_token *ids, size_t n) {
    flerp_want w = want_of(L);
    w.masks = masks_of(L, H, NULL, 0);
    const flerp_status st = flerp_forward(L->ex, L->seq, ids, n, &w, L->rows, NULL, 0);
    if (st != FLERP_OK) return st;
    if (L->rows->rows != 1) {
        R->error = "a forward that asked one row returned another count";
        return FLERP_E_FAILED;
    }
    ++R->forwards, R->entered += (int64_t)n;
    if (H->entered) H->entered(H->ctx, ids, n);
    return FLERP_OK;
}

/* The drafts after ids[0] into ids[1..]: the chain the head gives while the margin holds (step form), or `steps`
 * held on the device (held form: their values come back with the verify). Returns the count, or a negative status. */
static int draft(const flerp_loop *L, const flerp_loop_hooks *H, int step_form, int64_t steps, flerp_token *ids) {
    if (!step_form) {
        if (steps < 1) return 0;
        const flerp_status st = flerp_draft_hold(L->ex, L->seq, ids[0], (int32_t)steps);
        return st != FLERP_OK ? (int)st : (int)steps;
    }
    int k = 0;
    for (int64_t s = 0; s < steps; ++s) {
        flerp_draft d;
        const flerp_status st = flerp_draft_step(L->ex, L->seq, ids[k], (int32_t)s, &d);
        if (st != FLERP_OK) return (int)st;
        if (d.best - d.second < (float)L->margin || is_end(L, d.token)) break;
        if (H->draft_ok && !H->draft_ok(H->ctx, k, d.token)) break;
        ids[++k] = d.token;
    }
    return k;
}

static flerp_status verify(const flerp_loop *L, const flerp_loop_hooks *H, int step_form, flerp_token *ids, int k) {
    flerp_verify_run run;
    flerp_want w = want_of(L);
    memset(&run, 0, sizeof(run));
    run.size = sizeof(run), run.seq = L->seq, run.out = L->rows, run.ids = ids;
    run.masks = masks_of(L, H, ids + 1, (size_t)k);
    if (!step_form) run.masks = L->masks; /* held drafts are unknown until the verify returns them */
    if (step_form) run.n_ids = (size_t)k + 1;
    else run.n_ids = 1, run.held = k, run.held_out = ids + 1, run.held_cap = MAX_RUN - 1;
    return flerp_verify(L->ex, &run, 1, &w);
}

/* ids fed ahead of the picked token `id` (hooks.before), then id, in one forward. Nonzero: no room (finish length). */
static flerp_status feed_before(const flerp_loop *L, const flerp_loop_hooks *H, flerp_loop_result *R, flerp_token id,
                                const flerp_token *pre, int n_pre, int *no_room) {
    flerp_token run[MAX_RUN];
    *no_room = 0;
    if (R->completion + n_pre >= L->max_tokens) {
        *no_room = 1;
        return FLERP_OK;
    }
    if (n_pre + 1 > MAX_RUN) {
        R->error = "hooks.before gave more ids than one forward of the loop holds";
        return FLERP_E_FAILED;
    }
    memcpy(run, pre, (size_t)n_pre * sizeof(flerp_token));
    run[n_pre] = id;
    R->completion += n_pre;
    if (H->emit) {
        for (int i = 0; i < n_pre; ++i) H->emit(H->ctx, pre[i], 1);
        H->emit(H->ctx, id, 0); /* no stop string completes here: none of it is content */
    }
    const flerp_status st = forward(L, H, R, run, (size_t)n_pre + 1);
    if (st != FLERP_OK) return st;
    if (H->step) H->step(H->ctx, FLERP_FED_STEP, 0, n_pre), H->step(H->ctx, 0, 0, 1);
    return FLERP_OK;
}

flerp_status flerp_decode_loop(const flerp_loop *L, const flerp_loop_hooks *hooks, flerp_loop_result *R) {
    /* the hooks as far as the caller's size: a caller compiled against a shorter struct has the later hooks null */
    flerp_loop_hooks hooks_known;
    const flerp_loop_hooks *H = &hooks_known;
    memset(&hooks_known, 0, sizeof(hooks_known));
    if (hooks == NULL || hooks->size < FLERP_HAS(flerp_loop_hooks, pick)) {
        R->finish = NULL, R->error = "flerp_loop_hooks is null or its size is short of pick";
        return FLERP_E_ARGUMENT;
    }
    memcpy(&hooks_known, hooks, hooks->size < sizeof(hooks_known) ? hooks->size : sizeof(hooks_known));
    if (H->pick == NULL) {
        R->finish = NULL, R->error = "flerp_loop_hooks.pick is required";
        return FLERP_E_ARGUMENT;
    }
    const int step_form = (L->caps->flags & FLERP_CAP_DRAFT_STEP) != 0 && L->drafts > 0;
    const int held_form = !step_form && (L->caps->flags & FLERP_CAP_DRAFT_HELD) != 0 && L->drafts > 0;
    int64_t depth = L->drafts;
    if (depth > L->caps->max_draft) depth = L->caps->max_draft;
    if (depth > L->caps->max_verify_rows - 1) depth = L->caps->max_verify_rows - 1;
    if (depth > MAX_RUN - 1) depth = MAX_RUN - 1;
    flerp_token ids[MAX_RUN], again[MAX_RUN], carry = -1;
    int64_t next_row = 0; /* the row of L->rows the next token is picked from */
    const uint32_t size = R->size;
    memset(R, 0, sizeof(*R));
    R->size = size;
#define TRY(call) do { const flerp_status st_ = (call); if (st_ != FLERP_OK) return st_; } while (0)
    for (;;) {
        if (H->cancelled && H->cancelled(H->ctx)) {
            R->finish = "cancelled";
            break;
        }
        const flerp_token *fed = NULL;
        const int n_fed = fed_at(H, R->entered + (carry >= 0 ? 1 : 0), &fed);
        if (carry >= 0 && n_fed > 0) { /* ids are due while a token is carried: the carried token alone first */
            TRY(forward(L, H, R, &carry, 1));
            carry = -1, next_row = 0;
            if (H->step) H->step(H->ctx, 0, 0, 1);
        }
        if (carry < 0 && n_fed > 0) {
            if (R->completion + n_fed >= L->max_tokens) { /* no room left for them and an answer */
                R->finish = "length";
                break;
            }
            R->completion += n_fed;
            if (H->emit)
                for (int i = 0; i < n_fed; ++i) H->emit(H->ctx, fed[i], 1);
            TRY(forward(L, H, R, fed, (size_t)n_fed));
            next_row = 0;
            if (H->step) H->step(H->ctx, FLERP_FED_STEP, 0, n_fed);
            continue;
        }
        flerp_token id;
        if (carry >= 0) { /* picked, counted and emitted at the rejected position: straight to drafting */
            id = carry, carry = -1;
        } else {
            id = H->pick(H->ctx, L->rows, next_row);
            ++R->completion;
            if (is_end(L, id)) { /* the end of turn is not emitted */
                R->finish = "stop";
                break;
            }
            const flerp_token *pre = NULL;
            const int n_pre = H->before ? H->before(H->ctx, id, &pre) : 0;
            if (n_pre > 0) {
                int no_room = 0;
                TRY(feed_before(L, H, R, id, pre, n_pre, &no_room));
                if (no_room) {
                    R->finish = "length";
                    break;
                }
                next_row = 0;
                continue;
            }
            if (H->emit && H->emit(H->ctx, id, 0)) {
                R->finish = "stop";
                break;
            }
            if (R->completion >= L->max_tokens) {
                R->finish = "length";
                break;
            }
        }
        /* no drafts when fed ids follow id directly (the engine's stop_due) */
        if ((step_form || held_form) && !stops_drafts(H, R->entered + 1)) {
            const int64_t room = L->max_tokens - R->completion;
            const int64_t steps = depth < room ? depth : room;
            ids[0] = id;
            int k = draft(L, H, step_form, steps, ids);
            if (k < 0) return (flerp_status)k;
            if (k > 0) {
                int redone = 0;
                TRY(verify(L, H, step_form, ids, k));
                /* held form: an executor with a draft depth rule of its own may hold fewer than asked (qwen35moe_abi
                   draft_value=on); its verify returns 1 + what it held, and that is the round's k */
                if (!step_form && L->rows->rows >= 1 && L->rows->rows < k + 1) k = (int)L->rows->rows - 1;
                if (L->rows->rows != k + 1) {
                    R->error = "a verify returned a row count other than 1 + its drafts";
                    return FLERP_E_FAILED;
                }
                if (H->redo && H->redo(H->ctx, R->verifies, L->rows, k + 1, 0)) { /* undo the verify, run it again */
                    memcpy(again, ids, sizeof(ids));
                    TRY(flerp_drop(L->ex, L->seq));
                    ++R->redone, redone = 1;
                    int k2 = draft(L, H, step_form, steps, ids);
                    if (k2 < 0) return (flerp_status)k2;
                    if (step_form) {
                        if (k2 == k) TRY(verify(L, H, step_form, ids, k));
                    } else {
                        /* held form: ask the depth of the first round again (k2 == steps) and take the round's k from
                           the returned rows as the first verify did, so an executor that held fewer than asked
                           (qwen35moe_abi draft_value=on) compares equal rather than failing as a changed chain */
                        TRY(verify(L, H, step_form, ids, k2));
                        if (L->rows->rows >= 1 && L->rows->rows < k2 + 1) k2 = (int)L->rows->rows - 1;
                    }
                    if (k2 != k || L->rows->rows != k + 1 || memcmp(again, ids, (size_t)(k + 1) * sizeof(flerp_token))) {
                        R->error = "the draft chain changed after a drop";
                        return FLERP_E_FAILED;
                    }
                    H->redo(H->ctx, R->verifies, L->rows, k + 1, 1);
                }
                flerp_loop_verify info;
                memset(&info, 0, sizeof(info));
                info.index = R->verifies, info.at = R->entered, info.token = id, info.drafts = k, info.redone = redone;
                info.rows = L->rows;
                ++R->verifies, R->drafted += k;
                /* accept drafts while the token picked from the row is the draft */
                int j = 0, ended = 0, cut = 0;
                flerp_token v = -1;
                for (; j < k; ++j) {
                    v = H->pick(H->ctx, L->rows, j);
                    if (v != ids[j + 1] || is_end(L, v)) break; /* a held draft may be an end id: never accepted */
                    ++R->completion, ++R->accepted;
                    const int stopped = H->emit ? H->emit(H->ctx, v, 0) : 0;
                    if (stopped || R->completion >= L->max_tokens) {
                        R->finish = stopped ? "stop" : "length", ended = 1, ++j;
                        break;
                    }
                    /* fed ids are due after this draft with drafts after it still unchecked: stop accepting here */
                    if (j + 1 < k && stops_drafts(H, R->entered + j + 2)) {
                        ++j, cut = 1;
                        break;
                    }
                }
                info.accepted = j, info.cut = cut, info.ended = ended;
                if (H->verified) H->verified(H->ctx, &info);
                if (cut || ended || j == k) { /* id and j accepted drafts stay; row j is the next token's */
                    if (j < k) TRY(flerp_keep_prefix(L->ex, L->seq, j + 1, -1));
                    else TRY(flerp_keep(L->ex, L->seq));
                    R->entered += j + 1;
                    if (H->entered) H->entered(H->ctx, ids, (size_t)j + 1);
                    if (H->step) H->step(H->ctx, k, j, j + 1);
                    if (ended) break;
                    next_row = j;
                    continue;
                }
                /* draft j rejected: v is that position's token; it is carried into the next step's verify */
                ++R->completion;
                const flerp_token *pre = NULL;
                const int n_pre = !is_end(L, v) && H->before ? H->before(H->ctx, v, &pre) : 0;
                TRY(flerp_keep_prefix(L->ex, L->seq, j + 1, n_pre > 0 ? -1 : v));
                R->entered += j + 1;
                if (H->entered) H->entered(H->ctx, ids, (size_t)j + 1);
                if (H->step) H->step(H->ctx, k, j, j + 1);
                if (is_end(L, v)) {
                    R->finish = "stop";
                    break;
                }
                if (n_pre > 0) { /* fed with its ids in a forward of its own, not carried */
                    int no_room = 0;
                    TRY(feed_before(L, H, R, v, pre, n_pre, &no_room));
                    if (no_room) {
                        R->finish = "length";
                        break;
                    }
                    next_row = 0;
                    continue;
                }
                if (H->emit && H->emit(H->ctx, v, 0)) {
                    R->finish = "stop";
                    break;
                }
                if (R->completion >= L->max_tokens) {
                    R->finish = "length";
                    break;
                }
                carry = v;
                continue;
            }
        }
        TRY(forward(L, H, R, &id, 1));
        next_row = 0;
        if (H->step) H->step(H->ctx, 0, 0, 1);
    }
#undef TRY
    return FLERP_OK;
}
