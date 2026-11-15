/* FROZEN: executor.flerp.fake_abi as of ABI version 2 (commit 825cc98): the version 2 executor of the cross-version
 * tests in tests/test_flerp_serve.py. Never edit. */
/* A stand-in executor behind include/flerp/executor.h, for testing callers of the ABI without a GPU: no weights,
 * no device. Its "model" is a hash: the token after a context is a function of the context's last token and its
 * length, so every way of grouping ids into forwards gives the same tokens (drafts are output-invariant), and a
 * caller's loop can be checked against itself. What it offers is chosen at flerp_open, so one caller can be run
 * against each shape of executor the ABI allows:
 *
 *   vocab=N         ids are [0, N) (required); row=N the floats of a full row (default vocab)
 *   lo=A hi=B       the model's tokens come from [A, B) (default 0 and vocab)
 *   reply=N         after N tokens past the prompt the model gives end=ID (default 24 and vocab - 1)
 *   deliveries=LIST of full,candidates,tokens (default all three)
 *   draft=none|step|held   the draft head's form (default none); every 5th draft is wrong, and a step draft's
 *                   margin is 0.5 where the position is a multiple of 7 and 3 elsewhere
 *   sampled=on|off  FLERP_CAP_SAMPLED_TOKENS (default off); top_k_max=N, needs_top_k=on|off: flerp_caps version 2
 *   masks=on|off    FLERP_CAP_MASKS, a logit bias only (default on)
 *   positions=N     (default 4096); seqs=N (default 1)
 *   abi=1|2         what flerp_get_caps reports and fills (default 2)
 * A sampled draw (tokens delivery, temperature > 0) is the row's candidate number (seed + position) mod min(top_k, 3):
 * a function of the seed, not a distribution. */
#include "flerp/executor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define K 20
#define MAX_ROWS 16
#define MAX_SEQS 8

typedef struct {
    int open, pending, sampled, top_k;
    uint64_t seed;
    int64_t pos, prompt, v_rows;
    flerp_token last, v_ids[MAX_ROWS];
    flerp_token held[MAX_ROWS];
    int n_held;
} seq_t;

struct flerp_executor {
    int64_t vocab, row, lo, hi, reply, positions;
    flerp_token end;
    uint32_t deliveries;
    int draft, sampled, masks, top_k_max, needs_top_k, abi, n_seqs;
    seq_t seqs[MAX_SEQS];
    flerp_stats stats;
    char error[256], describe[256];
    float *rowbuf;
};

static char g_open_error[256];

static flerp_status fail(const flerp_executor *ex, flerp_status st, const char *what) {
    snprintf(((flerp_executor *)ex)->error, sizeof(ex->error), "%s", what);
    return st;
}

static uint32_t mix(uint64_t x) {
    x ^= x >> 33, x *= 0xff51afd7ed558ccdULL, x ^= x >> 33, x *= 0xc4ceb9fe1a85ec53ULL, x ^= x >> 33;
    return (uint32_t)x;
}

/* The model: the token after a context of `len` ids that ends with `last`. */
static flerp_token model(const flerp_executor *ex, const seq_t *q, flerp_token last, int64_t len) {
    if (q->prompt >= 0 && len - q->prompt >= ex->reply) return ex->end;
    return (flerp_token)(ex->lo + mix(((uint64_t)(uint32_t)last << 32) ^ (uint64_t)len) % (uint64_t)(ex->hi - ex->lo));
}

/* The full row after that context: the model's token at 10, its successor at 8, the rest below 5. */
static void full_row(const flerp_executor *ex, const seq_t *q, flerp_token last, int64_t len, const flerp_masks *m, float *x) {
    const flerp_token t = model(ex, q, last, len);
    for (int64_t i = 0; i < ex->row; ++i) x[i] = i < ex->vocab ? (float)(mix((uint64_t)i * 0x9e3779b97f4a7c15ULL + (uint64_t)len) % 1000) / 250.0f : -1e30f;
    x[t] = 10.0f;
    x[(t + 1) % ex->vocab] = 8.0f;
    if (m != NULL)
        for (size_t i = 0; i < m->n_bias; ++i) x[m->bias_id[i]] += m->bias[i];
}

static void top_k(const float *x, int64_t n, flerp_token *id, float *v) {
    for (int c = 0; c < K; ++c) id[c] = -1, v[c] = -INFINITY;
    for (int64_t i = 0; i < n; ++i) {
        if (!(x[i] > v[K - 1])) continue; /* value descending, the lower id first on a tie */
        int c = K - 1;
        for (; c > 0 && x[i] > v[c - 1]; --c) id[c] = id[c - 1], v[c] = v[c - 1];
        id[c] = (flerp_token)i, v[c] = x[i];
    }
}

/* Row `r` of a result: the context ends with `last` and is `len` long. */
static flerp_status deliver(flerp_executor *ex, seq_t *q, flerp_token last, int64_t len, const flerp_want *w, const flerp_masks *m,
                            flerp_rows *out, int64_t r) {
    const int64_t n = w->n_valid > 0 && w->n_valid < ex->vocab ? w->n_valid : ex->vocab;
    if (!(ex->deliveries & w->delivery)) return fail(ex, FLERP_E_UNSUPPORTED, "fake: this delivery is not offered (caps.deliveries)");
    if (m != NULL && (!ex->masks || m->rows != 0)) return fail(ex, FLERP_E_UNSUPPORTED, "fake: masks are not offered (a logit bias only, with masks=on)");
    if (out == NULL) return fail(ex, FLERP_E_ARGUMENT, "results were asked for without a flerp_rows");
    if (w->delivery == FLERP_OUT_FULL) {
        if (out->full == NULL || out->full_cap < (size_t)((r + 1) * ex->row)) return fail(ex, FLERP_E_ARGUMENT, "flerp_rows.full is too small");
        full_row(ex, q, last, len, m, out->full + r * ex->row);
    } else {
        flerp_token id[K];
        float v[K];
        full_row(ex, q, last, len, m, ex->rowbuf);
        top_k(ex->rowbuf, n, id, v);
        if (w->delivery == FLERP_OUT_CANDIDATES) {
            if (out->cand_id == NULL || out->cand_logit == NULL || out->cand_cap < (size_t)((r + 1) * K)) return fail(ex, FLERP_E_ARGUMENT, "flerp_rows.cand_* are too small");
            memcpy(out->cand_id + r * K, id, sizeof(id)), memcpy(out->cand_logit + r * K, v, sizeof(v));
        } else if (w->delivery == FLERP_OUT_TOKENS) {
            if (out->tokens == NULL || out->tokens_cap < (size_t)(r + 1)) return fail(ex, FLERP_E_ARGUMENT, "flerp_rows.tokens is too small");
            int pick = 0;
            if (q->sampled) pick = (int)((q->seed + (uint64_t)len) % (uint64_t)(q->top_k > 0 && q->top_k < 3 ? q->top_k : 3));
            out->tokens[r] = id[pick];
        } else return fail(ex, FLERP_E_ARGUMENT, "fake: not one FLERP_OUT_* value");
    }
    if (out->nan != NULL && out->nan_cap > (size_t)r) out->nan[r] = 0;
    out->rows = r + 1;
    return FLERP_OK;
}

static const char *value_of(const flerp_open_args *a, const char *key, const char *otherwise) {
    for (size_t i = 0; i < a->n_config; ++i)
        if (!strcmp(a->config[i].key, key)) return a->config[i].value;
    return otherwise;
}

flerp_status flerp_open(const flerp_open_args *args, flerp_executor **out) {
    static const char *const keys[] = {"vocab", "row", "lo", "hi", "reply", "end", "deliveries", "draft", "sampled", "top_k_max",
                                       "needs_top_k", "masks", "positions", "seqs", "abi"};
    g_open_error[0] = 0;
    if (args == NULL || out == NULL) return FLERP_E_ARGUMENT;
    for (size_t i = 0; i < args->n_config; ++i) {
        int known = 0;
        for (size_t k = 0; k < sizeof(keys) / sizeof(*keys); ++k) known |= !strcmp(args->config[i].key, keys[k]);
        if (!known) {
            snprintf(g_open_error, sizeof(g_open_error), "flerp_open (fake): unknown key '%s'", args->config[i].key);
            return FLERP_E_CONFIG;
        }
    }
    flerp_executor *ex = calloc(1, sizeof(*ex));
    ex->vocab = atoll(value_of(args, "vocab", "0"));
    if (ex->vocab < 4) {
        snprintf(g_open_error, sizeof(g_open_error), "flerp_open (fake): vocab is required (>= 4)");
        free(ex);
        return FLERP_E_CONFIG;
    }
    ex->row = atoll(value_of(args, "row", "0"));
    if (ex->row < ex->vocab) ex->row = ex->vocab;
    ex->lo = atoll(value_of(args, "lo", "0"));
    ex->hi = atoll(value_of(args, "hi", "0"));
    if (ex->hi <= ex->lo || ex->hi > ex->vocab) ex->hi = ex->vocab;
    ex->reply = atoll(value_of(args, "reply", "24"));
    ex->end = (flerp_token)atoll(value_of(args, "end", "-1"));
    if (ex->end < 0) ex->end = (flerp_token)(ex->vocab - 1);
    ex->positions = atoll(value_of(args, "positions", "4096"));
    ex->n_seqs = atoi(value_of(args, "seqs", "1"));
    if (ex->n_seqs < 1 || ex->n_seqs > MAX_SEQS) ex->n_seqs = 1;
    const char *d = value_of(args, "deliveries", "full,candidates,tokens");
    if (strstr(d, "full")) ex->deliveries |= FLERP_OUT_FULL;
    if (strstr(d, "candidates")) ex->deliveries |= FLERP_OUT_CANDIDATES;
    if (strstr(d, "tokens")) ex->deliveries |= FLERP_OUT_TOKENS;
    d = value_of(args, "draft", "none");
    ex->draft = !strcmp(d, "step") ? 1 : !strcmp(d, "held") ? 2 : 0;
    ex->sampled = !strcmp(value_of(args, "sampled", "off"), "on");
    ex->masks = !strcmp(value_of(args, "masks", "on"), "on");
    ex->top_k_max = atoi(value_of(args, "top_k_max", "0"));
    ex->needs_top_k = !strcmp(value_of(args, "needs_top_k", "off"), "on");
    ex->abi = atoi(value_of(args, "abi", "2"));
    ex->rowbuf = malloc((size_t)ex->row * sizeof(float));
    snprintf(ex->describe, sizeof(ex->describe), "fake executor (a hash, no weights): vocab %lld, deliveries %u, draft form %d, ABI %d",
             (long long)ex->vocab, ex->deliveries, ex->draft, ex->abi);
    *out = ex;
    return FLERP_OK;
}

void flerp_close(flerp_executor *ex) {
    if (ex) free(ex->rowbuf);
    free(ex);
}

flerp_status flerp_get_caps(const flerp_executor *ex, flerp_caps *out) {
    flerp_caps c;
    const size_t v1 = FLERP_HAS(flerp_caps, flags);
    if (out == NULL || out->size < v1) return fail(ex, FLERP_E_ARGUMENT, "flerp_caps is null or short");
    memset(&c, 0, sizeof(c));
    c.size = (uint32_t)(ex->abi < 2 ? v1 : out->size < sizeof(c) ? out->size : sizeof(c));
    c.abi = (uint32_t)ex->abi, c.positions = ex->positions, c.max_forward = 512, c.max_verify_rows = MAX_ROWS;
    c.logits_row = ex->deliveries & FLERP_OUT_FULL ? ex->row : 0;
    c.candidates = ex->deliveries & FLERP_OUT_CANDIDATES ? K : 0;
    c.seqs = ex->n_seqs, c.max_draft = ex->draft ? 8 : 0, c.deliveries = ex->deliveries;
    c.flags = FLERP_CAP_KEEP_PREFIX | (ex->draft == 1 ? FLERP_CAP_DRAFT_STEP : 0u) | (ex->draft == 2 ? FLERP_CAP_DRAFT_HELD : 0u) |
              (ex->masks ? FLERP_CAP_MASKS : 0u) | (ex->sampled ? FLERP_CAP_SAMPLED_TOKENS : 0u);
    if (ex->sampled) {
        c.sampling = FLERP_SAMPLE_TOP_K | FLERP_SAMPLE_SEED | (ex->needs_top_k ? FLERP_SAMPLE_NEEDS_TOP_K : 0u);
        c.top_k_max = ex->top_k_max;
    }
    memcpy(out, &c, c.size);
    return FLERP_OK;
}

const char *flerp_describe(const flerp_executor *ex) { return ex->describe; }
const char *flerp_state_fingerprint(const flerp_executor *ex) { (void)ex; return "fake"; }
const char *flerp_last_error(const flerp_executor *ex) { return ex ? ex->error : g_open_error; }
int flerp_failed(const flerp_executor *ex) { (void)ex; return 0; }

static seq_t *seq_of(flerp_executor *ex, int32_t seq) {
    return seq >= 0 && seq < ex->n_seqs && ex->seqs[seq].open ? &ex->seqs[seq] : NULL;
}

flerp_status flerp_seq_new(flerp_executor *ex, const flerp_seq_args *args, int32_t *seq) {
    for (int i = 0; i < ex->n_seqs; ++i) {
        seq_t *q = &ex->seqs[i];
        if (q->open) continue;
        memset(q, 0, sizeof(*q));
        q->prompt = -1;
        if (args != NULL && args->sampling != NULL && args->sampling->temperature > 0) {
            const flerp_sampling *s = args->sampling;
            if (!ex->sampled) return fail(ex, FLERP_E_UNSUPPORTED, "fake: this executor does not draw sampled tokens (sampled=on)");
            if (s->top_k == 0 && ex->needs_top_k) return fail(ex, FLERP_E_ARGUMENT, "fake: a sampled seq needs top_k");
            if (ex->top_k_max > 0 && s->top_k > ex->top_k_max) return fail(ex, FLERP_E_ARGUMENT, "fake: top_k above top_k_max");
            if (s->top_p < 1 || s->min_p > 0) return fail(ex, FLERP_E_ARGUMENT, "fake: top_p and min_p are not taken");
            q->sampled = 1, q->seed = s->seed, q->top_k = s->top_k;
        }
        q->open = 1;
        *seq = i;
        return FLERP_OK;
    }
    return fail(ex, FLERP_E_FULL, "fake: every seq is open");
}

flerp_status flerp_seq_free(flerp_executor *ex, int32_t seq) {
    seq_t *q = seq_of(ex, seq);
    if (!q) return fail(ex, FLERP_E_ARGUMENT, "fake: the seq is not open");
    q->open = 0;
    return FLERP_OK;
}

flerp_status flerp_seq_reset(flerp_executor *ex, int32_t seq) {
    seq_t *q = seq_of(ex, seq);
    if (!q) return fail(ex, FLERP_E_ARGUMENT, "fake: the seq is not open");
    if (q->pending) return fail(ex, FLERP_E_STATE, "fake: a verify is pending");
    q->pos = 0, q->prompt = -1, q->n_held = 0;
    return FLERP_OK;
}

int64_t flerp_seq_pos(const flerp_executor *ex, int32_t seq) {
    const seq_t *q = seq_of((flerp_executor *)ex, seq);
    return q ? q->pos : FLERP_E_ARGUMENT;
}

flerp_status flerp_forward(flerp_executor *ex, int32_t seq, const flerp_token *ids, size_t n, const flerp_want *want, flerp_rows *out,
                           const flerp_token *lookahead, size_t n_lookahead) {
    seq_t *q = seq_of(ex, seq);
    (void)lookahead, (void)n_lookahead;
    if (!q) return fail(ex, FLERP_E_ARGUMENT, "fake: the seq is not open");
    if (q->pending) return fail(ex, FLERP_E_STATE, "fake: a verify is pending: keep, keep_prefix or drop first");
    if (ids == NULL || n == 0) return fail(ex, FLERP_E_ARGUMENT, "flerp_forward: no ids");
    const int64_t rows = want ? want->rows : 0;
    if (rows < 0 || rows > 1) return fail(ex, FLERP_E_UNSUPPORTED, "fake: a forward returns its last row only");
    if (n > 512) return fail(ex, FLERP_E_FULL, "flerp_forward: more ids than caps.max_forward");
    if (q->pos + (int64_t)n > ex->positions) return fail(ex, FLERP_E_FULL, "fake: the seq is out of positions");
    for (size_t i = 0; i < n; ++i)
        if (ids[i] < 0 || ids[i] >= ex->vocab) return fail(ex, FLERP_E_ARGUMENT, "fake: an id outside the vocabulary");
    q->pos += (int64_t)n, q->last = ids[n - 1], q->n_held = 0;
    ++ex->stats.forwards, ex->stats.forward_tokens += (int64_t)n;
    if (rows == 0) return FLERP_OK;
    if (q->prompt < 0) q->prompt = q->pos; /* the reply starts where the first result is asked */
    return deliver(ex, q, q->last, q->pos, want, want->masks, out, 0);
}

/* The head's draft after a context that ends with `last` at length `len`: the model's token, wrong every 5th. */
static flerp_token draft_of(const flerp_executor *ex, const seq_t *q, flerp_token last, int64_t len) {
    const flerp_token t = model(ex, q, last, len);
    return len % 5 == 0 && t != ex->end ? (flerp_token)(ex->lo + (t - ex->lo + 1) % (ex->hi - ex->lo)) : t;
}

flerp_status flerp_draft_step(flerp_executor *ex, int32_t seq, flerp_token token, int32_t step, flerp_draft *out) {
    seq_t *q = seq_of(ex, seq);
    if (!q || out == NULL) return fail(ex, FLERP_E_ARGUMENT, "fake: the seq is not open");
    if (ex->draft != 1) return fail(ex, FLERP_E_UNSUPPORTED, "fake: no stepped draft head (draft=step)");
    if (q->pending) return fail(ex, FLERP_E_STATE, "fake: a verify is pending");
    const int64_t len = q->pos + 1 + step; /* the context: the seq, the pending token, the chain so far */
    out->token = draft_of(ex, q, token, len);
    out->best = 10.0f, out->second = len % 7 == 0 ? 9.5f : 7.0f, out->probability = NAN;
    ++ex->stats.draft_tokens;
    return FLERP_OK;
}

flerp_status flerp_draft_hold(flerp_executor *ex, int32_t seq, flerp_token pending, int32_t depth) {
    seq_t *q = seq_of(ex, seq);
    if (!q) return fail(ex, FLERP_E_ARGUMENT, "fake: the seq is not open");
    if (ex->draft != 2) return fail(ex, FLERP_E_UNSUPPORTED, "fake: no held drafts (draft=held)");
    if (q->pending) return fail(ex, FLERP_E_STATE, "fake: a verify is pending");
    if (depth < 0 || depth > 8) return fail(ex, FLERP_E_ARGUMENT, "fake: depth outside 0..caps.max_draft");
    flerp_token prev = pending;
    for (int i = 0; i < depth; ++i) prev = q->held[i] = draft_of(ex, q, prev, q->pos + 1 + i);
    q->n_held = depth;
    ex->stats.draft_tokens += depth;
    return FLERP_OK;
}

flerp_status flerp_verify(flerp_executor *ex, flerp_verify_run *runs, size_t n_runs, const flerp_want *want) {
    if (runs == NULL || n_runs != 1 || want == NULL) return fail(ex, FLERP_E_ARGUMENT, "fake: one run a verify");
    flerp_verify_run *r = &runs[0];
    seq_t *q = seq_of(ex, r->seq);
    if (!q) return fail(ex, FLERP_E_ARGUMENT, "fake: the seq is not open");
    if (q->pending) return fail(ex, FLERP_E_STATE, "fake: a verify is pending");
    if (r->ids == NULL || r->n_ids == 0) return fail(ex, FLERP_E_ARGUMENT, "flerp_verify: a run without ids");
    int64_t rows = (int64_t)r->n_ids;
    if (r->held > 0) {
        if (ex->draft != 2 || r->n_ids != 1 || r->held != q->n_held) return fail(ex, FLERP_E_ARGUMENT, "fake: run.held is not the drafts held for the seq");
        if (r->held_out == NULL || r->held_cap < (size_t)r->held) return fail(ex, FLERP_E_ARGUMENT, "flerp_verify_run.held_out is too small");
        rows = 1 + r->held;
    }
    if (rows > MAX_ROWS) return fail(ex, FLERP_E_FULL, "flerp_verify: more rows than caps.max_verify_rows");
    q->v_ids[0] = r->ids[0];
    for (int64_t i = 1; i < rows; ++i) q->v_ids[i] = r->held > 0 ? q->held[i - 1] : r->ids[i];
    if (r->held > 0) memcpy(r->held_out, q->held, (size_t)r->held * sizeof(flerp_token));
    q->pending = 1, q->v_rows = rows;
    ++ex->stats.verifies, ex->stats.verify_rows += rows;
    for (int64_t i = 0; i < rows; ++i) {
        const flerp_status st = deliver(ex, q, q->v_ids[i], q->pos + i + 1, want, r->masks, r->out, i);
        if (st != FLERP_OK) return st;
    }
    return FLERP_OK;
}

static flerp_status resolve(flerp_executor *ex, int32_t seq, int64_t rows) {
    seq_t *q = seq_of(ex, seq);
    if (!q) return fail(ex, FLERP_E_ARGUMENT, "fake: the seq is not open");
    if (!q->pending) return fail(ex, FLERP_E_STATE, "fake: no verify is pending");
    if (rows < 0) rows = q->v_rows;
    if (rows > q->v_rows) return fail(ex, FLERP_E_ARGUMENT, "keep: rows outside the pending verify");
    if (rows > 0) q->pos += rows, q->last = q->v_ids[rows - 1];
    ex->stats.kept_rows += rows, ex->stats.dropped_rows += q->v_rows - rows;
    q->pending = 0, q->n_held = 0;
    return FLERP_OK;
}

flerp_status flerp_keep(flerp_executor *ex, int32_t seq) { return resolve(ex, seq, -1); }
flerp_status flerp_keep_prefix(flerp_executor *ex, int32_t seq, int64_t rows, flerp_token carry) {
    const seq_t *q = seq_of(ex, seq);
    (void)carry;
    if (q && q->pending && (rows < 1 || rows >= q->v_rows)) return fail(ex, FLERP_E_ARGUMENT, "flerp_keep_prefix: rows must be 1 .. the run's rows - 1");
    return resolve(ex, seq, rows);
}
flerp_status flerp_drop(flerp_executor *ex, int32_t seq) { return resolve(ex, seq, 0); }

#define NO_SNAPSHOTS return fail(ex, FLERP_E_UNSUPPORTED, "fake: no snapshots")
flerp_status flerp_snapshot_save(flerp_executor *ex, int32_t seq, flerp_snapshot *out) { (void)seq, (void)out; NO_SNAPSHOTS; }
flerp_status flerp_snapshot_restore(flerp_executor *ex, int32_t seq, flerp_snapshot snap) { (void)seq, (void)snap; NO_SNAPSHOTS; }
flerp_status flerp_snapshot_drop(flerp_executor *ex, int32_t seq, flerp_snapshot snap) { (void)seq, (void)snap; NO_SNAPSHOTS; }
int64_t flerp_snapshot_pos(const flerp_executor *ex, int32_t seq, flerp_snapshot snap) { (void)ex, (void)seq, (void)snap; return FLERP_E_UNSUPPORTED; }
int64_t flerp_snapshot_bytes(const flerp_executor *ex, int32_t seq, flerp_snapshot snap, int64_t from) { (void)ex, (void)seq, (void)snap, (void)from; return FLERP_E_UNSUPPORTED; }
flerp_status flerp_snapshot_export(flerp_executor *ex, int32_t seq, flerp_snapshot snap, int64_t from, void *bytes, size_t cap, size_t *written) {
    (void)seq, (void)snap, (void)from, (void)bytes, (void)cap, (void)written; NO_SNAPSHOTS;
}
flerp_status flerp_state_import(flerp_executor *ex, int32_t seq, const void *bytes, size_t n_bytes, int64_t n, int64_t from) {
    (void)seq, (void)bytes, (void)n_bytes, (void)n, (void)from; NO_SNAPSHOTS;
}
flerp_status flerp_snapshot_write(flerp_executor *ex, int32_t seq, flerp_snapshot snap, const char *dir, uint64_t key) { (void)seq, (void)snap, (void)dir, (void)key; NO_SNAPSHOTS; }
flerp_status flerp_snapshot_read(flerp_executor *ex, int32_t seq, const char *dir, uint64_t key, int64_t at, flerp_snapshot *out) {
    (void)seq, (void)dir, (void)key, (void)at, (void)out; NO_SNAPSHOTS;
}

flerp_status flerp_get_stats(const flerp_executor *ex, flerp_stats *out) {
    if (out == NULL || out->size < sizeof(flerp_stats)) return fail(ex, FLERP_E_ARGUMENT, "flerp_stats is null or short");
    const uint32_t size = out->size;
    *out = ex->stats;
    out->size = size, out->device_bytes = 0;
    return FLERP_OK;
}

size_t flerp_stats_json(const flerp_executor *ex, char *out, size_t cap) {
    (void)ex;
    if (out != NULL && cap > 2) memcpy(out, "{}", 3);
    return 2;
}
