/* executor.flerp.fake_abi with a scripted reply, for tests of a front end's reply handling (tests/test_flerp_reply.py).
 *
 * The stand-in's own model is a hash, so its replies are no text a reply parser would split. This wrapper links the
 * stand-in unchanged (its source is included, three entry points renamed) and puts a script on top: open key
 * script=FILE names a file of token ids, read at each flerp_seq_new. After the prompt (the first forward that asks for
 * rows), every row's top token is the script's next id - the script advances over each id that enters the seq and
 * equals its next id, so ids a caller feeds on the model's behalf (a thinking stop, a think close) do not move it -
 * and the stand-in's end id once the script is spent.
 *
 * Masks (flerp_masks with rows): the stand-in takes a logit bias only, so the wrapper hands it the bias and applies
 * each row's mask to the delivered row itself, as an executor that offers masks does: a refused id leaves a full
 * row (-inf), leaves the candidates (the rest move up), and a drawn token falls to the lowest allowed id. An empty script file leaves the stand-in's rows.
 * Open key masks_rows=off reports FLERP_CAP_MASKS without row masks (a refusal path); masks=off (the stand-in's own
 * key) takes FLERP_CAP_MASKS away. Drafts are the stand-in's: use draft=none with a script.
 * Open key delay_ms=N makes every forward and verify take N ms, so a test can hold the front end's generation lane
 * (its queue, /health under load, a client that leaves) without a model. */
#define flerp_open fake_inner_open
#define flerp_forward fake_inner_forward
#define flerp_verify fake_inner_verify
#define flerp_seq_new fake_inner_seq_new
#include "../../catalog/flerp/executor/fake_abi/src/flerp/executor/fake_abi.c"
#undef flerp_open
#undef flerp_forward
#undef flerp_verify
#undef flerp_seq_new

#include <stdint.h>
#include <time.h>

flerp_status flerp_open(const flerp_open_args *args, flerp_executor **out);
flerp_status flerp_forward(flerp_executor *ex, int32_t seq, const flerp_token *ids, size_t n, const flerp_want *want, flerp_rows *out,
                           const flerp_token *lookahead, size_t n_lookahead);
flerp_status flerp_verify(flerp_executor *ex, flerp_verify_run *runs, size_t n_runs, const flerp_want *want);
flerp_status flerp_seq_new(flerp_executor *ex, const flerp_seq_args *args, int32_t *seq);

#define SCRIPT_MAX 8192
static char g_script_path[1024];
static int g_rows_masks = 1;
static long g_delay_ms = 0;

static void delay(void) {
    if (g_delay_ms <= 0) return;
    struct timespec ts;
    ts.tv_sec = g_delay_ms / 1000, ts.tv_nsec = (g_delay_ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}
static flerp_token g_script[SCRIPT_MAX];
static size_t g_n_script, g_cursor[MAX_SEQS];
static int g_armed[MAX_SEQS];

flerp_status flerp_open(const flerp_open_args *args, flerp_executor **out) {
    flerp_kv kv[64];
    size_t n = 0;
    g_script_path[0] = 0, g_rows_masks = 1, g_delay_ms = 0;
    for (size_t i = 0; args && i < args->n_config && n < 64; ++i) {
        if (!strcmp(args->config[i].key, "script")) snprintf(g_script_path, sizeof(g_script_path), "%s", args->config[i].value);
        else if (!strcmp(args->config[i].key, "masks_rows")) g_rows_masks = strcmp(args->config[i].value, "off") != 0;
        else if (!strcmp(args->config[i].key, "delay_ms")) g_delay_ms = atol(args->config[i].value);
        else kv[n++] = args->config[i];
    }
    flerp_open_args a = *args;
    a.config = kv, a.n_config = n;
    return fake_inner_open(&a, out);
}

flerp_status flerp_seq_new(flerp_executor *ex, const flerp_seq_args *args, int32_t *seq) {
    const flerp_status st = fake_inner_seq_new(ex, args, seq);
    if (st != FLERP_OK) return st;
    g_n_script = 0;
    if (g_script_path[0]) {
        FILE *f = fopen(g_script_path, "r");
        long v;
        if (f) {
            while (g_n_script < SCRIPT_MAX && fscanf(f, "%ld", &v) == 1) g_script[g_n_script++] = (flerp_token)v;
            fclose(f);
        }
    }
    g_cursor[*seq] = 0, g_armed[*seq] = 0;
    return FLERP_OK;
}

static void advance(int32_t seq, const flerp_token *ids, size_t n) {
    if (!g_armed[seq]) return;
    for (size_t i = 0; i < n; ++i)
        if (g_cursor[seq] < g_n_script && ids[i] == g_script[g_cursor[seq]]) ++g_cursor[seq];
}

/* -1: no script (an empty file): the stand-in's own rows, the masks still applied */
static flerp_token next_of(const flerp_executor *ex, int32_t seq, size_t at) {
    (void)seq;
    if (g_n_script == 0) return -1;
    return at < g_n_script ? g_script[at] : ex->end;
}

static int allowed(const flerp_masks *m, int64_t r, flerp_token id) {
    if (m == NULL || m->rows == 0 || m->bits == NULL) return 1;
    if (id < 0 || (int64_t)id / 32 >= m->words) return 0;
    return (m->bits[r * m->words + id / 32] >> (id % 32)) & 1u;
}

/* Row r of `out`, whose context ends with `last` at length `len`: the script's token t on top (t < 0: none), then
 * the row's mask applied where the logits are - candidates and draws come from the masked row, as on an executor
 * that applies masks before its top-k. */
static void script_row(flerp_executor *ex, seq_t *q, flerp_token last, int64_t len, uint32_t delivery, const flerp_masks *bias,
                       flerp_rows *out, int64_t r, flerp_token t, const flerp_masks *m) {
    float *x = delivery == FLERP_OUT_FULL ? out->full + r * ex->row : ex->rowbuf;
    if (delivery != FLERP_OUT_FULL) full_row(ex, q, last, len, bias, x);
    if (t >= 0) x[t] = 50.0f;
    for (int64_t i = 0; i < ex->vocab; ++i)
        if (!allowed(m, r, (flerp_token)i)) x[i] = -INFINITY;
    if (delivery == FLERP_OUT_FULL) return;
    flerp_token id[K];
    float v[K];
    top_k(x, ex->vocab, id, v);
    for (int c = 0; c < K; ++c)
        if (id[c] < 0 || !(v[c] > -INFINITY)) id[c] = INT32_MAX, v[c] = -INFINITY; /* fewer allowed ids than candidates: padding */
    if (delivery == FLERP_OUT_CANDIDATES) memcpy(out->cand_id + r * K, id, sizeof(id)), memcpy(out->cand_logit + r * K, v, sizeof(v));
    else out->tokens[r] = id[0];
}

/* the stand-in takes a bias only: row masks stay with the wrapper */
static const flerp_masks *bias_of(const flerp_masks *m, flerp_masks *keep) {
    if (m == NULL || m->rows == 0) return m;
    if (m->n_bias == 0) return NULL;
    *keep = *m;
    keep->rows = 0, keep->words = 0, keep->bits = NULL;
    return keep;
}

static flerp_status masks_ok(flerp_executor *ex, const flerp_masks *m) {
    if (m != NULL && m->rows > 0 && !g_rows_masks) return fail(ex, FLERP_E_UNSUPPORTED, "script fake: row masks are off (masks_rows=off)");
    return FLERP_OK;
}

flerp_status flerp_forward(flerp_executor *ex, int32_t seq, const flerp_token *ids, size_t n, const flerp_want *want, flerp_rows *out,
                           const flerp_token *lookahead, size_t n_lookahead) {
    const flerp_masks *m = want ? want->masks : NULL;
    flerp_status st = masks_ok(ex, m);
    if (st != FLERP_OK) return st;
    delay();
    flerp_want w;
    flerp_masks keep;
    if (want) w = *want, w.masks = bias_of(m, &keep);
    st = fake_inner_forward(ex, seq, ids, n, want ? &w : NULL, out, lookahead, n_lookahead);
    if (st != FLERP_OK || seq < 0 || seq >= MAX_SEQS) return st;
    advance(seq, ids, n);
    if (want && want->rows > 0 && out) {
        g_armed[seq] = 1;
        seq_t *q = seq_of(ex, seq);
        const flerp_token t = next_of(ex, seq, g_cursor[seq]);
        if (t >= 0 || (m && m->rows > 0)) script_row(ex, q, q->ids[q->pos - 1], q->pos, want->delivery, w.masks, out, out->rows - 1, t, m);
    }
    return FLERP_OK;
}

flerp_status flerp_verify(flerp_executor *ex, flerp_verify_run *runs, size_t n_runs, const flerp_want *want) {
    if (runs == NULL || n_runs != 1) return fake_inner_verify(ex, runs, n_runs, want);
    const flerp_masks *m = runs[0].masks;
    flerp_status st = masks_ok(ex, m);
    if (st != FLERP_OK) return st;
    delay();
    flerp_masks keep;
    flerp_verify_run run = runs[0];
    run.masks = bias_of(m, &keep);
    st = fake_inner_verify(ex, &run, 1, want);
    runs[0].out = run.out;
    if (st != FLERP_OK) return st;
    const int32_t seq = run.seq;
    seq_t *q = seq_of(ex, seq);
    size_t at = g_cursor[seq];
    for (int64_t r = 0; r < run.out->rows; ++r) { /* row r follows v_ids[0..r]: the script after them */
        if (at < g_n_script && q->v_ids[r] == g_script[at]) ++at;
        const flerp_token t = next_of(ex, seq, at);
        if (t >= 0 || (m && m->rows > 0)) script_row(ex, q, q->v_ids[r], q->pos + r + 1, want->delivery, run.masks, run.out, r, t, m);
    }
    return FLERP_OK;
}
