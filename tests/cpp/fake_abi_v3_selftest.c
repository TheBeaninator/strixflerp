/* executor.flerp.fake_abi through every version 3 call of include/flerp/executor.h (tests/test_flerp_serve.py).
 * Built three ways:
 *   (default)        against the version 3 header and the version 3 stand-in: every call, then the stand-in opened
 *                    as a version 2 executor (abi=2), then version 2 sized structs from a version 3 build
 *   -DV2_CALLER      against the frozen version 2 header (tests/cpp/abi_v2) and the version 3 stand-in: a binary
 *                    built before version 3 existed drives a version 3 executor
 *   -DV3_ON_V2       against the version 3 header and the frozen version 2 stand-in: a version 3 caller finds a
 *                    version 2 executor and uses only what its caps say
 * Prints each failed check and exits with the number of failures. */
#include "flerp/executor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
#define CHECK(cond)                                                                \
    do {                                                                           \
        if (!(cond)) ++failures, printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    } while (0)

static flerp_executor *open_with(const char *const *kv, size_t n) {
    flerp_kv c[24];
    for (size_t i = 0; i < n; ++i) c[i].key = kv[2 * i], c[i].value = kv[2 * i + 1];
    flerp_open_args oa;
    memset(&oa, 0, sizeof(oa));
    oa.size = sizeof(oa), oa.recipe = "t", oa.config = c, oa.n_config = n;
    flerp_executor *ex = NULL;
    if (flerp_open(&oa, &ex) != FLERP_OK) {
        printf("FAIL open: %s\n", flerp_last_error(NULL));
        exit(100);
    }
    return ex;
}

/* Greedy decode of `n_new` tokens after `prompt` on a fresh seq: the plain way, one token a forward. */
static int decode(flerp_executor *ex, int32_t seq, const flerp_token *prompt, size_t n, int n_new, flerp_token *out) {
    flerp_want w;
    memset(&w, 0, sizeof(w));
    w.size = sizeof(w), w.delivery = FLERP_OUT_TOKENS, w.rows = n_new > 0; /* no result asked: the prompt goes on */
    flerp_token t;
    flerp_rows r;
    memset(&r, 0, sizeof(r));
    r.size = sizeof(r), r.tokens = &t, r.tokens_cap = 1;
    if (n > 0 && flerp_forward(ex, seq, prompt, n, &w, &r, NULL, 0) != FLERP_OK) return -1;
    for (int i = 0; i < n_new; ++i) {
        out[i] = t;
        if (i + 1 < n_new && flerp_forward(ex, seq, &t, 1, &w, &r, NULL, 0) != FLERP_OK) return -1;
    }
    return 0;
}

#if !defined(V2_CALLER) && !defined(V3_ON_V2)
typedef struct {
    unsigned char *buf;
    size_t n, cap, at;
    int calls;
} mem_t;

static int sink(void *ctx, const void *b, size_t n) {
    mem_t *m = ctx;
    if (m->n + n > m->cap) m->cap = (m->n + n) * 2, m->buf = realloc(m->buf, m->cap);
    memcpy(m->buf + m->n, b, n), m->n += n, ++m->calls;
    return 0;
}
static int stop_sink(void *ctx, const void *b, size_t n) { (void)ctx, (void)b, (void)n; return 1; }
static int64_t source(void *ctx, void *b, size_t cap) { /* 5 bytes a call: a stream, not one buffer */
    mem_t *m = ctx;
    size_t k = m->n - m->at < cap ? m->n - m->at : cap;
    if (k > 5) k = 5;
    memcpy(b, m->buf + m->at, k), m->at += k;
    return (int64_t)k;
}

static const char *const ALL[] = {"vocab", "5000", "lo", "100", "hi", "4000", "reply", "40", "features", "all", "seqs", "3"};

static void every_call(void) {
    flerp_executor *ex = open_with(ALL, 6);
    flerp_caps c;
    memset(&c, 0, sizeof(c));
    c.size = sizeof(c);
    CHECK(flerp_get_caps(ex, &c) == FLERP_OK);
    CHECK(c.abi == 3 && c.size == sizeof(flerp_caps));
    const uint32_t v3 = FLERP_CAP_STATE_PREFIX | FLERP_CAP_SNAPSHOT_MARK | FLERP_CAP_SNAPSHOT_SHARED | FLERP_CAP_STATE_PARK | FLERP_CAP_STEER |
                        FLERP_CAP_STEER_SPAN | FLERP_CAP_IMAGE_ROWS | FLERP_CAP_IMAGE_PREPARE | FLERP_CAP_CONSTRAINT | FLERP_CAP_SEQ_POLICY |
                        FLERP_CAP_SEQ_BATCHABLE | FLERP_CAP_PREFETCH | FLERP_CAP_RENEW;
    CHECK((c.flags & v3) == v3);
    CHECK(c.steer_dim == 4 && c.steer_layers == 8 && c.image_token == 4998 && c.image_hidden == 4 && c.images > 0 && c.constraints == 31);

    flerp_seq_args sa;
    memset(&sa, 0, sizeof(sa));
    sa.size = sizeof(sa);
    int32_t a, b;
    CHECK(flerp_seq_new(ex, &sa, &a) == FLERP_OK);
    CHECK(flerp_seq_new(ex, &sa, &b) == FLERP_OK);
    flerp_token prompt[64], plain[20], got[20];
    for (int i = 0; i < 64; ++i) prompt[i] = 200 + i;
    CHECK(decode(ex, a, prompt, 64, 20, plain) == 0);

    /* state with its prefix ids: export a snapshot, import it into another seq, the continuation is the same */
    CHECK(flerp_seq_reset(ex, a) == FLERP_OK);
    flerp_snapshot mark, early;
    CHECK(flerp_snapshot_mark(ex, a, 32, &mark) == FLERP_OK);
    CHECK(flerp_snapshot_mark(ex, a, 80, &early) == FLERP_OK);          /* past the prompt: dropped by the result */
    CHECK(flerp_snapshot_pos(ex, a, mark) == FLERP_E_STATE);            /* not reached yet */
    CHECK(decode(ex, a, prompt, 64, 1, got) == 0);
    CHECK(flerp_snapshot_pos(ex, a, mark) == 32);
    CHECK(flerp_snapshot_pos(ex, a, early) == FLERP_E_ARGUMENT);
    mem_t m = {0};
    flerp_state_desc d;
    memset(&d, 0, sizeof(d));
    d.size = sizeof(d);
    CHECK(flerp_state_export2(ex, a, mark, 0, &d, sink, &m) == FLERP_OK);
    CHECK(d.tokens == 32 && d.from == 0 && d.n_prefix == 32 && d.bytes == (int64_t)m.n && !strcmp(d.fingerprint, flerp_state_fingerprint(ex)));
    CHECK(d.prefix_ids != NULL && memcmp(d.prefix_ids, prompt, 32 * sizeof(flerp_token)) == 0);
    CHECK(flerp_state_export2(ex, a, mark, 0, &d, stop_sink, NULL) == FLERP_E_ARGUMENT);
    CHECK(!flerp_failed(ex));
    flerp_state_desc in = d;
    in.prefix_ids = prompt, in.bytes = -1;
    flerp_state_desc no_prefix = in;
    no_prefix.prefix_ids = NULL, no_prefix.n_prefix = 0;
    CHECK(flerp_state_import2(ex, b, &no_prefix, source, &m) == FLERP_E_STATE);   /* StrixFold's refusal, now with a reason */
    flerp_token wrong[32];
    memcpy(wrong, prompt, sizeof(wrong)), wrong[7] = 1;
    flerp_state_desc bad = in;
    bad.prefix_ids = wrong, m.at = 0;
    CHECK(flerp_state_import2(ex, b, &bad, source, &m) == FLERP_E_STATE);
    bad = in, bad.fingerprint = "other", m.at = 0;
    CHECK(flerp_state_import2(ex, b, &bad, source, &m) == FLERP_E_STATE);
    m.at = 0;
    CHECK(flerp_state_import2(ex, b, &in, source, &m) == FLERP_OK);
    CHECK(flerp_seq_pos(ex, b) == 32);
    CHECK(decode(ex, b, prompt + 32, 32, 20, got) == 0);
    CHECK(memcmp(got, plain, sizeof(got)) == 0);
    /* a delta [32, 64) onto a seq that holds the first 32 */
    flerp_snapshot full;
    CHECK(flerp_seq_reset(ex, a) == FLERP_OK);
    CHECK(decode(ex, a, prompt, 64, 0, got) == 0);
    CHECK(flerp_snapshot_save(ex, a, &full) == FLERP_OK);
    mem_t m2 = {0};
    flerp_state_desc dd;
    memset(&dd, 0, sizeof(dd));
    dd.size = sizeof(dd);
    CHECK(flerp_state_export2(ex, a, full, 32, &dd, sink, &m2) == FLERP_OK);
    CHECK(dd.tokens == 64 && dd.from == 32 && m2.n == 32 + 4 * 32);
    CHECK(flerp_seq_reset(ex, b) == FLERP_OK);
    flerp_state_desc din = dd;
    din.prefix_ids = prompt;
    m2.at = 0;
    CHECK(flerp_state_import2(ex, b, &din, source, &m2) == FLERP_E_STATE);    /* the seq does not hold [0, 32) */
    m.at = 0;
    CHECK(flerp_state_import2(ex, b, &in, source, &m) == FLERP_OK);
    m2.at = 0;
    CHECK(flerp_state_import2(ex, b, &din, source, &m2) == FLERP_OK);
    CHECK(flerp_seq_pos(ex, b) == 64);
    flerp_token pend = plain[0];
    (void)pend;

    /* shared: survives its seq, restores into another */
    flerp_snapshot shared;
    CHECK(flerp_snapshot_share(ex, a, full, &shared) == FLERP_OK);
    CHECK(flerp_seq_free(ex, a) == FLERP_OK);
    CHECK(flerp_snapshot_pos(ex, a, full) == FLERP_E_ARGUMENT);          /* freed with its seq */
    CHECK(flerp_seq_reset(ex, b) == FLERP_OK);
    CHECK(flerp_snapshot_restore(ex, b, shared) == FLERP_OK);
    CHECK(flerp_seq_pos(ex, b) == 64);
    flerp_want w;
    memset(&w, 0, sizeof(w));
    w.size = sizeof(w), w.delivery = FLERP_OUT_TOKENS, w.rows = 1;
    flerp_token t;
    flerp_rows r;
    memset(&r, 0, sizeof(r));
    r.size = sizeof(r), r.tokens = &t, r.tokens_cap = 1;
    CHECK(flerp_forward(ex, b, &prompt[63], 1, &w, &r, NULL, 0) == FLERP_OK);   /* any id: the row exists */
    CHECK(flerp_snapshot_drop(ex, b, shared) == FLERP_E_ARGUMENT);
    CHECK(flerp_snapshot_drop(ex, -1, shared) == FLERP_OK);

    /* park and resume */
    CHECK(flerp_seq_new(ex, &sa, &a) == FLERP_OK);
    CHECK(decode(ex, a, prompt, 40, 0, got) == 0);
    flerp_snapshot parked;
    CHECK(flerp_seq_park(ex, a, &parked) == FLERP_OK);
    CHECK(flerp_seq_pos(ex, a) == 0);
    CHECK(flerp_seq_resume(ex, a, parked) == FLERP_OK);
    CHECK(flerp_seq_pos(ex, a) == 40);
    CHECK(flerp_seq_resume(ex, a, parked) == FLERP_E_ARGUMENT);          /* consumed */
    CHECK(decode(ex, a, prompt + 40, 24, 20, got) == 0);
    CHECK(memcmp(got, plain, sizeof(got)) == 0);

    /* steering: a vector changes the reply, scale 0 and dropping the item give it back, a span covers [from, to) */
    const float v[4] = {0.5f, -1.0f, 2.0f, 0.25f};
    uint32_t h;
    CHECK(flerp_steer_load(ex, "calm", 3, v, 3, &h) == FLERP_E_ARGUMENT);
    CHECK(flerp_steer_load(ex, "calm", 3, v, 4, &h) == FLERP_OK && h != 0);
    flerp_steer_item it;
    memset(&it, 0, sizeof(it));
    it.size = sizeof(it), it.vector = h, it.scale = 1.5f, it.from = 0, it.to = -1;
    CHECK(flerp_seq_reset(ex, a) == FLERP_OK);
    CHECK(flerp_seq_steer(ex, a, &it, 1) == FLERP_OK);
    CHECK(decode(ex, a, prompt, 64, 20, got) == 0);
    CHECK(memcmp(got, plain, sizeof(got)) != 0);
    CHECK(flerp_steer_drop(ex, h) == FLERP_E_STATE);
    it.from = 70, it.to = 75;                                             /* a span after the 6th reply token */
    CHECK(flerp_seq_reset(ex, a) == FLERP_OK);
    CHECK(flerp_seq_steer(ex, a, &it, 1) == FLERP_OK);
    CHECK(decode(ex, a, prompt, 64, 20, got) == 0);
    CHECK(memcmp(got, plain, 6 * sizeof(flerp_token)) == 0 && memcmp(got, plain, sizeof(got)) != 0);
    it.from = 5, it.to = 5;
    CHECK(flerp_seq_steer(ex, a, &it, 1) == FLERP_E_ARGUMENT);
    CHECK(flerp_seq_steer(ex, a, NULL, 0) == FLERP_OK);
    CHECK(flerp_steer_drop(ex, h) == FLERP_OK);
    CHECK(flerp_seq_reset(ex, a) == FLERP_OK);
    CHECK(decode(ex, a, prompt, 64, 20, got) == 0);
    CHECK(memcmp(got, plain, sizeof(got)) == 0);

    /* images: rows and the executor's own tower; a different picture, a different reply; such a seq verifies alone */
    CHECK(flerp_seq_batchable(ex, a) == 1);
    flerp_token ip[64];
    memcpy(ip, prompt, sizeof(ip));
    for (int i = 56; i < 64; ++i) ip[i] = c.image_token;          /* at the prompt's end: the next token reads it */
    float rows[8 * 4], rows2[8 * 4];
    for (int i = 0; i < 32; ++i) rows[i] = (float)i, rows2[i] = (float)i + (i == 31);
    flerp_image_span sp;
    memset(&sp, 0, sizeof(sp));
    sp.size = sizeof(sp), sp.at = 56, sp.tokens = 8, sp.grid_t = 1, sp.grid_h = 2, sp.grid_w = 4, sp.rows = rows;
    flerp_token g1[20], g2[20];
    CHECK(flerp_seq_reset(ex, a) == FLERP_OK);
    CHECK(flerp_forward_spans(ex, a, ip, 64, &sp, 1, &w, &r, NULL, 0) == FLERP_OK);
    CHECK(flerp_seq_batchable(ex, a) == 0);
    g1[0] = t;
    CHECK(decode(ex, a, NULL, 0, 0, g1) == 0);
    sp.rows = rows2;
    CHECK(flerp_seq_reset(ex, a) == FLERP_OK);
    CHECK(flerp_seq_batchable(ex, a) == 1);
    CHECK(flerp_forward_spans(ex, a, ip, 64, &sp, 1, &w, &r, NULL, 0) == FLERP_OK);
    g2[0] = t;
    CHECK(g1[0] != g2[0]);
    sp.at = 55;                                                          /* position 55 is not the image id */
    CHECK(flerp_seq_reset(ex, a) == FLERP_OK);
    CHECK(flerp_forward_spans(ex, a, ip, 64, &sp, 1, &w, &r, NULL, 0) == FLERP_E_ARGUMENT);
    const char picture[] = "\x89PNG fake picture bytes";
    flerp_image_span pp;
    memset(&pp, 0, sizeof(pp));
    pp.size = sizeof(pp);
    CHECK(flerp_image_prepare(ex, picture, sizeof(picture), &pp) == FLERP_OK);
    CHECK(pp.tokens > 0 && pp.tokens == (int64_t)pp.grid_t * pp.grid_h * pp.grid_w);
    flerp_token pq[64];
    memcpy(pq, prompt, sizeof(pq));
    for (int64_t i = 0; i < pp.tokens; ++i) pq[20 + i] = c.image_token;
    pp.at = 20, pp.payload = picture, pp.payload_bytes = sizeof(picture);
    CHECK(flerp_forward_spans(ex, a, pq, 64, &pp, 1, &w, &r, NULL, 0) == FLERP_OK);
    pp.tokens += 1;
    CHECK(flerp_seq_reset(ex, a) == FLERP_OK);
    CHECK(flerp_forward_spans(ex, a, pq, 64, &pp, 1, &w, &r, NULL, 0) == FLERP_E_ARGUMENT);

    /* slots: prefetch is a hint, renew keeps the tokens */
    CHECK(flerp_seq_reset(ex, a) == FLERP_OK);
    CHECK(flerp_prefetch(ex, a, prompt, 8) == FLERP_OK);
    CHECK(flerp_seq_can_renew(ex, a) == 1);
    CHECK(decode(ex, a, prompt, 64, 0, got) == 0);
    CHECK(flerp_seq_renew(ex, a) == FLERP_OK);
    CHECK(decode(ex, a, NULL, 0, 0, got) == 0);
    {
        flerp_token g[20];
        CHECK(flerp_seq_reset(ex, a) == FLERP_OK);
        CHECK(decode(ex, a, prompt, 32, 0, g) == 0);
        CHECK(flerp_seq_renew(ex, a) == FLERP_OK);
        CHECK(decode(ex, a, prompt + 32, 32, 20, g) == 0);
        CHECK(memcmp(g, plain, sizeof(g)) == 0);
    }
    char js[128];
    CHECK(flerp_stats_json(ex, js, sizeof(js)) > 2 && strstr(js, "\"renews\": 2") != NULL);

    /* a constraint for device-side draws, and the stop policy */
    flerp_constraint con;
    memset(&con, 0, sizeof(con));
    con.size = sizeof(con), con.kind = FLERP_CONSTRAINT_JSON_SCHEMA, con.text = "{\"type\":\"object\"}", con.n_text = strlen(con.text);
    flerp_seq_args sc = sa;
    sc.constraint = &con;
    int32_t s3;
    CHECK(flerp_seq_free(ex, a) == FLERP_OK);
    CHECK(flerp_seq_new(ex, &sc, &s3) == FLERP_OK);
    CHECK(decode(ex, s3, prompt, 64, 20, got) == 0);
    uint32_t par = (uint32_t)got[0] & 1u;
    int same = 1;
    for (int i = 0; i < 20; ++i) same &= ((uint32_t)got[i] & 1u) == par;
    CHECK(same);
    w.delivery = FLERP_OUT_CANDIDATES;
    CHECK(flerp_seq_reset(ex, s3) == FLERP_OK);
    {
        flerp_token ci[20];
        float cl[20];
        flerp_rows rc = r;
        rc.cand_id = ci, rc.cand_logit = cl, rc.cand_cap = 20;
        CHECK(flerp_forward(ex, s3, prompt, 64, &w, &rc, NULL, 0) == FLERP_E_UNSUPPORTED);   /* the caller masks rows itself */
    }
    w.delivery = FLERP_OUT_TOKENS;
    con.kind = 3;
    CHECK(flerp_seq_free(ex, s3) == FLERP_OK);
    CHECK(flerp_seq_new(ex, &sc, &s3) == FLERP_E_ARGUMENT);
    const flerp_token close[3] = {11, 12, 13}, eos[1] = {plain[9]};
    flerp_seq_args sp2 = sa;
    sp2.think_budget = 4, sp2.think_close = close, sp2.n_think_close = 3, sp2.eos = eos, sp2.n_eos = 1, sp2.loop_guard = 64;
    CHECK(flerp_seq_new(ex, &sp2, &s3) == FLERP_OK);
    CHECK(decode(ex, s3, prompt, 64, 7, got) == 0);
    CHECK(memcmp(got, plain, 4 * sizeof(flerp_token)) == 0 && got[4] == 11 && got[5] == 12 && got[6] == 13);
    CHECK(flerp_seq_reset(ex, s3) == FLERP_OK);
    CHECK(flerp_forward(ex, s3, prompt, 64, &w, &r, NULL, 0) == FLERP_OK);
    CHECK(flerp_forward(ex, s3, eos, 1, &w, &r, NULL, 0) == FLERP_E_STATE);       /* the reply ended */
    CHECK(flerp_forward(ex, s3, prompt, 1, &w, &r, NULL, 0) == FLERP_E_STATE);
    CHECK(flerp_seq_reset(ex, s3) == FLERP_OK);
    CHECK(flerp_forward(ex, s3, prompt, 64, &w, &r, NULL, 0) == FLERP_OK);
    flerp_close(ex);
    free(m.buf), free(m2.buf);
}

/* The same stand-in reporting version 2: caps stop at top_k_max, every version 3 call is refused. */
static void as_version_2(void) {
    static const char *const kv[] = {"vocab", "5000", "abi", "2"};
    flerp_executor *ex = open_with(kv, 2);
    flerp_caps c;
    memset(&c, 0x5A, sizeof(c));
    c.size = sizeof(c);
    CHECK(flerp_get_caps(ex, &c) == FLERP_OK);
    CHECK(c.abi == 2 && c.size == FLERP_CAPS_V2_SIZE);
    unsigned char *raw = (unsigned char *)&c;
    int untouched = 1;
    for (size_t i = FLERP_CAPS_V2_SIZE; i < sizeof(c); ++i) untouched &= raw[i] == 0x5A;
    CHECK(untouched);
    flerp_seq_args sa;
    memset(&sa, 0, sizeof(sa));
    sa.size = sizeof(sa);
    flerp_constraint con = {sizeof(con), FLERP_CONSTRAINT_REGEX, "a+", 2};
    sa.constraint = &con;                                     /* a version 2 executor reads only its own struct */
    int32_t s;
    CHECK(flerp_seq_new(ex, &sa, &s) == FLERP_OK);
    flerp_snapshot h = 0;
    uint32_t vh = 0;
    flerp_state_desc d;
    memset(&d, 0, sizeof(d));
    d.size = sizeof(d);
    flerp_image_span sp;
    memset(&sp, 0, sizeof(sp));
    sp.size = sizeof(sp);
    const float v[4] = {0};
    CHECK(flerp_state_export2(ex, s, 1, 0, &d, stop_sink, NULL) == FLERP_E_UNSUPPORTED);
    CHECK(flerp_state_import2(ex, s, &d, source, NULL) == FLERP_E_UNSUPPORTED);
    CHECK(flerp_snapshot_mark(ex, s, 8, &h) == FLERP_E_UNSUPPORTED);
    CHECK(flerp_snapshot_share(ex, s, 1, &h) == FLERP_E_UNSUPPORTED);
    CHECK(flerp_seq_park(ex, s, &h) == FLERP_E_UNSUPPORTED);
    CHECK(flerp_seq_resume(ex, s, 1) == FLERP_E_UNSUPPORTED);
    CHECK(flerp_steer_load(ex, "x", 0, v, 4, &vh) == FLERP_E_UNSUPPORTED);
    CHECK(flerp_steer_drop(ex, 1) == FLERP_E_UNSUPPORTED);
    CHECK(flerp_seq_steer(ex, s, NULL, 0) == FLERP_E_UNSUPPORTED);
    CHECK(flerp_image_prepare(ex, "x", 1, &sp) == FLERP_E_UNSUPPORTED);
    CHECK(flerp_forward_spans(ex, s, (const flerp_token[]){1}, 1, &sp, 1, NULL, NULL, NULL, 0) == FLERP_E_UNSUPPORTED);
    CHECK(flerp_seq_batchable(ex, s) == FLERP_E_UNSUPPORTED);
    CHECK(flerp_prefetch(ex, s, NULL, 0) == FLERP_E_UNSUPPORTED);
    CHECK(flerp_seq_can_renew(ex, s) == FLERP_E_UNSUPPORTED);
    CHECK(flerp_seq_renew(ex, s) == FLERP_E_UNSUPPORTED);
    flerp_close(ex);
    static const char *const bad[] = {"vocab", "5000", "abi", "2", "features", "steer"};
    flerp_kv kb[3] = {{bad[0], bad[1]}, {bad[2], bad[3]}, {bad[4], bad[5]}};
    flerp_open_args oa = {sizeof(oa), "t", kb, 3};
    CHECK(flerp_open(&oa, &ex) == FLERP_E_CONFIG);            /* version 3 features on a version 2 executor */
}

/* Version 2 sized structs from a version 3 build: the version 3 fields past `size` are never read. */
static void v2_sized_structs(void) {
    static const char *const kv[] = {"vocab", "5000", "features", "all"};
    flerp_executor *ex = open_with(kv, 2);
    flerp_seq_args sa;
    memset(&sa, 0xFF, sizeof(sa));                            /* garbage past the version 2 struct */
    sa.size = (uint32_t)FLERP_SEQ_ARGS_V2_SIZE, sa.sampling = NULL, sa.max_new = 0, sa.drafts = 0;
    int32_t s;
    CHECK(flerp_seq_new(ex, &sa, &s) == FLERP_OK);
    flerp_token p[8] = {1, 2, 3, 4, 5, 6, 7, 8}, g[4];
    CHECK(decode(ex, s, p, 8, 4, g) == 0);
    flerp_close(ex);
}
#endif

#ifdef V3_ON_V2
/* A version 3 caller over the frozen version 2 stand-in: links only version 2 calls and decides from caps. */
static void v3_on_v2(void) {
    static const char *const kv[] = {"vocab", "5000"};
    flerp_executor *ex = open_with(kv, 1);
    flerp_caps c;
    memset(&c, 0x5A, sizeof(c));
    c.size = sizeof(c);
    CHECK(flerp_get_caps(ex, &c) == FLERP_OK);
    CHECK(c.abi == 2 && c.size == FLERP_CAPS_V2_SIZE);        /* the version 2 executor knows 72 bytes */
    const int v3 = c.abi >= 3 && c.size >= FLERP_HAS(flerp_caps, images);
    CHECK(!v3);
    unsigned char *raw = (unsigned char *)&c;
    int untouched = 1;
    for (size_t i = FLERP_CAPS_V2_SIZE; i < sizeof(c); ++i) untouched &= raw[i] == 0x5A;
    CHECK(untouched);
    flerp_seq_args sa;
    memset(&sa, 0, sizeof(sa));
    sa.size = sizeof(sa);                                     /* a longer struct than the executor knows */
    int32_t s;
    CHECK(flerp_seq_new(ex, &sa, &s) == FLERP_OK);
    flerp_token p[8] = {1, 2, 3, 4, 5, 6, 7, 8}, g[4];
    CHECK(decode(ex, s, p, 8, 4, g) == 0);
    flerp_close(ex);
}
#endif

#ifdef V2_CALLER
/* A caller built against the version 2 header over the version 3 stand-in with every feature on. */
static void v2_caller(void) {
    static const char *const kv[] = {"vocab", "5000", "features", "all", "draft", "step"};
    flerp_executor *ex = open_with(kv, 3);
    struct {
        flerp_caps c;
        unsigned char guard[64];
    } g;
    memset(&g, 0, sizeof(g));
    memset(g.guard, 0xCD, sizeof(g.guard));
    g.c.size = sizeof(g.c);
    CHECK(flerp_get_caps(ex, &g.c) == FLERP_OK);
    int intact = 1;
    for (size_t i = 0; i < sizeof(g.guard); ++i) intact &= g.guard[i] == 0xCD;
    CHECK(intact);                                            /* never written past the version 2 struct */
    CHECK(g.c.abi == 3 && g.c.size == sizeof(flerp_caps) && (g.c.flags & FLERP_CAP_DRAFT_STEP));
    flerp_seq_args sa;
    memset(&sa, 0, sizeof(sa));
    sa.size = sizeof(sa), sa.drafts = 1;
    int32_t s;
    CHECK(flerp_seq_new(ex, &sa, &s) == FLERP_OK);
    flerp_token p[8] = {1, 2, 3, 4, 5, 6, 7, 8}, plain[6];
    CHECK(decode(ex, s, p, 8, 6, plain) == 0);
    /* one verify of the pending token and a stepped draft, as a version 2 serving loop makes it */
    CHECK(flerp_seq_reset(ex, s) == FLERP_OK);
    flerp_want w;
    memset(&w, 0, sizeof(w));
    w.size = sizeof(w), w.delivery = FLERP_OUT_TOKENS, w.rows = 1;
    flerp_token t, vt[2];
    flerp_rows r;
    memset(&r, 0, sizeof(r));
    r.size = sizeof(r), r.tokens = &t, r.tokens_cap = 1;
    CHECK(flerp_forward(ex, s, p, 8, &w, &r, NULL, 0) == FLERP_OK);
    flerp_draft d;
    CHECK(flerp_draft_step(ex, s, t, 0, &d) == FLERP_OK);
    flerp_token ids[2] = {t, d.token};
    flerp_rows vr;
    memset(&vr, 0, sizeof(vr));
    vr.size = sizeof(vr), vr.tokens = vt, vr.tokens_cap = 2;
    flerp_verify_run run;
    memset(&run, 0, sizeof(run));
    run.size = sizeof(run), run.seq = s, run.ids = ids, run.n_ids = 2, run.out = &vr;
    CHECK(flerp_verify(ex, &run, 1, &w) == FLERP_OK);
    CHECK(t == plain[0] && vt[0] == plain[1]);
    CHECK(flerp_keep_prefix(ex, s, 1, vt[0]) == FLERP_OK || flerp_keep(ex, s) == FLERP_OK);
    flerp_close(ex);
}
#endif

int main(void) {
#if defined(V2_CALLER)
    v2_caller();
#elif defined(V3_ON_V2)
    v3_on_v2();
#else
    every_call();
    as_version_2();
    v2_sized_structs();
#endif
    printf("%d failures\n", failures);
    return failures;
}
