/* Executor ABI conformance: drives whichever executor it is linked with through include/flerp/executor.h only.
 *
 *   conformance --out FILE [--delivery tokens|candidates|full] key=value ... [--case NAME] case options ...
 *
 * key=value pairs go to flerp_open as they are. The executor is opened once; every `--case NAME` starts another
 * case on it (options before the first --case form one case named "default"). The result is one JSON object with
 * a "cases" array. Exit status 0 only when every check of every case passed.
 *
 * Case options:
 *   --ids FILE          the prompt's token ids (decimal, whitespace separated)
 *   --paths LIST        which paths run, comma separated (default plain,speculative,snapshot):
 *     plain             prompt forward, then one token a forward, N tokens (--new N)
 *     speculative       prompt forward, then rounds of --draft D drafts + verify + keep / keep_prefix; every fourth
 *                       round is first dropped (flerp_drop) and verified again (--no-drop leaves that out)
 *     snapshot          a save at prompt position P (--snap-at P), M tokens decoded (--snap-new M), the save
 *                       restored, M tokens again; with FLERP_CAP_SNAPSHOT_BYTES also exported, the seq reset,
 *                       imported, and decoded a third time. The reference is the same prompt forwarded with the
 *                       same cuts and no save
 *     loop              the decode loop of strixite's serving engine (Engine::run, serve/engine.cpp at 20937dc)
 *                       written over the ABI: sample; draft with flerp_draft_step while the head's top-2 margin
 *                       reaches --mtp-margin, at most --mtp-draft and never past --max-tokens; one verify of the
 *                       token and its drafts; keep, or keep_prefix and carry the corrected token into the next
 *                       verify. It writes the tokens that entered the sequence and one {drafted, accepted, emitted}
 *                       record a step, the form of the server's own capture files (serve/capture.hpp), so the two
 *                       can be compared record for record: same forwards, same rows a forward
 *     loop-drop         the loop again with every fourth verify dropped and run again; the second run's rows must
 *                       be the first's (candidates: ids and logits bit for bit) and the result the loop's
 *   --cut P             (repeatable) a prompt forward ends at position P, as a serving loop cuts at its boundaries
 *   --turn P            loop: a snapshot is saved when a prompt forward ends at P (the engine's turn snapshot)
 *   --fed AT:ID,ID,..   (repeatable) loop: ids the serving loop feeds instead of sampling once AT tokens have
 *                       entered the sequence after the prompt (the engine's thinking-budget stop text)
 *   --end-ids ID,ID     loop: a sampled id that ends the turn; --n-valid V: tokens are drawn from ids [0, V)
 *   --cross-shape observe   comparisons between paths whose forwards carry different row counts are written as
 *                       "observe_*" and are not checks. For an executor whose logits depend on the rows of a
 *                       forward (strixite: docs/EXECUTOR-ABI.md, "Forward shape") they are not expected to hold;
 *                       the default (require) is for one that promises output-invariant drafts (StrixFold)
 *   --no-snapshot       same as leaving snapshot out of --paths
 * Each round is traced on stderr (with candidates: the top-2 margin of each row).
 */
#include "flerp/decode_loop.h"
#include "flerp/executor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    long at;
    flerp_token ids[128];
    int n;
} fed_t;

typedef struct {
    const char *name, *ids_path;
    size_t n_new, snap_new;
    long snap_at, cuts[8], max_tokens, n_valid, turn;
    int depth, n_cuts, no_drop, observe, mtp_draft, n_end, n_fed;
    int run_plain, run_spec, run_snap, run_loop, run_loop_drop;
    double mtp_margin;
    flerp_token end_ids[8];
    fed_t fed[8];
} case_t;

static flerp_executor *ex;
static flerp_caps caps;
static uint32_t delivery;
static FILE *out;
static int failures;
static const case_t *cur; /* the case being run: its cuts and n_valid shape every forward */
static int in_case;       /* the JSON writer is inside a case object */

static void die(const char *what) {
    fprintf(stderr, "conformance: %s: %s\n", what, ex ? flerp_last_error(ex) : "");
    if (out) fprintf(out, in_case ? "\"error\": \"%s\"}]}\n" : "\"error\": \"%s\"}\n", what), fclose(out);
    exit(2);
}

#define CHECK(call) do { if ((call) != FLERP_OK) die(#call); } while (0)

static void check(const char *name, int ok) {
    fprintf(out, "\"check_%s\": %s,\n", name, ok ? "true" : "false");
    fprintf(stderr, "conformance: %-34s %s\n", name, ok ? "ok" : "FAILED");
    if (!ok) ++failures;
}

/* A comparison across forward shapes: a check, or with --cross-shape observe a recorded observation. */
static void cross(const char *name, int ok) {
    if (!cur->observe) {
        check(name, ok);
        return;
    }
    fprintf(out, "\"observe_%s\": %s,\n", name, ok ? "true" : "false");
    fprintf(stderr, "conformance: %-34s %s (observation)\n", name, ok ? "equal" : "differs");
}

static void put_str(const char *name, const char *v) {
    fprintf(out, "\"%s\": \"", name);
    for (; *v; ++v) {
        if (*v == '"' || *v == '\\') fputc('\\', out);
        fputc((unsigned char)*v < 32 ? ' ' : *v, out);
    }
    fprintf(out, "\",\n");
}

static void put_ids(const char *name, const flerp_token *t, size_t n) {
    fprintf(out, "\"%s\": [", name);
    for (size_t i = 0; i < n; ++i) fprintf(out, "%s%d", i ? ", " : "", (int)t[i]);
    fprintf(out, "],\n");
}

static size_t first_diff(const flerp_token *a, const flerp_token *b, size_t n) {
    size_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}

/* Result buffers for up to `buf_rows` rows of the chosen delivery; greedy() reduces a row to its greedy token. */
static float *full_buf, *cand_logit;
static flerp_token *cand_id, *tok_buf;
static int64_t buf_rows;

static void rows_init(flerp_rows *r) {
    memset(r, 0, sizeof(*r));
    r->size = sizeof(*r);
    if (delivery == FLERP_OUT_TOKENS) r->tokens = tok_buf, r->tokens_cap = (size_t)buf_rows;
    else if (delivery == FLERP_OUT_CANDIDATES)
        r->cand_id = cand_id, r->cand_logit = cand_logit, r->cand_cap = (size_t)(buf_rows * caps.candidates);
    else r->full = full_buf, r->full_cap = (size_t)(buf_rows * caps.logits_row);
}

static flerp_token greedy(const flerp_rows *r, int64_t row) {
    if (delivery == FLERP_OUT_TOKENS) return r->tokens[row];
    if (delivery == FLERP_OUT_CANDIDATES) return r->cand_id[row * caps.candidates];
    const float *x = r->full + row * caps.logits_row;
    const int64_t n = cur->n_valid > 0 && cur->n_valid < caps.logits_row ? cur->n_valid : caps.logits_row;
    int64_t best = 0;
    for (int64_t i = 1; i < n; ++i) if (x[i] > x[best]) best = i; /* the lower id on a tie */
    return (flerp_token)best;
}

static flerp_want want_rows(int64_t rows) {
    flerp_want w;
    memset(&w, 0, sizeof(w));
    w.size = sizeof(w), w.delivery = rows ? delivery : FLERP_OUT_NONE, w.rows = rows;
    w.n_valid = rows ? cur->n_valid : 0;
    return w;
}

/* ids[from, n) into the seq in forwards of at most caps.max_forward, ending at the case's cuts; the token after
 * the last id when `last`. `engine`: as strixite's engine forwards a prompt - the next forward's ids go along as
 * the lookahead hint, and a snapshot is saved where a forward ends at the case's turn point (*turn_snap). The
 * forwards are flerp_feed_ids' (include/flerp/decode_loop.h), the ones the front end makes. */
static flerp_token feed(int32_t seq, const flerp_token *ids, size_t from, size_t n, int last, int engine, flerp_snapshot *turn_snap) {
    flerp_rows r;
    flerp_feed f;
    int64_t cuts[8];
    const flerp_want w = want_rows(1);
    memset(&f, 0, sizeof(f));
    f.size = sizeof(f), f.ex = ex, f.seq = seq, f.max_forward = caps.max_forward, f.lookahead = engine, f.snapshot_at = -1;
    if (n > 1) { /* prompt forwards only */
        for (int c = 0; c < cur->n_cuts; ++c) cuts[c] = cur->cuts[c];
        f.cuts = cuts, f.n_cuts = (size_t)cur->n_cuts;
    }
    if (engine && turn_snap && (caps.flags & FLERP_CAP_SNAPSHOT) && (caps.flags & FLERP_CAP_SNAPSHOT_ANYWHERE))
        f.snapshot_at = cur->turn, f.snapshot = turn_snap;
    rows_init(&r);
    CHECK(flerp_feed_ids(&f, ids, from, n, last ? &w : NULL, &r));
    if (!last) return -1;
    if (r.rows != 1) die("a forward that asked one row returned another count");
    return greedy(&r, 0);
}

static void plain(int32_t seq, flerp_token first, flerp_token *t, size_t n) {
    t[0] = first;
    for (size_t i = 1; i < n; ++i) t[i] = feed(seq, &t[i - 1], 0, 1, 1, 0, NULL);
}

/* Rounds of a fixed draft depth + verify. Returns the rounds run; counts kept and rejected drafts and dropped rounds. */
static size_t speculative(int32_t seq, flerp_token first, flerp_token *t, size_t n, int depth, size_t *drafted,
                          size_t *accepted, size_t *dropped) {
    const int step = (caps.flags & FLERP_CAP_DRAFT_STEP) != 0, held = (caps.flags & FLERP_CAP_DRAFT_HELD) != 0;
    flerp_token ids[64], drafts[64];
    size_t have = 1, rounds = 0;
    t[0] = first;
    if (depth > caps.max_draft) depth = caps.max_draft;
    if (depth > (int)caps.max_verify_rows - 1) depth = (int)caps.max_verify_rows - 1;
    if (depth > 63) depth = 63;
    while (have < n) {
        int k = depth;
        if ((size_t)k > n - have - 1) k = (int)(n - have - 1);
        if (!step && !held) k = 0;
        const flerp_token pending = t[have - 1];
        flerp_rows r;
        flerp_verify_run run;
        const flerp_want w = want_rows(1);
        for (int attempt = 0; attempt < 2; ++attempt) {
            /* the executor's own drafts: a chain handed back, or held on the device and named by count */
            if (step) {
                flerp_draft d;
                flerp_token prev = pending;
                for (int s = 0; s < k; ++s) {
                    CHECK(flerp_draft_step(ex, seq, prev, s, &d));
                    if (attempt == 1 && drafts[s] != d.token) die("the draft chain changed after a drop");
                    drafts[s] = prev = d.token;
                }
            } else if (held && k > 0) {
                CHECK(flerp_draft_hold(ex, seq, pending, k));
            }
            memset(&run, 0, sizeof(run));
            run.size = sizeof(run), run.seq = seq, run.out = &r;
            ids[0] = pending;
            if (step) memcpy(ids + 1, drafts, (size_t)k * sizeof(flerp_token));
            run.ids = ids, run.n_ids = step ? (size_t)k + 1 : 1;
            if (held) run.held = k, run.held_out = drafts, run.held_cap = 64;
            rows_init(&r);
            CHECK(flerp_verify(ex, &run, 1, &w));
            if (r.rows != k + 1) die("a verify returned a row count other than 1 + its drafts");
            if (attempt == 0 && rounds % 4 == 3 && !cur->no_drop) { /* undo the whole verify, then run the round again */
                if (flerp_forward(ex, seq, ids, 1, &w, &r, NULL, 0) != FLERP_E_STATE) die("a forward was accepted while a verify was pending");
                CHECK(flerp_drop(ex, seq));
                ++*dropped;
                continue;
            }
            break;
        }
        int j = 0;
        while (j < k && greedy(&r, j) == drafts[j]) ++j;
        const flerp_token v = greedy(&r, j);
        fprintf(stderr, "round %zu at %zu: pending %d, %d drafts, %d accepted, next %d%s", rounds, have, (int)pending, k, j, (int)v,
                rounds % 4 == 3 && !cur->no_drop ? ", dropped once" : "");
        if (delivery == FLERP_OUT_CANDIDATES)
            for (int row = 0; row <= j; ++row)
                fprintf(stderr, " [row %d top %d margin %.4f]", row, (int)r.cand_id[row * caps.candidates],
                        (double)(r.cand_logit[row * caps.candidates] - r.cand_logit[row * caps.candidates + 1]));
        fputc('\n', stderr);
        if (j == k) CHECK(flerp_keep(ex, seq));
        else CHECK(flerp_keep_prefix(ex, seq, j + 1, v));
        memcpy(t + have, drafts, (size_t)j * sizeof(flerp_token));
        t[have + (size_t)j] = v;
        have += (size_t)j + 1;
        *drafted += (size_t)k, *accepted += (size_t)j;
        ++rounds;
    }
    return rounds;
}

/* --- the engine's decode loop over the ABI ------------------------------------------------------------------ */
typedef struct {
    flerp_token *tok;            /* tokens that entered the sequence after the prompt */
    size_t n, cap;
    unsigned char (*steps)[3];   /* {drafted, accepted, emitted} a step; drafted 255 = fed ids (kThinkingStopStep) */
    size_t n_steps;
    const char *finish;
    long completion;             /* the engine's completion_tokens: sampled and fed, forwarded or not */
    size_t verifies, redone, drafted, accepted;
    int redo_rows_equal, drops;
    int32_t seq;
    flerp_token first_tok[64];
} loop_out;

/* The loop's hooks: greedy picks, the case's fed ids, the records, and for loop-drop every fourth verify undone. */
static flerp_token hook_pick(void *ctx, const flerp_rows *rows, int64_t row) {
    (void)ctx;
    return greedy(rows, row);
}

static int hook_fed(void *ctx, int64_t at, const flerp_token **ids) {
    (void)ctx;
    for (int i = 0; i < cur->n_fed; ++i)
        if (cur->fed[i].at == at) {
            *ids = cur->fed[i].ids;
            return cur->fed[i].n;
        }
    return 0;
}

static void hook_step(void *ctx, int drafted, int accepted, int emitted) {
    loop_out *o = ctx;
    if (o->n_steps + 1 > o->cap) die("the loop ran past its buffers");
    o->steps[o->n_steps][0] = (unsigned char)drafted, o->steps[o->n_steps][1] = (unsigned char)accepted;
    o->steps[o->n_steps++][2] = (unsigned char)emitted;
    if (drafted == FLERP_FED_STEP) fprintf(stderr, "step %zu: %d ids fed\n", o->n_steps - 1, emitted);
}

static void hook_entered(void *ctx, const flerp_token *ids, size_t n) {
    loop_out *o = ctx;
    if (o->n + n > o->cap) die("the loop ran past its buffers");
    memcpy(o->tok + o->n, ids, n * sizeof(flerp_token)), o->n += n;
}

static int hook_redo(void *ctx, int64_t verify, const flerp_rows *r, int64_t n_rows, int again) {
    loop_out *o = ctx;
    const size_t ids_bytes = (size_t)(n_rows * caps.candidates) * sizeof(flerp_token);
    const size_t logit_bytes = (size_t)(n_rows * caps.candidates) * sizeof(float);
    char *keep = (char *)full_buf; /* free with the candidates delivery: the first run's rows */
    if (again) { /* the same forward again: the same rows */
        for (int64_t row = 0; row < n_rows; ++row) o->redo_rows_equal &= greedy(r, row) == o->first_tok[row];
        if (delivery == FLERP_OUT_CANDIDATES)
            o->redo_rows_equal &= !memcmp(keep, cand_id, ids_bytes) &&
                                  !memcmp(keep + (size_t)(buf_rows * caps.candidates) * sizeof(flerp_token), cand_logit, logit_bytes);
        return 0;
    }
    if (!o->drops || verify % 4 != 3) return 0;
    for (int64_t row = 0; row < n_rows; ++row) o->first_tok[row] = greedy(r, row);
    if (delivery == FLERP_OUT_CANDIDATES) {
        memcpy(keep, cand_id, ids_bytes);
        memcpy(keep + (size_t)(buf_rows * caps.candidates) * sizeof(flerp_token), cand_logit, logit_bytes);
    }
    flerp_rows probe;
    const flerp_want w = want_rows(1);
    const flerp_token one = 0;
    memset(&probe, 0, sizeof(probe));
    probe.size = sizeof(probe);
    if (flerp_forward(ex, o->seq, &one, 1, &w, &probe, NULL, 0) != FLERP_E_STATE) die("a forward was accepted while a verify was pending");
    return 1;
}

static void hook_verified(void *ctx, const flerp_loop_verify *v) {
    const loop_out *o = ctx;
    fprintf(stderr, "step %zu at %lld: token %d, %d drafts, %d accepted%s%s", o->n_steps, (long long)v->at, (int)v->token, v->drafts,
            v->accepted, v->redone ? ", dropped once" : "", v->cut ? ", cut at a fed stop" : v->ended ? ", ended" : "");
    if (delivery == FLERP_OUT_CANDIDATES)
        for (int row = 0; row <= v->accepted && row <= v->drafts; ++row)
            fprintf(stderr, " [row %d top %d margin %.4f]", row, (int)v->rows->cand_id[row * caps.candidates],
                    (double)(v->rows->cand_logit[row * caps.candidates] - v->rows->cand_logit[row * caps.candidates + 1]));
    fputc('\n', stderr);
}

/* Engine::run from the prompt's forward to the end of the request (serve/engine.cpp 1256-1691 at 20937dc), for a
 * text request without a response format, logprobs, nudges or several slots: flerp_decode_loop, the loop the front
 * end runs. What the engine decides from the reply's text - the thinking budget's stop - comes from the case
 * (--fed): the ids, and how many tokens the sequence holds when they are fed. `drops`: every fourth verify is
 * dropped once and run again. */
static void engine_loop(int32_t seq, const flerp_token *prompt, size_t n, int drops, loop_out *o) {
    flerp_snapshot turn_snap = 0;
    flerp_rows r;
    flerp_loop L;
    flerp_loop_hooks H;
    flerp_loop_result R;
    memset(o, 0, sizeof(*o));
    o->cap = (size_t)cur->max_tokens + 160;
    o->tok = calloc(o->cap, sizeof(*o->tok)), o->steps = calloc(o->cap, sizeof(*o->steps));
    o->redo_rows_equal = 1, o->drops = drops, o->seq = seq;
    /* the prompt, cut and hinted as the engine does it; its last row stays in the loop's buffers */
    {
        flerp_feed f;
        int64_t cuts[8];
        const flerp_want w = want_rows(1);
        memset(&f, 0, sizeof(f));
        f.size = sizeof(f), f.ex = ex, f.seq = seq, f.max_forward = caps.max_forward, f.lookahead = 1, f.snapshot_at = -1;
        for (int c = 0; c < cur->n_cuts; ++c) cuts[c] = cur->cuts[c];
        f.cuts = cuts, f.n_cuts = (size_t)cur->n_cuts;
        if ((caps.flags & FLERP_CAP_SNAPSHOT) && (caps.flags & FLERP_CAP_SNAPSHOT_ANYWHERE)) f.snapshot_at = cur->turn, f.snapshot = &turn_snap;
        rows_init(&r);
        CHECK(flerp_feed_ids(&f, prompt, 0, n, &w, &r));
        if (r.rows != 1) die("a forward that asked one row returned another count");
    }
    memset(&L, 0, sizeof(L));
    L.size = sizeof(L), L.ex = ex, L.seq = seq, L.caps = &caps, L.delivery = delivery, L.n_valid = cur->n_valid, L.rows = &r;
    L.max_tokens = cur->max_tokens, L.drafts = cur->mtp_draft, L.margin = cur->mtp_margin;
    L.end_ids = cur->end_ids, L.n_end = (size_t)cur->n_end;
    memset(&H, 0, sizeof(H));
    H.ctx = o, H.pick = hook_pick, H.fed = hook_fed, H.redo = hook_redo, H.step = hook_step, H.entered = hook_entered;
    H.verified = hook_verified;
    memset(&R, 0, sizeof(R));
    R.size = sizeof(R);
    if (flerp_decode_loop(&L, &H, &R) != FLERP_OK) die(R.error ? R.error : "flerp_decode_loop");
    o->finish = R.finish, o->completion = (long)R.completion;
    o->verifies = (size_t)R.verifies, o->redone = (size_t)R.redone, o->drafted = (size_t)R.drafted, o->accepted = (size_t)R.accepted;
    if (turn_snap) CHECK(flerp_snapshot_drop(ex, seq, turn_snap));
}

static void put_loop(const char *name, const loop_out *o) {
    put_ids(name, o->tok, o->n);
    fprintf(out, "\"%s_steps\": [", name);
    for (size_t i = 0; i < o->n_steps; ++i)
        fprintf(out, "%s[%d, %d, %d]", i ? ", " : "", o->steps[i][0], o->steps[i][1], o->steps[i][2]);
    fprintf(out, "],\n\"%s_finish\": \"%s\", \"%s_completion_tokens\": %ld, \"%s_verifies\": %zu, \"%s_drafted\": %zu, "
                 "\"%s_accepted\": %zu, \"%s_dropped_verifies\": %zu,\n",
            name, o->finish, name, o->completion, name, o->verifies, name, o->drafted, name, o->accepted, name, o->redone);
}

static flerp_token *read_ids(const char *path, size_t *n_out) {
    FILE *f = fopen(path, "r");
    if (!f) {
        perror(path);
        exit(2);
    }
    size_t n = 0, cap = 1024;
    flerp_token *ids = malloc(cap * sizeof(*ids));
    long v;
    while (fscanf(f, "%ld", &v) == 1) {
        if (n == cap) ids = realloc(ids, (cap *= 2) * sizeof(*ids));
        ids[n++] = (flerp_token)v;
    }
    fclose(f);
    *n_out = n;
    return ids;
}

static int parse_list(const char *s, flerp_token *ids, int cap) {
    int n = 0;
    while (*s && n < cap) {
        char *end;
        ids[n++] = (flerp_token)strtol(s, &end, 10);
        if (end == s) return -1;
        s = *end == ',' ? end + 1 : end;
        if (*end && *end != ',') return -1;
    }
    return *s ? -1 : n;
}

static void run_case(const case_t *c, int32_t seq) {
    size_t n = 0;
    flerp_token *prompt = read_ids(c->ids_path, &n);
    if (n < 2) die("the prompt holds fewer than 2 ids");
    cur = c;
    const size_t n_new = c->n_new;
    flerp_token *a = calloc(n_new + 1, sizeof(*a)), *b = calloc(n_new + 1, sizeof(*b));
    fprintf(stderr, "conformance: case %s (%zu prompt ids)\n", c->name, n);
    in_case = 1;
    fprintf(out, "{\"name\": \"%s\", \"prompt_tokens\": %zu, \"new\": %zu, \"no_drop\": %s, \"cross_shape\": \"%s\", \"n_valid\": %ld,\n",
            c->name, n, n_new, c->no_drop ? "true" : "false", c->observe ? "observe" : "require", c->n_valid);
    flerp_token cut_ids[8];
    for (int i = 0; i < c->n_cuts; ++i) cut_ids[i] = (flerp_token)c->cuts[i];
    put_ids("cuts", cut_ids, (size_t)c->n_cuts);

    if (c->run_plain) {
        CHECK(flerp_seq_reset(ex, seq));
        check("reset_position", flerp_seq_pos(ex, seq) == 0);
        plain(seq, feed(seq, prompt, 0, n, 1, 0, NULL), a, n_new);
        put_ids("plain", a, n_new);
        check("plain_position", flerp_seq_pos(ex, seq) == (int64_t)(n + n_new - 1));
    }
    if (c->run_spec) {
        size_t drafted = 0, accepted = 0, dropped = 0;
        CHECK(flerp_seq_reset(ex, seq));
        const size_t rounds = speculative(seq, feed(seq, prompt, 0, n, 1, 0, NULL), b, n_new, c->depth, &drafted, &accepted, &dropped);
        put_ids("speculative", b, n_new);
        fprintf(out, "\"rounds\": %zu, \"drafted\": %zu, \"accepted\": %zu, \"dropped_rounds\": %zu,\n", rounds, drafted, accepted, dropped);
        if (c->run_plain) {
            fprintf(out, "\"speculative_first_difference\": %zu,\n", first_diff(a, b, n_new));
            cross("speculative_equals_plain", !memcmp(a, b, n_new * sizeof(*a)));
        }
        check("speculative_position", flerp_seq_pos(ex, seq) == (int64_t)(n + n_new - 1));
        check("verify_keep_drop_exercised", rounds > 0 && (n_new < 6 || c->no_drop || dropped > 0));
    }
    if (c->run_loop || c->run_loop_drop) {
        loop_out l, ld;
        CHECK(flerp_seq_reset(ex, seq));
        engine_loop(seq, prompt, n, 0, &l);
        put_loop("loop", &l);
        check("loop_position", flerp_seq_pos(ex, seq) == (int64_t)(n + l.n));
        if (c->run_loop_drop) {
            CHECK(flerp_seq_reset(ex, seq));
            engine_loop(seq, prompt, n, 1, &ld);
            put_loop("loop_drop", &ld);
            check("loop_drop_equals_loop", ld.n == l.n && !memcmp(ld.tok, l.tok, l.n * sizeof(*l.tok)) &&
                                               ld.n_steps == l.n_steps && !memcmp(ld.steps, l.steps, l.n_steps * 3));
            check("dropped_verify_rows_equal_again", ld.redo_rows_equal);
            check("verify_drop_exercised", ld.redone > 0 || l.verifies < 4);
        }
        if (c->run_plain) { /* one row a forward against the engine's rows a forward */
            const size_t m = l.n < n_new ? l.n : n_new;
            fprintf(out, "\"loop_first_difference_from_plain\": %zu,\n", first_diff(a, l.tok, m));
            if (l.verifies > 0) fprintf(out, "\"observe_loop_equals_plain\": %s,\n", first_diff(a, l.tok, m) == m ? "true" : "false");
        }
    }
    if (c->run_snap && (caps.flags & FLERP_CAP_SNAPSHOT)) {
        const int anywhere = (caps.flags & FLERP_CAP_SNAPSHOT_ANYWHERE) != 0;
        const long snap_at = c->snap_at < 0 ? (long)(n / 2) : c->snap_at;
        if (snap_at < 1 || (size_t)snap_at >= n) die("--snap-at must be inside the prompt");
        const size_t snap_new = c->run_plain && c->snap_new > n_new ? n_new : c->snap_new;
        flerp_token *s0 = calloc(snap_new, sizeof(*s0)), *s1 = calloc(snap_new, sizeof(*s1)), *s2 = calloc(snap_new, sizeof(*s2));
        flerp_snapshot snap = 0;
        /* the reference: the same forwards (the prompt cut at the save's position too), no save */
        CHECK(flerp_seq_reset(ex, seq));
        feed(seq, prompt, 0, (size_t)snap_at, 0, 0, NULL);
        plain(seq, feed(seq, prompt, (size_t)snap_at, n, 1, 0, NULL), s0, snap_new);
        CHECK(flerp_seq_reset(ex, seq));
        feed(seq, prompt, 0, (size_t)snap_at, 0, 0, NULL);
        CHECK(flerp_snapshot_save(ex, seq, &snap));
        plain(seq, feed(seq, prompt, (size_t)snap_at, n, 1, 0, NULL), s1, snap_new);
        check("snapshot_position", flerp_snapshot_pos(ex, seq, snap) == snap_at);
        if (!anywhere) CHECK(flerp_seq_reset(ex, seq)); /* such an executor restores only into a prompt pass */
        CHECK(flerp_snapshot_restore(ex, seq, snap));
        check("restored_position", flerp_seq_pos(ex, seq) == snap_at);
        plain(seq, feed(seq, prompt, (size_t)snap_at, n, 1, 0, NULL), s2, snap_new);
        fprintf(out, "\"snapshot_at\": %ld,\n", snap_at);
        put_ids("snapshot_reference", s0, snap_new);
        put_ids("snapshot_first", s1, snap_new);
        put_ids("snapshot_restored", s2, snap_new);
        check("snapshot_save_changes_nothing", !memcmp(s0, s1, snap_new * sizeof(*s1)));
        check("snapshot_restore_equal", !memcmp(s1, s2, snap_new * sizeof(*s1)));
        if (c->run_plain) cross("snapshot_equals_plain", !memcmp(s1, a, snap_new * sizeof(*s1)));
        if (caps.flags & FLERP_CAP_SNAPSHOT_BYTES) {
            const int64_t bytes = flerp_snapshot_bytes(ex, seq, snap, 0);
            size_t written = 0;
            void *state = bytes > 0 ? malloc((size_t)bytes) : NULL;
            if (!state) die("flerp_snapshot_bytes");
            CHECK(flerp_snapshot_export(ex, seq, snap, 0, state, (size_t)bytes, &written));
            CHECK(flerp_seq_reset(ex, seq));
            CHECK(flerp_state_import(ex, seq, state, written, snap_at, 0));
            check("imported_position", flerp_seq_pos(ex, seq) == snap_at);
            plain(seq, feed(seq, prompt, (size_t)snap_at, n, 1, 0, NULL), s2, snap_new);
            put_ids("snapshot_imported", s2, snap_new);
            fprintf(out, "\"snapshot_bytes\": %lld,\n", (long long)bytes);
            check("snapshot_import_equal", written == (size_t)bytes && !memcmp(s1, s2, snap_new * sizeof(*s1)));
            free(state);
        } else {
            CHECK(flerp_snapshot_drop(ex, seq, snap));
        }
    } else if (c->run_snap) {
        fprintf(out, "\"snapshot\": \"not offered (caps.flags)\",\n");
    }
    fprintf(out, "\"done\": true}");
    in_case = 0;
    free(prompt), free(a), free(b);
}

static int usage(void) {
    fprintf(stderr, "usage: conformance --out FILE [--delivery tokens|candidates|full] key=value ... [--case NAME] --ids FILE "
                    "[--paths plain,speculative,snapshot,loop,loop-drop] [--new N] [--draft D] [--snap-at P] [--snap-new M] [--cut P]... "
                    "[--no-drop] [--no-snapshot] [--cross-shape require|observe] [--max-tokens N] [--mtp-draft D] [--mtp-margin M] "
                    "[--end-ids ID,ID] [--n-valid V] [--turn P] [--fed AT:ID,ID,..]... [--case NAME ...]\n");
    return 2;
}

int main(int argc, char **argv) {
    const char *out_path = NULL, *want_delivery = NULL;
    static case_t cases[32];
    int n_cases = 0;
    flerp_kv kv[64];
    size_t n_kv = 0;
    case_t *c = NULL;
    for (int i = 1; i < argc; ++i) {
        char *eq = strchr(argv[i], '=');
        const int more = i + 1 < argc;
        if (!strcmp(argv[i], "--out") && more) { out_path = argv[++i]; continue; }
        if (!strcmp(argv[i], "--delivery") && more) { want_delivery = argv[++i]; continue; }
        if (eq && n_kv < 64 && strncmp(argv[i], "--", 2)) { *eq = 0, kv[n_kv].key = argv[i], kv[n_kv++].value = eq + 1; continue; }
        if (c == NULL || !strcmp(argv[i], "--case")) {
            if (n_cases == 32) return usage();
            c = &cases[n_cases++];
            c->name = "default", c->n_new = 32, c->snap_new = 8, c->snap_at = -1, c->depth = 4, c->turn = -1;
            c->run_plain = c->run_spec = c->run_snap = 1, c->max_tokens = 0, c->mtp_draft = 7, c->mtp_margin = 1.5;
            if (!strcmp(argv[i], "--case")) {
                if (!more) return usage();
                c->name = argv[++i];
                continue;
            }
        }
        if (!strcmp(argv[i], "--ids") && more) c->ids_path = argv[++i];
        else if (!strcmp(argv[i], "--new") && more) c->n_new = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--draft") && more) c->depth = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--snap-at") && more) c->snap_at = atol(argv[++i]);
        else if (!strcmp(argv[i], "--snap-new") && more) c->snap_new = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--cut") && more && c->n_cuts < 8) c->cuts[c->n_cuts++] = atol(argv[++i]);
        else if (!strcmp(argv[i], "--no-drop")) c->no_drop = 1;
        else if (!strcmp(argv[i], "--no-snapshot")) c->run_snap = 0;
        else if (!strcmp(argv[i], "--max-tokens") && more) c->max_tokens = atol(argv[++i]);
        else if (!strcmp(argv[i], "--mtp-draft") && more) c->mtp_draft = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mtp-margin") && more) c->mtp_margin = atof(argv[++i]);
        else if (!strcmp(argv[i], "--n-valid") && more) c->n_valid = atol(argv[++i]);
        else if (!strcmp(argv[i], "--turn") && more) c->turn = atol(argv[++i]);
        else if (!strcmp(argv[i], "--end-ids") && more) {
            if ((c->n_end = parse_list(argv[++i], c->end_ids, 8)) < 0) return usage();
        } else if (!strcmp(argv[i], "--fed") && more && c->n_fed < 8) {
            char *colon = strchr(argv[++i], ':');
            fed_t *f = &c->fed[c->n_fed++];
            if (!colon || (f->n = parse_list(colon + 1, f->ids, 128)) < 1) return usage();
            f->at = atol(argv[i]);
        } else if (!strcmp(argv[i], "--cross-shape") && more) {
            ++i;
            if (strcmp(argv[i], "observe") && strcmp(argv[i], "require")) return usage();
            c->observe = !strcmp(argv[i], "observe");
        } else if (!strcmp(argv[i], "--paths") && more) {
            char *p = argv[++i];
            c->run_plain = c->run_spec = c->run_snap = c->run_loop = c->run_loop_drop = 0;
            for (char *t = strtok(p, ","); t; t = strtok(NULL, ",")) {
                if (!strcmp(t, "plain")) c->run_plain = 1;
                else if (!strcmp(t, "speculative")) c->run_spec = 1;
                else if (!strcmp(t, "snapshot")) c->run_snap = 1;
                else if (!strcmp(t, "loop")) c->run_loop = 1;
                else if (!strcmp(t, "loop-drop")) c->run_loop = c->run_loop_drop = 1;
                else return usage();
            }
        } else return usage();
    }
    if (!out_path || n_cases == 0) return usage();
    for (int i = 0; i < n_cases; ++i) {
        c = &cases[i];
        if (!c->ids_path || ((c->run_plain || c->run_spec) && c->n_new < 2) || (c->run_loop && c->max_tokens < 1))
            return fprintf(stderr, "conformance: case %s: --ids is required, --new >= 2 for plain and speculative, --max-tokens for loop\n", c->name), 2;
    }

    flerp_open_args oa = {sizeof(oa), "conformance", kv, n_kv};
    if (flerp_open(&oa, &ex) != FLERP_OK) return fprintf(stderr, "conformance: flerp_open: %s\n", flerp_last_error(NULL)), 2;
    out = fopen(out_path, "w");
    if (!out) return perror(out_path), 2;
    memset(&caps, 0, sizeof(caps));
    caps.size = sizeof(caps);
    cur = &cases[0];
    CHECK(flerp_get_caps(ex, &caps));
    delivery = caps.deliveries & FLERP_OUT_TOKENS ? FLERP_OUT_TOKENS
               : caps.deliveries & FLERP_OUT_CANDIDATES ? FLERP_OUT_CANDIDATES : FLERP_OUT_FULL;
    if (want_delivery) {
        delivery = !strcmp(want_delivery, "tokens") ? FLERP_OUT_TOKENS : !strcmp(want_delivery, "candidates") ? FLERP_OUT_CANDIDATES : FLERP_OUT_FULL;
        if (!(caps.deliveries & delivery)) die("the executor does not offer the asked --delivery");
    }
    buf_rows = caps.max_verify_rows > 0 ? caps.max_verify_rows : 1;
    const int64_t K = caps.candidates ? caps.candidates : 1;
    tok_buf = malloc((size_t)buf_rows * sizeof(flerp_token));
    cand_id = malloc((size_t)(buf_rows * K) * sizeof(flerp_token));
    cand_logit = malloc((size_t)(buf_rows * K) * sizeof(float));
    /* with full rows: the rows; with candidates: room for a dropped verify's rows (ids, then logits) */
    full_buf = delivery == FLERP_OUT_FULL ? malloc((size_t)(buf_rows * caps.logits_row) * sizeof(float))
                                          : malloc((size_t)(buf_rows * K) * (sizeof(flerp_token) + sizeof(float)));
    fprintf(out, "{\"abi\": %u,\n", caps.abi);
    put_str("describe", flerp_describe(ex));
    put_str("fingerprint", flerp_state_fingerprint(ex));
    fprintf(out, "\"caps\": {\"positions\": %lld, \"max_forward\": %lld, \"max_verify_rows\": %lld, \"logits_row\": %lld, "
                 "\"candidates\": %d, \"seqs\": %d, \"snapshots\": %d, \"max_draft\": %d, \"deliveries\": %u, \"flags\": %u},\n",
            (long long)caps.positions, (long long)caps.max_forward, (long long)caps.max_verify_rows, (long long)caps.logits_row,
            caps.candidates, caps.seqs, caps.snapshots, caps.max_draft, caps.deliveries, caps.flags);
    fprintf(out, "\"delivery\": %u,\n", delivery);
    check("abi_version", caps.abi >= 2 && caps.abi <= FLERP_EXECUTOR_ABI);

    flerp_seq_args sa;
    memset(&sa, 0, sizeof(sa));
    sa.size = sizeof(sa), sa.drafts = 1;
    for (int i = 0; i < n_cases; ++i) {
        const int64_t want = (int64_t)(cases[i].max_tokens > (long)cases[i].n_new ? (size_t)cases[i].max_tokens : cases[i].n_new) + 64;
        if (want > sa.max_new) sa.max_new = want;
    }
    int32_t seq = -1;
    CHECK(flerp_seq_new(ex, &sa, &seq));
    fprintf(out, "\"cases\": [\n");
    for (int i = 0; i < n_cases; ++i) {
        if (i) fprintf(out, ",\n");
        run_case(&cases[i], seq);
    }
    fprintf(out, "],\n");

    flerp_stats st;
    memset(&st, 0, sizeof(st));
    st.size = sizeof(st);
    CHECK(flerp_get_stats(ex, &st));
    char js[1024];
    flerp_stats_json(ex, js, sizeof(js));
    fprintf(out, "\"stats\": {\"forwards\": %lld, \"forward_tokens\": %lld, \"verifies\": %lld, \"verify_rows\": %lld, \"kept_rows\": %lld, "
                 "\"dropped_rows\": %lld, \"draft_tokens\": %lld, \"snapshots_saved\": %lld, \"snapshots_restored\": %lld, \"origin\": %s},\n",
            (long long)st.forwards, (long long)st.forward_tokens, (long long)st.verifies, (long long)st.verify_rows, (long long)st.kept_rows,
            (long long)st.dropped_rows, (long long)st.draft_tokens, (long long)st.snapshots_saved, (long long)st.snapshots_restored, js);
    CHECK(flerp_seq_free(ex, seq));
    fprintf(out, "\"failures\": %d}\n", failures);
    fclose(out); /* the result is complete on disk before the executor is closed */
    fprintf(stderr, "conformance: %s (%d failed); closing the executor\n", failures ? "FAILED" : "passed", failures);
    flerp_close(ex);
    fprintf(stderr, "conformance: the executor is closed\n");
    return failures ? 1 : 0;
}
