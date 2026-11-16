/* A stand-in executor behind include/flerp/executor.h, for testing callers of the ABI without a GPU: no weights,
 * no device. Its "model" is a hash: the token after a context is a function of the context's last id, its length
 * and the steering over its last position, so every way of grouping ids into forwards gives the same tokens (drafts
 * are output-invariant), and a caller's loop can be checked against itself. What it offers is chosen at flerp_open,
 * so one caller can be run against each shape of executor the ABI allows:
 *
 *   vocab=N         ids are [0, N) (required); row=N the floats of a full row (default vocab)
 *   lo=A hi=B       the model's tokens come from [A, B) (default 0 and vocab)
 *   reply=N         after N tokens past the prompt the model gives end=ID (default 24 and vocab - 1)
 *   deliveries=LIST of full,candidates,tokens (default all three)
 *   draft=none|step|held   the draft head's form (default none); every 5th draft is wrong, and a step draft's
 *                   margin is 0.5 where the position is a multiple of 7 and 3 elsewhere
 *   hold_cap=N      draft=held: hold at most N drafts however many are asked, and a verify of the held drafts returns
 *                   1 + what was held (an executor with a depth rule of its own, as qwen35moe_abi draft_value=on;
 *                   default 0: hold what is asked and refuse a verify naming another count)
 *   sampled=on|off  FLERP_CAP_SAMPLED_TOKENS (default off); top_k_max=N, needs_top_k=on|off: flerp_caps version 2
 *   masks=on|off    FLERP_CAP_MASKS, a logit bias only (default on)
 *   positions=N     (default 4096); seqs=N (default 1)
 *   abi=1|2|3       what flerp_get_caps reports and fills, and which calls answer (default 3)
 *   features=LIST   version 3 capabilities (default none; with abi < 3 they are refused at flerp_open):
 *                     state       SNAPSHOT, STATE_PREFIX, SNAPSHOT_MARK, SNAPSHOT_SHARED, STATE_PARK (keep_step=N)
 *                     steer       STEER, STEER_SPAN: 4 floats a vector, 8 layers
 *                     image-rows, image-prepare   IMAGE_ROWS / IMAGE_PREPARE: 4 floats a row, image_token=ID
 *                                 (default vocab - 2)
 *                     constraint  CONSTRAINT, every kind; seq-policy  SEQ_POLICY
 *                     slots       SEQ_BATCHABLE, PREFETCH, RENEW
 *                     all         every one of them
 * A sampled draw (tokens delivery, temperature > 0) is the row's candidate number (seed + position) mod min(top_k, 3):
 * a function of the seed, not a distribution. A constrained draw (tokens delivery) is the best candidate whose id
 * has the parity the constraint's text hashes to. An image span's positions hold ids hashed from its rows or its
 * payload, so a different picture gives a different reply. A state is the ids it covers: export writes them after a
 * 32-byte header, import requires the prefix ids to equal them (StrixFold's rule). */
#include "flerp/executor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define K 20
#define MAX_ROWS 16
#define MAX_SEQS 8
#define MAX_SNAPS 64
#define MAX_VECTORS 16
#define MAX_ITEMS 8
#define MAX_POLICY 16
#define STEER_DIM 4
#define STEER_LAYERS 8
#define IMAGE_HIDDEN 4
#define IMAGE_MAX_ROWS 1024
#define IMAGES 4

enum { F_STATE = 1, F_STEER = 2, F_IMAGE_ROWS = 4, F_IMAGE_PREPARE = 8, F_CONSTRAINT = 16, F_POLICY = 32, F_SLOTS = 64 };

typedef struct {
    uint32_t vector;
    float scale;
    int64_t from, to;
} item_t;

typedef struct {
    int open, pending, sampled, top_k, image, ended;
    uint64_t seed;
    int64_t pos, prompt, v_rows;
    flerp_token *ids; /* [positions]: the ids the seq holds, image positions as their hashed ids */
    flerp_token v_ids[MAX_ROWS];
    flerp_token held[MAX_ROWS];
    int n_held;
    item_t items[MAX_ITEMS];
    size_t n_items;
    int constrained;
    uint32_t parity;
    flerp_token eos[MAX_POLICY], close[MAX_POLICY];
    size_t n_eos, n_close;
    int32_t budget, loop_guard;
} seq_t;

typedef struct {
    int used;
    int owner;    /* the seq, -1 shared, -2 parked */
    int64_t pos;  /* -1: a mark not reached yet */
    int64_t at;
    uint32_t gen;
    flerp_token *ids;
} snap_t;

typedef struct {
    int used;
    uint32_t salt;
} vector_t;

struct flerp_executor {
    int64_t vocab, row, lo, hi, reply, positions, keep_step;
    flerp_token end, image_token;
    uint32_t deliveries;
    int draft, hold_cap, sampled, masks, top_k_max, needs_top_k, abi, n_seqs, features;
    seq_t seqs[MAX_SEQS];
    snap_t snaps[MAX_SNAPS];
    vector_t vectors[MAX_VECTORS];
    uint32_t gen;
    int64_t prefetched, renews;
    flerp_stats stats;
    char error[256], describe[256], fingerprint[96];
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

static uint64_t hash_bytes(const void *p, size_t n) {
    const unsigned char *b = p;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 0x100000001b3ULL;
    return h;
}

static uint32_t caps_flags(const flerp_executor *ex) {
    uint32_t f = FLERP_CAP_KEEP_PREFIX | (ex->draft == 1 ? FLERP_CAP_DRAFT_STEP : 0u) | (ex->draft == 2 ? FLERP_CAP_DRAFT_HELD : 0u) |
                 (ex->masks ? FLERP_CAP_MASKS : 0u) | (ex->sampled ? FLERP_CAP_SAMPLED_TOKENS : 0u);
    if (ex->features & F_STATE)
        f |= FLERP_CAP_SNAPSHOT | FLERP_CAP_STATE_PREFIX | FLERP_CAP_SNAPSHOT_MARK | FLERP_CAP_SNAPSHOT_SHARED | FLERP_CAP_STATE_PARK |
             (ex->keep_step == 0 ? FLERP_CAP_SNAPSHOT_ANYWHERE : 0u);
    if (ex->features & F_STEER) f |= FLERP_CAP_STEER | FLERP_CAP_STEER_SPAN;
    if (ex->features & (F_IMAGE_ROWS | F_IMAGE_PREPARE)) f |= FLERP_CAP_VISION;
    if (ex->features & F_IMAGE_ROWS) f |= FLERP_CAP_IMAGE_ROWS;
    if (ex->features & F_IMAGE_PREPARE) f |= FLERP_CAP_IMAGE_PREPARE;
    if (ex->features & F_CONSTRAINT) f |= FLERP_CAP_CONSTRAINT;
    if (ex->features & F_POLICY) f |= FLERP_CAP_SEQ_POLICY;
    if (ex->features & F_SLOTS) f |= FLERP_CAP_SEQ_BATCHABLE | FLERP_CAP_PREFETCH | FLERP_CAP_RENEW;
    return f;
}

/* A version 3 call: refused unless the executor is version 3 and offers the flag. */
#define NEED(flag, name)                                                                                                   \
    do {                                                                                                                   \
        if (ex->abi < 3 || !(caps_flags(ex) & (flag))) return fail(ex, FLERP_E_UNSUPPORTED, "fake: " name " is not offered"); \
    } while (0)

/* The steering over position p of the seq. */
static uint32_t steer_salt(const flerp_executor *ex, const seq_t *q, int64_t p) {
    uint32_t salt = 0;
    for (size_t i = 0; i < q->n_items; ++i) {
        const item_t *it = &q->items[i];
        if (it->scale == 0.0f || p < it->from || (it->to >= 0 && p >= it->to)) continue;
        uint32_t bits;
        memcpy(&bits, &it->scale, sizeof(bits));
        salt ^= mix(((uint64_t)ex->vectors[it->vector - 1].salt << 32) ^ bits);
    }
    return salt;
}

/* The model: the token after a context of `len` ids that ends with `last`. */
static flerp_token model(const flerp_executor *ex, const seq_t *q, flerp_token last, int64_t len) {
    if (q->prompt >= 0 && len - q->prompt >= ex->reply) return ex->end;
    const uint64_t salt = (uint64_t)steer_salt(ex, q, len - 1) * 0x9e3779b97f4a7c15ULL;
    return (flerp_token)(ex->lo + mix((((uint64_t)(uint32_t)last << 32) ^ (uint64_t)len) ^ salt) % (uint64_t)(ex->hi - ex->lo));
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
    if (q->constrained && w->delivery != FLERP_OUT_TOKENS)
        return fail(ex, FLERP_E_UNSUPPORTED, "fake: a constrained seq is drawn on the executor (FLERP_OUT_TOKENS only)");
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
            if (q->constrained)
                while (pick < K - 1 && ((uint32_t)id[pick] & 1u) != q->parity) ++pick;
            flerp_token t = id[pick];
            const int64_t drawn = q->prompt >= 0 ? len - q->prompt : 0; /* reply draws before this one */
            if (q->n_close > 0 && drawn >= q->budget && drawn < q->budget + (int64_t)q->n_close) t = q->close[drawn - q->budget];
            out->tokens[r] = t;
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

static void free_ex(flerp_executor *ex) {
    if (ex == NULL) return;
    for (int i = 0; i < MAX_SEQS; ++i) free(ex->seqs[i].ids);
    for (int i = 0; i < MAX_SNAPS; ++i) free(ex->snaps[i].ids);
    free(ex->rowbuf);
    free(ex);
}

flerp_status flerp_open(const flerp_open_args *args, flerp_executor **out) {
    static const char *const keys[] = {"vocab", "row", "lo", "hi", "reply", "end", "deliveries", "draft", "hold_cap", "sampled", "top_k_max",
                                       "needs_top_k", "masks", "positions", "seqs", "abi", "features", "keep_step", "image_token"};
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
    ex->image_token = (flerp_token)atoll(value_of(args, "image_token", "-1"));
    if (ex->image_token < 0 || ex->image_token >= ex->vocab) ex->image_token = (flerp_token)(ex->vocab - 2);
    ex->positions = atoll(value_of(args, "positions", "4096"));
    if (ex->positions < 1) ex->positions = 4096;
    ex->keep_step = atoll(value_of(args, "keep_step", "0"));
    if (ex->keep_step < 0) ex->keep_step = 0;
    ex->n_seqs = atoi(value_of(args, "seqs", "1"));
    if (ex->n_seqs < 1 || ex->n_seqs > MAX_SEQS) ex->n_seqs = 1;
    const char *d = value_of(args, "deliveries", "full,candidates,tokens");
    if (strstr(d, "full")) ex->deliveries |= FLERP_OUT_FULL;
    if (strstr(d, "candidates")) ex->deliveries |= FLERP_OUT_CANDIDATES;
    if (strstr(d, "tokens")) ex->deliveries |= FLERP_OUT_TOKENS;
    d = value_of(args, "draft", "none");
    ex->draft = !strcmp(d, "step") ? 1 : !strcmp(d, "held") ? 2 : 0;
    ex->hold_cap = atoi(value_of(args, "hold_cap", "0"));
    ex->sampled = !strcmp(value_of(args, "sampled", "off"), "on");
    ex->masks = !strcmp(value_of(args, "masks", "on"), "on");
    ex->top_k_max = atoi(value_of(args, "top_k_max", "0"));
    ex->needs_top_k = !strcmp(value_of(args, "needs_top_k", "off"), "on");
    ex->abi = atoi(value_of(args, "abi", "3"));
    if (ex->abi < 1 || ex->abi > FLERP_EXECUTOR_ABI) ex->abi = FLERP_EXECUTOR_ABI;
    d = value_of(args, "features", "");
    const int all = strstr(d, "all") != NULL;
    ex->features = (all || strstr(d, "state") ? F_STATE : 0) | (all || strstr(d, "steer") ? F_STEER : 0) |
                   (all || strstr(d, "image-rows") ? F_IMAGE_ROWS : 0) | (all || strstr(d, "image-prepare") ? F_IMAGE_PREPARE : 0) |
                   (all || strstr(d, "constraint") ? F_CONSTRAINT : 0) | (all || strstr(d, "seq-policy") ? F_POLICY : 0) |
                   (all || strstr(d, "slots") ? F_SLOTS : 0);
    if (ex->features && ex->abi < 3) {
        snprintf(g_open_error, sizeof(g_open_error), "flerp_open (fake): features need abi=3");
        free(ex);
        return FLERP_E_CONFIG;
    }
    ex->rowbuf = malloc((size_t)ex->row * sizeof(float));
    for (int i = 0; i < ex->n_seqs; ++i) ex->seqs[i].ids = calloc((size_t)ex->positions, sizeof(flerp_token));
    snprintf(ex->describe, sizeof(ex->describe), "fake executor (a hash, no weights): vocab %lld, deliveries %u, draft form %d, ABI %d, features %d",
             (long long)ex->vocab, ex->deliveries, ex->draft, ex->abi, ex->features);
    snprintf(ex->fingerprint, sizeof(ex->fingerprint), "fake/%lld/%lld/%lld/%lld", (long long)ex->vocab, (long long)ex->lo,
             (long long)ex->hi, (long long)ex->reply);
    *out = ex;
    return FLERP_OK;
}

void flerp_close(flerp_executor *ex) { free_ex(ex); }

flerp_status flerp_get_caps(const flerp_executor *ex, flerp_caps *out) {
    flerp_caps c;
    const size_t v1 = FLERP_HAS(flerp_caps, flags);
    if (out == NULL || out->size < v1) return fail(ex, FLERP_E_ARGUMENT, "flerp_caps is null or short");
    const size_t known = ex->abi < 2 ? v1 : ex->abi < 3 ? FLERP_CAPS_V2_SIZE : sizeof(c);
    memset(&c, 0, sizeof(c));
    c.size = (uint32_t)(ex->abi < 2 ? v1 : out->size < known ? out->size : known);
    c.abi = (uint32_t)ex->abi, c.positions = ex->positions, c.max_forward = 512, c.max_verify_rows = MAX_ROWS;
    c.logits_row = ex->deliveries & FLERP_OUT_FULL ? ex->row : 0;
    c.candidates = ex->deliveries & FLERP_OUT_CANDIDATES ? K : 0;
    c.seqs = ex->n_seqs, c.max_draft = ex->draft ? 8 : 0, c.deliveries = ex->deliveries;
    c.snapshots = ex->features & F_STATE ? 8 : 0;
    c.flags = caps_flags(ex);
    if (ex->sampled) {
        c.sampling = FLERP_SAMPLE_TOP_K | FLERP_SAMPLE_SEED | (ex->needs_top_k ? FLERP_SAMPLE_NEEDS_TOP_K : 0u);
        c.top_k_max = ex->top_k_max;
    }
    c.keep_step = ex->features & F_STATE ? ex->keep_step : 0;
    c.constraints = ex->features & F_CONSTRAINT ? FLERP_CONSTRAINT_JSON_SCHEMA | FLERP_CONSTRAINT_REGEX | FLERP_CONSTRAINT_CHOICE |
                                                       FLERP_CONSTRAINT_GRAMMAR | FLERP_CONSTRAINT_TOOL_CALL_REQUIRED
                                                 : 0u;
    if (ex->features & F_STEER) c.steer_dim = STEER_DIM, c.steer_layers = STEER_LAYERS;
    if (ex->features & (F_IMAGE_ROWS | F_IMAGE_PREPARE)) {
        c.image_token = ex->image_token, c.image_max_rows = IMAGE_MAX_ROWS, c.images = IMAGES;
        c.image_hidden = ex->features & F_IMAGE_ROWS ? IMAGE_HIDDEN : 0;
    }
    memcpy(out, &c, c.size);
    return FLERP_OK;
}

const char *flerp_describe(const flerp_executor *ex) { return ex->describe; }
const char *flerp_state_fingerprint(const flerp_executor *ex) { return ex->fingerprint; }
const char *flerp_last_error(const flerp_executor *ex) { return ex ? ex->error : g_open_error; }
int flerp_failed(const flerp_executor *ex) { (void)ex; return 0; }

static seq_t *seq_of(flerp_executor *ex, int32_t seq) {
    return seq >= 0 && seq < ex->n_seqs && ex->seqs[seq].open ? &ex->seqs[seq] : NULL;
}

/* --- snapshots (features=state) ------------------------------------------------------------------------------ */
static flerp_snapshot handle_of(const flerp_executor *ex, int i) { return ((uint64_t)ex->snaps[i].gen << 32) | (uint64_t)(i + 1); }

static snap_t *snap_of(flerp_executor *ex, flerp_snapshot h) {
    const uint64_t i = (h & 0xffffffffu) - 1;
    if (h == 0 || i >= MAX_SNAPS || !ex->snaps[i].used || ex->snaps[i].gen != (uint32_t)(h >> 32)) return NULL;
    return &ex->snaps[i];
}

static void snap_free(snap_t *s) {
    free(s->ids);
    s->ids = NULL, s->used = 0;
}

static void drop_owned(flerp_executor *ex, int owner) {
    for (int i = 0; i < MAX_SNAPS; ++i)
        if (ex->snaps[i].used && ex->snaps[i].owner == owner) snap_free(&ex->snaps[i]);
}

/* A new snapshot of `owner` holding the first `pos` ids of `ids` (pos -1: a mark at `at`, ids taken later). */
static flerp_status snap_new(flerp_executor *ex, int owner, const flerp_token *ids, int64_t pos, int64_t at, flerp_snapshot *out) {
    int held = 0, slot = -1;
    for (int i = 0; i < MAX_SNAPS; ++i) {
        if (ex->snaps[i].used && ex->snaps[i].owner == owner && owner >= 0) ++held;
        if (!ex->snaps[i].used && slot < 0) slot = i;
    }
    if (slot < 0 || (owner >= 0 && held >= 8)) return fail(ex, FLERP_E_FULL, "fake: no room for another snapshot");
    snap_t *s = &ex->snaps[slot];
    s->ids = calloc((size_t)ex->positions, sizeof(flerp_token));
    if (pos > 0) memcpy(s->ids, ids, (size_t)pos * sizeof(flerp_token));
    s->used = 1, s->owner = owner, s->pos = pos, s->at = at, s->gen = ++ex->gen;
    *out = handle_of(ex, slot);
    return FLERP_OK;
}

static void set_state(seq_t *q, const flerp_token *ids, int64_t pos) {
    memcpy(q->ids, ids, (size_t)pos * sizeof(flerp_token));
    q->pos = pos, q->prompt = -1, q->n_held = 0, q->ended = 0;
}

flerp_status flerp_seq_new(flerp_executor *ex, const flerp_seq_args *args, int32_t *seq) {
    for (int i = 0; i < ex->n_seqs; ++i) {
        seq_t *q = &ex->seqs[i];
        if (q->open) continue;
        flerp_token *ids = q->ids;
        memset(q, 0, sizeof(*q));
        q->ids = ids, q->prompt = -1;
        if (args != NULL && args->sampling != NULL && args->sampling->temperature > 0) {
            const flerp_sampling *s = args->sampling;
            if (!ex->sampled) return fail(ex, FLERP_E_UNSUPPORTED, "fake: this executor does not draw sampled tokens (sampled=on)");
            if (s->top_k == 0 && ex->needs_top_k) return fail(ex, FLERP_E_ARGUMENT, "fake: a sampled seq needs top_k");
            if (ex->top_k_max > 0 && s->top_k > ex->top_k_max) return fail(ex, FLERP_E_ARGUMENT, "fake: top_k above top_k_max");
            if (s->top_p < 1 || s->min_p > 0) return fail(ex, FLERP_E_ARGUMENT, "fake: top_p and min_p are not taken");
            q->sampled = 1, q->seed = s->seed, q->top_k = s->top_k;
        }
        /* version 3 fields: read only from a version 3 executor and as far as the caller's struct goes */
        const size_t size = args != NULL && ex->abi >= 3 ? args->size : 0;
        if (size >= FLERP_HAS(flerp_seq_args, constraint) && args->constraint != NULL) {
            const flerp_constraint *c = args->constraint;
            if (!(ex->features & F_CONSTRAINT)) return fail(ex, FLERP_E_UNSUPPORTED, "fake: no constraints (features=constraint)");
            if (c->size < FLERP_HAS(flerp_constraint, n_text) || c->kind == 0 || (c->kind & (c->kind - 1)) || c->kind > FLERP_CONSTRAINT_TOOL_CALL_REQUIRED ||
                c->text == NULL)
                return fail(ex, FLERP_E_ARGUMENT, "fake: a constraint is one FLERP_CONSTRAINT_* kind with its text");
            q->constrained = 1, q->parity = (uint32_t)(hash_bytes(c->text, c->n_text) ^ c->kind) & 1u;
        }
        if (size >= FLERP_HAS(flerp_seq_args, loop_guard) &&
            (args->n_eos > 0 || args->n_think_close > 0 || args->loop_guard != 0)) {
            if (!(ex->features & F_POLICY)) return fail(ex, FLERP_E_UNSUPPORTED, "fake: no stop policy (features=seq-policy)");
            if (args->n_eos > MAX_POLICY || args->n_think_close > MAX_POLICY || (args->n_eos && !args->eos) ||
                (args->n_think_close && !args->think_close) || (args->n_think_close && args->think_budget < 0))
                return fail(ex, FLERP_E_ARGUMENT, "fake: eos and think_close hold at most 16 ids; think_budget >= 0");
            if (args->n_eos) memcpy(q->eos, args->eos, args->n_eos * sizeof(flerp_token));
            if (args->n_think_close) memcpy(q->close, args->think_close, args->n_think_close * sizeof(flerp_token));
            q->n_eos = args->n_eos, q->n_close = args->n_think_close, q->budget = args->think_budget, q->loop_guard = args->loop_guard;
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
    drop_owned(ex, seq);
    return FLERP_OK;
}

flerp_status flerp_seq_reset(flerp_executor *ex, int32_t seq) {
    seq_t *q = seq_of(ex, seq);
    if (!q) return fail(ex, FLERP_E_ARGUMENT, "fake: the seq is not open");
    if (q->pending) return fail(ex, FLERP_E_STATE, "fake: a verify is pending");
    q->pos = 0, q->prompt = -1, q->n_held = 0, q->image = 0, q->ended = 0;
    return FLERP_OK;
}

int64_t flerp_seq_pos(const flerp_executor *ex, int32_t seq) {
    const seq_t *q = seq_of((flerp_executor *)ex, seq);
    return q ? q->pos : FLERP_E_ARGUMENT;
}

static int is_eos(const seq_t *q, flerp_token t) {
    for (size_t i = 0; i < q->n_eos; ++i)
        if (q->eos[i] == t) return 1;
    return 0;
}

/* Refuses feeding past an end id after the prompt (stop policy). */
static flerp_status check_feed(flerp_executor *ex, seq_t *q, const flerp_token *ids, size_t n) {
    if (q->ended) return fail(ex, FLERP_E_STATE, "fake: the reply ended (an eos id was fed); reset the seq");
    if (q->prompt >= 0)
        for (size_t i = 0; i < n; ++i)
            if (is_eos(q, ids[i])) {
                q->ended = 1;
                return fail(ex, FLERP_E_STATE, "fake: an eos id was fed after the prompt: the reply ended");
            }
    return FLERP_OK;
}

static uint64_t span_hash(const flerp_executor *ex, const flerp_image_span *s) {
    return s->rows != NULL ? hash_bytes(s->rows, (size_t)s->tokens * IMAGE_HIDDEN * sizeof(float)) : hash_bytes(s->payload, s->payload_bytes) ^ (uint64_t)ex->vocab;
}

static void prepare(const void *payload, size_t n, flerp_image_span *out) {
    const int32_t w = 2 + (int32_t)(hash_bytes(payload, n) % 7);
    out->grid_t = 1, out->grid_h = 2, out->grid_w = w, out->tokens = 2 * w;
}

flerp_status flerp_forward_spans(flerp_executor *ex, int32_t seq, const flerp_token *ids, size_t n, const flerp_image_span *spans,
                                 size_t n_spans, const flerp_want *want, flerp_rows *out, const flerp_token *lookahead, size_t n_lookahead) {
    seq_t *q = seq_of(ex, seq);
    (void)lookahead, (void)n_lookahead;
    if (n_spans > 0) NEED(FLERP_CAP_IMAGE_ROWS | FLERP_CAP_IMAGE_PREPARE, "flerp_forward_spans");
    if (!q) return fail(ex, FLERP_E_ARGUMENT, "fake: the seq is not open");
    if (q->pending) return fail(ex, FLERP_E_STATE, "fake: a verify is pending: keep, keep_prefix or drop first");
    if (ids == NULL || n == 0) return fail(ex, FLERP_E_ARGUMENT, "flerp_forward: no ids");
    const int64_t rows = want ? want->rows : 0;
    if (rows < 0 || rows > 1) return fail(ex, FLERP_E_UNSUPPORTED, "fake: a forward returns its last row only");
    if (n > 512) return fail(ex, FLERP_E_FULL, "flerp_forward: more ids than caps.max_forward");
    if (q->pos + (int64_t)n > ex->positions) return fail(ex, FLERP_E_FULL, "fake: the seq is out of positions");
    for (size_t i = 0; i < n; ++i)
        if (ids[i] < 0 || ids[i] >= ex->vocab) return fail(ex, FLERP_E_ARGUMENT, "fake: an id outside the vocabulary");
    if (n_spans > IMAGES) return fail(ex, FLERP_E_FULL, "fake: more image spans than caps.images");
    int64_t prev_end = q->pos;
    for (size_t k = 0; k < n_spans; ++k) {
        const flerp_image_span *s = &spans[k];
        if (s->size < FLERP_HAS(flerp_image_span, payload_bytes) || s->tokens < 1 || s->tokens > IMAGE_MAX_ROWS || s->at < prev_end ||
            s->at + s->tokens > q->pos + (int64_t)n)
            return fail(ex, FLERP_E_ARGUMENT, "fake: a span outside the forward, overlapping, or of no rows");
        for (int64_t p = s->at; p < s->at + s->tokens; ++p)
            if (ids[p - q->pos] != ex->image_token) return fail(ex, FLERP_E_ARGUMENT, "fake: a span's ids are not caps.image_token");
        if (s->rows != NULL) {
            if (!(ex->features & F_IMAGE_ROWS)) return fail(ex, FLERP_E_UNSUPPORTED, "fake: no finished rows (features=image-rows)");
        } else if (s->payload != NULL) {
            if (!(ex->features & F_IMAGE_PREPARE)) return fail(ex, FLERP_E_UNSUPPORTED, "fake: no tower of its own (features=image-prepare)");
            flerp_image_span p;
            prepare(s->payload, s->payload_bytes, &p);
            if (p.tokens != s->tokens) return fail(ex, FLERP_E_ARGUMENT, "fake: the span's tokens are not flerp_image_prepare's");
        } else return fail(ex, FLERP_E_ARGUMENT, "fake: a span with neither rows nor a payload");
        prev_end = s->at + s->tokens;
    }
    flerp_status st = check_feed(ex, q, ids, n);
    if (st != FLERP_OK) return st;
    memcpy(q->ids + q->pos, ids, n * sizeof(flerp_token));
    for (size_t k = 0; k < n_spans; ++k) {
        const uint64_t h = span_hash(ex, &spans[k]);
        for (int64_t j = 0; j < spans[k].tokens; ++j)
            q->ids[spans[k].at + j] = (flerp_token)(ex->lo + mix(h ^ (uint64_t)j) % (uint64_t)(ex->hi - ex->lo));
        q->image = 1;
    }
    const int64_t from = q->pos;
    q->pos += (int64_t)n, q->n_held = 0;
    ++ex->stats.forwards, ex->stats.forward_tokens += (int64_t)n;
    for (int i = 0; i < MAX_SNAPS; ++i) { /* marks this forward reached; marks left behind when a result is asked are dropped */
        snap_t *s = &ex->snaps[i];
        if (!s->used || s->owner != seq || s->pos >= 0) continue;
        if (s->at > from && s->at <= q->pos) {
            memcpy(s->ids, q->ids, (size_t)s->at * sizeof(flerp_token));
            s->pos = s->at, ++ex->stats.snapshots_saved;
        } else if (rows > 0) snap_free(s);
    }
    if (rows == 0) return FLERP_OK;
    if (q->prompt < 0) q->prompt = q->pos; /* the reply starts where the first result is asked */
    return deliver(ex, q, q->ids[q->pos - 1], q->pos, want, want->masks, out, 0);
}

flerp_status flerp_forward(flerp_executor *ex, int32_t seq, const flerp_token *ids, size_t n, const flerp_want *want, flerp_rows *out,
                           const flerp_token *lookahead, size_t n_lookahead) {
    return flerp_forward_spans(ex, seq, ids, n, NULL, 0, want, out, lookahead, n_lookahead);
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
    if (ex->hold_cap > 0 && depth > ex->hold_cap) depth = ex->hold_cap; /* its own depth rule: fewer than asked */
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
        /* run.held names what the caller asked to hold; with hold_cap the seq holds at most that many and the verify
           returns 1 + what it holds (the caller takes the round's size from the rows) */
        const int held_ok = ex->hold_cap > 0 ? r->held >= q->n_held : r->held == q->n_held;
        if (ex->draft != 2 || r->n_ids != 1 || !held_ok) return fail(ex, FLERP_E_ARGUMENT, "fake: run.held is not the drafts held for the seq");
        if (r->held_out == NULL || r->held_cap < (size_t)q->n_held) return fail(ex, FLERP_E_ARGUMENT, "flerp_verify_run.held_out is too small");
        rows = 1 + q->n_held;
    }
    if (rows > MAX_ROWS) return fail(ex, FLERP_E_FULL, "flerp_verify: more rows than caps.max_verify_rows");
    if (q->pos + rows > ex->positions) return fail(ex, FLERP_E_FULL, "fake: the seq is out of positions");
    flerp_status st = check_feed(ex, q, r->ids, 1);
    if (st != FLERP_OK) return st;
    q->v_ids[0] = r->ids[0];
    for (int64_t i = 1; i < rows; ++i) q->v_ids[i] = r->held > 0 ? q->held[i - 1] : r->ids[i];
    if (r->held > 0) memcpy(r->held_out, q->held, (size_t)q->n_held * sizeof(flerp_token));
    q->pending = 1, q->v_rows = rows;
    ++ex->stats.verifies, ex->stats.verify_rows += rows;
    for (int64_t i = 0; i < rows; ++i) {
        st = deliver(ex, q, q->v_ids[i], q->pos + i + 1, want, r->masks, r->out, i);
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
    if (rows > 0) memcpy(q->ids + q->pos, q->v_ids, (size_t)rows * sizeof(flerp_token)), q->pos += rows;
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

/* --- snapshots: save / restore / drop / pos with features=state; host bytes and files stay unsupported --------- */
#define NEED_STATE(name)                                                                                                    \
    do {                                                                                                                    \
        if (!(ex->features & F_STATE)) return fail(ex, FLERP_E_UNSUPPORTED, "fake: no snapshots (features=state): " name); \
    } while (0)

flerp_status flerp_snapshot_save(flerp_executor *ex, int32_t seq, flerp_snapshot *out) {
    NEED_STATE("flerp_snapshot_save");
    seq_t *q = seq_of(ex, seq);
    if (!q || out == NULL) return fail(ex, FLERP_E_ARGUMENT, "fake: the seq is not open");
    if (q->pending) return fail(ex, FLERP_E_STATE, "fake: a verify is pending");
    if (ex->keep_step > 0 && q->pos % ex->keep_step != 0) return fail(ex, FLERP_E_STATE, "fake: a save only at a multiple of caps.keep_step");
    const flerp_status st = snap_new(ex, seq, q->ids, q->pos, q->pos, out);
    if (st == FLERP_OK) ++ex->stats.snapshots_saved;
    return st;
}

flerp_status flerp_snapshot_restore(flerp_executor *ex, int32_t seq, flerp_snapshot snap) {
    NEED_STATE("flerp_snapshot_restore");
    seq_t *q = seq_of(ex, seq);
    snap_t *s = snap_of(ex, snap);
    if (!q || !s || s->owner == -2 || (s->owner >= 0 && s->owner != seq)) return fail(ex, FLERP_E_ARGUMENT, "fake: not a snapshot of this seq");
    if (q->pending || s->pos < 0) return fail(ex, FLERP_E_STATE, "fake: a verify is pending, or the mark was not reached");
    set_state(q, s->ids, s->pos);
    ++ex->stats.snapshots_restored;
    return FLERP_OK;
}

flerp_status flerp_snapshot_drop(flerp_executor *ex, int32_t seq, flerp_snapshot snap) {
    NEED_STATE("flerp_snapshot_drop");
    snap_t *s = snap_of(ex, snap);
    if (!s || s->owner == -2 || (s->owner == -1 ? seq != -1 : s->owner != seq))
        return fail(ex, FLERP_E_ARGUMENT, "fake: not a snapshot of this seq (a shared one is dropped with seq -1)");
    snap_free(s);
    return FLERP_OK;
}

int64_t flerp_snapshot_pos(const flerp_executor *ex, int32_t seq, flerp_snapshot snap) {
    if (!(ex->features & F_STATE)) return FLERP_E_UNSUPPORTED;
    const snap_t *s = snap_of((flerp_executor *)ex, snap);
    if (!s || s->owner == -2 || (s->owner >= 0 && s->owner != seq)) return FLERP_E_ARGUMENT;
    return s->pos < 0 ? FLERP_E_STATE : s->pos;
}

#define NO_BYTES return fail(ex, FLERP_E_UNSUPPORTED, "fake: no host bytes or files (version 3: flerp_state_export2)")
int64_t flerp_snapshot_bytes(const flerp_executor *ex, int32_t seq, flerp_snapshot snap, int64_t from) { (void)ex, (void)seq, (void)snap, (void)from; return FLERP_E_UNSUPPORTED; }
flerp_status flerp_snapshot_export(flerp_executor *ex, int32_t seq, flerp_snapshot snap, int64_t from, void *bytes, size_t cap, size_t *written) {
    (void)seq, (void)snap, (void)from, (void)bytes, (void)cap, (void)written; NO_BYTES;
}
flerp_status flerp_state_import(flerp_executor *ex, int32_t seq, const void *bytes, size_t n_bytes, int64_t n, int64_t from) {
    (void)seq, (void)bytes, (void)n_bytes, (void)n, (void)from; NO_BYTES;
}
flerp_status flerp_snapshot_write(flerp_executor *ex, int32_t seq, flerp_snapshot snap, const char *dir, uint64_t key) { (void)seq, (void)snap, (void)dir, (void)key; NO_BYTES; }
flerp_status flerp_snapshot_read(flerp_executor *ex, int32_t seq, const char *dir, uint64_t key, int64_t at, flerp_snapshot *out) {
    (void)seq, (void)dir, (void)key, (void)at, (void)out; NO_BYTES;
}

/* --- version 3: state with its prefix ids ---------------------------------------------------------------------- */
static void put64(unsigned char *b, int64_t v) {
    for (int i = 0; i < 8; ++i) b[i] = (unsigned char)((uint64_t)v >> (8 * i));
}
static int64_t get64(const unsigned char *b) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= (uint64_t)b[i] << (8 * i);
    return (int64_t)v;
}

flerp_status flerp_state_export2(flerp_executor *ex, int32_t seq, flerp_snapshot snap, int64_t from, flerp_state_desc *desc,
                                 flerp_sink sink, void *ctx) {
    NEED(FLERP_CAP_STATE_PREFIX, "flerp_state_export2");
    snap_t *s = snap_of(ex, snap);
    if (!s || s->owner == -2 || (s->owner >= 0 && s->owner != seq) || desc == NULL || sink == NULL ||
        desc->size < FLERP_HAS(flerp_state_desc, bytes))
        return fail(ex, FLERP_E_ARGUMENT, "fake: not a snapshot of this seq, or no desc or sink");
    if (s->pos < 0) return fail(ex, FLERP_E_STATE, "fake: the mark was not reached");
    if (from < 0 || from > s->pos || (ex->keep_step > 0 && from % ex->keep_step != 0))
        return fail(ex, FLERP_E_ARGUMENT, "fake: from outside [0, the snapshot's position] or off caps.keep_step");
    desc->tokens = s->pos, desc->from = from, desc->prefix_ids = s->ids, desc->n_prefix = (size_t)s->pos;
    desc->fingerprint = ex->fingerprint, desc->bytes = 32 + 4 * (s->pos - from);
    unsigned char head[32], chunk[4096];
    memcpy(head, "FAKESTA3", 8), put64(head + 8, s->pos), put64(head + 16, from), put64(head + 24, (int64_t)hash_bytes(ex->fingerprint, strlen(ex->fingerprint)));
    if (sink(ctx, head, sizeof(head)) != 0) return fail(ex, FLERP_E_ARGUMENT, "fake: the sink stopped the export");
    size_t fill = 0;
    for (int64_t p = from; p < s->pos; ++p) {
        const uint32_t v = (uint32_t)s->ids[p];
        for (int i = 0; i < 4; ++i) chunk[fill++] = (unsigned char)(v >> (8 * i));
        if (fill == sizeof(chunk) || p + 1 == s->pos) {
            if (sink(ctx, chunk, fill) != 0) return fail(ex, FLERP_E_ARGUMENT, "fake: the sink stopped the export");
            fill = 0;
        }
    }
    return FLERP_OK;
}

flerp_status flerp_state_import2(flerp_executor *ex, int32_t seq, const flerp_state_desc *desc, flerp_source source, void *ctx) {
    NEED(FLERP_CAP_STATE_PREFIX, "flerp_state_import2");
    seq_t *q = seq_of(ex, seq);
    if (!q || desc == NULL || source == NULL || desc->size < FLERP_HAS(flerp_state_desc, bytes))
        return fail(ex, FLERP_E_ARGUMENT, "fake: the seq is not open, or no desc or source");
    if (q->pending) return fail(ex, FLERP_E_STATE, "fake: a verify is pending");
    if (desc->tokens < 0 || desc->tokens > ex->positions || desc->from < 0 || desc->from > desc->tokens)
        return fail(ex, FLERP_E_ARGUMENT, "fake: tokens or from out of range");
    if (desc->prefix_ids == NULL || desc->n_prefix != (size_t)desc->tokens)
        return fail(ex, FLERP_E_STATE, "fake: a state resumes only with its prefix ids (n_prefix == tokens)");
    if (desc->fingerprint == NULL || strcmp(desc->fingerprint, ex->fingerprint) != 0)
        return fail(ex, FLERP_E_STATE, "fake: the state's fingerprint is not this executor's");
    if (q->pos != desc->from) return fail(ex, FLERP_E_STATE, "fake: the seq is not at the state's from (0 for a whole state)");
    for (int64_t p = 0; p < desc->from; ++p)
        if (q->ids[p] != desc->prefix_ids[p]) return fail(ex, FLERP_E_STATE, "fake: a delta on a seq that holds other ids");
    const size_t want = 32 + 4 * (size_t)(desc->tokens - desc->from);
    unsigned char *buf = malloc(want + 1);
    size_t got = 0;
    for (;;) {
        const int64_t r = source(ctx, buf + got, want + 1 - got);
        if (r < 0) return free(buf), fail(ex, FLERP_E_ARGUMENT, "fake: the source failed");
        if (r == 0) break;
        got += (size_t)r;
        if (got > want) return free(buf), fail(ex, FLERP_E_STATE, "fake: more bytes than the state holds");
    }
    flerp_status st = FLERP_OK;
    if (got != want || memcmp(buf, "FAKESTA3", 8) != 0 || get64(buf + 8) != desc->tokens || get64(buf + 16) != desc->from)
        st = fail(ex, FLERP_E_STATE, "fake: the bytes are not a state of these positions");
    for (int64_t p = desc->from; st == FLERP_OK && p < desc->tokens; ++p) {
        const unsigned char *b = buf + 32 + 4 * (p - desc->from);
        if ((flerp_token)((uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24) != desc->prefix_ids[p])
            st = fail(ex, FLERP_E_STATE, "fake: the bytes are not the prefix ids' state");
    }
    free(buf);
    if (st != FLERP_OK) return st;
    drop_owned(ex, seq);
    set_state(q, desc->prefix_ids, desc->tokens);
    return FLERP_OK;
}

flerp_status flerp_snapshot_mark(flerp_executor *ex, int32_t seq, int64_t at, flerp_snapshot *out) {
    NEED(FLERP_CAP_SNAPSHOT_MARK, "flerp_snapshot_mark");
    seq_t *q = seq_of(ex, seq);
    if (!q || out == NULL) return fail(ex, FLERP_E_ARGUMENT, "fake: the seq is not open");
    if (at <= q->pos || at > ex->positions || (ex->keep_step > 0 && at % ex->keep_step != 0))
        return fail(ex, FLERP_E_ARGUMENT, "fake: a mark past the seq's position, at a multiple of caps.keep_step");
    return snap_new(ex, seq, NULL, -1, at, out);
}

flerp_status flerp_snapshot_share(flerp_executor *ex, int32_t seq, flerp_snapshot snap, flerp_snapshot *out) {
    NEED(FLERP_CAP_SNAPSHOT_SHARED, "flerp_snapshot_share");
    snap_t *s = snap_of(ex, snap);
    if (!s || s->owner != seq || out == NULL) return fail(ex, FLERP_E_ARGUMENT, "fake: not a snapshot of this seq");
    if (s->pos < 0) return fail(ex, FLERP_E_STATE, "fake: the mark was not reached");
    return snap_new(ex, -1, s->ids, s->pos, s->pos, out);
}

flerp_status flerp_seq_park(flerp_executor *ex, int32_t seq, flerp_snapshot *out) {
    NEED(FLERP_CAP_STATE_PARK, "flerp_seq_park");
    seq_t *q = seq_of(ex, seq);
    if (!q || out == NULL) return fail(ex, FLERP_E_ARGUMENT, "fake: the seq is not open");
    if (q->pending) return fail(ex, FLERP_E_STATE, "fake: a verify is pending");
    const flerp_status st = snap_new(ex, -2, q->ids, q->pos, q->pos, out);
    if (st != FLERP_OK) return st;
    q->pos = 0, q->prompt = -1, q->n_held = 0, q->image = 0, q->ended = 0;
    return FLERP_OK;
}

flerp_status flerp_seq_resume(flerp_executor *ex, int32_t seq, flerp_snapshot parked) {
    NEED(FLERP_CAP_STATE_PARK, "flerp_seq_resume");
    seq_t *q = seq_of(ex, seq);
    snap_t *s = snap_of(ex, parked);
    if (!q || !s || s->owner != -2) return fail(ex, FLERP_E_ARGUMENT, "fake: not a parked state");
    if (q->pending || q->pos != 0) return fail(ex, FLERP_E_STATE, "fake: resume into a reset seq");
    set_state(q, s->ids, s->pos);
    snap_free(s);
    return FLERP_OK;
}

/* --- version 3: steering --------------------------------------------------------------------------------------- */
flerp_status flerp_steer_load(flerp_executor *ex, const char *name, int32_t layer, const float *v, int32_t dim, uint32_t *out) {
    NEED(FLERP_CAP_STEER, "flerp_steer_load");
    (void)name;
    if (v == NULL || out == NULL || dim != STEER_DIM || layer < 0 || layer >= STEER_LAYERS)
        return fail(ex, FLERP_E_ARGUMENT, "fake: a vector is caps.steer_dim floats at a layer below caps.steer_layers");
    for (int i = 0; i < MAX_VECTORS; ++i)
        if (!ex->vectors[i].used) {
            ex->vectors[i].used = 1, ex->vectors[i].salt = mix(hash_bytes(v, (size_t)dim * sizeof(float)) ^ (uint64_t)layer);
            *out = (uint32_t)(i + 1);
            return FLERP_OK;
        }
    return fail(ex, FLERP_E_FULL, "fake: no room for another vector");
}

flerp_status flerp_steer_drop(flerp_executor *ex, uint32_t vector) {
    NEED(FLERP_CAP_STEER, "flerp_steer_drop");
    if (vector == 0 || vector > MAX_VECTORS || !ex->vectors[vector - 1].used) return fail(ex, FLERP_E_ARGUMENT, "fake: not a loaded vector");
    for (int s = 0; s < ex->n_seqs; ++s)
        for (size_t i = 0; ex->seqs[s].open && i < ex->seqs[s].n_items; ++i)
            if (ex->seqs[s].items[i].vector == vector) return fail(ex, FLERP_E_STATE, "fake: a seq steers with the vector");
    ex->vectors[vector - 1].used = 0;
    return FLERP_OK;
}

flerp_status flerp_seq_steer(flerp_executor *ex, int32_t seq, const flerp_steer_item *items, size_t n) {
    NEED(FLERP_CAP_STEER, "flerp_seq_steer");
    seq_t *q = seq_of(ex, seq);
    if (!q || (n > 0 && items == NULL) || n > MAX_ITEMS) return fail(ex, FLERP_E_ARGUMENT, "fake: the seq is not open, or more than 8 items");
    if (q->pending) return fail(ex, FLERP_E_STATE, "fake: a verify is pending");
    for (size_t i = 0; i < n; ++i) {
        const flerp_steer_item *it = &items[i];
        if (it->size < FLERP_HAS(flerp_steer_item, to) || it->vector == 0 || it->vector > MAX_VECTORS || !ex->vectors[it->vector - 1].used)
            return fail(ex, FLERP_E_ARGUMENT, "fake: an item names no loaded vector");
        if (!(it->from == 0 && it->to == -1) && (it->from < 0 || (it->to >= 0 && it->to <= it->from)))
            return fail(ex, FLERP_E_ARGUMENT, "fake: an item's span is not [from, to)");
    }
    for (size_t i = 0; i < n; ++i) q->items[i] = (item_t){items[i].vector, items[i].scale, items[i].from, items[i].to};
    q->n_items = n;
    return FLERP_OK;
}

/* --- version 3: images ----------------------------------------------------------------------------------------- */
flerp_status flerp_image_prepare(flerp_executor *ex, const void *payload, size_t n, flerp_image_span *out) {
    NEED(FLERP_CAP_IMAGE_PREPARE, "flerp_image_prepare");
    if (payload == NULL || n == 0 || out == NULL || out->size < FLERP_HAS(flerp_image_span, payload_bytes))
        return fail(ex, FLERP_E_ARGUMENT, "fake: no payload or a short span");
    prepare(payload, n, out);
    return FLERP_OK;
}

/* --- version 3: slots ------------------------------------------------------------------------------------------ */
int flerp_seq_batchable(const flerp_executor *ex, int32_t seq) {
    if (ex->abi < 3 || !(caps_flags(ex) & FLERP_CAP_SEQ_BATCHABLE)) return FLERP_E_UNSUPPORTED;
    const seq_t *q = seq_of((flerp_executor *)ex, seq);
    return q ? !q->image : FLERP_E_ARGUMENT;
}

flerp_status flerp_prefetch(flerp_executor *ex, int32_t seq, const flerp_token *ids, size_t n) {
    NEED(FLERP_CAP_PREFETCH, "flerp_prefetch");
    if (!seq_of(ex, seq) || (n > 0 && ids == NULL)) return fail(ex, FLERP_E_ARGUMENT, "fake: the seq is not open");
    for (size_t i = 0; i < n; ++i)
        if (ids[i] < 0 || ids[i] >= ex->vocab) return fail(ex, FLERP_E_ARGUMENT, "fake: an id outside the vocabulary");
    ex->prefetched += (int64_t)n;
    return FLERP_OK;
}

int flerp_seq_can_renew(const flerp_executor *ex, int32_t seq) {
    if (ex->abi < 3 || !(caps_flags(ex) & FLERP_CAP_RENEW)) return FLERP_E_UNSUPPORTED;
    const seq_t *q = seq_of((flerp_executor *)ex, seq);
    return q ? !q->pending : FLERP_E_ARGUMENT;
}

flerp_status flerp_seq_renew(flerp_executor *ex, int32_t seq) {
    NEED(FLERP_CAP_RENEW, "flerp_seq_renew");
    const int can = flerp_seq_can_renew(ex, seq);
    if (can < 0) return fail(ex, FLERP_E_ARGUMENT, "fake: the seq is not open");
    if (!can) return fail(ex, FLERP_E_STATE, "fake: a verify is pending");
    ++ex->renews; /* the position and ids are kept, so the tokens after a renew are the tokens without one */
    return FLERP_OK;
}

flerp_status flerp_get_stats(const flerp_executor *ex, flerp_stats *out) {
    if (out == NULL || out->size < sizeof(flerp_stats)) return fail(ex, FLERP_E_ARGUMENT, "flerp_stats is null or short");
    const uint32_t size = out->size;
    *out = ex->stats;
    out->size = size, out->device_bytes = 0;
    return FLERP_OK;
}

size_t flerp_stats_json(const flerp_executor *ex, char *out, size_t cap) {
    char buf[128];
    int n = 2;
    if (ex->features & F_SLOTS)
        n = snprintf(buf, sizeof(buf), "{\"prefetch_ids\": %lld, \"renews\": %lld}", (long long)ex->prefetched, (long long)ex->renews);
    else
        memcpy(buf, "{}", 3);
    if (out != NULL && cap > 0) {
        const size_t w = (size_t)n < cap - 1 ? (size_t)n : cap - 1;
        memcpy(out, buf, w), out[w] = 0;
    }
    return (size_t)n;
}
