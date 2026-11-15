/* FROZEN: include/flerp/executor.h as of version 2 (commit 825cc98), kept so tests/test_flerp_serve.py can build a
 * version 2 caller and a version 2 executor against the version 3 ones. Never edit. */
/* strixflerp executor ABI, version 2 (SPEC.md section 6).
 *
 * One C ABI that a model-family executor sits behind, written from the two real ones in the catalogue:
 *   strixite   LmBackend             catalog/strixite/runtime/engine/src/serve/engine.hpp
 *   StrixFold  lanes Backend vtable  catalog/strixfold/runtime/lanes/src/zig/src/core/lanes/backend.zig
 *              and its engine host   catalog/strixfold/runtime/lanes/src/zig/src/core/engine_api.zig
 * docs/EXECUTOR-ABI.md maps every call of both onto this file and names what does not map.
 *
 * Shape. An executor is one loaded model in one process. It holds `seqs` conversations ("seq"), each at its own
 * position. A serving loop feeds token ids into a seq and takes back a result for the last rows it asked for. The
 * two origins differ in what that result is, so the caller picks a delivery the executor offers (caps.deliveries):
 *
 *   FLERP_OUT_FULL        full logits rows (strixite; StrixFold only for sampled rows, see the mapping)
 *   FLERP_OUT_CANDIDATES  each row's top `caps.candidates` (id, logit), value descending, lower id first on a tie
 *   FLERP_OUT_TOKENS      the token the executor draws for each row with the seq's sampling at the row's position.
 *                         StrixFold draws on the device and never shows logits for greedy rows; strixite's adapter
 *                         implements it for temperature 0 only (the row's argmax, lower id first on a tie)
 *
 * Speculation. `flerp_verify` is a forward whose effect on the seqs it names is pending: before any other call on
 * a seq of the verify, exactly one of flerp_keep (all rows), flerp_keep_prefix (the first `rows`), flerp_drop.
 * Drafts come from the caller, or from the executor's own draft head in one of two forms (caps.flags):
 *   FLERP_CAP_DRAFT_STEP  flerp_draft_step: one chained draft token a call, returned to the caller (strixite MTP)
 *   FLERP_CAP_DRAFT_HELD  flerp_draft_hold: the executor drafts `depth` tokens and keeps them on the device; the
 *                         next verify of that seq names them by count (run.held) and returns them (StrixFold)
 *
 * Rules for every call: returns FLERP_OK or a negative flerp_status; on failure flerp_last_error() describes it and
 * output arguments are unspecified. Buffers are the caller's; the executor never keeps a pointer past the call
 * unless a comment says so. Not thread safe: one thread drives an executor (both origins serve from one thread).
 * Every struct starts with `size` = sizeof the struct as the caller compiled it, so fields can be appended.
 *
 * Versions. Version 2 appends to version 1 and changes nothing in it: flerp_caps.sampling and .top_k_max (what an
 * executor that draws on the device accepts in flerp_sampling, so a front end can refuse a request before it runs).
 * A struct the executor fills (flerp_caps, flerp_stats) is written for at most the caller's `size` bytes and comes
 * back with `size` = the bytes the executor knows: a version 2 caller of a version 1 executor finds the appended
 * fields zero and `size` short of them; a version 1 caller of a version 2 executor is never written past its struct.
 */
#ifndef FLERP_EXECUTOR_H
#define FLERP_EXECUTOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FLERP_EXECUTOR_ABI 2

typedef struct flerp_executor flerp_executor; /* opaque */
typedef int32_t flerp_token;
typedef uint64_t flerp_snapshot;              /* an executor's name for a saved state; 0 is never a snapshot */

typedef enum flerp_status {
    FLERP_OK = 0,
    FLERP_E_ARGUMENT = -1,     /* a bad argument: out of range, a buffer too small, a struct too short */
    FLERP_E_STATE = -2,        /* out of order: a verify is pending, nothing is pending, the seq is not open */
    FLERP_E_UNSUPPORTED = -3,  /* this executor does not have the call or the delivery (see caps) */
    FLERP_E_FULL = -4,         /* no room: positions, seqs, snapshots, rows of one forward */
    FLERP_E_FAILED = -5,       /* the backend failed; the executor may only be closed (flerp_failed() is 1) */
    FLERP_E_CONFIG = -6        /* flerp_open: a missing or bad key */
} flerp_status;

enum { /* flerp_caps.deliveries, flerp_want.delivery */
    FLERP_OUT_NONE = 0,
    FLERP_OUT_FULL = 1,
    FLERP_OUT_CANDIDATES = 2,
    FLERP_OUT_TOKENS = 4
};

enum { /* flerp_caps.flags */
    FLERP_CAP_DRAFT_STEP = 1u << 0,      /* flerp_draft_step */
    FLERP_CAP_DRAFT_HELD = 1u << 1,      /* flerp_draft_hold, flerp_verify_run.held */
    FLERP_CAP_VISION = 1u << 2,          /* the model reads images (no call in this version carries one) */
    FLERP_CAP_MASKS = 1u << 3,           /* flerp_want.masks and .bias */
    FLERP_CAP_VERIFY_BATCH = 1u << 4,    /* one flerp_verify may carry runs of several seqs */
    FLERP_CAP_KEEP_PREFIX = 1u << 5,     /* flerp_keep_prefix without running the kept rows again */
    FLERP_CAP_SNAPSHOT = 1u << 6,        /* flerp_snapshot_save / restore / drop / pos */
    FLERP_CAP_SNAPSHOT_BYTES = 1u << 7,  /* flerp_snapshot_export / flerp_state_import (host bytes, deltas) */
    FLERP_CAP_SNAPSHOT_FILES = 1u << 8,  /* flerp_snapshot_write / read (the executor's own files under a dir) */
    FLERP_CAP_SNAPSHOT_ANYWHERE = 1u << 9, /* a save may be taken at any position; without it only where the
                                             executor says (StrixFold: at prompt chunk ends, before any decode) */
    FLERP_CAP_SAMPLED_TOKENS = 1u << 10, /* FLERP_OUT_TOKENS with temperature > 0 (keyed sampler) */
    FLERP_CAP_FORWARD_ROWS = 1u << 11    /* a forward may return more than its last row (want.rows > 1) */
};

enum { /* flerp_caps.sampling (version 2): what FLERP_OUT_TOKENS draws with temperature > 0 accept */
    FLERP_SAMPLE_TOP_K = 1u << 0,        /* flerp_sampling.top_k > 0 */
    FLERP_SAMPLE_TOP_P = 1u << 1,        /* flerp_sampling.top_p < 1 */
    FLERP_SAMPLE_MIN_P = 1u << 2,        /* flerp_sampling.min_p > 0 */
    FLERP_SAMPLE_SEED = 1u << 3,         /* the draws are a function of flerp_sampling.seed */
    FLERP_SAMPLE_NEEDS_TOP_K = 1u << 4   /* top_k = 0 (the whole vocabulary) is refused: a sampled seq sets top_k */
};

typedef struct flerp_caps {
    uint32_t size;
    uint32_t abi;               /* FLERP_EXECUTOR_ABI of the executor */
    int64_t positions;          /* positions one seq may hold */
    int64_t max_forward;        /* token ids of one flerp_forward */
    int64_t max_verify_rows;    /* rows of one flerp_verify, all runs together */
    int64_t logits_row;         /* floats of one full logits row (>= the tokenizer's size) */
    int32_t candidates;         /* candidates a row with FLERP_OUT_CANDIDATES (0: none) */
    int32_t seqs;               /* conversations held at once */
    int32_t snapshots;          /* snapshots one seq may hold at once (0: none) */
    int32_t max_draft;          /* drafts of one chain or hold (0: no draft head) */
    uint32_t deliveries;        /* FLERP_OUT_* offered */
    uint32_t flags;             /* FLERP_CAP_* */
    /* version 2 (present when the returned `size` covers them) */
    uint32_t sampling;          /* FLERP_SAMPLE_*; 0 without FLERP_CAP_SAMPLED_TOKENS */
    int32_t top_k_max;          /* the largest flerp_sampling.top_k of a sampled seq (0: no bound) */
} flerp_caps;
/* The bytes of a struct up to and including `field`: `caps.size >= FLERP_HAS(flerp_caps, top_k_max)`. */
#define FLERP_HAS(type, field) (offsetof(type, field) + sizeof(((type *)0)->field))

typedef struct flerp_kv {
    const char *key, *value;
} flerp_kv;

/* What an executor loads. `recipe` is the recipe name; `config` the deployment's facts as strings. Keys are the
 * executor's own (docs/EXECUTOR-ABI.md lists them per adapter); an unknown key is refused, never ignored. */
typedef struct flerp_open_args {
    uint32_t size;
    const char *recipe;
    const flerp_kv *config;
    size_t n_config;
} flerp_open_args;

typedef struct flerp_sampling { /* how FLERP_OUT_TOKENS draws; temperature 0 = greedy, the rest unread */
    uint32_t size;
    double temperature, top_p, min_p;
    int32_t top_k;
    uint64_t seed;
} flerp_sampling;

typedef struct flerp_seq_args {
    uint32_t size;
    const flerp_sampling *sampling; /* null: greedy */
    int64_t max_new;                /* reply tokens the seq may take after its prompt (0: no limit asked).
                                       StrixFold sizes the seq's extent of the pool from it */
    int32_t drafts;                 /* 0: the seq never uses the executor's draft head */
} flerp_seq_args;

typedef struct flerp_masks { /* per returned row: bit (id % 32) of word (id / 32) set = id allowed */
    uint32_t size;
    int64_t rows, words;
    const uint32_t *bits;        /* [rows, words] */
    const flerp_token *bias_id;  /* a logit bias added to every row first: ascending unique ids ... */
    const float *bias;           /* ... and their values */
    size_t n_bias;
} flerp_masks;

typedef struct flerp_want {
    uint32_t size;
    uint32_t delivery;           /* one FLERP_OUT_* value; FLERP_OUT_NONE with rows = 0 */
    int64_t rows;                /* results for the last `rows` rows of the forward (0: none) */
    int64_t n_valid;             /* candidates and tokens come from ids [0, n_valid) (0: the whole row) */
    const flerp_masks *masks;    /* null: none; masks->rows == rows */
} flerp_want;

/* Where results go: the caller's buffers and their capacities. Only the buffers of the asked delivery are read.
 * On return `rows` is the number of rows written. */
typedef struct flerp_rows {
    uint32_t size;
    int64_t rows;
    float *full;            size_t full_cap;      /* [rows, caps.logits_row] */
    flerp_token *cand_id;   float *cand_logit;    size_t cand_cap; /* [rows, caps.candidates] each */
    flerp_token *tokens;    size_t tokens_cap;    /* [rows] */
    uint8_t *nan;           size_t nan_cap;       /* [rows], optional: 1 when a row held a NaN */
} flerp_rows;

typedef struct flerp_verify_run {
    uint32_t size;
    int32_t seq;
    const flerp_token *ids;  /* the token before the drafts (the seq's pending token), then the caller's drafts */
    size_t n_ids;
    int32_t held;            /* FLERP_CAP_DRAFT_HELD: drafts the executor holds for the seq go after ids[0]
                                (then n_ids must be 1); their values come back in held_out */
    flerp_token *held_out;   size_t held_cap;
    const flerp_masks *masks; /* null, or one mask per row */
    flerp_rows *out;         /* every row of the run: 1 + held + (n_ids - 1) rows */
} flerp_verify_run;

typedef struct flerp_draft { /* one draft token and how sure the head is */
    flerp_token token;
    float best, second;      /* the head's best and second-best logit (strixite's Top2); NaN when not known */
    double probability;      /* the head's probability for the token; NaN when not known */
} flerp_draft;

typedef struct flerp_stats {
    uint32_t size;
    int64_t forwards, forward_tokens;   /* flerp_forward calls and the ids they ran */
    int64_t verifies, verify_rows;      /* flerp_verify calls and their rows */
    int64_t kept_rows, dropped_rows;    /* rows kept and rows undone by keep / keep_prefix / drop */
    int64_t draft_tokens;               /* tokens the draft head produced */
    int64_t snapshots_saved, snapshots_restored;
    int64_t device_bytes;               /* device memory the executor holds; -1 when it keeps no count */
} flerp_stats;

/* --- lifetime ------------------------------------------------------------------------------------------------ */
flerp_status flerp_open(const flerp_open_args *args, flerp_executor **out);
void flerp_close(flerp_executor *ex);
flerp_status flerp_get_caps(const flerp_executor *ex, flerp_caps *out);
/* A line for /health and logs, and what a saved state fits (model, weights, layout, dtype). The strings live as
 * long as the executor. */
const char *flerp_describe(const flerp_executor *ex);
const char *flerp_state_fingerprint(const flerp_executor *ex);
/* The last failure's text on this executor (ex null: the last flerp_open's). Valid until the next call. */
const char *flerp_last_error(const flerp_executor *ex);
int flerp_failed(const flerp_executor *ex);

/* --- conversations ------------------------------------------------------------------------------------------- */
flerp_status flerp_seq_new(flerp_executor *ex, const flerp_seq_args *args, int32_t *seq);
flerp_status flerp_seq_free(flerp_executor *ex, int32_t seq);   /* also drops a pending verify and its snapshots */
flerp_status flerp_seq_reset(flerp_executor *ex, int32_t seq);  /* back to position 0, same settings */
int64_t flerp_seq_pos(const flerp_executor *ex, int32_t seq);   /* tokens the seq holds; negative: flerp_status */

/* --- forward ------------------------------------------------------------------------------------------------- */
/* Runs `ids` at the seq's position and advances it by n. want->rows results for the last rows go to `out`.
 * `lookahead` (may be null): the ids of the forward after this one, a hint only. */
flerp_status flerp_forward(flerp_executor *ex, int32_t seq, const flerp_token *ids, size_t n,
                           const flerp_want *want, flerp_rows *out,
                           const flerp_token *lookahead, size_t n_lookahead);

/* --- speculation --------------------------------------------------------------------------------------------- */
/* The draft head's token after `token` (step 0: the seq's last sampled token; step > 0: the previous step's
 * draft). Steps of one chain are consecutive calls; a chain ends at the next flerp_verify of the seq. */
flerp_status flerp_draft_step(flerp_executor *ex, int32_t seq, flerp_token token, int32_t step, flerp_draft *out);
/* Hold `depth` drafts on the device for the seq's next verify (0: none). `pending` is the seq's last drawn token,
 * not yet fed: the token the next verify starts with. */
flerp_status flerp_draft_hold(flerp_executor *ex, int32_t seq, flerp_token pending, int32_t depth);
/* One forward over every run. Each run's rows come back in run->out with the delivery of `want` (want->rows is
 * unread: a verify returns every row). Each seq of the runs is pending until keep / keep_prefix / drop. */
flerp_status flerp_verify(flerp_executor *ex, flerp_verify_run *runs, size_t n_runs, const flerp_want *want);
flerp_status flerp_keep(flerp_executor *ex, int32_t seq);
/* Keep the first `rows` rows of the seq's run (1 <= rows < the run's rows). `carry`: the token the seq's next
 * forward starts with (the corrected sample), or -1. */
flerp_status flerp_keep_prefix(flerp_executor *ex, int32_t seq, int64_t rows, flerp_token carry);
flerp_status flerp_drop(flerp_executor *ex, int32_t seq);

/* --- snapshots ----------------------------------------------------------------------------------------------- */
/* Save the seq's state at its position; restore makes the seq's state the snapshot's again (its position too).
 * A snapshot belongs to the seq that saved it and stays valid until dropped, the seq is freed, or (without
 * FLERP_CAP_SNAPSHOT_BYTES semantics to the contrary) a flerp_state_import on that seq. */
flerp_status flerp_snapshot_save(flerp_executor *ex, int32_t seq, flerp_snapshot *out);
flerp_status flerp_snapshot_restore(flerp_executor *ex, int32_t seq, flerp_snapshot snap);
flerp_status flerp_snapshot_drop(flerp_executor *ex, int32_t seq, flerp_snapshot snap);
int64_t flerp_snapshot_pos(const flerp_executor *ex, int32_t seq, flerp_snapshot snap); /* negative: flerp_status */
/* Host bytes (FLERP_CAP_SNAPSHOT_BYTES). from > 0: a delta, only what changed since position `from`.
 * flerp_snapshot_bytes: the size export will write, or negative. flerp_state_import: `bytes` become the seq's state
 * for positions [0, n) (a delta right after its base: import(base, from, 0) then import(delta, n, from)); every
 * snapshot of the seq is invalid afterwards. */
int64_t flerp_snapshot_bytes(const flerp_executor *ex, int32_t seq, flerp_snapshot snap, int64_t from);
flerp_status flerp_snapshot_export(flerp_executor *ex, int32_t seq, flerp_snapshot snap, int64_t from,
                                   void *bytes, size_t cap, size_t *written);
flerp_status flerp_state_import(flerp_executor *ex, int32_t seq, const void *bytes, size_t n_bytes,
                                int64_t n, int64_t from);
/* The executor's own files (FLERP_CAP_SNAPSHOT_FILES): a snapshot written under `dir` as `key`, and read back
 * as a snapshot of `seq` at `at` tokens. */
flerp_status flerp_snapshot_write(flerp_executor *ex, int32_t seq, flerp_snapshot snap, const char *dir, uint64_t key);
flerp_status flerp_snapshot_read(flerp_executor *ex, int32_t seq, const char *dir, uint64_t key, int64_t at,
                                 flerp_snapshot *out);

/* --- counters ------------------------------------------------------------------------------------------------ */
flerp_status flerp_get_stats(const flerp_executor *ex, flerp_stats *out);
/* The origin's own counters as one JSON object (strixite BackendStats, StrixFold drafter telemetry); writes at
 * most cap bytes with a terminating 0 and returns the length needed. */
size_t flerp_stats_json(const flerp_executor *ex, char *out, size_t cap);

#ifdef __cplusplus
}
#endif
#endif /* FLERP_EXECUTOR_H */
