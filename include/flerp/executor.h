/* strixflerp executor ABI, version 3 (SPEC.md section 6).
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
 *                         An executor with a draft depth rule of its own (qwen35moe_abi draft_value=on) may hold
 *                         fewer than `depth`; its verify then returns 1 + the held count rows (out->rows), and a
 *                         caller reads the count from there (catalog/flerp/runtime/decode_loop does)
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
 * Version 3 appends to version 2 the same way (docs/EXECUTOR-ABI.md, "Version 3"): state with its prefix ids, marked,
 * shared and parked snapshots, steering, image spans, constraints and stop policy for device-side draws, and three
 * slot calls. Each is behind a FLERP_CAP_* flag. An executor reporting abi >= 3 exports every version 3 call and
 * answers FLERP_E_UNSUPPORTED where its flag is absent; a caller calls none of them unless caps.abi >= 3 and the flag
 * is set. Structs the caller fills (flerp_seq_args) are read only as far as their `size`: an executor ignores the
 * version 3 fields of a version 2 caller, and a version 2 executor ignores them always, so a caller checks the flag.
 */
#ifndef FLERP_EXECUTOR_H
#define FLERP_EXECUTOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FLERP_EXECUTOR_ABI 3

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
    FLERP_CAP_FORWARD_ROWS = 1u << 11,   /* a forward may return more than its last row (want.rows > 1) */
    /* version 3 */
    FLERP_CAP_STATE_PREFIX = 1u << 12,   /* flerp_state_export2 / flerp_state_import2: streamed state with its prefix ids */
    FLERP_CAP_SNAPSHOT_MARK = 1u << 13,  /* flerp_snapshot_mark: a save the next prompt forward takes at a position */
    FLERP_CAP_SNAPSHOT_SHARED = 1u << 14, /* flerp_snapshot_share: a snapshot restorable into any seq */
    FLERP_CAP_STATE_PARK = 1u << 15,     /* flerp_seq_park / flerp_seq_resume */
    FLERP_CAP_STEER = 1u << 16,          /* flerp_steer_load / flerp_steer_drop / flerp_seq_steer, the whole seq */
    FLERP_CAP_STEER_SPAN = 1u << 17,     /* flerp_steer_item.from / .to: a vector over positions [from, to) */
    FLERP_CAP_IMAGE_ROWS = 1u << 18,     /* flerp_forward_spans with finished rows (flerp_image_span.rows) */
    FLERP_CAP_IMAGE_PREPARE = 1u << 19,  /* flerp_image_prepare and spans with a payload: the executor's own tower */
    FLERP_CAP_CONSTRAINT = 1u << 20,     /* flerp_seq_args.constraint for FLERP_OUT_TOKENS draws (caps.constraints) */
    FLERP_CAP_SEQ_POLICY = 1u << 21,     /* flerp_seq_args.eos / think_budget / think_close / loop_guard */
    FLERP_CAP_SEQ_BATCHABLE = 1u << 22,  /* flerp_seq_batchable */
    FLERP_CAP_PREFETCH = 1u << 23,       /* flerp_prefetch */
    FLERP_CAP_RENEW = 1u << 24           /* flerp_seq_renew / flerp_seq_can_renew */
};

enum { /* flerp_caps.constraints, flerp_constraint.kind (version 3) */
    FLERP_CONSTRAINT_JSON_SCHEMA = 1u << 0,      /* text: a JSON Schema */
    FLERP_CONSTRAINT_REGEX = 1u << 1,            /* text: a regular expression */
    FLERP_CONSTRAINT_CHOICE = 1u << 2,           /* text: the choices, each ended by a 0 byte, then an empty one */
    FLERP_CONSTRAINT_GRAMMAR = 1u << 3,          /* text: a GBNF grammar */
    FLERP_CONSTRAINT_TOOL_CALL_REQUIRED = 1u << 4 /* text: the tools' JSON array; the reply must be a call */
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
    /* version 3 */
    int64_t keep_step;          /* positions a marked or saved state must be a multiple of (0: anywhere) */
    uint32_t constraints;       /* FLERP_CONSTRAINT_* taken with FLERP_CAP_CONSTRAINT */
    int32_t steer_dim;          /* floats of one steering vector (FLERP_CAP_STEER) */
    int32_t steer_layers;       /* layers a vector may be loaded at: [0, steer_layers) */
    flerp_token image_token;    /* the id a prompt holds at each position of an image span */
    int32_t image_hidden;       /* floats of one image row (FLERP_CAP_IMAGE_ROWS) */
    int64_t image_max_rows;     /* rows of one image span */
    int32_t images;             /* image spans one forward may carry */
} flerp_caps;
/* The bytes of a struct up to and including `field`: `caps.size >= FLERP_HAS(flerp_caps, top_k_max)`. */
#define FLERP_HAS(type, field) (offsetof(type, field) + sizeof(((type *)0)->field))
#define FLERP_CAPS_V2_SIZE FLERP_HAS(flerp_caps, top_k_max)

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
    /* version 3, read only when `size` covers them */
    const struct flerp_constraint *constraint; /* null: none. FLERP_CAP_CONSTRAINT; FLERP_OUT_TOKENS draws only */
    const flerp_token *eos;         /* FLERP_CAP_SEQ_POLICY: ids that end the reply: a forward or verify that feeds */
    size_t n_eos;                   /*   one after the prompt is refused (FLERP_E_STATE) until the seq is reset */
    int32_t think_budget;           /* FLERP_CAP_SEQ_POLICY: reply draws before think_close is forced */
    const flerp_token *think_close; /*   the ids forced, one a draw, in order (none: n_think_close 0) */
    size_t n_think_close;
    int32_t loop_guard;             /* FLERP_CAP_SEQ_POLICY: the executor's own repetition guard (0: off) */
} flerp_seq_args;
#define FLERP_SEQ_ARGS_V2_SIZE FLERP_HAS(flerp_seq_args, drafts)

typedef struct flerp_constraint { /* version 3 */
    uint32_t size;
    uint32_t kind;                  /* one FLERP_CONSTRAINT_* */
    const char *text;               /* the constraint, kind's form; read during flerp_seq_new only */
    size_t n_text;
} flerp_constraint;

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

/* --- version 3: state with its prefix ids --------------------------------------------------------------------- */
/* What a state is. Export fills it (prefix_ids live until the next call on the executor); import reads it. A state
 * holds positions [from, tokens); from > 0 is a delta on a seq that holds [0, from) of the same ids. */
typedef struct flerp_state_desc {
    uint32_t size;
    int64_t tokens;                 /* positions the state covers */
    int64_t from;                   /* 0: whole; > 0: a delta */
    const flerp_token *prefix_ids;  /* the ids of positions [0, tokens): an executor that resumes a prompt pass from */
    size_t n_prefix;                /*   a state needs them (StrixFold); n_prefix == tokens */
    const char *fingerprint;        /* flerp_state_fingerprint of the executor that wrote it */
    int64_t bytes;                  /* the stream's length (export: filled before the first sink call; import: -1 unknown) */
} flerp_state_desc;
/* Streamed bytes, because a state is 0.5 to 1.3 GB. A sink returns 0 to go on, anything else to stop the export
 * (FLERP_E_FAILED is not raised: the call returns FLERP_E_ARGUMENT and the executor stays usable). A source writes at
 * most cap bytes and returns how many, 0 at the end, negative on failure. */
typedef int (*flerp_sink)(void *ctx, const void *bytes, size_t n);
typedef int64_t (*flerp_source)(void *ctx, void *bytes, size_t cap);
flerp_status flerp_state_export2(flerp_executor *ex, int32_t seq, flerp_snapshot snap, int64_t from, flerp_state_desc *desc,
                                 flerp_sink sink, void *ctx);
/* The seq (open, nothing pending; at position 0, or at desc->from for a delta) takes the state; every snapshot of
 * the seq is invalid afterwards. A wrong fingerprint or prefix is FLERP_E_STATE. */
flerp_status flerp_state_import2(flerp_executor *ex, int32_t seq, const flerp_state_desc *desc, flerp_source source, void *ctx);
/* FLERP_CAP_SNAPSHOT_MARK: a save the seq's prompt forwards take when the seq reaches position `at` (a multiple of
 * caps.keep_step, past the seq's position). The handle is valid at once; flerp_snapshot_pos is negative
 * (FLERP_E_STATE) until a forward reached `at`. A mark not reached when a result row is asked for is dropped. */
flerp_status flerp_snapshot_mark(flerp_executor *ex, int32_t seq, int64_t at, flerp_snapshot *out);
/* FLERP_CAP_SNAPSHOT_SHARED: a copy of the seq's snapshot that belongs to no seq: it survives the seq, restores into
 * any seq (flerp_snapshot_restore), and is dropped with flerp_snapshot_drop(ex, -1, shared). */
flerp_status flerp_snapshot_share(flerp_executor *ex, int32_t seq, flerp_snapshot snap, flerp_snapshot *out);
/* FLERP_CAP_STATE_PARK: the seq's state leaves the device's working set (strixite's paged and small states) and the
 * seq is reset; resume puts it back into an open, reset seq and consumes the handle. */
flerp_status flerp_seq_park(flerp_executor *ex, int32_t seq, flerp_snapshot *out);
flerp_status flerp_seq_resume(flerp_executor *ex, int32_t seq, flerp_snapshot parked);

/* --- version 3: steering ------------------------------------------------------------------------------------- */
typedef struct flerp_steer_item {
    uint32_t size;
    uint32_t vector;                /* a flerp_steer_load handle */
    float scale;
    int64_t from, to;               /* FLERP_CAP_STEER_SPAN: positions [from, to) (to = -1: open); else 0 and -1 */
} flerp_steer_item;
/* A vector of caps.steer_dim floats added at `layer`; `name` is the caller's, for logs. Handles are never 0. */
flerp_status flerp_steer_load(flerp_executor *ex, const char *name, int32_t layer, const float *v, int32_t dim, uint32_t *out);
flerp_status flerp_steer_drop(flerp_executor *ex, uint32_t vector);   /* FLERP_E_STATE while a seq uses it */
/* The seq's vectors from its next forward on (n = 0: none); replaces the previous set. */
flerp_status flerp_seq_steer(flerp_executor *ex, int32_t seq, const flerp_steer_item *items, size_t n);

/* --- version 3: images --------------------------------------------------------------------------------------- */
typedef struct flerp_image_span {
    uint32_t size;
    int64_t at;                     /* the span's first position; ids [at, at + tokens) are caps.image_token */
    int64_t tokens;
    int32_t grid_t, grid_h, grid_w; /* the rows' grid (mRoPE positions) */
    const float *rows;              /* FLERP_CAP_IMAGE_ROWS: [tokens, caps.image_hidden] */
    const void *payload;            /* FLERP_CAP_IMAGE_PREPARE: the picture's file bytes, rows null */
    size_t payload_bytes;
} flerp_image_span;
/* The span a picture takes in the executor's own tower: fills tokens and grid of `out` (at, rows, payload unread). */
flerp_status flerp_image_prepare(flerp_executor *ex, const void *payload, size_t n, flerp_image_span *out);
/* flerp_forward whose ids carry image spans; spans lie inside the forward, ascending, not overlapping. */
flerp_status flerp_forward_spans(flerp_executor *ex, int32_t seq, const flerp_token *ids, size_t n,
                                 const flerp_image_span *spans, size_t n_spans, const flerp_want *want, flerp_rows *out,
                                 const flerp_token *lookahead, size_t n_lookahead);

/* --- version 3: slots ---------------------------------------------------------------------------------------- */
int flerp_seq_batchable(const flerp_executor *ex, int32_t seq);  /* 1: may share a verify; 0: verifies alone; < 0 */
/* A hint: ids the seq's next forward or draft step will run (strixite's prefetch_ple). */
flerp_status flerp_prefetch(flerp_executor *ex, int32_t seq, const flerp_token *ids, size_t n);
int flerp_seq_can_renew(const flerp_executor *ex, int32_t seq); /* 1, 0, or negative flerp_status */
/* The soft reset that rebuilds the seq's session state with its position and ids kept (strixite renew). */
flerp_status flerp_seq_renew(flerp_executor *ex, int32_t seq);

/* --- counters ------------------------------------------------------------------------------------------------ */
flerp_status flerp_get_stats(const flerp_executor *ex, flerp_stats *out);
/* The origin's own counters as one JSON object (strixite BackendStats, StrixFold drafter telemetry); writes at
 * most cap bytes with a terminating 0 and returns the length needed. */
size_t flerp_stats_json(const flerp_executor *ex, char *out, size_t cap);

#ifdef __cplusplus
}
#endif
#endif /* FLERP_EXECUTOR_H */
