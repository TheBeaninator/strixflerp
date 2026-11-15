/* strixflerp reply ABI, version 1: a model's generated tokens -> the parts of an OpenAI reply (reasoning, content,
 * tool calls), behind one C interface.
 *
 * How a reply splits belongs to the model's chat template (where its think block ends, how it spells a tool call),
 * not to the engine that runs it, so a front end (serve.flerp.http) reaches it through this header and a recipe
 * binds the module that provides it (text.reply/flerp-v1/-), next to the text module (include/flerp/text.h):
 *   runtime.flerp.reply_strixite   strixite's OutputParser and ThinkWatch (serve/output_parser.hpp,
 *                                  serve/think_nudge.hpp): Qwen3.8-Flash-Next's <think> and <tool_call> forms
 * The module also names the texts a decode loop feeds into a reply on the model's behalf (the thinking budget's stop,
 * the think close ahead of a tool call, the thinking nudges) and says when a nudge is due; the front end decides
 * whether to feed them and tokenizes them with the text module.
 *
 * Rules: a call returns FLERP_OK / a count, or a negative flerp_status; flerp_reply_last_error says why. Not thread
 * safe: one thread drives a flerp_reply_model and its replies. Event pointers live until the next call on the reply.
 */
#ifndef FLERP_REPLY_H
#define FLERP_REPLY_H

#include "flerp/executor.h" /* flerp_token, flerp_kv, flerp_status */

#ifdef __cplusplus
extern "C" {
#endif

#define FLERP_REPLY_ABI 1

typedef struct flerp_reply_model flerp_reply_model; /* opaque: the model's tokenizer and reply forms */
typedef struct flerp_reply flerp_reply;             /* opaque: one reply being parsed */

/* Keys are the module's own; strixite's are the text module's (tokenizer = the tokenizer.json), so a front end may
 * hand both modules the same keys. An unknown key is refused. */
flerp_status flerp_reply_open(const flerp_kv *config, size_t n_config, flerp_reply_model **out);
void flerp_reply_close(flerp_reply_model *m);
const char *flerp_reply_last_error(const flerp_reply_model *m); /* m null: the last flerp_reply_open's */
const char *flerp_reply_describe(const flerp_reply_model *m);

/* A text the decode loop may feed on the model's behalf, by name, or null for a name the module does not have:
 *   "thinking_stop"           fed when a thinking budget is spent: closes the think block
 *   "think_close_before_call" fed ahead of a tool call begun inside the think block
 *   "nudge1", "nudge2"        fed into a think block going in circles (never a close)
 *   "call_begin"              the token text that opens a tool call */
const char *flerp_reply_text(const flerp_reply_model *m, const char *name);

typedef struct flerp_reply_config {
    uint32_t size;
    int thinking;              /* the prompt left generation inside the think block */
    const char *tools;         /* the request's tools as JSON text (an array), or null: types the call arguments */
    size_t n_tools;
    const char *const *stop;   /* stop strings (non-empty), applied to content */
    size_t n_stop;
    uint64_t id_seed;          /* tool call ids */
    int trim_ws;               /* trim each part's leading and trailing white space (chat) */
    int literal_tags;          /* a tag token is structure only right after a newline */
    int stream_calls;          /* tool calls as they are generated (FLERP_REPLY_CALL_START / _ARGS) */
    int nudge;                 /* follow the reasoning for flerp_reply_state.nudge_due */
} flerp_reply_config;

flerp_status flerp_reply_begin(flerp_reply_model *m, const flerp_reply_config *cfg, flerp_reply **out);
void flerp_reply_end(flerp_reply *r);

enum {
    FLERP_REPLY_REASONING = 0,  /* text: a piece of reasoning */
    FLERP_REPLY_CONTENT = 1,    /* text: a piece of content */
    FLERP_REPLY_CALL = 2,       /* a whole tool call: id, name, arguments (JSON text of an object) */
    FLERP_REPLY_CALL_START = 3, /* a streamed call's id and name */
    FLERP_REPLY_CALL_ARGS = 4   /* text: a piece of the streamed call's arguments' JSON text */
};

typedef struct flerp_reply_event {
    uint32_t kind;
    const char *text;          /* REASONING, CONTENT, CALL_ARGS; the CALL's arguments */
    size_t n_text;
    const char *id, *name;     /* CALL, CALL_START */
} flerp_reply_event;

/* One generated token (never an end id: the loop ends on those). `fed`: the loop fed it on the model's behalf (it is
 * not followed for the nudge). Returns 1 when a stop string completed (the reply ends; this token's text past the
 * stop string is dropped), 0 otherwise, or a negative status. The token's events go to *events / *n_events. */
int flerp_reply_feed(flerp_reply *r, flerp_token token, int fed, const flerp_reply_event **events, size_t *n_events);
/* The reply is over: held text released, a call still open dropped. */
flerp_status flerp_reply_finish(flerp_reply *r, const flerp_reply_event **events, size_t *n_events);

typedef struct flerp_reply_state {
    uint32_t size;
    int in_reasoning;          /* inside the think block */
    int in_call;               /* between a call's open and close */
    int after_newline;         /* the last token fed ended in a newline (or none was fed) */
    int nudge_due;             /* 0, or the nudge (1, 2) to feed now */
    int dropped_partial_call;  /* after finish: a call was open at the end */
    int64_t reasoning_tokens, tool_calls, malformed_calls, invalid_bytes;
} flerp_reply_state;

flerp_status flerp_reply_get_state(const flerp_reply *r, flerp_reply_state *out);
/* The nudge flerp_reply_state.nudge_due named was fed. */
void flerp_reply_nudge_fired(flerp_reply *r);

/* --- constraints: a structured answer as token masks the host hands an executor (FLERP_CAP_MASKS) ------------------
 * The module compiles the constraint with the model's grammar and tokenizer and answers, for the reply so far and a
 * verify's drafts, which ids each row may give (flerp_masks.bits: bit id % 32 of word id / 32). Before the think
 * block closes a row refuses only the end of the turn; after it, the grammar's tokens, and the end of the turn
 * where the grammar is complete. */
typedef struct flerp_reply_constraint flerp_reply_constraint; /* opaque */

enum {
    FLERP_REPLY_JSON_OBJECT = 1, /* any JSON object (text unused) */
    FLERP_REPLY_JSON_SCHEMA = 2  /* text: a JSON Schema */
};

/* in_content: the prompt has thinking off (the grammar applies from the first token). width: the ids a mask row
 * covers (the executor's logits row). FLERP_E_ARGUMENT: the schema cannot be enforced, and flerp_reply_last_error
 * says why in the client's terms. */
flerp_status flerp_reply_constrain(flerp_reply_model *m, uint32_t kind, const char *text, size_t n_text, int in_content,
                                   int64_t width, flerp_reply_constraint **out);
void flerp_reply_constraint_free(flerp_reply_constraint *c);
int64_t flerp_reply_constraint_words(const flerp_reply_constraint *c); /* words a mask row */
/* Rows for the reply so far followed by pending[0, r), r = 0 .. n_pending, into bits[(n_pending + 1) * words]. */
flerp_status flerp_reply_constraint_masks(flerp_reply_constraint *c, const flerp_token *pending, size_t n_pending, uint32_t *bits,
                                          size_t n_bits);
/* 1: `token` may follow the reply so far and pending[0, n_pending); 0: it may not. */
int flerp_reply_constraint_allows(flerp_reply_constraint *c, const flerp_token *pending, size_t n_pending, flerp_token token);
/* A token joined the reply (picked or fed). FLERP_E_ARGUMENT: the constraint refuses it (a mask was not applied). */
flerp_status flerp_reply_constraint_feed(flerp_reply_constraint *c, flerp_token token);

#ifdef __cplusplus
}
#endif
#endif /* FLERP_REPLY_H */
