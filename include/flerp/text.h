/* strixflerp text ABI, version 1: a model's tokenizer and chat template behind one C interface.
 *
 * Tokenizer and template belong to the model, not to the engine that runs it, so a front end (serve.flerp.http)
 * reaches them through this header and a recipe binds the module that provides it (text.model/flerp-v1/-):
 *   runtime.flerp.text_strixite   strixite's Tokenizer and render_chat (serve/tokenizer.hpp, serve/chat_template.hpp)
 *   runtime.flerp.text_strixfold  StrixFold's HfText (zig/src/server/hf_text.zig: its tokenizer and Jinja engine)
 * The modules are thin: no tokenizer is written here.
 *
 * Rules: a call returns FLERP_OK / a count, or a negative flerp_status; flerp_text_last_error says why. Not thread
 * safe: one thread drives a flerp_text. Buffers are the caller's; a count larger than the capacity given means the
 * buffer was too small and nothing useful was written.
 */
#ifndef FLERP_TEXT_H
#define FLERP_TEXT_H

#include "flerp/executor.h" /* flerp_token, flerp_kv, flerp_status */

#ifdef __cplusplus
extern "C" {
#endif

#define FLERP_TEXT_ABI 1

typedef struct flerp_text flerp_text; /* opaque */

/* Keys are the module's own (strixite: tokenizer = the tokenizer.json; StrixFold: model_dir = the checkpoint
 * directory); an unknown key is refused. */
flerp_status flerp_text_open(const flerp_kv *config, size_t n_config, flerp_text **out);
void flerp_text_close(flerp_text *t);
const char *flerp_text_last_error(const flerp_text *t); /* t null: the last flerp_text_open's */
const char *flerp_text_describe(const flerp_text *t);   /* lives as long as t */
int64_t flerp_text_vocab(const flerp_text *t);          /* ids are [0, vocab): what a sampler may draw from */

/* The ids of `text` (UTF-8), a special token's text as that token - except inside `plain` (n_plain byte ranges
 * [begin, end), ascending, as flerp_text_render_chat reports them), where it is ordinary text. Returns the count. */
int64_t flerp_text_encode(flerp_text *t, const char *text, size_t n, const size_t *plain, size_t n_plain,
                          flerp_token *out, size_t cap);
/* One token's bytes (a token may end inside a UTF-8 sequence: the caller joins them). Returns the length. */
int64_t flerp_text_token_bytes(flerp_text *t, flerp_token id, char *out, size_t cap);
/* The ids that end a reply (the model's end of turn and end of text). Returns the count. */
int64_t flerp_text_end_ids(const flerp_text *t, flerp_token *out, size_t cap);
/* The id of a token named by its text ("<|im_start|>"), or -1. */
flerp_token flerp_text_token_id(const flerp_text *t, const char *piece);

typedef struct flerp_text_chat {
    uint32_t size;
    const char *text;     /* the rendered prompt; lives until the next call on t */
    size_t n_text;
    const size_t *plain;  /* byte ranges of `text` that came from the request and not from the template */
    size_t n_plain;       /* ranges (2 entries each) */
} flerp_text_chat;

/* The chat template on a chat-completions request: `request` is the request's JSON object as text. The module
 * reads "messages" and the template switches its model has (enable_thinking, reasoning_effort,
 * chat_template_kwargs, ...). FLERP_E_ARGUMENT: the request is the client's error (flerp_text_last_error is the
 * message for it); FLERP_E_UNSUPPORTED: the model has no chat template. */
flerp_status flerp_text_render_chat(flerp_text *t, const char *request, size_t n, flerp_text_chat *out);

#ifdef __cplusplus
}
#endif
#endif /* FLERP_TEXT_H */
