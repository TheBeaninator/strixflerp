/* include/flerp/reply.h for a recipe whose model has no reply module yet: flerp_reply_open says so
 * (FLERP_E_UNSUPPORTED), and the front end returns chat replies whole and refuses what needs the split. */
#include "flerp/reply.h"

#include <stddef.h>

flerp_status flerp_reply_open(const flerp_kv *config, size_t n_config, flerp_reply_model **out) {
    (void)config, (void)n_config;
    if (out) *out = NULL;
    return FLERP_E_UNSUPPORTED;
}
void flerp_reply_close(flerp_reply_model *m) { (void)m; }
const char *flerp_reply_last_error(const flerp_reply_model *m) {
    (void)m;
    return "runtime.flerp.reply_none: this recipe binds no reply module";
}
const char *flerp_reply_describe(const flerp_reply_model *m) {
    (void)m;
    return "none";
}
const char *flerp_reply_text(const flerp_reply_model *m, const char *name) {
    (void)m, (void)name;
    return NULL;
}
flerp_status flerp_reply_begin(flerp_reply_model *m, const flerp_reply_config *cfg, flerp_reply **out) {
    (void)m, (void)cfg, (void)out;
    return FLERP_E_UNSUPPORTED;
}
void flerp_reply_end(flerp_reply *r) { (void)r; }
int flerp_reply_feed(flerp_reply *r, flerp_token token, int fed, const flerp_reply_event **events, size_t *n_events) {
    (void)r, (void)token, (void)fed, (void)events, (void)n_events;
    return FLERP_E_UNSUPPORTED;
}
flerp_status flerp_reply_finish(flerp_reply *r, const flerp_reply_event **events, size_t *n_events) {
    (void)r, (void)events, (void)n_events;
    return FLERP_E_UNSUPPORTED;
}
flerp_status flerp_reply_get_state(const flerp_reply *r, flerp_reply_state *out) {
    (void)r, (void)out;
    return FLERP_E_UNSUPPORTED;
}
void flerp_reply_nudge_fired(flerp_reply *r) { (void)r; }
flerp_status flerp_reply_constrain(flerp_reply_model *m, uint32_t kind, const char *text, size_t n_text, int in_content,
                                   int64_t width, flerp_reply_constraint **out) {
    (void)m, (void)kind, (void)text, (void)n_text, (void)in_content, (void)width, (void)out;
    return FLERP_E_UNSUPPORTED;
}
void flerp_reply_constraint_free(flerp_reply_constraint *c) { (void)c; }
int64_t flerp_reply_constraint_words(const flerp_reply_constraint *c) {
    (void)c;
    return 0;
}
flerp_status flerp_reply_constraint_masks(flerp_reply_constraint *c, const flerp_token *pending, size_t n_pending, uint32_t *bits,
                                          size_t n_bits) {
    (void)c, (void)pending, (void)n_pending, (void)bits, (void)n_bits;
    return FLERP_E_UNSUPPORTED;
}
int flerp_reply_constraint_allows(flerp_reply_constraint *c, const flerp_token *pending, size_t n_pending, flerp_token token) {
    (void)c, (void)pending, (void)n_pending, (void)token;
    return 0;
}
flerp_status flerp_reply_constraint_feed(flerp_reply_constraint *c, flerp_token token) {
    (void)c, (void)token;
    return FLERP_E_UNSUPPORTED;
}
