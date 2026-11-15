// include/flerp/reply.h over strixite's own reply parser and thinking watch (serve/output_parser.hpp,
// serve/think_nudge.hpp): Qwen3.8-Flash-Next's reply forms. Nothing is re-implemented here: the parser is the
// origin's OutputParser, the nudge trigger its ThinkWatch, and the fed texts its engine's constants
// (serve/engine.hpp kThinkingStop and kThinkCloseBeforeCall at 20937dc, serve/think_nudge.hpp), so a reply splits into
// the bytes the origin's server sends.
//
// flerp_reply_open keys: tokenizer (the checkpoint's tokenizer.json, required) - the text module's key.
// The nudge watch follows the origin's served policy (think-nudge-rate 0.25, think-nudge-min-tokens 3072,
// think-nudge-wording commit: server_config.cpp defaults).
#include "flerp/reply.h"

#include "serve/grammar.hpp"
#include "serve/json.hpp"
#include "serve/json_schema.hpp"
#include "serve/output_parser.hpp"
#include "serve/structured_output.hpp"
#include "serve/think_nudge.hpp"
#include "serve/tokenizer.hpp"

#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace strix;

namespace {
std::string g_open_error;

// serve/engine.hpp at 20937dc (engine constants; the engine module itself needs the GPU, so they are named here)
constexpr const char *kThinkingStop =
    "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly "
    "now.\n</think>\n\n";
constexpr const char *kThinkCloseBeforeCall = "</think>\n\n";
}  // namespace

struct flerp_reply_model {
    std::unique_ptr<Tokenizer> tok;
    std::string describe;
    mutable std::string error;
    int32_t think_end = -1;
    std::vector<int32_t> end_ids;
    std::map<int64_t, std::unique_ptr<TokenMasker>> maskers;  // by mask width
    // compiled constraints kept across requests, most recent first (serve/grammar_cache.hpp's rule: a repeated schema
    // costs neither the compile nor its masks again); at most 16
    std::list<std::pair<std::string, std::shared_ptr<grammar::GrammarAutomaton>>> automata;
};

struct flerp_reply_constraint {
    flerp_reply_model *m;
    std::shared_ptr<grammar::GrammarAutomaton> automaton;
    std::unique_ptr<StructuredOutput> so;
    int64_t words = 0;
};

struct flerp_reply {
    flerp_reply_model *m;
    json::Value tools;
    bool has_tools = false;
    std::unique_ptr<OutputParser> parser;
    std::unique_ptr<ThinkWatch> watch;
    std::vector<OutputEvent> raw;
    std::vector<std::string> args;  // CALL arguments' JSON text, alive with `out`
    std::vector<flerp_reply_event> out;

    void publish(const flerp_reply_event **events, size_t *n) {
        out.clear(), args.clear();
        args.reserve(raw.size());
        for (const OutputEvent &e : raw) {
            flerp_reply_event v{};
            switch (e.kind) {
            case OutputEvent::Kind::Reasoning: v.kind = FLERP_REPLY_REASONING; break;
            case OutputEvent::Kind::Content: v.kind = FLERP_REPLY_CONTENT; break;
            case OutputEvent::Kind::ToolCall: v.kind = FLERP_REPLY_CALL; break;
            case OutputEvent::Kind::ToolCallStart: v.kind = FLERP_REPLY_CALL_START; break;
            case OutputEvent::Kind::ToolCallArgs: v.kind = FLERP_REPLY_CALL_ARGS; break;
            }
            if (e.kind == OutputEvent::Kind::ToolCall) {
                args.push_back(e.call.arguments.dump());
                v.text = args.back().data(), v.n_text = args.back().size();
            } else {
                v.text = e.text.data(), v.n_text = e.text.size();
            }
            if (e.kind == OutputEvent::Kind::ToolCall || e.kind == OutputEvent::Kind::ToolCallStart)
                v.id = e.call.id.c_str(), v.name = e.call.name.c_str();
            out.push_back(v);
        }
        if (events) *events = out.data();
        if (n) *n = out.size();
    }
};

extern "C" {

flerp_status flerp_reply_open(const flerp_kv *config, size_t n_config, flerp_reply_model **out) {
    g_open_error.clear();
    std::string path;
    for (size_t i = 0; i < n_config; ++i) {
        const std::string key = config[i].key ? config[i].key : "";
        if (key != "tokenizer") {
            g_open_error = "flerp_reply_open (strixite): unknown key '" + key + "' (tokenizer)";
            return FLERP_E_CONFIG;
        }
        path = config[i].value ? config[i].value : "";
    }
    if (out == nullptr || path.empty()) {
        g_open_error = "flerp_reply_open (strixite): tokenizer (the checkpoint's tokenizer.json) is required";
        return FLERP_E_CONFIG;
    }
    auto m = std::make_unique<flerp_reply_model>();
    try {
        m->tok = std::make_unique<Tokenizer>(path);
        m->think_end = m->tok->id_of("</think>");
        (void)m->tok->id_of("<tool_call>"), (void)m->tok->id_of("</tool_call>");
        m->end_ids = {m->tok->id_of("<|im_end|>"), m->tok->id_of("<|endoftext|>")};  // the engine's im_end_, eot_
        m->describe = "strixite reply parser (Qwen3.8-Flash-Next: <think>, <tool_call> XML calls), " + path;
    } catch (const std::exception &e) {
        g_open_error = std::string("flerp_reply_open (strixite): ") + e.what();
        return FLERP_E_FAILED;
    }
    *out = m.release();
    return FLERP_OK;
}

void flerp_reply_close(flerp_reply_model *m) { delete m; }
const char *flerp_reply_last_error(const flerp_reply_model *m) { return m ? m->error.c_str() : g_open_error.c_str(); }
const char *flerp_reply_describe(const flerp_reply_model *m) { return m->describe.c_str(); }

const char *flerp_reply_text(const flerp_reply_model *, const char *name) {
    const std::string n = name ? name : "";
    if (n == "thinking_stop") return kThinkingStop;
    if (n == "think_close_before_call") return kThinkCloseBeforeCall;
    if (n == "nudge1") return think_nudge1_text("commit");
    if (n == "nudge2") return kThinkNudge2;
    if (n == "call_begin") return "<tool_call>";
    return nullptr;
}

flerp_status flerp_reply_begin(flerp_reply_model *m, const flerp_reply_config *cfg, flerp_reply **out) {
    if (cfg == nullptr || cfg->size < sizeof(flerp_reply_config) || out == nullptr) {
        m->error = "flerp_reply_config is null or short";
        return FLERP_E_ARGUMENT;
    }
    auto r = std::make_unique<flerp_reply>();
    r->m = m;
    try {
        if (cfg->tools != nullptr && cfg->n_tools > 0) {
            r->tools = json::Value::parse(std::string_view(cfg->tools, cfg->n_tools));
            r->has_tools = true;
        }
        OutputParser::Config pc;
        pc.thinking = cfg->thinking != 0;
        pc.tools = r->has_tools ? &r->tools : nullptr;
        for (size_t i = 0; i < cfg->n_stop; ++i) pc.stop.emplace_back(cfg->stop[i]);
        pc.id_seed = cfg->id_seed;
        pc.trim_ws = cfg->trim_ws != 0;
        pc.literal_tags = cfg->literal_tags != 0;
        pc.stream_calls = cfg->stream_calls != 0;
        r->parser = std::make_unique<OutputParser>(*m->tok, pc);
        if (cfg->nudge) r->watch = std::make_unique<ThinkWatch>(ThinkWatch::Policy{});
    } catch (const std::exception &e) {
        m->error = std::string("flerp_reply_begin (strixite): ") + e.what();
        return FLERP_E_ARGUMENT;
    }
    *out = r.release();
    return FLERP_OK;
}

void flerp_reply_end(flerp_reply *r) { delete r; }

int flerp_reply_feed(flerp_reply *r, flerp_token token, int fed, const flerp_reply_event **events, size_t *n_events) {
    try {
        r->raw.clear();
        const bool stopped = r->parser->feed(token, r->raw);
        // the engine's watch_tok: a sampled token that left the parser in the think block
        if (r->watch && !fed && r->parser->in_reasoning()) r->watch->observe(token, r->m->tok->decode({token}));
        r->publish(events, n_events);
        return stopped ? 1 : 0;
    } catch (const std::exception &e) {
        r->m->error = std::string("flerp_reply_feed (strixite): ") + e.what();
        return FLERP_E_FAILED;
    }
}

flerp_status flerp_reply_finish(flerp_reply *r, const flerp_reply_event **events, size_t *n_events) {
    try {
        r->raw.clear();
        r->parser->finish(r->raw);
        r->publish(events, n_events);
        return FLERP_OK;
    } catch (const std::exception &e) {
        r->m->error = std::string("flerp_reply_finish (strixite): ") + e.what();
        return FLERP_E_FAILED;
    }
}

flerp_status flerp_reply_get_state(const flerp_reply *r, flerp_reply_state *out) {
    if (out == nullptr || out->size < sizeof(flerp_reply_state)) return FLERP_E_ARGUMENT;
    const OutputParser &p = *r->parser;
    out->in_reasoning = p.in_reasoning(), out->in_call = p.in_call(), out->after_newline = p.after_newline();
    out->nudge_due = r->watch && p.in_reasoning() ? r->watch->due() : 0;
    out->dropped_partial_call = p.dropped_partial_call();
    out->reasoning_tokens = p.reasoning_tokens(), out->tool_calls = p.tool_calls();
    out->malformed_calls = p.malformed_calls(), out->invalid_bytes = p.invalid_bytes();
    return FLERP_OK;
}

void flerp_reply_nudge_fired(flerp_reply *r) {
    if (r->watch) r->watch->fired();
}

flerp_status flerp_reply_constrain(flerp_reply_model *m, uint32_t kind, const char *text, size_t n_text, int in_content, int64_t width,
                                   flerp_reply_constraint **out) {
    if (out == nullptr || width < m->tok->size()) {
        m->error = "flerp_reply_constrain: no out, or a mask width below the tokenizer's ids";
        return FLERP_E_ARGUMENT;
    }
    try {
        std::string key;
        json::Value schema;
        if (kind == FLERP_REPLY_JSON_OBJECT) key = "json_object";
        else if (kind == FLERP_REPLY_JSON_SCHEMA) schema = json::Value::parse(std::string_view(text, n_text)), key = "json_schema " + schema.dump();
        else {
            m->error = "flerp_reply_constrain: unknown kind " + std::to_string(kind);
            return FLERP_E_UNSUPPORTED;
        }
        std::shared_ptr<grammar::GrammarAutomaton> a;
        for (auto it = m->automata.begin(); it != m->automata.end(); ++it)
            if (it->first == key) {
                a = it->second;
                m->automata.splice(m->automata.begin(), m->automata, it);
                break;
            }
        if (!a) {
            a = std::make_shared<grammar::GrammarAutomaton>(kind == FLERP_REPLY_JSON_OBJECT ? json_schema::compile_any_object()
                                                                                          : json_schema::compile(schema, "response_format.json_schema.schema"));
            m->automata.emplace_front(key, a);
            if (m->automata.size() > 16) m->automata.pop_back();
        }
        std::unique_ptr<TokenMasker> &masker = m->maskers[width];
        if (!masker) masker = std::make_unique<TokenMasker>(*m->tok, (int32_t)width);
        auto c = std::make_unique<flerp_reply_constraint>();
        c->m = m, c->automaton = a, c->words = masker->mask_words();
        c->so = std::make_unique<StructuredOutput>(a, *masker, *m->tok, m->think_end, m->end_ids, in_content != 0);
        *out = c.release();
        return FLERP_OK;
    } catch (const std::exception &e) {
        // the check's own prefix (function, file:line, condition) is the server log's; the client gets the reason
        const std::string full = e.what();
        const size_t at = full.find("failed: ");
        m->error = at == std::string::npos ? full : full.substr(at + 8);
        return FLERP_E_ARGUMENT;
    }
}

void flerp_reply_constraint_free(flerp_reply_constraint *c) { delete c; }

int64_t flerp_reply_constraint_words(const flerp_reply_constraint *c) { return c->words; }

flerp_status flerp_reply_constraint_masks(flerp_reply_constraint *c, const flerp_token *pending, size_t n_pending, uint32_t *bits, size_t n_bits) {
    try {
        const LogitMasks lm = c->so->masks_for(std::vector<int32_t>(pending, pending + n_pending));
        if (lm.bits.size() > n_bits) {
            c->m->error = "flerp_reply_constraint_masks: the bits buffer is too small";
            return FLERP_E_ARGUMENT;
        }
        std::copy(lm.bits.begin(), lm.bits.end(), bits);
        return FLERP_OK;
    } catch (const std::exception &e) {
        c->m->error = std::string("flerp_reply_constraint_masks: ") + e.what();
        return FLERP_E_FAILED;
    }
}

int flerp_reply_constraint_allows(flerp_reply_constraint *c, const flerp_token *pending, size_t n_pending, flerp_token token) {
    try {
        StructuredOutput::Cursor at = c->so->cursor();
        for (size_t i = 0; i < n_pending; ++i) {
            if (!c->so->allows(at, pending[i])) return 0;
            at = c->so->after(at, pending[i]);
        }
        return c->so->allows(at, token) ? 1 : 0;
    } catch (const std::exception &) {
        return 0;
    }
}

flerp_status flerp_reply_constraint_feed(flerp_reply_constraint *c, flerp_token token) {
    try {
        c->so->feed(token);
        return FLERP_OK;
    } catch (const std::exception &e) {
        c->m->error = std::string("flerp_reply_constraint_feed: ") + e.what();
        return FLERP_E_ARGUMENT;
    }
}

}  // extern "C"
