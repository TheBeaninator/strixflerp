// include/flerp/text.h over strixite's own tokenizer and chat template (serve/tokenizer.hpp, serve/chat_template.hpp):
// Qwen3.8-Flash-Next's text. Nothing is re-implemented here; the request's template switches are read the way
// strixite's request parser reads them (serve/openai.cpp at 20937dc, "Thinking on / off"), so a request renders to
// the prompt the origin's server renders it to.
//
// flerp_text_open keys: tokenizer (the checkpoint's tokenizer.json, required).
// Not read here, because they are serving policy of strixite's HTTP layer and not the template's: the server's
// thinking default, thinking left out for a small max_tokens, thinking budgets. A request that names "tools" is
// rendered with them (the template's own tool section); whether tools are served is the front end's decision.
#include "flerp/text.h"

#include "common/check.hpp"
#include "serve/chat_template.hpp"
#include "serve/json.hpp"
#include "serve/tokenizer.hpp"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace strix;

struct flerp_text {
    std::unique_ptr<Tokenizer> tok;
    std::string describe;
    mutable std::string error;
    std::string rendered;
    std::vector<size_t> plain;
    std::vector<flerp_token> end_ids;
};

namespace {
std::string g_open_error;

struct Refused {  // the client's error in a chat request
    std::string what;
};

bool get_bool(const json::Value &v, const std::string &name) {
    if (!v.is_bool()) throw Refused{"'" + name + "' must be true or false"};
    return v.as_bool(name);
}

// serve/openai.cpp: the three spellings of thinking on / off, reasoning_effort (a level, or "none" = off) and
// chat_template_kwargs. Two that disagree are refused.
ChatTemplateOptions options_of(const json::Value &body) {
    ChatTemplateOptions o;
    std::vector<std::pair<std::string, bool>> on_off;
    const auto effort = [&](const json::Value &v, const std::string &name) {
        if (!v.is_string()) throw Refused{"'" + name + "' must be a string"};
        const std::string e = v.as_string(name);
        if (e == "none") on_off.emplace_back(name, false);
        else if (e == "xhigh" || e == "medium" || e == "low") o.reasoning_effort = e;
        else if (e == "high") o.reasoning_effort = "xhigh";  // OpenAI's top level is the template's top level
        else throw Refused{"'" + name + "' must be one of none, low, medium, high, xhigh; got \"" + e + "\""};
    };
    if (const json::Value *v = body.find("reasoning_effort"); v && !v->is_null()) effort(*v, "reasoning_effort");
    if (const json::Value *v = body.find("enable_thinking"); v && !v->is_null())
        on_off.emplace_back("enable_thinking", get_bool(*v, "enable_thinking"));
    if (const json::Value *t = body.find("thinking"); t && t->is_object())
        if (const json::Value *ty = t->find("type"); ty && !ty->is_null()) {
            if (!ty->is_string() || (ty->as_string("thinking.type") != "enabled" && ty->as_string("thinking.type") != "disabled"))
                throw Refused{"'thinking.type' must be \"enabled\" or \"disabled\""};
            on_off.emplace_back("thinking.type", ty->as_string("thinking.type") == "enabled");
        }
    if (const json::Value *k = body.find("chat_template_kwargs"); k && !k->is_null()) {
        if (!k->is_object()) throw Refused{"'chat_template_kwargs' must be an object"};
        for (const auto &[key, v] : k->as_object("chat_template_kwargs")) {
            if (v.is_null()) continue;
            const std::string name = "chat_template_kwargs." + key;
            if (key == "enable_thinking") on_off.emplace_back(name, get_bool(v, name));
            else if (key == "preserve_thinking") o.preserve_thinking = get_bool(v, name);
            else if (key == "reasoning_effort") effort(v, name);
            else throw Refused{"chat_template_kwargs '" + key + "' is not supported (enable_thinking, preserve_thinking, reasoning_effort)"};
        }
    }
    for (const auto &[name, on] : on_off)
        if (on != on_off[0].second)
            throw Refused{"thinking on / off disagree: '" + on_off[0].first + "' says " + (on_off[0].second ? "on" : "off") + ", '" +
                          name + "' says " + (on ? "on" : "off")};
    if (!on_off.empty()) o.enable_thinking = on_off[0].second;
    return o;
}

template <class F> int64_t guarded(const flerp_text *t, F &&f) {
    try {
        return f();
    } catch (const std::exception &e) {
        t->error = e.what();
        return FLERP_E_ARGUMENT;
    }
}
}  // namespace

extern "C" {

flerp_status flerp_text_open(const flerp_kv *config, size_t n_config, flerp_text **out) {
    g_open_error.clear();
    std::string path;
    for (size_t i = 0; i < n_config; ++i) {
        const std::string key = config[i].key ? config[i].key : "";
        if (key != "tokenizer") {
            g_open_error = "flerp_text_open (strixite): unknown key '" + key + "' (tokenizer)";
            return FLERP_E_CONFIG;
        }
        path = config[i].value ? config[i].value : "";
    }
    if (out == nullptr || path.empty()) {
        g_open_error = "flerp_text_open (strixite): tokenizer (the checkpoint's tokenizer.json) is required";
        return FLERP_E_CONFIG;
    }
    auto t = std::make_unique<flerp_text>();
    try {
        t->tok = std::make_unique<Tokenizer>(path);
        // the ids Engine ends a turn at (serve/engine.cpp: im_end_, eot_)
        t->end_ids = {t->tok->id_of("<|im_end|>"), t->tok->id_of("<|endoftext|>")};
        t->describe = "strixite tokenizer and chat template (Qwen3.8-Flash-Next), " + std::to_string(t->tok->size()) + " tokens, " + path;
    } catch (const std::exception &e) {
        g_open_error = std::string("flerp_text_open (strixite): ") + e.what();
        return FLERP_E_FAILED;
    }
    *out = t.release();
    return FLERP_OK;
}

void flerp_text_close(flerp_text *t) { delete t; }
const char *flerp_text_last_error(const flerp_text *t) { return t ? t->error.c_str() : g_open_error.c_str(); }
const char *flerp_text_describe(const flerp_text *t) { return t->describe.c_str(); }
int64_t flerp_text_vocab(const flerp_text *t) { return t->tok->size(); }

int64_t flerp_text_encode(flerp_text *t, const char *text, size_t n, const size_t *plain, size_t n_plain, flerp_token *out, size_t cap) {
    return guarded(t, [&]() -> int64_t {
        std::vector<std::pair<size_t, size_t>> spans(n_plain);
        for (size_t i = 0; i < n_plain; ++i) spans[i] = {plain[2 * i], plain[2 * i + 1]};
        const std::string_view sv(text, n);
        const std::vector<int32_t> ids = n_plain > 0 ? t->tok->encode(sv, spans) : t->tok->encode(sv);
        if (ids.size() <= cap && out != nullptr) std::memcpy(out, ids.data(), ids.size() * sizeof(flerp_token));
        return (int64_t)ids.size();
    });
}

int64_t flerp_text_token_bytes(flerp_text *t, flerp_token id, char *out, size_t cap) {
    return guarded(t, [&]() -> int64_t {
        const std::string &b = t->tok->token_bytes(id);
        if (b.size() <= cap && out != nullptr) std::memcpy(out, b.data(), b.size());
        return (int64_t)b.size();
    });
}

int64_t flerp_text_end_ids(const flerp_text *t, flerp_token *out, size_t cap) {
    if (t->end_ids.size() <= cap && out != nullptr) std::memcpy(out, t->end_ids.data(), t->end_ids.size() * sizeof(flerp_token));
    return (int64_t)t->end_ids.size();
}

flerp_token flerp_text_token_id(const flerp_text *t, const char *piece) {
    try {
        return t->tok->id_of(piece);
    } catch (const std::exception &) {
        return -1;
    }
}

flerp_status flerp_text_render_chat(flerp_text *t, const char *request, size_t n, flerp_text_chat *out) {
    if (out == nullptr || out->size < sizeof(flerp_text_chat)) {
        t->error = "flerp_text_chat is null or short";
        return FLERP_E_ARGUMENT;
    }
    try {
        const json::Value body = json::Value::parse(std::string_view(request, n));
        const json::Value *messages = body.find("messages");
        if (messages == nullptr) throw Refused{"'messages' is required"};
        const json::Value *tools = body.find("tools");
        if (tools != nullptr && !(tools->is_array() && !tools->as_array("tools").empty())) tools = nullptr;
        std::vector<std::pair<size_t, size_t>> spans;
        t->rendered = render_chat(*messages, tools, options_of(body), &spans);
        t->plain.clear();
        for (const auto &[b, e] : spans) t->plain.push_back(b), t->plain.push_back(e);
    } catch (const Refused &r) {
        t->error = r.what;
        return FLERP_E_ARGUMENT;
    } catch (const std::exception &e) {  // the template's own refusals and JSON type errors: the client's request
        t->error = e.what();
        return FLERP_E_ARGUMENT;
    }
    out->text = t->rendered.data(), out->n_text = t->rendered.size();
    out->plain = t->plain.data(), out->n_plain = t->plain.size() / 2;
    return FLERP_OK;
}

}  // extern "C"
