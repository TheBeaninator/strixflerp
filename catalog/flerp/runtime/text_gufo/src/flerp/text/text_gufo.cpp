// SPDX-License-Identifier: MIT
// include/flerp/text.h over gufo's tokenizer and chat template for the Qwen 3.5 / 3.6 family (GGUF files of
// executor.gufo.qwen and executor.flerp.qwen35moe). Nothing is re-implemented here:
//   tokenizer   gufo::tokenization::QwenTokenizer::CreateFromGguf, the one the executor's loader builds
//               (catalog/flerp/executor/qwen35moe/src/flerp/qwen35moe/src/models/qwen/hip/model_loader.cpp 436-437;
//               the origin's src/models/qwen/hip/model_loader.cpp 347-348);
//   template    the check the loader runs first, QwenChatTemplate::ValidateGgufTemplate (adapted loader 426-429,
//               origin 337-340): a Qwen3.8 file must carry a pinned template, another Qwen file passes; then the
//               prompt is rendered by QwenChatTemplate::Render, gufo's compiled Qwen ChatML formatter, with the
//               options gufo's server resolves (ResolveQwenChatOptions; the rendered-prompt bound of the context,
//               src/cli/serve/inference_backend.cpp QwenChatOptions). gufo's server renders every Qwen file this way;
//               the file's own Jinja template is read only for its SHA-256 and profile (in the describe line);
//   encoding    QwenChatTemplate::EncodeRendered: message content as text (no control tokens), the rest as framing;
//               a completions prompt the same way with no content spans, as the generator encodes --prompt-file
//               (tool.flerp.qwen35moe_generate, generate.cpp 833).
// A request's thinking switches are read as gufo's server reads them (src/cli/serve/openai_chat.cpp,
// ParseReasoningOptions): thinking.type, reasoning_effort, chat_template_kwargs.{enable_thinking, thinking_mode,
// reasoning_effort, preserve_thinking}, plus the top-level enable_thinking that strixite's server takes; two that
// disagree are refused. Messages: system, developer, user, assistant (content a string or text parts;
// reasoning_content on an assistant turn). Tool messages and assistant tool calls are refused (the front end
// serves no tools).
//
// flerp_text_open keys: weights (the GGUF file, required), context (positions, for the rendered-prompt bound;
// default 4096).
#include "flerp/text.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <optional>
#include <string>
#include <vector>

#include "serve/json.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/reasoning.hpp"
#include "src/families/qwen/text/chat_template.hpp"
#include "src/families/qwen/text/tokenizer.hpp"

namespace tk = gufo::tokenization;
using strix::json::Value;

struct flerp_text {
    std::unique_ptr<tk::QwenTokenizer> tok;
    std::uint32_t context = 4096;
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

bool get_bool(const Value &v, const std::string &name) {
    if (!v.is_bool()) throw Refused{"'" + name + "' must be true or false"};
    return v.as_bool(name);
}

const Value *present(const Value &body, const char *key) {
    const Value *v = body.find(key);
    return v != nullptr && !v->is_null() ? v : nullptr;
}

gufo::ReasoningEffort effort_of(const Value &v, const std::string &name) {
    if (!v.is_string()) throw Refused{"'" + name + "' must be a string"};
    const std::string e = v.as_string(name);
    if (e == "minimal") return gufo::ReasoningEffort::kMinimal;
    if (e == "low") return gufo::ReasoningEffort::kLow;
    if (e == "medium") return gufo::ReasoningEffort::kMedium;
    if (e == "high") return gufo::ReasoningEffort::kHigh;
    if (e == "xhigh") return gufo::ReasoningEffort::kXHigh;
    if (e == "max") return gufo::ReasoningEffort::kMax;
    throw Refused{"'" + name + "' must be one of none, minimal, low, medium, high, xhigh, max; got \"" + e + "\""};
}

// gufo's ParseReasoningOptions (openai_chat.cpp) and the top-level enable_thinking; "none" as an effort = off.
gufo::ReasoningOptions reasoning_of(const Value &body) {
    gufo::ReasoningOptions r;
    std::vector<std::pair<std::string, bool>> on_off;
    const auto effort = [&](const Value &v, const std::string &name) {
        if (v.is_string() && v.as_string(name) == "none") on_off.emplace_back(name, false);
        else r.effort = effort_of(v, name);
    };
    if (const Value *t = present(body, "thinking")) {
        const Value *ty = t->is_object() ? t->find("type") : nullptr;
        if (ty == nullptr || !ty->is_string() || (ty->as_string("thinking.type") != "enabled" && ty->as_string("thinking.type") != "disabled"))
            throw Refused{"'thinking.type' must be enabled or disabled"};
        on_off.emplace_back("thinking.type", ty->as_string("thinking.type") == "enabled");
    }
    if (const Value *v = present(body, "reasoning_effort")) effort(*v, "reasoning_effort");
    if (const Value *v = present(body, "enable_thinking")) on_off.emplace_back("enable_thinking", get_bool(*v, "enable_thinking"));
    if (const Value *k = present(body, "chat_template_kwargs")) {
        if (!k->is_object()) throw Refused{"'chat_template_kwargs' must be an object"};
        for (const auto &[key, v] : k->as_object("chat_template_kwargs")) {
            if (v.is_null()) continue;
            const std::string name = "chat_template_kwargs." + key;
            if (key == "enable_thinking") on_off.emplace_back(name, get_bool(v, name));
            else if (key == "preserve_thinking") r.preserve_thinking = get_bool(v, name);
            else if (key == "reasoning_effort") effort(v, name);
            else if (key == "thinking_mode") {
                if (!v.is_string()) throw Refused{"'" + name + "' must be a string"};
                const std::string m = v.as_string(name);
                if (m == "thinking" || m == "on") on_off.emplace_back(name, true);
                else if (m == "chat" || m == "off" || m == "none") on_off.emplace_back(name, false);
                else if (m != "auto") throw Refused{"'" + name + "' must be auto, thinking, or chat"};
            } else {
                throw Refused{"chat_template_kwargs '" + key + "' is not supported (enable_thinking, thinking_mode, reasoning_effort, preserve_thinking)"};
            }
        }
    }
    for (const auto &[name, on] : on_off)
        if (on != on_off[0].second)
            throw Refused{"thinking on / off disagree: '" + on_off[0].first + "' says " + (on_off[0].second ? "on" : "off") + ", '" + name +
                          "' says " + (on ? "on" : "off")};
    if (!on_off.empty()) r.enabled = on_off[0].second;
    return r;
}

std::string text_of(const Value &content, const std::string &where) {
    if (content.is_null()) return "";
    if (content.is_string()) return content.as_string(where);
    if (!content.is_array()) throw Refused{"'" + where + "' must be a string or an array of text parts"};
    std::string out;
    for (const Value &part : content.as_array(where)) {
        const Value *type = part.is_object() ? part.find("type") : nullptr;
        const Value *text = part.is_object() ? part.find("text") : nullptr;
        if (type == nullptr || !type->is_string() || type->as_string(where) != "text" || text == nullptr || !text->is_string())
            throw Refused{"'" + where + "': only text parts are served"};
        out += text->as_string(where);
    }
    return out;
}

std::vector<tk::ChatMessage> messages_of(const Value &body) {
    const Value *messages = body.find("messages");
    if (messages == nullptr || !messages->is_array()) throw Refused{"'messages' must be an array"};
    std::vector<tk::ChatMessage> out;
    size_t i = 0;
    for (const Value &m : messages->as_array("messages")) {
        const std::string at = "messages[" + std::to_string(i++) + "]";
        if (!m.is_object()) throw Refused{"'" + at + "' must be an object"};
        const Value *role = m.find("role");
        if (role == nullptr || !role->is_string()) throw Refused{"'" + at + ".role' must be a string"};
        const std::string r = role->as_string(at + ".role");
        tk::ChatMessage msg;
        if (r == "system") msg.role = tk::ChatRole::kSystem;
        else if (r == "developer") msg.role = tk::ChatRole::kDeveloper;
        else if (r == "user") msg.role = tk::ChatRole::kUser;
        else if (r == "assistant") msg.role = tk::ChatRole::kAssistant;
        else if (r == "tool") throw Refused{"'" + at + "': tool messages are not served (no tool calling)"};
        else throw Refused{"'" + at + ".role' must be system, developer, user or assistant"};
        if (const Value *c = m.find("content")) msg.content = text_of(*c, at + ".content");
        if (const Value *tc = present(m, "tool_calls"); tc && !(tc->is_array() && tc->as_array(at + ".tool_calls").empty()))
            throw Refused{"'" + at + ".tool_calls': tool calls are not served"};
        if (const Value *rc = present(m, "reasoning_content")) {
            if (r != "assistant" || !rc->is_string()) throw Refused{"'reasoning_content' is only valid as a string on assistant messages"};
            msg.thought = rc->as_string(at + ".reasoning_content");
        }
        if (const Value *name = present(m, "name")) {
            if (!name->is_string()) throw Refused{"'" + at + ".name' must be a string"};
            msg.name = name->as_string(at + ".name");
        }
        out.push_back(std::move(msg));
    }
    if (out.empty()) throw Refused{"'messages' is empty"};
    return out;
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
    auto t = std::make_unique<flerp_text>();
    for (size_t i = 0; i < n_config; ++i) {
        const std::string key = config[i].key ? config[i].key : "";
        const std::string value = config[i].value ? config[i].value : "";
        if (key == "weights") {
            path = value;
        } else if (key == "context") {
            char *end = nullptr;
            const unsigned long v = std::strtoul(value.c_str(), &end, 10);
            if (value.empty() || *end != 0 || v < 16 || v > (1UL << 24)) {
                g_open_error = "flerp_text_open (gufo): context must be a whole number from 16";
                return FLERP_E_CONFIG;
            }
            t->context = (std::uint32_t)v;
        } else {
            g_open_error = "flerp_text_open (gufo): unknown key '" + key + "' (weights, context)";
            return FLERP_E_CONFIG;
        }
    }
    if (out == nullptr || path.empty()) {
        g_open_error = "flerp_text_open (gufo): weights (the GGUF file) is required";
        return FLERP_E_CONFIG;
    }
    try {
        std::string error;
        const std::unique_ptr<gufo::core::GgufReader> reader = gufo::core::GgufReader::OpenFile(path, &error);
        if (!reader) throw std::runtime_error("GGUF " + path + ": " + error);
        if (!tk::QwenChatTemplate::ValidateGgufTemplate(*reader, &error)) throw std::runtime_error("chat template: " + error);
        t->tok = tk::QwenTokenizer::CreateFromGguf(*reader, &error);
        if (!t->tok) throw std::runtime_error("tokenizer: " + error);
        std::string tmpl = "no chat template in the file";
        std::string tmpl_error;
        if (const auto file_template = tk::QwenChatTemplate::CreateFromGguf(*reader, &tmpl_error))
            tmpl = "the file's template " + std::string(file_template->GetTemplateId()) + " sha256 " + std::string(file_template->GetTemplateSha256());
        // the ids a reply ends at: the tokenizer's stop tokens (QwenTokenizer::IsStopToken: eos and <|endoftext|>),
        // the generator's stop rule
        const flerp_token eos = (flerp_token)t->tok->GetEosTokenId();
        if (t->tok->IsStopToken((tk::TokenId)eos)) t->end_ids.push_back(eos);
        if (const auto eot = t->tok->FindSpecialToken("<|endoftext|>"); eot && t->tok->IsStopToken(*eot) && (flerp_token)*eot != eos)
            t->end_ids.push_back((flerp_token)*eot);
        if (t->end_ids.empty()) throw std::runtime_error("the tokenizer has no stop token");
        t->describe = "gufo Qwen tokenizer (" + std::to_string(t->tok->GetVocabSize()) + " tokens) and compiled Qwen ChatML formatter; " + tmpl +
                      "; " + path;
    } catch (const std::exception &e) {
        g_open_error = std::string("flerp_text_open (gufo): ") + e.what();
        return FLERP_E_FAILED;
    }
    *out = t.release();
    return FLERP_OK;
}

void flerp_text_close(flerp_text *t) { delete t; }
const char *flerp_text_last_error(const flerp_text *t) { return t ? t->error.c_str() : g_open_error.c_str(); }
const char *flerp_text_describe(const flerp_text *t) { return t->describe.c_str(); }
int64_t flerp_text_vocab(const flerp_text *t) { return (int64_t)t->tok->GetVocabSize(); }

int64_t flerp_text_encode(flerp_text *t, const char *text, size_t n, const size_t *plain, size_t n_plain, flerp_token *out, size_t cap) {
    return guarded(t, [&]() -> int64_t {
        std::vector<tk::ContentSpan> spans(n_plain);
        for (size_t i = 0; i < n_plain; ++i) {
            if (plain[2 * i + 1] < plain[2 * i] || plain[2 * i + 1] > n) throw std::invalid_argument("a plain range outside the text");
            spans[i] = {plain[2 * i], plain[2 * i + 1] - plain[2 * i]};
        }
        const std::string_view sv(text, n);
        const std::vector<tk::TokenId> ids = tk::QwenChatTemplate::EncodeRendered(*t->tok, sv, 0, n, spans, {});
        if (ids.size() <= cap && out != nullptr)
            for (size_t i = 0; i < ids.size(); ++i) out[i] = (flerp_token)ids[i];
        return (int64_t)ids.size();
    });
}

int64_t flerp_text_token_bytes(flerp_text *t, flerp_token id, char *out, size_t cap) {
    return guarded(t, [&]() -> int64_t {
        if (id < 0 || (size_t)id >= t->tok->GetVocabSize()) throw std::invalid_argument("token id " + std::to_string(id) + " is outside the vocabulary");
        const std::string b = t->tok->DecodeTokenCopy((tk::TokenId)id);
        if (b.size() <= cap && out != nullptr) std::memcpy(out, b.data(), b.size());
        return (int64_t)b.size();
    });
}

int64_t flerp_text_end_ids(const flerp_text *t, flerp_token *out, size_t cap) {
    if (t->end_ids.size() <= cap && out != nullptr) std::memcpy(out, t->end_ids.data(), t->end_ids.size() * sizeof(flerp_token));
    return (int64_t)t->end_ids.size();
}

flerp_token flerp_text_token_id(const flerp_text *t, const char *piece) {
    const auto id = t->tok->FindSpecialToken(piece);
    return id ? (flerp_token)*id : -1;
}

flerp_status flerp_text_render_chat(flerp_text *t, const char *request, size_t n, flerp_text_chat *out) {
    if (out == nullptr || out->size < sizeof(flerp_text_chat)) {
        t->error = "flerp_text_chat is null or short";
        return FLERP_E_ARGUMENT;
    }
    try {
        const Value body = Value::parse(std::string_view(request, n));
        const std::vector<tk::ChatMessage> messages = messages_of(body);
        tk::ChatTemplateOptions options = tk::ResolveQwenChatOptions(reasoning_of(body));
        options.max_output_bytes = tk::RenderedPromptBoundBytes(t->context);
        std::vector<tk::ContentSpan> spans;
        std::string error;
        const std::optional<std::string> rendered = tk::QwenChatTemplate::Render(messages, {}, options, &error, nullptr, nullptr, &spans);
        if (!rendered) throw Refused{"the chat template: " + error};
        t->rendered = *rendered;
        t->plain.clear();
        for (const tk::ContentSpan &s : spans) t->plain.push_back(s.offset), t->plain.push_back(s.offset + s.size);
    } catch (const Refused &r) {
        t->error = r.what;
        return FLERP_E_ARGUMENT;
    } catch (const std::exception &e) {  // JSON type errors: the client's request
        t->error = e.what();
        return FLERP_E_ARGUMENT;
    }
    out->text = t->rendered.data(), out->n_text = t->rendered.size();
    out->plain = t->plain.data(), out->n_plain = t->plain.size() / 2;
    return FLERP_OK;
}

}  // extern "C"
