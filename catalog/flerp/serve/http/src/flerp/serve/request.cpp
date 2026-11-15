#include "flerp/serve/request.hpp"

#include <algorithm>
#include <cmath>

namespace flerp_serve {

using strix::json::Value;

std::string error_body(const ApiError &e) {
    Value err = Value::object();
    err.set("message", Value::string(e.message));
    err.set("type", Value::string(e.status >= 500 ? "server_error" : "invalid_request_error"));
    err.set("param", e.param.empty() ? Value::null() : Value::string(e.param));
    err.set("code", e.code.empty() ? Value::null() : Value::string(e.code));
    Value out = Value::object();
    out.set("error", std::move(err));
    return out.dump();
}

namespace {

[[noreturn]] void bad(const std::string &param, const std::string &message) { throw ApiError{400, message, param, ""}; }
// A feature strixflerp's origins serve and this front end does not (README.md, "The front end").
[[noreturn]] void not_served(const std::string &param, const std::string &what) {
    throw ApiError{400, what + " is not served by this front end (serve.flerp.http)", param, "not_served"};
}
// A request the executor of this recipe cannot honour, by its capability flags.
[[noreturn]] void cannot(const std::string &param, const std::string &why) {
    throw ApiError{400, "'" + param + "' cannot be honoured by this recipe's executor: " + why, param, "executor_cannot"};
}

const Value *field(const Value &body, const char *key) {
    const Value *v = body.find(key);
    return v != nullptr && !v->is_null() ? v : nullptr;
}

double number(const Value &v, const char *name) {
    if (!v.is_number()) bad(name, std::string("'") + name + "' must be a number");
    return v.as_double(name);
}

int64_t integer(const Value &v, const char *name) {
    if (!v.is_int()) bad(name, std::string("'") + name + "' must be an integer");
    return v.as_int(name);
}

bool boolean(const Value &v, const char *name) {
    if (!v.is_bool()) bad(name, std::string("'") + name + "' must be true or false");
    return v.as_bool(name);
}

bool truthy(const Value &v) { return !(v.is_null() || (v.is_bool() && !v.as_bool("")) || (v.is_int() && v.as_int("") == 0)); }

// strixite's request parser (serve/openai.cpp check_tools at d19945e): the shape the template and the reply parser read.
void check_tools(const Value &tools) {
    if (!tools.is_array()) bad("tools", "'tools' must be an array");
    const auto &ts = tools.as_array("tools");
    for (size_t k = 0; k < ts.size(); ++k) {
        const std::string p = "tools[" + std::to_string(k) + "]";
        const Value &t = ts[k];
        if (!t.is_object()) bad(p, "'" + p + "' must be an object");
        const Value *type = t.find("type");
        if (!type || !type->is_string() || type->as_string("type") != "function") bad(p + ".type", "'" + p + ".type' must be \"function\"");
        const Value *fn = t.find("function");
        if (!fn || !fn->is_object()) bad(p + ".function", "'" + p + ".function' must be an object");
        const Value *name = fn->find("name");
        if (!name || !name->is_string() || name->as_string("name").empty())
            bad(p + ".function.name", "'" + p + ".function.name' must be a non-empty string");
        const Value *params = fn->find("parameters");
        if (params && !params->is_null() && !params->is_object())
            bad(p + ".function.parameters", "'" + p + ".function.parameters' must be an object");
    }
}

// Tools, response_format and thinking, read as strixite's request parser reads them (serve/openai.cpp).
void read_reply_fields(const Value &body, Request &r) {
    if (const Value *v = field(body, "response_format")) {
        if (!v->is_object()) bad("response_format", "'response_format' must be an object");
        const Value *type = v->find("type");
        const std::string kind = type && type->is_string() ? type->as_string("type") : std::string("(none)");
        if (kind == "json_object") {
            r.format = Request::Format::JsonObject;
        } else if (kind == "json_schema") {
            const Value *spec = v->find("json_schema");
            if (!spec || !spec->is_object())
                bad("response_format.json_schema", "response_format type 'json_schema' needs a 'json_schema' object ({\"name\": ..., \"schema\": {...}})");
            const Value *schema = spec->find("schema");
            if (!schema || !(schema->is_object() || schema->is_bool()))
                bad("response_format.json_schema.schema", "response_format.json_schema needs a 'schema' (an object)");
            if (const Value *strict = spec->find("strict"); strict && !strict->is_null()) boolean(*strict, "response_format.json_schema.strict");
            if (const Value *name = spec->find("name"); name && !name->is_null()) {
                if (!name->is_string()) bad("response_format.json_schema.name", "'response_format.json_schema.name' must be a string");
                r.schema_name = name->as_string("response_format.json_schema.name");
            }
            r.format = Request::Format::JsonSchema;
            r.schema = schema->dump();
        } else if (kind != "text") {
            bad("response_format", "response_format type '" + kind + "' is not supported (text, json_object, json_schema)");
        }
    }
    std::string tool_choice = "auto";
    if (const Value *v = field(body, "tool_choice")) {
        if (!v->is_string() || (v->as_string("tool_choice") != "auto" && v->as_string("tool_choice") != "none"))
            not_served("tool_choice", "tool_choice " + (v->is_string() ? "'" + v->as_string("tool_choice") + "'" : std::string("(a named tool)")) +
                                          " (tool_choice \"auto\" and \"none\" are served)");
        tool_choice = v->as_string("tool_choice");
    }
    if (const Value *v = field(body, "tools")) {
        check_tools(*v);
        if (tool_choice == "auto" && !v->as_array("tools").empty()) r.tools = v->dump();
    }
    r.tools_dropped = field(body, "tools") != nullptr && r.tools.empty();
    if (r.format != Request::Format::Text && !r.tools.empty())
        bad("response_format", "response_format json_object / json_schema together with tools is not supported (the "
                               "answer is either a JSON document or tool calls) - send no tools, or tool_choice \"none\"");
    // the thinking budget: four spellings; two that disagree are refused
    std::vector<std::pair<std::string, int64_t>> budgets;
    const auto take = [&](const Value *v, const std::string &name) {
        if (!v || v->is_null()) return;
        const int64_t b = integer(*v, name.c_str());
        if (b < 0) bad(name, "'" + name + "' must be >= 0, got " + std::to_string(b));
        budgets.emplace_back(name, b);
    };
    take(field(body, "max_thinking_tokens"), "max_thinking_tokens");
    if (const Value *o = field(body, "reasoning"); o && o->is_object()) take(o->find("max_tokens"), "reasoning.max_tokens");
    if (const Value *o = field(body, "thinking"); o && o->is_object()) take(o->find("budget_tokens"), "thinking.budget_tokens");
    if (const Value *o = field(body, "chat_template_kwargs"); o && o->is_object()) take(o->find("thinking_budget"), "chat_template_kwargs.thinking_budget");
    for (const auto &[name, b] : budgets)
        if (b != budgets[0].second)
            bad(name, "thinking budgets disagree: '" + budgets[0].first + "' = " + std::to_string(budgets[0].second) + ", '" + name + "' = " + std::to_string(b));
    if (!budgets.empty()) r.thinking_budget = budgets[0].second;
    // thinking on / off (the text module refuses spellings that disagree) and whether the request asked for thinking
    const auto on_off = [&](bool on) {
        if (!r.thinking_on.has_value()) r.thinking_on = on;
    };
    const auto effort = [&](const Value &v) {
        if (v.is_string() && v.as_string("reasoning_effort") == "none") on_off(false);
        else r.effort_asked = true;
    };
    if (const Value *v = field(body, "reasoning_effort")) effort(*v);
    if (const Value *v = field(body, "enable_thinking"); v && v->is_bool()) on_off(v->as_bool("enable_thinking"));
    if (const Value *o = field(body, "thinking"); o && o->is_object())
        if (const Value *t = o->find("type"); t && t->is_string()) on_off(t->as_string("thinking.type") == "enabled");
    if (const Value *k = field(body, "chat_template_kwargs"); k && k->is_object()) {
        if (const Value *v = k->find("enable_thinking"); v && v->is_bool()) on_off(v->as_bool("enable_thinking"));
        if (const Value *v = k->find("reasoning_effort"); v && !v->is_null()) effort(*v);
    }
}

// Fields of features outside this front end's scope: named so the client hears why, not ignored.
void refuse_out_of_scope(const Value &body, bool chat) {
    for (const char *key : {"grammar", "json_schema", "guided_json", "guided_grammar", "guided_regex", "guided_choice", "structured_outputs"})
        if (field(body, key)) not_served(key, std::string("structured output ('") + key + "')");
    for (const char *key : {"functions", "function_call"})
        if (field(body, key)) not_served(key, "tool calling with the legacy 'functions' fields (send 'tools')");
    if (field(body, "steer")) not_served("steer", "activation steering");
    if (const Value *v = field(body, "cache_prompt"); v && truthy(*v)) not_served("cache_prompt", "the prompt cache");
    if (!chat) {
        if (field(body, "response_format")) not_served("response_format", "response_format on /v1/completions");
        if (field(body, "tools")) not_served("tools", "tools on /v1/completions");
    }
    if (const Value *v = field(body, "n"); v && integer(*v, "n") != 1) not_served("n", "more than one choice (n)");
    if (!chat) {
        if (const Value *v = field(body, "echo"); v && truthy(*v)) not_served("echo", "echo");
        if (field(body, "suffix")) not_served("suffix", "suffix");
        if (const Value *v = field(body, "best_of"); v && integer(*v, "best_of") != 1) not_served("best_of", "best_of");
        return;
    }
    // vision: any content part that is not text
    const Value *messages = body.find("messages");
    if (messages == nullptr || !messages->is_array()) bad("messages", "'messages' must be an array");
    size_t i = 0;
    for (const Value &m : messages->as_array("messages")) {
        const Value *content = m.is_object() ? m.find("content") : nullptr;
        if (content != nullptr && content->is_array())
            for (const Value &part : content->as_array("content")) {
                const Value *type = part.is_object() ? part.find("type") : nullptr;
                if (type == nullptr || !type->is_string() || type->as_string("type") != "text")
                    not_served("messages[" + std::to_string(i) + "].content",
                               "vision and other media (a content part of type '" + (type && type->is_string() ? type->as_string("type") : std::string("?")) + "')");
            }
        ++i;
    }
}

}  // namespace

Request parse_request(const Value &body, bool chat, const Facts &f) {
    if (!body.is_object()) bad("", "the request body must be a JSON object");
    refuse_out_of_scope(body, chat);
    Request r;
    r.chat = chat;
    r.sampling = f.defaults;
    r.sampling.size = sizeof(flerp_sampling);
    if (const Value *v = field(body, "model"); v && v->is_string()) r.model = v->as_string("model");
    if (chat) {
        read_reply_fields(body, r);
        r.body = body.dump();
    } else {
        const Value *p = body.find("prompt");
        if (p == nullptr || !p->is_string()) bad("prompt", "'prompt' must be a string (token ids and batches of prompts are not served)");
        r.prompt = p->as_string("prompt");
    }
    if (const Value *v = field(body, "stream")) r.stream = boolean(*v, "stream");
    if (const Value *o = field(body, "stream_options"); o && o->is_object())
        if (const Value *v = o->find("include_usage"); v && !v->is_null()) r.include_usage = boolean(*v, "stream_options.include_usage");
    if (const Value *v = field(body, "flerp_trace")) r.trace = boolean(*v, "flerp_trace");
    for (const char *key : {"max_tokens", "max_completion_tokens"})
        if (const Value *v = field(body, key)) {
            const int64_t n = integer(*v, key);
            if (n < 1) bad(key, std::string("'") + key + "' must be >= 1");
            if (r.max_tokens && *r.max_tokens != n) bad(key, "'max_tokens' and 'max_completion_tokens' disagree");
            r.max_tokens = n;
        }
    if (const Value *v = field(body, "temperature")) r.sampling.temperature = number(*v, "temperature");
    if (const Value *v = field(body, "top_p")) r.sampling.top_p = number(*v, "top_p");
    if (const Value *v = field(body, "min_p")) r.sampling.min_p = number(*v, "min_p");
    if (const Value *v = field(body, "top_k")) r.sampling.top_k = (int32_t)std::clamp<int64_t>(integer(*v, "top_k"), -1, INT32_MAX);
    if (const Value *v = field(body, "seed")) {
        const int64_t s = integer(*v, "seed");
        if (s < 0) bad("seed", "'seed' must be >= 0");
        r.sampling.seed = (uint64_t)s, r.seeded = true;
    }
    const flerp_sampling &s = r.sampling;
    if (!std::isfinite(s.temperature) || s.temperature < 0 || s.temperature > 100) bad("temperature", "'temperature' must be in [0, 100]");
    if (!std::isfinite(s.top_p) || s.top_p <= 0 || s.top_p > 1) bad("top_p", "'top_p' must be in (0, 1]");
    if (!std::isfinite(s.min_p) || s.min_p < 0 || s.min_p >= 1) bad("min_p", "'min_p' must be in [0, 1)");
    if (s.top_k < 0) bad("top_k", "'top_k' must be >= 0 (0 = off)");
    if (const Value *v = field(body, "stop")) {
        if (v->is_string()) r.stop.push_back(v->as_string("stop"));
        else if (v->is_array())
            for (const Value &x : v->as_array("stop")) {
                if (!x.is_string()) bad("stop", "'stop' must be a string or an array of strings");
                r.stop.push_back(x.as_string("stop"));
            }
        else bad("stop", "'stop' must be a string or an array of strings");
        if (r.stop.size() > 16) bad("stop", "'stop' takes at most 16 strings");
        for (const std::string &x : r.stop)
            if (x.empty()) bad("stop", "'stop' strings must not be empty");
    }
    // logit_bias: OpenAI's {"<token id>": bias in [-100, 100]}
    if (const Value *v = field(body, "logit_bias")) {
        if (!v->is_object()) bad("logit_bias", "'logit_bias' must be an object of token id -> bias");
        std::vector<std::pair<int64_t, float>> entries;
        for (const auto &[key, b] : v->as_object("logit_bias")) {
            char *end = nullptr;
            const long long id = std::strtoll(key.c_str(), &end, 10);
            if (key.empty() || *end != 0 || id < 0 || id >= f.vocab)
                bad("logit_bias", "logit_bias key '" + key + "' is not a token id in [0, " + std::to_string(f.vocab) + ")");
            const double x = number(b, "logit_bias");
            if (!std::isfinite(x) || std::fabs(x) > 100) bad("logit_bias", "logit_bias of token " + key + " must be in [-100, 100]");
            if (x != 0) entries.emplace_back(id, (float)x);
        }
        std::sort(entries.begin(), entries.end());
        for (size_t i = 1; i < entries.size(); ++i)
            if (entries[i].first == entries[i - 1].first) bad("logit_bias", "logit_bias names token " + std::to_string(entries[i].first) + " twice");
        for (const auto &[id, x] : entries) r.bias_id.push_back((flerp_token)id), r.bias.push_back(x);
    }
    if (const Value *v = field(body, "logprobs")) {
        if (chat) r.logprobs = boolean(*v, "logprobs");
        else if (!(v->is_int() && v->as_int("logprobs") == 0) && !(v->is_bool() && !v->as_bool("logprobs")))
            not_served("logprobs", "logprobs on /v1/completions (ask /v1/chat/completions)");
    }
    if (const Value *v = field(body, "top_logprobs")) {
        const int64_t n = integer(*v, "top_logprobs");
        if (n < 0 || n > 20) bad("top_logprobs", "'top_logprobs' must be in [0, 20]");
        if (!r.logprobs && n > 0) bad("top_logprobs", "'top_logprobs' needs \"logprobs\": true");
        r.top_logprobs = (int)n;
    }
    if (r.logprobs && r.stream) not_served("logprobs", "logprobs on a streamed response");
    if (chat && !f.reply) {  // no reply module: nothing splits the reply, so nothing that needs the split is served
        if (!r.tools.empty()) not_served("tools", "tool calling with this recipe (it binds no reply module)");
        if (r.thinking_budget) not_served("max_thinking_tokens", "a thinking budget with this recipe (it binds no reply module)");
        if (r.format != Request::Format::Text) not_served("response_format", "structured output with this recipe (it binds no reply module)");
    }
    if (r.format != Request::Format::Text && !(f.caps.flags & FLERP_CAP_MASKS))
        cannot("response_format", "it takes no token masks (no FLERP_CAP_MASKS)");
    if (r.format != Request::Format::Text && !f.host_sampling && s.temperature > 0)
        cannot("response_format", "its tokens are drawn on the device and a constraint rides on rows the host samples: send temperature 0");

    // What this recipe's executor can honour, from its capability flags.
    const flerp_caps &c = f.caps;
    if (!r.bias_id.empty() && !(c.flags & FLERP_CAP_MASKS))
        cannot("logit_bias", "it takes no logit bias (no FLERP_CAP_MASKS)");
    if (r.logprobs && !(f.host_sampling && (c.deliveries & FLERP_OUT_FULL)))
        cannot("logprobs", f.host_sampling ? "it hands no full logits rows to the host (no FLERP_OUT_FULL)"
                                           : "it hands no logits to the host: its tokens are drawn on the device");
    if (!f.host_sampling && s.temperature > 0) {
        if (!(c.flags & FLERP_CAP_SAMPLED_TOKENS))
            cannot("temperature", "it draws greedy tokens only (no FLERP_CAP_SAMPLED_TOKENS): send temperature 0");
        if (f.caps_v2) {
            if (s.top_k == 0 && (c.sampling & FLERP_SAMPLE_NEEDS_TOP_K))
                cannot("top_k", "it samples only with top_k set" + (c.top_k_max > 0 ? ", from 1 to " + std::to_string(c.top_k_max) : std::string()) +
                                    ": send top_k, or temperature 0");
            if (s.top_k > 0 && !(c.sampling & FLERP_SAMPLE_TOP_K)) cannot("top_k", "its sampler has no top_k");
            if (c.top_k_max > 0 && s.top_k > c.top_k_max) cannot("top_k", "its largest top_k is " + std::to_string(c.top_k_max));
            if (s.top_p < 1 && !(c.sampling & FLERP_SAMPLE_TOP_P)) cannot("top_p", "its sampler has no top_p");
            if (s.min_p > 0 && !(c.sampling & FLERP_SAMPLE_MIN_P)) cannot("min_p", "its sampler has no min_p");
            if (r.seeded && !(c.sampling & FLERP_SAMPLE_SEED)) cannot("seed", "its draws do not follow a seed");
        }
    }
    return r;
}

}  // namespace flerp_serve
