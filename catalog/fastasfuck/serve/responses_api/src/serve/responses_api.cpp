#include "serve/responses_api.hpp"

#include "common/check.hpp"
#include "serve/chat_template.hpp"
#include "serve/openai.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace strix {

namespace {

[[noreturn]] void bad_req(const std::string &param, const std::string &msg, const std::string &code = {}) {
    throw ApiError(400, msg, param, "invalid_request_error", code);
}

const json::Value *find_field(const json::Value &body, const char *name) {
    const json::Value *v = body.find(name);
    return v && !v->is_null() ? v : nullptr;
}

std::string get_str(const json::Value &v, const char *name) {
    if (!v.is_string()) bad_req(name, std::string("'") + name + "' must be a string");
    return v.as_string(name);
}

int64_t get_integer(const json::Value &v, const char *name) {
    if (v.is_number() && !v.is_int()) {
        const double d = v.as_double(name);
        if (d == std::floor(d) && std::fabs(d) < 9e15) return (int64_t)d;
    }
    if (!v.is_int()) bad_req(name, std::string("'") + name + "' must be an integer");
    return v.as_int(name);
}

double get_flt(const json::Value &v, const char *name) {
    if (!v.is_number()) bad_req(name, std::string("'") + name + "' must be a number");
    return v.as_double(name);
}

bool get_boolean(const json::Value &v, const char *name) {
    if (!v.is_bool()) bad_req(name, std::string("'") + name + "' must be a boolean");
    return v.as_bool(name);
}

// Responses API supports flat tools: {"type": "function", "name": "foo", "description": "bar", "parameters": {...}}
// Wrap into nested {"type": "function", "function": {...}} if flat, preserving backward compatibility with nested.
json::Value normalize_tools_for_render(const json::Value &tools) {
    if (!tools.is_array()) return json::Value();
    json::Value norm = json::Value::array();
    for (const json::Value &t : tools.as_array("tools")) {
        if (!t.is_object()) continue;
        if (t.find("function") && t.find("function")->is_object()) {
            norm.push(t);
        } else if (t.find("name") && t.find("name")->is_string()) {
            json::Value wrapped = json::Value::object();
            wrapped.set("type", json::Value::string("function"));
            json::Value fn = json::Value::object();
            fn.set("name", json::Value::string(t.find("name")->as_string("name")));
            if (const json::Value *d = t.find("description"); d && d->is_string())
                fn.set("description", json::Value::string(d->as_string("description")));
            if (const json::Value *p = t.find("parameters"); p && p->is_object())
                fn.set("parameters", *p);
            wrapped.set("function", std::move(fn));
            norm.push(std::move(wrapped));
        } else {
            norm.push(t);
        }
    }
    return norm;
}

}  // namespace

uint64_t compute_responses_config_hash(const std::optional<std::string> &instructions, const json::Value &tools) {
    std::string s = instructions.value_or("");
    s += "|";
    if (tools.is_array()) s += tools.dump();
    uint64_t h = 14695981039346656037ull;
    for (uint8_t c : s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

ResponsesRequest parse_responses_request(const json::Value &body, const SamplingParams &defaults) {
    if (!body.is_object()) bad_req("", "Request body must be a JSON object");
    ResponsesRequest r;
    r.sampling = defaults;

    if (const json::Value *m = find_field(body, "model")) r.model = get_str(*m, "model");
    if (const json::Value *ins = find_field(body, "instructions")) r.instructions = get_str(*ins, "instructions");
    if (const json::Value *prev = find_field(body, "previous_response_id")) r.previous_response_id = get_str(*prev, "previous_response_id");
    if (const json::Value *st = find_field(body, "stream")) r.stream = get_boolean(*st, "stream");
    // store: false in Responses API does NOT disable in-memory continuation; we always allow ResponseStore chaining
    if (const json::Value *store = find_field(body, "store")) r.store = get_boolean(*store, "store");

    const json::Value *mot = find_field(body, "max_output_tokens");
    if (!mot) mot = find_field(body, "max_tokens");
    if (mot) {
        r.max_output_tokens = get_integer(*mot, "max_output_tokens");
        if (*r.max_output_tokens < 1) bad_req("max_output_tokens", "'max_output_tokens' must be >= 1");
    }

    if (const json::Value *v = find_field(body, "temperature")) {
        r.sampling.temperature = get_flt(*v, "temperature");
        if (!(r.sampling.temperature >= 0 && r.sampling.temperature <= 2)) bad_req("temperature", "'temperature' must be in [0, 2]");
    }
    if (const json::Value *v = find_field(body, "top_p")) {
        r.sampling.top_p = get_flt(*v, "top_p");
        if (!(r.sampling.top_p > 0 && r.sampling.top_p <= 1)) bad_req("top_p", "'top_p' must be in (0, 1]");
    }
    if (const json::Value *v = find_field(body, "top_k")) {
        r.sampling.top_k = get_integer(*v, "top_k");
        if (r.sampling.top_k < 0) bad_req("top_k", "'top_k' must be >= 0");
    }
    if (const json::Value *v = find_field(body, "seed")) r.sampling.seed = (uint64_t)get_integer(*v, "seed");

    // Reasoning effort mapping: reasoning: {"effort": "none" | "low" | "medium" | "high"}
    if (const json::Value *re = find_field(body, "reasoning")) {
        if (re->is_object()) {
            if (const json::Value *eff = find_field(*re, "effort"); eff && eff->is_string()) {
                const std::string effort_str = eff->as_string("reasoning.effort");
                if (effort_str == "none") {
                    r.template_options.enable_thinking = false;
                } else if (effort_str == "low" || effort_str == "medium" || effort_str == "high" || effort_str == "xhigh") {
                    r.template_options.enable_thinking = true;
                    r.template_options.reasoning_effort = (effort_str == "high") ? "xhigh" : effort_str;
                }
            }
        }
    }
    if (const json::Value *reff = find_field(body, "reasoning_effort"); reff && reff->is_string()) {
        const std::string effort_str = reff->as_string("reasoning_effort");
        if (effort_str == "none") r.template_options.enable_thinking = false;
        else if (effort_str == "low" || effort_str == "medium" || effort_str == "high" || effort_str == "xhigh") {
            r.template_options.enable_thinking = true;
            r.template_options.reasoning_effort = (effort_str == "high") ? "xhigh" : effort_str;
        }
    }

    // Tools (flat or nested)
    if (const json::Value *tc = find_field(body, "tool_choice")) {
        if (tc->is_string()) {
            r.tool_choice = tc->as_string("tool_choice");
        } else if (tc->is_object()) {
            if (const json::Value *name = tc->find("name"); name && name->is_string()) {
                r.tool_choice = name->as_string("tool_choice.name");
            } else if (const json::Value *fn = tc->find("function"); fn && fn->is_object()) {
                if (const json::Value *name = fn->find("name"); name && name->is_string()) {
                    r.tool_choice = name->as_string("tool_choice.function.name");
                }
            }
        }
    }
    if (const json::Value *t = find_field(body, "tools")) {
        if (!t->is_array()) bad_req("tools", "'tools' must be an array");
        r.tools = normalize_tools_for_render(*t);
    }

    // Input parsing: string or array of items
    const json::Value *inp = find_field(body, "input");
    if (!inp) bad_req("input", "'input' is required");
    if (inp->is_string()) {
        ResponsesInputItem item;
        item.kind = ResponsesInputItem::Kind::Message;
        item.role = "user";
        item.content = inp->as_string("input");
        r.input_items.push_back(std::move(item));
    } else if (inp->is_array()) {
        for (const json::Value &val : inp->as_array("input")) {
            if (!val.is_object()) bad_req("input", "input items must be objects");
            std::string type = "message";
            if (const json::Value *tp = find_field(val, "type")) type = get_str(*tp, "type");

            if (type == "message") {
                ResponsesInputItem item;
                item.kind = ResponsesInputItem::Kind::Message;
                if (const json::Value *rl = find_field(val, "role")) item.role = get_str(*rl, "role");
                if (const json::Value *c = find_field(val, "content")) {
                    if (c->is_string()) {
                        item.content = c->as_string("content");
                    } else if (c->is_array()) {
                        for (const json::Value &p : c->as_array("content")) {
                            if (p.is_object()) {
                                if (const json::Value *txt = find_field(p, "text")) item.content += txt->as_string("text");
                            }
                        }
                    }
                }
                r.input_items.push_back(std::move(item));
            } else if (type == "function_call") {
                ResponsesInputItem item;
                item.kind = ResponsesInputItem::Kind::FunctionCall;
                if (const json::Value *cid = find_field(val, "call_id"); cid && cid->is_string()) {
                    item.call_id = cid->as_string("call_id");
                } else if (const json::Value *id = find_field(val, "id"); id && id->is_string()) {
                    item.call_id = id->as_string("id");
                }
                if (const json::Value *nm = find_field(val, "name")) item.name = get_str(*nm, "name");
                if (const json::Value *arg = find_field(val, "arguments")) {
                    if (arg->is_string()) item.arguments = arg->as_string("arguments");
                    else if (arg->is_object()) item.arguments = arg->dump();
                }
                r.input_items.push_back(std::move(item));
            } else if (type == "function_call_output") {
                ResponsesInputItem item;
                item.kind = ResponsesInputItem::Kind::FunctionCallOutput;
                if (const json::Value *cid = find_field(val, "call_id"); cid && cid->is_string()) {
                    item.call_id = cid->as_string("call_id");
                } else if (const json::Value *id = find_field(val, "id"); id && id->is_string()) {
                    item.call_id = id->as_string("id");
                }
                if (const json::Value *out = find_field(val, "output")) {
                    if (out->is_string()) {
                        item.output = out->as_string("output");
                    } else if (out->is_array()) {
                        for (const json::Value &p : out->as_array("output")) {
                            if (p.is_object()) {
                                if (const json::Value *txt = find_field(p, "text"); txt && txt->is_string()) {
                                    item.output += txt->as_string("text");
                                }
                            }
                        }
                    }
                }
                r.input_items.push_back(std::move(item));
            } else if (type == "reasoning") {
                // OpenAI Responses API reasoning items from prior turns: safely ignore
                continue;
            }
        }
    } else {
        bad_req("input", "'input' must be a string or array of items");
    }

    return r;
}

std::vector<int32_t> build_responses_prompt(
    ResponsesRequest &req,
    ResponseStore &store,
    const Tokenizer &tok,
    bool &is_continuation,
    int64_t &prefix_len
) {
    std::vector<std::pair<size_t, size_t>> plain_spans;

    if (req.previous_response_id.empty()) {
        // Cold session: render full prompt from scratch
        json::Value messages = json::Value::array();
        std::vector<json::Value> msg_list;
        if (req.instructions.has_value() && !req.instructions->empty()) {
            json::Value sys = json::Value::object();
            sys.set("role", json::Value::string("system"));
            sys.set("content", json::Value::string(*req.instructions));
            msg_list.push_back(std::move(sys));
        }

        for (const auto &item : req.input_items) {
            if (item.kind == ResponsesInputItem::Kind::Message) {
                json::Value m = json::Value::object();
                std::string role = item.role;
                if (role == "developer") role = "system";
                m.set("role", json::Value::string(role));
                m.set("content", json::Value::string(item.content));
                msg_list.push_back(std::move(m));
            } else if (item.kind == ResponsesInputItem::Kind::FunctionCall) {
                json::Value tc = json::Value::object();
                tc.set("id", json::Value::string(item.call_id));
                tc.set("type", json::Value::string("function"));
                json::Value fn = json::Value::object();
                fn.set("name", json::Value::string(item.name));
                fn.set("arguments", json::Value::string(item.arguments));
                tc.set("function", std::move(fn));

                bool attached = false;
                if (!msg_list.empty()) {
                    auto *last_role = msg_list.back().find("role");
                    if (last_role && last_role->as_string("role") == "assistant") {
                        auto *calls = msg_list.back().find("tool_calls");
                        if (calls && calls->is_array()) {
                            calls->push(std::move(tc));
                        } else {
                            json::Value arr = json::Value::array();
                            arr.push(std::move(tc));
                            msg_list.back().set("tool_calls", std::move(arr));
                        }
                        attached = true;
                    }
                }
                if (!attached) {
                    json::Value asst = json::Value::object();
                    asst.set("role", json::Value::string("assistant"));
                    asst.set("content", json::Value::string(""));
                    json::Value arr = json::Value::array();
                    arr.push(std::move(tc));
                    asst.set("tool_calls", std::move(arr));
                    msg_list.push_back(std::move(asst));
                }
            } else if (item.kind == ResponsesInputItem::Kind::FunctionCallOutput) {
                json::Value m = json::Value::object();
                m.set("role", json::Value::string("tool"));
                m.set("content", json::Value::string(item.output));
                m.set("tool_call_id", json::Value::string(item.call_id));
                msg_list.push_back(std::move(m));
            }
        }

        for (auto &m : msg_list) {
            messages.push(std::move(m));
        }

        const json::Value *tools_ptr = req.tools.is_array() && !req.tools.as_array("tools").empty() ? &req.tools : nullptr;
        std::string rendered = render_chat(messages, tools_ptr, req.template_options, &plain_spans, false);
        is_continuation = false;
        prefix_len = 0;
        return tok.encode(rendered, plain_spans);
    }

    // Continuation with previous_response_id
    auto prev = store.get(req.previous_response_id);
    if (!prev.has_value()) {
        bad_req("previous_response_id", "Previous response '" + req.previous_response_id + "' not found or evicted from memory.",
                "previous_response_not_found");
    }

    // Inherit instructions and tools if omitted in new turn
    if (!req.instructions.has_value() && prev->instructions.has_value()) req.instructions = prev->instructions;
    if (req.tools.is_null() && prev->tools.has_value()) req.tools = *prev->tools;

    const uint64_t cur_hash = compute_responses_config_hash(req.instructions, req.tools);
    if (prev->config_hash != 0 && cur_hash != prev->config_hash) {
        bad_req("previous_response_id", "Configuration (instructions/tools) changed since previous response.",
                "previous_response_not_found");
    }

    // Splicing: previous turn tokens + delta
    std::string delta_str;
    const int32_t im_end_id = tok.id_of("<|im_end|>");
    if (prev->tokens.empty() || prev->tokens.back() != im_end_id) {
        delta_str += "<|im_end|>\n";
    }

    // Handle tool results: group consecutive tool outputs inside one user turn
    std::vector<const ResponsesInputItem *> tool_outputs;
    for (const auto &item : req.input_items) {
        if (item.kind == ResponsesInputItem::Kind::FunctionCallOutput) tool_outputs.push_back(&item);
    }

    if (!tool_outputs.empty()) {
        delta_str += "<|im_start|>user";
        for (const auto *item : tool_outputs) {
            delta_str += "\n<tool_response>\n";
            const size_t s = delta_str.size();
            delta_str += item->output;
            plain_spans.emplace_back(s, delta_str.size());
            delta_str += "\n</tool_response>";
        }
        delta_str += "<|im_end|>\n";
    }

    // Handle user/assistant messages in input
    for (const auto &item : req.input_items) {
        if (item.kind == ResponsesInputItem::Kind::Message) {
            std::string role = item.role;
            if (role == "developer") role = "system";
            delta_str += "<|im_start|>" + role + "\n";
            const size_t s = delta_str.size();
            delta_str += item.content;
            plain_spans.emplace_back(s, delta_str.size());
            delta_str += "<|im_end|>\n";
        }
    }

    // Assistant generation prompt with thinking flag
    delta_str += "<|im_start|>assistant\n";
    const bool thinking = req.template_options.enable_thinking.value_or(true);
    if (thinking) delta_str += "<think>\n";
    else delta_str += "<think>\n\n</think>\n\n";

    std::vector<int32_t> delta_tokens = tok.encode(delta_str, plain_spans);

    std::vector<int32_t> prompt = prev->tokens;
    prefix_len = (int64_t)prompt.size();
    prompt.insert(prompt.end(), delta_tokens.begin(), delta_tokens.end());
    is_continuation = true;
    return prompt;
}

std::string response_event_frame(int64_t seq, const std::string &type, json::Value payload) {
    payload.set("type", json::Value::string(type));
    payload.set("sequence_number", json::Value::integer(seq));
    return payload.dump();
}

std::string response_created_event(int64_t seq, const std::string &resp_id, const std::string &model, int64_t created_at) {
    json::Value r = json::Value::object();
    r.set("id", json::Value::string(resp_id));
    r.set("object", json::Value::string("response"));
    r.set("status", json::Value::string("in_progress"));
    r.set("model", json::Value::string(model));
    r.set("created_at", json::Value::integer(created_at));
    r.set("output", json::Value::array());
    r.set("error", json::Value::null());
    r.set("incomplete_details", json::Value::null());
    json::Value b = json::Value::object();
    b.set("response", std::move(r));
    return response_event_frame(seq, "response.created", std::move(b));
}

std::string response_in_progress_event(int64_t seq, const std::string &resp_id, const std::string &model, int64_t created_at) {
    json::Value r = json::Value::object();
    r.set("id", json::Value::string(resp_id));
    r.set("object", json::Value::string("response"));
    r.set("status", json::Value::string("in_progress"));
    r.set("model", json::Value::string(model));
    r.set("created_at", json::Value::integer(created_at));
    r.set("output", json::Value::array());
    r.set("error", json::Value::null());
    r.set("incomplete_details", json::Value::null());
    json::Value b = json::Value::object();
    b.set("response", std::move(r));
    return response_event_frame(seq, "response.in_progress", std::move(b));
}

std::string response_output_item_added_event(int64_t seq, const std::string &resp_id, int output_index, const json::Value &item) {
    json::Value b = json::Value::object();
    b.set("response_id", json::Value::string(resp_id));
    b.set("output_index", json::Value::integer(output_index));
    b.set("item", item);
    return response_event_frame(seq, "response.output_item.added", std::move(b));
}

std::string response_output_item_done_event(int64_t seq, const std::string &resp_id, int output_index, const json::Value &item) {
    json::Value b = json::Value::object();
    b.set("response_id", json::Value::string(resp_id));
    b.set("output_index", json::Value::integer(output_index));
    b.set("item", item);
    return response_event_frame(seq, "response.output_item.done", std::move(b));
}

std::string response_content_part_added_event(int64_t seq, const std::string &resp_id, const std::string &item_id, int output_index, int content_index) {
    json::Value p = json::Value::object();
    p.set("type", json::Value::string("output_text"));
    p.set("text", json::Value::string(""));
    json::Value b = json::Value::object();
    b.set("response_id", json::Value::string(resp_id));
    b.set("item_id", json::Value::string(item_id));
    b.set("output_index", json::Value::integer(output_index));
    b.set("content_index", json::Value::integer(content_index));
    b.set("part", std::move(p));
    return response_event_frame(seq, "response.content_part.added", std::move(b));
}

std::string response_output_text_delta_event(int64_t seq, const std::string &resp_id, const std::string &item_id, int output_index, int content_index, const std::string &delta) {
    json::Value b = json::Value::object();
    b.set("response_id", json::Value::string(resp_id));
    b.set("item_id", json::Value::string(item_id));
    b.set("output_index", json::Value::integer(output_index));
    b.set("content_index", json::Value::integer(content_index));
    b.set("delta", json::Value::string(delta));
    return response_event_frame(seq, "response.output_text.delta", std::move(b));
}

std::string response_output_text_done_event(int64_t seq, const std::string &resp_id, const std::string &item_id, int output_index, int content_index, const std::string &text) {
    json::Value b = json::Value::object();
    b.set("response_id", json::Value::string(resp_id));
    b.set("item_id", json::Value::string(item_id));
    b.set("output_index", json::Value::integer(output_index));
    b.set("content_index", json::Value::integer(content_index));
    b.set("text", json::Value::string(text));
    return response_event_frame(seq, "response.output_text.done", std::move(b));
}

std::string response_content_part_done_event(int64_t seq, const std::string &resp_id, const std::string &item_id, int output_index, int content_index, const std::string &text) {
    json::Value p = json::Value::object();
    p.set("type", json::Value::string("output_text"));
    p.set("text", json::Value::string(text));
    json::Value b = json::Value::object();
    b.set("response_id", json::Value::string(resp_id));
    b.set("item_id", json::Value::string(item_id));
    b.set("output_index", json::Value::integer(output_index));
    b.set("content_index", json::Value::integer(content_index));
    b.set("part", std::move(p));
    return response_event_frame(seq, "response.content_part.done", std::move(b));
}

std::string response_reasoning_text_delta_event(int64_t seq, const std::string &resp_id, const std::string &item_id, int output_index, const std::string &delta) {
    json::Value b = json::Value::object();
    b.set("response_id", json::Value::string(resp_id));
    b.set("item_id", json::Value::string(item_id));
    b.set("output_index", json::Value::integer(output_index));
    b.set("delta", json::Value::string(delta));
    return response_event_frame(seq, "response.reasoning_text.delta", std::move(b));
}

std::string response_reasoning_text_done_event(int64_t seq, const std::string &resp_id, const std::string &item_id, int output_index, const std::string &text) {
    json::Value b = json::Value::object();
    b.set("response_id", json::Value::string(resp_id));
    b.set("item_id", json::Value::string(item_id));
    b.set("output_index", json::Value::integer(output_index));
    b.set("text", json::Value::string(text));
    return response_event_frame(seq, "response.reasoning_text.done", std::move(b));
}

std::string response_function_call_args_delta_event(int64_t seq, const std::string &resp_id, const std::string &item_id, int output_index, const std::string &call_id, const std::string &delta) {
    json::Value b = json::Value::object();
    b.set("response_id", json::Value::string(resp_id));
    b.set("item_id", json::Value::string(item_id));
    b.set("output_index", json::Value::integer(output_index));
    b.set("call_id", json::Value::string(call_id));
    b.set("delta", json::Value::string(delta));
    return response_event_frame(seq, "response.function_call_arguments.delta", std::move(b));
}

std::string response_function_call_args_done_event(int64_t seq, const std::string &resp_id, const std::string &item_id, int output_index, const std::string &call_id, const std::string &arguments) {
    json::Value b = json::Value::object();
    b.set("response_id", json::Value::string(resp_id));
    b.set("item_id", json::Value::string(item_id));
    b.set("output_index", json::Value::integer(output_index));
    b.set("call_id", json::Value::string(call_id));
    b.set("arguments", json::Value::string(arguments));
    return response_event_frame(seq, "response.function_call_arguments.done", std::move(b));
}

std::string response_completed_event(
    int64_t seq, const std::string &resp_id, const std::string &model, int64_t created_at,
    const json::Value &output_items, const GenerationResult &res,
    bool incomplete
) {
    json::Value u = json::Value::object();
    u.set("input_tokens", json::Value::integer(res.prompt_tokens));
    u.set("output_tokens", json::Value::integer(res.completion_tokens));
    u.set("total_tokens", json::Value::integer(res.prompt_tokens + res.completion_tokens));
    json::Value itd = json::Value::object();
    itd.set("cached_tokens", json::Value::integer(res.cached_tokens));
    u.set("input_tokens_details", std::move(itd));

    json::Value r = json::Value::object();
    r.set("id", json::Value::string(resp_id));
    r.set("object", json::Value::string("response"));
    r.set("status", json::Value::string(incomplete ? "incomplete" : "completed"));
    r.set("model", json::Value::string(model));
    r.set("created_at", json::Value::integer(created_at));
    r.set("output", output_items);
    r.set("usage", std::move(u));
    r.set("error", json::Value::null());
    if (incomplete) {
        json::Value inc = json::Value::object();
        std::string reason = (res.finish_reason == "length") ? "max_output_tokens" : "cancelled";
        inc.set("reason", json::Value::string(reason));
        r.set("incomplete_details", std::move(inc));
    } else {
        r.set("incomplete_details", json::Value::null());
    }

    json::Value b = json::Value::object();
    b.set("response", std::move(r));
    return response_event_frame(seq, incomplete ? "response.incomplete" : "response.completed", std::move(b));
}

std::string response_non_stream_body(
    const std::string &resp_id, const std::string &model, int64_t created_at,
    const json::Value &output_items, const GenerationResult &res,
    bool incomplete
) {
    json::Value u = json::Value::object();
    u.set("input_tokens", json::Value::integer(res.prompt_tokens));
    u.set("output_tokens", json::Value::integer(res.completion_tokens));
    u.set("total_tokens", json::Value::integer(res.prompt_tokens + res.completion_tokens));
    json::Value itd = json::Value::object();
    itd.set("cached_tokens", json::Value::integer(res.cached_tokens));
    u.set("input_tokens_details", std::move(itd));

    json::Value r = json::Value::object();
    r.set("id", json::Value::string(resp_id));
    r.set("object", json::Value::string("response"));
    r.set("status", json::Value::string(incomplete ? "incomplete" : "completed"));
    r.set("model", json::Value::string(model));
    r.set("created_at", json::Value::integer(created_at));
    r.set("output", output_items);
    r.set("usage", std::move(u));
    r.set("error", json::Value::null());
    if (incomplete) {
        json::Value inc = json::Value::object();
        std::string reason = (res.finish_reason == "length") ? "max_output_tokens" : "cancelled";
        inc.set("reason", json::Value::string(reason));
        r.set("incomplete_details", std::move(inc));
    } else {
        r.set("incomplete_details", json::Value::null());
    }
    return r.dump();
}

}  // namespace strix
