// serve.flerp.http: the request side of the OpenAI surface. What a request may ask is decided here from what the
// binary's executor says it offers (flerp_caps), never from which engine it is.
#pragma once

#include "flerp/executor.h"
#include "serve/json.hpp"

#include <optional>
#include <string>
#include <vector>

namespace flerp_serve {

// An error the client caused (or a feature this front end does not serve): HTTP status and an OpenAI error body.
struct ApiError {
    int status;
    std::string message, param, code;
};
std::string error_body(const ApiError &e);

// What this binary offers, fixed at startup.
struct Facts {
    flerp_caps caps{};
    bool caps_v2 = false;        // the executor filled flerp_caps.sampling and .top_k_max
    bool host_sampling = false;  // tokens are sampled on the host from rows the executor hands back
    int64_t vocab = 0;           // the tokenizer's size
    bool reply = false;          // a reply module splits chat replies (include/flerp/reply.h)
    flerp_sampling defaults{};   // for the fields a request leaves out
};

struct Request {
    bool chat = false;
    std::string body;    // chat: the request's JSON text, for the model's chat template
    std::string prompt;  // completions: the raw prompt
    std::string model;
    bool stream = false, include_usage = false, trace = false;
    std::optional<int64_t> max_tokens;
    flerp_sampling sampling{};
    bool seeded = false;
    std::vector<std::string> stop;
    std::vector<flerp_token> bias_id;  // ascending
    std::vector<float> bias;
    bool logprobs = false;
    int top_logprobs = 0;
    // chat: the reply (include/flerp/reply.h) and its constraint, as strixite's request parser reads them
    std::string tools;                     // the tools the template renders and the reply parser types (JSON), or empty
    bool tools_dropped = false;            // 'tools' sent and not rendered (tool_choice "none", or an empty array)
    std::optional<int64_t> thinking_budget;
    std::optional<bool> thinking_on;       // thinking switched on or off by the request (any spelling)
    bool effort_asked = false;             // a reasoning_effort level
    enum class Format { Text, JsonObject, JsonSchema } format = Format::Text;
    std::string schema, schema_name;       // json_schema: the schema (JSON) and its name
    bool thinking_requested() const { return (thinking_on && *thinking_on) || effort_asked || thinking_budget.has_value(); }
};

// Validates a parsed body. Throws ApiError: 400 for a malformed field, a feature this front end does not serve
// (code "not_served") or one this recipe's executor cannot honour (code "executor_cannot").
Request parse_request(const strix::json::Value &body, bool chat, const Facts &facts);

}  // namespace flerp_serve
