#pragma once

#include "serve/chat_template.hpp"
#include "serve/engine.hpp"
#include "serve/json.hpp"
#include "serve/response_store.hpp"
#include "serve/tokenizer.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace strix {

struct ResponsesInputItem {
    enum class Kind { Message, FunctionCall, FunctionCallOutput };
    Kind kind = Kind::Message;
    std::string role = "user";  // "user", "assistant", "system", "developer"
    std::string content;
    std::string call_id;
    std::string name;
    std::string arguments;
    std::string output;
};

struct ResponsesRequest {
    std::string model;
    std::optional<std::string> instructions;
    std::vector<ResponsesInputItem> input_items;
    std::string previous_response_id;
    json::Value tools;
    std::string tool_choice = "auto";
    bool stream = false;
    bool store = true;
    std::optional<int64_t> max_output_tokens;
    SamplingParams sampling;
    std::vector<std::string> stop;
    ChatTemplateOptions template_options;
    std::optional<int64_t> thinking_budget;
};

// Compute hash of instructions and tools
uint64_t compute_responses_config_hash(const std::optional<std::string> &instructions, const json::Value &tools);

// Parses a POST /v1/responses request body
ResponsesRequest parse_responses_request(const json::Value &body, const SamplingParams &defaults);

// Build tokens: either whole prompt (first turn), or spliced delta on top of previous_response_id.
// Inherits instructions and tools from previous response if omitted in the new request.
std::vector<int32_t> build_responses_prompt(
    ResponsesRequest &req,
    ResponseStore &store,
    const Tokenizer &tok,
    bool &is_continuation,
    int64_t &prefix_len
);

// Event formatters
std::string response_event_frame(int64_t seq, const std::string &type, json::Value payload);

std::string response_created_event(int64_t seq, const std::string &resp_id, const std::string &model, int64_t created_at);
std::string response_in_progress_event(int64_t seq, const std::string &resp_id, const std::string &model, int64_t created_at);

std::string response_output_item_added_event(int64_t seq, const std::string &resp_id, int output_index, const json::Value &item);
std::string response_output_item_done_event(int64_t seq, const std::string &resp_id, int output_index, const json::Value &item);

std::string response_content_part_added_event(int64_t seq, const std::string &resp_id, const std::string &item_id, int output_index, int content_index);
std::string response_output_text_delta_event(int64_t seq, const std::string &resp_id, const std::string &item_id, int output_index, int content_index, const std::string &delta);
std::string response_output_text_done_event(int64_t seq, const std::string &resp_id, const std::string &item_id, int output_index, int content_index, const std::string &text);
std::string response_content_part_done_event(int64_t seq, const std::string &resp_id, const std::string &item_id, int output_index, int content_index, const std::string &text);

std::string response_reasoning_text_delta_event(int64_t seq, const std::string &resp_id, const std::string &item_id, int output_index, const std::string &delta);
std::string response_reasoning_text_done_event(int64_t seq, const std::string &resp_id, const std::string &item_id, int output_index, const std::string &text);

std::string response_function_call_args_delta_event(int64_t seq, const std::string &resp_id, const std::string &item_id, int output_index, const std::string &call_id, const std::string &delta);
std::string response_function_call_args_done_event(int64_t seq, const std::string &resp_id, const std::string &item_id, int output_index, const std::string &call_id, const std::string &arguments);

std::string response_completed_event(
    int64_t seq, const std::string &resp_id, const std::string &model, int64_t created_at,
    const json::Value &output_items, const GenerationResult &res,
    bool incomplete = false
);

std::string response_non_stream_body(
    const std::string &resp_id, const std::string &model, int64_t created_at,
    const json::Value &output_items, const GenerationResult &res,
    bool incomplete = false
);

}  // namespace strix
