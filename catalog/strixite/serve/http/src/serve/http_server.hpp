#pragma once

// The HTTP surface on cpp-httplib: POST /v1/chat/completions (JSON or SSE),
// GET /v1/models, /health, /metrics (Prometheus text), /cache (prefix-reuse counters). Requests are
// rendered and tokenized on the HTTP thread, then queued on the Engine; a streaming response relays the engine's
// events as they come, with SSE comment keep-alives while the request waits or prefills; a client that goes away
// cancels its request.

#include "serve/engine.hpp"
#include "serve/json.hpp"
#include "serve/grammar_cache.hpp"
#include "serve/tokenizer.hpp"
#include "serve/vision_request.hpp"

#include <atomic>
#include <memory>
#include <string>

namespace httplib {
class Server;
}

namespace strix {

// What a request's "steer" field is resolved with (serve/steer_library: the vectors live with the model).
class SteerResolver {
public:
    virtual ~SteerResolver() = default;
    // [{"vector": name, "scale": s, "layer": L (default: the vector's own)}, ...] -> a set; null for an empty list.
    // Throws (strix::Error) naming what is wrong.
    virtual std::shared_ptr<const SteerSet> resolve(const json::Value &steer) const = 0;
    // What GET /steer answers: {"vectors": [{"name", "layer", "values", "hash"}, ...]}.
    virtual json::Value list() const = 0;
    // Read the directory again (POST /steer/reload): new and changed vectors serve from the next request. Requests
    // already resolved keep the vectors they hold. Returns list().
    virtual json::Value reload() const = 0;
};

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 5300;
    std::string model_id = "qwen3.8-flash-next";
    SamplingParams defaults;  // generation_config.json's
    // Thinking for requests that don't say (no enable_thinking in any spelling, no reasoning_effort, no budget): false
    // renders them without the think block (config `thinking`).
    bool thinking_default = true;
    // tool-grammar: every request with tools runs under the tool-call grammar (serve/tool_grammar): once a call
    // opens, only a declared tool and its parameter names can follow. Off: only tool_choice required / a name do.
    bool tool_grammar = true;
    // thinking-cap: the cap on thinking tokens for requests that name no budget of their own (0 = off). A request's own
    // budget always wins. thinking-cap-grace: past the cap, at most this many more thinking tokens while the engine
    // waits for a sentence end (GenerationRequest::thinking_budget_grace).
    int64_t thinking_cap = 0, thinking_cap_grace = 0;
    double keepalive_s = 10;  // SSE comment interval while nothing else is sent
    size_t max_body_bytes = 64u << 20;
    // Seconds a connection may go silent while its request body is still arriving before the read is abandoned and
    // the (retryable, x-should-retry) 408 goes out; inactivity per socket read, not total upload time (strixite #15:
    // httplib's 5 s default killed whole-history agent uploads over weak Wi-Fi with an empty, unlogged 400).
    int request_read_timeout_s = 30;
    // Vision (Phase D): non-null when the server loaded with vision on - the image parts parse, preprocess
    // (on the HTTP thread) and reach the engine; /health reports engine_.backend().has_vision().
    // vision_preprocess_limit (config vision-preprocess-limit): how many image preprocesses may run at once -
    // see vision::PreprocessGate (serve/vision_request.hpp).
    const vision::ServedVision *vision = nullptr;
    // Activation steering (config steer-dir): non-null when the server loaded a vector library; requests may then
    // carry "steer" (runtime/steer.hpp). Null: a request with "steer" is a 400.
    const SteerResolver *steer = nullptr;
    int64_t vision_preprocess_limit = 2;
    // --logprobs (card 31cf21e3d1): /v1/chat/completions takes "logprobs": true with "top_logprobs" up to 20 and
    // answers with per-token logprobs. Off (the default): such requests stay the same 400 they were before.
    bool logprobs = false;
    // --cache-xfer (card a4a320429d): POST /cache/export and /cache/import move the best cached entry of a
    // conversation between servers as disk file bytes (CTXMOVE-XFER). Off (the default): the routes don't exist -
    // they move conversation contents, and the engine port has no auth; the network is the trust boundary.
    bool cache_xfer = false;
};

class HttpServer {
public:
    HttpServer(Engine &engine, const Tokenizer &tok, ServerConfig cfg);
    ~HttpServer();
    // Binds (port 0 = any free port; returns the port) - then serve() blocks until stop().
    int bind();
    void serve();
    void stop();

private:
    Engine &engine_;
    const Tokenizer &tok_;
    ServerConfig cfg_;
    int64_t started_;
    std::atomic<int64_t> next_id_{0};  // request ids for the log ("req <id> ...")
    GrammarCache grammars_;            // compiled response formats (structured output), shared across requests
    vision::PreprocessGate vision_gate_;  // image preprocesses in flight (cfg_.vision_preprocess_limit)
    vision::DecodeMemo vision_memo_;      // grids of the images already decoded (their repeats skip the decode)
    std::unique_ptr<httplib::Server> svr_;
    void routes();
    // The prompt token ids a chat-completions body renders to: the request route's own parse, chat template and
    // tokenizer (so the tokens are the ones a served request of this body would have cached). For the transfer
    // routes; throws ApiError(400) the way the request route does for a body that doesn't parse or render.
    std::vector<int32_t> chat_tokens(const std::string &body);  // not const: an image's grid goes through vision_memo_
};

}  // namespace strix
