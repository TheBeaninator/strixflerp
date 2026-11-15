// serve.flerp.http: an OpenAI-compatible HTTP server in front of include/flerp/executor.h. One source tree for
// every executor: the binary is whatever executor adapter (executor.abi/flerp-v1/-), model text module
// (text.model/flerp-v1/-) and host sampler (sample.host/flerp-v1/-) its recipe binds. Nothing here names an engine;
// what a request may ask follows from flerp_caps.
//
//   strixflerp-<recipe> [--host H] [--port N] [--model-id ID] [--drafts N] [--draft-margin M] [--prompt-cut TOKEN]
//                       [--delivery auto|tokens] [--generation-config FILE] [--think-nudge on|off] [--think-end-guard on|off]
//                       [--literal-tags on|off] [--think-close-on-call on|off] [--thinking on|off]
//                       executor.KEY=VALUE ... text.KEY=VALUE ...
//
//   executor.K=V, text.K=V  the keys of flerp_open and flerp_text_open, as the bound modules name them
//   --drafts N              most drafts a verify (default: 7 for a draft head that steps, caps.max_draft for one that
//                           holds its drafts; 0: one token a forward). --draft-margin M (default 1.5): the top-2 logit
//                           margin a stepped draft needs. Both defaults are strixite's served ones (mtp-draft, mtp-margin)
//   --prompt-cut TOKEN      a prompt's forward ends at the last TOKEN of the prompt (strixite's engine cuts at the
//                           last <|im_start|>; its logits depend on the cut: docs/EXECUTOR-ABI.md, "Forward shape")
//   --delivery tokens       take the executor's own draws even where it offers logits (default auto: sample on the
//                           host where rows are offered, let the executor draw where only tokens are)
//   --generation-config F   a Hugging Face generation_config.json: temperature, top_k, top_p for requests that
//                           leave them out (otherwise 1, off, 1)
//
//   --think-nudge .. --thinking   strixite's reply policy (its server_config.cpp switches of the same names), all on
//                           as served. A chat reply is split by the model's reply module (include/flerp/reply.h,
//                           opened with the text module's keys) into reasoning_content, content and tool_calls.
//
// Routes: POST /v1/chat/completions, POST /v1/completions (JSON or SSE), GET /v1/models, GET /health, POST /tokenize.
// One conversation at a time: requests queue and run one after another on the thread that opened the executor
// (the ABI is driven by one thread); HTTP threads only parse, wait and write.
#include "flerp/decode_loop.h"
#include "flerp/executor.h"
#include "flerp/sampler.h"
#include "flerp/reply.h"
#include "flerp/serve/request.hpp"
#include "flerp/text.h"

#include "httplib.h"
#include "serve/json.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <limits>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace flerp_serve;
using strix::json::Value;

namespace {

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// --- what crosses between an HTTP thread and the executor's thread -------------------------------------------------
struct Event {
    enum class Kind { Start, Data, Done, Error } kind = Kind::Data;
    std::string body;  // Data: one SSE payload; Done: the whole response of a request that does not stream; Error: its body
    int status = 200;
};

struct Job {
    enum class What { Generate, Tokenize } what = What::Generate;
    Request req;
    std::string text;  // Tokenize
    std::string id;
    int64_t created = 0;
    std::atomic<bool> cancelled{false};
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Event> events;
    void push(Event e) {
        {
            std::lock_guard<std::mutex> lock(mu);
            events.push_back(std::move(e));
        }
        cv.notify_all();
    }
    bool pop(Event &e, double seconds) {
        std::unique_lock<std::mutex> lock(mu);
        if (!cv.wait_for(lock, std::chrono::duration<double>(seconds), [&] { return !events.empty(); })) return false;
        e = std::move(events.front());
        events.pop_front();
        return true;
    }
};

struct Config {
    std::string host = "127.0.0.1", model_id = "strixflerp", prompt_cut, delivery = "auto", generation_config;
    int port = 5400;
    int drafts = -1;
    double margin = 1.5;
    std::vector<std::pair<std::string, std::string>> executor, text;
    // the reply policy of strixite's engine (server_config.cpp: think-nudge, think-end-guard, literal-tags,
    // think-close-on-call, thinking), all on as served; off for like-for-like comparisons
    bool think_nudge = true, think_end_guard = true, literal_tags = true, think_close_on_call = true, thinking = true;
};

// --- the server's state: everything below the queue belongs to the executor's thread ---------------------------------
struct State {
    Config cfg;
    Facts facts;
    flerp_executor *ex = nullptr;
    flerp_text *text = nullptr;
    flerp_reply_model *reply = nullptr;  // the model's reply forms (include/flerp/reply.h)
    std::string executor_describe, text_describe, reply_describe;
    // the ids the reply policy feeds, tokenized with the model's text: the thinking budget's stop, the think close
    // ahead of a tool call, the two nudges; and the token that opens a call
    std::vector<flerp_token> thinking_stop, think_close, nudge[2];
    flerp_token call_begin = -1;
    std::vector<flerp_token> end_ids;
    flerp_token cut_id = -1;
    int drafts = 0;
    // result buffers for caps.max_verify_rows rows
    std::vector<flerp_token> tokens, cand_id;
    std::vector<float> cand_logit, full;
    std::vector<uint8_t> nan;
    // the queue
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::shared_ptr<Job>> queue;
    std::atomic<bool> stopping{false}, busy{false}, failed{false};
    std::atomic<int64_t> next_id{0}, served{0};
};

State *g_state = nullptr;
httplib::Server *g_server = nullptr;

void on_signal(int) {
    if (g_state) g_state->stopping = true;
    if (g_server) g_server->stop();
    if (g_state) g_state->cv.notify_all();
}

// --- text helpers -------------------------------------------------------------------------------------------------
// The length of the longest prefix of s[from, n) that does not end inside a UTF-8 sequence.
size_t utf8_whole(const std::string &s, size_t n) {
    size_t i = n;
    int back = 0;
    while (i > 0 && back < 4 && ((unsigned char)s[i - 1] & 0xC0) == 0x80) --i, ++back;
    if (i == 0) return n;
    const unsigned char lead = (unsigned char)s[i - 1];
    const int need = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
    return need > back + 1 ? i - 1 : n;
}

// Bytes as valid UTF-8 for a JSON string: each byte of an invalid sequence becomes U+FFFD.
std::string utf8_clean(const std::string &s) {
    if (strix::json::valid_utf8(s)) return s;
    std::string out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        const size_t len = c < 0x80 ? 1 : c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 0;
        if (len > 0 && i + len <= s.size() && strix::json::valid_utf8(std::string_view(s).substr(i, len))) {
            out.append(s, i, len), i += len;
        } else {
            out += "\xEF\xBF\xBD", ++i;
        }
    }
    return out;
}

std::string token_bytes(State &st, flerp_token id) {
    char buf[512];
    const int64_t n = flerp_text_token_bytes(st.text, id, buf, sizeof(buf));
    if (n < 0) return "";
    if ((size_t)n <= sizeof(buf)) return std::string(buf, (size_t)n);
    std::string big((size_t)n, 0);
    flerp_text_token_bytes(st.text, id, big.data(), big.size());
    return big;
}

std::vector<flerp_token> encode(State &st, const char *text, size_t n, const size_t *plain, size_t n_plain) {
    std::vector<flerp_token> ids(n + 16);
    int64_t got = flerp_text_encode(st.text, text, n, plain, n_plain, ids.data(), ids.size());
    if (got > (int64_t)ids.size()) {
        ids.resize((size_t)got);
        got = flerp_text_encode(st.text, text, n, plain, n_plain, ids.data(), ids.size());
    }
    if (got < 0) throw ApiError{400, std::string("the text does not tokenize: ") + flerp_text_last_error(st.text), "", ""};
    ids.resize((size_t)got);
    return ids;
}

Value ids_json(const std::vector<flerp_token> &ids) {
    Value a = Value::array();
    for (flerp_token t : ids) a.push(Value::integer(t));
    return a;
}

// --- one generation -----------------------------------------------------------------------------------------------
struct Logprob {
    flerp_token token;
    double logprob;
    std::vector<std::pair<flerp_token, double>> top;
};

struct Gen {
    State &st;
    Job &job;
    const Request &rq;
    flerp_rows rows{};
    uint32_t delivery = FLERP_OUT_TOKENS;
    flerp_sampler *sampler = nullptr;
    std::vector<flerp_token> prompt, emitted, sequence, unsent;
    std::vector<std::array<int, 3>> steps;
    std::string text;  // the reply's bytes
    size_t sent = 0, max_stop = 0;
    bool stop_hit = false;
    std::string pick_error;
    Logprob picked{};  // of the last picked row (logprobs)
    std::vector<Logprob> logprobs;
    // chat: the reply split (include/flerp/reply.h) and the engine's reply policy over it
    flerp_reply *reply = nullptr;
    flerp_reply_constraint *constraint = nullptr;  // response_format: masks on every row (FLERP_CAP_MASKS)
    std::vector<uint32_t> mask_bits;
    std::vector<flerp_token> chain;                 // the draft chain being grown (draft_ok)
    flerp_masks row_masks{};
    std::string reasoning, content;
    std::vector<Value> calls;  // whole calls, for a response that does not stream
    int tool_index = 0;
    int64_t count = 0, max_tokens = 0;  // completion tokens so far (picked and fed), and the most
    int64_t budget = -1;                // thinking tokens before the budget's stop is fed (-1: none)
    bool budget_hit = false;
    enum class Feeding { None, Budget, Nudge, CallClose } feeding = Feeding::None;
    int64_t nudges = 0, end_blocked = 0, closed_for_call = 0;
    std::string reply_error;
    Gen(State &s, Job &j) : st(s), job(j), rq(j.req) {}
    ~Gen() {
        if (sampler) flerp_sampler_free(sampler);
        if (reply) flerp_reply_end(reply);
        if (constraint) flerp_reply_constraint_free(constraint);
    }
    flerp_reply_state reply_state() const {
        flerp_reply_state r{};
        r.size = sizeof(r);
        flerp_reply_get_state(reply, &r);
        return r;
    }

    Value base(const char *object) const {
        Value v = Value::object();
        v.set("id", Value::string(job.id));
        v.set("object", Value::string(object));
        v.set("created", Value::integer(job.created));
        v.set("model", Value::string(st.cfg.model_id));
        return v;
    }

    // One streamed chunk with a piece of text (and, traced, the ids emitted since the last chunk).
    void send_text(const std::string &piece, const char *finish) {
        Value v = base(rq.chat ? "chat.completion.chunk" : "text_completion");
        Value choice = Value::object();
        choice.set("index", Value::integer(0));
        if (rq.chat) {
            Value delta = Value::object();
            if (finish == nullptr || !piece.empty()) delta.set("content", Value::string(utf8_clean(piece)));
            choice.set("delta", std::move(delta));
        } else {
            choice.set("text", Value::string(utf8_clean(piece)));
        }
        choice.set("finish_reason", finish ? Value::string(finish) : Value::null());
        Value choices = Value::array();
        choices.push(std::move(choice));
        v.set("choices", std::move(choices));
        if (rq.trace) {
            Value t = Value::object();
            t.set("token_ids", ids_json(unsent));
            v.set("flerp", std::move(t));
        }
        unsent.clear();
        job.push({Event::Kind::Data, v.dump(), 200});
    }

    // Streams what is safe to show: not the tail a stop string could still complete in, not half a character.
    void flush(bool final) {
        if (!rq.stream) return;
        size_t limit = text.size();
        if (!final) {
            if (max_stop > 1) limit = limit > max_stop - 1 ? limit - (max_stop - 1) : 0;
            limit = utf8_whole(text, limit);
        }
        if (limit > sent) {
            send_text(text.substr(sent, limit - sent), nullptr);
            sent = limit;
        }
    }

    static flerp_token pick(void *ctx, const flerp_rows *rows, int64_t row) {
        Gen &g = *(Gen *)ctx;
        flerp_token t = -1;
        if (g.reply && g.st.cfg.think_end_guard && g.delivery != FLERP_OUT_TOKENS) {
            const flerp_reply_state rs = g.reply_state();
            if ((rs.in_reasoning || rs.in_call) && g.ban_end(rows, row)) ++g.end_blocked;
        }
        if (g.delivery == FLERP_OUT_TOKENS) {
            t = rows->tokens[row];
        } else if (flerp_sampler_pick(g.sampler, g.delivery, rows, row, g.st.facts.caps.candidates, g.st.facts.caps.logits_row, &t) != FLERP_OK) {
            if (g.pick_error.empty()) g.pick_error = flerp_sampler_last_error();
            return g.st.end_ids.empty() ? 0 : g.st.end_ids[0];  // ends the loop; the request fails with pick_error
        }
        if (g.rq.logprobs) g.note_logprob(rows->full + row * g.st.facts.caps.logits_row, t);
        return t;
    }

    // The engine's think-end guard (serve/sampler.cpp ban_tokens): the end ids out of a row; true when the row's top
    // was one of them.
    bool ban_end(const flerp_rows *rows, int64_t row) {
        const std::vector<flerp_token> &ids = st.end_ids;
        const auto banned = [&](flerp_token id) { return std::find(ids.begin(), ids.end(), id) != ids.end(); };
        const float ninf = -std::numeric_limits<float>::infinity();
        if (delivery == FLERP_OUT_FULL) {
            const int64_t n = st.facts.caps.logits_row;
            float *x = rows->full + row * n;
            const flerp_token top = (flerp_token)(std::max_element(x, x + n) - x);
            for (flerp_token id : ids)
                if (id >= 0 && id < n) x[id] = ninf;
            return banned(top);
        }
        const int64_t c = st.facts.caps.candidates;
        float *v = rows->cand_logit + row * c;
        flerp_token *id = rows->cand_id + row * c;
        const bool top = banned(id[0]);
        int64_t w = 0;
        for (int64_t i = 0; i < c; ++i)
            if (!banned(id[i])) v[w] = v[i], id[w] = id[i], ++w;
        for (; w < c; ++w) v[w] = ninf, id[w] = std::numeric_limits<flerp_token>::max();
        return top;
    }

    // One chunk of a streamed chat reply with this delta (and, traced, the ids emitted since the last chunk).
    void send_delta(Value delta) {
        Value v = base("chat.completion.chunk");
        Value choice = Value::object();
        choice.set("index", Value::integer(0));
        choice.set("delta", std::move(delta));
        choice.set("finish_reason", Value::null());
        Value choices = Value::array();
        choices.push(std::move(choice));
        v.set("choices", std::move(choices));
        if (rq.trace) {
            Value t = Value::object();
            t.set("token_ids", ids_json(unsent));
            v.set("flerp", std::move(t));
        }
        unsent.clear();
        job.push({Event::Kind::Data, v.dump(), 200});
    }

    // strixite's tool call shapes (serve/openai.cpp tool_call_delta / _start_delta / _args_delta)
    static Value call_json(const flerp_reply_event &e, int index, bool whole) {
        Value fn = Value::object();
        fn.set("name", Value::string(e.name ? e.name : ""));
        fn.set("arguments", Value::string(whole ? std::string(e.text, e.n_text) : std::string()));
        Value tc = Value::object();
        tc.set("index", Value::integer(index));
        tc.set("id", Value::string(e.id ? e.id : ""));
        tc.set("type", Value::string("function"));
        tc.set("function", std::move(fn));
        return tc;
    }

    // The reply's events: streamed as strixite's server streams them, or kept for the response.
    void on_events(const flerp_reply_event *ev, size_t n) {
        for (size_t k = 0; k < n; ++k) {
            const flerp_reply_event &e = ev[k];
            const std::string text(e.text ? e.text : "", e.n_text);
            if (!rq.stream) {
                if (e.kind == FLERP_REPLY_REASONING) reasoning += text;
                else if (e.kind == FLERP_REPLY_CONTENT) content += text;
                else if (e.kind == FLERP_REPLY_CALL) calls.push_back(call_json(e, (int)calls.size(), true));
                continue;
            }
            Value d = Value::object();
            if (e.kind == FLERP_REPLY_REASONING) d.set("reasoning_content", Value::string(text));
            else if (e.kind == FLERP_REPLY_CONTENT) d.set("content", Value::string(text));
            else {
                Value tcs = Value::array();
                if (e.kind == FLERP_REPLY_CALL) tcs.push(call_json(e, tool_index++, true));
                else if (e.kind == FLERP_REPLY_CALL_START) tcs.push(call_json(e, tool_index++, false));
                else {  // consecutive argument pieces (one token can complete several) as one delta
                    std::string piece = text;
                    while (k + 1 < n && ev[k + 1].kind == FLERP_REPLY_CALL_ARGS) ++k, piece.append(ev[k].text, ev[k].n_text);
                    Value fn = Value::object();
                    fn.set("arguments", Value::string(piece));
                    Value tc = Value::object();
                    tc.set("index", Value::integer(tool_index - 1));
                    tc.set("function", std::move(fn));
                    tcs.push(std::move(tc));
                }
                d.set("tool_calls", std::move(tcs));
            }
            send_delta(std::move(d));
        }
    }

    // --- the engine's reply policy, as decode-loop hooks (strixite serve/engine.cpp Engine::run) ---
    // The thinking budget's stop is due (the engine's stop_due, grace 0).
    bool stop_due() const {
        if (budget < 0 || budget_hit) return false;
        const flerp_reply_state rs = reply_state();
        return rs.in_reasoning && rs.reasoning_tokens >= budget;
    }
    static int fed(void *ctx, int64_t, const flerp_token **ids) {
        Gen &g = *(Gen *)ctx;
        if (!g.reply) return 0;
        if (g.stop_due()) {
            g.feeding = Feeding::Budget;
            *ids = g.st.thinking_stop.data();
            return (int)g.st.thinking_stop.size();
        }
        if (g.st.cfg.think_nudge) {  // a nudge into a think block going in circles, when it fits
            const int d = g.reply_state().nudge_due;
            if (d > 0) {
                const std::vector<flerp_token> &nt = g.st.nudge[d - 1];
                if (g.count + (int64_t)nt.size() < g.max_tokens) {
                    g.feeding = Feeding::Nudge;
                    *ids = nt.data();
                    return (int)nt.size();
                }
            }
        }
        return 0;
    }
    static int fed_stops_drafts(void *ctx, int64_t) { return ((Gen *)ctx)->stop_due() ? 1 : 0; }
    // A tool call begun inside the think block, right after a newline: the think close fed ahead of it.
    bool misplaced_call(flerp_token t) const {
        if (!reply || constraint || !st.cfg.think_close_on_call || t != st.call_begin || st.think_close.empty()) return false;
        const flerp_reply_state rs = reply_state();
        return rs.in_reasoning && rs.after_newline;
    }
    static int before(void *ctx, flerp_token token, const flerp_token **ids) {
        Gen &g = *(Gen *)ctx;
        if (!g.misplaced_call(token)) return 0;
        g.feeding = Feeding::CallClose;
        *ids = g.st.think_close.data();
        return (int)g.st.think_close.size();
    }
    // No <tool_call> drafted inside the think block (it might need the close fed ahead of it), and no draft the
    // constraint refuses (it would be rejected anyway: the engine does not verify it).
    static int draft_ok(void *ctx, int k, flerp_token d) {
        Gen &g = *(Gen *)ctx;
        if (k == 0) g.chain.clear();
        if (g.reply && g.st.cfg.think_close_on_call && d == g.st.call_begin && g.reply_state().in_reasoning) return 0;
        if (g.constraint) {
            if (!flerp_reply_constraint_allows(g.constraint, g.chain.data(), g.chain.size(), d)) return 0;
            g.chain.push_back(d);
        }
        return 1;
    }
    // The constraint's masks for a forward (pending empty) or a verify's rows, with the request's logit bias.
    static const flerp_masks *masks(void *ctx, const flerp_token *pending, size_t n) {
        Gen &g = *(Gen *)ctx;
        if (!g.constraint) return nullptr;
        const int64_t words = flerp_reply_constraint_words(g.constraint);
        g.mask_bits.resize((n + 1) * (size_t)words);
        if (flerp_reply_constraint_masks(g.constraint, pending, n, g.mask_bits.data(), g.mask_bits.size()) != FLERP_OK) {
            if (g.reply_error.empty()) g.reply_error = flerp_reply_last_error(g.st.reply);
            return nullptr;
        }
        g.row_masks.size = sizeof(flerp_masks), g.row_masks.rows = (int64_t)n + 1, g.row_masks.words = words, g.row_masks.bits = g.mask_bits.data();
        g.row_masks.bias_id = g.rq.bias_id.data(), g.row_masks.bias = g.rq.bias.data(), g.row_masks.n_bias = g.rq.bias_id.size();
        return &g.row_masks;
    }

    // The picked token's log-probability and the row's top ones, from the full row over the tokenizer's ids.
    void note_logprob(const float *x, flerp_token t) {
        const int64_t n = st.facts.vocab;
        float mx = x[0];
        for (int64_t i = 1; i < n; ++i) mx = std::max(mx, x[i]);
        double sum = 0;
        for (int64_t i = 0; i < n; ++i) sum += std::exp((double)x[i] - mx);
        const double lse = mx + std::log(sum);
        picked = Logprob{t, (double)x[t] - lse, {}};
        std::vector<std::pair<float, flerp_token>> top;  // value descending, the lower id first on a tie
        const size_t k = (size_t)rq.top_logprobs;
        for (int64_t i = 0; k > 0 && i < n; ++i) {
            if (top.size() == k && !(x[i] > top.back().first)) continue;
            auto at = std::upper_bound(top.begin(), top.end(), std::pair<float, flerp_token>{x[i], (flerp_token)i},
                                       [](const auto &a, const auto &b) { return a.first > b.first; });
            top.insert(at, {x[i], (flerp_token)i});
            if (top.size() > k) top.pop_back();
        }
        for (const auto &[v, id] : top) picked.top.emplace_back(id, (double)v - lse);
    }

    static int emit(void *ctx, flerp_token token, int fed) {
        Gen &g = *(Gen *)ctx;
        const size_t before = g.text.size();
        g.text += token_bytes(g.st, token);
        g.emitted.push_back(token), g.unsent.push_back(token);
        ++g.count;
        if (g.rq.logprobs && !fed) g.logprobs.push_back(g.picked);
        if (g.constraint && flerp_reply_constraint_feed(g.constraint, token) != FLERP_OK) {
            if (g.reply_error.empty()) g.reply_error = flerp_reply_last_error(g.st.reply);
            return 1;
        }
        if (g.reply) {  // chat: the reply's own split; its stop strings apply to content
            if (fed && g.feeding != Feeding::None) {  // the first id of a fed run
                if (g.feeding == Feeding::Budget) g.budget_hit = true;
                else if (g.feeding == Feeding::Nudge) flerp_reply_nudge_fired(g.reply), ++g.nudges;
                else ++g.closed_for_call;
                g.feeding = Feeding::None;
            }
            const flerp_reply_event *ev = nullptr;
            size_t n = 0;
            const int stopped = flerp_reply_feed(g.reply, token, fed, &ev, &n);
            if (stopped < 0) {
                if (g.reply_error.empty()) g.reply_error = flerp_reply_last_error(g.st.reply);
                return 1;
            }
            g.on_events(ev, n);
            return fed ? 0 : stopped;
        }
        if (!fed && g.max_stop > 0) {  // a stop string completed by this token ends the reply before the string
            const size_t from = before > g.max_stop - 1 ? before - (g.max_stop - 1) : 0;
            size_t hit = std::string::npos;
            for (const std::string &s : g.rq.stop) hit = std::min(hit, g.text.find(s, from));
            if (hit != std::string::npos) {
                g.text.resize(hit);
                g.stop_hit = true;
                return 1;
            }
        }
        g.flush(false);
        return 0;
    }
    static void step(void *ctx, int drafted, int accepted, int emitted) { ((Gen *)ctx)->steps.push_back({drafted, accepted, emitted}); }
    static void entered(void *ctx, const flerp_token *ids, size_t n) {
        Gen &g = *(Gen *)ctx;
        g.sequence.insert(g.sequence.end(), ids, ids + n);
    }
    static int cancelled(void *ctx) {
        const Gen &g = *(Gen *)ctx;
        return g.job.cancelled.load() || g.st.stopping.load() || !g.pick_error.empty() || !g.reply_error.empty();
    }
};

[[noreturn]] void executor_error(State &st, flerp_status status, const char *what) {
    const std::string why = std::string(what) + ": " + flerp_last_error(st.ex);
    if (flerp_failed(st.ex)) st.failed = true;
    // a refusal of the request by the executor is the client's; a failed backend is the server's
    if (status == FLERP_E_ARGUMENT || status == FLERP_E_UNSUPPORTED || status == FLERP_E_FULL)
        throw ApiError{400, why, "", "executor_cannot"};
    throw ApiError{500, why, "", "executor_failed"};
}

void generate(State &st, Job &job) {
    const Request &rq = job.req;
    const flerp_caps &caps = st.facts.caps;
    Gen g(st, job);
    // the prompt: the model's template and tokenizer. Chat follows strixite's request route (serve/http_server.cpp):
    // tools rendered only when served (tool_choice "auto"), the server's thinking default, and with no room to think
    // (a small max_tokens and no thinking asked for) a second render without the think block.
    std::optional<bool> thinking_on = rq.thinking_on;
    if (rq.chat && !st.cfg.thinking && !thinking_on && !rq.thinking_requested()) thinking_on = false;
    const auto render = [&](bool thinking_off) {
        std::string body = rq.body;
        if (rq.tools_dropped || thinking_off) {
            Value b = Value::parse(rq.body);
            if (rq.tools_dropped) b.set("tools", Value::array());
            if (thinking_off) b.set("enable_thinking", Value::boolean(false));
            body = b.dump();
        }
        flerp_text_chat chat{};
        chat.size = sizeof(chat);
        const flerp_status ts = flerp_text_render_chat(st.text, body.data(), body.size(), &chat);
        if (ts == FLERP_E_UNSUPPORTED) throw ApiError{400, std::string("this model has no chat template: ") + flerp_text_last_error(st.text), "messages", "not_served"};
        if (ts != FLERP_OK) throw ApiError{400, flerp_text_last_error(st.text), "messages", ""};
        g.prompt = encode(st, chat.text, chat.n_text, chat.plain, chat.n_plain);
    };
    int64_t P = 0, room = 0, max_tokens = 0;
    bool clamped = false;
    const auto measure = [&] {
        P = (int64_t)g.prompt.size();
        if (P < 1) throw ApiError{400, "the prompt is empty", rq.chat ? "messages" : "prompt", ""};
        room = caps.positions - P;
        if (room < 1)
            throw ApiError{400, "the prompt is " + std::to_string(P) + " tokens and the context holds " + std::to_string(caps.positions), rq.chat ? "messages" : "prompt",
                           "context_length_exceeded"};
        max_tokens = std::min<int64_t>(rq.max_tokens.value_or(room), room);
        clamped = !rq.max_tokens || *rq.max_tokens > room;
    };
    // strixite's thinking_budget_for (serve/openai.cpp): an explicit budget, or all but the answer's room
    const auto budget_for = [&](int64_t mt) -> int64_t {
        if (thinking_on && !*thinking_on) return -1;
        if (rq.thinking_budget) return *rq.thinking_budget;
        const int64_t floor = rq.thinking_requested() ? 64 : 1024;
        return std::max<int64_t>(0, mt - std::max<int64_t>(floor, mt * 15 / 100));
    };
    bool thinking_skipped = false;
    if (rq.chat) {
        render(thinking_on && !*thinking_on);
        measure();
        if (st.reply && !(thinking_on && !*thinking_on) && !rq.thinking_requested() && budget_for(max_tokens) == 0) {
            thinking_on = false, thinking_skipped = true;
            render(true);
            measure();
        }
    } else {
        g.prompt = encode(st, rq.prompt.data(), rq.prompt.size(), nullptr, 0);
        measure();
    }
    g.max_tokens = max_tokens;
    if (rq.chat && st.reply) {  // the reply's split
        std::vector<const char *> stops;
        for (const std::string &x : rq.stop) stops.push_back(x.c_str());
        flerp_reply_config rc{};
        rc.size = sizeof(rc);
        rc.thinking = !(thinking_on && !*thinking_on);
        rc.tools = rq.tools.empty() ? nullptr : rq.tools.data(), rc.n_tools = rq.tools.size();
        rc.stop = stops.data(), rc.n_stop = stops.size();
        rc.id_seed = std::random_device{}();
        // literal tags not under a response format's grammar, whose think-block end is the token (serve/engine.cpp)
        rc.trim_ws = 1, rc.literal_tags = st.cfg.literal_tags && rq.format == Request::Format::Text, rc.stream_calls = rq.stream;
        rc.nudge = st.cfg.think_nudge;
        if (flerp_reply_begin(st.reply, &rc, &g.reply) != FLERP_OK) throw ApiError{500, std::string("the reply parser: ") + flerp_reply_last_error(st.reply), "", "internal"};
        g.budget = rc.thinking ? budget_for(max_tokens) : -1;
        if (rq.format != Request::Format::Text) {  // the response format, compiled for this executor's logits row
            const bool object = rq.format == Request::Format::JsonObject;
            const flerp_status cs = flerp_reply_constrain(st.reply, object ? FLERP_REPLY_JSON_OBJECT : FLERP_REPLY_JSON_SCHEMA, rq.schema.data(),
                                                          rq.schema.size(), !rc.thinking, caps.logits_row > 0 ? caps.logits_row : st.facts.vocab, &g.constraint);
            if (cs == FLERP_E_ARGUMENT) throw ApiError{400, flerp_reply_last_error(st.reply), "response_format", ""};
            if (cs != FLERP_OK) throw ApiError{500, std::string("the response format: ") + flerp_reply_last_error(st.reply), "response_format", "internal"};
        }
    }
    if (!rq.chat)  // chat: the reply parser applies stop strings to its content
        for (const std::string &s : rq.stop) g.max_stop = std::max(g.max_stop, s.size());

    // who samples, and from what
    flerp_sampling sampling = rq.sampling;
    sampling.size = sizeof(sampling);
    if (!rq.seeded) sampling.seed = std::random_device{}();
    if (st.facts.host_sampling) {
        const flerp_status ss = flerp_sampler_new(&sampling, st.facts.vocab, &g.sampler);
        if (ss != FLERP_OK) throw ApiError{ss == FLERP_E_ARGUMENT ? 400 : 500, std::string("sampling: ") + flerp_sampler_last_error(), "", ""};
        const bool cands = (caps.deliveries & FLERP_OUT_CANDIDATES) && !rq.logprobs && flerp_sampler_takes_candidates(g.sampler, caps.candidates);
        if (!cands && !(caps.deliveries & FLERP_OUT_FULL))
            throw ApiError{400, "these sampling parameters need full logits rows and this recipe's executor hands back candidates only: send top_k from 1 to " +
                                    std::to_string(caps.candidates) + ", or temperature 0", "top_k", "executor_cannot"};
        g.delivery = cands ? FLERP_OUT_CANDIDATES : FLERP_OUT_FULL;
    }
    g.rows.size = sizeof(flerp_rows);
    const size_t R = (size_t)std::max<int64_t>(1, caps.max_verify_rows);
    g.rows.nan = st.nan.data(), g.rows.nan_cap = st.nan.size();
    if (g.delivery == FLERP_OUT_TOKENS) g.rows.tokens = st.tokens.data(), g.rows.tokens_cap = st.tokens.size();
    else if (g.delivery == FLERP_OUT_CANDIDATES)
        g.rows.cand_id = st.cand_id.data(), g.rows.cand_logit = st.cand_logit.data(), g.rows.cand_cap = st.cand_id.size();
    else {
        if (st.full.empty()) st.full.resize(R * (size_t)caps.logits_row);
        g.rows.full = st.full.data(), g.rows.full_cap = st.full.size();
    }
    flerp_masks bias{};
    bias.size = sizeof(bias), bias.bias_id = rq.bias_id.data(), bias.bias = rq.bias.data(), bias.n_bias = rq.bias_id.size();
    const flerp_masks *masks = rq.bias_id.empty() ? nullptr : &bias;

    flerp_seq_args sa{};
    sa.size = sizeof(sa), sa.max_new = max_tokens, sa.drafts = st.drafts;   // the most drafts a verify of this seq holds (was a flag: 1, card 3f6d20b703)
    sa.sampling = !st.facts.host_sampling && sampling.temperature > 0 ? &sampling : nullptr;
    int32_t seq = -1;
    if (const flerp_status s = flerp_seq_new(st.ex, &sa, &seq); s != FLERP_OK) executor_error(st, s, "flerp_seq_new");
    struct Free {
        State &st;
        int32_t seq;
        ~Free() { flerp_seq_free(st.ex, seq); }
    } free_seq{st, seq};

    job.push({Event::Kind::Start, "", 200});
    if (rq.stream && rq.chat) {  // the role first, as OpenAI streams it
        Value v = g.base("chat.completion.chunk");
        Value delta = Value::object();
        delta.set("role", Value::string("assistant"));
        delta.set("content", Value::string(""));
        Value choice = Value::object();
        choice.set("index", Value::integer(0));
        choice.set("delta", std::move(delta));
        choice.set("finish_reason", Value::null());
        Value choices = Value::array();
        choices.push(std::move(choice));
        v.set("choices", std::move(choices));
        job.push({Event::Kind::Data, v.dump(), 200});
    }

    // the prompt's forwards, then the loop every caller of the ABI shares (include/flerp/decode_loop.h)
    const double t0 = now_s();
    flerp_feed feed{};
    feed.size = sizeof(feed), feed.ex = st.ex, feed.seq = seq, feed.max_forward = caps.max_forward, feed.lookahead = 1, feed.snapshot_at = -1;
    int64_t cut = -1;
    if (st.cut_id >= 0)
        for (int64_t k = P - 1; k > 0 && cut < 0; --k)
            if (g.prompt[(size_t)k] == st.cut_id) cut = k;
    if (cut > 0) feed.cuts = &cut, feed.n_cuts = 1;
    flerp_want want{};
    want.size = sizeof(want), want.delivery = g.delivery, want.rows = 1, want.n_valid = st.facts.vocab, want.masks = masks;
    if (g.constraint) want.masks = Gen::masks(&g, nullptr, 0);  // the prompt's last row is the answer's first
    if (const flerp_status s = flerp_feed_ids(&feed, g.prompt.data(), 0, (size_t)P, &want, &g.rows); s != FLERP_OK) executor_error(st, s, "the prompt's forward");
    const double t1 = now_s();

    flerp_loop L{};
    L.size = sizeof(L), L.ex = st.ex, L.seq = seq, L.caps = &caps, L.delivery = g.delivery, L.n_valid = st.facts.vocab, L.masks = masks;
    L.rows = &g.rows, L.max_tokens = max_tokens, L.drafts = st.drafts, L.margin = st.cfg.margin;
    L.end_ids = st.end_ids.data(), L.n_end = st.end_ids.size();
    flerp_loop_hooks H{};
    H.ctx = &g, H.pick = Gen::pick, H.emit = Gen::emit, H.step = Gen::step, H.entered = Gen::entered, H.cancelled = Gen::cancelled;
    if (g.reply) H.fed = Gen::fed, H.fed_stops_drafts = Gen::fed_stops_drafts, H.before = Gen::before, H.draft_ok = Gen::draft_ok, H.masks = Gen::masks;
    flerp_loop_result res{};
    res.size = sizeof(res);
    const flerp_status ls = flerp_decode_loop(&L, &H, &res);
    const double t2 = now_s();
    if (ls != FLERP_OK) {
        if (res.error) throw ApiError{500, std::string("the decode loop: ") + res.error, "", "executor_failed"};
        executor_error(st, ls, "the decode loop");
    }
    if (!g.pick_error.empty()) throw ApiError{500, "sampling: " + g.pick_error, "", "sampler_failed"};
    if (!g.reply_error.empty()) throw ApiError{500, "the reply parser: " + g.reply_error, "", "internal"};
    std::string finish = res.finish ? res.finish : "stop";
    flerp_reply_state rs{};
    rs.size = sizeof(rs);
    if (g.reply && finish != "cancelled") {
        const flerp_reply_event *ev = nullptr;
        size_t n = 0;
        if (flerp_reply_finish(g.reply, &ev, &n) != FLERP_OK) throw ApiError{500, std::string("the reply parser: ") + flerp_reply_last_error(st.reply), "", "internal"};
        g.on_events(ev, n);
        flerp_reply_get_state(g.reply, &rs);
        if (finish == "stop" && rs.tool_calls + rs.malformed_calls > 0) finish = "tool_calls";
    }

    // the response
    Value usage = Value::object();
    usage.set("prompt_tokens", Value::integer(P));
    usage.set("completion_tokens", Value::integer(res.completion));
    usage.set("total_tokens", Value::integer(P + res.completion));
    if (g.reply) {
        Value cd = Value::object();
        cd.set("reasoning_tokens", Value::integer(rs.reasoning_tokens));
        usage.set("completion_tokens_details", std::move(cd));
    }
    Value timings = Value::object();
    const double prompt_ms = (t1 - t0) * 1e3, decode_ms = (t2 - t1) * 1e3;
    timings.set("prompt_n", Value::integer(P));
    timings.set("prompt_ms", Value::number(prompt_ms));
    timings.set("prompt_per_second", Value::number(prompt_ms > 0 ? (double)P / (prompt_ms / 1e3) : 0.0));
    timings.set("predicted_n", Value::integer(res.completion));
    timings.set("predicted_ms", Value::number(decode_ms));
    // as strixite's "generate" line counts it: the first token comes with the prompt's forward
    timings.set("predicted_per_second", Value::number(decode_ms > 0 && res.completion > 1 ? (double)(res.completion - 1) / (decode_ms / 1e3) : 0.0));
    Value flerp = Value::object();
    if (rq.trace) {
        flerp.set("prompt_token_ids", ids_json(g.prompt));
        flerp.set("token_ids", ids_json(g.emitted));
        flerp.set("sequence_ids", ids_json(g.sequence));
        Value steps = Value::array();
        for (const auto &s : g.steps) {
            Value one = Value::array();
            one.push(Value::integer(s[0])), one.push(Value::integer(s[1])), one.push(Value::integer(s[2]));
            steps.push(std::move(one));
        }
        flerp.set("steps", std::move(steps));
        flerp.set("delivery", Value::string(g.delivery == FLERP_OUT_TOKENS ? "tokens" : g.delivery == FLERP_OUT_CANDIDATES ? "candidates" : "full"));
        flerp.set("sampled_by", Value::string(st.facts.host_sampling ? "host" : "executor"));
        flerp.set("prompt_cut", Value::integer(cut));
        flerp.set("max_tokens", Value::integer(max_tokens));
        flerp.set("max_tokens_clamped", Value::boolean(clamped));
        flerp.set("forwards", Value::integer(res.forwards));
        flerp.set("verifies", Value::integer(res.verifies));
        flerp.set("drafted", Value::integer(res.drafted));
        flerp.set("accepted", Value::integer(res.accepted));
        flerp.set("seed", Value::string(std::to_string(sampling.seed)));
    }
    if (g.reply) {  // what the reply policy did (strixite's "strix" extension rows of the same names)
        Value r = Value::object();
        r.set("thinking", Value::string(thinking_skipped ? "skipped" : (thinking_on && !*thinking_on) ? "off" : "on"));
        r.set("thinking_budget", Value::integer(g.budget));
        r.set("thinking_budget_hit", Value::boolean(g.budget_hit));
        if (g.nudges > 0) r.set("think_nudges", Value::integer(g.nudges));
        if (g.end_blocked > 0) r.set("think_end_blocked", Value::integer(g.end_blocked));
        if (g.closed_for_call > 0) r.set("think_closed_for_call", Value::integer(g.closed_for_call));
        if (rs.dropped_partial_call) r.set("dropped_partial_tool_call", Value::boolean(true));
        if (rs.malformed_calls > 0) r.set("malformed_tool_calls", Value::integer(rs.malformed_calls));
        flerp.set("reply", std::move(r));
    }
    if (finish == "cancelled") {
        job.push({Event::Kind::Error, error_body({499, "the request was cancelled", "", "cancelled"}), 499});
        return;
    }
    if (rq.stream) {
        if (!g.reply) g.flush(true);
        g.send_text("", finish.c_str());
        if (rq.include_usage || rq.trace) {
            Value v = g.base(rq.chat ? "chat.completion.chunk" : "text_completion");
            v.set("choices", Value::array());
            v.set("usage", std::move(usage));
            v.set("timings", std::move(timings));
            if (rq.trace) v.set("flerp", std::move(flerp));
            job.push({Event::Kind::Data, v.dump(), 200});
        }
        job.push({Event::Kind::Done, "", 200});
        return;
    }
    Value v = g.base(rq.chat ? "chat.completion" : "text_completion");
    Value choice = Value::object();
    choice.set("index", Value::integer(0));
    if (rq.chat) {
        Value message = Value::object();
        message.set("role", Value::string("assistant"));
        // strixite's message (serve/openai.cpp completion_body): content null when the reply is tool calls only
        if (!g.reply) message.set("content", Value::string(utf8_clean(g.text)));  // no reply module: the text whole
        else message.set("content", g.content.empty() && !g.calls.empty() ? Value::null() : Value::string(g.content));
        if (!g.reasoning.empty()) message.set("reasoning_content", Value::string(g.reasoning));
        if (!g.calls.empty()) {
            Value tcs = Value::array();
            for (Value &c : g.calls) tcs.push(std::move(c));
            message.set("tool_calls", std::move(tcs));
        }
        choice.set("message", std::move(message));
        if (rq.logprobs) {
            Value content = Value::array();
            const auto entry = [&](flerp_token id, double lp) {
                const std::string b = token_bytes(st, id);
                Value e = Value::object();
                e.set("token", Value::string(utf8_clean(b)));
                e.set("token_id", Value::integer(id));
                e.set("logprob", Value::number(lp));
                Value bytes = Value::array();
                for (unsigned char c : b) bytes.push(Value::integer(c));
                e.set("bytes", std::move(bytes));
                return e;
            };
            for (const Logprob &lp : g.logprobs) {
                Value e = entry(lp.token, lp.logprob);
                Value top = Value::array();
                for (const auto &[id, x] : lp.top) top.push(entry(id, x));
                e.set("top_logprobs", std::move(top));
                content.push(std::move(e));
            }
            Value lp = Value::object();
            lp.set("content", std::move(content));
            choice.set("logprobs", std::move(lp));
        } else {
            choice.set("logprobs", Value::null());
        }
    } else {
        choice.set("text", Value::string(utf8_clean(g.text)));
        choice.set("logprobs", Value::null());
    }
    choice.set("finish_reason", Value::string(finish));
    Value choices = Value::array();
    choices.push(std::move(choice));
    v.set("choices", std::move(choices));
    v.set("usage", std::move(usage));
    v.set("timings", std::move(timings));
    if (rq.trace) v.set("flerp", std::move(flerp));
    job.push({Event::Kind::Done, v.dump(), 200});
}

void run_job(State &st, Job &job) {
    try {
        if (st.failed) throw ApiError{503, "the executor failed and must be restarted", "", "executor_failed"};
        if (job.what == Job::What::Tokenize) {
            Value v = Value::object();
            v.set("tokens", ids_json(encode(st, job.text.data(), job.text.size(), nullptr, 0)));
            job.push({Event::Kind::Done, v.dump(), 200});
            return;
        }
        generate(st, job);
    } catch (const ApiError &e) {
        job.push({Event::Kind::Error, error_body(e), e.status});
    } catch (const std::exception &e) {
        job.push({Event::Kind::Error, error_body({500, e.what(), "", "internal"}), 500});
    }
}

// --- HTTP -----------------------------------------------------------------------------------------------------------
void reply_error(httplib::Response &res, const ApiError &e) {
    res.status = e.status;
    res.set_content(error_body(e), "application/json");
}

void submit(State &st, const std::shared_ptr<Job> &job) {
    {
        std::lock_guard<std::mutex> lock(st.mu);
        st.queue.push_back(job);
    }
    st.cv.notify_all();
}

void completions(State &st, const httplib::Request &req, httplib::Response &res, bool chat) {
    auto job = std::make_shared<Job>();
    try {
        Value body;
        try {
            body = Value::parse(req.body);
        } catch (const std::exception &e) {
            throw ApiError{400, std::string("the request body is not JSON: ") + e.what(), "", ""};
        }
        try {
            job->req = parse_request(body, chat, st.facts);
        } catch (const ApiError &) {
            throw;
        } catch (const std::exception &e) {  // a type error from the JSON accessors
            throw ApiError{400, e.what(), "", ""};
        }
    } catch (const ApiError &e) {
        return reply_error(res, e);
    }
    job->id = (chat ? "chatcmpl-flerp-" : "cmpl-flerp-") + std::to_string(++st.next_id);
    job->created = (int64_t)std::time(nullptr);
    submit(st, job);
    Event first;
    while (!job->pop(first, 1.0))
        if (st.stopping) return reply_error(res, {503, "the server is stopping", "", ""});
    if (first.kind == Event::Kind::Error) {  // refused before anything ran: a plain HTTP error, streamed or not
        res.status = first.status;
        return res.set_content(first.body, "application/json");
    }
    if (!job->req.stream) {
        Event e = first;
        while (e.kind != Event::Kind::Done && e.kind != Event::Kind::Error)
            while (!job->pop(e, 1.0))
                if (st.stopping) return reply_error(res, {503, "the server is stopping", "", ""});
        res.status = e.status;
        return res.set_content(e.body, "application/json");
    }
    res.set_header("Cache-Control", "no-cache");
    res.set_chunked_content_provider("text/event-stream", [job, &st](size_t, httplib::DataSink &out) {
        const auto frame = [&](const std::string &f) {
            if (!out.write(f.data(), f.size())) {
                job->cancelled = true;
                return false;
            }
            return true;
        };
        Event e;
        for (;;) {
            if (!job->pop(e, 10.0)) {  // nothing to send yet (a long prefill): an SSE comment keeps the connection
                if (!frame(": keep-alive\n\n")) return false;
                if (st.stopping) break;
                continue;
            }
            if (e.kind == Event::Kind::Data) {
                if (!frame("data: " + e.body + "\n\n")) return false;
            } else if (e.kind == Event::Kind::Error) {
                frame("data: " + e.body + "\n\n");
                break;
            } else if (e.kind == Event::Kind::Done) {
                frame("data: [DONE]\n\n");
                break;
            }
        }
        out.done();
        return true;
    });
}

Value deliveries_json(uint32_t d) {
    Value a = Value::array();
    if (d & FLERP_OUT_FULL) a.push(Value::string("full"));
    if (d & FLERP_OUT_CANDIDATES) a.push(Value::string("candidates"));
    if (d & FLERP_OUT_TOKENS) a.push(Value::string("tokens"));
    return a;
}

void routes(State &st, httplib::Server &svr) {
    svr.Post("/v1/chat/completions", [&st](const httplib::Request &req, httplib::Response &res) { completions(st, req, res, true); });
    svr.Post("/v1/completions", [&st](const httplib::Request &req, httplib::Response &res) { completions(st, req, res, false); });
    svr.Get("/v1/models", [&st](const httplib::Request &, httplib::Response &res) {
        Value m = Value::object();
        m.set("id", Value::string(st.cfg.model_id));
        m.set("object", Value::string("model"));
        m.set("owned_by", Value::string("strixflerp"));
        Value data = Value::array();
        data.push(std::move(m));
        Value v = Value::object();
        v.set("object", Value::string("list"));
        v.set("data", std::move(data));
        res.set_content(v.dump(), "application/json");
    });
    svr.Get("/health", [&st](const httplib::Request &, httplib::Response &res) {
        const flerp_caps &c = st.facts.caps;
        Value v = Value::object();
        v.set("status", Value::string(st.failed ? "failed" : "ok"));
        v.set("model", Value::string(st.cfg.model_id));
        v.set("front_end", Value::string("serve.flerp.http"));
        v.set("executor", Value::string(st.executor_describe));
        v.set("executor_abi", Value::integer(c.abi));
        v.set("deliveries", deliveries_json(c.deliveries));
        v.set("sampled_by", Value::string(st.facts.host_sampling ? "host" : "executor"));
        v.set("sampler", Value::string(st.facts.host_sampling ? flerp_sampler_describe() : "the executor's draws"));
        v.set("text", Value::string(st.text_describe));
        v.set("reply", Value::string(st.reply_describe));
        v.set("draft_form", Value::string(c.flags & FLERP_CAP_DRAFT_STEP ? "step" : c.flags & FLERP_CAP_DRAFT_HELD ? "held" : "none"));
        v.set("drafts", Value::integer(st.drafts));
        v.set("context", Value::integer(c.positions));
        v.set("conversations_at_once", Value::integer(1));
        v.set("busy", Value::boolean(st.busy.load()));
        v.set("requests_served", Value::integer(st.served.load()));
        // the fields FreeFitty's router reads (docs/FLERP-SWAP-PLAN.md section A3): one conversation at a time is one
        // slot whatever caps.seqs says, since requests run one after another on the executor's thread
        size_t queued = 0;
        {
            std::lock_guard<std::mutex> lock(st.mu);
            queued = st.queue.size();
        }
        v.set("slots", Value::integer(1));
        v.set("busy_slots", Value::integer(st.busy.load() ? 1 : 0));
        v.set("queued", Value::integer((int64_t)queued));
        v.set("context_length", Value::integer(c.positions));
        if (st.failed) res.status = 503;
        res.set_content(v.dump(), "application/json");
    });
    svr.Post("/tokenize", [&st](const httplib::Request &req, httplib::Response &res) {
        auto job = std::make_shared<Job>();
        job->what = Job::What::Tokenize;
        try {
            const Value body = Value::parse(req.body);
            const Value *c = body.find("content");
            if (c == nullptr) c = body.find("prompt");
            if (c == nullptr || !c->is_string()) throw std::runtime_error("'content' must be a string");
            job->text = c->as_string("content");
        } catch (const std::exception &e) {
            return reply_error(res, {400, std::string("/tokenize takes {\"content\": \"text\"}: ") + e.what(), "content", ""});
        }
        submit(st, job);
        Event e;
        while (!job->pop(e, 1.0))
            if (st.stopping) return reply_error(res, {503, "the server is stopping", "", ""});
        res.status = e.status;
        res.set_content(e.body, "application/json");
    });
    // routes the origins' servers have and this front end does not serve: said, not a bare 404
    const auto not_here = [](const char *what) {
        return [what](const httplib::Request &, httplib::Response &res) {
            reply_error(res, {501, std::string(what) + " is not served by this front end (serve.flerp.http)", "", "not_served"});
        };
    };
    for (const char *path : {"/cache/export", "/cache/import", "/cache/probe"}) svr.Post(path, not_here("the prompt cache"));
    svr.Get("/cache", not_here("the prompt cache"));
    svr.Get("/steer", not_here("activation steering"));
    svr.Post("/steer/reload", not_here("activation steering"));
    svr.Post("/v1/messages", not_here("the Anthropic messages API"));
    svr.Post("/v1/responses", not_here("the responses API"));
    svr.Post("/v1/embeddings", not_here("embeddings"));
}

int usage() {
    std::fprintf(stderr, "usage: strixflerp-<recipe> [--host H] [--port N] [--model-id ID] [--drafts N] [--draft-margin M] [--prompt-cut TOKEN] "
                         "[--delivery auto|tokens] [--generation-config FILE] [--think-nudge|--think-end-guard|--literal-tags|--think-close-on-call|--thinking on|off] "
                         "executor.KEY=VALUE ... text.KEY=VALUE ...\n");
    return 2;
}

}  // namespace

// The entry a host that owns process start-up calls (the Zig build of a Zig-exported executor); main() otherwise.
extern "C" int flerp_host_main(int argc, char **argv);

extern "C" int flerp_host_main(int argc, char **argv) {
    State st;
    Config &cfg = st.cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool more = i + 1 < argc;
        if (a == "--host" && more) cfg.host = argv[++i];
        else if (a == "--port" && more) cfg.port = std::atoi(argv[++i]);
        else if (a == "--model-id" && more) cfg.model_id = argv[++i];
        else if (a == "--drafts" && more) cfg.drafts = std::atoi(argv[++i]);
        else if (a == "--draft-margin" && more) cfg.margin = std::atof(argv[++i]);
        else if (a == "--prompt-cut" && more) cfg.prompt_cut = argv[++i];
        else if (a == "--delivery" && more) cfg.delivery = argv[++i];
        else if (a == "--generation-config" && more) cfg.generation_config = argv[++i];
        else if ((a == "--think-nudge" || a == "--think-end-guard" || a == "--literal-tags" || a == "--think-close-on-call" || a == "--thinking") && more) {
            const std::string v = argv[++i];
            if (v != "on" && v != "off") return usage();
            bool &b = a == "--think-nudge" ? cfg.think_nudge : a == "--think-end-guard" ? cfg.think_end_guard : a == "--literal-tags" ? cfg.literal_tags
                    : a == "--think-close-on-call" ? cfg.think_close_on_call : cfg.thinking;
            b = v == "on";
        }
        else if (a.rfind("executor.", 0) == 0 && a.find('=') != std::string::npos)
            cfg.executor.emplace_back(a.substr(9, a.find('=') - 9), a.substr(a.find('=') + 1));
        else if (a.rfind("text.", 0) == 0 && a.find('=') != std::string::npos)
            cfg.text.emplace_back(a.substr(5, a.find('=') - 5), a.substr(a.find('=') + 1));
        else return usage();
    }
    if (cfg.delivery != "auto" && cfg.delivery != "tokens") return usage();
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    // the model's text
    std::vector<flerp_kv> kv;
    for (const auto &[k, v] : cfg.text) kv.push_back({k.c_str(), v.c_str()});
    if (flerp_text_open(kv.data(), kv.size(), &st.text) != FLERP_OK) {
        std::fprintf(stderr, "startup: text: %s\n", flerp_text_last_error(nullptr));
        return 1;
    }
    st.text_describe = flerp_text_describe(st.text);
    st.facts.vocab = flerp_text_vocab(st.text);
    st.end_ids.resize(8);
    st.end_ids.resize((size_t)std::max<int64_t>(0, flerp_text_end_ids(st.text, st.end_ids.data(), st.end_ids.size())));
    std::printf("startup: text %s\n", st.text_describe.c_str());
    // the model's reply forms, opened with the text module's keys (include/flerp/reply.h)
    // (FLERP_E_UNSUPPORTED: the recipe binds no reply module; chat replies are then returned whole and tools refused)
    if (const flerp_status rs = flerp_reply_open(kv.data(), kv.size(), &st.reply); rs == FLERP_E_UNSUPPORTED) {
        st.reply = nullptr;
        st.reply_describe = std::string("none: ") + flerp_reply_last_error(nullptr);
    } else if (rs != FLERP_OK) {
        std::fprintf(stderr, "startup: reply: %s\n", flerp_reply_last_error(nullptr));
        flerp_text_close(st.text);
        return 1;
    } else {
        st.reply_describe = flerp_reply_describe(st.reply);
    }
    st.facts.reply = st.reply != nullptr;
    if (st.reply) {
        const auto ids_of = [&](const char *name) {
            const char *t = flerp_reply_text(st.reply, name);
            return t ? encode(st, t, std::strlen(t), nullptr, 0) : std::vector<flerp_token>{};
        };
        st.thinking_stop = ids_of("thinking_stop"), st.think_close = ids_of("think_close_before_call");
        st.nudge[0] = ids_of("nudge1"), st.nudge[1] = ids_of("nudge2");
        if (const char *cb = flerp_reply_text(st.reply, "call_begin")) st.call_begin = flerp_text_token_id(st.text, cb);
    }
    std::printf("startup: reply %s; think-nudge %s, think-end-guard %s, literal-tags %s, think-close-on-call %s, thinking %s\n", st.reply_describe.c_str(),
                cfg.think_nudge ? "on" : "off", cfg.think_end_guard ? "on" : "off", cfg.literal_tags ? "on" : "off", cfg.think_close_on_call ? "on" : "off",
                cfg.thinking ? "on" : "off");
    if (!cfg.prompt_cut.empty()) {
        st.cut_id = flerp_text_token_id(st.text, cfg.prompt_cut.c_str());
        if (st.cut_id < 0) {
            std::fprintf(stderr, "startup: --prompt-cut %s is not a token of this model\n", cfg.prompt_cut.c_str());
            return 1;
        }
    }
    st.facts.defaults.size = sizeof(flerp_sampling);
    st.facts.defaults.temperature = 1, st.facts.defaults.top_p = 1, st.facts.defaults.top_k = 0;
    if (!cfg.generation_config.empty()) {
        try {
            std::ifstream f(cfg.generation_config);
            std::stringstream ss;
            ss << f.rdbuf();
            const Value g = Value::parse(ss.str());
            if (const Value *v = g.find("temperature"); v && v->is_number()) st.facts.defaults.temperature = v->as_double("temperature");
            if (const Value *v = g.find("top_p"); v && v->is_number()) st.facts.defaults.top_p = v->as_double("top_p");
            if (const Value *v = g.find("top_k"); v && v->is_int()) st.facts.defaults.top_k = (int32_t)v->as_int("top_k");
        } catch (const std::exception &e) {
            std::fprintf(stderr, "startup: --generation-config %s: %s\n", cfg.generation_config.c_str(), e.what());
            return 1;
        }
    }

    // the executor
    kv.clear();
    for (const auto &[k, v] : cfg.executor) kv.push_back({k.c_str(), v.c_str()});
    flerp_open_args oa{};
    oa.size = sizeof(oa), oa.recipe = cfg.model_id.c_str(), oa.config = kv.data(), oa.n_config = kv.size();
    const double t_open = now_s();
    if (flerp_open(&oa, &st.ex) != FLERP_OK) {
        std::fprintf(stderr, "startup: executor: %s\n", flerp_last_error(nullptr));
        if (st.reply) flerp_reply_close(st.reply);
        flerp_text_close(st.text);
        return 1;
    }
    const auto shut = [&](int rc) {
        flerp_close(st.ex);
        if (st.reply) flerp_reply_close(st.reply);
        flerp_text_close(st.text);
        return rc;
    };
    flerp_caps &caps = st.facts.caps;
    caps.size = sizeof(caps);
    if (flerp_get_caps(st.ex, &caps) != FLERP_OK) {
        std::fprintf(stderr, "startup: flerp_get_caps: %s\n", flerp_last_error(st.ex));
        return shut(1);
    }
    st.facts.caps_v2 = caps.abi >= 2 && caps.size >= FLERP_HAS(flerp_caps, top_k_max);
    st.executor_describe = flerp_describe(st.ex);
    const bool logits = (caps.deliveries & (FLERP_OUT_FULL | FLERP_OUT_CANDIDATES)) != 0;
    st.facts.host_sampling = logits && cfg.delivery != "tokens";
    if (!st.facts.host_sampling && !(caps.deliveries & FLERP_OUT_TOKENS)) {
        std::fprintf(stderr, "startup: the executor offers no tokens delivery (caps.deliveries %u)\n", caps.deliveries);
        return shut(1);
    }
    if ((caps.deliveries & FLERP_OUT_FULL) && caps.logits_row < st.facts.vocab) {
        std::fprintf(stderr, "startup: the executor's logits row (%lld) is shorter than the tokenizer (%lld tokens): the text module is another model's\n",
                     (long long)caps.logits_row, (long long)st.facts.vocab);
        return shut(1);
    }
    const bool step = caps.flags & FLERP_CAP_DRAFT_STEP, held = caps.flags & FLERP_CAP_DRAFT_HELD;
    st.drafts = !step && !held ? 0 : cfg.drafts >= 0 ? std::min(cfg.drafts, caps.max_draft) : step ? std::min(7, caps.max_draft) : caps.max_draft;
    const size_t R = (size_t)std::max<int64_t>(1, caps.max_verify_rows);
    st.tokens.resize(R), st.nan.resize(R);
    st.cand_id.resize(R * (size_t)std::max(1, caps.candidates)), st.cand_logit.resize(st.cand_id.size());
    std::printf("startup: executor %s (loaded in %.1f s)\n", st.executor_describe.c_str(), now_s() - t_open);
    std::printf("startup: executor ABI %u, deliveries %u, flags %u, %lld positions, %lld rows a verify, %d seqs\n", caps.abi, caps.deliveries, caps.flags,
                (long long)caps.positions, (long long)caps.max_verify_rows, caps.seqs);
    std::printf("startup: tokens are %s; drafts %d a verify (%s), margin %g; prompt cut %s\n",
                st.facts.host_sampling ? (std::string("sampled on the host by ") + flerp_sampler_describe()).c_str() : "drawn by the executor", st.drafts,
                step ? "stepped" : held ? "held" : "no draft head", cfg.margin, cfg.prompt_cut.empty() ? "none" : cfg.prompt_cut.c_str());

    httplib::Server svr;
    svr.set_payload_max_length(64u << 20);
    svr.set_read_timeout(30, 0);
    routes(st, svr);
    if (!svr.bind_to_port(cfg.host, cfg.port)) {
        std::fprintf(stderr, "startup: cannot bind %s:%d\n", cfg.host.c_str(), cfg.port);
        return shut(1);
    }
    g_state = &st, g_server = &svr;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::signal(SIGPIPE, SIG_IGN);
    std::thread http([&] { svr.listen_after_bind(); });
    std::printf("startup: serving on http://%s:%d (one conversation at a time)\n", cfg.host.c_str(), cfg.port);

    // this thread opened the executor and is the only one that calls it
    for (;;) {
        std::shared_ptr<Job> job;
        {
            std::unique_lock<std::mutex> lock(st.mu);
            st.cv.wait_for(lock, std::chrono::milliseconds(200), [&] { return st.stopping.load() || !st.queue.empty(); });
            if (st.stopping) break;
            if (st.queue.empty()) continue;
            job = st.queue.front();
            st.queue.pop_front();
        }
        st.busy = true;
        if (job->cancelled) job->push({Event::Kind::Error, error_body({499, "the request was cancelled", "", "cancelled"}), 499});
        else run_job(st, *job);
        st.busy = false;
        ++st.served;
    }
    svr.stop();
    http.join();
    g_server = nullptr, g_state = nullptr;
    std::printf("shutdown: closing the executor\n");
    const int rc = shut(0);
    std::printf("shutdown: the executor is closed\n");
    return rc;
}

#ifndef FLERP_HOSTED
int main(int argc, char **argv) { return flerp_host_main(argc, argv); }
#endif
