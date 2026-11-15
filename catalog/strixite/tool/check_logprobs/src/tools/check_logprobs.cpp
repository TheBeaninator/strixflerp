// Per-token logprobs, host only (card 31cf21e3d1, ENG-LOGPROB): the row math of serve/logprobs.hpp, the engine's
// capture sites driven by a scripted fake backend (serve/engine.hpp) - greedy text, the thinking budget's fed stop,
// MTP-accepted and MTP-rejected tokens - which rows a request takes (full logits with logprobs, the candidate fast
// path without), the /v1/chat/completions parsing with and without the server's --logprobs, and the response body's
// choice.logprobs shape. The model's tokenizer.json is all it needs; no model, no GPU.
//
//   check_logprobs <tokenizer.json>

#include "common/check.hpp"
#include "serve/engine.hpp"
#include "serve/json.hpp"
#include "serve/logprobs.hpp"
#include "serve/openai.hpp"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace strix;

namespace {

int g_fail = 0;

#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                        \
            ++g_fail;                                                                                                  \
        }                                                                                                              \
    } while (0)

#define CHECK_CLOSE(a, b)                                                                                              \
    do {                                                                                                               \
        const double va = (a), vb = (b);                                                                               \
        if (!(std::fabs(va - vb) <= 1e-9 * (1.0 + std::fabs(vb)))) {                                                    \
            std::fprintf(stderr, "FAIL %s:%d: %s = %.17g, %s = %.17g\n", __FILE__, __LINE__, #a, va, #b, vb);            \
            ++g_fail;                                                                                                  \
        }                                                                                                              \
    } while (0)

// ---------------------------------------------------------------- the naive reference

struct Ref {
    bool nan = false;
    double lse = 0;
    std::vector<std::pair<double, int32_t>> top;  // (logit, id) desc, lower id first on ties
};

Ref ref_row(const std::vector<float> &x, int64_t n_valid, int64_t k) {
    Ref r;
    std::vector<double> v((size_t)n_valid);
    bool any = false;
    double max_v = 0;
    for (int64_t id = 0; id < n_valid; ++id) {
        v[(size_t)id] = x[(size_t)id];
        if (std::isnan(v[(size_t)id]) || v[(size_t)id] > std::numeric_limits<float>::max()) r.nan = true;
        if (std::isfinite(v[(size_t)id])) {
            if (!any || v[(size_t)id] > max_v) max_v = v[(size_t)id];
            any = true;
        }
    }
    if (!any || r.nan) {
        r.nan = true;
        return r;
    }
    double sum = 0;
    for (double d : v)
        if (std::isfinite(d)) sum += std::exp(d - max_v);
    r.lse = max_v + std::log(sum);
    for (int64_t id = 0; id < n_valid; ++id) r.top.emplace_back(v[(size_t)id], (int32_t)id);
    std::sort(r.top.begin(), r.top.end(),
              [](const std::pair<double, int32_t> &a, const std::pair<double, int32_t> &b) {
                  return a.first != b.first ? a.first > b.first : a.second < b.second;
              });
    if (k >= 0 && r.top.size() > (size_t)k) r.top.resize((size_t)k);
    return r;
}

// ---------------------------------------------------------------- the scripted fake backend

// Logits as a function of the token just generated (ctx) and the id (v): a bounded pseudo-random base, with +300 on
// the scripted successor - which is then the row's unique argmax. The row is logits_row() floats: n_valid usable
// ones plus three padding floats of 1e30, which a correct row_logprobs never looks at (it scans [0, n_valid) only).
class FakeBackend : public LmBackend {
public:
    int64_t capacity() const override { return 4096; }
    int64_t max_chunk() const override { return 8192; }
    int64_t logits_row() const override { return n_valid_ + 3; }
    int64_t n_valid() const { return n_valid_; }

    void setup(int64_t n_valid, int32_t im_end) { n_valid_ = n_valid, im_end_ = im_end; }
    void script(int32_t after, int32_t chosen) { next_[after] = chosen; }
    void draft_override(int32_t after, int32_t d) { drafts_[after] = d; }
    void expect_mtp(bool on) { mtp_ = on; }
    int32_t argmax(int32_t ctx) const {
        const auto it = next_.find(ctx);
        return it == next_.end() ? im_end_ : it->second;
    }
    float logit_for(int32_t ctx, int64_t v) const {
        const double base = -(double)(((uint32_t)ctx * 2654435761u + (uint32_t)v * 40503u) % 997u) / 10.0;
        return (float)(v < n_valid_ && v == argmax(ctx) ? base + 300.0 : base);
    }
    // The full row (usable floats plus padding) for context ctx.
    std::vector<float> row_for(int32_t ctx) const {
        std::vector<float> r((size_t)logits_row(), 1e30f);
        for (int64_t v = 0; v < n_valid_; ++v) r[(size_t)v] = logit_for(ctx, v);
        return r;
    }
    int64_t full_calls = 0, cand_calls = 0;  // which rows the requests took

    bool has_mtp() const override { return mtp_; }
    Top2 forward_mtp_top2(int32_t token_id, int64_t) override {
        const auto it = drafts_.find(token_id);
        const int32_t d = it == drafts_.end() ? argmax(token_id) : it->second;
        return Top2{d, 100.0f, 0.0f, false};
    }

    LogitRows forward_rows(const std::vector<int32_t> &ids, int64_t n_logits, bool cands, int64_t,
                           const LogitMasks *) override {
        return rows_of(ids, n_logits, cands);
    }
    LogitRows forward_verify_rows(const std::vector<int32_t> &ids, int64_t n_logits, bool cands, int64_t,
                                  const LogitMasks *) override {
        snap_[kDraftSlot] = pos_;  // what the base forward_verify would have saved
        return rows_of(ids, n_logits, cands);
    }
    std::vector<float> forward(const std::vector<int32_t> &ids, bool want_logits) override {
        pos_ += (int64_t)ids.size();
        return want_logits ? row_for(ids.back()) : std::vector<float>{};
    }
    void keep_verify() override {}
    void keep_verify_prefix(const std::vector<int32_t> &, int64_t rows) override {
        pos_ = snap_[kDraftSlot] + rows;
    }

    int64_t pos() const override { return pos_; }
    void reset() override { pos_ = 0; }
    void save_snapshot(int slot) override { snap_[slot] = pos_; }
    int64_t snapshot_pos(int slot) const override { return snap_[slot]; }
    bool can_restore_snapshot(int slot) const override { return snap_[slot] >= 0; }
    void restore_snapshot(int slot) override { pos_ = snap_[slot]; }
    void export_snapshot(int, HostBuffer &out, int64_t) override { out.resize(0); }
    uint64_t snapshot_bytes(int, int64_t) const override { return 0; }
    void import_state(const HostBuffer &, int64_t n, int64_t) override { pos_ = n; }
    std::string state_fingerprint() const override { return "fake"; }
    std::string describe() const override { return "fake"; }

private:
    LogitRows rows_of(const std::vector<int32_t> &ids, int64_t n_logits, bool cands) {
        pos_ += (int64_t)ids.size();
        if (cands) ++cand_calls;
        else ++full_calls;
        if (n_logits == 0) return LogitRows::from_full({}, logits_row());
        std::vector<float> flat, cand_v;
        std::vector<int32_t> cand_id;
        for (int64_t j = (int64_t)ids.size() - n_logits; j < (int64_t)ids.size(); ++j) {
            const int32_t ctx = ids[(size_t)j];
            if (!cands) {
                const std::vector<float> r = row_for(ctx);
                flat.insert(flat.end(), r.begin(), r.end());
                continue;
            }
            // A candidate row: the scripted argmax first, then filler ids greedy never reaches.
            const float top_v = logit_for(ctx, argmax(ctx));
            for (int64_t c = 0; c < Sampler::kCandidates; ++c) {
                cand_v.push_back(c == 0 ? top_v : (float)-1e30);
                cand_id.push_back(c == 0 ? argmax(ctx) : (int32_t)c + 1);
            }
        }
        LogitRows l;
        l.rows = n_logits;
        if (cands) {
            l.row = 0;
            l.cands = Sampler::kCandidates;
            l.cand_v = std::move(cand_v);
            l.cand_id = std::move(cand_id);
            l.nan.assign((size_t)n_logits, 0);
        } else {
            l = LogitRows::from_full(std::move(flat), logits_row());
        }
        return l;
    }

    int64_t n_valid_ = 0, pos_ = 0;
    int32_t im_end_ = -1;
    bool mtp_ = false;
    std::map<int32_t, int32_t> next_, drafts_;
    int64_t snap_[4] = {-1, -1, -1, -1};
};

// ---------------------------------------------------------------- running the engine

struct TestSink : GenerationSink {
    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    GenerationResult res;
    std::string text, reasoning;

    void on_start() override {}
    void on_events(std::vector<OutputEvent> &ev) override {
        for (OutputEvent &e : ev) {
            if (e.kind == OutputEvent::Kind::Reasoning) reasoning += e.text;
            else if (e.kind == OutputEvent::Kind::Content) text += e.text;
        }
    }
    void on_done(const GenerationResult &r) override {
        {
            std::lock_guard<std::mutex> lock(mu);
            res = r;
        }
        std::lock_guard<std::mutex> lock(mu);
        done = true;
        cv.notify_all();
    }
    bool cancelled() override { return false; }
    GenerationResult wait() {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [this] { return done; });
        return res;
    }
};

struct Run {
    FakeBackend be;
    GenerationResult res;
    std::string text, reasoning;
};

Run run_engine(const Tokenizer &tok, const std::vector<int32_t> &prompt, const std::map<int32_t, int32_t> &script,
               const std::map<int32_t, int32_t> &drafts, bool logprobs, int64_t top, bool thinking,
               int64_t thinking_budget, bool mtp, int64_t max_tokens = 64) {
    Run run;
    run.be.setup(tok.size(), tok.id_of("<|im_end|>"));
    run.be.expect_mtp(mtp);
    for (const auto &[k, v] : script) run.be.script(k, v);
    for (const auto &[k, v] : drafts) run.be.draft_override(k, v);
    Engine::Options opts;
    opts.think_nudge = false;
    if (mtp) opts.mtp_draft = 2, opts.mtp_margin = 0.0f;
    GenerationRequest req;
    req.prompt = prompt;
    req.max_tokens = max_tokens;
    req.sampling = SamplingParams{};
    req.sampling.temperature = 0;  // greedy
    req.parser.thinking = thinking;
    req.thinking_budget = thinking_budget;
    req.logprobs = logprobs;
    req.logprobs_top = top;
    req.id = 1;
    auto sink = std::make_shared<TestSink>();
    {
        Engine engine(run.be, tok, opts);
        engine.submit(std::move(req), sink);
        run.res = sink->wait();
    }
    run.text = sink->text;
    run.reasoning = sink->reasoning;
    return run;
}

// One sampled entry's expected numbers, recomputed (naive, double) from the fake's own row for context ctx.
void check_entry(const Tokenizer &tok, const FakeBackend &be, const TokenLogprob &e, int32_t ctx, int64_t top_k) {
    CHECK(e.id == be.argmax(ctx));
    CHECK(e.token == tok.token_bytes(e.id));
    CHECK(!e.fed);
    CHECK(e.has_logprob);
    if (!e.has_logprob) return;
    const Ref r = ref_row(be.row_for(ctx), be.n_valid(), top_k);
    CHECK_CLOSE(e.logprob, (double)be.logit_for(ctx, e.id) - r.lse);
    CHECK((int64_t)e.top.size() == std::min<int64_t>(top_k, (int64_t)r.top.size()));
    for (size_t j = 0; j < e.top.size() && j < r.top.size(); ++j) {
        CHECK(e.top[j].id == r.top[j].second);
        CHECK_CLOSE(e.top[j].logprob, r.top[j].first - r.lse);
        CHECK(e.top[j].token == tok.token_bytes(e.top[j].id));
    }
}

// ---------------------------------------------------------------- the engine and the math

void test_math() {
    // row_logprobs on a hand-made row: -inf entries out of everything, top-k by value, lower id first on ties.
    const float inf = -INFINITY;
    std::vector<float> x = {0.0f, inf, 2.0f, -1.0f, 2.0f, 0.5f};
    RowLogprobs r = row_logprobs(x.data(), (int64_t)x.size(), 3);
    CHECK(!r.nan);
    const double max_v = 2.0;
    double sum = 0;
    for (double v : {0.0, 2.0, -1.0, 2.0, 0.5}) sum += std::exp(v - max_v);
    CHECK_CLOSE(r.logsumexp, max_v + std::log(sum));
    CHECK(r.top.size() == 3);
    CHECK(r.top[0].id == 2 && r.top[1].id == 4);  // the tie: lower id first
    CHECK_CLOSE(r.top[0].logprob, 2.0 - r.logsumexp);
    CHECK(r.top[2].id == 5);
    RowLogprobs r0 = row_logprobs(x.data(), (int64_t)x.size(), 0);
    CHECK(r0.top.empty() && r0.logsumexp == r.logsumexp);
    std::vector<float> all_inf(4, inf);
    CHECK(row_logprobs(all_inf.data(), 4, 2).nan);
    CHECK(row_logprobs(x.data(), 6, 10).top.size() == 5);  // k beyond the row: capped at n_valid

    // utf8_sanitize: valid text passes; anything else becomes U+FFFD per maximal subpart (one for a cut sequence -
    // the token-boundary case - and one per byte that is itself no valid lead: overlong C0, stray continuation).
    const std::string FFFD = "\xEF\xBF\xBD";
    CHECK(utf8_sanitize("hello") == "hello");
    CHECK(utf8_sanitize("\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80") == "\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80");
    CHECK(utf8_sanitize("\xE2\x82") == FFFD);                              // cut at end of input: one
    CHECK(utf8_sanitize("a\xC0\x80" "b") == "a" + FFFD + FFFD + "b");      // overlong lead, then a stray continuation
    CHECK(utf8_sanitize("\xED\xA0\x80") == FFFD + FFFD + FFFD);            // surrogate: ED is no valid prefix
    CHECK(utf8_sanitize("\x80\x41") == FFFD + "A");                        // stray continuation
    CHECK(utf8_sanitize("\xF0\x9F\x98") == FFFD);                          // cut mid 4-byte: one
    CHECK(utf8_sanitize("\xE2\x82\x41") == FFFD + "A");                    // bad continuation ends the subpart
}

void test_engine(const Tokenizer &tok) {
    const int32_t im_end = tok.id_of("<|im_end|>");
    const std::vector<int32_t> prompt = tok.encode("<|im_start|>user\nhi\n\n<|im_start|>assistant\n");
    const int32_t w1 = tok.encode("yes")[0];  // no leading space: the parser trims the message's leading whitespace
    const int32_t w2 = tok.encode("now")[0];
    const int32_t other = tok.encode("later")[0];

    // A: greedy text - logprobs off, on with top 0, on with top 5. Same sampled tokens all three ways; off takes
    // candidate rows, both on-runs full rows.
    std::map<int32_t, int32_t> script;
    script[prompt.back()] = w1;
    script[w1] = w2;
    script[w2] = im_end;
    Run off = run_engine(tok, prompt, script, {}, false, 0, false, -1, false);
    CHECK(off.res.finish_reason == "stop");
    CHECK(off.res.completion_tokens == 3);  // w1, w2 and the <tool_call> that ended the turn (never emitted)
    CHECK(off.text == tok.decode({w1, w2}));
    CHECK(off.res.logprobs.empty());
    CHECK(off.be.cand_calls > 0 && off.be.full_calls == 0);

    Run top0 = run_engine(tok, prompt, script, {}, true, 0, false, -1, false);
    CHECK(top0.text == off.text && top0.res.completion_tokens == off.res.completion_tokens);
    CHECK(top0.be.cand_calls == 0);  // every row came back full
    CHECK(top0.res.logprobs.size() == 2);
    CHECK(top0.res.logprobs[0].has_logprob && top0.res.logprobs[1].has_logprob);
    CHECK(top0.res.logprobs[0].top.empty() && top0.res.logprobs[1].top.empty());

    Run top5 = run_engine(tok, prompt, script, {}, true, 5, false, -1, false);
    CHECK(top5.text == off.text);
    // Entries are the emitted tokens: everything counted except the end-of-turn that was never emitted.
    CHECK(top5.res.logprobs.size() + 1 == (size_t)top5.res.completion_tokens);
    {
        int32_t ctx = prompt.back();
        for (const TokenLogprob &e : top5.res.logprobs) {
            check_entry(tok, top5.be, e, ctx, 5);
            CHECK(!e.reasoning);
            ctx = e.id;
        }
        CHECK(ctx == w2);
    }

    // B: the thinking budget's fed stop. Two reasoning tokens, then the engine feeds kThinkingStop: entries for the
    // fed tokens, no logprob, marked fed; the count still matches completion_tokens.
    const std::vector<int32_t> stop_ids = tok.encode(kThinkingStop);
    const int32_t last_stop = stop_ids.back();
    std::map<int32_t, int32_t> sc = script;
    sc[prompt.back()] = w1, sc[w1] = w2, sc[w2] = w1;  // it would go on thinking
    sc[last_stop] = im_end;                             // and does, once the block is closed
    Run stop = run_engine(tok, prompt, sc, {}, true, 3, true, 2, false);
    CHECK(stop.res.finish_reason == "stop");
    CHECK(stop.res.thinking_budget_hit);
    CHECK(stop.res.logprobs.size() + 1 == (size_t)stop.res.completion_tokens);
    CHECK(stop.res.logprobs.size() == 2 + stop_ids.size());
    {
        int32_t ctx = prompt.back();
        for (size_t j = 0; j < stop.res.logprobs.size(); ++j) {
            const TokenLogprob &e = stop.res.logprobs[j];
            if (j < 2) {
                check_entry(tok, stop.be, e, ctx, 3);
                CHECK(e.reasoning);
                ctx = e.id;
            } else {
                const int32_t fed_id = stop_ids[j - 2];
                CHECK(e.fed && !e.has_logprob);
                CHECK(e.id == fed_id && e.token == tok.token_bytes(fed_id));
                ctx = fed_id;
            }
        }
        CHECK(stop.res.logprobs[2].fed && stop.res.logprobs[2].reasoning);  // the stop opens inside the reasoning
        CHECK(stop.res.fed_tokens == (int64_t)stop_ids.size());
        CHECK(!stop.reasoning.empty());
    }

    // C: MTP. Every draft accepted - entries for the sampled tokens, values from the verify rows. Then a rejected
    // draft at the last position, where the trunk wants the end of turn: accepted entries plus nothing for im_end.
    std::map<int32_t, int32_t> acc;
    acc[prompt.back()] = w1, acc[w1] = w2, acc[w2] = other, acc[other] = im_end;
    Run mtp = run_engine(tok, prompt, acc, {}, true, 4, false, -1, true);
    CHECK(mtp.res.finish_reason == "stop");
    CHECK(mtp.res.mtp_accepted_tokens >= 2);
    CHECK(mtp.text == tok.decode({w1, w2, other}));
    CHECK(mtp.res.logprobs.size() + 1 == (size_t)mtp.res.completion_tokens);
    {
        int32_t ctx = prompt.back();
        for (const TokenLogprob &e : mtp.res.logprobs) {
            check_entry(tok, mtp.be, e, ctx, 4);
            ctx = e.id;
        }
    }
    // The head drafts `other` after w2, the trunk's row says im_end: rejected there, and that ends the turn.
    std::map<int32_t, int32_t> rej_script{{prompt.back(), w1}, {w1, w2}, {w2, im_end}};
    std::map<int32_t, int32_t> rej_drafts{{w2, other}};
    Run rj = run_engine(tok, prompt, rej_script, rej_drafts, true, 4, false, -1, true);
    CHECK(rj.res.finish_reason == "stop");
    CHECK(rj.res.mtp_rollbacks == 1);
    CHECK(rj.text == tok.decode({w1, w2}));
    CHECK(rj.res.logprobs.size() + 1 == (size_t)rj.res.completion_tokens);
    {
        int32_t ctx = prompt.back();
        for (const TokenLogprob &e : rj.res.logprobs) {
            check_entry(tok, rj.be, e, ctx, 4);
            ctx = e.id;
        }
    }
}

// ---------------------------------------------------------------- the API layer

// Parses, expects `ok`, or a 400 whose message contains `msg`; `out` gets the request when it parsed.
void try_parse(const char *body, bool enabled, const char *msg, ChatRequest *out = nullptr) {
    json::Value b;
    try {
        b = json::Value::parse(body);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: test body does not parse: %s\n", e.what());
        ++g_fail;
        return;
    }
    ChatRequest r;
    bool thrown = false;
    std::string message;
    try {
        r = parse_chat_request(b, SamplingParams{}, /*vision_enabled=*/false, enabled);
    } catch (const ApiError &e) {
        thrown = true;
        message = e.what();
        CHECK(e.status == 400);
    }
    if (msg) {
        if (!thrown) std::fprintf(stderr, "FAIL: expected a 400 (%s) from: %s\n", msg, body);
        else if (message.find(msg) == std::string::npos)
            std::fprintf(stderr, "FAIL: 400 says \"%s\", expected to contain \"%s\"\n", message.c_str(), msg);
        if (!thrown || message.find(msg) == std::string::npos) ++g_fail;
    } else if (thrown) {
        std::fprintf(stderr, "FAIL: unexpected 400 (%s) from: %s\n", message.c_str(), body);
        ++g_fail;
    } else if (out) {
        *out = std::move(r);
    }
}

void test_parse() {
    const char *kMsg = "logprobs are not supported";
    // The flag off: exactly today's refusals, and "logprobs": false still passes through untouched.
    try_parse(R"({"messages":[{"role":"user","content":"hi"}],"logprobs":true})", false, kMsg);
    try_parse(R"({"messages":[{"role":"user","content":"hi"}],"logprobs":false})", false, nullptr);
    try_parse(R"({"messages":[{"role":"user","content":"hi"}],"top_logprobs":3})", false, kMsg);
    // On: accepted, validated.
    try_parse(R"({"messages":[{"role":"user","content":"hi"}],"logprobs":true,"top_logprobs":21})", true, "0..20");
    try_parse(R"({"messages":[{"role":"user","content":"hi"}],"logprobs":true,"top_logprobs":-1})", true, "0..20");
    try_parse(R"({"messages":[{"role":"user","content":"hi"}],"top_logprobs":2})", true, "requires");
    try_parse(R"({"messages":[{"role":"user","content":"hi"}],"logprobs":true,"stream":true})", true, "stream");
    // The values land on the request.
    ChatRequest r;
    try_parse(R"({"messages":[{"role":"user","content":"hi"}],"logprobs":true,"top_logprobs":7})", true, nullptr, &r);
    CHECK(r.logprobs && r.logprobs_top == 7);
    // Legacy /v1/completions: refused whatever the flag (its logprobs is a count).
    bool thrown = false;
    try {
        parse_completion_request(json::Value::parse(R"({"prompt":"hi","logprobs":2})"), SamplingParams{});
    } catch (const ApiError &e) {
        thrown = true;
        CHECK(std::string(e.what()).find(kMsg) != std::string::npos);
    }
    CHECK(thrown);
}

void test_body() {
    GenerationResult r;
    r.finish_reason = "stop";
    r.completion_tokens = 2;
    TokenLogprob e0;
    e0.id = 100;
    e0.token = "\xE2\x82";  // a cut euro sign: invalid alone, valid continued by the next token
    e0.has_logprob = true;
    e0.logprob = -0.25;
    e0.top.push_back(TopToken{100, -0.25, "\xE2\x82"});
    e0.top.push_back(TopToken{7, -3.5, "b"});
    TokenLogprob e1;
    e1.id = 200;
    e1.token = "\xAC!";  // completes it
    e1.fed = true;
    e1.reasoning = true;
    r.logprobs = {e0, e1};
    const ResponseMeta meta{"chatcmpl-x", "m", 1, "on"};
    const std::string body = completion_body(meta, "thinking", "hi!", {}, r, false, true);
    json::Value b = json::Value::parse(body);
    CHECK(!b.find("error"));
    const json::Value &choice = b.find("choices")->as_array("choices")[0];
    const json::Value *lp = choice.find("logprobs");
    CHECK(lp && lp->is_object());
    if (!lp || !lp->is_object()) return;
    CHECK(lp->find("refusal") && lp->find("refusal")->is_null());
    const std::vector<json::Value> &content = lp->find("content")->as_array("content");
    CHECK(content.size() == 2);
    const json::Value &c0 = content[0];
    CHECK(c0.find("token")->as_string("token") == "\xEF\xBF\xBD");  // sanitized for the wire
    CHECK_CLOSE(c0.find("logprob")->as_double("logprob"), -0.25);
    {
        std::string joined;
        for (const json::Value &x : c0.find("bytes")->as_array("bytes")) joined += (char)(unsigned char)x.as_int("byte");
        CHECK(joined == "\xE2\x82");
    }
    CHECK(c0.find("top_logprobs")->as_array("top_logprobs").size() == 2);
    CHECK(c0.find("top_logprobs")->as_array("top_logprobs")[1].find("token")->as_string("token") == "b");
    CHECK_CLOSE(c0.find("top_logprobs")->as_array("top_logprobs")[1].find("logprob")->as_double("logprob"), -3.5);
    const json::Value &c1 = content[1];
    CHECK(c1.find("logprob")->is_null());
    CHECK(c1.find("strix_fed") && c1.find("strix_fed")->is_bool() && c1.find("strix_fed")->as_bool("strix_fed"));
    CHECK(c1.find("strix_channel") && c1.find("strix_channel")->as_string("strix_channel") == "reasoning");
    CHECK(!c0.find("strix_fed") && !c0.find("strix_channel"));
    // A request that did not ask: the body is exactly what it always was (no logprobs key at all).
    json::Value b2 = json::Value::parse(completion_body(meta, "thinking", "hi!", {}, r, false, false));
    CHECK(b2.find("choices")->as_array("choices")[0].find("logprobs") == nullptr);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: check_logprobs <tokenizer.json>\n");
        return 2;
    }
    const Tokenizer tok(argv[1]);
    test_math();
    test_engine(tok);
    test_parse();
    test_body();
    if (g_fail) {
        std::fprintf(stderr, "%d check(s) FAILED\n", g_fail);
        return 1;
    }
    std::printf("check_logprobs: ok\n");
    return 0;
}
