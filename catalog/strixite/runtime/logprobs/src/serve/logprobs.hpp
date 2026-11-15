#pragma once

// Per-token logprobs for /v1/chat/completions requests with "logprobs": true ("top_logprobs" up to 20), behind the
// server's --logprobs (card 31cf21e3d1, ENG-LOGPROB). The wire shape is OpenAI's: each generated token carries the
// log-softmax logprob of what was sampled, over the very FP32 logits row the sampler saw (after the structured-output
// mask and the think guard, before temperature - the HF-reference meaning), plus the row's top tokens. The engine
// takes those rows as FULL logits (LogitRows::full): a logprob needs the row's normalisation constant, and only the
// full row has it. Tokens the engine fed itself (the thinking stop, the nudge, the close before a misplaced call)
// have no row of their own: they get an entry without a logprob, so the entries map 1:1 to the generation order.

#include <cstdint>
#include <string>
#include <vector>

#include "serve/sampler.hpp"
#include "serve/tokenizer.hpp"

namespace strix {

// One token of a row's top list: its id, its log-softmax logprob, and its bytes (filled by the recorder, who has
// the tokenizer; may end mid-UTF-8 sequence - the wire sanitizes, bytes carry the raw form).
struct TopToken {
    int32_t id = -1;
    double logprob = 0;
    std::string token;
};

// One entry of GenerationResult::logprobs: a token of the generation, in generation order (sampled and fed ones
// alike). openai.cpp serializes these into choice.logprobs.content.
struct TokenLogprob {
    int32_t id = -1;
    std::string token;         // the token's raw bytes; may end mid-UTF-8 sequence (OpenAI's bytes carries these)
    bool has_logprob = false;  // false: engine-fed, or a row with no finite value (then nothing is quotable)
    double logprob = 0;        // natural log of the sampled token's probability, when has_logprob
    bool fed = false;          // emitted by the engine, not sampled
    bool reasoning = false;    // was fed to the parser while the think block was open (strix_channel on the wire)
    std::vector<TopToken> top;  // descending logprob; empty unless has_logprob
};

// The log-softmax pieces of one logits row of n_valid ids: the logsumexp and the k biggest values (ties: lower id
// first). -inf entries contribute nothing to the sum and never enter the top list; any other non-finite value
// means nan (the row as the sampler would have seen it: it throws on such a row only when asked for it).
struct RowLogprobs {
    bool nan = false;
    double logsumexp = 0;
    std::vector<TopToken> top;  // size min(k, n_valid)
};
RowLogprobs row_logprobs(const float *x, int64_t n_valid, int64_t k);

// Fills e.logprob / e.top / e.has_logprob for token id sampled from the full logits row x[0, n_valid).
void record_sampled(TokenLogprob &e, const float *x, int64_t n_valid, int64_t k, int32_t id);

// What Engine::run carries when the request wants logprobs (null out: off, every method is a no-op).
class LogprobRecorder {
public:
    LogprobRecorder(std::vector<TokenLogprob> *out, const Tokenizer &tok, int64_t n_valid, int64_t top_k)
        : out_(out), tok_(&tok), n_valid_(n_valid), top_k_(top_k) {}
    bool off() const { return out_ == nullptr; }
    // A sampled token from rows.row(r) (a full row: logprobs requests take no candidate rows).
    void sampled(const LogitRows &rows, int64_t r, int32_t id, std::string bytes, bool reasoning) const;
    // An engine-fed token (no logits row of its own).
    void fed(int32_t id, std::string bytes, bool reasoning) const;

private:
    std::vector<TokenLogprob> *out_ = nullptr;
    const Tokenizer *tok_ = nullptr;
    int64_t n_valid_ = 0, top_k_ = 0;
};

// A token's bytes as valid UTF-8: maximal invalid / incomplete sub-sequences become one U+FFFD (the wire's token
// field is a JSON string; OpenAI's own servers sanitize the same way - bytes keeps the raw form).
std::string utf8_sanitize(const std::string &raw);

}  // namespace strix
