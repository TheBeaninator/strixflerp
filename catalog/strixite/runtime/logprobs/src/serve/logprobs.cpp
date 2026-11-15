#include "serve/logprobs.hpp"

#include <cmath>
#include <limits>

#include "common/check.hpp"

namespace strix {

RowLogprobs row_logprobs(const float *x, int64_t n_valid, int64_t k) {
    RowLogprobs out;
    double max_v = -std::numeric_limits<double>::infinity();
    bool any = false;
    for (int64_t id = 0; id < n_valid; ++id) {
        const double v = x[id];
        if (std::isnan(v) || v > std::numeric_limits<float>::max()) {
            out.nan = true;
            continue;
        }
        if (v == -std::numeric_limits<double>::infinity()) continue;  // a masked / padded entry
        if (!any || v > max_v) max_v = v;
        any = true;
    }
    if (!any) {
        out.nan = true;
        return out;
    }
    double sum = 0;
    for (int64_t id = 0; id < n_valid; ++id) {
        const double v = x[id];
        if (!std::isfinite(v)) continue;  // -inf contributes 0, NaN flagged above
        sum += std::exp(v - max_v);
        if (k <= 0) continue;
        // Keep the k largest (logprob desc, lower id first on ties) by insertion into a tiny sorted list.
        if (out.top.size() == (size_t)k && (v <= out.top.back().logprob && !(v == out.top.back().logprob && id < out.top.back().id)))
            continue;
        TopToken t{(int32_t)id, v, {}};
        size_t at = out.top.size();
        while (at > 0 && (out.top[at - 1].logprob < v || (out.top[at - 1].logprob == v && out.top[at - 1].id > id)))
            --at;
        out.top.insert(out.top.begin() + (long)at, t);
        if (out.top.size() > (size_t)k) out.top.pop_back();
    }
    out.logsumexp = max_v + std::log(sum);
    for (TopToken &t : out.top) t.logprob -= out.logsumexp;
    return out;
}

void record_sampled(TokenLogprob &e, const float *x, int64_t n_valid, int64_t k, int32_t id) {
    STRIX_CHECK(id >= 0 && id < n_valid, "record_sampled: id ", id, " outside [0, ", n_valid, ")");
    RowLogprobs r = row_logprobs(x, n_valid, k);
    if (r.nan) return;  // has_logprob stays false: nothing about this row is quotable
    const double lp = (double)x[id] - r.logsumexp;
    if (!std::isfinite(lp)) return;  // a -inf logit was sampled (all-masked row): not a JSON number
    e.has_logprob = true;
    e.logprob = lp;
    if (k > 0) e.top = std::move(r.top);
}

void LogprobRecorder::sampled(const LogitRows &rows, int64_t r, int32_t id, std::string bytes, bool reasoning) const {
    if (off()) return;
    STRIX_CHECK(rows.cands == 0, "LogprobRecorder: logprobs need full logits rows, got a candidate row");
    STRIX_CHECK(r >= 0 && r < rows.rows && rows.row >= n_valid_, "LogprobRecorder: row ", r, " outside [",
                rows.rows, ", ", rows.row, ")");
    TokenLogprob e;
    e.id = id;
    e.token = std::move(bytes);
    e.reasoning = reasoning;
    record_sampled(e, rows.full.data() + (size_t)r * (size_t)rows.row, n_valid_, top_k_, id);
    for (TopToken &t : e.top) t.token = tok_->token_bytes(t.id);
    out_->push_back(std::move(e));
}

void LogprobRecorder::fed(int32_t id, std::string bytes, bool reasoning) const {
    if (off()) return;
    TokenLogprob e;
    e.id = id;
    e.token = std::move(bytes);
    e.fed = true;
    e.reasoning = reasoning;
    out_->push_back(std::move(e));
}

std::string utf8_sanitize(const std::string &raw) {
    // The Unicode "maximal subpart" rule (table 3-7 of the standard, what Rust's from_utf8_lossy and the WHATWG
    // decoder do): a well-formed sequence passes; the bytes before an error that did form the start of one (a bad
    // range, a cut sequence) become exactly one U+FFFD. Token pieces are cut at token boundaries, so this is the
    // common case: a token that continues the character the next entry completes still shows as one U+FFFD alone.
    std::string out;
    out.reserve(raw.size());
    const auto *p = (const unsigned char *)raw.data();
    const size_t n = raw.size();
    const auto cont = [](unsigned char b) { return (b >> 6) == 2; };
    size_t i = 0;
    while (i < n) {
        const unsigned char c = p[i];
        if (c < 0x80) {  // runs of ASCII are the whole payload in practice: copy them at a time
            size_t j = i + 1;
            while (j < n && p[j] < 0x80) ++j;
            out.append(raw, i, j - i);
            i = j;
            continue;
        }
        // The sequence length the lead byte announces, 0 when the lead itself is never valid (a stray continuation,
        // C0/C1 (always overlong) or F5+); the second byte's range catches overlongs and surrogates (E0/ED) and the
        // > U+10FFFF leads (F0/F4).
        int len = 0;
        if (c >= 0xC2 && c <= 0xDF) len = 2;
        else if (c >= 0xE0 && c <= 0xEF) {
            len = 3;
            if (c == 0xE0 && i + 1 < n && p[i + 1] < 0xA0) len = 0;
            if (c == 0xED && i + 1 < n && p[i + 1] > 0x9F) len = 0;
        } else if (c >= 0xF0 && c <= 0xF4) {
            len = 4;
            if (c == 0xF0 && i + 1 < n && p[i + 1] < 0x90) len = 0;
            if (c == 0xF4 && i + 1 < n && p[i + 1] > 0x8F) len = 0;
        }
        size_t m = i + 1;  // consume the lead, then every continuation byte that is one
        while (len > 0 && m < i + (size_t)len && m < n && cont(p[m])) ++m;
        if (len > 0 && m == i + (size_t)len) {  // complete and well-formed
            out.append(raw, i, (size_t)len);
            i = m;
            continue;
        }
        out += "\xEF\xBF\xBD";  // U+FFFD for exactly the subpart: the lead plus the valid prefix of continuations
        i = m > i + 1 ? m : i + 1;
    }
    return out;
}

}  // namespace strix
