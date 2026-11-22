// strixflerp-qsa-overlap: how much of one full-attention layer's QSA block selection the next one repeats.
//
// The question it answers (IndexCache, owner's 2026-11-16 package): a cross-layer index cache reuses layer a's top-k
// blocks as layer b's candidates and rescoring only those, so it can be exact for a query only when b's own top-k
// sits inside a's. The upper bound on what such a cache recovers is the recall |S_a ∩ S_b| / |S_b|, measured here
// on the engine's own selections (the qsa_sel / qsa_nsel probe points of executor.strixite.qwen4exp), not modelled.
//
// One weights load; for each prompt length L the first L tokens of a real text are prefilled in chunks, then
// --decode greedy T = 1 steps run. Rows recorded: every --stride-th row of each prefill chunk past the dense limit,
// and every decode row. A row counts only when both layers kept a full k blocks (below that the indexer keeps every
// block and the overlap is 1 by construction).
//
//   strixflerp-qsa-overlap --weights W --ngram T --tokenizer J --text FILE --out DIR [--lens 8192,32768]
//       [--decode 16] [--stride 256] [--chunk 8192] [--capacity N]
//
// Writes DIR/overlap.jsonl, one line per (len, phase, layer pair): pair (layer a, layer b), distance in attention
// layers (1 = adjacent), rows, mean / p10 / p50 / p90 recall, the random baseline k / blocks, and the share of b's
// selection inside the 16 newest blocks and the first 4 (recency and sink, which any cache would keep anyway).
#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "kernels/qsa.hpp"  // kQsaBlock
#include "runtime/qwen4exp.hpp"
#include "serve/tokenizer.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace strix;

namespace {

std::string slurp(const std::string &p) {
    std::ifstream f(p, std::ios::binary);
    STRIX_CHECK(f.good(), "qsa_overlap: cannot read ", p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

std::vector<int64_t> split_i64(const std::string &s) {
    std::vector<int64_t> out;
    std::stringstream ss(s);
    for (std::string p; std::getline(ss, p, ',');)
        if (!p.empty()) out.push_back(std::stoll(p));
    return out;
}

// One recorded query row: its position and, per attention layer (in layer order), the selected blocks (ascending).
struct Row {
    int64_t pos = 0;
    std::vector<std::vector<int32_t>> sel;  // [attention layer index] -> blocks; empty = not a full-k row
};

struct Stats {
    std::vector<double> recall, jaccard;
    double baseline = 0, recent = 0, sink = 0;
};

double quantile(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[(size_t)std::min<double>((double)v.size() - 1, q * (double)(v.size() - 1) + 0.5)];
}

}  // namespace

int main(int argc, char **argv) {
    try {
        std::map<std::string, std::string> a;
        for (int i = 1; i < argc; ++i) {
            std::string k = argv[i];
            STRIX_CHECK(k.rfind("--", 0) == 0 && i + 1 < argc, "qsa_overlap: bad argument ", k);
            a[k.substr(2)] = argv[++i];
        }
        auto get = [&](const std::string &k, const std::string &d = "") { return a.count(k) ? a[k] : d; };
        const std::string out_dir = get("out");
        STRIX_CHECK(!get("weights").empty() && !get("ngram").empty() && !get("tokenizer").empty() && !get("text").empty() &&
                        !out_dir.empty(),
                    "usage: strixflerp-qsa-overlap --weights W --ngram T --tokenizer J --text FILE --out DIR [--lens L,..]");
        const std::vector<int64_t> lens = split_i64(get("lens", "8192,32768"));
        const int64_t n_decode = std::stoll(get("decode", "16")), stride = std::stoll(get("stride", "256"));
        const int64_t chunk = std::stoll(get("chunk", "8192"));
        const bool use_probe = get("probe", "1") != "0";   // --probe 0: no probe copies (the unlogged timing run, V10-QSA)
        auto mono_ns = [] { return (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
        const int64_t max_len = *std::max_element(lens.begin(), lens.end());
        const int64_t capacity = std::stoll(get("capacity", std::to_string(max_len + n_decode + 64)));

        const Tokenizer tok(get("tokenizer"));  // the constructor reads the file
        const std::vector<int32_t> all = tok.encode(slurp(get("text")));
        STRIX_CHECK((int64_t)all.size() >= max_len, "qsa_overlap: the text has ", all.size(), " tokens, ", max_len,
                    " wanted (no repeats: a repeated text would select its own copy)");

        const double t0 = now_s();
        Qwen4ExpModel model(get("weights"), get("ngram"), kernels::Act::BF16, false, 8388608, 1.0f);
        const auto &D = model.dims();
        std::vector<int64_t> attn;  // attention layer indices
        for (int64_t i = 0; i < D.layers; ++i)
            if (D.is_attention[(size_t)i]) attn.push_back(i);
        std::map<int64_t, size_t> attn_slot;
        for (size_t j = 0; j < attn.size(); ++j) attn_slot[attn[j]] = j;
        const int64_t k = D.qsa_budget / kernels::kQsaBlock;
        std::fprintf(stderr, "qsa_overlap: loaded in %.1f s; %zu attention layers, k %lld blocks, text %zu tokens\n",
                     now_s() - t0, attn.size(), (long long)k, all.size());

        Qwen4ExpSession ses(model, capacity, chunk, PrefillMath::WmmaBf16, false, parse_kv_format(get("kv", "q8")));
        std::ofstream out(out_dir + "/overlap.jsonl", std::ios::app);
        STRIX_CHECK(out.good(), "qsa_overlap: cannot write ", out_dir, "/overlap.jsonl");

        for (const int64_t L : lens) {
            std::map<std::string, std::vector<Row>> rows;  // phase -> rows
            int64_t base = 0;                              // position of row 0 of the current forward
            bool decoding = false;
            // per forward: the nsel row counts arrive after sel, so keep sel until nsel comes
            std::map<size_t, std::vector<int32_t>> pending_sel;
            std::vector<Row> cur;  // the current forward's recorded rows
            auto keep_row = [&](int64_t r) { return decoding || r % stride == stride - 1; };
            const Qwen4ExpProbe probe = [&](const std::string &name, const void *dev, int64_t T, int64_t cols, ProbeType type) {
                const size_t dot = name.find('.');
                if (dot == std::string::npos) return;
                const std::string point = name.substr(dot + 1);
                if (point != "qsa_sel" && point != "qsa_nsel") return;
                STRIX_CHECK(type == ProbeType::I32, "qsa_overlap: ", name, " is not int32");
                const int64_t layer = std::stoll(name.substr(1, dot - 1));
                const size_t slot = attn_slot.at(layer);
                std::vector<int32_t> h((size_t)(T * cols));
                STRIX_HIP_CHECK(hipMemcpy(h.data(), dev, h.size() * 4, hipMemcpyDeviceToHost), "probe copy ", name);
                if (point == "qsa_sel") {
                    STRIX_CHECK(cols == k, "qsa_overlap: qsa_sel has ", cols, " columns, k = ", k);
                    pending_sel[slot] = std::move(h);
                    return;
                }
                const std::vector<int32_t> &sel = pending_sel.at(slot);
                if (cur.empty())
                    for (int64_t r = 0; r < T; ++r)
                        if (keep_row(r)) {
                            Row x;
                            x.pos = base + r;
                            x.sel.resize(attn.size());
                            cur.push_back(x);
                        }
                for (Row &x : cur) {
                    const int64_t r = x.pos - base;
                    if (h[(size_t)r] != k) continue;  // the indexer kept every block: not a selection row
                    std::vector<int32_t> s(sel.begin() + r * k, sel.begin() + (r + 1) * k);
                    std::sort(s.begin(), s.end());
                    x.sel[slot] = std::move(s);
                }
            };
            auto run = [&](const std::vector<int32_t> &ids, int64_t want, const char *phase) {
                cur.clear(), pending_sel.clear();
                std::vector<float> lg = use_probe ? ses.forward(ids, want, probe) : ses.forward(ids, want);
                for (Row &x : cur) rows[phase].push_back(std::move(x));
                base += (int64_t)ids.size();
                return lg;
            };
            ses.reset();
            base = 0, decoding = false;
            const double tp = now_s();
            std::vector<float> lg;
            for (int64_t pos = 0; pos < L;) {
                const int64_t end = std::min(L, pos + chunk);
                lg = run(std::vector<int32_t>(all.begin() + pos, all.begin() + end), end == L ? 1 : 0, "prefill");
                pos = end;
            }
            const double prefill_s = now_s() - tp;
            decoding = true;
            std::fprintf(stderr, "{\"event\":\"phase\",\"len\":%lld,\"phase\":\"decode_start\",\"mono_ns\":%lld}\n", (long long)L, mono_ns());
            int32_t next = (int32_t)(std::max_element(lg.begin(), lg.end()) - lg.begin());
            for (int64_t g = 0; g < n_decode; ++g) {
                lg = run({next}, 1, "decode");
                next = (int32_t)(std::max_element(lg.begin(), lg.end()) - lg.begin());
            }
            std::fprintf(stderr, "{\"event\":\"phase\",\"len\":%lld,\"phase\":\"decode_end\",\"mono_ns\":%lld,\"decode\":%lld,\"pos\":%lld}\n",
                         (long long)L, mono_ns(), (long long)n_decode, (long long)base);
            std::fprintf(stderr, "qsa_overlap: len %lld prefill %.1f s, %zu prefill rows, %zu decode rows recorded\n",
                         (long long)L, prefill_s, rows["prefill"].size(), rows["decode"].size());

            for (const auto &[phase, rs] : rows)
                for (size_t ia = 0; ia + 1 < attn.size(); ++ia)
                    for (size_t ib = ia + 1; ib < attn.size(); ++ib) {
                        Stats st;
                        double base_sum = 0, rec_sum = 0, sink_sum = 0;
                        for (const Row &x : rs) {
                            const auto &sa = x.sel[ia], &sb = x.sel[ib];
                            if ((int64_t)sa.size() != k || (int64_t)sb.size() != k) continue;
                            std::vector<int32_t> both;
                            std::set_intersection(sa.begin(), sa.end(), sb.begin(), sb.end(), std::back_inserter(both));
                            st.recall.push_back((double)both.size() / (double)k);
                            st.jaccard.push_back((double)both.size() / (double)(2 * k - (int64_t)both.size()));
                            const int64_t blocks = (x.pos + 1) / kernels::kQsaBlock;
                            base_sum += (double)k / (double)blocks;
                            int64_t rec = 0, sink = 0;
                            for (int32_t b : sb) rec += b >= blocks - 16, sink += b < 4;
                            rec_sum += (double)rec / (double)k, sink_sum += (double)sink / (double)k;
                        }
                        const size_t n = st.recall.size();
                        if (n == 0) continue;
                        double mean = 0;
                        for (double v : st.recall) mean += v;
                        mean /= (double)n;
                        char line[512];
                        std::snprintf(line, sizeof line,
                                      "{\"len\":%lld,\"phase\":\"%s\",\"a\":%lld,\"b\":%lld,\"distance\":%zu,\"rows\":%zu,"
                                      "\"recall_mean\":%.4f,\"recall_p10\":%.4f,\"recall_p50\":%.4f,\"recall_p90\":%.4f,"
                                      "\"random_baseline\":%.4f,\"b_recent16_share\":%.4f,\"b_sink4_share\":%.4f,\"k\":%lld,"
                                      "\"kind\":\"cross_layer\",\"jaccard_mean\":%.4f,\"jaccard_p10\":%.4f,\"jaccard_p50\":%.4f}",
                                      (long long)L, phase.c_str(), (long long)attn[ia], (long long)attn[ib], ib - ia, n, mean,
                                      quantile(st.recall, 0.1), quantile(st.recall, 0.5), quantile(st.recall, 0.9),
                                      base_sum / (double)n, rec_sum / (double)n, sink_sum / (double)n, (long long)k,
                                      [&] { double m = 0; for (double v : st.jaccard) m += v; return m / (double)n; }(),
                                      quantile(st.jaccard, 0.1), quantile(st.jaccard, 0.5));
                        out << line << "\n";
                    }
            // V10-QSA (gate/qsa-index-stability.toml): token-to-token stability per indexer layer over
            // consecutive decode rows. Full set, and with the first 4 and last 16 blocks of each row removed.
            if (rows.count("decode"))
                for (size_t ia = 0; ia < attn.size(); ++ia) {
                    const auto &rs = rows.at("decode");
                    std::vector<double> jf, js;
                    double base_sum = 0;
                    for (size_t t = 1; t < rs.size(); ++t) {
                        const auto &s0 = rs[t - 1].sel[ia], &s1 = rs[t].sel[ia];
                        if ((int64_t)s0.size() != k || (int64_t)s1.size() != k) continue;
                        std::vector<int32_t> both;
                        std::set_intersection(s0.begin(), s0.end(), s1.begin(), s1.end(), std::back_inserter(both));
                        jf.push_back((double)both.size() / (double)(2 * k - (int64_t)both.size()));
                        const int64_t n0 = (rs[t - 1].pos + 1) / kernels::kQsaBlock, n1 = (rs[t].pos + 1) / kernels::kQsaBlock;
                        auto strip = [](const std::vector<int32_t> &v, int64_t n) {
                            std::vector<int32_t> o;
                            for (int32_t b : v) if (b >= 4 && b < n - 16) o.push_back(b);
                            return o;
                        };
                        const std::vector<int32_t> a = strip(s0, n0), b = strip(s1, n1);
                        std::vector<int32_t> ib, ub;
                        std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(ib));
                        std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(ub));
                        js.push_back(ub.empty() ? 1.0 : (double)ib.size() / (double)ub.size());
                        base_sum += (double)k / (double)(2 * n1 - k);
                    }
                    const size_t n = jf.size();
                    if (n == 0) continue;
                    auto mean = [](const std::vector<double> &v) { double m = 0; for (double x : v) m += x; return m / (double)v.size(); };
                    char line[512];
                    std::snprintf(line, sizeof line,
                                  "{\"len\":%lld,\"phase\":\"decode\",\"kind\":\"token_to_token\",\"layer\":%lld,\"rows\":%zu,"
                                  "\"jaccard_mean\":%.4f,\"jaccard_p10\":%.4f,\"jaccard_p50\":%.4f,\"jaccard_strip_mean\":%.4f,"
                                  "\"jaccard_strip_p10\":%.4f,\"random_baseline\":%.4f,\"k\":%lld}",
                                  (long long)L, (long long)attn[ia], n, mean(jf), quantile(jf, 0.1), quantile(jf, 0.5), mean(js),
                                  quantile(js, 0.1), base_sum / (double)n, (long long)k);
                    out << line << "\n";
                }
            out.flush();
        }
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "qsa_overlap: %s\n", e.what());
        return 1;
    }
}
