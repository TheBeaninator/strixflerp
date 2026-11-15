// strix_acts (STEER-1): mean activations of prompt sets at chosen layer points, for steering directions.
//
// One model + session load; each prompt of each set runs alone from an empty session, and the probe hook
// (Qwen4ExpProbe, the same one the golden suites read) hands every named point to this tool, which averages the
// selected token rows of each prompt and then the prompts of each set. A direction is a difference of two sets'
// means, taken outside this tool (Python), as are norms and projections.
//
//   strix_acts --weights W --ngram T --tokenizer J --jobs J --out DIR [--capacity N] [--chunk N] [--yarn F]
//              [--mtp on|off] [--mtp-vocab N] [--kv q8|bf16]   (defaults as the nodes serve: on, 65536, q8)
//
// jobs.json: {"points":["out","mlp_in"], "layers":[0,..] (default all), "skip":S, "last":K,
//             "sets":[{"name":"villainess","texts":["<|im_start|>system\n...", ...]}, ...]}
//   points  per-layer probe names: "out" (the H residual streams after the layer, H*d wide), "attn_in" / "mlp_in"
//           (the block's mixed-down input, d wide), "mixer" / "moe" (the block's output before it is mixed up)
//   skip    rows before this prompt position are left out (a shared system prefix); last: only the last K rows
//   texts   tokenized as given (chat markers included as text are special tokens, as the server renders them)
//   states  instead of (or beside) texts: [{"path":"x.kv","suffix":"<|im_start|>"}] - an at-rest context (a prompt
//           cache entry file, as /cache/export writes it: one entry, or a base then its delta) is imported into the
//           session and only the suffix is forwarded, so its rows read the residual streams at the end of that whole
//           context for the cost of a few tokens. The suffix defaults to "<|im_start|>", which is what follows every
//           Turn and Checkpoint entry (both end where a message is about to begin). skip/last count suffix rows.
// Writes DIR/<set>.<point>.f32: float32 [layers][cols] row-major, and DIR/meta.json (layers, cols, H, d, counts).
#include "common/bf16.hpp"
#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "runtime/qwen4exp.hpp"
#include "serve/json.hpp"
#include "serve/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace strix;

namespace {

std::string slurp(const std::string &p) {
    std::ifstream f(p, std::ios::binary);
    STRIX_CHECK(f.good(), "strix_acts: cannot read ", p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// One entry of a prompt cache file (serve/prompt_cache.cpp layout): its token count, delta base and state bytes.
struct Entry {
    uint64_t n = 0, base_n = 0;
    const uint8_t *state = nullptr;
    size_t bytes = 0;
};

std::vector<Entry> parse_entries(const std::string &body, const std::string &path) {
    static const char kMagic[8] = {'S', 'T', 'R', 'X', 'P', 'C', 'C', 'H'};
    std::vector<Entry> out;
    const uint8_t *b = reinterpret_cast<const uint8_t *>(body.data());
    size_t off = 0;
    while (off < body.size()) {
        STRIX_CHECK(body.size() - off >= 4096 && std::memcmp(b + off, kMagic, 8) == 0, path, ": not a prompt cache entry at ", off);
        auto rd = [&](size_t o, auto v) { std::memcpy(&v, b + off + o, sizeof(v)); return v; };
        const uint32_t version = rd(8, uint32_t{});
        Entry e;
        e.n = rd(16, uint64_t{});
        const uint64_t state_bytes = rd(24, uint64_t{}), state_at = rd(32, uint64_t{});
        e.base_n = version >= 2 ? rd(248, uint64_t{}) : 0;
        STRIX_CHECK(off + state_at + state_bytes <= body.size(), path, ": entry at ", off, " runs past the file");
        e.state = b + off + state_at, e.bytes = (size_t)state_bytes;
        out.push_back(e);
        off += (size_t)(state_at + state_bytes);
    }
    STRIX_CHECK(!out.empty(), path, ": empty");
    return out;
}

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

}  // namespace

int main(int argc, char **argv) {
    try {
        std::map<std::string, std::string> a;
        for (int i = 1; i < argc; ++i) {
            std::string k = argv[i];
            STRIX_CHECK(k.rfind("--", 0) == 0 && i + 1 < argc, "strix_acts: bad argument ", k);
            a[k.substr(2)] = argv[++i];
        }
        auto get = [&](const std::string &k, const std::string &d = "") { return a.count(k) ? a[k] : d; };
        const std::string weights = get("weights"), ngram = get("ngram"), tokj = get("tokenizer"), jobs_path = get("jobs"),
                          out_dir = get("out");
        STRIX_CHECK(!weights.empty() && !ngram.empty() && !tokj.empty() && !jobs_path.empty() && !out_dir.empty(),
                    "usage: strix_acts --weights W --ngram T --tokenizer J --jobs J --out DIR [--capacity N] [--chunk N] [--yarn F]");
        const int64_t capacity = std::stoll(get("capacity", "32768")), chunk = std::stoll(get("chunk", "8192"));
        const float yarn = std::stof(get("yarn", "1.0"));

        const json::Value jobs = json::Value::parse(slurp(jobs_path));
        std::vector<std::string> points;
        for (const json::Value &p : jobs.find("points")->as_array("points")) points.push_back(p.as_string("point"));
        const int64_t skip = jobs.find("skip") ? jobs.find("skip")->as_int("skip") : 0;
        const int64_t last = jobs.find("last") ? jobs.find("last")->as_int("last") : 0;

        const Tokenizer tok(tokj);
        const double t0 = now_s();
        Qwen4ExpModel model(weights, ngram, kernels::Act::BF16, false, 262144, yarn);
        // the draft head on, as the nodes serve: an at-rest state carries its rows, and its size must match
        const bool mtp = get("mtp", "on") != "off" && model.has_mtp();
        Qwen4ExpSession ses(model, capacity, chunk, PrefillMath::WmmaBf16, mtp, parse_kv_format(get("kv", "q8")));
        if (mtp) ses.set_mtp_vocab(std::stoll(get("mtp-vocab", "65536")));
        const auto &D = model.dims();
        std::vector<int64_t> layers;
        if (const json::Value *lv = jobs.find("layers"))
            for (const json::Value &l : lv->as_array("layers")) layers.push_back(l.as_int("layer"));
        else
            for (int64_t i = 0; i < D.layers; ++i) layers.push_back(i);
        std::map<std::string, size_t> want;  // "L{i}.{point}" -> slot
        for (size_t li = 0; li < layers.size(); ++li)
            for (size_t pi = 0; pi < points.size(); ++pi)
                want["L" + std::to_string(layers[li]) + "." + points[pi]] = li * points.size() + pi;
        std::fprintf(stderr, "strix_acts: loaded in %.1f s; %lld layers, H %lld, d %lld; %zu probe slots\n", now_s() - t0,
                     (long long)D.layers, (long long)D.H, (long long)D.d, want.size());

        std::vector<int64_t> cols(want.size(), 0);
        json::Value meta = json::Value::object();
        json::Value sets_meta = json::Value::object();
        for (const json::Value &set : jobs.find("sets")->as_array("sets")) {
            const std::string name = set.find("name")->as_string("set.name");
            std::vector<std::vector<double>> sum(want.size());  // per slot: sum over prompts of the prompt's mean row
            int64_t n_prompts = 0, n_rows = 0;
            struct Item { std::string text, state; };
            std::vector<Item> items;
            if (const json::Value *tv = set.find("texts"))
                for (const json::Value &t : tv->as_array("texts")) items.push_back({t.as_string("text"), ""});
            if (const json::Value *sv = set.find("states"))
                for (const json::Value &st : sv->as_array("states"))
                    items.push_back({st.find("suffix") ? st.find("suffix")->as_string("suffix") : "<|im_start|>",
                                     st.find("path")->as_string("state.path")});
            for (const Item &item : items) {
                const std::vector<int32_t> ids = tok.encode(item.text);
                const int64_t P = (int64_t)ids.size();
                STRIX_CHECK(P > 0, "strix_acts: an empty prompt or suffix");
                const int64_t lo = std::max(skip, last > 0 ? P - last : (int64_t)0);
                STRIX_CHECK(lo < P, "strix_acts: set ", name, ": a ", P, "-token prompt has no rows after skip/last");
                std::vector<std::vector<double>> acc(want.size());
                int64_t base = 0;
                std::vector<float> host;
                const Qwen4ExpProbe probe = [&](const std::string &pname, const void *dev, int64_t rows, int64_t c, ProbeType type) {
                    auto it = want.find(pname);
                    if (it == want.end()) return;
                    STRIX_CHECK(type == ProbeType::BF16 || type == ProbeType::F32, "strix_acts: ", pname, " is not a float point");
                    const size_t s = it->second;
                    if (cols[s] == 0) cols[s] = c;
                    STRIX_CHECK(cols[s] == c, "strix_acts: ", pname, " changed width");
                    if (acc[s].empty()) acc[s].assign((size_t)c, 0.0);
                    const size_t esz = type == ProbeType::BF16 ? 2 : 4;
                    std::vector<uint8_t> raw((size_t)(rows * c) * esz);
                    STRIX_HIP_CHECK(hipMemcpy(raw.data(), dev, raw.size(), hipMemcpyDeviceToHost), "probe copy ", pname);
                    for (int64_t r = 0; r < rows; ++r) {
                        const int64_t pos = base + r;
                        if (pos < lo) continue;
                        for (int64_t k = 0; k < c; ++k) {
                            const size_t i = (size_t)(r * c + k);
                            float v;
                            if (esz == 2) { uint16_t b; std::memcpy(&b, raw.data() + i * 2, 2); v = bf16_bits_to_f32(b); }
                            else std::memcpy(&v, raw.data() + i * 4, 4);
                            acc[s][(size_t)k] += v;
                        }
                    }
                };
                ses.reset();
                if (!item.state.empty()) {
                    // the at-rest context first: a whole entry, or a base then its delta, exactly as the cache imports
                    const std::string body = slurp(item.state);
                    for (const Entry &en : parse_entries(body, item.state)) {
                        STRIX_CHECK(ses.state_bytes((int64_t)en.n, (int64_t)en.base_n) == en.bytes, item.state, ": ", en.bytes,
                                    " state bytes, this model expects ", ses.state_bytes((int64_t)en.n, (int64_t)en.base_n),
                                    " (another model or format)");
                        std::vector<uint8_t> buf(en.state, en.state + en.bytes);
                        ses.import_state(buf.data(), buf.size(), (int64_t)en.n, (int64_t)en.base_n);
                    }
                    STRIX_CHECK(ses.pos() + P <= ses.capacity(), item.state, ": the context and suffix pass the capacity");
                }
                for (int64_t o = 0; o < P; o += chunk) {
                    const int64_t e = std::min(P, o + chunk);
                    base = o;
                    ses.forward(std::vector<int32_t>(ids.begin() + o, ids.begin() + e), 1, probe);
                }
                const double n = (double)(P - lo);
                for (size_t s = 0; s < want.size(); ++s) {
                    STRIX_CHECK(!acc[s].empty(), "strix_acts: the forward never reached a requested probe point");
                    if (sum[s].empty()) sum[s].assign(acc[s].size(), 0.0);
                    for (size_t k = 0; k < acc[s].size(); ++k) sum[s][k] += acc[s][k] / n;
                }
                ++n_prompts, n_rows += P - lo;
            }
            STRIX_CHECK(n_prompts > 0, "strix_acts: set ", name, " has no texts or states");
            for (size_t pi = 0; pi < points.size(); ++pi) {
                std::vector<float> out;
                for (size_t li = 0; li < layers.size(); ++li)
                    for (double v : sum[li * points.size() + pi]) out.push_back((float)(v / n_prompts));
                const std::string path = out_dir + "/" + name + "." + points[pi] + ".f32";
                std::ofstream f(path, std::ios::binary | std::ios::trunc);
                STRIX_CHECK(f.good(), "strix_acts: cannot write ", path);
                f.write(reinterpret_cast<const char *>(out.data()), (std::streamsize)(out.size() * 4));
            }
            json::Value sm = json::Value::object();
            sm.set("prompts", json::Value::integer(n_prompts));
            sm.set("rows", json::Value::integer(n_rows));
            sets_meta.set(name, std::move(sm));
            std::fprintf(stderr, "strix_acts: %s: %lld prompts, %lld rows\n", name.c_str(), (long long)n_prompts, (long long)n_rows);
        }
        json::Value lv = json::Value::array(), cv = json::Value::object(), pv = json::Value::array();
        for (int64_t l : layers) lv.push(json::Value::integer(l));
        for (size_t pi = 0; pi < points.size(); ++pi) {
            pv.push(json::Value::string(points[pi]));
            cv.set(points[pi], json::Value::integer(cols[pi]));
        }
        meta.set("layers", std::move(lv));
        meta.set("points", std::move(pv));
        meta.set("cols", std::move(cv));
        meta.set("H", json::Value::integer(D.H));
        meta.set("d", json::Value::integer(D.d));
        meta.set("skip", json::Value::integer(skip));
        meta.set("last", json::Value::integer(last));
        meta.set("sets", std::move(sets_meta));
        std::ofstream mf(out_dir + "/meta.json", std::ios::trunc);
        mf << meta.dump() << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "strix_acts: %s\n", e.what());
        return 1;
    }
}
