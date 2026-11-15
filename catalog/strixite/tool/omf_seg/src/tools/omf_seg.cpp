// omf_seg (OMF-R3, card 3fa039212b): batch save/restore/probe of Qwen4ExpSession states.
//
// One model + session load; a jobs file (JSON) drives a sequence of steps per job, so a whole experiment
// (export segments, import transformed states, probe read-backs, time everything) runs in one model load.
// State files are the prompt cache's own bytes (serve/prompt_cache.hpp layout, Qwen4ExpSession::export_state);
// transforms (quantize / prune / drop / low-rank / RoPE rebase) happen outside this tool, in Python, on those
// bytes - the tool imports whatever it is given (import_state checks the size against state_bytes(n, from);
// everything else about the bytes is the caller's responsibility - that is the whole point of this tool).
//
//   omf_seg --weights W --ngram T --tokenizer J --jobs jobs.json --out results.jsonl [--capacity N] [--chunk N]
//           [--yarn 1.0] [--mtp on|off] [--ngram-cache-rows N]
//
// jobs.json: {"jobs":[{"id":"..","steps":[{...},...]}, ...]} - step ops:
//   {"op":"reset"}                                   session to empty
//   {"op":"forward","tokens":[..]|"tokens_file":"f.bin"[,"chunk":N]}   prefill (chunked), timed
//   {"op":"export","path":"f.bin"[,"from":K]}        snapshot at pos(), export; writes path+".meta" {n,from,bytes,mtp}
//   {"op":"import","path":"f.bin"[,"n":N,"from":K]}  load a state (n/from default from its .meta; from>0 needs the
//                                                    matching base imported just before, like the prompt cache does)
//   {"op":"probe","tokens":..|"tokens_file":..[,"expect":..|"expect_file":..][,"max_new":M][,"keep":"after"]}
//        forward(prompt) (timed), then teacher-forced NLL of expect (chunked, positions continue), then greedy
//        decode of max_new tokens from where we are; reports gen ids/text, nats/token, timings. Without "keep",
//        restores the state to the step's start (probe is read-only for the imported segment).
#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "formats/strixw.hpp"
#include "kernels/linear_q8.hpp"
#include "runtime/qwen4exp.hpp"
#include "serve/json.hpp"
#include "serve/tokenizer.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace strix;

namespace {

double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Args {
    std::map<std::string, std::string> kv;
    std::string get(const std::string &k, const std::string &d = "") const {
        auto it = kv.find(k);
        return it == kv.end() ? d : it->second;
    }
    int64_t num(const std::string &k, int64_t d) const { return kv.count(k) ? std::stoll(kv.at(k)) : d; }
    double real(const std::string &k, double d) const { return kv.count(k) ? std::stod(kv.at(k)) : d; }
};

std::string slurp(const std::string &p) {
    std::ifstream f(p, std::ios::binary);
    STRIX_CHECK(f.good(), "omf_seg: cannot read ", p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void spit(const std::string &p, const void *data, size_t n) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    STRIX_CHECK(f.good(), "omf_seg: cannot write ", p);
    f.write(static_cast<const char *>(data), (std::streamsize)n);
    STRIX_CHECK(f.good(), "omf_seg: write failed for ", p);
}

// tokens / tokens_file / expect / expect_file -> ids
std::vector<int32_t> tokens_of(const json::Value &v, const char *k_plain, const char *k_file, const std::string &path,
                               const Tokenizer &tok, const char *k_text) {
    if (const json::Value *t = v.find(k_text)) return tok.encode(t->as_string(path));
    if (const json::Value *t = v.find(k_plain)) {
        const auto &a = t->as_array(path);
        std::vector<int32_t> ids;
        ids.reserve(a.size());
        for (size_t i = 0; i < a.size(); ++i) ids.push_back((int32_t)a[i].as_int(path));
        return ids;
    }
    if (const json::Value *f = v.find(k_file)) {
        const std::string raw = slurp(f->as_string(path));
        STRIX_CHECK(raw.size() % 4 == 0, path, ": file size ", raw.size(), " is not a multiple of 4");
        std::vector<int32_t> ids(raw.size() / 4);
        std::memcpy(ids.data(), raw.data(), raw.size());
        return ids;
    }
    return {};
}

int64_t argmax(const float *x, int64_t n) {
    int64_t b = 0;
    for (int64_t i = 1; i < n; ++i)
        if (x[i] > x[b]) b = i;
    return b;
}

double lse(const float *x, int64_t n) {
    double m = -INFINITY;
    for (int64_t i = 0; i < n; ++i) m = std::max(m, (double)x[i]);
    double s = 0;
    for (int64_t i = 0; i < n; ++i) s += std::exp((double)x[i] - m);
    return m + std::log(s);
}

void push_ints(json::Value &arr, const std::vector<int32_t> &ids) {
    for (int32_t t : ids) arr.push(json::Value::integer(t));
}

}  // namespace

int main(int argc, char **argv) {
    try {
        Args a;
        for (int i = 1; i < argc; ++i) {
            std::string k = argv[i];
            STRIX_CHECK(k.rfind("--", 0) == 0 && i + 1 < argc, "omf_seg: bad argument ", k);
            a.kv[k.substr(2)] = argv[++i];
        }
        const std::string weights = a.get("weights"), ngram = a.get("ngram"), tokj = a.get("tokenizer");
        const std::string jobs_path = a.get("jobs"), out_path = a.get("out", "/dev/stdout");
        STRIX_CHECK(!weights.empty() && !ngram.empty() && !tokj.empty() && !jobs_path.empty(),
                    "usage: omf_seg --weights W --ngram T --tokenizer J --jobs J [--out F] [--capacity N] [--chunk N] [--yarn F] [--mtp on|off] [--kv q8|bf16]");
        const int64_t capacity = a.num("capacity", 32768), chunk = a.num("chunk", 8192);
        const float yarn = (float)a.real("yarn", 1.0);
        const bool mtp_want = a.get("mtp", "on") != "off";
        const int64_t cache_rows = a.num("ngram-cache-rows", 262144);
        const int64_t mtp_vocab = a.num("mtp-vocab", 65536);
        const int64_t eos = a.num("eos", 248044);

        const Tokenizer tok(tokj);
        const json::Value jobs_doc = json::Value::parse(slurp(jobs_path));
        const json::Value *jobs_v = jobs_doc.find("jobs");
        STRIX_CHECK(jobs_v, "jobs file: no 'jobs'");
        const std::vector<json::Value> &jobs = jobs_v->as_array("jobs");

        const double t_load0 = now_ms();
        // v0.4.0 port: the model ctor drops the fork's TpConfig (single-rank, device-resident weights are the
        // upstream default WeightPlacement); mtp_vocab still goes to the session, as upstream's server does it.
        Qwen4ExpModel model(weights, ngram, kernels::Act::BF16, false, cache_rows, yarn);
        // K/V cache format of the states: q8 (v0.4.0's served default) or bf16 (the v0.3 entries). --kv selects.
        const KvFormat kv = parse_kv_format(a.get("kv", "q8"));
        const bool mtp = mtp_want && model.has_mtp();
        Qwen4ExpSession ses(model, capacity, chunk, PrefillMath::WmmaBf16, mtp, kv);
        if (model.has_mtp()) ses.set_mtp_vocab(mtp_vocab);
        const int64_t V = model.dims().vocab;
        std::fprintf(stderr, "omf_seg: loaded in %.1f s, mtp=%s kv=%s, L%lld hkv%lld hd%lld idx_d%lld vocab %lld cap %lld\n",
                     (now_ms() - t_load0) / 1000, mtp ? "on" : "off", kv_format_name(kv), (long long)model.dims().layers,
                     (long long)model.dims().hkv, (long long)model.dims().hd, (long long)model.dims().idx_d,
                     (long long)V, (long long)ses.capacity());

        Qwen4ExpSnapshot snap = ses.make_snapshot();
        std::vector<uint8_t> buf;
        int64_t last_full_n = -1;
        std::ofstream out(out_path, std::ios::app);
        STRIX_CHECK(out.good(), "omf_seg: cannot write ", out_path);

        for (size_t j = 0; j < jobs.size(); ++j) {
            const json::Value &job = jobs[j];
            std::string id = "job" + std::to_string(j);
            if (const json::Value *iv = job.find("id")) id = iv->as_string("job.id");
            json::Value res = json::Value::object();
            res.set("id", json::Value::string(id));
            json::Value steps_out = json::Value::array();
            std::string err;
            try {
                STRIX_CHECK(job.find("steps"), "job ", id, ": no steps");
                const std::vector<json::Value> &steps = job.find("steps")->as_array("steps");
                for (size_t k = 0; k < steps.size(); ++k) {
                    const json::Value &st = steps[k];
                    const std::string op = st.find("op") ? st.find("op")->as_string("step.op") : "";
                    const std::string path = "job " + id + " step " + std::to_string(k) + " (" + op + ")";
                    json::Value so = json::Value::object();
                    so.set("op", json::Value::string(op));
                    const double t0 = now_ms();
                    const int64_t pos_before = ses.pos();

                    if (op == "reset") {
                        ses.reset();
                    } else if (op == "encode") {
                        const std::string f = st.find("path") ? st.find("path")->as_string(path) : "";
                        const std::vector<int32_t> ids = tok.encode(st.find("text")->as_string(path));
                        spit(f, ids.data(), ids.size() * 4);
                        so.set("tokens", json::Value::integer((int64_t)ids.size()));
                    } else if (op == "forward") {
                        const std::vector<int32_t> ids = tokens_of(st, "tokens", "tokens_file", path, tok, "text");
                        const int64_t ch = st.find("chunk") ? st.find("chunk")->as_int(path) : chunk;
                        STRIX_CHECK(pos_before + (int64_t)ids.size() + 17 <= ses.capacity(),
                                    path, ": pos ", pos_before, " + ", ids.size(), " > capacity ", ses.capacity());
                        for (size_t o = 0; o < ids.size(); o += (size_t)ch) {
                            const size_t e = std::min(ids.size(), o + (size_t)std::max<int64_t>(1, ch));
                            ses.forward(std::vector<int32_t>(ids.begin() + (int64_t)o, ids.begin() + (int64_t)e), 1);
                        }
                        so.set("tokens", json::Value::integer((int64_t)ids.size()));
                    } else if (op == "export") {
                        const std::string f = st.find("path") ? st.find("path")->as_string(path) : "";
                        int64_t from = st.find("from") ? st.find("from")->as_int(path) : 0;
                        if (from < 0)
                            from = last_full_n;  // -1: the n of the last whole export in this run (delta chain)
                        STRIX_CHECK(from >= 0, path, ": no previous whole export to base the delta on");
                        const int64_t n = ses.pos();
                        STRIX_CHECK(n >= 1, path, ": nothing to export (pos ", n, ")");
                        ses.save(snap);
                        const size_t bytes = ses.state_bytes(n, from);
                        buf.resize(bytes);
                        ses.export_state(snap, buf.data(), bytes, from);
                        spit(f, buf.data(), bytes);
                        json::Value meta = json::Value::object();
                        meta.set("n", json::Value::integer(n));
                        meta.set("from", json::Value::integer(from));
                        meta.set("bytes", json::Value::integer((int64_t)bytes));
                        meta.set("mtp", json::Value::boolean(mtp));
                        const std::string md = meta.dump();
                        spit(f + ".meta", md.data(), md.size());
                        if (from == 0) last_full_n = n;
                        so.set("n", json::Value::integer(n));
                        so.set("bytes", json::Value::integer((int64_t)bytes));
                    } else if (op == "import") {
                        const std::string f = st.find("path") ? st.find("path")->as_string(path) : "";
                        int64_t n = 0, from = 0;
                        if (st.find("n")) n = st.find("n")->as_int(path);
                        if (st.find("from")) from = st.find("from")->as_int(path);
                        if (!st.find("n")) {
                            const json::Value meta = json::Value::parse(slurp(f + ".meta"));
                            n = meta.find("n")->as_int("meta.n");
                            if (const json::Value *fv = meta.find("from")) from = fv->as_int("meta.from");
                        }
                        const std::string raw = slurp(f);
                        STRIX_CHECK(ses.state_bytes(n, from) == raw.size(),
                                    path, ": file ", raw.size(), " bytes, state_bytes(", n, ",", from, ") = ",
                                    ses.state_bytes(n, from));
                        if (from == 0) ses.reset();
                        else STRIX_CHECK(ses.pos() == from, path, ": a delta needs its base imported just before (pos ",
                                         ses.pos(), ", from ", from, ")");
                        buf.assign(raw.begin(), raw.end());
                        ses.import_state(buf.data(), buf.size(), n, from);
                        so.set("n", json::Value::integer(n));
                        so.set("from", json::Value::integer(from));
                    } else if (op == "probe") {
                        const std::vector<int32_t> prompt = tokens_of(st, "tokens", "tokens_file", path, tok, "text");
                        const std::vector<int32_t> expect = tokens_of(st, "expect", "expect_file", path, tok, "expect_text");
                        const int64_t max_new = st.find("max_new") ? st.find("max_new")->as_int(path) : 24;
                        const bool keep_after = st.find("keep") != nullptr;
                        const int64_t need = (int64_t)prompt.size() + (int64_t)expect.size() + max_new + 8;
                        STRIX_CHECK(pos_before + need <= ses.capacity(), path, ": pos ", pos_before, " + ", need,
                                    " > capacity ", ses.capacity());
                        // Make the probe read-only for the imported state unless "keep".
                        ses.save(snap);
                        const bool can_rewind = true;  // fresh save at pos_before; forwards don't rewind
                        // 1) prompt forward (timed: TTFT for a restored state)
                        double fwd_ms = 0;
                        std::vector<float> logits;
                        {
                            const double f0 = now_ms();
                            for (size_t o = 0; o < prompt.size(); o += (size_t)chunk) {
                                const size_t e = std::min(prompt.size(), o + (size_t)chunk);
                                logits = ses.forward(std::vector<int32_t>(prompt.begin() + (int64_t)o,
                                                                          prompt.begin() + (int64_t)e), 1);
                            }
                            fwd_ms = now_ms() - f0;
                        }
                        so.set("prompt_ms", json::Value::number(fwd_ms));
                        so.set("prompt_tokens", json::Value::integer((int64_t)prompt.size()));
                        // 2) teacher-forced NLL of expect: chunked forwards, positions continue; row r of a
                        //    chunk covering expect[s..e) scores expect[s+r+1] (expect[0] scored by the prompt's logits).
                        if (!expect.empty()) {
                            STRIX_CHECK(!logits.empty(), path, ": no prompt logits to score expect[0]");
                            double nll = 0;
                            int64_t top1 = 0;
                            // expect[0] is scored by the prompt's last logits (the prediction at position P-1)
                            {
                                const double l = lse(logits.data(), V);
                                nll += l - logits[expect[0]];
                                if (argmax(logits.data(), V) == expect[0]) ++top1;
                            }
                            // chunks of 16 expect tokens; row r (0-based) of the chunk [s, s+len) is the
                            // prediction after expect[s+r], scoring expect[s+r+1]
                            for (size_t s = 0; s < expect.size();) {
                                const size_t e = std::min(expect.size(), s + 16);
                                const std::vector<int32_t> part(expect.begin() + (int64_t)s, expect.begin() + (int64_t)e);
                                logits = ses.forward(part, (int64_t)part.size());
                                for (size_t r = 0; r < part.size(); ++r) {
                                    const size_t target = s + r + 1;
                                    if (target >= expect.size()) break;
                                    const float *row = logits.data() + r * (size_t)V;
                                    const double l = lse(row, V);
                                    nll += l - row[expect[target]];
                                    if (argmax(row, V) == expect[target]) ++top1;
                                }
                                s = e;
                            }
                            so.set("nats_per_token", json::Value::number(nll / (double)expect.size()));
                            so.set("top1_match", json::Value::integer(top1));
                            so.set("expect_n", json::Value::integer((int64_t)expect.size()));
                        }
                        // 3) greedy continuation from where we are (logits = prediction at pos())
                        // Generation starts from the prediction after everything fed so far: keep only the last row.
                        if (!expect.empty() && !logits.empty())
                            logits = std::vector<float>(logits.end() - (long)V, logits.end());
                        std::vector<int32_t> gen;
                        if (max_new > 0) {
                            STRIX_CHECK(!logits.empty(), path, ": no logits to start generation");
                            const double g0 = now_ms();
                            int32_t nx = (int32_t)argmax(logits.data(), V);
                            for (int64_t g = 0; g < max_new; ++g) {
                                gen.push_back(nx);
                                if ((int64_t)nx == eos) break;
                                logits = ses.forward({nx}, 1);
                                nx = (int32_t)argmax(logits.data(), V);
                            }
                            so.set("gen_ms", json::Value::number(now_ms() - g0));
                            so.set("gen_n", json::Value::integer((int64_t)gen.size()));
                        }
                        json::Value gen_arr = json::Value::array();
                        push_ints(gen_arr, gen);
                        so.set("gen", gen_arr);
                        so.set("gen_text", json::Value::string(tok.decode(gen)));
                        if (!keep_after && can_rewind && ses.pos() > pos_before && pos_before >= 1) ses.restore(snap);
                        so.set("pos_after", json::Value::integer(ses.pos()));
                    } else {
                        STRIX_FAIL(path, ": unknown op");
                    }
                    so.set("ms", json::Value::number(now_ms() - t0));
                    so.set("pos_end", json::Value::integer(ses.pos()));
                    steps_out.push(so);
                }
            } catch (const std::exception &e) {
                err = e.what();
                try { ses.reset(); } catch (...) {}
            }
            res.set("steps", steps_out);
            if (!err.empty()) res.set("error", json::Value::string(err));
            out << res.dump() << "\n";
            out.flush();
            std::fprintf(stderr, "omf_seg: job %zu/%zu '%s' %s\n", j + 1, jobs.size(), id.c_str(),
                         err.empty() ? "ok" : err.c_str());
        }
        std::fprintf(stderr, "omf_seg: done\n");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "omf_seg FATAL: %s\n", e.what());
        return 1;
    }
}
