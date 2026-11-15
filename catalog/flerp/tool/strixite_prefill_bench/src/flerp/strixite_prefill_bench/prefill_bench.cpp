// strixflerp-prefill-bench: chunked prefill of fixed token ids through strixite's Qwen4ExpSession, the way the
// server's engine runs a prompt (serve/engine.cpp Engine::run: chunks of max_chunk rows, set_lookahead with the
// next chunk's ids before each forward, the last forward asking for logits), then greedy decode.
//
// One process loads the weights once; each variant builds its own session (max_tokens = its chunk), runs every
// prompt, and is destroyed before the next, so variants never share device memory. Variants alternate per round
// (round r runs them in order when r is even, reversed when odd) to spread drift across both.
//
//   strixflerp-prefill-bench --weights W --ngram T --out DIR --prompts 8192,32768 --variants NAME:chunk=8192[:ENV=V..],..
//       [--capacity N] [--rounds R] [--gen G] [--seed S] [--gdn-precision half-u] [--attention-precision exact]
//       [--kv q8] [--ngram-cache-rows N] [--logits-rows K] [--save-logits 0|1]
//
// Prompt ids: splitmix64(seed ^ length) mapped into [256, 151000), fixed for a (seed, length).
// Per prompt and variant it writes one JSON line to DIR/results.jsonl: chunk wall times (host, from the forward call
// to its return; the forward ends with a stream sync, so this is the chunk's full span), prefill tok/s, PLE stats
// deltas, sha256 of the last K logits rows (float32 bytes), their argmax, the greedy ids and decode tok/s (T = 1
// forwards, MTP off), device memory after the session was built and after the prompt. With --save-logits 1 the
// first round's last-K logits rows go to DIR/logits.<variant>.<len>.f32 (for KL / top-1 offline).
// Environment entries of a variant are set (setenv) before its session is built and unset afterwards: an adapted
// module reads its switch when the session is built.
#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "runtime/qwen4exp.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace strix;

namespace {

// --- sha256 (FIPS 180-4), small and self-contained -----------------------------------------------------------------
struct Sha256 {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint8_t buf[64];
    size_t n = 0;
    uint64_t total = 0;
    static uint32_t rotr(uint32_t x, int r) { return (x >> r) | (x << (32 - r)); }
    void block(const uint8_t *p) {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
            0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
            0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
            0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
            0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
            0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
        }
        h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += hh;
    }
    void add(const void *data, size_t len) {
        const uint8_t *p = static_cast<const uint8_t *>(data);
        total += len;
        while (len > 0) {
            const size_t take = std::min(len, 64 - n);
            std::memcpy(buf + n, p, take);
            n += take, p += take, len -= take;
            if (n == 64) block(buf), n = 0;
        }
    }
    std::string hex() {
        const uint64_t bits = total * 8;
        const uint8_t one = 0x80, zero = 0;
        add(&one, 1);
        while (n != 56) add(&zero, 1);
        uint8_t len[8];
        for (int i = 0; i < 8; ++i) len[i] = (uint8_t)(bits >> (56 - 8 * i));
        add(len, 8);
        char out[65];
        for (int i = 0; i < 8; ++i) std::snprintf(out + 8 * i, 9, "%08x", h[i]);
        return std::string(out, 64);
    }
};

uint64_t splitmix64(uint64_t &s) {
    uint64_t z = (s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

std::vector<int32_t> prompt_ids(uint64_t seed, int64_t len) {
    uint64_t s = seed ^ (uint64_t)len * 0x100000001b3ull;
    std::vector<int32_t> ids((size_t)len);
    for (auto &v : ids) v = 256 + (int32_t)(splitmix64(s) % (151000 - 256));
    return ids;
}

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
// CLOCK_MONOTONIC ns (steady_clock on Linux): the clock rocprofv3's kernel records use, so chunk spans line up with them.
int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// read_bytes of /proc/self/io: bytes this process caused to be read from storage (page-cache hits excluded).
int64_t io_read_bytes() {
    std::ifstream f("/proc/self/io");
    for (std::string k; f >> k;) {
        int64_t v = 0;
        f >> v;
        if (k == "read_bytes:") return v;
    }
    return -1;
}

std::vector<std::string> split(const std::string &s, char c) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    for (std::string part; std::getline(ss, part, c);)
        if (!part.empty()) out.push_back(part);
    return out;
}

struct Variant {
    std::string name;
    int64_t chunk = 8192;
    std::vector<std::pair<std::string, std::string>> env;
};

size_t dev_used() {
    size_t f = 0, t = 0;
    STRIX_HIP_CHECK(hipMemGetInfo(&f, &t), "hipMemGetInfo");
    return t - f;
}

std::string jnum(double v) {
    char b[64];
    std::snprintf(b, sizeof b, "%.6f", v);
    return b;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        std::map<std::string, std::string> a;
        for (int i = 1; i < argc; ++i) {
            std::string k = argv[i];
            STRIX_CHECK(k.rfind("--", 0) == 0 && i + 1 < argc, "prefill_bench: bad argument ", k);
            a[k.substr(2)] = argv[++i];
        }
        auto get = [&](const std::string &k, const std::string &d = "") { return a.count(k) ? a[k] : d; };
        const std::string out_dir = get("out");
        STRIX_CHECK(!get("weights").empty() && !get("ngram").empty() && !out_dir.empty() && !get("variants").empty() &&
                        !get("prompts").empty(),
                    "usage: strixflerp-prefill-bench --weights W --ngram T --out DIR --prompts L,.. --variants NAME:chunk=N[:K=V..],..");
        const int64_t capacity = std::stoll(get("capacity", "139264"));
        const int rounds = std::stoi(get("rounds", "3"));
        const int64_t gen = std::stoll(get("gen", "32"));
        const uint64_t seed = std::stoull(get("seed", "1234"));
        const int64_t kl = std::stoll(get("logits-rows", "16"));
        const bool save_logits = get("save-logits", "1") == "1";
        std::vector<int64_t> lens;
        for (const std::string &p : split(get("prompts"), ',')) lens.push_back(std::stoll(p));
        std::vector<Variant> variants;
        for (const std::string &v : split(get("variants"), ',')) {
            const auto parts = split(v, ':');
            Variant x;
            x.name = parts.at(0);
            for (size_t i = 1; i < parts.size(); ++i) {
                const size_t eq = parts[i].find('=');
                STRIX_CHECK(eq != std::string::npos, "prefill_bench: variant part '", parts[i], "' has no '='");
                const std::string k = parts[i].substr(0, eq), val = parts[i].substr(eq + 1);
                if (k == "chunk") x.chunk = std::stoll(val);
                else x.env.emplace_back(k, val);
            }
            variants.push_back(x);
        }
        PrecisionOptions precision;
        precision.gdn = parse_gdn_precision(get("gdn-precision", "half-u"));
        precision.attention = parse_attention_precision(get("attention-precision", "exact"));

        const double t0 = now_s();
        Qwen4ExpModel model(get("weights"), get("ngram"), kernels::Act::BF16, false,
                            std::stoll(get("ngram-cache-rows", "8388608")), 1.0f);
        const auto &D = model.dims();
        std::fprintf(stderr, "prefill_bench: weights loaded in %.1f s (%.1f GiB), device used %.2f GiB, %.2f GB read from storage\n",
                     now_s() - t0, (double)model.weights().data_bytes() / (1ull << 30), (double)dev_used() / (1ull << 30),
                     (double)io_read_bytes() * 1e-9);
        std::ofstream out(out_dir + "/results.jsonl", std::ios::app);
        STRIX_CHECK(out.good(), "prefill_bench: cannot write ", out_dir, "/results.jsonl");

        for (int r = 0; r < rounds; ++r) {
            std::vector<Variant> order = variants;
            if (r % 2 == 1) std::reverse(order.begin(), order.end());
            for (const Variant &v : order) {
                for (const auto &kv : v.env) setenv(kv.first.c_str(), kv.second.c_str(), 1);
                {
                    const size_t used0 = dev_used();
                    auto ses = std::make_unique<Qwen4ExpSession>(model, capacity, v.chunk, PrefillMath::WmmaBf16,
                                                                 false, parse_kv_format(get("kv", "q8")));
                    ses->set_precision(precision);
                    const size_t used_ses = dev_used();
                    for (int64_t L : lens) {
                        const std::vector<int32_t> ids = prompt_ids(seed, L);
                        ses->reset();
                        const auto p0 = ses->ple_stats();
                        std::vector<double> chunk_s;
                        std::vector<int64_t> chunk_ns;  // start, end of each chunk's forward call
                        std::vector<float> logits;
                        const double tp = now_s();
                        for (int64_t pos = 0; pos < L;) {
                            const int64_t end = std::min(L, pos + v.chunk);
                            if (end < L)
                                ses->set_lookahead(std::vector<int32_t>(ids.begin() + end,
                                                                        ids.begin() + std::min(L, end + v.chunk)));
                            const int64_t want = end == L ? std::min(kl, end - pos) : 0;
                            const double tc = now_s();
                            chunk_ns.push_back(now_ns());
                            logits = ses->forward(std::vector<int32_t>(ids.begin() + pos, ids.begin() + end), want);
                            chunk_s.push_back(now_s() - tc);
                            chunk_ns.push_back(now_ns());
                            pos = end;
                        }
                        const double prefill_s = now_s() - tp;
                        const auto p1 = ses->ple_stats();
                        const size_t used_after = dev_used();
                        const int64_t V = (int64_t)logits.size() / std::max<int64_t>(1, std::min(kl, L));
                        Sha256 sh;
                        sh.add(logits.data(), logits.size() * 4);
                        const std::string sha = sh.hex();
                        std::vector<int32_t> top;
                        for (size_t row = 0; row * (size_t)V < logits.size(); ++row) {
                            const float *x = logits.data() + row * (size_t)V;
                            top.push_back((int32_t)(std::max_element(x, x + V) - x));
                        }
                        if (save_logits && r == 0) {
                            std::ofstream lf(out_dir + "/logits." + v.name + "." + std::to_string(L) + ".f32",
                                             std::ios::binary);
                            lf.write(reinterpret_cast<const char *>(logits.data()), (std::streamsize)(logits.size() * 4));
                        }
                        // greedy decode, T = 1 forwards (MTP off)
                        std::vector<int32_t> gen_ids;
                        int32_t next = top.back();
                        const double td = now_s();
                        for (int64_t g = 0; g < gen && ses->pos() < capacity - 1; ++g) {
                            gen_ids.push_back(next);
                            const std::vector<float> lg = ses->forward({next}, 1);
                            next = (int32_t)(std::max_element(lg.begin(), lg.end()) - lg.begin());
                        }
                        const double decode_s = now_s() - td;
                        std::ostringstream j;
                        j << "{\"round\":" << r << ",\"variant\":\"" << v.name << "\",\"chunk\":" << v.chunk
                          << ",\"len\":" << L << ",\"prefill_s\":" << jnum(prefill_s)
                          << ",\"prefill_tps\":" << jnum((double)L / prefill_s) << ",\"chunk_s\":[";
                        for (size_t i = 0; i < chunk_s.size(); ++i) j << (i ? "," : "") << jnum(chunk_s[i]);
                        j << "],\"chunk_ns\":[";
                        for (size_t i = 0; i < chunk_ns.size(); ++i) j << (i ? "," : "") << chunk_ns[i];
                        j << "],\"io_read_bytes\":" << io_read_bytes() << ",\"ple_wait_s\":" << jnum(p1.wait_seconds - p0.wait_seconds)
                          << ",\"ple_gather_s\":" << jnum(p1.gather_seconds - p0.gather_seconds)
                          << ",\"ple_waits\":" << (p1.waits - p0.waits) << ",\"logits_rows\":" << top.size()
                          << ",\"vocab\":" << V << ",\"logits_sha256\":\"" << sha << "\",\"top1\":[";
                        for (size_t i = 0; i < top.size(); ++i) j << (i ? "," : "") << top[i];
                        j << "],\"gen_ids\":[";
                        for (size_t i = 0; i < gen_ids.size(); ++i) j << (i ? "," : "") << gen_ids[i];
                        j << "],\"decode_tps\":" << jnum((double)gen_ids.size() / decode_s)
                          << ",\"dev_used_before_session\":" << used0 << ",\"dev_used_session\":" << used_ses
                          << ",\"dev_used_after\":" << used_after << ",\"env\":{";
                        for (size_t i = 0; i < v.env.size(); ++i)
                            j << (i ? "," : "") << "\"" << v.env[i].first << "\":\"" << v.env[i].second << "\"";
                        j << "}}";
                        out << j.str() << "\n";
                        out.flush();
                        std::fprintf(stderr, "prefill_bench: r%d %s len %lld: %.1f tok/s prefill, %.2f tok/s decode, sha %.16s\n",
                                     r, v.name.c_str(), (long long)L, (double)L / prefill_s,
                                     (double)gen_ids.size() / decode_s, sha.c_str());
                    }
                }
                for (const auto &kv : v.env) unsetenv(kv.first.c_str());
            }
        }
        (void)D;
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "prefill_bench: %s\n", e.what());
        return 1;
    }
}
