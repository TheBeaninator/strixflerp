// SPDX-License-Identifier: MIT
// Shared by the interface gate drivers (gate/interface/drivers/*): case files, seeded inputs, device buffers,
// timing, comparisons, JSON output. Plain and slow on purpose; nothing here is on a serving path.
//
// A driver is run as:  driver <cases.txt> <out.json> <dump-dir>
// cases.txt: one case per line, `name key=value key=value ...` (written by gate/interface.py from the gate's
// config file). The driver writes one JSON object per case and, for parity cases, the provider's raw output
// to <dump-dir>/<case>.bin so that gate/interface.py can compare providers with each other bit for bit.
#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gate {

inline void hip_ok(hipError_t e, const char *what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

// --- cases ----------------------------------------------------------------------------------------------------
struct Case {
    std::string name;
    std::map<std::string, std::string> kv;
    std::string str(const std::string &k) const {
        auto it = kv.find(k);
        if (it == kv.end()) throw std::runtime_error("case " + name + ": missing " + k);
        return it->second;
    }
    std::string str(const std::string &k, const std::string &dflt) const {
        auto it = kv.find(k);
        return it == kv.end() ? dflt : it->second;
    }
    int64_t num(const std::string &k) const { return std::stoll(str(k)); }
    int64_t num(const std::string &k, int64_t dflt) const { return kv.count(k) ? std::stoll(str(k)) : dflt; }
};

inline std::vector<Case> read_cases(const char *path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error(std::string("cannot read ") + path);
    std::vector<Case> out;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        Case c;
        if (!(ss >> c.name) || c.name[0] == '#') continue;
        std::string tok;
        while (ss >> tok) {
            auto eq = tok.find('=');
            if (eq == std::string::npos) throw std::runtime_error("case " + c.name + ": bad token " + tok);
            c.kv[tok.substr(0, eq)] = tok.substr(eq + 1);
        }
        out.push_back(c);
    }
    return out;
}

// --- JSON (flat objects of numbers, booleans and strings) -----------------------------------------------------
struct Obj {
    std::vector<std::pair<std::string, std::string>> items;
    Obj &raw(const std::string &k, const std::string &v) { items.emplace_back(k, v); return *this; }
    Obj &s(const std::string &k, const std::string &v) {
        std::string q = "\"";
        for (char c : v) q += (c == '"' || c == '\\') ? std::string("\\") + c : c == '\n' ? std::string("\\n") : std::string(1, c);
        return raw(k, q + "\"");
    }
    Obj &b(const std::string &k, bool v) { return raw(k, v ? "true" : "false"); }
    Obj &i(const std::string &k, long long v) { return raw(k, std::to_string(v)); }
    Obj &d(const std::string &k, double v) {
        if (!std::isfinite(v)) return raw(k, v != v ? "\"nan\"" : v > 0 ? "\"inf\"" : "\"-inf\"");
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.9g", v);
        return raw(k, buf);
    }
    std::string text() const {
        std::string o = "{";
        for (size_t j = 0; j < items.size(); ++j) o += (j ? ", " : "") + ("\"" + items[j].first + "\": ") + items[j].second;
        return o + "}";
    }
};

inline void write_results(const char *path, const std::vector<Obj> &rows) {
    std::ofstream f(path);
    f << "[\n";
    for (size_t j = 0; j < rows.size(); ++j) f << " " << rows[j].text() << (j + 1 < rows.size() ? ",\n" : "\n");
    f << "]\n";
    if (!f) throw std::runtime_error(std::string("cannot write ") + path);
}

inline void dump(const std::string &dir, const std::string &name, const void *p, size_t bytes) {
    std::ofstream f(dir + "/" + name + ".bin", std::ios::binary);
    f.write(static_cast<const char *>(p), (std::streamsize)bytes);
    if (!f) throw std::runtime_error("cannot write dump for " + name);
}

// --- seeded inputs ---------------------------------------------------------------------------------------------
struct Rng {      // splitmix64: the same stream on every host
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    double uniform() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }      // [0, 1)
    float sym(float scale) { return (float)((uniform() * 2.0 - 1.0) * scale); }
    // roughly normal, sum of 4 uniforms: activations and logits look like this, not like a uniform
    float gauss(float sigma) { return (float)((uniform() + uniform() + uniform() + uniform() - 2.0) * 1.7320508 * sigma); }
};

inline uint32_t f32_bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline float bits_f32(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
inline float bf16_to_f32(uint16_t b) { return bits_f32((uint32_t)b << 16); }
inline uint16_t f32_to_bf16(float f) {      // round to nearest even, NaN kept quiet: torch's .to(bfloat16)
    uint32_t u = f32_bits(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
inline float half_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) << 31, exp = (h >> 10) & 0x1f, frac = h & 0x3ff;
    if (exp == 0) return bits_f32(sign) + (sign ? -1.0f : 1.0f) * std::ldexp((float)frac, -24);
    if (exp == 31) return bits_f32(sign | 0x7f800000u | (frac << 13));
    return bits_f32(sign | ((exp + 112) << 23) | (frac << 13));
}
inline uint16_t f32_to_half(float f) {      // round to nearest even; inputs here are finite and in range
    const uint32_t u = f32_bits(f), sign = (u >> 16) & 0x8000u;
    const int exp = (int)((u >> 23) & 0xff) - 127;
    uint32_t man = u & 0x7fffffu;
    if (exp < -24) return (uint16_t)sign;
    if (exp < -14) {
        man |= 0x800000u;
        const int shift = -exp - 14 + 13;
        uint32_t h = man >> shift;
        const uint32_t rem = man & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) ++h;
        return (uint16_t)(sign | h);
    }
    if (exp > 15) return (uint16_t)(sign | 0x7c00u);
    uint32_t h = ((uint32_t)(exp + 15) << 10) | (man >> 13);
    const uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;
    return (uint16_t)(sign | h);
}

// --- device buffers --------------------------------------------------------------------------------------------
template <typename T> struct Dev {
    T *p = nullptr;
    size_t n = 0;
    explicit Dev(size_t count) : n(count) { hip_ok(hipMalloc(&p, std::max<size_t>(count, 1) * sizeof(T)), "hipMalloc"); }
    Dev(const Dev &) = delete;
    Dev &operator=(const Dev &) = delete;
    ~Dev() { (void)hipFree(p); }
    void put(const std::vector<T> &h) { hip_ok(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "upload"); }
    void put(const T *h, size_t count) { hip_ok(hipMemcpy(p, h, count * sizeof(T), hipMemcpyHostToDevice), "upload"); }
    std::vector<T> get() const {
        std::vector<T> h(n);
        hip_ok(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "download");
        return h;
    }
    void fill_bytes(int v) { hip_ok(hipMemset(p, v, n * sizeof(T)), "hipMemset"); }
};

// --- timing ------------------------------------------------------------------------------------------------------
struct Timing {
    double median_us, min_us, max_us;
    int iters, repeats;
};

// Time per launch as the stream sees it: `iters` launches queued back to back, then one synchronize.
template <typename F> Timing time_launches(F &&launch, hipStream_t stream, int iters, int repeats) {
    for (int w = 0; w < 3; ++w) launch();
    hip_ok(hipStreamSynchronize(stream), "sync");
    std::vector<double> us;
    for (int r = 0; r < repeats; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int j = 0; j < iters; ++j) launch();
        hip_ok(hipStreamSynchronize(stream), "sync");
        const auto t1 = std::chrono::steady_clock::now();
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / iters);
    }
    std::sort(us.begin(), us.end());
    return {us[us.size() / 2], us.front(), us.back(), iters, repeats};
}

inline void put_timing(Obj &o, const Timing &t, double bytes_per_launch, int launches_per_call) {
    o.d("median_us", t.median_us).d("min_us", t.min_us).d("max_us", t.max_us).i("iters", t.iters).i("repeats", t.repeats);
    o.i("kernel_launches_per_call", launches_per_call).d("bytes_per_call", bytes_per_launch);
    o.d("gb_per_s", bytes_per_launch / (t.median_us * 1e-6) / 1e9);
}

// --- comparisons -------------------------------------------------------------------------------------------------
struct FloatDiff {
    long long n = 0, mismatched_bits = 0, nan_payload_only = 0, nan_class = 0;
    double max_abs = 0, max_rel = 0;
    bool identical() const { return mismatched_bits == 0; }
};

// Bit comparison of fp32 arrays. Two NaNs with different payloads are counted apart (nan_payload_only): the CPU
// and the GPU may propagate payloads differently and no engine promises them.
inline FloatDiff diff_f32(const float *got, const float *want, size_t n) {
    FloatDiff d;
    d.n = (long long)n;
    for (size_t j = 0; j < n; ++j) {
        if (f32_bits(got[j]) == f32_bits(want[j])) continue;
        const bool gn = got[j] != got[j], wn = want[j] != want[j];
        if (gn && wn) { ++d.nan_payload_only; continue; }
        ++d.mismatched_bits;
        if (gn != wn) { ++d.nan_class; continue; }
        const double a = std::fabs((double)got[j] - (double)want[j]);
        d.max_abs = std::max(d.max_abs, a);
        const double den = std::fabs((double)want[j]);
        if (den > 0) d.max_rel = std::max(d.max_rel, a / den);
    }
    return d;
}

inline void put_diff(Obj &o, const std::string &prefix, const FloatDiff &d) {
    o.b(prefix + "bit_identical", d.identical()).i(prefix + "mismatched", d.mismatched_bits);
    o.i(prefix + "nan_payload_only", d.nan_payload_only).i(prefix + "nan_class_differs", d.nan_class);
    o.d(prefix + "max_abs", d.max_abs).d(prefix + "max_rel", d.max_rel);
}

// Error of a fp32 dot product against the exact (double) one, in units of the bound that holds for any
// summation order of `terms` fp32 additions with up to `term_roundings` roundings inside each term:
//   |fl(sum) - sum| <= (terms - 1 + term_roundings) * u * sum |t_i|  (first order),  u = 2^-24.
// A value above 1 cannot be explained by summation order and rounding alone.
inline double order_bound(double sum_abs_terms, long long terms, int term_roundings) {
    return (double)(terms - 1 + term_roundings) * 5.9604644775390625e-08 * sum_abs_terms * 1.0001 + 1e-45;
}

template <typename Body> int run_driver(int argc, char **argv, const char *provider_note, Body &&body) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <cases.txt> <out.json> <dump-dir>\n", argv[0]);
        return 2;
    }
    try {
        hip_ok(hipInit(0), "hipInit");
        hipStream_t stream = nullptr;
        hip_ok(hipStreamCreate(&stream), "hipStreamCreate");
        std::vector<Obj> rows;
        for (const Case &c : read_cases(argv[1])) {
            Obj o;
            o.s("case", c.name).s("kind", c.str("kind"));
            try {
                body(c, o, stream, std::string(argv[3]));
                o.b("ran", true);
            } catch (const std::exception &e) {
                o.b("ran", false).s("error", e.what());
            }
            std::fprintf(stderr, "%s %s\n", provider_note, o.text().c_str());
            rows.push_back(o);
        }
        write_results(argv[2], rows);
        hip_ok(hipStreamDestroy(stream), "hipStreamDestroy");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "driver failed: %s\n", e.what());
        return 1;
    }
}

}  // namespace gate
