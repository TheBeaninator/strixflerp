// format.flerp.q4panel / q8panel / strixw_v5 unit test, host only (built by
// catalog/flerp/tool/convert_strixw_panel/src/flerp/convert_strixw_panel/build.sh OUT this.cpp).
// For every Q4/Q8 class shape of the fleet's Qwen3.8-Flash-Next file (gate/results/strixw-layout/header-diff.md), at
// both code widths: geometry, 128-byte alignment, strixw_v5 component size = the row-major q + scale + min bytes,
// every chunk and scale/min at the offset docs/STRIXW-LAYOUT-NEXT.md candidate A states (computed here from the
// formula, not from q4panel's helpers), exact round trip. Then refusals, and a tiny v5 file written and re-read.
// Prints one line per case; exit 0 = all passed.

#include "common/check.hpp"
#include "flerp/q4panel/q4panel.hpp"
#include "flerp/q8panel/q8panel.hpp"
#include "formats/strixw.hpp"

#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace strix;

namespace {

int fails = 0;
#define EXPECT(cond, ...)                                                                                               \
    do {                                                                                                                \
        if (!(cond)) {                                                                                                  \
            ++fails;                                                                                                    \
            std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #cond);                                                  \
        }                                                                                                               \
    } while (0)

uint64_t rng = 0x9e3779b97f4a7c15ull;
uint64_t next() {
    rng ^= rng << 13, rng ^= rng >> 7, rng ^= rng << 17;
    return rng;
}
void fill(uint8_t *p, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const uint64_t v = next();
        std::memcpy(p + i, &v, 8);
    }
    for (; i < n; ++i) p[i] = (uint8_t)next();
}

struct Shape {
    const char *cls;
    int64_t N, K, G, experts;
};

// Fleet shapes (header-diff.md): N, K, G as stored; experts stacked [E * n, K].
const Shape kShapes[] = {
    {"exp_gu", 655360, 2560, 128, 512},  {"exp_down", 1310720, 640, 32, 512}, {"sh_gu", 1280, 2560, 64, 0},
    {"sh_down", 2560, 640, 32, 0},       {"gdn_in", 16480, 2560, 128, 0},     {"gdn_out", 2560, 6144, 64, 0},
    {"attn_qkv", 13952, 2560, 64, 0},    {"attn_o", 2560, 6144, 64, 0},       {"ple_kv", 12800, 2560, 64, 0},
    {"lm_head", 248320, 2560, 64, 0},    {"mtp_fc", 2560, 2560, 64, 0},      
    {"exp_down(upstream g128)", 1310720, 640, 128, 512}, {"sh_down(upstream g64)", 2560, 640, 64, 0},
};
constexpr int64_t kDataRowsCap = 20480;  // rows filled with data per case (panels are independent; full K and G)

void one(const Shape &s, int bits) {
    const flerp::Panel16Geom g = flerp::panel16_geom(s.N, s.K, s.G, bits, s.cls);
    const int64_t rcb = bits == 4 ? s.K / 2 : s.K;
    EXPECT(g.panel_bytes == (uint64_t)(16 * rcb + 64 * s.K / s.G));
    EXPECT(g.total_bytes == (uint64_t)(s.N * rcb + 2 * 2 * s.N * (s.K / s.G)));
    const bool aligned = g.panel_bytes % 128 == 0 && g.codes_bytes % 128 == 0;
    // The design's claim (64K/G % 128 == 0 "for every class at G32/64/128 with K >= 256") holds for every fleet
    // class; it fails for K = 640 at G128 (64 * 640 / 128 = 320), the upstream exp_down layout, which is
    // therefore expected unaligned here (panels then start on 64-byte boundaries: 5440 = 85 * 64).
    const bool expect_aligned = !(s.K == 640 && s.G == 128);
    EXPECT(aligned == expect_aligned);
    if (!aligned) EXPECT(g.panel_bytes % 64 == 0);
    // strixw_v5 plans one Q component of exactly that size (the old encoding's three planes, no more no less).
    StrixwTensor t;
    t.name = s.cls, t.shape = {s.N, s.K}, t.group_size = s.G, t.experts = s.experts;
    t.encoding = bits == 4 ? StrixwEncoding::Q4Panel16 : StrixwEncoding::Q8Panel16;
    const auto pc = strixw_components(t);
    t.encoding = bits == 4 ? StrixwEncoding::Q4RowMajor : StrixwEncoding::Q8RowMajor;
    const auto rc = strixw_components(t);
    EXPECT(pc.size() == 1 && pc[0].role == StrixwRole::Q && pc[0].bytes == rc[0].bytes + rc[1].bytes + rc[2].bytes);
    if (s.experts) EXPECT((s.N / s.experts) % 16 == 0);

    // Data: the first min(N, cap) rows (a multiple of 16), all of K.
    const int64_t n = std::min<int64_t>(s.N, kDataRowsCap);
    const flerp::Panel16Geom d = flerp::panel16_geom(n, s.K, s.G, bits, s.cls);
    std::vector<uint8_t> q((size_t)(n * rcb)), sc((size_t)(n * d.groups * 2)), mn(sc.size()), pan(d.total_bytes);
    fill(q.data(), q.size()), fill(sc.data(), sc.size()), fill(mn.data(), mn.size());
    flerp::panel16_pack(d, q.data(), sc.data(), mn.data(), pan.data());
    // The design's offsets, written out here: panel p at p * P; chunk c of row r at (c * 16 + r) * 16;
    // group g's (scale, min) of row r at codes_bytes + (g * 16 + r) * 4.
    const uint64_t P = (uint64_t)(16 * rcb + 64 * s.K / s.G), CB = (uint64_t)(16 * rcb);
    uint64_t bad = 0;
    for (int64_t row = 0; row < n; ++row) {
        const int64_t p = row / 16, r = row % 16;
        for (int64_t c = 0; c < rcb / 16; ++c)
            bad += std::memcmp(pan.data() + p * P + (c * 16 + r) * 16, q.data() + row * rcb + c * 16, 16) != 0;
        for (int64_t gr = 0; gr < s.K / s.G; ++gr) {
            const uint8_t *e = pan.data() + p * P + CB + (gr * 16 + r) * 4;
            bad += std::memcmp(e, sc.data() + (row * (s.K / s.G) + gr) * 2, 2) != 0;
            bad += std::memcmp(e + 2, mn.data() + (row * (s.K / s.G) + gr) * 2, 2) != 0;
        }
    }
    EXPECT(bad == 0);
    std::vector<uint8_t> q2(q.size()), s2(sc.size()), m2(mn.size());
    flerp::panel16_unpack(d, pan.data(), q2.data(), s2.data(), m2.data());
    const bool rt = q2 == q && s2 == sc && m2 == mn;
    EXPECT(rt);
    std::printf("%-24s q%d N=%-7lld K=%-5lld G=%-3lld panels=%-6lld panel_bytes=%-6llu (codes %llu + s/m %llu) %s, "
                "data rows %lld: offsets %s, round trip %s\n",
                s.cls, bits, (long long)s.N, (long long)s.K, (long long)s.G, (long long)g.panels,
                (unsigned long long)g.panel_bytes, (unsigned long long)g.codes_bytes, (unsigned long long)g.sm_bytes,
                aligned ? "128-aligned" : "NOT-128-aligned", (long long)n, bad ? "WRONG" : "as designed",
                rt ? "exact" : "DIFFERENT");
}

template <class F> bool throws(F f) {
    try {
        f();
    } catch (const strix::Error &) {
        return true;
    }
    return false;
}

void typed_round_trip() {
    // Through the typed API and real quantizers: dequantized values identical after the round trip.
    std::vector<float> w(64 * 256);
    for (float &x : w) x = (float)((int64_t)(next() % 2001) - 1000) / 7919.0f;
    const Q4Weight a = quantize_q4(w.data(), 64, 256, 64);
    const Q4Weight b = flerp::from_panel16(flerp::to_panel16(a, "q4"), "q4");
    EXPECT(a.q == b.q && a.scale == b.scale && a.minv == b.minv && dequantize_q4(a) == dequantize_q4(b));
    const Q8Weight c = quantize_q8(w.data(), 64, 256, 32);
    const Q8Weight e = flerp::from_panel16(flerp::to_panel16(c, "q8"), "q8");
    EXPECT(c.q == e.q && c.scale == e.scale && c.minv == e.minv && dequantize_q8(c) == dequantize_q8(e));
    EXPECT(flerp::to_panel16(a, "q4").bytes() == a.bytes() && flerp::to_panel16(c, "q8").bytes() == c.bytes());
    std::printf("typed API: Q4 and Q8 quantized, panel round trip exact, dequantized values identical\n");
}

void refusals() {
    EXPECT(throws([] { flerp::panel16_geom(130, 256, 64, 4, "N%16"); }));
    EXPECT(throws([] { flerp::panel16_geom(2052, 256, 64, 8, "N%16"); }));
    EXPECT(throws([] { flerp::panel16_geom(32, 48, 32, 4, "K%32"); }));    // K not a multiple of 32 codes
    EXPECT(throws([] { flerp::panel16_geom(32, 160, 64, 4, "K%G"); }));    // 160 % 64 (the 4x160 split at G64)
    EXPECT(throws([] { flerp::panel16_geom(32, 256, 16, 4, "G"); }));      // unsupported G
    EXPECT(throws([] { flerp::panel16_geom(32, 256, 64, 6, "bits"); }));
    EXPECT(!throws([] { flerp::panel16_geom(32, 160, 32, 4, "160 at G32"); }));
    StrixwTensor t;  // experts whose per-expert rows are not whole panels
    t.name = "x", t.encoding = StrixwEncoding::Q4Panel16, t.shape = {4 * 24, 256}, t.group_size = 64, t.experts = 4;
    EXPECT(throws([&] { strixw_components(t); }));
    std::printf("refusals: N %% 16, K %% 32, K %% G, G, bits, per-expert rows %% 16\n");
}

void tiny_file(const std::string &dir) {
    const std::string path = dir + "/selftest.v5.strixw";
    std::vector<float> w(48 * 128);
    for (float &x : w) x = (float)((int64_t)(next() % 2001) - 1000) / 3001.0f;
    const flerp::Q8Panel16 p = flerp::to_panel16(quantize_q8(w.data(), 48, 128, 64), "tiny");
    {
        StrixwTensor t;
        t.name = "w", t.encoding = StrixwEncoding::Q8Panel16, t.shape = {48, 128}, t.group_size = 64;
        StrixwWriter wr(path, {{"layout", "x=q8g64p16"}}, {t});
        wr.write(0, StrixwRole::Q, p.data.data(), p.data.size());
        wr.finish();
    }
    StrixwFile f(path);
    uint32_t v = 0;
    std::FILE *fp = std::fopen(path.c_str(), "rb");
    std::fseek(fp, 8, SEEK_SET);
    EXPECT(std::fread(&v, 4, 1, fp) == 1);
    std::fclose(fp);
    f.verify_hashes(1);
    const StrixwTensor &t = f.get("w");
    EXPECT(v == 5 && t.encoding == StrixwEncoding::Q8Panel16 && t.components.size() == 1 &&
           std::memcmp(f.data(t, StrixwRole::Q), p.data.data(), p.data.size()) == 0);
    std::remove(path.c_str());
    std::printf("strixw v5 file: written, re-read (version %u, q8-panel16), hashes verified\n", v);
}

}  // namespace

int main(int argc, char **argv) {
    try {
        for (const Shape &s : kShapes)
            for (int bits : {4, 8}) one(s, bits);
        typed_round_trip();
        refusals();
        tiny_file(argc > 1 ? argv[1] : "/tmp");
    } catch (const std::exception &e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    std::printf("%s (%d failures)\n", fails ? "FAILED" : "ALL PASSED", fails);
    return fails ? 1 : 0;
}
