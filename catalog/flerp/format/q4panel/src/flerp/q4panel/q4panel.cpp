#include "flerp/q4panel/q4panel.hpp"

#include "common/check.hpp"

#include <cstring>

namespace flerp {

Panel16Geom panel16_geom(int64_t N, int64_t K, int64_t G, int bits, const char *what) {
    STRIX_CHECK(bits == 4 || bits == 8, what, ": panel16 codes are 4 or 8 bits, got ", bits);
    STRIX_CHECK(strix::q4_group_size_supported(G), what, ": group size ", G, " (must be 32, 64 or 128)");
    STRIX_CHECK(N >= kPanelRows && N % kPanelRows == 0, what, ": panel16 needs N % 16 == 0, got N = ", N);
    STRIX_CHECK(K >= 32 && K % 32 == 0 && K % G == 0, what, ": panel16 needs K a multiple of 32 codes and of G = ",
                G, ", got K = ", K);
    Panel16Geom g;
    g.N = N, g.K = K, g.G = G, g.bits = bits;
    g.row_code_bytes = bits == 4 ? K / 2 : K;
    g.chunks = g.row_code_bytes / kPanelChunkBytes;
    g.groups = K / G;
    g.panels = N / kPanelRows;
    g.codes_bytes = (uint64_t)(kPanelRows * g.row_code_bytes);
    g.sm_bytes = (uint64_t)(kPanelRows * g.groups * 4);
    g.panel_bytes = g.codes_bytes + g.sm_bytes;
    g.total_bytes = (uint64_t)g.panels * g.panel_bytes;
    return g;
}

uint64_t panel16_code_offset(const Panel16Geom &g, int64_t n, int64_t c) {
    return (uint64_t)(n / kPanelRows) * g.panel_bytes + (uint64_t)(c * kPanelRows + n % kPanelRows) * kPanelChunkBytes;
}

uint64_t panel16_sm_offset(const Panel16Geom &g, int64_t n, int64_t grp) {
    return (uint64_t)(n / kPanelRows) * g.panel_bytes + g.codes_bytes + (uint64_t)(grp * kPanelRows + n % kPanelRows) * 4;
}

void panel16_pack(const Panel16Geom &g, const uint8_t *q, const uint8_t *scale, const uint8_t *minv, uint8_t *out) {
    for (int64_t p = 0; p < g.panels; ++p) {
        uint8_t *P = out + (uint64_t)p * g.panel_bytes;
        for (int64_t c = 0; c < g.chunks; ++c)
            for (int64_t r = 0; r < kPanelRows; ++r)
                std::memcpy(P + (c * kPanelRows + r) * kPanelChunkBytes,
                            q + (uint64_t)(p * kPanelRows + r) * g.row_code_bytes + c * kPanelChunkBytes,
                            kPanelChunkBytes);
        uint8_t *S = P + g.codes_bytes;
        for (int64_t gr = 0; gr < g.groups; ++gr)
            for (int64_t r = 0; r < kPanelRows; ++r) {
                const uint64_t i = (uint64_t)((p * kPanelRows + r) * g.groups + gr) * 2;
                std::memcpy(S + (gr * kPanelRows + r) * 4, scale + i, 2);
                std::memcpy(S + (gr * kPanelRows + r) * 4 + 2, minv + i, 2);
            }
    }
}

void panel16_unpack(const Panel16Geom &g, const uint8_t *in, uint8_t *q, uint8_t *scale, uint8_t *minv) {
    for (int64_t p = 0; p < g.panels; ++p) {
        const uint8_t *P = in + (uint64_t)p * g.panel_bytes;
        for (int64_t c = 0; c < g.chunks; ++c)
            for (int64_t r = 0; r < kPanelRows; ++r)
                std::memcpy(q + (uint64_t)(p * kPanelRows + r) * g.row_code_bytes + c * kPanelChunkBytes,
                            P + (c * kPanelRows + r) * kPanelChunkBytes, kPanelChunkBytes);
        const uint8_t *S = P + g.codes_bytes;
        for (int64_t gr = 0; gr < g.groups; ++gr)
            for (int64_t r = 0; r < kPanelRows; ++r) {
                const uint64_t i = (uint64_t)((p * kPanelRows + r) * g.groups + gr) * 2;
                std::memcpy(scale + i, S + (gr * kPanelRows + r) * 4, 2);
                std::memcpy(minv + i, S + (gr * kPanelRows + r) * 4 + 2, 2);
            }
    }
}

uint64_t q4_panel16_bytes(int64_t N, int64_t K, int64_t G) { return panel16_geom(N, K, G, 4, "q4_panel16_bytes").total_bytes; }

Q4Panel16 to_panel16(const strix::Q4Weight &w, const char *what) {
    strix::check_q4(w, what);
    const Panel16Geom g = panel16_geom(w.N, w.K, w.G, 4, what);
    Q4Panel16 p;
    p.N = w.N, p.K = w.K, p.G = w.G;
    p.data.resize(g.total_bytes);
    panel16_pack(g, w.q.data(), reinterpret_cast<const uint8_t *>(w.scale.data()),
                 reinterpret_cast<const uint8_t *>(w.minv.data()), p.data.data());
    return p;
}

strix::Q4Weight from_panel16(const Q4Panel16 &p, const char *what) {
    const Panel16Geom g = panel16_geom(p.N, p.K, p.G, 4, what);
    STRIX_CHECK(p.data.size() == g.total_bytes, what, ": Q4Panel16 holds ", p.data.size(), " bytes, expected ",
                g.total_bytes);
    strix::Q4Weight w;
    w.N = p.N, w.K = p.K, w.G = p.G;
    w.q.resize((size_t)(p.N * p.K / 2));
    w.scale.resize((size_t)(p.N * g.groups));
    w.minv.resize((size_t)(p.N * g.groups));
    panel16_unpack(g, p.data.data(), w.q.data(), reinterpret_cast<uint8_t *>(w.scale.data()),
                   reinterpret_cast<uint8_t *>(w.minv.data()));
    return w;
}

}  // namespace flerp
