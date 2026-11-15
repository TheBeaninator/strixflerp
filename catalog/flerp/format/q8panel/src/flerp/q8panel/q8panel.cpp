#include "flerp/q8panel/q8panel.hpp"

#include "common/check.hpp"

namespace flerp {

uint64_t q8_panel16_bytes(int64_t N, int64_t K, int64_t G) { return panel16_geom(N, K, G, 8, "q8_panel16_bytes").total_bytes; }

Q8Panel16 to_panel16(const strix::Q8Weight &w, const char *what) {
    strix::check_q8(w, what);
    const Panel16Geom g = panel16_geom(w.N, w.K, w.G, 8, what);
    Q8Panel16 p;
    p.N = w.N, p.K = w.K, p.G = w.G;
    p.data.resize(g.total_bytes);
    panel16_pack(g, w.q.data(), reinterpret_cast<const uint8_t *>(w.scale.data()),
                 reinterpret_cast<const uint8_t *>(w.minv.data()), p.data.data());
    return p;
}

strix::Q8Weight from_panel16(const Q8Panel16 &p, const char *what) {
    const Panel16Geom g = panel16_geom(p.N, p.K, p.G, 8, what);
    STRIX_CHECK(p.data.size() == g.total_bytes, what, ": Q8Panel16 holds ", p.data.size(), " bytes, expected ",
                g.total_bytes);
    strix::Q8Weight w;
    w.N = p.N, w.K = p.K, w.G = p.G;
    w.q.resize((size_t)(p.N * p.K));
    w.scale.resize((size_t)(p.N * g.groups));
    w.minv.resize((size_t)(p.N * g.groups));
    panel16_unpack(g, p.data.data(), w.q.data(), reinterpret_cast<uint8_t *>(w.scale.data()),
                   reinterpret_cast<uint8_t *>(w.minv.data()));
    return w;
}

}  // namespace flerp
