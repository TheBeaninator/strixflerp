#pragma once

// format.flerp.q8panel: Q8Panel16, format.flerp.q4panel's panel layout at one code per byte
// (docs/STRIXW-LAYOUT-NEXT.md candidate A, "Q8P16"). For W [N, K], G 32/64/128, N % 16 == 0, K % 32 == 0,
// K % G == 0; panel p of 16 rows, P = 16K + 64K/G bytes:
//   codes  chunk c = 0 .. K/16-1, row r: the 16-byte chunk c of row 16p + r (16 codes, k = 16c .. 16c+15) at
//          panel byte (c * 16 + r) * 16
//   s/m    group g, row r: bf16 scale then bf16 min at panel byte 16K + (g * 16 + r) * 4
// 16-byte chunks (not Q8ChunkMajor's 32-byte ones) so a lane load is 16 B for both widths and a panel's chunk
// run is 256 B either way. Order-only; dequantized values bit-identical to the row-major strix::Q8Weight.

#include "flerp/q4panel/q4panel.hpp"
#include "formats/q8.hpp"

namespace flerp {

struct Q8Panel16 {
    int64_t N = 0, K = 0, G = 0;
    std::vector<uint8_t> data;
    size_t bytes() const { return data.size(); }
};

uint64_t q8_panel16_bytes(int64_t N, int64_t K, int64_t G);
Q8Panel16 to_panel16(const strix::Q8Weight &w, const char *what);
strix::Q8Weight from_panel16(const Q8Panel16 &p, const char *what);

}  // namespace flerp
