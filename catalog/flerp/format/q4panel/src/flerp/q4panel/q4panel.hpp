#pragma once

// format.flerp.q4panel: Q4Panel16, the panel-interleaved Q4 layout of docs/STRIXW-LAYOUT-NEXT.md section 2,
// candidate A. Order-only: the codes, scales and mins of a row-major strix::Q4Weight, re-laid out, nothing
// re-quantized, so the dequantized values are bit-identical.
//
// For W [N, K], group size G (32/64/128), N % 16 == 0, K % 32 == 0, K % G == 0. Rows go in panels of 16
// (panel p = rows 16p .. 16p+15). Panel p occupies bytes [p * P, (p + 1) * P) of one buffer, P = panel_bytes:
//   codes   chunk c = 0 .. K/32-1, row r = 0 .. 15: the 16-byte chunk c of row 16p + r (32 codes, today's
//           nibble order: low nibble = even k) at panel byte (c * 16 + r) * 16 - each chunk unchanged
//   s/m     group g = 0 .. K/G-1, row r = 0 .. 15: 4 bytes at panel byte 8K + (g * 16 + r) * 4:
//           bf16 scale[16p + r][g], then bf16 min[16p + r][g]
//   P = 8K + 64K/G (codes 16 rows x K/2 bytes, then 16 rows x K/G groups x 4 bytes).
// Q8Panel16 (format.flerp.q8panel) is the same with one code per byte: 16-byte chunks of 16 codes,
// K/16 chunks, P = 16K + 64K/G.
//
// The relayout itself is code-width agnostic (Panel16Geom::row_code_bytes), so it lives here once and q8panel
// calls it.

#include "formats/q4.hpp"

#include <cstdint>
#include <vector>

namespace flerp {

constexpr int64_t kPanelRows = 16;        // rows per panel (the WMMA N dimension)
constexpr int64_t kPanelChunkBytes = 16;  // one lane load; Q4: 32 codes, Q8: 16 codes

struct Panel16Geom {
    int64_t N = 0, K = 0, G = 0, bits = 0;
    int64_t row_code_bytes = 0;  // K/2 (Q4) or K (Q8)
    int64_t chunks = 0;          // 16-byte chunks per row
    int64_t groups = 0;          // K / G
    int64_t panels = 0;          // N / 16
    uint64_t codes_bytes = 0;    // per panel: 16 * row_code_bytes
    uint64_t sm_bytes = 0;       // per panel: 16 * groups * 4
    uint64_t panel_bytes = 0;    // codes_bytes + sm_bytes
    uint64_t total_bytes = 0;    // panels * panel_bytes (== the row-major q + scale + min bytes)
};

// Throws (naming `what`) unless bits is 4 or 8, G is 32/64/128, N % 16 == 0, K % 32 == 0 and K % G == 0.
Panel16Geom panel16_geom(int64_t N, int64_t K, int64_t G, int bits, const char *what);

// Row-major planes -> panels. q: N * row_code_bytes bytes; scale, min: N * groups BF16 values (as bytes,
// little-endian, any alignment); out: total_bytes. Rows [row0, row0 + rows) only when given (rows % 16 == 0,
// row0 % 16 == 0; the pointers are then to those rows' planes and out to their panels).
void panel16_pack(const Panel16Geom &g, const uint8_t *q, const uint8_t *scale, const uint8_t *minv, uint8_t *out);
// The inverse: panels -> row-major planes.
void panel16_unpack(const Panel16Geom &g, const uint8_t *in, uint8_t *q, uint8_t *scale, uint8_t *minv);

// Byte offset of chunk c of row n's codes, and of row n's group g scale (min is +2), inside the panel buffer.
uint64_t panel16_code_offset(const Panel16Geom &g, int64_t n, int64_t c);
uint64_t panel16_sm_offset(const Panel16Geom &g, int64_t n, int64_t grp);

// A separate type, so a row-major kernel cannot be handed panels (as for strix::Q4ChunkMajor).
struct Q4Panel16 {
    int64_t N = 0, K = 0, G = 0;
    std::vector<uint8_t> data;  // panels, total_bytes
    size_t bytes() const { return data.size(); }
};

uint64_t q4_panel16_bytes(int64_t N, int64_t K, int64_t G);  // throws on an unsuitable shape
Q4Panel16 to_panel16(const strix::Q4Weight &w, const char *what);
strix::Q4Weight from_panel16(const Q4Panel16 &p, const char *what);

}  // namespace flerp
