#pragma once

// format.flerp.panel_device: Q4Panel16 / Q8Panel16 tensors (format.flerp.q4panel, q8panel) resident on the device, as
// the kernels of the strixw panel layout take them (docs/STRIXW-LAYOUT-NEXT.md stages 3-5).
//
// One buffer per tensor: panel p (rows 16p .. 16p+15) at byte p * P, P = codes + 64K/G with codes = 8K (Q4) or 16K
// (Q8): chunk c of row 16p + r (16 bytes) at panel byte (16c + r) * 16, then (bf16 scale, bf16 min) of group g of
// row 16p + r at panel byte codes + (16g + r) * 4.
//
// Handing panels to the existing call sites. strixite's executor passes strix::Q4DeviceView / Q8DeviceView (inside a
// strix::QWeightView) to the kernels; a panel tensor travels in those same structs, tagged so that no row-major
// kernel can read it and no panel kernel can read a row-major one:
//   q = the panel buffer, scale = minv = nullptr, G = (encoding id << 32) | group size,
// the encoding id being the strixw one (StrixwEncoding::Q4Panel16 = 10, Q8Panel16 = 11, format.flerp.strixw_v5). Every
// row-major kernel of strixite refuses a null scale pointer or a group size outside 32/64/128; the panel kernels
// (kernel.flerp.linear_q4p, linear_q8p, wmma_q4p, moe_grouped_p) take a tagged view only, and their adapters send a
// call to the panel kernel or to strixite's by this tag. N may be below the panels' rows (a row prefix, as the MTP
// head truncation does): rows >= N are never stored.

#include "runtime/q4_device.hpp"
#include "runtime/q8_device.hpp"

#include <cstdint>

#if defined(__HIPCC__)
#define FLERP_PANEL_HD __host__ __device__
#else
#define FLERP_PANEL_HD
#endif

namespace flerp {

constexpr int64_t kQ4PanelEncoding = 10, kQ8PanelEncoding = 11;  // StrixwEncoding::Q4Panel16 / Q8Panel16
constexpr int kPanelEncodingShift = 32;

struct PanelView {
    const uint8_t *p = nullptr;
    int64_t N = 0, K = 0, G = 0;
    int bits = 0;  // 4 or 8
    // bytes of a panel's codes (8K or 16K) and of a whole panel
    FLERP_PANEL_HD int64_t codes_bytes() const { return bits == 4 ? 8 * K : 16 * K; }
    FLERP_PANEL_HD int64_t panel_bytes() const { return codes_bytes() + 64 * K / G; }
};

FLERP_PANEL_HD inline int64_t panel_tag(int64_t encoding, int64_t G) { return (encoding << kPanelEncodingShift) | G; }
FLERP_PANEL_HD inline int64_t panel_tag_encoding(int64_t tagged_G) { return tagged_G >> kPanelEncodingShift; }
FLERP_PANEL_HD inline int64_t panel_tag_group(int64_t tagged_G) {
    return tagged_G & (((int64_t)1 << kPanelEncodingShift) - 1);
}

// Whether a view carries panels (the tag above), not row-major planes.
inline bool is_panel(const strix::Q4DeviceView &v) {
    return panel_tag_encoding(v.G) == kQ4PanelEncoding && v.scale == nullptr && v.minv == nullptr;
}
inline bool is_panel(const strix::Q8DeviceView &v) {
    return panel_tag_encoding(v.G) == kQ8PanelEncoding && v.scale == nullptr && v.minv == nullptr;
}

// The tagged views (the loader makes them from a Q4Panel16 / Q8Panel16 tensor).
inline strix::Q4DeviceView q4_panel_view(const uint8_t *p, int64_t N, int64_t K, int64_t G) {
    return {p, nullptr, nullptr, N, K, panel_tag(kQ4PanelEncoding, G)};
}
inline strix::Q8DeviceView q8_panel_view(const uint8_t *p, int64_t N, int64_t K, int64_t G) {
    return {p, nullptr, nullptr, N, K, panel_tag(kQ8PanelEncoding, G)};
}

// Tagged view -> PanelView; G is the plain group size. The caller checks is_panel first.
inline PanelView panel_of(const strix::Q4DeviceView &v) { return {v.q, v.N, v.K, panel_tag_group(v.G), 4}; }
inline PanelView panel_of(const strix::Q8DeviceView &v) { return {v.q, v.N, v.K, panel_tag_group(v.G), 8}; }

#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC__)
// Device addressing. row: absolute row of the panel buffer; chunk: 16-byte chunk index along the row.
__device__ inline const uint8_t *panel_chunk(const uint8_t *p, int64_t P, int64_t row, int64_t chunk) {
    return p + (row >> 4) * P + ((chunk << 4) + (row & 15)) * 16;
}
// (scale | min << 16) of group g of `row` (both BF16 bits).
__device__ inline uint32_t panel_sm(const uint8_t *p, int64_t P, int64_t codes, int64_t row, int64_t g) {
    return *reinterpret_cast<const uint32_t *>(p + (row >> 4) * P + codes + ((g << 4) + (row & 15)) * 4);
}
#endif

}  // namespace flerp
