#pragma once

// kernel.flerp.wmma_q4p: kernels/wmma_gemm.hpp's WmmaWCodes (strixite) reading Q4Panel16 / Q8Panel16 tensors
// (runtime/panel_device.hpp). Included by .hip files only (wmma_q4p.hip, kernel.flerp.moe_grouped_p).
//
// Every thread fetches exactly the codes, scales and mins WmmaWCodes gives it - the same staged row r, the same K
// offset, the same (r, slot) of the scale window - so stage_f16 writes the same F16 bits to LDS and every WMMA after it
// sees the same operands: the outputs are bit-equal to the row-major kernels'. Only the addresses change, so that a
// wave's loads are whole cache lines:
//   codes  thread l takes staged row r = l / kLoadsPerRow, 16-byte load j = l % kLoadsPerRow (WmmaWCodes' mapping).
//          Q4 (2 loads a row): a wave = 16 rows x chunks c, c + 1 = one panel's 512 contiguous bytes; Q8 (4 a row):
//          8 rows x chunks c .. c + 3 = four 128-byte runs. Row-major: 16 or 8 rows, 32 bytes of each 128-byte line.
//   window (scale, min) of kWin = 8 groups x 128 rows: a 16-row panel's 8 groups are 512 contiguous bytes
//          ([g][r] x 4 bytes), so entry e is mapped to panel e / 128, group (e % 128) / 16, row (e % 128) % 16 - a wave
//          reads one 128-byte line, scale and min in one 4-byte load - and stage_window writes it to the (row, slot)
//          WmmaWCodes would have written it to (sw / mw [BN][kWin], the same LDS contents).
// Staged rows past rows_valid (or row_of < 0) and groups past the row's end are zero, as there.

#include "kernels/wmma_gemm.hpp"
#include "runtime/panel_device.hpp"

#include <cstdint>

namespace strix::kernels {

template <int BITS, int G, int BN>
struct WmmaWCodesP {
    static_assert(BITS == 4 || BITS == 8, "panels: Q4 / Q8");
    static constexpr int kCodesPerLoad = BITS == 4 ? 32 : 16, kLoadsPerRow = kWmmaKC / kCodesPerLoad;
    static constexpr int kLoads = BN * kLoadsPerRow / kWmmaThreads;
    static constexpr int kWin = 8, kWinLoads = BN * kWin / kWmmaThreads;
    static_assert(BN * kLoadsPerRow % kWmmaThreads == 0 && BN * kWin % kWmmaThreads == 0, "whole loads per thread");
    static_assert(BN % 16 == 0, "whole panels a tile");
    uint4 q[kLoads];
    uint16_t s[kWinLoads], m[kWinLoads];
    bool win = false;

    __device__ static bool opens_window(int64_t k0) { return k0 % G == 0 && (k0 / G) % kWin == 0; }
    __device__ static int64_t panel_bytes(int64_t K) { return (BITS == 4 ? 8 : 16) * K + 64 * K / G; }
    __device__ static int64_t codes_bytes(int64_t K) { return (BITS == 4 ? 8 : 16) * K; }
    // Window entry e -> its staged row and slot (the panel-major order above).
    __device__ static int win_row(int e) { return (e / 128) * 16 + (e % 128) % 16; }
    __device__ static int win_slot(int e) { return (e % 128) / 16; }

    // As WmmaWCodes::fetch_mapped: staged row r from global row row_of(r) (< 0: zeros). panels: the tensor's panel
    // buffer; scale / minv are unused (the panels carry them) and kept so call sites read as the origin's.
    template <typename RowFn>
    __device__ void fetch_mapped(const uint8_t *panels, const uint16_t *, const uint16_t *, RowFn row_of, int64_t K,
                                 int64_t k0) {
        const int64_t P = panel_bytes(K), codes = codes_bytes(K);
#pragma unroll
        for (int i = 0; i < kLoads; ++i) {
            const int l = (int)threadIdx.x + i * kWmmaThreads, r = l / kLoadsPerRow;
            const int64_t k = k0 + (l % kLoadsPerRow) * kCodesPerLoad, row = row_of(r);
            q[i] = row >= 0 ? *reinterpret_cast<const uint4 *>(flerp::panel_chunk(panels, P, row, k / kCodesPerLoad))
                            : make_uint4(0, 0, 0, 0);
        }
        win = opens_window(k0);
        if (win) {
            const int64_t groups = K / G, g0 = k0 / G;
#pragma unroll
            for (int i = 0; i < kWinLoads; ++i) {
                const int e = (int)threadIdx.x + i * kWmmaThreads;
                const int64_t g = g0 + win_slot(e), row = row_of(win_row(e));
                const uint32_t sm = row >= 0 && g < groups ? flerp::panel_sm(panels, P, codes, row, g) : 0u;
                s[i] = (uint16_t)(sm & 0xffffu), m[i] = (uint16_t)(sm >> 16);
            }
        }
    }
    // As WmmaWCodes::fetch (k_base 0): rows row_base + r, r < rows_valid.
    __device__ void fetch(const uint8_t *panels, const uint16_t *, const uint16_t *, int64_t row_base, int rows_valid,
                          int64_t K, int64_t k0) {
        fetch_mapped(panels, nullptr, nullptr, [&](int r) -> int64_t { return r < rows_valid ? row_base + r : -1; }, K,
                     k0);
    }
    __device__ void stage_window(uint16_t *sw, uint16_t *mw) const {
        if (win)
#pragma unroll
            for (int i = 0; i < kWinLoads; ++i) {
                const int e = (int)threadIdx.x + i * kWmmaThreads, at = win_row(e) * kWin + win_slot(e);
                sw[at] = s[i], mw[at] = m[i];  // [r][slot], WmmaWCodes' place for this (row, group)
            }
    }
    // WmmaWCodes::stage_f16, unchanged.
    __device__ void stage_f16(uint16_t *lds, const uint16_t *sw, const uint16_t *mw, int64_t k0) const {
#pragma unroll
        for (int i = 0; i < kLoads; ++i) {
            const int l = (int)threadIdx.x + i * kWmmaThreads, r = l / kLoadsPerRow;
            const int kin = (l % kLoadsPerRow) * kCodesPerLoad;
            const int slot = (int)(((k0 + kin) / G) % kWin);
            wmma_stage_codes_f16<BITS>(q[i], wmma_bf16_bits_to_f16x2(sw[r * kWin + slot]),
                                       wmma_bf16_bits_to_f16x2(mw[r * kWin + slot]), lds + r * kWmmaLd + kin);
        }
    }
};

}  // namespace strix::kernels
