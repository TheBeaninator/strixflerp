// SPDX-License-Identifier: MIT
// kernel.flerp.moe_kgrouped: grouped prefill GEMMs over the GGUF K-quant blocks as stored (Q4_K, Q6_K), for
// executor.flerp.qwen35moe prompt chunks (docs/PREFILL-ORNITH.md). The weights are read in place: no repack, no second
// copy, the decode kernels untouched. Each K step decodes a 128-row x 64-K weight tile from its blocks into LDS as F16
// (value = d*sc*q - dmin*m in FP32, one rounding to F16) and every row tile of the work item multiplies it on the
// gfx11 matrix units (v_wmma_f32_16x16x16_f16, FP32 accumulation over the whole K extent).
//
//   Group     ids [pairs] (pair = token * used + slot) -> per-expert counts, offsets, the pairs sorted by expert, and a
//             tile list (expert, first row) of BM-row work items. ids == nullptr: one group of every pair, in order
//             (the shared expert and dense projections).
//   ToF16     FP32 rows -> F16 rows (round to nearest even); a value outside F16 range sets *overflow.
//   GateUp    h[pos] = silu(x . gate_e) * (x . up_e) for each sorted pair, F16 out in sorted order, x row = token.
//   Down      y[pair] = h[pos] . down_e, FP32 out at the pair's own row (so the combine reads [token, slot] rows).
//   Linear    y[row] = x[row] . W (a dense projection: Down's kernel over the identity group).
//
// Every output row is a function of its own input row only (WMMA's per-element accumulation runs over K in a fixed
// order, whatever else shares the tile), so outputs are bit-identical whatever the grouping, the order within a group,
// the chunk cut or BM. Device pointers, asynchronous on `stream`.
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace flerp::kgrouped {

enum class WType : int { kQ4K = 12, kQ6K = 14 };  // ggml type ids

// Bytes of one weight row (K values) and of one expert's [rows, K] stack.
std::size_t RowBytes(WType type, std::size_t k) noexcept;

// The device scratch of a grouping: counts, offsets and fill [n_experts], the sorted pairs [pairs], the tile list
// [MaxTiles] and its length. Laid out from one pointer (256-byte aligned pieces).
struct Plan {
  int* counts = nullptr;
  int* offsets = nullptr;
  int* fill = nullptr;
  int* sorted = nullptr;
  int2* tiles = nullptr;
  int* n_tiles = nullptr;
  std::size_t pairs = 0;
  std::size_t n_experts = 0;
  int bm = 0;
  std::size_t max_tiles = 0;
};

std::size_t MaxTiles(std::size_t pairs, std::size_t n_experts, int bm) noexcept;
std::size_t PlanBytes(std::size_t pairs, std::size_t n_experts, int bm) noexcept;
Plan MakePlan(void* scratch, std::size_t pairs, std::size_t n_experts, int bm) noexcept;

// BM (16-row tiles x 2 x TM) for a grouping of `pairs` over `n_experts`: rows per active expert, 32 at up to 40, else 128
// (measured in gate/results/prefill-ornith; 64 is supported, never the fastest there).
int ChooseBm(std::size_t pairs, std::size_t n_experts) noexcept;

void Group(const std::int32_t* ids, Plan& plan, hipStream_t stream);

// PREFILL-KPROF (9506f4f116): live-tile items in GateUp / Down / Linear (F16 path): a wave skips the WMMAs of its
// 16-row tiles past the item's rows. Bit-identical outputs. -1 = FLERP_KG_LIVE (default off), 0 off, 1 on.
void SetLiveTiles(int mode) noexcept;
void ToF16(const float* x, std::uint16_t* xh, std::size_t n, unsigned* overflow, hipStream_t stream);

// x_share: pairs per token (the router's used count; 1 for the shared expert).
void GateUp(WType type, const void* w_gate, const void* w_up, const std::uint16_t* xh, std::size_t x_share,
            const Plan& plan, std::uint16_t* h, std::size_t n, std::size_t k, unsigned* overflow, hipStream_t stream);
void Down(WType type, const void* w_down, const std::uint16_t* h, const Plan& plan, float* y, std::size_t n,
          std::size_t k, hipStream_t stream);
// Dense: y [rows, n] = xh [rows, k] . W^T, W one [n, k] stack. plan: Group(nullptr) over `rows` pairs, one expert.
void Linear(WType type, const void* w, const std::uint16_t* xh, const Plan& plan, float* y, std::size_t n,
            std::size_t k, hipStream_t stream);

// Exact variant: weight codes exact in F16 with scale and min applied in FP32; x and h as hi/lo F16 pairs
// (lo = f16(v - f16(v))), two WMMAs a fragment. BM 32 at up to 40 rows an expert, else 64.
int ChooseBmExact(std::size_t pairs, std::size_t n_experts) noexcept;
void ToF16Split(const float* x, std::uint16_t* hi, std::uint16_t* lo, std::size_t n, unsigned* overflow,
                hipStream_t stream);
void GateUpExact(WType type, const void* w_gate, const void* w_up, const std::uint16_t* x_hi,
                 const std::uint16_t* x_lo, std::size_t x_share, const Plan& plan, std::uint16_t* h_hi,
                 std::uint16_t* h_lo, std::size_t n, std::size_t k, unsigned* overflow, hipStream_t stream);
void DownExact(WType type, const void* w_down, const std::uint16_t* h_hi, const std::uint16_t* h_lo, const Plan& plan,
               float* y, std::size_t n, std::size_t k, hipStream_t stream);
void LinearExact(WType type, const void* w, const std::uint16_t* x_hi, const std::uint16_t* x_lo, const Plan& plan,
                 float* y, std::size_t n, std::size_t k, hipStream_t stream);

}  // namespace flerp::kgrouped
