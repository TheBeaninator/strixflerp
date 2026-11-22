// SPDX-License-Identifier: MIT
// Coarse-then-exact scoring of a draft head (ORNITH-CRUCIBLE item 2, gate/ornith-crucible.toml): the recipe of
// Crucible's low-bit draft head (catalog/crucible/runtime/speculation: engine/runtime/draft_quant.h) re-expressed for
// a GGUF Q6_K / Q4_K output head read in place.
//
//   Build   once: a 2- or 4-bit copy of the head (asymmetric, 64-value groups, fp16 scale and minimum per group)
//           made on the device from the GGUF rows, and err_rms = max over rows of ||w_r - w~_r||_2 / sqrt(hidden).
//   Score   per draft step, four launches:
//     Coarse   coarse logits of every token from the low-bit copy, and the maximum of each 128-token block;
//     Select   the `keep` best blocks by coarse maximum (lower block first on a tie), and the best maximum outside them;
//     Rescore  the kept blocks' logits recomputed from the GGUF rows with gufo's QuantWarpBlockDot, the row dot of the
//              decode GEMV the exact head runs (LaunchGEMV -> LaunchQ8KBlockGEMV: Q8KBlockGEMVKernel_2Rows /
//              _2RowsWideQ4K), so each kept logit has the exact head's bits; written into `logits` in place;
//     Choose   the best and second-best exact logit of the kept blocks (finite values only, lower id on a tie: the
//              rule of gufo's ArgmaxKernel), tau = k * err_rms * ||x||_2, and fallback = !(second - outside >= tau).
//   The caller runs the exact head when fallback is set. Without fallback, every token outside the kept blocks has a
//   coarse logit below second - tau; when |coarse - exact| < tau for those tokens (k standard deviations of the
//   coarse error under the model |e . x| ~ err_rms ||x|| |z|), the exact top-1 and top-2 are inside the kept blocks,
//   so the draft id and the top-2 margin equal the exact head's. The gate measures how often that model fails.
#ifndef FLERP_DRAFT_COARSE_HPP_
#define FLERP_DRAFT_COARSE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace flerp::draft_coarse {

struct Result {
  std::uint32_t id{0};
  float best{0.0F};
  float second{0.0F};
  float outside{0.0F};   // the best coarse block maximum outside the kept blocks (-inf when none)
  float tau{0.0F};
  float xnorm{0.0F};
  std::uint32_t fallback{0};
  std::uint32_t pad{0};
};

constexpr int kBlockTokens = 128;
constexpr int kMaxKeep = 32;

class Coarse {
 public:
  ~Coarse();
  Coarse(const Coarse&) = delete;
  Coarse& operator=(const Coarse&) = delete;

  /// head: device rows [vocab][hidden] in `type` (Q6_K or Q4_K). bits 2 or 4; keep 1..32 blocks; hidden a multiple
  /// of 2048. Null with *error when the shape or type is not one this reads.
  [[nodiscard]] static std::unique_ptr<Coarse> Build(const void* head, gufo::core::GgmlType type, int vocab, int hidden,
                                                     int bits, int keep, float k_tau, hipStream_t stream,
                                                     std::string* error);

  /// x F32 [hidden] (device). logits F32 [vocab]: coarse values, the kept blocks' exact. result: device Result.
  void Score(const float* x, float* logits, Result* result, hipStream_t stream) const;
  /// The passes of Score one at a time (bench timing).
  void CoarsePass(const float* x, float* logits, hipStream_t stream) const;
  void SelectPass(hipStream_t stream) const;
  void RescorePass(const float* x, float* logits, hipStream_t stream) const;
  void ChoosePass(const float* x, Result* result, hipStream_t stream) const;

  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] float err_rms() const noexcept { return err_rms_; }
  [[nodiscard]] int bits() const noexcept { return bits_; }
  [[nodiscard]] int keep() const noexcept { return keep_; }
  [[nodiscard]] float k_tau() const noexcept { return k_tau_; }
  [[nodiscard]] int blocks() const noexcept { return blocks_; }
  /// The last Score's kept blocks (device, keep ints) and candidate logits (device, keep * 128 floats).
  [[nodiscard]] const int* chosen() const noexcept { return chosen_; }
  /// ||w_r - w~_r||_2 of every row (host, vocab floats), the exact per-row coarse error. The head multiplies the F32
  /// activation directly (QuantWarpBlockDot), so |exact_r - coarse_r| <= row_err()[r] * ||x||_2 up to FP32 rounding.
  [[nodiscard]] const std::vector<float>& row_err() const noexcept { return row_err_; }

 private:
  Coarse() = default;
  const void* head_{nullptr};
  gufo::core::GgmlType type_{gufo::core::GgmlType::kQ6_K};
  int vocab_{0}, hidden_{0}, bits_{0}, keep_{0}, blocks_{0};
  float k_tau_{8.0F};
  float err_rms_{0.0F};
  std::size_t bytes_{0};
  std::uint32_t* codes_{nullptr};
  __half* scale_{nullptr};
  __half* minv_{nullptr};
  float* maxima_{nullptr};
  int* chosen_{nullptr};
  float* outside_{nullptr};
  float* cand_{nullptr};
  std::vector<float> row_err_;
};

}  // namespace flerp::draft_coarse

#endif  // defined(ENGINE_ENABLE_HIP)
#endif  // FLERP_DRAFT_COARSE_HPP_
