// SPDX-License-Identifier: MIT
// executor.flerp.qwen35moe: the file's MTP (nextn) block as a draft head, for a qwen35moe GGUF whose header has
// nextn_predict_layers 1 (Qwen3.6-35B-A3B blk.40, Ornith-1.5 397B blk.60). The forward is llama.cpp's
// LLM_GRAPH_TYPE_DECODER_MTP graph for Qwen3.5/3.6 MoE (src/models/qwen35moe.cpp graph_mtp; driven as
// common/speculative.cpp common_speculative_impl_draft_mtp drives it), on this executor's own launches:
//
//   h      = output_norm(the target's pre-norm row)       (qwen35moe.cpp: "post-norm hidden state feeds both the LM
//                                                          head and the MTP seed")      -- StepTarget only
//   x      = concat(enorm(embed(token)), hnorm(h))         enorm first
//   y      = eh_proj . x
//   y     += attn_output(sigmoid(gate) * attention)        attn_norm, q/k/v, q_norm/k_norm, rope, this head's own KV
//   y     += LaunchMoeFfn(post_attention_norm(y))          the block's routed + gated shared experts
//   h_out  = shared_head_norm(y)                           (output_norm when the file has none)
//   logits = output . h_out                                (nextn.shared_head_head when the file has it)
//
// Positions are the token's own (llama.cpp): the pair (h_p, x_{p+1}) runs at position p + 1, and position 0 carries
// (zero h, x_0). A chain step feeds the previous step's h_out and its argmax at the next position. Attention reads
// positions [0, pos] of the head's KV, so a later step at a position overwrites what a chain step left there.
//
// The weights are the head's own device copy (one allocation): blk.N.* of the MTP block, plus token_embd when the
// target weights have none (a stage that does not start at block 0). They are read by pread from the GGUF or, with
// data_path, from a same-size copy at the same offsets (a placed stage's staged file). The target's output head and
// output_norm are used in place: the target must have the head (the last stage, or the whole model).
//
// Nothing here touches the target's arena, stream or graph: the head has its own stream and buffers, and a step
// waits for that stream before it returns.
//
// Coarse-then-exact head scoring (ORNITH-CRUCIBLE item 2, gate/ornith-crucible.toml; kernel.flerp.draft_coarse), off by
// default. FLERP_MTP_COARSE=q2|q4 builds a 2- or 4-bit copy of the head at Create; a logits step then scores every token
// from it, recomputes the FLERP_MTP_COARSE_BLOCKS (8) best 128-token blocks exactly (the exact head's row dot, so the
// same bits) and takes the draft from them, unless the coarse margin is under tau = FLERP_MTP_COARSE_K (8) x err_rms x
// ||x||: then the exact head runs as before. FLERP_MTP_COARSE_CHECK=1 also runs the exact head every step and counts
// misses (CoarseStatsJson; printed to stderr at destruction). The draft ids and the top-2 margin are the exact head's
// whenever the error model holds; the verify is exact either way, so the output never changes.
#ifndef FLERP_QWEN35MOE_MTP_HEAD_HPP_
#define FLERP_QWEN35MOE_MTP_HEAD_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/models/qwen/state.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::core {
class GgufReader;
}
namespace flerp::draft_coarse {
class Coarse;
struct Result;
}

namespace flerp::qwen35moe {

struct MtpHeadOptions {
  std::uint32_t max_context{4096};
  std::string data_path;   // empty: the GGUF the reader mapped
};

class MtpHead {
 public:
  ~MtpHead();
  MtpHead(const MtpHead&) = delete;
  MtpHead& operator=(const MtpHead&) = delete;

  /// target: the device weights of the executor that runs the model (or its last stage). Null with *error when the
  /// file has no usable MTP block or the target has no head.
  [[nodiscard]] static std::unique_ptr<MtpHead> Create(const gufo::core::GgufReader& reader,
                                                       const gufo::models::QwenModelWeights& target,
                                                       const MtpHeadOptions& options, std::string* error);

  /// (token, output_norm(d_target_prenorm_row)) at pos. Returns the head's argmax when logits are asked, else 0.
  std::uint32_t StepTarget(std::uint32_t token, const float* d_target_prenorm_row, std::uint32_t pos, bool logits);
  /// (token, zero h) at pos (llama.cpp's position 0).
  std::uint32_t StepZero(std::uint32_t token, std::uint32_t pos, bool logits);
  /// (token, the previous step's h_out) at pos: a chain step.
  std::uint32_t StepFeedback(std::uint32_t token, std::uint32_t pos, bool logits);

  /// The last step's h_out, logits (valid after a step that asked for them) and its input row h (after
  /// output_norm for a target step; zeros or the previous h_out otherwise). Copies to the host.
  [[nodiscard]] std::vector<float> CopyHidden() const;
  [[nodiscard]] std::vector<float> CopyLogits() const;
  [[nodiscard]] std::vector<float> CopyInput() const;
  /// The last logits step's argmax, best and second-best logit (the lower id first on a tie, as the GPU argmax), for
  /// the value-aware draft depth (draft_round.hpp). Reads the row through a pinned host buffer.
  struct Top2 {
    std::uint32_t id{0};
    float best{0.0F}, second{0.0F};
  };
  [[nodiscard]] Top2 LastTop2();

  [[nodiscard]] std::uint32_t block() const noexcept { return block_; }
  [[nodiscard]] std::size_t weight_bytes() const noexcept { return weight_bytes_; }
  [[nodiscard]] std::size_t buffer_bytes() const noexcept { return buffer_bytes_; }
  [[nodiscard]] double load_s() const noexcept { return load_s_; }
  [[nodiscard]] bool own_embedding() const noexcept { return own_embedding_; }
  [[nodiscard]] const gufo::models::QwenLayerWeights& layer() const noexcept { return layer_; }

  /// Coarse scoring counters (all zero when it is off).
  struct CoarseStats {
    std::uint64_t steps{0}, fallbacks{0}, checked{0};
    std::uint64_t wrong_ids{0};         // checked steps whose draft id differs from the exact head's argmax (must be 0)
    std::uint64_t miss_top1{0};         // checked steps whose exact top-1 is outside the kept blocks
    std::uint64_t miss_top2{0};         // ... exact top-1 or top-2 outside the kept blocks
    std::uint64_t unsafe{0};            // checked steps with no fallback and a top-1 or top-2 miss
    std::uint64_t rescore_mismatch{0};  // kept logits whose bits differ from the exact head's
    double max_z{0.0};                  // max over checked steps and tokens outside the kept blocks of |coarse - exact| / (err_rms ||x||)
    double build_s{0.0};
  };
  [[nodiscard]] bool coarse_on() const noexcept { return coarse_ != nullptr; }
  [[nodiscard]] const CoarseStats& coarse_stats() const noexcept { return cstats_; }
  [[nodiscard]] std::string CoarseStatsJson() const;

 private:
  MtpHead() = default;
  std::uint32_t Run(std::uint32_t token, const float* h_normed, std::uint32_t pos, bool logits);
  std::uint32_t HeadExact();
  std::uint32_t HeadCoarse();

  gufo::core::ModelConfig config_;
  std::uint32_t block_{0};
  std::uint32_t max_context_{0};
  gufo::models::QwenLayerWeights layer_;   // device pointers
  gufo::models::QwenTensorRef enorm_, hnorm_, eh_proj_, shared_head_norm_, embed_, head_, output_norm_;
  bool own_embedding_{false};
  void* d_weights_{nullptr};
  std::size_t weight_bytes_{0};
  std::size_t buffer_bytes_{0};
  double load_s_{0.0};
  hipStream_t stream_{nullptr};
  std::vector<void*> buffers_;
  float* d_h_{nullptr};          // the step's input row h (normed)
  float* d_embedding_{nullptr};
  float* d_fusion_{nullptr};     // [2H]
  float* d_hidden_{nullptr};
  float* d_normed_{nullptr};
  float* d_qg_{nullptr};
  float* d_q_{nullptr};
  float* d_k_{nullptr};
  float* d_v_{nullptr};
  float* d_gate_{nullptr};
  float* d_context_{nullptr};
  float* d_attn_out_{nullptr};
  float* d_ffn_out_{nullptr};
  float* d_moe_a_{nullptr};
  float* d_moe_b_{nullptr};
  float* d_moe_c_{nullptr};
  std::size_t moe_floats_{0};
  float* d_feedback_{nullptr};   // h_out
  float* d_logits_{nullptr};
  float* h_logits_{nullptr};     // pinned host copy of the logits row (LastTop2)
  float* d_kv_{nullptr};
  void* d_kv_f16_{nullptr};
  float* d_split_k_{nullptr};
  std::uint32_t* d_token_{nullptr};
  // coarse scoring (FLERP_MTP_COARSE)
  std::unique_ptr<flerp::draft_coarse::Coarse> coarse_;
  flerp::draft_coarse::Result* d_cres_{nullptr};
  flerp::draft_coarse::Result* h_cres_{nullptr};   // pinned
  bool coarse_check_{false};
  float* d_check_logits_{nullptr};
  CoarseStats cstats_;
};

}  // namespace flerp::qwen35moe

#endif  // defined(ENGINE_ENABLE_HIP)
#endif  // FLERP_QWEN35MOE_MTP_HEAD_HPP_
