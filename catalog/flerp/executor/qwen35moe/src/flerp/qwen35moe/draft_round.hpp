// SPDX-License-Identifier: MIT
// executor.flerp.qwen35moe: the last stage's side of speculative decoding with the file's own MTP head (ORNITH-O3 step
// two, gate/ornith-o3-spec.toml; survey option B). The drafts are verified as back-to-back single-row decode steps
// (stage.hpp SetRecurrentRing keeps the recurrent state of every row); this class decides what the head drafts and
// which of a round's rows are kept. It runs no target forward itself: the caller runs the stage and reports each row.
//
//   prompt   PromptRows(start, n, ids, rows): the head's KV pairs (h_p, x_{p+1}) at p + 1 for the chunk's rows, and
//            (zero h, x_0) at 0, as llama.cpp's draft-mtp and the O3 probe pair them (mtp_head.hpp). A chunk's last
//            row waits for the next chunk's first id, or for the token the target draws after the prompt.
//   frontier Frontier(f, x_f, row f - 1): the pair at f with logits = d_1, then chain steps d_2.. (each fed the
//            previous step's h_out and draft). How many: config.max_draft, fewer near the context's end, and with
//            value_aware the rule of Crucible draft_value.h (catalog/crucible/runtime/speculation): draft d + 1 is
//            computed while Q_d * p_next * T(d) > E(d) * (T(d + 1) - T(d)) and kept while
//            Q_d * p * T(d) > E(d) * row_ms, T(d) = base_ms + d * (row_ms + mtp_ms) (option B: a draft costs one more
//            row through the slowest stage and one MTP step), p from the head's top-1 / top-2 logit margin bucket,
//            calibrated by the conversation's own outcomes (DraftCalibration, Crucible's prior and decay).
//   round    BeginRound(p, k' + 1): frames j = 0..k' at positions p + j carry the pending token, then d_1..d_k'.
//            Frame(j, input, drawn, row) after the target ran frame j: drawn is its greedy id (GPU argmax, lower id
//            on a tie). While d_{j+1} == drawn the draft is kept and the head's pair (row, d_{j+1}) is written at
//            p + j + 1 ("refill": the probe's pairing, with the target's row in place of the chain's h_out). On the
//            first mismatch, or after frame k', the round's ids are d_1..d_n, drawn_n and the next frontier is
//            drafted at once (f = p + n + 1). NeedsTarget() is false for frames after that: their rows are
//            rejected whatever the target says, so the stage may skip them (Skipped).
//   lookup   (ORNITH-CRUCIBLE item 4, gate/ornith-crucible.toml [item4]; DraftConfig.lookup) prompt-lookup drafts, gufo
//            speculative/prompt_lookup.hpp's recipe: at a frontier, the last 3 committed ids are looked up among the
//            earlier committed ids (prompt and answer, the 16 most recent occurrences); the occurrence whose preceding
//            ids match furthest back wins (at most 64; counted only over the answer's span), and the ids that followed
//            it are the drafts, up to max_draft. A match of 12 ids is needed, or 5 when it continues where the last
//            mostly-kept copy left off (within 8 before .. 512 after). A frontier with such a match drafts by lookup
//            (the head still writes its pair at f, without logits, so later MTP drafts see a whole KV); otherwise the
//            MTP head drafts as above, at most mtp_max. No model cost; the round and the verify are unchanged.
// Greedy only. Exact: the kept ids are the target's own argmax of each kept row.
#ifndef FLERP_QWEN35MOE_DRAFT_ROUND_HPP_
#define FLERP_QWEN35MOE_DRAFT_ROUND_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include "flerp/qwen35moe/mtp_head.hpp"

namespace flerp::qwen35moe {

struct DraftCosts {
  float base_ms{75.2F};   // a round with no draft (one decode step, relay included)
  float row_ms{19.68F};   // one more back-to-back row (the slowest stage's step)
  float mtp_ms{4.83F};    // one MTP draft step
  [[nodiscard]] float StepMs(int drafts) const { return base_ms + static_cast<float>(drafts) * (row_ms + mtp_ms); }
};

struct DraftConfig {
  std::uint32_t max_draft{2};
  bool value_aware{false};
  bool lookup{false};          // prompt-lookup drafts first (item 4)
  std::uint32_t mtp_max{0};    // with lookup: the most MTP drafts a round (0: max_draft)
  DraftCosts costs;
};

/// Crucible draft_value.h DraftCalibration: acceptance by margin bucket, a prior plus decayed observations.
class DraftCalibration {
 public:
  static constexpr int kBuckets = 8;
  static int Bucket(float margin);
  [[nodiscard]] float P(int bucket) const;
  [[nodiscard]] float Mean() const;
  void Observe(const std::vector<int>& buckets, int proposed, int accepted);

 private:
  std::array<float, kBuckets> hits_{}, trials_{};
  float all_hits_{0.0F}, all_trials_{0.0F};
};

struct DraftStats {
  std::uint64_t rounds{0}, frames{0}, skipped{0}, proposed{0}, accepted{0};
  std::uint64_t draft_steps{0}, refill_steps{0}, prompt_steps{0};
  double draft_ms{0.0}, refill_ms{0.0}, prompt_ms{0.0}, top2_ms{0.0};
  std::vector<std::uint32_t> depths;   // k' chosen a round (value-aware or fixed)
  std::uint64_t frontiers{0}, lookup_rounds{0}, lookup_proposed{0}, lookup_accepted{0};
  std::uint64_t mtp_rounds{0}, mtp_proposed{0}, mtp_accepted{0};
};

class DraftRound {
 public:
  DraftRound(MtpHead& head, std::size_t hidden, std::uint32_t context);
  ~DraftRound();
  DraftRound(const DraftRound&) = delete;
  DraftRound& operator=(const DraftRound&) = delete;

  /// A new conversation (the head's KV is rewritten from position 0); resets the calibration and the stats.
  void Begin(const DraftConfig& config);
  void PromptRows(std::uint32_t start, std::size_t rows, const std::uint32_t* ids, const float* d_rows);
  void Frontier(std::uint32_t f, std::uint32_t token, const float* d_row);
  [[nodiscard]] const std::vector<std::uint32_t>& drafts() const noexcept { return drafts_; }

  void BeginRound(std::uint32_t start, std::uint32_t frames);
  [[nodiscard]] bool NeedsTarget() const noexcept { return !ended_; }
  void Frame(std::uint32_t j, std::uint32_t input, std::uint32_t drawn, const float* d_row);
  void Skipped(std::uint32_t j, std::uint32_t input);
  [[nodiscard]] bool RoundDone() const noexcept { return next_frame_ == frames_; }
  /// The round's kept ids (d_1..d_n, then the target's id after them); after a frontier, the token it drew.
  [[nodiscard]] const std::vector<std::uint32_t>& emitted() const noexcept { return emitted_; }
  /// The round's frames' drawn ids so far, in frame order (0xffffffff for a skipped frame).
  [[nodiscard]] const std::vector<std::uint32_t>& drawn() const noexcept { return drawn_; }
  [[nodiscard]] const DraftStats& stats() const noexcept { return stats_; }
  [[nodiscard]] std::string StatsJson() const;

 private:
  void CheckInput(std::uint32_t j, std::uint32_t input) const;
  void Commit(std::uint32_t pos, std::uint32_t id);
  /// Prompt-lookup drafts for the frontier at f (history_[0..f] committed): up to `limit` ids, *start = the history
  /// index of the first; empty when no match is long enough.
  std::vector<std::uint32_t> Lookup(std::uint32_t f, std::uint32_t limit, std::size_t* start);

  MtpHead& head_;
  std::size_t hidden_;
  std::uint32_t context_;
  DraftConfig config_;
  DraftCalibration cal_;
  DraftStats stats_;
  float* d_pending_row_{nullptr};   // a prompt chunk's last row, until the next id
  bool has_pending_{false};
  std::uint32_t pending_pos_{0};
  std::uint32_t pending_token_{0};  // the round's frame 0 input (the target's last drawn id)
  std::vector<std::uint32_t> drafts_;
  std::vector<int> buckets_;
  // the round in flight
  std::vector<std::uint32_t> round_drafts_;
  std::vector<int> round_buckets_;
  std::uint32_t round_start_{0}, frames_{0}, next_frame_{0}, kept_{0};
  bool ended_{false};           // the kept chain ended: later frames are rejected
  std::vector<std::uint32_t> emitted_;
  std::vector<std::uint32_t> drawn_;
  // prompt lookup
  std::vector<std::uint32_t> history_;   // committed id at each position
  static constexpr std::size_t kNone = static_cast<std::size_t>(-1);
  std::size_t request_start_{kNone};     // the first frontier's position (the answer's start)
  std::size_t resume_{kNone};            // where the last mostly-kept copy left off
  bool lookup_drafts_{false};            // the drafts held came from lookup
  std::size_t lookup_start_{0};
  bool round_lookup_{false};
  std::size_t round_lookup_start_{0};
};

}  // namespace flerp::qwen35moe

#endif  // defined(ENGINE_ENABLE_HIP)
#endif  // FLERP_QWEN35MOE_DRAFT_ROUND_HPP_
