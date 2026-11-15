// SPDX-License-Identifier: MIT
// See draft_round.hpp.
#if defined(ENGINE_ENABLE_HIP)
#include "flerp/qwen35moe/draft_round.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>

namespace flerp::qwen35moe {
namespace {

using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point a) { return std::chrono::duration<double, std::milli>(Clock::now() - a).count(); }

// Crucible draft_value.h: margin bucket edges and the prior acceptance of each bucket.
constexpr float kEdges[] = {1, 2, 3, 4, 6, 8, 12};
constexpr std::array<float, DraftCalibration::kBuckets> kPrior{0.35F, 0.43F, 0.51F, 0.75F, 0.87F, 0.94F, 0.99F, 0.99F};
constexpr float kWeight = 8.0F;
constexpr float kDecay = 0.98F;

// Crucible draft_value.h DraftValue, with option B's step cost (draft_round.hpp).
struct DraftValue {
  float q{1.0F}, e{1.0F};
  int drafts{0};
  [[nodiscard]] bool Try(const DraftCosts& c, float p_next) const {
    return q * p_next * c.StepMs(drafts) > e * (c.StepMs(drafts + 1) - c.StepMs(drafts));
  }
  [[nodiscard]] bool Keep(const DraftCosts& c, float p) const { return q * p * c.StepMs(drafts) > e * c.row_ms; }
  void Add(float p) {
    q *= p;
    e += q;
    ++drafts;
  }
};

std::string Num(double v) {
  char b[40];
  std::snprintf(b, sizeof(b), "%.6g", v);
  return b;
}

}  // namespace

int DraftCalibration::Bucket(float margin) {
  int b = 0;
  while (b < 7 && margin >= kEdges[b]) ++b;
  return b;
}

float DraftCalibration::P(int bucket) const {
  const auto b = static_cast<std::size_t>(std::clamp(bucket, 0, kBuckets - 1));
  return std::clamp((hits_[b] + kWeight * kPrior[b]) / (trials_[b] + kWeight), 0.02F, 0.995F);
}

float DraftCalibration::Mean() const { return std::clamp((all_hits_ + kWeight * 0.75F) / (all_trials_ + kWeight), 0.02F, 0.995F); }

void DraftCalibration::Observe(const std::vector<int>& buckets, int proposed, int accepted) {
  for (int k = 0; k < proposed && k <= accepted && k < static_cast<int>(buckets.size()); ++k) {
    const auto b = static_cast<std::size_t>(std::clamp(buckets[static_cast<std::size_t>(k)], 0, kBuckets - 1));
    const float hit = k < accepted ? 1.0F : 0.0F;
    hits_[b] = hits_[b] * kDecay + hit;
    trials_[b] = trials_[b] * kDecay + 1.0F;
    all_hits_ = all_hits_ * kDecay + hit;
    all_trials_ = all_trials_ * kDecay + 1.0F;
  }
}

DraftRound::DraftRound(MtpHead& head, std::size_t hidden, std::uint32_t context) : head_(head), hidden_(hidden), context_(context) {
  if (hipMalloc(reinterpret_cast<void**>(&d_pending_row_), hidden_ * sizeof(float)) != hipSuccess) {
    throw std::runtime_error("draft round: row buffer allocation failed");
  }
}

DraftRound::~DraftRound() {
  if (d_pending_row_ != nullptr) (void)hipFree(d_pending_row_);
}

void DraftRound::Begin(const DraftConfig& config) {
  config_ = config;
  cal_ = DraftCalibration();
  stats_ = DraftStats();
  has_pending_ = false;
  pending_pos_ = pending_token_ = 0;
  drafts_.clear();
  buckets_.clear();
  round_drafts_.clear();
  round_buckets_.clear();
  frames_ = next_frame_ = kept_ = round_start_ = 0;
  ended_ = false;
  emitted_.clear();
  drawn_.clear();
  history_.clear();
  request_start_ = resume_ = kNone;
  lookup_drafts_ = round_lookup_ = false;
  lookup_start_ = round_lookup_start_ = 0;
}

void DraftRound::Commit(std::uint32_t pos, std::uint32_t id) {
  if (history_.size() <= pos) history_.resize(static_cast<std::size_t>(pos) + 1, 0xffffffffU);
  history_[pos] = id;
}

std::vector<std::uint32_t> DraftRound::Lookup(std::uint32_t f, std::uint32_t limit, std::size_t* start) {
  constexpr std::size_t kKey = 3, kMinMatch = 12, kMaxMatch = 64, kMaxCandidates = 16, kResumeMin = 5, kResumeWindow = 512,
                        kResumeBack = 8;
  std::vector<std::uint32_t> out;
  const std::size_t n = static_cast<std::size_t>(f) + 1;
  if (request_start_ == kNone) {   // a request's first speculative step copies nothing
    request_start_ = f;
    return out;
  }
  if (limit == 0 || n < kKey + 1 || history_.size() < n) return out;
  const std::uint32_t* c = history_.data();
  const std::size_t span = n - request_start_;   // the answer's span bounds the query match
  std::size_t best_len = 0, best_i = 0, found = 0;
  for (std::size_t i = n - kKey - 1;; --i) {   // occurrences [i, i + 3) with a continuation at i + 3 <= n - 1
    if (c[i] == c[n - 3] && c[i + 1] == c[n - 2] && c[i + 2] == c[n - 1]) {
      std::size_t len = kKey;
      while (len < kMaxMatch && len < span && i >= len - kKey + 1 && c[i - 1 - (len - kKey)] == c[n - 4 - (len - kKey)]) ++len;
      if (len > best_len) best_len = len, best_i = i;
      if (++found == kMaxCandidates) break;
    }
    if (i == 0) break;
  }
  if (found == 0) return out;
  const std::size_t s = best_i + kKey;
  const bool resumed = resume_ != kNone && s + kResumeBack >= resume_ && s <= resume_ + kResumeWindow;
  if (best_len < (resumed ? kResumeMin : kMinMatch)) return out;
  for (std::size_t k = s; k < n && out.size() < limit; ++k) out.push_back(c[k]);
  *start = s;
  return out;
}

void DraftRound::PromptRows(std::uint32_t start, std::size_t rows, const std::uint32_t* ids, const float* d_rows) {
  if (rows == 0) return;
  if (hipDeviceSynchronize() != hipSuccess) throw std::runtime_error("draft round: device synchronize failed");
  const auto t0 = Clock::now();
  for (std::size_t i = 0; i < rows; ++i) Commit(start + static_cast<std::uint32_t>(i), ids[i]);
  for (std::size_t i = 0; i < rows; ++i) {
    const std::uint32_t p = start + static_cast<std::uint32_t>(i);
    if (p == 0) {
      (void)head_.StepZero(ids[0], 0, false);
      ++stats_.prompt_steps;
    }
    if (i == 0 && has_pending_) {
      if (pending_pos_ + 1 != p) throw std::logic_error("draft round: prompt chunks out of order");
      (void)head_.StepTarget(ids[0], d_pending_row_, p, false);
      ++stats_.prompt_steps;
      has_pending_ = false;
    }
    if (i + 1 < rows) {
      (void)head_.StepTarget(ids[i + 1], d_rows + i * hidden_, p + 1, false);
      ++stats_.prompt_steps;
    }
  }
  if (hipMemcpy(d_pending_row_, d_rows + (rows - 1) * hidden_, hidden_ * sizeof(float), hipMemcpyDeviceToDevice) != hipSuccess) {
    throw std::runtime_error("draft round: row copy failed");
  }
  has_pending_ = true;
  pending_pos_ = start + static_cast<std::uint32_t>(rows) - 1;
  stats_.prompt_ms += Ms(t0);
}

void DraftRound::Frontier(std::uint32_t f, std::uint32_t token, const float* d_row) {
  if (has_pending_) {
    if (pending_pos_ + 1 != f) throw std::logic_error("draft round: the frontier does not follow the prompt");
    has_pending_ = false;   // d_row is that same row
  }
  pending_token_ = token;
  emitted_.assign(1, token);
  drafts_.clear();
  buckets_.clear();
  lookup_drafts_ = false;
  Commit(f, token);
  if (f >= context_) return;   // no position left for the head (or for a round)
  const auto t0 = Clock::now();
  ++stats_.frontiers;
  // the next round runs frames at f .. f + k': at most context - 1 - f drafts
  std::uint32_t limit = std::min<std::uint32_t>(config_.max_draft, context_ - 1 - f);
  if (config_.lookup) {
    std::size_t start = 0;
    auto copy = Lookup(f, limit, &start);
    if (!copy.empty()) {   // drafts by lookup; the head writes its pair at f without logits
      (void)head_.StepTarget(token, d_row, f, false);
      ++stats_.draft_steps;
      drafts_ = std::move(copy);
      lookup_drafts_ = true;
      lookup_start_ = start;
      stats_.draft_ms += Ms(t0);
      return;
    }
    if (config_.mtp_max > 0) limit = std::min(limit, config_.mtp_max);
  }
  DraftValue v;
  const bool first = limit > 0 && (!config_.value_aware || v.Try(config_.costs, cal_.Mean()));
  std::uint32_t d = head_.StepTarget(token, d_row, f, first);
  ++stats_.draft_steps;
  for (std::uint32_t i = 0; first; ++i) {
    int bucket = 0;
    float p = 1.0F;
    if (config_.value_aware) {
      const auto t1 = Clock::now();
      const auto t2 = head_.LastTop2();
      stats_.top2_ms += Ms(t1);
      if (t2.id != d) throw std::logic_error("draft round: the head's host and device argmax differ");
      bucket = DraftCalibration::Bucket(t2.best - t2.second);
      p = cal_.P(bucket);
      if (!v.Keep(config_.costs, p)) break;
    }
    v.Add(p);
    drafts_.push_back(d);
    buckets_.push_back(bucket);
    if (i + 1 >= limit) break;
    if (config_.value_aware && !v.Try(config_.costs, cal_.Mean())) break;
    d = head_.StepFeedback(d, f + i + 1, true);
    ++stats_.draft_steps;
  }
  stats_.draft_ms += Ms(t0);
}

void DraftRound::BeginRound(std::uint32_t start, std::uint32_t frames) {
  if (frames == 0 || frames - 1 > drafts_.size()) {
    throw std::logic_error("draft round: a round of " + std::to_string(frames) + " frames with " + std::to_string(drafts_.size()) +
                           " drafts held");
  }
  if (!RoundDone()) throw std::logic_error("draft round: a round began before the last one ended");
  round_drafts_.assign(drafts_.begin(), drafts_.begin() + (frames - 1));
  round_buckets_.assign(buckets_.begin(), buckets_.begin() + std::min<std::size_t>(buckets_.size(), frames - 1));
  round_start_ = start;
  round_lookup_ = lookup_drafts_;
  round_lookup_start_ = lookup_start_;
  if (round_lookup_) {
    ++stats_.lookup_rounds;
    stats_.lookup_proposed += frames - 1;
  } else if (frames > 1) {
    ++stats_.mtp_rounds;
    stats_.mtp_proposed += frames - 1;
  }
  frames_ = frames;
  next_frame_ = 0;
  kept_ = 0;
  ended_ = false;
  emitted_.clear();
  drawn_.clear();
  ++stats_.rounds;
  stats_.proposed += frames - 1;
  stats_.depths.push_back(frames - 1);
}

void DraftRound::CheckInput(std::uint32_t j, std::uint32_t input) const {
  if (j != next_frame_ || j >= frames_) {
    throw std::logic_error("draft round: frame " + std::to_string(j) + " out of order (next " + std::to_string(next_frame_) + " of " +
                           std::to_string(frames_) + ")");
  }
  const std::uint32_t want = j == 0 ? pending_token_ : round_drafts_[j - 1];
  if (input != want) {
    throw std::logic_error("draft round: frame " + std::to_string(j) + " carries id " + std::to_string(input) + ", the round's is " +
                           std::to_string(want));
  }
}

void DraftRound::Frame(std::uint32_t j, std::uint32_t input, std::uint32_t drawn, const float* d_row) {
  CheckInput(j, input);
  ++next_frame_;
  ++stats_.frames;
  drawn_.push_back(drawn);
  if (ended_) return;   // a frame after the kept chain ended, run anyway (the stage did not skip it)
  const std::uint32_t k = frames_ - 1;
  if (j < k && round_drafts_[j] == drawn) {
    // draft j + 1 is kept: the head's pair at its position, from the target's row
    kept_ = j + 1;
    Commit(round_start_ + j + 1, round_drafts_[j]);
    const auto t0 = Clock::now();
    (void)head_.StepTarget(round_drafts_[j], d_row, round_start_ + j + 1, false);
    ++stats_.refill_steps;
    stats_.refill_ms += Ms(t0);
    return;
  }
  ended_ = true;
  stats_.accepted += kept_;
  if (round_lookup_) {
    stats_.lookup_accepted += kept_;
    resume_ = 2 * kept_ < k ? kNone : round_lookup_start_ + kept_;   // gufo PromptLookup::Observe
  } else {
    stats_.mtp_accepted += kept_;
  }
  cal_.Observe(round_buckets_, static_cast<int>(k), static_cast<int>(kept_));
  std::vector<std::uint32_t> ids(round_drafts_.begin(), round_drafts_.begin() + kept_);
  ids.push_back(drawn);
  Frontier(round_start_ + j + 1, drawn, d_row);
  emitted_ = std::move(ids);
}

void DraftRound::Skipped(std::uint32_t j, std::uint32_t input) {
  CheckInput(j, input);
  if (!ended_) throw std::logic_error("draft round: frame " + std::to_string(j) + " skipped before the chain ended");
  ++next_frame_;
  ++stats_.frames;
  ++stats_.skipped;
  drawn_.push_back(0xffffffffU);
}

std::string DraftRound::StatsJson() const {
  std::string depths;
  // a histogram of k' (index = depth)
  std::vector<std::uint64_t> h;
  for (const auto d : stats_.depths) {
    if (h.size() <= d) h.resize(d + 1, 0);
    ++h[d];
  }
  for (std::size_t i = 0; i < h.size(); ++i) depths += (i ? "," : "") + std::to_string(h[i]);
  return "{\"rounds\":" + std::to_string(stats_.rounds) + ",\"frames\":" + std::to_string(stats_.frames) + ",\"skipped\":" +
         std::to_string(stats_.skipped) + ",\"proposed\":" + std::to_string(stats_.proposed) + ",\"accepted\":" +
         std::to_string(stats_.accepted) + ",\"draft_steps\":" + std::to_string(stats_.draft_steps) + ",\"draft_ms\":" + Num(stats_.draft_ms) +
         ",\"refill_steps\":" + std::to_string(stats_.refill_steps) + ",\"refill_ms\":" + Num(stats_.refill_ms) + ",\"prompt_steps\":" +
         std::to_string(stats_.prompt_steps) + ",\"prompt_ms\":" + Num(stats_.prompt_ms) + ",\"top2_ms\":" + Num(stats_.top2_ms) +
         ",\"depth_histogram\":[" + depths + "],\"max_draft\":" + std::to_string(config_.max_draft) + ",\"value_aware\":" +
         (config_.value_aware ? "true" : "false") + ",\"lookup\":" + (config_.lookup ? "true" : "false") + ",\"mtp_max\":" +
         std::to_string(config_.mtp_max) + ",\"frontiers\":" + std::to_string(stats_.frontiers) + ",\"lookup_rounds\":" +
         std::to_string(stats_.lookup_rounds) + ",\"lookup_proposed\":" + std::to_string(stats_.lookup_proposed) +
         ",\"lookup_accepted\":" + std::to_string(stats_.lookup_accepted) + ",\"mtp_rounds\":" + std::to_string(stats_.mtp_rounds) +
         ",\"mtp_proposed\":" + std::to_string(stats_.mtp_proposed) + ",\"mtp_accepted\":" + std::to_string(stats_.mtp_accepted) + "}";
}

}  // namespace flerp::qwen35moe
#endif  // defined(ENGINE_ENABLE_HIP)
