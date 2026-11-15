// SPDX-License-Identifier: MIT
// See mtp_probe.hpp.
#include "mtp_probe.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace flerp_tool {
namespace {

constexpr std::uint32_t kTop = 20;

std::string Ids(const std::vector<std::uint32_t>& v) {
  std::string o = "[";
  for (std::size_t i = 0; i < v.size(); ++i) o += (i ? "," : "") + std::to_string(v[i]);
  return o + "]";
}

void Sync() {
  if (hipDeviceSynchronize() != hipSuccess) throw std::runtime_error("mtp probe: device synchronize failed");
}

}  // namespace

MtpProbe::MtpProbe(flerp::qwen35moe::MtpHead& head, std::uint32_t depth, std::size_t hidden)
    : head_(head), depth_(depth), hidden_(hidden) {
  if (depth_ == 0) throw std::invalid_argument("mtp probe: depth 0");
}

MtpProbe::~MtpProbe() {
  if (dump_ != nullptr) std::fclose(dump_);
}

void MtpProbe::ArmDump(const std::string& path, std::uint32_t frontiers) {
  if (dump_ != nullptr) std::fclose(dump_);
  dump_ = std::fopen(path.c_str(), "wb");
  if (dump_ == nullptr) throw std::runtime_error("mtp probe: cannot write " + path);
  const char magic[4] = {'M', 'T', 'P', 'D'};
  const std::int32_t head[3] = {static_cast<std::int32_t>(hidden_), static_cast<std::int32_t>(kTop),
                                static_cast<std::int32_t>(head_.block())};
  std::fwrite(magic, 1, 4, dump_);
  std::fwrite(head, sizeof(head), 1, dump_);
  dump_frontiers_left_ = frontiers;
  dump_records_ = 0;
}

void MtpProbe::Begin(const std::vector<std::uint32_t>& prompt) {
  prompt_ = prompt;
  ids_.clear();
  drafts_.clear();
  rows_seen_ = 0;
  step_ms_ = 0.0;
  steps_ = 0;
}

// One dump record: int32 kind (0 zero h, 1 target row, 2 chain), token, pos, logits, argmax; float32 h_in[H],
// h_out[H]; int32 top ids[20]; float32 top values[20] (zeros without logits).
void MtpProbe::Record(std::uint32_t kind, std::uint32_t token, std::uint32_t pos, bool logits, std::uint32_t argmax) {
  if (dump_ == nullptr) return;
  const std::int32_t rec[5] = {static_cast<std::int32_t>(kind), static_cast<std::int32_t>(token), static_cast<std::int32_t>(pos),
                               logits ? 1 : 0, static_cast<std::int32_t>(argmax)};
  const auto h_in = head_.CopyInput();
  const auto h_out = head_.CopyHidden();
  std::vector<std::int32_t> top_ids(kTop, 0);
  std::vector<float> top_vals(kTop, 0.0F);
  if (logits) {
    const auto l = head_.CopyLogits();
    std::vector<std::uint32_t> idx(l.size());
    std::iota(idx.begin(), idx.end(), 0U);
    std::partial_sort(idx.begin(), idx.begin() + kTop, idx.end(), [&](std::uint32_t a, std::uint32_t b) {
      return l[a] > l[b] || (l[a] == l[b] && a < b);
    });
    for (std::uint32_t i = 0; i < kTop; ++i) {
      top_ids[i] = static_cast<std::int32_t>(idx[i]);
      top_vals[i] = l[idx[i]];
    }
  }
  std::fwrite(rec, sizeof(rec), 1, dump_);
  std::fwrite(h_in.data(), sizeof(float), h_in.size(), dump_);
  std::fwrite(h_out.data(), sizeof(float), h_out.size(), dump_);
  std::fwrite(top_ids.data(), sizeof(std::int32_t), kTop, dump_);
  std::fwrite(top_vals.data(), sizeof(float), kTop, dump_);
  ++dump_records_;
}

void MtpProbe::PromptRows(std::uint32_t start, std::size_t rows, const float* d_rows) {
  Sync();
  const auto t0 = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < rows; ++i) {
    const std::uint32_t p = start + static_cast<std::uint32_t>(i);
    if (p == 0) {
      (void)head_.StepZero(prompt_.empty() ? 0 : prompt_[0], 0, false);
      ++steps_;
      Record(0, prompt_.empty() ? 0 : prompt_[0], 0, false, 0);
    }
    if (p + 1 < prompt_.size()) {
      (void)head_.StepTarget(prompt_[p + 1], d_rows + i * hidden_, p + 1, false);
      ++steps_;
      Record(1, prompt_[p + 1], p + 1, false, 0);
    }
  }
  rows_seen_ += rows;
  step_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void MtpProbe::Frontier(std::uint32_t f, std::uint32_t token, const float* d_row) {
  Sync();
  const auto t0 = std::chrono::steady_clock::now();
  ids_.push_back(token);
  std::vector<std::uint32_t> d(depth_);
  d[0] = head_.StepTarget(token, d_row, f, true);
  Record(1, token, f, true, d[0]);
  for (std::uint32_t k = 1; k < depth_; ++k) {
    d[k] = head_.StepFeedback(d[k - 1], f + k, true);
    Record(2, d[k - 1], f + k, true, d[k]);
  }
  steps_ += depth_;
  drafts_.push_back(std::move(d));
  step_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (dump_ != nullptr && dump_frontiers_left_ > 0 && --dump_frontiers_left_ == 0) {
    std::fclose(dump_);
    dump_ = nullptr;
  }
}

std::string MtpProbe::Json(std::size_t prompt_index, const std::string& extra) const {
  // quick tally: chain hits at each depth over frontiers whose truth is known
  std::string hits = "[", known = "[";
  for (std::uint32_t k = 1; k <= depth_; ++k) {
    std::size_t n = 0, h = 0;
    for (std::size_t j = 0; j < drafts_.size(); ++j) {
      if (j + k >= ids_.size()) break;
      ++n;
      bool ok = true;
      for (std::uint32_t i = 1; i <= k && ok; ++i) ok = drafts_[j][i - 1] == ids_[j + i];
      h += ok ? 1 : 0;
    }
    hits += (k > 1 ? "," : "") + std::to_string(h);
    known += (k > 1 ? "," : "") + std::to_string(n);
  }
  std::string drafts = "[";
  for (std::size_t j = 0; j < drafts_.size(); ++j) drafts += (j ? "," : "") + Ids(drafts_[j]);
  drafts += "]";
  char ms[64];
  std::snprintf(ms, sizeof(ms), "%.4f", steps_ ? step_ms_ / static_cast<double>(steps_) : 0.0);
  return "{\"event\":\"mtp_probe\",\"prompt\":" + std::to_string(prompt_index) + ",\"depth\":" + std::to_string(depth_) +
         ",\"block\":" + std::to_string(head_.block()) + ",\"prompt_tokens\":" + std::to_string(prompt_.size()) +
         ",\"prompt_rows_seen\":" + std::to_string(rows_seen_) + ",\"ids\":" + Ids(ids_) + ",\"drafts\":" + drafts +
         ",\"chain_hits\":" + hits + "],\"chain_known\":" + known + "],\"mtp_steps\":" + std::to_string(steps_) +
         ",\"mtp_ms_per_step\":" + ms + ",\"dump_records\":" + std::to_string(dump_records_) + extra + "}";
}

StagedMtpProbe::StagedMtpProbe(flerp::qwen35moe::MtpHead& head, std::uint32_t depth, std::size_t hidden,
                               std::vector<std::vector<std::uint32_t>> prompts, std::uint32_t rank)
    : probe_(head, depth, hidden), hidden_(hidden), prompts_(std::move(prompts)), rank_(rank) {
  if (prompts_.empty()) throw std::invalid_argument("mtp probe: the last rank needs the ids files (--ids-file)");
}

void StagedMtpProbe::Flush() {
  if (!open_) return;
  const std::size_t index = (session_ - 1) % prompts_.size();
  std::printf("%s\n", probe_.Json(index, ",\"rank\":" + std::to_string(rank_) + ",\"session\":" + std::to_string(session_ - 1)).c_str());
  std::fflush(stdout);
  open_ = false;
}

void StagedMtpProbe::OnReset() {
  Flush();
  ++session_;
  probe_.Begin(prompts_[(session_ - 1) % prompts_.size()]);
  open_ = true;
}

void StagedMtpProbe::OnClose() { Flush(); }

void StagedMtpProbe::OnForward(bool decode, std::uint32_t start, std::size_t rows, bool want, std::uint32_t token,
                               const float* d_rows) {
  if (!open_) return;
  if (!decode) {
    probe_.PromptRows(start, rows, d_rows);
    if (want) probe_.Frontier(start + static_cast<std::uint32_t>(rows), token, d_rows + (rows - 1) * hidden_);
  } else {
    probe_.Frontier(start + 1, token, d_rows);
  }
}

}  // namespace flerp_tool
