// SPDX-License-Identifier: MIT
// The MTP acceptance probe (ORNITH-O3 step one, gate/ornith-o3-mtp.toml): during a normal greedy run, the file's MTP
// head (flerp/qwen35moe/mtp_head.hpp) drafts DEPTH tokens at every frontier, and the probe records whether draft k
// equals the greedy id that actually comes k tokens later. It only reads: the ids the run produces are the run's own.
// Off unless --mtp-probe DEPTH is given.
//
// The target's pre-norm rows come through the stage output buffer (flerp/qwen35moe/stage.hpp d_output: a prompt chunk
// writes one row a token, a decode step row 0), which changes no arithmetic of the forward.
//
//   Begin(prompt)               a new conversation: the head's KV is rewritten from position 0
//   PromptRows(start, n, rows)  prompt positions [start, start + n): pairs (h_p, x_{p+1}) for p + 1 < prompt size,
//                               and (zero, x_0) at position 0
//   Frontier(f, x_f, row f-1)   the target produced x_f from position f - 1: pair (h_{f-1}, x_f) at f with logits =
//                               draft 1, then DEPTH - 1 chain steps; drafts[f - prompt] = {d_1 .. d_DEPTH}, d_k a guess
//                               at x_{f+k}
#ifndef FLERP_QWEN35MOE_TOOL_MTP_PROBE_HPP_
#define FLERP_QWEN35MOE_TOOL_MTP_PROBE_HPP_

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "flerp/qwen35moe/mtp_head.hpp"

namespace flerp_tool {

struct MtpProbeOptions {
  std::uint32_t depth{0};             // 0: off
  std::string data_file;              // where the head's bytes come from (empty: the GGUF)
  std::string dump_path;              // unit-check dump of the first prompt (generate path only)
  std::uint32_t dump_frontiers{8};
};

class MtpProbe {
 public:
  MtpProbe(flerp::qwen35moe::MtpHead& head, std::uint32_t depth, std::size_t hidden);
  ~MtpProbe();

  void Begin(const std::vector<std::uint32_t>& prompt);
  void PromptRows(std::uint32_t start, std::size_t rows, const float* d_rows);
  void Frontier(std::uint32_t f, std::uint32_t token, const float* d_row);

  /// Arms the dump of this conversation (every step until `frontiers` frontiers have been drafted).
  void ArmDump(const std::string& path, std::uint32_t frontiers);

  /// One `mtp_probe` event: the prompt, the ids the frontiers saw, every frontier's drafts, a quick tally.
  [[nodiscard]] std::string Json(std::size_t prompt_index, const std::string& extra = "") const;
  [[nodiscard]] const std::vector<std::uint32_t>& ids() const noexcept { return ids_; }
  [[nodiscard]] std::size_t prompt_rows_seen() const noexcept { return rows_seen_; }

 private:
  void Record(std::uint32_t kind, std::uint32_t token, std::uint32_t pos, bool logits, std::uint32_t argmax);

  flerp::qwen35moe::MtpHead& head_;
  std::uint32_t depth_;
  std::size_t hidden_;
  std::vector<std::uint32_t> prompt_;
  std::vector<std::uint32_t> ids_;                    // x_N, x_{N+1}, .. as the frontiers saw them
  std::vector<std::vector<std::uint32_t>> drafts_;    // [frontier][k]
  std::size_t rows_seen_{0};
  double step_ms_{0.0};
  std::size_t steps_{0};
  std::FILE* dump_{nullptr};
  std::uint32_t dump_frontiers_left_{0};
  std::size_t dump_records_{0};
};

/// The probe on the last rank of a --staged tcp world: that rank sees hidden rows, not tokens, so it is given the
/// same ids files as rank 0 and takes conversation s (the s-th RESET) to be prompt s mod (number of files), the order
/// in which rank 0 runs them. Its own GPU argmax is the token rank 0 receives. One `mtp_probe` event a conversation,
/// printed at the next RESET or at CLOSE.
class StagedMtpProbe {
 public:
  StagedMtpProbe(flerp::qwen35moe::MtpHead& head, std::uint32_t depth, std::size_t hidden,
                 std::vector<std::vector<std::uint32_t>> prompts, std::uint32_t rank);
  void OnReset();
  void OnClose();
  void OnForward(bool decode, std::uint32_t start, std::size_t rows, bool want, std::uint32_t token, const float* d_rows);

 private:
  void Flush();
  MtpProbe probe_;
  std::size_t hidden_;
  std::vector<std::vector<std::uint32_t>> prompts_;
  std::uint32_t rank_;
  std::size_t session_{0};
  bool open_{false};
};

}  // namespace flerp_tool

#endif  // FLERP_QWEN35MOE_TOOL_MTP_PROBE_HPP_
