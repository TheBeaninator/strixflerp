// SPDX-License-Identifier: MIT
// executor.flerp.qwen35moe_abi: the model as N stages joined by runtime.flerp.stage_relay, as rank 0 sees it (the
// Pipeline behind include/flerp/executor.h) and as ranks 1..N-1 run it (StageMain, the loop of
// tool.flerp.qwen35moe_stage). docs/PHASE5-FOURNODE-DESIGN.md section 3.9; the stage mechanics are stage P2's
// (tool.flerp.qwen35moe_generate, stage_driver.cpp), written again here as a library with a mirrored-call loop.
//
// Options are the flerp_open keys, the same strings for rank 0 (executor.KEY=VALUE of the front end) and for a
// stage process (KEY=VALUE arguments of flerp-qwen35moe-stage):
//   weights            the GGUF file (required)
//   context            positions of the one seq and of every stage's arena (default 4096)
//   world              stages (default 1)
//   rank               this process's stage; flerp_open takes 0 only, the stage tool 1..world-1
//   peers              world > 1: "loopback" (every stage in this process, views of one load) or
//                      "HOST[,HOST...]:PORT" (TCP; rank r listens on HOST_r:PORT + r; one HOST serves every rank)
//   placement          "even" (the stage plan's byte-balanced cuts, design 3.1) or "c0,c1,..,cW" (block cuts)
//   epoch              the relay's epoch, equal on every rank (default 1)
//   deadline_ms        no progress on a link inside a call within this fails the call (default 60000)
//   setup_deadline_ms  rank 0 waits this long at flerp_open for the stages to connect (default 1800000)
//   graph              on | off: the captured decode graph (default on, the origin's)
//   single_reader      on | off: the placed loader's single O_DIRECT reader (default on: the pod's I/O rule)
//   pause_ms           the single reader's pause between tensors (default 100)
//   data_file          read the bytes from this same-size copy instead (loom-swarm's sparse rank file)
//   swarm_helper, swarm_manifest, swarm_from (A:P[,A:P..]), swarm_conns
//                      the bytes from loom-swarm over the network straight into the device (placed_loader.hpp)
//   drafts             rank 0, TCP: the most drafts a round the last stage's MTP head gives a seq that asks for drafts
//                      (caps.max_draft; 0, the default: no drafts). ORNITH-O3 step two, gate/ornith-o3-spec.toml
//   draft_value        on | off, rank 0 with drafts: the value-aware draft depth (Crucible draft_value.h, draft_round.hpp)
//                      for every drafting session, so the last stage holds at most `drafts` a round, fewer when the rule
//                      says so, and flerp_draft_hold / flerp_verify take what it holds (card 3f6d20b703, default off)
//   draft_head         on | off, the last stage process: load the file's MTP block beside the stage (default off)
//   draft_costs        the last stage process: the value-aware rule's costs, base_ms,row_ms,mtp_ms (draft_round.hpp
//                      DraftCosts; default Ornith's 75.2,19.68,4.83)
//   mtp_data_file      the head's bytes from this same-size copy (default data_file, else weights)
//
// Every process loads through format.flerp.gguf_placed: world 1 and the loopback load the whole model once (the
// loopback's stages are views of it with their block ranges); a TCP rank loads only its own blocks (a world-1
// range plan). A process writes one JSON line a step to stdout: {"event": "loaded" | "waiting" | "connected" |
// "closed" | "terminated" | "relay_failed", ...}.
#ifndef FLERP_QWEN35MOE_ABI_STAGES_HPP_
#define FLERP_QWEN35MOE_ABI_STAGES_HPP_

#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace flerp::stage_relay {
struct Frame;
}

namespace flerp::qwen35moe_abi {

struct Options {
  std::string weights;
  std::uint32_t context{4096};
  std::uint32_t world{1};
  std::uint32_t rank{0};
  std::string peers;
  std::string placement{"even"};
  std::uint64_t epoch{1};
  std::uint32_t deadline_ms{60000};
  std::uint32_t setup_deadline_ms{1800000};
  bool graph{true};
  bool single_reader{true};
  std::uint32_t pause_ms{100};
  std::string data_file;
  std::string swarm_helper, swarm_manifest, swarm_from;
  std::uint32_t swarm_conns{4};
  std::uint32_t drafts{0};
  bool draft_head{false};
  bool draft_value{false};
  std::string draft_costs;
  std::string mtp_data_file;
};

/// One key: false (and *error) for an unknown key or a bad value. Never ignores a key.
bool SetKey(Options& o, const std::string& key, const std::string& value, std::string* error);
/// The keys together: "" when they make sense for `rank` (0: the adapter; > 0: a stage process).
std::string CheckOptions(const Options& o);

/// One line of JSON to stdout, flushed.
void Emit(const std::string& json);

/// What rank 0 gets back from one call: the token the last stage drew (its argmax) and, when asked, the full
/// logits row of the call's last position.
struct Reply {
  std::int32_t token{0};
  std::vector<float> logits;
};

/// A drafting session's round (Pipeline::Verify): the ids the last stage kept (its drafts while each equalled the
/// target's argmax of the row before, then the target's own id), every frame's drawn id, the next drafts it holds.
struct Round {
  std::vector<std::int32_t> kept, drawn, drafts;
};

class Pipeline {
 public:
  /// Loads rank 0's part (world 1: everything), opens the relay and waits for the stages (TCP), or builds the
  /// other stages in this process (loopback). Throws std::exception with the reason.
  explicit Pipeline(const Options& o);
  ~Pipeline();
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  /// One forward: `ids` at `start`. decode: a decode step (one id, the captured graph); else a prompt chunk of
  /// at most max_rows() ids. want: the last row's token is needed; full: its logits row too. Throws on failure,
  /// after which failed() is true and only Close() / the destructor may follow.
  Reply Forward(bool decode, std::span<const std::int32_t> ids, std::uint32_t start, bool want, bool full);
  /// Every stage back to position 0 (mirrored RESET), the decode graph switch with it. drafts > 0: a drafting
  /// session (TCP, opt.drafts >= drafts): prompt chunks carry their ids, every reply carries the last stage's next
  /// drafts (held()), and a decode step is a round of one frame. Throws on failure.
  void Reset(std::uint32_t drafts = 0);
  /// A drafting session: the pending id and the first `held` drafts the last stage holds, at start.. as back-to-back
  /// single-row decode steps (every frame run, so every frame's id comes back). Throws on failure.
  Round Verify(std::int32_t pending, std::uint32_t held, std::uint32_t start);
  /// The drafts the last stage holds for the session's pending id (after a prompt reply, a decode step or a verify).
  [[nodiscard]] const std::vector<std::int32_t>& held() const noexcept { return held_; }
  [[nodiscard]] std::uint32_t session_drafts() const noexcept { return session_drafts_; }
  /// CLOSE to every stage when the relay is still sound; never throws. The destructor calls it.
  void Close() noexcept;

  [[nodiscard]] bool failed() const noexcept { return failed_; }
  [[nodiscard]] const std::string& reason() const noexcept { return reason_; }
  [[nodiscard]] std::size_t max_rows() const noexcept { return max_rows_; }
  [[nodiscard]] std::uint32_t vocab() const noexcept { return vocab_; }
  [[nodiscard]] std::uint32_t context() const noexcept { return opt_.context; }
  [[nodiscard]] const std::string& describe() const noexcept { return describe_; }
  [[nodiscard]] const std::string& fingerprint() const noexcept { return fingerprint_; }
  [[nodiscard]] std::int64_t device_bytes() const noexcept { return device_bytes_; }

 private:
  struct Impl;
  [[noreturn]] void Fail(const std::string& why);
  void Pump();
  void TakeSpec(const flerp::stage_relay::Frame& f, Round* round);
  flerp::stage_relay::Frame SendRound(std::int32_t pending, std::uint32_t held, std::uint32_t start, const std::vector<float>& rows0);

  Options opt_;
  std::unique_ptr<Impl> impl_;
  std::size_t max_rows_{0};
  std::uint32_t vocab_{0};
  std::string describe_, fingerprint_, reason_;
  std::int64_t device_bytes_{-1};
  bool failed_{false}, closed_{false};
  std::uint32_t session_drafts_{0};
  std::vector<std::int32_t> held_;
};

/// A stage process (rank > 0, TCP): load the rank's blocks, connect, execute rank 0's mirrored calls until CLOSE.
/// `stop` is set by the caller's SIGTERM / SIGINT handler, which also calls InterruptStage(). Returns the exit
/// status: 0 after CLOSE or a stop, 3 when the relay failed otherwise, 1 when the load or the setup failed.
int StageMain(const Options& o, const std::atomic<bool>& stop);
/// Async-signal-safe: makes the stage's blocked receive fail (Relay::Interrupt).
void InterruptStage() noexcept;

}  // namespace flerp::qwen35moe_abi

#endif  // FLERP_QWEN35MOE_ABI_STAGES_HPP_
