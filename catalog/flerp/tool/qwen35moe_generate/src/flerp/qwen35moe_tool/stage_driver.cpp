// SPDX-License-Identifier: MIT
// flerp-qwen35moe-generate --staged: the model run as N stages joined by runtime.flerp.stage_relay (stage P2 of
// docs/PHASE5-FOURNODE-DESIGN.md). Rank 0 embeds and runs its blocks, middle ranks run theirs, the last rank runs
// the output norm and head and replies with the token id and (--reply full) the logits row. Each stage is an unchanged
// executor.flerp.qwen35moe executor over a block range, with the whole model's config and arena (design 3.4, 3.7);
// a stage's hidden rows go in and out through its stage buffers (flerp/qwen35moe/stage.hpp).
//
//   --staged loopback  one process, one GPU, one load: the whole model through the placed loader, then for each of
//                      --configs "W:CUTS:GRAPH;..." (CUTS = even | c0,c1,..,cW; GRAPH = on | off) the stages as views
//                      of that one load, joined by an in-process relay ring, every prompt traced. --fault-config
//                      "W:CUTS:GRAPH" then runs the injected faults (drop, stale-epoch, replay, bad-bytes on rank 1's
//                      outgoing frames, and a wrong epoch at the handshake) and checks every rank fails.
//   --staged tcp       one rank of N processes: --world W --rank R --cuts even|c0,..,cW --port P [--host A|A0,..,AW-1] --epoch E;
//                      this rank loads only its blocks (a world-1 range plan) and joins the TCP ring (rank r listens
//                      on port + r). Rank 0 runs every prompt once per --graph-modes (or --run-specs) entry, then CLOSE. --fault K:N
//                      injects a fault on this rank's outgoing frame N.
//                      --dump F --dump-layers L,.. --dump-rows R,..: whole MoE blocks of this rank's range as the
//                      device computed them (moe_ffn.hpp MoeDump), in the prompt chunks of the first prompt of the
//                      first run only (chunks run eagerly), written to F after CLOSE, for gate/moe_block_check.py.
//   common: --model F --context N --max-tokens N --ids-file F... [--single-reader --pause-ms N] [--deadline-ms N]
//           --rows pinned|pageable (default pinned): the stage's rows go to and from the device through fixed pinned
//           host buffers (hipHostMalloc, as Crucible's runtime.crucible.device_memory pinned_buffer.h), or, pageable,
//           through a fresh std::vector each call (the P2..P5 behaviour; 1.8 ms a call on Ornith's ranks 1-3).
//           --reply full|token (rank 0, loopback; default full): full asks the last stage for the whole logits row
//           (the gates hash it); token asks for the token id only, which the last stage's executor already picks on
//           the GPU (LaunchGPUArgmax / LaunchBatchedGPUArgmax: the lowest id wins a tie, as Argmax below).
//           --prompt-subchunks S (rank 0, loopback; default 1): each prompt chunk is cut into S sub-chunks (sizes
//           differ by at most 1) sent back to back, the replies read after the last one, so the stages of a TCP
//           ring work on different sub-chunks at once (StrixFold stagger.zig; PHASE5-FOURNODE-DESIGN.md:120).
//           --run-specs G:R:S[:K],.. (tcp rank 0): the runs of every prompt, each graph on|off, reply full|token,
//           S sub-chunks, K drafts a round (0 or absent: none; N: N a round; vN: at most N, value-aware; NpL or
//           vNpL: prompt-lookup drafts first, at most L, else at most N MTP drafts, gate/ornith-crucible.toml [item4]);
//           in place of --graph-modes.
//           Speculative decoding with the file's MTP head (ORNITH-O3 step two, gate/ornith-o3-spec.toml; tcp only,
//           reply token): the last rank given --draft-head loads the head (from --mtp-data-file, else --data-file,
//           else the GGUF) and drafts at every frontier (flerp/qwen35moe/draft_round.hpp); rank 0 sends the
//           pending id and the drafts as back-to-back single-row decode frames (relay kSpecRound); the last rank
//           keeps the longest matching prefix and answers once a round (kSpecReply: the kept ids, the next drafts).
//           Every rank keeps a ring of K + 1 recurrent states (stage.hpp SetRecurrentRing). --draft-costs B,R,M
//           (last rank; the value-aware rule's ms: a round, a row, an MTP step) and --draft-skip on|off (default
//           on: the last rank does not run the target on frames after the kept chain ended).
//           where the bytes come from (format.flerp.gguf_placed placed_loader.hpp): the --model file (default);
//           --data-file F [--punch-behind] a same-size copy at the same offsets (loom-swarm's sparse rank file);
//           --swarm-helper BIN --swarm-manifest M --swarm-from A:P[,A:P..] [--swarm-conns K] loom-swarm over the
//           network straight into the device, nothing on a disk (--model then supplies only the header)
//
// Events (JSON lines on stdout): `loaded`, `staged` (one a configuration and prompt: ids, sha256 of every logits row,
// timings; with drafts its rounds), `stage_timing` (tcp ranks > 0: one a session; the last rank's `draft` stats),
// `draft_head`, `fault_injected`, `relay_failed`, `fault_case` (loopback).
#include <hip/hip_runtime.h>
#include <signal.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "flerp/gguf_placed/placement.hpp"
#include "flerp/qwen35moe/moe_ffn.hpp"
#include "flerp/qwen35moe/stage.hpp"
#include "flerp/stage_relay/relay.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/families/qwen/text/tokenizer.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "flerp/qwen35moe/draft_round.hpp"
#include "flerp/qwen35moe/mtp_head.hpp"
#include "mtp_probe.hpp"

namespace flerp_tool {
namespace {

namespace relay = flerp::stage_relay;
using gufo::tokenization::TokenId;
using Clock = std::chrono::steady_clock;

std::vector<std::string> Hosts(const std::string& text);
double Ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
double UnixNow() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string Num(double v) {
  char b[40];
  std::snprintf(b, sizeof(b), "%.6g", v);
  return b;
}
std::string Str(const std::string& s) {
  std::string o = "\"";
  for (const char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    if (static_cast<unsigned char>(c) < 0x20) continue;
    o += c;
  }
  return o + "\"";
}
template <typename R>
std::string Ids(const R& r) {
  std::string o = "[";
  bool first = true;
  for (const auto v : r) {
    o += (first ? "" : ",") + std::to_string(static_cast<long long>(v));
    first = false;
  }
  return o + "]";
}
std::string Strs(const std::vector<std::string>& v) {
  std::string o = "[";
  for (std::size_t i = 0; i < v.size(); ++i) o += (i ? "," : "") + Str(v[i]);
  return o + "]";
}

void HipOk(hipError_t s, const char* what) {
  if (s != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(s));
}

std::string Sha(const void* data, std::size_t bytes) {
  return gufo::crypto::Sha256Hex(std::span<const std::uint8_t>(static_cast<const std::uint8_t*>(data), bytes));
}

std::span<const std::uint8_t> Bytes(const std::vector<float>& v) {
  return {reinterpret_cast<const std::uint8_t*>(v.data()), v.size() * sizeof(float)};
}
std::vector<float> Floats(const std::vector<std::uint8_t>& b) {
  std::vector<float> f(b.size() / sizeof(float));
  std::memcpy(f.data(), b.data(), f.size() * sizeof(float));
  return f;
}

// --- one stage: an executor over a block range ------------------------------------------------------------------
struct ChunkTime {
  std::size_t rows{0};
  double ms{0.0};
};

struct Timing {
  std::vector<ChunkTime> chunks;   // prompt chunks: forward + synchronize
  std::size_t decode_steps{0};
  double decode_ms{0.0};
  double copy_ms{0.0};             // host <-> device copies of the stage's rows
  double logits_ms{0.0};           // copy of the logits row (last stage)
};

struct Out {
  TokenId token{0};
  std::vector<float> rows;     // the stage's output rows (not the last stage), pageable mode
  const float* row_data{nullptr};   // the output rows (pinned mode: the stage's pinned buffer, valid until its next call)
  std::size_t row_floats{0};
  std::vector<float> logits;   // the last stage, when the call asked for the full row
  [[nodiscard]] std::span<const std::uint8_t> RowBytes() const {
    return {reinterpret_cast<const std::uint8_t*>(row_data), row_floats * sizeof(float)};
  }
};

enum class RowsMode { kPinned, kPageable };

class Stage {
 public:
  Stage(std::uint32_t rank, std::uint32_t world, std::shared_ptr<const gufo::hip::QwenGpuModel> model, std::uint32_t context,
        RowsMode rows_mode = RowsMode::kPinned)
      : rank_(rank), world_(world), model_(std::move(model)), pinned_(rows_mode == RowsMode::kPinned) {
    std::string error;
    exec_ = gufo::hip::QwenGpuExecutor::Create(model_, &error, context);
    if (!exec_) throw std::runtime_error("stage " + std::to_string(rank) + ": executor: " + error);
    const auto& w = model_->GetWeights();
    begin_ = w.BlockBegin();
    end_ = w.BlockEnd();
    hidden_ = w.config.hidden_size;
    max_rows_ = exec_->GetMaxPromptBatch();
    first_ = rank == 0;
    last_ = rank + 1 == world;
    if (first_ != (begin_ == 0) || last_ != w.HasHead()) {
      throw std::runtime_error("stage " + std::to_string(rank) + ": blocks [" + std::to_string(begin_) + ", " +
                               std::to_string(end_) + ") do not fit its place in the chain");
    }
    if (world_ > 1) {
      if (!first_) HipOk(hipMalloc(&d_in_, max_rows_ * hidden_ * sizeof(float)), "stage input");
      if (!last_) HipOk(hipMalloc(&d_out_, max_rows_ * hidden_ * sizeof(float)), "stage output");
      // Pinned host rows (Crucible pinned_buffer.h): a fixed buffer the DMA reads or writes directly, so no call
      // registers or stages pageable memory (runtime_memory_policy.h: that path stops every KFD queue).
      if (pinned_ && !first_) HipOk(hipHostMalloc(reinterpret_cast<void**>(&h_in_), max_rows_ * hidden_ * sizeof(float)), "pinned stage input");
      if (pinned_ && !last_) HipOk(hipHostMalloc(reinterpret_cast<void**>(&h_out_), max_rows_ * hidden_ * sizeof(float)), "pinned stage output");
      flerp::qwen35moe::BindStageIo(&w, {.d_input = d_in_, .d_output = d_out_});
    }
  }
  ~Stage() {
    flerp::qwen35moe::ReleaseRecurrentRing(exec_.get());
    exec_.reset();
    if (world_ > 1) flerp::qwen35moe::UnbindStageIo(&model_->GetWeights());
    if (d_in_) (void)hipFree(d_in_);
    if (d_out_) (void)hipFree(d_out_);
    if (h_in_) (void)hipHostFree(h_in_);
    if (h_out_) (void)hipHostFree(h_out_);
  }
  Stage(const Stage&) = delete;
  Stage& operator=(const Stage&) = delete;

  // A prompt chunk (decode = false) or a decode step. `tokens` matter on the first stage only; `input` (rows x
  // hidden FP32, any alignment) on the others. want: the call's logits (the last row's argmax); full: the logits row too.
  Out Run(bool decode, const std::vector<TokenId>& tokens, std::uint32_t start_pos, std::size_t rows, bool want, bool full,
          const void* input) {
    Out out;
    flerp::qwen35moe::MoeDump* dump = dump_;
    const bool arm = dump != nullptr && !decode && sessions_ == 1;
    if (arm) {
      dump->armed = true;
      dump->position_base = start_pos;
    }
    if (rows == 0 || rows > max_rows_ || (decode && rows != 1)) throw std::runtime_error("stage: bad row count");
    auto t0 = Clock::now();
    if (!first_) {
      const std::size_t bytes = rows * hidden_ * sizeof(float);
      if (pinned_) {
        std::memcpy(h_in_, input, bytes);
        HipOk(hipMemcpy(d_in_, h_in_, bytes, hipMemcpyHostToDevice), "copy into the stage input");
      } else {
        HipOk(hipMemcpy(d_in_, input, bytes, hipMemcpyHostToDevice), "copy into the stage input");
      }
    }
    const std::vector<TokenId> zeros(rows, 0);
    const std::vector<TokenId>& ids = first_ ? tokens : zeros;
    auto t1 = Clock::now();
    if (decode) {
      // Logits are asked on every stage, as the single-node trace asks them: that is what makes the step eligible
      // for the captured graph; a stage without the head answers token 0 (decode_step.cpp).
      out.token = exec_->ForwardToken(ids.at(0), start_pos, true);
    } else {
      out.token = exec_->ForwardPromptBatch(std::span<const TokenId>(ids.data(), rows), start_pos, want);
    }
    HipOk(hipDeviceSynchronize(), "synchronize");
    if (arm) dump->armed = false;
    auto t2 = Clock::now();
    if (!last_) {
      out.row_floats = rows * hidden_;
      if (pinned_) {
        HipOk(hipMemcpy(h_out_, d_out_, out.row_floats * sizeof(float), hipMemcpyDeviceToHost), "copy of the stage output");
        out.row_data = h_out_;
      } else {
        out.rows.resize(out.row_floats);
        HipOk(hipMemcpy(out.rows.data(), d_out_, out.rows.size() * sizeof(float), hipMemcpyDeviceToHost), "copy of the stage output");
        out.row_data = out.rows.data();
      }
    }
    auto t3 = Clock::now();
    if (last_ && want && full) {
      const auto l = exec_->CopyLastLogits();
      out.logits.assign(l.begin(), l.end());
    }
    auto t4 = Clock::now();
    if (decode) {
      timing_.decode_steps += 1;
      timing_.decode_ms += Ms(t1, t2);
    } else {
      timing_.chunks.push_back({rows, Ms(t1, t2)});
    }
    timing_.copy_ms += Ms(t0, t1) + Ms(t2, t3);
    timing_.logits_ms += Ms(t3, t4);
    return out;
  }

  // drafts: the session's most drafts a round; > 0 runs its decode steps on a ring of drafts + 1 recurrent states.
  void Reset(std::uint32_t drafts = 0) {
    exec_->Reset();
    flerp::qwen35moe::SetRecurrentRing(exec_.get(), drafts > 0 ? drafts + 1 : 0);
    drafts_ = drafts;
    timing_ = {};
    ++sessions_;
  }
  void SetDump(flerp::qwen35moe::MoeDump* dump) { dump_ = dump; }

  [[nodiscard]] std::string TimingJson() const {
    std::string chunks = "[";
    for (std::size_t i = 0; i < timing_.chunks.size(); ++i) {
      chunks += std::string(i ? "," : "") + "{\"rows\":" + std::to_string(timing_.chunks[i].rows) + ",\"ms\":" + Num(timing_.chunks[i].ms) +
                ",\"ms_per_block\":" + Num(timing_.chunks[i].ms / blocks()) + "}";
    }
    chunks += "]";
    const double per_step = timing_.decode_steps ? timing_.decode_ms / static_cast<double>(timing_.decode_steps) : 0.0;
    return "{\"rank\":" + std::to_string(rank_) + ",\"blocks\":[" + std::to_string(begin_) + "," + std::to_string(end_) +
           "],\"chunks\":" + chunks + ",\"decode_steps\":" + std::to_string(timing_.decode_steps) + ",\"decode_ms\":" + Num(timing_.decode_ms) +
           ",\"decode_ms_per_step\":" + Num(per_step) + ",\"decode_ms_per_block\":" + Num(per_step / blocks()) +
           ",\"copy_ms\":" + Num(timing_.copy_ms) + ",\"logits_copy_ms\":" + Num(timing_.logits_ms) + ",\"rows_mode\":" +
           (pinned_ ? "\"pinned\"" : "\"pageable\"") + RingJson() + "}";
  }
  [[nodiscard]] std::string RingJson() const {
    const auto* ring = flerp::qwen35moe::FindRecurrentRing(exec_.get());
    if (ring == nullptr || drafts_ == 0) return "";
    return ",\"ring\":{\"slots\":" + std::to_string(ring->slots) + ",\"steps\":" + std::to_string(ring->steps) + ",\"graphs_captured\":" +
           std::to_string(ring->captures) + ",\"bytes\":" + std::to_string(ring->bytes) + "}";
  }

  [[nodiscard]] double blocks() const { return static_cast<double>(end_ - begin_); }
  [[nodiscard]] bool last() const { return last_; }
  [[nodiscard]] bool pinned() const { return pinned_; }
  [[nodiscard]] std::size_t max_rows() const { return max_rows_; }
  [[nodiscard]] const Timing& timing() const { return timing_; }
  [[nodiscard]] gufo::hip::QwenGpuExecutor& exec() { return *exec_; }

 private:
  std::uint32_t rank_, world_;
  std::shared_ptr<const gufo::hip::QwenGpuModel> model_;
  std::unique_ptr<gufo::hip::QwenGpuExecutor> exec_;
  std::uint32_t begin_{0}, end_{0};
  std::size_t hidden_{0}, max_rows_{0};
  bool first_{false}, last_{false};
  bool pinned_{true};
  float* d_in_{nullptr};
  float* d_out_{nullptr};
  float* h_in_{nullptr};    // pinned host rows (pinned mode)
  float* h_out_{nullptr};
  Timing timing_;
  flerp::qwen35moe::MoeDump* dump_{nullptr};
  std::uint32_t sessions_{0};
  std::uint32_t drafts_{0};
};

// A rank > 0: one frame in, its stage, one frame out. Returns false after CLOSE.
// --mtp-probe on the last tcp rank (gate/ornith-o3-mtp.toml): the probe and its stage output rows, or null.
StagedMtpProbe* g_mtp_probe = nullptr;
float* g_mtp_rows = nullptr;
// --draft-head on the last tcp rank (gate/ornith-o3-spec.toml): the drafting side, its stage output rows (g_mtp_rows),
// whether the session drafts, and whether frames after the kept chain are skipped.
flerp::qwen35moe::DraftRound* g_draft = nullptr;
flerp::qwen35moe::DraftCosts g_draft_costs;
bool g_draft_on = false;
bool g_draft_skip = true;
bool g_draft_skip_session = true;   // this session's: off when its RESET asks for every row (kSpecAllRows)
std::size_t g_hidden = 0;

std::span<const std::uint8_t> Words(const std::vector<std::uint32_t>& w) {
  return {reinterpret_cast<const std::uint8_t*>(w.data()), w.size() * sizeof(std::uint32_t)};
}

// The last rank's answer to a frontier: [kept, drafts, frames, kept ids.., drafts.., drawn ids..] (relay kSpecReply).
void SendSpecReply(relay::Relay& r, const relay::FrameHeader& h, const flerp::qwen35moe::DraftRound& d) {
  const auto& kept = d.emitted();
  const auto& drafts = d.drafts();
  const auto& drawn = d.drawn();
  std::vector<std::uint32_t> w{static_cast<std::uint32_t>(kept.size()), static_cast<std::uint32_t>(drafts.size()),
                               static_cast<std::uint32_t>(drawn.size())};
  w.insert(w.end(), kept.begin(), kept.end());
  w.insert(w.end(), drafts.begin(), drafts.end());
  w.insert(w.end(), drawn.begin(), drawn.end());
  r.Send(relay::Op::kReply, h.start_pos, static_cast<std::uint32_t>(w.size()), relay::kSpecReply | relay::kWantLogits, kept.back(), Words(w),
         h.seq_slot);
}

bool ServeOne(Stage& stage, relay::Relay& r, int* session) {
  const relay::Frame f = r.Recv();
  const auto& h = f.header;
  switch (f.op()) {
    case relay::Op::kPrefillChunk:
    case relay::Op::kDecode: {
      const bool decode = f.op() == relay::Op::kDecode;
      const bool want = (h.flags & relay::kWantLogits) != 0U;
      const bool full = (h.flags & relay::kFullLogits) != 0U;
      const std::size_t row_bytes = static_cast<std::size_t>(h.rows) * g_hidden * sizeof(float);
      const bool with_ids = !decode && (h.flags & relay::kWithIds) != 0U;
      // pinned: the payload goes straight into the pinned buffer (Run); pageable: through a float vector, as before
      const auto input = stage.pinned() ? std::vector<float>{} : Floats(std::vector<std::uint8_t>(f.payload.begin(), f.payload.begin() + static_cast<std::ptrdiff_t>(row_bytes)));
      const void* in = stage.pinned() ? static_cast<const void*>(f.payload.data()) : input.data();
      const bool spec = stage.last() && g_draft_on;
      if (decode && (h.flags & relay::kSpecRound) != 0U) {
        // a frame of a speculative round
        if (!stage.last()) {
          const Out o = stage.Run(true, {}, h.start_pos, 1, want, false, in);
          r.Send(f.op(), h.start_pos, h.rows, h.flags, h.token, o.RowBytes(), h.seq_slot);
          return true;
        }
        if (!spec) throw std::runtime_error("a draft frame, and this session does not draft");
        const std::uint32_t j = relay::SpecFrameIndex(h.flags), frames = relay::SpecFrameCount(h.flags);
        if (j == 0) g_draft->BeginRound(h.start_pos, frames);
        if (g_draft_skip_session && !g_draft->NeedsTarget()) {
          g_draft->Skipped(j, h.token);
        } else {
          const Out o = stage.Run(true, {}, h.start_pos, 1, true, false, in);
          g_draft->Frame(j, h.token, static_cast<std::uint32_t>(o.token), g_mtp_rows);
        }
        if (g_draft->RoundDone()) SendSpecReply(r, h, *g_draft);
        return true;
      }
      if (spec && decode) throw std::runtime_error("a plain decode frame in a drafting session");
      const Out o = stage.Run(decode, {}, h.start_pos, h.rows, want, full, in);
      if (g_mtp_probe != nullptr && stage.last()) {
        g_mtp_probe->OnForward(decode, h.start_pos, h.rows, want, static_cast<std::uint32_t>(o.token), g_mtp_rows);
      }
      if (spec) {
        // a prompt chunk of a drafting session: the head's prompt pairs, and at the prompt's end the first drafts
        if (!with_ids) throw std::runtime_error("a prompt chunk without its ids in a drafting session");
        std::vector<std::uint32_t> ids(h.rows);
        std::memcpy(ids.data(), f.payload.data() + row_bytes, ids.size() * sizeof(std::uint32_t));
        g_draft->PromptRows(h.start_pos, h.rows, ids.data(), g_mtp_rows);
        if (want) {
          g_draft->Frontier(h.start_pos + h.rows, static_cast<std::uint32_t>(o.token), g_mtp_rows + (h.rows - 1) * g_hidden);
          SendSpecReply(r, h, *g_draft);
        } else {
          r.Send(relay::Op::kReply, h.start_pos, 1, 0, 0, {}, h.seq_slot);
        }
        return true;
      }
      if (stage.last()) {
        const std::uint32_t flags = want && full ? relay::kFullLogits | relay::kWantLogits : h.flags & relay::kWantLogits;
        r.Send(relay::Op::kReply, h.start_pos, 1, flags, static_cast<std::uint32_t>(o.token),
               want && full ? Bytes(o.logits) : std::span<const std::uint8_t>{}, h.seq_slot);
      } else if (with_ids) {
        std::vector<std::uint8_t> out(o.RowBytes().begin(), o.RowBytes().end());
        out.insert(out.end(), f.payload.begin() + static_cast<std::ptrdiff_t>(row_bytes), f.payload.end());
        r.Send(f.op(), h.start_pos, h.rows, h.flags, h.token, out, h.seq_slot);
      } else {
        r.Send(f.op(), h.start_pos, h.rows, h.flags, h.token, o.RowBytes(), h.seq_slot);
      }
      return true;
    }
    case relay::Op::kReset:
    case relay::Op::kClose: {
      // mirrored: forwarded (or answered) before this rank acts
      if (stage.last()) r.Send(relay::Op::kReply, 0, 1, 0, 0, {}, h.seq_slot);
      else r.Send(f.op(), 0, 0, h.flags, h.token, {}, h.seq_slot);
      if (session != nullptr && (f.op() == relay::Op::kClose || stage.timing().decode_steps + stage.timing().chunks.size() > 0)) {
        std::cout << "{\"event\":\"stage_timing\",\"rank\":" << r.config().rank << ",\"run\":" << (*session - 1) << ",\"timing\":" << stage.TimingJson()
                  << ",\"relay\":{\"frames_sent\":" << r.stats().frames_sent << ",\"bytes_sent\":" << r.stats().bytes_sent
                  << ",\"send_s\":" << Num(r.stats().send_s) << ",\"recv_s\":" << Num(r.stats().recv_s) << "}"
                  << (g_draft != nullptr && g_draft_on && stage.last() ? ",\"draft\":" + g_draft->StatsJson() : std::string()) << "}" << std::endl;
      }
      if (g_mtp_probe != nullptr && stage.last()) {
        if (f.op() == relay::Op::kClose) g_mtp_probe->OnClose();
        else g_mtp_probe->OnReset();
      }
      if (f.op() == relay::Op::kClose) return false;
      flerp::qwen35moe::SetDecodeGraph((h.flags & relay::kGraphOff) == 0U);
      const std::uint32_t drafts = (h.flags & relay::kSpecRound) != 0U ? h.token : 0U;
      stage.Reset(drafts);
      g_draft_on = false;
      if (drafts > 0 && stage.last()) {
        if (g_draft == nullptr) throw std::runtime_error("a drafting session, and the last rank has no --draft-head");
        flerp::qwen35moe::DraftConfig dc;
        dc.max_draft = drafts;
        dc.value_aware = (h.flags & relay::kSpecValue) != 0U;
        dc.lookup = (h.flags & relay::kSpecLookup) != 0U;
        dc.mtp_max = dc.lookup ? (h.flags >> 16) & 0xffU : 0U;
        dc.costs = g_draft_costs;
        g_draft->Begin(dc);
        g_draft_skip_session = g_draft_skip && (h.flags & relay::kSpecAllRows) == 0U;
        g_draft_on = true;
      }
      if (session != nullptr) ++*session;
      return true;
    }
    default:
      r.Abort(std::string("unexpected ") + relay::OpName(f.op()));
      throw relay::RelayError("rank " + std::to_string(r.config().rank) + ": unexpected " + relay::OpName(f.op()));
  }
}

// --- rank 0: the calls of a traced generation ------------------------------------------------------------------
struct CallResult {
  TokenId token{0};
  std::vector<float> logits;
  bool spec{false};                     // a kSpecReply: the kept ids and the next drafts
  std::vector<std::uint32_t> kept, drafts, drawn;
};

struct CallTiming {
  std::vector<double> chunk_wall_ms;
  std::size_t decode_calls{0};   // decode steps, or rounds with drafts
  double decode_wall_ms{0.0};
  std::size_t frames{0};         // decode frames sent (rounds: k' + 1 each)
};

// A spec reply's words: [kept, drafts, frames, kept ids.., drafts.., drawn ids..].
void ParseSpec(const relay::Frame& f, CallResult* res) {
  const std::size_t n = f.payload.size() / sizeof(std::uint32_t);
  std::vector<std::uint32_t> w(n);
  std::memcpy(w.data(), f.payload.data(), n * sizeof(std::uint32_t));
  if (n < 3 || w[0] == 0 || std::size_t{3} + w[0] + w[1] + w[2] != n) throw relay::RelayError("rank 0: a malformed spec reply");
  res->spec = true;
  const auto a = w.begin() + 3, b = a + w[0], c = b + w[1];
  res->kept.assign(a, b);
  res->drafts.assign(b, c);
  res->drawn.assign(c, w.end());
  res->token = static_cast<TokenId>(res->kept.back());
}

class Pipeline {
 public:
  Pipeline(Stage& rank0, relay::Relay* r, std::vector<std::pair<Stage*, relay::Relay*>> loop_ranks)
      : rank0_(rank0), relay_(r), loop_(std::move(loop_ranks)) {}

  // full: the last call of a prompt and every decode step ask for the whole logits row, else the token id only.
  // subchunks: a prompt chunk goes down the chain as that many sub-chunks, back to back (decode steps: one row).
  void SetReply(bool full, std::size_t subchunks) {
    full_ = full;
    subchunks_ = std::max<std::size_t>(subchunks, 1);
  }
  // drafts: the most drafts a round the next sessions ask of the last stage (0: none); value: value-aware depth.
  void SetDrafts(std::uint32_t drafts, bool value, bool lookup = false, std::uint32_t mtp_max = 0) {
    if (drafts > 0 && (relay_ == nullptr || !loop_.empty())) throw std::invalid_argument("drafts need --staged tcp");
    if (drafts > 0 && full_) throw std::invalid_argument("drafts need --reply token");
    if (drafts > 64) throw std::invalid_argument("at most 64 drafts a round");
    drafts_ = drafts;
    value_ = value;
    lookup_ = lookup && drafts > 0;
    mtp_max_ = mtp_max;
  }

  CallResult Call(bool decode, const std::vector<TokenId>& tokens, std::uint32_t start, bool want) {
    const auto t0 = Clock::now();
    const std::size_t rows = tokens.size();
    // Sub-chunk k has rows / parts rows, one more for k < rows % parts; only the last asks for logits. Rank 0 sends
    // each as soon as its own blocks have run it, without waiting for the reply of the one before; the replies are
    // read in order after the last send (the last stage answers every frame, so they are as many as the sends).
    const std::size_t parts = decode ? 1 : std::min(subchunks_, rows);
    CallResult res;
    std::size_t off = 0;
    for (std::size_t k = 0; k < parts; ++k) {
      const std::size_t n = rows / parts + (k < rows % parts ? 1 : 0);
      const bool last = k + 1 == parts;
      const bool w = want && last;
      const std::vector<TokenId> sub(tokens.begin() + static_cast<std::ptrdiff_t>(off), tokens.begin() + static_cast<std::ptrdiff_t>(off + n));
      const auto pos = static_cast<std::uint32_t>(start + off);
      Out o = rank0_.Run(decode, sub, pos, n, w, full_, nullptr);
      if (relay_ == nullptr) {
        if (last) {
          res.token = o.token;
          res.logits = std::move(o.logits);
        }
      } else if (drafts_ > 0 && !decode) {
        // a drafting session: the prompt rows carry their ids, for the last stage's MTP head
        const std::uint32_t flags = (w ? relay::kWantLogits : 0U) | relay::kWithIds;
        std::vector<std::uint8_t> payload(o.RowBytes().begin(), o.RowBytes().end());
        for (const TokenId id : sub) {
          const auto u = static_cast<std::uint32_t>(id);
          const auto* b = reinterpret_cast<const std::uint8_t*>(&u);
          payload.insert(payload.end(), b, b + sizeof(u));
        }
        relay_->Send(relay::Op::kPrefillChunk, pos, static_cast<std::uint32_t>(n), flags, 0, payload);
      } else {
        const std::uint32_t flags = w ? relay::kWantLogits | (full_ ? relay::kFullLogits : 0U) : 0U;
        relay_->Send(decode ? relay::Op::kDecode : relay::Op::kPrefillChunk, pos, static_cast<std::uint32_t>(n), flags, 0, o.RowBytes());
      }
      off += n;
    }
    if (relay_ != nullptr) {
      for (std::size_t k = 0; k < parts; ++k) Pump();   // loopback: each other rank serves one frame a pump
      for (std::size_t k = 0; k < parts; ++k) {
        relay::Frame f = relay_->Recv();
        if (f.op() != relay::Op::kReply) {
          relay_->Abort(std::string("expected REPLY, got ") + relay::OpName(f.op()));
          throw relay::RelayError("rank 0: expected REPLY");
        }
        if (k + 1 == parts) {
          res.token = static_cast<TokenId>(f.header.token);
          if ((f.header.flags & relay::kSpecReply) != 0U) ParseSpec(f, &res);
          else res.logits = Floats(f.payload);
        }
      }
    }
    const double ms = Ms(t0, Clock::now());
    if (decode) {
      timing_.decode_calls += 1;
      timing_.decode_wall_ms += ms;
      timing_.frames += 1;
    } else {
      timing_.chunk_wall_ms.push_back(ms);
    }
    return res;
  }

  // One speculative round: ids[0] (the pending id) and the drafts at start, start + 1, .. as back-to-back
  // single-row decode frames; the last stage answers once with the ids it keeps and the next drafts.
  CallResult Round(const std::vector<TokenId>& ids, std::uint32_t start) {
    const auto t0 = Clock::now();
    const auto frames = static_cast<std::uint32_t>(ids.size());
    for (std::uint32_t j = 0; j < frames; ++j) {
      Out o = rank0_.Run(true, {ids[j]}, start + j, 1, true, false, nullptr);
      relay_->Send(relay::Op::kDecode, start + j, 1, relay::kWantLogits | relay::SpecFrameFlags(j, frames), static_cast<std::uint32_t>(ids[j]),
                   o.RowBytes());
    }
    relay::Frame f = relay_->Recv();
    if (f.op() != relay::Op::kReply || (f.header.flags & relay::kSpecReply) == 0U) {
      relay_->Abort("expected a spec REPLY");
      throw relay::RelayError("rank 0: expected a spec REPLY");
    }
    CallResult res;
    ParseSpec(f, &res);
    timing_.decode_calls += 1;
    timing_.frames += frames;
    timing_.decode_wall_ms += Ms(t0, Clock::now());
    return res;
  }

  void Reset(bool graph) {
    std::uint32_t flags = graph ? 0U : relay::kGraphOff;
    if (drafts_ > 0) flags |= relay::kSpecRound | (value_ ? relay::kSpecValue : 0U);
    if (lookup_) flags |= relay::kSpecLookup | ((mtp_max_ & 0xffU) << 16);
    if (relay_ != nullptr) relay_->Send(relay::Op::kReset, 0, 0, flags, drafts_, {});
    flerp::qwen35moe::SetDecodeGraph(graph);
    rank0_.Reset(drafts_);
    timing_ = {};
    if (relay_ != nullptr) {
      Pump();
      if (relay_->Recv().op() != relay::Op::kReply) {
        relay_->Abort("RESET not answered");
        throw relay::RelayError("rank 0: RESET not answered");
      }
    }
  }

  void Close() {
    if (relay_ == nullptr) return;
    relay_->Send(relay::Op::kClose, 0, 0, 0, 0, {});
    Pump();
    (void)relay_->Recv();
  }

  // Loopback: the other ranks' loops, one step each, in ring order. A rank that throws is poisoned; the others
  // still take their turn (their receive then fails on the closed link).
  void Pump() {
    for (auto& [stage, r] : loop_) {
      try {
        (void)ServeOne(*stage, *r, nullptr);
      } catch (const relay::RelayError&) {
      }
    }
  }

  [[nodiscard]] const CallTiming& timing() const { return timing_; }
  [[nodiscard]] Stage& rank0() { return rank0_; }
  [[nodiscard]] bool full() const { return full_; }
  [[nodiscard]] std::size_t subchunks() const { return subchunks_; }
  [[nodiscard]] bool rank0_pinned() const { return rank0_.pinned(); }
  [[nodiscard]] std::uint32_t drafts() const { return drafts_; }
  [[nodiscard]] bool value() const { return value_; }
  [[nodiscard]] bool lookup() const { return lookup_; }
  [[nodiscard]] std::uint32_t mtp_max() const { return mtp_max_; }

 private:
  Stage& rank0_;
  relay::Relay* relay_;
  std::vector<std::pair<Stage*, relay::Relay*>> loop_;
  CallTiming timing_;
  bool full_{true};
  std::size_t subchunks_{1};
  std::uint32_t drafts_{0};
  bool value_{false};
  bool lookup_{false};
  std::uint32_t mtp_max_{0};
};

std::uint32_t Argmax(const std::vector<float>& v) {
  std::uint32_t best = 0;
  for (std::uint32_t i = 1; i < v.size(); ++i) {
    if (v[i] > v[best]) best = i;
  }
  return best;
}

struct Traced {
  std::vector<TokenId> ids;
  std::vector<std::string> sha;
  bool argmax_is_token{true};
  double prefill_s{0}, decode_s{0};
  // drafting sessions: a round's proposed drafts k' and kept drafts n, and the ids it returned (first round: the
  // prompt's reply, k' = n = 0)
  std::vector<std::pair<std::uint32_t, std::uint32_t>> rounds;
  std::vector<std::vector<std::uint32_t>> round_ids;
  bool accounting_ok{true};
  std::string accounting_error;
};

// The calls of generate.cpp's Trace (prefill, then a decode step a token, the logits row of every step copied),
// with the prompt cut into the arena's chunks as QwenGpuExecutor::ForwardPromptBatch cuts it (prefill.cpp).
Traced RunPrompt(Pipeline& p, const std::vector<TokenId>& prompt, std::size_t max_tokens, bool graph,
                 const gufo::tokenization::QwenTokenizer& tokenizer) {
  Traced t;
  p.Reset(graph);
  const std::size_t batch = p.rank0().max_rows();
  auto t0 = Clock::now();
  CallResult cur;
  for (std::size_t off = 0; off < prompt.size(); off += batch) {
    const std::size_t n = std::min(batch, prompt.size() - off);
    const bool last = off + n == prompt.size();
    cur = p.Call(false, std::vector<TokenId>(prompt.begin() + static_cast<std::ptrdiff_t>(off), prompt.begin() + static_cast<std::ptrdiff_t>(off + n)),
                 static_cast<std::uint32_t>(off), last);
  }
  t.prefill_s = std::chrono::duration<double>(Clock::now() - t0).count();
  auto t1 = Clock::now();
  std::size_t position = prompt.size();
  if (p.drafts() > 0) {
    // ORNITH-O3 step two: rounds of the pending id and the last stage's drafts. The ids are taken in order as the
    // drafts-off loop below takes them: a stop id ends the reply before it is taken; max_tokens ids end it.
    if (!cur.spec) throw std::runtime_error("a drafting session's prompt reply carries no drafts");
    const auto take = [&](std::uint32_t id) {
      if (tokenizer.IsStopToken(static_cast<TokenId>(id))) return false;
      t.ids.push_back(static_cast<TokenId>(id));
      return t.ids.size() < max_tokens;
    };
    t.rounds.emplace_back(0, 0);
    t.round_ids.push_back(cur.kept);
    bool more = max_tokens > 0 && take(cur.kept.back());
    while (more) {
      const std::size_t room = max_tokens - t.ids.size();
      std::vector<TokenId> ids{static_cast<TokenId>(cur.kept.back())};
      for (std::size_t i = 0; i < cur.drafts.size() && i < room; ++i) ids.push_back(static_cast<TokenId>(cur.drafts[i]));
      const std::vector<std::uint32_t> sent(cur.drafts.begin(), cur.drafts.begin() + static_cast<std::ptrdiff_t>(ids.size() - 1));
      cur = p.Round(ids, static_cast<std::uint32_t>(position));
      const std::size_t n = cur.kept.size() - 1;
      if (n > sent.size() || !std::equal(cur.kept.begin(), cur.kept.begin() + static_cast<std::ptrdiff_t>(n), sent.begin())) {
        t.accounting_ok = false;
        t.accounting_error = "round " + std::to_string(t.rounds.size()) + ": kept ids are not the drafts sent";
      }
      t.rounds.emplace_back(static_cast<std::uint32_t>(sent.size()), static_cast<std::uint32_t>(n));
      t.round_ids.push_back(cur.kept);
      position += cur.kept.size();
      for (const std::uint32_t id : cur.kept) {
        more = take(id);
        if (!more) break;
      }
    }
    t.decode_s = std::chrono::duration<double>(Clock::now() - t1).count();
    return t;
  }
  while (t.ids.size() < max_tokens) {
    if (p.full()) {   // a token-only reply has no row to hash or check
      t.sha.push_back(Sha(cur.logits.data(), cur.logits.size() * sizeof(float)));
      t.argmax_is_token = t.argmax_is_token && !cur.logits.empty() && Argmax(cur.logits) == static_cast<std::uint32_t>(cur.token);
    }
    if (tokenizer.IsStopToken(cur.token)) break;
    t.ids.push_back(cur.token);
    cur = p.Call(true, {cur.token}, static_cast<std::uint32_t>(position), true);
    ++position;
  }
  t.decode_s = std::chrono::duration<double>(Clock::now() - t1).count();
  return t;
}

// --- options ----------------------------------------------------------------------------------------------------
struct StageConfig {
  std::string text;
  std::uint32_t world{1};
  std::vector<std::uint32_t> cuts;   // empty: even (the stage plan's)
  bool graph{true};
};

std::vector<std::uint32_t> ParseList(const std::string& s) {
  std::vector<std::uint32_t> out;
  std::istringstream in(s);
  for (std::string item; std::getline(in, item, ',');) {
    if (!item.empty()) out.push_back(static_cast<std::uint32_t>(std::stoul(item)));
  }
  return out;
}

StageConfig ParseConfig(const std::string& text) {
  StageConfig c;
  c.text = text;
  const auto a = text.find(':'), b = text.rfind(':');
  if (a == std::string::npos || a == b) throw std::invalid_argument("config: want W:CUTS:GRAPH, got " + text);
  c.world = static_cast<std::uint32_t>(std::stoul(text.substr(0, a)));
  const std::string cuts = text.substr(a + 1, b - a - 1), graph = text.substr(b + 1);
  if (cuts != "even") c.cuts = ParseList(cuts);
  if (graph != "on" && graph != "off") throw std::invalid_argument("config: graph is on or off");
  c.graph = graph == "on";
  return c;
}

struct Options {
  std::string mode, model;
  std::uint32_t context{4096};
  std::size_t max_tokens{128};
  std::vector<std::string> id_files;
  flerp::qwen35moe::LoadOptions load;
  std::vector<StageConfig> configs;
  std::optional<StageConfig> fault_config;
  std::uint32_t world{1}, rank{0};
  std::string cuts{"even"};
  std::uint16_t port{39500};
  std::string host{"127.0.0.1"};
  std::uint64_t epoch{1};
  std::uint32_t deadline_ms{60000};
  std::uint32_t setup_deadline_ms{1800000};
  std::vector<bool> graph_modes{true};
  RowsMode rows_mode{RowsMode::kPinned};
  bool reply_full{true};
  std::size_t subchunks{1};
  struct RunSpec {
    bool graph{true};
    bool full{true};
    std::size_t subchunks{1};
    std::uint32_t drafts{0};   // ORNITH-O3 step two: drafts a round (0: none)
    bool value{false};         // ... at most, value-aware
    bool lookup{false};        // ORNITH-CRUCIBLE item 4: prompt-lookup drafts first (drafts = the lookup cap)
    std::uint32_t mtp_max{0};  // ... and at most this many MTP drafts
  };
  std::vector<RunSpec> run_specs;   // tcp rank 0: empty = one a --graph-modes entry with --reply and --prompt-subchunks
  std::string fault;
  std::string dump_path;
  std::uint32_t mtp_depth{0};      // --mtp-probe D: the last tcp rank drafts with the MTP head (mtp_probe.hpp)
  std::string mtp_data_file;        // --mtp-data-file F: the head's bytes (default: --data-file, else the GGUF)
  bool draft_head{false};           // --draft-head: the last tcp rank drafts with the MTP head (gate/ornith-o3-spec.toml)
  flerp::qwen35moe::DraftCosts draft_costs;   // --draft-costs B,R,M
  bool draft_skip{true};            // --draft-skip on|off
  std::vector<std::uint32_t> dump_layers, dump_rows;
};

// Cuts of a configuration: given, or the stage plan's for that world (byte-balanced, design 3.1).
std::vector<std::uint32_t> ResolveCuts(const gufo::core::GgufReader& reader, std::uint32_t world, const std::vector<std::uint32_t>& given,
                                       std::uint32_t blocks) {
  if (!given.empty()) {
    if (given.size() != world + 1 || given.front() != 0 || given.back() != blocks) throw std::invalid_argument("cuts do not cover the blocks");
    for (std::size_t i = 1; i < given.size(); ++i) {
      if (given[i] <= given[i - 1]) throw std::invalid_argument("cuts must increase");
    }
    return given;
  }
  std::string error;
  const auto plan = flerp::gguf_placed::StagePlan(reader, world, std::nullopt, &error);
  if (!plan) throw std::runtime_error("placement: " + error);
  return plan->cuts;
}

// What every rank of one configuration hashes into its frames: the model's whole-file plan, the cuts, the shapes.
std::array<std::uint8_t, 32> RelayPlan(const std::string& whole_plan_sha, const std::vector<std::uint32_t>& cuts, std::uint32_t hidden,
                                       std::uint32_t vocab, std::size_t max_rows) {
  std::string text = "flerp-stage-relay v1\nmodel-plan " + whole_plan_sha + "\ncuts";
  for (const auto c : cuts) text += " " + std::to_string(c);
  text += "\nhidden " + std::to_string(hidden) + "\nvocab " + std::to_string(vocab) + "\nmax_rows " + std::to_string(max_rows) + "\n";
  return relay::PlanFromHex(Sha(text.data(), text.size()));
}

std::vector<std::vector<TokenId>> ReadPrompts(const std::vector<std::string>& files) {
  std::vector<std::vector<TokenId>> out;
  for (const auto& f : files) {
    std::ifstream in(f);
    if (!in) throw std::runtime_error("cannot read " + f);
    std::vector<TokenId> ids;
    for (long long id = 0; in >> id;) ids.push_back(static_cast<TokenId>(id));
    out.push_back(std::move(ids));
  }
  return out;
}

relay::Config RelayConfig(const Options& o, std::uint32_t world, std::uint32_t rank, const std::array<std::uint8_t, 32>& plan,
                          const gufo::core::ModelConfig& mc, std::size_t max_rows) {
  relay::Config c;
  c.world = world;
  c.rank = rank;
  c.epoch = o.epoch;
  c.hidden = mc.hidden_size;
  c.vocab = mc.vocab_size;
  c.max_rows = static_cast<std::uint32_t>(max_rows);
  c.plan = plan;
  c.deadline = std::chrono::milliseconds(o.deadline_ms);
  return c;
}

std::string LoadedJson(const char* mode, double load_s) {
  const auto& rep = flerp::qwen35moe::LastLoadReport();
  return std::string("{\"event\":\"loaded\",\"mode\":") + Str(mode) + ",\"load_s\":" + Num(load_s) + ",\"placed\":{\"plan_sha256\":\"" +
         rep.plan_sha256 + "\",\"blocks\":[" + std::to_string(rep.block_begin) + "," + std::to_string(rep.block_end) +
         "],\"allocated_bytes\":" + std::to_string(rep.allocated_bytes) + ",\"plan_device_bytes\":" + std::to_string(rep.plan_device_bytes) +
         ",\"tensor_bytes\":" + std::to_string(rep.tensor_bytes) + ",\"fill_s\":" + Num(rep.fill_s) + ",\"pause_s\":" + Num(rep.pause_s) +
         ",\"single_reader\":" + (rep.single_reader ? "true" : "false") + ",\"direct_io\":" + (rep.direct_io ? "true" : "false") +
         ",\"hip_free_before\":" + std::to_string(rep.hip_free_before) + ",\"hip_free_after\":" + std::to_string(rep.hip_free_after) +
         ",\"hip_total\":" + std::to_string(rep.hip_total) + ",\"data_path\":" + Str(rep.data_path) + ",\"punched_bytes\":" +
         std::to_string(rep.punched_bytes) + ",\"swarm\":" + (rep.swarm ? "true" : "false") + ",\"swarm_bytes\":" + std::to_string(rep.swarm_bytes) +
         ",\"swarm_runs\":" + std::to_string(rep.swarm_runs) + ",\"swarm_stats\":" + (rep.swarm_stats.empty() ? std::string("null") : rep.swarm_stats) +
         ",\"read_gb_per_s\":" + Num(rep.fill_s > 0 ? static_cast<double>(rep.tensor_bytes) / rep.fill_s / 1e9 : 0.0) + "}}";
}

std::string StagedJson(const char* transport, const StageConfig& c, const std::vector<std::uint32_t>& cuts, std::size_t index,
                       std::size_t prompt_tokens, const Traced& t, const Pipeline& p, const std::string& stages_json, double stages_decode_ms,
                       double stages_chunk_ms, double last_logits_ms) {
  const auto& ct = p.timing();
  double chunk_wall = 0;
  for (const double v : ct.chunk_wall_ms) chunk_wall += v;
  const double steps = static_cast<double>(std::max<std::size_t>(ct.decode_calls, 1));
  const double chunks = static_cast<double>(std::max<std::size_t>(ct.chunk_wall_ms.size(), 1));
  std::string o = std::string("{\"event\":\"staged\",\"transport\":") + Str(transport) + ",\"config\":" + Str(c.text) +
                  ",\"world\":" + std::to_string(c.world) + ",\"cuts\":" + Ids(cuts) + ",\"graph\":" + (c.graph ? "true" : "false") +
                  ",\"prompt\":" + std::to_string(index) + ",\"prompt_tokens\":" + std::to_string(prompt_tokens) + ",\"ids\":" + Ids(t.ids) +
                  ",\"rows\":" + std::to_string(t.sha.size()) + ",\"logits_sha256\":" + Strs(t.sha) +
                  ",\"argmax_is_token\":" + (t.argmax_is_token ? "true" : "false") + ",\"reply\":" + (p.full() ? "\"full\"" : "\"token\"") +
                  ",\"subchunks\":" + std::to_string(p.subchunks()) + ",\"rows_mode\":" + (p.rank0_pinned() ? "\"pinned\"" : "\"pageable\"") +
                  ",\"prefill_s\":" + Num(t.prefill_s) +
                  ",\"decode_s\":" + Num(t.decode_s) + ",\"calls\":{\"chunks\":" + std::to_string(ct.chunk_wall_ms.size()) +
                  ",\"chunk_wall_ms\":" + Num(chunk_wall) + ",\"decode_calls\":" + std::to_string(ct.decode_calls) +
                  ",\"decode_wall_ms\":" + Num(ct.decode_wall_ms) + ",\"frames\":" + std::to_string(ct.frames) + "}";
  o += ",\"drafts\":" + std::to_string(p.drafts()) + ",\"value\":" + (p.value() ? "true" : "false") + ",\"lookup\":" +
       (p.lookup() ? "true" : "false") + ",\"mtp_max\":" + std::to_string(p.mtp_max());
  if (p.drafts() > 0) {
    std::string rounds = "[", ids = "[";
    for (std::size_t i = 0; i < t.rounds.size(); ++i) {
      rounds += std::string(i ? "," : "") + "[" + std::to_string(t.rounds[i].first) + "," + std::to_string(t.rounds[i].second) + "]";
      ids += std::string(i ? "," : "") + Ids(t.round_ids[i]);
    }
    o += ",\"rounds\":" + rounds + "],\"round_ids\":" + ids + "],\"accounting_ok\":" + (t.accounting_ok ? "true" : "false") +
         ",\"accounting_error\":" + Str(t.accounting_error);
  }
  if (!stages_json.empty()) {
    // relay overhead = wall time of rank 0's calls - the stages' forwards - the last stage's logits copy (which a
    // single node makes too): copies of rows to and from the device, framing, checks, the transport.
    o += ",\"stages\":" + stages_json + ",\"relay_overhead\":{\"decode_ms_per_token\":" +
         Num((ct.decode_wall_ms - stages_decode_ms - last_logits_ms) / steps) + ",\"chunk_ms_per_chunk\":" +
         Num((chunk_wall - stages_chunk_ms) / chunks) + ",\"note\":\"logits copies of the last stage are subtracted from decode only\"}";
  }
  return o + "}";
}

// --- loopback ---------------------------------------------------------------------------------------------------
struct LoopbackSet {
  std::vector<std::shared_ptr<const gufo::hip::QwenGpuModel>> views;
  std::vector<std::unique_ptr<Stage>> stages;
  std::vector<std::unique_ptr<relay::Relay>> relays;
};

std::shared_ptr<const gufo::hip::QwenGpuModel> View(const std::shared_ptr<const gufo::core::GgufReader>& reader,
                                                    const std::shared_ptr<const gufo::hip::QwenGpuModel>& base, std::uint32_t begin,
                                                    std::uint32_t end) {
  // The weights of the one load, re-ranged: the same device tensors, no regions of its own (nothing to free).
  auto weights = base->GetWeights();
  weights.block_begin = begin;
  weights.block_end = end;
  std::string error;
  auto tok = gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  if (!tok) throw std::runtime_error("tokenizer: " + error);
  return std::make_shared<const gufo::hip::QwenGpuModel>(reader, std::move(weights),
                                                         std::shared_ptr<const gufo::tokenization::QwenTokenizer>(std::move(tok)),
                                                         std::vector<gufo::hip::QwenGpuWeightRegion>{});
}

LoopbackSet BuildLoopback(const Options& o, const std::shared_ptr<const gufo::core::GgufReader>& reader,
                          const std::shared_ptr<const gufo::hip::QwenGpuModel>& base, const std::vector<std::uint32_t>& cuts,
                          const std::string& whole_sha, std::vector<std::uint64_t> epochs = {}) {
  LoopbackSet s;
  const std::uint32_t world = static_cast<std::uint32_t>(cuts.size() - 1);
  for (std::uint32_t r = 0; r < world; ++r) {
    s.views.push_back(world == 1 ? base : View(reader, base, cuts[r], cuts[r + 1]));
    s.stages.push_back(std::make_unique<Stage>(r, world, s.views.back(), o.context, o.rows_mode));
  }
  if (world > 1) {
    const auto& mc = base->GetConfig();
    const auto plan = RelayPlan(whole_sha, cuts, mc.hidden_size, mc.vocab_size, s.stages[0]->max_rows());
    std::vector<relay::Config> cfgs;
    for (std::uint32_t r = 0; r < world; ++r) {
      cfgs.push_back(RelayConfig(o, world, r, plan, mc, s.stages[0]->max_rows()));
      if (r < epochs.size()) cfgs.back().epoch = epochs[r];
    }
    s.relays = relay::LoopbackRing(cfgs);
  }
  return s;
}

Pipeline MakePipeline(LoopbackSet& s, const Options& o) {
  std::vector<std::pair<Stage*, relay::Relay*>> loop;
  for (std::size_t r = 1; r < s.stages.size(); ++r) loop.emplace_back(s.stages[r].get(), s.relays[r].get());
  Pipeline p(*s.stages[0], s.relays.empty() ? nullptr : s.relays[0].get(), std::move(loop));
  p.SetReply(o.reply_full, o.subchunks);
  return p;
}

int Loopback(const Options& o, const std::shared_ptr<const gufo::core::GgufReader>& reader) {
  flerp::qwen35moe::SetLoadOptions(o.load);
  const auto t0 = Clock::now();
  std::string error;
  auto base = gufo::hip::QwenGpuModel::CreateFromGguf(reader, &error);
  if (!base) throw std::runtime_error("load: " + error);
  std::cout << LoadedJson("loopback", std::chrono::duration<double>(Clock::now() - t0).count()) << std::endl;
  g_hidden = base->GetConfig().hidden_size;
  std::string plan_error;
  const auto whole = flerp::gguf_placed::StagePlan(*reader, 1, std::nullopt, &plan_error);
  if (!whole) throw std::runtime_error("placement: " + plan_error);
  const std::string whole_sha = whole->Sha256();
  const auto prompts = ReadPrompts(o.id_files);
  const std::uint32_t blocks = base->GetConfig().num_layers;
  int rc = 0;

  for (const auto& c : o.configs) {
    const auto cuts = ResolveCuts(*reader, c.world, c.cuts, blocks);
    auto set = BuildLoopback(o, reader, base, cuts, whole_sha);
    for (auto& r : set.relays) {
      if (r->poisoned()) throw std::runtime_error("loopback handshake failed: " + r->reason());
    }
    Pipeline p = MakePipeline(set, o);
    for (std::size_t i = 0; i < prompts.size(); ++i) {
      try {
        const Traced t = RunPrompt(p, prompts[i], o.max_tokens, c.graph, base->GetTokenizer());
        std::string stages = "[";
        double dec = 0, chunk = 0, last_logits = 0;
        for (std::size_t r = 0; r < set.stages.size(); ++r) {
          const auto& tm = set.stages[r]->timing();
          stages += std::string(r ? "," : "") + set.stages[r]->TimingJson();
          dec += tm.decode_ms;
          for (const auto& ch : tm.chunks) chunk += ch.ms;
          if (set.stages[r]->last()) last_logits = tm.logits_ms;
        }
        stages += "]";
        std::cout << StagedJson("loopback", c, cuts, i, prompts[i].size(), t, p, stages, dec, chunk, last_logits) << std::endl;
      } catch (const std::exception& e) {
        std::cout << "{\"event\":\"staged_failed\",\"config\":" << Str(c.text) << ",\"prompt\":" << i << ",\"error\":" << Str(e.what()) << "}"
                  << std::endl;
        rc = 1;
        break;
      }
    }
  }

  if (o.fault_config) {
    const auto& c = *o.fault_config;
    const auto cuts = ResolveCuts(*reader, c.world, c.cuts, blocks);
    const std::vector<TokenId>& prompt = prompts.front();
    using K = relay::Fault::Kind;
    // ordinal 6 on rank 1's link: HELLO, RESET, the prompt's chunk(s), then decode steps (a short prompt: step 3)
    for (const K kind : {K::kDrop, K::kStaleEpoch, K::kReplay, K::kBadBytes}) {
      auto set = BuildLoopback(o, reader, base, cuts, whole_sha);
      Pipeline p = MakePipeline(set, o);
      bool fired = false;
      set.relays[1]->SetFault({kind, 6}, [&](const relay::Fault&) { fired = true; });
      std::string rank0_error;
      const auto t0 = Clock::now();
      try {
        (void)RunPrompt(p, prompt, o.max_tokens, c.graph, base->GetTokenizer());
      } catch (const std::exception& e) {
        rank0_error = e.what();
      }
      // every rank still unpoisoned is given its next call, as its own loop would make it
      for (std::size_t r = 1; r < set.relays.size(); ++r) {
        if (!set.relays[r]->poisoned()) {
          try {
            (void)ServeOne(*set.stages[r], *set.relays[r], nullptr);
          } catch (const std::exception&) {
          }
        }
      }
      std::vector<std::string> reasons;
      bool all = true;
      for (auto& r : set.relays) {
        all = all && r->poisoned();
        reasons.push_back(r->reason());
      }
      std::cout << "{\"event\":\"fault_case\",\"transport\":\"loopback\",\"config\":" << Str(c.text) << ",\"fault\":" << Str(relay::FaultName(kind))
                << ",\"ordinal\":6,\"rank\":1,\"fired\":" << (fired ? "true" : "false") << ",\"rank0_call_failed\":" << (rank0_error.empty() ? "false" : "true")
                << ",\"rank0_error\":" << Str(rank0_error) << ",\"all_poisoned\":" << (all ? "true" : "false") << ",\"reasons\":" << Strs(reasons)
                << ",\"ms\":" << Num(Ms(t0, Clock::now())) << "}" << std::endl;
    }
    {
      std::vector<std::uint64_t> epochs(c.world, o.epoch);
      epochs[std::min<std::size_t>(2, c.world - 1)] = o.epoch - 1;
      auto set = BuildLoopback(o, reader, base, cuts, whole_sha, epochs);
      std::vector<std::string> reasons;
      bool any = false;
      for (auto& r : set.relays) {
        any = any || r->poisoned();
        reasons.push_back(r->reason());
      }
      std::cout << "{\"event\":\"fault_case\",\"transport\":\"loopback\",\"config\":" << Str(c.text)
                << ",\"fault\":\"wrong-epoch-at-handshake\",\"rank\":" << std::min<std::size_t>(2, c.world - 1) << ",\"fired\":true,\"refused\":"
                << (any ? "true" : "false") << ",\"reasons\":" << Strs(reasons) << "}" << std::endl;
    }
  }
  return rc;
}

// --host A or A0,A1,..: one address for every rank, or one a rank (rank r listens on its own, connects to r + 1's).
std::vector<std::string> Hosts(const std::string& text) {
  std::vector<std::string> out;
  std::istringstream in(text);
  for (std::string item; std::getline(in, item, ',');) {
    if (!item.empty()) out.push_back(item);
  }
  if (out.empty()) throw std::invalid_argument("--host names no address");
  return out;
}

// --- one TCP rank -----------------------------------------------------------------------------------------------
int TcpRank(const Options& o, const std::shared_ptr<const gufo::core::GgufReader>& reader) {
  ::signal(SIGPIPE, SIG_IGN);
  std::string error;
  const auto whole = flerp::gguf_placed::StagePlan(*reader, 1, std::nullopt, &error);
  if (!whole) throw std::runtime_error("placement: " + error);
  const std::uint32_t blocks = whole->blocks;
  const auto cuts = ResolveCuts(*reader, o.world, o.cuts == "even" ? std::vector<std::uint32_t>{} : ParseList(o.cuts), blocks);
  auto load = o.load;
  load.placed = true;
  load.world = 1;
  load.rank = 0;
  load.has_range = true;
  load.block_begin = cuts[o.rank];
  load.block_end = cuts[o.rank + 1];
  flerp::qwen35moe::SetLoadOptions(load);
  const auto t0 = Clock::now();
  auto model = gufo::hip::QwenGpuModel::CreateFromGguf(reader, &error);
  if (!model) throw std::runtime_error("load: " + error);
  std::cout << LoadedJson("tcp", std::chrono::duration<double>(Clock::now() - t0).count()) << std::endl;
  Stage stage(o.rank, o.world, model, o.context, o.rows_mode);
  // --mtp-probe (gate/ornith-o3-mtp.toml): on the last rank only, the file's MTP head beside the stage. Its rows come
  // through a stage output buffer bound here, before the first forward (the decode graph's key folds it); the ids
  // are the run's own. The other ranks ignore the flag.
  std::unique_ptr<flerp::qwen35moe::MtpHead> mtp_head;
  std::unique_ptr<StagedMtpProbe> mtp_probe;
  g_hidden = model->GetWeights().config.hidden_size;
  if (o.mtp_depth > 0 && o.draft_head) throw std::invalid_argument("--mtp-probe and --draft-head exclude each other");
  std::unique_ptr<flerp::qwen35moe::DraftRound> draft;
  if (o.draft_head && o.rank + 1 == o.world) {
    // ORNITH-O3 step two: the head beside the last stage, its rows through a stage output buffer bound before the
    // first forward (as the probe's), drafting only in sessions whose RESET asks for drafts
    const auto& w = model->GetWeights();
    HipOk(hipMalloc(reinterpret_cast<void**>(&g_mtp_rows), stage.max_rows() * w.config.hidden_size * sizeof(float)), "draft rows");
    auto io = flerp::qwen35moe::GetStageIo(&w);
    io.d_output = g_mtp_rows;
    flerp::qwen35moe::BindStageIo(&w, io);
    std::string mtp_error;
    const std::string mtp_data = o.mtp_data_file.empty() ? o.load.fill.data_path : o.mtp_data_file;
    mtp_head = flerp::qwen35moe::MtpHead::Create(*reader, w, {.max_context = o.context, .data_path = mtp_data}, &mtp_error);
    if (!mtp_head) throw std::runtime_error(mtp_error);
    draft = std::make_unique<flerp::qwen35moe::DraftRound>(*mtp_head, w.config.hidden_size, o.context);
    g_draft = draft.get();
    g_draft_costs = o.draft_costs;
    g_draft_skip = o.draft_skip;
    std::cout << "{\"event\":\"draft_head\",\"rank\":" << o.rank << ",\"block\":" << mtp_head->block() << ",\"weight_bytes\":" << mtp_head->weight_bytes()
              << ",\"buffer_bytes\":" << mtp_head->buffer_bytes() << ",\"load_s\":" << Num(mtp_head->load_s())
              << ",\"own_embedding\":" << (mtp_head->own_embedding() ? "true" : "false") << ",\"data_file\":" << Str(mtp_data)
              << ",\"costs_ms\":[" << Num(o.draft_costs.base_ms) << "," << Num(o.draft_costs.row_ms) << "," << Num(o.draft_costs.mtp_ms)
              << "],\"skip\":" << (o.draft_skip ? "true" : "false") << "}" << std::endl;
  }
  if (o.mtp_depth > 0 && o.rank + 1 == o.world) {
    const auto& w = model->GetWeights();
    HipOk(hipMalloc(reinterpret_cast<void**>(&g_mtp_rows), stage.max_rows() * w.config.hidden_size * sizeof(float)), "mtp probe rows");
    auto io = flerp::qwen35moe::GetStageIo(&w);
    io.d_output = g_mtp_rows;
    flerp::qwen35moe::BindStageIo(&w, io);
    std::string mtp_error;
    const std::string mtp_data = o.mtp_data_file.empty() ? o.load.fill.data_path : o.mtp_data_file;
    mtp_head = flerp::qwen35moe::MtpHead::Create(*reader, w, {.max_context = o.context, .data_path = mtp_data}, &mtp_error);
    if (!mtp_head) throw std::runtime_error(mtp_error);
    std::cout << "{\"event\":\"mtp_head\",\"rank\":" << o.rank << ",\"block\":" << mtp_head->block() << ",\"depth\":" << o.mtp_depth
              << ",\"weight_bytes\":" << mtp_head->weight_bytes() << ",\"buffer_bytes\":" << mtp_head->buffer_bytes()
              << ",\"load_s\":" << Num(mtp_head->load_s()) << ",\"own_embedding\":" << (mtp_head->own_embedding() ? "true" : "false")
              << ",\"data_file\":" << Str(mtp_data) << "}" << std::endl;
    std::vector<std::vector<std::uint32_t>> mtp_prompts;
    for (const auto& pr : ReadPrompts(o.id_files)) mtp_prompts.emplace_back(pr.begin(), pr.end());
    mtp_probe = std::make_unique<StagedMtpProbe>(*mtp_head, o.mtp_depth, w.config.hidden_size, std::move(mtp_prompts), o.rank);
    g_mtp_probe = mtp_probe.get();
  }
  flerp::qwen35moe::MoeDump dump;
  if (!o.dump_path.empty()) {
    dump.layers = o.dump_layers;
    dump.rows = o.dump_rows;
    flerp::qwen35moe::SetMoeDump(&dump);
    stage.SetDump(&dump);
  }
  const auto write_dump = [&] {
    if (o.dump_path.empty()) return;
    std::ofstream out(o.dump_path, std::ios::binary | std::ios::trunc);
    out.write(dump.bytes.data(), static_cast<std::streamsize>(dump.bytes.size()));
    std::cout << "{\"event\":\"moe_dump\",\"rank\":" << o.rank << ",\"path\":" << Str(o.dump_path) << ",\"records\":" << dump.records
              << ",\"bytes\":" << dump.bytes.size() << ",\"layers\":" << Ids(dump.layers) << ",\"rows\":" << Ids(dump.rows) << "}" << std::endl;
  };
  const auto& mc = model->GetConfig();
  const auto plan = RelayPlan(whole->Sha256(), cuts, mc.hidden_size, mc.vocab_size, stage.max_rows());
  std::unique_ptr<relay::Relay> r;
  const auto fail = [&](const std::string& why) {
    std::cout << "{\"event\":\"relay_failed\",\"rank\":" << o.rank << ",\"reason\":" << Str(why) << ",\"t_unix\":" << std::fixed << UnixNow()
              << std::defaultfloat << "}" << std::endl;
    return 3;
  };
  try {
    r = relay::ConnectRing(RelayConfig(o, o.world, o.rank, plan, mc, stage.max_rows()), Hosts(o.host), o.port,
                           std::chrono::milliseconds(o.setup_deadline_ms));
  } catch (const relay::RelayError& e) {
    return fail(e.what());
  }
  std::cout << "{\"event\":\"connected\",\"rank\":" << o.rank << ",\"world\":" << o.world << ",\"cuts\":" << Ids(cuts) << ",\"t_unix\":" << std::fixed
            << UnixNow() << std::defaultfloat << "}" << std::endl;
  if (!o.fault.empty()) {
    r->SetFault(relay::ParseFault(o.fault), [&](const relay::Fault& f) {
      std::cout << "{\"event\":\"fault_injected\",\"rank\":" << o.rank << ",\"fault\":" << Str(relay::FaultName(f.kind)) << ",\"ordinal\":" << f.ordinal
                << ",\"t_unix\":" << std::fixed << UnixNow() << std::defaultfloat << "}" << std::endl;
    });
  }
  try {
    if (o.rank != 0) {
      int session = 0;
      try {
        while (ServeOne(stage, *r, &session)) {
        }
      } catch (const relay::RelayError&) {
        throw;
      } catch (const std::exception& e) {
        r->Abort(std::string("rank ") + std::to_string(o.rank) + ": " + e.what());   // the ring fails now, not at a deadline
        throw;
      }
      write_dump();
      return 0;
    }
    const auto prompts = ReadPrompts(o.id_files);
    Pipeline p(stage, r.get(), {});
    std::vector<Options::RunSpec> specs = o.run_specs;
    if (specs.empty()) {
      for (const bool graph : o.graph_modes) specs.push_back({graph, o.reply_full, o.subchunks});
    }
    for (const auto& spec : specs) {
      const bool graph = spec.graph;
      p.SetReply(spec.full, spec.subchunks);
      p.SetDrafts(spec.drafts, spec.value, spec.lookup, spec.mtp_max);
      StageConfig c;
      c.world = o.world;
      c.cuts = cuts;
      c.graph = graph;
      c.text = std::to_string(o.world) + ":" + o.cuts + ":" + (graph ? "on" : "off");
      for (std::size_t i = 0; i < prompts.size(); ++i) {
        const Traced t = RunPrompt(p, prompts[i], o.max_tokens, graph, model->GetTokenizer());
        const auto& tm = stage.timing();
        double chunk = 0;
        for (const auto& ch : tm.chunks) chunk += ch.ms;
        // the other ranks' forwards are in their stage_timing events; the gate subtracts them
        std::cout << StagedJson("tcp", c, cuts, i, prompts[i].size(), t, p, "[" + stage.TimingJson() + "]", tm.decode_ms, chunk, 0.0) << std::endl;
      }
    }
    p.Close();
    write_dump();
  } catch (const relay::RelayError& e) {
    return fail(e.what());
  }
  return 0;
}

}  // namespace

int StagedMain(int argc, char** argv) {
  Options o;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      const auto v = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
        return argv[++i];
      };
      if (a == "--staged") o.mode = v();
      else if (a == "--model") o.model = v();
      else if (a == "--context") o.context = static_cast<std::uint32_t>(std::stoul(v()));
      else if (a == "--max-tokens") o.max_tokens = std::stoull(v());
      else if (a == "--ids-file") o.id_files.push_back(v());
      else if (a == "--single-reader") o.load.fill.single_reader = true;
      else if (a == "--pause-ms") o.load.fill.pause_ms = static_cast<std::uint32_t>(std::stoul(v()));
      else if (a == "--chunk-pause-ms") o.load.fill.chunk_pause_ms = static_cast<std::uint32_t>(std::stoul(v()));
      else if (a == "--data-file") { o.load.fill.data_path = v(); o.load.fill.single_reader = true; }
      else if (a == "--punch-behind") o.load.fill.punch_behind = true;
      else if (a == "--swarm-helper") o.load.fill.swarm.helper = v();
      else if (a == "--swarm-manifest") o.load.fill.swarm.manifest = v();
      else if (a == "--swarm-from") o.load.fill.swarm.sources = Hosts(v());
      else if (a == "--swarm-conns") o.load.fill.swarm.conns = static_cast<std::uint32_t>(std::stoul(v()));
      else if (a == "--configs") {
        std::istringstream in(v());
        for (std::string item; std::getline(in, item, ';');) {
          if (!item.empty()) o.configs.push_back(ParseConfig(item));
        }
      } else if (a == "--fault-config") o.fault_config = ParseConfig(v());
      else if (a == "--world") o.world = static_cast<std::uint32_t>(std::stoul(v()));
      else if (a == "--rank") o.rank = static_cast<std::uint32_t>(std::stoul(v()));
      else if (a == "--cuts") o.cuts = v();
      else if (a == "--port") o.port = static_cast<std::uint16_t>(std::stoul(v()));
      else if (a == "--host") o.host = v();
      else if (a == "--epoch") o.epoch = std::stoull(v());
      else if (a == "--deadline-ms") o.deadline_ms = static_cast<std::uint32_t>(std::stoul(v()));
      else if (a == "--setup-deadline-ms") o.setup_deadline_ms = static_cast<std::uint32_t>(std::stoul(v()));
      else if (a == "--graph-modes") {
        o.graph_modes.clear();
        std::istringstream in(v());
        for (std::string item; std::getline(in, item, ',');) o.graph_modes.push_back(item == "on");
      } else if (a == "--rows") {
        const std::string m = v();
        if (m != "pinned" && m != "pageable") throw std::invalid_argument("--rows is pinned or pageable");
        o.rows_mode = m == "pinned" ? RowsMode::kPinned : RowsMode::kPageable;
      } else if (a == "--reply") {
        const std::string m = v();
        if (m != "full" && m != "token") throw std::invalid_argument("--reply is full or token");
        o.reply_full = m == "full";
      } else if (a == "--prompt-subchunks") {
        o.subchunks = std::stoull(v());
        if (o.subchunks == 0) throw std::invalid_argument("--prompt-subchunks is at least 1");
      } else if (a == "--run-specs") {
        std::istringstream in(v());
        for (std::string item; std::getline(in, item, ',');) {
          if (item.empty()) continue;
          std::vector<std::string> parts;
          std::istringstream fields(item);
          for (std::string x; std::getline(fields, x, ':');) parts.push_back(x);
          if (parts.size() != 3 && parts.size() != 4) throw std::invalid_argument("--run-specs: want GRAPH:REPLY:SUB[:DRAFTS], got " + item);
          const std::string& g = parts[0];
          const std::string& rp = parts[1];
          const std::size_t sub = std::stoull(parts[2]);
          if ((g != "on" && g != "off") || (rp != "full" && rp != "token") || sub == 0) throw std::invalid_argument("--run-specs: bad " + item);
          Options::RunSpec spec{g == "on", rp == "full", sub};
          if (parts.size() == 4) {
            std::string d = parts[3];
            spec.value = !d.empty() && d[0] == 'v';
            if (spec.value) d = d.substr(1);
            if (const auto pp = d.find('p'); pp != std::string::npos) {   // KpL: MTP at most K, lookup at most L
              spec.lookup = true;
              spec.mtp_max = static_cast<std::uint32_t>(std::stoul(d.substr(0, pp)));
              const std::uint32_t cap = static_cast<std::uint32_t>(std::stoul(d.substr(pp + 1)));
              if (spec.mtp_max == 0 || cap == 0 || spec.mtp_max > 255) throw std::invalid_argument("--run-specs: bad KpL " + item);
              d = std::to_string(std::max(cap, spec.mtp_max));
            }
            spec.drafts = static_cast<std::uint32_t>(std::stoul(d));
            if (spec.drafts > 0 && spec.full) throw std::invalid_argument("--run-specs: drafts need reply token: " + item);
            if (spec.value && spec.drafts == 0) throw std::invalid_argument("--run-specs: v0: " + item);
          }
          o.run_specs.push_back(spec);
        }
      } else if (a == "--fault") o.fault = v();
      else if (a == "--dump") o.dump_path = v();
      else if (a == "--mtp-probe") o.mtp_depth = static_cast<std::uint32_t>(std::stoul(v()));
      else if (a == "--mtp-data-file") o.mtp_data_file = v();
      else if (a == "--draft-head") o.draft_head = true;
      else if (a == "--draft-skip") {
        const std::string m = v();
        if (m != "on" && m != "off") throw std::invalid_argument("--draft-skip is on or off");
        o.draft_skip = m == "on";
      } else if (a == "--draft-costs") {
        const std::string c = v();
        float b = 0, r = 0, m = 0;
        if (std::sscanf(c.c_str(), "%f,%f,%f", &b, &r, &m) != 3 || b <= 0 || r <= 0 || m < 0) throw std::invalid_argument("--draft-costs B,R,M (ms)");
        o.draft_costs = {b, r, m};
      }
      else if (a == "--dump-layers") o.dump_layers = ParseList(v());
      else if (a == "--dump-rows") o.dump_rows = ParseList(v());
      else if (a == "--help") {
        // every option this parser takes, one a line: gate/ornith_p5.py --plan checks its argv against this list
        for (const char* k : {"--staged", "--model", "--context", "--max-tokens", "--ids-file", "--single-reader", "--pause-ms",
                              "--chunk-pause-ms", "--data-file", "--punch-behind", "--swarm-helper", "--swarm-manifest", "--swarm-from",
                              "--swarm-conns", "--configs", "--fault-config", "--world", "--rank", "--cuts", "--port", "--host", "--epoch",
                              "--deadline-ms", "--setup-deadline-ms", "--graph-modes", "--fault", "--dump", "--dump-layers", "--dump-rows",
                              "--rows", "--reply", "--prompt-subchunks", "--run-specs", "--mtp-probe", "--mtp-data-file",
                              "--draft-head", "--draft-skip", "--draft-costs", "--help"})
          std::cout << k << "\n";
        return 0;
      }
      else throw std::invalid_argument("unknown option " + a);
    }
    if (o.model.empty() || (o.mode != "loopback" && o.mode != "tcp")) throw std::invalid_argument("--staged loopback|tcp and --model are required");
    if (o.mode == "loopback" && o.configs.empty()) throw std::invalid_argument("--configs is required");
    if ((o.mode == "loopback" || o.rank == 0) && o.id_files.empty()) throw std::invalid_argument("--ids-file is required");
    if (o.mode == "tcp" && (o.world < 2 || o.rank >= o.world)) throw std::invalid_argument("tcp: world >= 2 and rank < world");
    if (o.load.fill.swarm.enabled() && !o.load.fill.data_path.empty()) throw std::invalid_argument("--swarm-* and --data-file exclude each other");
    if (o.load.fill.punch_behind && o.load.fill.data_path.empty()) throw std::invalid_argument("--punch-behind needs --data-file");
    o.load.placed = true;
    std::string error;
    auto owner = gufo::core::GgufReader::OpenFile(o.model, &error);
    if (!owner) throw std::runtime_error("GGUF " + o.model + ": " + error);
    const std::shared_ptr<const gufo::core::GgufReader> reader(std::move(owner));
    return o.mode == "loopback" ? Loopback(o, reader) : TcpRank(o, reader);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    std::cout << "{\"event\":\"error\",\"rank\":" << o.rank << ",\"error\":" << Str(e.what()) << "}" << std::endl;
    return 1;
  }
}

}  // namespace flerp_tool
