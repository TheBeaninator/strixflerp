// SPDX-License-Identifier: MIT
// executor.flerp.qwen35moe_abi: rank 0's Pipeline and the stage loop of ranks 1..N-1 (stages.hpp). The stage
// mechanics are stage P2's (catalog/flerp/tool/qwen35moe_generate/src/flerp/qwen35moe_tool/stage_driver.cpp: Stage,
// ServeOne, View, RelayPlan), whose gate showed world 1, 2 and 4 give P1's logits bit for bit
// (gate/results/qwen36-35b-a3b.flerp-moe-stage/summary.md). What is new here: the calls arrive one at a time from
// the executor ABI instead of a traced generation, a stage process waits for them with no deadline between calls
// (a server is idle most of the time), and a stage process ends cleanly on SIGTERM. ORNITH-O3 step two
// (gate/ornith-o3-spec.toml): drafting sessions as stage_driver.cpp runs them (relay kSpecRound / kSpecReply, the
// last stage's flerp/qwen35moe/draft_round.hpp, a recurrent ring on every stage), with every frame run (kSpecAllRows)
// so a verify returns every row.
#include "flerp/qwen35moe_abi/stages.hpp"

#include <hip/hip_runtime.h>
#include <signal.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "flerp/gguf_placed/placement.hpp"
#include "flerp/qwen35moe/draft_round.hpp"
#include "flerp/qwen35moe/mtp_head.hpp"
#include "flerp/qwen35moe/stage.hpp"
#include "flerp/stage_relay/relay.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/families/qwen/text/tokenizer.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace flerp::qwen35moe_abi {
namespace {

namespace relay = flerp::stage_relay;
using gufo::tokenization::TokenId;
using Clock = std::chrono::steady_clock;

// A stage process waits this long for its caller's next call (a server's idle time): in effect no deadline.
constexpr std::chrono::milliseconds kIdle = std::chrono::hours(24 * 365);

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
std::string Ids(const std::vector<std::uint32_t>& v) {
  std::string o = "[";
  for (std::size_t i = 0; i < v.size(); ++i) o += (i ? "," : "") + std::to_string(v[i]);
  return o + "]";
}
double UnixNow() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }
// Event time stamps in ms: Num's %.6g printed every t_unix as 1.7945e+09 (FLERP-P5 run 11121559's logs had no times).
std::string UnixText() {
  char b[40];
  std::snprintf(b, sizeof(b), "%.3f", UnixNow());
  return b;
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
  if (!f.empty()) std::memcpy(f.data(), b.data(), f.size() * sizeof(float));
  return f;
}

std::uint32_t ParseU32(const std::string& key, const std::string& v) {
  std::size_t used = 0;
  unsigned long long x = 0;
  try {
    x = std::stoull(v, &used);
  } catch (const std::exception&) {
    used = 0;
  }
  if (used != v.size() || v.empty() || x > 0xFFFFFFFFULL) throw std::invalid_argument(key + " must be a whole number, got '" + v + "'");
  return static_cast<std::uint32_t>(x);
}
bool ParseOnOff(const std::string& key, const std::string& v) {
  if (v != "on" && v != "off") throw std::invalid_argument(key + " must be on or off, got '" + v + "'");
  return v == "on";
}
std::vector<std::uint32_t> ParseList(const std::string& key, const std::string& s) {
  std::vector<std::uint32_t> out;
  std::istringstream in(s);
  for (std::string item; std::getline(in, item, ',');) out.push_back(ParseU32(key, item));
  return out;
}

struct Peers {
  bool loopback{false};
  std::vector<std::string> hosts;
  std::uint16_t port{0};
};
Peers ParsePeers(const std::string& s) {
  Peers p;
  if (s == "loopback") {
    p.loopback = true;
    return p;
  }
  const auto colon = s.rfind(':');
  if (colon == std::string::npos || colon == 0) throw std::invalid_argument("peers must be loopback or HOST[,HOST...]:PORT, got '" + s + "'");
  const std::uint32_t port = ParseU32("peers port", s.substr(colon + 1));
  if (port == 0 || port > 65000) throw std::invalid_argument("peers: port out of range");
  p.port = static_cast<std::uint16_t>(port);
  std::istringstream in(s.substr(0, colon));
  for (std::string h; std::getline(in, h, ',');) {
    if (h.empty()) throw std::invalid_argument("peers: an empty host");
    p.hosts.push_back(h);
  }
  return p;
}

// --- one stage: an unchanged executor.flerp.qwen35moe executor over a block range (stage_driver.cpp's Stage) -------
struct Out {
  TokenId token{0};
  std::vector<float> rows;     // the stage's output rows (not the last stage)
  std::vector<float> logits;   // the last stage, when the call asked for the full row
};

class Stage {
 public:
  Stage(std::uint32_t rank, std::uint32_t world, std::shared_ptr<const gufo::hip::QwenGpuModel> model, std::uint32_t context)
      : world_(world), model_(std::move(model)) {
    std::string error;
    exec_ = gufo::hip::QwenGpuExecutor::Create(model_, &error, context);
    if (!exec_) throw std::runtime_error("stage " + std::to_string(rank) + ": executor: " + error);
    const auto& w = model_->GetWeights();
    hidden_ = w.config.hidden_size;
    max_rows_ = exec_->GetMaxPromptBatch();
    first_ = rank == 0;
    last_ = rank + 1 == world;
    if (first_ != (w.BlockBegin() == 0) || last_ != w.HasHead()) {
      throw std::runtime_error("stage " + std::to_string(rank) + ": blocks [" + std::to_string(w.BlockBegin()) + ", " +
                               std::to_string(w.BlockEnd()) + ") do not fit its place in the chain");
    }
    if (world_ > 1) {
      if (!first_) HipOk(hipMalloc(&d_in_, max_rows_ * hidden_ * sizeof(float)), "stage input");
      if (!last_) HipOk(hipMalloc(&d_out_, max_rows_ * hidden_ * sizeof(float)), "stage output");
      flerp::qwen35moe::BindStageIo(&w, {.d_input = d_in_, .d_output = d_out_});
    }
  }
  ~Stage() {
    flerp::qwen35moe::ReleaseRecurrentRing(exec_.get());
    exec_.reset();
    if (world_ > 1) flerp::qwen35moe::UnbindStageIo(&model_->GetWeights());
    if (d_in_) (void)hipFree(d_in_);
    if (d_out_) (void)hipFree(d_out_);
  }
  Stage(const Stage&) = delete;
  Stage& operator=(const Stage&) = delete;

  // A prompt chunk (decode = false) or a decode step. `tokens` matter on the first stage only; `input` (rows x
  // hidden) on the others. A decode step asks the logits on every stage, as stage P2 does: that keeps the step on
  // the captured graph; a stage without the head answers token 0.
  Out Run(bool decode, const std::vector<TokenId>& tokens, std::uint32_t start_pos, std::size_t rows, bool want, bool full,
          const float* input) {
    Out out;
    if (rows == 0 || rows > max_rows_ || (decode && rows != 1)) throw std::runtime_error("stage: bad row count " + std::to_string(rows));
    if (first_ && tokens.size() != rows) throw std::runtime_error("stage: token count differs from the rows");
    if (!first_) HipOk(hipMemcpy(d_in_, input, rows * hidden_ * sizeof(float), hipMemcpyHostToDevice), "copy into the stage input");
    const std::vector<TokenId> zeros(first_ ? 0 : rows, 0);
    const std::vector<TokenId>& ids = first_ ? tokens : zeros;
    if (decode) out.token = exec_->ForwardToken(ids.at(0), start_pos, true);
    else out.token = exec_->ForwardPromptBatch(std::span<const TokenId>(ids.data(), rows), start_pos, want);
    HipOk(hipDeviceSynchronize(), "synchronize");
    if (!last_) {
      out.rows.resize(rows * hidden_);
      HipOk(hipMemcpy(out.rows.data(), d_out_, out.rows.size() * sizeof(float), hipMemcpyDeviceToHost), "copy of the stage output");
    }
    if (last_ && want && full) {
      const auto l = exec_->CopyLastLogits();
      out.logits.assign(l.begin(), l.end());
    }
    return out;
  }

  // drafts > 0: the session's decode steps run on a ring of drafts + 1 recurrent states (stage.hpp).
  void Reset(std::uint32_t drafts = 0) {
    exec_->Reset();
    flerp::qwen35moe::SetRecurrentRing(exec_.get(), drafts > 0 ? drafts + 1 : 0);
  }
  // ORNITH-PCACHE: the stage's KV and recurrent state at `positions` kept apart from the arena (the executor's own
  // snapshot, device to device), and put back after a reset. One snapshot; a save replaces it.
  void SnapSave(std::uint32_t positions) {
    HipOk(hipDeviceSynchronize(), "synchronize before the snapshot");
    snap_.reset();
    snap_ = exec_->SaveSnapshot(positions);
    snap_pos_ = positions;
  }
  void SnapRestore(std::uint32_t drafts, std::uint32_t positions) {
    if (!snap_ || snap_pos_ != positions)
      throw std::runtime_error("stage: no snapshot at " + std::to_string(positions) + " positions (held " + std::to_string(snap_ ? snap_pos_ : 0) + ")");
    exec_->RestoreSnapshot(*snap_);
    flerp::qwen35moe::SetRecurrentRing(exec_.get(), drafts > 0 ? drafts + 1 : 0);
    HipOk(hipDeviceSynchronize(), "synchronize after the snapshot restore");
  }
  [[nodiscard]] bool last() const { return last_; }
  [[nodiscard]] std::size_t max_rows() const { return max_rows_; }

 private:
  std::uint32_t world_;
  std::shared_ptr<const gufo::hip::QwenGpuModel> model_;
  std::unique_ptr<gufo::hip::QwenGpuExecutor> exec_;
  std::unique_ptr<gufo::hip::QwenGpuSnapshot> snap_;
  std::uint32_t snap_pos_{0};
  std::size_t hidden_{0}, max_rows_{0};
  bool first_{false}, last_{false};
  float* d_in_{nullptr};
  float* d_out_{nullptr};
};

// The last stage's drafting side (a stage process given draft_head=on): the head's rows come through its stage output
// buffer; on while the session drafts.
flerp::qwen35moe::DraftRound* g_draft = nullptr;
flerp::qwen35moe::DraftCosts g_draft_costs;   // draft_costs (the value-aware rule)
float* g_draft_rows = nullptr;
bool g_draft_on = false;
std::size_t g_hidden = 0;

void SendSpecReply(relay::Relay& r, const relay::FrameHeader& h, const flerp::qwen35moe::DraftRound& d) {
  const auto& kept = d.emitted();
  const auto& drafts = d.drafts();
  const auto& drawn = d.drawn();
  std::vector<std::uint32_t> w{static_cast<std::uint32_t>(kept.size()), static_cast<std::uint32_t>(drafts.size()),
                               static_cast<std::uint32_t>(drawn.size())};
  w.insert(w.end(), kept.begin(), kept.end());
  w.insert(w.end(), drafts.begin(), drafts.end());
  w.insert(w.end(), drawn.begin(), drawn.end());
  r.Send(relay::Op::kReply, h.start_pos, static_cast<std::uint32_t>(w.size()), relay::kSpecReply | relay::kWantLogits, kept.back(),
         {reinterpret_cast<const std::uint8_t*>(w.data()), w.size() * sizeof(std::uint32_t)}, h.seq_slot);
}

// A rank > 0: one frame in, its stage, one frame out (stage_driver.cpp's ServeOne). Control frames are forwarded
// (or answered, by the last rank) before this rank acts on them. Returns false after CLOSE.
bool ServeOne(Stage& stage, relay::Relay& r, std::chrono::milliseconds header_deadline) {
  const relay::Frame f = r.Recv(header_deadline);
  const auto& h = f.header;
  switch (f.op()) {
    case relay::Op::kPrefillChunk:
    case relay::Op::kDecode: {
      const bool decode = f.op() == relay::Op::kDecode;
      const bool want = (h.flags & relay::kWantLogits) != 0U;
      const bool full = (h.flags & relay::kFullLogits) != 0U;
      const std::size_t row_bytes = static_cast<std::size_t>(h.rows) * g_hidden * sizeof(float);
      const bool with_ids = !decode && (h.flags & relay::kWithIds) != 0U;
      const auto input = Floats(std::vector<std::uint8_t>(f.payload.begin(), f.payload.begin() + static_cast<std::ptrdiff_t>(row_bytes)));
      const bool spec = stage.last() && g_draft_on;
      if (decode && (h.flags & relay::kSpecRound) != 0U) {
        if (!stage.last()) {
          const Out o = stage.Run(true, {}, h.start_pos, 1, want, false, input.data());
          r.Send(f.op(), h.start_pos, h.rows, h.flags, h.token, Bytes(o.rows), h.seq_slot);
          return true;
        }
        if (!spec) throw std::runtime_error("a draft frame, and this session does not draft");
        const std::uint32_t j = relay::SpecFrameIndex(h.flags), frames = relay::SpecFrameCount(h.flags);
        if (j == 0) g_draft->BeginRound(h.start_pos, frames);
        const Out o = stage.Run(true, {}, h.start_pos, 1, true, false, input.data());   // every frame: kSpecAllRows
        g_draft->Frame(j, h.token, static_cast<std::uint32_t>(o.token), g_draft_rows);
        if (g_draft->RoundDone()) SendSpecReply(r, h, *g_draft);
        return true;
      }
      if (spec && decode) throw std::runtime_error("a plain decode frame in a drafting session");
      const Out o = stage.Run(decode, {}, h.start_pos, h.rows, want, full, input.data());
      if (spec) {
        if (!with_ids) throw std::runtime_error("a prompt chunk without its ids in a drafting session");
        std::vector<std::uint32_t> ids(h.rows);
        std::memcpy(ids.data(), f.payload.data() + row_bytes, ids.size() * sizeof(std::uint32_t));
        g_draft->PromptRows(h.start_pos, h.rows, ids.data(), g_draft_rows);
        if (want) {
          g_draft->Frontier(h.start_pos + h.rows, static_cast<std::uint32_t>(o.token), g_draft_rows + (h.rows - 1) * g_hidden);
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
        const auto rows = Bytes(o.rows);
        std::vector<std::uint8_t> out(rows.begin(), rows.end());
        out.insert(out.end(), f.payload.begin() + static_cast<std::ptrdiff_t>(row_bytes), f.payload.end());
        r.Send(f.op(), h.start_pos, h.rows, h.flags, h.token, out, h.seq_slot);
      } else {
        r.Send(f.op(), h.start_pos, h.rows, h.flags, h.token, Bytes(o.rows), h.seq_slot);
      }
      return true;
    }
    case relay::Op::kReset:
    case relay::Op::kClose: {
      if (stage.last()) r.Send(relay::Op::kReply, 0, 1, 0, 0, {}, h.seq_slot);
      else r.Send(f.op(), h.start_pos, 0, h.flags, h.token, {}, h.seq_slot);
      if (f.op() == relay::Op::kClose) return false;
      if ((h.flags & relay::kSnapSave) != 0U) {   // ORNITH-PCACHE: keep the stage's state at h.start_pos positions, no reset
        stage.SnapSave(h.start_pos);
        if (stage.last() && g_draft_on) g_draft->SaveEdge();
        return true;
      }
      flerp::qwen35moe::SetDecodeGraph((h.flags & relay::kGraphOff) == 0U);
      const std::uint32_t drafts = (h.flags & relay::kSpecRound) != 0U ? h.token : 0U;
      const bool restore = (h.flags & relay::kSnapRestore) != 0U;
      if (restore) stage.SnapRestore(drafts, h.start_pos);
      else stage.Reset(drafts);
      g_draft_on = false;
      if (drafts > 0 && stage.last()) {
        if (g_draft == nullptr) throw std::runtime_error("a drafting session, and the last stage has no draft head (draft_head=on)");
        flerp::qwen35moe::DraftConfig dc;
        dc.max_draft = drafts;
        dc.value_aware = (h.flags & relay::kSpecValue) != 0U;
        dc.costs = g_draft_costs;
        g_draft->Begin(dc);
        g_draft_on = true;
        if (restore && !g_draft->RestoreEdge(h.start_pos)) throw std::runtime_error("a snapshot restore, and the draft head saved no prompt edge at " + std::to_string(h.start_pos));
      }
      return true;
    }
    default:
      r.Abort(std::string("unexpected ") + relay::OpName(f.op()));
      throw relay::RelayError("rank " + std::to_string(r.config().rank) + ": unexpected " + relay::OpName(f.op()));
  }
}

// --- loading ------------------------------------------------------------------------------------------------------
std::shared_ptr<const gufo::core::GgufReader> OpenReader(const std::string& path) {
  std::string error;
  auto owner = gufo::core::GgufReader::OpenFile(path, &error);
  if (!owner) throw std::runtime_error("GGUF " + path + ": " + error);
  return std::shared_ptr<const gufo::core::GgufReader>(std::move(owner));
}

std::vector<std::uint32_t> ResolveCuts(const gufo::core::GgufReader& reader, const Options& o, std::uint32_t blocks) {
  if (o.placement != "even") {
    const auto given = ParseList("placement", o.placement);
    if (given.size() != o.world + 1 || given.front() != 0 || given.back() != blocks)
      throw std::invalid_argument("placement: " + std::to_string(o.world + 1) + " cuts from 0 to " + std::to_string(blocks) + " are needed");
    for (std::size_t i = 1; i < given.size(); ++i)
      if (given[i] <= given[i - 1]) throw std::invalid_argument("placement: cuts must increase");
    return given;
  }
  std::string error;
  const auto plan = flerp::gguf_placed::StagePlan(reader, o.world, std::nullopt, &error);
  if (!plan) throw std::runtime_error("placement: " + error);
  return plan->cuts;
}

// What every rank hashes into its frames: the model's whole-file plan, the cuts, the shapes, the context.
std::array<std::uint8_t, 32> RelayPlan(const std::string& whole_plan_sha, const std::vector<std::uint32_t>& cuts, std::uint32_t hidden,
                                       std::uint32_t vocab, std::size_t max_rows, std::uint32_t context) {
  std::string text = "flerp-qwen35moe-abi v1\nmodel-plan " + whole_plan_sha + "\ncuts";
  for (const auto c : cuts) text += " " + std::to_string(c);
  text += "\nhidden " + std::to_string(hidden) + "\nvocab " + std::to_string(vocab) + "\nmax_rows " + std::to_string(max_rows) +
          "\ncontext " + std::to_string(context) + "\n";
  return relay::PlanFromHex(Sha(text.data(), text.size()));
}

relay::Config RelayConfig(const Options& o, std::uint32_t rank, const std::array<std::uint8_t, 32>& plan, const gufo::core::ModelConfig& mc,
                          std::size_t max_rows) {
  relay::Config c;
  c.world = o.world;
  c.rank = rank;
  c.epoch = o.epoch;
  c.hidden = mc.hidden_size;
  c.vocab = mc.vocab_size;
  c.max_rows = static_cast<std::uint32_t>(max_rows);
  c.plan = plan;
  c.deadline = std::chrono::milliseconds(o.deadline_ms);
  return c;
}

// The rank's part through format.flerp.gguf_placed: the whole model (`range` empty) or one block range as a world-1
// range plan (stage_driver.cpp's TcpRank), by the single direct reader when asked.
std::shared_ptr<const gufo::hip::QwenGpuModel> Load(const std::shared_ptr<const gufo::core::GgufReader>& reader, const Options& o,
                                                    std::optional<std::pair<std::uint32_t, std::uint32_t>> range, std::uint32_t rank,
                                                    const char* mode) {
  flerp::qwen35moe::LoadOptions load;
  load.placed = true;
  load.world = 1;
  load.rank = 0;
  if (range) {
    load.has_range = true;
    load.block_begin = range->first;
    load.block_end = range->second;
  }
  load.fill.single_reader = o.single_reader;
  load.fill.pause_ms = o.single_reader ? o.pause_ms : 0;
  if (!o.data_file.empty()) {
    load.fill.data_path = o.data_file;
    load.fill.single_reader = true;
    load.fill.pause_ms = 0;   // a worker-local copy: nobody else's disk
  }
  if (!o.swarm_helper.empty()) {
    load.fill.swarm.helper = o.swarm_helper;
    load.fill.swarm.manifest = o.swarm_manifest;
    load.fill.swarm.conns = o.swarm_conns;
    for (std::size_t a = 0, b; a <= o.swarm_from.size(); a = b + 1) {
      b = o.swarm_from.find(',', a);
      if (b == std::string::npos) b = o.swarm_from.size();
      if (b > a) load.fill.swarm.sources.push_back(o.swarm_from.substr(a, b - a));
    }
  }
  flerp::qwen35moe::SetLoadOptions(load);
  const auto t0 = Clock::now();
  std::string error;
  auto model = gufo::hip::QwenGpuModel::CreateFromGguf(reader, &error);
  if (!model) throw std::runtime_error("load: " + error);
  const auto& rep = flerp::qwen35moe::LastLoadReport();
  Emit(std::string("{\"event\":\"loaded\",\"mode\":") + Str(mode) + ",\"rank\":" + std::to_string(rank) + ",\"load_s\":" +
       Num(std::chrono::duration<double>(Clock::now() - t0).count()) + ",\"placed\":{\"plan_sha256\":\"" + rep.plan_sha256 + "\",\"blocks\":[" +
       std::to_string(rep.block_begin) + "," + std::to_string(rep.block_end) + "],\"allocated_bytes\":" + std::to_string(rep.allocated_bytes) +
       ",\"plan_device_bytes\":" + std::to_string(rep.plan_device_bytes) + ",\"tensor_bytes\":" + std::to_string(rep.tensor_bytes) +
       ",\"fill_s\":" + Num(rep.fill_s) + ",\"pause_s\":" + Num(rep.pause_s) + ",\"single_reader\":" + (rep.single_reader ? "true" : "false") +
       ",\"direct_io\":" + (rep.direct_io ? "true" : "false") + ",\"swarm\":" + (rep.swarm ? "true" : "false") +
       ",\"swarm_bytes\":" + std::to_string(rep.swarm_bytes) + ",\"data_path\":" + Str(rep.data_path) + ",\"hip_free_before\":" + std::to_string(rep.hip_free_before) +
       ",\"hip_free_after\":" + std::to_string(rep.hip_free_after) + ",\"hip_total\":" + std::to_string(rep.hip_total) + "},\"t_unix\":" +
       UnixText() + "}");
  return model;
}

// The weights of the one load, re-ranged: the same device tensors, no regions of its own (stage_driver.cpp's View).
std::shared_ptr<const gufo::hip::QwenGpuModel> View(const std::shared_ptr<const gufo::core::GgufReader>& reader,
                                                    const std::shared_ptr<const gufo::hip::QwenGpuModel>& base, std::uint32_t begin,
                                                    std::uint32_t end) {
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

std::string Basename(const std::string& p) {
  const auto s = p.rfind('/');
  return s == std::string::npos ? p : p.substr(s + 1);
}

std::atomic<relay::Relay*> g_stage_relay{nullptr};

}  // namespace

// --- options ------------------------------------------------------------------------------------------------------
bool SetKey(Options& o, const std::string& key, const std::string& value, std::string* error) {
  try {
    if (key == "weights") o.weights = value;
    else if (key == "context") o.context = ParseU32(key, value);
    else if (key == "world") o.world = ParseU32(key, value);
    else if (key == "rank") o.rank = ParseU32(key, value);
    else if (key == "peers") o.peers = value;
    else if (key == "placement") o.placement = value;
    else if (key == "epoch") o.epoch = ParseU32(key, value);
    else if (key == "deadline_ms") o.deadline_ms = ParseU32(key, value);
    else if (key == "setup_deadline_ms") o.setup_deadline_ms = ParseU32(key, value);
    else if (key == "graph") o.graph = ParseOnOff(key, value);
    else if (key == "single_reader") o.single_reader = ParseOnOff(key, value);
    else if (key == "pause_ms") o.pause_ms = ParseU32(key, value);
    else if (key == "data_file") o.data_file = value;
    else if (key == "swarm_helper") o.swarm_helper = value;
    else if (key == "swarm_manifest") o.swarm_manifest = value;
    else if (key == "swarm_from") o.swarm_from = value;
    else if (key == "swarm_conns") o.swarm_conns = ParseU32(key, value);
    else if (key == "drafts") o.drafts = ParseU32(key, value);
    else if (key == "draft_head") o.draft_head = ParseOnOff(key, value);
    else if (key == "draft_value") o.draft_value = ParseOnOff(key, value);
    else if (key == "draft_costs") o.draft_costs = value;
    else if (key == "mtp_data_file") o.mtp_data_file = value;
    else if (key == "subchunks") o.subchunks = value == "auto" ? 0U : ParseU32(key, value);
    else if (key == "subchunk_min_rows") o.subchunk_min_rows = ParseU32(key, value);
    else if (key == "in_flight") o.in_flight = ParseU32(key, value);
    else if (key == "reuse_tail") o.reuse_tail = ParseU32(key, value);
    else if (key == "reuse_grid") o.reuse_grid = ParseU32(key, value);
    else {
      if (error) *error = "unknown key '" + key + "' (weights, context, world, rank, peers, placement, epoch, deadline_ms, setup_deadline_ms, graph, single_reader, pause_ms, data_file, swarm_helper, swarm_manifest, swarm_from, swarm_conns, drafts, draft_value, draft_head, draft_costs, mtp_data_file, reuse_tail, reuse_grid, subchunks, subchunk_min_rows, in_flight)";
      return false;
    }
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return false;
  }
  return true;
}

std::string CheckOptions(const Options& o) {
  if (o.weights.empty()) return "weights (the GGUF file) is required";
  if (!o.swarm_helper.empty() && (o.swarm_manifest.empty() || o.swarm_from.empty())) return "swarm_helper needs swarm_manifest and swarm_from";
  if (!o.swarm_helper.empty() && !o.data_file.empty()) return "swarm_* and data_file exclude each other";
  if (o.context < 16) return "context must be at least 16";
  if (o.world < 1 || o.world > 64) return "world must be 1 to 64";
  if (o.rank >= o.world) return "rank must be below world";
  if (o.deadline_ms == 0) return "deadline_ms must be above 0";
  if (o.drafts > 64) return "drafts: at most 64 a round";
  if (o.drafts > 0 && (o.rank != 0 || o.world < 2)) return "drafts is rank 0's key, and needs world >= 2 over TCP (the last stage drafts)";
  if (o.draft_head && (o.rank == 0 || o.rank + 1 != o.world)) return "draft_head is the last stage process's key";
  if (o.draft_value && o.drafts == 0) return "draft_value needs drafts (rank 0)";
  if (!o.draft_costs.empty() && !o.draft_head) return "draft_costs needs draft_head (the last stage process)";
  if ((o.reuse_tail > 0 || o.reuse_grid > 0) && o.rank != 0) return "reuse_tail and reuse_grid are rank 0's keys";
  if (o.reuse_grid > 0 && o.reuse_tail == 0) return "reuse_grid needs reuse_tail";
  if (o.world == 1) {
    if (!o.peers.empty()) return "world 1 has no peers";
    return "";
  }
  if (o.peers.empty()) return "world > 1 needs peers (loopback or HOST[,HOST...]:PORT)";
  try {
    const Peers p = ParsePeers(o.peers);
    if (p.loopback && o.rank != 0) return "a loopback runs every stage in rank 0's process: no stage process has a loopback";
    if (!p.loopback && p.hosts.size() != 1 && p.hosts.size() != o.world) return "peers: one HOST, or one a rank";
  } catch (const std::exception& e) {
    return e.what();
  }
  if (o.placement != "even") {
    try {
      if (ParseList("placement", o.placement).size() != o.world + 1) return "placement: world + 1 cuts are needed";
    } catch (const std::exception& e) {
      return e.what();
    }
  }
  return "";
}

void Emit(const std::string& json) {
  std::fputs(json.c_str(), stdout);
  std::fputc('\n', stdout);
  std::fflush(stdout);
}

// --- rank 0 -------------------------------------------------------------------------------------------------------
struct Pipeline::Impl {
  std::shared_ptr<const gufo::core::GgufReader> reader;
  std::shared_ptr<const gufo::hip::QwenGpuModel> base;
  std::vector<std::shared_ptr<const gufo::hip::QwenGpuModel>> views;
  std::vector<std::unique_ptr<Stage>> stages;          // [0] is rank 0; a loopback's other ranks follow
  std::vector<std::unique_ptr<relay::Relay>> relays;   // loopback: one a rank; TCP: rank 0's
  relay::Relay* r0{nullptr};
  ~Impl() {
    // executors before the weights they run over; the views before the load they are views of
    stages.clear();
    relays.clear();
    views.clear();
    base.reset();
  }
};

Pipeline::Pipeline(const Options& o) : opt_(o), impl_(std::make_unique<Impl>()) {
  if (const std::string e = CheckOptions(o); !e.empty()) throw std::invalid_argument(e);
  if (o.rank != 0) throw std::invalid_argument("flerp_open is rank 0; ranks 1.." + std::to_string(o.world - 1) + " run flerp-qwen35moe-stage");
  Impl& I = *impl_;
  I.reader = OpenReader(o.weights);
  std::string error;
  const auto whole = flerp::gguf_placed::StagePlan(*I.reader, 1, std::nullopt, &error);
  if (!whole) throw std::runtime_error("placement: " + error);
  const std::string whole_sha = whole->Sha256();
  const auto cuts = ResolveCuts(*I.reader, o, whole->blocks);
  const Peers peers = o.world > 1 ? ParsePeers(o.peers) : Peers{};
  const bool tcp = o.world > 1 && !peers.loopback;
  I.base = Load(I.reader, o, tcp ? std::optional<std::pair<std::uint32_t, std::uint32_t>>({cuts[0], cuts[1]}) : std::nullopt, 0,
                o.world == 1 ? "world1" : tcp ? "tcp" : "loopback");
  device_bytes_ = static_cast<std::int64_t>(flerp::qwen35moe::LastLoadReport().allocated_bytes);
  const auto& mc = I.base->GetConfig();
  if (o.world > 1 && peers.loopback) {
    for (std::uint32_t r = 0; r < o.world; ++r) {
      I.views.push_back(View(I.reader, I.base, cuts[r], cuts[r + 1]));
      I.stages.push_back(std::make_unique<Stage>(r, o.world, I.views.back(), o.context));
    }
  } else {
    I.stages.push_back(std::make_unique<Stage>(0, o.world, I.base, o.context));
  }
  max_rows_ = I.stages[0]->max_rows();
  vocab_ = mc.vocab_size;
  g_hidden = mc.hidden_size;
  if (o.drafts > 0 && !tcp) throw std::invalid_argument("drafts need world >= 2 over TCP (the last stage process drafts)");
  const auto plan = RelayPlan(whole_sha, cuts, mc.hidden_size, mc.vocab_size, max_rows_, o.context);
  if (o.world > 1 && peers.loopback) {
    std::vector<relay::Config> cfgs;
    for (std::uint32_t r = 0; r < o.world; ++r) cfgs.push_back(RelayConfig(o, r, plan, mc, max_rows_));
    I.relays = relay::LoopbackRing(cfgs);
    for (auto& r : I.relays)
      if (r->poisoned()) throw std::runtime_error("loopback handshake: " + r->reason());
    I.r0 = I.relays[0].get();
  } else if (tcp) {
    Emit("{\"event\":\"waiting\",\"rank\":0,\"world\":" + std::to_string(o.world) + ",\"cuts\":" + Ids(cuts) + ",\"peers\":" + Str(o.peers) +
         ",\"t_unix\":" + UnixText() + "}");
    I.relays.push_back(relay::ConnectRing(RelayConfig(o, 0, plan, mc, max_rows_), peers.hosts, peers.port,
                                          std::chrono::milliseconds(o.setup_deadline_ms)));
    I.r0 = I.relays[0].get();
    Emit("{\"event\":\"connected\",\"rank\":0,\"world\":" + std::to_string(o.world) + ",\"t_unix\":" + UnixText() + "}");
  }
  std::string cut_text;
  for (std::size_t i = 0; i < cuts.size(); ++i) cut_text += (i ? "," : "") + std::to_string(cuts[i]);
  describe_ = "executor.flerp.qwen35moe_abi: " + Basename(o.weights) + ", " + std::to_string(whole->blocks) + " blocks, world " +
              std::to_string(o.world) + (o.world == 1 ? "" : peers.loopback ? " (loopback, one process)" : " (TCP " + o.peers + ")") +
              ", cuts " + cut_text + ", context " + std::to_string(o.context) + ", decode graph " + (o.graph ? "on" : "off") +
              ", one seq, greedy tokens or full rows" +
              (o.drafts > 0 ? ", MTP drafts held by the last stage (at most " + std::to_string(o.drafts) + " a round)" : std::string());
  fingerprint_ = "qwen35moe-abi v1; model plan " + whole_sha + "; cuts " + cut_text + "; context " + std::to_string(o.context);
}

Pipeline::~Pipeline() {
  Close();
  impl_.reset();
}

void Pipeline::Fail(const std::string& why) {
  if (!failed_) {
    failed_ = true;
    reason_ = why;
    if (impl_->r0 != nullptr) impl_->r0->Abort(why);
  }
  throw std::runtime_error(why);
}

// Loopback: the other ranks' loops, one frame each, in ring order. A rank whose relay fails is poisoned and its
// neighbours' receives then fail too; rank 0 sees it on its own receive.
void Pipeline::Pump() {
  Impl& I = *impl_;
  if (I.relays.size() < 2) return;
  for (std::size_t r = 1; r < I.stages.size(); ++r) {
    try {
      (void)ServeOne(*I.stages[r], *I.relays[r], std::chrono::milliseconds(opt_.deadline_ms));
    } catch (const relay::RelayError&) {
    }
  }
}

Reply Pipeline::Forward(bool decode, std::span<const std::int32_t> ids, std::uint32_t start, bool want, bool full) {
  if (failed_) throw std::runtime_error("the stages failed earlier: " + reason_);
  Impl& I = *impl_;
  Reply rep;
  try {
    std::vector<TokenId> tok(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) tok[i] = static_cast<TokenId>(ids[i]);
    if (I.r0 == nullptr) {
      Out o = I.stages[0]->Run(decode, tok, start, ids.size(), want, full, nullptr);
      rep.token = static_cast<std::int32_t>(o.token);
      rep.logits = std::move(o.logits);
      return rep;
    }
    if (session_drafts_ > 0 && full) Fail("a drafting session returns tokens only");
    if (decode) {
      // nothing of a prompt may be in flight before a decode step
      Drain();
      Out o = I.stages[0]->Run(true, tok, start, 1, want, full, nullptr);
      if (session_drafts_ > 0) {
        // a drafting session: a decode step is a round of one frame
        const relay::Frame f = SendRound(tok[0], 0, start, o.rows);
        TakeSpec(f, nullptr);
        rep.token = static_cast<std::int32_t>(f.header.token);
        return rep;
      }
      const std::uint32_t flags = want ? relay::kWantLogits | (full ? relay::kFullLogits : 0U) : 0U;
      I.r0->Send(relay::Op::kDecode, start, 1, flags, 0, Bytes(o.rows));
      ++unread_;
      Pump();
      const relay::Frame f = ReadReply();
      if (want && full && f.payload.size() != static_cast<std::size_t>(vocab_) * sizeof(float)) Fail("rank 0: the reply carries no logits row");
      rep.token = static_cast<std::int32_t>(f.header.token);
      rep.logits = Floats(f.payload);
      return rep;
    }
    // A prompt forward (ORNITH-SUBCHUNK 6abf28258c): relay::SubchunkParts(rows) sub-chunks back to back, sizes differing
    // by at most one row; only the last may ask for the row. With in_flight > 0 over TCP the replies are left unread
    // until in_flight are, or until a forward wants its row: the front end's next prompt forward then finds the ring
    // still busy instead of drained. A drafting session's chunks carry their ids (the last stage's MTP head).
    const bool keep = opt_.in_flight > 0 && I.relays.size() < 2;
    if (!keep) Drain();
    const std::size_t rows = tok.size();
    const std::size_t parts = relay::SubchunkParts(rows, opt_.subchunks, opt_.subchunk_min_rows);
    std::size_t off = 0;
    for (std::size_t k = 0; k < parts; ++k) {
      if (keep && unread_ >= opt_.in_flight) (void)ReadReply();
      const std::size_t n = rows / parts + (k < rows % parts ? 1 : 0);
      const bool last = k + 1 == parts;
      const bool w = want && last;
      const std::vector<TokenId> sub(tok.begin() + static_cast<std::ptrdiff_t>(off), tok.begin() + static_cast<std::ptrdiff_t>(off + n));
      const auto pos = static_cast<std::uint32_t>(start + off);
      Out o = I.stages[0]->Run(false, sub, pos, n, w, full && last, nullptr);
      if (session_drafts_ > 0) {
        std::vector<std::uint8_t> payload(reinterpret_cast<const std::uint8_t*>(o.rows.data()),
                                          reinterpret_cast<const std::uint8_t*>(o.rows.data() + o.rows.size()));
        for (const TokenId id : sub) {
          const auto u = static_cast<std::uint32_t>(id);
          payload.insert(payload.end(), reinterpret_cast<const std::uint8_t*>(&u), reinterpret_cast<const std::uint8_t*>(&u) + sizeof(u));
        }
        I.r0->Send(relay::Op::kPrefillChunk, pos, static_cast<std::uint32_t>(n), (w ? relay::kWantLogits : 0U) | relay::kWithIds, 0, payload);
      } else {
        const std::uint32_t flags = w ? relay::kWantLogits | (full ? relay::kFullLogits : 0U) : 0U;
        I.r0->Send(relay::Op::kPrefillChunk, pos, static_cast<std::uint32_t>(n), flags, 0, Bytes(o.rows));
      }
      ++unread_;
      off += n;
    }
    if (!keep) {
      for (std::size_t k = 0; k < parts; ++k) Pump();   // loopback: each other rank serves one frame a pump
    }
    if (keep && !want) return rep;   // rows 0: nothing to give back; the replies are read later
    while (unread_ > 1) (void)ReadReply();
    const relay::Frame f = ReadReply();
    rep.token = static_cast<std::int32_t>(f.header.token);
    if ((f.header.flags & relay::kSpecReply) != 0U) {
      TakeSpec(f, nullptr);
    } else {
      if (want && full && f.payload.size() != static_cast<std::size_t>(vocab_) * sizeof(float)) Fail("rank 0: the reply carries no logits row");
      rep.logits = Floats(f.payload);
    }
  } catch (const std::exception& e) {
    Fail(e.what());
  }
  return rep;
}

relay::Frame Pipeline::ReadReply() {
  Impl& I = *impl_;
  relay::Frame f = I.r0->Recv();
  if (f.op() != relay::Op::kReply) Fail(std::string("rank 0: expected REPLY, got ") + relay::OpName(f.op()));
  if (unread_ > 0) --unread_;
  return f;
}

void Pipeline::Drain() {
  Impl& I = *impl_;
  while (I.r0 != nullptr && unread_ > 0) (void)ReadReply();
}

// A spec reply's words: [kept, drafts, frames, kept ids.., drafts.., drawn ids..]; the drafts become held().
void Pipeline::TakeSpec(const relay::Frame& f, Round* round) {
  const std::size_t n = f.payload.size() / sizeof(std::uint32_t);
  std::vector<std::uint32_t> w(n);
  std::memcpy(w.data(), f.payload.data(), n * sizeof(std::uint32_t));
  if (n < 3 || w[0] == 0 || std::size_t{3} + w[0] + w[1] + w[2] != n) Fail("rank 0: a malformed spec reply");
  const auto a = w.begin() + 3, b = a + w[0], c = b + w[1];
  held_.assign(b, c);
  if (round != nullptr) {
    round->kept.assign(a, b);
    round->drafts.assign(b, c);
    round->drawn.assign(c, w.end());
  }
}

// Frames of one round: `pending` and the first `held` held drafts at start.. (o: rank 0's run of frame 0, done).
relay::Frame Pipeline::SendRound(std::int32_t pending, std::uint32_t held, std::uint32_t start, const std::vector<float>& rows0) {
  Impl& I = *impl_;
  const std::uint32_t frames = held + 1;
  for (std::uint32_t j = 0; j < frames; ++j) {
    const std::int32_t id = j == 0 ? pending : held_[j - 1];
    if (j == 0) {
      I.r0->Send(relay::Op::kDecode, start, 1, relay::kWantLogits | relay::SpecFrameFlags(0, frames), static_cast<std::uint32_t>(id), Bytes(rows0));
      continue;
    }
    const Out o = I.stages[0]->Run(true, {static_cast<TokenId>(id)}, start + j, 1, true, false, nullptr);
    I.r0->Send(relay::Op::kDecode, start + j, 1, relay::kWantLogits | relay::SpecFrameFlags(j, frames), static_cast<std::uint32_t>(id),
               Bytes(o.rows));
  }
  relay::Frame f = I.r0->Recv();
  if (f.op() != relay::Op::kReply || (f.header.flags & relay::kSpecReply) == 0U) Fail("rank 0: expected a spec REPLY");
  return f;
}

Round Pipeline::Verify(std::int32_t pending, std::uint32_t held, std::uint32_t start) {
  if (failed_) throw std::runtime_error("the stages failed earlier: " + reason_);
  try {
    Drain();
  } catch (const std::exception& e) {
    Fail(e.what());
  }
  if (session_drafts_ == 0) throw std::logic_error("verify outside a drafting session");
  if (held > held_.size()) throw std::logic_error("verify of " + std::to_string(held) + " drafts, " + std::to_string(held_.size()) + " held");
  Impl& I = *impl_;
  Round round;
  try {
    const Out o0 = I.stages[0]->Run(true, {static_cast<TokenId>(pending)}, start, 1, true, false, nullptr);
    const relay::Frame f = SendRound(pending, held, start, o0.rows);
    TakeSpec(f, &round);
    if (round.drawn.size() != held + 1) Fail("rank 0: a verify reply without every frame's id");
  } catch (const std::exception& e) {
    Fail(e.what());
  }
  return round;
}

void Pipeline::SnapSave(std::uint32_t positions) {
  if (failed_) throw std::runtime_error("the stages failed earlier: " + reason_);
  Impl& I = *impl_;
  try {
    if (I.r0 != nullptr) I.r0->Send(relay::Op::kReset, positions, 0, relay::kSnapSave, 0, {});
    I.stages[0]->SnapSave(positions);
    if (I.r0 != nullptr) {
      Pump();
      if (I.r0->Recv().op() != relay::Op::kReply) Fail("rank 0: SNAPSHOT not answered");
    }
  } catch (const std::exception& e) {
    Fail(e.what());
  }
}

void Pipeline::Reset(std::uint32_t drafts, std::uint32_t restore) {
  if (failed_) throw std::runtime_error("the stages failed earlier: " + reason_);
  if (drafts > opt_.drafts) throw std::invalid_argument("a session of " + std::to_string(drafts) + " drafts, the executor's most is " + std::to_string(opt_.drafts));
  Impl& I = *impl_;
  try {
    Drain();
    std::uint32_t flags = opt_.graph ? 0U : relay::kGraphOff;
    if (drafts > 0) flags |= relay::kSpecRound | relay::kSpecAllRows | (opt_.draft_value ? relay::kSpecValue : 0U);
    if (restore > 0) flags |= relay::kSnapRestore;
    if (I.r0 != nullptr) I.r0->Send(relay::Op::kReset, restore, 0, flags, drafts, {});
    flerp::qwen35moe::SetDecodeGraph(opt_.graph);
    session_drafts_ = drafts;
    held_.clear();
    if (restore > 0) I.stages[0]->SnapRestore(drafts, restore);
    else I.stages[0]->Reset(drafts);
    if (I.r0 != nullptr) {
      Pump();
      if (I.r0->Recv().op() != relay::Op::kReply) Fail("rank 0: RESET not answered");
    }
  } catch (const std::exception& e) {
    Fail(e.what());
  }
}

void Pipeline::Close() noexcept {
  if (closed_ || !impl_) return;
  closed_ = true;
  Impl& I = *impl_;
  if (I.r0 != nullptr && !failed_ && !I.r0->poisoned()) {
    try {
      Drain();
      I.r0->Send(relay::Op::kClose, 0, 0, 0, 0, {});
      Pump();
      (void)I.r0->Recv();
    } catch (const std::exception&) {
    }
  }
  try {
    Emit("{\"event\":\"closed\",\"rank\":0,\"failed\":" + std::string(failed_ ? "true" : "false") + ",\"t_unix\":" + UnixText() + "}");
  } catch (...) {
  }
}

// --- a stage process ----------------------------------------------------------------------------------------------
void InterruptStage() noexcept {
  relay::Relay* r = g_stage_relay.load();
  if (r != nullptr) r->Interrupt();
}

int StageMain(const Options& o, const std::atomic<bool>& stop) {
  ::signal(SIGPIPE, SIG_IGN);
  const auto event = [&](const char* kind, const std::string& extra) {
    Emit(std::string("{\"event\":\"") + kind + "\",\"rank\":" + std::to_string(o.rank) + extra + ",\"t_unix\":" + UnixText() + "}");
  };
  try {
    if (const std::string e = CheckOptions(o); !e.empty()) throw std::invalid_argument(e);
    if (o.rank == 0 || o.world < 2) throw std::invalid_argument("a stage process is rank 1..world-1 of a world of 2 or more");
    const Peers peers = ParsePeers(o.peers);
    const auto reader = OpenReader(o.weights);
    std::string error;
    const auto whole = flerp::gguf_placed::StagePlan(*reader, 1, std::nullopt, &error);
    if (!whole) throw std::runtime_error("placement: " + error);
    const auto cuts = ResolveCuts(*reader, o, whole->blocks);
    const auto model = Load(reader, o, std::make_pair(cuts[o.rank], cuts[o.rank + 1]), o.rank, "stage");
    if (stop.load()) {
      event("terminated", ",\"when\":\"after the load\"");
      return 0;
    }
    Stage stage(o.rank, o.world, model, o.context);
    g_hidden = model->GetConfig().hidden_size;
    std::unique_ptr<flerp::qwen35moe::MtpHead> head;
    std::unique_ptr<flerp::qwen35moe::DraftRound> draft;
    if (o.draft_head) {
      // the file's MTP block beside the last stage, its rows through a stage output buffer bound before the first forward
      const auto& w = model->GetWeights();
      HipOk(hipMalloc(reinterpret_cast<void**>(&g_draft_rows), stage.max_rows() * w.config.hidden_size * sizeof(float)), "draft rows");
      auto io = flerp::qwen35moe::GetStageIo(&w);
      io.d_output = g_draft_rows;
      flerp::qwen35moe::BindStageIo(&w, io);
      const std::string data = o.mtp_data_file.empty() ? o.data_file : o.mtp_data_file;
      std::string head_error;
      head = flerp::qwen35moe::MtpHead::Create(*reader, w, {.max_context = o.context, .data_path = data}, &head_error);
      if (!head) throw std::runtime_error(head_error);
      draft = std::make_unique<flerp::qwen35moe::DraftRound>(*head, w.config.hidden_size, o.context);
      g_draft = draft.get();
      if (!o.draft_costs.empty()) {
        float c[3];
        if (std::sscanf(o.draft_costs.c_str(), "%f,%f,%f", &c[0], &c[1], &c[2]) != 3 || c[0] <= 0 || c[1] <= 0 || c[2] < 0)
          throw std::invalid_argument("draft_costs: want base_ms,row_ms,mtp_ms");
        g_draft_costs = {c[0], c[1], c[2]};
      }
      event("draft_head", ",\"block\":" + std::to_string(head->block()) + ",\"weight_bytes\":" + std::to_string(head->weight_bytes()) +
                              ",\"load_s\":" + Num(head->load_s()) + ",\"data_file\":" + Str(data));
    }
    const auto& mc = model->GetConfig();
    const auto plan = RelayPlan(whole->Sha256(), cuts, mc.hidden_size, mc.vocab_size, stage.max_rows(), o.context);
    std::unique_ptr<relay::Relay> r;
    try {
      r = relay::ConnectRing(RelayConfig(o, o.rank, plan, mc, stage.max_rows()), peers.hosts, peers.port,
                             std::chrono::milliseconds(o.setup_deadline_ms));
    } catch (const relay::RelayError& e) {
      if (stop.load()) {
        event("terminated", ",\"when\":\"during the setup\"");
        return 0;
      }
      event("relay_failed", ",\"reason\":" + Str(e.what()));
      return 1;
    }
    g_stage_relay.store(r.get());
    if (stop.load()) r->Interrupt();   // a signal that came before the store
    event("connected", ",\"world\":" + std::to_string(o.world) + ",\"cuts\":" + Ids(cuts));
    int rc = 0;
    try {
      while (ServeOne(stage, *r, kIdle)) {
      }
      event("closed", "");
    } catch (const std::exception& e) {
      if (stop.load()) {
        event("terminated", ",\"when\":\"waiting for a call\"");
      } else {
        r->Abort(e.what());
        event("relay_failed", ",\"reason\":" + Str(e.what()));
        rc = 3;
      }
    }
    g_stage_relay.store(nullptr);
    r.reset();
    return rc;
  } catch (const std::exception& e) {
    if (stop.load()) {
      event("terminated", ",\"when\":\"during the load\"");
      return 0;
    }
    event("error", ",\"error\":" + Str(e.what()));
    return 1;
  }
}

}  // namespace flerp::qwen35moe_abi
