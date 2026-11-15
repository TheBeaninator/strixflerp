// SPDX-License-Identifier: MIT
// A greedy token generator over gufo's Qwen 3.5 executor with executor.flerp.qwen35moe laid over it, for recipe
// gates. flerp/gufo_qwen/generate.cpp (tool.flerp.gufo_qwen_generate) with a traced mode added. It prints token
// ids, not prose: one JSON object a line on stdout, everything else on stderr.
//
//   flerp-qwen35moe-generate --model FILE.gguf [--cpu] [--tokenize] [--context N] [--max-tokens N]
//                            [--trace] [--trace-top N] [--trace-layers 0,1,39] [--decode-prompt]
//                            (--prompt-file FILE | --ids-file FILE)...
//
// --prompt-file   a prompt as raw UTF-8 text: no chat template, encoded by the engine's tokenizer.
// --ids-file      a prompt as token ids separated by white space (the tokenizer is not asked).
// --cpu           the CPU reference (models::QwenGenerator) instead of the HIP executor.
// --tokenize      print each prompt's token ids and generate nothing (implies --cpu: no device is touched).
// --trace         HIP only. Instead of QwenGpuExecutor::Generate, the same calls it makes (ForwardPromptBatch, then
//                 ForwardToken a token; executor.cpp GenerateFromPrefix) after a Reset, and after each forward the
//                 logits row is copied to the host: the --trace-top largest logits and the row's log-sum-exp are
//                 printed for every generated position. For the prompt pass, of the blocks named by --trace-layers:
//                 the hidden state after the block (sums, norms, the corner values llama.cpp's eval callback
//                 prints) and what the router chose (ids in selection order, weights, the shared gate logit) for
//                 the first three and last three prompt tokens.
// --decode-prompt with --trace: the prompt goes through ForwardToken a token (the decode step's kernels: FP32
//                 activations everywhere) instead of ForwardPromptBatch (prompt chunks, whose attention and
//                 DeltaNet projections quantize activations). No hidden states or router choices are recorded.
// --force-file F  with --trace, one a prompt in order: token ids fed as the continuation in place of the tokens the
//                 engine draws (teacher forcing), so every engine is asked about the same context.
// --record-at L   with --force-file: the generated positions (0 = the first) whose logits are recorded; the run
//                 stops after the last. The CPU reference skips the LM head at the other positions.
// --arms A,B,D,.. HIP only (PREFILL-ORNITH): the whole prompt list once per arm in this one load: A = the per-token
//                 prompt experts, B = the grouped ones (flerp::qwen35moe::SetMoeGrouped), E = grouped exact (exact
//                 weight codes, hi/lo F16 activations), F = B + the dense projections on the grouped Linear (F16),
//                 X = E + the dense projections on the exact Linear, G = F and Y = X with the GQA WMMA prompt
//                 attention (kernel.flerp.attn_wmma_gqa), D = --decode-prompt with A, L = G with live-tile items
//                 (flerp::kgrouped::SetLiveTiles, PREFILL-KPROF), C = G with the 16-byte MoE combine
//                 and one F16 conversion per shared projection input (flerp::qwen35moe::SetMoeCombineFast, MOE-COMBINE),
//                 R = G with the tiled prompt router (flerp::moe::SetRouterTiled, ROUTER-GEMM).
//                 An `arm` event precedes each arm's events.
// --cpu --trace   the CPU reference, driven here a token at a time (models::ForwardModel, as QwenGenerator does),
//                 with the same record of every generated position.
//
// --placed        HIP only. Load through format.flerp.gguf_placed (flerp/qwen35moe/stage.hpp): the stage plan of
//                 (header, --world W, --rank R), or with --blocks B0:B1 (world 1) the blocks [B0, B1) only; one device
//                 allocation of the plan's bytes filled by byte-range reads. The `loaded` event then carries the
//                 loader's report (plan hash, bytes allocated, seconds, device memory before and after).
//                 --single-reader [--pause-ms N] [--chunk-pause-ms N]: one thread reading O_DIRECT, sleeping between
//                 tensors (and pieces), for a host whose disk and page cache others depend on.
// --plan-only     no device: print the plan of --world W (and --blocks) as one `plan` event and exit.
// --stage-bench   HIP only, with --placed: instead of generating, run the loaded block range as a stage - a prompt
//                 chunk of --chunk-rows rows and decode steps - first on the token ids of the first --ids-file (a
//                 range that starts at block 0), then on a fixed synthetic hidden-state input, and print one
//                 `stage_bench` event a phase: times, the sha256 of the stage's output rows, whether a repeat gave
//                 the same bytes. --dump FILE --dump-layers L,.. --dump-rows R,..: whole MoE blocks as the device
//                 computed them, in the record format of gate/moe_block_check.py (positions: a chunk row is its
//                 row; a decode step is chunk rows + its index; the synthetic phase adds 100000).
//
// --staged loopback|tcp   the model as N stages joined by runtime.flerp.stage_relay (stage_driver.cpp has the options).
//
// --mtp-probe D   HIP only, off by default (gate/ornith-o3-mtp.toml). The calls Generate makes (ProbeRun), and beside
//                 them the file's MTP head (flerp/qwen35moe/mtp_head.hpp) drafting D tokens at every frontier; one
//                 `mtp_probe` event a prompt after its `generated` event (mtp_probe.hpp). The ids are the run's own.
//                 --mtp-data-file F: the head's bytes from a same-size copy at the same offsets (default: the GGUF).
//                 --mtp-dump F [--mtp-dump-frontiers N]: every MTP step of the first prompt until N frontiers (8),
//                 inputs and outputs, for gate/ornith_o3_mtp.py ref.
//
// Without --trace the decode is the engine's own: QwenGpuExecutor::Generate at its default sampling (temperature
// 0, so the unmodified argmax) and its production execution policy. No draft backend is attached.
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>

#include "flerp/gguf_placed/placement.hpp"
#include "flerp/qwen35moe/stage.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/families/qwen/text/chat_template.hpp"
#include "src/families/qwen/text/tokenizer.hpp"
#include "src/models/qwen/forward.hpp"
#include "src/models/qwen/generator.hpp"
#include "src/models/qwen/ssm.hpp"
#include "src/models/qwen/state.hpp"
#include "flerp/qwen35moe/moe_ffn.hpp"
#include "flerp/moe_kgrouped/moe_kgrouped.hpp"
#include "flerp/moe_kquant/moe_kquant.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/modules/embed.hpp"
#include "src/models/qwen/modules/module_ctx.hpp"
#include "src/models/qwen/modules/unembed.hpp"
#include "flerp/qwen35moe/mtp_head.hpp"
#include "mtp_probe.hpp"

namespace {

using gufo::tokenization::TokenId;
using Clock = std::chrono::steady_clock;

double Since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string JsonString(std::string_view text) {
  std::string out = "\"";
  char buffer[8];
  for (const unsigned char c : text) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += static_cast<char>(c);
    } else if (c < 0x20) {
      std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
      out += buffer;
    } else {
      out += static_cast<char>(c);
    }
  }
  return out + "\"";
}

template <typename Range>
std::string JsonIds(const Range& ids) {
  std::string out = "[";
  bool first = true;
  for (const auto id : ids) {
    if (!first) out += ",";
    out += std::to_string(static_cast<std::int64_t>(id));
    first = false;
  }
  return out + "]";
}

struct Prompt {
  std::string source;
  bool is_ids{false};
};

struct Result {
  std::vector<TokenId> ids;
  std::string text;
  double first_token_s{0.0};
  double total_s{0.0};
};

template <typename Engine>
Result Run(Engine& engine, const std::vector<TokenId>& prompt, std::size_t max_tokens) {
  Result result;
  gufo::models::GenerationOptions options;
  options.max_new_tokens = max_tokens;
  const auto start = Clock::now();
  (void)engine.Generate(prompt, options, [&](TokenId token, std::string_view piece) {
    if (result.ids.empty()) result.first_token_s = Since(start);
    result.ids.push_back(token);
    result.text.append(piece);
    return true;
  });
  result.total_s = Since(start);
  return result;
}

// --mtp-probe: the calls Generate makes (QwenGpuExecutor::GenerateFromPrefix: the prompt in the arena's chunks, as
// ForwardPromptBatch cuts it, then ForwardToken a token, greedy), with the MTP probe beside them (mtp_probe.hpp). The
// target's rows come through the stage output buffer d_rows, set before the first forward.
Result ProbeRun(gufo::hip::QwenGpuExecutor& gpu, const std::vector<TokenId>& prompt, std::size_t max_tokens,
                flerp_tool::MtpProbe& probe, const float* d_rows) {
  Result result;
  const std::size_t hidden = gpu.GetConfig().hidden_size;
  const std::size_t batch = gpu.GetMaxPromptBatch();
  gpu.Reset();
  probe.Begin(std::vector<std::uint32_t>(prompt.begin(), prompt.end()));
  const auto start = Clock::now();
  TokenId next = 0;
  const float* last_row = d_rows;
  for (std::size_t off = 0; off < prompt.size(); off += batch) {
    const std::size_t n = std::min(batch, prompt.size() - off);
    const bool last = off + n == prompt.size();
    next = gpu.ForwardPromptBatch(std::span<const TokenId>(prompt.data() + off, n), static_cast<std::uint32_t>(off), last);
    probe.PromptRows(static_cast<std::uint32_t>(off), n, d_rows);
    last_row = d_rows + (n - 1) * hidden;
  }
  std::size_t position = prompt.size();
  while (result.ids.size() < max_tokens) {
    if (gpu.GetTokenizer().IsStopToken(next)) break;
    if (result.ids.empty()) result.first_token_s = Since(start);
    result.ids.push_back(next);
    result.text.append(gpu.GetTokenizer().DecodeToken(next));
    probe.Frontier(static_cast<std::uint32_t>(position), static_cast<std::uint32_t>(next), last_row);
    next = gpu.ForwardToken(next, static_cast<std::uint32_t>(position));
    last_row = d_rows;
    ++position;
  }
  result.total_s = Since(start);
  return result;
}

int Usage() {
  std::cerr << "usage: flerp-qwen35moe-generate --model FILE.gguf [--cpu] [--tokenize] [--context N] [--max-tokens N] "
               "[--trace] [--trace-top N] [--trace-layers L,L,...] (--prompt-file FILE | --ids-file FILE)...\n";
  return 2;
}

std::string JsonFloats(const float* values, std::size_t count) {
  std::string out = "[";
  char buffer[40];
  for (std::size_t i = 0; i < count; ++i) {
    std::snprintf(buffer, sizeof(buffer), "%s%.9g", i == 0 ? "" : ",", static_cast<double>(values[i]));
    out += buffer;
  }
  return out + "]";
}

std::string JsonDouble(double value) {
  char buffer[40];
  std::snprintf(buffer, sizeof(buffer), "%.12g", value);
  return buffer;
}

// The tokens llama.cpp's eval callback shows of a row of rows: the first three and the last three.
std::vector<std::size_t> CornerRows(std::size_t rows) {
  std::vector<std::size_t> out;
  for (std::size_t r = 0; r < rows; ++r) {
    if (r < 3 || r + 3 >= rows) out.push_back(r);
  }
  return out;
}

struct Step {
  std::size_t position{0};
  TokenId token{0};
  std::vector<std::pair<std::uint32_t, float>> top;
  double logsumexp{0.0};
  std::string sha256;   // of the whole copied logits row (HIP runs)
};

std::string Sha256Floats(std::span<const float> values) {
  return gufo::crypto::Sha256Hex(
      std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(values.data()), values.size_bytes()));
}

Step Record(std::span<const float> logits, std::size_t position, TokenId token, std::size_t top_n) {
  Step step;
  step.position = position;
  step.token = token;
  std::vector<std::uint32_t> order(logits.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<std::uint32_t>(i);
  top_n = std::min(top_n, order.size());
  std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(top_n), order.end(),
                    [&](std::uint32_t a, std::uint32_t b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); });
  const double max = top_n == 0 ? 0.0 : logits[order[0]];
  double sum = 0.0;
  for (const float v : logits) sum += std::exp(static_cast<double>(v) - max);
  step.logsumexp = max + std::log(sum);
  for (std::size_t i = 0; i < top_n; ++i) step.top.emplace_back(order[i], logits[order[i]]);
  return step;
}

// One prompt, traced. Returns false after printing an error.
std::string StepsJson(const std::vector<Step>& steps) {
  std::string out = "[";
  for (std::size_t i = 0; i < steps.size(); ++i) {
    out += std::string(i == 0 ? "" : ",") + "{\"position\":" + std::to_string(steps[i].position) + ",\"token\":" +
           std::to_string(static_cast<std::int64_t>(steps[i].token)) + ",\"logsumexp\":" + JsonDouble(steps[i].logsumexp) +
           (steps[i].sha256.empty() ? std::string() : ",\"sha256\":\"" + steps[i].sha256 + "\"") + ",\"top\":[";
    for (std::size_t j = 0; j < steps[i].top.size(); ++j) {
      out += std::string(j == 0 ? "" : ",") + "[" + std::to_string(steps[i].top[j].first) + "," +
             JsonDouble(steps[i].top[j].second) + "]";
    }
    out += "]}";
  }
  return out + "]";
}

// One prompt through the CPU reference, traced: QwenGenerator::Generate's forwards (generator.cpp), greedy.
bool TraceCpu(const gufo::core::GgufReader& reader, const gufo::tokenization::QwenTokenizer& tokenizer,
              const std::vector<TokenId>& prompt, std::size_t max_tokens, std::size_t top_n, std::uint32_t context,
              std::size_t index, const std::string& source, const std::vector<TokenId>* forced,
              const std::vector<std::size_t>& record_at) {
  std::string error;
  auto weights = gufo::models::QwenModelWeights::LoadFromGguf(reader, &error);
  if (!weights) {
    std::cerr << "error: " << error << "\n";
    return false;
  }
  const auto& config = weights->config;
  gufo::models::QwenKvCache kv(config.FullAttentionLayerCount(), config.num_key_value_heads, context, config.head_dim);
  gufo::models::QwenSsmCache ssm(config.num_layers, config.SsmQkvSize(), config.ssm_conv_kernel, config.ssm_time_step_rank,
                                 config.ssm_state_size, config.SsmValueSize());
  gufo::models::QwenScratchArena arena(config);
  kv.Reset();
  ssm.Reset();
  const auto start = Clock::now();
  if (forced != nullptr) {
    // ForwardModel's own steps (forward.cpp), with the LM head only where its logits are recorded.
    std::vector<TokenId> all(prompt);
    all.insert(all.end(), forced->begin(), forced->begin() + static_cast<std::ptrdiff_t>(max_tokens - 1));
    std::vector<Step> steps;
    double first_token_s = 0.0;
    for (std::size_t i = 0; i < all.size(); ++i) {
      const gufo::models::qwen::CpuModuleContext module_ctx;
      gufo::models::qwen::EmbedForward(module_ctx, all[i], weights->token_embd, config.hidden_size, arena.hidden);
      for (std::uint32_t l = 0; l < config.num_layers; ++l) {
        gufo::models::ForwardLayer(arena.hidden, weights->layers[l], config, kv, ssm, l, static_cast<std::uint32_t>(i), arena);
      }
      if (i + 1 < prompt.size()) continue;
      const std::size_t k = i + 1 - prompt.size();
      if (k == 0) first_token_s = Since(start);
      if (std::find(record_at.begin(), record_at.end(), k) == record_at.end()) continue;
      const gufo::models::qwen::CpuLayerContext layer_ctx(config, arena);
      gufo::models::qwen::UnembedForward(layer_ctx, weights->output_norm, weights->output, arena.hidden, arena.logits);
      steps.push_back(Record(arena.logits, k, static_cast<TokenId>(gufo::models::GreedyArgmax(arena.logits)), top_n));
    }
    std::cout << "{\"event\":\"traced\",\"mode\":\"cpu\",\"forced\":true,\"prompt\":" << index << ",\"source\":" << JsonString(source)
              << ",\"prompt_tokens\":" << prompt.size() << ",\"ids\":[],\"argmax_is_token\":true,\"first_token_s\":" << first_token_s
              << ",\"total_s\":" << Since(start) << ",\"hidden\":[],\"router\":[],\"steps\":" << StepsJson(steps) << "}" << std::endl;
    return true;
  }
  for (std::size_t p = 0; p < prompt.size(); ++p) {
    gufo::models::ForwardModel(prompt[p], static_cast<std::uint32_t>(p), *weights, kv, ssm, arena, arena.logits);
  }
  const double first_token_s = Since(start);
  std::vector<Step> steps;
  std::vector<TokenId> ids;
  std::size_t position = prompt.size();
  while (ids.size() < max_tokens) {
    const auto next = static_cast<TokenId>(gufo::models::GreedyArgmax(arena.logits));
    steps.push_back(Record(arena.logits, position, next, top_n));
    if (tokenizer.IsStopToken(next)) break;
    ids.push_back(next);
    if (ids.size() == max_tokens) break;
    gufo::models::ForwardModel(next, static_cast<std::uint32_t>(position), *weights, kv, ssm, arena, arena.logits);
    ++position;
  }
  std::cout << "{\"event\":\"traced\",\"mode\":\"cpu\",\"prompt\":" << index << ",\"source\":" << JsonString(source)
            << ",\"prompt_tokens\":" << prompt.size() << ",\"ids\":" << JsonIds(ids) << ",\"argmax_is_token\":true"
            << ",\"first_token_s\":" << first_token_s << ",\"total_s\":" << Since(start) << ",\"hidden\":[],\"router\":[]"
            << ",\"steps\":" << StepsJson(steps) << "}" << std::endl;
  return true;
}

bool Trace(gufo::hip::QwenGpuExecutor& gpu, const std::vector<TokenId>& prompt, std::size_t max_tokens, std::size_t top_n,
           const std::vector<std::uint32_t>& traced_layers, bool decode_prompt, std::size_t index, const std::string& source,
           const std::vector<TokenId>* forced, const std::vector<std::size_t>& record_at) {
  const std::vector<std::uint32_t> layers = decode_prompt ? std::vector<std::uint32_t>{} : traced_layers;
  const auto& config = gpu.GetConfig();
  const std::size_t hidden = config.hidden_size;
  gpu.Reset();
  flerp::qwen35moe::MoeTrace router;
  flerp::qwen35moe::SetMoeTrace(&router);
  if (!decode_prompt) gpu.SetPromptHiddenCapture(true, layers);
  const auto start = Clock::now();
  TokenId next = 0;
  if (decode_prompt) {
    for (std::size_t p = 0; p < prompt.size(); ++p) {
      next = gpu.ForwardToken(prompt[p], static_cast<std::uint32_t>(p), p + 1 == prompt.size());
    }
  } else {
    next = gpu.ForwardPromptBatch(prompt, 0, true);
  }
  const double first_token_s = Since(start);
  flerp::qwen35moe::SetMoeTrace(nullptr);
  const auto captured = gpu.GetPromptHiddenStates();
  const std::size_t tokens = prompt.size();
  std::string hidden_json = "[";
  if (!layers.empty() && captured.size() == tokens * layers.size() * hidden) {
    for (std::size_t li = 0; li < layers.size(); ++li) {
      const auto row = [&](std::size_t t) { return captured.data() + ((t * layers.size()) + li) * hidden; };
      double sum_all = 0.0;
      float sum_all_f32 = 0.0F;          // llama.cpp's eval callback sums in float, in this order
      for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t i = 0; i < hidden; ++i) {
          sum_all += row(t)[i];
          sum_all_f32 += row(t)[i];
        }
      }
      const float* last = row(tokens - 1);
      double l2 = 0.0, sum_last = 0.0;
      float sum_last_f32 = 0.0F;
      for (std::size_t i = 0; i < hidden; ++i) {
        l2 += static_cast<double>(last[i]) * last[i];
        sum_last += last[i];
        sum_last_f32 += last[i];
      }
      hidden_json += std::string(li == 0 ? "" : ",") + "{\"layer\":" + std::to_string(layers[li]) + ",\"sum_all_tokens\":" +
                     JsonDouble(sum_all) + ",\"sum_all_tokens_f32\":" + JsonDouble(sum_all_f32) + ",\"last_token_l2\":" +
                     JsonDouble(std::sqrt(l2)) + ",\"last_token_sum\":" + JsonDouble(sum_last) + ",\"last_token_sum_f32\":" +
                     JsonDouble(sum_last_f32) + ",\"corners\":[";
      bool first = true;
      for (const std::size_t t : CornerRows(tokens)) {
        hidden_json += std::string(first ? "" : ",") + "{\"token\":" + std::to_string(t) + ",\"first3\":" +
                       JsonFloats(row(t), 3) + ",\"last3\":" + JsonFloats(row(t) + hidden - 3, 3) + "}";
        first = false;
      }
      hidden_json += "]}";
    }
  }
  hidden_json += "]";
  if (!decode_prompt) gpu.SetPromptHiddenCapture(false);

  std::string router_json = "[";
  bool first_layer = true;
  // A prompt longer than the arena's batch goes in chunks: the trace has one record a block a chunk. The first
  // chunk holds the first tokens, the last chunk the last ones.
  for (const std::uint32_t layer : layers) {
    const flerp::qwen35moe::MoeTraceLayer* head = nullptr;
    const flerp::qwen35moe::MoeTraceLayer* tail = nullptr;
    for (const auto& rec : router.layers) {
      if (rec.layer != layer) continue;
      if (head == nullptr) head = &rec;
      tail = &rec;
    }
    if (head == nullptr) continue;
    router_json += std::string(first_layer ? "" : ",") + "{\"layer\":" + std::to_string(layer) + ",\"used\":" +
                   std::to_string(head->used) + ",\"tokens\":[";
    first_layer = false;
    bool first = true;
    const auto emit = [&](const flerp::qwen35moe::MoeTraceLayer& rec, std::size_t local, std::size_t global) {
      router_json += std::string(first ? "" : ",") + "{\"token\":" + std::to_string(global) + ",\"ids\":" +
                     JsonIds(std::span<const std::int32_t>(rec.ids.data() + local * rec.used, rec.used)) +
                     ",\"weights\":" + JsonFloats(rec.weights.data() + local * rec.used, rec.used);
      if (!rec.shared_gate_logit.empty())
        router_json += ",\"shared_gate_logit\":" + JsonFloats(rec.shared_gate_logit.data() + local, 1);
      router_json += "}";
      first = false;
    };
    for (std::size_t t = 0; t < std::min<std::size_t>(3, head->tokens); ++t) emit(*head, t, t);
    for (std::size_t t = tail->tokens >= 3 ? tail->tokens - 3 : 0; t < tail->tokens; ++t) {
      const std::size_t global = tokens - tail->tokens + t;
      if (global >= 3) emit(*tail, t, global);
    }
    router_json += "]}";
  }
  router_json += "]";

  std::vector<Step> steps;
  std::vector<TokenId> ids;
  std::size_t position = tokens;
  bool argmax_agrees = true;
  while (ids.size() < max_tokens) {
    const std::size_t k = ids.size();
    if (forced == nullptr || std::find(record_at.begin(), record_at.end(), k) != record_at.end()) {
      const auto logits = gpu.CopyLastLogits();
      steps.push_back(Record(logits, forced == nullptr ? position : k, next, top_n));
      steps.back().sha256 = Sha256Floats(logits);
      argmax_agrees = argmax_agrees && !steps.back().top.empty() && steps.back().top[0].first == static_cast<std::uint32_t>(next);
    }
    if (forced == nullptr && gpu.GetTokenizer().IsStopToken(next)) break;
    const TokenId feed = forced == nullptr ? next : (*forced)[k];
    ids.push_back(feed);
    if (forced != nullptr && ids.size() == max_tokens) break;
    next = gpu.ForwardToken(feed, static_cast<std::uint32_t>(position));
    ++position;
  }
  const double total_s = Since(start);
  const std::string steps_json = StepsJson(steps);
  std::cout << "{\"event\":\"traced\",\"mode\":" << JsonString(decode_prompt ? "decode-prompt" : "prefill") << ",\"prompt\":"
            << index << ",\"source\":" << JsonString(source) << ",\"prompt_tokens\":" << tokens << ",\"ids\":" << JsonIds(ids) << ",\"argmax_is_token\":"
            << (argmax_agrees ? "true" : "false") << ",\"first_token_s\":" << first_token_s << ",\"total_s\":" << total_s
            << ",\"hidden\":" << hidden_json << ",\"router\":" << router_json << ",\"steps\":" << steps_json << "}"
            << std::endl;
  return true;
}


// --- the loaded block range as a stage (--stage-bench) ----------------------------------------------------------
struct BenchOptions {
  std::size_t chunk_rows{1024};
  std::size_t decode_steps{48};
  std::string dump_path;
  std::vector<std::uint32_t> dump_layers;
  std::vector<std::uint32_t> dump_rows;
};

void HipOk(hipError_t status, const char* what) {
  if (status != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
}

struct RowStats {
  std::string sha256;
  std::size_t non_finite{0};
  double rms{0.0};
};

RowStats ReadRows(const float* device, std::size_t count, std::vector<float>& host) {
  host.resize(count);
  HipOk(hipDeviceSynchronize(), "synchronize");
  HipOk(hipMemcpy(host.data(), device, count * sizeof(float), hipMemcpyDeviceToHost), "copy of the stage output");
  RowStats out;
  out.sha256 = Sha256Floats(host);
  double sum = 0.0;
  for (const float v : host) {
    if (!std::isfinite(v)) ++out.non_finite;
    else sum += static_cast<double>(v) * v;
  }
  out.rms = std::sqrt(sum / static_cast<double>(std::max<std::size_t>(count, 1)));
  return out;
}

std::string StatsJson(const RowStats& s) {
  return "{\"sha256\":\"" + s.sha256 + "\",\"non_finite\":" + std::to_string(s.non_finite) + ",\"rms\":" + JsonDouble(s.rms) + "}";
}

// Mean milliseconds of `steps` decode steps from `first_pos`; the first one apart (it may capture the graph).
struct DecodeTiming {
  double first_ms{0.0};
  double mean_ms{0.0};
  std::size_t steps{0};
};

DecodeTiming TimeDecode(gufo::hip::QwenGpuExecutor& gpu, const std::vector<TokenId>& tokens, std::size_t first_index,
                        std::uint32_t first_pos, std::size_t steps, bool logits) {
  DecodeTiming t;
  double rest = 0.0;
  for (std::size_t i = 0; i < steps; ++i) {
    const TokenId token = tokens.empty() ? 0 : tokens[(first_index + i) % tokens.size()];
    const auto start = Clock::now();
    (void)gpu.ForwardToken(token, first_pos + static_cast<std::uint32_t>(i), logits);
    HipOk(hipDeviceSynchronize(), "synchronize");
    const double ms = Since(start) * 1e3;
    if (i == 0) t.first_ms = ms;
    else rest += ms;
  }
  t.steps = steps;
  t.mean_ms = steps > 1 ? rest / static_cast<double>(steps - 1) : t.first_ms;
  return t;
}

std::string TimingJson(const DecodeTiming& t) {
  return "{\"steps\":" + std::to_string(t.steps) + ",\"first_ms\":" + JsonDouble(t.first_ms) + ",\"mean_ms_after_first\":" +
         JsonDouble(t.mean_ms) + "}";
}

bool StageBench(gufo::hip::QwenGpuExecutor& gpu, const std::vector<TokenId>& ids, const BenchOptions& opt, std::uint32_t context) {
  const auto& report = flerp::qwen35moe::LastLoadReport();
  const auto& config = gpu.GetConfig();
  const std::size_t hidden = config.hidden_size;
  const std::size_t rows = opt.chunk_rows;
  const std::uint32_t begin = report.placed ? report.block_begin : 0;
  const std::uint32_t end = report.placed ? report.block_end : config.num_layers;
  const std::size_t blocks = end - begin;
  if (rows == 0 || rows > gpu.GetMaxPromptBatch() || rows + opt.decode_steps + 8 >= context) {
    std::cerr << "error: --chunk-rows " << rows << " with " << opt.decode_steps << " decode steps does not fit a batch of "
              << gpu.GetMaxPromptBatch() << " and a context of " << context << "\n";
    return false;
  }
  float* d_in = nullptr;
  float* d_out = nullptr;
  HipOk(hipMalloc(&d_in, rows * hidden * sizeof(float)), "allocation of the stage input");
  HipOk(hipMalloc(&d_out, rows * hidden * sizeof(float)), "allocation of the stage output");
  flerp::qwen35moe::MoeDump dump;
  dump.layers = opt.dump_layers;
  dump.rows = opt.dump_rows;
  if (std::find(dump.rows.begin(), dump.rows.end(), 0U) == dump.rows.end()) dump.rows.push_back(0);   // a decode step is row 0
  flerp::qwen35moe::SetMoeDump(&dump);
  std::vector<float> host;
  constexpr std::size_t kDumpedDecodeSteps = 3;

  const auto phase = [&](const char* name, bool synthetic, std::uint32_t dump_base) {
    const std::vector<TokenId> none;
    const std::vector<TokenId>& toks = synthetic ? none : ids;
    std::vector<TokenId> chunk(rows, 0);
    if (!synthetic) std::copy(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(rows), chunk.begin());
    const auto run_chunk = [&](bool armed, double& seconds) {
      gpu.Reset();
      dump.armed = armed && !dump.layers.empty();
      dump.position_base = dump_base;
      const auto start = Clock::now();
      (void)gpu.ForwardPromptBatch(chunk, 0, false);
      HipOk(hipDeviceSynchronize(), "synchronize");
      seconds = Since(start);
      dump.armed = false;
      return ReadRows(d_out, rows * hidden, host);
    };
    double chunk_first_s = 0.0, chunk_again_s = 0.0, chunk_third_s = 0.0;
    const RowStats first = run_chunk(true, chunk_first_s);
    // decode steps after the chunk, eager and dumped, then timed at this context
    const std::size_t dumped_before = dump.records;
    for (std::size_t i = 0; i < kDumpedDecodeSteps; ++i) {
      dump.armed = !dump.layers.empty();
      dump.position_base = dump_base + static_cast<std::uint32_t>(rows + i);
      (void)gpu.ForwardToken(toks.empty() ? 0 : toks[(rows + i) % toks.size()], static_cast<std::uint32_t>(rows + i), false);
      HipOk(hipDeviceSynchronize(), "synchronize");
      dump.armed = false;
    }
    const RowStats after_decode = ReadRows(d_out, hidden, host);
    const std::size_t at = rows + kDumpedDecodeSteps;
    // A graph captured without the stage input would replay without it: with a synthetic input only eager steps.
    DecodeTiming at_context_graph{};
    if (!synthetic) at_context_graph = TimeDecode(gpu, toks, at, static_cast<std::uint32_t>(at), 8, true);
    const DecodeTiming at_context_eager =
        TimeDecode(gpu, toks, at + 8, static_cast<std::uint32_t>(at + 8), 8, false);
    const RowStats again = run_chunk(false, chunk_again_s);
    const RowStats third = run_chunk(false, chunk_third_s);
    // decode from an empty context
    gpu.Reset();
    DecodeTiming fresh_graph{};
    std::size_t pos = 0;
    if (!synthetic) {
      fresh_graph = TimeDecode(gpu, toks, 0, 0, opt.decode_steps, true);
      pos = opt.decode_steps;
    }
    const RowStats fresh_out = ReadRows(d_out, hidden, host);
    const DecodeTiming fresh_eager = TimeDecode(gpu, toks, pos, static_cast<std::uint32_t>(pos), opt.decode_steps, false);
    const auto per_block = [&](double ms) { return JsonDouble(ms / static_cast<double>(blocks)); };
    std::cout << "{\"event\":\"stage_bench\",\"phase\":" << JsonString(name) << ",\"blocks\":[" << begin << "," << end << "],\"chunk_rows\":" << rows
              << ",\"hidden\":" << hidden << ",\"chunk_s\":[" << JsonDouble(chunk_first_s) << "," << JsonDouble(chunk_again_s) << "," << JsonDouble(chunk_third_s)
              << "],\"chunk_first_run_dumped\":" << (dump.layers.empty() ? "false" : "true")
              << ",\"chunk_ms_per_block\":" << per_block(std::min(chunk_again_s, chunk_third_s) * 1e3)
              << ",\"chunk_rows_per_s\":" << JsonDouble(static_cast<double>(rows) / std::min(chunk_again_s, chunk_third_s))
              << ",\"chunk_out\":" << StatsJson(first) << ",\"chunk_out_again\":" << StatsJson(again) << ",\"chunk_out_third\":" << StatsJson(third)
              << ",\"chunk_repeats_equal\":" << (first.sha256 == again.sha256 && again.sha256 == third.sha256 ? "true" : "false")
              << ",\"decode_out_after_chunk\":" << StatsJson(after_decode) << ",\"decode_dumped_records\":" << (dump.records - dumped_before)
              << ",\"decode_at_context\":{\"context\":" << at << ",\"with_logits_request\":" << TimingJson(at_context_graph)
              << ",\"eager\":" << TimingJson(at_context_eager) << "}"
              << ",\"decode_fresh\":{\"with_logits_request\":" << TimingJson(fresh_graph) << ",\"eager\":" << TimingJson(fresh_eager)
              << ",\"out_after_with_logits\":" << StatsJson(fresh_out) << "}"
              << ",\"decode_ms_per_block\":{\"with_logits_request\":" << (synthetic ? "null" : per_block(fresh_graph.mean_ms))
              << ",\"eager\":" << per_block(fresh_eager.mean_ms) << "}}" << std::endl;
  };

  bool ok = true;
  try {
    if (begin == 0 && ids.size() >= rows + 32) {
      flerp::qwen35moe::SetStageIo({.d_input = nullptr, .d_output = d_out});
      phase("tokens", false, 0);
    } else {
      std::cout << "{\"event\":\"stage_bench_skipped\",\"phase\":\"tokens\",\"why\":"
                << JsonString(begin != 0 ? "the range does not start at block 0" : "the ids file is shorter than the chunk plus 32") << "}" << std::endl;
    }
    // a fixed synthetic hidden-state input: xorshift32 from one seed, values in (-0.1, 0.1)
    std::vector<float> synth(rows * hidden);
    std::uint32_t x = 0x9E3779B9U;
    for (auto& v : synth) {
      x ^= x << 13U;
      x ^= x >> 17U;
      x ^= x << 5U;
      v = static_cast<float>(static_cast<std::int32_t>(x)) * (0.1F / 2147483648.0F);
    }
    HipOk(hipMemcpy(d_in, synth.data(), synth.size() * sizeof(float), hipMemcpyHostToDevice), "copy of the synthetic input");
    std::cout << "{\"event\":\"synthetic_input\",\"rows\":" << rows << ",\"hidden\":" << hidden << ",\"sha256\":\"" << Sha256Floats(synth) << "\"}" << std::endl;
    flerp::qwen35moe::SetStageIo({.d_input = d_in, .d_output = d_out});
    phase("synthetic", true, 100000);
  } catch (const std::exception& e) {
    std::cerr << "error: stage bench: " << e.what() << "\n";
    ok = false;
  }
  flerp::qwen35moe::SetMoeDump(nullptr);
  flerp::qwen35moe::SetStageIo({});
  if (!opt.dump_path.empty()) {
    std::ofstream out(opt.dump_path, std::ios::binary | std::ios::trunc);
    out.write(dump.bytes.data(), static_cast<std::streamsize>(dump.bytes.size()));
    std::cout << "{\"event\":\"moe_dump\",\"path\":" << JsonString(opt.dump_path) << ",\"records\":" << dump.records << ",\"bytes\":" << dump.bytes.size()
              << ",\"layers\":" << JsonIds(dump.layers) << ",\"rows\":" << JsonIds(dump.rows) << "}" << std::endl;
  }
  (void)hipFree(d_in);
  (void)hipFree(d_out);
  return ok;
}

std::string PlanJson(const flerp::gguf_placed::Plan& plan) {
  std::string out = "{\"event\":\"plan\",\"sha256\":\"" + plan.Sha256() + "\",\"world\":" + std::to_string(plan.world) + ",\"blocks\":" +
                    std::to_string(plan.blocks) + ",\"cuts\":" + JsonIds(plan.cuts) + ",\"file_bytes\":" + std::to_string(plan.file_bytes) +
                    ",\"unloaded\":" + std::to_string(plan.unloaded_tensors) + ",\"unloaded_bytes\":" + std::to_string(plan.unloaded_bytes) + ",\"ranks\":[";
  for (std::size_t r = 0; r < plan.ranks.size(); ++r) {
    const auto& rk = plan.ranks[r];
    out += std::string(r == 0 ? "" : ",") + "{\"rank\":" + std::to_string(rk.rank) + ",\"blocks\":[" + std::to_string(rk.block_begin) + "," +
           std::to_string(rk.block_end) + "],\"tensors\":" + std::to_string(rk.entries.size()) + ",\"tensor_bytes\":" +
           std::to_string(rk.tensor_bytes) + ",\"device_bytes\":" + std::to_string(rk.device_bytes) + "}";
  }
  return out + "]}";
}

}  // namespace

namespace flerp_tool {
int StagedMain(int argc, char** argv);   // stage_driver.cpp
}

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::string_view(argv[i]) == "--staged") return flerp_tool::StagedMain(argc, argv);
  }
  std::string model_path;
  std::vector<Prompt> prompts;
  std::size_t max_tokens = 64;
  std::uint32_t context = 4096;
  bool cpu = false;
  bool tokenize_only = false;
  bool trace = false;
  bool decode_prompt = false;
  std::string arms;
  std::vector<std::string> force_files;
  std::vector<std::size_t> record_at;
  std::size_t trace_top = 20;
  std::vector<std::uint32_t> trace_layers;
  flerp::qwen35moe::LoadOptions load_options;
  bool plan_only = false;
  bool stage_bench = false;
  BenchOptions bench;
  flerp_tool::MtpProbeOptions mtp;
  const auto list = [](const char* v) {
    std::vector<std::uint32_t> out;
    std::istringstream in(v);
    for (std::string item; std::getline(in, item, ',');) {
      if (!item.empty()) out.push_back(static_cast<std::uint32_t>(std::strtoul(item.c_str(), nullptr, 10)));
    }
    return out;
  };
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    const auto value = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
    if (arg == "--cpu") {
      cpu = true;
    } else if (arg == "--tokenize") {
      cpu = tokenize_only = true;
    } else if (arg == "--trace") {
      trace = true;
    } else if (arg == "--arms") {
      const char* v = value();
      if (v == nullptr) return Usage();
      arms = v;
    } else if (arg == "--decode-prompt") {
      decode_prompt = true;
    } else if (arg == "--placed") {
      load_options.placed = true;
    } else if (arg == "--single-reader") {
      load_options.fill.single_reader = true;
    } else if (arg == "--pause-ms" || arg == "--chunk-pause-ms") {
      const char* v = value();
      if (v == nullptr) return Usage();
      (arg == "--pause-ms" ? load_options.fill.pause_ms : load_options.fill.chunk_pause_ms) =
          static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10));
    } else if (arg == "--data-file" || arg == "--swarm-helper" || arg == "--swarm-manifest" || arg == "--swarm-from" || arg == "--swarm-conns") {
      // where the placed loader's bytes come from (stage_driver.cpp has the words)
      const char* v = value();
      if (v == nullptr) return Usage();
      auto& f = load_options.fill;
      if (arg == "--data-file") f.data_path = v, f.single_reader = true;
      else if (arg == "--swarm-helper") f.swarm.helper = v;
      else if (arg == "--swarm-manifest") f.swarm.manifest = v;
      else if (arg == "--swarm-conns") f.swarm.conns = static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10));
      else {
        f.swarm.sources.clear();
        std::string t(v);
        for (std::size_t a = 0, b; a <= t.size(); a = b + 1) {
          b = t.find(',', a);
          if (b == std::string::npos) b = t.size();
          if (b > a) f.swarm.sources.push_back(t.substr(a, b - a));
        }
      }
    } else if (arg == "--punch-behind") {
      load_options.fill.punch_behind = true;
    } else if (arg == "--plan-only") {
      plan_only = cpu = true;
    } else if (arg == "--stage-bench") {
      stage_bench = true;
    } else if (arg == "--world" || arg == "--rank" || arg == "--blocks" || arg == "--chunk-rows" || arg == "--decode-steps" ||
               arg == "--dump" || arg == "--dump-layers" || arg == "--dump-rows") {
      const char* v = value();
      if (v == nullptr) return Usage();
      if (arg == "--world") load_options.world = static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10));
      else if (arg == "--rank") load_options.rank = static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10));
      else if (arg == "--chunk-rows") bench.chunk_rows = std::strtoull(v, nullptr, 10);
      else if (arg == "--decode-steps") bench.decode_steps = std::strtoull(v, nullptr, 10);
      else if (arg == "--dump") bench.dump_path = v;
      else if (arg == "--dump-layers") bench.dump_layers = list(v);
      else if (arg == "--dump-rows") bench.dump_rows = list(v);
      else {
        const std::string text(v);
        const auto colon = text.find(':');
        if (colon == std::string::npos) return Usage();
        load_options.has_range = true;
        load_options.block_begin = static_cast<std::uint32_t>(std::strtoul(text.substr(0, colon).c_str(), nullptr, 10));
        load_options.block_end = static_cast<std::uint32_t>(std::strtoul(text.substr(colon + 1).c_str(), nullptr, 10));
      }
    } else if (arg == "--mtp-probe" || arg == "--mtp-data-file" || arg == "--mtp-dump" || arg == "--mtp-dump-frontiers") {
      const char* v = value();
      if (v == nullptr) return Usage();
      if (arg == "--mtp-probe") mtp.depth = static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10));
      else if (arg == "--mtp-data-file") mtp.data_file = v;
      else if (arg == "--mtp-dump") mtp.dump_path = v;
      else mtp.dump_frontiers = static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10));
    } else if (arg == "--trace-top" || arg == "--trace-layers" || arg == "--force-file" || arg == "--record-at") {
      const char* v = value();
      if (v == nullptr) return Usage();
      if (arg == "--trace-top") {
        trace_top = std::strtoull(v, nullptr, 10);
      } else if (arg == "--force-file") {
        force_files.emplace_back(v);
      } else if (arg == "--record-at") {
        std::istringstream in(v);
        for (std::string item; std::getline(in, item, ',');) {
          if (!item.empty()) record_at.push_back(std::strtoull(item.c_str(), nullptr, 10));
        }
      } else {
        std::istringstream in(v);
        for (std::string item; std::getline(in, item, ',');) {
          if (!item.empty()) trace_layers.push_back(static_cast<std::uint32_t>(std::strtoul(item.c_str(), nullptr, 10)));
        }
      }
    } else if (arg == "--model" || arg == "--prompt-file" || arg == "--ids-file" || arg == "--max-tokens" ||
               arg == "--context") {
      const char* v = value();
      if (v == nullptr) return Usage();
      if (arg == "--model") model_path = v;
      else if (arg == "--prompt-file") prompts.push_back({v, false});
      else if (arg == "--ids-file") prompts.push_back({v, true});
      else if (arg == "--max-tokens") max_tokens = std::strtoull(v, nullptr, 10);
      else context = static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10));
    } else {
      return Usage();
    }
  }
  if (model_path.empty() || (prompts.empty() && !plan_only) || max_tokens == 0 || (decode_prompt && (cpu || !trace))) return Usage();
  if (stage_bench && (cpu || trace || !load_options.placed)) return Usage();
  if (load_options.has_range && !(load_options.placed || plan_only)) return Usage();
  if (!force_files.empty() && (!trace || force_files.size() != prompts.size() || record_at.empty())) return Usage();
  if (mtp.depth > 0 && (cpu || trace || stage_bench || load_options.has_range)) return Usage();
  if (mtp.depth == 0 && (!mtp.data_file.empty() || !mtp.dump_path.empty())) return Usage();

  const auto load_start = Clock::now();
  std::string error;
  auto owner = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (!owner) {
    std::cerr << "error: GGUF " << model_path << ": " << error << "\n";
    return 1;
  }
  const std::shared_ptr<const gufo::core::GgufReader> reader(std::move(owner));
  const std::string architecture(reader->GetMetadataString("general.architecture").value_or(""));
  if (plan_only) {
    const auto plan = flerp::gguf_placed::StagePlan(
        *reader, load_options.world,
        load_options.has_range ? std::optional<flerp::gguf_placed::BlockRange>(
                                     flerp::gguf_placed::BlockRange{load_options.block_begin, load_options.block_end})
                               : std::nullopt,
        &error);
    if (!plan) {
      std::cerr << "error: placement: " << error << "\n";
      return 1;
    }
    std::cout << PlanJson(*plan) << std::endl;
    return 0;
  }
  flerp::qwen35moe::SetLoadOptions(load_options);

  std::unique_ptr<gufo::hip::QwenGpuExecutor> gpu;
  std::unique_ptr<gufo::models::QwenGenerator> host;
  if (cpu) {
    host = gufo::models::QwenGenerator::CreateFromGguf(*reader, &error);
  } else {
    gpu = gufo::hip::QwenGpuExecutor::CreateFromGguf(reader, &error, context);
  }
  if (!gpu && !host) {
    std::cerr << "error: " << (cpu ? "Qwen CPU generator" : "Qwen GPU executor") << " (general.architecture="
              << architecture << "): " << error << "\n";
    return 1;
  }
  const double load_s = Since(load_start);
  const auto& config = cpu ? host->GetConfig() : gpu->GetConfig();
  const auto& tokenizer = cpu ? host->GetTokenizer() : gpu->GetTokenizer();
  std::cout << "{\"event\":\"loaded\",\"engine\":" << JsonString(cpu ? "flerp-qwen35moe-cpu" : "flerp-qwen35moe-hip")
            << ",\"architecture\":" << JsonString(architecture) << ",\"layers\":" << config.num_layers
            << ",\"hidden\":" << config.hidden_size << ",\"intermediate\":" << config.intermediate_size
            << ",\"heads\":" << config.num_attention_heads << ",\"kv_heads\":" << config.num_key_value_heads
            << ",\"vocab\":" << config.vocab_size << ",\"mtp_layers\":" << config.mtp_num_layers
            << ",\"context\":" << context << ",\"load_s\":" << load_s;
  // The MoE sizes as the header gives them (the executor's own are per block: QwenLayerWeights::moe).
  const std::string arch_prefix = architecture + ".";
  const auto key = [&](const char* name) { return reader->GetMetadataUint32(arch_prefix + name).value_or(0); };
  std::cout << ",\"expert_count\":" << key("expert_count") << ",\"expert_used_count\":" << key("expert_used_count")
            << ",\"expert_feed_forward_length\":" << key("expert_feed_forward_length")
            << ",\"expert_shared_feed_forward_length\":" << key("expert_shared_feed_forward_length");
  if (gpu) {
    const auto usage = gpu->GetMemoryUsage();
    std::cout << ",\"device_bytes\":" << usage.TotalBytes()
              << ",\"weight_resident_bytes\":" << gpu->GetModel().GetResidentBytes()
              << ",\"weight_regions\":" << gpu->GetModel().GetWeightRegionCount()
              << ",\"max_prompt_batch\":" << gpu->GetMaxPromptBatch()
              << ",\"arena_estimate_bytes\":{\"context\":" << gufo::hip::QwenGpuExecutor::EstimateMemoryUsage(config, context).TotalBytes()
              << ",\"32768\":" << gufo::hip::QwenGpuExecutor::EstimateMemoryUsage(config, 32768).TotalBytes() << "}";
    const auto& rep = flerp::qwen35moe::LastLoadReport();
    std::cout << ",\"loader\":" << JsonString(rep.placed ? "placed" : "origin");
    if (rep.placed) {
      std::cout << ",\"placed\":{\"plan_sha256\":\"" << rep.plan_sha256 << "\",\"world\":" << rep.world << ",\"rank\":" << rep.rank
                << ",\"blocks\":[" << rep.block_begin << "," << rep.block_end << "],\"model_blocks\":" << rep.blocks
                << ",\"plan_device_bytes\":" << rep.plan_device_bytes << ",\"allocated_bytes\":" << rep.allocated_bytes
                << ",\"tensor_bytes\":" << rep.tensor_bytes << ",\"tensors\":" << rep.tensors
                << ",\"unloaded_tensors\":" << rep.unloaded_tensors << ",\"unloaded_bytes\":" << rep.unloaded_bytes
                << ",\"plan_s\":" << JsonDouble(rep.plan_s) << ",\"alloc_s\":" << JsonDouble(rep.alloc_s) << ",\"fill_s\":" << JsonDouble(rep.fill_s)
                << ",\"read_gb_per_s\":" << JsonDouble(rep.fill_s > 0 ? static_cast<double>(rep.tensor_bytes) / 1e9 / rep.fill_s : 0.0)
                << ",\"single_reader\":" << (rep.single_reader ? "true" : "false") << ",\"direct_io\":" << (rep.direct_io ? "true" : "false")
                << ",\"read_calls\":" << rep.read_calls << ",\"pause_s\":" << JsonDouble(rep.pause_s)
                << ",\"data_path\":" << JsonString(rep.data_path) << ",\"punched_bytes\":" << rep.punched_bytes
                << ",\"swarm\":" << (rep.swarm ? "true" : "false") << ",\"swarm_bytes\":" << rep.swarm_bytes << ",\"swarm_runs\":" << rep.swarm_runs
                << ",\"swarm_stats\":" << (rep.swarm_stats.empty() ? std::string("null") : rep.swarm_stats)
                << ",\"hip_free_before\":" << rep.hip_free_before << ",\"hip_free_after\":" << rep.hip_free_after
                << ",\"hip_total\":" << rep.hip_total << ",\"rss_file_before_fill\":" << rep.rss_file_before
                << ",\"rss_file_after_fill\":" << rep.rss_file_after << "}";
    }
  }
  std::cout << "}" << std::endl;

  // --mtp-probe: the stage output rows (set before the first forward: the decode graph's key folds them) and the head.
  std::unique_ptr<flerp::qwen35moe::MtpHead> mtp_head;
  std::unique_ptr<flerp_tool::MtpProbe> probe;
  float* d_probe_rows = nullptr;
  if (mtp.depth > 0) {
    const std::size_t rows = gpu->GetMaxPromptBatch();
    if (hipMalloc(reinterpret_cast<void**>(&d_probe_rows), rows * config.hidden_size * sizeof(float)) != hipSuccess) {
      std::cerr << "error: mtp probe: row buffer\n";
      return 1;
    }
    flerp::qwen35moe::SetStageIo({.d_input = nullptr, .d_output = d_probe_rows});
    mtp_head = flerp::qwen35moe::MtpHead::Create(*reader, gpu->GetModel().GetWeights(), {.max_context = context, .data_path = mtp.data_file},
                                                 &error);
    if (!mtp_head) {
      std::cerr << "error: " << error << "\n";
      return 1;
    }
    const auto& ml = mtp_head->layer();
    std::cout << "{\"event\":\"mtp_head\",\"block\":" << mtp_head->block() << ",\"depth\":" << mtp.depth
              << ",\"weight_bytes\":" << mtp_head->weight_bytes() << ",\"buffer_bytes\":" << mtp_head->buffer_bytes()
              << ",\"load_s\":" << mtp_head->load_s() << ",\"own_embedding\":" << (mtp_head->own_embedding() ? "true" : "false")
              << ",\"experts\":" << ml.moe.expert_count << ",\"used\":" << ml.moe.expert_used << ",\"expert_ff\":" << ml.moe.expert_ff
              << ",\"shared_ff\":" << ml.moe.shared_ff << ",\"data_file\":" << JsonString(mtp.data_file) << "}" << std::endl;
    probe = std::make_unique<flerp_tool::MtpProbe>(*mtp_head, mtp.depth, config.hidden_size);
  }

  std::vector<char> arm_list;
  for (const char c : arms) if (std::strchr("ABDEFXGYLCR", c) != nullptr && c != 0) arm_list.push_back(c);
  if (arm_list.empty()) arm_list.push_back('-');
  const bool decode_prompt_flag = decode_prompt;
  for (const char arm : arm_list) {
  if (arm != '-') {
    flerp::qwen35moe::SetMoeGrouped((arm == 'B' || arm == 'F' || arm == 'G' || arm == 'L' || arm == 'C' || arm == 'R') ? 1 : ((arm == 'E' || arm == 'X' || arm == 'Y') ? 2 : 0));
    flerp::qwen35moe::SetDenseGrouped((arm == 'F' || arm == 'G' || arm == 'L' || arm == 'C' || arm == 'R') ? 1 : ((arm == 'X' || arm == 'Y') ? 2 : 0));
    flerp::qwen35moe::SetAttnGqa((arm == 'G' || arm == 'Y' || arm == 'L' || arm == 'C' || arm == 'R') ? 1 : 0);
    flerp::kgrouped::SetLiveTiles(arm == 'L' ? 1 : 0);   // PREFILL-KPROF (9506f4f116): L = G + live-tile items
    flerp::qwen35moe::SetMoeCombineFast(arm == 'C' ? 1 : 0);   // MOE-COMBINE (2dc55e67f0): C = G + combine + one ToF16
    flerp::moe::SetRouterTiled(arm == 'R' ? 1 : 0);   // ROUTER-GEMM (c6473d6063): R = G + the tiled prompt router
    decode_prompt = decode_prompt_flag || arm == 'D';
    std::cout << "{\"event\":\"arm\",\"arm\":\"" << arm << "\"}" << std::endl;
  }
  for (std::size_t index = 0; index < prompts.size(); ++index) {
    std::ifstream file(prompts[index].source, std::ios::binary);
    if (!file) {
      std::cerr << "error: cannot read " << prompts[index].source << "\n";
      return 1;
    }
    const std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::vector<TokenId> ids;
    if (prompts[index].is_ids) {
      std::istringstream in(content);
      for (long long id = 0; in >> id;) ids.push_back(static_cast<TokenId>(id));
    } else {
      ids = gufo::tokenization::QwenChatTemplate::EncodeRendered(tokenizer, content, 0, content.size(), {}, {});
    }
    if (!stage_bench && (ids.empty() || ids.size() + max_tokens >= context)) {
      std::cerr << "error: prompt " << index << " has " << ids.size() << " tokens; with " << max_tokens
                << " new tokens it must fit a context of " << context << "\n";
      return 1;
    }
    if (stage_bench) {
      // one bench, on the first ids file; a prompt with too few ids runs the synthetic phase only
      return StageBench(*gpu, ids, bench, context) ? 0 : 1;
    }
    if (tokenize_only) {
      std::cout << "{\"event\":\"tokenized\",\"prompt\":" << index << ",\"source\":"
                << JsonString(prompts[index].source) << ",\"prompt_ids\":" << JsonIds(ids) << "}" << std::endl;
      continue;
    }
    if (trace) {
      for (auto& layer : trace_layers) layer = std::min<std::uint32_t>(layer, config.num_layers - 1);
      std::sort(trace_layers.begin(), trace_layers.end());
      trace_layers.erase(std::unique(trace_layers.begin(), trace_layers.end()), trace_layers.end());
      try {
        std::vector<TokenId> forced;
        std::size_t tokens_wanted = max_tokens;
        if (!force_files.empty()) {
          std::ifstream in(force_files[index]);
          for (long long id = 0; in >> id;) forced.push_back(static_cast<TokenId>(id));
          tokens_wanted = *std::max_element(record_at.begin(), record_at.end()) + 1;
          if (forced.size() + 1 < tokens_wanted || ids.size() + tokens_wanted >= context) {
            std::cerr << "error: prompt " << index << ": the forced continuation has " << forced.size() << " tokens, "
                      << tokens_wanted - 1 << " are needed, within the context\n";
            return 1;
          }
        }
        const auto* force = force_files.empty() ? nullptr : &forced;
        const bool ok = cpu ? TraceCpu(*reader, tokenizer, ids, tokens_wanted, trace_top, context, index, prompts[index].source, force, record_at)
                            : Trace(*gpu, ids, tokens_wanted, trace_top, trace_layers, decode_prompt, index, prompts[index].source, force, record_at);
        if (!ok) return 1;
      } catch (const std::exception& exception) {
        std::cerr << "error: prompt " << index << ": " << exception.what() << "\n";
        return 1;
      }
      continue;
    }
    Result result;
    try {
      if (probe) {
        if (index == 0 && !mtp.dump_path.empty()) probe->ArmDump(mtp.dump_path, mtp.dump_frontiers);
        result = ProbeRun(*gpu, ids, max_tokens, *probe, d_probe_rows);
      } else {
        result = cpu ? Run(*host, ids, max_tokens) : Run(*gpu, ids, max_tokens);
      }
    } catch (const std::exception& exception) {
      std::cerr << "error: prompt " << index << ": " << exception.what() << "\n";
      return 1;
    }
    const double decode_s = result.total_s - result.first_token_s;
    const double decode_tps =
        result.ids.size() > 1 && decode_s > 0.0 ? static_cast<double>(result.ids.size() - 1) / decode_s : 0.0;
    std::cout << "{\"event\":\"generated\",\"prompt\":" << index << ",\"source\":" << JsonString(prompts[index].source)
              << ",\"prompt_ids\":" << JsonIds(ids) << ",\"ids\":" << JsonIds(result.ids)
              << ",\"text\":" << JsonString(result.text) << ",\"first_token_s\":" << result.first_token_s
              << ",\"total_s\":" << result.total_s << ",\"decode_tok_s\":" << decode_tps << "}" << std::endl;
    if (probe) std::cout << probe->Json(index) << std::endl;
  }
  }
  return 0;
}
