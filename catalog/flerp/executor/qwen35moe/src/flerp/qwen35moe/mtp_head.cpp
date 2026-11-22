// SPDX-License-Identifier: MIT
// See mtp_head.hpp.
#if defined(ENGINE_ENABLE_HIP)
#include "flerp/qwen35moe/mtp_head.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <gufo/ext/gguf_reader.hpp>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "flerp/draft_coarse/draft_coarse.hpp"
#include "flerp/qwen35moe/moe_ffn.hpp"
#include "src/families/qwen/hip_common/attention_policy.hpp"
#include "src/families/qwen/kernels/ops.hpp"

namespace flerp::qwen35moe {
namespace {

using gufo::core::GgmlType;
using gufo::models::QwenTensorRef;

void Ok(hipError_t status, const char* what) {
  if (status != hipSuccess) throw std::runtime_error(std::string("mtp head: ") + what + ": " + hipGetErrorString(status));
}

struct Wanted {
  std::string name;
  QwenTensorRef* ref;
  std::size_t elements;   // 0: any (checked elsewhere)
  bool required;
  bool norm;              // read as float rows by the kernels: must be F32
};

}  // namespace

MtpHead::~MtpHead() {
  if (stream_ != nullptr) (void)hipStreamSynchronize(stream_);
  if (coarse_) std::cerr << CoarseStatsJson() << std::endl;
  if (coarse_trace_ != nullptr) std::fclose(coarse_trace_);
  coarse_.reset();
  if (h_cres_ != nullptr) (void)hipHostFree(h_cres_);
  for (void* p : buffers_) (void)hipFree(p);
  if (d_weights_ != nullptr) (void)hipFree(d_weights_);
  if (h_logits_ != nullptr) (void)hipHostFree(h_logits_);
  if (stream_ != nullptr) (void)hipStreamDestroy(stream_);
}

std::unique_ptr<MtpHead> MtpHead::Create(const gufo::core::GgufReader& reader, const gufo::models::QwenModelWeights& target,
                                         const MtpHeadOptions& options, std::string* error) {
  const auto fail = [&](const std::string& why) -> std::unique_ptr<MtpHead> {
    if (error != nullptr) *error = "mtp head: " + why;
    return nullptr;
  };
  const auto t0 = std::chrono::steady_clock::now();
  std::unique_ptr<MtpHead> head(new MtpHead());
  auto& h = *head;
  h.config_ = target.config;
  const auto& c = h.config_;
  const std::string arch = c.architecture + ".";
  const std::uint32_t nextn = reader.GetMetadataUint32(arch + "nextn_predict_layers").value_or(0);
  if (nextn != 1) return fail("the header's " + arch + "nextn_predict_layers is " + std::to_string(nextn) + ", this head runs exactly 1");
  if (!target.HasHead() || target.output.empty() || target.output_norm.empty()) {
    return fail("the target weights have no output head (the MTP head runs beside the last stage)");
  }
  if (options.max_context == 0 || options.max_context > c.context_length) return fail("max_context out of range");
  h.block_ = c.num_layers;
  h.max_context_ = options.max_context;
  const std::string p = "blk." + std::to_string(h.block_) + ".";
  const std::size_t H = c.hidden_size, A = c.AttentionSize(), V = c.vocab_size;
  const std::size_t KV = static_cast<std::size_t>(c.num_key_value_heads) * c.head_dim;
  const auto expert_count = reader.GetMetadataUint32(arch + "expert_count").value_or(0);
  const auto expert_used = reader.GetMetadataUint32(arch + "expert_used_count").value_or(0);
  if (expert_count == 0 || expert_used == 0 || expert_used > expert_count || expert_count > 1024 || expert_used > 32) {
    return fail("unusable expert_count / expert_used_count");
  }

  auto& l = h.layer_;
  QwenTensorRef embed_tokens, shared_head_head;
  std::vector<Wanted> want = {
      {p + "attn_norm.weight", &l.attn_norm, H, true, true},
      {p + "post_attention_norm.weight", &l.ffn_norm, H, true, true},
      {p + "attn_q.weight", &l.attn_q, 2 * A * H, true, false},
      {p + "attn_k.weight", &l.attn_k, KV * H, true, false},
      {p + "attn_v.weight", &l.attn_v, KV * H, true, false},
      {p + "attn_output.weight", &l.attn_output, H * A, true, false},
      {p + "attn_q_norm.weight", &l.attn_q_norm, c.head_dim, true, true},
      {p + "attn_k_norm.weight", &l.attn_k_norm, c.head_dim, true, true},
      {p + "ffn_gate_inp.weight", &l.ffn_gate_inp, static_cast<std::size_t>(expert_count) * H, true, true},
      {p + "ffn_gate_exps.weight", &l.ffn_gate_exps, 0, true, false},
      {p + "ffn_up_exps.weight", &l.ffn_up_exps, 0, true, false},
      {p + "ffn_down_exps.weight", &l.ffn_down_exps, 0, true, false},
      {p + "ffn_gate_inp_shexp.weight", &l.ffn_gate_inp_shexp, H, false, true},
      {p + "ffn_gate_shexp.weight", &l.ffn_gate_shexp, 0, false, false},
      {p + "ffn_up_shexp.weight", &l.ffn_up_shexp, 0, false, false},
      {p + "ffn_down_shexp.weight", &l.ffn_down_shexp, 0, false, false},
      {p + "nextn.eh_proj.weight", &h.eh_proj_, 2 * H * H, true, false},
      {p + "nextn.enorm.weight", &h.enorm_, H, true, true},
      {p + "nextn.hnorm.weight", &h.hnorm_, H, true, true},
      {p + "nextn.shared_head_norm.weight", &h.shared_head_norm_, H, false, true},
      {p + "nextn.embed_tokens.weight", &embed_tokens, V * H, false, false},
      {p + "nextn.shared_head_head.weight", &shared_head_head, V * H, false, false},
  };
  if (target.token_embd.empty()) want.push_back({"token_embd.weight", &h.embed_, V * H, true, false});

  // Host references first (the reader's mapping), then one device allocation and a copy of each.
  const auto regions = reader.GetMappedRegions();
  struct Copy {
    QwenTensorRef* ref;
    std::uint64_t file_offset;
    std::size_t bytes;
    std::size_t device_offset;
    std::string path;
  };
  std::vector<Copy> copies;
  std::size_t total = 0;
  for (const auto& w : want) {
    const auto* t = reader.FindTensor(w.name);
    if (t == nullptr || t->data == nullptr) {
      if (w.required) return fail("missing " + w.name);
      continue;
    }
    QwenTensorRef ref{.data = t->data, .type = t->type, .num_elements = static_cast<std::size_t>(t->ElementCount())};
    if (w.elements != 0 && ref.num_elements != w.elements) {
      return fail(w.name + ": " + std::to_string(ref.num_elements) + " elements, expected " + std::to_string(w.elements));
    }
    if (w.norm && ref.type != GgmlType::kF32) return fail(w.name + " is not F32");
    const std::size_t bytes = ref.EncodedSizeBytes();
    if (bytes == 0) return fail(w.name + ": unknown encoded size");
    const auto addr = reinterpret_cast<std::uintptr_t>(t->data);
    const gufo::core::GgufMappedRegion* region = nullptr;
    for (const auto& r : regions) {
      const auto base = reinterpret_cast<std::uintptr_t>(r.data);
      if (addr >= base && addr - base + bytes <= r.size) region = &r;
    }
    if (region == nullptr) return fail(w.name + " is outside the reader's mapping");
    const std::uint64_t off = addr - reinterpret_cast<std::uintptr_t>(region->data);
    total = (total + 255) & ~std::size_t{255};
    copies.push_back({w.ref, off, bytes, total,
                      options.data_path.empty() ? region->source_path.string() : options.data_path});
    total += bytes;
    *w.ref = ref;
  }

  // The expert stacks' widths, as weights.cpp LoadMoeBlock reads them.
  if (l.ffn_up_exps.num_elements % (static_cast<std::size_t>(expert_count) * H) != 0) return fail("ffn_up_exps shape");
  const std::size_t expert_ff = l.ffn_up_exps.num_elements / (static_cast<std::size_t>(expert_count) * H);
  if (l.ffn_gate_exps.num_elements != l.ffn_up_exps.num_elements || l.ffn_down_exps.num_elements != l.ffn_up_exps.num_elements) {
    return fail("expert stack shapes differ");
  }
  std::size_t shared_ff = 0;
  if (!l.ffn_up_shexp.empty()) {
    shared_ff = l.ffn_up_shexp.num_elements / H;
    if (l.ffn_gate_inp_shexp.empty() || l.ffn_gate_shexp.num_elements != shared_ff * H || l.ffn_down_shexp.num_elements != shared_ff * H) {
      return fail("shared expert shapes");
    }
  }
  l.is_full_attention = true;
  l.moe.expert_count = expert_count;
  l.moe.expert_used = expert_used;
  l.moe.expert_ff = static_cast<std::uint32_t>(expert_ff);
  l.moe.shared_ff = static_cast<std::uint32_t>(shared_ff);

  try {
    Ok(hipMalloc(&h.d_weights_, total), "weights allocation");
    h.weight_bytes_ = total;
    constexpr std::size_t kPiece = std::size_t{64} << 20;
    std::vector<char> staging(kPiece);
    std::string open_path;
    int fd = -1;
    for (auto& cp : copies) {
      if (cp.path != open_path) {
        if (fd >= 0) ::close(fd);
        fd = ::open(cp.path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) throw std::runtime_error("mtp head: cannot open " + cp.path);
        open_path = cp.path;
      }
      char* dst = static_cast<char*>(h.d_weights_) + cp.device_offset;
      for (std::size_t done = 0; done < cp.bytes;) {
        const std::size_t n = std::min(kPiece, cp.bytes - done);
        std::size_t got = 0;
        while (got < n) {
          const ssize_t r = ::pread(fd, staging.data() + got, n - got, static_cast<off_t>(cp.file_offset + done + got));
          if (r <= 0) {
            ::close(fd);
            throw std::runtime_error("mtp head: short read of " + cp.path);
          }
          got += static_cast<std::size_t>(r);
        }
        Ok(hipMemcpy(dst + done, staging.data(), n, hipMemcpyHostToDevice), "weights copy");
        done += n;
      }
      cp.ref->data = dst;
      cp.ref->available_bytes = cp.bytes;
    }
    if (fd >= 0) ::close(fd);
  } catch (const std::exception& e) {
    return fail(e.what());
  }

  h.own_embedding_ = !embed_tokens.empty() || target.token_embd.empty();
  h.embed_ = !embed_tokens.empty() ? embed_tokens : (target.token_embd.empty() ? h.embed_ : target.token_embd);
  h.head_ = !shared_head_head.empty() ? shared_head_head : target.output;
  h.output_norm_ = target.output_norm;
  if (h.shared_head_norm_.empty()) h.shared_head_norm_ = target.output_norm;

  try {
    Ok(hipStreamCreateWithFlags(&h.stream_, hipStreamNonBlocking), "stream");
    const auto alloc = [&](auto*& ptr, std::size_t bytes) {
      void* q = nullptr;
      Ok(hipMalloc(&q, bytes), "buffer allocation");
      Ok(hipMemset(q, 0, bytes), "buffer clear");
      h.buffers_.push_back(q);
      h.buffer_bytes_ += bytes;
      ptr = static_cast<std::remove_reference_t<decltype(ptr)>>(q);
    };
    const std::size_t total_kv = KV * h.max_context_;
    h.moe_floats_ = l.moe.ScratchFloats(H);
    alloc(h.d_h_, H * sizeof(float));
    alloc(h.d_embedding_, H * sizeof(float));
    alloc(h.d_fusion_, 2 * H * sizeof(float));
    alloc(h.d_hidden_, H * sizeof(float));
    alloc(h.d_normed_, H * sizeof(float));
    alloc(h.d_qg_, 2 * A * sizeof(float));
    alloc(h.d_q_, A * sizeof(float));
    alloc(h.d_k_, KV * sizeof(float));
    alloc(h.d_v_, KV * sizeof(float));
    alloc(h.d_gate_, A * sizeof(float));
    alloc(h.d_context_, A * sizeof(float));
    alloc(h.d_attn_out_, H * sizeof(float));
    alloc(h.d_ffn_out_, H * sizeof(float));
    alloc(h.d_moe_a_, h.moe_floats_ * sizeof(float));
    alloc(h.d_moe_b_, h.moe_floats_ * sizeof(float));
    alloc(h.d_moe_c_, h.moe_floats_ * sizeof(float));
    alloc(h.d_feedback_, H * sizeof(float));
    alloc(h.d_logits_, V * sizeof(float));
    alloc(h.d_kv_, 2 * total_kv * sizeof(float));
    alloc(h.d_kv_f16_, 2 * total_kv * sizeof(std::uint16_t));
    alloc(h.d_split_k_, gufo::hip::detail::DecodeAttentionScratchElements(c.num_attention_heads, c.head_dim) * sizeof(float));
    alloc(h.d_token_, sizeof(std::uint32_t));
  } catch (const std::exception& e) {
    return fail(e.what());
  }

  // FLERP_MTP_COARSE=q2|q4: the coarse copy of the head (kernel.flerp.draft_coarse), built here once.
  if (const char* cv = std::getenv("FLERP_MTP_COARSE"); cv != nullptr && (std::strcmp(cv, "q2") == 0 || std::strcmp(cv, "q4") == 0)) {
    const auto env_num = [](const char* name, double def) {
      const char* v = std::getenv(name);
      return v != nullptr ? std::strtod(v, nullptr) : def;
    };
    const int bits = cv[1] - '0';
    const int keep = static_cast<int>(env_num("FLERP_MTP_COARSE_BLOCKS", 8));
    const float k = static_cast<float>(env_num("FLERP_MTP_COARSE_K", 8.0));
    const char* chk = std::getenv("FLERP_MTP_COARSE_CHECK");
    const auto tb = std::chrono::steady_clock::now();
    std::string why;
    h.coarse_ = flerp::draft_coarse::Coarse::Build(h.head_.data, h.head_.type, static_cast<int>(V), static_cast<int>(H), bits, keep,
                                                   k, h.stream_, &why);
    if (!h.coarse_) return fail(why);
    h.cstats_.build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tb).count();
    h.coarse_check_ = chk != nullptr && std::strcmp(chk, "0") != 0;
    try {
      const auto alloc = [&](auto*& ptr, std::size_t bytes) {
        void* q = nullptr;
        Ok(hipMalloc(&q, bytes), "buffer allocation");
        Ok(hipMemset(q, 0, bytes), "buffer clear");
        h.buffers_.push_back(q);
        h.buffer_bytes_ += bytes;
        ptr = static_cast<std::remove_reference_t<decltype(ptr)>>(q);
      };
      alloc(h.d_cres_, sizeof(flerp::draft_coarse::Result));
      if (h.coarse_check_) alloc(h.d_check_logits_, V * sizeof(float));
      Ok(hipHostMalloc(reinterpret_cast<void**>(&h.h_cres_), sizeof(flerp::draft_coarse::Result)), "pinned result");
    } catch (const std::exception& e) {
      return fail(e.what());
    }
    h.buffer_bytes_ += h.coarse_->bytes();
  }
  h.load_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return head;
}

std::uint32_t MtpHead::StepTarget(std::uint32_t token, const float* d_target_prenorm_row, std::uint32_t pos, bool logits) {
  gufo::hip::LaunchRMSNorm(d_target_prenorm_row, static_cast<const float*>(output_norm_.data), d_h_, config_.hidden_size, 1e-6F, stream_);
  return Run(token, d_h_, pos, logits);
}

std::uint32_t MtpHead::StepZero(std::uint32_t token, std::uint32_t pos, bool logits) {
  Ok(hipMemsetAsync(d_h_, 0, config_.hidden_size * sizeof(float), stream_), "zero h");
  return Run(token, d_h_, pos, logits);
}

std::uint32_t MtpHead::StepFeedback(std::uint32_t token, std::uint32_t pos, bool logits) {
  Ok(hipMemcpyAsync(d_h_, d_feedback_, config_.hidden_size * sizeof(float), hipMemcpyDeviceToDevice, stream_), "feedback h");
  return Run(token, d_h_, pos, logits);
}

std::uint32_t MtpHead::Run(std::uint32_t token, const float* h_normed, std::uint32_t pos, bool logits) {
  using namespace gufo::hip;
  if (pos >= max_context_) throw std::out_of_range("mtp head: position past the head's context");
  const auto& c = config_;
  const std::size_t H = c.hidden_size, A = c.AttentionSize(), V = c.vocab_size;
  const std::size_t KV = static_cast<std::size_t>(c.num_key_value_heads) * c.head_dim;
  const std::size_t total_kv = KV * max_context_;
  const auto f32 = [](const QwenTensorRef& t) { return static_cast<const float*>(t.data); };

  // x = concat(enorm(embed(token)), hnorm(h)); y = eh_proj . x
  LaunchEmbeddingLookup(embed_.data, embed_.type, token, d_embedding_, H, stream_);
  LaunchRMSNorm(d_embedding_, f32(enorm_), d_fusion_, H, 1e-6F, stream_);
  LaunchRMSNorm(h_normed, f32(hnorm_), d_fusion_ + H, H, 1e-6F, stream_);
  LaunchGEMV(eh_proj_.data, eh_proj_.type, d_fusion_, d_hidden_, H, 2 * H, stream_);

  // gated attention over the head's own KV
  LaunchRMSNorm(d_hidden_, f32(layer_.attn_norm), d_normed_, H, 1e-6F, stream_);
  LaunchFusedQKVProjections(layer_.attn_q.data, layer_.attn_q.type, layer_.attn_k.data, layer_.attn_k.type, layer_.attn_v.data,
                            layer_.attn_v.type, d_normed_, d_qg_, d_k_, d_v_, 2 * A, KV, H, stream_);
  LaunchUnpackQG(d_qg_, d_q_, d_gate_, c.num_attention_heads, c.head_dim, stream_);
  LaunchPerHeadRMSNorm(d_q_, f32(layer_.attn_q_norm), d_q_, c.num_attention_heads, c.head_dim, 1e-6F, stream_);
  LaunchPerHeadRMSNorm(d_k_, f32(layer_.attn_k_norm), d_k_, c.num_key_value_heads, c.head_dim, 1e-6F, stream_);
  LaunchRoPE(d_q_, d_k_, c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.rotary_dim, pos, c.rope_theta, stream_, nullptr);
  LaunchAttention(d_q_, d_k_, d_v_, d_gate_, d_kv_, d_kv_ + total_kv, d_kv_f16_, static_cast<std::uint16_t*>(d_kv_f16_) + total_kv,
                  d_context_, 0, pos, max_context_, c.num_attention_heads, c.num_key_value_heads, c.head_dim, stream_, d_split_k_);
  LaunchGEMV(layer_.attn_output.data, layer_.attn_output.type, d_context_, d_attn_out_, H, A, stream_);
  LaunchResidualAdd(d_hidden_, d_attn_out_, d_hidden_, H, stream_);

  // the block's mixture of experts
  LaunchRMSNorm(d_hidden_, f32(layer_.ffn_norm), d_normed_, H, 1e-6F, stream_);
  LaunchMoeFfn(layer_, d_normed_, d_ffn_out_, 1, H, {.a = d_moe_a_, .b = d_moe_b_, .c = d_moe_c_, .floats_per_token = moe_floats_},
               stream_, block_, false);
  LaunchResidualAdd(d_hidden_, d_ffn_out_, d_hidden_, H, stream_);
  LaunchRMSNorm(d_hidden_, f32(shared_head_norm_), d_feedback_, H, 1e-6F, stream_);

  std::uint32_t out = 0;
  if (logits) {
    if (coarse_) return HeadCoarse();
    LaunchGEMV(head_.data, head_.type, d_feedback_, d_logits_, V, H, stream_);
    LaunchGPUArgmax(d_logits_, d_token_, V, stream_);
    Ok(hipMemcpyAsync(&out, d_token_, sizeof(out), hipMemcpyDeviceToHost, stream_), "token copy");
  }
  Ok(hipStreamSynchronize(stream_), "step");
  return out;
}

// The head as before: every logit from the GGUF rows, then the GPU argmax.
std::uint32_t MtpHead::HeadExact() {
  using namespace gufo::hip;
  const std::size_t H = config_.hidden_size, V = config_.vocab_size;
  std::uint32_t out = 0;
  LaunchGEMV(head_.data, head_.type, d_feedback_, d_logits_, V, H, stream_);
  LaunchGPUArgmax(d_logits_, d_token_, V, stream_);
  Ok(hipMemcpyAsync(&out, d_token_, sizeof(out), hipMemcpyDeviceToHost, stream_), "token copy");
  Ok(hipStreamSynchronize(stream_), "step");
  return out;
}

// Coarse scores, the kept blocks exact, the choice; the exact head when the coarse margin is under tau.
std::uint32_t MtpHead::HeadCoarse() {
  using namespace gufo::hip;
  const std::size_t H = config_.hidden_size, V = config_.vocab_size;
  coarse_->Score(d_feedback_, d_logits_, d_cres_, stream_);
  Ok(hipMemcpyAsync(h_cres_, d_cres_, sizeof(*h_cres_), hipMemcpyDeviceToHost, stream_), "coarse result copy");
  Ok(hipStreamSynchronize(stream_), "coarse step");
  const flerp::draft_coarse::Result r = *h_cres_;
  ++cstats_.steps;
  std::vector<float> coarse_row;
  std::vector<int> kept;
  if (coarse_check_) {  // the coarse row (kept blocks exact) before a fallback overwrites it
    coarse_row.resize(V);
    kept.resize(static_cast<std::size_t>(coarse_->keep()));
    Ok(hipMemcpy(coarse_row.data(), d_logits_, V * sizeof(float), hipMemcpyDeviceToHost), "check copy");
    Ok(hipMemcpy(kept.data(), coarse_->chosen(), kept.size() * sizeof(int), hipMemcpyDeviceToHost), "check copy");
  }
  std::uint32_t out = r.id;
  if (r.fallback != 0) {
    ++cstats_.fallbacks;
    out = HeadExact();
  }
  if (coarse_check_) {
    ++cstats_.checked;
    LaunchGEMV(head_.data, head_.type, d_feedback_, d_check_logits_, V, H, stream_);
    LaunchGPUArgmax(d_check_logits_, d_token_, V, stream_);
    std::uint32_t exact_id = 0;
    std::vector<float> exact(V);
    Ok(hipMemcpyAsync(&exact_id, d_token_, sizeof(exact_id), hipMemcpyDeviceToHost, stream_), "check copy");
    Ok(hipMemcpyAsync(exact.data(), d_check_logits_, V * sizeof(float), hipMemcpyDeviceToHost, stream_), "check copy");
    Ok(hipStreamSynchronize(stream_), "check");
    if (out != exact_id) ++cstats_.wrong_ids;
    std::vector<char> in(V, 0);
    for (const int b : kept) {
      for (std::size_t t = 0; b >= 0 && t < flerp::draft_coarse::kBlockTokens; ++t) {
        const std::size_t tok = static_cast<std::size_t>(b) * flerp::draft_coarse::kBlockTokens + t;
        if (tok >= V) break;
        in[tok] = 1;
        if (std::memcmp(&coarse_row[tok], &exact[tok], sizeof(float)) != 0) ++cstats_.rescore_mismatch;
      }
    }
    // exact top-2 (finite only, lower id first on a tie)
    std::size_t t1 = V, t2 = V;
    for (std::size_t t = 0; t < V; ++t) {
      const float v = exact[t];
      if (!std::isfinite(v)) continue;
      if (t1 == V || v > exact[t1]) {
        t2 = t1;
        t1 = t;
      } else if (t2 == V || v > exact[t2]) {
        t2 = t;
      }
    }
    const bool m1 = t1 < V && in[t1] == 0;
    const bool m2 = m1 || (t2 < V && in[t2] == 0);
    if (m1) ++cstats_.miss_top1;
    if (m2) ++cstats_.miss_top2;
    if (m2 && r.fallback == 0) ++cstats_.unsafe;
    // V10-BLOCKBOUND: per-row and per-block certificates (gate/draft-coarse-blockbound.toml). The head multiplies the
    // F32 x directly (QuantWarpBlockDot), so |exact_t - coarse_t| <= row_err[t] * ||x||; slack covers FP32 summation.
    {
      constexpr std::size_t kB = flerp::draft_coarse::kBlockTokens;
      const std::vector<float>& re = coarse_->row_err();
      if (block_err_.empty()) {
        block_err_.assign((V + kB - 1) / kB, 0.0F);
        for (std::size_t t = 0; t < V; ++t) block_err_[t / kB] = std::max(block_err_[t / kB], re[t]);
      }
      const double xn = static_cast<double>(r.xnorm);
      const double slack = 1e-5 * (std::fabs(static_cast<double>(r.best)) + 1.0);
      double ub_row = -INFINITY, ub_block = -INFINITY;
      for (std::size_t b = 0; b < block_err_.size(); ++b) {
        if (in[b * kB] != 0) continue;
        double cmax = -INFINITY;
        for (std::size_t t = b * kB; t < std::min(V, (b + 1) * kB); ++t) {
          if (!std::isfinite(coarse_row[t])) continue;
          const double c = coarse_row[t], bound = static_cast<double>(re[t]) * xn;
          cmax = std::max(cmax, c);
          ub_row = std::max(ub_row, c + bound);
          if (std::isfinite(exact[t])) {
            const double d = std::fabs(c - static_cast<double>(exact[t]));
            if (d > bound + slack) ++cstats_.bound_violations;
            if (bound > 0.0) cstats_.max_ratio = std::max(cstats_.max_ratio, d / bound);
          }
        }
        ub_block = std::max(ub_block, cmax + static_cast<double>(block_err_[b]) * xn);
      }
      const double best = r.best, second = r.second;
      const bool c1b = ub_block + slack < best, c2b = ub_block + slack < second;
      const bool c1r = ub_row + slack < best, c2r = ub_row + slack < second;
      cstats_.cert1_block += c1b; cstats_.cert2_block += c2b; cstats_.cert1_row += c1r; cstats_.cert2_row += c2r;
      if (((c1b || c1r) && m1) || ((c2b || c2r) && m2)) ++cstats_.false_cert;
      if (coarse_trace_ == nullptr) {
        if (const char* tp = std::getenv("FLERP_MTP_COARSE_TRACE"); tp != nullptr && *tp != 0) coarse_trace_ = std::fopen(tp, "w");
      }
      if (coarse_trace_ != nullptr) {
        std::fprintf(coarse_trace_,
                     "{\"step\":%llu,\"best\":%.9g,\"second\":%.9g,\"xnorm\":%.9g,\"ub_block\":%.9g,\"ub_row\":%.9g,"
                     "\"outside\":%.9g,\"tau\":%.9g,\"fallback\":%u,\"miss1\":%d,\"miss2\":%d,\"c1b\":%d,\"c2b\":%d,\"c1r\":%d,\"c2r\":%d}\n",
                     static_cast<unsigned long long>(cstats_.checked), best, second, xn, ub_block, ub_row,
                     static_cast<double>(r.outside), static_cast<double>(r.tau), r.fallback, m1 ? 1 : 0, m2 ? 1 : 0,
                     c1b ? 1 : 0, c2b ? 1 : 0, c1r ? 1 : 0, c2r ? 1 : 0);
      }
    }
    const double scale = static_cast<double>(coarse_->err_rms()) * static_cast<double>(r.xnorm);
    if (scale > 0.0) {
      for (std::size_t t = 0; t < V; ++t) {
        if (in[t] != 0 || !std::isfinite(exact[t]) || !std::isfinite(coarse_row[t])) continue;
        cstats_.max_z = std::max(cstats_.max_z, std::fabs(static_cast<double>(coarse_row[t]) - exact[t]) / scale);
      }
    }
  }
  return out;
}

std::string MtpHead::CoarseStatsJson() const {
  std::ostringstream o;
  o << "{\"event\":\"mtp_coarse\",\"on\":" << (coarse_ ? "true" : "false");
  if (coarse_) {
    o << ",\"bits\":" << coarse_->bits() << ",\"keep\":" << coarse_->keep() << ",\"k\":" << coarse_->k_tau()
      << ",\"err_rms\":" << coarse_->err_rms() << ",\"bytes\":" << coarse_->bytes() << ",\"build_s\":" << cstats_.build_s
      << ",\"check\":" << (coarse_check_ ? "true" : "false");
  }
  o << ",\"steps\":" << cstats_.steps << ",\"fallbacks\":" << cstats_.fallbacks << ",\"checked\":" << cstats_.checked
    << ",\"wrong_ids\":" << cstats_.wrong_ids << ",\"miss_top1\":" << cstats_.miss_top1 << ",\"miss_top2\":" << cstats_.miss_top2
    << ",\"unsafe\":" << cstats_.unsafe << ",\"rescore_mismatch\":" << cstats_.rescore_mismatch << ",\"max_z\":" << cstats_.max_z
    << ",\"cert1_block\":" << cstats_.cert1_block << ",\"cert2_block\":" << cstats_.cert2_block << ",\"cert1_row\":" << cstats_.cert1_row
    << ",\"cert2_row\":" << cstats_.cert2_row << ",\"false_cert\":" << cstats_.false_cert << ",\"bound_violations\":" << cstats_.bound_violations
    << ",\"max_ratio\":" << cstats_.max_ratio << "}";
  return o.str();
}

MtpHead::Top2 MtpHead::LastTop2() {
  const std::size_t V = config_.vocab_size;
  if (h_logits_ == nullptr) Ok(hipHostMalloc(reinterpret_cast<void**>(&h_logits_), V * sizeof(float)), "pinned logits");
  Ok(hipMemcpyAsync(h_logits_, d_logits_, V * sizeof(float), hipMemcpyDeviceToHost, stream_), "logits copy");
  Ok(hipStreamSynchronize(stream_), "logits copy");
  Top2 t;
  t.best = h_logits_[0];
  t.second = -3.0e38F;
  for (std::uint32_t i = 1; i < V; ++i) {
    const float v = h_logits_[i];
    if (v > t.best) {
      t.second = t.best;
      t.best = v;
      t.id = i;
    } else if (v > t.second) {
      t.second = v;
    }
  }
  return t;
}

std::vector<float> MtpHead::CopyHidden() const {
  std::vector<float> v(config_.hidden_size);
  Ok(hipMemcpy(v.data(), d_feedback_, v.size() * sizeof(float), hipMemcpyDeviceToHost), "hidden copy");
  return v;
}

std::vector<float> MtpHead::CopyLogits() const {
  std::vector<float> v(config_.vocab_size);
  Ok(hipMemcpy(v.data(), d_logits_, v.size() * sizeof(float), hipMemcpyDeviceToHost), "logits copy");
  return v;
}

std::vector<float> MtpHead::CopyInput() const {
  std::vector<float> v(config_.hidden_size);
  Ok(hipMemcpy(v.data(), d_h_, v.size() * sizeof(float), hipMemcpyDeviceToHost), "input copy");
  return v;
}

}  // namespace flerp::qwen35moe
#endif  // defined(ENGINE_ENABLE_HIP)
