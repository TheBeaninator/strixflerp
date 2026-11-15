// SPDX-License-Identifier: MIT
// See moe_ffn.hpp.
#if defined(ENGINE_ENABLE_HIP)
#include "flerp/qwen35moe/moe_ffn.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <gufo/ext/experimental/hip_utils.hpp>
#include <iostream>

#include "flerp/moe_kgrouped/moe_kgrouped.hpp"
#include "flerp/moe_kquant/moe_kquant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace flerp::qwen35moe {
namespace {
MoeTrace* g_trace = nullptr;
MoeDump* g_dump = nullptr;

template <typename T>
void Append(std::vector<char>& out, const T* values, std::size_t count) {
  const char* p = reinterpret_cast<const char*>(values);
  out.insert(out.end(), p, p + (count * sizeof(T)));
}
}  // namespace

namespace {
// PREFILL-ORNITH (docs/PREFILL-ORNITH.md, gate/prefill-ornith.toml): prompt chunks of at least FLERP_MOE_GROUPED_MIN
// rows (64) take kernel.flerp.moe_kgrouped when FLERP_MOE_GROUPED=1 (F16) or 2 (exact codes, hi/lo F16) and every expert and shared matrix is Q4_K or
// Q6_K: rows grouped by expert, each weight tile decoded once into LDS for all of an expert's rows. Off by default.
// The router, the trace, the dump and the combine are the same calls; only the four IndexedExperts launches change.
int g_grouped_override = -1;  // SetMoeGrouped: -1 = the environment, 0 off, 1 on
// 0 off, 1 grouped F16, 2 grouped exact (exact weight codes, hi/lo F16 activations).
int GroupedMode() {
  static const int env_mode = [] {
    const char* v = std::getenv("FLERP_MOE_GROUPED");
    if (v == nullptr || std::strcmp(v, "0") == 0) return 0;
    return std::strcmp(v, "2") == 0 ? 2 : 1;
  }();
  return g_grouped_override >= 0 ? g_grouped_override : env_mode;
}
std::size_t GroupedMin() {
  static const std::size_t n = [] {
    const char* v = std::getenv("FLERP_MOE_GROUPED_MIN");
    return v != nullptr ? static_cast<std::size_t>(std::strtoull(v, nullptr, 10)) : std::size_t{64};
  }();
  return n;
}
bool KType(gufo::core::GgmlType t, flerp::kgrouped::WType& out) {
  if (t == gufo::core::GgmlType::kQ4_K) {
    out = flerp::kgrouped::WType::kQ4K;
    return true;
  }
  if (t == gufo::core::GgmlType::kQ6_K) {
    out = flerp::kgrouped::WType::kQ6K;
    return true;
  }
  return false;
}
std::size_t g_fallbacks = 0;
unsigned* g_flag_host = nullptr;

// The grouped launches. Region c holds, from its start: h F16 [T k, F], shared h F16 [T, S], x F16 [T, H], then the
// two plans and the overflow flag (256-byte aligned). Returns false (nothing launched) when it does not fit or a type
// is not a K-quant this kernel reads; true after it ran, with *overflow telling whether an F16 value overflowed.
bool LaunchGrouped(const gufo::models::QwenLayerWeights& layer, const float* x, std::size_t tokens, std::size_t hidden,
                   MoeScratch scratch, const std::int32_t* ids, float* expert_out, float* shared_out,
                   hipStream_t stream, bool exact, bool& overflow) {
  namespace kg = flerp::kgrouped;
  const auto& moe = layer.moe;
  kg::WType tg, tu, td, sg = kg::WType::kQ4K, su = kg::WType::kQ4K, sd = kg::WType::kQ4K;
  if (!KType(layer.ffn_gate_exps.type, tg) || !KType(layer.ffn_up_exps.type, tu) || tg != tu ||
      !KType(layer.ffn_down_exps.type, td) || hidden % 256 != 0 || moe.expert_ff % 256 != 0 ||
      moe.expert_count > 1024)
    return false;
  if (moe.has_shared() && (!KType(layer.ffn_gate_shexp.type, sg) || !KType(layer.ffn_up_shexp.type, su) || sg != su ||
                           !KType(layer.ffn_down_shexp.type, sd) || moe.shared_ff % 256 != 0))
    return false;
  const std::size_t used = moe.expert_used, ff = moe.expert_ff, sff = moe.shared_ff, experts = moe.expert_count;
  const std::size_t pairs = tokens * used;
  const int bm = exact ? kg::ChooseBmExact(pairs, experts) : kg::ChooseBm(pairs, experts);
  const auto align = [](std::size_t v) { return (v + 255) & ~static_cast<std::size_t>(255); };
  char* base = reinterpret_cast<char*>(scratch.c);
  auto* h = reinterpret_cast<std::uint16_t*>(base);
  std::size_t off = align(pairs * ff * 2);
  auto* sh = reinterpret_cast<std::uint16_t*>(base + off);
  off += align(tokens * sff * 2);
  auto* xh = reinterpret_cast<std::uint16_t*>(base + off);
  off += align(tokens * hidden * 2);
  std::uint16_t *xl = nullptr, *hl = nullptr, *shl = nullptr;   // exact: the lo halves
  if (exact) {
    xl = reinterpret_cast<std::uint16_t*>(base + off);
    off += align(tokens * hidden * 2);
    hl = reinterpret_cast<std::uint16_t*>(base + off);
    off += align(pairs * ff * 2);
    shl = reinterpret_cast<std::uint16_t*>(base + off);
    off += align(tokens * sff * 2);
  }
  void* plan_mem = base + off;
  off += align(kg::PlanBytes(pairs, experts, bm));
  void* splan_mem = base + off;
  off += align(kg::PlanBytes(tokens, 1, bm));
  auto* flag = reinterpret_cast<unsigned*>(base + off);
  off += 256;
  if (off > tokens * scratch.floats_per_token * sizeof(float)) return false;
  if (g_flag_host == nullptr) HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&g_flag_host), sizeof(unsigned)));
  HIP_CHECK(hipMemsetAsync(flag, 0, sizeof(unsigned), stream));
  kg::Plan plan = kg::MakePlan(plan_mem, pairs, experts, bm);
  kg::Group(ids, plan, stream);
  if (exact) {
    kg::ToF16Split(x, xh, xl, tokens * hidden, flag, stream);
    kg::GateUpExact(tg, layer.ffn_gate_exps.data, layer.ffn_up_exps.data, xh, xl, used, plan, h, hl, ff, hidden, flag,
                    stream);
    kg::DownExact(td, layer.ffn_down_exps.data, h, hl, plan, expert_out, hidden, ff, stream);
  } else {
    kg::ToF16(x, xh, tokens * hidden, flag, stream);
    kg::GateUp(tg, layer.ffn_gate_exps.data, layer.ffn_up_exps.data, xh, used, plan, h, ff, hidden, flag, stream);
    kg::Down(td, layer.ffn_down_exps.data, h, plan, expert_out, hidden, ff, stream);
  }
  if (moe.has_shared()) {
    kg::Plan splan = kg::MakePlan(splan_mem, tokens, 1, bm);
    kg::Group(nullptr, splan, stream);
    if (exact) {
      kg::GateUpExact(sg, layer.ffn_gate_shexp.data, layer.ffn_up_shexp.data, xh, xl, 1, splan, sh, shl, sff, hidden,
                      flag, stream);
      kg::DownExact(sd, layer.ffn_down_shexp.data, sh, shl, splan, shared_out, hidden, sff, stream);
    } else {
      kg::GateUp(sg, layer.ffn_gate_shexp.data, layer.ffn_up_shexp.data, xh, 1, splan, sh, sff, hidden, flag, stream);
      kg::Down(sd, layer.ffn_down_shexp.data, sh, splan, shared_out, hidden, sff, stream);
    }
  }
  HIP_CHECK(hipMemcpyAsync(g_flag_host, flag, sizeof(unsigned), hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  overflow = *g_flag_host != 0;
  if (unsigned* df = DenseOverflowFlag(); df != nullptr) {  // dense projections: counted and reported, no fallback
    unsigned v = 0;
    HIP_CHECK(hipMemcpy(&v, df, sizeof(unsigned), hipMemcpyDeviceToHost));
    if (v != 0) {
      std::cerr << "flerp::qwen35moe: F16 overflow in a grouped dense projection (clamped)\n";
      HIP_CHECK(hipMemset(df, 0, sizeof(unsigned)));
    }
  }
  return true;
}
}  // namespace

void SetMoeTrace(MoeTrace* trace) noexcept { g_trace = trace; }
void SetMoeGrouped(int mode) noexcept { g_grouped_override = mode; }

namespace {
int g_dense_override = -1;
struct DenseScratch {
  void* p = nullptr;
  std::size_t bytes = 0;
  unsigned* flag = nullptr;
  std::size_t overflows_seen = 0;
} g_dense;
}  // namespace

void SetDenseGrouped(int mode) noexcept { g_dense_override = mode; }

int DenseGroupedMode() noexcept {
  static const int env_mode = [] {
    const char* v = std::getenv("FLERP_DENSE_GROUPED");
    if (v == nullptr || std::strcmp(v, "0") == 0) return 0;
    return std::strcmp(v, "2") == 0 ? 2 : 1;
  }();
  return g_dense_override >= 0 ? g_dense_override : env_mode;
}

bool LaunchDenseGrouped(const void* w, gufo::core::GgmlType type, const float* x, float* y, std::size_t rows,
                        std::size_t n, std::size_t k, bool exact, hipStream_t stream) {
  namespace kg = flerp::kgrouped;
  kg::WType t;
  if (!KType(type, t) || k % 256 != 0 || rows == 0) return false;
  const int bm = exact ? 64 : 128;
  const auto align = [](std::size_t v) { return (v + 255) & ~static_cast<std::size_t>(255); };
  const std::size_t need = 2 * align(rows * k * 2) + align(kg::PlanBytes(rows, 1, bm));
  if (need > g_dense.bytes) {
    if (g_dense.p != nullptr) {
      HIP_CHECK(hipStreamSynchronize(stream));
      HIP_CHECK(hipFree(g_dense.p));
    }
    HIP_CHECK(hipMalloc(&g_dense.p, need));
    g_dense.bytes = need;
  }
  if (g_dense.flag == nullptr) {
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&g_dense.flag), sizeof(unsigned)));
    HIP_CHECK(hipMemsetAsync(g_dense.flag, 0, sizeof(unsigned), stream));
  }
  char* b = static_cast<char*>(g_dense.p);
  auto* xh = reinterpret_cast<std::uint16_t*>(b);
  auto* xl = reinterpret_cast<std::uint16_t*>(b + align(rows * k * 2));
  void* plan_mem = b + 2 * align(rows * k * 2);
  kg::Plan plan = kg::MakePlan(plan_mem, rows, 1, bm);
  kg::Group(nullptr, plan, stream);
  if (exact) {
    kg::ToF16Split(x, xh, xl, rows * k, g_dense.flag, stream);
    kg::LinearExact(t, w, xh, xl, plan, y, n, k, stream);
  } else {
    kg::ToF16(x, xh, rows * k, g_dense.flag, stream);
    kg::Linear(t, w, xh, plan, y, n, k, stream);
  }
  return true;
}

unsigned* DenseOverflowFlag() noexcept { return g_dense.flag; }

namespace {
int g_attn_override = -1;
}  // namespace
void SetAttnGqa(int mode) noexcept { g_attn_override = mode; }
int AttnGqaMode() noexcept {
  static const int env_mode = [] {
    const char* v = std::getenv("FLERP_ATTN_GQA");
    return (v != nullptr && std::strcmp(v, "0") != 0) ? 1 : 0;
  }();
  return g_attn_override >= 0 ? g_attn_override : env_mode;
}
void SetMoeDump(MoeDump* dump) noexcept { g_dump = dump; }

void LaunchMoeFfn(const gufo::models::QwenLayerWeights& layer, const float* x, float* out, std::size_t tokens,
                  std::size_t hidden, MoeScratch scratch, hipStream_t stream, std::uint32_t layer_index,
                  bool traceable) {
  namespace qfn = gufo::models::qwen38_flash_next::rocm;
  const auto& moe = layer.moe;
  const std::size_t experts = moe.expert_count;
  const std::size_t used = moe.expert_used;
  const std::size_t ff = moe.expert_ff;
  const std::size_t shared_ff = moe.shared_ff;
  if (tokens == 0) {
    return;
  }
  if (!moe.enabled() || scratch.a == nullptr || scratch.b == nullptr || scratch.c == nullptr ||
      scratch.floats_per_token < moe.ScratchFloats(hidden)) {
    std::cerr << "flerp::qwen35moe::LaunchMoeFfn: block " << layer_index << " needs " << moe.ScratchFloats(hidden)
              << " scratch floats a token, the arena has " << scratch.floats_per_token << "\n";
    std::abort();
  }
  const std::size_t stride = experts + 1;
  float* expert_out = scratch.a;
  float* logits = scratch.b;
  auto* ids = reinterpret_cast<std::int32_t*>(scratch.b + (tokens * stride));
  float* weights = scratch.b + (tokens * (stride + used));
  float* shared_out = scratch.b + (tokens * (stride + (2 * used)));
  float* act = scratch.c;
  float* shared_act = scratch.c + (tokens * used * ff);
  const auto n = static_cast<std::uint32_t>(tokens);

  if (!moe.has_shared()) {
    // No shared expert: the epilogue still reads a gate logit and a shared row. Zeros give sigmoid(0) * 0.
    HIP_CHECK(hipMemsetAsync(scratch.b, 0, tokens * (stride + (2 * used) + hidden) * sizeof(float), stream));
  }
  flerp::moe::GateLogits(static_cast<const float*>(layer.ffn_gate_inp.data),
                         moe.has_shared() ? static_cast<const float*>(layer.ffn_gate_inp_shexp.data) : nullptr, x,
                         logits, tokens, experts, hidden, stride, stream);
  qfn::RouterTopK(logits, static_cast<std::uint32_t>(stride), ids, weights, n, static_cast<std::uint32_t>(experts),
                  static_cast<std::uint32_t>(used), stream);
  if (traceable && g_trace != nullptr) {
    MoeTraceLayer rec;
    rec.layer = layer_index;
    rec.tokens = tokens;
    rec.used = static_cast<std::uint32_t>(used);
    rec.ids.resize(tokens * used);
    rec.weights.resize(tokens * used);
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(rec.ids.data(), ids, rec.ids.size() * sizeof(std::int32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(rec.weights.data(), weights, rec.weights.size() * sizeof(float), hipMemcpyDeviceToHost));
    if (moe.has_shared()) {
      std::vector<float> rows(tokens * stride);
      HIP_CHECK(hipMemcpy(rows.data(), logits, rows.size() * sizeof(float), hipMemcpyDeviceToHost));
      rec.shared_gate_logit.resize(tokens);
      for (std::size_t t = 0; t < tokens; ++t) {
        rec.shared_gate_logit[t] = rows[(t * stride) + experts];
      }
    }
    g_trace->layers.push_back(std::move(rec));
  }
  bool grouped = false;
  const int mode = GroupedMode();
  if (traceable && mode != 0 && tokens >= GroupedMin()) {
    bool overflow = false;
    grouped = LaunchGrouped(layer, x, tokens, hidden, scratch, ids, expert_out, shared_out, stream, mode == 2, overflow);
    if (grouped && overflow) {
      // An F16 value would have overflowed: this block's feed forward runs again on the per-token path (exact
      // inputs: x, ids and weights are untouched).
      ++g_fallbacks;
      std::cerr << "flerp::qwen35moe: block " << layer_index << ", " << tokens
                << " rows: F16 overflow in the grouped experts, block re-run on the per-token path (fallbacks "
                << g_fallbacks << ")\n";
      grouped = false;
    }
  }
  if (!grouped) {
  flerp::moe::IndexedExperts(layer.ffn_gate_exps.data, layer.ffn_gate_exps.type, layer.ffn_up_exps.data,
                             layer.ffn_up_exps.type, ids, x, act, tokens, used, used, ff, hidden, experts, stream);
  flerp::moe::IndexedExperts(layer.ffn_down_exps.data, layer.ffn_down_exps.type, nullptr, layer.ffn_down_exps.type,
                             ids, act, expert_out, tokens, used, 1, hidden, ff, experts, stream);
  }
  if (!grouped && moe.has_shared()) {
    flerp::moe::IndexedExperts(layer.ffn_gate_shexp.data, layer.ffn_gate_shexp.type, layer.ffn_up_shexp.data,
                               layer.ffn_up_shexp.type, nullptr, x, shared_act, tokens, 1, 1, shared_ff, hidden, 1,
                               stream);
    flerp::moe::IndexedExperts(layer.ffn_down_shexp.data, layer.ffn_down_shexp.type, nullptr,
                               layer.ffn_down_shexp.type, nullptr, shared_act, shared_out, tokens, 1, 1, hidden,
                               shared_ff, 1, stream);
  }
  qfn::MoeEpilogue(expert_out, weights, shared_out, logits + experts, static_cast<std::uint32_t>(stride), out, n,
                   static_cast<std::uint32_t>(used), static_cast<std::uint32_t>(hidden), stream);
  if (g_dump != nullptr && g_dump->armed &&
      std::find(g_dump->layers.begin(), g_dump->layers.end(), layer_index) != g_dump->layers.end()) {
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<float> row_x(hidden), row_out(hidden), row_w(used), row_logits(stride);
    std::vector<std::int32_t> row_ids(used);
    for (const std::uint32_t t : g_dump->rows) {
      if (t >= tokens) continue;
      HIP_CHECK(hipMemcpy(row_x.data(), x + (t * hidden), hidden * sizeof(float), hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(row_ids.data(), ids + (t * used), used * sizeof(std::int32_t), hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(row_w.data(), weights + (t * used), used * sizeof(float), hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(row_logits.data(), logits + (t * stride), stride * sizeof(float), hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(row_out.data(), out + (t * hidden), hidden * sizeof(float), hipMemcpyDeviceToHost));
      const std::int32_t head[5] = {static_cast<std::int32_t>(layer_index),
                                    static_cast<std::int32_t>(g_dump->position_base + t),
                                    static_cast<std::int32_t>(hidden), static_cast<std::int32_t>(used),
                                    static_cast<std::int32_t>(experts)};
      const float gate = moe.has_shared() ? row_logits[experts] : 0.0F;
      Append(g_dump->bytes, head, 5);
      Append(g_dump->bytes, row_x.data(), hidden);
      Append(g_dump->bytes, row_ids.data(), used);
      Append(g_dump->bytes, row_w.data(), used);
      Append(g_dump->bytes, &gate, 1);
      Append(g_dump->bytes, row_out.data(), hidden);
      ++g_dump->records;
    }
  }
}

}  // namespace flerp::qwen35moe
#endif  // defined(ENGINE_ENABLE_HIP)
