// SPDX-License-Identifier: MIT
// See stage.hpp.
#include "flerp/qwen35moe/stage.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

#include <atomic>
#include <map>
#include <mutex>

namespace flerp::qwen35moe {
namespace {
LoadOptions g_options;
StageIo g_io;
std::atomic<bool> g_decode_graph{true};
std::mutex g_bound_mu;
std::map<const void*, StageIo>& Bound() {
  static std::map<const void*, StageIo> bound;
  return bound;
}
flerp::gguf_placed::LoadReport& Report() {
  static flerp::gguf_placed::LoadReport report;
  return report;
}
}  // namespace

void SetLoadOptions(const LoadOptions& options) noexcept { g_options = options; }
const LoadOptions& GetLoadOptions() noexcept { return g_options; }
const flerp::gguf_placed::LoadReport& LastLoadReport() noexcept { return Report(); }
void SetLastLoadReport(const flerp::gguf_placed::LoadReport& report) { Report() = report; }
void SetStageIo(const StageIo& io) noexcept { g_io = io; }
const StageIo& GetStageIo() noexcept { return g_io; }
void BindStageIo(const void* weights, const StageIo& io) {
  std::lock_guard<std::mutex> lock(g_bound_mu);
  Bound()[weights] = io;
}
void UnbindStageIo(const void* weights) noexcept {
  std::lock_guard<std::mutex> lock(g_bound_mu);
  Bound().erase(weights);
}
StageIo GetStageIo(const void* weights) noexcept {
  std::lock_guard<std::mutex> lock(g_bound_mu);
  const auto it = Bound().find(weights);
  return it == Bound().end() ? g_io : it->second;
}
void SetDecodeGraph(bool enabled) noexcept { g_decode_graph = enabled; }
bool DecodeGraphEnabled() noexcept { return g_decode_graph; }

namespace {
std::map<const void*, RecurrentRing>& Rings() {
  static std::map<const void*, RecurrentRing> rings;
  return rings;
}
}  // namespace

void SetRecurrentRing(const void* executor, std::uint32_t slots) {
  auto& ring = Rings()[executor];
  ring.slots = slots < 2 ? 0 : slots;
  ring.started = false;
}

void ResetRecurrentRing(const void* executor) noexcept {
  const auto it = Rings().find(executor);
  if (it != Rings().end()) it->second.started = false;
}

void ReleaseRecurrentRing(const void* executor) noexcept {
  const auto it = Rings().find(executor);
  if (it == Rings().end()) return;
  it->second.graphs.reset();   // the graphs first: they name the buffers
#if defined(ENGINE_ENABLE_HIP)
  for (float* p : it->second.conv) (void)hipFree(p);
  for (void* p : it->second.delta) (void)hipFree(p);
#endif
  Rings().erase(it);
}

RecurrentRing* FindRecurrentRing(const void* executor) noexcept {
  const auto it = Rings().find(executor);
  return it == Rings().end() ? nullptr : &it->second;
}

}  // namespace flerp::qwen35moe
