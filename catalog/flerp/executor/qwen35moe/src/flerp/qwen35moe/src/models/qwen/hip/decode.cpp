// flerp (executor.flerp.qwen35moe): gufo's src/models/qwen/hip/decode.cpp (f070a17) with the stage's block range
// and buffers folded into the graph key and the decode graph switch of flerp/qwen35moe/stage.hpp, and (ORNITH-O3 step
// two) the recurrent ring of stage.hpp SetRecurrentRing with one captured graph per slot pair; marked below.
#if defined(ENGINE_ENABLE_HIP)
#include <gufo/ext/experimental/dispatch_telemetry.hpp>
#include <gufo/ext/experimental/hip_utils.hpp>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>

#include "src/families/qwen/hip_common/attention_policy.hpp"
#include "src/models/qwen/hip/detail/decode_step.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "flerp/qwen35moe/stage.hpp"

namespace gufo::hip {

namespace {
// flerp (executor.flerp.qwen35moe): the decode graphs of a recurrent ring (flerp/qwen35moe/stage.hpp
// SetRecurrentRing), one a workload identity (the identity folds the ring's read and write slots).
using RingGraphs = std::map<std::pair<std::uint64_t, std::uint64_t>,
                            std::unique_ptr<detail::HipGraphDecodeExecutor>>;
}  // namespace

tokenization::TokenId QwenGpuExecutor::ForwardToken(
    tokenization::TokenId token_id, std::uint32_t pos, bool compute_logits) {
  CheckReset();
  if (pos >= arena_.GetMaxContext()) {
    throw std::length_error("token position exceeds the GPU context length");
  }
  models::qwen::ContinuationWrite write(continuation_hooks_, pos, pos + 1);

  if (replaying_ssm_state_ && !compute_logits) {
    if (arena_.CanReplaySsmPosition(pos)) {
      ReplaySsmState(pos);
      write.Commit(pos + 1, false);
      return 0;
    }
    replaying_ssm_state_ = false;
    arena_.DisableSsmReplayCapture();
  } else if (replaying_ssm_state_) {
    replaying_ssm_state_ = false;
    arena_.DisableSsmReplayCapture();
  }

  arena_.MarkSsmReplayPosition(pos);
  // One recurrent pass per step. Its buffer pair is part of the graph key below
  // because the captured launches bake the state pointers.
  // flerp (executor.flerp.qwen35moe): on a recurrent ring (stage.hpp
  // SetRecurrentRing) the pass reads slot(pos) and writes slot(pos + 1).
  auto* ring = flerp::qwen35moe::FindRecurrentRing(this);
  const bool on_ring = ring != nullptr && ring->slots >= 2;
  QwenRecurrentStateIo recurrent_state{};
  std::uint64_t recurrent_identity = 0;
  if (on_ring) {
    if (arena_.has_saved_state_ || replaying_ssm_state_) {
      throw std::logic_error(
          "executor.flerp.qwen35moe: a recurrent ring with the rollback pair");
    }
    const auto& cfg = arena_.config_;
    const std::size_t conv_bytes = static_cast<std::size_t>(cfg.SsmLayerCount()) *
                                   cfg.SsmQkvSize() * cfg.ssm_conv_kernel *
                                   sizeof(float);
    const std::size_t delta_bytes =
        static_cast<std::size_t>(cfg.SsmLayerCount()) *
        cfg.ssm_time_step_rank * cfg.ssm_state_size * cfg.SsmValueSize() *
        QwenRecurrentStateElementBytes(arena_.policy_.recurrent_state_storage);
    while (ring->conv.size() + 1 < ring->slots) {
      float* conv = nullptr;
      void* delta = nullptr;
      HIP_CHECK(hipMalloc(&conv, conv_bytes));
      const hipError_t st = hipMalloc(&delta, delta_bytes);
      if (st != hipSuccess) {
        (void)hipFree(conv);
        HIP_CHECK(st);
      }
      HIP_CHECK(hipMemsetAsync(conv, 0, conv_bytes, arena_.stream));
      HIP_CHECK(hipMemsetAsync(delta, 0, delta_bytes, arena_.stream));
      ring->conv.push_back(conv);
      ring->delta.push_back(delta);
      ring->bytes += conv_bytes + delta_bytes;
    }
    if (!ring->started) {
      ring->started = true;
      ring->base = pos;
    }
    if (pos < ring->base) {
      throw std::logic_error(
          "executor.flerp.qwen35moe: a decode step before the ring's first");
    }
    const std::uint32_t read = (pos - ring->base) % ring->slots;
    const std::uint32_t write = (read + 1) % ring->slots;
    const std::uint8_t live = arena_.live_recurrent_;
    const auto conv = [&](std::uint32_t slot) {
      return slot == 0 ? arena_.d_ssm_conv_buffers_[live] : ring->conv[slot - 1];
    };
    const auto delta = [&](std::uint32_t slot) {
      return slot == 0 ? arena_.d_ssm_deltanet_buffers_[live]
                       : ring->delta[slot - 1];
    };
    recurrent_state = {.conv_in = conv(read),
                       .conv_out = conv(write),
                       .recurrent_in = delta(read),
                       .recurrent_out = delta(write)};
    recurrent_identity = (std::uint64_t{1} << 40) |
                         (static_cast<std::uint64_t>(read) << 20) | write;
    ++ring->steps;
  } else {
    recurrent_state = arena_.BeginRecurrentWrite();
    recurrent_identity = arena_.RecurrentIoIdentity();
  }
  last_hidden_offset_ = 0;
  const auto& config = weights_.config;
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  const bool use_split_k_decode = detail::IsSplitKDecodeAttentionSupported(
      sequence_length, config.num_attention_heads, config.num_key_value_heads,
      config.head_dim);

  auto scratch = arena_.GetScratchView();
  auto* d_out_token = scratch.decode.sampled_token.data();

  // Copy token_id and pos to GPU device memory
  const std::uint32_t in_params[2] = {token_id, pos};
  HIP_CHECK(hipMemcpyAsync(scratch.decode.prompt_tokens.data(), in_params,
                           sizeof(in_params), hipMemcpyHostToDevice,
                           arena_.stream));

  EmitDecodeRouteTelemetry(weights_, policy_);
  auto graph_key = graph_key_;
  const auto target_layers = arena_.GetTargetLayerCapture();
  graph_key.workload_identity = ExtendQwenGraphWorkloadIdentity(
      graph_key.workload_identity, target_layers.size());
  for (const auto layer : target_layers) {
    graph_key.workload_identity =
        ExtendQwenGraphWorkloadIdentity(graph_key.workload_identity, layer);
  }
  graph_key.workload_identity = ExtendQwenGraphWorkloadIdentity(
      graph_key.workload_identity, recurrent_identity);
  // flerp (executor.flerp.qwen35moe): a stage's graph is the graph of its own
  // blocks reading and writing its own buffers (docs/PHASE5-FOURNODE-DESIGN.md
  // section 3.5): fold the block range and the stage buffers, which the
  // captured copies bake, so a graph is never replayed for another range.
  {
    const auto stage_io = flerp::qwen35moe::GetStageIo(&weights_);
    const std::uint64_t stage_fields[] = {
        weights_.BlockBegin(), weights_.BlockEnd(),
        reinterpret_cast<std::uintptr_t>(stage_io.d_input),
        reinterpret_cast<std::uintptr_t>(stage_io.d_output)};
    for (const std::uint64_t field : stage_fields) {
      graph_key.workload_identity =
          ExtendQwenGraphWorkloadIdentity(graph_key.workload_identity, field);
    }
  }
  // flerp (executor.flerp.qwen35moe): a ring step uses the ring's graph of
  // its slot pair; every other step the origin's one graph.
  detail::HipGraphDecodeExecutor* graph_executor = &graph_executor_;
  if (on_ring) {
    if (!ring->graphs) {
      ring->graphs = std::shared_ptr<void>(new RingGraphs(), [](void* p) {
        delete static_cast<RingGraphs*>(p);
      });
    }
    auto& graphs = *static_cast<RingGraphs*>(ring->graphs.get());
    auto& slot = graphs[{graph_key.execution_identity,
                         graph_key.workload_identity}];
    if (!slot) slot = std::make_unique<detail::HipGraphDecodeExecutor>();
    graph_executor = slot.get();
  }
  auto& graph_executor_ref = *graph_executor;
  const QwenGraphRejection graph_rejections = ResolveQwenGraphRejections(
      compute_logits, use_split_k_decode,
      graph_executor_ref.IsEnabled() &&
          // flerp (executor.flerp.qwen35moe): the process-wide decode graph
          // switch (flerp/qwen35moe/stage.hpp); off runs the step eagerly.
          flerp::qwen35moe::DecodeGraphEnabled() &&
          pos >= vision_input_.layout().PrefixLength());
  detail::EmitQwenGraphEligibility(
      graph_key.execution_identity, graph_key.workload_identity,
      static_cast<std::uint32_t>(graph_rejections));

  const auto launch_captured_graph = [&]() {
    if (!graph_executor_ref.Launch(arena_.stream, graph_key)) {
      graph_executor_ref.Reset();
      throw std::runtime_error(
          "Qwen HIP graph launch failed; captured graph was invalidated");
    }
  };

  if (graph_rejections == QwenGraphRejection::kNone) {
    if (graph_executor_ref.IsCapturedFor(graph_key)) {
      launch_captured_graph();
    } else {
      // Target feature taps change the captured copy operations.
      if (graph_executor_ref.IsCaptured())
        graph_executor_ref.Reset();
      const bool ok =
          graph_executor_ref.TryCapture(arena_.stream, graph_key, [&]() {
            ExecuteDecodeStep(arena_, weights_, policy_, token_id, pos,
                              compute_logits, recurrent_state, &vision_input_);
          });
      if (ok) {
        if (on_ring) ++ring->captures;
        launch_captured_graph();
      } else {
        ExecuteDecodeStep(arena_, weights_, policy_, token_id, pos,
                          compute_logits, recurrent_state, &vision_input_);
      }
    }
  } else {
    ExecuteDecodeStep(arena_, weights_, policy_, token_id, pos, compute_logits,
                      recurrent_state, &vision_input_);
  }

  if (!compute_logits) {
    write.Commit(pos + 1, false);
    return 0;
  }

  std::uint32_t next_token_id = 0;
  HIP_CHECK(hipMemcpyAsync(&next_token_id, d_out_token, sizeof(std::uint32_t),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));

  const auto result = CheckedSampleToken(next_token_id, config.vocab_size);
  write.Commit(pos + 1, true);
  return result;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
