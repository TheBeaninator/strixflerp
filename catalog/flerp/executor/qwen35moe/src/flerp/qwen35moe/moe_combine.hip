// MOE-COMBINE (card 2dc55e67f0, gate/moe-combine.toml): the MoE combine of a prompt chunk, four adjacent elements a
// thread. gufo's MoeEpilogueKernel (catalog/gufo/kernel/qfn_kernels, kernels.hip.cpp) gives each thread one float and
// launches tokens x dim/256 blocks of narrow 4-byte loads: 48-51 GB/s on K (gate/results/moe-combine), 7 ms a 35B block.
// This kernel keeps its per-element expression - the slots summed in order from a pinned zero, each step pinned
// (Materialize(acc + w * v)), then acc + SigmoidF(g) * sh - and is built in flerp_qfn_kernels with the same options as
// kernels.hip.cpp (-O3 -ffast-math -fno-finite-math-only), so every output word is the same; only the loads widen to
// 16 bytes (189-203 GB/s on K, the rate of a plain read of the same bytes).
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace flerp::qwen35moe {
namespace {

constexpr unsigned kCombineThreads = 256;
constexpr std::uint32_t kCombineMaxSlots = 16;

// SigmoidF and Materialize as kernels.hip.cpp defines them.
__device__ __forceinline__ float CombineSigmoidF(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float CombineMaterialize(float v) {
  asm volatile("" : "+v"(v));
  return v;
}

__global__ void MoeCombineVec4Kernel(const float* expert_out, const float* weights, const float* shared_out,
                                     const float* gate, std::uint32_t gate_stride, float* out, std::uint32_t k,
                                     std::uint32_t dim) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
  if (i >= dim) return;
  const float* rows = expert_out + static_cast<std::size_t>(t) * k * dim + i;
  float w[kCombineMaxSlots];
  float4 v[kCombineMaxSlots];
#pragma unroll
  for (std::uint32_t s = 0; s < kCombineMaxSlots; ++s) {
    w[s] = s < k ? weights[t * k + s] : 0.0f;
    v[s] = s < k ? *reinterpret_cast<const float4*>(rows + static_cast<std::size_t>(s) * dim)
                 : float4{0.0f, 0.0f, 0.0f, 0.0f};
  }
  const std::size_t idx = static_cast<std::size_t>(t) * dim + i;
  const float4 sh = *reinterpret_cast<const float4*>(shared_out + idx);
  const float g = gate[static_cast<std::size_t>(t) * gate_stride];
  float a0 = CombineMaterialize(0.0f), a1 = CombineMaterialize(0.0f), a2 = CombineMaterialize(0.0f),
        a3 = CombineMaterialize(0.0f);
#pragma unroll
  for (std::uint32_t s = 0; s < kCombineMaxSlots; ++s) {
    if (s < k) {
      a0 = CombineMaterialize(a0 + w[s] * v[s].x);
      a1 = CombineMaterialize(a1 + w[s] * v[s].y);
      a2 = CombineMaterialize(a2 + w[s] * v[s].z);
      a3 = CombineMaterialize(a3 + w[s] * v[s].w);
    }
  }
  float4 o;
  o.x = a0 + CombineSigmoidF(g) * sh.x;
  o.y = a1 + CombineSigmoidF(g) * sh.y;
  o.z = a2 + CombineSigmoidF(g) * sh.z;
  o.w = a3 + CombineSigmoidF(g) * sh.w;
  *reinterpret_cast<float4*>(out + idx) = o;
}

}  // namespace

// false (nothing launched) when the shape is outside the kernel: dim not a multiple of 4, more than 16 slots, or
// unaligned rows; the caller then runs gufo's MoeEpilogue.
bool MoeCombineVec4(const float* expert_out, const float* weights, const float* shared_out, const float* gate,
                    std::uint32_t gate_stride, float* out, std::uint32_t n_tokens, std::uint32_t k, std::uint32_t dim,
                    hipStream_t stream) {
  const auto a16 = [](const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15U) == 0; };
  if (n_tokens == 0 || dim % 4 != 0 || k == 0 || k > kCombineMaxSlots || !a16(expert_out) || !a16(shared_out) ||
      !a16(out))
    return false;
  const unsigned blocks = (dim / 4 + kCombineThreads - 1) / kCombineThreads;
  MoeCombineVec4Kernel<<<dim3(n_tokens, blocks), dim3(kCombineThreads), 0, stream>>>(expert_out, weights, shared_out,
                                                                                    gate, gate_stride, out, k, dim);
  return hipGetLastError() == hipSuccess;
}

}  // namespace flerp::qwen35moe
