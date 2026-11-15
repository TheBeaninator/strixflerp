// SPDX-License-Identifier: MIT
// strixite's logits_topk (top 20 candidates per logits row, LogitCand[20] + a NaN word per row) on gufo's
// LaunchGPUSortLogits(logits, workspace, stream, keep = 20) - the adapter docs/INTERFACES.md calls `cands-layout`.
//
// What gufo's launcher gives: one row at a time, the exact sorted prefix of the row (value descending, lower id
// first on a tie, -0 equal to +0) as two arrays in a GpuSamplingWorkspace: sorted_logits and sorted_token_ids.
// What the adapter adds, with two small kernels of its own:
//   pack      (sorted_logits[i], sorted_token_ids[i]) for i < 20 into LogitCand; slots past the row's logits get
//             strixite's filler (-inf, INT32_MAX);
//   nan_scan  strixite flags a row that holds a NaN among its first n_valid logits; gufo has no such output (it
//             turns every non-finite logit into -inf before sorting), so the adapter reads the row once itself.
// Memory: gufo's own AllocateGpuSamplingWorkspace calls hipMalloc; strixite's kernels never allocate and take a
// caller-owned workspace. The adapter keeps strixite's rule: logits_topk_workspace_bytes is taken over too and
// returns strixite's own size plus room for gufo's four row-sized arrays, and the GpuSamplingWorkspace handed to
// gufo points into that caller buffer. vocab_size = n_valid, which is also how the LM head's padding stays out.
//
// Where the contracts differ, and what happens there:
//   allow-mask (structured output)  gufo has none: the call goes to strixite's kernel (exact, counted as an
//                                    origin launch).
//   n_valid <= 1024                 gufo sorts such a row with hipcub's DeviceRadixSort, which needs temp
//                                    storage from its own allocator: the call goes to strixite's kernel.
//   +inf logits                     strixite ranks +inf first; gufo maps it to -inf. NOT reproduced.
//   NaN rows                        the flag is exact. The candidates differ only when fewer than 20 logits of
//                                    the row are above -inf (gufo lists the NaN's id as a -inf entry, strixite
//                                    leaves it out); the sampler refuses a flagged row either way.
//   cost                            5 kernel launches per row + 1 per call, against strixite's 2 per call.
// The module gate (gate/interface/logits-topk.toml) measures every line of this list.
#include "kernels/logits_topk.hpp"

#include "common/hip_check.hpp"
#include "flerp/adapter_rt.hpp"
#include "src/families/qwen/kernels/ops/token.hpp"

#include <climits>
#include <cmath>

namespace strix::kernels {

size_t logits_topk_workspace_bytes_flerp_origin(int64_t rows, int64_t n_valid);
void logits_topk_flerp_origin(const float *logits, int64_t rows, int64_t ld, int64_t n_valid, LogitCand *out,
                              uint32_t *nan, void *workspace, size_t workspace_bytes, hipStream_t stream,
                              const uint32_t *mask, int64_t mask_words);

namespace {

constexpr int kScanThreads = 256;
constexpr int64_t kScanPerBlock = 64 * kScanThreads;      // logits one workgroup reads for NaNs
constexpr int64_t kGufoTile = 1024;                       // gufo's kCandidateTile: at or below it, it radix-sorts

struct Layout {
    size_t origin_bytes;      // strixite's own workspace, kept first so a delegated call finds it as before
    size_t gufo_off;          // adjusted_logits, sorted_logits, token_ids, sorted_token_ids: n_valid x 4 bytes each
    size_t flags_off;         // [rows][blocks] uint32: a block's logits hold a NaN
    size_t total;
    int64_t blocks;
};

Layout layout(int64_t rows, int64_t n_valid) {
    Layout l{};
    l.origin_bytes = (logits_topk_workspace_bytes_flerp_origin(rows, n_valid) + 15) / 16 * 16;
    l.gufo_off = l.origin_bytes;
    l.blocks = (n_valid + kScanPerBlock - 1) / kScanPerBlock;
    l.flags_off = l.gufo_off + (size_t)n_valid * 16;
    l.total = l.flags_off + (size_t)(rows * l.blocks) * sizeof(uint32_t);
    return l;
}

__global__ void __launch_bounds__(kScanThreads)
    nan_scan(const float *logits, int64_t ld, int n_valid, uint32_t *flags, int blocks) {
    __shared__ uint32_t any;
    const float *row = logits + (int64_t)blockIdx.y * ld;
    const int64_t base = (int64_t)blockIdx.x * kScanPerBlock;
    const int64_t end = base + kScanPerBlock < n_valid ? base + kScanPerBlock : n_valid;
    bool bad = false;
    for (int64_t i = base + threadIdx.x; i < end; i += kScanThreads)
        bad |= (__float_as_uint(row[i]) & 0x7fffffffu) > 0x7f800000u;      // NaN by its bits, whatever the math flags
    if (threadIdx.x == 0) any = 0;
    __syncthreads();
    if (bad) any = 1;      // every writer stores the same value
    __syncthreads();
    if (threadIdx.x == 0) flags[(int64_t)blockIdx.y * blocks + blockIdx.x] = any;
}

__global__ void __launch_bounds__(32) pack(const float *sorted_logits, const uint32_t *sorted_ids, int n_valid,
                                           const uint32_t *flags, int blocks, LogitCand *out, uint32_t *nan) {
    const int j = threadIdx.x;
    if (j < kLogitCands) {
        const uint32_t id = j < n_valid ? sorted_ids[j] : 0xffffffffu;
        LogitCand c;
        if (id < (uint32_t)n_valid) {
            c.v = sorted_logits[j];
            c.id = (int32_t)id;
        } else {
            c.v = -INFINITY;
            c.id = INT32_MAX;
        }
        out[j] = c;
    }
    if (j == 0) {
        uint32_t f = 0;
        for (int b = 0; b < blocks; ++b) f |= flags[b];
        *nan = f;
    }
}

}  // namespace

size_t logits_topk_workspace_bytes(int64_t rows, int64_t n_valid) { return layout(rows, n_valid).total; }

void logits_topk(const float *logits, int64_t rows, int64_t ld, int64_t n_valid, LogitCand *out, uint32_t *nan,
                 void *workspace, size_t workspace_bytes, hipStream_t stream, const uint32_t *mask, int64_t mask_words) {
    static flerp::Tally tally("kernel.flerp.logits_topk_gufo", "kernel.gufo.qwen_sampling:LaunchGPUSortLogits");
    const bool args_ok = logits && out && nan && workspace && rows >= 1 && rows <= 65535 && n_valid >= 1 &&
                         n_valid <= kLogitsTopkMaxValid && ld >= n_valid &&
                         reinterpret_cast<uintptr_t>(workspace) % 16 == 0;
    if (!args_ok || mask != nullptr || n_valid <= kGufoTile || workspace_bytes < layout(rows, n_valid).total) {
        tally.origin();      // strixite's kernel: exact for masks and short rows, and it reports argument errors itself
        logits_topk_flerp_origin(logits, rows, ld, n_valid, out, nan, workspace,
                                 workspace_bytes < layout(rows, n_valid).origin_bytes ? workspace_bytes
                                                                                      : layout(rows, n_valid).origin_bytes,
                                 stream, mask, mask_words);
        return;
    }
    tally.foreign();
    const Layout l = layout(rows, n_valid);
    auto *base = static_cast<unsigned char *>(workspace);
    auto *flags = reinterpret_cast<uint32_t *>(base + l.flags_off);
    nan_scan<<<dim3((unsigned)l.blocks, (unsigned)rows), kScanThreads, 0, stream>>>(logits, ld, (int)n_valid, flags,
                                                                                    (int)l.blocks);
    STRIX_HIP_CHECK_LAUNCH("logits_topk (gufo adapter) nan_scan rows=", rows, " n_valid=", n_valid);
    gufo::hip::GpuSamplingWorkspace ws{};
    ws.adjusted_logits = reinterpret_cast<float *>(base + l.gufo_off);
    ws.sorted_logits = ws.adjusted_logits + n_valid;
    ws.token_ids = reinterpret_cast<std::uint32_t *>(ws.sorted_logits + n_valid);
    ws.sorted_token_ids = ws.token_ids + n_valid;
    ws.vocab_size = (std::size_t)n_valid;
    for (int64_t r = 0; r < rows; ++r) {      // one row at a time: gufo's workspace holds one row
        gufo::hip::LaunchGPUSortLogits(logits + r * ld, &ws, stream, (std::size_t)kLogitCands);
        pack<<<1, 32, 0, stream>>>(ws.sorted_logits, ws.sorted_token_ids, (int)n_valid, flags + r * l.blocks,
                                   (int)l.blocks, out + r * kLogitCands, nan + r);
        STRIX_HIP_CHECK_LAUNCH("logits_topk (gufo adapter) row ", r);
    }
}

}  // namespace strix::kernels
