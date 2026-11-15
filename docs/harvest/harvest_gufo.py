#!/usr/bin/env python3
"""Harvest gufo (neuhaus RDMA fork, f070a17) into catalog/gufo. Run: python3 docs/harvest/harvest_gufo.py
Kernel modules take their translation units plus the headers only they include; headers shared across kernel modules
live in header-only modules, found with docs/harvest/incgraph.py."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harvestlib import Origin, M, P, run
from incgraph import closure

O = Origin('gufo', 'gufo-neuhaus', 'https://github.com/neuhaus/gufo', 'f070a17', 'rdma', 'MIT',
           ['gufo contributors', 'neuhaus'], 'therock-10.1')
ROOT = O.roots[0]
QK = 'src/families/qwen/kernels/'
QFN = 'src/models/qwen38_flash_next/'
DS4 = 'src/models/deepseek_v4_flash/'
DSR = DS4 + 'kernels/rocm/'

claimed = set()


def take(patterns, exclude=()):
    """Origin files matching the patterns that no earlier module took."""
    out = [f for f in O.expand(patterns, exclude) if f not in claimed]
    claimed.update(out)
    return out


def tu_closure(tus):
    c = closure(ROOT, ['.', 'include', 'src', DSR.rstrip('/'), DSR + 'detail', DS4 + 'runtime', DSR + 'mmq', QFN + 'kernels/rocm/mmq'], tus)
    return sorted({f for fs in c.values() for f in fs})


HIPFLAGS = ['-x', 'hip', '-std=gnu++20', '-O2', '-UNDEBUG', '-DENGINE_ENABLE_HIP=1', '-c']
FAST = ['-x', 'hip', '-std=gnu++20', '-O3', '-ffast-math', '-fno-finite-math-only', '-Wno-unused-command-line-argument', '-UNDEBUG', '-c']
MMQ = ['-x', 'hip', '-std=gnu++17', '-O3', '-ffast-math', '-fno-finite-math-only', '-Wno-unused-value', '-Wno-unused-command-line-argument', '-fPIC', '-UNDEBUG', '-c']
W64 = ['-mwavefrontsize64']
DEC = 'decode and MTP/DFlash verification, 1..16 rows, memory-bound'
PF = 'prefill batches (64..1024 tokens), compute-bound'
NOTBUILT = 'c++ host code: built only inside a recipe (phase 3 and later)'
mods = []

# ------------------------------------------------------------------ header-only modules first (they own shared headers)
FMT_PATTERNS = ['src/core/gguf_reader.cpp', 'src/core/gguf_identity.hpp', 'src/core/gguf_identity.cpp', 'src/core/quant/']
qwen_tus = sorted(O.expand([QK + '*.hip'])) + ['src/models/qwen/hip/batched_ssm.hip']
all_kernel_tus = qwen_tus + [QFN + 'kernels/rocm/kernels.hip.cpp', QFN + 'kernels/rocm/w8a8_wave64.hip', DSR + 'backend.hip.cpp', 'src/core/hip/snapshot_copy.hip'] + \
    O.expand([QFN + 'kernels/rocm/mmq/*.hip.cpp', DSR + 'mmq/*.hip.cpp'])
allc = tu_closure(all_kernel_tus)
core_hdr = take([f for f in allc if (f.startswith('src/core/') or f.startswith('include/')) and not f.endswith(('.hip', '.cpp'))] + ['src/core/hip/libstdcxx_noinline.hpp'])
fmt_files = take(FMT_PATTERNS)   # after core_headers: gguf_reader.hpp and quant/ggml_dequant.hpp are headers kernels include
qwen_hdr = take([f for f in tu_closure(qwen_tus) if f.startswith('src/families/qwen/') and f.endswith('.hpp')])

mods.append(M('format.gufo.gguf_kquant', 'GGUF reader and ggml quant layouts: K-quants (Q2_K..Q8_K), Q8_0, IQ2 grids; host dequantizers and a host quantized GEMM used as oracles',
              fmt_files, lang='c++', compile='n/a', upstream='block layouts and IQ grids after llama.cpp / ggml (MIT, The ggml authors)',
              provides=[P('format.loader/gguf/-', 'gufo::core::GgufReader, GgmlType', 'exact (mmap of the GGUF file; weights stay in their GGUF encoding on the device)', 'load time'),
                        P('format.device-view/gguf-kquant/-', 'block_q8_0 (QK 32), block_q4_K / q5_K / q6_K / q8_K (QK 256), IQ2_XXS/XS/S grids: src/core/quant/ggml_dequant.hpp', 'raw GGUF blocks (layouts declared in src/core/quant/ggml_dequant.hpp, which lives in kernel.gufo.core_headers because kernels include it); ggml_dequant.cpp is the CPU oracle the kernels are compared with', 'n/a')],
              requires=['module:kernel.gufo.core_headers'], reason=NOTBUILT))
mods.append(M('kernel.gufo.core_headers', 'Header-only: the gufo core and public extension headers that kernel translation units include (HIP utilities, dispatch telemetry, model config, sampling state, JSON constraint types)',
              core_hdr, lang='hip-c++', includes=['src', 'src/include'], compile='header-only',
              provides=[P('kernel-abi.gufo-core/-/-', 'src/core/hip/hip_utils.hpp, src/core/hip/detail/*.hpp, include/gufo/ext/*.hpp', 'n/a', 'n/a')],
              reason='header-only; compiled inside every kernel.gufo.* module'))
mods.append(M('kernel.gufo.qwen_ops', 'Header-only: launch APIs of the shared Qwen HIP operators (kernels/ops/*.hpp), quant block helpers, route and attention policies, and kernel-private headers shared between translation units',
              qwen_hdr, lang='hip-c++', includes=['src'], compile='header-only',
              provides=[P('kernel-abi.gufo-qwen-ops/-/-', 'gufo::hip::Launch* declarations (ops/gemm.hpp, attention.hpp, ssm.hpp, token.hpp, norm_residual.hpp, swiglu.hpp, prefill_fp16.hpp)', 'n/a', 'n/a')],
              requires=['module:kernel.gufo.core_headers'],
              reason='header-only; compiled inside the kernel.gufo.qwen_* modules'))

QREQ = ['module:kernel.gufo.qwen_ops', 'module:kernel.gufo.core_headers']


def qk(name, files, summary, provides, extra=(), sflags=None, notes=None):
    srcs = [QK + f for f in files] + list(extra)
    mods.append(M(f'kernel.gufo.{name}', summary, take(srcs), provides=provides, sources=srcs, includes=['src'], flags=HIPFLAGS,
                  source_flags=sflags, requires=QREQ + ['format.device-view/gguf-kquant/-'], notes=notes, variants=len(srcs),
                  parity='origin-test:tests/kernels'))


qk('qwen_linear_decode', ['gemv.hip', 'gemv_quant.hip', 'fused.hip', 'qkv.hip', 'swiglu.hip', 'small_batch_wave64.hip', 'small_batch_quant16_wave64.hip', 'verify_gemm_27b.hip'],
   'Decode and verification projections on GGUF weights: F32/BF16 GEMV, K-quant GEMV decoded in-kernel, fused QKV and fused gate/up SwiGLU, exact wave64 small-batch kernels for draft verification',
   [P('linear.decode/gguf-kquant/f32.gufo', 'gufo::hip::LaunchGEMV, LaunchQ8KBlockGEMV, LaunchFusedQKVProjections, LaunchFusedSwiGLUGEMV (ops/gemm.hpp, ops/attention.hpp, ops/swiglu.hpp)', 'fp32 activations, weights decoded in-kernel from packed GGUF blocks, fp32 accumulate; Q8_K path quantizes the activation to int8 and uses an integer MAC with one fp scale at block end', DEC),
    P('linear.decode/bf16/f32.gufo', 'gufo::hip::LaunchGEMV (is_bf16), LaunchExactBf16GEMMFp32SmallBatch', 'fp32 x, bf16 W, fp32 accumulate, fp32 out', DEC),
    P('linear.verify/gguf-kquant/f32.exact-rows', 'gufo::hip::LaunchVerifyGemm27, LaunchKQuantSixteenWave64, LaunchPackedSwiGLUSixteenWave64', 'every lane walks its sub-blocks in the same order and lanes reduce in the same butterfly for every block shape: rows of a verification batch are bit-equal to the one-row decode kernel', '9..16 verification rows, wave64')],
   sflags={QK + 'small_batch_wave64.hip': W64,
           QK + 'small_batch_quant16_wave64.hip': W64 + ['-gline-tables-only', '-Xarch_device', '-mllvm=-amdgpu-sched-strategy=iterative-ilp']})
qk('qwen_linear_prefill', ['prefill_gemm.hip', 'prefill_quant_gemm.hip', 'prefill_quant_wave64.hip', 'prefill_fp16.hip', 'hipblas_gemm.hip', 'hipblaslt_gemm.hip', 'prefill_swiglu.hip'],
   'Prefill projections: blocked WMMA GEMM directly on K-quant blocks, the Q8_1 activation-quantized route, the FP16 route, hipBLAS / hipBLASLt BF16 GEMM with cached plans, batched SwiGLU',
   [P('linear.prefill/gguf-kquant/bf16-f32', 'gufo::hip::LaunchBatchedQuantGEMM, LaunchBatchedQuantGEMMBf16, LaunchBatchedQuantGEMMFp32, LaunchBatchedQuantGEMMPreQuantized', 'no dequantize-to-BF16: each output row reads the quant blocks directly; activations bf16 (or pre-quantized Q8_1 tiles, int8 MAC), fp32 accumulate', PF),
    P('linear.prefill/gguf-kquant/f16-f32', 'ops/prefill_fp16.hpp launchers', 'packed weights scaled in fp32, rounded to fp16 in-kernel, fp16 activations, fp32 accumulation in K16 order', PF),
    P('linear.prefill/bf16/bf16-f32.hipblaslt', 'gufo::hip::HipblasltGemmPlans::RunBf16, LaunchHipblasGEMM, LaunchHipblasGEMMBF16', 'library GEMM; summation order is the library plan', PF)],
   sflags={QK + 'prefill_quant_gemm.hip': ['-fno-slp-vectorize'], QK + 'prefill_quant_wave64.hip': W64 + ['-fno-slp-vectorize']})
qk('qwen_attention', ['attention_decode.hip', 'attention_decode_graph.hip', 'attention_decode_verify_wave64.hip', 'attention_batched.hip', 'attention_tile.hip', 'attention_wmma.hip', 'unpack.hip', 'prefill_unpack.hip'],
   'Grouped-query softmax attention with gating over a canonical FP16 KV cache: decode with split-K partials, verification rows, batched causal tiles, masked prefill on WMMA',
   [P('attention.full/kv-f16/f32.gufo', 'gufo::hip::LaunchAttention, LaunchCausalDecodeAttention, LaunchBatchedAttention, LaunchBatchedAttentionGemm, LaunchBatchedAttentionTile, LaunchQwenWmmaAttention, LaunchUnpackQG', 'K/V canonical FP16 (FP32 in the validation fallback), fp32 scores and softmax, sigmoid gate; split halves exchange partial log-sum-exp; WMMA path validated against a CPU double-precision reference', DEC + '; ' + PF)],
   sflags={QK + 'attention_decode_verify_wave64.hip': W64})
qk('qwen_gdn', ['ssm.hip', 'ssm_recurrence.hip', 'ssm_decode_recurrence.hip', 'ssm_row_split.hip', 'prefill_ssm.hip'],
   'Gated DeltaNet (called SSM in gufo): fused input projections, causal conv1d, token-serial DeltaNet recurrence, row-split recurrence with fused post-norm and gate, rollback capture',
   [P('gdn.recurrence/-/f32.gufo', 'gufo::hip::LaunchSSMConvRecurrence, LaunchSSMConvRecurrenceRows, LaunchSSMConvRecurrenceBatch, LaunchBatchedSSMConvRecurrence, LaunchBatchedSSMConvRecurrenceRowSplit, LaunchCaptureBatchedSsmReplay (ops/ssm.hpp)', 'fp32 state; each row arithmetic and order identical to the one-row-per-launch form (bit-exact across batch shapes); in-place and pinned-buffer passes bit-identical', DEC + '; ' + PF),
    P('gdn.front/gguf-kquant/f32', 'gufo::hip::LaunchFusedSSMInputProjections, LaunchBatchedFusedSSMInputProjections', 'fused q,k,v,gate,alpha,beta projections on GGUF weights', DEC)],
   extra=['src/models/qwen/hip/batched_ssm.hip'])
qk('qwen_norm_rope', ['norm.hip', 'prefill_norm.hip', 'rope.hip', 'prefill_rope.hip', 'residual.hip', 'prefill_residual.hip', 'embed.hip', 'prefill_embed.hip'],
   'RMSNorm (whole row and per head), RoPE, fused Q/K norm + RoPE + KV-cache write, residual add, embedding lookup; single-token and batched forms',
   [P('norm.rms/-/f32.gufo', 'gufo::hip::LaunchRMSNorm, LaunchPerHeadRMSNorm, LaunchBatchedRMSNorm', 'out = x / sqrt(mean(x^2) + eps) * weight in fp32; optional BF16 output in the batched form', DEC),
    P('rope.apply/-/f32.gufo', 'gufo::hip::LaunchRoPE, LaunchBatchedRoPE', 'fp32 rotation', DEC),
    P('attention.qk-norm-rope-store/kv-f16/f32', 'gufo::hip::LaunchFusedQKNormRoPEKvWrite, LaunchBatchedFusedQKNormRoPEKvWrite', 'per-head RMSNorm, RoPE and the K/V write into every selected cache representation in one launch', DEC),
    P('residual.add/-/f32.out-of-place', 'gufo::hip::LaunchResidualAdd, LaunchBatchedResidualAdd', 'fp32 add', DEC),
    P('embed.lookup/gguf-kquant/f32', 'gufo::hip::LaunchEmbeddingLookup, LaunchBatchedEmbeddingLookup', 'row dequantized to fp32', DEC)])
qk('qwen_sampling', ['sample.hip'],
   'Device-side sampling: bounded tile-winner top-k (radix order, lowest id on ties), penalties, argmax, and lossless speculative accept / residual sampling for verification rows',
   [P('sample.topk/-/f32.gufo', 'gufo::hip::LaunchGPUSortLogits, LaunchGPUSampling, LaunchGPUArgmax, LaunchBatchedGPUArgmax (ops/token.hpp)', 'ordered float bits + original id: radix order with lowest-id ties and signed-zero equivalence; non-finite values sort last', '1 row x vocab'),
    P('speculation.accept/-/f32.gufo', 'gufo::hip::LaunchGPUSpeculativeSampling', 'accepts the draft token with the lossless speculative ratio, else samples the residual target-minus-draft distribution', 'verification rows')])
qk('qwen_dflash', ['dflash_kernels.hip'],
   'DFlash-2 draft kernels: a draft-private Q8_0 copy of the LM head and the draft graph operators (drafts only steer verification, the verifier still runs the BF16 head)',
   [P('draft.dflash/gguf-q8_0/f32', 'gufo::hip::LaunchDFlashQuantizeBf16ToQ8_0, LaunchDFlashNonCausalAttention*, LaunchDFlashGroupedDynamicConv, LaunchDFlashSelectorStep / Batch, LaunchDFlashSiLUMul (dflash_kernels.hpp)', 'draft-only numerics: cannot change an emitted token', 'draft block, weight-bandwidth bound')])

# qwen38 flash next
qfn_tus = [QFN + 'kernels/rocm/kernels.hip.cpp', QFN + 'kernels/rocm/w8a8_wave64.hip']
qfn_files = take([f for f in tu_closure(qfn_tus) if f.startswith(QFN) and '/mmq/' not in f])
mods.append(M('kernel.gufo.qfn_kernels', 'Qwen3.8-Flash-Next model-private kernels in one translation unit (83 kernels): embedding, grouped RMSNorm, hyper-connection mix / combine, PLE, Gated DeltaNet, QSA pooling / select, attention, router top-k, MoE epilogue, MTP head, argmax, W8A8 int8 WMMA GEMM',
              qfn_files, sources=qfn_tus, includes=['src'], flags=FAST + ['-DQFN_ROCM_BUILD=1', '-DENGINE_ENABLE_HIP=1'], source_flags={QFN + 'kernels/rocm/w8a8_wave64.hip': W64},
              requires=['module:kernel.gufo.core_headers', 'module:kernel.gufo.qwen_ops', 'module:kernel.gufo.qfn_mmq', 'format.device-view/gguf-kquant/-'], variants=83,
              provides=[
                  P('hc.mix/gguf-q8_0/f32', 'gufo::models::qwen38_flash_next::rocm::HcMixEpilogue*, HcCombine*, HcMixQ8Fused, HcApplyDeferred (kernels.hpp)', 'fp32 residual streams (4 x 2560, rank 320); fused decode form is byte for byte the separate launches; F16 and tiled Q8 mirrors byte-identical to qfn_mmq_quantize_q8_1; compiled -ffast-math', DEC + '; ' + PF),
                  P('gdn.recurrence/-/f32.qfn', 'GdnPrep, GdnPrepKq, GdnPrepAb, Gdn, GdnEpilogue, RestoreGdnState, RollingSnapshot, SsmConv*', 'fp32 state; rollback keeps the state after the first token and exact fp32 keys / decays / betas / errors for later tokens (no inverse, no quantization)', DEC),
                  P('ple.inject/gguf-q8_0/f32', 'PleGate, PleConv, PleInject, HistoryShift', 'fp32', DEC),
                  P('indexer.qsa/-/f32.qfn', 'PoolBlocks, SelectScore, CompactSelection', 'fp32 block-pooled keys and scores', DEC),
                  P('attention.full/kv-f16/f32.qfn', 'PrepareAttention, Rope, StoreKv, Attention, AttentionMerge, WmmaCausalAttention', 'fp32 math over stored K/V rows; split attention merged from partials', DEC + '; ' + PF),
                  P('moe.route/-/f32.qfn', 'RouterTopK, ExpertCounts, RoutedPadBounds, RoutedScatter', 'top-k on fp32 router logits', DEC),
                  P('moe.combine/-/f32.qfn', 'MoeEpilogue, MoeEpilogueVec4, HcCombineMoeF16', 'weighted sum of expert rows plus the gated shared expert; F16 expert rows in the fused form', DEC),
                  P('linear.prefill/gguf-q8_0/i8-f32.w8a8', 'QuantizeQ8Tiled*, W8A8BlockedWmmaGEMM, W8A8Tail', 'activations quantized per 32-wide block to int8 (Q8_1 tiles), int8 WMMA integer accumulate, one fp scale per block pair', PF),
                  P('linear.decode/f32+bf16+f16/f32.narrow', 'Narrow, SmallGemm, DenseF16GEMM, ExpandQ8_0', 'fp32 accumulate; for the narrow projections the quantized tier does not cover (routers, alpha/beta, indexer, inject)', DEC),
                  P('norm.rms/-/f32.grouped', 'RmsNorm, RmsNormCached', 'grouped RMSNorm sharing one gamma row, fp32', DEC),
                  P('act.swiglu/-/f32.qfn', 'Swiglu, SwigluHalf, SwigluQ8, SiluScale, SigmoidMul', 'fp32; F16 and Q8 output forms', DEC),
                  P('sample.argmax/-/f32.penalty', 'ArgmaxPartial, ArgmaxFinish, SplitArgmax*, PenaltyArgmax*, GatherArgmaxCandidates, VerificationCandidateTile', 'argmax and candidate tiles, lowest id on ties', '1..16 rows x vocab'),
                  P('draft.mtp/gguf-q8_0/f32', 'MtpHidden, MtpHiddenVec4, MtpAddEmbeddingVec4, MtpCandidateTile, MaskDraftLogits, GatherDraftLogits, RemapDraftIds', 'draft-only numerics', 'draft steps'),
                  P('collective.device-stage/-/f32', 'StagePartial, WaitValue, AddRowsBroadcast', 'TP2 partial staging on the device; reduction adds the peer partial in rank order', 'TP2')],
              notes='One 9,352-line translation unit: the kernels cannot be compiled or swapped separately without editing it. kernels.hpp is the launch API.'))
qfn_mmq_tus = O.expand([QFN + 'kernels/rocm/mmq/*.hip.cpp'])
mods.append(M('kernel.gufo.qfn_mmq', 'Quantized matrix kernels for Qwen3.8-Flash-Next derived from llama.cpp ggml-cuda (mmq / mmvq / mmid / quantize), built with HIP; plus the wave64 MoE kernel',
              take([QFN + 'kernels/rocm/mmq/']), sources=qfn_mmq_tus,
              includes=['src', 'src/' + QFN + 'kernels/rocm/mmq'], flags=MMQ + ['-DGGML_HIP_NO_VMM', '-DENGINE_ENABLE_HIP=1'],
              source_flags={QFN + 'kernels/rocm/mmq/moe_wave64.hip.cpp': W64}, variants=len(qfn_mmq_tus),
              upstream='llama.cpp 5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a ggml/src/ggml-cuda (MIT, The ggml authors); see src/.../mmq/VENDOR.md',
              requires=['module:kernel.gufo.core_headers'],
              provides=[P('linear.prefill/gguf-kquant/q8_1-f32.mmq', 'qfn_mmq_* C ABI (qfn_mmq.h): mul_mat_q, quantize_q8_1', 'ggml mmq: activations quantized to Q8_1 (int8 per 32-block with fp scale and sum), integer dot products, fp32 scale; compiled -ffast-math', PF),
                        P('linear.decode/gguf-kquant/f32.mmvq', 'qfn_mmq_* C ABI: mul_mat_vec_q', 'ggml mmvq vec-dot per row', DEC),
                        P('moe.experts-decode/gguf-kquant/f32.mmid', 'qfn_mmq_* C ABI: mul_mat_id, moe_wave64', 'ggml mul_mat_id grouping by expert', DEC)]))

# deepseek v4 flash (ds4 import)
ds4_closure = [f for f in tu_closure([DSR + 'backend.hip.cpp']) if f.startswith(DS4) and '/mmq/' not in f]
D = DSR + 'detail/ds4_rocm_'
DS4UP = 'DS4 84cc882352757baf628a1776badf7cc54d584e28 (MIT, see src/.../LICENSE.ds4 and UPSTREAM.md); converted to native ROCm/HIP by gufo'
DSINC = ['src', 'src/' + DSR.rstrip('/'), 'src/' + DSR + 'detail', 'src/' + DS4 + 'runtime']
VIA = 'header fragment of one translation unit; compiles only inside kernel.gufo.ds4_backend'


def dsh(name, files, summary, provides, extra=()):
    mods.append(M(f'kernel.gufo.{name}', summary, take([D + f + '.hip.hpp' for f in files] + list(extra)), lang='hip-c++', includes=DSINC,
                  compile='via:kernel.gufo.ds4_backend', provides=provides, upstream=DS4UP, reason=VIA, authors=['antirez (DS4)', 'gufo contributors']))


dsh('ds4_attention', ['attention', 'attention_launch', 'decode_attention', 'compressor', 'fp8_kv', 'fp8_kv_launch'],
    'DeepSeek V4 Flash attention: prefill / decode over raw and mixed (compressed) KV, the DSpark support attention, the KV compressor, FP8 KV quantization and raw-cache store',
    [P('attention.sparse-sink/kv-fp8-e4m3/f32', 'ds4_gpu_attention_* (resident_api.h, backend.h)', 'fp32 activations; KV rows quantized to FP8 (E4M3); single-pass rocWMMA prefill kernel; compiled -ffast-math', DEC + '; ' + PF),
     P('kv.store/fp8-e4m3/f32', 'fp8_kv_quantize_kernel, ds4_gpu_fp8_kv_* launchers', 'E4M3 round to nearest of fp32 values', DEC),
     P('kv.compress/-/f32', 'ds4_gpu_compressor_* launchers', 'ratio-4 pooling of raw rows into compressed rows at compression boundaries', DEC)])
dsh('ds4_indexer', ['indexer', 'indexer_score', 'indexer_topk'],
    'DeepSeek V4 Flash sparse-attention indexer: 64 heads of width 128 scored one 128-thread block a score, exact MSD radix top-k, plus the DSpark Markov correction',
    [P('indexer.score/kv-f32/f32', 'ds4_gpu_indexer_score_* launchers', 'fp32, each head owns its slot (no inter-warp sync inside the loop)', DEC),
     P('indexer.topk/-/f32.radix', 'ds4_gpu_indexer_topk_* launchers', 'exact MSD radix selection: +0 and -0 tie, then the lower index wins', DEC),
     P('draft.markov/gguf/f32', 'DSpark Markov correction kernels in ds4_rocm_indexer.hip.hpp', 'argmax(logits + W2 * W1[prev]) without materializing the full correction', 'draft steps')])
dsh('ds4_moe', ['moe', 'moe_launch', 'router', 'shared_expert', 'iq2_gate', 'q2_down'],
    'DeepSeek V4 Flash routed MoE: router, IQ2 gate/up experts, Q2_K down projection on WMMA with row groups of 128/64, shared expert, routed sum',
    [P('moe.route/-/f32.ds4', 'router kernel + ds4_gpu_router_* launchers', 'only the final score is rounded: intermediate fp32 transcendental rounding varies across ROCm libraries and can change routing', DEC),
     P('moe.experts-decode/gguf-iq2+q2_k/f32', 'ds4_gpu_routed_moe_* launchers (47 kernels)', 'IQ2_XXS gate/up decoded through grid tables keeping the established scalar product order; Q2_K down; fp32 accumulate; -ffast-math', DEC),
     P('moe.experts-prefill/gguf-iq2+q2_k/f32', 'ds4_gpu_routed_moe_* launchers (batched, WMMA q2 down)', 'rows grouped by expert in 128-row groups then a 64-row tail', PF)],
    extra=[DSR + 'iq2_tables.hip.hpp'])
dsh('ds4_hc', ['hc', 'hc_output_launch', 'output'],
    'DeepSeek V4 Flash hyper-connections (called hierarchical composition in the source): split / weighted sum / expand over the 4 x 4096 row, fused pre-block chain, output and pointwise kernels',
    [P('hc.mix/gguf-f16/f32.ds4', 'ds4_gpu_hc_* launchers (resident_api.h)', 'fp32 row of 4 x 4096; fused pre-block (norm, mix projection, split, weighted sum) for the supported shape, else the separate chain', DEC + '; ' + PF)])
dsh('ds4_linear', ['common', 'matmul', 'q8', 'q8_decode', 'f16_decode', 'hipblaslt', 'embedding_launch', 'misc_launch'],
    'DeepSeek V4 Flash dense projections: Q8_0 matmul and grouped outputs, narrow Q8 projections shared with the numerical oracle, blocked F16 GEMM, F16 decode, hipBLASLt state, embedding',
    [P('linear.decode/gguf-q8_0/f32.ds4', 'ds4_gpu_matmul_q8_0_* launchers', 'Q8_0 blocks (QK 32), fp32 accumulate, the scalar kernel compiled multiply/FMA order under fast-math', DEC),
     P('linear.decode/f16/f32.ds4', 'ds4_gpu_matmul_f16_* launchers, ds4_gpu_matmul_f16_f16_input_tensor', 'F16 weights; optional published F16 mirror of the activation', DEC),
     P('linear.prefill/f16/f16-f32.ds4', 'blocked F16 GEMM (attention output B projection), hipBLASLt routes', 'F16 operands, fp32 accumulate; library plan order on the hipBLASLt route', PF),
     P('embed.lookup/gguf-f16/f32', 'ds4_gpu_embedding_* launchers', 'exact row copy and widening', DEC)])
dsh('ds4_norm_rope', ['norm', 'norm_rope'],
    'DeepSeek V4 Flash RMS norm with the row held in registers, and the fused head RMS norm plus rotated tail',
    [P('norm.rms/-/f32.ds4', 'ds4_gpu_rms_norm_* launchers', 'fp32; fused form scales the whole row, tail included, exactly as the standalone norm', DEC),
     P('rope.heads/-/f32.ds4-tail', 'ds4_gpu_rope_tail_tensor, ds4_gpu_head_rms_norm_rope_tail_*', 'rotation of the head tail dims in fp32', DEC)])
mods.append(M('kernel.gufo.ds4_dspark', 'DSpark sampling kernels: eight corrected logits per request without a full-vocabulary probability buffer',
              take([DSR + 'detail/ds4_dspark_sampling.hip.hpp']), lang='hip-c++', includes=DSINC, compile='via:kernel.gufo.ds4_backend', upstream=DS4UP,
              provides=[P('draft.dspark-sampling/-/f32', 'ds4_dspark_sampling kernels', 'bounded proposal q with exact rejection / residual correction on the host (dspark_sampler.hpp)', 'draft cycles')],
              reason=VIA, authors=['antirez (DS4)', 'gufo contributors']))
mods.append(M('kernel.gufo.ds4_backend', 'The DeepSeek V4 Flash ROCm backend translation unit: runtime, device helpers and the include order that makes one object of every kernel.gufo.ds4_* fragment',
              take([f for f in ds4_closure] + [DSR + 'backend.hip.cpp']), sources=[DSR + 'backend.hip.cpp'], includes=DSINC,
              flags=FAST + ['-D_GNU_SOURCE', '-DDS4_ROCM_BUILD=1', '-DENGINE_ENABLE_HIP=1'], upstream=DS4UP, authors=['antirez (DS4)', 'gufo contributors'],
              requires=['module:kernel.gufo.ds4_attention', 'module:kernel.gufo.ds4_indexer', 'module:kernel.gufo.ds4_moe', 'module:kernel.gufo.ds4_hc',
                        'module:kernel.gufo.ds4_linear', 'module:kernel.gufo.ds4_norm_rope', 'module:kernel.gufo.ds4_dspark', 'module:kernel.gufo.ds4_mmq',
                        'module:kernel.gufo.core_headers'],
              provides=[P('executor.device-graph/deepseek-v4-flash/gguf', 'ds4_gpu_* C API (backend.h, resident_api.h)', 'single-device resident ROCm route with one authoritative state layout', 'solo')],
              notes='backend.hip.cpp includes 26 detail headers in a fixed order; they share static state, so the family modules are fragments, not separately linkable units.'))
ds4_mmq_tus = O.expand([DSR + 'mmq/*.hip.cpp'])
mods.append(M('kernel.gufo.ds4_mmq', 'Quantized matrix kernels of the DS4 import, derived from llama.cpp ggml-cuda with the DS4-GB10 DSpark changes, built with HIP under ds4_ggml_hip_* symbols',
              take([DSR + 'mmq/']), sources=ds4_mmq_tus, includes=['src', 'src/' + DSR + 'mmq', 'src/' + DSR.rstrip('/')],
              flags=MMQ + ['-DGGML_USE_HIP', '-DGGML_HIP_NO_VMM', '-DDS4_ROCM_WMMA_MMQ_Y=64'], variants=len(ds4_mmq_tus),
              upstream='llama.cpp 5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a ggml/src/ggml-cuda (MIT) plus xangel82/DS4-GB10-GX10-DSpark-CUDA 910501e; see src/.../mmq/VENDOR.md',
              requires=['module:kernel.gufo.core_headers'],
              provides=[P('linear.prefill/gguf-kquant/q8_1-f32.mmq-ds4', 'ds4_mmq.h raw-pointer C ABI (ds4_ggml_hip_*)', 'ggml mmq numerics, WMMA y tile 64; -ffast-math', PF),
                        P('linear.decode/gguf-kquant/f32.mmvq-ds4', 'ds4_mmq.h mul_mat_vec_q entries', 'ggml mmvq', DEC)]))
mods.append(M('kernel.gufo.snapshot_copy', 'Device copy kernel for slot snapshots (KV and recurrent state transfer between live buffers and snapshot storage)',
              take(['src/core/hip/snapshot_copy.hip']), sources=['src/core/hip/snapshot_copy.hip'], includes=['src'], flags=HIPFLAGS,
              requires=['module:kernel.gufo.core_headers'],
              provides=[P('cache.snapshot-copy/-/-', 'gufo::hip snapshot copy launchers (src/core/hip/snapshot_copy.hpp)', 'exact byte copy', 'snapshot save / restore')]))

# ------------------------------------------------------------------ runtime
mods.append(M('runtime.gufo.tp_verbs', 'TP2 over native InfiniBand / RoCE v2 verbs: two-rank RC communicator (one RDMA write with immediate per partial), GID selection, the TP partition of the model, and the serving-side TP control, executor and disk store',
              take([QFN + 'kernels/rocm/verbs.hpp', QFN + 'kernels/rocm/verbs.cpp', QFN + 'kernels/rocm/communicator.hpp', QFN + 'distributed/',
                    'src/cli/serve/tp_control.*', 'src/cli/serve/tp_executor.*', 'src/cli/serve/tp_disk_store.*', 'src/cli/serve/tp_snapshot_gate.hpp', 'RDMA_TEMP_STATE.md']),
              lang='c++', compile='n/a', authors=['neuhaus', 'gufo contributors'],
              provides=[P('collective.exchange-partial/-/f32.gufo', 'gufo::models::qwen38_flash_next::rocm::Communicator (ExchangePartial, StartPartial / FinishPartial, QueuePartial), CreateIbrverbsCommunicator', 'bytes only; every collective carries and validates its operation ordinal; fixed send and receive windows mapped by both ranks', 'TP2; event-driven completion channel')],
              requires=['collective.device-stage/-/f32'], reason=NOTBUILT,
              notes='RDMA_TEMP_STATE.md (repo root) says this branch is an unqualified merge checkpoint with one open correctness bug in prefix/checkpoint reuse.'))
mods.append(M('runtime.gufo.speculative', 'Speculation framework: draft backend interface, prompt-lookup drafts, the lossless speculative verifier, and the Qwen verifier',
              take(['src/core/speculative/', 'src/models/qwen/hip/qwen_speculative_verifier.*']), lang='c++', compile='n/a',
              provides=[P('speculation.window-verify/-/-.lossless', 'gufo::core::speculative::SpeculativeVerifier, DraftBackend, PromptLookup', 'lossless: accepted tokens follow the target distribution (exact rejection / residual sampling)', 'draft width 1..16')],
              requires=['speculation.accept/-/f32.gufo'], reason=NOTBUILT))
mods.append(M('runtime.gufo.dflash', 'DFlash speculator for Qwen: draft model weights, policy, executor, batch and backend',
              take(['src/models/qwen/hip/dflash/', 'src/models/qwen/hip/dflash.hpp', 'src/models/qwen/dflash_policy.hpp', 'src/models/qwen/dflash_weights.*']), lang='c++', compile='n/a',
              provides=[P('speculation.draft/dflash/f32', 'gufo::models::qwen::hip::dflash', 'draft-only numerics', 'draft block')],
              requires=['draft.dflash/gguf-q8_0/f32', 'speculation.window-verify/-/-.lossless'], reason=NOTBUILT))
mods.append(M('runtime.gufo.mtp', 'MTP speculators: the Qwen MTP head (model, executor, backend, reference) and the Qwen3.8-Flash-Next MTP policy, costs and n-gram tables',
              take(['src/models/qwen/hip/mtp/', 'src/models/qwen/hip/mtp.hpp', 'src/models/qwen/mtp_reference.*', QFN + 'mtp_policy.hpp', QFN + 'mtp_costs.hpp',
                    QFN + 'ngram.*', QFN + 'draft_vocab.hpp']), lang='c++', compile='n/a',
              provides=[P('speculation.draft/mtp/f32', 'gufo::models::qwen::hip::mtp, qwen38_flash_next mtp_policy', 'draft-only numerics', 'draft steps')],
              requires=['draft.mtp/gguf-q8_0/f32', 'speculation.window-verify/-/-.lossless'], reason=NOTBUILT))
mods.append(M('runtime.gufo.dspark', 'DSpark speculation for DeepSeek V4 Flash: sampler bridge with bounded proposals and exact rejection / residual correction, policy',
              take([DS4 + 'dspark_sampler.hpp', DS4 + 'runtime/dspark_policy.h', DS4 + 'runtime/dspark_internal.h']), lang='c++', compile='n/a', upstream=DS4UP,
              includes=['src', 'src/' + DS4 + 'runtime'],
              provides=[P('speculation.draft/dspark/f32', 'gufo::models::deepseek_v4_flash::DsparkSamplerBridge', 'target filters and penalties always see the complete vocabulary; a rejected draw becomes the next cycle anchor', 'draft cycles')],
              requires=['draft.dspark-sampling/-/f32', 'draft.markov/gguf/f32'], reason=NOTBUILT))
mods.append(M('runtime.gufo.cache', 'Continuation and prefix caches: slot store, prefix index, checkpoints, disk catalog and ledger, retention, streaming; the serving-side continuation cache, identity and disk store; component text cache',
              take(['src/cache/', 'src/cli/serve/continuation_*', 'src/cli/serve/component_text_cache.*', 'src/cli/serve/component_cache_metrics.hpp',
                    'src/core/hip/component_cache_resources.*', 'src/core/hip/committed_backing.*', 'src/core/hip/staging.*', 'src/core/hip/transfer_pool.*']),
              lang='c++', compile='n/a',
              provides=[P('cache.prompt-state/-/-.gufo', 'gufo::cache (slot.hpp, prefix_index.hpp, checkpoint.hpp, disk.hpp), ContinuationCache', 'exact reuse of a saved slot state for a matching token prefix', 'RAM and disk tiers')],
              requires=['cache.snapshot-copy/-/-'], reason=NOTBUILT))
mods.append(M('runtime.gufo.hip_host', 'HIP host services: host allocations, managed allocation, heap placement, weight upload, snapshot transfer, hipBLASLt plan database, bandwidth probe',
              take(['src/core/hip/'], exclude=['src/core/hip/allocation_benchmark.hip']), lang='c++', compile='n/a',
              provides=[P('device.memory/hip/-', 'gufo::hip host_allocations, managed_alloc, weight_upload, snapshot_transfer', 'n/a', 'unified memory on gfx1151')],
              requires=['module:kernel.gufo.core_headers'], reason=NOTBUILT))
mods.append(M('runtime.gufo.sampling', 'Host sampling and constrained decoding: sampler state and penalties, JSON-schema constraint (lexeme and regex compilers), reasoning and session modes',
              take(['src/core/sampling.*', 'src/core/json_constraint.*', 'src/core/json_schema_*', 'src/core/text_sampling_defaults.hpp', 'src/core/reasoning.hpp',
                    'src/core/session_mode.hpp', 'src/core/utf8.hpp', 'src/core/json.hpp', 'src/core/cancellable_gate.hpp', 'src/core/mapped_prefetch.hpp', 'src/core/model_config.hpp']),
              lang='c++', compile='n/a',
              provides=[P('sample.host/-/f32.gufo', 'gufo::sampling::SamplerState', 'fp32 logits; penalties as unique-token records', 'host'),
                        P('grammar.constraint/-/-.gufo', 'gufo::core::JsonConstraint', 'n/a', 'host')],
              requires=['module:kernel.gufo.core_headers'], reason=NOTBUILT,
              notes='The matching .hpp files that kernel translation units include are in kernel.gufo.core_headers.'))
mods.append(M('runtime.gufo.platform', 'Platform leaves (POSIX side): mapped files, page pinning, prefault, publish, sockets, environment, shutdown signal; crypto, logging, diagnostics',
              take(['src/core/platform/', 'src/core/crypto/', 'src/core/logging/', 'src/core/diagnostics/'], exclude=['**_windows.cpp']), lang='c++', compile='n/a',
              provides=[P('host.platform/posix/-', 'gufo::platform', 'n/a', 'host')], reason=NOTBUILT))
mods.append(M('runtime.gufo.ext_api', 'The public extension API (include/gufo/ext) and its runtime: provider registry, options, host budget, device probe, text contracts',
              take(['include/', 'src/ext/'], exclude=['include/gufo/ext/embeddings.hpp']), lang='c++', compile='n/a',
              includes=['src', 'src/include'],
              provides=[P('executor.provider-api/gufo-ext/-', 'gufo::ext (adapter.hpp, registry.hpp, continuation.hpp, draft_backend.hpp, speculative_verifier.hpp)', 'n/a', 'one provider per model family')],
              reason=NOTBUILT))

# ------------------------------------------------------------------ executors
mods.append(M('executor.gufo.qwen38_flash_next', 'Qwen3.8-Flash-Next engine: config, weights, device model, ROCm executor and batch paths, continuation layout and adapter, RoPE scaling, CPU reference',
              take([QFN], exclude=[QFN + 'functional/**']), lang='c++', compile='n/a',
              provides=[P('executor.model/qwen3.8-flash-next/gguf', 'gufo::models::qwen38_flash_next (engine.hpp), provider entry in extension.json', 'fp32 activations on GGUF weights; TP2 adds partials in rank order; MTP drafts verified losslessly', 'solo and TP2 (RDMA, unqualified branch)')],
              requires=['hc.mix/gguf-q8_0/f32', 'gdn.recurrence/-/f32.qfn', 'ple.inject/gguf-q8_0/f32', 'indexer.qsa/-/f32.qfn', 'attention.full/kv-f16/f32.qfn', 'moe.route/-/f32.qfn',
                        'moe.combine/-/f32.qfn', 'linear.prefill/gguf-q8_0/i8-f32.w8a8', 'linear.prefill/gguf-kquant/q8_1-f32.mmq', 'linear.decode/gguf-kquant/f32.mmvq',
                        'moe.experts-decode/gguf-kquant/f32.mmid', 'draft.mtp/gguf-q8_0/f32', 'speculation.draft/mtp/f32', 'collective.exchange-partial/-/f32.gufo',
                        'format.loader/gguf/-', 'cache.prompt-state/-/-.gufo', 'device.memory/hip/-', 'executor.provider-api/gufo-ext/-'],
              reason=NOTBUILT))
mods.append(M('executor.gufo.deepseek_v4_flash', 'DeepSeek V4 Flash engine (DS4 import): stable gufo engine API over the imported model data, tokenizer, request sessions, snapshots and ROCm graph execution; chat template and tool calls',
              take([DS4], exclude=[DS4 + 'functional/**', DS4 + 'reference/**']), lang='c++', compile='n/a', upstream=DS4UP, authors=['antirez (DS4)', 'gufo contributors'],
              includes=['src', 'src/' + DS4 + 'runtime'],
              provides=[P('executor.model/deepseek-v4-flash/gguf', 'gufo::models::deepseek_v4_flash (engine.hpp)', 'single-device resident ROCm route; DSpark drafts with exact correction', 'solo only: the DS4 distributed / TP paths are not built or shipped (UPSTREAM.md)')],
              requires=['executor.device-graph/deepseek-v4-flash/gguf', 'speculation.draft/dspark/f32', 'format.loader/gguf/-', 'executor.provider-api/gufo-ext/-'],
              reason=NOTBUILT))
mods.append(M('runtime.gufo.image', 'Image input helpers: the image and pixel types the Qwen vision headers name, decoding (PNG, JPEG, WebP), fetching by URL, resizing',
              take(['src/core/image.*']), lang='c++', compile='n/a',
              provides=[P('media.image-decode/-/-.gufo', 'gufo::core image.hpp', 'n/a', 'host; needs libpng, libjpeg, libwebp and libcurl')],
              reason=NOTBUILT, notes='Harvested for phase 5: executor.gufo.qwen names these types through the vision headers. A text-only recipe carries the header and does not compile image.cpp.'))
mods.append(M('executor.gufo.qwen_vision', 'Qwen vision input: the image encoder (ViT tower on HIP), its device input, the prompt layout with image spans and the RoPE layout the text executor reads, the image cache',
              take(['src/families/qwen/vision/']), lang='c++', compile='n/a',
              provides=[P('executor.vision/qwen3.5/gguf', 'gufo::models::qwen::vision (encoder.hpp, device_input.hpp, prompt.hpp, image_cache.hpp)', 'fp32 on the mmproj GGUF', 'solo')],
              requires=['format.loader/gguf/-', 'device.memory/hip/-'],
              reason=NOTBUILT, notes='Harvested for phase 5: src/models/qwen/hip/executor.hpp includes vision/device_input.hpp and the origin links gufo_qwen_vision into gufo_qwen27b, so the text executor does not build without it.'))
mods.append(M('executor.gufo.qwen', 'Qwen (3.5 / 3.6 family) engine: forward, state, generator, weights, the HIP executor (arena, decode step, batched decode, prefill chunks, SSM replay), composable modules, tokenizer, chat template, config',
              take(['src/models/qwen/', 'src/families/qwen/'], exclude=['src/models/qwen/functional/**', 'src/families/qwen/vision/**', 'src/families/qwen/text/reference/**', '**/AGENTS.md', '**/README.md']),
              lang='c++', compile='n/a',
              provides=[P('executor.model/qwen3.5/gguf', 'gufo::models::qwen (generator.hpp, hip/executor.hpp)', 'fp32 activations on GGUF weights, canonical FP16 KV', 'solo')],
              requires=['linear.decode/gguf-kquant/f32.gufo', 'linear.prefill/gguf-kquant/bf16-f32', 'attention.full/kv-f16/f32.gufo', 'gdn.recurrence/-/f32.gufo', 'norm.rms/-/f32.gufo', 'rope.apply/-/f32.gufo',
                        'sample.topk/-/f32.gufo', 'speculation.draft/dflash/f32', 'speculation.draft/mtp/f32', 'format.loader/gguf/-', 'cache.prompt-state/-/-.gufo', 'device.memory/hip/-'],
              reason=NOTBUILT, notes='src/models/qwen/hip/attention.hip (4 lines, an include shim) is here, not in a kernel module.'))
mods.append(M('serve.gufo.http', 'gufo serve: HTTP server, OpenAI chat API, text generation scheduler and model runner, inference backend, logprobs, stop sequences, response formats',
              take(['src/cli/serve/'], exclude=['src/cli/serve/asr_*', 'src/cli/serve/audio_*', 'src/cli/serve/tts_*', 'src/cli/serve/image_api.*', 'src/cli/serve/video_*',
                                               'src/cli/serve/embedding*', 'src/cli/serve/websocket.*']),
              lang='c++', compile='n/a',
              provides=[P('serve.http/openai/-.gufo', 'gufo serve (serve.cpp, http_server.cpp, openai_chat.cpp)', 'n/a', 'host')],
              requires=['executor.provider-api/gufo-ext/-', 'cache.prompt-state/-/-.gufo', 'sample.host/-/f32.gufo'], reason=NOTBUILT))

ORIGIN_TOML = '''
# gufo with the neuhaus RDMA fork: C++20 + HIP engine for gfx1151 (Qwen, Qwen3.8-Flash-Next, DeepSeek V4 Flash).
name      = "gufo"
repo      = "https://github.com/neuhaus/gufo"
commit    = "f070a17"
branch    = "rdma"                                       # unqualified merge checkpoint, see RDMA_TEMP_STATE.md
upstream  = "https://github.com/gufo-org/gufo (version.txt 0.28.1; merged upstream/main 12effe36)"
pod_tree  = "/data/ark-inbox/src/github.com__neuhaus__gufo"
reference_checkout = "~/Code/strixflerp-upstream/gufo-neuhaus"

[licence]
spdx      = "MIT"
files     = ["LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"]        # copied verbatim next to this file; licenses/ and src/models/deepseek_v4_flash/LICENSE.ds4 in the origin
third_party = "DeepSeek V4 Flash engine adapted from DS4 84cc8823 (MIT); quantized matrix kernels from llama.cpp 5c0e9468 ggml-cuda (MIT) with xangel82/DS4-GB10-GX10-DSpark-CUDA 910501e changes"

[toolchain]
host      = "C++20, CMake + Ninja; pinned Nix flake (ROCm 10.1.0, HIP clang) in the origin"
device    = "therock-10.1"                               # toolchains.toml: /data/tools/therock-10.1.0a20260823 on control-node
libraries = "hipBLAS, hipBLASLt, hipCUB, rocPRIM, rocWMMA; libibverbs for the TP2 RDMA adapter (GUFO_ENABLE_TP2_RDMA)"

[compile]                                                # CMake's HIP language: clang++ -x hip, one object per source
driver    = "<rocm>/lib/llvm/bin/clang++"
arch_flag = "--offload-arch=gfx1151"
rocm_flag = "--rocm-path=<rocm>"
flags     = ["-x", "hip", "-std=gnu++20", "-O2", "-UNDEBUG", "-DENGINE_ENABLE_HIP=1", "-c"]
system_includes = ["include"]
output    = "host object with the gfx1151 device code embedded (.o), one per translation unit"
note      = "RelWithDebInfo adds -g in the origin; the gate leaves it out to keep objects small. Per-source options (-mwavefrontsize64, -fno-slp-vectorize, iterative-ilp) are copied from src/families/qwen/kernels/CMakeLists.txt into [build.source_flags]."

[kernels]
loading   = "statically linked: each .hip translation unit carries its kernels and host launchers (gufo::hip::Launch*); no runtime kernel registry"
dispatch  = "host launch functions choose kernels by shape and GGUF type (src/families/qwen/hip_common/gemm_route.hpp, execution_policy.hpp)"
'''

LEFT_OUT = {
    'src/models/{qwen3_asr,qwen3_tts,minimax_h3,qwen_image_21,embedding_gemma2}, src/cli/{transcribe,video}, src/cli/serve/{asr,tts,audio,image,video,embedding,websocket}*': 'audio, image, video and embedding models and their serving endpoints',
    'src/core/hip/allocation_benchmark.hip, src/cli/{bench,diagnose,eval,prompt}, src/eval, src/testing, tests/, tools/': 'benchmarks, diagnostics, evaluation harness, tests, tools',
    'src/core/platform/*_windows.cpp, BUILD-WINDOWS.md, cmake/toolchains/windows-therock.cmake': 'Windows platform code',
    'src/cli/{main.cpp,banner.*,arg_parser.hpp,extension_diagnostics.*,sampling_options.hpp}': 'the CLI front end outside serve',
    'src/models/*/functional, src/models/deepseek_v4_flash/reference, src/families/qwen/text/reference': 'functional-test descriptors and reference templates',
    'CMakeLists.txt, cmake/, flake.nix, vcpkg.json, docs/, assets/, schemas/, *.md at the root': 'build system and documentation (the per-target compile options are recorded in the manifests)',
}

if __name__ == '__main__':
    run(O, mods, ORIGIN_TOML, LEFT_OUT, ['LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md'])
