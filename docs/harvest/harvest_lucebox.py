#!/usr/bin/env python3
"""Harvest Lucebox (Luce-Org/lucebox, a70ab3c) into catalog/lucebox.
Run: python3 docs/harvest/harvest_lucebox.py

Card ORIGIN-LUCEBOX-HARVEST 0bd0e0db19; verdict of ORIGIN-LUCEBOX c9e9b2a327 (/data/pod/reports/2026-11/origin-lucebox.md).
Scope: the HIP kernel families of the report. Lucebox's kernels are of two kinds:
  - ggml ops in the vendored ggml (server/deps/llama.cpp/ggml/src/ggml-cuda/*.cu, MIT, The ggml authors), compiled as HIP
    by ggml/src/ggml-hip/CMakeLists.txt; 28 of those .cu files are Lucebox's own (absent from upstream ggml-org/llama.cpp);
  - nine HIP translation units of the hub's own server (server/src, Apache-2.0), added with LANGUAGE HIP in
    server/CMakeLists.txt and reaching HIP through the server/hip_compat shim headers.
Each family is one module that owns its .cu files and the headers named after them; two header-only modules carry the
#include closure of those units (ggml_headers: the ggml side, server_headers: the server side). The upstream ggml-cuda
files that carry Lucebox edits (mmvq, mmid, mmq, fattn, ...) stay out: without lucebox-ggml luce-dflash@6fbe72d in the
ark their edits cannot be told from upstream drift (report, section ark additions).

The reference copy is control-node's ark clone /data/ark-inbox/src/github.com__Luce-Org__lucebox (a depth-1 clone at
a70ab3c). The laptop copy ~/Code/strixflerp-upstream-supplement/lucebox is 'git archive a70ab3c' of it with LFS
smudging off (/data/pod/plans/0bd0e0db19/lucebox-a70ab3c.tar.xz, sha256 e9e0d120...; the tar inside b2f57d97...); LUCEBOX_CHECKOUT names another copy."""
import os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvestlib, incgraph
from harvestlib import Origin, M, P, run

CHECKOUT = os.environ.get('LUCEBOX_CHECKOUT', os.path.join(harvestlib.SUPPLEMENT, 'lucebox'))
COMMIT = 'a70ab3ca25a945d97580043230dc33984aaf1fb2'
G = 'server/deps/llama.cpp/ggml'
CU = G + '/src/ggml-cuda'


class TreeOrigin(Origin):
    """The origin's file set is the archive tree of COMMIT (no submodules: Block-Sparse-Attention is not in it)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.roots = [CHECKOUT]
        if not os.path.isfile(os.path.join(CHECKOUT, CU, 'ds4-hc.cu')):
            raise SystemExit(f'{CHECKOUT} does not look like lucebox a70ab3c (no {CU}/ds4-hc.cu)')


O = TreeOrigin('lucebox', 'lucebox', 'https://github.com/Luce-Org/lucebox', COMMIT, 'main', 'Apache-2.0', ['Luce-Org'], 'therock-10.2')

HIP_COMPAT_GGML = 'server/src/hip_compat'    # server/CMakeLists.txt:516: target_include_directories(ggml-hip BEFORE PRIVATE src/hip_compat)
GGML_ROOTS = [HIP_COMPAT_GGML, G + '/include', G + '/src', CU]
# server/CMakeLists.txt:510-515: ggml-hip PRIVATE defines for CUDA spellings the vendored HIP shim does not map
GGML_FLAGS = ['-DcublasSgemmStridedBatched=hipblasSgemmStridedBatched', '-DcudaStreamCaptureStatus=hipStreamCaptureStatus',
              '-DcudaStreamCaptureStatusNone=hipStreamCaptureStatusNone', '-DcudaStreamIsCapturing=hipStreamIsCapturing']
SERVER_ROOTS = ['server/include', 'server/src', 'server/hip_compat', G + '/src', G + '/include', 'server/deps/llama.cpp/common']
GGML_SPDX = 'MIT'
SERVER_SPDX = 'Apache-2.0'
SERVER_FLAGS = ['-DLUCE_BACKEND_HIP=1', '-DLUCE_HAVE_DRAFT_TOPK=1', '-DLUCE_HAVE_KVFLASH_FUSED_MASS=1']
NOT_RUN = 'Nothing was run on a GPU or timed on the pod.'

F = {}
def fam(name, summary, section, ggml, server, headers, provides, why, sflags=None):
    """One family: ggml .cu names (in ggml-cuda), server .cu paths, headers it owns, what it provides."""
    F[name] = dict(summary=summary, section=section, ggml=ggml, server=server, headers=headers, provides=provides,
                   why=why, sflags=sflags or {})

fam('hc', 'DeepSeek4 hyper-connections as fused ggml ops (pre, post and mix modes on the f32 residual streams), the W8A8 Q8 activation tiles of the HC down projection, the stream mix with an optional q8_1 output, and the server-side DeepSeek4 HC kernels',
    '7. Hyper-connections',
    ['ds4-hc.cu', 'hc-cn.cu', 'hc-mix.cu'], ['server/src/deepseek4/deepseek4_hc_cuda.cu'],
    [CU + '/ds4-hc.cuh', CU + '/hc-cn.cuh', CU + '/hc-mix.cuh', CU + '/hc-match.inc', 'server/src/deepseek4/deepseek4_hc_cuda.h'],
    [P('hc.mix/-/f32.lucebox', 'GGML ops dispatched from ggml-cuda.cu: ds4-hc.cu (modes pre / post), hc-mix.cu, hc-cn.cu (ggml_backend_cuda_context, ggml_tensor)',
       'f32 residual streams and mix; streams summed in order 0..hc-1 (hc-mix.cu header); q8 output as MMVQ quantize_q8_1 writes it',
       'DeepSeek V4 / V4.1 Flash (4 streams) and the qwen4exp family; decode and prefill'),
     P('hc.mix/-/f32.lucebox-server', 'server/src/deepseek4/deepseek4_hc_cuda.h C entry points',
       'f32; contract in deepseek4_hc_cuda.h', 'DeepSeek V4 server graph')],
    'A third DeepSeek-V4 hyper-connection implementation on gfx1151 beside kernel.gufo.ds4_hc and StrixFold, with a W8A8 down-projection path neither has')
fam('indexer', 'DeepSeek V4 sparse-attention indexer scores as a ggml op, a rocWMMA 2.x path with a scalar fallback for older or header-less ROCm, kept bit-aligned with the official DeepSeek V4 graph',
    '5. Sparse-attention indexer and QSA', ['ds4-indexer.cu'], [], [CU + '/ds4-indexer.cuh'],
    [P('attn.indexer-score/-/f32.lucebox', 'GGML op in ds4-indexer.cu', 'bit-for-bit aligned with the official DeepSeek V4 graph (file header); rocWMMA only when ROCWMMA_VERSION_MAJOR > 1',
       'DeepSeek V4 / V4.1 Flash prefill and decode')],
    'The second DeepSeek V4 indexer in the catalogue written as a WMMA kernel for gfx1151, against kernel.gufo.ds4_indexer and StrixFold indexer_score')
fam('qsa', 'QSA (Qwen3.8-Flash-Next sparse attention) on gfx11 / gfx12 WMMA fragments with the sigmoid gate folded into the store, the prefill union builder, decode variants, and a sort of the selected ids per query',
    '5. Sparse-attention indexer and QSA', ['qsa.cu', 'qsa-ids.cu'], [],
    [CU + '/qsa.cuh', CU + '/qsa-decode.cuh', CU + '/qsa-decode-wmma.cuh', CU + '/qsa-ids.cuh'],
    [P('attn.softmax-gqa/kv-f16/f16.lucebox-qsa', 'GGML ops in qsa.cu', 'F16 store byte-identical to the plain path with the gate folded in (server/CMakeLists.txt test_qsa3_gate_parity)',
       'Qwen3.8-Flash-Next prefill (union) and decode'),
     P('attn.qsa-ids/-/-', 'GGML op in qsa-ids.cu', 'sorts only the selected ids, never scores; one CTA a query', 'Qwen3.8-Flash-Next')],
    'A further QSA provider beside kernel.strixite.qsa, kernel.libstrix.attn and gufo qfn, with a decode WMMA path')
fam('paged_attn', 'paged decode attention with the context split into partitions (one per 1024 tokens at most), and its RDNA WMMA variant',
    'none (INTERFACES.md has no attention section)', ['paged-attn.cu'], [],
    [CU + '/paged-attn.cuh', CU + '/paged-attn-common.cuh', CU + '/paged-attn-rdna-wmma.cuh'],
    [P('attn.softmax-gqa/kv-f16/f16.lucebox-paged', 'GGML op in paged-attn.cu', 'half head-dim row and a float2 per output row a partition (file header)',
       'decode on a paged KV cache; partitions bounded so a 2048-row chunk at 118K context fits (file header)')],
    'Paged attention with an RDNA WMMA path; the catalogue has paged attention only in libstrix (torch layouts)')
fam('attn_prefill', 'prefill attention: the chunked-prefill FlashAttention driver with strided dequant per chunk, the GGML_OP_FLASH_ATTN_SPARSE dispatch, the rocWMMA block-sparse flashprefill kernels (PFlash) and their BSA-compatible launcher',
    'none (attention); the sparse selection is nearest 5', ['fattn-chunked.cu', 'fattn-sparse.cu'],
    ['server/src/flashprefill_kernels.hip.cu', 'server/src/bsa_launcher_hip.cu'],
    [CU + '/fattn-chunked.cuh', CU + '/fattn-sparse.cuh', CU + '/fattn-rdna-wmma.cuh'],
    [P('attn.prefill-causal/kv-f16/f16.lucebox-chunked', 'GGML op in fattn-chunked.cu', 'K/V extracted per chunk at chunk_len x nh_kv x D (file header)', 'long prefill'),
     P('attn.block-sparse/kv-bf16/bf16.lucebox-pflash', 'fattn-sparse.cu dispatch; server flashprefill_kernels.hip.cu and the extern "C" launcher of bsa_launcher_hip.cu',
       'BF16 Q/K/V in flash_prefill_forward_bf16 layout, rocWMMA 16x16x16 tiles (file header)',
       'speculative and long prefill; built by the origin only with LUCE_HIP_SM80_EQUIV=ON (flashprefill) and LUCE_ENABLE_BSA=ON (launcher), both off in Dockerfile.rocm')],
    'The only block-sparse prefill attention for gfx1151 in the catalogue (PFlash)',
    {'server/src/flashprefill_kernels.hip.cu': ['-DLUCE_HAVE_FLASHPREFILL=1', '-DLUCE_CUDA_MIN_SM=80'],
     'server/src/bsa_launcher_hip.cu': ['-DLUCE_HAVE_BSA=1', '-DLUCE_HAVE_FLASHPREFILL=1', '-DLUCE_CUDA_MIN_SM=80']})
fam('gdn', 'Gated DeltaNet pieces: SiLU conv with the q/k L2 norm folded in, the replay-log commit of speculative chains (paged and single-sequence), the gated norm, and the server SpecLA DeltaConstruct commit',
    '6. Gated DeltaNet', ['gdn-conv.cu', 'gdn-chain-commit.cu', 'gdn-replay-log.cu', 'gated-norm.cu'], ['server/src/common/specla_commit_cuda.cu'],
    [CU + '/gdn-conv.cuh', CU + '/gated-norm.cuh', 'server/src/common/specla_commit_cuda.h'],
    [P('mixer.gdn-front/-/f32.lucebox', 'GGML op in gdn-conv.cu', 'rms_norm(eps) * post_scale + bias per 128-channel head with the thread mapping of rms_norm_scale_f32 (file header)', 'GDN prefill and verify'),
     P('mixer.gdn-commit/-/f32.lucebox', 'gdn-replay-log.cu, gdn-chain-commit.cu (ggml-cuda-gdn-chain.h)', 'commit of the accepted prefix of a speculative chain from the replay log', 'speculative decode on hybrid GDN models'),
     P('norm.gated/-/f32.lucebox', 'GGML op in gated-norm.cu', 'sigmoid(z) * x, with or without the RMS norm; the reduction of rms_norm_f32<256, true> (file header)', 'GDN heads and attention output gate'),
     P('mixer.gdn-commit/-/f32.lucebox-specla', 'server/src/common/specla_commit_cuda.h', 'fused DeltaConstruct commit (file header)', 'SpecLA speculative decode')],
    'Speculative-decode state commit for GDN layers is what strixite and libstrix solve differently (MTP decode kernels); this is a third way')
fam('moe', 'MoE expert kernels: fused decode expert dot products on GGUF i-quants, the fused post-MMID route reduction with the shared-expert add, and an F16 WMMA routed-expert GEMM for the ROCmFP2 / ROCmFP3 MIX expert types',
    '4. MoE: router, experts, combine', ['moe-fused.cu', 'moe-fused-combine.cu', 'mix-wmma-moe.cu'], [],
    [CU + '/moe-fused.cuh', CU + '/moe-fused-combine.cuh', CU + '/mix-wmma-moe.cuh'],
    [P('moe.experts-decode/gguf-iquant/f32.lucebox', 'GGML op in moe-fused.cu', 'one weight row of 256 values against the input vector per dot (file header)', 'decode'),
     P('moe.combine/-/f32.lucebox', 'GGML op in moe-fused-combine.cu', 'dst[t, d] = shared_out[t, d] + sum over used experts of down_e[t, e, d] * weights[t, e] (file header)', 'decode and prefill'),
     P('moe.experts-prefill/rocmfp-mix/f16.lucebox', 'GGML MUL_MAT_ID path in mix-wmma-moe.cu', 'F16 WMMA; decodes each weight tile once instead of per 64-route tile (file header)', 'large mul_mat_id batches (prefill), qtypes 105, 106, 107')],
    'The ROCmFP MIX expert GEMM reads a weight format no catalogue module reads; the fused combine is a candidate against strixite moe_grouped')
fam('gemv', 'small-M GEMV / GEMM kernels: IQ4_NL, Q8 / F16 and W8A8 tiles staged in LDS with a v_perm LUT decode, small-M helpers, and an MXFP8 GEMV',
    '1. Decode GEMV', ['mmb.cu', 'mxfp8.cu'], [],
    [CU + '/mmb.cuh', CU + '/mmb-q8f16.cuh', CU + '/mmb-quant.cuh', CU + '/mmb-small-m.cuh', CU + '/mmb-w8a8.cuh',
     CU + '/small-m-gemm.cuh', CU + '/small-m-qgemm.cuh', CU + '/mxfp8.cuh'],
    [P('linear.decode/gguf-iquant+q8/bf16.lucebox-mmb', 'GGML ops in mmb.cu', 'IQ4_NL decoded to bf16 in LDS, LUT applied with v_perm_b32 (file header)', 'decode and small M'),
     P('linear.decode/mxfp8/f32.lucebox', 'GGML op in mxfp8.cu', 'one coalesced 256-byte code read a block, scale group of 32 (file header)', 'decode')],
    'Decode GEMV is the largest slice of decode time; these are new providers to set against kernel.strixite.linear_q8 and the hipfire GEMVs')
fam('rocmfp', 'runtime decode of the ROCmFP2 and ROCmFP3 MIX weight types (per-expert codebook and mode from a per-tensor registry) for the MMVQ and MMID paths, with the ROCmFP4 HIP codebook and scale headers',
    '1. Decode GEMV and 4. MoE (weight format)', ['rocmfp2_mix.cu', 'rocmfp3_mix.cu'], [],
    [CU + '/rocmfp2_mix.cuh', CU + '/rocmfp3_mix.cuh', G + '/rocmfp4/rocmfp4_hip_codebook.cuh', G + '/rocmfp4/rocmfp4_hip_scale.cuh'],
    [P('format.decode/rocmfp2-mix+rocmfp3-mix/f16.lucebox', 'to_fp16 converters and registry in rocmfp2_mix.cu, rocmfp3_mix.cu', 'GGML_TYPE_Q2_1_ROCMFP2_MIX (106) and Q3_1_ROCMFP3_MIX (105) decoded with tables staged to device by the deepseek4 loader (file header)',
       'DeepSeek V4 expert weights')],
    "ROCmFP is Lucebox's own expert format (2- and 3-bit mixes) behind its MAX+ figures; no catalogue module reads it")
fam('kvquant', 'Walsh-Hadamard rotation for TurboQuant KV (strided source, scalar on HIP) and the TQ3 code helpers',
    '9. Quantized KV', ['turbo-wht.cu'], [], [CU + '/turbo-wht.cuh', CU + '/tq3-quant.cuh'],
    [P('kvcache.rotate/tq3/f32.lucebox', 'GGML op in turbo-wht.cu', 'established scalar implementation on HIP (file header)', 'KV store and read with TurboQuant TQ3 codes')],
    'A rotated 3-bit KV format beside strixite kv8 and StrixFold FP4 KV')
fam('sampling', 'small-k top-k with softmax probabilities over device-resident vocabulary rows for draft candidates, and the server DDTree draft top-K with log-probs',
    '10. Sampling and top-k', ['topk-rows.cu'], ['server/src/common/geometric_draft_topk_cuda.cu'],
    ['server/src/common/geometric_draft_topk_cuda.h'],
    [P('logits.topk/-/f32.lucebox', 'GGML op in topk-rows.cu; server geometric_draft_topk_cuda.h', 'k <= 8 candidates out of a ~100k vocab with softmax probabilities (file header)', 'draft-head candidates, verify width, draft trees')],
    'Draft-candidate top-k for speculative decoding, against kernel.strixite.logits_topk')
fam('inkling2', 'fused HIP kernels of Inkling2-Loop and its bundled DSpark speculator, all graph-safe (deterministic grids, pool allocations)',
    '11. Drafts and speculation', ['inkling2-ops.cu'], [], [CU + '/inkling2-ops.cuh'],
    [P('draft.dspark/-/f32.lucebox-inkling2', 'GGML ops in inkling2-ops.cu', 'launches asynchronously on ctx.stream(), HIP-graph and CUDA-graph safe (file header)', 'Inkling2-Loop and its DSpark speculator')],
    "The DSpark speculator kernels are what the card ORIGIN-LUCEBOX named as Lucebox's value beside the plain kernels")
fam('kvflash', 'the exact chunk attention-mass scorer of the KVFlash GPU scorer (KV selection)',
    'none (KV selection; nearest 9)', [], ['server/src/common/kvflash_fused_mass_cuda.cu'], ['server/src/common/kvflash_fused_mass_cuda.h'],
    [P('kvcache.score-mass/-/f32.lucebox', 'server/src/common/kvflash_fused_mass_cuda.h', 'exact chunk mass (header)', 'KVFlash long-context KV selection')],
    'No catalogue module scores KV chunks for eviction or selection')
fam('norm', 'per-token RMSNorm times weight for the HIP chunk-B graph, built in every HIP build',
    '8. Norm and RoPE', [], ['server/src/rms_norm_hip.cu'], [],
    [P('norm.rmsnorm/-/f32.lucebox', 'server/src/rms_norm_hip.cu', 'per-token RMS norm then weight multiply (file header)', 'qwen3 chunk-B graph path')],
    'A small second RMSNorm provider; recorded for completeness of the HIP units')
fam('collective', 'device halves of the cluster transports: f32 / bf16 conversion for the RCCL compressed allreduce, and row pack and add of the lean decode-path reduction behind stream memory waits',
    '12. TP collective', [], ['server/src/cluster/cluster_comm_kernels.cu', 'server/src/cluster/fast_reduce_kernels.cu'], [],
    [P('collective.allreduce-rccl-bf16/-/f32.lucebox', 'server/src/cluster/cluster_comm_kernels.cu', 'f32 to bf16 by hand, summed after the RCCL exchange (file header)', 'multi-box expert parallel (LUCE_CLUSTER=ON)'),
     P('collective.allreduce-rdma/-/bf16.lucebox', 'server/src/cluster/fast_reduce_kernels.cu', 'pack, wait (hipStreamWaitValue32 in fast_reduce.cpp), add (file header)', 'decode-path reduction over raw RDMA (LUCE_CLUSTER=ON)')],
    "A second RDMA decode reduction to read against StrixFold's transports")
fam('vision', 'the four DS4V vision tower ops: a hipBLASLt BF16 bias matmul, and F32 softmax, rotary and AV kernels that keep PyTorch eager arithmetic',
    '13. Image input for DeepSeek-V4.1', ['vision-av.cu', 'vision-bias.cu', 'vision-rotary.cu', 'vision-softmax.cu'], [],
    [CU + '/vision-av.cuh', CU + '/vision-bias.cuh', CU + '/vision-rotary.cuh', CU + '/vision-softmax.cuh', CU + '/vision-softmax-kernels.cuh'],
    [P('vision.ds4v-ops/bf16/f32.lucebox', 'GGML_OP_MUL_MAT_BIAS_BF16, GGML_OP_RMS_NORM_VISION_F32, GGML_OP_SOFT_MAX_VISION_F32, GGML_OP_MUL_MAT_VISION_AV_F32 (VENDOR.md)',
       'F32 eager materialisation as PyTorch 3d3aa83 does it (vision-rotary.cu header); vision-softmax.cu built with -fno-fast-math -ffp-contract=off',
       'DS4V vision tower; built only with hipBLASLt (GGML_HIP_DS4V_VISION)')],
    'A second DeepSeek V4.1 image-input path beside strixfold_pr6',
    {CU + '/vision-softmax.cu': ['-fno-fast-math', '-ffp-contract=off']})
fam('ple', 'the fused PLE (per-layer embedding) conv matcher of the qwen4exp graph',
    'none', ['ple-conv.cu'], [], [CU + '/ple-conv.cuh'],
    [P('embed.ple-conv/-/f32.lucebox', 'GGML op in ple-conv.cu', 'matched subgraph replaced by one fused kernel; LUCE_PLE_FUSE_LEGACY=1 keeps the unfused graph (file header)', 'qwen4exp per-layer embeddings')],
    'Model-specific; recorded so every Lucebox HIP unit has a home')

FAMILIES = list(F)


def units(f):
    return [CU + '/' + s for s in F[f]['ggml']] + F[f]['server']


def closures():
    files = set(O.files())
    g = incgraph.closure(CHECKOUT, GGML_ROOTS, [u for f in FAMILIES for u in units(f) if u.startswith(G)])
    s = incgraph.closure(CHECKOUT, SERVER_ROOTS, [u for f in FAMILIES for u in units(f) if u.startswith('server/src')])
    allf = set()
    for d in (g, s):
        for v in d.values():
            allf.update(v)
    return sorted(f for f in allf if f in files), {**g, **s}


ALL, CLOSURE = closures()
OWNED = {}
for _f in FAMILIES:
    for _p in units(_f) + [h for h in F[_f]['headers'] if os.path.isfile(os.path.join(CHECKOUT, h))]:
        OWNED[_p] = _f
HDR_GGML = sorted(set([p for p in ALL if p not in OWNED and p.startswith((G, HIP_COMPAT_GGML))] +
                      ['server/deps/llama.cpp/LICENSE', 'server/deps/llama.cpp/VENDOR.md']))
HDR_SERVER = [p for p in ALL if p not in OWNED and not p.startswith((G, HIP_COMPAT_GGML))]


def fam_requires(f):
    """Header modules and other families whose files this family's units include."""
    req = ['module:kernel.lucebox.ggml_headers']
    if F[f]['server']:
        req.append('module:kernel.lucebox.server_headers')
    for u in units(f):
        for p in CLOSURE.get(u, []):
            o = OWNED.get(p)
            if o and o != f and 'module:kernel.lucebox.' + o not in req:
                req.append('module:kernel.lucebox.' + o)
    return req


def mods():
    out = [
        M('kernel.lucebox.ggml_headers',
          'the ggml side of the #include closure of every Lucebox ggml-cuda kernel unit: ggml public headers, ggml-impl and ggml-common, the ggml-cuda common, convert, mma, MMQ / MMVQ, fattn and vendor (hip.h) headers, with VENDOR.md and the vendored ggml LICENSE',
          HDR_GGML, compile='header-only', reason='header-only; compiled inside every kernel.lucebox.* family module',
          includes=['src/' + r for r in GGML_ROOTS], spdx=GGML_SPDX, authors=['The ggml authors', 'Luce-Org'],
          upstream='vendored ggml: Luce-Org/lucebox-ggml luce-dflash@6fbe72d plus PRs 35, 37, 39, llama.cpp 22298 and the hub-local patch set of lucebox 505 (server/deps/llama.cpp/VENDOR.md)',
          provides=[P('kernel-abi.lucebox-ggml/-/-', 'ggml/include/*.h, ggml/src/ggml-impl.h, ggml/src/ggml-common.h, ggml-cuda/common.cuh and vendors/hip.h', 'n/a', 'n/a')],
          why='Lucebox kernels are ggml ops: they cannot compile without these headers'),
        M('kernel.lucebox.server_headers',
          'the server side of the #include closure of the nine Lucebox server HIP units: server/include, the server/src headers they reach, and the server/hip_compat CUDA-to-HIP shim headers',
          HDR_SERVER, compile='header-only', reason='header-only; compiled inside the kernel.lucebox.* modules with server units',
          includes=['src/server/hip_compat', 'src/server/include', 'src/server/src', 'src/server/deps/llama.cpp/common'],
          provides=[P('kernel-abi.lucebox-server/-/-', 'server/hip_compat/*.h and the server/src C headers of the HIP units', 'n/a', 'n/a')],
          spdx=SERVER_SPDX, why='the server units include their C contracts and the hip_compat shim from here'),
    ]
    for f in FAMILIES:
        d = F[f]
        srcs = units(f)
        paths = srcs + [h for h in d['headers'] if os.path.isfile(os.path.join(CHECKOUT, h))]
        spdx = ' AND '.join(sorted({GGML_SPDX if p.startswith(G) else SERVER_SPDX for p in srcs}))
        sflags = {}
        for s in d['server']:
            sflags[s] = SERVER_FLAGS + d['sflags'].get(s, [])
        for s in srcs:
            if s.startswith(G):
                sflags[s] = GGML_FLAGS + d['sflags'].get(s, [])
        incs = sorted({'src/' + os.path.dirname(p) for p in paths})
        if any(s.startswith(G) for s in srcs):
            incs.append('src/' + G + '/src')     # ggml units include "ggml-cuda/<x>.cuh" from ggml/src
        tags = '; '.join(p['interface'] for p in d['provides'])
        notes = (f"INTERFACES.md section: {d['section']}. Interface tags: {tags}. Units: " + ', '.join(srcs) +
                 ". Licence: ggml-side files MIT (server/deps/llama.cpp/LICENSE, in kernel.lucebox.ggml_headers), server-side Apache-2.0 (LICENSE). "
                 "The ggml units are ggml backend ops (ggml_backend_cuda_context, ggml_tensor) dispatched from ggml-cuda.cu, which is not harvested; "
                 "a strixflerp recipe needs an adapter to call them. " + NOT_RUN)
        authors = ['Luce-Org', 'The ggml authors'] if any(p.startswith(G) for p in srcs) else ['Luce-Org']
        out.append(M('kernel.lucebox.' + f, d['summary'], paths, provides=d['provides'], requires=fam_requires(f),
                     sources=srcs, includes=incs, spdx=spdx, source_flags=sflags or None, notes=notes, why=d['why'],
                     authors=authors))
    return out


ORIGIN_TOML = '''
# Lucebox (lucebox-hub): an inference engine on a vendored ggml fork for DeepSeek V4 / V4.1 Flash, Qwen3.8-Flash-Next,
# Inkling2 and others on gfx1151 / gfx1201, with DFlash2 / MTP / DSpark speculation, hot experts resident and the rest
# streamed from SSD, KVFlash and PFlash (card ORIGIN-LUCEBOX-HARVEST 0bd0e0db19; report /data/pod/reports/2026-11/origin-lucebox.md).
name      = "lucebox"
repo      = "https://github.com/Luce-Org/lucebox"
commit    = "a70ab3ca25a945d97580043230dc33984aaf1fb2"
branch    = "main"
version   = "head of 2026-11-23 (docs(readme): add Inkling 2 Mini to head-to-head benchmark chart (#1069)); vendored ggml per server/deps/llama.cpp/VENDOR.md: lucebox-ggml luce-dflash@6fbe72d + PRs 35, 37, 39, llama.cpp 22298, lucebox 505"
reference_checkout = "control-node:/data/ark-inbox/src/github.com__Luce-Org__lucebox (ark copy, a depth-1 clone at a70ab3c); laptop: ~/Code/strixflerp-upstream-supplement/lucebox (git archive of that commit, LFS smudging off)"

[licence]
spdx      = "Apache-2.0"
files     = ["LICENSE"]                                     # copied verbatim next to this file; server/deps/llama.cpp/LICENSE (MIT) is copied beside it as LICENSE-ggml
third_party = "The vendored ggml (server/deps/llama.cpp) is MIT, The ggml authors (server/deps/llama.cpp/LICENSE, copied here as LICENSE-ggml and carried in kernel.lucebox.ggml_headers); Lucebox's own ggml-cuda kernels sit under that file and carry no per-file licence header. server/ is Apache-2.0 (LICENSE). vision-rotary.cu follows the arithmetic of PyTorch 3d3aa83 aten/src/ATen/native/cuda (BSD-3-Clause upstream) by its header. Block-Sparse-Attention (submodule, Apache-2.0) is not in the archive and not harvested."

[toolchain]
host      = "C++17, CMake 3.25 or newer (ROCm 7.2's CMake files), Ninja; nlohmann/json 9cca280 by FetchContent"
device    = "therock-10.2"                               # the origin builds on ROCm 7.2.4 (Dockerfile.rocm); control-node's /opt/rocm-7.2.4 has no rocwmma, hipcub or hipblas headers, therock-10.2 has them
build     = "Dockerfile.rocm: cmake -S server -DLUCE_GPU_BACKEND=hip -DLUCE_HIP_ARCHITECTURES=gfx1151 -DLUCE_FA_ALL_QUANTS=OFF -DLUCE_ENABLE_BSA=OFF, targets test_dflash luce_server test_server_unit; ggml-hip globs ggml-cuda/*.cu as LANGUAGE HIP; server HIP units by set_source_files_properties(LANGUAGE HIP) in server/CMakeLists.txt:806-1036"

[compile]                                                # CMake's HIP language: clang++ -x hip, one object per translation unit
driver    = "<rocm>/lib/llvm/bin/clang++"
arch_flag = "--offload-arch=gfx1151"
rocm_flag = "--rocm-path=<rocm>"
flags     = ["-x", "hip", "-std=c++17", "-O3", "-DNDEBUG", "-DGGML_USE_HIP", "-DGGML_SCHED_MAX_COPIES=4", "-D_XOPEN_SOURCE=600", "-D_GNU_SOURCE", "-DGGML_BUILD", "-DGGML_SHARED", "-DGGML_BACKEND_BUILD", "-DGGML_BACKEND_SHARED", "-DGGML_HIP_DS4V_VISION", "-DGGML_CUDA_PEER_MAX_BATCH_SIZE=128", "-c"]
system_includes = ["include"]
output    = "host object with the gfx1151 device code embedded (.o), one per translation unit"
note      = "Defines from ggml/src/CMakeLists.txt (GGML_SCHED_MAX_COPIES, _XOPEN_SOURCE, _GNU_SOURCE, GGML_BUILD / GGML_SHARED, GGML_BACKEND_BUILD / GGML_BACKEND_SHARED), ggml-hip/CMakeLists.txt (GGML_USE_HIP, GGML_HIP_DS4V_VISION with hipBLASLt present) and ggml-cuda/CMakeLists.txt (GGML_CUDA_PEER_MAX_BATCH_SIZE=128, its default), Release -O3 -DNDEBUG. ggml units add the four CUDA-spelling defines server/CMakeLists.txt:510-515 gives ggml-hip and see server/src/hip_compat first (kernel.lucebox.ggml_headers); server units add LUCE_BACKEND_HIP=1, LUCE_HAVE_DRAFT_TOPK=1, LUCE_HAVE_KVFLASH_FUSED_MASS=1 and per-unit LUCE_HAVE_FLASHPREFILL / LUCE_HAVE_BSA as server/CMakeLists.txt sets them ([build].source_options)."

[kernels]
loading   = "linked: the ggml units into libggml-hip.so (ggml_backend_cuda_* ops dispatched from ggml-cuda.cu by GGML op), the server units into luce_common; no code-object kit"
selection = "ggml op dispatch in ggml-cuda.cu and the model graphs of server/src (deepseek4, qwen35, qwen4exp, inkling2, ...); environment switches such as LUCE_PLE_FUSE_LEGACY and LUCE_FP_USE_BSA"
models    = "DeepSeek V4 / V4.1 Flash (+ DS4V vision), Qwen3.5 / 3.8 (Flash-Next), qwen4exp, Inkling2-Loop with DSpark, gemma4, laguna, bailingmoe3 (server/src). Claimed: DSV4.1 Flash 25.4 tok/s on R9700 + Ryzen AI MAX+ 128 GB, DSV4 Flash 27.9 tok/s on MAX+ alone, Flash-Next 84.8 tok/s on R9700 + MAX+ (README)"
'''

LEFT_OUT = {
    G + ' (the rest)': 'the ggml base: upstream ggml-cuda kernels (70 of them with Lucebox edits: mmvq, mmid, mmq, fattn, rope, norm, cpy, ggml-cuda.cu, ...), template instances, the CPU and other backends; Lucebox edits cannot be told from upstream drift until lucebox-ggml luce-dflash@6fbe72d is in the ark',
    CU + '/{fattn-wmma-f16,diagmask,copy-batch}.cu': 'older upstream ggml files that upstream has since removed or renamed (batch-copy.cu), not Lucebox kernels',
    'server/src (host code)': 'the engine: model graphs, scheduler, KV cache, server, placement and SSD streaming; C++ host code, not kernels',
    'server/src/moss_image, server/src/{flashprefill_f16,flashprefill_kernels,flashprefill_scalar,bsa_fwd_inst,bsa_launcher}.cu': 'CUDA-only units (image generation, sm_60 to sm_80 prefill, CUTLASS BSA)',
    'server/deps/Block-Sparse-Attention/': 'git submodule (mit-han-lab, 49d6c39, CUDA / CUTLASS): not in the archive; the HIP build replaces it with bsa_launcher_hip.cu',
    'optimizations/': 'kvflash, pflash, paged_attention, spark, image_gen hold no kernel source; megakernel is CUDA only',
    'server/test, conftest.py, harness/, scripts/': 'tests, benches and scripts',
    'docs/, assets/, README.md, CONTRIBUTING.md, variables.md, share/': 'documentation and assets (LFS pointers in the archive)',
    'Dockerfile*, docker-bake.hcl, Makefile, pyproject.toml, uv.lock, server/CMakeLists.txt, server/cmake': 'build and packaging; the compile flags are recorded in ORIGIN.toml [compile]',
}


def kernels_md():
    bt = chr(96)
    q = lambda s: bt + s + bt
    o = ['# Lucebox HIP kernel families', '',
         'Generated by ' + q('docs/harvest/harvest_lucebox.py') + ' (card ORIGIN-LUCEBOX-HARVEST 0bd0e0db19) from the family table of',
         q('/data/pod/reports/2026-11/origin-lucebox.md') + '. Paths are origin-relative; G = ' + q(CU) + '.', '',
         '| module | units | INTERFACES.md section | interfaces |', '|---|---|---|---|']
    nu = 0
    for f in FAMILIES:
        u = units(f)
        nu += len(u)
        o.append('| %s | %s | %s | %s |' % (q('kernel.lucebox.' + f), ', '.join(q(x.replace(CU + '/', 'G/')) for x in u),
                 F[f]['section'], ', '.join(q(p['interface']) for p in F[f]['provides'])))
    o += ['', '%d families, %d translation units (%d ggml-side, %d server-side). Header-only: %s (%d files), %s (%d files).'
          % (len(FAMILIES), nu, sum(len(F[f]['ggml']) for f in FAMILIES), sum(len(F[f]['server']) for f in FAMILIES),
             q('kernel.lucebox.ggml_headers'), len(HDR_GGML), q('kernel.lucebox.server_headers'), len(HDR_SERVER)),
          '', '## Not done here', '', NOT_RUN + ' Every kernel is a ggml op or a server C entry point; a swap into a strixflerp recipe needs an adapter.']
    open(os.path.join(harvestlib.REPO, 'catalog', O.name, 'KERNELS.md'), 'w').write('\n'.join(o) + '\n')


if __name__ == '__main__':
    run(O, mods(), ORIGIN_TOML, LEFT_OUT, ['LICENSE'])
    # The ark clone has the Block-Sparse-Attention submodule checked out (CUDA / CUTLASS); git archive leaves it out.
    ot = os.path.join(harvestlib.REPO, 'catalog', O.name, 'ORIGIN.toml')
    s = open(ot).read()
    assert 'unharvested = [\n' in s
    open(ot, 'w').write(s.replace('unharvested = [\n', 'unharvested = [\n  "server/deps/Block-Sparse-Attention/**",\n', 1))
    shutil.copyfile(O.find('server/deps/llama.cpp/LICENSE'), os.path.join(harvestlib.REPO, 'catalog', O.name, 'LICENSE-ggml'))
    kernels_md()
