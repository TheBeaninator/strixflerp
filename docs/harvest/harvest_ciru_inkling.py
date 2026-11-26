#!/usr/bin/env python3
"""Harvest Ciru's Inkling 2 Mini CIRU Halo Agent (ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151, b3870f35926f,
release v1.5.0) into catalog/ciru_inkling.
Run: python3 docs/harvest/harvest_ciru_inkling.py

Card ORIGIN-CIRU-INKLING 8bd576841e (ark intake 7546d7e, owner 2026-11-24).
Scope: the native gfx1151 HIP engine of the release, overlay/vllm/models/inkling2/amd/ops: seven .hip translation units
and their shared header hip_common.cuh, one kernel module a translation unit plus a header-only module. The rest of
the release (the vLLM overlay and patch, the Triton/Gluon fallbacks and the ctypes loader in Python, the install and
serve scripts, the bundled ROCr library and the prebuilt kernel library) is listed as unharvested.

The engine builds the seven units into one shared object with one hipcc line (ops/hip_runtime.py: hipcc -shared -fPIC
-O3 --offload-arch=gfx1151 -DINKLING_HIP_ABI_VERSION=3 -DINKLING_HIP_SRC_HASH_RAW=<hash>) and calls it through ctypes:
every kernel is reached by an extern "C" launch_* function on raw device pointers, strides and a stream. No unit
includes a PyTorch header, so each compiles on its own; the compile gate compiles each to one object with the same
flags and -c in place of -shared. The 12-hex source hash of the origin (sha256 over the eight file names and bytes,
ops/hip_runtime.py compute_hip_src_hash) is recomputed here and must equal the name of the prebuilt library the
release ships (scripts/rocm/hip_inkling_gfx1151_671b45b80972.so).

Reference checkout: the control-node ark copy /data/ark-inbox/src/github.com__ciru-ai__Inkling-2-Mini-CIRU-Halo-Agent-gfx1151#<commit>
(a depth-1 clone whose .git has no refs directory, so the commit is read from .git/HEAD), else the laptop copy
~/Code/strixflerp-upstream-supplement/ciru_inkling (rsync of that ark copy, 2026-11-24). CIRU_INKLING_CHECKOUT names
another copy. The script refuses a checkout whose .git/HEAD is not COMMIT or whose files differ from the release's
MANIFEST.sha256."""
import hashlib, os, re, sys, tomllib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvestlib
from harvestlib import Origin, M, P, run

COMMIT = 'b3870f35926fb322f398e6c373a93098534f8fdb'
POD_ARK = '/data/ark-inbox/src/github.com__ciru-ai__Inkling-2-Mini-CIRU-Halo-Agent-gfx1151#' + COMMIT
CHECKOUT = os.environ.get('CIRU_INKLING_CHECKOUT') or (
    POD_ARK if os.path.isdir(POD_ARK) else os.path.join(harvestlib.SUPPLEMENT, 'ciru_inkling'))
OPS = 'overlay/vllm/models/inkling2/amd/ops/'
TUS = ('hip_moe_prefill.hip', 'hip_attn_prefill.hip', 'hip_glue_prefill.hip', 'hip_attn_decode.hip',
       'hip_quant_decode.hip', 'hip_moe_decode.hip', 'hip_glue_decode.hip')     # ops/hip_runtime.py _HIP_TU_FILES
HASH_FILES = ('hip_common.cuh',) + TUS                                          # _HIP_SOURCE_FILES, in that order
SRC_HASH = '671b45b80972'
PREBUILT = 'scripts/rocm/hip_inkling_gfx1151_%s.so' % SRC_HASH


class ArkOrigin(Origin):
    """The file set is the checkout without .git; the commit comes from .git/HEAD and every file the release's
    MANIFEST.sha256 names must hash as listed."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.roots = [CHECKOUT]
        head = open(os.path.join(CHECKOUT, '.git', 'HEAD')).read().strip()
        if head != COMMIT:
            raise SystemExit(f'{CHECKOUT}: .git/HEAD is {head}, the harvest pins {COMMIT}')
        bad = []
        for line in open(os.path.join(CHECKOUT, 'MANIFEST.sha256')):
            if line.strip():
                digest, name = line.split(None, 1)
                if harvestlib.sha256(os.path.join(CHECKOUT, name.strip())) != digest:
                    bad.append(name.strip())
        if bad:
            raise SystemExit(f'{CHECKOUT}: differs from MANIFEST.sha256: ' + ' '.join(bad))

    def files(self):
        if self._files is None:
            out = []
            for d, dirs, fs in os.walk(CHECKOUT):
                if d == CHECKOUT:
                    dirs[:] = [x for x in dirs if x != '.git']
                out += [os.path.relpath(os.path.join(d, f), CHECKOUT) for f in fs]
            self._files = sorted(out)
        return self._files


O = ArkOrigin('ciru_inkling', 'ciru_inkling', 'https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151',
              COMMIT, 'main', 'Apache-2.0', ['Ciru (CIRU AI)'], 'therock-10.0')

h = hashlib.sha256()
for n in HASH_FILES:
    h.update(n.encode()); h.update(b'\0'); h.update(open(O.find(OPS + n), 'rb').read()); h.update(b'\0')
if h.hexdigest()[:12] != SRC_HASH:
    raise SystemExit(f'source hash {h.hexdigest()[:12]} is not {SRC_HASH}, the prebuilt library the release ships')
PREBUILT_SHA = harvestlib.sha256(O.find(PREBUILT))
PREBUILT_BYTES = os.path.getsize(O.find(PREBUILT))

_res = harvestlib.load_compile_results()
_runs = [_res['kernel.ciru_inkling.' + t[4:-4]] for t in TUS if 'kernel.ciru_inkling.' + t[4:-4] in _res]
GATE = ('Compile gate: %d of %d units build (%s on %s, %s; gate/results/compile/kernel.ciru_inkling.<unit>.json)' % (
    sum(r['status'] == 'pass' for r in _runs), len(TUS), _runs[0]['toolchain'], _runs[0]['host'], min(r['date'] for r in _runs)[:10])
    if _runs else 'Compile gate: not run yet')

FLAGS = ['-fPIC', '-O3', '-DINKLING_HIP_ABI_VERSION=3', '-DINKLING_HIP_SRC_HASH_RAW=' + SRC_HASH, '-c']
FORMATS = 'nvfp4+int4-g32+ciru-c1'
BANKS = ('four expert banks chosen by a mode argument (BANK_MODE 0 source: the checkpoint\'s NVFP4, E2M1 nibbles with positive '
         'E4M3 group-16 scales and an FP32 outer scale an expert; 10 mixed1: per-expert kind 0 (NVFP4) or 1 (signed INT4 G32, BF16 '
         'scales); 20 mixed2: kind 0 or 2 (Flash C1: 4-bit codes with group tags, FP16 row multipliers and FP16 [256, 2] affine '
         'tables an expert); 2 coda: C1 only)')
NOT_TIMED = 'No timing was taken on the pod; figures are the author\'s (README.md, v1.5.0).'


def K(entries, tag, beside, what, numeric=None, regime=None):
    """One group of launch functions of a unit. tag: the strixflerp interface it matches, as a full interface name
    (the class is the name without its .tag), or None. beside: catalogue kernel modules of other origins that provide
    the same <family>.<op>; kernels_md() checks that each exists and provides that op."""
    return dict(entries=entries, tag=tag, beside=beside, what=what, numeric=numeric, regime=regime)


def bank_names(kind):
    return ['launch_%s_%s' % (b, kind) for b in ('source', 'mixed1', 'mixed2', 'coda')]


UNITS = {
 'moe_prefill': [
  K([n + s for n in bank_names('w13') for s in ('_f16_xbf16', '_bf16_xbf16', '_f16', '_bf16')] + bank_names('w2'),
    'moe.experts-prefill/%s/bf16.ciru-inkling' % FORMATS,
    ['kernel.strixite.moe_grouped', 'kernel.crucible.moe_prefill', 'kernel.hipfire.moe', 'kernel.radiance.halo_moe'],
    'one templated kernel (moe_wmma_k32_unified_kernel) for both expert projections of prefill over a sorted route list: '
    'a block owns a 128-row (paired) or 64-row (tail) block of routes of one expert and 128 (w13) or 256 (w2, N % 256 == 0) '
    'output columns, K in steps of 32 on the f16 WMMA, weights unpacked from the packed bank in registers; 16-row fragments '
    'that are all padding are skipped; w13 fuses the softcapped SwiGLU (gate_cap, up_cap)',
    'activations FP16 or BF16 (the _xbf16 entries read BF16 X and convert when staging); w13 output FP16 or BF16, w2 output BF16; '
    'weights from ' + BANKS,
    'prefill; 8 routes a token (Inkling 2 Mini: 128 experts, top 8, hidden 2304); N % 128 == 0'),
  K(['launch_pair_m64'], None, [],
    'one-warp planner that scans the 64-row block expert ids of the route plan into unified descriptors (128-row pairs of '
    'adjacent blocks of one expert, 64-row tails) and their counts, on the device (no host sync)'),
  K(['launch_hip_moe_reduce'], 'moe.combine/-/bf16.ciru-inkling-slot-order',
    ['kernel.gufo.qfn_kernels', 'kernel.hipfire.moe', 'kernel.radiance.halo_moe'],
    'sums the 8 routed rows of each token into its output row',
    'FP32 sum in slot order 0..7, rounded once to BF16 (RTNE): deterministic, bit-exact by the file header',
    'prefill and decode (lowered to M >= 1 in v1.5.0); 288 threads a row for N <= 2304'),
  K(['inkling_hip_abi_version', 'inkling_hip_src_hash'], None, [],
    'ABI metadata the loader checks before it uses the library: INKLING_HIP_ABI_VERSION (3) and the source hash the '
    'library was compiled from (INKLING_HIP_SRC_HASH_RAW)'),
 ],
 'attn_prefill': [
  K(['launch_hip_kv_conv_1w'], 'kvcache.store/kv-bf16/bf16.ciru-inkling-conv',
    ['kernel.gufo.ds4_attention', 'kernel.hipfire.kv_cache', 'kernel.strixfold.triton_kit'],
    'Inkling\'s K/V short convolution fused with the paged cache write: one warp a (token tile, KV head) applies the width-4 '
    'causal depthwise conv to K and V (taps from the conv-state cache across request boundaries), scales and clips, and '
    'writes the results to the paged K/V cache and the new inputs to the conv-state cache',
    'BF16 in and out, FP32 conv math; head_dim 128, 32 lanes x 4 BF16; static or dynamic (per-token) conv weights',
    'prefill; any KV block size; positions int32 or int64'),
  K(['launch_hip_qr_clip'], None, [],
    'exact per-head, per-channel L2 clip of Q and of the relative-position query R before the post-8K temperature (tau), '
    '16 lanes a head'),
  K(['launch_hip_gather_paged_kv_hnd'], None, [],
    'copies one request\'s pages of the paged K/V cache into contiguous [KVH, N, D] arrays, the long-context far part of '
    'hybrid prefill'),
  K(['launch_hip_flash_rel_attn_prefill', 'launch_hip_flash_rel_attn_near_merge'],
    'attn.prefill-causal/kv-bf16/bf16.ciru-inkling-rel',
    ['kernel.flerp.attn_wmma_gqa', 'kernel.radiance.halo_attn'],
    'causal flash attention over the paged cache with Inkling\'s learned relative-position bias: a block owns 16 query tokens '
    'of one head and computes K Q^T and V^T P^T on the BF16 WMMA (12.5 KiB LDS, 2 barriers a 16-key tile); keys farther '
    'than EXTENT take no bias (far phase), nearer keys take R . RelProjT[distance] (near phase); sliding window for local '
    'layers; a fused output gate and attention-sink epilogue; near_merge runs the near part and merges a far partial '
    '(OutFar, LseFar) computed over gathered pages',
    'BF16 Q, K, V, online softmax in FP32 with exp2 (scale_log2); R in BF16 or FP16 (r_fp16); optional BF16 rounding of the bias',
    'prefill; head_dim 128, GQA groups of num_heads / 8 KV heads, any page size'),
 ],
 'glue_prefill': [
  K(['launch_q4_coeffs'], None, [],
    'coefficients of the 4-query block-attention-residual message (D = 2304, one stacked block) without materialising the mix'),
  K(['launch_hip_sconv_prefill'], None, [],
    'Inkling\'s static short convolution of the residual streams for prefill (BT = 64 tokens, width 4, 8 channels a thread) '
    'over a paged sliding-window conv-state cache, safe at request and tile boundaries, with optional addend and residual '
    'inputs and FP32 or BF16 outputs'),
  K(['launch_hip_loop_zeromat_prefill'], None, [],
    'Inkling\'s loop controller for prefill, three passes a 16-token tile: down projection [16, 2304] x [2304, 64], up '
    'projection and the gain / forget blend of the destination stream with the message, then RMSNorm times the stream\'s '
    'norm weight; nearest relative in the catalogue: the hyper-connection mixes (hc.mix-*), which this is not'),
  K(['launch_hip_attn_res_prefill'], None, [],
    'Inkling\'s block attention residuals: each token mixes the stacked earlier block outputs and the last one with a '
    'softmax over query . normalised value (D = 2304, 288 threads x 8 elements; one-pass register cache up to 12 stacked '
    'blocks, two passes with an LDS softmax buffer above that), optional addend and RMSNorm of the result'),
  K(['launch_hip_router_prefill'], 'moe.gate-logits/bf16/bf16-f32.ciru-inkling',
    ['kernel.flerp.moe_kquant', 'kernel.strixfold.moe_gate'],
    'router GEMM for prefill on the BF16 WMMA, X read once a 128-column block, two outputs: FP32 scores and the '
    'bias-corrected scores the top-k reads',
    'BF16 X and router weight, FP32 accumulate and outputs',
    'prefill; 128 experts, hidden 2304'),
 ],
 'attn_decode': [
  K(['launch_hip_kv_conv_decode'], 'kvcache.store/kv-bf16/bf16.ciru-inkling-conv-decode',
    ['kernel.gufo.ds4_attention', 'kernel.hipfire.kv_cache', 'kernel.strixfold.triton_kit'],
    'the K/V short convolution and paged cache write for decode and speculative verify: one warp a (KV head, request), no '
    'LDS and no barriers, int64 cache addressing',
    'BF16 in and out, FP32 conv math; head_dim 128',
    'decode, several query tokens a request (cu_seqlens_q); positions int32 or int64'),
  K(['launch_hip_flash_rel_decode_split', 'launch_hip_decode_combine'],
    'attn.softmax-gqa/kv-bf16/bf16.ciru-inkling-rel-split',
    ['kernel.strixite.attention', 'kernel.gufo.qwen_attention', 'kernel.hipfire.attention', 'kernel.radiance.halo_attn'],
    'split-KV flash decode attention with the relative-position bias, for target layers (with bias, tau after 8K) and the '
    'DSpark draft layers: a block owns (split, KV head, request) and the query rows of that KV head, a branchless far loop '
    'without bias over most keys, then a near / local / tail loop with the bias and exact masks; a cooperative BN = 32 '
    'variant packs 32 query rows (B = 1, M = 6..8) so K and V are read once; decode_combine merges the splits and applies '
    'the output gate and the sink',
    'BF16 Q / K / V on the BF16 WMMA, FP32 partials and log-sum-exp per split; int64 paged addressing up to 1,048,576 tokens',
    'decode and speculative verify; head_dim 128, 8 KV heads; author figure: 227.7 GB/s at 128K context (README.md:31)'),
 ],
 'quant_decode': [
  K(['launch_hip_w6_wmma'], 'linear.decode/ciru-w6-g64/bf16.ciru-inkling',
    ['kernel.strixite.linear_q56', 'kernel.hipfire.gemv_hfq', 'kernel.gufo.qwen_linear_decode'],
    'single-pass W6 linear for small M on the f16 WMMA: a block owns 64 output rows (4 warps x 16) and an M tile of 16, 32 '
    'or 48 rows, weights streamed as a 4-bit plane (P4, 16-byte loads) and a 2-bit plane (P2, 8-byte loads) with one scale '
    'a row and 64-wide K step, software-pipelined; optional split-K; a QKV form reads separate Q, K and V inputs for its '
    'three output ranges; optional softcap and FP32 output (the LM head)',
    'BF16 activations, signed 6-bit weights with a BF16 scale a (row, K64 group), FP32 accumulate; BF16 or FP32 out',
    'decode and verify, M <= 48 a launch (w6_qkv, w6_wo, w6_head); author figure: the B = 8, M = 40 W6/W4 linear step '
    '9.85 ms -> 6.32 ms (README.md:32)'),
  K(['launch_hip_w4_wmma'], 'linear.decode/ciru-w4/bf16.ciru-inkling',
    ['kernel.strixite.linear_q4', 'kernel.flerp.linear_q4p', 'kernel.hipfire.gemv_mq'],
    'the same single-pass small-M WMMA linear on packed 4-bit weights with a scale a (row, K64 group), optional bias and '
    'split-K',
    'BF16 activations, FP32 accumulate, BF16 or FP32 out',
    'decode and verify, M <= 48 a launch'),
  K(['launch_hip_reduce_splits'], None, [],
    'split-K reducer shared by the W6, W4 and DSpark W8 kernels, with the optional softcap and FP32 output'),
  K(['launch_hip_w6_head_gemv_m1'], 'logits.head/ciru-w6-g32+g64/bf16-f32.ciru-inkling',
    ['kernel.radiance.halo_gemm_fp8'],
    'M = 1 GEMV for the W6 head: the whole weight record set of a row prefetched (up to 6 loads), v_dot2_f32_bf16, group '
    'scales of 32 or 64 weights, softcap epilogue',
    'BF16 activations, signed 6-bit weights in 16-weight records (one 64-bit and one 32-bit word), FP32 dot, BF16 or FP32 logits',
    'single-sequence decode, widths 2304 / 3072'),
  K(['launch_hip_w4_markov_argmax'], 'draft.markov/ciru-w4/bf16.ciru-inkling-argmax',
    ['kernel.gufo.ds4_indexer', 'kernel.strixfold.triton_kit'],
    'the DSpark draft\'s rank-256 Markov fold fused with greedy selection: W4 weights against the latent, added to the base '
    'logits and bias, softcap, and an argmax with an exact tie-break, 8 lanes a row and 16-byte loads',
    'BF16 latent and base logits, FP32 dot (v_dot2), softcap then argmax with an exact tie-break',
    'B = 1 draft step (the fused greedy fast path); author figure 248.0 GB/s (README.md:32)'),
 ],
 'moe_decode': [
  K(['launch_hip_moe_decode_plan'], None, [],
    'one-warp greedy planner that pairs adjacent 16-row route blocks of one expert into 32-row blocks on the device '
    '(no host sync, safe inside a captured HIP graph)'),
  K(['launch_hip_moe_decode_unified'], 'moe.experts-decode/%s/bf16.ciru-inkling-m16' % FORMATS,
    ['kernel.strixite.linear_q4', 'kernel.crucible.moe_decode', 'kernel.hipfire.moe', 'kernel.radiance.halo_moe'],
    'single pass a expert for decode and verify batches: a block owns one 16- or 32-row route block and a 128-row tile of '
    'the bank, weights unpacked from registers (no LDS copy of B) on a K64 software pipeline and reused for both 16-row '
    'halves of a paired block; FUSED_ACT 1 is w13 with the softcapped SwiGLU, 0 is w2',
    'X in FP16 or BF16 (converted when staged), w13 output FP16 or BF16, w2 output BF16; weights from ' + BANKS,
    'decode with up to 8 sequences and draft tokens (the B = 8 multi-stream path); author figures: about 220-225 GB/s, '
    'B = 8 throughput +13.0% (8 x 256 tokens, 194.1 tok/s) and +17.4-18.7% (8 x 512, 205.5-207.8 tok/s) (README.md:33)'),
 ],
 'glue_decode': [
  K(['launch_hip_sconv_decode'], None, [],
    'the static short convolution of the residual streams for decode (8 tokens a block, width 4, one channel a thread), '
    'aliasing-safe, with the prefill unit\'s addend / residual options'),
  K(['launch_hip_loop_decode_down', 'launch_hip_loop_decode_blend_rms', 'launch_hip_loop_decode_blend_rms_2pass'], None, [],
    'the loop controller for decode: down projection of the RMS-normalised (destination, message) pair, then the up '
    'projection, gain / forget blend and RMSNorm in one pass or two (one read of each weight)'),
  K(['launch_hip_pair_rmsnorm', 'launch_hip_blend_rmsnorm', 'launch_hip_mix_rmsnorm', 'launch_hip_rms_norm'],
    'norm.rmsnorm/-/bf16.ciru-inkling',
    ['kernel.gufo.qwen_norm_rope', 'kernel.hipfire.norm_rope', 'kernel.strixfold.aux'],
    'RMSNorm forms: pair (destination and message normalised side by side, the loop controller\'s input), blend (gain / forget '
    'blend then RMSNorm), mix (the DSpark draft\'s weighted sum of target hidden-state taps then RMSNorm) and a plain row RMSNorm',
    'FP32 math, BF16 or FP32 in and out (in_f32 / out_f32 / nw_f32 switches)',
    'decode rows; hidden 2304'),
  K(['launch_hip_router_gemm_m8'], 'moe.gate-logits/bf16/bf16-f32.ciru-inkling-m8',
    ['kernel.flerp.moe_kquant', 'kernel.strixfold.moe_gate'],
    'router GEMM for M <= 8 rows, optionally with the sigmoid and bias correction fused',
    'BF16 X and weight, FP32 logits', 'decode; 128 experts'),
  K(['launch_hip_route_m8'], 'moe.router/-/f32.ciru-inkling-m8',
    ['kernel.strixite.moe_router', 'kernel.crucible.ops', 'kernel.hipfire.moe'],
    'sigmoid, bias-corrected stable top-8 of 128 experts with 256 threads, gamma weights, and optionally the expert-sorted '
    'route list and padded 16-row blocks the decode MoE plan reads',
    'FP32; bit-exact IEEE-754 selection order by the file header', 'decode, M <= 8 tokens'),
  K(['launch_hip_qk_norm_rope'], 'pos.rope-heads/-/bf16.ciru-inkling-qk-norm',
    ['kernel.gufo.ds4_norm_rope', 'kernel.strixfold.rope_heads'],
    'in-place per-head RMSNorm of Q and K (optional) and rotary embedding from a cos / sin cache by position, the DSpark '
    'draft layers\' attention prologue',
    'BF16 in place, FP32 math', 'draft decode'),
  K(['launch_hip_dspark_w8'], 'linear.decode/i8-g128/bf16.ciru-inkling-dspark',
    ['kernel.strixite.linear_q8', 'kernel.flerp.linear_q8p', 'kernel.gufo.qfn_kernels'],
    'the DSpark draft projections on signed W8-G128 weights in N64 / K64 tiles, M = 8 rows on the WMMA, optional split-K '
    '(reduced by launch_hip_reduce_splits)',
    'BF16 activations, int8 weights with a scale a 128-wide group, FP32 accumulate, BF16 out', 'draft decode, M = 8'),
  K(['launch_hip_silu_and_mul'], 'act.swiglu/-/bf16.ciru-inkling',
    ['kernel.strixite.swiglu', 'kernel.gufo.qfn_kernels', 'kernel.strixfold.dense_bf16'],
    'SiLU(gate) * up over [rows, 2N] rows', 'BF16, FP32 math', 'draft decode'),
  K(['launch_hip_add_softcap'], None, [],
    'adds two row-strided arrays and applies the softcap, the draft logits epilogue'),
  K(['launch_hip_markov_gate'], 'draft.markov/-/bf16.ciru-inkling-gate',
    ['kernel.gufo.ds4_indexer', 'kernel.strixfold.triton_kit'],
    'the DSpark Markov gate: combines the raw draft projection with the carried state into the gated output and the next '
    'rank-sized latent',
    'BF16, FP32 math', 'draft decode'),
 ],
}

SUMMARY = {
 'moe_prefill': 'Inkling 2 Mini MoE prefill (hip_moe_prefill.hip): one WMMA kernel for both expert projections over the NVFP4 source bank and Ciru\'s fitted INT4 / C1 banks, its 64-row block planner and the slot-order route reduction; also the ABI metadata of the library',
 'attn_prefill': 'Inkling 2 Mini attention prefill (hip_attn_prefill.hip): K/V short convolution with the paged cache write, Q / R clip, paged K/V gather, and BF16-WMMA flash attention with the learned relative-position bias, window, output gate and sink',
 'glue_prefill': 'Inkling 2 Mini prefill glue (hip_glue_prefill.hip): static short convolution of the residual streams, the loop controller, block attention residuals with their message coefficients, and the two-output router GEMM',
 'attn_decode': 'Inkling 2 Mini decode attention (hip_attn_decode.hip): K/V short convolution with the paged cache write and split-KV flash decode attention with the relative-position bias, a 32-row cooperative variant and the gate / sink combine, int64 paged addressing to 1M tokens',
 'quant_decode': 'Inkling 2 Mini quantized decode linears (hip_quant_decode.hip): single-pass small-M W6 and W4 WMMA linears, their split-K reducer, the M = 1 W6 head GEMV and the W4 Markov fold with greedy argmax',
 'moe_decode': 'Inkling 2 Mini MoE decode (hip_moe_decode.hip): the 16-to-32-row route block planner and one single-pass-per-expert WMMA kernel over all four expert banks, the multi-stream (B = 8) decode path',
 'glue_decode': 'Inkling 2 Mini decode glue and DSpark draft kernels (hip_glue_decode.hip): short convolution, loop controller, RMSNorm forms, M <= 8 router GEMM and top-8 route, and the draft\'s QK norm + RoPE, W8 projections, SiLU-and-mul, softcap add and Markov gate',
}

WHY = {
 'moe_decode': 'the card names the 8-stream decode kernels first: this unit carries the author\'s 194-208 tok/s B = 8 decode (README.md:19, 33) and bears on the M = 8 gap of WRENCH-M8 e9d0b002da and LIVE-KTRACE 2007c996cf: one pass a expert over 16 / 32-row route blocks, weights unpacked in registers and shared by both halves of a paired block',
 'quant_decode': 'the small-M (16 / 32 / 48 rows) single-pass W6 / W4 WMMA linears are the dense half of the same B = 8 decode step (9.85 -> 6.32 ms by the author, README.md:32), the other multi-slot shape the card asks for',
 'attn_decode': 'a split-KV decode attention with a 32-row cooperative variant for B = 1, M = 6..8 verify batches, the third kernel of the multi-slot decode step, with int64 paged addressing to 1M tokens',
}


def tag_class(tag):
    head, last = tag.rsplit('/', 1)
    return head + '/' + last.split('.')[0]


COMMON = 'module:kernel.ciru_inkling.common'
mods = []
mods.append(M('kernel.ciru_inkling.common',
    'Header-only: hip_common.cuh, the shared device helpers of the seven Inkling 2 Mini HIP units (BF16 <-> FP32 with RTNE, '
    'WMMA fragment vector types, warp reductions) and the ABI version and source hash macros',
    [OPS + 'hip_common.cuh'], includes=['src/' + OPS.rstrip('/')], compile='header-only',
    provides=[P('kernel-abi.ciru-inkling-common/-/-', OPS + 'hip_common.cuh (INKLING_HIP_ABI_VERSION, INKLING_HIP_SRC_HASH, half16 / short16 / float8 vector types, bf16 helpers)', 'n/a', 'n/a')],
    notes='Interface tag: none. Every unit includes it by a quoted include; the ABI version (3) and the source hash are compile-time macros the loader (ops/hip_runtime.py, not harvested) reads back through inkling_hip_abi_version / inkling_hip_src_hash.',
    reason='header-only; compiled inside every kernel.ciru_inkling.* unit'))

for name, rows in UNITS.items():
    tu = OPS + 'hip_%s.hip' % name
    provides = [P(r['tag'], ', '.join(r['entries']) + ' (extern "C", ' + tu.rsplit('/', 1)[1] + ')', r['numeric'], r['regime'])
                for r in rows if r['tag']]
    tags = '; '.join('%s: %s' % (r['entries'][0] + (' and %d more' % (len(r['entries']) - 1) if len(r['entries']) > 1 else ''),
                                 tag_class(r['tag']) if r['tag'] else 'none') for r in rows)
    mods.append(M('kernel.ciru_inkling.' + name, SUMMARY[name], [tu], provides=provides, sources=[tu], includes=['src/' + OPS.rstrip('/')],
                  flags=FLAGS, requires=[COMMON], why=WHY.get(name),
                  notes='Interface tags by launch group: ' + tags + ' (catalog/ciru_inkling/KERNELS.md has the table). '
                        'Launch functions take raw device pointers, element strides and a stream (ctypes from ops/hip_runtime.py); '
                        'none takes a torch tensor. ' + NOT_TIMED))

# every HIP source of the origin is in a module, and the units are the origin's _HIP_TU_FILES
hips = sorted(f for f in O.files() if f.endswith(('.hip', '.cuh', '.cu')))
if hips != sorted([OPS + 'hip_common.cuh'] + [OPS + t for t in TUS]):
    raise SystemExit('HIP sources differ from the harvest\'s list: ' + ' '.join(hips))
rt = open(O.find(OPS + 'hip_runtime.py')).read()
if tuple(re.findall(r'"(hip_\w+\.hip)"', rt.split('_HIP_TU_FILES', 1)[1].split(')', 1)[0])) != TUS:
    raise SystemExit('ops/hip_runtime.py _HIP_TU_FILES is not the harvest\'s TUS')
for name, rows in UNITS.items():
    text = open(O.find(OPS + 'hip_%s.hip' % name)).read()
    defined = set(re.findall(r'^(?:extern "C" )?(?:int|void|const char\*)\s+((?:launch_|inkling_)\w+)\s*\([^)]*\)\s*\{', text, re.M)) | \
        set(re.findall(r'^DEF_W(?:13|2)_LAUNCH\((\w+)', text, re.M))
    listed = {e for r in rows for e in r['entries']}
    if defined != listed:
        raise SystemExit(f'hip_{name}.hip: entry points differ from UNITS: ' + ' '.join(sorted(defined ^ listed)))

ORIGIN_TOML = '''
# Inkling 2 Mini CIRU Halo Agent: Ciru's runtime for Thinking Machines Lab's Inkling 2 Mini on one Strix Halo (gfx1151),
# a tuned ROCm vLLM with a complete native RDNA 3.5 HIP prefill and decode engine (seven translation units, 299 kernels by
# the author), DSpark speculative decoding and Ciru's fitted expert weight banks (card ORIGIN-CIRU-INKLING 8bd576841e).
name      = "ciru_inkling"
repo      = "https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151"
commit    = "b3870f35926fb322f398e6c373a93098534f8fdb"
branch    = "main"
version   = "v1.5.0 (commit of 2026-11-23: Release v1.5.0: all-HIP RDNA 3.5 prefill and decode engine); HIP ABI version 3; source hash 671b45b80972"
reference_checkout = "control-node: /data/ark-inbox/src/github.com__ciru-ai__Inkling-2-Mini-CIRU-Halo-Agent-gfx1151#b3870f35926fb322f398e6c373a93098534f8fdb (ark intake 7546d7e, a depth-1 clone; its .git has no refs directory, so git commands fail there and the harvest reads .git/HEAD); laptop: ~/Code/strixflerp-upstream-supplement/ciru_inkling (rsync of that copy, 2026-11-24). Every file named in the release's MANIFEST.sha256 matches it in both"

[licence]
spdx      = "Apache-2.0"
files     = ["LICENSE", "NOTICE"]                          # copied verbatim next to this file
third_party = "NOTICE: Ciru's runtime changes, fitted weights and packaging are Apache-2.0; the runtime incorporates modified vLLM code (Apache-2.0) and the Python overlay files carry the vLLM contributors' SPDX header; the seven .hip units and hip_common.cuh carry only 'SPDX-License-Identifier: Apache-2.0, Copyright (c) 2026 CIRU AI'. The bundled ROCr library (scripts/rocm, AMD ROCr 6b0e43f3 with the idle-polling fix 46558b7a; its own licence files) and the model weights (Thinking Machines Lab, Apache-2.0, not in this repository) are not harvested."

[toolchain]
host      = "python 3.12 with PyTorch 2.13.0+rocm10.0.0, Triton 3.8.0 and vLLM 0.34.0+rocm100 patched by patches/vllm-inkling2-gfx1151.patch and the overlay (README.md:58, scripts/install.sh); the HIP library itself links only libamdhip64.so.7 and libc"
device    = "therock-10.0"                               # the release pins the ROCm 10.0 SDK wheels (README.md:48, 58)
build     = "ops/hip_runtime.py builds one shared object from the seven units when no prebuilt library matches the source hash: hipcc -shared -fPIC -O3 --offload-arch=gfx1151 -Wl,--as-needed -DINKLING_HIP_ABI_VERSION=3 -DINKLING_HIP_SRC_HASH_RAW=<hash> -I<ops> <seven .hip> -o hip_inkling_gfx1151_<hash>.so. The release ships that library prebuilt: scripts/rocm/hip_inkling_gfx1151_671b45b80972.so, %(bytes)d bytes, sha256 %(sha)s (MANIFEST.sha256); 671b45b80972 is the hash of exactly the harvested sources (recomputed by the harvest script). It is not harvested and was not rebuilt or compared here"

[compile]                                                # the origin's hipcc line, one object per unit (-c for -shared)
driver    = "<rocm>/bin/hipcc"
arch_flag = "--offload-arch=gfx1151"
rocm_flag = "--rocm-path=<rocm>"
flags     = ["-fPIC", "-O3", "-DINKLING_HIP_ABI_VERSION=3", "-DINKLING_HIP_SRC_HASH_RAW=671b45b80972", "-c"]
output    = "host object with the gfx1151 device code embedded (.o), one per translation unit"
note      = "the origin links the seven units into one shared object (-shared -Wl,--as-needed); the gate compiles each unit on its own with -c. hip_glue_decode.hip declares launch_hip_reduce_splits, which hip_quant_decode.hip defines, so the units link only together. %(gate)s"

[kernels]
loading   = "ctypes: ops/hip_runtime.py loads the prebuilt library (or $INKLING_AMD_HIP_SO, or a cached or freshly compiled one), checks inkling_hip_abi_version() == 3, the source hash and a list of required launch symbols, and falls back to the Triton / Gluon kernels when any check fails; seven per-unit kill switches (README.md:30)"
selection = "the Python ops (overlay/vllm/models/inkling2/amd/ops/*.py) pick a HIP launch or the Triton / Gluon fallback per call from shape and mode; HIP graphs are captured over 1..8 active sequences"
models    = "thinkingmachines/Inkling-2-Mini (48 layers, hidden 2304, 128 experts top 8, 8 KV heads of 128) with its DSpark draft, on the original NVFP4 experts or Ciru's fitted banks. Author figures, v1.5.0 (README.md:19): 2,299.4 client / 2,421.9 server prompt tok/s at 4K, 50.2 server decode tok/s at 128K, 194-208 tok/s 8-stream decode"
''' % {'bytes': PREBUILT_BYTES, 'sha': PREBUILT_SHA, 'gate': GATE}

LEFT_OUT = {
    'overlay/vllm/ (except ops/*.hip and ops/hip_common.cuh)': 'the vLLM model, attention, MoE, DSpark, scheduler and model-runner overlay in Python, the ctypes loader ops/hip_runtime.py and the Triton / Gluon fallback kernels; written against vLLM internals',
    'patches/vllm-inkling2-gfx1151.patch': 'the vLLM patch the installer applies (1 MB)',
    'scripts/': 'install, serve and smoke-test scripts, the bundled ROCr library with its patch and licences, and the prebuilt kernel library (hashes recorded in ORIGIN.toml)',
    'assets/, README.md, INSTALL.md': 'documentation; README.md was read for the module records',
    'MANIFEST.sha256, WEIGHTS.sha256, .gitignore': 'release manifests (the harvest checks the checkout against MANIFEST.sha256) and repository plumbing',
    'LICENSE, NOTICE': 'copied beside ORIGIN.toml',
}


def providers():
    out = {}
    cat = os.path.join(harvestlib.REPO, 'catalog')
    for origin in sorted(os.listdir(cat)):
        if origin == O.name or not os.path.isdir(os.path.join(cat, origin)):
            continue
        for kind in sorted(os.listdir(os.path.join(cat, origin))):
            kd = os.path.join(cat, origin, kind)
            if not os.path.isdir(kd):
                continue
            for name in sorted(os.listdir(kd)):
                p = os.path.join(kd, name, 'module.toml')
                if os.path.isfile(p):
                    t = tomllib.load(open(p, 'rb'))
                    out[t['id']] = [x.get('interface', '') for x in t.get('provides', [])]
    return out


def kernels_md():
    prov = providers()
    o = ['# Inkling 2 Mini CIRU HIP kernels by strixflerp interface', '',
         'One row for each group of `extern "C"` launch functions in the seven HIP units of `%s` at b3870f3 (v1.5.0).' % OPS.rstrip('/'),
         'Card ORIGIN-CIRU-INKLING 8bd576841e. Written by `docs/harvest/harvest_ciru_inkling.py` from its `UNITS` table, read',
         'from source and the origin\'s README.md; the compile gate builds each unit for gfx1151, nothing was run.', '',
         '- **tag**: the strixflerp interface class the group matches, `<family>.<op>/<weight format>/<dtype>`, or none. The',
         '  module\'s `[[provides]]` entry carries the same name with a `.ciru-inkling...` variant tag.',
         '- **same op in the catalogue**: kernel modules of other origins that provide that `<family>.<op>`; the harvest script',
         '  fails if one of them does not. Same op is not same operands: these kernels take raw pointers in the layouts of',
         '  Ciru\'s vLLM overlay (paged K/V cache with a block table, route plans from the planners, the packed banks).', '',
         '| unit | lines | launch functions | tag | same op in the catalogue |', '|---|---:|---|---|---|']
    tagged = untagged = 0
    for name, rows in UNITS.items():
        f = 'hip_%s.hip' % name
        lines = len(open(O.find(OPS + f)).read().splitlines())
        for r in rows:
            if r['tag']:
                op = r['tag'].split('/')[0] + '/'
                for b in r['beside']:
                    if b not in prov or not any(i.startswith(op) for i in prov[b]):
                        raise SystemExit(f'{f}: {b} does not provide {op}')
                tagged += 1
            else:
                untagged += 1
            ents = ['`%s`' % e for e in r['entries']]
            if len(ents) > 4:
                ents = ents[:2] + ['... %d in all' % len(ents)]
            o.append('| `%s` | %d | %s | %s | %s |' % (f, lines, ', '.join(ents),
                                                     '`%s`' % tag_class(r['tag']) if r['tag'] else 'none',
                                                     ', '.join('`%s`' % b for b in r['beside']) or '-'))
    n_entries = sum(len(r['entries']) for rows in UNITS.values() for r in rows)
    o += ['', '%d units, %d launch groups (%d entry points): %d carry a tag, %d are none (planners, helpers, ABI metadata and '
          'the Inkling-specific short convolution, loop controller and block attention residuals, which no catalogue module provides).'
          % (len(UNITS), tagged + untagged, n_entries, tagged, untagged),
          '`hip_common.cuh` is the header-only module `kernel.ciru_inkling.common`.', '',
          '## What each group does', '']
    for name, rows in UNITS.items():
        for r in rows:
            o.append('- `hip_%s.hip` `%s`%s: %s.' % (name, r['entries'][0], (' (+%d)' % (len(r['entries']) - 1) if len(r['entries']) > 1 else ''), r['what']))
    o += ['', '## Not done here', '',
          'No kernel was run or timed on the pod and the prebuilt library was not compared with a rebuild. A timing of the B = 8',
          'decode units against the strixite modules is its own experiment card; it needs an adapter from strixflerp\'s weight',
          'views to these packed banks, or Ciru\'s fitted weights, which are not in this repository.']
    open(os.path.join(harvestlib.REPO, 'catalog', O.name, 'KERNELS.md'), 'w').write('\n'.join(o) + '\n')


if __name__ == '__main__':
    run(O, mods, ORIGIN_TOML, LEFT_OUT, ['LICENSE', 'NOTICE'])
    kernels_md()
