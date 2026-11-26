#!/usr/bin/env python3
"""Harvest libstrix (asf0/libstrix, 4f1c06f) into catalog/libstrix.
Run: python3 docs/harvest/harvest_libstrix.py

Card ORIGIN-LIBSTRIX dd6f98d883; verdict of KSRC-SURVEY f7de2c34b1 (/data/pod/reports/2026-11/ksrc-survey.md, row 1).
Scope: the 22 kernels of csrc/{attn,gdn,gemm,moe,norm} as one module a family, the shared device header with the unit
that pins its WMMA fragment layout, the torch op registration with the origin's build files and kernel table, and
the per-kernel tests and bench scripts. The vLLM plugin (python/strix_vllm), the NPU shim (npu/), the Docker image
and recipes, the model-level benches, tools/ and the docs are listed as unharvested.

libstrix builds one shared object, python/strix/_C.so: every .hip file holds its device kernels and a host function
that takes torch::Tensor arguments, checks the geometry and launches, and csrc/torch_ops.cpp registers those
functions as torch.ops.strix.*. A kernel is therefore called with torch tensors in vLLM's layouts (paged KV cache
with a block table, moe_align_block_size sorted ids, compressed-tensors and Quark weight packings), which is another
shape than strixite's strix::kernels calls on strixw device views. Each file is tagged with the strixflerp interface
class it matches (KINDS below, written to catalog/libstrix/KERNELS.md and into the module notes). Nothing is
compiled or run by this harvest, so every module is [gate] status = "harvested".

The reference checkout is the laptop copy ~/Code/strixflerp-upstream-supplement/libstrix (4f1c06f, moved by git
bundle from the control-node ark copy /data/ark-inbox/src/github.com__asf0__libstrix while that was a depth-1 clone at
the commit). Both control-node ark copies have since moved to e3a1c45 (shallow, 2026-11-23) and no longer hold 4f1c06f,
so this script refuses them (HEAD != COMMIT). LIBSTRIX_CHECKOUT names another copy."""
import os, re, subprocess, sys, tomllib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvestlib
from harvestlib import Origin, M, P, run

CHECKOUT = os.environ.get('LIBSTRIX_CHECKOUT', os.path.join(harvestlib.SUPPLEMENT, 'libstrix'))
COMMIT = '4f1c06f55d520c734a66101e8c31fc5dd9dedeb3'


class GitOrigin(Origin):
    """The origin's file set is git's (git ls-files of the checkout), and the checkout must sit on COMMIT."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.roots = [CHECKOUT]
        head = subprocess.run(['git', '-C', CHECKOUT, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        if head != COMMIT:
            raise SystemExit(f'{CHECKOUT} is at {head}, the harvest pins {COMMIT}')
        dirty = subprocess.run(['git', '-C', CHECKOUT, 'status', '--porcelain'], capture_output=True, text=True, check=True).stdout
        if dirty.strip():
            raise SystemExit(f'{CHECKOUT} has local changes; the harvest copies files verbatim from the commit')

    def files(self):
        if self._files is None:
            out = subprocess.run(['git', '-C', CHECKOUT, 'ls-files'], capture_output=True, text=True, check=True).stdout
            self._files = sorted(f for f in out.splitlines() if f)
        return self._files


O = GitOrigin('libstrix', 'libstrix', 'https://github.com/asf0/libstrix', COMMIT, 'main', 'MIT', ['asf0'], 'therock-10.1')
FAMILIES = ('attn', 'gdn', 'gemm', 'moe', 'norm')
LIB = ('a translation unit of python/strix/_C.so: built by the origin CMake (CMakeLists.txt globs every csrc/**/*.hip beside '
       'csrc/torch_ops.cpp; HIP standard 20, -O3 -ffp-contract=fast -munsafe-fp-atomics, gfx1151, linked against libtorch). '
       'The host half of each file takes torch::Tensor arguments, so no unit compiles without the PyTorch headers; not compiled by this harvest')
VLLM = 'vLLM (Apache-2.0, vllm-project/vllm) and the asf0/strix-vllm fork (branch gfx1151)'
NOT_TIMED = 'No timing was taken on the pod.'


def K(file, ops, tag, beside, what, numeric=None, regime=None):
    """One kernel file. ops: the torch.ops.strix entry points it defines. tag: the strixflerp interface it matches, as a
    full interface name (the class is the name without its .tag), or None. beside: catalogue kernel modules of the
    same <family>.<op>; kernels_md() checks that each exists and provides that op."""
    return dict(file=file, ops=ops, tag=tag, beside=beside, what=what, numeric=numeric, regime=regime)


KINDS = {
 'attn': [
  K('attn_decode_bf16.hip', ['attn_decode_bf16'], 'attn.softmax-gqa/kv-bf16+i8/bf16.libstrix-decode',
    ['kernel.strixite.attention', 'kernel.gufo.qwen_attention', 'kernel.hipfire.attention', 'kernel.tensorfold.hip_attention'],
    'paged attention for pure-decode batches, one query token a sequence: a block owns one (sequence, KV head, key segment) and all query heads of that KV head, stages each K/V chunk in LDS once, and runs an online softmax per head; split-KV over key segments, optional sliding window and attention sinks',
    'bf16 Q and K/V, or an int8 K/V cache with a scale per token and head (k_scale, v_scale); the segments are combined first and the sink logit is folded in once after that (file header)',
    'decode; head 128 GQA 8, head 256 GQA 2 or 8, head 512 GQA 8 (docs/kernels.md:77)'),
  K('attn_prefill_bf16.hip', ['attn_prefill_bf16'], 'attn.prefill-causal/kv-bf16+i8/bf16.libstrix',
    ['kernel.flerp.attn_wmma_gqa'],
    'causal paged attention for prefill and speculative verify over cu_seqlens_q: a block owns 16 query tokens of one sequence and HPB query heads of one KV head, a head is D / 128 waves that split head_dim, and everything is computed transposed (S^T = K Q^T, O^T = V^T P^T) so a WMMA lane owns one query column',
    'bf16 Q and K/V or an int8 K/V cache; WMMA scores and P V with per-lane softmax statistics; per-shape scheduling flags (mask_skip, hoist) change the schedule only and every shape stays bit-compatible with the tests (file header)',
    'prefill; head 64 GQA 4, head 128 GQA 8, head 256 GQA 2 / 6 / 8, head 512 GQA 8, any KV block size, optional sliding window; sinks on unsplit launches only (docs/kernels.md:76)'),
  K('attn_prefill32_bf16.hip', ['attn_prefill32_bf16'], 'attn.prefill-causal/kv-bf16/bf16.libstrix-d512x32',
    ['kernel.flerp.attn_wmma_gqa'],
    'head_dim 512 causal prefill with 32 query columns sharing one K/V stage: eight waves change role by phase (score: 2 column tiles x 4 D-slices; softmax: one wave a column tile; P V: 64 V dims a wave for both tiles)',
    'bf16 K/V; partial scores are kept as bf16 in LDS, about 3e-3 relative error on P by the file header',
    'prefill; head 512, no window, one KV segment'),
  K('qsa_indexer_bf16.hip', ['qsa_indexer_bf16'], 'attn.qsa-indexer/-/bf16.libstrix-scores',
    ['kernel.strixite.qsa', 'kernel.crucible.qsa', 'kernel.gufo.qfn_kernels', 'kernel.hipfire.indexer_topk', 'kernel.tensorfold.hip_flashnext'],
    'Qwen3.8-Flash-Next QSA indexer scores over the paged compressed-key cache: logits[r, c] = sum over heads of relu(q[r, h] . k[c]) / divisor for c < visible[r], -inf past it; a block owns 64 rows x 64 keys and stages the key tile in LDS once for all heads',
    'bf16 q and compressed keys; scores only: the top-k over the logits stays with the caller (vLLM)',
    'single-request prefill of 128 rows or more; 4 indexer heads, head_dim 128, one KV head (docs/kernels.md:86)'),
  K('qsa_kv_to_f16.hip', ['qsa_kv_to_f16'], None, [],
    'rewrites one layer\'s visible K/V from the paged cache into compact fp16 arrays indexed by token (exact from a bf16 cache, q * scale rounded once from an int8 cache), the copy qsa_union_attn_bf16 gathers from'),
  K('qsa_union_bf16.hip', ['qsa_union_attn_bf16'], 'attn.softmax-gqa/kv-bf16+i8/bf16.libstrix-qsa-union',
    ['kernel.strixite.attention', 'kernel.gufo.qfn_kernels', 'kernel.hipfire.attention'],
    'Qwen3.8-Flash-Next QSA sparse attention over row unions: 16 consecutive query rows share one sorted union of their selected tokens (a build kernel makes the union list and one membership bit a row from a bitmap and prefix popcount in LDS), the attention kernel gathers K/V for the union once and masks each score by membership, in attn_prefill_bf16\'s transposed WMMA layout',
    'per (row, head) the softmax over that row\'s own selection, the masked-out scores being the extra cost; a bf16 cache is read in place or through the compact fp16 copy, an int8 cache always through the fp16 copy (docs/kernels.md:85)',
    'single-request prefill of 1,024 rows or more; head_dim 256, GQA 12, at most 131,072 positions (docs/kernels.md:85)'),
 ],
 'gdn': [
  K('gdn_chunk_k128_v128_c64_bf16.hip', ['gdn_chunk_fwd_k128_v128_c64_bf16'], 'mixer.gdn-prefill/-/bf16.libstrix-c64',
    ['kernel.crucible.gdn'],
    'gated delta rule, chunked prefill over cu_seqlens: kernel 1 computes A = (I + strict_lower(beta k.k e^dgc))^-1 and P = scale tril(q.k e^dgc) for every (chunk, head) in parallel, kernel 2 walks one sequence\'s chunks for one head and a 64-column slice of V with the state in registers',
    'bf16 q / k / v, fp32 gates; the recurrent state is read and written as fp32 or bf16 (vLLM\'s --mamba-ssm-cache-dtype); no in-kernel L2 norm',
    'prefill; head_k = head_v = 128, chunks of 64 tokens'),
  K('gdn_conv_post_bf16.hip', ['gdn_causal_conv1d_w4_bf16', 'gdn_post_conv_bf16'], 'mixer.gdn-front/-/bf16.libstrix',
    ['kernel.gufo.qwen_gdn', 'kernel.hipfire.gdn'],
    'GDN prefill front end: the width-4 causal depthwise conv over mixed_qkv with a 3-row carried state a sequence, then the post-conv split into q / k / v with L2 norm and the g and beta gates; feature blocks are the fastest grid index so concurrent blocks read the same token rows',
    'bf16 x and conv state, L2 norm on, plain g (docs/kernels.md:74)',
    'prefill; conv width 4, heads of 128, varlen over cu_seqlens'),
  K('gdn_decode_mtp_bf16.hip', ['gdn_decode_mtp_bf16'], 'mixer.gdn/-/bf16.libstrix-mtp',
    ['kernel.strixite.gdn', 'kernel.crucible.gdn', 'kernel.gufo.qfn_kernels', 'kernel.gufo.qwen_gdn', 'kernel.hipfire.gdn', 'kernel.tensorfold.hip_recurrence'],
    'GDN speculative decode (MTP) for one request x value head a block: L2 norm of q / k, gating, the delta-rule recurrence over the request\'s draft tokens from the last accepted state, a state write per token slot and the gated RMS norm of the output; a HIP port of vLLM\'s CUDA-only csrc/gdn/fused_gdn_decode_kernel.cu',
    'same math and thread layout as the vLLM kernel by the file header; bf16 handled as raw uint16',
    'decode with draft tokens; the schema of vLLM\'s _C::fused_gdn_decode_post_conv_mtp'),
 ],
 'gemm': [
  K('fp8_dequant.hip', ['fp8_dequant'], None, [],
    'float8_e4m3fn [N, K] with 128x128 block scales or per-output-channel scales to a bf16 [N, K] copy, which the plugin hands to hipBLASLt for large prefill batches'),
  K('fp8_wmma_gemm.hip', ['fp8_wmma_gemm', 'fp8_wmma_gemm_cfg', 'fp8_wmma_gemm_raw', 'fp8_wmma_gemm_raw_cfg'],
    'linear.prefill/fp8-e4m3/bf16.libstrix', ['kernel.strixite.linear_wmma', 'kernel.hipfire.gemm_float'],
    'FP8 W8A16 GEMM C[M, N] = A[M, K] x W[N, K]^T: e4m3 bytes are decoded to bf16 as they are staged into LDS and the main loop is a bf16 WMMA GEMM; the raw variant turns the bytes into fp16 (value / 2^8) with shifts and masks, runs the fp16 WMMA and applies the block scale to the fp32 accumulator each K step',
    'bf16 activations, never quantized; e4m3 to bf16 is exact, so the scale multiply is the only rounding (bf16 staging), and the raw path keeps the weights exact with fp32 products (file header)',
    'prefill batches between the skinny kernel and hipBLASLt: M <= 256 raw, M < 768 otherwise; 128x128 block scales (K % 128 == 0) or per-channel scales (K % 64 == 0) (docs/kernels.md:82)'),
  K('hc_up_gate_mix_bf16.hip', ['hc_up_gate_mix_bf16'], 'hc.mix-prefill/bf16/bf16.libstrix-up-gate',
    ['kernel.strixite.hc_wmma', 'kernel.crucible.hc'],
    'Qwen3.8-Flash-Next hyper-connection up projection fused with the gate mix: y[m, c] = (1/4) sum over 4 streams of sigmoid(bf16(a[m] . w[s * C + c])) * x[m, s * C + c]; a block owns BM rows x BN columns of y and all four streams, K (the low rank) runs in 64-wide steps, and the [M, 4C] gate is never written to memory',
    'the gate is rounded to bf16 before the sigmoid like the unfused GEMM\'s output, and the four streams are added in fp32 in stream order 0..3, hc_gate_mix\'s order (file header)',
    'prefill; hc_count 4, bf16, K % 64 == 0, batches of 512 rows or more (docs/kernels.md:87)'),
  K('mxfp4_gemv.hip', ['mxfp4_gemv'], 'linear.decode/mxfp4/bf16.libstrix', ['kernel.hipfire.gemv_mfp4'],
    'MXFP4 W4A16 GEMV: one wave an output row, eight rows a block, A staged in LDS per K chunk, weights decoded in registers straight from the packed stream (16 weight bytes and one e8m0 scale byte a 32-K block)',
    'bf16 activations; an e2m1 nibble becomes its bf16 pattern by one add with the block exponent folded in, exact because the value has at most one mantissa bit',
    'decode; the plugin routes M <= 2 here (docs/kernels.md:83), the kernel takes M <= 8; Quark layout'),
  K('mxfp4_wmma_gemm.hip', ['mxfp4_wmma_gemm', 'mxfp4_wmma_gemm_cfg', 'mxfp4_dequant'], 'linear.prefill/mxfp4/bf16.libstrix',
    ['kernel.hipfire.gemm_mfp4'],
    'MXFP4 W4A16 GEMM on Quark\'s packed layout (weight u8 [N, K/2], one e8m0 scale byte per 32 elements): weights are decoded to bf16 as they are staged into LDS, then a bf16 WMMA GEMM; no dequantized weight copy',
    'bf16 activations without the fp4 activation quantize of vLLM\'s emulation kernel; the decode is exact',
    'the plugin routes M > 2 here; K % 128 == 0 (docs/kernels.md:83)'),
  K('w4a16_wmma_gemm.hip', ['w4a16_wmma_gemm', 'w4a16_dequant'], 'linear.prefill/ct-int4/bf16.libstrix',
    ['kernel.strixite.linear_wmma', 'kernel.flerp.wmma_q4p'],
    'W4A16 GEMM for prefill on vLLM\'s RDNAHybridW4A16 layout (symmetric uint4b8 as an int8 [N, K/2] view of ExLlama-shuffled int32 words, bf16 group scales): weights are dequantized to bf16 ((nibble - 8) * scale) as they are staged into LDS, then a bf16 WMMA GEMM',
    'bf16 activations, no activation quantization; each weight is rounded once to bf16 at staging',
    'prefill; the plugin routes group 32 with M > 128 here (docs/kernels.md:81)'),
  K('w8a16_skinny_gemm.hip', ['w8a16_skinny_gemm', 'fp8_skinny_gemm'], 'linear.decode/ct-int8+fp8-e4m3/bf16.libstrix',
    ['kernel.strixite.linear_q8', 'kernel.flerp.linear_q8p', 'kernel.gufo.qfn_kernels'],
    'W8A16 skinny GEMM for decode: uint8b128 weights [N, K] with per-channel or group-128 scales against bf16 / fp16 activations, and the same file\'s skinny GEMM on float8_e4m3fn weights (the e4m3 byte read as an fp16 holding value / 2^8); derived from vLLM\'s csrc/rocm/skinny_gemms.cu through the strix-vllm fork (Apache-2.0)',
    'activations stay bf16 / fp16; the dense_w8 use reads an int8 copy of a bf16 weight, each copy checked against bf16 at load (docs/kernels.md:88)',
    'decode; the op takes M <= 8, the plugin routes M <= 5 (docs/kernels.md:79); K % 128 == 0 for compressed-tensors weights'),
  K('w8a8_wmma_gemm.hip', ['w8a8_wmma_gemm', 'w8a8_wmma_gemm_cfg', 'w4a8_wmma_gemm'], 'linear.prefill/ct-int8+ct-int4/i8-f32.libstrix',
    ['kernel.gufo.qfn_kernels'],
    'W8A8 GEMM on the iu8 16x16x16 WMMA for larger M: uint8b128 weights read raw, activations quantized to int8 per (row, 128-element K group) inside the op, the i32 accumulator folded into fp32 with s_w * s_a at every group boundary; w4a8_wmma_gemm is the int4-weight entry of the same file; derived from vLLM\'s csrc/rocm/skinny_gemms.cu through the strix-vllm fork (Apache-2.0)',
    'out[m, n] = sum over groups of s_a[m][g] * s_w[n][g] * (integer dot of the group), q = clamp(rint(x / s_a), -127, 127), s_a = max |x| of the group / 127 (file header); activations are quantized, so this is not the W8A16 result',
    'prefill; K % 128 == 0; the plugin routes W8A8 linears and, for int4, M > 5 here (docs/kernels.md:80-81)'),
 ],
 'moe': [
  K('moe_gemm_bf16.hip', ['moe_gemm_bf16'], 'moe.experts-prefill/bf16/bf16.libstrix',
    ['kernel.strixite.moe_grouped', 'kernel.flerp.moe_grouped_p', 'kernel.hipfire.moe'],
    'grouped expert GEMM over vLLM\'s moe_align_block_size layout, C[sid, n] = sum_k A[sid / top_k, k] * B[expert(sid), n, k] (* w[sid]): a block owns BM sorted ids of one expert and BN output columns; the w13 GEMM fuses SiLU or GELU-tanh and the gate * up product, each activation its own template instantiation',
    'bf16 operands on the bf16 WMMA; the SiLU instantiation keeps the exact expression its published numbers were measured on (file header)',
    'both expert GEMMs of vLLM\'s TritonExperts; no quantization, LoRA, bias or expert parallelism (docs/kernels.md:69)'),
  K('moe_gemm_w4a16.hip', ['moe_gemm_w4a16'], 'moe.experts-prefill/ct-int4/bf16.libstrix',
    ['kernel.strixite.moe_grouped', 'kernel.flerp.moe_kgrouped'],
    'the same grouped expert GEMM on int4 weights in vLLM\'s Triton WNA16 layout (uint8 [E, N, K/2], plain nibble order, bf16 group scales): weights are dequantized as they are staged into LDS, or with DEFER staged as the exact bf16 128 + q and the group scale applied to the fp32 accumulator at the group\'s end',
    'bf16 activations; DEFER keeps the weights unrounded: acc += s[n] * (raw - 136 * rowsum(A)), the row sums from one extra WMMA against a ones fragment (file header)',
    'uint4 with group 32 / 64 / 128, no zero-point (docs/kernels.md:70)'),
  K('moe_gemm_w4a8.hip', ['moe_gemm_w4a8', 'moe_act_quant_i8'], 'moe.experts-prefill/ct-int4/i8-f32.libstrix',
    ['kernel.strixite.moe_grouped'],
    'the grouped expert GEMM on the iu8 WMMA: activations pre-quantized to int8 per (row, 128-element group) by moe_act_quant_i8, int4 weights turned into int8 at the LDS commit with two byte permutes and an add / xor per 4 values, the i32 accumulator folded into fp32 with sa * sw every 128-wide K step',
    'C[sid, n] = sum over groups of sa * sw * integer dot (file header); activations are quantized, so it changes numerics against the W4A16 kernel and is on for Qwen3.8-Flash-Next only by default (docs/kernels.md:10)',
    'batches of 1,024 tokens or more; uint4 with group 128, K % 128 == 0 (docs/kernels.md:84)'),
  K('moe_gemm_w8a16.hip', ['moe_gemm_w8a16', 'moe_gemm_fp8', 'moe_gemm_fp8_raw'], 'moe.experts-prefill/ct-int8+fp8-e4m3/bf16.libstrix',
    ['kernel.strixite.moe_grouped'],
    'the grouped expert GEMM on one-byte weights, the format a template argument: int8 with a scale per output channel, float8_e4m3fn with per-channel or 128x128 block scales, and two raw FP8 forms that turn the bytes into fp16 with shifts and masks, run the fp16 WMMA and apply the scale to the fp32 accumulator',
    'activations stay bf16 (fp16 for the raw forms), not quantized; in the raw forms the weights are exact and the fused activation is written as fp16, or as bf16 when silu(gate) * up leaves the fp16 range (file header)',
    'int8: K % 64 == 0, N % 128 == 0, opt-in because it measured slower than Triton\'s int8 WMMA (docs/kernels.md:71); FP8: N % 128 == 0 (docs/kernels.md:72)'),
 ],
 'norm': [
  K('gdn_gated_rmsnorm_h128_bf16.hip', ['gdn_gated_rmsnorm_h128_bf16'], 'norm.rmsnorm-gated/-/bf16.libstrix-h128',
    ['kernel.strixite.norm'],
    'the GDN output norm out = rms_norm(x) * w * act(z) over 128-channel rows (RMSNormGated with norm_before_gate=True), 16 lanes a row and 8 channels (one 16-byte load) a lane',
    'bf16 rows; the gate is SiLU (activation 0) or sigmoid (1), named at every call',
    'rows of 128 channels (docs/kernels.md:67)'),
 ],
}

SUMMARY = {
 'attn': 'libstrix\' six attention kernels on vLLM\'s paged KV cache: decode and causal prefill attention on bf16 or int8 K/V (head 64 to 512, GQA, sinks, sliding window), a 32-column head-512 prefill kernel, and for Qwen3.8-Flash-Next the QSA indexer scores, the compact fp16 K/V copy and sparse attention over 16-row unions of the selected tokens',
 'gdn': 'libstrix\' three Gated DeltaNet kernels for the Qwen3.x GDN layers: the chunked delta rule for prefill (heads of 128, chunks of 64, an A / P kernel and a state-walking kernel), the prefill front end (width-4 causal conv, then q / k / v with L2 norm and gates) and a HIP port of vLLM\'s fused speculative-decode kernel',
 'gemm': 'libstrix\' eight dense linear kernels on quantized checkpoints as vLLM stores them: skinny decode GEMMs and a GEMV with bf16 activations (W8A16, FP8, MXFP4), WMMA prefill GEMMs that decode weights to bf16 while staging them into LDS (W4A16, FP8, MXFP4), an iu8 WMMA GEMM with int8 activations per 128-element group (W8A8, W4A8), an FP8 dequant, and the Qwen3.8-Flash-Next hyper-connection up projection fused with its gate mix',
 'moe': 'libstrix\' four grouped expert GEMMs over vLLM\'s moe_align_block_size layout, one block per (BM sorted ids of one expert, BN output columns), with the SiLU or GELU-tanh gate and the gate * up product fused into the first GEMM: bf16 weights, int4 with bf16 activations, int4 with int8 activations on the iu8 WMMA, and one-byte weights (int8, FP8, raw FP8)',
 'norm': 'libstrix\' gated RMSNorm for the GDN output, out = rms_norm(x) * w * act(z) on 128-channel bf16 rows with a SiLU or sigmoid gate, 16 lanes a row and 8 channels (one 16-byte load) a lane',
}

# Why each module exists (SPEC.md section 5; tools/describe-audit.py wants one per module, none shared).
WHY = {
 'kernel.libstrix.attn': 'the survey gave libstrix a harvest verdict for this family first: qsa_union and qsa_indexer are a permissively licensed second implementation of Flash-Next\'s QSA prefill beside strixite\'s, and the origin reports the union kernel at 1.5x and the indexer at 1.7-1.9x the gfx1151 Triton kernels (docs/kernels.md:85-86), a comparison kernel.strixite.qsa and attention have not been put through',
 'kernel.libstrix.gdn': 'the survey names GDN chunk prefill among the prefill surfaces with one implementation in the catalogue; this MIT one differs from kernel.strixite.gdn in ways its source states (64-token chunks against 32, an A / P kernel for every chunk in parallel before the state walk, a bf16 state accepted in place), so a timing card can test whether any of them transfers',
 'kernel.libstrix.gemm': 'hc_up_gate_mix is a fusion the survey found nowhere else (the origin: 6.7 to 5.2 ms a site at 16k rows with no gate in memory, docs/kernels.md:87), and the skinny W8A16 kernel is what takes the origin\'s Flash-Next decode from 51 to 37 ms a step (docs/kernels.md:88), to be set against kernel.strixite.linear_q8 on the same model; the FP8 and MXFP4 kernels read checkpoint formats no strixite kernel reads',
 'kernel.libstrix.moe': 'kernel.strixite.moe_grouped buckets routes itself and reads strixw Q4 / Q5 rows; these kernels do the grouped product on the sorted-id layout a vLLM host already has and on four other weight encodings, with two ideas to try against it: the deferred group scale that keeps int4 weights unrounded, and int8 activations on the iu8 WMMA (1.2-1.34x the W4A16 tile at 16k Flash-Next tokens, docs/kernels.md:84)',
 'kernel.libstrix.norm': 'the GDN output norm runs on every token and value head of each GDN layer, and this is a second gated RMSNorm on 128-channel bf16 rows beside kernel.strixite.norm; at 140 lines it is one of the smallest libstrix kernels, so it is a cheap one to carry through an adapter first',
 'kernel.libstrix.common': 'every libstrix kernel but three includes this header for its lane permutes, lane-group sums and WMMA fragment helpers, so no family module can be read or compiled without it; the test unit pins the fragment layout those helpers assume on gfx1151',
 'kernel.libstrix.torch_ops': 'a libstrix kernel has no C ABI of its own: the schema registered here is the only statement of each entry point\'s operands, and the registry table is the only statement of the geometry each kernel accepts; an adapter to strixflerp\'s executor ABI starts from these two files',
 'tool.libstrix.kernel_tests': 'the survey counted a bench script a kernel among the reasons libstrix is the cheapest origin to harvest; these scripts and the torch-reference tests define what correct and faster mean for each kernel in the origin\'s own terms, which the timing card against the strixite modules needs before it writes its own',
}

mods, claimed = [], set()


def take(names):
    out = []
    for n in names:
        hits = [f for f in O.expand([n]) if f not in claimed]
        if not hits:
            raise SystemExit(f'nothing left for {n}')
        out += hits
    claimed.update(out)
    return out


def mod(id, summary, names, provides, notes, lang='hip-c++', compile='n/a', reason=LIB, requires=(), sources=None, spdx=None,
        upstream=None, toolchain=None):
    files = take(names)
    if sources is None:
        sources = [f for f in files if f.endswith('.hip')]
    incs = sorted({'src/' + os.path.dirname(f) for f in files if f.endswith(('.h', '.hpp', '.hip', '.cpp'))})
    mods.append(M(id, summary, files, provides=provides, requires=requires, lang=lang, sources=sources, includes=incs,
                  compile=compile, reason=reason, notes=notes, why=WHY[id], upstream=upstream, spdx=spdx, toolchain=toolchain))


COMMON = 'module:kernel.libstrix.common'
FAMILY_NOTES = {
 'attn': 'Operands are vLLM\'s: key_cache / value_cache are views of the packed paged cache addressed through block_table, int8 caches carry k_scale / v_scale. Origin numbers (docs/kernels.md:85-86): qsa_union 1.5x the gfx1151 Triton kernel, more as the context grows; qsa_indexer 1.7-1.9x the tl.dot kernel. ' + NOT_TIMED,
 'gdn': 'gdn_decode_mtp_bf16.hip carries SPDX MIT and a vLLM copyright line (a port of vLLM\'s csrc/gdn/fused_gdn_decode_kernel.cu); the origin\'s NOTICE does not list it among the Apache-2.0 files, and this module records the licence the file states. ' + NOT_TIMED,
 'gemm': 'Licence: w8a16_skinny_gemm.hip and w8a8_wmma_gemm.hip are Apache-2.0 (derived from vLLM, NOTICE), the other six files MIT; LICENSES/Apache-2.0.txt rides in this module. Origin numbers: hc_up_gate_mix 6.7 to 5.2 ms a site at 16k rows (docs/kernels.md:87); int8 decode copies read by w8a16_skinny_gemm take Qwen3.8-Flash-Next decode from 51 to 37 ms a step, GSM8K 93 against 92 (docs/kernels.md:88). The survey notes that as an engine the origin is not faster than strixite on decode. ' + NOT_TIMED,
 'moe': 'Routing and the sorted-id layout (sorted_ids, expert_ids, num_post_padded) are the caller\'s, vLLM\'s moe_align_block_size. Expert weight rows should not sit a large power of two apart: 4 KiB-strided rows read 4x slower on gfx1151 by the origin\'s measurement (moe_gemm_bf16.hip header), so the plugin pads the row pitch. The class is moe.experts-prefill, the grouped form beside kernel.strixite.moe_grouped; vLLM calls these kernels for its expert GEMMs at decode as well. ' + NOT_TIMED,
 'norm': 'The entry returns a new tensor; activation 0 is SiLU and 1 is sigmoid. ' + NOT_TIMED,
}
FAMILY_SPDX = {'gemm': 'MIT AND Apache-2.0'}
FAMILY_UP = {
 'attn': 'operand layouts and the kernels replaced are those of ' + VLLM + '; the file headers name no vLLM code in these files',
 'gdn': 'gdn_decode_mtp_bf16.hip is a HIP port of vLLM\'s csrc/gdn/fused_gdn_decode_kernel.cu (' + VLLM + '); the chunk kernel replaces ChunkGatedDeltaRule, the FLA chunk prefill vLLM calls (docs/kernels.md:73)',
 'gemm': 'w8a16_skinny_gemm.hip and w8a8_wmma_gemm.hip are derived from vLLM\'s csrc/rocm/skinny_gemms.cu through the asf0/strix-vllm fork (Apache-2.0, NOTICE); weight layouts are compressed-tensors\' and Quark\'s as vLLM loads them',
 'moe': 'the sorted-id layout and the weight formats are those of ' + VLLM + ' (moe_align_block_size, the Triton WNA16, int8 and FP8 expert paths)',
 'norm': 'the operator is RMSNormGated as vLLM\'s Qwen GDN layers call it (docs/kernels.md:67)',
}


def tag_class(tag):
    """The interface class of a tag: the name without its .tag (flerplib.catalog's rule)."""
    head, last = tag.rsplit('/', 1)
    return head + '/' + last.split('.')[0]


def family(fam):
    rows = KINDS[fam]
    names = ['csrc/%s/%s' % (fam, r['file']) for r in rows] + (['LICENSES/Apache-2.0.txt'] if fam == 'gemm' else [])
    provides = [P(r['tag'], ', '.join('torch.ops.strix.' + o for o in r['ops']) + ' (csrc/' + fam + '/' + r['file'] + '; schemas in csrc/torch_ops.cpp)',
                  r['numeric'], r['regime']) for r in rows if r['tag']]
    tags = '; '.join('%s %s' % (r['file'], tag_class(r['tag']) if r['tag'] else 'none') for r in rows)
    mod('kernel.libstrix.' + fam, SUMMARY[fam], names, provides,
        'Interface tags by file: ' + tags + ' (catalog/libstrix/KERNELS.md has the table). ' + FAMILY_NOTES[fam],
        requires=[COMMON], spdx=FAMILY_SPDX.get(fam), upstream=FAMILY_UP[fam])


mod('kernel.libstrix.common', 'Shared gfx1151 (RDNA3.5, wave32) device helpers of the libstrix kernels (csrc/strix_common.h): DPP quad permutes, ds_swizzle and v_permlanex16 lane selects behind a shuffle-xor that skips ds_bpermute\'s LDS round trip for power-of-two strides, sums over aligned lane groups and the WMMA fragment helpers; with the one-wave test unit that pins the fragment layout (D[16, 16] = A[16, K] @ B[16, K]^T)',
    ['csrc/strix_common.h', 'csrc/testing/wmma_layout_test.hip'],
    [P('kernel-abi.libstrix-device/-/-', 'namespace strix in csrc/strix_common.h; torch.ops.strix._wmma_bf16_tile (csrc/testing/wmma_layout_test.hip)', 'n/a', 'n/a')],
    'Interface tag: none. fp8_dequant.hip, w8a16_skinny_gemm.hip and w8a8_wmma_gemm.hip do not include the header. docs/kernels.md "Hardware notes that shaped the kernels" gives three rules of the origin: pad weight row pitch (rows 4-16 KiB apart read at the same column offset are 4-8x slower), keep staging registers in ext_vector_type vectors, use strix::barrier() for LDS hand-offs.')

for fam in FAMILIES:
    family(fam)

N_OPS = len(re.findall(r'm\.impl\(', open(O.find('csrc/torch_ops.cpp')).read()))
mod('kernel.libstrix.torch_ops', 'libstrix\' calling surface and build: csrc/torch_ops.cpp declares the host entry of every kernel and registers %d schemas as torch.ops.strix.* (TORCH_LIBRARY(strix), implemented for the CUDA dispatch key), python/strix loads _C.so, registers fake implementations and holds the kernel table (registry.py: one row a kernel with the geometry it accepts and how it is built, queried by select / explain / geometries), and CMakeLists.txt with build.sh define the one shared object' % N_OPS,
    ['csrc/torch_ops.cpp', 'CMakeLists.txt', 'build.sh', 'python/strix/'],
    [P('kernel-abi.libstrix-torch-ops/-/-', 'torch.ops.strix.* (%d schemas, csrc/torch_ops.cpp); strix.select(op, **geometry), strix.explain, strix.geometries, strix.kernels, strix.ops (python/strix/registry.py)' % N_OPS,
       'n/a here: each kernel module states its own arithmetic', 'host calls on torch tensors; a kernel rejects any shape it was not built for, and select() returning None means the caller keeps its own path (README.md)')],
    'Interface tag: none. This is the layer an adapter replaces: strixflerp executors call kernels through by-value argument structs or strix::kernels functions, not through torch. The _cfg ops are bench-only (fixed tile configs for bench/kernels). The vLLM plugin that builds its dispatch gates from the table (python/strix_vllm) is not harvested.',
    sources=['csrc/torch_ops.cpp'],
    reason='the registration unit of python/strix/_C.so (compiled as HIP by the origin CMake, linked against libtorch); needs the PyTorch headers; not compiled by this harvest',
    requires=[COMMON] + ['module:kernel.libstrix.' + f for f in FAMILIES])

mod('tool.libstrix.kernel_tests', 'libstrix\' per-kernel checks, all Python through torch.ops.strix: tests/ holds the pytest files that compare kernels with torch references (conftest.py, an architecture gate, a file a kernel or weight format, and the tests of the registry table and of bench/results.py), and bench/kernels/ holds a timing script a kernel (attention prefill, GDN chunk and conv, gated RMSNorm, MoE bf16 / WNA16 / FP8, FP8 and MXFP4 GEMM and GEMV, W8A16 decode, W8A8, QSA indexer / sparse / union, the dense_w8 row sweep and two NPU projection benches)',
    ['tests/', 'bench/kernels/'],
    [P('verify.kernel-reference/-/-.libstrix', 'pytest tests/; python bench/kernels/<kernel>.py', 'each test states its own reference and tolerance', 'offline, on a gfx1151 GPU with _C.so built; not run by this harvest')],
    'Interface tag: none. Not run here: the pod has not built _C.so. bench/model, bench/container, bench/results.py, bench/run_guarded.sh and bench/upstream.sh (model-level and container checks) are not harvested.',
    lang='python', compile='n/a', reason='python: not compiled', sources=[], toolchain='',
    requires=['module:kernel.libstrix.torch_ops'])

# every csrc/<family>/*.hip is in its family module, and nothing else of csrc is left over
hips = sorted(f for f in O.files() if re.fullmatch(r'csrc/(%s)/[^/]+\.hip' % '|'.join(FAMILIES), f))
declared = sorted('csrc/%s/%s' % (fam, r['file']) for fam in FAMILIES for r in KINDS[fam])
if hips != declared or len(hips) != 22:
    raise SystemExit('csrc kernel files differ from KINDS: ' + ' '.join(sorted(set(hips) ^ set(declared))))
left = sorted(f for f in O.expand(['csrc/']) if f not in claimed)
if left:
    raise SystemExit('csrc files in no module: ' + ' '.join(left))

ORIGIN_TOML = '''
# libstrix: HIP kernels tuned for AMD Strix Halo (gfx1151, wave32) and a vLLM plugin that swaps them into stock vLLM:
# attention on the paged KV cache, Gated DeltaNet, grouped MoE expert GEMMs, dense GEMMs on W8A16 / W8A8 / W4A16 / FP8 /
# MXFP4 checkpoints, and the Qwen3.8-Flash-Next QSA and hyper-connection kernels. One .hip file a kernel, each compiled
# for one geometry named in its entry point (card ORIGIN-LIBSTRIX dd6f98d883).
name      = "libstrix"
repo      = "https://github.com/asf0/libstrix"
commit    = "4f1c06f55d520c734a66101e8c31fc5dd9dedeb3"
branch    = "main"
version   = "libstrix 0.0.1 (pyproject.toml:7); head of 2026-11-08 (docker: MTP_VOCAB (Flash-Next MTP draft vocab subset), NPU + MTP at GPU_UTIL 0.80)"
reference_checkout = "laptop: ~/Code/strixflerp-upstream-supplement/libstrix (holds 4f1c06f55d52, moved by git bundle from the control-node ark copy when it was a depth-1 clone at that commit; every harvested file byte-identical). The control-node ark copies /data/ark-inbox/src/github.com__asf0__libstrix and /data/ark/software/src/github.com__asf0__libstrix are shallow and now at e3a1c45 (updated 2026-11-23), so they no longer hold 4f1c06f"

[licence]
spdx      = "MIT"
files     = ["LICENSE", "NOTICE"]                          # copied verbatim next to this file
third_party = "NOTICE: libstrix is MIT except three files that hold vLLM code under Apache-2.0 and keep that licence in their header: csrc/gemm/w8a16_skinny_gemm.hip and csrc/gemm/w8a8_wmma_gemm.hip (derived from vLLM's csrc/rocm/skinny_gemms.cu; both in kernel.libstrix.gemm, which is MIT AND Apache-2.0 and carries LICENSES/Apache-2.0.txt) and python/strix_vllm/triton_unified_attention.py (vendored, not harvested). pyproject.toml declares the package MIT AND Apache-2.0. csrc/gdn/gdn_decode_mtp_bf16.hip states MIT with a vLLM copyright line (a port of vLLM's csrc/gdn/fused_gdn_decode_kernel.cu) and is not in NOTICE. Weights are not part of the tree."

[toolchain]
host      = "C++20, CMake 3.26 or newer with Ninja (build.sh); python 3.10 or newer with PyTorch for ROCm, whose headers and libraries the one shared object is built against"
device    = "therock-10.1"                               # README.md:53 and docker/Dockerfile:65-67: ROCm 10.1.0 wheels with torch 2.14.0+rocm10.1.0 for gfx1151; the pod has not built it
build     = "build.sh: cmake -G Ninja with CMAKE_HIP_COMPILER = $ROCM_PATH/lib/llvm/bin/clang++ -> python/strix/_C.so (target _C: csrc/torch_ops.cpp and every csrc/**/*.hip, HIP standard 20, -O3 -ffp-contract=fast -munsafe-fp-atomics, defines __HIP_PLATFORM_AMD__ and USE_ROCM, STRIX_GPU_ARCH gfx1151, linked against TORCH_LIBRARIES; CMakeLists.txt:16-25). There is no hipcc line per kernel and no kernel builds without libtorch, so this file has no [compile] recipe"

[kernels]
loading   = "linked into one torch extension: python/strix/__init__.py loads _C.so with torch.ops.load_library and the kernels are called as torch.ops.strix.<entry>; device kernels are launched by the host function in the same .hip file. There is no code-object kit, manifest or kernel id"
selection = "strix.select(op, **geometry) over python/strix/registry.py returns the kernel for a shape or None; in vLLM the plugin (python/strix_vllm, entry point vllm.general_plugins strix) builds its gates from that table and STRIX_DISABLE / STRIX_ENABLE turn kernels off and on by name (docs/kernels.md)"
models    = "as a vLLM plugin: Qwen3.6, Qwen3.8 (27B and Flash-Next), gemma-4, LFM2.5, MiniCPM-V, Muse-Glimmer and Flint (README.md); the QSA and hyper-connection kernels are Qwen3.8-Flash-Next's. Engine level, Flash-Next AWQ at 128k: 1,518 tok/s prefill, 24.2 tok/s decode (docs/performance.md:90)"
'''

LEFT_OUT = {
    'python/strix_vllm/': 'the vLLM plugin: dispatch gates, weight repacking and the NPU bank loop written against vLLM internals; it also holds the vendored Apache-2.0 triton_unified_attention.py',
    'npu/': 'a Rust shim over hipfire\'s XDNA2 trunk transport for the NPU prefill projections',
    'bench/model/, bench/container/, bench/results.py, bench/run_guarded.sh, bench/upstream.sh': 'model-level and container checks that need a served vLLM',
    'docker/': 'the image build and the serving recipes',
    'tools/': 'four developer scripts (trace analysis and conversion, attention precision, prefill tuning)',
    'docs/, README.md, AGENTS.md, CLAUDE.md': 'documentation; docs/kernels.md and docs/performance.md were read for the module records and KERNELS.md',
    'pyproject.toml, .github/, .gitignore, .dockerignore': 'packaging and repository plumbing',
    'LICENSE, NOTICE': 'copied beside ORIGIN.toml; LICENSES/Apache-2.0.txt is in kernel.libstrix.gemm',
}


def providers():
    """{module id: [interface, ...]} of the catalogue outside this origin."""
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
    """catalog/libstrix/KERNELS.md: the 22 kernel files, each with its interface tag."""
    prov = providers()
    o = ['# libstrix kernels by strixflerp interface', '',
         'One row for each of the 22 kernel files of `csrc/{attn,gdn,gemm,moe,norm}` at 4f1c06f. Card ORIGIN-LIBSTRIX',
         'dd6f98d883. Written by `docs/harvest/harvest_libstrix.py` from its `KINDS` table; read from source and from the',
         'origin\'s `docs/kernels.md`, nothing here was compiled or run.', '',
         '- **tag**: the strixflerp interface class the file matches, `<family>.<op>/<weight format>/<dtype>`, or none. The',
         '  module\'s `[[provides]]` entry carries the same name with a `.libstrix` variant tag.',
         '- **same op in the catalogue**: kernel modules of other origins that provide that `<family>.<op>`; the harvest',
         '  script fails if one of them does not. The same op is not the same operands: every libstrix kernel takes torch',
         '  tensors in vLLM\'s layouts (`csrc/torch_ops.cpp`), so each row needs an adapter before it can stand in for a',
         '  strixflerp module.', '',
         '| file | lines | module | entry points (`torch.ops.strix.`) | tag | same op in the catalogue |', '|---|---:|---|---|---|---|']
    counts = {}
    for fam in FAMILIES:
        for r in KINDS[fam]:
            rel = 'csrc/%s/%s' % (fam, r['file'])
            lines = len(open(O.find(rel)).read().splitlines())
            if r['tag']:
                op = r['tag'].split('/')[0] + '/'
                for b in r['beside']:
                    if b not in prov or not any(i.startswith(op) for i in prov[b]):
                        raise SystemExit(f'{rel}: {b} does not provide {op}')
            counts[fam] = counts.get(fam, 0) + 1
            o.append('| `%s` | %d | `kernel.libstrix.%s` | %s | %s | %s |' % (
                rel, lines, fam, ', '.join('`%s`' % x for x in r['ops']),
                '`%s`' % tag_class(r['tag']) if r['tag'] else 'none',
                ', '.join('`%s`' % b for b in r['beside']) or '-'))
    untagged = [r['file'] for fam in FAMILIES for r in KINDS[fam] if not r['tag']]
    total = sum(counts.values())
    o += ['', '%d files: %s. %d carry a tag, %d are none (%s, copies that feed another kernel).'
          % (total, ', '.join('%s %d' % (f, counts[f]) for f in FAMILIES), total - len(untagged), len(untagged),
             ' and '.join('`%s`' % u for u in untagged)),
          '`csrc/testing/wmma_layout_test.hip`, the 23rd `.hip` file of the origin, is a test unit in `kernel.libstrix.common`.', '',
          '## What each kernel does', '']
    for fam in FAMILIES:
        for r in KINDS[fam]:
            o.append('- `%s`: %s.' % (r['file'], r['what']))
    o += ['', '## Not done here', '',
          'No kernel was compiled, linked or timed. A timing against the strixite modules is its own experiment card; it needs',
          '`_C.so` built against PyTorch for ROCm, or an adapter that lifts a kernel out of its torch host function.']
    open(os.path.join(harvestlib.REPO, 'catalog', O.name, 'KERNELS.md'), 'w').write('\n'.join(o) + '\n')


if __name__ == '__main__':
    run(O, mods, ORIGIN_TOML, LEFT_OUT, ['LICENSE', 'NOTICE'])
    kernels_md()
