#!/usr/bin/env python3
"""Harvest Strata (Niko1221/Strata at release v0.1.40, 1cbcacbcae29) into catalog/strata.
Run: python3 docs/harvest/harvest_strata.py

Card ORIGIN-STRATA ef16e87953. Strata is a C++/CUDA engine for Qwen3.8-Flash-Next on one consumer GPU with the routed
experts streamed from host memory; its kernels are CUDA-shaped .cu files that the HIP build compiles as HIP for wave32
RDNA (cmake/hip_backend.cmake: every src/**/*.cu gets LANGUAGE HIP and the shim include/strata/hip_compat/cuda_runtime.h
is force-included). v0.1.40 is the build ciru-ai/strataflash measured on Strix Halo (75.8 tok/s decode).

Scope: every HIP kernel source of the engine, one kernel module a family (FAMILIES below): src/kernels/cuda/*.cu (44
files and s26_tsum.cuh), src/prefill/*.cu (7 engine files and 2 parity programs), src/core/{device,pinned,
remote_expert_opt}.cu and tests/core/dma_batch_parity.cu, with the headers they include, the HIP compatibility shim and the
vendored llama.cpp block layouts (third_party/ggml). Each kernel file is tagged with the strixflerp interface it matches
(catalog/strata/KERNELS.md and the module [[provides]]). The host engine (src/core/*.cpp, src/prefill/prefill.cpp,
src/program, src/spec, ...), the CPU expert kernels, the SYCL backend, the server, tools, tests and docs are listed as
unharvested.

The reference checkout is the ark copy on control-node
(/data/ark-inbox/src/github.com__Niko1221__Strata#1cbcacbcae2953f3be9edc46369f0c875bc6ab8b, a tree without git objects);
the laptop holds the same files, moved by a tarball through control-node get, in ~/Code/strixflerp-upstream-supplement/strata.
STRATA_CHECKOUT names another copy. The tree digest below pins the bytes, since there is no git object to check."""
import hashlib, os, re, sys, tomllib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvestlib
from harvestlib import Origin, M, P, run

# Strata's sources are .cu / .cuh: let harvestlib's #include scan (include_requires) read them too, for this origin only
harvestlib.C_FAMILY = tuple(harvestlib.C_FAMILY) + ('.cu', '.cuh')

CHECKOUT = os.environ.get('STRATA_CHECKOUT', os.path.join(harvestlib.SUPPLEMENT, 'strata'))
COMMIT = '1cbcacbcae2953f3be9edc46369f0c875bc6ab8b'
# sha256 over the lines "<sha256 of the file>  <origin-relative path>\n" of every file, sorted by path (.git left out)
TREE_SHA256 = '525320cee1a3e7d8440edea5c689dae533ba8a95a71da1fb792eff1292242a5a'
SKIP = ('.git', 'build')


def tree_files(root):
    out = []
    for d, dirs, names in os.walk(root):
        if d == root:
            dirs[:] = [x for x in dirs if x not in SKIP]
            names = [n for n in names if n not in SKIP]
        out += [os.path.relpath(os.path.join(d, n), root) for n in names]
    return sorted(out)


def tree_digest(root, files):
    h = hashlib.sha256()
    for f in files:
        h.update(f'{harvestlib.sha256(os.path.join(root, f))}  {f}\n'.encode())
    return h.hexdigest()


class TreeOrigin(Origin):
    """The origin's file set is the checkout's files (a git tree export: no .git objects to ask), pinned by digest."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.roots = [CHECKOUT]
        head = os.path.join(CHECKOUT, '.git', 'HEAD')
        if os.path.isfile(head) and open(head).read().strip() != COMMIT:
            raise SystemExit(f'{head} names {open(head).read().strip()}, the harvest pins {COMMIT}')
        got = tree_digest(CHECKOUT, self.files())
        if got != TREE_SHA256:
            raise SystemExit(f'{CHECKOUT}: tree digest {got}, the harvest pins {TREE_SHA256} (Strata {COMMIT[:12]})')

    def files(self):
        if self._files is None:
            self._files = tree_files(CHECKOUT)
        return self._files


O = TreeOrigin('strata', 'strata', 'https://github.com/Niko1221/Strata', COMMIT, 'main (release tag v0.1.40)', 'MIT',
               ['Niko1221', 'Strata contributors'], 'therock-10.2')
GGML = ['Niko1221', 'Strata contributors', 'The ggml authors (llama.cpp 3cf03257, MIT)']
NOT_TIMED = 'No timing was taken on the pod.'
LLAMA = 'llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT), the commit third_party/ggml/VERSION.txt pins'


def K(file, tag, what, api, numeric='', regime=''):
    """One kernel file (origin-relative). tag: the strixflerp interface it matches, as a full interface name (the class
    is the name without its .tag), or None. api: its host entry points."""
    return dict(file=file, tag=tag, what=what, api=api, numeric=numeric, regime=regime)


KC = 'src/kernels/cuda/'
PF = 'src/prefill/'
IK = 'include/strata/kernels/'
IP = 'include/strata/prefill/'
IC = 'include/strata/core/'

# family -> (summary, why, kernel files, other paths (headers), extra M() arguments)
FAMILIES = {
 'common': dict(
  summary='Strata\'s HIP compatibility layer and shared device headers: include/strata/hip_compat (cuda_runtime.h, the shim the HIP build force-includes into every .cu, which maps the CUDA runtime, warp intrinsics and cuBLAS spellings to HIP, plus intrinsics.hpp with the RDNA dot4 and DPP helpers), the older platform/hip_compat copy, the bit-exact bf16 / f16 conversions, dp4a, the gfx-architecture predicates, the q8_1 finiteness check, the emulation switches, the S26 tree sum (s26_tsum.cuh) and llama.cpp\'s block layouts and codebook grids (third_party/ggml/ggml-common.h, MIT)',
  why='every other kernel.strata module compiles against these headers, so they are the first thing an adapter or a recipe needs; hip_compat is also the whole of Strata\'s CUDA-to-HIP port in six files',
  files=[],
  paths=['include/strata/hip_compat/', 'include/strata/platform/hip_compat/', IK + 'bf16_bits.hpp', IK + 'f16_bits.hpp',
         IK + 'dp4a.hpp', IK + 'gfx_arch.hpp', IK + 'q8_1_finite.hpp', IC + 'emulate.hpp', KC + 's26_tsum.cuh', 'third_party/ggml/'],
  extra=dict(compile='header-only', reason='header-only; compiled inside every other kernel.strata.* module',
             includes=['src/include/strata/hip_compat', 'src/include', 'src/third_party/ggml', 'src/src/kernels/cuda'],
             authors=GGML, provides=[P('kernel-abi.strata-hip-compat/-/-', 'include/strata/hip_compat/cuda_runtime.h (force-included with -include by the origin HIP build), intrinsics.hpp', 'CUDA runtime, warp shuffle / vote and cuBLAS names mapped to HIP; wave32 assumed (warpSize 32)', 'wave32 RDNA: gfx1100 / gfx1201 validated by the origin, gfx1151 listed as unvalidated in cmake/hip_backend.cmake at v0.1.40')],
             notes='Interface tag: none for a kernel (headers only); the module provides the hip_compat ABI name. include/strata/platform/hip_compat is an older copy of the shim that no harvested source includes; it is kept here because it is HIP compatibility code of the same engine. third_party/ggml carries its own LICENSE (llama.cpp, MIT).')),
 's_gemv': dict(
  summary='Strata\'s GEMVs over its own S-family weight planes (docs/pack-format.md: S2 = Q2_0\'s canonical form, codes packed 4 a byte LSB-first with one FP32 scale a 64-element group; S4 / S8 with a codebook and an offset): the S2 decode, the naive S2 GEMV, two measured-neutral S2 variants kept as negative results (s2_gemv_fast, s2_gemv_quads), the S2 GEMV over Q8_0 activations and the generic S-family GEMV with FP16, Q8_K and Q8_0 activation forms',
  why='the S2 expert matvec is the engine\'s hottest decode path (48 layers x 10 experts x 3 roles a token over 31.64 GiB of Q2_0 weights), and these are its reference kernels; strixflerp has no 2-bit Q2_0-plane GEMV of this shape',
  files=[
   K(KC + 'dequant_s2.cu', None, 'S2 codes and scales to FP32, (code - 1) * scale, one thread an element: the decode every S2 GEMV inlines, kept host-callable as the parity reference', 'strata::kernels::dequant_s2'),
   K(KC + 's2_gemv.cu', 'linear.decode/strata-s2/f16-f32.strata', 'the naive S2 GEMV, one thread an output row, dequantizing on the fly with FP32 accumulation inside the row', 'strata::kernels::s2_gemv', 'FP16 activations, FP32 accumulation and output', 'decode, one token'),
   K(KC + 's2_gemv_fast.cu', 'linear.decode/strata-s2/f16-f32.strata-fast', 'two levers on the S2 GEMV (vector loads and a split reduction) that measured negative or neutral; the file says so and is kept as the record', 'strata::kernels::s2_gemv_fast', 'as s2_gemv', 'decode; not the engine\'s fast path (file header)'),
   K(KC + 's2_gemv_quads.cu', 'linear.decode/strata-s2/f16-f32.strata-quads', 'S2 with the code load amortised over four elements, a test of the instruction-bound hypothesis', 'strata::kernels::s2_gemv_quads', 'as s2_gemv', 'decode'),
   K(KC + 's2_gemv_q8.cu', 'linear.decode/strata-s2/q8_0-f32.strata', 'the S2 GEMV over Q8_0-quantized activations, ggml_mul_mat\'s vec_dot_type rule for Q2_0, so GPU and CPU expert paths agree on a token', 'strata::kernels::s2_gemv_q8', 'Q8_0 activation blocks, integer dot per block, FP32 output', 'decode'),
   K(KC + 's_gemv.cu', 'linear.decode/strata-s/f16+q8_k+q8_0-f32.strata', 'one GEMV for S2, S4 and S8 (value = cb[code] * scale + offset), templated on the code width, with split and async variants and Q8_K / Q8_0 activation forms', 'strata::kernels::s_gemv, s_gemv_split, s_gemv_split_async, s_gemv_q8k(_split), s_gemv_q8_0_split', 'FP16, Q8_K or Q8_0 activations, FP32 output', 'decode'),
  ],
  paths=[IK + 'dequant_s2.hpp', IK + 's2_gemv.hpp', IK + 's2_gemv_q8.hpp', IK + 's_gemv.hpp'], extra={}),
 'gguf_gemv': dict(
  summary='decode GEMVs on native GGUF blocks, adapted from llama.cpp\'s CUDA MMVQ at 3cf03257: Q2_0, Q4_0, Q5_0, Q8_0, Q3_K-Q6_K, IQ4_NL and IQ4_XS against q8_1 activations (native_mmvq), the i-quant formats IQ1_M-IQ3_S, IQ4_NL with Q2_0 and the UD-Q4_K_XL K-quants (iq_kernels: vec_dot, dequantizers, embedding rows), and the activation quantizers to Q8_0 / Q8_K the way ggml converts src1 (quantize_act)',
  why='Strata runs the IQ2_XS / IQ3_XXS and Unsloth UD-Q4_K_XL model files through these, so they are the GGUF k-quant and i-quant decode path of a Qwen3.8-Flash-Next engine measured on Strix Halo, beside gufo\'s and hipfire\'s',
  files=[
   K(KC + 'native_mmvq.cu', 'linear.decode/gguf-kquant+iq4+q2_0/q8_1-f32.strata-mmvq', 'llama.cpp\'s generic MMVQ (quantize.cu, vecdotq.cuh, mmvq.cu) for ten GGUF block types against q8_1 activations, with a fused SwiGLU-then-quantize and multi-column forms', 'strata::kernels::native_quantize_q8_1, native_swiglu_quantize_q8_1, native_<type>_mmvq, native_<type>_f32', 'q8_1 activations, llama.cpp\'s integer dots and FP32 accumulation; the CUDA build compiles it with --use_fast_math, the HIP build without', 'decode; one to a few columns'),
   K(KC + 'iq_kernels.cu', 'linear.decode/gguf-iq+kquant/q8_1-f32.strata-iq', 'the i-quant and UD-Q4_K_XL formats on the GPU: llama.cpp\'s vec_dot_*_q8_1, dequantizers and q8_1 quantizer, transcribed, plus embedding-row gathers', 'strata::kernels::iq_mmvq, quantize_q8_1_rows, iq_dequant_f16 / f32, iq_embed_rows', 'q8_1 activations; block layouts and grids from third_party/ggml/ggml-common.h', 'decode'),
   K(KC + 'quantize_act.cu', 'quant.activation/-/q8_0+q8_k.strata', 'activation quantization to Q8_0 and Q8_K (and back) the way ggml_mul_mat converts src1 to the weight\'s vec_dot_type', 'strata::kernels::quantize_q8_0(_scaled), quantize_q8_K, dequant_q8_0, dequant_q8_K', 'Q8_K\'s multiply and magic-bias rounding kept as separate FP32 operations on HIP (-ffp-contract=off, CMakeLists.txt)', 'any row length that is a multiple of the block'),
  ],
  paths=[IK + 'native_mmvq.hpp', IK + 'iq_kernels.hpp', IK + 'quantize_act.hpp'],
  extra=dict(authors=GGML, source_flags={KC + 'quantize_act.cu': ['-ffp-contract=off']})),
 'bf16_gemv': dict(
  summary='GEMVs for BF16 weights: the naive and the warp-split BF16 x BF16 GEMV (bf16_gemv) and llama.cpp\'s MMVF with FP32 activations, single, multi-column and multi-matrix forms (native_bf16)',
  why='six tensors a layer stay BF16 in a Strata pack (router, shared-expert gate, hyper-connection projections), so every token runs these; a second BF16 GEMV family next to strixite\'s and flerp\'s dense kernels',
  files=[
   K(KC + 'bf16_gemv.cu', 'linear.dense/bf16/bf16-f32.strata', 'the BF16 GEMV: one thread a row as the reference, one warp a row with lanes striding the reduction for the shapes that need it', 'strata::kernels::bf16_gemv, bf16_gemv_split', 'BF16 activations and weights, FP32 accumulation and output', 'decode'),
   K(KC + 'native_bf16.cu', 'linear.dense/bf16/f32.strata-mmvf', 'llama.cpp\'s FP32-activation MMVF (mmvf.cu) for a contiguous BF16 matrix: no fusion, ids or channels', 'strata::kernels::bf16_gemv_fp32_mmvf(_cols, _multi, _multi_aux)', 'FP32 activations, BF16 weights, llama.cpp\'s reduction order', 'decode, one to a few columns'),
  ],
  paths=[IK + 'bf16_gemv.hpp'], extra=dict(authors=GGML)),
 'moe': dict(
  summary='the MoE block\'s GPU kernels: the router (softmax over all experts, stable top-k, renormalise with ggml\'s 2^-14 clamp; and llama.cpp\'s topk-moe for 512 experts / top 10), the shared expert (silu-gated FFN with a sigmoid scalar gate) with moe_combine, llama.cpp\'s weighted reduction of the routed parts, and R4\'s grouped GPU expert tier over S2 expert blobs (hit selection, grouped GEMV, scatter-add)',
  why='Qwen3.8-Flash-Next routes 10 of 512 experts a token, and Strata\'s grouped GPU tier is what hides the streamed CPU experts; the router and combine are the same ops as the catalogue\'s moe.router and moe.combine providers',
  files=[
   K(KC + 'router_top10.cu', 'moe.router/-/f32.strata-top10', 'the router\'s softmax / top-k / renormalise half over (n_tokens, n_expert) logits, with a variant search', 'strata::kernels::router_top10, router_top10_variant', 'FP32 logits; ties break by index; ggml\'s 2^-14 lower clamp on the renormaliser', 'any expert count; top 10 in the engine'),
   K(KC + 'native_router.cu', 'moe.router/-/f32.strata-native512', 'llama.cpp\'s topk-moe for one token, 512 experts, top 10, softmax, no selection bias', 'strata::kernels::native_router_top10(_multi)', 'finite FP32 logits, I32 ids and FP32 weights out; matches the pinned CUDA contract', '512 experts, k = 10 only'),
   K(KC + 'native_moe.cu', 'moe.combine/-/f32.strata-native', 'llama.cpp\'s moe-weighted-reduction for one token or a batch, also over the GPU hit rows, with an optional shared-expert gate', 'strata::kernels::native_moe_combine(_multi, _multi_hits, _multi_hits_gated)', 'FP32 parts and weights, k in [1, 15]', 'decode and verify windows'),
   K(KC + 'shared_expert.cu', 'moe.experts-decode/strata-s+bf16/f32.strata-shared', 'the shared expert: silu(x Wg) * (x Wu) then Wd on S-family or BF16 weights, a sigmoid scalar gate per token, and moe_combine of the routed parts with it', 'strata::kernels::shared_expert(_multi), moe_combine', 'Q8_0 / Q8_K / BF16 activation forms per weight form; FP32 output', 'decode'),
   K(KC + 's2_expert_grouped.cu', 'moe.experts-decode/strata-s2/q8_0-f32.strata-grouped', 'R4\'s GPU expert tier: picks the routed experts that are resident on the GPU, runs their S2 gate / up / down GEMVs grouped (one warp an output row, the CPU expert\'s arithmetic over the same bytes) and adds the results into the routed parts', 'strata::kernels::moe_hit_select(_multi), moe_hit_grouped_s2(_dev, _multi), moe_hit_add, moe_grouped_s2, moe_group_resident', 'Q8_0 activations, the CPU pool\'s integer arithmetic, FP32 out', 'decode; experts resident in a VRAM cache'),
  ],
  paths=[IK + 'router_top10.hpp', IK + 'native_router.hpp', IK + 'native_moe.hpp', IK + 'shared_expert.hpp', IK + 's2_expert_grouped.hpp'],
  extra=dict(authors=GGML)),
 'gdn': dict(
  summary='the gated delta-net (GDN) layer\'s non-projection kernels: the reference recurrence, conv step, L2 norm, beta and gate and output norm (gdn), the conv+L2 front end and the step with its output norm in one kernel (fused_gdn), llama.cpp\'s gated_delta_net recurrence with Strata\'s state layout (native_gdn) and llama.cpp\'s conv / SiLU / L2 / gate / output-norm front and back ends (native_gdn_preprocess); src/prefill/gdn_rec_parity.cu is the prompt path\'s recurrence parity program',
  why='36 of Qwen3.8-Flash-Next\'s 48 layers are GDN, so the decode step is the per-token recurrence these kernels own; fused_gdn is Strata\'s measured fix for the native step\'s uncoalesced state reads',
  files=[
   K(KC + 'gdn.cu', 'mixer.gdn/-/f32.strata', 'the GDN step (one thread owns a (j, h) column end to end, no barrier), the depthwise conv step, L2 norm, beta gate and the gated output norm, transcribed from ref/gdn.py', 'strata::kernels::gdn_step, gdn_conv_step, gdn_l2_norm, gdn_beta_gate, gdn_out_norm', 'FP32 state and activations', 'decode, S = 128'),
   K(KC + 'fused_gdn.cu', 'mixer.gdn/-/f32.strata-fused', 'the conv with L2 norm, the alpha / beta projections, and the delta-net step fused with its RMS output norm, laid out so a warp reads contiguous state', 'strata::kernels::fused_gdn_conv_l2, fused_gdn_ab, fused_gdn_step_norm', 'FP32 state, BF16 alpha / beta weights', 'decode, heads of 128'),
   K(KC + 'native_gdn.cu', 'mixer.gdn/-/f32.strata-native', 'llama.cpp\'s gated_delta_net recurrence for one token with Strata\'s state addressing: a warp owns one (head, column) and keeps its rows in registers', 'strata::kernels::native_gdn_step', 'FP32; q not pre-scaled, the scale goes on the readout as in the pinned operator; CUDA compiles it with --use_fast_math', 'decode, S = 128, h_v a multiple of h_k'),
   K(KC + 'native_gdn_preprocess.cu', 'mixer.gdn-front/-/f32.strata-native', 'llama.cpp\'s GDN front and back ends: four-tap conv then SiLU, L2 norm, beta and gate, gated output norm', 'strata::kernels::native_gdn_conv_silu, native_gdn_l2_norm, native_gdn_beta_gate, native_gdn_gate, native_gdn_out_norm', 'FP32; fast-math on CUDA', 'decode'),
   K(PF + 'gdn_rec_parity.cu', None, 'a stand-alone parity program for the prompt path\'s chunked GDN recurrence (its own reference kernels and main); not a library unit', 'main()'),
  ],
  paths=[IK + 'gdn.hpp', IK + 'fused_gdn.hpp', IK + 'native_gdn.hpp', IK + 'native_gdn_preprocess.hpp'],
  extra=dict(authors=GGML)),
 'hc': dict(
  summary='the hyper-connection (gated residual, gr) kernels of Qwen3.8-Flash-Next\'s 4-stream residual: gr_read / gr_write after llama.cpp\'s build_hc_mix / build_hc_combine (gr), the read in two kernels with the previous write folded in (fused_gr), llama.cpp\'s weighted RMSNorm and GR post-operations (native_gr_norm, native_gr_postops), and a control vector applied to the residual stream after a layer\'s FFN write (cvec)',
  why='the hyper-connection read and write run 96 + 96 times a token; fused_gr cuts the native path\'s seven kernels to two, and the catalogue already holds hc.mix providers from strixite, crucible and radiance to set it beside',
  files=[
   K(KC + 'gr.cu', 'hc.mix/bf16/f32.strata', 'gr_read (norm, down, SiLU, up, gate and mean, inject) and gr_write over the four residual streams, single-token and multi-token', 'strata::kernels::gr_read(_multi), gr_write(_multi), gr_workspace_init', 'FP32 residual, BF16 projection weights, optional FP32-activation MMVF', 'decode and verify windows; hc = 4'),
   K(KC + 'fused_gr.cu', 'hc.mix/bf16/f32.strata-fused', 'the hyper-connection read in two kernels with the previous half\'s write folded in, several variants selected at run time', 'strata::kernels::fused_gr_read(_multi), fused_gr_supported, fused_gr_set_fast', 'BF16 weights unpacked to FP32 once, the same 8-FMA order as the unfused path', 'decode; n_embd 2560, hc 4, low rank 320 in the engine'),
   K(KC + 'native_gr_norm.cu', 'norm.rmsnorm/-/f32.strata-native-gr', 'llama.cpp\'s weighted FP32 RMSNorm for contiguous rows, independent statistics a row', 'strata::kernels::native_gr_rms_norm_weighted(_multi)', 'FP32, the pinned XOR reduction tree; fast-math on CUDA', 'any row length; 256 threads below 1024 columns, else 1024'),
   K(KC + 'native_gr_postops.cu', 'hc.mix-post/-/f32.strata-native', 'llama.cpp\'s single-token GR post-operations (dsv4-hc.cu, scale, unary): down SiLU, the pre-gated mix and the post combine', 'strata::kernels::native_gr_down_silu, native_gr_pre_gated(_multi), native_gr_post(_multi)', 'FP32', 'decode'),
   K(KC + 'cvec.cu', 'steer.add-rows/-/f32.strata-cvec', 'a control vector on the residual stream (llama.cpp\'s llama_adapter_cvec with the projection-mode extension): after a layer\'s FFN write, h . v over the stream and the update, bitwise the write fused_gr would fold', 'strata::kernels::cvec_upload, cvec_apply, cvec_set_enabled', 'FP32', 'one direction a layer over a layer range'),
  ],
  paths=[IK + 'gr.hpp', IK + 'fused_gr.hpp', IK + 'native_gr_norm.hpp', IK + 'native_gr_postops.hpp', IK + 'cvec.hpp'],
  extra=dict(authors=GGML)),
 'qsa': dict(
  summary='the QSA (sparse full-attention) layers: the reference cache, indexer, top-k and attention (qsa), split-K decode attention over the selected cells straight from the KV pools (qsa_decode_attn), the prompt path\'s attention on matrix cores (qsa_prompt_attn), the indexer\'s block scores and radix top-k for many queries at long context (qsa_select), and llama.cpp-pinned pieces: the QSA norms and gate, the indexer key cache (pooling, norm, IMRoPE), the block scores and the short-context F16 flash attention',
  why='QSA is the attention of Qwen3.8-Flash-Next\'s 12 full-attention layers and decides long-context decode and prefill cost; the indexer, top-k and sparse attention are the ops of strixite\'s, crucible\'s and gufo\'s qsa modules',
  files=[
   K(KC + 'qsa.cu', 'attn.qsa-indexer/-/f32.strata-ref', 'the spec-of-record QSA: KV append and gather on the paged FP16 pools, the indexer key append and pooled block scores, top-512 selection and attention over the selected cells, transcribed from ref/qsa.py', 'strata::kernels::kv_append, kv_gather, indexer_key_append, qsa_index, topk_512, qsa_attend', 'FP32 scores (an FP64 block a row in qsa_index), FP16 K / V pools', 'decode; 24 query heads, 2 KV heads, head 256'),
   K(KC + 'qsa_decode_attn.cu', 'attn.softmax-gqa/kv-f16+i8+q4_0/f32.strata-decode', 'split-K decode attention over the selected cells, reading the KV pools through the page table, with per-format bodies for FP16, INT8, Q4_0 and the hybrid K8V4 cache', 'strata::kernels::qsa_decode_attn_step, qsa_decode_attn_batch', 'FP32 accumulation; K / V decoded per format (Q4_0 V rotated back by the Hadamard transform)', 'decode and verify windows over about 2,051 selected cells at 4K'),
   K(KC + 'qsa_prompt_attn.cu', 'attn.softmax-gqa/kv-f16+i8+q4_0/f32.strata-prompt', 'the prompt path\'s QSA attention on matrix cores (MMA on CUDA, an RDNA WMMA kernel on HIP), replacing one-query-at-a-time decode attention in the prompt', 'strata::kernels::qsa_prompt_attn_batch', 'FP16 / INT8 K and V; the decode kernel stays the reference', 'prefill chunks; refuses a device without the matrix-core path'),
   K(KC + 'qsa_select.cu', 'attn.indexer-topk/-/f32.strata', 'the indexer\'s block scores (warp kernel and a matrix-core one) and a radix top-k over blocks for many queries at once, with a cluster variant and a reference', 'strata::kernels::qsa_block_scores(_tc), qsa_block_topk(_cluster, _ref)', 'FP32 scores; the radix select is exact on the key bits', 'prefill and long-context decode; capacity up to 262,144 cells'),
   K(KC + 'native_qsa.cu', 'norm.rmsnorm/-/f32.strata-native-qsa', 'llama.cpp\'s single-token QSA norms (weighted and grouped RMSNorm) and the output gate', 'strata::kernels::native_qsa_rms_norm_weighted, native_qsa_rms_norm_grouped, native_qsa_gate_apply', 'FP32; fast-math on CUDA', 'decode'),
   K(KC + 'native_qsa_indexer.cu', 'attn.qsa-indexer/-/f32.strata-native-keys', 'llama.cpp\'s indexer key cache: F16 raw keys, F32 pooling, weighted RMSNorm and text IMRoPE (set-rows, norm, rope)', 'strata::kernels::native_qsa_indexer_append(_batch, _steps)', 'F16 raw-key cache, F32 pooled keys', 'single sequence; idx_dim 128, idx_block 4, n_rot 64'),
   K(KC + 'native_qsa_score.cu', 'attn.indexer-score/-/f32.strata-native', 'llama.cpp\'s MMF dispatch for the indexer block scores: pooled [128, max_blocks] against query [128, 4], n_kv padded to 256', 'strata::kernels::native_qsa_score', 'F32; the padded tile as in the pinned CUDA path', 'single text sequence, four-cell blocks, top-k budget 2048'),
   K(KC + 'native_flash_attn.cu', 'attn.softmax-gqa/kv-f16/f32.strata-native-fa', 'llama.cpp\'s F16 vector FlashAttention specialised to Q 24 x 256, KV 2 x 256, one query', 'strata::kernels::native_flash_attn_short_step', 'F16 K / V, F32 q and output, scale 1/16, no ALiBi, softcap or sink', 'short contexts only, chosen explicitly'),
  ],
  paths=[IK + 'qsa.hpp', IK + 'qsa_decode_attn.hpp', IK + 'qsa_prompt_attn.hpp', IK + 'qsa_select.hpp', IK + 'native_qsa.hpp',
         IK + 'native_qsa_indexer.hpp', IK + 'native_qsa_score.hpp', IK + 'native_flash_attn.hpp'],
  extra=dict(authors=GGML)),
 'kvcache': dict(
  summary='the QSA layers\' KV storage: Q4_0 KV with a 256-point Walsh-Hadamard rotation (kv_q4, from PR #21 by code-martin), INT8 KV with an FP16 scale a 64-value group (kv_q8), and KV streaming, where the authoritative K / V live in pinned device-mapped host memory and VRAM holds a ring of pages resolved through the page table (kv_stream)',
  why='KV size is what lets a 262K context fit beside streamed experts on a 12 GB card, and the rotated Q4_0 and the host-backed page ring are two KV forms strixflerp does not have yet',
  files=[
   K(KC + 'kv_q4.cu', 'kvcache.store/q4_0-wht/f32.strata', 'FWHT-256 rotation (one warp a row, orthonormal, its own inverse) then ggml q4_0 for K and V, append (step, steps, batch) and gather', 'strata::kernels::fwht256_cuda, kv_append_q4(_step, _steps), kv_gather_q4_step', 'q4_0 blocks of rotated values; a deterministic group maximum added on merge', 'QSA layers; 576 B a cell for K and V of both heads'),
   K(KC + 'kv_q8.cu', 'kvcache.quant8/-/f32.strata', 'INT8 K / V codes with one FP16 scale a 64-value group a cell and KV head; append and gather', 'strata::kernels::kv_append_q8_step(s), kv_gather_q8_step', 'symmetric int8, FP16 scales', 'QSA layers'),
   K(KC + 'kv_stream.cu', 'kvcache.tiering/-/-.strata-stream', 'KV streaming: resolve the selected pages into the VRAM slot ring, stage and unstage blocks to and from pinned host memory, ring tables and restore', 'strata::kernels::kv_stream_reset, kv_stream_resolve, kv_ring_table, kv_ring_restore, kv_stage_from_host, kv_unstage_to_host', 'bytes copied as stored (FP16, INT8 or Q4_0)', 'long contexts; pages of 4 cells (docs/kv-streaming-design.md)'),
  ],
  paths=[IK + 'kv_q4.hpp', IK + 'kv_q8.hpp', IK + 'kv_stream.hpp'], extra=dict(authors=['Niko1221', 'Strata contributors', 'code-martin (PR #21)'])),
 'rope': dict(
  summary='rotary position embedding: NEOX partial RoPE (64 of 256 dimensions) with the YaRN-style scaling tables (rope, rope_scaling.hpp, mrope.hpp), and llama.cpp\'s text-only IMRoPE with the fused QSA norm + RoPE (native_rope)',
  why='every QSA layer rotates q and k, and the two kernels are the reference and the llama.cpp-pinned form of the same op that strixite\'s and gufo\'s pos.rope modules provide',
  files=[
   K(KC + 'rope.cu', 'pos.rope/-/f32.strata-neox', 'NEOX pairing (i, i + n_rot/2), partial rotation from host-built cos / sin tables', 'strata::kernels::rope_neox_apply, build_rope_table, rope_table_set', 'FP32; tables built in FP64 on the host', 'head 256, n_rot 64'),
   K(KC + 'native_rope.cu', 'pos.rope/-/f32.strata-imrope', 'llama.cpp\'s rope_multi for text positions (equal across the four IMRoPE sections), and the QSA RMSNorm fused with it', 'strata::kernels::native_rope_apply, native_qsa_rms_norm_rope', 'FP32; the pinned rope_yarn arithmetic', 'head 128 or 256, 64 rotated channels'),
  ],
  paths=[IK + 'rope.hpp', IK + 'native_rope.hpp', IK + 'mrope.hpp', IK + 'rope_scaling.hpp'], extra=dict(authors=GGML)),
 'ple': dict(
  summary='the per-layer embedding (PLE) block\'s GPU half (build_ple of the qwen4exp graph): sixteen gathered n-gram rows of 160 through the key / value projections, grouped norms and a dilated causal conv into a residual-stack update (ple), and llama.cpp\'s post-projection operations for it (native_ple_postops)',
  why='the PLE block is part of every Qwen3.8-Flash-Next layer and runs a token-history conv; the catalogue\'s ple.inject providers come from other models, so this is the Flash-Next form',
  files=[
   K(KC + 'ple.cu', 'ple.inject/-/f32.strata', 'the PLE block for one token: gathered rows, key / value projections, grouped norms, the conv over the history and the residual update', 'strata::kernels::ple_block(_projected), ple_history_advance', 'FP32 with BF16 or quantized projection weights', 'decode; 16 rows of 160, history 9, conv width 4 dilation 3'),
   K(KC + 'native_ple_postops.cu', 'ple.inject/-/f32.strata-native-postops', 'llama.cpp\'s build_ple after the two projections: norms, conv, reduction and the update, single token and batched', 'strata::kernels::native_ple_postops(_batch)', 'FP32, the pinned reduce_rows / sumrows order', 'embd 2560, streams 4, history 9'),
  ],
  paths=[IK + 'ple.hpp', IK + 'native_ple_postops.hpp', IK + 'ngram.hpp'], extra=dict(authors=GGML)),
 'elementwise': dict(
  summary='the glue kernels a layer needs between its GEMVs, in one file with one parity test: embedding gather from S-family codes, the GDN gate (softplus with ggml\'s large-x branch), scale, add, SiLU, FP32 to FP16 / BF16 casts and a weighted RMSNorm',
  why='glue written once with one float convention; the embedding gather and the GDN gate run every token',
  files=[
   K(KC + 'elementwise.cu', 'tensor.ops/-/f32.strata-glue', 'embedding_gather, gdn_gate, scale_inplace, add_inplace, silu_inplace, f32_to_f16_bulk, f32_to_bf16_bulk, rms_norm_weighted and helpers', 'strata::kernels::embedding_gather, gdn_gate, scale_inplace, add_inplace, f32_to_f16_bulk, f32_to_bf16_bulk, silu_inplace, rms_norm_weighted', 'FP32; compiled with -ffp-contract=off on HIP so the embedding gather keeps separately rounded operations', 'any length'),
  ],
  paths=[IK + 'elementwise.hpp'], extra=dict(source_flags={KC + 'elementwise.cu': ['-ffp-contract=off']})),
 'sampling': dict(
  summary='the sampler chain in llama.cpp\'s order (penalties, top-k, top-p, min-p, temperature, pick) on the GPU, a greedy cluster path, and the coupled draft sampler that stages and samples a speculative draft over a sub-vocabulary',
  why='sampling on the device keeps the logits off the bus; the chain order matches llama.cpp, which strixflerp\'s sample.next-token providers are judged against',
  files=[
   K(KC + 'sampler.cu', 'sample.next-token/-/f32.strata', 'penalties, top-k, top-p, min-p, temperature and pick, plus sample_greedy_cluster and the coupled-draft stage / sample', 'strata::kernels::sample_tokens, sample_greedy_cluster, coupled_draft_stage, coupled_draft_sample', 'FP32 logits; seeded pick', 'any vocabulary; Qwen3.8 151,936'),
  ],
  paths=[IK + 'sampler.hpp', IC + 'coupled_draft.hpp'], extra={}),
 'verify': dict(
  summary='kernels of the speculative verify window, where the last accepted token and T-1 drafts go through a layer at once: the GDN conv / alpha-beta / step-and-norm over T tokens with a commit that keeps only the accepted prefix, device flags to wait on, GPU timestamps and the resident-expert plan',
  why='the verify window must reproduce plain decode bit for bit (the per-token arithmetic is copied from fused_gdn and elementwise in the same order), which is the property strixflerp\'s speculation gates check',
  files=[
   K(KC + 'verify_kernels.cu', 'mixer.gdn/-/f32.strata-verify', 'gdn_conv_l2_multi, gdn_conv_commit, gdn_ab_multi, gdn_step_norm_multi, wait_flag_ge(_or), gpu_stamp, resident_plan', 'strata::kernels::gdn_conv_l2_multi, gdn_conv_commit, gdn_ab_multi, gdn_step_norm_multi, wait_flag_ge, gpu_stamp, resident_plan', 'FP32, bitwise equal to the single-token kernels by construction', 'verify windows of T tokens'),
  ],
  paths=[IK + 'verify_kernels.hpp'], extra={}),
 'prefill': dict(
  summary='batched prompt processing (plan v0.3 P5): native GGUF blocks to BF16 / F16 / F32 on the device (dequant_bf16), the chunk\'s projections Y = X W^T through hipBLAS / hipBLASLt with a solution table and the BF16-pair trick (gemm), the element-wise and stateful kernels of a prompt chunk (hyper-connection norm and mix, GDN conv and state, PLE conv, SwiGLU) (kernels), and an RDNA3 WMMA FP16 / BF16 GEMM (wmma_gemm, HIP only); src/prefill/gemm_bf16_parity.cu is a parity program',
  why='Strata\'s prompt path is the 762.5 tok/s cold 64K prefill ciru-ai/strataflash measured on Strix Halo, and wmma_gemm is a gfx11 WMMA GEMM beside strixite\'s linear_wmma and flerp\'s attn_wmma_gqa',
  files=[
   K(KC + 'dequant_bf16.cu', None, 'native GGUF blocks (Q2_0, Q4_0, Q5_0, Q8_0, Q3_K-Q6_K, IQ4_NL, IQ4_XS) to BF16, F16 or F32 rows, one thread a 32-element group, transcribed from llama.cpp\'s dequantize_row_*: a staging copy for the BLAS GEMMs', 'strata::kernels::dequant_bf16, dequant_f16(_ld), dequant_f32'),
   K(PF + 'gemm.cu', 'linear.prefill/bf16/bf16-f32.strata-blas', 'the prompt chunk\'s projections: BF16 or F16 weights (dequantized into a reusable scratch when native GGUF) times the activations through hipBLAS / hipBLASLt (tuned solution tables tools/hip/*-hipblaslt-*.txt), an exact path for the HC down projection, MMQ when built', 'strata::prefill::Gemm::bf16, f16, native, bf16_hcd_exact', 'BF16 / F16 products with FP32 output; STRATA_PREFILL_BF16X2 adds the BF16 residual of the activation for ~16 mantissa bits', 'prefill chunks of T tokens'),
   K(PF + 'kernels.cu', 'hc.mix-prefill/bf16/f32.strata-chunk', 'the prompt chunk\'s element-wise and stateful kernels: hyper-connection norm, mix, up-mix and write with the next norm (and the control vector), GDN conv and state walk, PLE conv, SwiGLU and casts', 'strata::prefill::gr_norm(_rs), gr_mix_r, gr_upmix, gr_write_norm_rs, gr_write_cvec_norm_rs, gr_silu and the GDN / PLE chunk kernels', 'FP32 with F16 / BF16 copies for the GEMMs; saturated SwiGLU in the FP16 path', 'prefill chunks; recurrences walk the chunk in order inside one launch'),
   K(PF + 'wmma_gemm.cu', 'linear.prefill/f16+bf16/f32.strata-wmma', 'Y = beta Y + X W^T on RDNA3 WMMA intrinsics for FP16 and BF16 operands (gfx11 only; STRATA_WMMA_GFX11 compiles the intrinsics, a device check decides whether they run)', 'strata::prefill::wmma_gemm_* (src/prefill/wmma_gemm.h)', 'F16 / BF16 operands, FP32 accumulation', 'prefill; gfx1100 tuned, gfx11 family'),
   K(PF + 'gemm_bf16_parity.cu', None, 'a stand-alone parity program for the BF16 GEMM path against a reference; not a library unit', 'main()'),
  ],
  paths=[IK + 'dequant_bf16.hpp', IP + 'gemm.hpp', IP + 'kernels.hpp', PF + 'wmma_gemm.h', PF + 'hipblaslt_tuning.hpp'],
  extra=dict(authors=GGML, source_flags={PF + 'gemm.cu': ['-DSTRATA_HIPBLASLT_AVAILABLE=1']})),
 'prefill_moe_fused': dict(
  summary='#136: the prompt path\'s routed experts of a Q2_0 pack on Strata\'s own int8 matrix-core kernels with the token grouping on the GPU (moe_fused); opt-in STRATA_PF_FUSED=1, built into strata_mmq only with -DSTRATA_PREFILL_MMQ=ON on HIP, where the gfx11 WMMA kernels run. Its native-GGUF sibling moe_fused_iq.cu includes llama.cpp\'s ggml.h and is in kernel.strata.prefill_moe_ggml',
  why='the routed experts are most of a prompt chunk\'s weight traffic; these group the (token, k) pairs on the device and read each expert once, beside the catalogue\'s moe.experts-prefill providers',
  files=[
   K(PF + 'moe_fused.cu', 'moe.experts-prefill/gguf-q2_0/i8-f32.strata-fused', 'grouping of the routed (token, k) pairs by expert on the GPU, int8 activation quantization, and the Q2_0 gate / up, SwiGLU and down products on int8 matrix cores', 'strata::prefill::fused::group, quantize_act, experts', 'int8 activations a 32-value block with an FP32 scale, Q2_0 weights d_w (q - 1); integer dots, FP32 output', 'prefill; gfx11 WMMA on HIP, sm_80+ on CUDA'),
  ],
  paths=[IP + 'moe_fused.hpp'],
  extra=dict(flags_extra=['-DGGML_USE_HIP=1'])),
 'prefill_moe_ggml': dict(
  summary='the prompt path\'s routed-expert sources that build against llama.cpp\'s ggml (the checkout the origin CMake fetches): prompt-speed plan step 2b, the experts through llama.cpp\'s MMQ kernels (ggml-cuda mmq.cuh) with expert gathers into a group buffer, q8_1 activations and one launch a product (moe_mmq), the few host symbols of ggml-cuda.cu that MMQ references (ggml_cuda_host), the resident-expert sort (mmq_resident_sort.hpp), and #136\'s fused int8 experts for native GGUF packs (moe_fused_iq, which takes the block types from ggml.h); opt-in -DSTRATA_PREFILL_MMQ=ON on HIP',
  why='MMQ keeps prompt experts quantized (one read of a 1.4-2 MB expert instead of 10 MB of FP16), the same idea as gufo\'s qfn_mmq module, and moe_fused_iq is the native-pack form of the fused int8 experts; all three need llama.cpp sources or headers the Strata tree does not contain',
  files=[
   K(PF + 'moe_mmq.cu', 'moe.experts-prefill/gguf-kquant/q8_1-f32.strata-mmq', 'expert gathers (one copy launch an expert as its blob arrives), q8_1 quantization of the routed activations and llama.cpp MMQ launches for the pack\'s types', 'strata::prefill::mmq::quantize, gather_native, Product::run, supported, fits', 'q8_1 activations, llama.cpp\'s MMQ integer tiles, FP32 output', 'prefill; types q2_0, iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq4_nl, iq4_xs, q8_0 (+ q5_0, K-quants by option)'),
   K(PF + 'moe_fused_iq.cu', 'moe.experts-prefill/gguf-iq+kquant/i8-f32.strata-fused', 'the same fused int8 experts for native GGUF expert slices, formats per layer from expert_layout', 'strata::prefill::fused::experts_native, quantize_act_native, native_supported', 'int8 activations; llama.cpp block layouts from ggml-common.h', 'prefill; gfx11 WMMA on HIP'),
   K(PF + 'ggml_cuda_host.cu', None, 'host-side ggml-cuda symbols (device info with GGML\'s AMD cc encoding, pool, logging) that MMQ and quantize reference, so strata_mmq links without the rest of ggml-cuda.cu', 'ggml_cuda_info and friends'),
  ],
  paths=[IP + 'moe_mmq.hpp', PF + 'mmq_resident_sort.hpp', IP + 'moe_fused_iq.hpp'],
  extra=dict(authors=GGML, compile='n/a', reason='needs llama.cpp\'s ggml at the commit in third_party/ggml/VERSION.txt, which the origin CMake fetches and the Strata tree does not contain: ggml/include/ggml.h (moe_fused_iq.cu) and ggml/src/ggml-cuda (common.cuh, mmq.cuh, quantize.cu and the mmq-instance-<type>.cu files; moe_mmq.cu, ggml_cuda_host.cu); the HIP build compiles these only with -DSTRATA_PREFILL_MMQ=ON; not compiled by this harvest')),
 'core': dict(
  summary='the .cu files of Strata\'s runtime core: the device arena, NaN-filled allocations and the runtime device check against the compiled architectures (device), pinned and device-mapped host memory with large-page and Windows paths (pinned), the remote-expert decode optimisation\'s row-skip kernels over the helper caches (remote_expert_opt), and tests/core/dma_batch_parity.cu, the batched-copy ordering test; with the core and CPU-expert headers they include',
  why='these are the HIP translation units of the runtime the kernels run in: the device check is what refuses a gfx1151 binary built for another target, and pinned memory carries the streamed experts and the KV ring',
  files=[
   K('src/core/device.cu', 'device.info/-/-.strata', 'DeviceArena (one allocation a planner region, bump sub-allocation), a NaN fill kernel, and the runtime device facts and check against STRATA_HIP_ARCHS', 'strata::core::DeviceArena, device_info, device_check', '', 'startup'),
   K('src/core/pinned.cu', None, 'pinned and device-mapped host buffers (registration, large pages, aliasing); no device kernel', 'strata::core::PinnedBuffer'),
   K('src/core/remote_expert_opt.cu', None, 'Remote Expert Decode Optimization v1: two kernels that skip rows already returned by the remote sum when copying mapped expert rows', 'strata::core::RemoteExpertOpt'),
   K('tests/core/dma_batch_parity.cu', None, 'a stand-alone test of the batched DMA copies\' ordering (flag order behind queued copies); not a library unit', 'main()'),
  ],
  paths=[IC + 'device.hpp', IC + 'pinned.hpp', 'include/strata/platform/memory.hpp', IC + 'remote_expert_opt.hpp',
         IC + 'remote_experts.hpp', IC + 'expert_cache.hpp', IC + 'expert_source.hpp', IC + 'exchange_storage.hpp', IC + 'hit_hook.hpp',
         IC + 'on_device.hpp', IC + 'vmm.hpp', IC + 'dma_batch.hpp', IK + 'cpu/expert.hpp', IK + 'cpu/expert_layout.hpp',
         IK + 'cpu/native_expert.hpp', IK + 'cpu/pool.hpp'],
  extra=dict(source_skip=['tests/core/dma_batch_parity.cu'])),
}
TESTS = {PF + 'gdn_rec_parity.cu', PF + 'gemm_bf16_parity.cu', 'tests/core/dma_batch_parity.cu'}
COMMON = 'module:kernel.strata.common'


def strip_tag(tag):
    """The interface class: the full name without its variant tag (the tag rides on the dtype field)."""
    head, _, last = tag.rpartition('/')
    return head + '/' + last.split('.')[0]


def provides_for(fam, files):
    out = []
    for k in files:
        if k['tag']:
            out.append(P(k['tag'], f"{k['api']} ({k['file']})", k['numeric'] or 'see the file header', k['regime'] or 'see the file header'))
    return out


mods = []
for fam, spec in FAMILIES.items():
    extra = dict(spec['extra'])
    files = spec['files']
    flags_extra = extra.pop('flags_extra', None)
    skip = set(extra.pop('source_skip', [])) | TESTS
    tagged = [k for k in files if k['tag']]
    untagged = [k for k in files if not k['tag']]
    tagnote = '; '.join(f"{os.path.basename(k['file'])} {strip_tag(k['tag'])}" for k in tagged)
    nonenote = ', '.join(os.path.basename(k['file']) for k in untagged)
    notes = extra.pop('notes', None)
    if notes is None:
        notes = ('Interface tags by file: ' + (tagnote or 'none')) + (f'; none: {nonenote}' if nonenote else '') + \
            '. catalog/strata/KERNELS.md has the table. ' + NOT_TIMED
    sources = [k['file'] for k in files if k['file'].endswith('.cu') and k['file'] not in skip]
    provides = extra.pop('provides', None) or provides_for(fam, files)
    kw = dict(provides=provides, requires=[] if fam == 'common' else [COMMON], sources=sources if extra.get('compile', 'standalone') == 'standalone' else [],
              includes=extra.pop('includes', ['src/include']), why=spec['why'], notes=notes)
    if flags_extra:
        kw['source_flags'] = {s: flags_extra for s in sources}
    sf = extra.pop('source_flags', None)
    if sf:
        kw['source_flags'] = {**kw.get('source_flags', {}), **sf}
    kw.update(extra)
    mods.append(M('kernel.strata.' + fam, spec['summary'], [k['file'] for k in files] + spec['paths'], **kw))

# every .cu / .cuh of the engine is in a family module
cus = sorted(f for f in O.files() if f.endswith(('.cu', '.cuh')) and not f.startswith(('docs/', 'bench/results/')))
declared = sorted([k['file'] for s in FAMILIES.values() for k in s['files']] + [KC + 's26_tsum.cuh'])
if cus != declared:
    raise SystemExit('HIP kernel sources differ from FAMILIES: ' + ' '.join(sorted(set(cus) ^ set(declared))))

ORIGIN_TOML = '''
# Strata: a C++ engine for Qwen3.8-Flash-Next on one consumer GPU (CUDA, HIP for wave32 RDNA, SYCL, CPU), the routed
# experts streamed from host memory into a CPU pool and a GPU expert tier; an OpenAI / Anthropic-compatible server.
# Its GPU kernels are CUDA-shaped .cu files that the HIP build compiles as HIP through a force-included compatibility
# header (card ORIGIN-STRATA ef16e87953).
name      = "strata"
repo      = "https://github.com/Niko1221/Strata"
commit    = "1cbcacbcae2953f3be9edc46369f0c875bc6ab8b"
branch    = "main"
version   = "release tag v0.1.40, the build ciru-ai/strataflash measured on Strix Halo: 75.8 tok/s decode, 762.5 tok/s cold 64K prefill, tools 77/100 (card ef16e87953); main HEAD 2af30cc (v0.2.21.1) is also in the ark and was not harvested"
reference_checkout = "control-node:/data/ark-inbox/src/github.com__Niko1221__Strata#1cbcacbcae2953f3be9edc46369f0c875bc6ab8b (ark copy: the tree at the commit, .git without objects, ark 7546d7e); laptop: ~/Code/strixflerp-upstream-supplement/strata (the same files, by tarball through control-node get); docs/harvest/harvest_strata.py pins the tree digest"

[licence]
spdx      = "MIT"
files     = ["LICENSE"]                                  # copied verbatim next to this file
third_party = "LICENSE: MIT, Copyright (c) 2026 Niko1221 and the Strata contributors. third_party/ggml (ggml-common.h, VERSION.txt, LICENSE) is llama.cpp's, MIT, The ggml authors (in kernel.strata.common). The native_*.cu kernels, iq_kernels.cu and dequant_bf16.cu say in their headers that they are adapted or transcribed from llama.cpp 3cf03257 (MIT) and carry its copyright line; kv_q4.cu is from PR #21 by code-martin. src/prefill/moe_mmq.cu compiles llama.cpp's MMQ sources, which are not in the tree. ciru-ai/strataflash (the measurement repo) has no licence file and was not harvested. Weights are not part of the tree."

[toolchain]
host      = "C++20 (CMAKE_CXX_STANDARD 20), CMake with the HIP language; the HIP sources get no -std of their own (no HIP_STANDARD is set), so clang's default applies"
device    = "therock-10.2"                               # hipBLAS, hipBLASLt and rocBLAS from the same ROCm; the origin names ROCm 7.x / HIP SDK 7.2 (Windows) and leaves the toolkit to the user
build     = "cmake -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 (cmake/hip_backend.cmake: every src/**/*.cu, bench/**/*.cu and tests/**/*.cu gets LANGUAGE HIP; strata_hip_runtime adds -include include/strata/hip_compat/cuda_runtime.h, include/strata/hip_compat before include/, STRATA_USE_HIP=1, STRATA_HIP_ARCHS, STRATA_WMMA_GFX11=1 for gfx11 targets); Release (-O3 -DNDEBUG), position-independent code; libraries strata_core, strata_kernels, strata_prefill (+ strata_mmq with -DSTRATA_PREFILL_MMQ=ON) link into the strata program. gfx1151 is in the unvalidated list at v0.1.40 (configure warns)"

[compile]                                                # CMake's HIP language: clang++ -x hip, one object per .cu
driver    = "<rocm>/lib/llvm/bin/clang++"
arch_flag = "--offload-arch=gfx1151"
rocm_flag = "--rocm-path=<rocm>"
flags     = ["-x", "hip", "-O3", "-DNDEBUG", "-fPIC", "-DSTRATA_USE_HIP=1", "-DSTRATA_HIP_ARCHS=\\"gfx1151\\"", "-DSTRATA_WMMA_GFX11=1", "-include", "cuda_runtime.h", "-c"]
system_includes = ["include"]
output    = "host object with the gfx1151 device code embedded (.o), one per translation unit"
gate      = "gate/compile/run.py --origin strata on control-node's CPU, 2026-11-24, therock-10.2: 16 of 16 standalone modules and 51 of 51 units compile for gfx1151 (gate/results/compile/kernel.strata.*.json); kernel.strata.common is header-only and kernel.strata.prefill_moe_ggml needs llama.cpp's ggml, so neither is compiled"
note      = "-include cuda_runtime.h resolves through the -I list to kernel.strata.common's include/strata/hip_compat, as the origin's absolute -include path does. Per-source options from CMakeLists.txt are in [build] source_options: -ffp-contract=off for elementwise.cu and quantize_act.cu, STRATA_HIPBLASLT_AVAILABLE=1 for gemm.cu (defined when hipBLASLt is found, as in TheRock), GGML_USE_HIP=1 for the strata_mmq sources. -Wall -Wextra (strata_warnings) are left out."

[kernels]
loading   = "statically linked: each .cu holds its kernels and the host launchers declared in include/strata/kernels/*.hpp and include/strata/prefill/*.hpp; launches go to an explicit stream; the decode step is captured into a HIP graph (src/core/graph.cpp). No code-object kit or kernel registry"
selection = "the layer and session code pick a kernel by format, shape and switch: STRATA_* environment variables and set_enabled() calls choose the llama.cpp-pinned native_* kernels or Strata's own, fused or split forms (docs/)"
models    = "Qwen3.8-Flash-Next (qwen4exp in llama.cpp terms: 48 layers, 36 GDN and 12 QSA, 512 experts top 10, 4-stream hyper-connections, PLE) from Strata packs (Q2_0 as S2) or native GGUF (IQ2_XS, IQ3_XXS, Unsloth UD-Q4_K_XL); README.md"
'''

LEFT_OUT = {
    'src/core/*.cpp, src/core/*_main.cpp, src/artifact, src/plan, src/platform, src/program, src/spec, src/ngram, src/kernels/ngram.cpp': 'the host engine: session, layer graph, expert sources and caches, the GGUF / pack reader, the planner, generation and speculative control; they launch the kernels but hold none',
    'src/prefill/prefill.cpp': 'the prompt path\'s host orchestration over kernel.strata.prefill and the MoE prompt modules',
    'src/kernels/cpu, include/strata/kernels/cpu (except the four headers remote_expert_opt.cu includes)': 'the CPU expert pool and its AVX / AVX2 / AVX-512 kernels: not HIP',
    'src/kernels/*_parity.cpp, *_bench.cpp, *_test.cpp, tests/, src/core/*_test.cpp': 'parity programs, benches and unit tests (host .cpp, some compiled as HIP); the .cu test programs are in their family modules',
    'sycl/': 'the SYCL backend (Intel GPUs)',
    'serve/, chat.py, setup.py, setup.sh, SETUP.bat, START-HERE.bat, update.sh, UPDATE.bat, Dockerfile, docker-entrypoint.sh, requirements.txt': 'the server, the chat client, installers and container',
    'tools/, bench/, ref/, data/': 'tools (including the hipBLASLt tuning tables and tuner), benches with recorded results, the Python reference model the kernels were transcribed from, and data files',
    'docs/, README*.md, AGENTS.md, SECURITY.md': 'documentation; read for the module records',
    'CMakeLists.txt, cmake/, .github/, .gitignore, .gitattributes, .dockerignore': 'build system and repository plumbing (the HIP compile options are recorded in ORIGIN.toml [compile])',
    'LICENSE': 'copied beside ORIGIN.toml',
}


def providers():
    """{module id: [interface, ...]} of the catalogue outside this origin."""
    out = {}
    cat = os.path.join(harvestlib.REPO, 'catalog')
    for origin in sorted(os.listdir(cat)):
        if origin == O.name or not os.path.isdir(os.path.join(cat, origin)):
            continue
        for d, _, fs in os.walk(os.path.join(cat, origin)):
            if 'module.toml' in fs:
                t = tomllib.load(open(os.path.join(d, 'module.toml'), 'rb'))
                out[t['id']] = [x.get('interface', '') for x in t.get('provides', [])]
    return out


def same_op(tag, prov):
    """Modules of other origins that provide the tag's <family>.<op> (with the tag moved into the op when the dtype
    is a dash, as harvestlib.iface does)."""
    op = tag.split('/')[0]
    hits = []
    for mid, ifs in sorted(prov.items()):
        for i in ifs:
            first = i.split('/')[0]
            if first == op or (i.endswith('/-') and first.startswith(op + '-')):
                hits.append(mid)
                break
    return hits


def kernels_md():
    """catalog/strata/KERNELS.md: every HIP kernel source, each with its interface tag."""
    prov = providers()
    bt = '\x60'
    o = ['# Strata kernels by strixflerp interface', '',
         'One row for each HIP kernel source of Strata at v0.1.40 (1cbcacbcae29). Card ORIGIN-STRATA ef16e87953. Written by',
         bt + 'docs/harvest/harvest_strata.py' + bt + ' from its ' + bt + 'FAMILIES' + bt + ' table, read from the sources and their headers.', '',
         '- **tag**: the strixflerp interface class the file matches, ' + bt + '<family>.<op>/<weight format>/<dtype>' + bt + ', or none. The',
         '  module\'s ' + bt + '[[provides]]' + bt + ' entry carries the full name with a ' + bt + '.strata' + bt + ' variant tag.',
         '- **same op in the catalogue**: kernel modules of other origins that provide that ' + bt + '<family>.<op>' + bt + '. The same op is',
         '  not the same operands: every Strata kernel takes raw device pointers in Strata\'s layouts (S-family planes, the QSA',
         '  page table, the 4-stream residual), so each row needs an adapter before it can stand in for a strixflerp module.', '',
         '| file | lines | module | tag | same op in the catalogue |', '|---|---:|---|---|---|']
    counts, untagged, total = {}, [], 0
    for fam, spec in FAMILIES.items():
        for k in spec['files']:
            lines = len(open(O.find(k['file']), errors='replace').read().splitlines())
            hits = same_op(k['tag'], prov) if k['tag'] else []
            counts[fam] = counts.get(fam, 0) + 1
            total += 1
            if not k['tag']:
                untagged.append(k['file'])
            shown = ', '.join(bt + h + bt for h in hits[:6]) + (f' and {len(hits) - 6} more' if len(hits) > 6 else '')
            o.append('| %s%s%s | %d | %skernel.strata.%s%s | %s | %s |' % (
                bt, k['file'], bt, lines, bt, fam, bt, (bt + strip_tag(harvestlib.iface(k['tag'])) + bt) if k['tag'] else 'none', shown or '-'))
    o += ['', '%d files: %s. %d carry a tag, %d are none (%s).'
          % (total, ', '.join('%s %d' % (f, counts[f]) for f in counts), total - len(untagged), len(untagged),
             ', '.join(bt + os.path.basename(u) + bt for u in untagged)),
          bt + 'src/kernels/cuda/s26_tsum.cuh' + bt + ' (the S26 tree sum, a device header) is in ' + bt + 'kernel.strata.common' + bt + '.', '',
          '## What each kernel does', '']
    for fam, spec in FAMILIES.items():
        for k in spec['files']:
            o.append('- %s%s%s: %s.' % (bt, k['file'], bt, k['what']))
    o += ['', '## Not done here', '',
          'No kernel was run or timed on the pod. The compile gate (' + bt + 'gate/compile/run.py --origin strata' + bt + ', control-node CPU, gfx1151)',
          'compiles the standalone modules; ' + bt + 'kernel.strata.prefill_moe_ggml' + bt + ' needs llama.cpp\'s ggml headers and ggml-cuda sources and is not compiled.',
          'A timing against the strixite modules is its own experiment card.']
    open(os.path.join(harvestlib.REPO, 'catalog', O.name, 'KERNELS.md'), 'w').write('\n'.join(o) + '\n')


if __name__ == '__main__':
    run(O, mods, ORIGIN_TOML, LEFT_OUT, ['LICENSE'])
    kernels_md()
