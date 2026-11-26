#!/usr/bin/env python3
"""Harvest radiance (codeberg.org/StillDeadcode/radiance, 38c182a) into catalog/radiance.
Run: python3 docs/harvest/harvest_radiance.py

Card ORIGIN-RADIANCE b0c1db5993. Scope: libhalo, radiance's kernel library for Strix Halo (gfx1151), and the plugin
ABI headers its units include (abi/). The engine core, the model plugins (arch/), the other kernel libraries
(libr4d, libr3d, libavx, libref), the quantisers and the tests are listed as unharvested.

libhalo units are ported from libr4d (gfx1201) by a rename transform and then re-measured on gfx1151
(libhalo/PORTING.md). Each .hip is one translation unit of libhalo.so, compiled by radiance's CMake
(rad_hipify, -ffp-contract=off) against abi/; nothing is compiled by this harvest, so every module is
[gate] status = "harvested".

The reference checkout is the ark copy on control-node (/data/ark-inbox/src/codeberg.org__StillDeadcode__radiance);
the laptop holds libhalo/, abi/, LICENSE, NOTICE and the origin's file list (git ls-files, radiance-files.txt) in
~/Code/strixflerp-upstream-supplement/radiance. RADIANCE_CHECKOUT names another copy (the full ark checkout works)."""
import os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvestlib
from harvestlib import Origin, M, P, run

CHECKOUT = os.environ.get('RADIANCE_CHECKOUT', os.path.join(harvestlib.SUPPLEMENT, 'radiance'))
COMMIT = '38c182a2baa35e92bc1104460d7410b0daf2439a'


class ListedOrigin(Origin):
    """The origin's file set is git's (git ls-files of the checkout, or the saved list beside a partial copy)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.roots = [CHECKOUT]

    def files(self):
        if self._files is None:
            lst = os.path.join(CHECKOUT, 'radiance-files.txt')
            if os.path.isdir(os.path.join(CHECKOUT, '.git')):
                out = subprocess.run(['git', '-C', CHECKOUT, 'ls-files'], capture_output=True, text=True, check=True).stdout
            else:
                out = open(lst).read()
            self._files = sorted(f for f in out.splitlines() if f)
        return self._files


O = ListedOrigin('radiance', 'radiance', 'https://codeberg.org/StillDeadcode/radiance', COMMIT, 'main', 'Apache-2.0',
                 ['Deadcode', 'radiance contributors'], 'therock-10.1')
L = 'libhalo/'
NA = 'a translation unit of libhalo.so: built by radiance CMake (rad_hipify, -ffp-contract=off) against abi/, not compiled by this harvest'
DEC = 'decode and verify heights (1..64 rows), a weight stream bound by ~256 GB/s LPDDR5X'
PF = 'prefill row counts (tiled), compute-bound on the wave32 WMMA cores'
mods, claimed = [], set()


def take(names):
    out = []
    for n in names:
        hits = [f for f in O.expand([n if '/' in n else L + n]) if f not in claimed]
        if not hits:
            raise SystemExit(f'nothing left for {n}')
        out += hits
    claimed.update(out)
    return out


def mod(id, summary, names, provides, notes=None, lang='hip-c++', authors=None, compile='n/a', reason=NA, requires=()):
    files = take(names)
    ext = ('.hip',) if lang == 'hip-c++' else ('.cpp',)
    sources = [f for f in files if f.endswith(ext)] if compile != 'header-only' else []
    incs = sorted({'src/' + os.path.dirname(f) for f in files if f.endswith(('.h', '.hip', '.cpp'))})
    mods.append(M(id, summary, files, provides=provides, requires=requires, lang=lang, sources=sources, includes=incs,
                  compile=compile, reason=reason, notes=notes, authors=authors, why=WHY[id],
                  toolchain='' if lang == 'python' else None))


# Why each module exists (SPEC.md section 5; tools/describe-audit.py wants one per module, none shared).
WHY = {
 'runtime.radiance.plugin_abi': 'radiance keeps model plugins and kernel libraries apart behind one C ABI so a card gets its own kernel library without touching a model; libhalo cannot be read or compiled without these headers',
 'kernel.radiance.halo_common': 'gfx1151 and gfx1201 disagree on the WMMA fragment order and gfx1151 lacks the fp8 and bf16 pack instructions, so every ported unit needs one measured layer that states the lane-to-element rules once',
 'kernel.radiance.halo_chain': 'a decode step is hundreds of small dependent launches and each AQL barrier leaves the GPU idle about 2 us; a chain moves the ordering into the kernels so the queue never waits',
 'kernel.radiance.halo_hc': 'the gated residual is read and written around every block (97 mixes a Flash-Next token), so its bytes and its launch count are a fixed tax on decode that a second hand-written gfx1151 implementation lets us compare against strixite\'s',
 'kernel.radiance.halo_mhc': 'GLM and DeepSeek-V4.1 mix residual streams with a Sinkhorn-normalised matrix rather than Qwen\'s low-rank pair, a different kernel shape that the pod meets on StrixFold',
 'kernel.radiance.halo_qsa': 'sparse attention only pays if choosing the blocks is cheap and its output is directly what attention reads; splitting the indexer into five ops lets decode run only the ones a step needs',
 'kernel.radiance.halo_kv': 'with indexer-selected attention a step reads a small part of the context, so the full KV can be parked and only a ring kept resident; that frees memory for more sequences or bigger experts',
 'kernel.radiance.halo_gdn': 'three of every four Flash-Next layers are gated delta nets, with a recurrence that is a per-token update at decode and a chunked triangular solve at prefill; each phase needs its own kernel to reach the part\'s limits',
 'kernel.radiance.halo_moe': 'routed experts are most of the weight bytes a token reads; sorting the routed rows by expert lets one launch read each expert once for every row and stream that picked it',
 'kernel.radiance.halo_router': 'the router is tiny work behind a full launch, so radiance folds its GEMM, selection and slot sort into one launch to stop paying the launch floor three times a layer',
 'kernel.radiance.halo_gemm_int': 'decode on Strix Halo is weight bytes over memory bandwidth, so integer weights at 2 to 8 bits with integer activations are how radiance buys speed and how it fits a 113 GiB model in a rank\'s memory',
 'kernel.radiance.halo_gemm_bf16': 'unquantised projections remain for protected experts, small models, biased layers and LoRA, and they are the reference every quantised GEMM is checked against',
 'kernel.radiance.halo_gemm_fp8': 'most published checkpoints ship block-scaled FP8, and gfx1151 has no fp8 matrix instruction, so serving them needs an exact widen to f16 inside the GEMM instead of a converted copy of the weights',
 'kernel.radiance.halo_gemm_mxfp4': 'MXFP4 is the 4-bit float form vendors publish for MoE experts; reading it directly halves the bytes of an 8-bit stream at decode without a requantisation step of our own',
 'kernel.radiance.halo_quant_act': 'an integer GEMM needs integer activations, and a separate quantiser launch per junction would cost more than the GEMM saves, so the producer that already holds the row writes the codes',
 'kernel.radiance.halo_attn': 'attention kernels are shape-specialised (head size, grouped-query ratio, KV dtype, window), and each shape had to be re-derived for gfx1151\'s fragment order; this is the complete set for the models radiance serves',
 'kernel.radiance.halo_allreduce': 'tensor parallelism spends its time in collectives, so radiance runs them as device kernels with a compressed wire and fuses the work that always follows a reduce',
 'kernel.radiance.halo_sample': 'sampling on the host forces a logits copy and a round trip every token; drawing on the device keeps tokens there and lets speculative sampling accept or reject without leaving the stream',
 'kernel.radiance.halo_ple': 'Qwen4-Exp adds a hashed n-gram embedding at one layer; computing the row ids and the gate on the device keeps the table lookup off the host path of a decode step',
 'kernel.radiance.halo_dflash': 'a block-diffusion drafter proposes a whole block in one pass, a different speculation shape from the MTP head, and needs its own convolution and path-selection kernels',
 'kernel.radiance.halo_model_ops': 'every op in a step needs a device launch because buffers never visit the host, so the many tiny ops are collected and, where a model issues hundreds of them, joined into single kernels',
 'kernel.radiance.halo_glm_family': 'GLM-5.3 and Ling-3.0 use latent attention, KDA recurrence and sigmoid routing; the kernels are kept so those families can be compared with StrixFold\'s when the pod serves one',
 'kernel.radiance.halo_media': 'radiance serves image, video and audio models through the same kernel library; kept for the vision tower and for any media recipe the factory later needs',
 'runtime.radiance.halo_rows': 'selection by declared rows with measured bands is how one engine picks the right kernel per shape and how a half-ported library still loads and reports its gaps by name',
 'tool.radiance.halo_selftest': 'a kernel library ported by renaming is only trusted through cases recorded once and replayed against every library, plus a check that the port still answers the same interface',
}

COMMON = 'module:kernel.radiance.halo_common'

mod('runtime.radiance.plugin_abi',
    'radiance\'s plugin ABI headers: the op and row declarations a kernel library answers (rad_abi.h, rad_plugin.h), the chained-launch record type (RadKernelChainFn), weight layout and encoding tags, the Philox draw, the sampler and YaRN contracts',
    ['abi/'], [P('kernel-abi.radiance-plugin/-/-', 'RadArgs, RadStream, RadKernelInfo rows, rad_plugin_open (abi/rad_abi.h, abi/rad_plugin.h)', 'n/a', 'n/a')],
    lang='c++', compile='header-only', reason='headers only; every libhalo unit includes them')

mod('kernel.radiance.halo_common',
    'libhalo\'s shared device layer for gfx1151: the plain C entry points (halo.h), argument readers, the one launch path (halo_emit.h), wave32 WMMA fragment algebra and 16-bit operand traits (16 x 16-bit fragments, k = e, lanes 16-31 replicate), software bf16 and E4M3 packs for the instructions gfx1151 lacks, a wave-local Walsh-Hadamard and the register-blocked f32 tile',
    ['halo.h', 'halo_args.h', 'halo_common.h', 'halo_emit.h', 'halo_launch.h', 'halo_enc.h', 'halo_wmma.h', 'halo_dt16.h',
     'halo_fwht.h', 'halo_q8.h', 'halo_f32_tile.h', 'halo_plugin.h'],
    [P('kernel-abi.radiance-halo/-/-', 'halo_* C entries (halo.h), halo_rad_<row> shims (halo_plugin.h), halo::mma, DT16<0|1>, halo::pk_bf16', 'n/a', 'n/a')],
    compile='header-only', reason='headers only; compiled inside every kernel.radiance.halo_* unit',
    requires=['module:runtime.radiance.plugin_abi'])

mod('kernel.radiance.halo_chain',
    'Chained launches: a kernel launched without the AQL barrier bit waits in its own prologue on a completion counter of the kernels before it and adds to it when done (the r4dchain name marker is what the AQL backend keys on)',
    ['halo_chain.h'],
    [P('runtime.chain/-/-.radiance', 'halo_chain.h wait / done bracket; chained forms are named *_r4dchain_* per family', 'none: schedule only', 'decode steps of many small dependent launches')],
    notes='The shape behind runtime.flerp.chain (CHAIN-0 4a97e0f695). CHAIN-1 40d8943aa2 measured it on strixite\'s real decode kernels: stalls at the fill that would pay, +0.16 ms a step at the fill that ran clean (/data/pod/reports/2026-11/chain1.md).',
    compile='header-only', reason='header only', requires=[COMMON])

mod('kernel.radiance.halo_hc',
    'Qwen4-Exp\'s gated residual (hyper-connections) in four ops: hc_enter, hc_read (grouped RMS norm of the sub-streams, the low-rank mixing pair on f16 WMMA, bf16 or E4M3 weights), hc_write and hc_project; plus mtp_enter, the MTP head\'s entry over the same stored rows. In hc_write the last of a group\'s workgroups to arrive (an arrival counter) does that group\'s quantisation',
    ['halo_hc_bf16.hip', 'halo_hc_read.hip', 'halo_hc_f16.h', 'halo_hc_fp8.h', 'halo_mtp_enter.hip'],
    [P('hc.mix/radiance-e4m3+bf16/bf16', 'halo_hc_enter, halo_hc_read_bf16 / _e4m3, halo_hc_write, halo_hc_project', 'fp32 accumulate over f16 WMMA operands; an E4M3 byte widened exactly to f16; one rounding to bf16 on store', DEC + '; the same rows at prefill'),
     P('draft.mtp-enter/radiance-e4m3+bf16/bf16', 'halo_mtp_enter_bf16 / _e4m3', 'as hc_read', 'one launch a draft step')],
    notes='Compare kernel.strixite.hc (Q8 G64 mixes, 97 down + 97 up launches a decode step, 3.88 ms of 26.9 ms: hc-decode.md) and HC-Q4 89f1c108d8 (Q4 mixes fail fidelity, KL 0.123). The arrival-counter fold is the pattern FOLD-1 75f1d18d2f / FOLD-2 8c7f1aab87 use.',
    requires=[COMMON])

mod('kernel.radiance.halo_mhc',
    'GLM\'s manifold hyper-connections: parallel projections, then one workgroup a token forms the Sinkhorn mixing matrix; f32 mixing weights',
    ['halo_mhc.hip'],
    [P('hc.mix-decode/glm-mhc-f32/bf16', 'halo_mhc_* (halo_mhc.hip)', 'f32 Sinkhorn iterations in one workgroup a token', 'decode and prefill rows')],
    notes='Same job as kernel.strixfold.mhc_mix (DeepSeek-V4.1 mHC, 80 mix sites); SPEED-FUSE-1 b7269694f8 fuses StrixFold\'s chain into one launch a site.',
    requires=[COMMON])

mod('kernel.radiance.halo_qsa',
    'The QSA indexer of Qwen3.8-Flash-Next\'s full-attention layers as five ops: the work list of blocks completing this step, the compressed key of each 4-token block (128 dims), the block score (sum over index heads of relu(q.bk)), the exact top-k selection written as the page-id table the paged attention reads, and the raw-key tail of tokens whose block is not complete',
    ['halo_qsa_work_bf16.hip', 'halo_qsa_block_key_bf16.hip', 'halo_qsa_score_bf16.hip', 'halo_qsa_select_bf16.hip', 'halo_qsa_tail_store_bf16.hip'],
    [P('attn.qsa-indexer/-/bf16.radiance', 'halo_qsa_work, halo_qsa_block_key, halo_qsa_score, halo_qsa_select, halo_qsa_tail_store', 'bf16 operands, fp32 accumulate; exact top-k', 'decode (one query) and prefill (a chunk of queries); selection feeds attn page ids directly')],
    notes='V10-QSA 8ed7cc2c42 measured strixite\'s indexer at 0.8-1.7 % of decode GPU time and selection stability at Jaccard 0.43-0.65 token to token: cost is not the lever; the page-id output is what kv_resolve (kernel.radiance.halo_kv) streams KV by.',
    requires=[COMMON])

mod('kernel.radiance.halo_kv',
    'KV streaming on the device: kv_resolve turns a streamed group\'s selected block ids into the ring slots that hold them, copies an absent block in from the pinned host mirror and rewrites the selection in place; kv_len_trim gives per-sequence context lengths less a trim',
    ['halo_kv_resolve.hip', 'halo_kv_len_trim.hip'],
    [P('kvcache.tiering/-/-.radiance-stream', 'halo_rad_kv_resolve, halo_rad_kv_len_trim', 'byte moves; the selection decides what is resident', 'indexer-selected attention only: a step reads far less than the context (docs/KV-STREAMING.md in the origin)')],
    notes='On unified memory the host mirror and the ring are the same RAM pool, so the win on Strix Halo is not VRAM; the shape that carries over is a per-slot resident ring with the full context parked, which is what a slot switch needs.',
    requires=[COMMON])

mod('kernel.radiance.halo_gdn',
    'Gated delta net (the 36 linear-attention layers of Qwen3.8-Flash-Next): the depthwise causal convolution with the work fused around it per serving phase, the chunk preamble as one kernel (the (I + strict_lower(...))^-1 solve, chunk 64), the chunked scan on WMMA for prefill, the recurrent state update for decode (fp32 state), and the gated RMS norm alone or fused with the block Hadamard and int8 quantiser that feeds out_proj',
    ['halo_gdn_args.h', 'halo_gdn_wmma.h', 'halo_gdn_conv_tap.h', 'halo_gdn_conv_w4_h128_bf16.hip', 'halo_gdn_kkt_solve.hip',
     'halo_gdn_chunk_scan.hip', 'halo_gdn_recurrent_update_k128_v128_bf16_fp32state.hip', 'halo_gdn_gated_rmsnorm_h128_bf16.hip',
     'halo_gdn_gated_norm_had_quant_i8.hip'],
    [P('mixer.gdn/-/bf16.radiance', 'halo_gdn_conv_*, halo_gdn_kkt_solve_k128_c64_bf16, halo_gdn_chunk_scan_k128_v128_c64_bf16, halo_gdn_recurrent_update_k128_v128_bf16_fp32state', 'bf16 in, fp32 state; chunked scan on f16 WMMA with overflow guards', 'prefill: chunks of 64; decode: one recurrent update a token'),
     P('norm.rmsnorm-gated/-/bf16.radiance', 'halo_gdn_gated_rmsnorm_h128_bf16, halo_gdn_gated_norm_had_quant_i8', 'fp32 sum of squares; the fused form writes rotated int8 codes', 'one launch a GDN layer')],
    notes='strixite: kernel.strixite.gdn (delta_rule one block per value head), rmsnorm_gated and gdn_front as separate launches (FOLD-2 8c7f1aab87 folds them); the GDN input projections are 3.94 ms of a 32.9 ms decode step (strixite-perf-ranking.md).',
    authors=['Deadcode', 'ggz14', 'radiance contributors'], requires=[COMMON])

mod('kernel.radiance.halo_moe',
    'Routed experts: the stable counting sort from per-token expert ids to an expert-sorted slot list and its inverse, the weighted gather back, and the grouped expert GEMM at every routed height in three weight forms: bf16, quantised (E4M3 or int codes decoded to f16 for the f16 WMMA, since gfx1151 has no fp8 matrix instruction) and OCP-MXFP4; plus the per-expert and shared-expert row scales',
    ['halo_moe.hip', 'halo_moe_sort.h', 'halo_moe_gather.h', 'halo_moe_gemm_bf16.hip', 'halo_moe_gemm_q.hip', 'halo_moe_mxfp4.hip',
     'halo_expert_scale_bf16.hip', 'halo_scale_rows_bf16.hip'],
    [P('moe.experts-decode/radiance-q+mxfp4/bf16', 'halo_moe_gemm_q, halo_moe_gemm_mxfp4a8 at decode heights; halo_moe_scatter, halo_moe_gather', 'fp32 accumulate; a row\'s bits do not depend on the height it is batched at', DEC + '; rows of several streams sorted by expert into one launch'),
     P('moe.experts-prefill/radiance-q+mxfp4/bf16', 'halo_moe_gemm_q, halo_moe_gemm_bf16, halo_moe_gemm_mxfp4a8', 'as decode', PF),
     P('moe.combine/-/bf16.radiance', 'halo_moe_gather (weighted gather of slot rows back to tokens)', 'fp32 sum in slot order', 'one launch a layer')],
    notes='The expert-sorted slot list is the batching shape MULTI-DECODE defc70210f needs: S streams\' routed rows share one grouped launch and each expert\'s weights are read once for all rows that picked it.',
    authors=['Deadcode', 'ggz14', 'slydlake', 'keptn42', 'radiance contributors'], requires=[COMMON])

mod('kernel.radiance.halo_router',
    'MoE routing: softmax plus exact top-k a wave a row, sigmoid routing with a correction bias (GLM), and the fused forms: selection and the slot sort in one workgroup, and the router GEMM, its top-k, the slot sort and the shared expert\'s gate in one launch',
    ['halo_router_topk.h', 'halo_router_topk_bf16.hip', 'halo_router_sigmoid.h', 'halo_router_sigmoid_topk_f32.hip',
     'halo_router_scatter.hip', 'halo_router_gemm_scatter.hip'],
    [P('moe.router/-/bf16.radiance', 'halo_router_topk, halo_router_topk_scatter, halo_router_gemm_scatter, halo_router_sigmoid_topk', 'bf16 logits, exact top-k, ties to the lower id; the fused rows share one row body with the plain ones', 'decode: one launch a layer for GEMM + top-k + sort')],
    notes='strixite runs router GEMV, top-k and the grouping as separate launches (48 router launches a step; FOLD-2 folds the top-k into the last block of the router GEMV).',
    authors=['Deadcode', 'keptn42', 'radiance contributors'], requires=[COMMON, 'module:kernel.radiance.halo_moe'])

mod('kernel.radiance.halo_gemm_int',
    'Integer-weight GEMMs: 2-, 4-, 5- and 8-bit weights against int8 activations (w2a8, w4a8, w8a8, the asymmetric w4a8i8 / w5a8i8) or f16 activations (w4a16, w5a16, w8a16), each as a skinny form up to 64 rows and a tiled prefill form with double-buffered LDS, and the 4-bit weight unpacked to row-major bf16 for prefill',
    ['halo_gemm_args.h', 'halo_gemm_w2a8_nt.hip', 'halo_gemm_w4a8_nt_m64.hip', 'halo_gemm_w4a8_tiled.hip', 'halo_gemm_w8a8_nt_m64.hip',
     'halo_gemm_w8a8_tiled.hip', 'halo_gemm_wpa8i8_tiled.hip', 'halo_gemm_w4a16_nt_m64.hip', 'halo_gemm_w5a16_nt_m64.hip',
     'halo_gemm_w8a16_nt_m64.hip', 'halo_dequant_w4_bf16.hip'],
    [P('linear.decode/radiance-int/i8+f16-f32', 'halo_gemm_w4a8_nt_m64, halo_gemm_w8a8_nt_m64, halo_gemm_w4a16_nt_m64, halo_gemm_w5a16_nt_m64, halo_gemm_w8a16_nt_m64, halo_gemm_w2a8_nt', 'integer or f16 products, fp32 accumulate, a scale per (row, group of K); int8 activations are written by the producer (kernel.radiance.halo_quant_act)', DEC),
     P('linear.prefill/radiance-int/i8-f32', 'halo_gemm_w4a8_tiled, halo_gemm_w8a8_tiled, halo_gemm_w4a8i8_tiled / w5a8i8_tiled, halo_dequant_w4_bf16', 'as decode', PF)],
    notes='The W8A8 trunk is radiance\'s served form of Qwen3.8-Flash-Next (int8 trunk and lm_head, w4 experts: docs/MOE-W4.md, KL 0.0789 all / 0.0239 text vs bf16, floor 0.0399 / 0.0111). strixite serves Q8 weights with bf16 activations (linear_q8); int8 activations halve the activation bytes a verify GEMM re-reads at M = 2..16 (the PF-13 surface).',
    authors=['Deadcode', 'keptn42', 'radiance contributors'], requires=[COMMON])

mod('kernel.radiance.halo_gemm_bf16',
    'Unquantised GEMMs: bf16 skinny forms (16 and any M up to 64), the tiled prefill form, the biased and f32 variants, the f32 routing and index gate projection with a fixed accumulation order, and runtime LoRA shrink / expand with a slot per row',
    ['halo_gemm_bf16_nt_m16.hip', 'halo_gemm_bf16_nt_m64.hip', 'halo_gemm_bf16_nt_tiled.hip', 'halo_gemm_nt_bias.hip',
     'halo_gemm_f32_nt_bias.hip', 'halo_gemm_nt_f32.hip', 'halo_lora_bf16.hip'],
    [P('linear.dense/bf16/bf16.radiance', 'halo_gemm_bf16_nt_m16, halo_gemm_bf16_nt_m64, halo_gemm_bf16_nt_tiled, halo_gemm_bf16_nt_bias_tiled, halo_gemm_nt_f32_bf16', 'bf16 in and out, fp32 accumulate', DEC + '; tiled at prefill'),
     P('adapter.lora/bf16/bf16', 'halo_lora_shrink, halo_lora_expand (per-row slot)', 'fp32 accumulate', 'any row count; a different adapter per row of one batch')],
    notes='Per-row LoRA slots serve several adapters over one base model in one batch: the shape for lens / expert variants sharing a worker\'s weights.',
    requires=[COMMON])

mod('kernel.radiance.halo_gemm_fp8',
    'Block-scaled FP8 (E4M3) GEMMs on a part with no fp8 matrix instruction: the matvec against the checkpoint\'s own bytes (fp8a16_m1), the fp8a8 skinny, tiled and biased rows, canonical E4M3 with exact f32 block scales, the stored layouts, and the lm_head projection over a head whose rows carry their scales (fp8 and i8)',
    ['halo_fp8_args.h', 'halo_fp8_frag.h', 'halo_fp8_lm.h', 'halo_fp8_wmma.h', 'halo_fp8_layout.cpp', 'halo_gemm_fp8a16_m1.hip',
     'halo_gemm_fp8a8.hip', 'halo_gemm_fp8_f32scale.hip', 'halo_logits_gemm.hip'],
    [P('linear.decode/radiance-fp8/bf16+e4m3-f32', 'halo_gemm_fp8a16_nt_m1, halo_gemm_fp8a8_nt_m16', 'E4M3 widened exactly to f16 (halo_fp8::e4m3_pk), fp32 accumulate', DEC),
     P('linear.prefill/radiance-fp8/e4m3-f32', 'halo_gemm_fp8a8_tiled, halo_gemm_fp8_f32scale', 'as decode', PF),
     P('logits.head/radiance-fp8+i8/f32', 'halo_logits_gemm_fp8, halo_logits_gemm_i8', 'f32 logits', 'M rows over the vocabulary')],
    requires=[COMMON])

mod('kernel.radiance.halo_gemm_mxfp4',
    'OCP-MXFP4 weight x fp8 activation GEMMs: the decode GEMV (M <= 8) and its wide form (9..64) whose output bits do not depend on the height, the K-split decode form up to 128 rows, the skinny m64, the gate_up form with the gated fp8 quantiser in its epilogue, and the tiled prefill GEMM',
    ['halo_mxfp4.h', 'halo_gemm_mxfp4a8_gemv.hip', 'halo_gemm_mxfp4a8_decode.hip', 'halo_gemm_mxfp4a8_nt_m64.hip',
     'halo_gemm_mxfp4a8_gated.hip', 'halo_gemm_mxfp4a8_tiled.hip'],
    [P('linear.decode/radiance-mxfp4/e4m3-f32', 'halo_gemm_mxfp4a8_gemv, _gemv_wide, _decode, _nt_m64, _gated_gemv', 'e2m1 nibbles with an E8M0 scale per 32, fp32 accumulate; row-height invariant', DEC),
     P('linear.prefill/radiance-mxfp4/e4m3-f32', 'halo_gemm_mxfp4a8_tiled', 'as decode', PF)],
    notes='A 4-bit weight stream is half the bytes of Q8 at decode, where time is bytes over 226 GB/s. hipfire\'s mfp4 (linear.decode/hipfire-mfp4/f32) is the catalogue\'s other 4-bit float form.',
    authors=['ggz14', 'slydlake', 'Deadcode', 'radiance contributors'], requires=[COMMON, 'module:kernel.radiance.halo_gemm_fp8'])

mod('kernel.radiance.halo_quant_act',
    'Junction kernels that write the quantised activation the next GEMM reads, so a quantised trunk adds no quantiser launch: per-row int8 and per-(row, 128) E4M3 quantisers alone, and fused behind their producers (residual add + RMS norm, RMS norm + block Hadamard, gated product + Hadamard, gelu(gate) * up, and producers that already hold the row in registers); ParoQuant\'s Givens rotation dense on the matrix cores or to f16; and the GPTQ Gram accumulation H += X^T X',
    ['halo_quant_act_i8.hip', 'halo_quant_act_fp8.hip', 'halo_had_quant_act_i8.hip', 'halo_rmsnorm_quant_i8.hip',
     'halo_rmsnorm_had_quant_i8.hip', 'halo_gated_had_quant_i8.hip', 'halo_gelu_mul_quant_i8.hip', 'halo_fused_quant_fp8.hip',
     'halo_givens_dense.hip', 'halo_givens_rotate.hip', 'halo_gram_accum_fp8.hip'],
    [P('quant.activation/-/i8+e4m3', 'halo_quant_act_i8, halo_quant_act_fp8, halo_rmsnorm_quant_i8, halo_rmsnorm_had_quant_i8, halo_gated_had_quant_i8, halo_gelu_mul_quant_i8', 'per-row symmetric int8, or E4M3 with an f32 scale per (row, 128 of K), written in the consumer GEMM\'s fragment order', 'one launch a junction, decode and prefill'),
     P('quant.calibration-gram/-/f32', 'halo_gram_accum_fp8', 'f32 Gram matrix over the served activations', 'offline calibration on the device')],
    notes='The fusion that paid in radiance is producer + quantiser (it removes a launch AND a pass over the row). SFX-PKG measured strixite\'s plain boundary fusion at 0.9 % of a step; this family is the reason to revisit it only together with int8 activations.',
    authors=['Deadcode', 'keptn42', 'radiance contributors'], requires=[COMMON])

mod('kernel.radiance.halo_attn',
    'Attention on gfx1151 in a transposed core (S^T = K.Q^T, a lane owns a query row): paged decode / MTP-verify at head_dim 256 and 128 over grouped-query ratios 1, 2, 4, 6, 8 and 12 (12 = Qwen3.8-Flash-Next\'s QSA layers, 24 query heads over 2 KV heads; 6 with a 4-token page is its TP4 leg), sliding windows, bf16 / f16 / E4M3 KV, the prefill form, dense and prefix-self forms, and the fused QKV preambles (norm, partial RoPE, gate)',
    ['halo_attn_args.h', 'halo_attn_decode_h256_gqa6.hip', 'halo_attn_paged_gqa8.hip', 'halo_attn_paged_h128_gqa1.hip',
     'halo_attn_paged_h128_gqa2.hip', 'halo_attn_paged_h128_gqa2w.hip', 'halo_attn_paged_h128_gqa4.hip', 'halo_attn_paged_h256_gqa12.hip',
     'halo_attn_paged_h256_gqa2.hip', 'halo_attn_paged_h256_gqa4.hip', 'halo_attn_paged_h256_gqa6.hip', 'halo_attn_paged_h256_gqa6_bs4.hip',
     'halo_attn_paged_window.hip', 'halo_attn_prefill_h256_gqa6.hip', 'halo_attn_dense_bf16.hip', 'halo_attn_gen.hip',
     'halo_qk_norm_rope_gate.hip', 'halo_qkv_norm_rope.hip'],
    [P('attn.softmax-gqa/kv-bf16+f16+e4m3/bf16.radiance', 'halo_attn_paged_* (h128 / h256, gqa 1..12), halo_attn_decode_h256_gqa6', 'f16 operands on WMMA, lazy reference-max online softmax, fp32 statistics', 'decode and MTP verify over the indexer\'s page-id table'),
     P('attn.prefill-causal/kv-bf16+e4m3/bf16.radiance', 'halo_attn_prefill_h256_gqa6, halo_attn_dense_bf16, halo_attn_prefix_self', 'as decode', PF),
     P('attn.qk-norm-rope-store/-/bf16.radiance', 'halo_qk_norm_rope_gate, halo_qkv_norm_rope', 'per-head RMS norm, partial NeoX RoPE, gate; one pass', 'one launch a layer')],
    notes='libhalo/halo_dt16.h is the whole gfx12 -> gfx1151 porting story for attention bodies.',
    requires=[COMMON])

mod('kernel.radiance.halo_allreduce',
    'Tensor-parallel collectives as device kernels: one-shot and two-shot (reduce-scatter + all-gather) all-reduce and all-gather at 2, 3, 4 and 8 ranks over peer-mapped memory, on an exact wire (bf16 / f16 / f32) or a lossy one (Walsh-Hadamard rotated 6-bit, three-tier int8), with the handshake, link-hold and shard rules; and the all-reduce fused with the residual add, RMS norm, block Hadamard and int8 quantiser that always follow it',
    ['halo_ar_entry.hip', 'halo_ar_gather_2rank.hip', 'halo_ar_gather_Nrank.hip', 'halo_ar_oneshot_2rank_exact.hip',
     'halo_ar_oneshot_2rank_wht6.h', 'halo_ar_oneshot_2rank_wht6.hip', 'halo_ar_oneshot_Nrank_exact.hip',
     'halo_ar_twoshot_Nrank_exact.hip', 'halo_ar_twoshot_Nrank_ti8.hip', 'halo_ar_ln_had_quant_i8.hip', 'halo_ar_hc.h',
     'halo_ar_links.h', 'halo_ar_seq.h', 'halo_ar_shards.h', 'halo_ar_ti8.h', 'halo_ar_wide.h', 'halo_p2p_copy2d.hip'],
    [P('collective.allreduce-p2p/-/bf16+f32', 'halo_ar_oneshot_{2,3,4,8}rank_exact, halo_ar_twoshot_{3,4,8}rank_exact, halo_ar_gather_{2,3,4,8}rank', 'exact wire: the sum in rank order', 'peer GPUs on one PCIe fabric; a decode or prefill collective a layer'),
     P('collective.allreduce-p2p/-/wht6+ti8', 'halo_ar_oneshot_2rank_wht6, halo_ar_twoshot_{3,4}rank_ti8, halo_ar_ln_had_quant_i8', 'lossy wire: Hadamard-rotated 6-bit or tiered int8 payload', 'as above, where the link is the bound')],
    notes='Written for cards that map each other\'s memory over PCIe; the pod\'s ranks are separate machines on a 100 GbE RoCE rail (runtime.strixfold.coll_rdma, kernel.strixfold.coll_proxy). What carries over: the compressed wire (the rail carries fp32 partials today) and the fused all-reduce + norm + quantise epilogue. docs/TP4.md in the origin lists every TP4 lever tried and what closed it.',
    authors=['tcclaviger', 'goodman-b', 'keptn42', 'Deadcode', 'radiance contributors'], requires=[COMMON])

mod('kernel.radiance.halo_sample',
    'Sampling on the device: the vLLM chain for one position in one workgroup (temperature, softmax, top-k, top-p, min-p, renormalise, draw) with the second draw and accept probability speculative sampling needs, the decomposed stage ops, the windowed no-repeat-ngram ban, the exact per-row top-R of a wide bf16 matrix for a sharded draft head, and the global argmax over gathered (id, value) pairs',
    ['halo_sample_chain_f32.hip', 'halo_sample_stages_f32.hip', 'halo_sample_ngram_f32.hip', 'halo_rowtopk_bf16.hip', 'halo_pair_argmax_f32.hip'],
    [P('sample.next-token/-/f32.radiance', 'halo_sample_chain_f32, halo_sample_stages_*', 'f32, Philox draws keyed by seed and position (abi/rad_philox.h)', 'one workgroup a position; tokens stay on the device'),
     P('logits.topk/-/bf16.radiance', 'halo_rowtopk_bf16, halo_pair_argmax', 'exact top-R, ties to the lower id', 'draft head shards')],
    notes='Sampled (temperature > 0) speculative decoding with the accept test on the device; strixite samples on the host from the top 20 candidates.',
    requires=[COMMON, 'module:runtime.radiance.plugin_abi'])

mod('kernel.radiance.halo_ple',
    'Qwen4-Exp\'s Per-Layer Embedding: the hashed n-gram row ids on the device, the dot-product gate and the dilated depthwise convolution with its silu and residual add',
    ['halo_ngram_ids.hip', 'halo_ple_gate_bf16.hip', 'halo_ple_conv_bf16.hip'],
    [P('ple.inject/-/bf16.radiance', 'halo_ngram_ids, halo_ple_gate_bf16, halo_ple_conv_bf16', 'int64 wrapping hash; bf16 rows, fp32 accumulate', 'one layer a step; the table itself is held in RAM by the engine')],
    notes='strixite reads n-gram rows through a row cache from NVMe or NFS (PLE wait 333 s of 51,600 engine seconds on worker B, max 4.7 s); radiance keeps the table in RAM (NOTICE: keptn42).',
    requires=[COMMON])

mod('kernel.radiance.halo_dflash',
    'The DFlash2 block-diffusion drafter\'s kernels: the grouped dynamic depthwise convolution (fused) and the candidate selector (edge scores and the path walk, one workgroup a sequence)',
    ['halo_dflash_conv_body.h', 'halo_dflash_conv_t2_g16_bf16.hip', 'halo_dflash_select_bf16.hip'],
    [P('speculation.draft/dflash/bf16.radiance', 'halo_dflash_conv_t2_g16_bf16, halo_dflash_select_bf16', 'bf16, fp32 accumulate', 'a whole draft block in one backbone pass')],
    requires=[COMMON])

mod('kernel.radiance.halo_model_ops',
    'The small ops between the GEMMs: embedding gathers, norms, elementwise ops, RoPE and KV stores (halo_model_bf16), the encoder ops, Gemma 4\'s sandwich-norm joins each as one kernel instead of a chain (~250 norms and ~90 adds a step otherwise), and a prefetch kernel that reads a weight\'s leading bytes into the on-die cache ahead of its GEMM',
    ['halo_model_bf16.hip', 'halo_encoder_bf16.hip', 'halo_norm_join_bf16.hip', 'halo_prefetch.hip'],
    [P('norm.rmsnorm/-/bf16.radiance', 'halo_rmsnorm_*, halo_norm_join_* (halo_model_bf16.hip, halo_norm_join_bf16.hip)', 'fp32 sum of squares, one rounding to bf16', 'one launch a junction'),
     P('runtime.weight-prefetch/-/-', 'halo_rad_prefetch', 'reads only', 'issued on a second stream while a collective or another kernel holds the main one')],
    notes='halo_prefetch is the shape of PF-9 06f047e361 (MALL prefetch of the next weights from a second stream).',
    requires=[COMMON])

mod('kernel.radiance.halo_glm_family',
    'Kernels of model families the pod does not serve on radiance\'s path: GLM-5.3 routing and blocks (halo_glm), KDA exact recurrence with speculative candidate snapshots, absorbed MLA attention over one 512-wide latent a token, and the DSA indexer (pool, query, score)',
    ['halo_glm.hip', 'halo_kda.hip', 'halo_mla.hip', 'halo_dsa.hip'],
    [P('attn.sparse-sink/kv-latent/bf16.radiance-mla', 'halo_mla_*, halo_dsa_* , halo_kda_*, halo_glm_*', 'fp32 softmax statistics merged across waves', 'GLM-5.3 / Ling-3.0 decode and prefill')],
    notes='DSA and MLA are the DeepSeek-V4.1 / GLM surfaces StrixFold\'s sparse_attn and indexer_score cover (STRIX-META widgets A, B, F).',
    authors=['Deadcode', 'keptn42', 'radiance contributors'], requires=[COMMON, 'module:kernel.radiance.halo_moe', 'module:kernel.radiance.halo_gemm_fp8'])

mod('kernel.radiance.halo_media',
    'Image, video, audio and vision-tower kernels: diffusion transformer norms and sampler steps, LTX-2 and video ops, video sparse attention, the upscaler\'s 3-D convolution, vocoder convolutions, NHWC convolution as an implicit GEMM, ViT and SAM attention and the vision tower\'s norm and position embedding',
    ['halo_gen.hip', 'halo_ltx.hip', 'halo_video.hip', 'halo_vsa.hip', 'halo_upscaler.hip', 'halo_audio.hip', 'halo_conv2d_nhwc.hip',
     'halo_conv_t1d.hip', 'halo_attn_vit_h72_bf16.hip', 'halo_sam_attn_bf16.hip', 'halo_vit_norm_embed.hip'],
    [P('media.generation-ops/-/bf16.radiance', 'halo_* rows of docs/OPS.md "Image generation", "Video generation", "Audio generation", "LTX-2"', 'bf16, fp32 accumulate', 'image, video and audio models; vision towers')],
    notes='No strixflerp interface matches: recorded for the vision tower (FLERP-VISION cb7f93a5f5) and for a later media recipe.',
    requires=[COMMON, 'module:runtime.radiance.plugin_abi'])

mod('runtime.radiance.halo_rows',
    'libhalo\'s declaration side on the host: the row table the engine selects kernels from (a row = op, constraint band, priority, tunables, optional chain and fusion claims), the shape and extent functions, the weight layout tags with gfx1151\'s fragment order, the fusion pass over chains of unfused ops, and the generated weak stubs that answer RAD_E_UNSUPPORTED for units not yet ported',
    ['halo_rows.cpp', 'halo_shapes.cpp', 'halo_layout.cpp', 'halo_fuse.cpp', 'halo_stubs.cpp'],
    [P('kernel-select.rows/radiance/-', 'RadKernelInfo rows, rad_plugin_open (halo_rows.cpp); halo_fuse.cpp fusion claims', 'n/a', 'selection by op, shape band and measured priority; declines machines without a gfx1151')],
    notes='A row table with measured constraint bands per shape is what strixflerp\'s champion ledger does per model by hand; the stub count as a porting scoreboard is the same idea as [gate] status.',
    lang='c++', requires=[COMMON, 'module:runtime.radiance.plugin_abi'])

mod('tool.radiance.halo_selftest',
    'libhalo\'s checks and build: halo_selftest (every row against recorded cases shared with libr4d), halo_parity (is libhalo the same plugin interface as libr4d), the stub and unit-list generator, the CMake file and the porting guide',
    ['halo_selftest.cpp', 'halo_parity.cpp', 'gen_stubs.py', 'CMakeLists.txt', 'README.md', 'PORTING.md'],
    [P('verify.kernel-selftest/radiance/-', 'halo_selftest <libhalo.so> --case <row>; halo_parity <libr4d.so> <libhalo.so>; python3 libhalo/gen_stubs.py --check', 'against libref outputs', 'offline, on the card')],
    lang='c++', requires=[COMMON, 'module:runtime.radiance.plugin_abi'])

left = [f for f in O.expand([L]) if f not in claimed]
if left:
    raise SystemExit('libhalo files in no module: ' + ' '.join(left))

ORIGIN_TOML = '''
# radiance: an inference server for AMD GPUs in C++ and HIP (language, image, video, speech and music models);
# architectures, kernel libraries and quantisers are .so plugins. Harvested here: libhalo, its kernel library for
# Strix Halo (gfx1151), and the plugin ABI headers (card ORIGIN-RADIANCE b0c1db5993).
name      = "radiance"
repo      = "https://codeberg.org/StillDeadcode/radiance"
commit    = "38c182a2baa35e92bc1104460d7410b0daf2439a"
branch    = "main"
version   = "head of 2026-11-16 (chat: a tool-call delimiter is not a turn terminator)"
reference_checkout = "laptop: ~/Code/strixflerp-upstream-supplement/radiance (libhalo/, abi/, LICENSE, NOTICE, git ls-files list; holds 38c182a2baa3, every harvested file byte-identical). The control-node ark copy /data/ark-inbox/src/codeberg.org__StillDeadcode__radiance is shallow and now at 689f554 (updated 2026-11-21), so it no longer holds 38c182a2baa3"

[licence]
spdx      = "Apache-2.0"
files     = ["LICENSE", "NOTICE"]                          # copied verbatim next to this file
third_party = "NOTICE names the contributors (tcclaviger: N-rank all-reduce; ggz14: MXFP4 GEMM, GDN overflow guards; slydlake: MXFP4 decode and prefill GEMMs; goodman-b: four-rank tensor parallelism for Qwen3.8-Flash-Next; keptn42: fused router launch, routed GEMM decode loads, n-gram table in RAM, and more). vendor/ in the origin is not harvested. Weights are not part of the tree."

[toolchain]
host      = "C++ and CMake (CMakePresets.json); libhalo also builds standalone against an installed radiance (cmake -S libhalo -DCMAKE_PREFIX_PATH=<radiance prefix>)"
device    = "ROCm HIP for gfx1151; built only when RAD_GPU_TARGETS names gfx1151"
build     = "cmake --preset <preset> -DRAD_GPU_TARGETS=gfx1151 -> radiance_home/kernels/libhalo.so; every unit compiles -ffp-contract=off (rad_hipify)"

[kernels]
loading   = "libhalo.so is a kernel plugin: rad_plugin_open declares rows (op, constraint band, priority) and the engine selects a row per op and shape; libraries are taken in file-name order, so libhalo comes ahead of libr3d on a gfx1151; unported units are weak stubs returning RAD_E_UNSUPPORTED"
models    = "arch/qwen4exp_fp8 is the origin's Qwen3.8-Flash-Next plugin (served container: w4 experts, int8 trunk, MTP, vision; 113 GiB); not harvested"
'''

LEFT_OUT = {
    'core/, plugin/, integrations/, dashboard/, deploy/, docker/, scripts/, tools/, cmake/, build.sh, Dockerfile': 'the engine, its server and fleet code, packaging and tools: a second serving stack, read for shapes, not vendored',
    'arch/': 'model plugins (qwen4exp_fp8 is Qwen3.8-Flash-Next); a later card harvests the one the pod serves if a recipe needs it',
    'libr4d/, libr3d/, libavx/, libref/, libquant/': 'kernel libraries for other targets (gfx1201, generic RDNA3, CPU), the reference library and the quantisers',
    'tests/, data/, vendor/, docs/, spec.md, README.md': 'tests, recorded cases, third-party code and documentation (docs/TP4.md, MOE-W4.md, IQR.md, KV-STREAMING.md, QSA.md were read for the roadmap)',
}

if __name__ == '__main__':
    run(O, mods, ORIGIN_TOML, LEFT_OUT, ['LICENSE', 'NOTICE'])
