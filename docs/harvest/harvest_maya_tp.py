#!/usr/bin/env python3
"""Harvest Maya-TP (uv-genai/maya-tp at 6951bd7, tp-v1.1.45 on Project Maya v1.1.58) into catalog/maya_tp.
Run: python3 docs/harvest/harvest_maya_tp.py
     python3 docs/harvest/harvest_maya_tp.py --base-diff <project-maya HEAD tree> <project-maya v1.1.78 tree>
         # rewrite data/maya_tp/base-diff.tsv and data/maya_tp/project-maya-head-glm.tsv (run on control-node)

Card ORIGIN-MAYA-TP 2982083a71. Maya-TP is a derived project of Project Maya (mw00/project-maya, itself grown out of
Niko1221/Strata, so its sources still say strata::) for two AMD Strix Halo hosts joined by Thunderbolt / USB4 RDMA
(usb4_rdma, RoCE v2): intra-expert tensor parallel for GLM-5.3-Flash (GLM-5 DSA), DeepSeek-V4.1-Flash and MiniMax-M3.1,
a GPU-staged queued all-reduce ported from gufo's Ibrverbs, MTP / DSpark / DFlash speculative decoding under TP2 and
the Maya server driving the pair.

Scope ("the more modules the merrier", the card): the engine is split into many family modules, every kernel family,
transport, scheduler, speculative-decode and serve piece its own module (FAMILIES below), each with a describe saying
its shape and where it would go in strixflerp. The TP2 delta (src/net, src/core/tp_context.cu, include/strata/net,
the tp_* headers, src/ds4/ds4_tp_*, src/mm3/mm3_tp.*, tools/tp2, tools/llama-swap and the TP tests) is in modules of
its own. data/maya_tp/base-diff.tsv lists every file against the newest Project Maya in the ark (53885ce, v1.1.81)
and against v1.1.78 (1016876, the base of catalog/kestrel); data/maya_tp/project-maya-head-glm.tsv lists the
GLM-5.3-Flash engine files Project Maya changed after v1.1.78 (recorded, not harvested: the card asks for maya-tp's
modules). b-iurea/project-maya (a research spec in the ark) is named in ORIGIN.toml as a reference only.

The reference checkout is the ark copy on control-node
(/data/ark-inbox/src/github.com__uv-genai__maya-tp#6951bd7efb3c76b088e663645f274431b22dd913, a shallow clone);
the laptop holds the same files, moved by a tarball through control-node get, in ~/Code/strixflerp-upstream-supplement/
maya_tp. MAYA_TP_CHECKOUT names another copy. The tree digest below pins the bytes (.git left out)."""
import hashlib, os, re, sys, tomllib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvestlib
from harvestlib import Origin, M, P, run

harvestlib.C_FAMILY = tuple(harvestlib.C_FAMILY) + ('.cu', '.cuh')

ARK = '/data/ark-inbox/src/github.com__uv-genai__maya-tp#6951bd7efb3c76b088e663645f274431b22dd913'
CHECKOUT = os.environ.get('MAYA_TP_CHECKOUT', os.path.join(harvestlib.SUPPLEMENT, 'maya_tp'))
COMMIT = '6951bd7efb3c76b088e663645f274431b22dd913'
HEAD_MAYA = '53885cef8faf0d3b3f7e28c0b1537a056c1f3ecf'
V78_MAYA = '1016876a387e332fc5bf8bb82030fca7dc2821b2'
TREE_SHA256 = 'c66ece83c2f572a596331d8a512a294bbee5265ed5e07f32586ae8b9aead170b'
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'maya_tp')
SKIP = ('.git',)


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


def write_base_diff(head, v78):
    """data/maya_tp/base-diff.tsv: every maya-tp file against Project Maya HEAD and v1.1.78 (sha256);
    data/maya_tp/project-maya-head-glm.tsv: the GLM engine files Project Maya changed between v1.1.78 and HEAD."""
    def hashes(root):
        return {f: harvestlib.sha256(os.path.join(root, f)) for f in tree_files(root)}
    tp, hd, old = hashes(CHECKOUT), hashes(head), hashes(v78)
    os.makedirs(DATA, exist_ok=True)
    rows = []
    for f in sorted(set(tp) | set(hd)):
        a, b, c = tp.get(f, '-'), hd.get(f, '-'), old.get(f, '-')
        st = 'A' if f not in hd and f not in old else 'D' if f not in tp else '=' if a == b else 'M78' if a == c else 'M'
        if st == 'D' and f in old and f not in hd:
            continue
        rows.append((st, f, a, b, c))
    with open(os.path.join(DATA, 'base-diff.tsv'), 'w') as o:
        o.write('# uv-genai/maya-tp %s against mw00/project-maya HEAD %s (v1.1.81) and v1.1.78 %s (sha256 of each file).\n' % (COMMIT, HEAD_MAYA, V78_MAYA))
        o.write('# Written by docs/harvest/harvest_maya_tp.py --base-diff. A: only in maya-tp (no file of that path in either Maya);\n')
        o.write('# =: same bytes as Maya HEAD; M78: same bytes as v1.1.78 but not HEAD; M: differs from both; D: in Maya HEAD, not in maya-tp.\n')
        o.write('# Maya-TP integrates upstream v1.1.54-v1.1.58 (README) and keeps the strata:: names Maya later renamed to maya::\n')
        o.write('# (include/strata, src/core/... includes), so most M rows are that rename and upstream drift, not TP work.\n')
        o.write('status\tpath\tmaya_tp\tmaya_head\tmaya_v1.1.78\n')
        for r in rows:
            o.write('\t'.join(r) + '\n')
    glm = [f for f in sorted(set(hd) | set(old)) if re.search(r'(^|/)(glm|glmd)_|glm53|moe_mmq', f)
           and not f.startswith(('docs/', 'bench/results/')) and hd.get(f) != old.get(f)]
    with open(os.path.join(DATA, 'project-maya-head-glm.tsv'), 'w') as o:
        o.write('# The GLM engine files of mw00/project-maya that differ between v1.1.78 (%s) and HEAD %s (v1.1.81).\n' % (V78_MAYA, HEAD_MAYA))
        o.write('# Written by docs/harvest/harvest_maya_tp.py --base-diff. Recorded for the card, not harvested.\n')
        o.write('path\tmaya_v1.1.78\tmaya_head\tin_maya_tp\n')
        for f in glm:
            o.write('%s\t%s\t%s\t%s\n' % (f, old.get(f, '-'), hd.get(f, '-'), 'yes' if f in tp else 'no'))
    counts = {}
    for r in rows:
        counts[r[0]] = counts.get(r[0], 0) + 1
    print('base-diff.tsv: %d rows %s; project-maya-head-glm.tsv: %d files' % (len(rows), counts, len(glm)))


if len(sys.argv) == 4 and sys.argv[1] == '--base-diff':
    write_base_diff(sys.argv[2], sys.argv[3])
    sys.exit(0)


class TreeOrigin(Origin):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.roots = [CHECKOUT]
        head = os.path.join(CHECKOUT, '.git', 'HEAD')
        if os.path.isfile(head) and open(head).read().strip() != COMMIT:
            raise SystemExit(f'{head} names {open(head).read().strip()}, the harvest pins {COMMIT}')
        got = tree_digest(CHECKOUT, self.files())
        if got != TREE_SHA256:
            raise SystemExit(f'{CHECKOUT}: tree digest {got}, the harvest pins {TREE_SHA256} (maya-tp {COMMIT[:12]})')

    def files(self):
        if self._files is None:
            self._files = tree_files(CHECKOUT)
        return self._files


O = TreeOrigin('maya_tp', 'maya_tp', 'https://github.com/uv-genai/maya-tp', COMMIT, 'main (tp-v1.1.45, merge of PR #184)', 'MIT',
               ['Maya-TP contributors (uv-genai)', 'Peasant Smith and the Project Maya contributors', 'Niko1221 and the Strata contributors'],
               'therock-10.2')
LLAMA = ['Maya-TP contributors (uv-genai)', 'Peasant Smith and the Project Maya contributors', 'The ggml authors (llama.cpp 3cf03257, MIT)']
KC, IK, IC, IN, PF = 'src/kernels/cuda/', 'include/strata/kernels/', 'include/strata/core/', 'include/strata/net/', 'src/prefill/'
DS, MM, CPU = 'src/ds4/', 'src/mm3/', 'src/kernels/cpu/'
TP = 'TP2 delta: '


def F(id, summary, why, paths, provides, compile=None, lang='hip-c++', notes='', flags=None, tp=False, nosrc=()):
    return dict(id=id, summary=summary, why=why, paths=paths, provides=provides, compile=compile, lang=lang,
                notes=notes, flags=flags or {}, tp=tp, nosrc=nosrc)


def kc(*names):
    return [KC + n for n in names]


NEEDS_LLAMA = 'needs llama.cpp ggml-cuda sources (MMQ, the pinned 3cf03257 checkout) that the maya-tp tree does not contain'
HOST = 'host engine code; the origin builds it only inside its CMake engine targets (strata_engine, maya-glm, maya-ds4, maya-mm3), so it is harvested, not compiled alone'

FAMILIES = [
 # ---- kernels: the Strata / Maya single-host families -------------------------------------------------------------
 F('kernel.maya_tp.common', 'Maya-TP\'s HIP compatibility layer and shared device headers: include/strata/hip_compat (cuda_runtime.h, force-included into every .cu, mapping CUDA runtime, warp intrinsics and cuBLAS names to HIP, with the wave64 logical-32-lane mapping for gfx908), the kernel headers no other family owns (bf16 / f16 bits, dp4a, gfx-architecture predicates) and llama.cpp\'s block layouts (third_party/ggml/ggml-common.h)',
   'every other kernel.maya_tp module compiles against these headers; it is the CUDA-to-HIP port the whole engine rides on, a newer copy of kernel.strata.common / kernel.kestrel.hip_compat',
   ['include/strata/hip_compat/', 'third_party/ggml/ggml-common.h', 'third_party/ggml/VERSION.txt', 'third_party/ggml/LICENSE'],
   [P('kernel-abi.maya-tp-hip-compat/-/-', 'include/strata/hip_compat/cuda_runtime.h (force-included with -include), intrinsics.hpp', 'CUDA names mapped to HIP; wave32 assumed, wave64 through a logical 32-lane mapping', 'gfx1151 (the fork\'s target), gfx1100 / gfx1201 / gfx908 in cmake/hip_backend.cmake')],
   compile='header-only'),
 F('kernel.maya_tp.s_gemv', 'GEMVs over the S-family weight planes (S2 = Q2_0 codes packed four a byte with one FP32 scale per 64 values; S4 / S8 with a codebook): the S2 decode, the naive S2 GEMV, the measured-neutral fast and quads variants kept as records, the S2 GEMV over Q8_0 activations and the generic S-family GEMV with FP16, Q8_K and Q8_0 activation forms',
   'the Qwen-family expert matvec of the Strata lineage as Maya-TP carries it; a 2-bit Q2_0-plane GEMV shape strixflerp has only from kernel.strata.s_gemv',
   kc('dequant_s2.cu', 's2_gemv.cu', 's2_gemv_fast.cu', 's2_gemv_quads.cu', 's2_gemv_q8.cu', 's_gemv.cu'),
   [P('linear.decode/strata-s/f16+q8_k+q8_0-f32.maya-tp', 'strata::kernels::s2_gemv, s2_gemv_q8, s_gemv(_split, _q8k, _q8_0_split)', 'FP16 / Q8_0 / Q8_K activations, FP32 accumulation', 'decode, one token')]),
 F('kernel.maya_tp.gguf_gemv', 'decode GEMVs on native GGUF blocks adapted from llama.cpp\'s MMVQ at 3cf03257: Q2_0-Q8_0, the K-quants and IQ4 against q8_1 activations (native_mmvq) and the i-quants IQ1_M-IQ3_S with the K-quant helpers (iq_kernels, iq_kquant.cuh)',
   'GLM-5.3-Flash\'s Maya-S / M / L packs are IQ2_XXS / IQ3 / K-quant GGUFs, so this is the dense and shared-weight decode GEMV under the TP2 split as well as one host',
   kc('native_mmvq.cu', 'iq_kernels.cu', 'iq_kquant.cuh'),
   [P('linear.decode/gguf-kquant+iq/q8_1-f32.maya-tp-mmvq', 'strata::kernels::native_<type>_mmvq, iq_mmvq, quantize_q8_1_rows', 'q8_1 activations, llama.cpp\'s integer dots, FP32 out', 'decode, one to a few columns')]),
 F('kernel.maya_tp.quantize_act', 'activation quantization to Q8_0 and Q8_K (and back) the way ggml_mul_mat converts src1 to the weight\'s vec_dot_type, compiled with -ffp-contract=off so Q8_K\'s multiply and magic-bias rounding stay separate FP32 operations',
   'the activation side every GGUF expert GEMV needs; its exact rounding is what keeps the GPU and CPU expert lanes agreeing token for token',
   kc('quantize_act.cu'),
   [P('quant.activation/-/q8_0+q8_k.maya-tp', 'strata::kernels::quantize_q8_0, quantize_q8_K, dequant_q8_0 / _K', 'separately rounded FP32 operations (-ffp-contract=off)', 'any block-multiple row')],
   flags={KC + 'quantize_act.cu': ['-ffp-contract=off']}),
 F('kernel.maya_tp.bf16_gemv', 'BF16 weight GEMVs and conversions: the naive and warp-split BF16 GEMV, llama.cpp\'s FP32-activation MMVF (single, multi-column, multi-matrix) and the BF16 dequantizers',
   'routers, hyper-connection projections and other BF16 tensors run through these every token; a third BF16 GEMV family beside strixite\'s and strata\'s',
   kc('bf16_gemv.cu', 'native_bf16.cu', 'dequant_bf16.cu'),
   [P('linear.dense/bf16/bf16-f32.maya-tp', 'strata::kernels::bf16_gemv(_split), bf16_gemv_fp32_mmvf(_cols, _multi)', 'BF16 weights, FP32 accumulation', 'decode')]),
 F('kernel.maya_tp.router', 'the MoE routers: softmax / stable top-k / renormalise (router_top10), llama.cpp\'s topk-moe for 512 experts top 10 (native_router) and GLM-5\'s sigmoid router with selection bias and group limits (router_sigmoid)',
   'GLM-5.3-Flash routes with a sigmoid router; under TP2 both ranks run the router on the same summed residual so routing stays identical and no expert ids cross the link',
   kc('router_top10.cu', 'native_router.cu', 'router_sigmoid.cu'),
   [P('moe.router/-/f32.maya-tp-sigmoid', 'strata::kernels::router_sigmoid', 'FP32 logits, sigmoid scores with bias, top-k by index ties', 'GLM-5 routing'),
    P('moe.router/-/f32.maya-tp-top10', 'strata::kernels::router_top10, native_router_top10', 'FP32 softmax top-k', 'Qwen-family routing')]),
 F('kernel.maya_tp.moe_experts', 'the MoE block\'s expert kernels: llama.cpp\'s moe-weighted reduction (native_moe), the shared expert with its sigmoid gate and moe_combine (shared_expert), and the grouped GPU expert tier over S2 blobs (s2_expert_grouped: hit selection, grouped GEMV, scatter-add)',
   'the combine step is where a TP2 rank\'s half-width expert partial is formed before the all-reduce; the grouped tier is the VRAM-resident expert path beside the streamed CPU lane',
   kc('native_moe.cu', 'shared_expert.cu', 's2_expert_grouped.cu'),
   [P('moe.combine/-/f32.maya-tp', 'strata::kernels::native_moe_combine(_multi, _hits), moe_combine', 'FP32 parts and weights', 'decode and verify windows'),
    P('moe.experts-decode/strata-s2+bf16/q8_0-f32.maya-tp-grouped', 'strata::kernels::shared_expert, moe_hit_grouped_s2', 'Q8_0 / BF16 forms, FP32 out', 'decode, resident experts')]),
 F('kernel.maya_tp.gdn', 'the gated delta-net layer kernels: the reference step with conv, L2 norm, beta gate and output norm (gdn), the fused conv+L2 and step+norm (fused_gdn), llama.cpp\'s gated_delta_net recurrence (native_gdn) and its front / back ends (native_gdn_preprocess)',
   'the linear-attention recurrence of the Qwen-Next family Maya also serves; the same shapes as kernel.strata.gdn in a later revision',
   kc('gdn.cu', 'fused_gdn.cu', 'native_gdn.cu', 'native_gdn_preprocess.cu'),
   [P('mixer.gdn/-/f32.maya-tp', 'strata::kernels::gdn_step, fused_gdn_step_norm, native_gdn_step', 'FP32 state', 'decode, heads of 128')]),
 F('kernel.maya_tp.hyperconn', 'the gated residual / hyper-connection kernels: gr_read and gr_write (gr), their fused form (fused_gr) and llama.cpp\'s RMS norm and post-ops of the hyper-connection streams (native_gr_norm, native_gr_postops)',
   'the 4-stream residual of the Qwen-Next family; under TP2 the all-reduce lands on this residual, so its layout fixes what crosses the link',
   kc('gr.cu', 'fused_gr.cu', 'native_gr_norm.cu', 'native_gr_postops.cu'),
   [P('hc.mix/bf16/f32.maya-tp', 'strata::kernels::gr_read, gr_write, fused_gr_*, native_gr_norm / _postops', 'BF16 projections, FP32 streams', 'decode')]),
 F('kernel.maya_tp.qsa', 'the QSA sparse-attention family: the cache, indexer and attention (qsa), the decode and prompt attention kernels, the page selector (qsa_select) and llama.cpp\'s QSA attention, indexer and score kernels (native_qsa*)',
   'Maya\'s sparse attention for the Qwen-Next family; a full set of QSA decode / prompt / select kernels in one place, beside kernel.strata.qsa and kernel.kestrel.qsa_score',
   kc('qsa.cu', 'qsa_decode_attn.cu', 'qsa_prompt_attn.cu', 'qsa_select.cu', 'native_qsa.cu', 'native_qsa_indexer.cu', 'native_qsa_score.cu'),
   [P('attn.qsa-indexer/-/f32.maya-tp', 'strata::kernels::qsa_*, native_qsa_indexer, native_qsa_score', 'FP32 scores, FP16 / INT8 / Q4_0 keys', 'decode and prompt'),
    P('attn.softmax-gqa/kv-f16+i8+q4_0/f32.maya-tp-qsa', 'strata::kernels::qsa_decode_attn, qsa_prompt_attn', 'FP32 softmax', 'selected pages')]),
 F('kernel.maya_tp.kvcache', 'KV cache formats and streaming: Q4_0 KV with a Walsh-Hadamard rotation (kv_q4), INT8 KV (kv_q8) and the host-resident KV stream with its page moves (kv_stream)',
   'long contexts on a 128 GB APU live or die by KV bytes; three KV shapes ready to sit behind strixflerp\'s kvcache.store interface',
   kc('kv_q4.cu', 'kv_q8.cu', 'kv_stream.cu'),
   [P('kvcache.store/q4_0-wht+i8/f32.maya-tp', 'strata::kernels::kv_q4_*, kv_q8_*, kv_stream_*', 'Q4_0 with WHT, INT8 per-row scales', 'decode and prompt')]),
 F('kernel.maya_tp.rope', 'rotary embedding: NEOX partial RoPE (rope) and llama.cpp\'s rope_multi / rope_yarn numerical contract (native_rope)',
   'every attention family needs a rope whose bits match the reference; two forms with a stated numerical contract',
   kc('rope.cu', 'native_rope.cu'),
   [P('pos.rope/-/f32.maya-tp', 'strata::kernels::rope_neox, native_rope', 'FP32', 'decode and prompt')]),
 F('kernel.maya_tp.ple', 'the per-layer embedding (PLE) block\'s GPU half (ple) and llama.cpp\'s PLE post-ops (native_ple_postops)',
   'PLE is a Qwen-Next family input path whose table is streamed from disk; the GPU half is what a strixflerp PLE injector would call',
   kc('ple.cu', 'native_ple_postops.cu'),
   [P('ple.inject/-/f32.maya-tp', 'strata::kernels::ple_*, native_ple_postops', 'FP32', 'decode')]),
 F('kernel.maya_tp.elementwise', 'the glue kernels (elementwise: residual adds, gathers, scaling, compiled -ffp-contract=off) and the control-vector row add (cvec)',
   'small kernels every engine needs between the big ones; the contract flag shows which glue operations have rounding that matters',
   kc('elementwise.cu', 'cvec.cu'),
   [P('tensor.ops/-/f32.maya-tp-glue', 'strata::kernels::elementwise ops, cvec_add_rows', 'separately rounded FP32 (-ffp-contract=off)', 'any')],
   flags={KC + 'elementwise.cu': ['-ffp-contract=off']}),
 F('kernel.maya_tp.sampler', 'the sampler chain in llama.cpp\'s order (penalties, top-k, top-p, min-p, temperature, keyed draw) on the GPU',
   'under TP2 rank 0 samples and sends only token decisions over TCP, so the sampler is a rank-0-only piece; a GPU sampler beside sample.next-token providers',
   kc('sampler.cu'),
   [P('sample.next-token/-/f32.maya-tp', 'strata::kernels::sampler_*', 'FP32 logits, keyed RNG', 'one or several rows')]),
 F('kernel.maya_tp.verify', 'the speculative verify window kernels: padded verification outputs, acceptance compare and the window\'s argmax / logit picks (verify_kernels), with Maya-TP\'s HIP optimisations for padded verify outputs',
   'MTP and DSpark under TP2 verify a window on both ranks; these kernels decide acceptance on rank 0 and are the shape a strixflerp window-verify module takes',
   kc('verify_kernels.cu'),
   [P('speculation.window-verify/-/f32.maya-tp', 'strata::kernels::verify_*', 'FP32 logits, greedy or keyed compare', 'verify windows of 1-16 rows')]),
 F('kernel.maya_tp.flash_attn', 'llama.cpp\'s flash-attention kernel specialised for the engine\'s shapes (native_flash_attn, from 3cf03257)',
   'a dense softmax-GQA attention with a pinned numerical reference, for the full-attention layers and prompts',
   kc('native_flash_attn.cu'),
   [P('attn.softmax-gqa/kv-f16/f32.maya-tp-native-fa', 'strata::kernels::native_flash_attn', 'FP16 KV, FP32 softmax', 'decode and prompt')]),
 F('kernel.maya_tp.exl3', 'the EXL3 trellis-quant kernels: the current decode / prompt GEMVs and expert kernels (exl3, compiled -ffp-contract=on so fused multiply-adds sit where the source writes them) and v1.1.8\'s legacy kernels for pre-sm_80 GPUs (exl3_legacy)',
   'EXL3 is Maya\'s second weight family (DeepSeek / MiniMax / GLM EXL3 packs); the same op as kernel.kestrel.exl3 at Maya-TP\'s revision',
   kc('exl3.cu', 'exl3_legacy.cu'),
   [P('linear.decode/exl3/f32-f32.maya-tp', 'strata::kernels::exl3_*', 'EXL3 codebooks and trellis, FP32 out; fused where written (-ffp-contract=on)', 'decode and prompt')],
   flags={KC + 'exl3.cu': ['-ffp-contract=on']}),
 # ---- kernels: GLM-5 / glm5-next / glm-dsa ---------------------------------------------------------------------------
 F('kernel.maya_tp.glm_fast', 'the GLM fast-path kernels (glm_fast.cu, 7,453 lines): AMD IQ2_XXS / IQ2_S / IQ3_XXS decode experts with packed sign expansion and LDS codebooks, the dense GEMV gf::mv with its block-type extensions (glmd_mv_ext, glmd_mv_dots) and the DSA top-pool chooser (dsa_topk)',
   'the hottest GLM-5.3-Flash decode kernels; under TP2 each rank runs every selected expert on its half of n_ff with these, so they are the per-rank expert body of the intra-expert split',
   kc('glm_fast.cu', 'dsa_topk.cuh', 'glmd_mv_ext.cuh', 'glmd_mv_dots.cuh'),
   [P('moe.experts-decode/gguf-iq2+iq3/f32.maya-tp', 'strata::kernels glm fast-path expert launches (include/strata/kernels/glm_fast.hpp)', 'q8_1-style integer dots, FP32 out', 'decode; half n_ff per rank under TP2')]),
 F('kernel.maya_tp.glm_batch', 'GLM\'s batched (several rows / conversations) expert and dense kernels (glm_batch.cu): the multi-row forms of the fast-path GEMVs used by multi-sequence decode and verify windows',
   'multi-slot serving and MTP windows run several rows at once; this is the batched expert body that the TP2 batch frames feed',
   kc('glm_batch.cu'),
   [P('moe.experts-decode/gguf-iq2+iq3/f32.maya-tp-rows', 'strata::kernels glm_batch launches (glm_batch.hpp)', 'FP32 out', 'decode, up to 64 rows')]),
 F('kernel.maya_tp.glm_dsa', 'the glm5-next DSA lightning indexer and absorbed-MLA attention (glm_dsa.cu)',
   'GLM-5\'s DeepSeek-style sparse attention; under TP2 attention heads are split across the ranks and this is the per-rank head-group kernel',
   kc('glm_dsa.cu'),
   [P('attn.sparse-dsa-mla/-/f32.maya-tp', 'strata::kernels glm_dsa launches (glm_dsa.hpp)', 'FP32 scores and softmax over the selected tokens', 'decode')]),
 F('kernel.maya_tp.glm_kda', 'the glm5-next KDA (kimi-delta-attention style) pieces: gates, recurrence and readout over the KDA state layouts (glm_kda.cu)',
   'the linear-attention half of glm5-next\'s hybrid; a recurrence shape strixflerp only has as GDN',
   kc('glm_kda.cu'),
   [P('mixer.kda/-/f32.maya-tp', 'strata::kernels glm_kda launches (glm_kda.hpp)', 'FP32 state', 'decode')]),
 F('kernel.maya_tp.glm_hc', 'GLM-5.3-Flash\'s manifold hyper-connection (mHC) read and write halves (glm_hc.cu)',
   'the mHC residual is what the two TP2 all-reduce points complete; its read / write kernels fix the residual layout on both ranks',
   kc('glm_hc.cu'),
   [P('hc.mix/bf16/f32.maya-tp-mhc', 'strata::kernels glm_hc launches (glm_hc.hpp)', 'FP32 streams', 'decode')]),
 F('kernel.maya_tp.glm_kv_stream', 'the GLM latent KV stream (glm_kv_stream.cu) and the INT8 latent records (glm_lat8.cuh, MAYA_GLM_KV_INT8)',
   'MLA latent KV in INT8 halves the per-token bytes of long GLM contexts; a kvcache shape for latent attention',
   kc('glm_kv_stream.cu', 'glm_lat8.cuh'),
   [P('kvcache.store/mla-latent-i8/f32.maya-tp', 'strata::kernels glm_kv_stream launches', 'INT8 latent with scales', 'decode')]),
 F('kernel.maya_tp.glm_ffn', 'the glm5-next clamped SwiGLU (glm_ffn.cu)',
   'the activation of every GLM FFN and expert; under TP2 it runs on each rank\'s half of n_ff before the down share',
   kc('glm_ffn.cu'),
   [P('act.swiglu-clamped/-/f32.maya-tp', 'strata::kernels glm_ffn launches (glm_ffn.hpp)', 'FP32 with a clamp', 'any')]),
 F('kernel.maya_tp.glmd_decode', 'the glm-dsa decode attention kernels: mla2\'s decode attention (glmd_dec), the B4.3-B4.5 fast kernels (glmd_fast), the token top-k selector (glmd_topk), the NORM RoPE (glmd_rope), the indexer\'s 128-point Walsh-Hadamard rotation (glmd_had), the arch spellings (glmd_arch), the captured-part position block (glmd_graph) and the library marker (glmd_stub)',
   'the GLM-5.3 (DSA) decode attention as Maya\'s glm-dsa port built it; the per-rank head-group attention under TP2',
   kc('glmd_dec.cu', 'glmd_fast.cu', 'glmd_topk.cuh', 'glmd_rope.cuh', 'glmd_had.cuh', 'glmd_arch.cuh', 'glmd_graph.cu', 'glmd_stub.cu'),
   [P('attn.sparse-dsa-mla/-/f32.maya-tp-glmd', 'strata::kernels glmd_dec / glmd_fast launches', 'FP32 softmax over the top-k tokens', 'decode')]),
 F('kernel.maya_tp.glmd_mixer', 'the glm-dsa decode mixer\'s own kernels (glmd_mixer.cu, B4.6 / B14.3 / B14.4)',
   'the per-layer attention mixer glue of the DSA decode, the step whose output is the first of the two TP2 all-reduce points',
   kc('glmd_mixer.cu'),
   [P('attn.dsa-mixer/-/f32.maya-tp', 'strata::kernels glmd_mixer launches (glmd_mixer.hpp)', 'FP32', 'decode')]),
 F('kernel.maya_tp.glmd_batch', 'the glm-dsa batched prompt kernels (glmd_batch.cu, B9.2-B9.5)',
   'GLM prompt attention over many rows; TP2 prefill chunks are bounded by the transport window and land here',
   kc('glmd_batch.cu'),
   [P('attn.prefill-causal/-/f32.maya-tp-glmd', 'strata::kernels glmd_batch launches (glmd_batch.hpp)', 'FP16 / FP32', 'prompt')]),
 F('kernel.maya_tp.glmd_volta', 'the glm-dsa Volta / native kernels: lightning-indexer scorers with WMMA or CUDA cores and F32 / FP16 keys (glmd_volta_idx) and the fused sparse-gather MLA decode (glmd_volta_mla)',
   'a second, matrix-core indexer and MLA decode for the same op; the WMMA scorer is the shape gfx1151\'s WMMA would take',
   kc('glmd_volta_idx.cu', 'glmd_volta_mla.cu'),
   [P('attn.indexer-score/-/f32.maya-tp-volta', 'strata::kernels glmd_volta launches (glmd_volta.hpp)', 'F32 / FP16 keys, WMMA or scalar', 'decode')]),
 F('kernel.maya_tp.glmd_ref', 'the glm-dsa F32 reference kernels (glmd_ref.cu, B3.4) and the device halves of the wave-9 guards: the FP16 activation range guard and friends (glmd_guards.cu)',
   'a reference forward on the GPU is what a parity gate compares the fast path against; the guards are the runtime numerics checks',
   kc('glmd_ref.cu', 'glmd_guards.cu'),
   [P('verify.kernel-reference-maya-tp/-/-', 'strata::kernels glmd_ref, glmd guards', 'F32 reference', 'parity runs')]),
 # ---- kernels: TP2 ------------------------------------------------------------------------------------------------------
 F('kernel.maya_tp.tp_kernels', TP + 'the GPU half of a queued TP2 exchange (src/net/tp_kernels.cu, include/strata/net/tp_kernels.hpp): StagePartial copies the partial into the registered send window and publishes a ready value at system scope from the last block, WaitValue spins one wave on the peer\'s arrival flag with a wall-clock timeout, WaitAdd fuses the wait with the broadcast add of the peer partial into the residual streams',
   'this is the GPU-staged queued transport the card names: the CPU never round-trips inside a decode exchange, so it is the piece strixflerp\'s collective.device-stage interface wants for a two-node rail',
   ['src/net/tp_kernels.cu'],
   [P('collective.device-stage/-/f32.maya-tp', 'strata::net::StagePartial, WaitValue, WaitAdd, AddRowsBroadcast', 'FP32 partials; system-scope release / acquire on 64-bit flags', 'two ranks, decode and prefill windows')], tp=True),
 F('kernel.maya_tp.peer_probe', 'a peer-access probe program (src/kernels/peer_probe.cu): checks hipDeviceCanAccessPeer and times peer copies between local GPUs',
   'the single-host multi-GPU check Maya runs before choosing a split; the counterpart of the two-host link checks in tools/tp2',
   ['src/kernels/peer_probe.cu'],
   [P('verify.peer-access/-/-', 'main() (peer_probe.cu)', 'copy timing', 'several GPUs on one host')]),
 # ---- kernels: prefill ----------------------------------------------------------------------------------------------------
 F('kernel.maya_tp.prefill', 'the prompt path\'s kernels: the batched prompt kernels (src/prefill/kernels.cu), the dense GEMM dispatch through hipBLAS / hipBLASLt (gemm.cu) and the hipBLASLt solution-table parser (hipblaslt_tuning.hpp)',
   'prefill throughput is GEMM-bound; a tuned hipBLASLt dispatch with tables per gfx target is a shape strixflerp\'s linear.prefill providers can take directly',
   [PF + 'kernels.cu', PF + 'gemm.cu', PF + 'hipblaslt_tuning.hpp'],
   [P('linear.prefill/bf16+f16/f32.maya-tp-blaslt', 'strata::prefill::gemm_*, prefill kernels (include/strata/prefill/)', 'BF16 / FP16 in, FP32 accumulate', 'prompt chunks')],
   flags={PF + 'gemm.cu': ['-DSTRATA_HIPBLASLT_AVAILABLE=1']}),
 F('kernel.maya_tp.prefill_moe_rdna4', 'HIP-only GLM prompt experts adapted from Niko1221/Strata fb58e0d (src/prefill/moe_fused_rdna4.cu): IQ prompt experts with int8 activation codes and FP32 scales per 32 values, fused gate / up / down',
   'GLM prompt experts without llama.cpp\'s MMQ; the RDNA fused expert shape beside kernel.kestrel.prefill_moe',
   [PF + 'moe_fused_rdna4.cu'],
   [P('moe.experts-prefill/gguf-iq+kquant/i8-f32.maya-tp-fused', 'strata::prefill moe_fused_rdna4 launches', 'int8 activation codes, FP32 scales', 'prompt')],
   compile='n/a', notes='Not compiled: it includes ggml-cuda\'s common.cuh, so it ' + NEEDS_LLAMA + ' (the compile gate of 2026-11-25 stopped at that include)'),
 F('kernel.maya_tp.prefill_moe_mmq', 'the llama.cpp MMQ view of the prompt experts: moe_mmq.cu and glm-dsa\'s moe_mmq_strata.cu compile ggml-cuda\'s MMQ from the pinned llama.cpp checkout, ggml_cuda_host.cu supplies the host symbols, mmq-fused-swiglu.patch is the fused-SwiGLU patch applied to it',
   'the HIP build requires this path (CMakeLists refuses HIP without STRATA_PREFILL_MMQ); it records exactly which llama.cpp MMQ pieces and patch a GLM prompt path needs',
   [PF + 'moe_mmq.cu', PF + 'moe_mmq_strata.cu', PF + 'ggml_cuda_host.cu', 'third_party/ggml/mmq-fused-swiglu.patch'],
   [P('moe.experts-prefill/gguf-kquant/q8_1-f32.maya-tp-mmq', 'strata::prefill::moe_mmq_*', 'llama.cpp MMQ arithmetic', 'prompt')],
   compile='n/a', notes='Not compiled: ' + NEEDS_LLAMA + '.'),
 # ---- kernels: DeepSeek-V4.1 (maya-ds4) and MiniMax-M3.1 (maya-mm3) ----------------------------------------------------
 F('kernel.maya_tp.ds4_kernels', 'DeepSeek-V4.1\'s own device kernels for a window of T tokens (ds4_kernels.cu, 5,111 lines): attention with its compressed KV, hyper-connections, router and the window forms, with the BF16 / FP8 E4M3 conversions (ds4_numerics), programmatic dependent launch (ds4_pdl) and the Q6_K value reader (ds4_q6k)',
   'DeepSeek-V4.1-Flash is the other TP2 model; these are the per-rank window kernels the TP2 DSpark path verifies with',
   [DS + 'ds4_kernels.cu', DS + 'ds4_kernels.cuh', DS + 'ds4_numerics.cuh', DS + 'ds4_pdl.cuh', DS + 'ds4_q6k.cuh'],
   [P('attn.sparse-sink/-/f32.maya-tp-ds4', 'ds4 kernel launches (ds4_kernels.cuh)', 'BF16 / FP8 E4M3 conversions, no fast math', 'windows of 1-16 tokens')]),
 F('kernel.maya_tp.ds4_iq', 'DeepSeek-V4.1\'s low-bit routed experts in ggml block types on the device (ds4_iq.cu): the i-quants with IQ1 codes (ds4_iq1_codes), the 512-wide i-quant and MXFP4 kernels (ds4_iq512_kern.inc, ds4_mx_kern.inc)',
   'the routed-expert body of maya-ds4; under TP2 its block-aligned slices are what each rank uploads',
   [DS + 'ds4_iq.cu', DS + 'ds4_iq.cuh', DS + 'ds4_iq1_codes.hpp', DS + 'ds4_iq512_kern.inc', DS + 'ds4_mx_kern.inc'],
   [P('moe.experts-decode/gguf-iq+mxfp4/f32.maya-tp-ds4', 'ds4 IQ launches (ds4_iq.cuh)', 'ggml block types, FP32 out', 'decode and windows')]),
 F('kernel.maya_tp.ds4_gguf', 'a DeepSeek-V4.1 GGUF\'s BF16 matrices and MXFP4 experts on the device (ds4_gguf.cu / .cuh, unsloth\'s DeepSeek-V4.1-Flash-1017 GGUFs)',
   'the dense side of maya-ds4\'s GGUF path; the TP2 dense loader splits these matrices by output or input',
   [DS + 'ds4_gguf.cu', DS + 'ds4_gguf.cuh'],
   [P('linear.decode/bf16+mxfp4/f32.maya-tp-ds4', 'ds4 GGUF launches (ds4_gguf.cuh)', 'BF16 / MXFP4', 'decode')]),
 F('kernel.maya_tp.ds4_lt', 'DeepSeek-V4.1\'s prompt-path dense GEMMs through hipBLASLt (ds4_lt.cu / ds4_lt.hpp)',
   'measured prompt GEMMs for DeepSeek shapes on AMD; a hipBLASLt provider for linear.prefill',
   [DS + 'ds4_lt.cu', DS + 'ds4_lt.hpp'],
   [P('linear.prefill/bf16/f32.maya-tp-ds4-lt', 'ds4 lt GEMM (ds4_lt.hpp)', 'BF16, FP32 accumulate', 'prompt')],
   flags={DS + 'ds4_lt.cu': ['-DSTRATA_HIPBLASLT_AVAILABLE=1']}),
 F('kernel.maya_tp.ds4_vision', 'DeepSeek-V4.1\'s vision tower on one GPU (ds4_vision.cu / .cuh, DeepSeek\'s inference/vision.py)',
   'image input for the DeepSeek engine; a vision encoder module strixflerp has none of',
   [DS + 'ds4_vision.cu', DS + 'ds4_vision.cuh'],
   [P('vision.encoder/-/f32.maya-tp-ds4', 'ds4 vision tower (ds4_vision.cuh)', 'BF16 / FP32', 'one image at a time')]),
 F('kernel.maya_tp.ds4_pcie', 'the decode path\'s PCIe lane on the device (ds4_pcie.cuh / ds4_pcie.hpp): the GPU pulls the experts the CPU lane does not hold over the bus',
   'an expert-streaming lane for dGPU hosts; on Strix Halo\'s unified memory it is the shape to avoid, recorded for completeness',
   [DS + 'ds4_pcie.cuh', DS + 'ds4_pcie.hpp'],
   [P('moe.expert-stream-pcie/-/-', 'ds4 pcie lane (ds4_pcie.hpp)', 'bytes moved, no arithmetic', 'decode on a dGPU')], compile='header-only'),
 F('kernel.maya_tp.mm3_kernels', 'maya-mm3\'s kernels around the EXL3 matrices for MiniMax-M3.1-Flash (mm3_kernels.cu / .cuh): attention, norms, router and glue',
   'the MiniMax engine\'s per-layer kernels; its TP path cuts these layers over the group members',
   [MM + 'mm3_kernels.cu', MM + 'mm3_kernels.cuh'],
   [P('attn.softmax-gqa/kv-f16/f32.maya-tp-mm3', 'mm3 kernel launches (mm3_kernels.cuh)', 'FP32', 'decode and prompt')]),
 F('kernel.maya_tp.mm3_exl3', 'maya-mm3\'s EXL3 device kernels (mm3_exl3.cu / .hpp), a copy of the shared EXL3 kernels shaped for MiniMax-M3.1',
   'EXL3 GEMVs specialised for one model family; compare with kernel.maya_tp.exl3 for what the specialisation buys',
   [MM + 'mm3_exl3.cu', MM + 'mm3_exl3.hpp'],
   [P('linear.decode/exl3/f32-f32.maya-tp-mm3', 'mm3 EXL3 launches (mm3_exl3.hpp)', 'EXL3 trellis, FP32 out', 'decode')]),
 F('kernel.maya_tp.mm3_gguf', 'maya-mm3\'s GGUF weights on the device (mm3_gguf.cu / .cuh / mm3_gguf_host.hpp): activation rounding, the dense and expert paths over MiniMax-M3.1\'s Maya quants',
   'the GGUF counterpart of mm3_exl3 for the same model: the same MiniMax-M3.1 layers over ggml block types, so the two show what each weight family costs on one engine',
   [MM + 'mm3_gguf.cu', MM + 'mm3_gguf.cuh', MM + 'mm3_gguf_host.hpp'],
   [P('moe.experts-decode/gguf-kquant/f32.maya-tp-mm3', 'mm3 GGUF launches (mm3_gguf.cuh)', 'ggml block types', 'decode')]),
 F('kernel.maya_tp.mm3_gv', 'maya-mm3\'s dense GEMVs over a GGUF\'s K-quant and Q8_0 matrices (mm3_gv.cu / .cuh): attention, shared and dense FFN weights',
   'a compact K-quant / Q8_0 GEMV family with ggml block layouts, independent of the MMVQ adaptation',
   [MM + 'mm3_gv.cu', MM + 'mm3_gv.cuh'],
   [P('linear.decode/gguf-kquant+q8_0/f32.maya-tp-mm3-gv', 'mm3 GEMV launches (mm3_gv.cuh)', 'FP32 out', 'decode')]),
 F('kernel.maya_tp.mm3_vision', 'MiniMax-M3.1-Flash\'s vision tower (mm3_vision.cu / .hpp)',
   'the second vision encoder in the tree, for the MiniMax engine',
   [MM + 'mm3_vision.cu', MM + 'mm3_vision.hpp'],
   [P('vision.encoder/-/f32.maya-tp-mm3', 'mm3 vision tower (mm3_vision.hpp)', 'BF16 / FP32', 'one image at a time')]),
 # ---- kernels: CPU expert lane -------------------------------------------------------------------------------------------
 F('kernel.maya_tp.cpu_iq', 'the CPU expert rows in SIMD: i-quant rows in 256-bit and 512-bit lanes for several tokens (iq_avx2, iq_avx512), the one-token IQ2_XXS row (iq_avx512_one), Q3_K / Q2_K against Q8_K (kq_avx512), Q2_0 rows with the activation quantizer (q2_avx2) and the Q2_0 expert kernel (expert)',
   'the CPU lane that runs experts not in VRAM; Strix Halo\'s Zen 5 cores with AVX-512 are a real expert tier and these are its kernels',
   [CPU + 'iq_avx2.cpp', CPU + 'iq_avx512.cpp', CPU + 'iq_avx512_one.cpp', CPU + 'kq_avx512.cpp', CPU + 'q2_avx2.cpp', CPU + 'expert.cpp'],
   [P('moe.experts-decode-cpu/gguf-iq+kquant/q8_k-f32.maya-tp', 'strata::kernels::cpu rows (include/strata/kernels/cpu/)', 'integer dots, FP32 out, -ffp-contract=off on iq_avx2', 'decode and windows on the CPU')],
   compile='n/a', lang='c++', notes='Not compiled by the gate: per-source x86 options (-mavx2 / -mavx512*) come from CMakeLists.txt and the gate compiles hip-c++ only.'),
 F('kernel.maya_tp.cpu_exl3', 'EXL3 routed experts on the CPU (exl3_cpu), the scalar EXL3 reference (exl3_ref) and the CPU EXL3 bench (exl3_cpu_bench)',
   'an EXL3 CPU lane so trellis-quant experts can spill to the cores; the reference doubles as a parity oracle',
   [CPU + 'exl3_cpu.cpp', CPU + 'exl3_ref.cpp', CPU + 'exl3_cpu_bench.cpp', CPU + 'exl3_cpu_avx512.inc'],
   [P('moe.experts-decode-cpu/exl3/f32.maya-tp', 'strata::kernels::cpu exl3 (exl3_cpu.hpp, exl3_ref.hpp)', 'FP32', 'decode on the CPU')],
   compile='n/a', lang='c++', notes='Not compiled by the gate (host C++, x86 options).'),
 F('kernel.maya_tp.cpu_pool', 'the CPU expert pool and its tables: the worker pool protocol (pool), the per-layer expert table (expert_layout), native GGUF experts through ggml-cpu (native_expert) and the portable fallback (portable)',
   'the scheduler side of the CPU lane: how experts are handed to cores while the GPU works; the overlap that makes a CPU tier pay',
   [CPU + 'pool.cpp', CPU + 'expert_layout.cpp', CPU + 'native_expert.cpp', CPU + 'portable.cpp'],
   [P('sched.cpu-expert-pool/-/-', 'strata::kernels::cpu::Pool (pool.hpp)', 'no arithmetic of its own', 'decode with a CPU lane')],
   compile='n/a', lang='c++', notes='Not compiled by the gate (host C++).'),
 F('kernel.maya_tp.cpu_glm_lane', 'the glm-dsa CPU expert lane\'s lockstep 4-row decode GEMV (glm_lane_row4, glm_lane_row4_avx2, B14.9)',
   'GLM experts on the CPU for several rows at once, the CPU half of multi-slot decode',
   [CPU + 'glm_lane_row4.cpp', CPU + 'glm_lane_row4_avx2.cpp', CPU + 'glm_lane_row4_x86.hpp'],
   [P('moe.experts-decode-cpu/gguf-iq/f32.maya-tp-row4', 'glm_lane_row4 (glm_lane_row4.hpp)', 'FP32', 'up to 4 rows')],
   compile='n/a', lang='c++', notes='Not compiled by the gate (host C++, AVX2 options).'),
 # ---- runtime: TP2 transport, collectives and partition --------------------------------------------------------------------
 F('runtime.maya_tp.tp_rdma', TP + 'the two-host RDMA transport over RoCE v2 RC queue pairs on Thunderbolt usb4_rdma (src/net/verbs.cpp, 1,632 lines, gufo\'s Ibrverbs ported to strata::net): one RDMA write with immediate data per exchange into the peer\'s rotating fixed-address receive window (hipHostRegister-mapped so the GPU reads it), queued exchanges completed by a flag the NIC raises, a TCP bootstrap for QP metadata and peer-loss detection, RoCE v2 GID choice by the netdev\'s IPv4 address (rdma_gid) and control-socket state classification',
   'the transport the card is about: ~37 Gb/s and ~6 us tiny-write round trips between two Strix Halo hosts with no switch; it is the shape for a strixflerp two-node rail transport beside gufo\'s tp_verbs, and the window / immediate-data design carries to the pod\'s 100 GbE RoCE rail',
   ['src/net/verbs.cpp', 'src/net/rdma_gid.cpp', 'src/net/control_socket_state.hpp'],
   [P('transport.verbs-maya-tp/-/-', 'strata::net::IbrverbsCommunicator (include/strata/net/verbs.hpp), SelectRoceV2Gid', 'byte moves only; the peer sums', 'two ranks, RoCE v2 RC'),
    P('collective.exchange-partial/-/f32.maya-tp', 'Communicator::ExchangePartial / QueuePartial over RDMA windows', 'FP32 partials, commutative two-operand sum on both ranks', 'two ranks')],
   tp=True, compile='standalone', notes='Kin: runtime.gufo.tp_verbs (the source of the port; wire structs GUFT v7 / GUCH v5 / GURD byte-compatible with gufo).'),
 F('runtime.maya_tp.tp_comm_abi', TP + 'the ordered communication boundary and its wire identity (include/strata/net/communicator.hpp, tp_wire.hpp): ExchangePartial / QueuePartial / SendControl / RecvControl with an exclusive operation scope per control request and an operation ordinal on every exchange; the GUFT handshake, GUCH collective header and GURD readiness word, byte-compatible with gufo',
   'the interface any two-rank transport implements so engines sum against it without knowing the wire; the operation ordinal is the guard that stops a collective binding to the wrong request, a rule strixflerp\'s collective.interface can adopt',
   [],
   [P('collective.interface-maya-tp/-/-', 'strata::net::Communicator, Wire / CollectiveHeader structs', 'no arithmetic; ordering and identity', 'world 1 or 2')], compile='header-only', tp=True),
 F('runtime.maya_tp.tp_context', TP + 'the GLM engine\'s two-rank tensor-parallel state (src/core/tp_context.cu, tp_context.hpp, tp_partition.hpp, tp_collective.hpp): the intra-expert split (each rank holds [ff_begin, ff_begin + ff_count) of every expert\'s and dense FFN\'s intermediate dimension as byte spans of the existing quant blob, down shares on quant-block boundaries), the two all-reduce points per block (after the mixer out, after the MoE + shared sum), agree_min votes for exact integer decisions and the prefill batch bound under the transport window',
   'the heart of maya-tp\'s GLM-5.3 intra-expert TP: routing stays identical on both ranks and no rank waits on the other\'s experts; the partition rules transfer directly to a strixflerp TP2 executor for any MoE',
   ['src/core/tp_context.cu'],
   [P('collective.allreduce-tp2/-/f32.maya-tp', 'strata::core::TpContext::all_reduce, agree_min, drain; TpPartition, TpShare, tp_prefill_limit', 'FP32 partials summed local + peer on both ranks (same bits)', 'two ranks; world 1 is a no-op'),
    P('partition.intra-expert-tp2/-/-', 'strata::TpPartition (tp_partition.hpp)', 'byte spans of the quant blob, block-aligned', 'GLM-5 expert and dense FFN widths')], tp=True),
 F('runtime.maya_tp.glm_tp_batch', TP + 'GLM\'s TP2 application frames (include/strata/core/glm_tp_batch.hpp): versioned multi-slot batch plans and commits (up to 4 slots, 64 rows) that rank 0 sends rank 1 over the control channel, with limits, members, generations and positions',
   'multi-conversation serving over the pair needs both ranks to step the same slots in the same order; this is the scheduler frame that keeps them aligned',
   [],
   [P('sched.tp2-batch-frames-glm/-/-', 'strata::core::glm_tp encode / decode of Plan and Commit', 'integer frames', 'up to 4 slots, 64 rows')], compile='header-only', tp=True),
 F('runtime.maya_tp.ds4_tp_control', TP + 'DeepSeek TP2\'s control plane (src/ds4/ds4_tp_control, ds4_tp_channel, ds4_tp_ack, ds4_tp_chunk, ds4_tp_identity, ds4_tp_preflight, ds4_tp_controller): a bounded, versioned, endian-independent command codec (Feed / Step / Verify / Verdict / Reset / Shutdown / Bind / Rows / BatchVerify / BatchCommit), identity exchange and preflight checks, accepted / completed acknowledgements, the shared row bound under the collective window and rank 0\'s controller that poisons the pair on a failure after a message may have reached the worker',
   'a two-rank control protocol whose failure rule (never retry a command or restart one rank) is the hard-won part; it is the shape of a strixflerp TP2 rank-0 controller',
   [DS + 'ds4_tp_control.hpp', DS + 'ds4_tp_channel.hpp', DS + 'ds4_tp_ack.hpp', DS + 'ds4_tp_chunk.hpp', DS + 'ds4_tp_identity.hpp', DS + 'ds4_tp_preflight.hpp', DS + 'ds4_tp_controller.hpp'],
   [P('sched.tp2-control-ds4/-/-', 'strata::ds4::tp2::Controller, encode / decode of Control, exchange_identity, chunk_rows', 'integer frames', 'two ranks')], compile='header-only', tp=True),
 F('runtime.maya_tp.ds4_tp_batch', TP + 'DeepSeek TP2\'s opt-in multi-sequence batch metadata (ds4_tp_batch.hpp, ds4_tp_batch_state.hpp): batch plans (up to 4 slots, 16 rows) and commits with emitted / pending tokens, and the per-slot logical state that refuses stale, skipped or out-of-order operations',
   'the batched-serving scheduler state of the DeepSeek pair; with glm_tp_batch it shows one protocol shape serving two engines',
   [DS + 'ds4_tp_batch.hpp', DS + 'ds4_tp_batch_state.hpp'],
   [P('sched.tp2-batch-frames-ds4/-/-', 'strata::ds4::tp2::BatchPlan, BatchCommit, BatchState', 'integer frames', 'up to 4 slots')], compile='header-only', tp=True),
 F('runtime.maya_tp.ds4_tp_worker', TP + 'DeepSeek TP2\'s rank-1 side (ds4_tp_worker.hpp, ds4_tp_worker_state.hpp, ds4_tp_model_worker.hpp): the worker loop that receives control frames, its state machine and the model worker that runs the matching forward on its slices',
   'the follower half of a two-rank engine; the state machine mirrors the controller\'s so either side can detect a broken pair',
   [DS + 'ds4_tp_worker.hpp', DS + 'ds4_tp_worker_state.hpp', DS + 'ds4_tp_model_worker.hpp'],
   [P('sched.tp2-worker-ds4/-/-', 'strata::ds4::tp2::Worker, WorkerState, ModelWorker', 'no arithmetic of its own', 'rank 1 of 2')], compile='header-only', tp=True),
 F('runtime.maya_tp.ds4_tp_loader', TP + 'DeepSeek TP2\'s rank-local weight loading (ds4_tp_partition, ds4_tp_pack, ds4_tp_dense_loader, ds4_tp_rank_loader, ds4_tp_upload.cuh, ds4_tp_expert_pool.cuh): the partition of every tensor by output rows or input blocks, the pack description, the dense loader that owns one split matrix, the rank loader and the HIP upload of block-aligned GGUF expert slices into a bounded expert pool',
   'each rank loads only its share from the GGUF, which is what lets two 128 GB hosts hold a model neither holds alone; the loader rules carry to any strixflerp TP2 loader',
   [DS + 'ds4_tp_partition.hpp', DS + 'ds4_tp_pack.hpp', DS + 'ds4_tp_dense_loader.hpp', DS + 'ds4_tp_rank_loader.hpp', DS + 'ds4_tp_upload.cuh', DS + 'ds4_tp_expert_pool.cuh'],
   [P('format.loader-tp2-slices/-/-', 'strata::ds4::tp2::RankDenseLoader, RankLoader, partition helpers, upload / expert pool', 'byte spans, block-aligned', 'two ranks')], compile='header-only', tp=True),
 F('runtime.maya_tp.mm3_tp', TP + 'maya-mm3\'s tensor-parallel path (mm3_tp.cu / .hpp, 2,884 lines): GPUs in groups of tp, each group a run of layers, every layer cut over its members (query heads and their KV heads, index heads, o_proj columns, 1/tp of every FFN width); the residual stays on the host, members\' partials come back through pinned memory with a counter each and are summed in member order, routed experts outside VRAM run on the CPU lane as tp sub-experts',
   'a single-host multi-GPU TP with a host-side reduction, the contrast case to the two-host RDMA TP; the member-order host sum is a deterministic all-reduce without peer access',
   [MM + 'mm3_tp.cu', MM + 'mm3_tp.hpp'],
   [P('collective.host-sum-tp/-/f32.maya-tp-mm3', 'mm3::TpEngine (mm3_tp.hpp)', 'FP32 partials summed on the host in member order', 'several GPUs, one host')], compile='n/a', tp=True,
   notes='Not compiled alone: ' + HOST + '.'),
 # ---- runtime: GLM engine --------------------------------------------------------------------------------------------------
 F('runtime.maya_tp.glm_model', 'the glm5-next / GLM-5.3-Flash runner (glm_model.cu, 4,206 lines) with several conversations decoded together (glm_rows, glm_multi), the residual seams (glm_residual), the state arena and its layout descriptor (glm_state_bind, glm_state_layout), the memory map and split-device rules',
   'the executor that the TP2 context plugs into: where the two all-reduce points sit in a layer and how a rank\'s slices are bound; the executor shape a strixflerp GLM recipe would follow',
   ['src/core/glm_model.cu', 'src/core/glm_rows.cu', 'src/core/glm_rows_state.hpp', 'src/core/glm_multi.cu', 'src/core/glm_residual.cu', 'src/core/glm_residual.hpp', 'src/core/glm_state_bind.cpp', 'src/core/glm_state_layout.cpp', 'src/core/glm_memory.hpp'],
   [P('executor.glm5-next/-/-', 'strata::Glm5Model (include/strata/core/glm_model.hpp)', 'per-kernel numerics', 'one host or one rank of two')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 F('runtime.maya_tp.glm_fast_path', 'the glm5-next FAST decode path (glm_fast_path.cu, 8,758 lines, with glm_fast_state.hpp, the opt-in layer graph capture glm_layer_graphs.hpp and the per-request decode knobs glm_req_knobs.cpp): per-layer captured graphs, the fast binder\'s weights and the decode loop that calls the TP2 all-reduce',
   'the hot loop of GLM-5.3-Flash decode, including where Maya-TP hooks the queued exchange; the reference for wiring a device-stage collective into a graph-captured decode',
   ['src/core/glm_fast_path.cu', 'src/core/glm_fast_state.hpp', 'src/core/glm_layer_graphs.hpp', 'src/core/glm_req_knobs.cpp'],
   [P('executor.glm5-next-fast-decode/-/-', 'Glm5Model fast-path members', 'per-kernel numerics', 'decode')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 F('runtime.maya_tp.glm_prefill', 'the glm5-next batched prompt path (glm_prefill.cu, 5,273 lines) with its FP16 weight types (glm_prefill_types.hpp)',
   'GLM prompt processing in chunks; under TP2 both ranks agree on chunk sizes that fit the transport\'s receive window',
   ['src/core/glm_prefill.cu', 'src/core/glm_prefill_types.hpp'],
   [P('executor.glm5-next-prefill/-/-', 'Glm5Model prompt members', 'FP16 weights, FP32 accumulate', 'prompt chunks')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 F('runtime.maya_tp.glm_dsa_decode', 'the glm-dsa (GLM-5.3 DSA) decode side: the fast binder and arena (glm_dsa_fast, glm_dsa_fast_plan), the per-part decode graph cache, the decode attention mixer (glm_dsa_mixer and its no-GPU stub), the geometry reader and validation (glm_dsa_geometry, glm_geometry_check), the layer split rules (glm_dsa_split), the host halves of the guards and the seam-dump policy',
   'GLM-5.3 with DeepSeek sparse attention as Maya ported it; the binder / geometry split is how an engine accepts several GLM variants from one pack reader',
   ['src/core/glm_dsa_fast.cu', 'src/core/glm_dsa_fast.hpp', 'src/core/glm_dsa_fast_plan.cpp', 'src/core/glm_dsa_fast_plan.hpp', 'src/core/glm_dsa_graphs.hpp', 'src/core/glm_dsa_mixer.cu', 'src/core/glm_dsa_mixer.hpp', 'src/core/glm_dsa_mixer_stub.cpp', 'src/core/glm_dsa_geometry.cpp', 'src/core/glm_geometry_check.cpp', 'src/core/glm_dsa_split.cpp', 'src/core/glm_guards_host.cpp', 'src/core/glm_cb_dump.hpp'],
   [P('executor.glm-dsa-decode/-/-', 'glm-dsa fast binder and mixer', 'per-kernel numerics', 'decode')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 F('runtime.maya_tp.glm_dsa_prompt', 'the glm-dsa prompt side: the prompt layer (glm_prefill_dsa), the batched prompt plan (glm_dsa_prompt_plan), the prompt\'s host share on a node-split layer and the expert streaming driver over PCIe links with its pure plan (glm_dsa_prefill_share, glm_dsa_prefill_stream, glm_dsa_prefill_stream_plan)',
   'prompt processing for DSA GLM with experts streamed while computing; the stream plan is a reusable overlap scheduler',
   ['src/core/glm_prefill_dsa.cu', 'src/core/glm_prefill_dsa.hpp', 'src/core/glm_dsa_prompt_plan.cpp', 'src/core/glm_dsa_prefill_share.cpp', 'src/core/glm_dsa_prefill_stream.cpp', 'src/core/glm_dsa_prefill_stream.hpp', 'src/core/glm_dsa_prefill_stream_plan.cpp'],
   [P('executor.glm-dsa-prefill/-/-', 'glm-dsa prompt layer and stream driver', 'FP16 / FP32', 'prompt')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 F('runtime.maya_tp.glm_dsa_reference', 'the glm-dsa F32 reference forward (glm_dsa_ref.cu with its host pieces and no-GPU stub) and the CPU twins of the reference, fast and Volta decode kernels (glm_dsa_cpu_ref, glm_dsa_cpu_fast_twin, glm_dsa_cpu_volta_twin)',
   'a reference forward and CPU twins are the oracle a parity gate needs for every GLM DSA kernel; they transfer as strixflerp verify references',
   ['src/core/glm_dsa_ref.cu', 'src/core/glm_dsa_ref_host.cpp', 'src/core/glm_dsa_ref_stub.cpp', 'src/core/glm_dsa_cpu_ref.cpp', 'src/core/glm_dsa_cpu_fast_twin.cpp', 'src/core/glm_dsa_cpu_volta_twin.cpp'],
   [P('verify.reference-forward-glm-dsa/-/-', 'Glm5Model::step_layers_dsa, CPU twins', 'F32', 'parity runs')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 F('runtime.maya_tp.glm_mtp', 'GLM MTP speculative decoding: the one-GPU MTP decode where the NextN block drafts (glm_mtp), several conversations drafting at once (glm_mtp_rows), the NextN carrier\'s sequence state (glm_mtp_state, glm_mtp_state_model) and the speculative verify window on the trunk (glm_window)',
   'MTP under TP2 is maya-tp\'s biggest measured win (+29.6% code, two drafts): rank 0 coordinates drafts and acceptance while trunk partials are summed; the shape a strixflerp speculation.draft-mtp module takes',
   ['src/core/glm_mtp.cu', 'src/core/glm_mtp_rows.cu', 'src/core/glm_mtp_state.cpp', 'src/core/glm_mtp_state_model.cu', 'src/core/glm_window.cu'],
   [P('speculation.draft-mtp/-/-', 'Glm5Model MTP members (mtp_rows.hpp, glm_mtp_state.hpp)', 'per-kernel numerics', 'drafts 1-7')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 F('runtime.maya_tp.glm_dflash', 'the DFlash2 block drafter and its decode on the GLM trunk (glm_dflash.cu, 2,075 lines) with the adaptive draft-length policy (glm_draft_adaptive.hpp)',
   'a block-diffusion style drafter for GLM beside MTP; a second speculative path with its own acceptance policy',
   ['src/core/glm_dflash.cu'],
   [P('speculation.draft-dflash/-/-', 'GlmDflash (glm_dflash.hpp)', 'per-kernel numerics', 'block drafts')], compile='n/a', notes='Not compiled alone: ' + HOST + '. The DFlash2 draft weights are not part of the tree.'),
 F('runtime.maya_tp.glm_ngram', 'the n-gram draft matcher and its decode_spec session (glm_ngram_draft) and head-group pricing with acceptance accounting (glm_dsa_spec)',
   'free drafts from the context itself; the pricing / acceptance accounting decides when speculation pays',
   ['src/core/glm_ngram_draft.cpp', 'src/core/glm_dsa_spec.cpp'],
   [P('speculation.draft-ngram/-/-', 'glm_ngram_draft.hpp, glm_dsa_spec.hpp', 'integer matching', 'decode')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 F('runtime.maya_tp.glm_tiers', 'GLM expert memory tiers: the run-time tiers over NUMA / VRAM / host (glm_tier_numa), their host-side rules (glm_tier_plan), the node-split CPU lane arithmetic (glm_tier_lane), heat-decay promotion / eviction (glm_heat_evict), the slow-path pool size classes (glm_pool_classes) and the file-order warm-up (glm_warm_seq, glm_warm_seq_model)',
   'where each expert lives and when it moves is the scheduler that decides decode speed for models bigger than VRAM; heat-decay eviction is a policy strixflerp has no module for',
   ['src/core/glm_tier_numa.cu', 'src/core/glm_tier_numa.hpp', 'src/core/glm_tier_plan.cpp', 'src/core/glm_tier_plan.hpp', 'src/core/glm_tier_lane.cpp', 'src/core/glm_tier_lane.hpp', 'src/core/glm_heat_evict.cpp', 'src/core/glm_pool_classes.cpp', 'src/core/glm_warm_seq.cpp', 'src/core/glm_warm_seq_model.cu'],
   [P('sched.expert-tiers/-/-', 'glm_tier_numa.hpp, glm_tier_plan.hpp, glm_heat_evict.hpp, glm_warm_seq.hpp', 'no arithmetic of its own', 'models larger than VRAM')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 F('runtime.maya_tp.glm_prompt_cache', 'the content-addressed NVMe prompt cache for GLM (glm_prompt_cache store, its engine glue and model side)',
   'cached follow-ups are what make a coding agent fast; a content-addressed prompt cache beside cache.prompt-gufo and cache.prompt-strixfold',
   ['src/core/glm_prompt_cache.cpp', 'src/core/glm_prompt_cache_engine.cpp', 'src/core/glm_prompt_cache_model.cu'],
   [P('cache.prompt-maya-tp/-/-', 'glm_prompt_cache.hpp', 'state bytes, no arithmetic', 'follow-up turns')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 F('runtime.maya_tp.glm_tools', 'GLM engine side tools: refusal ablation on the GLM residual (glm_ablate.cu / .hpp, ablate_capture.hpp), the CPU routed-expert speed probe (glm_cpu_bench) and the Maya v1.1.3 feature sync against glm-dsa (glm_maya_sync)',
   'the steering / ablation hook is the same shape as strixflerp\'s steer modules; the CPU expert probe sizes a CPU lane before it is used',
   ['src/core/glm_ablate.cu', 'src/core/glm_ablate.hpp', 'src/core/ablate_capture.hpp', 'src/core/glm_cpu_bench.cpp', 'src/core/glm_maya_sync.cpp'],
   [P('steer.ablate-direction/-/f32.maya-tp', 'glm_ablate.hpp', 'FP32 projection', 'decode')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 # ---- runtime: shared engine core ------------------------------------------------------------------------------------------
 F('runtime.maya_tp.core_session', 'the Strata-lineage engine core: one token through all layers with per-layer graphs (session), the composed GDN layer (layer), the graph registry, the shape checks (layout), the dense-weight loader (weights), native dense and head layers, the device arena (device.cu) and the pinned host arena with the parallel expert load (pinned.cu)',
   'the executor skeleton every Maya engine reuses; the per-layer graph and arena rules carry to strixflerp executors',
   ['src/core/session.cpp', 'src/core/layer.cpp', 'src/core/graph.cpp', 'src/core/layout.cpp', 'src/core/weights.cpp', 'src/core/native_dense.cpp', 'src/core/native_head.cpp', 'src/core/device.cu', 'src/core/pinned.cu'],
   [P('executor.strata-core/-/-', 'strata::core::Session, Layer, GraphRegistry, Weights (include/strata/core/)', 'per-kernel numerics', 'one host')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 F('runtime.maya_tp.expert_store', 'the expert storage tier: R4\'s slot storage and residency table (expert_cache), the expert-source adapter with its three-clause contract (expert_source, 2,114 lines), remote experts (remote_experts) and the hit hook (hit_hook.hpp)',
   'the VRAM expert cache and its source contract decide which experts are resident; a cache module that sits under any MoE executor',
   ['src/core/expert_cache.cpp', 'src/core/expert_source.cpp', 'src/core/remote_experts.cpp'],
   [P('sched.expert-cache/-/-', 'strata::core::ExpertCache, ExpertSource, RemoteExperts', 'no arithmetic of its own', 'decode with experts larger than VRAM')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 F('runtime.maya_tp.verify_mtp', 'the Strata-lineage speculative verify and MTP drivers (verify.cpp, mtp.cpp) with the MTP row helpers',
   'the engine-agnostic verify window driver that the GLM, DeepSeek and MiniMax engines reuse; under TP2 rank 0 owns acceptance',
   ['src/core/verify.cpp', 'src/core/mtp.cpp'],
   [P('speculation.window-verify-driver/-/-', 'include/strata/core/verify.hpp, mtp.hpp', 'greedy or keyed acceptance', 'verify windows')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 F('runtime.maya_tp.spec_controller', 'speculative-decoding control: the controller, the draft-length policy with its cost model and the suffix drafter (src/spec), plus the n-gram matcher kernel host code (src/kernels/ngram.cpp)',
   'draft-length choice is where MTP gains are won or lost (seven drafts cost 38.6% on the technical prompt); a policy module strixflerp\'s speculation family can take whole',
   ['src/spec/controller.cpp', 'src/spec/draft_policy.cpp', 'src/spec/suffix_drafter.cpp', 'src/kernels/ngram.cpp'],
   [P('speculation.policy/-/-', 'strata::spec::Controller, DraftPolicy, SuffixDrafter (include/strata/spec/)', 'cost model in tokens / s', 'any drafter')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 F('runtime.maya_tp.prefill_host', 'the prompt path\'s host orchestration (src/prefill/prefill.cpp, 2,026 lines) over the prefill kernels and MoE prompt modules',
   'chunking, GEMM choice and expert batching for prompts; the driver a strixflerp prefill recipe would mirror',
   [PF + 'prefill.cpp'],
   [P('executor.prefill-driver/-/-', 'strata::prefill (include/strata/prefill/prefill.hpp)', 'per-kernel numerics', 'prompt')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 F('runtime.maya_tp.generate', 'the generate program (src/program/generate.cpp, 10,336 lines): the strata generate / maya engine process loop that serves the HTTP front end, with the conversation cache and logits selection',
   'the engine process the server talks to, including the TP2 rank-0 / rank-1 paths and slot handling; it shows how an engine exposes TP2 to a server',
   ['src/program/generate.cpp'],
   [P('serve.program-maya-tp/-/-', 'main() of the generate program, conv_cache.hpp, logits_selection.hpp', 'no arithmetic of its own', 'one host or one rank of two')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 F('runtime.maya_tp.ds4_engine', 'maya-ds4, DeepSeek-V4.1-Flash on Maya (ds4_model.cu, 22,390 lines; ds4_model.hpp, ds4_multi_attn, ds4_logprobs, ds4_evt, ds4_marks, ds4_watchdog, ds4_path_policy, ds4_prefill_stats, ds4_prompt_mem, ds4_markov_q8, ds4_ablate, ds4_chunk_policy)',
   'the DeepSeek executor the TP2 DSpark path runs on both ranks; the reference for a DeepSeek-V4.1 TP2 recipe beside strixfold\'s TP4',
   [DS + 'ds4_model.cu', DS + 'ds4_model.hpp', DS + 'ds4_multi_attn.hpp', DS + 'ds4_logprobs.hpp', DS + 'ds4_evt.hpp', DS + 'ds4_marks.hpp', DS + 'ds4_watchdog.hpp', DS + 'ds4_path_policy.hpp', DS + 'ds4_prefill_stats.hpp', DS + 'ds4_prompt_mem.hpp', DS + 'ds4_markov_q8.hpp', DS + 'ds4_ablate.hpp', DS + 'ds4_chunk_policy.hpp'],
   [P('executor.dsv41-maya-tp/-/-', 'ds4 Model (ds4_model.hpp)', 'per-kernel numerics', 'one host or one rank of two')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 F('runtime.maya_tp.ds4_main', 'the maya-ds4 program (ds4_main.cpp, 4,122 lines): options, the TP2 rank setup, DSpark drafting and the serve loop',
   'where the DeepSeek pair is launched and its control channel wired; the entry a TP2 launcher calls on each host',
   [DS + 'ds4_main.cpp'],
   [P('serve.program-ds4-maya-tp/-/-', 'main() (ds4_main.cpp)', 'no arithmetic of its own', 'one host or rank 0 / 1')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 F('runtime.maya_tp.ds4_store', 'maya-ds4\'s weight and state storage: the expert store and pool (ds4_store, ds4_pool), the engram (prompt state) file I/O (ds4_engram_io), RAM slices, borrowed buffers, the pack info reader (ds4_packinfo) and the OS helpers (ds4_os)',
   'how DeepSeek weights and saved states move between disk, host and GPU; the engram file is a prompt-state cache format',
   [DS + 'ds4_store.hpp', DS + 'ds4_pool.hpp', DS + 'ds4_engram_io.hpp', DS + 'ds4_ram_slice.hpp', DS + 'ds4_borrow.hpp', DS + 'ds4_packinfo.cpp', DS + 'ds4_packinfo.hpp', DS + 'ds4_os.hpp'],
   [P('cache.prompt-engram-maya-tp/-/-', 'ds4 store / engram I/O', 'bytes', 'DeepSeek states and experts')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 F('runtime.maya_tp.ds4_cpu', 'maya-ds4\'s CPU lane: GGUF experts on the CPU (ds4_gguf_cpu), the i-quant CPU rows (ds4_iq_cpu), EXL3 on the CPU (ds4_cpu_exl3), row preparation (ds4_cpu_prepare_rows) and the host all-reduce (ds4_host_ar)',
   'DeepSeek experts on Zen 5 cores, and a host-side all-reduce for single-host multi-GPU, both reusable outside maya-ds4',
   [DS + 'ds4_gguf_cpu.cpp', DS + 'ds4_gguf_cpu.hpp', DS + 'ds4_iq_cpu.cpp', DS + 'ds4_cpu_exl3.hpp', DS + 'ds4_cpu_prepare_rows.hpp', DS + 'ds4_host_ar.cpp'],
   [P('moe.experts-decode-cpu/gguf-iq+exl3/f32.maya-tp-ds4', 'ds4 CPU lane', 'FP32', 'decode on the CPU')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 F('runtime.maya_tp.mm3_engine', 'maya-mm3, MiniMax-M3.1-Flash on Maya: the model with loading, placement and forward (mm3_model), the program (mm3_main), the CPU expert lane (mm3_cpu, mm3_cpu_gg), sampling, slots, lane plans, pack and geometry readers and the weight scan',
   'the third engine of the tree and the host for the single-host mm3 TP; the placement-over-GPUs logic is reusable',
   [MM + 'mm3_model.cu', MM + 'mm3_model.hpp', MM + 'mm3_main.cpp', MM + 'mm3_cpu.cpp', MM + 'mm3_cpu_gg.cpp', MM + 'mm3_cpu.hpp', MM + 'mm3_host.cpp', MM + 'mm3_host.hpp', MM + 'mm3_sample.hpp', MM + 'mm3_slots.hpp', MM + 'mm3_lane_plan.hpp', MM + 'mm3_pack.hpp', MM + 'mm3_geo.hpp', MM + 'mm3_wscan.hpp', MM + 'mm3_os.hpp', MM + 'mm3_env.hpp', MM + 'mm3_cands.hpp', MM + 'mm3_ablate.hpp'],
   [P('executor.minimax-m31-maya-tp/-/-', 'mm3::Model (mm3_model.hpp)', 'per-kernel numerics', 'one or several GPUs')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 F('runtime.maya_tp.mm3_dspark', 'maya-mm3\'s DSpark draft (mm3_dspark.cu / .hpp)',
   'DSpark is the drafter maya-tp runs under TP2 for DeepSeek; the MiniMax copy shows the drafter separated from its engine',
   [MM + 'mm3_dspark.cu', MM + 'mm3_dspark.hpp'],
   [P('speculation.draft-dspark/-/-', 'mm3 DSpark (mm3_dspark.hpp)', 'per-kernel numerics', 'block drafts')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 F('runtime.maya_tp.platform', 'platform helpers: direct (O_DIRECT) file reads, memory queries, the memory planner CLI (src/plan) and the platform / plan headers (env switches and aliases, idle gate, stall watch, GPU-shared memory, dedicated memory, CPU relax, commit)',
   'unified-memory accounting and direct I/O are what make weight streaming work on an APU; small pieces any engine can take',
   ['src/platform/direct_file.cpp', 'src/platform/memory.cpp', 'src/plan/plan_main.cpp', 'include/strata/platform/', 'include/strata/plan/'],
   [P('platform.memory-io/-/-', 'strata::platform (include/strata/platform/), strata-plan', 'no arithmetic', 'any host')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 F('format.maya_tp.gguf_reader', 'the header-only GGUF reader and dequantizer with their CLIs (src/artifact: strata-gguf, strata-dequant; include/strata/artifact)',
   'a small dependency-free GGUF reader is the first piece of any loader; a format.reader provider',
   ['src/artifact/gguf_reader.cpp', 'src/artifact/dequant.cpp', 'include/strata/artifact/'],
   [P('format.reader-maya-tp/-/-', 'strata::artifact gguf_reader.hpp, dequant.hpp', 'dequant to FP32', 'any GGUF')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 F('format.maya_tp.ple_reader', 'the PLE table reader (src/ngram/ple_reader.cpp, include/strata/ngram/ple_reader.hpp): the per-layer embedding table streamed from disk',
   'the format side of the PLE input path; pairs with kernel.maya_tp.ple',
   ['src/ngram/ple_reader.cpp', 'include/strata/ngram/'],
   [P('format.reader-ple/-/-', 'strata::ngram::PleReader', 'bytes', 'decode')], compile='n/a', notes='Not compiled alone: ' + HOST + '.'),
 F('runtime.maya_tp.core_headers', 'the engine headers of include/strata/core and include/strata/kernels that no other module owns by name (glm_model.hpp, glm_arch, glm_env, session, layer, verify, mtp, prefill pieces and the kernel launch headers shared across families)',
   'a single header module so every engine and kernel module above resolves its #include lines the way the origin\'s include/ directory does',
   [], [P('kernel-abi.maya-tp-engine-headers/-/-', 'include/strata/core/*.hpp, include/strata/kernels/*.hpp', 'declarations only', 'all engines')], compile='header-only'),
 # ---- serve -------------------------------------------------------------------------------------------------------------
 F('serve.maya_tp.server', 'the Maya HTTP server (serve/server.py and its helpers: frontend, models and model API, model pulls, GPU map, telemetry, savings, loop guard, update): OpenAI- and Anthropic-compatible endpoints, streaming, cached follow-ups, cancellation, thinking budgets, and in maya-tp the TP2 peer health and recovery reporting',
   'the serving shape that drives a TP2 pair as one model; peer-loss handling at the HTTP layer is what strixflerp\'s serve modules lack for multi-node engines',
   ['serve/server.py', 'serve/frontend.py', 'serve/models.py', 'serve/model_api.py', 'serve/model_pull.py', 'serve/gpu_map.py', 'serve/telemetry.py', 'serve/savings.py', 'serve/loop_guard.py', 'serve/update.py', 'serve/__init__.py'],
   [P('serve.http-maya-tp/-/-', 'serve/server.py', 'no arithmetic', 'one engine or a TP2 pair')], compile='n/a', lang='python'),
 F('serve.maya_tp.responses', 'the Responses API layer (serve/responses.py, response_config, response_jobs, response_schema, response_store): background jobs, stored responses and schema-constrained output',
   'an OpenAI Responses-compatible layer with stored jobs, reusable over any engine behind strixflerp\'s serve front',
   ['serve/responses.py', 'serve/response_config.py', 'serve/response_jobs.py', 'serve/response_schema.py', 'serve/response_store.py'],
   [P('serve.responses-api/-/-', 'serve/responses.py', 'no arithmetic', 'HTTP')], compile='n/a', lang='python'),
 F('serve.maya_tp.templates', 'chat formatting for the served models: GLM-5 / glm-dsa templates and product settings (glm5_chat_template.jinja, glm_dsa_template.py, glm_dsa_product.py, glm_dsa_tune.py), DeepSeek-V4.1 encoding (ds41_encoding.py) and the generic template',
   'getting the GLM-5.3 and DeepSeek chat formats byte-exact is a precondition for tool use; templates are cheap to transfer and expensive to get wrong',
   ['serve/glm5_chat_template.jinja', 'serve/chat_template.jinja', 'serve/glm_dsa_template.py', 'serve/glm_dsa_product.py', 'serve/glm_dsa_tune.py', 'serve/ds41_encoding.py', 'serve/chat_golden.json'],
   [P('serve.chat-template-glm5/-/-', 'serve/glm_dsa_template.py, glm5_chat_template.jinja', 'text', 'GLM-5.x, DeepSeek-V4.1')], compile='n/a', lang='python'),
 F('serve.maya_tp.tools_mcp', 'tool and attachment handling in the server: MCP client (mcp.py), attachments (attach.py) and PDF text extraction (pdf.py, pdf_text.py)',
   'coding-agent use needs tools and file inputs; an MCP client at the serve layer is a piece strixflerp\'s serve has no counterpart for',
   ['serve/mcp.py', 'serve/attach.py', 'serve/pdf.py', 'serve/pdf_text.py'],
   [P('serve.tools-mcp/-/-', 'serve/mcp.py', 'no arithmetic', 'HTTP')], compile='n/a', lang='python'),
 F('serve.maya_tp.vision_front', 'the server side of image input for DeepSeek-V4.1 and MiniMax-M3.1 (serve/ds4_vision.py, serve/mm3_vision.py)',
   'image preprocessing in front of the two vision towers (kernel.maya_tp.ds4_vision, mm3_vision)',
   ['serve/ds4_vision.py', 'serve/mm3_vision.py'],
   [P('serve.vision-preprocess/-/-', 'serve/ds4_vision.py, mm3_vision.py', 'image to patches', 'HTTP')], compile='n/a', lang='python'),
 # ---- tools -------------------------------------------------------------------------------------------------------------
 F('tool.maya_tp.tp2_launch', TP + 'launching a TP2 pair: the Python launcher (tools/tp2/launch.py, owned_exec.py), the rank scripts (rank0.sh, rank1.sh, maya-rank0.sh, maya-rank1.sh, model_env.sh), the run / profile / no-all-reduce scripts and the llama-swap wrappers (tools/llama-swap/tp2.sh, tp2-ds4.sh, tools/tp2/tp2-swap.sh) that start / stop both ranks and forward MTP draft count, capacity and context',
   'one command that brings up both hosts and tears them down together is what makes a two-node engine operable; the llama-swap wrapper shape fits the pod\'s llama-swap',
   ['tools/tp2/launch.py', 'tools/tp2/owned_exec.py', 'tools/tp2/rank0.sh', 'tools/tp2/rank1.sh', 'tools/tp2/maya-rank0.sh', 'tools/tp2/maya-rank1.sh', 'tools/tp2/model_env.sh', 'tools/tp2/run_bench.sh', 'tools/tp2/run_comm_test.sh', 'tools/tp2/run_noar.sh', 'tools/tp2/run_prof.sh', 'tools/tp2/tp2-swap.sh', 'tools/llama-swap/'],
   [P('tool.tp2-launch/-/-', 'tools/tp2/launch.py, tools/llama-swap/tp2.sh', 'no arithmetic', 'two hosts over usb4_rdma')], compile='n/a', lang='python', tp=True),
 F('tool.maya_tp.tp2_validation', TP + 'validating and benchmarking a TP2 pair: acceptance suite (acceptance.sh), split audit, greedy parity, MTP acceptance and bench, prefill bench and cancel, continuation boundaries, long-loop reuse, collective regressions and batch validation (tools/tp2/*.py)',
   'the regression harness that proved TP2 / MTP on Maya-S, M and L; parity and collective-regression checks are what a strixflerp TP2 gate needs before any timing',
   ['tools/tp2/acceptance.sh', 'tools/tp2/audit_split.py', 'tools/tp2/greedy_parity.py', 'tools/tp2/mtp_acceptance.py', 'tools/tp2/bench_mtp.py', 'tools/tp2/bench_prefill.py', 'tools/tp2/prefill_cancel.py', 'tools/tp2/continuation_boundaries.py', 'tools/tp2/long_loop_reuse.py', 'tools/tp2/regression_collectives.py', 'tools/tp2/validate_batch.py'],
   [P('verify.tp2-parity-suite/-/-', 'tools/tp2/*.py', 'greedy text parity, acceptance rates, tok/s', 'two hosts')], compile='n/a', lang='python', tp=True),
 F('tool.maya_tp.tp_tests', TP + 'the TP unit and pair tests: tests/tp_*.cpp (collective, RDMA communicator, context, control-socket state, partition, slice), tests/ds4_tp_*.cpp / .cu / .py (channel, control, batch, chunk, identity, loaders, workers, sharded DSpark / rollback / sample, pair smoke), tests/glm_tp*, the down-column band test, the server\'s TP2 boundary and peer-loss tests and tools/test_tp*.py / tools/test_*tp2*.py with the TP calibration script',
   'tests are the executable spec of the TP2 protocol and partition rules; they port with the modules they test',
   ['tests/tp_collective_test.cpp', 'tests/tp_comm_rdma_test.cpp', 'tests/tp_context_test.cpp', 'tests/tp_control_socket_state_test.cpp', 'tests/tp_partition_test.cpp', 'tests/tp_slice_test.cpp', 'tests/ds4_tp_*', 'tests/glm_tp*', CPU + 'native_down_cols_test.cpp', 'serve/test_tp2_boundaries.py', 'serve/test_tp2_peer_loss.py', 'tools/test_tp*.py', 'tools/test_*tp2*.py', 'tools/calibrate_glm_tp.py'],
   [P('verify.tp2-unit-tests/-/-', 'the tests\' main() / unittest classes', 'exact protocol and partition checks', 'one host (fakes) or two hosts (rdma tests)')], compile='n/a', lang='c++', tp=True),
 F('tool.maya_tp.kernel_parity', 'the kernel parity programs and tests (src/kernels/*_parity.cpp, *_test.cpp, qsa_select_bench.cpp and the oracle vectors .inc)',
   'each kernel family\'s parity program is its oracle; they are what a strixflerp parity gate would run against these modules',
   ['src/kernels/*_parity.cpp', 'src/kernels/*_test.cpp', 'src/kernels/qsa_select_bench.cpp', 'src/kernels/*.inc'],
   [P('verify.kernel-parity-maya-tp/-/-', 'the parity programs\' main()', 'per-family tolerances', 'one GPU')], compile='n/a', lang='hip-c++'),
 F('tool.maya_tp.hip_tests', 'the HIP-specific tests and benches (tests/hip): expert-cache staging, expert kernel bench, GLM attention / mv / prefill benches, handoff, intrinsics, native QSA score, PLE IQ4, prefill fused / MMQ / hipBLASLt / native batch parity',
   'the AMD-only checks of the kernels on gfx1151, including the WMMA intrinsics test; the bench programs are the timing harness for a later experiment card',
   ['tests/hip/'],
   [P('verify.hip-kernel-tests-maya-tp/-/-', 'tests/hip/*.cpp / .cu main()', 'per-test tolerances', 'one AMD GPU')], compile='n/a', lang='hip-c++'),
 F('tool.maya_tp.hipblaslt_tuning', 'the hipBLASLt tuning tables and tuner (tools/hip): solution tables per gfx target and hipBLASLt version (gfx1151 GLM-5.3 tables included), the tuner tune_hipblaslt.cpp and the prefill bench driver',
   'measured GEMM solution tables for gfx1151 are directly reusable by any strixflerp prefill module on the same toolchain',
   ['tools/hip/'],
   [P('tool.hipblaslt-tables/-/-', 'tools/hip/tune_hipblaslt.cpp, gfx*-hipblaslt-*.txt', 'solution ids per GEMM shape', 'gfx1100 / gfx1151 / gfx1201')], compile='n/a', lang='c++'),
 F('tool.maya_tp.engine_probes', 'the engine\'s small measurement programs: strata-device (GPU, plan and arena report), strata-load (expert arena load), strata-overlap (does streaming overlap compute?) and the concurrency probe (do the CPU expert path and the GPU stream run at once?)',
   'each answers one yes / no question about a host before an engine is tuned; the overlap and concurrency probes matter on unified memory',
   ['src/core/device_main.cpp', 'src/core/load_main.cpp', 'src/core/overlap_main.cpp', 'src/core/concurrent_main.cpp'],
   [P('tool.engine-probes/-/-', 'main() of each program', 'timings', 'one host')], compile='n/a', lang='hip-c++'),
]

FAMILY_IDS = [f['id'] for f in FAMILIES]
assert len(set(FAMILY_IDS)) == len(FAMILY_IDS)

# ---- headers: include/strata/** goes to the module that owns a source of the same stem, else to a header module -----------
files = O.files()
expl = {}
for f in FAMILIES:
    out = O.expand(f['paths']) if f['paths'] else []
    if f['paths'] and not out:
        raise SystemExit(f"{f['id']}: no origin files match {f['paths']}")
    expl[f['id']] = out
taken = {p: mid for mid, ps in expl.items() for p in ps}
dupe = [p for p in taken if sum(p in ps for ps in expl.values()) > 1]
if dupe:
    raise SystemExit('files in two families: ' + ' '.join(dupe[:20]))
stem_owner = {}
for mid, ps in expl.items():
    for p in ps:
        if p.startswith(('src/', 'serve/')) and p.endswith(('.cu', '.cpp')):
            stem_owner.setdefault(os.path.splitext(os.path.basename(p))[0], mid)
HEADER_HOME = {'include/strata/net/communicator.hpp': 'runtime.maya_tp.tp_comm_abi', IN + 'tp_wire.hpp': 'runtime.maya_tp.tp_comm_abi',
               IN + 'verbs.hpp': 'runtime.maya_tp.tp_rdma', IN + 'rdma_gid.hpp': 'runtime.maya_tp.tp_rdma',
               IN + 'tp_kernels.hpp': 'kernel.maya_tp.tp_kernels',
               IC + 'tp_context.hpp': 'runtime.maya_tp.tp_context', IC + 'tp_partition.hpp': 'runtime.maya_tp.tp_context',
               IC + 'tp_collective.hpp': 'runtime.maya_tp.tp_context', IC + 'glm_tp_batch.hpp': 'runtime.maya_tp.glm_tp_batch',
               IC + 'glm_draft_adaptive.hpp': 'runtime.maya_tp.glm_dflash', IC + 'mtp_rows.hpp': 'runtime.maya_tp.glm_mtp',
               IC + 'hit_hook.hpp': 'runtime.maya_tp.expert_store', IC + 'glm_env.hpp': 'runtime.maya_tp.glm_fast_path',
               'include/strata/prefill/moe_mmq_strata.hpp': 'kernel.maya_tp.prefill_moe_mmq',
               'include/strata/prefill/failure.hpp': 'runtime.maya_tp.prefill_host',
               'include/strata/program/conv_cache.hpp': 'runtime.maya_tp.generate', 'include/strata/program/logits_selection.hpp': 'runtime.maya_tp.generate'}
for h in files:
    if not h.startswith('include/strata/') or h in taken:
        continue
    stem = os.path.splitext(os.path.basename(h))[0]
    if h in HEADER_HOME:
        home = HEADER_HOME[h]
    elif h.startswith('include/strata/kernels/cpu/'):
        home = stem_owner.get(stem) or 'kernel.maya_tp.cpu_pool'
    elif h.startswith('include/strata/spec/'):
        home = 'runtime.maya_tp.spec_controller'
    elif stem in stem_owner and not stem_owner[stem].startswith('tool.'):
        home = stem_owner[stem]
    elif h.startswith('include/strata/kernels/') and stem in ('bf16_bits', 'f16_bits', 'dp4a'):
        home = 'kernel.maya_tp.common'
    else:
        home = 'runtime.maya_tp.core_headers'
    expl[home].append(h)
    taken[h] = home

INCLUDES = ['src/include/strata/hip_compat', 'src/include', 'src/third_party/ggml', 'src/src/kernels/cuda', 'src/src/ds4', 'src/src/mm3', 'src/src/core', 'src/src/net']
mods = []
for f in FAMILIES:
    paths = sorted(expl[f['id']])
    kind = f['id'].split('.')[0]
    comp = f['compile'] or ('standalone' if f['lang'] == 'hip-c++' and any(p.endswith('.cu') for p in paths) else 'n/a')
    sources = [p for p in paths if p.endswith('.cu') and p not in f['nosrc']] if comp == 'standalone' else []
    if comp == 'standalone' and f['id'] == 'runtime.maya_tp.tp_rdma':
        sources = [p for p in paths if p.endswith('.cpp')]
    if comp == 'standalone' and not sources:
        comp = 'header-only' if f['lang'] == 'hip-c++' else 'n/a'
    reason = None
    if comp == 'n/a':
        reason = (f['notes'] or HOST).replace('Not compiled alone: ', '').replace('Not compiled by the gate: ', '').replace('Not compiled: ', '')[:300]
    elif comp == 'header-only':
        reason = 'header-only; compiled inside the modules that include it'
    tags = ', '.join(p['interface'] for p in f['provides'])
    notes = 'Interface tag: %s. %s%s' % (tags if kind == 'kernel' or f['provides'] else 'none', (f['notes'] + ' ') if f['notes'] else '',
                                         'Part of the TP2 delta over Project Maya (data/maya_tp/base-diff.tsv rows A). ' if f['tp'] else '')
    notes += 'Nothing was run or timed on the pod.'
    flags = dict(f['flags'])
    if f['id'] == 'runtime.maya_tp.tp_rdma':
        # strata_net is a plain C++ target (CMakeLists.txt): hip::host and libibverbs, -include format, no hip_compat force-include
        flags = {}
        kw_flags = ['-x', 'c++', '-std=c++20', '-O3', '-DNDEBUG', '-fPIC', '-D__HIP_PLATFORM_AMD__=1', '-include', 'format', '-c']
    else:
        kw_flags = None
    kw = dict(provides=f['provides'], requires=[] if f['id'] == 'kernel.maya_tp.common' or kind != 'kernel' else ['module:kernel.maya_tp.common'],
              sources=sources, includes=INCLUDES if f['lang'] in ('hip-c++', 'c++') else [], why=f['why'], notes=notes, compile=comp,
              reason=reason, lang=f['lang'])
    if f['lang'] == 'python':
        kw['toolchain'] = ''
    if kw_flags:
        kw['flags'] = kw_flags
    if flags and sources:
        kw['source_flags'] = {k: v for k, v in flags.items() if k in sources}
    if f['id'] in ('kernel.maya_tp.common',) or kind == 'kernel' and f['lang'] == 'hip-c++':
        kw['authors'] = LLAMA if any('native_' in p or 'third_party/ggml' in p for p in paths) else None
    mods.append(M(f['id'], f['summary'], paths, **kw))

ORIGIN_TOML = '''
# Maya-TP: tensor-parallel inference for large MoE models on two AMD Strix Halo hosts joined by Thunderbolt / USB4
# RDMA (usb4_rdma, RoCE v2), derived from Project Maya (mw00/project-maya, itself grown out of Niko1221/Strata, so the
# sources keep the strata:: names). GLM-5.3-Flash with intra-expert TP2 and MTP, DeepSeek-V4.1-Flash TP2 with DSpark,
# a GPU-staged queued all-reduce over RDMA write windows ported from gufo (card ORIGIN-MAYA-TP 2982083a71).
name      = "maya_tp"
repo      = "https://github.com/uv-genai/maya-tp"
commit    = "6951bd7efb3c76b088e663645f274431b22dd913"
branch    = "main"
version   = "tp-v1.1.45 (VERSION 1.1.45), head commit 'Merge pull request #184 from uv-genai/claude/v65-accepted' (2026-11-18); integrates upstream Project Maya v1.1.54-v1.1.58 (README, issue #159). Measured by the origin (README, docs/benchmarks): GLM-5.3-Flash Maya-S IQ2_XXS 18.55 tok/s on one host, 25.82 on the pair, 28.60 with MTP one draft (technical), 33.53 with two drafts (code); link ~37 Gb/s large transfers, ~6 us tiny RDMA write round trips"
base      = "mw00/project-maya: maya-tp integrates v1.1.54-v1.1.58; the ark holds Project Maya HEAD 53885cef8faf (v1.1.81) and v1.1.78 1016876a387e (the base of catalog/kestrel), not v1.1.58. docs/harvest/data/maya_tp/base-diff.tsv lists every maya-tp file against both (A = only in maya-tp: the TP2 delta and the strata-named headers; M = changed). docs/harvest/data/maya_tp/project-maya-head-glm.tsv lists the GLM-5.3-Flash engine files Project Maya changed after v1.1.78 (recorded, not harvested)"
reference = "b-iurea/project-maya (ark /data/ark-inbox/src/github.com__b-iurea__project-maya): a research spec, named here as a reference note; no module comes from it"
reference_checkout = "control-node:/data/ark-inbox/src/github.com__uv-genai__maya-tp#6951bd7efb3c76b088e663645f274431b22dd913 (ark copy, a shallow clone, .git/HEAD 6951bd7e); laptop: ~/Code/strixflerp-upstream-supplement/maya_tp (the same 3,985 files, by tarball through control-node get); docs/harvest/harvest_maya_tp.py pins the tree digest"

[licence]
spdx      = "MIT"
files     = ["LICENSE"]                                  # copied verbatim next to this file
third_party = "LICENSE: MIT, Copyright (c) 2026 Peasant Smith and the Project Maya contributors; Copyright (c) 2026 Niko1221 and the Strata contributors. The native_*.cu kernels say they are adapted from llama.cpp 3cf03257 (MIT); third_party/ggml (ggml-common.h, VERSION.txt, LICENSE, mmq-fused-swiglu.patch) is llama.cpp's, MIT; third_party/exllamav3 and third_party/beellama.cpp hold only their MIT licence files (EXL3 arithmetic in exl3.cu follows exllamav3); moe_fused_rdna4.cu is adapted from Niko1221/Strata fb58e0d (MIT); src/net is a port of gufo's Ibrverbs. LICENSE.deepseek-v41, LICENSE.minimax-m31 and LICENSE.qwen-community-1.0 are model licences and are not harvested. third_party/wheels (Python wheels) is not harvested. Weights are not part of the tree."

[toolchain]
host      = "C++20 (CMAKE_CXX_STANDARD 20), CMake with the HIP language; -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON (required for HIP) -DSTRATA_ENABLE_TP2_RDMA=ON (libibverbs) for the pair; MAYA_* options map to STRATA_*"
device    = "therock-10.2"                               # the origin targets ROCm 7.x on gfx1151 (Linux); the catalogue's gfx1151 toolchain is named
build     = "cmake/hip_backend.cmake: every src/**/*.cu, bench/**/*.cu and tests/**/*.cu gets LANGUAGE HIP; strata_hip_runtime force-includes include/strata/hip_compat/cuda_runtime.h, puts include/strata/hip_compat before include/, defines STRATA_USE_HIP=1 and STRATA_HIP_ARCHS; strata_net (src/net/verbs.cpp, rdma_gid.cpp, tp_kernels.cu) adds -include format for C++ and links libibverbs"

[compile]                                                # CMake's HIP language: clang++ -x hip, one object per .cu
driver    = "<rocm>/lib/llvm/bin/clang++"
arch_flag = "--offload-arch=gfx1151"
rocm_flag = "--rocm-path=<rocm>"
flags     = ["-x", "hip", "-std=c++20", "-O3", "-DNDEBUG", "-fPIC", "-DSTRATA_USE_HIP=1", "-DSTRATA_HIP_ARCHS=\\"gfx1151\\"", "-DSTRATA_VERSION=\\"1.1.45\\"", "-include", "cuda_runtime.h", "-c"]
system_includes = ["include"]
output    = "host object with the gfx1151 device code embedded (.o), one per translation unit"
note      = "-include cuda_runtime.h resolves through the -I list to kernel.maya_tp.common's include/strata/hip_compat, as the origin's absolute -include path does. Per-source options from CMakeLists.txt are in each module's [build] source_options (-ffp-contract=off for elementwise.cu and quantize_act.cu, -ffp-contract=on for exl3.cu, STRATA_HIPBLASLT_AVAILABLE=1 for gemm.cu and ds4_lt.cu, -include format for src/net/*.cpp). -Wall -Wextra (strata_warnings) are left out."

[kernels]
loading   = "statically linked: each .cu holds its kernels and host launchers declared in include/strata/**; engines (maya-glm / strata generate, maya-ds4, maya-mm3) capture decode steps into HIP graphs"
selection = "by model architecture from the pack (GLM-5 / glm5-next / glm-dsa, DeepSeek-V4.1, MiniMax-M3.1, Qwen-Next), gcnArchName branches and MAYA_* / STRATA_* environment switches; TP2 by MAYA_TP_* settings and the rank launchers in tools/tp2"
models    = "GLM-5.3-Flash Maya-XS / S / M / L (IQ2_XXS to K-quant GGUF), DeepSeek-V4.1-Flash Maya-XS / S (TP2 with DSpark), MiniMax-M3.1-Flash, Qwen-Next family (README)"

[tp2]
transport = "RoCE v2 RC queue pairs over Thunderbolt usb4_rdma (src/net/verbs.cpp): one RDMA write with immediate data per exchange into rotating fixed-address receive windows mapped with hipHostRegister; TCP bootstrap for QP metadata, control frames and peer-loss detection; wire GUFT v7 / GUCH v5 / GURD byte-compatible with gufo"
queued    = "GPU-staged: StagePartial publishes a ready flag at system scope, the NIC completion raises the arrival flag, WaitAdd spins on it on the stream and adds the peer partial into the residual (src/net/tp_kernels.cu); the README calls the queued collectives 23-24% faster on DeepSeek XS than one host"
partition = "intra-expert: each rank holds half of every routed / shared expert's and dense FFN's intermediate dimension (2048 -> 1024, 12288 -> 6144), byte spans of the quant blob, down shares block-aligned; attention heads split; two all-reduce points per block (mixer out, MoE + shared sum); routing replicated"
modules   = "kernel.maya_tp.tp_kernels, runtime.maya_tp.{tp_rdma, tp_comm_abi, tp_context, glm_tp_batch, ds4_tp_control, ds4_tp_batch, ds4_tp_worker, ds4_tp_loader, mm3_tp}, tool.maya_tp.{tp2_launch, tp2_validation, tp_tests}"
'''

LEFT_OUT = {
    'docs/, README.md, CHANGELOG.md, AGENTS.md, CONTRIBUTING.md, SECURITY.md, Handoff*.md, initial-prompt.md': 'documentation, benchmark reports and the upstream integration notes (docs/upstream, 1,824 files); read for the module records',
    'data/': 'GLM synthetic dumps and fixtures (glm5-synth-dumps, glm-dsa, exl3 fixtures)',
    'tests/ (outside tool.maya_tp.tp_tests and hip_tests)': 'the engine\'s other unit and model tests (glm_dsa, ds4, mm3, hy4, platform)',
    'tools/ (outside tools/tp2, tools/llama-swap and tools/hip)': 'packers, quant tools, calibration, setup UI, benches and their tests',
    'serve/web, serve/test_*.py (except the TP2 tests), serve fakes and fixtures': 'the browser UI (HTML / JS / CSS) and server tests',
    'src/kernels/cpu tests and stress programs (except native_down_cols_test.cpp)': 'CPU lane tests',
    'src/program/conv_cache_test.cpp, src/spec/*_test.cpp, src/artifact/*_test.cpp, src/ngram/*_test.cpp, src/platform/memory_test.cpp, src/prefill/failure_test.cpp, src/core/*_test.cpp': 'unit tests of harvested host code',
    'src/ds4/ds4_vision.cu? no: harvested; src/ds4 headers not named in a module': 'none: every src/ds4 file is in a module',
    'CMakeLists.txt, cmake/, .github/, setup.py, setup.sh, maya.py, maya.sh, chat.py, START-MAYA.bat, VERSION': 'build system, installers and launch scripts (the HIP compile options are in ORIGIN.toml [compile])',
    'third_party/wheels, third_party/exllamav3, third_party/beellama.cpp, ref/, bench/, assets/': 'vendored wheels, licence-only directories, the reference upstream llama.cpp notes, benches and images',
    'LICENSE.deepseek-v41, LICENSE.minimax-m31, LICENSE.qwen-community-1.0': 'model licences',
    'LICENSE': 'copied beside ORIGIN.toml',
}
del LEFT_OUT['src/ds4/ds4_vision.cu? no: harvested; src/ds4 headers not named in a module']


def modules_md(owner):
    bt = '\x60'
    o = ['# Maya-TP modules', '',
         'Every module of catalog/maya_tp (uv-genai/maya-tp 6951bd7, tp-v1.1.45). Card ORIGIN-MAYA-TP 2982083a71. Written by',
         bt + 'docs/harvest/harvest_maya_tp.py' + bt + ' from its ' + bt + 'FAMILIES' + bt + ' table. **TP2** marks the modules of the two-host tensor-parallel',
         'delta over Project Maya; the rest is the engine Maya-TP carries (Project Maya v1.1.58 with the strata:: names).', '',
         '| module | TP2 | files | lines | interface | compile |', '|---|---|---:|---:|---|---|']
    tot_f = tot_l = 0
    for f, m in zip(FAMILIES, mods):
        ps = [p for p, mid in owner.items() if mid == m.id]
        lines = sum(len(open(O.find(p), 'rb').read().splitlines()) for p in ps)
        tot_f += len(ps); tot_l += lines
        o.append('| %s%s%s | %s | %d | %d | %s | %s |' % (bt, m.id, bt, 'TP2' if f['tp'] else '', len(ps), lines,
                 ', '.join(bt + p['interface'] + bt for p in m.provides) or 'none', m.compile))
    o += ['', '%d modules, %d files, %d lines; %d TP2 modules.' % (len(mods), tot_f, tot_l, sum(1 for f in FAMILIES if f['tp'])), '',
          '## Not done here', '',
          'No kernel or engine was run or timed on the pod, and no RDMA link was opened (the pod nodes have no Thunderbolt',
          'link between them). The compile gate (' + bt + 'gate/compile/run.py --origin maya_tp' + bt + ', control-node CPU, gfx1151) compiles the',
          'standalone modules; the host engine modules are harvested only (their CMake targets link dozens of units together).']
    open(os.path.join(harvestlib.REPO, 'catalog', O.name, 'MODULES.md'), 'w').write('\n'.join(o) + '\n')


if __name__ == '__main__':
    owner = run(O, mods, ORIGIN_TOML, LEFT_OUT, ['LICENSE'])
    modules_md(owner)
