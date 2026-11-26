#!/usr/bin/env python3
"""Harvest upstream TensorFold (ashhart/TensorFold main 000b534, 2026-11-12) into catalog/tensorfold.
Run: python3 docs/harvest/harvest_tensorfold.py

The checkout is ~/Code/strixflerp-upstream-supplement/TensorFold, a shallow read-only clone; nothing of it is built or
run here. Upstream is 11,340 files (185 MB, most of it docs, receipts, the app and tests), so the harvest takes:
the HIP backend whole (kernels, Zig runtime, collectives, serve glue, the families it registers), the CUDA kernels and
families for the models the pod cares about (DeepSeek-V4.1, GLM-5.3-Flash, Kimi K3, Nemotron, Qwen3.5 / Flash Next),
the host pieces StrixFold also holds (core, fabric, lanes, prompt cache, server, checkpoint CLI) and the build files of
the HIP half. Everything else is named in ORIGIN.toml [harvest] unharvested and summarised in HARVEST.json left_out.

The second job: the relation to catalog/strixfold. StrixFold is the pod's fork by way of bertholomus/TensorFold's
deepseek-v41-tp2 branch, cut from an older upstream. Where a file sits at the same path in both, this script records
per module the StrixFold module that holds it and the size of the difference (changed lines), not a duplicate verdict.
"""
import difflib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvestlib
from harvestlib import Origin, M, P, run, REPO, tlist

CHECKOUT = os.environ.get('TENSORFOLD_CHECKOUT', os.path.join(harvestlib.SUPPLEMENT, 'TensorFold'))
COMMIT = '000b534a284524bbe9a4cc1f3e8996baf80af754'


class GitOrigin(Origin):
    """The checkout is a git clone: .git is not an origin file."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.roots = [CHECKOUT]

    def files(self):
        if self._files is None:
            out = []
            for d, dirs, fs in os.walk(self.roots[0]):
                if d == self.roots[0]:
                    dirs[:] = [x for x in dirs if x not in ('.git', 'build')]
                out += [os.path.relpath(os.path.join(d, f), self.roots[0]) for f in fs]
            self._files = sorted(out)
        return self._files


O = GitOrigin('tensorfold', 'TensorFold', 'https://github.com/ashhart/TensorFold', COMMIT, 'main', 'Apache-2.0',
              ['TensorFold contributors'], 'therock-10.2')
HK = 'zig/kernels/hip/'
CK = 'zig/kernels/cuda/'
FA = 'zig/src/families/'
HIPFLAGS = ['--genco', '-D__HIP_PLATFORM_AMD__=1', '-DUSE_ROCM=1', '-DHIPBLAS_V2', '-fPIC', '-DCUDA_HAS_FP16=1',
            '-DHIP_ENABLE_WARP_SYNC_BUILTINS=1', '-std=c++20', '-fno-gpu-rdc', '-mno-wavefrontsize64', '-ffp-contract=off',
            '-DTF_WAVE=32', '-DTF_DOT2_F16=1', '-DTF_DOT2_BF16=1', '-DTF_SDOT4=1', '-DTF_SDOT8=1', '-DTF_MATRIX=1',
            '-DTF_DOT2_SIGNED_ZERO=1']
DEC = 'decode rounds (1..N lanes), memory-bound'
PF = 'prompt chunks, compute-bound'
NOT_GFX = 'CUDA source: harvested to read its recipe; not built for gfx1151'
ZIG = 'zig: harvested as data; not assembled (no tensorfold recipe yet)'
mods = []
claimed = set()


def take(patterns, exclude=()):
    out = [f for f in O.expand(patterns, exclude) if f not in claimed]
    claimed.update(out)
    if not out:
        raise SystemExit(f'nothing left for {patterns}')
    return out


def add(id, summary, patterns, provides, exclude=(), lang='zig', sources=(), compile='n/a', reason=None, includes=None,
        spdx=None, upstream=None, notes=None, requires=(), flags=()):
    files = take(patterns, exclude)
    if reason is None:
        reason = {'zig': ZIG, 'cuda': NOT_GFX, 'python': 'python: not compiled'}.get(lang)
    mods.append(M(id, summary, files, provides=provides, requires=requires, lang=lang, sources=list(sources),
                  includes=includes if includes is not None else (['src/' + HK] if lang == 'hip-c++' else []),
                  compile=compile, reason=reason, spdx=spdx, upstream=upstream, notes=notes, flags=flags,
                  toolchain='' if lang in ('python', 'cuda') else None))


def hipk(id, summary, patterns, provides, sources=(), **k):
    add(id, summary, [HK + p for p in patterns], provides, lang='hip-c++', sources=[HK + s for s in sources],
        compile='standalone' if sources else 'header-only',
        reason=None if sources else 'header-only; included by the group sources of the other kernel.tensorfold.hip_* modules',
        flags=HIPFLAGS if sources else (), **k)


# ------------------------------------------------------------------------------------------------- HIP kernels
# zig/build/hip.zig compiles one code object per "group" source with -I zig/kernels/hip; the headers are shared.
hipk('kernel.tensorfold.hip_common', 'HIP device basics for RDNA2/3/3.5/4: arch switches (TF_WAVE, dot2, sdot, matrix units), vector types, dot2 helpers, WMMA fragments, the columns-controls contract header',
     ['common/', 'columns_controls.hpp'],
     [P('kernel-abi.tensorfold-hip/-/-', 'common/arch.hpp, vec.hpp, dot2.hpp, wmma.hpp', 'n/a', 'n/a')])
hipk('kernel.tensorfold.hip_quant', 'Weight decoders for the HIP GEMV/GEMM tiles: MLX affine (4/6/8-bit, group scales and biases) pieces and tiles, BF16 decoder, dense tiles, activation quantisation',
     ['quant/'],
     [P('format.device-view/mlx-affine/-', 'quant/mlx.hpp, mlx_decoder.hpp, mlx_pieces.hpp, mlx_tiles.hpp', 'exact dequant (q*scale+bias) in fp32', 'n/a'),
      P('format.device-view/bf16/-.tensorfold', 'quant/bf16_decoder.hpp, dense_tiles.hpp', 'exact', 'n/a')])
hipk('kernel.tensorfold.hip_tiles', 'Matrix tiles: dot2 streaming GEMV (fp16/bf16 dot2, column-split, fused epilogues), WMMA gemm for gfx11 and gfx12 (wmma12), k-split gemm, stream and plan tables',
     ['tiles/'],
     [P('linear.decode/mlx-affine/f32.tensorfold-dot2', 'tiles/dot2.hip, dot2_tiles.hip (affine_dot2_stream_*)', 'fp32 accumulate in a fixed lane order', DEC),
      P('linear.prefill/mlx-affine/f32.tensorfold-wmma', 'tiles/gemm.hpp, matrix_gemm.hpp, wmma12_gemm.hpp, gemm_kp.hpp', 'WMMA fp32 accumulate', PF)],
     sources=['tiles/dot2.hip', 'tiles/dot2_tiles.hip'])
hipk('kernel.tensorfold.hip_ops', 'Model-free ops: RMS/layer norms, RoPE, elementwise, activations, MoE routing and expert GEMV (Qwen and Nemotron shapes), TP shard helpers, sampling draws, dense text rows, vision ops, VL positions',
     ['ops/'],
     [P('norm.rms/-/f32.tensorfold', 'ops/norms.hpp', 'fp32', 'all'),
      P('rope.apply/-/f32.tensorfold', 'ops/rope.hpp', 'fp32', 'all'),
      P('moe.experts-decode/mlx-affine/f32.tensorfold', 'ops/moe.hpp, ops/nemotron_moe.hpp', 'per-expert fp32, fixed combine order', DEC),
      P('sample.draw/-/f32.tensorfold', 'ops/draw.hpp', 'keyed draws on the GPU', DEC)],
     sources=['ops/ops.hip', 'ops/act.hip', 'ops/vision.hip'])
hipk('kernel.tensorfold.hip_attention', 'Attention: paged GQA decode with split-K, flash-style prefill (attention_fa), prefill tiles',
     ['attention/'],
     [P('attention.full/kv-f16/f32.tensorfold', 'attention/attention.hip, attention_fa.hip, prefill.hip', 'fp32 softmax', 'decode and prefill')],
     sources=['attention/attention.hip', 'attention/prefill.hip'])
hipk('kernel.tensorfold.hip_decode', 'Decode round planning on the GPU: page tables, the per-round launch plan',
     ['decode/'],
     [P('kv.pages/-/-.tensorfold', 'decode/pages.hpp, plan.hpp', 'n/a', DEC)],
     sources=['decode/decode.hip', 'decode/plan.hip'])
hipk('kernel.tensorfold.hip_recurrence', 'Linear-time mixers: Gated DeltaNet (Qwen3.5 / Flash Next) chunked prefill and recurrent decode; Mamba-2 selective scan and conv (Nemotron hybrid)',
     ['recurrence/'],
     [P('gdn.recurrence/-/f32.tensorfold', 'recurrence/gated_delta.hip', 'fp32 state', 'decode and prefill'),
      P('mixer.mamba2/-/f32.tensorfold', 'recurrence/mamba.hip', 'fp32 state', 'decode and prefill')],
     sources=['recurrence/gated_delta.hip', 'recurrence/mamba.hip'])
hipk('kernel.tensorfold.hip_flashnext', 'Qwen Flash Next on HIP: QSA indexer, sparse attention and its lane split, n-gram (PLE) rows, stream kernels',
     ['flashnext/'],
     [P('indexer.qsa/-/f32.tensorfold', 'flashnext/indexer.hpp', 'fp32 scores, top-k', 'decode and prefill'),
      P('attention.sparse-sink/kv-f16/f32.tensorfold', 'flashnext/sparse.hpp, sparse_lanes.hpp', 'fp32 softmax', 'decode and prefill')],
     sources=['flashnext/flashnext.hip'])
hipk('kernel.tensorfold.hip_draft', 'MTP / drafter kernels: draft rows, acceptance on the GPU',
     ['draft/'], [P('speculation.draft/mtp/f32.tensorfold', 'draft/draft.hip', 'drafts only', DEC)], sources=['draft/draft.hip'])
hipk('kernel.tensorfold.hip_vision_ocr', 'Vision and OCR towers: GLM-OCR pointwise ops (PyTorch reduction order), LightOnOCR and Qwen vision (strict IEEE builds)',
     ['glm_ocr/', 'lighton_ocr.hip', 'qwen_vision.hip'],
     [P('vision.tower-ops/bf16/f32.tensorfold', 'glm_ocr/pointwise.hip, lighton_ocr.hip, qwen_vision.hip', 'fp32; strict IEEE for the last two', 'prompt')],
     sources=['glm_ocr/pointwise.hip', 'lighton_ocr.hip', 'qwen_vision.hip'], spdx='Apache-2.0 AND BSD-3-Clause',
     upstream='glm_ocr/pointwise.hip adapts PyTorch 8514f05 Reduce.cuh and layer_norm_kernel.cu (BSD-3-Clause; THIRD_PARTY_NOTICES.md)')
hipk('kernel.tensorfold.hip_joint_schema', 'Structured-output token masks on the GPU (joint schema)',
     ['joint_schema/'], [P('grammar.mask/-/-.tensorfold', 'joint_schema/joint_schema.hip', 'exact', DEC)], sources=['joint_schema/joint_schema.hip'])
hipk('kernel.tensorfold.hip_affine', 'Stand-alone affine projection object (affine.hip, built separately by zig/build/hip.zig for the affine tuner)',
     ['affine.hip'], [P('linear.decode/mlx-affine/f32.tensorfold-tune', 'affine.hip', 'fp32', DEC)], sources=['affine.hip'])

# ------------------------------------------------------------------------------------------------- CUDA kernels (recipes)
add('kernel.tensorfold.cuda_common', 'Shared CUDA headers: cp.async, MMA fragments for quantised matmul, KV pages, MLA and KDA helpers, fp pairs, fold',
    [CK + 'common/'], [P('kernel-abi.tensorfold-cuda/-/-', 'common/*.cuh', 'n/a', 'n/a')], lang='cuda')
add('kernel.tensorfold.cuda_dsv41', 'DeepSeek-V4.1 CUDA kernels: sparse attention and split-K, hyper-connection mix (mHC) with prefetch, router GEMV, top-k, KV glue and split, Engram gate, prefill glue, L2 pacing/prefetch, sampling',
    [CK + 'deepseek_v41/'],
    [P('attention.sparse-sink/kv-fp8/f32.tensorfold-cuda', 'deepseek_v41/attn_cuda.cu, attn_split.cuh, kvsplit.cu', 'fp32 softmax', 'decode and prefill'),
     P('hc.mix/bf16/f32.tensorfold-cuda', 'deepseek_v41/mhc_cuda.cu, mhc_pf.cu, mhc_tail.cuh', 'fp32', 'decode and prefill'),
     P('moe.route/-/f32.tensorfold-cuda', 'deepseek_v41/router_gemv.cu, router_glue.cu, router_narrow.cuh', 'fp32', 'all')],
    lang='cuda', upstream='DeepSeek-V4.1 CUDA sources (THIRD_PARTY_NOTICES.md "DeepSeek-V4.1 CUDA sources")')
add('kernel.tensorfold.cuda_exl3', 'EXL3 on CUDA: tile decode straight into MMA fragments, linear and grouped-expert GEMV, prefill dense (pfdense k8..k16), x3gm grouped GEMM plan/load/epilogue, segmented experts; EXLR8 codecs',
    [CK + 'exl3_*', CK + 'exl3_block/', CK + 'exlr8*.cu'],
    [P('linear.decode/exl3/f32.tensorfold-cuda', 'exl3_linear.cu, exl3_block/linear.cu, lin2.cu', 'fp32 accumulate', DEC),
     P('moe.experts-prefill/exl3/f32.tensorfold-cuda', 'exl3_experts.cu, exl3_block/x3gm*.cu, x3seg.cu, experts_grouped.cuh', 'rows independent, K ranges fixed by shape, warps summed in order', PF)],
    lang='cuda', spdx='Apache-2.0 AND MIT', upstream='EXL3 after ExLlamaV3 (MIT, Copyright (c) 2025 Turboderp), per-file headers')
add('kernel.tensorfold.cuda_glm', 'GLM-5.3-Flash CUDA kernels: KDA (Kimi delta attention) mixer, DSA indexer pools and scores, glue (attention, latent, router MMA, MoE, hyper-connections, norms, rows), DFlash drafter, quant, vision',
    [CK + 'glm_*'],
    [P('mixer.kda/-/f32.tensorfold-cuda', 'glm_kda.cu', 'fp32 state', 'decode and prefill'),
     P('indexer.score/-/f32.tensorfold-glm', 'glm_index.cu, glm_index_pools.cuh, glm_index_scores.cuh', 'fp32', 'decode and prefill')],
    lang='cuda')
add('kernel.tensorfold.cuda_nemotron', 'Nemotron 3.5 hybrid CUDA kernels: Mamba mixer, latent MoE, routing, attention, keyed sampling ops',
    [CK + 'nemotron_*'],
    [P('mixer.mamba2/-/f32.tensorfold-cuda', 'nemotron_mamba.cu', 'fp32 state', 'decode and prefill'),
     P('moe.experts-decode/nvfp4/f32.tensorfold-nemotron', 'nemotron_latent.cu, nemotron_route.cu', 'fp32', DEC)],
    lang='cuda')
add('kernel.tensorfold.cuda_flashnext', 'Qwen Flash Next CUDA kernels: GDN (fused, lanes, TPR), QSA tile, sparse attention lanes, hyper-connections dense, pages, MTP, batching',
    [CK + 'flashnext_*'],
    [P('gdn.recurrence/-/f32.tensorfold-cuda', 'flashnext_gdn.cu, flashnext_gdn_fused.cuh, flashnext_gdn_lanes.cuh, flashnext_gdn_tpr.cuh', 'fp32 state', 'decode and prefill')],
    lang='cuda')
add('kernel.tensorfold.cuda_experts', 'CUDA MoE expert kernels: affine/NVFP4/FP8 experts, packed and prefill experts, expert rows, CK lanes, generic MoE glue and the lane GEMV',
    [CK + 'experts*', CK + 'nvfp4_experts*', CK + 'fp8_experts.cu', CK + 'ck_experts.cu', CK + 'ck_lane.cu', CK + 'moe.cu', CK + 'moe_glue.cu', CK + 'lane_gemv.cu'],
    [P('moe.experts-prefill/affine/f32.tensorfold-cuda', 'experts_prefill.cu, experts3_prefill.cu', 'fp32', PF)], lang='cuda')
add('kernel.tensorfold.cuda_roce_gather', 'Cross-rank gather over RoCE with GPU-side completion (roce_gather.cu) and the rank sum used by TP decode',
    [CK + 'roce_gather.cu', CK + 'rank_sum.cu'],
    [P('collective.allgather-roce/-/f32.tensorfold-cuda', 'roce_gather.cu, rank_sum.cu', 'bytes; sum in rank order', 'TP decode')], lang='cuda')

# ------------------------------------------------------------------------------------------------- runtime (zig)
add('runtime.tensorfold.hip', 'The HIP backend runtime: raw HIP driver binding, device caps and arch detect, memory and arena, modules and code objects, typed launches per kernel group, graphs (capture, split, cache), streams, hipBLAS, optional AOTriton attention, admission and policy, per-arch tuning tables',
    ['zig/src/hip/'], exclude=['zig/src/hip/comm/**', 'zig/src/hip/kernel_tests/**', 'zig/src/hip/kernel_tests.zig'],
    provides=[P('device.facade/hip/-.tensorfold', 'hip/root.zig, runtime.zig, launch.zig, graph.zig', 'n/a', 'n/a'),
              P('tune.table/hip/-', 'hip/tuning/gfx1151.zon (+ gfx1030, gfx1100, gfx1201), tuning/search.zig', 'n/a', 'n/a')],
    spdx='Apache-2.0 AND MIT', upstream='hip/attention_aotriton.zig follows the AOTriton 0.11.1 v1 forward ABI (MIT; LICENSES/AOTriton-MIT.txt)')
add('runtime.tensorfold.hip_comm', 'HIP tensor-parallel group: RCCL by dlopen, a TCP link transport, loopback, rank identity handshake',
    ['zig/src/hip/comm/'],
    [P('collective.interface/-/-.tensorfold-hip', 'hip/comm/group.zig, rccl.zig, link.zig, loopback.zig, identity.zig', 'bytes', 'TP on one host or over links')])
add('runtime.tensorfold.hip_serve', 'HIP serving glue between a family and the server: offers, options, pages, prefix reuse, snapshots, draws, cancel, costs',
    ['zig/src/hip_serve/'], [P('serve.engine/-/-.tensorfold-hip', 'hip_serve/serve.zig, backend.zig', 'n/a', 'serving')])
add('runtime.tensorfold.fabric', 'Host fabric library: verbs ABI and queue pairs, memory registration, RDMA links, mailbox / packet / wire framing',
    ['zig/src/fabric/'], [P('transport.verbs/-/-.tensorfold', 'fabric/verbs.zig, verbs_qp.zig, verbs_link.zig, registration.zig', 'n/a', 'RC queue pairs over RoCE')])
add('runtime.tensorfold.lanes', 'Lane core: the round loop over concurrent streams, the backend vtable, draft depth and acceptance, the engine API, stagger',
    ['zig/src/core/lanes/', 'zig/src/core/lane_host.zig', 'zig/src/core/draft_depth.zig', 'zig/src/core/stagger.zig', 'zig/src/core/engine_api.zig',
     'zig/src/core/lane_host_test.zig', 'zig/src/core/lane_host_reuse_test.zig'],
    [P('executor.lane-backend/-/-.tensorfold', 'core/lanes/backend.zig', 'n/a', 'N streams a round'),
     P('sample.keyed/-/f64.tensorfold', 'core/lanes/sampling.zig', 'keyed exact sampler', 'per token')])
add('runtime.tensorfold.prompt_cache', 'Exact prompt reuse: states at prompt-pass chunk ends, on-disk Imprint',
    ['zig/src/core/prompt_cache.zig', 'zig/src/core/prompt_imprint.zig', 'zig/src/core/segments.zig', 'zig/src/core/keepalive.zig',
     'zig/src/core/prompt_cache_test.zig', 'zig/src/core/prompt_imprint_test.zig'],
    [P('cache.prompt-state/-/-.tensorfold', 'core/prompt_cache.zig, prompt_imprint.zig', 'exact reuse', 'chunk-end granularity')])
add('runtime.tensorfold.core', 'Host core: tokenizer, chat template engine, small helpers the families import (same file set as runtime.strixfold.core)',
    ['zig/src/core/tokenizer/', 'zig/src/core/template/', 'zig/src/core/root.zig', 'zig/src/core/affine_mm.zig', 'zig/src/core/copy_index.zig',
     'zig/src/core/fence.zig', 'zig/src/core/frags.zig', 'zig/src/core/hc.zig', 'zig/src/core/ids_json.zig', 'zig/src/core/moe_route.zig',
     'zig/src/core/row_projection.zig'],
    [P('text.tokenizer/-/-.tensorfold', 'core/tokenizer/tokenizer.zig', 'n/a', 'host'), P('text.chat-template/jinja/-.tensorfold', 'core/template/template.zig', 'n/a', 'host')])
add('runtime.tensorfold.engine_host', 'The HIP engine host: the family registry the server opens (qwen3_5, qwen3_5 MoE, flashnext, nemotron, glm_ocr, qwen3, dots.mocr, Nemotron Parse, LightOnOCR), refusals, memory budget and cache fit',
    ['zig/src/native/hip.zig', 'zig/src/native/hip_families.zig', 'zig/src/native/none.zig', 'zig/src/native/cache_fit.zig', 'zig/src/native/cuda_memory.zig',
     'zig/src/native/draft_sources_hip_test.zig'],
    [P('executor.host/hip/-.tensorfold', 'native/hip.zig registry, native/hip_families.zig', 'n/a', 'one engine a process')])

# ------------------------------------------------------------------------------------------------- formats
add('format.tensorfold.safetensors', 'Host readers: safetensors headers and tensors, checkpoint index, O_DIRECT reads, npy',
    ['zig/src/core/safetensors.zig', 'zig/src/core/direct_io.zig', 'zig/src/core/checkpoint.zig', 'zig/src/core/npy.zig'],
    [P('format.reader/safetensors/-.tensorfold', 'core/safetensors.zig, checkpoint.zig', 'exact', 'load time')])
add('format.tensorfold.quant', 'Checkpoint quant formats on the host: MLX affine, BF16, FP8, dense and expert packing, conversion, cache keys',
    ['zig/src/core/quant/'], [P('format.loader/mlx-affine/-', 'core/quant/mlx_affine.zig, convert.zig, experts.zig', 'exact', 'load time')])
add('format.tensorfold.exl3', 'EXL3 and EXLR8 on the host: format and layout, load and prepare, projection decode, EXLR8 codec, manifest, node split, experts, prompt path',
    ['zig/src/core/exl3_*.zig', 'zig/src/core/exlr8*.zig'],
    [P('format.loader/exl3-safetensors/-.tensorfold', 'core/exl3_format.zig, exl3_layout.zig, exl3_load.zig', 'exact', 'load time'),
     P('format.loader/exlr8/-', 'core/exlr8*.zig', 'exact', 'load time')],
    spdx='Apache-2.0 AND MIT', upstream='EXL3 after ExLlamaV3 (MIT, Turboderp)')

# ------------------------------------------------------------------------------------------------- executors (families)
FAMS = [
    ('qwen3_5', 'Qwen3.5 / 3.6 / 3.8 dense and MoE with Gated DeltaNet (HIP, CUDA, Metal; HIP qualified on gfx1151 at v1.0.12 with MLX 4-bit)', 'qwen3.5', 'mlx-affine', ()),
    ('flashnext', 'Qwen3.8 Flash Next: GDN + QSA sparse attention + PLE n-grams + hyper-connections + MTP (HIP path in flashnext/hip, CUDA, Metal)', 'qwen3.8-flash-next', 'mlx-affine', ()),
    ('nemotron', 'Nemotron 3.5 hybrid Mamba-2 / attention / latent MoE with MTP (HIP path in nemotron/hip, CUDA, Metal)', 'nemotron-3.5', 'mlx-affine', ()),
    ('qwen3', 'Qwen3 dense (embedding / reranker readouts on HIP, CUDA)', 'qwen3', 'bf16', ()),
    ('glm_ocr', 'GLM-OCR on HIP', 'glm-ocr', 'bf16', ()),
    ('dots_mocr', 'dots.mocr OCR on HIP (gfx1201 checked)', 'dots.mocr', 'bf16', ()),
    ('nemotron_parse', 'Nemotron Parse on HIP', 'nemotron-parse', 'bf16', ()),
    ('lighton_ocr', 'LightOnOCR on HIP', 'lighton-ocr', 'bf16', ()),
    ('deepseek_v41', 'DeepSeek-V4.1 upstream family: a separate rewrite from StrixFold dsv41 (CUDA and Metal; no HIP path upstream; no release binary serves it until its whole-model gate passes)', 'deepseek-v4.1', 'exl3', ('fixtures/**',)),
    ('glm', 'GLM-5.3-Flash: KDA + DSA sparse attention + MoE, MTP / DFlash (CUDA TP2/TP3, paired Metal; no HIP path)', 'glm-5.3-flash', 'nvfp4', ()),
    ('glm_dsa', 'GLM DSA (sparse attention) host pieces', 'glm-dsa', '-', ()),
    ('kimi_k3', 'Kimi K3 / K3.1 text adapter (Metal; no HIP path)', 'kimi-k3', 'mlx-affine', ()),
]
for fam, summ, model, fmt, ex in FAMS:
    add(f'executor.tensorfold.{fam}', summ, [FA + fam + '/'], exclude=[FA + fam + '/' + e for e in ex],
        provides=[P(f'executor.model/{model}/{fmt}.tensorfold', f'families/{fam}', 'per family receipts (docs/receipts upstream)', 'serving')],
        spdx='Apache-2.0 AND LicenseRef-Kimi-K3' if fam == 'kimi_k3' else None,
        notes=('fixtures/ (3.4 MB of recorded launches and checkpoint forms) left out' if fam == 'deepseek_v41' else
               'fixtures/ are Moonshot AI release files under the Kimi K3 / K3.1-Pro licences (catalog/tensorfold/LICENSES)' if fam == 'kimi_k3' else None))

# ------------------------------------------------------------------------------------------------- serve, tools
add('serve.tensorfold.http', 'The native HTTP server: OpenAI chat / responses and Anthropic messages, SSE, tool calls, auth, metrics',
    ['zig/src/server/'], [P('serve.http/openai+anthropic/-.tensorfold', 'server/main.zig, serve.zig', 'n/a', 'host')],
    spdx='Apache-2.0 AND LicenseRef-Kimi-K3', notes='server/kimi_prompt.zig, kimi_values.zig and fixtures/kimi adapt Kimi K3 release code (THIRD_PARTY_NOTICES.md)')
add('tool.tensorfold.checkpoint_cli', 'Checkpoint command line (models / info / pull)',
    ['zig/src/cli/cli.zig', 'zig/src/cli/hub.zig', 'zig/src/cli/info.zig', 'zig/src/cli/models.zig', 'zig/src/cli/pull.zig'],
    [P('tool.checkpoint-cli/-/-.tensorfold', 'cli/cli.zig', 'n/a', 'host')])
add('tool.tensorfold.zig_build', 'The build graph of the HIP half: root build.zig, zig/build/hip.zig (16 kernel groups, flags, caps switches), hip_contract.zig, lighton_hip.zig, the pinned Zig version',
    ['build.zig', 'build.zig.zon', '.zig-version', 'zig/build/hip.zig', 'zig/build/hip_contract.zig', 'zig/build/lighton_hip.zig'],
    [P('build.graph/zig/-.tensorfold', 'zig build (hip half: zig/build/hip.zig)', 'n/a', 'build time')],
    notes='zig/build is ignored by this repository\'s .gitignore (build/); the files are tracked with git add -f, as StrixFold\'s are.')

# The root texts, LICENSES/ and the fixture licence THIRD_PARTY_NOTICES.md links, copied beside ORIGIN.toml.
LICENCE_FILES = (['LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md'] + [f for f in O.files() if f.startswith('LICENSES/')]
                 + ['zig/src/server/fixtures/deepseek_v41_encoder/LICENSE'])

ORIGIN_TOML = f'''
# Upstream TensorFold: Zig native inference engine with Metal, CUDA, HIP, XPU and Vulkan backends.
# StrixFold (catalog/strixfold) is the pod's fork of an older TensorFold by way of bertholomus/TensorFold deepseek-v41-tp2.
name      = "tensorfold"
repo      = "https://github.com/ashhart/TensorFold"
commit    = "{COMMIT}"
branch    = "main"
date      = "2026-11-12"
reference_checkout = "~/Code/strixflerp-upstream-supplement/TensorFold (shallow, read-only; ark manifest repos.tsv also tracks it)"
relation  = "HARVEST.json relation_to_strixfold: per module, the StrixFold module holding the same path and the changed-line count"

[licence]
spdx      = "Apache-2.0"
# copied verbatim next to this file at their origin paths (LICENCE_FILES in docs/harvest/harvest_tensorfold.py)
files     = {tlist(LICENCE_FILES)}
source    = "the reference checkout above (git clone at {COMMIT[:9]}); byte-identical to the pod ark copies control-node:/data/ark-inbox/src/github.com__ashhart__TensorFold and /data/ark/software/src/github.com__ashhart__TensorFold (checked 2026-11-26). Those ark copies are at 3795e605, which adds LICENSES/Tencent-Hy-1.0.txt (not in {COMMIT[:9]}): run flerp diff-origin against the {COMMIT[:9]} clone, not the ark copies"
third_party = "Code before 0.6.0 keeps its MIT notice (LICENSES/MIT.txt). EXL3 sources after ExLlamaV3 (MIT, Turboderp); GLM-OCR pointwise after PyTorch (BSD-3-Clause); AOTriton ABI adapter (MIT); Kimi K3 fixtures and prompt adapter under the Kimi K3 / K3.1-Pro licences. See THIRD_PARTY_NOTICES.md."

[toolchain]
host      = "zig 0.17.0 (.zig-version)"
device    = "therock-10.2"                       # what the pod would build with; upstream documents ROCm 7.13 (docs/known-issues.md): ROCm 7.13 or Ubuntu HIP 7.1 / clang 21; the gfx1151 receipt (v1.0.12) used Ubuntu HIP 7.1.52801
targets   = "gfx1030, gfx1100, gfx1151, gfx1201 (zig/src/hip/tuning/*.zon)"

[compile]                                        # zig/build/hip.zig compileSource
driver    = "<rocm>/bin/hipcc"
flags     = {json.dumps(HIPFLAGS)}
arch_flag = "--offload-arch=gfx1151"
rocm_flag = "--rocm-path=<rocm> --rocm-device-lib-path=<rocm>/lib/llvm/amdgcn/bitcode"
include   = "-I zig/kernels/hip"
strict    = ["-O2", "-fno-fast-math", "-fdenormal-fp-math=ieee"]   # lighton_ocr.hip, qwen_vision.hip
output    = "one code object per group source (16 groups, zig/build/hip.zig `groups`), embedded in the binary"
caps_note = "TF_* switches come from zig/src/hip/caps.zig for the arch; the values in flags are gfx1151's row of that table (matrix wmma11, dot2_signed_zero true); not yet checked by a compile gate"

[kernels]
loading   = "code objects embedded by @embedFile; zig/src/hip/kernels.zig names the groups, launches/*.zig bind kernels by symbol with typed arguments"
'''

LEFT_OUT = {
    'docs/, *.md at the root, assets/, packaging/, Dockerfile, setup.py, .github/, web/, app/': 'documentation, receipts, packaging, the web and desktop apps (docs read for docs/TENSORFOLD.md)',
    'zig/tests/, zig/test_support/, zig/*.zig at the zig/ root, zig/src/hip/kernel_tests*, tools/': 'tests, gate programs, benches and Python tooling',
    'zig/kernels/metal/, zig/src/metal/, zig/kernels/vulkan/, zig/src/vulkan/, zig/kernels/xpu/, zig/src/xpu/': 'Metal, Vulkan and Intel XPU backends: no pod hardware',
    'zig/kernels/cuda/* (except deepseek_v41, exl3*, glm_*, nemotron_*, flashnext_*, experts / moe, roce_gather, common), zig/src/cuda/, zig/src/native/cuda*': 'CUDA backend runtime and kernels of families with no pod use',
    'zig/src/families/* except qwen3_5, flashnext, nemotron, qwen3, glm_ocr, dots_mocr, nemotron_parse, lighton_ocr, deepseek_v41, glm, glm_dsa, kimi_k3': 'families with neither a HIP path nor a pod model (gemma4, lfm*, minimax, step5, drama3, moss_image, ...)',
    'zig/src/families/deepseek_v41/fixtures/': 'recorded fixtures (3.4 MB)',
    'zig/src/core/* except the files of runtime.tensorfold.{core,lanes,prompt_cache}, format.tensorfold.{safetensors,quant,exl3}': 'the rest of the backend-neutral core (430 entries: contracts, living weights, digest cache, references)',
    'zig/src/{cluster,living,sessions,vision,video,platform,grammar}/, zig/src/native/* except hip*, cache_fit, cuda_memory, none': 'cluster manager, living weights, sessions, other engine hosts',
    'zig/kernels/hip/{runtime_tests,contract_controls}.hip': 'runtime and contract test kernels',
    'zig/build/* except hip.zig, hip_contract.zig, lighton_hip.zig; zig/runtime_sources, zig/tools': 'build halves of other backends',
}


def strixfold_relation(owner):
    """{tensorfold module: {strixfold module: {same_path, identical, changed_lines}}} for files at the same path."""
    sf = json.load(open(os.path.join(REPO, 'catalog', 'strixfold', 'HARVEST.json')))['owner']
    out = {}
    for f, m in owner.items():
        sm = sf.get(f)
        if sm is None:
            continue
        k, _, n = sm.split('.', 2)
        a = open(os.path.join(REPO, 'catalog', 'strixfold', k, n, 'src', f), 'rb').read()
        b = open(O.find(f), 'rb').read()
        r = out.setdefault(m, {}).setdefault(sm, {'same_path': 0, 'identical': 0, 'changed_lines': 0})
        r['same_path'] += 1
        if a == b:
            r['identical'] += 1
        else:
            r['changed_lines'] += sum(1 for l in difflib.unified_diff(a.decode(errors='replace').splitlines(),
                                                                     b.decode(errors='replace').splitlines(), n=0)
                                      if l[:1] in '+-' and l[:3] not in ('+++', '---'))
    return out


if __name__ == '__main__':
    owner = run(O, mods, ORIGIN_TOML, LEFT_OUT, LICENCE_FILES)
    rel = strixfold_relation(owner)
    p = os.path.join(REPO, 'catalog', 'tensorfold', 'HARVEST.json')
    idx = json.load(open(p))
    idx['relation_to_strixfold'] = rel
    idx['relation_note'] = ('same origin-relative path in catalog/strixfold (StrixFold f64b5ff, cut from an older TensorFold); '
                            'changed_lines = +/- lines of a zero-context unified diff summed over the module. StrixFold\'s dsv41 '
                            'family (zig/src/families/dsv41) and HIP kit (zig/kernels/hip/dsv41) have no counterpart path upstream: '
                            'upstream\'s DeepSeek family is zig/src/families/deepseek_v41, a separate rewrite (4 shared basenames, all changed).')
    json.dump(idx, open(p, 'w'), indent=1, sort_keys=True)
    for m, r in sorted(rel.items()):
        print(m, r)
