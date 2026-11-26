#!/usr/bin/env python3
"""Harvest DwarfStar (antirez/ds4) and five GLM / ROCm branches of it into catalog/ds4.
Run on control-node, where the ark copies are:
    python3 docs/harvest/harvest_ds4.py                 # rebuild catalog/ds4 and docs/harvest/data/ds4/*.tsv
    python3 docs/harvest/harvest_ds4.py --compose DIR   # write the reference tree flerp diff-origin --origin ds4 reads
    ./flerp diff-origin --origin ds4 DIR

Card ORIGIN-DS4 31224b4ca3. One origin, six trees:
  base     antirez/ds4 main b3f50180ec2c, at its own paths (catalog/ds4/<kind>/<name>/src/<path>);
  overlays davidcanar/ds4 glm-tp-rocm 8ebf67295a4f (PR #1024), kyuz0/ds4 perf/rocm-gfx1151-glm53-flash 4bdbcc206828,
           perf/rocm-gfx1151-glm53-flash-prefill-restoration 4fef372a4631, perf/rocm-gfx1151-glm53-performance
           f46631e483e1 and rocm-multi-node 89c1ea982df2. Each overlay is recorded as its base-diff against the base
           (docs/harvest/data/ds4/<slug>.tsv: every file it adds or changes, and every base file it lacks); the files it
           adds or changes in the GPU, tensor-parallel, transport and engine families are harvested as their own module
           set under the origin-relative prefix overlay/<slug>/, one module per family, named <prefix>_<family>.
The origin's file set is the base tree plus overlay/<slug>/<path> for every added or changed file of each overlay; the
composed reference tree (--compose) is exactly that, as symlinks into the ark copies, so diff-origin compares every
vendored byte with the ark. Each ark copy is a shallow git clone: the script checks HEAD against the pin and that the
work tree is clean before reading it.

Families (FAM below): every ROCm kernel header under rocm/ is its own module (they are fragments of the single ROCm
translation unit ds4_rocm.cu, compiled through kernel.ds4.rocm_backend), the vendored llama.cpp MMQ kernels split by
translation unit, the CUDA backend, the shared model kernels (DeepSeek V4.1, the two vision towers, Qwen3.8 Flash Next),
and one module per host C subsystem: engine, GPU API, TP transport, distributed runtime, SSD streaming, KV store,
speculative drafter, server, agent, CLI and the GGUF tools. Metal (metal/, ds4_metal.m), tests, benches, quality and
imatrix data and docs are listed as unharvested. Nothing here compiles or runs; gate.status = "builds" comes from
gate/results/compile (gate/compile/run.py --origin ds4 on control-node, CPU only)."""
import difflib, hashlib, os, re, subprocess, sys, tomllib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvestlib
from harvestlib import Origin, M, P, run

ARK = os.environ.get('DS4_ARK', '/data/ark-inbox/src')
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'ds4')
BASE = dict(dir='github.com__antirez__ds4', repo='https://github.com/antirez/ds4', branch='main',
            commit='b3f50180ec2c7324892be1f50bc83555ccb8d5e7', date='2026-11-03')
SKIP = ('.git', 'build')

# slug, module prefix, ark directory, repo, branch, commit, title, why
OVERLAYS = [
 dict(slug='glm_tp_rocm', prefix='glmtp', dir='github.com__davidcanar__ds4#8ebf67295a4f4663c326aa415145a35175e02b84',
      repo='https://github.com/davidcanar/ds4', branch='glm-tp-rocm', commit='8ebf67295a4f4663c326aa415145a35175e02b84', date='2026-10-23',
      authors=['davidcanar', 'The ds4.c authors', 'The ggml authors', 'DeepSeek'],
      title='davidcanar/ds4 glm-tp-rocm 8ebf672 (antirez/ds4 PR #1024, stacked on #813, #835 and #1018)',
      why='GLM-5.3-Flash tensor parallel across two ROCm machines: the KDA heads split between the ranks, the routed experts split 50/50, and the TP slab in host-mapped pinned memory that GPU kernels and the CPU transport thread share on Strix Halo (rocm/ds4_rocm_tp.cuh); the branch head is "GLM TP: keep the transport slab unified on ROCm builds"'),
 dict(slug='kyuz0_glm53_flash', prefix='k53flash', dir='github.com__kyuz0__ds4#perf__rocm-gfx1151-glm53-flash',
      repo='https://github.com/kyuz0/ds4', branch='perf/rocm-gfx1151-glm53-flash', commit='4bdbcc206828ab4e7cdfc45a1052cb5f35400fdc', date='2026-09-01',
      authors=['kyuz0', 'The ds4.c authors', 'The ggml authors'],
      title='kyuz0/ds4 perf/rocm-gfx1151-glm53-flash 4bdbcc2',
      why='kyuz0\'s gfx1151 GLM-5.3-Flash decode branch: on-the-fly WMMA Q8_0 batch GEMMs and partial HC expand (ds4_rocm_q8.cuh), the GLM dense-compact causal attention and row-wise Q8 projections (ds4_rocm_glm.cuh) and a stand-alone selected multi-head attention kernel (rocm/ds4_rocm_glm_selected_multihead.cu); the tree is a fork from before the base\'s V4.1 and Qwen work, so its base-diff also shows files the base changed later'),
 dict(slug='kyuz0_glm53_prefill', prefix='k53prefill', dir='github.com__kyuz0__ds4#perf__rocm-gfx1151-glm53-flash-prefill-restoration',
      repo='https://github.com/kyuz0/ds4', branch='perf/rocm-gfx1151-glm53-flash-prefill-restoration', commit='4fef372a46315bd6b34b0e58b7fc91a7e5f4f21f', date='2026-09-01',
      authors=['kyuz0', 'The ds4.c authors', 'The ggml authors'],
      title='kyuz0/ds4 perf/rocm-gfx1151-glm53-flash-prefill-restoration 4fef372',
      why='the glm53-flash branch with the GLM-5.3-Flash prefill path restored and accelerated (head commit "rocm: accelerate GLM 5.3 Flash prefill"); same fork point as kyuz0_glm53_flash'),
 dict(slug='kyuz0_glm53_perf', prefix='k53perf', dir='github.com__kyuz0__ds4#perf__rocm-gfx1151-glm53-performance',
      repo='https://github.com/kyuz0/ds4', branch='perf/rocm-gfx1151-glm53-performance', commit='f46631e483e19d13f23d3c4bced7ade0d086a3ad', date='2026-09-25',
      authors=['kyuz0', 'The ds4.c authors', 'The ggml authors'],
      title='kyuz0/ds4 perf/rocm-gfx1151-glm53-performance f46631e',
      why='kyuz0\'s later gfx1151 branch: the Halo kernels imported from MadRocK-AI/ds4 551219f (frozen WMMA GEMMs for the attention output projections and shared expert, IQ2 S9 MoE producers, Q2 and Q8 panels, direct-query attention), DeepSeek V4.1 ROCm primitives, Qwen3.8 Flash Next on gfx1151, a gufo-derived DSV4 attention, and two-machine ROCm tensor parallel over TCP or RoCE with its own slot-generation queue (docs/CLUSTERING_ROCM.md); head commit "rocm: avoid low-confidence GLM MTP verification on gfx1151"'),
 dict(slug='kyuz0_rocm_multi_node', prefix='kmultinode', dir='github.com__kyuz0__ds4#rocm-multi-node',
      repo='https://github.com/kyuz0/ds4', branch='rocm-multi-node', commit='89c1ea982df20b7578f95bb10d894f5447146fc3', date='2026-06-16',
      authors=['kyuz0', 'The ds4.c authors', 'The ggml authors'],
      title='kyuz0/ds4 rocm-multi-node 89c1ea9 (head commit by antirez, "Fix ROCm MTP startup residency")',
      why='an early ROCm multi-node line of ds4, five months older than the base: the ROCm backend before it grew V4.1, GLM-5.3 and Qwen kernels, with the pipeline-parallel distributed runtime; kept as the record of where the ROCm multi-node work started'),
]
OVL = {o['slug']: o for o in OVERLAYS}


def git(d, *args):
    return subprocess.run(['git', '-C', d] + list(args), capture_output=True, text=True).stdout.strip()


def check_pin(t):
    d = os.path.join(ARK, t['dir'])
    head = git(d, 'rev-parse', 'HEAD')
    if head != t['commit']:
        raise SystemExit(f'{d}: HEAD is {head!r}, the harvest pins {t["commit"]}')
    dirty = git(d, 'status', '--porcelain')
    if dirty:
        raise SystemExit(f'{d}: the work tree differs from {t["commit"][:12]}:\n{dirty[:600]}')
    return d


def tree_files(root):
    out = []
    for d, dirs, names in os.walk(root):
        if d == root:
            dirs[:] = [x for x in dirs if x not in SKIP]
            names = [n for n in names if n not in SKIP]
        out += [os.path.relpath(os.path.join(d, n), root) for n in names]
    return sorted(out)


def blob(path):
    data = open(path, 'rb').read()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


class DS4Origin(Origin):
    """Base files at their paths plus overlay/<slug>/<path> for every file an overlay adds or changes."""
    def __init__(self):
        super().__init__('ds4', 'ds4', BASE['repo'], BASE['commit'], BASE['branch'], 'MIT',
                         ['The ds4.c authors', 'The ggml authors', 'DeepSeek'], 'therock-10.2')
        self.base_dir = check_pin(BASE)
        self.base = {f: blob(os.path.join(self.base_dir, f)) for f in tree_files(self.base_dir)}
        self.map = {f: os.path.join(self.base_dir, f) for f in self.base}
        self.delta = {}
        for o in OVERLAYS:
            od = check_pin(o)
            mine = {f: blob(os.path.join(od, f)) for f in tree_files(od)}
            rows = []
            for f in sorted(set(mine) | set(self.base)):
                if f not in self.base:
                    rows.append(('A', f, '-', mine[f]))
                elif f not in mine:
                    rows.append(('D', f, self.base[f], '-'))
                elif mine[f] != self.base[f]:
                    rows.append(('M', f, self.base[f], mine[f]))
            self.delta[o['slug']] = rows
            for st, f, _, _ in rows:
                if st != 'D':
                    self.map[f'overlay/{o["slug"]}/{f}'] = os.path.join(od, f)
        self.roots = []

    def files(self):
        if self._files is None:
            self._files = sorted(self.map)
        return self._files

    def find(self, rel):
        return self.map[rel]


# ---------------------------------------------------------------------------------------------------------------
# Families. A rule maps an origin path (base-relative, or overlay-relative) to a family; FAM holds what each family is.
RULES = [
 (r'ds4_rocm\.cu|ds4_rocm\.h|ds4_iq2_tables_cuda\.inc', 'rocm_backend'),
 (r'ds4_rocm_compat\.cu', 'rocm_compat'),
 (r'ds4_rocm_unavailable\.cu', 'rocm_unavailable'),
 (r'rocm/halo/moe/.*|rocm/ds4_rocm_halo_moe\.h', 'rocm_halo_moe'),
 (r'rocm/halo/q2/.*|rocm/ds4_rocm_halo_q2\.(h|cuh)', 'rocm_halo_q2'),
 (r'rocm/halo/shared_gu/.*', 'rocm_halo_shared_gu'),
 (r'rocm/halo/attention/.*|rocm/ds4_rocm_halo_attention\.cuh', 'rocm_halo_attention'),
 (r'rocm/halo/output_a/.*|rocm/ds4_rocm_halo_output_a\.cuh', 'rocm_halo_output_a'),
 (r'rocm/halo/output_a_uncached/.*|rocm/ds4_rocm_halo_output_a_uncached\.cuh', 'rocm_halo_output_a_uncached'),
 (r'rocm/halo/output_b/.*|rocm/ds4_rocm_halo_output_b_module\.cuh', 'rocm_halo_output_b'),
 (r'rocm/v41_gate_up/.*|rocm/ds4_rocm_v41_gate_up\.h', 'rocm_v41_gate_up'),
 (r'rocm/ds4_rocm_(\w+)\.(cuh|h|cu)', r'rocm_\1'),
 (r'ds4_deepseek41_cuda\.cuh|ds4_deepseek41_gpu\.h', 'dsv41'),
 (r'ds4_glm53_vision_gpu\.cuh', 'glm53_vision'),
 (r'ds4_deepseek4_vision_gpu\.cuh', 'dsv4_vision'),
 (r'ds4_qwen4_cuda\.cuh', 'qwen4_cuda'),
 (r'ds4_cuda\.cu', 'cuda_backend'),
 (r'cuda/mmq/(common\.cuh|ggml-common\.h|ggml\.h|ggml-impl\.h|ggml-cuda\.h|mma\.cuh|vecdotq\.cuh|unary\.cuh|VENDOR\.md|vendors/.*)', 'mmq_common'),
 (r'cuda/mmq/ds4_ggml_stubs\.(cu|h)|ds4_rocm_memory\.h', 'mmq_ggml_stubs'),
 (r'cuda/mmq/(ds4_mmq\.(cu|h)|mmq\.cuh)', 'mmq_moe'),
 (r'cuda/mmq/quantize\.(cu|cuh)', 'mmq_quantize'),
 (r'cuda/mmq/mmid\.(cu|cuh)', 'mmq_mmid'),
 (r'cuda/mmq/mmvq\.(cu|cuh)', 'mmq_mmvq'),
 (r'cuda/mmq/(ds4_mmq_d2r\.(cu|cuh)|test/d2r_stubs\.cu)', 'mmq_d2r'),
 (r'cuda/mmq/ds4_repack\.(cu|h)', 'mmq_repack'),
 (r'ds4\.c|ds4\.h', 'engine'),
 (r'ds4_gpu\.h|ds4_gpu_copy\.h|ds4_gpu_mgpu\.h', 'gpu_api'),
 (r'ds4_gpu_tp\.h', 'gpu_tp'),
 (r'ds4_gpu_args\.(c|h)', 'gpu_args'),
 (r'ds4_tp\.(c|h)', 'tp'),
 (r'ds4_tp_io\.h', 'tp_io'),
 (r'ds4_tp_roce\.h', 'tp_roce'),
 (r'tp_odl_test\.c', 'tp_odl_test'),
 (r'ds4_distributed\.(c|h)', 'distributed'),
 (r'ds4_kvstore\.(c|h)', 'kvstore'),
 (r'ds4_ssd\.(c|h)', 'ssd'),
 (r'ds4_streaming_hotlist(_glm52)?\.inc', 'streaming_hotlist'),
 (r'ds4_engram\.(c|h)', 'engram'),
 (r'ds4_layer_pack\.(c|h)', 'layer_pack'),
 (r'ds4_prompt_prefix\.(c|h)', 'prompt_prefix'),
 (r'ds4_deepseek41_dspark\.h', 'dspark'),
 (r'ds4_image\.(c|h)', 'image'),
 (r'third_party/iris/.*', 'iris'),
 (r'ds4_qwen4_unicode\.inc', 'qwen4_unicode'),
 (r'ds4_qwen4_vision\.h', 'qwen4_vision'),
 (r'ds4_linux_memory\.h', 'linux_memory'),
 (r'rax\.(c|h)|rax_malloc\.h', 'rax'),
 (r'linenoise\.(c|h)', 'linenoise'),
 (r'ds4_server\.c|ds4_tool_text\.h', 'server'),
 (r'ds4_web\.(c|h)', 'web'),
 (r'ds4_agent\.c', 'agent'),
 (r'ds4_cli\.c', 'cli'),
 (r'ds4_eval\.c|ds4_eval_cases\.(c|h)', 'eval'),
 (r'ds4_bench\.c', 'bench'),
 (r'ds4_help\.(c|h)', 'help'),
 (r'gguf-tools/(quants\.(c|h)|deepseek4-quantize\.c|Makefile)', 'gguf_quants'),
 (r'gguf-tools/(glm53_full_quantize|glm53_quantize|glm53_manifest|glm53_validate_gguf)\.py', 'gguf_glm53'),
 (r'gguf-tools/(deepseek41_quantize|deepseek41_metadata|deepseek41_validate_gguf|deepseek41_dspark|merge_deepseek41_shards)\.py', 'gguf_dsv41'),
 (r'gguf-tools/(qwen4_\w+|gen_qwen4_unicode)\.py', 'gguf_qwen4'),
 (r'gguf-tools/(deepseek4_vision|deepseek41_vision|glm53_vision)\.py', 'gguf_vision'),
 (r'dir-steering/tools/\w+\.py', 'dir_steering'),
]
RULES = [(re.compile(rx), name) for rx, name in RULES]


def family(path):
    for rx, name in RULES:
        m = rx.fullmatch(path)
        if m:
            return m.expand(name)
    return None


# families an overlay contributes modules for; its other changed files are in its base-diff only
OVERLAY_FAMILIES = re.compile(r'rocm_.*|mmq_.*|dsv41|glm53_vision|dsv4_vision|engine|gpu_api|gpu_tp|tp|tp_io|tp_roce|tp_odl_test|distributed|dspark|linux_memory')


def F(kind, tag, iface, summary, why, lang=None, api='', numeric='', regime=''):
    return dict(kind=kind, tag=tag, iface=iface, summary=summary, why=why, lang=lang, api=api, numeric=numeric, regime=regime)


TU = 'a fragment of the single ROCm translation unit ds4_rocm.cu, where its kernels and helpers stay static'
FAM = {
 'rocm_backend': F('kernel', None, 'kernel-abi.ds4-rocm-backend/-/-', 'ds4\'s ROCm backend translation unit: ds4_rocm.cu sets the HIP spellings (64-bit warp masks, hipBLASLt / rocBLAS names) and #includes every rocm/*.cuh module plus the shared V4.1, GLM-5.3 vision and V4 vision kernels in a fixed order into one object; ds4_rocm.h declares the ROCm-only entry points and ds4_iq2_tables_cuda.inc holds the IQ2_XXS grids', 'the one place where the ROCm kernel modules become a backend: the ds4_gpu.h API (tensor-resident, device-owned activations and KV across a whole prefill / decode command sequence) is implemented here for gfx1151', api='ds4_gpu_* (ds4_gpu.h), ds4_rocm_* (ds4_rocm.h)', regime='Strix Halo gfx1151; eager default-stream execution (no graph capture on ROCm)'),
 'rocm_compat': F('kernel', None, 'device.ds4-rocm-memory/hip/-', 'the ROCm host shims compiled as their own object (ds4_rocm_compat.cu): device allocation through the ds4_rocm_memory.h arena (in kernel.ds4.mmq_ggml_stubs), the multi-GPU entry points of ds4_gpu_mgpu.h, and the decode-graph hooks that are CUDA-only (ROCm decodes eagerly)', 'how ds4 sizes and places device memory on a unified-memory APU, the part of a ROCm backend that is not kernels', api='ds4_gpu_* memory and device hooks'),
 'rocm_unavailable': F('kernel', None, 'kernel-abi.ds4-rocm-unavailable/-/-', 'C-linkage stubs for the CUDA-only fused and multi-device hooks that the shared graph code references but ROCm validation never selects', 'keeps the ROCm link closed without #ifdefs in ds4.c; the list of what ROCm lacks against CUDA'),
 'rocm_attention': F('kernel', 'attn.softmax-gqa', 'attn.softmax-gqa/kv-f16+fp8/f32.ds4', 'DeepSeek V4 attention kernels for raw and mixed (FP8-compressed) KV: prefill raw / mixed with softmax tiles, decode mixed and indexed (8-head online, one-fast), the 16-head WMMA mixed kernel, the 64x64 WMMA output-B projection, head pack / unpack and the non-causal raw batch attention of the DSpark draft block; ' + TU + '', 'the attention core of the DSV4 / V4.1 graph on gfx1151, with WMMA paths; the catalogue\'s attn.softmax-gqa providers are the comparison set', numeric='F32 accumulation; F16 / FP8 KV; rocWMMA where the head count allows', regime='prefill and decode'),
 'rocm_attention_launch': F('kernel', None, 'kernel-abi.ds4-rocm-attention-launch/-/-', 'host launch glue for the attention kernels: shape and KV-format dispatch, split choices and the eligibility rules; ' + TU + '', 'where ds4 decides which attention kernel runs for a given model, context and batch; the policy half of rocm_attention'),
 'rocm_common': F('kernel', 'embed.lookup', 'embed.lookup/q8_0+f16/f32.ds4', 'shared device helpers and the first kernels of the TU: token embedding into the HC streams (F16 and Q8_0), F16 / F32 dense matmuls with shared-x warp rows and ordered chunks, HC repeat and slot packing; ' + TU + '', 'the helpers every later rocm module uses, and the embedding step of every model'),
 'rocm_compressor': F('kernel', 'kvcache.compress', 'kvcache.compress/-/f32.ds4', 'the DSV4 KV compressor: ratio-4 shift, pooled prefill and update, row set and store; ' + TU + '', 'DSV4\'s compressed-KV attention needs it every layer; the catalogue\'s kvcache.compress providers are the comparison'),
 'rocm_current_api_compat': F('kernel', None, 'kernel-abi.ds4-rocm-api-compat/-/-', 'adapters from the current ds4_gpu.h API to the ROCm kernels\' older entry points, kept in one header; ' + TU + '', 'the seam between engine API drift and the kernel modules'),
 'rocm_deepseek4_vision': F('kernel', 'vision.ds4v-ops', 'vision.ds4v-ops/bf16/f32.ds4-rocm', 'ROCm kernels of the DeepSeek V4 vision path: visual mixed-KV pack and softmax attention and the parallel visual router select; ' + TU + '', 'the vision tower\'s attention on gfx1151'),
 'rocm_embedding_launch': F('kernel', None, 'kernel-abi.ds4-rocm-embedding-launch/-/-', 'launch glue for the embedding kernels; ' + TU + '', 'policy half of the embedding step'),
 'rocm_fp8_kv': F('kernel', 'kvcache.store', 'kvcache.store/fp8/f32.ds4', 'FP8 KV quantization and the raw-KV batch store; ' + TU + '', 'the KV write of every attention layer'),
 'rocm_fp8_kv_launch': F('kernel', None, 'kernel-abi.ds4-rocm-fp8-kv-launch/-/-', 'launch glue for the FP8 KV kernels; ' + TU + '', 'policy half of the KV write'),
 'rocm_glm': F('kernel', 'mixer.kda', 'mixer.kda/-/f32.ds4-glm', 'the GLM-5.x ROCm kernels, mirroring the Metal GLM graph entry points: KDA decode and the three-stage KDA prefill (prepare, recurrence, output), BF16 embedding and matvecs, Q4_K x Q8_K matmul, GLM attention (full, indexed split decode with reduce, LoRA-indexed), KV-cache build and compact-KV QKV norm, the indexer (scores, RoPE tail, pool update) and the causal / selected GEMM attention path (gather, RoPE to F16, softmax, scatter); ' + TU + '', 'the GLM-5.3 and GLM-5.3-Flash graph on gfx1151, KDA included; the base of every GLM overlay in this origin', numeric='F32 state; BF16 / Q8_0 / Q4_K weights', regime='prefill and decode'),
 'rocm_hc': F('kernel', 'hc.mix', 'hc.mix/-/f32.ds4', 'the hyper-connection (HC) kernels: split Sinkhorn, weighted sums (fused with norm), and the 4-stream expand / expand-add in F32 and half; ' + TU + '', 'DSV4\'s 4-stream residual runs these every sublayer; the catalogue\'s hc.mix providers are the comparison set'),
 'rocm_hc_output_launch': F('kernel', None, 'kernel-abi.ds4-rocm-hc-output-launch/-/-', 'launch glue for the HC output path; ' + TU + '', 'policy half of the HC write'),
 'rocm_hipblaslt': F('kernel', None, 'tune.ds4-hipblaslt-plan/hip/-', 'hipBLASLt state, plan cache and version checks for the GEMMs that go to the library (HIP only); ' + TU + '', 'how ds4 keeps hipBLASLt planning out of the hot path on ROCm'),
 'rocm_indexer': F('kernel', 'attn.indexer-topk', 'attn.indexer-topk/-/f32.ds4', 'the DSV4 lightning indexer: Hadamard FP4 keys, scores (direct, batched, WMMA-128), top-k (pow2, chunked, tree merge, 1024, 8192 via hipcub), top-1 rows, masks, argmax and the DSpark Markov argmax; ' + TU + '', 'sparse attention selection for DSV4 / V4.1; beside the catalogue\'s attn.indexer-topk providers'),
 'rocm_matmul': F('kernel', 'linear.prefill', 'linear.prefill/f16/f32.ds4-wmma', 'F16 matmuls on WMMA for small and tiny M (small-M, tiny-M 24, tiny-batch wave) and their dispatch; ' + TU + '', 'the projections of short batches (verify windows, small prefills) on gfx1151 matrix cores'),
 'rocm_misc_launch': F('kernel', None, 'kernel-abi.ds4-rocm-misc-launch/-/-', 'launch glue for the pointwise and output kernels; ' + TU + '', 'policy half of the small kernels'),
 'rocm_moe': F('kernel', 'moe.experts-decode', 'moe.experts-decode/iq2_xxs+q2_k+q4_k+mxfp4/f32.ds4', 'the routed-MoE kernels: active-expert and tile building, gate / up / mid decode (LUT, MXFP4, Q4_K, IQ2) and down products in many tilings (expert tiles of 4 / 8 / 16 rows, quarter-warp, sum-6, hotlist WMMA for Q2_K); ' + TU + '', 'the expert matvecs dominate decode for DSV4 and GLM; this is ds4\'s whole gfx1151 MoE kernel set', numeric='quantized weights with F32 accumulation', regime='decode and small batches'),
 'rocm_moe_launch': F('kernel', None, 'kernel-abi.ds4-rocm-moe-launch/-/-', 'host launch glue and policy for the routed MoE: workspace sizing, tile plans, format dispatch; ' + TU + '', 'where ds4 decides which MoE kernel runs'),
 'rocm_norm_rope': F('kernel', 'norm.rmsnorm', 'norm.rmsnorm/-/f32.ds4', 'RMSNorm (plain, weighted, per-head, QKV rows) and the RoPE tail, alone and fused with the head norm; ' + TU + '', 'every layer\'s norms and rotary step'),
 'rocm_output': F('kernel', 'act.swiglu', 'act.swiglu/-/f32.ds4', 'pointwise kernels: SwiGLU, add / add3, zero, the HC output weights and the directional-steering projection; ' + TU + '', 'the small ops between the big ones; steering is ds4\'s dir-steering hook'),
 'rocm_q8': F('kernel', 'linear.decode', 'linear.decode/q8_0/f32.ds4', 'Q8_0 matmuls: dequant to F16 / F32, warp-8 and shared-x row kernels, pre-quantized activations, pairs, batch with token tiles and WMMA row tiles, grouped-output Q8 with partial sums, the HC-expand fused forms and the fused shared-expert gate / up SwiGLU; ' + TU + '', 'Q8_0 is the format of the dense projections in ds4\'s GGUFs, so this is most of the non-expert decode time'),
 'rocm_router': F('kernel', 'moe.router', 'moe.router/-/f32.ds4', 'the warp top-k router select; ' + TU + '', 'expert routing for DSV4 / V4.1'),
 'rocm_runtime': F('kernel', None, 'kernel-abi.ds4-rocm-runtime/-/-', 'the ROCm runtime body: device tensors and arenas, streams, rocBLAS / hipBLAS handles and tuned solution indices, the ds4_gpu_* entry points that are not kernel launches, and Q8_0 dequant helpers; ' + TU + '', 'the biggest rocm module: the executor around the kernels'),
 'rocm_shared_expert': F('kernel', None, 'kernel-abi.ds4-rocm-shared-expert/-/-', 'launch glue for the shared expert; ' + TU + '', 'policy half of the shared-expert FFN'),
 'rocm_tp': F('kernel', 'collective.device-stage', 'collective.device-stage/-/-.ds4-tp', 'device side of two-machine tensor parallelism on ROCm: kernels that publish GPU arrival into the transport slab and park the stream until the CPU transport has exchanged the partials, and the copy / add / release around them; ' + TU + '', 'the GPU half of ds4\'s TP exchange; a TP-over-network shape strixflerp\'s collective modules can be compared with'),
 'rocm_glm_selected_multihead': F('kernel', None, 'attn.ds4-glm-selected-multihead/-/f32', 'a stand-alone HIP translation unit with the GLM selected (sparse, indexer-chosen) multi-head attention kernel', 'GLM-5.3-Flash decode attention over the selected cells in one kernel for all heads'),
 'rocm_halo_moe': F('kernel', 'moe.experts-prefill', 'moe.experts-prefill/iq2_xxs/f32.ds4-halo-s9', 'the Halo IQ2_XXS MoE producers ("S9"): co-computed gate / up with a register epilogue in two tilings (wide, twin) and two variants (public, fresh), their descriptors, the IQ2 signed-grid feed and tables, and the runtime entry that sits on the native MMQ quantizer, stream 0', 'frozen, qualified gfx1151 expert GEMMs for prefill chunks of 2048 / 4096 rows', numeric='native MMQ q8_1 activations; fused SwiGLU multiply order kept', regime='prefill, 2048-row physical workspace'),
 'rocm_halo_q2': F('kernel', 'linear.prefill', 'linear.prefill/q2_k/f16-f32.ds4-halo', 'the Halo Q2 panels: direct and 64-wide staged WMMA kernels, a prepared-weight GEMM on Adel Johar\'s MIT GEMM template, and the runtime that gathers and prepares Q2 weights', 'a gfx1151 Q2 prefill GEMM with a native K8 correction kept'),
 'rocm_halo_shared_gu': F('kernel', 'linear.prefill', 'linear.prefill/q8_0/f16-f32.ds4-halo-shared-gu', 'the Halo shared-expert gate / up GEMM: weight and activation preparation and a fixed-shape LDS-scoped WMMA kernel on Adel Johar\'s MIT GEMM template', 'a qualified shared-expert GEMM for 2K / 4K prefill on gfx1151'),
 'rocm_halo_attention': F('kernel', 'attn.prefill-causal', 'attn.prefill-causal/kv-f16/f32.ds4-halo', 'the Halo half-precision fused direct-QK attention with compact KV access and its support headers', 'frozen gfx1151 prefill attention that keeps the native FP16 WMMA order'),
 'rocm_halo_output_a': F('kernel', 'linear.prefill', 'linear.prefill/f16/f32.ds4-halo-output-a', 'the Halo grouped attention-output projection A: heads-first WMMA with a half epilogue, on Adel Johar\'s MIT GEMM template (candidate/)', 'a frozen gfx1151 output-projection GEMM'),
 'rocm_halo_output_a_uncached': F('kernel', 'linear.prefill', 'linear.prefill/f16/f32.ds4-halo-output-a-uncached', 'the Halo output projection A without the cached partial: XOR-ranked fragment stores into a 32 KiB partial buffer', 'the uncached variant of output A'),
 'rocm_halo_output_b': F('kernel', 'linear.prefill', 'linear.prefill/f16/f32.ds4-halo-output-b', 'the Halo output projection B as gfx1151 assembly (s4_d2.s, MIT, its LICENSE beside it) and the module header that includes the generated code.inc', 'a hand-scheduled gfx1151 GEMM; code.inc is generated at build time by scripts/build_halo_output_b.py and is not in the tree'),
 'rocm_v41_gate_up': F('kernel', 'moe.experts-prefill', 'moe.experts-prefill/iq2_xxs/f32.ds4-v41-gate-up', 'the DeepSeek V4.1 IQ2 gate / up adapter of the Halo S9 producers: wide and twin tiles, descriptors, IQ2 feed and grid, runtime on the native MMQ quantizer', 'V4.1 dimensions and routing capacity on the qualified V4 producers, 2048-row workspace'),
 'rocm_halo_indexer': F('kernel', 'attn.indexer-score', 'attn.indexer-score/-/f32.ds4-halo', 'the Halo resident-key indexer score kernel (native FP16 WMMA K order, head by head)', 'indexer scores over resident keys for long prefills'),
 'rocm_halo_hc': F('kernel', 'hc.mix-prefill', 'hc.mix-prefill/-/f32.ds4-halo', 'the Halo HC RMS-half kernel with the global FP32 boundary kept explicit', 'the HC norm of the Halo prefill path'),
 'rocm_halo_direct_qk': F('kernel', 'attn.prefill-causal', 'attn.prefill-causal/kv-f16/f32.ds4-halo-direct-qk', 'the Halo direct-query FP32 producer with native FP16 WMMA transforms and online softmax / PV', 'prefill attention without cache conversion or extra scratch'),
 'rocm_halo_static_query': F('kernel', 'attn.softmax-gqa', 'attn.softmax-gqa/kv-f16/f32.ds4-halo-static-q', 'the Halo register-query attention kernel, warp reductions and softmax in their original order', 'a register-resident query attention form'),
 'rocm_halo_q8': F('kernel', 'linear.prefill', 'linear.prefill/q8_0/f16-f32.ds4-halo-q8', 'the Halo K16-panel Q8 kernels (FP16 conversion, cooperative K16, native QA)', 'Q8_0 prefill GEMMs with the WMMA operand roles kept'),
 'rocm_halo_shared_down': F('kernel', 'linear.prefill', 'linear.prefill/q8_0/f32.ds4-halo-shared-down', 'the Halo shared-expert down GEMM with a fixed K32 schedule, qualified for 2K / 4K', 'the shared expert\'s down projection in Halo prefill'),
 'rocm_halo_shapes': F('kernel', None, 'kernel-abi.ds4-halo-shapes/-/-', 'the model-qualified shape selector of the Halo kernels (later 2048-token chunks only, 64-bit arithmetic)', 'decides when a Halo kernel may run; everything else takes the native path'),
 'rocm_hc_sgemm': F('kernel', 'hc.mix-prefill', 'hc.mix-prefill/-/f32.ds4-sgemm', 'an HC-only F32 SGEMM plan over the graph\'s lent head storage on stream 0, no device allocation of its own', 'the HC projections of a prefill chunk through one SGEMM'),
 'rocm_qwen4': F('kernel', 'mixer.gdn', 'mixer.gdn/-/f32.ds4-qwen4-rocm', 'Qwen3.8 Flash Next on gfx1151: the GDN scan / prep / output, conv, HC norm and mix, indexer score / select, router, MoE lists and tiles, matvecs and matrices (WMMA expert prefill on IQ2_XXS / Q2_K), MTP stage and combine', 'a whole Qwen3.8-Flash-Next kernel set for gfx1151 following the CPU contracts and the CUDA path'),
 'rocm_qwen4_attention_wmma': F('kernel', 'attn.prefill-causal', 'attn.prefill-causal/kv-f16/f32.ds4-qwen4-wmma', 'experimental residual-corrected Qwen prefill attention on WMMA: 16 queries with the causal-tail accumulator in registers, dense and sparse (union-mask) windows', 'a QSA prefill attention on matrix cores with up to 262,144 tokens of context'),
 'rocm_attention_gufo': F('kernel', 'attn.sparse-sink', 'attn.sparse-sink/kv-f16/f32.ds4-gufo', 'a gufo-derived DeepSeek V4 attention dataflow (from gufo-org/gufo 9da89d64, MIT): single-pass indexed attention with Q staged transposed, top_k 512', 'a challenger to the native DSV4 attention; the catalogue already has gufo\'s own attn.sparse-sink to compare'),
 'rocm_v41': F('kernel', None, 'kernel-abi.ds4-v41-rocm/-/-', 'DeepSeek V4.1 baseline primitives for ROCm: candidates, carry, engram (with LDS token reuse), gather, grouped Q8 (blocks4, WMMA), BF16 and F16 token-reuse kernels, with explicit OCML division to keep tuned arithmetic', 'the V4.1 graph\'s own ops on gfx1151'),
 'dsv41': F('kernel', None, 'kernel-abi.ds4-dsv41/-/-', 'DeepSeek V4.1-specific GPU operations shared by CUDA and ROCm (included after the shared kernels): BF16, candidates, carry, causal and index scores, engram, gather, indexer, pool, quantize, RoPE and the shared-expert gate / up / down / mask / owner kernels; ds4_deepseek41_gpu.h declares them', 'the V4.1 graph\'s ops in one place for both GPU backends'),
 'glm53_vision': F('kernel', 'vision.tower-ops', 'vision.tower-ops/bf16/f32.ds4-glm53', 'GLM-5.3 vision tower operations shared by CUDA and ROCm: attention, bias, layernorm + GELU, QKV RoPE, reorder, RMS, HC scatter, SwiGLU bias', 'the GLM-5.3 image path on the GPU'),
 'dsv4_vision': F('kernel', 'vision.ds4v-ops', 'vision.ds4v-ops/bf16/f32.ds4', 'DeepSeek V4 Flash Vision-Exp operations shared by CUDA and ROCm: residual add, aligner reorder, GELU bias, QKV RoPE, BF16 rounding, SwiGLU split', 'the DSV4 vision tower on the GPU'),
 'qwen4_cuda': F('kernel', 'mixer.gdn', 'mixer.gdn/-/f32.ds4-qwen4-cuda', 'Qwen3.8 Flash Next on CUDA, included by ds4_cuda.cu: GDN, attention, HC, indexer, router, MoE, matvec / matrix (tensor-core) and MTP kernels', 'the CUDA reference that the gfx1151 Qwen port (k53perf_rocm_qwen4) follows', lang='cuda'),
 'cuda_backend': F('kernel', None, 'kernel-abi.ds4-cuda-backend/-/-', 'ds4\'s CUDA backend (one translation unit, about 36,000 lines): the ds4_gpu API on CUDA with multi-GPU, graph capture, DSpark and the shared model kernels', 'the CUDA twin of rocm_backend; the ROCm modules were split out of its kernels and many ROCm headers still say "included from ds4_cuda.cu"', lang='cuda'),
 'mmq_common': F('kernel', None, 'kernel-abi.ds4-ggml-mmq-common/-/-', 'the vendored llama.cpp ggml-cuda headers the MMQ kernels compile against (common.cuh, mma.cuh, vecdotq.cuh, unary.cuh, ggml-common.h, the HIP / MUSA / CUDA vendor maps) and the thin ggml.h / ggml-impl.h / ggml-cuda.h redirects; VENDOR.md records the upstream pin (llama.cpp 5c0e9468, 2026-05-14)', 'the base every mmq_* module needs; ROCm builds them with -DGGML_USE_HIP'),
 'mmq_ggml_stubs': F('kernel', None, 'kernel-abi.ds4-ggml-stubs/-/-', 'the minimum ggml API surface (declarations and a few definitions) that the vendored MMQ code references, so it links without the ggml runtime, and ds4_rocm_memory.h, the ROCm device-memory policy (an arena for small tensors, dedicated spans for large ones) that the stubs back ggml\'s pool with and ds4_rocm_compat.cu allocates through; the header sits in this module because ds4_ggml_stubs.cu reaches it as ../../ds4_rocm_memory.h, and a relative include resolves only inside one module', 'how ds4 lifts llama.cpp kernels out of ggml'),
 'mmq_moe': F('kernel', 'moe.experts-decode', 'moe.experts-decode/iq2_xxs+q2_k+q8_0/q8_1-f32.ds4-mmq', 'ds4\'s quantized-matmul C ABI over llama.cpp\'s mul_mat_q (mmq.cuh): dense Q8_0 / Q2_K / IQ2_XXS and MoE _id variants, the aligned-artifact MoE vec kernels (IQ2_XXS gate / up / mid with dedup, Q2_K, Q8_0 dense), the MXFP4 staged gate / up and the sum-6 down', 'llama.cpp\'s MMQ as a ds4 kernel library; the ROCm build uses it for the expert and dense quantized GEMMs', numeric='q8_1 activations, integer dots, F32 out', regime='decode and prefill'),
 'mmq_quantize': F('kernel', 'quant.activation', 'quant.activation/-/q8_1+mxfp4+nvfp4.ds4-mmq', 'llama.cpp\'s activation quantizers for MMQ: q8_1, MMQ q8_1 / MXFP4 / NVFP4 tiles', 'the src1 conversion every MMQ product needs'),
 'mmq_mmid': F('kernel', None, 'moe.ds4-mmid/-/-', 'llama.cpp\'s mm_ids helper: expert-id to row mapping for the MoE matmuls', 'the gather step of MMQ\'s MoE path'),
 'mmq_mmvq': F('kernel', 'linear.decode', 'linear.decode/gguf-kquant+iq/q8_1-f32.ds4-mmvq', 'llama.cpp\'s mul_mat_vec_q and its MoE form', 'the one-token quantized GEMV of the MMQ family'),
 'mmq_d2r': F('kernel', None, 'moe.ds4-d2r/-/-', 'the gated D2R path: the Q2_K MoE down-GEMM and dense Q8_0 on the kind-5 aligned artifact (ds4_mmq_d2r.cu, CUDA build) and the ROCm stubs that stand in for it (test/d2r_stubs.cu, what the ROCm build links)', 'a derived-to-resident GEMM path and its ROCm placeholder'),
 'mmq_repack': F('kernel', 'linear.row-layout', 'linear.row-layout/iq2_xxs+q2_k+q8_0/-.ds4-aligned', 'the aligned structure-of-arrays repack library shared by the weight server and the engine\'s self-load: catalog walk, candidate predicates, parallel repack, FNV-1a artifact hash, IQ2_XXS / Q2_K / Q8_0 aligned builds', 'the weight layout the aligned MMQ kernels read; a shape for strixflerp\'s format.device-view ideas'),
 'engine': F('runtime', 'executor.model', 'executor.model/gguf/-.ds4', 'ds4.c and ds4.h, the engine: mmap GGUF loading, the fixed DeepSeek V4 / V4.1 / Pro, GLM-5.2 / 5.3 / 5.3-Flash and Qwen3.8-Flash-Next layouts and their graph drivers, CPU reference kernels, sessions (one mutable inference timeline owning KV and logits; ds4_session_sync reuses, extends or rebuilds), MTP and DSpark speculation control, the TP and distributed hooks, tokenizer wiring', 'the whole host side of the engine in one deliberately vertical file; every backend and transport module plugs into it', lang='c', api='ds4_engine_*, ds4_session_* (ds4.h)'),
 'gpu_api': F('runtime', None, 'executor.ds4-gpu-api/-/-', 'the backend ABI: ds4_gpu.h (tensor-resident GPU executor API: opaque device tensors and every op the graph issues), ds4_gpu_copy.h (device copy spans) and ds4_gpu_mgpu.h (multi-GPU types and tensor APIs)', 'the contract every backend (Metal, CUDA, ROCm) implements; what an adapter to strixflerp kernels would target', lang='c', api='ds4_gpu_*'),
 'gpu_tp': F('runtime', 'collective.interface', 'collective.interface/-/-.ds4-gpu-tp', 'the TP gate interface between the graph and a backend: gate sequences both ranks must issue, the optional pinned slab shared by an integrated GPU and the transport, arrival / release ordering', 'how a backend joins the TP exchange without knowing the transport', lang='c'),
 'gpu_args': F('runtime', None, 'tool.ds4-gpu-args/-/-', 'the shared parser for --gpu-vram and --gpu-devices', 'multi-GPU placement options', lang='c'),
 'tp': F('runtime', 'collective.interface', 'collective.interface/-/-.ds4-tp', 'the tensor-parallel transport and lockstep protocol: two ranks run the same graph, each with one contiguous half of the routed experts; rank 0 mirrors every session call to rank 1 over a TCP control socket, and inside each token the partial block outputs go through a registered slab by two-sided RDMA send / recv (Thunderbolt UC on Metal, RoCE RC on Linux) or a full-duplex TCP socket; versioned wire features (command mailbox, verify flags, DSpark agreement)', 'two-machine TP over a network link, the shape the pod\'s own TP4 work lives in', lang='c', api='ds4_tp_* (ds4_tp.h)'),
 'tp_io': F('runtime', None, 'transport.ds4-tp-io/-/-', 'full-duplex socket I/O for the TP exchange: both directions advance independently of socket buffer capacity', 'the TCP data path of TP', lang='c'),
 'tp_roce': F('runtime', 'transport.verbs', 'transport.verbs/-/-.ds4-roce', 'an optional Linux RC SEND / RECV adapter over libibverbs, loaded lazily: buffers are aligned host memory pinned once by ibv_reg_mr (no GPU MR), QP teardown before deregistration, quarantine of buffers a provider cannot release', 'RoCE for two-machine TP on Linux / ROCm', lang='c'),
 'tp_odl_test': F('tool', None, 'tool.ds4-tp-odl-test/-/-', 'a transport-level A/B for the TP data plane without a model: synthetic hello identity, slab attach barrier and gate exchanges over odl, tcp or rdma, leader and worker', 'validates and benchmarks a TP transport with no weights', lang='c'),
 'distributed': F('runtime', 'transport.stage-relay', 'transport.stage-relay/-/f32.ds4', 'the distributed (pipeline-parallel) runtime: workers execute contiguous layer slices with the local graph-slice entry points; the coordinator session delegates sync / eval / save / load; KV snapshots stay topology-independent', 'gluing several machines\' RAM for larger models; the catalogue\'s transport.stage-relay is the comparison', lang='c', api='ds4_dist_* (ds4_distributed.h)'),
 'kvstore': F('runtime', None, 'cache.ds4-kv-disk/-/-', 'disk KV checkpoint files: the shared layout and payload helpers under the server\'s byte-prefix cache and the agent\'s sessions, with a graph-payload ABI byte', 'durable KV for long sessions', lang='c'),
 'ssd': F('runtime', 'runtime.weight-prefetch', 'runtime.weight-prefetch/-/-.ds4-ssd', 'SSD streaming of routed experts: budgets and reads for models larger than RAM', 'running GLM 5.3 or DeepSeek V4 Pro on 128 GB machines', lang='c'),
 'streaming_hotlist': F('runtime', None, 'format.ds4-expert-hotlist/-/-', 'generated expert hotlists (sorted by hits per weight) that SSD streaming keeps resident: DeepSeek V4 Pro and GLM-5.2', 'the data half of SSD streaming', lang='c'),
 'engram': F('runtime', None, 'runtime.ds4-engram/-/-', 'the DeepSeek V4.1 engram tables: loading and lookup, on disk or resident', 'V4.1\'s n-gram memory', lang='c'),
 'layer_pack': F('runtime', None, 'runtime.ds4-layer-pack/-/-', 'the monotonic-contiguous multi-tier layer placement packer (devices then CPU), pure C99', 'how layers are placed across GPUs and the CPU tier', lang='c'),
 'prompt_prefix': F('runtime', 'cache.prompt', 'cache.prompt/-/-.ds4-prefix', 'role-marker prompt-prefix detection for prefix reuse', 'prompt caching across turns', lang='c'),
 'dspark': F('runtime', 'speculation.draft', 'speculation.draft/-/-.ds4-dspark', 'the internal DeepSeek V4.1 drafter graph (DSpark): a 128-token draft window, five draft rows and verify rows, Markov and confidence heads, verify / rewind state; included after the target graph in ds4.c', 'ds4\'s speculative decoding for V4.1; spec-decode is its own module', lang='c'),
 'image': F('runtime', 'vision.preprocess', 'vision.preprocess/-/-.ds4', 'image decoding and preprocessing for the vision models', 'turns request images into the vision towers\' input', lang='c'),
 'iris': F('runtime', 'vision.image-decode', 'vision.image-decode/jpeg+png/-.ds4-iris', 'iris: single-header JPEG and PNG decoders (MIT, its LICENSE beside them)', 'dependency-free image decode', lang='c'),
 'qwen4_unicode': F('runtime', 'text.tokenizer', 'text.tokenizer/qwen/-.ds4-unicode', 'generated Unicode tables for the Qwen tokenizer', 'tokenizer pre-split for Qwen3.8', lang='c'),
 'qwen4_vision': F('runtime', None, 'vision.ds4-qwen4-vision-abi/-/-', 'the Qwen3.8 vision types and entry points shared by the engine and the GPU backends', 'the Qwen vision contract', lang='c'),
 'linux_memory': F('runtime', None, 'device.ds4-linux-memory/-/-', 'Linux available-memory accounting with CMA pages excluded (they back movable allocations, not pinned GPU pages)', 'memory budgeting on unified-memory APUs', lang='c'),
 'rax': F('runtime', None, 'runtime.ds4-rax/-/-', 'rax, antirez\'s radix tree (BSD-3-Clause)', 'prefix lookups in the server and agent', lang='c'),
 'linenoise': F('tool', None, 'tool.ds4-linenoise/-/-', 'linenoise line editing (BSD-2-Clause)', 'the CLI and agent prompt', lang='c'),
 'server': F('serve', 'serve.engine', 'serve.engine/-/-.ds4', 'ds4-server: OpenAI / Anthropic-compatible local HTTP server; per-connection threads queue jobs to resident session workers, a model coordinator batches decode-ready sessions and serializes bounded prefill quanta; tool-text escaping (ds4_tool_text.h)', 'a micro-batching serve front end over one engine', lang='c'),
 'web': F('serve', None, 'serve.ds4-web/-/-', 'the built-in web UI served by ds4-server', 'browser front end', lang='c'),
 'agent': F('tool', None, 'tool.ds4-agent/-/-', 'ds4-agent, the terminal coding agent on the engine API', 'an agent loop with compaction and sessions', lang='c'),
 'cli': F('tool', None, 'tool.ds4-cli/-/-', 'the ds4 CLI: one-shot and interactive chat with a live session', 'the reference client', lang='c'),
 'eval': F('tool', None, 'tool.ds4-eval/-/-', 'ds4-eval: built-in benchmark integration test (GPQA Diamond, SuperGPQA, AIME 2025, COMPSEC; hard suite) with its cases', 'quality check on the real model', lang='c'),
 'bench': F('tool', None, 'tool.ds4-bench/-/-', 'the throughput benchmark: prefill at context frontiers and fixed greedy decode, snapshot and restore outside the timing', 'how ds4 measures itself', lang='c'),
 'help': F('tool', None, 'tool.ds4-help/-/-', 'the shared --help text', 'option documentation', lang='c'),
 'gguf_quants': F('tool', 'format.convert', 'format.convert/gguf/-.ds4-quants', 'the GGUF quantization facade in C (float conversion, q4_1, q8_0, q2_K, q4_K, iq2_xxs; derived from GGML\'s MIT quantizers) and the DeepSeek V4 Flash / Pro HF to GGUF quantizer (FP8 / FP4 dequant, imatrix), with their Makefile', 'how ds4\'s GGUFs are made', lang='c'),
 'gguf_glm53': F('tool', 'format.convert', 'format.convert/gguf/-.ds4-glm53', 'GLM-5.3 quantization scripts: full and recipe quantizers, manifest and GGUF validation', 'how the GLM-5.3 GGUFs are made', lang='python'),
 'gguf_dsv41': F('tool', 'format.convert', 'format.convert/gguf/-.ds4-dsv41', 'DeepSeek V4.1 scripts: quantizer, metadata, validation, DSpark support GGUF and shard merge', 'how the V4.1 GGUFs and the DSpark support file are made', lang='python'),
 'gguf_qwen4': F('tool', 'format.convert', 'format.convert/gguf/-.ds4-qwen4', 'Qwen3.8 Flash Next scripts: convert, IQ2, native n-grams, pack and the Unicode table generator', 'how the Qwen packs are made', lang='python'),
 'gguf_vision': F('tool', 'tool.vision-extract', 'tool.vision-extract/safetensors/bf16.ds4', 'vision tower extraction scripts for DeepSeek V4, V4.1 and GLM-5.3', 'how the vision GGUFs are made', lang='python'),
 'dir_steering': F('tool', None, 'tool.ds4-dir-steering/-/-', 'directional steering tools: build a direction from contrast prompts and sweep it', 'produces the vectors rocm_output\'s steering projection applies', lang='python'),
}

# Hand-written notes on what an overlay's module adds (beyond the generated line counts and new kernels).
OV_NOTES = {
 ('glm_tp_rocm', 'rocm_tp'): 'New in this branch: the ROCm TP device side. The slab lives in host-mapped pinned memory (ds4_gpu_tensor_alloc_shared), coherent on Strix Halo with HSA_XNACK=1; ds4_gpu_tp_gate_encode appends a flag-set kernel (fenced with __threadfence_system) and a spin kernel around each exchange on the in-order default stream, plus split and big gates, a stage copy and a K-sliced pre-quantized Q8_0 matmul for the split projections.',
 ('glm_tp_rocm', 'tp'): 'The branch grows the TP protocol for GLM-5.3-Flash on ROCm (+1,717 lines in ds4_tp.c): the KDA head split and the 50/50 routed-expert split across the two ranks.',
 ('glm_tp_rocm', 'engine'): 'The GLM TP graph changes in ds4.c: which heads and experts each rank owns and where the gates go.',
 ('glm_tp_rocm', 'rocm_glm'): 'GLM kernels for the split: K-sliced and multi-row BF16 matvecs, indexed attention split into scores, PV and reduce, an indexer GEMM prep / reduce.',
 ('glm_tp_rocm', 'distributed'): '+1,171 lines in ds4_distributed.c.',
 ('glm_tp_rocm', 'tp_odl_test'): 'New in this branch.',
 ('kyuz0_glm53_flash', 'rocm_glm_selected_multihead'): 'New in this branch, built as its own object (ROCM_GLM_KERNEL_OBJS, -std=c++17).',
 ('kyuz0_glm53_prefill', 'rocm_glm_selected_multihead'): 'Same file as in kyuz0_glm53_flash.',
 ('kyuz0_glm53_perf', 'rocm_tp'): 'A different ROCm TP device side from glm_tp_rocm\'s: one compute stream and one CPU transport service, each queue slot with exact ready / done / consumed generations; the service never calls HIP and the GPU releases a slot only after its guarded consumer has finished.',
 ('kyuz0_glm53_perf', 'tp'): 'TP rewritten for the ROCm cluster mode of docs/CLUSTERING_ROCM.md (two gfx1151 machines, experts split equally, TCP or RoCE).',
}
COMMON_NOTE = 'Verbatim; the module is one family of the ds4 harvest (docs/harvest/harvest_ds4.py, card 31224b4ca3).'

GATE_FAILS = ('kernel.ds4.rocm_backend, glmtp_rocm_backend and k53flash_rocm_backend stop on one error, a host-side rsqrtf call in '
              'the base rocm/ds4_rocm_deepseek4_vision.cuh:239 that the HIP headers of TheRock 10.0, 10.1 and 10.2 have no host overload for '
              '(the rest of each translation unit is accepted; k53prefill and kmultinode, whose trees lack that header, build); '
              'k53perf_rocm_backend needs rocm/halo/output_b/code.inc, which that branch\'s Makefile generates from s4_d2.s at build time; '
              'k53perf_rocm_halo_moe and k53perf_rocm_halo_q2 (runtime.hip) reach base files by ../../../ relative includes out of the overlay prefix; '
              'k53prefill_mmq_ggml_stubs and k53prefill_mmq_moe see both the base\'s and the branch\'s cuda/mmq/vendors/hip.h, because the '
              'unchanged common.cuh includes the copy beside it')
ROCM_FLAGS = ['-x', 'hip', '-O3', '-ffast-math', '-fno-finite-math-only', '-pthread', '-D__HIP_PLATFORM_AMD__', '-Wno-unused-command-line-argument', '-c']
MMQ_EXTRA = ['-std=c++17', '-DGGML_USE_HIP', '-DDS4_HIP_MMQ_Y=64']


def tree_std(slug):
    return ['-std=c++20'] if slug == 'kyuz0_glm53_perf' else []


def lines_of(path):
    return len(open(path, 'rb').read().splitlines())


GLOBAL = re.compile(rb'__global__[^;{]*?\b(\w+)\s*\(', re.S)


def kernels_of(path):
    return set(k.decode() for k in GLOBAL.findall(open(path, 'rb').read()))


def build(O):
    """[(M, family, slug or None)] for the base and every overlay."""
    groups = {}                                   # (slug, family) -> [origin path]
    for f in O.base:
        fam = family(f)
        if fam:
            groups.setdefault((None, fam), []).append(f)
    for o in OVERLAYS:
        for st, f, _, _ in O.delta[o['slug']]:
            fam = None if st == 'D' else 'engine' if f == 'LICENSE' else family(f)
            if fam and OVERLAY_FAMILIES.fullmatch(fam):
                groups.setdefault((o['slug'], fam), []).append(f)
    for key in groups:
        if key[1] not in FAM:
            raise SystemExit(f'family {key[1]} ({groups[key][:3]}) has no FAM entry')
    present = {}                                  # slug -> set of families
    for slug, fam in groups:
        present.setdefault(slug, set()).add(fam)

    def mid(slug, fam):
        meta = FAM[fam]
        return f'{meta["kind"]}.ds4.' + (fam if slug is None else f'{OVL[slug]["prefix"]}_{fam}')

    def pre(slug):
        return '' if slug is None else f'overlay/{slug}/'

    out = []
    for (slug, fam), files in sorted(groups.items(), key=lambda kv: (kv[0][0] or '', kv[0][1])):
        meta = FAM[fam]
        p = pre(slug)
        paths = sorted(p + f for f in files)
        lang = meta['lang'] or 'hip-c++'
        srcs, compile, reason, flags = [], 'n/a', None, None
        base_inc = ('src/' + p).rstrip('/')
        includes = [base_inc]
        if fam.startswith('rocm_') or fam in ('dsv41', 'glm53_vision', 'dsv4_vision'):
            includes = [base_inc, base_inc + '/rocm']
        if fam.startswith('mmq_'):
            includes = [base_inc, base_inc + '/cuda/mmq']
        if lang == 'hip-c++':
            cu = [x for x in paths if x.endswith(('.cu', '.hip'))]
            if fam == 'mmq_d2r':
                cu = [x for x in cu if x.endswith('test/d2r_stubs.cu')]
            if cu:
                srcs, compile = cu, 'standalone'
                std = tree_std(slug)
                if fam.startswith('mmq_'):
                    flags = ROCM_FLAGS[:-1] + std + MMQ_EXTRA + ['-c']
                elif fam in ('rocm_halo_moe', 'rocm_v41_gate_up'):
                    flags = ROCM_FLAGS[:-1] + std + ['-DGGML_USE_HIP', '-c']
                elif fam == 'rocm_glm_selected_multihead':
                    flags = ROCM_FLAGS[:-1] + std + ['-std=c++17', '-c']
                elif std:
                    flags = ROCM_FLAGS[:-1] + std + ['-c']
            elif fam == 'mmq_common':
                compile = 'via:' + mid(slug if 'mmq_moe' in present.get(slug, ()) else None, 'mmq_moe')
            else:
                compile = 'via:' + mid(slug if 'rocm_backend' in present.get(slug, ()) else None, 'rocm_backend')
        elif lang == 'cuda':
            reason = 'CUDA source (nvcc in the origin Makefile); the compile gate builds hip-c++ modules only'
        elif lang == 'c':
            reason = 'host C99 / C11 built by the origin Makefile with cc; the compile gate builds hip-c++ modules only'
        else:
            reason = 'Python tool; nothing to compile'
        # requires: an overlay module asks for its own overlay's modules first, then the base's, so that the include
        # path finds the overlay's copy of a header before the base's
        req = []
        if fam == 'rocm_backend' or (fam.startswith('rocm_halo') or fam in ('rocm_v41_gate_up',)):
            want = sorted(x for x in FAM if x.startswith('rocm_')) + ['dsv41', 'glm53_vision', 'dsv4_vision', 'gpu_api', 'gpu_tp', 'image', 'qwen4_vision', 'linux_memory', 'mmq_common', 'mmq_ggml_stubs', 'mmq_moe']
            for s in ([slug] if slug else []) + [None]:
                for w in want + ([fam] if s is None and slug else []):
                    if w in present.get(s, ()) and not (s == slug and w == fam):
                        r = 'module:' + mid(s, w)
                        if r not in req:
                            req.append(r)
        elif fam.startswith('mmq_') and fam != 'mmq_common':
            for s in ([slug] if slug else []) + [None]:
                for w in ('mmq_common', 'mmq_ggml_stubs', 'mmq_quantize', 'mmq_mmid', 'mmq_mmvq', 'mmq_moe', 'mmq_d2r', 'mmq_repack', 'linux_memory'):
                    if w in present.get(s, ()) and not (s == slug and w == fam):
                        r = 'module:' + mid(s, w)
                        if r not in req:
                            req.append(r)
        elif fam in ('rocm_compat', 'rocm_glm_selected_multihead'):
            for s in ([slug] if slug else []) + [None]:
                for w in ('mmq_ggml_stubs', 'gpu_api', 'gpu_tp', 'gpu_args', 'linux_memory') + ((fam,) if s is None and slug else ()):
                    if w in present.get(s, ()) and not (s == slug and w == fam):
                        r = 'module:' + mid(s, w)
                        if r not in req:
                            req.append(r)
        # describe
        if slug is None:
            summary, why = meta['summary'], meta['why']
            authors = None
            notes = COMMON_NOTE
        else:
            o = OVL[slug]
            parts, newk = [], set()
            rows = {f: st for st, f, _, _ in O.delta[slug]}
            for f in files:
                theirs = O.find(p + f)
                if rows[f] == 'A':
                    parts.append(f'{f} added ({lines_of(theirs)} lines)')
                    newk |= kernels_of(theirs)
                else:
                    mine = O.find(f)
                    a = open(mine, encoding='utf-8', errors='surrogateescape').read().splitlines()
                    b = open(theirs, encoding='utf-8', errors='surrogateescape').read().splitlines()
                    plus = minus = 0
                    for l in difflib.unified_diff(a, b, lineterm='', n=0):
                        if l.startswith('+') and not l.startswith('+++'):
                            plus += 1
                        elif l.startswith('-') and not l.startswith('---'):
                            minus += 1
                    parts.append(f'{f} changed (+{plus} -{minus} lines)')
                    newk |= kernels_of(theirs) - kernels_of(mine)
            nk = sorted(newk)
            summary = (f'{o["title"]}: its copy of the {fam} family against antirez/ds4 b3f5018: ' + '; '.join(parts) +
                       (f'. Kernels not in the base: {", ".join(nk[:24])}' + (f' and {len(nk) - 24} more' if len(nk) > 24 else '') if nk else '') +
                       f'. The family: {meta["summary"]}')
            why = o['why'] + '. The family in the base: ' + meta['why']
            authors = o['authors'][:]
            if fam.startswith('rocm_halo') or fam == 'rocm_v41_gate_up':
                authors += ['MadRocK-AI/ds4 551219f (Halo import, docs/HALO_THIRD_PARTY.md)']
                if fam in ('rocm_halo_q2', 'rocm_halo_shared_gu', 'rocm_halo_output_a', 'rocm_halo_output_b'):
                    authors += ['Adel Johar (MIT, 2024)']
            if fam == 'rocm_attention_gufo':
                authors += ['gufo-org/gufo 9da89d64 (MIT)']
            notes = ' '.join(x for x in (OV_NOTES.get((slug, fam)), f'Overlay {slug}: files under overlay/{slug}/ are the branch\'s bytes at {o["commit"][:12]}; unchanged files of the branch are the base modules\' (docs/harvest/data/ds4/{slug}.tsv).', COMMON_NOTE) if x)
        if fam.startswith('mmq_'):
            authors = (authors or ['The ds4.c authors', 'DeepSeek']) + ['The ggml authors (llama.cpp 5c0e9468, MIT)']
        spdx = {'rax': 'BSD-3-Clause', 'linenoise': 'BSD-2-Clause'}.get(fam)
        iface = meta['iface']
        if slug is not None:
            head, _, last = iface.rpartition('/')
            dtype, _, tag = last.partition('.')
            iface = f'{head}/{dtype}.{tag or "ds4"}-{OVL[slug]["prefix"]}'
        prov = [P(iface, meta['api'] or 'see summary', meta['numeric'] or 'as the origin builds it', meta['regime'] or 'as the origin runs it')]
        m = M(mid(slug, fam), summary, paths, provides=prov, requires=req, lang=lang, sources=srcs, includes=includes,
              flags=flags or (), authors=authors, spdx=spdx, notes=notes, compile=compile, reason=reason, why=why)
        m.overlay = OVL.get(slug)
        m.family, m.slug, m.tag = fam, slug, meta['tag']
        out.append(m)
    return out


ORIGIN_TOML = '''
# DwarfStar (antirez/ds4): a small native C inference engine for DeepSeek V4 / V4.1 Flash (and Pro), GLM-5.2 / 5.3 /
# 5.3-Flash and Qwen3.8 Flash Next, with Metal, CUDA and ROCm (Strix Halo gfx1151) backends, SSD expert streaming,
# two-machine tensor parallel and multi-machine pipeline parallel, a micro-batching server and an agent. Harvested with
# five GLM / ROCm branches of it as overlays (card ORIGIN-DS4 31224b4ca3).
name      = "ds4"
repo      = "https://github.com/antirez/ds4"
commit    = "b3f50180ec2c7324892be1f50bc83555ccb8d5e7"
branch    = "main"
version   = "main at 2026-11-03 ('Read half-aligned Qwen Q2 weight rows correctly on Metal')"
reference_checkout = "control-node: python3 docs/harvest/harvest_ds4.py --compose DIR writes the reference tree flerp diff-origin --origin ds4 reads: the base ark copy /data/ark-inbox/src/github.com__antirez__ds4 (a shallow git clone, HEAD b3f50180ec2c, clean) at its own paths and overlay/<slug>/<path> for every file each overlay adds or changes, as symlinks into the overlay's ark copy; the harvest checks each copy's HEAD against its pin and that its work tree is clean"

[licence]
spdx      = "MIT"
files     = LICENCE_FILES                                  # copied verbatim next to this file
third_party = "LICENSE: MIT, The ds4.c authors (2026), The ggml authors (2023-2026), DeepSeek (2023). cuda/mmq is vendored from llama.cpp's ggml-cuda at 5c0e9468 (MIT, The ggml authors; VENDOR.md). rax is BSD-3-Clause and linenoise BSD-2-Clause (their headers). third_party/iris is MIT (its LICENSE, in runtime.ds4.iris). licenses/Apache-2.0.txt covers redistributed benchmark material in the eval data (EVAL_DATA.md), not harvested. Overlays: davidcanar/ds4 carries the base LICENSE unchanged; the kyuz0/ds4 branches carry an MIT LICENSE without the DeepSeek line (overlay/<slug>/LICENSE, harvested in that overlay's engine module). kyuz0_glm53_perf's Halo kernels come from MadRocK-AI/ds4 551219f under the repository MIT licence; the GEMM template headers under rocm/halo/output_a, rocm/halo/shared_gu/kernel and rocm/halo/q2/prepared keep Adel Johar's MIT notice (2024), rocm/halo/output_b/LICENSE is that MIT licence for the assembly (docs/HALO_THIRD_PARTY.md, rocm/halo/moe/NOTICE.md, rocm/v41_gate_up/NOTICE.md); rocm/ds4_rocm_attention_gufo.cuh is adapted from gufo-org/gufo 9da89d64 (MIT). No weights are in any tree."

[toolchain]
host      = "C99 (-std=c99, -D_GNU_SOURCE, -O3 -ffast-math -fno-finite-math-only) for the engine; C11 for the quality tools; make"
device    = "therock-10.2"                               # the origin names hipcc from /opt/rocm and gfx1151 by default (ROCM_ARCH)
build     = "make strix-halo (alias make rocm): ds4.c, ds4_tp.c, ds4_distributed.c, ds4_ssd.c and the other host .c files with cc -DDS4_ROCM_BUILD -fPIC; ds4_rocm.cu, ds4_rocm_compat.cu and ds4_rocm_unavailable.cu with hipcc ROCM_CFLAGS; cuda/mmq/{ds4_ggml_stubs,ds4_mmq,quantize,mmid,mmvq}.cu and test/d2r_stubs.cu with ROCM_MMQ_FLAGS (-std=c++17 -DGGML_USE_HIP -DDS4_HIP_MMQ_Y=64 -Icuda/mmq); linked by hipcc with -lhipblas -lhipblaslt -lrocblas. kyuz0_glm53_perf adds the Halo objects (rocm/halo/**/*.hip, rocm/v41_gate_up/**/*.hip) and builds with -std=c++20; kyuz0_glm53_flash / _prefill add rocm/ds4_rocm_glm_selected_multihead.cu (-std=c++17)"

[compile]                                                # hipcc ROCM_CFLAGS: clang++ -x hip, one object per translation unit
driver    = "<rocm>/lib/llvm/bin/clang++"
arch_flag = "--offload-arch=gfx1151"
rocm_flag = "--rocm-path=<rocm>"
flags     = ["-x", "hip", "-O3", "-ffast-math", "-fno-finite-math-only", "-pthread", "-D__HIP_PLATFORM_AMD__", "-Wno-unused-command-line-argument", "-c"]
system_includes = ["include"]
output    = "host object with the gfx1151 device code embedded (.o), one per translation unit"
gate      = "GATE_LINE"
note      = "ROCM_CFLAGS without -g (debug info only). The ROCm kernel headers rocm/*.cuh are fragments of ds4_rocm.cu: each is its own module with [gate] compile = via:<the backend module>, which compiles them all. The MMQ modules carry ROCM_MMQ_FLAGS in [build] options, the kyuz0_glm53_perf modules -std=c++20 (that branch's ROCM_CFLAGS). -include paths follow [[requires]] order, overlay modules first, so an overlay's copy of a header is found before the base's."

[kernels]
loading   = "statically linked: one ROCm translation unit (ds4_rocm.cu) holds the kernels of every rocm/*.cuh as static functions, plus the MMQ objects (and the Halo objects on kyuz0_glm53_perf); launches go to the default stream, eagerly (no graph capture on ROCm)"
selection = "the graph code in ds4.c calls the ds4_gpu_* API (ds4_gpu.h); each *_launch.cuh picks the kernel by model, shape, quant format and environment switches (DS4_* variables)"
models    = "DeepSeek V4 Flash / Pro and V4.1 Flash (with DSpark drafter and engram), GLM-5.2, GLM-5.3, GLM-5.3-Flash (KDA), Qwen3.8 Flash Next, plus their vision towers; GGUF only (README.md, docs/MODELS.md)"
'''


def overlay_toml(O):
    out = []
    for o in OVERLAYS:
        rows = O.delta[o['slug']]
        n = {k: sum(1 for r in rows if r[0] == k) for k in 'AMD'}
        out.append(f'''
[[overlay]]
slug      = "{o['slug']}"
prefix    = "{o['prefix']}"                    # module names <kind>.ds4.<prefix>_<family>
repo      = "{o['repo']}"
branch    = "{o['branch']}"
commit    = "{o['commit']}"
date      = "{o['date']}"
reference_checkout = "control-node:/data/ark-inbox/src/{o['dir']}"
base_diff = "docs/harvest/data/ds4/{o['slug']}.tsv"   # {n['A']} added, {n['M']} changed, {n['D']} base files absent, against b3f50180ec2c
what      = {harvestlib.tq(o['why'])}''')
    return ''.join(out)


LEFT_OUT = {
    'metal/, ds4_metal.m, ds4_metal_device.h': 'the Metal backend (Apple GPUs); strixflerp has no Metal language and the pod no Apple GPU',
    'tests/, cuda/mmq/test/ (except d2r_stubs.cu), speed-bench/': 'tests, parity programs and benches with recorded results',
    'gguf-tools/imatrix, gguf-tools/mixed, gguf-tools/quality-testing, gguf-tools/tests, gguf-tools/README.md': 'imatrix data, mixed recipes, the quality-testing corpora and tool tests',
    'docs/, misc/, *.md, logo.svg, licenses/': 'documentation and the Apache-2.0 text for redistributed eval data; read for the module records',
    'dir-steering/examples, dir-steering/out, dir-steering/README.md': 'steering example prompts and an output vector',
    'Makefile, download_model.sh, .github': 'build system and helpers (the ROCm compile options are in ORIGIN.toml [compile])',
    'overlay/<slug>/ files outside the GPU, TP, transport and engine families': 'each overlay\'s copy of the server, CLI, agent, Metal, tests, docs and the other host tools: listed in its base-diff, not harvested',
    'LICENSE': 'copied beside ORIGIN.toml (each kyuz0 overlay\'s own LICENSE is in its engine module)',
}


def base_diff(O):
    os.makedirs(DATA, exist_ok=True)
    owner = {}
    for m in MODS:
        for p in m.paths:
            owner[p] = m.id
    for o in OVERLAYS:
        lines = [f'# {o["repo"]} {o["branch"]} {o["commit"]} against antirez/ds4 main {BASE["commit"]} (git blob ids of the ark copies).',
                 '# Written by docs/harvest/harvest_ds4.py; A added by the branch, M changed, D a base file the branch lacks; module = where the branch file is harvested.',
                 'status\tpath\tbase_blob\tbranch_blob\tmodule']
        for st, f, b, t in O.delta[o['slug']]:
            lines.append(f'{st}\t{f}\t{b}\t{t}\t{owner.get(f"overlay/{o["slug"]}/{f}", "-")}')
        open(os.path.join(DATA, o['slug'] + '.tsv'), 'w').write('\n'.join(lines) + '\n')


def providers():
    out = {}
    cat = os.path.join(harvestlib.REPO, 'catalog')
    for origin in sorted(os.listdir(cat)):
        if origin == 'ds4' or not os.path.isdir(os.path.join(cat, origin)):
            continue
        for d, _, fs in os.walk(os.path.join(cat, origin)):
            if 'module.toml' in fs:
                t = tomllib.load(open(os.path.join(d, 'module.toml'), 'rb'))
                out[t['id']] = [x.get('interface', '') for x in t.get('provides', [])]
    return out


def same_op(tag, prov):
    hits = []
    for mid, ifs in sorted(prov.items()):
        for i in ifs:
            first = i.split('/')[0]
            if first == tag or first.startswith(tag + '-'):
                hits.append(mid)
                break
    return hits


def modules_md(O):
    prov = providers()
    bt = '\x60'
    o = ['# ds4 modules by strixflerp interface', '',
         'One row for each module of catalog/ds4 (card ORIGIN-DS4 31224b4ca3), written by ' + bt + 'docs/harvest/harvest_ds4.py' + bt + '.',
         'Base modules hold antirez/ds4 b3f50180ec2c; overlay modules (name prefix glmtp_, k53flash_, k53prefill_, k53perf_, kmultinode_)',
         'hold one branch\'s added or changed files of that family under ' + bt + 'overlay/<slug>/' + bt + '.', '',
         '- **tag**: the existing strixflerp interface class (' + bt + '<family>.<op>' + bt + ') the module matches, or none. Every module also',
         '  provides its own full interface name (module.toml ' + bt + '[[provides]]' + bt + '), with a ds4 variant tag.',
         '- **same op in the catalogue**: modules of other origins that provide the tagged class. The same op is not the same operands:',
         '  ds4 kernels take ds4_gpu tensors in ds4\'s layouts, so each needs an adapter before it can stand in for a strixflerp module.', '',
         '| module | files | lines | tag | same op in the catalogue |', '|---|---:|---:|---|---|']
    tagged = 0
    by = {}
    for m in MODS:
        n = sum(lines_of(O.find(p)) for p in m.paths)
        hits = same_op(m.tag, prov) if m.tag else []
        tagged += bool(m.tag)
        by[m.slug or 'base'] = by.get(m.slug or 'base', 0) + 1
        shown = ', '.join(bt + h + bt for h in hits[:5]) + (f' and {len(hits) - 5} more' if len(hits) > 5 else '')
        o.append(f'| {bt}{m.id}{bt} | {len(m.paths)} | {n} | ' + (bt + m.tag + bt if m.tag else 'none') + f' | {shown or "-"} |')
    o += ['', f'{len(MODS)} modules ({", ".join(f"{k} {v}" for k, v in by.items())}); {tagged} carry a tag, {len(MODS) - tagged} are none.', '',
          '## Overlays', '']
    for ov in OVERLAYS:
        rows = O.delta[ov['slug']]
        n = {k: sum(1 for r in rows if r[0] == k) for k in 'AMD'}
        o.append(f'- {bt}{ov["slug"]}{bt} ({ov["prefix"]}_*): {ov["title"]}. {n["A"]} file(s) added, {n["M"]} changed, {n["D"]} base file(s) absent '
                 f'(docs/harvest/data/ds4/{ov["slug"]}.tsv). {ov["why"]}.')
    o += ['', '## Not done here', '', 'No kernel was run or timed on the pod. The compile gate (' + bt + 'gate/compile/run.py --origin ds4' + bt +
          ', control-node CPU, gfx1151) compiles the standalone hip-c++ modules; host C, CUDA and Python modules are not compiled.']
    open(os.path.join(harvestlib.REPO, 'catalog', 'ds4', 'MODULES.md'), 'w').write('\n'.join(o) + '\n')


def compose(O, out):
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit(f'{out} exists and is not empty')
    for rel in O.files():
        dst = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.symlink(O.find(rel), dst)
    print(f'compose: {len(O.files())} files under {out}')


_write_module = harvestlib.write_module


def write_module(origin, m, results, extra_requires=()):
    ov = getattr(m, 'overlay', None)
    saved = (origin.repo, origin.commit, origin.branch)
    if ov:
        origin.repo, origin.commit, origin.branch = ov['repo'], ov['commit'], ov['branch']
    try:
        return _write_module(origin, m, results, extra_requires)
    finally:
        origin.repo, origin.commit, origin.branch = saved


harvestlib.write_module = write_module

if __name__ == '__main__':
    O = DS4Origin()
    if len(sys.argv) == 3 and sys.argv[1] == '--compose':
        compose(O, sys.argv[2])
        raise SystemExit(0)
    MODS = sorted(build(O), key=lambda m: (m.kind, m.name))
    lic = ['LICENSE']
    res = harvestlib.load_compile_results()
    gated = [m for m in MODS if m.compile == 'standalone']
    passed = [m for m in gated if res.get(m.id, {}).get('status') == 'pass']
    gate_line = (f'gate/compile/run.py --origin ds4 on control-node CPU, therock-10.2: {len(passed)} of {len(gated)} standalone modules compile for gfx1151 '
                 '(gate/results/compile/<module>.json); the via: fragments compile inside their backend module' +
                 (f'. {len(gated) - len(passed)} fail: ' + GATE_FAILS if len(passed) < len(gated) else '') if any(m.id in res for m in gated)
                 else 'not run yet')
    toml = ORIGIN_TOML.replace('LICENCE_FILES', harvestlib.tlist(lic)).replace('GATE_LINE', gate_line) + overlay_toml(O)
    run(O, MODS, toml, LEFT_OUT, lic)
    base_diff(O)
    modules_md(O)
