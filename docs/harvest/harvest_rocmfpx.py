#!/usr/bin/env python3
"""Harvest Ciru's ROCmFPX low-bit HIP paths (ciru-ai/ROCmFPX, efd1ef5) into catalog/rocmfpx.
Run: python3 docs/harvest/harvest_rocmfpx.py

Card ORIGIN-ROCMFPX 38c84c0fac. ROCmFPX is a llama.cpp tree: official ggml-org/llama.cpp history up to 15586e2d (tag
b10297 plus one commit, UPSTREAM.md) with Ciru's ROCmFPX layer on top: the GGUF weight types ROCmFP2 (Q2_0_ROCMFPX,
2.5 bpw S40 codebook), ROCmFP3, ROCmFP4 (dual- and single-scale), ROCmFP6, ROCmFP7 (signed Q7, the DualView format) and
ROCmFP8, their CPU reference code, and their HIP paths in ggml-cuda (MMQ, MMVQ, dequantize / copy / get_rows /
set_rows, quantized-KV flash attention). Only that layer is harvested, as one kernel module a family plus a format
module; the unchanged llama.cpp headers the ROCmFPX translation units include ride in kernel.rocmfpx.common so every
family compiles standalone.

What counts as the ROCmFPX layer. The base commit 15586e2d is not on the island (the ark holds ggml-org/llama.cpp only
as depth-1 copies of other commits) and relay-host's llama.cpp clone predates it, so the delta cannot be taken with git.
A file is ROCmFPX material here when it lies in ggml/rocmfpx or ggml/rocmfp4, or its text carries a ROCmFPX marker (MARK:
the ROCmFP type names, the Q7 DualView / panel16 identifiers). check_layer() fails the run when a marked file under
ggml-cuda, ggml-hip, ggml/rocmfpx or ggml/rocmfp4 is neither in a module nor named in LEFT_OUT_MARKED, and when a
family file carries no marker and is not a declared base file. Changed llama.cpp files (mmq.cuh, vecdotq.cuh, mmvq.cu,
...) are taken whole: the ROCmFPX code sits inside them as extra type cases.

The reference checkout is the laptop copy ~/Code/strixflerp-upstream-supplement/rocmfpx (efd1ef5, moved from the
control-node ark copy /data/ark-inbox/src/github.com__ciru-ai__ROCmFPX#efd1ef5a17db576168d849831e1ef3346582881e, a
depth-1 clone whose .git lacks refs/, so git reads it only with a refs directory added in a copy). ROCMFPX_CHECKOUT
names another git copy at the commit. flerp diff-origin reads the ark copy directly. Nothing here compiles or runs;
gate.status = "builds" comes from gate/results/compile (gate/compile/run.py on control-node, CPU only)."""
import os, re, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvestlib
from harvestlib import Origin, M, P, run

CHECKOUT = os.environ.get('ROCMFPX_CHECKOUT', os.path.join(harvestlib.SUPPLEMENT, 'rocmfpx'))
COMMIT = 'efd1ef5a17db576168d849831e1ef3346582881e'
BASE = '15586e2d7165570fb3aa7c26e0d442e289ef69de'
CU = 'ggml/src/ggml-cuda/'
MARK = re.compile(r'rocmfp|q7_0_rocm|dualview|q7-q8|q7_q8|panel16', re.I)
INC = ['src/' + CU.rstrip('/'), 'src/ggml/src', 'src/ggml/include', 'src/ggml/rocmfpx', 'src/ggml/rocmfp4']
INC_ROOTS = [i[len('src/'):] for i in INC]
AUTHORS = ['Ciru (ciru-ai)', 'the ggml authors']
NOT_TIMED = 'No timing was taken on the pod.'
TICK = chr(96)        # a backtick; the markdown below writes it as SECT and swaps it in at the end
SECT = chr(167)


class GitOrigin(Origin):
    """The origin's file set is git's (git ls-files of the checkout), and the checkout must sit on COMMIT, clean."""
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


O = GitOrigin('rocmfpx', 'rocmfpx', 'https://github.com/ciru-ai/ROCmFPX', COMMIT, 'main', 'MIT', AUTHORS, 'therock-10.1')

# The seven GGUF weight types of the layer (ggml/include/ggml.h, ggml/src/ggml.c type traits, ggml/rocmfpx/rocmfpx.h,
# ggml/rocmfp4/rocmfp4.h): key, ggml type, id, values a block, bytes a block, layout, MMQ tile loader, MMVQ dot.
FORMATS = [
 ('fp2', 'Q2_0_ROCMFPX', 108, 32, 10, '2-bit codes from the frozen S40 codebook (four codes a byte decode to an int8 quad) and two UE4M3 half-block scales',
  'load_tiles_rocmfpx_fp2', 'vec_dot_rocmfpx_fp2_q8_1'),
 ('fp3', 'Q3_0_ROCMFPX', 104, 32, 14, '3-bit codes on the levels 0, +-1, +-2, +-4 and two UE4M3 half-block scales chosen by reconstruction MSE',
  'load_tiles_rocmfpx_fp3', 'vec_dot_rocmfpx_fp3_q8_1'),
 ('fp4', 'Q4_0_ROCMFP4', 100, 32, 18, 'packed E2M1-derived 4-bit codes (Codebook10) and two UE4M3 half-block scales',
  'load_tiles_rocmfp4', 'vec_dot_rocmfp4_q8_1'),
 ('fp4_fast', 'Q4_0_ROCMFP4_FAST', 101, 32, 17, 'the same 32 ROCmFP4 nibbles with one UE4M3 scale for the block (the speed layout)',
  'load_tiles_rocmfp4_fast', 'vec_dot_rocmfp4_fast_q8_1'),
 ('fp6', 'Q6_0_ROCMFPX', 102, 32, 26, '6-bit signed-magnitude codes up to 31 and two UE4M3 half-block scales chosen by reconstruction MSE',
  'load_tiles_rocmfpx_fp6', 'vec_dot_rocmfpx_fp6_q8_1'),
 ('fp7', 'Q7_0_ROCMFPX', 107, 256, 240, 'DualView: a 256-value macroblock of eight 32-value groups, 7-bit signed codes (8 x 28 bytes) then eight FP16 group scales; the same integers are exposed to the GPU as an exact signed-Q8 view for prefill',
  'load_tiles_rocmfpx_fp7', 'vec_dot_rocmfpx_q7_q8_1'),
 ('fp8', 'Q8_0_ROCMFPX', 103, 32, 33, 'signed int8 codes clamped to +-127 and one UE4M3 block scale',
  'load_tiles_rocmfpx_fp8', 'vec_dot_rocmfpx_fp8_q8_1'),
]
TYPE_FILE = {'fp2': 'q2_0_rocmfpx', 'fp3': 'q3_0_rocmfpx', 'fp4': 'q4_0_rocmfp4', 'fp4_fast': 'q4_0_rocmfp4_fast',
             'fp6': 'q6_0_rocmfpx', 'fp7': 'q7_0_rocmfpx', 'fp8': 'q8_0_rocmfpx'}
TYPE_NAME = {k: t for k, t, *_ in FORMATS}
FA_TYPES = ['fp3', 'fp4', 'fp4_fast', 'fp6', 'fp8']        # quantized-KV flash-attention instances (K and V the same type)
WF = 'gguf-rocmfpx'


def F(file, what, tag=None, beside=(), base=False):
    """One file of a family (path relative to ggml/src/ggml-cuda). tag: the strixflerp interface it matches (full name;
    the class is the name without its variant tag) or None. beside: catalogue kernel modules of the same <family>.<op>;
    checked to exist and to provide that op. base: an unchanged llama.cpp file kept beside its ROCmFPX neighbours."""
    return dict(file=file, what=what, tag=tag, beside=list(beside), base=base)


MMQ_BESIDE = ['kernel.gufo.ds4_mmq', 'kernel.gufo.qfn_mmq', 'kernel.gufo.qwen_linear_prefill', 'kernel.hipfire.gemm_ggml']
MMQ_MOE_BESIDE = ['kernel.flerp.moe_kgrouped', 'kernel.gufo.ds4_moe']
MMVQ_BESIDE = ['kernel.hipfire.gemv_ggml', 'kernel.flerp.gguf_gemv_hipfire', 'kernel.gufo.qwen_linear_decode', 'kernel.gufo.ds4_linear', 'kernel.gufo.qfn_mmq']
MMVQ_MOE_BESIDE = ['kernel.flerp.moe_kquant', 'kernel.gufo.ds4_moe', 'kernel.gufo.qfn_mmq']

FAMILIES = {
 'mmq': dict(
  summary='ROCmFPX MMQ: ggml\'s tiled integer matrix multiply (MUL_MAT and MUL_MAT_ID for prompt batches) with ROCmFPX tile loaders for '
          'all seven types (ROCmFP2, 3, 4, 4-fast, 6, 7, 8) against q8_1 activations, one template instance a type, the dispatcher, and '
          'the ROCmFP7 DualView fast paths: a direct MMA dot on the Q7 codes, a gfx1151 panel16 weight cache, and the exact signed-Q8 '
          'compute view that sends Q7 prefill through the ordinary Q8_0 MMQ',
  why='Ciru\'s HIP prefill path for the format family the catalogue\'s ladder lacks (2.5 to 8.25 bits a weight, UE4M3 or FP16 scales); the '
      'gguf-kquant MMQ providers kernel.gufo.ds4_mmq / qfn_mmq and kernel.hipfire.gemm_ggml are the same ggml lineage on K-quants',
  files=[
   F('mmq.cuh', 'ggml\'s MMQ kernel templates with the ROCmFPX load_tiles_* loaders, the mmq_type_traits of the seven types, and the Q7 MMA-direct and panel16 dots',
     'linear.prefill/%s/f32.rocmfpx-mmq' % WF, MMQ_BESIDE),
   F('mmq.cu', 'ggml_cuda_mul_mat_q with the ROCmFPX type cases, the MUL_MAT_ID expert path through mmid.cuh, and the Q7 handoff to the panel16 cache or the Q8 view',
     'moe.experts-prefill/%s/f32.rocmfpx-mmq' % WF, MMQ_MOE_BESIDE),
   F('quantize.cuh', 'ggml\'s q8_1 activation quantizer declarations (block_q8_1_mmq comes from mmq.cuh, which it includes); unchanged base file kept beside mmq.cuh', base=True),
   F('q7-view-mode.cuh', 'the Q7 view selector: GGML_ROCM_GFX1151_Q7_PANEL16_CACHE and GGML_ROCM_GFX1151_Q7_Q8_VIEW pick panel16, Q8 view or none'),
   F('q7-panel16.cuh', 'the gfx1151 panel16 device layout of canonical Q7 weights (one panel a [16 rows, 256 K] tile, quartets in the mirrored wave32 order)'),
   F('q7-panel16-cache.cuh', 'declarations of the process-wide panel16 cache and its MMQ instrumentation (defined in ggml-cuda.cu, not harvested)'),
   F('q7-q8-view.cuh', 'the Q7 to standard Q8_0 compute view: sign-extends the stored 7-bit codes and keeps the FP16 group scale, so one Q7 macroblock is eight Q8_0 blocks'),
   F('q7-q8-view-cache.cuh', 'declarations of the Q8-view cache (an allowlist of Qwen MMQ projections; GET_ROWS and MMVQ always read canonical Q7; defined in ggml-cuda.cu, not harvested)'),
   F('template-instances/generate_cu_files.py', 'the generator of the template-instance files, with the ROCmFPX types in its MMQ and FA-vector lists'),
  ] + [F('template-instances/mmq-instance-%s.cu' % TYPE_FILE[k], 'DECL_MMQ_CASE(GGML_TYPE_%s): the MMQ kernels of that type' % t) for k, t, *_ in FORMATS],
  sources=['template-instances/mmq-instance-%s.cu' % TYPE_FILE[k] for k, *_ in FORMATS] + ['mmq.cu'],
  numeric='q8_1 activations (ggml quantize_q8_1); integer dot products of the decoded codes with int8 activations, scaled by the decoded UE4M3 (FP16 for ROCmFP7) weight scale and the q8_1 scale, accumulated in f32; f32 output',
  regime='prompt batches (ggml_cuda_should_use_mmq; the origin build sets GGML_CUDA_FORCE_MMQ), dense and MoE expert matrices; the Q7 panel16 and Q8-view paths are gfx1151-only and opt-in by environment',
 ),
 'mmvq': dict(
  summary='ROCmFPX MMVQ: ggml\'s quantized matrix-vector multiply (MUL_MAT and MUL_MAT_ID for one to a few tokens) with the ROCmFPX type cases, the vec_dot '
          'dispatch of all seven types and the origin\'s ROCmFPX tuning defines (rows per block for MoE, VDR), reading canonical blocks including Q7',
  why='the decode path of the same seven formats; its arithmetic is the per-block dot products of vecdotq.cuh (kernel.rocmfpx.common), which the MMQ and flash-attention '
      'families reuse',
  files=[
   F('mmvq.cu', 'ggml\'s mul_mat_vec_q with the ROCmFPX cases in the vec-dot, VDR and rows-per-block tables and the MUL_MAT_ID expert path (GGML_ROCMFPX_MOE_MMVQ_ROWS_PER_BLOCK and friends are build-time tuning defines)',
     'linear.decode/%s/f32.rocmfpx-mmvq' % WF, MMVQ_BESIDE),
  ],
  own=('moe.experts-decode/%s/f32.rocmfpx-mmvq' % WF, 'mmvq.cu: ggml_cuda_mul_mat_vec_q with ids (MUL_MAT_ID), one expert channel a token', MMVQ_MOE_BESIDE),
  sources=['mmvq.cu'],
  numeric='q8_1 activations; per-block integer dot of the decoded codes (vec_dot_rocmfp*_q8_1 in vecdotq.cuh) times the weight scale and the q8_1 scale, f32 accumulation and output',
  regime='decode and small batches (up to ggml\'s MMVQ column limit), dense and MoE experts',
 ),
 'convert': dict(
  summary='ROCmFPX conversions on the HIP backend: dequantize to f16 / f32 for the BLAS and copy paths, CPY between f32 / f16 and the ROCmFPX types (quantize on the GPU), '
          'GET_ROWS from ROCmFPX tensors (embedding lookup), SET_ROWS into ROCmFPX tensors (quantized KV store) and the standalone ROCmFP4 dequant kernel',
  why='the operations the origin requires before it claims runtime support for a type (ggml/rocmfpx/README.md: CPY, GET_ROWS, SET_ROWS, MUL_MAT, MUL_MAT_ID); '
      'GET_ROWS and SET_ROWS are what an embedding table or a KV cache in a ROCmFPX type needs',
  files=[
   F('convert.cu', 'ggml\'s to-f16 / to-f32 dequantize kernels with the ROCmFPX cases'),
   F('getrows.cu', 'ggml\'s GET_ROWS with the ROCmFPX dequantizers', 'embed.lookup/%s/f32.rocmfpx' % WF, ['kernel.gufo.ds4_linear', 'kernel.gufo.qwen_norm_rope']),
   F('set-rows.cu', 'ggml\'s SET_ROWS with the ROCmFPX quantizers (the store of a quantized KV cache)', 'kvcache.store/%s/f32.rocmfpx' % WF, ['kernel.hipfire.kv_cache']),
   F('cpy.cu', 'ggml\'s CPY with f32 / f16 to ROCmFPX quantization and back'),
   F('cpy-utils.cuh', 'device quantizers of the ROCmFPX types for CPY and SET_ROWS (the scale search on the GPU)'),
   F('../../rocmfp4/rocmfp4_hip.cu', 'standalone ROCmFP4 dequantize kernel for integration tests (rocmfp4_dequantize_q4_0_f32_kernel), linked into ggml-hip'),
  ],
  sources=['convert.cu', 'getrows.cu', 'set-rows.cu', 'cpy.cu', '../../rocmfp4/rocmfp4_hip.cu'],
  numeric='dequantize: integer code times decoded UE4M3 (FP16 for ROCmFP7) scale; quantize: the device scale search of cpy-utils.cuh (the CPU reference in ggml/rocmfpx and ggml/rocmfp4 is the contract)',
  regime='any shape; elementwise and row copies',
 ),
 'fattn_vec': dict(
  summary='ROCmFPX quantized-KV flash attention: ggml\'s vector flash-attention kernel (decode) with K and V in ROCmFP3, ROCmFP4, ROCmFP4-fast, ROCmFP6 or ROCmFP8, '
          'one template instance a type pair, and the shared dequantize / dot helpers of fattn-common.cuh',
  why='a KV cache in the ROCmFPX types is attended without a dequantized copy; the catalogue\'s quantized-KV attention providers are hipfire\'s (q8_0 KV) and radiance\'s (e4m3)',
  files=[
   F('fattn-vec.cuh', 'ggml\'s flash_attn_ext_vec with the ROCmFPX K / V cases', 'attn.softmax-gqa/kv-rocmfpx/f32.rocmfpx-vec', ['kernel.hipfire.attention', 'kernel.gufo.qwen_attention']),
   F('fattn-common.cuh', 'K-dot and V-dequantize helpers of the flash-attention kernels with the ROCmFPX types'),
  ] + [F('template-instances/fattn-vec-instance-%s-%s.cu' % (TYPE_FILE[k], TYPE_FILE[k]), 'DECL_FATTN_VEC_CASE for K = V = %s' % TYPE_NAME[k]) for k in FA_TYPES],
  sources=['template-instances/fattn-vec-instance-%s-%s.cu' % (TYPE_FILE[k], TYPE_FILE[k]) for k in FA_TYPES],
  numeric='f16 Q against dequantized K blocks (or the q8_1 dot path of fattn-common.cuh), online softmax in f32, V dequantized per block; f32 output',
  regime='decode (one or a few query rows a head); the origin compiles these five pairs without GGML_CUDA_FA_ALL_QUANTS (ggml-hip/CMakeLists.txt)',
 ),
}
FAMILY_ORDER = ['mmq', 'mmvq', 'convert', 'fattn_vec']

# Shared ROCmFPX headers (kernel.rocmfpx.common) besides the base closure.
COMMON_ROCMFPX = ['ggml/include/ggml.h', 'ggml/src/ggml-common.h', CU + 'common.cuh', CU + 'vecdotq.cuh', CU + 'dequantize.cuh',
                  'ggml/rocmfpx/rocmfpx.h', 'ggml/rocmfpx/q7-q8-view.h', 'ggml/rocmfpx/rocmfpx_hip_codebook.cuh',
                  'ggml/rocmfp4/rocmfp4.h', 'ggml/rocmfp4/rocmfp4_hip_codebook.cuh', 'ggml/rocmfp4/rocmfp4_hip_scale.cuh']
FORMAT_FILES = ['ggml/rocmfpx/rocmfpx.c', 'ggml/rocmfpx/rocmfp2_reference.c', 'ggml/rocmfpx/rocmfp2_reference.h',
                'ggml/rocmfpx/test_rocmfpx.c', 'ggml/rocmfpx/test_rocmfp2_reference.c', 'ggml/rocmfpx/README.md',
                'ggml/rocmfp4/rocmfp4.c', 'ggml/rocmfp4/README.md']
# Marked files in the HIP backend and the two ROCmFPX directories that are in no module, and why.
LEFT_OUT_MARKED = {
 CU + 'ggml-cuda.cu': 'the backend: supports_op and the MMQ / MMVQ / BLAS choice with the ROCmFPX cases, and the definitions of the Q7 panel16 and Q8-view caches; its include closure is the whole ggml-cuda directory and it builds only as all of libggml-hip',
 CU + 'fattn.cu': 'the flash-attention dispatcher (picks the vector kernel for the ROCmFPX KV types); its closure pulls the tile, MMA and WMMA flash-attention kernels of the base',
 CU + 'CMakeLists.txt': 'build file of the CUDA backend (registers the ROCmFPX template instances)',
 'ggml/src/ggml-hip/CMakeLists.txt': 'build file of the HIP backend: adds ../../rocmfp4/rocmfp4_hip.cu and the five ROCmFPX FA-vector pairs; the compile line it produces is ORIGIN.toml [compile]',
}


def tag_class(tag):
    head, wf, dt = tag.split('/')
    return '/'.join([head, wf, dt.split('.')[0]])


def resolve(frm, name, quoted):
    cands = ([os.path.normpath(os.path.join(os.path.dirname(frm), name))] if quoted else []) + \
        [os.path.normpath(os.path.join(r, name)) for r in INC_ROOTS]
    have = set(O.files())
    return next((c for c in cands if c in have), None)


_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*(["<])([^">]+)[">]', re.M)


def closure(roots):
    """origin files reached from roots through #include lines resolved like the ggml-hip target (beside the file, then INC_ROOTS)."""
    seen, todo = set(), list(roots)
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        for q, name in _INCLUDE.findall(open(O.find(f), errors='replace').read()):
            hit = resolve(f, name, q == '"')
            if hit and hit not in seen:
                todo.append(hit)
    return seen


def marked(path):
    return MARK.search(path) is not None or MARK.search(open(O.find(path), errors='replace').read()) is not None


def family_files(fam):
    return [os.path.normpath(CU + r['file']) for r in FAMILIES[fam]['files']]


def check_layer(declared, base_hdrs):
    """Every marked file of the HIP backend and the ROCmFPX directories is in a module or LEFT_OUT_MARKED, and every
    module file that is not a base file carries a marker. Returns the layer (the marked files of that scope)."""
    scope = [f for f in O.files() if f.startswith((CU, 'ggml/src/ggml-hip/', 'ggml/rocmfpx/', 'ggml/rocmfp4/'))]
    layer = sorted(f for f in scope if f.startswith(('ggml/rocmfpx/', 'ggml/rocmfp4/')) or marked(f))
    missing = [f for f in layer if f not in declared and f not in LEFT_OUT_MARKED]
    if missing:
        raise SystemExit('ROCmFPX files in no module and not in LEFT_OUT_MARKED: ' + ' '.join(missing))
    base_named = {os.path.normpath(CU + r['file']) for fam in FAMILY_ORDER for r in FAMILIES[fam]['files'] if r['base']}
    unmarked = [f for f in declared if f not in base_hdrs and f not in base_named and not marked(f)
                and not f.startswith(('ggml/rocmfpx/', 'ggml/rocmfp4/'))]
    if unmarked:
        raise SystemExit('module files with no ROCmFPX marker that are not base files: ' + ' '.join(unmarked))
    return layer


def providers():
    """{module id: [interfaces]} over the catalogue's other origins."""
    import tomllib
    out = {}
    for d, _, fs in os.walk(os.path.join(harvestlib.REPO, 'catalog')):
        if 'module.toml' in fs and (os.sep + 'rocmfpx' + os.sep) not in (d + os.sep):
            m = tomllib.load(open(os.path.join(d, 'module.toml'), 'rb'))
            out[m['id']] = [p['interface'] for p in m.get('provides', [])]
    return out


def build_mods():
    mods = []
    family = {fam: family_files(fam) for fam in FAMILY_ORDER}
    roots = [f for fam in FAMILY_ORDER for f in family[fam] if f.endswith(('.cu', '.cuh', '.h'))] + COMMON_ROCMFPX
    reach = closure(roots)
    owned = set(f for fs in family.values() for f in fs) | set(COMMON_ROCMFPX) | set(FORMAT_FILES)
    stray = sorted(f for f in reach if f not in owned and marked(f))
    if stray:
        raise SystemExit('ROCmFPX files reached by includes but in no module: ' + ' '.join(stray))
    base_hdrs = sorted(f for f in reach if f not in owned)
    layer = check_layer(sorted(owned | set(base_hdrs)), set(base_hdrs))

    fmt = 'format.device-view/%s/-' % WF
    mods.append(M('kernel.rocmfpx.common',
        'Header-only layer the ROCmFPX kernel units include: the ROCmFPX type ids (ggml.h) and block layouts (ggml-common.h, ggml/rocmfpx/rocmfpx.h, '
        'ggml/rocmfp4/rocmfp4.h), the device codebooks and UE4M3 scale decoders, ggml-cuda\'s common.cuh, dequantize.cuh and vecdotq.cuh with the '
        'per-block ROCmFPX dot products (vec_dot_rocmfp*_q8_1) that MMVQ, MMQ and flash attention share, and the %d llama.cpp headers with no ROCmFPX '
        'marker that those files reach through #include' % len(base_hdrs),
        COMMON_ROCMFPX + base_hdrs,
        [P('kernel-abi.rocmfpx-ggml-cuda/-/-', 'ggml-cuda/common.cuh (ggml_backend_cuda_context, the CUDA-to-HIP vendor mapping), vecdotq.cuh '
           '(vec_dot_rocmfp4_q8_1, vec_dot_rocmfp4_fast_q8_1, vec_dot_rocmfpx_fp2 / fp3 / fp6 / fp8_q8_1, vec_dot_rocmfpx_q7_q8_1)',
           'n/a here', 'compiled inside every kernel.rocmfpx.* unit')],
        why='SPEC.md gives headers that several modules include a module of their own; the vec_dot functions here are the arithmetic core of every '
            'ROCmFPX family, and the base headers (llama.cpp 15586e2d) are taken only so that the families compile standalone; the rest of the base '
            'is not harvested',
        notes='Interface tag: none. ROCmFPX files (carry ROCmFPX code): %s. Base files (no ROCmFPX marker) listed by the include closure that '
              'docs/harvest/harvest_rocmfpx.py computes from the family files: %s.' % (', '.join(COMMON_ROCMFPX), ', '.join(base_hdrs)),
        includes=INC, sources=[], compile='header-only', reason='header-only; compiled inside every other kernel.rocmfpx.* unit',
        authors=AUTHORS,
        upstream='ggml-org/llama.cpp %s (tag b10297 plus one commit): ggml.h, ggml-common.h, common.cuh, dequantize.cuh and vecdotq.cuh are llama.cpp files with ROCmFPX code added; the base headers carry no ROCmFPX marker' % BASE))
    pv = providers()
    for fam in FAMILY_ORDER:
        d = FAMILIES[fam]
        provides, seen = [], set()
        for r in d['files']:
            if r['tag'] and r['tag'] not in seen:
                seen.add(r['tag'])
                provides.append(P(r['tag'], '%s: %s' % (os.path.normpath(CU + r['file']), r['what']), d['numeric'], d['regime']))
        if d.get('own'):
            provides.append(P(d['own'][0], d['own'][1], d['numeric'], d['regime']))
        for tag, beside in [(r['tag'], r['beside']) for r in d['files'] if r['tag']] + ([(d['own'][0], d['own'][2])] if d.get('own') else []):
            op = tag.split('/')[0] + '/'
            for b in beside:
                if b not in pv or not any(i.startswith(op) for i in pv[b]):
                    raise SystemExit(f'{fam}: {b} does not provide {op}')
        tags = '; '.join('%s %s' % (os.path.basename(r['file']), tag_class(r['tag']) if r['tag'] else 'none') for r in d['files']
                         if not r['file'].startswith('template-instances/'))
        base_named = [os.path.basename(r['file']) for r in d['files'] if r['base']]
        notes = ('Interface tags by file: %s; the template instances none (catalog/rocmfpx/KERNELS.md has the table). %s%s' %
                 (tags, ('Unchanged llama.cpp file kept here: %s. ' % ', '.join(base_named)) if base_named else '', NOT_TIMED))
        mods.append(M('kernel.rocmfpx.' + fam, d['summary'], family[fam], provides, why=d['why'], notes=notes,
                      sources=[os.path.normpath(CU + s) for s in d['sources']], includes=INC,
                      requires=['module:kernel.rocmfpx.common', fmt], authors=AUTHORS,
                      upstream='ggml-org/llama.cpp %s: the llama.cpp files of this family carry the ROCmFPX code as extra type cases; the q7-*.cuh, the ROCmFP template instances and rocmfp4_hip.cu are ROCmFPX files' % BASE))
    # module requires from #include lines of .cu / .cuh files too (harvestlib.include_requires reads the C family only)
    owner = {}
    for m in mods:
        for f in O.expand(m.paths, m.exclude):
            owner[f] = m.id
    for m in mods:
        need = []
        for f in O.expand(m.paths, m.exclude):
            if not f.endswith(('.cu', '.cuh', '.h', '.c', '.cpp', '.hpp', '.inc')):
                continue        # generate_cu_files.py holds #include lines as template text
            for q, name in _INCLUDE.findall(open(O.find(f), errors='replace').read()):
                dep = owner.get(resolve(f, name, q == '"'))
                if dep and dep != m.id and 'module:' + dep not in list(m.requires) + need:
                    need.append('module:' + dep)
        m.requires = list(m.requires) + sorted(need)
    mods.append(M('format.rocmfpx.reference',
        'CPU reference of the seven ROCmFPX GGUF weight types: quantize (reconstruction-MSE UE4M3 scale search per 16-weight half block for FP3 / FP6, '
        'the S40 codebook search for FP2, signed Q7 groups with FP16 scales), dequantize and dot for ROCmFP2, 3, 6, 7 and 8 (ggml/rocmfpx) and ROCmFP4 '
        'dual- and single-scale (ggml/rocmfp4), with the origin\'s unit tests and format notes; the block structs live in kernel.rocmfpx.common',
        FORMAT_FILES,
        [P(fmt, 'GGML_TYPE_Q2_0_ROCMFPX = 108, Q3_0 = 104, Q4_0_ROCMFP4 = 100, Q4_0_ROCMFP4_FAST = 101, Q6_0 = 102, Q7_0 = 107, Q8_0 = 103 '
           '(ggml/include/ggml.h); block_rocmfp2 / 3 / 6 / 7 / 8 (ggml/rocmfpx/rocmfpx.h), block_rocmfp4 / block_rocmfp4_fast (ggml/rocmfp4/rocmfp4.h); '
           'the quantize / dequantize / dot routines of rocmfpx.c, rocmfp2_reference.c and rocmfp4.c',
           'value = integer code times decoded scale: finite unsigned UE4M3 scale bytes (0x7f and sign-bit bytes invalid) for FP2 / 3 / 4 / 6 / 8, FP16 group scales for FP7; '
           'bits a weight: FP2 2.5, FP3 3.5, FP4 4.5, FP4-fast 4.25, FP6 6.5, FP7 7.5, FP8 8.25',
           '32-value blocks (FP7: 256-value macroblocks of eight 32-value groups); GGUF tensors of llama.cpp models')],
        why='the format ladder the catalogue lacks: 2.5 to 8.25 bits a weight on one block and scale contract, with a CPU reference that the HIP families are held to',
        notes='Interface tag: none (no catalogue format reads ROCmFPX blocks). ggml.c / ggml-quants.c register the types (type traits, quantize dispatch) '
              'and are left out as base files. Tests: test_rocmfpx.c and test_rocmfp2_reference.c (scripts/check-rocmfpx-reference.sh and '
              'check-rocmfp2-reference.sh build them with cc); not run here.',
        lang='c', sources=[], includes=[], compile='n/a', toolchain='', authors=['Ciru (ciru-ai)'],
        reason='CPU reference compiled into libggml-base by ggml/src/CMakeLists.txt with the rest of ggml; the compile gate builds hip-c++ modules only',
        requires=['module:kernel.rocmfpx.common']))
    return mods, base_hdrs, layer


FLAGS = ["-x", "hip", "-std=gnu++17", "-O3", "-DNDEBUG", "-fPIC", "-funsafe-math-optimizations", "-ffast-math",
         "-DGGML_BACKEND_BUILD", "-DGGML_BACKEND_SHARED", "-DGGML_CUDA_FORCE_MMQ", "-DGGML_HIP_GRAPHS", "-DGGML_HIP_NO_VMM",
         "-DGGML_SCHED_MAX_COPIES=4", "-DGGML_SHARED", "-DGGML_USE_HIP", "-DUSE_PROF_API=1", "-D_GNU_SOURCE", "-D_XOPEN_SOURCE=600",
         "-D__HIP_PLATFORM_AMD__=1", "-D__HIP_ROCclr__=1", "-Dggml_hip_EXPORTS", "-c"]

ORIGIN_TOML = '''
# ROCmFPX by Ciru: llama.cpp with a family of low-bit GGUF weight formats for AMD (ROCmFP2 2.5 bpw S40 codebook, ROCmFP3,
# ROCmFP4, ROCmFP6, ROCmFP7 / DualView, ROCmFP8; UE4M3 or FP16 scales on 32-value blocks), their CPU reference and their
# HIP/ROCm paths (MMQ, MMVQ, conversions, quantized-KV flash attention), plus Strix quality recipes and validation
# scripts. Only the ROCmFPX layer of the HIP backend and the two ROCmFPX directories is harvested (card ORIGIN-ROCMFPX 38c84c0fac).
name      = "rocmfpx"
repo      = "https://github.com/ciru-ai/ROCmFPX"
commit    = "efd1ef5a17db576168d849831e1ef3346582881e"
branch    = "main"
version   = "main HEAD of 2026-09-28 ('Remove unused scheduled Actions jobs'), tag kairic-edge-qwen38-27b-v1.3 (ark manifest row); GitHub fork of charlie12345/ROCmFPX, whose ark copy (fb08d7cd, 2026-09-23) is not used"
base      = "ggml-org/llama.cpp 15586e2d7165570fb3aa7c26e0d442e289ef69de (tag b10297 plus one commit, UPSTREAM.md); not on the island, so the ROCmFPX layer is told apart by ROCmFPX markers in the files (docs/harvest/harvest_rocmfpx.py)"
reference_checkout = "control-node:/data/ark-inbox/src/github.com__ciru-ai__ROCmFPX#efd1ef5a17db576168d849831e1ef3346582881e (ark copy, ark 7546d7e, a depth-1 clone whose .git lacks refs/); laptop: ~/Code/strixflerp-upstream-supplement/rocmfpx (the same commit: the ark copy's .git with refs/ added, checked out)"

[licence]
spdx      = "MIT"
files     = ["LICENSE"]                                  # copied verbatim next to this file
third_party = "LICENSE is llama.cpp's MIT licence (Copyright 2023-2026 The ggml authors) and covers the whole tree; README.md and docs/UPSTREAM-ATTRIBUTION.md credit Ciru for the ROCmFPX work and keep upstream authorship; docs/ROCmFPX-UPSTREAM-CREDITS.md names the TurboQuant KV work (Tom Turney), not harvested; licenses/LICENSE-jsonhpp belongs to vendored JSON code outside the harvest. Weights are not part of the tree"

[toolchain]
host      = "C++17, CMake + Ninja (scripts/build-strix-rocmfp4-mtp.sh: GGML_HIP=ON, GGML_HIP_FORCE_MMQ=ON, GGML_HIP_ROCWMMA_FATTN=OFF, CMAKE_HIP_ARCHITECTURES=gfx1151, Release)"
device    = "therock-10.1"                               # README.md: validated with a pinned TheRock ROCm 7.15 development toolchain on Radeon 8060S (gfx1151); the pod's gate uses TheRock 10.1
build     = "ggml/src/ggml-hip/CMakeLists.txt compiles every ggml-cuda/*.cu, the mmq-instance files, five ROCmFPX FA-vector pairs and ../../rocmfp4/rocmfp4_hip.cu as HIP into libggml-hip; ggml/rocmfpx/rocmfpx.c joins libggml-base (ggml/src/CMakeLists.txt)"

[compile]                                                # CMake's HIP language: clang++ -x hip, one object per .cu
driver    = "<rocm>/lib/llvm/bin/clang++"
arch_flag = "--offload-arch=gfx1151"
rocm_flag = "--rocm-path=<rocm>"
flags     = FLAGS
system_includes = ["include"]
output    = "host object with the gfx1151 device code embedded (.o), one per translation unit"
note      = "flags are the ggml-hip target's compile line from compile_commands.json of a CMake configure (no build) of the ark copy on control-node with scripts/build-strix-rocmfp4-mtp.sh's options (Vulkan, server and tests off), TheRock 10.1 (/data/pod/plans/38c84c0fac/cmake-probe/compile_commands.json). The origin's include roots are ggml/src and ggml/include; the modules add ggml/src/ggml-cuda, ggml/rocmfpx and ggml/rocmfp4 because the catalogue splits one directory over several modules"

[kernels]
loading   = "statically linked into libggml-hip: each .cu translation unit carries its kernels and host launchers; no code-object kit or kernel registry"
selection = "ggml-cuda.cu (not harvested) picks MMQ for prompt batches and MMVQ for decode by type and batch (GGML_CUDA_FORCE_MMQ in the origin build); ROCmFP7 prefill goes through the Q8 view or the panel16 cache when GGML_ROCM_GFX1151_Q7_Q8_VIEW or GGML_ROCM_GFX1151_Q7_PANEL16_CACHE is set (q7-view-mode.cuh)"
models    = "llama.cpp GGUF models quantized to ROCmFPX types (tools/quantize presets, docs/recipes: Qwen3.5 MoE, Step 3.5, Qwen3.8-27B ActiveFPX, Ornith 35B DualView)"
'''.replace('flags     = FLAGS', 'flags     = ' + harvestlib.tlist(FLAGS))

LEFT_OUT = {
 'ggml/ (everything but the files in the modules)': 'the base engine, llama.cpp 15586e2d; kernel.rocmfpx.common carries the base headers the ROCmFPX units include',
 'ggml/src/ggml-cuda/ggml-cuda.cu, fattn.cu, CMakeLists.txt, ggml/src/ggml-hip/CMakeLists.txt': 'ROCmFPX-marked files of the HIP backend left out: ' + '; '.join('%s: %s' % (k, v) for k, v in LEFT_OUT_MARKED.items()),
 'ggml/src/ggml.c, ggml-quants.c, ggml-cpu/, ggml/src/CMakeLists.txt': 'the type registration and CPU backend cases of the ROCmFPX types inside the base C library (type traits, quantize dispatch, CPU ops)',
 'ggml/src/ggml-vulkan/': 'the Vulkan backend with ROCmFPX dequant / matvec / MMQ shaders (FP2 to FP8 and ROCmFP4): not the HIP path',
 'ggml/src/ggml-hip/fattn-kv-batched.cu, fattn-vec-turbo-mixed.cu and the turbo3 / turbo4 instances': 'TurboQuant KV-cache quantization integrated from another contributor (docs/ROCmFPX-UPSTREAM-CREDITS.md), not a ROCmFPX format',
 'src/, include/, common/, tools/ (quantize presets), tests/, gguf-py/': 'libllama, the quantize tool with the ROCmFPX presets and tensor routing, backend-op and Q7 view tests, the Python GGUF readers: runtime and tooling, not kernels',
 'scripts/, docs/ (recipes, experiment notes), .github/, .devops/, app/, skills/, vendor/, the rest of the tree': 'validation, sweep, quantization and serving scripts, Strix quality recipes and research notes, CI, the llama.cpp front ends and vendored code',
}


def kernels_md(base_hdrs, layer):
    o = ['# rocmfpx kernels: the ROCmFPX layer of ggml-cuda by family, with the strixflerp interface each file matches', '',
         'Generated by §docs/harvest/harvest_rocmfpx.py§; edit the script, not this file. Origin ciru-ai/ROCmFPX §%s§ on llama.cpp §%s§.' % (COMMIT[:12], BASE[:12]), '',
         'A tag is a full interface name; the class it matches is the name without its variant tag. "beside" lists catalogue kernel modules',
         'of the same §<family>.<op>§; they read other weight formats (no catalogue module reads ROCmFPX blocks). Template instances and helper',
         'headers carry no tag.', '',
         '## The seven weight types', '',
         '| type | ggml id | values / block | bytes / block | bits / weight | MMQ instance | MMQ tile loader | MMVQ dot | FA-vector KV | layout |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for k, t, tid, n, b, what, lt, vd in FORMATS:
        o.append('| §%s§ | %d | %d | %d | %.2f | §mmq-instance-%s.cu§ | §%s§ | §%s§ | %s | %s |' % (
            t, tid, n, b, 8.0 * b / n, TYPE_FILE[k], lt, vd, 'yes' if k in FA_TYPES else 'no', what))
    o += ['', '## Files by family', '', '| file | lines | module | tag (interface) | class | beside |', '|---|---|---|---|---|---|']
    count = 0
    for fam in FAMILY_ORDER:
        for r in FAMILIES[fam]['files']:
            rel = os.path.normpath(CU + r['file'])
            lines = len(open(O.find(rel), errors='replace').read().splitlines())
            count += 1
            o.append('| §%s§ | %d | §kernel.rocmfpx.%s§ | %s | %s | %s |' % (
                rel, lines, fam, '§%s§' % r['tag'] if r['tag'] else 'none', '§%s§' % tag_class(r['tag']) if r['tag'] else '-',
                ', '.join('§%s§' % b for b in r['beside']) or '-'))
        own = FAMILIES[fam].get('own')
        if own:
            o.append('| (mmvq.cu, MUL_MAT_ID) | | §kernel.rocmfpx.%s§ | §%s§ | §%s§ | %s |' % (fam, own[0], tag_class(own[0]), ', '.join('§%s§' % b for b in own[2])))
    o += ['', '%d family files; kernel.rocmfpx.common holds %d ROCmFPX headers and %d llama.cpp headers with no ROCmFPX marker; format.rocmfpx.reference holds %d files.' %
          (count, len(COMMON_ROCMFPX), len(base_hdrs), len(FORMAT_FILES)),
          '%d files of ggml-cuda, ggml-hip, ggml/rocmfpx and ggml/rocmfp4 are ROCmFPX material (in ggml/rocmfpx or ggml/rocmfp4, or carrying a ROCmFPX marker); all are in a module except %s.' %
          (len(layer), ', '.join('§%s§' % k for k in LEFT_OUT_MARKED)), '',
          '## What each file does', '']
    for fam in FAMILY_ORDER:
        for r in FAMILIES[fam]['files']:
            o.append('- §%s§: %s.' % (os.path.normpath(CU + r['file']), r['what']))
    o += ['', '## Not done here', '',
          'No kernel was run or timed. A comparison against the catalogue\'s GGUF providers needs weights in a ROCmFPX type (or a converter into',
          'ROCmFPX blocks) and an adapter from ggml\'s tensor launch signature; both are their own cards.']
    open(os.path.join(harvestlib.REPO, 'catalog', O.name, 'KERNELS.md'), 'w').write('\n'.join(o).replace(SECT, TICK) + '\n')


if __name__ == '__main__':
    mods, base_hdrs, layer = build_mods()
    run(O, mods, ORIGIN_TOML, LEFT_OUT, ['LICENSE'])
    kernels_md(base_hdrs, layer)
