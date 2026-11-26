#!/usr/bin/env python3
"""Harvest fastasfuck (Cheblan/fastasfuck, release 1.1.0 = 73bd170) into catalog/fastasfuck.
Run: python3 docs/harvest/harvest_fastasfuck.py        (--tree prints the checkout's tree hash and writes nothing;
     --delta OLD_CHECKOUT rewrites docs/harvest/data/fastasfuck/delta-35015b5.tsv from a copy of the 35015b5 tree)

Card ORIGIN-FASTASFUCK 80ae224db4 harvested main 35015b5 (the kernels); card FASTASFUCK-110 17eead1048 moves the pin to
the 1.1.0 tag and harvests the whole engine, one module a unit. Scope: the 25 HIP kernel files of kernels/ as one module
a kernel family (20 families), the shared device and host headers they include (kernel.fastasfuck.common) and the Q4 /
Q5 / Q6 / Q8 host formats with their device views (format.fastasfuck.qn_device); then the host engine split the way
catalog/strixite splits the same files: common/ helpers, formats/ (quantizers, .faf, the converter), runtime/ (loaders,
PLE, paged buffers, the qwen4exp forward), serve/ (engine, grammar, tokenizer, OpenAI and Responses APIs, server), and
every program and script of tools/ as its own tool module. The vendored third-party code, deploy/ and docs/ are listed
as unharvested. docs/harvest/data/fastasfuck/delta-35015b5.tsv lists every file 1.1.0 added or changed against 35015b5;
delta-35015b5.md maps them onto modules.

fastasfuck is a fork of strixite (AGENTS.md "Why We Forked": upstream is the private strix-infer that strixite publishes;
CMakeLists.txt still names the project strix-infer and the library strix_kernels). Its kernels are strixite's kernels:
for every kernel file the script names the catalog/strixite module that holds the same kernel and counts the lines
that differ from it (difflib, against catalog/strixite at the commit its ORIGIN.toml pins). Each file is tagged with
the interface classes that strixite module provides, and the script fails when that module no longer provides one of
them. Host modules carry their strixite counterpart's interfaces when they do the same job, their own where fastasfuck
added or changed it (.faf, Responses API, MTP calibration, vocabulary filter, the 1.1.0 --prebuilt import). The fork
point is not known here: a difference can be fastasfuck's change or a strixite change after the fork.

fastasfuck compiles every kernel file as its own translation unit with the ROCm clang (no hipcc, no CMake HIP
language: CMakeLists.txt), so each family module is [gate].compile = "standalone" and gate/compile/run.py
compiles it with ORIGIN.toml [compile]. Host C++ is plain C++ for that clang (no -x hip), which the compile gate does
not build: those modules are n/a (header-only for headers). The reference copy is the ark copy on control-node
(/data/ark-inbox/src/github.com__Cheblan__fastasfuck#73bd1703..., a depth-1 clone); the laptop holds the same files in
~/Code/strixflerp-upstream-supplement/fastasfuck-1.1.0 (that directory without .git, by tar over ssh; the 35015b5
copy stays in .../fastasfuck for --delta). The script reads FASTASFUCK_CHECKOUT when set, else the laptop copy when it
exists, else the ark copy (so it runs as is on control-node); .git is left out either way. The tree hash in ORIGIN.toml
is checked before anything is written.

Each host module's [gate] reason names the CMake target(s) that compile its .cpp files, read from the checkout's
CMakeLists.txt (add_library / add_executable / target_sources); a .cpp in no target stops the script."""
import difflib, hashlib, os, re, sys, tomllib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvestlib
from harvestlib import Origin, M, P, run

COMMIT = '73bd1703cbe04f1556b17a3c36bd05fd7ee0372e'       # tag 1.1.0
LAPTOP_COPY = os.path.join(harvestlib.SUPPLEMENT, 'fastasfuck-1.1.0')
ARK_COPY = '/data/ark-inbox/src/github.com__Cheblan__fastasfuck#' + COMMIT     # control-node
CHECKOUT = os.environ.get('FASTASFUCK_CHECKOUT') or (LAPTOP_COPY if os.path.isdir(LAPTOP_COPY) else ARK_COPY)
if not os.path.isdir(CHECKOUT):
    raise SystemExit(f'fastasfuck checkout not found: set FASTASFUCK_CHECKOUT, or provide {LAPTOP_COPY} or {ARK_COPY}')
# sha256 of the sha256sum listing of every file (paths relative to the checkout, .git left out, byte order):
#   cd <checkout> && find . -type f -not -path './.git/*' -printf '%P\n' | LC_ALL=C sort | xargs -d '\n' sha256sum | sha256sum
TREE_SHA256 = '1cacae21215aa376152480386e39a6e9940bc9e8bc1ad25a5d224956243146f0'
OLD_TREE_SHA256 = '819cae8b82bc20e4239b8748aa20ad314ac3a23c36a04364da9edcfa55e3f561'   # the 35015b5 tree, for --delta


def tree_sha256(files):
    listing = ''.join('%s  %s\n' % (harvestlib.sha256(os.path.join(CHECKOUT, f)), f) for f in files)
    return hashlib.sha256(listing.encode()).hexdigest()


class DirOrigin(Origin):
    """The origin's file set is the checkout directory without .git; its listing hash must be TREE_SHA256."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.roots = [CHECKOUT]
        if '--tree' not in sys.argv:
            got = tree_sha256(self.files())
            if got != TREE_SHA256:
                raise SystemExit(f'{CHECKOUT}: tree sha256 {got}, the harvest pins {TREE_SHA256}')

    def files(self):
        if self._files is None:
            out = []
            for d, ds, fs in os.walk(CHECKOUT):
                ds[:] = [x for x in ds if not (d == CHECKOUT and x == '.git')]
                out += [os.path.relpath(os.path.join(d, f), CHECKOUT) for f in fs]
            self._files = sorted(out, key=lambda s: s.encode())
        return self._files


O = DirOrigin('fastasfuck', 'fastasfuck', 'https://github.com/Cheblan/fastasfuck', COMMIT, '1.1.0', 'AGPL-3.0-only', ['Cheblan'], 'therock-10.1')
STRIXITE = os.path.join(harvestlib.REPO, 'catalog', 'strixite')
STRIXITE_COMMIT = tomllib.load(open(os.path.join(STRIXITE, 'ORIGIN.toml'), 'rb'))['commit'][:7]
NOT_TIMED = 'Nothing was run or timed on the pod; the origin\'s own figures (README.md) are unverified here.'


def K(file, cpt, cfile, ifaces, what):
    """One kernel or header file. cpt: the catalog/strixite module holding the same kernel; cfile: its file there
    (the same path unless renamed); ifaces: the interfaces of cpt this file provides (checked against cpt)."""
    return dict(file=file, cpt=cpt, cfile=cfile or file, ifaces=ifaces, what=what)


def BF(c):
    return [c + '/bf16', c + '/f32']


FAMILIES = {
 'attention_gqa': [
  K('kernels/attention_gqa.hip', 'kernel.strixite.attention', 'kernels/attention.hip', BF('attn.softmax-gqa/-'),
    'full-attention layers: fused q / k zero-centred RMSNorm plus partial RoPE jobs that also write the K/V cache (qk_norm_rope, qk_norm_rope_jobs), then FP32 online-softmax GQA attention over the cache in scalar and WMMA forms, and gathered forms over a paged K/V (attention_gathered, attention_gathered_wmma), on a BF16 or 8-bit (kv8) cache'),
  K('kernels/attention_gqa.hpp', 'kernel.strixite.attention', 'kernels/attention.hpp', [], 'declarations and workspace sizes')],
 'embedding': [
  K('kernels/embedding.hip', 'kernel.strixite.embedding', None, BF('embed.lookup/bf16') + BF('embed.lookup/strixw-q8'),
    'token embedding gather from a BF16 or Q8 table into F32 or BF16 rows; an out-of-range id zeroes its row and records the first bad (position, id) in a device error word (embedding_lookup, check / report / reset_embedding_error)'),
  K('kernels/embedding.hpp', 'kernel.strixite.embedding', None, [], 'declarations')],
 'gdn': [
  K('kernels/gdn.hip', 'kernel.strixite.gdn', None, BF('mixer.gdn/-'),
    'Gated DeltaNet mixer in two launches: gdn_front (width-4 causal depthwise conv plus silu over the merged in_proj rows, beta and g gates) and delta_rule (the recurrent state update and the output rows)'),
  K('kernels/gdn.hpp', 'kernel.strixite.gdn', None, [], 'declarations')],
 'hc': [
  K('kernels/hc.hip', 'kernel.strixite.hc', None, BF('hc.mix/strixw-q4') + BF('hc.mix/strixw-q8'),
    'per-token hyper-connection mixes over the 4 residual streams on Q4 / Q8 weights (hc_mix_down, hc_mix_up, hc_inject); in this file hc_mix_up is instantiated for 1 to 8 tokens a block sharing each weight load (file comment)'),
  K('kernels/hc.hpp', 'kernel.strixite.hc', None, [], 'declarations')],
 'hc_wmma': [
  K('kernels/hc_wmma.hip', 'kernel.strixite.hc_wmma', None, BF('hc.mix-prefill/strixw-q4') + BF('hc.mix-prefill/strixw-q8'),
    'multi-token hyper-connection down / up mixes on the WMMA units (hc_mix_down_wmma, hc_mix_up_wmma, hc_inject_inv); the file comments describe a fused 4-stream hc_down with no DRAM partials, a 128-row up tile and loads two K steps ahead'),
  K('kernels/hc_wmma.hpp', 'kernel.strixite.hc_wmma', None, [], 'declarations and workspace size')],
 'kv8': [
  K('kernels/kv8.hip', 'kernel.strixite.kv8', None, BF('kvcache.quant8/-'),
    'the optional 8-bit K/V cache of the full-attention layers: kv8_commit quantizes each completed 32-row group (K per channel, V per 32-channel block), kv8_dequant reads it back'),
  K('kernels/kv8.hpp', 'kernel.strixite.kv8', None, [], 'the Kv8View layout and sizes'),
  K('kernels/kv8_math.hpp', 'kernel.strixite.kv8', None, [], 'device quantize / dequantize helpers')],
 'linear': [
  K('kernels/linear.hip', 'kernel.strixite.linear', None, BF('linear.dense/bf16'), 'plain BF16-weight matmul y = x W^T with FP32 accumulation (linear_bf16w)'),
  K('kernels/linear_q4.hip', 'kernel.strixite.linear_q4', None, BF('linear.decode/strixw-q4') + BF('moe.experts-decode/strixw-q4') + BF('moe.experts-decode/strixw-q5'),
    'Q4 group-affine decode GEMV (linear_q4) plus the per-token MoE expert gather, fused gather + SwiGLU and combine for Q4 and Q5 experts; the file adds exact short-M unrolling for M = 1-11 (MTP verify, file comment)'),
  K('kernels/linear_q5.hip', 'kernel.strixite.linear_q56', 'kernels/linear_q56.hip', BF('linear.decode/strixw-q5'),
    'Q5 decode GEMV (linear_q5): low Q4 plane plus a 1-bit high plane; strixite builds Q5 and Q6 from one template file, linear_q56.hip'),
  K('kernels/linear_q6.hip', 'kernel.strixite.linear_q56', 'kernels/linear_q56.hip', BF('linear.decode/strixw-q6'),
    'Q6 decode GEMV (linear_q6): low Q4 plane plus a 2-bit high plane, the other half of strixite\'s linear_q56.hip'),
  K('kernels/linear_q8.hip', 'kernel.strixite.linear_q8', None, BF('linear.decode/strixw-q8'),
    'Q8 decode GEMV (linear_q8) with BF16 activations through v_dot2; exact short-M unrolling for M = 1-11 (file comment)'),
  K('kernels/linear_wmma.hip', 'kernel.strixite.linear_wmma', None,
    BF('linear.prefill/strixw-q4') + BF('linear.prefill/strixw-q5') + BF('linear.prefill/strixw-q6') + BF('linear.prefill/strixw-q8') + BF('linear.prefill/bf16'),
    'prefill matmul on the WMMA units for Q4 / Q5 / Q6 / Q8 and BF16 weights (linear_q4_wmma ... linear_bf16w_wmma); the file comments: Q4 / Q6 / Q8 staged as F16, Q5 as BF16 scaled codes, and a 64-row tile for Q5 and Q6 because 128 rows spill past 256 VGPRs'),
  K('kernels/linear.hpp', 'kernel.strixite.linear', None, [], 'declarations'),
  K('kernels/linear_q4.hpp', 'kernel.strixite.linear_q4', None, [], 'declarations'),
  K('kernels/linear_q5.hpp', 'kernel.strixite.linear_q56', None, [], 'declarations'),
  K('kernels/linear_q6.hpp', 'kernel.strixite.linear_q56', None, [], 'declarations'),
  K('kernels/linear_q8.hpp', 'kernel.strixite.linear_q8', None, [], 'declarations'),
  K('kernels/linear_wmma.hpp', 'kernel.strixite.linear_wmma', None, [], 'declarations')],
 'logits_bias': [
  K('kernels/logits_bias.hip', 'kernel.strixite.logits_bias', None, ['logits.bias/-/-'],
    'adds a request\'s OpenAI logit_bias entries to its rows of the device FP32 logits, in place, before the top-k pick'),
  K('kernels/logits_bias.hpp', 'kernel.strixite.logits_bias', None, [], 'declarations')],
 'logits_topk': [
  K('kernels/logits_topk.hip', 'kernel.strixite.logits_topk', None, ['logits.topk/-/-'],
    'top-20 candidates of each logits row by rank counting, two levels of 8,192 inputs a workgroup (logits_topk)'),
  K('kernels/logits_topk.hpp', 'kernel.strixite.logits_topk', None, [], 'declarations and workspace size')],
 'moe_grouped': [
  K('kernels/moe_grouped.hip', 'kernel.strixite.moe_grouped', None, BF('moe.experts-prefill/strixw-q4') + BF('moe.experts-prefill/strixw-q5'),
    'MoE experts for multi-token forwards: routes bucketed by expert once a layer (moe_group_routes), then grouped gather, gather + SwiGLU and combine over Q4 / Q5 expert rows; the file comments add 16-byte stores, BF16 partials through LDS and a fused combine-reduce + hc_inject_inv for prefill'),
  K('kernels/moe_grouped.hpp', 'kernel.strixite.moe_grouped', None, [], 'declarations and workspace size')],
 'moe_router': [
  K('kernels/moe_router.hip', 'kernel.strixite.moe_router', None, BF('moe.router/-'),
    'per-token top-k router in registers with renormalized weights and the shared-expert slot (moe_router, moe_shared_add, moe_shared_add_inject)'),
  K('kernels/moe_router.hpp', 'kernel.strixite.moe_router', None, [], 'declarations')],
 'mtp_pick': [
  K('kernels/mtp_pick.hip', 'kernel.strixite.mtp_pick', None, ['draft.mtp-pick/-/-'],
    'one-block reduction over the draft head\'s logits: argmax, runner-up value and a NaN flag, packed with the forward\'s error words (mtp_pick)'),
  K('kernels/mtp_pick.hpp', 'kernel.strixite.mtp_pick', None, [], 'declarations')],
 'norm': [
  K('kernels/norm.hip', 'kernel.strixite.norm', None, BF('norm.rmsnorm-zc/-') + BF('norm.rmsnorm-gated/-'),
    'zero-centred RMSNorm, its grouped form and the gated RMSNorm with a silu or sigmoid gate (rmsnorm_zc, group_rmsnorm_zc, rmsnorm_gated)'),
  K('kernels/norm.hpp', 'kernel.strixite.norm', None, [], 'declarations')],
 'ple': [
  K('kernels/ple.hip', 'kernel.strixite.ple', None, BF('ple.inject/-'),
    'PLE injection in two launches: ple_gate (signed-sqrt sigmoid gate per token and stream) and ple_conv (dilated causal depthwise conv)'),
  K('kernels/ple.hpp', 'kernel.strixite.ple', None, [], 'declarations')],
 'qsa_indexer': [
  K('kernels/qsa_indexer.hip', 'kernel.strixite.qsa', 'kernels/qsa.hip', BF('attn.qsa-indexer/-'),
    'QSA indexer of the full-attention layers: block keys (qsa_block_keys), relu-dot scores summed over heads (qsa_scores) and the top-k block pick (qsa_topk), with an explicit cooperative-launch path that Auto never picks (file comment)'),
  K('kernels/qsa_indexer.hpp', 'kernel.strixite.qsa', 'kernels/qsa.hpp', [], 'declarations and workspace size')],
 'residual': [
  K('kernels/residual.hip', 'kernel.strixite.residual', None, BF('residual.add/-'), 'element-wise in-place x += y with an FP32 add and one rounding (residual_add)'),
  K('kernels/residual.hpp', 'kernel.strixite.residual', None, [], 'declarations')],
 'rope': [
  K('kernels/rope.hip', 'kernel.strixite.rope', None, BF('pos.rope/-'),
    'partial rotate-half RoPE in place inside the merged qkv rows, YaRN blending and interleaved mRoPE positions (rope)'),
  K('kernels/rope.hpp', 'kernel.strixite.rope', None, [], 'declarations and MropePos'),
  K('kernels/rope_math.hpp', 'kernel.strixite.rope_math', None, ['pos.rope-math/-/-'], 'device inline cos / sin and pair rotation helpers')],
 'swiglu': [
  K('kernels/swiglu.hip', 'kernel.strixite.swiglu', None, BF('act.swiglu/-'), 'SwiGLU h = silu(gate) * up over [gate | up] rows (swiglu)'),
  K('kernels/swiglu.hpp', 'kernel.strixite.swiglu', None, [], 'declarations')],
 'vision': [
  K('kernels/vision.hip', 'kernel.strixite.vision', None, BF('vision.tower-ops/bf16'),
    'the vision tower\'s kernels: bias-add linear, LayerNorm, GELU, position add, axial 2D RoPE on q / k, attention, embedding scatter and merge regroup (vision_*)'),
  K('kernels/vision.hpp', 'kernel.strixite.vision', None, [], 'declarations')],
 'wmma': [
  K('kernels/wmma.hip', 'kernel.strixite.wmma', None, ['gemm.wmma-tile/-/bf16'], 'the host entry wmma_tile_product that checks the tile helpers on one tile'),
  K('kernels/wmma.hpp', 'kernel.strixite.wmma', None, [], 'wave32 WMMA tile helpers and fragment loaders'),
  K('kernels/wmma_gemm.hpp', 'kernel.strixite.wmma_gemm', None, ['gemm.wmma-blocks/-/bf16'],
    'device building blocks of the tiled WMMA GEMMs (LDS staging, software pipelining, Q4 / Q6 / Q8 code unpacking)')],
}
COMMON_FILES = [
  K('common/hip_runtime.hpp', 'kernel.strixite.hip_common', None, [], 'the single include point for hip_runtime.h'),
  K('common/hip_check.hpp', 'kernel.strixite.hip_common', None, ['device.hip-check/-/-'], 'STRIX_HIP_CHECK'),
  K('common/check.hpp', 'runtime.strixite.check', None, ['host.check/-/-'], 'host checks that throw with context'),
  K('kernels/device_info.hpp', 'kernel.strixite.device_info', None, ['device.info/-/-'], 'cached wavefront size and multiprocessor count'),
  K('kernels/qsplit.hpp', 'kernel.strixite.qsplit', None, ['linear.row-layout/strixw-q4/-'], 'the split low / high plane expert row layout'),
  K('kernels/types.hpp', None, None, [], 'shared activation type helpers; strixite has no such file'),
  K('kernels/math_utils.hpp', None, None, [], 'shared device math helpers (BF16 conversions among them); strixite has no such file'),
]
FORMAT_FILES = [K(f'formats/q{b}.hpp', f'format.strixite.q{b}', None, [f'format.host/strixw-q{b}/-'], f'host Q{b} weight') for b in (4, 5, 6, 8)] + \
    [K(f'runtime/q{b}_device.hpp', f'format.strixite.q{b}_device', None, [f'format.device-view/strixw-q{b}/-'], f'Q{b} device view') for b in (4, 5, 6, 8)] + \
    [K('runtime/device_buffer.hpp', 'runtime.strixite.device_buffer', None, ['device.buffer/-/-'], 'owning device buffer')]

SUMMARY = {
 'attention_gqa': 'fastasfuck\'s full-attention kernels (strixite\'s attention.hip renamed): fused q / k RMSNorm + partial RoPE + K/V cache write, then FP32 online-softmax GQA attention over a BF16 or 8-bit cache, scalar, WMMA and gathered (paged) forms',
 'embedding': 'fastasfuck\'s token embedding gather from a BF16 or Q8 table into F32 or BF16 rows with an out-of-range id error word',
 'gdn': 'fastasfuck\'s Gated DeltaNet mixer: the conv + gates front end and the delta-rule recurrence',
 'hc': 'fastasfuck\'s per-token hyper-connection down / up / inject mixes over 4 residual streams on Q4 / Q8 weights',
 'hc_wmma': 'fastasfuck\'s multi-token hyper-connection mixes on the WMMA units, with a fused 4-stream down mix',
 'kv8': 'fastasfuck\'s 8-bit K/V cache: group quantize on commit and dequantize, with the shared device math',
 'linear': 'fastasfuck\'s dense linear kernels: BF16-weight matmul, Q4 / Q5 / Q6 / Q8 decode GEMVs (with the per-token MoE expert gather and combine in linear_q4.hip) and the WMMA prefill matmul for all five weight types',
 'logits_bias': 'fastasfuck\'s OpenAI logit_bias applied in place to the device logits',
 'logits_topk': 'fastasfuck\'s top-20 candidates per logits row by rank counting',
 'moe_grouped': 'fastasfuck\'s grouped MoE expert kernels for multi-token forwards on Q4 / Q5 expert rows, with a fused combine-reduce + hc inject for prefill',
 'moe_router': 'fastasfuck\'s MoE router: top-k, renormalized weights and the shared-expert slot',
 'mtp_pick': 'fastasfuck\'s draft-head argmax, runner-up and NaN flag packed with the forward\'s error words',
 'norm': 'fastasfuck\'s RMSNorm kernels: zero-centred, grouped and gated',
 'ple': 'fastasfuck\'s PLE injection: gate and dilated causal conv',
 'qsa_indexer': 'fastasfuck\'s QSA indexer (strixite\'s qsa.hip renamed): block keys, scores and top-k block pick',
 'residual': 'fastasfuck\'s element-wise in-place residual add x += y on F32 or BF16 rows, an FP32 add with one rounding to the activation dtype',
 'rope': 'fastasfuck\'s partial RoPE with YaRN and interleaved mRoPE, and its device math header',
 'swiglu': 'fastasfuck\'s SwiGLU over [gate | up] rows',
 'vision': 'fastasfuck\'s vision tower kernels and its bias-add linear',
 'wmma': 'fastasfuck\'s wave32 WMMA tile helpers, the tile-check entry point and the tiled-GEMM building blocks',
}
WHY_FAMILY = ('owner asked for fastasfuck as an origin (2026-11-24): an engine that serves Qwen3.8-Flash-Next on one Strix Halo from '
              'a fork of strixite\'s kernels; this module holds its %s so a timing card can set it against %s, the strixite code it was forked from')

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


def counterpart(k):
    """(lines that differ from the strixite file, lines of the fastasfuck file); None when strixite has no such file."""
    if not k['cpt']:
        return None
    kind, _, name = k['cpt'].split('.', 2)
    path = os.path.join(STRIXITE, kind, name, 'src', k['cfile'])
    if not os.path.isfile(path):
        raise SystemExit(f"{k['file']}: {k['cpt']} has no {k['cfile']}")
    a = open(O.find(k['file'])).read().splitlines()
    b = open(path).read().splitlines()
    return sum(1 for l in difflib.unified_diff(a, b, lineterm='', n=0) if l[:1] in '+-' and l[:3] not in ('+++', '---')), len(a)


_STRIX = {}


def strixite_provides(mid):
    if mid not in _STRIX:
        kind, _, name = mid.split('.', 2)
        t = tomllib.load(open(os.path.join(STRIXITE, kind, name, 'module.toml'), 'rb'))
        _STRIX[mid] = {p['interface']: p for p in t.get('provides', [])}
    return _STRIX[mid]


def entries(hpp):
    """Host entry points declared in a header: names of top-level function declarations."""
    text = open(O.find(hpp)).read()
    return sorted(set(re.findall(r'^\s*(?:void|hipError_t|int|bool|size_t|std::size_t|float)\s+([a-z_0-9]+)\s*\(', text, re.M)))


def same_text(k, d):
    if d is None:
        return 'new in fastasfuck, no strixite counterpart'
    if d[0] == 0:
        return 'byte-identical to %s %s at %s' % (k['cpt'], k['cfile'], STRIXITE_COMMIT)
    return '%d changed lines against %s %s at %s' % (d[0], k['cpt'], k['cfile'], STRIXITE_COMMIT)


def provides_for(files):
    out = []
    for k in files:
        if not k['ifaces']:
            continue
        sp = strixite_provides(k['cpt'])
        d = counterpart(k)
        hpp = k['file'][:-4] + '.hpp' if k['file'].endswith('.hip') else k['file']
        ep = entries(hpp) if os.path.isfile(os.path.join(CHECKOUT, hpp)) else []
        for i in k['ifaces']:
            if i not in sp:
                raise SystemExit(f"{k['file']}: {k['cpt']} does not provide {i}")
            api = ('strix::kernels::' + ', '.join(ep) + ' (' + hpp + ')') if ep else k['file']
            if d and d[0] == 0:
                num = 'the same source as %s (%s), so its arithmetic: %s' % (k['cpt'], same_text(k, d), sp[i].get('numeric', ''))
            else:
                num = 'derived from %s (%s); the arithmetic was not compared here' % (k['cpt'], same_text(k, d))
            out.append(P(i, api, num, 'as %s: %s' % (k['cpt'], sp[i].get('regime', ''))))
    return out


def mod(id, summary, files, provides, notes, why, lang='hip-c++', compile='standalone', reason=None):
    got = take([k['file'] for k in files])
    sources = [f for f in got if f.endswith('.hip')]
    mods.append(M(id, summary, got, provides=provides, lang=lang, sources=sources, includes=['src'],
                  compile=compile, reason=reason, notes=notes, why=why))


def tag_line(files, only_tagged_headers=True):
    return '; '.join('%s %s (%s)' % (k['file'], ', '.join(sorted({i.rsplit('/', 1)[0] for i in k['ifaces']})) or 'none', same_text(k, counterpart(k)))
                     for k in files if k['file'].endswith('.hip') or k['ifaces'] or not only_tagged_headers)


mod('kernel.fastasfuck.common',
    'fastasfuck\'s shared headers: the HIP runtime include point and STRIX_HIP_CHECK (common/), host checks, the cached device info, the split-plane expert row layout and the shared type and math helpers of kernels/',
    COMMON_FILES, provides_for(COMMON_FILES),
    'Interface tags by file: ' + tag_line(COMMON_FILES, False) + '.',
    'every fastasfuck kernel file includes common/hip_check.hpp and most include kernels/types.hpp, so no family module compiles without these headers; types.hpp and math_utils.hpp are the two shared headers strixite does not have',
    compile='header-only', reason='header-only; compiled inside every kernel.fastasfuck.* family module that includes it')
mod('format.fastasfuck.qn_device',
    'fastasfuck\'s Q4 / Q5 / Q6 / Q8 weight formats as the kernels see them: the host weight structs (formats/q*.hpp, group-affine codes with BF16 scale and min) and their device views and owning uploads (runtime/q*_device.hpp) over the owning device buffer',
    FORMAT_FILES, provides_for(FORMAT_FILES),
    'Interface tags by file: ' + tag_line(FORMAT_FILES) + '. The codes are therefore strixw\'s Q4 / Q5 / Q6 / Q8 layouts; the .faf container fastasfuck loads them from (formats/faf.cpp, runtime/faf_loader.cpp) and the .cpp files of formats/ (converters, quantizers) are not harvested.',
    'the linear, hc, embedding and MoE kernel headers include runtime/q*_device.hpp, so these headers are needed to compile them; they also show that fastasfuck\'s q4g128 / q6g64 / q8g64 weights are strixw\'s block layouts in another container',
    lang='c++', compile='header-only', reason='header-only; the format headers the kernel headers include')

for fam, files in FAMILIES.items():
    hips = [k for k in files if k['file'].endswith('.hip')]
    cpts = sorted({k['cpt'] for k in hips})
    mod('kernel.fastasfuck.' + fam, SUMMARY[fam], files, provides_for(files),
        'Interface tags by file: ' + tag_line(files) + ' (catalog/fastasfuck/KERNELS.md has the table). ' + NOT_TIMED,
        WHY_FAMILY % ('%s kernel file%s' % (fam, 's' if len(hips) > 1 else ''), (', '.join(cpts[:-1]) + ' and ' + cpts[-1]) if len(cpts) > 1 else cpts[0]))

# every kernels/*.hip is in its family module, and every file of kernels/ is in some module
hips = sorted(f for f in O.files() if re.fullmatch(r'kernels/[^/]+\.hip', f))
declared = sorted(k['file'] for fs in FAMILIES.values() for k in fs if k['file'].endswith('.hip'))
if hips != declared or len(hips) != 25:
    raise SystemExit('kernels/*.hip differ from FAMILIES: ' + ' '.join(sorted(set(hips) ^ set(declared))))
left = sorted(f for f in O.expand(['kernels/']) if f not in claimed)
if left:
    raise SystemExit('kernels/ files in no module: ' + ' '.join(left))


# ---------------------------------------------------------------- the engine around the kernels (card 17eead1048)
# One module a unit of the host engine, split the way catalog/strixite splits the same files (fastasfuck is its
# fork), so each module has a strixite counterpart to set it against where strixite has the file. CMake builds these
# as host C++ with the ROCm clang (no -x hip, CMakeLists.txt), so the compile gate, which builds hip-c++ modules,
# does not compile them: [gate].compile is n/a (header-only for headers) and the reason names the CMake target.


def cmake_targets():
    """{origin .cpp path: [target, ...]} from CMakeLists.txt: add_library(name ...), add_executable(name ...) and
    target_sources(name [PRIVATE|PUBLIC|INTERFACE] ...); executables read 'executable <name>', libraries first."""
    text = re.sub(r'#[^\n]*', '', open(os.path.join(CHECKOUT, 'CMakeLists.txt')).read())
    out = {}
    for cmd, body in re.findall(r'\b(add_library|add_executable|target_sources)\s*\(([^)]*)\)', text):
        words = body.split()
        if not words or words[0].startswith('$'):
            continue
        name = words[0]
        for w in words[1:]:
            if w.endswith('.cpp') and not w.startswith('$'):
                out.setdefault(w, [])
                out[w].append(name)
    exe = set(re.findall(r'\badd_executable\s*\(\s*([A-Za-z0-9_]+)', text))
    return {f: sorted(dict.fromkeys(ts), key=lambda t: (t in exe, t)) for f, ts in out.items()}, exe


_TARGETS, _EXES = cmake_targets()


def targets_of(f):
    ts = _TARGETS.get(f)
    if not ts:
        raise SystemExit(f'{f}: in no add_library / add_executable / target_sources of CMakeLists.txt')
    return [('executable ' + t) if t in _EXES else t for t in ts]


def H(id, files, cpt, summary, why, ifaces=None, lang='c++'):
    """One host module. files: origin paths; cpt: the catalog/strixite module holding the same files, or None;
    ifaces: None copies cpt's [[provides]] (the same job), else [(interface, api, numeric, regime), ...]."""
    return dict(id=id, files=files, cpt=cpt, summary=summary, why=why, ifaces=ifaces, lang=lang)


def T(name, summary, why, cpt=None, ifaces=None, path=None, lang=None):
    path = path or 'tools/%s.%s' % (name, 'py' if lang == 'python' else 'cpp')
    lang = lang or ('python' if path.endswith('.py') else 'c++')
    iname = name.replace('_', '-')
    if ifaces is None and cpt is None:
        ifaces = [('tool.%s/-/-' % iname, path, 'n/a', 'offline tool' if lang == 'python' else 'host program')]
    return H('tool.fastasfuck.' + name, [path], cpt, summary, why, ifaces, lang)


HOST = [
 # common/
 H('runtime.fastasfuck.args', ['common/args.hpp'], 'runtime.strixite.args',
   'fastasfuck\'s header-only checked integer and double parsing for its command-line tools, failing through STRIX_CHECK with the argument\'s name and range',
   'every converter, bench and test program of fastasfuck parses its flags through this header, so a tool module cannot be read without it'),
 H('runtime.fastasfuck.bf16', ['common/bf16.hpp'], 'runtime.strixite.bf16',
   'fastasfuck\'s header-only host BF16 widening and round-to-nearest-even narrowing by bit manipulation',
   'the host converters and loaders round to BF16 with these helpers, the host twin of the device conversions in the kernels'),
 H('runtime.fastasfuck.log', ['common/log.hpp'], 'runtime.strixite.log',
   'fastasfuck\'s header-only printf-style one-line logger with syslog levels, a per-slot tag and a journal priority prefix',
   'the engine and server log through it; the module lets a recipe swap logging without touching the engine files'),
 H('runtime.fastasfuck.trace', ['common/trace.hpp'], 'runtime.strixite.trace',
   'fastasfuck\'s ROCTX range macros for rocprofv3 timelines, compiled to nothing unless STRIX_ROCTX is defined',
   'tools/profile_run.py reads the ranges these macros emit (forward, layer, sublayer), so profiles of fastasfuck depend on this header'),
 # formats/
 H('format.fastasfuck.q4', ['formats/q4.cpp'], 'format.strixite.q4',
   'fastasfuck\'s Q4 reference quantizer and dequantizer behind formats/q4.hpp: 4-bit codes with BF16 scale and min per group, row append and a chunk-major relayout',
   'the 1.1.0 GPTQ tools promise codes bit-identical to this file (tools/quantize_qwen_gptq.py docstring), so it is the reference a calibrated Q4 tensor is checked against'),
 H('format.fastasfuck.q5', ['formats/q5.cpp'], 'format.strixite.q5',
   'fastasfuck\'s Q5 host quantizer: the Q4 low-nibble plane plus a 1-bit high plane per row with Q4\'s BF16 scale and min',
   'Q5 is one of the four weight widths a .faf layout string may name; this is the host side the Q5 decode and prefill kernels read'),
 H('format.fastasfuck.q6', ['formats/q6.cpp'], 'format.strixite.q6',
   'fastasfuck\'s Q6 host quantizer: the Q4 low plane plus a 2-bit high plane per row, 6.25 bits per weight at G128',
   'the shipped .faf keeps every dense attention, GDN-out and shared-expert weight at q6g64 (README.md), so this quantizer writes a large share of the served model'),
 H('format.fastasfuck.q8', ['formats/q8.cpp'], 'format.strixite.q8',
   'fastasfuck\'s Q8 host quantizer: one unsigned byte per weight with asymmetric BF16 scale and min per group and a chunk-major copy for the HC up-mix',
   'the hyper-connection and LM-head weights ship at q8g64, so the MTP draft acceptance the author tunes for rests on this format'),
 H('format.fastasfuck.q4_from_q8', ['formats/q4_from_q8.cpp', 'formats/q4_from_q8.hpp'], 'format.strixite.q4_from_q8',
   'fastasfuck\'s Q8 to Q4 requantizer: exact Q8 dequantize then Q4 quantize, threaded over row chunks with byte-identical output',
   'it lets a Q4 layout be derived from an existing Q8 conversion without re-reading the BF16 checkpoint'),
 H('format.fastasfuck.safetensors', ['formats/safetensors.cpp', 'formats/safetensors.hpp'], 'format.strixite.safetensors',
   'fastasfuck\'s read-only safetensors reader: header JSON to name, dtype, shape and offset per tensor over an mmap of the file',
   'the converter reads the HF checkpoint and, since 1.1.0, the prebuilt GPTQ shards through this reader'),
 H('format.fastasfuck.merge_plan', ['formats/merge_plan.cpp', 'formats/merge_plan.hpp'], 'format.strixite.merge_plan',
   'fastasfuck\'s table of linear weights that read the same input (GDN qkv+z+b+a, attention q+k+v plus indexer, expert gate+up), row-concatenated at conversion',
   'the merged rows are why one GEMV serves several projections; the converter and the forward agree on them through this table'),
 H('format.fastasfuck.ngram_table', ['formats/ngram_table.cpp', 'formats/ngram_table.hpp'], 'format.strixite.ngram_table',
   'fastasfuck\'s ngram.table container: a hashed 512-byte header with the checkpoint fingerprint, then fixed-size BF16, FP8 or int8 PLE rows from offset 4096',
   'the PLE n-gram rows are too large for memory, so the served engine reads them from this file one positioned read a row'),
 H('format.fastasfuck.faf', ['formats/faf.cpp', 'formats/faf.hpp'], None,
   'fastasfuck\'s .faf weights container: header, hashed index and a data region with every tensor in its kernel\'s layout (Q4/Q5/Q6/Q8 row- or chunk-major, BF16, F32, I64), a streaming writer and an mmap reader',
   '.faf is what fastasfuck renamed strixite\'s strixw container to and extended (format v4 adds Q6 and Q5 row-major); every served model and every 1.1.0 calibrated package is one of these files',
   [('format.container/faf/-', 'strix::FafWriter, strix::FafFile (formats/faf.hpp)', 'n/a: tensors are stored as written, each component hashed', 'host, offline write and mmap read')]),
 H('format.fastasfuck.convert_qwen4exp', ['formats/convert_qwen4exp.cpp', 'formats/convert_qwen4exp.hpp'], 'format.strixite.convert_qwen4exp',
   'fastasfuck\'s converter library from Qwen3.8-Flash-Next safetensors to weights.faf: merge plan, per-class --layout quantization, the HC fold and, new in 1.1.0, a --prebuilt store that takes GPTQ-calibrated <tensor>.{q,scale,minv} instead of quantizing',
   'the 1.1.0 release is this file: PrebuiltStore lets offline GPTQ weights enter a .faf with the same layout and bytes as round-to-nearest ones and records the calibration and its fingerprint in the metadata',
   [('format.convert/faf/-', 'strix::convert_qwen4exp (formats/convert_qwen4exp.hpp, ConvertOptions)', 'round-to-nearest per-group quantization as format.fastasfuck.q4 .. q8; a --prebuilt tensor is copied as given, its bytes and dtype checked', 'host, offline'),
    ('format.prebuilt-import/faf/-', 'ConvertOptions::prebuilt_dir (--prebuilt DIR of *.safetensors plus calib_meta.json)', 'exact copy of the prebuilt codes, BF16 scale and min; every prebuilt base must be consumed', 'host, offline (new in 1.1.0)')]),
 H('format.fastasfuck.safetensors_loader', ['runtime/weight_loader.cpp', 'runtime/weight_loader.hpp'], 'format.strixite.safetensors_loader',
   'fastasfuck\'s bulk safetensors loader: concurrent pread workers into pinned host memory, then fdatasync and page-cache eviction',
   'the Qwen3.5 BF16 path and the tests load checkpoints through it without leaving the file in the page cache of a unified-memory machine'),
 H('format.fastasfuck.faf_loader', ['runtime/faf_loader.cpp', 'runtime/faf_loader.hpp'], 'format.strixite.strixw_loader',
   'fastasfuck\'s .faf to device loader: one allocation for the data region, concurrent preads through two event-gated pinned staging buffers, typed Q4/Q5/Q6/Q8 views, and a HostExperts placement that leaves the expert stacks in pinned host memory',
   'the event gate on staging-buffer reuse is a gfx1151 host-buffer hazard the file documents; the HostExperts placement lets a smaller discrete card run the same kernels on the same bytes',
   [('format.device-loader/faf/-', 'strix::FafDevice (runtime/faf_loader.hpp)', 'n/a: bytes copied unchanged, alignment carried from the file', 'host to device, one GPU (GTT on gfx1151)')]),
 # runtime/
 H('runtime.fastasfuck.ngram_rows', ['runtime/ngram_rows.cpp', 'runtime/ngram_rows.hpp'], 'runtime.strixite.ngram_rows',
   'fastasfuck\'s PLE row source that gathers n-gram rows straight from the checkpoint\'s ngram_embedding shards with deduplicated preads and no cache',
   'a reference source for the PLE rows that needs no converted table, used to check the table path against the checkpoint'),
 H('runtime.fastasfuck.ngram_table', ['runtime/ngram_table.cpp', 'runtime/ngram_table.hpp'], 'runtime.strixite.ngram_table',
   'fastasfuck\'s serving PLE row source: on-demand reads from the converted ngram.table with deduplication, up to 64 concurrent preads, BF16 decode and an LRU row cache',
   'every decoded token needs 16 PLE rows; this reader keeps that off the critical path on an SSD-backed table'),
 H('runtime.fastasfuck.paged_buffer', ['runtime/paged_buffer.hpp'], 'runtime.strixite.paged_buffer',
   'fastasfuck\'s header-only growable device arrays on HIP virtual memory: one reserved range, physical pages mapped in as the used prefix grows',
   'the K/V caches and parked states of up to 4 slots grow in place through it without reallocating or copying'),
 H('runtime.fastasfuck.paged_state', ['runtime/paged_state.hpp'], 'runtime.strixite.paged_state',
   'fastasfuck\'s HIP-free description of a parked conversation for host code: generation id, byte counts and host-readable chunk addresses',
   'the prompt cache saves and restores parked states through this header without including HIP'),
 H('runtime.fastasfuck.ple_hash', ['runtime/ple_hash.cpp', 'runtime/ple_hash.hpp'], 'runtime.strixite.ple_hash',
   'fastasfuck\'s host computation of the PLE n-gram row ids: order-2 and order-3 token hashes reduced modulo each head\'s prime vocabulary',
   'the row ids decide which PLE rows a token reads, so the hash has to match the checkpoint\'s training-time rule bit for bit'),
 H('runtime.fastasfuck.qweight', ['runtime/qweight.hpp'], 'runtime.strixite.qweight',
   'fastasfuck\'s header-only quantized-weight view that records the Q4/Q5/Q6/Q8 format a .faf tensor declares and dispatches to the matching GEMV, WMMA or grouped-expert kernel',
   'it is the switch that lets one forward run any per-class layout a .faf names without recompiling'),
 H('executor.fastasfuck.qwen3_5', ['runtime/qwen3_5.cpp', 'runtime/qwen3_5.hpp'], 'executor.strixite.qwen3_5',
   'fastasfuck\'s Qwen3.5 text forward on unquantized BF16 linears: 24 GDN or full-attention layers with SwiGLU MLP and a tied LM head',
   'a small BF16 model path kept beside the Flash-Next engine, useful as an unquantized reference for the shared kernels'),
 H('executor.fastasfuck.qwen4exp', ['runtime/qwen4exp.cpp', 'runtime/qwen4exp.hpp'], 'executor.strixite.qwen4exp',
   'fastasfuck\'s Qwen3.8-Flash-Next engine: loads weights.faf and the n-gram table, runs 4 hyper-connected streams through 36 GDN and 12 QSA attention layers with PLE and a top-10-of-512 MoE, chunked prefill, MTP drafts and batched verify',
   'the forward that ties every kernel family together; set against executor.strixite.qwen4exp it shows what the fork changed in scheduling and precision'),
 H('executor.fastasfuck.vision_tower', ['runtime/vision_tower.cpp', 'runtime/vision_tower.hpp'], 'executor.strixite.vision_tower',
   'fastasfuck\'s device-resident ViT and patch merger loaded from vision.faf as exact BF16 rows, one image per forward',
   'image prompts go through it before the language forward splices the merged rows in'),
 H('executor.fastasfuck.qwen4exp_backend', ['serve/qwen4exp_backend.cpp', 'serve/qwen4exp_backend.hpp'], 'executor.strixite.qwen4exp_backend',
   'fastasfuck\'s LmBackend over one Qwen4ExpSession with snapshot slots, a paged K/V budget and MTP drafts; its saved-state fingerprint gains a cal-<prebuilt_fingerprint> tag in 1.1.0',
   'the 1.1.0 fingerprint tag keeps prompt-cache entries built on GPTQ weights apart from entries built on round-to-nearest weights that share the source fingerprint and layout'),
 # serve/ (host libraries of strix_serve)
 H('runtime.fastasfuck.capture', ['serve/capture.cpp', 'serve/capture.hpp'], 'runtime.strixite.capture',
   'fastasfuck\'s opt-in per-request capture files: a JSON header, prompt and generated token ids, and a drafted/accepted/emitted record per decode step',
   'the captures are the raw data for MTP acceptance studies such as tools/measure_verify_curve and the adaptive draft margin'),
 H('runtime.fastasfuck.chat_template', ['serve/chat_template.cpp', 'serve/chat_template.hpp'], 'runtime.strixite.chat_template',
   'fastasfuck\'s hand-rendered Qwen3.8-Flash-Next chat template in C++, byte-identical to transformers\' apply_chat_template with tools and thinking switches',
   'prompts must match the reference tokens for prefix reuse and quality, and the server renders every chat request through this file'),
 H('runtime.fastasfuck.engine', ['serve/engine.cpp', 'serve/engine.hpp', 'serve/seq_backend.cpp', 'serve/seq_backend.hpp'], 'runtime.strixite.engine',
   'fastasfuck\'s serving loop: one worker per slot behind LmBackend, longest-prefix resume, chunked prefill, sampling, parsing and streaming with MTP draft-and-verify, and SeqBackend batching several slots\' verifies into one forward',
   'the 104 tok/s 4-slot figure the author quotes comes from this loop batching slots; it is the module a slots experiment would swap'),
 H('runtime.fastasfuck.grammar', ['serve/grammar.cpp', 'serve/grammar.hpp'], 'runtime.strixite.grammar',
   'fastasfuck\'s byte-level context-free grammar matcher: a nondeterministic pushdown automaton tracking every live parse with a depth bound',
   'structured output and tool calls are enforced through this matcher token by token'),
 H('runtime.fastasfuck.grammar_cache', ['serve/grammar_cache.cpp', 'serve/grammar_cache.hpp'], 'runtime.strixite.grammar_cache',
   'fastasfuck\'s mutex-guarded LRU of compiled grammar automata keyed by schema or tools, each keeping the token masks it has filled',
   'repeat requests with the same schema or tools skip grammar compilation and mask building'),
 H('runtime.fastasfuck.host_buffer', ['serve/host_buffer.cpp', 'serve/host_buffer.hpp'], 'runtime.strixite.host_buffer',
   'fastasfuck\'s growable host byte buffer on anonymous mmap that pre-faults pages and grows with mremap',
   'the prompt cache\'s host tier holds saved states in these buffers so growth never copies a state'),
 H('runtime.fastasfuck.json', ['serve/json.cpp', 'serve/json.hpp'], 'runtime.strixite.json',
   'fastasfuck\'s strict RFC 8259 JSON value, parser and printers, one of them reproducing Python json.dumps byte for byte',
   'the chat template\'s tojson must print exactly as Python does for prompts to match, and the whole server speaks JSON through this file'),
 H('runtime.fastasfuck.json_schema', ['serve/json_schema.cpp', 'serve/json_schema.hpp'], 'runtime.strixite.json_schema',
   'fastasfuck\'s compiler from a JSON Schema subset to a byte grammar that only admits texts it can promise are valid',
   'response_format json_schema requests become grammars through it'),
 H('runtime.fastasfuck.serve_log', ['serve/log.hpp'], 'runtime.strixite.serve_log',
   'fastasfuck\'s request log layout helpers: fixed-column label rows, rate-first speed rows and hand comma-grouped numbers',
   'the per-request speed lines the author\'s benches scrape are printed by these helpers'),
 H('runtime.fastasfuck.memory_governor', ['serve/memory_governor.hpp'], 'runtime.strixite.memory_governor',
   'fastasfuck\'s header-only owner of the unified-memory RAM budget: a MemAvailable floor and pure room and shortfall rules on one reading',
   'on a 128 GB unified-memory machine running a 65 GiB model, every cache decision has to agree on one budget or the machine hangs'),
 H('runtime.fastasfuck.memory_plan', ['serve/memory_plan.cpp', 'serve/memory_plan.hpp'], 'runtime.strixite.memory_plan',
   'fastasfuck\'s startup RAM check after every GPU buffer exists, failing with the settings that would fit',
   'the server refuses to start rather than hang later when the model, K/V and cache margin do not fit'),
 H('runtime.fastasfuck.mtp_calib', ['serve/mtp_calib.cpp', 'serve/mtp_calib.hpp'], None,
   'fastasfuck\'s MTP latency calibration: measures draft and verify step times on the running GPU (warmup, sample counts, tolerance), caches the profile and falls back to built-in gfx1151 figures',
   'the engine picks how many tokens to draft from measured verify cost versus draft cost; strixite has no such file, it is fastasfuck\'s own addition',
   [('draft.mtp-calibration/-/-', 'strix::MtpCalibrationConfig, strix::MtpHardwareProfile (serve/mtp_calib.hpp)', 'timings in ms, fitted by the file; the draft length chosen from them changes speed, not the tokens', 'host, timed on one GPU')]),
 H('runtime.fastasfuck.output_parser', ['serve/output_parser.cpp', 'serve/output_parser.hpp'], 'runtime.strixite.output_parser',
   'fastasfuck\'s incremental parser from generated tokens to reasoning, content and schema-typed tool-call arguments',
   'streamed tool calls in the OpenAI and Responses APIs are cut from the model\'s XML call form here'),
 H('runtime.fastasfuck.prompt_cache', ['serve/prompt_cache.cpp', 'serve/prompt_cache.hpp'], 'runtime.strixite.prompt_cache',
   'fastasfuck\'s saved session states keyed by token ids: a RAM tier that shrinks with MemAvailable and checksummed files written under a write budget',
   'multi-turn agents resume from a saved state instead of prefilling again; the 1.1.0 calibration tag in the state fingerprint guards these entries'),
 H('runtime.fastasfuck.resume_loss', ['serve/resume_loss.cpp', 'serve/resume_loss.hpp'], 'runtime.strixite.resume_loss',
   'fastasfuck\'s one-line report of where a prompt left its nearest saved state, raised to a warning when the loss costs seconds of prefill',
   'it tells an operator why a turn prefilled from scratch, which the prompt cache alone does not show'),
 H('runtime.fastasfuck.response_store', ['serve/response_store.cpp', 'serve/response_store.hpp'], None,
   'fastasfuck\'s bounded in-memory store of past Responses API turns (tokens, instructions, tools, config hash) by response id, pruned by entry count and bytes',
   'previous_response_id in the Responses API resumes from these stored turns; strixite has no Responses API',
   [('serve.response-store/-/-', 'strix::ResponseStore, strix::StoredResponse (serve/response_store.hpp)', 'n/a', 'host')]),
 H('runtime.fastasfuck.sampler', ['serve/sampler.cpp', 'serve/sampler.hpp'], 'runtime.strixite.sampler',
   'fastasfuck\'s host next-token sampler in transformers\' warper order over a full logits row or the GPU\'s top-20 candidates',
   'temperature 0.6 throughput figures and T = 0 bit-identity checks in the benches both go through this sampler'),
 H('runtime.fastasfuck.structured_output', ['serve/structured_output.cpp', 'serve/structured_output.hpp'], 'runtime.strixite.structured_output',
   'fastasfuck\'s per-generation response_format cursor: free reasoning, then only grammar tokens, one allowed-token mask per logits row',
   'it joins the grammar matcher to the sampler so constrained output still allows thinking first'),
 H('runtime.fastasfuck.think_nudge', ['serve/think_nudge.cpp', 'serve/think_nudge.hpp'], 'runtime.strixite.think_nudge',
   'fastasfuck\'s ThinkWatch: detects self-repeating long reasoning by 8-gram hashes and feeds a short nudge sentence at a paragraph break',
   'long thinking loops waste decode time; this is the engine-side guard against them'),
 H('runtime.fastasfuck.token_mask', ['serve/token_mask.cpp', 'serve/token_mask.hpp'], 'runtime.strixite.token_mask',
   'fastasfuck\'s allowed-token mask builder walking the byte-sorted vocabulary with shared-prefix reuse and per-state caching',
   'masking all 248k tokens per step would be too slow without the prefix walk this file does'),
 H('runtime.fastasfuck.tokenizer', ['serve/tokenizer.cpp', 'serve/tokenizer.hpp'], 'runtime.strixite.tokenizer',
   'fastasfuck\'s hand-written byte-level BPE for Qwen3.8-Flash-Next loaded and checked against tokenizer.json',
   'no Python or HF tokenizer runs in the server; token ids for every request come from this file'),
 H('runtime.fastasfuck.unicode', ['serve/unicode.cpp', 'serve/unicode.hpp', 'serve/unicode_tables.inc'], 'runtime.strixite.unicode',
   'fastasfuck\'s UTF-8 repair, Unicode classes and NFC composition from tables generated off Python\'s unicodedata',
   'the tokenizer\'s pre-tokenizer and the chat template\'s strip need Python-identical Unicode behaviour'),
 H('runtime.fastasfuck.vision_preprocess', ['serve/vision_preprocess.cpp', 'serve/vision_preprocess.hpp'], 'runtime.strixite.vision_preprocess',
   'fastasfuck\'s host image path: sniffed decode (stb, libwebp), smart_resize with an antialiased bicubic filter, normalization and patch merge',
   'pixel values must match the HF processor for the vision tower\'s output to match; this is the host half of image prompts'),
 H('runtime.fastasfuck.vision_request', ['serve/vision_request.cpp', 'serve/vision_request.hpp'], 'runtime.strixite.vision_request',
   'fastasfuck\'s serving side of image prompts: base64 data URLs, image pad expansion, mRoPE positions and content-derived cache ids',
   'it lets image prompts share the prompt cache safely: the same image bytes hit, different images miss'),
 H('runtime.fastasfuck.vocab_filter', ['serve/vocab_filter.cpp', 'serve/vocab_filter.hpp'], None,
   'fastasfuck\'s vocabulary filter: classifies tokens into script sections (Latin, Cyrillic, CJK, code, ...) and masks the ones a preset such as en, en+ru or no-cjk leaves out',
   'a fastasfuck-only feature with no strixite counterpart: it narrows sampling to the scripts an operator expects',
   [('text.vocab-filter/-/-', 'strix::vocab_filter::parse_filter and the section masks (serve/vocab_filter.hpp)', 'masked tokens never sampled; unmasked logits unchanged', 'host')]),
 H('serve.fastasfuck.http', ['serve/http_server.cpp', 'serve/http_server.hpp', 'serve/server_config.cpp', 'serve/server_config.hpp'], 'serve.strixite.http',
   'fastasfuck\'s HTTP surface on cpp-httplib (chat and text completions as JSON or SSE, models, health, metrics, cache routes) and its option table read from a key = value config',
   'the server\'s routes and every operator setting live here; the vendored cpp-httplib is not harvested'),
 H('serve.fastasfuck.openai', ['serve/openai.cpp', 'serve/openai.hpp'], 'serve.strixite.openai',
   'fastasfuck\'s field-by-field validation of OpenAI chat-completions and completions bodies into engine requests, and the response, SSE and error bodies',
   'OpenAI clients reach the engine through it; it decides which request fields are honoured'),
 H('serve.fastasfuck.responses_api', ['serve/responses_api.cpp', 'serve/responses_api.hpp'], None,
   'fastasfuck\'s OpenAI Responses API: input items (messages, function calls and outputs), instructions, previous_response_id chaining through the response store, tools and streaming events',
   'the Responses API is what fastasfuck adds over strixite\'s chat-completions server for agent clients that use it',
   [('serve.api/openai-responses/-', 'strix::ResponsesRequest and its handlers (serve/responses_api.hpp)', 'n/a', 'host, over runtime.fastasfuck.engine')]),
 H('serve.fastasfuck.fastasfuck_server', ['tools/fastasfuck_server.cpp'], 'serve.strixite.strix_server',
   'fastasfuck\'s server main program: option table and config file, startup memory plan, loads weights.faf, the n-gram table and the tokenizer, serves on port 5300 until SIGTERM',
   'the program the author\'s throughput figures come from; it is strixite\'s strix_server renamed and carried on the fork'),
]

HOST_TOOLS = [
 T('check_stream_calls', 'fastasfuck\'s host-only differential test: streamed tool calls fed byte by byte and in random pieces must match the buffered output parser', 'it is the guard that streaming tool-call output does not diverge from the buffered parser', cpt='tool.strixite.check_stream_calls'),
 T('convert_ngram_table', 'fastasfuck\'s converter from the checkpoint\'s 128 ngram_embedding shards to one exact BF16 ngram.table with a sampled re-read check', 'the served PLE table is made by this tool and then transcoded to 8 bits', cpt='tool.strixite.convert_ngram_table'),
 T('convert_qwen4exp', 'fastasfuck\'s convert_qwen4exp program: HF safetensors to weights.faf by --layout, with --mtp-from, --vision and, new in 1.1.0, --prebuilt DIR for GPTQ-calibrated tensors', 'the 1.1.0 release adds the --prebuilt flag here, the entry point to pair tools/quantize_qwen_gptq.py output with the converter', cpt='tool.strixite.convert_qwen4exp'),
 T('transcode_ngram_table', 'fastasfuck\'s BF16 ngram.table to FP8 or int8 row transcoder with a bit-exact read-back check and row error statistics', 'the published n-gram table is the int8 one this tool writes from the BF16 table, at half the size on SSD', cpt='tool.strixite.transcode_ngram_table'),
 T('inspect_faf', 'fastasfuck\'s .faf inspector: prints metadata and the tensor index, --verify re-hashes every component', 'the quick check that a converted or calibrated .faf is whole and what layout it carries'),
 T('eval_ppl_kld', 'fastasfuck\'s perplexity and KL-divergence evaluator comparing two weight layouts token by token with raw logit dumps', 'quantization choices in fastasfuck are judged by held-out KLD; this is the on-engine measurement'),
 T('measure_verify_curve', 'fastasfuck\'s bench of the verify forward latency for M = 1..16 drafted tokens solo and batched over 4 slots on real weights', 'the MTP calibration fallback figures and the draft-length policy come from this curve'),
 T('test_linear_q6', 'fastasfuck\'s Q6 decode and WMMA linear kernel test against the host Q6 reference on random weights', 'the Q6 kernels serve every dense weight of the shipped layout, so they get their own correctness test'),
 T('test_ngram_cache', 'fastasfuck\'s test of the ngram.table reader and its LRU row cache against direct reads', 'a wrong cached PLE row would silently change every token after it; this test pins the cache'),
 T('test_split_invariance', 'fastasfuck\'s proof tool that replaying a prefix of verify rows gives a bit-exact GDN recurrent state', 'MTP rollback in a recurrent model relies on this invariance; the test is the evidence for it'),
 T('test_vision_batchable_edge_cases', 'fastasfuck\'s test of image requests batched across slots on the real backend: pads, positions and cache ids in edge cases', 'mixing image and text slots in one batched verify is easy to get wrong; this test covers the cases the author hit'),
 T('breakdown_split', 'fastasfuck\'s per-kernel layer breakdown bench on gfx1151 (GDN, dense and QSA attention, MoE, HC, LM head, MTP head) with MALL flushes', 'it shows where a decode step spends its time kernel by kernel, the input for any kernel swap'),
 T('bench_threshold_gemv_vs_wmma', 'fastasfuck\'s bench of the GEMV versus WMMA crossover in tokens for the linear, delta-rule, HC and grouped MoE kernels', 'the engine switches from GEMV to WMMA at a token count; this bench is where those thresholds come from'),
 T('audit_kernels', 'fastasfuck\'s script that disassembles a built binary\'s gfx1151 code objects and audits their kernels (registers, spills, LDS)', 'register spills on gfx1151 cost decode speed; this script lists them per kernel', lang='python'),
 T('bench_adaptive_margin', 'fastasfuck\'s sweep of the adaptive MTP margin slope against a running server', 'the steepness with which the engine lengthens MTP drafts as acceptance rises was picked by this sweep', lang='python'),
 T('profile_run', 'fastasfuck\'s rocprofv3 driver: runs a ROCTX build of the server on real weights, sends T = 1..512 queries and summarizes regions and kernels from the trace database', 'the end-to-end profile of the served engine; it reads the ranges common/trace.hpp emits', lang='python'),
 T('quantize_qwen_gptq', 'fastasfuck\'s production GPU GPTQ quantizer (new in 1.1.0): full-rank FP64 ActOrder GPTQ with static groups and Shrink512 over the experts, writing <tensor>.{q,scale,minv} bit-compatible with formats/q4.cpp plus calibration metadata', 'the 1.1.0 headline (-30.7% held-out KLD against round-to-nearest at the same bytes and speed, README.md) is produced by this script and fed to convert_qwen4exp --prebuilt', lang='python'),
 T('sim_qwen_quant', 'fastasfuck\'s layer-by-layer BF16 to Q4/Q5/Q6/Q8 Pareto simulator in pure PyTorch for qwen4exp, with bit-identical fake quantizers (new in 1.1.0)', 'the shared base the other 1.1.0 calibration simulators import: its forward and held-out document define the KLD they report', lang='python'),
 T('sim_qwen_calib', 'fastasfuck\'s held-out simulator of zero-speed-penalty calibration methods at the shipped layout, Hessians from a disjoint corpus (new in 1.1.0)', 'the first calibration round of the 1.1.0 study; later rounds import its corpus and helpers', lang='python'),
 T('sim_qwen_awq_gptq', 'fastasfuck\'s simulator of AWQ smoothing folded into the hyper-connection norms plus full-rank GPTQ at the shipped layout (new in 1.1.0)', 'it tested whether AWQ helps on top of GPTQ without kernel changes, the second round of the 1.1.0 study', lang='python'),
 T('sim_qwen_gptq_r3', 'fastasfuck\'s round-3 simulator of six pure-GPTQ variants (damping, importance clip, ActOrder static groups, expert shrinkage) (new in 1.1.0)', 'round 3 picked the ActOrder static-groups recipe that the production quantizer ships', lang='python'),
 T('sim_qwen_wave2', 'fastasfuck\'s round-4 simulator: Q6 hotspot GPTQ, coordinate-descent refits and asymmetric GPTAQ on top of the round-3 recipe (new in 1.1.0)', 'it is where the README\'s GPTQ versus GPTAQ ablation comes from', lang='python'),
 T('sim_qwen_fp64_mtp', 'fastasfuck\'s FP64-Hessian trunk plus MTP-head GPTQ simulator on the quantized trunk\'s outputs (new in 1.1.0)', 'it checks that calibrating the trunk does not cost MTP draft acceptance, and whether the draft head should be calibrated too', lang='python'),
 T('solve_qwen_quant', 'fastasfuck\'s CPU multiple-choice knapsack (MILP) optimizer picking a per-class or per-layer quantization layout under a byte budget from measured KLD (new in 1.1.0)', 'it turns the simulators\' sensitivity data into a layout without running the GPU', lang='python'),
 T('bench_common', 'fastasfuck\'s shared bench prompts, statistics and streaming OpenAI client for the tools/bench suite', 'every bench script imports it, so the suite\'s prompts and statistics are defined once here', path='tools/bench/common.py'),
 T('bench_ab_quant', 'fastasfuck\'s A/B bench of two quantization layouts on one server config: sterile solo runs with bit-identity, MTP acceptance, 4-slot concurrency and 4k/8k prefill (new in 1.1.0)', 'the 1.1.0 speed-parity claim for GPTQ against round-to-nearest weights was measured with this bench', path='tools/bench/bench_ab_quant.py'),
 T('bench_chunk_pareto', 'fastasfuck\'s two-phase sweep of the prefill chunk and chunk-shared sizes: solo saturation, then multi-agent replay', 'the prefill chunk and chunk-shared sizes in the served config were chosen by this sweep', path='tools/bench/bench_chunk_pareto.py'),
 T('bench_mixed_concurrency', 'fastasfuck\'s mixed decode and prefill contention bench (3D+1P, 2D+2P, 1D+3P) measuring step-latency percentiles in the overlap window', 'it shows how prefill on one slot slows decoding slots, the cost the chunk settings trade', path='tools/bench/bench_mixed_concurrency.py'),
 T('bench_multislot_scaling', 'fastasfuck\'s 1, 2 and 4 slot concurrency bench of aggregate and per-slot speed and inter-token latency', 'the 4-slot throughput figure in the README is this bench\'s number', path='tools/bench/bench_multislot_scaling.py'),
 T('bench_prefill_scaling', 'fastasfuck\'s prefill throughput and time-to-first-token bench from 4k to 128k context on non-repeating source text', 'long-context prefill speed, the other half of the README figures, is measured here', path='tools/bench/bench_prefill_scaling.py'),
 T('bench_solo_sterile', 'fastasfuck\'s single-slot T = 0 decode bench with dynamic bit-identity across passes', 'the solo tok/s figure and the determinism check behind it come from this bench', path='tools/bench/bench_solo_sterile.py'),
 T('run_mixed_triad', 'fastasfuck\'s runner that restarts the server per chunk-shared candidate and runs the mixed-concurrency bench for each', 'it automates the three-way comparison behind the chosen chunk-shared size', path='tools/bench/run_mixed_triad.py'),
]
HOST += HOST_TOOLS


def host_counterpart(f, cpt):
    """(changed lines, lines) of origin file f against the same path in catalog/strixite module cpt; None when there
    is no counterpart file."""
    if not cpt:
        return None
    kind, _, name = cpt.split('.', 2)
    path = os.path.join(STRIXITE, kind, name, 'src', f)
    if not os.path.isfile(path):
        return None
    a = open(O.find(f), errors='replace').read().splitlines()
    b = open(path, errors='replace').read().splitlines()
    return sum(1 for l in difflib.unified_diff(a, b, lineterm='', n=0) if l[:1] in '+-' and l[:3] not in ('+++', '---')), len(a)


def host_same(f, cpt):
    d = host_counterpart(f, cpt)
    if d is None:
        return 'new in fastasfuck' if not cpt else 'no file of that path in ' + cpt
    return 'byte-identical to %s at %s' % (cpt, STRIXITE_COMMIT) if d[0] == 0 else '%d changed lines against %s at %s' % (d[0], cpt, STRIXITE_COMMIT)


_PYIMP = re.compile(r'^(?:import\s+([A-Za-z_]\w*)|from\s+([A-Za-z_]\w*)\s+import\b)', re.M)


def host_mod(h):
    files = h['files']
    got = take(files)
    if h['ifaces'] is None:
        sp = strixite_provides(h['cpt'])
        ds = [host_counterpart(f, h['cpt']) for f in files]
        same = all(d is not None and d[0] == 0 for d in ds)
        prov = []
        for i, p in sp.items():
            num = ('the same source as %s, so its behaviour: %s' % (h['cpt'], p.get('numeric', 'n/a'))) if same else \
                  'derived from %s (%s); not compared here' % (h['cpt'], '; '.join('%s %s' % (f, host_same(f, h['cpt'])) for f in files))
            prov.append(P(i, p.get('api', files[0]), num, 'as %s: %s' % (h['cpt'], p.get('regime', ''))))
    else:
        prov = [P(i, a, n, r) for i, a, n, r in h['ifaces']]
    srcs = [f for f in got if f.endswith('.cpp')] if h['lang'] == 'c++' else []
    tgt = list(dict.fromkeys(t for f in srcs for t in targets_of(f)))
    if h['lang'] == 'python':
        compile, reason = 'n/a', 'a Python script run with python3 (torch, numpy or scipy where it imports them); nothing to compile'
    elif not srcs:
        compile, reason = 'header-only', 'host C++ header; the compile gate builds hip-c++ modules only'
    else:
        compile, reason = 'n/a', 'host C++ built by the origin\'s CMake (%s) with the ROCm clang as plain C++, no -x hip; the compile gate builds hip-c++ modules only' % ', '.join(tgt)
    reqs = []
    if h['lang'] == 'python':      # sibling imports (tools/ and tools/bench/ put their own directory on sys.path)
        names = {os.path.splitext(os.path.basename(x['files'][0]))[0]: x['id'] for x in HOST
                 if x['lang'] == 'python' and os.path.dirname(x['files'][0]) == os.path.dirname(files[0])}
        for a, b in _PYIMP.findall(open(O.find(files[0])).read()):
            n = a or b
            if n in names and names[n] != h['id']:
                reqs.append('module:' + names[n])
    notes = 'Files vs strixite: ' + '; '.join('%s %s' % (f, host_same(f, h['cpt'])) for f in files) + '. ' + NOT_TIMED
    if h['ifaces'] is not None and h['cpt']:
        notes += ' Interface tag: its own, because fastasfuck changed the job (strixite\'s ' + h['cpt'] + ' provides ' + \
                 ', '.join(sorted(strixite_provides(h['cpt']))) + ').'
    mods.append(M(h['id'], h['summary'], got, provides=prov, requires=sorted(set(reqs)), lang=h['lang'], sources=srcs,
                  includes=['src'] if h['lang'] == 'c++' else [], compile=compile, reason=reason, notes=notes, why=h['why']))


for h in HOST:
    host_mod(h)


# ---------------------------------------------------------------- delta against the previous pin (card 17eead1048)
OLD_COMMIT = '35015b5dfb6428a8200ebd427efb7b8ff5380b1c'
DATA = os.path.join(harvestlib.REPO, 'docs', 'harvest', 'data', 'fastasfuck')
DELTA_TSV = os.path.join(DATA, 'delta-35015b5.tsv')
DELTA_MD = os.path.join(DATA, 'delta-35015b5.md')


def plain_files(root):
    out = []
    for d, ds, fs in os.walk(root):
        ds[:] = [x for x in ds if not (d == root and x == '.git')]
        out += [os.path.relpath(os.path.join(d, f), root) for f in fs]
    return sorted(out, key=lambda s: s.encode())


def write_delta(old):
    """--delta OLD_CHECKOUT: compare the 35015b5 tree (a copy of its ark directory without .git) with this checkout."""
    if hashlib.sha256(''.join('%s  %s\n' % (harvestlib.sha256(os.path.join(old, f)), f) for f in plain_files(old)).encode()).hexdigest() != OLD_TREE_SHA256:
        raise SystemExit(f'{old}: not the 35015b5 tree (tree sha256 differs from {OLD_TREE_SHA256})')
    a, b = set(plain_files(old)), set(O.files())
    rows = []
    for f in sorted(a | b, key=lambda s: s.encode()):
        ha = harvestlib.sha256(os.path.join(old, f)) if f in a else '-'
        hb = harvestlib.sha256(O.find(f)) if f in b else '-'
        if ha == hb:
            continue
        la = open(os.path.join(old, f), errors='replace').read().splitlines() if f in a else []
        lb = open(O.find(f), errors='replace').read().splitlines() if f in b else []
        d = [l for l in difflib.unified_diff(la, lb, lineterm='', n=0) if l[:1] in '+-' and l[:3] not in ('+++', '---')]
        rows.append(('A' if f not in a else 'D' if f not in b else 'M', f, ha, hb, sum(l[0] == '+' for l in d), sum(l[0] == '-' for l in d)))
    os.makedirs(DATA, exist_ok=True)
    with open(DELTA_TSV, 'w') as fh:
        fh.write('# Cheblan/fastasfuck %s (tag 1.1.0) against %s, the previous harvest pin (sha256 of each file).\n' % (COMMIT, OLD_COMMIT))
        fh.write('# Written by docs/harvest/harvest_fastasfuck.py --delta from the two ark copies; A added, M changed, D removed in 1.1.0. Card 17eead1048.\n')
        fh.write('status\tpath\told_sha256\tnew_sha256\tadded\tremoved\n')
        for r in rows:
            fh.write('\t'.join(map(str, r)) + '\n')
    print(f'{DELTA_TSV}: {len(rows)} rows')


def read_delta():
    rows = []
    for line in open(DELTA_TSV):
        if line.startswith('#') or line.startswith('status\t'):
            continue
        s, f, ha, hb, ad, rm = line.rstrip('\n').split('\t')
        if s != 'D' and harvestlib.sha256(O.find(f)) != hb:
            raise SystemExit(f'{DELTA_TSV}: {f} does not match the checkout')
        rows.append(dict(status=s, path=f, added=int(ad), removed=int(rm)))
    return rows


def delta_md(owner):
    rows = read_delta()
    if any(r['path'].startswith(('kernels/', 'runtime/', 'common/')) for r in rows):
        raise SystemExit('delta_md: the text says no kernels/, runtime/ or common/ file changed; it no longer holds')
    q = '\x60'
    bymod = {}
    for r in rows:
        bymod.setdefault(owner.get(r['path'], '(not harvested)'), []).append(r)
    o = ['# fastasfuck 1.1.0 against 35015b5', '',
         'What Cheblan/fastasfuck release 1.1.0 (§%s§) changed against the previous harvest pin §%s§ (main, 2026-11-24).' % (COMMIT[:12], OLD_COMMIT[:12]),
         'File rows: §delta-35015b5.tsv§ beside this file. Written by §docs/harvest/harvest_fastasfuck.py§. Card FASTASFUCK-110 17eead1048.', '',
         '%d files differ: %d added, %d changed, %d removed. No file under §kernels/§, §runtime/§ or §common/§ changed, so every kernel' % (
             len(rows), sum(r['status'] == 'A' for r in rows), sum(r['status'] == 'M' for r in rows), sum(r['status'] == 'D' for r in rows)),
         'family module is byte-for-byte the 35015b5 harvest. 1.1.0 is a weights-quality release: offline GPTQ calibration',
         '(the §tools/*.py§ simulators and quantizer), a §--prebuilt§ path in the converter that takes the calibrated tensors',
         'into a §.faf§ of the same layout and bytes, a calibration tag in the prompt-cache state fingerprint, and the README',
         'and §docs/WEIGHTS.md§ text that reports the result (author figures, unverified here).', '',
         '## By module', '', '| module | files A/M/D | lines +/- |', '|---|---|---:|']
    for m in sorted(bymod, key=lambda m: (m == '(not harvested)', m)):
        rs = bymod[m]
        o.append('| %s | %d/%d/%d | +%d -%d |' % (m if m.startswith('(') else '§%s§' % m, sum(r['status'] == 'A' for r in rs), sum(r['status'] == 'M' for r in rs),
                                                 sum(r['status'] == 'D' for r in rs), sum(r['added'] for r in rs), sum(r['removed'] for r in rs)))
    o += ['', '## By file', '', '| status | file | lines +/- | module |', '|---|---|---:|---|']
    for r in rows:
        o.append('| %s | §%s§ | +%d -%d | %s |' % (r['status'], r['path'], r['added'], r['removed'], '§%s§' % owner[r['path']] if r['path'] in owner else 'not harvested'))
    open(DELTA_MD, 'w').write('\n'.join(o).replace('§', q) + '\n')
    return rows


def modules_md(owner, delta):
    """catalog/fastasfuck/MODULES.md: one row a module, with its 1.1.0 delta and strixite counterpart."""
    q = '\x60'
    prov = providers()
    changed = {}
    for r in delta:
        if r['path'] in owner:
            changed.setdefault(owner[r['path']], []).append(r['status'])
    o = ['# fastasfuck modules', '',
         'One row for each module of catalog/fastasfuck at 1.1.0 (§%s§). Card FASTASFUCK-110 17eead1048 (first harvest' % COMMIT[:12],
         'ORIGIN-FASTASFUCK 80ae224db4). Written by §docs/harvest/harvest_fastasfuck.py§; read from source, nothing was run or timed.', '',
         '- **1.1.0**: files of the module added (A) or changed (M) since 35015b5 (§docs/harvest/data/fastasfuck/delta-35015b5.md§); - = unchanged.',
         '- **strixite counterpart**: the catalog/strixite module holding the same files (fastasfuck is a fork of strixite).',
         '- **same op elsewhere**: modules of other origins that provide the same §<family>.<op>§.', '',
         '| module | lang | files | lines | compile | 1.1.0 | strixite counterpart | tag | same op elsewhere |', '|---|---|---:|---:|---|---|---|---|---|']
    cpt_of = {h['id']: h['cpt'] for h in HOST}
    for fam, fs in FAMILIES.items():
        cpt_of['kernel.fastasfuck.' + fam] = ', '.join(sorted({k['cpt'] for k in fs if k['cpt']}))
    cpt_of['kernel.fastasfuck.common'] = 'kernel.strixite.hip_common and others'
    cpt_of['format.fastasfuck.qn_device'] = 'format.strixite.q4 .. q8_device'
    for m in mods:
        files = O.expand(m.paths)
        lines = sum(len(open(O.find(f), errors='replace').read().splitlines()) for f in files)
        tags = sorted({p['interface'] for p in m.provides})
        ops = sorted({t.split('/')[0] for t in tags})
        others = sorted(x for x, ifs in prov.items() if x.split('.')[1] != 'strixite' and any(i.split('/')[0] in ops for i in ifs))
        ch = changed.get(m.id, [])
        o.append('| §%s§ | %s | %d | %d | %s | %s | %s | %s | %s |' % (
            m.id, m.lang, len(files), lines, m.compile, ('%dA %dM' % (ch.count('A'), ch.count('M'))) if ch else '-',
            '§%s§' % cpt_of[m.id] if cpt_of.get(m.id) else 'none', ', '.join('§%s§' % t for t in tags[:3]) + (' +%d' % (len(tags) - 3) if len(tags) > 3 else ''),
            (', '.join('§%s§' % x for x in others[:5]) + (', +%d more' % (len(others) - 5) if len(others) > 5 else '')) or '-'))
    kinds = {}
    for m in mods:
        kinds[m.kind] = kinds.get(m.kind, 0) + 1
    o += ['', '%d modules: %s.' % (len(mods), ', '.join('%d %s' % (v, k) for k, v in sorted(kinds.items())))]
    open(os.path.join(harvestlib.REPO, 'catalog', O.name, 'MODULES.md'), 'w').write('\n'.join(o).replace('§', q) + '\n')


LICENCE_FILES = ['LICENSE', 'THIRD_PARTY.md', 'third_party/cpp-httplib/LICENSE', 'third_party/libwebp/COPYING',
                 'third_party/libwebp/PATENTS']     # the same list as ORIGIN_TOML [licence] files

ORIGIN_TOML = '''
# fastasfuck: a C++20 + HIP inference engine for Qwen3.8-Flash-Next on one AMD Strix Halo (gfx1151, wave32 WMMA), with an
# OpenAI-compatible server (chat completions and the Responses API), MTP speculative decoding and up to 4 concurrent slots.
# A fork of strixite (AGENTS.md "Why We Forked"); its 25 kernel files are strixite's kernels, some renamed or changed
# (catalog/fastasfuck/KERNELS.md counts the changed lines of each). Card ORIGIN-FASTASFUCK 80ae224db4 harvested the kernels at
# main 35015b5; card FASTASFUCK-110 17eead1048 moved the pin to release 1.1.0 and harvested the engine around them
# (catalog/fastasfuck/MODULES.md). 1.1.0 changed no kernel: docs/harvest/data/fastasfuck/delta-35015b5.md.
name      = "fastasfuck"
repo      = "https://github.com/Cheblan/fastasfuck"
commit    = "73bd1703cbe04f1556b17a3c36bd05fd7ee0372e"
branch    = "1.1.0"
version   = "release tag 1.1.0 (73bd170) as pinned in the ark (repos.tsv, ark commit 9d833fc, 2026-11-25); the previous harvest pinned main 35015b5, and the delta against it is docs/harvest/data/fastasfuck/delta-35015b5.tsv (no kernel changed: GPTQ calibration tools, the converter's --prebuilt import, a calibration tag in the state fingerprint, docs)"
reference_checkout = "control-node:/data/ark-inbox/src/github.com__Cheblan__fastasfuck#73bd1703cbe04f1556b17a3c36bd05fd7ee0372e (ark copy, a depth-1 clone); laptop: ~/Code/strixflerp-upstream-supplement/fastasfuck-1.1.0 (that directory without .git)"
tree_sha256 = "@TREE@"   # sha256 of the sorted sha256sum listing of every file without .git (docs/harvest/harvest_fastasfuck.py)

[licence]
spdx      = "AGPL-3.0-only"
# copied verbatim next to this file at their origin paths: LICENSE, THIRD_PARTY.md and the vendored licence texts it names
files     = ["LICENSE", "THIRD_PARTY.md", "third_party/cpp-httplib/LICENSE", "third_party/libwebp/COPYING", "third_party/libwebp/PATENTS"]
source    = "the laptop copy above, byte-identical to the ark copy control-node:/data/ark-inbox/src/github.com__Cheblan__fastasfuck#73bd1703cbe04f1556b17a3c36bd05fd7ee0372e (checked 2026-11-26)"
note      = "README.md and its badge say AGPL-3.0 with no version clause; LICENSE is byte-identical to strixite's GNU AGPL v3 text and the kernels derive from strixite, which is AGPL-3.0-only, so the catalogue records AGPL-3.0-only. Copied verbatim with attribution, not relicensed. THIRD_PARTY.md says the engine was written from scratch; the kernel files say otherwise (a fork of strixite, whose THIRD_PARTY.md has the same sentence)."
third_party = "vendored and not harvested: cpp-httplib (MIT), stb (MIT OR Unlicense), libwebp decoder (BSD-3-Clause), Unicode tables (THIRD_PARTY.md); the cpp-httplib and libwebp licence texts THIRD_PARTY.md names ride beside this file, stb's licence is at the end of each header (not harvested). Model weights are not in the tree."

[toolchain]
host      = "C++20, CMake 3.28 or newer with Ninja (CMakePresets.json, preset strix, Release); the ROCm clang compiles host and device; a GCC 13-16 libstdc++"
device    = "therock-10.1"                               # README.md:140: tested with ROCm 10.1 / HIP 7.16 (Arch rocm-hip-sdk)

[compile]                                                # CMakeLists.txt:70-187: no hipcc, every kernels/*.hip is a CXX source with -x hip, Release flags
driver    = "<rocm>/lib/llvm/bin/clang++"
arch_flag = "--offload-arch=gfx1151"
rocm_flag = "--rocm-path=<rocm>"
flags     = ["-x", "hip", "-std=c++20", "-O3", "-march=znver5", "-DNDEBUG", "-D__HIP_PLATFORM_AMD__", "-Wall", "-Wextra", "-Wno-unused-parameter", "-c"]
system_includes = ["include"]
output    = "host object with the gfx1151 device code embedded (.o), one per kernels/*.hip"
note      = "Release adds -flto (host LTO); the gate leaves it out so each unit gives a plain object. The origin's include root is the repository root (target_include_directories), here each module's src."

[kernels]
loading   = "statically linked: the 25 kernel files form the static library strix_kernels (CMakeLists.txt:181-187); each .hip holds its device kernels and host launchers in namespace strix::kernels; no runtime kernel registry"
models    = "Qwen3.8-Flash-Next (and its vision tower), q4g128 / q6g64 / q8g64 weights in .faf files (docs/WEIGHTS.md); author's figures 73.8 tok/s solo, 104.4 tok/s over 4 slots (README.md); 1.1.0 adds a GPTQ-calibrated package of 65.51 GiB with -30.7% held-out KLD against round-to-nearest and 74.16 tok/s solo MTP (README.md); all unverified here"
'''

LEFT_OUT = {
    'third_party/': 'vendored cpp-httplib, stb, libwebp and the stb_impl unit: other licences (THIRD_PARTY.md), not fastasfuck code; serve.fastasfuck.http and runtime.fastasfuck.vision_preprocess include them as system headers',
    'deploy/': 'container and service files of the author\'s deployment',
    'docs/': 'documentation; docs/WEIGHTS.md (changed in 1.1.0) was read for ORIGIN.toml and the delta notes',
    'README.md, AGENTS.md, CONTRIBUTING.md': 'documentation, read for ORIGIN.toml',
    'CMakeLists.txt, CMakePresets.json, .gitignore, .dockerignore, .containerignore': 'build files; the kernel compile command is recorded in ORIGIN.toml [compile] and each host module\'s CMake target in its [gate] reason',
    'LICENSE, THIRD_PARTY.md, third_party/cpp-httplib/LICENSE, third_party/libwebp/{COPYING,PATENTS}': 'copied beside ORIGIN.toml',
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
    prov = providers()
    q = '\x60'
    o = ['# fastasfuck kernels by strixflerp interface', '',
         'One row for each of the 25 kernel files of §kernels/§ at 1.1.0 (%s; byte-identical to 35015b5). Cards ORIGIN-FASTASFUCK' % COMMIT[:7],
         '80ae224db4 and FASTASFUCK-110 17eead1048. Written by',
         '§docs/harvest/harvest_fastasfuck.py§; nothing here was run or timed. fastasfuck is a fork of strixite, so every file',
         'has a strixite counterpart, and the tag of a file is the interface classes that counterpart module provides.', '',
         '- **vs strixite**: lines that differ (difflib, unified, no context) from the counterpart file in §catalog/strixite§ at',
         '  %s. §0§ is byte-identical. The fork point is not known here, so a difference can be a fastasfuck change or a' % STRIXITE_COMMIT,
         '  strixite change made after the fork (catalog/strixite is v0.7.0 plus the pod\'s patches).',
         '- **other providers**: kernel modules of other origins that provide the same §<family>.<op>§.', '',
         '| file | lines | module | strixite counterpart | vs strixite | tag | other providers |', '|---|---:|---|---|---:|---|---|']
    n_same = n_mod = 0
    for fam, fs in FAMILIES.items():
        for k in fs:
            if not k['file'].endswith('.hip'):
                continue
            d = counterpart(k)
            n_same += d[0] == 0
            n_mod += d[0] != 0
            ops = sorted({i.split('/')[0] + '/' for i in k['ifaces']})
            others = sorted(m for m, ifs in prov.items() if m != k['cpt'] and m.startswith('kernel.') and any(i.startswith(op) for op in ops for i in ifs))
            classes = sorted({i.rsplit('/', 1)[0] + '/*' if i.split('/')[2] != '-' else i for i in k['ifaces']})
            o.append('| §%s§ | %d | §kernel.fastasfuck.%s§ | §%s§ §%s§ | %d | %s | %s |' % (
                k['file'], d[1], fam, k['cpt'], k['cfile'], d[0], ', '.join('§%s§' % c for c in classes),
                ', '.join('§%s§' % m for m in others) or '-'))
    o += ['', '25 kernel files: %d byte-identical to strixite, %d changed. Every file carries a tag.' % (n_same, n_mod), '',
          '## Headers', '', '| file | module | strixite counterpart | vs strixite |', '|---|---|---|---:|']
    for fam, fs in [('common', COMMON_FILES), ('qn_device', FORMAT_FILES)] + list(FAMILIES.items()):
        for k in fs:
            if k['file'].endswith('.hip'):
                continue
            d = counterpart(k)
            mid = ('format.fastasfuck.' if fam == 'qn_device' else 'kernel.fastasfuck.') + fam
            o.append('| §%s§ | §%s§ | %s | %s |' % (k['file'], mid, '§%s§ §%s§' % (k['cpt'], k['cfile']) if k['cpt'] else 'none', d[0] if d else 'new'))
    o += ['', '## What each kernel file does', '']
    for fam, fs in FAMILIES.items():
        for k in fs:
            if k['file'].endswith('.hip'):
                o.append('- §%s§: %s.' % (k['file'], k['what']))
    o += ['', '## Not done here', '',
          'No kernel was run or timed. The compile gate (§gate/compile/run.py§) compiles each family module for gfx1151 on',
          'control-node with ORIGIN.toml [compile]; its results are §gate/results/compile/kernel.fastasfuck.*.json§.',
          'A timing against the strixite modules is its own experiment card.']
    text = '\n'.join(o).replace('§', q) + '\n'
    open(os.path.join(harvestlib.REPO, 'catalog', O.name, 'KERNELS.md'), 'w').write(text)


if __name__ == '__main__':
    if '--tree' in sys.argv:
        print(tree_sha256(O.files()))
        sys.exit(0)
    if len(sys.argv) == 3 and sys.argv[1] == '--delta':
        write_delta(sys.argv[2])
        sys.exit(0)
    run(O, mods, ORIGIN_TOML.replace('@TREE@', TREE_SHA256), LEFT_OUT, LICENCE_FILES)
    kernels_md()
    owner = {f: m.id for m in mods for f in O.expand(m.paths)}
    delta = delta_md(owner)
    modules_md(owner, delta)
