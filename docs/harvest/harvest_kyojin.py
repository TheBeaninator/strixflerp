#!/usr/bin/env python3
"""Harvest Kyojin (Yamz-Labs/kyojin 116a883, ExLlamaV3 with ROCm / Strix Halo work) into catalog/kyojin.
Run: python3 docs/harvest/harvest_kyojin.py
     python3 docs/harvest/harvest_kyojin.py --base-diff <the base: ark copy of turboderp-org/exllamav3 8c593e4e>   # rewrite data/kyojin/*.tsv

Card ORIGIN-KYOJIN 692a13d6d0. Kyojin is a fork of turboderp-org/exllamav3. Only its delta over the base named in the card
(turboderp-org/exllamav3 master 8c593e4e, v1.6.7, the ark copy) is harvested, as ciru_iu4 did: data/kyojin/base-diff.tsv lists
every file Kyojin adds (A), changes (M) or removes (D) against that base (git blob ids), data/kyojin/tree.tsv the blob id of
every file of the pinned tree. The script checks the checkout against tree.tsv, that every A / M file inside SCOPE is in exactly
one module, that no module holds a file outside the delta except the unchanged base headers the delta's C / HIP files include
(kernel.kyojin.ext_base, so the families compile standalone), and derives module requires from #include lines.

The reference checkout is the ark copy on control-node (/data/ark-inbox/src/github.com__Yamz-Labs__kyojin#116a883...), a plain
tree whose leftover .git holds HEAD but no objects; name another copy of the same tree with KYOJIN_CHECKOUT. Nothing here
compiles or runs; gate.status = "builds" comes from gate/results/compile (gate/compile/run.py on control-node, CPU only)."""
import hashlib, json, os, re, sys, tomllib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvestlib
from harvestlib import Origin, M, P, run

ARK = '/data/ark-inbox/src/github.com__Yamz-Labs__kyojin#116a883ff213a842619bf633089b7e0083aa86ff'
CHECKOUT = os.environ.get('KYOJIN_CHECKOUT', ARK)
COMMIT = '116a883ff213a842619bf633089b7e0083aa86ff'
BASE_ARK = '/data/ark-inbox/src/github.com__turboderp-org__exllamav3'
BASE = '8c593e4e5c88da76a4aaf8f1b1838bb7c1fce4e0'
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'kyojin')
BASE_DIFF = os.path.join(DATA, 'base-diff.tsv')
TREE = os.path.join(DATA, 'tree.tsv')
EXT = 'exllamav3/exllamav3_ext/'
AUTHORS = ['Yamz Labs (Kyojin contributors)', 'turboderp (ExLlamaV3, MIT)']
NOT_TIMED = 'No timing was taken on the pod.'
SPDX = 'MIT'
TOOLCHAIN = 'therock-10.2'


def blob_id(path):
    data = open(path, 'rb').read()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def tree_blobs(root):
    out = {}
    for d, dirs, fs in os.walk(root):
        if d == root:
            dirs[:] = [x for x in dirs if x != '.git']
            fs = [x for x in fs if x != '.git']
        for f in fs:
            p = os.path.join(d, f)
            out[os.path.relpath(p, root)] = blob_id(p)
    return out


def git_head(root):
    """The commit a plain ark copy was cloned at: .git/HEAD, or the branch it names."""
    h = open(os.path.join(root, '.git', 'HEAD')).read().strip()
    if h.startswith('ref: '):
        ref = h[5:]
        p = os.path.join(root, '.git', ref)
        if os.path.isfile(p):
            return open(p).read().strip()
        for line in open(os.path.join(root, '.git', 'packed-refs')):
            if line.rstrip().endswith(' ' + ref):
                return line.split()[0]
    return h


def write_base_diff(base):
    for d, want in ((CHECKOUT, COMMIT), (base, BASE)):
        head = git_head(d)
        if head != want:
            raise SystemExit(f'{d} is at {head}, want {want}')
    ours, theirs = tree_blobs(CHECKOUT), tree_blobs(base)
    os.makedirs(DATA, exist_ok=True)
    with open(TREE, 'w') as f:
        f.write('# Yamz-Labs/kyojin %s: git blob id and path of every file (read from the ark copy, .git skipped).\n' % COMMIT)
        for p in sorted(ours):
            f.write('%s\t%s\n' % (ours[p], p))
    rows = []
    for p in sorted(set(ours) | set(theirs)):
        if p not in theirs:
            rows.append(('A', p, '-', ours[p]))
        elif p not in ours:
            rows.append(('D', p, theirs[p], '-'))
        elif ours[p] != theirs[p]:
            rows.append(('M', p, theirs[p], ours[p]))
    with open(BASE_DIFF, 'w') as f:
        f.write('# Yamz-Labs/kyojin %s against its base turboderp-org/exllamav3 %s (git blob ids).\n' % (COMMIT, BASE))
        f.write('# Written by docs/harvest/harvest_kyojin.py --base-diff; A added, M changed, D removed by Kyojin.\n')
        f.write('# base read from: plain tree %s (.git/HEAD ref master = %s)\n' % (base, BASE))
        f.write('status\tpath\tbase_blob\tkyojin_blob\n')
        for r in rows:
            f.write('\t'.join(r) + '\n')
    print(f'{BASE_DIFF}: {len(rows)} rows ({len(ours)} kyojin files, {len(theirs)} base files)')


def read_base_diff():
    rows = {}
    for line in open(BASE_DIFF):
        if line.startswith('#') or line.startswith('status\t'):
            continue
        st, path, _, kb = line.rstrip('\n').split('\t')
        rows[path] = (st, kb)
    return rows


def read_tree():
    return dict(reversed(l.rstrip('\n').split('\t', 1)) for l in open(TREE) if not l.startswith('#'))


class TreeOrigin(Origin):
    """The checkout must hold exactly the files of COMMIT with the blob ids of data/kyojin/tree.tsv (.git ignored)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.roots = [CHECKOUT]
        self.blobs = read_tree()
        have = self.files()
        missing, extra = sorted(set(self.blobs) - set(have)), sorted(set(have) - set(self.blobs))
        wrong = [p for p in have if p in self.blobs and blob_id(os.path.join(CHECKOUT, p)) != self.blobs[p]]
        if missing or extra or wrong:
            raise SystemExit(f'{CHECKOUT} is not the tree of {COMMIT}: missing {missing[:5]}, extra {extra[:5]}, differing {wrong[:5]}')
        if os.path.isfile(os.path.join(CHECKOUT, '.git', 'HEAD')) and git_head(CHECKOUT) != COMMIT:
            raise SystemExit(f'{CHECKOUT}/.git/HEAD is not {COMMIT}')

    def files(self):
        if self._files is None:
            out = []
            for d, dirs, fs in os.walk(CHECKOUT):
                if d == CHECKOUT:
                    dirs[:] = [x for x in dirs if x != '.git']
                    fs = [x for x in fs if x != '.git']
                out += [os.path.relpath(os.path.join(d, f), CHECKOUT) for f in fs]
            self._files = sorted(out)
        return self._files


# What is harvested: the A / M files under these patterns (Origin.expand syntax). Everything else of the delta is LEFT_OUT.
SCOPE = [EXT, 'exllamav3/gm_ext/', 'exllamav3/modules/attention_fn/', 'exllamav3/modules/dsv41_fn/', 'exllamav3/modules/gr_fn/',
         'exllamav3/modules/ple_fn/', 'exllamav3/modules/gated_delta_net_fn/', 'exllamav3/vendor/fla/', 'exllamav3/loader/',
         'exllamav3/generator/', 'exllamav3/cache/', 'exllamav3/foreign_quant/', 'exllamav3/dsv41/', 'hipsp/',
         'exllamav3/util/hip_compiler.py', 'exllamav3/util/hip_lib.py', 'exllamav3/util/kernel_seed.py', 'exllamav3/util/jit_log.py',
         'exllamav3/util/tune_cache.py', 'exllamav3/util/knobs.py', 'exllamav3/util/row_inv.py', 'exllamav3/util/disk_store.py',
         'exllamav3/util/attention_mode.py', 'exllamav3/bridge.py', 'exllamav3/serve_split.py', 'exllamav3/device_map.py',
         'exllamav3/ext_fallbacks.py', 'exllamav3/guard.py', 'exllamav3/guard_common.py', 'exllamav3/memory_guard.py',
         'exllamav3/model/cuda_graph.py', 'exllamav3/model/host_table.py', 'exllamav3/model/dense_warmup.py',
         'exllamav3/model/dense_gemm_tune_seed.txt', 'exllamav3/model/model_tp*.py', 'exllamav3/model/moe_cpu_host.py',
         'exllamav3/modules/ablit_runtime.py', 'exllamav3/modules/block_graph.py', 'exllamav3/modules/draft2s.py',
         'exllamav3/modules/dsv41.py', 'exllamav3/modules/expert_store.py', 'exllamav3/modules/foreign_linear.py',
         'exllamav3/modules/fq_moe_prefill*.py', 'exllamav3/modules/hip_glue.py', 'exllamav3/modules/kpool_indexer.py',
         'exllamav3/modules/moe_fused.py', 'exllamav3/modules/moe_mixedk.py', 'exllamav3/modules/qwd3_pf.py',
         'exllamav3/modules/router_topk.py', 'exllamav3/modules/row_split.py',
         'tools/glm/serve.py', 'tools/glm/serve_*.py', 'tools/glm/image_cache*.py', 'tools/glm/SERVE.md',
         'tools/qwen/serve.py', 'tools/qwen/serve_metrics.py', 'tools/qwen/serve_ping.py', 'tools/qwen/serve_reqlog.py',
         'tools/qwen/image_cache.py', 'tools/qwen/SERVE.md', 'tools/qwen/spec_policy.py',
         'tools/hy4/serve.py', 'tools/hy4/serve_core.py', 'tools/hy4/engine.py', 'tools/hy4/mu_engine.py', 'tools/hy4/draft_policy.py',
         'tools/hy4/depth_gate.py', 'tools/hy4/ngram.py', 'tools/hy4/spec2/',
         'tools/kyojin_serve_split.py', 'tools/loopguard.py', 'tools/serve_compat.py', 'tools/l21_prompt_encode_cache.py',
         'tools/startup_health.py', 'tools/admin_knobs.py', 'tools/slot_check.py']
INC_ROOTS = [EXT.rstrip('/')]          # setup.py's include root of the extension; quoted includes resolve next to the file first
TORCH = re.compile(r'^(torch/|ATen/|c10/|pybind11/|Python\.h|cublas|cuda_runtime|cuda_fp16|cuda_bf16|mma\.h|cuda\.h|nvrtc)')
C_LIKE = ('.c', '.cc', '.cpp', '.h', '.hpp', '.hip', '.inc', '.cu', '.cuh')
JIT = ['-x', 'hip', '-std=c++17', '-O3', '-ffp-contract=off', '--offload-device-only', '-c']


def K(id, paths, tag, summary, why, numeric='fp16 / bf16 operands, fp32 accumulation (file comments)', regime='gfx1151 serving path',
      **kw):
    """One module of the table. paths: Origin.expand patterns over the origin. tag: the strixflerp interface it matches (full
    name with a .kyojin-* variant tag) or None. Keywords: lang, jit (the origin compiles it with hipcc --genco at first use:
    a device-only code object), spdx, upstream, notes, sources (override), cuda (CUDA-only source), host (host code)."""
    return dict(id=id, paths=paths, tag=tag, summary=summary, why=why, numeric=numeric, regime=regime, **kw)


X = lambda *names: [EXT + n for n in names]
MODS = [
 # ---- shared headers --------------------------------------------------------------------------------------------------
 K('kernel.kyojin.ext_portability', X('compat.cuh', 'compat_rocm.cuh', 'port.cuh', 'ptx.cuh', 'util.cuh', 'reduction.cuh', 'pad_pitch.h', 'hip/hip_mma.cuh'),
   None, 'The portability layer the Kyojin kernels share: port.cuh (one-source HIP gfx11.5 / CUDA sm_86 macros), compat_rocm.cuh (maskless shuffles and the CUDA-to-HIP spellings), hip/hip_mma.cuh (the HIP WMMA fragment vocabulary and mma_ab_h_hip, gfx11 and gfx12 layouts), ptx.cuh, util.cuh, reduction.cuh and pad_pitch.h (row-padded activation index arithmetic shared with a CPU bounds proof)',
   'every kernel family below includes these; a strixflerp adapter that lifts a Kyojin kernel needs this layer or its own equivalent', 'n/a here: headers', 'compiled inside every kernel.kyojin.* unit'),
 # ---- elementwise, norm, rope, activation -----------------------------------------------------------------------------
 K('kernel.kyojin.activation', X('activation.cu', 'activation_rocm.cu'), 'act.swiglu/-/f32.kyojin-ext',
   'Gated activations of the extension (silu(x) * y and the gelu / relu2 forms, in place when the output aliases an input) and activation_rocm.cu, a ROCm-only x * sigmoid(y @ w) + z gate',
   'the MLP and shared-expert activations of every model Kyojin serves; the ROCm file is Kyojin\'s', 'fp16 / bf16 in and out, fp32 math'),
 K('kernel.kyojin.fused_elt', X('fused_elt.cuh', 'fused_elt_rocm.cu'), 'norm.rmsnorm-gated/-/bf16.kyojin-fused-elt',
   'ROCm-only fused elementwise ops: each replaces a torch kernel chain of ext_fallbacks.py with one launch and the same per-element arithmetic (rmsnorm(x [* act(g)]) * (w + bias), sigmoid / silu gates), no fast-math intrinsics',
   'on ROCm the base extension leaves these to torch chains of 5-8 launches; one launch each is a decode-step launch-count lever', 'bf16 / f32 inputs, fp32 math, bf16 round-to-nearest-even out (c10::BFloat16 semantics)'),
 K('kernel.kyojin.elementwise', X('add.cu', 'softcap.cu', 'histogram.cu'), 'residual.add/-/f32.kyojin-ext',
   'Small elementwise kernels of the extension that Kyojin changed: residual add (with dtype mixes), logit soft-capping and a value histogram',
   'changed for the ROCm build; small but on every step', 'fp16 / bf16 / fp32 operands'),
 K('kernel.kyojin.norm', X('norm.cu'), 'norm.rmsnorm/-/f32.kyojin-ext',
   'RMS norm and gated RMS norm of the extension (warp-shuffle reductions across lanes and warps), with Kyojin\'s ROCm changes',
   'every layer; the gated form is the GDN / KDA output norm', 'fp16 / bf16 in, fp32 reduction'),
 K('kernel.kyojin.rope', X('rope.cu'), 'pos.rope/-/f32.kyojin-ext',
   'Rotary position embedding of the extension (neox / gptj styles, partial rotary, fused q / k), with Kyojin\'s ROCm changes',
   'every attention layer', 'fp16 q / k, fp32 sin / cos math'),
 K('kernel.kyojin.attention_ext', X('attention.cu'), None,
   'The extension\'s attention helpers (attention.cu) as Kyojin changed them: the small attention kernels and cache plumbing the libtorch attention blocks launch',
   'the eager side of the graph-captured attention blocks (runtime.kyojin.bc_attention)'),
 K('kernel.kyojin.routing', X('routing.cu', 'routing.cuh', 'routing_std_gfx12.cu', 'routing_std_gfx12.cuh'), 'moe.router/-/f32.kyojin-ext',
   'MoE routers: the extension\'s top-k routing kernels (softmax / sigmoid-bias / grouped top-k, normalisation and scaling) and a gfx12 standard-router variant',
   'every MoE layer\'s router; GLM-5.3-Flash and Qwen3.8-Flash-Next route through it', 'fp32 logits and weights'),
 K('kernel.kyojin.ple', X('ple.cu', 'ple.cuh'), 'ple.inject/-/f32.kyojin',
   'PLE stream gate of Qwen3.8-Flash-Next: out = sigmoid(signed_sqrt(gate * gate_scale)) * value, the shared value row broadcast over the hyper-connection streams',
   'the per-layer-embedding injection of the Flash-Next trunk', 'fp32 gate math, fp16 / bf16 rows'),
 K('kernel.kyojin.softmax_sampling', X('generator/sampling_basic.cu', 'generator/sampling_extra.cu', 'generator/sampling_fused.cu', 'generator/gumbel.cu'),
   'sample.next-token/-/f32.kyojin',
   'On-device sampling: argmax / top-k / top-p / min-p basics, a fused terminal sampling step that computes temperature / min-P / Gumbel pipelines in one pass over the logits without materialising a softmax, and the Gumbel noise kernel',
   'sampling stays on the device so a captured decode graph needs no host round trip per token', 'fp32 logits'),
 K('kernel.kyojin.penalties', X('generator/rep_pen.cu', 'generator/dry.cu'), 'logits.bias/-/-.kyojin-penalties',
   'Repetition penalties on the device: frequency / presence / repetition penalty over a token window and DRY (llama.cpp llama_sampler_dry_apply semantics, single-token breakers) in one launch per step',
   'keeps penalty sampling inside the captured step', 'fp32 logits'),
 K('kernel.kyojin.page_copy', X('generator/cache.cu'), 'cache.snapshot-copy/-/-.kyojin',
   'Paged KV cache page copy and rotation kernels (16-byte aligned per-CTA chunks; small DSA pools of 1792 B pages)',
   'prefix reuse, forks and defragmentation of the paged cache', 'byte copies'),
 K('kernel.kyojin.strings', X('generator/strings.cpp'), 'text.output-parser/-/-.kyojin-stop-strings',
   'Stop-string matcher of the generator: compares a UTF-32 tail against a packed list, reporting full, partial (overlapping the end) or no match',
   'stop strings without a Python loop per token', 'n/a', 'host', host=True),
 # ---- kv cache -------------------------------------------------------------------------------------------------------
 K('kernel.kyojin.kvcache_quant', X('cache/q_cache.cu', 'cache/q_cache_kernels.cuh', 'cache/lmq.cuh'), 'kvcache.quant8/-/f32.kyojin-hadamard',
   'Quantized KV cache writer / reader of the extension as Kyojin changed it: Hadamard-rotated (H4 in registers, H8 across the subgroup) 2..8-bit groups with fp16 scales, and lmq.cuh\'s cubic codebook (encode / decode in a few FMAs)',
   'the quantized paged KV cache of the Kyojin servers (CUDA build; on ROCm the Triton writer in kernel.kyojin.triton_qwrite stands in)', 'fp16 K / V in, packed 2..8-bit codes plus an fp16 scale per 32-value group'),
 # ---- attention selection / sparse -------------------------------------------------------------------------------------
 K('kernel.kyojin.dsa_topk', X('dsa_topk.cu', 'dsa_topk.cuh'), 'attn.indexer-topk/-/i32.kyojin-radix',
   'DSA / QSA top-k selection: a descending radix bucket search per row (one block per row) giving ascending indices then -1 padding',
   'the selection step of DeepSeek sparse attention and Qwen QSA; GLM-5.3-Flash\'s DSA layers select through it', 'fp16 scores, int32 indices'),
 K('kernel.kyojin.dsa_topk_gfx12', X('dsa_topk_gfx12.cu', 'dsa_topk_gfx12.cuh'), 'attn.indexer-topk/-/i32.kyojin-gfx12',
   'QSA selection rewritten for RDNA (the ROCm path of ext.dsa_topk): fp16 (R, T) rows read from memory once, k <= 512 into int32, bitwise identical to dsa_topk; one 1024-thread block a row, four sweeps (coarse and fine histogram, count, write)',
   'the ROCm default of the selection kernel', 'fp16 scores, int32 indices'),
 K('kernel.kyojin.dsa_topk_par', X('dsa_topk_par.cu', 'dsa_topk_par.cuh'), 'attn.indexer-topk/-/i32.kyojin-par',
   'QSA selection with many blocks per row, bitwise identical to dsa_topk_gfx12: the row sweeps run on the whole GPU, which a decode-batch-1 selection over a 32K..512K pool row needs',
   'long-context decode: one block per row leaves the GPU idle at batch 1', 'fp16 scores, int32 indices', 'decode, long context'),
 K('kernel.kyojin.mla_dec_valu', X('mla_dec_valu.cu', 'mla_dec_valu.cuh'), 'attn.sparse-sink/kv-latent/bf16.kyojin-valu-absorb',
   'MLA decode absorb / unfold for up to 8 rows on VALU (ROCm only), bit-identical to the Triton WMMA kernels of attention_fn/mla_triton.py (v_wmma_f32_16x16x16_f16 equals a chain of 8 v_dot2_f32_f16 over ascending k)',
   'GLM-5.3-Flash MLA decode: the few-row absorb is cheaper on VALU than padding to a WMMA tile', 'fp16 operands, fp32 accumulation in the WMMA order', 'decode, R <= 8 rows'),
 K('kernel.kyojin.dsv4_compress', X('dsv4_compress.cu'), 'kvcache.compress/-/f32.kyojin',
   'DeepSeek-V4 KV compressor kernels (the compressed-attention pool update) as Kyojin changed them',
   'DeepSeek V4 / V4.1 compressed attention; kin of the catalogue\'s kvcache.compress providers', 'fp32 state, bf16 / fp16 rows'),
 K('kernel.kyojin.dsv4_pool_quant', X('dsv4_pool_quant.cu'), 'kvcache.store/fp8-e4m3/f32.kyojin-pool',
   'DeepSeek-V4 compressed pool quantization (pool rows to fp8 with scales)', 'the DSV4 pool cache footprint', 'fp8 e4m3 rows with fp32 scales'),
 # ---- recurrent mixers -----------------------------------------------------------------------------------------------
 K('kernel.kyojin.gdn', X('gdn.cu', 'gdn.cuh'), 'mixer.gdn/-/f32.kyojin-ext',
   'Gated DeltaNet recurrent kernels of the extension, much extended by Kyojin (2,160 lines): register-resident-state decode variants, multi-row verify, state rewind for speculative rounds, and timing-only ablation arms',
   'the linear-attention layers of Qwen3.8-Flash-Next and Kimi-Linear decode through it', 'fp32 state, bf16 / fp16 q / k / v', 'decode and verify'),
 K('kernel.kyojin.kda_inter', X('kda_inter.cu', 'kda_inter.cuh'), 'mixer.kda/-/f32.kyojin-inter',
   'KDA inter-solve off-diagonal dots (kdaC7): one program per (head, 64-row chunk), full-cache-line reads, gated operands a = x * exp2(g_x - g_n), bitwise equal to vendor/fla/kda_inter_split.py\'s inter_off_kernel',
   'Kimi delta attention chunked prefill', 'bf16 operands, fp32 dots', 'prefill'),
 K('kernel.kyojin.dflash2', X('dflash2.cu', 'dflash2.cuh'), 'speculation.draft/dflash/f32.kyojin-conv',
   'DFlash2 drafter kernels ported from exllamav3 1.6.0: the grouped dynamic causal convolution over a draft block (out[b, t, c] = sum_k (base[k, c] + dyn[b, t, k, c / group]) * x[b, t - k, c], taps within the block)',
   'the DFlash2 block drafter of speculative decoding', 'fp16 / bf16 in, fp32 accumulation', 'decode, draft block',
   upstream='ported from exllamav3 1.6.0 (MIT, turboderp)'),
 K('kernel.kyojin.hc_mix', X('hc_mix.cu', 'hc_mix.cuh', 'hc_mix_q8r.cuh'), 'hc.mix/-/f32.kyojin',
   'Fused mHC hyper-connection mix and HyperHead collapse (2,432 lines), plus hc_mix_q8r.cuh: a one-pass int8 gated-residual mix for 2..4 rows (decode verify / MTP) that fetches each weight byte once instead of R times',
   'hyper-connection models (DeepSeek V4.x, Hy4-flash, Qwen gated residual) mix streams every layer; the R-row int8 form is a verify-step lever', 'fp32 mix math; int8 weights with scales in the q8r form'),
 K('kernel.kyojin.ngram', X('ngram.cu', 'ngram.cuh', 'ngram_gather_win.cpp'), 'ple.ngram-rows/exl3-ngram/-.kyojin',
   'N-gram embedding rows: device gather of hashed n-gram rows, and the scattered disk gather of rows on Windows (overlapped ReadFile at high queue depth)',
   'n-gram embedding tables (Engram-style) too large to hold resident', 'rows copied verbatim'),
 # ---- dense GEMM -----------------------------------------------------------------------------------------------------
 K('kernel.kyojin.hgemm', X('hgemm.cu', 'hgemm.cuh', 'hgemm_f16acc.cu'), 'linear.prefill/f16/f16-f32.kyojin-hgemm',
   'Dense fp16 GEMM of the extension with Kyojin\'s row-padded operand support (a non-power-of-two row pitch such as K=6144 rows of 12288 B) and hgemm_f16acc.cu, an fp16-accumulator tensor-core GEMM flushing to fp32 across K (a GeForce rate lever)',
   'dense projections; the padded-pitch form avoids channel conflicts', 'fp16 operands, fp32 accumulation (f16acc: fp16 partials folded to fp32 per k-slice)'),
 K('kernel.kyojin.moe_unswizzle', X('moe_unswizzle.cu', 'moe_unswizzle.cuh'), None,
   'Unswizzle of CPU-offloaded expert weights streamed back to the GPU for prefill (the band-swizzled VBMI tile order to the GPU order)',
   'pairs with the CPU expert offload (runtime.kyojin.cpu_moe_mul1)', 'byte permutation'),
 # ---- EXL3 quant kernels ---------------------------------------------------------------------------------------------
 K('kernel.kyojin.exl3_codec', X('quant/bits_k.cuh', 'quant/codebook.cuh', 'quant/exl3_dq.cuh', 'quant/exl3_devctx.cu', 'quant/exl3_devctx.cuh', 'quant/util.cu'),
   'format.device-view/exl3/-.kyojin',
   'EXL3 trellis decode primitives as Kyojin changed them: bit-window extraction (exl3_dq.cuh), the 3INST / mcg / mul1 codebooks, the per-device context (SM count, locks, the MoE expert scheduler tickets) and quant utilities',
   'every EXL3 kernel below decodes weights through these; the codebooks define the EXL3 numerics', 'trellis codes decoded to fp16 through the codebook hash'),
 K('kernel.kyojin.exl3_kernel_map', X('quant/exl3_kernel_map.cu', 'quant/exl3_kernel_map.cuh'), 'dispatch.kernel-table/exl3/-.kyojin',
   'Kernel selection table of the EXL3 GEMM: (K bits, codebook, shape class) to kernel instance and launch geometry', 'which EXL3 kernel serves a shape', 'n/a'),
 K('kernel.kyojin.exl3_gemm', X('quant/exl3_gemm.cu', 'quant/exl3_gemm.cuh', 'quant/exl3_gemm_inner.cuh', 'quant/exl3_gemm_kernel.cuh'),
   'linear.prefill/exl3/f16-f32.kyojin',
   'EXL3 GEMM (trellis decode fused into the MMA tiles) with Kyojin\'s changes, including fp16 accumulation folded into fp32 once per k-slice for m=1',
   'the EXL3 dense projections at prefill and multi-row verify', 'fp16 activations, fp32 accumulation'),
 K('kernel.kyojin.exl3_gemv', X('quant/exl3_gemv.cu', 'quant/exl3_gemv.cuh', 'quant/exl3_gemv_kernel.cuh'), 'linear.decode/exl3/f16-f32.kyojin-qtip',
   'Small-m EXL3 GEMV, QTIP-style on the unmodified EXL3 format: warps split k and never synchronise in the main loop, B streams straight to registers; launched from exl3_gemm when the heuristic applies',
   'decode-bound projections at batch 1..few', 'fp16 activations, fp32 accumulation', 'decode, small m'),
 K('kernel.kyojin.exl3_gemv_int8', X('quant/exl3_gemv_int8.cu', 'quant/exl3_gemv_int8.cuh', 'quant/exl3_gemv_int8_kernel.cuh', 'quant/comp_units/exl3_gemv_int8_*'),
   'linear.decode/exl3/f32.kyojin-int8',
   'Fused int8-activation GEMV for mul1 (cb 2) tensors: one cooperative launch covering the input Hadamard, activation quantization, dp4a GEMV and output Hadamard; plain int8 or an error-feedback residual mode (~15-16 bit effective activations); per-K instances in comp_units',
   'decode GEMV at int8 rate; opt-in on ROCm until measured', 'int8 activations (with an optional residual pass), dp4a, fp32 accumulation', 'decode'),
 K('kernel.kyojin.exl3_dec', X('quant/exl3_dec.cu', 'quant/exl3_dec.cuh'), 'linear.decode/exl3/f32.kyojin-dec-rdna35',
   'Batch-1 EXL3 decode kernels for RDNA3.5 (gfx1151, 3,431 lines): one lane owns one 16x16 trellis tile, the bit windows sit at compile-time positions (fully unrolled decode, no LDS staging, no shuffles), fp32 accumulation through v_dot2_f32_f16, the input Hadamard fused',
   'Kyojin\'s Strix Halo batch-1 decode GEMV; the shape a strixflerp linear.decode/exl3 provider for gfx1151 would take', 'fp16 activations, v_dot2_f32_f16 into fp32', 'decode, batch 1'),
 K('kernel.kyojin.exl3_gemv_rdna', [EXT + 'rocm/quant/'], 'linear.decode/exl3/f16-f32.kyojin-rdna',
   'RDNA tiles of the EXL3 GEMV (rocm/quant: the wave32 kernel, its tile loaders and launch front end)', 'the ROCm GEMV path before exl3_dec', 'fp16 activations, fp32 accumulation', 'decode'),
 K('kernel.kyojin.exl3_moe', X('quant/exl3_moe.cu', 'quant/exl3_moe.cuh', 'quant/exl3_moe_common.cuh', 'quant/exl3_moe_kernel.cuh', 'quant/comp_units/exl3_moe_inst_*', 'quant/comp_units/exl3_moe_instances.cuh'),
   'moe.experts-decode/exl3/f16-f32.kyojin',
   'EXL3 MoE expert kernels: input Hadamard, gate / up GEMMs, activation, down GEMM and output Hadamard per (K, codebook, N tile, M tile) instance, with M_TILE 32 / 64 instances for experts with many tokens and the [count_lo, count_hi] split',
   'routed experts of the EXL3 packs (GLM-5.3-Flash-EXL3-Yamz, Qwen3.8-Flash-Next-EXL3-Yamz)', 'fp16 activations, fp32 accumulation'),
 K('kernel.kyojin.exl3_moe_valu', X('quant/exl3_moe_valu.cuh', 'quant/exl3_moe_valu_idx.h'), 'moe.experts-decode/exl3/f32.kyojin-valu',
   'VALU grouped-MoE decode matvec for gfx11.5 (default on for K1 / K2 / K3 experts): replaces the WMMA / LDS body of the grouped and dedup decode launches; index arithmetic in a pure C++ header shared with a CPU bounds checker',
   'decode experts at batch 1 are matvecs: VALU beats a padded WMMA tile', 'fp16 activations, v_dot2 into fp32', 'decode'),
 K('kernel.kyojin.exl3_moe_coop', X('quant/exl3_moe_coop.cu', 'quant/exl3_moe_coop.cuh', 'quant/exl3_moe_coop_kernel.cuh', 'quant/comp_units/exl3_moe_coop_instances.cuh'),
   'moe.experts-decode/exl3/f16-f32.kyojin-coop',
   'Fused decode-shaped MoE (bsz 1..MAX_BSZN): two ordinary launches per layer run every (token, expert) slot through gate / up, activation and down, replacing the per-projection mgemm graphs',
   'launch count of the MoE decode step', 'fp16 activations, fp32 accumulation', 'decode'),
 K('kernel.kyojin.exl3_coop_autotune', X('quant/coop_autotune.cu', 'quant/coop_autotune.cuh'), 'tune.table/hip/-.kyojin-coop',
   'Autotuner of the cooperative EXL3 GEMM: times candidate kernels on the first call of every (row class, K, N, bits, dtype, device) and caches the winner (runtime.kyojin.tune_cache persists it per pack)',
   'day-0 tuning of the GEMM per pack and device', 'n/a'),
 K('kernel.kyojin.exl3_moe_fused', X('quant/exl3_moe_fused.cu', 'quant/exl3_moe_fused.cuh', 'quant/exl3_moe_fused_api.h', 'quant/exl3_moe_fused_idx.h'),
   'moe.experts-decode/exl3/f32.kyojin-fused-half',
   'Fused MoE half-layer decode for gfx11.5 (opt-in EXL3_MOE_FUSED): one persistent launch per half-layer for 1..4 rows covering hyper-connection dots, finalize / mix, router, top-k and expert grouping, gate / up (VALU trellis matvec), silu * up, down and the Hadamard; every index through a pure C++ plan shared with a CPU bounds checker',
   'the whole MoE half of a decode layer in one launch: the launch-bound Strix Halo decode step\'s biggest lever', 'fp16 activations, fp32 accumulation', 'decode, 1..4 rows'),
 K('kernel.kyojin.exl3_moe_fused7', X('quant/exl3_moe_fused7*'), 'moe.experts-decode/exl3/f32.kyojin-fused7',
   'core7 of the fused MoE half-layer: front end rewritten for memory-level parallelism (every global load of a work item issued before its first use), same arithmetic order and byte-identical results as core5; compile-time ablation variants',
   'a memory-latency rewrite of the fused half-layer', 'fp16 activations, fp32 accumulation', 'decode'),
 K('kernel.kyojin.exl3_moe_fused9', X('quant/exl3_moe_fused9*'), 'moe.experts-decode/exl3/f32.kyojin-fused9',
   'core9: the mf7 kernel plus cache touches (read-only prefetch loads whose results are dropped) for finalize, router and shared-expert rows, with always-inline copies of the outlined phases',
   'prefetching the next phase\'s weights while the current one computes', 'fp16 activations, fp32 accumulation', 'decode'),
 K('kernel.kyojin.exl3_moe_grouped', X('quant/exl3_moe_grouped.cu', 'quant/exl3_moe_grouped.cuh', 'quant/exl3_moe_gemm.cuh'), 'moe.experts-prefill/exl3/f16-f32.kyojin-grouped',
   'Grouped prefill tier of the fused MoE: token-parallel Hadamards and a grouped GEMM where one CTA owns one (expert, row tile, 128-column tile) output block, for experts with token counts in [count_lo, count_hi]',
   'mid-size expert batches at prefill', 'fp16 activations, fp32 accumulation', 'prefill'),
 K('kernel.kyojin.exl3_moe_prefill', X('quant/exl3_moe_prefill.cu', 'quant/exl3_moe_prefill.cuh'), 'moe.experts-prefill/exl3/f16-f32.kyojin-wmma',
   'Grouped EXL3 MoE prefill for gfx11.5 WMMA (2,027 lines): tokens sorted by expert, every projection one launch over all experts (8 launches a layer, independent of the expert count), trellis tiles decoded into LDS inside the GEMM, gate + up in one launch',
   'Kyojin\'s Strix Halo MoE prefill (the 580-633 tok/s GLM-5.3-Flash prefill rides on it)', 'fp16 activations, WMMA f32 accumulation', 'prefill'),
 K('kernel.kyojin.exl3_quantize', X('quant/quantize.cu', 'quant/quantize.cuh', 'quant/quantize_tiles_kernel.cuh', 'quant/quantize_tiles_optimized.cuh', 'quant/quantize_tiles_rdna.cuh', 'quant/comp_units/quantize_tiles_instances.cuh'),
   'format.convert/exl3/-.kyojin-trellis-quantize',
   'EXL3 trellis quantizer (Viterbi over tail-biting rings): the templated kernel, sm_120-tuned dense specialisations and quantize_tiles_rdna.cuh, a grouped gfx11 quantizer for K = 2..4 giving the same indices',
   'conversion-time kernels; the RDNA form lets a Strix Halo quantize EXL3 itself', 'half arithmetic, first-minimum tie rule', 'conversion'),
 K('kernel.kyojin.exl3_hadamard', X('quant/hadamard.cu', 'hadamard.cpp'), 'rotate.fwht/-/f32.kyojin',
   'Hadamard rotations of EXL3 (device butterflies and the host Paley construction of non-power-of-two sizes)', 'the input / output rotations of every EXL3 projection', 'fp16 / fp32'),
 K('kernel.kyojin.exl3_pack', X('quant/pack.cu', 'quant/reconstruct.cu'), 'format.convert/exl3/-.kyojin-pack',
   'EXL3 tile packing and reconstruction (packed 16x128 tiles back to fp16 weights)', 'conversion and the dequant fallback', 'trellis codes to fp16'),
 # ---- llama.cpp-derived ROCm MMQ and routed F16 ------------------------------------------------------------------------
 K('kernel.kyojin.gmmq', [EXT + 'rocm/gmmq/'], 'moe.experts-prefill/gguf-kquant/q8_1-f32.kyojin-gmmq',
   'Routed (MoE) HIP MMQ for GGUF Q4_K / Q5_K / Q5_1 / Q8_0: out[t * n_used + s] = dequant(W[ids[t, s]]) @ x[t] with q8_1 activations, adapted from Gufo 07e1191\'s mmq (itself llama.cpp 5c0e946 mmq / mmid / quantize), one mul_mat_q_case per translation unit, with a torch binding',
   'GGUF MoE prefill (Hy4-flash, generic GGUF MoE); compare the gufo mmq it came from', 'q8_1 activations, integer MMA, fp32 out', 'prefill',
   upstream='adapted from Gufo 07e1191 src/models/qwen38_flash_next/kernels/rocm/mmq (MIT), itself from llama.cpp 5c0e946 ggml-cuda mmq / mmid / quantize (MIT); every file keeps its licence header'),
 K('kernel.kyojin.routed_f16', [EXT + 'rocm/routed_f16/'], 'moe.experts-prefill/gguf-kquant/f16-f32.kyojin-routed-f16',
   'Routed F16-WMMA prefill MoE for gfx1151: RoutedPairF16 / RoutedDownF16 / pad-bounds / scatter kernels and the MoE epilogue (combine with the shared-expert term), adapted from Gufo',
   'GGUF MoE prefill through dequant to fp16 tiles and WMMA', 'fp16 WMMA, fp32 accumulation', 'prefill',
   upstream='adapted from Gufo 07e1191 kernels.hip.cpp (RoutedPairF16Kernel, RoutedDownF16Kernel, RoutedPadBoundsKernel, RoutedScatterKernel, MoeEpilogueVec4Kernel; MIT)'),
 # ---- foreign quant (GGUF / NVFP4 / FP8 weights on the EXL3 engine) ----------------------------------------------------
 K('kernel.kyojin.fq_common', X('foreign_quant/fq_api.h', 'foreign_quant/fq_mmvq_api.h', 'foreign_quant/fq_tables.cuh', 'foreign_quant/fq_iq3_tables.cuh', 'foreign_quant/fq_dequant.cuh', 'foreign_quant/fq_gemv.cuh', 'foreign_quant/fq_kq_item.cuh', 'foreign_quant/fq_kq_wmma.cuh'),
   'format.device-view/gguf-kquant/-.kyojin-fq',
   'Shared layer of the foreign-quant kernels (GGUF k-quants and IQ formats, NVFP4, FP8 on the EXL3 engine): host entry points, IQ2_XXS / IQ3_XXS tables (verbatim llama.cpp), device-inline block decoders, lane dot products, the Q4_K / Q6_K v2 items and the gfx11 WMMA multi-row path',
   'lets Kyojin run GGUF weights unconverted (Hy4-flash Q3_K, generic GGUF); every fq_* kernel includes it', 'bit-exact block decode, fp32 dots', 'decode and prefill'),
 K('kernel.kyojin.fq_dequant', X('foreign_quant/fq_dequant.cu', 'foreign_quant/fq_dequant_kq.cu'), 'format.decode/gguf-kquant/f16.kyojin',
   'Foreign-quant dequant to fp16: Q8_0 / IQ2_XXS / NVFP4 (streaming 2D grids) and the k-quants Q4_K / Q5_K / Q6_K / IQ4_XS for the unfused batched-prefill path (dequant then fp16 GEMM)',
   'the prefill weight path for foreign formats', 'exact dequant to fp16', 'prefill'),
 K('kernel.kyojin.fq_gemm', X('foreign_quant/fq_gemm.cu'), 'linear.decode/gguf-iquant+q8/f32.kyojin-fq-gemm',
   'Fused dequant-GEMV / GEMM for foreign-quant weights (M = 1..16): weights dequantised on the fly with the bit-exact block arithmetic', 'the first decode path of foreign formats', 'fp32 accumulation, fp16 out', 'decode'),
 K('kernel.kyojin.fq_gemv', X('foreign_quant/fq_gemv.cu'), 'linear.decode/gguf-iquant+q8/f32.kyojin-fq-gemv',
   'Cooperative decode GEMV for foreign-quant weights (M = 1..16): threads share a row\'s weight stream instead of each reading a whole dot product', 'decode GEMV bandwidth for GGUF weights', 'fp32 accumulation', 'decode'),
 K('kernel.kyojin.fq_gemv_fp8', X('foreign_quant/fq_gemv_fp8.cu'), 'linear.decode/fp8-e4m3/f32.kyojin',
   'FP8 E4M3 decode GEMV in C++ (wave32 on RDNA) replacing a Triton GEMV that streamed only 42-55 GB/s on gfx1151', 'fp8 checkpoints at decode', 'fp8 weights decoded to fp32, fp32 accumulation', 'decode'),
 K('kernel.kyojin.fq_gemv_q8', X('foreign_quant/fq_gemv_q8.cu'), 'linear.decode/gguf-q8_0/f32.kyojin-q8_1',
   'q8_1-activation decode kernels (llama.cpp mmvq design): activations quantized once per row into a cached scratch, integer dp4a dots against Q8_0 / IQ weights', 'integer-rate decode for GGUF weights', 'q8_1 activations, dp4a, fp32 out', 'decode'),
 K('kernel.kyojin.fq_gemv_kq', X('foreign_quant/fq_gemv_kq.cu'), 'linear.decode/gguf-kquant/f32.kyojin-q8_1',
   'q8_1-activation decode kernels for K-quant and IQ4_XS weights: port of the llama.cpp mmvq vmmq arms with Kyojin\'s v2 items and multi-row forms', 'GGUF k-quant decode GEMV', 'q8_1 activations, dp4a, fp32 out', 'decode'),
 K('kernel.kyojin.fq_mmvq', X('foreign_quant/fq_mmvq.cu', 'foreign_quant/ggml_mmvq/'), 'linear.decode/gguf-iq+kquant/q8_1-f32.kyojin-mmvq',
   'IQ-format decode GEMV, llama.cpp mmvq style (CUDA and HIP, wave32): q8_1 activations, the ggml vec_dot_iq*_q8_1 dots (ggml_mmvq/: IQ, K, K46, legacy and MXFP4 vec-dots ported from llama.cpp), 1..8 activation columns a launch, plus a grouped decode MoE for IQ experts',
   'IQ-quantized GGUF weights at decode', 'q8_1 activations, dp4a, fp32 out', 'decode',
   upstream='ggml_mmvq/*.cuh and parts of fq_mmvq.cu from llama.cpp 7182ef7 / 61dbf49 (MIT, THIRD_PARTY_NOTICES.md)'),
 K('kernel.kyojin.fq_mmq_few', X('foreign_quant/fq_mmq_few.cu'), 'linear.verify/gguf-kquant/f32.kyojin-mmq-few',
   'Few-row MMQ (2..24 activation rows) for dense GGUF Q8_0 / Q4_K / Q5_K / Q6_K / Q3_K / IQ4_XS: int8 MMA on raw GGUF rows streamed through a per-warp cp.async ring, activations quantized to q8_1 in the fragment layout',
   'the verify shape of speculative decoding (a few rows) on GGUF weights', 'q8_1 activations, int8 MMA, fp32 accumulation', 'verify, 2..24 rows',
   upstream='mul_mat_q_few of llama.cpp PR #31533 (merge 0fb8f15770, MIT); the Q3_K and IQ4_XS decoders are Kyojin\'s'),
 K('kernel.kyojin.fq_moe', X('foreign_quant/fq_moe.cu'), 'moe.experts-decode/gguf-iquant/f32.kyojin-grouped',
   'Grouped decode MoE for foreign-quant experts (gateless relu2, IQ2_XXS): the selected experts\' up and down GEMVs in two launches plus a weighted fold, replacing a per-expert loop of 18 launches', 'MoE decode launch count on GGUF weights', 'fp32 accumulation', 'decode'),
 K('kernel.kyojin.fq_moe_q8', X('foreign_quant/fq_moe_q8.cu'), 'moe.experts-decode/gguf-iquant/f32.kyojin-q8_1',
   'Grouped decode MoE with q8_1 activations: same (slot, tile) grid as fq_moe.cu but integer dp4a dots over once-per-step quantized activations', 'integer-rate MoE decode', 'q8_1 activations, dp4a', 'decode'),
 K('kernel.kyojin.fq_moe_kq', X('foreign_quant/fq_moe_kq.cu'), 'moe.experts-decode/gguf-kquant/f32.kyojin-q8_1',
   'Grouped decode MoE with q8_1 activations for K-quant / IQ4_XS experts (gate + up, down, fold / combine launches; v2 items)', 'Hy4-flash Q3_K and generic GGUF MoE decode', 'q8_1 activations, dp4a', 'decode'),
 K('kernel.kyojin.fq_mlp', X('foreign_quant/fq_mlp.cu'), 'act.swiglu/-/f32.kyojin-fq-relu2',
   'Fused shared-expert up GEMV + relu2 epilogue (torch-exact) for foreign-quant MLPs, batch 1', 'one launch fewer per shared expert', 'fp32 accumulation, half out', 'decode'),
 K('kernel.kyojin.hip_kq_moe_dec', X('hip_kq/moe_dec.cu', 'hip_kq/build_moe.sh') + ['exllamav3/loader/moe_dec_hip.py'], 'moe.experts-decode/gguf-kquant/f32.kyojin-co',
   'Decode MoE kernels for GGUF k-quants as a device-only gfx1151 code object (built by hip_kq/build_moe.sh, loaded by loader/moe_dec_hip.py): same arithmetic and accumulation order as fq_moe_kq.cu, with unrolled-load / LDS-activation v3 arms',
   'a code object loaded beside the extension: the shape of a strixflerp kernel kit', 'q8_1 activations, dp4a, bit-equal to fq_moe_kq', 'decode', jit=True),
 # ---- small JIT extension of generic GGUF MoE ------------------------------------------------------------------------
 K('kernel.kyojin.gm_moe', ['exllamav3/gm_ext/gm_moe.cu', 'exllamav3/gm_ext/gm_moe.cuh', 'exllamav3/gm_ext/gm_moe.hip', 'exllamav3/gm_ext/gm_kq_item.cuh'], 'moe.experts-decode/gguf-kquant/f32.kyojin-gm',
   'Generic GGUF MoE decode for one token: Q4_K gate / up and Q4_K / Q6_K down experts, SwiGLU, no shared expert, same arithmetic as fq_moe_kq.cu (the .hip file is the hipify output the build compiles)', 'generic GGUF MoE models without a model-specific path', 'q8_1 activations, dp4a', 'decode', sources=['exllamav3/gm_ext/gm_moe.hip']),
 K('kernel.kyojin.gm_pfmoe', ['exllamav3/gm_ext/gm_pfmoe.cu', 'exllamav3/gm_ext/gm_pfmoe.hip'], 'moe.experts-prefill/gguf-kquant/f16-f32.kyojin-gm',
   'Generic GGUF MoE prefill experts for gfx1151 WMMA: fused dequant of the stored k-quant bytes and a grouped fp16 GEMM over expert row blocks, per-expert pointer tables, no row floor', 'generic GGUF MoE prefill', 'fp16 WMMA, fp32 accumulation', 'prefill', sources=['exllamav3/gm_ext/gm_pfmoe.hip']),
 K('kernel.kyojin.gm_router', ['exllamav3/gm_ext/gm_router.cu', 'exllamav3/gm_ext/gm_router.cuh', 'exllamav3/gm_ext/gm_router.hip'], 'moe.router/-/f32.kyojin-gm',
   'Decode-only standard MoE router for up to eight rows (gfx12 tuned): the gate held as a persistent contiguous (experts, hidden) transpose so each GEMV is coalesced', 'router GEMV coalescing at decode', 'fp32', 'decode', sources=['exllamav3/gm_ext/gm_router.hip']),
 K('kernel.kyojin.gm_dense_q8', ['exllamav3/gm_ext/gm_dense_q8.cu', 'exllamav3/gm_ext/gm_dense_q8.cuh'], 'linear.decode/gguf-q8_0/f32.kyojin-gm-cols',
   'Dense Q8_0 x q8_1 decode GEMV, few rows, column-tiled (HY4_DENSE_V2): one block walks consecutive columns with the thread-to-chunk map of fq_gemv_q8', 'Hy4 dense trunk decode', 'q8_1 activations, dp4a', 'decode'),
 K('runtime.kyojin.gm_ext', ['exllamav3/gm_ext/gm_bindings.cpp', 'exllamav3/gm_ext/__init__.py'], None,
   'The small JIT-built extension beside exllamav3_ext (gm_bindings.cpp binds the gm_* kernels; __init__.py builds or loads it, get() returns None when absent)', 'how a side extension ships kernels without rebuilding the main one', 'n/a', 'host', host=True),
 # ---- attention kernels written as standalone HIP (JIT, code objects) --------------------------------------------------
 K('kernel.kyojin.qdec_hip', ['exllamav3/modules/attention_fn/qdec.hip', 'exllamav3/modules/attention_fn/qdec_hip.py'], 'attn.softmax-gqa/kv-f16+i8+q4_0/f32.kyojin-qdec',
   'Quantized-KV flash-decoding split kernel for gfx11.5 (wave32 WMMA f16): QK^T and PV straight from the packed bit-plane words of the quantized cache, per-split partials reduced by the combine kernel; compiled at first use with hipcc --genco and launched through the HIP module API',
   'decode attention over the 2..8-bit paged cache without a dequant pass', 'dequant (raw - (2^(B-1) - 0.5)) * scale * 2^-(B-1) in fp16, fp32 softmax and accumulation', 'decode', jit=True,
   source_flags={'exllamav3/modules/attention_fn/qdec.hip': ['-DHD=256', '-DKB=4', '-DVB=4', '-DNKV=2', '-DNQH=24', '-DQLEN=1', '-DPAGE=256', '-DNW=4', '-DDW=2', '-DCAUSAL=1', '-DHAS_SINKS=0', '-DKEY_LO=0', '-DTQ=0', '-DVPF=1', '-DSMM1=0', '-DSCALE=0.0625f']},
   notes='The loader specialises the kernel per call with -D defines (qdec_hip.py _compile); the gate compiles one decode configuration: head 256, 24 query / 2 KV heads, 4-bit K and V, one query row, 256-token pages, NW=4, DW=2 (what make_cfg picks for that shape).'),
 K('kernel.kyojin.qdec_cuda', ['exllamav3/modules/attention_fn/cu/', 'exllamav3/modules/attention_fn/qdec_cuda.py'], 'attn.softmax-gqa/kv-f16+i8+q4_0/f32.kyojin-qdec-cuda',
   'CUDA twin of qdec (sm_80+, NVRTC at first use): tensor-core QK^T / PV from packed bit planes, plus qdec_rows.cu for verify steps with many query rows per KV head (up to ~56)', 'the NVIDIA side of the same cache format; the rows form is the verify-step shape', 'fp16 MMA, fp32 accumulation', 'decode and verify', lang='cuda'),
 K('kernel.kyojin.qsa_prefill', ['exllamav3/modules/attention_fn/qsa_prefill.hip', 'exllamav3/modules/attention_fn/qsa_prefill_hip.py', 'exllamav3/modules/attention_fn/qsa_proof/'], 'attn.softmax-gqa/kv-f16/f32.kyojin-qsa-prefill',
   'Qwen QSA sparse prefill attention for gfx11.5 WMMA f16, one wave per (row, kv head): replaces the Triton split + combine pair for fp16 paged caches with bit-identical per-row arithmetic (32-key tiles over the selected keys); qsa_proof/ holds the bounds-proof stamp the loader checks',
   'Qwen3.8-Flash-Next long-prompt prefill', 'fp16 WMMA, fp32 softmax', 'prefill', jit=True),
 K('kernel.kyojin.qsa_prefill_wmma', ['exllamav3/modules/attention_fn/qsa_prefill_wmma.hip'], 'attn.softmax-gqa/kv-f16/f32.kyojin-qsa-union4',
   'QSA sparse prefill, 4-row union variant (8 waves a block): 16-key tiles gathered once from the union of the rows\' selections (LDS bitonic union, per-row visibility masks), K / V prefetch in registers, V transposed in LDS; structure after Gufo\'s WmmaCausalAttentionKernel, no Gufo source copied',
   'shares tile loads across rows whose selections overlap', 'fp16 WMMA, fp32 softmax', 'prefill', jit=True,
   upstream='structure follows Gufo WmmaCausalAttentionKernel (MIT); written for Kyojin\'s index-list selection (THIRD_PARTY_NOTICES.md)'),
 # ---- Triton kernels (python, JIT) ---------------------------------------------------------------------------------
 K('kernel.kyojin.triton_paged', ['exllamav3/modules/attention_fn/triton_paged.py'], 'attn.softmax-gqa/kv-f16+i8+q4_0/f32.kyojin-triton-paged',
   'Triton paged attention of the engine as Kyojin changed it (2,694 lines): flash-decoding split / combine over the paged fp16 and quantized caches, prefill and verify forms', 'the generic attention path on ROCm', 'fp16 operands, fp32 softmax', lang='python'),
 K('kernel.kyojin.triton_paged_cu', ['exllamav3/modules/attention_fn/triton_paged_cu.py'], 'attn.softmax-gqa/kv-f16/f32.kyojin-triton-paged-cu',
   'CUDA-only paged decode attention (flash-decoding split + combine) ported from upstream exllamav3 1.6.0', 'the NVIDIA decode path', 'fp16, fp32 softmax', lang='python', upstream='ported from exllamav3 1.6.0 modules/attention_fn/triton_paged.py (MIT, turboderp)'),
 K('kernel.kyojin.triton_dsa', ['exllamav3/modules/attention_fn/dsa_triton.py'], 'attn.sparse-sink/kv-latent/bf16.kyojin-triton-dsa',
   'Production DSA (DeepSeek Sparse Attention) Triton kernels (1,877 lines): constexpr toggles for a sliding-window phase, sinks, gathered phase, indexer feeds', 'GLM-5.3-Flash / DeepSeek V4 sparse attention on ROCm', 'fp16 / bf16, fp32 softmax', lang='python'),
 K('kernel.kyojin.triton_mla', ['exllamav3/modules/attention_fn/mla_triton.py', 'exllamav3/modules/attention_fn/mla_sinks.py'], 'attn.sparse-sink/kv-latent/bf16.kyojin-triton-mla',
   'Triton kernels for multi-head latent attention (compressed latent and shared RoPE key in the cache; absorb / unfold WMMA kernels) and mla_sinks.py, the shared maths of per-head sinks, sliding window and sigmoid output gate',
   'GLM-5.3-Flash MLA decode and prefill', 'fp16 WMMA, fp32 accumulation', lang='python'),
 K('kernel.kyojin.triton_qsa', ['exllamav3/modules/attention_fn/qsa_triton.py'], 'attn.qsa-indexer/-/f32.kyojin-triton',
   'Triton kernels of the QSA indexer stages of the graph-captured decode path (raw-key plane append, scoring shared with the DSA / kpool stack)', 'Qwen3.8-Flash-Next decode selection', 'fp16 keys, fp32 scores', lang='python'),
 K('kernel.kyojin.triton_kvwin', ['exllamav3/modules/attention_fn/kvwin_triton.py'], 'attn.softmax-gqa/kv-f16/f32.kyojin-kvwin',
   'KVWIN kernels: the fp16 ring writer and the fp16 ring partial of the flash-decoding split (exact sink and recent tokens beside a quantized pool)', 'quality of a quantized cache at its recent window', 'fp16 ring, fp32 softmax', lang='python'),
 K('kernel.kyojin.triton_tq', ['exllamav3/modules/attention_fn/tq_triton.py'], 'kvcache.rotate/tq3/f32.kyojin',
   'TurboQuant cache kernels: paged window dequant for the prefill staging scratch (32-value groups, power-of-two bit planes, one fp16 norm per head vector)', 'the TurboQuant KV cache (runtime.kyojin.kvcache_tq)', 'Lloyd-Max codes times an fp16 norm', lang='python'),
 K('kernel.kyojin.triton_qdeq', ['exllamav3/modules/attention_fn/qdeq_triton.py', 'exllamav3/modules/attention_fn/qdeq_window_triton.py'], 'kvcache.quant8/-/f32.kyojin-dequant-window',
   'Quantized paged cache to fp16 staging: the 8-bit H32-rotated cache for the HIP sparse prefill kernel, and any 2..8-bit plane cache to a compact fp16 window (same order as the torch fallback)', 'prefill over a quantized cache', 'exact dequant to fp16', lang='python'),
 K('kernel.kyojin.triton_qwrite', ['exllamav3/modules/attention_fn/qwrite_triton.py'], 'kvcache.quant8/-/f32.kyojin-triton-writer',
   'Native Triton 8-bit paged cache writer replacing the torch fallback of ext.quant_cache_paged on ROCm (q_cache.cu is not in the ROCm build): same format and float32 operation order (unnormalised H32 butterfly, scale = absmax + eps)', 'the ROCm cache write', 'fp32 math, 8-bit codes', lang='python'),
 # ---- attention runtime pieces --------------------------------------------------------------------------------------
 K('runtime.kyojin.attn_bc', ['exllamav3/modules/attention_fn/bc_attn.py', 'exllamav3/modules/attention_fn/bc_dsa.py', 'exllamav3/modules/attention_fn/bc_mla.py'], 'executor.device-graph/exl3/-.kyojin-bc-attn',
   'Python side of the graph-captured attention blocks (GQA, DSA, MLA): buffers, capture and patch-and-replay of each attention step', 'the captured decode step of each attention family', 'n/a', 'decode', lang='python'),
 K('runtime.kyojin.attn_dispatch', ['exllamav3/modules/attention_fn/dispatch.py', 'exllamav3/modules/attention_fn/common.py', 'exllamav3/modules/attention_fn/torch.py', 'exllamav3/modules/attention_fn/bighead_scalar.py'], None,
   'Attention backend dispatch (hint caches, per-shape handler lists), shared helpers, the torch reference path and the scalar big-head path', 'which attention kernel serves a call', 'n/a', lang='python'),
 K('runtime.kyojin.ck_pagedkv', ['exllamav3/modules/attention_fn/ck_pagedkv*.py'], 'attn.prefill-causal/kv-f16/f32.kyojin-ck',
   'Optional Composable Kernel paged prefill backend (MiMo qk192, gfx1151, batch one) and its sparse form: page selection and compaction, an 8-bit cache adapter and the install hook (EXL3_CK_PAGEDKV)', 'CK FMHA as a prefill backend', 'fp16', 'prefill', lang='python'),
 # ---- DS4.1 kernels (JIT) and Engram ---------------------------------------------------------------------------------
 K('kernel.kyojin.dsv41_ops', ['exllamav3/modules/dsv41_fn/dsv41_hip.hip', 'exllamav3/modules/dsv41_fn/dsv41_num.h', 'exllamav3/modules/dsv41_fn/dsv41_hip.py', 'exllamav3/modules/dsv41_fn/stub/'], 'kernel-abi.dsv41-hip/-/-.kyojin',
   'DeepSeek-V4.1-Flash small-op kernels for gfx1151 in one code object (1,123 lines): each replaces a chain of torch ops of the CPU-exact reference; dsv41_num.h holds the fp8 / fp4 number formats as plain C++ shared bit for bit with a CPU build',
   'DS4.1 on one Strix Halo; the number-format header is a reusable bit-exact contract', 'explicit-rounding fp8 / fp4 fake-quant arithmetic', jit=True),
 K('kernel.kyojin.dsv41_idx', ['exllamav3/modules/dsv41_fn/dsv41_idx.hip', 'exllamav3/modules/dsv41_fn/dsv41_idx.py'], 'attn.indexer-score/-/f32.kyojin-dsv41',
   'DS4.1 indexer scoring (WMMA f16 to f32): score[r, t] = sum_h w[r, h] * relu(q[r, h] . k[t]) over 32 heads x 128, a dense pool scan and a candidate-slot scan', 'DSA indexer cost at long context', 'fp16 q / k, fp32 accumulation', jit=True),
 K('kernel.kyojin.dsv41_moe', ['exllamav3/modules/dsv41_fn/dsv41_moe.hip', 'exllamav3/modules/dsv41_fn/dsv41_moe.py'], 'moe.experts-decode/exl3/f32.kyojin-dsv41',
   'DS4.1 batch-1 routed experts of a layer in 5 launches for all K groups at once (top-6), bit-identical to the per-K-group exl3_moe_gfx12_k3 path', 'MoE decode launch count', 'fp16 activations, fp32 accumulation', 'decode', jit=True),
 K('kernel.kyojin.dsv41_fuse', ['exllamav3/modules/dsv41_fn/dsv41_fuse.hip', 'exllamav3/modules/dsv41_fn/dsv41_fuse.py'], 'act.swiglu/-/f32.kyojin-dsv41-glue',
   'DS4.1 decode glue fusion: swiglu, MoE tail, router scores and router top-k with K-group slots, each one launch with the same operation order as the unfused chain', 'the small launches between the FFN GEMVs', 'fp32', 'decode', jit=True),
 K('kernel.kyojin.ds41_pf', ['exllamav3/modules/dsv41_fn/ds41_pf.hip', 'exllamav3/modules/dsv41_fn/ds41_pf.py'], 'attn.sparse-sink/kv-latent/bf16.kyojin-ds41-pf',
   'DS4.1 prefill kernels: sparse attention (MQA, 64 heads x 512, one shared latent kv, per-row key lists, per-head sink) on WMMA f16, routed-expert gather-sum, rope rows', 'DS4.1 prefill', 'fp16 WMMA, fp32 softmax', 'prefill', jit=True),
 K('runtime.kyojin.engram_io', ['exllamav3/modules/dsv41_fn/engram_gather.cpp', 'exllamav3/modules/dsv41_fn/engram_io.py', 'exllamav3/modules/dsv41_fn/engram_dev.py', 'exllamav3/modules/dsv41_fn/engram_prefetch.py'],
   'ple.ngram-rows/ngram-table/-.kyojin-engram',
   'Engram n-gram table I/O: a threaded O_DIRECT row gather (plain C ABI, ctypes, no torch), device staging and prefetch of the next rows', 'n-gram tables too large for memory, read from NVMe per step', 'rows copied verbatim', host=True),
 K('runtime.kyojin.engram_io_c', ['exllamav3/loader/engram_io.c'], 'ple.ngram-rows/ngram-table/-.kyojin-pread',
   'GIL-free parallel row reads for the Engram table (one pread per distinct row into a flat buffer), C', 'the Hy4 Engram path', 'rows copied verbatim', lang='c', host=True),
 K('kernel.kyojin.gr_mix', ['exllamav3/modules/gr_fn/'], 'hc.mix-prefill/f16/f32.kyojin-gr',
   'Gated-residual mix chain of the Qwen prefill for gfx11.5 WMMA f16: dm = normed @ proj^T, t = silu(dm/H), the up projection and the sigmoid mean mix as hand-written kernels', 'Qwen3.8-Flash-Next prefill', 'fp16 WMMA, fp32 math, rne16 rounding', 'prefill', jit=True),
 K('kernel.kyojin.ple_hip', ['exllamav3/modules/ple_fn/'], 'ple.inject/-/f32.kyojin-hip',
   'Hand HIP kernels for the two torch fallbacks left in the Qwen prefill: PLELayer.forward_streams_reference after the key / value projections, and deinterleave_qg; bitwise identical to the torch chains', 'prefill launch count', 'fp32', 'prefill', jit=True),
 K('kernel.kyojin.gdn_fn_conv1d', ['exllamav3/modules/gated_delta_net_fn/conv1d.py'], 'mixer.gdn-front/-/f32.kyojin-conv1d', 'Causal conv1d of the GDN front end (Triton and torch paths) as Kyojin changed it', 'GDN layers', 'fp32', lang='python'),
 K('kernel.kyojin.gdn_fn_rule', ['exllamav3/modules/gated_delta_net_fn/gated_delta_rule.py'], 'mixer.gdn-prefill/-/f32.kyojin-rule', 'Gated delta rule dispatch (chunked prefill and recurrent decode forms)', 'GDN layers', 'fp32 state', lang='python'),
 K('kernel.kyojin.gdn_fn_gr_pf', ['exllamav3/modules/gated_delta_net_fn/gr_pf.py'], 'hc.mix-prefill/f16/f32.kyojin-gr-pf', 'Fused prefill pieces of GatedResidual._mix (grouped rms_norm, gate application) in Triton, replacing ~13 fp32 torch kernels on ROCm', 'Qwen prefill', 'fp32', 'prefill', lang='python'),
 K('kernel.kyojin.kda_fused', ['exllamav3/modules/gated_delta_net_fn/kda_gate.py', 'exllamav3/modules/gated_delta_net_fn/kda_norm.py'], 'norm.rmsnorm-gated/-/f32.kyojin-kda',
   'Fused KDA forget gate and gated RMS norm (Triton), replacing torch chains (norm.cu is not built on ROCm)', 'Kimi-Linear on ROCm', 'fp32', lang='python'),
 # ---- vendored fla (Triton) and its HIP replacements ------------------------------------------------------------------
 K('kernel.kyojin.fla_gdn_hf', ['exllamav3/vendor/fla/hip/gdn_hf.hip', 'exllamav3/vendor/fla/hip/gdn_hf_hip.py', 'exllamav3/vendor/fla/hip/LICENSE.hipfire-APACHE', 'exllamav3/vendor/fla/hip/NOTICE.hipfire', 'exllamav3/vendor/fla/hip/__init__.py'],
   'mixer.gdn-prefill/-/f32.kyojin-hipfire', 'Gated delta chunk WMMA kernel for gfx1151 adapted from hipfire (gated_delta_chunk_wmma.gfx1151.hip), with its Apache-2.0 licence and NOTICE', 'GDN chunked prefill on WMMA; kin of the hipfire GDN kernels', 'bf16 WMMA, fp32 state', 'prefill', jit=True, spdx='Apache-2.0',
   upstream='hipfire v0.4.2.1-13-gbfb90b9 kernels/src/gated_delta_chunk_wmma.gfx1151.hip (Apache-2.0)'),
 K('kernel.kyojin.fla_kda_hf', ['exllamav3/vendor/fla/hip/kda_hf.hip', 'exllamav3/vendor/fla/hip/kda_hf_hip.py'], 'mixer.kda/-/f32.kyojin-hipfire', 'KDA chunk kernel derived from hipfire\'s gated delta chunk WMMA kernel', 'KDA chunked prefill', 'bf16 WMMA, fp32 state', 'prefill', jit=True, spdx='Apache-2.0',
   upstream='derived from hipfire v0.4.2.1-13-gbfb90b9 gated_delta_chunk_wmma.gfx1151.hip (Apache-2.0)'),
 K('kernel.kyojin.fla_gdn_fused_h', ['exllamav3/vendor/fla/hip/gdn_fused_h.hip', 'exllamav3/vendor/fla/hip/gdn_fused_h_hip.py'], 'mixer.gdn-prefill/-/f32.kyojin-fused-h',
   'GDN fused chunk kernel for gfx11.5 WMMA: recompute_w_u + chunk_gated_delta_rule_fwd_h + chunk_fwd_o of the fla Triton chain in one kernel', 'GDN prefill launch count and traffic', 'bf16 WMMA, fp32 state', 'prefill', jit=True),
 K('kernel.kyojin.fla_kda_fused_h', ['exllamav3/vendor/fla/hip/kda_fused_h.hip', 'exllamav3/vendor/fla/hip/kda_fused_h_hip.py'], 'mixer.kda/-/f32.kyojin-fused-h',
   'KDA fused chunk recurrence (output fused) for gfx11.5 / gfx12 WMMA, matching the Triton chunk_kda_fwd_kernel_h_fused bitwise', 'Kimi-Linear prefill', 'bf16 products, fp32 state', 'prefill', jit=True),
 K('kernel.kyojin.fla_triton_gdn', ['exllamav3/vendor/fla/gdn_*.py', 'exllamav3/vendor/fla/chunk_*.py', 'exllamav3/vendor/fla/solve_tril.py', 'exllamav3/vendor/fla/cumsum.py', 'exllamav3/vendor/fla/l2norm.py'], 'mixer.gdn-prefill/-/f32.kyojin-fla',
   'Vendored flash-linear-attention Triton chain for GDN as Kyojin changed and extended it (gdn_fused, gdn_kkt_shared, gdn_pf added)', 'GDN prefill on any GPU with Triton', 'bf16 / fp32', 'prefill', lang='python', upstream='flash-linear-attention (MIT, Songlin Yang, Yu Zhang, Zhiyuan Li; exllamav3/vendor/fla/LICENSE)'),
 K('kernel.kyojin.fla_triton_kda', ['exllamav3/vendor/fla/kda_*.py', 'exllamav3/vendor/fla/gla_chunk.py'], 'mixer.kda/-/f32.kyojin-fla',
   'Vendored fla Triton chain for KDA / GLA, with Kyojin\'s fused intra, inter-split and fused-h forms', 'KDA prefill', 'bf16 / fp32', 'prefill', lang='python', upstream='flash-linear-attention (MIT)'),
 K('kernel.kyojin.fla_common', ['exllamav3/vendor/fla/__init__.py', 'exllamav3/vendor/fla/utils.py'], None, 'Package glue of the vendored fla (exports, utils)', 'needed by both fla chains', 'n/a', lang='python'),
 # ---- hipsp (standalone sparse MLA) --------------------------------------------------------------------------------
 K('kernel.kyojin.sp_mla', ['hipsp/sp_mla.cu', 'hipsp/occ.cu', 'hipsp/sp_idx.h', 'hipsp/sp_hip.py', 'hipsp/build.sh', 'hipsp/host_guard_test.cpp', 'hipsp/test_sp_hip.py', 'hipsp/mb_hip.py'], 'attn.sparse-sink/kv-latent/bf16.kyojin-sp-mla',
   'Sparse MLA prefill kernel for gfx1151 (WMMA f16, wave32): one workgroup of 16 waves = one query x 64 heads over latent rows [N, 576]; visible keys = selected groups + the own open group; address math in sp_idx.h shared with a guard-page host test',
   'GLM-5.3-Flash / DSA sparse MLA prefill as a standalone library', 'fp16 WMMA, fp32 softmax, saturated fp16 out', 'prefill', sources=['hipsp/sp_mla.cu'], notes='occ.cu is an occupancy probe that includes sp_mla.cu (a main() asking hipOccupancyMaxActiveBlocksPerMultiprocessor for sp_mla<4, true>); it no longer matches the template parameters of sp_mla in this tree (the gate run of 22:59Z: address of overloaded function sp_mla does not match), so it rides here as a file, not a source.', flags=['-x', 'hip', '-std=c++17', '-O3', '-fPIC', '-c']),
 K('tool.kyojin.hipsp_probes', ['hipsp/wpeak.cu', 'hipsp/wpeak2.cu'], 'tool.kernel-bench/-/-.kyojin-wmma-peak', 'WMMA peak-rate probes for gfx1151: two kernels that issue back-to-back v_wmma f16 ops to measure the matrix-core ceiling', 'roofline numbers for the sparse MLA work', 'n/a', 'offline', flags=['-x', 'hip', '-std=c++17', '-O3', '-fPIC', '-c']),
 # ---- host runtime of the extension ----------------------------------------------------------------------------------
 K('runtime.kyojin.ext_bindings', X('bindings.cpp', 'build_config.py'), 'executor.provider-api/exl3-ext/-.kyojin',
   'The extension\'s pybind11 module (every op bound, CUDA and ROCm branches) and build_config.py (which sources build on ROCm)', 'the call surface of every kernel above', 'n/a', 'host', host=True),
 K('runtime.kyojin.graph', X('graph.cu', 'graph.cuh'), 'executor.device-graph/exl3/-.kyojin-graph',
   'CUDA / HIP graph capture and node patching of the extension (record once, patch kernel arguments, replay)', 'captured decode steps; launch overhead dominates the Strix Halo decode step', 'n/a', 'decode'),
 K('runtime.kyojin.stloader', X('stloader.cpp', 'stloader.h', 'stloader_cu.cu'), 'format.reader/safetensors/-.kyojin-stloader',
   'Safetensors loader: coalesced reads through pinned ring buffers per worker thread, gaps read through rather than splitting runs', 'model load time from NVMe', 'bytes', 'load', host=True),
 K('runtime.kyojin.host_mem', X('cuda_host.cpp', 'cuda_host.h', 'cuda_drv.cpp', 'cuda_drv.h', 'triton_kernel.cpp'), 'device.memory/hip/-.kyojin-host-register',
   'Host-memory registration (shared arena, CPU expert offload segment), driver API entry points resolved at run time, and launching precompiled Triton kernels through the driver API', 'zero-copy host memory on the unified-memory APU', 'n/a', 'host', host=True),
 K('runtime.kyojin.cpu_moe_mul1', X('cpu/moe_mul1.cpp', 'cpu/moe_mul1.h', 'avx2_target.cpp', 'avx2_target.h', 'avx512_target.cpp', 'avx512_target.h'), 'moe.experts-decode/exl3/f32.kyojin-cpu',
   'CPU MoE expert GEMM for mul1 (cb2) EXL3 tensors (2,490 lines, AVX-512 VBMI / AVX2): the mul1 codebook is affine in a byte-sum, so experts offloaded to the CPU decode without a table; CPU feature probes',
   'experts that do not fit GPU memory run on the Zen 5 cores', 'byte-sum codebook, fp32 accumulation', 'decode, CPU', host=True),
 K('runtime.kyojin.moe_handoff', X('cpu/moe_handoff.cu', 'cpu/moe_handoff.h', 'cpu/moe_handoff_amdgpu.cu'), 'collective.device-proxy/-/-.kyojin-moe-handoff',
   'Persistent-worker handoff for CPU-offloaded experts: pinned shared-memory flags, GPU publish / wait kernels and a spin-parked child process consuming a job ring; moe_handoff_amdgpu.cu is the HIP port',
   'GPU-CPU expert handoff without host syncs: a transport between the device stream and CPU workers', 'n/a', 'decode'),
 K('runtime.kyojin.tp_allreduce_cpu', X('parallel/all_reduce_cpu.cu', 'parallel/all_reduce_cpu_avx2.cpp', 'parallel/all_reduce_cpu_avx2.h', 'parallel/all_reduce_cpu_avx512.cpp'), 'collective.allreduce-p2p/-/bf16+f32.kyojin-cpu',
   'CPU-assisted all-reduce of the native tensor-parallel backend: devices stage 128 KB wire chunks to pinned memory, a worker pool of the CPU-reduce process accumulates (AVX2 / AVX-512 BF16 adds) and the result flows back', 'tensor parallelism without peer access (the multi-device split)', 'bf16 / fp32 adds', 'decode and prefill'),
 K('runtime.kyojin.tp_broadcast_gather', X('parallel/broadcast.cu', 'parallel/gather.cu'), 'collective.allgather-tcp/-/f32.kyojin-pinned',
   'Broadcast and gather collectives over pinned host memory with per-block stage flags', 'the TP backend\'s other collectives', 'byte copies'),
 K('runtime.kyojin.tp_context', X('parallel/context.cu', 'parallel/context.cuh', 'parallel/barrier_inner.cuh', 'parallel/timeout.cuh'), 'collective.interface/-/-.kyojin',
   'TP context, epoch barrier and the sticky timeout (once one collective times out, every later wait aborts)', 'what the collectives above share', 'n/a'),
 K('runtime.kyojin.sam', X('sam.cpp', 'sam.h', 'sam_bc.h'), 'draft.ddtree/-/f32.kyojin-sam',
   'Suffix automaton over the token history (BC_SAM): incremental extend and longest-match lookup, the prompt-lookup draft source of speculative decoding', 'n-gram / prompt-lookup drafting in C++', 'n/a', 'decode', host=True),
 K('runtime.kyojin.bc_attention', X('libtorch/attention.cpp', 'libtorch/attention.h', 'libtorch/attention_bc.h'), 'executor.device-graph/exl3/-.kyojin-bc-gqa',
   'Graph-captured decode attention block (first run eager, second captured, later patch-and-replay): q / k / v projections, fused q / k norm and rope, cache write and attention in one replay', 'decode launch overhead', 'n/a', 'decode'),
 K('runtime.kyojin.bc_blocksparse_mlp', X('libtorch/blocksparse_mlp.cpp', 'libtorch/blocksparse_mlp.h', 'libtorch/blocksparse_mlp_bc.h'), 'executor.device-graph/exl3/-.kyojin-bc-moe',
   'Graph-captured block-sparse MLP (MoE) block of the extension', 'MoE decode step', 'n/a', 'decode'),
 K('runtime.kyojin.bc_dsv4_attn', X('libtorch/dsv4_attn.cpp', 'libtorch/dsv4_attn.h'), 'executor.device-graph/deepseek-v4-flash/gguf.kyojin-bc-dsa',
   'Whole-attention-step graph for DeepSeek-V4 DSA decode: staged input, q_a / q_norm / q_b, wkv, fused-norm rope, compressors, indexer scoring with capture-safe top-k, split / combine', 'DSV4 decode', 'n/a', 'decode'),
 K('runtime.kyojin.bc_gated_delta_net', X('libtorch/gated_delta_net.cpp', 'libtorch/gated_delta_net.h', 'libtorch/gated_delta_net_bc.h'), 'executor.device-graph/exl3/-.kyojin-bc-gdn',
   'Graph-captured Gated DeltaNet block (BC_GatedDeltaNetSplit pattern)', 'GDN decode step', 'n/a', 'decode'),
 K('runtime.kyojin.bc_mla_attention', X('libtorch/mla_attention.cpp'), 'executor.device-graph/exl3/-.kyojin-bc-mla',
   'Graph-captured MLA attention block (998 lines)', 'GLM-5.3-Flash MLA decode step', 'n/a', 'decode'),
 K('runtime.kyojin.bc_mlp', X('libtorch/mlp.cpp', 'libtorch/mlp.h', 'libtorch/mlp_bc.h', 'libtorch/linear.h', 'libtorch/linear_bc.h'), 'executor.device-graph/exl3/-.kyojin-bc-mlp',
   'Graph-captured MLP and linear blocks: gate and up as one fused MGEMM when pointer tables are given, its own scratch for multi-token runs', 'dense MLP decode', 'n/a', 'decode'),
 # ---- python runtime: scheduler, spec decode, caches ----------------------------------------------------------------
 K('runtime.kyojin.generator', ['exllamav3/generator/generator.py', 'exllamav3/generator/job.py', 'exllamav3/generator/async_generator.py'], 'executor.decode-loop/exl3/-.kyojin',
   'The paged generator and job model as Kyojin rewrote them (2,894 + 2,492 lines): continuous batching, chunked prefill, speculative rounds (MTP, DFlash2, n-gram), page sharing and the async front', 'the scheduler every Kyojin server runs', 'n/a', 'serving', lang='python'),
 K('runtime.kyojin.sched_chunk_il', ['exllamav3/generator/chunk_il.py'], 'executor.decode-loop/-/-.kyojin-chunk-interleave',
   'Chunked-prefill interleave planner (default on since 1.8.8): decode rounds of running jobs are interleaved between prefill chunks of a new prompt; pure python, no torch', 'decode latency while a long prompt prefills', 'n/a', 'serving', lang='python'),
 K('runtime.kyojin.sched_prefill_share', ['exllamav3/generator/prefill_share.py'], 'cache.prompt/-/-.kyojin-prefill-share',
   'Concurrent prefill of one shared prefix: jobs starting together on one prompt prefix wait for the page owner instead of prefilling it twice', 'agent fan-out with a common prompt', 'n/a', 'serving', lang='python'),
 K('runtime.kyojin.sched_row_groups', ['exllamav3/generator/row_groups.py', 'exllamav3/util/row_inv.py'], 'executor.decode-loop/-/-.kyojin-row-invariant',
   'Row invariance of multi-job and multi-row verify forwards: jobs grouped so a row gets the same bits whatever else is decoded with it (EXL3_ROW_INV)', 'determinism across batch composition', 'n/a', 'serving', lang='python'),
 K('runtime.kyojin.spec_sample', ['exllamav3/generator/spec_sample.py'], 'speculation.accept/-/f32.kyojin', 'Speculative sampling and acceptance of drafted tokens (greedy and sampled rules)', 'MTP / DFlash2 / n-gram verify', 'fp32 probabilities', 'decode', lang='python'),
 K('runtime.kyojin.draft_depth', ['exllamav3/generator/draft_depth.py'], 'speculation.draft/dflash/f32.kyojin-depth',
   'Acceptance-driven draft depth for block drafters (DFlash2): an EMA of accepted tokens picks full depth, a short block or off per request', 'draft cost when acceptance drops', 'n/a', 'decode', lang='python'),
 K('runtime.kyojin.draft_confidence', ['exllamav3/generator/draft_confidence.py'], 'speculation.draft/mtp/f32.kyojin-confidence', 'Confidence-gated draft length (stop drafting when the draft\'s probability falls)', 'MTP draft length', 'fp32', 'decode', lang='python'),
 K('runtime.kyojin.draft2s', ['exllamav3/modules/draft2s.py'], 'speculation.draft/mtp/f32.kyojin-draft2s', 'Two-stage drafter module (draft2s) of the speculative path', 'speculative decoding drafts', 'n/a', 'decode', lang='python'),
 K('runtime.kyojin.loop_detect', ['exllamav3/generator/loop_detect.py'], 'serve.think-nudge/-/-.kyojin-loop-detect', 'Repetition loop detection on the generated stream', 'stops runaway loops in long thinking', 'n/a', 'serving', lang='python'),
 K('runtime.kyojin.sampler', ['exllamav3/generator/sampler/'], 'sample.next-token/-/f32.kyojin-stack', 'Sampler stacks and presets (custom samplers, collapsed stacks that map onto the fused device sampler)', 'sampling settings to device kernels', 'fp32', lang='python'),
 K('runtime.kyojin.pagetable', ['exllamav3/generator/pagetable.py'], 'kvcache.parked-state/-/-.kyojin-pagetable', 'Paged KV page table: hashing, sharing, reference counts and eviction', 'prefix reuse', 'n/a', lang='python'),
 K('runtime.kyojin.cpu_cache', ['exllamav3/generator/cpu_cache.py'], 'kvcache.tiering/exl3-pages/-.kyojin-cpu', 'Host-memory tier of the paged cache (pages swapped to CPU memory)', 'more cached prefixes than device memory holds', 'n/a', lang='python'),
 K('runtime.kyojin.prefix_disk', ['exllamav3/generator/prefix_disk.py', 'exllamav3/util/disk_store.py'], 'cache.prompt/-/-.kyojin-nvme',
   'Persistent prompt cache on NVMe: content-addressed pages (blake2b of page ids and the previous hash) survive lane unloads, model swaps and reboots; disk_store.py is the namespace, atomic background writes, LRU byte cap and free-space floor', 'a resumed conversation pays only its new tokens', 'n/a', lang='python'),
 K('runtime.kyojin.slot_store', ['exllamav3/generator/slot_store.py'], 'serve.resume-loss/-/-.kyojin-slots', 'llama.cpp-compatible slot persistence (GET /slots, save / restore / erase) for the paged generator', 'swap orchestrators keep lane caches warm on disk', 'n/a', lang='python'),
 K('runtime.kyojin.kvcache_core', ['exllamav3/cache/cache.py', 'exllamav3/cache/__init__.py'], 'kvcache.store/kv-bf16/bf16.kyojin', 'Cache layer base and registry as Kyojin changed them', 'every cache layer', 'n/a', lang='python'),
 K('runtime.kyojin.kvcache_dsa', ['exllamav3/cache/dsa.py'], 'kvcache.compress/-/f32.kyojin-dsa-cache', 'DSA cache layer (compressed pools, indexer keys)', 'DSA models', 'n/a', lang='python'),
 K('runtime.kyojin.kvcache_mla', ['exllamav3/cache/mla.py'], 'kvcache.store/kv-bf16/bf16.kyojin-mla', 'MLA latent cache layer: one compressed latent (ckv, kv_lora_rank) and one shared RoPE key (kpe) per token in pages, instead of per-head K and V', 'MLA models', 'n/a', lang='python'),
 K('runtime.kyojin.kvcache_qsa', ['exllamav3/cache/qsa.py'], 'kvcache.store/kv-bf16/bf16.kyojin-qsa', 'QSA cache layer (raw-key planes for the indexer)', 'Qwen3.8-Flash-Next', 'n/a', lang='python'),
 K('runtime.kyojin.kvcache_recurrent', ['exllamav3/cache/recurrent.py', 'exllamav3/cache/recurrent_util.py'], 'kvcache.parked-state/-/-.kyojin-recurrent', 'Recurrent-state cache (GDN / KDA / Mamba states, checkpoints for speculative rewind)', 'linear-attention models', 'fp32 states', lang='python'),
 K('runtime.kyojin.kvcache_tq', ['exllamav3/cache/tq.py', 'exllamav3/cache/tq_codebooks.py'], 'kvcache.rotate/tq3/f32.kyojin-turboquant',
   'TurboQuant KV cache layer: per head vector y = R x (Hadamard x random signs), n = ||y||, codes = nearest Lloyd-Max centroid of y / n; codebooks for (D, bits) from the Lloyd iteration on (1 - t^2)^((D-3)/2)', 'a 2..4-bit KV cache with a principled codebook', 'Lloyd-Max codes, one fp16 norm', lang='python'),
 K('runtime.kyojin.kvcache_kvwin', ['exllamav3/cache/kvwin.py'], 'kvcache.store/kv-bf16/bf16.kyojin-kvwin', 'KVWIN: exact fp16 sink tokens and a recent-window ring per job on top of a complete quantized pool', 'quality of quantized caches', 'fp16 ring', lang='python'),
 # ---- loader levers (Hy4-flash, GGUF) ------------------------------------------------------------------------------
 K('format.kyojin.gguf_loader', ['exllamav3/loader/gguf_*.py', 'exllamav3/loader/gguf_blocks/'], 'format.loader/gguf/-.kyojin',
   'GGUF loading for the EXL3 engine: the tensor collection (raw quantized payloads with the SafetensorsCollection call shape), the block registry that composes a generic loader from per-family blocks, GGUF-to-HF config, tokenizer from tokenizer.ggml.* (pre-tokenizer table generated from llama.cpp)', 'runs GGUF files unconverted', 'raw bytes', 'load', lang='python'),
 K('format.kyojin.safetensors_loader', ['exllamav3/loader/safetensors.py'], 'format.reader/safetensors/-.kyojin-py', 'Python safetensors collection as Kyojin changed it (1,293 lines)', 'EXL3 pack loading', 'bytes', 'load', lang='python'),
 K('executor.kyojin.hy4_forward', ['exllamav3/loader/hy4_forward.py', 'exllamav3/loader/mla_hy4.py', 'exllamav3/loader/ablit_hy4.py'], 'executor.model/hy4-flash/gguf.kyojin',
   'Hy4-flash whole-model forward (slow but correct, torch only), its CPU fp32 attention reference and the opt-in decensor edit hook (HY4_ABLIT_SPEC)', 'the reference every Hy4 lever is checked against', 'fp32 reference', lang='python'),
 K('runtime.kyojin.hy4_attn_levers', ['exllamav3/loader/attn_eager.py', 'exllamav3/loader/attn_grouped.py', 'exllamav3/loader/rope_rows.py', 'exllamav3/loader/kv_fp8.py', 'exllamav3/loader/kv_fused.py', 'exllamav3/loader/mla_bgemv.py'],
   'attn.qk-norm-rope-store/-/bf16.kyojin-hy4', 'Hy4 attention decode levers, each behind a flag: fewer eager launches, grouped projections on the stored bytes, positional rope rows, fp8 latent K cache, one-kernel K-row writes, absorbed MLA matrices read from stored Q8_0', 'Hy4-flash decode', 'see each file', 'decode', lang='python'),
 K('runtime.kyojin.hy4_prefill', ['exllamav3/loader/prefill_attn.py', 'exllamav3/loader/prefill_unabs.py', 'exllamav3/loader/prefill_kernels.py', 'exllamav3/loader/ck_mla.py'], 'moe.experts-prefill/gguf-kquant/f16-f32.kyojin-triton',
   'Hy4 prefill kernels in Triton (ROCm gfx11 WMMA through tl.dot): fused dequant + GEMM on stored GGUF bytes, grouped / sorted-token MoE, flash-style absorbed MLA prefill, un-absorbed MLA helpers, CK fmha behind ckmla_fwd', 'Hy4-flash prefill (928-987 tok/s served, README)', 'fp16 WMMA, fp32 accumulation', 'prefill', lang='python'),
 K('runtime.kyojin.hy4_moe_levers', ['exllamav3/loader/kq_moe_hip.py', 'exllamav3/loader/moe_i8.py', 'exllamav3/loader/router_gemv.py', 'exllamav3/loader/dense_req.py', 'exllamav3/loader/head_bf16.py', 'exllamav3/loader/ref_fused.py'],
   'moe.router/-/f32.kyojin-hy4', 'Hy4 MoE / dense decode levers: HIP WMMA grouped expert GEMM on stored k-quants, int8 draft experts, one-launch router GEMV, run-time dense requant arms, bf16 head mat-vec, fused router select', 'Hy4-flash decode', 'see each file', 'decode', lang='python'),
 K('runtime.kyojin.hy4_engram', ['exllamav3/loader/engram_rows.py', 'exllamav3/loader/engram_gpu.py', 'exllamav3/loader/engram_fused.py', 'exllamav3/loader/engram_stage.py'], 'ple.ngram-rows/ngram-table/-.kyojin-hy4',
   'Hy4 Engram: row lookup (E4M3 / E8M0 tables, F8 rows), the GPU Engram (distinct rows preaded, nothing of the hidden state leaves the device), fused Engram block in two Triton launches, vectorised host staging for graph replays', 'Hy4-flash Engram layers', 'exact row decode', lang='python'),
 K('runtime.kyojin.hy4_indexer', ['exllamav3/loader/indexer_fused.py', 'exllamav3/loader/pool_fused.py', 'exllamav3/loader/lc_sparse.py'], 'attn.indexer-topk/-/i32.kyojin-kpool',
   'k-pool sparse attention pieces: device-resident pooled-key indexer with no host sync, one-kernel pool update, long-context decode kernels (exact top-n group selection)', 'Hy4 long-context decode', 'fp32 scores', 'decode', lang='python'),
 K('runtime.kyojin.hy4_hc', ['exllamav3/loader/hc_fused.py'], 'hc.mix/-/f32.kyojin-hy4-triton', 'Fused hyper-connection sites and norms (Triton), generic by shape', 'Hy4 decode', 'fp32', lang='python'),
 K('runtime.kyojin.hy4_graph', ['exllamav3/loader/step_graph.py', 'exllamav3/loader/step_mu.py', 'exllamav3/loader/step_nosync.py', 'exllamav3/loader/graph_bucket.py', 'exllamav3/loader/mu_slots.py', 'exllamav3/loader/mu_debug.py', 'exllamav3/loader/lever_fires.py'],
   'executor.device-graph/hy4-flash/gguf.kyojin', 'Hy4 decode step as one HIP graph per context bucket, multi-session replay (rows of several sessions in one replay), the host-sync-free driver, context buckets, per-session ops, fault localisation and lever hit counters', 'the ~4,000 small launches of the eager step become one replay', 'n/a', 'decode', lang='python'),
 K('runtime.kyojin.hy4_mtp', ['exllamav3/loader/mtp_head.py', 'exllamav3/loader/mtp_gate.py'], 'draft.mtp/gguf-q8_0/f32.kyojin-hy4',
   'Hy4 MTP draft head (the GGUF next-n block with two Triton expert kernels) and the split draft chain (base graph + tail graph, HY4_SPEC2_GATE)', 'Hy4 speculative decode (tau 2.7-2.9)', 'n/a', 'decode', lang='python'),
 # ---- runtime support ----------------------------------------------------------------------------------------------
 K('runtime.kyojin.hip_jit', ['exllamav3/util/hip_compiler.py', 'exllamav3/util/hip_lib.py'], None, 'One resolver for the hipcc that builds every JIT HIP kernel (EXL3_HIPCC, the rocm-sdk wheel, PATH, /opt/rocm) and the module loader for the resulting code objects', 'how the JIT kernels above are built and loaded', 'n/a', 'host', lang='python'),
 K('runtime.kyojin.triton_seed', ['exllamav3/util/kernel_seed.py', 'exllamav3/util/jit_log.py'], 'verify.kit-manifest/triton-cache/-.kyojin', 'Pre-compiled Triton kernels shipped with the code (kernel_seed tarball per Triton version and arch) and a log of Triton JIT compiles', 'cold-start compile time of a fresh container', 'n/a', 'host', lang='python'),
 K('runtime.kyojin.tune_cache', ['exllamav3/util/tune_cache.py', 'exllamav3/model/dense_gemm_tune_seed.txt', 'exllamav3/model/dense_warmup.py'], 'tune.table/hip/-.kyojin-pack', 'Per-pack tuning cache of the EXL3 GEMM autotuner, the shipped dense GEMM seed and the dense warm-up', 'tuned kernels from the first request', 'n/a', 'host', lang='python'),
 K('runtime.kyojin.knobs', ['exllamav3/util/knobs.py', 'exllamav3/util/attention_mode.py'], None, 'Run-time switchable A/B knobs (read from the environment at call time, one writer) and the frozen attention policy', 'one-load A/B of levers', 'n/a', 'host', lang='python'),
 K('runtime.kyojin.ext_fallbacks', ['exllamav3/ext_fallbacks.py'], 'tensor.ops/-/f32.kyojin-fallbacks', 'Torch fallbacks for extension ops not built on ROCm (the chains fused_elt and the Triton writers replace)', 'the reference arithmetic of the ROCm replacements', 'torch arithmetic', lang='python'),
 K('runtime.kyojin.guard', ['exllamav3/guard.py', 'exllamav3/guard_common.py', 'exllamav3/memory_guard.py'], 'host.memory-governor/-/-.kyojin', 'Memory guards: GTT / host-memory admission and the live free-memory floor of a server', 'a 128 GB APU shared with the OS', 'n/a', 'host', lang='python'),
 K('runtime.kyojin.cuda_graph', ['exllamav3/model/cuda_graph.py', 'exllamav3/model/host_table.py'], 'executor.device-graph/exl3/-.kyojin-py-graph', 'Python graph capture helpers and host pointer tables of the model', 'captured steps', 'n/a', 'decode', lang='python'),
 K('runtime.kyojin.block_graph', ['exllamav3/modules/block_graph.py'], 'executor.device-graph/exl3/-.kyojin-block-graph', 'Per-block graph capture of a decoder layer', 'decode launch overhead', 'n/a', 'decode', lang='python'),
 K('runtime.kyojin.hip_glue', ['exllamav3/modules/hip_glue.py'], None, 'Glue between modules and the JIT HIP kernels (loads and routes to the code objects)', 'ROCm kernel routing', 'n/a', lang='python'),
 K('runtime.kyojin.expert_store', ['exllamav3/modules/expert_store.py', 'exllamav3/modules/moe_mixedk.py'], 'runtime.weight-prefetch/-/-.kyojin-experts', 'Expert store (expert tiering between host and device with prefetch and protection) and mixed-K expert stacks', 'models larger than device memory', 'n/a', lang='python'),
 K('runtime.kyojin.moe_fused_py', ['exllamav3/modules/moe_fused.py', 'exllamav3/modules/router_topk.py'], 'moe.router/-/f32.kyojin-py', 'Python front of the fused MoE half-layer and the router top-k module', 'MoE decode', 'n/a', 'decode', lang='python'),
 K('runtime.kyojin.foreign_linear', ['exllamav3/modules/foreign_linear.py', 'exllamav3/modules/fq_moe_prefill*.py'], 'linear.dispatch/gguf/-.kyojin', 'Foreign-quant linear module and the foreign MoE prefill paths (generic, IQ, KQ): picks the fq_* kernel by format and row count', 'GGUF weights inside the EXL3 engine', 'n/a', lang='python'),
 K('runtime.kyojin.kpool_indexer', ['exllamav3/modules/kpool_indexer.py'], 'attn.qsa-indexer/-/f32.kyojin-kpool', 'k-pool DSA indexer module (Hy4 row-list form)', 'Hy4 sparse attention', 'fp32 scores', lang='python'),
 K('runtime.kyojin.qwd3_pf', ['exllamav3/modules/qwd3_pf.py'], 'runtime.weight-prefetch/-/-.kyojin-mall', 'Weight prefetch into the infinity cache (MALL, 32 MB, ~700 GB/s on re-reads vs 236 GB/s DRAM) during the latency-bound kernels of a decode token: read-only side-stream reads of weights a later streaming kernel will read, results bit-identical by construction (EXL3_QWD3_PF, default off)', 'decode on Strix Halo is memory bound; warming the MALL during latency-bound kernels hides DRAM time', 'n/a: no arithmetic, reads only', 'decode', lang='python'),
 K('runtime.kyojin.ablit_runtime', ['exllamav3/modules/ablit_runtime.py'], 'steer.add-rows/-/f32.kyojin-ablit', 'Run-time abliteration edit (a projection removed from writer outputs, no weights changed)', 'kin of the catalogue\'s steering modules', 'fp32', lang='python'),
 K('runtime.kyojin.dsv41_module', ['exllamav3/modules/dsv41.py'], 'executor.model/deepseek-v4.1-flash/exl3.kyojin-module', 'DS4.1 module of the EXL3 engine (glue to the dsv41_fn kernels)', 'DS4.1 serving', 'n/a', lang='python'),
 # ---- transport (multi-device split) -------------------------------------------------------------------------------
 K('runtime.kyojin.layer_split', ['exllamav3/serve_split.py', 'exllamav3/bridge.py', 'exllamav3/device_map.py', 'exllamav3/modules/row_split.py'], 'transport.stage-relay/-/f32.kyojin-bridge',
   'Contiguous layer split across two devices: the serve-side split plan, an async host-staged bridge (part0 and part1 in either order; on RX 9070 over USB4 + Strix Halo there is no fast peer path, so activations stage through host memory), the device map and row split', 'an eGPU or second device beside a Strix Halo', 'activations copied verbatim', 'serving', lang='python'),
 K('runtime.kyojin.tp_python', ['exllamav3/model/model_tp*.py', 'exllamav3/model/moe_cpu_host.py'], 'collective.interface/-/-.kyojin-tp',
   'Python side of the native tensor-parallel backend (allocation, backend, shared arena, functions) and the CPU MoE host as Kyojin changed them', 'TP across devices with the CPU-assisted collectives', 'n/a', lang='python'),
 # ---- foreign quant python ------------------------------------------------------------------------------------------
 K('format.kyojin.fq_formats', ['exllamav3/foreign_quant/k_quants.py', 'exllamav3/foreign_quant/iq2_xxs.py', 'exllamav3/foreign_quant/nvfp4.py', 'exllamav3/foreign_quant/q8_0.py', 'exllamav3/foreign_quant/legacy_dq.py', 'exllamav3/foreign_quant/gap_formats.py', 'exllamav3/foreign_quant/gguf_tables.py', 'exllamav3/foreign_quant/gguf_fast.py', 'exllamav3/foreign_quant/iq_grids_gen.py', 'exllamav3/foreign_quant/iq_seeds.py', 'exllamav3/foreign_quant/vendor_input.py', 'exllamav3/foreign_quant/__init__.py'],
   'format.device-view/gguf-kquant/-.kyojin-py', 'Python reference decoders of the foreign formats (k-quants, IQ2_XXS, NVFP4, Q8_0, legacy, gap formats), GGUF tables and the IQ grid generator that writes fq_tables.cuh', 'the bit-exact references the fq_* kernels are checked against', 'exact decode', lang='python'),
 K('kernel.kyojin.fq_triton', ['exllamav3/foreign_quant/triton_*.py', 'exllamav3/foreign_quant/moe_mxfp4_prefill.py'], 'moe.experts-prefill/mxfp4/bf16.kyojin-triton',
   'Triton kernels of the foreign formats: IQ and gap-format GEMVs, row gathers, shared-gate, f16 GEMV, MXFP4 MoE and its prefill', 'foreign formats where no HIP kernel exists', 'fp32 accumulation', lang='python'),
 # ---- DS4.1 python engine -------------------------------------------------------------------------------------------
 K('executor.kyojin.dsv41_reference', ['exllamav3/dsv41/ops.py', 'exllamav3/dsv41/quant.py', 'exllamav3/dsv41/engram.py', 'exllamav3/dsv41/model.py', 'exllamav3/dsv41/config.py', 'exllamav3/dsv41/linear.py', 'exllamav3/dsv41/experts.py', 'exllamav3/dsv41/iq2_tables.py', 'exllamav3/dsv41/weights.py', 'exllamav3/dsv41/safetensors_io.py', 'exllamav3/dsv41/gguf.py', 'exllamav3/dsv41/__init__.py'],
   'executor.model/deepseek-v4.1-flash/exl3.kyojin-ref', 'DeepSeek-V4.1-Flash CPU-exact reference engine (ops, fp8 / fp4 quant, Engram, model, weights I/O)', 'the reference the DS4.1 HIP kernels match bit for bit', 'explicit-rounding fp32 reference', lang='python'),
 K('serve.kyojin.dsv41_serve', ['exllamav3/dsv41/serve_*.py', 'exllamav3/dsv41/prefix_cache.py', 'exllamav3/dsv41/spec.py', 'exllamav3/dsv41/dspark.py', 'exllamav3/dsv41/draft_head.py', 'exllamav3/dsv41/dec2.py'], 'serve.engine/-/-.kyojin-dsv41',
   'DS4.1 serving model: cache, graph, MoE and ops of the serving path, prefix cache, speculative decoding (DSpark drafts, draft head, dec2)', 'DS4.1 served', 'n/a', 'serving', lang='python'),
 K('tool.kyojin.dsv41_convert', ['exllamav3/dsv41/convert_driver.py', 'exllamav3/dsv41/pack.py', 'exllamav3/dsv41/packspec.py', 'exllamav3/dsv41/calib.py', 'exllamav3/dsv41/stream_calib.py', 'exllamav3/dsv41/synth_pack.py'], 'tool.convert-qwen4exp/-/-.kyojin-dsv41',
   'DS4.1 pack conversion: driver, pack layout and spec, calibration (streaming), synthetic packs for tests', 'the public DS4.1 conversion (the Yamz EXL3 conversion pipeline for the published GLM / Qwen packs is private)', 'n/a', 'offline', lang='python'),
 K('executor.kyojin.dsv41_vision', ['exllamav3/dsv41/vision.py', 'exllamav3/dsv41/vision_spec.py'], 'vision.tower/deepseek-v4.1/bf16.kyojin', 'DS4.1 vision tower (image patches to embeddings for the DS4.1 trunk) and its tensor spec', 'DS4.1 images', 'bf16', lang='python'),
 # ---- serving -----------------------------------------------------------------------------------------------------
 K('serve.kyojin.glm_server', ['tools/glm/serve.py', 'tools/glm/serve_*.py', 'tools/glm/image_cache*.py', 'tools/glm/SERVE.md'], 'serve.http/openai/-.kyojin-glm',
   'GLM-5.3-Flash OpenAI-compatible server (aiohttp): chat template, streaming, MTP depth, request log, metrics, health ping, image embedding cache (memory and disk), acceptance metrics', 'the server behind the 26-32 tok/s MTP decode and 580-633 tok/s prefill numbers on one Strix Halo', 'n/a', 'serving', lang='python'),
 K('serve.kyojin.qwen_server', ['tools/qwen/serve.py', 'tools/qwen/serve_metrics.py', 'tools/qwen/serve_ping.py', 'tools/qwen/serve_reqlog.py', 'tools/qwen/image_cache.py', 'tools/qwen/SERVE.md', 'tools/qwen/spec_policy.py'], 'serve.http/openai/-.kyojin-qwen',
   'Qwen server (Qwen3.8-Flash-Next and the 27B): the same front with its speculative policy (draft policy mix, MTP / n-gram)', 'Qwen serving', 'n/a', 'serving', lang='python'),
 K('serve.kyojin.hy4_server', ['tools/hy4/serve.py', 'tools/hy4/serve_core.py', 'tools/hy4/engine.py', 'tools/hy4/mu_engine.py', 'tools/hy4/draft_policy.py', 'tools/hy4/depth_gate.py', 'tools/hy4/ngram.py', 'tools/hy4/spec2/'], 'serve.http/openai/-.kyojin-hy4',
   'Hy4-flash server (GGUF Q3_K, 64K context): engine and multi-session engine, speculative depth gate, draft policy, n-gram drafts, spec2 driver', 'GGUF serving on one Strix Halo (23-29 tok/s spec decode, README)', 'n/a', 'serving', lang='python'),
 K('serve.kyojin.serve_common', ['tools/kyojin_serve_split.py', 'tools/loopguard.py', 'tools/serve_compat.py', 'tools/l21_prompt_encode_cache.py', 'tools/startup_health.py', 'tools/admin_knobs.py', 'tools/slot_check.py'], 'serve.api/openai/-.kyojin-common',
   'Pieces the servers share: split flags, answer-phase loop guard, vLLM-style compatibility (token-id prompts, cache_salt, /tokenize), incremental prompt encode cache, /health during load, admin knob endpoint, slot check', 'common server behaviour', 'n/a', 'serving', lang='python'),
]


def resolve(origin, frm, name, quoted):
    cands = ([os.path.normpath(os.path.join(os.path.dirname(frm), name))] if quoted else []) + \
        [os.path.normpath(os.path.join(r, name)) for r in INC_ROOTS]
    return next((c for c in cands if c in origin.fileset), None)


INC_RX = re.compile(r'^[ \t]*#[ \t]*include[ \t]*(["<])([^">]+)[">]', re.M)


def includes(origin, f):
    return INC_RX.findall(open(origin.find(f), errors='replace').read())


def torch_bound(origin, f, memo):
    """True when f reaches a torch / pybind11 / CUDA-only header through its includes (inside the origin)."""
    if f in memo:
        return memo[f]
    memo[f] = False
    hit = False
    for q, name in includes(origin, f):
        if TORCH.match(name):
            hit = True
            break
        r = resolve(origin, f, name, q == '"')
        if r and torch_bound(origin, r, memo):
            hit = True
            break
    memo[f] = hit
    return hit


KRX = re.compile(r'__global__\s+(?:__launch_bounds__\s*\((?:[^()]|\([^()]*\))*\)\s*)?(?:static\s+)?(?:inline\s+)?void\s+(?:__launch_bounds__\s*\((?:[^()]|\([^()]*\))*\)\s*)?(\w+)\s*\(', re.S)


def kernel_names(origin, files):
    out = []
    for f in files:
        if f.endswith(('.cu', '.cuh', '.hip')):
            for k in KRX.findall(open(origin.find(f), errors='replace').read()):
                if k not in out and k != '__launch_bounds__':
                    out.append(k)
    return out


def providers_of_others(origin):
    """{module id: [interface, ...]} of the catalogue outside this origin."""
    out = {}
    cat = os.path.join(harvestlib.REPO, 'catalog')
    for o in sorted(os.listdir(cat)):
        if o == origin or not os.path.isdir(os.path.join(cat, o)):
            continue
        for d, _, fs in os.walk(os.path.join(cat, o)):
            if 'module.toml' in fs:
                t = tomllib.load(open(os.path.join(d, 'module.toml'), 'rb'))
                out[t['id']] = [x.get('interface', '') for x in t.get('provides', [])]
    return out


def destination(d, cls, lang, prov):
    """One clause for the module's why: where its shape would land in strixflerp (transfer shapes, not parts)."""
    how = {'python': 'a port of the python / Triton logic', 'cuda': 'a HIP port of the CUDA twin',
           'c++': 'the host code behind the flerp ABI', 'c': 'the host code behind the flerp ABI'}.get(lang, 'an adapter from its launcher to the flerp module ABI')
    if not cls:
        return 'no catalogue interface matches; it stays a reference for %s, kept beside the module it serves' % how
    op = cls.split('/')[0]
    same = lambda i: '/'.join([re.sub(r'-[a-z0-9_]+$', '', i.split('/')[0]) if i.split('/')[0] != op else op] + i.split('/')[1:2] + [i.split('/')[2].split('.')[0]]) == cls if i.count('/') == 2 else False
    beside = sorted(m for m, ifs in prov.items() if any(same(i) or i.split('/')[0] == op and '/'.join(i.split('/')[:2]) + '/' + i.split('/')[2].split('.')[0] == cls for i in ifs))
    if not beside:
        beside = sorted(m for m, ifs in prov.items() if any(i.split('/')[0] == op or i.split('/')[0].startswith(op + '-') for i in ifs))
    if beside:
        return '%s %s provider beside %s%s, through %s' % ('an' if cls[0] in 'aeiou' else 'a', cls, ', '.join(beside[:3]), (' and %d more' % (len(beside) - 3)) if len(beside) > 3 else '', how)
    return 'a new %s provider (no other origin provides %s yet), through %s' % (cls, op, how)


def lang_of(d, files):
    if d.get('lang'):
        return d['lang']
    if any(f.endswith(('.cu', '.cuh', '.hip')) for f in files):
        return 'hip-c++'
    if any(f.endswith(('.cpp', '.cc')) for f in files):
        return 'c++'
    if any(f.endswith('.c') for f in files):
        return 'c'
    if any(f.endswith(('.h', '.hpp')) for f in files):
        return 'hip-c++'
    return 'python'


def main():
    O = TreeOrigin('kyojin', 'kyojin', 'https://github.com/Yamz-Labs/kyojin', COMMIT, 'main', SPDX, AUTHORS, TOOLCHAIN)
    O.fileset = set(O.files())
    diff = read_base_diff()
    bad = [p for p, (st, kb) in diff.items() if st != 'D' and O.blobs.get(p) != kb]
    if bad:
        raise SystemExit('base-diff.tsv does not match the checkout: ' + ' '.join(bad[:10]))
    delta = {p for p, (st, _) in diff.items() if st != 'D'}
    in_scope = set(O.expand(SCOPE)) & delta

    owned = {}
    for d in MODS:
        # a pattern may also match unchanged base files (comp_units instances); only the delta is taken, and a plain
        # path that names an unchanged file is an error
        files = [f for f in O.expand(d['paths']) if f in delta]
        if not files:
            raise SystemExit(f'{d["id"]}: no Kyojin file matches {d["paths"]}')
        for p in d['paths']:
            if '*' not in p and not p.endswith('/') and p not in delta:
                raise SystemExit(f'{d["id"]}: {p} is not added or changed by Kyojin')
        for f in files:
            if f in owned:
                raise SystemExit(f'{f}: in both {owned[f]} and {d["id"]}')
            owned[f] = d['id']
        d['files'] = files
    missing = sorted(in_scope - set(owned))
    if missing:
        raise SystemExit(f'{len(missing)} in-scope Kyojin files are in no module: ' + ' '.join(missing[:30]))
    outside = sorted(set(owned) - in_scope)
    if outside:
        raise SystemExit('module files outside SCOPE: ' + ' '.join(outside[:10]))

    # unchanged base headers the delta's C / HIP files include: kernel.kyojin.ext_base, so the families compile standalone
    base_hdrs, todo, seen = set(), [f for f in owned if f.endswith(C_LIKE)], set()
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        for q, name in includes(O, f):
            r = resolve(O, f, name, q == '"')
            if r is None:
                continue
            if r in delta and r not in owned:
                raise SystemExit(f'{f} includes {r}, a Kyojin file in no module')
            if r not in delta:
                base_hdrs.add(r)
            todo.append(r)
    base_hdrs = sorted(base_hdrs)
    owner = dict(owned)
    for f in base_hdrs:
        owner[f] = 'kernel.kyojin.ext_base'

    prov = providers_of_others(O.name)
    memo = {}
    mods = []
    if base_hdrs:
        mods.append(M('kernel.kyojin.ext_base',
            'Unchanged ExLlamaV3 base files (turboderp-org/exllamav3 8c593e4e) that the Kyojin C / HIP files include: %d headers of exllamav3_ext' % len(base_hdrs),
            base_hdrs, [P('kernel-abi.kyojin-exl3-base/-/-', ', '.join(os.path.basename(f) for f in base_hdrs[:8]) + (' ...' if len(base_hdrs) > 8 else ''),
                          'n/a here: headers', 'compiled inside the kernel.kyojin.* units that include them')],
            why='SPEC.md gives headers that several modules include a module of their own; these are base files, taken only so that the Kyojin families compile standalone; the rest of the base is not harvested',
            notes='Interface tag: none. Files: the include closure that docs/harvest/harvest_kyojin.py computes from the Kyojin files; each is byte-identical to the base (base-diff.tsv has no row for it).',
            includes=['src/' + INC_ROOTS[0]], sources=[], compile='header-only', reason='header-only; compiled inside the units that include it',
            authors=['turboderp (ExLlamaV3)']))
    for d in MODS:
        files = d['files']
        lang = lang_of(d, files)
        compiled = {'hip-c++': ('.hip', '.cpp', '.cc', '.cu'), 'c++': ('.cpp', '.cc'), 'c': ('.c',)}.get(lang, ())
        if 'sources' in d:
            sources = d['sources']
        else:
            sources = [f for f in files if f.endswith(compiled)]
        reason, flags, toolchain = None, (), None
        tb = [s for s in sources if torch_bound(O, s, memo)]
        if lang == 'python':
            compile, toolchain = 'n/a', ''
            reason = 'python: not compiled' + ('; the Triton kernels in it are JIT-compiled at run time' if d['id'].startswith('kernel.') else '')
        elif lang == 'cuda':
            compile = 'n/a'
            reason = 'CUDA-only twin (sm_80+ tensor-core PTX, NVRTC at first use on NVIDIA); not a gfx1151 unit'
        elif not sources:
            compile, reason = 'header-only', 'header-only; compiled inside the units that include it'
        elif tb:
            compile = 'n/a'
            reason = ('needs the PyTorch C++ extension headers (torch / ATen / c10 / pybind11) of a ROCm torch build, not on the gate host: %s' % ', '.join(os.path.basename(s) for s in tb[:6]) + (' ...' if len(tb) > 6 else ''))
        elif lang in ('c++', 'c'):
            compile = 'n/a'
            reason = 'host %s, outside the HIP compile gate' % lang
        else:
            compile = 'standalone'
            flags = d.get('flags') or (JIT if d.get('jit') else ())
        knames = kernel_names(O, files)
        tag = d['tag']
        api = '%s%s' % (', '.join(os.path.basename(f) for f in files[:8]) + (' ...' if len(files) > 8 else ''),
                        ('; kernels: ' + ', '.join(knames[:10]) + (' ...' if len(knames) > 10 else '')) if knames else '')
        if tag:
            provides = [P(tag, api, d['numeric'], d['regime'])]
        else:
            provides = [P('kernel-abi.kyojin-%s/-/-' % d['id'].split('.', 2)[2].replace('_', '-'), api, d['numeric'], d['regime'])]
        added = [os.path.basename(f) for f in files if diff[f][0] == 'A']
        changed = [os.path.basename(f) for f in files if diff[f][0] == 'M']
        cls = None
        if tag:
            head, wf, dt = harvestlib.iface(tag).split('/')
            cls = '/'.join([re.sub(r'-kyojin.*$', '', head), wf, dt.split('.')[0]])
        notes = 'Interface tag: %s. Against the base turboderp-org/exllamav3 8c593e4e: %d added%s, %d changed%s. %s' % (
            cls or 'none', len(added), (' (%s)' % ', '.join(added[:6]) + (' ...' if len(added) > 6 else '')) if added else '',
            len(changed), (' (%s)' % ', '.join(changed[:6]) + (' ...' if len(changed) > 6 else '')) if changed else '', NOT_TIMED)
        if d.get('notes'):
            notes += ' ' + d['notes']
        if d.get('jit') and compile == 'standalone':
            notes += ' The origin builds it at first use with hipcc --genco --offload-arch=gfx1151 -O3 (-ffp-contract=off) into a code object; the gate compiles it device-only the same way.'
        incs = ['src/' + INC_ROOTS[0]] if lang != 'python' and any(f.startswith(EXT) for f in files) else []
        mods.append(M(d['id'], d['summary'], files, provides, why=d['why'] + '. Where the shape would go: ' + destination(d, cls, lang, prov), notes=notes, lang=lang, sources=sources,
                      includes=incs, compile=compile, reason=reason, flags=flags,
                      toolchain=toolchain, spdx=d.get('spdx'), upstream=d.get('upstream'),
                      source_flags=d.get('source_flags') if compile == 'standalone' else None))

    # requires from #include lines of every C-like file (harvestlib.include_requires does not read .cu / .cuh)
    for m in mods:
        need = []
        for f in O.expand(m.paths, m.exclude):
            if not f.endswith(C_LIKE):
                continue
            for q, name in includes(O, f):
                dep = owner.get(resolve(O, f, name, q == '"'))
                if dep and dep != m.id and 'module:' + dep not in need:
                    need.append('module:' + dep)
        m.requires = sorted(need)
    counts = {k: sum(1 for st, _ in diff.values() if st == k) for k in 'AMD'}
    delta_line = '%d rows: %d files added (A), %d changed (M), %d base files absent from Kyojin (D)' % (len(diff), counts['A'], counts['M'], counts['D'])
    run(O, mods, ORIGIN_TOML.replace('@@DELTA@@', delta_line), LEFT_OUT, ['LICENSE', 'THIRD_PARTY_NOTICES.md'])
    modules_md(O, mods, diff)


ORIGIN_TOML = '''
# Kyojin: Yamz Labs' fork of ExLlamaV3 for 100 GB-class models on one desktop machine, with the ROCm / Strix Halo (gfx1151) work
# that runs GLM-5.3-Flash on one Strix Halo (580-633 tok/s prefill, 26-32 tok/s MTP decode at 8K, served). Only what Kyojin adds or
# changes over turboderp-org/exllamav3 is harvested (card ORIGIN-KYOJIN 692a13d6d0).
name      = "kyojin"
repo      = "https://github.com/Yamz-Labs/kyojin"
commit    = "116a883ff213a842619bf633089b7e0083aa86ff"
branch    = "main"
version   = "1.9.3 (exllamav3/version.py); README figures measured on Kyojin 1.8.10 (Strix Halo) and 1.8.9 (RTX, Mac), Hy4-flash on 1.9.1"
base      = "turboderp-org/exllamav3 master 8c593e4e5c88da76a4aaf8f1b1838bb7c1fce4e0 (v1.6.7), the ark copy control-node:/data/ark-inbox/src/github.com__turboderp-org__exllamav3 (.git/HEAD ref master); the delta is docs/harvest/data/kyojin/base-diff.tsv, computed from the two ark copies. Kyojin's own files name exllamav3 1.6.0 as the version some kernels were ported from, so its fork point is older than this base: a base file that moved on upstream after the fork also shows as changed. base-diff.tsv: @@DELTA@@"
reference_checkout = "control-node:/data/ark-inbox/src/github.com__Yamz-Labs__kyojin#116a883ff213a842619bf633089b7e0083aa86ff (ark copy, a plain tree whose .git holds HEAD = 116a883 but no objects: the harvest checks every file against docs/harvest/data/kyojin/tree.tsv, their git blob ids)"
missing   = "the Yamz conversion pipeline that produced the published EXL3-Yamz packs (GLM-5.3-Flash-EXL3-Yamz, Qwen3.8-Flash-Next-EXL3-Yamz, DeepSeek-V4.1-Flash-EXL3-Yamz) is private and not in the repository; only the ExLlamaV3 converter and the DS4.1 pack tools are public"

[licence]
spdx      = "MIT"
files     = ["LICENSE", "THIRD_PARTY_NOTICES.md"]                  # copied verbatim next to this file
third_party = "THIRD_PARTY_NOTICES.md: rocm/gmmq and rocm/routed_f16 adapted from Gufo 07e1191 (MIT), itself from llama.cpp 5c0e946 (MIT); foreign_quant/ggml_mmvq and parts of fq_mmvq.cu / fq_mmq_few.cu from llama.cpp 7182ef7 / 61dbf49 / PR #31533 (MIT); qsa_prefill_wmma.hip follows Gufo's WmmaCausalAttentionKernel structure (no source copied). vendor/fla is flash-linear-attention (MIT); vendor/fla/hip/gdn_hf.hip and kda_hf.hip are adapted from hipfire (Apache-2.0; LICENSE.hipfire-APACHE and NOTICE.hipfire ride in kernel.kyojin.fla_gdn_hf, and those two modules carry spdx Apache-2.0). The base ExLlamaV3 is MIT (Copyright (c) 2025 Turboderp). The presets/ decensor tensors and the published weights are not harvested"

[toolchain]
host      = "Python 3.12, PyTorch ROCm 7 wheels; the extension builds through setup.py (torch.utils.cpp_extension, hipify on ROCm, -O3 -DUSE_ROCM); JIT HIP kernels through exllamav3/util/hip_compiler.py's hipcc (the rocm-sdk wheel's clang 23 by default)"
device    = "therock-10.2"                              # the rocm-sdk wheels of doc/install.md are TheRock builds; the gate uses therock-10.2
build     = "bash build.sh / pip install -e . (the torch extension exllamav3_ext); tools/hy4/install_strix.sh (Hy4 kernels); hipcc --genco at first use for attention_fn/*.hip, dsv41_fn/*.hip, gr_fn, ple_fn, vendor/fla/hip, hip_kq and hipsp"

[compile]                                                # clang -x hip, one object per translation unit
driver    = "<rocm>/lib/llvm/bin/clang++"
arch_flag = "--offload-arch=gfx1151"
rocm_flag = "--rocm-path=<rocm>"
flags     = ["-x", "hip", "-std=c++17", "-O3", "-DUSE_ROCM", "-D__HIP_PLATFORM_AMD__=1", "-fPIC", "-c"]
system_includes = ["include"]
output    = "host object with the gfx1151 device code embedded (.o) for extension units; a device-only object for the JIT kernels (the origin's hipcc --genco code objects)"
note      = "extension units: setup.py's ROCm flags (-O3 -DUSE_ROCM) on the hipified sources; the gate compiles only units that reach no torch / ATen / c10 / pybind11 header (the others need a ROCm PyTorch build's C++ headers). JIT units: [build] options carry -ffp-contract=off and --offload-device-only, as exllamav3/util/hip_compiler.py and the loaders build them"

[kernels]
loading   = "two ways: the torch extension exllamav3_ext (pybind11, every op bound in bindings.cpp; gm_ext a second small one) and hipcc --genco code objects loaded with the HIP module API by the python loaders (qdec_hip.py, qsa_prefill_hip.py, dsv41_*.py, gr_mix_hip.py, ple_hip.py, vendor/fla/hip/*_hip.py, hipsp/sp_hip.py, loader/moe_dec_hip.py); Triton kernels JIT at run time with a shipped seed cache"
dispatch  = "python modules pick a path by shape, format and flag; environment switches (EXL3_*, HY4_*) and run-time knobs (exllamav3/util/knobs.py) select levers; graph-captured blocks (libtorch/*_bc.h, loader/step_graph.py) replay whole decode steps"
models    = "GLM-5.3-Flash (EXL3-Yamz pack, MTP; one Strix Halo: 32 tok/s decode, 633 tok/s prefill at 8K served, README of 1.8.10; the card cites 580 tok/s prefill and 26-30 tok/s MTP decode), Qwen3.8-Flash-Next (53 / 1,576), Hy4-flash GGUF Q3_K (spec decode 23-29 tok/s, prefill 928-987 at 4K-64K), DeepSeek-V4.1-Flash, Qwen3.8-27B on RTX 3090"
'''

LEFT_OUT = {
    'exllamav3/architecture/': 'model architecture definitions (42 changed or added); model code, not kernels or runtime pieces',
    'exllamav3/modules/*.py (changed base modules: attn, linear, mlp, block_sparse_mlp*, gated_delta_net, mla_attn, transformer, ...), exllamav3/modules/arch_specific/, exllamav3/modules/quant/': 'the engine module graph as Kyojin changed it; the kernels and runtime pieces they call are harvested',
    'exllamav3/conversion/, convert.py, util/': 'the ExLlamaV3 converter as Kyojin changed it; the Yamz conversion pipeline behind the published packs is private and not in the repository',
    'exllamav3/model/ (other than the harvested files), exllamav3/util/ (other than the harvested files), exllamav3/tokenizer/, exllamav3/integration/, exllamav3/ext.py, exllamav3/model_init.py, exllamav3/__init__.py, exllamav3/version.py': 'engine plumbing changed for the fork',
    'exllamav3/kernel_seed/': 'a pre-compiled Triton cache tarball (binary, regenerated by tools/kernel_seed); runtime.kyojin.triton_seed holds its loader',
    'tests/, eval/, examples/, science/, ref/': 'tests, evaluations, examples and CPU references; not run here',
    'tools/ (everything but the serve files in serve.kyojin.*)': 'benchmarks, probes, A/B scripts, qualification, conversion and release tooling',
    'doc/, docs/, assets/, *.md, presets/, docker/, scripts/, .github/, nix files, build.sh, env.sh, requirements*, pyproject.toml, setup.py, uv.lock': 'documentation, decensor presets (tensors), containers, CI and packaging',
    'LICENSE, THIRD_PARTY_NOTICES.md': 'copied beside ORIGIN.toml',
}


def modules_md(O, mods, diff):
    BT = chr(96)
    prov = {}
    cat = os.path.join(harvestlib.REPO, 'catalog')
    for o in sorted(os.listdir(cat)):
        if o == O.name or not os.path.isdir(os.path.join(cat, o)):
            continue
        for d, _, fs in os.walk(os.path.join(cat, o)):
            if 'module.toml' in fs:
                t = tomllib.load(open(os.path.join(d, 'module.toml'), 'rb'))
                prov[t['id']] = [x.get('interface', '') for x in t.get('provides', [])]
    q = lambda s: BT + s + BT
    lines_total = 0
    o = ['# Kyojin modules by strixflerp interface', '',
         'One row for each module of catalog/kyojin: what Yamz-Labs/kyojin at 116a883 adds (A) or changes (M) against its base',
         'turboderp-org/exllamav3 8c593e4e (%s), split by family. Card ORIGIN-KYOJIN 692a13d6d0.' % q('docs/harvest/data/kyojin/base-diff.tsv'),
         'Written by %s; read from source, nothing here was run.' % q('docs/harvest/harvest_kyojin.py'), '',
         '- **tag**: the strixflerp interface class the module matches, or none (its [[provides]] then names a kernel-abi.kyojin-* shape).',
         '- **same op in the catalogue**: modules of other origins that provide that family.op; the same op is not the same call.', '',
         '| module | lang | files (A/M) | lines | compile | tag | same op in the catalogue |', '|---|---|---|---:|---|---|---|']
    tagged = 0
    for m in mods:
        files = O.expand(m.paths, m.exclude)
        n = sum(len(open(O.find(f), 'rb').read().splitlines()) for f in files)
        lines_total += n
        a = sum(1 for f in files if diff.get(f, ('=',))[0] == 'A')
        c = sum(1 for f in files if diff.get(f, ('=',))[0] == 'M')
        iface = m.provides[0]['interface']
        if iface.startswith('kernel-abi.kyojin'):
            cls, beside = None, []
        else:
            tagged += 1
            head, wf, dt = iface.split('/')
            op = re.sub(r'-kyojin.*$', '', head)
            cls = '/'.join([op, wf, dt.split('.')[0]])
            beside = sorted(mid for mid, ifs in prov.items() if any(i.split('/')[0] == op or i.split('/')[0].startswith(op + '-') for i in ifs))
            more = len(beside) - 6
            beside = beside[:6] + (['+%d more' % more] if more > 0 else [])
        o.append('| %s | %s | %d (%d/%d) | %d | %s | %s | %s |' % (q(m.id), m.lang, len(files), a, c, n, m.compile,
                 q(cls) if cls else 'none', ', '.join(b if b.startswith('+') else q(b) for b in beside) or '-'))
    o += ['', '%d modules, %d carry a tag, %d are none; %d lines.' % (len(mods), tagged, len(mods) - tagged, lines_total), '',
          '## Not done here', '', 'Nothing was run or timed. The compile gate (gate/compile/run.py on control-node, CPU only) compiles the HIP',
          'modules that reach no PyTorch header; the torch-extension units need a ROCm PyTorch build\'s C++ headers and stay harvested.',
          'The Yamz conversion pipeline behind the published EXL3-Yamz packs is private and missing.']
    open(os.path.join(harvestlib.REPO, 'catalog', O.name, 'MODULES.md'), 'w').write('\n'.join(o) + '\n')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--base-diff':
        write_base_diff(sys.argv[2])
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit(__doc__)
