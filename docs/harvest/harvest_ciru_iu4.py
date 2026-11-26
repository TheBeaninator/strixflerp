#!/usr/bin/env python3
"""Harvest Ciru's Qwen3.8-Flash Strix runtime (ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4, 47cb923) into catalog/ciru_iu4.
Run: python3 docs/harvest/harvest_ciru_iu4.py
     python3 docs/harvest/harvest_ciru_iu4.py --base-diff <the base: ark copy or git checkout>   # rewrite data/ciru_iu4/base-diff.tsv

Card ORIGIN-CIRU-IU4 314b9e37c3. The origin is a llama.cpp tree: the CIRU runtime is pwilkin's Strix Halo reference
engine (pwilkin/llama.cpp f5daaa3c, "reference_commit" of docs/qualification/v4.0.0/source-identity.json and
"base_revision" of docs/qualification/v3.1.0/BUILD-SPEC.json, itself a llama.cpp fork) plus what Ciru added and changed
on top. Only that delta is harvested: every file of ggml/src/ggml-cuda and ggml/src/ggml-hip that the CIRU tree adds or
changes against the base is in exactly one kernel family module, the C1 (IU4) weight type is a format module and the
CIRUPLE1 pager another; the unchanged base headers those files include ride in kernel.ciru_iu4.cuda_common so that each
family compiles standalone. data/ciru_iu4/base-diff.tsv lists the delta (status, path, base blob, CIRU blob); this
script checks the CIRU blobs against the checkout and that the kernel modules cover the ggml-cuda / ggml-hip rows.

The reference checkout is the ark copy on control-node (/data/ark-inbox/src/github.com__ciru-ai__Qwen3.8-Flash-CIRU-STRIX-IU4
#47cb9239fa4913aa5a7c4b36dc21ca63e69da184, ark 7546d7e), the default; run the script on control-node, or name another copy
of the same commit with CIRU_IU4_CHECKOUT. The base is in the ark too (ark a3db8d4): on control-node,
  python3 docs/harvest/harvest_ciru_iu4.py --base-diff '/data/ark-inbox/src/github.com__pwilkin__llama.cpp#f5daaa3cfa6358e5dd398911ec741813745a5440'
rewrites base-diff.tsv from the two ark copies. Nothing here compiles or runs;
gate.status = "builds" comes from gate/results/compile (gate/compile/run.py on control-node, CPU only)."""
import hashlib, os, re, subprocess, sys, tomllib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvestlib
from harvestlib import Origin, M, P, run

ARK = '/data/ark-inbox/src/github.com__ciru-ai__Qwen3.8-Flash-CIRU-STRIX-IU4#47cb9239fa4913aa5a7c4b36dc21ca63e69da184'
CHECKOUT = os.environ.get('CIRU_IU4_CHECKOUT', ARK)
COMMIT = '47cb9239fa4913aa5a7c4b36dc21ca63e69da184'
BASE = 'f5daaa3cfa6358e5dd398911ec741813745a5440'
BASE_DIFF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'ciru_iu4', 'base-diff.tsv')
TREE = os.path.join(os.path.dirname(BASE_DIFF), 'tree.tsv')     # git blob id of every file of the commit
CU = 'ggml/src/ggml-cuda/'


def git(*args, cwd=None):
    return subprocess.run(['git', '-C', cwd or CHECKOUT] + list(args), capture_output=True, text=True, check=True).stdout


def blobs(checkout):
    out = {}
    for line in git('ls-files', '-s', cwd=checkout).splitlines():
        meta, path = line.split('\t', 1)
        out[path] = meta.split()[1]
    return out


def blob_id(path):
    data = open(path, 'rb').read()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def tree_blobs(root):
    """{path: git blob id} of a plain tree (an ark copy), a top-level .git ignored."""
    out = {}
    for d, dirs, fs in os.walk(root):
        if d == root:
            dirs[:] = [x for x in dirs if x != '.git']
            fs = [x for x in fs if x != '.git']
        for f in fs:
            p = os.path.join(d, f)
            out[os.path.relpath(p, root)] = blob_id(p)
    return out


def write_base_diff(base):
    """data/ciru_iu4/base-diff.tsv from the commit (CIRU_IU4_CHECKOUT) and the base (pwilkin/llama.cpp f5daaa3c).
    Each side is a git checkout (HEAD must be the pinned commit) or a plain tree such as the ark copies on control-node
    (/data/ark-inbox/src/github.com__pwilkin__llama.cpp#f5daaa3c..., ark a3db8d4, whose leftover .git directory is not a
    repository and is skipped), read file by file. A git checkout of
    the commit also rewrites tree.tsv; a plain tree of it must match tree.tsv first."""
    sides = []
    for d, want in ((CHECKOUT, COMMIT), (base, BASE)):
        r = subprocess.run(['git', '-C', d, 'rev-parse', '--show-toplevel', 'HEAD'], capture_output=True, text=True)
        lines = r.stdout.split()
        if r.returncode == 0 and os.path.realpath(lines[0]) == os.path.realpath(d):   # a git checkout of its own
            head = lines[1]
            if head != want:
                raise SystemExit(f'{d} is at {head}, want {want}')
            sides.append((blobs(d), 'git checkout'))
        else:
            sides.append((tree_blobs(d), 'plain tree ' + d))
    (ours, how_ours), (theirs, how_base) = sides
    if how_ours == 'git checkout':
        with open(TREE, 'w') as f:
            f.write('# git ls-files -s of ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4 %s: blob id and path of every file.\n' % COMMIT)
            for p in sorted(ours):
                f.write('%s\t%s\n' % (ours[p], p))
    elif ours != read_tree():
        raise SystemExit(f'{CHECKOUT} is not the tree of {COMMIT} (data/ciru_iu4/tree.tsv)')
    rows = []
    for p in sorted(set(ours) | set(theirs)):
        if p not in theirs:
            rows.append(('A', p, '-', ours[p]))
        elif p not in ours:
            rows.append(('D', p, theirs[p], '-'))
        elif ours[p] != theirs[p]:
            rows.append(('M', p, theirs[p], ours[p]))
    os.makedirs(os.path.dirname(BASE_DIFF), exist_ok=True)
    with open(BASE_DIFF, 'w') as f:
        f.write('# ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4 %s against its base pwilkin/llama.cpp %s (git blob ids).\n' % (COMMIT, BASE))
        f.write('# Written by docs/harvest/harvest_ciru_iu4.py --base-diff; A added, M changed, D removed by CIRU.\n')
        f.write('# base read from: %s\n' % how_base)
        f.write('status\tpath\tbase_blob\tciru_blob\n')
        for r in rows:
            f.write('\t'.join(r) + '\n')
    print(f'{BASE_DIFF}: {len(rows)} rows ({len(theirs)} base files, {how_base})')


def read_base_diff():
    rows = {}
    for line in open(BASE_DIFF):
        if line.startswith('#') or line.startswith('status\t'):
            continue
        st, path, _, cb = line.rstrip('\n').split('\t')
        rows[path] = (st, cb)
    return rows


def read_tree():
    return dict(reversed(l.rstrip('\n').split('\t', 1)) for l in open(TREE) if not l.startswith('#'))


class TreeOrigin(Origin):
    """The checkout (the ark copy: a plain tree, no .git) must hold exactly the files of COMMIT, each with the git blob id
    data/ciru_iu4/tree.tsv records; a .git at the top is ignored, as flerp diff-origin does."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.roots = [CHECKOUT]
        self.blobs = read_tree()
        have = self.files()
        missing, extra = sorted(set(self.blobs) - set(have)), sorted(set(have) - set(self.blobs))
        wrong = [p for p in have if p in self.blobs and blob_id(os.path.join(CHECKOUT, p)) != self.blobs[p]]
        if missing or extra or wrong:
            raise SystemExit(f'{CHECKOUT} is not the tree of {COMMIT}: missing {missing[:5]}, extra {extra[:5]}, differing {wrong[:5]}')

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


AUTHORS = ['Ciru (ciru-ai)', 'pwilkin', 'llama.cpp / ggml contributors']
NOT_TIMED = 'No timing was taken on the pod.'
INC = ['src/' + CU.rstrip('/'), 'src/ggml/src', 'src/ggml/include']   # the ggml-cuda target's include roots
INC_ROOTS = [CU.rstrip('/'), 'ggml/src', 'ggml/include']


def F(file, kernels, tag, what, changed=None):
    """One CIRU file of a family. kernels: its device entry points (new ones, for a changed file). tag: the strixflerp
    interface it matches (full name; the class is the name without its variant tag) or None. changed: for a file CIRU
    changed rather than added, a line on what the change is."""
    return dict(file=file, kernels=kernels, tag=tag, what=what, changed=changed)


FAMILIES = {
 'c1': dict(
  summary='CIRU\'s native IU4 routed-expert projections for the C1 weight type (GGML_TYPE_C1_G16_K2560 / K640): MUL_MAT_ID '
          'over 4-bit codes with a G16 affine table, run on the gfx1151 iu4 WMMA (__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32) or '
          'its same-value IU8 arm; vector paths (one wave per row, or four rows sharing an activation packet) for up to eight '
          'tokens, a grouped WMMA projection (four N16 waves share each per-expert table) for longer contiguous batches, and an '
          'F32 reference kernel',
  why='the IU4 expert format is what lets CIRU hold Qwen3.8-Flash-Next\'s routed experts at 4.5175 bits a weight and multiply them '
      'on the integer WMMA without dequantizing; no catalogue kernel reads C1, and the other iu4 WMMA users are hipfire\'s, on its own formats '
      '(qwen4_moe_iu4_sym.gfx1151.hip in kernel.hipfire.moe, the *_iu4 kernels of kernel.hipfire.gemm_mq), the nearest thing to compare against',
  files=[
   F('c1.cu', ['c1_pack_q8_runtime', 'c1_vector_runtime', 'c1_mat_id_reference'], 'moe.experts-decode/c1-g16/f32.ciru-iu4',
     'the C1 translation unit: packs the activation to Q8 (K32 split, centred or not), runs the vector paths for short batches '
     'and the grouped projection for longer ones, and keeps the F32 reference; ggml_cuda_mul_mat_id_c1 is the entry the backend calls'),
   F('c1.cuh', [], None, 'declares ggml_cuda_mul_mat_id_c1'),
   F('c1-grouped.cuh', ['project', 'route_count', 'route_describe', 'route_scatter'], 'moe.experts-prefill/c1-g16/f32.ciru-iu4',
     'grouped C1 projection for more than eight tokens: routes are counted and scattered per expert, then four N16 waves share '
     'each expert\'s table in a CTA; the frozen 4.5175-bpw GGUF values are consumed directly'),
   F('c1-vector-r4.cuh', ['c1_vector_r4_runtime', 'c1_vector_r4_lut_runtime'], 'moe.experts-decode/c1-g16/f32.ciru-iu4-r4',
     'four adjacent output rows share one activation packet and one wave (with and without a table LUT); the reduction matches the one-row path'),
  ],
  sources=['c1.cu'],
  numeric='weights: per output row K/2 bytes of 4-bit codes then K/16 bytes of table indices (one index per 16 values into a '
          'per-expert FP16 affine table [2, 256]), times an FP16 row multiplier; activations packed to int8 (c1_pack_q8_runtime), '
          'integer dot on the iu4 or iu8 WMMA, the sum correction applied with an FMA after rounding the dot (file comments)',
  regime='Qwen3.8-Flash-Next routed experts: gate / up K2560, down K640; vector paths up to 8 tokens, grouped WMMA above',
 ),
 'mmb': dict(
  summary='CIRU\'s additions to the pwilkin MMB (BF16 WMMA matrix-batch) path: routed C1 expert kernels with and without the '
          'fused GLU, a Q5_K tile loader from HaloBox for prefill, the hyper-connection down / inject Q8_0 pair kernel, and the '
          'BF16 v4 merge of the routed MoE weighted reduction',
  why='these are the prefill-side expert and dense kernels CIRU\'s v5 / v6 releases measure; the C1 MMB path and the HC Q8_0 pair '
      'have no counterpart in the catalogue, and the Q5_K loader is HaloBox\'s, which is not otherwise harvested',
  files=[
   F('mmb.cu', ['mmb_routed_c1_kernel', 'mmb_routed_c1_glu_kernel', 'mmb_f32_halo_kernel', 'mmb_cvt_bf16_f16'],
     'moe.experts-prefill/c1-g16/bf16.ciru-mmb',
     'MMB translation unit; CIRU adds the routed C1 kernels (the C1 row decoded into the BF16 WMMA tile, LLAMA_C1_MMB / '
     'LLAMA_C1_GROUP4 / LLAMA_C1_PACKED_HALF), the HaloBox 25fda3478 Q5_K tile loader (mmb_f32_halo_kernel, CIRU_HALO_PREFILL / '
     'CIRU_HALO_Q5_MMB) and ggml_cuda_mmb_hc_q8_pair', changed='+708 -46 lines against the base'),
   F('mmb.cuh', [], None, 'MMB declarations; CIRU adds ggml_cuda_mmb_hc_q8_pair', changed='+1 line'),
   F('hc_q8_pair.cuh', [], None, 'interface of the HC Q8_0 pair prototype (Wdown [10240, 320] and Winject [10240, 4] as separate GGUF Q8_0 matrices)'),
   F('hc_q8_pair_kernel.cuh', ['hc_q8_pair_kernel'], 'hc.mix-prefill/gguf-q8_0/f32.ciru-q8-pair',
     'one kernel computes the hyper-connection down projection and the inject projection from the MMB BF16 activation with '
     'the exact mmb_dq_row68 arithmetic, storing both GGML F32 outputs (CIRU_HC_Q8_FUSE / LLAMA_HC_Q8_PAIR)'),
   F('moe-weighted-reduction.cu', ['moe_weighted_reduction_bf16_v4_merge'], 'moe.combine/-/bf16.ciru-v4-merge',
     'routed-expert weighted reduction; CIRU adds a BF16 v4 kernel that merges the following ADD (rounded after the reduction, as the unfused graph)',
     changed='+43 -1 lines'),
  ],
  sources=['mmb.cu', 'moe-weighted-reduction.cu'],
  numeric='BF16 WMMA with F32 accumulation; C1 and Q8_0 rows are decoded to BF16 (round to nearest even) in the LDS tile; the '
          'Q5_K loader builds q+32 and subtracts 32 packed, exact for q in [0, 31] (file comments)',
  regime='prefill: routed experts above LLAMA_C1_MMB_MIN_T tokens, HC down / inject with T up to 128-token tiles',
 ),
 'q5': dict(
  summary='CIRU\'s Q5_K kernels for the dense layers: a short-batch exact FP32 GEMM adapted from Gufo\'s '
          'SmallBatchKQuantExactFp32GEMMKernel (a 4-token kernel and an opt-in 5-token one, wave64), the hook that routes MMVQ to it, an isolated '
          'Q5_K wave64 prefill GEMM around the unmodified Gufo 990fdce WKQuantA8BlockedWmmaGEMMKernel excerpt, and the Q8_1 '
          'activation quantizer change that keeps the stored scale for both the dot and its offset correction',
  why='short batches of 4 and 5 tokens are the MTP verify shape (draft depth 3 and 4), and CIRU replaces MMVQ there with an exact FP32 path; '
      'the wave64 GEMM is a default-off experiment that carries a Gufo kernel into ggml, comparable with kernel.gufo.qwen_linear_prefill',
  files=[
   F('q5_short.cu', ['q5_short_f32_t4_kernel', 'q5_short_f32_t5_kernel'], 'linear.verify/gguf-q5k/f32.ciru-short',
     'Gufo-derived short-batch Q5_K x F32 GEMM, a 4-token kernel and a 5-token one behind CIRU_Q5_SHORT_F32_T5, compiled wave64 with -fno-fast-math (CIRU_Q5_SHORT_GFX1151); MIT notice of Gufo in the file'),
   F('q5_short.cuh', [], None, 'declares ggml_cuda_q5_short_f32'),
   F('q5_wave64.cu', ['pack_q8_and_sums', 'zero_q8_tail'], 'linear.prefill/gguf-q5k/f32.ciru-wave64',
     'Q5_K decode, Gufo-format Q8 activation packing with sums and launch glue around the Gufo GEMM; default absent from every serving runtime (CIRU_Q5_GUFO_WAVE64)'),
   F('q5_wave64.cuh', [], None, 'declares the wave64 operator (scratch from the ggml HIP pool, the current stream)'),
   F('donor_wkquant_kernel.cuh', ['WKQuantA8BlockedWmmaGEMMKernel'], None,
     'unmodified excerpt of Gufo 990fdce src/models/qwen/hip/kernels/prefill_quant_gemm.hpp, compiled inside q5_wave64.cu'),
   F('mmvq.cu', [], 'linear.decode/gguf-kquant/f32.ciru-mmvq', 'ggml\'s MMVQ; CIRU routes Q5_K short batches to ggml_cuda_q5_short_f32', changed='+9 lines'),
   F('quantize.cu', [], None, 'ggml\'s activation quantizer; CIRU_Q5_CONSISTENT_SUM uses the stored scale for both the integer dot and its offset correction', changed='+23 -4 lines'),
  ],
  sources=['q5_short.cu', 'q5_wave64.cu', 'mmvq.cu', 'quantize.cu'],
  source_flags={'q5_short.cu': ['-mwavefrontsize64', '-O2', '-fno-fast-math', '-DCIRU_Q5_SHORT_GFX1151=1'], 'q5_wave64.cu': ['-mwavefrontsize64']},
  numeric='Q5_K weights decoded exactly; q5_short multiplies in FP32 (exact FP32 GEMM, no activation quantization); the wave64 GEMM '
          'quantizes activations to Q8 with block sums (Gufo A8 format)',
  regime='q5_short: 4 or 5 tokens; wave64 GEMM: prefill, off by default',
 ),
 'attn': dict(
  summary='CIRU\'s attention changes on the pwilkin tree: the QSA3 sparse attention reworked (one 128-lane workgroup a query '
          'merging sorted selected rows, query grouping), a per-query selected-key QSA decode path for MTP and appends of up to '
          '512 tokens (plain and WMMA partials with a merge), the compact lightning-indexer form whose mask comes from per-row '
          'limits, an index-mask fallback in flash attention, mask-bound fixes in the shared FA code and the MMA kernel, and an '
          'argsort for rows longer than a block (QSA\'s 2051 selected cells, wide image prompts)',
  why='QSA is Qwen3.8-Flash-Next\'s sparse attention; CIRU\'s decode path and grouping are what its "sequence-safe QSA" and '
      'concurrent MTP releases measure, against kernel.strixite.qsa and the other QSA providers',
  files=[
   F('qsa.cu', ['qsa3_launch'], 'attn.softmax-gqa/kv-f16/f32.ciru-qsa3',
     'QSA3 sparse attention for long ubatches; CIRU reworks kernel B (one workgroup of 128 lanes a query, each lane merging the '
     'sorted selected rows of its key range) and adds query grouping (CIRU_QSA_QUERY_GROUP)', changed='+107 -71 lines'),
   F('qsa-decode.cu', ['qsa_decode_unique_ids', 'qsa_decode_partial', 'qsa_decode_merge'], 'attn.softmax-gqa/kv-f16/f32.ciru-qsa-decode',
     'per-query selected-key attention for MTP and appends up to 512 (516) rows: unique selected ids, split partials over the selection, then a merge'),
   F('qsa-decode.cuh', [], None, 'declares the decode path and its support test (replays up to four cached tokens before a 512-token append)'),
   F('qsa-decode-wmma.cuh', ['qsa_decode_wmma_partial'], 'attn.softmax-gqa/kv-f16/f32.ciru-qsa-decode-wmma', 'WMMA form of the decode partials'),
   F('lightning-indexer.cu', [], 'attn.qsa-indexer/-/f32.ciru-compact',
     'QSA lightning indexer; CIRU adds the compact form (ggml_lightning_indexer_compact: a key is visible when its limit is below the query\'s, instead of an F16 mask)',
     changed='+25 -8 lines'),
   F('fattn.cu', ['flash_attn_index_mask_set', 'flash_attn_index_mask_clear', 'flash_attn_index_mask_empty'], 'attn.softmax-gqa/kv-f16/f32.ciru-fattn',
     'flash-attention dispatch; CIRU adds the QSA decode route (LLAMA_QSA_PER_QUERY), an index-mask fallback with bounded scratch when QSA3 declines a large prefill, and F32 accumulation for the 256-wide RDNA3.5 path',
     changed='+95 lines'),
   F('fattn-common.cuh', [], None, 'shared FA code; CIRU fixes mask views that hold exactly the live query rows, matches sequence broadcasting, and adds CIRU_FA_LEGACY_LDS_OCCUPANCY (the 64 KiB LDS occupancy bound of HIP 7.15 for reproducible split order)',
     changed='+23 -3 lines'),
   F('fattn-mma-f16.cuh', [], None, 'MMA flash attention; CIRU checks every mask element unless a finite probe keeps a dense tile on the normal path', changed='+19 lines'),
   F('argsort.cu', [], 'attn.indexer-topk/-/i32.ciru-wide-argsort', 'bitonic argsort; CIRU lets a row exceed the per-block thread limit (each thread handles disjoint pairs per stage)', changed='+26 -25 lines'),
  ],
  sources=['qsa.cu', 'qsa-decode.cu', 'lightning-indexer.cu', 'fattn.cu', 'argsort.cu'],
  numeric='F16 K/V (the release runs an f16 KV cache), F32 softmax and accumulation; QSA selects keys per query from the indexer scores',
  regime='QSA3 for long ubatches, the per-query decode path for MTP and appends up to 512 tokens; dense flash attention otherwise',
 ),
 'gdn': dict(
  summary='CIRU\'s chunked Gated DeltaNet prefill for RDNA3.5 (S_v = 128, up to eight trailing state slots, one sequence): a '
          'prep kernel builds the in-chunk T\' = (I + A)^-1 diag(beta) and P, a scan kernel carries the state across chunks and '
          'writes the output with its gated RMS norm (and an optional BF16 copy), a third kernel stores the trailing state '
          'snapshots MTP rollback needs; the q / k L2 norms and the short conv before the GDN node are folded in',
  why='CIRU names the chunked GDN gate as its v5.0 prefill lead (after HaloBox c750386bb); it is the third chunked GDN prefill in the '
      'catalogue after kernel.crucible.gdn and libstrix\'s, and the only one that also folds the norms and keeps MTP snapshots',
  files=[
   F('gated_delta_net.cu', ['gdn_chunk_prep_kernel', 'gdn_chunk_scan_kernel', 'gdn_chunk_tail_snapshots_kernel'], 'mixer.gdn-prefill/-/f32.ciru-chunk',
     'chunked prefill beside the base recurrent kernel; GGML_GDN_CHUNK / GGML_GDN_CHUNK_MIN gate it, CIRU_GDN_MULTI_TILE is an off-by-default variant', changed='+813 -9 lines'),
   F('gated_delta_net.cuh', [], None, 'declares the eligibility test and the q / k norm, conv and output-norm registrations the backend makes before the node', changed='+16 lines'),
  ],
  sources=['gated_delta_net.cu'],
  numeric='F32 state and arithmetic; per chunk A[r][s] = beta_r (k_r . k_s) exp(G_r - G_s), U = T\' (V - diag(e^G) K S0), O = scale diag(e^G) Q S0 + P U (file header)',
  regime='prefill of one sequence, head_v 128, non-KDA; decode stays on the base recurrent kernel',
 ),
 'ple_conv': dict(
  summary='CIRU\'s change to pwilkin\'s fused PLE causal convolution (ple_conv_kernel and ple_concat_tail, matched in the graph '
          'from CONCAT / CONT taps / SILU): the depthwise weights may be BF16 as well as F16 (a template flag decodes the raw 16 bits), and a log switch',
  why='part of the Flash-Next PLE path whose exact rounding CIRU qualifies against; small, but a changed kernel file',
  files=[
   F('ple-conv.cu', [], None, 'fused width-4, dilation-3 depthwise causal convolution of the PLE stream with its carried tail; CIRU accepts BF16 weights beside F16 and adds CIRU_PLE_CONV_LOG', changed='+24 -6 lines'),
  ],
  sources=['ple-conv.cu'],
  own=('ple.conv-causal/f16+bf16/f32.ciru', 'ple-conv.cu: ple_conv_kernel, ple_concat_tail (graph match in ggml-cuda.cu)'),
  numeric='F32 state and output; F16 or BF16 weights widened to F32; multiply and add rounded separately on AMD (the base\'s inline asm)',
  regime='Qwen3.8-Flash-Next PLE stream, prefill and decode',
 ),
 'backend': dict(
  summary='The CIRU form of ggml-cuda.cu, the HIP backend\'s op dispatch and graph evaluation (+480 lines over the base): exact '
          'short batches (CIRU_EXACT_BATCH_MAX), routing to the C1, Q5_K, QSA-decode, chunked-GDN and HC kernels with the graph '
          'matches that feed them, and HC capture / merge / replay helpers (hc-*.inc, fixture.h; the shared device header common.cuh '
          'stays in kernel.ciru_iu4.cuda_common); plus ggml-hip/CMakeLists.txt with the gfx1151 per-source options of the Q5_K units',
  why='every CIRU kernel is reached through this file, and its environment switches are the knobs CIRU\'s qualified launch '
      'profiles set; an adapter that lifts a CIRU kernel out of ggml starts from the call here',
  files=[
   F('ggml-cuda.cu', [], None, 'the backend translation unit; CIRU adds the exact-batch path, the C1 / Q5_K / HC / GDN routing and graph matching', changed='+480 -9 lines'),
   F('hc-capture.inc', [], None, 'hyper-connection capture and replay priming (ciru_hc_capture_step / output / replay_prime / save_bytes), included in ggml-cuda.cu'),
   F('hc-merge.inc', [], None, 'exact HC merge scheduling (ciru_hc_exact_merge, ciru_hc_schedule_shared), included in ggml-cuda.cu'),
   F('hc-replay.inc', [], None, 'diagnostic exports for the standalone component driver, included in ggml-cuda.cu'),
   F('fixture.h', [], None, 'fixture record type the HC capture writes'),
  ],
  extra=['ggml/src/ggml-hip/CMakeLists.txt'],
  sources=['ggml-cuda.cu'],
  numeric='n/a here: each kernel module states its own arithmetic',
  regime='host dispatch of the ggml graph on the gfx1151 HIP backend',
 ),
}

KINDS_ORDER = ['c1', 'mmb', 'q5', 'attn', 'gdn', 'ple_conv', 'backend']
SPDX = 'MIT'


FORMATS = {
 'c1_g16': dict(
  summary='The C1 weight type CIRU adds to ggml for its IU4 routed experts: GGML_TYPE_C1_G16_K2560 (block 2560 values in 1440 '
          'bytes) and GGML_TYPE_C1_G16_K640 (640 in 360) in ggml.h and the ggml.c type traits, and the same two ids in gguf-py; '
          'a row is K/2 bytes of 4-bit codes followed by K/16 bytes of table indices, and two FP16 tensors ride beside each '
          'expert weight (row multipliers [rows, n_expert], affine table [2, 256, n_expert])',
  why='the device view the kernel.ciru_iu4.c1 and mmb kernels read; the files are the whole ggml.h / ggml.c / constants.py, of which CIRU changed 16, 32 and 4 lines',
  paths=['ggml/include/ggml.h', 'ggml/src/ggml.c', 'gguf-py/gguf/constants.py'],
  provides=[P('format.device-view/c1-g16/-', 'GGML_TYPE_C1_G16_K2560 = 43, GGML_TYPE_C1_G16_K640 = 44 (ggml/include/ggml.h); type traits blck_size / type_size (ggml/src/ggml.c); GGMLQuantizationType.C1_G16_* (gguf-py/gguf/constants.py)',
              'codes are unsigned 4-bit (code - 8 as int8 on the device); value = table[index] affine of the code, times the row multiplier (src/models/qwen4exp.cpp loads the aux tensors, not harvested)',
              'Qwen3.8-Flash-Next routed experts: gate / up K2560, down K640'),
            P('graph.op-lightning-indexer-compact/-/-', 'ggml_lightning_indexer_compact(ctx, q, k, weights, mask, limits, full_blocks, first) (ggml.h / ggml.c)',
              'builds GGML_OP_LIGHTNING_INDEXER with src[4] = I32 limits; kernel.ciru_iu4.attn computes it', 'QSA indexer over compact per-row limits')],
  notes='Also in these files and changed by CIRU: the ggml_lightning_indexer_compact graph op (ggml.h, ggml.c). The loader that creates the C1 '
        'aux tensors and checks their types is in src/models/qwen4exp.cpp (load_one, lines 380-410), which is model code and not harvested. '
        'The CPU backend refuses C1 (ggml-cpu.cpp), not harvested. Interface tag: none (no catalogue format reads C1).',
  lang='c', sources=[], includes=['src/ggml/include'], compile='n/a', toolchain='',
  reason='type definitions inside ggml\'s core C library; compiled with all of ggml by the origin CMake, not standalone',
 ),
 'ple_pager': dict(
  summary='CIRU\'s SSD-paged PLE embedding table (CIRUPLE1): the Qwen3.8-Flash-Next per-layer-embedding rows stay in a payload '
          'file and are paged into a bounded host cache by the model as tokens arrive, instead of being held in memory',
  why='the PLE table is the 52 GB part of the release (docs/PROVENANCE.md); this pager is how CIRU serves it on a 128 GB machine, beside '
      'strixite\'s and crucible\'s PLE row readers',
  paths=['src/models/qwen4exp-ple-pager.cpp', 'src/models/qwen4exp-ple-pager.h'],
  provides=[P('ple.rows-paged/ciruple1/-', 'qwen4exp-ple-pager.h (the pager class and its sidecar file layout)',
              'rows copied verbatim from the payload file', 'host side, one model')],
  notes='Interface tag: none (ple.ngram-rows is the nearest catalogue class, but CIRUPLE1 pages plain rows, not n-gram hashes). Includes llama '
        'internals (llama-model.h and friends) that are not harvested.',
  lang='c++', sources=[], includes=[], compile='n/a', toolchain='',
  reason='host code of libllama: includes llama internals that are not harvested; builds only inside the origin tree',
 ),
}


def resolve(origin, frm, name, quoted):
    cands = ([os.path.normpath(os.path.join(os.path.dirname(frm), name))] if quoted else []) + \
        [os.path.normpath(os.path.join(r, name)) for r in INC_ROOTS]
    have = set(origin.files())
    return next((c for c in cands if c in have), None)


def closure(origin, roots):
    """origin files reached from roots through quoted and angle includes resolved in the ggml-cuda target's roots."""
    seen, todo = set(), list(roots)
    rx = re.compile(r'^[ \t]*#[ \t]*include[ \t]*(["<])([^">]+)[">]', re.M)
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        for q, name in rx.findall(open(origin.find(f), errors='replace').read()):
            hit = resolve(origin, f, name, q == '"')
            if hit and hit not in seen:
                todo.append(hit)
    return seen


def main():
    O = TreeOrigin('ciru_iu4', 'ciru_iu4', 'https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4', COMMIT, 'main', SPDX, AUTHORS, 'therock-10.1')
    diff = read_base_diff()
    have = O.blobs
    bad = [p for p, (st, cb) in diff.items() if st != 'D' and have.get(p) != cb]
    if bad:
        raise SystemExit('base-diff.tsv does not match the checkout: ' + ' '.join(bad[:10]))
    ciru = {p for p, (st, _) in diff.items() if st != 'D'}
    kernel_rows = sorted(p for p in ciru if p.startswith((CU, 'ggml/src/ggml-hip/')) and '/template-instances/' not in p)
    declared = sorted([CU + r['file'] for fam in KINDS_ORDER for r in FAMILIES[fam]['files']] +
                      [e for fam in KINDS_ORDER for e in FAMILIES[fam].get('extra', [])])
    if kernel_rows != sorted(declared + [CU + 'common.cuh']):     # common.cuh is in kernel.ciru_iu4.cuda_common
        raise SystemExit('ggml-cuda / ggml-hip files CIRU adds or changes differ from FAMILIES: ' + ' '.join(sorted(set(kernel_rows) ^ set(declared + [CU + 'common.cuh']))))
    for fm in FORMATS.values():
        for p in fm['paths']:
            if p not in ciru:
                raise SystemExit(f'{p}: in a format module but not added or changed by CIRU')

    mods = []
    # base headers the CIRU units include, with common.cuh (CIRU-changed), so every family compiles standalone
    roots = [p for p in declared if p.endswith(('.cu', '.cuh', '.h', '.inc'))]
    reach = closure(O, roots)
    base_hdrs = sorted(p for p in reach if p not in ciru)
    common = [CU + 'common.cuh'] + base_hdrs
    if CU + 'common.cuh' not in ciru:
        raise SystemExit('common.cuh is no longer changed by CIRU: revisit kernel.ciru_iu4.cuda_common')
    stray = sorted(p for p in reach if p in ciru and p not in declared and p != CU + 'common.cuh' and
                   all(p not in fm['paths'] for fm in FORMATS.values()))
    if stray:
        raise SystemExit('CIRU-changed files reached by includes but in no module: ' + ' '.join(stray))
    mods.append(M('kernel.ciru_iu4.cuda_common',
        'Header-only layer the CIRU kernel units include: ggml-cuda\'s shared device header common.cuh (CIRU changes two lines) and '
        'the %d unchanged base headers of ggml-cuda, ggml/src and ggml/include that the CIRU files reach through #include '
        '(vendors/hip.h, the quantization block and conversion helpers, MMA / WMMA wrappers, the ggml backend headers)' % len(base_hdrs),
        common, [P('kernel-abi.ciru-ggml-cuda/-/-', 'ggml-cuda/common.cuh: ggml_backend_cuda_context, the CUDA-to-HIP vendor mapping, warp and block helpers',
                   'n/a here', 'compiled inside every kernel.ciru_iu4.* unit')],
        why='SPEC.md gives headers that several modules include a module of their own; these are base files (pwilkin/llama.cpp f5daaa3c) except common.cuh, '
            'taken only so that the CIRU families compile standalone; the rest of the base is not harvested',
        notes='Interface tag: none. Files: common.cuh (CIRU, +2 -2 lines against the base) and %d base files listed by the include closure that '
              'docs/harvest/harvest_ciru_iu4.py computes from the CIRU kernel files.' % len(base_hdrs),
        includes=INC, sources=[], compile='header-only', reason='header-only; compiled inside every other kernel.ciru_iu4.* unit',
        requires=['module:format.ciru_iu4.c1_g16']))
    for fam in KINDS_ORDER:
        d = FAMILIES[fam]
        files = [CU + r['file'] for r in d['files']] + d.get('extra', [])
        provides, seen = [], set()
        for r in d['files']:
            if r['tag'] and r['tag'] not in seen:
                seen.add(r['tag'])
                provides.append(P(r['tag'], '%s: %s' % (r['file'], ', '.join(r['kernels']) or 'host path'), d['numeric'], d['regime']))
        if d.get('own'):
            provides.append(P(d['own'][0], d['own'][1], d['numeric'], d['regime']))
        if not provides:
            provides.append(P('kernel-abi.ciru-%s/-/-' % fam.replace('_', '-'), ', '.join(r['file'] for r in d['files']),
                              d['numeric'], d['regime']))
        tags = '; '.join('%s %s' % (r['file'], tag_class(r['tag']) if r['tag'] else 'none') for r in d['files'])
        changed = '; '.join('%s %s' % (r['file'], r['changed']) for r in d['files'] if r['changed'])
        added = ', '.join(r['file'] for r in d['files'] if not r['changed'])
        notes = 'Interface tags by file: %s (catalog/ciru_iu4/KERNELS.md has the table). Against the base pwilkin/llama.cpp f5daaa3c: %s%s. %s' % (
            tags, ('added ' + added + '; ') if added else '', ('changed ' + changed) if changed else 'nothing changed', NOT_TIMED)
        if fam == 'backend':
            notes += ' ggml-hip/CMakeLists.txt is the origin build file (+8 lines: -mwavefrontsize64 and -fno-fast-math for the two Q5_K units on gfx1151); not compiled.'
        mods.append(M('kernel.ciru_iu4.' + fam, d['summary'], files, provides, why=d['why'], notes=notes,
                      sources=[CU + s for s in d['sources']], includes=INC,
                      source_flags={CU + k: v for k, v in d.get('source_flags', {}).items()} or None,
                      requires=['module:kernel.ciru_iu4.cuda_common'],
                      upstream=UPSTREAM.get(fam)))
    # module requires from #include lines of .cu / .cuh files too (harvestlib.include_requires reads the C family only)
    owner = {}
    for m in mods:
        for f in O.expand(m.paths, m.exclude):
            owner[f] = m.id
    rx = re.compile(r'^[ \t]*#[ \t]*include[ \t]*(["<])([^">]+)[">]', re.M)
    for m in mods:
        need = []
        for f in O.expand(m.paths, m.exclude):
            for q, name in rx.findall(open(O.find(f), errors='replace').read()):
                hit = resolve(O, f, name, q == '"')
                dep = owner.get(hit)
                if dep and dep != m.id and 'module:' + dep not in m.requires + need:
                    need.append('module:' + dep)
        m.requires = list(m.requires) + sorted(need)
    for name in ('c1_g16', 'ple_pager'):
        d = FORMATS[name]
        mods.append(M('format.ciru_iu4.' + name, d['summary'], d['paths'], d['provides'], why=d['why'], notes=d['notes'],
                      lang=d['lang'], sources=d['sources'], includes=d['includes'], compile=d['compile'], toolchain=d['toolchain'],
                      reason=d['reason']))
    run(O, mods, ORIGIN_TOML, LEFT_OUT, ['LICENSE', 'THIRD_PARTY_NOTICES.md'])
    kernels_md(O)


UPSTREAM = {
 'q5': 'q5_short.cu adapts Gufo\'s SmallBatchKQuantExactFp32GEMMKernel and Q5_K decoder (MIT notice in the file); donor_wkquant_kernel.cuh is an unmodified excerpt of Gufo 990fdce45640 src/models/qwen/hip/kernels/prefill_quant_gemm.hpp (MIT)',
 'mmb': 'the Q5_K tile loader of mmb_f32_halo_kernel follows HaloBox (halo-box/strix-llama.cpp) 25fda3478 (MIT); the MMB path itself is pwilkin\'s',
 'gdn': 'the chunked GDN gate availability follows HaloBox c750386bb (THIRD_PARTY_NOTICES.md)',
 'attn': 'QSA, the lightning indexer and the MMB / FA kernels these files change are pwilkin\'s Strix Halo reference work on llama.cpp',
}


def tag_class(tag):
    head, wf, dt = tag.split('/')
    return '/'.join([head, wf, dt.split('.')[0]])


ORIGIN_TOML = '''
# Qwen3.8 Flash CIRU Strix IU4: Ciru's llama.cpp-derived runtime for Qwen3.8-Flash-Next on AMD Strix Halo (gfx1151): a compact
# IU4 routed-expert weight type (C1) on the iu4 WMMA, MTP drafting, sparse attention (QSA) decode paths, chunked Gated DeltaNet
# prefill and an SSD-paged PLE table, on top of pwilkin's Strix Halo reference engine. Only what CIRU adds or changes over
# that base is harvested (card ORIGIN-CIRU-IU4 314b9e37c3).
name      = "ciru_iu4"
repo      = "https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4"
commit    = "47cb9239fa4913aa5a7c4b36dc21ca63e69da184"
branch    = "main"
version   = "runtime 6.0.0 (SOURCE_IDENTITY.json, CIRU_RELEASE.json); head commit 'Release Flash v6.1.0 hybrid', whose v6.1 runtime is a patch on HaloBox b1064 in contrib/flash-v6.1 (not harvested)"
base      = "pwilkin/llama.cpp f5daaa3cfa6358e5dd398911ec741813745a5440 (reference_commit of docs/qualification/v4.0.0/source-identity.json, base_revision of docs/qualification/v3.1.0/BUILD-SPEC.json); the delta is docs/harvest/data/ciru_iu4/base-diff.tsv, computed on control-node from the ark copies of both (base: ark a3db8d4). docs/PROVENANCE.md names ggml-org/llama.cpp f5e85d43 as the clean public base of the v1 / v2 lineage"
reference_checkout = "control-node:/data/ark-inbox/src/github.com__ciru-ai__Qwen3.8-Flash-CIRU-STRIX-IU4#47cb9239fa4913aa5a7c4b36dc21ca63e69da184 (ark copy, ark 7546d7e, a plain tree without .git: the harvest checks all 4,001 files against docs/harvest/data/ciru_iu4/tree.tsv, the git blob ids of 47cb923); the harvest runs on control-node against it"

[licence]
spdx      = "MIT"
files     = ["LICENSE", "THIRD_PARTY_NOTICES.md"]                  # copied verbatim next to this file
third_party = "THIRD_PARTY_NOTICES.md: a derivative of ggml-org/llama.cpp under its MIT licence; Qwen experimental / MTP support adapted from Ryan Monsurate's llama.cpp work (MIT); Gufo Q5 kernels (q5_short.cu with its own MIT notice, donor_wkquant_kernel.cuh an unmodified Gufo 990fdce excerpt, MIT); HaloBox and pwilkin Strix Halo work (MIT); AMD OPUS helper headers and hip_minimal.hpp (MIT, not in the harvested files). Model weights, the MTP draft and the PLE payload are under the Qwen Community License 1.0 and are not part of the tree"

[toolchain]
host      = "C++17, CMake + Ninja (scripts/ciru/build-linux-amd.sh: GGML_HIP=ON, GGML_HIP_NO_VMM=ON, GGML_HIP_GRAPHS=ON, GGML_HIP_MMQ_MFMA=ON, BUILD_SHARED_LIBS=ON, Release)"
device    = "therock-10.1"                               # CIRU_RELEASE.json: ROCm SDK 10.0.0 with pwilkin's HIP / ROCr 7dda3ac6; contrib/flash-v6.1 builds with TheRock 10.1.0a20260910
build     = "ggml/src/ggml-hip/CMakeLists.txt compiles every ggml-cuda/*.cu as HIP into libggml-hip; q5_short.cu and q5_wave64.cu get -mwavefrontsize64 (and -O2 -g -fno-fast-math -DCIRU_Q5_SHORT_GFX1151=1) on gfx1151"

[compile]                                                # CMake's HIP language: clang++ -x hip, one object per .cu
driver    = "<rocm>/lib/llvm/bin/clang++"
arch_flag = "--offload-arch=gfx1151"
rocm_flag = "--rocm-path=<rocm>"
flags     = FLAGS
system_includes = ["include"]
output    = "host object with the gfx1151 device code embedded (.o), one per translation unit"
note      = "flags are the ggml-hip target's compile line from compile_commands.json of a CMake 4.4.3 configure of the ark copy on control-node (therock-10.1) with scripts/ciru/build-linux-amd.sh's options (GGML_HIP_NO_VMM, GGML_HIP_GRAPHS, GGML_HIP_MMQ_MFMA, shared libraries, Release; /data/pod/plans/314b9e37c3/compile-commands-excerpt.txt), in the same set; the module's own include roots replace -I ggml/src and -I ggml/include; the gate leaves -g out of q5_short.cu's per-source options"

[kernels]
loading   = "linked into libggml-hip: kernels are launched by host functions in the same .cu file, which ggml-cuda.cu's op dispatch calls for each graph node; no code-object kit or kernel registry"
dispatch  = "ggml-cuda.cu picks a path by op, tensor type and shape, with CIRU's environment switches (CIRU_*, LLAMA_C1_*, LLAMA_QSA_PER_QUERY, GGML_GDN_CHUNK, ...) set by the launch profiles in profiles/ and scripts/ciru/run-server.sh"
models    = "Qwen3.8-Flash-Next (IU4 target GGUF, Q8_0 MTP draft, CIRUPLE1 PLE payload); author figures for v6.1.0 hybrid: 4K uncached prefill 1,363.63 -> 1,626.27 tok/s, decode 42.2 -> 42.3 tok/s (contrib/flash-v6.1/qualification.json)"
'''

FLAGS = ["-x", "hip", "-std=gnu++17", "-O3", "-DNDEBUG", "-fPIC", "-DGGML_USE_HIP", "-DGGML_HIP_NO_VMM", "-DGGML_HIP_GRAPHS",
         "-DGGML_SHARED", "-DGGML_BACKEND_BUILD", "-DGGML_BACKEND_SHARED", "-DGGML_SCHED_MAX_COPIES=4",
         "-D_GNU_SOURCE", "-D_XOPEN_SOURCE=600", "-DUSE_PROF_API=1", "-D__HIP_PLATFORM_AMD__=1", "-D__HIP_ROCclr__=1", "-Dggml_hip_EXPORTS"]
# ggml/cmake/common.cmake ggml_cuda_fattn_vec_instances with the default GGML_CUDA_FA_QUANTS (ggml/CMakeLists.txt:207): one
# GGML_CUDA_FA_<K>_<V> definition a K / V type pair, 1 for the four compiled pairs, and the list itself
FA_TYPES = ['q4_0', 'q4_1', 'q5_0', 'q5_1', 'q8_0', 'bf16', 'f16']
FA_PAIRS = ['q4_0-q4_0', 'q8_0-q8_0', 'f16-f16', 'bf16-bf16']
FLAGS += ['-DGGML_CUDA_FA_QUANTS="%s"' % ','.join(FA_PAIRS)] + \
    ['-DGGML_CUDA_FA_%s_%s=%d' % (k.upper(), v.upper(), int('%s-%s' % (k, v) in FA_PAIRS)) for v in FA_TYPES for k in FA_TYPES] + ['-c']

LEFT_OUT = {
    'ggml/ (everything but the files in the modules)': 'the base engine, pwilkin/llama.cpp f5daaa3c, unchanged by CIRU; kernel.ciru_iu4.cuda_common carries the base headers the CIRU kernels include',
    'ggml/src/ggml-cpu/ggml-cpu.cpp, ggml/src/ggml-cpu/ops.cpp, ggml/src/ggml-backend.cpp, ggml/src/ggml-openvino/': 'CIRU changes outside the HIP backend: the CPU backend refuses C1 and computes the compact lightning indexer, a backend scheduling change, two OpenVINO fixes',
    'src/ (llama-batch, llama-context, llama-graph, llama-kv-cache, llama-kv-cells, llama-memory-*, llama-model, models/qwen4exp.cpp, models/qwen35moe.cpp, models/block-graph.inc, ciru-mtp-shortlist.h, qsa-prefix-state.h)': 'CIRU\'s runtime changes in libllama: MTP drafting and rollback, the hybrid memory and KV cells, the Qwen3.8-Flash-Next graph and the C1 aux-tensor loader; runtime-level, not kernels (a runtime harvest is its own card)',
    'common/, tools/server/, tools/llama-bench/, include/llama.h': 'speculative-decoding policy, server and bench changes',
    'contrib/flash-v6.1/': 'the v6.1.0 runtime as a patch on HaloBox b1064 (halo-box/strix-llama.cpp b9f3c0e8) with its build and weight recipes; the patched tree, its C1 / GDN / hyper-connection kernel changes and the two GDN sidecars (scripts/ciru/flash-v6/*.hip) exist only after applying it, and b9f3c0e8 is not in the ark',
    'tests/ (CIRU tests: test-fattn-mask-bounds.cu, test-qsa-*, test-nextn-row-order.cpp, test-backend-uma-input.cpp, test-c1-small-batch-dispatch.py, test-ciru-launcher.py)': 'tests; not run here',
    'scripts/ciru/, profiles/, patches/, checksums/, results/, benchmarks/, docs/, ui/, *.md, *.json at the root': 'build and launch scripts, launch profiles, the H121 patch already applied in src, qualification records, benchmarks and documentation',
    'LICENSE, THIRD_PARTY_NOTICES.md': 'copied beside ORIGIN.toml',
}


def providers(origin):
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


def kernels_md(O):
    """catalog/ciru_iu4/KERNELS.md: every ggml-cuda file CIRU adds or changes, with its interface tag."""
    prov = providers(O.name)
    diff = read_base_diff()
    o = ['# CIRU IU4 kernels by strixflerp interface', '',
         'One row for each file of `ggml/src/ggml-cuda` that ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4 at 47cb923 adds (A) or changes (M)',
         'against its base pwilkin/llama.cpp f5daaa3c (`docs/harvest/data/ciru_iu4/base-diff.tsv`). Card ORIGIN-CIRU-IU4 314b9e37c3.',
         'Written by `docs/harvest/harvest_ciru_iu4.py` from its `FAMILIES` table; read from source, nothing here was run.', '',
         '- **tag**: the strixflerp interface class the file matches, `<family>.<op>/<weight format>/<dtype>`, or none. The module\'s',
         '  `[[provides]]` entry carries the same name with a `.ciru-*` variant tag.',
         '- **same op in the catalogue**: kernel modules of other origins that provide that `<family>.<op>`. The same op is not the',
         '  same call: every CIRU kernel is reached through ggml-cuda.cu\'s dispatch of a ggml graph node, so a row needs an adapter',
         '  before it stands in for a strixflerp module.', '',
         '| file | A/M | lines | module | new device kernels | tag | same op in the catalogue |', '|---|---|---:|---|---|---|---|']
    n, tagged = 0, 0
    for fam in KINDS_ORDER:
        for r in FAMILIES[fam]['files']:
            rel = CU + r['file']
            lines = len(open(O.find(rel), errors='replace').read().splitlines())
            beside = []
            if r['tag']:
                op = r['tag'].split('/')[0] + '/'
                beside = sorted(m for m, ifs in prov.items() if m.startswith('kernel.') and any(i.startswith(op) for i in ifs))
                tagged += 1
            n += 1
            o.append('| `%s` | %s | %d | `kernel.ciru_iu4.%s` | %s | %s | %s |' % (
                r['file'], diff[rel][0], lines, fam, ', '.join('`%s`' % k for k in r['kernels']) or '-',
                '`%s`' % tag_class(r['tag']) if r['tag'] else 'none', ', '.join('`%s`' % b for b in beside) or '-'))
    added = sum(1 for fam in KINDS_ORDER for r in FAMILIES[fam]['files'] if diff[CU + r['file']][0] == 'A')
    o += ['', '%d files: %d added, %d changed; %d carry a tag, %d are none (headers, includes and host glue). '
          '`ggml/src/ggml-hip/CMakeLists.txt` (changed, in `kernel.ciru_iu4.backend`) is the build file.' % (n, added, n - added, tagged, n - tagged), '',
          '## What each file does', '']
    for fam in KINDS_ORDER:
        for r in FAMILIES[fam]['files']:
            o.append('- `%s` (%s): %s%s.' % (r['file'], fam, r['what'], ('; ' + r['changed']) if r['changed'] else ''))
    o += ['', '## Not done here', '',
          'No kernel was run or timed. The compile gate (`gate/compile/run.py` on control-node, CPU only) compiles each family\'s',
          'translation units for gfx1151 with the ggml-hip target\'s flags. The v6.1.0 kernel changes are in a patch on HaloBox b1064',
          '(`contrib/flash-v6.1/source/halobox-b1064-flash-v6.1.patch`) and are not in this table.']
    open(os.path.join(harvestlib.REPO, 'catalog', O.name, 'KERNELS.md'), 'w').write('\n'.join(o) + '\n')


ORIGIN_TOML = ORIGIN_TOML.replace('flags     = FLAGS', 'flags     = ' + harvestlib.tlist(FLAGS))

if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--base-diff':
        write_base_diff(sys.argv[2])
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit(__doc__)
