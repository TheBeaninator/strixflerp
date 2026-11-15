#!/usr/bin/env python3
"""Harvest hipfire v0.4.5 (efa8fbb) into catalog/hipfire. Run: python3 docs/harvest/harvest_hipfire.py
Kernel sources are grouped into one module per family (hipfire_units.FAMILIES); each module's units.json lists the
compile units hipfire itself builds from those sources (crates/rdna-compute/src/kernels.rs and friends), restricted to
gfx1151 material. Rust crates are taken whole (src/, Cargo.toml, map.md): a crate cannot be split and still build."""
import collections, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harvestlib import Origin, M, P, run, sha256
from hipfire_units import parse_units, family, arch_ok, FOREIGN
from incgraph import closure

O = Origin('hipfire', 'hipfire', 'https://github.com/warpfront/hipfire', 'efa8fbb', 'master', 'Apache-2.0 OR MIT',
           ['Kaden Schutt', 'hipfire contributors'], 'therock-10.2')
ROOT = O.roots[0]
KS = 'kernels/src/'
NOTBUILT = 'rust crate: built only inside a recipe (phase 3 and later); cargo needs the whole workspace and vendored dependencies'
FLAG_TAG = re.compile(r'^\s*//\s*HIPFIRE_COMPILER_FLAGS:(.*)$', re.M)
# Sources that are in the tree for gfx1151-or-generic by name but do not compile for gfx1151; filled from the compile
# gate's own errors (gate/results/compile), each with the compiler's reason. They stay in src/ but leave units.json.
NOT_GFX1151 = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'hipfire', 'not_gfx1151.json'))) \
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'hipfire', 'not_gfx1151.json')) else {}

hips = [f for f in sorted(os.listdir(os.path.join(ROOT, KS))) if f.endswith('.hip')]
kept = [f for f in hips if arch_ok(f)]
fam = collections.defaultdict(list)
for f in kept:
    fam[family(f)].append(KS + f)
left_names = fam.pop('LEFTOUT', [])
foreign = [f for f in hips if not arch_ok(f)]

# gfx12_weight_cache_policy.inc is a prelude spliced into generic and gfx1151 units too, whatever its name says
hdr = [KS + f for f in sorted(os.listdir(os.path.join(ROOT, KS))) if f.endswith('.inc') or (f.endswith(('.h', '.cuh')) and arch_ok(f))]
ggml = sorted({f for fs in closure(ROOT, ['third_party/ggml/src', 'third_party/ggml/include', 'third_party/ggml/src/ggml-cuda'],
                                   ['third_party/ggml/src/ggml-cuda/vecdotq.cuh']).values() for f in fs})
hdr += ggml + ['third_party/ggml/LICENSE', 'third_party/ggml/PIN']
owner = {f: 'common_headers' for f in hdr}
for k, fs in fam.items():
    for f in fs:
        owner[f] = k

# ---- compile units
units_raw = parse_units(ROOT)
mod_units = collections.defaultdict(list)
skipped = collections.Counter()
in_unit = set()
for cname, u in sorted(units_raw.items()):
    files = [p['file'] for p in u['parts'] if 'file' in p]
    if FOREIGN.search(cname.lower()):
        skipped['foreign arch in the constant name'] += 1
        continue
    if any(f not in owner for f in files):
        skipped['needs a source outside the gfx1151 set (other arch, image / audio family)'] += 1
        continue
    main = [f for f in files if f.endswith('.hip')][-1]
    name = re.sub(r'_src$', '', cname.lower())
    text = ''.join(p.get('text', '') for p in u['parts'])
    parts = [{'path': f'kernel/{owner[p["file"]]}/src/{p["file"]}'} if 'file' in p else {'text': p['text']} for p in u['parts']]
    flags = []
    for f in files:
        for m in FLAG_TAG.finditer(open(os.path.join(ROOT, f), errors='replace').read()):
            flags += m.group(1).split()
    for m in FLAG_TAG.finditer(text):
        flags += m.group(1).split()
    if name in NOT_GFX1151:
        skipped['does not compile for gfx1151 (docs/harvest/data/hipfire/not_gfx1151.json)'] += 1
        in_unit.update(files)
        continue
    mod_units[owner[main]].append({'name': name, 'const': cname, 'registry': u['registry'], 'parts': parts, 'flags': flags})
    in_unit.update(files)
for k, fs in fam.items():
    for f in fs:
        if f not in in_unit:
            name = 'file_' + re.sub(r'[^a-z0-9]+', '_', os.path.basename(f)[:-4].lower())
            if name in NOT_GFX1151:
                skipped['does not compile for gfx1151 (docs/harvest/data/hipfire/not_gfx1151.json)'] += 1
                continue
            flags = [t for m in FLAG_TAG.finditer(open(os.path.join(ROOT, f), errors='replace').read()) for t in m.group(1).split()]
            mod_units[k].append({'name': name, 'const': None, 'registry': None, 'parts': [{'path': f'kernel/{k}/src/{f}'}], 'flags': flags})

# sources whose only units are excluded as not-gfx1151 ('arch') leave the harvest; the others stay in src/ without a unit
used_by = collections.defaultdict(set)
for cname, u in units_raw.items():
    for p_ in u['parts']:
        if 'file' in p_:
            used_by[p_['file']].add(re.sub(r'_src$', '', cname.lower()))
dropped_arch, uncovered = [], collections.Counter()
live_units = {u['name'] for us in mod_units.values() for u in us}
for k in list(fam):
    keep_f = []
    for f in fam[k]:
        names = {n for n in used_by.get(f, set())} | {'file_' + re.sub(r'[^a-z0-9]+', '_', os.path.basename(f)[:-4].lower())}
        excl = [NOT_GFX1151[n] for n in names if n in NOT_GFX1151]
        if not (names & live_units) and excl and all(e['class'] == 'arch' for e in excl):
            dropped_arch.append(f)
            continue
        if not (names & live_units):
            uncovered[k] += 1
        keep_f.append(f)
    fam[k] = keep_f
DEC = 'decode, 1 row (multirow / batched variants up to 8..16 rows), memory-bound'
PF = 'prefill batches, compute-bound (WMMA, MMQ)'
F32 = 'fp32 activations and fp32 accumulate; weights decoded in-kernel; summation order is per kernel variant, so bits differ between variants and from other engines'


def nvar(k):
    return len(mod_units[k])


def ex(k, n=5):
    us = [u['name'] for u in mod_units[k]]
    return ', '.join(us[:n]) + (f' (+{len(us) - n} more units, see units.json)' if len(us) > n else '')


def kmod(k, summary, provides, requires=(), notes=None):
    deps = sorted({p['path'].split('/')[1] for u in mod_units[k] for p in u['parts'] if 'path' in p} - {k})
    return M(f'kernel.hipfire.{k}', summary + f' ({len(fam[k])} sources, {nvar(k)} gfx1151 compile units)', fam[k],
             display_paths=[KS + '<' + k + ' family: see HARVEST.json and units.json>'],
             provides=provides, includes=['src/kernels/src'], units='units.json', variants=nvar(k),
             requires=['module:kernel.hipfire.common_headers'] + [f'module:kernel.hipfire.{d}' for d in deps] + list(requires),
             extra_files={'units.json': json.dumps({'origin_registry': 'crates/rdna-compute/src/kernels.rs (+ other rdna-compute/src/*.rs constants)',
                                                    'note': 'a unit is the concatenation of its parts, compiled as one HIP source, as hipfire builds it; path parts are relative to catalog/hipfire',
                                                    'units': mod_units[k]}, indent=1)},
             notes=((notes + ' ') if notes else '') + (f'{uncovered[k]} of the sources have no compile unit here (assembled by Rust code in the origin, or broken in the origin): see docs/harvest/data/hipfire/not_gfx1151.json.' if uncovered[k] else '') or None,
             parity='origin-test:crates/rdna-compute/tests')


mods = [
    M('kernel.hipfire.common_headers', 'Header-only: shared kernel headers (turbo_common, Givens / FWHT rotation, KV slot descriptors, MMQ bodies, device wait, TP exchange) and the vendored ggml-cuda vecdotq headers',
      hdr, lang='hip-c++', includes=['src/kernels/src', 'src/third_party/ggml/src', 'src/third_party/ggml/src/ggml-cuda', 'src/third_party/ggml/include'], compile='header-only',
      provides=[P('kernel-abi.hipfire-common/-/-', 'turbo_common.h, givens_common.h, mq_fwht256.h, kv_slot_desc*.h, tp_exchange16.h, *_mmq_body.cuh', 'n/a', 'n/a')],
      upstream='third_party/ggml: see third_party/ggml/PIN (MIT, The ggml authors)', reason='header-only; included by the kernel.hipfire.* family modules'),
    kmod('gemv_hfq', 'Decode GEMV on hipfire HFQ weights: flat N-bit codes with one f32 scale / zero per group (HFQ2..HFQ8, groups 64..1024), HFP4-G32 (FP4 E2M1 + UE8M0), PARO; residual, multirow, wave64 and fused QKV / gate-up forms',
         [P('linear.decode/hipfire-hfq/f32', ex('gemv_hfq'), F32, DEC)], requires=['format.device-view/hipfire-hfq/-']),
    kmod('gemv_mq', 'Decode GEMV on hipfire MQ weights: FWHT-rotated HFQ (MQ2..MQ8 G256, the dual-fp16-scale V2 family, Lloyd / LUT codebooks, MQ4C); residual, multirow and fused forms',
         [P('linear.decode/hipfire-mq/f32', ex('gemv_mq'), F32 + '; the input must be rotated first (rotate.fwht), a fused rotate exists for some epilogues', DEC)],
         requires=['format.device-view/hipfire-mq/-', 'rotate.fwht/-/f32']),
    kmod('gemv_mfp4', 'Decode GEMV on MFP4-G32 / MFP3 weights (FP4 codes with per-32 scales, E8 lattice variants), including the indexed MoE gate-up and down forms for gfx1151',
         [P('linear.decode/hipfire-mfp4/f32', ex('gemv_mfp4'), F32, DEC)], requires=['format.device-view/hipfire-mfp4/-']),
    kmod('gemv_ggml', 'Decode GEMV on GGUF blocks (Q4_K, Q6_K, Q8_0 and its wide / staged variants, Q4F16, TQ2, BQ1) and the fused QKV / gate-up forms on Q4_K and Q8_0',
         [P('linear.decode/gguf-kquant/f32.hipfire', ex('gemv_ggml'), F32 + '; Q8_0 block = f16 scale + 32 int8, Q4_K block = 144 bytes per 256 weights', DEC)],
         requires=['format.device-view/gguf-kquant/-.hipfire']),
    kmod('gemv_float', 'Decode GEMV on float weights: F32, F16 and BF16 with fp32 x, plus the fused Q/K L2-norm scale',
         [P('linear.decode/f32+f16+bf16/f32.hipfire', ex('gemv_float'), 'fp32 accumulate, shared-memory reduction across wavefronts', DEC),
          P('linear.dense/bf16/f32.hipfire', 'gemv_bf16_xf32 (W [M, K] raw bf16, x [K] f32, y [M] f32)', 'bf16 widened losslessly (16-bit shift), fp32 input and fp32 accumulate; the origin calls it its KLD / PPL oracle path', '1 row')]),
    kmod('gemm_hfq', 'Prefill GEMM on HFQ weights: WMMA, dot2 and MMQ bodies; residual, QKV, QKVZA and gate-up multi-output forms',
         [P('linear.prefill/hipfire-hfq/f32', ex('gemm_hfq'), 'weights decoded to f16 tiles for WMMA (fp32 accumulate) or integer MMQ with activation quantization, per variant', PF)],
         requires=['format.device-view/hipfire-hfq/-']),
    kmod('gemm_mq', 'Prefill GEMM on MQ weights: f16 WMMA, the exact integer routes (IU4 and I8 WMMA with a block_i8_128 activation quantizer), MMQ; residual, QKV, QKVZA and gate-up forms',
         [P('linear.prefill/hipfire-mq/f32', ex('gemm_mq'), 'f16 WMMA with fp32 accumulate, or exact-integer (iu4 / i8) WMMA with activations quantized per 128-block and one fp scale per block pair', PF),
          P('act.quantize/i8-128/f32', 'block_i8_128_quant, act_quant_* units', 'per-128-block int8 activation quantizer feeding the integer GEMMs', PF)],
         requires=['format.device-view/hipfire-mq/-', 'rotate.fwht/-/f32']),
    kmod('gemm_mfp4', 'Prefill GEMM on MFP4-G32 weights (f16 WMMA, E8 variants) and the MFP4 dequantizer',
         [P('linear.prefill/hipfire-mfp4/f32', ex('gemm_mfp4'), 'f16 WMMA, fp32 accumulate', PF)], requires=['format.device-view/hipfire-mfp4/-']),
    kmod('gemm_ggml', 'Prefill GEMM on GGUF Q8_0 (WMMA) and the TQ2 / BQ1 low-bit formats',
         [P('linear.prefill/gguf-q8_0/f32.hipfire', ex('gemm_ggml'), 'Q8_0 blocks on WMMA, fp32 accumulate', PF)], requires=['format.device-view/gguf-kquant/-.hipfire']),
    kmod('gemm_float', 'Prefill GEMM on float weights: F16 x F16 WMMA, F32, BF16',
         [P('linear.prefill/f16+bf16+f32/f32.hipfire', ex('gemm_float'), 'f16 or bf16 WMMA operands, fp32 accumulate', PF)]),
    kmod('attention', 'Softmax attention: single-head causal, GQA, Flash-Decoding (per-chunk partials of max, sum, output with an online-softmax reduce) over F32 / F16 / BF16 / Q8_0 / int8 / HFQ / asymmetric FWHT and Givens rotated KV, prefill on WMMA, sliding window',
         [P('attention.full/kv-f16+bf16+q8_0+rotated/f32.hipfire', ex('attention'), 'fp32 scores and online softmax; KV rows decoded from their storage format in-kernel; split chunks merged by an online-softmax correction (order fixed by chunk index)', DEC + '; ' + PF)],
         requires=['kv.store/hipfire-kv/f32']),
    kmod('kv_cache', 'KV cache writers and transcoders: F32 / F16 / BF16, Q8_0, int8, HFQ4 / HFQ8, asymmetric K with FWHT or Givens rotation at 2 / 3 / 4 bits, Lloyd-coded V, fold and compaction, sliding-window ring',
         [P('kv.store/hipfire-kv/f32', ex('kv_cache'), 'quantization per format: Q8_0 = f16 scale + 32 int8 per 32 elements; fp16 by true RNE; rotated low-bit K after a FWHT / Givens rotation; transcodes down a bit ladder as context grows', DEC),
]),
    kmod('ds4_sparse_attn', 'DeepSeek V4 sparse attention: sliding-window + top-k attention (scalar and WMMA, direct and gathered), the KV compressor (softmax pooling, ring write, overlap concat), indexed attention, tri-attention scoring',
         [P('attention.sparse-sink/kv-f16/f32.hipfire', ex('ds4_sparse_attn'), 'fp32 math over the SWA window plus the top-k compressed keys', DEC + '; ' + PF),
          P('kv.compress/-/f32.hipfire', 'compressor_* units', 'softmax pooling of raw rows into compressed rows', DEC)]),
    kmod('indexer_topk', 'Sparse-attention indexer: ReLU scores (scalar and WMMA), compressed-key scoring, top-k (bounded, parallel, two-stage), top-k KV gather; Qwen4 QSA scoring, split-K decode and filter-select',
         [P('indexer.score/kv-f32/f32.hipfire', 'indexer_relu_score_*, indexer_compressed_k_score', 'fp32 relu(q . k) summed over heads', DEC),
          P('indexer.topk/-/f32.hipfire', 'indexer_top_k*, indexer_topk_twostage, topk8_select', 'exact top-k on fp32 scores', DEC),
          P('indexer.qsa/-/bf16+f32.hipfire', 'qsa_bf16_scores, qsa_filter_select, qsa_decode_splitk*', 'exact serial-FMA BF16 pooled-key scoring; exact sampled-threshold selection with certified compaction and radix repair', DEC)]),
    kmod('moe', 'Mixture-of-experts: router softmax / top-k renorm, hash router, indexed and batched expert gate-up and down GEMV / GEMM across HFQ, MQ, MFP4 and Q8_0, scatter and combine, expert-parallel row gather',
         [P('moe.route/-/f32.hipfire', 'moe_router_softmax*, moe_topk_renorm*, moe_softmax_topk*, hash_router_normalize*, deepseek4_moe_topk*', 'fp32 softmax and top-k renormalization', DEC),
          P('moe.experts-decode/hipfire-hfq+mq+mfp4/f32', ex('moe', 4), F32 + '; indexed by expert id, k = 8 forms', DEC),
          P('moe.experts-prefill/hipfire-hfq+mq+mfp4/f32', 'gemm_*_moe_* units (grouped MMQ, WMMA)', 'grouped by expert, WMMA or MMQ per variant', PF),
          P('moe.combine/-/f32.hipfire', 'moe_down_combine*, qwen4_moe_scatter*, ffn_moe_combine_policy', 'weighted sum of expert outputs in slot order', DEC)],
         requires=['format.device-view/hipfire-hfq/-', 'format.device-view/hipfire-mq/-', 'format.device-view/hipfire-mfp4/-']),
    kmod('hc', 'Hyper-connections: 4-stream mix, control compute / finalize, Sinkhorn 4x4, alpha apply, input map, stream init from the embedding; Qwen4 fused down and the gfx1151 hyper-read up kernels; stream-mix aggregate / broadcast',
         [P('hc.mix/hipfire/f32', ex('hc'), 'fp32 over 4 residual streams; Sinkhorn-normalized 4x4 mixing', DEC + '; ' + PF)]),
    kmod('gdn', 'Gated DeltaNet: the recurrence with F32 / Q8 / Q4 state (tiled LDS, tree and lanes variants), chunked WMMA forms for gfx1151, conv1d + SiLU front, alpha gate, gated norm',
         [P('gdn.recurrence/-/f32.hipfire', ex('gdn'), 'fp32 state S[128x128] per head tiled in 8-row chunks; Q8 / Q4 state variants quantize the state', DEC + '; ' + PF),
          P('gdn.front/-/f32.hipfire', 'conv1d_silu*, alpha_gate, gated_norm', 'causal conv width 4 + SiLU, fp32', DEC)]),
    kmod('norm_rope', 'RMSNorm (row-split, residual-add and at-slot forms), L2 norm, LayerNorm, RoPE in its interleaved / half-split / partial / tail / YaRN / mRoPE forms, Givens and FWHT (MQ) rotations',
         [P('norm.rms/-/f32.hipfire', 'rmsnorm, rmsnorm_rowsplit, rmsnorm_residual_add, fused_rmsnorm_mq_rotate', 'fp32', DEC),
          P('rope.apply/-/f32.hipfire', 'rope_partial_*, rope_tail_*, rope_mrope_*, rope_batched', 'fp32 rotation; partial (first rot dims) and tail (last dims) layouts, YaRN scaling', DEC),
          P('rotate.fwht/-/f32', 'mq_rotate_x_*, rotate_x_mq_awq, givens_rotate*', 'FWHT-256 sign-rotated input for MQ weights; Givens rotations for KV', DEC)]),
    kmod('sampling', 'Device sampling: argmax (single, batched, token chain), top-k logits, top-p sampling (parallel), softmax with temperature, logit penalties and softcap, greedy accept',
         [P('sample.topk/-/f32.hipfire', 'topk_logits, sample_top_p*, softmax_temp_batched, logit_penalty_table', 'fp32', '1 row x vocab'),
          P('sample.argmax/-/f32.hipfire', 'argmax, argmax_batched, argmax_token_chain', 'argmax', '1..N rows x vocab')]),
    kmod('speculation', 'Draft and verify helpers: DFlash draft attention (scalar and WMMA over f16 KV), DDTree mask / Gumbel top-k / walk, sampled-accept and chain-accept, DSpark staged KV and lazy sample-accept, draft row packing, verify twins',
         [P('draft.dflash/hipfire/f32', 'attention_dflash* units', 'draft-only numerics', 'draft blocks of 32..128 rows'),
          P('speculation.accept/-/f32.hipfire', 'chain_accept_spec, dflash_sampled_accept_sparse_q, dspark_sample_accept_lazy, spec_sampling_cpu_ar*', 'lossless accept / residual sampling on the device', 'verification rows'),
          P('draft.ddtree/-/f32', 'ddtree_build_attn_mask, ddtree_gumbel_topk_batched, ddtree_swor_walk', 'draft tree construction', 'draft trees')]),
    kmod('embedding', 'Embedding lookup from F32 / F16 / BF16 / Q8 / Q4_K / HFQ4 / MQ4V2 tables, single and batched, and external scatter',
         [P('embed.lookup/hipfire-hfq+q8+bf16/f32', ex('embedding'), 'row dequantized to fp32', DEC)]),
    kmod('tp', 'Tensor-parallel and plumbing kernels: 16-way peer exchange, slow-link prefill staging, railgun copy / fill, device counters',
         [P('collective.device-stage/-/f32.hipfire', 'tp_exchange16, slowlink_prefill', 'bytes only', 'TP peers')]),
    kmod('tensor_ops', 'Pointwise and tensor utilities: add, mul, scale, SiLU / sigmoid / GELU forms, casts and dequantizers to F16 / BF16, deinterleave, repeat-interleave, transpose, strided copies, NPU trunk consumers',
         [P('tensor.ops/-/f32.hipfire', ex('tensor_ops', 8), 'fp32 pointwise; casts round to nearest even', 'any rows'),
          P('residual.add/-/f32.hipfire', 'add_inplace_f32 (a[i] += b[i]), add_f32 (c = a + b), broadcast_add_rows_f32', 'one IEEE fp32 add per element', 'element-wise')]),
]

# ---- prebuilt PeaceMaker artefacts for gfx1151
pm = [f for f in O.expand(['kernels/*gfx1151*', 'kernels/*gfx1151*/**']) if 'negative' not in f and not f.endswith(('.s', '.proof.json', '.isa'))]
pre = []
for f in pm:
    if f.endswith(('.hxaco', '.co')):
        pre.append(dict(file='src/' + f, sha256=sha256(O.find(f)), bytes=os.path.getsize(O.find(f)), kind='hxaco bundle' if f.endswith('.hxaco') else 'code object'))
mods.append(M('kernel.hipfire.pm_gfx1151', f'Prebuilt PeaceMaker kernels for gfx1151: {sum(1 for p in pre if p["kind"] == "hxaco bundle")} .hxaco bundles (typed machine code emitted without hipcc, certified against hipcc twins), with their M7 identity records and shape contracts',
              pm, lang='hsaco', includes=[], compile='prebuilt', prebuilt=pre, variants=len(pre),
              provides=[P('linear.decode/hipfire-mq/f32.pm', 'mq4_rows_x4_pm_gfx1151, qwen4_mq4_x4_pm_gfx1151, qwen4_mq6_x4_pm_gfx1151, bf16_rows_wide_pm_gfx1151, legacy_mq4_rowtiles_pm_gfx1151', 'certified equal to the hipcc twin kernel on the shape contract (proof.json); exact-columns row tiles', '4 rows a tile, decode'),
                        P('moe.experts-decode/hipfire-mq+e8p/f32.pm', 'mq_i4_moe_gemv_*_pm_gfx1151, e8p_i4_moe_gemv_pm_gfx1151, qwen4_moe_iu4_sym_*_pm_gfx1151, qwen4_moe_gate_up_stream8_pm_gfx1151, qwen4_shared_gu_pm_gfx1151', 'integer-4 MoE GEMV per shape contract (r128_down, r256_gate_up)', DEC),
                        P('indexer.qsa/-/bf16+f32.pm', 'qsa_select_pm_gfx1151, qsa_select_bf16_pm_gfx1151, qsa_gather_pm_gfx1151', 'exact selection, certified against the hipcc arm', DEC),
                        P('speculation.accept/-/f32.pm', 'spec_accept_ordered_pm_gfx1151, spec_sampling_rows_parallel_pm_gfx1151, mtp_dense_det_pm_gfx1151, mtp_q8_det_pm_gfx1151', 'deterministic MTP draft heads and ordered accept', 'draft rows')],
              reason='prebuilt code objects: emitted by the PeaceMaker generator (crates/peacemaker-*), not by hipcc; hashes are recorded in [[prebuilt]]',
              notes='The matching generator crates (peacemaker-*, railgun-*) are not harvested. The assembly listings (.s, .isa) and proof files (.proof.json, 20 MB) are generator outputs and are left out, as is kernels/qwen4_moe_gfx1151_negative (deliberately wrong objects for the certifier tests).'))


def crate(mid, name, summary, provides, requires=(), exclude=(), notes=None, extra=()):
    return M(mid, summary, [f'crates/{name}/src/', f'crates/{name}/Cargo.toml', f'crates/{name}/map.md', *extra], exclude=list(exclude),
             display_paths=[f'crates/{name}/src/', f'crates/{name}/Cargo.toml', f'crates/{name}/map.md', *extra],
             lang='rust', compile='n/a', provides=provides, requires=list(requires), reason=NOTBUILT, notes=notes)


mods += [
    crate('format.hipfire.quantize', 'hipfire-quantize', 'The hipfire quantizer / converter: HFQ, MQ (FWHT rotation, symmetric, Lloyd MQ4V2), HFP4 / MFP4, E8 lattice and GPTQ, Q4, GGUF input, per-model pipelines (Qwen4 slabs and PLE hot packs, DeepSeek)',
          [P('format.converter/hipfire-hfq+mq+mfp4/-', 'hipfire-quantize (cli.rs, pipeline.rs, quant_mq.rs, quant_hfp4.rs, quant_e8.rs, hfq.rs)', 'calibrated quantization; the .hfq container and qt numbers (docs/quant-formats/qt-register.txt)', 'offline')],
          extra=['docs/quant-formats/'],
          notes='The crate is taken whole, so it also carries image-model pipelines (pipeline_flux.rs, raon.rs, vision_sidecar.rs) that are outside this pass.'),
    crate('format.hipfire.loader', 'hipfire-loader', 'Top-of-DAG model loader and carrier registry: admission, batch staging, native GGUF path, Qwen4 slab loading, speculation build',
          [P('format.loader/hipfire-hfq/-', 'hipfire_loader (Carrier registry, admission.rs, decide.rs)', 'weights uploaded verbatim (passthrough qt) or host-decoded per the qt register', 'load time'),
           P('format.device-view/hipfire-hfq/-', 'HFQ group blocks: [f32 scale][f32 zero][codes]', 'n/a', 'n/a'),
           P('format.device-view/hipfire-mq/-', 'MQ blocks: HFQ payload of FWHT-rotated weights; V2 header = two fp16 scale / zero pairs', 'n/a', 'n/a'),
           P('format.device-view/hipfire-mfp4/-', 'MFP4-G32: FP4 codes with per-32 scales', 'n/a', 'n/a'),
           P('format.device-view/gguf-kquant/-.hipfire', 'gguf_native: GGUF blocks as stored', 'n/a', 'n/a')]),
    crate('runtime.hipfire.kernel_abi', 'hipfire-kernel-abi', 'KernelAbiV1: one typed contract per kernel family (kernarg fields, operand formats and extents, admitted domain, numeric contract id, per-export arch / block / LDS / grid), validated before enqueue',
          [P('verify.kernel-abi/hipfire-v1/-', 'hipfire_kernel_abi::{KernelAbiV1, validate_invocation, encode_into, check_export}', 'rejects a wrong arch / export / dtype / format / alignment / extent / domain / grid / block / LDS before the launch', 'per launch')]),
    crate('runtime.hipfire.dispatch', 'hipfire-dispatch', 'Kernel dispatch families and pipelines: GEMV / GEMM / attention / fused QKV / MoE / rotation tables choosing a kernel by format, arch and shape; batched layer pipelines, CPU split and offload plans',
          [P('dispatch.kernel-table/hipfire/-', 'hipfire_dispatch::families::{gemv, gemm, attention, fused_qkv, moe, rotation, kv_tier}, tables::*', 'runtime dispatch by (format, arch, shape); this is the layer strixflerp replaces with static assembly', 'per launch')],
          requires=['verify.kernel-abi/hipfire-v1/-', 'device.facade/hip/-.hipfire']),
    crate('runtime.hipfire.rdna_compute', 'rdna-compute', 'The kernel launcher: source registry (kernels.rs maps every kernel name to its HIP source text), JIT compile with a content-keyed cache and packaged kernel packs, arch capability table, typed launch wrappers per family',
          [P('device.facade/hip/-.hipfire', 'rdna_compute::{Gpu, KernelCompiler, ArchCaps, kernels, kernel_registry}', 'n/a', 'JIT compile on first use, then cached code objects'),
           P('verify.kernel-pack/hsaco/-', 'rdna_compute::compiler (pack index, hash validation), code_object.rs', 'content-keyed objects; packaged objects re-verified by hash before load', 'load time')],
          exclude=['crates/rdna-compute/src/cdna/**'],
          notes='kernels.rs is the generator input the units.json files of the kernel modules were derived from. src/cdna (gfx942) is left out, so the crate does not build as is.'),
    crate('runtime.hipfire.hip_bridge', 'hip-bridge', 'Safe Rust FFI to the AMD HIP runtime by dlopen: device, streams, memory, modules, launches',
          [P('device.ffi/hip/-', 'hip_bridge', 'n/a', 'n/a')]),
    crate('runtime.hipfire.runtime', 'hipfire-runtime', 'The inference orchestrator: KV backends and adaptive KV ladder, session and prefix caches, sampler, speculation (DFlash, DSpark core and block controller, DDTree, n-gram and lookup drafts, speculation economics), device mesh / expert-parallel / TP shards, weight stores and pager, tokenizer, serve',
          [P('cache.prompt-state/-/-.hipfire', 'hipfire_runtime::{session_cache, prefix_index, checkpoint_pool, verified_cache}', 'snapshot restore is cold-exact; live continuation is session-exact (arch-qwen4 map.md)', 'RAM tiers'),
           P('kv.tiering/hipfire-kv/-', 'hipfire_runtime::{kv_backend, kv_mode, kv_adaptive}', 'KV format chosen per mode; adaptive ladder transcodes older rows to fewer bits', 'per session'),
           P('speculation.draft/dflash/f32.hipfire', 'hipfire_runtime::{dflash, dflash2_spec, dflash_online, ddtree}', 'draft-only numerics', 'draft blocks'),
           P('speculation.draft/dspark/f32.hipfire', 'hipfire_runtime::{dspark_core, dspark_block_controller}', 'draft-only numerics', 'draft blocks'),
           P('speculation.window-verify/-/-.hipfire', 'hipfire_runtime::{spec, spec_sampling, spec_economics, spec_retire}', 'lossless sampled accept; a measured AR floor decides whether speculation runs', 'per request'),
           P('sample.host/-/f32.hipfire', 'hipfire_runtime::{sampler, batch_sampler}', 'fp32', 'host'),
           P('collective.mesh/-/-.hipfire', 'hipfire_runtime::{device_mesh, mesh_split, ep, ep_tile_exchange, tp_shard, multi_gpu}', 'expert-parallel and TP shards over a device mesh', 'multi-GPU / multi-node'),
           P('serve.daemon/hipfire/-', 'hipfire_runtime::serve', 'n/a', 'host')],
          requires=['device.facade/hip/-.hipfire', 'dispatch.kernel-table/hipfire/-'],
          exclude=['crates/hipfire-runtime/src/imagedec.rs'],
          notes='One crate holds caches, speculation, sampling, mesh and serve; they cannot be harvested as separate modules without splitting the crate. imagedec.rs (image decode) is left out.'),
    crate('executor.hipfire.arch_qwen4', 'hipfire-arch-qwen4', 'Native Qwen4 / Qwen3.8-Flash-Next (arch 16): strict config and admission, typed weights and GPU state, bounded PLE rows, the Qwen4Bundle owner, GPU forward and exact AR lanes, native MTP transactions, expert slabs / tiers / hot sets, mesh TP and slow-link prefill',
          [P('executor.model/qwen3.8-flash-next/hipfire-mq', 'hipfire_arch_qwen4::{Qwen4Bundle, gpu_forward, lane, mtp_spec}', 'fp32 activations on MQ4 / MQ6 weights; exact AR lanes never touch the bundle state; MTP ships a measured AR floor', 'solo, expert-parallel mesh'),
           P('speculation.draft/mtp/f32.hipfire', 'hipfire_arch_qwen4::{mtp_gpu, mtp_lanes, mtp_spec}', 'draft-only numerics, deterministic PM draft heads', 'draft steps')],
          requires=['linear.decode/hipfire-mq/f32', 'linear.prefill/hipfire-mq/f32', 'moe.experts-decode/hipfire-hfq+mq+mfp4/f32', 'moe.route/-/f32.hipfire', 'hc.mix/hipfire/f32',
                    'gdn.recurrence/-/f32.hipfire', 'indexer.qsa/-/bf16+f32.hipfire', 'attention.full/kv-f16+bf16+q8_0+rotated/f32.hipfire', 'norm.rms/-/f32.hipfire', 'rope.apply/-/f32.hipfire',
                    'format.loader/hipfire-hfq/-', 'device.facade/hip/-.hipfire', 'dispatch.kernel-table/hipfire/-', 'cache.prompt-state/-/-.hipfire'],
          exclude=['crates/hipfire-arch-qwen4/src/npu_trunk/**']),
    crate('executor.hipfire.arch_deepseek4', 'hipfire-arch-deepseek4', 'DeepSeek V4 Flash (arch 9): hyper-connections, compressed-KV indexer, tail-only RoPE, Q/O-LoRA, raw SWA; bundle, weights, state, forward; the DSpark and MTP speculators and speculative decode',
          [P('executor.model/deepseek-v4-flash/hipfire-mfp4', 'hipfire_arch_deepseek4::{Deepseek4Bundle, forward, program}', 'fp32 activations; expert-parallel rank state in LoadedModel.ep', 'solo, expert-parallel'),
           P('speculation.draft/dspark/f32.hipfire-ds4', 'hipfire_arch_deepseek4::{dspark_speculator, spec_decode, spec_impl}', 'draft-only numerics', 'draft blocks'),
           P('speculation.draft/mtp/f32.hipfire-ds4', 'hipfire_arch_deepseek4::{mtp, mtp_speculator}', 'no measured AR floor: speculation.mtp = auto keeps plain AR (map.md)', 'draft steps')],
          requires=['attention.sparse-sink/kv-f16/f32.hipfire', 'indexer.score/kv-f32/f32.hipfire', 'indexer.topk/-/f32.hipfire', 'moe.experts-decode/hipfire-hfq+mq+mfp4/f32', 'hc.mix/hipfire/f32',
                    'linear.decode/hipfire-mfp4/f32', 'rope.apply/-/f32.hipfire', 'format.loader/hipfire-hfq/-', 'device.facade/hip/-.hipfire', 'dispatch.kernel-table/hipfire/-']),
]

n_gate_units = sum(len(v) for v in mod_units.values())
ORIGIN_TOML = f'''
# hipfire v0.4.5: Rust workspace + HIP kernels for RDNA GPUs; gfx1151 material only in this catalogue.
name      = "hipfire"
repo      = "https://github.com/warpfront/hipfire"
commit    = "efa8fbb"
branch    = "master"                                     # "Merge v0.4.5 into master"
version   = "0.4.5"
pod_tree  = "/data/ark-inbox/src/github.com__warpfront__hipfire"
reference_checkout = "~/Code/strixflerp-upstream/hipfire"

[licence]
spdx      = "Apache-2.0 OR MIT"
files     = ["LICENSE", "LICENSE-APACHE", "LICENSE-MIT", "NOTICE"]   # copied verbatim next to this file
note      = "Cargo.toml says license = Apache-2.0 and the source headers carry SPDX Apache-2.0; the repository ships both licence texts"

[toolchain]
host      = "Rust workspace (cargo), ROCm loaded dynamically (hip-bridge dlopen)"
device    = "therock-10.2"                               # the README asks for ROCm 7.2 or newer on gfx1151; the gate uses TheRock 10.2 (an earlier run with /opt/rocm-7.2.4 compiled 971 of these units clean, 2026-11-10)
second_path = "PeaceMaker emits typed machine code without hipcc for a subset of kernels (kernels/*.hxaco), certified against the hipcc twins"

[compile]                                                # crates/rdna-compute/src/compiler.rs core_hipcc_flags + rocm_root_flags
driver    = "<rocm>/bin/hipcc"
arch_flag = "--offload-arch=gfx1151"
rocm_flags = ["--rocm-path=<rocm>", "--hip-path=<rocm>"]
flags     = ["--genco", "-O3", "--no-offload-compress", "-fuse-cuid=none", "-mcode-object-version=6"]
system_includes = ["include"]
object_ext = ".hsaco"
per_source = "lines `// HIPFIRE_COMPILER_FLAGS: ...` in a source add flags (compiler.rs per_kernel_flags); recorded per unit in units.json"
output    = "one offload bundle per compile unit (hipcc --genco)"
note      = "A compile unit is a Rust string constant: one source, or defines + sources concatenated (kernels.rs). Headers that the origin splices as text are found through -I here."

[kernels]
loading   = "JIT: hipcc at first use into a content-keyed cache ($HOME/.hipfire_kernels/<arch>), or a packaged, hash-verified kernel pack; PeaceMaker .hxaco bundles are embedded in the binary (include_bytes)"
dispatch  = "runtime tables in hipfire-dispatch choose a kernel by format, arch and shape; contracts in hipfire-kernel-abi"
sources_total = {len(hips)}
sources_gfx1151_or_generic = {len(kept)}
sources_other_arch = {len(foreign)}
sources_left_out_non_llm = {len(left_names)}
sources_dropped_not_gfx1151 = {len(dropped_arch)}
sources_without_unit = {sum(uncovered.values())}
registry_constants = {len(units_raw)}
gate_units = {n_gate_units}
units_skipped = {json.dumps([f'{v}: {k}' for k, v in skipped.items()])}
'''

LEFT_OUT = {
    f'kernels/src: {len(foreign)} sources for other architectures (gfx1030, gfx1100, gfx11_dgpu, gfx12, gfx1200, gfx1201, gfx906, gfx942) and kernels/src/pm_cdna': 'not gfx1151 material',
    f'kernels/src: {len(left_names)} image / audio / bench sources': 'VAE, ViT, vision-language preprocessing, Flux attention and RoPE, Gemma4, RAON, PFlash (legacy), bench, emu and calibration kernels: ' + ' '.join(os.path.basename(f)[:-4] for f in left_names),
    f'kernels/src: {len(dropped_arch)} generically named sources that only build for other architectures': 'the compiler refuses them for gfx1151 (gfx1201-only #error such as the FP8 E4M3 KV writers and attention, dot1-insts wave64 dp4a variants, ggml shims): ' + ' '.join(os.path.basename(f)[:-4] for f in dropped_arch),
    'kernels/*gfx1100*, kernels/*gfx1201*, kernels/pm-decode/gfx1201, kernels/qwen4_moe_gfx11*_negative': 'other-arch prebuilt artefacts and negative certifier fixtures',
    'crates/hipfire-arch-{cohere2moe,diffusion,dots-ocr,embeddinggemma2,gemma4,lfm2moe,lfm2-vl,llama,maple,minimax,muse-glimmer,qwen2,qwen35,qwen35-embed,qwen35-vl,raon,toy}': 'other model families (audio / image / video / embedding, or not served on the pod)',
    'crates/peacemaker-*, railgun*, redline*, radiowave, saddle-*, pm-census, hsa-bridge, va-bridge, hipfire-xdna, npu-tools, hipfire-cpu, hipfire-atlas': 'kernel generator, certifier, record / replay runtime, NPU and CPU back ends: not needed to compile or launch the hipcc kernels',
    'crates/hipfire-{cli,client,config,daemon,detect,engine,generate,registry,reap,tui,pflash,ds4-parent,dispatch-tests}': 'product binaries, config, registry and research crates',
    'crates/*/tests, crates/*/examples, tests/, bench/, benchmarks/, experiments/, autoresearch/, docs/ (except docs/quant-formats), scripts/, tools/': 'tests, examples, benchmarks, documentation, scripts',
}

if __name__ == '__main__':
    run(O, mods, ORIGIN_TOML, LEFT_OUT, ['LICENSE', 'LICENSE-APACHE', 'LICENSE-MIT', 'NOTICE'])
    print('units', n_gate_units, 'skipped', dict(skipped))
