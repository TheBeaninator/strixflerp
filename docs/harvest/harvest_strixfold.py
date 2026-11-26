#!/usr/bin/env python3
"""Harvest StrixFold (the pod's fork, main f44d736) into catalog/strixfold. Run from anywhere: python3 docs/harvest/harvest_strixfold.py
Needs ~/Code/strixflerp-upstream/strixfold, the supplement (zig/build, missing from the reference copy) and
docs/harvest/data/strixfold/ (kit manifests fetched from control-node, see docs/HARVEST.md)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harvestlib import Origin, M, P, run, REPO, tlist

O = Origin('strixfold', 'strixfold', 'https://github.com/Shoestring-dev/Strixfold', 'f64b5ff', 'main', 'Apache-2.0',
           ['StrixFold contributors', 'TensorFold contributors'], 'therock-10.2', supplement='strixfold')
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'strixfold')
K = 'zig/kernels/hip/dsv41/'
F = 'zig/src/families/dsv41/'
REG = json.load(open(O.find(K + 'registry.json')))
SYMS = {s['source']: sorted({k['symbol'] for k in s['kernels']}) for s in REG['sources']}
NKERN = {s['source']: len(s['kernels']) for s in REG['sources']}


def api(src, limit=6):
    s = SYMS[src]
    return ', '.join(s[:limit]) + (f' (+{len(s) - limit} more, see registry.json)' if len(s) > limit else '')


ABI = 'module:kernel.strixfold.dsv41_abi'
DEC = 'decode / verify windows, 1..128 rows, memory-bound'
PF = 'prompt chunks (64..1024 rows), compute-bound on WMMA'
KITFLAGS = ['--offload-device-only', '--no-gpu-bundle-output', '-O3', '-std=c++20', '-fno-gpu-rdc',
            '-mno-wavefrontsize64', '-ffp-contract=off', '-fno-fast-math', '-fdenormal-fp-math=ieee',
            '-D__HIP_PLATFORM_AMD__']


def kern(name, src, summary, provides, extra_paths=(), requires=(), authors=None, notes=None):
    return M(f'kernel.strixfold.{name}', summary, [K + src, *extra_paths], provides=provides, sources=[K + src],
             includes=['src/' + K.rstrip('/')], flags=KITFLAGS, requires=[ABI, *requires], authors=authors, notes=notes,
             variants=NKERN[src], parity='origin-test:tools/dsv41-hip/tri/run_tests.sh')


tri_sums = open(os.path.join(DATA, 'tri-SHA256SUMS')).read()
aot = json.load(open(os.path.join(DATA, 'aot.json')))
tri_fns = sorted({k['fn'] for k in aot['kernels']})
hipkit = json.load(open(os.path.join(DATA, 'dsv41-kit.manifest.json')))

modules = [
    # ---------------------------------------------------------------- kernels (hip-c++)
    M('kernel.strixfold.dsv41_abi',
      'The dsv41 HIP kernel ABI header: version word and every by-value argument struct, shared by hipcc and Zig (translate-c)',
      [K + 'abi.h'], lang='hip-c++', includes=['src/' + K.rstrip('/')], compile='header-only',
      provides=[P('kernel-abi.dsv41-hip/-/-', 'TF_DS_HIP_ABI_VERSION, tf_ds_*_args structs',
                  'fixed-width fields, natural alignment, size and offset static asserts per struct', 'n/a')],
      reason='header-only; compiled inside every other kernel.strixfold.* module'),
    kern('glue', 'glue.hip', 'Byte-moving and widening glue: bf16/f16 to f32, rank-ordered add, row scatter/gather, strided copy, argmax',
         [P('tensor.cast/-/bf16-f32', 'tf_ds_glue_bf16_to_f32, tf_ds_glue_f16_to_f32', 'exact widening', 'any rows, memory-bound'),
          P('residual.add2/-/f32-bf16', 'tf_ds_glue_add2_bf16', 'one IEEE fp32 add of two fp32 partials then one rounding to bf16 (clang float->__bf16, RNE); this is the TP rank-ordered reduction of a 2-rank all-gather', 'any rows, memory-bound'),
          P('tensor.rows/-/-', 'tf_ds_glue_scatter_rows, tf_ds_glue_gather_rows, tf_ds_glue_strided_copy, tf_ds_glue_copy_rows16', 'exact byte moves', 'any rows, memory-bound'),
          P('sample.argmax/-/f32', 'tf_ds_glue_argmax_rows', 'argmax, first index on ties (torch.argmax)', 'rows x vocab fp32 logits, one block of 1024 a row')]),
    kern('aux', 'aux.hip', 'Indexer top-k selection helpers, torch-order RMSNorm, compress2, complex-FMA RoPE, bf16 scale, mean of 4 streams',
         [P('indexer.topk/-/i64-keys', 'tf_ds_aux_topk_i64, tf_ds_aux_topk_indices, tf_ds_aux_cand_fast, tf_ds_aux_block_max, tf_ds_aux_pool_pick, tf_ds_aux_apply_pool, tf_ds_aux_iota_vis',
            'exact: int64 keys = score bits in total order above, ~index below (ties to the lower index, -0 as +0)', DEC),
          P('norm.rms/-/bf16.torch-order', 'tf_ds_aux_rms_norm', 'torch arithmetic at torch rounding points (explicit _rn intrinsics, no contraction); byte-equal to the served torch forward', 'rows x d, memory-bound'),
          P('kv.compress2/-/bf16', 'tf_ds_aux_compress2', 'softmax-weighted pair compression as torch computes it (compress2_softmax)', 'window rows'),
          P('rope.apply/-/bf16.complex-fma', 'tf_ds_aux_rope', 'complex rotation with the CUDA kernel FMA placement (rope_complex_fma), rounded to bf16', 'rows x heads'),
          P('draft.hc-mean4/-/bf16', 'tf_ds_aux_hc_mean4', 'mean of the 4 mHC streams, fp32 then bf16', 'draft rows')],
         notes='Bodies ported from TensorFold CUDA sources zig/kernels/cuda/dsv41_ops.cu and dsv41_torch.cu (sha256 in the file header).'),
    kern('sparse_attn', 'aux_attn.hip', 'DeepSeek-V4.1 sparse attention with a sink (32 heads, head dim 512 a rank): 128-key window ring plus indexer picks, split partials or one pass',
         [P('attention.sparse-sink/kv-bf16/bf16', 'tf_ds_aux_sparse_attn_h32_d512', 'scores fp32, online softmax over 16-key chunks, p rounded to bf16 before the PV product, acc fp32; split mode writes (m, l, acc) partials in _sparse_attn_part layout, one-pass writes bf16(acc / (l + exp(sink - m))); contract sparse_attn_sink_bf16p', DEC + '; also prompt path one-pass'),
          P('attention.sparse-sink/kv-fp4-e2m1/bf16', 'tf_ds_aux_sparse_attn_h32_d512', 'as kv-bf16, keys decoded exactly from packed FP4 (E2M1 codes, E4M3 scale per 16 elements)', DEC)],
         requires=['kv.store/fp4-e2m1/bf16']),
    kern('indexer_score', 'aux_index.hip', 'Indexer scores of a prompt chunk on WMMA: 32 index heads of dim 128 against the FP4 compressed-key cache',
         [P('indexer.score/kv-fp4-e2m1/bf16', 'tf_ds_aux_index_rows_h32_d128', 'score = sum_h relu(q_h . k) * w_h in fp32, fixed head order; k decoded exactly from FP4 (E8M0 scale per 32); -inf past the visible count; optional int64 top-k keys and 64-key tile maxima; within index_scores_fp32_bound of the fp64 reference, not Triton bits', PF)],
         requires=['kv.store/fp4-e2m1/bf16']),
    kern('moe_gate', 'aux_gate.hip', 'MoE router gate logits (and indexer head weights) of a prompt chunk: x bf16 times fp16 weights on bf16 WMMA with an exact hi/lo split',
         [P('moe.gate-logits/f16/bf16-f32', api('aux_gate.hip'), 'each fp16 weight split exactly into two bf16, products exact, WMMA fp32 accumulation (not IEEE RNE), within gamma(5120) of the exact sum (torch fp32 GEMM contract); row-invariant', PF)]),
    kern('mhc_mix', 'aux_hc.hip', 'mHC (hyper-connection) posted streams and mixes of a prompt chunk: 4 streams of 5120, 40 mix blocks of 512',
         [P('hc.mix-prefill/dsv41-mhc-f32/bf16', 'tf_ds_aux_hc_mix_pf_d5120', 'fp32 fma chains in column order, post contraction as _hc_post; hc_mix_fp32_bound (not Triton bits)', PF)]),
    kern('rope_heads', 'aux_rope.hip', 'RoPE on attention heads in place, bit-identical to the Triton _rope_heads',
         [P('rope.heads/-/bf16.hip', 'tf_ds_aux_rope_heads', 'pairs (2j, 2j+1) of the last rd dims; re = fma(x_e, c, -(x_o s)), im = fma(x_e, s, x_o c); each rounded to bf16 RNE; bit-equal to every recorded _rope_heads variant (t_hiprope)', 'rows x heads, memory-bound')]),
    kern('dense_bf16', 'dense.hip', 'BF16-weight dense projections for fine-tuned packs (attn.wo_b, shared expert w2): row-invariant GEMV and SwiGLU',
         [P('linear.decode/bf16/bf16-f32.row-invariant', 'tf_ds_dense_gemv', 'w and x bf16 (products exact in fp32), one fma per element in fixed lane order, 32 lane partials summed by xor shuffles 16,8,4,2,1; y fp32 stored or accumulated with one fp32 add; a row gets the same bits whatever shares the window', DEC),
          P('act.swiglu/-/bf16.dsv41', 'tf_ds_dense_swiglu', 'DeepSeek SwiGLU with its clamps (swiglu_bf16)', DEC)],
         notes='Prompt chunks of these weights go through hipBLAS GemmEx, not this kernel.'),
    kern('exl3', 'exl3.hip', 'EXL3 trellis-coded linears and routed experts: exact MUL1 tile decode, v_dot2_f32_f16 for decode windows, f16 WMMA for prompt chunks',
         [P('linear.decode/exl3/f16-f32.row-invariant', 'tf_ds_exl3_rot_many, tf_ds_exl3_glinear_k<K2>_w<W> (12 variants)', 'input rotated (Hadamard 128, sign vectors) into fp16; each 16x16 weight tile decoded exactly once a wave (MUL1), v_dot2_f32_f16 with fp32 accumulation in a fixed order (split-K partials and wave sums added in fixed order); contract exl3.linear of numerics.json (tolerance vs CUDA bits, not bit-equal)', '1..63 rows glinear; memory-bound'),
          P('linear.prefill/exl3/f16-f32', 'tf_ds_exl3_rot_in, tf_ds_exl3_gemm_k<K2>_m<M> (8), tf_ds_exl3_unpack_k<K2> (6)', 'v_wmma_f32_16x16x16_f16, fp16 operands, fp32 accumulation; contract exl3.linear', PF),
          P('moe.experts-decode/exl3/bf16', 'tf_ds_exl3_experts_decode_prep, tf_ds_exl3_experts_grouped_cp_lo<L>_hi<H>_epi<E> (6), tf_ds_exl3_experts_gateup_epilogue, tf_ds_exl3_experts_down_combine', 'epoch / ready protocol of the decode experts, SwiGLU and combine epilogues in the CUDA order; contract exl3.experts_grouped', DEC),
          P('moe.experts-prefill/exl3/bf16.grouped', 'tf_ds_exl3_experts_group_count, _group_place, _work_list, _rot_in, _grouped_mma_* (3), _grouped_rows_* (3)', 'picks grouped by expert; contract exl3.experts_grouped', PF)],
         requires=['format.device-view/exl3/-'],
         notes='Tile format after ExLlamaV3 (MIT, Copyright (c) 2025 Turboderp). exl3.l2_prefetch is a no-output hint kernel.'),
    kern('exl3_pf_experts', 'exl3_pf.hip', 'Routed experts for prompt chunks: picks grouped into jobs of 64 or 128 rows, gate/up and down on f16 WMMA, deterministic combine',
         [P('moe.experts-prefill/exl3/bf16.jobs', api('exl3_pf.hip', 4), 'exact MUL1 decode, v_wmma_f32_16x16x16_f16 with fp32 accumulation, H128 * svh, DeepSeek SwiGLU with clamps, combine = fma chain of wts * z in slot order from 0; no split K, no atomics: deterministic; contract exl3.experts_grouped', PF)],
         requires=['format.device-view/exl3/-']),
    kern('coll_proxy', 'coll.hip', 'GPU half of the slot collective: stage copies this rank block into a coherent host slab and publishes it, collect waits for the peer block',
         [P('collective.device-proxy/-/-', 'tf_ds_coll_stage, tf_ds_coll_collect', 'exact byte moves; ready / done / consumed generation words with system-scope release / acquire, bounded waits, abort word', 'one gather a launch pair; capturable in a HIP graph')],
         notes='Host side: runtime.strixfold.coll_tcp (tcp_proxy.zig) or runtime.strixfold.coll_rdma (rdma_proxy.zig).'),
    M('kernel.strixfold.triton_kit',
      f'The Triton-compiled half of the dsv41 kit: {len(tri_sums.splitlines())} gfx1151 code objects for {len(tri_fns)} kernel functions ({len(aot["kernels"])} recorded variants), built ahead of time from tri_kernels.py',
      ['tools/dsv41-hip/tri/tri_kernels.py', 'tools/dsv41-hip/tri/aot_build.py', 'tools/dsv41-hip/tri/variants.py',
       'tools/dsv41-hip/tri/make_port.py', 'tools/dsv41-hip/tri/common.py', 'tools/dsv41-hip/tri/kit_table.py',
       'tools/dsv41-hip/tri/SHA256SUMS', 'tools/dsv41-hip/tri/fetch_upstream.sh', 'tools/dsv41-hip/tri/README.md'],
      lang='triton-aot', includes=[], compile='prebuilt', variants=len(aot['kernels']),
      toolchain='',
      upstream='bertholomus/TensorFold @ 222a9c0b6ecaa9fd3c5dbded65a668524c9510ff: deepseek_v41/cuda/{kernels,markov}.py, cuda/exl3/prefill.py (ported by make_port.py)',
      provides=[
          P('hc.mix-decode/dsv41-mhc-f32/bf16', 'tri._hc_partial, _hc_finish, _hc_finish_s, _hc_sinkhorn, _hc_mix_part, _hc_post, _hc_post_only, _collapse, _collapse_norm', 'Triton codegen is the reference: bits are whatever this kit computes; the served engine and the Zig port launch the same objects', DEC),
          P('hc.mix-prefill/dsv41-mhc-f32/bf16.triton-bits', 'tri._hc_mix_pf', 'Triton bits (kernel.strixfold.mhc_mix replaces it within hc_mix_fp32_bound)', PF),
          P('norm.rms/-/bf16.triton-bits', 'tri._rmsnorm, _rmsnorm_rot, _q_kv_norm, _kv_norm_rope', 'Triton bits; norm fused with rotation / RoPE in the _rot and _rope variants', DEC),
          P('rope.heads/-/bf16.triton-bits', 'tri._rope_heads', 'Triton bits (kernel.strixfold.rope_heads is bit-equal)', DEC),
          P('moe.route/-/bf16-f32', 'tri._rowmm2, _rowmm2r, _rowmm_parts, _route, _route_parts', 'router row GEMMs in split parts then top-k route; Triton bits', DEC),
          P('indexer.score/kv-fp4-e2m1/bf16.triton-bits', 'tri._index_score, _index_score_rows, _index_score_cand, _score_keys, _gather_tiles', 'Triton bits', DEC),
          P('indexer.topk/-/i64-keys.triton', 'tri._topk_sel, _topk_finish', 'exact selection on int64 keys', DEC),
          P('attention.sparse-sink/kv-fp4-e2m1/bf16.triton-bits', 'tri._sparse_attn_part, _sparse_attn_merge, _sparse_attn_merge_rot2', 'split partials (m, l, acc) merged in fixed order; Triton bits (kernel.strixfold.sparse_attn replaces the part kernel within its contract)', DEC),
          P('kv.store/fp4-e2m1/bf16', 'tri._fp4_store, _fp4qd_p2', 'FP4 (E2M1) codes with an E4M3 scale per 16 elements (attention keys) or E8M0 per 32 (indexer keys); E4M3 round trip written as _e4m3_rt in the port', DEC),
          P('draft.markov/f16/bf16', 'tri._bias, _score, _finish, _pick, _tap', 'DSpark Markov loop kernels; drafts are output-invariant (verified by the target)', 'draft rows'),
          P('linear.prefill/exl3/f16-f32.triton', 'tri._gemm', 'prompt GEMM of exl3/prefill.py; Triton bits', PF),
          P('embed.lookup/bf16/bf16.dsv41', 'tri._embed_init, _engram_gate', 'embedding rows into the 4 mHC streams; Engram gate', DEC)],
      prebuilt=[dict(kit='dsv41 Triton AOT kit', objects=len(tri_sums.splitlines()),
                     where='control-node:/data/pod/plans/af9276f296/tri-kit-gfx1151/hsaco/<sha256>.hsaco (rsynced from worker A ~/tfp/tri/kit/gfx1151)',
                     manifest='prebuilt/aot.json', manifest_sha256='490a69467c9bfc914e33a2e5d4e9cebd120b04abfa0923b6eecc6bca290000b8',
                     hashes='prebuilt/SHA256SUMS', hashes_digest='sha256 of the sorted object hashes = f65752636db279aab49c8c6efe7df6eed5781893b95bc8532820db30fa0a62f4')],
      extra_files={'prebuilt/aot.json': open(os.path.join(DATA, 'aot.json')).read(), 'prebuilt/SHA256SUMS': tri_sums},
      reason='prebuilt code objects (Triton AOT); rebuilding needs the pinned Triton wheel and python 3.13 venv, which exist on workers F and A only',
      notes='Object file names are their sha256 (checked on control-node for all 158). The .hsaco files are not committed; prebuilt/aot.json is the kit manifest as built.'),

    # ---------------------------------------------------------------- formats
    M('format.strixfold.exl3', 'EXL3 pack loader and device layouts for one TP rank: linears in strips, experts in one trellis buffer with pointer, width and scale tables; checkpoint split and rank cache',
      [F + 'weights.zig', F + 'checkpoint.zig', F + 'rank_cache.zig', F + 'rank_cache_main.zig', F + 'plan.zig',
       ],
      lang='zig', compile='n/a', closure='zig:families/dsv41', 
      provides=[P('format.device-view/exl3/-', 'weights.zig Weights, plan.zig Split, rank_cache.zig', 'bytes as the Python loader writes them (digests per device tensor against weights.py)', 'load time'),
                P('format.loader/exl3-safetensors/-', 'checkpoint.zig, rank_cache.zig (TFDSRK01)', 'exact', 'load time')],
      requires=['device.facade/hip/-', 'format.reader/safetensors/-.strixfold'], reason='zig: built only as part of the executor (phase 3)'),
    M('format.strixfold.safetensors', 'Host readers the core library exports: safetensors headers and tensors, the checkpoint index, O_DIRECT reads, npy files',
      ['zig/src/core/safetensors.zig', 'zig/src/core/direct_io.zig', 'zig/src/core/checkpoint.zig', 'zig/src/core/npy.zig'], lang='zig', compile='n/a', closure='zig:core', 
      provides=[P('format.reader/safetensors/-.strixfold', 'core/safetensors.zig, core/checkpoint.zig, core/direct_io.zig, core/npy.zig', 'exact', 'load time')],
      reason='zig: built as part of the core library (phase 3)'),
    M('format.strixfold.engram', 'Engram n-gram embedding shards: row ids per position (hash rule), FP8 row decode, O_DIRECT / io_uring row reads from the original FP8 tables on NVMe',
      [F + 'engram.zig', F + 'engram_io.zig', F + 'engram_aio.zig', F + 'engram_check_main.zig'], lang='zig', compile='n/a', closure='zig:families/dsv41', 
      provides=[P('format.loader/engram-fp8/-', 'engram.zig, engram_io.zig, engram_aio.zig', 'row ids as numpy computes them; FP8 (E4M3) rows decoded exactly', 'one read per (position, table) a decode round; NVMe-bound')],
      reason='zig: built only as part of the executor (phase 3)'),

    # ---------------------------------------------------------------- runtime
    M('runtime.strixfold.gpu_facade', 'Backend-neutral GPU surface checked at compile time against each backend, with the HIP backend: context, streams, memory, modules, typed kernel launch, graphs, Triton AOT launch',
      ['zig/src/gpu/interface.zig', 'zig/src/gpu/hip_facade.zig', 'zig/src/gpu/staging.zig', 'zig/src/gpu/kernel.zig', 'zig/src/gpu/hip/'],
      lang='zig', compile='n/a', requires=['verify.kit-manifest/hsaco/-', 'module:kernel.strixfold.dsv41_abi'],
      provides=[P('device.facade/hip/-', 'gpu/interface.zig (top, Driver, Stream, Kernel, Graph), gpu/hip/root.zig', 'n/a', 'kernel launch about 0.024 ms on gfx1151; HIP graphs capture decode rounds')],
      reason='zig: built only as part of the executor (phase 3)',
      notes='cuda_facade.zig and zig/src/cuda are not harvested; interface.zig still names both facades.'),
    M('runtime.strixfold.kit', 'Kernel kit loading and verification: raw gfx1151 code objects described by a manifest generated from each object AMDGPU metadata note and re-verified at load; kernel registry and the build step that makes the kit',
      ['zig/src/gpu/hsaco.zig', 'zig/src/gpu/msgpack.zig', 'zig/src/gpu/manifest.zig', 'zig/src/gpu/kit_tool.zig', K + 'registry.json', 'tools/dsv41-hip/kernel_registry.py',
       'tools/dsv41-hip/kernels.json', 'tools/dsv41-hip/KERNELS.md', 'tools/dsv41-hip/numerics/'],
      authors=['StrixFold contributors', 'TensorFold contributors', 'pod'], lang='zig', compile='n/a',
      provides=[P('verify.kit-manifest/hsaco/-', 'tf-hip-kit scan/merge/verify/coverage, gpu.Kit, gpu.Kernel(id, Params).bind', 'load-time check: exact arch, wave size, per-kernel argument table against the hsaco metadata, sha256 of every object', 'load time'),
                P('verify.numerics-contract/-/-', 'tools/dsv41-hip/numerics/numerics.json (w-ref contract 1.7.0)', 'per-operator tolerance bands against fp64 references; the contract the non-bit-exact kernels are judged under', 'n/a')],
      prebuilt=[dict(kit='dsv41 HIP kit as built on control-node by zig build (test kit, HIP objects only)', objects=len(hipkit['objects']),
                     kernels=len(hipkit['kernels']), where='control-node:/data/pod/plans/strixfold/fork/zig-out/native-hip/test/dsv41-kit',
                     manifest='prebuilt/dsv41-kit.manifest.json', toolchain=hipkit['toolchain']['hipcc']),
                dict(kit='dsv41 serving kit (HIP + Triton) on workers F and A', objects=169, kernels=302,
                     manifest_sha256='94bf37dbcba2400d6bca5ec6f38626b092fba9eddb71168ca83c4bd68fbf745a',
                     where='workers F and A: ~/tfp/src/TensorFold/zig-out/native-hip (not read in this pass; figures from /data/pod/plans/strixfold/PATHS.md r6)')],
      extra_files={'prebuilt/dsv41-kit.manifest.json': open(os.path.join(DATA, 'dsv41-kit.manifest.json')).read()},
      reason='zig: built only as part of the executor (phase 3)'),
    M('runtime.strixfold.coll', 'The TP collective interface (Coll): stream-ordered all-gather of fp32 partials in rank order, split begin/end, abort, identity handshake; the ranks control link and fan',
      [F + 'coll.zig', F + 'fan.zig', F + 'link.zig'], lang='zig', compile='n/a', closure='zig:families/dsv41', 
      provides=[P('collective.interface/-/-', 'coll.zig Coll (VTable.all_gather, the gatherBegin / gatherEnd halves, abort, close), Identity, DType, Site', 'a collective only moves bytes; every reduction is an all-gather of fp32 partials each rank adds in rank order, so results are identical on every rank and independent of the transport', 'TP2; first failure poisons the Coll')],
      reason='zig: built only as part of the executor (phase 3)'),
    M('runtime.strixfold.coll_tcp', 'Coll over host-staged TCP on the USB4 / Ethernet links: eager tcp_host, and tcp_proxy (two kernels on the model stream plus a busy-polling proxy thread, graph-capturable)',
      [F + 'tcp_coll.zig', F + 'tcp_proxy.zig', F + 'tcp_wire.zig'], authors=['StrixFold contributors', 'TensorFold contributors', 'pod'], lang='zig', compile='n/a', closure='zig:families/dsv41', 
      provides=[P('collective.allgather-tcp/-/f32', 'tcp_coll.zig (tcp_host), tcp_proxy.zig', 'bytes only; site, layer, byte count and generation checked against the peer on every call', 'TP2; decode slices and prompt blocks')],
      requires=['collective.interface/-/-', 'collective.device-proxy/-/-', 'device.facade/hip/-'],
      reason='zig: built only as part of the executor (phase 3)'),
    M('runtime.strixfold.coll_rdma', 'Coll over RoCE verbs: tcp_proxy slot collective with one RC RDMA write-with-immediate per staged slot (pod card SFR-2); the older one-hop RDMA ring and its sequence bookkeeping',
      [F + 'rdma_proxy.zig', F + 'rdma.zig', F + 'rdma_seq.zig', F + 'rdma_bench_main.zig', 'tools/dsv41-hip/rdma_check.py'],
      lang='zig', compile='n/a', closure='zig:families/dsv41', authors=['pod', 'StrixFold contributors', 'TensorFold contributors'],
      provides=[P('collective.allgather-rdma/-/f32', 'rdma_proxy.zig (TF_COLL_KIND=rdma)', 'bytes only; 64-byte trailer (magic, epoch, sequence, bytes, site, rank) checked per slot; gathers larger than a slot fall back to tcp_host', 'TP2 decode slices; measured 30.8 decode tok/s on F+A (pod /data/pod/plans/af9276f296/NOTES.md)')],
      requires=['collective.interface/-/-', 'collective.device-proxy/-/-', 'module:runtime.strixfold.coll_tcp', 'module:runtime.strixfold.fabric'],
      reason='zig: built only as part of the executor (phase 3)',
      notes='rdma_proxy.zig and rdma_check.py are pod-authored (commits 3da08b5, 7bf2623). rdma.zig is the CUDA-era verbs ring (f44d736: used on CUDA only).'),
    M('runtime.strixfold.coll_nccl', 'Coll over NCCL opened at run time (the CUDA transport); kept because RCCL exports the same API, not exercised on HIP',
      [F + 'comm.zig', F + 'comm_main.zig'], lang='zig', compile='n/a', closure='zig:families/dsv41', 
      provides=[P('collective.allgather-nccl/-/f32', 'comm.zig Comm.coll', 'bytes only; no header, no mismatch check', 'TP2 / TP4 on CUDA')],
      requires=['collective.interface/-/-'],
      reason='zig: in the assembled tree (the family root imports comm.zig) but not reached by the HIP programs; the NCCL path is unverified on gfx1151'),
    M('runtime.strixfold.fabric', 'Host fabric library: verbs ABI and queue pairs, memory registration, RDMA links, mailbox / packet / wire framing used by the transports and the cluster layer',
      ['zig/src/fabric/'], authors=['StrixFold contributors', 'TensorFold contributors', 'pod'], lang='zig', compile='n/a',
      provides=[P('transport.verbs/-/-', 'fabric/verbs.zig, verbs_qp.zig, verbs_link.zig, registration.zig', 'n/a', 'RC queue pairs over RoCE')],
      reason='zig: built only as part of the executor (phase 3)'),
    M('runtime.strixfold.lanes', 'TensorFold 1.0 lane core: the round loop over concurrent streams, the backend vtable it drives (forward, draft, verify windows, keep / drop), draft depth and acceptance, events',
      ['zig/src/core/lanes/*.zig', 'zig/src/core/lane_host.zig', 'zig/src/core/draft_depth.zig', 'zig/src/core/stagger.zig', 'zig/src/core/engine_api.zig',
       '!zig/src/core/lanes/sampling.zig', 'zig/src/core/lane_host_test.zig', 'zig/src/core/lane_host_reuse_test.zig'], lang='zig', compile='n/a', closure='zig:core', 
      provides=[P('executor.lane-backend/-/-', 'core/lanes/backend.zig Backend vtable (Feed, Window, ...)', 'n/a', 'N streams a round; drawn tokens stay on the GPU behind handles'),
                P('speculation.window-verify/-/-.keyed', 'core/lanes/accept.zig, depth.zig, proposer.zig, windows.zig', 'output-invariant: a verify row draws as the serial run at its position (keyed sampler), so drafts never change the emitted tokens', 'draft depth chosen from a cost profile')],
      requires=['sample.keyed/-/f64'], reason='zig: built only as part of the executor (phase 3)'),
    M('runtime.strixfold.sampling', 'The keyed exact sampler: Gumbel draws keyed by (seed, position, token id) so a verify row draws exactly as the serial run; top-k / top-p / min-p in float64 as numpy computes them',
      ['zig/src/core/lanes/sampling.zig', F + 'sampling.zig'], lang='zig', compile='n/a', closure='zig:families/dsv41', 
      provides=[P('sample.keyed/-/f64', 'core/lanes/sampling.zig Sampling, families/dsv41/sampling.zig', 'float64, libm exp/log, numpy pairwise sum order; bit for bit with the Python exact_sampling.py', 'top_k + margin candidates a row, host side')],
      reason='zig: built only as part of the executor (phase 3)'),
    M('runtime.strixfold.grammar', 'Constrained decoding: regex and JSON-schema programs compiled to a token-trie machine',
      ['zig/src/core/lanes/grammar/'], lang='zig', compile='n/a', closure='zig:core', 
      provides=[P('grammar.constraint/-/-.strixfold', 'core/lanes/grammar/grammar.zig', 'n/a', 'host side, per sampled row')],
      reason='zig: built only as part of the executor (phase 3)'),
    M('runtime.strixfold.dspark_draft', 'The DSpark drafter on one TP rank: window rings, target taps absorbed, N streams drafts and confidences from one pass, the Markov loop over a split vocabulary, the confidence head',
      [F + 'draft.zig', F + 'tri_markov.zig', F + 'markov_tokens.zig', F + 'scheduler_costs.zig', 'tools/draft_vocab.py', 'tools/draft_vocab_extend.py'],
      lang='zig', compile='n/a', closure='zig:families/dsv41', 
      provides=[P('speculation.draft/dspark/bf16', 'draft.zig Drafter, tri_markov.zig', 'drafts are proposals only: the target verifies every row, so the output is draft-invariant', 'N streams a pass')],
      requires=['draft.markov/f16/bf16', 'collective.interface/-/-'],
      reason='zig: built only as part of the executor (phase 3)',
      notes='tri_markov.zig also launches the prompt GEMM tri._gemm.'),
    M('runtime.strixfold.prompt_cache', 'Exact prompt reuse for any family: states kept at prompt-pass chunk ends, found by their tokens; states on disk (Imprint) outlive the server',
      ['zig/src/core/prompt_cache.zig', 'zig/src/core/prompt_imprint.zig', 'zig/src/core/segments.zig', 'zig/src/core/keepalive.zig', 'zig/src/core/prompt_cache_test.zig', 'zig/src/core/prompt_imprint_test.zig'],
      lang='zig', compile='n/a', closure='zig:core', 
      provides=[P('cache.prompt-state/-/-.strixfold', 'core/prompt_cache.zig Snapshots vtable, core/prompt_imprint.zig Imprint', 'exact reuse: a restored state is byte-equal to the one saved', 'chunk-end granularity')],
      reason='zig: built only as part of the executor (phase 3)'),
    M('runtime.strixfold.core', 'Host core library the family and server import: tokenizer, chat template engine, small shared helpers',
      ['zig/src/core/tokenizer/', 'zig/src/core/template/', 'zig/src/core/root.zig', 'zig/src/core/affine_mm.zig', 'zig/src/core/copy_index.zig',
       'zig/src/core/fence.zig', 'zig/src/core/frags.zig', 'zig/src/core/hc.zig', 'zig/src/core/ids_json.zig', 'zig/src/core/moe_route.zig',
       'zig/src/core/row_projection.zig'], lang='zig', compile='n/a', closure='zig:core', requires=['format.reader/safetensors/-.strixfold'],
      provides=[P('text.tokenizer/-/-.strixfold', 'core/tokenizer/tokenizer.zig', 'n/a', 'host'), P('text.chat-template/jinja/-.strixfold', 'core/template/template.zig', 'n/a', 'host')],
      reason='zig: built only as part of the executor (phase 3)'),

    # ---------------------------------------------------------------- executor, serve
    M('executor.strixfold.dsv41', 'DeepSeek-V4.1-Flash, one TP rank: prompt chunks, decode / verify rounds, the Triton and HIP kernel launch sites, behind the lane backend vtable and the native family interface',
      [F + '*.zig', F + 'hipblas_abi_check.c', F + 'fixtures/launches.json', F + 'fixtures/derived_launches.json'],
      exclude=[F + x for x in ('weights.zig', 'checkpoint.zig', 'rank_cache.zig', 'rank_cache_main.zig', 'plan.zig', 'engram.zig', 'engram_io.zig', 'engram_aio.zig',
                               'engram_check_main.zig', 'coll.zig', 'fan.zig', 'link.zig', 'tcp_coll.zig', 'tcp_proxy.zig', 'tcp_wire.zig', 'rdma_proxy.zig', 'rdma.zig',
                               'rdma_seq.zig', 'rdma_bench_main.zig', 'comm.zig', 'comm_main.zig', 'sampling.zig', 'draft.zig', 'tri_markov.zig', 'markov_tokens.zig',
                               'scheduler_costs.zig', 'hip_test.zig',
                               'vit.zig', 'vit_main.zig', 'picture.zig', 'picture_main.zig', 'pil.zig', 'fmha.zig', 'vision.zig')],
      sources=[F + 'lanes_main.zig', F + 'layer_main.zig'],
      lang='zig', compile='n/a', closure='zig:families/dsv41', 
      provides=[P('executor.model/deepseek-v4.1-flash/exl3', 'lanes.zig (lane Backend), native.zig (native family), tf-dsv41-lanes, tf-dsv41-layer', 'token identity with the served Python lane is the origin gate (layer_main.zig against recordings); decode rows are row-invariant, drafts output-invariant', 'TP2 over two gfx1151 nodes; 30.8 decode tok/s, 1,226..1,432 prefill tok/s measured on F+A over RDMA')],
      requires=['executor.lane-backend/-/-', 'device.facade/hip/-', 'verify.kit-manifest/hsaco/-', 'collective.interface/-/-', 'collective.allgather-tcp/-/f32', 'collective.allgather-rdma/-/f32', 'collective.allgather-nccl/-/f32',
                'format.device-view/exl3/-', 'format.loader/engram-fp8/-', 'sample.keyed/-/f64', 'speculation.draft/dspark/bf16', 'cache.prompt-state/-/-.strixfold',
                'linear.decode/exl3/f16-f32.row-invariant', 'linear.prefill/exl3/f16-f32', 'moe.experts-decode/exl3/bf16', 'moe.experts-prefill/exl3/bf16.jobs',
                'moe.gate-logits/f16/bf16-f32', 'moe.route/-/bf16-f32', 'attention.sparse-sink/kv-fp4-e2m1/bf16', 'indexer.score/kv-fp4-e2m1/bf16',
                'indexer.topk/-/i64-keys', 'hc.mix-decode/dsv41-mhc-f32/bf16', 'hc.mix-prefill/dsv41-mhc-f32/bf16', 'rope.heads/-/bf16.triton-bits', 'rope.heads/-/bf16.hip',
                'norm.rms/-/bf16.triton-bits', 'kv.store/fp4-e2m1/bf16', 'linear.decode/bf16/bf16-f32.row-invariant', 'residual.add2/-/f32-bf16',
                'module:runtime.strixfold.core', 'module:kernel.strixfold.dsv41_abi', 'module:executor.strixfold.dsv41_vision',
                'module:runtime.strixfold.fabric'],
      reason='zig: assembled and gated in phase 3 (token identity on the worker pair)',
      authors=['StrixFold contributors', 'TensorFold contributors', 'pod'],
      notes='native.zig carries a pod patch (f44d736). cublas.zig / cublaslt.zig are dlopen shims the family imports unconditionally and stay; the *2d* files (the TP4 split, exercised on CUDA only) are included because round.zig, prompt.zig and draft.zig import them; model.zig imports vision.zig unconditionally, so the vision tower is its own module (executor.strixfold.dsv41_vision) that this one requires; fixtures/*.json are embedded by tri.zig.'),
    M('executor.strixfold.dsv41_vision', 'The DeepSeek-V4.1 vision tower on one rank: picture decode and resize as PIL does it, ViT blocks, fused multi-head attention, the image rows handed to the model',
      [F + 'vision.zig', F + 'vit.zig', F + 'picture.zig', F + 'pil.zig', F + 'fmha.zig'], lang='zig', compile='n/a', closure='zig:families/dsv41', 
      provides=[P('vision.tower/deepseek-v4.1/bf16', 'vision.zig, vit.zig (Vit), picture.zig, pil.zig, fmha.zig', 'not measured here: the phase 3 gate sends text only', 'one image a request, prompt path')],
      reason='zig: in the assembled tree because model.zig imports vision.zig, but a HIP build analyses none of it (gpu.features.vision is false, native.zig); never compiled or run here',
      notes='Imports model.zig, cublas.zig, cublaslt.zig and exact.zig of executor.strixfold.dsv41: the two modules are one Zig import closure and are cut apart for the record only. vit_main.zig and picture_main.zig (stand-alone checkers) are not harvested.'),
    M('runtime.strixfold.engine_host', 'The HIP engine host the server links as native_engines: opens a family for a model directory, memory budget and cache fit, the engine API handed to the HTTP side',
      ['zig/src/native/hip.zig', 'zig/src/native/none.zig', 'zig/src/native/cache_fit.zig', 'zig/src/native/cuda_memory.zig'], lang='zig', compile='n/a',
      provides=[P('executor.host/hip/-', 'native/hip.zig (open, families), native/cache_fit.zig, native/cuda_memory.zig (budget)', 'n/a', 'one engine a process')],
      requires=['executor.model/deepseek-v4.1-flash/exl3', 'executor.lane-backend/-/-', 'device.facade/hip/-'],
      reason='zig: assembled in phase 3',
      notes='cuda_memory.zig is the memory budget arithmetic hip.zig imports under that name; it holds no CUDA call.'),
    M('tool.strixfold.checkpoint_cli', 'The checkpoint command line the server binary carries (tensorfold-native models / info / pull): hub cache lookup and download',
      ['zig/src/cli/cli.zig', 'zig/src/cli/hub.zig', 'zig/src/cli/info.zig', 'zig/src/cli/models.zig', 'zig/src/cli/pull.zig'], lang='zig', compile='n/a',
      provides=[P('tool.checkpoint-cli/-/-', 'cli/cli.zig', 'n/a', 'host')],
      requires=['executor.host/hip/-'], reason='zig: assembled in phase 3 (imported by server/main.zig as checkpoint_cli)'),
    M('tool.strixfold.zig_build', 'The origin build graph: root build.zig, the HIP half (code objects by hipcc, the kit by tf-hip-kit scan, the native-hip step), the CUDA and dist halves build.zig imports, the pinned Zig version, the build wrapper',
      ['build.zig', 'build.zig.zon', '.zig-version', 'zig/build/hip_dsv41.zig', 'zig/build/cuda.zig', 'zig/build/dist.zig', 'tools/dsv41-hip/zig_build.sh'],
      lang='zig', compile='n/a', authors=['StrixFold contributors', 'TensorFold contributors', 'pod'],
      provides=[P('build.graph/zig/-', 'zig build native-hip -Dhipcc= -Drocm= -Dhip-triton-kit=', 'n/a', 'build time')],
      reason='zig: the build files of the phase 3 recipe (flerplib/assemble_zig.py lays the tree out and runs them unchanged)',
      notes='zig/build/*.zig were missing from the laptop reference copy (it excluded directories named build) and come from the pod tree by way of ~/Code/strixflerp-upstream-supplement. cuda.zig and dist.zig are needed because build.zig imports them; their targets name files that are not harvested and are never made by the native-hip step.'),
    M('serve.strixfold.http', 'TensorFold 1.0 HTTP server: OpenAI chat / responses and Anthropic messages translation, SSE, tool-call parsing, auth, metrics, compaction',
      ['zig/src/server/'], lang='zig', compile='n/a', sources=['zig/src/server/main.zig'],
      provides=[P('serve.http/openai+anthropic/-.strixfold', 'server/main.zig, server/serve.zig', 'n/a', 'host')],
      requires=['executor.lane-backend/-/-', 'grammar.constraint/-/-.strixfold', 'text.tokenizer/-/-.strixfold', 'text.chat-template/jinja/-.strixfold',
                'executor.host/hip/-', 'tool.checkpoint-cli/-/-', 'cache.prompt-state/-/-.strixfold'],
      reason='zig: assembled in phase 3'),
]

LICENCE_FILES = ['LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md'] + [f for f in O.files() if f.startswith('LICENSES/')]

ORIGIN_TOML = f'''
# StrixFold: the pod's fork of StrixFold / TensorFold with the AMD HIP backend for DeepSeek-V4.1-Flash (TP2).
name      = "strixfold"
repo      = "https://github.com/Shoestring-dev/Strixfold"
commit    = "f64b5ff"
branch    = "main"
pod_tree  = "/data/pod/plans/strixfold/fork"            # the pod's git tree; 8 pod commits on top of Shoestring-dev 88ea779 (re-cut from f44d736 on 2026-11-11)
upstream  = "https://github.com/ashhart/TensorFold by way of https://github.com/bertholomus/TensorFold (222a9c0b for the Triton kernels)"
reference_checkout = "~/Code/strixflerp-upstream/strixfold (+ ~/Code/strixflerp-upstream-supplement/strixfold/zig/build, dropped by the copy)"

[licence]
spdx      = "Apache-2.0"
# copied verbatim next to this file at their origin paths: the root texts and LICENSES/, which THIRD_PARTY_NOTICES.md cites
files     = {tlist(LICENCE_FILES)}
source    = "the reference checkout above, byte-identical to f64b5ff in control-node:/data/pod/plans/strixfold/fork (git show f64b5ff:<path>, checked 2026-11-26)"
third_party = "EXL3 tile format after ExLlamaV3 (MIT, Turboderp); DeepSeek reference code (MIT); see THIRD_PARTY_NOTICES.md"

[toolchain]
host      = "zig 0.17.0 (/home/pod/opt on control-node)"
device    = "therock-10.2"                               # /data/models/tools/therock-10.2: HIP 7.17.26392, AMD clang 24
documented_pin = "ROCm 10.0.0-4 (HIP 7.15, clang 23) under /opt/rocm/core-10.0 (tools/dsv41-hip/BUILD_AND_RUN.md); the pod builds with TheRock 10.2 instead"
triton    = "triton 3.8.0+git675c5987.rocm10.1.0a20260822 (cp313 wheel, sha256 1bab2eeb...), ahead of time, target hip:gfx1151:32"
blas      = "hipBLAS by dlopen (libhipblas.so.3) for fp32 GEMMs"

[compile]                                                # how the origin compiles a HIP kernel source (zig/build/hip_dsv41.zig)
driver    = "<rocm>/bin/hipcc"
flags     = {json.dumps(KITFLAGS)}
arch_flag = "--offload-arch=gfx1151"
rocm_flag = "--rocm-path=<rocm>"
output    = "raw gfx1151 ELF code object (.hsaco), one per .hip source"

[kernels]
loading   = "a manifest-described kit of raw gfx1151 code objects; the manifest is generated from each object's AMDGPU metadata note and re-verified at load (gpu/hip/kit.zig); kernels are bound by id with a typed parameter struct (gpu.Kernel)"
registry  = ["zig/kernels/hip/dsv41/registry.json", "tools/dsv41-hip/kernels.json", "tools/dsv41-hip/KERNELS.md"]
hip_sources = {len([s for s in SYMS if s not in ('abi_echo.hip', 'smoke.hip')])}
triton_objects = {len(tri_sums.splitlines())}
'''

LEFT_OUT = {
    'zig/src/cuda, zig/kernels/cuda, zig/src/gpu/cuda_facade.zig, zig/src/native/cuda*.zig, zig/src/cli/cuda_*': 'CUDA backend',
    'zig/src/metal, zig/kernels/metal, zig/src/native/metal.zig, zig/src/core/checkpoint_metal.zig': 'Metal backend',
    'zig/src/families/{flashnext,glm,kimi_k3,nemotron,qwen3_5}, zig/src/native/{flashnext_host,glm_host,qwen35}.zig': 'model families with no HIP path',
    'zig/src/families/dsv41/{vit_main,picture_main}.zig': 'stand-alone vision checkers (the tower itself is executor.strixfold.dsv41_vision)',
    'zig/src/families/dsv41/hip_test.zig, zig/tests, zig/src/families/dsv41/fixtures (except the two launch recordings tri.zig embeds)': 'the family test root, the test programs and their fixtures. The *_test.zig, tests.zig and fake.zig files next to the sources ARE harvested: Zig 0.17 loads every file a compiled file imports, test blocks included',
    'zig/kernels/hip/dsv41/{abi_echo,smoke}.hip': 'ABI echo and runtime smoke kernels (test kit only)',
    'zig/src/cluster, zig/src/cli/{cuda_*,tests}.zig': 'multi-node cluster manager, CUDA command lines and CLI tests: not on the dsv41 HIP serving path',
    'tools/dsv41-hip (except the registry, numerics contract, tri generator inputs, zig_build.sh, rdma_check.py)': 'qualification, bench, deploy and input-preparation scripts, dev harnesses',
    'docs, assets, packaging, *.md at the root, setup.py, tools/release, tools/zig, tools/dsv41-zig': 'documentation, packaging, release tooling',
}

RECIPE_RESULT = os.path.join(REPO, 'gate', 'results', 'dsv41-flash.strixfold-tp2')


def recipe_status():
    """Zig modules build only inside the recipe: when its gate's build stage passed on control-node, the modules of that
    assembly move to status "builds" with the gate's summary as evidence (the manifests say so, not this script alone)."""
    try:
        summary = json.load(open(os.path.join(RECIPE_RESULT, 'summary.json')))
        assembled = json.load(open(os.path.join(RECIPE_RESULT, 'assembly.json')))['modules']
    except (OSError, ValueError):
        return 0
    if not summary.get('stages', {}).get('build', {}).get('pass'):
        return 0
    n = 0
    for m in modules:
        # in the tree and loaded by the compiler, but behind a feature the HIP build has off: Zig analyses none of it
        if m.lang != 'zig' or m.id not in assembled or m.id in ('executor.strixfold.dsv41_vision', 'runtime.strixfold.coll_nccl'):
            continue
        path = os.path.join(REPO, 'catalog', 'strixfold', m.kind, m.name, 'module.toml')
        text = open(path).read()
        head, gate = text.rsplit('[gate]', 1)
        lines = [l for l in gate.splitlines() if l.strip() and not l.startswith(('status', 'evidence', 'reason'))]
        lines += ['status    = "builds"', 'evidence  = "gate/results/dsv41-flash.strixfold-tp2/summary.json"',
                  'reason    = "zig: builds as part of the recipe dsv41-flash.strixfold-tp2 on control-node (zig 0.17.0, TheRock 10.2); Zig compiles what the built programs reach, not every declaration; not run on a GPU from that build"']
        open(path, 'w').write(head + '[gate]\n' + '\n'.join(lines) + '\n')
        n += 1
    return n


if __name__ == '__main__':
    run(O, modules, ORIGIN_TOML, LEFT_OUT, LICENCE_FILES)
    print(f'strixfold: {recipe_status()} zig modules at status builds from the recipe gate')
