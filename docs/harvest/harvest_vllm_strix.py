#!/usr/bin/env python3
"""Harvest vllm-strix-halo (davidcanar/vllm-strix-halo at fe5ac10) into catalog/vllm_strix.
Run: python3 docs/harvest/harvest_vllm_strix.py               (rebuild catalog/vllm_strix)
     python3 docs/harvest/harvest_vllm_strix.py --side-gates  (control-node CPU: py_compile every .py, git apply --numstat
                                                               every .patch / .diff, gcc the two OdinLink C helpers)
The HIP units go through the catalogue's own gate: python3 gate/compile/run.py --origin vllm_strix.

Card ORIGIN-VLLM-STRIX d2a7b91dec. vllm-strix-halo serves GLM-5.3-Flash (AWQ W4A16, DFlash2 k=3, CUDA graphs) and
DeepSeek-V4-Flash (native vLLM 0.31 port, DSpark k=5) tensor-parallel over two Strix Halos (gfx1151) joined by one
Thunderbolt / USB4 cable, on vLLM pinned at 73859fec. The fabric is OdinLink (Geramy/OdinLink-Five, its own origin
catalog/odinlink5): RCCL rides the odl_tb5 net plugin for prefill collectives, odl_ar2 carries the decode all-reduces
and odl_mq moves the EngineCore <-> worker data plane off TCP.

Scope: the repo's own work, one module a piece (FAMILIES below): the HIP kernels (int4 / MXFP4 MoE GEMVs, W8A16 /
FP8 / BF16 / FP32 GEMVs and the fused router, the odl_ar2 all-reduce), the Triton kernels it adds to vLLM, every
patch and install hook it applies to vLLM (sparse indexer fixes, prefix-cache, speculative decoding, router, MoE,
CUDA-graph, gfx1151 platform gates, the DS4 native port), the OdinLink hooks and host kit, the serving scripts and
the bench / probe / trace tools. Not harvested: vLLM itself (container/pinned-vllm and pinned-triton are vLLM and
Triton files with the patches applied, kept by the origin for byte-exact rebuilds), the two prebuilt .so libraries,
binary data, documentation and history (README.md, PATCHES.md and the HANDOFF notes are read for the records).

The reference checkout is the ark copy on control-node
(/data/ark-inbox/src/github.com__davidcanar__vllm-strix-halo#fe5ac1016737c1f68e04772e3ae9e7567bc40f22, a tree with a
shallow .git); the laptop holds the same files, moved by a tarball through control-node get, in
~/Code/strixflerp-upstream-supplement/vllm_strix. VLLM_STRIX_CHECKOUT names another copy. The tree digest pins the bytes."""
import ast, datetime, glob, hashlib, json, os, re, shutil, subprocess, sys, tempfile, tomllib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvestlib
from harvestlib import Origin, M, P, run

CHECKOUT = os.environ.get('VLLM_STRIX_CHECKOUT', os.path.join(harvestlib.SUPPLEMENT, 'vllm_strix'))
COMMIT = 'fe5ac1016737c1f68e04772e3ae9e7567bc40f22'
# sha256 over the lines "<sha256 of the file>  <origin-relative path>\n" of every file, sorted by path (.git left out)
TREE_SHA256 = '480b9fad2d62c95ccc23a0b9db91e211fbaf0a10c81440755b8c791e64e1dc04'
SKIP = ('.git',)
ODL5 = os.environ.get('ODINLINK5_CHECKOUT', os.path.join(harvestlib.SUPPLEMENT, 'odinlink5'))


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


class TreeOrigin(Origin):
    """The origin's file set is the checkout's files (an ark tree), pinned by .git/HEAD and the tree digest."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.roots = [CHECKOUT]
        head = os.path.join(CHECKOUT, '.git', 'HEAD')
        if os.path.isfile(head) and open(head).read().strip() != COMMIT:
            raise SystemExit(f'{head} names {open(head).read().strip()}, the harvest pins {COMMIT}')
        got = tree_digest(CHECKOUT, self.files())
        if got != TREE_SHA256 and os.environ.get('VLLM_STRIX_PRINT_DIGEST'):
            print(got); sys.exit(0)
        if got != TREE_SHA256:
            raise SystemExit(f'{CHECKOUT}: tree digest {got}, the harvest pins {TREE_SHA256} (vllm-strix-halo {COMMIT[:12]})')

    def files(self):
        if self._files is None:
            self._files = tree_files(CHECKOUT)
        return self._files


O = TreeOrigin('vllm_strix', 'vllm_strix', 'https://github.com/davidcanar/vllm-strix-halo', COMMIT, 'main',
               'Apache-2.0', ['davidcanar (vllm-strix-halo)'], '')
CP = 'container/patches/'
DS4REF = CP + 'ds4-native-fixed-refs/'
SC = 'scripts/'
H = 'host/'
SP = 'container/rootfs/opt/venv/lib/python3.12/site-packages/'
AR2 = ['odinlink/ar2/odl_ar2.hip', 'container/native/odl_ar2.hip']
VLLM = ['vLLM project contributors (upstream code; this repo ports or patches it)', 'davidcanar (vllm-strix-halo)']
DS4V = ['davidcanar (vllm-strix-halo)', 'AlexKGwyn/ds4-vllm (design of the DS4_ODL_AR2 hook and odl_mq, Apache-2.0)']
NOT_RUN = 'Nothing was run: no GPU, no worker, no weights; the pod has no Thunderbolt pair (card d2a7b91dec).'
RIG = 'two Strix Halo (gfx1151, Radeon 8060S) boxes, TP=2 over one Thunderbolt / USB4 cable'
PATCHLIKE = ('.patch', '.diff')


def F(id, paths, summary, why, provides=(), tag=None, requires=(), sources=(), notes='', authors=None, lang=None,
      exclude=(), compile=None):
    return dict(id=id, paths=list(paths), summary=summary, why=why, provides=list(provides), tag=tag,
                requires=list(requires), sources=list(sources), notes=notes, authors=authors, lang=lang,
                exclude=list(exclude), compile=compile)


FAMILIES = [
 # ---------------------------------------------------------------- HIP kernels
 F('kernel.vllm_strix.moe_int4_gemv', [CP + 'vsh_moe_int4.hip'],
   'decode-specialised MoE expert GEMVs for gfx1151 in one HIP unit, wvSplitK style (16 contiguous packed bytes a lane, one wave an output row): vsh_moe_int4_gemv on moe_align blocks (AWQ W4A16, symmetric uint4b8, group 128), vsh_moe_int4_direct (gate-up then down straight from the routed ids, no moe_align), vsh_moe_topk_sum, vsh_moe_swiglu, and five generations of the MXFP4 direct expert kernel (v1 to v5: row-pair passes, fp16 v_perm decode plus dot2, K-chunked staging)',
   'routed MoE was 90 ms of a GLM-5.3 decode step on the stock Triton fused_moe_kernel_gptq_awq (70-80 GB/s, padded to BLOCK_SIZE_M) and 37 ms with this kernel (PATCHES 28.5); the MXFP4 v5 kernel took DeepSeek-V4 routed experts from ~99 ms a step to parity with the June stack (PATCHES 38-41). The shape, a skinny expert GEMV over routed ids that skips the align pass, is the decode MoE strixflerp needs for AWQ int4 and MXFP4 checkpoints',
   provides=[P('moe.experts-decode/awq-int4/bf16', 'extern "C" vsh_moe_int4_gemv, vsh_moe_int4_direct, vsh_moe_topk_sum, vsh_moe_swiglu (stream first, bf16 activations)', 'fp32 accumulate, bf16 out; symmetric uint4b8 with one scale per 128 K', 'decode, <= 64 token-expert pairs, E=288 N=1024 (GLM-5.3-Flash TP=2)'),
             P('moe.experts-decode/mxfp4/bf16', 'extern "C" vsh_moe_mxfp4_direct, vsh_moe_mxfp4_v2 .. v5 (phase 0 = w13, 1 = w2; mt / rp / rc tile knobs)', 'e2m1 values with e8m0 block scales decoded in-kernel, fp32 accumulate', 'decode, E=256 (DeepSeek-V4-Flash TP=2), tuned on real routing dumps')],
   tag='moe.experts-decode', sources=[CP + 'vsh_moe_int4.hip'],
   notes='Interface tag: moe.experts-decode (formats awq-int4 and mxfp4 are new to the catalogue). The prebuilt libvsh_moe_int4.so the origin ships is not harvested. The ctypes wrapper and its vLLM hooks are runtime.vllm_strix.moe_int4_dispatch and runtime.vllm_strix.mxfp4_direct. ' + NOT_RUN),
 F('kernel.vllm_strix.w8a16_gemv', [CP + 'vsh_w8a16.hip'],
   'skinny decode GEMVs for gfx1151 in one HIP unit (one wave an output row, VEC x 16 contiguous bytes a lane): W8A16 (int8 weights, one fp32 scale per 128 K, bf16 in / out) and its dequant, block-scaled FP8 (128x128, ue8m0) GEMV and dequant, BF16-weight and FP32-weight GEMVs with fp32 output for router gates, and vsh_sigmoid_bias_topk, the fused sigmoid + correction bias + deterministic top-8 + renormalise router',
   'the GLM AWQ checkpoint keeps ~49 ms a step of BF16 linears and DeepSeek-V4 its block-FP8 linears; on gfx1151 hipBLASLt and the Triton block GEMM run those at 3-55 GB/s for M <= 8, and these GEMVs replace them (PATCHES 28.6, 30-32, 36). A W8A16 / FP8 decode GEMV with a router fused in is the small-M linear strixflerp lacks for BF16 and FP8 checkpoints',
   provides=[P('linear.decode/w8a16-g128/bf16', 'extern "C" vsh_w8a16_gemv, vsh_w8_dequant', 'int8 symmetric weights, fp32 scale per 128 K, fp32 accumulate, bf16 out', 'M <= 8 rows'),
             P('linear.decode/fp8-block128/bf16', 'extern "C" vsh_fp8_gemv, vsh_fp8_dequant (multi-row variants by argument)', 'e4m3 weights, ue8m0 128x128 block scales, fp32 accumulate', 'decode sizes, K up to 12288 (DSpark drafter fc)'),
             P('moe.gate-logits/bf16/bf16-f32', 'extern "C" vsh_w16_gemv, vsh_w32_gemv', 'bf16 or fp32 weights, fp32 accumulate and output (rel err ~2e-6 for w32)', 'router gates 288 x 4096 and 256 x 4096, M <= 8'),
             P('moe.router/-/f32.sigmoid-bias-topk', 'extern "C" vsh_sigmoid_bias_topk', 'sigmoid, + correction bias, top-k by biased value desc then index asc, renormalised unbiased weights', 'GLM-5.3 top-8 of 288')],
   tag='linear.decode', sources=[CP + 'vsh_w8a16.hip'],
   notes='Interface tag: linear.decode (also moe.gate-logits and moe.router). The prebuilt libvsh_w8a16.so is not harvested. The load-time requantiser and hooks are runtime.vllm_strix.w8a16_requant, gate_bf16, gate_lmhead, fused_router, fp8_gemv_hook. ' + NOT_RUN),
 F('runtime.vllm_strix.odl_ar2', AR2 + ['odinlink/ar2/odl_ar2.py', SP + 'odl_ar2.py'],
   'odl_ar2, a two-rank decode all-reduce over OdinLink streams: on the stream a D2H copy into a pinned send slot, a doorbell kernel and a wait+add kernel that spins on a pinned per-slot flag and adds the peer slot in place (UMA zero-copy); a CPU progress thread sends each round with one STREAM_SEND ioctl (<= 64 KiB latency path, 4024-byte fragments) and polls nonblocking STREAM_RECV; rendezvous by a TCP barrier; ctypes wrapper odl_ar2.py',
   'decode all-reduces are tiny (a 48 KiB message is 13 frames) and latency-bound; RCCL over a net plugin pays proxy and syscall cost per step, so the rig carries them on a dedicated path and leaves RCCL the big prefill collectives. Stream order makes message k round k, so slot reuse needs no extra handshake. This is the shape a two-host strixflerp pair over USB4 would use for its per-layer partial sums',
   provides=[P('collective.allreduce-p2p/-/bf16.odl-stream', 'extern "C" odl2_init(rank, dev_index, peer_ip, port), odl2_enqueue(stream, dst, src, count, dtype), odl2_seq, odl2_shutdown; python odl_ar2.py', 'elementwise SUM, dtype bf16 / fp16 / fp32, bf16 and fp16 accumulate in fp32; both ranks compute the same sum', 'two ranks, one OdinLink device, decode-sized messages; progress thread spins hot for 5 ms after activity then sleeps 200 us')],
   tag='collective.allreduce-p2p', sources=['odinlink/ar2/odl_ar2.hip'], requires=['module:runtime.odinlink5.lib'],
   authors=DS4V,
   notes='Interface tag: collective.allreduce-p2p (the catalogue providers run over RoCE; this one over OdinLink streams). container/native/odl_ar2.hip and site-packages/odl_ar2.py are byte-identical copies of odinlink/ar2/ (the image build reads the former); both kept, one compiled. Needs <odl_tb5/odl_tb5.h> from runtime.odinlink5.lib (OdinLink-Five 0315225 in the catalogue; the origin builds against 4534f58 + odinlink-local.patch, runtime.vllm_strix.odinlink_local_patch). The vLLM hook is runtime.vllm_strix.odl_ar2_hook. ' + NOT_RUN),
 F('tool.vllm_strix.odl_ar2_bench', ['odinlink/ar2/odl_ar2_bench.cpp'],
   'standalone two-rank bench for odl_ar2: loads libodl_ar2.so, runs warm-up and timed all-reduces of decode-sized bf16 / fp16 / fp32 buffers on a HIP stream and checks the sums against the expected rank pattern',
   'the all-reduce has to be measured and checked without vLLM in the way: this bench isolates the transport and the wait+add kernels, the first thing to run when bringing a USB4 pair up',
   provides=[P('tool.collective-bench/-/-', 'odl_ar2_bench <rank> <peer_ip> ... (main in odl_ar2_bench.cpp)', 'checks exact sums', 'two hosts, decode-sized messages')],
   tag=None, sources=['odinlink/ar2/odl_ar2_bench.cpp'],
   notes='Interface tag: none (tool). Compiled as HIP host code (-x hip). ' + NOT_RUN),
 F('tool.vllm_strix.fp8_sweep', [SC + 'fp8sweep.hip', SC + 'fp8sweep.py', SC + 'time12288.py'],
   'tile sweep for the FP8 block-scaled GEMV: fp8sweep.hip holds the kernel with ROWS output rows a wave (activation reused across rows) behind one sweep(variant, ...) entry; fp8sweep.py and time12288.py time the variants against vsh_fp8_gemv at DeepSeek-V4 decode shapes and at K=12288 (DSpark drafter fc, row stride 12544)',
   'the multi-row FP8 GEMV in kernel.vllm_strix.w8a16_gemv was picked by this sweep (PATCHES 38, 40); the sweep is how its tile choice would be re-derived for another checkpoint or clock',
   provides=[P('tool.kernel-bench/fp8-block128/-', 'extern "C" sweep(variant, stream, W, S, X, Y, N, K, M, ldw, ss, sx, sy); python drivers', 'timing only', 'decode shapes')],
   tag='tool.kernel-bench', sources=[SC + 'fp8sweep.hip'],
   notes='Interface tag: tool.kernel-bench. ' + NOT_RUN),
 F('tool.vllm_strix.w8_sweep', [SC + 'w8sweep.hip', SC + 'w8sweep.py', SC + 'bf16_bw.py'],
   'tile sweep for the W8A16 GEMV (ROWS rows a wave, VEC x 16 contiguous bytes a lane a step) behind sweep(variant, ...), its ctypes driver, and bf16_bw.py, a torch bandwidth probe for BF16 GEMV at the same shapes',
   'the W8A16 GEMV parameters were chosen by this sweep against measured BF16 bandwidth (PATCHES 23, 28.6); the pair says how close to the memory roof a skinny int8 GEMV gets on gfx1151',
   provides=[P('tool.kernel-bench/w8a16-g128/-', 'extern "C" sweep(variant, stream, W, S, X, Y, N, K, M, sx, sy)', 'timing only', 'M <= 8')],
   tag='tool.kernel-bench', sources=[SC + 'w8sweep.hip'],
   notes='Interface tag: tool.kernel-bench. ' + NOT_RUN),
 # ---------------------------------------------------------------- Triton kernels added to vLLM
 F('kernel.vllm_strix.kpool_paged_logits', [CP + 'vsh_kpool_paged_logits.py', SC + 'test_kpool_logits.py', SC + 'test_prefill_logits.py'],
   'Triton decode indexer-logits reader for the GLM-5.3 kpool cache on gfx1151: each indexer page is [page_size x 128 fp8 e4m3fn in 16x16 SHUFFLE order][page_size fp32 scales], and the kernel scores every pool of the page-level block table directly (no fp8 copy), with unit tests on the real cache layout for decode and prefill',
   'aiter stage1 scored this layout at random on gfx1151, so decode attention past ~2K tokens attended to arbitrary pools (one of the four stacked long-context bugs, PATCHES 27); the reader also dropped a 12.6 ms a step fp8 copy. A paged, layout-aware indexer scorer is what a strixflerp GLM-5.3 sparse-attention path needs',
   provides=[P('attn.indexer-score/fp8-kpool-shuffle/f32', 'vsh_kpool_paged_logits(q_fp8 / q, weights, kv_cache, block_table, seq_lens, ...) (Triton)', 'fp8 e4m3fn K with fp32 per-pool scales, fp32 logits', 'decode, page = 64 pools = 256 tokens')],
   tag='attn.indexer-score',
   notes='Interface tag: attn.indexer-score. The install hook is runtime.vllm_strix.kpool_paged_logits_hook. ' + NOT_RUN),
 F('kernel.vllm_strix.idx_logits', [CP + 'vsh_idx_logits.py', SC + 'test_idx_logits.py', SC + 'idx_bench.py'],
   'Triton prefill indexer logits for DeepSeek-V4 on gfx1151 on fp16 WMMA: logits[q, k] = kv_scale[k] * sum_h w[q, h] * relu(Q[q, h, :] . K[k, :]) over k in [ks[q], ke[q]), several query rows a program so K tiles are reused, with its exactness test against aiter fp8_mqa_logits and a context sweep of the indexer ops',
   'the indexer scores every earlier compressed position, the quadratic term of DS4 prefill; aiter ran it at ~3 TFLOPS on gfx1151, and this kernel is one of the three changes that took DS4 prefill to ~460 tok/s at 8K and 331 at 128K (PATCHES 42)',
   provides=[P('attn.indexer-score/fp8-ds4/f32.wmma', 'vsh_idx_logits(q, kv, kv_scale, weights, ks, ke) (Triton)', 'fp16 WMMA dot of dequantised operands, fp32 logits; selected top-k exact vs aiter', 'prefill, 2048-token chunks')],
   tag='attn.indexer-score',
   notes='Interface tag: attn.indexer-score. The install hook is runtime.vllm_strix.ds4_idx_logits_hook. ' + NOT_RUN),
 F('kernel.vllm_strix.sparse_attn_split', [CP + 'vsh_sparse_attn_split.py', SC + 'test_split.py'],
   'Triton split-KV (flash-decoding) variant of vLLM\'s _sparse_attn_prefill_ragged_kernel for GLM-5.3 sparse MLA decode: the ~2048 selected latent keys (512-dim) of each query are split across programs and combined by log-sum-exp, with a unit test against the original kernel',
   'GLM decode ran its sparse MLA through the prefill ragged kernel with one program per (query, 16-head block): 8 workgroups at M=4 and 2 at M=1 on a 40-CU GPU (PATCHES 28.4). Split-KV over the selected keys is the decode shape for any top-k sparse attention',
   provides=[P('attn.sparse-mla/-/bf16.split-kv', 'vsh_sparse_attn_split(...) (Triton; same arguments as the ragged prefill kernel)', 'bf16 in, fp32 softmax and accumulation, log-sum-exp combine', 'decode, small query counts (MTP verify rows, draft steps)')],
   tag=None,
   notes='Interface tag: none (the catalogue\'s sparse attention is attn.sparse-sink, a different selection). Hook: runtime.vllm_strix.sparse_attn_split_hook. ' + NOT_RUN),
 # ---------------------------------------------------------------- dispatch wrappers and vLLM hooks around the kernels
 F('runtime.vllm_strix.moe_int4_dispatch', [CP + 'vsh_moe_int4.py', CP + 'vsh-moe-int4-hip.py', SC + 'moe_mt.py'],
   'ctypes wrapper for libvsh_moe_int4.so (vsh_moe_int4.py: gemv, direct_moe, topk_sum, swiglu, MXFP4 entry points with shape checks) and the install hook that routes decode-sized int4 fused-MoE calls (<= 64 token-expert pairs) from the Triton gptq_awq kernel to the HIP GEMV behind VSH_MOE_INT4_HIP; moe_mt.py times the tile options',
   'the kernel is only useful once vLLM calls it: this is the seam that swaps one MoE implementation for another inside the fused-MoE layer by batch size, the same seam a strixflerp adapter would occupy',
   provides=[P('moe.experts-decode-dispatch/awq-int4/-', 'vsh_moe_int4.gemv / direct_moe (python); env VSH_MOE_INT4_HIP', 'n/a (dispatch)', 'decode only; prefill stays on Triton')],
   tag=None, requires=['module:kernel.vllm_strix.moe_int4_gemv'],
   notes='Interface tag: none. Byte-identical copy in container/pinned-vllm/vllm/model_executor/layers/fused_moe/vsh_moe_int4.py (not harvested). ' + NOT_RUN),
 F('runtime.vllm_strix.moe_direct', [CP + 'vsh-moe-direct.py', SC + 'test_moe_direct.py', SC + 'test_pad_ids.py'],
   'decode MoE straight to HIP: a hook in CompressedTensorsWNA16MoEMethod.apply that, for <= 64 token-expert pairs, calls vsh_moe_int4.direct_moe (three launches: gate-up, SwiGLU, down + top-k sum) instead of the modular kernel; tests against the stock pipeline and for padding rows (expert id -1 contributes nothing)',
   'the modular kernel spent ~0.4 ms of host Python a layer on prepare / finalize, workspace sizing, moe_align_block_size and five launches (PATCHES 31); at 42 layers that was a large part of a CPU-bound step. Collapsing decode MoE to three launches is the host-side half of a fast decode MoE',
   provides=[P('moe.experts-decode-direct/awq-int4/bf16', 'hook on CompressedTensorsWNA16MoEMethod.apply; env VSH_MOE_DIRECT', 'same as the gemv path', 'decode, <= 64 pairs')],
   tag='moe.experts-decode', requires=['module:runtime.vllm_strix.moe_int4_dispatch'], notes='Interface tag: moe.experts-decode (host path). ' + NOT_RUN),
 F('runtime.vllm_strix.mxfp4_direct', [CP + 'vsh-mxfp4-direct.py', SC + 'test_mxfp4_direct.py'],
   'direct HIP MXFP4 MoE for DeepSeek-V4 decode: a hook that, for <= 64 token-expert pairs on gfx1151, replaces UnfusedOAITritonExperts (triton_kernels matmul with in-kernel upcast, routing-data build, activation, index_add combine) with the vsh_moe_mxfp4 kernels on the same swizzled weights; test against the stock pipeline',
   'the stock path took ~99 ms of the DS4 decode step (PATCHES 38); the direct kernels brought DS4 to June-stack parity and then past it with the v5 kernel. The shape, MXFP4 experts decoded in the GEMV itself, fits strixflerp\'s DeepSeek-V4.1 MXFP4 experts',
   provides=[P('moe.experts-decode-direct/mxfp4/bf16', 'hook on the OAI Triton experts; env VSH_MXFP4_DIRECT (version and tile from env)', 'e2m1 + e8m0 decoded in-kernel, fp32 accumulate', 'decode, E=256')],
   tag='moe.experts-decode', requires=['module:runtime.vllm_strix.moe_int4_dispatch'], notes='Interface tag: moe.experts-decode. ' + NOT_RUN),
 F('tool.vllm_strix.mxfp4_sweeps', [SC + p for p in ('moe_dump_ids.py', 'moe_phases.py', 'moe_replay.py', 'moe_v4sweep.py', 'moe_v5phase.py', 'moe_v5sweep.py', 'moe_versions.py')],
   'MXFP4 direct-MoE tuning kit: a routing dump of real DS4 decode (moe_dump_ids.py), replay of that routing with per-phase timing (moe_replay.py), and sweeps of the v3 / v4 / v5 kernels by phase and tile at DS4 TP=2 decode shapes with correctness against v1',
   'v5 was tuned on recorded routing rather than uniform random ids, which changed the winner (PATCHES 41); a routing-replay sweep is the method to reuse for any MoE kernel tuned for a real model',
   provides=[P('tool.kernel-bench/mxfp4/-.routing-replay', 'python scripts (VSH_MX_DUMP_IDS=prefix records ids)', 'timing and max-abs-diff vs v1', 'decode, E=256')],
   tag='tool.kernel-bench', notes='Interface tag: tool.kernel-bench. The routing dump the origin keeps (scripts/data/ds4_dspark_routing.pt) is binary data and is not harvested. ' + NOT_RUN),
 F('runtime.vllm_strix.w8a16_requant', [CP + 'vsh_w8a16.py', CP + 'vsh-w8a16-linear.py', SC + 'test_w8a16.py'],
   'load-time int8 (W8A16, symmetric, group 128) for BF16 linears: after loading, UnquantizedLinearMethod quantises eligible layers (bf16, 2-D, K in a fixed set 1024..8192, N >= 32, not kv_b_proj / wk_weights_proj / embeddings / lm_head / vision) and frees the BF16 copy, and apply() runs the HIP W8A16 GEMV for M <= 8; vsh_w8a16.py is also the ctypes wrapper for every libvsh_w8a16.so entry point',
   'the AWQ checkpoint leaves attention, KDA / MLA projections, shared experts and dense MLPs in BF16, ~49 ms a step of skinny GEMV; int8 at load halves the bytes without a new checkpoint (PATCHES 23, 28.6). Requantise-at-load is a cheap way for strixflerp to serve BF16 leftovers of a quantised checkpoint',
   provides=[P('linear.requant-load/bf16-to-w8a16-g128/-', 'vsh_w8a16.maybe_quantize(layer) / apply (python); env VSH_W8A16', 'per-group absmax int8, fp32 scales', 'load time; decode M <= 8 on the GEMV, larger M dequantises')],
   tag=None, requires=['module:kernel.vllm_strix.w8a16_gemv'], notes='Interface tag: none (no catalogue module requantises at load). Copy in container/pinned-vllm/vllm/model_executor/layers/vsh_w8a16.py (not harvested). ' + NOT_RUN),
 F('runtime.vllm_strix.gate_lmhead', [CP + 'vsh-gate-lmhead.py', SC + 'test_w32.py'],
   'two decode linears off hipBLASLt: GateLinear with fp32 weights routed to the w32 GEMV for <= 8 rows (tier "vsh"), and an int8 copy of lm_head used for target and drafter logits; test of the w32 GEMV',
   'the fp32 router gate cost ~150 us x 42 layers a step through x.float() + an fp32 hipBLASLt GEMM, and the BF16 lm_head is the largest single GEMV of a step; both paid more than their arithmetic (PATCHES 30)',
   provides=[P('moe.gate-logits/f32/f32', 'hook on GateLinear (tier vsh); env VSH_ROUTER_GEMV', 'fp32 weights and accumulate, rel err ~2e-6', 'M <= 8'),
             P('logits.lm-head/w8a16-g128/bf16', 'int8 lm_head copy; env VSH_W8A16_LMHEAD', 'int8 group-128 weights', 'decode, target and DFlash2 drafter')],
   tag='moe.gate-logits', requires=['module:runtime.vllm_strix.w8a16_requant'], notes='Interface tag: moe.gate-logits. ' + NOT_RUN),
 F('runtime.vllm_strix.gate_bf16', [CP + 'vsh-gate-bf16.py', SC + 'test_gate_bf16.py', SC + 'test_gate_ds4.py', SC + 'test_gate_engage.py'],
   'bf16-weight, fp32-output router gate GEMV: GateLinear tier 4 (torch.mm(out_dtype=fp32) on hipBLASLt) is routed to vsh_w16_gemv for <= 8 rows; tests that GLM\'s 288 x 4096 and DS4\'s 256 x 4096 gates engage it and match',
   'on the 73859fec pin GLM-5.3\'s gate is BF16 with an fp32 output and took ~150 us a layer, ~6 ms a step (PATCHES 32); the same GEMV serves DS4\'s gate',
   provides=[P('moe.gate-logits/bf16/f32', 'hook on GateLinear tier 4; env VSH_ROUTER_GEMV', 'bf16 weights, fp32 accumulate and output', 'M <= 8')],
   tag='moe.gate-logits', requires=['module:runtime.vllm_strix.w8a16_requant'], notes='Interface tag: moe.gate-logits. ' + NOT_RUN),
 F('runtime.vllm_strix.fused_router', [CP + 'vsh-fused-router.py', SC + 'test_fused_router.py', SC + 'test_fused_router_nan.py'],
   'one HIP kernel for GLM routing: GroupedTopKRouter on ROCm is replaced by vsh_sigmoid_bias_topk (sigmoid, correction bias, deterministic top-8 by biased value then index, renormalise the unbiased weights); tests for equality with the stock router and for NaN inputs',
   'the torch.compile\'d grouped_topk cost ~200 us of CPU plus ~50 us of Dynamo guards a MoE layer, 42 layers a step (PATCHES 31, 33): routing was host-bound, not GPU-bound',
   provides=[P('moe.router/-/f32.sigmoid-bias-topk-fused', 'hook on GroupedTopKRouter; env VSH_FUSED_ROUTER', 'deterministic order; same experts and weights as the stock router', 'decode and prefill, top-8 of 288')],
   tag='moe.router', requires=['module:runtime.vllm_strix.w8a16_requant'], notes='Interface tag: moe.router. ' + NOT_RUN),
 F('runtime.vllm_strix.router_det_topk', [CP + 'vsh-moe-router-deterministic-topk.patch', 'upstream/test_grouped_topk_determinism.py', 'upstream/PR-BODY.md'],
   'deterministic MoE-router top-k for vLLM\'s grouped_topk_router: _deterministic_topk_indices selects experts by a stable descending sort so ties break by index, with the upstream test and the ready-to-send pull-request body (vLLM #55514 is still open)',
   'torch.topk on ROCm breaks ties differently run to run, so the same prompt routed differently and greedy decoding diverged; a deterministic router is a precondition for reproducible MoE serving (PATCHES 14)',
   provides=[P('moe.router/-/f32.stable-topk', 'patch to vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py', 'ties broken by lower expert index', 'all batch sizes')],
   tag='moe.router', notes='Interface tag: moe.router. PR-BODY.md is the origin\'s draft upstream PR text. ' + NOT_RUN),
 F('runtime.vllm_strix.router_packed_topk', [CP + 'vsh-router-packed-topk.py', SC + 'test_router.py'],
   'the same deterministic expert order without the slow sort: the router packs (biased score, expert index) into one sortable key and takes torch.topk of the packed keys, replacing the stable descending sort; test against the sort version',
   'inside grouped_topk\'s torch.compile region inductor lowered the stable sort into a persistent Triton kernel launched with 2 x 32 threads, 242 us a call, 45 calls a step, ~10.9 ms (PATCHES 28.3)',
   provides=[P('moe.router/-/f32.packed-key-topk', 'patch script over grouped_topk_router', 'identical order to router_det_topk', 'decode')],
   tag='moe.router', requires=['module:runtime.vllm_strix.router_det_topk'], notes='Interface tag: moe.router. ' + NOT_RUN),
 F('runtime.vllm_strix.fp8_gemv_hook', [CP + 'vsh-fp8-gemv.py', SC + 'test_fp8_gemv.py', SC + 'test_fp8_preq.py'],
   'HIP FP8 block-scaled GEMV for DeepSeek-V4\'s decode-sized linears: TritonFp8BlockScaledMMKernel (fp8 activation quant + Triton block GEMM) is replaced by vsh_fp8_gemv on bf16 activations for small M; tests against an exact dequantised reference, the stock W8A8 path and a pre-quantised route',
   'gfx1151 has no fp8 dot in Triton, and the block GEMM reached 3-11 GB/s at decode sizes, ~87 % of the DS4 decode step (PATCHES 35, 36)',
   provides=[P('linear.decode/fp8-block128/bf16.hook', 'hook on TritonFp8BlockScaledMMKernel.apply', 'fp32 accumulate on dequantised fp8', 'decode M <= 8')],
   tag='linear.decode', requires=['module:runtime.vllm_strix.w8a16_requant'], notes='Interface tag: linear.decode. ' + NOT_RUN),
 F('runtime.vllm_strix.fp8_prefill', [CP + 'vsh-fp8-prefill.py', SC + 'test_fp8_prefill.py'],
   'DeepSeek-V4 block-FP8 linears at prefill sizes: the fp8 weight is dequantised once to an exact bf16 copy by a HIP kernel and the GEMM runs on hipBLASLt, instead of fp8 activation quant + Triton _w8a8_triton_block_scaled_mm; test against fp32 and the stock path',
   'the Triton block GEMM took 2.5-9 ms a linear at a 512-token chunk, 49 % of DS4 prefill GPU time, because gfx1151 has no fp8 matrix path (PATCHES 41): trading memory for a bf16 copy is the right call on a 128 GB UMA box',
   provides=[P('linear.prefill/fp8-block128/bf16.dequant-hipblaslt', 'hook on the block-FP8 linear for M > 8', 'exact bf16 dequant, hipBLASLt bf16 GEMM', 'prefill chunks')],
   tag='linear.prefill', requires=['module:runtime.vllm_strix.w8a16_requant'], notes='Interface tag: linear.prefill. ' + NOT_RUN),
 F('runtime.vllm_strix.sparse_attn_split_hook', [CP + 'vsh-sparse-attn-split.py'],
   'install hook that routes GLM sparse MLA decode with small query counts (MTP verify rows, draft steps) from the ragged prefill kernel to the split-KV kernel',
   'the split kernel only pays when the query count is small; the hook carries that crossover so prefill keeps the original kernel (PATCHES 28.4)',
   provides=[P('attn.sparse-mla-dispatch/-/-', 'patch script over vllm/v1/attention/ops/rocm_aiter_mla_sparse.py', 'n/a (dispatch)', 'decode')],
   tag=None, requires=['module:kernel.vllm_strix.sparse_attn_split'], notes='Interface tag: none. ' + NOT_RUN),
 # ---------------------------------------------------------------- sparse indexer fixes (GLM-5.3 long context, PATCHES 27)
 F('runtime.vllm_strix.kpool_paged_logits_hook', [CP + 'vsh-kpool-paged-logits.py'],
   'install hook that points GLM-5.3\'s decode indexer at the paged kpool logits reader, behind VSH_KPOOL_PAGED_LOGITS',
   'without it decode scored pools at random past ~2K tokens on gfx1151; with the other three section-27 fixes it made retrieval pass to 255K (PATCHES 27, 27.1)',
   provides=[P('attn.indexer-score-dispatch/fp8-kpool-shuffle/-', 'patch script over vllm/models/glm5next/amd/sparse_indexer.py', 'n/a', 'decode')],
   tag='attn.indexer-score', requires=['module:kernel.vllm_strix.kpool_paged_logits'], notes='Interface tag: attn.indexer-score. ' + NOT_RUN),
 F('runtime.vllm_strix.pr59412_indexer_blocks', [CP + '59412-indexer.diff', CP + '59412-pooled-indexer-kernel-blocks.py'],
   'port of upstream vLLM PR #59412 (fixes #58858): on ROCm the DSA indexer backend advertised kernel block sizes [1, MultipleOf(16)], so the hybrid KV manager\'s block became the kernel block for the MLA / indexer group; the port makes kernel blocks page-aligned to the kpool pages (64 pools = 256 tokens)',
   'with mismatched blocks every pool past ~4K tokens aliased onto one page, the second of the four long-context bugs (PATCHES 27); the lesson for strixflerp is that the KV block of a pooled indexer must equal its page',
   provides=[P('kvcache.block-align-indexer/-/-', 'diff to vllm/v1/attention/backends/mla/indexer.py plus its install script', 'n/a', 'GLM-5.3 kpool, any context')],
   tag=None, authors=VLLM, notes='Interface tag: none. ' + NOT_RUN),
 F('runtime.vllm_strix.idx_force_tail', [CP + 'vsh-idx-force-tail-boost.py', CP + 'vsh-idx-force-tail-sparse_indexer.py'],
   'force-tail for the GLM sparse indexer: the 128 most recent pools are boosted before top-k, so they are always selected without overwriting chosen columns; vsh-idx-force-tail-sparse_indexer.py is the earlier column-overwrite version (sparse_indexer.py replaced whole), kept by the origin and superseded',
   'the first version wrote recent pools over the last 128 columns of an unsorted top-k output and discarded ~1/4 of the indexer\'s picks at random (PATCHES 27); boosting before selection is the correct way to guarantee a local window in top-k sparse attention',
   provides=[P('attn.indexer-topk/-/f32.force-tail', 'patch script over glm5next/amd/sparse_indexer.py', 'recent pools get +inf-like boost before top-k', 'decode and prefill')],
   tag='attn.indexer-topk', authors=VLLM, notes='Interface tag: attn.indexer-topk. ' + NOT_RUN),
 F('runtime.vllm_strix.idx_fp8max_fnuz', [CP + 'vsh-idx-fp8max-fnuz.py'],
   'indexer K quantised to an fp8 max of 224 instead of 448 so the e4m3fn to fnuz cast on the paged-MQA logits path cannot overflow to NaN, behind VSH_IDX_FP8MAX_FNUZ',
   'gfx1151 reports float8_e4m3fn (max 448) while the aiter logits path reads fnuz (max 240): every vector\'s absmax element became NaN after the cast and decode scored NaN (PATCHES 27). Any fp8 cache crossing the fn / fnuz line on RDNA needs this headroom',
   provides=[P('attn.indexer-quant/fp8-e4m3fn/-.fnuz-safe', 'patch script over glm5next/amd/ops/kpool_compress.py', 'absmax scaled to 224', 'GLM-5.3 kpool writer')],
   tag=None, notes='Interface tag: none. ' + NOT_RUN),
 F('runtime.vllm_strix.idx_pool_subchunks', [CP + 'vsh-idx-pool-aligned-subchunks-indexer.py'],
   'GLM indexer prefill split into pool-aligned sub-chunks: vLLM\'s mla/indexer.py (replaced whole) cuts prefill indexer work at pool boundaries so no chunk splits a 4-token pool',
   'a chunk boundary inside a pool made the compressed pool see half its tokens; aligning sub-chunks to pools keeps the kpool compression exact across chunked prefill',
   provides=[P('attn.indexer-chunking/-/-.pool-aligned', 'replacement vllm/v1/attention/backends/mla/indexer.py', 'n/a', 'chunked prefill')],
   tag=None, authors=VLLM, notes='Interface tag: none. A whole vLLM file with the change inside. ' + NOT_RUN),
 F('runtime.vllm_strix.idx_bt_gather_v4', [CP + 'vsh-idx-bt-gather-v4-ops.py'],
   'gather-site block-table expansion for the GLM indexer (a replacement ops file): the page-level block table is expanded to pool level where the gather reads it; superseded by the section-27 fixes and inert in the current image',
   'kept as the record of the first attempt at the aliasing bug, which treated the symptom at the gather instead of the block size (PATCHES 27); useful to recognise the failure mode',
   provides=[P('attn.indexer-gather/-/-.bt-expand', 'replacement ops file (whole)', 'n/a', 'inert')],
   tag=None, authors=VLLM, notes='Interface tag: none. Superseded upstream of this rig; harvested for the record. ' + NOT_RUN),
 F('runtime.vllm_strix.ragged_compact_ops', [CP + 'upstream-window-ragged-compact-ops.py'],
   'upstream\'s ragged-index rewrite of the sparse-attention window ops, carried as a whole replacement file ahead of the pin: compacted ragged indices for windowed sparse attention',
   'the rig needed the upstream rewrite before its pin had it; carrying a forward port as a whole file is how the origin tracks upstream fixes between pin bumps',
   provides=[P('attn.sparse-window-ops/-/-.ragged-compact', 'replacement ops file (whole)', 'n/a', 'GLM-5.3 sparse attention')],
   tag=None, authors=VLLM, notes='Interface tag: none. ' + NOT_RUN),
 F('runtime.vllm_strix.pr55222_workspace', [CP + '55222-glm5next-attention.py', CP + '55222-indexer.py'],
   'ports of upstream vLLM PR #55222 (workspace units) as whole files for glm5next/common/attention.py and the MLA indexer backend: indexer and GLM attention workspaces sized in units the ROCm kernels accept',
   'without the port the rig could not size the indexer workspace on its pin; carried until a pin bump made it upstream',
   provides=[P('attn.workspace-units/-/-', 'replacement vllm/models/glm5next/common/attention.py and v1/attention/backends/mla/indexer.py', 'n/a', 'GLM-5.3')],
   tag=None, authors=VLLM, notes='Interface tag: none. ' + NOT_RUN),
 F('runtime.vllm_strix.pr57979_kda_stride', [CP + '57979-kda-fused_recurrent.py', CP + '57979-kda-kernels.py'],
   'port of upstream vLLM PR #57979, the KDA (Kimi delta attention) stride fix, as whole files of the flash-linear-attention derived Triton kernels used by GLM-5.3\'s linear-attention layers (fused_recurrent.py, kernels.py)',
   'the stride bug corrupted KDA state on non-contiguous inputs; GLM-5.3-Flash is a hybrid of KDA and sparse MLA, so the linear-attention kernels are half the model',
   provides=[P('mixer.kda/-/bf16.stride-fix', 'replacement glm5next/amd/ops/third_party/kda/{fused_recurrent,kernels}.py (Triton)', 'fp32 state', 'GLM-5.3 KDA layers')],
   tag=None, authors=['Songlin Yang, Yu Zhang (flash-linear-attention)', 'vLLM project contributors', 'davidcanar (port)'],
   notes='Interface tag: none (the catalogue has mixer.gdn, a different recurrence). ' + NOT_RUN),
 # ---------------------------------------------------------------- prefix cache, speculative decoding, scheduler
 F('runtime.vllm_strix.apc_align', [CP + 'vsh-glm53-apc-align.py'],
   'prefix-cache alignment for GLM-5.3 with EAGLE / DFlash: the EAGLE last-block drop is resolved to pure-drafter KV groups instead of every group, so prefix hits stop losing a 2304-token page a lookup, behind VSH_GLM53_APC_ALIGN',
   'kv_cache_utils returned early for GLM-5-Next before annotating EAGLE groups, so every group was flagged and every hit dropped a page (PATCHES 18); with the fix a 13.9K repeat went from ~50 s to 0.95 s TTFT',
   provides=[P('cache.prompt-align-spec/-/-', 'patch script over vllm/v1/core/kv_cache_utils.py and kv_cache_coordinator.py', 'n/a', 'prefix caching with a drafter')],
   tag='cache.prompt', notes='Interface tag: cache.prompt (the catalogue\'s cache.prompt-* providers). ' + NOT_RUN),
 F('runtime.vllm_strix.apc_retention', [CP + 'vsh-apc-retention.py'],
   'adds --prefix-cache-retention-interval 2304 (2176 with speculation off) to the GLM serve script behind VSH_GLM53_APC_RETENTION, so KDA / Mamba state checkpoints are kept on the attention page grid',
   'the default retains only semantic checkpoints, the sparsest setting for hybrid models, and the hybrid min() then tied every attention hit to a Mamba checkpoint, so the first repeat missed (PATCHES 20, 34); a hybrid prefix cache needs state checkpoints on the attention grid',
   provides=[P('cache.prompt-state-interval/-/-', 'patch script over host/vsh-manual-serve.sh (a stock vLLM flag)', 'n/a', 'GLM-5.3 hybrid KV')],
   tag='cache.prompt', notes='Interface tag: cache.prompt. ' + NOT_RUN),
 F('runtime.vllm_strix.adaptive_k', [CP + 'vsh-adaptive-k.py', CP + 'vsh-adaptive-k-json-gate.py'],
   'adaptive draft length for speculative decoding wired for async scheduling: an EMA of acceptance sets the verified draft length per step in the scheduler (not in update_draft_token_ids, which async scheduling never calls), with a live JSON override and the fix that lets the JSON force reach the gate; installed, instrumented, off',
   'measured off because step time does not depend on draft length on this rig (PATCHES 17), but the scheduler seam and the EMA policy are what strixflerp would need on hardware where it does',
   provides=[P('speculation.draft-length/-/-.ema', 'patch scripts over vllm/v1/core/sched/scheduler.py; env VSH_ADAPTIVE_K, VSH_ADAPTIVE_K_FORCE, JSON override file', 'n/a', 'async scheduling with a drafter')],
   tag='speculation.draft', notes='Interface tag: speculation.draft (policy, not a drafter). ' + NOT_RUN),
 F('runtime.vllm_strix.dflash_drafter_kv_group', [CP + 'vsh-dflash-drafter-kv-group.patch'],
   'puts the DFlash2 drafter\'s KV into its own group in _get_kv_cache_groups_glm5_next, so the drafter\'s attention layers get a KV spec that matches their backend (TRITON_ATTN) instead of the target\'s sparse MLA groups',
   'DFlash2 k=3 is the default drafter and the fastest mode on this rig (20-28 tok/s, PATCHES 28.7, 30); it could not run until its KV was grouped apart from the target',
   provides=[P('speculation.draft/dflash/bf16.kv-group', 'patch to vllm/v1/core/kv_cache_utils.py', 'n/a', 'GLM-5.3 with the DFlash2 drafter')],
   tag='speculation.draft', notes='Interface tag: speculation.draft. ' + NOT_RUN),
 F('runtime.vllm_strix.eagle3_aux_states', [CP + 'vsh-glm-eagle3-aux-states.patch'],
   'auxiliary hidden states for EAGLE3-style drafters from the GLM-5-Next model: the patch makes glm5next/common/model.py collect the hidden states of chosen layers and return them beside the output',
   'EAGLE3 and DFlash drafters read intermediate target layers, which the GLM model did not expose; this is the model-side tap every multi-layer drafter needs',
   provides=[P('speculation.aux-hidden/-/bf16', 'patch to vllm/models/glm5next/common/model.py', 'n/a', 'target forward with a drafter')],
   tag='speculation.draft', notes='Interface tag: speculation.draft. ' + NOT_RUN),
 F('runtime.vllm_strix.triton_ptr_cache', [CP + 'vsh-triton-ptr-cache.py'],
   'memoises Triton AMD\'s pointer to device-pointer lookup: extractPointer() in triton/backends/amd/driver.c called hipPointerGetAttribute on every launch for every pointer argument, and the patch caches the answer per pointer; validated bit-exact, off',
   'root-caused with gdb on the live worker and measured at ~0.3 % of a step, so it stays off (PATCHES 21); recorded because every Triton launch on ROCm pays it',
   provides=[P('runtime.launch-pointer-cache/-/-', 'patch script over the pinned triton/backends/amd/driver.c', 'n/a', 'every Triton launch')],
   tag=None, notes='Interface tag: none. ' + NOT_RUN),
 F('runtime.vllm_strix.cg_eager_collectives', [CP + 'vsh-cg-eager-collectives.py', SC + 'nccl_graph.py', SC + 'nccl_graph2.py'],
   'CUDA graphs with TP collectives kept out: with VLLM_USE_BREAKABLE_CUDAGRAPH=1 and PIECEWISE mode, every GroupCoordinator collective breaks the capture and runs eagerly; with the two-rank RCCL repros (graphed all-reduce, graphed + eager on two communicators) that found the hang',
   'FULL graphs captured the RCCL-over-OdinLink all-reduces and the first replay hung with both ranks spinning in ncclDevKernel (PATCHES 29); breaking at collectives made graphs work and moved decode to 95-99 ms a step (PATCHES 33)',
   provides=[P('runtime.graph-capture-break-collectives/-/-', 'patch script over vllm/distributed/parallel_state.py; env VSH_CG_EAGER_COLLECTIVES', 'n/a', 'decode, graph sizes 1/2/4/8')],
   tag=None, notes='Interface tag: none. ' + NOT_RUN),
 F('runtime.vllm_strix.shm_spin', [CP + 'vsh-shm-spin.py'],
   'makes the SHM MessageQueue reader\'s busy-spin window tunable: SpinCondition.wait() yields in a loop for busy_loop_s after every read before falling back to the zmq poll, and the hook exposes that window (default stock)',
   'the EngineCore <-> worker control plane spins a full core between steps by default; on a box whose CPU and iGPU share one power budget that spin can cost GPU clock (PATCHES 19, 28.1)',
   provides=[P('runtime.mq-spin-window/-/-', 'patch script over shm_broadcast.py; env', 'n/a', 'control plane')],
   tag=None, notes='Interface tag: none. ' + NOT_RUN),
 F('runtime.vllm_strix.wna16_empty_cache', [CP + 'vsh-wna16-layerwise-empty-cache.patch'],
   'a per-layer torch.cuda.empty_cache() in the WNA16 MoE weight conversion, so each layer\'s transpose staging is released before the next layer loads',
   'on gfx1151 UMA reserved-but-free allocator segments bill against system RAM; 42 layers of dead reservation OOM-killed the worker at load at every config, and the fix let it init at 82.4 GiB (PATCHES 15.4). Any loader on a UMA box needs this discipline',
   provides=[P('format.load-release-staging/-/-', 'patch to compressed_tensors_moe_wna16.py', 'n/a', 'load time')],
   tag=None, notes='Interface tag: none. ' + NOT_RUN),
 # ---------------------------------------------------------------- gfx1151 platform gates
 F('runtime.vllm_strix.aiter_gfx1151_gate', [CP + 'vsh-aiter-gfx1151-gate.patch'],
   'unlocks aiter on gfx1151 in vllm/_aiter_ops.py, where the stock check enabled it only for CDNA',
   'the GLM AMD path calls aiter ops (sparse MLA, kpool, mHC); without the gate they were refused on RDNA 3.5 and the model would not run on Strix Halo',
   provides=[P('runtime.platform-gate/-/-.aiter-gfx1151', 'patch to vllm/_aiter_ops.py', 'n/a', 'gfx1151')], tag=None, notes='Interface tag: none. ' + NOT_RUN),
 F('runtime.vllm_strix.fp8_fnuz_mqa', [CP + 'vsh-fp8-fnuz-mqa.patch', SP + 'aiter/ops/triton/attention/pa_mqa_logits.py'],
   'fp8 fnuz handling for the paged MQA logits on gfx1151: the vLLM side picks the fnuz-aware aiter fp8_mqa_logits, and the overlay pa_mqa_logits.py replaces the base image\'s old-layout aiter file',
   'the base image\'s aiter assumed gfx94x fp8; on gfx1151 the indexer logits read the cache with the wrong fp8 flavour (PATCHES 15.1, 27)',
   provides=[P('attn.indexer-score/fp8-fnuz/f32.paged-mqa', 'patch to rocm_aiter_mla_sparse.py + aiter overlay (Triton)', 'fnuz-aware', 'decode indexer')],
   tag='attn.indexer-score', authors=['AMD aiter contributors (overlay file)', 'davidcanar (vllm-strix-halo)'], notes='Interface tag: attn.indexer-score. ' + NOT_RUN),
 F('runtime.vllm_strix.mhc_no_tilelang', [CP + 'vsh-mhc-no-tilelang-gfx1151.patch'],
   'keeps vllm/model_executor/layers/mhc.py off the TileLang mHC path on gfx1151 (the import is gated to gfx942)',
   'TileLang has no gfx1151 target, so importing it broke DS4 and GLM layers that use manifold hyper-connections; a two-line gate kept until upstream\'s TileLang migration is checked (PATCHES 15.1)',
   provides=[P('runtime.platform-gate/-/-.mhc-tilelang', 'patch to mhc.py', 'n/a', 'gfx1151')], tag=None, notes='Interface tag: none. ' + NOT_RUN),
 # ---------------------------------------------------------------- DeepSeek-V4 native port
 F('runtime.vllm_strix.ds4_native_port', [DS4REF],
   'the DeepSeek-V4-Flash native port to vLLM 0.31 on gfx1151 as reference files: dspark.py and mtp.py from upstream #57568 (DSpark and MTP drafters), compressor.py, sparse_mla.py, sparse_swa.py and rocm_aiter_mla_sparse.py with the rig\'s fixes, the state the vsh-ds4-* patches apply to',
   'DS4 serves correctly on this port since PATCHES 37 and at 24-27 tok/s with DSpark k=5 (PATCHES 41-43), faster than the June delegate stack; the files are the base strixflerp would read for a DS4 TP2 path',
   provides=[P('runtime.model-port/-/-.ds4-vllm031', 'replacement vLLM files (whole)', 'n/a', 'DeepSeek-V4-Flash(-Vision-Exp), TP=2, eager')],
   tag=None, authors=VLLM, notes='Interface tag: none. ' + NOT_RUN),
 F('runtime.vllm_strix.ds4_gfx1151_gates', [CP + p for p in ('vsh-ds4-c4a-decode-layout-gfx1151.patch', 'vsh-ds4-generic-gemm-gfx1151.patch', 'vsh-ds4-no-aiter-fp8-gfx1151.patch', 'vsh-ds4-no-attn-preshuffle-gfx1151.patch', 'vsh-ds4-no-gateup-preshuffle-gfx1151.patch')],
   'five gfx1151 gates for the DS4 AMD path: the C4A block-flat decode layout enabled for gfx1151 (a note: applied inline, no diff body), the MI300-only aiter blockscale GEMM replaced by the generic GEMM, aiter\'s fp8 scaled-mm refused (no fp8 matrix path), and the attention and gate-up B-preshuffles skipped (they exist only for aiter linears)',
   'each gate removes one CDNA assumption from the DS4 model code; together they are the list of what DS4 on RDNA 3.5 must not call',
   provides=[P('runtime.platform-gate/-/-.ds4-gfx1151', 'patches to models/deepseek_v4/amd/{rocm,model}.py and scaled_mm/aiter.py', 'n/a', 'gfx1151')],
   tag=None, notes='Interface tag: none. vsh-ds4-c4a-decode-layout-gfx1151.patch holds only a two-line note; the change is inside ds4-native-fixed-refs/rocm_aiter_mla_sparse.py. ' + NOT_RUN),
 F('runtime.vllm_strix.ds4_decode_inv_rope', [CP + 'vsh-ds4-decode-inv-rope.py'],
   'fixes DeepSeek-V4 decode on gfx1151: rocm_sparse_attn_decode reported every row as inverse-RoPE\'d when given positions while its gfx1151 path never applied the rotation, so the hook makes the standalone rocm_inverse_rope_rows_ pass run',
   'this was the DS4 decode corruption: outputs were rotated attention values (PATCHES 37); the kind of silent mismatch between a fused epilogue and its fallback that strixflerp parity tests exist to catch',
   provides=[P('pos.inverse-rope-rows/-/bf16', 'patch script over models/deepseek_v4/amd/rocm.py', 'exact', 'decode')], tag=None, notes='Interface tag: none. ' + NOT_RUN),
 F('runtime.vllm_strix.ds4_decode_rows', [CP + 'vsh-ds4-decode-rows.py'],
   'explicit cap for the gfx1151 paged-MQA decode logits workspace: (heads, decode_rows, max_model_len) fp32 is 128 MiB a row at 512K, and the hook bounds decode_rows by max_num_seqs x (1 + spec) where the worker cannot read the config',
   'without the cap the 512K context and 2048-token chunks did not fit the 6 GiB KV pin (PATCHES 42)',
   provides=[P('kvcache.workspace-cap/-/-.ds4-decode-logits', 'patch script over the DS4 indexer', 'n/a', '512K context')], tag=None, notes='Interface tag: none. ' + NOT_RUN),
 F('runtime.vllm_strix.ds4_idx_hadamard', [CP + 'vsh-ds4-idx-hadamard-K.patch', CP + 'vsh-ds4-idx-hadamard-K-031.patch', CP + 'vsh-ds4-idx-hadamard-Q-031.patch'],
   'Hadamard rotation for the DS4 indexer\'s fp4 quantisation: K in fused_compress_quant_cache.py and Q in fused_indexer_q.py are rotated before fp32x2 to fp4x2 packing (the -031 pair targets the 0.31 tree, the plain K patch the June tree)',
   'rotation spreads outliers before 4-bit quantisation; the gfx950 path does it in hardware-specific code, and gfx1151 needed it in the Triton kernels to keep indexer selection',
   provides=[P('attn.indexer-quant/fp4/-.hadamard', 'patches to models/deepseek_v4/common/ops/{fused_compress_quant_cache,fused_indexer_q}.py (Triton)', 'Hadamard then e2m1 pack', 'DS4 indexer')],
   tag=None, notes='Interface tag: none. ' + NOT_RUN),
 F('runtime.vllm_strix.ds4_idx_logits_hook', [CP + 'vsh-ds4-idx-logits.py'],
   'install hook that routes DS4 prefill indexer logits from aiter\'s Triton _fp8_mqa_logits_kernel to vsh_idx_logits',
   'the indexer is the quadratic part of DS4 prefill; this hook is the seam that switched it to the WMMA kernel (PATCHES 42)',
   provides=[P('attn.indexer-score-dispatch/fp8-ds4/-', 'patch script over v1/attention/ops/rocm_aiter_mla_sparse.py', 'n/a', 'prefill')],
   tag='attn.indexer-score', requires=['module:kernel.vllm_strix.idx_logits'], notes='Interface tag: attn.indexer-score. ' + NOT_RUN),
 F('runtime.vllm_strix.ds4_mhc_cfg', [CP + 'vsh-ds4-mhc-cfg.py', SC + 'test_mhc_cfg.py', SC + 'test_mhc_small.py', SC + 'mhc_aiter_sweep.py', SC + 'mhc_aiter_unfused.py'],
   'gfx1151 tile configs and a fused / unfused crossover for aiter\'s manifold hyper-connection (mHC) kernels, with the sweeps that found them (fused post+pre GEMM tiles, unfused mhc_post / mhc_pre by split-k) and equality / timing tests',
   'aiter ships mHC tables for gfx950 / gfx942 / gfx1250 only; its fallback ran the fused GEMM at 13-16 GB/s, 2.6 ms a call at a 512-token chunk (PATCHES 42); GLM uses the tables too (PATCHES 44). strixflerp\'s DeepSeek-V4.1 path has the same 4-stream mHC',
   provides=[P('mixer.hyper-connection/-/bf16.gfx1151-tiles', 'patch script installing aiter mHC configs and the crossover (Triton)', 'same outputs as the default config', 'DS4 hc 4, hidden 4096')],
   tag=None, authors=['davidcanar (vllm-strix-halo)'], notes='Interface tag: none. ' + NOT_RUN),
 F('runtime.vllm_strix.ds4_mmf32', [CP + 'vsh-ds4-mmf32.py'],
   'DS4 compressor and indexer-compressor scores (torch.mm(hidden, fused_wkv_wgate.weight.T, out_dtype=float32), bf16 weights) routed to vsh_w8a16.mm_f32, the w16 GEMV for small batches, with a gate-hook diagnostic',
   'hipBLASLt ran these at ~55 GB/s for 6-row decode batches (PATCHES 40); one more case of a bf16-weight fp32-out GEMV',
   provides=[P('linear.decode/bf16/f32.ds4-compressor', 'patch script over models/deepseek_v4/attention.py', 'fp32 accumulate and output', 'decode')],
   tag='linear.decode', requires=['module:runtime.vllm_strix.w8a16_requant'], notes='Interface tag: linear.decode. ' + NOT_RUN),
 F('runtime.vllm_strix.cgfusion_soft', [CP + 'cgfusion-soft.py'],
   'makes the DS4 AMD path\'s CUDA-graph fusion check soft: models/deepseek_v4/amd/rocm.py warns instead of raising when graph fusion is unavailable',
   'a hard failure in the fusion probe blocked DS4 boots on gfx1151 (PATCHES 35); soft-failing kept eager and piecewise modes available',
   provides=[P('runtime.platform-gate/-/-.ds4-cgfusion', 'patch script over models/deepseek_v4/amd/rocm.py', 'n/a', 'boot')], tag=None, notes='Interface tag: none. ' + NOT_RUN),
 # ---------------------------------------------------------------- tuned configs
 F('runtime.vllm_strix.moe_tile_configs', [H + 'moe-configs/'],
   'tuned fused-MoE tile tables for the Radeon 8060S at E=288, N=1024 (int4_w4a16 and the default dtype) under the file names the runtime actually asks for',
   'the first tuned table never loaded because its name did not match what vLLM looks up, and a stale one cost ~6 % (PATCHES 6, 23); tile tables are data a strixflerp MoE recipe would carry the same way',
   provides=[P('moe.tile-config/awq-int4/-', 'E=288,N=1024,device_name=AMD_Radeon_8060S[,dtype=int4_w4a16].json', 'n/a', 'GLM-5.3-Flash TP=2 prefill')],
   tag=None, notes='Interface tag: none. ' + NOT_RUN),
 F('runtime.vllm_strix.aiter_gemm_configs', [SP + 'aiter/ops/triton/configs/gemm/'],
   'gfx1151 tile tables for aiter\'s Triton GEMMs (A8W8, A8W8 blockscale, preshuffled blockscale, batched A8W8 with per-token-group prequant) by N and K, copied into the image\'s aiter config directory',
   'aiter ships these tables for CDNA only and falls back to untuned tiles elsewhere; they are the gfx1151 shapes of DS4 and GLM linears the origin measured',
   provides=[P('linear.tile-config/fp8-a8w8/-', 'gfx1151-GEMM-A8W8*.json, gfx1151-BATCHED_GEMM-*.json', 'n/a', 'per (N, K)')],
   tag=None, authors=['davidcanar (vllm-strix-halo)'], notes='Interface tag: none. ' + NOT_RUN),
 # ---------------------------------------------------------------- OdinLink plumbing on the vLLM side
 F('runtime.vllm_strix.odl_ar2_hook', [CP + 'vsh-odl-ar2-allreduce.patch'],
   'the vLLM hook that sends decode-sized all-reduces in CudaCommunicator.all_reduce to odl_ar2 when DS4_ODL_AR2=1 and the size fits, everything else to RCCL',
   'one branch in the communicator is all it takes to give small collectives their own transport; ported in intent from AlexKGwyn/ds4-vllm and re-anchored on the 73859fec pin (PATCHES 15)',
   provides=[P('collective.allreduce-dispatch/-/-.size-split', 'patch to vllm/distributed/device_communicators/cuda_communicator.py; env DS4_ODL_AR2', 'n/a', 'TP=2')],
   tag='collective.allreduce-p2p', requires=['module:runtime.vllm_strix.odl_ar2'], authors=DS4V, notes='Interface tag: collective.allreduce-p2p (dispatch). ' + NOT_RUN),
 F('runtime.vllm_strix.odl_mq', [SP + 'odl_mq.py'],
   'odl_mq, an OdinLink stream data plane for vLLM\'s MessageQueue remote readers: the writer opens an auto-assigned stream and advertises it, a cross-box reader answers with a 4-byte hello from its own stream before its zmq subscribe, and data then moves as [4-byte length][<= 4000-byte frames] carrying pickle out-of-band buffers; zmq stays for the handshake and is the fallback',
   'the per-step input broadcast (EngineCore box1 to the TP1 worker on box2) and the response queue crossed thunderbolt-net TCP and blocked each decode step; moving them onto streams removed the per-step TCP dependency (PATCHES 15.2). A two-host strixflerp pair has the same control-plane hop',
   provides=[P('transport.message-queue-odl/-/-', 'python odl_mq (ctypes over libodl_tb5.so.0); env ODL_MQ_DEV, ODL_MQ_LIB', 'bytes', 'single cross-box reader a queue; fail-open to zmq')],
   tag=None, requires=['module:runtime.odinlink5.lib'], authors=DS4V, notes='Interface tag: none (transport.stage-relay moves activations, not control messages). ' + NOT_RUN),
 F('runtime.vllm_strix.odl_mq_hook', [CP + 'vsh-odl-mq-dataplane.patch'],
   'the shm_broadcast hook that lets MessageQueue Handles carry an odl_mq stream id and makes remote readers and writers use odl_mq when the device and library are present (DS4_ODL_MQ), fail-open to zmq on any init failure',
   'the data plane only helps if the queue picks it up without breaking same-box readers; the hook declines for a same-host "remote" reader and keeps zmq for it',
   provides=[P('transport.message-queue-dispatch/-/-', 'patch to vllm/distributed/device_communicators/shm_broadcast.py', 'n/a', 'TP=2 across boxes')],
   tag=None, requires=['module:runtime.vllm_strix.odl_mq'], authors=DS4V, notes='Interface tag: none. ' + NOT_RUN),
 F('runtime.vllm_strix.odinlink_local_patch', ['odinlink/odinlink-local.patch', 'container/odinlink/odinlink-local.patch'],
   'five fixes on OdinLink-Five 4534f58 and nothing else: connection restart after a failed DMA verify, verify that survives a peer re-login, the RCCL plugin\'s logger-ABI segfault (a 2-argument call into the 6-argument ncclDebugLogger_t), marking a reassembly buffer bad when a fragment drops, and preallocated RX assembly buffers (rx_asm_max 2 MiB, rx_asm_pool 4) so the drop path stops firing; registers only the tb_protocol_handler fields mainline defines so odl_tb5.ko builds on a stock kernel',
   'upstream grew the RX assembly buffer with kmalloc(GFP_ATOMIC), an order-8 request that fails on a UMA box with weights in RAM, then delivered the short message as complete: the all-reduce returned plausible wrong data and the model answered questions nobody asked (odinlink/README.md). This is the hardening any OdinLink deployment on a pod pair would need',
   provides=[P('transport.odl-hardening/-/-', 'git apply on OdinLink-Five 4534f58 (driver, rccl plugin)', 'n/a', 'one cable, stock kernel')],
   tag=None, requires=['module:runtime.odinlink5.driver_core', 'module:runtime.odinlink5.rccl_net_plugin'],
   notes='Interface tag: none. The two copies are byte-identical (host build and image build). Written against 4534f58; the catalogue\'s OdinLink-Five is 0315225 (the ark copy), where git apply --check passes the driver hunks (with offsets) and fails the rccl/src/odl_tb5_plugin.c hunk, whose context upstream rewrote (gate/results/compile/vllm_strix.side-gates.json); whether 0315225 already carries the logger-ABI fix was not established. ' + NOT_RUN),
 F('runtime.vllm_strix.odl_state', ['odinlink/odl_state.c'],
   'odl-state, the readiness helper: opens /dev/odl_tb5_0 and _1 through libodl_tb5, prints each device\'s peer state and link speed; odinlink.service gates boot on it reaching READY',
   'serving must not start before the cross-host handshake is READY; a 20-line probe on the public API is the gate the systemd unit and the cluster scripts use',
   provides=[P('tool.link-state/-/-', 'odl-state (main)', 'n/a', 'two devices')], tag=None, requires=['module:runtime.odinlink5.lib'], notes='Interface tag: none. ' + NOT_RUN),
 F('tool.vllm_strix.odl_pingpong', ['odinlink/odl_pingpong.c'],
   'RTT latency bench over the OdinLink stream layer: server echoes, client sends [4-byte size][payload in <= ODL_CHUNK frames] on one stream and records CLOCK_MONOTONIC round trips; min / p50 / p90 / p99 / max and the raw RTTs to a file',
   'the numbers that justified odl_ar2 over RCCL for decode came from this bench; the framing mirrors odl_stress and the RCCL plugin, so it measures what the collectives see',
   provides=[P('tool.link-bench/-/-.rtt', 'odl_pingpong server|client <dev> <msg_bytes> <iters> [warmup]', 'timing', 'one stream')], tag=None, requires=['module:runtime.odinlink5.lib'], notes='Interface tag: none. ' + NOT_RUN),
 F('runtime.vllm_strix.nhi_throttle', ['odinlink/nhi-throttle-mod/'],
   'nhi_throttle, a tiny kernel module that rewrites the USB4 / Thunderbolt NHI MSI-X interrupt throttling registers (REG_INT_THROTTLING_RATE 0x38c00 + 4 x vector, 256 ns units) on every active NHI at load (insmod nhi_throttle.ko ns=8000)',
   'mainline hardcodes ~128 us of interrupt moderation on every NHI vector, which was the ~65 us one-way latency floor measured on usb4_rdma; it is the knob below every USB4 transport\'s latency',
   provides=[P('transport.nhi-irq-throttle/-/-', 'kernel module parameter ns', 'n/a', 'AMD / Intel USB4 NHI')],
   tag=None, notes='Interface tag: none. GPL-2.0 per its SPDX line; not built (no kernel tree for the pod kernel in the gate). ' + NOT_RUN, compile='n/a'),
 F('runtime.vllm_strix.odinlink_host_kit', ['odinlink/build-odinlink.sh', 'odinlink/install-odinlink.sh', 'odinlink/odl-swap.sh', 'odinlink/uninstall-tbv.sh', 'odinlink/71-odl-tb5.rules', 'odinlink/systemd/', 'odinlink/README.md'],
   'the OdinLink host kit: fetch OdinLink-Five at the pin and apply the local patch, build odl_tb5.ko against stock kernel headers and libodl_tb5 (no sudo), install driver, library, udev rule, odl-state and odinlink.service, and odl-swap.sh, the load path (MOK signing when a key is enrolled, unload stale thunderbolt_ibverbs, load, wait for READY); plus uninstall-tbv.sh for rigs that ran the old patched-core stack',
   'the transport is only as good as its bring-up: the kit encodes the single-cable rule (mainline gives callbacks no source XDomain, so two links put the peer at the same route) and leaves every module parameter but ring size at default',
   provides=[P('tool.transport-install/-/-.odinlink', 'shell scripts, udev rule, systemd unit', 'n/a', 'Ubuntu-class host, stock kernel, one cable')],
   tag=None, requires=['module:runtime.vllm_strix.odinlink_local_patch'], notes='Interface tag: none. Read, not run. ' + NOT_RUN),
 # ---------------------------------------------------------------- serving
 F('serve.vllm_strix.launcher', ['vllm-strix-halo.sh', H + 'vsh-config', H + 'vsh-config.yaml', H + 'systemd/'],
   'the single entry point: vllm-strix-halo.sh <glm53|ds4> start|stop|status|logs over both boxes, the flat vsh-config.yaml (every key becomes a VSH_* env var through vsh-config) and the systemd units that hold the whole two-box stack',
   'one config file and one verb per model is how the rig switches GLM and DS4 on the same container and port; strixflerp\'s serve recipes need the same single source of truth for a two-host pair',
   provides=[P('serve.cluster-launcher/-/-', 'vllm-strix-halo.sh, vsh-config (python), vsh-config.yaml, vsh-glm.service, vsh-ds4.service', 'n/a', RIG)],
   tag=None, notes='Interface tag: none. ' + NOT_RUN),
 F('serve.vllm_strix.cluster_env', [H + 'vsh-cluster-env.sh', H + 'vsh-cluster-env.odl.sh', H + 'vsh-cluster-env.tcp.sh'],
   'the transport profiles: the canonical NCCL / memory / Ray environment both boxes source, and the odl profile (RCCL over the odl_tb5 net plugin, NCCL_MAX_NCHANNELS=4, DS4_ODL_AR2=1, odl_mq, control bootstrap on thunderbolt0) or the tcp fallback (RCCL sockets on thunderbolt0, DS4_ODL_AR2=0)',
   'a transport is mostly environment; keeping the two profiles side by side made the fallback one config key (PATCHES 15.3)',
   provides=[P('serve.transport-profile/-/-', 'sourced shell', 'n/a', RIG)], tag=None, notes='Interface tag: none. ' + NOT_RUN),
 F('serve.vllm_strix.manual_serve', [H + 'vsh-manual-serve.sh', H + 'vsh-ds4-manual-serve.sh', H + 'vsh-cluster-restart.sh', H + 'vsh-cluster-down.sh', H + 'vsh-ds4-restart.sh', H + 'vsh-ds4-reserve.sh'],
   'the serve and lifecycle scripts: the GLM and DS4 vllm serve launchers run inside the container (every vLLM flag and VSH_* switch in one place), and the restart / teardown / bisect-reserve flows (teardown, Ray on both boxes, serve, verify; box2 driven over ssh)',
   'the flags these launchers pass are the measured configuration of the rig (graph mode, spec method, chunk size, retention interval, KV pin); the teardown order (serve unit, stranded procs, Ray) is what keeps GTT from being stranded',
   provides=[P('serve.vllm-launch/-/-', 'shell', 'n/a', RIG)], tag=None, notes='Interface tag: none. ' + NOT_RUN),
 F('serve.vllm_strix.warmup', [H + 'vsh-warmup.py'],
   'post-boot warm-up: a transient unit started once the API answers fires dummy requests that trigger and cache the JIT compiles, so the first real request does not pay them',
   'for ~2.5 minutes after a boot the warm-up prefill holds the engine (README); without it the first user request ate the Triton compile latency',
   provides=[P('serve.warmup/-/-', 'python, run by vsh-cluster-restart.sh', 'n/a', 'after boot')], tag=None, notes='Interface tag: none. ' + NOT_RUN),
 F('serve.vllm_strix.toolcall_failopen', [CP + 'vsh-toolcall-failopen.py'],
   'tool-call fail-open for the glm47 parser: when a tool call is emitted inside a reasoning span that never closes, the turn returns the text and the call instead of finish_reason=stop with empty content',
   'about three turns in five ended with empty content and up to twelve thousand generated tokens in no field (HANDOVER-tool-calls.md); a serve layer must never swallow output it cannot parse',
   provides=[P('serve.toolcall-parse/-/-.fail-open', 'patch script over the glm47 tool parser', 'n/a', 'OpenAI chat API')], tag='serve.http', notes='Interface tag: serve.http (response shaping). ' + NOT_RUN),
 F('tool.vllm_strix.toolcall_drop_log', [CP + 'vsh-toolcall-drop-log.py'],
   'pure instrumentation for the tool-call parser: writes the full accumulated generation to VSH_TOOLCALL_DROP_DIR when the parser discards text inside an unterminated reasoning span',
   'the fail-open fix was found from these dumps; it is the probe to run whenever a parser and a model disagree',
   provides=[P('tool.parser-drop-log/-/-', 'patch script; env VSH_TOOLCALL_DROP_DIR', 'n/a', 'serve')], tag=None, notes='Interface tag: none. ' + NOT_RUN),
 F('tool.vllm_strix.sync_instr', [CP + 'vsh-sync-instr.py'],
   'measures what the KDA chunk-index host sync costs: instruments prepare_chunk_indices (a .tolist() device to host round trip on the chunked KDA path)',
   'part of the host-sync audit that showed the rig GPU-bound rather than CPU-bound (PATCHES 22)',
   provides=[P('tool.host-sync-probe/-/-', 'patch script', 'timing', 'prefill')], tag=None, notes='Interface tag: none. ' + NOT_RUN),
 F('tool.vllm_strix.gpu_sclk', [H + 'vsh-gpu-sclk.sh'],
   'caps or restores the iGPU maximum SCLK through pp_od_clk_voltage (non-persistent), set from glm53_gpu_sclk_max',
   'on Strix Halo the CPU and iGPU share one budget, and during decode the CPU was clamped to 2.0 GHz; capping the GPU clock at 2100 MHz freed the CPU and raised tok/s (PATCHES 28.1)',
   provides=[P('tool.gpu-clock-cap/-/-', 'vsh-gpu-sclk.sh <MHz|auto>', 'n/a', 'one box')], tag=None, notes='Interface tag: none. Read, not run. ' + NOT_RUN),
 F('tool.vllm_strix.container_ops', [H + 'container-heal.sh', H + 'deploy.sh', 'container/build.sh', 'container/Dockerfile'],
   'image and host plumbing: the two-stage Dockerfile (OdinLink userspace build stage, then vLLM rebuilt at 73859fec on the kyuz0 TheRock gfx1151 base with libodl_ar2 compiled by hipcc and the patches applied), build.sh, the one-time host deployment and container-heal.sh, which reconciles a podman container whose DB says Up while crun is dead',
   'the Dockerfile is the authoritative order in which every patch in this catalogue is applied and every native is compiled (the -O3 --offload-arch=gfx1151 -fPIC line the compile gate reuses)',
   provides=[P('tool.image-build/-/-', 'Dockerfile, shell', 'n/a', 'podman, both boxes')], tag=None, notes='Interface tag: none. Read, not run. ' + NOT_RUN),
 # ---------------------------------------------------------------- benches and probes
 F('tool.vllm_strix.decode_bench', [SC + p for p in ('vsh_bench.py', 'vsh_ab.py', 'stepbench.py', 'ttft.py', 'decode_at_ctx.py', 'dflash_bench.sh', 'glm_regress.sh', 'steady.py')],
   'the rig\'s decode benches: vsh_bench (decode / TTFT with speculative-decoding receipts), vsh_ab (matched-length A/B for drafters, MTP vs DFlash2), stepbench (unprofiled step time from SSE chunk timing), TTFT with unique prompts, decode deep in a long context, the DFlash2 acceptance curve, the GLM regression battery and a CPU-frequency recorder',
   'tok/s moves with draft acceptance, so the rig measures at matched lengths and per workload (JSON / tools / prose); these are the receipts behind the README tables (PATCHES 24, 25, 44)',
   provides=[P('tool.decode-bench/-/-.spec-receipts', 'python and shell against the OpenAI endpoint', 'timing', 'GLM-5.3 and DS4 serve')], tag='tool.prefill-bench', notes='Interface tag: tool.prefill-bench (the catalogue\'s nearest bench class). ' + NOT_RUN),
 F('tool.vllm_strix.context_probes', [SC + p for p in ('midneedle.py', 'needlepos.py', 'ds4ctx.py', 'ds4ctxwatch.sh', 'ds4len.py')],
   'long-context retrieval probes: three watchwords at 25 / 50 / 75 % of a filler prompt (GLM and DS4), the pool ids that hold them, a context sweep with a MemAvailable watcher on both boxes, and a length / position battery',
   'mid-context needles exposed the four stacked indexer bugs that tail needles missed (PATCHES 27); the probes are the acceptance test for any sparse-attention change',
   provides=[P('tool.context-probe/-/-', 'python, shell', 'pass / fail per needle', '6.8K-512K')], tag=None, notes='Interface tag: none. ' + NOT_RUN),
 F('tool.vllm_strix.quality_probes', [SC + 'nll.py', SC + 'data/nll_agents.txt', SC + 'data/nll_patches.txt', SC + 'data/nll_pyfile.txt', SC + 'greedy_repro.py', SC + 'drift.py', SC + 'effort_test.py', SC + 'loop_probe.py'],
   'quality A/B across builds: mean per-token NLL of fixed texts via prompt_logprobs (with the three texts), greedy reproducibility over N identical requests with distinct cache salts, output drift, the reasoning-effort template check and the degenerate-loop probe',
   'every kernel swap here was accepted on NLL and greedy equality, not only on speed; greedy is reproducible only below ~2K prompt tokens on this stack (README), which these probes measure',
   provides=[P('tool.quality-ab/-/-', 'python', 'NLL, byte equality', 'serve')], tag=None, notes='Interface tag: none. ' + NOT_RUN),
 F('tool.vllm_strix.toolcall_probes', [SC + p for p in ('replay.py', 'replay_n.py', 'tool_count.py', 'proxy.py', 'render_prompt.py')],
   'tool-calling probes: verbatim replay of a captured opencode request, the rate of early stops without a tool call over N replays, the dependence on the number of tools offered, a transparent logging proxy and the rendered chat template',
   'the 8/8 tool-calling result and the fail-open fix came from replaying real client traffic rather than synthetic prompts (PATCHES 26, 27.1)',
   provides=[P('tool.toolcall-replay/-/-', 'python', 'pass / fail', 'serve')], tag=None, notes='Interface tag: none. ' + NOT_RUN),
 F('tool.vllm_strix.trace_analysis', [SC + p for p in ('trace_dtoh.py', 'trace_gap_analyze.py', 'trace_share.py', 'trace_sync.py', 'ds4trace.py', 'ds4-trace-agg.py', 'ds4who.py', 'ds4prefill.py', 'vsh_kpl_hook.py', 'vsh_kpl_hook4.py')],
   'torch-profiler trace analysis: device to host copies (pipeline drains), where a decode step\'s wall time goes, kernel-class share per step / token, blocking host ops, DS4 decode and prefill GPU time by kernel family, which host op launched a kernel (correlation ids), and two live hooks for indexer selection',
   'every PATCHES section starts from one of these readings (step wall vs GPU busy, top kernels, host self-time); they turn a trace into the one number that decides the next change',
   provides=[P('tool.trace-analysis/-/-', 'python over chrome-trace JSON', 'n/a', 'offline')], tag=None, notes='Interface tag: none. ' + NOT_RUN),
 F('tool.vllm_strix.ds4_probes', [SC + p for p in ('ds4probe.py', 'ds4speed.py', 'ds4speed_sampled.py', 'ds4image.py', 'ds4-gen.py', 'ds4_render_prompt.py')],
   'DeepSeek-V4 bring-up probes: the Phase-0 battery, decode speed and coherence (greedy and server-default sampling), image-input probes on synthetic images, a counting request and the rendered prompt with thinking on and off',
   'the DS4 port was declared correct and fast on these probes (PATCHES 35-43); the same battery is a ready acceptance set for a strixflerp DS4 path',
   provides=[P('tool.model-probe/-/-.ds4', 'python against the endpoint', 'pass / fail, tok/s', 'DS4 serve')], tag=None, notes='Interface tag: none. ' + NOT_RUN),
 F('tool.vllm_strix.debug_hooks', [CP + 'debug/'],
   'debug-only patches kept by the origin: dumps of GLM indexer gather / plan / kpool / map / score / top-k (whole-file sparse_indexer and ops replacements), block-table expansion experiments, live indexer-selection diagnostics, a HIP API cost probe, the cgfusion diagnostic and an early int4 MoE GEMV with its test',
   'these hooks localised the long-context bugs one stage at a time (PATCHES 27); they show where to tap a sparse indexer to see what it selected',
   provides=[P('tool.debug-hooks/-/-', 'python patch scripts and replacement files (inert by default)', 'n/a', 'debug')], tag=None, authors=VLLM, notes='Interface tag: none. ' + NOT_RUN),
]

LEFT_OUT = {
    'container/pinned-vllm/, container/pinned-triton/': 'vLLM and Triton files with the patches applied, kept by the origin for byte-exact rebuilds: upstream code, not this repo\'s (the vsh_* files inside are byte-identical to container/patches copies, harvested there)',
    'container/patches/libvsh_*.so': 'prebuilt gfx1151 libraries of vsh_moe_int4.hip and vsh_w8a16.hip; the sources are harvested and compiled by the gate',
    'scripts/data/ds4_dspark_routing.pt': 'binary routing dump of real DS4 decode (input to tool.vllm_strix.mxfp4_sweeps)',
    'scripts/download-models.sh': 'downloads model weights (the catalogue holds no weights)',
    'host/__pycache__/, *.pre-ds4native, container/Dockerfile.pre-ds4fnuz': 'bytecode and pre-migration backups',
    'README.md, PATCHES.md, PENDINGWORK.md, AGENTS.md, FIXSPEC.md, FRESH-EYES-20tps.md, HANDOFF-*.md, HANDOVER-tool-calls.md, host/PATCHES.md': 'documentation and history; read for the module records and ORIGIN.toml',
    'LICENSE, THIRD_PARTY_NOTICES.md': 'copied beside ORIGIN.toml',
    '.gitignore': 'repository plumbing',
}

# Licence texts the harvested files name but the tree does not ship: {catalog path: repo-relative canonical copy}.
CANONICAL = {'LICENSES/GPL-2.0': 'docs/harvest/data/licences/linux-LICENSES-preferred-GPL-2.0',
             'LICENSES/aiter-MIT.txt': 'docs/harvest/data/licences/aiter-775b7a1-LICENSE'}

ORIGIN_TOML = '''
# vllm-strix-halo: vLLM serving GLM-5.3-Flash and DeepSeek-V4-Flash tensor-parallel (TP=2) over two AMD Strix Halo
# boxes (gfx1151) joined by one Thunderbolt / USB4 cable through the OdinLink driver, with gfx1151 kernels, DFlash2 /
# DSpark speculative decoding and the patches that made long context correct (card ORIGIN-VLLM-STRIX d2a7b91dec).
name      = "vllm_strix"
repo      = "https://github.com/davidcanar/vllm-strix-halo"
commit    = "fe5ac1016737c1f68e04772e3ae9e7567bc40f22"
branch    = "main"
version   = "main HEAD 2026-10-06 (README status 2026-10-06, PATCHES sections 1-45); vLLM pinned at 73859fec (0.31.0.dev0) on docker.io/kyuz0/vllm-therock-gfx1151:rocm10.0.0-torch2.11.0-vllm0.30.0; OdinLink-Five pinned at 4534f58 + odinlink/odinlink-local.patch"
reference_checkout = "control-node:/data/ark-inbox/src/github.com__davidcanar__vllm-strix-halo#fe5ac1016737c1f68e04772e3ae9e7567bc40f22 (ark copy: the tree at the commit with a shallow .git, ark card a200ae59e2); laptop: ~/Code/strixflerp-upstream-supplement/vllm_strix (the same 329 files, by tarball through control-node get); docs/harvest/harvest_vllm_strix.py pins the tree digest"

[licence]
spdx      = "Apache-2.0"
files     = ["LICENSE", "THIRD_PARTY_NOTICES.md"]        # copied verbatim next to this file
# Licences the harvested files name that the tree ships no text for, copied verbatim from canonical copies
# (docs/harvest/harvest_vllm_strix.py CANONICAL; sources in docs/harvest/data/licences/SOURCES.tsv)
canonical_files = ["LICENSES/GPL-2.0", "LICENSES/aiter-MIT.txt"]
canonical_source = "LICENSES/GPL-2.0: the Linux kernel's LICENSES/preferred/GPL-2.0 (for odinlink/nhi-throttle-mod/nhi_throttle.c, SPDX GPL-2.0); LICENSES/aiter-MIT.txt: ROCm/aiter's LICENSE at 775b7a1 (for the aiter overlay pa_mqa_logits.py, SPDX MIT); both from control-node ark copies, docs/harvest/data/licences/SOURCES.tsv"
third_party = "LICENSE: Apache-2.0. THIRD_PARTY_NOTICES.md: vLLM (Apache-2.0; the 55222-*, 57979-*, 59412-*, upstream-window-*, vsh-idx-*-indexer / ops and ds4-native-fixed-refs files are vLLM files or ports and keep its SPDX headers; 57979-kda-* also carry flash-linear-attention's Songlin Yang, Yu Zhang); AlexKGwyn/ds4-vllm (Apache-2.0: the odl_ar2 / odl_mq design and the host orchestration pattern); the aiter overlay pa_mqa_logits.py is AMD aiter's (MIT by its SPDX line; Copyright (C) 2024-2026, Advanced Micro Devices, Inc.; aiter's LICENSE is LICENSES/aiter-MIT.txt); odinlink/nhi-throttle-mod/nhi_throttle.c is GPL-2.0 by its SPDX line. Model weights are not part of the tree."

[toolchain]
host      = "python 3.12 venv inside the vllm-strix-halo image (vLLM 0.31.0.dev0+git73859fec rebuilt from source, PyTorch 2.11, Triton 3.8, aiter, ROCm 10.0 TheRock); podman on Ubuntu hosts"
device    = "therock-10.2"                               # the gate's toolchain; the origin compiles with the venv's hipcc (ROCm 10.0)
build     = "container/build.sh -> container/Dockerfile: stage odinlink-build builds OdinLink-Five 4534f58 + odinlink-local.patch userspace (libodl_tb5, librccl_net_odl_tb5); the final stage rebuilds vLLM at 73859fec, applies container/patches in order, copies pinned-vllm / pinned-triton and the site-packages overlay, and compiles odl_ar2.hip with /opt/venv/bin/hipcc -O3 --offload-arch=gfx1151 -fPIC -shared -lodl_tb5; libvsh_moe_int4.so and libvsh_w8a16.so are prebuilt and copied"

[compile]                                                # hipcc as the Dockerfile calls it: clang++ -x hip, one object a unit
driver    = "<rocm>/lib/llvm/bin/clang++"
arch_flag = "--offload-arch=gfx1151"
rocm_flag = "--rocm-path=<rocm>"
flags     = ["-x", "hip", "-O3", "-fPIC", "-c"]
system_includes = ["include"]
output    = "host object with the gfx1151 device code embedded (.o), one per translation unit"
gate      = "gate/compile/run.py --origin vllm_strix on control-node's CPU, nice 19, therock-10.2, 2026-11-25: 6 of 6 standalone HIP modules and 6 of 6 units compile for gfx1151 (gate/results/compile/<module>.json; log /data/pod/plans/d2a7b91dec/compile-gate2.log). python3 docs/harvest/harvest_vllm_strix.py --side-gates: 144 of 144 .py files byte-compile; 19 of 20 .patch / .diff parse with git apply --numstat (vsh-ds4-c4a-decode-layout-gfx1151.patch is a two-line note with no diff); odl_state.c and odl_pingpong.c compile with gcc against runtime.odinlink5.lib; odinlink-local.patch (written for OdinLink-Five 4534f58) does not apply to the catalogue's 0315225: its driver hunks apply with offsets, the rccl/src/odl_tb5_plugin.c hunk fails where upstream rewrote the plugin-loaded log lines (gate/results/compile/vllm_strix.side-gates.json)"

[kernels]
loading   = "HIP: shared libraries loaded with ctypes (libvsh_moe_int4.so, libvsh_w8a16.so, libodl_ar2.so); Triton: JIT at serving time inside vLLM; patches installed at image build"
selection = "environment switches (VSH_*, DS4_ODL_AR2, DS4_ODL_MQ) and batch-size crossovers inside each hook: decode-sized calls (M <= 8 rows, <= 64 token-expert pairs) go to the HIP kernels, the rest stay on vLLM's paths"
models    = "GLM-5.3-Flash AWQ W4A16 (wtdcode/GLM-5.3-Flash-AWQ-W4A16) with the DFlash2 drafter k=3; DeepSeek-V4-Flash-Vision-Exp (MXFP4 experts, block-FP8 linears) with DSpark k=5"

[numbers]                                                # the origin's README and PATCHES, 2026-10-06; not measured here
glm53     = "decode 23-27 / 25-29 / 19.6-19.9 tok/s JSON / tools / prose (DFlash2 k=3, CUDA graphs PIECEWISE, 95-99 ms a step; 12.0 before PATCHES 28); prefill 278-294 tok/s cold at 5.8K-19.9K; TTFT 13.9K ~50 s cold, 0.95 s cached; 512K context, KV pool 1,263,415 tokens; needles 3/3 to 128K, tail needles to 255K; tools 8/8"
ds4       = "decode 24.4 / 24.7 tok/s prose / JSON with default sampling (DSpark k=5, eager), 15.4 spec off; prefill ~460 tok/s at 8K, 331 at 128K; 512K context, KV pool 901,584 tokens; TTFT 6.6 min at 128K"
transport = "OdinLink (odl_tb5) over one TB4 / USB4 cable: RCCL via the odl net plugin for prefill collectives (NCCL_MAX_NCHANNELS=4), odl_ar2 for decode all-reduces, odl_mq for the EngineCore <-> worker data plane; tcp fallback profile"
'''


def lang_of(f):
    if f['lang']:
        return f['lang']
    files = O.expand(f['paths'], f['exclude'])
    if f['sources']:
        return 'c' if all(s.endswith('.c') for s in f['sources']) else 'hip-c++'
    if any(p.endswith(('.c', '.h')) for p in files):
        return 'c'
    code = [p for p in files if not p.endswith(('.md', '.txt', '.json', '.rules', '.service', '.yaml'))]
    if code and all(p.endswith(PATCHLIKE) for p in code):
        return 'patch'
    if code and all(p.endswith('.sh') or os.path.basename(p) in ('Dockerfile', 'Makefile') or '/systemd/' in p for p in code):
        return 'shell'
    if any(p.endswith('.py') for p in files) or any(p.endswith('.json') for p in files):
        return 'python'
    return 'shell'


GATE_NOTE = {
    'python': 'python: not compiled; python3 docs/harvest/harvest_vllm_strix.py --side-gates byte-compiles every .py on control-node (gate/results/compile/vllm_strix.side-gates.json)',
    'patch': 'patch: applies to vLLM 73859fec or OdinLink-Five 4534f58, neither of which is on the island; --side-gates parses each with git apply --numstat (gate/results/compile/vllm_strix.side-gates.json)',
    'shell': 'shell: scripts and units, read and not run (host plumbing for the two-box rig)',
    'c': 'c: compiled by --side-gates with gcc against runtime.odinlink5.lib on control-node CPU (gate/results/compile/<module>.json)',
}


def mods():
    out = []
    for f in FAMILIES:
        lang = lang_of(f)
        hip = lang == 'hip-c++'
        comp = f['compile'] or ('standalone' if hip else 'n/a')
        reason = None if hip else GATE_NOTE[lang]
        if f['id'] == 'runtime.vllm_strix.nhi_throttle':
            reason = 'linux kernel module: not built (needs the target kernel\'s headers; the control-node kernel is not the rig\'s)'
        out.append(M(f['id'], f['summary'], f['paths'], provides=f['provides'], requires=f['requires'], lang=lang,
                     sources=f['sources'] if lang in ('hip-c++',) else (), includes=['src'] if hip else [],
                     toolchain='therock-10.2' if hip else '', authors=f['authors'], notes=f['notes'], why=f['why'],
                     compile=comp, reason=reason, exclude=f['exclude']))
    return out


def modules_md():
    """catalog/vllm_strix/MODULES.md: each module with its lang, files, lines, provides and interface tag."""
    bt = '\x60'
    o = ['# vllm-strix-halo modules by strixflerp interface', '',
         'One row for each module of vllm-strix-halo at fe5ac10. Card ORIGIN-VLLM-STRIX d2a7b91dec. Written by',
         bt + 'docs/harvest/harvest_vllm_strix.py' + bt + ' from its ' + bt + 'FAMILIES' + bt + ' table.', '',
         '- **tag**: the strixflerp interface class the module matches (a class some catalogue module already provides), or',
         '  none. The module\'s ' + bt + '[[provides]]' + bt + ' names what it would be called.',
         '- The transport under the rig (OdinLink-Five: driver, library, RCCL / NCCL net plugins) is its own origin,',
         '  ' + bt + 'catalog/odinlink5' + bt + '.', '',
         '| module | lang | files | lines | provides | tag |', '|---|---|---:|---:|---|---|']
    tot_f = tot_l = 0
    tags = 0
    for f in FAMILIES:
        files = O.expand(f['paths'], f['exclude'])
        lines = sum(len(open(O.find(p), errors='replace').read().splitlines()) for p in files)
        tot_f += len(files)
        tot_l += lines
        tags += 1 if f['tag'] else 0
        o.append('| %s%s%s | %s | %d | %d | %s | %s |' % (bt, f['id'], bt, lang_of(f), len(files), lines,
                 ', '.join(bt + p['interface'] + bt for p in f['provides']), (bt + f['tag'] + bt) if f['tag'] else 'none'))
    o += ['', f'{len(FAMILIES)} modules, {tot_f} files, {tot_l} lines; {tags} carry a tag, {len(FAMILIES) - tags} are none.', '',
          'Not harvested (ORIGIN.toml [harvest] unharvested lists every file):', '']
    o += [f'- {bt}{k}{bt}: {v}' for k, v in LEFT_OUT.items()]
    open(os.path.join(harvestlib.REPO, 'catalog', O.name, 'MODULES.md'), 'w').write('\n'.join(o) + '\n')


def sh(cmd, cwd=None, env=None, inp=None):
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, input=inp)
    return p.returncode, (p.stdout + p.stderr)[-2000:]


def side_gates():
    """control-node CPU: py_compile every .py, git apply --numstat every .patch / .diff, gcc the two OdinLink C helpers
    against runtime.odinlink5.lib, and git apply --check of odinlink-local.patch on the catalogue's OdinLink-Five."""
    root = os.path.join(harvestlib.REPO, 'catalog', O.name)
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    py, pt = [], []
    for f in FAMILIES:
        kind, _, name = f['id'].split('.', 2)
        src = os.path.join(root, kind, name, 'src')
        for d, _, fs in os.walk(src):
            for n in sorted(fs):
                p = os.path.join(d, n)
                rel = os.path.relpath(p, src)
                if n.endswith('.py'):
                    try:
                        compile(open(p, 'rb').read(), rel, 'exec', dont_inherit=True)
                        py.append(dict(module=f['id'], file=rel, ok=True))
                    except SyntaxError as e:
                        py.append(dict(module=f['id'], file=rel, ok=False, error=f'{e.msg} line {e.lineno}'))
                elif n.endswith(PATCHLIKE):
                    rc, out = sh(['git', 'apply', '--numstat', p], cwd=tempfile.gettempdir())
                    pt.append(dict(module=f['id'], file=rel, rc=rc, numstat=out.strip().splitlines()[:12]))
    # the two C helpers against the catalogue's OdinLink-Five library headers
    libinc = os.path.join(harvestlib.REPO, 'catalog', 'odinlink5', 'runtime', 'lib', 'src', 'lib', 'include')
    cres = {}
    cc = os.environ.get('CC', 'gcc')
    rc_v, ccv = sh([cc, '--version'])
    for mid, rel in (('runtime.vllm_strix.odl_state', 'odinlink/odl_state.c'), ('tool.vllm_strix.odl_pingpong', 'odinlink/odl_pingpong.c')):
        kind, _, name = mid.split('.', 2)
        srcf = os.path.join(root, kind, name, 'src', rel)
        tmp = tempfile.mkdtemp(prefix='vsh-c-')
        cmd = [cc, '-std=gnu11', '-O2', '-Wall', '-c', '-I', libinc, srcf, '-o', os.path.join(tmp, 'x.o')]
        t0 = datetime.datetime.now()
        rc, out = sh(cmd)
        ok = rc == 0 and os.path.getsize(os.path.join(tmp, 'x.o')) > 0
        res = dict(module=mid, gate='compile-c', arch='x86_64 (host code)', host=os.uname().nodename, date=now,
                   toolchain=ccv.splitlines()[0] if ccv else cc, units=[dict(unit=rel, cmd=' '.join(cmd).replace(harvestlib.REPO + '/', ''), rc=rc,
                   seconds=round((datetime.datetime.now() - t0).total_seconds(), 2), tail=out[-800:])],
                   status='pass' if ok else 'fail')
        if not ok:
            res['reason'] = out[-300:]
        shutil.rmtree(tmp, ignore_errors=True)
        json.dump(res, open(os.path.join(harvestlib.REPO, 'gate', 'results', 'compile', mid + '.json'), 'w'), indent=1)
        open(os.path.join(harvestlib.REPO, 'gate', 'results', 'compile', mid + '.json'), 'a').write('\n')
        cres[mid] = res['status']
    # does odinlink-local.patch (written for 4534f58) still apply to the catalogue's OdinLink-Five 0315225?
    apply = None
    if os.path.isdir(ODL5):
        tmp = tempfile.mkdtemp(prefix='vsh-odl-')
        shutil.copytree(ODL5, os.path.join(tmp, 't'), ignore=shutil.ignore_patterns('.git'))
        lp = os.path.join(root, 'runtime', 'odinlink_local_patch', 'src', 'odinlink', 'odinlink-local.patch')
        rc, out = sh(['git', 'apply', '--check', '-v', lp], cwd=os.path.join(tmp, 't'))
        apply = dict(target='OdinLink-Five 0315225 (' + ODL5 + ')', rc=rc, tail=out[-1500:])
        shutil.rmtree(tmp, ignore_errors=True)
    res = dict(gate='side-gates', origin=O.name, host=os.uname().nodename, date=now,
               python=dict(files=len(py), ok=sum(1 for r in py if r['ok']), failed=[r for r in py if not r['ok']]),
               patches=dict(files=len(pt), parsed=sum(1 for r in pt if r['rc'] == 0), results=pt),
               c_helpers=cres, odinlink_local_patch_on_0315225=apply)
    out = os.path.join(harvestlib.REPO, 'gate', 'results', 'compile', 'vllm_strix.side-gates.json')
    json.dump(res, open(out, 'w'), indent=1)
    open(out, 'a').write('\n')
    print(f"side gates: python {res['python']['ok']} of {res['python']['files']} byte-compile; patches {res['patches']['parsed']} of "
          f"{res['patches']['files']} parse; C helpers {cres}; odinlink-local.patch on 0315225: "
          f"{'applies' if apply and apply['rc'] == 0 else 'does not apply' if apply else 'not checked'}")
    return 0


if __name__ == '__main__':
    if '--side-gates' in sys.argv:
        sys.exit(side_gates())
    run(O, mods(), ORIGIN_TOML, LEFT_OUT, ['LICENSE', 'THIRD_PARTY_NOTICES.md'], CANONICAL)
    modules_md()
