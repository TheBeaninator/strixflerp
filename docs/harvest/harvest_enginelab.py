#!/usr/bin/env python3
"""Harvest the Strix Halo engine lab (hyukudan/strix-halo-engine-lab at 0598cf6dc030) into catalog/enginelab, then its
companion USB4 kernel patch series into catalog/enginelab_usb4 (docs/harvest/harvest_enginelab_usb4.py).
Run: python3 docs/harvest/harvest_enginelab.py            (both origins)
     python3 docs/harvest/harvest_enginelab.py --classify (print the family table and any unassigned file, write nothing)

Card ORIGIN-ENGINELAB bc74e1a17d. The engine lab is a C++20 / HIP inference engine written for Strix Halo (gfx1151,
wave32, 128 GB UMA) that serves Qwen3.8-Flash-Next, DeepSeek V4.1 Flash, MiniMax M3.1 Flash and GLM-5.3 Flash on one
node or on two nodes joined by USB4, with optional CUDA routes for an RTX PRO 6000. Its claims are an EXACT profile
(bitwise its declared reference), exact two-rank tensor-parallel decode, MTP and wide copy-verify drafting, DSpark
drafting, WMMA prefill and a USB4 transport with a GPU-direct DMA-BUF path (README.md, measured 2026-11-21).

Scope: every file under src/, include/, apps/, lab/ and third_party/, split into families (FAMILIES below; the more
modules the merrier: every kernel, transport, scheduler, spec-decode and serve piece is its own module). A file goes to
the first family whose pattern matches its origin path; the script exits if a file under those roots matches none.
tests/, scripts/, tools/, docs/, cmake/, packaging/, models/, data/ and the rest are listed as unharvested. Each module
is tagged with the strixflerp interface class it matches, or none; its [[provides]] then names the shape it would carry.

The reference checkout is the ark copy on control-node
(/data/ark-inbox/src/github.com__hyukudan__strix-halo-engine-lab#0598cf6dc030260c633482a4ef4b38fdd9d5502b, a tree without
git objects); the laptop holds the same files, moved by a tarball through control-node get, in
~/Code/strixflerp-upstream-supplement/enginelab. ENGINELAB_CHECKOUT names another copy. The tree digest pins the bytes."""
import hashlib, os, re, subprocess, sys, tomllib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvestlib
from harvestlib import Origin, M, P, run

harvestlib.C_FAMILY = tuple(harvestlib.C_FAMILY) + ('.cu', '.cuh', '.inl')

CHECKOUT = os.environ.get('ENGINELAB_CHECKOUT', os.path.join(harvestlib.SUPPLEMENT, 'enginelab'))
COMMIT = '0598cf6dc030260c633482a4ef4b38fdd9d5502b'
# sha256 over the lines "<sha256 of the file>  <origin-relative path>\n" of every file, sorted by path (.git left out)
TREE_SHA256 = '99bc296df6a8cc9ef1ba63e40d88c35cc1c34fe872c9cc077d0ce4b5ea12b0ff'
SKIP = ('.git', 'build')
ROOTS = ('src/', 'include/', 'apps/', 'lab/', 'third_party/')
# tests/ headers that harvested sources #include (the rest of tests/ is unharvested)
TEST_HOOKS = ('tests/hip_qwen_rotary_watchdog.hpp', 'tests/hip_qwen_rotary_fixtures.inc', 'tests/hip_mtp_model_qualification.hpp')


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
    """The origin's file set is the checkout's files (a git tree export without objects), pinned by digest."""
    def __init__(self, *a, checkout, commit, digest, **k):
        super().__init__(*a, **k)
        self.roots, self._checkout = [checkout], checkout
        head = os.path.join(checkout, '.git', 'HEAD')
        if os.path.isfile(head) and open(head).read().strip() != commit:
            raise SystemExit(f'{head} names {open(head).read().strip()}, the harvest pins {commit}')
        got = tree_digest(checkout, self.files())
        if digest and got != digest:
            raise SystemExit(f'{checkout}: tree digest {got}, the harvest pins {digest} ({commit[:12]})')
        self.digest = got

    def files(self):
        if self._files is None:
            self._files = tree_files(self._checkout)
        return self._files


# ---------------------------------------------------------------------------------------------- family patterns
def r(*stems):
    """Files directly in src/ or include/strix/ whose name (less a hip_ prefix) starts with one of the stems."""
    return [rf'(src|include/strix)/(hip_)?{s}[^/]*' for s in stems]


def d(*dirs):
    """Whole subtrees src/<dir>/ and include/strix/<dir>/."""
    return [rf'(src|include/strix)/{x}/.*' for x in dirs]


def F(kind, name, pats, summary, why, iface, api, numeric='see the file headers', regime='see the file headers', tag=False, requires=()):
    """One family. iface: the interface it provides. tag=True: iface's class (less the .enginelab tag) is a class the
    catalogue already has, so the module is tagged with it; False: no catalogue interface matches (tag none) and iface is
    the name the shape would carry."""
    return dict(kind=kind, name=name, pats=[re.compile(p) for p in pats], summary=summary, why=why, iface=iface, api=api,
                numeric=numeric, regime=regime, tag=tag, requires=list(requires))


EXACT = 'EXACT profile: bitwise the engine\'s declared reference (fixed fmaf order, -ffp-contract=off, denormals kept); FAST levers change bits only behind the quality gate (AGENTS.md)'

SC = 'src/core/'
FAMILIES = [
 # ------------------------------------------------------------------ shared kernel core (src/core), AGENTS.md "shared homes"
 F('kernel', 'core_tiled_wmma', [SC + r'(wmma_tiled|tiled_core|f16_tile_gemm|a4_tiles|v14_core_policy|vgpr_budget|wave_uniform|lds_rows|NOTICE)[^/]*'],
   'the shared tiled prefill core adopted from strix-trellis (namespace strix::core): the register-blocked, double-buffered WMMA GEMM for gfx1151 (wmma_tiled.hip: a block owns BM x BN outputs, decodes each weight tile once into LDS, every wave reuses its A / B fragments; v_wmma_f32_16x16x16_f16 with ascending K16 order from +0), the FP16 tile GEMM and its small-M form, the A4 tiles, the v14 core policy, the VGPR budget and wave-uniform helpers, and src/core/NOTICE.md',
   'this is the WMMA prefill engine every model of the lab shares at 8-32K prompts: the decomposition changes but the per-output K16 order does not, so it keeps the EXACT bits of the plain WMMA core; strixflerp\'s gemm.wmma-tile providers are the slot it would fill',
   'gemm.wmma-tile/trellis+iu4+f16/f16-f32.enginelab', 'strix::core wmma tiled launchers (src/core/wmma_tiled.hpp), f16_tile_gemm*.hpp', 'FP16 operands, FP32 accumulation in ascending K16 tiles; PackedIU4 FAST v1 rescales I32 sums at K64 boundaries (opt-in)', 'prefill, M >= 16 rows; tile 128 / 256 rows, rasterisation group 2 (STRIX_TRELLIS_WMMA_RASTER)', tag=True),
 F('kernel', 'core_trellis', [SC + r'(trellis_|exl3_rows)[^/]*'],
   'the one EXL3 trellis weight codec of the engine (trellis_format.hpp, trellis_codec.hpp: ring, codebook and reader primitives from strix-trellis, gathered without changing their arithmetic) and the EXL3 row plan',
   'DeepSeek V4.1, GLM and MiniMax keep K4 / K5 trellis weights; this is the decode the dense and MoE trellis kernels inline, next to catalog/strix_trellis\'s own copy',
   'linear.decode/exl3-trellis/f16-f32.enginelab', 'strix::core trellis codec headers', 'decoded FP16 weights, bit-exact with the strix-trellis reference', 'decode and prefill, K2-K5 trellis', tag=True),
 F('kernel', 'core_iu4', [SC + r'(iu4|i4_shadow|w4a16|f16_dense_iu4|dense_iu4|dense_linear_iu4)[^/]*'],
   'stored INT4 (IU4) support: the IU4 pack builder and plans, IU4 x A4 dense projections with their layout and rotation, IU4 row plans and x-pairs, on-the-fly IU4, the INT4 shadow and the small-row W4A16 decoder with streaming stages and shared dequantisation (adopted from strix-trellis, extended here)',
   'FAST_I4 is the lab\'s lossy 4-bit identity for GLM / MiniMax / DS41 dense weights; the W4A16 streaming decoder is the decode GEMV shape for those packs',
   'linear.decode/iu4/f16-f32.enginelab', 'strix::core iu4*.hpp, w4a16.hpp', 'INT4 weights with group scales; A4 or FP16 activations; FAST identity (changes bits, gated)', 'decode 1-8 rows; prefill tiles through core_tiled_wmma', tag=True),
 F('kernel', 'core_integer', [SC + r'(integer_|moe_i4|chunk_contract)[^/]*'],
   'the integer operand for DeepSeek V4.1 GGUF blocks (integer_operand.hpp, integer_math.hpp, the plain / tall / wide / grouped integer main loops and the grouped schedule) and the MoE INT4 arithmetic and chain plans',
   'an exact integer-dot path over GGUF k-quant blocks shared by the dense and the grouped MoE kernels; the lab\'s local extension of the adopted core',
   'linear.decode/gguf-kquant/a8-f32.enginelab-integer', 'strix::core integer_*.hpp / *.inc main loops', 'integer dot products per block, FP32 scale application in a fixed order', 'decode and small-batch verify windows', tag=True),
 F('kernel', 'core_bf16', [SC + r'(bf16_|fma_bf16_rows|tall_bf16_)[^/]*'],
   'BF16 kernels of the shared core: dense row plans and narrow plans, the BF16 LM head, the BF16 projection batch, the BF16 reference, the FMA BF16 rows and the tall BF16 plan and reference',
   'routers, hyper-connection projections and heads stay BF16 in every model; these are the shared BF16 GEMV / tall forms with host references for the EXACT gate',
   'linear.dense/bf16/bf16-f32.enginelab', 'strix::core bf16_*.hpp', 'BF16 weights, FP32 accumulation in a declared order', 'decode rows and tall prefill', tag=True),
 F('kernel', 'core_f16_dense', [SC + r'(f16_|dense_)[^/]*'],
   'FP16 dense forms of the shared core: tall and hi-lo FP16 dense, Q8 x A8 dense, the hi-lo math, the dense linear and row / MMA-group / pair-map / f32-input-major plans and the dense model configuration',
   'the dense projection family every engine binds for its non-expert linears, with the plans that pick tile and split per shape',
   'linear.dense/f16+q8_0/f16-f32.enginelab', 'strix::core dense_*.hpp, f16_dense_*.hpp, dense_model_config.cpp', 'FP16 or Q8 weights, FP32 accumulation; hi-lo split keeps FP32-equivalent sums', 'decode and prefill', tag=True),
 F('kernel', 'core_gemv_rows', [SC + r'(gemv_|rows_plan|row_|direct_row_fetch|fma_rows|resident_|step_rows|step_inputs|decode_state_rows|q8_|lowbit_f16|microscaled_f16|linear_|layer_read_ahead)[^/]*'],
   'the decode GEMV rows family: the GEMV policy and its direct, look-ahead and staged row main loops, row layouts and transforms, resident Q8 rows and row plans, step rows / inputs plans, Q8_0 transpose, low-bit and microscaled FP16 staging, the linear plan table and references, direct row fetch and layer read-ahead',
   'one kernel family for 1-8 row decode projections across weight formats, with the staging that hides UMA reads; the shape strixflerp\'s linear.decode providers fill',
   'linear.decode/q8_0+lowbit+microscaled/f16-f32.enginelab-rows', 'strix::core gemv_rows.hpp, rows_plan.hpp, linear_plan*.hpp', 'FP32 accumulation per row in a fixed order', 'decode, 1-8 rows', tag=True),
 F('kernel', 'core_tall', [SC + r'tall_[^/]*'],
   'tall (many-row) linear plans: the tall chunk policy, tall family selection and the tall K plan',
   'the prefill-side split of the same projections when M grows past the rows family',
   'linear.prefill/-/f16-f32.enginelab-tall', 'strix::core tall_*.hpp', 'as the rows family', 'prefill chunks', tag=True),
 F('kernel', 'core_hc', [SC + r'(hc_|residual_|controller_|mix_)[^/]*'],
   'the hyper-connection core: HC geometry, expand rows, lane chains, the residual mix and its plan, reference and HIP transition, controller mix and controller RMS rows, the mix catalog and latency plan',
   'GLM-5.3 and DeepSeek V4.1 carry multi-stream hyper-connections (mHC); this is the one shared HC family (hc_pre / hc_post chains) the engines bind',
   'hc.mix/-/f32.enginelab', 'strix::core hc_*.hpp, residual_mix*.hpp', EXACT, 'decode and prefill', tag=True),
 F('kernel', 'core_head', [SC + r'(head_|screened_head|hip_screened_head|bf16_lm_head|markov_head|hip_bf16_embedding_rows|token_sampling_device_body)[^/]*'],
   'LM head and output kernels of the shared core: head argmax plans and emit, the screened head (a cheap screen then the exact rows that can win), the Markov head device / reference / round, BF16 embedding rows and the device token-sampling body',
   'the greedy head is a large decode cost; the screened head keeps the exact argmax while reading fewer vocabulary rows',
   'logits.head/bf16/f32.enginelab', 'strix::core head_*.hpp, screened_head*.hpp', 'exact argmax over FP32 logits', 'decode', tag=True),
 F('kernel', 'core_short_conv', [SC + r'short_convolution[^/]*'],
   'the short causal convolution and its ring buffer state',
   'the depthwise conv step of recurrent (GDN / Mamba-style) layers, kept in the shared core',
   'mixer.short-conv/-/f32.enginelab', 'strix::core short_convolution*.hpp'),
 F('kernel', 'core_tp_join', [SC + r'tp_join[^/]*'],
   'the exact tensor-parallel join: tp_join.hpp and tp_join_sum.hpp sum the two ranks\' partial outputs in a declared order so a TP2 result is bitwise the single-rank one',
   'the arithmetic half of the lab\'s exact TP2 decode: partials are joined in the same order the single-node kernel would have accumulated them',
   'collective.tp-join/-/f32.enginelab', 'strix::core tp_join*.hpp', 'fixed-order FP32 sums: bitwise the single-rank reference', 'two ranks, decode'),
 F('kernel', 'core_hadamard', [SC + r'hadamard[^/]*'],
   'the Hadamard transform helper of the shared core', 'rotations ahead of low-bit operands (IU4 x A4)',
   'rotate.fwht/-/f16.enginelab', 'strix::core hadamard.hpp', tag=True),
 F('kernel', 'core_activation_quant', [SC + r'activation_quant[^/]*'],
   'the shared activation quantiser (activation_quant.hpp)', 'A8 / A4 activation forms for the integer and IU4 kernels',
   'quant.activation/-/a8+a4.enginelab', 'strix::core activation_quant.hpp', tag=True),
 F('kernel', 'core_lookup', [SC + r'lookup_[^/]*'],
   'the lookup catalog and lookup plan of the shared core', 'table-driven selection of kernel variants by shape',
   'kernel-select.lookup/-/-', 'strix::core lookup_*.hpp'),
 F('kernel', 'core_common', [SC + r'.*'],
   'the remaining shared-core headers: backend selection (backend.hpp, backend_hip.hpp), byte budget and range, GLU pair, glu / pointwise / round-store, ordered reduce, numeric partition, NVFP4 BF16 table, null perturbation, block-draft reference, GPU port and SHA-256, weight archive and certificate, weight decode, file source, load queue and slice loader, host lane and free memory, layer set, load cohort, looped layers, window state buffers, HIP host ready, range weight loader and weight uploader',
   'the plumbing the shared kernels and loaders compile against; every other core module includes some of it',
   'kernel-abi.enginelab-core/-/-', 'strix::core headers', tag=True),
 # ------------------------------------------------------------------ attention core (src/attn_core, src/attn, include/strix/attn)
 F('kernel', 'attn_core_flash', [r'src/attn_core/(dense_flash|dense_decode|banded_flash|flash_|tile_stage|wmma_frag|wmma_types|online_softmax|masks|relative_bias|position_split|two_term_model)[^/]*'],
   'the shared dense flash attention: dense decode and its flash form with key split and budget, dense and banded flash prefill, the flash core, tile, prefill policy and tile stage, WMMA fragments and types, online softmax, masks, relative bias, position split and the two-term cost model',
   'one attention family for every engine\'s dense GQA layers, decode and prefill, WMMA on gfx1151',
   'attn.softmax-gqa/-/bf16-f32.enginelab', 'src/attn_core flash headers', 'FP16 WMMA scores, FP32 online softmax', 'decode and prefill', tag=True),
 F('kernel', 'attn_core_sparse_mla', [r'src/attn_core/(sparse_flash|selected_|eager_selected)[^/]*'],
   'selected-token (sparse) attention: the sparse flash MLA, row-group and WMMA forms with key split, the exact selected MLA with its rule and VOPD score body, eager selected attention and the selected block plan and flash prefill',
   'DeepSeek V4.1 and GLM-5.3 attend over a selected token set (DSA / MLA); this is the shared sparse half with an exact rule next to each fast form',
   'attn.sparse-mla/-/bf16-f32.enginelab', 'src/attn_core sparse / selected headers', EXACT, 'decode and prefill over selected tokens'),
 F('kernel', 'attn_core_select', [r'src/attn_core/(select|group_select|group_score|group_pool|index_score|eager_softmax)[^/]*'],
   'top-k selection for sparse attention: select, radix-3 select and threshold select with their rules and spec, the group select kernel, group score scan and pool, index score (band, mode, WMMA) and the eager softmax rule',
   'the indexer half of DSA-style attention: score the cache, pick the top tokens exactly',
   'attn.indexer-topk/-/f32.enginelab', 'src/attn_core select headers', 'exact top-k with declared tie order', 'decode and prefill', tag=True),
 F('kernel', 'attn_core_prepared', [r'src/attn_core/.*'],
   'prepared attention: prepared GQA and BF16 attention with lane plans and prepared step, chunk decode and chunk exact combine (rule), decode and row-group plans, cache rows and shadow, recurrent attention state, and the block / causal GQA references',
   'the plan-and-bind layer the engines call into, with exact chunk combine so a split decode keeps the single-pass bits',
   'attn.softmax-gqa/-/bf16-f32.enginelab-prepared', 'src/attn_core prepared headers', EXACT, 'decode', tag=True),
 F('kernel', 'attn_contracts', [r'src/attn/.*', r'include/strix/attn/.*'],
   'attention contracts and host references: the MLA contract and reference, the group indexer and its reference, group pool, FP8 plan, cache rows / shadow and prepared lanes (include/strix/attn)',
   'the declared operand contracts the attention kernels are tested against',
   'verify.numerics-contract/-/-.enginelab-attn', 'include/strix/attn/*.hpp, src/attn/*_ref.cpp', tag=True),
 # ------------------------------------------------------------------ MoE core
 F('kernel', 'moe_core_prefill', [r'src/moe_core/(hip_|one_split_prefill|qwen_grouped_tiles|qwen_k_|dense_a8_small|kquant_block_reads|exl3_trellis)[^/]*'],
   'grouped MoE prefill kernel bodies: the HIP MoE prefill cores (p1, core2, g3, da16, GSQ FP16 / WMMA, F16x project and down bodies, k-quant down, LDS epilogue), the DS41 grouped prefill and its variants, the one-split prefill, Qwen grouped tiles and k-quant decode, dense A8 small, k-quant block reads and the EXL3 trellis copy',
   'MoE prefill is where WMMA pays most; this is the shared grouped-GEMM family that Qwen, DS41, GLM and MiniMax prefill bind',
   'moe.experts-prefill/gguf-kquant+gsq+exl3/f16-f32.enginelab', 'src/moe_core/hip_*.inc bodies', 'FP32 accumulation, per-tile fixed order', 'prefill', tag=True),
 F('kernel', 'moe_core_microscaled', [r'src/moe_core/(microscaled|nvfp4_act|int4_layer)[^/]*'],
   'microscaled (MX / NVFP4-style) grouped MoE: the R1 microscaled geometry, pipe and CUDA plans, NVFP4 activation rule and the INT4 layer policy',
   'the low-bit expert path whose plans are backend-neutral (CUDA and HIP both bind them)',
   'moe.experts-decode/nvfp4+mx/f16-f32.enginelab', 'src/moe_core/microscaled_*.hpp', tag=True),
 F('kernel', 'moe_core_grouped', [r'src/moe_core/(grouped|device_grouped|expert_major|class_tree|pair_gather)[^/]*'],
   'the grouped expert GEMM family: grouped core, dispatch, policy and rows, grouped F16 / A8-F16 / integer decode / IU4 x A4 (contract, plan) / W4A4 / microscaled forms, grouped tall plan, device grouped plan, expert-major plan, class tree and pair gather',
   'one MoE grouped GEMM family per AGENTS.md: every engine\'s routed experts in decode and verify windows run through it',
   'moe.experts-decode/iu4+w4a4+f16/f16-f32.enginelab-grouped', 'src/moe_core/grouped_*.hpp', 'FP32 accumulation in a declared order per expert', 'decode and small verify windows', tag=True),
 F('kernel', 'moe_core_route', [r'src/moe_core/.*'],
   'MoE routing and decode: route, route plan R1, route prune, route project and wave kernels, the HIP route tap, R1 route and tail, decode and decode schedule, device plan and index, expert activation and the host reference',
   'the router half of the shared MoE core (softmax / sigmoid top-k and the projections that feed it)',
   'moe.router/-/f32.enginelab', 'src/moe_core/route*.hpp, decode*.hpp', 'exact top-k, declared ties', 'decode and prefill', tag=True),
 F('runtime', 'moe_policy', [r'include/strix/moe/.*'],
   'MoE contracts (include/strix/moe): activation policy, combine, declared experts, expert union, route file, route policy, route project and route-tap exchange',
   'the engine-neutral MoE declarations: how routed outputs combine and how a pair exchanges route taps',
   'moe.combine/-/f32.enginelab', 'include/strix/moe/*.hpp', tag=True),

 # ------------------------------------------------------------------ transports and two-node pieces
 F('runtime', 'xport_crc', [r'src/xport/(crc32c_fold|device_crc32c|row_fold)[^/]*', r'src/dist_crc32c[^/]*'],
   'wire integrity: CRC32C folding on the device (device_crc32c.hip) and the host (vector and parallel CRC32C, the vector form from Intel ISA-L, BSD-3-Clause), and row folds',
   'every USB4 message carries a checksum; the device form lets a GPU-direct message be checked without a host copy, and deferred checksums were part of the measured MiniMax exchange win',
   'verify.wire-crc32c/-/u32.enginelab', 'strix::xport device_crc32c.hpp, src/dist_crc32c_*.hpp', 'CRC32C (Castagnoli)', 'every exchanged message'),
 F('runtime', 'xport_tp_gather', [r'src/xport/(tp_gather|tp_join_|bf16_wire|pair_device_link)[^/]*'],
   'the tensor-parallel slot gathers: the one instantiation of the shared Channel for TP gathers (tp_gather.hip and its traits), the TP join BF16 / exec / map HIP pieces, the BF16 wire format and the pair device link',
   'the transport half of exact TP2 decode: each rank\'s slot partials travel over the pair link and are joined in the declared order (core_tp_join)',
   'collective.tp-gather/-/bf16-f32.enginelab', 'strix::xport tp_gather_hip.hpp, tp_join_*_hip.hpp', 'bytes moved verbatim; the join order is fixed', 'two ranks, every decode layer'),
 F('runtime', 'xport_kvshard', [r'src/xport/kvshard[^/]*'],
   'KV-shard exchange: the KV-shard channel (host and HIP), its traits and oracle',
   'a two-rank prefill shards the KV cache by position and exchanges shards (GLM kvshard prefill)',
   'kvcache.shard-exchange/-/bf16.enginelab', 'strix::xport kvshard_channel*.hpp'),
 F('runtime', 'xport_channel', [r'src/xport/.*'],
   'the shared exchange channel (src/xport): backend and transport backend, borrowed transport, channel (device, HIP, impl.hip, communicator), exchange format (host and HIP), framed exchange, split frame, HIP pipe staging and staging, the HIP channel runtime, small progress, stats, wire and the DS41 adapter',
   'one channel that every engine\'s pair traffic goes through (AGENTS.md: two-node pipelines, transports, exchange overlap are shared code)',
   'transport.channel/-/-', 'strix::xport channel.hpp, channel_hip.hpp'),
 F('runtime', 'dist_communicator', [r'src/(dist_communicator|transport|dist_latency_trace)[^/]*', r'include/strix/(transport|dist/(communicator|exchange_|transport|open_transport|mesh|link_cost|pipeline_timeline))[^/]*'],
   'the two-rank communicator and message transport (include/strix/dist): whole messages in order between two ranks, backends in-process pair, TCP and the host path of USB4 streams striped over a cable\'s streams; persistent progress threads, exchange with an on-wire callback, abort and close semantics; exchange direction and transport, transport caps and spec, mesh, link cost, pipeline timeline and latency trace',
   'the message layer the pipelines and lockstep control run on; the catalogue\'s transports are RoCE verbs and TCP, this adds a USB4 host-stream backend behind the same exchange contract',
   'transport.message/-/-', 'strix::dist Transport / Communicator (include/strix/dist/transport.hpp, communicator.hpp)', 'bytes; payload + trailer, in order', 'two ranks; TCP for bring-up, USB4 streams in production'),
 F('runtime', 'dist_gpu_direct', [r'src/(dist_gpu_direct)[^/]*', r'include/strix/dist/gpu_direct[^/]*'],
   'GPU-direct exchange: setup and copies between the two iGPUs\' memory through the USB4 DMA-BUF stream extension (dist_gpu_direct*.cpp / .hip)',
   'moves activations iGPU to iGPU with no host staging; the routing per message class picks it for the sizes where it wins',
   'transport.usb4-gpu-direct/-/-', 'strix::dist gpu_direct.hpp, gpu_direct_copy.hpp', 'bytes, verified', 'two hosts on USB4 with the strix_gpu_stream UAPI (catalog/enginelab_usb4)'),
 F('runtime', 'dist_host_staged', [r'src/(dist_host_copy|dist_staged_ready)[^/]*', r'include/strix/dist/.*'],
   'host-path staging: host copies, host waits and the staged-ready protocol with its HIP kernels (a producer publishes a ready word the consumer polls)',
   'producer-published buffers and an in-kernel wait were measured to cut MiniMax exchange overhead from 2.91 to 1.89 ms a token (usb4-driver README)',
   'transport.staged-ready/-/-', 'strix::dist staged_ready*.hpp, host_copy.hpp'),
 F('runtime', 'distributed_control', [r'src/(distributed_(control_wire|greedy_control|remote_control|token_transaction|position_completion|flow))[^/]*', r'include/strix/distributed_(control_wire|greedy_control|remote_control|token_transaction|position_completion|flow)[^/]*'],
   'distributed control: the control wire, greedy control, remote control, token transactions, position completion and flow control between ranks',
   'the control plane of a two-node decode: which rank commits which token and when, with replay-safe transactions',
   'runtime.distributed-control/-/-', 'src/distributed_*.cpp'),
 F('runtime', 'distributed_runtime', [r'(src|include/strix)/distributed_[^/]*'],
   'the distributed runtime: data plane, native backend, placement, runtime, session and stage resources / state',
   'binds a model\'s stage graph to two nodes: what lives where and how stages hand off',
   'runtime.distributed/-/-', 'src/distributed_runtime.cpp, distributed_session.cpp'),
 F('runtime', 'direct_transport', [r'(src|include/strix)/(hip_)?(direct_transport_|linux_direct_transport|direct_stream_control)[^/]*'],
   'the native USB4 stream transport: driver, Linux driver over /dev/tbstreamN, completion, frames, GPU export, lease, native backend and session, pool and pool memory, stream control',
   'the userspace half of the USB4STREAM kernel series (catalog/enginelab_usb4): bounded queues, fail-closed recovery, DMA quiescence before reuse',
   'transport.usb4-stream/-/-', 'strix direct_transport_*.hpp', 'bytes; frames from the stream device', 'two hosts on the patched kernel'),
 F('runtime', 'direct_gpu_stream', [r'(src|include/strix)/(direct_gpu_|hip_direct_gpu|linux_direct_gpu_stream|gpu_queue_session|linux_gpu_queue|gpu_visibility)[^/]*'],
   'GPU-direct streams: boundary staging, pool allocator, stream commissioning and the GPU side, the Linux GPU-stream UAPI binding, GPU queue sessions and the GPU visibility probe and pattern',
   'commissioning proves the USB4 controller sees the iGPU memory it was given before any real traffic flows',
   'transport.usb4-gpu-stream/-/-', 'strix direct_gpu_*.hpp, linux_direct_gpu_stream_uapi.hpp'),
 F('runtime', 'dual_link', [r'(src|include/strix)/dual_link_[^/]*'],
   'dual-link runtime and telemetry: two USB4 cables as one link', 'about 37 Gb/s a direction with four streams over two cables (usb4-driver README)',
   'transport.dual-link/-/-', 'strix dual_link_runtime.hpp'),
 F('runtime', 'pair_link', [r'include/strix/pair/(link|gather_|bf16_wire|producer_publish|embed_frame|pipe)[^/]*'],
   'pair link pieces (include/strix/pair): link, link flags and mux, gather link and fn, BF16 wire, producer publish (host and HIP), embed frame and pipe',
   'how two ranks move activation frames, including producer-published buffers a consumer kernel waits on',
   'transport.pair-link/-/-', 'include/strix/pair/link*.hpp'),
 F('runtime', 'pair_pipeline', [r'include/strix/pair/(layer_pipeline|layercut|stage_map|chain|prefill_group|window_|split_plan|planner|placement_inputs|balance|regime|relayout|column_split)[^/]*'],
   'two-node pipelines (include/strix/pair): layer pipeline, layer cut, stage map, chains and chain topology, prefill groups, window overlap and window ring, split plan, planner, placement inputs, balance, regime, relayout and column split',
   'the pipeline-parallel shape of the DS41 and MiniMax pairs: where to cut the layers and how to overlap windows with the exchange',
   'runtime.pair-pipeline/-/-', 'include/strix/pair/*.hpp'),
 F('runtime', 'pair_state', [r'include/strix/pair/.*'],
   'pair state and lockstep (include/strix/pair): lockstep control bookkeeping (rank, round and last completed round agreed before acting), state declarations, owners and relay, TP inputs and TP layout',
   'the guard rails of a two-rank decode: a dropped, duplicated or stale control message is caught before either side acts',
   'runtime.pair-lockstep/-/-', 'include/strix/pair/lockstep.hpp, tp_layout.hpp'),
 # ------------------------------------------------------------------ DeepSeek V4.1
 F('kernel', 'ds41_expert_tp', [r'(src|include/strix)/(hip_)?(ds41_expert_tp|ds41_shexp_tp|ds41_engram_tp)[^/]*'],
   'DeepSeek V4.1 expert tensor parallelism over two ranks (identity ds41-expert-tp2-i1152-v1): every routed expert split by its intermediate dimension, rank r holding 1152 gate / up rows (IQ2_XXS) and the matching 36 Q2_K groups of each down row, the boundary block stored on both ranks; plus shared-expert TP on the trellis library and engram TP',
   'expert tensor-parallel decode: each rank does half of every expert instead of half of the experts, so load is even whatever the router picks; the shape a TP2 MoE decode would take in strixflerp',
   'moe.experts-tp2/iq2_xxs+q2_k/f32.enginelab', 'strix ds41_expert_tp.hpp (STRIX_DS41_EXPERT_TP=1), ds41_shexp_tp.hpp', 'existing Q2_K / IQ2_XXS readers over each rank\'s packed half rows; partials joined exactly', 'two ranks, decode', ),
 F('runtime', 'ds41_expert_tier', [r'(src|include/strix)/(ds41_expert_rpc|ds41_expert_server|hip_ds41_expert_backend|ds41_expert_backend|ds41_hot_placement|ds41_host_pin|ds41_host_promote|ds41_tier_stats|ds41_touch_batch|ds41_replica_census|ds41_memory_budget)[^/]*'],
   'DS41 expert tiering: the expert RPC and server, the HIP expert backend, hot placement, host pinning (plan, admission) and promotion, tier statistics, touch batches, replica census and the memory budget',
   'keeps hot experts in fast memory and serves cold ones from a peer or host: the residency side of a 2-node DS41',
   'runtime.expert-tiering/-/-', 'src/ds41_expert_server.cpp, ds41_hot_placement.cpp'),
 F('executor', 'ds41_dspark', [r'(src|include/strix)/(hip_)?(ds41_dspark|deepseek41_dspark|deepseek41_batch_dspark|ds41_spec_tail|ds41_pipeline_dspark)[^/]*'],
   'DSpark speculative drafting for DeepSeek V4.1: the DSpark driver, context, files and vocabulary, the DSpark target and batch DSpark forward, the speculative tail and the pipelined DSpark',
   'the drafting that lifts DS41 pair decode from 26.8 to 41.7-60.0 tok/s with exact outputs (README, f42edd05e)',
   'draft.dspark/-/f32.enginelab-ds41', 'src/ds41_dspark*.cpp, hip_ds41_dspark.hip', EXACT, 'pair decode', tag=True),
 F('executor', 'ds41_pipeline', [r'(src|include/strix)/(hip_)?(ds41_pipeline_|ds41_decode_xport|deepseek41_prefill_xport|ds41_state_transfer|ds41_staged_ready|deepseek41_state_|ds41_step_trace|ds41_hetero)[^/]*'],
   'the DeepSeek V4.1 two-node path: the pipeline front, multi-front and wire, decode transport, prefill transport (HIP) and its traits, state transfer, frames and pieces, staged ready and step trace',
   'DS41 runs across two Strix Halos over USB4 (2K 792, 32K 1,224 tok/s prefill; 26.8 tok/s decode, README): this is the layer-split pipeline that does it',
   'runtime.pipeline-2node/-/-', 'src/ds41_pipeline_front.cpp, hip_deepseek41_prefill_xport.hip'),
 F('kernel', 'ds41_trellis', [r'(src|include/strix)/(hip_)?(ds41_trellis|ds41_scalar_codecs)[^/]*'],
   'DS41 trellis kernels: trellis linear and grouped trellis (HIP) and the scalar codecs', 'the K4 / K5 trellis dense path of the DS41 mix',
   'linear.decode/exl3-trellis/f16-f32.enginelab-ds41', 'hip_ds41_trellis.hip, hip_ds41_trellis_group.hip', tag=True),
 F('kernel', 'ds41_rows', [r'(src|include/strix)/(hip_)?(ds41_f16_rows|ds41_f8_rows|ds41_native_fp8|ds41_integer_operand|ds41_formats|ds41_packed|ds41_iq2_xxs_grid|ds41_dense_|ds41_rn_)[^/]*'],
   'DS41 dense rows: F16 rows, fast F8 rows, native FP8, the integer operand, formats, packed weights, the IQ2_XXS grid (from ggml, MIT), the dense adapter, FP16 dense, geometry and plan, and the RN wide rows (STRIX_DS41_RN_WIDE: a per site x format x rows table choosing between the engine kernel and multi-row forms for Q8_0 / U6 rows 2-8, all variants bitwise equal)',
   'the non-expert projections of DS41 in its native FP8 / GGUF forms',
   'linear.decode/fp8+gguf/f16-f32.enginelab-ds41', 'hip_ds41_f16_rows.hip, hip_ds41_f8_rows_fast.hip, ds41_native_fp8.cpp', tag=True),
 F('kernel', 'ds41_routes', [r'(src|include/strix)/(hip_)?(ds41_routes_fast|ds41_moe)[^/]*'],
   'DS41 MoE routing: the fast routes (HIP, with its body) and the MoE layer', 'the DS41 router and routed-expert layer',
   'moe.router/-/f32.enginelab-ds41', 'hip_ds41_routes_fast.hip, hip_ds41_rn_*.hip', tag=True),
 F('kernel', 'ds41_attention', [r'(src|include/strix)/(hip_)?(ds41_attention|deepseek41_attention_)[^/]*'],
   'DS41 attention: the HIP attention with F16 WMMA, flash and QK pieces, the attention host and state',
   'the DS41 compressed / selected attention on gfx1151',
   'attn.ds41/-/bf16-f32.enginelab', 'hip_ds41_attention.hip'),
 F('kernel', 'ds41_engram', [r'(src|include/strix)/(hip_)?ds41_engram[^/]*', r'src/engram/.*', r'include/strix/engram/.*'],
   'engram lookup: the hasher, read-ahead, row prep, row reader and spec (src/engram, include/strix/engram) with its HIP inject and row kernels, and DS41\'s engram and diagnostics',
   'a hashed n-gram embedding table read ahead of the layer that injects it; StrixFold\'s engram is the pod\'s counterpart',
   'embed.engram/-/bf16.enginelab', 'strix::engram row_reader.hpp, inject.hpp'),
 F('executor', 'ds41_prefill', [r'(src|include/strix)/(hip_)?(ds41_prefill|deepseek41_prefill)[^/]*'],
   'DS41 prefill: the HIP prefill and its relaxed form, adapter, precision, route and tuning, the DeepSeek prefill and its levers',
   'the prompt path of DS41 (docs/ds41-prefill-tuning.md)',
   'executor.prefill/-/-.enginelab-ds41', 'hip_ds41_prefill.hip, hip_deepseek41_prefill.hip'),
 F('runtime', 'ds41_rtx', [r'(src|include/strix)/(ds41_rtx|rtx_state_import)[^/]*'],
   'the optional RTX PRO route for DS41: RTX budget, drafter select, profile and schedule, and RTX state import',
   'a discrete GPU joins a Strix Halo node for drafting or prefill (docs/discrete-gpu.md)',
   'runtime.hetero-route/-/-', 'src/ds41_rtx_*.cpp, rtx_state_import.cpp'),
 F('serve', 'ds41_serve', [r'(src|include/strix)/(deepseek41_(serve|prefix_cache|session_cache|cache_lease))[^/]*', r'src/deepseek41/http_session_owner[^/]*'],
   'DS41 serving: serve cache policy, observation and scheduling, the prefix and session caches, the cache lease and the HTTP session owner',
   'how DS41 keeps conversations warm across requests on a pair',
   'serve.engine/-/-.enginelab-ds41', 'src/deepseek41_*cache.cpp, src/deepseek41/http_session_owner.cpp', tag=True),
 F('executor', 'ds41_vision', [r'src/deepseek41/vision_[^/]*'],
   'DS41 vision: the vision adapter and splice', 'image rows spliced into the DS41 prompt', 'vision.splice/-/bf16.enginelab-ds41', 'src/deepseek41/vision_*.cpp'),
 F('executor', 'ds41_engine', [r'(src|include/strix)/(hip_)?(ds41_|deepseek41_)[^/]*', r'src/deepseek41/.*'],
   'the DeepSeek V4.1 engine: model, family, plan, describe and model description, host / device / batch forward and batch schedule, layer, pair defaults and pair profile table, exact flags and fast profile, the file reader, weight overlay and read options, prepared load and the MSVC builtins shim',
   'the binding of DS41 to the shared cores: what is model-specific and nothing else (AGENTS.md)',
   'executor.model/-/-.enginelab-ds41', 'src/deepseek41_model.cpp, hip_deepseek41_device_forward.hip', tag=True),

 # ------------------------------------------------------------------ speculative decoding
 F('kernel', 'mtp_draft_head', r('mtp_draft_head', 'mtp_fast_draft', 'mtp_projection', 'mtp_shared_expert', 'mtp_attention', 'mtp_input_fusion', 'mtp_prime_rows'),
   'the MTP draft head on the device: draft head, fast draft, projection, shared expert, attention (and its contract), input fusion and prime rows',
   'the multi-token-prediction head that proposes tokens for the verify window; device side',
   'draft.mtp/-/f32.enginelab-head', 'hip_mtp_draft_head.hip, hip_mtp_fast_draft.hip, hip_mtp_input_fusion.hip', tag=True),
 F('runtime', 'mtp_drafters', r('mtp_draft_', 'mtp_context_draft', 'mtp_frame_draft', 'mtp_ngram_draft', 'compiled_mtp_draft_plan', 'qualified_mtp_draft_backend'),
   'MTP drafting control: draft activation, binding, contract, cycle input, profiles, qualification, readiness, split, state contract and memory, tensor audit and vocabulary; the context, frame and n-gram drafters, the compiled draft plan and the qualified draft backend',
   'which drafter proposes what each cycle (MTP head, copy spans from the context, n-grams), qualified before it may run',
   'speculation.draft/-/i32.enginelab-mtp', 'src/mtp_draft_*.cpp, mtp_context_draft.cpp, mtp_ngram_draft.cpp', tag=True),
 F('runtime', 'mtp_forward', r('mtp_forward_', 'qualified_mtp_forward_runtime', 'mtp_payload_geometry', 'mtp_output_contract', 'mtp_logits_dump', 'mtp_step_logits_digest', 'mtp_native_validation', 'mtp_shared_execution_witness'),
   'the MTP forward: contract, golden file, payload, replay, runtime and semantics; the qualified forward runtime, payload geometry, output contract, logits dump and step digest, native validation and shared execution witness',
   'proves a draft forward computes what its contract says before its proposals are trusted',
   'draft.mtp-forward/-/f32.enginelab', 'src/mtp_forward_*.cpp'),
 F('runtime', 'mtp_window', r('mtp_window_', 'mtp_target_', 'mtp_batch_window_controller', 'mtp_depth_', 'mtp_cohort_partition', 'mtp_selection_transaction'),
   'MTP verify windows: window controller and verifier (HIP), target checkpoint (plan), target hidden, target window (payload, policy), batch and depth window controllers, cohort partition and selection transaction',
   'verification bit-identical to greedy: every committed row is a verified target row, so MTP never changes the tokens, only the speed (README: GLM pair 25.7 -> 40.4-66.5 tok/s with MTP + wide)',
   'speculation.window-verify-lossless/-/-.enginelab-mtp', 'hip_mtp_window_verifier.hip, src/mtp_window_controller.cpp', EXACT, 'decode, 8-row windows', tag=True),
 F('runtime', 'mtp_runtime', r('mtp_'),
   'the MTP runtime: runtime and greedy runtime (HIP), resources, asset loader, sidecar binding, cost-table session and the rest of the MTP host pieces',
   'ties the draft head, drafters and verify windows into one decode loop per engine',
   'draft.mtp/-/f32.enginelab-runtime', 'hip_mtp_runtime.hip, hip_mtp_greedy_runtime.hip', tag=True),
 F('runtime', 'spec_wide_window', [r'(src|include/strix)/spec/(wide_window|tree_window|verify_cost_table|commit_tail|row_rule|draft_rows)[^/]*'],
   'wide verify windows (include/strix/spec): the backend-neutral pricing of a wide window (rows-1 copied proposals plus the pending token in one forward of rows rows, against the narrow path, with a declared keep stride), tree windows, the verify cost table, commit tail, row rule and draft rows',
   'MTP wide drafting: a long, confident copy span is verified in one wide forward instead of several 8-row windows; tokens never change with the decision, only the cost',
   'speculation.window-verify-wide/-/-', 'strix::spec wide_window.hpp, verify_cost_table.hpp', 'host only; the decision changes cost, never tokens', 'one request, long copy spans (edit / JSON / code)'),
 F('runtime', 'spec_block_draft', [r'(src|include/strix)/spec/block_[^/]*'],
   'block drafting (include/strix/spec, src/spec): block candidates, block drafter and policy, block-parallel verification and the block-draft reference',
   'parallel block proposals verified in one pass',
   'speculation.draft/-/i32.enginelab-block', 'strix::spec block_drafter.hpp', tag=True),
 F('runtime', 'spec_core', [r'(src|include/strix)/spec/.*'],
   'speculation core (include/strix/spec, src/spec): drafting and draft declarations, the draft-verify runner, engine target, greedy continuation and prefix, MTP chain and defer, pair round',
   'the engine-neutral contract every drafter and verifier binds to, including the pair-round form for two ranks',
   'speculation.accept/-/-.enginelab-core', 'strix::spec drafting.hpp, engine_target.hpp', tag=True),
 F('runtime', 'speculation_sampling', r('speculation_sampling', 'speculation_rng', 'compiled_sampled_speculation'),
   'sampled speculation: sampling, sampling cycle and executor, the keyed RNG and the compiled sampled-speculation backend',
   'speculative decoding under temperature sampling with a reproducible keyed RNG, so a sampled stream is replayable',
   'speculation.accept/-/-.enginelab-sampled', 'src/speculation_sampling*.cpp, speculation_rng.cpp', tag=True),
 F('runtime', 'speculative_verify', r('speculative_verify', 'compiled_speculative_verify', 'compiled_speculation_backend'),
   'speculative verification: verify binding, plan and runtime, the compiled verify and the compiled speculation backend',
   'binds a verify window to a compiled execution plan',
   'speculation.window-verify/-/-.enginelab', 'src/speculative_verify_*.cpp', tag=True),
 F('runtime', 'speculation_control', r('speculation', 'draft_proposal_state'),
   'the speculation controller: speculation, controller, cycle, executor, measurement and multi-draft, and the draft proposal state',
   'picks drafter and window per cycle from measured costs',
   'speculation.controller/-/-', 'src/speculation_controller.cpp'),
 F('runtime', 'prompt_lookup', r('prompt_lookup', 'sparse_lookup', 'sparse_select_policy') + [r'(src|include/strix)/lookup/.*'],
   'prompt lookup drafting and sparse lookup: the prompt-lookup draft and runtime (HIP), sparse lookup and its executor, the sparse select policy and the lookup index',
   'copy-from-context proposals that feed the wide windows (the edit rows of the README reach 153-155 tok/s served)',
   'draft.prompt-lookup/-/i32.enginelab', 'src/prompt_lookup_draft.cpp, hip_prompt_lookup_runtime.hip'),
 F('kernel', 'dspark_markov', [r'src/dspark_common/.*'],
   'DSpark Markov candidate kernels shared by the DSpark engines', 'the candidate step of DSpark drafting', 'draft.markov/-/f32.enginelab', 'src/dspark_common/markov_cand_kernels.*', tag=True),
 # ------------------------------------------------------------------ Qwen sparse attention, GDN, HC, PLE
 F('kernel', 'qsa_prefill', r('qsa_fast_prefill', 'qsa_prefill_', 'qsa_index_', 'qsa_pooled_score', 'qsa_stage_score', 'qsa_scatter', 'qsa_score_scratch', 'qsa_cohort_part2', 'qsa_committed_pool'),
   'QSA (Qwen3.8-Flash-Next sparse attention) prefill kernels: the fast prefill profile (union of TR rows\' selections in 4-token blocks, WMMA S^T = K x Q^T and O^T += V^T x P^T with FP32 accumulation, power-of-two Q scaling) with its KV-pair, pipe, shadow and share forms, prefill batch, budget, flash, index and stages, index band and WMMA score, pooled and stage scores, scatter, score scratch, cohort part 2 and committed pool',
   'WMMA sparse prefill at 8-32K for the 12 QSA layers of Qwen3.8-Flash-Next',
   'attn.qsa-indexer/-/bf16-f32.enginelab-prefill', 'hip_qsa_*.inc (included by hip_indexed_attention_launcher.hip)', 'FP16 WMMA, FP32 accumulation and softmax', 'prefill groups of more than 8 rows, head_dim 256', tag=True),
 F('runtime', 'qsa_plan', r('qsa_', 'compact_qsa'),
   'QSA plans and diagnostics: selection plan, input prefetch, cache-only mode and the selection / GQA-reuse diagnostics',
   'what the QSA kernels read and when',
   'attn.qsa-ids/-/i32.enginelab', 'src/qsa_selection_plan.cpp', tag=True),
 F('kernel', 'indexed_attention', r('indexed_attention'),
   'the indexed attention launcher (HIP; it includes the QSA kernel bodies), state and workspace', 'the decode-side entry of QSA attention',
   'attn.qsa-indexer/-/bf16-f32.enginelab-decode', 'hip_indexed_attention_launcher.hip', tag=True),
 F('kernel', 'gdn_prefill', r('gdn_prefill', 'gdn_chunk', 'gdn_fp16_chunk', 'gdn_qpf2k', 'gdn_islands'),
   'gated delta-net prefill: the prefill scan and overlap, chunked and FP16 chunked recurrence, the qpf2k prefill and 64-wide gates, and islands',
   'chunked GDN for the 36 linear-attention layers of Qwen3.8-Flash-Next at long prompts',
   'mixer.gdn-prefill/-/f32.enginelab', 'hip_gdn_prefill_scan.inc, hip_gdn_fp16_chunk.inc', tag=True),
 F('kernel', 'gdn_decode', r('gdn_'),
   'gated delta-net decode: the GDN launcher (HIP) and decode fuse, recurrence rows, recurrent layout, LDS index and storage, dims, identity, output admission and the wave8 / layout diagnostics',
   'the per-token GDN step and the state layout it reads',
   'mixer.gdn/-/f32.enginelab', 'hip_gdn_launcher.hip', EXACT, 'decode', tag=True),
 F('kernel', 'ops_recur', [r'src/ops/recur/.*'],
   'shared recurrence ops (src/ops/recur): gated-delta policy, prologue, rows (body), step (body) and window, the recurrent decode plan and step',
   'the backend-neutral recurrence family (one per AGENTS.md) the GDN engines bind',
   'mixer.gdn/-/f32.enginelab-ops', 'src/ops/recur/gated_delta_*.hpp', tag=True),
 F('runtime', 'recurrent_state', r('recurrent_'),
   'recurrent state: the recurrent plan, state policy and state type', 'how GDN state is held, checkpointed and rolled back for verify windows',
   'kvcache.recurrent-state/-/f32.enginelab', 'src/recurrent_plan.cpp'),
 F('kernel', 'hc_prefill', r('qpf2k_hc'),
   'hyper-connection prefill (qpf2k): the HC prefill with inverse reuse (and layout), out16 and output3 bodies', 'the prompt-path HC mix',
   'hc.mix-prefill/-/f32.enginelab', 'hip_qpf2k_hc*.inc', tag=True),
 F('kernel', 'hc_hyper', r('hc_', 'hyper_mix', 'lane_chain_hyper', 'repacked_hyper') + [r'(src|include/strix)/hc/.*'],
   'hyper-connection decode: the hyper-mix launcher (HIP), fast WMMA HC, wide2 VOPD bodies, repacked down layout and pack, parallel projection oracle, projection prefetch, lane-chain hyper output, the HC contract and host reference, and their tests',
   'the decode-side HC mix with VOPD dual-issue bodies on gfx1151',
   'hc.mix-decode/-/f32.enginelab', 'hip_hyper_mix_launcher.hip, hip_hc_*.inc', EXACT, 'decode', tag=True),
 F('runtime', 'qpf2k_schedule', r('qpf2k'),
   'the qpf2k prefill schedule defaults and pinned buffers (STRIX_QPF2K_SCHED, the code-default exact long-context items)', 'the default prompt schedule of the Qwen prefill',
   'tune.prefill-schedule/-/-', 'include/strix/qpf2k_defaults.hpp'),
 F('kernel', 'ple', r('ple_', 'graph_ple'),
   'per-layer embeddings (PLE): direct, discard safety, gap witness, the PLE launcher (HIP), lookup, projection tile, reclaim, rolling, rows site, serve policy, staged gather, workspace and the graph PLE',
   'Qwen3.8-Flash-Next\'s PLE rows, gathered and injected per layer with reclaimable memory',
   'ple.inject/-/bf16.enginelab', 'hip_ple_launcher.hip, src/ple_*.cpp', tag=True),
 # ------------------------------------------------------------------ heads, sampling, logprobs
 F('kernel', 'token_sampling', r('token_sampling'),
   'token sampling: host and device samplers (HIP) and timing', 'one sampler shared by the replica and served paths',
   'sample.next-token/-/f32.enginelab', 'hip_token_sampling.hip, hip_token_sampling_device.hip', tag=True),
 F('kernel', 'logits_head', r('lm_head_launcher', 'head_argmax', 'argmax_core', 'decode_head_rows', 'decode_integer_head', 'logit_rows'),
   'the LM head launcher (HIP), head argmax and argmax core, decode head rows, the integer head and logit rows', 'greedy decode output',
   'logits.argmax/-/f32.enginelab', 'hip_lm_head_launcher.hip', 'exact argmax', 'decode', tag=True),
 F('runtime', 'logprobs', r('logprob_partials', 'cuda_logprob', 'sampled_target_logits', 'teacher_logit'),
   'log-probabilities: logprob partials (HIP and CUDA spellings), the sampled target-logits reader and the teacher logit spool', 'logprobs for the API and the quality gate',
   'sample.logprobs/-/f32.enginelab', 'hip_logprob_partials.hip', tag=True),
 # ------------------------------------------------------------------ MoE launchers and schedules (src root)
 F('kernel', 'moe_prefill_launch', r('moe_prefill_', 'moe_tile64', 'router_fast_prefill'),
   'MoE prefill launch pieces: prefill commit, route, small, tiers and tile64 bodies, tile64 down body and the fast-prefill router', 'the Qwen MoE prompt path',
   'moe.experts-prefill/-/f16-f32.enginelab-launch', 'hip_moe_prefill_*.inc', tag=True),
 F('kernel', 'moe_q4k_a8', r('moe_q4k_a8', 'moe_a8_down', 'q51_'),
   'Q4_K x A8 MoE: the Q4K-A8 contract, known rows, probe and runtime, the A8 down contract, q51 grouped down (contract, probe), known rows and slot-parallel contract',
   'an integer A8 expert path over Q4_K experts',
   'moe.experts-decode/q4_k/a8-f32.enginelab', 'hip_moe_q4k_a8_*.hip', tag=True),
 F('kernel', 'moe_launch', r('moe_experts_launcher', 'moe_route_', 'moe_direct_b', 'moe_overlap_dispatcher', 'moe_slot_merge', 'moe_batched'),
   'MoE decode launchers: experts launcher, route launcher, route BF16 and prune, direct-B, the overlap dispatcher, slot merge and the batched activation diagnostic', 'the Qwen MoE decode launches and their overlap',
   'moe.experts-decode/-/f16-f32.enginelab-launch', 'hip_moe_experts_launcher.hip, hip_moe_overlap_dispatcher.hip', tag=True),
 F('runtime', 'moe_schedule', r('moe_', 'deferred_moe', 'expert_stream', 'route_receipt', 'wave_router'),
   'MoE schedules: layer runtime and schedule, plan, token cost backend and schedule, fork-join arena, deferred MoE status, expert stream epoch, route receipt and the wave router contract', 'scheduling routed experts across lanes and windows',
   'moe.schedule/-/-', 'src/moe_layer_schedule.cpp'),
 F('kernel', 'shared_expert', r('shared_expert'),
   'the shared expert: launcher (HIP), prefetch and rows diagnostic', 'the always-on expert of each MoE layer', 'moe.shared-expert/-/f16-f32.enginelab', 'hip_shared_expert_launcher.hip'),
 # ------------------------------------------------------------------ dense, rows, projections (src root)
 F('kernel', 'dense_launch', r('dense_'),
   'dense launch pieces: A8 G64 dense, FP16 tiled and WMMA dense, dense rows tile and wave, the experimental dense trellis (HIP) and its pack, dense model config',
   'the engine-side dense projections over the shared core',
   'linear.dense/-/f16-f32.enginelab-launch', 'hip_dense_trellis.hip, hip_dense_f16_wmma.hpp', tag=True),
 F('kernel', 'q8_rows', r('q8_rows', 'q8_projection', 'q8_0_rule'),
   'Q8 rows: the Q8 rows GEMV, IU4 rows, lane VOPD and VOPD bodies, sites, the Q8 projection manifest and organization, the Q8_0 rule', 'Q8_0 decode rows with VOPD dual issue',
   'linear.decode/q8_0/f16-f32.enginelab', 'hip_q8_rows_gemv.hpp', tag=True),
 F('kernel', 'rows_select', r('rows_', 'r1_rows', 'wave_fit', 'wave32_dpp', 'qwen_dpp_reduce', 'fma_rows_policy', 'split_k_policy', 'a8_row_pitch'),
   'row-kernel selection: rows fit and launch registry, R1 row tiles, wave fit, wave32 DPP and the Qwen DPP reduce, FMA rows policy, split-K policy and A8 row pitch', 'picks the rows kernel and split per shape',
   'kernel-select.rows/-/-.enginelab', 'include/strix/rows_launch_registry.hpp', tag=True),
 F('kernel', 'integer_projection', r('integer_projection', 'packed_projection', 'projection_reduction', 'onfly_i4', 'qwen_onfly_i4', 'low_precision', 'gsq_a8', 'decode_projection_group', 'decode_weight_sweep'),
   'integer and packed projections: the integer projection (HIP), its execution and runtime, packed projection, projection reduction, on-the-fly INT4 (generic and Qwen), low precision, GSQ A8 groups, decode projection groups and the weight sweep',
   'integer-dot projections for the GSQ and on-the-fly INT4 FAST levers',
   'linear.decode/gsq+iu4/a8-f32.enginelab', 'hip_integer_projection.hip', tag=True),

 # ------------------------------------------------------------------ execution, graphs, launch
 F('runtime', 'execution', r('execution_', 'compiled_execution'),
   'compiled execution: arena, binding, buffers, IO, memory, plan, resources, runtime, telemetry, tensor access and tensor binding, and the compiled execution',
   'a model forward compiled once into a plan of bound buffers and launches, replayed per step',
   'executor.compiled-plan/-/-', 'src/execution_*.cpp'),
 F('runtime', 'prepared_views', r('prepared_'),
   'prepared execution: arena views, prepared execution, the prepared serial session and tensor views', 'zero-copy views of prepared weights and buffers',
   'executor.prepared-views/-/-', 'src/prepared_execution.cpp'),
 F('runtime', 'launch', r('dispatch_', 'launch_', 'launcher_catalog', 'runtime_launchers', 'kernel_registry', 'event_queue_trace', 'embedding_launcher'),
   'launch plumbing: dispatch fuse and resources, launch context, dispatch and overlap, the launcher catalog, runtime launchers (HIP), kernel registry, event queue trace and the embedding launcher', 'one kernel table every engine dispatches through',
   'dispatch.kernel-table/-/-.enginelab', 'src/kernel_registry.cpp, hip_runtime_launchers.hip', tag=True),
 F('runtime', 'exec_program', [r'src/exec/(arena|catalog\.cpp|compiler|descriptor|draft_runner|generic_engine|profiler_cpu|scheduled_stage_graph|step_replay_row|executor_cpu)[^/]*', r'include/strix/exec/.*'],
   'the exec layer (src/exec, include/strix/exec): arena, catalog, cohort programs, compiler, descriptor, draft runner, generic engine, program, step lanes, width graphs and width replay, the CPU executor and profiler and the scheduled stage graph',
   'a backend-neutral program compiler: stage graphs lowered to width-specialised programs and replayed',
   'executor.device-graph/-/-.enginelab', 'strix::exec compiler.hpp, generic_engine.hpp', tag=True),
 F('runtime', 'graph', [r'(src|include/strix)/graph/.*'] + r('graph_', 'serial_graph'),
   'model graphs (src/graph, include/strix/graph): chains, launch plan, model declaration, program segments, rows plan, split view, stage graph and values; graph attention state, eligibility, inputs and operator control; the serial graph (HIP)', 'the declared stage graph a model is written as',
   'build.graph/-/-.enginelab', 'strix::graph stage_graph.hpp', tag=True),
 F('runtime', 'model_graph', [r'src/model_graph/.*'],
   'model-graph adapters (src/model_graph): the model graph and plan, stage bridge, and the DeepSeek V4.1, dense and Qwen adapters', 'binds a model to the stage graph',
   'executor.model-graph/-/-', 'src/model_graph/*.cpp'),
 # ------------------------------------------------------------------ sessions, caches, prefill plans (src root)
 F('runtime', 'session_snapshot', r('session_'),
   'session state: executor handoff, restore, snapshot (memory, transfer, HIP transfer), state cache, layout and memory', 'save, move and restore a decode session (KV, recurrent state, positions)',
   'cache.snapshot-copy/-/-.enginelab', 'src/session_snapshot.cpp, hip_session_snapshot_transfer.cpp', tag=True),
 F('runtime', 'serial_session', r('serial_'),
   'the serial (one-request) decode path: decode reference and replay, host state shadow, prefix cache and lifecycle, request and session', 'the reference decode loop the EXACT gate compares against',
   'executor.decode-loop/-/-.enginelab', 'src/serial_session.cpp, serial_decode_reference.cpp', tag=True),
 F('serve', 'serving_cohort', r('serving_', 'shared_serving', 'greedy_serving', 'cohort_', 'batch_'),
   'serving cohorts and the scheduler: the serving cohort (HIP), greedy backend, prefix cache, shared cohort, greedy serving scheduler, cohort small / step plans, batch nested prefill and batch service handoff',
   'batches concurrent requests into cohorts that decode together (README: Qwen HTTP four slots)',
   'serve.scheduler-cohort/-/-', 'src/greedy_serving_scheduler.cpp, hip_serving_cohort.hip'),
 F('runtime', 'prefix_state', r('prefix_state_cache', 'state_cache', 'context_key', 'block_summary', 'repeat_digests'),
   'prefix state caching: the prefix state cache, state-cache snapshot, context key index, block summary state and repeat digests', 'reuse of a prompt prefix\'s state across requests',
   'cache.prompt/-/-.enginelab', 'src/prefix_state_cache.cpp', tag=True),
 F('runtime', 'known_prompt', r('known_'),
   'known-prompt prefill: cancellation, limits, prefix prefill and runtime, prefix window (HIP), prompt backend and chunks, window cohort, output policy and rows binding', 'prefill of a prompt already known in full, in chunks that can be cancelled',
   'prefill.known-prompt/-/-', 'src/known_prompt_chunks.cpp'),
 F('runtime', 'prefill_plan', r('prefill_', 'attention_split_plan', 'attn_row', 'attn_rowgroup'),
   'prefill planning: dense plan, headless prefill, identity, MoE policy, piece plan, pool, row liveness, stage reservation; the attention split plan, row share and row-group union', 'how a long prompt is cut and scheduled',
   'prefill.plan/-/-', 'include/strix/prefill_piece_plan.hpp'),
 F('runtime', 'handoff', r('handoff', 'final_generation', 'generation_output', 'model_input', 'decode_execution_options', 'decode_numerical_profile', 'kv_storage', 'ctx_exact'),
   'handoff and output: handoff lanes and policy, final generation state export, generation output (HIP, dispatch, diagnostic), model input, decode execution options and numerical profile, KV storage and exact-context memory', 'prefill-to-decode handoff and the output stream',
   'executor.handoff/-/-', 'src/handoff.cpp, hip_generation_output.hip'),
 # ------------------------------------------------------------------ weights and model files
 F('format', 'gguf', r('gguf_', 'ggml_', 'iq4_'),
   'GGUF: reader and writer, ggml dequantisation and weight formats, IQ4_NL and IQ4_XS', 'the GGUF container every Strix Halo model ships in',
   'format.reader/gguf/-.enginelab', 'src/gguf_reader.cpp', tag=True),
 F('format', 'qwen_grids', r('qwen_iq'),
   'the i-quant codebook grids for Qwen (IQ2_S, IQ3_S, IQ3_XXS; from ggml, MIT)', 'grid tables the i-quant readers need', 'format.iq-grid/iq2_s+iq3_s+iq3_xxs/-', 'include/strix/qwen_iq*_grid.inc'),
 F('runtime', 'model_registry', r('model_'),
   'model registry and residency: content hash, contract, copy-residency layout, description, family registry, manifest, mapping accounting, registry, residency (HIP, diagnostic), runtime binding and tensor index', 'one registry the CLI, server and planner read',
   'format.model-registry/-/-', 'src/model_registry.cpp, model_manifest.cpp'),
 F('format', 'weight_files', r('weight_', 'mapped_model_file', 'file_map', 'file_snapshot', 'unbuffered_file', 'asset_manifest', 'bp_codec', 'compact_stage_images', 'rounded_stage_image', 'fnv', 'arithmetic_lineage'),
   'model files: weight archive and pack, mapped model file, file map and snapshot, unbuffered file reads, asset manifest, BP codec, compact and rounded stage images, FNV and arithmetic lineage', 'reading weights from NVMe into UMA without page-cache double copies',
   'format.container/-/-.enginelab', 'src/mapped_model_file.cpp, unbuffered_file.hpp', tag=True),
 F('format', 'weight_formats', [r'(src|include/strix)/weight/formats/.*'],
   'weight formats (src/weight/formats, include/strix/weight/formats): catalog, channel quant, DS41, EXL3, FP8, ggml, GSQ, microscaled (matrix, encode), NVFP4, packed, plain, IEEE half and scalar float', 'every weight encoding the engine reads, one decoder each',
   'format.decode/-/-.enginelab', 'src/weight/formats/*.cpp', tag=True),
 F('format', 'weight_adapters', [r'src/weight/adapters/.*'],
   'per-model tensor naming maps (src/weight/adapters): DS41, Gemma 4, GLM, H3, Hy4 (and DFly), MiniMax, Qwen, vision and the schema', 'the model-specific data the shared loader consumes (AGENTS.md: naming maps as data)',
   'format.layout/-/-.enginelab', 'src/weight/adapters/*.cpp', tag=True),
 F('format', 'weight_readers', [r'src/weight/readers/.*'],
   'weight readers: GGUF, pack and safetensors', 'one reader per container', 'format.reader/gguf+safetensors/-.enginelab-readers', 'src/weight/readers/*.cpp', tag=True),
 F('format', 'hip_weight', [r'src/hip/weight/.*'] + r('q6p_'),
   'device weight sources (src/hip/weight): ggml and GSQ sources, GSQ FP16 and group codecs, GSQ plan and partial trace, trellis sources and partial trace, registry, runtime and source; and the Q6P pack (pack, registry, weights: a 6-bit packed copy that decodes to exactly the target tensor\'s bytes), which ggml_sources.hpp includes', 'weights decoded and uploaded on the device',
   'format.device-loader/-/-.enginelab', 'src/hip/weight/*.cpp', tag=True),
 F('format', 'weight_core', [r'(src|include/strix)/weight/.*'],
   'the shared weight layer (src/weight, include/strix/weight): admission, block draft, cache, calibration moments and provenance, confined files, contracts, decision table, decoder, family, host low-bit (GPTQ) and Q8, image identity, levers, load, low-bit pack, model binding, native runtime, operand cache plan, reference, registry, resident slices, resolve, row pitch, runtime, source, tensor access / launch plan / map, typed slices and types',
   'one loader for every model and backend; models bring only their naming maps',
   'format.loader/-/-.enginelab', 'src/weight/load.cpp, registry.cpp', tag=True),
 # ------------------------------------------------------------------ device, memory, residency, deployment, profiles
 F('runtime', 'device', [r'(src|include/strix)/device/.*'] + r('device_', 'platform_probe', 'hip_gpu_visibility'),
   'devices (src/device, include/strix/device): budget, cost identity (CUDA, HIP), declarations, eager code load, gates (HIP, none), CUDA / HIP / Intel enumerators, link class, low-bit rows, process budget and topology, profiles, runtime host memory, sets and topology, nvidia-smi and xpu-smi readers; device allocation errors, profile and the platform probe',
   'no device, node count or link is compiled in: everything comes from these declarations or a calibration at load (AGENTS.md)',
   'device.info/-/-.enginelab', 'strix::device declaration.hpp, topology.hpp', tag=True),
 F('runtime', 'memory', r('host_', 'pinned_host', 'memory_plan', 'uma_', 'linux_uma', 'runtime_allocator', 'registered_backing', 'owned_layer'),
   'memory: host memory and upload (backend), host phase timing, pinned host buffers, the memory plan, UMA owner inventory and Linux UMA observation, the runtime allocator (HIP), registered backing lifetime and owned layer ordinals', 'one memory owner on 128 GB unified memory: who holds which bytes',
   'device.memory/-/-.enginelab', 'src/runtime_allocator.cpp, hip_runtime_allocator.cpp', tag=True),
 F('runtime', 'residency', [r'(src|include/strix)/residency/.*'],
   'residency (src/residency, include/strix/residency): copy source, direct read, expert residency, replacement and the residency CLI', 'which experts and layers stay resident and how the rest are read in',
   'runtime.weight-prefetch/-/-.enginelab', 'strix::residency expert_residency.hpp', tag=True),
 F('runtime', 'deploy', [r'(src|include/strix)/deploy/.*'],
   'deployment planning (src/deploy, include/strix/deploy): binding, device, footprint, link, memory plan, MiniMax layer cut, plan, route and spec', 'budgets measured device memory for the requested KV context, workspace and weight formats before loading (docs/deployment.md)',
   'serve.memory-plan/-/-.enginelab', 'strix::deploy plan.hpp', tag=True),
 F('runtime', 'levers', r('lever', 'turbo', 'fast_profile', 'engine_profile', 'env_', 'cost_table_cache', 'partition_search') + [r'(src|include/strix)/profiles?/.*'],
   'levers and profiles: the lever registry and cache, turbo, fast and engine profiles, environment arms and flags, the cost-table cache and partition search; src/profile (catalog levers, model presets, multi-device) and include/strix/profile(s) (per-model profile tables)',
   'the EXACT / FAST / TURBO tiers as data: a lever is a registry row with identity, receipts, regime and per-tier default (AGENTS.md)',
   'tune.table/-/-.enginelab', 'include/strix/lever.hpp, src/profile/catalog_levers.cpp', tag=True),
 F('runtime', 'receipts', [r'include/strix/receipt/.*'] + r('raw_baseline_recording', 'offline_'),
   'receipts and offline tuning: the receipt writer, validator, step-profile report and legacy form; raw baseline recording; offline compile provenance, tuning cache and preparation', 'every measured claim comes with a receipt the tools can check',
   'verify.receipt/-/-', 'include/strix/receipt/writer.hpp'),
 F('runtime', 'numeric', [r'include/strix/numeric/.*'],
   'numerics declarations: declared numerics, gfx1151 math and exact rsqrt', 'the bit-level contracts the EXACT profile rests on',
   'verify.numerics-contract/-/-.enginelab', 'include/strix/numeric/*.hpp', tag=True),
 F('runtime', 'state_txn', [r'(src|include/strix)/state/.*'],
   'state declarations and transactions (src/state, include/strix/state): decl, lane view, table and txn', 'declared session state with transactional updates',
   'runtime.state-table/-/-', 'strix::state table.hpp'),
 F('runtime', 'power', [r'(src|include/strix)/power/.*'],
   'the power sampler', 'samples power during benchmarks', 'host.power-sampler/-/-', 'src/power/sampler.cpp'),
 F('runtime', 'support_matrix', [r'(src|include/strix)/support/.*'],
   'the support matrix (models x routes x profiles)', 'what is supported where, read by the CLI and the docs check', 'host.support-matrix/-/-', 'src/support/matrix.cpp'),
 F('format', 'model_data', [r'include/strix/model_data/.*', r'(src|include/strix)/model/.*'],
   'model data: per-model constant tables (DeepSeek V4.1 Flash, GLM-5.3 Flash, Hy4 Flash, Inkling2 Mini, MiniMax M3.1 Flash, Nemotron 3.5 Bolt E4B, Qwen3.8-27B, Qwen3.8-Flash-Next) and the constants / constant profiles', 'model facts as data',
   'format.model-constants/-/-', 'include/strix/model_data/*.hpp'),
 F('runtime', 'host_common', [r'include/strix/(core|hip)/.*'] + r('oracles', 'check', 'preflight', 'setup_quiescence', 'native_setup', 'startup_markers', 'runtime_telemetry', 'build_info', 'msvc_math', 'cuda_', 'qpersist', 'prompt_terminal', 'serve_cli_parity', 'arch_caps'),
   'host common: qpersist and its diagnostic (switches for verify-window fused GDN / QSA tails, bit-identical to the separate launches, and the exact MTP draft-attention kernels), prompt terminal diagnostics, serve / CLI parity; include/strix/core (A8 G32, chunked dot, dense activation, dense reference, dense site policy, rows), include/strix/hip (arch, legacy intrinsics and launch) with the arch caps header they include, oracles, check, preflight, setup quiescence, startup markers, runtime telemetry, the build-info template, the MSVC math builtins and the CUDA spelling headers of the root', 'small shared headers most modules include',
   'kernel-abi.enginelab-common/-/-', 'include/strix/core/*.hpp, include/strix/hip/*.hpp'),

 # ------------------------------------------------------------------ session and serve (include/strix/session, src/session, include/strix/serve, src/serve)
 F('runtime', 'session_kv', [r'(src|include/strix)/session/(kv|prefix_cache|ring_plan|state_image|mark_plan|checkpoint_io|growing_visit|step_replay|recurrent_state)[^/]*'],
   'session KV and state (include/strix/session, src/session): KV, KV capacity, pages, precision and row append, prefix cache, ring plan, state image, mark plan, checkpoint IO, growing visit checkpoints, step replay and recurrent state', 'the engine-neutral KV / state store every engine session uses',
   'kvcache.store/-/bf16.enginelab', 'strix::session kv.hpp, kv_pages.hpp', tag=True),
 F('runtime', 'session_embed', [r'(src|include/strix)/session/embed_[^/]*'],
   'embed rows and splices: embed rows (file), embed splice (host, device, HIP)', 'image / audio rows spliced into the token embedding stream',
   'vision.image-rows/-/bf16.enginelab', 'strix::session embed_splice.hpp', tag=True),
 F('runtime', 'session_engine', [r'(src|include/strix)/session/.*'],
   'the session engine contract (include/strix/session, src/session): backends (CPU, HIP), engine, CLI output, cycle pack (HIP), draft-verify runner, head-gate CLI, linear verify step, parallel jobs, prefill session, prompt windows and the speculation default',
   'the contract an engine implements so the shared serve layer can drive it', 'executor.backend/-/-.enginelab', 'strix::session engine.hpp', tag=True),
 F('serve', 'serve_http', [r'(src|include/strix)/serve/(http_|api|ollama|health|request_fields|json\.|media|session_api)[^/]*'],
   'the HTTP front (include/strix/serve, src/serve): HTTP server and client, the OpenAI-style API and session API, the Ollama API, health, request fields, JSON and media', 'strix-serve\'s wire surface',
   'serve.http/-/-.enginelab', 'strix::serve http_server.hpp, api.hpp', tag=True),
 F('serve', 'serve_tokenizer', [r'(src|include/strix)/serve/(bpe_tokenizer|tokenizer|o200k_|unicode_tables|token_memo)[^/]*'],
   'tokenizers: BPE tokenizer, tokenizer interface and UTF-8, the o200k pre-tokenizer and its Unicode tables, the Unicode tables and the token memo', 'one tokenizer stack for every model',
   'text.tokenizer/-/-.enginelab', 'strix::serve bpe_tokenizer.hpp', tag=True),
 F('serve', 'serve_chat', [r'(src|include/strix)/serve/(chat_template|deepseek_chat|glm_chat|hy4_chat|inkling2_chat)[^/]*'],
   'chat templates: the template engine and kit, and the DeepSeek, GLM, Hy4 and Inkling 2 chat formats', 'rendering a conversation to tokens per model',
   'text.chat-template/-/-.enginelab', 'strix::serve chat_template.hpp', tag=True),
 F('serve', 'serve_constraint', [r'(src|include/strix)/serve/(json_constraint|constraint_selection)[^/]*'],
   'structured output: the JSON constraint and constraint selection', 'JSON-constrained decoding', 'grammar.json-schema/-/-.enginelab', 'strix::serve json_constraint.hpp', tag=True),
 F('serve', 'serve_pflash', [r'(src|include/strix)/serve/(pflash_|prompt_compress|prompt_scorer)[^/]*', r'(src|include/strix)/pflash/.*'],
   'PFlash prompt compression: per-model regions (DeepSeek, GLM, MiniMax, Qwen), the prompt compressor and scorer, the CPU drafter and scorer and the GGUF probe (src/pflash)', 'shrinks a long prompt before prefill by scoring regions with a small drafter',
   'serve.prompt-compress/-/-', 'strix::serve pflash_regions.hpp'),
 F('serve', 'serve_cohort', [r'(src|include/strix)/serve/(cohort_|fixed_kv_cohort|replicated_cohort|batch_sampling|job_sampling|pair_sampling|slot_capacity|row_budget|step_planner|short_first|decode_form|cycle_receipt)[^/]*'],
   'serving scheduler pieces: cohort decode, passes and session, fixed-KV cohort, replicated cohort, batch / job / pair sampling, slot capacity, row budget, the step planner, short-first ordering, decode form and cycle receipts', 'the scheduler that turns requests into decode cycles',
   'serve.scheduler/-/-', 'strix::serve step_planner.hpp, cohort_decode.hpp'),
 F('serve', 'serve_replica', [r'(src|include/strix)/serve/replica_[^/]*'],
   'replica routing: the replica router and proxy', 'Qwen on two nodes as two replicas behind one router (strix_router)', 'serve.replica-router/-/-', 'strix::serve replica_router.hpp'),
 F('serve', 'serve_prefix', [r'(src|include/strix)/serve/(prefix_|prompt_|conversation_keep|queued_prefix|cache_budget_rule|prefill_)[^/]*'],
   'serve-side prompt caching: prefix directory, join and cache-queue binding, prompt join, piece, service and stream, conversation keep, queued-prefix protection, cache budget rule, prefill chunk fallback, group and interleave', 'keeps conversations warm and joins new prompts onto cached prefixes',
   'cache.prompt/-/-.enginelab-serve', 'strix::serve prefix_directory.hpp', tag=True),
 F('serve', 'serve_session', [r'(src|include/strix)/serve/.*'],
   'serve sessions and runners: engine, run, serve config, deployment, session runtime and reservation, serve session, serial session owner and generation CLI, resident plan and session, native handoff, startup phases and warm-up, aux lane and units, file key', 'the engine-neutral serving runner every model plugs into',
   'serve.engine/-/-.enginelab', 'strix::serve engine.hpp, resident_session.hpp', tag=True),
 F('tool', 'cli', [r'(src|include/strix)/cli/.*'],
   'the strix CLI library: cli, detail, doctor, inspection and process', 'strix run / doctor / inspect', 'tool.cli/-/-.enginelab', 'src/cli/cli.cpp'),
 # ------------------------------------------------------------------ shared ops
 F('kernel', 'ops_norm', [r'src/ops/norm/.*'], 'normalisation ops: norm (device), numeric traits, presets, reference, RMS BF16 rows and row reduction', 'the shared RMSNorm family',
   'norm.rmsnorm/-/f32.enginelab', 'src/ops/norm/*.hpp', EXACT, 'any', tag=True),
 F('kernel', 'ops_glu', [r'src/ops/glu/.*'], 'GLU ops: GLU (device), Q8 rows GLU pair and SwiGLU rows', 'the gated FFN activation', 'act.swiglu/-/f32.enginelab', 'src/ops/glu/*.hpp', tag=True),
 F('kernel', 'ops_rope', [r'src/ops/rope/.*'], 'RoPE ops: rope (device) and split-half device', 'rotary embeddings', 'pos.rope/-/f32.enginelab', 'src/ops/rope/*.hpp', tag=True),
 F('kernel', 'ops_a8', [r'src/ops/a8/.*', r'src/ops/chain/.*'], 'A8 activation ops: group and quant device kernels; and the chain prep plan', 'activation quantisation for integer kernels', 'act.quantize/-/a8.enginelab', 'src/ops/a8/*.hpp', tag=True),
 # ------------------------------------------------------------------ Qwen family
 F('executor', 'qwen_prefill', r('qwen_prefill_'),
   'Qwen prefill: balance, chain, handoff, pair and pair staging, pipe dump, pipeline, schedule, stages and trace', 'the Qwen3.8 prompt pipeline, including the two-node pair prefill',
   'executor.prefill/-/-.enginelab-qwen', 'include/strix/qwen_prefill_pipeline.hpp'),
 F('kernel', 'qwen_rotary', r('qwen_rotary', 'qwen_yarn', 'rotary_descriptor'),
   'Qwen rotary: the rotary table (HIP) and diagnostic, the YaRN table and the rotary descriptor', 'long-context rotary scaling', 'pos.rope/-/f32.enginelab-qwen', 'hip_qwen_rotary_table.hip', tag=True),
 F('executor', 'qwen_engine', r('qwen', 'qwen27', 'qwen35') + [r'(src|include/strix)/qwen(27|35)?/.*'],
   'the Qwen engines: Qwen3.8-Flash-Next and Qwen3.8-27B model descriptions, MoE prepared and BF16 router, all-on, profile table, target state digest, weight packs, the Qwen 3.5 config, family, HIP engine and spec costs, pair plan and prefill controls / windows (include/strix/qwen35), the 27B model and the HTTP session owner and service',
   'the binding of Qwen3.8 to the shared cores; the fastest one-node path of the lab (51 tok/s MTP at 32K, 153-155 edit tok/s served)',
   'executor.model/-/-.enginelab-qwen', 'src/qwen35/hip/engine.hip, hip_qwen_moe_prepared.cpp', tag=True),
 # ------------------------------------------------------------------ GLM-5.3 Flash
 F('kernel', 'glm_prefill_wmma', [r'src/glm/prefill/glm_pf_(attn_wmma|kda_wmma|mla_gemm_vopd|mla_wmma|router_wmma|wmma)[^/]*'],
   'GLM WMMA prefill: WMMA attention, KDA, MLA (GEMM VOPD, WMMA), router, and the generic, MoE and x WMMA prefill', 'GLM-5.3 prefill at 8-32K (617 / 595 tok/s on the pair)',
   'linear.prefill/-/f16-f32.enginelab-glm', 'src/glm/prefill/glm_pf_*wmma*.hip', tag=True),
 F('executor', 'glm_prefill', [r'src/glm/prefill/.*'],
   'GLM prefill: HC expand backward, KV-shard prefill, MoE traits, aux, capacity and dense policies, exact (MoE, x), IU4, load admission, MoE, path, variants, workspace and the prefill driver', 'the GLM prompt path around its WMMA kernels',
   'executor.prefill/-/-.enginelab-glm', 'src/glm/prefill/glm_prefill.cpp', requires=['kernel.strix_trellis.c_api']),
 F('executor', 'glm_kvshard', [r'src/glm/engine/glm_kvshard[^/]*'],
   'GLM KV sharding: shard, dump, exchange, layout and math', 'two ranks each hold half the KV by position', 'kvcache.shard/-/bf16.enginelab-glm', 'src/glm/engine/glm_kvshard*.cpp'),
 F('executor', 'glm_engine', [r'src/glm/(engine|host|common)/.*', r'include/strix/glm/.*'] + r('glm_'),
   'the GLM-5.3 engine: engine, embed splice, fast attention, fast INT4 defaults and pack, fast MoE, JSON, library, memory, pack and the R1b generated identity (bitwise equal to the host emulation by construction); host, HC and math headers; the pair profile table',
   'GLM-5.3 Flash on two nodes: 25.7 tok/s plain, 40.4-66.5 with MTP + wide (README)', 'executor.model/-/-.enginelab-glm', 'src/glm/engine/engine.cpp, glm_r1b.hip', EXACT, 'pair', tag=True, requires=['kernel.strix_trellis.c_api']),
 F('kernel', 'glm_kernels', [r'src/glm/kernels/.*'], 'GLM kernels: DSA, norm, tier A and the reduction tree', 'the GLM-specific device kernels (DSA selection, tier-A chains)', 'attn.dsa/-/bf16-f32.enginelab-glm', 'src/glm/kernels/*.hip'),
 F('kernel', 'glm_attn', [r'src/glm/attn/.*'], 'GLM attention: fast attention, tier B and the DSA policy', 'GLM\'s sparse attention decode', 'attn.dsa/-/bf16-f32.enginelab-glm-attn', 'src/glm/attn/*'),
 F('kernel', 'glm_moe', [r'src/glm/moe/.*'], 'GLM MoE and its trellis device kernels', 'GLM routed experts on trellis weights', 'moe.experts-decode/exl3-trellis/f16-f32.enginelab-glm', 'src/glm/moe/*', tag=True),
 F('runtime', 'glm_spec', [r'src/glm/spec/.*'], 'GLM speculation: device, host, merge and run', 'MTP + wide on the GLM pair', 'speculation.window-verify-lossless/-/-.enginelab-glm', 'src/glm/spec/*', tag=True),
 F('serve', 'glm_serve', [r'src/glm/serve/.*'], 'GLM serving: control, import, lead, prefix cache, self-test, vision and the session owner', 'strix-serve for the GLM pair', 'serve.engine/-/-.enginelab-glm', 'src/glm/serve/*', tag=True),
 F('runtime', 'glm_load', [r'src/glm/load/.*'], 'GLM loading: model description, peer load, rank plan (graph, main)', 'which rank loads which layers', 'format.loader/-/-.enginelab-glm', 'src/glm/load/*', tag=True),
 F('runtime', 'glm_xfer', [r'src/glm/xfer/.*'], 'GLM exchange: post exchange and fast post, sequence parallel, traits, the transfer (device, exchange) and its bench', 'GLM\'s pair exchange over the shared channel', 'transport.glm-exchange/-/-', 'src/glm/xfer/*'),
 F('executor', 'glm_vision', [r'src/glm/vision/.*'], 'GLM vision', 'image input for GLM', 'vision.splice/-/bf16.enginelab-glm', 'src/glm/vision/*'),
 F('runtime', 'glm_rtxpf', [r'src/glm/rtxpf/.*'], 'GLM RTX prefill route', 'prefill on an attached RTX PRO', 'runtime.hetero-route/-/-.enginelab-glm', 'src/glm/rtxpf/*'),
 # ------------------------------------------------------------------ MiniMax M3.1 Flash
 F('kernel', 'minimax_attn', [r'src/minimax/attn/.*'] + r('minimax_msa'), 'MiniMax attention: index projection, MSA fused, flash policy, in-block layout, norm-rope and select dump; the MSA (HIP) entry', 'MiniMax\'s sparse attention (docs/minimax-msa.md)', 'attn.msa/-/bf16-f32.enginelab-minimax', 'src/minimax/attn/*, hip_minimax_msa.hip'),
 F('kernel', 'minimax_moe', [r'src/minimax/moe/.*'] + r('minimax_moe'), 'MiniMax MoE: mmx MoE and traits, prefill (core, IU4) and trellis device; the MoE (HIP) entry', 'MiniMax routed experts', 'moe.experts-decode/exl3-trellis+iu4/f16-f32.enginelab-minimax', 'src/minimax/moe/*', tag=True),
 F('runtime', 'minimax_pfx', [r'src/minimax/pfx/.*'], 'MiniMax prefill exchange: CRC, exchange, route and wire', 'the prefill table ring over the pair (docs/minimax-prefill-table-ring.md)', 'transport.minimax-prefill/-/-', 'src/minimax/pfx/*'),
 F('runtime', 'minimax_rtxpf', [r'src/minimax/rtxpf/.*'], 'MiniMax RTX FP32 identity (device, reference, tests)', 'the RTX prefill route with an FP32 identity', 'runtime.hetero-route/-/-.enginelab-minimax', 'src/minimax/rtxpf/*'),
 F('runtime', 'minimax_session', [r'src/minimax/session/.*'], 'MiniMax session runtime and cache', 'MiniMax decode sessions', 'cache.snapshot-copy/-/-.enginelab-minimax', 'src/minimax/session/*', tag=True),
 F('runtime', 'minimax_xfer', [r'src/minimax/xfer/.*'], 'MiniMax exchange: traits, transfer and device transfer', 'the pair exchange measured at 2.91 -> 1.89 ms a token', 'transport.minimax-exchange/-/-', 'src/minimax/xfer/*'),
 F('executor', 'minimax_dspark', [r'src/minimax/dspark/.*'], 'MiniMax DSpark: HIP DSpark, Markov candidates, auto mode, DSpark files and host, n-gram drafter, reference and tests', 'DSpark auto on MiniMax: 23.3 -> 32.4 code tok/s (README)', 'draft.dspark/-/f32.enginelab-minimax', 'src/minimax/dspark/*', tag=True),
 F('serve', 'minimax_serve', [r'src/minimax/serve/.*'], 'MiniMax serving: API, chat, control, DSpark policy, lead, observe, scheduling, session, tokenizer, paired state, serve runner and runtime, session engine and service, fixtures', 'strix-serve for the MiniMax pair', 'serve.engine/-/-.enginelab-minimax', 'src/minimax/serve/*', tag=True),
 F('executor', 'minimax_engine', [r'src/minimax/.*', r'include/strix/minimax/.*'] + r('minimax_'),
   'the MiniMax M3.1 engine: deployment, linear drafter, engine, common, defaults, fast INT4, kernels, pack, profile table, prefill timeline, route identity; model assets, description and graph; include/strix/minimax; the layer cut', 'MiniMax M3.1 Flash on two nodes (24.2 tok/s 1K, 621.6 prefill at 8K)',
   'executor.model/-/-.enginelab-minimax', 'src/minimax/engine/engine.cpp', tag=True, requires=['kernel.strix_trellis.c_api']),

 # ------------------------------------------------------------------ src/gemma4 (the Gemma-4-style engine that serves Nemotron 3.5 Bolt E4B; dflash.hpp: "Nemotron Bolt E4B DFlash"), H3 (MiniMax H3 audio / video / image), Hy4, Inkling 2
 F('executor', 'gemma4_prefill', [r'src/gemma4/prefill/.*'], 'Gemma 4 prefill (src/gemma4/prefill)', 'the Gemma 4 prompt path', 'executor.prefill/-/-.enginelab-gemma4', 'src/gemma4/prefill/*'),
 F('runtime', 'gemma4_dflash', [r'src/gemma4/dflash/.*'], 'the DFlash drafter of Nemotron 3.5 Bolt E4B (host reference in the declared arithmetic g4-dflash-v1.1: five pre-norm layers projecting committed target taps into K/V, a block of draft rows attended non-causally) (src/gemma4/dflash)', 'block drafting for the Bolt E4B engine; the file says it is pending device and full-workload qualification', 'draft.dflash/-/f32.enginelab-gemma4', 'src/gemma4/dflash/*', tag=True),
 F('executor', 'gemma4_model', [r'src/gemma4/model/.*'], 'Gemma 4 model description and layers (src/gemma4/model)', 'the model binding', 'executor.model/-/-.enginelab-gemma4', 'src/gemma4/model/*', tag=True),
 F('kernel', 'gemma4_kernels', [r'src/gemma4/kernels/.*'], 'Gemma 4 device kernels (src/gemma4/kernels)', 'its device kernels', 'kernel-abi.enginelab-gemma4/-/-', 'src/gemma4/kernels/*'),
 F('runtime', 'gemma4_ref', [r'src/gemma4/ref/.*'], 'Gemma 4 host references (src/gemma4/ref)', 'references for the EXACT gate', 'verify.kernel-reference/-/-.enginelab-gemma4', 'src/gemma4/ref/*'),
 F('executor', 'gemma4_features', [r'src/gemma4/features/.*'], 'Gemma 4 features (per-layer inputs, media features) (src/gemma4/features)', 'model features', 'executor.features/-/-.enginelab-gemma4', 'src/gemma4/features/*'),
 F('executor', 'gemma4_engine', [r'src/gemma4/engine/.*'], 'the Gemma 4 engine (src/gemma4/engine)', 'the engine driver', 'executor.engine/-/-.enginelab-gemma4', 'src/gemma4/engine/*'),
 F('runtime', 'gemma4_eagle', [r'src/gemma4/eagle/.*'], 'Gemma 4 EAGLE / MTP drafter (with the drafter export script) (src/gemma4/eagle)', 'EAGLE-style drafting', 'draft.eagle/-/f32.enginelab-gemma4', 'src/gemma4/eagle/*'),
 F('runtime', 'gemma4_kv', [r'src/gemma4/kv/.*'], 'Gemma 4 KV cache (sliding and global layers) (src/gemma4/kv)', 'its KV layout', 'kvcache.store/-/bf16.enginelab-gemma4', 'src/gemma4/kv/*', tag=True),
 F('kernel', 'gemma4_exl3', [r'src/gemma4/exl3/.*'], 'Gemma 4 EXL3 device decode (with its notices) (src/gemma4/exl3)', 'EXL3 weights for Gemma 4', 'linear.decode/exl3-trellis/f16-f32.enginelab-gemma4', 'src/gemma4/exl3/*', tag=True),
 F('runtime', 'gemma4_contracts', [r'src/gemma4/contracts/.*'], 'Gemma 4 numeric contracts (src/gemma4/contracts)', 'its declared contracts', 'verify.numerics-contract/-/-.enginelab-gemma4', 'src/gemma4/contracts/*', tag=True),
 F('serve', 'gemma4_serve', [r'src/gemma4/serve/.*'], 'Gemma 4 serving (with the Unicode licence notice) (src/gemma4/serve)', 'strix-serve for Gemma 4', 'serve.engine/-/-.enginelab-gemma4', 'src/gemma4/serve/*', tag=True),
 F('kernel', 'h3_vae', [r'src/h3/vae/.*'], 'H3 image VAE (src/h3/vae)', 'the decoder of MiniMax-H3-Image', 'media.vae/-/f32.enginelab-h3', 'src/h3/vae/*'),
 F('executor', 'h3_te', [r'src/h3/te/.*'], 'H3 text encoder (src/h3/te)', 'the text encoder of the image model', 'media.text-encoder/-/f32.enginelab-h3', 'src/h3/te/*'),
 F('kernel', 'h3_attn', [r'src/h3/attn/.*'], 'H3 attention (src/h3/attn)', 'diffusion transformer attention', 'attn.dit/-/bf16-f32.enginelab-h3', 'src/h3/attn/*'),
 F('runtime', 'h3_common', [r'src/h3/common/.*'], 'H3 common headers (src/h3/common)', 'shared H3 pieces', 'kernel-abi.enginelab-h3/-/-', 'src/h3/common/*'),
 F('runtime', 'h3_levers', [r'src/h3/levers/.*'], 'H3 levers (sparse attention study and others) (src/h3/levers)', 'its FAST levers', 'tune.table/-/-.enginelab-h3', 'src/h3/levers/*', tag=True),
 F('runtime', 'h3_dist', [r'src/h3/dist/.*'], 'H3 distributed (two-node) pieces (src/h3/dist)', 'H3 on a pair', 'runtime.distributed/-/-.enginelab-h3', 'src/h3/dist/*'),
 F('executor', 'h3_block', [r'src/h3/block/.*'], 'H3 transformer blocks (src/h3/block)', 'the DiT blocks', 'executor.block/-/-.enginelab-h3', 'src/h3/block/*'),
 F('kernel', 'h3_gemm', [r'src/h3/gemm/.*'], 'H3 GEMMs (src/h3/gemm)', 'its GEMM kernels', 'gemm.wmma-tile/-/f16-f32.enginelab-h3', 'src/h3/gemm/*', tag=True),
 F('executor', 'h3_model', [r'src/h3/.*'], 'the H3 model root (src/h3)', 'MiniMax H3 audio + video and MiniMax-H3-Image (text to image), opt-in builds', 'executor.model/-/-.enginelab-h3', 'src/h3/*', tag=True),
 F('kernel', 'hy4_hip', [r'src/hy4/hip/.*'], 'Hy4 HIP forward, plan and rows', 'Hy4-flash on gfx1151 (16.8 tok/s one node, 1,270 prefill on the pair)', 'executor.forward/-/-.enginelab-hy4', 'src/hy4/hip/*', tag=True),
 F('runtime', 'hy4_ref', [r'src/hy4/ref/.*'], 'Hy4 host references (DFly and host)', 'the BASE reference of Hy4 FAST arms', 'verify.kernel-reference/-/-.enginelab-hy4', 'src/hy4/ref/*'),
 F('executor', 'hy4_engine', [r'src/hy4/.*', r'include/strix/hy4/.*'], 'the Hy4 engine root and serving (session service, include/strix/hy4)', 'Hy4-flash binding', 'executor.model/-/-.enginelab-hy4', 'src/hy4/*', tag=True),
 F('executor', 'inkling2_cuda', [r'src/inkling2/[^/]*_cuda[^/]*'], 'Inkling 2 Mini CUDA pieces (attention, block candidates, controller operands, dense, MoE, program, serve)', 'the RTX route of Inkling 2 Mini', 'executor.model/-/-.enginelab-inkling2-cuda', 'src/inkling2/*_cuda.cu'),
 F('kernel', 'inkling2_attention', [r'src/inkling2/inkling2_mini_attention[^/]*'], 'Inkling 2 Mini attention (HIP, operands, prepare)', 'its attention', 'attn.softmax-gqa/-/bf16-f32.enginelab-inkling2', 'src/inkling2/inkling2_mini_attention*', tag=True),
 F('kernel', 'inkling2_moe', [r'src/inkling2/inkling2_mini_(moe|expert)[^/]*'], 'Inkling 2 Mini MoE and experts (HIP, dispatch, reference, layout)', 'its experts', 'moe.experts-decode/-/f16-f32.enginelab-inkling2', 'src/inkling2/inkling2_mini_moe*', tag=True),
 F('runtime', 'inkling2_dspark', [r'src/inkling2/inkling2_mini_(dspark|resident_dspark)[^/]*'], 'Inkling 2 Mini DSpark (checkpoint, context, reference, taps, resident)', 'DSpark drafting for Inkling 2', 'draft.dspark/-/f32.enginelab-inkling2', 'src/inkling2/inkling2_mini_dspark*', tag=True),
 F('executor', 'inkling2_resident', [r'src/inkling2/inkling2_mini_resident[^/]*'], 'Inkling 2 Mini resident backend, decoder, forward, route and weights', 'the all-resident decode of Inkling 2', 'executor.forward/-/-.enginelab-inkling2', 'src/inkling2/inkling2_mini_resident*', tag=True),
 F('serve', 'inkling2_serve', [r'src/inkling2/inkling2_mini_serve[^/]*'], 'Inkling 2 Mini serving (device, HIP, install, options)', 'strix-serve for Inkling 2', 'serve.engine/-/-.enginelab-inkling2', 'src/inkling2/inkling2_mini_serve*', tag=True),
 F('executor', 'inkling2_engine', [r'src/inkling2/.*'], 'the Inkling 2 Mini engine: calibration, checkpoint, config, controller and residual device code, dense HIP, forward, host, messages, prefill rows, schedule, state, weights and the vision tower', 'Inkling 2 Mini binding (catalog/ciru_inkling has another engine for it)', 'executor.model/-/-.enginelab-inkling2', 'src/inkling2/*', tag=True),
 F('executor', 'vision', [r'(src|include/strix)/vision/.*'], 'the shared vision tower (src/vision, include/strix/vision): API media, decode, embed cache and spans, encoder (CUDA, layout, plan), GGUF weights, host math, image, lookup weights, patchify, preprocess, projector, reference, resize and serial media', 'one vision encoder every vision model binds', 'vision.tower/-/bf16.enginelab', 'strix::vision encoder.hpp', tag=True),
 # ------------------------------------------------------------------ other backends
 F('runtime', 'backend_cpu', [r'src/cpu/.*'], 'the CPU backend (src/cpu: arch, core, weight)', 'a CPU route and reference for the same contracts', 'executor.lane-backend/-/-.enginelab-cpu', 'src/cpu/*', tag=True),
 F('runtime', 'backend_intel', [r'src/intel/.*'], 'the Intel GPU backend (src/intel: arch, core, weight)', 'an XPU route', 'executor.lane-backend/-/-.enginelab-intel', 'src/intel/*', tag=True),
 F('runtime', 'backend_metal', [r'src/metal/.*'], 'the Metal backend core (src/metal/core)', 'an Apple GPU route', 'executor.lane-backend/-/-.enginelab-metal', 'src/metal/*', tag=True),
 F('runtime', 'backend_vulkan', [r'src/vulkan/.*'], 'the Vulkan backend core (src/vulkan/core)', 'a Vulkan route', 'executor.lane-backend/-/-.enginelab-vulkan', 'src/vulkan/*', tag=True),
 # ------------------------------------------------------------------ CUDA routes (RTX PRO 6000; not built on the pod)
 F('kernel', 'cuda_attn', [r'src/cuda/core/(attn|attention|rowgroup)[^/]*'], 'CUDA attention: decode, flash (8-bit, split decode), lanes, production, row-group launch and union, selected blocks, tiles', 'the RTX attention route', 'attn.softmax-gqa/-/bf16-f32.enginelab-cuda', 'src/cuda/core/attn_*.cu', tag=True),
 F('kernel', 'cuda_moe', [r'src/cuda/core/(moe_|route_|grouped_w4a4)[^/]*'], 'CUDA MoE: A8 (MMA), microscaled R1 (step, pipe), R1 route plan, turbo I8, route project / select / softmax, grouped W4A4', 'the RTX MoE route', 'moe.experts-decode/-/f16-f32.enginelab-cuda', 'src/cuda/core/moe_*.cu', tag=True),
 F('kernel', 'cuda_recur', [r'src/cuda/core/recur[^/]*'], 'CUDA recurrence: chunk, decode, lanes, math, norm, parity and plan', 'GDN on the RTX route', 'mixer.gdn/-/f32.enginelab-cuda', 'src/cuda/core/recur_*.cu', tag=True),
 F('kernel', 'cuda_select', [r'src/cuda/core/select[^/]*'], 'CUDA selection: catalog, pick body, ReLU schedules, rows, score and segments', 'sparse attention selection on CUDA', 'attn.indexer-topk/-/f32.enginelab-cuda', 'src/cuda/core/select_*.cu', tag=True),
 F('kernel', 'cuda_head', [r'src/cuda/core/head_[^/]*'], 'CUDA heads: argmax (final, partial), emit and screened head', 'the RTX greedy head', 'logits.argmax/-/f32.enginelab-cuda', 'src/cuda/core/head_*.cu', tag=True),
 F('kernel', 'cuda_nvfp4', [r'src/cuda/core/nvfp4[^/]*'], 'CUDA NVFP4: NVFP4 and NVFP4 x A8 with references and tiles', 'Blackwell FP4 weights', 'linear.decode/nvfp4/f16-f32.enginelab-cuda', 'src/cuda/core/nvfp4*.cu', tag=True),
 F('kernel', 'cuda_mix', [r'src/cuda/core/mix[^/]*'], 'CUDA hyper-connection mix: body, latency plan, transition', 'HC on the RTX route', 'hc.mix/-/f32.enginelab-cuda', 'src/cuda/core/mix_*.cu', tag=True),
 F('kernel', 'cuda_dense', [r'src/cuda/core/(dense|imma|mma_tile|native_mma|native_reuse)[^/]*'], 'CUDA dense: dense, IMMA dense and tiles, MMA tiles, native MMA and reuse', 'dense projections on tensor cores', 'linear.dense/-/f16-f32.enginelab-cuda', 'src/cuda/core/dense/*', tag=True),
 F('kernel', 'cuda_core_common', [r'src/cuda/core/.*', r'include/strix/cuda_core/.*', r'src/cuda/(arch|weight)/.*', r'src/cuda/[^/]*'], 'the CUDA core rest: AMD math tables, common, reduce, PTX, norms, chain, step inputs / rows / row store, trellis, turbo I8, vision, embed splice, lookup, token sampling, weight uploader, arch (SM120) and weight; include/strix/cuda_core', 'the shared CUDA kernel home (AGENTS.md: kernel bodies per backend in src/cuda/core)', 'kernel-abi.enginelab-cuda/-/-', 'src/cuda/core/*'),
 F('executor', 'cuda_ds41', [r'src/cuda/models/ds41/.*', r'include/strix/cuda/ds41_[^/]*'], 'DeepSeek V4.1 on CUDA (src/cuda/models/ds41, include/strix/cuda/ds41_*)', 'the RTX DS41 route', 'executor.model/-/-.enginelab-ds41-cuda', 'src/cuda/models/ds41/*', tag=True),
 F('executor', 'cuda_qwen', [r'src/cuda/models/qwen/.*', r'include/strix/cuda/qwen_[^/]*', r'src/qwen35/cuda/.*'], 'Qwen on CUDA (src/cuda/models/qwen, include/strix/cuda/qwen_*, src/qwen35/cuda)', 'the RTX Qwen route (85-96 tok/s serial, 600+ with MTP + wide on JSON)', 'executor.model/-/-.enginelab-qwen-cuda', 'src/cuda/models/qwen/*', tag=True),
 F('executor', 'cuda_minimax', [r'src/cuda/models/minimax/.*'], 'MiniMax on CUDA', 'the RTX MiniMax route', 'executor.model/-/-.enginelab-minimax-cuda', 'src/cuda/models/minimax/*', tag=True),
 F('executor', 'cuda_glm', [r'src/cuda/models/glm/.*'], 'GLM on CUDA (with its producer and bench)', 'the RTX GLM route', 'executor.model/-/-.enginelab-glm-cuda', 'src/cuda/models/glm/*', tag=True),
 F('runtime', 'cuda_exec', [r'src/exec/.*', r'include/strix/cuda/.*'], 'the CUDA exec layer: catalog decode, step rows and WR1 / WR2, bound transfers, decode epilogue, lowering and transition, program recipe, the CUDA executor and profiler; include/strix/cuda (common)', 'compiled CUDA programs for the RTX route', 'executor.device-graph/-/-.enginelab-cuda', 'src/exec/*cuda*', tag=True),

 # ------------------------------------------------------------------ lab/ (one module a lab)
 F('kernel', 'lab_basic_kernels', [r'lab/basic_kernels/.*'], 'the basic HIP kernels lab (vector ops, reductions) with its CMake and README (lab/basic_kernels)', 'the first kernels brought up on gfx1151', 'kernel-research.enginelab-lab-basic-kernels/-/-', 'lab/basic_kernels'),
 F('tool', 'lab_cross_compile', [r'lab/cross_compile/.*'], 'the cross-compile probe: a HIP probe and its prebuilt gfx1151 code object and object (lab/cross_compile/out) (lab/cross_compile)', 'checks offline compilation for gfx1151', 'tool.enginelab-lab-cross-compile/-/-', 'lab/cross_compile'),
 F('tool', 'lab_direct_gpu_stream', [r'lab/direct_gpu_stream/.*'], 'the direct GPU stream commissioning lab: bounded TCP control, observe driver, visibility contract, kernels and campaign, commissioning and visibility notes, the kernel side and tests (lab/direct_gpu_stream)', 'commissioning of the USB4 GPU-direct path before production', 'tool.enginelab-lab-direct-gpu-stream/-/-', 'lab/direct_gpu_stream'),
 F('tool', 'lab_gpu_queue', [r'lab/gpu_queue/.*'], 'the GPU queue lab: native queue main and memory (lab/gpu_queue)', 'user-mode GPU queue experiments', 'tool.enginelab-lab-gpu-queue/-/-', 'lab/gpu_queue'),
 F('tool', 'lab_gpu_visibility', [r'lab/gpu_visibility/.*'], 'the GPU visibility lab (lab/gpu_visibility)', 'proves iGPU memory written by the NIC is visible to kernels', 'tool.enginelab-lab-gpu-visibility/-/-', 'lab/gpu_visibility'),
 F('tool', 'lab_gsq', [r'lab/gsq/.*'], 'the GSQ lab: content identity, tensor dequantisation and a dequant comparison against llama.cpp (lab/gsq)', 'GSQ format checks', 'tool.enginelab-lab-gsq/-/-', 'lab/gsq'),
 F('kernel', 'lab_integer_projection', [r'lab/integer_projection/.*'], 'the integer projection lab kernel with offline compile scripts (lab/integer_projection)', 'the INT projection kernel before it moved to src', 'kernel-research.enginelab-lab-integer-projection/-/-', 'lab/integer_projection'),
 F('kernel', 'lab_low_precision', [r'lab/low_precision/.*'], 'the low-precision lab: intrinsic probe, compile probe and host reference (lab/low_precision)', 'gfx1151 low-precision intrinsics', 'kernel-research.enginelab-lab-low-precision/-/-', 'lab/low_precision'),
 F('kernel', 'lab_moe_experts', [r'lab/moe_experts/.*'], 'the MoE experts lab kernel (lab/moe_experts)', 'grouped expert kernel prototype', 'kernel-research.enginelab-lab-moe-experts/-/-', 'lab/moe_experts'),
 F('kernel', 'lab_moe_iq4_xs', [r'lab/moe_iq4_xs/.*'], 'the MoE IQ4_XS lab kernel (lab/moe_iq4_xs)', 'IQ4_XS expert prototype', 'kernel-research.enginelab-lab-moe-iq4-xs/-/-', 'lab/moe_iq4_xs'),
 F('kernel', 'lab_moe_q8_0', [r'lab/moe_q8_0/.*'], 'the MoE Q8_0 lab kernel with its qualification script and test (lab/moe_q8_0)', 'Q8_0 expert prototype', 'kernel-research.enginelab-lab-moe-q8-0/-/-', 'lab/moe_q8_0'),
 F('kernel', 'lab_moe_routing', [r'lab/moe_routing/.*'], 'the MoE routing lab: flat and hierarchical routing kernels with qualification (lab/moe_routing)', 'router prototypes', 'kernel-research.enginelab-lab-moe-routing/-/-', 'lab/moe_routing'),
 F('tool', 'lab_native_campaign', [r'lab/native_campaign/.*'], 'the native numerical campaign lab (lab/native_campaign)', 'numerical measurement campaigns', 'tool.enginelab-lab-native-campaign/-/-', 'lab/native_campaign'),
 F('kernel', 'lab_packed_projection', [r'lab/packed_projection/.*'], 'the packed projection lab: IQ4_XS and Q8 projections (lab/packed_projection)', 'packed GEMV prototypes', 'kernel-research.enginelab-lab-packed-projection/-/-', 'lab/packed_projection'),
 F('kernel', 'lab_q4k_a8', [r'lab/q4k_a8/.*'], 'the Q4_K x A8 dot4 compile probe (lab/q4k_a8)', 'the dot4 instruction path for Q4K-A8', 'kernel-research.enginelab-lab-q4k-a8/-/-', 'lab/q4k_a8'),
 F('tool', 'lab_qwen_output_split', [r'lab/qwen_output_split/.*'], 'the Qwen output split lab: native target, host tests (lab/qwen_output_split)', 'splitting the Qwen output head', 'tool.enginelab-lab-qwen-output-split/-/-', 'lab/qwen_output_split'),
 F('kernel', 'lab_recurrent_gdn', [r'lab/recurrent_gdn/.*'], 'the recurrent GDN lab kernel (lab/recurrent_gdn)', 'GDN recurrence prototype', 'kernel-research.enginelab-lab-recurrent-gdn/-/-', 'lab/recurrent_gdn'),
 F('kernel', 'lab_sparse_lookup', [r'lab/sparse_lookup/.*'], 'the sparse lookup lab (lab/sparse_lookup)', 'sparse lookup prototype', 'kernel-research.enginelab-lab-sparse-lookup/-/-', 'lab/sparse_lookup'),
 F('tool', 'lab_transport', [r'lab/transport/.*'], 'the transport lab (lab/transport)', 'transport bring-up experiments', 'tool.enginelab-lab-transport/-/-', 'lab/transport'),
 F('tool', 'lab_uma_residency', [r'lab/uma_residency/.*'], 'the UMA residency lab (lab/uma_residency)', 'UMA residency measurements', 'tool.enginelab-lab-uma-residency/-/-', 'lab/uma_residency'),

 # ------------------------------------------------------------------ apps/ (the programs)
 F('tool', 'apps_ds41_cuda', [r'apps/cuda_ds41_[^/]*'], 'the CUDA DeepSeek V4.1 programs: decode, hot decode and replay, multi front, parts host check, pipeline front, prefill (service), replica (budget, serve), session cache and draft', 'the RTX DS41 binaries', 'tool.enginelab-ds41-cuda/-/-', 'apps/cuda_ds41_*'),
 F('tool', 'apps_qwen_cuda', [r'apps/cuda_qwen[^/]*'], 'the CUDA Qwen programs: Qwen 3.5 cohort, decode and spec, logits, prefill service, prefix cache, serve and policy', 'the RTX Qwen binaries', 'tool.enginelab-qwen-cuda/-/-', 'apps/cuda_qwen*'),
 F('tool', 'apps_ds41', [r'apps/(deepseek41_|ds41_)[^/]*'], 'the DeepSeek V4.1 programs: batch, deploy, DSpark (rank, replay), expert server, inspect, lockstep, long context, pipeline front and replay, prefill (continue, quality, resident, pair), prefix, quality arms, rank, requant eval, serve, trimmed, engram pages, hetero stage, prepare load, read bench, state send and XS probes', 'the DS41 pair binaries (deepseek41_rank_main is one rank of the two-node path)', 'tool.enginelab-ds41/-/-', 'apps/deepseek41_*.cpp'),
 F('tool', 'apps_glm', [r'apps/glm_[^/]*'], 'the GLM programs: auto rows, deploy (owners, provider), model facts, prefill main, rank (defaults, receipt, startup, transport), route levers, serve and vision', 'the GLM pair binaries', 'tool.enginelab-glm/-/-', 'apps/glm_rank_main.cpp'),
 F('tool', 'apps_minimax', [r'apps/minimax_[^/]*'], 'the MiniMax programs: layer cut, rank and serve', 'the MiniMax pair binaries', 'tool.enginelab-minimax/-/-', 'apps/minimax_rank_main.cpp'),
 F('tool', 'apps_gemma4', [r'apps/gemma4_[^/]*'], 'the Gemma 4 programs: decode, gate, prefill A/B, serve and tokenize', 'Gemma 4 binaries', 'tool.enginelab-gemma4/-/-', 'apps/gemma4_*'),
 F('tool', 'apps_hy4', [r'apps/hy4_[^/]*'], 'the Hy4 programs: DFly, rank, reference and rows', 'Hy4 binaries', 'tool.enginelab-hy4/-/-', 'apps/hy4_*'),
 F('tool', 'apps_inkling2', [r'apps/inkling2_[^/]*'], 'the Inkling 2 Mini programs: CLI, CPU, CUDA, GPU main, HIP, inspect and media', 'Inkling 2 binaries', 'tool.enginelab-inkling2/-/-', 'apps/inkling2_mini_*'),
 F('tool', 'apps_h3', [r'apps/h3_[^/]*'], 'the H3 image program', 'text to image', 'tool.enginelab-h3/-/-', 'apps/h3_image_main.cpp'),
 F('tool', 'apps_mtp', [r'apps/(hip_mtp_|mtp_|hip_qwen_prefill_mtp)[^/]*'], 'MTP application pieces: the MTP cohort bridge and tiled cohort, the decode bridge, wide copy verify (a long confident copy span verified in ONE rows-row forward of the tiled prefill tail owner instead of 8-row windows, state snapshotted first), the prefill-to-MTP handoff, MTP asset verification, manifest audit, prompt phase and window memory reservation', 'where MTP wide drafting is wired into the decode CLI and server', 'speculation.window-verify-wide/-/-.apps', 'apps/hip_mtp_wide_copy.hpp'),
 F('tool', 'apps_qwen', [r'apps/(qwen_|hip_qwen35_)[^/]*'], 'the Qwen programs and headers: Qwen 3.5 decode, pair and spec mains, cache admission and family contract, chain plan, deploy and provider, effective config, HTTP memory, description hook, MoE direct-B and BF16 route, MTP bundle and owner plan, pair transport, prefill external memory / options / plan, prompt organization, rotary capabilities, serial prompt certificate, service health, tiled prompt plan and the YaRN admission / memory / owner inventory', 'the Qwen one-node and replica binaries', 'tool.enginelab-qwen/-/-', 'apps/hip_qwen35_decode_main.hip'),
 F('tool', 'apps_hip_decode', [r'apps/hip_[^/]*'], 'the HIP decode program: decode token entry and main, decode experiment, options and receipt JSON, deferred MoE status, HTTP serve, model residency main, prefix state cache, runtime allocator main and session store', 'strix run\'s HIP decode path', 'tool.generate-ids/-/-.enginelab', 'apps/hip_decode_token_main.cpp', tag=True),
 F('tool', 'apps_strix', [r'apps/(strix|tokenize_main|support_matrix_main|platform_probe_main|device_plan_main|memory_plan_main|model_|inspect_gguf|gguf_pack_check|execution_|serial_decode_reference_main)[^/]*'], 'the strix front programs: strix, strix-serve, strix-router, strix-bench, model describe, inspection bindings, tokenize, support matrix, platform probe, device and memory plans, model manifest, GGUF inspect and pack check, execution tensor audit and inventory receipt, serial decode reference', 'the user-facing binaries', 'tool.launcher/-/-.enginelab', 'apps/strix_main.cpp, strix_serve_main.cpp', tag=True),
 F('tool', 'apps_dist', [r'apps/(dist_|xfer_probe)[^/]*'], 'transport probes: GPU, host exchange bench, TCP probe and the transfer probe', 'bring-up of the pair link', 'tool.enginelab-dist-probe/-/-', 'apps/dist_*_main.cpp'),
 F('tool', 'apps_serve_common', [r'apps/.*'], 'shared application headers: TP2 resident (launch environment snapshot of every lever, 8-field TP2 agreement and per-slot barrier), HTTP session completion and slot capacity, known-prefill reservation, PLE reclaim mapping and resident fallback, prefix-state cache identity, QSA input receipt, RTX state-import routes, runtime owner inventory and shared prefetch receipts, state cache stats, bitcalc measurement, deferred MoE receipt and the offline tuning build context template', 'the glue the resident TP2 servers share', 'serve.resident-tp2/-/-', 'apps/resident_tp2.hpp'),
 F('runtime', 'test_hooks', [re.escape(t) for t in TEST_HOOKS], 'the three tests/ headers that harvested sources include: the Qwen rotary watchdog and fixtures (included by hip_indexed_attention_launcher.hip) and the MTP model qualification (included by apps/hip_decode_token_main.cpp)', 'without them two compile units cannot build as harvested; the rest of tests/ stays unharvested', 'verify.kernel-selftest/-/-.enginelab', 'tests/hip_qwen_rotary_watchdog.hpp, tests/hip_mtp_model_qualification.hpp', tag=True),
 F('runtime', 'stb_image', [r'third_party/.*'], 'stb_image.h (Sean Barrett; MIT or public domain), vendored', 'image decoding for the vision paths', 'media.image-decode/-/u8.enginelab', 'third_party/stb/stb_image.h', 'bytes to RGB', 'any image input', tag=True),
]


# ---------------------------------------------------------------------------------------------- assembly
AUTHORS = ['hyukudan', 'strix-halo-engine-lab contributors']
LICENCE_FILES = ['LICENSE', 'LICENSES/ISA-L-CRC32C.txt', 'LICENSES/stb_image.txt', 'LICENSES/strix-trellis-Apache-2.0.txt']
SPDX_EXTRA = {   # module -> spdx expression where vendored third-party text sits beside the Apache-2.0 code
    'runtime.enginelab.xport_crc': 'Apache-2.0 AND BSD-3-Clause',      # src/dist_crc32c_vector.hpp: Intel ISA-L CRC32C (LICENSES/ISA-L-CRC32C.txt)
    'format.enginelab.qwen_grids': 'Apache-2.0 AND MIT',               # the i-quant grids come from ggml
    'kernel.enginelab.ds41_rows': 'Apache-2.0 AND MIT',                # include/strix/ds41_iq2_xxs_grid.inc, from ggml
    'runtime.enginelab.stb_image': 'MIT OR Unlicense',                 # stb_image.h's own dual licence (LICENSES/stb_image.txt)
    'serve.enginelab.gemma4_serve': 'Apache-2.0 AND Unicode-3.0',      # notices/UNICODE-LICENSE.txt covers its Unicode tables
}
NOT_TIMED = 'Nothing of this module was run or timed on the pod (card bc74e1a17d: no GPU, no workers, no weights).'
_INC = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"([^"]+\.hip)"', re.M)


def harvest_files(origin):
    return [f for f in origin.files() if (f.startswith(ROOTS) or f in TEST_HOOKS) and '/__pycache__/' not in f]


def assign(origin):
    owner, fams = {}, {f['name']: [] for f in FAMILIES}
    names = [f['name'] for f in FAMILIES]
    if len(set(names)) != len(names):
        raise SystemExit('duplicate family names: ' + ' '.join(sorted({n for n in names if names.count(n) > 1})))
    missing = []
    for path in harvest_files(origin):
        fam = next((f for f in FAMILIES if any(p.fullmatch(path) for p in f['pats'])), None)
        if fam is None:
            missing.append(path)
            continue
        fams[fam['name']].append(path)
        owner[path] = fam['name']
    return fams, missing


def included_hip(origin, files):
    """.hip files that another file #includes: fragments, not translation units."""
    out = set()
    for f in files:
        if f.endswith(harvestlib.C_FAMILY) and '/tests/' not in f:
            for name in _INC.findall(open(origin.find(f), errors='replace').read()):
                out.add(os.path.basename(name))
    return out


def lang_of(files):
    ext = {os.path.splitext(f)[1] for f in files}
    if '.hip' in ext:
        return 'hip-c++'
    if ext & {'.cu', '.cuh'}:
        return 'cuda'
    if ext & {'.cpp', '.cc', '.hpp', '.h', '.inc', '.inl', '.in'}:
        return 'c++'
    if '.c' in ext:
        return 'c'
    if '.py' in ext:
        return 'python'
    return 'c++'


def strip_tag(name):
    head, _, last = name.rpartition('/')
    return head + '/' + last.split('.')[0]


_QINC = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"([^"]+)"', re.M)


def relative_roots(origin, fams):
    """Extra include roots a module exposes so that the origin's quoted relative includes into it ("channel_hip.hpp",
    "../core/tp_join.hpp", "../common/glm_math.hpp") still resolve when the including file sits in another module: for an
    include REL of file F that names T = dirname(F)/REL in module B, B gets the directory D of its own tree with
    D/REL == T (D = an m-th-level directory under T's base for m leading ../). An include that src/ or include/ already
    resolves needs nothing; one with no such D in B (deeper ../ chains) stays unresolved and its module fails the gate."""
    owner = {f: n for n, fs in fams.items() for f in fs}
    dirs = {n: {os.path.dirname(f) for f in fs} for n, fs in fams.items()}
    alldirs = {n: {'/'.join(d.split('/')[:i]) for d in ds for i in range(1, d.count('/') + 2)} for n, ds in dirs.items()}
    extra = {n: set() for n in fams}
    for f, a in owner.items():
        if not f.endswith(harvestlib.C_FAMILY):
            continue
        for rel in _QINC.findall(open(origin.find(f), errors='replace').read()):
            t = os.path.normpath(os.path.join(os.path.dirname(f), rel))
            b = owner.get(t)
            if b is None or b == a or any(os.path.normpath(os.path.join(r, rel)) == t for r in ('src', 'include')):
                continue
            parts = rel.split('/')
            m = len([p for p in parts if p == '..'])
            q = '/'.join(p for p in parts if p != '..')
            if not t.endswith(q):
                continue
            base = t[:len(t) - len(q)].rstrip('/')
            if m == 0:
                cands = [base]
            elif not base:              # the include climbs to the origin root: any depth-m directory of B
                cands = sorted(d for d in alldirs[b] if d.count('/') == m - 1)
            else:
                cands = sorted(d for d in alldirs[b] if d.startswith(base + '/') and d[len(base) + 1:].count('/') == m - 1)
            if cands:
                extra[b].add(cands[0])
    return extra


def build_modules(origin):
    fams, missing = assign(origin)
    if missing:
        raise SystemExit(f'{len(missing)} files under {ROOTS} match no family, e.g. ' + ' '.join(missing[:20]))
    extra_roots = relative_roots(origin, fams)
    frag = included_hip(origin, harvest_files(origin))
    mods = []
    for f in FAMILIES:
        files = fams[f['name']]
        if not files:
            raise SystemExit(f"family {f['name']} matches no file")
        mid = f"{f['kind']}.enginelab.{f['name']}"
        lang = lang_of(files)
        cls = strip_tag(harvestlib.iface(f['iface'])) if f['tag'] else None
        tagnote = (f"Interface tag: {strip_tag(f['iface'])} (a class the catalogue already has); the module provides it with an enginelab variant tag." if f['tag'] else
                   f"Interface tag: none (no catalogue interface matches); [[provides]] names the shape it would carry, {harvestlib.iface(f['iface'])}.")
        if lang == 'hip-c++':
            sources = [x for x in files if x.endswith('.hip') and os.path.basename(x) not in frag]
            hosts = [x for x in files if x.endswith(('.cpp', '.cc'))]
            compile_, reason = ('standalone', None) if sources else ('header-only', 'its .hip files are fragments #included by another unit')
            build = (f"{len(sources)} HIP unit(s) compiled standalone by the gate" + (f"; {len(hosts)} host .cpp file(s) are compiled by the origin's CMake host toolchain and are not gate units" if hosts else ''))
        elif lang == 'cuda':
            sources, compile_, reason = [], 'n/a', 'CUDA source for the optional RTX PRO route: no CUDA toolkit on the island and no NVIDIA device on the pod, so it is not compiled'
            build = 'CUDA (RTX PRO 6000 route)'
        elif lang in ('c++', 'c'):
            ext = ('.cpp', '.cc') if lang == 'c++' else ('.c',)
            sources = [x for x in files if x.endswith(ext)]
            if sources:
                compile_, reason = 'n/a', 'host C++ (or C) compiled by the origin\'s CMake with its host toolchain; the compile gate builds hip-c++ modules only'
            else:
                compile_, reason = 'header-only', 'host headers only; compiled inside the modules that include them'
            build = f'{len(sources)} host unit(s)'
        else:
            sources, compile_, reason, build = [], 'n/a', f'{lang} files: not compiled', lang
        notes = f"{tagnote} {build}. {NOT_TIMED}"
        prov = [P(f['iface'], f['api'], f['numeric'], f['regime'])]
        includes = ['src/src', 'src/include', 'src/third_party', 'src/apps', 'src/tests'] + ['src/' + d for d in sorted(extra_roots[f['name']])]
        if f['requires']:
            notes += ' Requires ' + ', '.join(f['requires']) + ': the origin builds these against the external strix-trellis library (CMakeLists.txt STRIX_DS41_TRELLIS_INCLUDE_DIR, include <strix_trellis.h>), which is catalog/strix_trellis here.'
        if f['tag']:
            where = (f"where it would go: beside the catalogue's {strip_tag(f['iface'])} providers as the enginelab variant "
                     f"(same op, the engine lab's own operand layouts, so an adapter comes first)")
        elif f['kind'] == 'tool':
            where = f"where it would go: kept as a reference program or lab ({harvestlib.iface(f['iface'])}), not linked into an engine"
        else:
            where = f"where it would go: a new strixflerp interface {harvestlib.iface(f['iface'])}, which no catalogue module provides yet"
        why_full = f"{f['why']}; {where}"
        tops = sorted({'/'.join(x.split('/')[:2]) for x in files})
        summary = f['summary'] if len(f['summary']) >= 80 else f"{f['summary']}: {len(files)} origin files under {', '.join(tops[:4])}"
        mods.append(M(mid, summary, files, provides=prov, lang=lang, requires=['module:' + x for x in f['requires']], sources=sources, includes=includes,
                      compile=compile_, reason=reason, why=why_full, notes=notes, spdx=SPDX_EXTRA.get(mid),
                      authors=AUTHORS, toolchain='therock-10.2' if lang == 'hip-c++' else ''))
        mods[-1]._tag_class = cls
    return mods


def fast_expand(self, patterns, exclude=()):
    """Exact origin paths (this harvest lists every file) resolve directly; anything else goes through Origin.expand."""
    have = set(self.files())
    if not exclude and all(p in have for p in patterns):
        return sorted(set(patterns))
    return Origin.expand(self, patterns, exclude)


TreeOrigin.expand = fast_expand

ORIGIN_TOML = '''
# The Strix Halo engine lab: a C++20 / HIP inference engine written from scratch for AMD Strix Halo (gfx1151, wave32,
# 128 GB unified memory), one node or two nodes over USB4, optional CUDA routes for an RTX PRO 6000. EXACT profile =
# bitwise its declared reference; FAST / TURBO levers are registry rows behind a quality gate (AGENTS.md). Card
# ORIGIN-ENGINELAB bc74e1a17d; the companion USB4 kernel series is origin enginelab_usb4.
name      = "enginelab"
repo      = "https://github.com/hyukudan/strix-halo-engine-lab"
commit    = "0598cf6dc030260c633482a4ef4b38fdd9d5502b"
branch    = "main"
version   = "project strix_engine_lab 0.1.0 (CMakeLists.txt); README results measured 2026-11-21 on main at d3a89ae16 and f42edd05e: DeepSeek V4.1 Flash pair 26.85 tok/s plain, 41.70-60.04 with DSpark, prefill 792 / 1,124 / 1,224 tok/s at 2K / 8K / 32K; GLM-5.3 Flash pair 25.73 plain, 40.37 (code) and 66.48 (prose) with MTP + wide, prefill 617 / 595 at 8K / 32K; MiniMax M3.1 Flash pair 23.3 plain, 32.4 code with DSpark auto; Qwen3.8-Flash-Next one node 51.0 MTP at 32K, served 38-155 by content class; Qwen3.8-27B 13.9 one node, 20.6 TP pair; Hy4-flash 16.8 one node, 22.3 TP pair"
reference_checkout = "control-node:/data/ark-inbox/src/github.com__hyukudan__strix-halo-engine-lab#0598cf6dc030260c633482a4ef4b38fdd9d5502b (ark copy: the tree at the commit, .git without objects); laptop: ~/Code/strixflerp-upstream-supplement/enginelab (the same files, by tarball through control-node get); docs/harvest/harvest_enginelab.py pins the tree digest"

[licence]
spdx      = "Apache-2.0"
files     = ["LICENSE", "LICENSES/ISA-L-CRC32C.txt", "LICENSES/stb_image.txt", "LICENSES/strix-trellis-Apache-2.0.txt"]   # copied verbatim next to this file
third_party = "LICENSE: Apache-2.0. LICENSES/: Intel ISA-L CRC32C (BSD-3-Clause; src/dist_crc32c_vector.hpp, in runtime.enginelab.xport_crc), stb_image (MIT or public domain; third_party/stb, in runtime.enginelab.stb_image) and strix-trellis (Apache-2.0; the tiled core, IU4 pack builder, W4A16 decoder and trellis codec of src/core were adopted from it, src/core/NOTICE.md). include/strix/qwen_iq*_grid.inc and ds41_iq2_xxs_grid.inc carry ggml's copyright (MIT). src/gemma4/exl3/notices (Apache-2.0 NOTICE) and src/gemma4/serve/notices/UNICODE-LICENSE.txt (Unicode License v3) are kept beside the code they cover. The EXL3 format is exllamav3's (MIT); QTIP is Tseng et al.; no code of either is included (src/core/NOTICE.md). Weights and the README's measurement kits are not part of the tree."

[toolchain]
host      = "C++20 (CMAKE_CXX_STANDARD 20), CMake; Linux and ROCm on Strix Halo; Windows and CUDA builds exist for the RTX route"
device    = "therock-10.2"                               # the origin names ROCm and leaves the version to the user; the pod's TheRock 10.2 is what the compile gate used
build     = "cmake with HIP enabled and STRIX_HIP_ARCHS gfx1151 (cmake/StrixHipArchs.cmake: STRIX_HIP_WAVE32_FLAG = -mno-wavefrontsize64); the HIP targets set HIP_STANDARD 20 and -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee -fhip-fp32-correctly-rounded-divide-sqrt (CMakeLists.txt, e.g. strix_tiled_core and the GLM tier flags at 7578); include roots src/ and include/ (target_include_directories ... src include); strix_hip_runtime defines STRIX_DPP_REDUCE and opt-in STRIX_HAS_EXPERIMENTAL_* switches. About 4,000 lines of CMake (CMakeLists.txt, cmake/**) are not harvested"

[compile]                                                # CMake's HIP language: clang++ -x hip, one object per .hip
driver    = "<rocm>/lib/llvm/bin/clang++"
arch_flag = "--offload-arch=gfx1151"
rocm_flag = "--rocm-path=<rocm>"
flags     = ["-x", "hip", "-std=c++20", "-O3", "-DNDEBUG", "-fPIC", "-mno-wavefrontsize64", "-fno-fast-math", "-ffp-contract=off", "-fdenormal-fp-math=ieee", "-fhip-fp32-correctly-rounded-divide-sqrt", "-DSTRIX_DPP_REDUCE=1", "-c"]
gate      = "gate/compile/run.py --module <id> for each of the 93 standalone hip-c++ modules (six workers, --jobs 4 each), control-node CPU, nice 19, therock-10.2, 2026-11-26 on card/bc74e1a17d @ 5bd68981e80b: 77 of 93 modules and 144 of 174 units compile for gfx1151 (gate/results/compile/<module>.json). The 16 failing modules stop at two-level relative includes into shallow modules (../../xport/channel_hip.hpp, ../../moe_core/exl3_trellis.hpp, ../../attn_core/*, ../../core/*, ../../hip_q8_rows_*, ../../model_graph/*, ../../dspark_common/*), which a per-module compile cannot resolve; the origin compiles them in one tree"
note      = "Include roots per module are src/src and src/include (the origin's src/ and include/), so a module's #include lines resolve through the modules it requires; harvestlib adds those requires from the #include lines. Generated headers (include/strix/build_info.hpp from build_info.hpp.in, apps/offline_tuning_build_context.hpp from its .in) are not generated here."

[kernels]
loading   = "statically linked: kernels are HIP functions launched by the engine's own launchers (src/hip_*_launcher.hip, src/runtime_launchers.cpp, the kernel registry); decode steps are captured as serial graphs; no code-object kit"
selection = "levers as data: strix::profile resolves --profile exact|fast|turbo per regime from registry rows with versioned identities and receipts; STRIX_* environment switches and model TOMLs (models/*.toml, not harvested) select routes; the deployment planner budgets memory per device declaration"
models    = "Qwen3.8-Flash-Next, DeepSeek V4.1 Flash, MiniMax M3.1 Flash, GLM-5.3 Flash in production; Qwen3.8-27B, Hy4-flash, Nemotron 3.5 Bolt E4B, MiniMax H3 (audio + video), MiniMax-H3-Image, Gemma 4 and Inkling 2 Mini as opt-in builds (README.md, models/registry.toml)"
'''

LEFT_OUT = {
    'tests/': '1,991 files of unit, parity and GPU tests with fixtures; read for the module records, not harvested',
    'scripts/, tools/': 'benchmark, measurement, packaging and audit scripts (read, never run)',
    'docs/, README*.md, AGENTS.md, REPO_DESCRIPTION.txt': 'documentation; the module describes are written from it',
    'CMakeLists.txt, cmake/, packaging/, .github/': 'the build system (HIP flags recorded in ORIGIN.toml [compile])',
    'models/, data/, recipes/, release/, support/, artifacts/, examples/': 'model TOMLs, device declarations, draft vocabularies and other data',
    '.audit555-*.log, .pr567-*.log': 'build logs committed by the origin',
    'LICENSE, LICENSES/': 'copied beside ORIGIN.toml',
}


def modules_md(mods):
    bt = '\x60'
    by_kind = {}
    for m in mods:
        by_kind[m.kind] = by_kind.get(m.kind, 0) + 1
    tagged = sum(1 for m in mods if m._tag_class)
    total = sum(len(m.paths) for m in mods)
    o = ['# Strix Halo engine lab modules', '',
         f'{len(mods)} modules, {total} files of hyukudan/strix-halo-engine-lab at {COMMIT[:12]} (card ORIGIN-ENGINELAB bc74e1a17d). Written by',
         bt + 'docs/harvest/harvest_enginelab.py' + bt + ' from its ' + bt + 'FAMILIES' + bt + ' table.', '',
         '- **tag**: the strixflerp interface class the family matches, or none (then **provides** is the name its shape would carry).',
         '- **lang / gate**: hip-c++ modules list their .hip units for the compile gate; host C++ and CUDA modules are not compiled here.', '',
         'By kind: ' + ', '.join(f'{k} {v}' for k, v in sorted(by_kind.items())) + f'. {tagged} carry a catalogue tag, {len(mods) - tagged} are none.', '',
         '| module | files | lang | units | tag | provides |', '|---|---:|---|---:|---|---|']
    for m in mods:
        o.append(f'| {bt}{m.id}{bt} | {len(m.paths)} | {m.lang} | {len(m.sources)} | ' +
                 (bt + m._tag_class + bt if m._tag_class else 'none') + f' | {bt}{m.provides[0]["interface"]}{bt} |')
    o += ['', '## What each module is', '']
    for m in mods:
        o.append(f'- {bt}{m.id}{bt}: {m.summary}. Why: {m.why}.')
    o += ['', '## Not done here', '', NOT_TIMED, 'The shapes the card names (exact TP2 decode, expert TP decode, MTP wide drafting, WMMA prefill at 8-32K, the',
          'DSV4.1 two-node path) are in core_tp_join + xport_tp_gather + pair_state, ds41_expert_tp, spec_wide_window + apps_mtp + mtp_window,',
          'core_tiled_wmma + qsa_prefill + glm_prefill_wmma + moe_core_prefill, and ds41_pipeline + dist_communicator + direct_transport.']
    open(os.path.join(harvestlib.REPO, 'catalog', 'enginelab', 'MODULES.md'), 'w').write('\n'.join(o) + '\n')


O = TreeOrigin('enginelab', 'enginelab', 'https://github.com/hyukudan/strix-halo-engine-lab', COMMIT, 'main', 'Apache-2.0',
               AUTHORS, 'therock-10.2', checkout=CHECKOUT, commit=COMMIT, digest=TREE_SHA256)

if __name__ == '__main__':
    mods = build_modules(O)
    if '--classify' in sys.argv:
        for m in mods:
            print(f'{m.id:52} {len(m.paths):5} {m.lang:8} units={len(m.sources):3} {m.compile}')
        print(f'{len(mods)} modules, {sum(len(m.paths) for m in mods)} files, tree digest {O.digest}')
        sys.exit(0)
    # run() prunes the non-module directories of catalog/enginelab, so LICENSES/ is removed first and rebuilt by the copy
    harvestlib.shutil.rmtree(os.path.join(harvestlib.REPO, 'catalog', 'enginelab', 'LICENSES'), ignore_errors=True)
    _copy = harvestlib.shutil.copyfile
    def _copy_mkdir(src, dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        return _copy(src, dst)
    harvestlib.shutil.copyfile = _copy_mkdir
    run(O, mods, ORIGIN_TOML, LEFT_OUT, LICENCE_FILES)
    modules_md(mods)
    if '--no-usb4' not in sys.argv:
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'harvest_enginelab_usb4.py')], check=True)
