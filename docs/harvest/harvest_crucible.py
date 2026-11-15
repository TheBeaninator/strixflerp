#!/usr/bin/env python3
"""Harvest Crucible (jaedync/crucible-strix-halo, d6af02c) into catalog/crucible. Run: python3 docs/harvest/harvest_crucible.py

Crucible's engine is ONE HIP translation unit: engine/engine.hip includes every header under engine/ (642 kernels,
docs/BUILD.md). So the kernel, format and runtime modules below are header fragments of that unit
([gate] compile = "via:executor.crucible.flashnext"); the executor module holds engine.hip and is built by the
recipe qwen38-flash-next.crucible with the origin's own Makefile (tool.crucible.build).

Headers include each other with paths relative to the including file ("../gfx1151/common.h") or to engine/
("model/flashnext/model.h"), so every module exports src/engine and each of its own header directories as include
roots; the include-derived requires below come from those.

The reference checkout is a read-only clone in ~/Code/strixflerp-upstream-supplement/crucible-strix-halo (nothing of
it is run on the laptop)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvestlib
from harvestlib import Origin, M, P, run

CHECKOUT = os.environ.get('CRUCIBLE_CHECKOUT', os.path.join(harvestlib.SUPPLEMENT, 'crucible-strix-halo'))


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


O = GitOrigin('crucible', 'crucible-strix-halo', 'https://github.com/jaedync/crucible-strix-halo',
              'd6af02cd739d554c43a62ef8d0e98b6f59f2dfd7', 'main', 'Apache-2.0', ['jaedync'], 'rocm-7.2.4')
E = 'engine/'
EXEC = 'executor.crucible.flashnext'
VIA = 'via:' + EXEC
VIA_REASON = 'header fragment of the one translation unit engine/engine.hip; compiles only inside ' + EXEC
EXL3_MIT = 'Apache-2.0 AND MIT'
EXL3_UP = 'EXL3 format after ExLlamaV3 d3739fd393337b1ff4d6c2a342b12f0c87a9592f (MIT, Turboderp; catalog/crucible/NOTICE)'
DEC = 'decode and MTP verify, 1..8 rows (R), memory-bound'
PF = 'prompt chunks (up to 2048 rows a chunk), compute-bound on WMMA'
C_FAMILY = ('.h', '.hpp', '.hip', '.inc', '.cpp')
mods = []
claimed = set()


def take(patterns, exclude=()):
    out = [f for f in O.expand(patterns, exclude) if f not in claimed]
    claimed.update(out)
    if not out:
        raise SystemExit(f'nothing left for {patterns}')
    return out


def roots(files):
    ds = {'src/' + os.path.dirname(f) for f in files if f.endswith(C_FAMILY)}
    if any(f.startswith(E) for f in files):
        ds.add('src/engine')
    return sorted(ds)


def hdr(id, summary, patterns, provides, spdx=None, upstream=None, notes=None, compile=VIA, reason=VIA_REASON,
        sources=(), lang='hip-c++', requires=()):
    files = take([E + p if not p.startswith(('engine/', 'tools/', 'ref/', 'server/', 'crucible', 'Makefile', 'VERSION'))
                  else p for p in patterns])
    mods.append(M(id, summary, files, provides=provides, requires=requires, lang=lang, sources=list(sources),
                  includes=roots(files), compile=compile, reason=reason, spdx=spdx, upstream=upstream, notes=notes,
                  toolchain='' if lang == 'python' else None))


# ------------------------------------------------------------------------------------------------- kernels
hdr('kernel.crucible.device_common', 'Header-only gfx1151 wave32 device helpers: shared math and loads, WMMA primitives with fp32 high/low expansions, LDS stream ring, compute-only barriers for loader waves',
    ['gfx1151/common.h', 'gfx1151/prefill_wmma.h', 'gfx1151/lds_stream_ring.h', 'gfx1151/persistent_sync.h'],
    [P('kernel-abi.crucible-device/-/-', 'crucible:: device helpers (gfx1151/common.h), Wmma* fragments (gfx1151/prefill_wmma.h), LdsStreamRing, persistent barriers', 'n/a', 'n/a'),
     P('gemm.wmma-tile/-/f16-f32.crucible', 'gfx1151/prefill_wmma.h', 'v_wmma_f32_16x16x16_f16, fp32 accumulate; high/low fp16 split preserves fp32 state contractions', PF)])
hdr('kernel.crucible.exl3_gemv', 'EXL3 decode matvec (mul1 trellis codebook) for 1..8 activation rows: input Hadamard transform, multi-matrix GEMV over front-ordered tile chunks, output transform; dense ring schedules with lookahead weight prefetch',
    ['exl3/exl3.h', 'exl3/kernels/exl3_gemv.h', 'exl3/kernels/lane_window.h', 'exl3/exl3_ring.h', 'exl3/ring_rows.h', 'exl3/row_policy.h',
     'exl3/front_order.h', 'exl3/decode_split.h', 'exl3/lookahead.h', 'exl3/balanced_items.h', 'exl3/dense_rows.h', 'exl3/plans.h'],
    [P('linear.decode/exl3/f16-f32.crucible', 'LinInKernel, LinGemvKernel, LinOutKernel, LinRingFlowKernel (exl3/exl3.h, exl3_ring.h); plans in exl3/plans.h', 'y = H_O(W_hat^T H_I(x*suh))*svh; codebook decoded exactly per lane, fp32 accumulate in a fixed split order; each row of R keeps the R=1 contraction order', DEC)],
    spdx=EXL3_MIT, upstream=EXL3_UP,
    notes='Own HIP code; the activation path and the codebook constants follow ExLlamaV3 (exl3.h:3, lane_window.h:24), so the module carries the MIT notice as well (NOTICE).')
hdr('kernel.crucible.exl3_gemm', 'EXL3 prefill GEMM on gfx11 wave32 WMMA: trellis tiles decoded into LDS and fed to v_wmma, two column tiles share one LDS publication, a half-precision path, packed input transforms',
    ['exl3/kernels/exl3_gemm.h', 'exl3/kernels/exl3_gemm_pair.h', 'exl3/kernels/exl3_moe_window.h', 'exl3/prefill_half.h', 'exl3/prefill_items.h', 'exl3/prefill_linear.h'],
    [P('linear.prefill/exl3/f16-f32.crucible', 'Exl3GemmKernel (exl3/kernels/exl3_gemm.h), PrefillGemmOut*Kernel, PrefillLinIn*Kernel (exl3/prefill_linear.h)', 'tile decoded to fp16 in LDS, WMMA fp32 accumulate, input transform and epilogue shared with decode', PF)],
    spdx=EXL3_MIT, upstream=EXL3_UP)
hdr('kernel.crucible.moe_decode', 'Decode routed experts: one-launch MoE stage, stream-grid expert body with b128/b64 weight loads and no LDS transpose, row-sparse experts for verify forwards, shared-expert-first scheduling',
    ['layers/moe_stage.h', 'layers/moe_sg.h', 'layers/moe_b128.h', 'layers/moe_b128_switch.h', 'layers/moe_rows.h', 'layers/moe_chunk_map.h',
     'layers/moe_shared_first.h', 'layers/moe_window_policy.h', 'layers/expert_count.h'],
    [P('moe.experts-decode/exl3/f16-f32.crucible', 'MoeStageKernel, MoeSgKernel, MoeFusedKernel, MoeRows*Kernel', 'exact: each routed slot keeps its R=1 order; stable combine by router slot', DEC)])
hdr('kernel.crucible.moe_prefill', 'Prefill MoE: stable expert-major dispatch without atomics (count, prefix, scatter), grouped expert GEMM items of 192 rows x 128 columns on WMMA, pair path (8-wave, two K sets), stable combine',
    ['layers/prefill_moe.h', 'layers/prefill_moe_tile.h', 'layers/prefill_moe_pair.h', 'layers/prefill_moe_kset.h', 'layers/kernels/prefill_moe.h'],
    [P('moe.experts-prefill/exl3/f16-f32.crucible', 'PrefillMoe*Kernel (layers/prefill_moe.h, prefill_moe_tile.h), MoeCount/Prefix/ScatterKernel (layers/kernels/prefill_moe.h)', 'per router slot exact; expert-major order stable, no atomics on output order', PF),
     P('moe.dispatch/-/-.crucible-stable', 'MoeCountKernel, MoePrefixExpertsKernel, MoeScatterKernel', 'exact (integer counting sort by expert)', PF)])
hdr('kernel.crucible.gdn', 'Gated DeltaNet: decode core for 1..8 rows, one-launch verify core, chunk-64 WY/UT prefill (gram, inverse, state carry), persistent R=1 attention-side prototype',
    ['layers/gdn_rows.h', 'layers/gdn_verify.h', 'layers/gdn_verify_switch.h', 'layers/persistent_gdn.h', 'layers/kernels/persistent_gdn_load.h',
     'layers/kernels/prefill_gdn.h', 'layers/packed_state.h'],
    [P('mixer.gdn/-/f32.crucible', 'GdnKernel (ops.h) / GdnVerifyKernel (gdn_verify.h)', 'fp32 recurrent state, fixed-order accumulate', DEC),
     P('mixer.gdn-prefill/-/f32.crucible-wy64', 'GdnPrefill*Kernel (layers/kernels/prefill_gdn.h)', 'chunk-64 WY/UT form: not the decode recurrence order (tolerance, not bits)', PF)])
hdr('kernel.crucible.hc', 'Hyper-connections: decode HC read (dots, mid, mix) on WMMA with fp16 or int8 weights, exact multi-row paths, prefill HC with packed operands',
    ['layers/hc_wmma.h', 'layers/hc_wmma2.h', 'layers/hc_frag.h', 'layers/hc_int8.h', 'layers/persistent_hc.h', 'layers/prefill_hc.h',
     'layers/kernels/hc_multirow.h', 'exl3/hc8_layout.h'],
    [P('hc.mix/f16/f32.crucible', 'HcWmmaKernel, HcW2Kernel (hc_wmma*.h), HcDots/HcMixKernel (ops.h)', 'fp32 accumulate on WMMA; int8 variant (CRUCIBLE_HC_WMMA=2) is a tolerance contract', DEC),
     P('hc.mix-prefill/f16/f32.crucible', 'PrefillHc*Kernel (layers/prefill_hc.h)', 'fp32 accumulate', PF)])
hdr('kernel.crucible.qsa', 'Sparse attention (QSA): exact fp32 indexer scores, tile-local top-512, split-K sparse attention for long decode, prefill union flash over four query rows sharing KV, opt-in input capture',
    ['layers/qsa_long.h', 'layers/qsa_capture.h', 'layers/qsa_capture_policy.h', 'layers/kernels/prefill_qsa.h', 'layers/kernels/prefill_qsa_score.h',
     'layers/kernels/prefill_qsa_score_sgpr.h', 'layers/kernels/prefill_qsa_select.h', 'layers/kernels/prefill_qsa_union.h', 'layers/kernels/qsa_union_lds.h'],
    [P('attn.qsa-indexer/-/f32.crucible', 'QsaScore*Kernel, QsaTop*Kernel (qsa_long.h), prefill_qsa_select.h', 'exact fp32 indexer, exact top-512', 'decode and prefill'),
     P('attn.sparse-sink/kv-f16/f32.crucible', 'QsaGqaSplitKernel, QsaMergeKernel, QsaAttnTailKernel (qsa_long.h), QsaUnionFlashKernel (prefill_qsa_union.h)', 'fp32 softmax, split-K merge', 'decode and prefill')])
hdr('kernel.crucible.ops', 'Non-EXL3 operators of the Qwen3.8-Flash-Next forward: embed, norms, RoPE (incl. generated YaRN tables), router, dense fp16 rows, PLE gather and mix, MTP fuse, argmax; numerics mode switch',
    ['layers/ops.h', 'layers/persistent_ops.h', 'layers/numerics.h', 'layers/dense_rows.h', 'layers/rope_tables.h', 'layers/prefill_ple.h', 'layers/nonfinite_probe.h'],
    [P('moe.router/-/f32.crucible', 'MoeRouteKernel (layers/ops.h)', 'fp32 logits, top-k stable', DEC),
     P('pos.rope/-/f32.crucible', 'RopePlainFillKernel, rope_tables.h (YaRN x2/x4)', 'tables generated from tools/yarn reference json', 'all'),
     P('ple.inject/-/f32.crucible', 'PleGatherKernel, PleMixKernel, PleMixSpanKernel (ops.h), prefill_ple.h', 'fp32', 'all'),
     P('logits.argmax/-/f32.crucible', 'ArgmaxKernel', 'exact', 'decode')],
    spdx=EXL3_MIT, upstream=EXL3_UP, notes='ops.h:65 folds HC weights as ExLlamaV3 does; carries the MIT notice too.')
hdr('kernel.crucible.vision', 'Vision tower kernels (patchify, patch embed, attention, layer norm, GELU); EXL3 linears run through the prefill GEMM',
    ['layers/kernels/vision.h'],
    [P('vision.tower-ops/exl3/f32.crucible', 'PatchifyKernel, PatchEmbedKernel, AttentionKernel, LayerNormKernel, GeluKernel (layers/kernels/vision.h)', 'fp32', 'one image per request')])

# ------------------------------------------------------------------------------------------------- formats
hdr('format.crucible.exl3', 'EXL3 checkpoint reader: bounded safetensors header parse + mmap of sharded packs, 256-byte-unit static layout, load task split, decode capability list (integer and half-integer K widths)',
    ['exl3/safetensors.h', 'exl3/static_layout.h', 'exl3/load_tasks.h', 'exl3/decode_support.h'],
    [P('format.loader/exl3-safetensors/-.crucible', 'crucible safetensors reader (exl3/safetensors.h), StaticLayout, LoadTasks', 'exact (bytes mmapped; tiles kept in EXL3 encoding on the device)', 'load time'),
     P('format.device-view/exl3/-', 'trellis tiles 16x16, K 1..8 and 1.5..6.5 (exl3/decode_support.h, kernels/lane_window.h)', 'raw EXL3 tiles', 'n/a')],
    spdx=EXL3_MIT, upstream=EXL3_UP)
hdr('format.crucible.flashnext_weights', 'Qwen3.8-Flash-Next pack geometry and load decisions: layer schedule, named geometry, per-matrix width and front order, CPU-only weight validation before allocation, BF16->F16 conversion',
    ['model/flashnext/geometry.h', 'model/flashnext/load_plan.h', 'model/flashnext/weight_validation.h', 'model/flashnext/bf16_half.h'],
    [P('format.layout/exl3-flashnext/-', 'geometry.h, load_plan.h, weight_validation.h', 'exact', 'load time')],
    spdx=EXL3_MIT, upstream=EXL3_UP)
hdr('format.crucible.ngram16', 'PLE n-gram table: the EXL3 n-gram trellis row codec (device and host), and the opt-in 16-bit table built from the original BF16 weights (fetch, convert, error check)',
    ['layers/ple_row_codec.h', 'model/flashnext/ple_host.h', 'tools/ngram16/codec.py', 'tools/ngram16/common.py', 'tools/ngram16/convert.py',
     'tools/ngram16/error_check.py', 'tools/ngram16/fetch.py', 'tools/ngram16/fp16_scan.py', 'tools/ngram16/synth.py', 'tools/ngram16/make_fixture.py'],
    [P('ple.ngram-rows/exl3-ngram/-', 'PleGatherKernel row codec (layers/ple_row_codec.h), host gather from the mmapped checkpoint (model/flashnext/ple_host.h)', 'codec bit-exact with exllamav3 ngram_codec.py', 'all'),
     P('tool.convert-ngram-table/bf16/-', 'tools/ngram16/fetch.py, convert.py', 'bf16 copy of the original table', 'offline')],
    spdx=EXL3_MIT, upstream='n-gram row codec ported from exllamav3 ngram_codec.py (MIT, Turboderp; tools/ngram16/codec.py:1)')

# ------------------------------------------------------------------------------------------------- runtime
hdr('runtime.crucible.device_memory', 'Device and host memory policy on the unified-memory APU: pinned host buffers and allocator, SVM migration rules that avoid stopping KFD queues, warm-up page touching',
    ['gfx1151/pinned_allocator.h', 'gfx1151/pinned_buffer.h', 'gfx1151/runtime_memory_policy.h', 'gfx1151/warmup.h', 'gfx1151/migration_wait.h'],
    [P('device.memory/hip/-.crucible', 'PinnedBuffer, PinnedAllocator, runtime_memory_policy.h', 'n/a', 'n/a')])
hdr('runtime.crucible.speculation', 'MTP speculative decoding: draft head policy, low-bit (q4/q2) and dense fp16 draft heads, acceptance prior from the MTP top-2 margin, value-aware draft length, draft vocabulary, verify spans',
    ['runtime/draft_*.h', 'runtime/spec_span.h', 'runtime/spec_steps.h', 'runtime/mtp_bulk.h'],
    [P('speculation.draft/mtp/f32.crucible', 'draft_ops.h, draft_quant.h (DraftQGemvKernel), draft_dense.h, draft_policy.h, draft_value.h', 'drafts only; verification commits the target tokens (lossless)', DEC),
     P('speculation.window-verify-lossless/-/-.crucible', 'draft_accept.h, spec_span.h', 'exact', DEC)])
hdr('runtime.crucible.sampling', 'Sampling and generation: own RNG (no std distribution), generation options, decode policy defaults, resident request acceptance',
    ['runtime/sampling.h', 'runtime/sampling_generation.h', 'runtime/gen_options.h', 'runtime/generation.h', 'runtime/decode_policy.h'],
    [P('sample.next-token/-/f32.crucible', 'sampling.h, sampling_generation.h', 'host sampler, portable RNG', 'per token')])
hdr('runtime.crucible.conversation_cache', 'Conversation and prefix cache: immutable entries owning KV bytes, resume, snapshots in pinned chunks, lazy saves, in-slot checkpoints, exact storage addresses, tenant keys',
    ['runtime/conversation_*.h', 'runtime/lazy_snapshot.h', 'runtime/snapshot_bytes.h', 'runtime/slot_checkpoint*.h', 'runtime/tenant.h', 'layers/cache_storage.h', 'runtime/retained_prefix.h'],
    [P('cache.prompt-crucible/-/-', 'ConversationCache (conversation_cache.h), snapshot_bytes.h, slot_checkpoints.h', 'exact (bit-identical resume)', 'serving')])
hdr('runtime.crucible.state_disk', 'NVMe tier for conversation states: block storage, file formats with checksums, state epoch (code identity) and the switches that are part of a state identity, SHA-256',
    ['runtime/state_disk.h', 'runtime/state_blocks.h', 'runtime/state_format.h', 'runtime/state_epoch.h', 'runtime/state_switches.h', 'runtime/sha256.h'],
    [P('kvcache.tiering/crucible-state/-', 'state_disk.h, state_format.h', 'exact', 'serving')])
hdr('runtime.crucible.batch', 'Batched serving contracts: start-time admission, CPU boundary checks, row inputs kept alive across queued copies, packed verify span validation, rows per packed pass',
    ['runtime/batch_admission.h', 'runtime/batch_contract.h', 'runtime/batch_inputs.h', 'runtime/batch_spec_contract.h', 'runtime/pass_rows.h', 'runtime/packed_rows_option.h'],
    [P('serve.memory-plan/-/-.crucible', 'batch_admission.h', 'n/a', 'serving')])
hdr('runtime.crucible.ablation', 'Opt-in runtime abliteration (CRUCIBLE_ABLATE): sidecar loading (Heretic ExLlamaV3 format), per-request mode, device half',
    ['layers/ablation.h', 'layers/ablation_device.h', 'runtime/ablate_request.h'],
    [P('steer.add-rows/-/f32.crucible', 'ablation.h, ablation_device.h', 'fp32', 'all')],
    spdx=EXL3_MIT, upstream='sidecar format of the Heretic ExLlamaV3 runtime (ablation.h:6)')

# ------------------------------------------------------------------------------------------------- executor
hdr(EXEC, 'Qwen3.8-Flash-Next EXL3 executor: the Engine (engine/engine.hip, one translation unit holding every kernel), model loader, layer loop, prefill engine with chunking, interleave and stage arena, batched and packed-verify forwards, switch registry',
    ['engine/engine.hip', 'switches.h', 'model/flashnext/model.h', 'model/flashnext/active_config.h', 'model/flashnext/draft_chain_switch.h',
     'model/flashnext/fixed_state.h', 'model/flashnext/mtp_sidecar.h', 'model/flashnext/prefill_launch.h', 'model/flashnext/vision.h', 'model/flashnext/multimodal.h',
     'runtime/layer_loop.h', 'runtime/prefill_*.h', 'runtime/prefill_engine.inc', 'runtime/stage_arena.h', 'runtime/stage_plan.h', 'runtime/batch_forward.h',
     'runtime/batch_spec_forward.h', 'runtime/device_zero.h', 'runtime/dump_io.h', 'runtime/prompt.h', 'runtime/sequence_state.h', 'runtime/rope_mode.h',
     'runtime/rope_policy.h', 'runtime/bounds.h'],
    [P('executor.model/qwen3.8-flash-next/exl3', 'crucible::Engine (engine/engine.hip), Loader (model/flashnext/model.h); driven by the JSON-line protocol of serve.crucible.engine_serve', 'per docs/FLASHNEXT_FORWARD.md; CRUCIBLE_EXACT selects the exact numerics mode', 'one GPU, 1..4 sessions')],
    sources=['engine/engine.hip'], compile='n/a',
    reason='the whole engine is one translation unit; it builds in the recipe qwen38-flash-next.crucible (origin Makefile), not in the per-module compile gate')

# ------------------------------------------------------------------------------------------------- serve
hdr('serve.crucible.engine_serve', 'The engine side of serving: JSON-line protocol (serial and batched), batch sessions and conversation LRU, disk tier glue, slot bases, logprobs, cache status and miss ledger',
    ['serve/*.h'],
    [P('serve.engine/-/-.crucible', 'serve.h, batched_serve.h, batch_session.h (JSON lines on stdin/stdout)', 'n/a', 'serving')])
mods_server = take(['server/'], exclude=['server/tests/**'])
mods.append(M('serve.crucible.openai', 'Python OpenAI- and Anthropic-compatible HTTP frontend over the engine process: chat template, tool calls, thinking budget, batch scheduler, vision requests, metrics',
              mods_server, lang='python', toolchain='', compile='n/a', reason='python: not compiled', includes=[],
              provides=[P('serve.http/openai+anthropic/-.crucible', 'python -m server (server/__main__.py, api.py, http.py)', 'n/a', 'serving')],
              requires=['serve.engine/-/-.crucible']))

# ------------------------------------------------------------------------------------------------- tools
hdr('tool.crucible.engine_cli', 'The engine binary command line: gen (one capture, --jobs), batch CLI, prefill bench, debug schedule, version',
    ['cli/*.h'], [P('tool.generate-ids/crucible/-', 'crucible gen CKPT ... (cli/gen_cli.h), prefill bench (cli/prefill_bench.h)', 'n/a', 'n/a')])
cli_py = take(['crucible', 'tools/crucible_cli/'])
mods.append(M('tool.crucible.cli', 'The ./crucible launcher and its Python CLI: setup, doctor, packs, service, update, kernel parameters',
              cli_py, lang='python', toolchain='', compile='n/a', reason='python: not compiled', includes=[],
              provides=[P('tool.launcher/crucible/-', './crucible (tools/crucible_cli/cli.py)', 'n/a', 'n/a')]))
build_files = take(['Makefile', 'VERSION', 'tools/hipcc_checked.sh', 'tools/devsplit.py', 'tools/devsplit_ir.py', 'tools/devsplit_groups.tsv',
                    'tools/devsplit_facts.tsv', 'tools/isa_diff.py', 'tools/release_engine.py', 'tools/state_epoch.py', 'tools/state_epoch_pins.tsv',
                    'tools/state_epoch_sources.txt', 'tools/generate_persistent_ops.py'])
mods.append(M('tool.crucible.build', 'The origin build: Makefile, the checked hipcc wrapper, devsplit (parallel split device compile with the same ISA), ISA diff, release stamp and state epoch stamp',
              build_files, lang='python', toolchain='', compile='n/a', reason='build scripts', includes=[],
              provides=[P('build.graph/make/-', 'make build/engine/crucible[.mono] (Makefile)', 'n/a', 'n/a')]))
ref_files = take(['ref/', 'tools/yarn/'], exclude=['ref/test_*.py', 'ref/bf16/test_*.py', 'ref/bf16/conftest.py'])
mods.append(M('tool.crucible.reference', 'CPU references: NumPy EXL3 codec and matvec (bit-exact codebook and decoder states), fp64 layer oracle, prefill and vision references, YaRN RoPE table generator',
              ref_files, lang='python', toolchain='', compile='n/a', reason='python: not compiled', includes=[],
              spdx=EXL3_MIT, upstream=EXL3_UP,
              provides=[P('verify.numerics-contract/exl3/-.crucible', 'ref/exl3_ref.py, ref/oracle.py, tools/yarn/rope_header.py', 'fp32/fp64 CPU', 'offline')]))

ORIGIN_TOML = '''
# Crucible: C++20 + HIP inference engine for EXL3 checkpoints of Qwen3.8-Flash-Next on one Strix Halo (gfx1151),
# with an OpenAI-compatible Python server and MTP speculative decoding.
name      = "crucible"
repo      = "https://github.com/jaedync/crucible-strix-halo"
commit    = "d6af02cd739d554c43a62ef8d0e98b6f59f2dfd7"
branch    = "main"
version   = "see VERSION in tool.crucible.build"
reference_checkout = "~/Code/strixflerp-upstream-supplement/crucible-strix-halo"

[licence]
spdx      = "Apache-2.0"
files     = ["LICENSE", "NOTICE"]                          # copied verbatim next to this file
third_party = "EXL3 format reference ExLlamaV3 d3739fd393337b1ff4d6c2a342b12f0c87a9592f (MIT, Turboderp, text in NOTICE); modules whose code follows or ports it (EXL3 kernels, reader, n-gram codec, CPU references, ops) are 'Apache-2.0 AND MIT'; the rest is Apache-2.0. Weights (Qwen Community License 1.0) are not part of the tree."

[toolchain]
host      = "C++20 single translation unit, GNU make; python3 for the build helpers and the server"
device    = "rocm-7.2.4"                                 # docs/BUILD.md: tested with ROCm 7.2.4 on gfx1151
build     = "make build/engine/crucible (devsplit: 64 parallel parts, same ISA) or build/engine/crucible.mono (plain hipcc, one clang process, about 20 min)"

[compile]                                                # Makefile ENGINE_FLAGS
driver    = "<rocm>/bin/hipcc"
arch_flag = "--offload-arch=gfx1151"
rocm_flag = "--rocm-path=<rocm>"
flags     = ["-O3", "-std=c++20", "-I<rocm>/include", "-pthread"]
output    = "one linked binary build/engine/crucible holding all 642 kernels"

[kernels]
loading   = "statically compiled into the one engine binary; switches (engine/switches.h, CRUCIBLE_*) pick kernel variants at run time"
'''

LEFT_OUT = {
    'bench/, tests/, server/tests/, ref/test_*.py, conftest.py, .github/': 'benchmarks, probes, tests and CI',
    'docs/, *.md at the root, engine/**/README.md, CHANGELOG.md, SECURITY.md, CONTRIBUTING.md': 'documentation (read for docs/CRUCIBLE.md)',
    'install.sh, Dockerfile, deploy/, requirements-cpu.txt, .gitignore': 'installation, container and systemd packaging',
    'training/': 'MTP draft tuning scripts',
    'tools/* (except the build helpers, crucible_cli, ngram16, yarn)': 'gate, probe, replay, soak and analysis scripts',
}

if __name__ == '__main__':
    run(O, mods, ORIGIN_TOML, LEFT_OUT, ['LICENSE', 'NOTICE'])
