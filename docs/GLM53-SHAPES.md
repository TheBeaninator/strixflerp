# GLM-5.3 harvest: every shape ranked for DSV4.1 TP4 and a GLM-5.3-Flash TP4 recipe

Card GLM53-SHAPES 7b11987286 (epic GLM53-HARVEST ed900d7238). Written 2026-11-26 from the catalogue at Buzz main
bcd23c7f6: the module.toml, MODULES.md and ORIGIN.toml files of the eight GLM53 origins, `docs/SHAPE-TRANSFERS.md`,
`docs/PREFILL-DSV41.md` and `docs/DSV41-TP4.md`, and the TP4 collective source in `catalog/strixfold_tp4`. No GPU, worker
or weights were used. Every gain below is inferred, and its arithmetic is given. No idea is closed because of its format:
each shape says how it is re-expressed in the target (`docs/SHAPE-TRANSFERS.md`, the principle).

## 1. What was harvested

| origin | card | upstream @ pin | modules | what it is |
|---|---|---|---:|---|
| ciru_link | 682cd99ba2 | ciru-ai/CiruStrixLink @ 1c7f1ae | 11 | Go USB4 transport and console, plus a 16-patch Linux 7.2.2 USB4STREAM series with DMA-BUF import (zero-copy GPU buffers on the NHI rings) |
| vllm_strix | d2a7b91dec | davidcanar/vllm-strix-halo @ fe5ac10 | 78 | vLLM TP2 for GLM-5.3-Flash and DeepSeek-V4-Flash over OdinLink: gfx1151 kernels, DFlash2/DSpark, long-context fixes |
| odinlink5 | d2a7b91dec | Geramy/OdinLink-Five @ 0315225 | 15 | Thunderbolt 5 / USB4 host-to-host driver, library, RCCL/NCCL net plugins, a verbs provider |
| kyojin | 692a13d6d0 | Yamz-Labs/kyojin @ 116a883 | 182 | ExLlamaV3 fork: GLM-5.3-Flash on one Strix Halo (580-633 tok/s prefill, 26-32 MTP decode), DS4.1 EXL3 kernels |
| ds4 | 31224b4ca3 | antirez/ds4 @ b3f5018 + 5 overlays | 217 | DwarfStar C engine (DSV4/4.1, GLM-5.x) with davidcanar glm-tp-rocm (PR #1024: KDA head split, 50/50 routed-expert TP) and four kyuz0 gfx1151 branches (Halo kernels, RoCE TP) |
| maya_tp | 2982083a71 | uv-genai/maya-tp @ 6951bd7 | 99 | Project Maya TP2: GLM-5.3-Flash intra-expert TP + MTP, DS4.1 TP2 + DSpark, RoCE v2 RDMA transport ported from gufo |
| enginelab | bc74e1a17d | hyukudan/strix-halo-engine-lab @ 0598cf6 | 254 | C++20/HIP engine, EXACT profile bitwise to its reference: DS4.1 pair 26.9 plain / 41.7-60.0 DSpark, prefill 792 / 1,124 / 1,224 tok/s at 2K / 8K / 32K; GLM-5.3-Flash pair 25.7 plain / 40.4-66.5 MTP + wide, prefill 617 / 595 |
| enginelab_usb4 | bc74e1a17d | hyukudan/strix-halo-usb4-driver @ d5de947 | 16 | the engine lab's USB4 kernel series (49 stable + 3 experimental patches) |

872 modules in all. fastasfuck was refreshed to 1.1.0 (d08e5a54d) but adds no GLM53 module; it is not ranked here.
Appendix A lists every one of the 872 module ids under its shape family.

## 2. The two targets and what bounds them

**(a) DSV4.1 TP4 on F D C A** (`dsv41-flash.strixfold-tp4`, served build 72c0f083fca1, conf v9). Prompt passes are about
75 % of wall time; prefill runs at about 1,200-1,400 tok/s cold; single-stream decode 52 tok/s; 16 slots; 100 GbE
RoCE rail (`TRANSPORT=rdma`, `COLL_SLOT_KIB=8704`, `TF_COLL_LANES=4`, `PREFILL_SPLIT=256`, `PREFILL_ZIP2=1`,
`TP4_FILL=1`, `VGATHER=1`, `L2_PREFETCH=1`).

The fact that orders the ranking: **TP4 prompts run at about the TP2 pair's rate.** The pair measured 1,205-1,452 tok/s
(`docs/PREFILL-DSV41.md` section 2); four ranks give 1,200-1,400. Doubling the ranks halved each rank's share of the
kernel work in `docs/PREFILL-DSV41.md` section 3 (1.46 s of kernel time a rank per 2,048 rows at TP2, 70 % of it the
routed experts), so the time went somewhere that does not shrink with more ranks (inferred; the TP4 phase timer
`tf-dsv41-lanes --profile 2` is the measurement that settles it):

1. **Per-job fixed cost of the experts pass.** Every (expert, row tile) job decodes that expert's matrices; at TP2 that
   was ~15 ms a pass whatever the rows. At TP4 a rank holds a quarter of every expert, so the job count stays at ~385
   while the work per job halves: the fixed share grows.
2. **Gather bytes.** `rdma_core.zig` (world 4) gathers rank g's block into place g of every rank's slot and sends a
   trailer to every peer. If the blocks are the n x 5120 fp32 partials of `docs/PREFILL-DSV41.md`, each rank sends its
   block to three peers instead of one. At 2,048 rows that is 2 gathers x 20 layers x 2,048 x 5,120 x 4 B x 3 peers =
   5.0 GB out of each rank a chunk, ~0.45 s at ~11 GB/s, against ~0.7 s of rank kernel time: the zip can hide it only
   if nothing else competes. (Not verified: the TP4 executor's `prompt2d` / `w-pf` layout may already send quarters;
   the C2 card counts the bytes first.)
3. **Work every rank repeats.** The mHC mixes and the Triton `_hc_finish` (Sinkhorn, 20 iterations), the router
   logits, the indexer top-512 at the index sources and the Engram host reads are replicated per rank, so their share
   doubles when the per-rank GEMM work halves.
4. **Prompt passes that need not run.** With 16 slots of agent traffic, requests that share a prefix and start together
   each run the shared chunks; kept states exist only at 2,048-row chunk ends.

**(b) A future GLM-5.3-Flash TP4 recipe.** GLM-5.3-Flash (`glm5-next` in the origins) is a DSA model: a lightning indexer
and absorbed sparse MLA, KDA (Kimi delta attention) recurrent layers, mHC streams, routed experts (288 experts per
vllm_strix's tile tables) and an MTP NextN head. The best public numbers are the engine lab's pair (617 tok/s prefill
at 8K, 25.7 plain, 40.4-66.5 MTP + wide decode, exact) and Kyojin's single box (580-633 prefill, 26-32 MTP). Nobody runs it
at TP4 or on RoCE. It serves only in windows or after the owner decides; DSV4.1 TP4 stays the standing engine.

## 3. Shape families (every module is in one; Appendix A has the ids)

Each family gives the shape, where it would go in strixflerp and how it is re-expressed there, the gate, and the expected
gain for target (a) and target (b).

### F01 MoE expert prefill (grouped expert GEMMs) (30 modules)

- **Shape.** Rows sorted or bucketed by expert, one launch (or a fixed few) per projection over all experts, weight tile decoded once into LDS inside the GEMM, gate and up computed together, an epilogue that combines with the shared expert. Variants: EXL3 trellis tiles (kyojin exl3_moe_prefill / exl3_moe_grouped), IQ2/Q2_K panels on frozen WMMA producers (ds4 k53perf halo S9 "wide / twin" tiles for 2,048 / 4,096-row chunks), int8-activation IQ experts (maya prefill_moe_rdna4), llama.cpp MMQ views (ds4 mmq_moe, maya prefill_moe_mmq), and the engine lab's shared grouped family (moe_core_prefill, moe_core_grouped, microscaled, Q4_K x A8).
- **Where, re-expressed.** `moe.experts-prefill/exl3/bf16.jobs` inside `kernel.strixfold.exl3_pf_experts` (TP2) and its TP4 slice in `executor.strixfold_tp4.dsv41`; for GLM, the experts pass of the GLM executor. Re-expression: keep StrixFold's EXL3 3-bit tiles and its per-output fp32 order, take the launch shape (one persistent launch over (expert, row-tile) items, slot-count tile classes, decode-once-into-LDS); GGUF/IQ variants re-expressed onto EXL3 tiles by swapping the inner decode, never by changing the weights.
- **Gate.** bit identity against `exl3_pf_experts` (`gate/prefill_dsv41_bench.py` already counts differing outputs: 0 of 10.5 M is the bar) when the per-output K order is kept; FIDELITY otherwise (int8 activations, IQ re-quantisation).
- **Gain (a) DSV4.1 TP4.** high: experts are ~70 % of a TP2 rank's prefill kernel time; at TP4 each rank holds a quarter of every expert, so the per-job fixed part grows as a share. Inferred +10-25 % TP4 prefill.
- **Gain (b) GLM-5.3-Flash TP4.** high: the GLM prefill rides on this (kyojin's 580-633 tok/s one-box GLM prefill is exl3_moe_prefill).

### F02 Dense prompt GEMMs, tile cores and GEMM tuning (50 modules)

- **Shape.** Register-blocked, double-buffered WMMA GEMM where a block owns BM x BN outputs, decodes each weight tile once into LDS and every wave reuses its fragments, with ascending K16 order (enginelab core_tiled_wmma, from strix-trellis); per-pack autotuned kernel choice cached on first call (kyojin exl3_coop_autotune, tune_cache, exl3_kernel_map); frozen WMMA GEMMs for fixed shapes (ds4 k53perf halo output_a / output_b / shared_gu / shared_down); dequantise-once-to-bf16 then hipBLASLt for formats without a matrix path (vllm fp8_prefill, maya ds4_lt); integer (IU4, A8, Q4_K x A8) operand forms.
- **Where, re-expressed.** `linear.prefill/exl3/f16-f32` (`kernel.strixfold.exl3` `exl3.gemm` for wq_a, wkv, wq_b, wo_a, wo_b and engram_wkv); a `tune.table` record next to the kit. Re-expression: the tile/blocking and autotune choices are written for EXL3 trellis tiles; the K16 order is what makes it bit-exact.
- **Gate.** bit identity when the per-output K order is unchanged (core_tiled_wmma claims exactly this); FIDELITY for integer-activation forms.
- **Gain (a) DSV4.1 TP4.** medium: the five EXL3 projections are 17 % of TP2 prefill kernel time at 20-27 TFLOP/s; a 1.5x tile core is ~5-6 % of prefill.
- **Gain (b) GLM-5.3-Flash TP4.** medium: same role in the GLM attention projections.

### F03 Weight formats, codecs, quantisers and loaders (41 modules)

- **Shape.** One trellis codec (ring, codebook, reader) shared by every dense and MoE kernel (enginelab core_trellis, kyojin exl3_codec/pack/quantize/hadamard); GGUF/IQ/NVFP4 readers and dequantisers; pack converters (kyojin dsv41_convert, ds4 gguf_* tools); the tensor-name adapters and weight catalog of the engine lab.
- **Where, re-expressed.** `format.*` and the `kernel.strixfold.exl3` codec headers; the GLM recipe's weight pack. Re-expression: a codec is the one place a format lives; transfers keep the target's codec and borrow the readers' loading shapes (coalesced pinned-ring reads, kyojin stloader).
- **Gate.** bit identity of decoded tiles against the target codec on every tensor of a pack; FIDELITY for any re-quantisation.
- **Gain (a) DSV4.1 TP4.** low: no served-path change; the DS4.1 pack converters matter only for a re-pack.
- **Gain (b) GLM-5.3-Flash TP4.** high as a prerequisite: the GLM recipe needs one codec and one pack (EXL3 K4/K5 trellis per the engine lab, or the public IQ2_XXS GGUFs per maya-tp).

### F04 Decode GEMVs and row kernels (71 modules)

- **Shape.** Split-K / wvSplitK GEMVs with 16 contiguous bytes a lane; q8_1-activation integer dots (llama.cpp mmvq) for K-quants and IQ; VALU matvecs instead of padded WMMA at batch 1; look-ahead row plans (enginelab core_gemv_rows); FP8 block-scaled GEMV in C++ replacing a slow Triton one; the common/compat headers these compile against.
- **Where, re-expressed.** `linear.decode/exl3-trellis/f16-f32` and the decode linears of `executor.strixfold_tp4.dsv41`; `kernel.flerp.moe_kquant` for GGUF executors. Re-expression onto EXL3: lane-load and split-K mapping carry, the inner decode is the trellis one.
- **Gate.** bit identity when the order is kept (R1 showed it can be); FIDELITY otherwise.
- **Gain (a) DSV4.1 TP4.** low-medium: decode is ~25 % of wall time; the TP4 decode linears are already split four ways.
- **Gain (b) GLM-5.3-Flash TP4.** medium: GLM decode at TP4 is bandwidth-bound per rank.

### F05 MoE decode, routers and the fused MoE half-layer (61 modules)

- **Shape.** The whole MoE half of a decode layer in one persistent launch (kyojin exl3_moe_fused 7/9 with read-only cache touches), all K groups of a token in 5 launches (dsv41_moe), VALU grouped matvecs, deterministic packed top-k routers (vllm router_packed_topk: (score, index) packed into one sortable key), sigmoid-bias top-k in one kernel, decode-sized direct-to-HIP MoE hooks.
- **Where, re-expressed.** `moe.experts-decode/exl3/f32` and `moe.router/-/bf16-f32` in `executor.strixfold_tp4.dsv41` decode; `kernel.strixfold_tp4.exl3_mlp`. Re-expression: keep the TP4 expert slice and the router's tie rule; borrow the launch fusion.
- **Gate.** bit identity (dsv41_moe is bit-identical to the per-group path by construction); router tie rule must match exactly.
- **Gain (a) DSV4.1 TP4.** medium for decode (launch-bound steps), small for wall time (~25 % decode share).
- **Gain (b) GLM-5.3-Flash TP4.** medium-high: GLM decode at 288 experts top-8 is launch-bound the same way.

### F06 Sparse attention, indexers, top-k selection and compressed KV (DSA / QSA / MLA) (93 modules)

- **Shape.** WMMA f16 indexer scoring `sum_h w[r,h] * relu(q[r,h].k[t])` with dense-pool and candidate-slot scans (kyojin dsv41_idx); radix bucket top-k with many blocks per row for batch-1 long pools, bitwise equal to the one-block form (dsa_topk_par); sparse MLA prefill as one workgroup per query over 64 heads (sp_mla); Hadamard-rotated FP4 index keys (ds4 rocm_indexer, vllm ds4_idx_hadamard); pool-aligned indexer sub-chunks so chunked prefill never splits a compression pool (vllm idx_pool_subchunks); force-tail of recent pools; split-KV flash-decoding for small query counts (vllm sparse_attn_split).
- **Where, re-expressed.** `attn.indexer-score/kv-fp4-e2m1/bf16`, `attn.indexer-topk/-/i64-keys`, `attn.sparse-sink/kv-fp4-e2m1/bf16` in `kernel.strixfold.sparse_attn` and the index-source layers of the TP4 executor. Re-expression: keep FP4 E2M1 keys and the exact top-512 with its tie rule; borrow the scan/WMMA mapping and the multi-block selection.
- **Gate.** bit identity for selection (exact top-k, same ties) and for scores when the order is kept; FIDELITY if score precision changes.
- **Gain (a) DSV4.1 TP4.** medium and growing with context: indexer scores are 4-8 % of a chunk at 32K and double per doubling of context; the top-512 selection is unmeasured. Agent prompts at 64K+ make this the third prefill lever.
- **Gain (b) GLM-5.3-Flash TP4.** high: GLM-5.3-Flash is a DSA + absorbed-MLA model; the indexer and sparse MLA are its attention.

### F07 Dense / flash attention and KV cache formats (29 modules)

- **Shape.** Quantised-KV flash-decoding split kernels on WMMA (kyojin qdec_hip), TurboQuant and 8-bit Hadamard-rotated paged caches, KVWIN exact fp16 sink plus recent-window ring, paged page tables with sharing and eviction, FP8 KV stores.
- **Where, re-expressed.** `kvcache.*` of a future GQA executor; for DSV4.1 the window ring is already bf16 and the compressed keys FP4. Re-expression: the sink-plus-window ring is the DSV4.1 window ring; the paged table is the shape a 16-slot prefix share needs (see F14).
- **Gate.** bit identity for layout-only changes; FIDELITY for any KV quantisation.
- **Gain (a) DSV4.1 TP4.** low: DSV4.1 KV is already compressed.
- **Gain (b) GLM-5.3-Flash TP4.** low-medium: GLM MLA latents are small; INT8 latent records (maya glm_lat8) are a FIDELITY lever for very long contexts.

### F08 Hyper-connections (mHC) (15 modules)

- **Shape.** Fused mHC mix with the HyperHead collapse (kyojin hc_mix, 2,432 lines) and a one-pass int8 R-row mix that reads each weight byte once for 2-4 verify rows; HC prefill with inverse reuse (enginelab hc_prefill); split Sinkhorn and an SGEMM form (ds4 rocm_hc, k53perf hc_sgemm); a fused / unfused crossover table (vllm ds4_mhc_cfg).
- **Where, re-expressed.** `hc.mix-prefill/dsv41-mhc-f32/bf16` (`kernel.strixfold.mhc_mix` `aux.hc_mix_pf`) plus the Triton `_hc_finish` (Sinkhorn 20 iterations, collapse, RMSNorm). Re-expression: fuse the finish into the mix at 5120 width (gufo's 4096 form is the same shape).
- **Gate.** bit identity if the Sinkhorn iteration order and fp32 rounding points are kept; FIDELITY otherwise.
- **Gain (a) DSV4.1 TP4.** medium at TP4: the mixes are 4 % of TP2 kernel time and every rank repeats them, so their share roughly doubles at TP4, plus the unmeasured Triton finish. Inferred +3-6 % prefill.
- **Gain (b) GLM-5.3-Flash TP4.** medium: GLM-5.3-Flash also mixes mHC streams every layer (maya glm_hc).

### F09 Linear attention and recurrence (GDN, KDA) (24 modules)

- **Shape.** Chunked WMMA delta-rule kernels with recompute fused (kyojin fla_gdn_fused_h, fla_kda_fused_h, hipfire-derived gdn_hf / kda_hf), KDA inter-chunk off-diagonal dots, fused forget gate + gated norm, GDN prefill scan with overlap (enginelab gdn_prefill), recurrent-state caches with speculative checkpoints.
- **Where, re-expressed.** no DSV4.1 slot (no linear-attention layers). GLM-5.3-Flash has KDA layers: the GLM executor's recurrent mixer; `ssm.*` interfaces for Qwen executors.
- **Gate.** FIDELITY (chunk size changes summation order) unless the chunk shape is kept.
- **Gain (a) DSV4.1 TP4.** none for DSV4.1 (no such layers); not closed: the chunked-scan schedule is the shape for any future hybrid model.
- **Gain (b) GLM-5.3-Flash TP4.** high: KDA is part of every GLM-5.3-Flash layer stack; ds4 glmtp splits KDA heads across ranks.

### F10 Norm, RoPE, elementwise glue, heads, sampling, chat templates (44 modules)

- **Shape.** Fused elementwise chains replacing torch chains, per-head norms with RoPE tail, on-device samplers in llama.cpp order, LM-head argmax plans, PLE gates, stop-string matchers, chat templates and tool-call fail-open.
- **Where, re-expressed.** `norm.*`, `pos.rope`, `residual.add`, `sample.*` and the serve front. Re-expression: launch fusion into the neighbouring kernel.
- **Gate.** bit identity for fusion; exact-match tests for templates and parsers.
- **Gain (a) DSV4.1 TP4.** low: under 2 % of prefill together; vllm toolcall_failopen is a serving-correctness item worth a look for agent traffic.
- **Gain (b) GLM-5.3-Flash TP4.** low-medium: the GLM chat template and glm47 tool parser are needed by the recipe's serve front.

### F11 Speculative decoding: DSpark, MTP, DFlash, wide verify windows, draft control (41 modules)

- **Shape.** Wide verify windows: a long confident copy span verified in one wide forward instead of several 8-row windows, priced by a cost table, tokens unchanged by the decision (enginelab spec_wide_window, mtp_window); pipelined DSpark (ds41_dspark: pair decode 26.8 -> 41.7-60.0 exact); acceptance-EMA draft depth and confidence-gated length (kyojin draft_depth, draft_confidence; vllm adaptive_k; maya spec_controller with a cost model); suffix-automaton and n-gram drafters; MTP heads.
- **Where, re-expressed.** `speculation.*` / `draft.dspark` and the TP4 executor's Drafter and `kernel.strixfold_tp4.aux_markov`. Re-expression: verify width becomes another forward-shape class of the TP4 executor; the cost table is re-measured on the rail.
- **Gate.** greedy tokens equal to drafts-off at the same build (SPEC point 3); with TP4_FILL=1 not exact, gate against the same build's own reference.
- **Gain (a) DSV4.1 TP4.** medium: decode is ~25 % of wall time; +20-40 % decode on copy-heavy agent replies is +5-10 % wall.
- **Gain (b) GLM-5.3-Flash TP4.** high: MTP + wide is what lifts the engine lab's GLM pair from 25.7 to 40-66 tok/s.

### F12 Tensor-parallel partitioning and collectives (56 modules)

- **Shape.** Exact TP join: partials summed in a declared order so a TP result is bitwise the single-rank one (enginelab core_tp_join, xport_tp_gather with a BF16 wire); intra-expert split by intermediate dimension with down rows on quant-block boundaries, two all-reduce points a block and agree_min votes for integer decisions (maya tp_context, enginelab ds41_expert_tp); KDA head split plus 50/50 routed experts (ds4 glmtp); KV sharded by position (enginelab glm_kvshard); an operation ordinal on every exchange (maya tp_comm_abi); GPU arrive/wait/add kernels around a host-mapped slab (ds4 rocm_tp, maya tp_kernels); a latency path for small all-reduces separate from bulk (vllm odl_ar2).
- **Where, re-expressed.** `runtime.strixfold_tp4.coll_rdma` (rdma_proxy world 4, lanes, swap12) and the gather sites of `executor.strixfold_tp4.dsv41`. Re-expression: the rail already runs RC writes with immediate data into slots; what transfers is the reduce-scatter-in-declared-order shape (fewer bytes per rank than an all-gather of fp32 partials at world 4) and the ordinal guard.
- **Gate.** bit identity when the declared summation order equals the current one (prompt-logits sha and ids on the gate's eight prompts); a BF16 wire is FIDELITY.
- **Gain (a) DSV4.1 TP4.** medium-high and the least certain: TP4 prefill runs at about the TP2 rate, so something does not scale; an all-gather of n x 5120 fp32 partials sends each rank's block to three peers. Inferred +5-20 % if gather time is not hidden.
- **Gain (b) GLM-5.3-Flash TP4.** high: the GLM recipe at TP4 is built on these partition rules.

### F13 Link transports and their drivers (USB4 / Thunderbolt / RoCE) (60 modules)

- **Shape.** Completion suppression and NAPI-style hybrid RX polling, DMA-BUF import so the NIC moves GPU-owned buffers with no CPU in the payload path (ciru tbstream_dmabuf, enginelab gpu_stream), MSI-X clear flushing and ring priming, write-with-immediate into rotating fixed-address windows the GPU polls (maya tp_rdma), lazy RC SEND/RECV over libibverbs with QP-before-MR teardown and buffer quarantine (ds4 k53perf_tp_roce), CRC32C on the device for wire integrity (enginelab xport_crc).
- **Where, re-expressed.** the pod rail is 100 GbE RoCE on CX6 through the SN2700, not USB4: every USB4 item is re-expressed on verbs. DMA-BUF zero copy -> a GPU-memory MR (peer-direct / dma-buf MR on the CX6) instead of the coherent host slab; hybrid RX polling -> CQ busy-poll with event fallback in the proxy thread; device CRC -> a debug-mode check on rdma_proxy trailers.
- **Gate.** bit identity (a transport moves bytes; equal prompt-logits sha and ids), plus the existing trailer/site checks.
- **Gain (a) DSV4.1 TP4.** low-medium: the rail already does RC writes; a GPU-memory MR removes one host-slab hop for chunked prefill gathers. Not closed by format: the ideas are verbs-level.
- **Gain (b) GLM-5.3-Flash TP4.** medium: the GLM recipe reuses the rail transport.

### F14 Prompt cache, prefix sharing and prefill scheduling (19 modules)

- **Shape.** Concurrent prefill of one shared prefix: jobs starting together on one prompt wait for the page owner instead of prefilling twice (kyojin sched_prefill_share); content-addressed NVMe pages chained by hash (kyojin prefix_disk, maya glm_prompt_cache); known-prompt cancellable chunked prefill and a prefix window (enginelab known_prompt, prefix_state, serve_prefix); chunked-prefill interleave so running decodes are not starved (sched_chunk_il); retention intervals aligned to speculative block drops (vllm apc_retention / apc_align); role-marker prefix detection (ds4 prompt_prefix); prompt compression by a small scorer (enginelab serve_pflash).
- **Where, re-expressed.** `cache.prompt` of the TP4 serve (`runtime.strixfold_tp4.core`, `lanes.zig` keeps at 2,048-row chunk ends; sf-disk) and the TP4 server's request queue. Re-expression: page owner = the slot that first claims a chunk-end key; content address = the existing state key chained over 2,048-row chunks.
- **Gate.** bit identity: a state resumed from a shared owner must equal the state the request would compute (the sf-disk invariant the TP4 gate already checks); pflash changes the prompt, so it is a quality gate, not FIDELITY.
- **Gain (a) DSV4.1 TP4.** high, traffic-dependent: prompt passes are ~75 % of wall time; every duplicated prefix pass removed is wall time saved one for one. Sizing needs only the serve logs (route records, request prefixes), no hardware.
- **Gain (b) GLM-5.3-Flash TP4.** high: the same front serves GLM.

### F15 Executors, engines, graphs and model families (123 modules)

- **Shape.** Whole-engine forms: graph-captured blocks replayed per step (kyojin bc_*, hy4_graph), compiled execution plans and prepared views (enginelab exec_program, execution), per-model engines (DS4.1, GLM-5.3, MiniMax, Qwen, Gemma 4, Hy4, Inkling, H3) and their prefill drivers, CPU reference engines (kyojin dsv41_reference).
- **Where, re-expressed.** `executor.model/*` beside `executor.strixfold_tp4.dsv41`; the GLM recipe takes one of these as its base (section 4). Re-expression: engines are composed, not swapped: their schedules (prefill drivers, step graphs) are the shapes.
- **Gate.** the executor's own reference (tokens and prompt-logits sha) plus FIDELITY against a dequantised FP32 reference.
- **Gain (a) DSV4.1 TP4.** low as parts; the CPU-exact DS4.1 reference (kyojin dsv41_reference) is a FIDELITY reference candidate for DSV4.1.
- **Gain (b) GLM-5.3-Flash TP4.** high: the GLM recipe base (enginelab glm_engine / glm_prefill, maya glm_model, kyojin glm_server).

### F16 Memory, residency, expert tiers, Engram I/O and weight prefetch (32 modules)

- **Shape.** Read-only side-stream prefetch of weights into the 32 MB MALL during latency-bound kernels, bit-identical by construction (kyojin qwd3_pf); GIL-free parallel pread / O_DIRECT row gathers for Engram (engram_io_c, enginelab ds41_engram read-ahead); expert tiers with prefetch and SSD streaming hotlists; memory admission guards and CMA-aware MemAvailable.
- **Where, re-expressed.** the TP4 decode step (L2_PREFETCH already +3.93 %), the Engram host step of `executor.strixfold_tp4.dsv41`, PLACE-managed placements. Re-expression: MALL prefetch extends L2_PREFETCH to the next streaming kernel's weights.
- **Gate.** bit identity (prefetch and I/O never touch math).
- **Gain (a) DSV4.1 TP4.** medium for decode (+3-8 % decode, inferred from the L2_PREFETCH result), low for wall time; Engram reads are replicated per rank and unmeasured.
- **Gain (b) GLM-5.3-Flash TP4.** medium: GLM has no Engram; MALL prefetch carries.

### F17 Serving fronts, benches, probes and tooling (35 modules)

- **Shape.** HTTP/Responses fronts, warm-up units, quality A/B probes (per-token NLL over fixed texts), context-retrieval probes, trace analysis, kernel parity programs.
- **Where, re-expressed.** `serve.*` and `gate/` tooling. Re-expression: the probes become gate inputs (vllm quality_probes and context_probes are ready-made FIDELITY / long-context checks).
- **Gate.** n/a (tooling); the probes are gates themselves.
- **Gain (a) DSV4.1 TP4.** low (tooling); the long-context watchword probes are a cheap check for any indexer change.
- **Gain (b) GLM-5.3-Flash TP4.** medium: the recipe needs a serve front and probes.

### F18 Non-ROCm backends (CUDA, Metal, Vulkan, Intel, RTX routes, NCCL plugins) (28 modules)

- **Shape.** CUDA twins of the HIP kernels (NVFP4 microscaled MoE, IMMA dense, MMA tiles, CUDA graphs), Metal/Vulkan/Intel backends, RTX PRO prefill routes.
- **Where, re-expressed.** no pod hardware runs them; the shapes still transfer: microscaled (MX/NVFP4-style) block scales -> a per-32 scale layout for an int8-activation expert route on gfx1151 WMMA (iu8 at the f16 rate); CUDA graph capture -> HIP graph capture (already used by the TP4 GRAPHS switch).
- **Gate.** as the HIP re-expression requires (bit identity for graph capture, FIDELITY for microscaled activations).
- **Gain (a) DSV4.1 TP4.** low: read as designs, not parts.
- **Gain (b) GLM-5.3-Flash TP4.** low.

### F19 Vision towers and image I/O (20 modules)

- **Shape.** Vision towers for DS4.1, GLM-5.3, MiniMax and Qwen, image decoders, the visual mixed-KV pack.
- **Where, re-expressed.** the TP4 vision path (vision.hip on rank 0, dormant in the text gate); a GLM vision recipe later.
- **Gate.** FIDELITY against the tower's reference; bit identity for decoder swaps.
- **Gain (a) DSV4.1 TP4.** low for text prefill; relevant only to image prompts.
- **Gain (b) GLM-5.3-Flash TP4.** low (GLM vision is out of the first recipe).

## 4. Ranking for (a) DSV4.1 TP4

Gains are of TP4 prefill throughput (tok/s) unless marked; wall time = 0.75 x prefill gain + 0.25 x decode gain.

| rank | transfer | modules (shape donors) | target in strixflerp | gate | expected gain | card |
|---|---|---|---|---|---|---|
| 1 | **C1 Grouped EXL3 experts pass at the TP4 slice**: one persistent launch per projection over (expert, row-tile) items, tokens sorted by expert, each trellis tile decoded once into LDS and shared by all row tiles of the expert, gate + up in one launch, register-blocked double-buffered WMMA with ascending K16 order | kernel.kyojin.exl3_moe_prefill, kernel.kyojin.exl3_moe_grouped, kernel.enginelab.core_tiled_wmma, kernel.enginelab.moe_core_prefill, kernel.ds4.k53perf_rocm_halo_moe | `kernel.strixfold.exl3_pf_experts` as a derived module under `catalog/flerp`, bound by the TP4 recipe through `[build.overlay]` | bit identity: 0 differing outputs in `gate/prefill_dsv41_bench.py` at TP4 slice shapes, then prompt-logits sha + ids on the 8 gate prompts | +10-25 % prefill (the fixed ~15 ms a TP2 pass is the R2 target; halving it was +20 % at TP2, and it is a larger share at TP4) | c058e5b8a3 |
| 2 | **C2 Exact reduce-scatter of the prefill partials in declared order**: each rank sums one column quarter of the four partials in the current rank order, then gathers the summed quarters; fewer bytes per rank than gathering whole fp32 partials at world 4, same fp32 adds in the same order | kernel.enginelab.core_tp_join, runtime.enginelab.xport_tp_gather, runtime.maya_tp.tp_context, runtime.maya_tp.tp_comm_abi, kernel.maya_tp.tp_kernels | `runtime.strixfold_tp4.coll_rdma` (rdma_proxy world 4) and the gather sites of `executor.strixfold_tp4.dsv41` | bit identity (prompt-logits sha + ids); a BF16 wire variant is FIDELITY and a separate arm | +5-20 % prefill if gathers are not hidden (section 2 item 2); zero if the TP4 layout already sends quarters, which the card checks first | 077b919eb5 |
| 3 | **C3 Shared-prefix prefill and finer kept states**: requests that start together on one prefix wait for the owner's chunk-end state instead of prefilling it again; content-addressed chunk keys chained over 2,048-row chunks | runtime.kyojin.sched_prefill_share, runtime.kyojin.prefix_disk, runtime.enginelab.prefix_state, runtime.enginelab.known_prompt, serve.enginelab.serve_prefix, runtime.maya_tp.glm_prompt_cache, runtime.vllm_strix.apc_retention, runtime.ds4.prompt_prefix | the TP4 server's request queue and `cache.prompt` (`lanes.zig` keeps, sf-disk) | bit identity: the state a waiter resumes from equals the state it would compute (the sf-disk identity) | traffic-dependent and possibly the largest: every duplicated prefix pass is prompt time removed one for one; sized from serve logs before any code | 11b89e522d |
| 4 | **C4 Wide verify windows and acceptance-driven draft depth** for the TP4 drafter: verify a long confident copy span in one wide forward, priced by a measured cost table; drop to short or no drafts when the acceptance EMA falls | runtime.enginelab.spec_wide_window, runtime.enginelab.mtp_window, executor.enginelab.ds41_dspark, runtime.kyojin.draft_depth, runtime.kyojin.draft_confidence, runtime.vllm_strix.adaptive_k, runtime.maya_tp.spec_controller | the Drafter of `executor.strixfold_tp4.dsv41` and its `aux_markov` candidates | greedy ids equal to drafts-off on the same build; row invariance across verify widths (runtime.kyojin.sched_row_groups is the shape) | +20-40 % decode on copy-heavy replies (the engine lab's DS4.1 DSpark gives 1.55-2.24x over plain; our drafts already take part of it), ~+5-10 % wall | c3dca97e7b |
| 5 | **C5 Fused mHC finish + mix in prefill**: the Sinkhorn finish, collapse and RMSNorm fused into the posted-stream mix at 5120 width, one launch twice a layer | kernel.kyojin.hc_mix, kernel.enginelab.hc_prefill, kernel.enginelab.core_hc, kernel.ds4.k53perf_rocm_hc_sgemm, kernel.ds4.rocm_hc, runtime.vllm_strix.ds4_mhc_cfg | `kernel.strixfold.mhc_mix` (`aux.hc_mix_pf`) plus the Triton `_hc_finish` | bit identity if the iteration order and rounding points are kept, else FIDELITY | +3-6 % prefill (4 % of TP2 kernel time, replicated per rank so ~8 % at TP4, plus the unmeasured Triton finish) | a56f8ef357 |
| 6 | Indexer scoring on WMMA and multi-block top-k at long context; pool-aligned indexer sub-chunks | kernel.kyojin.dsv41_idx, kernel.kyojin.dsa_topk_par, kernel.ds4.rocm_indexer, kernel.ds4.k53perf_rocm_halo_indexer, runtime.vllm_strix.idx_pool_subchunks | `attn.indexer-score`, `attn.indexer-topk` | bit identity for selection | 4-8 % of a chunk at 32K, doubling per context doubling: +2-4 % at 32K, more at 128K | (not carded) |
| 7 | Dense prompt GEMM tile core with per-pack autotune | kernel.enginelab.core_tiled_wmma, kernel.kyojin.exl3_gemm, kernel.kyojin.exl3_coop_autotune, runtime.kyojin.tune_cache | `kernel.strixfold.exl3` `exl3.gemm` | bit identity (K16 order) | ~5 % prefill (17 % share at 20-27 TFLOP/s) | (not carded) |
| 8 | MALL weight prefetch in decode, extending L2_PREFETCH | runtime.kyojin.qwd3_pf | TP4 decode step | bit identity | +3-8 % decode (L2_PREFETCH alone gave +3.93 %), ~1-2 % wall | (not carded) |
| 9 | GPU-memory MR for the rail slots (the DMA-BUF zero-copy idea on verbs) and CQ busy-poll | runtime.ciru_link.tbstream_dmabuf, runtime.enginelab_usb4.stream_rx_latency, runtime.maya_tp.tp_rdma, runtime.ds4.k53perf_tp_roce | `runtime.strixfold_tp4.coll_rdma` | bit identity | small unless C2 shows the host slab hop matters | (not carded) |
| 10 | Fused MoE half-layer for decode (one persistent launch) | kernel.kyojin.exl3_moe_fused, kernel.kyojin.dsv41_moe, kernel.kyojin.exl3_moe_valu | TP4 decode experts | bit identity | +10-20 % decode if launch-bound, ~3-5 % wall | (not carded) |

Every other family (F03, F04, F07, F09, F10, F13 drivers, F15-F19) is ranked below these for (a): either it has no
DSV4.1 slot today (F09: no linear-attention layers) or its share of the TP4 wall time is under ~2 %. None is closed:
section 3 says how each is re-expressed.

## 5. Ranking for (b) GLM-5.3-Flash TP4

| rank | shape | donors | why first |
|---|---|---|---|
| 1 | Intra-expert TP at world 4: every routed and dense FFN split by intermediate dimension on quant-block boundaries, two all-reduce points a block, agree_min votes for integer decisions | runtime.maya_tp.tp_context, kernel.enginelab.ds41_expert_tp, kernel.enginelab.core_tp_join | routing stays identical on every rank and load is even whatever the router picks; generalises from 2 to 4 ranks by slicing quarters |
| 2 | Attention split: KDA heads across ranks (ds4 glmtp), KV sharded by position for the MLA latents (enginelab glm_kvshard) | kernel.ds4.glmtp_rocm_tp, runtime.ds4.glmtp_tp, executor.enginelab.glm_kvshard, runtime.enginelab.xport_kvshard | the only published GLM-5.3 attention splits |
| 3 | Grouped EXL3 MoE prefill (C1 is the same shape) | kernel.kyojin.exl3_moe_prefill, kernel.enginelab.moe_core_prefill | the GLM prefill rides on it |
| 4 | Sparse MLA prefill + DSA indexer + exact top-k | kernel.kyojin.sp_mla, kernel.kyojin.dsa_topk_par, kernel.enginelab.attn_core_sparse_mla, kernel.maya_tp.glm_dsa, runtime.vllm_strix.idx_pool_subchunks | GLM's attention is DSA |
| 5 | MTP + wide verify windows | runtime.enginelab.glm_spec, runtime.enginelab.spec_wide_window, runtime.maya_tp.glm_mtp, kernel.kyojin.dflash2 | 25.7 -> 40-66 tok/s on the pair |
| 6 | Chunked KDA scan | kernel.kyojin.fla_kda_fused_h, kernel.kyojin.kda_inter, kernel.maya_tp.glm_kda | every GLM layer stack has KDA layers |
| 7 | Rail transport: the TP4 rdma_proxy (world 4, lanes) reused, with maya's operation ordinal | runtime.strixfold_tp4.coll_rdma (catalogue), runtime.maya_tp.tp_comm_abi | the pod already has a world-4 RoCE collective; nothing USB4 is needed |
| 8 | Prompt cache (C3's shape) | runtime.maya_tp.glm_prompt_cache, runtime.kyojin.prefix_disk | agent traffic |

### The recipe (card 4b63cad73f)

- **Base engine: the engine lab's GLM-5.3 engine** (executor.enginelab.glm_engine, glm_prefill, glm_kvshard,
  runtime.enginelab.glm_load / glm_spec / glm_xfer, kernel.enginelab.glm_attn / glm_kernels / glm_moe /
  glm_prefill_wmma). Why: the best published GLM-5.3-Flash numbers on this GPU, an EXACT profile that is bitwise its
  declared reference (the same discipline as strixflerp's gates), and EXL3 K4/K5 trellis weights, the StrixFold format
  family. Its compile gate already builds 77 of 93 standalone modules for gfx1151; the 16 that fail only need the
  origin's single include tree, which a recipe assembly gives.
- **World 2 -> 4.** Expert TP re-cut from 1152-row halves to quarters on the same quant-block rule; KV shard by position
  over four ranks; the exact TP join extended to four partials in a declared order, so world 4 stays bitwise world 1.
- **Transport re-expressed.** The engine lab's channel has USB4 backends (direct_transport, dist_gpu_direct) and a
  host-staged one (dist_host_staged). The rail gets a verbs backend: RC writes with immediate data into rotating
  windows, the shape of StrixFold's world-4 rdma_proxy and maya's tp_rdma, with an operation ordinal on every exchange.
- **Weights only through the tools.** The source checkpoint and the converted pack are registered in
  `/data/pod/weights/models.tsv` and staged with `pod-weights stage` on control-node; worker copies go through
  `loom-swarm.place place request` (read `place status` first; StrixFold's Engram placements are protected). No raw
  copies, no cold-storage loads.
- **Gate against a reference.** PASS = (1) TP4 token ids and prompt-logits sha equal to the same engine at world 1 on the
  same pack, which the exact join promises, and (2) the FIDELITY rule (`docs/FIDELITY.md`: KL against the dequantised
  FP32 reference, top-1 agreement) on the gate prompts. The fallback reference if world 1 does not fit one node is the
  engine lab's pair (world 2) on the same pack.
- **When it runs.** Only in a window the pod picks, with the TP4 serve brought back afterwards; GLM does not displace
  DSV4.1 as the standing engine. The window run is its own experiment card written from the gate's `--plan`.
- Not chosen as base, and why: vLLM TP=4 over RCCL (vllm_strix's patches) is the least new code but the slowest
  published numbers (~290 prefill, 20-28 decode at TP2); maya-tp's IQ2_XXS GGUF path is TP2-only with a lower-bit pack;
  Kyojin is one box and its EXL3-Yamz conversion pipeline is private. Each stays a donor of shapes (section 5 table).

## 6. Cards

Created under GLM53-HARVEST ed900d7238 with card-intake, all in todo. C2 and C3 are research cards: their first act is to size the gain (gather bytes from source; duplicate prefix passes from serve logs), and a DROP verdict closes them. The kernel and drafter cards build the module and commit the gate criteria; each hardware run is its own experiment card.


| card | id | rank |
|---|---|---|
| GLM53-XFER-EXPERTS: grouped EXL3 experts pass for the TP4 slice in strixflerp (work) | c058e5b8a3 | (a) 1 |
| GLM53-XFER-RS: count TP4 prefill gather bytes and decide an exact reduce-scatter (research) | 077b919eb5 | (a) 2 |
| GLM53-XFER-PREFIX: size shared-prefix prefill on the TP4 serve from its logs (research) | 11b89e522d | (a) 3 |
| GLM53-XFER-WIDE: wide verify windows and draft-depth control for the TP4 drafter (work) | c3dca97e7b | (a) 4 |
| GLM53-XFER-MHC: fused mHC finish and mix for DSV4.1 TP4 prefill (work) | a56f8ef357 | (a) 5 |
| GLM53-TP4-RECIPE: GLM-5.3-Flash TP4 recipe on the engine lab's GLM engine (work) | 4b63cad73f | (b) |

## 7. Measured, inferred, not verified

Read from source here: the module records, the TP4 slot collective's world-4 layout (`rdma_core.zig`, `rdma_proxy.zig`:
RC write with immediate data, chunked gathers over lanes, nothing on tcp_host at world 4). Cited: every origin number
(their READMEs, through the ORIGIN.toml files), the TP2 kernel table and the TP4 serving facts. Inferred: every gain, the
four "does not scale" causes in section 2, the gather byte count at TP4. Not verified: any number on F D C A, the TP4
per-phase split, GLM-5.3-Flash's exact expert count and layer mix.

## Appendix A. Every GLM53 module by family (872)

The family header in section 3 carries the shape, where it goes, the gate and the gain; each row says what the module
holds. ds4 overlay rows name their branch: glmtp (davidcanar glm-tp-rocm 8ebf672); k53flash (kyuz0 glm53-flash 4bdbcc2); k53perf (kyuz0 glm53-performance f46631e); k53prefill (kyuz0 glm53-flash-prefill-restoration 4fef372); kmultinode (kyuz0 rocm-multi-node 89c1ea9).

### F01 MoE expert prefill (grouped expert GEMMs) (30)

| module | what it holds |
|---|---|
| `kernel.ds4.k53flash_mmq_moe` | [k53flash overlay] mmq_moe family: cuda/mmq/ds4_mmq.cu changed (+61 -782 lines); cuda/mmq/ds4_mmq.h changed (+1 -3... |
| `kernel.ds4.k53perf_mmq_moe` | [k53perf overlay] mmq_moe family: cuda/mmq/ds4_mmq.cu changed (+113 -773 lines); cuda/mmq/ds4_mmq.h changed (+22 -0... |
| `kernel.ds4.k53perf_rocm_halo_moe` | [k53perf overlay] rocm_halo_moe family: rocm/ds4_rocm_halo_moe.h added (10 lines); rocm/halo/moe/NOTICE.md added (8... |
| `kernel.ds4.k53perf_rocm_halo_q2` | [k53perf overlay] rocm_halo_q2 family: rocm/ds4_rocm_halo_q2.cuh added (6 lines); rocm/ds4_rocm_halo_q2.h added (7... |
| `kernel.ds4.k53perf_rocm_v41_gate_up` | [k53perf overlay] rocm_v41_gate_up family: rocm/ds4_rocm_v41_gate_up.h added (10 lines); rocm/v41_gate_up/NOTICE.md... |
| `kernel.ds4.k53prefill_mmq_moe` | [k53prefill overlay] mmq_moe family: cuda/mmq/ds4_mmq.cu changed (+61 -782 lines); cuda/mmq/ds4_mmq.h changed (+4 -7... |
| `kernel.ds4.mmq_d2r` | the gated D2R path: the Q2_K MoE down-GEMM and dense Q8_0 on the kind-5 aligned artifact (ds4_mmq_d2r.cu, CUDA... |
| `kernel.ds4.mmq_mmid` | llama.cpp's mm_ids helper: expert-id to row mapping for the MoE matmuls |
| `kernel.ds4.mmq_moe` | ds4's quantized-matmul C ABI over llama.cpp's mul_mat_q (mmq.cuh): dense Q8_0 / Q2_K / IQ2_XXS and MoE _id variants,... |
| `kernel.enginelab.glm_moe` | GLM MoE and its trellis device kernels: 3 origin files under src/glm |
| `kernel.enginelab.glm_prefill_wmma` | GLM WMMA prefill: WMMA attention, KDA, MLA (GEMM VOPD, WMMA), router, and the generic, MoE and x WMMA prefill |
| `kernel.enginelab.inkling2_moe` | Inkling 2 Mini MoE and experts (HIP, dispatch, reference, layout): 6 origin files under src/inkling2 |
| `kernel.enginelab.lab_moe_experts` | the MoE experts lab kernel (lab/moe_experts): 3 origin files under lab/moe_experts |
| `kernel.enginelab.lab_moe_iq4_xs` | the MoE IQ4_XS lab kernel (lab/moe_iq4_xs): 3 origin files under lab/moe_iq4_xs |
| `kernel.enginelab.lab_moe_q8_0` | the MoE Q8_0 lab kernel with its qualification script and test (lab/moe_q8_0): 5 origin files under lab/moe_q8_0 |
| `kernel.enginelab.minimax_moe` | MiniMax MoE: mmx MoE and traits, prefill (core, IU4) and trellis device; the MoE (HIP) entry |
| `kernel.enginelab.moe_core_grouped` | the grouped expert GEMM family: grouped core, dispatch, policy and rows, grouped F16 / A8-F16 / integer decode / IU4... |
| `kernel.enginelab.moe_core_microscaled` | microscaled (MX / NVFP4-style) grouped MoE: the R1 microscaled geometry, pipe and CUDA plans, NVFP4 activation rule... |
| `kernel.enginelab.moe_core_prefill` | grouped MoE prefill kernel bodies: the HIP MoE prefill cores (p1, core2, g3, da16, GSQ FP16 / WMMA, F16x project and... |
| `kernel.enginelab.moe_prefill_launch` | MoE prefill launch pieces: prefill commit, route, small, tiers and tile64 bodies, tile64 down body and the... |
| `kernel.enginelab.moe_q4k_a8` | Q4_K x A8 MoE: the Q4K-A8 contract, known rows, probe and runtime, the A8 down contract, q51 grouped down (contract,... |
| `kernel.kyojin.exl3_moe_grouped` | Grouped prefill tier of the fused MoE: token-parallel Hadamards and a grouped GEMM where one CTA owns one (expert,... |
| `kernel.kyojin.exl3_moe_prefill` | Grouped EXL3 MoE prefill for gfx11.5 WMMA (2,027 lines): tokens sorted by expert, every projection one launch over... |
| `kernel.kyojin.gm_pfmoe` | Generic GGUF MoE prefill experts for gfx1151 WMMA: fused dequant of the stored k-quant bytes and a grouped fp16 GEMM... |
| `kernel.kyojin.gmmq` | Routed (MoE) HIP MMQ for GGUF Q4_K / Q5_K / Q5_1 / Q8_0: out[t * n_used + s] = dequant(W[ids[t, s]]) @ x[t] with... |
| `kernel.kyojin.moe_unswizzle` | Unswizzle of CPU-offloaded expert weights streamed back to the GPU for prefill (the band-swizzled VBMI tile order to... |
| `kernel.kyojin.routed_f16` | Routed F16-WMMA prefill MoE for gfx1151: RoutedPairF16 / RoutedDownF16 / pad-bounds / scatter kernels and the MoE... |
| `kernel.maya_tp.glm_batch` | GLM's batched (several rows / conversations) expert and dense kernels (glm_batch.cu): the multi-row forms of the... |
| `kernel.maya_tp.prefill_moe_mmq` | the llama.cpp MMQ view of the prompt experts: moe_mmq.cu and glm-dsa's moe_mmq_strata.cu compile ggml-cuda's MMQ... |
| `kernel.maya_tp.prefill_moe_rdna4` | HIP-only GLM prompt experts adapted from Niko1221/Strata fb58e0d (src/prefill/moe_fused_rdna4.cu): IQ prompt experts... |

### F02 Dense prompt GEMMs, tile cores and GEMM tuning (50)

| module | what it holds |
|---|---|
| `kernel.ds4.glmtp_rocm_matmul` | [glmtp overlay] rocm_matmul family: rocm/ds4_rocm_matmul.cuh changed (+105 -0 lines). Kernels not in the base:... |
| `kernel.ds4.k53flash_rocm_hipblaslt` | [k53flash overlay] rocm_hipblaslt family: rocm/ds4_rocm_hipblaslt.cuh changed (+4 -180 lines). The family: hipBLASLt... |
| `kernel.ds4.k53flash_rocm_matmul` | [k53flash overlay] rocm_matmul family: rocm/ds4_rocm_matmul.cuh changed (+4 -35 lines). The family: F16 matmuls on... |
| `kernel.ds4.k53perf_rocm_halo_output_a` | [k53perf overlay] rocm_halo_output_a family: rocm/ds4_rocm_halo_output_a.cuh added (18 lines);... |
| `kernel.ds4.k53perf_rocm_halo_output_a_uncached` | [k53perf overlay] rocm_halo_output_a_uncached family: rocm/ds4_rocm_halo_output_a_uncached.cuh added (81 lines);... |
| `kernel.ds4.k53perf_rocm_halo_output_b` | [k53perf overlay] rocm_halo_output_b family: rocm/ds4_rocm_halo_output_b_module.cuh added (34 lines);... |
| `kernel.ds4.k53perf_rocm_halo_q8` | [k53perf overlay] rocm_halo_q8 family: rocm/ds4_rocm_halo_q8.cuh added (241 lines). Kernels not in the base:... |
| `kernel.ds4.k53perf_rocm_halo_shapes` | [k53perf overlay] rocm_halo_shapes family: rocm/ds4_rocm_halo_shapes.h added (69 lines). The family: the... |
| `kernel.ds4.k53perf_rocm_halo_shared_down` | [k53perf overlay] rocm_halo_shared_down family: rocm/ds4_rocm_halo_shared_down.cuh added (111 lines). Kernels not in... |
| `kernel.ds4.k53perf_rocm_halo_shared_gu` | [k53perf overlay] rocm_halo_shared_gu family: rocm/halo/shared_gu/convert.hpp added (15 lines);... |
| `kernel.ds4.k53perf_rocm_hipblaslt` | [k53perf overlay] rocm_hipblaslt family: rocm/ds4_rocm_hipblaslt.cuh changed (+20 -6 lines). The family: hipBLASLt... |
| `kernel.ds4.k53perf_rocm_matmul` | [k53perf overlay] rocm_matmul family: rocm/ds4_rocm_matmul.cuh changed (+133 -26 lines). The family: F16 matmuls on... |
| `kernel.ds4.k53prefill_rocm_hipblaslt` | [k53prefill overlay] rocm_hipblaslt family: rocm/ds4_rocm_hipblaslt.cuh changed (+4 -180 lines). The family:... |
| `kernel.ds4.k53prefill_rocm_matmul` | [k53prefill overlay] rocm_matmul family: rocm/ds4_rocm_matmul.cuh changed (+4 -35 lines). The family: F16 matmuls on... |
| `kernel.ds4.kmultinode_rocm_hipblaslt` | [kmultinode overlay] rocm_hipblaslt family: rocm/ds4_rocm_hipblaslt.cuh changed (+4 -180 lines). The family:... |
| `kernel.ds4.kmultinode_rocm_matmul` | [kmultinode overlay] rocm_matmul family: rocm/ds4_rocm_matmul.cuh changed (+26 -535 lines). The family: F16 matmuls... |
| `kernel.ds4.mmq_quantize` | llama.cpp's activation quantizers for MMQ: q8_1, MMQ q8_1 / MXFP4 / NVFP4 tiles |
| `kernel.ds4.rocm_hipblaslt` | hipBLASLt state, plan cache and version checks for the GEMMs that go to the library (HIP only); a fragment of the... |
| `kernel.ds4.rocm_matmul` | F16 matmuls on WMMA for small and tiny M (small-M, tiny-M 24, tiny-batch wave) and their dispatch; a fragment of the... |
| `kernel.enginelab.core_activation_quant` | the shared activation quantiser (activation_quant.hpp): 1 origin files under src/core |
| `kernel.enginelab.core_bf16` | BF16 kernels of the shared core: dense row plans and narrow plans, the BF16 LM head, the BF16 projection batch, the... |
| `kernel.enginelab.core_f16_dense` | FP16 dense forms of the shared core: tall and hi-lo FP16 dense, Q8 x A8 dense, the hi-lo math, the dense linear and... |
| `kernel.enginelab.core_integer` | the integer operand for DeepSeek V4.1 GGUF blocks (integer_operand.hpp, integer_math.hpp, the plain / tall / wide /... |
| `kernel.enginelab.core_iu4` | stored INT4 (IU4) support: the IU4 pack builder and plans, IU4 x A4 dense projections with their layout and... |
| `kernel.enginelab.core_lookup` | the lookup catalog and lookup plan of the shared core: 2 origin files under src/core |
| `kernel.enginelab.core_tall` | tall (many-row) linear plans: the tall chunk policy, tall family selection and the tall K plan |
| `kernel.enginelab.core_tiled_wmma` | the shared tiled prefill core adopted from strix-trellis (namespace strix::core): the register-blocked,... |
| `kernel.enginelab.dense_launch` | dense launch pieces: A8 G64 dense, FP16 tiled and WMMA dense, dense rows tile and wave, the experimental dense... |
| `kernel.enginelab.h3_gemm` | H3 GEMMs (src/h3/gemm): 6 origin files under src/h3 |
| `kernel.enginelab.integer_projection` | integer and packed projections: the integer projection (HIP), its execution and runtime, packed projection,... |
| `kernel.enginelab.lab_integer_projection` | the integer projection lab kernel with offline compile scripts (lab/integer_projection) |
| `kernel.enginelab.lab_low_precision` | the low-precision lab: intrinsic probe, compile probe and host reference (lab/low_precision) |
| `kernel.enginelab.lab_packed_projection` | the packed projection lab: IQ4_XS and Q8 projections (lab/packed_projection): 4 origin files under lab/packed_projection |
| `kernel.enginelab.lab_q4k_a8` | the Q4_K x A8 dot4 compile probe (lab/q4k_a8): 1 origin files under lab/q4k_a8 |
| `kernel.enginelab.ops_a8` | A8 activation ops: group and quant device kernels; and the chain prep plan: 3 origin files under src/ops |
| `kernel.kyojin.exl3_coop_autotune` | Autotuner of the cooperative EXL3 GEMM: times candidate kernels on the first call of every (row class, K, N, bits,... |
| `kernel.kyojin.exl3_gemm` | EXL3 GEMM (trellis decode fused into the MMA tiles) with Kyojin's changes, including fp16 accumulation folded into... |
| `kernel.kyojin.exl3_kernel_map` | Kernel selection table of the EXL3 GEMM: (K bits, codebook, shape class) to kernel instance and launch geometry |
| `kernel.kyojin.fq_gemm` | Fused dequant-GEMV / GEMM for foreign-quant weights (M = 1..16): weights dequantised on the fly with the bit-exact... |
| `kernel.kyojin.fq_mmq_few` | Few-row MMQ (2..24 activation rows) for dense GGUF Q8_0 / Q4_K / Q5_K / Q6_K / Q3_K / IQ4_XS: int8 MMA on raw GGUF... |
| `kernel.kyojin.hgemm` | Dense fp16 GEMM of the extension with Kyojin's row-padded operand support (a non-power-of-two row pitch such as... |
| `kernel.maya_tp.ds4_lt` | DeepSeek-V4.1's prompt-path dense GEMMs through hipBLASLt (ds4_lt.cu / ds4_lt.hpp) |
| `kernel.maya_tp.prefill` | the prompt path's kernels: the batched prompt kernels (src/prefill/kernels.cu), the dense GEMM dispatch through... |
| `kernel.maya_tp.quantize_act` | activation quantization to Q8_0 and Q8_K (and back) the way ggml_mul_mat converts src1 to the weight's vec_dot_type,... |
| `runtime.kyojin.tune_cache` | Per-pack tuning cache of the EXL3 GEMM autotuner, the shipped dense GEMM seed and the dense warm-up |
| `runtime.vllm_strix.aiter_gemm_configs` | gfx1151 tile tables for aiter's Triton GEMMs (A8W8, A8W8 blockscale, preshuffled blockscale, batched A8W8 with... |
| `runtime.vllm_strix.fp8_prefill` | DeepSeek-V4 block-FP8 linears at prefill sizes: the fp8 weight is dequantised once to an exact bf16 copy by a HIP... |
| `tool.enginelab.lab_native_campaign` | the native numerical campaign lab (lab/native_campaign): 6 origin files under lab/native_campaign |
| `tool.kyojin.hipsp_probes` | WMMA peak-rate probes for gfx1151: two kernels that issue back-to-back v_wmma f16 ops to measure the matrix-core ceiling |
| `tool.maya_tp.hipblaslt_tuning` | the hipBLASLt tuning tables and tuner (tools/hip): solution tables per gfx target and hipBLASLt version (gfx1151... |

### F03 Weight formats, codecs, quantisers and loaders (41)

| module | what it holds |
|---|---|
| `format.enginelab.gguf` | GGUF: reader and writer, ggml dequantisation and weight formats, IQ4_NL and IQ4_XS |
| `format.enginelab.hip_weight` | device weight sources (src/hip/weight): ggml and GSQ sources, GSQ FP16 and group codecs, GSQ plan and partial trace,... |
| `format.enginelab.model_data` | model data: per-model constant tables (DeepSeek V4.1 Flash, GLM-5.3 Flash, Hy4 Flash, Inkling2 Mini, MiniMax M3.1... |
| `format.enginelab.qwen_grids` | the i-quant codebook grids for Qwen (IQ2_S, IQ3_S, IQ3_XXS; from ggml, MIT): 3 origin files under include/strix |
| `format.enginelab.weight_adapters` | per-model tensor naming maps (src/weight/adapters): DS41, Gemma 4, GLM, H3, Hy4 (and DFly), MiniMax, Qwen, vision... |
| `format.enginelab.weight_core` | the shared weight layer (src/weight, include/strix/weight): admission, block draft, cache, calibration moments and... |
| `format.enginelab.weight_files` | model files: weight archive and pack, mapped model file, file map and snapshot, unbuffered file reads, asset... |
| `format.enginelab.weight_formats` | weight formats (src/weight/formats, include/strix/weight/formats): catalog, channel quant, DS41, EXL3, FP8, ggml,... |
| `format.enginelab.weight_readers` | weight readers: GGUF, pack and safetensors: 3 origin files under src/weight |
| `format.kyojin.fq_formats` | Python reference decoders of the foreign formats (k-quants, IQ2_XXS, NVFP4, Q8_0, legacy, gap formats), GGUF tables... |
| `format.kyojin.gguf_loader` | GGUF loading for the EXL3 engine: the tensor collection (raw quantized payloads with the SafetensorsCollection call... |
| `format.kyojin.safetensors_loader` | Python safetensors collection as Kyojin changed it (1,293 lines) |
| `format.maya_tp.gguf_reader` | the header-only GGUF reader and dequantizer with their CLIs (src/artifact: strata-gguf, strata-dequant;... |
| `format.maya_tp.ple_reader` | the PLE table reader (src/ngram/ple_reader.cpp, include/strata/ngram/ple_reader.hpp): the per-layer embedding table... |
| `kernel.enginelab.core_hadamard` | the Hadamard transform helper of the shared core: 1 origin files under src/core |
| `kernel.enginelab.core_trellis` | the one EXL3 trellis weight codec of the engine (trellis_format.hpp, trellis_codec.hpp: ring, codebook and reader... |
| `kernel.enginelab.ds41_trellis` | DS41 trellis kernels: trellis linear and grouped trellis (HIP) and the scalar codecs |
| `kernel.enginelab.gemma4_exl3` | Gemma 4 EXL3 device decode (with its notices) (src/gemma4/exl3): 12 origin files under src/gemma4 |
| `kernel.kyojin.exl3_codec` | EXL3 trellis decode primitives as Kyojin changed them: bit-window extraction (exl3_dq.cuh), the 3INST / mcg / mul1... |
| `kernel.kyojin.exl3_hadamard` | Hadamard rotations of EXL3 (device butterflies and the host Paley construction of non-power-of-two sizes) |
| `kernel.kyojin.exl3_pack` | EXL3 tile packing and reconstruction (packed 16x128 tiles back to fp16 weights) |
| `kernel.kyojin.exl3_quantize` | EXL3 trellis quantizer (Viterbi over tail-biting rings): the templated kernel, sm_120-tuned dense specialisations... |
| `kernel.kyojin.fq_common` | Shared layer of the foreign-quant kernels (GGUF k-quants and IQ formats, NVFP4, FP8 on the EXL3 engine): host entry... |
| `kernel.kyojin.fq_dequant` | Foreign-quant dequant to fp16: Q8_0 / IQ2_XXS / NVFP4 (streaming 2D grids) and the k-quants Q4_K / Q5_K / Q6_K /... |
| `kernel.kyojin.fq_mlp` | Fused shared-expert up GEMV + relu2 epilogue (torch-exact) for foreign-quant MLPs, batch 1 |
| `kernel.kyojin.fq_triton` | Triton kernels of the foreign formats: IQ and gap-format GEMVs, row gathers, shared-gate, f16 GEMV, MXFP4 MoE and... |
| `kernel.maya_tp.cpu_exl3` | EXL3 routed experts on the CPU (exl3_cpu), the scalar EXL3 reference (exl3_ref) and the CPU EXL3 bench (exl3_cpu_bench) |
| `kernel.maya_tp.cpu_iq` | the CPU expert rows in SIMD: i-quant rows in 256-bit and 512-bit lanes for several tokens (iq_avx2, iq_avx512), the... |
| `kernel.maya_tp.ds4_gguf` | a DeepSeek-V4.1 GGUF's BF16 matrices and MXFP4 experts on the device (ds4_gguf.cu / .cuh, unsloth's... |
| `kernel.maya_tp.ds4_iq` | DeepSeek-V4.1's low-bit routed experts in ggml block types on the device (ds4_iq.cu): the i-quants with IQ1 codes... |
| `kernel.maya_tp.exl3` | the EXL3 trellis-quant kernels: the current decode / prompt GEMVs and expert kernels (exl3, compiled... |
| `kernel.maya_tp.mm3_exl3` | maya-mm3's EXL3 device kernels (mm3_exl3.cu / .hpp), a copy of the shared EXL3 kernels shaped for MiniMax-M3.1 |
| `kernel.maya_tp.mm3_gguf` | maya-mm3's GGUF weights on the device (mm3_gguf.cu / .cuh / mm3_gguf_host.hpp): activation rounding, the dense and... |
| `kernel.maya_tp.mm3_gv` | maya-mm3's dense GEMVs over a GGUF's K-quant and Q8_0 matrices (mm3_gv.cu / .cuh): attention, shared and dense FFN... |
| `runtime.ds4.layer_pack` | the monotonic-contiguous multi-tier layer placement packer (devices then CPU), pure C99 |
| `runtime.kyojin.foreign_linear` | Foreign-quant linear module and the foreign MoE prefill paths (generic, IQ, KQ): picks the fq_* kernel by format and... |
| `tool.ds4.gguf_glm53` | GLM-5.3 quantization scripts: full and recipe quantizers, manifest and GGUF validation |
| `tool.ds4.gguf_quants` | the GGUF quantization facade in C (float conversion, q4_1, q8_0, q2_K, q4_K, iq2_xxs; derived from GGML's MIT... |
| `tool.ds4.gguf_qwen4` | Qwen3.8 Flash Next scripts: convert, IQ2, native n-grams, pack and the Unicode table generator |
| `tool.enginelab.lab_gsq` | the GSQ lab: content identity, tensor dequantisation and a dequant comparison against llama.cpp (lab/gsq) |
| `tool.kyojin.dsv41_convert` | DS4.1 pack conversion: driver, pack layout and spec, calibration (streaming), synthetic packs for tests |

### F04 Decode GEMVs and row kernels (71)

| module | what it holds |
|---|---|
| `kernel.ds4.dsv41` | DeepSeek V4.1-specific GPU operations shared by CUDA and ROCm (included after the shared kernels): BF16, candidates,... |
| `kernel.ds4.glmtp_dsv41` | [glmtp overlay] dsv41 family: ds4_deepseek41_cuda.cuh changed (+30 -249 lines); ds4_deepseek41_gpu.h changed (+2... |
| `kernel.ds4.glmtp_rocm_common` | [glmtp overlay] rocm_common family: rocm/ds4_rocm_common.cuh changed (+12 -2 lines). The family: shared device... |
| `kernel.ds4.glmtp_rocm_compat` | [glmtp overlay] rocm_compat family: ds4_rocm_compat.cu changed (+5 -4 lines). The family: the ROCm host shims... |
| `kernel.ds4.glmtp_rocm_q8` | [glmtp overlay] rocm_q8 family: rocm/ds4_rocm_q8.cuh changed (+45 -0 lines). Kernels not in the base:... |
| `kernel.ds4.glmtp_rocm_unavailable` | [glmtp overlay] rocm_unavailable family: ds4_rocm_unavailable.cu changed (+0 -9 lines). The family: C-linkage stubs... |
| `kernel.ds4.k53flash_mmq_common` | [k53flash overlay] mmq_common family: cuda/mmq/VENDOR.md changed (+0 -8 lines). The family: the vendored llama.cpp... |
| `kernel.ds4.k53flash_mmq_ggml_stubs` | [k53flash overlay] mmq_ggml_stubs family: cuda/mmq/ds4_ggml_stubs.cu changed (+0 -2 lines). The family: the minimum... |
| `kernel.ds4.k53flash_mmq_mmvq` | [k53flash overlay] mmq_mmvq family: cuda/mmq/mmvq.cu changed (+16 -19 lines). The family: llama.cpp's mul_mat_vec_q... |
| `kernel.ds4.k53flash_mmq_repack` | [k53flash overlay] mmq_repack family: cuda/mmq/ds4_repack.cu changed (+4 -38 lines); cuda/mmq/ds4_repack.h changed... |
| `kernel.ds4.k53flash_rocm_compat` | [k53flash overlay] rocm_compat family: ds4_rocm_compat.cu changed (+13 -6 lines). The family: the ROCm host shims... |
| `kernel.ds4.k53flash_rocm_q8` | [k53flash overlay] rocm_q8 family: rocm/ds4_rocm_q8.cuh changed (+449 -0 lines). Kernels not in the base:... |
| `kernel.ds4.k53flash_rocm_unavailable` | [k53flash overlay] rocm_unavailable family: ds4_rocm_unavailable.cu changed (+0 -3 lines). The family: C-linkage... |
| `kernel.ds4.k53perf_dsv41` | [k53perf overlay] dsv41 family: ds4_deepseek41_cuda.cuh changed (+224 -524 lines); ds4_deepseek41_gpu.h changed (+79... |
| `kernel.ds4.k53perf_mmq_common` | [k53perf overlay] mmq_common family: cuda/mmq/VENDOR.md changed (+0 -8 lines). The family: the vendored llama.cpp... |
| `kernel.ds4.k53perf_mmq_mmvq` | [k53perf overlay] mmq_mmvq family: cuda/mmq/mmvq.cu changed (+16 -19 lines). The family: llama.cpp's mul_mat_vec_q... |
| `kernel.ds4.k53perf_mmq_repack` | [k53perf overlay] mmq_repack family: cuda/mmq/ds4_repack.cu changed (+0 -1 lines). The family: the aligned... |
| `kernel.ds4.k53perf_rocm_compat` | [k53perf overlay] rocm_compat family: ds4_rocm_compat.cu changed (+12 -4 lines). The family: the ROCm host shims... |
| `kernel.ds4.k53perf_rocm_current_api_compat` | [k53perf overlay] rocm_current_api_compat family: rocm/ds4_rocm_current_api_compat.cuh changed (+99 -2 lines). The... |
| `kernel.ds4.k53perf_rocm_q8` | [k53perf overlay] rocm_q8 family: rocm/ds4_rocm_q8.cuh changed (+62 -4 lines). Kernels not in the base:... |
| `kernel.ds4.k53perf_rocm_unavailable` | [k53perf overlay] rocm_unavailable family: ds4_rocm_unavailable.cu changed (+0 -3 lines). The family: C-linkage... |
| `kernel.ds4.k53perf_rocm_v41` | [k53perf overlay] rocm_v41 family: rocm/ds4_rocm_v41.cuh added (2096 lines). Kernels not in the base:... |
| `kernel.ds4.k53prefill_mmq_common` | [k53prefill overlay] mmq_common family: cuda/mmq/VENDOR.md changed (+0 -8 lines); cuda/mmq/vendors/hip.h changed (+0... |
| `kernel.ds4.k53prefill_mmq_ggml_stubs` | [k53prefill overlay] mmq_ggml_stubs family: cuda/mmq/ds4_ggml_stubs.cu changed (+4 -65 lines). The family: the... |
| `kernel.ds4.k53prefill_mmq_mmvq` | [k53prefill overlay] mmq_mmvq family: cuda/mmq/mmvq.cu changed (+16 -19 lines). The family: llama.cpp's... |
| `kernel.ds4.k53prefill_mmq_repack` | [k53prefill overlay] mmq_repack family: cuda/mmq/ds4_repack.cu changed (+4 -38 lines); cuda/mmq/ds4_repack.h changed... |
| `kernel.ds4.k53prefill_rocm_compat` | [k53prefill overlay] rocm_compat family: ds4_rocm_compat.cu changed (+13 -6 lines). The family: the ROCm host shims... |
| `kernel.ds4.k53prefill_rocm_q8` | [k53prefill overlay] rocm_q8 family: rocm/ds4_rocm_q8.cuh changed (+449 -0 lines). Kernels not in the base:... |
| `kernel.ds4.k53prefill_rocm_unavailable` | [k53prefill overlay] rocm_unavailable family: ds4_rocm_unavailable.cu changed (+0 -3 lines). The family: C-linkage... |
| `kernel.ds4.kmultinode_rocm_common` | [kmultinode overlay] rocm_common family: rocm/ds4_rocm_common.cuh changed (+1 -41 lines). The family: shared device... |
| `kernel.ds4.kmultinode_rocm_current_api_compat` | [kmultinode overlay] rocm_current_api_compat family: rocm/ds4_rocm_current_api_compat.cuh changed (+85 -229 lines).... |
| `kernel.ds4.kmultinode_rocm_q8` | [kmultinode overlay] rocm_q8 family: rocm/ds4_rocm_q8.cuh changed (+442 -144 lines). Kernels not in the base:... |
| `kernel.ds4.mmq_common` | the vendored llama.cpp ggml-cuda headers the MMQ kernels compile against (common.cuh, mma.cuh, vecdotq.cuh,... |
| `kernel.ds4.mmq_ggml_stubs` | the minimum ggml API surface (declarations and a few definitions) that the vendored MMQ code references, so it links... |
| `kernel.ds4.mmq_mmvq` | llama.cpp's mul_mat_vec_q and its MoE form |
| `kernel.ds4.mmq_repack` | the aligned structure-of-arrays repack library shared by the weight server and the engine's self-load: catalog walk,... |
| `kernel.ds4.rocm_common` | shared device helpers and the first kernels of the TU: token embedding into the HC streams (F16 and Q8_0), F16 / F32... |
| `kernel.ds4.rocm_compat` | the ROCm host shims compiled as their own object (ds4_rocm_compat.cu): device allocation through the... |
| `kernel.ds4.rocm_current_api_compat` | adapters from the current ds4_gpu.h API to the ROCm kernels' older entry points, kept in one header; a fragment of... |
| `kernel.ds4.rocm_q8` | Q8_0 matmuls: dequant to F16 / F32, warp-8 and shared-x row kernels, pre-quantized activations, pairs, batch with... |
| `kernel.ds4.rocm_unavailable` | C-linkage stubs for the CUDA-only fused and multi-device hooks that the shared graph code references but ROCm... |
| `kernel.enginelab.core_common` | the remaining shared-core headers: backend selection (backend.hpp, backend_hip.hpp), byte budget and range, GLU... |
| `kernel.enginelab.core_gemv_rows` | the decode GEMV rows family: the GEMV policy and its direct, look-ahead and staged row main loops, row layouts and... |
| `kernel.enginelab.ds41_rows` | DS41 dense rows: F16 rows, fast F8 rows, native FP8, the integer operand, formats, packed weights, the IQ2_XXS grid... |
| `kernel.enginelab.lab_basic_kernels` | the basic HIP kernels lab (vector ops, reductions) with its CMake and README (lab/basic_kernels) |
| `kernel.enginelab.q8_rows` | Q8 rows: the Q8 rows GEMV, IU4 rows, lane VOPD and VOPD bodies, sites, the Q8 projection manifest and organization,... |
| `kernel.enginelab.rows_select` | row-kernel selection: rows fit and launch registry, R1 row tiles, wave fit, wave32 DPP and the Qwen DPP reduce, FMA... |
| `kernel.kyojin.dsv41_ops` | DeepSeek-V4.1-Flash small-op kernels for gfx1151 in one code object (1,123 lines): each replaces a chain of torch... |
| `kernel.kyojin.exl3_dec` | Batch-1 EXL3 decode kernels for RDNA3.5 (gfx1151, 3,431 lines): one lane owns one 16x16 trellis tile, the bit... |
| `kernel.kyojin.exl3_gemv` | Small-m EXL3 GEMV, QTIP-style on the unmodified EXL3 format: warps split k and never synchronise in the main loop, B... |
| `kernel.kyojin.exl3_gemv_int8` | Fused int8-activation GEMV for mul1 (cb 2) tensors: one cooperative launch covering the input Hadamard, activation... |
| `kernel.kyojin.exl3_gemv_rdna` | RDNA tiles of the EXL3 GEMV (rocm/quant: the wave32 kernel, its tile loaders and launch front end) |
| `kernel.kyojin.ext_base` | Unchanged ExLlamaV3 base files (turboderp-org/exllamav3 8c593e4e) that the Kyojin C / HIP files include: 49 headers... |
| `kernel.kyojin.ext_portability` | The portability layer the Kyojin kernels share: port.cuh (one-source HIP gfx11.5 / CUDA sm_86 macros),... |
| `kernel.kyojin.fq_gemv` | Cooperative decode GEMV for foreign-quant weights (M = 1..16): threads share a row's weight stream instead of each... |
| `kernel.kyojin.fq_gemv_fp8` | FP8 E4M3 decode GEMV in C++ (wave32 on RDNA) replacing a Triton GEMV that streamed only 42-55 GB/s on gfx1151 |
| `kernel.kyojin.fq_gemv_kq` | q8_1-activation decode kernels for K-quant and IQ4_XS weights: port of the llama.cpp mmvq vmmq arms with Kyojin's v2... |
| `kernel.kyojin.fq_gemv_q8` | q8_1-activation decode kernels (llama.cpp mmvq design): activations quantized once per row into a cached scratch,... |
| `kernel.kyojin.fq_mmvq` | IQ-format decode GEMV, llama.cpp mmvq style (CUDA and HIP, wave32): q8_1 activations, the ggml vec_dot_iq*_q8_1 dots... |
| `kernel.kyojin.gm_dense_q8` | Dense Q8_0 x q8_1 decode GEMV, few rows, column-tiled (HY4_DENSE_V2): one block walks consecutive columns with the... |
| `kernel.maya_tp.bf16_gemv` | BF16 weight GEMVs and conversions: the naive and warp-split BF16 GEMV, llama.cpp's FP32-activation MMVF (single,... |
| `kernel.maya_tp.common` | Maya-TP's HIP compatibility layer and shared device headers: include/strata/hip_compat (cuda_runtime.h,... |
| `kernel.maya_tp.gguf_gemv` | decode GEMVs on native GGUF blocks adapted from llama.cpp's MMVQ at 3cf03257: Q2_0-Q8_0, the K-quants and IQ4... |
| `kernel.maya_tp.s_gemv` | GEMVs over the S-family weight planes (S2 = Q2_0 codes packed four a byte with one FP32 scale per 64 values; S4 / S8... |
| `kernel.vllm_strix.w8a16_gemv` | skinny decode GEMVs for gfx1151 in one HIP unit (one wave an output row, VEC x 16 contiguous bytes a lane): W8A16... |
| `runtime.vllm_strix.fp8_gemv_hook` | HIP FP8 block-scaled GEMV for DeepSeek-V4's decode-sized linears: TritonFp8BlockScaledMMKernel (fp8 activation quant... |
| `runtime.vllm_strix.w8a16_requant` | load-time int8 (W8A16, symmetric, group 128) for BF16 linears: after loading, UnquantizedLinearMethod quantises... |
| `tool.ds4.gguf_dsv41` | DeepSeek V4.1 scripts: quantizer, metadata, validation, DSpark support GGUF and shard merge |
| `tool.enginelab.lab_cross_compile` | the cross-compile probe: a HIP probe and its prebuilt gfx1151 code object and object (lab/cross_compile/out)... |
| `tool.vllm_strix.fp8_sweep` | tile sweep for the FP8 block-scaled GEMV: fp8sweep.hip holds the kernel with ROWS output rows a wave (activation... |
| `tool.vllm_strix.w8_sweep` | tile sweep for the W8A16 GEMV (ROWS rows a wave, VEC x 16 contiguous bytes a lane a step) behind sweep(variant,... |

### F05 MoE decode, routers and the fused MoE half-layer (61)

| module | what it holds |
|---|---|
| `kernel.ds4.glmtp_rocm_moe` | [glmtp overlay] rocm_moe family: rocm/ds4_rocm_moe.cuh changed (+290 -26 lines). Kernels not in the base:... |
| `kernel.ds4.glmtp_rocm_moe_launch` | [glmtp overlay] rocm_moe_launch family: rocm/ds4_rocm_moe_launch.cuh changed (+309 -50 lines). The family: host... |
| `kernel.ds4.k53flash_rocm_moe` | [k53flash overlay] rocm_moe family: rocm/ds4_rocm_moe.cuh changed (+141 -0 lines). The family: the routed-MoE... |
| `kernel.ds4.k53flash_rocm_moe_launch` | [k53flash overlay] rocm_moe_launch family: rocm/ds4_rocm_moe_launch.cuh changed (+12 -1 lines). The family: host... |
| `kernel.ds4.k53flash_rocm_router` | [k53flash overlay] rocm_router family: rocm/ds4_rocm_router.cuh changed (+4 -15 lines). The family: the warp top-k... |
| `kernel.ds4.k53perf_rocm_moe` | [k53perf overlay] rocm_moe family: rocm/ds4_rocm_moe.cuh changed (+525 -25 lines). Kernels not in the base:... |
| `kernel.ds4.k53perf_rocm_moe_launch` | [k53perf overlay] rocm_moe_launch family: rocm/ds4_rocm_moe_launch.cuh changed (+369 -64 lines). The family: host... |
| `kernel.ds4.k53perf_rocm_router` | [k53perf overlay] rocm_router family: rocm/ds4_rocm_router.cuh changed (+22 -10 lines). The family: the warp top-k... |
| `kernel.ds4.k53prefill_rocm_moe` | [k53prefill overlay] rocm_moe family: rocm/ds4_rocm_moe.cuh changed (+141 -0 lines). The family: the routed-MoE... |
| `kernel.ds4.k53prefill_rocm_moe_launch` | [k53prefill overlay] rocm_moe_launch family: rocm/ds4_rocm_moe_launch.cuh changed (+12 -1 lines). The family: host... |
| `kernel.ds4.k53prefill_rocm_router` | [k53prefill overlay] rocm_router family: rocm/ds4_rocm_router.cuh changed (+4 -15 lines). The family: the warp top-k... |
| `kernel.ds4.kmultinode_rocm_moe` | [kmultinode overlay] rocm_moe family: rocm/ds4_rocm_moe.cuh changed (+194 -1640 lines). The family: the routed-MoE... |
| `kernel.ds4.kmultinode_rocm_moe_launch` | [kmultinode overlay] rocm_moe_launch family: rocm/ds4_rocm_moe_launch.cuh changed (+206 -1666 lines). The family:... |
| `kernel.ds4.kmultinode_rocm_router` | [kmultinode overlay] rocm_router family: rocm/ds4_rocm_router.cuh changed (+60 -130 lines). The family: the warp... |
| `kernel.ds4.kmultinode_rocm_shared_expert` | [kmultinode overlay] rocm_shared_expert family: rocm/ds4_rocm_shared_expert.cuh changed (+0 -145 lines). The family:... |
| `kernel.ds4.rocm_moe` | the routed-MoE kernels: active-expert and tile building, gate / up / mid decode (LUT, MXFP4, Q4_K, IQ2) and down... |
| `kernel.ds4.rocm_moe_launch` | host launch glue and policy for the routed MoE: workspace sizing, tile plans, format dispatch; a fragment of the... |
| `kernel.ds4.rocm_router` | the warp top-k router select; a fragment of the single ROCm translation unit ds4_rocm.cu, where its kernels and... |
| `kernel.ds4.rocm_shared_expert` | launch glue for the shared expert; a fragment of the single ROCm translation unit ds4_rocm.cu, where its kernels and... |
| `kernel.enginelab.ds41_routes` | DS41 MoE routing: the fast routes (HIP, with its body) and the MoE layer: 5 origin files under include/strix,... |
| `kernel.enginelab.lab_moe_routing` | the MoE routing lab: flat and hierarchical routing kernels with qualification (lab/moe_routing) |
| `kernel.enginelab.moe_core_route` | MoE routing and decode: route, route plan R1, route prune, route project and wave kernels, the HIP route tap, R1... |
| `kernel.enginelab.moe_launch` | MoE decode launchers: experts launcher, route launcher, route BF16 and prune, direct-B, the overlap dispatcher, slot... |
| `kernel.enginelab.shared_expert` | the shared expert: launcher (HIP), prefetch and rows diagnostic: 6 origin files under include/strix,... |
| `kernel.kyojin.dsv41_fuse` | DS4.1 decode glue fusion: swiglu, MoE tail, router scores and router top-k with K-group slots, each one launch with... |
| `kernel.kyojin.dsv41_moe` | DS4.1 batch-1 routed experts of a layer in 5 launches for all K groups at once (top-6), bit-identical to the... |
| `kernel.kyojin.exl3_moe` | EXL3 MoE expert kernels: input Hadamard, gate / up GEMMs, activation, down GEMM and output Hadamard per (K,... |
| `kernel.kyojin.exl3_moe_coop` | Fused decode-shaped MoE (bsz 1..MAX_BSZN): two ordinary launches per layer run every (token, expert) slot through... |
| `kernel.kyojin.exl3_moe_fused` | Fused MoE half-layer decode for gfx11.5 (opt-in EXL3_MOE_FUSED): one persistent launch per half-layer for 1..4 rows... |
| `kernel.kyojin.exl3_moe_fused7` | core7 of the fused MoE half-layer: front end rewritten for memory-level parallelism (every global load of a work... |
| `kernel.kyojin.exl3_moe_fused9` | core9: the mf7 kernel plus cache touches (read-only prefetch loads whose results are dropped) for finalize, router... |
| `kernel.kyojin.exl3_moe_valu` | VALU grouped-MoE decode matvec for gfx11.5 (default on for K1 / K2 / K3 experts): replaces the WMMA / LDS body of... |
| `kernel.kyojin.fq_moe` | Grouped decode MoE for foreign-quant experts (gateless relu2, IQ2_XXS): the selected experts' up and down GEMVs in... |
| `kernel.kyojin.fq_moe_kq` | Grouped decode MoE with q8_1 activations for K-quant / IQ4_XS experts (gate + up, down, fold / combine launches; v2... |
| `kernel.kyojin.fq_moe_q8` | Grouped decode MoE with q8_1 activations: same (slot, tile) grid as fq_moe.cu but integer dp4a dots over... |
| `kernel.kyojin.gm_moe` | Generic GGUF MoE decode for one token: Q4_K gate / up and Q4_K / Q6_K down experts, SwiGLU, no shared expert, same... |
| `kernel.kyojin.gm_router` | Decode-only standard MoE router for up to eight rows (gfx12 tuned): the gate held as a persistent contiguous... |
| `kernel.kyojin.hip_kq_moe_dec` | Decode MoE kernels for GGUF k-quants as a device-only gfx1151 code object (built by hip_kq/build_moe.sh, loaded by... |
| `kernel.kyojin.routing` | MoE routers: the extension's top-k routing kernels (softmax / sigmoid-bias / grouped top-k, normalisation and... |
| `kernel.maya_tp.glm_fast` | the GLM fast-path kernels (glm_fast.cu, 7,453 lines): AMD IQ2_XXS / IQ2_S / IQ3_XXS decode experts with packed sign... |
| `kernel.maya_tp.glm_ffn` | the glm5-next clamped SwiGLU (glm_ffn.cu) |
| `kernel.maya_tp.moe_experts` | the MoE block's expert kernels: llama.cpp's moe-weighted reduction (native_moe), the shared expert with its sigmoid... |
| `kernel.maya_tp.router` | the MoE routers: softmax / stable top-k / renormalise (router_top10), llama.cpp's topk-moe for 512 experts top 10... |
| `kernel.vllm_strix.moe_int4_gemv` | decode-specialised MoE expert GEMVs for gfx1151 in one HIP unit, wvSplitK style (16 contiguous packed bytes a lane,... |
| `runtime.enginelab.moe_policy` | MoE contracts (include/strix/moe): activation policy, combine, declared experts, expert union, route file, route... |
| `runtime.enginelab.moe_schedule` | MoE schedules: layer runtime and schedule, plan, token cost backend and schedule, fork-join arena, deferred MoE... |
| `runtime.kyojin.cpu_moe_mul1` | CPU MoE expert GEMM for mul1 (cb2) EXL3 tensors (2,490 lines, AVX-512 VBMI / AVX2): the mul1 codebook is affine in a... |
| `runtime.kyojin.hy4_moe_levers` | Hy4 MoE / dense decode levers: HIP WMMA grouped expert GEMM on stored k-quants, int8 draft experts, one-launch... |
| `runtime.kyojin.moe_fused_py` | Python front of the fused MoE half-layer and the router top-k module |
| `runtime.kyojin.moe_handoff` | Persistent-worker handoff for CPU-offloaded experts: pinned shared-memory flags, GPU publish / wait kernels and a... |
| `runtime.maya_tp.glm_fast_path` | the glm5-next FAST decode path (glm_fast_path.cu, 8,758 lines, with glm_fast_state.hpp, the opt-in layer graph... |
| `runtime.vllm_strix.fused_router` | one HIP kernel for GLM routing: GroupedTopKRouter on ROCm is replaced by vsh_sigmoid_bias_topk (sigmoid, correction... |
| `runtime.vllm_strix.gate_bf16` | bf16-weight, fp32-output router gate GEMV: GateLinear tier 4 (torch.mm(out_dtype=fp32) on hipBLASLt) is routed to... |
| `runtime.vllm_strix.gate_lmhead` | two decode linears off hipBLASLt: GateLinear with fp32 weights routed to the w32 GEMV for <= 8 rows (tier \"vsh\"),... |
| `runtime.vllm_strix.moe_direct` | decode MoE straight to HIP: a hook in CompressedTensorsWNA16MoEMethod.apply that, for <= 64 token-expert pairs,... |
| `runtime.vllm_strix.moe_int4_dispatch` | ctypes wrapper for libvsh_moe_int4.so (vsh_moe_int4.py: gemv, direct_moe, topk_sum, swiglu, MXFP4 entry points with... |
| `runtime.vllm_strix.moe_tile_configs` | tuned fused-MoE tile tables for the Radeon 8060S at E=288, N=1024 (int4_w4a16 and the default dtype) under the file... |
| `runtime.vllm_strix.mxfp4_direct` | direct HIP MXFP4 MoE for DeepSeek-V4 decode: a hook that, for <= 64 token-expert pairs on gfx1151, replaces... |
| `runtime.vllm_strix.router_det_topk` | deterministic MoE-router top-k for vLLM's grouped_topk_router: _deterministic_topk_indices selects experts by a... |
| `runtime.vllm_strix.router_packed_topk` | the same deterministic expert order without the slow sort: the router packs (biased score, expert index) into one... |
| `tool.vllm_strix.mxfp4_sweeps` | MXFP4 direct-MoE tuning kit: a routing dump of real DS4 decode (moe_dump_ids.py), replay of that routing with... |

### F06 Sparse attention, indexers, top-k selection and compressed KV (DSA / QSA / MLA) (93)

| module | what it holds |
|---|---|
| `kernel.ds4.glmtp_rocm_attention` | [glmtp overlay] rocm_attention family: rocm/ds4_rocm_attention.cuh changed (+12 -5 lines). The family: DeepSeek V4... |
| `kernel.ds4.glmtp_rocm_attention_launch` | [glmtp overlay] rocm_attention_launch family: rocm/ds4_rocm_attention_launch.cuh changed (+19 -3 lines). The family:... |
| `kernel.ds4.glmtp_rocm_glm` | [glmtp overlay] rocm_glm family: rocm/ds4_rocm_glm.cuh changed (+1094 -87 lines). Kernels not in the base:... |
| `kernel.ds4.glmtp_rocm_indexer` | [glmtp overlay] rocm_indexer family: rocm/ds4_rocm_indexer.cuh changed (+236 -7 lines). Kernels not in the base:... |
| `kernel.ds4.k53flash_rocm_attention` | [k53flash overlay] rocm_attention family: rocm/ds4_rocm_attention.cuh changed (+86 -0 lines). Kernels not in the... |
| `kernel.ds4.k53flash_rocm_attention_launch` | [k53flash overlay] rocm_attention_launch family: rocm/ds4_rocm_attention_launch.cuh changed (+1 -12 lines). The... |
| `kernel.ds4.k53flash_rocm_glm` | [k53flash overlay] rocm_glm family: rocm/ds4_rocm_glm.cuh changed (+374 -86 lines). Kernels not in the base:... |
| `kernel.ds4.k53flash_rocm_glm_selected_multihead` | [k53flash overlay] rocm_glm_selected_multihead family: rocm/ds4_rocm_glm_selected_multihead.cu added (234 lines).... |
| `kernel.ds4.k53flash_rocm_indexer` | [k53flash overlay] rocm_indexer family: rocm/ds4_rocm_indexer.cuh changed (+1 -3 lines). The family: the DSV4... |
| `kernel.ds4.k53perf_rocm_attention` | [k53perf overlay] rocm_attention family: rocm/ds4_rocm_attention.cuh changed (+88 -0 lines). Kernels not in the... |
| `kernel.ds4.k53perf_rocm_attention_gufo` | [k53perf overlay] rocm_attention_gufo family: rocm/ds4_rocm_attention_gufo.cuh added (360 lines). Kernels not in the... |
| `kernel.ds4.k53perf_rocm_attention_launch` | [k53perf overlay] rocm_attention_launch family: rocm/ds4_rocm_attention_launch.cuh changed (+244 -2 lines). The... |
| `kernel.ds4.k53perf_rocm_glm` | [k53perf overlay] rocm_glm family: rocm/ds4_rocm_glm.cuh changed (+244 -7 lines). Kernels not in the base:... |
| `kernel.ds4.k53perf_rocm_halo_attention` | [k53perf overlay] rocm_halo_attention family: rocm/ds4_rocm_halo_attention.cuh added (34 lines);... |
| `kernel.ds4.k53perf_rocm_halo_direct_qk` | [k53perf overlay] rocm_halo_direct_qk family: rocm/ds4_rocm_halo_direct_qk.cuh added (391 lines). Kernels not in the... |
| `kernel.ds4.k53perf_rocm_halo_indexer` | [k53perf overlay] rocm_halo_indexer family: rocm/ds4_rocm_halo_indexer.cuh added (106 lines). Kernels not in the... |
| `kernel.ds4.k53perf_rocm_halo_static_query` | [k53perf overlay] rocm_halo_static_query family: rocm/ds4_rocm_halo_static_query.cuh added (147 lines). Kernels not... |
| `kernel.ds4.k53perf_rocm_indexer` | [k53perf overlay] rocm_indexer family: rocm/ds4_rocm_indexer.cuh changed (+17 -3 lines). The family: the DSV4... |
| `kernel.ds4.k53prefill_rocm_attention` | [k53prefill overlay] rocm_attention family: rocm/ds4_rocm_attention.cuh changed (+86 -0 lines). Kernels not in the... |
| `kernel.ds4.k53prefill_rocm_attention_launch` | [k53prefill overlay] rocm_attention_launch family: rocm/ds4_rocm_attention_launch.cuh changed (+1 -12 lines). The... |
| `kernel.ds4.k53prefill_rocm_glm` | [k53prefill overlay] rocm_glm family: rocm/ds4_rocm_glm.cuh changed (+414 -132 lines). Kernels not in the base:... |
| `kernel.ds4.k53prefill_rocm_glm_selected_multihead` | [k53prefill overlay] rocm_glm_selected_multihead family: rocm/ds4_rocm_glm_selected_multihead.cu added (234 lines).... |
| `kernel.ds4.k53prefill_rocm_indexer` | [k53prefill overlay] rocm_indexer family: rocm/ds4_rocm_indexer.cuh changed (+1 -3 lines). The family: the DSV4... |
| `kernel.ds4.kmultinode_rocm_attention` | [kmultinode overlay] rocm_attention family: rocm/ds4_rocm_attention.cuh changed (+93 -563 lines). Kernels not in the... |
| `kernel.ds4.kmultinode_rocm_attention_launch` | [kmultinode overlay] rocm_attention_launch family: rocm/ds4_rocm_attention_launch.cuh changed (+45 -431 lines). The... |
| `kernel.ds4.kmultinode_rocm_compressor` | [kmultinode overlay] rocm_compressor family: rocm/ds4_rocm_compressor.cuh changed (+5 -12 lines). The family: the... |
| `kernel.ds4.kmultinode_rocm_indexer` | [kmultinode overlay] rocm_indexer family: rocm/ds4_rocm_indexer.cuh changed (+8 -269 lines). The family: the DSV4... |
| `kernel.ds4.rocm_attention` | DeepSeek V4 attention kernels for raw and mixed (FP8-compressed) KV: prefill raw / mixed with softmax tiles, decode... |
| `kernel.ds4.rocm_attention_launch` | host launch glue for the attention kernels: shape and KV-format dispatch, split choices and the eligibility rules; a... |
| `kernel.ds4.rocm_compressor` | the DSV4 KV compressor: ratio-4 shift, pooled prefill and update, row set and store; a fragment of the single ROCm... |
| `kernel.ds4.rocm_glm` | the GLM-5.x ROCm kernels, mirroring the Metal GLM graph entry points: KDA decode and the three-stage KDA prefill... |
| `kernel.ds4.rocm_indexer` | the DSV4 lightning indexer: Hadamard FP4 keys, scores (direct, batched, WMMA-128), top-k (pow2, chunked, tree merge,... |
| `kernel.enginelab.attn_contracts` | attention contracts and host references: the MLA contract and reference, the group indexer and its reference, group... |
| `kernel.enginelab.attn_core_select` | top-k selection for sparse attention: select, radix-3 select and threshold select with their rules and spec, the... |
| `kernel.enginelab.attn_core_sparse_mla` | selected-token (sparse) attention: the sparse flash MLA, row-group and WMMA forms with key split, the exact selected... |
| `kernel.enginelab.ds41_attention` | DS41 attention: the HIP attention with F16 WMMA, flash and QK pieces, the attention host and state |
| `kernel.enginelab.glm_attn` | GLM attention: fast attention, tier B and the DSA policy: 5 origin files under src/glm |
| `kernel.enginelab.indexed_attention` | the indexed attention launcher (HIP; it includes the QSA kernel bodies), state and workspace |
| `kernel.enginelab.lab_sparse_lookup` | the sparse lookup lab (lab/sparse_lookup): 7 origin files under lab/sparse_lookup |
| `kernel.enginelab.minimax_attn` | MiniMax attention: index projection, MSA fused, flash policy, in-block layout, norm-rope and select dump; the MSA... |
| `kernel.enginelab.qsa_prefill` | QSA (Qwen3.8-Flash-Next sparse attention) prefill kernels: the fast prefill profile (union of TR rows' selections in... |
| `kernel.kyojin.ds41_pf` | DS4.1 prefill kernels: sparse attention (MQA, 64 heads x 512, one shared latent kv, per-row key lists, per-head... |
| `kernel.kyojin.dsa_topk` | DSA / QSA top-k selection: a descending radix bucket search per row (one block per row) giving ascending indices... |
| `kernel.kyojin.dsa_topk_gfx12` | QSA selection rewritten for RDNA (the ROCm path of ext.dsa_topk): fp16 (R, T) rows read from memory once, k <= 512... |
| `kernel.kyojin.dsa_topk_par` | QSA selection with many blocks per row, bitwise identical to dsa_topk_gfx12: the row sweeps run on the whole GPU,... |
| `kernel.kyojin.dsv41_idx` | DS4.1 indexer scoring (WMMA f16 to f32): score[r, t] = sum_h w[r, h] * relu(q[r, h] . k[t]) over 32 heads x 128, a... |
| `kernel.kyojin.dsv4_compress` | DeepSeek-V4 KV compressor kernels (the compressed-attention pool update) as Kyojin changed them |
| `kernel.kyojin.dsv4_pool_quant` | DeepSeek-V4 compressed pool quantization (pool rows to fp8 with scales) |
| `kernel.kyojin.mla_dec_valu` | MLA decode absorb / unfold for up to 8 rows on VALU (ROCm only), bit-identical to the Triton WMMA kernels of... |
| `kernel.kyojin.qsa_prefill` | Qwen QSA sparse prefill attention for gfx11.5 WMMA f16, one wave per (row, kv head): replaces the Triton split +... |
| `kernel.kyojin.qsa_prefill_wmma` | QSA sparse prefill, 4-row union variant (8 waves a block): 16-key tiles gathered once from the union of the rows'... |
| `kernel.kyojin.sp_mla` | Sparse MLA prefill kernel for gfx1151 (WMMA f16, wave32): one workgroup of 16 waves = one query x 64 heads over... |
| `kernel.kyojin.triton_dsa` | Production DSA (DeepSeek Sparse Attention) Triton kernels (1,877 lines): constexpr toggles for a sliding-window... |
| `kernel.kyojin.triton_mla` | Triton kernels for multi-head latent attention (compressed latent and shared RoPE key in the cache; absorb / unfold... |
| `kernel.kyojin.triton_qsa` | Triton kernels of the QSA indexer stages of the graph-captured decode path (raw-key plane append, scoring shared... |
| `kernel.maya_tp.ds4_kernels` | DeepSeek-V4.1's own device kernels for a window of T tokens (ds4_kernels.cu, 5,111 lines): attention with its... |
| `kernel.maya_tp.glm_dsa` | the glm5-next DSA lightning indexer and absorbed-MLA attention (glm_dsa.cu) |
| `kernel.maya_tp.glm_kv_stream` | the GLM latent KV stream (glm_kv_stream.cu) and the INT8 latent records (glm_lat8.cuh, MAYA_GLM_KV_INT8) |
| `kernel.maya_tp.glmd_batch` | the glm-dsa batched prompt kernels (glmd_batch.cu, B9.2-B9.5) |
| `kernel.maya_tp.glmd_decode` | the glm-dsa decode attention kernels: mla2's decode attention (glmd_dec), the B4.3-B4.5 fast kernels (glmd_fast),... |
| `kernel.maya_tp.glmd_mixer` | the glm-dsa decode mixer's own kernels (glmd_mixer.cu, B4.6 / B14.3 / B14.4) |
| `kernel.maya_tp.glmd_ref` | the glm-dsa F32 reference kernels (glmd_ref.cu, B3.4) and the device halves of the wave-9 guards: the FP16... |
| `kernel.maya_tp.glmd_volta` | the glm-dsa Volta / native kernels: lightning-indexer scorers with WMMA or CUDA cores and F32 / FP16 keys... |
| `kernel.maya_tp.qsa` | the QSA sparse-attention family: the cache, indexer and attention (qsa), the decode and prompt attention kernels,... |
| `kernel.vllm_strix.idx_logits` | Triton prefill indexer logits for DeepSeek-V4 on gfx1151 on fp16 WMMA: logits[q, k] = kv_scale[k] * sum_h w[q, h] *... |
| `kernel.vllm_strix.kpool_paged_logits` | Triton decode indexer-logits reader for the GLM-5.3 kpool cache on gfx1151: each indexer page is [page_size x 128... |
| `kernel.vllm_strix.sparse_attn_split` | Triton split-KV (flash-decoding) variant of vLLM's _sparse_attn_prefill_ragged_kernel for GLM-5.3 sparse MLA decode:... |
| `runtime.enginelab.qsa_plan` | QSA plans and diagnostics: selection plan, input prefetch, cache-only mode and the selection / GQA-reuse diagnostics |
| `runtime.kyojin.bc_blocksparse_mlp` | Graph-captured block-sparse MLP (MoE) block of the extension |
| `runtime.kyojin.bc_mla_attention` | Graph-captured MLA attention block (998 lines) |
| `runtime.kyojin.hy4_indexer` | k-pool sparse attention pieces: device-resident pooled-key indexer with no host sync, one-kernel pool update,... |
| `runtime.kyojin.kpool_indexer` | k-pool DSA indexer module (Hy4 row-list form) |
| `runtime.kyojin.kvcache_dsa` | DSA cache layer (compressed pools, indexer keys) |
| `runtime.kyojin.kvcache_mla` | MLA latent cache layer: one compressed latent (ckv, kv_lora_rank) and one shared RoPE key (kpe) per token in pages,... |
| `runtime.kyojin.kvcache_qsa` | QSA cache layer (raw-key planes for the indexer) |
| `runtime.maya_tp.glm_dsa_decode` | the glm-dsa (GLM-5.3 DSA) decode side: the fast binder and arena (glm_dsa_fast, glm_dsa_fast_plan), the per-part... |
| `runtime.maya_tp.glm_dsa_prompt` | the glm-dsa prompt side: the prompt layer (glm_prefill_dsa), the batched prompt plan (glm_dsa_prompt_plan), the... |
| `runtime.maya_tp.glm_dsa_reference` | the glm-dsa F32 reference forward (glm_dsa_ref.cu with its host pieces and no-GPU stub) and the CPU twins of the... |
| `runtime.vllm_strix.ds4_decode_inv_rope` | fixes DeepSeek-V4 decode on gfx1151: rocm_sparse_attn_decode reported every row as inverse-RoPE'd when given... |
| `runtime.vllm_strix.ds4_decode_rows` | explicit cap for the gfx1151 paged-MQA decode logits workspace: (heads, decode_rows, max_model_len) fp32 is 128 MiB... |
| `runtime.vllm_strix.ds4_idx_hadamard` | Hadamard rotation for the DS4 indexer's fp4 quantisation: K in fused_compress_quant_cache.py and Q in... |
| `runtime.vllm_strix.ds4_idx_logits_hook` | install hook that routes DS4 prefill indexer logits from aiter's Triton _fp8_mqa_logits_kernel to vsh_idx_logits |
| `runtime.vllm_strix.fp8_fnuz_mqa` | fp8 fnuz handling for the paged MQA logits on gfx1151: the vLLM side picks the fnuz-aware aiter fp8_mqa_logits, and... |
| `runtime.vllm_strix.idx_bt_gather_v4` | gather-site block-table expansion for the GLM indexer (a replacement ops file): the page-level block table is... |
| `runtime.vllm_strix.idx_force_tail` | force-tail for the GLM sparse indexer: the 128 most recent pools are boosted before top-k, so they are always... |
| `runtime.vllm_strix.idx_fp8max_fnuz` | indexer K quantised to an fp8 max of 224 instead of 448 so the e4m3fn to fnuz cast on the paged-MQA logits path... |
| `runtime.vllm_strix.idx_pool_subchunks` | GLM indexer prefill split into pool-aligned sub-chunks: vLLM's mla/indexer.py (replaced whole) cuts prefill indexer... |
| `runtime.vllm_strix.kpool_paged_logits_hook` | install hook that points GLM-5.3's decode indexer at the paged kpool logits reader, behind VSH_KPOOL_PAGED_LOGITS |
| `runtime.vllm_strix.pr55222_workspace` | ports of upstream vLLM PR #55222 (workspace units) as whole files for glm5next/common/attention.py and the MLA... |
| `runtime.vllm_strix.pr59412_indexer_blocks` | port of upstream vLLM PR #59412 (fixes #58858): on ROCm the DSA indexer backend advertised kernel block sizes [1,... |
| `runtime.vllm_strix.ragged_compact_ops` | upstream's ragged-index rewrite of the sparse-attention window ops, carried as a whole replacement file ahead of the... |
| `runtime.vllm_strix.sparse_attn_split_hook` | install hook that routes GLM sparse MLA decode with small query counts (MTP verify rows, draft steps) from the... |
| `tool.vllm_strix.context_probes` | long-context retrieval probes: three watchwords at 25 / 50 / 75 % of a filler prompt (GLM and DS4), the pool ids... |

### F07 Dense / flash attention and KV cache formats (29)

| module | what it holds |
|---|---|
| `kernel.ds4.k53perf_rocm_qwen4_attention_wmma` | [k53perf overlay] rocm_qwen4_attention_wmma family: rocm/ds4_rocm_qwen4_attention_wmma.cuh added (605 lines).... |
| `kernel.ds4.rocm_fp8_kv` | FP8 KV quantization and the raw-KV batch store; a fragment of the single ROCm translation unit ds4_rocm.cu, where... |
| `kernel.ds4.rocm_fp8_kv_launch` | launch glue for the FP8 KV kernels; a fragment of the single ROCm translation unit ds4_rocm.cu, where its kernels... |
| `kernel.enginelab.attn_core_flash` | the shared dense flash attention: dense decode and its flash form with key split and budget, dense and banded flash... |
| `kernel.enginelab.attn_core_prepared` | prepared attention: prepared GQA and BF16 attention with lane plans and prepared step, chunk decode and chunk exact... |
| `kernel.enginelab.h3_attn` | H3 attention (src/h3/attn): 4 origin files under src/h3 |
| `kernel.enginelab.inkling2_attention` | Inkling 2 Mini attention (HIP, operands, prepare): 5 origin files under src/inkling2 |
| `kernel.kyojin.attention_ext` | The extension's attention helpers (attention.cu) as Kyojin changed them: the small attention kernels and cache... |
| `kernel.kyojin.kvcache_quant` | Quantized KV cache writer / reader of the extension as Kyojin changed it: Hadamard-rotated (H4 in registers, H8... |
| `kernel.kyojin.page_copy` | Paged KV cache page copy and rotation kernels (16-byte aligned per-CTA chunks; small DSA pools of 1792 B pages) |
| `kernel.kyojin.qdec_hip` | Quantized-KV flash-decoding split kernel for gfx11.5 (wave32 WMMA f16): QK^T and PV straight from the packed... |
| `kernel.kyojin.triton_kvwin` | KVWIN kernels: the fp16 ring writer and the fp16 ring partial of the flash-decoding split (exact sink and recent... |
| `kernel.kyojin.triton_paged` | Triton paged attention of the engine as Kyojin changed it (2,694 lines): flash-decoding split / combine over the... |
| `kernel.kyojin.triton_qdeq` | Quantized paged cache to fp16 staging: the 8-bit H32-rotated cache for the HIP sparse prefill kernel, and any... |
| `kernel.kyojin.triton_qwrite` | Native Triton 8-bit paged cache writer replacing the torch fallback of ext.quant_cache_paged on ROCm (q_cache.cu is... |
| `kernel.kyojin.triton_tq` | TurboQuant cache kernels: paged window dequant for the prefill staging scratch (32-value groups, power-of-two bit... |
| `kernel.maya_tp.flash_attn` | llama.cpp's flash-attention kernel specialised for the engine's shapes (native_flash_attn, from 3cf03257) |
| `kernel.maya_tp.kvcache` | KV cache formats and streaming: Q4_0 KV with a Walsh-Hadamard rotation (kv_q4), INT8 KV (kv_q8) and the... |
| `runtime.enginelab.session_kv` | session KV and state (include/strix/session, src/session): KV, KV capacity, pages, precision and row append, prefix... |
| `runtime.kyojin.attn_bc` | Python side of the graph-captured attention blocks (GQA, DSA, MLA): buffers, capture and patch-and-replay of each... |
| `runtime.kyojin.attn_dispatch` | Attention backend dispatch (hint caches, per-shape handler lists), shared helpers, the torch reference path and the... |
| `runtime.kyojin.bc_attention` | Graph-captured decode attention block (first run eager, second captured, later patch-and-replay): q / k / v... |
| `runtime.kyojin.ck_pagedkv` | Optional Composable Kernel paged prefill backend (MiMo qk192, gfx1151, batch one) and its sparse form: page... |
| `runtime.kyojin.cpu_cache` | Host-memory tier of the paged cache (pages swapped to CPU memory) |
| `runtime.kyojin.hy4_attn_levers` | Hy4 attention decode levers, each behind a flag: fewer eager launches, grouped projections on the stored bytes,... |
| `runtime.kyojin.kvcache_core` | Cache layer base and registry as Kyojin changed them |
| `runtime.kyojin.kvcache_kvwin` | KVWIN: exact fp16 sink tokens and a recent-window ring per job on top of a complete quantized pool |
| `runtime.kyojin.kvcache_tq` | TurboQuant KV cache layer: per head vector y = R x (Hadamard x random signs), n = //y//, codes = nearest Lloyd-Max... |
| `runtime.kyojin.pagetable` | Paged KV page table: hashing, sharing, reference counts and eviction |

### F08 Hyper-connections (mHC) (15)

| module | what it holds |
|---|---|
| `kernel.ds4.k53perf_rocm_halo_hc` | [k53perf overlay] rocm_halo_hc family: rocm/ds4_rocm_halo_hc.cuh added (40 lines). Kernels not in the base:... |
| `kernel.ds4.k53perf_rocm_hc_sgemm` | [k53perf overlay] rocm_hc_sgemm family: rocm/ds4_rocm_hc_sgemm.cuh added (105 lines). The family: an HC-only F32... |
| `kernel.ds4.kmultinode_rocm_hc_output_launch` | [kmultinode overlay] rocm_hc_output_launch family: rocm/ds4_rocm_hc_output_launch.cuh changed (+32 -38 lines). The... |
| `kernel.ds4.rocm_hc` | the hyper-connection (HC) kernels: split Sinkhorn, weighted sums (fused with norm), and the 4-stream expand /... |
| `kernel.ds4.rocm_hc_output_launch` | launch glue for the HC output path; a fragment of the single ROCm translation unit ds4_rocm.cu, where its kernels... |
| `kernel.enginelab.core_hc` | the hyper-connection core: HC geometry, expand rows, lane chains, the residual mix and its plan, reference and HIP... |
| `kernel.enginelab.hc_hyper` | hyper-connection decode: the hyper-mix launcher (HIP), fast WMMA HC, wide2 VOPD bodies, repacked down layout and... |
| `kernel.enginelab.hc_prefill` | hyper-connection prefill (qpf2k): the HC prefill with inverse reuse (and layout), out16 and output3 bodies |
| `kernel.kyojin.gr_mix` | Gated-residual mix chain of the Qwen prefill for gfx11.5 WMMA f16: dm = normed @ proj^T, t = silu(dm/H), the up... |
| `kernel.kyojin.hc_mix` | Fused mHC hyper-connection mix and HyperHead collapse (2,432 lines), plus hc_mix_q8r.cuh: a one-pass int8... |
| `kernel.maya_tp.glm_hc` | GLM-5.3-Flash's manifold hyper-connection (mHC) read and write halves (glm_hc.cu) |
| `kernel.maya_tp.hyperconn` | the gated residual / hyper-connection kernels: gr_read and gr_write (gr), their fused form (fused_gr) and... |
| `runtime.kyojin.hy4_hc` | Fused hyper-connection sites and norms (Triton), generic by shape |
| `runtime.vllm_strix.ds4_mhc_cfg` | gfx1151 tile configs and a fused / unfused crossover for aiter's manifold hyper-connection (mHC) kernels, with the... |
| `runtime.vllm_strix.mhc_no_tilelang` | keeps vllm/model_executor/layers/mhc.py off the TileLang mHC path on gfx1151 (the import is gated to gfx942) |

### F09 Linear attention and recurrence (GDN, KDA) (24)

| module | what it holds |
|---|---|
| `kernel.enginelab.core_short_conv` | the short causal convolution and its ring buffer state: 2 origin files under src/core |
| `kernel.enginelab.gdn_decode` | gated delta-net decode: the GDN launcher (HIP) and decode fuse, recurrence rows, recurrent layout, LDS index and... |
| `kernel.enginelab.gdn_prefill` | gated delta-net prefill: the prefill scan and overlap, chunked and FP16 chunked recurrence, the qpf2k prefill and... |
| `kernel.enginelab.lab_recurrent_gdn` | the recurrent GDN lab kernel (lab/recurrent_gdn): 3 origin files under lab/recurrent_gdn |
| `kernel.enginelab.ops_recur` | shared recurrence ops (src/ops/recur): gated-delta policy, prologue, rows (body), step (body) and window, the... |
| `kernel.kyojin.fla_common` | Package glue of the vendored fla (exports, utils) |
| `kernel.kyojin.fla_gdn_fused_h` | GDN fused chunk kernel for gfx11.5 WMMA: recompute_w_u + chunk_gated_delta_rule_fwd_h + chunk_fwd_o of the fla... |
| `kernel.kyojin.fla_gdn_hf` | Gated delta chunk WMMA kernel for gfx1151 adapted from hipfire (gated_delta_chunk_wmma.gfx1151.hip), with its... |
| `kernel.kyojin.fla_kda_fused_h` | KDA fused chunk recurrence (output fused) for gfx11.5 / gfx12 WMMA, matching the Triton chunk_kda_fwd_kernel_h_fused... |
| `kernel.kyojin.fla_kda_hf` | KDA chunk kernel derived from hipfire's gated delta chunk WMMA kernel |
| `kernel.kyojin.fla_triton_gdn` | Vendored flash-linear-attention Triton chain for GDN as Kyojin changed and extended it (gdn_fused, gdn_kkt_shared,... |
| `kernel.kyojin.fla_triton_kda` | Vendored fla Triton chain for KDA / GLA, with Kyojin's fused intra, inter-split and fused-h forms |
| `kernel.kyojin.gdn` | Gated DeltaNet recurrent kernels of the extension, much extended by Kyojin (2,160 lines): register-resident-state... |
| `kernel.kyojin.gdn_fn_conv1d` | Causal conv1d of the GDN front end (Triton and torch paths) as Kyojin changed it |
| `kernel.kyojin.gdn_fn_gr_pf` | Fused prefill pieces of GatedResidual._mix (grouped rms_norm, gate application) in Triton, replacing ~13 fp32 torch... |
| `kernel.kyojin.gdn_fn_rule` | Gated delta rule dispatch (chunked prefill and recurrent decode forms) |
| `kernel.kyojin.kda_fused` | Fused KDA forget gate and gated RMS norm (Triton), replacing torch chains (norm.cu is not built on ROCm) |
| `kernel.kyojin.kda_inter` | KDA inter-solve off-diagonal dots (kdaC7): one program per (head, 64-row chunk), full-cache-line reads, gated... |
| `kernel.maya_tp.gdn` | the gated delta-net layer kernels: the reference step with conv, L2 norm, beta gate and output norm (gdn), the fused... |
| `kernel.maya_tp.glm_kda` | the glm5-next KDA (kimi-delta-attention style) pieces: gates, recurrence and readout over the KDA state layouts... |
| `runtime.enginelab.recurrent_state` | recurrent state: the recurrent plan, state policy and state type: 4 origin files under include/strix,... |
| `runtime.kyojin.kvcache_recurrent` | Recurrent-state cache (GDN / KDA / Mamba states, checkpoints for speculative rewind) |
| `runtime.vllm_strix.pr57979_kda_stride` | port of upstream vLLM PR #57979, the KDA (Kimi delta attention) stride fix, as whole files of the... |
| `tool.vllm_strix.sync_instr` | measures what the KDA chunk-index host sync costs: instruments prepare_chunk_indices (a .tolist() device to host... |

### F10 Norm, RoPE, elementwise glue, heads, sampling, chat templates (44)

| module | what it holds |
|---|---|
| `kernel.ds4.glmtp_rocm_norm_rope` | [glmtp overlay] rocm_norm_rope family: rocm/ds4_rocm_norm_rope.cuh changed (+57 -34 lines). The family: RMSNorm... |
| `kernel.ds4.k53perf_rocm_norm_rope` | [k53perf overlay] rocm_norm_rope family: rocm/ds4_rocm_norm_rope.cuh changed (+9 -0 lines). The family: RMSNorm... |
| `kernel.ds4.kmultinode_rocm_embedding_launch` | [kmultinode overlay] rocm_embedding_launch family: rocm/ds4_rocm_embedding_launch.cuh changed (+0 -38 lines). The... |
| `kernel.ds4.kmultinode_rocm_norm_rope` | [kmultinode overlay] rocm_norm_rope family: rocm/ds4_rocm_norm_rope.cuh changed (+0 -21 lines). The family: RMSNorm... |
| `kernel.ds4.kmultinode_rocm_output` | [kmultinode overlay] rocm_output family: rocm/ds4_rocm_output.cuh changed (+1 -11 lines). The family: pointwise... |
| `kernel.ds4.rocm_embedding_launch` | launch glue for the embedding kernels; a fragment of the single ROCm translation unit ds4_rocm.cu, where its kernels... |
| `kernel.ds4.rocm_norm_rope` | RMSNorm (plain, weighted, per-head, QKV rows) and the RoPE tail, alone and fused with the head norm; a fragment of... |
| `kernel.ds4.rocm_output` | pointwise kernels: SwiGLU, add / add3, zero, the HC output weights and the directional-steering projection; a... |
| `kernel.enginelab.core_head` | LM head and output kernels of the shared core: head argmax plans and emit, the screened head (a cheap screen then... |
| `kernel.enginelab.logits_head` | the LM head launcher (HIP), head argmax and argmax core, decode head rows, the integer head and logit rows |
| `kernel.enginelab.ops_glu` | GLU ops: GLU (device), Q8 rows GLU pair and SwiGLU rows: 4 origin files under src/ops |
| `kernel.enginelab.ops_norm` | normalisation ops: norm (device), numeric traits, presets, reference, RMS BF16 rows and row reduction |
| `kernel.enginelab.ops_rope` | RoPE ops: rope (device) and split-half device: 3 origin files under src/ops |
| `kernel.enginelab.ple` | per-layer embeddings (PLE): direct, discard safety, gap witness, the PLE launcher (HIP), lookup, projection tile,... |
| `kernel.enginelab.qwen_rotary` | Qwen rotary: the rotary table (HIP) and diagnostic, the YaRN table and the rotary descriptor |
| `kernel.enginelab.token_sampling` | token sampling: host and device samplers (HIP) and timing: 9 origin files under include/strix,... |
| `kernel.kyojin.activation` | Gated activations of the extension (silu(x) * y and the gelu / relu2 forms, in place when the output aliases an... |
| `kernel.kyojin.elementwise` | Small elementwise kernels of the extension that Kyojin changed: residual add (with dtype mixes), logit soft-capping... |
| `kernel.kyojin.fused_elt` | ROCm-only fused elementwise ops: each replaces a torch kernel chain of ext_fallbacks.py with one launch and the same... |
| `kernel.kyojin.ngram` | N-gram embedding rows: device gather of hashed n-gram rows, and the scattered disk gather of rows on Windows... |
| `kernel.kyojin.norm` | RMS norm and gated RMS norm of the extension (warp-shuffle reductions across lanes and warps), with Kyojin's ROCm... |
| `kernel.kyojin.penalties` | Repetition penalties on the device: frequency / presence / repetition penalty over a token window and DRY (llama.cpp... |
| `kernel.kyojin.ple` | PLE stream gate of Qwen3.8-Flash-Next: out = sigmoid(signed_sqrt(gate * gate_scale)) * value, the shared value row... |
| `kernel.kyojin.ple_hip` | Hand HIP kernels for the two torch fallbacks left in the Qwen prefill: PLELayer.forward_streams_reference after the... |
| `kernel.kyojin.rope` | Rotary position embedding of the extension (neox / gptj styles, partial rotary, fused q / k), with Kyojin's ROCm changes |
| `kernel.kyojin.softmax_sampling` | On-device sampling: argmax / top-k / top-p / min-p basics, a fused terminal sampling step that computes temperature... |
| `kernel.kyojin.strings` | Stop-string matcher of the generator: compares a UTF-32 tail against a packed list, reporting full, partial... |
| `kernel.maya_tp.elementwise` | the glue kernels (elementwise: residual adds, gathers, scaling, compiled -ffp-contract=off) and the control-vector... |
| `kernel.maya_tp.ple` | the per-layer embedding (PLE) block's GPU half (ple) and llama.cpp's PLE post-ops (native_ple_postops) |
| `kernel.maya_tp.rope` | rotary embedding: NEOX partial RoPE (rope) and llama.cpp's rope_multi / rope_yarn numerical contract (native_rope) |
| `kernel.maya_tp.sampler` | the sampler chain in llama.cpp's order (penalties, top-k, top-p, min-p, temperature, keyed draw) on the GPU |
| `runtime.enginelab.logprobs` | log-probabilities: logprob partials (HIP and CUDA spellings), the sampled target-logits reader and the teacher logit... |
| `runtime.enginelab.session_embed` | embed rows and splices: embed rows (file), embed splice (host, device, HIP): 7 origin files under include/strix,... |
| `runtime.kyojin.hip_glue` | Glue between modules and the JIT HIP kernels (loads and routes to the code objects) |
| `runtime.kyojin.sampler` | Sampler stacks and presets (custom samplers, collapsed stacks that map onto the fused device sampler) |
| `runtime.maya_tp.core_headers` | the engine headers of include/strata/core and include/strata/kernels that no other module owns by name... |
| `serve.enginelab.serve_chat` | chat templates: the template engine and kit, and the DeepSeek, GLM, Hy4 and Inkling 2 chat formats |
| `serve.enginelab.serve_constraint` | structured output: the JSON constraint and constraint selection: 4 origin files under include/strix, src/serve |
| `serve.enginelab.serve_tokenizer` | tokenizers: BPE tokenizer, tokenizer interface and UTF-8, the o200k pre-tokenizer and its Unicode tables, the... |
| `serve.maya_tp.templates` | chat formatting for the served models: GLM-5 / glm-dsa templates and product settings (glm5_chat_template.jinja,... |
| `serve.vllm_strix.toolcall_failopen` | tool-call fail-open for the glm47 parser: when a tool call is emitted inside a reasoning span that never closes, the... |
| `tool.vllm_strix.gpu_sclk` | caps or restores the iGPU maximum SCLK through pp_od_clk_voltage (non-persistent), set from glm53_gpu_sclk_max |
| `tool.vllm_strix.toolcall_drop_log` | pure instrumentation for the tool-call parser: writes the full accumulated generation to VSH_TOOLCALL_DROP_DIR when... |
| `tool.vllm_strix.toolcall_probes` | tool-calling probes: verbatim replay of a captured opencode request, the rate of early stops without a tool call... |

### F11 Speculative decoding: DSpark, MTP, DFlash, wide verify windows, draft control (41)

| module | what it holds |
|---|---|
| `executor.enginelab.ds41_dspark` | DSpark speculative drafting for DeepSeek V4.1: the DSpark driver, context, files and vocabulary, the DSpark target... |
| `executor.enginelab.minimax_dspark` | MiniMax DSpark: HIP DSpark, Markov candidates, auto mode, DSpark files and host, n-gram drafter, reference and tests |
| `kernel.enginelab.dspark_markov` | DSpark Markov candidate kernels shared by the DSpark engines: 1 origin files under src/dspark_common |
| `kernel.enginelab.mtp_draft_head` | the MTP draft head on the device: draft head, fast draft, projection, shared expert, attention (and its contract),... |
| `kernel.kyojin.dflash2` | DFlash2 drafter kernels ported from exllamav3 1.6.0: the grouped dynamic causal convolution over a draft block... |
| `kernel.maya_tp.verify` | the speculative verify window kernels: padded verification outputs, acceptance compare and the window's argmax /... |
| `runtime.ds4.dspark` | the internal DeepSeek V4.1 drafter graph (DSpark): a 128-token draft window, five draft rows and verify rows, Markov... |
| `runtime.ds4.glmtp_dspark` | [glmtp overlay] dspark family: ds4_deepseek41_dspark.h changed (+1 -1 lines). The family: the internal DeepSeek V4.1... |
| `runtime.enginelab.gemma4_dflash` | the DFlash drafter of Nemotron 3.5 Bolt E4B (host reference in the declared arithmetic g4-dflash-v1.1: five pre-norm... |
| `runtime.enginelab.gemma4_eagle` | Gemma 4 EAGLE / MTP drafter (with the drafter export script) (src/gemma4/eagle): 16 origin files under src/gemma4 |
| `runtime.enginelab.glm_spec` | GLM speculation: device, host, merge and run: 8 origin files under src/glm |
| `runtime.enginelab.inkling2_dspark` | Inkling 2 Mini DSpark (checkpoint, context, reference, taps, resident): 6 origin files under src/inkling2 |
| `runtime.enginelab.mtp_drafters` | MTP drafting control: draft activation, binding, contract, cycle input, profiles, qualification, readiness, split,... |
| `runtime.enginelab.mtp_forward` | the MTP forward: contract, golden file, payload, replay, runtime and semantics; the qualified forward runtime,... |
| `runtime.enginelab.mtp_runtime` | the MTP runtime: runtime and greedy runtime (HIP), resources, asset loader, sidecar binding, cost-table session and... |
| `runtime.enginelab.mtp_window` | MTP verify windows: window controller and verifier (HIP), target checkpoint (plan), target hidden, target window... |
| `runtime.enginelab.prompt_lookup` | prompt lookup drafting and sparse lookup: the prompt-lookup draft and runtime (HIP), sparse lookup and its executor,... |
| `runtime.enginelab.spec_block_draft` | block drafting (include/strix/spec, src/spec): block candidates, block drafter and policy, block-parallel... |
| `runtime.enginelab.spec_core` | speculation core (include/strix/spec, src/spec): drafting and draft declarations, the draft-verify runner, engine... |
| `runtime.enginelab.spec_wide_window` | wide verify windows (include/strix/spec): the backend-neutral pricing of a wide window (rows-1 copied proposals plus... |
| `runtime.enginelab.speculation_control` | the speculation controller: speculation, controller, cycle, executor, measurement and multi-draft, and the draft... |
| `runtime.enginelab.speculation_sampling` | sampled speculation: sampling, sampling cycle and executor, the keyed RNG and the compiled sampled-speculation backend |
| `runtime.enginelab.speculative_verify` | speculative verification: verify binding, plan and runtime, the compiled verify and the compiled speculation backend |
| `runtime.kyojin.draft2s` | Two-stage drafter module (draft2s) of the speculative path |
| `runtime.kyojin.draft_confidence` | Confidence-gated draft length (stop drafting when the draft's probability falls) |
| `runtime.kyojin.draft_depth` | Acceptance-driven draft depth for block drafters (DFlash2): an EMA of accepted tokens picks full depth, a short... |
| `runtime.kyojin.hy4_mtp` | Hy4 MTP draft head (the GGUF next-n block with two Triton expert kernels) and the split draft chain (base graph +... |
| `runtime.kyojin.loop_detect` | Repetition loop detection on the generated stream |
| `runtime.kyojin.sam` | Suffix automaton over the token history (BC_SAM): incremental extend and longest-match lookup, the prompt-lookup... |
| `runtime.kyojin.spec_sample` | Speculative sampling and acceptance of drafted tokens (greedy and sampled rules) |
| `runtime.maya_tp.glm_dflash` | the DFlash2 block drafter and its decode on the GLM trunk (glm_dflash.cu, 2,075 lines) with the adaptive... |
| `runtime.maya_tp.glm_mtp` | GLM MTP speculative decoding: the one-GPU MTP decode where the NextN block drafts (glm_mtp), several conversations... |
| `runtime.maya_tp.glm_ngram` | the n-gram draft matcher and its decode_spec session (glm_ngram_draft) and head-group pricing with acceptance... |
| `runtime.maya_tp.mm3_dspark` | maya-mm3's DSpark draft (mm3_dspark.cu / .hpp) |
| `runtime.maya_tp.spec_controller` | speculative-decoding control: the controller, the draft-length policy with its cost model and the suffix drafter... |
| `runtime.maya_tp.verify_mtp` | the Strata-lineage speculative verify and MTP drivers (verify.cpp, mtp.cpp) with the MTP row helpers |
| `runtime.vllm_strix.adaptive_k` | adaptive draft length for speculative decoding wired for async scheduling: an EMA of acceptance sets the verified... |
| `runtime.vllm_strix.apc_align` | prefix-cache alignment for GLM-5.3 with EAGLE / DFlash: the EAGLE last-block drop is resolved to pure-drafter KV... |
| `runtime.vllm_strix.dflash_drafter_kv_group` | puts the DFlash2 drafter's KV into its own group in _get_kv_cache_groups_glm5_next, so the drafter's attention... |
| `runtime.vllm_strix.eagle3_aux_states` | auxiliary hidden states for EAGLE3-style drafters from the GLM-5-Next model: the patch makes... |
| `tool.enginelab.apps_mtp` | MTP application pieces: the MTP cohort bridge and tiled cohort, the decode bridge, wide copy verify (a long... |

### F12 Tensor-parallel partitioning and collectives (56)

| module | what it holds |
|---|---|
| `executor.enginelab.ds41_pipeline` | the DeepSeek V4.1 two-node path: the pipeline front, multi-front and wire, decode transport, prefill transport (HIP)... |
| `executor.enginelab.glm_kvshard` | GLM KV sharding: shard, dump, exchange, layout and math: 6 origin files under src/glm |
| `kernel.ds4.glmtp_rocm_tp` | [glmtp overlay] rocm_tp family: rocm/ds4_rocm_tp.cuh added (1073 lines). Kernels not in the base:... |
| `kernel.ds4.k53perf_rocm_tp` | [k53perf overlay] rocm_tp family: rocm/ds4_rocm_tp.cuh added (324 lines). Kernels not in the base: rocm_tp_add,... |
| `kernel.enginelab.core_tp_join` | the exact tensor-parallel join: tp_join.hpp and tp_join_sum.hpp sum the two ranks' partial outputs in a declared... |
| `kernel.enginelab.ds41_expert_tp` | DeepSeek V4.1 expert tensor parallelism over two ranks (identity ds41-expert-tp2-i1152-v1): every routed expert... |
| `kernel.maya_tp.tp_kernels` | TP2 delta: the GPU half of a queued TP2 exchange (src/net/tp_kernels.cu, include/strata/net/tp_kernels.hpp):... |
| `runtime.ds4.distributed` | the distributed (pipeline-parallel) runtime: workers execute contiguous layer slices with the local graph-slice... |
| `runtime.ds4.glmtp_distributed` | [glmtp overlay] distributed family: ds4_distributed.c changed (+1171 -26 lines); ds4_distributed.h changed (+16 -0... |
| `runtime.ds4.glmtp_gpu_tp` | [glmtp overlay] gpu_tp family: ds4_gpu_tp.h changed (+41 -2 lines). The family: the TP gate interface between the... |
| `runtime.ds4.glmtp_tp` | [glmtp overlay] tp family: ds4_tp.c changed (+1717 -205 lines); ds4_tp.h changed (+61 -4 lines). The family: the... |
| `runtime.ds4.gpu_tp` | the TP gate interface between the graph and a backend: gate sequences both ranks must issue, the optional pinned... |
| `runtime.ds4.k53flash_tp` | [k53flash overlay] tp family: ds4_tp.c changed (+288 -1777 lines); ds4_tp.h changed (+7 -113 lines). The family: the... |
| `runtime.ds4.k53perf_gpu_tp` | [k53perf overlay] gpu_tp family: ds4_gpu_tp.h changed (+19 -7 lines). The family: the TP gate interface between the... |
| `runtime.ds4.k53perf_tp` | [k53perf overlay] tp family: ds4_tp.c changed (+390 -844 lines); ds4_tp.h changed (+10 -82 lines). The family: the... |
| `runtime.ds4.k53prefill_tp` | [k53prefill overlay] tp family: ds4_tp.c changed (+288 -1777 lines); ds4_tp.h changed (+7 -113 lines). The family:... |
| `runtime.ds4.kmultinode_distributed` | [kmultinode overlay] distributed family: ds4_distributed.c changed (+11 -34 lines). The family: the distributed... |
| `runtime.ds4.tp` | the tensor-parallel transport and lockstep protocol: two ranks run the same graph, each with one contiguous half of... |
| `runtime.enginelab.dist_communicator` | the two-rank communicator and message transport (include/strix/dist): whole messages in order between two ranks,... |
| `runtime.enginelab.dist_gpu_direct` | GPU-direct exchange: setup and copies between the two iGPUs' memory through the USB4 DMA-BUF stream extension... |
| `runtime.enginelab.dist_host_staged` | host-path staging: host copies, host waits and the staged-ready protocol with its HIP kernels (a producer publishes... |
| `runtime.enginelab.distributed_control` | distributed control: the control wire, greedy control, remote control, token transactions, position completion and... |
| `runtime.enginelab.distributed_runtime` | the distributed runtime: data plane, native backend, placement, runtime, session and stage resources / state |
| `runtime.enginelab.glm_xfer` | GLM exchange: post exchange and fast post, sequence parallel, traits, the transfer (device, exchange) and its bench |
| `runtime.enginelab.handoff` | handoff and output: handoff lanes and policy, final generation state export, generation output (HIP, dispatch,... |
| `runtime.enginelab.minimax_pfx` | MiniMax prefill exchange: CRC, exchange, route and wire: 6 origin files under src/minimax |
| `runtime.enginelab.minimax_xfer` | MiniMax exchange: traits, transfer and device transfer: 4 origin files under src/minimax |
| `runtime.enginelab.pair_link` | pair link pieces (include/strix/pair): link, link flags and mux, gather link and fn, BF16 wire, producer publish... |
| `runtime.enginelab.pair_pipeline` | two-node pipelines (include/strix/pair): layer pipeline, layer cut, stage map, chains and chain topology, prefill... |
| `runtime.enginelab.pair_state` | pair state and lockstep (include/strix/pair): lockstep control bookkeeping (rank, round and last completed round... |
| `runtime.enginelab.xport_kvshard` | KV-shard exchange: the KV-shard channel (host and HIP), its traits and oracle: 4 origin files under src/xport |
| `runtime.enginelab.xport_tp_gather` | the tensor-parallel slot gathers: the one instantiation of the shared Channel for TP gathers (tp_gather.hip and its... |
| `runtime.kyojin.layer_split` | Contiguous layer split across two devices: the serve-side split plan, an async host-staged bridge (part0 and part1... |
| `runtime.kyojin.tp_allreduce_cpu` | CPU-assisted all-reduce of the native tensor-parallel backend: devices stage 128 KB wire chunks to pinned memory, a... |
| `runtime.kyojin.tp_broadcast_gather` | Broadcast and gather collectives over pinned host memory with per-block stage flags |
| `runtime.kyojin.tp_context` | TP context, epoch barrier and the sticky timeout (once one collective times out, every later wait aborts) |
| `runtime.kyojin.tp_python` | Python side of the native tensor-parallel backend (allocation, backend, shared arena, functions) and the CPU MoE... |
| `runtime.maya_tp.ds4_tp_batch` | TP2 delta: DeepSeek TP2's opt-in multi-sequence batch metadata (ds4_tp_batch.hpp, ds4_tp_batch_state.hpp): batch... |
| `runtime.maya_tp.ds4_tp_control` | TP2 delta: DeepSeek TP2's control plane (src/ds4/ds4_tp_control, ds4_tp_channel, ds4_tp_ack, ds4_tp_chunk,... |
| `runtime.maya_tp.ds4_tp_loader` | TP2 delta: DeepSeek TP2's rank-local weight loading (ds4_tp_partition, ds4_tp_pack, ds4_tp_dense_loader,... |
| `runtime.maya_tp.ds4_tp_worker` | TP2 delta: DeepSeek TP2's rank-1 side (ds4_tp_worker.hpp, ds4_tp_worker_state.hpp, ds4_tp_model_worker.hpp): the... |
| `runtime.maya_tp.glm_tp_batch` | TP2 delta: GLM's TP2 application frames (include/strata/core/glm_tp_batch.hpp): versioned multi-slot batch plans and... |
| `runtime.maya_tp.mm3_tp` | TP2 delta: maya-mm3's tensor-parallel path (mm3_tp.cu / .hpp, 2,884 lines): GPUs in groups of tp, each group a run... |
| `runtime.maya_tp.tp_comm_abi` | TP2 delta: the ordered communication boundary and its wire identity (include/strata/net/communicator.hpp,... |
| `runtime.maya_tp.tp_context` | TP2 delta: the GLM engine's two-rank tensor-parallel state (src/core/tp_context.cu, tp_context.hpp,... |
| `runtime.vllm_strix.cg_eager_collectives` | CUDA graphs with TP collectives kept out: with VLLM_USE_BREAKABLE_CUDAGRAPH=1 and PIECEWISE mode, every... |
| `runtime.vllm_strix.odl_ar2` | odl_ar2, a two-rank decode all-reduce over OdinLink streams: on the stream a D2H copy into a pinned send slot, a... |
| `runtime.vllm_strix.odl_ar2_hook` | the vLLM hook that sends decode-sized all-reduces in CudaCommunicator.all_reduce to odl_ar2 when DS4_ODL_AR2=1 and... |
| `runtime.vllm_strix.odl_mq` | odl_mq, an OdinLink stream data plane for vLLM's MessageQueue remote readers: the writer opens an auto-assigned... |
| `runtime.vllm_strix.odl_mq_hook` | the shm_broadcast hook that lets MessageQueue Handles carry an odl_mq stream id and makes remote readers and writers... |
| `runtime.vllm_strix.shm_spin` | makes the SHM MessageQueue reader's busy-spin window tunable: SpinCondition.wait() yields in a loop for busy_loop_s... |
| `serve.vllm_strix.cluster_env` | the transport profiles: the canonical NCCL / memory / Ray environment both boxes source, and the odl profile (RCCL... |
| `tool.maya_tp.tp2_launch` | TP2 delta: launching a TP2 pair: the Python launcher (tools/tp2/launch.py, owned_exec.py), the rank scripts... |
| `tool.maya_tp.tp2_validation` | TP2 delta: validating and benchmarking a TP2 pair: acceptance suite (acceptance.sh), split audit, greedy parity, MTP... |
| `tool.maya_tp.tp_tests` | TP2 delta: the TP unit and pair tests: tests/tp_*.cpp (collective, RDMA communicator, context, control-socket state,... |
| `tool.vllm_strix.odl_ar2_bench` | standalone two-rank bench for odl_ar2: loads libodl_ar2.so, runs warm-up and timed all-reduces of decode-sized bf16... |

### F13 Link transports and their drivers (USB4 / Thunderbolt / RoCE) (60)

| module | what it holds |
|---|---|
| `kernel.maya_tp.peer_probe` | a peer-access probe program (src/kernels/peer_probe.cu): checks hipDeviceCanAccessPeer and times peer copies between... |
| `runtime.ciru_link.nhi_endpoint` | the NHI endpoint lifecycle in Go (internal/transport): Inspect discovers thunderbolt-net, the stream service and... |
| `runtime.ciru_link.tbstream_dmabuf` | import of GPU-owned DMA-BUFs into thunderbolt_stream: a DMA-BUF import probe with the scatter-gather helpers in... |
| `runtime.ciru_link.tbstream_hopid` | HopID ownership between thunderbolt-net and thunderbolt_stream: the network driver records which receive HopID it... |
| `runtime.ciru_link.tbstream_ring_fixes` | NHI ring reliability fixes: flush posted MSI-X interrupt clears (0011), prime the RX ring before enabling the DMA... |
| `runtime.ciru_link.tbstream_zerocopy` | the base of the Linux 7.2.2 USB4STREAM patch series: NHI rings that suppress per-frame completions (0001), the... |
| `runtime.ciru_link.usb4net` | USB4NET discovery and setup in Go (internal/link): finds thunderbolt-net interfaces and Thunderbolt paths, loads the... |
| `runtime.ds4.k53perf_tp_io` | [k53perf overlay] tp_io family: ds4_tp_io.h added (75 lines). The family: full-duplex socket I/O for the TP... |
| `runtime.ds4.k53perf_tp_roce` | [k53perf overlay] tp_roce family: ds4_tp_roce.h added (209 lines). The family: an optional Linux RC SEND / RECV... |
| `runtime.enginelab.direct_gpu_stream` | GPU-direct streams: boundary staging, pool allocator, stream commissioning and the GPU side, the Linux GPU-stream... |
| `runtime.enginelab.direct_transport` | the native USB4 stream transport: driver, Linux driver over /dev/tbstreamN, completion, frames, GPU export, lease,... |
| `runtime.enginelab.dual_link` | dual-link runtime and telemetry: two USB4 cables as one link: 4 origin files under include/strix,... |
| `runtime.enginelab.xport_channel` | the shared exchange channel (src/xport): backend and transport backend, borrowed transport, channel (device, HIP,... |
| `runtime.enginelab.xport_crc` | wire integrity: CRC32C folding on the device (device_crc32c.hip) and the host (vector and parallel CRC32C, the... |
| `runtime.enginelab_usb4.dma_recovery` | DMA lifecycle and recovery: DMA progress, idle reset and quarantine reaping (0006), external dependency errors... |
| `runtime.enginelab_usb4.experimental` | the three experimental patches on top of the stable series (patches/experimental: 0049 host bounded RX poll window,... |
| `runtime.enginelab_usb4.gpu_queue` | the GPU stream queue: batched GPU ring submission (0003), fence and stop lifetime (0004), GPU queues, fences and... |
| `runtime.enginelab_usb4.gpu_stream_driver` | the DMA-BUF GPU stream driver as captured after 0001: overlay/drivers/thunderbolt/gpu_stream.c, gpu_stream.h and... |
| `runtime.enginelab_usb4.gpu_stream_uapi` | the experimental GPU stream UAPI header overlay/include/uapi/linux/strix_gpu_stream.h (as captured after patch 0001) |
| `runtime.enginelab_usb4.host_completion` | host completion path: completion path timing (0040), host completion latency tracepoints (0046), host RX-only polled... |
| `runtime.enginelab_usb4.kernel_backports` | an unmodified upstream Linux fix applied after the series: drm/amdkfd dma_buf reference leak in get_dmabuf_info... |
| `runtime.enginelab_usb4.link_training` | link and XDomain: credit accounting and link diagnostics (0002), kernel-doc for the stream / tbnet / CM ops members... |
| `runtime.enginelab_usb4.nhi_overlay` | the rest of the 0001 capture: overlay/drivers/thunderbolt (domain.c, nhi.c / nhi.h / nhi_regs.h, pci.c, stream.c,... |
| `runtime.enginelab_usb4.nhi_teardown` | AMD NHI teardown: CM unlinks tunnels on stop (0012), AMD interface reset when idle (0018), NHI shutdown releases... |
| `runtime.enginelab_usb4.stream_lifecycle` | stream open / close / IO correctness: open error path and frame arrays (0014), detach RAM rings and domain / XDomain... |
| `runtime.enginelab_usb4.stream_rx_latency` | stream RX completion and latency: NAPI-style hybrid RX completion (0010), per-stream NAPI and IRQ steering (0011),... |
| `runtime.enginelab_usb4.usb4stream_base` | patch 0001, the USB4STREAM source as the series starts it: configfs-created /dev/tbstreamN character devices for... |
| `runtime.maya_tp.tp_rdma` | TP2 delta: the two-host RDMA transport over RoCE v2 RC queue pairs on Thunderbolt usb4_rdma (src/net/verbs.cpp,... |
| `runtime.odinlink5.compress` | optional link compression: a wire header [odl_compress_header / compressed payload / padding], an nvCOMP backend... |
| `runtime.odinlink5.driver_apple` | the Apple-Silicon transport backend: Apple NHI DMA on the ACIO fabric from analysis of macOS's AppleThunderboltNHI... |
| `runtime.odinlink5.driver_core` | the odl_tb5 kernel driver core: module lifecycle (advertises the OdinLink protocol in XDomain property directories,... |
| `runtime.odinlink5.driver_nhi` | the NHI data path of the driver: the transport backend that wraps the kernel's tb_ring API and XDomain path... |
| `runtime.odinlink5.lib` | libodl_tb5, the userspace API over /dev/odl_tb5_N: open / close / peer state, the stream API (stream_open by filter... |
| `runtime.odinlink5.mac_rdma` | Linux to Mac memory over Thunderbolt 5: a macOS kext (OdinLinkNHI, OdinLinkRDMA) that posts 4-KiB RX descriptors... |
| `runtime.odinlink5.nccl_net_plugin` | the NCCL net plugin (libnccl-net-ODL_TB5.so) for CUDA hosts: the same stream transport with up to 16 devices, 64... |
| `runtime.odinlink5.rccl_net_plugin` | the RCCL net plugin (librccl_net_odl_tb5.so, symlinked librccl-net.so for RCCL's probe): ncclNet_v7 over the stream... |
| `runtime.odinlink5.verbs_provider` | a libibverbs provider over OdinLink: devices, protection domains, memory regions, completion queues, queue pairs and... |
| `runtime.vllm_strix.nhi_throttle` | nhi_throttle, a tiny kernel module that rewrites the USB4 / Thunderbolt NHI MSI-X interrupt throttling registers... |
| `runtime.vllm_strix.odinlink_host_kit` | the OdinLink host kit: fetch OdinLink-Five at the pin and apply the local patch, build odl_tb5.ko against stock... |
| `runtime.vllm_strix.odinlink_local_patch` | five fixes on OdinLink-Five 4534f58 and nothing else: connection restart after a failed DMA verify, verify that... |
| `runtime.vllm_strix.odl_state` | odl-state, the readiness helper: opens /dev/odl_tb5_0 and _1 through libodl_tb5, prints each device's peer state and... |
| `tool.ciru_link.cli` | the ciru-strixlink command (cmd/ciru-strixlink/main.go) with go.mod and the Makefile: version, prerequisites, setup... |
| `tool.ciru_link.linktest` | a dependency-free TCP link test (internal/bench: token-authenticated protocol on port 55321, upload / download... |
| `tool.ciru_link.prereq` | host prerequisites and install plans in Go: a versioned, UI-safe report of the kernel, modules, tools (ip, ethtool,... |
| `tool.ciru_link.tbstream_diag` | zero-copy progress diagnostics for thunderbolt_stream: ring and slot progress counters in nhi.c and stream.c with... |
| `tool.ds4.glmtp_tp_odl_test` | [glmtp overlay] tp_odl_test family: tp_odl_test.c added (203 lines). The family: a transport-level A/B for the TP... |
| `tool.enginelab.apps_dist` | transport probes: GPU, host exchange bench, TCP probe and the transfer probe: 4 origin files under... |
| `tool.enginelab.lab_direct_gpu_stream` | the direct GPU stream commissioning lab: bounded TCP control, observe driver, visibility contract, kernels and... |
| `tool.enginelab.lab_gpu_queue` | the GPU queue lab: native queue main and memory (lab/gpu_queue): 4 origin files under lab/gpu_queue |
| `tool.enginelab.lab_transport` | the transport lab (lab/transport): 6 origin files under lab/transport |
| `tool.enginelab_usb4.bench` | stream benchmarks: pingpong.c (round-trip latency), stream_mem_bench.c (throughput) and tools/lab/run_stream_bench.py |
| `tool.enginelab_usb4.diag` | diagnostics and build helpers: usb4_diagnose.py (controller / ring topology and budget), usb4_linkstat.py,... |
| `tool.enginelab_usb4.lab` | the lab setup: usb4stream_describe.py and its udev rule and refresh service, setup-usb4stream-lab, node_check.sh,... |
| `tool.odinlink5.cli` | odl_tb5_cli, the link test tool: server mode answers, client mode runs bandwidth, latency, latency-under-load,... |
| `tool.odinlink5.daemon` | odl-tb5-daemon, a GLib background service: D-Bus API for peer status (com.odinlink.Tb5Daemon), cable plug / unplug... |
| `tool.odinlink5.packaging` | build and packaging: the top-level CMake (driver via Kbuild, library, plugins, verbs, daemon and tray with... |
| `tool.odinlink5.remote_tensors` | Python tensor offload over Thunderbolt IP: the TB-Bridge client / server (pure Python, 25-40 Gb/s over TB5 IP,... |
| `tool.odinlink5.tests` | smoke and stress tests: device ioctl, library API and plugin suites in one plain-C runner, an RCCL DMA-BUF test,... |
| `tool.odinlink5.tray` | the GTK / Ayatana AppIndicator tray app: peers window, link tests with charts and result parsing, RCCL stats view,... |
| `tool.vllm_strix.odl_pingpong` | RTT latency bench over the OdinLink stream layer: server echoes, client sends [4-byte size][payload in <= ODL_CHUNK... |

### F14 Prompt cache, prefix sharing and prefill scheduling (19)

| module | what it holds |
|---|---|
| `runtime.ds4.kvstore` | disk KV checkpoint files: the shared layout and payload helpers under the server's byte-prefix cache and the agent's... |
| `runtime.ds4.prompt_prefix` | role-marker prompt-prefix detection for prefix reuse |
| `runtime.enginelab.known_prompt` | known-prompt prefill: cancellation, limits, prefix prefill and runtime, prefix window (HIP), prompt backend and... |
| `runtime.enginelab.prefill_plan` | prefill planning: dense plan, headless prefill, identity, MoE policy, piece plan, pool, row liveness, stage... |
| `runtime.enginelab.prefix_state` | prefix state caching: the prefix state cache, state-cache snapshot, context key index, block summary state and... |
| `runtime.enginelab.qpf2k_schedule` | the qpf2k prefill schedule defaults and pinned buffers (STRIX_QPF2K_SCHED, the code-default exact long-context items) |
| `runtime.enginelab.session_snapshot` | session state: executor handoff, restore, snapshot (memory, transfer, HIP transfer), state cache, layout and memory |
| `runtime.enginelab.state_txn` | state declarations and transactions (src/state, include/strix/state): decl, lane view, table and txn |
| `runtime.kyojin.prefix_disk` | Persistent prompt cache on NVMe: content-addressed pages (blake2b of page ids and the previous hash) survive lane... |
| `runtime.kyojin.sched_chunk_il` | Chunked-prefill interleave planner (default on since 1.8.8): decode rounds of running jobs are interleaved between... |
| `runtime.kyojin.sched_prefill_share` | Concurrent prefill of one shared prefix: jobs starting together on one prompt prefix wait for the page owner instead... |
| `runtime.kyojin.sched_row_groups` | Row invariance of multi-job and multi-row verify forwards: jobs grouped so a row gets the same bits whatever else is... |
| `runtime.kyojin.slot_store` | llama.cpp-compatible slot persistence (GET /slots, save / restore / erase) for the paged generator |
| `runtime.maya_tp.glm_prompt_cache` | the content-addressed NVMe prompt cache for GLM (glm_prompt_cache store, its engine glue and model side) |
| `runtime.maya_tp.prefill_host` | the prompt path's host orchestration (src/prefill/prefill.cpp, 2,026 lines) over the prefill kernels and MoE prompt... |
| `runtime.vllm_strix.apc_retention` | adds --prefix-cache-retention-interval 2304 (2176 with speculation off) to the GLM serve script behind... |
| `serve.enginelab.serve_pflash` | PFlash prompt compression: per-model regions (DeepSeek, GLM, MiniMax, Qwen), the prompt compressor and scorer, the... |
| `serve.enginelab.serve_prefix` | serve-side prompt caching: prefix directory, join and cache-queue binding, prompt join, piece, service and stream,... |
| `serve.vllm_strix.warmup` | post-boot warm-up: a transient unit started once the API answers fires dummy requests that trigger and cache the JIT... |

### F15 Executors, engines, graphs and model families (123)

| module | what it holds |
|---|---|
| `executor.enginelab.ds41_engine` | the DeepSeek V4.1 engine: model, family, plan, describe and model description, host / device / batch forward and... |
| `executor.enginelab.ds41_prefill` | DS41 prefill: the HIP prefill and its relaxed form, adapter, precision, route and tuning, the DeepSeek prefill and... |
| `executor.enginelab.gemma4_engine` | the Gemma 4 engine (src/gemma4/engine): 26 origin files under src/gemma4 |
| `executor.enginelab.gemma4_features` | Gemma 4 features (per-layer inputs, media features) (src/gemma4/features): 5 origin files under src/gemma4 |
| `executor.enginelab.gemma4_model` | Gemma 4 model description and layers (src/gemma4/model): 7 origin files under src/gemma4 |
| `executor.enginelab.gemma4_prefill` | Gemma 4 prefill (src/gemma4/prefill): 22 origin files under src/gemma4 |
| `executor.enginelab.glm_engine` | the GLM-5.3 engine: engine, embed splice, fast attention, fast INT4 defaults and pack, fast MoE, JSON, library,... |
| `executor.enginelab.glm_prefill` | GLM prefill: HC expand backward, KV-shard prefill, MoE traits, aux, capacity and dense policies, exact (MoE, x),... |
| `executor.enginelab.h3_block` | H3 transformer blocks (src/h3/block): 21 origin files under src/h3 |
| `executor.enginelab.h3_model` | the H3 model root (src/h3): 1 origin files under src/h3 |
| `executor.enginelab.h3_te` | H3 text encoder (src/h3/te): 10 origin files under src/h3 |
| `executor.enginelab.hy4_engine` | the Hy4 engine root and serving (session service, include/strix/hy4): 7 origin files under include/strix, src/hy4 |
| `executor.enginelab.inkling2_engine` | the Inkling 2 Mini engine: calibration, checkpoint, config, controller and residual device code, dense HIP, forward,... |
| `executor.enginelab.inkling2_resident` | Inkling 2 Mini resident backend, decoder, forward, route and weights: 6 origin files under src/inkling2 |
| `executor.enginelab.minimax_engine` | the MiniMax M3.1 engine: deployment, linear drafter, engine, common, defaults, fast INT4, kernels, pack, profile... |
| `executor.enginelab.qwen_engine` | the Qwen engines: Qwen3.8-Flash-Next and Qwen3.8-27B model descriptions, MoE prepared and BF16 router, all-on,... |
| `executor.enginelab.qwen_prefill` | Qwen prefill: balance, chain, handoff, pair and pair staging, pipe dump, pipeline, schedule, stages and trace |
| `executor.kyojin.dsv41_reference` | DeepSeek-V4.1-Flash CPU-exact reference engine (ops, fp8 / fp4 quant, Engram, model, weights I/O) |
| `executor.kyojin.hy4_forward` | Hy4-flash whole-model forward (slow but correct, torch only), its CPU fp32 attention reference and the opt-in... |
| `kernel.ds4.glmtp_rocm_backend` | [glmtp overlay] rocm_backend family: ds4_rocm.cu changed (+16 -78 lines); ds4_rocm.h changed (+4 -0 lines). The... |
| `kernel.ds4.glmtp_rocm_runtime` | [glmtp overlay] rocm_runtime family: rocm/ds4_rocm_runtime.cuh changed (+75 -4 lines). The family: the ROCm runtime... |
| `kernel.ds4.k53flash_rocm_backend` | [k53flash overlay] rocm_backend family: ds4_rocm.cu changed (+3 -1 lines); ds4_rocm.h changed (+5 -11 lines). The... |
| `kernel.ds4.k53flash_rocm_runtime` | [k53flash overlay] rocm_runtime family: rocm/ds4_rocm_runtime.cuh changed (+48 -17 lines). The family: the ROCm... |
| `kernel.ds4.k53perf_rocm_backend` | [k53perf overlay] rocm_backend family: ds4_rocm.cu changed (+32 -31 lines); ds4_rocm.h changed (+1 -6 lines). The... |
| `kernel.ds4.k53perf_rocm_qwen4` | [k53perf overlay] rocm_qwen4 family: rocm/ds4_rocm_qwen4.cuh added (2614 lines). Kernels not in the base: argmax,... |
| `kernel.ds4.k53perf_rocm_runtime` | [k53perf overlay] rocm_runtime family: rocm/ds4_rocm_runtime.cuh changed (+847 -103 lines). The family: the ROCm... |
| `kernel.ds4.k53prefill_rocm_backend` | [k53prefill overlay] rocm_backend family: ds4_rocm.cu changed (+3 -4 lines); ds4_rocm.h changed (+5 -11 lines). The... |
| `kernel.ds4.k53prefill_rocm_runtime` | [k53prefill overlay] rocm_runtime family: rocm/ds4_rocm_runtime.cuh changed (+48 -17 lines). The family: the ROCm... |
| `kernel.ds4.kmultinode_rocm_backend` | [kmultinode overlay] rocm_backend family: ds4_rocm.cu changed (+2 -134 lines); ds4_rocm.h changed (+5 -11 lines).... |
| `kernel.ds4.kmultinode_rocm_misc_launch` | [kmultinode overlay] rocm_misc_launch family: rocm/ds4_rocm_misc_launch.cuh changed (+0 -51 lines). The family:... |
| `kernel.ds4.kmultinode_rocm_runtime` | [kmultinode overlay] rocm_runtime family: rocm/ds4_rocm_runtime.cuh changed (+85 -4814 lines). The family: the ROCm... |
| `kernel.ds4.rocm_backend` | ds4's ROCm backend translation unit: ds4_rocm.cu sets the HIP spellings (64-bit warp masks, hipBLASLt / rocBLAS... |
| `kernel.ds4.rocm_misc_launch` | launch glue for the pointwise and output kernels; a fragment of the single ROCm translation unit ds4_rocm.cu, where... |
| `kernel.ds4.rocm_runtime` | the ROCm runtime body: device tensors and arenas, streams, rocBLAS / hipBLAS handles and tuned solution indices, the... |
| `kernel.enginelab.gemma4_kernels` | Gemma 4 device kernels (src/gemma4/kernels): 25 origin files under src/gemma4 |
| `kernel.enginelab.glm_kernels` | GLM kernels: DSA, norm, tier A and the reduction tree: 6 origin files under src/glm |
| `kernel.enginelab.hy4_hip` | Hy4 HIP forward, plan and rows: 5 origin files under src/hy4 |
| `kernel.maya_tp.cpu_glm_lane` | the glm-dsa CPU expert lane's lockstep 4-row decode GEMV (glm_lane_row4, glm_lane_row4_avx2, B14.9) |
| `kernel.maya_tp.mm3_kernels` | maya-mm3's kernels around the EXL3 matrices for MiniMax-M3.1-Flash (mm3_kernels.cu / .cuh): attention, norms, router... |
| `runtime.ds4.engine` | ds4.c and ds4.h, the engine: mmap GGUF loading, the fixed DeepSeek V4 / V4.1 / Pro, GLM-5.2 / 5.3 / 5.3-Flash and... |
| `runtime.ds4.glmtp_engine` | [glmtp overlay] engine family: ds4.c changed (+2816 -2920 lines); ds4.h changed (+104 -0 lines). The family: ds4.c... |
| `runtime.ds4.glmtp_gpu_api` | [glmtp overlay] gpu_api family: ds4_gpu.h changed (+99 -216 lines). The family: the backend ABI: ds4_gpu.h... |
| `runtime.ds4.gpu_api` | the backend ABI: ds4_gpu.h (tensor-resident GPU executor API: opaque device tensors and every op the graph issues),... |
| `runtime.ds4.k53flash_engine` | [k53flash overlay] engine family: LICENSE changed (+0 -1 lines); ds4.c changed (+1317 -16567 lines); ds4.h changed... |
| `runtime.ds4.k53flash_gpu_api` | [k53flash overlay] gpu_api family: ds4_gpu.h changed (+67 -762 lines). The family: the backend ABI: ds4_gpu.h... |
| `runtime.ds4.k53perf_engine` | [k53perf overlay] engine family: ds4.c changed (+3962 -5764 lines); ds4.h changed (+5 -21 lines). The family: ds4.c... |
| `runtime.ds4.k53perf_gpu_api` | [k53perf overlay] gpu_api family: ds4_gpu.h changed (+62 -353 lines). The family: the backend ABI: ds4_gpu.h... |
| `runtime.ds4.k53prefill_engine` | [k53prefill overlay] engine family: LICENSE changed (+0 -1 lines); ds4.c changed (+1308 -17336 lines); ds4.h changed... |
| `runtime.ds4.k53prefill_gpu_api` | [k53prefill overlay] gpu_api family: ds4_gpu.h changed (+67 -864 lines). The family: the backend ABI: ds4_gpu.h... |
| `runtime.ds4.kmultinode_engine` | [kmultinode overlay] engine family: LICENSE changed (+0 -1 lines); ds4.c changed (+2546 -65702 lines); ds4.h changed... |
| `runtime.ds4.kmultinode_gpu_api` | [kmultinode overlay] gpu_api family: ds4_gpu.h changed (+41 -2932 lines). The family: the backend ABI: ds4_gpu.h... |
| `runtime.ds4.qwen4_unicode` | generated Unicode tables for the Qwen tokenizer |
| `runtime.ds4.rax` | rax, antirez's radix tree (BSD-3-Clause) |
| `runtime.enginelab.backend_cpu` | the CPU backend (src/cpu: arch, core, weight): 24 origin files under src/cpu |
| `runtime.enginelab.exec_program` | the exec layer (src/exec, include/strix/exec): arena, catalog, cohort programs, compiler, descriptor, draft runner,... |
| `runtime.enginelab.execution` | compiled execution: arena, binding, buffers, IO, memory, plan, resources, runtime, telemetry, tensor access and... |
| `runtime.enginelab.gemma4_contracts` | Gemma 4 numeric contracts (src/gemma4/contracts): 2 origin files under src/gemma4 |
| `runtime.enginelab.gemma4_kv` | Gemma 4 KV cache (sliding and global layers) (src/gemma4/kv): 5 origin files under src/gemma4 |
| `runtime.enginelab.gemma4_ref` | Gemma 4 host references (src/gemma4/ref): 7 origin files under src/gemma4 |
| `runtime.enginelab.glm_load` | GLM loading: model description, peer load, rank plan (graph, main): 8 origin files under src/glm |
| `runtime.enginelab.graph` | model graphs (src/graph, include/strix/graph): chains, launch plan, model declaration, program segments, rows plan,... |
| `runtime.enginelab.h3_common` | H3 common headers (src/h3/common): 3 origin files under src/h3 |
| `runtime.enginelab.h3_dist` | H3 distributed (two-node) pieces (src/h3/dist): 12 origin files under src/h3 |
| `runtime.enginelab.h3_levers` | H3 levers (sparse attention study and others) (src/h3/levers): 13 origin files under src/h3 |
| `runtime.enginelab.host_common` | host common: qpersist and its diagnostic (switches for verify-window fused GDN / QSA tails, bit-identical to the... |
| `runtime.enginelab.hy4_ref` | Hy4 host references (DFly and host): 4 origin files under src/hy4 |
| `runtime.enginelab.launch` | launch plumbing: dispatch fuse and resources, launch context, dispatch and overlap, the launcher catalog, runtime... |
| `runtime.enginelab.levers` | levers and profiles: the lever registry and cache, turbo, fast and engine profiles, environment arms and flags, the... |
| `runtime.enginelab.minimax_session` | MiniMax session runtime and cache: 3 origin files under src/minimax |
| `runtime.enginelab.model_graph` | model-graph adapters (src/model_graph): the model graph and plan, stage bridge, and the DeepSeek V4.1, dense and... |
| `runtime.enginelab.numeric` | numerics declarations: declared numerics, gfx1151 math and exact rsqrt: 3 origin files under include/strix |
| `runtime.enginelab.prepared_views` | prepared execution: arena views, prepared execution, the prepared serial session and tensor views |
| `runtime.enginelab.receipts` | receipts and offline tuning: the receipt writer, validator, step-profile report and legacy form; raw baseline... |
| `runtime.enginelab.serial_session` | the serial (one-request) decode path: decode reference and replay, host state shadow, prefix cache and lifecycle,... |
| `runtime.enginelab.session_engine` | the session engine contract (include/strix/session, src/session): backends (CPU, HIP), engine, CLI output, cycle... |
| `runtime.enginelab.support_matrix` | the support matrix (models x routes x profiles): 2 origin files under include/strix, src/support |
| `runtime.enginelab.test_hooks` | the three tests/ headers that harvested sources include: the Qwen rotary watchdog and fixtures (included by... |
| `runtime.kyojin.ablit_runtime` | Run-time abliteration edit (a projection removed from writer outputs, no weights changed) |
| `runtime.kyojin.bc_dsv4_attn` | Whole-attention-step graph for DeepSeek-V4 DSA decode: staged input, q_a / q_norm / q_b, wkv, fused-norm rope,... |
| `runtime.kyojin.bc_gated_delta_net` | Graph-captured Gated DeltaNet block (BC_GatedDeltaNetSplit pattern) |
| `runtime.kyojin.bc_mlp` | Graph-captured MLP and linear blocks: gate and up as one fused MGEMM when pointer tables are given, its own scratch... |
| `runtime.kyojin.block_graph` | Per-block graph capture of a decoder layer |
| `runtime.kyojin.dsv41_module` | DS4.1 module of the EXL3 engine (glue to the dsv41_fn kernels) |
| `runtime.kyojin.ext_bindings` | The extension's pybind11 module (every op bound, CUDA and ROCm branches) and build_config.py (which sources build on... |
| `runtime.kyojin.ext_fallbacks` | Torch fallbacks for extension ops not built on ROCm (the chains fused_elt and the Triton writers replace) |
| `runtime.kyojin.generator` | The paged generator and job model as Kyojin rewrote them (2,894 + 2,492 lines): continuous batching, chunked... |
| `runtime.kyojin.gm_ext` | The small JIT-built extension beside exllamav3_ext (gm_bindings.cpp binds the gm_* kernels; __init__.py builds or... |
| `runtime.kyojin.graph` | CUDA / HIP graph capture and node patching of the extension (record once, patch kernel arguments, replay) |
| `runtime.kyojin.hip_jit` | One resolver for the hipcc that builds every JIT HIP kernel (EXL3_HIPCC, the rocm-sdk wheel, PATH, /opt/rocm) and... |
| `runtime.kyojin.hy4_graph` | Hy4 decode step as one HIP graph per context bucket, multi-session replay (rows of several sessions in one replay),... |
| `runtime.kyojin.hy4_prefill` | Hy4 prefill kernels in Triton (ROCm gfx11 WMMA through tl.dot): fused dequant + GEMM on stored GGUF bytes, grouped /... |
| `runtime.kyojin.knobs` | Run-time switchable A/B knobs (read from the environment at call time, one writer) and the frozen attention policy |
| `runtime.maya_tp.core_session` | the Strata-lineage engine core: one token through all layers with per-layer graphs (session), the composed GDN layer... |
| `runtime.maya_tp.ds4_engine` | maya-ds4, DeepSeek-V4.1-Flash on Maya (ds4_model.cu, 22,390 lines; ds4_model.hpp, ds4_multi_attn, ds4_logprobs,... |
| `runtime.maya_tp.ds4_main` | the maya-ds4 program (ds4_main.cpp, 4,122 lines): options, the TP2 rank setup, DSpark drafting and the serve loop |
| `runtime.maya_tp.generate` | the generate program (src/program/generate.cpp, 10,336 lines): the strata generate / maya engine process loop that... |
| `runtime.maya_tp.glm_model` | the glm5-next / GLM-5.3-Flash runner (glm_model.cu, 4,206 lines) with several conversations decoded together... |
| `runtime.maya_tp.glm_prefill` | the glm5-next batched prompt path (glm_prefill.cu, 5,273 lines) with its FP16 weight types (glm_prefill_types.hpp) |
| `runtime.maya_tp.glm_tools` | GLM engine side tools: refusal ablation on the GLM residual (glm_ablate.cu / .hpp, ablate_capture.hpp), the CPU... |
| `runtime.maya_tp.mm3_engine` | maya-mm3, MiniMax-M3.1-Flash on Maya: the model with loading, placement and forward (mm3_model), the program... |
| `runtime.vllm_strix.aiter_gfx1151_gate` | unlocks aiter on gfx1151 in vllm/_aiter_ops.py, where the stock check enabled it only for CDNA |
| `runtime.vllm_strix.cgfusion_soft` | makes the DS4 AMD path's CUDA-graph fusion check soft: models/deepseek_v4/amd/rocm.py warns instead of raising when... |
| `runtime.vllm_strix.ds4_gfx1151_gates` | five gfx1151 gates for the DS4 AMD path: the C4A block-flat decode layout enabled for gfx1151 (a note: applied... |
| `runtime.vllm_strix.ds4_mmf32` | DS4 compressor and indexer-compressor scores (torch.mm(hidden, fused_wkv_wgate.weight.T, out_dtype=float32), bf16... |
| `runtime.vllm_strix.ds4_native_port` | the DeepSeek-V4-Flash native port to vLLM 0.31 on gfx1151 as reference files: dspark.py and mtp.py from upstream... |
| `runtime.vllm_strix.triton_ptr_cache` | memoises Triton AMD's pointer to device-pointer lookup: extractPointer() in triton/backends/amd/driver.c called... |
| `tool.ds4.agent` | ds4-agent, the terminal coding agent on the engine API |
| `tool.ds4.cli` | the ds4 CLI: one-shot and interactive chat with a live session |
| `tool.ds4.help` | the shared --help text |
| `tool.ds4.linenoise` | linenoise line editing (BSD-2-Clause) |
| `tool.enginelab.apps_ds41` | the DeepSeek V4.1 programs: batch, deploy, DSpark (rank, replay), expert server, inspect, lockstep, long context,... |
| `tool.enginelab.apps_gemma4` | the Gemma 4 programs: decode, gate, prefill A/B, serve and tokenize: 5 origin files under... |
| `tool.enginelab.apps_glm` | the GLM programs: auto rows, deploy (owners, provider), model facts, prefill main, rank (defaults, receipt, startup,... |
| `tool.enginelab.apps_h3` | the H3 image program: 1 origin files under apps/h3_image_main.cpp |
| `tool.enginelab.apps_hip_decode` | the HIP decode program: decode token entry and main, decode experiment, options and receipt JSON, deferred MoE... |
| `tool.enginelab.apps_hy4` | the Hy4 programs: DFly, rank, reference and rows: 4 origin files under apps/hy4_dfly_main.cpp,... |
| `tool.enginelab.apps_inkling2` | the Inkling 2 Mini programs: CLI, CPU, CUDA, GPU main, HIP, inspect and media: 7 origin files under... |
| `tool.enginelab.apps_minimax` | the MiniMax programs: layer cut, rank and serve: 3 origin files under apps/minimax_layercut.inc,... |
| `tool.enginelab.apps_qwen` | the Qwen programs and headers: Qwen 3.5 decode, pair and spec mains, cache admission and family contract, chain... |
| `tool.enginelab.apps_serve_common` | shared application headers: TP2 resident (launch environment snapshot of every lever, 8-field TP2 agreement and... |
| `tool.enginelab.apps_strix` | the strix front programs: strix, strix-serve, strix-router, strix-bench, model describe, inspection bindings,... |
| `tool.enginelab.cli` | the strix CLI library: cli, detail, doctor, inspection and process: 6 origin files under include/strix, src/cli |
| `tool.maya_tp.engine_probes` | the engine's small measurement programs: strata-device (GPU, plan and arena report), strata-load (expert arena... |

### F16 Memory, residency, expert tiers, Engram I/O and weight prefetch (32)

| module | what it holds |
|---|---|
| `kernel.enginelab.ds41_engram` | engram lookup: the hasher, read-ahead, row prep, row reader and spec (src/engram, include/strix/engram) with its HIP... |
| `kernel.maya_tp.cpu_pool` | the CPU expert pool and its tables: the worker pool protocol (pool), the per-layer expert table (expert_layout),... |
| `kernel.maya_tp.ds4_pcie` | the decode path's PCIe lane on the device (ds4_pcie.cuh / ds4_pcie.hpp): the GPU pulls the experts the CPU lane does... |
| `runtime.ds4.engram` | the DeepSeek V4.1 engram tables: loading and lookup, on disk or resident |
| `runtime.ds4.gpu_args` | the shared parser for --gpu-vram and --gpu-devices |
| `runtime.ds4.linux_memory` | Linux available-memory accounting with CMA pages excluded (they back movable allocations, not pinned GPU pages) |
| `runtime.ds4.ssd` | SSD streaming of routed experts: budgets and reads for models larger than RAM |
| `runtime.ds4.streaming_hotlist` | generated expert hotlists (sorted by hits per weight) that SSD streaming keeps resident: DeepSeek V4 Pro and GLM-5.2 |
| `runtime.enginelab.deploy` | deployment planning (src/deploy, include/strix/deploy): binding, device, footprint, link, memory plan, MiniMax layer... |
| `runtime.enginelab.device` | devices (src/device, include/strix/device): budget, cost identity (CUDA, HIP), declarations, eager code load, gates... |
| `runtime.enginelab.ds41_expert_tier` | DS41 expert tiering: the expert RPC and server, the HIP expert backend, hot placement, host pinning (plan,... |
| `runtime.enginelab.memory` | memory: host memory and upload (backend), host phase timing, pinned host buffers, the memory plan, UMA owner... |
| `runtime.enginelab.model_registry` | model registry and residency: content hash, contract, copy-residency layout, description, family registry, manifest,... |
| `runtime.enginelab.power` | the power sampler: 2 origin files under include/strix, src/power |
| `runtime.enginelab.residency` | residency (src/residency, include/strix/residency): copy source, direct read, expert residency, replacement and the... |
| `runtime.kyojin.engram_io` | Engram n-gram table I/O: a threaded O_DIRECT row gather (plain C ABI, ctypes, no torch), device staging and prefetch... |
| `runtime.kyojin.engram_io_c` | GIL-free parallel row reads for the Engram table (one pread per distinct row into a flat buffer), C |
| `runtime.kyojin.expert_store` | Expert store (expert tiering between host and device with prefetch and protection) and mixed-K expert stacks |
| `runtime.kyojin.guard` | Memory guards: GTT / host-memory admission and the live free-memory floor of a server |
| `runtime.kyojin.host_mem` | Host-memory registration (shared arena, CPU expert offload segment), driver API entry points resolved at run time,... |
| `runtime.kyojin.hy4_engram` | Hy4 Engram: row lookup (E4M3 / E8M0 tables, F8 rows), the GPU Engram (distinct rows preaded, nothing of the hidden... |
| `runtime.kyojin.qwd3_pf` | Weight prefetch into the infinity cache (MALL, 32 MB, ~700 GB/s on re-reads vs 236 GB/s DRAM) during the... |
| `runtime.kyojin.stloader` | Safetensors loader: coalesced reads through pinned ring buffers per worker thread, gaps read through rather than... |
| `runtime.maya_tp.ds4_cpu` | maya-ds4's CPU lane: GGUF experts on the CPU (ds4_gguf_cpu), the i-quant CPU rows (ds4_iq_cpu), EXL3 on the CPU... |
| `runtime.maya_tp.ds4_store` | maya-ds4's weight and state storage: the expert store and pool (ds4_store, ds4_pool), the engram (prompt state) file... |
| `runtime.maya_tp.expert_store` | the expert storage tier: R4's slot storage and residency table (expert_cache), the expert-source adapter with its... |
| `runtime.maya_tp.glm_tiers` | GLM expert memory tiers: the run-time tiers over NUMA / VRAM / host (glm_tier_numa), their host-side rules... |
| `runtime.maya_tp.platform` | platform helpers: direct (O_DIRECT) file reads, memory queries, the memory planner CLI (src/plan) and the platform /... |
| `runtime.vllm_strix.wna16_empty_cache` | a per-layer torch.cuda.empty_cache() in the WNA16 MoE weight conversion, so each layer's transpose staging is... |
| `tool.enginelab.lab_gpu_visibility` | the GPU visibility lab (lab/gpu_visibility): 3 origin files under lab/gpu_visibility |
| `tool.enginelab.lab_qwen_output_split` | the Qwen output split lab: native target, host tests (lab/qwen_output_split): 9 origin files under lab/qwen_output_split |
| `tool.enginelab.lab_uma_residency` | the UMA residency lab (lab/uma_residency): 4 origin files under lab/uma_residency |

### F17 Serving fronts, benches, probes and tooling (35)

| module | what it holds |
|---|---|
| `runtime.kyojin.triton_seed` | Pre-compiled Triton kernels shipped with the code (kernel_seed tarball per Triton version and arch) and a log of... |
| `serve.ciru_link.console` | the embedded operator console and peer agent (internal/ui): a loopback HTTP console on 7749 with a static page... |
| `serve.ds4.server` | ds4-server: OpenAI / Anthropic-compatible local HTTP server; per-connection threads queue jobs to resident session... |
| `serve.ds4.web` | the built-in web UI served by ds4-server |
| `serve.enginelab.ds41_serve` | DS41 serving: serve cache policy, observation and scheduling, the prefix and session caches, the cache lease and the... |
| `serve.enginelab.gemma4_serve` | Gemma 4 serving (with the Unicode licence notice) (src/gemma4/serve): 12 origin files under src/gemma4 |
| `serve.enginelab.glm_serve` | GLM serving: control, import, lead, prefix cache, self-test, vision and the session owner |
| `serve.enginelab.inkling2_serve` | Inkling 2 Mini serving (device, HIP, install, options): 6 origin files under src/inkling2 |
| `serve.enginelab.minimax_serve` | MiniMax serving: API, chat, control, DSpark policy, lead, observe, scheduling, session, tokenizer, paired state,... |
| `serve.enginelab.serve_cohort` | serving scheduler pieces: cohort decode, passes and session, fixed-KV cohort, replicated cohort, batch / job / pair... |
| `serve.enginelab.serve_http` | the HTTP front (include/strix/serve, src/serve): HTTP server and client, the OpenAI-style API and session API, the... |
| `serve.enginelab.serve_replica` | replica routing: the replica router and proxy: 4 origin files under include/strix, src/serve |
| `serve.enginelab.serve_session` | serve sessions and runners: engine, run, serve config, deployment, session runtime and reservation, serve session,... |
| `serve.enginelab.serving_cohort` | serving cohorts and the scheduler: the serving cohort (HIP), greedy backend, prefix cache, shared cohort, greedy... |
| `serve.kyojin.dsv41_serve` | DS4.1 serving model: cache, graph, MoE and ops of the serving path, prefix cache, speculative decoding (DSpark... |
| `serve.kyojin.glm_server` | GLM-5.3-Flash OpenAI-compatible server (aiohttp): chat template, streaming, MTP depth, request log, metrics, health... |
| `serve.kyojin.hy4_server` | Hy4-flash server (GGUF Q3_K, 64K context): engine and multi-session engine, speculative depth gate, draft policy,... |
| `serve.kyojin.qwen_server` | Qwen server (Qwen3.8-Flash-Next and the 27B): the same front with its speculative policy (draft policy mix, MTP /... |
| `serve.kyojin.serve_common` | Pieces the servers share: split flags, answer-phase loop guard, vLLM-style compatibility (token-id prompts,... |
| `serve.maya_tp.responses` | the Responses API layer (serve/responses.py, response_config, response_jobs, response_schema, response_store):... |
| `serve.maya_tp.server` | the Maya HTTP server (serve/server.py and its helpers: frontend, models and model API, model pulls, GPU map,... |
| `serve.maya_tp.tools_mcp` | tool and attachment handling in the server: MCP client (mcp.py), attachments (attach.py) and PDF text extraction... |
| `serve.vllm_strix.launcher` | the single entry point: vllm-strix-halo.sh <glm53/ds4> start/stop/status/logs over both boxes, the flat... |
| `serve.vllm_strix.manual_serve` | the serve and lifecycle scripts: the GLM and DS4 vllm serve launchers run inside the container (every vLLM flag and... |
| `tool.ds4.bench` | the throughput benchmark: prefill at context frontiers and fixed greedy decode, snapshot and restore outside the timing |
| `tool.ds4.dir_steering` | directional steering tools: build a direction from contrast prompts and sweep it |
| `tool.ds4.eval` | ds4-eval: built-in benchmark integration test (GPQA Diamond, SuperGPQA, AIME 2025, COMPSEC; hard suite) with its cases |
| `tool.maya_tp.hip_tests` | the HIP-specific tests and benches (tests/hip): expert-cache staging, expert kernel bench, GLM attention / mv /... |
| `tool.maya_tp.kernel_parity` | the kernel parity programs and tests (src/kernels/*_parity.cpp, *_test.cpp, qsa_select_bench.cpp and the oracle... |
| `tool.vllm_strix.container_ops` | image and host plumbing: the two-stage Dockerfile (OdinLink userspace build stage, then vLLM rebuilt at 73859fec on... |
| `tool.vllm_strix.debug_hooks` | debug-only patches kept by the origin: dumps of GLM indexer gather / plan / kpool / map / score / top-k (whole-file... |
| `tool.vllm_strix.decode_bench` | the rig's decode benches: vsh_bench (decode / TTFT with speculative-decoding receipts), vsh_ab (matched-length A/B... |
| `tool.vllm_strix.ds4_probes` | DeepSeek-V4 bring-up probes: the Phase-0 battery, decode speed and coherence (greedy and server-default sampling),... |
| `tool.vllm_strix.quality_probes` | quality A/B across builds: mean per-token NLL of fixed texts via prompt_logprobs (with the three texts), greedy... |
| `tool.vllm_strix.trace_analysis` | torch-profiler trace analysis: device to host copies (pipeline drains), where a decode step's wall time goes,... |

### F18 Non-ROCm backends (CUDA, Metal, Vulkan, Intel, RTX routes, NCCL plugins) (28)

| module | what it holds |
|---|---|
| `executor.enginelab.cuda_ds41` | DeepSeek V4.1 on CUDA (src/cuda/models/ds41, include/strix/cuda/ds41_*): 56 origin files under include/strix, src/cuda |
| `executor.enginelab.cuda_glm` | GLM on CUDA (with its producer and bench): 84 origin files under src/cuda |
| `executor.enginelab.cuda_minimax` | MiniMax on CUDA: 17 origin files under src/cuda |
| `executor.enginelab.cuda_qwen` | Qwen on CUDA (src/cuda/models/qwen, include/strix/cuda/qwen_*, src/qwen35/cuda): 40 origin files under... |
| `executor.enginelab.inkling2_cuda` | Inkling 2 Mini CUDA pieces (attention, block candidates, controller operands, dense, MoE, program, serve) |
| `kernel.ds4.cuda_backend` | ds4's CUDA backend (one translation unit, about 36,000 lines): the ds4_gpu API on CUDA with multi-GPU, graph... |
| `kernel.ds4.qwen4_cuda` | Qwen3.8 Flash Next on CUDA, included by ds4_cuda.cu: GDN, attention, HC, indexer, router, MoE, matvec / matrix... |
| `kernel.enginelab.cuda_attn` | CUDA attention: decode, flash (8-bit, split decode), lanes, production, row-group launch and union, selected blocks,... |
| `kernel.enginelab.cuda_core_common` | the CUDA core rest: AMD math tables, common, reduce, PTX, norms, chain, step inputs / rows / row store, trellis,... |
| `kernel.enginelab.cuda_dense` | CUDA dense: dense, IMMA dense and tiles, MMA tiles, native MMA and reuse: 5 origin files under src/cuda |
| `kernel.enginelab.cuda_head` | CUDA heads: argmax (final, partial), emit and screened head: 5 origin files under src/cuda |
| `kernel.enginelab.cuda_mix` | CUDA hyper-connection mix: body, latency plan, transition: 5 origin files under src/cuda |
| `kernel.enginelab.cuda_moe` | CUDA MoE: A8 (MMA), microscaled R1 (step, pipe), R1 route plan, turbo I8, route project / select / softmax, grouped W4A4 |
| `kernel.enginelab.cuda_nvfp4` | CUDA NVFP4: NVFP4 and NVFP4 x A8 with references and tiles: 5 origin files under src/cuda |
| `kernel.enginelab.cuda_recur` | CUDA recurrence: chunk, decode, lanes, math, norm, parity and plan: 9 origin files under src/cuda |
| `kernel.enginelab.cuda_select` | CUDA selection: catalog, pick body, ReLU schedules, rows, score and segments: 8 origin files under src/cuda |
| `kernel.kyojin.qdec_cuda` | CUDA twin of qdec (sm_80+, NVRTC at first use): tensor-core QK^T / PV from packed bit planes, plus qdec_rows.cu for... |
| `kernel.kyojin.triton_paged_cu` | CUDA-only paged decode attention (flash-decoding split + combine) ported from upstream exllamav3 1.6.0 |
| `runtime.enginelab.backend_intel` | the Intel GPU backend (src/intel: arch, core, weight): 15 origin files under src/intel |
| `runtime.enginelab.backend_metal` | the Metal backend core (src/metal/core): 4 origin files under src/metal |
| `runtime.enginelab.backend_vulkan` | the Vulkan backend core (src/vulkan/core): 4 origin files under src/vulkan |
| `runtime.enginelab.cuda_exec` | the CUDA exec layer: catalog decode, step rows and WR1 / WR2, bound transfers, decode epilogue, lowering and... |
| `runtime.enginelab.ds41_rtx` | the optional RTX PRO route for DS41: RTX budget, drafter select, profile and schedule, and RTX state import |
| `runtime.enginelab.glm_rtxpf` | GLM RTX prefill route: 4 origin files under src/glm |
| `runtime.enginelab.minimax_rtxpf` | MiniMax RTX FP32 identity (device, reference, tests): 5 origin files under src/minimax |
| `runtime.kyojin.cuda_graph` | Python graph capture helpers and host pointer tables of the model |
| `tool.enginelab.apps_ds41_cuda` | the CUDA DeepSeek V4.1 programs: decode, hot decode and replay, multi front, parts host check, pipeline front,... |
| `tool.enginelab.apps_qwen_cuda` | the CUDA Qwen programs: Qwen 3.5 cohort, decode and spec, logits, prefill service, prefix cache, serve and policy |

### F19 Vision towers and image I/O (20)

| module | what it holds |
|---|---|
| `executor.enginelab.ds41_vision` | DS41 vision: the vision adapter and splice: 3 origin files under src/deepseek41 |
| `executor.enginelab.glm_vision` | GLM vision: 2 origin files under src/glm |
| `executor.enginelab.vision` | the shared vision tower (src/vision, include/strix/vision): API media, decode, embed cache and spans, encoder (CUDA,... |
| `executor.kyojin.dsv41_vision` | DS4.1 vision tower (image patches to embeddings for the DS4.1 trunk) and its tensor spec |
| `kernel.ds4.dsv4_vision` | DeepSeek V4 Flash Vision-Exp operations shared by CUDA and ROCm: residual add, aligner reorder, GELU bias, QKV RoPE,... |
| `kernel.ds4.glm53_vision` | GLM-5.3 vision tower operations shared by CUDA and ROCm: attention, bias, layernorm + GELU, QKV RoPE, reorder, RMS,... |
| `kernel.ds4.k53flash_glm53_vision` | [k53flash overlay] glm53_vision family: ds4_glm53_vision_gpu.cuh changed (+0 -1 lines). The family: GLM-5.3 vision... |
| `kernel.ds4.k53perf_dsv4_vision` | [k53perf overlay] dsv4_vision family: ds4_deepseek4_vision_gpu.cuh changed (+141 -13 lines). Kernels not in the... |
| `kernel.ds4.k53perf_glm53_vision` | [k53perf overlay] glm53_vision family: ds4_glm53_vision_gpu.cuh changed (+58 -4 lines). Kernels not in the base:... |
| `kernel.ds4.k53prefill_glm53_vision` | [k53prefill overlay] glm53_vision family: ds4_glm53_vision_gpu.cuh changed (+0 -1 lines). The family: GLM-5.3 vision... |
| `kernel.ds4.rocm_deepseek4_vision` | ROCm kernels of the DeepSeek V4 vision path: visual mixed-KV pack and softmax attention and the parallel visual... |
| `kernel.enginelab.h3_vae` | H3 image VAE (src/h3/vae): 30 origin files under src/h3 |
| `kernel.maya_tp.ds4_vision` | DeepSeek-V4.1's vision tower on one GPU (ds4_vision.cu / .cuh, DeepSeek's inference/vision.py) |
| `kernel.maya_tp.mm3_vision` | MiniMax-M3.1-Flash's vision tower (mm3_vision.cu / .hpp) |
| `runtime.ds4.image` | image decoding and preprocessing for the vision models |
| `runtime.ds4.iris` | iris: single-header JPEG and PNG decoders (MIT, its LICENSE beside them) |
| `runtime.ds4.qwen4_vision` | the Qwen3.8 vision types and entry points shared by the engine and the GPU backends |
| `runtime.enginelab.stb_image` | stb_image.h (Sean Barrett; MIT or public domain), vendored: 1 origin files under third_party/stb |
| `serve.maya_tp.vision_front` | the server side of image input for DeepSeek-V4.1 and MiniMax-M3.1 (serve/ds4_vision.py, serve/mm3_vision.py) |
| `tool.ds4.gguf_vision` | vision tower extraction scripts for DeepSeek V4, V4.1 and GLM-5.3 |

