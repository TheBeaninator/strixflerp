# Prefill of DeepSeek-V4.1-Flash on StrixFold TP2: where the time goes and what could make it faster

Recipe `dsv41-flash.strixfold-tp2` (fork f64b5ff, two ranks, the pod serves it on F and A). Paths without a root are
under `catalog/strixfold/`; `X/` is `executor/dsv41/src/zig/src/families/dsv41/`, `K/` is `kernel/`. "Measured" means
a number with a file behind it; everything else is marked inferred. Written 2026-11-11.

## 1. The prefill path

**Chunks.** A prompt runs in chunks of 2,048 rows that start at multiples of 2,048 (`X/model.zig:49-50`, a constant;
`CHUNK_ROWS` in the conf is recorded, not read). With `TF_DS_PREFILL_SPLIT=256` (the pod's setting, kept above 16K
context by `PREFILL_SPLIT_LARGE=1`) a chunk whose first half would hold 256 rows or more runs as two halves of up to
1,024 rows "zipped" layer by layer (`X/model.zig:831-845`, `fillSplit` `:1003-1146`); otherwise as one
(`fillWhole` `:862-974`). Row counts decide the kernels: more than 128 rows take the prompt GEMM, 1 to 128 the grouped
decode linears (`X/prompt.zig:784-796`); 64 rows or more take the prompt experts (`:1157-1161`); more than 16 rows
the one-pass attention and the `gate_logits` router input (`:826`, `:1125-1136`).

**Bounded replay.** Only the encoder half (layers 0-19) runs every row. At layer 20 (`layers / 2`) a chunk is cut to
its rows at or past `prompt length - window` (`replayCut`, `X/model.zig:900-912`, `X/prompt.zig:1631`), so the
decoder half (20-39) and the head run at most the last 128 rows of the prompt, once per prompt. A long prompt's cost
is therefore 20 layers per row, plus a fixed tail.

**One encoder layer on one rank, n rows** (`X/model.zig:938-955`, in order):

| step | kernels (module) | weights read | dtypes |
|---|---|---|---|
| attention mixes | `aux.hc_mix_pf` (`kernel.strixfold.mhc_mix`) posts the gathered MoE partial of the layer before into the 4 streams and makes 24 mixes; Triton `_hc_finish` (Sinkhorn 20 iterations, collapse, RMSNorm) (`X/tri_hc.zig:202-204`) | `hc_attn` fp32 [24, 20480] | streams bf16 [n, 4, 5120]; partials fp32 |
| attention in | `exl3.rot_in` + `exl3.gemm` (`kernel.strixfold.exl3`) for `wq_a` 5120 to 1280 and `wkv` 5120 to 512 (`X/prompt.zig:828-830`); Triton RMSNorm; `wq_b` 1280 to 16384 (this rank's 32 heads of 512) (`:834-835`); `aux.rope_heads`; Triton `_kv_norm_rope` (`:859`) | EXL3 trellis, decoded inside the GEMM per (256- or 128-row tile, 128-column strip) (`X/exl3_prefill.zig:117-146`) | x bf16, rotated rows fp16, fp32 accumulate, bf16 out |
| compressed layers (ratio 2: layers 2-19) | at a kv source (2, 8, 14): `comp_wkv`, `comp_wgate` to fp32, `aux.compress2`, RMSNorm, RoPE, FP4 store of latents and index keys (`:946-1001`). At an index source (2, 8, 14): `idx_wq_b` 1280 to 4096, RoPE, FP4 queries, `aux.gate_logits` (ne 32), `aux.index_rows` over every compressed key the rows can see, then the top-512 selection (`:1015-1085`); the other compressed layers reuse that selection | FP4 E2M1 key cache | bf16 queries, fp32 scores as int64 keys |
| sparse attention | `aux.sparse_attn` one pass (`kernel.strixfold.sparse_attn`): 128 window keys (bf16) + 512 picked latents (FP4), 32 heads of 512, sink (`:885-904`, `X/tri_attn.zig:76-116`) | none (sink vector) | bf16 q and keys, fp32 accumulate, bf16 out |
| attention out | `aux.rope_heads` inverse; `wo_a` 4 groups of 4096 to 1024; `wo_b` 4096 to 5120 into fp32 (`:905-939`) | EXL3 | partial fp32 [n, 5120] |
| **gather** | all-gather of the attention partial: n x 5120 x 4 bytes a rank (`:1099-1102`) | | fp32 |
| FFN mixes | as the attention mixes, posting the gathered attention partial (`:1106-1111`) | `hc_ffn` | |
| MoE | `aux.gate_logits` (5120 to 384, fp16 weight) and Triton `_route` (top 6 + the shared expert in the last slot) (`:1125-1136`); `exl3.pf_experts.*` (`kernel.strixfold.exl3_pf_experts`): count, place, rot, gate/up, down, combine (`X/exl3_experts.zig:202-252`) | a rank's table: 385 experts x (5120 to 1152, twice, and 1152 to 5120), EXL3 at 2 to 3 bits; **each (expert, up to 64 rows) job decodes that expert's three matrices once** (`K/exl3_pf_experts/src/zig/kernels/hip/dsv41/exl3_pf.hip:5-12`) | x bf16, fp16 WMMA, fp32 accumulate, partial fp32 |
| **gather** | all-gather of the MoE partial, n x 5120 x 4 bytes | | fp32 |

Engram layers (1 and 14) add a host step (n-gram hashing, O_DIRECT row reads, on a read-ahead thread:
`X/model.zig:879-937`), one GEMM (3072 to 25600) and their own gathers (n x 25600 x 4 bytes, in two column halves on
HIP). The head gathers one row of logits once per prompt.

**What crosses the wire.** Per encoder layer two fp32 gathers of n x 5120 x 4 bytes each way: 20.5 MiB a 1,024-row
half, 41 MiB a whole chunk. Per 2,048-row chunk 40 x 41 MiB + 2 x 200 MiB = 2.0 GiB a rank a direction, 1.0 MB a
prompt token (origin `tools/dsv41-hip/COLLECTIVES-DESIGN.md` section 5.3). The slot collective (`tcp_proxy`, and
`rdma_proxy` under `TRANSPORT=rdma`) carries gathers up to one slot, 4,352 KiB; **everything larger goes through
`tcp_host`**, so prefill is on host-staged TCP whatever the transport (`runtime/coll_rdma/src/zig/src/families/dsv41/rdma_proxy.zig:5`,
`:88-91`; the served log line "larger gathers through tcp_host"). `tcp_host` copies the rank's block to pinned host
memory with `hipMemcpyDtoHAsync` in 4 MiB pieces, exchanges each over one TCP socket and uploads the peer's block
(`runtime/coll_tcp/src/zig/src/families/dsv41/tcp_coll.zig:271-321`). Unsplit, a gather blocks the host in stream
order. Split, each half's gather is begun after its step and ended after the other half's next step
(`Zip.next`, `X/model.zig:1037-1043`): one gather in flight, moving while the GPU runs the other half.

**Prompt cache.** States are kept only at chunk ends (multiples of 2,048) and restore only into a prompt pass
(`X/lanes.zig:436`, `:751-752`; `runtime/prompt_cache`). A hit resumes at the last kept chunk end; a learned state on
disk (`LEARN_DIR`) does the same across restarts.

**Settings that reach prefill.** `TF_DS_PREFILL_SPLIT` / `_ZIP` (`X/model.zig:261-262`); `TF_DS_AUX` (HIP kernels or
Triton, `X/tri_aux.zig:1-15`); `TF_COLL_PIPE_MIB` / `TF_COLL_CHUNK_MIB` / `TF_COLL_STAGING` / `TF_TP_STRIPE`
(`tcp_coll.zig:10-25`); `TF_COLL_SLOT_KIB`; `ENGRAM_AIO`; `PROMPT_CACHE_GIB`, `LEARN_DIR`. Not settings: the chunk
rows, the 64/128-row expert job size (chosen by picks per expert, `X/exl3_experts.zig:211`), the GEMM tile
(`X/exl3_prefill.zig:121-124`).

**A profiling switch exists and has never been run on the pod.** `tf-dsv41-lanes --profile 2` (installed by the same
build step as the server) times each prompt chunk on the GPU by phase: start, embed, engram, mix_attn, attn,
gather_a, mix_ffn, moe, gather_m, head, and finer marks inside attention and MoE (`X/model.zig:871-972`,
`X/round.zig:858-860`, `X/lanes_main.zig:131`, `:153`, `:167-171`). It is in the ranks' handshake and it runs a split
chunk's halves one after the other. No output of it is in `gate/results` or under `/data/pod`.

## 2. Measurements that already existed

All end to end (TTFT or tokens per second). No breakdown of a prompt chunk by stage or kernel was on record anywhere.

| what | number | where |
|---|---|---|
| 1,818-token prompt (one chunk, tail included) | 1,205 to 1,241 tok/s, TTFT 1.48 to 1.51 s, the same under TCP, RDMA, striping, drafts on or off, 1 or 4 streams, two ROCm builds | pod `/data/pod/reports/2026-11/strixfold.md:138-142`, `strixfold-rdma.md:100,226`, `strixfold-drafts.md:67,202`, `strixfold-rocm.md:33`, `strixfold-slots.md:122`; today `/data/pod/plans/4af0e1079a/run-{1-tcp,2-rdma}/rank0-run/bench.json` |
| about 8K tokens | 1,424 to 1,452 tok/s (TTFT 5.4 to 5.6 s); one older build 1,622 | `strixfold-drafts.md:78,198`, `strixfold-rdma.md:101-102,127`, `/data/pod/plans/779c7ca147/probe-8k-arm2.jsonl` |
| 129,018 tokens, 128K context, 1 stream, 2026-11-06 | TTFT 89.6 s, 1,440 tok/s | `/data/pod/plans/779c7ca147/probe-128k.jsonl`, `strixfold.md:148` |
| 65,391 tokens cold, the current serving conf (4 streams, 524,288 context) | TTFT 62.1 s, 1,055 tok/s | `/data/pod/plans/d74bd09137/bench-20261111.txt`, card d74bd09137 |
| prompt cache, learned states | 8,415 tokens resumed at 8,192: TTFT 0.54 s against 6.29 s; a learned 63,488-token prefix: TTFT 2.2 to 3.5 s | `sff-cache-on.md:180-278`, card d74bd09137 |
| the split, on the origin's USB4 pair | +13 to +21 % at 4K and 8K; +34 % at 63K context; prompt logits SHA equal split against whole | origin `COLLECTIVES-DESIGN.md` 5.3, `serve.conf` `PREFILL_SPLIT_LARGE` |
| transport alone, origin's USB4 pair | 512 to 810 ms per 2,048-row chunk | origin `COLLECTIVES-DESIGN.md` 5.3 |
| this recipe's gate | 3,176-token request with 146 new tokens: 5.4 s wall (no TTFT recorded) | `gate/results/dsv41-flash.strixfold-tp2/summary.json` |

Already tried on the pod for prefill and flat: RDMA for the slot collective (prefill gathers never used it), a second
proxy thread and striping (SPEED-STRIPE 49040a0ad9), drafts (SPEED-DRAFTS 9dd37dcd97), 4 streams (SPEED-SLOTS
36c2c1f5e0), the ROCm build. Open on the board: SFR-5 b8212e7bbc (its design note re-aims it at prefill gathers over
RDMA), SPEED-PARITY 705f856f99, SPEED-FUSE-1 b7269694f8 (the mHC chain as one launch). Never done on the pod: the split
on against off, a chunk-size comparison, the phase timer.

In-source timings by the origin's authors, which the bench below reproduces: sparse attention 4.9 ms at 2,048 rows
with 512 picks (`K/sparse_attn/src/zig/kernels/hip/dsv41/aux_attn.hip:24-25`; here 4.92 and 5.20), mHC mixes 0.66 ms
at 1,024 rows (`K/mhc_mix/src/zig/kernels/hip/dsv41/aux_hc.hip:22-23`; here 0.77).

## 3. The time budget (measured on one GPU, kernels alone)

`gate/prefill_dsv41_bench.py` loads the kit the recipe's build stage produced (manifest sha256 `76936479...`, equal
to the served deployment's) and launches each prefill kernel by its symbol with the argument structs of
`kernel.strixfold.dsv41_abi`, on random weights and activations of a rank's shapes. Run on control-node,
2026-11-11T18:26Z, 3 minutes under the control-gpu lease: `gate/results/prefill-dsv41/bench.json` (95 timings, 6
checks); `rerun-outliers/` repeats five cases whose first timing was disturbed. Milliseconds a launch:

| kernel, one rank | 512 rows | 1,024 rows | 2,048 rows | 4,096 rows | rate at 1,024 |
|---|---|---|---|---|---|
| routed experts, whole pass (385 experts at 3 bits; uniform routing) | 20.8 | 24.7 | 35.6 | 60.6 | 10.3 TFLOP/s |
| the same, skewed routing (a third of the experts take two thirds of the picks) | 20.2 | 25.1 | 38.1 | 62.4 | |
| the same at 2 bits, uniform | 18.6 | 23.3 | 34.2 | 61.3 | |
| `wq_a` 5120 to 1280 | 0.31 | 0.54 | 1.18 | 2.42 | 27.6 |
| `wkv` 5120 to 512 | 0.26 | 0.33 | 0.66 | 1.45 | 19.7 |
| `wq_b` 1280 to 16384 | 0.94 | 1.76 | 3.47 | 7.09 | 24.9 |
| `wo_a`, 4 groups of 4096 to 1024 | 0.94 | 1.74 | 3.37 | 8.73 | 22.8 |
| `wo_b` 4096 to 5120 | 0.81 | 1.73 | 4.14 | 8.48 | 26.0 |
| sparse attention, 128 window keys + 512 picks | 1.39 | 2.62 | 4.92 | 9.79 | 16.4 |
| sparse attention, window only (layers 0, 1) | 0.33 | 0.69 | 1.18 | 2.49 | 12.5 |
| mHC posted streams + mixes (twice a layer) | 0.43 | 0.77 | 1.46 | 3.00 | 163 GB/s |
| router gate logits (384) | 0.27 | 0.37 | 0.76 | 1.71 | 11.0 |
| RoPE on 32 heads of 512 (twice a layer) | 0.02 | 0.06 | 0.16 | 0.80 | |
| indexer scores over 16,384 / 32,768 / 65,536 keys (index sources only) | 2.5 / 4.9 / 10.1 | 5.6 / 9.9 / 20.0 | 10.8 / 19.6 / 39.9 | 23.8 / 41.5 / 77.7 | 27 |
| `engram_wkv` column half (Engram layers only) | 1.62 | 3.12 | 6.22 | 12.8 | 26.2 |

One encoder layer on a 1,024-row half, summed from that table: **35.4 ms, of which the routed experts are 24.7 ms
(70 %)**, the five EXL3 projections 6.1 ms (17 %), attention 2.6 ms (7 %), the mHC mixes 1.5 ms (4 %), gate logits and
RoPE 0.5 ms. A 2,048-row chunk as served (two halves, 20 encoder layers, the kv-source, index-source and Engram
extras at position 32K): **1.46 s of kernel time a rank, 0.99 s of it in the experts.** That is 1,400 tokens per
second if nothing else cost anything.

Not in the sum (not measured: they are Triton-compiled or host steps): `_hc_finish` twice a layer, two RMSNorms,
`_kv_norm_rope`, `_route`, the FP4 stores, the top-512 selection at the three index sources, the Engram host reads,
launch gaps, and every collective. The decoder tail (layers 20 to 39 on at most 128 rows, the grouped decode kernels)
was not measured either; the 1,818-token case (1.49 s for one chunk, where the table gives about 1.26 s for the
encoder rows) suggests about 0.2 s a prompt for it (inferred).

**How this relates to the pair.** The pair's chunk takes 1.42 s at 1,440 tok/s (2026-11-06) and 1.94 s at 1,055 tok/s
(today's conf). The kernel sum from another machine of the same model, 1.46 s, is the size of the pair's whole chunk
time. So the pair's prefill is mostly GPU kernel time and the zip already hides most of the gathers; that reading
is consistent with every transport experiment on the pod leaving prefill flat. It is an inference: control-node and
the workers do not run the same power setting, a CPU-bound job was running on control-node during the bench (load
average 22 to 30; the memory-bound kernels may read low), and nothing here saw a collective. The split between
kernel time, unhidden gather time and host time on the pair needs the phase timer (section 6).

**The finding in the table.** The experts' pass costs 20.8 ms at 512 rows and 60.6 ms at 4,096: about 15 ms a pass
whatever the row count, then about 11 ms per 1,024 rows. The fixed part is the decode: every (expert, up to 64 or 128
rows) job decodes that expert's three matrices once, and from 512 rows up a pass touches every one of the 385 experts
(392 to 471 jobs in every case). Per phase at 1,024 rows: gate/up 14.3 ms, down 6.9, rotation 0.9, combine 0.8,
grouping 0.06. The split runs the experts once per half (`prompt.moe` inside the per-half loop, `X/model.zig:1111-1114`),
so a 2,048-row chunk pays the decode twice in each of 20 layers: 2 x 24.7 = 49.3 ms where one pass over the same
2,048 rows is 35.6 ms.

## 4. The other engines on the same operations

| StrixFold prefill kernel (interface) | gufo DS4 | hipfire DS4 | strixite (Qwen) | drop-in? / shape to transfer |
|---|---|---|---|---|
| chunk rows | 4,096 a chunk, whole prompt up to 4,096 (`catalog/gufo/executor/deepseek_v4_flash/src/src/models/deepseek_v4_flash/runtime/rocm_graph.cpp:42-56`) | 1,024 (`HIPFIRE_DEEPSEEK4_PP_BATCH`, `catalog/hipfire/executor/arch_deepseek4/src/crates/hipfire-arch-deepseek4/src/carrier.rs:151`) | 16,384, or 2,048 while other slots decode | a schedule, not a kernel |
| routed experts (`moe.experts-prefill/exl3/bf16.jobs`) | `kernel.gufo.ds4_moe`: rows grouped by expert in groups of 128 then 64, IQ2 gate/up and Q2_K down on WMMA | `kernel.hipfire.moe`: grouped GEMM from 128 rows, MQ2 Lloyd weights, an i8-WMMA route | `kernel.strixite.moe_grouped`, strixw Q4/Q5 | not a drop-in; strixite's single-launch tile classes onto the EXL3 pass (R2). All four group rows by expert |
| prompt GEMM (`linear.prefill/exl3/f16-f32`) | `ds4_linear`: Q8_0 dequantised once to an F16 mirror, hipBLASLt or its own WMMA GEMM | `gemm_mfp4` / `gemm_mq` / `gemm_hfq` | `linear_wmma`, 64-row tiles | no (`docs/INTERFACES.md:110-114`). StrixFold's already runs at 20 to 27 TFLOP/s |
| indexer scores (`attn.indexer-score/kv-fp4-e2m1/bf16`) | `ds4_indexer`: f32 keys, 64 heads, 128-key tiles | `indexer_topk`: f32 or f16 keys | - | no: key storage and head count differ (`docs/INTERFACES.md:165-167`) |
| top-k selection (`attn.indexer-topk/-/i64-keys`) | exact MSD radix on f32 scores | bounded exact merge | - | with a `score-keys` adapter, judged, not gated (`docs/INTERFACES.md:168-171`); the selection's cost here is unmeasured |
| sparse attention (`attn.sparse-sink/kv-fp4-e2m1/bf16`) | `ds4_attention`: f32 q, f32 or f16 keys, 16-row tiles | `ds4_sparse_attn`: f32, direct or gathered | - | no: FP4 keys and bf16 activations exist only in StrixFold |
| mHC mixes (`hc.mix-prefill/dsv41-mhc-f32/bf16`) | `ds4_hc`, 4 x 4096, fused norm + mix + split from 128 rows | `hc`, five batched kernels | `hc_wmma` | no: 5120 against 4096, weights read (`docs/INTERFACES.md:198-200`) |
| router (`moe.router/-/bf16-f32`), RMSNorm, RoPE, residual | all fp32 | all fp32 | bf16 or fp32 | router and RoPE judged not swappable (TP split, Triton bits); residual add is folded into the mixes here. Together under 2 % of the budget |

Nothing in the 70 % is a drop-in part; the shapes worth transferring are in `docs/SHAPE-TRANSFERS.md` (R2, R3). The one design difference that matters is in the first row: gufo runs the same model
family in 4,096-row chunks, and StrixFold's own expert kernel is 39 % cheaper per row at 4,096 rows than at 1,024.

## 5. Candidates, ranked

Gains are of kernel time per 2,048 prompt rows (1.46 s in section 3) unless said otherwise; a gain on the pair is the
same fraction only as far as the pair is kernel-bound (section 3, inferred).

**1. One experts pass per 2,048 rows or more, with the zip kept: zip two whole chunks.** New executor code.
Mechanism: `fillSplit` on halves of 2,048 rows (a 4,096-row unit) instead of 1,024. Arithmetic: experts 35.6 ms
instead of 2 x 24.7 per layer per 2,048 rows, x 20 layers = **-0.27 s of 1.46 s (-19 %, +23 % tokens per second)**;
every other kernel is linear in rows (table). Measured at kernel level, both halves of the claim:
`gate/results/prefill-dsv41/bench.json` `checks` (rows as one pass against the same rows as two passes: 0 of
10,485,760 fp32 outputs differ, uniform and skewed routing, 3-bit and 2-bit tables, also at 256 rows) and the
timings above. Token identity: every prefill kernel is written row-invariant and the origin measured split against
whole with equal prompt-logits SHA-256, so the gate's criterion (ids and prompt-logits sha equal to the reference)
stays as it is and should hold; it has to be run. Size: `chunk_rows` is one constant but it is also the grid of kept
states, so the honest form keeps 2,048-row chunks as the unit of state and zips chunk k with chunk k + 1: a state at
the first chunk's end then needs that chunk's window rings saved before the second overwrites them (144 x 512 bf16
a layer). Costs: a second set of 2,048-row buffers a rank (the split's own cost was +0.62 to 0.68 GiB a rank; inferred
about the same again, against `PEAK_GIB=106` of 110), and learned states under `/cache/tfp-learned` must keep
restoring at 2,048-row ends (63,488 is not a multiple of 4,096). A derived module of `executor.strixfold.dsv41`
(adapted `model.zig`), not a kernel. Not written: it cannot run without the pair.

**2. `TF_DS_PREFILL_SPLIT` off. A setting.** The same -0.27 s of kernel time (one pass per chunk; kernel sum 1.15 s
against 1.42 for 20 ordinary layers), paid for with every gather back on the critical path: 2.0 GiB each way per
chunk through `tcp_host`. On the origin's USB4 pair that was 0.6 to 0.8 s and the split won; on the rail it has
never been measured. If a chunk's gathers cost less than 0.27 s there, off is faster today with no code. Bit-exact
by the origin's measurement. This is the cheapest decisive test and also measures what candidate 3 can be worth.

**3. Prefill gathers off host-staged TCP (the board's SFR-5).** Owned by the other session. Worth at most the
gather time the zip does not hide, which nobody has measured; section 3 suggests it is small today. It becomes
valuable together with 2 or 4.

**4. One experts pass for both halves inside today's chunk.** New executor code, smaller than 1: route each half,
run `exl3.pf_experts` once over the 2,048 rows, keep the attention zip. Same -0.27 s; but the MoE gathers of both
halves and the second half's attention gather then have nothing to overlap (three of four gathers a layer, 20.5 MiB
each way), so it only pays if a serial 20 MiB gather costs under about 4.5 ms. Decided by the same measurement as 2.

**5. Larger units still (4,096 rows a pass).** 30.3 ms per 2,048 rows against 49.3: -0.38 s (-26 %). Same code as 1
with twice the buffers again; only if memory allows.

**6. The indexer at long context.** `aux.index_rows` is linear in the keys a row sees: 9.9 ms a half at position
64K, 20 ms at 128K, at the three index sources, so 4 % of a chunk at 64K and 8 % at 128K, more beyond; plus the
selection, unmeasured. The kernel runs at 27 TFLOP/s, so the lever is fewer keys (the candidate pool of layer 20
onward applied earlier), which changes the model's arithmetic and so the gate's criterion: not proposed.

**7. The mHC chain as one launch (SPEED-FUSE-1).** The mixes are 1.5 ms of a 35.4 ms layer-half at 163 GB/s; with
`_hc_finish` (unmeasured) the chain is perhaps 6 % of prefill. A decode gain, not a prefill one.

**8. Prompt GEMMs, attention, router.** 17 %, 7 % and 1 %, at 16 to 27 TFLOP/s. No candidate.

Not a candidate by the owner's ruling: the workers' power mode. It is the likeliest reason the pair went from 1,440
to 1,055 tok/s between 2026-11-06 and today (inferred from dates only; the conf also changed).

## 6. The pair window: 15 minutes, three boots

Drafts stay on. Never edit `/data/pod/plans/strixfold/tp2-128k.conf`: copy it to a scratch conf with
`RUN_ROOT` of its own, `LEARN_DIR=` (empty, so `/cache/tfp-learned` is not opened), `PROMPT_CACHE_GIB=0`,
`LOGITS_SHA=1`, everything else as served. Tools are the deployment's own (`~/tfp/deploy/f64b5ff2e28b/tools/dsv41-hip`).
One prompt for all arms: 8,192 token ids (four full chunks, no partial one) plus a short question; `max_tokens 4`.

| arm | change | run | record |
|---|---|---|---|
| A | none | `qual/qual_bench.sh bench CONF A --reps 3 --prefill 8000` (origin `BUILD_AND_RUN.md:115`) | TTFT, prompt tokens, the `prompt_logits` sha line, rank 0's "prompt pass" line (host gather counts) |
| B | `PREFILL_SPLIT=` (empty) | the same | the same; the sha must equal arm A's |
| C | arm A's conf through `tf-dsv41-lanes ... --profile 2` on both ranks (`qual/lane_rank.sh CONF 0 DIR --requests FILE --parallel 1 --profile 2` and `... 1 DIR --profile 2`; `lane_gate.sh` passes extra arguments to rank 0 only, and the phase timer is in the handshake, so both need it) | one request | each rank's `prefill_phases_ms` line and `chunks` |

Also record on both workers before and after: power profile, GPU clock, load average. About 90 s of load and under
30 s of requests per boot.

Reading it: arm C gives the per-stage budget on the real pair, gathers included (it runs the halves one after the
other, so `gather_a + gather_m` is the serial gather cost G of a chunk, and `moe` should be near 70 % of the rest
if section 3 holds there). Arm B against A gives the same G from the other side: B - A = G_unhidden - 0.27 s scaled
to the workers. If B is faster, switch the split off and candidates 3 and 4 are next. If B is slower by more than
about 0.3 s a chunk, gathers are expensive, the zip stays, and candidate 1 is the one to build. If `moe` is not the
largest phase in arm C, section 3 does not transfer and the ranking is wrong.

## 7. Did the modular layout help?

Where it did: the kit is a manifest of named symbols with one argument struct each, and the ABI header is its own
module, so a 400-line driver with no engine behind it ran the served code objects alone at served shapes and
produced the first per-kernel prefill budget for this engine in three minutes of one GPU, without the pair and
without a model. The origin has the pieces for the same (its Triton and EXL3 harnesses) but they need its work area
and are not what the pod has; what the pod had after a week of tuning was end-to-end numbers only. And
`flerp interfaces` answered "is there a better part elsewhere" in one command per operation: no, for every kernel
in the 70 %.

Where it did not: the gain found is not a part. It is how many rows share one launch, decided in `model.zig`
inside `executor.strixfold.dsv41`, which the catalogue holds as one Zig import closure with no seam at the chunk
schedule. The hypothesis "see where the time goes" held; "swap in a better part" did not: nothing in the 70 % is a drop-in
part, and the hot kernel's cost is set by its caller. The shapes worth transferring are in `docs/SHAPE-TRANSFERS.md`
(R2, R3). A module-level gate could
check the candidate's premise (output equality and speed of the origin kernel under the two schedules) and nothing
more.

## 8. Measured, inferred, not verified

Measured here: the kernel table, the experts' row invariance, the kit's identity. Read from source with file and
line: the path map. Measured by others and cited: section 2. Inferred: that the pair is kernel-bound; the decoder
tail's 0.2 s; candidate 1's memory cost; the power-mode explanation. Not verified: any number on F and A; the cost
of a prefill gather on the rail; the Triton kernels, the top-512 selection and the Engram host step; expert bit
widths per layer (3 and 2 bits were both run and differ by 6 % at most); real routing (uniform and one skewed
distribution were run; `TF_DS_PF_HIST` would give the real one); the exact arguments of `qual_bench.sh` and
`lane_rank.sh` on the deployed tree.

**Status of the window.** `gate/prefill_dsv41_pair.py` runs these three arms (and arm A again if time allows) inside
one hold, with the deployment's own `first_light.sh` and `qual/lane_rank.sh` as transient user units, and brings the
launcher's serving run back before it releases. It has not run: on 2026-11-11 at 18:39Z, inside an agreed window,
the session's permission system refused the start (taking the live serving pair down needs the owner's own approval).
Nothing was held, stopped or copied on F or A. Every number in this document is therefore still from one GPU.

## 9. Candidate 1 as built: units of two chunks (`executor.flerp.dsv41_zip2`), not yet run on the pair

Written 2026-11-11 after the A/B run (`gate/results/prefill-dsv41/pair-2026-11-11T1853Z/summary.md`: split on 1.44 s
a chunk, off 1.82 s, equal logits). Pod card SPEED-ZIP2 18b8d5d277. Recipe `dsv41-flash.strixfold-tp2-zip2`; the
derived module holds adapted copies of `model.zig` and `lanes.zig` (`catalog/flerp/executor/dsv41_zip2`), laid over
the origin's by the recipe's `[build.overlay]`. No catalogue file of the origin changed.

**The setting.** `TF_DS_PREFILL_ZIP2`, read by every rank and compared in the peers' handshake when set. Unset or 0:
the origin's buffers, cuts, frames and handshake text. 1: a prompt unit is two whole 2,048-row chunks. 2: the
second chunk of a unit may also be the prompt's last, partial one (at least max(`PREFILL_SPLIT`, 256) rows). It needs
`PREFILL_SPLIT` and its zip on. The serving scripts have no conf key for it (`serve_lib.sh` builds a rank's whole
environment from the conf); the measurement adds one line to a scratch copy of `serve_lib.sh`.

**The schedule.** `Model.unitRows(start, len, replay)` gives the rows of the unit at `start`: 4,096 when the chunk at
`start` is whole and before the replay cut and a second whole chunk follows; else the origin's chunk. Rank 0 sends
the unit in the `fill_chunk` frame it already sends. A 4,096-row unit runs the origin's `fillSplit` with halves of
2,048 rows in two 2,048-row buffer sets (the origin: 1,024 in two of 1,024): per layer, chunk k's attention, chunk
k + 1's attention, chunk k's experts, chunk k + 1's experts, each step followed by the end of the gather before it
and the begin of its own. So every gather still moves behind a kernel step of the other chunk, and the experts run
once per 2,048 rows: 35.6 ms a pass where two 1,024-row passes are 49.3 ms (section 3). Any other unit (a short
prompt, a prompt's odd chunk, the chunk that holds the replay cut when no whole chunk precedes it, a profile) takes
the origin's path, cut where the origin cuts it (`splitAt` with caps 1,024 / 1,024 whatever the buffers hold). The
8,163-token prompt of the baseline is one unit of 4,096 rows and two single chunks under 1 (half of it gains), two
units under 2. A 65,391-token prompt is 15 units and two single chunks under 1, 16 units under 2.

**Kept states stay on the 2,048-row grid.** A state is one slot's window ring of every layer and the positional
stores of the ratio-2 layers (`snapCopy`), plus compressed rows that stay in the extent. Inside a unit the second
chunk overwrites the rings, so the first chunk's are copied as it leaves them: after its attention at each encoder
layer, and after its replay cut for the cut layer and the decoder layers, into one buffer of a state's size in
`snapCopy`'s layout (`Model.midSave`, about 11 MiB a rank). The prompt pass calls the cache's keep for the unit's
first chunk end under the same condition as for any chunk end, and `snapCopy` then hands out that buffer; rank 0
tells rank 1 with one more byte in the `snap_save` frame (only under the setting). The state's layout word and size
are unchanged, so files and exports of either build have the same form. Resuming from a kept state is unchanged
(a prompt pass starts at a chunk end; from there the units form as above).

**A kept state's identity is not the schedule's to keep.** `native.zig learnIdentity` hashes the conf file's sha256
and the deployed commit, among others. The states under `/cache/tfp-learned` are read back only by a server whose
conf bytes and `DEPLOY.json` commit are the ones that wrote them. If the setting becomes a line of the serving conf,
those states are not read (as after any conf edit today); they import and serve only if the setting reaches the
ranks without the conf's bytes changing. The measurement does exactly that to test the requirement (one conf file,
a symlink for the deployment, a one-line setting file).

**Memory, computed, not measured.** Two 2,048-row buffer sets in place of two of 1,024: 2,048 more rows of
`Chunk.init`'s per-row buffers. Summed from `prompt.zig:336-515` and `exl3_experts.zig:312-316` at this model's
sizes: about 1.09 MB a row (Engram exchange buffers 0.36 MB, experts scratch 0.30 MB, the rest 0.43 MB), so **about 2.1 GiB a rank**, three times the card's estimate; plus 32 MiB of pinned staging and about 60 MiB of
host Engram rows. Measured anchors: `/health` 103.45 GiB active as served; the arena is 4.513 GB with the split and
3.806 GB without (rank 1's load line, the A and B arms), so the per-set fixed part is 0.66 GiB and is not paid
again. Expected `/health` with the setting on: about 105.6 GiB against `PEAK_GIB=106` and `MEMORY_LIMIT_GB=110`,
before the prompt cache's 4 GiB of kept states. Off, the build allocates what the origin does. The arms record the
real numbers.

**What has been shown without the pair** (`gate/results/prefill-dsv41/zip2-build/`):
- the recipe assembles (314 files, two of them the derived module's) and builds on control-node; its kit is the
  deployed one byte for byte (169 objects, manifest 76936479...);
- the origin's host tests (`zig build test-dsv41-hip-host`, no GPU work) on an export of the fork with the two files
  laid over it: 146 pass, 8 skipped, 1 failed of 155 in the family's suite, against 144 / 8 / 1 of 153 on the export
  as it is. The one failure is the origin's own in both (a test that expects `frames=2` where the protocol is 4);
  the two tests added here (the units cover a prompt, end on the grid, and keep the first chunk before the cut) pass;
- the base recipe's check, origin and build stages still pass (314 of 314 files, the kit equal);
- the premise at kernel level was measured before (section 3 and `bench.json` `checks`: one experts pass over 2,048
  rows equals two over its halves bit for bit).
No one-GPU test of the schedule itself exists: a prompt pass needs both ranks' weights and the link.

**What has not been shown:** that the setting-off build gives the deployment's sha and time, that the setting-on
build gives the same prompt logits, the time per unit, the memory, and that a state kept inside a unit equals the
reference state. `gate/prefill_dsv41_zip2.py` measures each (arms A, Z0, Z1, SA, SZ, Z2; its header says what each
does). It has not run.

**Candidate 4 (one experts pass for both halves of today's chunk) is not simpler.** It needs a second routing
buffer layout or a combined 2,048-row experts scratch beside the two half-chunk sets, a new step order inside
`fillSplit`, and it leaves three of four gathers a layer with nothing to hide behind. The unit schedule reuses
`fillSplit` unchanged in its order (about 100 changed or added lines of `model.zig` outside tests and comments) and keeps
every gather hidden; its cost is the memory above. Candidate 4 costs no memory, so it is the fallback if 2.1 GiB a
rank does not fit, at the price of a serial 20 MiB gather three times a layer.
