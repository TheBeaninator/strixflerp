# Crucible and strixite, module by module

Written 2026-11-11 for the author of Crucible. It compares Crucible (at `d6af02c`) with strixite v0.7.0-pod (at
`20937dc`). Both serve Qwen3.8-Flash-Next on one Strix Halo (Ryzen AI Max+ 395, Radeon 8060S / gfx1151, 128 GB). The
HTTP and serving layers are left out.

**Labels.** Every number carries one of three labels. **[measured]** means we ran it on our own machines of this
model. **[claimed]** means the number comes from that engine's own docs. **[inferred]** means our own arithmetic.
**Crucible measured 2026-11-13 [M7]** (section 0). Sections 1-5 were written before that run; their Crucible
numbers are the project's own claims (README, `docs/RESULTS.md`), and section 0 says where our measurement agrees.

## 0. Measured side by side [M7]

Crucible `d6af02c` (rebuilt from the catalogue, `turboderp/Qwen3.8-Flash-Next-exl3` 4.05bpw_h6_ng6 staged on NVMe)
against strixite v0.7.0-pod (the fleet's `strix_server`, served strixw file) on control-node, the same token ids, one
engine loaded at a time. GPU clock high (dpm `high`, sclk 2.5-2.9 GHz) for every row except the strixite decode
server, after which sclk read 868 MHz / dpm `low`; the host load average was 7-20 from other pod work. Medians of 3.
Crucible prefill: one cold process per repeat. strixite prefill: the prefill gate's bench (chunk 8,192), one load,
round 0 first.

| Row | Crucible tok/s | strixite tok/s | Crucible / strixite |
|---|---|---|---|
| Prefill, cold 4K | 1,645 (1,606-1,688) | 872 (574-1,439) | 1.89 |
| Prefill, cold 32K | 1,667 (1,653-1,696) | 1,399 (846-1,461) | 1.19 |
| Prefill, cold 128K | 1,564 (1,528-1,573) | 1,230 (940-1,342) | 1.27 |
| Decode 256, code, no drafts | 31.9 | 32.1 | 0.99 |
| Decode 256, code, MTP default (Crucible spec 7; strixite MTP on) | 80.6 | 76.6 | 1.05 |
| Decode 256, prose, no drafts | 31.7 | 28.9 | 1.10 |
| Decode 256, prose, MTP default | 45.5 | 39.3 | 1.16 |

Against the claims: prefill 1,667 at 32K and 1,564 at 128K is 91% and 88% of the README's 1,831 / 1,782.
Decode without drafts 31.7-31.9 is below the claimed 34.2. With drafts, code 80.6 beats the claimed 71.2
(4.00 drafts accepted per forward, 84.6%), prose 45.5 is below the claimed 58.4 (0.98 per forward, 67.4%). strixite's
prefill is noisy (one slow repeat per length); its decode matches section 1's M1 range. So: prefill, Crucible ahead
by 1.2-1.3x at 32K-128K; decode, even without drafts, Crucible 5-16% ahead with them.

**Tokens** (Crucible gate, criteria fixed before the run): the export build and the catalogue-assembled build gave
the same greedy ids, repeat for repeat, on the strixite gate's three prompts (128 tokens), and `--spec 3` gave the
same ids as `--spec 0` (drafts lossless): PASS. First position where Crucible's greedy ids leave strixite's on the
same prompt ids (EXL3 4.05 against strixw; a fidelity record, not a criterion): short chat with thinking 5,
long chat 0, raw completion 6. The divergences are wording choices (synonymous openings), as expected from two
different quantizations of the same model.

**Citations.** Crucible paths are relative to the root of its repository (`engine/...`, `docs/...`). strixite paths
are relative to its own tree (`src/...`, `docs/...`). Our measurements are named M1-M6 (listed at the end).

**What strixflerp is.** strixflerp is the owner's catalogue of engine modules. They are harvested verbatim from
several Strix Halo engines: strixite, Crucible, hipfire, Gufo and StrixFold (Halogen and SGLang are listed as
origins, with no code harvested yet). Each module has a kind
(format, kernel, executor, runtime, serve or tool) and a named interface. Recipes assemble modules into binaries.
Hardware gates check that an assembled build matches its origin engine, bit for bit and in speed.

"Cross-engine reuse" here does not mean copying code between engines. It means taking a module's *recipe* (its
algorithm, data layout, work mapping and scheduling) and re-expressing it for another weight format: the same recipe
in a different size pan.

strixflerp is a personal research tool. It is not released and not supported: there is no download, no roadmap, and
no invitation to contribute.

## 1. Summary: answering "is strixite's decode better somehow?"

**Mostly no.**

**Decode without drafts is about even.** Crucible claims 34.2 tok/s on the 4.05 pack and 39.1 on the 3.05 pack
(README:28). strixite measured 32.1-33.5 [M1], and its own docs say 31.5 (`docs/how-it-got-fast.md:130`). Both
engines read about 5 GB per token: Crucible says 5.36 GB at 4.05 (`docs/how-it-got-fast.md:12`), and we estimate
~5.0-5.1 GB for strixite [inferred, from `docs/weights.md:17-30`]. Both engines run at ~70-77% of the measured read
bandwidth (`docs/how-it-got-fast.md:96`; strixite 31.5 tok/s at ~226 GB/s, `docs/how-it-got-fast.md:58` [inferred]).

**With drafts, Crucible is level on code and ahead on prose.**

- Crucible claims 71.2 tok/s on code and 58.4 on prose (README:25-26).
- strixite's own docs say 59-71 on code and 41-54 on prose (`docs/how-it-got-fast.md:133`).
- We measured strixite at 47.8-67.0 on our full-power machine [M1], and 42-46 on a serving machine in a low-power
  profile [M4, M5].

**Crucible's speculation is also exact, and strixite's is not.** Crucible's gate 3 requires the same bits with
drafts on and off (`docs/how-it-got-fast.md:66-77`). In strixite we measured that changing only the draft head's
quantization changed the greedy output on 7 of 20 prompts, while a control arm was identical on 20 of 20 [M5].

**Prefill: Crucible is ahead.** It claims 1,831 tok/s at a cold 32K prompt (README:20). strixite measures
~1,220 (1,167-1,293 at 7.5K-30K [M4]; 1,188-1,245 at 16K-128K [M6]).

**These figures are not like for like:**

- Weights: EXL3 trellis at 3.05 or 4.05 bpw, against strixw with Q4 G128 expert gate/up, Q4 G32 expert down on the
  served file, and Q8 G64 dense (`docs/weights.md:17-30`; served header [M6]).
- Power: Crucible ran a balanced profile at ~2.3 GHz (`docs/RESULTS.md:302`). M4 and M5 ran a low-power profile.
- Sessions: Crucible ran 4 sessions of 262K (README:16). strixite ran one slot of 262K.

**One strixite decode recipe looks worth a test in Crucible: 8-bit hyper-connection (HC) weights.**

- strixite stores all HC mixes as Q8 G64, 0.65 GiB (`docs/weights.md:24`).
- Crucible reads them as fp16 by default: 1,281 MB of a 5,595 MB decode token (`engine/layers/hc_int8.h:3-4`;
  `CRUCIBLE_HC_WMMA` defaults to 1, `engine/switches.h:59`).
- strixite's whole published layout, with HC at 8-bit, measured mean KL 0.028 against full precision
  (`docs/weights.md:90`). That is a different method from Crucible's 0.0230 over 896 positions, so the two figures
  are not directly comparable.

The rest of strixite's decode stack has close Crucible counterparts:

| strixite | Crucible |
|---|---|
| MTP chain with a margin gate | `engine/runtime/draft_accept.h` |
| 65,536-token draft vocabulary | `engine/runtime/draft_vocab.h` |
| n-gram rows kept off the GPU | `engine/model/flashnext/ple_host.h` |
| saved states | conversation cache + disk tier |

These look like parallel evolution, not a lead.

## 2. Module map

Module ids follow strixflerp's `kind.origin.name` scheme. Each "Stronger recipe" verdict is a judgement on the
evidence cited, not a measurement.

| Kind / family | Crucible | strixite | Stronger recipe, and why |
|---|---|---|---|
| format: weights | `format.crucible.exl3`: EXL3 trellis, 2-8 bits per matrix, mean 3.05/4.05 (`docs/how-it-got-fast.md:19-21`). 16x16 tiles repacked column-block-major (`engine/exl3/kernels/exl3_gemv.h:12-14`) | `format.strixite.strixw`, `q4`, `q8`: round-to-nearest affine, row-major codes plus separate scale/min planes. 4.41 bpw published (`docs/weights.md:5-8`) | Crucible on bytes and quality per bit (claimed KL 0.0230 at 4.05, `docs/how-it-got-fast.md:28`). Its tiles are already in WMMA fragment shape. strixw is decode-first: one row streams per wave (`src/kernels/linear_q4.hpp:6-12`) |
| format: n-gram table | `format.crucible.ngram16`: 6-bit pack "ng6", 39.0 GB at 4.05 (`engine/model/flashnext/ple_host.h:2-3`) | `format.strixite.ngram_table`: 8-bit rows, 48 GiB, on SSD (`docs/how-it-got-fast.md:161-165`) | Crucible (smaller); quality effect not compared |
| kernel: dense GEMV (decode) | `kernel.crucible.exl3_gemv`: lane owns 8 trellis states; 3 dword loads; fixed-order `v_dot2` chain per row, so a row's bits do not depend on how many rows share the pass (`exl3_gemv.h:4-10`). Front order for concurrent blocks (`engine/exl3/front_order.h`) | `kernel.strixite.linear_q4`, `linear_q8`: split-K over 1-8 waves; 32 codes per 16-byte load; one FMA per weight; 8 rows per pass at larger M (`src/kernels/linear_q4.hpp:6-12`) | Even on speed (section 1). Crucible on exactness: its batch-invariant accumulation is what strixite lacks [M5] |
| kernel: expert decode | `kernel.crucible.moe_decode`: stream grid. Persistent 8-wave blocks claim column-block units (whole K) from a global counter, with no block barrier; 88-95% of the byte floor in its bench; shared down last (`engine/layers/moe_sg.h:1-13`) | `kernel.strixite.moe_router` + per-expert Q4 GEMVs; routed experts 7.09 of 32.9 ms per token at 64K (`docs/how-it-got-fast.md:63-70`) | Crucible: one launch, barrier-free, and exact by construction |
| kernel: prefill GEMM (dense) | `kernel.crucible.exl3_gemm`: tile decoded once into LDS for all row tiles; half-precision kernel +4.4-4.8% per call (`CHANGELOG.md:408-409`) | `kernel.strixite.linear_wmma`: 64x128 tiles, K step 64, Q4/Q8 dequantized to F16 in LDS (`src/kernels/linear_wmma.hip:64-68`) | Crucible: its format avoids the row-gather transpose |
| kernel: grouped experts (prefill) | `kernel.crucible.moe_prefill`: 192x128 items, one trellis decode serving 12 row tiles (`engine/layers/prefill_moe_tile.h:1-9`); gate+up in one 8-wave block with SwiGLU in the epilogue (`engine/layers/prefill_moe_pair.h:1-7`); bank-conflict-aware LDS planes (`prefill_moe_tile.h:20-25`) | `kernel.strixite.moe_grouped`: route, then gather+SwiGLU, then combine; F16 dequant per tile; codes staged packed into LDS, because direct row loads were 1.65x slower (`src/kernels/moe_grouped.hip:347-353`) | Crucible: more rows per decoded tile and no fp32 gate/up round trip |
| kernel: attention (QSA) | `kernel.crucible.qsa`: 4-row union flash, one KV load per tile (`engine/layers/kernels/prefill_qsa_union.h:1-5`; +2.6% / +3.9%, `CHANGELOG.md:357-364`); fused tile-local top-512 (`prefill_qsa_select.h:1`); exact fp32 indexer in decode (`engine/layers/qsa_long.h:1-7`) | `kernel.strixite.attention`, `qsa`: WMMA attention; one-pass filter top-k, 99 to 44 ms per layer at 471K, +6.6% on a 490K prefill (`docs/how-it-got-fast.md:213-216`) | Crucible on the 4-row union. strixite's filter top-k is worth comparing at depth |
| kernel: gated delta / GDN | `kernel.crucible.gdn`: chunk-64 WY/UT on WMMA (`engine/layers/kernels/prefill_gdn.h:1-10`); single-launch verify core with row and state waves (`engine/layers/gdn_verify.h:1-22`) | `kernel.strixite.gdn`: chunked `delta_rule_wmma`, 13% of a prefill chunk [P1]; verify writes a spare state, so keep or drop is a pointer swap (`src/runtime/qwen4exp.hpp:476-482,777-780`) | Crucible on prefill (WY form). Verify: both avoid a snapshot copy |
| kernel: HC mixes | `kernel.crucible.hc`: fp16 by default, int8 behind a switch marked as a quality change (`engine/switches.h:59`, `engine/layers/hc_int8.h:1-9`) | `kernel.strixite.hc`, `hc_wmma`: Q8 G64, chunk-major up-mix copy for coalesced decode loads; 16% of a prefill chunk [P1] | strixite on decode bytes (section 1); neither measured on the other's quality method |
| kernel: norms / rope | `kernel.crucible.ops`: generated YaRN tables, one mode per sequence (`engine/runtime/rope_mode.h:1-3`) | `kernel.strixite.norm`, `rope`, `rope_math`: YaRN x2 to 512K, no measured cost (`docs/how-it-got-fast.md:157-159`) | Even |
| kernel: n-gram embedding (PLE) | host gather from the page cache with read hints (`ple_host.h:1-19`); prefill gather on a host thread overlapped with device work | `runtime.strixite.ngram_table`, `kernel.strixite.ple`: 8M-row RAM cache, 64 SSD reader threads (`src/runtime/ngram_table.cpp:20-21`); prefetch during drafting, 1.3% (`docs/how-it-got-fast.md:176`) | Even; different mechanisms for the same problem |
| MTP / draft | `runtime.crucible.speculation`: margin prior (`engine/runtime/draft_accept.h:5-13`); q2/q4 head copy for coarse logits, then the 8 best 128-token blocks recomputed exactly (`engine/runtime/draft_quant.h:1-11`); int8 HC at MTP sites + overlapped lead pass (`switches.h:52`); packed verify of up to 16 rows (`docs/ARCHITECTURE.md:186`); optional n-gram drafts (`:207`) | `kernel.strixite.mtp_pick` + executor: up to 7 drafts with a top-2 margin gate; Q4 G64 head over the 65,536 most frequent ids (`src/runtime/qwen4exp.cpp:229-281`); prompt-lookup drafts rejected for too little gain (`docs/how-it-got-fast.md:248`) | Crucible: coarse-then-exact scoring of the full vocabulary, plus exactness |
| sampling | `runtime.crucible.sampling`: keyed sampling stream, so a seed gives the same ids at any draft depth (`docs/ARCHITECTURE.md:211-212`) | `runtime.strixite.sampler`, `logits_topk`: top-20 picked on the GPU (`docs/how-it-got-fast.md:187`) | Crucible on reproducibility |
| executor | `executor.crucible.flashnext`: 4K planning grid, bulk launches of 16,384+128 rows, GDN/QSA cores in 2,048-row parts (`engine/model/flashnext/prefill_launch.h:13-29`); last-layer pruning (`engine/runtime/prefill_prune.h:1-13`) | `executor.strixite.qwen4exp`: every kernel per 8,192-token chunk; grouped experts from 6 tokens, WMMA from 12 (`src/runtime/qwen4exp.hpp:724,730`) | Crucible (section 3) |
| runtime: memory arena | `runtime.crucible.device_memory`; stage arena reusing row buffers whose live ranges do not overlap (`engine/runtime/stage_arena.h:1`) | `runtime.strixite.memory_plan`, `memory_governor`, `paged_buffer` | Crucible for prefill row buffers |
| runtime: state / prompt cache | `runtime.crucible.conversation_cache`, `state_disk`: immutable snapshots; NVMe blocks of 4,096 rows named by SHA-256 (`engine/runtime/state_blocks.h:1-20`); state epoch | `runtime.strixite.prompt_cache`, `paged_state`: base + one delta, never chained; 0.2 GB instead of 11.5 GB deep in a conversation (`src/serve/prompt_cache.hpp:50-62`; `docs/how-it-got-fast.md:233-235`) | Even; two good designs for the same problem |

## 3. Prefill, stage by stage

**Where the gap sits [inferred; different builds and clocks].**

- strixite's only per-kernel profile [P1] (an October build, 8,192-token chunk, 16K depth) shows 5,523 ms of GPU
  time per chunk. That is 1,483 tok/s of kernel time alone.
- At ~1,220 tok/s end to end, a chunk takes ~6.7 s of wall time.
- Crucible claims 1,831 tok/s, or 0.546 ms per token. strixite spends 0.674 ms per token inside kernels and ~0.15
  ms per token outside them.
- So roughly half of the gap is kernel time and half is time between kernels.
- Per-kernel efficiency is similar: Crucible claims 53-56% of a measured 48.2 TFLOPS (README:187,
  `docs/RESULTS.md:241`). strixite's experts and dense GEMMs come to ~52% and ~70% of that same ceiling [inferred
  from P1]. The difference is passes and gaps, not kernel quality.

**Stages that cost more time in strixite.**

| Stage (strixite share of chunk [P1]) | strixite mechanism | Crucible mechanism | Likely contribution |
|---|---|---|---|
| Routed experts (32.7%: `grouped_wmma_f16` 1,519 ms, `combine_reduce` 186 ms) | 8,192 tokens x 10 slots / 512 experts = ~160 slots per expert, so two 128-slot passes at ~62% fill. Weights are dequantized twice and gate/up are written out before SwiGLU | 16,384-row launches (~320 rows per expert); 192-row items decode a tile once for 12 row tiles; gate/up paired with the SwiGLU epilogue; counting-sort dispatch with no atomics (`engine/layers/kernels/prefill_moe.h:1`) | Largest kernel-side share |
| Dense projections (GDN 17.9%, attention 5.5%) | Row-major Q4/Q8 staged through LDS: 128 scattered 32-byte pieces per K step (`src/kernels/moe_grouped.hip:349-353` and the same pattern in `linear_wmma`) | Column-block-major 16x16 tiles already in fragment order; half-precision GEMM | The layout recipe transfers: strixw re-laid in fragment-friendly panels (being built; see C2) |
| HC mixes (15.8%) | Q8 WMMA mixes plus `hc_inject_inv` | `engine/layers/prefill_hc.h` (not profiled by us) | Unknown until traced |
| GDN core (13.0%) | chunked `delta_rule_wmma` | chunk-64 WY/UT on WMMA | Probably Crucible |
| Attention (10.6%) | `attention_partial_wmma` | 4-row union flash (+2.6% at 32K, claimed) | Small, Crucible |
| Outside kernels (~18% of wall [inferred]) | **First chunk: 2.69 s of a 10.8 s chunk idle; later chunks 0.28-0.38 s** [M2, 32K, traced]. Cold n-gram gather costs 494 ms for 65,536 rows (`src/runtime/ngram_table.cpp:20-21`) | Fewer, larger launches; no small tail launch (`docs/how-it-got-fast.md:90-91`); PLE gather overlapped on a host thread; stage arena | About half the total gap |
| Work skipped | none | last-layer row pruning (`prefill_prune.h`) | Small at 32K |

**What we have already tried in strixite.**

- **Chunk 16,384 instead of 8,192 (no other change)** [M2]: outputs bit-identical, but prefill was -4.8% / -10.1% /
  +3.6% at 8K / 32K / 128K. This is noisy and does not pass the gate.
  - The traced first chunk got bigger (14.2 s GPU, 2.2 s gap). A bigger chunk alone does not reproduce Crucible's
    split: Crucible runs the GEMMs over 16K rows while the GDN/QSA cores stay in 2K parts, and adds an arena.
- **A combine-kernel variant** [M3]: bit-identical, but no consistent gain (+0.2% at 32K, +11.6% at 128K, -43% at
  8K against a noisy base).
- **Expert-down G128 instead of G32** [M4]: prefill unchanged (1,167-1,293 against 1,176-1,291 tok/s).

Layout bytes alone are not where the prefill time goes.

## 4. Decode in detail: is strixite's decode better?

1. **Bytes per token: about the same.**
   - Crucible: 5.36 GB at 4.05, 4.62 GB at 3.05 (`docs/how-it-got-fast.md:11-12`).
   - strixite, published layout [inferred from `docs/weights.md:17-30`]: experts 61 GiB x 10/512, plus Q8/Q4 dense
     3.5 GiB, comes to ~5.0 GB. The served file adds ~0.07 GB because its expert down is G32.
   - Where the bytes go differs:
     - strixite puts 8 bits everywhere it reads every token. Its 64K step profile spends 10.6 of 32.9 ms on Q8 dense
       and the LM head (`docs/how-it-got-fast.md:63-70`).
     - Crucible spends fewer bits on dense matrices, but reads HC at fp16 (`hc_int8.h:3-4`).
2. **Bandwidth efficiency: equal.** Both are at ~70-77% of measured read bandwidth (section 1).
   - Crucible keeps gaps down with a barrier-free stream grid and a MALL prefetch on a second stream (1.15-1.21x on
     this model's layer sizes, `docs/how-it-got-fast.md:86-89`).
   - strixite measured 3.18 ms of host time and gaps across ~1,130 launches per token (`docs/how-it-got-fast.md:70`),
     and found HIP graphs slower (`docs/how-it-got-fast.md:243-244`).
   - Crucible's prefetch is the transferable piece: it is format-free.
3. **Drafting: Crucible is ahead on prose, level on code, and ahead on exactness.**
   - Crucible claims 4.69 tokens accepted per pass on code (`docs/how-it-got-fast.md:44-45`).
   - strixite: 94% of first drafts correct on code (`docs/how-it-got-fast.md:133`). Our acceptance rate is 0.57-0.66
     on the long decodes and 0.79 on the quality prompts [M5].
   - [M5] shows that strixite's verify pass is not batch-invariant. Its split-K GEMV and its "8 rows per pass" path
     for larger M (`src/kernels/linear_q4.hpp:6-12`) are the likely cause [inferred].
4. **Long context: flat in both.**
   - strixite: 48.9 tok/s at 0-128K and 48.5 at 384-512K with drafting (`docs/how-it-got-fast.md:157-158`).
   - Crucible: 52.6 tok/s at 490K over five workloads (README:60).
5. **Batching.** Crucible's packed multi-session verify (up to 16 rows) has no strixite counterpart in this
   comparison. strixite was measured serving one slot.

**Verdict:** no strixite decode module is clearly better. The 8-bit HC layout is the one difference worth a test.
strixite's base+delta state saving and its filter top-k at depth are good alternative designs, but not demonstrated
wins over Crucible's.

## 5. What the meta-engine view shows

Viewed module by module, the two engines' sub-recipes mostly line up one to one (section 2). In each family, the
difference is a recipe that can be expressed independently of the weight format. Each transfer below is that recipe
re-expressed for the other engine's format and checked by a gate.

**Crucible to strixite (strixw Q4/Q8):**

| # | Recipe | Re-expressed as | Gate that would prove it |
|---|---|---|---|
| C1 | Bulk launch: GEMMs over 16K rows, cores in 2K parts, plus stage arena | New layer loop in `executor.strixite.qwen4exp`; kernels unchanged | Logits sha256 and greedy ids equal to base at 8K/32K/128K; prefill tok/s ABAB, every repeat above the base max; memory recorded |
| C2 | 192-row decode-once expert items, gate/up pair with SwiGLU epilogue | A new kernel module for strixw Q4 that dequantizes a tile into LDS once for 12 row tiles | Bit-equal to `moe_grouped` at the served shapes (N 1,280 x 512, K 2,560 G128; down K 640 G32); TFLOP/s at 80/160/320 slots per expert |
| C3 | Host-gap closing | Overlap the first chunk's n-gram gather with the previous layer group | Wall-minus-GPU per chunk (the 2.7 s in M2 should shrink); bit-identical |
| C4 | Batch-invariant decode accumulation | Fixed per-row K order in `linear_q4/q8` regardless of M | Greedy output identical with drafts on and off and across draft-head variants (the M5 test) |
| C5 | Stream-grid experts, MALL prefetch | Units of "16 rows x whole K" over row-major Q4; a second-stream prefetch | Decode tok/s ABAB at 1 and 4 rows; token identity |
| C6 | Coarse-then-exact draft scoring | q2 shadow of the full LM head with the best blocks rescored from Q8 | Acceptance and tok/s on code and prose; output identical (needs C4 first) |

**strixite to Crucible (EXL3):**

| # | Recipe | Re-expressed as | Gate |
|---|---|---|---|
| S1 | 8-bit HC weights with a chunk-major up-mix copy for decode | Crucible's existing `hc_int8` path as the default, plus a decode-only chunk-major copy | KL and top-1 against BF16 on Crucible's 896-position method; decode tok/s ABAB (HC is 1,281 of 5,595 MB per token at fp16) |
| S2 | Filter top-k for the indexer at depth | Sampled cut-off then exact pick, in `prefill_qsa_select` / `qsa_long` | Same top-512 sets (exact); prefill at 256K-490K |
| S3 | Base + one-delta state saving | Alternative to SHA blocks for deep turns | Bytes written per turn and restore time; same bits as a miss |

**What we will measure first, once Crucible's 4.05 pack lands.** Crucible and strixite run on the same machine and
power profile, from the same token ids:

- Cold prefill at 4K / 32K / 128K under a kernel trace, giving per-stage time and wall-minus-GPU per chunk on both
  sides.
- Decode with and without drafts on code and prose, 3 repeats, median.
- Quality for both packs against BF16 on the same method (KL over the reference's top tokens, top-1 agreement).

That trace turns sections 3 and 4 from inferred into measured, and ranks C1-C3 by the gap they actually close. Until
then, Crucible's numbers above remain its own claims, and our gap decomposition is arithmetic across different builds
and clocks.

## Measurement records (ours)

| Id | What | Conditions |
|---|---|---|
| M1 | strixite rebuilt from the catalogue against the origin build; decode tok/s with and without MTP on 3 prompts; token identity over 18 requests | 2026-11-11; one machine; full-power profile |
| M2 | Prefill gate: chunk 16,384 against 8,192; logits and greedy identity; per-chunk timeline (wall / GPU-busy / gap) of a traced 32K request | 2026-11-12 UTC; same machine as M1 |
| M3 | Prefill gate: combine-kernel variant | 2026-11-12 UTC; same machine as M1 |
| M4 | Layout A/B: expert down G128 against G32; decode ~2K/~30K, prefill 7.5K/30K | 2026-11-11; serving machine, low-power profile |
| M5 | Draft head Q8 against Q4: A/B with a repeated control arm, 20 prompts, acceptance and greedy identity | 2026-11-12 UTC; serving machine, low-power profile |
| M6 | Served-file header and prefill range across 16K-128K | 2026-11 |
| M7 | Crucible recipe gate, stages tokens and speed (`gate/results/qwen38-flash-next.crucible/`: tokens.json, speed.json) | 2026-11-13 UTC; control-node, same machine as M1 |
| P1 | rocprofv3 per-kernel prefill profile, depth 16K, chunk 8,192 | 2026-10; older strixite build |
