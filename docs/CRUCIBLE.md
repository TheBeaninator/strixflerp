# Crucible: what it does differently, and what strixflerp should take from it

2026-11-11, agent. Read-only review of Crucible `d6af02c` (catalogue origin `crucible`, 27 modules,
`catalog/crucible/`) and its docs; no GPU was used, nothing of Crucible was run. Paths are origin-relative
(`catalog/crucible/<kind>/<module>/src/<path>`). "Origin" = Crucible's code or docs; "pod" = a pod measurement;
"inferred" = my arithmetic. strixite figures come from `docs/STRIXW-DESIGN.md` and the strixw prefill report
(pod `/data/pod/reports/2026-11/2026-11-11-strixw-prefill.md`).

"Taking a module" below never means dropping Crucible's code into another engine. It means taking its **shape**
(the problem, the algorithm, the data layout and work mapping, the trick that makes it fast) and re-expressing it
in the target's format and situation. Format differences are adaptation work, not a stop sign.

## 1. The headline numbers, and how comparable they are

| | Crucible v0.34 (origin README, balanced, 4.05 pack) | strixite fleet (pod) |
|---|---|---|
| cold prefill 32K / 130K | 1,831 / 1,782 tok/s | ~1,188-1,245 tok/s (16K-128K) |
| decode 1 session, MTP | 71 (code) / 58 (prose) tok/s | ~42-46 tok/s |
| decode, no drafts | 34.2 tok/s (4.05), 39.1 (3.05) | not on record |
| 4 sessions total | 118-129 tok/s | n/a (one slot a worker in the fleet profile) |
| weights read a token (decode) | 5.36 GB (4.05), 4.62 GB (3.05) (`docs/how-it-got-fast.md:11-12`) | Q4 experts (expert down Q4 **G32**), Q8 dense: more (header in STRIXW-DESIGN) |
| quality | KL 0.0230 vs BF16, 896 positions (`how-it-got-fast.md:26-30`) | not measured on the same method |

Not like for like: different quantization (EXL3 trellis at 4.05 bpw vs strixw Q4/Q8), different power modes (the
fleet's C D are balanced 85 W, A B F quiet 54 W; Crucible's rows ran at ~2.3 GHz balanced, `docs/RESULTS.md:302`),
different clients. The `speed` stage of `gate/qwen38-flash-next.crucible.toml` measures both on one host, one
prompt set, once the EXL3 pack lands. Until then: **Crucible claims ~1.5x strixite's cold prefill and ~1.4-1.6x
its MTP decode.**

## 2. Prefill: why it is faster

The origin's account: cold prefill is at 53-56% of a *measured* 48.2 TFLOPS f16 WMMA ceiling
(`docs/RESULTS.md:241`, `README.md:187`). strixite's PF-CF profile puts its experts at ~43% and its dense GEMMs at
~57% of an *assumed* 59 TFLOPS (strixw report section 2); on Crucible's measured ceiling those are ~52% and ~70%
(inferred, clocks differ). So per-kernel efficiency is not where 1.5x comes from. What differs is how much work and
how many passes a token costs, and what the GPU does between kernels:

1. **GEMM rows decoupled from core rows** (`model/flashnext/prefill_launch.h:13-29`, `runtime/prefill_rows.h:5-6`).
   A prompt is planned in 4,096-row chunks, but four chunks merge into one *bulk launch* of 16,384 rows (+128 tail
   join) for every GEMM (MoE experts, dense projections), while GDN and QSA cores run inside the launch in
   2,048-row parts so their scratch stays small. Each decoded expert weight tile serves 4x the rows of a chunk.
   Measured: 16,384 vs 8,192 launch rows +3.1% (32K) / +3.0% (130K) for 3.3 GB more device memory (stage arena
   4,656 vs 2,346 MiB, `prefill_launch.h:24-27`), bit-identical; 32,768 rows +4.7% for 9.8 GB. strixite runs every
   kernel on the same 8,192-row chunk: ~160 expert slots a pass, two Wide passes at ~62% fill (strixw report).
2. **MoE expert GEMM items of 192 rows x 128 columns, one trellis decode for 12 row tiles**
   (`layers/prefill_moe_tile.h:1-9`). Each wave decodes one weight tile per k step into LDS; every wave then runs
   3 row tiles x 4 decoded column tiles. Gate and up run in one 8-wave block (waves 0-3 gate, 4-7 up) and the
   SwiGLU "mid" runs in the epilogue, so no fp32 gate/up partials go to memory (`layers/prefill_moe_pair.h:1-7`).
   LDS planes are laid out against bank conflicts (`prefill_moe_tile.h:20-25`: a 12-word stride had 9% conflicts).
   Dispatch is a stable expert-major counting sort without atomics (`layers/kernels/prefill_moe.h:1`).
3. **Fewer, larger launches; host gaps closed.** "No small launch at the end" (`how-it-got-fast.md:90-91`), cold
   prompts launched in pairs on the roadmap (`README.md:187`), a stage arena that reuses row buffers whose live
   ranges do not overlap (`runtime/stage_arena.h:1`, `stage_plan.h:1`), PLE rows gathered on a host thread that
   overlaps device work (`model/flashnext/ple_host.h:216`, `CRUCIBLE_PLE_FETCH`). strixite loses ~18% of a chunk's
   wall time outside kernels (inferred in the strixw report, to be re-measured).
4. **Work not done:** last-layer row pruning (only rows a reader needs run the last layer's tail;
   `runtime/prefill_prune.h:1-10`, after gufo); chunk-64 WY/UT GDN on WMMA (`layers/kernels/prefill_gdn.h:1-10`);
   QSA prefill attention where 4 query rows share one key list and one KV load
   (`layers/kernels/prefill_qsa_union.h:1-5`; +2.6% at 32K, +3.9% at 130K, `CHANGELOG.md:357-364`); a dense
   half-precision GEMM kernel +4.4-4.8% a call, +1.3% cold 32K, equal bytes (`CHANGELOG.md:408-409`).
5. **EXL3 tiles are already WMMA-shaped.** A 16x16 trellis tile is the WMMA fragment; Crucible stores tiles
   column-block-major (`exl3/kernels/exl3_gemv.h:11-13`) so a wave streams contiguous memory, and decodes the tile
   once into LDS for all row tiles. strixw is row-major with separate scale/min planes: its WMMA kernels must stage
   128 scattered row pieces per k step (the "1.65x lesson", strixw report section 2). This is the one advantage
   that is format-bound, and section 4 says how to get it anyway.

## 3. Decode, speculation, memory, long context

- **Decode GEMV** (`exl3/kernels/exl3_gemv.h:1-13`): lane t owns 8 trellis states in one contiguous 7K+16-bit
  window, three dword loads and two funnel shifts per tile; `v_dot2_f32_f16`, one accumulator chain per (row,
  column) in fixed tile order, so a row's bits never depend on how many rows share the pass (exact batching and
  drafting). Dense matrices get a "front order" so concurrent blocks read side by side (`exl3/front_order.h:1-5`).
- **Decode experts: a stream grid** (`layers/moe_sg.h:1-13`): one launch does gate/up, mid and down of all expert
  jobs; G persistent 8-wave blocks claim units (column blocks x whole K) from a global counter, no block barrier,
  88-95% of the byte floor in its bench; the shared expert's down units last.
- **MALL prefetch** (`exl3/lookahead.h`, `how-it-got-fast.md:86-89`): a small kernel on a second stream reads the
  next weights into the 32 MiB MALL while compute runs: 1.15-1.21x on a chain of this model's layer sizes, nothing
  for >= 4 MiB chunks. Serial decode is at 70-77% of measured bandwidth (`how-it-got-fast.md:95-97`).
- **Speculation**: the checkpoint's MTP head, verification commits every token (`docs/ARCHITECTURE.md:205-214`);
  a 4- or 2-bit copy of the head rows scores coarse logits, then the 8 best 128-token blocks are recomputed from
  the EXL3 tiles (`runtime/draft_quant.h:1-8`); draft length from the MTP top-2 margin (`runtime/draft_accept.h`);
  int8 HC on the MTP sites and the next MTP pass overlapped with the server's step (`CRUCIBLE_DRAFT_CHAIN=hc8,lead`,
  `engine/switches.h:52`); packed verification of up to 16 rows across sessions (`ARCHITECTURE.md:184-187`).
  4.69 tokens accepted per pass on code (`how-it-got-fast.md:44-45`).
- **State**: fp32 GDN state plus a 16-row history and a PLE ring a sequence (`model/flashnext/fixed_state.h`);
  QSA KV 2 heads x 256 (`geometry.h:60-65`); one session of 524,288 tokens is 13.84 GB (README); conversation
  cache with immutable entries, an NVMe tier and a code-identity "state epoch" (`runtime/state_*.h`). The 39 GB
  n-gram table can stay on the host page cache: 16 rows a token (`ple_host.h:1-7`, `CRUCIBLE_PLE_HOST`).
- **Long context**: plain RoPE to 262,144, YaRN x2 / x4 to 524,288 / 1,048,576 from generated tables; a sequence
  keeps one mode from token 0, a change is a full prefill (`runtime/rope_mode.h:1-3`). Measured: 490K cold
  prefill 1,436 tok/s, decode with drafts 52.6 tok/s over five workloads (README "Long context").
- **Exactness as an engineering rule**: chunk plan fixed by the request, packed rows equal single rows, cache hit
  equals miss, and three GPU gates enforce it (`ARCHITECTURE.md:216-241`); lossy speedups are switchable off with
  `CRUCIBLE_EXACT=1`. This is the same discipline as strixflerp's bit-identity gates.

## 4. Module shapes, re-expressed

For each: (a) the shape; then how it lands in (S) strixite (strixw Q4/Q8 row-major, or the new panel layout),
(F) StrixFold (EXL3 2-3 bit, DSV4.1, TP2), (Q) strixflerp's own qwen35moe/GGUF executor; and the gate.
Ranked by expected gain for the fleet (prefill gap first, then decode).

**4.1 Bulk launch: GEMMs over 16-32K rows, cores over 2K parts** (`prefill_launch.h`, `prefill_rows.h`).
(a) Planning chunk != GEMM launch: the expert and dense GEMMs see all rows of 4 chunks, GDN/QSA cores run in parts;
a stage arena keeps memory bounded. (S) strixite's `qwen4exp` prefill loop runs all kernels per `--chunk`. New
code: a layer loop that runs projections and the expert GEMM over a 16,384-row launch and the delta-rule and
attention kernels over 2,048-4,096-row sub-parts, plus an arena plan; kernels unchanged (`grouped_wmma_f16`
Wide256 already takes large slot counts). Carries over unchanged: every kernel's per-token math, so expected
**bit-identical** (GDN chunk boundaries must stay at the same positions: keep the sub-part size a multiple of the
current chunk's GDN chunking). Expected: +3-5% from fewer passes alone, more if the second 62%-full Wide pass
disappears (320 slots/expert fill 2.5 Wide tiles). Gate: token hash identity vs v070tg on the strixite prefill
gate's prompts; bench at 16K/64K/128K; GTT cost recorded. (F) Same shape over StrixFold's 64-1,024-row expert
jobs where the per-pass cost is ~15 ms "whatever the row count" (`docs/PREFILL-DSV41.md:138`): the largest
relative win of the three, TP2 adds the rule that both ranks cut the same launches. (Q) The GGUF executor's
prompt batching gets the same split; gate: bit-identity vs its own world-1 run.

**4.2 Decode-once, 12-row-tile expert items with fused gate/up/SwiGLU** (`prefill_moe_tile.h`, `prefill_moe_pair.h`).
(a) One weight tile decoded per k step serves 192 rows; gate and up share one block and the activation is applied
in the epilogue; no fp32 partials round-trip. (S) strixite dequantizes Q4 to F16 into LDS per tile already; what
changes is the item geometry (192 slot rows x 128 columns, 3x4 tiles a wave instead of its 64x128 tiles) and the
gate/up pairing with the SwiGLU epilogue (today gather+SwiGLU is a separate step and `combine_reduce` costs 186 ms
a chunk). Bit-identity holds if each output keeps its fp32 WMMA chain over K in the same order, as Crucible proves
for its own variants. Expected: part of the ~9% the strixw report puts on expert-kernel efficiency. Gate: module
gate vs `moe_grouped` on fleet shapes (N 1,280 x 512, K 2,560 G128; down K 640 G32), bit-equal, bench by symbol.
(F) StrixFold's EXL3 expert kernel decodes tiles too; the change is the larger item and the pairing: exact by
the same argument. (Q) Q4_K/Q6_K dequant per tile into LDS is the GGUF form of the same idea.

**4.3 WMMA-native tile order for prefill weights** (EXL3 16x16 tiles, column-block-major).
(a) Weights stored so one k step of one column block is one contiguous run and already in fragment order.
(S) This is strixw's "tile-major copy" option: the panel layout work (`docs/STRIXW-LAYOUT-NEXT.md`) is the place.
For dense matrices a load-time copy (+~2.4 GB GTT) costs nothing to decode; for experts only an in-place layout
that decode also reads fits in GTT, so it must come with a panel-aware decode GEMV (the panel decode2 gate already
exists). Gate: bit-identity of GEMM outputs, decode tok/s per class not worse. (F) Already EXL3 tiles; check
StrixFold's tile order against Crucible's column-block-major. (Q) GGUF blocks are 256-wide rows: a load-time
repack is new code with a full-precision fidelity check only if the dequant order changes.

**4.4 Host gap closing** (merged launches, PLE gather on a host thread, stage arena, no tail launch).
(a) Keep the GPU queue full during prefill. (S) Measure first: rocprofv3 wall minus kernel time per chunk on
v070tg (strixw report section 4, item 1); then overlap strixite's n-gram SSD gather (a cold 65,536-row gather is
494 ms) with the previous layer group exactly as `ple_host.h:216` does. Bit-identical by construction. Up to ~18%.
(F, Q) The same profile first.

**4.5 Decode stream grid for experts** (`moe_sg.h`). (a) Persistent blocks pull (matrix, column-block) units from
a counter; no barriers; whole-K per unit. (S) strixite's decode experts are split-K GEMVs over row-major Q4: the
unit becomes "16 output rows x whole K" and the claim order becomes job-major with shared expert last; exact if
each row's K order is kept. Gate: bit-identity of decode tokens, tok/s at 1 and 4 rows. (F) StrixFold's epoch /
ready decode experts are the closest relative; compare at the TP2 shard shapes. (Q) GGUF Q4_K rows, same.

**4.6 MALL lookahead prefetch** (`lookahead.h`). (a) A second-stream kernel touches the next layer's weights into
the 32 MiB MALL. Format-free: carries over to all three unchanged except which address ranges come next. Exact
(no math). Origin: 1.15-1.21x on small chains. Gate: decode tok/s ABAB, token identity.

**4.7 Low-bit draft-head scoring with exact rescoring of the best blocks** (`draft_quant.h`). (a) Coarse 2/4-bit
logits pick 8 candidate 128-token blocks; those are recomputed exactly; drafts only. (S) strixite's MTP head is
Q8: a q2/q4 shadow of its LM head rows for drafting, verification unchanged; tokens provably identical (lossless
speculation), only acceptance changes. Gate: token identity with drafts on/off, acceptance and tok/s on code and
prose. (F) DSV4.1's MTP head, same. (Q) n/a until it drafts.

**4.8 Four-row QSA union flash** (`prefill_qsa_union.h`) and **chunk-64 WY GDN** (`prefill_gdn.h`): same model
family as strixite, so the shapes transfer to strixite's attention (10.6% of a chunk) and delta-rule (13%) kernels
directly; the WY form changes summation order (fidelity rule, not bit-identity); the union flash keeps each row's
selection exact (bit-identity possible if the per-row reduction order is kept).

## 5. What RIVALS.md says about strixite and hipfire

Nothing. RIVALS.md (and the whole tree: README, CHANGELOG, docs, engine) never names strixite, hipfire or StrixFold.
It compares only Halogen 0.18.14 and Gufo 0.22.0, measured by the Crucible author on one machine, Oct 23-26
(`docs/RIVALS.md:7-33`): Halogen leads cold prefill (1,938 vs Crucible 1,614 tok/s at 32K then) and one-session
decode; Crucible leads 4- and 8-session totals. Gufo's last-layer pruning is credited in the code
(`runtime/prefill_prune.h:1`). For the pod: Halogen's rows (1,938-2,121 tok/s at 32K) say the prefill ceiling on
this chip is well above both Crucible and strixite.

## 6. Tests strixflerp should run (prioritised)

1. **Serve the fleet model with Crucible outright on one worker** (gate stages `tokens`, `speed`, prepared in
   `gate/crucible_recipe.py --plan`): once `/data/ark/models/turboderp__Qwen3.8-Flash-Next-exl3` (revision
   4.05bpw_h6_ng6) lands, run Crucible and strixite on the same token ids on one host and power mode: cold prefill
   at 4K/32K/128K, decode with and without MTP on code and prose, 3 repeats, median; plus quality on the
   FIDELITY.md method (KL vs the BF16 reference's top tokens, top-1 agreement) for both packs. This turns every
   number in section 1 into a same-host number and is the cheapest decision input.
2. **Bulk launch in strixite (4.1)**: first the zero-code proxy, `--chunk 16384` vs 8192 ABAB (strixw report
   item 3, hash check); then the GEMM/core split. Measure TTFT at 16K/64K/128K and the token hashes.
3. **Wall-minus-kernel profile of v070tg** (4.4): rocprofv3 on a cold 16K and 64K prompt; the result sizes 4.4.
4. **Expert item microbench (4.2)**: a strixflerp kernel module re-expressing the 192x128 decode-once pair item
   on strixw Q4 at fleet shapes; bit-equality and TFLOP/s against `grouped_wmma_f16`, 80/160/320 slots/expert.
5. **MALL prefetch in strixite decode (4.6)**: ABAB decode tok/s, token identity.
6. **Draft-head q2 shadow (4.7)** on strixite: acceptance, tok/s, token identity.
7. **StrixFold bulk launch (4.1F)**: the DSV4.1 per-pass cost makes this the biggest relative prefill win there;
   gate as in `docs/PREFILL-DSV41.md`.
