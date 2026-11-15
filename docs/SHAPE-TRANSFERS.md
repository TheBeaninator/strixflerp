# Shape transfers: re-reading every "not swappable" as adaptation work

Status 2026-11-12. Review of docs, gate results, the pod report `2026-11-11-strixw-prefill.md` and the session
transcript of 2026-11-10..12. No GPU was used for this document; every gain below is an estimate with its arithmetic.

## The principle (the owner's correction, 2026-11-12)

Using one engine's component in another does **not** mean dropping the part in as-is. It means taking the module's
**shape** - the problem it solves, its key idea, its data layout, work mapping and scheduling, what makes it fast -
and re-expressing it to fit the target engine and weight format. A format mismatch is adaptation work, not a dead
end. "Can module X's bytes be called from engine Y" is the wrong question; "does X's idea pay in Y, and what does
re-writing it cost" is the right one.

What stays true from the old framing: a *compiled* kernel reads one format, so a drop-in swap across formats does not
exist. That is a statement about code objects, and it was being used as a statement about ideas.

Gates for a transfer: **bit identity** against the target's current kernel when the target's math (rounding points
and per-output summation order) is kept; otherwise the full-precision rule of `docs/FIDELITY.md` (KL against the
dequantized-weights FP32 reference, top-1, R's margin), never "equal to the other engine".

## 1. Instances found (30)

| # | where | the closing phrase (short) |
|---|---|---|
| 1 | `SPEC.md:16-17` | "The weight format is baked into every weight-reading kernel ... A 'universal GEMV' does not exist" |
| 2 | `docs/SOURCES.md:23` | section title restating 1 as a sourced fact |
| 3 | `docs/INTERFACES.md:65-66` | decode GEMV "not swappable across formats ... a swap means re-quantizing" |
| 4 | `docs/INTERFACES.md:74-76` | hipfire `ggml_mmvq` gfx1201-only: "not swappable today" |
| 5 | `docs/INTERFACES.md:110-114` | prefill GEMM "not swappable across engines" (rounding points, activation quantizers) |
| 6 | `docs/INTERFACES.md:132` | MoE experts "not swappable (weight formats)" |
| 7 | `docs/INTERFACES.md:141-149` | hipfire router "not swappable" (k=8, tie rule, one token a launch) |
| 8 | `docs/INTERFACES.md:150-151` | StrixFold router/gate "not swappable" (TP split, Triton bits) |
| 9 | `docs/INTERFACES.md:165-167` | indexer scores "not swappable" (FP4 keys, head counts) |
| 10 | `docs/INTERFACES.md:173-175` | QSA "not swappable as is" |
| 11 | `docs/INTERFACES.md:184-186` | Gated DeltaNet "not swappable ... an executor-level change" |
| 12 | `docs/INTERFACES.md:198-200` | hyper-connections "not swappable" (strixw vs Q8_0, folded norm) |
| 13 | `docs/INTERFACES.md:220-226` | RMSNorm (bf16) and RoPE "not swappable" |
| 14 | `docs/INTERFACES.md:237-241` | quantized KV "not swappable ... would have to be written again" |
| 15 | `docs/INTERFACES.md:268-269` | samplers "not swappable" |
| 16 | `docs/INTERFACES.md:280-283` | speculation "not swappable at module level" |
| 17 | `docs/PREFILL-DSV41.md:151` | routed experts: "no: weight format baked in" |
| 18 | `docs/PREFILL-DSV41.md:152` | prompt GEMM: "no" |
| 19 | `docs/PREFILL-DSV41.md:153,155` | indexer scores, sparse attention: "no" |
| 20 | `docs/PREFILL-DSV41.md:156` | mHC: "no: 5120 against 4096" |
| 21 | `docs/PREFILL-DSV41.md:157` | router, RoPE "judged not swappable" |
| 22 | `docs/PREFILL-DSV41.md:159,250-251` | "Nothing in the 70 % is a swap"; "swap in a better part did not, because the hot kernel's weight format is baked in" |
| 23 | transcript 2026-11-11T18:34 | "no other engine has a part for anything in the 70%" |
| 24 | pod `2026-11-11-strixw-prefill.md` row 10 | "kernel swaps from gufo or hipfire: weight-reading kernels are baked to their formats; none reads strixw Q4"; "its i8-WMMA MoE route is an idea, not a part" |
| 25 | transcript 2026-11-11T23:10 | "A strixw Q4 matrix can only be read by strixite's Q4 kernels, which is why weight-reading kernels can't be swapped" |
| 26 | transcript 2026-11-11T23:34 | "Ruled out: ... Kernels from other engines can't read strixw" |
| 27 | transcript 2026-11-11T00:29 / 23:01 | "No swap was a speed win"; overlap "narrower than the proposal assumed" (GEMV, prefill, GDN, HC, KV, speculators) |
| 28 | transcript 2026-11-12T00:47 | EXL3: "the problem the panel layout fixes ... doesn't exist here" (panel idea doesn't carry over) |
| 29 | transcript 2026-11-12T05:02 | Crucible "reads EXL3, not strixw, so its kernels can't be swapped into strixite ... a replacement engine rather than a parts donor" |
| 30 | `docs/PHASE5-GLM52-GAPMAP.md:386` | "the format and attention work does not carry over" |

Not counted (correct use of the word, or not a format argument): `docs/VISION-DSV41.md:148` (torch bits),
`docs/STRIXW-LAYOUT-NEXT.md:52,62` (B rejected on K divisibility, E on memory), `docs/PHASE5-FOURNODE-DESIGN.md:24`
(split c rejected on the numeric contract). `docs/PHASE5-ORNITH-GAPMAP.md:77` is the model to follow: "What carries
over is the idea, and it fits the file" (gufo's FF-slice TP re-expressed for four ranks on K-quant blocks).

## 2. Reframes

Measured inputs used throughout:
- strixite prefill kernel shares (PF-CF, pre-v0.5.0, pod report section 1): routed experts 32.7 %
  (`grouped_wmma_f16` 1,519 ms of 5,523 per 8,192-token chunk), GDN projections 17.9 %, HC 15.8 %, GDN core 13.0 %,
  attention 10.6 %, attention projections 5.5 %; wall ~6.7 s vs kernel 5.5 s per chunk.
- StrixFold experts pass (`docs/PREFILL-DSV41.md` section 3; `gate/results/exl3-pf-bytes/{a,b}.json`, median of
  repeats 2-5): as served 18.7 / 23.5 ms at 512 / 1,024 rows (3-bit); with trellis decode stubbed out 16.0 / 19.9 ms.
  So unpacking is ~15 %; the fit without decode is ~12 ms intercept + ~3.9 ms per 512 rows: **the fixed per-pass
  cost is mostly not decode and not bytes** (2-bit vs 3-bit saves nothing at 4,096 rows).
- Panel kernels (`gate/results/strixw-panel-kernels/README.md`): 489/489 byte-equal; prefill PASS (dense 67-99 % of
  origin time, experts 93.6-98.4 %); decode dense FAIL (108-190 % of origin time).
- qwen35moe executor: 35B Q4_K_M decode 56.4 tok/s moving ~1.8 GB a token, ~100 GB/s, half the 208-226 GB/s read
  peak (`docs/PHASE5-FOURNODE-DESIGN.md:105`); `kernel.flerp.moe_kquant` is gufo's `QuantWarpBlockDot`, one wave a
  row (`docs/INTERFACES.md` section 4 table).

### R1 (from 1, 3, 6, 17, 24-26) Group-affine GEMV shape onto K-quant blocks, for the qwen35moe executor
Source shape: `kernel.strixite.linear_q4` - one row streamed by 1-8 waves (split-K ~2,048 codes a wave), 16 B = 32
codes per lane load inside one group, `s*sum(x*q) + m*sum(x)` per chunk with `sum(x)` shared across rows, expert
gate and up rows of one expert computed together (`linear_q4.hip:97-121,162-165,244-276`). Fast because every lane
issues full 16 B loads and the min term costs one FMA per group, not per weight.
Target: Q4_K is the same math - 8 sub-blocks of 32 with 6-bit scale and min under fp16 `d`/`dmin`, 144 B per 256
weights (144 = 9 x 16, so 16 B loads stay aligned). Carries over: split-K mapping, min-fold, gate+up pairing, x-sum
reuse across the 8 experts of a token. Changes: unpack the 12-byte packed scales once per block per wave; Q6_K needs
its own unpack. New code: `kernel.flerp.moe_kquant_split` (Q4_K/Q6_K, `moe.experts-decode/gguf-kquant/f32.splitk`).
Gate: FIDELITY rule (order differs from gufo); at module level, output within fp32 order noise of `moe_kquant`.
Gain: if expert reads go from ~100 to ~170 GB/s, 35B decode 56 -> ~85-95 tok/s on one GPU; Ornith four-node decode
~13 tok/s rises by the expert share of its per-token time (inferred 1.3-1.5x). Risk: the 100 GB/s may be launch and
non-expert time, not the GEMV; profile first.
**Folded (2026-11-12, `gate/xfer-r1-fold.toml`, `gate/results/xfer-r1-fold/summary.md`).** Arm a (gufo's order, 16 B
loads) is now `moe_kquant`'s Q4_K route and, by the same recipe, its Q6_K route: bit-equal to the old kernel on every
35B and Ornith expert shape at 1-1,024 tokens, and the 35B gate prompts give P1's ids and logits sha256 at world 1 and
2 (loopback). Kernel GB/s at 1 token, old -> new (7 alternating rounds): 35B gate/up 131 -> 164, down Q6_K 139 -> 200,
down Q4_K 101 -> 144; Ornith gate/up 176 -> 196, down Q6_K 183 -> 211. 35B world-1 decode 57.0 -> 61.4 tok/s
(+7.8 %; two load-valid processes a binary, per-prompt spread about +-2 tok/s), which the bytes predict (expert reads 5.4 -> 4.1 ms of a 17.6 ms token). **Below the +20 % bar**: even
experts at read peak (~250 GB/s, ~2.2 ms) leave a ~14 ms token; the other ~12 ms (attention/GDN, dense
projections, router, launches, logits) are unprofiled and are where the rest of the 20 % has to come from. Split-K
(arm b) is not worth its order change at these shapes.

### R2 (from 6, 17, 22, 23, 28) strixite's expert scheduling onto StrixFold's EXL3 experts pass
Source shape: `kernel.strixite.moe_grouped` - one launch over (expert, slot-tile) work items, tile class picked by
slots (Narrow / Wide / Wide256, bit-identical to each other), codes staged packed into LDS with coalesced loads
(the "1.65x lesson", `moe_grouped.hip:352-353`); gufo `ds4_moe` groups rows by expert in 128 then 64.
Target: StrixFold `exl3_pf_experts` pays ~12 ms a pass that is neither trellis decode nor DRAM bytes (above),
across 392-471 jobs: ~25-30 us a job of fixed cost. Carries over: single persistent launch, slot-count tile classes,
per-job setup hoisted. Changes: tiles are EXL3 16x16 trellis tiles, so the K step and LDS staging are EXL3's.
Keep the per-output fp32 order -> bit identity with today's kernel (the gate `prefill_dsv41_bench.py` already checks
0 of 10.5 M outputs differing). Gain: halving the intercept saves 6 ms x 2 passes x 20 layers = 0.24 s of 1.46 s
kernel time per 2,048 rows, **~+20 % StrixFold prefill**, additive to the zip-two-chunks candidate only in part
(the zip removes one of the two passes). Risk: the intercept may be L2/occupancy, not job overhead.

### R3 (from 28, 29) The panel idea for EXL3 is about work mapping, not bytes
"Already tile-shaped" answered the byte question. The shape question: EXL3 decode is compute-heavy (15 % of the
experts pass even at prefill), and the panel layout's real idea is "one stored copy serves both the GEMV lane load
and the WMMA K-slice". Re-expressed for EXL3: decode each 16x16 trellis tile once per pass into an LDS panel shared by
all slot-tiles of that expert (today's 512-row pass decodes some experts' tiles more than once if they span tiles).
Gain: up to the 15 % unpack share of the experts pass (~0.1 s of 1.46 s, ~7 %). Gate: bit identity (decode is exact).

### R4 (from 1, 25, 26, 29) Crucible as a shape donor to strixite, not only a replacement
Crucible claims 1,831 tok/s cold 32K prefill on the fleet's model and hardware vs strixite ~1,220 (pod report
section 1) and upstream 1,370. Whatever it does differently in the experts pass, GDN chunked scan, HC or scheduling
is a shape that can be written for strixw (Q4 G128/G32 and Q4P16). Its EXL3 trellis decode does not transfer; its
work mapping, tile classes, fusion boundaries and chunk schedule all can. Gain: the ceiling is the whole gap,
**up to +50 %**; real value unknown until its kernels are read and timed by symbol. Risk: the claim is unmeasured;
part of its lead may be EXL3's smaller bytes (2-3 bit), which strixw Q4 cannot have.

### R5 (from 5, 18, 24) Integer-activation expert GEMM (hipfire i8-WMMA, gufo W8A8) for strixite and StrixFold
Shape: quantize x per 32 or 128 to int8 once, multiply raw codes on iu8 WMMA, apply weight scale/min and x scale in
fp32 after the integer accumulate. On gfx11 iu8 runs at the F16 rate (pod report row 9), so the win is not rate: it
removes the per-tile dequantize-to-F16 into LDS (strixite) and the per-weight float decode (EXL3 needs an integer
codebook form, harder). Gain: strixite's dequant share of `grouped_wmma_f16` is unmeasured; if it is like EXL3's
15 %, ~5 % of prefill. Gate: FIDELITY (activation quantization changes logits). Risk: fails the fidelity criteria.

### R6 (from 16, 27) Speculation shape for the qwen35moe executor (Ornith, Qwen3-Coder)
Shape: draft k tokens cheaply (n-gram/prompt lookup in hipfire and strixite's `ngram`; MTP where the checkpoint has
it), verify k+1 rows in one forward. Decode on 4 nodes is bandwidth-bound, so a 4-row verify reads the same ~weights
as 1 row (experts touched grow with distinct routes - the cost to measure). Carries over: draft/verify/keep_prefix
ABI (`docs/EXECUTOR-ABI.md`), already common to all four. Changes: multi-row path through `moe_kquant` (it already
takes any number of tokens) and the stage relay. Gate: greedy accepted tokens equal to the target's own greedy at
the same forward shape (SPEC point 3); FIDELITY for the verify rows. Gain: coding prompts with 1.5-2.5 accepted
tokens a step give ~1.4-2x on ~13 tok/s. Risk: expert union per verify erodes the bandwidth argument at 512 experts.

### R7 (from 3, 4) ggml mmvq for gfx1151
The llama.cpp `mul_mat_vec_q` shape (Q8_1 activations, dp4a-style integer dots) is in gufo for gfx1151 already;
hipfire's gfx1201 object is irrelevant. Transfer to the qwen35moe executor's dense K-quant GEMVs. Gain: small
(dense is ~3 % of Ornith weights, `PHASE5-ORNITH-GAPMAP.md:79`). FIDELITY gate.

### R8 (from 11) gufo's batch-invariant row arithmetic onto strixite
Shape: "each row's arithmetic identical to the one-row form (bit-exact across batch shapes)" (`ssm.hpp:26-29`).
strixite's tokens split across forward-shape classes (SPEC point 3: MTP on vs off differs from token 43). Writing
strixite's grouped/WMMA kernels to keep the one-row summation order per output gives **shape invariance**: a new
capability (MTP, chunk size and slot count stop changing tokens; every cross-shape gate becomes a hard check).
Cost: speed of the prefill kernels; possibly several %. Gate: tokens equal across shapes.

### R9 (from 12) HC: fused decode form and TP de-replication
gufo's fused HC decode ("byte for byte the separate launches") and hipfire's `qwen4_hc_down_fused` re-expressed on
strixite's Q4/Q8 folded-norm rows: fewer launches at decode (launch ~2.8 us each, SPEC point 5). In prefill HC is
15.8 % and replicated per TP rank (PF-CF); the shape to borrow is the FF-slice split of gufo's TP (R1's sibling).
Gain: prefill up to ~8 % on TP2 if HC is split, decode a few %. Gate: bit identity for fusion; TP split changes order.

### R10 (from 11) GDN chunked scan
hipfire's chunked WMMA delta rule with 8-row state chunks vs strixite `delta_rule_wmma` (514 ms, 9.3 % of a chunk).
Compare tiling and state residency; Q8 state is a FIDELITY question. Gain unknown, likely a few %.

### R11 (from 14) Per-channel K8 KV for the qwen35moe executor and Crucible-style long context
strixite `kv8` (K per channel over 32 rows after norm+RoPE, unquantized tail) re-written for the GGUF executor's
F16 KV: halves KV bytes, a capability for long contexts on Ornith (2 KV heads makes KV small already, so the gain is
mostly for Qwen3-Coder long prompts). FIDELITY gate.

### R12 (from 7, 8, 13, 15, 21) Router, norm, RoPE, sampler
Re-expressible (e.g. a k=10 batched softmax-top-k in hipfire's shape; StrixFold's Triton-bit router as HIP), all
under 2 % of time. Low value; recorded so they are not re-closed as "impossible".

### R13 (from 9, 10, 19) Indexer and QSA
The shape (pooled-key scores, exact top-k with a tie rule) transfers; StrixFold's FP4 keys are already the
cheapest form. hipfire's sampled-threshold selection could cut the unmeasured top-512 cost at long context.
Value only past 64K context (4-8 % of a chunk, `PREFILL-DSV41.md` candidate 6).

### R14 (from 20, 30) mHC 4x4096 vs 4x5120; GLM format work
gufo's fused norm+mix+split pre-block from 128 rows is a fusion shape that fits 5120 as well as 4096 (SPEED-FUSE-1,
~6 % of StrixFold prefill). GLM-5.2 is withdrawn; the latent-attention item stays parked.

### R15 (from 2, 5) bf16 GEMM
hipBLASLt-plan order for bf16 weights; no current consumer. Parked.

## 3. Ranking

| rank | idea | target | expected gain | cost | why |
|---|---|---|---|---|---|
| 1 | R4 Crucible shape read | strixite prefill (+decode) | up to +50 % prefill (gap 1,220 -> 1,831) | read + symbol bench, days | largest ceiling, same model and GPU |
| 2 | R2 strixite scheduling into EXL3 experts | StrixFold prefill | ~+20 % | one kernel, bit-identity gate exists | the 12 ms intercept is measured and unexplained |
| 3 | R1 split-K group-affine GEMV on Q4_K | Ornith / Qwen3-Coder decode | 35B 56 -> 85-95 tok/s; Ornith 1.3-1.5x | one kernel, one GPU | half of read peak is on the table |
| 4 | R6 speculation in the GGUF executor | Ornith four-node decode | 1.4-2x on coding | executor work | four-node decode is the slowest thing the fleet serves |
| 5 | R8 batch-invariant rows | strixite correctness | capability | kernel rework, speed risk | turns every shape gate into a hard check |
| 6 | R3 decode-once EXL3 tiles | StrixFold prefill | ~+7 % | one kernel | bounded by the measured 15 % unpack |
| 7 | R9 HC fused/split | strixite decode, TP prefill | few % / ~8 % | moderate | 15.8 % share |
| 8 | R5 integer-activation experts | strixite prefill | ~5 % | moderate + fidelity risk | dequant share unmeasured |

Not shortlisted: R7, R10-R15 (each under ~3 % or needs a model the fleet does not run). The panel layout itself
(Q4P16) is already in flight: its measured prefill kernel gain (dense 67-99 %, experts ~94 % of origin time at the
fleet's M8192, about 3-4 % of prefill) is below the design's inferred 9-13 %, and its decode kernels must first reach
parity; StrixFold's load-time "strips" (one wave = one contiguous run, `docs/PREFILL-DSV41.md`; transcript 00:47) is
the work-mapping shape the panel decode GEMV should copy.

## 4. Shortlist: first experiments

1. **R4 Crucible.** Once its weights land: serve it on control-node's GPU, rocprofv3 kernel trace of one cold 32K
   prompt, beside the same trace of strixite (pod report section 4 step 1). Produce the PF-CF stage table for both.
   Continue on any stage where Crucible is >= 1.3x faster per token; that stage's kernel is the next read.
2. **R2 EXL3 intercept.** control-node GPU, `gate/exl3_pf_bytes.py` driver at 128/256/512/1,024/2,048 rows, arms a/b,
   plus a rocprofv3 trace counting launches and per-job durations. Continue if the per-job fixed cost (intercept /
   jobs) is >= 15 us and is launch or setup time a single persistent launch removes.
3. **R1 K-quant split-K.** control-node GPU, microbench `moe_kquant` against a split-K Q4_K expert GEMV at the 35B
   shapes (256 experts, 8 used, width 512, K 2,048) with real `Qwen3.6-35B-A3B-Q4_K_M.gguf` stacks; measure GB/s.
   Continue if >= 160 GB/s and the recipe's decode rises >= 20 % under the FIDELITY criteria.
4. **R6 speculation.** No GPU first: replay the qwen35moe gate prompts' recorded greedy outputs through a prompt-lookup
   drafter and count accepted tokens per step; then one GPU: time 1/2/4/8-row forwards of the 35B. Continue if
   accepted >= 1.5 a step and a 4-row forward costs <= 1.6x a 1-row forward.
5. **R8 shape invariance.** One GPU: run `grouped_wmma_f16` and `linear_wmma` at M = 1, 4, 12, 2,048 on the same rows
   and count outputs differing from the one-row result; list which kernels break it. Continue if the breakers are
   few kernels whose one-row order costs <= 5 % at prefill.
6. **R3 decode-once.** Count per pass how many times each EXL3 tile is decoded (instrument the a-arm kit). Continue if
   > 1.2 decodes per tile at 512-1,024 rows.

## 5. Proposed rewording (not applied)

- `SPEC.md:16-17`, section 1 point 2 (owner: "yeah, the weight format is baked in, but that doesn't mean you can't
  use the same recipe later with a different size pan"), replace with:
  "2. The weight format is baked into every compiled weight-reading kernel: strixw Q4/Q8 groups, EXL3, ggml K-quants,
  hipfire MQ/MFP4 and FP8 each fix a different inner loop, and a kernel only reads the format it was written for.
  Its recipe is not baked in: the algorithm, tiling, work mapping and scheduling are reusable and get re-cooked for
  another format, a different size pan. That re-cooking is what cross-engine use means in strixflerp, gated by bit
  identity where the math is unchanged and by the fidelity rule (`docs/FIDELITY.md`) where it is not
  (`docs/SHAPE-TRANSFERS.md`)."
- `docs/SOURCES.md:23` title: "Each compiled weight-reading kernel reads one format (shapes transfer)".
- `docs/INTERFACES.md:65-66`: "not a drop-in across formats (a kernel binary reads one format). The shape - split-K
  mapping, min-fold, gate+up pairing - transfers; see `docs/SHAPE-TRANSFERS.md` R1."
- `docs/INTERFACES.md` verdict vocabulary: keep "swappable as is / with adapter" for drop-ins, and replace bare
  "not swappable" with "not a drop-in; shape transfer: <one line> (SHAPE-TRANSFERS Rn)" at :110, :132, :184, :198,
  :224, :237, :280.
- `docs/INTERFACES.md:239-241`: "The storage idea that travels is strixite's (per-channel K ranges after norm + RoPE);
  it is a candidate for the GGUF executor's KV (SHAPE-TRANSFERS R11)."
- `docs/PREFILL-DSV41.md:148` column header "swappable into this recipe?" -> "drop-in? / shape to transfer", with the
  experts row reading "not a drop-in; strixite's single-launch tile classes onto the EXL3 pass (R2)".
- `docs/PREFILL-DSV41.md:159` and `:249-251`: "Nothing in the 70 % is a drop-in part; the shapes worth transferring
  are in `docs/SHAPE-TRANSFERS.md` (R2, R3)." and drop "because the hot kernel's weight format is baked in".
- pod report `2026-11-11-strixw-prefill.md` row 10 (pod write, for the owner or the steward): "not drop-ins; the
  shapes to transfer are hipfire's i8-WMMA expert route (R5) and Crucible's prefill (R4)".
