# PREFILL-ORNITH results (card 8ef0a645e7)

Criteria: `gate/prefill-ornith.toml`, committed **ad8ea24 before any GPU run**; thresholds never loosened. One
amendment to that file, 21dc46f, also before the first fidelity run: the KL method wording (a side keeps a top 1,000,
so a union cannot be scored); no threshold changed.

Every number below is read out of a committed file in this directory; the file is named on each line. Nothing here
is re-measured by hand, and no criterion was re-judged after the fact.

| group | case | verdict | evidence |
|---|---|---|---|
| kernel | K1, K2, K3, K4 | **PASS** | `check-c1.jsonl` (+ `check-c1.result.txt`, job `20261114T023120Z-run-x`) |
| kernel | exact variant, same K1-K4 | PASS | `exact-x1.jsonl` (job `20261114T033240Z-run-x`) |
| speed | S1, S2, S3, S4 | **PASS** | `speed-s1.jsonl` (old path, `20261114T031514Z-run-x`) + `speed2-s2.jsonl` (new default order, `20261114T031923Z-run-x`) |
| projection | P1 | **PASS** | computed below from those two files and the P4 baseline |
| fidelity | F1 | **FAIL** | `35b-r2-verdict.json` |
| fidelity | F2 | **FAIL** | `35b-r2-verdict.json` |
| fidelity | F3 | **FAIL** | `35b-r2-verdict.json` |
| fidelity | F4 | **FAIL** | `35b-r2-verdict.json` |
| fidelity | F5 (count + report) | reported: **0** fallbacks | `35b-r2-verdict.json` |
| fidelity speed | T1 | **PASS** (4.10x; 4.47x on the 2,048-row speed pass) | `35b-r2-verdict.json`, `35b-speed3.txt` |

So: the kernels are correct and fast (S1-S4, K1-K4, T1 all pass); the model-level fidelity of the F16 grouped arm
does **not** meet the committed bounds, and the card's deliverable is the measurement plus the fix staged for it
(arm E, exact codes), not a turned-on path.

## 1. Kernel correctness (K1-K4) — PASS

Against a float64 reference written from llama.cpp's `dequantize_row_q4_K` / `q6_K` (no code shared with the kernel),
hot routing (4 experts in every token's top 10, so groups cross every BM's tile boundary), 192 tokens, 64 checked, all
10 slots plus the shared expert. `check-c1.jsonl`:

| case | rel_rms | max abs / rms(ref) | non-finite | limit |
|---|---|---|---|---|
| routed, down Q6_K | 5.10e-4 | 3.12e-3 | 0 | K1: 2e-3 / 2e-2 |
| shared, down Q6_K | 5.08e-4 | 2.89e-3 | 0 | K1 |
| routed, down Q4_K | 5.09e-4 | 3.21e-3 | 0 | K1 |
| shared, down Q4_K | 5.09e-4 | 2.78e-3 | 0 | K1 |
| Linear (12288, 4096) Q6_K | 2.94e-4 | 1.47e-3 | 0 | K2 |
| Linear (8192, 4096) Q4_K | 2.92e-4 | 1.54e-3 | 0 | K2 |
| Linear (4096, 8192) Q4_K | 2.95e-4 | 1.35e-3 | 0 | K2 |

- **K1 PASS** (routed + shared, both down types: 5.1e-4 <= 2e-3, max 3.2e-3 <= 2e-2, 0 non-finite).
- **K2 PASS** (three dense shapes, <= 3.0e-4).
- **K3 PASS**: `differing_floats_across_bm` = 0 and `differing_floats_across_cut` = 0 in both routed cases — outputs
  are bit-identical across BM 32/64/128 and across a 69 + 123 chunk cut.
- **K4 PASS**: `overflow` = 0 in all 46 check and speed records of the three K jobs (the `--check-dense` records carry
  no overflow field; the dense kernel has no F16 range check by design — its output is FP32).
- **Record** (`[kernel_correctness] record`, not a criterion): the current path against the same reference is
  rel_rms 2.4-2.9e-7 — i.e. the grouped path's error is ~1,800x the current path's, which is what F16 rounding of
  weights and x costs (docs/PREFILL-ORNITH.md section 1). The current path is itself not the reference.
- Exact variant (`exact-x1.jsonl`): rel_rms 6.4-7.1e-7 routed/shared, 3.1-5.3e-7 Linear — within 3x of the current
  path's own error, with the same K3 bit-invariance and 0 overflow.

## 2. Kernel speed (S1-S4) — PASS

Old = the current path's kernels in the same binary, arithmetic unchanged (1,024 rows reads ~83 GB a pass, so it is
bandwidth-bound: 198-215 GB/s requested). New = grouped, the column-fast grid order, which is the default in the
committed kernel (`moe_kgrouped.hip:673`); the tiles-fast order is kept only as `FLERP_KG_ORDER=0` for the comparison.
Uniform routing, down Q6_K (the heavier half), best BM per row count; `speed-s1.jsonl` (old, and new before the order
default was switched) and `speed2-s2.jsonl` (new default order, dense).

| rows | old routed ms | new routed ms (BM) | speedup | new TFLOP/s | old TFLOP/s | new shared ms |
|---|---|---|---|---|---|---|
| 1,024 | 402.318 | 30.676 (32) | **13.1x** | 8.4 | 0.64 | 1.416 |
| 4,096 | 1,628.992 | 55.303 (128) | **29.5x** | 18.6 | 0.63 | 3.824 |
| 16,384 | 6,518.526 | 178.509 (128) | **36.5x** | 23.1 | 0.63 | 15.881 |

The grid order is worth 1.5-2.2x (tiles-fast: 44.594 / 97.725 / 385.215 ms; 1.45x, 1.77x, 2.16x); at 16,384 rows the
old order misses S2 (10.7 < 20 TFLOP/s).
Skewed routing at 4,096 rows (`--skew`, `speed-s1.jsonl`, new order at the time): 108.388 ms — padding to tile
boundaries, not load imbalance, is what costs there.

- **S1 PASS**: 13.1x >= 10x at 1,024 rows (routed pass = group + F16 x + gate/up + down, against the current path's
  gate/up + down).
- **S2 PASS**: 23.1 TFLOP/s >= 20 at 16,384 rows.
- **S3 PASS**: 18.6 TFLOP/s >= 12 at 4,096 rows.
- **S4 PASS**: Linear (dense), best BM, all >= 20 TFLOP/s —
  (12288, 4096) Q6_K: 25.3 at 4,096 rows, 24.1 at 16,384 (BM 128);
  (4096, 8192) Q4_K: 28.2 at 4,096, 28.4 at 16,384 (BM 128). Also measured: (8192, 4096) Q4_K 29.3 / 27.2,
  (16384, 4096) Q4_K 29.8 / 28.1.
- Load and temperature were recorded per case by the kbench backend (`*.result.txt`: K loadavg at start and end, Tctl
  54-66 C peak on the speed jobs; the check job waited 2,531 s in the queue and ran under loadavg 8-11, which affects
  absolute rates, not the new/old ratios measured in the same jobs).
- Profile (`profile-p1.result.txt`, `20261114T032032Z-profile-x`): the grouped kernels carry 0 scratch (no spills),
  144-256 VGPR, 23.5-38.9 KiB LDS; the in-place code reads are not a separate stall source at these shapes.
- Exact variant: 37.789 / 107.965 / 364.004 ms = 10.6x / 15.1x / 17.9x the current path (BM 32 / 32 / 64) — about
  1.2-2.0x the cost of the F16 grouped arm, still 10x+ the current path.

## 3. Projection (P1) — PASS

`rest` per the toml = 472 ms (P4 baseline, `gate/results/ornith15-397b.p4/orn-0.jsonl`) - the current path's measured
routed + shared at 1,024 rows (402.318 + 21.307 = 423.625 ms, `speed-s1.jsonl`) = **48.4 ms** a block.
`stage ms = 15 x (new routed + new shared + rest x rows / 1,024)` (`speed2-s2.jsonl`, experts only, dense unchanged):

All columns are **per block** except the last two (stage = 15 blocks):

| rows a chunk | new routed + shared | rest (48.4 x rows/1,024) | a block | stage ms | rows/s a stage |
|---|---|---|---|---|---|
| 1,024 | 30.676 + 1.416 = 32.1 | 48.4 | 80.5 | 1,208 | **848** |
| 4,096 | 55.303 + 3.824 = 59.1 | 193.6 | 252.7 | 3,790 | **1,081** |
| 16,384 | 178.509 + 15.881 = 194.4 | 774.3 | 968.7 | 14,531 | **1,128** |

- **P1 PASS**: 1,081 rows/s at a 4,096-row chunk >= 1,000 (7.5x the 144 rows/s measured baseline).
- Projected, not measured end to end: it uses a measured baseline, measured kernel times and the 472 ms P4 block, and
  assumes the rest of the block is linear in rows and that dense/GDN/attention/norms are untouched. The rest now
  dominates (77 % at 4,096 rows), which is why step 2 (dense on Linear, bulk chunks) is the next lever; with Linear at
  the measured 24-30 TFLOP/s the same arithmetic reaches ~2,700-3,300 rows/s (estimate, docs section 5).
- End to end (estimate, labelled): ~4 pipelined stages run at about the slowest stage's rate; attention at depth is
  added by FLOP count only. Not measured on Ornith - that needs the 4-node window.

## 4. Fidelity (F1-F5) — F1, F2, F3, F4 FAIL; F5 reported

35B proxy (qwen36-35b-a3b Q4_K_M, one GPU, control-node's gpu-queue), same executor binary: A = `FLERP_MOE_GROUPED=0`
(current path), B = `=1` (grouped F16), D = `--decode-prompt` (FP32 activations, the FIDELITY.md yardstick). 8 prompts
x 32 positions = 256 positions. `35b-r2-verdict.json`, `35b-r1-kl-by-prompt.txt`, block checks
`35b-blockcheck-{A,B}.json`, FP32-reference judge `35b-fp32-verdict-ref-D.json`.

| criterion | bound (committed) | measured | verdict |
|---|---|---|---|
| F1 mean KL(A\|\|B) | <= 0.002 | **0.009613** | FAIL (4.8x) |
| F1 max KL(A\|\|B) | <= 0.02 | **1.3936** | FAIL (70x; prompt 6) |
| F2 top-1 agreement | >= 98 % | **96.48 %** (247/256) | FAIL |
| F2 every miss at A's top-2 margin < 0.3 nats | 0 exceptions | 2 misses at 0.984 and 0.438 | FAIL |
| F3 mean KL(D\|\|B) | <= 1.25 x 0.008864 + 0.0002 = 0.011280 | **0.012497** | FAIL |
| F4 greedy ids equal, 64 tokens | >= 7 of 8 prompts | **1 of 8** | FAIL |
| F5 overflow fallbacks | count and report | 0 traced, 0 greedy processes | reported |

Reference points, both measured, neither a criterion:
- The current path is itself **0.008864** mean KL from the FP32 yardstick D (F1's bound is 0.002 against the current
  path): the F16 grouped arm is 0.0125 from D, i.e. it adds about 0.0036 nats on top of an existing 0.0089.
- Judged directly against D (`35b-fp32-verdict-ref-D.json`): mean KL(D\|\|A) 0.017562 / KL(D\|\|B) 0.019295; top-1
  97.27 % for both; greedy equal 3 of 8 for A, 2 of 8 for B. (Different KL direction and cover set from F1/F3, so the
  two tables do not add up to each other.)
- The per-prompt KL is dominated by one prompt (mean 0.047, max 1.394 on prompt 6; the rest are 0.0009-0.0098), and
  greedy divergences cluster at positions where A's own top-2 margin is small — so this is rounding amplified by
  decoding, not a wrong result on some blocks.
- `35b-blockcheck-{A,B}.json`: an independent numpy check of the same block arithmetic. The grouped kernel matches a
  float model of itself (F16 weights + F16 x + FP32 accumulate) to 0.2 % of rms median. So this is **F16 rounding, not
  a bug in the kernel**: the arithmetic does exactly what the design says, and what the design says is not faithful
  enough at the committed bounds.

**T1 PASS**: 2,048-token prompt, median of 3 interleaved A/B reps in one load — A 6.935 s, B 1.692 s = **4.10x**
(>= 3x); the separate speed pass (`35b-speed3.txt`) gives 6.90 -> 1.54 s = 4.47x, and 3,000 tokens 10.23 -> 2.56 s =
4.0x.

## 5. What is not passed, and what is staged

- The grouped F16 arm (B) meets every kernel and speed criterion and fails F1-F4 as committed. It stays **off by
  default** (`FLERP_MOE_GROUPED` unset = current path); nothing turns it on.
- The committed fix is the exact arm E (exact F16 weight codes with scale/min in the FP32 accumulator, hi/lo F16
  activations, `FLERP_MOE_GROUPED=2`, commit 7ac7975): kernel error 6.4e-7 instead of 5e-4, 10.6-17.9x the current
  path. Its F1-F4 verdict on the 35B is child card **a74fe1bdc1** (arms A, B, E against D), with the hypothesis
  recorded there: KL(D\|\|E) at or below KL(D\|\|A). If E also fails F1 against A, the honest outcome is that grouped
  prefill is not faithful enough to turn on at these bounds, and the 2-3x matrix cost is the reason.
- Not done in this card (docs section 8): dense projections on Linear (step 2), bulk chunks with the stage arena, the
  Ornith 4-node window (owner's go), and a panel repack if it were ever needed.
