# 61208c12ad verdict: MoE combine with 16-byte loads

VERDICT: CONFIRMED
GATE: PASS

The original combine kernel was bound by its launch shape (one 4-byte load a thread), not by memory: the same expression
with 16-byte loads ran at 3.9x the original in the kernel bench with every output word equal. In the model, combine plus
ToF16 fell from 348 to 109 ms a 4,096-row chunk (predicted ~115, confirm line 150), output stayed identical, and prefill
rose 10 % at 2,048 tokens and 13 % at 16,384.

## Gate (from `tools/flerp-research judge`)

| metric | rule | measured | result |
|---|---|---|---|
| combine_plus_tof16_ms | <= 150 | 109.37 | pass |
| c0_words_differ | == 0 | 0 | pass |
| c2_mismatches | == 0 | 0 | pass |
| c3_ratio_2048 | >= 1.05 | 1.1004 | pass |
| c3_ratio_16384 | >= 1.05 | 1.1295 | pass |
| c4_scratch_bytes | == 0 | 0 | pass |

## Numbers

| | baseline (arm G) | candidate (arm C) | change |
|---|---|---|---|
| combine, ms a chunk | 279 | 70.4 | -75 % |
| ToF16, ms a chunk | 69 | 39.0 | -44 % |
| prefill tok/s, 2,048 tokens (median of 3) | 1,979.3 | 2,178.0 | +10.0 % |
| prefill tok/s, 16,384 tokens (median of 3) | 1,889.1 | 2,133.7 | +12.9 % |
| kernel bench GB/s, 35B dims (4,096 x 8 x 2,048) | 48.2 | 188.7 | 3.9x |
| kernel bench GB/s, larger dims (4,096 x 10 x 4,096) | 51.4 | 202.9 | 3.9x |

Identity: 108 recorded steps, 0 logits sha256 or generated-id mismatches; 0 of 8,388,608 and 0 of 16,777,216 output
words differ in the kernel bench. ISA: 0 scratch bytes with both toolchains (74 and 75 VGPRs).

## Limits

The limits in force (card.md LIMITS) and how often each one was hit.

| limit | value | hit |
|---|---|---|
| model reps | 3 interleaved reps a prompt length, median | 3 of 3 reps ran at each length and arm; the median left out arm G's slow first 2,048-token rep (1.249 s against 1.035 and 1.030) |
| kernel bench reps | 7 a dim, caches flushed | 7 of 7 ran at both dims |
| speed bar | 1.05x at 2,048 and 16,384 tokens | 0 of 2 lengths under it (1.100x and 1.129x) |
| prediction lines | confirm at 150 ms or less, refute above 250 ms | 0 crossings of the refute line; 109.37 ms is inside the confirm line |
| identity bar | 0 differing words, sha256 or ids | 0 differences in 108 recorded steps and 25,165,824 output words |
| generated tokens | 2 a prompt | 54 of 54 traced prompts stopped at the cap, as designed |
| context | 16,640 tokens | 0 prompts truncated (longest 16,384 ids) |
| sampling | greedy | 54 of 54 traced prompts greedy (argmax is the token) |
| trace warm-up | passes 1-2 dropped | 2 of 5 profile passes dropped |
| hang guard | 3,000 s process timeout | 0 (the model gate ran 87.7 s) |

## Premises found wrong

None: the cause bench put the cost in the kernel's launch shape, as the mechanism said, not in memory or in the writer
before it.

## Caveats

- Model numbers are from the 35B proxy recipe; the larger dims were run only in the kernel bench.
- The kernel trace's first two profile passes are warm-up and excluded (passes 3-5 measured); the arm G 2,048-token reps
  include one slow first rep (1.249 s against 1.035 and 1.030), and the median is used.
- No champion change: the promotion proposal was withdrawn and the lever enters the prefill champion through a later
  experiment.

## Evidence

`results/`: `cause.md` and `k-cause1.result.txt` (kernel bench), `verdict.json` (all checks), `gate-verdict.json` (model
gate identity and speed with per-rep times), `cprof-summary.md` (profile by kernel class), `isa-*.tsv`, `summary.json` (the
metrics judge reads), `MANIFEST.sha256`. The raw kernel trace and gate JSON lines are in `gate/results/moe-combine/`.
Merge: `flerp-research merge-plan` lists this folder and `catalog/flerp/executor/qwen35moe` (CONFIRMED).
