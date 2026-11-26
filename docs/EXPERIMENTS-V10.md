# strixflerp experiment package v10 (pre-registered)

Card FLERP-V10 f0609bca0f, parent STRIX-META bd830ae1ca. Supersedes the v9 draft (owner's #owner-room post
2026-11-17 00:52Z, media `8ab528e7...`) and folds in every item of the owner's six-trap review of v9 (01:00Z).
The five gate files below were committed **before any hardware run** of this package. They are never loosened after
a run; an amendment is added below its original line with a date and a reason.

The v9 thresholds were guesses. Where reading the mechanism gave a better prediction, v10 states that prediction and
keeps v9's guess in the TOML as `v9_guess`, so both can be checked against the result.

## What is already measured (not re-run)

Card SFX-PKG 1fd5fa6482 (agent, report `/data/pod/reports/2026-11/2026-11-16-strixflerp-package.md`):

| item | result | effect on v10 |
|---|---|---|
| Panel p16 whole model | dequant byte-identical, greedy ids identical 8/8 prompts; prefill 1.046-1.112x, decode 1.003-1.005x; promoted `qwen38-fnext.prefill-best` | closed |
| Fused residual/norm/gate | 288 launches a token, 192 removable, 2.21 us a launch vs 0.82 us grid barrier: 0.27 ms of a 30.2 ms step (0.9 %) | closed, not built |
| QSA cross-layer recall | adjacent indexer layers, decode: 0.727 at 8K, 0.647 at 32K (random 0.250 / 0.062) | exp 4 records token-to-token, which this did not |

## Facts that shape v10

- **The control GPU queue never starts a job today.** `tools/gpu-queue.sh` waits for GTT use under 1 GiB.
  llama-swap keeps `qwen38-27b` loaded (`ttl: 0`, about 43 GB GTT) on control-node. Bench box K, through `pod-kbench`
  (queue-only, exclusive), is the route for every GPU arm in v10 except exp 4, which runs on a worker. K is the same Strix Halo part (gfx1151, 128 GB).
  Experiment 1 therefore measures K's floor. A control-node arm is added only if the owner frees that GPU.
- **Experiment 3 has no hidden states on disk.** The draft_coarse run kept per-arm counters only. Those counters
  already split part of the question: on the q2 arm the true argmax fell outside the kept blocks on 21 of 6,144 steps,
  yet the global margin certified only 1. The coarse copy is good; the test is weak. A GPU capture run comes first.
- **QSA pins nothing.** `qsa_topk` returns the min(k, blocks) highest-scoring blocks. The query's 0-3 tail tokens are
  attended outside k. SFX-PKG found sink and newest blocks are about 0.2 % and 3 % of a selection by content, not by
  rule. Exp 4 still reports Jaccard with and without the first 4 and last 16 blocks, as the review asks.
- **Loopback cannot judge pipelining.** `gate/ornith-subchunk.toml` already records that four ranks on one GPU cannot
  overlap stages. Exp 5 measures each stage's compute alone and builds the four-node makespan from those times.
- **The 35B is 256 experts, 8 used** (GGUF `expert_count`, `expert_used_count`). Exp 2 and 5 read both from the
  header at run time rather than from this text.
- **GPU telemetry on K** (sysfs, card0): `gpu_busy_percent`, `gpu_metrics` (gfxclk, socket power, temperatures, DRAM
  read and write counters where the metrics version carries them), `pp_dpm_sclk`/`fclk`/`mclk`, hwmon `power1_average`,
  `temp1_input`, `freq1_input`. No `rocm-smi` bandwidth query is relied on. A counter that reads N/A is recorded as N/A
  and judges nothing.

## Order

1. Exp 1, noise floor (K). Its spread sets the rep counts of 2 and 5 by the formula in its TOML.
2. Exp 3 capture (K) and exp 4 (one worker window), independently. Then exp 3 replay (CPU, no GPU).
3. Exp 2, many streams (K).
4. Exp 5, sub-chunk stage sweep (K).

K's kbench queue runs one job at a time, so nothing contends with exp 1's quiet arm.

## The review's six traps and where each landed

| review item | fix | file |
|---|---|---|
| 1 Exp 2: expert fan-out | 4 distinct pinned prompts; unique experts per layer per step logged; prediction from a byte model that includes the union | throughput-35b.toml |
| 1 Exp 2: prose prompt length | prompt_tokens = 2048, gen_tokens = 64; prefix cache off and KV pages asserted disjoint | throughput-35b.toml |
| 1 Exp 2: 1-ULP flip cascades | exact-id match judged free-running; logit deltas measured teacher-forced on the single-stream ids | throughput-35b.toml |
| 1 Exp 2: stop vs identity fail | stop_rule names runtime/HIP faults only; identity mismatch dumps diffs and continues | throughput-35b.toml |
| 2 Exp 3: Frobenius trap | block bound = max over the block's rows of the exact row norm ‖w_i − ŵ_i‖₂ (per-row norms stored) | draft-coarse-blockbound.toml |
| 2 Exp 3: activation quantization | bound and reference both use the x the kernel actually multiplies (its quantized x̂ if it quantizes), plus a ‖w‖‖x − x̂‖ term otherwise | draft-coarse-blockbound.toml |
| 2 Exp 3: disk check first | done before writing: not on disk; capture run is step 2 | this doc |
| 3 Exp 5: loopback hides bubbles | per-stage compute measured alone (world 1, one quarter's layer range); four-node makespan reconstructed | prefill-subchunk-sweep.toml |
| 3 Exp 5: outer chunk fixed at 4K | outer call = 16,384 rows for every sub-chunk size; M = 16384 / rows recorded | prefill-subchunk-sweep.toml |
| 3 Exp 5: expert normalization | knee reported in rows per expert (rows × used / experts), both models' values from their headers | prefill-subchunk-sweep.toml |
| 4 Exp 4: gate logic | two verdicts, cost and stability, each with its own confirm and refute | qsa-index-stability.toml |
| 4 Exp 4: 64K coverage, sync | contexts 4K, 16K, 64K tokens (1,000 / 4,000 / 16,000 blocks) of real decode; selections in a preallocated device buffer copied once at exit; timing from a separate unlogged run | qsa-index-stability.toml |
| 4 Exp 4: pinned blocks | no pinned blocks exist; reported with and without first 4 / last 16 anyway | qsa-index-stability.toml |
| 5 Exp 1: gfx1151 telemetry | sources above; gfxclk, fclk, mclk, socket power, temps logged per rep; stop on a gfxclk drop | noise-floor.toml |
| 5 Exp 1: load per rep | load < 1.0 checked at every rep boundary, not only arm ends | noise-floor.toml |
| 5 Exp 1: range statistic | range/median, IQR/median and CV recorded; judged on CV; the reps formula is stated | noise-floor.toml |

## Not in this package

| item | where it stands |
|---|---|
| Panel layout and kernel | promoted (SFX-PKG) |
| Fused residual/norm/gate kernel | 0.9 % of a step, not built (SFX-PKG) |
| Cross-layer index reuse as a kernel | recall 0.65 at 32K; reopened only if exp 4's cost verdict says the indexer is worth optimizing |

## Open before each run card starts (resources, not criteria)

- Exp 4 runs in a WINJOB window on one worker (C or D), not on K (owner 2026-11-17: "use a different node"). K's
  `/dev/shm` holds 66.8 GB, under the 74.6 GB file. Every worker already holds it at
  `/cache/models/qwen38-fnext-tpn/weights.strixw`, so nothing is staged.
- Exp 1, 2 and 5 use the 35B on K. The run cards keep it placed across consecutive jobs instead of evicting and
  restaging between them.
