# Flash-Next on Strix Halo for the software factory: an ordered experiment roadmap

agent, 2026-11-17 (10:10 AM PT). Asked by owner: review the StrixFold modules of strixflerp, the board and
factmap, then the new strixflerp modules (harvest where needed), and give an ordered set of experiments. No hardware
was used for this report. Nothing here is carded yet.

"The model" is taken to be Qwen3.8-Flash-Next, the model every FreeFitty thread runs on the five workers.

## 1. What was reviewed

- strixflerp catalogue: 35 StrixFold modules + 6 of PR #6, the 58 flerp modules, champions ledger, EXPERIMENTS.md,
  v10 package (docs/EXPERIMENTS-V10.md on card/f0609bca0f) and its five reports, the sub-threshold ledger,
  decode-launch-design.md, chain1.md, hc-q4.md, strix-zport.md, strixite-slots.md and -2, strixite-perf-ranking.md,
  flashnext-vs-published.md.
- Board: STRIX-META bd830ae1ca (76 open cards), FLERP-SWAP 259695a55c, SF-FF dc5772a82f, the v10 run cards.
- factmap: 24 strixflerp and 51 StrixFold articles (top 10 of each read).
- New origin, harvested today: radiance libhalo, card ORIGIN-RADIANCE b0c1db5993, branch card/b0c1db5993 @
  ad84e82e6071, now in review. 25 modules, 188 files. The pod had spent two FreeFitty rounds on it with a survey
  and no catalogue output. Owner's other 11 kernel sources (KSRC-SURVEY f7de2c34b1) are surveyed on the pod
  (ksrc-rows.tsv) and not harvested; the report for that card is still owed by the pod.

## 2. The finding that sets the order

The factory is short of slots, and most single-stream levers are spent.

Live, 2026-11-17 17:55Z (FreeFitty /health): four workers up, 2 slots each, all 8 busy, 12 to 14 threads waiting on
each worker (54 waiting in all). Worker D was held.

Worker B's lifetime counters (strixite /metrics, 1,122 requests):

| per request | tokens | engine time |
|---|---|---|
| prompt | 27,600 (61.5 % served from cache) | |
| prefilled | 10,600 | 10.9 s (24 % of engine time) |
| generated | 797 | 35.1 s (76 %) |

- Decode: 22.7 tok/s per slot, about 45 tok/s per worker. One request alone on the same engine decodes at 66 to 72
  tok/s (flashnext-vs-published.md, code prompts).
- Two slots give 1.17x one slot (strixite-slots.md: 48.0 vs 41.7 tok/s). The byte model allows 1.35x.
- Each slot-step commits 2.41 tokens from 2.90 rows (1 real + 1.90 drafts). The first draft is accepted on 90.4 % of
  steps. Drafts after the first return 0.56 tokens a row.
- A decode step at one row is 32.9 ms: dense 8-bit weights 10.6, routed experts 7.1, hyper-connection mixes 4.1,
  GDN input projections 3.9, the rest 3.9, launch gaps 3.2. Only the experts must be re-read per row (measured 6 ms a
  row). The other 26 ms could be read once for every row in a step.

So the arithmetic says rows are cheap and the fleet spends them badly: 2 slots with deep drafts, where the same
step could carry 4 or more threads. A step of 8 rows should cost about 26 + 8 x 6 = 74 ms plus the draft heads. The
measured two-slot step is about 140 ms for 5.8 rows. That gap, and the slot count, are the two largest levers on
the board. Neither is carded for Flash-Next (V10-STREAMS 19aa2f3c03 asks the question on the 35B proxy only).

Second: 10,600 prefilled tokens a request is far more than a tool result. Some of it is recompute after a thread
loses its slot. Nobody has measured how much.

What the single-stream work has returned in the last three days, for scale: panel p16 +5 to 11 % prefill, PF-13
+1 to 2 % decode, FOLD-1 +1.5 % decode. Chained launches, HIP graphs, index reuse, the coarse draft head bound,
2,048-row sub-chunks, the Z prefill ports and Q4 hyper-connections were each built or bounded and closed.

## 3. What each source contributes

| source | shape worth carrying to Flash-Next | used in |
|---|---|---|
| runtime.strixfold.lanes | one round loop for many streams; draft depth chosen from a measured cost profile | E4 |
| kernel.strixfold.dense_bf16 (also crucible's EXL3 GEMV, radiance's MXFP4 GEMV) | a row's bits do not depend on how many rows share the launch | E3 |
| runtime.strixfold.prompt_cache + Imprint, SFF-PARK learned states, shared_messages | states kept at chunk ends, on disk under an identity, shared across threads | E5, E6 |
| runtime.strixfold.sampling | draws keyed by (seed, position, token id), so a sampled reply does not depend on batching | E3 |
| kernel.radiance.halo_moe + halo_router | routed rows of all streams sorted by expert into one launch; router GEMM + top-k + sort in one launch | E2, E8 |
| kernel.radiance.halo_kv (KV streaming) | full context parked, a ring of selected blocks resident per sequence | E5 |
| kernel.radiance.halo_quant_act, halo_gemm_int, docs/MOE-W4.md, docs/IQR.md | calibrated (GPTQ) narrow weights with a device-side Gram pass; producers write int8 activations | E7 |
| kernel.radiance.halo_hc | 8-bit float mixes; the arrival-counter fold | E7, E8 |
| docs/TP4.md (radiance) | a list of TP4 levers tried and lost; step size vs concurrency trade | E9; no TP work |

StrixFold itself (DeepSeek-V4.1 on a pair) stays parked as owner set it. Nothing below needs the pair. The one
approved window (SFU-RECEIPTS 9718972d74) still rides after the strixite work.

## 4. The experiments, in order

Every hardware experiment follows docs/EXPERIMENTS.md (gate TOML committed first, resources block, monitor). The
fidelity floor is identity (same ids as the stream run alone) unless the experiment changes weights, in which case
it is KL against the BF16 checkpoint (/mnt/ark/models/Qwen__Qwen3.8-Flash-Next, staged first, never run from the ark).
V10-NOISE fixed the rep count: 3 reps a side separate a 3 % effect on a quiet box. Each gate also reports one arm
under real serving load, because V10-NOISE showed a busy host moves a step by a third.

### Stage 0: the yardstick (no GPU, then one window)

**E0 FACTORY-CENSUS.** Seven days of FreeFitty turn logs joined with strixite counters, no hardware.
- Reports per request: new tokens since the thread's last turn, prefilled tokens, cached tokens, generated tokens,
  queue wait, whether the thread kept its slot; and per card: engine seconds and wall seconds.
- Prediction: at least half of prefilled tokens are recompute (prefilled minus new); queue wait is the largest part
  of a thread's wall time.
- Decides: the weight of every later stage. Recompute under 20 % closes E5 and E6. Queue wait under 20 % of wall
  demotes stage 1 below stage 3.

**E1 FACTORY-REPLAY.** Freeze a replay of real traffic (FLERP-CORPUS 74f780b3ac, FLERP-REPLAYKIT ca92617154 and
FLERP-SELFID bab6510b88 are already on the board for this) and play it at saturation against one worker in a WINJOB
window.
- Metric: committed tokens a second per worker at saturation, p50 and p95 turn time, identity per stream.
- This becomes the Flash-Next "throughput" champion area. Today that area holds a one-stream placeholder.
- Decides nothing alone; every later gate is judged on it.

### Stage 1: the row budget (predicted largest gain)

**E2 ROW-GRID.** Step time T(S, D) for S = 1, 2, 4, 6, 8 slots and D = 0, 1, 2, 3 drafts, on the replay, with a
kernel trace of one cell.
- Prediction: T = 26 ms + 6 ms a row + the draft heads. 4 slots x 1 draft gives at least 1.5x the committed tokens
  a second of today's 2 slots. Refuted below 1.2x.
- If T grows by 14 ms or more a row, the trace names the kernel that re-reads shared weights per row, and that
  kernel is the next card before any slot change (the expert-sorted launch of halo_moe is the first shape to try).
- Decides: the fleet's slot count and draft depth. strixite takes --total-slots 1 to 8 today; memory per slot is
  settled by E5.

**E3 ROW-INVARIANCE.** Does a thread's reply change with its neighbours?
- Known: strixite-slots.md section 8 found greedy output at two slots depends on what the other slot does.
- Measure the rate on the replay at S = 2 and 4, then rebuild the few-row GEMV and the verify batch so each row
  keeps its solo arithmetic order (the dense_bf16 shape). Gate: ids identical to solo on every stream, speed within
  2 % of the unordered kernel.
- Decides: whether the throughput champion can use the identity floor, and whether a review or a replay of a card
  can be reproduced exactly. Without it every later gate needs a KL floor and more reps.

**E4 LANE-POLICY.** Draft depth and slot fill chosen per step from queue depth and the E2 table.
- Idle worker: deep drafts for one thread's speed. Saturated worker: one draft, every slot filled.
- Gate: beats the best fixed setting of E2 by 5 % at each of three load levels on the replay, or the fixed setting
  ships and this closes.

### Stage 2: threads that outnumber slots

**E5 SLOT-PARK.** Skipped if E0 shows little recompute.
- Census first: state bytes per live token and per slot, and how many slots fit at the real thread lengths (60K to
  74K live tokens per slot today; each slot is allocated at full capacity).
- Then keep a waiting thread's state parked and swap it into a slot without prefill (strixite already has parked
  generations; the ring-plus-parked-context shape is radiance's, the on-disk identity is Imprint's).
- Prediction: prefilled tokens a request fall from 10,600 to new tokens plus at most one chunk. Engine time falls by
  the recompute share of the 24 % prefill slice.
- Gate: replay at today's ratio (7 threads a slot), ids identical, prefill seconds a request.

**E6 SHARED-PREFIX.** Only if E5's census shows the hoisted prefix (tools, akb describe, expert lens) is 15 % or
more of live tokens.
- Store the common prefix state once per worker and let every slot reference it.
- Gate: state bytes per slot, first-token time of a new thread, ids identical.

### Stage 3: bytes per token (multiplies every stage above)

**E7 BYTE-DIET.** Decode time is weight bytes over 226 GB/s, so narrower weights are the only large single-stream
lever left. Round-to-nearest Q4 mixes failed (HC-Q4: 1.46 ms faster, KL 0.123). The family has not been swept.
- Step 1, no speed claim: measure today's served file against BF16 (KL, top-1, greedy agreement) on the replay
  prompts. That number is the floor. It does not exist yet for Flash-Next.
- Step 2, sweep on CPU and K: mixes at Q6, Q5, Q4; dense trunk and LM head at Q6; each by round-to-nearest and by
  GPTQ with a Gram pass over real activations; int8 activations as a separate arm.
- Prediction: GPTQ Q6 mixes plus Q6 dense hold the floor and save at least 3 ms of a 32.9 ms step. Refuted if no
  cell under the floor saves 1 ms.
- Decides: the next weights file. QUANT-TOKENS 17bcc3f01b already owns the "what does quantisation cost in answers"
  method; this feeds it.

**E8 STACK-DECODE** (card 09d26982ac exists) with FOLD-2 8c7f1aab87 and the one-launch router. Run after E7,
because a format change rewrites the same kernels. Bound: 5 to 7 % decode.

### Stage 4: prefill as agent turns use it

**E9 TURN-PREFILL.** From E0's histogram of prefill sizes.
- Small appends pay the fixed 0.45 s a chunk (STRIX-CHUNKCOST 8ba96d6c22 exists). A new thread's first token waits
  for another slot's whole 8,192-token chunk (5 to 11 s).
- With more slots this interference grows, so it is gated at the E2 slot count: p95 stall of the other slots' decode
  during a prefill, and first-token time for a 2,000-token append.
- Decides: chunk size and the prefill/decode interleave at the new slot count.

### Stage 5: promote and package

**E10 SERVE-CHAMPION.** One config, qwen38-fnext.serve, holding the winners.
- One worker runs it for 24 h on live FreeFitty traffic beside a control worker over the same hours.
- Gate: completed rounds an hour per worker, 0 errors, identity on a sampled replay.
- On a pass it goes into the champions ledger and is what podcore's engine layer installs (P3g). The slot and cache
  work of stages 1 and 2 is the content of FLERP-SLOTS e5b4edba91 and FLERP-CACHE db829c8a0b, so it lands in the
  strixflerp front end.

## 5. Predicted size, stated as predictions

| stage | lever | predicted gain on tokens a second per worker | confidence |
|---|---|---|---|
| 1 | slots and draft policy at today's kernels | 1.2x to 1.5x | mechanism read, row cost measured at 2 slots only |
| 1 | row scaling at the roofline | up to a further 1.3x | byte model; the 140 ms step is unexplained by 40 to 60 ms |
| 2 | no recompute on slot switch | up to 1.2x of engine time | unmeasured until E0 |
| 3 | narrower calibrated weights | 1.1x | Q4 measured fast and unfaithful; Q6 and GPTQ untested |
| 3 | launch folds | 1.05x | bounded by the ledger |

## 6. Closed, do not re-run

Chained launches (CHAIN-1: stalls or slower), HIP graphs, cross-layer and token-to-token QSA reuse, the coarse draft
block bound, 2,048-row sub-chunks, router and combine ports from arm Z, hyper-connection kernel occupancy, Q4 mixes
by round-to-nearest. Tensor parallelism for Flash-Next is left out on purpose: the model fits one node and the
queue is independent threads.

## 7. Prerequisites that are not experiments

- Bench box K cannot hold Flash-Next locally (66.8 GB /dev/shm against a 74.6 GB file; its NVMe is unmounted), so
  each K process reads the file over the 2.5 GbE mount for about 5 minutes. Mounting K's NVMe as a PLACE tier
  removes that from every gate above.
- /data on control-node had 68 GB free at 17:47Z and pod-weights refuses stages; E7 needs room for BF16 slices.
- The BF16 checkpoint must be staged through pod-weights before E7.
