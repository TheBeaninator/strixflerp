# AMD Loom: shapes for strixflerp, ranked

Written 2026-11-14 (card LOOM-HARVEST d0ef20240d) from loom-skills `df8af7c`-era sources and hrx-loom 0.1.0, all on
the ark since ark 31c2682: src `loom-fun/loom-skills`, `loom-fun/weaver`, `loom-fun/loom-dev`, `ROCm/hrx-system`;
reference `binaries/hrx-loom` (wheel); models `loom-fun/*-loom` (51 program packages). Nothing was built or run.
Labels as in `docs/TENSORFOLD.md`: **[claimed]** Loom's own reports, **[measured]** ours, **[inferred]** our arithmetic.

The principle (`docs/SHAPE-TRANSFERS.md`): a module's recipe (algorithm, layout, work mapping, schedule) carried
into our executor. Gate: bit identity where rounding points and summation order are kept, `docs/FIDELITY.md` where not.

## 1. What Loom is

- **hrx-loom** (ROCm/hrx-system, Apache-2.0 WITH LLVM-exception): compiler (`loomc`), IREE-based runtime and
  `loom-serve`, an OpenAI-compatible server that runs a *model program*: `.loom` files (a vector IR close to MLIR)
  plus a `model.json` that names the GGUF by origin and sha256. No engine changes per model; the program is the model.
- **loom-skills** (MIT, AMD AI group): agent skills that *generate* those programs from a GGUF (Python generators
  `starter/gen_llm.py`, `gen_moe.py`, `gen_gdn.py`, `gen_flash2*.py`, `gen_gemm.py`, `draft/`), test kernels
  against llama.cpp goldens, and optimize under a measured controller. The catalogue
  `skills/loom-optimize/reference/optimization-catalog.md` records every lever with its number, including the ones
  that lost. That catalogue is the main shape source below.
- **weaver** (MIT): a portable CLI around the same loop with a bundled local agent.
- **51 ports** on HF `loom-fun`, all from the same skill: Ornith-1.5 35B-A3B and 122B, Qwen3.6-35B-A3B (our proxy
  GGUF), Qwen3.8-Flash-Next (strixite's model), Qwen3.8-27B, gemma-4, gpt-oss-20b and more.

## 2. Numbers that matter to us

DevLab Strix Halo runners (Ryzen AI MAX+ 395, the pod's chip), Q4_K_M, one node, client-timed, **[claimed]**. Every
Loom and llama.cpp number below is the `performance.devlab` block of the model's summary in the ark,
`loom-fun/loom-dev` `summaries/<checkpoint>.json` (DevLab runs of 2026-11-13: 43400168275 for Qwen3.6, 43392400146
for both Ornith-1.5 models, 43376548212 for Qwen3.8-Flash-Next); the drafting column is that file's
`performance.drafting`. llama.cpp is build b1374 with its own drafter (`draft-dflash` / `draft-mtp`).

| model (summary file) | metric | Loom | llama.cpp ROCm | ours |
|---|---|---|---|---|
| Qwen3.6-35B-A3B, our 35B proxy GGUF (`Qwen3.6-35B-A3B-Q4_K_M.json`) | prefill 2k | 2,406.2 | 1,316.3 | 1,974 (`ornith397.prefill-best` on the proxy) **[measured]** |
| 〃 | prefill 16k / 32k | 2,331.7 / 2,026.5 | 1,229.9 / 1,048.3 | 1,937 at 16k **[measured]** |
| 〃 | decode, Loom DFlash depth 3, llama.cpp n-max 7 | 89.1 | 57.1 | no proxy decode champion |
| 〃 | 4 streams aggregate | 165.2 | 127.6 | not measured |
| Ornith-1.5-35B-A3B (`Ornith-1.5-35B-A3B-Q4_K_M.json`) | prefill 2k / 32k | 2,378.4 / 2,014.0 | 1,218.8 / 985.5 | — |
| 〃 | decode, Loom DFlash depth 1, llama.cpp n-max 8 | 72.5 | 38.8 | — |
| Ornith-1.5-122B (`Ornith-1.5-122B-Q4_K_M.json`) | prefill 2k | 901.3 (x2.15) | 420.0 | — |
| 〃 | decode, Loom DFlash depth 1, llama.cpp n-max 15 | 27.7 (x2.72) | 10.2 | — |
| Qwen3.8-Flash-Next (`Qwen3.8-Flash-Next-UD-Q4_K_XL.json`) | prefill 2k | 1,154.6 | 442.3 | strixite recipe (compare in LOOM-RIVAL) |
| 〃 | decode, Loom MTP depth 3, llama.cpp draft-mtp | 41.9 | 34.4 | 〃 |

None of these decode rows is drafts-off: every Loom decode figure uses its drafter at the depth shown.

Loom's roofline tool reports no per-model efficiency for these runs. Its fitted constants are 0.33 of the WMMA peak for
prefill end to end and 0.85 of bandwidth for 4-stream MoE decode, fitted on other ports (loom-skills
`skills/loom-roofline/roofline.py` lines 27-33: prefill from Bolt-E4B 2,339 / 7,094, 4-stream MoE from Flint 0.74 and
LFM2.5-24B-A2B 0.91; `reference/calibration.md` line 41: "Prefill end to end is ~0.25–0.33 of WMMA peak")
**[claimed]**. Its fidelity rule is golden token comparison against llama.cpp with near-tie analysis, not a KL floor;
LOOM-RIVAL measures it against our FP32 reference so the two meet on one scale.

The proxy gap on the same GGUF and chip is +21.9 % at 2k (2,406.2 vs 1,974) and +20.4 % at 16k (2,331.7 vs 1,937), and
the nearest Loom prefill number to our 1,974 is Ornith-1.5-35B's 2,378.4 (+20.5 %). That gap is why the prefill shapes
rank first. One change since the table: PREFILL-LRC 5083008fe9 (arm Z = live tiles + tiled router + fast combine,
branch card/5083008fe9, not merged at this writing) measured 2,725.5 tok/s at 2k and 2,560.3 at 16k on the proxy
(`gate/results/prefill-lrc/zgate/verdict-gate.json` on that branch) **[measured]**, above Loom's proxy row. It needs
`FLERP_KG_LIVE=1`, a knob not yet on Buzz main, so LOOM-RIVAL either sets it or compares against the 1,974
row, and says which.

## 3. Shapes and their ports

### L1 Padding-free MoE prefill (Loom "Chunk size", rounds 1-2 and the qwen35moe pass)
Shape: every per-token-row kernel in a prefill epoch gets a live-row twin that does nothing on a padding row except
write row scale 0; the GEMM skips any row tile whose scales are all 0; the router skips all-padding workgroups; the
bucketed expert rows skip their unused tiles. On top: the grouped expert GEMM prefetches the next K step into
registers and skips MMAs of empty 16-row waves; `moe_combine` runs one workgroup per token (an expert row read
contiguously instead of in 8 far-apart pieces); `gdn_norm_gate` packs 8 heads per workgroup.
Loom: Ornith-1.0-35B prefill 1k 1,838 to 2,423 (+32 %), Qwen3.5-35B-A3B +22 %, 2k +24-42 %, decode tie, outputs
bit-identical **[claimed]**.
Port: our grouped K-quant path (`docs/PREFILL-ORNITH.md` section 2, the `MoeEpilogue` combine) has every one of these
roles. Live rows come from the chunk's row count, which the executor already knows; the combine becomes
token-major. Skips and prefetch keep summation order, so the gate is logits sha256 equal to `prefill-best`.
For the 397B over four nodes, the same skips matter more: 256-row sub-chunks over 512 experts leave most 16-row waves
empty **[inferred]**, which is where `ornith397.prefill-best` missed its 2,500 tok/s projection.
Card: LOOM-MOEPF 81a8399a82.

### L2 Proposal head and drafter bytes (Loom "Drafter bytes per step")
Shape: count a draft step's bytes beyond the trunk; convert at load: drafter BF16 to Q8_0, and the proposals read a
Q4_0 copy of the output head while verification keeps the exact head, so outputs cannot change.
Loom: Bolt-E4B depth 3 82.6 to 98.0 tok/s **[claimed]**.
Port: our MTP drafts (3 per step on `ornith397.decode-best`) each run the full head over a 248k vocabulary. A Q4_0 copy
made at load serves the draft passes; the verify pass and the champion's identity floor are untouched. Gate: ids
equal drafts-off byte for byte. Proxy first, then an Ornith window.
Card: LOOM-DRAFTHEAD e6c34ff2ee.

### L3 MoE decode launch shape (Loom "MoE decode")
Shape: rows per workgroup chosen so a gate/up launch has about 3,000 workgroups (powers of two 2..8); the router
scores 4 experts per wave in one loop (4 weight loads in flight, same summation order, bit-identical); expert
activations int8 with dot4 against the codes (llama.cpp mmvq numerics).
Loom: Qwen3-30B-A3B 14.39 to 12.97 ms a token (1.11x) **[claimed]**.
Port: workgroup count rule recomputed for top-8/256 (proxy) and top-10/512 (Ornith); router change is identity-gated;
int8 activations change numerics, so `docs/FIDELITY.md` judges that arm.
Card: LOOM-MOEDEC 7a663a62f7.

### L4 Gated delta-net recurrence (Loom "Gated delta-net recurrence")
Shape: row-split recurrence, a group of 8 lanes owns 4 value columns and each lane 16 key rows (64 state values in
registers, 384 workgroups instead of 1,536); two kernels, decode grid and prefill grid, a span runs in exactly one,
chosen by its own length, so batching never changes its numerics; channels spread over several workgroups when fewer
than 64 tokens are in flight; stores batched out of the step loop and loads one step ahead.
Loom: Qwen3.8-27B prefill +2.8 %, decode +5-8 % on the 16 x 2 grid **[claimed]**; Qwen3.5-0.8B decode +18 % from
the channel spread **[claimed]**.
Port: our GDN kernels for qwen35moe (proxy and Ornith) take the same split. The chunked scan (SHAPE-TRANSFERS R10)
stays the bigger lever and its own card.
Card: LOOM-GDN 54f59e0832.

### L5 Gates and measurement (loom-benchmark, loom-optimize "Measurement pitfalls", loom-roofline)
- **Consistency gate**: concurrent outputs equal sequential outputs, and a prompt repeated alone repeats exactly. It
  caught 3 of Loom's 8 real bugs (mixed tiles, padding, races) **[claimed]**. Our throughput floor states this rule;
  this is the tool for it.
- **Negative control** in every kernel A/B: a case expected to fail when the output equals the initial fill, because
  two kernels that both write nothing agree.
- **Roofline per metric** from the GGUF's tensors (bytes a token for decode, FLOPs a token for prefill, weights once
  per 4 tokens for 4 streams, drafting folded in by measured acceptance); each champion row gets its % of ceiling.
- **Pitfalls**: the 32 MB MALL inflates microbenchmarks whose tensors fit; WMMA on constant operands runs 52-57 TOPS
  vs about 40 on real data (power limit), so compute A/Bs use real weights and random activations; a nonce at the
  start of each timed prompt defeats prefix caching; A/B interleaved on one server with a noise-band verdict.
Card: LOOM-GATES 97d65d08a3.

### L6 Smaller shapes (folded into the cards above, or noted for later)
- **Last-layer output rows** (`LAST_OUT`): prefill runs the last layer's post-attention work only on the rows that
  feed the head; +1 % **[claimed]**. Off when MTP reads the trunk's final rows, so for us only drafts-off prefill.
- **Argmax head**: 128-row vocabulary chunks write (max, argmax), no logits buffer; greedy only. Saves V x 4 bytes and a
  pass a token; our identity gates are greedy, so it fits gate runs and greedy serving.
- **Adaptive rows per workgroup** for dense matvecs (halve R while n / R is under a target; keep 1- and 4-token R
  equal or the consistency gate fails): 54.2 to 59.9 tok/s on Bolt-E4B **[claimed]**. Applies to our dense projections
  in decode.
- **Listed fallback attention**: compact the tokens flash2 does not cover into a list with an atomic counter and run
  split-K attention over it with an indirect grid, so padding never does attention work.
- **Per-token RoPE + KV store**: one 256-lane workgroup per token loops over all heads, angles computed once; 6x the
  per-head version **[claimed]**.
- **Chunk size vs padding**: an epoch runs at the next power-of-two shape; 4,096-row chunks gain +22-38 % on MoE at 2k+
  but cost short prompts until the live-row skips (L1) exist. Our 4,096-row chunk choice (`prefill-best` G4 run 2) and
  L1 go together.

### L7 The agent loop (process shape, for FreeFitty pod work)
loom-skills is a worked example of agents doing kernel work under a controller: `perf.py` logs each build and prints
CONTINUE / REVIEW / BUDGET / DONE; every design choice and its rejected options go to `decisions.jsonl` with numbers;
"settled, don't re-run" lists stop fresh agents repeating lost A/Bs; `loom-improve` measures the skills themselves with
fresh-agent trials on a reference suite. Our champions ledger plays the controller's role for configs; the
"settled" and "did not help" lists are the piece we lack, and the LOOM port cards record theirs in their summaries.

## 4. Ranking and cards (all under STRIX-META bd830ae1ca)

| rank | card | shape | expected gain | gate |
|---|---|---|---|---|
| 0 | LOOM-RIVAL 133f14a170 | both engines on our GPU and GGUF | decides the order below | numbers + greedy agreement |
| 1 | LOOM-MOEPF 81a8399a82 | L1 | prefill 2k from 1,974 toward Loom's 2,406.2 on the proxy (+21.9 %), counting only the skips PREFILL-LRC 5083008fe9 has not already taken (it reached 2,725.5 with live tiles) | sha256 = prefill-best, else KL <= 0.0176 |
| 2 | LOOM-DRAFTHEAD e6c34ff2ee | L2 | decode +8 % at 3 drafts | ids identical |
| 3 | LOOM-MOEDEC 7a663a62f7 | L3 | decode +8 % drafts off | router identity; int8 by FIDELITY |
| 4 | LOOM-GDN 54f59e0832 | L4 | decode +3 %, prefill flat or up | KL <= 0.0176 |
| 5 | LOOM-GATES 97d65d08a3 | L5 | tools, no tok/s | pytest |
| 6 | LOOM-ORIGIN 91d7edc392 | catalog/loom | shape library, no tok/s | flerp diff-origin |

Every port lands as a new named config beside the proxy champions the gates already run (`configs/ornith397.<area>-<card>`, e.g. `ornith397.prefill-moepf`; these configs run the 35B proxy GGUF), with the defaults and the existing `ornith397.*` champions unchanged;
promotion goes through `flerp champion propose`. A change to shared prefill or decode kernels is cleared with the
session testing Ornith on strixflerp before its branch merges.
