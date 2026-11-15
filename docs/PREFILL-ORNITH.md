# Prefill of Ornith-1.5 397B: grouped K-quant kernels over the GGUF blocks

Pod card PREFILL-ORNITH 8ef0a645e7 (parent STRIX-META bd830ae1ca). Criteria: `gate/prefill-ornith.toml`, committed
before any GPU run. Kernel: `catalog/flerp/kernel/moe_kgrouped`. Bench: `gate/prefill_ornith_bench.hip`. Written
2026-11-14. "Measured" means a file behind it; everything else is labelled estimate.

## 0. Where prefill stands

- **Measured.** About 125 tok/s end to end: 1,808 tokens in 14.5 s at 16 pipelined sub-chunks, 44 s unpipelined
  (FLERP-P5 run 11132252). Stage 0 runs a 1,024-row chunk in 7.08 s: 472 ms a block, 144 rows/s a stage
  (`gate/results/ornith15-397b.p4/orn-0.jsonl`).
- **Cause.** Every (token, slot) pair reads its expert's rows (`moe_kquant.hip:271-300`, launched by
  `moe_ffn.cpp:86-95`), and so does the shared expert. That is about 82 MB a token a block: 84 GB for 1,024 rows,
  about 0.42 s at 200 GB/s. Nearly all of the 472 ms is this (estimate; the bench measures it).
- **What the arithmetic allows.** A block costs about 510 MFLOP a row:
  - routed experts: 252 MFLOP;
  - shared expert: 25 MFLOP;
  - dense projections: 236 MFLOP in a GDN block, 210 in an attention block;
  - router: 4 MFLOP.

  That is 7.65 GFLOP a row a stage of 15 blocks. At 25 TFLOP/s (half the measured 48 TFLOP/s WMMA peak, where
  strixite and Crucible's grouped kernels run) a stage would do about 3,300 rows/s (estimate).

## 1. Weight layout (a): read the GGUF blocks in place

| option | bytes | decode | fidelity | verdict |
|---|---|---|---|---|
| Lossless repack into WMMA panels, a second prefill-only copy | +54 GiB a stage (15 blocks x 3.6 GiB of experts). Stage 0 has 64 GiB free after its 57 GiB load; a 16K-row arena and 256K of KV need the rest | untouched | bit-identical weights | **no**: does not fit beside decode's copy |
| Lossless repack in place, both paths read panels | none | every decode kernel rewritten and re-gated (DECODE-KERN3, ORNITH-KERN3) | bit-identical | not now: puts tuned decode at risk for a prefill gain |
| True requant (strixw, EXL3) | about -5 % (strixw) to -19 % (EXL3 4.0) | rewritten | changes the model; needs the BF16 source (about 800 GB, not on the pod) | **no**: no source, and it is a different model |
| **Decode from the GGUF block into LDS** | none | untouched | weights exact up to one F16 rounding of each dequantized value | **chosen** |

Why in place works here:

- **The layout is already good for this.** A Q4_K row is 16 blocks of 144 contiguous bytes (K 4,096). Each 64-K step
  of a 128-row tile reads one 16-byte header and 32 contiguous code bytes per row: aligned 16-byte loads, and every
  byte of each cache line gets used over 4 consecutive steps.
- **Q6_K costs only a little more.** Its 210-byte blocks are 2-byte aligned and take 3 loads a part.
- **Panels would buy little.** strixite's lesson (`docs/CRUCIBLE-VS-STRIXITE.md` section 3, M4) is that layout bytes
  are not where prefill time goes. What matters is how many rows share one decoded tile.
- **Decode cost is small.** It is about 3 VALU operations a weight (code, FMA, F16 convert). For 6.4 G weights a
  block that is about 2 ms at 1,024 rows, under the matrix work.

**Numerics.** The value is `d*sc*q - dmin*m` (Q4_K) or `d*sc*(q-32)` (Q6_K), computed in FP32 and rounded once to
F16 in LDS. This is strixite's MoeMath::WmmaF16. x is rounded to F16 too. The relative error is about 5e-4 for each,
far below the 4- and 6-bit quantization steps. The int8 alternative (llama.cpp/gufo MMQ with Q8_1 activations) has
coarser activations. FIDELITY.md shows the current prompt pass loses most of its KL to exactly that Q8_1 step, so
F16 is the more faithful choice.

**F16 range.** x and h = silu(g)·u are checked against 65,504. A value out of range sets a device flag. The executor
then re-runs that block's feed forward on the current path, which is exact, and counts the fallback.

## 2. Grouped expert kernels (b)

The router is unchanged: GateLogits wide plus RouterTopK give ids and weights `[T, 10]`. Then:

1. **Group** (`kgrouped::Group`):
   - a count kernel builds the histogram with atomics;
   - one 1,024-thread scan turns it into offsets and a device tile list of (expert, first row) BM-row items;
   - a place kernel writes the sorted pair list.

   The order within a group comes from atomics and does not matter: every output row is a function of its own
   input row only. WMMA accumulates each element over K in a fixed order, whatever else shares the tile. Outputs are
   therefore bit-identical for any order, any BM and any chunk cut. Criterion K3 checks this, and it is also what
   O2's pipelined sub-chunks want.
2. **ToF16**: x to F16 once (`[T, 4096]`, 8 KB a row).
3. **GateUp** (Crucible's pair with the SwiGLU epilogue):
   - a block holds BM rows of one expert's group and 64 h columns, i.e. 64 gate rows plus the same 64 up rows
     (128 weight rows);
   - each 64-K step decodes the 128 x 64 weight tile into LDS once, and every row tile of the block uses it;
   - each wave owns gate and up for the same 16 columns, so `h = silu(g)·u` forms in registers;
   - h is stored as F16 in sorted order, `[pairs, 1024]`.
4. **Down**: the same kernel over 128 output rows of `ffn_down_exps`, K = 1,024. It reads h by sorted position and
   writes FP32 to the pair's own row `[T x 10, 4096]`, which is the layout today's `MoeEpilogue` reads (combine
   unchanged: slots summed in order, plus sigmoid(gate) x shared).
5. **Shared expert**: GateUp and Down over the identity group (one expert, every token).

**Tiles for gfx1151** (`v_wmma_f32_16x16x16_f16`, wave32):

- **Block shape.** 256 threads (8 waves), BK = 64. The wave grid is 2 (rows) x 4 (columns). A wave holds TM = BM/32
  row tiles x 2 column tiles, which is 64 accumulator VGPRs at BM 128.
- **LDS.** x tile BM x 72 F16 plus weight tile 128 x 72 F16 (rows padded to 144 bytes, 16-byte aligned). That is
  36 KiB at BM 128, so 3 blocks fit a WGP.
- **Software pipeline.** The next step's x and code bytes go into registers while this step's WMMAs run.
- **BM per launch.** BM is 32, 64 or 128, picked from rows per active expert. At 1,024 rows an expert has about
  20 rows (BM 32), at 4,096 about 80, at 16,384 about 320 (BM 128, 3 tiles).
- **Register budget.** Mind it: the ISA scan shows strixite's moe_grouped losing a wave under therock 10.1.
  `k_profile` reports VGPRs and scratch for these kernels.

## 3. Dense projections (c)

- `kgrouped::Linear` is Down's kernel over the identity group. It takes the attention and GDN in/out projections
  (`attn_qkv`, `attn_gate`, `ssm_out`, `attn_q/k/v`, `attn_output`) as F16-activation WMMA GEMMs on the same
  in-place blocks.
- Today these take gufo's Q8_1 MMQ (`LaunchBatchedQuantGEMMPreQuantized`, `prefill_chunk.cpp:172-188`). Its rate on
  Ornith shapes is not measured. It sits inside the "rest" of a block, i.e. 472 ms minus the experts.
- Moving them to Linear is worth doing only where the bench beats the rest's share. It also removes the Q8_1 step
  that dominates prompt-pass KL (FIDELITY.md).
- Step 2, after the experts.

## 4. Executor (d)

- **Step 1** (built): `LaunchMoeFfn` takes the grouped path when tokens >= 64, the expert and shared types are
  Q4_K or Q6_K, and `FLERP_MOE_GROUPED` is not 0. Scratch comes from the arena's existing MoE regions: x F16, h and
  the plan sit in region c, beside the current path's act. The overflow flag is read once a block, which also
  decides the fallback.
- **Step 2**: bulk chunks.
  - Raise `max_prompt_batch` from 2,048 to 8,192-16,384 rows. The dense GEMMs run over the whole chunk.
  - The GDN conv/recurrence and attention cores run in 2,048-row parts (their kernels already take a row range).
  - The experts run in parts of at most 4,096 rows, so the FP32 `[pairs, 4096]` partials stay at 0.66 GB, not
    2.7 GB. At 4,096 rows a part is near compute-bound, so re-reading the experts per part costs little (estimate
    below).
- **Stage arena.** One allocation sized for the largest part, reused by every block (Crucible's arena). Per row it
  holds hidden FP32 16 KB, x F16 8 KB, dense outputs about 100 KB, and per expert part 205 KB a row x 4,096.
- **Pipelining.** O2 already pipelines sub-chunks across stages. Bulk chunks change the sub-chunk size, not the
  relay.

## 5. Expected rows/s a stage (e): estimates

Per block, the new routed and shared experts take max(weight bytes / B, FLOPs / F) plus padding. B is about
200 GB/s and F about 25 TFLOP/s; weight bytes are about 3.9 GB a block.

| rows a chunk | experts new (estimate) | rest a block (estimate: 472 - 440, linear in rows) | stage ms | rows/s a stage |
|---|---|---|---|---|
| 1,024 | about 22 ms (bytes-bound, BM 32 padding) | about 32 ms | 15 x 54 = 0.8 s | about 1,260 |
| 4,096 | about 50 ms | about 128 ms | 2.7 s | about 1,530 |
| 16,384 | about 190 ms (4 parts) | about 512 ms | 10.5 s | about 1,560 |
| 16,384, dense on Linear as well | about 190 ms | Linear about 150 ms + cores (unknown) | about 5-6 s | about 2,700-3,300 |

- **Experts alone.** Today's 1,024-row chunk would go from 7.1 s to about 0.8 s a stage, 9x (estimate).
- **Beyond 1,024 rows.** The rest of the block decides, so dense projections and bulk chunks are the second lever.
- **End to end.** Four pipelined stages run at about the slowest stage's rate:
  - **32K tokens**: about 21 s with grouped experts only, about 11 s with dense on Linear as well. Attention at
    that depth adds about 2 GFLOP a row a stage.
  - **256K tokens**: about 170 s, or about 100 s plus attention. Attention averages 128K keys, about 16 GFLOP a row
    a stage, so about 280 s in all at 20 TFLOP/s attention.

  All of these are estimates.
- **Inputs to correct after the bench.** The bench gives measured expert times and the measured current-path time,
  so "rest" becomes 472 ms minus a measured number. Results are in `gate/results/prefill-ornith/summary.md`.

## 6. Fidelity (f)

- **Why exact equality is out.** The grouped GEMM changes the arithmetic (F16 weights and x, WMMA FP32 accumulation
  in place of the FP32 row dot), so prompt logits are not bit-identical to the per-token path. They are, however,
  bit-stable across grouping, BM and chunk cuts (K3).
- **Kernel level** (K1, K2): against a float64 reference written from llama.cpp's dequantizers, with no code
  shared with the kernel. The current path's error is recorded beside it.
- **Model level** (F1-F5): on the 35B proxy (same family, one GPU), A = current path, B = grouped,
  D = `--decode-prompt` (FP32 activations, the FIDELITY.md yardstick). Over 8 prompts x 32 positions (prefill end
  plus 31 teacher-forced decode positions, which read the KV and DeltaNet state the prefill wrote):
  - KL(A||B): mean <= 0.002, max <= 0.02;
  - top-1 agreement >= 98 %, with every miss on a margin under 0.3 nats;
  - KL(D||B) <= 1.25 x KL(D||A) + 0.0002;
  - greedy ids equal for at least 7 of 8 prompts over 64 tokens;
  - overflow fallbacks counted.
- **Ornith itself.** The gate is not re-run on Ornith until a 4-node window (owner's go). Ornith prompt logits are
  checked then against run 11132252's ids under the same F1-F4.

## 7. Measured (2026-11-14, gate/results/prefill-ornith)

- **Kernels** (K): K1-K4 pass, with rel_rms 5e-4. The routed pass is 13.1x / 29.5x / 36.5x the current path at
  1,024 / 4,096 / 16,384 rows (30.7 / 55.3 / 178.5 ms; 8.4 / 18.6 / 23.1 TFLOP/s). Linear runs at 24-30 TFLOP/s.
  The column-fast grid order was worth 1.4-2.2x and is the default.
- **Projected stage rows/s** with the experts alone: 848 / 1,081 / 1,128. The rest of the block now dominates (step 2).
- **35B whole model**: prompt pass 4.5x (297 to 1,326 tok/s at 2,048 tokens).
  - Fidelity F1-F4 fail as committed: KL(A||B) mean 0.0096.
  - The numpy block check shows F16 rounding, not a bug.
  - The current path is 0.0089 from the FP32 yardstick itself.
  - The fix, if needed, is exact codes plus hi/lo F16 activations.

## 8. Not done here

- Dense projections on Linear (step 2) and bulk chunks with the arena.
- The Ornith window.
- A panel repack, if `k_profile` shows the in-place loads limiting at 16K rows. The bench separates load from math
  by BM.
