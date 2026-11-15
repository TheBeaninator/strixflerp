# kernel.strixite.moe_grouped: design record

## What it is for
MoE experts for multi-token forwards (prefill, MTP verify): slots grouped by expert so each active expert's weights
are read once per tile of slots, not once per (token, slot) (`src/kernels/moe_grouped.hpp:3-9`).

## Why (origin's measurement)
At T = 512 the 5,120 slots of a layer touch ~all 512 experts ~10 times each; the per-token decode kernels were 55%
of prefill time (rocprofv3) (`moe_grouped.hpp:3-5`).

## Choices (origin's code)
- Three steps per layer: `moe_group_routes` once (both projections share it), grouped gather, grouped combine
  (`moe_grouped.hpp:12-17`, `:43-63`). Experts stacked [E*N, K] (`:57-63`).
- Math modes: F32 on the vector units (decode semantics, min-fold), WmmaBf16, and WmmaF16 for Q4 experts: x as F16,
  weights dequantized to F16 (`code * scale + min`, one rounding), FP32 accumulation over the whole K
  (`moe_grouped.hpp:30-40`). WmmaF16 picks Narrow (64-slot tiles, each wave decodes its 16 rows in registers), Wide
  (128 slots x 128 rows, each weight tile dequantized once into LDS) or Wide256 by token count; all three are
  bit-identical; Wide from M >= 4096 (`moe_grouped.hpp:33-38`; `moe_grouped.hip:1019`).
- In the F16 kernel the codes go to LDS still packed (coalesced block loads), then each wave decodes its rows; "loading
  each wave's rows straight from memory instead was 1.65x slower: scattered code and 2-byte scale loads, what costs
  is cache lines touched, not bytes" (`moe_grouped.hip:348-353`).
- Scale/min windows: kWin = 8 groups per row kept in LDS, refreshed every 8 groups; consecutive threads read
  consecutive groups of a row (a 16-byte run) (`moe_grouped.hip:374`, `:415-425`). G in 32/64/128, "a 32-code half-step lies in one group" (`:370`).
- SwiGLU fused into the gather epilogue (gate rows in waves 0-3, up rows in 4-7; h rounded once) and an optional F16
  h handoff to the down kernel (`moe_grouped.hpp:51-57`; `moe_grouped.hip:359-361`).
- Shared expert's gated add can ride in the combine's reduce (`MoeSharedAdd`; `executor.strixite.qwen4exp`
  `qwen4exp.cpp:3177-3182`).

## What it buys (measured)
With scaled codes and 64-token expert tiles prefill reached ~1,000 tok/s; 2k to 8k chunks added +8% and +5%
(`strixite docs/how-it-got-fast.md:194-218`).

## Costs and limits
- In PF-CF the experts stage was 1,805 ms of 5,523 (33%): `grouped_wmma_f16_kernel` 1,519 ms, `combine_reduce`
  186 ms (pod `/data/pod/reports/2026-10/strixite-results/pfcf.json` (card PF-CF 4546bb4318: build pfq1 ab50cc4, one node, depth 16,384, chunk 8,192, rocprofv3; an October build, before v0.5.0's grouped-expert rewrite reached the fleet)).
- Arithmetic (inferred): per token 10 experts x (1,280 x 2,560 + 2,560 x 640) MAC x 48 layers = 2.36 G MAC; per
  8,192-token chunk 38.7 TFLOP; at 1,519 ms ~25 TFLOP/s, ~43% of the assumed F16 peak, below the dense GEMM's ~57%.
- Tile fill (inferred): 8,192 tokens x 10 slots / 512 experts = 160 slots per expert per layer on average; 128-slot
  Wide tiles run 2 passes per expert at ~62% fill, re-reading that expert's weights per pass: ~2.8 GB of expert
  weights per layer per chunk, ~136 GB per chunk, ~0.65 s at ~210 GB/s if nothing is cached.
- Q4 only on the F16 path; Q5 stays on BF16 (`qweight.hpp:122-124`). The fleet G32 expert-down refreshes the scale
  window every 256 codes (K = 640: 3 windows per row).

## Coupled to
`format.strixite.q4` (row-major planes, G), `q5_device`, `merge_plan` (stacking, [gate | up] adjacency),
`kernel.strixite.wmma_gemm` (F16 helpers), `moe_router` (ids), `runtime.strixite.qweight`, `executor.strixite.qwen4exp`.

## Prefill remarks
The biggest single kernel in prefill. Levers here: slots per weight pass (bigger chunks), tile shape and occupancy,
combine_reduce, and the weight layout (a tile-contiguous copy would remove the LDS repack the 1.65x lesson forced,
but a second copy of the 68 GB stacks does not fit; see the prefill report).
