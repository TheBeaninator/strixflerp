# kernel.strixite.linear_q4: design record

## What it is for
The decode GEMV for Q4 weights (dense projections and the per-token MoE gather/combine), plus the fused expert
gather+SwiGLU.

## Choices (origin's code)
- Built for decode (small M, memory-bound): 1-8 waves per output row (split-K), each lane loads 32 codes per 16-byte
  load inside one group (G >= 32) and computes `s*sum(x*q) + m*sum(x)` per chunk, one FMA per weight; M == 1 has its
  own instantiation; larger M runs 8 rows per pass, "correct, but prefill is not what this kernel is built for"
  (`src/kernels/linear_q4.hpp:6-12`).
- Min-fold: per 32-code chunk `dot = sum x*q`, `xsum = sum x`, `acc = s*dot + m*xsum` (`linear_q4.hip:97-121`,
  `:177-199`). BF16 inputs use `v_dot2_f32_bf16`, xsum as a dot2 with (1,1); codes are exact in BF16; BF16 scale =
  bits << 16 (`linear_q4.hip:56-60`). At M = 8 the scalar form ran 2.3x its M = 1 time (`:57-59`).
- Group size: kernels template on `w.G` (32/64/128) (`linear_q4.hip:100-101`, dispatch `:334-338`; check `:403`).
- Split-K: ~2048 codes per wave measured best (`linear_q4.hip:342-350`).
- 16-byte alignment of the code array is checked (`:407-408`); it holds because strixw aligns components
  (`format.strixite.strixw`).
- Experts: E experts stacked as one [E*N, K] matrix, rows e*N..e*N+N-1, ids from the device, bad id = NaN + error
  flag, never an out-of-bounds read (`linear_q4.hip:162-165`). The gather+SwiGLU computes gate row j and up row j of
  one expert (the merged [gate I | up I] rows) and stores silu(g)*u, bit-identical to gather then swiglu
  (`:244-276`).

## What it buys
One FMA per weight and one 16-byte load per 32 codes: decode reads weights near bandwidth. The fused SwiGLU saves a
launch and the [M, A, 2I] round trip.

## Costs and limits
Per-row scale/min loads are 2-byte scattered reads (fine for decode, costly for tiled GEMM, see
`kernel.strixite.moe_grouped`). Weights are re-read for every 8 rows of x.

## Coupled to
`format.strixite.q4` / `q4_device`, `qsplit` (HB template, Q5 reuse), `merge_plan` (stacking and gate/up adjacency),
`runtime.strixite.qweight`, `executor.strixite.qwen4exp`.

## Prefill remarks
Used in prefill only below the grouped / WMMA thresholds (`kGroupedMinTokens = 6`, `kWmmaMinTokens = 12`,
`executor.strixite.qwen4exp` `qwen4exp.hpp:724,730`). Before the grouped kernels, the per-token expert GEMVs were
55% of prefill time at T = 512 (`kernel.strixite.moe_grouped` `moe_grouped.hpp:3-5`).
