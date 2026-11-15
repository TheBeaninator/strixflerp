# kernel.strixite.linear_wmma: design record

## What it is for
Multi-token (prefill) dense linear on the gfx11 matrix units for Q4/Q5/Q6/Q8 row-major weights, plus BF16-weight
WMMA for the router (`src/kernels/linear_wmma.hpp:3-7`, `:21-28`).

## Choices (origin's code)
- Tiles: 64 rows of x by 128 rows of W, K in steps of 64 (K % 64 == 0), 8 waves as 2 x 4, each 32 x 32
  (`linear_wmma.hpp:6-7`; `linear_wmma.hip:14`).
- F16 path (Q4/Q8): codes dequantized once while staged, `code * scale + min` to F16 in LDS, the two waves sharing a
  weight fragment read it decoded; x staged as F16; "no row sums, no per-group accumulator rescale"; a step that
  opens a scale window stages it first (`linear_wmma.hip:64-68`). The decode kernels' min-fold is dropped here.
- BF16 path: x rounded to BF16, raw codes on the matrix units, scale and min applied per group in FP32
  (`linear_wmma.hpp:3-6`).
- Shared helpers (`WmmaWCodes`, scale windows of kWin groups) live in `kernel.strixite.wmma_gemm`.

## What it buys (measured)
Matrix units moved prefill from 141 to 652 tok/s; staging weights as scaled codes took it to ~1,000
(`strixite docs/how-it-got-fast.md:194-213`). In the PF-CF profile `linear_wmma_f16_kernel` was 1,424 ms of a
5,523 ms GPU-busy 8,192-token chunk (26%) (pod `/data/pod/reports/2026-10/strixite-results/pfcf.json` (card PF-CF 4546bb4318: build pfq1 ab50cc4, one node, depth 16,384, chunk 8,192, rocprofv3; an October build, before v0.5.0's grouped-expert rewrite reached the fleet)).

## Costs and limits (inferred)
- Row-major codes: each 16-byte code load of a 128-row tile hits a different row (1,280 bytes apart for K = 2560),
  so a K step touches 128 cache lines partly; reuse across K steps depends on L2.
- Scales/mins come from separate planes: a window load per kWin groups per tile.
- Arithmetic on the PF-CF numbers: the dense projections are ~2.93 G MAC per token (GDN in/out x 36, attention
  qkv/o x 12, shared expert x 48), 48 TFLOP per 8,192-token chunk; at 1,424 ms that is ~34 TFLOP/s, ~57% of an
  assumed ~59 TFLOP/s F16 WMMA peak for gfx1151 (40 CUs, 512 F16 FLOP/clk/CU, ~2.9 GHz; not measured here).

## Coupled to
`format.strixite.q4`, `q5`, `q6`, `q8` and their device views, `kernel.strixite.wmma_gemm`, `runtime.strixite.qweight`
(`linear_qw_wmma` dispatch), `executor.strixite.qwen4exp` (`kWmmaMinTokens = 12`, `qwen4exp.hpp:730`).

## Prefill remarks
This is the dense prefill GEMM. A tile-major weight layout (codes of a 128-row x 64-code tile contiguous, its
scales/mins adjacent) would make each tile one contiguous read; it would need a new encoding or a load-time copy
(the dense projections are ~3.5 GB in the fleet file).
