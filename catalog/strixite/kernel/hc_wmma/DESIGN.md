# kernel.strixite.hc_wmma: design record

## What it is for
The hyper-connection mixes on the matrix units for multi-token forwards: x/h rounded once to F16, weights dequantized
to F16 once, FP32 accumulation; norms, gates and the stream combine stay FP32. Weights read once per 64-token tile
(`src/kernels/hc_wmma.hpp:1-14`).

## Choices (origin's code)
- Up: block of 32 positions j for all 4 streams = 128 rows of W_up chunk-major; codes, then a window of kWin
  groups' scales/mins, fetched with consecutive lanes on consecutive rows, which the [K/G][N] chunk-major layout
  makes coalesced; staged and dequantized to F16 in LDS (`hc_wmma.hip:151-155`).
- Requires H == 4, d % 64 == 0 (2,560), r % 64 == 0 (320) (`hc_wmma.hpp:14`).

## Cost (measured)
PF-CF: the HC stage was 874 ms of 5,523 per 8,192-token chunk (16%): `hc_up_wmma` 380 ms, `hc_down_wmma` 288 ms,
`hc_inject_inv` 180 ms (pod `/data/pod/reports/2026-10/strixite-results/pfcf.json` (card PF-CF 4546bb4318: build pfq1 ab50cc4, one node, depth 16,384, chunk 8,192, rocprofv3; an October build, before v0.5.0's grouped-expert rewrite reached the fleet)). Inferred: ~10 TFLOP per chunk at ~12 TFLOP/s; small r = 320 and the
elementwise steps keep it far from the matrix peak. HC is replicated, not split, across TP ranks (same file,
`finding`).

## Coupled to
`kernel.strixite.hc`, `format.strixite.q8` / `q4` (chunk-major), `kernel.strixite.wmma_gemm`, `executor.strixite.qwen4exp`.

## Prefill remarks
Weight bytes are small (0.7 GB); this stage is bound by the shape and the fused elementwise work, not the format.
