# kernel.strixite.ple: design record

## What it is for
qwen4_exp PLE injection on the GPU after the host has gathered the n-gram rows e [T, 2560] and the merged
[W_key; W_val] projection has produced kv: `ple_gate` (per token and stream, norms folded into sums) and
`ple_conv` (causal conv over the value history, gated add into the streams) (`src/kernels/ple.hpp:3-13`, `:35-39`).

## Choices
- The kv projection is one merged Q8 weight (fleet: [12800, 2560] Q8 G64) (`format.strixite.merge_plan`).
- Row data arrives as BF16 from pinned host buffers (`executor.strixite.qwen4exp` `qwen4exp.cpp:1041-1043`), decoded
  from the table's 8-bit rows on the host (`format.strixite.ngram_table`).

## Cost
PF-CF: the PLE stage was 22 ms of 5,523 per chunk (0.4%) (pod `/data/pod/reports/2026-10/strixite-results/pfcf.json`).
The GPU side is negligible; the cost is the SSD gather feeding it (see `runtime.strixite.ngram_table`).

## Coupled to
`runtime.strixite.ngram_table`, `runtime.strixite.ple_hash`, `format.strixite.ngram_table`, `executor.strixite.qwen4exp`.
