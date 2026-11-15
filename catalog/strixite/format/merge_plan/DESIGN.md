# format.strixite.merge_plan: design record

## What it is for
The list of weights that read the same input and are therefore stored as one row-concatenated tensor and run as
one launch.

## Choices (origin's code)
- Why: small projections are launch-bound, "512x2560 at <50% of read bandwidth vs ~89% for 12288x2560"
  (`src/formats/merge_plan.hpp:3-6`).
- Exact by construction: rows are appended at quantization time (`append_rows_q4`), so a merged launch computes each
  row as a separate launch would, provided members share K, format and group size (`merge_plan.hpp:8-11`).
- Groups (`src/formats/merge_plan.cpp:18-40`): GDN `in_proj` = qkv + z + b + a; attention `qkv` = q + k + v (+ the
  QSA indexer's `index_qk_proj` for qwen4_exp); MoE `experts_gate_up` = routed gate_up + shared gate + up, and
  `experts_down` = routed down + shared down (the shared expert as expert #E of the stack); `router` = 512 router
  rows + the shared-expert gate row (FP32 logits, one [T, 513] output). PLE kv is also one merged weight
  (`kernel.strixite.ple` `ple.hpp:3-5`).

## What the fleet does with it
The plan proposes the shared expert as expert #512, but the converter only stacks it when its class has the
experts' format; the fleet's `sh_gu`/`sh_down` are Q8 while experts are Q4, so the stack is 512 experts and the
shared expert is separate (`shared_expert = separate` in the fleet file; `convert_qwen4exp.cpp:503-519`).

## Coupled to
`format.strixite.q4` (append_rows_q4), `convert_qwen4exp` (applies the plan), `executor.strixite.qwen4exp` (reads
slices by row offset), `kernel.strixite.linear_q4` (gate/up rows of an expert adjacent: fused gather+SwiGLU),
`moe_router` (513-wide logits).

## Prefill remarks (inferred)
Merges help decode (launch count) more than prefill, where each GEMM is large anyway. They also fix the row order
a prefill kernel sees: an expert's [gate I | up I] rows are adjacent, which the fused SwiGLU gather relies on.
