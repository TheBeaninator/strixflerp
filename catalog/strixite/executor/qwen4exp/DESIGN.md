# executor.strixite.qwen4exp: design record (format-facing paths only)

## What it is for
The qwen4_exp engine (Qwen3.8-Flash-Next on the fleet): loads the strixw weights and n-gram table, runs prefill in
chunks and decode with MTP drafts. This record covers only how it uses the weight format.

## Format-facing choices (origin's code)
- Thresholds: grouped experts from `kGroupedMinTokens = 6`, matrix units from `kWmmaMinTokens = 12`
  (`src/runtime/qwen4exp.hpp:724,730`). Below them the decode kernels run.
- Experts in prefill (`qwen4exp.cpp:3170-3220`): `moe_group_routes`, then the fused gather+SwiGLU (WmmaF16, Q4) and
  the grouped combine; Q4 experts take F16, "Q5 stays on BF16 scaled codes until its kernel moves over"
  (`:3185-3190`); F16 h handoff only for Q4 down with BF16 activations (`:3195-3196`).
- Shared expert: a separate pair of dense GEMVs/GEMMs (the fleet's Q8 shared expert cannot be in the Q4 stack); in
  prefill it runs first and its gated add rides in the combine's reduce, bit-identical to a separate add
  (`:3170-3182`).
- Draft head: a Q4 G64 copy of the Q8 LM head's first `mtp_vocab` (65,536) rows made at load
  (`:229-281`, used at `:3895-3906`).
- PLE: pinned host buffers for gathered rows (`:1041-1043`), prefetch during drafting/lookahead (`:1214-1248`),
  async gather for prefill chunks (`:1250-1276`).
- Integrity: refuses an n-gram table from another checkpoint (`:167-172`).

## Prefill cost (measured)
PF-CF, one node, depth 16k, chunk 8,192 (pod `/data/pod/reports/2026-10/strixite-results/pfcf.json` (card PF-CF 4546bb4318: build pfq1 ab50cc4, one node, depth 16,384, chunk 8,192, rocprofv3; an October build, before v0.5.0's grouped-expert rewrite reached the fleet)): GPU-busy 5,523 ms per chunk (1,483 tok/s of kernel time). Stages: experts
1,805 ms (33%), GDN projections 988 (18%), HC 874 (16%), GDN core 716 (13%), attention 585 (11%), attention
projections 302 (5%), shared expert 152 (3%), router 42, QSA 35, PLE 22. The fleet measures 1,188-1,245 tok/s end
to end from 16k to 128k (pod `/data/pod/reports/2026-11/ctxmove-slots.md:52-61`), so ~15-20% of wall time is not
kernel time (inferred; builds differ).

## Coupled to
Every format and kernel module in `docs/STRIXW-DESIGN.md`.

## Prefill remarks
The executor decides chunk size (`--chunk 8192` on the fleet), which math each stage gets, and when the shared
expert runs. A load-time repack would be wired here or in `strixw_loader` and selected through `runtime.strixite.qweight`.
