# kernel.strixite.linear_q56: design record

## What it is for
Decode GEMV for Q5 and Q6 weights: one kernel template over a Q4 low plane plus an HB-bit high plane per row
(`src/kernels/linear_q56.hip:1`).

## Choices (origin's code)
- Row = `K/2` low-plane bytes then `K*HB/8` high-plane bytes (`linear_q56.hip:96-98`); the loop is linear_q4's
  (16-byte low load, scale/min per group, min-fold) with the high bits OR-ed into the code (`:100-108`).
- Same split-K and group-size templating as linear_q4.

## Status in the fleet
No Q5/Q6 tensor in the fleet file; this kernel is not on the fleet's path.

## Coupled to
`format.strixite.q5`, `q6`, `kernel.strixite.qsplit` (the expert-side twin of the high-plane read),
`runtime.strixite.qweight`.

## Prefill remarks
Prefill of Q5/Q6 dense weights goes to `linear_wmma` (`linear_q5_wmma`, `linear_q6_wmma`).
