# kernel.strixite.qsplit: design record

## What it is for
The row layout of Q4-based expert weights as the expert kernels read it, generalized to Q5: K/2 low-plane bytes
then K/8 high-bit bytes; `QHigh32<HB>` loads the 32 high bits of a 32-code chunk and shifts them into bit 4; with
HB = 0 the code compiles away (`src/kernels/qsplit.hpp:1-31`).

## Why (origin's code)
Lets one expert-kernel loop serve Q4 and Q5 (`kernel.strixite.linear_q4` `linear_q4.hip:177-199`, `:503-528`):
Q5 support costs no Q4 speed.

## Coupled to
`format.strixite.q4`, `q5`, `kernel.strixite.linear_q4`, `moe_grouped` (Q5 grouped paths).

## Prefill remarks
Q5 experts' prefill stays on BF16 math (`executor.strixite.qwen4exp` `qwen4exp.cpp:3185-3190`); qsplit's layout is
not used by the F16 WMMA expert kernels. Unused by the fleet (no Q5).
