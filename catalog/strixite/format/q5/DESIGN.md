# format.strixite.q5: design record

## What it is for
5-bit asymmetric per-group format for classes that tolerate less than 8 bits but want more than 4.

## Choices (origin's code)
- Per row two planes back to back: the Q4 low plane [K/2] (same nibble order) and a 1-bit high plane [K/8]; scale
  and min as Q4 (`src/formats/q5.hpp:3-10`). 5.25 bits per weight at G128; K % 128 == 0 keeps both planes 16-byte
  aligned (`q5.hpp:13-14`).
- The low plane being Q4's packing lets the kernels reuse the Q4 loop and add the high bit
  (`kernel.strixite.qsplit`).

## Status in the fleet
No Q5 tensor in the fleet file (read 2026-11-11, see `format.strixite.strixw/DESIGN.md`). The converter accepts Q5
experts (`convert_qwen4exp.cpp:346-349`).

## Prefill remarks
Q5 experts stay on BF16 scaled-code math in prefill "until its kernel moves over"; the F16 handoff between gather
and down refuses Q5 (`executor.strixite.qwen4exp` `qwen4exp.cpp:3185-3190`; `runtime.strixite.qweight`
`qweight.hpp:122-124`). A Q5 expert layout would therefore prefill slower than Q4 today (not measured here).

## Coupled to
`q4` (low plane), `q5_device`, `kernel.strixite.linear_q56`, `qsplit`, `linear_wmma` (`linear_q5_wmma`),
`moe_grouped` (Q5 grouped entry points).
