# format.strixite.q6: design record

## What it is for
6-bit asymmetric per-group format for dense classes.

## Choices (origin's code)
- Two planes per row: Q4 low plane [K/2] plus a 2-bit high plane [K/4] (`src/formats/q6.hpp:3-9`); 6.25 bits per
  weight at G128; K % 64 == 0 for 16-byte-aligned planes (`q6.hpp:12-13`).
- The converter allows Q6 for dense classes only; experts must be Q4 or Q5 (`convert_qwen4exp.cpp:346-349`), HC
  mixes not Q5/Q6 (`:350-353`).

## Status in the fleet
No Q6 tensor in the fleet file.

## Coupled to
`q4` (low plane), `q6_device`, `kernel.strixite.linear_q56`, `linear_wmma` (`linear_q6_wmma`), `runtime.strixite.qweight`.

## Prefill remarks
Only the dense WMMA path reads it; no grouped-expert kernel exists for Q6.
