# runtime.strixite.qweight: design record

## What it is for
A dense or expert weight in whichever format the layout gave it (Q4, Q5, Q6, Q8 row-major), with `linear_qw*`
dispatching to the matching kernel; the format is read from the strixw file at load, never guessed
(`src/runtime/qweight.hpp:1-5`).

## Choices (origin's code)
- One switch on `bits` per call site: decode GEMVs, WMMA prefill, grouped experts.
- Grouped combine with F16 h needs Q4 experts; Q5 is refused there (`qweight.hpp:118-124`).

## Coupled to
All format device views, `kernel.strixite.linear_q4`, `linear_q56`, `linear_q8`, `linear_wmma`, `moe_grouped`,
`format.strixite.strixw_loader` (`qw()`), `executor.strixite.qwen4exp`.

## Prefill remarks
The place where a prefill-only layout would be selected: a `QWeightView` could carry an optional second (tile-major)
view used by `linear_qw_wmma` / the grouped kernels while decode keeps the row-major one. (inferred)
