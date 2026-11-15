# format.strixite.q4: design record

## What it is for
The 4-bit weight format: definition, reference quantizer, row append, dequantizer, and the chunk-major relayout.

## Choices (origin's code and docs)
- Asymmetric per-group: `w = q * scale + min`, q in 0..15, scale and min stored as BF16 bits, one pair per group of
  G codes along K; G in {32, 64, 128} (`src/formats/q4.hpp:3-7`, `:26`).
- Separate planes: codes [N, K/2] (low nibble = even k), scales [N, K/G], mins [N, K/G] (`q4.hpp:4-5`, struct
  `:15-23`).
- Why asymmetric: "a symmetric 4-bit format lost accuracy at every group size" (`strixite docs/weights.md:69-73`;
  `how-it-got-fast.md:252`).
- Quantizer bit-identical to the reference quantizer the layouts were measured with (`q4.hpp:7-8`).
- Rows append exactly (`append_rows_q4`, `q4.hpp:34-37`): groups run along K inside a row, so concatenating rows
  never splits a group. This is what makes the load-time merges (`format.strixite.merge_plan`) and the stacked
  expert matrices exact.
- Chunk-major copy: chunk c of row n at `q[(c*N + n)*16]`, scale/min of group g at `[g*N + n]`, a separate type so a
  kernel cannot be handed the wrong one (`q4.hpp:43-55`). Its only reader is the HC up-mix (`kernel.strixite.hc`,
  `hc_wmma`). The fleet file stores HC mix_up as Q8 chunk-major, not Q4, so Q4ChunkMajor is unexercised there.

## What it buys
- 4.25 bits per weight at G128, 4.5 at G64 (header arithmetic; `weights.md:66-73`). G64 to G128 kept accuracy for
  5.6% fewer bytes and 1.9% less time per token (`strixite docs/how-it-got-fast.md:84-93`).
- Decode kernels fold the min: per 32-code chunk `acc += s*sum(x*q) + m*sum(x)`, one FMA per weight
  (`kernel.strixite.linear_q4`).

## Costs and limits
- Row-major codes with scales/mins in other planes: a GEMM tile needs three separate streams per K step
  (codes, scales, mins). (inferred)
- G is per tensor; kernels template on it (32/64/128 only).
- At G32 the scale+min overhead is 1 bit per weight (25% on top of 4); the fleet's expert-down class is G32.

## Coupled to
`q4_device` (device residency), `strixw` (encodings 1 and 2), `merge_plan` / `convert_qwen4exp` (append_rows_q4),
`q4_from_q8` (quantize_q4), `q5` / `q6` (reuse the low plane), every Q4 kernel.

## Prefill remarks
The WMMA prefill kernels do not use the min-fold; they dequantize each code to F16 once (see
`kernel.strixite.linear_wmma`, `moe_grouped`). For them the format is just "three planes, row-major"; a tile-major
variant would be a new encoding.
