# format.strixite.q8: design record

## What it is for
8-bit asymmetric per-group format: the dense path's default in the fleet (attention, GDN out, HC mixes, shared
expert, PLE kv, LM head, MTP fc).

## Choices (origin's code and docs)
- Q4's scheme at 8 bits: codes [N, K] one byte each, BF16 scale and min per group, quantizer bit-identical to the
  reference (`src/formats/q8.hpp:3-11`). 8.5 bits per weight at G64.
- Chunk-major copy for the HC up-mix (32-code chunks, 32 bytes) (`q8.hpp:38-46`). Q8ChunkMajor is the only
  chunk-major encoding the fleet file uses (HC mix_up, 100 tensors).
- Why 8 bits for the dense path: every token reads it; an all-4-bit layout agreed with the full model at 194 of 242
  positions against 224 with every dense part at 8 bits (`strixite docs/weights.md:46-53`).

## Costs and limits
Q8 cannot be used for routed experts (no Q8 gather/combine; `convert_qwen4exp.cpp:346-349`), which is why the
shared expert at Q8 cannot be stacked with the Q4 routed experts (see `format.strixite.convert_qwen4exp`).

## Coupled to
`q8_device`, `kernel.strixite.linear_q8`, `linear_wmma` (Q8 F16 path), `hc` / `hc_wmma` (Q8 overloads),
`q4_from_q8` (source of the draft head), `runtime.strixite.qweight`.

## Prefill remarks
Dense Q8 projections prefill through `linear_wmma_f16_kernel` with codes dequantized to F16 in LDS; at 8 bits the
weight stream per tile is twice Q4's, but the dense path is small (~3.5 GB) and the kernel ran at ~57% of the F16
matrix peak in the origin-era profile (inferred arithmetic; see the prefill report).
