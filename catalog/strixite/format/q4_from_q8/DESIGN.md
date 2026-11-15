# format.strixite.q4_from_q8: design record

## What it is for
Q8 to Q4 requantization at load: the MTP draft head scores the LM head's first `mtp_vocab` rows on every draft call;
those rows are Q8 in the served layout (178 MB at 65,536 rows); the draft reads a Q4 copy (94 MB) made once
(`src/formats/q4_from_q8.hpp:3-6`).

## Choices (origin's code and docs)
- Defined as dequantize-the-Q8-exactly then `quantize_q4` with group G: the Q4 nearest to what the engine serves,
  not the converter's Q4 from BF16 (`q4_from_q8.hpp:8-10`).
- Row prefix: row-major storage makes the first `rows` rows a prefix of each plane (`executor.strixite.qwen4exp`
  `qwen4exp.cpp:251`), and threads quantize contiguous row chunks independently, byte-identical to one call
  (`q4_from_q8.hpp:12-13`).
- Served as G64 over 65,536 rows (`qwen4exp.cpp:276-282`; draft head selection `:3895-3906`).
- A smaller draft vocabulary took a draft from 3.9 to 1.7 ms (`strixite docs/how-it-got-fast.md:120`).

## What it buys / costs
Buys a cheaper draft (draft output only proposes; the verify decides every token, so outputs stay the full
model's). Costs 94 MB of device memory.

## Coupled to
`q8` (dequantize_q8), `q4` (quantize_q4, append_rows_q4), `q4_device` (upload), `executor.strixite.qwen4exp`.

## Prefill remarks
This is the engine's precedent for a load-time derived weight copy (read back, transform on host, upload). A
prefill repack could follow the same pattern, but on the GPU: the expert stacks are 68 GB, too large for a host
round trip at startup. (inferred)
