# kernel.strixite.hc: design record

## What it is for
Hyper-connection mixes per token (decode, verify): `hc_mix_down` (Q8 or Q4 row-major W_down) and `hc_mix_up`
(chunk-major W_up), with norms, gates and the stream combine fused (`src/kernels/hc.hpp:38-50`).

## Choices (origin's code)
- `hc_mix_up` reads W_up chunk-major ([H*d, r], r = 320): each thread owns a row and consecutive threads' rows are
  adjacent in that layout, so a wave's 16-byte loads are one contiguous run (`hc.hpp:41-49`; `hc.hip:166-171`).
  This is the only consumer of the chunk-major encodings (`format.strixite.q4` `q4.hpp:43-55`, `q8` `q8.hpp:38-46`).
- Loads each weight chunk once and dots it with every token's h in LDS; at T = 5 (Q8, the served verify) 27.2 us
  with a block per token -> 25.5 us (`hc.hip:166-171`).

## Fleet
HC mix_up is Q8 chunk-major G64 and mix_down Q8 row-major G64 in the fleet file.

## Coupled to
`format.strixite.q4`, `q8` (chunk-major types), `q8_device`, `kernel.strixite.hc_wmma` (same math, prefill),
`executor.strixite.qwen4exp` (`qw_chunk_major` load, `qwen4exp.cpp:51`).

## Prefill remarks
Prefill uses `hc_wmma` from 12 tokens. The chunk-major layout is a precedent for storing a kernel-specific layout
in the file: here the up-mix's per-row ownership dictated it.
