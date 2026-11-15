# kernel.strixite.linear_q8: design record

## What it is for
Decode GEMV for Q8 weights: the fleet's dense path (attention, GDN out, shared expert, PLE kv, LM head).

## Choices (origin's code)
- linear_q4's structure at one code per byte: 16 codes per 16-byte load inside one group, `s*sum(x*q) + m*sum(x)`
  per chunk, split-K 1-8 waves, M == 1 instantiation (`src/kernels/linear_q8.hpp:2-5`).
- BF16 activations through `v_dot2_f32_bf16`; codes 0..255 exact in BF16; xsum as a dot2 against (1,1). The scalar
  form at M = 5 was instruction-bound (35-37% of wave cycles waiting to issue, 3+ VALU per weight per row); this is
  ~1 (`linear_q8.hip:54-60`).

## Coupled to
`format.strixite.q8` / `q8_device`, `runtime.strixite.qweight`, `executor.strixite.qwen4exp`.

## Prefill remarks
Not used at prefill sizes (>= 12 tokens go to `linear_wmma`).
