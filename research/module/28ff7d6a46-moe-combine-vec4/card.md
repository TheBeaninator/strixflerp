TITLE: Does the qwen35moe MoE combine run near read bandwidth with 16-byte loads
AREA: module
ASKED BY: strixflerp maintainers
APPLIES TO: strixflerp: whether the vectorized combine replaces the original epilogue for prompt chunks in executor.flerp.qwen35moe
SERIES: moe-prefill
QUESTION: Does giving each thread four adjacent outputs read with 16-byte loads make the MoE combine run near read bandwidth with the same output words?
WHY: if it does, the combine stops costing 13.6 % of a 4,096-row prompt chunk and the faster kernel becomes the prompt-chunk path.
MECHANISM: the combine moves 335 MB a 35B block, yet the original kernel runs at 43-48 GB/s with one 4-byte load a thread over tokens x dim/256 blocks, while a plain read of the same bytes runs at 196 GB/s, so it is load-issue bound, not memory bound
PREDICTION: combine plus ToF16 falls from 348 ms to about 115 ms a 4,096-row chunk; confirmed at 150 ms or less, refuted above 250 ms
METHOD:
  code: strixflerp@ad7270763b753f09a811e7ae75781d9963c3e412
  subject: catalog/flerp/executor/qwen35moe (moe_combine.hip) in recipe qwen36-35b-a3b.flerp-moe
  baseline: the same build with FLERP_MOE_COMBINE_FAST unset (the original epilogue), interleaved in the same load
  settings: FLERP_MOE_COMBINE_FAST=1
  model: Qwen3.6-35B-A3B Q4_K_M (GGUF)
  boxes: 1
MODULE:
  path: catalog/flerp/executor/qwen35moe
  replaces: gufo's MoeEpilogueKernel for prompt chunks (opt-in setting; decode untouched)
  interface: existing slot executor.model/qwen35moe/gguf
  builds: the recipe's CMake, compiled into flerp_qfn_kernels with the kernels unit's options
  tests: tests/test_moe_serve_stage.py
MEASURE: combine and ToF16 ms a chunk, read from a kernel trace
  prefill tok/s at 2,048 and 16,384 tokens, from the model gate's summary
  output words differing in the kernel bench; scratch bytes from the ISA scan
EXACTNESS: bit-exact vs baseline: kernel bench output words, and the logits sha256 and generated ids of every model rep
REPS: 3 interleaved model reps, because the speed bar is 5 % and the per-rep spread was under 5 %
GIVES: the commit with the vectorized combine and its cause bench (gate/combine_bench.hip)
NOTE: Written on 2026-11-20 from the strixflerp experiment MOE-COMBINE to show a complete card. gate.toml restates gate/moe-combine.toml, committed before any hardware run (internal commit; public in strixflerp@ad7270763b75 with the combine and its bench); no threshold was changed. The raw kernel trace and gate JSON lines stay in gate/results/moe-combine/.
