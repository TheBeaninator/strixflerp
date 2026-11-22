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
LIMITS:
  model reps = 3 interleaved reps a prompt length (G,C,G,C,G,C in one load), median taken; source: chosen; why: the speed bar is 5 % and the per-rep spread was under 5 %; changes-outcome: yes; justified: the median of 3 absorbs one slow rep and the mechanism predicts about 1.11x, twice the bar
  kernel bench reps = 7 a dim, caches flushed before each; source: chosen; why: the cause rule compares bandwidths the mechanism puts about 4x apart; changes-outcome: yes; justified: 43-48 against 196 GB/s is far outside the spread of 7 flushed reps
  speed bar = candidate tok/s at least 1.05x baseline at 2,048 and at 16,384 tokens; source: chosen; why: smaller gains sit inside the per-rep spread on one box; changes-outcome: yes; justified: the mechanism predicts about 1.11x, so 1.05x separates a real gain from noise
  prediction lines = confirmed at 150 ms or less, refuted above 250 ms, combine plus ToF16 a chunk; source: chosen; why: from the mechanism's ~80 ms combine and ~35 ms ToF16; changes-outcome: yes; justified: set from the mechanism in gate/moe-combine.toml before any run and never moved
  identity bar = 0 differing output words, logits sha256 or generated ids; source: chosen; why: a speed lever must not change any output; changes-outcome: yes; justified: the new kernel computes the same expression in the same order, so any differing word is a real change
  generated tokens = 2 a prompt (--max-tokens 2); source: chosen; why: identity is read at the prompt's last position and one decode step; changes-outcome: yes; justified: the new combine runs only in prompt chunks and decode is untouched, so those two steps cover every output it can change
  context = 16,640 tokens; source: chosen; why: holds the 16,384-id prompt and its 2 generated tokens; changes-outcome: no
  sampling = greedy (argmax), no temperature, top_p or top_k; source: chosen; why: identity compares generated ids, which only greedy decoding fixes; changes-outcome: no
  trace warm-up = profile passes 1-2 dropped, median of passes 3-5; source: chosen; why: the first passes pay code-object load and cold caches; changes-outcome: yes; justified: in the earlier prefill profile one warm-up pass left the first measured pass still warming, so two are dropped (gate/prefill-kprof.toml amendment 1, before this run)
  hang guard = model gate process timeout 3,000 s; source: chosen; why: a hang ends the process with no result file, so it counts as a harness fault, never as a speed or identity result; changes-outcome: no
GIVES: the commit with the vectorized combine and its cause bench (gate/combine_bench.hip)
NOTE: Written on 2026-11-20 from the strixflerp experiment MOE-COMBINE to show a complete card. LIMITS was added the same day, after the run: every line is a limit the run used, read from gate/moe-combine.toml, gate/lever_gate.py and gate/prefill-kprof.toml as committed before it. gate.toml restates gate/moe-combine.toml, committed before any hardware run (internal commit; public in strixflerp@ad7270763b75 with the combine and its bench); no threshold was changed. The raw kernel trace and gate JSON lines stay in gate/results/moe-combine/.
