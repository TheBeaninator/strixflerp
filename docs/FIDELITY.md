# Fidelity: what a recipe is measured against

## The principle

A recipe's north star is a **full-precision evaluation of the weights as shipped**: the stored (quantized) weights
dequantized, every activation in FP32, no activation quantization, no reordering that changes more than rounding.
That is the model the file defines. An engine is good to the extent its output distribution is close to that one.
Other engines on the same inputs are measured against the same reference and recorded beside it. They are
yardsticks, not criteria: holding one engine to another makes the other's rounding the target (the qwen35moe
recipe's gpu stage did that and failed a tolerance against llama.cpp that says nothing about which side is off).

A reference written here has no standing by itself. It needs a check that shares no code with it.

## The metrics

At a position, with every engine fed the same context (prompt ids plus a recorded continuation), against the
reference's distribution R:

- **KL(R || engine)** in nats, over the tokens both sides kept (the top 1,000 here), with the mass of R they hold;
- **max |log p_R - log p_engine|** over tokens with p_R >= 1e-3 (a floor: below it differences are large and do
  not matter);
- **top-1**: the engine's best token is R's; and **R's margin** between its two best, which says whether a
  disagreement would be a tie.

Stages `reference` and `fidelity` of `gate/moe_recipe.py`; criteria in `[fidelity]` of the recipe's gate config,
committed (a8658da) before any run.

## What was found (2026-11-11, control-node; `gate/results/qwen36-35b-a3b.flerp-moe/fidelity.md`)

Qwen3.6-35B-A3B Q4_K_M, 4 prompts x 4 positions, reference = the executor's CPU reference:

| engine | mean KL | largest KL | largest max-difference above the floor | top-1 is the reference's |
|---|---|---|---|---|
| strixflerp, prompt pass (the origin's prompt chunks) | 0.00099 | 0.00566 | 0.60 | 16 of 16 |
| strixflerp, prompt through the decode step | 0.00013 | 0.00078 | 0.23 | 16 of 16 |
| strixflerp, prompt pass with FP32 activations (a switch) | 0.00013 | 0.00078 | 0.25 | 16 of 16 |
| llama.cpp (CPU) | 0.00065 | 0.00212 | 0.88 | 16 of 16 |

**Verdict under the committed criteria: failed**, on two counts.
1. The long prompt's first two positions through the prompt pass: KL 0.0057 and 0.0038 against llama.cpp's 0.0013
   and 0.0007, over the allowed factor of 3 (floor 0.002). Everything is under the absolute bound of 0.02.
2. Three positions of one prompt, for both held paths, on the coverage rule: the reference's top 1,000 tokens hold
   98.1 to 99.5% of its mass there, under the 99.9% the rule asks. That is a limit of the measurement (1,000 tokens
   a side), written into the criteria before the distributions were seen. It was not changed.

**Where the prompt-pass gap comes from: activation quantization in the origin's prompt chunks.** A chunk's norm
writes BF16 rows (`catalog/gufo/executor/qwen/src/src/models/qwen/hip/prefill_chunk.cpp:249-252`), they are
quantized to tiled Q8_1, 8 bits (`:255-257`, `LaunchQuantizeActivationQ8_1`), and every projection on K-quant
weights reads that form (`:144-146`, `LaunchBatchedQuantGEMMPreQuantized`). The origin has no switch for it. The
adapted copy has one for measuring (`FLERP_PREFILL_FP32_ACTIVATIONS`: the same chunk, the projections reading FP32
rows): with it the prompt pass gives the decode step's numbers (mean KL 0.00013 both; on `rivers` position 0 the
max difference goes from 0.52 to 0.001). The MoE feed forward is not in that path: it reads FP32 rows either way.

The dense model (decider-2b) through the same binary, the origin's unmodified feed forward: prompt pass mean KL
0.0015 (largest 0.0137), decode step 0.0006, llama.cpp 0.0006. The origin's own prompt pass scores like the MoE
recipe's, and it would fail the same criteria.

## The reference's standing

- **The MoE block**: `gate/moe_block_check.py` (numpy, llama.cpp's gguf-py dequantization, float64) on 126 real
  block inputs (blocks 0, 19, 39; 42 positions): the same experts in the same order in all 126, weights within
  1.3e-7, block outputs within 5e-6 (5e-5 of their RMS). Independent of the executor's code for the K-quant
  dequantization and the MoE math; the inputs themselves come from the reference's own forward.
- **Everything else** (attention, Gated DeltaNet, norms, LM head: gufo's CPU forward): on the dense model, against
  llama.cpp run on an F32 conversion of the same file, 12 positions: log-probabilities above the floor within
  0.004 nats. Independent end to end, but on the dense model, not on the 35B file (its F32 form is about 140 GB).
- Not checked independently: the 35B file's attention and DeltaNet path as a whole.

## What is not known

- Why the decode step itself sits up to 0.23 nats from the CPU reference at a few positions of the MoE model
  (0.001 on others, and on the dense model's short prompts). Not investigated; a router choice flipping on
  rounding and the F16 KV cache on the long prompt are candidates, neither shown.
- The dense long prompt's reference is llama.cpp on F32, not the CPU reference: the two were compared on short
  prompts only.
- One run, 16 positions a model, greedy contexts from llama.cpp.

## Cost

CPU reference, 8 threads: 100 to 290 s for a short prompt and 24 positions of continuation; 4,529 s (75 min) for
the 1,808-token prompt, about 2.5 s a token. Dense: about 55 s a short prompt. GPU: one hold of 163 s.

## A note from another model: DeepSeek-V4.1's vision tower in BF16 (2026-11-11)

The same principle, one lesson. The gate of `runtime.flerp.vision_sidecar` held BF16 image rows to an FP32
evaluation of the tower by a bound derived from the format (a random walk of BF16 roundings, a row allowed cosine
0.9855), and failed: not the worker, which is bit-identical to DeepSeek's own `vision.py` in BF16, but the tower
as shipped. In BF16 a typical aligner row sits at cosine 0.9999 from FP32 (the median on all nine pictures), and a
few rows an image sit 10 to 40 times further than the median row without being short: 9 rows of 4 images under
0.9855 (`tiny` 2, lowest 0.979; `text` 1, 0.867; `cap` 2, 0.899; `over` 4, 0.935), none on the other five
pictures (lowest 0.989 to 0.996); 8 to 12% of an image's values equal the FP32 result rounded to BF16. BF16 is
the dtype DeepSeek's code runs the tower in, so the model was presumably trained and used with these rows; what
they cost downstream was not measured. Two things follow. A full-precision reference tells how far the shipped
arithmetic is from the weights' exact evaluation, and that distance can belong to the model, not to an engine:
the criterion for an implementation was restated by the owner as bit equality with the reference in BF16
(`gate/vision-dsv41-cpu312.toml`, `[restated]`), and the FP32 distance stays a record. And a bound "from first
principles" on accumulated rounding had no standing here: nothing bounded how much 32 blocks amplify an error.
Evidence: `gate/results/vision-dsv41-cpu312/summary.md` and `posthoc-bf16-rows.json`; `docs/VISION-DSV41.md`.
