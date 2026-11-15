# Interfaces: where the engines overlap

This is the overlap map of the catalogue after the phase 2 harvest: for each job, who provides it, with what weight
format, activation dtype, numeric contract and memory-ownership assumption, and whether two providers can be swapped.
It is written from the kernels' own header comments and launch signatures; every row cites the origin file.
Verdicts are of two kinds and say which: **judged** (what the sources say; nothing was run) and **measured**
(phase 4 ran it on control-node's gfx1151 and the evidence path is given). The five swaps that were measured are
collected in "The five swaps, measured" at the end; every other verdict is still judged.

Paths are origin-relative. In the tree they are under `catalog/<origin>/<kind>/<module>/src/<path>`. `flerp interfaces`
prints the same names with their providers and consumers.

## How to read the names

`<family>.<op>/<weight-format-or-dash>/<activation-dtype-or-dash>`, with these conventions (the ones beyond SPEC.md
are proposed in `docs/SPEC-PROPOSALS.md`):

- The family and op words are shared across origins (`linear.decode`, `linear.prefill`, `linear.dense`,
  `moe.router`, `moe.experts-decode`, `attn.softmax-gqa`, `attn.sparse-sink`, `attn.indexer-score`,
  `attn.qsa-indexer`, `mixer.gdn`, `hc.mix`, `norm.rmsnorm`, `pos.rope`, `kvcache.store`, `logits.topk`,
  `residual.add`, `collective.*`). Same words, same job.
- The weight-format field carries the storage format the inner loop is written for (`strixw-q4`, `exl3`,
  `gguf-kquant`, `hipfire-mq`, ...). For attention it carries the KV storage format (`kv-f16`, `kv-fp4-e2m1`).
- The dtype field is the activation dtype; `in-out` when they differ (`bf16-f32`).
- A `.tag` after the dtype (`f32.gufo`, `bf16.triton-bits`) marks a provider whose numeric contract or operand
  layout is not equal to another provider with the same first three fields. **Two modules carry the same full name
  only when one can stand in for the other in a recipe.** No two modules of one origin do today: where an origin
  compiles interchangeable alternatives into one binary and picks at run time, each has its own name, because
  `flerp assemble` refuses an interface that two modules of one assembly provide. Those are StrixFold's three
  collective transports (`collective.allgather-tcp/-/f32`, `-rdma`, `-nccl`, behind one vtable) and its two RoPE
  kernels (`pos.rope-heads/-/bf16.hip`, the HIP kernel, and `pos.rope-heads/-/bf16.triton-bits`, the Triton
  object the origin tests it against bit for bit); `executor.strixfold.dsv41` requires all five.
- So an overlap shows up as a shared prefix (the overlap class, SPEC.md section 2), and the tag is the gap that an
  adapter or a parity gate has to close. Adapters are modules under `catalog/flerp/kernel/`; `flerp interfaces`
  lists each with the foreign module it launches.

Model families matter for every row below: strixite, gufo `qwen38_flash_next` and hipfire `arch-qwen4` serve
**Qwen3.8-Flash-Next**; StrixFold `dsv41`, gufo `deepseek_v4_flash` and hipfire `arch-deepseek4` serve
**DeepSeek V4 / V4.1 Flash**. A kernel for hyper-connections of 4 x 2560 (Qwen) is not a provider for 4 x 5120 or
4 x 4096 (DeepSeek), whatever its name.

Memory ownership, once, since it is the same per engine in every table:

| engine | who owns device buffers | how a kernel receives them | how kernels are loaded |
|---|---|---|---|
| strixite | the caller (executor); kernels never allocate; "y must not overlap x" is a stated precondition | raw device pointers + sizes + `hipStream_t` in a C++ launch function (`kernels/*.hpp`) | statically linked translation units |
| StrixFold | `model.zig` owns every buffer of a rank; the kit owns nothing | one by-value C struct of `uint64_t` device addresses per kernel (`zig/kernels/hip/dsv41/abi.h`), or plain fixed-width parameters; Triton kernels take Triton's parameter list | raw `.hsaco` objects from a manifest-verified kit (`zig/src/gpu/hip/kit.zig`) |
| gufo | the model executor (`hip/arena.cpp`, `kernels/rocm/device_model.cpp`); scratch workspaces are passed in by the caller | device pointers + shape in `gufo::hip::Launch*` host functions (`src/families/qwen/kernels/ops/*.hpp`), or a C API (`ds4_gpu_*`, `qfn_mmq_*`) | statically linked translation units; DS4 keeps static device state inside its one TU |
| hipfire | the runtime (`GpuTensor` in `crates/rdna-compute`); kernels never allocate | `extern "C"` kernel parameters packed as kernargs by typed launch wrappers (`crates/hipfire-kernel-abi`) | JIT by `hipcc --genco` into a content-keyed cache, or embedded `.hxaco` bundles |

## 1. Decode GEMV (small M, memory-bound)

| provider | weight format | activations | numeric contract | source |
|---|---|---|---|---|
| `kernel.strixite.linear_q4` / `linear_q56` / `linear_q8` | strixw Q4 / Q5 / Q6 / Q8: group-affine codes, bf16 scale and min per group of 32..128 | bf16 or f32 in, same or f32 out | fp32 accumulate as `s * sum(x*q) + m * sum(x)` per chunk, split-K partials meet in LDS, one rounding on store; all ranks of a TP run use this order (bit identity across ranks) | `kernels/linear_q4.hpp`, `linear_q5.hpp`, `linear_q6.hpp`, `linear_q8.hpp` |
| `kernel.strixfold.exl3` | EXL3 trellis tiles (16 x 16, K2 half-bits a value), decoded exactly per wave | fp16 rotated input, fp32 out | `v_dot2_f32_f16`, fp32 accumulate in a fixed order; **row-invariant** (a row's bits do not depend on the window); meets the `exl3.linear` tolerance contract, not the CUDA build's bits | `zig/kernels/hip/dsv41/exl3.hip:1-24`, `tools/dsv41-hip/numerics/numerics.json` |
| `kernel.gufo.qwen_linear_decode` | GGUF blocks: Q8_0 (f16 scale + 32 int8), Q4_K / Q5_K / Q6_K (QK 256), Q8_K | f32 in, f32 out | weights decoded in-kernel, fp32 accumulate; the Q8_K path quantizes the activation to int8 and uses an integer MAC | `src/families/qwen/kernels/gemv.hip`, `gemv_quant.hip`, `ops/gemm.hpp:92-103` |
| `kernel.gufo.qfn_mmq`, `kernel.gufo.ds4_mmq` | GGUF K-quants | f32 | ggml `mul_mat_vec_q` ported from llama.cpp, `-ffast-math` | `src/models/*/kernels/rocm/mmq/mmvq.hip.cpp`, `mmq/VENDOR.md` |
| `kernel.gufo.ds4_linear` | GGUF Q8_0, F16 | f32 (optional published F16 mirror) | fp32 accumulate, the scalar kernel's multiply/FMA order kept under fast-math | `src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_q8.hip.hpp`, `ds4_rocm_q8_decode.hip.hpp` |
| `kernel.hipfire.gemv_ggml` | GGUF Q8_0, Q4_K (144 bytes per 256 weights) | f32 in, f32 out | weights decoded in-kernel, fp32 accumulate, one warp per row with shuffle reduction | `kernels/src/gemv_q8_0.hip:7-15`, `gemv_q4k.hip:22-31` |
| `kernel.hipfire.gemv_hfq` / `gemv_mq` / `gemv_mfp4` | HFQ (f32 scale and zero per group), MQ (FWHT-rotated HFQ, V2 = two fp16 scale pairs), MFP4-G32 | f32 | fp32 accumulate; MQ needs the input rotated first (`rotate.fwht`) | `crates/rdna-compute/src/kernels.rs:2515-2554`, `kernels/src/gemv_*.hip` |
| `kernel.hipfire.pm_gfx1151` | MQ4 / MQ6 (prebuilt) | f32 | certified equal to its hipcc twin on a shape contract | `kernels/mq4_rows_x4_pm_gfx1151.hxaco`, `*.contract.json` |

Verdicts:
- strixw, EXL3, GGUF and hipfire formats: **not a drop-in across formats** (a kernel binary reads one format; SPEC
  fact 2). The shape - split-K mapping, min-fold, gate+up pairing - transfers; see `docs/SHAPE-TRANSFERS.md` R1.
- `linear.decode/gguf-kquant/f32.gufo` and `linear.decode/gguf-kquant/f32.hipfire`: **swappable with the adapter
  `kernel.flerp.gguf_gemv_hipfire`, measured at module level** (swap 3 below). Same weight bytes (GGUF Q8_0 and
  Q4_K blocks as stored), same fp32 x and y, same shape arguments. Correction to the judged version: the launcher
  to adapt is `LaunchQ8KBlockGEMV(A, type, x, y, M, K, stream)` (`ops/gemm.hpp:99-103`, `gemv_quant.hip`), the
  one that reads packed blocks; `LaunchGEMV` routes to it for quantized types and aborts on the rest
  (`gemv.hip:176-196`). The sums are added in a different order, so the outputs are not bit-equal. No recipe of
  this repository consumes a GGUF GEMV yet, so there is no token-level result.
- The ggml-derived mmvq kernels in gufo and hipfire's `ggml_mmvq` port come from the same llama.cpp file, but
  hipfire's port is shipped only as a gfx1201 object (`kernels/ggml_mmvq_shim_gfx1201.co`) and did not compile for
  gfx1151 in the gate: **not swappable today.**

## 2. Dense BF16 GEMV (router logits, fine-tuned projections)

| provider | weights | activations | numeric contract | source |
|---|---|---|---|---|
| `kernel.strixite.linear` (`linear.dense/bf16/{bf16,f32}`) | raw bf16 `[N, K]` row-major | bf16 or f32 in; out in the same dtype, or bf16 in and f32 out with the accumulator stored unrounded | fp32 accumulate, one rounding at the end. The header says "deliberately simple and untuned"; the source has a tuned decode path for aligned operands with `K % 8 == 0`: one `fmaf` per element, 32 lanes, split over KS waves when `K > 1280` (`kernels/linear.hip:62-142`), and the simple strided kernel only otherwise | `kernels/linear.hpp:1-25`, `kernels/linear.hip` |
| `kernel.strixfold.dense_bf16` (`linear.dense/bf16/bf16-f32.row-invariant`) | raw bf16 `[N, K]` | bf16 in, f32 out stored or accumulated | products exact in fp32, one `fmaf` per element in fixed lane order, 32 partials summed by xor shuffles 16, 8, 4, 2, 1; row-invariant | `zig/kernels/hip/dsv41/dense.hip:1-22,122-124` |
| `kernel.gufo.qwen_linear_decode` (`linear.dense/bf16/f32.gufo`) | raw bf16 | f32 in, f32 out | fp32 accumulate | `src/families/qwen/kernels/gemv.hip`, `ops/gemm.hpp:92-97,141` |
| `kernel.hipfire.gemv_float` (`linear.dense/bf16/f32.hipfire`) | raw bf16 `[M, K]` | f32 in, f32 out | bf16 widened by a 16-bit shift, fp32 accumulate | `kernels/src/gemv_bf16_xf32.hip:7-36` |

Verdict: **swappable with the adapter `kernel.flerp.dense_bf16_strixfold`, measured** (swap 4 below) for the
bf16-in, f32-out case (strixite's router logits: `out_act = F32`), between strixite and StrixFold. Same weight
layout, same input dtype, same unrounded fp32 output. The adapter maps
`linear_bf16w(x, w, y, M, N, K, act, out_act, stream)` onto
`tf_ds_dense_gemv(w, x, y, k, n, rows, ldx, ldy, accumulate = 0)`. Measured: the two kernels give the same bits
when strixite does not split K (K = 1280), and differ in the last fp32 bits at the router's K = 2560, where each
kernel's bits are reproduced by a CPU emulation of its own documented order, so the order is the whole
difference. StrixFold's kernel needs K to be a multiple of 128 and 16-byte aligned operands; outside that the
adapter hands the call back to strixite's kernel. The gufo and hipfire kernels take f32 input, so they need an
extra cast in front of a bf16 pipeline: judged **swappable with a launch adapter + `cast-bf16-f32`**, at the cost
of one more launch; not attempted.

## 3. Prefill GEMM (many rows, compute-bound on the matrix units)

| provider | weight format | activations | numeric contract | source |
|---|---|---|---|---|
| `kernel.strixite.linear_wmma` | strixw Q4 / Q5 / Q6 / Q8, bf16 | x rounded to bf16 | raw codes on `v_wmma_f32_16x16x16_bf16`, group scale and min applied in fp32, fp32 accumulate, one rounding of y; tiles 64 x 128, K % 64 == 0 | `kernels/linear_wmma.hpp:1-12`, `kernels/wmma.hpp` |
| `kernel.strixfold.exl3` (+ Triton `_gemm` in `kernel.strixfold.triton_kit`) | EXL3 | fp16 rotated rows | `v_wmma_f32_16x16x16_f16`, fp32 accumulate; `exl3.linear` contract | `zig/kernels/hip/dsv41/exl3.hip`, `tools/dsv41-hip/tri/tri_kernels.py` |
| `kernel.gufo.qwen_linear_prefill` | GGUF K-quants; bf16 | bf16, fp16, or Q8_1-quantized int8 tiles | direct on quant blocks (no dequantize pass); the fp16 route rounds weights to fp16 in-kernel; hipBLASLt route for bf16 weights | `src/families/qwen/kernels/prefill_quant_gemm.hip`, `prefill_fp16.hip`, `hipblaslt_gemm.hip`, `ops/gemm.hpp:178-200`, `ops/prefill_fp16.hpp:20-22` |
| `kernel.gufo.qfn_kernels` (`.w8a8`) | GGUF Q8_0 | int8 tiles (activations quantized per 32-block) | int8 WMMA, integer accumulate, one fp scale per block pair | `src/models/qwen38_flash_next/kernels/rocm/kernels.hpp:224-226` |
| `kernel.gufo.qfn_mmq`, `ds4_mmq` | GGUF K-quants | Q8_1 | ggml mmq | `mmq/mmq.hpp`, `mmq/VENDOR.md` |
| `kernel.hipfire.gemm_hfq` / `gemm_mq` / `gemm_mfp4` / `gemm_ggml` / `gemm_float` | HFQ, MQ, MFP4, Q8_0, F16 / BF16 | f32 in; f16 or int8 tiles inside | f16 WMMA with fp32 accumulate, or exact-integer (iu4 / i8) WMMA after a per-128-block activation quantizer | `kernels/src/gemm_*.hip`, `block_i8_128_quant.hip`, `crates/rdna-compute/src/kernels.rs:5709-5730` |

Verdict: **not a drop-in** across engines; shape transfer: integer-activation and bf16 GEMM routes (SHAPE-TRANSFERS R5,
R15). Every provider rounds activations at a different point (bf16, fp16, int8
per 32, int8 per 128) and three of them quantize activations, which changes the result by more than summation order.
Inside gufo the Q8_0 W8A8 route and the mmq route are close (both Q8_1 activations), but they are one executor's
internal routes, not catalogue-level alternatives. The only cross-engine case worth a gate later is bf16-weight GEMM:
strixite `linear.prefill/bf16/*` against gufo's hipBLASLt route (library plan order; tolerance parity).

## 4. MoE: router, experts, combine

| provider | job | weight format | activations | numeric contract | source |
|---|---|---|---|---|---|
| `kernel.strixite.moe_router` | top-k + renormalized weights + shared slot | - | bf16 or f32 logits | all fp32; selection on logits, ties to the lower expert; `w_j = exp(l_j - l_max) / sum`; coef stored fp32 | `kernels/moe_router.hpp:1-38` |
| `kernel.gufo.qfn_kernels` (`moe.router/-/f32.qfn`) | `RouterTopK` | - | f32 | top-k on fp32 logits | `src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp` (kernel `RouterTopK`) |
| `kernel.gufo.ds4_moe` (`.ds4`) | router | - | f32 | only the final score is rounded (transcendentals differ across ROCm libraries) | `detail/ds4_rocm_router.hip.hpp:1-2` |
| `kernel.hipfire.moe` (`.hipfire`) | `moe_softmax_topk_k8`, `moe_topk_renorm_*` | - | f32 | fp32 softmax and top-k renormalization, k fixed per kernel (8, 10) | `kernels/src/moe_softmax_topk_k8.hip`, `moe_topk_renorm_k8.hip` |
| `kernel.strixfold.triton_kit` (`moe.router/-/bf16-f32`), `kernel.strixfold.moe_gate` | gate logits + route | fp16 gate weights | bf16 x | Triton bits; HIP gate: exact hi/lo bf16 split of fp16 weights, WMMA fp32 accumulate within gamma(5120) | `zig/kernels/hip/dsv41/aux_gate.hip:1-12` |
| `kernel.strixite.linear_q4` (`moe.experts-decode`), `moe_grouped` (`-prefill`) | experts | strixw Q4 / Q5 | bf16 / f32 | fp32, one rounding on store; grouped form differs only in summation order | `kernels/moe_grouped.hpp:1-14` |
| `kernel.strixfold.exl3`, `exl3_pf_experts` | experts | EXL3 | bf16 | `exl3.experts_grouped` contract; deterministic combine (fma chain in slot order) | `zig/kernels/hip/dsv41/exl3_pf.hip:1-15` |
| `kernel.gufo.ds4_moe`, `qfn_mmq` (`mmid`) | experts | GGUF IQ2_XXS + Q2_K, K-quants | f32 | per format | `detail/ds4_rocm_moe.hip.hpp`, `ds4_rocm_iq2_gate.hip.hpp`, `ds4_rocm_q2_down.hip.hpp` |
| `kernel.hipfire.moe`, `pm_gfx1151` | experts | HFQ, MQ, MFP4, E8P | f32 | per format | `kernels/src/gemv_*_moe_*.hip`, `kernels/*_pm_gfx1151.hxaco` |
| `kernel.flerp.moe_kquant` (`moe.experts-decode/gguf-kquant/f32.indexed`, `moe.gate-logits/gguf-f32/f32`) | experts of the ids a device buffer holds (one projection, or gate/up with SwiGLU), any number of tokens; router logits with the shared gate logit beside them | GGUF expert stacks (last dimension the expert): Q8_0, Q8_K, Q3_K..Q6_K, IQ4_NL, IQ4_XS, IQ3_S, F32, BF16, F16; F32 router | f32 | gufo's `QuantWarpBlockDot` per selected row (fp32 activations, fp32 accumulate), one wave a row; owns no memory, sizes are arguments | `flerp/moe_kquant/moe_kquant.hpp` (written here; consumer `executor.flerp.qwen35moe`) |

Verdicts:
- Experts: **not a drop-in** (weight formats; job 1's argument); shape transfer: expert scheduling and i8-WMMA routes
  (SHAPE-TRANSFERS R2, R5).
- Router top-k for Qwen3.8-Flash-Next (swap 5 below, measured):
  - strixite against gufo qfn: **swappable with the adapter `kernel.flerp.moe_router_gufo`.** fp32 logits rows
    `stride` apart, `ids [M, k]` int32 and weights `[M, k]` fp32 on both sides. The pod's checkpoint keeps the
    shared expert separate (`inspect_strixw`: `shared_expert separate`), so strixite's router runs without its
    shared slot and the layouts are equal; with a stacked shared expert strixite's rows are `k + 1` wide and the
    adapter hands the call back. gufo selects on probabilities, not logits, and has no rule for a non-finite
    logit (the adapter adds strixite's with one more launch). Measured: the same ids as the reference in every
    case, exact ties and one-ulp near ties included; weights within 6 fp32 steps, not bit for bit.
  - strixite against hipfire: **not swappable**, three reasons, the first decisive. (1) k: the checkpoint routes
    `num_experts_per_tok = 10`; hipfire's softmax-top-k kernel is compiled for k = 8, and its top-10 kernel
    (`moe_topk_renorm_top10_batched.hip`) takes pre-softmaxed probabilities in rows exactly `n_exp` apart, where
    strixite's logits rows are `E + 1` apart and no softmaxed copy exists: it would need a softmax launch and a
    scratch buffer that strixite's `moe_router` signature does not carry (operand layout and memory ownership).
    (2) Tie rule, measured at k = 8: hipfire's arg-max keeps the lower thread on a tie (thread t holds experts
    t, t + 256), so with 16 equal top logits it returns experts 256, 260, 100, 356, .. where strixite's contract
    gives 2, 5, 31, 44, ..: a different expert set, in 43 of 64 rows of the tie case. (3) One token per launch for
    the k = 8 kernel (its batched twin needs rows `n_exp` apart): 17 ms for a 2,048-token chunk against 0.09 ms.
- StrixFold's router and gate are split across TP ranks and work on bf16 with Triton's bits: **not swappable** with
  the single-device fp32 routers.

## 5. Sparse-attention indexer and QSA

| provider | model | keys | numeric contract | source |
|---|---|---|---|---|
| `kernel.strixite.qsa` (`attn.qsa-indexer/-/{bf16,f32}`) | Qwen3.8-FN | 4-token block keys, activation dtype | block mean in fp32 then cast; `score = (1/sqrt(D)) sum_h relu(q_h . K_b)` fp32; top-k a set, ties to the lower block | `kernels/qsa.hpp:1-20` |
| `kernel.gufo.qfn_kernels` (`.qfn`) | Qwen3.8-FN | f32 pooled blocks | `PoolBlocks`, `SelectScore`, `CompactSelection`, fp32 | `kernels.hip.cpp` |
| `kernel.hipfire.indexer_topk`, `pm_gfx1151` (`bf16+f32`) | Qwen3.8-FN | bf16 pooled keys | "exact serial-FMA BF16 pooled-key scoring"; exact sampled-threshold selection with certified compaction | `crates/rdna-compute/src/kernels.rs:9-13`, `kernels/src/qsa_bf16_scores.hip`, `qsa_filter_select.hip` |
| `kernel.strixfold.indexer_score`, `aux`, `triton_kit` | DSV4.1 | FP4 (E2M1, E8M0 scale per 32) compressed keys, 32 heads x 128 | `sum_h relu(q_h . k) * w_h` fp32 in a fixed head order; int64 keys for top-k (score bits above, ~index below) | `zig/kernels/hip/dsv41/aux_index.hip:1-16`, `aux.hip` |
| `kernel.gufo.ds4_indexer` | DSV4 | f32 keys, 64 heads x 128 | fp32; exact MSD radix top-k, +0 and -0 tie then lower index | `detail/ds4_rocm_indexer_score.hip.hpp:1-2`, `ds4_rocm_indexer_topk.hip.hpp:1-2` |
| `kernel.hipfire.indexer_topk` | DSV4 | f32 compressed keys | ReLU scores (scalar and WMMA), top-k bounded / two-stage | `kernels/src/indexer_relu_score_batched.hip`, `indexer_top_k*.hip` |

Verdicts:
- Scores: **not swappable** between StrixFold (FP4 keys, per-head weights `w_h`, 32 heads) and gufo / hipfire DS4
  (f32 keys, 64 heads): different key storage and different head counts, which also says the two engines are not
  serving the same checkpoint layout.
- Top-k selection on a score row is exact everywhere, with the same tie rule in the three that state one.
  StrixFold `attn.indexer-topk/-/i64-keys` against gufo `attn.indexer-topk/-/f32.radix`: **swappable with the
  adapter `score-keys`** (fp32 score to StrixFold's int64 key and back). It is exact, so parity is equality, but it
  only pays off once the score kernels agree, so it is not in the first five.
- QSA for Qwen3.8-FN: strixite's fp32 scores on activation-dtype keys against hipfire's serial-FMA bf16 scores:
  **not swappable as is** (FMA placement differs, and hipfire selects with a sampled threshold plus repair where
  strixite ranks). The selected set should match except on ties; a parity gate on the set is the cheap way to know.

## 6. Gated DeltaNet

| provider | state | numeric contract | source |
|---|---|---|---|
| `kernel.strixite.gdn` (`mixer.gdn/-/{bf16,f32}`) | activation-dtype inputs, fp32 gates | conv width 4 + SiLU in fp32, `qkv_out` rounded once, beta and g fp32; q/k not normalized here | `kernels/gdn.hpp:1-14` |
| `kernel.gufo.qwen_gdn` (`.gufo`), `kernel.gufo.qfn_kernels` (`.qfn`) | fp32 state | each row's arithmetic identical to the one-row form (bit-exact across batch shapes); rollback keeps exact fp32 keys / decays / betas | `src/families/qwen/kernels/ops/ssm.hpp:26-29,76-83`, `kernels.hpp:26-27` |
| `kernel.hipfire.gdn` (`.hipfire`) | fp32, Q8 or Q4 state | `S[128 x 128]` per head tiled in 8-row chunks; chunked WMMA forms for gfx1151 | `crates/rdna-compute/src/kernels.rs:9578-9582`, `kernels/src/gated_delta_*.hip` |

Verdict: **not a drop-in**; shape transfer: hipfire's chunked delta-rule tiling (SHAPE-TRANSFERS R10). The three split
the mixer at different places (strixite: front here, delta rule in its
executor's own kernel; gufo: projections fused with the recurrence; hipfire: optional quantized state) and keep the
recurrent state in different dtypes. A GDN swap is an executor-level change.

## 7. Hyper-connections

| provider | model | weights | numeric contract | source |
|---|---|---|---|---|
| `kernel.strixite.hc`, `hc_wmma` | Qwen3.8-FN, 4 x 2560, rank 320 | strixw Q4 / Q8 with the norm weight folded in | everything fp32 inside, `u` rounded once; prefill form rounds x and h to f16 | `kernels/hc.hpp:1-14`, `hc_wmma.hpp:1-12` |
| `kernel.gufo.qfn_kernels` (`hc.mix/gguf-q8_0/f32`) | Qwen3.8-FN, 4 x 2560, rank 320 | GGUF Q8_0 | fp32 streams; fused decode form byte for byte the separate launches; `-ffast-math` | `kernels.hpp:61-176` |
| `kernel.hipfire.hc` | Qwen4 and DS4 | hipfire formats | fp32; Sinkhorn 4 x 4 | `kernels/src/hc_*.hip`, `qwen4_hc_down_fused.gfx1151.hip` |
| `kernel.strixfold.mhc_mix`, `triton_kit` | DSV4.1, 4 x 5120 | fp32 mix weights | fp32 fma chains in column order; the HIP kernel is within `hc_mix_fp32_bound` of the Triton one, not bit-equal | `zig/kernels/hip/dsv41/aux_hc.hip:1-24` |
| `kernel.gufo.ds4_hc` | DSV4, 4 x 4096 | GGUF F16 | fp32 row; fused pre-block for the supported shape | `detail/ds4_rocm_hc.hip.hpp`, `ds4_rocm_hc_output_launch.hip.hpp` |

Verdict: **not a drop-in**; shape transfer: fused HC forms and the 128-row pre-block (SHAPE-TRANSFERS R9, R14).
Weight formats differ within a model family (strixw against Q8_0), and strixite folds
the norm weight into the quantized rows where gufo keeps it separate. Note the hidden sizes: StrixFold's kernel is
`d5120` and gufo's DS4 pre-block is 4 x 4096, so those two are not the same model configuration.

## 8. Norm and RoPE

| provider | job | dtype | numeric contract | source |
|---|---|---|---|---|
| `kernel.strixite.norm` (`norm.rmsnorm-zc`, `-gated`) | RMSNorm, weight `(1 + w)` | bf16 / f32 | fp32 math, casts at the reference's points | `kernels/norm.hpp:1-14` |
| `kernel.strixfold.aux` (`.torch-order`), `triton_kit` (`.triton-bits`) | RMSNorm | bf16 | torch arithmetic at torch's rounding points / Triton's bits | `zig/kernels/hip/dsv41/aux.hip:1-13` |
| `kernel.gufo.qwen_norm_rope` (`.gufo`), `qfn_kernels` (`.grouped`), `ds4_norm_rope` (`.ds4`) | RMSNorm, plain weight | f32 | `x / sqrt(mean(x^2) + eps) * weight` | `ops/norm_residual.hpp:12-17`, `kernels.hpp:55-56` |
| `kernel.hipfire.norm_rope` (`.hipfire`) | RMSNorm | f32 | fp32 | `kernels/src/rmsnorm.hip` |
| `kernel.strixite.rope` (`pos.rope`) | partial RoPE, rotate-half pairs `(i, i + rot/2)` | bf16 / f32 | transformers' arithmetic, products and sum rounded separately, no FMA | `kernels/rope.hpp:1-11` |
| `kernel.strixfold.rope_heads` (`pos.rope-heads/-/bf16.hip`) and `triton_kit` (`pos.rope-heads/-/bf16.triton-bits`) | RoPE on heads, interleaved pairs `(2j, 2j + 1)` of the last dims | bf16 | `fma(x_e, c, -(x_o s))`, each rounded to bf16; HIP kernel bit-equal to Triton's `_rope_heads` | `zig/kernels/hip/dsv41/aux_rope.hip:1-12` |
| `kernel.strixfold.aux` (`.complex-fma`) | RoPE, CUDA FMA placement | bf16 | `rope_complex_fma` | `zig/kernels/hip/dsv41/aux.hip` |
| `kernel.gufo.qwen_norm_rope`, `ds4_norm_rope` | RoPE, tail rotation | f32 | fp32 | `ops/token.hpp:123-131`, `detail/ds4_rocm_norm_rope.hip.hpp` |
| `kernel.hipfire.norm_rope` | RoPE: partial, tail, interleaved, half-split, YaRN, mRoPE | f32 | fp32 | `kernels/src/rope_*.hip` |

Verdicts:
- `pos.rope-heads/-/bf16.hip` for `pos.rope-heads/-/bf16.triton-bits`: **swappable as is** (the HIP kernel for
  the Triton object). The two are one overlap class under two tags because the kit holds both and the executor
  requires both; the origin already has the bit test (`tools/dsv41-hip/tri/t_hiprope.py`, not harvested).
- RMSNorm across engines: **swappable with the adapter `norm-weight`** between the fp32 providers (gufo and hipfire:
  same formula, plain weight), not bit-equal in general because of `rsqrt` against `1 / sqrt` and the mean's
  summation order. strixite's needs `w' = 1 + w` and has bf16 rounding points that the fp32 kernels do not: **not
  swappable** in a bf16 run.
- RoPE: **not a drop-in** across engines; shape transfer: re-expressible, low value (SHAPE-TRANSFERS R12). Pair layout (rotate-half against interleaved), which dims rotate (first
  against tail), and FMA use all differ, and each engine matches its own reference's bits.

## 9. Quantized KV

| provider | K / V storage | read by | numeric contract | source |
|---|---|---|---|---|
| `kernel.strixite.kv8` (`kvcache.quant8`) | K: 8 bits per channel over groups of 32 rows; V: 8 bits per block of 32 channels; fp16 scale and zero | `kernel.strixite.attention` | `value = fma(code, scale, zero)` fp32; newest incomplete group kept unquantized in a tail | `kernels/kv8.hpp:1-14` |
| `kernel.strixfold.triton_kit` (`kvcache.store/fp4-e2m1/bf16`) | FP4 E2M1 codes, E4M3 scale per 16 (attention) or E8M0 per 32 (indexer) | `kernel.strixfold.sparse_attn`, `indexer_score` | exact decode to bf16 | `zig/kernels/hip/dsv41/aux_attn.hip:8-10`, `aux_index.hip:8-9` |
| `kernel.gufo.ds4_attention` (`kvcache.store/fp8-e4m3/f32`) | FP8 E4M3 | same module | E4M3 rounding of fp32 | `detail/ds4_rocm_fp8_kv.hip.hpp` |
| `kernel.gufo.qwen_attention` | canonical FP16 | same module | fp16 | `ops/attention.hpp:27-31` |
| `kernel.hipfire.kv_cache` (`kvcache.store/hipfire-kv/f32`) | F16, BF16, Q8_0, int8, HFQ4 / 8, rotated low-bit K (FWHT or Givens, 2 / 3 / 4 bits), Lloyd V; transcodes down a ladder | `kernel.hipfire.attention` | per format | `crates/rdna-compute/src/kernels.rs:8365-8424`, `kernels/src/kv_cache_write_*.hip` |

Verdict: **not a drop-in**; shape transfer: per-channel K8 KV (SHAPE-TRANSFERS R11). A KV format is fixed by the writer and the attention kernel that reads it, exactly like
a weight format. hipfire's FP8 E4M3 KV writers refuse to compile for anything but gfx1201 (`#error` in
`kernels/src/kv_cache_write_fp8_e4m3*.hip`), so on gfx1151 gufo's DS4 backend is the only FP8 KV provider. The
storage idea that travels is strixite's (per-channel K ranges after norm + RoPE); it is a candidate for the GGUF
executor's KV (SHAPE-TRANSFERS R11).

## 10. Sampling and top-k

| provider | job | numeric contract | output | source |
|---|---|---|---|---|
| `kernel.strixite.logits_topk` | top 20 candidates per row | exact: value descending, lower id first on a tie; NaN flagged | `LogitCand[20]` (8 bytes each) per row + NaN word | `kernels/logits_topk.hpp:1-43` |
| `kernel.strixite.mtp_pick` | best, runner-up, NaN flag | exact, lowest index on a tie | one small struct | `kernels/mtp_pick.hpp:1-10` |
| `kernel.gufo.qwen_sampling` (`logits.topk/-/f32.gufo`) | sorted prefix of `keep` logits; sampling; speculative accept | exact: equal logits keep ascending ids, non-finite last | `sorted_logits`, `sorted_token_ids` in a caller workspace | `ops/token.hpp:27-37,70-74`, `sample.hip:1-4` |
| `kernel.gufo.qfn_kernels` (`logits.argmax/-/f32.penalty`) | argmax, candidate tiles, penalties | lowest id on ties | device token / candidate rows | `kernels.hip.cpp` (`ArgmaxPartial`, `ArgmaxFinish`) |
| `kernel.strixfold.glue` (`logits.argmax/-/f32`) | argmax of rows | first index on ties (torch.argmax) | index per row | `zig/kernels/hip/dsv41/glue.hip` (`tf_ds_glue_argmax_rows`) |
| `kernel.hipfire.sampling` | top-k (4 per thread x 256 = 1024 candidates, host reduces to 20), argmax, top-p | fp32 | 1024 x (u32 id, f32 value) | `kernels/src/topk_logits.hip:7-36`, `argmax_batched.hip:7-16` |
| `runtime.strixfold.sampling` | keyed Gumbel sampler, float64, host | bit for bit with the Python sampler | token | `zig/src/core/lanes/sampling.zig`, `zig/src/families/dsv41/sampling.zig:1-9` |

Verdicts:
- Top-k candidates, strixite against gufo: **swappable with the adapter `kernel.flerp.logits_topk_gufo`,
  measured** (swap 2 below). Both are exact selections on fp32 logits with the same order (value descending,
  lower id first, -0 equal to +0), so the 20 candidates are the same bytes. The adapter calls
  `LaunchGPUSortLogits(logits, workspace, stream, keep = 20)` once per row with `vocab_size = n_valid` and a
  workspace that points into strixite's caller-owned buffer, packs `(sorted_logits[i], sorted_token_ids[i])` into
  `LogitCand`, and scans the row for NaN itself (gufo has no NaN output). Two gaps remain and are documented in
  the adapter: gufo maps +inf to -inf (strixite ranks it first), and on a NaN row with fewer than 20 logits above
  -inf gufo lists the NaN's id. Allow-masks and rows of 1,024 logits or fewer go back to strixite's kernel.
- hipfire's top-k returns 1024 unsorted candidates for the host to reduce: **swappable with `cands-layout` plus a
  host reduce**, which gives back the copy and scan that strixite's kernel was written to remove.
- Argmax (StrixFold, gufo, hipfire): **swappable with a trivial launch adapter**; exact, same tie rule where stated
  (hipfire's tie rule is not stated in the source header and must be checked).
- The samplers themselves (strixite's host sampler, StrixFold's keyed float64 sampler, gufo's device sampler) are
  **not swappable**: StrixFold's draft invariance depends on the key `(seed, position, token id)`.

## 11. Drafts and speculation

| provider | mechanism | invariant | source |
|---|---|---|---|
| strixite MTP (`draft.mtp-pick`, executor) | MTP head, greedy pick | drafts verified by the target | `kernels/mtp_pick.hpp` |
| `runtime.strixfold.dspark_draft` + `lanes` (`speculation.window-verify-keyed`) | DSpark drafter, Markov loop, verify windows | output-invariant: a verify row draws exactly as the serial run at its position | `zig/src/families/dsv41/draft.zig:1-11`, `zig/src/core/lanes/backend.zig` |
| `runtime.gufo.speculative` (`-lossless`), `dflash`, `mtp`, `dspark` | DFlash-2, MTP, DSpark; device accept / residual sampling | lossless speculative sampling (accepted tokens follow the target distribution) | `ops/token.hpp:88-93`, `src/core/speculative/speculative_verifier.hpp`, `src/models/deepseek_v4_flash/dspark_sampler.hpp:17-24` |
| `runtime.hipfire.runtime`, executors (`speculation.draft/*`, `speculation.window-verify-hipfire`) | DFlash, DSpark, MTP, DDTree, n-gram | lossless sampled accept; a measured AR floor decides whether speculation runs | `crates/hipfire-arch-qwen4/map.md`, `crates/hipfire-arch-deepseek4/map.md` |

Verdict: **not a drop-in** at module level; shape transfer: speculation shape for the qwen35moe executor
(SHAPE-TRANSFERS R6). "Output-invariant" (StrixFold: same tokens as the serial run, by a keyed
sampler) and "lossless" (gufo, hipfire: same distribution, different random draws) are different guarantees, and
each speculator is bound to its executor's verify path. What can be shared later is the executor ABI's
`draft` / `verify` / `keep_prefix` shape (SPEC section 6), which all four fit.

## 12. TP collective

| provider | operation | transport | checks | source |
|---|---|---|---|---|
| strixite TP (`strixite-tp-rdma-v2`, not in this catalogue) | every rank sums fp32 partials in rank order, one rounding | RDMA | bit identity across ranks | `strixite-tp-rdma-v2/docs/tp-rdma.md:25-27` |
| `runtime.strixfold.coll` (`collective.interface/-/-`) | the vtable: stream-ordered all-gather of fp32 partials in rank order, split begin / end, abort | - | identity handshake, first failure poisons | `zig/src/families/dsv41/coll.zig:1-15` |
| `runtime.strixfold.coll_tcp` (`collective.allgather-tcp/-/f32`) | all-gather | host-staged TCP (USB4 / Ethernet), eager or proxy kernels | site, layer, bytes, generation per call | `tcp_coll.zig:1-12`, `tcp_proxy.zig` |
| `runtime.strixfold.coll_rdma` (`collective.allgather-rdma/-/f32`) | all-gather | RC RDMA write-with-immediate per slot (RoCE) | 64-byte trailer per slot | `rdma_proxy.zig:1-12` |
| `runtime.strixfold.coll_nccl` (`collective.allgather-nccl/-/f32`) | all-gather | NCCL / RCCL API by dlopen | none | `comm.zig` |
| `kernel.strixfold.coll_proxy` (`collective.device-proxy`) | stage / collect on the model stream (graph-capturable) | coherent host slab | bounded waits, abort word | `zig/kernels/hip/dsv41/coll.hip:1-13` |
| `runtime.gufo.tp_verbs` (`collective.exchange-partial/-/f32.gufo`) | exchange one fp32 partial with the peer; start / finish and queued forms | RC verbs, InfiniBand or RoCE v2, write-with-immediate, event-driven completions | operation ordinal per collective | `src/models/qwen38_flash_next/kernels/rocm/communicator.hpp:23-88`, `verbs.hpp:1-43` |
| `kernel.gufo.qfn_kernels` (`collective.device-stage`) | `StagePartial`, `WaitValue`, `AddRowsBroadcast` | - | - | `kernels.hip.cpp` |
| SGLang + RCCL (not harvested) | NCCL-style collectives | RCCL over the RoCE rail | - | `catalog/sglang/ORIGIN.toml` |

Verdicts:
- The three StrixFold transports: **swappable as is**, at run time and not in a recipe; they implement one vtable
  and the model's numbers do not depend on the transport (a collective only moves bytes). The origin compiles all
  three into one binary and picks with `TF_COLL_KIND`, so each has its own interface name
  (`collective.allgather-tcp/-/f32`, `-rdma`, `-nccl`) and the executor requires all three.
- StrixFold `Coll` against gufo's `Communicator`: **swappable with the adapter `coll-communicator`.** Both move
  fp32 partials between two ranks with one RDMA write-with-immediate per exchange and leave the reduction to the
  caller. The adapter maps the vtable's `all_gather(stream, site, send, recv, bytes)`
  (`coll.zig:99-103`) onto `ExchangePartial(data, bytes, stream, error)`, which returns a device pointer to the
  peer's partial (`communicator.hpp:44-47`), then copies both blocks into `recv` in rank order. StrixFold's per-call
  site and generation check has to ride on gufo's `BeginOperation` / `EndOperation` scope and operation ordinal.
  Gating it needs the worker pair.
- RCCL / NCCL: **swappable with StrixFold's existing `coll_nccl`** in principle (same API), unverified on gfx1151.

## 13. Image input for DeepSeek-V4.1 (`strixfold_pr6`, harvested, not run)

Names for what Shoestring-dev/Strixfold pull request #6 provides and requires. Everything here is **judged** from
the PR's sources (`catalog/strixfold_pr6/.../src/`, cited as origin paths); nothing was compiled or run.
`docs/VISION-DSV41.md` has the path end to end.

| interface | provider | what crosses it | source |
|---|---|---|---|
| `vision.tower/deepseek-v4.1/bf16.cpu-torch` | `runtime.strixfold_pr6.vision_sidecar` | `Vision.prepare` (image bytes to a patch payload and a span length, on the request's thread) and `rowsBatch` (payloads to rows, on the lane thread) | `zig/src/families/dsv41/vision_cpu.zig:65-88,169-199` |
| `vision.tower/deepseek-v4.1/bf16` | `executor.strixfold.dsv41_vision` (CUDA builds only) | the same hook, the tower on the GPU | `zig/src/families/dsv41/vision.zig` |
| `vision.worker-pipe/vcpu-v1/bf16` | `runtime.strixfold_pr6.vision_sidecar` | parent to child and back, below | `vision_cpu.zig:10-11,132-166`, `vision_cpu.py:104-151`, `vision_pipe.zig` |
| `vision.image-rows/deepseek-v4.1/bf16` | `executor.strixfold_pr6.dsv41_image_rows` | the row hand-off into the prompt pass, below | `zig/src/families/dsv41/lanes.zig:674-733`, `model.zig:733-749` |
| `cache.prefix-key-image/-/-` | `runtime.strixfold_pr6.image_key` | the cache-key contract, below | `zig/src/core/lanes/image_key.zig`, `zig/src/core/lane_host.zig:78-99,333-338` |
| `serve.image-spans/-/-` | `serve.strixfold_pr6.image_prompt` | image parts to `Image` spans of the rendered prompt | `zig/src/server/prompt.zig:60,70-137` |
| `vision.attn-softmax/-/f32` | `kernel.strixfold_pr6.vision_softmax` | `normalize_scores(float *scores, int keys)`: in-place row softmax of one tile | `zig/src/families/dsv41/vision_softmax.hip` |
| `tool.vision-extract/safetensors/bf16` | `tool.strixfold_pr6.prepare_vision` | checkpoint shards to `vision.safetensors` (266 tensors) and `gate_bias_vl.safetensors` (43) | `tools/dsv41-hip/prepare_vision.py` |

**The row hand-off** (`vision.image-rows/deepseek-v4.1/bf16`).
- An image is a span of the prompt: `Image{at, tokens, bytes, rows}` (`core/lanes/stream.zig:102-107`). The
  server writes `tokens` copies of the image token id at `at` (`server/prompt.zig:133-134`); the family checks
  every id of the span is that token and replaces it by -1 (`lanes.zig:681-685`).
- Rows: BF16, shape `[tokens][5120]` (10,240 bytes a row), little-endian, in prompt order, one block for all
  images of the request. For a patch grid of `nh x nw` (14-pixel patches) the aligner's grid is
  `lh = (nh + 2) / 3`, `lw = (nw + 2) / 3` and `tokens = lh * (lw + 1) + 2`: `image_start`, then `lh` lines of
  `lw` aligner rows and one `image_newline`, then `image_end` (`vision_cpu.py:96-101`, `picture.zig:41-44`).
  Bounds: at most 9,216 patches and 1,024 rows an image (10.5 MB of rows).
- Where they enter: after `fillBegin` and before the prompt's chunks, `Model.setImages(spans, rows)` copies the
  rows to a device buffer and sets `Span{at, len, row}` (`model.zig:733-749`); the prompt pass reads those rows in
  place of the embedding rows of the span, and the routers use `gate_bias_vl` for span positions
  (`families/dsv41/prompt.zig:1216-1230` in `catalog/strixfold`). Rank 0 alone encodes; a follower gets spans and rows in wire frames of 32 KiB
  (`lanes.zig:715-733`) and needs only the bias file.
- Requires of the executor: the image token id and hidden size 5120 from the checkpoint's config, the 43
  `layers.N.ffn.gate.bias_vl` tensors loaded (`vision_cpu.zig:35`), `TF_DS_MODALITIES=text_image`.
- With a restored prefix, images that end at or before the restore point are skipped (no rows are made for them);
  an image that straddles it is an error (`lanes.zig:677-690`).

**The worker protocol** (`vision.worker-pipe/vcpu-v1/bf16`). One child process an uncached image:
`<python> -I -c <source> <weights path> <weights sha256> [<code object path> <its sha256>]`
(`vision_cpu.zig:132-138`); environment `TF_DS_VISION_TIMEOUT`, `TF_DS_VISION_THREADS`, and for the HIP form
`TF_HIP_DEVICE`, `TF_DS_VISION_RESERVE_MIB`.
- stdin: 32 bytes, six little-endian u32 (`best_h, best_w, n_vit_h, n_vit_w, n_llm_h, n_llm_w`) and 8 zero bytes;
  then `n_vit_h * n_vit_w * 588` BF16 values, the patches as `picture.zig` makes them (`[patch][channel][row][column]`,
  `(v / 255 - 0.5) / 0.5`); then end of file. The child rejects a grid that is not consistent or over the bounds
  (`vision_cpu.py:104-113`) and a payload of another length (`:136-138`).
- stdout: `"VCPU"`, u32 rows, u64 byte count (= rows x 10,240), the rows; then end of file and exit status 0. The
  parent refuses anything else, a trailing byte included (`vision_cpu.zig:150-166`).
- One deadline covers spawn, write and read; the request's cancel check is polled every 100 ms; the child is
  killed on any error path (`vision_pipe.zig:15-33`, `vision_cpu.zig:188-190`). The child sets `signal.alarm`
  before importing torch (`vision_cpu.py:155`).
- The child verifies the weights file's sha256 on every start (`vision_cpu.py:11-16`) and requires exactly 266
  BF16 tensors named `vision.*`, `aligner.*`, `image_*` (`:24-36`).

**The cache-key contract** (`cache.prefix-key-image/-/-`). The prompt cache keys kept states by token ids, and all
images share one token id. With `Vision.cache_images` (`TF_DS_VISION_CACHE_MIB > 0`) and the prompt cache on:
- the cache sees a shadow prompt: the request's ids with the first 8 ids of each image span replaced by
  `0x80000000 | word`, the eight u32 words of `sha256(rows flag, span length, payload bytes)`
  (`image_key.zig:9-28`). Bit 31 keeps them outside the vocabulary. The model never sees the shadow; a span
  shorter than 8 tokens, or spans out of order, give no key and the request runs uncached;
- a state is kept, and so restored, only at a cut that is not inside an image span (`validCut`,
  `image_key.zig:31-37`, `lane_host.zig:88`);
- a request with an image key takes no `shared_prefixes` (`lane_host.zig:336`);
- the key is over the *prepared payload* (grid and patches), so it changes with the picture preparation, not with
  the tower: two servers with different towers but one preparation would share keys. The row cache
  (`vision_cache.zig`, sha256 of the payload to rows, an LRU of at most 512 MiB) has the same property. Both are
  process-local.

Verdicts (judged): `bf16.cpu-torch` and the CUDA tower are one overlap class and the PR claims the CPU form equals
the reference; the gate in `docs/VISION-DSV41.md` is what would show it. `vision.tower-ops/bf16/*`
(`kernel.strixite.vision`) and `executor.vision/qwen3.5/gguf` (`executor.gufo.qwen_vision`) are other models'
towers: kernels to build a native tower from, not providers of these names.

**Measured since (2026-11-11, CPU only):** `vision.worker-pipe/vcpu-v1/bf16.env-setting`
(`runtime.flerp.vision_sidecar`) is the child side of the worker pipe with the torch pin as a checked setting
(`FLERP_VISION_ENV`; records in the module's `vision-env.json`). Same framing, same rows: on Python 3.12 and torch
2.11.0+cpu its rows are bit-identical to DeepSeek's `vision.py` run in BF16, on 11 images
(`gate/results/vision-dsv41-cpu312/summary.md`; that gate failed one criterion, a bound on BF16 against FP32 that
the reference's own BF16 rows break; `docs/VISION-DSV41.md`, "What ran"). The tag is there because the setting is
an addition to the origin's contract, not because the bytes differ. The parent side (`vision_cpu.zig`) is unrun.

## The five swaps, measured

Phase 4 took the five cheapest cross-engine swaps and ran each on control-node's gfx1151 (2026-11-10). "Cross-engine"
means the provider comes from another origin, is built by that origin's compile recipe (SPEC.md section 5), and is
called through the launch API of the consumer's engine by an adapter under `catalog/flerp/kernel/`. Two levels of
evidence:

- **module gate** (`flerp gate interface <name>`): every provider against a CPU reference
  (`gate/interface/reference/`), with pass criteria fixed in `gate/interface/<name>.toml` before the first run.
  Evidence: `gate/results/interface/<name>/summary.md` and `result.json`.
- **recipe gate** (`flerp gate recipe qwen38-flash-next.<variant>`): strixite's server assembled with the
  foreign kernel bound in, against the same reference strixite binary as the unmodified recipe, 18 greedy
  requests (three prompts, three repeats, MTP drafting on and off). Evidence:
  `gate/results/qwen38-flash-next.<variant>/summary.md`. The gate also reads each adapter's own launch count
  from the server log: a swap whose foreign kernel never ran fails.

No vendored file was modified for any of them: `flerp diff-origin --origin strixite` is still 269 of 269 files
byte-identical in every variant's gate. The adapter takes over the launch symbol by a compile-time rename.

Times are medians of nine repeats of back-to-back launches with one synchronize; the host's one-minute load
average at the start of each run is in the result files and quoted below (another job was compiling on
control-node for most of them: between 8 and 39).

| swap | providers | module gate | parity class, measured | recipe variant | recipe gate |
|---|---|---|---|---|---|
| 1 residual add, fp32 | strixite, gufo, hipfire | PASS (all three) | bit-identical to the reference and to each other, 9 of 9 cases | none: strixite's Qwen3.8 executor never calls it | - |
| 2 top-20 logits candidates | strixite, gufo | PASS (both) | bit-identical on the 12 required cases; gufo differs on +inf logits and on a NaN row with fewer than 20 finite logits (reported) | `topk-gufo` | PASS, token identity: all 18 requests give the reference's prompt ids, generated ids and MTP steps |
| 3 GGUF decode GEMV, Q8_0 and Q4_K | gufo, hipfire | PASS (both) | tolerance: both within 0.3% of the any-order bound; not bit-equal to each other in any of 14 cases | none: no recipe consumes a GGUF GEMV | - |
| 4 BF16 dense GEMV, router logits | strixite, StrixFold | PASS (both) | tolerance: both within 0.05% of the any-order bound; same bits at K = 1280, last bits differ at K = 2560; each kernel's bits reproduced by an emulation of its own order | `dense-strixfold` | PASS under token-agreement (module gate passed; tokens measured, not identical): 15 of 18 requests leave the reference's tokens, the earliest at generated token 5 |
| 5 MoE router top-k | strixite, gufo, hipfire | strixite PASS, gufo PASS, hipfire FAIL | gufo: ids equal the reference's in all 14 cases, weights within 6 fp32 steps; hipfire: wrong expert set on ties | `router-gufo` (gufo only) | PASS under token-agreement (module gate passed; tokens measured, not identical): 15 of 18 requests leave the reference's tokens, the earliest at generated token 5 |

All three recipe swaps in one binary (`three-engines`: gufo's top-k, StrixFold's router logits, gufo's router, in
strixite's executor): PASS under token-agreement (module gate passed; tokens measured, not identical): 12 of 18 requests leave the reference's tokens, the earliest at generated token 5. In the assembled server, `tf_ds_dense_gemv` launched 36,732 times with MTP on and 95,472 with MTP off, 0 calls handed back to strixite's kernel; `LaunchGPUSortLogits` launched 720 times with MTP on and 1,992 with MTP off, 0 calls handed back to strixite's kernel; `RouterTopK` launched 37,164 times with MTP on and 95,904 with MTP off, 0 calls handed back to strixite's kernel. First divergence per request (the three repeats agree unless two values are given): short-chat-thinking MTP on: token 154 of 164; long-chat MTP on: none of 255; raw-completion MTP on: token 44 of 256; short-chat-thinking MTP off: token 5 of 150; long-chat MTP off: none of 255; raw-completion MTP off: token 43 of 255.

Decode tokens per second: each recipe next to the reference strixite binary measured in the same gate run (median
of three repeats per prompt; `strixite` is the unmodified recipe). "Same tokens: no" marks a prompt whose text
left the reference's: from the first different token the two servers decode different text, and with MTP on the
draft acceptance changes with it, so that row is not a like-for-like speed comparison. Speed is recorded for the
variants and is not one of their pass criteria.

| recipe | MTP | prompt | reference t/s | recipe t/s | diff | same tokens | host load, 1 min (ref / recipe) |
|---|---|---|---|---|---|---|---|
| `strixite` | on | short-chat-thinking | 55.39 | 55.37 | -0.0% | yes | ? / ? |
| `strixite` | on | long-chat | 66.88 | 67.01 | +0.2% | yes | ? / ? |
| `strixite` | on | raw-completion | 47.8 | 49.46 | +3.5% | yes | ? / ? |
| `strixite` | off | short-chat-thinking | 32.07 | 33.45 | +4.3% | yes | ? / ? |
| `strixite` | off | long-chat | 32.36 | 32.8 | +1.4% | yes | ? / ? |
| `strixite` | off | raw-completion | 32.75 | 33.36 | +1.9% | yes | ? / ? |
| `topk-gufo` | on | short-chat-thinking | 55.25 | 55.2 | -0.1% | yes | 38.82 / 23.05 |
| `topk-gufo` | on | long-chat | 66.74 | 65.9 | -1.3% | yes | 38.82 / 23.05 |
| `topk-gufo` | on | raw-completion | 49.79 | 49.55 | -0.5% | yes | 38.82 / 23.05 |
| `topk-gufo` | off | short-chat-thinking | 32.13 | 31.44 | -2.2% | yes | 12.76 / 8.23 |
| `topk-gufo` | off | long-chat | 32.79 | 31.78 | -3.1% | yes | 12.76 / 8.23 |
| `topk-gufo` | off | raw-completion | 33.2 | 22.84 | -31.2% | yes | 12.76 / 8.23 |
| `dense-strixfold` | on | short-chat-thinking | 52.99 | 59.04 | +11.4% | no | 15.2 / 27.1 |
| `dense-strixfold` | on | long-chat | 65.04 | 66.94 | +2.9% | yes | 15.2 / 27.1 |
| `dense-strixfold` | on | raw-completion | 48.93 | 47.34 | -3.3% | no | 15.2 / 27.1 |
| `dense-strixfold` | off | short-chat-thinking | 33.42 | 31.79 | -4.9% | no | 17.65 / 10.57 |
| `dense-strixfold` | off | long-chat | 32.93 | 31.76 | -3.5% | no | 17.65 / 10.57 |
| `dense-strixfold` | off | raw-completion | 33.53 | 32.66 | -2.6% | no | 17.65 / 10.57 |
| `router-gufo` | on | short-chat-thinking | 51.99 | 49.22 | -5.3% | no | 10.5 / 12.18 |
| `router-gufo` | on | long-chat | 63.3 | 67.3 | +6.3% | no | 10.5 / 12.18 |
| `router-gufo` | on | raw-completion | 47.24 | 49.26 | +4.3% | no | 10.5 / 12.18 |
| `router-gufo` | off | short-chat-thinking | 32.69 | 32.26 | -1.3% | no | 15.37 / 11.26 |
| `router-gufo` | off | long-chat | 32.29 | 32.03 | -0.8% | yes | 15.37 / 11.26 |
| `router-gufo` | off | raw-completion | 32.84 | 32.96 | +0.3% | no | 15.37 / 11.26 |
| `three-engines` | on | short-chat-thinking | 56.59 | 56.3 | -0.5% | no | 10.15 / 14.06 |
| `three-engines` | on | long-chat | 67.56 | 67.61 | +0.1% | yes | 10.15 / 14.06 |
| `three-engines` | on | raw-completion | 50.75 | 43.26 | -14.8% | no | 10.15 / 14.06 |
| `three-engines` | off | short-chat-thinking | 30.5 | 32.57 | +6.8% | no | 16.16 / 10.42 |
| `three-engines` | off | long-chat | 30.42 | 32.12 | +5.6% | yes | 16.16 / 10.42 |
| `three-engines` | off | raw-completion | 30.94 | 32.45 | +4.9% | no | 16.16 / 10.42 |

Two rows need a word. `topk-gufo`, MTP off, raw-completion: the variant's three repeats were 12.9, 22.8 and
33.2 t/s against the reference's 32.8 to 33.3, while the host's load average rose from 8 to 30 under that server;
the median is the other job's doing and the third repeat is the comparable one. The other `topk-gufo` rows show
what the module gate predicts for five launches a row in place of two: 0 to 3% slower. The unmodified recipe's
rows are from its run of 22:44Z, before the gate recorded the host load.

What the token measurements of swaps 4 and 5 mean. Both change the router's numbers in their last bits only (the
module gates bound it), and both move the greedy text: 15 of 18 requests, first at generated token 5 of the
short chat prompt. The same evidence shows how little it takes: the reference strixite binary against itself,
MTP drafting on against off (the same kernels at other batch shapes), already differs at token 95 of that prompt
and token 43 of the raw completion, and agrees on the long chat. The positions where the variants leave the
reference (5, 11, 43 to 53, 95, 154) are where this model's greedy choice is that close. So "token identity" is
a property this model gives only to bit-exact swaps, which is why swap 2 has it and swaps 4 and 5 are reported
as agreement measurements. Whether the diverged text is as good is not something these gates measure.

The unmodified recipe after the assembler changes of phase 4: `qwen38-flash-next.strixite` was assembled and
built again (stages check, origin, build, unit, 2026-11-11T00:13Z and 00:28Z) and its 13 binaries have the sha256 of the
ones that passed the GPU stage on 2026-11-10T22:44Z, so that GPU result stands for the current tree
(`gate/results/qwen38-flash-next.strixite/summary.json`; its GPU stage was not run again).

### 1. Residual add, fp32: measured, bit-identical; no recipe consumer

`kernel.strixite.residual` against `kernel.gufo.qwen_norm_rope` (`LaunchResidualAdd`, through
`kernel.flerp.residual_add_gufo`) and `kernel.hipfire.tensor_ops` (`add_inplace_f32`, a code object, through
`kernel.flerp.residual_add_hipfire`). Evidence: `gate/results/interface/residual-add/`.

- Parity: all three give the CPU reference's bits on 9 of 9 cases (sizes around the 256-thread block, one decode
  row, 1,000,003 elements, a 20,971,520-element prefill block; values include cancellations, signed zeros,
  subnormals, overflow, inf - inf and NaN; `y == x`). Pairwise, the three outputs are byte-identical.
- Time per launch: 10,240 elements (one token's four streams) 2.01 us strixite, 2.19 us gufo, 1.98 us hipfire;
  20,971,520 elements 1,207 us, 1,208 us, 1,247 us (208, 208 and 202 GB/s over the three operand streams). Host
  load 22.6 for strixite and gufo, 11.1 for hipfire (its provider was run again through `flerp gate module`).
- What the adapters had to cover: gufo writes a third buffer, so the in-place call passes `out == a`, which is
  outside what gufo's `__restrict__` qualifiers promise (harmless for one load and one store per element, and
  measured so); hipfire has no host launcher and an `int` element count; neither has a BF16 form.
- **No recipe variant, and why.** The judged list expected this swap to prove the plumbing inside a strixite
  recipe "for an f32-activation run". There is no such call to rebind: `runtime/qwen4exp.cpp` includes
  `kernels/residual.hpp` (line 22) and never calls `residual_add`; its residual updates are fused into
  `hc_inject` and `moe_shared_add_inject`. The two callers are the vision tower (`runtime/vision_tower.cpp:296,301`,
  activation dtype BF16 in the server) and the Qwen3.5 baseline forward, which no binary links. A variant bound
  to either adapter would build and would launch the foreign kernel zero times, which the recipe gate counts as
  a swap that did not run. The plumbing proof moved to swap 2.

### 2. Top-20 logits candidates: measured, exact; recipe variant passes with token identity

`kernel.strixite.logits_topk` against `kernel.gufo.qwen_sampling` (`LaunchGPUSortLogits`, keep = 20) through
`kernel.flerp.logits_topk_gufo`. Evidence: `gate/results/interface/logits-topk/`,
`gate/results/qwen38-flash-next.topk-gufo/`.

- Parity: both providers give the reference's bytes (20 `LogitCand` per row and the NaN words) on the 12
  required cases: 1, 2 and 9 rows of 248,077 logits in a 248,320-wide row, rows full of ties, signed zeros,
  padding that must not win, rows that are mostly -inf, a 4,097-logit vocabulary, the allow-mask and the
  1,000-logit row (those two reach strixite's kernel through the adapter by design: 2 launches of 5,964).
- Reported, not required: with a +inf logit gufo returns the finite maximum where strixite returns +inf; on a
  NaN row with fewer than 20 logits above -inf gufo lists the NaN's id as a -inf entry (the NaN flag itself is
  exact in every case).
- Time per call at 248,077 logits: 1 row 79.9 us strixite, 472 us gufo; 4 rows 133 us, 390 us; 16 rows 375 us,
  1,532 us. strixite needs 2 launches per call, the adapter 5 per row plus 1. (In this run gufo's one-row call
  cost more than its four-row call; not explained. Host load 22 to 23.)
- Recipe: PASS, token identity: all 18 requests give the reference's prompt ids, generated ids and MTP steps. In the assembled server, `LaunchGPUSortLogits` launched 666 times with MTP on and 1,989 with MTP off, 0 calls handed back to strixite's kernel.

### 3. GGUF decode GEMV, Q8_0 and Q4_K: measured at module level, tolerance parity

`kernel.gufo.qwen_linear_decode` (`LaunchQ8KBlockGEMV`) against `kernel.hipfire.gemv_ggml` (`gemv_q8_0`,
`gemv_q4k`, code objects) through `kernel.flerp.gguf_gemv_hipfire`. Evidence:
`gate/results/interface/gguf-gemv/`.

- Parity: 14 of 14 cases within the any-order bound for both (worst 0.27% of it for gufo, 0.18% for hipfire):
  1, 2, 2,559, 2,560 and 10,240 rows at K = 2,560, K = 6,144 with x over ten decades, K = 256. The two differ
  from each other in every case: 86% of all outputs, by at most 7.6e-6 in the cases with ordinary inputs and by
  at most 0.19 in the two cases whose x spans ten decades.
- Is the summation order the only difference? For hipfire, yes, shown: a CPU emulation of its source's order
  (per lane `(d * q) * x` folded with one fma, lanes reduced by shuffles) reproduces its bits in 14 of 14 cases.
  For gufo, shown in 12 of 14: its two-row Q8_0 kernel (block sums, both steps fused; 5 of 5 even-row cases)
  and its Q4_K sub-16 route (`scale * dot - offset * xsum` with the scale product fused; 7 of 7). Not shown for
  its warp-per-row Q8_0 kernel, which an odd row count selects: the 2,559-row case matches no emulation, and
  the one-row case has a single output, whose match with an emulation of the other kernel is not evidence.
- Time per launch: Q8_0 2,560 x 2,560 (7.0 MB) 13.5 us gufo, 16.8 us hipfire; 16,384 x 2,560 (44.6 MB) 197 us
  and 206 us (227 and 217 GB/s); Q4_K 2,560 x 2,560 (3.7 MB) 14.2 us and 17.0 us; 16,384 x 2,560 (23.6 MB)
  78 us and 90 us. Host load 9.2. The large Q4_K shape works out to 303 GB/s of weight bytes for gufo and 263 GB/s for hipfire, above the 226 GB/s read peak strixite's document reports for this GPU; that was not investigated, so read those two as times per launch and not as bandwidth figures.
- No recipe: nothing in this repository consumes a GGUF GEMV yet; the token-level check waits for a gufo
  recipe. The adapter takes over `LaunchQ8KBlockGEMV`; gufo's own `LaunchGEMV` in the same module keeps its
  kernel (the rename covers the module's own translation units).

### 4. BF16 dense GEMV for the router logits: measured, order-only difference; recipe variant measured

`kernel.strixite.linear` (`linear_bf16w`, BF16 in, FP32 out) against `kernel.strixfold.dense_bf16`
(`tf_ds_dense_gemv`, a code object built with StrixFold's flags and TheRock 10.2) through
`kernel.flerp.dense_bf16_strixfold`. Evidence: `gate/results/interface/dense-bf16/`,
`gate/results/qwen38-flash-next.dense-strixfold/`.

- Parity: 12 of 12 cases within the any-order bound for both (worst 0.05% of it). At the router's shape
  (513 x 2,560, 1 to 128 rows) about 70% of the outputs differ between the two kernels, by at most 7.2e-7; with
  inputs over twelve decades the largest gap is 0.078 on outputs up to 10^4.
- The summation order is the only difference, shown: strixite's bits equal a CPU emulation of its rows path
  (KS = 2 waves at K = 2,560) on all 11 aligned cases, StrixFold's equal an emulation of its lane order on all 10
  cases it ran, and at K = 1,280, where strixite does not split, the two kernels are bit-identical. (The
  unaligned-w case takes strixite's strided kernel in both providers; its order was not emulated.)
- Corrections to the judged version: strixite's kernel is not "the untuned baseline" (see section 2), and at one
  row it is the faster one: 5.6 us against 7.1 us; 16 rows 27.1 us against 25.7 us; 128 rows 175 us against
  204 us. Host load 15.6. No speed-up from this swap at decode.
- Recipe: PASS under token-agreement (module gate passed; tokens measured, not identical): 15 of 18 requests leave the reference's tokens, the earliest at generated token 5. In the assembled server, `tf_ds_dense_gemv` launched 34,587 times with MTP on and 97,056 with MTP off, 0 calls handed back to strixite's kernel. First divergence per request (the three repeats agree unless two values are given): short-chat-thinking MTP on: token 5 of 164; long-chat MTP on: none of 255; raw-completion MTP on: token 46 of 256; short-chat-thinking MTP off: token 95 of 150; long-chat MTP off: token 11 of 255; raw-completion MTP off: token 43 of 255.

### 5. MoE router top-k: measured; gufo swappable, hipfire not

`kernel.strixite.moe_router` against `kernel.gufo.qfn_kernels` (`RouterTopK`) through
`kernel.flerp.moe_router_gufo`, and `kernel.hipfire.moe` (`moe_softmax_topk_renorm_k8`) through
`kernel.flerp.moe_router_hipfire`. Evidence: `gate/results/interface/moe-router/`,
`gate/results/qwen38-flash-next.router-gufo/`. The checkpoint routes 10 of 512 experts with the shared expert
kept separate, so the router runs without a shared slot.

- gufo: PASS. Selected ids equal the reference's, slot for slot, in all 14 required cases (k = 10 and k = 8;
  1, 16 and 2,048 tokens; rows full of exact ties; 16 equal logits across the k-th place; a NaN and a +inf token,
  which get strixite's ids -1, NaN weights and error word through the adapter's fix-up launch) and also in the
  two reported cases with logits one ulp apart. Weights within 5.3 fp32 steps of the exact value (strixite's own:
  3.5); the two are not bit-equal in any case.
- hipfire: FAIL, at k = 8, the only k its kernel has. On rows of exact ties it returns a different order in 63 of
  64 rows and a different expert set in 43; with 16 equal top logits it picks 256, 260, 100, 356, 44, 300, 2, 258
  where the contract gives 2, 5, 31, 44, 70, 100, 129, 256. Cause in the source: its arg-max reduction keeps the
  lower thread on a tie, and thread t holds experts t and t + 256 (`moe_softmax_topk_k8.hip`). It passes the
  cases without ties. It is not bound by any recipe: k = 8 against the checkpoint's 10 already rules it out
  (section 4).
- Time per call, k = 10: 1 token 28.1 us strixite, 36.6 us gufo (2 launches); 16 tokens 28.6 us, 37.1 us;
  2,048 tokens 100 us, 344 us. hipfire at k = 8: 11.8 us for 1 token, 135 us for 16, 17.1 ms for 2,048 (one
  launch per token) against strixite's 25.5 us, 26.0 us, 90.6 us. Host load 11.9.
- Recipe: PASS under token-agreement (module gate passed; tokens measured, not identical): 15 of 18 requests leave the reference's tokens, the earliest at generated token 5. In the assembled server, `RouterTopK` launched 31,227 times with MTP on and 95,760 with MTP off, 0 calls handed back to strixite's kernel. First divergence per request (the three repeats agree unless two values are given): short-chat-thinking MTP on: token 5 of 164; long-chat MTP on: token 11 of 255; raw-completion MTP on: token 53 of 256; short-chat-thinking MTP off: token 5 of 150; long-chat MTP off: none of 255; raw-completion MTP off: token 43 of 255.

Next after these, and the first that needs two nodes: StrixFold's `Coll` on gufo's verbs `Communicator`. Judged the
numerically safest swap in the catalogue (bytes only); gating it needs the worker pair and a phase 3 executor.

## Gaps this pass leaves open

- Every verdict outside "The five swaps, measured" is still judged from sources; for those, `gate.status` is at most
  `builds`.
- The recipe gate's 18 requests are greedy and unmasked. They do not exercise the allow-mask path of the top-k
  swap (which the adapter hands back to strixite's kernel), sampling at temperature above zero, or vision.
- Token agreement of the two order-changing swaps is a measurement over three prompts of up to 256 generated
  tokens, not a bound: a longer or different prompt can diverge earlier or later.
- The speed numbers were taken while another job compiled on control-node; the result files carry the load
  average and the per-repeat values.
- The summation order of gufo's warp-per-row Q8_0 GEMV kernel and of the unaligned path of strixite's BF16
  linear is not reproduced by an emulation; both are inside the bound.
- gufo's kernel sources carry few header comments; rounding points inside its fused kernels were not traced line
  by line beyond the kernels gated here.
- hipfire's per-variant numerics were read for the representative kernels cited above, not for all 1,046 units.
  Its argmax tie rule is still unread; its router's is now measured (section 4).
- The strixite rows describe `catalog/strixite` as harvested; its TP path (`strixite-tp-rdma-v2`) is not in the
  catalogue yet, so job 12 has no strixite module.
