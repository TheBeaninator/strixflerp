# Phase 5 gap map: GLM-5.2 (colibri int4) on four nodes

Written 2026-11-10 from a read-only pass. Nothing was built, loaded on a GPU or run on a worker.
Phase 5 is SPEC.md section 9 item 5: a recipe for a model no origin serves on this pod, assembled from catalogue
modules, with the missing modules written under origin `flerp`.

Evidence marks: **[R]** read in code, a file header or a report (path and line given); **[I]** inferred from what
was read; **[N]** not checked.

Paths: `catalog/...`, `gate/...`, `docs/...` are in this tree. `upstream/...` is `~/Code/strixflerp-upstream/...`.
`pod:` is control-node. `colibri:` is `pod:/data/ark/src/github.com__JustVugg__colibri` at commit `ce370e8`
(2026-09-25), line numbers from that commit. `model:` is
`pod:/data/ark/models/jlnsrk__GLM-5.2-colibri-int4/colibri`.

## 0. Findings that change the plan

1. **The container is not an SGLang or HF checkpoint.** It is the private format of colibri, a pure-C CPU engine
   (Apache-2.0) whose source is on the pod. `model:README.md` says so and names the converter and its flags
   (`convert_fp8_to_int4.py --ebits 4 --io-bits 8`). **[R]**
2. **The pod never served this model.** The four-node SGLang deployment was GLM-5.3-Flash FP8, a different
   architecture (section 1.4). No engine on the pod has produced a GLM-5.2 token. **[R]**
3. **The DSA indexer weights are not in the container.** The converter skips them; colibri then runs dense causal
   attention, which equals the model's sparse attention only up to `index_topk` = 2,048 tokens of context
   (section 1.3). **[R]**
4. **`config.json` describes the source checkpoint, not the bytes.** It still carries the FP8 `quantization_config`
   and `dtype: bfloat16`; the file headers hold `U8` and `F32` only. **[R]**
5. **Most kernels exist in the catalogue, in hipfire, at a neighbouring format**, including latent (MLA) attention
   and the bias-aware router. None of them has passed a hardware parity gate here; all are compile-gated only
   (section 2). **[R]**
6. **The only token-level reference is colibri's own CPU path**, which has not been built on the pod (section 1.5).
7. **StrixFold's four-rank split exists in the harvested source but is switched off on HIP**
   (`world4 = false`), and it is cut to DeepSeek-V4.1's shapes (section 3). **[R]**

## 1. The model

### 1.1 Files and tensor census [R]

Header-only read of all 144 shards (`out-00000..00140` plus `out-mtp-00000..00002`): 118,478 tensors,
378,893,907,712 bytes (352.9 GiB). No `__metadata__` block in any shard. Dtypes: `U8` 377.59 GB, `F32` 1.30 GB.
Other files: `config.json`, `generation_config.json` (temperature 1.0, top_p 0.95), `tokenizer.json`,
`tokenizer_config.json`, `README.md`. No `chat_template.jinja`.

| tensor (per layer unless noted) | logical shape O x I | stored | bits | layers |
|---|---|---|---|---|
| `model.embed_tokens.weight` | 154880 x 6144 | U8 951,582,720 + `.qs` F32 [154880] | int8-row | once |
| `lm_head.weight` | 154880 x 6144 | same | int8-row | once |
| `self_attn.q_a_proj` | 2048 x 6144 | U8 6,291,456 + `.qs` [2048] | int4-row | 0..77 |
| `self_attn.q_b_proj` | 16384 x 2048 (64 heads x 256) | U8 16,777,216 + `.qs` [16384] | int4-row | 0..77 |
| `self_attn.kv_a_proj_with_mqa` | 576 x 6144 (512 latent + 64 rope) | U8 1,769,472 + `.qs` [576] | int4-row | 0..77 |
| `self_attn.kv_b_proj` | 28672 x 512 (64 x (192 + 256)) | U8 7,340,032 + `.qs` [28672] | int4-row | 0..77 |
| `self_attn.o_proj` | 6144 x 16384 | U8 50,331,648 + `.qs` [6144] | int4-row | 0..77 |
| `q_a_layernorm`, `kv_a_layernorm`, `input_layernorm`, `post_attention_layernorm` | [2048], [512], [6144], [6144] | F32 | - | 0..78 |
| `mlp.{gate,up}_proj`, `mlp.down_proj` (dense MLP) | 12288 x 6144, 6144 x 12288 | U8 37,748,736 each | int4-row | 0..2 |
| `mlp.gate.weight`, `mlp.gate.e_score_correction_bias` (router) | 256 x 6144, [256] | F32 | - | 3..78 |
| `mlp.shared_experts.{gate,up,down}_proj` | 2048 x 6144, 6144 x 2048 | U8 6,291,456 each | int4-row | 3..77 |
| `mlp.experts.E.{gate,up,down}_proj`, E = 0..255 | 2048 x 6144, 6144 x 2048 | U8 6,291,456 each | int4-row | 3..78 |
| layer 78 (MTP): attention, shared expert, `eh_proj` 6144 x 12288 | as above | U8 at one byte a weight | int8-row | 78 |
| layer 78: `enorm`, `hnorm`, `shared_head.norm` | [6144] | F32 | - | 78 |
| `model.norm.weight` | [6144] | F32 | - | once |

Byte totals: routed experts 342.0 GiB (97 %); attention 6.0 GiB; shared experts 1.3 GiB; embedding and head
1.77 GiB; routers 0.45 GiB; MTP layer 4.78 GiB (its 256 experts are int4, the rest int8). Per layer: 0.18 GiB
(dense layers 0..2), 4.61 GiB (sparse layers 3..77). Routed experts counted from the headers: 76 x 256 = 19,456
(the README says 21,504; the headers are what is on disk).

### 1.2 The encoding [R]

Defined by `colibri:c/tools/convert_fp8_to_int4.py` and `colibri:docs/FORMATS.md:60-66`.

- **int4-row** (colibri `fmt=2`), converter lines 140-153: per output row, `s = max(max|w| / 7, 1e-8)`,
  `q = clip(rint(w / s), -8, 7)`, stored as the nibble `q + 8`. Two nibbles a byte: **low nibble = even column,
  high nibble = odd column**. `w = (nibble - 8) * s[row]`. One F32 scale per row in `<name>.qs`. No groups, no
  separate zero point (the offset is the constant 8).
- **int8-row** (`fmt=1`), lines 133-138: `s = max|w| / 127`, signed int8 bytes, `w = q * s[row]`.
- **F32 kept**: norms, router weight and bias (converter `classify`, lines 257-294).
- **Skipped in the main pass**: everything named `indexer` or `indexers_proj`, and the MTP layer; the MTP layer
  comes from a second pass (`--mtp`, int8 by default because an int4 MTP head gives near-zero acceptance, lines
  575-585). An `--indexer` pass would write `out-idx-*`; this directory has none.
- The engine identifies a tensor's format from its byte count and its `.qs` length (FORMATS.md:55-58), so a loader
  written here must apply the same rule or carry its own table.
- The source was `zai-org/GLM-5.2-FP8` (e4m3, 128 x 128 block scales), dequantized to f32 and re-quantized. The
  FP8 checkpoint is not on the pod. **[R]** (`find` over the model directories)

### 1.3 Forward pass [R unless marked]

From `colibri:c/colibri.c`, whose header (lines 1-17) calls it a faithful copy of transformers'
`modeling_glm_moe_dsa.py`. The transformers file itself was not read. **[N]**

```
x = embed[token]                                         (int8-row row lookup, :2671)
for layer in 0..77:                                      (layer_forward_rows, :7217-7321)
    n = rmsnorm(x, input_layernorm)                      (x * rsqrt(mean(x^2) + 1e-5) * w, :1564)
    x += attention(n)
    n = rmsnorm(x, post_attention_layernorm)
    x += dense_mlp(n) if layer < 3 else moe(n)
logits = lm_head @ rmsnorm(x, model.norm)                (:7541, :7593)
```

Attention (MLA, `attention_rows` :4391; the scoring core at :4530-4590 was skimmed, not read line by line):

- `qr = rmsnorm(q_a x, q_a_layernorm)`; `q = q_b qr` gives 64 heads x 256 = [192 nope | 64 rope].
- `c = kv_a x` gives [512 | 64]; the 512 are RMS-normed with `kv_a_layernorm`, the 64 are the rope key shared by
  all heads. **The KV state is these 576 numbers a token a layer** (colibri keeps them F32 by default).
- `kv_b` expands the 512 latent to per-head [192 k_nope | 256 v]; colibri also has "absorbed" forms that fold
  `kv_b` into the query and the output instead.
- RoPE on the 64 rope dims of q and k: `theta = 8,000,000` read from `rope_parameters.rope_theta` (:1729-1730),
  angle `pos * theta^(-2j/64)`, pairs (2j, 2j+1) rotated and written de-interleaved to [j] and [32 + j]
  (:1651-1683).
- Scores: `q . [k_nope | k_rope] / sqrt(256)` (:1779), causal softmax, values 256 a head, then `o_proj` over
  64 x 256 = 16,384.
- **DSA**: `has_dsa` is set only if `self_attn.indexer.wq_b.weight` exists (:2604-2614). It does not, so every
  position attends to every earlier key. The file header (:7-8) states this is the model's function only while the
  sequence is at most `index_topk` = 2,048.

MoE (`moe` :5414): `score = sigmoid(gate.weight @ n)`; selection on `score + e_score_correction_bias` (:5507),
top 8 of 256 by repeated argmax (:5380-5388); weights are the unbiased scores of the picks, divided by their sum
(`norm_topk_prob`, :5635) and multiplied by `routed_scaling_factor` 2.5 (:5636). Each expert is
`down(silu(gate n) * up n)` (:1596-1648). One shared expert of the same shape is added unweighted (:6566).
Dense MLP in layers 0..2 is the same SwiGLU at width 12,288 (:6643).

MTP (`mtp_draft` :7668-7706): `eh_proj([rmsnorm(embed(tok), enorm) | rmsnorm(h, hnorm)])`, one more layer
(layer 78, its own KV row), `shared_head.norm`, then the main `lm_head`. Not needed for first tokens.

Text: `tokenizer.json` is a `tokenizers` BPE (154,820 vocabulary, 321,649 merges, 36 added tokens, regex `Split` +
`ByteLevel` pre-tokenizer, no normalizer). Its sha256 equals GLM-5.3-Flash's `tokenizer.json`
(`19e77364...`). **[R]** End ids 154820 `<|endoftext|>`, 154827 `<|user|>`, 154829 `<|observation|>`. The raw
turn colibri sends is `[gMASK]<sop><|user|>{prompt}<|assistant|>` (`colibri:c/coli:1444`). The official GLM-5.2
`chat_template.jinja` (sha256 `172dc74a...`, named in `colibri:c/tests/test_glm52_chat_template.py`) is not in
the model directory; whether GLM-5.3-Flash's template on the pod is the same file was not checked. **[N]**

### 1.4 What the pod served on four nodes before [R]

- **GLM-5.3-Flash FP8** (`zai-org/GLM-5.3-Flash`, `model_type glm5_next`): 45 layers + MTP layer 45, hidden 4096,
  288 experts, 34 KDA (linear attention) and 11 DSA layers, four hyper-connected streams; 76,108 tensors,
  328,326,771,576 bytes at `pod:/data/models/glm-5.3-flash-nvme`
  (`pod:/data/pod/reports/2026-10/pod/glm53-flash-performance-2026-10-29.md:47-64`). Launcher
  `pod:/data/models/tools/sglang-rocm/glm-tp4.sh` (line 1 names the model; later revisions load a striped view
  under `/cache/models/glm-5.3-flash-nvme-view`). 9.5 decode tok/s single stream (same report, section 1.1).
  `catalog/sglang/ORIGIN.toml:1` records the same.
- That is not this directory and not this architecture: GLM-5.2 is `glm_moe_dsa` with 78 uniform MLA layers, no
  KDA, no hyper-connections.
- SGLang 0.5.20 on the pod does carry a `GlmMoeDsaForCausalLM` class
  (`pod:/data/projects/sglang-0.5.20/python/sglang/srt/models/glm4_moe.py:1445`). It loads HF checkpoints, not
  this container, and the FP8 checkpoint (756 GB per `model:README.md`) is neither on the pod nor small enough
  for 4 x 110 GiB. So SGLang cannot be the reference. **[R for the facts, I for the conclusion]**
- Elsewhere on the pod: a GLM-5.3 `UD-Q4_K_XL` GGUF, 436 GiB in 11 parts (`pod:/mnt/nv1/models/glm-5.3`), and
  `Justvugg__GLM-5.3-colibri-int4-g64` (grouped int4, a different model). Neither was opened. **[N]**
- Reports that mention this container only list it in inventories
  (`pod:/data/pod/reports/2026-10/review-diversity-2026-10-26.md:25`: "colibri runtime, served: no").

### 1.5 A token-level reference for the gate

| candidate | state | verdict |
|---|---|---|
| colibri CPU engine on this container, `TF=1` teacher forcing (`colibri.c:15-17`) plus greedy continuation | source on the pod; **no binary built** (`c/glm` absent) **[R]**; gcc 13.3 and 32 cores present **[R]** | the reference. It defines the container's meaning (format inference, rounding, dense attention). One teacher-forced pass over a fixed token sequence gives an argmax for every position. |
| colibri `ROUTE_TRACE` (router picks per position and layer, `:5637`) | same binary | a per-layer check: equal expert picks on all 76 routers localize a fault to a layer. Recommended as a stage of the gate. |
| colibri tiny oracle `c/ref_glm.json` (12 prompt ids, 32 teacher-forced predictions against transformers) | file present **[R]**; the tiny model directory it needs was not looked for **[N]** | proves the colibri build itself before trusting it. |
| SGLang `GlmMoeDsaForCausalLM` | needs the FP8 checkpoint | not available. |
| a numpy forward written from section 1.3 | not written | optional second opinion; slow; would be new code with the same reading of the format. |

Expectation to set before the first run **[I]**: CPU f32 summation order differs from any GPU kernel, so the
criterion is `token-agreement` (SPEC section 7), not identity. colibri's own README reports 30 to 32 of 32
teacher-forced positions against transformers with two near-ties (`colibri:README.md:315-317`).

Cost of the reference **[I]**: the container sits on an SMR hard disk (`/dev/sdc1`, ST8000DM004, mounted
`/mnt/ark`) **[R]**. A decode step reads about 11 GiB of routed experts, so free-running generation from that disk
is minutes a token when cold. A teacher-forced pass reads each layer's union of picked experts once. control-node
had 1 GiB of RAM available when looked at (124 GiB total, 122 used) **[R]**, and colibri wants 16 to 25 GiB, so
the reference run needs either a quiet control-node or a worker's CPU with the container on its `/cache` NVMe.

## 2. Operation by operation

Status column: the module's `[gate] status` in its manifest. For every StrixFold, gufo and hipfire kernel module
that is `builds`: it compiles for gfx1151 with its origin's command (`gate/results/compile/`), nothing more. The
only kernels with a hardware parity result here are the five interface gates (README.md:136-142): residual add,
logits top-20, GGUF GEMV (Q8_0, Q4_K), BF16 dense GEMV, MoE router top-k.

Fit: **same** = same math and same weight format; **math** = same math, different weight or state format;
**none** = nothing in the catalogue.

| operation | catalogue module(s) | engine, format, dtype | status | fit | what remains |
|---|---|---|---|---|---|
| read the container | `format.strixite.safetensors`, `format.strixite.safetensors_loader`, `format.strixfold.safetensors` | generic safetensors readers | builds; strixite's are exercised by the passed strixite recipe | same (container level) | a `format.flerp.colibri_rows` module: name table, format inference, device views, a layer range a node |
| int4-row GEMV, 1 to 16 rows (q_a, q_b, kv_a, o, dense MLP, shared expert) | `kernel.hipfire.gemv_hfq` (`gemv_hfq4g256.hip`) | hipfire HFQ4-G256: 136 bytes a group of 256 = f32 scale, f32 zero, 128 code bytes, low nibble first (`:61,:109-116`); f32 activations | builds | math | transcode at load: `scale = s[row]`, `zero = -8 s[row]` for every group of the row. Same nibble order. +6.25 % bytes. Last-bit differences from colibri. Or a new native kernel (section 5, item 8) |
| same | `kernel.strixite.linear_q4` | strixw Q4: BF16 scale and min a group of 32/64/128 (`catalog/strixite/format/q4/src/formats/q4.hpp:3-6`) | builds; passed inside the strixite recipe at its own shapes | math, **lossy** | not recommended: an F32 row scale does not survive BF16 |
| int4 routed experts, 8 of 256 by index | `kernel.hipfire.moe` (`gemv_hfq4g256_moe_gate_up_k8_indexed_batched`, `..._moe_down_indexed*`) | HFQ4-G256, expert pointer table + top-k indices, f32 | builds | math | same transcode; widths 2048 / 6144 are multiples of 256 |
| same | `kernel.strixite.linear_q4` expert gather, `kernel.strixfold.exl3`, `kernel.gufo.ds4_moe`, `kernel.gufo.qfn_mmq` | strixw Q4/Q5; EXL3 trellis; GGUF IQ2/Q2_K; GGUF K-quants | builds / builds / harvested / builds | math, formats that cannot hold these bytes without re-quantizing | not usable without the FP8 source |
| int8-row GEMV (lm_head 154880 x 6144; MTP layer) | `kernel.hipfire.gemv_hfq` (`gemv_hfq8g256.hip`: 264 bytes a group, f32 scale and zero, u8 codes, `:19-28`) | HFQ8-G256, f32 | builds | math | transcode `code = q + 128`, `zero = -128 s` |
| embedding lookup, int8-row | `kernel.hipfire.embedding`, `kernel.strixite.embedding` | HFQ4, GGUF Q8, Q4_K, BF16, strixw Q8 tables | builds | none at this format | one row of 6,144 values: dequantize on the host for first tokens, or a 20-line kernel |
| RMSNorm (weight, eps 1e-5, no bias) | `kernel.hipfire.norm_rope` (`rmsnorm.hip`), `kernel.gufo.qwen_norm_rope`, `kernel.strixite.norm`, `kernel.strixfold.aux` | f32 / f32 / bf16+f32 / torch-order bf16 | builds | same for the f32 ones **[I: formula not diffed line by line]** | interface gate |
| RoPE on a 64-dim tail, interleaved pairs, theta 8e6 | `kernel.hipfire.norm_rope` (`rope_tail_interleaved*.hip`, `rope_partial_interleaved.hip`) | f32, in place | builds | math up to layout | hipfire rotates pairs in place (`rope_partial_interleaved.hip:20-37`); colibri de-interleaves. Scores are equal if q and k use the same kernel. The `tail` variants' arguments were not read **[N]** |
| MLA attention over a latent KV | `kernel.hipfire.attention` (`latent_attention_{absorb,decode,decompress,flash,split,serial,gate}.hip`, `module.toml:80-86`) | BF16 cache rows [latent + tail] = [512 + 64], f32 queries, geometry by argument (heads, nope, tail, v, latent); limits `wa + wb <= 2048`, multiples of 8, `dv <= 2048` (`latent_attention_decode.hip:8-45`) | builds | math, **KV format BF16 where colibri keeps F32** | GLM-5.2's 64 / 192 / 64 / 256 / 512 fits the limits. No executor in the catalogue calls these kernels; upstream only hipfire's un-harvested Raon family does (`catalog/hipfire/HARVEST.json:12`, `upstream/hipfire/crates/hipfire-arch-raon/src/mla.rs:5-17`). `kv_b` must be handed over as BF16 or F32 (dequantize 28672 x 512 once a layer) |
| MLA attention | `kernel.strixfold.sparse_attn`, `kernel.gufo.ds4_attention`, `kernel.hipfire.ds4_sparse_attn`, `kernel.strixite.attention` | DeepSeek-V4 sink + window attention; GQA softmax | builds / harvested / builds / builds | none (different attention) | - |
| DSA indexer and top-2048 selection | `kernel.strixfold.indexer_score`, `kernel.strixfold.aux`, `kernel.hipfire.indexer_topk`, `kernel.gufo.ds4_indexer` | 32 heads x 128 scoring, top-k | builds / builds / builds / harvested | math, nearby | **blocked on weights that are not on the pod**, not on code |
| router logits (256 x 6144 F32) | `kernel.hipfire.gemv_float`, `kernel.strixite.linear` (BF16 weights), `kernel.strixfold.moe_gate` (fp16) | f32 GEMV | builds | same for hipfire's f32 GEMV | - |
| router: sigmoid, bias-aware top-8, renormalize, x 2.5 | `kernel.hipfire.moe` (`deepseek4_moe_topk_bias_aware.hip:11-27`: select on `scores + bias`, weights from unbiased scores, divided by their sum, times a route scale; 256 experts, k <= 32) + `kernel.hipfire.tensor_ops` (`sigmoid.hip`) | f32 | builds | same **[I: tie rule not compared; colibri takes the lowest index]** | interface gate with tie cases. Note hipfire's other router (`moe_softmax_topk_renorm_k8`) **fails** the tie cases of the existing `moe-router` gate (README.md:142) |
| router | `kernel.strixite.moe_router`, gufo `RouterTopK` | softmax-renormalized | builds; both pass `moe-router` | none (softmax, no bias) | - |
| SiLU x up | `kernel.hipfire.tensor_ops` (`silu_mul.hip`), `kernel.strixite.swiglu` | f32; bf16/f32 merged layout | builds | same | - |
| weighted expert sum, residual add | `kernel.hipfire.moe` (combine), residual add from strixite / gufo / hipfire | f32 | builds; residual add **passed** bit-identical for all three (README.md:138) | same | - |
| final head and logits | int8-row GEMV above; `kernel.strixite.logits_topk` (benched), `kernel.strixfold.glue` argmax, `kernel.hipfire.sampling` | f32 rows | benched / builds / builds | same | hand full rows to the host for first tokens |
| sampling | `runtime.flerp.sampler_strixite` behind `sampler.h` | host, full rows | parity | same | none |
| decode loop, HTTP front end | `runtime.flerp.decode_loop`, `serve.flerp.http` | executor ABI | parity (over strixite) | same | needs an executor behind `executor.h` |
| KV state, snapshots, prompt cache | `runtime.strixite.prompt_cache`, `runtime.strixfold.prompt_cache`, `kernel.hipfire.kv_cache` | each tied to its executor's state layout | builds | none for a 576-wide latent row | first tokens need only an append-only buffer and reset |
| tokenizer | `runtime.flerp.text_strixfold` (StrixFold `HfText`: a checkpoint's `tokenizer.json` + Jinja template), `runtime.flerp.text_strixite` (Qwen's) | Zig / C++ | builds / parity (Qwen only) | same **[I]**: upstream StrixFold serves GLM-5.3-Flash, whose `tokenizer.json` is byte-identical | a test that encodes fixed strings to colibri's ids |
| chat template | same module | Jinja | builds | file missing | the template file; raw turn string until then |
| executor (the layer loop) | `executor.strixfold.dsv41`, `executor.strixite.qwen4exp`, `executor.gufo.*`, `executor.hipfire.*` | each one model family | - | none | new: `executor.flerp.glm_moe_dsa` |
| MTP draft | `runtime.gufo.mtp`, strixite's MTP in `executor.strixite.qwen4exp` | Qwen heads | harvested / builds | none | later; the ABI's `flerp_draft_step` form fits |

Two sources outside the catalogue that cover more:

- **colibri's own GPU backend** (`colibri:c/backend_cuda.cu`, built for AMD with `HIP=1` through
  `backend_gpu_compat.h`, `colibri:c/Makefile:343-347`): int4-row kernels at exactly this format
  (`quant_matmul` :508, `w4a16_matmul` :693, `w4a16_gate_up` :726, grouped expert kernels :795-933), MLA absorb
  attention (:1056-1149), RMSNorm, RoPE, router logits and select (:2684-2817). Apache-2.0. It would be a new
  origin `colibri`: **same math and same format**. Whether it compiles for gfx1151 with TheRock was not tried.
  **[N]** Its README names gfx1201 (RDNA4), not gfx1151.
- **Upstream StrixFold's `glm` family** (`upstream/strixfold/zig/src/families/glm/`, 5,516 lines) is GLM-5.3-Flash
  on Metal (`glm.zig:1`, `kernels.zig:3` imports `metal`; `config.zig:1-45` fixes hidden 4096, 288 experts, KDA +
  MLA). `upstream/strixfold/docs/recipes/glm-5.3-flash.md` adds two-rank CUDA. No HIP path, a different model.
  Useful as a second reading of MLA + indexer + MTP ordering, not as code to assemble.

## 3. Multi-node

### 3.1 How StrixFold splits two nodes [R]

- "Lanes" (`runtime.strixfold.lanes`) are concurrent streams inside a round, not nodes. The cross-node part is the
  collective (`runtime.strixfold.coll`): every reduction in the model is a stream-ordered all-gather of fp32
  partials that each rank then adds in rank order, so a transport only moves bytes
  (`catalog/strixfold/runtime/coll/src/zig/src/families/dsv41/coll.zig:1-15`).
- The split is tensor-parallel by columns: 32 attention heads a rank, each expert's intermediate halved (gate/up by
  output column, down by input row), the vocabulary halved, Engram by hash column, the rest replicated
  (`catalog/strixfold/format/exl3/src/zig/src/families/dsv41/plan.zig:1-7`).
- Transports: `coll_tcp`, `coll_rdma` (RC write-with-immediate per slot; measured 30.8 decode tok/s on F+A,
  `catalog/strixfold/runtime/coll_rdma/module.toml:36`), `coll_nccl` (CUDA).

### 3.2 What is hard-wired

| item | evidence | consequence for four ranks of GLM-5.2 |
|---|---|---|
| HIP build refuses world 4 | `runtime/gpu_facade/src/zig/src/gpu/hip_facade.zig:11` (`.world4 = false`); `executor/dsv41/.../model.zig:207-208` ("the 2D split is CUDA's") | the four-node path has never run on gfx1151 |
| the four-node split is "2D": two TP2 pairs, quarters in whole 128-column blocks, with DeepSeek-V4.1's numbers in the code (vocabulary 1,010 blocks, 253 / 252 a quarter) | `prompt2d.zig:1-10`, `round2d.zig:1-12`, `draft2d.zig:15-22,57`, `plan.zig:6-7` | not reusable for another model's shapes |
| its exchanges go over NCCL, with RDMA rings for decode-size parts | `prompt2d.zig:6-10`, `ring2d.zig:1-7`; `coll.zig:4` ("the four-node split's point-to-point exchange (CUDA's NCCL only)") | no HIP transport for it in the catalogue |
| the executor is the dsv41 family: EXL3 weights, Triton kit, mHC streams, Engram | `executor/dsv41/module.toml:3,74-75` | none of the family code applies to GLM-5.2 |
| world count in the collective itself | `rdma_seq.zig:5` (`max_ranks = 8`), `rdma.zig:197` (accepts `2 <= world <= max_ranks`); `fabric/collective.zig:10` (64) | **not** limited to two; the byte-moving layer is general |
| a check program | `check_main.zig:39` ("this engine branch splits over one or two ranks only") | consistent with the above |

So from StrixFold, four ranks of GLM-5.2 can take the collective interface and the RDMA slot transport (Zig,
status builds, exercised on the pair), and nothing above them.

### 3.3 strixite-tp-rdma-v2 [R]

`upstream/strixite-tp-rdma-v2/docs/tp-rdma.md:1-50` and `runtime/tp_comm.hpp:3,57,74-101`:

- `TpComm`: world N in {2, 4}, RDMA write-with-immediate over libibverbs, all-reduce of FP32 partials summed in
  rank order with one rounding (bit-identical activations on every rank), all-gather, a reduce-scatter form for
  large exchanges, a TCP control mesh.
- Split loader `TpConfig{world, rank}` over `.strixw`; mirror protocol (`tp_mirror`: rank 0 sends every
  state-changing call to the other ranks before making it); split draft head; kernel changes for the narrow
  per-rank shapes.
- The document says it ran on 2 or 4 Strix Halo machines through the SN2700.
- **Not in the catalogue** (`docs/INTERFACES.md:288,531`). `tp_comm.{hpp,hip}` is model-agnostic C++ and the most
  direct four-rank communicator available; `tp_mirror` is written against strixite's `LmBackend`, which
  `executor.h` was derived from.
- gufo's `runtime.gufo.tp_verbs` is two-rank only and marked by its own notes as an unqualified merge checkpoint
  (`catalog/gufo/runtime/tp_verbs/module.toml:3-4`).

### 3.4 What four nodes need for this model

Three ways to cut it. All three fit memory (section 4).

| cut | what crosses nodes each token | new code | bandwidth ceiling **[I]** |
|---|---|---|---|
| **A. Layer stages** (node k owns a contiguous block of layers) | the residual stream, 6,144 f32 a row, three hops; control calls mirrored | a small stage relay (TCP is enough at 24.6 kB a row) | one node's: about 20 GB read a token over about 200 GB/s is roughly 10 tok/s |
| B. Expert-parallel (attention on one node, each node holds 64 of every layer's 256 experts) | one scatter and gather a layer, 75 a token | relay + routing of picks to owners; colibri's cluster mode does this over TCP (`colibri.c:3246-3384`, README "Local cluster mode") | experts read in parallel, attention serial |
| C. Tensor-parallel 4 (every rank holds a quarter of every weight) | about 2 all-reduces a layer, 156+ a token | harvest `tp_comm`, a split loader for row formats, mirror protocol | four nodes' |

Row scales make cut C straightforward for this format **[I]**: gate/up split by output rows keep their own scale;
down split by input columns shares its row scale across ranks and the partials add.

For first tokens cut A is the smallest: every node runs the single-node executor on a layer range, and because a
hop moves bytes, the tokens do not depend on where the cuts are. That property is testable on one node.

## 4. Memory plan

Per-node GPU memory of about 110 GiB is the caller's figure. The SGLang report's per-rank account (weights 75.7 +
KV 16.4 + recurrent state 4.4 + 15.9 free, in GB) sums to 112.4 GB, which is 104.7 GiB if those are decimal.
**[N: not measured on a worker]**

Cut A, native int4-row bytes:

| node | holds | weights GiB |
|---|---|---|
| A | embedding 0.89, layers 0..2 (0.54), layers 3..21 (19 x 4.61) | 89.0 |
| B | layers 22..40 | 87.6 |
| C | layers 41..59 | 87.6 |
| D | layers 60..77 (18 x 4.61), `model.norm`, `lm_head` 0.89, MTP layer 4.78 | 88.7 |
| total | | 352.9 |

- With the HFQ4-G256 transcode (+6.25 % on nibble payload): about 93 to 94.5 GiB a node, 375 GiB in all.
  `kv_b` handed to hipfire's kernels as F32 adds 58.7 MB a layer (4.6 GiB in all); as BF16 half that.
- KV: 576 values a token a layer. As BF16 that is 1,152 bytes; a node with 19 layers holds 21.9 kB a token:
  0.67 GiB at 32,768 tokens, 2.7 GiB at 131,072. Whole model: 90 kB a token (5.5 GiB at 65,536).
- Headroom a node: about 21 GiB native, about 16 GiB transcoded, before workspace. It fits on either figure for
  node memory, with less room at 104.7 GiB.
- Cut C would hold 88.2 GiB a rank plus the whole KV on every rank; cut B about 85.5 GiB of experts a node plus
  11 GiB of dense weights on the attention node (96.5 GiB there).
- Context that is *correct* is a separate limit: without indexer weights the model's own function is only
  reproduced up to 2,048 tokens (section 1.3).
- Staging: 353 GiB has to leave an SMR disk on control-node and reach four `/cache` NVMe drives (each node needs
  only its own shards; layer-to-shard mapping is in the headers). control-node's NVMe has 197 GiB free **[R]**, so
  it cannot hold a full copy.
- Bytes read a decoded token **[I, from the census]**: sparse layer 252 MB (attention 82.5, shared expert 18.9,
  eight experts 151) x 75, dense layers 0.6 GB, head 0.95 GB: about 20.4 GB.

## 5. Work items, smallest honest path to first tokens

Sizes are rough agent-days and assume the pod's GPU and worker windows are available when asked for; they are not.

| # | item | new or adapted | size | the gate that proves it |
|---|---|---|---|---|
| 1 | **Reference.** Build colibri's CPU `glm` on the pod; prove the build on its tiny oracle; stage the container where it can be read; record teacher-forced predictions, greedy continuations and `ROUTE_TRACE` for a fixed prompt set (all under 2,048 tokens), with the container's sha256 list and colibri's commit | no engine code; a gate runner `gate/glm52_reference.py` and its config written **before** the first run | 1 to 2 | the tiny oracle reproduces `ref_glm.json`; the reference files are committed with hashes; a second run gives the same tokens |
| 2 | `format.flerp.colibri_rows`: name table and format inference as colibri does it, host dequantizers, load of a layer range to the device, transcode-on-load to HFQ4/HFQ8 groups | new, on `format.strixite.safetensors*` | 1 to 2 | host test: dequantized rows equal colibri's for sampled tensors of every kind, bit for bit; the census of section 1.1 reproduced from code |
| 3 | Interface gates for the hipfire kernels this recipe binds: HFQ4-G256 GEMV, HFQ8-G256 GEMV, indexed expert gate/up and down, bias-aware router (with ties), latent attention decode at 64 x (192 + 64), 256 values, 512 latent, tail RoPE | gate configs and CPU references, new; kernels as harvested, launched as embedded code objects through `kernel.flerp.adapter_rt` the way phase 4 does | 3 to 4 | `flerp gate interface <name>`: parity against a CPU reference written from colibri's formulas, within the any-order summation bound; GB/s recorded |
| 4 | `executor.flerp.glm_moe_dsa` behind `executor.h`: layer loop of section 1.3 over a layer range, append-only latent KV, reset, full logits rows to the host; no drafts, no snapshots, one row a forward first | new C++, modelled on `executor.flerp.strixite_abi` for the ABI side and on `colibri.c` for order of operations | 4 to 6 | **partial-model gate on one GPU**: layers 0..k that fit control-node's free GPU memory, teacher-forced, router picks equal to the reference's `ROUTE_TRACE` for those layers on every position (ties excepted and listed) |
| 5 | `runtime.flerp.stage_relay`: one binary a node, layer range by deployment key, residual rows and mirrored control calls over TCP on the rail | new (strixite's `tp_mirror` is the pattern; not harvested) | 2 to 3 | cut invariance on one node: the same partial model as one stage and as two processes gives byte-equal residual rows |
| 6 | Recipe `glm52-colibri.flerp-stage4` (`topology = "tp4"` needs a new value such as `stage4`; a SPEC change in the same commit), text binding (`runtime.flerp.text_strixfold` or a C++ text module), `serve.flerp.http` | recipe and bindings new; modules existing | 1 to 2 | tokenizer ids equal colibri's on fixed strings; `flerp check`, `flerp assemble` with no hand edits |
| 7 | **Four-node run** on A B C D | launcher for the gate, under the pod's hold rules | 1 to 2 plus the window | recipe gate, criterion `token-agreement`: teacher-forced agreement with the reference position by position, first divergence reported with the two logits; greedy continuations compared; tok/s recorded |

After first tokens, in the order they pay:

| # | item | size | note |
|---|---|---|---|
| 8 | native int4-row and int8-row kernels (`kernel.flerp.linear_rows`), or harvest colibri's HIP backend as origin `colibri` | 3 to 5 | removes the transcode, saves about 22 GiB, and can match colibri's arithmetic more closely |
| 9 | multi-row forwards and prefill GEMM (`kernel.hipfire.gemm_hfq`), verify with keep / drop | 3 to 5 | prompt speed; needed before MTP |
| 10 | MTP drafts from layer 78 (int8) through `flerp_draft_step` | 2 to 3 | colibri reports 2.2 to 2.8 tokens a forward |
| 11 | cut B or C for decode speed (harvest `tp_comm` from strixite-tp-rdma-v2 as modules) | 5 to 10 | the stage cut is bound by one node's memory bandwidth |
| 12 | DSA indexer | blocked | needs `out-idx-*`, which needs the 756 GB FP8 download; the kernels are near (section 2) |
| 13 | snapshots, prompt cache, KV quantization | 3+ | none of the existing caches fit a latent KV row |

First tokens are items 1 to 7: about 13 to 21 agent-days, not counting waits for hardware.

Constraints that are not engineering **[R from SPEC.md section 10; pod state not re-checked]**: workers serve the
pod's roles, and the only worker use SPEC describes is phase 3's F+A pair gate under `pod-quad hold`. A four-node
gate on A B C D is not covered and needs its rule written into SPEC first; control-node's GPU is shared and the
pod's jobs come first.

### The single riskiest unknown

**Whether colibri's CPU path can be made to produce a trustworthy reference for this exact container on this pod.**
Every gate in the table compares against it, and it is the only definition of what these bytes mean (format
inference by byte count, rounding, dense attention in place of DSA). It has never been built or run here, the
container is on a slow shingled disk, control-node's RAM is almost fully committed, and the container's own
integrity against `pod:/home/pod/ark-hashes/jlnsrk__GLM-5.2-colibri-int4.sha256` was not verified in this
pass. If the reference cannot be produced, phase 5 has a model that emits tokens and nothing to hold them to.
Item 1 is first for that reason and is cheap to try.

Second, and a limit on usefulness rather than on first tokens: behaviour beyond 2,048 tokens of context. Without
the indexer the engine computes dense attention, which is not what the model was trained with, and nothing on the
pod can say how much that costs.

## 6. For comparison: Qwen3-Coder-480B-A35B as a Q4_K GGUF

Not on the pod; the architecture facts in this section are from memory of the public model card and were **not
verified** [N]: a plain Qwen3 MoE, 62 layers, hidden 6144, grouped-query softmax attention with full RoPE, 160
experts with 8 a token, softmax router with renormalized top-k, no shared expert, no MTP head, no indexer; a
Q4_K_M file is about 290 GB and mixes Q4_K and Q6_K blocks.

| operation | catalogue | fit |
|---|---|---|
| container and weight format | `format.gufo.gguf_kquant` (harvested), `format.hipfire.loader` GGUF path (harvested) | **same format**: no transcode, no re-quantization |
| decode GEMV on Q4_K / Q6_K / Q8_0 | `kernel.gufo.qwen_linear_decode`, `kernel.hipfire.gemv_ggml`, adapter `kernel.flerp.gguf_gemv_hipfire` | same; **the one weight-reading interface with a passed hardware gate** for two providers (`gguf-gemv`, Q8_0 and Q4_K; README.md:140) |
| prefill GEMM on K-quants | `kernel.gufo.qwen_linear_prefill`, `kernel.gufo.qfn_mmq` | same; builds |
| routed experts on K-quants | `kernel.gufo.qfn_mmq` (`.mmid`), hipfire's fused gate-up on Q4_K | same format; written for 512-expert Qwen3.8 shapes, shape limits not checked |
| router (softmax top-k, renormalize) | `kernel.strixite.moe_router`, gufo `RouterTopK` (both pass `moe-router`), `kernel.flerp.moe_router_gufo` | same; gated |
| GQA attention + RoPE + F16 KV | `kernel.gufo.qwen_attention`, `kernel.gufo.qwen_norm_rope`, `kernel.hipfire.attention`, `kernel.strixite.attention` | same; builds |
| sampling, logits, residual | gated providers as in section 2 | same |
| tokenizer, template | `runtime.flerp.text_strixite` is Qwen's tokenizer class and ChatML-style template (parity for Qwen3.8); a GGUF carries its own tokenizer and template | near; not checked against Qwen3-Coder's template |
| executor | `executor.gufo.qwen` is the Qwen3.5 / 3.6 dense engine with Gated DeltaNet and appears to have no expert code (a search of its sources for an expert count found nothing; its manifest names no MoE); `executor.gufo.qwen38_flash_next` and `executor.strixite.qwen4exp` are the hyper-connected Flash-Next model | **none**: a new plain-MoE executor, simpler than GLM-5.2's (no latent attention, no MTP) |
| four nodes | the same gap as section 3 | same work (items 5, 11) |
| reference | llama.cpp reads the file as it is | far easier than colibri: a standard engine, standard format |

In short: for the eventual target the format, the weight-reading kernels, the router and the reference are
already in hand and partly gated; the missing pieces are the executor and the four-node cut. GLM-5.2 additionally
needs a format module, a transcode or new kernels, latent attention that no executor here has driven, and a
reference built from a third-party CPU engine. The executor skeleton, the stage relay, the recipe shape and the
four-node gate runner written for GLM-5.2 carry over; the format and attention work does not.

## 7. Not verified in this pass

- Worker GPU memory available to one process (taken as about 110 GiB).
- That colibri builds and runs on the pod; its speed here; whether its tiny GLM fixture is present.
- The container's hashes against the recorded sha256 list.
- transformers' `modeling_glm_moe_dsa.py` (colibri's claim of fidelity is taken from its header and README).
- The attention scoring loop of colibri (`:4530-4590`) line by line; the `absorb` variants.
- Any hipfire kernel's numerics: all are compile-gated only. hipfire's RMSNorm formula, the `rope_tail_*`
  arguments, the router's tie rule and the indexed expert kernels' shape limits were not compared with colibri's.
- That colibri's `backend_cuda.cu` compiles for gfx1151.
- That StrixFold's `HfText` tokenizes GLM's `tokenizer.json` correctly on HIP builds (inferred from upstream
  serving GLM-5.3-Flash with the identical file).
- The official GLM-5.2 chat template (not in the model directory).
- Whether A B C D can be held together now (SPEC section 10 as written does not cover it).
- Every Qwen3-Coder-480B fact in section 6.
