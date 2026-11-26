# Strata kernels by strixflerp interface

One row for each HIP kernel source of Strata at v0.1.40 (1cbcacbcae29). Card ORIGIN-STRATA ef16e87953. Written by
`docs/harvest/harvest_strata.py` from its `FAMILIES` table, read from the sources and their headers.

- **tag**: the strixflerp interface class the file matches, `<family>.<op>/<weight format>/<dtype>`, or none. The
  module's `[[provides]]` entry carries the full name with a `.strata` variant tag.
- **same op in the catalogue**: kernel modules of other origins that provide that `<family>.<op>`. The same op is
  not the same operands: every Strata kernel takes raw device pointers in Strata's layouts (S-family planes, the QSA
  page table, the 4-stream residual), so each row needs an adapter before it can stand in for a strixflerp module.

| file | lines | module | tag | same op in the catalogue |
|---|---:|---|---|---|
| `src/kernels/cuda/dequant_s2.cu` | 51 | `kernel.strata.s_gemv` | none | - |
| `src/kernels/cuda/s2_gemv.cu` | 63 | `kernel.strata.s_gemv` | `linear.decode/strata-s2/f16-f32` | `kernel.ciru_inkling.glue_decode`, `kernel.ciru_inkling.quant_decode`, `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemv`, `kernel.fastasfuck.linear`, `kernel.flerp.gguf_gemv_hipfire` and 29 more |
| `src/kernels/cuda/s2_gemv_fast.cu` | 153 | `kernel.strata.s_gemv` | `linear.decode/strata-s2/f16-f32` | `kernel.ciru_inkling.glue_decode`, `kernel.ciru_inkling.quant_decode`, `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemv`, `kernel.fastasfuck.linear`, `kernel.flerp.gguf_gemv_hipfire` and 29 more |
| `src/kernels/cuda/s2_gemv_quads.cu` | 93 | `kernel.strata.s_gemv` | `linear.decode/strata-s2/f16-f32` | `kernel.ciru_inkling.glue_decode`, `kernel.ciru_inkling.quant_decode`, `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemv`, `kernel.fastasfuck.linear`, `kernel.flerp.gguf_gemv_hipfire` and 29 more |
| `src/kernels/cuda/s2_gemv_q8.cu` | 96 | `kernel.strata.s_gemv` | `linear.decode/strata-s2/q8_0-f32` | `kernel.ciru_inkling.glue_decode`, `kernel.ciru_inkling.quant_decode`, `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemv`, `kernel.fastasfuck.linear`, `kernel.flerp.gguf_gemv_hipfire` and 29 more |
| `src/kernels/cuda/s_gemv.cu` | 694 | `kernel.strata.s_gemv` | `linear.decode/strata-s/f16+q8_k+q8_0-f32` | `kernel.ciru_inkling.glue_decode`, `kernel.ciru_inkling.quant_decode`, `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemv`, `kernel.fastasfuck.linear`, `kernel.flerp.gguf_gemv_hipfire` and 29 more |
| `src/kernels/cuda/native_mmvq.cu` | 2239 | `kernel.strata.gguf_gemv` | `linear.decode/gguf-kquant+iq4+q2_0/q8_1-f32` | `kernel.ciru_inkling.glue_decode`, `kernel.ciru_inkling.quant_decode`, `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemv`, `kernel.fastasfuck.linear`, `kernel.flerp.gguf_gemv_hipfire` and 29 more |
| `src/kernels/cuda/iq_kernels.cu` | 3533 | `kernel.strata.gguf_gemv` | `linear.decode/gguf-iq+kquant/q8_1-f32` | `kernel.ciru_inkling.glue_decode`, `kernel.ciru_inkling.quant_decode`, `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemv`, `kernel.fastasfuck.linear`, `kernel.flerp.gguf_gemv_hipfire` and 29 more |
| `src/kernels/cuda/quantize_act.cu` | 352 | `kernel.strata.gguf_gemv` | `quant.activation/-/q8_0+q8_k` | `kernel.radiance.halo_quant_act` |
| `src/kernels/cuda/bf16_gemv.cu` | 141 | `kernel.strata.bf16_gemv` | `linear.dense/bf16/bf16-f32` | `kernel.fastasfuck.linear`, `kernel.flerp.dense_bf16_strixfold`, `kernel.gufo.qwen_linear_decode`, `kernel.hipfire.gemv_float`, `kernel.radiance.halo_gemm_bf16`, `kernel.strixfold.dense_bf16` and 1 more |
| `src/kernels/cuda/native_bf16.cu` | 370 | `kernel.strata.bf16_gemv` | `linear.dense/bf16/f32` | `kernel.fastasfuck.linear`, `kernel.flerp.dense_bf16_strixfold`, `kernel.gufo.qwen_linear_decode`, `kernel.hipfire.gemv_float`, `kernel.radiance.halo_gemm_bf16`, `kernel.strixfold.dense_bf16` and 1 more |
| `src/kernels/cuda/router_top10.cu` | 497 | `kernel.strata.moe` | `moe.router/-/f32` | `kernel.ciru_inkling.glue_decode`, `kernel.crucible.ops`, `kernel.fastasfuck.moe_router`, `kernel.flerp.moe_router_gufo`, `kernel.flerp.moe_router_hipfire`, `kernel.gufo.ds4_moe` and 6 more |
| `src/kernels/cuda/native_router.cu` | 179 | `kernel.strata.moe` | `moe.router/-/f32` | `kernel.ciru_inkling.glue_decode`, `kernel.crucible.ops`, `kernel.fastasfuck.moe_router`, `kernel.flerp.moe_router_gufo`, `kernel.flerp.moe_router_hipfire`, `kernel.gufo.ds4_moe` and 6 more |
| `src/kernels/cuda/native_moe.cu` | 170 | `kernel.strata.moe` | `moe.combine/-/f32` | `kernel.ciru_inkling.moe_prefill`, `kernel.ciru_iu4.mmb`, `kernel.gufo.qfn_kernels`, `kernel.hipfire.moe`, `kernel.lucebox.moe`, `kernel.radiance.halo_moe` |
| `src/kernels/cuda/shared_expert.cu` | 419 | `kernel.strata.moe` | `moe.experts-decode/strata-s+bf16/f32` | `kernel.ciru_inkling.moe_decode`, `kernel.ciru_iu4.c1`, `kernel.crucible.moe_decode`, `kernel.fastasfuck.linear`, `kernel.flerp.linear_q4p`, `kernel.flerp.moe_kquant` and 15 more |
| `src/kernels/cuda/s2_expert_grouped.cu` | 1236 | `kernel.strata.moe` | `moe.experts-decode/strata-s2/q8_0-f32` | `kernel.ciru_inkling.moe_decode`, `kernel.ciru_iu4.c1`, `kernel.crucible.moe_decode`, `kernel.fastasfuck.linear`, `kernel.flerp.linear_q4p`, `kernel.flerp.moe_kquant` and 15 more |
| `src/kernels/cuda/gdn.cu` | 248 | `kernel.strata.gdn` | `mixer.gdn/-/f32` | `kernel.crucible.gdn`, `kernel.fastasfuck.gdn`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_gdn`, `kernel.hipfire.gdn`, `kernel.libstrix.gdn` and 4 more |
| `src/kernels/cuda/fused_gdn.cu` | 166 | `kernel.strata.gdn` | `mixer.gdn/-/f32` | `kernel.crucible.gdn`, `kernel.fastasfuck.gdn`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_gdn`, `kernel.hipfire.gdn`, `kernel.libstrix.gdn` and 4 more |
| `src/kernels/cuda/native_gdn.cu` | 122 | `kernel.strata.gdn` | `mixer.gdn/-/f32` | `kernel.crucible.gdn`, `kernel.fastasfuck.gdn`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_gdn`, `kernel.hipfire.gdn`, `kernel.libstrix.gdn` and 4 more |
| `src/kernels/cuda/native_gdn_preprocess.cu` | 199 | `kernel.strata.gdn` | `mixer.gdn-front/-/f32` | `kernel.gufo.qwen_gdn`, `kernel.hipfire.gdn`, `kernel.libstrix.gdn`, `kernel.lucebox.gdn` |
| `src/prefill/gdn_rec_parity.cu` | 886 | `kernel.strata.gdn` | none | - |
| `src/kernels/cuda/gr.cu` | 499 | `kernel.strata.hc` | `hc.mix/bf16/f32` | `kernel.crucible.hc`, `kernel.fastasfuck.hc`, `kernel.gufo.ds4_hc`, `kernel.gufo.qfn_kernels`, `kernel.hipfire.hc`, `kernel.lucebox.hc` and 3 more |
| `src/kernels/cuda/fused_gr.cu` | 1963 | `kernel.strata.hc` | `hc.mix/bf16/f32` | `kernel.crucible.hc`, `kernel.fastasfuck.hc`, `kernel.gufo.ds4_hc`, `kernel.gufo.qfn_kernels`, `kernel.hipfire.hc`, `kernel.lucebox.hc` and 3 more |
| `src/kernels/cuda/native_gr_norm.cu` | 109 | `kernel.strata.hc` | `norm.rmsnorm/-/f32` | `kernel.ciru_inkling.glue_decode`, `kernel.gufo.ds4_norm_rope`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_norm_rope`, `kernel.hipfire.norm_rope`, `kernel.lucebox.norm` and 4 more |
| `src/kernels/cuda/native_gr_postops.cu` | 138 | `kernel.strata.hc` | `hc.mix-post/-/f32` | - |
| `src/kernels/cuda/cvec.cu` | 174 | `kernel.strata.hc` | `steer.add-rows/-/f32` | `kernel.strixite.steer`, `runtime.crucible.ablation` |
| `src/kernels/cuda/qsa.cu` | 821 | `kernel.strata.qsa` | `attn.qsa-indexer/-/f32` | `kernel.ciru_iu4.attn`, `kernel.crucible.qsa`, `kernel.fastasfuck.qsa_indexer`, `kernel.gufo.qfn_kernels`, `kernel.hipfire.indexer_topk`, `kernel.hipfire.pm_gfx1151` and 5 more |
| `src/kernels/cuda/qsa_decode_attn.cu` | 582 | `kernel.strata.qsa` | `attn.softmax-gqa/kv-f16+i8+q4_0/f32` | `kernel.ciru_inkling.attn_decode`, `kernel.ciru_iu4.attn`, `kernel.fastasfuck.attention_gqa`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_attention`, `kernel.hipfire.attention` and 6 more |
| `src/kernels/cuda/qsa_prompt_attn.cu` | 1522 | `kernel.strata.qsa` | `attn.softmax-gqa/kv-f16+i8+q4_0/f32` | `kernel.ciru_inkling.attn_decode`, `kernel.ciru_iu4.attn`, `kernel.fastasfuck.attention_gqa`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_attention`, `kernel.hipfire.attention` and 6 more |
| `src/kernels/cuda/qsa_select.cu` | 1279 | `kernel.strata.qsa` | `attn.indexer-topk/-/f32` | `kernel.ciru_iu4.attn`, `kernel.gufo.ds4_indexer`, `kernel.hipfire.indexer_topk`, `kernel.strixfold.aux`, `kernel.strixfold.triton_kit` |
| `src/kernels/cuda/native_qsa.cu` | 167 | `kernel.strata.qsa` | `norm.rmsnorm/-/f32` | `kernel.ciru_inkling.glue_decode`, `kernel.gufo.ds4_norm_rope`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_norm_rope`, `kernel.hipfire.norm_rope`, `kernel.lucebox.norm` and 4 more |
| `src/kernels/cuda/native_qsa_indexer.cu` | 308 | `kernel.strata.qsa` | `attn.qsa-indexer/-/f32` | `kernel.ciru_iu4.attn`, `kernel.crucible.qsa`, `kernel.fastasfuck.qsa_indexer`, `kernel.gufo.qfn_kernels`, `kernel.hipfire.indexer_topk`, `kernel.hipfire.pm_gfx1151` and 5 more |
| `src/kernels/cuda/native_qsa_score.cu` | 213 | `kernel.strata.qsa` | `attn.indexer-score/-/f32` | `kernel.gufo.ds4_indexer`, `kernel.hipfire.indexer_topk`, `kernel.lucebox.indexer`, `kernel.strixfold.indexer_score`, `kernel.strixfold.triton_kit`, `kernel.tensorfold.cuda_glm` |
| `src/kernels/cuda/native_flash_attn.cu` | 212 | `kernel.strata.qsa` | `attn.softmax-gqa/kv-f16/f32` | `kernel.ciru_inkling.attn_decode`, `kernel.ciru_iu4.attn`, `kernel.fastasfuck.attention_gqa`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_attention`, `kernel.hipfire.attention` and 6 more |
| `src/kernels/cuda/kv_q4.cu` | 241 | `kernel.strata.kvcache` | `kvcache.store/q4_0-wht/f32` | `kernel.ciru_inkling.attn_decode`, `kernel.ciru_inkling.attn_prefill`, `kernel.gufo.ds4_attention`, `kernel.hipfire.kv_cache`, `kernel.strixfold.triton_kit` |
| `src/kernels/cuda/kv_q8.cu` | 147 | `kernel.strata.kvcache` | `kvcache.quant8/-/f32` | `kernel.fastasfuck.kv8`, `kernel.strixite.kv8` |
| `src/kernels/cuda/kv_stream.cu` | 286 | `kernel.strata.kvcache` | `kvcache.tiering-strata-stream/-/-` | `kernel.radiance.halo_kv`, `runtime.crucible.state_disk`, `runtime.hipfire.runtime` |
| `src/kernels/cuda/rope.cu` | 136 | `kernel.strata.rope` | `pos.rope/-/f32` | `kernel.crucible.ops`, `kernel.fastasfuck.rope`, `kernel.gufo.qwen_norm_rope`, `kernel.hipfire.norm_rope`, `kernel.strixfold.aux`, `kernel.strixite.rope` and 2 more |
| `src/kernels/cuda/native_rope.cu` | 245 | `kernel.strata.rope` | `pos.rope/-/f32` | `kernel.crucible.ops`, `kernel.fastasfuck.rope`, `kernel.gufo.qwen_norm_rope`, `kernel.hipfire.norm_rope`, `kernel.strixfold.aux`, `kernel.strixite.rope` and 2 more |
| `src/kernels/cuda/ple.cu` | 390 | `kernel.strata.ple` | `ple.inject/-/f32` | `kernel.crucible.ops`, `kernel.fastasfuck.ple`, `kernel.gufo.qfn_kernels`, `kernel.radiance.halo_ple`, `kernel.strixite.ple` |
| `src/kernels/cuda/native_ple_postops.cu` | 221 | `kernel.strata.ple` | `ple.inject/-/f32` | `kernel.crucible.ops`, `kernel.fastasfuck.ple`, `kernel.gufo.qfn_kernels`, `kernel.radiance.halo_ple`, `kernel.strixite.ple` |
| `src/kernels/cuda/elementwise.cu` | 446 | `kernel.strata.elementwise` | `tensor.ops/-/f32` | `kernel.hipfire.tensor_ops` |
| `src/kernels/cuda/sampler.cu` | 1081 | `kernel.strata.sampling` | `sample.next-token/-/f32` | `kernel.radiance.halo_sample`, `runtime.crucible.sampling`, `runtime.gufo.sampling`, `runtime.hipfire.runtime`, `runtime.strixfold.sampling`, `runtime.strixite.sampler` and 1 more |
| `src/kernels/cuda/verify_kernels.cu` | 1219 | `kernel.strata.verify` | `mixer.gdn/-/f32` | `kernel.crucible.gdn`, `kernel.fastasfuck.gdn`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_gdn`, `kernel.hipfire.gdn`, `kernel.libstrix.gdn` and 4 more |
| `src/kernels/cuda/dequant_bf16.cu` | 272 | `kernel.strata.prefill` | none | - |
| `src/prefill/gemm.cu` | 827 | `kernel.strata.prefill` | `linear.prefill/bf16/bf16-f32` | `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemm`, `kernel.fastasfuck.linear`, `kernel.flerp.wmma_q4p`, `kernel.gufo.ds4_linear`, `kernel.gufo.ds4_mmq` and 19 more |
| `src/prefill/kernels.cu` | 1755 | `kernel.strata.prefill` | `hc.mix-prefill/bf16/f32` | `kernel.ciru_iu4.mmb`, `kernel.crucible.hc`, `kernel.fastasfuck.hc_wmma`, `kernel.libstrix.gemm`, `kernel.strixfold.mhc_mix`, `kernel.strixfold.triton_kit` and 1 more |
| `src/prefill/wmma_gemm.cu` | 737 | `kernel.strata.prefill` | `linear.prefill/f16+bf16/f32` | `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemm`, `kernel.fastasfuck.linear`, `kernel.flerp.wmma_q4p`, `kernel.gufo.ds4_linear`, `kernel.gufo.ds4_mmq` and 19 more |
| `src/prefill/gemm_bf16_parity.cu` | 155 | `kernel.strata.prefill` | none | - |
| `src/prefill/moe_fused.cu` | 727 | `kernel.strata.prefill_moe_fused` | `moe.experts-prefill/gguf-q2_0/i8-f32` | `kernel.ciru_inkling.moe_prefill`, `kernel.ciru_iu4.c1`, `kernel.ciru_iu4.mmb`, `kernel.crucible.moe_prefill`, `kernel.fastasfuck.moe_grouped`, `kernel.flerp.moe_grouped_p` and 13 more |
| `src/prefill/moe_mmq.cu` | 298 | `kernel.strata.prefill_moe_ggml` | `moe.experts-prefill/gguf-kquant/q8_1-f32` | `kernel.ciru_inkling.moe_prefill`, `kernel.ciru_iu4.c1`, `kernel.ciru_iu4.mmb`, `kernel.crucible.moe_prefill`, `kernel.fastasfuck.moe_grouped`, `kernel.flerp.moe_grouped_p` and 13 more |
| `src/prefill/moe_fused_iq.cu` | 1111 | `kernel.strata.prefill_moe_ggml` | `moe.experts-prefill/gguf-iq+kquant/i8-f32` | `kernel.ciru_inkling.moe_prefill`, `kernel.ciru_iu4.c1`, `kernel.ciru_iu4.mmb`, `kernel.crucible.moe_prefill`, `kernel.fastasfuck.moe_grouped`, `kernel.flerp.moe_grouped_p` and 13 more |
| `src/prefill/ggml_cuda_host.cu` | 147 | `kernel.strata.prefill_moe_ggml` | none | - |
| `src/core/device.cu` | 364 | `kernel.strata.core` | `device.info-strata/-/-` | `kernel.fastasfuck.common`, `kernel.strixite.device_info` |
| `src/core/pinned.cu` | 800 | `kernel.strata.core` | none | - |
| `src/core/remote_expert_opt.cu` | 194 | `kernel.strata.core` | none | - |
| `tests/core/dma_batch_parity.cu` | 107 | `kernel.strata.core` | none | - |

57 files: s_gemv 6, gguf_gemv 3, bf16_gemv 2, moe 5, gdn 5, hc 5, qsa 8, kvcache 3, rope 2, ple 2, elementwise 1, sampling 1, verify 1, prefill 5, prefill_moe_fused 1, prefill_moe_ggml 3, core 4. 49 carry a tag, 8 are none (`dequant_s2.cu`, `gdn_rec_parity.cu`, `dequant_bf16.cu`, `gemm_bf16_parity.cu`, `ggml_cuda_host.cu`, `pinned.cu`, `remote_expert_opt.cu`, `dma_batch_parity.cu`).
`src/kernels/cuda/s26_tsum.cuh` (the S26 tree sum, a device header) is in `kernel.strata.common`.

## What each kernel does

- `src/kernels/cuda/dequant_s2.cu`: S2 codes and scales to FP32, (code - 1) * scale, one thread an element: the decode every S2 GEMV inlines, kept host-callable as the parity reference.
- `src/kernels/cuda/s2_gemv.cu`: the naive S2 GEMV, one thread an output row, dequantizing on the fly with FP32 accumulation inside the row.
- `src/kernels/cuda/s2_gemv_fast.cu`: two levers on the S2 GEMV (vector loads and a split reduction) that measured negative or neutral; the file says so and is kept as the record.
- `src/kernels/cuda/s2_gemv_quads.cu`: S2 with the code load amortised over four elements, a test of the instruction-bound hypothesis.
- `src/kernels/cuda/s2_gemv_q8.cu`: the S2 GEMV over Q8_0-quantized activations, ggml_mul_mat's vec_dot_type rule for Q2_0, so GPU and CPU expert paths agree on a token.
- `src/kernels/cuda/s_gemv.cu`: one GEMV for S2, S4 and S8 (value = cb[code] * scale + offset), templated on the code width, with split and async variants and Q8_K / Q8_0 activation forms.
- `src/kernels/cuda/native_mmvq.cu`: llama.cpp's generic MMVQ (quantize.cu, vecdotq.cuh, mmvq.cu) for ten GGUF block types against q8_1 activations, with a fused SwiGLU-then-quantize and multi-column forms.
- `src/kernels/cuda/iq_kernels.cu`: the i-quant and UD-Q4_K_XL formats on the GPU: llama.cpp's vec_dot_*_q8_1, dequantizers and q8_1 quantizer, transcribed, plus embedding-row gathers.
- `src/kernels/cuda/quantize_act.cu`: activation quantization to Q8_0 and Q8_K (and back) the way ggml_mul_mat converts src1 to the weight's vec_dot_type.
- `src/kernels/cuda/bf16_gemv.cu`: the BF16 GEMV: one thread a row as the reference, one warp a row with lanes striding the reduction for the shapes that need it.
- `src/kernels/cuda/native_bf16.cu`: llama.cpp's FP32-activation MMVF (mmvf.cu) for a contiguous BF16 matrix: no fusion, ids or channels.
- `src/kernels/cuda/router_top10.cu`: the router's softmax / top-k / renormalise half over (n_tokens, n_expert) logits, with a variant search.
- `src/kernels/cuda/native_router.cu`: llama.cpp's topk-moe for one token, 512 experts, top 10, softmax, no selection bias.
- `src/kernels/cuda/native_moe.cu`: llama.cpp's moe-weighted-reduction for one token or a batch, also over the GPU hit rows, with an optional shared-expert gate.
- `src/kernels/cuda/shared_expert.cu`: the shared expert: silu(x Wg) * (x Wu) then Wd on S-family or BF16 weights, a sigmoid scalar gate per token, and moe_combine of the routed parts with it.
- `src/kernels/cuda/s2_expert_grouped.cu`: R4's GPU expert tier: picks the routed experts that are resident on the GPU, runs their S2 gate / up / down GEMVs grouped (one warp an output row, the CPU expert's arithmetic over the same bytes) and adds the results into the routed parts.
- `src/kernels/cuda/gdn.cu`: the GDN step (one thread owns a (j, h) column end to end, no barrier), the depthwise conv step, L2 norm, beta gate and the gated output norm, transcribed from ref/gdn.py.
- `src/kernels/cuda/fused_gdn.cu`: the conv with L2 norm, the alpha / beta projections, and the delta-net step fused with its RMS output norm, laid out so a warp reads contiguous state.
- `src/kernels/cuda/native_gdn.cu`: llama.cpp's gated_delta_net recurrence for one token with Strata's state addressing: a warp owns one (head, column) and keeps its rows in registers.
- `src/kernels/cuda/native_gdn_preprocess.cu`: llama.cpp's GDN front and back ends: four-tap conv then SiLU, L2 norm, beta and gate, gated output norm.
- `src/prefill/gdn_rec_parity.cu`: a stand-alone parity program for the prompt path's chunked GDN recurrence (its own reference kernels and main); not a library unit.
- `src/kernels/cuda/gr.cu`: gr_read (norm, down, SiLU, up, gate and mean, inject) and gr_write over the four residual streams, single-token and multi-token.
- `src/kernels/cuda/fused_gr.cu`: the hyper-connection read in two kernels with the previous half's write folded in, several variants selected at run time.
- `src/kernels/cuda/native_gr_norm.cu`: llama.cpp's weighted FP32 RMSNorm for contiguous rows, independent statistics a row.
- `src/kernels/cuda/native_gr_postops.cu`: llama.cpp's single-token GR post-operations (dsv4-hc.cu, scale, unary): down SiLU, the pre-gated mix and the post combine.
- `src/kernels/cuda/cvec.cu`: a control vector on the residual stream (llama.cpp's llama_adapter_cvec with the projection-mode extension): after a layer's FFN write, h . v over the stream and the update, bitwise the write fused_gr would fold.
- `src/kernels/cuda/qsa.cu`: the spec-of-record QSA: KV append and gather on the paged FP16 pools, the indexer key append and pooled block scores, top-512 selection and attention over the selected cells, transcribed from ref/qsa.py.
- `src/kernels/cuda/qsa_decode_attn.cu`: split-K decode attention over the selected cells, reading the KV pools through the page table, with per-format bodies for FP16, INT8, Q4_0 and the hybrid K8V4 cache.
- `src/kernels/cuda/qsa_prompt_attn.cu`: the prompt path's QSA attention on matrix cores (MMA on CUDA, an RDNA WMMA kernel on HIP), replacing one-query-at-a-time decode attention in the prompt.
- `src/kernels/cuda/qsa_select.cu`: the indexer's block scores (warp kernel and a matrix-core one) and a radix top-k over blocks for many queries at once, with a cluster variant and a reference.
- `src/kernels/cuda/native_qsa.cu`: llama.cpp's single-token QSA norms (weighted and grouped RMSNorm) and the output gate.
- `src/kernels/cuda/native_qsa_indexer.cu`: llama.cpp's indexer key cache: F16 raw keys, F32 pooling, weighted RMSNorm and text IMRoPE (set-rows, norm, rope).
- `src/kernels/cuda/native_qsa_score.cu`: llama.cpp's MMF dispatch for the indexer block scores: pooled [128, max_blocks] against query [128, 4], n_kv padded to 256.
- `src/kernels/cuda/native_flash_attn.cu`: llama.cpp's F16 vector FlashAttention specialised to Q 24 x 256, KV 2 x 256, one query.
- `src/kernels/cuda/kv_q4.cu`: FWHT-256 rotation (one warp a row, orthonormal, its own inverse) then ggml q4_0 for K and V, append (step, steps, batch) and gather.
- `src/kernels/cuda/kv_q8.cu`: INT8 K / V codes with one FP16 scale a 64-value group a cell and KV head; append and gather.
- `src/kernels/cuda/kv_stream.cu`: KV streaming: resolve the selected pages into the VRAM slot ring, stage and unstage blocks to and from pinned host memory, ring tables and restore.
- `src/kernels/cuda/rope.cu`: NEOX pairing (i, i + n_rot/2), partial rotation from host-built cos / sin tables.
- `src/kernels/cuda/native_rope.cu`: llama.cpp's rope_multi for text positions (equal across the four IMRoPE sections), and the QSA RMSNorm fused with it.
- `src/kernels/cuda/ple.cu`: the PLE block for one token: gathered rows, key / value projections, grouped norms, the conv over the history and the residual update.
- `src/kernels/cuda/native_ple_postops.cu`: llama.cpp's build_ple after the two projections: norms, conv, reduction and the update, single token and batched.
- `src/kernels/cuda/elementwise.cu`: embedding_gather, gdn_gate, scale_inplace, add_inplace, silu_inplace, f32_to_f16_bulk, f32_to_bf16_bulk, rms_norm_weighted and helpers.
- `src/kernels/cuda/sampler.cu`: penalties, top-k, top-p, min-p, temperature and pick, plus sample_greedy_cluster and the coupled-draft stage / sample.
- `src/kernels/cuda/verify_kernels.cu`: gdn_conv_l2_multi, gdn_conv_commit, gdn_ab_multi, gdn_step_norm_multi, wait_flag_ge(_or), gpu_stamp, resident_plan.
- `src/kernels/cuda/dequant_bf16.cu`: native GGUF blocks (Q2_0, Q4_0, Q5_0, Q8_0, Q3_K-Q6_K, IQ4_NL, IQ4_XS) to BF16, F16 or F32 rows, one thread a 32-element group, transcribed from llama.cpp's dequantize_row_*: a staging copy for the BLAS GEMMs.
- `src/prefill/gemm.cu`: the prompt chunk's projections: BF16 or F16 weights (dequantized into a reusable scratch when native GGUF) times the activations through hipBLAS / hipBLASLt (tuned solution tables tools/hip/*-hipblaslt-*.txt), an exact path for the HC down projection, MMQ when built.
- `src/prefill/kernels.cu`: the prompt chunk's element-wise and stateful kernels: hyper-connection norm, mix, up-mix and write with the next norm (and the control vector), GDN conv and state walk, PLE conv, SwiGLU and casts.
- `src/prefill/wmma_gemm.cu`: Y = beta Y + X W^T on RDNA3 WMMA intrinsics for FP16 and BF16 operands (gfx11 only; STRATA_WMMA_GFX11 compiles the intrinsics, a device check decides whether they run).
- `src/prefill/gemm_bf16_parity.cu`: a stand-alone parity program for the BF16 GEMM path against a reference; not a library unit.
- `src/prefill/moe_fused.cu`: grouping of the routed (token, k) pairs by expert on the GPU, int8 activation quantization, and the Q2_0 gate / up, SwiGLU and down products on int8 matrix cores.
- `src/prefill/moe_mmq.cu`: expert gathers (one copy launch an expert as its blob arrives), q8_1 quantization of the routed activations and llama.cpp MMQ launches for the pack's types.
- `src/prefill/moe_fused_iq.cu`: the same fused int8 experts for native GGUF expert slices, formats per layer from expert_layout.
- `src/prefill/ggml_cuda_host.cu`: host-side ggml-cuda symbols (device info with GGML's AMD cc encoding, pool, logging) that MMQ and quantize reference, so strata_mmq links without the rest of ggml-cuda.cu.
- `src/core/device.cu`: DeviceArena (one allocation a planner region, bump sub-allocation), a NaN fill kernel, and the runtime device facts and check against STRATA_HIP_ARCHS.
- `src/core/pinned.cu`: pinned and device-mapped host buffers (registration, large pages, aliasing); no device kernel.
- `src/core/remote_expert_opt.cu`: Remote Expert Decode Optimization v1: two kernels that skip rows already returned by the remote sum when copying mapped expert rows.
- `tests/core/dma_batch_parity.cu`: a stand-alone test of the batched DMA copies' ordering (flag order behind queued copies); not a library unit.

## Not done here

No kernel was run or timed on the pod. The compile gate (`gate/compile/run.py --origin strata`, control-node CPU, gfx1151)
compiles the standalone modules; `kernel.strata.prefill_moe_ggml` needs llama.cpp's ggml headers and ggml-cuda sources and is not compiled.
A timing against the strixite modules is its own experiment card.
