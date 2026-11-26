# ds4 modules by strixflerp interface

One row for each module of catalog/ds4 (card ORIGIN-DS4 31224b4ca3), written by `docs/harvest/harvest_ds4.py`.
Base modules hold antirez/ds4 b3f50180ec2c; overlay modules (name prefix glmtp_, k53flash_, k53prefill_, k53perf_, kmultinode_)
hold one branch's added or changed files of that family under `overlay/<slug>/`.

- **tag**: the existing strixflerp interface class (`<family>.<op>`) the module matches, or none. Every module also
  provides its own full interface name (module.toml `[[provides]]`), with a ds4 variant tag.
- **same op in the catalogue**: modules of other origins that provide the tagged class. The same op is not the same operands:
  ds4 kernels take ds4_gpu tensors in ds4's layouts, so each needs an adapter before it can stand in for a strixflerp module.

| module | files | lines | tag | same op in the catalogue |
|---|---:|---:|---|---|
| `kernel.ds4.cuda_backend` | 1 | 36457 | none | - |
| `kernel.ds4.dsv41` | 2 | 1332 | none | - |
| `kernel.ds4.dsv4_vision` | 1 | 444 | `vision.ds4v-ops` | `kernel.lucebox.vision` |
| `kernel.ds4.glm53_vision` | 1 | 607 | `vision.tower-ops` | `kernel.crucible.vision`, `kernel.fastasfuck.vision`, `kernel.strixite.vision`, `kernel.tensorfold.hip_vision_ocr` |
| `kernel.ds4.glmtp_dsv41` | 2 | 950 | none | - |
| `kernel.ds4.glmtp_rocm_attention` | 1 | 1954 | `attn.softmax-gqa` | `kernel.ciru_inkling.attn_decode`, `kernel.ciru_iu4.attn`, `kernel.fastasfuck.attention_gqa`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_attention` and 9 more |
| `kernel.ds4.glmtp_rocm_attention_launch` | 1 | 1760 | none | - |
| `kernel.ds4.glmtp_rocm_backend` | 2 | 350 | none | - |
| `kernel.ds4.glmtp_rocm_common` | 1 | 427 | `embed.lookup` | `kernel.fastasfuck.embedding`, `kernel.gufo.ds4_linear`, `kernel.gufo.qwen_norm_rope`, `kernel.hipfire.embedding`, `kernel.rocmfpx.convert` and 3 more |
| `kernel.ds4.glmtp_rocm_compat` | 1 | 406 | none | - |
| `kernel.ds4.glmtp_rocm_glm` | 1 | 6319 | `mixer.kda` | `kernel.tensorfold.cuda_glm`, `runtime.vllm_strix.pr57979_kda_stride` |
| `kernel.ds4.glmtp_rocm_indexer` | 1 | 1537 | `attn.indexer-topk` | `kernel.ciru_iu4.attn`, `kernel.gufo.ds4_indexer`, `kernel.hipfire.indexer_topk`, `kernel.strata.qsa`, `kernel.strixfold.aux` and 3 more |
| `kernel.ds4.glmtp_rocm_matmul` | 1 | 1405 | `linear.prefill` | `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemm`, `kernel.fastasfuck.linear`, `kernel.flerp.wmma_q4p`, `kernel.gufo.ds4_linear` and 25 more |
| `kernel.ds4.glmtp_rocm_moe` | 1 | 5732 | `moe.experts-decode` | `kernel.ciru_inkling.moe_decode`, `kernel.ciru_iu4.c1`, `kernel.crucible.moe_decode`, `kernel.fastasfuck.linear`, `kernel.flerp.linear_q4p` and 23 more |
| `kernel.ds4.glmtp_rocm_moe_launch` | 1 | 3090 | none | - |
| `kernel.ds4.glmtp_rocm_norm_rope` | 1 | 635 | `norm.rmsnorm` | `kernel.ciru_inkling.glue_decode`, `kernel.fastasfuck.norm`, `kernel.gufo.ds4_norm_rope`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_norm_rope` and 12 more |
| `kernel.ds4.glmtp_rocm_q8` | 1 | 1493 | `linear.decode` | `kernel.ciru_inkling.glue_decode`, `kernel.ciru_inkling.quant_decode`, `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemv`, `kernel.fastasfuck.linear` and 37 more |
| `kernel.ds4.glmtp_rocm_runtime` | 1 | 6500 | none | - |
| `kernel.ds4.glmtp_rocm_tp` | 1 | 1073 | `collective.device-stage` | `kernel.gufo.qfn_kernels`, `kernel.hipfire.tp` |
| `kernel.ds4.glmtp_rocm_unavailable` | 1 | 33 | none | - |
| `kernel.ds4.k53flash_glm53_vision` | 1 | 606 | `vision.tower-ops` | `kernel.crucible.vision`, `kernel.fastasfuck.vision`, `kernel.strixite.vision`, `kernel.tensorfold.hip_vision_ocr` |
| `kernel.ds4.k53flash_mmq_common` | 1 | 149 | none | - |
| `kernel.ds4.k53flash_mmq_ggml_stubs` | 1 | 236 | none | - |
| `kernel.ds4.k53flash_mmq_mmvq` | 1 | 1183 | `linear.decode` | `kernel.ciru_inkling.glue_decode`, `kernel.ciru_inkling.quant_decode`, `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemv`, `kernel.fastasfuck.linear` and 37 more |
| `kernel.ds4.k53flash_mmq_moe` | 3 | 10344 | `moe.experts-decode` | `kernel.ciru_inkling.moe_decode`, `kernel.ciru_iu4.c1`, `kernel.crucible.moe_decode`, `kernel.fastasfuck.linear`, `kernel.flerp.linear_q4p` and 23 more |
| `kernel.ds4.k53flash_mmq_repack` | 2 | 1252 | `linear.row-layout` | `kernel.fastasfuck.common`, `kernel.strixite.qsplit` |
| `kernel.ds4.k53flash_rocm_attention` | 1 | 2033 | `attn.softmax-gqa` | `kernel.ciru_inkling.attn_decode`, `kernel.ciru_iu4.attn`, `kernel.fastasfuck.attention_gqa`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_attention` and 9 more |
| `kernel.ds4.k53flash_rocm_attention_launch` | 1 | 1733 | none | - |
| `kernel.ds4.k53flash_rocm_backend` | 2 | 404 | none | - |
| `kernel.ds4.k53flash_rocm_compat` | 1 | 412 | none | - |
| `kernel.ds4.k53flash_rocm_glm` | 1 | 5600 | `mixer.kda` | `kernel.tensorfold.cuda_glm`, `runtime.vllm_strix.pr57979_kda_stride` |
| `kernel.ds4.k53flash_rocm_glm_selected_multihead` | 1 | 234 | none | - |
| `kernel.ds4.k53flash_rocm_hipblaslt` | 1 | 137 | none | - |
| `kernel.ds4.k53flash_rocm_indexer` | 1 | 1306 | `attn.indexer-topk` | `kernel.ciru_iu4.attn`, `kernel.gufo.ds4_indexer`, `kernel.hipfire.indexer_topk`, `kernel.strata.qsa`, `kernel.strixfold.aux` and 3 more |
| `kernel.ds4.k53flash_rocm_matmul` | 1 | 1269 | `linear.prefill` | `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemm`, `kernel.fastasfuck.linear`, `kernel.flerp.wmma_q4p`, `kernel.gufo.ds4_linear` and 25 more |
| `kernel.ds4.k53flash_rocm_moe` | 1 | 5609 | `moe.experts-decode` | `kernel.ciru_inkling.moe_decode`, `kernel.ciru_iu4.c1`, `kernel.crucible.moe_decode`, `kernel.fastasfuck.linear`, `kernel.flerp.linear_q4p` and 23 more |
| `kernel.ds4.k53flash_rocm_moe_launch` | 1 | 2842 | none | - |
| `kernel.ds4.k53flash_rocm_q8` | 1 | 1897 | `linear.decode` | `kernel.ciru_inkling.glue_decode`, `kernel.ciru_inkling.quant_decode`, `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemv`, `kernel.fastasfuck.linear` and 37 more |
| `kernel.ds4.k53flash_rocm_router` | 1 | 258 | `moe.router` | `kernel.ciru_inkling.glue_decode`, `kernel.crucible.ops`, `kernel.fastasfuck.moe_router`, `kernel.flerp.moe_router_gufo`, `kernel.flerp.moe_router_hipfire` and 13 more |
| `kernel.ds4.k53flash_rocm_runtime` | 1 | 6460 | none | - |
| `kernel.ds4.k53flash_rocm_unavailable` | 1 | 39 | none | - |
| `kernel.ds4.k53perf_dsv41` | 2 | 921 | none | - |
| `kernel.ds4.k53perf_dsv4_vision` | 1 | 572 | `vision.ds4v-ops` | `kernel.lucebox.vision` |
| `kernel.ds4.k53perf_glm53_vision` | 1 | 661 | `vision.tower-ops` | `kernel.crucible.vision`, `kernel.fastasfuck.vision`, `kernel.strixite.vision`, `kernel.tensorfold.hip_vision_ocr` |
| `kernel.ds4.k53perf_mmq_common` | 1 | 149 | none | - |
| `kernel.ds4.k53perf_mmq_mmvq` | 1 | 1183 | `linear.decode` | `kernel.ciru_inkling.glue_decode`, `kernel.ciru_inkling.quant_decode`, `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemv`, `kernel.fastasfuck.linear` and 37 more |
| `kernel.ds4.k53perf_mmq_moe` | 3 | 10712 | `moe.experts-decode` | `kernel.ciru_inkling.moe_decode`, `kernel.ciru_iu4.c1`, `kernel.crucible.moe_decode`, `kernel.fastasfuck.linear`, `kernel.flerp.linear_q4p` and 23 more |
| `kernel.ds4.k53perf_mmq_repack` | 1 | 1159 | `linear.row-layout` | `kernel.fastasfuck.common`, `kernel.strixite.qsplit` |
| `kernel.ds4.k53perf_rocm_attention` | 1 | 2035 | `attn.softmax-gqa` | `kernel.ciru_inkling.attn_decode`, `kernel.ciru_iu4.attn`, `kernel.fastasfuck.attention_gqa`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_attention` and 9 more |
| `kernel.ds4.k53perf_rocm_attention_gufo` | 1 | 360 | `attn.sparse-sink` | `kernel.crucible.qsa`, `kernel.gufo.ds4_attention`, `kernel.hipfire.ds4_sparse_attn`, `kernel.radiance.halo_glm_family`, `kernel.strixfold.sparse_attn` and 5 more |
| `kernel.ds4.k53perf_rocm_attention_launch` | 1 | 1986 | none | - |
| `kernel.ds4.k53perf_rocm_backend` | 2 | 404 | none | - |
| `kernel.ds4.k53perf_rocm_compat` | 1 | 413 | none | - |
| `kernel.ds4.k53perf_rocm_current_api_compat` | 1 | 419 | none | - |
| `kernel.ds4.k53perf_rocm_glm` | 1 | 5549 | `mixer.kda` | `kernel.tensorfold.cuda_glm`, `runtime.vllm_strix.pr57979_kda_stride` |
| `kernel.ds4.k53perf_rocm_halo_attention` | 7 | 519 | `attn.prefill-causal` | `kernel.ciru_inkling.attn_prefill`, `kernel.flerp.attn_wmma_gqa`, `kernel.libstrix.attn`, `kernel.lucebox.attn_prefill`, `kernel.radiance.halo_attn` |
| `kernel.ds4.k53perf_rocm_halo_direct_qk` | 1 | 391 | `attn.prefill-causal` | `kernel.ciru_inkling.attn_prefill`, `kernel.flerp.attn_wmma_gqa`, `kernel.libstrix.attn`, `kernel.lucebox.attn_prefill`, `kernel.radiance.halo_attn` |
| `kernel.ds4.k53perf_rocm_halo_hc` | 1 | 40 | `hc.mix-prefill` | `kernel.ciru_iu4.mmb`, `kernel.crucible.hc`, `kernel.fastasfuck.hc_wmma`, `kernel.libstrix.gemm`, `kernel.strata.prefill` and 4 more |
| `kernel.ds4.k53perf_rocm_halo_indexer` | 1 | 106 | `attn.indexer-score` | `kernel.gufo.ds4_indexer`, `kernel.hipfire.indexer_topk`, `kernel.lucebox.indexer`, `kernel.strata.qsa`, `kernel.strixfold.indexer_score` and 8 more |
| `kernel.ds4.k53perf_rocm_halo_moe` | 14 | 1279 | `moe.experts-prefill` | `kernel.ciru_inkling.moe_prefill`, `kernel.ciru_iu4.c1`, `kernel.ciru_iu4.mmb`, `kernel.crucible.moe_prefill`, `kernel.fastasfuck.moe_grouped` and 18 more |
| `kernel.ds4.k53perf_rocm_halo_output_a` | 8 | 2242 | `linear.prefill` | `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemm`, `kernel.fastasfuck.linear`, `kernel.flerp.wmma_q4p`, `kernel.gufo.ds4_linear` and 25 more |
| `kernel.ds4.k53perf_rocm_halo_output_a_uncached` | 2 | 93 | `linear.prefill` | `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemm`, `kernel.fastasfuck.linear`, `kernel.flerp.wmma_q4p`, `kernel.gufo.ds4_linear` and 25 more |
| `kernel.ds4.k53perf_rocm_halo_output_b` | 3 | 5125 | `linear.prefill` | `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemm`, `kernel.fastasfuck.linear`, `kernel.flerp.wmma_q4p`, `kernel.gufo.ds4_linear` and 25 more |
| `kernel.ds4.k53perf_rocm_halo_q2` | 16 | 4352 | `linear.prefill` | `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemm`, `kernel.fastasfuck.linear`, `kernel.flerp.wmma_q4p`, `kernel.gufo.ds4_linear` and 25 more |
| `kernel.ds4.k53perf_rocm_halo_q8` | 1 | 241 | `linear.prefill` | `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemm`, `kernel.fastasfuck.linear`, `kernel.flerp.wmma_q4p`, `kernel.gufo.ds4_linear` and 25 more |
| `kernel.ds4.k53perf_rocm_halo_shapes` | 1 | 69 | none | - |
| `kernel.ds4.k53perf_rocm_halo_shared_down` | 1 | 111 | `linear.prefill` | `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemm`, `kernel.fastasfuck.linear`, `kernel.flerp.wmma_q4p`, `kernel.gufo.ds4_linear` and 25 more |
| `kernel.ds4.k53perf_rocm_halo_shared_gu` | 10 | 4025 | `linear.prefill` | `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemm`, `kernel.fastasfuck.linear`, `kernel.flerp.wmma_q4p`, `kernel.gufo.ds4_linear` and 25 more |
| `kernel.ds4.k53perf_rocm_halo_static_query` | 1 | 147 | `attn.softmax-gqa` | `kernel.ciru_inkling.attn_decode`, `kernel.ciru_iu4.attn`, `kernel.fastasfuck.attention_gqa`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_attention` and 9 more |
| `kernel.ds4.k53perf_rocm_hc_sgemm` | 1 | 105 | `hc.mix-prefill` | `kernel.ciru_iu4.mmb`, `kernel.crucible.hc`, `kernel.fastasfuck.hc_wmma`, `kernel.libstrix.gemm`, `kernel.strata.prefill` and 4 more |
| `kernel.ds4.k53perf_rocm_hipblaslt` | 1 | 327 | none | - |
| `kernel.ds4.k53perf_rocm_indexer` | 1 | 1322 | `attn.indexer-topk` | `kernel.ciru_iu4.attn`, `kernel.gufo.ds4_indexer`, `kernel.hipfire.indexer_topk`, `kernel.strata.qsa`, `kernel.strixfold.aux` and 3 more |
| `kernel.ds4.k53perf_rocm_matmul` | 1 | 1407 | `linear.prefill` | `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemm`, `kernel.fastasfuck.linear`, `kernel.flerp.wmma_q4p`, `kernel.gufo.ds4_linear` and 25 more |
| `kernel.ds4.k53perf_rocm_moe` | 1 | 5968 | `moe.experts-decode` | `kernel.ciru_inkling.moe_decode`, `kernel.ciru_iu4.c1`, `kernel.crucible.moe_decode`, `kernel.fastasfuck.linear`, `kernel.flerp.linear_q4p` and 23 more |
| `kernel.ds4.k53perf_rocm_moe_launch` | 1 | 3136 | none | - |
| `kernel.ds4.k53perf_rocm_norm_rope` | 1 | 621 | `norm.rmsnorm` | `kernel.ciru_inkling.glue_decode`, `kernel.fastasfuck.norm`, `kernel.gufo.ds4_norm_rope`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_norm_rope` and 12 more |
| `kernel.ds4.k53perf_rocm_q8` | 1 | 1506 | `linear.decode` | `kernel.ciru_inkling.glue_decode`, `kernel.ciru_inkling.quant_decode`, `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemv`, `kernel.fastasfuck.linear` and 37 more |
| `kernel.ds4.k53perf_rocm_qwen4` | 1 | 2614 | `mixer.gdn` | `kernel.ciru_iu4.gdn`, `kernel.crucible.gdn`, `kernel.fastasfuck.gdn`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_gdn` and 9 more |
| `kernel.ds4.k53perf_rocm_qwen4_attention_wmma` | 1 | 605 | `attn.prefill-causal` | `kernel.ciru_inkling.attn_prefill`, `kernel.flerp.attn_wmma_gqa`, `kernel.libstrix.attn`, `kernel.lucebox.attn_prefill`, `kernel.radiance.halo_attn` |
| `kernel.ds4.k53perf_rocm_router` | 1 | 281 | `moe.router` | `kernel.ciru_inkling.glue_decode`, `kernel.crucible.ops`, `kernel.fastasfuck.moe_router`, `kernel.flerp.moe_router_gufo`, `kernel.flerp.moe_router_hipfire` and 13 more |
| `kernel.ds4.k53perf_rocm_runtime` | 1 | 7173 | none | - |
| `kernel.ds4.k53perf_rocm_tp` | 1 | 324 | `collective.device-stage` | `kernel.gufo.qfn_kernels`, `kernel.hipfire.tp` |
| `kernel.ds4.k53perf_rocm_unavailable` | 1 | 39 | none | - |
| `kernel.ds4.k53perf_rocm_v41` | 1 | 2096 | none | - |
| `kernel.ds4.k53perf_rocm_v41_gate_up` | 9 | 693 | `moe.experts-prefill` | `kernel.ciru_inkling.moe_prefill`, `kernel.ciru_iu4.c1`, `kernel.ciru_iu4.mmb`, `kernel.crucible.moe_prefill`, `kernel.fastasfuck.moe_grouped` and 18 more |
| `kernel.ds4.k53prefill_glm53_vision` | 1 | 606 | `vision.tower-ops` | `kernel.crucible.vision`, `kernel.fastasfuck.vision`, `kernel.strixite.vision`, `kernel.tensorfold.hip_vision_ocr` |
| `kernel.ds4.k53prefill_mmq_common` | 2 | 456 | none | - |
| `kernel.ds4.k53prefill_mmq_ggml_stubs` | 1 | 177 | none | - |
| `kernel.ds4.k53prefill_mmq_mmvq` | 1 | 1183 | `linear.decode` | `kernel.ciru_inkling.glue_decode`, `kernel.ciru_inkling.quant_decode`, `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemv`, `kernel.fastasfuck.linear` and 37 more |
| `kernel.ds4.k53prefill_mmq_moe` | 3 | 10320 | `moe.experts-decode` | `kernel.ciru_inkling.moe_decode`, `kernel.ciru_iu4.c1`, `kernel.crucible.moe_decode`, `kernel.fastasfuck.linear`, `kernel.flerp.linear_q4p` and 23 more |
| `kernel.ds4.k53prefill_mmq_repack` | 2 | 1252 | `linear.row-layout` | `kernel.fastasfuck.common`, `kernel.strixite.qsplit` |
| `kernel.ds4.k53prefill_rocm_attention` | 1 | 2033 | `attn.softmax-gqa` | `kernel.ciru_inkling.attn_decode`, `kernel.ciru_iu4.attn`, `kernel.fastasfuck.attention_gqa`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_attention` and 9 more |
| `kernel.ds4.k53prefill_rocm_attention_launch` | 1 | 1733 | none | - |
| `kernel.ds4.k53prefill_rocm_backend` | 2 | 401 | none | - |
| `kernel.ds4.k53prefill_rocm_compat` | 1 | 412 | none | - |
| `kernel.ds4.k53prefill_rocm_glm` | 1 | 5594 | `mixer.kda` | `kernel.tensorfold.cuda_glm`, `runtime.vllm_strix.pr57979_kda_stride` |
| `kernel.ds4.k53prefill_rocm_glm_selected_multihead` | 1 | 234 | none | - |
| `kernel.ds4.k53prefill_rocm_hipblaslt` | 1 | 137 | none | - |
| `kernel.ds4.k53prefill_rocm_indexer` | 1 | 1306 | `attn.indexer-topk` | `kernel.ciru_iu4.attn`, `kernel.gufo.ds4_indexer`, `kernel.hipfire.indexer_topk`, `kernel.strata.qsa`, `kernel.strixfold.aux` and 3 more |
| `kernel.ds4.k53prefill_rocm_matmul` | 1 | 1269 | `linear.prefill` | `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemm`, `kernel.fastasfuck.linear`, `kernel.flerp.wmma_q4p`, `kernel.gufo.ds4_linear` and 25 more |
| `kernel.ds4.k53prefill_rocm_moe` | 1 | 5609 | `moe.experts-decode` | `kernel.ciru_inkling.moe_decode`, `kernel.ciru_iu4.c1`, `kernel.crucible.moe_decode`, `kernel.fastasfuck.linear`, `kernel.flerp.linear_q4p` and 23 more |
| `kernel.ds4.k53prefill_rocm_moe_launch` | 1 | 2842 | none | - |
| `kernel.ds4.k53prefill_rocm_q8` | 1 | 1897 | `linear.decode` | `kernel.ciru_inkling.glue_decode`, `kernel.ciru_inkling.quant_decode`, `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemv`, `kernel.fastasfuck.linear` and 37 more |
| `kernel.ds4.k53prefill_rocm_router` | 1 | 258 | `moe.router` | `kernel.ciru_inkling.glue_decode`, `kernel.crucible.ops`, `kernel.fastasfuck.moe_router`, `kernel.flerp.moe_router_gufo`, `kernel.flerp.moe_router_hipfire` and 13 more |
| `kernel.ds4.k53prefill_rocm_runtime` | 1 | 6460 | none | - |
| `kernel.ds4.k53prefill_rocm_unavailable` | 1 | 39 | none | - |
| `kernel.ds4.kmultinode_rocm_attention` | 1 | 1477 | `attn.softmax-gqa` | `kernel.ciru_inkling.attn_decode`, `kernel.ciru_iu4.attn`, `kernel.fastasfuck.attention_gqa`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_attention` and 9 more |
| `kernel.ds4.kmultinode_rocm_attention_launch` | 1 | 1358 | none | - |
| `kernel.ds4.kmultinode_rocm_backend` | 2 | 270 | none | - |
| `kernel.ds4.kmultinode_rocm_common` | 1 | 377 | `embed.lookup` | `kernel.fastasfuck.embedding`, `kernel.gufo.ds4_linear`, `kernel.gufo.qwen_norm_rope`, `kernel.hipfire.embedding`, `kernel.rocmfpx.convert` and 3 more |
| `kernel.ds4.kmultinode_rocm_compressor` | 1 | 557 | `kvcache.compress` | `kernel.gufo.ds4_attention`, `kernel.hipfire.ds4_sparse_attn` |
| `kernel.ds4.kmultinode_rocm_current_api_compat` | 1 | 178 | none | - |
| `kernel.ds4.kmultinode_rocm_embedding_launch` | 1 | 113 | none | - |
| `kernel.ds4.kmultinode_rocm_hc_output_launch` | 1 | 437 | none | - |
| `kernel.ds4.kmultinode_rocm_hipblaslt` | 1 | 137 | none | - |
| `kernel.ds4.kmultinode_rocm_indexer` | 1 | 1047 | `attn.indexer-topk` | `kernel.ciru_iu4.attn`, `kernel.gufo.ds4_indexer`, `kernel.hipfire.indexer_topk`, `kernel.strata.qsa`, `kernel.strixfold.aux` and 3 more |
| `kernel.ds4.kmultinode_rocm_matmul` | 1 | 791 | `linear.prefill` | `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemm`, `kernel.fastasfuck.linear`, `kernel.flerp.wmma_q4p`, `kernel.gufo.ds4_linear` and 25 more |
| `kernel.ds4.kmultinode_rocm_misc_launch` | 1 | 31 | none | - |
| `kernel.ds4.kmultinode_rocm_moe` | 1 | 4022 | `moe.experts-decode` | `kernel.ciru_inkling.moe_decode`, `kernel.ciru_iu4.c1`, `kernel.crucible.moe_decode`, `kernel.fastasfuck.linear`, `kernel.flerp.linear_q4p` and 23 more |
| `kernel.ds4.kmultinode_rocm_moe_launch` | 1 | 1371 | none | - |
| `kernel.ds4.kmultinode_rocm_norm_rope` | 1 | 591 | `norm.rmsnorm` | `kernel.ciru_inkling.glue_decode`, `kernel.fastasfuck.norm`, `kernel.gufo.ds4_norm_rope`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_norm_rope` and 12 more |
| `kernel.ds4.kmultinode_rocm_output` | 1 | 77 | `act.swiglu` | `kernel.ciru_inkling.glue_decode`, `kernel.fastasfuck.swiglu`, `kernel.gufo.qfn_kernels`, `kernel.strixfold.dense_bf16`, `kernel.strixfold_tp4.dense_bf16` and 1 more |
| `kernel.ds4.kmultinode_rocm_q8` | 1 | 1746 | `linear.decode` | `kernel.ciru_inkling.glue_decode`, `kernel.ciru_inkling.quant_decode`, `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemv`, `kernel.fastasfuck.linear` and 37 more |
| `kernel.ds4.kmultinode_rocm_router` | 1 | 199 | `moe.router` | `kernel.ciru_inkling.glue_decode`, `kernel.crucible.ops`, `kernel.fastasfuck.moe_router`, `kernel.flerp.moe_router_gufo`, `kernel.flerp.moe_router_hipfire` and 13 more |
| `kernel.ds4.kmultinode_rocm_runtime` | 1 | 1700 | none | - |
| `kernel.ds4.kmultinode_rocm_shared_expert` | 1 | 310 | none | - |
| `kernel.ds4.mmq_common` | 11 | 6782 | none | - |
| `kernel.ds4.mmq_d2r` | 3 | 3505 | none | - |
| `kernel.ds4.mmq_ggml_stubs` | 3 | 603 | none | - |
| `kernel.ds4.mmq_mmid` | 2 | 348 | none | - |
| `kernel.ds4.mmq_mmvq` | 2 | 1220 | `linear.decode` | `kernel.ciru_inkling.glue_decode`, `kernel.ciru_inkling.quant_decode`, `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemv`, `kernel.fastasfuck.linear` and 37 more |
| `kernel.ds4.mmq_moe` | 3 | 11103 | `moe.experts-decode` | `kernel.ciru_inkling.moe_decode`, `kernel.ciru_iu4.c1`, `kernel.crucible.moe_decode`, `kernel.fastasfuck.linear`, `kernel.flerp.linear_q4p` and 23 more |
| `kernel.ds4.mmq_quantize` | 2 | 484 | `quant.activation` | `kernel.radiance.halo_quant_act`, `kernel.strata.gguf_gemv` |
| `kernel.ds4.mmq_repack` | 2 | 1293 | `linear.row-layout` | `kernel.fastasfuck.common`, `kernel.strixite.qsplit` |
| `kernel.ds4.qwen4_cuda` | 1 | 2356 | `mixer.gdn` | `kernel.ciru_iu4.gdn`, `kernel.crucible.gdn`, `kernel.fastasfuck.gdn`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_gdn` and 9 more |
| `kernel.ds4.rocm_attention` | 1 | 1947 | `attn.softmax-gqa` | `kernel.ciru_inkling.attn_decode`, `kernel.ciru_iu4.attn`, `kernel.fastasfuck.attention_gqa`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_attention` and 9 more |
| `kernel.ds4.rocm_attention_launch` | 1 | 1744 | none | - |
| `kernel.ds4.rocm_backend` | 3 | 485 | none | - |
| `kernel.ds4.rocm_common` | 1 | 417 | `embed.lookup` | `kernel.fastasfuck.embedding`, `kernel.gufo.ds4_linear`, `kernel.gufo.qwen_norm_rope`, `kernel.hipfire.embedding`, `kernel.rocmfpx.convert` and 3 more |
| `kernel.ds4.rocm_compat` | 1 | 405 | none | - |
| `kernel.ds4.rocm_compressor` | 1 | 564 | `kvcache.compress` | `kernel.gufo.ds4_attention`, `kernel.hipfire.ds4_sparse_attn` |
| `kernel.ds4.rocm_current_api_compat` | 1 | 322 | none | - |
| `kernel.ds4.rocm_deepseek4_vision` | 1 | 324 | `vision.ds4v-ops` | `kernel.lucebox.vision` |
| `kernel.ds4.rocm_embedding_launch` | 1 | 151 | none | - |
| `kernel.ds4.rocm_fp8_kv` | 1 | 40 | `kvcache.store` | `kernel.ciru_inkling.attn_decode`, `kernel.ciru_inkling.attn_prefill`, `kernel.gufo.ds4_attention`, `kernel.hipfire.kv_cache`, `kernel.rocmfpx.convert` and 3 more |
| `kernel.ds4.rocm_fp8_kv_launch` | 1 | 9 | none | - |
| `kernel.ds4.rocm_glm` | 1 | 5312 | `mixer.kda` | `kernel.tensorfold.cuda_glm`, `runtime.vllm_strix.pr57979_kda_stride` |
| `kernel.ds4.rocm_hc` | 1 | 373 | `hc.mix` | `kernel.ciru_iu4.mmb`, `kernel.crucible.hc`, `kernel.fastasfuck.hc`, `kernel.fastasfuck.hc_wmma`, `kernel.gufo.ds4_hc` and 14 more |
| `kernel.ds4.rocm_hc_output_launch` | 1 | 443 | none | - |
| `kernel.ds4.rocm_hipblaslt` | 1 | 313 | none | - |
| `kernel.ds4.rocm_indexer` | 1 | 1308 | `attn.indexer-topk` | `kernel.ciru_iu4.attn`, `kernel.gufo.ds4_indexer`, `kernel.hipfire.indexer_topk`, `kernel.strata.qsa`, `kernel.strixfold.aux` and 3 more |
| `kernel.ds4.rocm_matmul` | 1 | 1300 | `linear.prefill` | `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemm`, `kernel.fastasfuck.linear`, `kernel.flerp.wmma_q4p`, `kernel.gufo.ds4_linear` and 25 more |
| `kernel.ds4.rocm_misc_launch` | 1 | 82 | none | - |
| `kernel.ds4.rocm_moe` | 1 | 5468 | `moe.experts-decode` | `kernel.ciru_inkling.moe_decode`, `kernel.ciru_iu4.c1`, `kernel.crucible.moe_decode`, `kernel.fastasfuck.linear`, `kernel.flerp.linear_q4p` and 23 more |
| `kernel.ds4.rocm_moe_launch` | 1 | 2831 | none | - |
| `kernel.ds4.rocm_norm_rope` | 1 | 612 | `norm.rmsnorm` | `kernel.ciru_inkling.glue_decode`, `kernel.fastasfuck.norm`, `kernel.gufo.ds4_norm_rope`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_norm_rope` and 12 more |
| `kernel.ds4.rocm_output` | 1 | 87 | `act.swiglu` | `kernel.ciru_inkling.glue_decode`, `kernel.fastasfuck.swiglu`, `kernel.gufo.qfn_kernels`, `kernel.strixfold.dense_bf16`, `kernel.strixfold_tp4.dense_bf16` and 1 more |
| `kernel.ds4.rocm_q8` | 1 | 1448 | `linear.decode` | `kernel.ciru_inkling.glue_decode`, `kernel.ciru_inkling.quant_decode`, `kernel.ciru_iu4.q5`, `kernel.crucible.exl3_gemv`, `kernel.fastasfuck.linear` and 37 more |
| `kernel.ds4.rocm_router` | 1 | 269 | `moe.router` | `kernel.ciru_inkling.glue_decode`, `kernel.crucible.ops`, `kernel.fastasfuck.moe_router`, `kernel.flerp.moe_router_gufo`, `kernel.flerp.moe_router_hipfire` and 13 more |
| `kernel.ds4.rocm_runtime` | 1 | 6429 | none | - |
| `kernel.ds4.rocm_shared_expert` | 1 | 455 | none | - |
| `kernel.ds4.rocm_unavailable` | 1 | 42 | none | - |
| `runtime.ds4.distributed` | 2 | 8562 | `transport.stage-relay` | `runtime.flerp.stage_relay` |
| `runtime.ds4.dspark` | 1 | 471 | `speculation.draft` | `executor.hipfire.arch_deepseek4`, `executor.hipfire.arch_qwen4`, `kernel.radiance.halo_dflash`, `kernel.tensorfold.hip_draft`, `runtime.crucible.speculation` and 8 more |
| `runtime.ds4.engine` | 2 | 90450 | `executor.model` | `executor.crucible.flashnext`, `executor.flerp.dsv41_zip2`, `executor.flerp.qwen35moe`, `executor.gufo.deepseek_v4_flash`, `executor.gufo.qwen` and 17 more |
| `runtime.ds4.engram` | 2 | 770 | none | - |
| `runtime.ds4.glmtp_distributed` | 2 | 9723 | `transport.stage-relay` | `runtime.flerp.stage_relay` |
| `runtime.ds4.glmtp_dspark` | 1 | 471 | `speculation.draft` | `executor.hipfire.arch_deepseek4`, `executor.hipfire.arch_qwen4`, `kernel.radiance.halo_dflash`, `kernel.tensorfold.hip_draft`, `runtime.crucible.speculation` and 8 more |
| `runtime.ds4.glmtp_engine` | 2 | 90450 | `executor.model` | `executor.crucible.flashnext`, `executor.flerp.dsv41_zip2`, `executor.flerp.qwen35moe`, `executor.gufo.deepseek_v4_flash`, `executor.gufo.qwen` and 17 more |
| `runtime.ds4.glmtp_gpu_api` | 1 | 3772 | none | - |
| `runtime.ds4.glmtp_gpu_tp` | 1 | 102 | `collective.interface` | `runtime.strixfold.coll`, `runtime.strixfold_tp4.coll`, `runtime.tensorfold.hip_comm` |
| `runtime.ds4.glmtp_tp` | 2 | 5879 | `collective.interface` | `runtime.strixfold.coll`, `runtime.strixfold_tp4.coll`, `runtime.tensorfold.hip_comm` |
| `runtime.ds4.gpu_api` | 3 | 4150 | none | - |
| `runtime.ds4.gpu_args` | 2 | 314 | none | - |
| `runtime.ds4.gpu_tp` | 1 | 63 | `collective.interface` | `runtime.strixfold.coll`, `runtime.strixfold_tp4.coll`, `runtime.tensorfold.hip_comm` |
| `runtime.ds4.image` | 2 | 1361 | `vision.preprocess` | `runtime.strixite.vision_preprocess` |
| `runtime.ds4.iris` | 3 | 2625 | `vision.image-decode` | `runtime.strixite.libwebp`, `runtime.strixite.stb` |
| `runtime.ds4.k53flash_engine` | 3 | 75144 | `executor.model` | `executor.crucible.flashnext`, `executor.flerp.dsv41_zip2`, `executor.flerp.qwen35moe`, `executor.gufo.deepseek_v4_flash`, `executor.gufo.qwen` and 17 more |
| `runtime.ds4.k53flash_gpu_api` | 1 | 3194 | none | - |
| `runtime.ds4.k53flash_tp` | 2 | 2715 | `collective.interface` | `runtime.strixfold.coll`, `runtime.strixfold_tp4.coll`, `runtime.tensorfold.hip_comm` |
| `runtime.ds4.k53perf_engine` | 2 | 88632 | `executor.model` | `executor.crucible.flashnext`, `executor.flerp.dsv41_zip2`, `executor.flerp.qwen35moe`, `executor.gufo.deepseek_v4_flash`, `executor.gufo.qwen` and 17 more |
| `runtime.ds4.k53perf_gpu_api` | 1 | 3598 | none | - |
| `runtime.ds4.k53perf_gpu_tp` | 1 | 75 | `collective.interface` | `runtime.strixfold.coll`, `runtime.strixfold_tp4.coll`, `runtime.tensorfold.hip_comm` |
| `runtime.ds4.k53perf_tp` | 2 | 3784 | `collective.interface` | `runtime.strixfold.coll`, `runtime.strixfold_tp4.coll`, `runtime.tensorfold.hip_comm` |
| `runtime.ds4.k53perf_tp_io` | 1 | 75 | none | - |
| `runtime.ds4.k53perf_tp_roce` | 1 | 209 | `transport.verbs` | `runtime.odinlink5.verbs_provider`, `runtime.strixfold.fabric`, `runtime.tensorfold.fabric` |
| `runtime.ds4.k53prefill_engine` | 3 | 74355 | `executor.model` | `executor.crucible.flashnext`, `executor.flerp.dsv41_zip2`, `executor.flerp.qwen35moe`, `executor.gufo.deepseek_v4_flash`, `executor.gufo.qwen` and 17 more |
| `runtime.ds4.k53prefill_gpu_api` | 1 | 3092 | none | - |
| `runtime.ds4.k53prefill_tp` | 2 | 2715 | `collective.interface` | `runtime.strixfold.coll`, `runtime.strixfold_tp4.coll`, `runtime.tensorfold.hip_comm` |
| `runtime.ds4.kmultinode_distributed` | 1 | 8414 | `transport.stage-relay` | `runtime.flerp.stage_relay` |
| `runtime.ds4.kmultinode_engine` | 3 | 26976 | `executor.model` | `executor.crucible.flashnext`, `executor.flerp.dsv41_zip2`, `executor.flerp.qwen35moe`, `executor.gufo.deepseek_v4_flash`, `executor.gufo.qwen` and 17 more |
| `runtime.ds4.kmultinode_gpu_api` | 1 | 998 | none | - |
| `runtime.ds4.kvstore` | 2 | 1571 | none | - |
| `runtime.ds4.layer_pack` | 2 | 226 | none | - |
| `runtime.ds4.linux_memory` | 1 | 37 | none | - |
| `runtime.ds4.prompt_prefix` | 2 | 298 | `cache.prompt` | `runtime.crucible.conversation_cache`, `runtime.gufo.cache`, `runtime.hipfire.runtime`, `runtime.strixfold.prompt_cache`, `runtime.strixfold_tp4.prompt_cache` and 4 more |
| `runtime.ds4.qwen4_unicode` | 1 | 310 | `text.tokenizer` | `runtime.strixfold.core`, `runtime.strixfold_tp4.core`, `runtime.strixite.tokenizer`, `runtime.tensorfold.core` |
| `runtime.ds4.qwen4_vision` | 1 | 19 | none | - |
| `runtime.ds4.rax` | 3 | 3091 | none | - |
| `runtime.ds4.ssd` | 2 | 253 | `runtime.weight-prefetch` | `kernel.radiance.halo_model_ops` |
| `runtime.ds4.streaming_hotlist` | 2 | 19843 | none | - |
| `runtime.ds4.tp` | 2 | 4310 | `collective.interface` | `runtime.strixfold.coll`, `runtime.strixfold_tp4.coll`, `runtime.tensorfold.hip_comm` |
| `serve.ds4.server` | 2 | 23262 | `serve.engine` | `runtime.strixite.engine`, `runtime.tensorfold.hip_serve`, `serve.crucible.engine_serve` |
| `serve.ds4.web` | 2 | 1487 | none | - |
| `tool.ds4.agent` | 1 | 13635 | none | - |
| `tool.ds4.bench` | 1 | 1062 | none | - |
| `tool.ds4.cli` | 1 | 2438 | none | - |
| `tool.ds4.dir_steering` | 2 | 295 | none | - |
| `tool.ds4.eval` | 3 | 6167 | none | - |
| `tool.ds4.gguf_dsv41` | 5 | 1184 | `format.convert` | `format.flerp.convert_qwen4exp_p16`, `format.strixite.convert_qwen4exp` |
| `tool.ds4.gguf_glm53` | 4 | 2137 | `format.convert` | `format.flerp.convert_qwen4exp_p16`, `format.strixite.convert_qwen4exp` |
| `tool.ds4.gguf_quants` | 4 | 4616 | `format.convert` | `format.flerp.convert_qwen4exp_p16`, `format.strixite.convert_qwen4exp` |
| `tool.ds4.gguf_qwen4` | 6 | 4707 | `format.convert` | `format.flerp.convert_qwen4exp_p16`, `format.strixite.convert_qwen4exp` |
| `tool.ds4.gguf_vision` | 3 | 928 | `tool.vision-extract` | `tool.strixfold_pr6.prepare_vision` |
| `tool.ds4.glmtp_tp_odl_test` | 1 | 203 | none | - |
| `tool.ds4.help` | 2 | 628 | none | - |
| `tool.ds4.linenoise` | 2 | 2965 | none | - |

217 modules (base 73, glm_tp_rocm 23, kyuz0_glm53_flash 24, kyuz0_glm53_perf 50, kyuz0_glm53_prefill 24, kyuz0_rocm_multi_node 23); 125 carry a tag, 92 are none.

## Overlays

- `glm_tp_rocm` (glmtp_*): davidcanar/ds4 glm-tp-rocm 8ebf672 (antirez/ds4 PR #1024, stacked on #813, #835 and #1018). 2 file(s) added, 78 changed, 30 base file(s) absent (docs/harvest/data/ds4/glm_tp_rocm.tsv). GLM-5.3-Flash tensor parallel across two ROCm machines: the KDA heads split between the ranks, the routed experts split 50/50, and the TP slab in host-mapped pinned memory that GPU kernels and the CPU transport thread share on Strix Halo (rocm/ds4_rocm_tp.cuh); the branch head is "GLM TP: keep the transport slab unified on ROCm builds".
- `kyuz0_glm53_flash` (k53flash_*): kyuz0/ds4 perf/rocm-gfx1151-glm53-flash 4bdbcc2. 1 file(s) added, 100 changed, 1385 base file(s) absent (docs/harvest/data/ds4/kyuz0_glm53_flash.tsv). kyuz0's gfx1151 GLM-5.3-Flash decode branch: on-the-fly WMMA Q8_0 batch GEMMs and partial HC expand (ds4_rocm_q8.cuh), the GLM dense-compact causal attention and row-wise Q8 projections (ds4_rocm_glm.cuh) and a stand-alone selected multi-head attention kernel (rocm/ds4_rocm_glm_selected_multihead.cu); the tree is a fork from before the base's V4.1 and Qwen work, so its base-diff also shows files the base changed later.
- `kyuz0_glm53_prefill` (k53prefill_*): kyuz0/ds4 perf/rocm-gfx1151-glm53-flash-prefill-restoration 4fef372. 1 file(s) added, 100 changed, 1710 base file(s) absent (docs/harvest/data/ds4/kyuz0_glm53_prefill.tsv). the glm53-flash branch with the GLM-5.3-Flash prefill path restored and accelerated (head commit "rocm: accelerate GLM 5.3 Flash prefill"); same fork point as kyuz0_glm53_flash.
- `kyuz0_glm53_perf` (k53perf_*): kyuz0/ds4 perf/rocm-gfx1151-glm53-performance f46631e. 111 file(s) added, 114 changed, 66 base file(s) absent (docs/harvest/data/ds4/kyuz0_glm53_perf.tsv). kyuz0's later gfx1151 branch: the Halo kernels imported from MadRocK-AI/ds4 551219f (frozen WMMA GEMMs for the attention output projections and shared expert, IQ2 S9 MoE producers, Q2 and Q8 panels, direct-query attention), DeepSeek V4.1 ROCm primitives, Qwen3.8 Flash Next on gfx1151, a gufo-derived DSV4 attention, and two-machine ROCm tensor parallel over TCP or RoCE with its own slot-generation queue (docs/CLUSTERING_ROCM.md); head commit "rocm: avoid low-confidence GLM MTP verification on gfx1151".
- `kyuz0_rocm_multi_node` (kmultinode_*): kyuz0/ds4 rocm-multi-node 89c1ea9 (head commit by antirez, "Fix ROCm MTP startup residency"). 13 file(s) added, 79 changed, 3380 base file(s) absent (docs/harvest/data/ds4/kyuz0_rocm_multi_node.tsv). an early ROCm multi-node line of ds4, five months older than the base: the ROCm backend before it grew V4.1, GLM-5.3 and Qwen kernels, with the pipeline-parallel distributed runtime; kept as the record of where the ROCm multi-node work started.

## Not done here

No kernel was run or timed on the pod. The compile gate (`gate/compile/run.py --origin ds4`, control-node CPU, gfx1151) compiles the standalone hip-c++ modules; host C, CUDA and Python modules are not compiled.
