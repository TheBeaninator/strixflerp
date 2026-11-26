# CIRU IU4 kernels by strixflerp interface

One row for each file of `ggml/src/ggml-cuda` that ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4 at 47cb923 adds (A) or changes (M)
against its base pwilkin/llama.cpp f5daaa3c (`docs/harvest/data/ciru_iu4/base-diff.tsv`). Card ORIGIN-CIRU-IU4 314b9e37c3.
Written by `docs/harvest/harvest_ciru_iu4.py` from its `FAMILIES` table; read from source, nothing here was run.

- **tag**: the strixflerp interface class the file matches, `<family>.<op>/<weight format>/<dtype>`, or none. The module's
  `[[provides]]` entry carries the same name with a `.ciru-*` variant tag.
- **same op in the catalogue**: kernel modules of other origins that provide that `<family>.<op>`. The same op is not the
  same call: every CIRU kernel is reached through ggml-cuda.cu's dispatch of a ggml graph node, so a row needs an adapter
  before it stands in for a strixflerp module.

| file | A/M | lines | module | new device kernels | tag | same op in the catalogue |
|---|---|---:|---|---|---|---|
| `c1.cu` | A | 354 | `kernel.ciru_iu4.c1` | `c1_pack_q8_runtime`, `c1_vector_runtime`, `c1_mat_id_reference` | `moe.experts-decode/c1-g16/f32` | `kernel.ciru_inkling.moe_decode`, `kernel.crucible.moe_decode`, `kernel.fastasfuck.linear`, `kernel.flerp.linear_q4p`, `kernel.flerp.moe_kquant`, `kernel.flerp.moe_sg`, `kernel.gufo.ds4_moe`, `kernel.gufo.qfn_mmq`, `kernel.hipfire.moe`, `kernel.hipfire.pm_gfx1151`, `kernel.kestrel.amd_experts`, `kernel.kestrel.ds4`, `kernel.kestrel.exl3`, `kernel.lucebox.moe`, `kernel.radiance.halo_moe`, `kernel.strix_trellis.exl3_tp_moe`, `kernel.strixfold.exl3`, `kernel.strixite.linear_q4`, `kernel.tensorfold.cuda_nemotron`, `kernel.tensorfold.hip_ops` |
| `c1.cuh` | A | 8 | `kernel.ciru_iu4.c1` | - | none | - |
| `c1-grouped.cuh` | A | 287 | `kernel.ciru_iu4.c1` | `project`, `route_count`, `route_describe`, `route_scatter` | `moe.experts-prefill/c1-g16/f32` | `kernel.ciru_inkling.moe_prefill`, `kernel.crucible.moe_prefill`, `kernel.fastasfuck.moe_grouped`, `kernel.flerp.moe_grouped_p`, `kernel.flerp.moe_grouped_pf`, `kernel.flerp.moe_kgrouped`, `kernel.gufo.ds4_moe`, `kernel.hipfire.moe`, `kernel.kestrel.prefill_moe`, `kernel.libstrix.moe`, `kernel.lucebox.moe`, `kernel.radiance.halo_moe`, `kernel.strixfold.exl3`, `kernel.strixfold.exl3_pf_experts`, `kernel.strixite.moe_grouped`, `kernel.tensorfold.cuda_exl3`, `kernel.tensorfold.cuda_experts` |
| `c1-vector-r4.cuh` | A | 214 | `kernel.ciru_iu4.c1` | `c1_vector_r4_runtime`, `c1_vector_r4_lut_runtime` | `moe.experts-decode/c1-g16/f32` | `kernel.ciru_inkling.moe_decode`, `kernel.crucible.moe_decode`, `kernel.fastasfuck.linear`, `kernel.flerp.linear_q4p`, `kernel.flerp.moe_kquant`, `kernel.flerp.moe_sg`, `kernel.gufo.ds4_moe`, `kernel.gufo.qfn_mmq`, `kernel.hipfire.moe`, `kernel.hipfire.pm_gfx1151`, `kernel.kestrel.amd_experts`, `kernel.kestrel.ds4`, `kernel.kestrel.exl3`, `kernel.lucebox.moe`, `kernel.radiance.halo_moe`, `kernel.strix_trellis.exl3_tp_moe`, `kernel.strixfold.exl3`, `kernel.strixite.linear_q4`, `kernel.tensorfold.cuda_nemotron`, `kernel.tensorfold.hip_ops` |
| `mmb.cu` | M | 1543 | `kernel.ciru_iu4.mmb` | `mmb_routed_c1_kernel`, `mmb_routed_c1_glu_kernel`, `mmb_f32_halo_kernel`, `mmb_cvt_bf16_f16` | `moe.experts-prefill/c1-g16/bf16` | `kernel.ciru_inkling.moe_prefill`, `kernel.crucible.moe_prefill`, `kernel.fastasfuck.moe_grouped`, `kernel.flerp.moe_grouped_p`, `kernel.flerp.moe_grouped_pf`, `kernel.flerp.moe_kgrouped`, `kernel.gufo.ds4_moe`, `kernel.hipfire.moe`, `kernel.kestrel.prefill_moe`, `kernel.libstrix.moe`, `kernel.lucebox.moe`, `kernel.radiance.halo_moe`, `kernel.strixfold.exl3`, `kernel.strixfold.exl3_pf_experts`, `kernel.strixite.moe_grouped`, `kernel.tensorfold.cuda_exl3`, `kernel.tensorfold.cuda_experts` |
| `mmb.cuh` | M | 28 | `kernel.ciru_iu4.mmb` | - | none | - |
| `hc_q8_pair.cuh` | A | 13 | `kernel.ciru_iu4.mmb` | - | none | - |
| `hc_q8_pair_kernel.cuh` | A | 217 | `kernel.ciru_iu4.mmb` | `hc_q8_pair_kernel` | `hc.mix-prefill/gguf-q8_0/f32` | `kernel.crucible.hc`, `kernel.fastasfuck.hc_wmma`, `kernel.libstrix.gemm`, `kernel.strixfold.mhc_mix`, `kernel.strixfold.triton_kit`, `kernel.strixite.hc_wmma` |
| `moe-weighted-reduction.cu` | M | 256 | `kernel.ciru_iu4.mmb` | `moe_weighted_reduction_bf16_v4_merge` | `moe.combine/-/bf16` | `kernel.ciru_inkling.moe_prefill`, `kernel.gufo.qfn_kernels`, `kernel.hipfire.moe`, `kernel.lucebox.moe`, `kernel.radiance.halo_moe` |
| `q5_short.cu` | A | 266 | `kernel.ciru_iu4.q5` | `q5_short_f32_t4_kernel`, `q5_short_f32_t5_kernel` | `linear.verify/gguf-q5k/f32` | `kernel.gufo.qwen_linear_decode` |
| `q5_short.cuh` | A | 5 | `kernel.ciru_iu4.q5` | - | none | - |
| `q5_wave64.cu` | A | 173 | `kernel.ciru_iu4.q5` | `pack_q8_and_sums`, `zero_q8_tail` | `linear.prefill/gguf-q5k/f32` | `kernel.crucible.exl3_gemm`, `kernel.fastasfuck.linear`, `kernel.flerp.wmma_q4p`, `kernel.gufo.ds4_linear`, `kernel.gufo.ds4_mmq`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qfn_mmq`, `kernel.gufo.qwen_linear_prefill`, `kernel.hipfire.gemm_float`, `kernel.hipfire.gemm_ggml`, `kernel.hipfire.gemm_hfq`, `kernel.hipfire.gemm_mfp4`, `kernel.hipfire.gemm_mq`, `kernel.kestrel.exl3`, `kernel.kestrel.hipblaslt`, `kernel.libstrix.gemm`, `kernel.radiance.halo_gemm_fp8`, `kernel.radiance.halo_gemm_int`, `kernel.radiance.halo_gemm_mxfp4`, `kernel.strix_trellis.exl3_wmma_gemm`, `kernel.strixfold.exl3`, `kernel.strixfold.triton_kit`, `kernel.strixite.linear_wmma`, `kernel.tensorfold.hip_tiles` |
| `q5_wave64.cuh` | A | 12 | `kernel.ciru_iu4.q5` | - | none | - |
| `donor_wkquant_kernel.cuh` | A | 419 | `kernel.ciru_iu4.q5` | `WKQuantA8BlockedWmmaGEMMKernel` | none | - |
| `mmvq.cu` | M | 1616 | `kernel.ciru_iu4.q5` | - | `linear.decode/gguf-kquant/f32` | `kernel.ciru_inkling.glue_decode`, `kernel.ciru_inkling.quant_decode`, `kernel.crucible.exl3_gemv`, `kernel.fastasfuck.linear`, `kernel.flerp.gguf_gemv_hipfire`, `kernel.flerp.linear_q4p`, `kernel.flerp.linear_q8p`, `kernel.gufo.ds4_linear`, `kernel.gufo.ds4_mmq`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qfn_mmq`, `kernel.gufo.qwen_linear_decode`, `kernel.hipfire.gemv_float`, `kernel.hipfire.gemv_ggml`, `kernel.hipfire.gemv_hfq`, `kernel.hipfire.gemv_mfp4`, `kernel.hipfire.gemv_mq`, `kernel.hipfire.pm_gfx1151`, `kernel.kestrel.exl3`, `kernel.libstrix.gemm`, `kernel.lucebox.gemv`, `kernel.radiance.halo_gemm_fp8`, `kernel.radiance.halo_gemm_int`, `kernel.radiance.halo_gemm_mxfp4`, `kernel.strix_trellis.exl3_dense_r1`, `kernel.strix_trellis.exl3_gemv`, `kernel.strix_trellis.exl3_tp_dense`, `kernel.strixfold.exl3`, `kernel.strixite.linear_q4`, `kernel.strixite.linear_q56`, `kernel.strixite.linear_q8`, `kernel.tensorfold.cuda_exl3`, `kernel.tensorfold.hip_affine`, `kernel.tensorfold.hip_tiles` |
| `quantize.cu` | M | 716 | `kernel.ciru_iu4.q5` | - | none | - |
| `qsa.cu` | M | 481 | `kernel.ciru_iu4.attn` | `qsa3_launch` | `attn.softmax-gqa/kv-f16/f32` | `kernel.ciru_inkling.attn_decode`, `kernel.fastasfuck.attention_gqa`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_attention`, `kernel.hipfire.attention`, `kernel.libstrix.attn`, `kernel.lucebox.paged_attn`, `kernel.lucebox.qsa`, `kernel.radiance.halo_attn`, `kernel.strixite.attention`, `kernel.tensorfold.hip_attention` |
| `qsa-decode.cu` | A | 173 | `kernel.ciru_iu4.attn` | `qsa_decode_unique_ids`, `qsa_decode_partial`, `qsa_decode_merge` | `attn.softmax-gqa/kv-f16/f32` | `kernel.ciru_inkling.attn_decode`, `kernel.fastasfuck.attention_gqa`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_attention`, `kernel.hipfire.attention`, `kernel.libstrix.attn`, `kernel.lucebox.paged_attn`, `kernel.lucebox.qsa`, `kernel.radiance.halo_attn`, `kernel.strixite.attention`, `kernel.tensorfold.hip_attention` |
| `qsa-decode.cuh` | A | 8 | `kernel.ciru_iu4.attn` | - | none | - |
| `qsa-decode-wmma.cuh` | A | 121 | `kernel.ciru_iu4.attn` | `qsa_decode_wmma_partial` | `attn.softmax-gqa/kv-f16/f32` | `kernel.ciru_inkling.attn_decode`, `kernel.fastasfuck.attention_gqa`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_attention`, `kernel.hipfire.attention`, `kernel.libstrix.attn`, `kernel.lucebox.paged_attn`, `kernel.lucebox.qsa`, `kernel.radiance.halo_attn`, `kernel.strixite.attention`, `kernel.tensorfold.hip_attention` |
| `lightning-indexer.cu` | M | 722 | `kernel.ciru_iu4.attn` | - | `attn.qsa-indexer/-/f32` | `kernel.crucible.qsa`, `kernel.fastasfuck.qsa_indexer`, `kernel.gufo.qfn_kernels`, `kernel.hipfire.indexer_topk`, `kernel.hipfire.pm_gfx1151`, `kernel.kestrel.qsa_score`, `kernel.libstrix.attn`, `kernel.radiance.halo_qsa`, `kernel.strixite.qsa`, `kernel.tensorfold.hip_flashnext` |
| `fattn.cu` | M | 849 | `kernel.ciru_iu4.attn` | `flash_attn_index_mask_set`, `flash_attn_index_mask_clear`, `flash_attn_index_mask_empty` | `attn.softmax-gqa/kv-f16/f32` | `kernel.ciru_inkling.attn_decode`, `kernel.fastasfuck.attention_gqa`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_attention`, `kernel.hipfire.attention`, `kernel.libstrix.attn`, `kernel.lucebox.paged_attn`, `kernel.lucebox.qsa`, `kernel.radiance.halo_attn`, `kernel.strixite.attention`, `kernel.tensorfold.hip_attention` |
| `fattn-common.cuh` | M | 1318 | `kernel.ciru_iu4.attn` | - | none | - |
| `fattn-mma-f16.cuh` | M | 2161 | `kernel.ciru_iu4.attn` | - | none | - |
| `argsort.cu` | M | 293 | `kernel.ciru_iu4.attn` | - | `attn.indexer-topk/-/i32` | `kernel.gufo.ds4_indexer`, `kernel.hipfire.indexer_topk`, `kernel.strixfold.aux`, `kernel.strixfold.triton_kit` |
| `gated_delta_net.cu` | M | 1341 | `kernel.ciru_iu4.gdn` | `gdn_chunk_prep_kernel`, `gdn_chunk_scan_kernel`, `gdn_chunk_tail_snapshots_kernel` | `mixer.gdn-prefill/-/f32` | `kernel.crucible.gdn`, `kernel.libstrix.gdn` |
| `gated_delta_net.cuh` | M | 30 | `kernel.ciru_iu4.gdn` | - | none | - |
| `ple-conv.cu` | M | 151 | `kernel.ciru_iu4.ple_conv` | - | none | - |
| `ggml-cuda.cu` | M | 7013 | `kernel.ciru_iu4.backend` | - | none | - |
| `hc-capture.inc` | A | 100 | `kernel.ciru_iu4.backend` | - | none | - |
| `hc-merge.inc` | A | 56 | `kernel.ciru_iu4.backend` | - | none | - |
| `hc-replay.inc` | A | 15 | `kernel.ciru_iu4.backend` | - | none | - |
| `fixture.h` | A | 13 | `kernel.ciru_iu4.backend` | - | none | - |

33 files: 18 added, 15 changed; 16 carry a tag, 17 are none (headers, includes and host glue). `ggml/src/ggml-hip/CMakeLists.txt` (changed, in `kernel.ciru_iu4.backend`) is the build file.

## What each file does

- `c1.cu` (c1): the C1 translation unit: packs the activation to Q8 (K32 split, centred or not), runs the vector paths for short batches and the grouped projection for longer ones, and keeps the F32 reference; ggml_cuda_mul_mat_id_c1 is the entry the backend calls.
- `c1.cuh` (c1): declares ggml_cuda_mul_mat_id_c1.
- `c1-grouped.cuh` (c1): grouped C1 projection for more than eight tokens: routes are counted and scattered per expert, then four N16 waves share each expert's table in a CTA; the frozen 4.5175-bpw GGUF values are consumed directly.
- `c1-vector-r4.cuh` (c1): four adjacent output rows share one activation packet and one wave (with and without a table LUT); the reduction matches the one-row path.
- `mmb.cu` (mmb): MMB translation unit; CIRU adds the routed C1 kernels (the C1 row decoded into the BF16 WMMA tile, LLAMA_C1_MMB / LLAMA_C1_GROUP4 / LLAMA_C1_PACKED_HALF), the HaloBox 25fda3478 Q5_K tile loader (mmb_f32_halo_kernel, CIRU_HALO_PREFILL / CIRU_HALO_Q5_MMB) and ggml_cuda_mmb_hc_q8_pair; +708 -46 lines against the base.
- `mmb.cuh` (mmb): MMB declarations; CIRU adds ggml_cuda_mmb_hc_q8_pair; +1 line.
- `hc_q8_pair.cuh` (mmb): interface of the HC Q8_0 pair prototype (Wdown [10240, 320] and Winject [10240, 4] as separate GGUF Q8_0 matrices).
- `hc_q8_pair_kernel.cuh` (mmb): one kernel computes the hyper-connection down projection and the inject projection from the MMB BF16 activation with the exact mmb_dq_row68 arithmetic, storing both GGML F32 outputs (CIRU_HC_Q8_FUSE / LLAMA_HC_Q8_PAIR).
- `moe-weighted-reduction.cu` (mmb): routed-expert weighted reduction; CIRU adds a BF16 v4 kernel that merges the following ADD (rounded after the reduction, as the unfused graph); +43 -1 lines.
- `q5_short.cu` (q5): Gufo-derived short-batch Q5_K x F32 GEMM, a 4-token kernel and a 5-token one behind CIRU_Q5_SHORT_F32_T5, compiled wave64 with -fno-fast-math (CIRU_Q5_SHORT_GFX1151); MIT notice of Gufo in the file.
- `q5_short.cuh` (q5): declares ggml_cuda_q5_short_f32.
- `q5_wave64.cu` (q5): Q5_K decode, Gufo-format Q8 activation packing with sums and launch glue around the Gufo GEMM; default absent from every serving runtime (CIRU_Q5_GUFO_WAVE64).
- `q5_wave64.cuh` (q5): declares the wave64 operator (scratch from the ggml HIP pool, the current stream).
- `donor_wkquant_kernel.cuh` (q5): unmodified excerpt of Gufo 990fdce src/models/qwen/hip/kernels/prefill_quant_gemm.hpp, compiled inside q5_wave64.cu.
- `mmvq.cu` (q5): ggml's MMVQ; CIRU routes Q5_K short batches to ggml_cuda_q5_short_f32; +9 lines.
- `quantize.cu` (q5): ggml's activation quantizer; CIRU_Q5_CONSISTENT_SUM uses the stored scale for both the integer dot and its offset correction; +23 -4 lines.
- `qsa.cu` (attn): QSA3 sparse attention for long ubatches; CIRU reworks kernel B (one workgroup of 128 lanes a query, each lane merging the sorted selected rows of its key range) and adds query grouping (CIRU_QSA_QUERY_GROUP); +107 -71 lines.
- `qsa-decode.cu` (attn): per-query selected-key attention for MTP and appends up to 512 (516) rows: unique selected ids, split partials over the selection, then a merge.
- `qsa-decode.cuh` (attn): declares the decode path and its support test (replays up to four cached tokens before a 512-token append).
- `qsa-decode-wmma.cuh` (attn): WMMA form of the decode partials.
- `lightning-indexer.cu` (attn): QSA lightning indexer; CIRU adds the compact form (ggml_lightning_indexer_compact: a key is visible when its limit is below the query's, instead of an F16 mask); +25 -8 lines.
- `fattn.cu` (attn): flash-attention dispatch; CIRU adds the QSA decode route (LLAMA_QSA_PER_QUERY), an index-mask fallback with bounded scratch when QSA3 declines a large prefill, and F32 accumulation for the 256-wide RDNA3.5 path; +95 lines.
- `fattn-common.cuh` (attn): shared FA code; CIRU fixes mask views that hold exactly the live query rows, matches sequence broadcasting, and adds CIRU_FA_LEGACY_LDS_OCCUPANCY (the 64 KiB LDS occupancy bound of HIP 7.15 for reproducible split order); +23 -3 lines.
- `fattn-mma-f16.cuh` (attn): MMA flash attention; CIRU checks every mask element unless a finite probe keeps a dense tile on the normal path; +19 lines.
- `argsort.cu` (attn): bitonic argsort; CIRU lets a row exceed the per-block thread limit (each thread handles disjoint pairs per stage); +26 -25 lines.
- `gated_delta_net.cu` (gdn): chunked prefill beside the base recurrent kernel; GGML_GDN_CHUNK / GGML_GDN_CHUNK_MIN gate it, CIRU_GDN_MULTI_TILE is an off-by-default variant; +813 -9 lines.
- `gated_delta_net.cuh` (gdn): declares the eligibility test and the q / k norm, conv and output-norm registrations the backend makes before the node; +16 lines.
- `ple-conv.cu` (ple_conv): fused width-4, dilation-3 depthwise causal convolution of the PLE stream with its carried tail; CIRU accepts BF16 weights beside F16 and adds CIRU_PLE_CONV_LOG; +24 -6 lines.
- `ggml-cuda.cu` (backend): the backend translation unit; CIRU adds the exact-batch path, the C1 / Q5_K / HC / GDN routing and graph matching; +480 -9 lines.
- `hc-capture.inc` (backend): hyper-connection capture and replay priming (ciru_hc_capture_step / output / replay_prime / save_bytes), included in ggml-cuda.cu.
- `hc-merge.inc` (backend): exact HC merge scheduling (ciru_hc_exact_merge, ciru_hc_schedule_shared), included in ggml-cuda.cu.
- `hc-replay.inc` (backend): diagnostic exports for the standalone component driver, included in ggml-cuda.cu.
- `fixture.h` (backend): fixture record type the HC capture writes.

## Not done here

No kernel was run or timed. The compile gate (`gate/compile/run.py` on control-node, CPU only) compiles each family's
translation units for gfx1151 with the ggml-hip target's flags. The v6.1.0 kernel changes are in a patch on HaloBox b1064
(`contrib/flash-v6.1/source/halobox-b1064-flash-v6.1.patch`) and are not in this table.
