# Inkling 2 Mini CIRU HIP kernels by strixflerp interface

One row for each group of `extern "C"` launch functions in the seven HIP units of `overlay/vllm/models/inkling2/amd/ops` at b3870f3 (v1.5.0).
Card ORIGIN-CIRU-INKLING 8bd576841e. Written by `docs/harvest/harvest_ciru_inkling.py` from its `UNITS` table, read
from source and the origin's README.md; the compile gate builds each unit for gfx1151, nothing was run.

- **tag**: the strixflerp interface class the group matches, `<family>.<op>/<weight format>/<dtype>`, or none. The
  module's `[[provides]]` entry carries the same name with a `.ciru-inkling...` variant tag.
- **same op in the catalogue**: kernel modules of other origins that provide that `<family>.<op>`; the harvest script
  fails if one of them does not. Same op is not same operands: these kernels take raw pointers in the layouts of
  Ciru's vLLM overlay (paged K/V cache with a block table, route plans from the planners, the packed banks).

| unit | lines | launch functions | tag | same op in the catalogue |
|---|---:|---|---|---|
| `hip_moe_prefill.hip` | 478 | `launch_source_w13_f16_xbf16`, `launch_source_w13_bf16_xbf16`, ... 20 in all | `moe.experts-prefill/nvfp4+int4-g32+ciru-c1/bf16` | `kernel.strixite.moe_grouped`, `kernel.crucible.moe_prefill`, `kernel.hipfire.moe`, `kernel.radiance.halo_moe` |
| `hip_moe_prefill.hip` | 478 | `launch_pair_m64` | none | - |
| `hip_moe_prefill.hip` | 478 | `launch_hip_moe_reduce` | `moe.combine/-/bf16` | `kernel.gufo.qfn_kernels`, `kernel.hipfire.moe`, `kernel.radiance.halo_moe` |
| `hip_moe_prefill.hip` | 478 | `inkling_hip_abi_version`, `inkling_hip_src_hash` | none | - |
| `hip_attn_prefill.hip` | 942 | `launch_hip_kv_conv_1w` | `kvcache.store/kv-bf16/bf16` | `kernel.gufo.ds4_attention`, `kernel.hipfire.kv_cache`, `kernel.strixfold.triton_kit` |
| `hip_attn_prefill.hip` | 942 | `launch_hip_qr_clip` | none | - |
| `hip_attn_prefill.hip` | 942 | `launch_hip_gather_paged_kv_hnd` | none | - |
| `hip_attn_prefill.hip` | 942 | `launch_hip_flash_rel_attn_prefill`, `launch_hip_flash_rel_attn_near_merge` | `attn.prefill-causal/kv-bf16/bf16` | `kernel.flerp.attn_wmma_gqa`, `kernel.radiance.halo_attn` |
| `hip_glue_prefill.hip` | 1121 | `launch_q4_coeffs` | none | - |
| `hip_glue_prefill.hip` | 1121 | `launch_hip_sconv_prefill` | none | - |
| `hip_glue_prefill.hip` | 1121 | `launch_hip_loop_zeromat_prefill` | none | - |
| `hip_glue_prefill.hip` | 1121 | `launch_hip_attn_res_prefill` | none | - |
| `hip_glue_prefill.hip` | 1121 | `launch_hip_router_prefill` | `moe.gate-logits/bf16/bf16-f32` | `kernel.flerp.moe_kquant`, `kernel.strixfold.moe_gate` |
| `hip_attn_decode.hip` | 1217 | `launch_hip_kv_conv_decode` | `kvcache.store/kv-bf16/bf16` | `kernel.gufo.ds4_attention`, `kernel.hipfire.kv_cache`, `kernel.strixfold.triton_kit` |
| `hip_attn_decode.hip` | 1217 | `launch_hip_flash_rel_decode_split`, `launch_hip_decode_combine` | `attn.softmax-gqa/kv-bf16/bf16` | `kernel.strixite.attention`, `kernel.gufo.qwen_attention`, `kernel.hipfire.attention`, `kernel.radiance.halo_attn` |
| `hip_quant_decode.hip` | 838 | `launch_hip_w6_wmma` | `linear.decode/ciru-w6-g64/bf16` | `kernel.strixite.linear_q56`, `kernel.hipfire.gemv_hfq`, `kernel.gufo.qwen_linear_decode` |
| `hip_quant_decode.hip` | 838 | `launch_hip_w4_wmma` | `linear.decode/ciru-w4/bf16` | `kernel.strixite.linear_q4`, `kernel.flerp.linear_q4p`, `kernel.hipfire.gemv_mq` |
| `hip_quant_decode.hip` | 838 | `launch_hip_reduce_splits` | none | - |
| `hip_quant_decode.hip` | 838 | `launch_hip_w6_head_gemv_m1` | `logits.head/ciru-w6-g32+g64/bf16-f32` | `kernel.radiance.halo_gemm_fp8` |
| `hip_quant_decode.hip` | 838 | `launch_hip_w4_markov_argmax` | `draft.markov/ciru-w4/bf16` | `kernel.gufo.ds4_indexer`, `kernel.strixfold.triton_kit` |
| `hip_moe_decode.hip` | 437 | `launch_hip_moe_decode_plan` | none | - |
| `hip_moe_decode.hip` | 437 | `launch_hip_moe_decode_unified` | `moe.experts-decode/nvfp4+int4-g32+ciru-c1/bf16` | `kernel.strixite.linear_q4`, `kernel.crucible.moe_decode`, `kernel.hipfire.moe`, `kernel.radiance.halo_moe` |
| `hip_glue_decode.hip` | 1426 | `launch_hip_sconv_decode` | none | - |
| `hip_glue_decode.hip` | 1426 | `launch_hip_loop_decode_down`, `launch_hip_loop_decode_blend_rms`, `launch_hip_loop_decode_blend_rms_2pass` | none | - |
| `hip_glue_decode.hip` | 1426 | `launch_hip_pair_rmsnorm`, `launch_hip_blend_rmsnorm`, `launch_hip_mix_rmsnorm`, `launch_hip_rms_norm` | `norm.rmsnorm/-/bf16` | `kernel.gufo.qwen_norm_rope`, `kernel.hipfire.norm_rope`, `kernel.strixfold.aux` |
| `hip_glue_decode.hip` | 1426 | `launch_hip_router_gemm_m8` | `moe.gate-logits/bf16/bf16-f32` | `kernel.flerp.moe_kquant`, `kernel.strixfold.moe_gate` |
| `hip_glue_decode.hip` | 1426 | `launch_hip_route_m8` | `moe.router/-/f32` | `kernel.strixite.moe_router`, `kernel.crucible.ops`, `kernel.hipfire.moe` |
| `hip_glue_decode.hip` | 1426 | `launch_hip_qk_norm_rope` | `pos.rope-heads/-/bf16` | `kernel.gufo.ds4_norm_rope`, `kernel.strixfold.rope_heads` |
| `hip_glue_decode.hip` | 1426 | `launch_hip_dspark_w8` | `linear.decode/i8-g128/bf16` | `kernel.strixite.linear_q8`, `kernel.flerp.linear_q8p`, `kernel.gufo.qfn_kernels` |
| `hip_glue_decode.hip` | 1426 | `launch_hip_silu_and_mul` | `act.swiglu/-/bf16` | `kernel.strixite.swiglu`, `kernel.gufo.qfn_kernels`, `kernel.strixfold.dense_bf16` |
| `hip_glue_decode.hip` | 1426 | `launch_hip_add_softcap` | none | - |
| `hip_glue_decode.hip` | 1426 | `launch_hip_markov_gate` | `draft.markov/-/bf16` | `kernel.gufo.ds4_indexer`, `kernel.strixfold.triton_kit` |

7 units, 32 launch groups (59 entry points): 19 carry a tag, 13 are none (planners, helpers, ABI metadata and the Inkling-specific short convolution, loop controller and block attention residuals, which no catalogue module provides).
`hip_common.cuh` is the header-only module `kernel.ciru_inkling.common`.

## What each group does

- `hip_moe_prefill.hip` `launch_source_w13_f16_xbf16` (+19): one templated kernel (moe_wmma_k32_unified_kernel) for both expert projections of prefill over a sorted route list: a block owns a 128-row (paired) or 64-row (tail) block of routes of one expert and 128 (w13) or 256 (w2, N % 256 == 0) output columns, K in steps of 32 on the f16 WMMA, weights unpacked from the packed bank in registers; 16-row fragments that are all padding are skipped; w13 fuses the softcapped SwiGLU (gate_cap, up_cap).
- `hip_moe_prefill.hip` `launch_pair_m64`: one-warp planner that scans the 64-row block expert ids of the route plan into unified descriptors (128-row pairs of adjacent blocks of one expert, 64-row tails) and their counts, on the device (no host sync).
- `hip_moe_prefill.hip` `launch_hip_moe_reduce`: sums the 8 routed rows of each token into its output row.
- `hip_moe_prefill.hip` `inkling_hip_abi_version` (+1): ABI metadata the loader checks before it uses the library: INKLING_HIP_ABI_VERSION (3) and the source hash the library was compiled from (INKLING_HIP_SRC_HASH_RAW).
- `hip_attn_prefill.hip` `launch_hip_kv_conv_1w`: Inkling's K/V short convolution fused with the paged cache write: one warp a (token tile, KV head) applies the width-4 causal depthwise conv to K and V (taps from the conv-state cache across request boundaries), scales and clips, and writes the results to the paged K/V cache and the new inputs to the conv-state cache.
- `hip_attn_prefill.hip` `launch_hip_qr_clip`: exact per-head, per-channel L2 clip of Q and of the relative-position query R before the post-8K temperature (tau), 16 lanes a head.
- `hip_attn_prefill.hip` `launch_hip_gather_paged_kv_hnd`: copies one request's pages of the paged K/V cache into contiguous [KVH, N, D] arrays, the long-context far part of hybrid prefill.
- `hip_attn_prefill.hip` `launch_hip_flash_rel_attn_prefill` (+1): causal flash attention over the paged cache with Inkling's learned relative-position bias: a block owns 16 query tokens of one head and computes K Q^T and V^T P^T on the BF16 WMMA (12.5 KiB LDS, 2 barriers a 16-key tile); keys farther than EXTENT take no bias (far phase), nearer keys take R . RelProjT[distance] (near phase); sliding window for local layers; a fused output gate and attention-sink epilogue; near_merge runs the near part and merges a far partial (OutFar, LseFar) computed over gathered pages.
- `hip_glue_prefill.hip` `launch_q4_coeffs`: coefficients of the 4-query block-attention-residual message (D = 2304, one stacked block) without materialising the mix.
- `hip_glue_prefill.hip` `launch_hip_sconv_prefill`: Inkling's static short convolution of the residual streams for prefill (BT = 64 tokens, width 4, 8 channels a thread) over a paged sliding-window conv-state cache, safe at request and tile boundaries, with optional addend and residual inputs and FP32 or BF16 outputs.
- `hip_glue_prefill.hip` `launch_hip_loop_zeromat_prefill`: Inkling's loop controller for prefill, three passes a 16-token tile: down projection [16, 2304] x [2304, 64], up projection and the gain / forget blend of the destination stream with the message, then RMSNorm times the stream's norm weight; nearest relative in the catalogue: the hyper-connection mixes (hc.mix-*), which this is not.
- `hip_glue_prefill.hip` `launch_hip_attn_res_prefill`: Inkling's block attention residuals: each token mixes the stacked earlier block outputs and the last one with a softmax over query . normalised value (D = 2304, 288 threads x 8 elements; one-pass register cache up to 12 stacked blocks, two passes with an LDS softmax buffer above that), optional addend and RMSNorm of the result.
- `hip_glue_prefill.hip` `launch_hip_router_prefill`: router GEMM for prefill on the BF16 WMMA, X read once a 128-column block, two outputs: FP32 scores and the bias-corrected scores the top-k reads.
- `hip_attn_decode.hip` `launch_hip_kv_conv_decode`: the K/V short convolution and paged cache write for decode and speculative verify: one warp a (KV head, request), no LDS and no barriers, int64 cache addressing.
- `hip_attn_decode.hip` `launch_hip_flash_rel_decode_split` (+1): split-KV flash decode attention with the relative-position bias, for target layers (with bias, tau after 8K) and the DSpark draft layers: a block owns (split, KV head, request) and the query rows of that KV head, a branchless far loop without bias over most keys, then a near / local / tail loop with the bias and exact masks; a cooperative BN = 32 variant packs 32 query rows (B = 1, M = 6..8) so K and V are read once; decode_combine merges the splits and applies the output gate and the sink.
- `hip_quant_decode.hip` `launch_hip_w6_wmma`: single-pass W6 linear for small M on the f16 WMMA: a block owns 64 output rows (4 warps x 16) and an M tile of 16, 32 or 48 rows, weights streamed as a 4-bit plane (P4, 16-byte loads) and a 2-bit plane (P2, 8-byte loads) with one scale a row and 64-wide K step, software-pipelined; optional split-K; a QKV form reads separate Q, K and V inputs for its three output ranges; optional softcap and FP32 output (the LM head).
- `hip_quant_decode.hip` `launch_hip_w4_wmma`: the same single-pass small-M WMMA linear on packed 4-bit weights with a scale a (row, K64 group), optional bias and split-K.
- `hip_quant_decode.hip` `launch_hip_reduce_splits`: split-K reducer shared by the W6, W4 and DSpark W8 kernels, with the optional softcap and FP32 output.
- `hip_quant_decode.hip` `launch_hip_w6_head_gemv_m1`: M = 1 GEMV for the W6 head: the whole weight record set of a row prefetched (up to 6 loads), v_dot2_f32_bf16, group scales of 32 or 64 weights, softcap epilogue.
- `hip_quant_decode.hip` `launch_hip_w4_markov_argmax`: the DSpark draft's rank-256 Markov fold fused with greedy selection: W4 weights against the latent, added to the base logits and bias, softcap, and an argmax with an exact tie-break, 8 lanes a row and 16-byte loads.
- `hip_moe_decode.hip` `launch_hip_moe_decode_plan`: one-warp greedy planner that pairs adjacent 16-row route blocks of one expert into 32-row blocks on the device (no host sync, safe inside a captured HIP graph).
- `hip_moe_decode.hip` `launch_hip_moe_decode_unified`: single pass a expert for decode and verify batches: a block owns one 16- or 32-row route block and a 128-row tile of the bank, weights unpacked from registers (no LDS copy of B) on a K64 software pipeline and reused for both 16-row halves of a paired block; FUSED_ACT 1 is w13 with the softcapped SwiGLU, 0 is w2.
- `hip_glue_decode.hip` `launch_hip_sconv_decode`: the static short convolution of the residual streams for decode (8 tokens a block, width 4, one channel a thread), aliasing-safe, with the prefill unit's addend / residual options.
- `hip_glue_decode.hip` `launch_hip_loop_decode_down` (+2): the loop controller for decode: down projection of the RMS-normalised (destination, message) pair, then the up projection, gain / forget blend and RMSNorm in one pass or two (one read of each weight).
- `hip_glue_decode.hip` `launch_hip_pair_rmsnorm` (+3): RMSNorm forms: pair (destination and message normalised side by side, the loop controller's input), blend (gain / forget blend then RMSNorm), mix (the DSpark draft's weighted sum of target hidden-state taps then RMSNorm) and a plain row RMSNorm.
- `hip_glue_decode.hip` `launch_hip_router_gemm_m8`: router GEMM for M <= 8 rows, optionally with the sigmoid and bias correction fused.
- `hip_glue_decode.hip` `launch_hip_route_m8`: sigmoid, bias-corrected stable top-8 of 128 experts with 256 threads, gamma weights, and optionally the expert-sorted route list and padded 16-row blocks the decode MoE plan reads.
- `hip_glue_decode.hip` `launch_hip_qk_norm_rope`: in-place per-head RMSNorm of Q and K (optional) and rotary embedding from a cos / sin cache by position, the DSpark draft layers' attention prologue.
- `hip_glue_decode.hip` `launch_hip_dspark_w8`: the DSpark draft projections on signed W8-G128 weights in N64 / K64 tiles, M = 8 rows on the WMMA, optional split-K (reduced by launch_hip_reduce_splits).
- `hip_glue_decode.hip` `launch_hip_silu_and_mul`: SiLU(gate) * up over [rows, 2N] rows.
- `hip_glue_decode.hip` `launch_hip_add_softcap`: adds two row-strided arrays and applies the softcap, the draft logits epilogue.
- `hip_glue_decode.hip` `launch_hip_markov_gate`: the DSpark Markov gate: combines the raw draft projection with the carried state into the gated output and the next rank-sized latent.

## Not done here

No kernel was run or timed on the pod and the prebuilt library was not compared with a rebuild. A timing of the B = 8
decode units against the strixite modules is its own experiment card; it needs an adapter from strixflerp's weight
views to these packed banks, or Ciru's fitted weights, which are not in this repository.
