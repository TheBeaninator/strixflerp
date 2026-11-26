# libstrix kernels by strixflerp interface

One row for each of the 22 kernel files of `csrc/{attn,gdn,gemm,moe,norm}` at 4f1c06f. Card ORIGIN-LIBSTRIX
dd6f98d883. Written by `docs/harvest/harvest_libstrix.py` from its `KINDS` table; read from source and from the
origin's `docs/kernels.md`, nothing here was compiled or run.

- **tag**: the strixflerp interface class the file matches, `<family>.<op>/<weight format>/<dtype>`, or none. The
  module's `[[provides]]` entry carries the same name with a `.libstrix` variant tag.
- **same op in the catalogue**: kernel modules of other origins that provide that `<family>.<op>`; the harvest
  script fails if one of them does not. The same op is not the same operands: every libstrix kernel takes torch
  tensors in vLLM's layouts (`csrc/torch_ops.cpp`), so each row needs an adapter before it can stand in for a
  strixflerp module.

| file | lines | module | entry points (`torch.ops.strix.`) | tag | same op in the catalogue |
|---|---:|---|---|---|---|
| `csrc/attn/attn_decode_bf16.hip` | 432 | `kernel.libstrix.attn` | `attn_decode_bf16` | `attn.softmax-gqa/kv-bf16+i8/bf16` | `kernel.strixite.attention`, `kernel.gufo.qwen_attention`, `kernel.hipfire.attention`, `kernel.tensorfold.hip_attention` |
| `csrc/attn/attn_prefill_bf16.hip` | 664 | `kernel.libstrix.attn` | `attn_prefill_bf16` | `attn.prefill-causal/kv-bf16+i8/bf16` | `kernel.flerp.attn_wmma_gqa` |
| `csrc/attn/attn_prefill32_bf16.hip` | 324 | `kernel.libstrix.attn` | `attn_prefill32_bf16` | `attn.prefill-causal/kv-bf16/bf16` | `kernel.flerp.attn_wmma_gqa` |
| `csrc/attn/qsa_indexer_bf16.hip` | 172 | `kernel.libstrix.attn` | `qsa_indexer_bf16` | `attn.qsa-indexer/-/bf16` | `kernel.strixite.qsa`, `kernel.crucible.qsa`, `kernel.gufo.qfn_kernels`, `kernel.hipfire.indexer_topk`, `kernel.tensorfold.hip_flashnext` |
| `csrc/attn/qsa_kv_to_f16.hip` | 137 | `kernel.libstrix.attn` | `qsa_kv_to_f16` | none | - |
| `csrc/attn/qsa_union_bf16.hip` | 422 | `kernel.libstrix.attn` | `qsa_union_attn_bf16` | `attn.softmax-gqa/kv-bf16+i8/bf16` | `kernel.strixite.attention`, `kernel.gufo.qfn_kernels`, `kernel.hipfire.attention` |
| `csrc/gdn/gdn_chunk_k128_v128_c64_bf16.hip` | 557 | `kernel.libstrix.gdn` | `gdn_chunk_fwd_k128_v128_c64_bf16` | `mixer.gdn-prefill/-/bf16` | `kernel.crucible.gdn` |
| `csrc/gdn/gdn_conv_post_bf16.hip` | 303 | `kernel.libstrix.gdn` | `gdn_causal_conv1d_w4_bf16`, `gdn_post_conv_bf16` | `mixer.gdn-front/-/bf16` | `kernel.gufo.qwen_gdn`, `kernel.hipfire.gdn` |
| `csrc/gdn/gdn_decode_mtp_bf16.hip` | 295 | `kernel.libstrix.gdn` | `gdn_decode_mtp_bf16` | `mixer.gdn/-/bf16` | `kernel.strixite.gdn`, `kernel.crucible.gdn`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qwen_gdn`, `kernel.hipfire.gdn`, `kernel.tensorfold.hip_recurrence` |
| `csrc/gemm/fp8_dequant.hip` | 92 | `kernel.libstrix.gemm` | `fp8_dequant` | none | - |
| `csrc/gemm/fp8_wmma_gemm.hip` | 346 | `kernel.libstrix.gemm` | `fp8_wmma_gemm`, `fp8_wmma_gemm_cfg`, `fp8_wmma_gemm_raw`, `fp8_wmma_gemm_raw_cfg` | `linear.prefill/fp8-e4m3/bf16` | `kernel.strixite.linear_wmma`, `kernel.hipfire.gemm_float` |
| `csrc/gemm/hc_up_gate_mix_bf16.hip` | 201 | `kernel.libstrix.gemm` | `hc_up_gate_mix_bf16` | `hc.mix-prefill/bf16/bf16` | `kernel.strixite.hc_wmma`, `kernel.crucible.hc` |
| `csrc/gemm/mxfp4_gemv.hip` | 149 | `kernel.libstrix.gemm` | `mxfp4_gemv` | `linear.decode/mxfp4/bf16` | `kernel.hipfire.gemv_mfp4` |
| `csrc/gemm/mxfp4_wmma_gemm.hip` | 392 | `kernel.libstrix.gemm` | `mxfp4_wmma_gemm`, `mxfp4_wmma_gemm_cfg`, `mxfp4_dequant` | `linear.prefill/mxfp4/bf16` | `kernel.hipfire.gemm_mfp4` |
| `csrc/gemm/w4a16_wmma_gemm.hip` | 247 | `kernel.libstrix.gemm` | `w4a16_wmma_gemm`, `w4a16_dequant` | `linear.prefill/ct-int4/bf16` | `kernel.strixite.linear_wmma`, `kernel.flerp.wmma_q4p` |
| `csrc/gemm/w8a16_skinny_gemm.hip` | 776 | `kernel.libstrix.gemm` | `w8a16_skinny_gemm`, `fp8_skinny_gemm` | `linear.decode/ct-int8+fp8-e4m3/bf16` | `kernel.strixite.linear_q8`, `kernel.flerp.linear_q8p`, `kernel.gufo.qfn_kernels` |
| `csrc/gemm/w8a8_wmma_gemm.hip` | 945 | `kernel.libstrix.gemm` | `w8a8_wmma_gemm`, `w8a8_wmma_gemm_cfg`, `w4a8_wmma_gemm` | `linear.prefill/ct-int8+ct-int4/i8-f32` | `kernel.gufo.qfn_kernels` |
| `csrc/moe/moe_gemm_bf16.hip` | 384 | `kernel.libstrix.moe` | `moe_gemm_bf16` | `moe.experts-prefill/bf16/bf16` | `kernel.strixite.moe_grouped`, `kernel.flerp.moe_grouped_p`, `kernel.hipfire.moe` |
| `csrc/moe/moe_gemm_w4a16.hip` | 488 | `kernel.libstrix.moe` | `moe_gemm_w4a16` | `moe.experts-prefill/ct-int4/bf16` | `kernel.strixite.moe_grouped`, `kernel.flerp.moe_kgrouped` |
| `csrc/moe/moe_gemm_w4a8.hip` | 466 | `kernel.libstrix.moe` | `moe_gemm_w4a8`, `moe_act_quant_i8` | `moe.experts-prefill/ct-int4/i8-f32` | `kernel.strixite.moe_grouped` |
| `csrc/moe/moe_gemm_w8a16.hip` | 625 | `kernel.libstrix.moe` | `moe_gemm_w8a16`, `moe_gemm_fp8`, `moe_gemm_fp8_raw` | `moe.experts-prefill/ct-int8+fp8-e4m3/bf16` | `kernel.strixite.moe_grouped` |
| `csrc/norm/gdn_gated_rmsnorm_h128_bf16.hip` | 140 | `kernel.libstrix.norm` | `gdn_gated_rmsnorm_h128_bf16` | `norm.rmsnorm-gated/-/bf16` | `kernel.strixite.norm` |

22 files: attn 6, gdn 3, gemm 8, moe 4, norm 1. 20 carry a tag, 2 are none (`qsa_kv_to_f16.hip` and `fp8_dequant.hip`, copies that feed another kernel).
`csrc/testing/wmma_layout_test.hip`, the 23rd `.hip` file of the origin, is a test unit in `kernel.libstrix.common`.

## What each kernel does

- `attn_decode_bf16.hip`: paged attention for pure-decode batches, one query token a sequence: a block owns one (sequence, KV head, key segment) and all query heads of that KV head, stages each K/V chunk in LDS once, and runs an online softmax per head; split-KV over key segments, optional sliding window and attention sinks.
- `attn_prefill_bf16.hip`: causal paged attention for prefill and speculative verify over cu_seqlens_q: a block owns 16 query tokens of one sequence and HPB query heads of one KV head, a head is D / 128 waves that split head_dim, and everything is computed transposed (S^T = K Q^T, O^T = V^T P^T) so a WMMA lane owns one query column.
- `attn_prefill32_bf16.hip`: head_dim 512 causal prefill with 32 query columns sharing one K/V stage: eight waves change role by phase (score: 2 column tiles x 4 D-slices; softmax: one wave a column tile; P V: 64 V dims a wave for both tiles).
- `qsa_indexer_bf16.hip`: Qwen3.8-Flash-Next QSA indexer scores over the paged compressed-key cache: logits[r, c] = sum over heads of relu(q[r, h] . k[c]) / divisor for c < visible[r], -inf past it; a block owns 64 rows x 64 keys and stages the key tile in LDS once for all heads.
- `qsa_kv_to_f16.hip`: rewrites one layer's visible K/V from the paged cache into compact fp16 arrays indexed by token (exact from a bf16 cache, q * scale rounded once from an int8 cache), the copy qsa_union_attn_bf16 gathers from.
- `qsa_union_bf16.hip`: Qwen3.8-Flash-Next QSA sparse attention over row unions: 16 consecutive query rows share one sorted union of their selected tokens (a build kernel makes the union list and one membership bit a row from a bitmap and prefix popcount in LDS), the attention kernel gathers K/V for the union once and masks each score by membership, in attn_prefill_bf16's transposed WMMA layout.
- `gdn_chunk_k128_v128_c64_bf16.hip`: gated delta rule, chunked prefill over cu_seqlens: kernel 1 computes A = (I + strict_lower(beta k.k e^dgc))^-1 and P = scale tril(q.k e^dgc) for every (chunk, head) in parallel, kernel 2 walks one sequence's chunks for one head and a 64-column slice of V with the state in registers.
- `gdn_conv_post_bf16.hip`: GDN prefill front end: the width-4 causal depthwise conv over mixed_qkv with a 3-row carried state a sequence, then the post-conv split into q / k / v with L2 norm and the g and beta gates; feature blocks are the fastest grid index so concurrent blocks read the same token rows.
- `gdn_decode_mtp_bf16.hip`: GDN speculative decode (MTP) for one request x value head a block: L2 norm of q / k, gating, the delta-rule recurrence over the request's draft tokens from the last accepted state, a state write per token slot and the gated RMS norm of the output; a HIP port of vLLM's CUDA-only csrc/gdn/fused_gdn_decode_kernel.cu.
- `fp8_dequant.hip`: float8_e4m3fn [N, K] with 128x128 block scales or per-output-channel scales to a bf16 [N, K] copy, which the plugin hands to hipBLASLt for large prefill batches.
- `fp8_wmma_gemm.hip`: FP8 W8A16 GEMM C[M, N] = A[M, K] x W[N, K]^T: e4m3 bytes are decoded to bf16 as they are staged into LDS and the main loop is a bf16 WMMA GEMM; the raw variant turns the bytes into fp16 (value / 2^8) with shifts and masks, runs the fp16 WMMA and applies the block scale to the fp32 accumulator each K step.
- `hc_up_gate_mix_bf16.hip`: Qwen3.8-Flash-Next hyper-connection up projection fused with the gate mix: y[m, c] = (1/4) sum over 4 streams of sigmoid(bf16(a[m] . w[s * C + c])) * x[m, s * C + c]; a block owns BM rows x BN columns of y and all four streams, K (the low rank) runs in 64-wide steps, and the [M, 4C] gate is never written to memory.
- `mxfp4_gemv.hip`: MXFP4 W4A16 GEMV: one wave an output row, eight rows a block, A staged in LDS per K chunk, weights decoded in registers straight from the packed stream (16 weight bytes and one e8m0 scale byte a 32-K block).
- `mxfp4_wmma_gemm.hip`: MXFP4 W4A16 GEMM on Quark's packed layout (weight u8 [N, K/2], one e8m0 scale byte per 32 elements): weights are decoded to bf16 as they are staged into LDS, then a bf16 WMMA GEMM; no dequantized weight copy.
- `w4a16_wmma_gemm.hip`: W4A16 GEMM for prefill on vLLM's RDNAHybridW4A16 layout (symmetric uint4b8 as an int8 [N, K/2] view of ExLlama-shuffled int32 words, bf16 group scales): weights are dequantized to bf16 ((nibble - 8) * scale) as they are staged into LDS, then a bf16 WMMA GEMM.
- `w8a16_skinny_gemm.hip`: W8A16 skinny GEMM for decode: uint8b128 weights [N, K] with per-channel or group-128 scales against bf16 / fp16 activations, and the same file's skinny GEMM on float8_e4m3fn weights (the e4m3 byte read as an fp16 holding value / 2^8); derived from vLLM's csrc/rocm/skinny_gemms.cu through the strix-vllm fork (Apache-2.0).
- `w8a8_wmma_gemm.hip`: W8A8 GEMM on the iu8 16x16x16 WMMA for larger M: uint8b128 weights read raw, activations quantized to int8 per (row, 128-element K group) inside the op, the i32 accumulator folded into fp32 with s_w * s_a at every group boundary; w4a8_wmma_gemm is the int4-weight entry of the same file; derived from vLLM's csrc/rocm/skinny_gemms.cu through the strix-vllm fork (Apache-2.0).
- `moe_gemm_bf16.hip`: grouped expert GEMM over vLLM's moe_align_block_size layout, C[sid, n] = sum_k A[sid / top_k, k] * B[expert(sid), n, k] (* w[sid]): a block owns BM sorted ids of one expert and BN output columns; the w13 GEMM fuses SiLU or GELU-tanh and the gate * up product, each activation its own template instantiation.
- `moe_gemm_w4a16.hip`: the same grouped expert GEMM on int4 weights in vLLM's Triton WNA16 layout (uint8 [E, N, K/2], plain nibble order, bf16 group scales): weights are dequantized as they are staged into LDS, or with DEFER staged as the exact bf16 128 + q and the group scale applied to the fp32 accumulator at the group's end.
- `moe_gemm_w4a8.hip`: the grouped expert GEMM on the iu8 WMMA: activations pre-quantized to int8 per (row, 128-element group) by moe_act_quant_i8, int4 weights turned into int8 at the LDS commit with two byte permutes and an add / xor per 4 values, the i32 accumulator folded into fp32 with sa * sw every 128-wide K step.
- `moe_gemm_w8a16.hip`: the grouped expert GEMM on one-byte weights, the format a template argument: int8 with a scale per output channel, float8_e4m3fn with per-channel or 128x128 block scales, and two raw FP8 forms that turn the bytes into fp16 with shifts and masks, run the fp16 WMMA and apply the scale to the fp32 accumulator.
- `gdn_gated_rmsnorm_h128_bf16.hip`: the GDN output norm out = rms_norm(x) * w * act(z) over 128-channel rows (RMSNormGated with norm_before_gate=True), 16 lanes a row and 8 channels (one 16-byte load) a lane.

## Not done here

No kernel was compiled, linked or timed. A timing against the strixite modules is its own experiment card; it needs
`_C.so` built against PyTorch for ROCm, or an adapter that lifts a kernel out of its torch host function.
