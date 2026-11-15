# 35B world-1 decode: per-kernel time of one step (rocprofv3 kernel trace)

Runner `gate/decode_profile.py` (+ `gate/decode_profile_run.sh` for the lease). Binary: (`build/kern3/after/out/flerp-qwen35moe-generate`), `--staged loopback`, world 1, prompt `code` (20 tokens), context 4096, control-node GPU.
Decode kernels = trace(64 tokens) minus trace(32 tokens), same prompt: 32 decode steps a graph mode, no prefill or load in them. GPU ms are kernel execution time (End - Start), summed; the step's wall time is the unprofiled decode ms a step.

## Runs

| case | rc | s | load start -> end | decode ms/step (tool) | tok/s (tool) | decode calls |
|---|---|---|---|---|---|---|
| on-64 | 0 | 83.3 | 5.68 -> 4.97 | 14.84 | 67.07 | [64] |
| on-32 | 0 | 83.0 | 4.97 -> 3.44 | 16.23 | 61.34 | [32] |
| off-64 | 0 | 83.4 | 3.44 -> 3.22 | 14.57 | 68.08 | [64] |
| off-32 | 0 | 83.1 | 3.22 -> 2.36 | 15.12 | 65.63 | [32] |
| on-64-noprof | 0 | 83.2 | 2.36 -> 2.07 | 14.51 | 68.65 | [64] |

## Graph on: 32 decode steps; 617.0 kernels a token, 11.529 GPU ms a token; tool's decode ms/step 14.51

Trace rows: 40630 (64 tokens), 20886 (32 tokens).

### By role

| role | ms a token | share of GPU ms | kernels a token |
|---|---:|---:|---:|
| experts: routed + shared (IndexedExpertsKernel) | 3.771 | 32.7% | 160.0 |
| GDN: fused input projections | 2.289 | 19.9% | 30.0 |
| lm head (output.weight Q6_K GEMV) | 1.952 | 16.9% | 1.0 |
| out projections (attn_output / ssm_out, Q4_K GEMV) | 1.062 | 9.2% | 40.0 |
| GDN: recurrence + conv + q/g unpack | 0.688 | 6.0% | 70.0 |
| attention: fused QKV projections | 0.678 | 5.9% | 10.0 |
| router: gate logits (F32 ffn_gate_inp GEMV) | 0.507 | 4.4% | 40.0 |
| norms | 0.376 | 3.3% | 81.0 |
| attention: core + qk-norm/rope/kv write | 0.314 | 2.7% | 20.0 |
| router: top-k + MoE epilogue | 0.256 | 2.2% | 80.0 |
| other | 0.029 | 0.3% | 0.0 |
| argmax / embed / residual / copies | -0.394 | -3.4% | 85.0 |

### By kernel (decode kernels only)

| kernel | role | ms a token | share | launches a token | us a launch |
|---|---|---:|---:|---:|---:|
| `flerp::moe::IndexedExpertsKernel<` | experts: routed + shared (IndexedExpertsKernel) | 3.498 | 30.3% | 120.0 | 29.1 |
| `gufo::hip::Wave32FusedSSMInputProjectionsKernel_1Row<4u>` | GDN: fused input projections | 2.289 | 19.9% | 30.0 | 76.3 |
| `gufo::hip::Q8KBlockGEMVKernel_2Rows<4u, T0>` | lm head (output.weight Q6_K GEMV) | 1.952 | 16.9% | 1.0 | 1951.6 |
| `gufo::hip::Q8KBlockGEMVKernel_2RowsWideQ4K<8u>` | out projections (attn_output / ssm_out, Q4_K GEMV) | 1.062 | 9.2% | 40.0 | 26.5 |
| `gufo::hip::Wave32FusedQKVProjectionsKernel_1Row<4u>` | attention: fused QKV projections | 0.678 | 5.9% | 10.0 | 67.8 |
| `_ZN4gufo3hip24DeltaNetRecurrenceKernelITkNS0_6detail25QwenRecurrentStateElementEfLb1ELb1EL` | GDN: recurrence + conv + q/g unpack | 0.584 | 5.1% | 30.0 | 19.5 |
| `flerp::moe::GateLogitsWideKernel` | router: gate logits (F32 ffn_gate_inp GEMV) | 0.507 | 4.4% | 40.0 | 12.7 |
| `gufo::hip::RMSNormKernel<0ul>` | norms | 0.375 | 3.3% | 81.0 | 4.6 |
| `gufo::hip::QwenDecodeOnlineAttentionHalfPtrKernel` | attention: core + qk-norm/rope/kv write | 0.274 | 2.4% | 10.0 | 27.4 |
| `flerp::moe::IndexedExpertsKernel<` | experts: routed + shared (IndexedExpertsKernel) | 0.274 | 2.4% | 40.0 | 6.8 |
| `gufo::models::qwen38_flash_next::rocm::RouterTopKKernel<512u>` | router: top-k + MoE epilogue | 0.182 | 1.6% | 40.0 | 4.5 |
| `gufo::hip::SSMConvKernel<false, decltype` | GDN: recurrence + conv + q/g unpack | 0.089 | 0.8% | 30.0 | 3.0 |
| `gufo::models::qwen38_flash_next::rocm::MoeEpilogueKernel` | router: top-k + MoE epilogue | 0.074 | 0.6% | 40.0 | 1.9 |
| `gufo::hip::ResidualAddKernel` | argmax / embed / residual / copies | 0.073 | 0.6% | 80.0 | 0.9 |
| `gufo::hip::FusedQKNormRoPEKvWriteKernel` | attention: core + qk-norm/rope/kv write | 0.039 | 0.3% | 10.0 | 3.9 |
| `gufo::hip::WKQuantA8BlockedWmmaGEMMKernel<128, 64, 4, 8, 1, T12` | other | 0.025 | 0.2% | 0.0 | 0.0 |
| `gufo::hip::UnpackQGKernel` | GDN: recurrence + conv + q/g unpack | 0.013 | 0.1% | 10.0 | 1.3 |
| `gufo::hip::BatchedArgmaxKernel<true, false>` | argmax / embed / residual / copies | 0.010 | 0.1% | 1.0 | 10.5 |
| `__amd_rocclr_copyBuffer` | argmax / embed / residual / copies | 0.004 | 0.0% | 2.0 | 2.0 |
| `gufo::hip::EmbeddingLookupPtrKernel` | argmax / embed / residual / copies | 0.003 | 0.0% | 1.0 | 3.3 |
| `gufo::hip::BatchedArgmaxKernel<false, true>` | argmax / embed / residual / copies | 0.002 | 0.0% | 1.0 | 1.6 |
| `__amd_rocclr_fillBufferUnAligned` | argmax / embed / residual / copies | -0.486 | -4.2% | 0.0 | 0.0 |

## Graph off: 32 decode steps; 617.0 kernels a token, 11.859 GPU ms a token; tool's decode ms/step 14.57

Trace rows: 40630 (64 tokens), 20886 (32 tokens).

### By role

| role | ms a token | share of GPU ms | kernels a token |
|---|---:|---:|---:|
| experts: routed + shared (IndexedExpertsKernel) | 3.403 | 28.7% | 160.0 |
| GDN: fused input projections | 2.818 | 23.8% | 30.0 |
| lm head (output.weight Q6_K GEMV) | 1.882 | 15.9% | 1.0 |
| out projections (attn_output / ssm_out, Q4_K GEMV) | 1.024 | 8.6% | 40.0 |
| attention: fused QKV projections | 0.658 | 5.6% | 10.0 |
| GDN: recurrence + conv + q/g unpack | 0.652 | 5.5% | 70.0 |
| router: gate logits (F32 ffn_gate_inp GEMV) | 0.625 | 5.3% | 40.0 |
| norms | 0.350 | 3.0% | 81.0 |
| attention: core + qk-norm/rope/kv write | 0.298 | 2.5% | 20.0 |
| router: top-k + MoE epilogue | 0.231 | 1.9% | 80.0 |
| argmax / embed / residual / copies | 0.092 | 0.8% | 85.0 |
| other | -0.175 | -1.5% | 0.0 |

### By kernel (decode kernels only)

| kernel | role | ms a token | share | launches a token | us a launch |
|---|---|---:|---:|---:|---:|
| `gufo::hip::Wave32FusedSSMInputProjectionsKernel_1Row<4u>` | GDN: fused input projections | 2.818 | 23.8% | 30.0 | 93.9 |
| `flerp::moe::IndexedExpertsKernel<` | experts: routed + shared (IndexedExpertsKernel) | 2.613 | 22.0% | 120.0 | 21.8 |
| `gufo::hip::Q8KBlockGEMVKernel_2Rows<4u, T0>` | lm head (output.weight Q6_K GEMV) | 1.882 | 15.9% | 1.0 | 1881.6 |
| `gufo::hip::Q8KBlockGEMVKernel_2RowsWideQ4K<8u>` | out projections (attn_output / ssm_out, Q4_K GEMV) | 1.024 | 8.6% | 40.0 | 25.6 |
| `flerp::moe::IndexedExpertsKernel<` | experts: routed + shared (IndexedExpertsKernel) | 0.790 | 6.7% | 40.0 | 19.7 |
| `gufo::hip::Wave32FusedQKVProjectionsKernel_1Row<4u>` | attention: fused QKV projections | 0.658 | 5.6% | 10.0 | 65.8 |
| `flerp::moe::GateLogitsWideKernel` | router: gate logits (F32 ffn_gate_inp GEMV) | 0.625 | 5.3% | 40.0 | 15.6 |
| `_ZN4gufo3hip24DeltaNetRecurrenceKernelITkNS0_6detail25QwenRecurrentStateElementEfLb1ELb1EL` | GDN: recurrence + conv + q/g unpack | 0.552 | 4.7% | 30.0 | 18.4 |
| `gufo::hip::RMSNormKernel<0ul>` | norms | 0.353 | 3.0% | 81.0 | 4.4 |
| `gufo::hip::QwenDecodeOnlineAttentionHalfPtrKernel` | attention: core + qk-norm/rope/kv write | 0.262 | 2.2% | 10.0 | 26.2 |
| `gufo::models::qwen38_flash_next::rocm::RouterTopKKernel<512u>` | router: top-k + MoE epilogue | 0.162 | 1.4% | 40.0 | 4.1 |
| `gufo::hip::SSMConvKernel<false, decltype` | GDN: recurrence + conv + q/g unpack | 0.091 | 0.8% | 30.0 | 3.0 |
| `gufo::hip::ResidualAddKernel` | argmax / embed / residual / copies | 0.075 | 0.6% | 80.0 | 0.9 |
| `gufo::models::qwen38_flash_next::rocm::MoeEpilogueKernel` | router: top-k + MoE epilogue | 0.069 | 0.6% | 40.0 | 1.7 |
| `gufo::hip::FusedQKNormRoPEKvWriteKernel` | attention: core + qk-norm/rope/kv write | 0.037 | 0.3% | 10.0 | 3.7 |
| `gufo::hip::UnpackQGKernel` | GDN: recurrence + conv + q/g unpack | 0.012 | 0.1% | 10.0 | 1.2 |
| `gufo::hip::BatchedArgmaxKernel<true, false>` | argmax / embed / residual / copies | 0.010 | 0.1% | 1.0 | 9.7 |
| `gufo::hip::EmbeddingLookupPtrKernel` | argmax / embed / residual / copies | 0.003 | 0.0% | 1.0 | 2.9 |
| `gufo::hip::BatchedArgmaxKernel<false, true>` | argmax / embed / residual / copies | 0.002 | 0.0% | 1.0 | 1.8 |
| `__amd_rocclr_copyBuffer` | argmax / embed / residual / copies | 0.002 | 0.0% | 2.0 | 0.9 |
| `gufo::hip::WKQuantA8BlockedWmmaGEMMKernel<128, 64, 4, 8, 1, T12` | other | -0.045 | -0.4% | 0.0 | 0.0 |
| `gufo::hip::WKQuantA8BlockedWmmaGEMMKernel<128, 64, 4, 8, 1, T14` | other | -0.129 | -1.1% | 0.0 | 0.0 |
