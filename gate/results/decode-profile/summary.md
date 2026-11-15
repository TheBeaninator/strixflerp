# 35B world-1 decode: per-kernel time of one step (rocprofv3 kernel trace)

Runner `gate/decode_profile.py` (+ `gate/decode_profile_run.sh` for the lease). Binary: the R1-fold `after` build (`build/xfer-r1-fold/after`), `--staged loopback`, world 1, prompt `code` (20 tokens), context 4096, control-node GPU.
Decode kernels = trace(64 tokens) minus trace(32 tokens), same prompt: 32 decode steps a graph mode, no prefill or load in them. GPU ms are kernel execution time (End - Start), summed; the step's wall time is the unprofiled decode ms a step.

## Runs

| case | rc | s | load start -> end | decode ms/step (tool) | tok/s (tool) | decode calls |
|---|---|---|---|---|---|---|
| on-64 | 0 | 98.8 | 5.92 -> 7.76 | 20.13 | 49.51 | [64] |
| on-32 | 0 | 94.1 | 7.76 -> 7.75 | 19.65 | 50.72 | [32] |
| off-64 | 0 | 88.3 | 7.75 -> 7.96 | 24.29 | 40.96 | [64] |
| off-32 | 0 | 83.3 | 7.96 -> 9.07 | 16.71 | 59.43 | [32] |
| on-64-noprof | 0 | 98.1 | 9.07 -> 8.96 | 16.32 | 61.06 | [64] |

## Graph on: 32 decode steps; 617.0 kernels a token, 16.734 GPU ms a token; tool's decode ms/step 16.32

Trace rows: 40630 (64 tokens), 20886 (32 tokens).

### By role

| role | ms a token | share of GPU ms | kernels a token |
|---|---:|---:|---:|
| experts: routed + shared (IndexedExpertsKernel) | 4.388 | 26.2% | 160.0 |
| GDN: fused input projections | 3.365 | 20.1% | 30.0 |
| lm head (output.weight Q6_K GEMV) | 2.934 | 17.5% | 1.0 |
| out projections (attn_output / ssm_out, Q4_K GEMV) | 1.710 | 10.2% | 40.0 |
| router: gate logits (F32 ffn_gate_inp GEMV) | 1.693 | 10.1% | 40.0 |
| attention: fused QKV projections | 0.802 | 4.8% | 10.0 |
| GDN: recurrence + conv + q/g unpack | 0.648 | 3.9% | 70.0 |
| attention: core + qk-norm/rope/kv write | 0.547 | 3.3% | 20.0 |
| norms | 0.350 | 2.1% | 81.0 |
| router: top-k + MoE epilogue | 0.224 | 1.3% | 80.0 |
| argmax / embed / residual / copies | 0.095 | 0.6% | 85.0 |
| other | -0.021 | -0.1% | 0.0 |

### By kernel (decode kernels only)

| kernel | role | ms a token | share | launches a token | us a launch |
|---|---|---:|---:|---:|---:|
| `flerp::moe::IndexedExpertsKernel<` | experts: routed + shared (IndexedExpertsKernel) | 3.574 | 21.4% | 120.0 | 29.8 |
| `gufo::hip::Wave32FusedSSMInputProjectionsKernel_1Row<4u>` | GDN: fused input projections | 3.365 | 20.1% | 30.0 | 112.2 |
| `gufo::hip::Q8KBlockGEMVKernel_2Rows<4u, T0>` | lm head (output.weight Q6_K GEMV) | 2.934 | 17.5% | 1.0 | 2933.8 |
| `gufo::hip::Q8KBlockGEMVKernel_2Rows<4u, T12` | out projections (attn_output / ssm_out, Q4_K GEMV) | 1.710 | 10.2% | 40.0 | 42.8 |
| `flerp::moe::GateLogitsKernel` | router: gate logits (F32 ffn_gate_inp GEMV) | 1.693 | 10.1% | 40.0 | 42.3 |
| `flerp::moe::IndexedExpertsKernel<` | experts: routed + shared (IndexedExpertsKernel) | 0.814 | 4.9% | 40.0 | 20.4 |
| `gufo::hip::Wave32FusedQKVProjectionsKernel_1Row<4u>` | attention: fused QKV projections | 0.802 | 4.8% | 10.0 | 80.2 |
| `_ZN4gufo3hip24DeltaNetRecurrenceKernelITkNS0_6detail25QwenRecurrentStateElementEfLb1ELb1EL` | GDN: recurrence + conv + q/g unpack | 0.558 | 3.3% | 30.0 | 18.6 |
| `gufo::hip::QwenDecodeOnlineAttentionHalfPtrKernel` | attention: core + qk-norm/rope/kv write | 0.510 | 3.1% | 10.0 | 51.0 |
| `gufo::hip::RMSNormKernel<0ul>` | norms | 0.351 | 2.1% | 81.0 | 4.3 |
| `gufo::models::qwen38_flash_next::rocm::RouterTopKKernel<512u>` | router: top-k + MoE epilogue | 0.157 | 0.9% | 40.0 | 3.9 |
| `gufo::hip::SSMConvKernel<false, decltype` | GDN: recurrence + conv + q/g unpack | 0.079 | 0.5% | 30.0 | 2.6 |
| `gufo::hip::ResidualAddKernel` | argmax / embed / residual / copies | 0.072 | 0.4% | 80.0 | 0.9 |
| `gufo::models::qwen38_flash_next::rocm::MoeEpilogueKernel` | router: top-k + MoE epilogue | 0.067 | 0.4% | 40.0 | 1.7 |
| `gufo::hip::FusedQKNormRoPEKvWriteKernel` | attention: core + qk-norm/rope/kv write | 0.037 | 0.2% | 10.0 | 3.7 |
| `gufo::hip::UnpackQGKernel` | GDN: recurrence + conv + q/g unpack | 0.012 | 0.1% | 10.0 | 1.2 |
| `gufo::hip::BatchedArgmaxKernel<true, false>` | argmax / embed / residual / copies | 0.010 | 0.1% | 1.0 | 10.4 |
| `__amd_rocclr_copyBuffer` | argmax / embed / residual / copies | 0.010 | 0.1% | 2.0 | 5.1 |
| `gufo::hip::EmbeddingLookupPtrKernel` | argmax / embed / residual / copies | 0.003 | 0.0% | 1.0 | 3.3 |
| `gufo::hip::BatchedArgmaxKernel<false, true>` | argmax / embed / residual / copies | 0.001 | 0.0% | 1.0 | 1.4 |
| `gufo::hip::WKQuantA8BlockedWmmaGEMMKernel<128, 64, 4, 8, 1, T14` | other | -0.005 | -0.0% | 0.0 | 0.0 |
| `gufo::hip::WKQuantA8BlockedWmmaGEMMKernel<128, 64, 4, 8, 1, T12` | other | -0.016 | -0.1% | 0.0 | 0.0 |

## Graph off: 32 decode steps; 617.0 kernels a token, 23.821 GPU ms a token; tool's decode ms/step 24.29

Trace rows: 40630 (64 tokens), 20886 (32 tokens).

### By role

| role | ms a token | share of GPU ms | kernels a token |
|---|---:|---:|---:|
| experts: routed + shared (IndexedExpertsKernel) | 8.557 | 35.9% | 160.0 |
| GDN: fused input projections | 4.506 | 18.9% | 30.0 |
| lm head (output.weight Q6_K GEMV) | 3.757 | 15.8% | 1.0 |
| router: gate logits (F32 ffn_gate_inp GEMV) | 1.774 | 7.4% | 40.0 |
| attention: fused QKV projections | 1.751 | 7.4% | 10.0 |
| out projections (attn_output / ssm_out, Q4_K GEMV) | 1.739 | 7.3% | 40.0 |
| GDN: recurrence + conv + q/g unpack | 0.661 | 2.8% | 70.0 |
| norms | 0.397 | 1.7% | 81.0 |
| attention: core + qk-norm/rope/kv write | 0.326 | 1.4% | 20.0 |
| router: top-k + MoE epilogue | 0.259 | 1.1% | 80.0 |
| argmax / embed / residual / copies | 0.093 | 0.4% | 85.0 |
| other | 0.001 | 0.0% | 0.0 |

### By kernel (decode kernels only)

| kernel | role | ms a token | share | launches a token | us a launch |
|---|---|---:|---:|---:|---:|
| `flerp::moe::IndexedExpertsKernel<` | experts: routed + shared (IndexedExpertsKernel) | 7.435 | 31.2% | 120.0 | 62.0 |
| `gufo::hip::Wave32FusedSSMInputProjectionsKernel_1Row<4u>` | GDN: fused input projections | 4.506 | 18.9% | 30.0 | 150.2 |
| `gufo::hip::Q8KBlockGEMVKernel_2Rows<4u, T0>` | lm head (output.weight Q6_K GEMV) | 3.757 | 15.8% | 1.0 | 3757.1 |
| `flerp::moe::GateLogitsKernel` | router: gate logits (F32 ffn_gate_inp GEMV) | 1.774 | 7.4% | 40.0 | 44.3 |
| `gufo::hip::Wave32FusedQKVProjectionsKernel_1Row<4u>` | attention: fused QKV projections | 1.751 | 7.4% | 10.0 | 175.1 |
| `gufo::hip::Q8KBlockGEMVKernel_2Rows<4u, T12` | out projections (attn_output / ssm_out, Q4_K GEMV) | 1.739 | 7.3% | 40.0 | 43.5 |
| `flerp::moe::IndexedExpertsKernel<` | experts: routed + shared (IndexedExpertsKernel) | 1.122 | 4.7% | 40.0 | 28.1 |
| `_ZN4gufo3hip24DeltaNetRecurrenceKernelITkNS0_6detail25QwenRecurrentStateElementEfLb1ELb1EL` | GDN: recurrence + conv + q/g unpack | 0.564 | 2.4% | 30.0 | 18.8 |
| `gufo::hip::RMSNormKernel<0ul>` | norms | 0.397 | 1.7% | 81.0 | 4.9 |
| `gufo::hip::QwenDecodeOnlineAttentionHalfPtrKernel` | attention: core + qk-norm/rope/kv write | 0.287 | 1.2% | 10.0 | 28.7 |
| `gufo::models::qwen38_flash_next::rocm::RouterTopKKernel<512u>` | router: top-k + MoE epilogue | 0.183 | 0.8% | 40.0 | 4.6 |
| `gufo::hip::SSMConvKernel<false, decltype` | GDN: recurrence + conv + q/g unpack | 0.084 | 0.4% | 30.0 | 2.8 |
| `gufo::hip::ResidualAddKernel` | argmax / embed / residual / copies | 0.082 | 0.3% | 80.0 | 1.0 |
| `gufo::models::qwen38_flash_next::rocm::MoeEpilogueKernel` | router: top-k + MoE epilogue | 0.076 | 0.3% | 40.0 | 1.9 |
| `gufo::hip::FusedQKNormRoPEKvWriteKernel` | attention: core + qk-norm/rope/kv write | 0.039 | 0.2% | 10.0 | 3.9 |
| `gufo::hip::UnpackQGKernel` | GDN: recurrence + conv + q/g unpack | 0.013 | 0.1% | 10.0 | 1.3 |
| `gufo::hip::BatchedArgmaxKernel<true, false>` | argmax / embed / residual / copies | 0.010 | 0.0% | 1.0 | 10.3 |
| `gufo::hip::EmbeddingLookupPtrKernel` | argmax / embed / residual / copies | 0.003 | 0.0% | 1.0 | 2.9 |
| `gufo::hip::BatchedArgmaxKernel<false, true>` | argmax / embed / residual / copies | 0.002 | 0.0% | 1.0 | 2.2 |
| `__amd_rocclr_copyBuffer` | argmax / embed / residual / copies | -0.000 | -0.0% | 2.0 | -0.1 |

## Reading (written by hand 2026-11-13, from the graph-on table)

- The graph-on step is GPU-bound: 16.73 GPU ms of kernels (profiled) against 16.32 ms unprofiled wall, 617 launches a
  token. Launch gaps are not a target under the graph. The graph-off profiled 64-token run (24.3 ms/step against
  16.7 for off-32) was slowed by something else on the host. Its absolute numbers are not used.
- Bandwidth from census bytes: router `ffn_gate_inp` F32 [2048,256] = 2 MiB x 40 in 1.693 ms = **50 GB/s**.
  `output.weight` Q6_K (417 MB) in 2.934 ms = **142 GB/s**. Q4_K out projections ([4096,2048] = 4.7 MB) at 42.8 us =
  **110 GB/s**. Expert reads (IndexedExpertsKernel) are 4.39 ms.
- The +20 % target (68.3 tok/s, 14.63 ms) needs 1.69 ms off 16.32.

| rank | target | now ms | expected ms saved |
|---|---|---:|---:|
| 1 | router GateLogitsKernel: F32 GEMV at 50 GB/s -> ~180 GB/s (or bf16 weights, or fuse into the preceding norm) | 1.69 | ~1.2 |
| 2 | lm head Q6_K GEMV (1 launch): 142 -> ~190 GB/s, e.g. the R1 split-K recipe | 2.93 | ~0.7 |
| 3 | Q4_K out projections (40 x Q8KBlockGEMV T12): 110 -> ~180 GB/s, e.g. the R1 recipe | 1.71 | ~0.65 |
| (4) | GDN fused input projections (30 launches, 112 us each). Bytes a layer not measured, so the saving is unestimated | 3.37 | ? |

Ranks 1 to 3 together are ~2.55 ms -> ~13.8 ms/step, ~72 tok/s (+27 % over 56.95).
