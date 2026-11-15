# Interface gate: moe.router/-/bf16

Host control-node, 2026-11-10T23:39:08Z. Config `gate/interface/moe-router.toml` (sha256 c1fa0139eb473c36). Consumer API: `strix::kernels::moe_router(logits, ld = E + 1, M, E, k, nullptr, 0, 0, ids, coef, Act::F32, err, stream)`.

Overall: **FAIL**

| provider | module that runs | built | parity vs CPU reference | adapter launches (foreign / origin) | gate |
|---|---|---|---|---|---|
| strixite | `kernel.strixite.moe_router` | yes | differs from the CPU reference: max abs 0, max rel 2.08e-07, worst error 0.109 of the gate's bound | - | PASS |
| gufo | `kernel.gufo.qfn_kernels` | yes | differs from the CPU reference: max abs 0, max rel 3.19e-07, worst error 0.167 of the gate's bound | 54574 / 0 | PASS |
| hipfire | `kernel.hipfire.moe` | yes | differs from the CPU reference: max abs 0, max rel 2.8e-07, worst error 0.147 of the gate's bound | 27287 / 0 | FAIL |

Time per call (median of the repeats, launches queued back to back then one synchronize). A call is one launch of the consumer API; adapters may need several kernel launches for it. GB/s is the size of the call's operands divided by that time: for operands of a few MB it is not a memory bandwidth (they stay in the GPU's caches between launches), for the large shapes it is.

| provider | case | shape | median us | min..max us | GB/s | kernel launches | host load (1 min) |
|---|---|---|---|---|---|---|---|
| strixite | k10-bench-decode-1 | M=1, E=512, k=10 (one decode token) | 28.12 | 28.08..28.13 | 0.1 | 1 per call | 11.91 |
| strixite | k10-bench-verify-16 | M=16, E=512, k=10 (a verify window) | 28.57 | 28.55..28.61 | 1.2 | 1 per call | 11.91 |
| strixite | k10-bench-prefill-2048 | M=2048, E=512, k=10 (a prefill chunk) | 100.34 | 100.21..101.19 | 43.4 | 1 per call | 11.91 |
| strixite | k8-bench-decode-1 | M=1, E=512, k=8 (one decode token) | 25.53 | 25.50..25.57 | 0.1 | 1 per call | 11.91 |
| strixite | k8-bench-verify-16 | M=16, E=512, k=8 (a verify window) | 25.96 | 25.94..25.98 | 1.3 | 1 per call | 11.91 |
| strixite | k8-bench-prefill-2048 | M=2048, E=512, k=8 (a prefill chunk) | 90.61 | 90.08..90.88 | 47.7 | 1 per call | 11.91 |
| gufo | k10-bench-decode-1 | M=1, E=512, k=10 (one decode token) | 36.63 | 36.55..36.77 | 0.1 | 2 per call | 11.91 |
| gufo | k10-bench-verify-16 | M=16, E=512, k=10 (a verify window) | 37.14 | 37.09..37.17 | 0.9 | 2 per call | 11.91 |
| gufo | k10-bench-prefill-2048 | M=2048, E=512, k=10 (a prefill chunk) | 344.02 | 343.45..344.22 | 12.7 | 2 per call | 11.91 |
| gufo | k8-bench-decode-1 | M=1, E=512, k=8 (one decode token) | 33.86 | 33.84..33.91 | 0.1 | 2 per call | 11.91 |
| gufo | k8-bench-verify-16 | M=16, E=512, k=8 (a verify window) | 34.42 | 29.94..34.56 | 1.0 | 2 per call | 11.91 |
| gufo | k8-bench-prefill-2048 | M=2048, E=512, k=8 (a prefill chunk) | 73.43 | 62.71..76.06 | 58.9 | 2 per call | 11.91 |
| hipfire | k8-bench-decode-1 | M=1, E=512, k=8 (one decode token) | 11.81 | 11.79..11.81 | 0.2 | M + 1 per call | 11.44 |
| hipfire | k8-bench-verify-16 | M=16, E=512, k=8 (a verify window) | 135.01 | 134.97..135.55 | 0.3 | M + 1 per call | 11.44 |
| hipfire | k8-bench-prefill-2048 | M=2048, E=512, k=8 (a prefill chunk) | 17078.17 | 17070.88..17111.79 | 0.3 | M + 1 per call | 11.44 |

Providers against each other (raw outputs of the parity cases):

| a | b | cases identical | cases differing | largest gap |
|---|---|---|---|---|
| strixite | gufo | 0 | 16 | 0 |
| strixite | hipfire | 0 | 8 | 0 |
| gufo | hipfire | 0 | 8 | 0 |

Notes:

- gufo / k10-near-ties-8: differs from the reference (reported, not a pass criterion): mismatched=64, max_abs=0, structure_ok=True
- gufo / k8-near-ties-8: differs from the reference (reported, not a pass criterion): mismatched=25, max_abs=0, structure_ok=True
- hipfire / k8-near-ties-8: differs from the reference (reported, not a pass criterion): mismatched=20, max_abs=0, structure_ok=True
- hipfire: FAIL - k8-exact-ties-64: required order-bound, got bit_identical=False, mismatched=71, err_over_bound=0.0449996107, structure_ok=False, max_abs=0
- hipfire: FAIL - k8-boundary-tie-4: required order-bound, got bit_identical=False, mismatched=4, err_over_bound=0, structure_ok=False, max_abs=0
