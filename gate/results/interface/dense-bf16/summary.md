# Interface gate: linear.dense/bf16/bf16

Host control-node, 2026-11-10T23:38:40Z. Config `gate/interface/dense-bf16.toml` (sha256 fc56ca76e766889e). Consumer API: `strix::kernels::linear_bf16w(x, w, y, M, N, K, Act::BF16, Act::F32, stream)`.

Overall: **PASS**

| provider | module that runs | built | parity vs CPU reference | adapter launches (foreign / origin) | gate |
|---|---|---|---|---|---|
| strixite | `kernel.strixite.linear` | yes | differs from the CPU reference: max abs 0.0625, max rel 0.00323, worst error 0.000506 of the gate's bound; a CPU emulation of the kernel's documented order (strixite-rows, strixite-rows = strixfold-lanes (the same order at this K)) reproduces its bits in 11 of 12 cases | - | PASS |
| strixfold | `kernel.strixfold.dense_bf16` | yes | differs from the CPU reference: max abs 0.0625, max rel 0.00403, worst error 0.000473 of the gate's bound; a CPU emulation of the kernel's documented order (strixfold-lanes, strixite-rows = strixfold-lanes (the same order at this K), strixite-rows) reproduces its bits in 11 of 12 cases | 28819 / 2 | PASS |

Time per call (median of the repeats, launches queued back to back then one synchronize). A call is one launch of the consumer API; adapters may need several kernel launches for it. GB/s is the size of the call's operands divided by that time: for operands of a few MB it is not a memory bandwidth (they stay in the GPU's caches between launches), for the large shapes it is.

| provider | case | shape | median us | min..max us | GB/s | kernel launches | host load (1 min) |
|---|---|---|---|---|---|---|---|
| strixite | bench-decode-1x513x2560 | M=1, N=513, K=2560 (router logits, one decode token) | 5.56 | 5.54..5.68 | 473.8 | 1 | 15.59 |
| strixite | bench-verify-16x513x2560 | M=16, N=513, K=2560 (a verify window) | 27.07 | 26.43..27.21 | 101.3 | 1 | 15.59 |
| strixite | bench-128x513x2560 | M=128, N=513, K=2560 (the widest window StrixFold's kernel is specified for; strixite's prefill takes its WMMA kernel) | 174.97 | 172.01..177.09 | 20.3 | 1 | 15.59 |
| strixfold | bench-decode-1x513x2560 | M=1, N=513, K=2560 (router logits, one decode token) | 7.06 | 7.05..7.33 | 373.0 | 1 | 15.59 |
| strixfold | bench-verify-16x513x2560 | M=16, N=513, K=2560 (a verify window) | 25.67 | 24.97..26.54 | 106.8 | 1 | 15.59 |
| strixfold | bench-128x513x2560 | M=128, N=513, K=2560 (the widest window StrixFold's kernel is specified for; strixite's prefill takes its WMMA kernel) | 204.40 | 199.05..208.81 | 17.3 | 1 | 15.59 |

Providers against each other (raw outputs of the parity cases):

| a | b | cases identical | cases differing | largest gap |
|---|---|---|---|---|
| strixite | strixfold | 4 | 8 | 0.0781 |

Notes:

- strixite: no order emulation reproduces unaligned-w-2x513x2560 (within the bound; the order of that path is not shown)
- strixfold: no order emulation reproduces unaligned-w-2x513x2560 (within the bound; the order of that path is not shown)
