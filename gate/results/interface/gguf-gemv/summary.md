# Interface gate: linear.decode/gguf-kquant/f32

Host control-node, 2026-11-11T00:09:19Z. Config `gate/interface/gguf-gemv.toml` (sha256 15f9e3b9ae327270). Consumer API: `gufo::hip::LaunchQ8KBlockGEMV(A, type, x, y, M, K, stream)`.

Overall: **PASS**

| provider | module that runs | built | parity vs CPU reference | adapter launches (foreign / origin) | gate |
|---|---|---|---|---|---|
| gufo | `kernel.gufo.qwen_linear_decode` | yes | differs from the CPU reference: max abs 0.0938, max rel 0.0159, worst error 0.00274 of the gate's bound; a CPU emulation of the kernel's documented order (gufo-block-sums (mul-add, mul-add), gufo-block-sums (fma, fma), gufo-sub16 (fma dot, fma scale)) reproduces its bits in 13 of 14 cases | - | PASS |
| hipfire | `kernel.hipfire.gemv_ggml` | yes | differs from the CPU reference: max abs 0.188, max rel 0.00532, worst error 0.00181 of the gate's bound; a CPU emulation of the kernel's documented order (hipfire-lanes (fma)) reproduces its bits in 14 of 14 cases | 21626 / 0 | PASS |

Time per call (median of the repeats, launches queued back to back then one synchronize). A call is one launch of the consumer API; adapters may need several kernel launches for it. GB/s is the size of the call's operands divided by that time: for operands of a few MB it is not a memory bandwidth (they stay in the GPU's caches between launches), for the large shapes it is.

| provider | case | shape | median us | min..max us | GB/s | kernel launches | host load (1 min) |
|---|---|---|---|---|---|---|---|
| gufo | q8_0-bench-2560x2560 | M=2560, K=2560 q8_0 (7.0 MB of weights: a decode projection) | 13.48 | 13.18..13.84 | 517.9 | 1 | 9.21 |
| gufo | q8_0-bench-16384x2560 | M=16384, K=2560 q8_0 (44.6 MB: the large shape; a GEMV has one x row, prefill is a GEMM) | 196.56 | 191.26..200.19 | 227.1 | 1 | 9.21 |
| gufo | q4k-bench-2560x2560 | M=2560, K=2560 q4_k (3.7 MB of weights: a decode projection) | 14.22 | 13.64..14.34 | 260.8 | 1 | 9.21 |
| gufo | q4k-bench-16384x2560 | M=16384, K=2560 q4_k (23.6 MB: the large shape; a GEMV has one x row, prefill is a GEMM) | 77.99 | 72.50..78.90 | 303.5 | 1 | 9.21 |
| hipfire | q8_0-bench-2560x2560 | M=2560, K=2560 q8_0 (7.0 MB of weights: a decode projection) | 16.83 | 15.41..17.10 | 414.8 | 1 | 9.21 |
| hipfire | q8_0-bench-16384x2560 | M=16384, K=2560 q8_0 (44.6 MB: the large shape; a GEMV has one x row, prefill is a GEMM) | 206.06 | 203.04..209.68 | 216.6 | 1 | 9.21 |
| hipfire | q4k-bench-2560x2560 | M=2560, K=2560 q4_k (3.7 MB of weights: a decode projection) | 17.02 | 16.15..17.26 | 217.8 | 1 | 9.21 |
| hipfire | q4k-bench-16384x2560 | M=16384, K=2560 q4_k (23.6 MB: the large shape; a GEMV has one x row, prefill is a GEMM) | 89.91 | 85.99..93.56 | 263.2 | 1 | 9.21 |

Providers against each other (raw outputs of the parity cases):

| a | b | cases identical | cases differing | largest gap |
|---|---|---|---|---|
| gufo | hipfire | 0 | 14 | 0.188 |

Notes:

- gufo: no order emulation reproduces q8_0-odd-2559rows-k2560 (within the bound; the order of that path is not shown)
