# Interface gate: linear.decode/strixw-q4/bf16

Host control-node, 2026-11-24T19:04:48Z. Config `gate/interface/strixw-decode.toml` (sha256 efbb77a111e23a78). Consumer API: `strix::kernels::linear_q4 / linear_q5 / linear_q6 / linear_q8(x, view, y, M, act, out_act, stream)`. Re-judged 2026-11-24T19:14:32Z from the recorded driver output (nothing re-run).

Overall: **PASS**

| provider | module that runs | built | parity vs CPU reference | adapter launches (foreign / origin) | gate |
|---|---|---|---|---|---|
| strixite-q4 | `kernel.strixite.linear_q4` | yes | differs from the CPU reference: max abs 0.000854, max rel 0.00606, worst error 1.13 of the gate's bound | - | PASS |
| strixite-q8 | `kernel.strixite.linear_q8` | yes | differs from the CPU reference: max abs 0.000854, max rel 0.00613, worst error 1.23 of the gate's bound | - | PASS |
| strixite-q56 | `kernel.strixite.linear_q56` | yes | differs from the CPU reference: max abs 0.00104, max rel 0.00469, worst error 1.67 of the gate's bound | - | PASS |
| flerp-q4p | `kernel.flerp.linear_q4p` | yes | differs from the CPU reference: max abs 0.000854, max rel 0.00606, worst error 1.13 of the gate's bound | 9013 / 0 | PASS |
| flerp-q8p | `kernel.flerp.linear_q8p` | yes | differs from the CPU reference: max abs 0.000854, max rel 0.00613, worst error 1.23 of the gate's bound | 9013 / 0 | PASS |

Time per call (median of the repeats, launches queued back to back then one synchronize). A call is one launch of the consumer API; adapters may need several kernel launches for it. GB/s is the size of the call's operands divided by that time: for operands of a few MB it is not a memory bandwidth (they stay in the GPU's caches between launches), for the large shapes it is.

| provider | case | shape | median us | min..max us | GB/s | kernel launches | host load (1 min) |
|---|---|---|---|---|---|---|---|
| strixite-q4 | q4-bench-m1-4096x2048 | M=1, N=4096, K=2048, G=64, BF16 in and out: a decode projection | 54.91 | 54.80..55.12 | 86.2 | 1 | 21.09 |
| strixite-q4 | q4-bench-m8-4096x2048 | M=8, N=4096, K=2048, G=64, BF16 in and out: a decode projection | 46.66 | 37.18..148.81 | 103.2 | 1 | 21.09 |
| strixite-q8 | q8-bench-m1-4096x2048 | M=1, N=4096, K=2048, G=64, BF16 in and out: a decode projection | 21.65 | 20.45..21.93 | 412.2 | 1 | 21.09 |
| strixite-q8 | q8-bench-m8-4096x2048 | M=8, N=4096, K=2048, G=64, BF16 in and out: a decode projection | 56.35 | 54.51..56.73 | 159.9 | 1 | 21.09 |
| strixite-q56 | q5-bench-m1-4096x2048 | M=1, N=4096, K=2048, G=64, BF16 in and out: a decode projection | 32.36 | 29.31..33.16 | 178.6 | 1 | 21.09 |
| strixite-q56 | q5-bench-m8-4096x2048 | M=8, N=4096, K=2048, G=64, BF16 in and out: a decode projection | 133.22 | 107.73..134.38 | 44.0 | 1 | 21.09 |
| strixite-q56 | q6-bench-m1-4096x2048 | M=1, N=4096, K=2048, G=64, BF16 in and out: a decode projection | 23.42 | 21.04..23.87 | 291.5 | 1 | 21.09 |
| strixite-q56 | q6-bench-m8-4096x2048 | M=8, N=4096, K=2048, G=64, BF16 in and out: a decode projection | 88.83 | 88.49..89.58 | 77.8 | 1 | 21.09 |
| flerp-q4p | q4-bench-m1-4096x2048 | M=1, N=4096, K=2048, G=64, BF16 in and out: a decode projection | 22.25 | 19.82..23.58 | 212.7 | 1 | 21.32 |
| flerp-q4p | q4-bench-m8-4096x2048 | M=8, N=4096, K=2048, G=64, BF16 in and out: a decode projection | 54.36 | 51.20..57.99 | 88.6 | 1 | 21.32 |
| flerp-q8p | q8-bench-m1-4096x2048 | M=1, N=4096, K=2048, G=64, BF16 in and out: a decode projection | 20.31 | 19.61..24.16 | 439.4 | 1 | 21.32 |
| flerp-q8p | q8-bench-m8-4096x2048 | M=8, N=4096, K=2048, G=64, BF16 in and out: a decode projection | 42.78 | 42.16..51.62 | 210.7 | 1 | 21.32 |

Providers against each other (raw outputs of the parity cases):

| a | b | cases identical | cases differing | largest gap |
|---|---|---|---|---|
| strixite-q4 | strixite-q8 | 0 | 0 | 0 |
| strixite-q4 | strixite-q56 | 0 | 0 | 0 |
| strixite-q4 | flerp-q4p | 7 | 0 | 0 |
| strixite-q4 | flerp-q8p | 0 | 0 | 0 |
| strixite-q8 | strixite-q56 | 0 | 0 | 0 |
| strixite-q8 | flerp-q4p | 0 | 0 | 0 |
| strixite-q8 | flerp-q8p | 7 | 0 | 0 |
| strixite-q56 | flerp-q4p | 0 | 0 | 0 |
| strixite-q56 | flerp-q8p | 0 | 0 | 0 |
| flerp-q4p | flerp-q8p | 0 | 0 | 0 |
