# ORNITH-KERN3: DECODE-KERN3 kernels at Ornith-1.5 397B dims (synthetic codes), backend kbench

Criteria `gate/ornith-kern3.toml` (committed before the GPU run). Bench build/kern3/decode-kern3 (catalogue at 83d5366).

Run: kernel-cold: job 20261113T212112Z-run-lcold (/data/pod/kbench/runs/7a1a58e793/20261113T212112Z-run-lcold), thermals {'tctl_start': 20.4, 'tctl_peak': 33.6, 'w_mean': 23.0, 'w_peak': 44.1, 'hot_frac': 0.0}. Method: bench box K (ltsp-k, gfx1151, balanced 85 W, quiet cooling, weak die contact), pod-kbench k_run, one lease for the whole stage, ROCm 7.2.4 runtime libs (the binary is built with therock 10.1 clang on control-node); each case's process starts at Tctl <= 50 C (<= 90 s wait) and K's loadavg <= 6; the bench alternates ref/new rounds inside one process, so no ABBA process order is added.

26 (case, x, tokens) results, 1300934 outputs compared; with any differing byte: 0 -> **PASS**. Cases off-load or failed: 0 

| case | rows x k | differing | ref GB/s | new GB/s | new/ref | ref us | new us |
|---|---|---|---|---|---|---|---|
| router-4096x512 | 513 x 4096 | 0 of 1217862 | 85.5 (84.9..85.8) | 194.0 (189.3..194.3) | 2.269 | 98.49 | 43.42 |
| outproj-q4k-4096x8192 | 4096 x 8192 | 0 of 8192 | 126.7 (126.4..127.2) | 161.5 (160.1..161.6) | 1.275 | 149.23 | 117.07 |
| ssm-alpha-q4k-64x4096 | 64 x 4096 | 0 of 128 | - | - | - | - | - |
| attn-kv-q4k-512x4096 | 512 x 4096 | 0 of 1024 | - | - | - | - | - |
| attn-gate-q4k-8192x4096 | 8192 x 4096 | 0 of 16384 | - | - | - | - | - |
| attn-q-q4k-16384x4096 | 16384 x 4096 | 0 of 32768 | - | - | - | - | - |
| attn-qkv-q4k-12288x4096 | 12288 x 4096 | 0 of 24576 | - | - | - | - | - |

## Projection (computed, not measured; criteria [projection])

Saving a block 0.087 ms (router 0.055 + outproj 0.032).
Stage (15 blocks): 18.5 -> 17.19 ms a step.
World 4 token: 83.0 -> 77.77 ms = 12.05 -> 12.86 tok/s (projected).
