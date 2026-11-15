# DECODE-KERN3: router, lm head and Q4_K out-projection kernels, memory access only

Criteria `gate/decode-kern3.toml` (committed before the GPU runs). Backend kbench: bench box K (ltsp-k, gfx1151, balanced 85 W, quiet cooling, weak die contact), pod-kbench k_run, one lease for the whole stage, ROCm 7.2.4 runtime libs (the binary is built with therock 10.1 clang on control-node); each case's process starts at Tctl <= 50 C (<= 90 s wait) and K's loadavg <= 6; the bench alternates ref/new rounds inside one process, so no ABBA process order is added.

## Kernel (kernel-cold): FINAL source, cache-cold ([kernel_cold]): the run the continue rule is judged on

kernel-cold: job 20261113T212202Z-run-lcold (/data/pod/kbench/runs/7a1a58e793/20261113T212202Z-run-lcold), thermals {'tctl_start': 21.5, 'tctl_peak': 38.1, 'w_mean': 28.6, 'w_peak': 67.1, 'hot_frac': 0.0}. Method: bench box K (ltsp-k, gfx1151, balanced 85 W, quiet cooling, weak die contact), pod-kbench k_run, one lease for the whole stage, ROCm 7.2.4 runtime libs (the binary is built with therock 10.1 clang on control-node); each case's process starts at Tctl <= 50 C (<= 90 s wait) and K's loadavg <= 6; the bench alternates ref/new rounds inside one process, so no ABBA process order is added.

28 (case, x, tokens) results, 204390 outputs compared; results with any differing byte: 0 -> **PASS** (byte equality). Cases off-load or failed: 0 

| case | type | rows x k | outputs differing (all x, tokens) | ref GB/s | new GB/s | new / ref |
|---|---|---|---|---|---|---|
| syn-router-4096x512 | f32 | 513 x 4096 | 0 of 155952 | 85.7 (85.4..86.2) | 194.0 (190.4..194.5) | 2.264 |
| syn-router-noshared | f32 | 128 x 2048 | 0 of 2304 | 34.7 (34.5..34.8) | 132.5 (131.9..132.9) | 3.822 |
| syn-router-hidden1536 | f32 | 65 x 1536 | 0 of 1170 | 24.3 (22.8..24.4) | 24.3 (24.1..24.6) | 1.002 |
| syn-q4k-oddM | q4k | 1001 x 2048 | 0 of 2002 | 115.6 (110.5..115.7) | 115.5 (115.3..115.8) | 1.000 |
| syn-q6k-oddM | q6k | 1001 x 2048 | 0 of 2002 | 145.7 (143.0..146.0) | 145.4 (143.0..145.6) | 0.998 |
| syn-q4k-k256 | q4k | 4096 x 256 | 0 of 8192 | 67.0 (63.3..67.2) | 73.8 (73.5..74.1) | 1.101 |
| syn-q6k-k256 | q6k | 4096 x 256 | 0 of 8192 | 73.7 (71.3..73.9) | 73.5 (73.1..74.0) | 0.997 |
| syn-q4k-k5120 | q4k | 6144 x 5120 | 0 of 12288 | 162.0 (160.9..162.8) | 173.3 (171.5..173.9) | 1.070 |
| syn-q6k-k5120 | q6k | 6144 x 5120 | 0 of 12288 | 194.4 (193.6..194.6) | 193.6 (191.7..194.4) | 0.996 |

