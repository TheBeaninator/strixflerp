# ATTN-PROJ: Ornith-1.5 397B K=4096 input projections (fused q/k/v and GDN qkv/gate/alpha/beta), memory access only

Criteria `gate/attn-proj.toml` (committed before any GPU run). Backend kbench: bench box K (ltsp-k, gfx1151, balanced 85 W, quiet cooling, weak die contact), pod-kbench k_run, one lease for the whole stage, ROCm 7.2.4 runtime libs (the binary is built with therock 10.1 clang on control-node); each case's process starts at Tctl <= 50 C (<= 90 s wait) and K's loadavg <= 6; the bench alternates ref/new rounds inside one process, so no ABBA process order is added. Old = the one-row fused launchers as at the criteria commit (gate/attn_proj_ref.hip); new = the catalogue's.

kernel-cold: job 20261113T212131Z-run-lcold (/data/pod/kbench/runs/7a1a58e793/20261113T212131Z-run-lcold), thermals {'tctl_start': 20.9, 'tctl_peak': 48.6, 'w_mean': 78.2, 'w_peak': 105.0, 'hot_frac': 0.0}. Method: bench box K (ltsp-k, gfx1151, balanced 85 W, quiet cooling, weak die contact), pod-kbench k_run, one lease for the whole stage, ROCm 7.2.4 runtime libs (the binary is built with therock 10.1 clang on control-node); each case's process starts at Tctl <= 50 C (<= 90 s wait) and K's loadavg <= 6; the bench alternates ref/new rounds inside one process, so no ABBA process order is added.

## [kernel] final source, cache-cold

204 (case, x, tokens) results, 12008868 outputs compared three ways (new, old, verify path); results with any differing byte or a touched guard: 0 -> **PASS**. Cases off-load or failed: 0 

| case | segments | MB | differing new/old, verify/old (all x, tokens) | old GB/s | new GB/s | new/old | old us | new us | new % of 250 |
|---|---|---|---|---|---|---|---|---|---|
| attn-v4 | q 16384 q4k, k 512 q4k, v 512 q4k | 40.1 | 0, 0 of 1183744 | 172.2 (171.8..173.1) | 212.5 (211.6..213.0) | 1.234 | 232.97 | 188.86 | 85 % |
| attn-v6 | q 16384 q4k, k 512 q4k, v 512 q6k | 40.6 | 0, 0 of 1183744 | 172.5 (171.7..173.2) | 213.0 (211.0..214.1) | 1.235 | 235.79 | 190.91 | 85 % |
| gdn-q4 | qkv 12288 q4k, gate 8192 q4k, alpha 64 q4k, beta 64 q4k | 47.5 | 0, 0 of 1401344 | 173.4 (172.5..174.3) | 212.0 (211.3..213.2) | 1.223 | 273.97 | 224.05 | 85 % |
| gdn-q6 | qkv 12288 q6k, gate 8192 q4k, alpha 64 q4k, beta 64 q4k | 60.5 | 0, 0 of 1401344 | 203.2 (201.6..203.9) | 212.0 (211.3..212.3) | 1.043 | 297.6 | 285.24 | 85 % |
| attn-q | q 16384 q4k | 37.7 | 0, 0 of 1114112 | 172.3 (171.6..173.6) | 209.2 (208.7..209.9) | 1.214 | 219.17 | 180.55 | 84 % |
| attn-kv | k 512 q4k, v 512 q4k | 2.4 | 0, 0 of 69632 | 132.4 (129.9..132.9) | 150.6 (148.1..151.0) | 1.137 | 17.94 | 15.77 | 60 % |
| attn-gate | gate 8192 q4k | 18.9 | 0, 0 of 557056 | 162.4 (161.5..163.3) | 200.5 (196.5..202.1) | 1.234 | 116.3 | 94.24 | 80 % |
| attn-qkv | qkv 12288 q4k | 28.3 | 0, 0 of 835584 | 168.5 (168.1..169.0) | 207.1 (205.1..208.4) | 1.229 | 168.14 | 136.81 | 83 % |
| ssm-alpha | alpha 64 q4k, beta 64 q4k | 0.3 | 0, 0 of 8704 | 39.3 (38.3..39.5) | 45.8 (45.4..46.9) | 1.164 | 7.91 | 6.8 | 18 % |
| attn-qkv-q6 | qkv 12288 q6k | 41.3 | 0, 0 of 835584 | 206.3 (205.3..207.1) | 206.2 (205.0..207.5) | 0.999 | 200.22 | 200.34 | 82 % |
| q35-gdn-blk0 | qkv 8192 q6k, gate 4096 q4k, alpha 32 q4k, beta 32 q4k | 18.6 | 0, 0 of 839936 | 188.0 (185.0..188.6) | 196.0 (193.5..196.8) | 1.043 | 98.76 | 94.72 | 78 % |
| q35-gdn-blk5 | qkv 8192 q4k, gate 4096 q4k, alpha 32 q4k, beta 32 q4k | 14.2 | 0, 0 of 839936 | 164.4 (162.9..164.8) | 195.7 (190.3..196.7) | 1.191 | 86.62 | 72.76 | 78 % |
| q35-attn-blk3 | q 8192 q4k, k 512 q4k, v 512 q6k | 10.9 | 0, 0 of 626688 | 158.3 (156.6..159.1) | 187.9 (186.9..189.7) | 1.186 | 68.81 | 58.0 | 75 % |
| q35-attn-blk11 | q 8192 q4k, k 512 q4k, v 512 q4k | 10.6 | 0, 0 of 626688 | 157.4 (157.1..157.7) | 188.9 (184.0..189.8) | 1.200 | 67.51 | 56.26 | 76 % |
| edge-gdn-odd | qkv 1002 q4k, gate 1001 q4k, alpha 3 q4k, beta 3 q4k | 4.6 | 0, 0 of 136612 | - | - | - | - | - | - |
| edge-attn-k256 | q 2048 q4k, k 256 q4k, v 256 q4k | 0.4 | 0, 0 of 174080 | - | - | - | - | - | - |
| edge-attn-k5120 | q 2048 q4k, k 256 q4k, v 256 q6k | 7.7 | 0, 0 of 174080 | - | - | - | - | - | - |

continue_rule (new >= old cold GB/s on every Ornith fused and per-shape case, 0 differing): **PASS** 

## [share] input projections' share of a block (old kernel; block = 18.5 ms / 15 = 1233 us)

| block type | old us | share | new us | share |
|---|---|---|---|---|
| attn-v4 | 232.97 | 18.9 % | 188.86 | 15.3 % |
| attn-v6 | 235.79 | 19.1 % | 190.91 | 15.5 % |
| gdn-q4 | 273.97 | 22.2 % | 224.05 | 18.2 % |
| gdn-q6 | 297.6 | 24.1 % | 285.24 | 23.1 % |

## [projection] computed, not measured

Saving: stage 0 0.432 ms a step (8 gdn-q6 + 4 gdn-q4 + 1 attn-v6 + 2 attn-v4), all 60 blocks 2.087 ms a token.
This kernel alone: stage 0 18.5 -> 18.07 ms; token 83.0 -> 80.91 ms = 12.05 -> 12.36 tok/s (projected).
With ORNITH-KERN3 router + outproj (0.088 ms a block): stage 0 18.5 -> 16.75 ms; token 83.0 -> 75.63 ms = 12.05 -> 13.22 tok/s (projected).

