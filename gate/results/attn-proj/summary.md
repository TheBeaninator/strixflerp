# ATTN-PROJ: Ornith-1.5 397B K=4096 input projections (fused q/k/v and GDN qkv/gate/alpha/beta), memory access only

Criteria `gate/attn-proj.toml` (committed before any GPU run). Host control-node, lease control-gpu as agent. Old = the one-row fused launchers as at the criteria commit (gate/attn_proj_ref.hip); new = the catalogue's.

## [tune] variants (rows a wave R, waves a block W), Ornith fused cases, cold, 1 token

| variant | R W | differing | attn-v4 new us (GB/s) | attn-v6 new us (GB/s) | gdn-q4 new us (GB/s) | gdn-q6 new us (GB/s) | sum new us |
|---|---|---|---|---|---|---|---|
| v1 | 2 8 | 0 | 199.23 (201.4) | 197.99 (205.4) | 232.13 (204.6) | 292.47 (206.8) | 921.8 |
| v2 | 2 4 | 0 | 200.0 (200.6) | 200.54 (202.8) | 236.67 (200.7) | 295.99 (204.3) | 933.2 |
| v3 | 4 4 | 0 | 216.77 (185.1) | 218.67 (186.0) | 257.04 (184.8) | 307.56 (196.6) | 1000.0 |
| v4 | 4 8 | 0 | 211.24 (189.9) | 211.21 (192.5) | 246.08 (193.0) | 304.4 (198.7) | 972.9 |

Pick ([tune] rule): **v1** (R W = 2 8), sum 921.8 us.

## [kernel] final source, cache-cold

204 (case, x, tokens) results, 12008868 outputs compared three ways (new, old, verify path); results with any differing byte or a touched guard: 0 -> **PASS**. Cases off-load or failed: 0 

| case | segments | MB | differing new/old, verify/old (all x, tokens) | old GB/s | new GB/s | new/old | old us | new us | new % of 250 |
|---|---|---|---|---|---|---|---|---|---|
| attn-v4 | q 16384 q4k, k 512 q4k, v 512 q4k | 40.1 | 0, 0 of 1183744 | 171.7 (162.7..173.2) | 206.2 (203.1..207.6) | 1.201 | 233.71 | 194.58 | 82 % |
| attn-v6 | q 16384 q4k, k 512 q4k, v 512 q6k | 40.6 | 0, 0 of 1183744 | 173.8 (173.0..174.3) | 207.9 (206.1..208.8) | 1.196 | 233.99 | 195.61 | 83 % |
| gdn-q4 | qkv 12288 q4k, gate 8192 q4k, alpha 64 q4k, beta 64 q4k | 47.5 | 0, 0 of 1401344 | 174.6 (172.0..175.1) | 206.9 (205.2..207.4) | 1.185 | 272.02 | 229.53 | 83 % |
| gdn-q6 | qkv 12288 q6k, gate 8192 q4k, alpha 64 q4k, beta 64 q4k | 60.5 | 0, 0 of 1401344 | 199.1 (198.2..199.7) | 207.8 (206.6..208.4) | 1.044 | 303.81 | 291.01 | 83 % |
| attn-q | q 16384 q4k | 37.7 | 0, 0 of 1114112 | 171.0 (168.3..173.0) | 200.6 (198.2..201.6) | 1.173 | 220.9 | 188.31 | 80 % |
| attn-kv | k 512 q4k, v 512 q4k | 2.4 | 0, 0 of 69632 | 131.1 (130.2..133.8) | 146.8 (145.8..151.1) | 1.120 | 18.13 | 16.18 | 59 % |
| attn-gate | gate 8192 q4k | 18.9 | 0, 0 of 557056 | 155.9 (153.9..161.6) | 189.2 (185.6..191.7) | 1.213 | 121.19 | 99.87 | 76 % |
| attn-qkv | qkv 12288 q4k | 28.3 | 0, 0 of 835584 | 172.6 (167.7..173.0) | 199.9 (196.8..202.1) | 1.158 | 164.17 | 141.74 | 80 % |
| ssm-alpha | alpha 64 q4k, beta 64 q4k | 0.3 | 0, 0 of 8704 | 41.8 (40.3..41.9) | 49.9 (49.8..50.0) | 1.194 | 7.44 | 6.24 | 20 % |
| attn-qkv-q6 | qkv 12288 q6k | 41.3 | 0, 0 of 835584 | 203.5 (202.9..203.9) | 203.8 (203.4..204.3) | 1.001 | 202.96 | 202.67 | 82 % |
| q35-gdn-blk0 | qkv 8192 q6k, gate 4096 q4k, alpha 32 q4k, beta 32 q4k | 18.6 | 0, 0 of 839936 | 180.8 (174.8..182.0) | 188.6 (185.4..190.7) | 1.043 | 102.69 | 98.41 | 75 % |
| q35-gdn-blk5 | qkv 8192 q4k, gate 4096 q4k, alpha 32 q4k, beta 32 q4k | 14.2 | 0, 0 of 839936 | 157.5 (148.6..158.1) | 183.9 (180.6..185.1) | 1.168 | 90.4 | 77.41 | 74 % |
| q35-attn-blk3 | q 8192 q4k, k 512 q4k, v 512 q6k | 10.9 | 0, 0 of 626688 | 148.5 (147.8..148.6) | 189.7 (187.8..190.5) | 1.278 | 73.39 | 57.43 | 76 % |
| q35-attn-blk11 | q 8192 q4k, k 512 q4k, v 512 q4k | 10.6 | 0, 0 of 626688 | 145.2 (144.8..145.9) | 189.5 (188.6..190.3) | 1.305 | 73.19 | 56.08 | 76 % |
| edge-gdn-odd | qkv 1002 q4k, gate 1001 q4k, alpha 3 q4k, beta 3 q4k | 4.6 | 0, 0 of 136612 | - | - | - | - | - | - |
| edge-attn-k256 | q 2048 q4k, k 256 q4k, v 256 q4k | 0.4 | 0, 0 of 174080 | - | - | - | - | - | - |
| edge-attn-k5120 | q 2048 q4k, k 256 q4k, v 256 q6k | 7.7 | 0, 0 of 174080 | - | - | - | - | - | - |

continue_rule (new >= old cold GB/s on every Ornith fused and per-shape case, 0 differing): **PASS** 

## [share] input projections' share of a block (old kernel; block = 18.5 ms / 15 = 1233 us)

| block type | old us | share | new us | share |
|---|---|---|---|---|
| attn-v4 | 233.71 | 18.9 % | 194.58 | 15.8 % |
| attn-v6 | 233.99 | 19.0 % | 195.61 | 15.9 % |
| gdn-q4 | 272.02 | 22.1 % | 229.53 | 18.6 % |
| gdn-q6 | 303.81 | 24.6 % | 291.01 | 23.6 % |

## [projection] computed, not measured

Saving: stage 0 0.389 ms a step (8 gdn-q6 + 4 gdn-q4 + 1 attn-v6 + 2 attn-v4), all 60 blocks 1.841 ms a token.
This kernel alone: stage 0 18.5 -> 18.11 ms; token 83.0 -> 81.16 ms = 12.05 -> 12.32 tok/s (projected).
With ORNITH-KERN3 router + outproj (0.088 ms a block): stage 0 18.5 -> 16.79 ms; token 83.0 -> 75.88 ms = 12.05 -> 13.18 tok/s (projected).

## [model] 35B, 4 gate prompts x 6 configs, --staged loopback

- run 0 `before` (sha256 98fee387fd98): exit 0, 173.5 s, load 3.72 -> 4.87; 24 results judged against P1 ids + 128 logits sha256; failures: 0 
- run 1 `after` (sha256 04d88470250d): exit 0, 168.2 s, load 4.87 -> 3.62; 24 results judged against P1 ids + 128 logits sha256; failures: 0 
- run 2 `before` (sha256 98fee387fd98): exit 0, 170.4 s, load 3.62 -> 3.20; 24 results judged against P1 ids + 128 logits sha256; failures: 0 
- run 3 `after` (sha256 04d88470250d): exit 0, 168.7 s, load 3.20 -> 4.30; 24 results judged against P1 ids + 128 logits sha256; failures: 0 
- run 4 `after` (sha256 04d88470250d): exit 0, 169.2 s, load 4.30 -> 4.14; 24 results judged against P1 ids + 128 logits sha256; failures: 0 
- run 5 `before` (sha256 98fee387fd98): exit 0, 170.1 s, load 4.14 -> 2.84; 24 results judged against P1 ids + 128 logits sha256; failures: 0 

Decode tok/s world 1 graph on, all prompts, median (min..max): before 70.19 (69.08..71.66), after 72.46 (70.35..74.26), after/before 1.032.

