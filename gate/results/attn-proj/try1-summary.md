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
| attn-v4 | q 16384 q4k, k 512 q4k, v 512 q4k | 40.1 | 0, 0 of 1183744 | 172.7 (172.4..173.4) | 206.9 (206.0..207.9) | 1.198 | 232.28 | 193.91 | 83 % |
| attn-v6 | q 16384 q4k, k 512 q4k, v 512 q6k | 40.6 | 0, 0 of 1183744 | 174.1 (171.6..177.5) | 208.0 (206.9..209.4) | 1.195 | 233.58 | 195.5 | 83 % |
| gdn-q4 | qkv 12288 q4k, gate 8192 q4k, alpha 64 q4k, beta 64 q4k | 47.5 | 0, 0 of 1401344 | 176.1 (173.9..180.1) | 207.0 (205.4..208.9) | 1.176 | 269.73 | 229.44 | 83 % |
| gdn-q6 | qkv 12288 q6k, gate 8192 q4k, alpha 64 q4k, beta 64 q4k | 60.5 | 0, 0 of 1401344 | 201.0 (200.4..201.4) | 209.2 (208.0..209.6) | 1.041 | 300.84 | 289.08 | 84 % |
| attn-q | q 16384 q4k | 37.7 | 0, 0 of 1114112 | 174.0 (172.1..178.5) | 203.8 (201.4..205.3) | 1.171 | 217.06 | 185.31 | 82 % |
| attn-kv | k 512 q4k, v 512 q4k | 2.4 | 0, 0 of 69632 | 135.3 (133.3..136.2) | 153.4 (152.4..154.2) | 1.134 | 17.56 | 15.49 | 61 % |
| attn-gate | gate 8192 q4k | 18.9 | 0, 0 of 557056 | 163.8 (157.9..164.4) | 194.6 (189.3..195.1) | 1.188 | 115.36 | 97.1 | 78 % |
| attn-qkv | qkv 12288 q4k | 28.3 | 0, 0 of 835584 | 169.6 (167.8..173.5) | 199.4 (198.7..202.8) | 1.176 | 167.06 | 142.05 | 80 % |
| ssm-alpha | alpha 64 q4k, beta 64 q4k | 0.3 | 0, 0 of 8704 | 41.7 (39.8..41.9) | 41.0 (40.9..41.2) | 0.983 | 7.46 | 7.59 | 16 % |
| attn-qkv-q6 | qkv 12288 q6k | 41.3 | 0, 0 of 835584 | 204.8 (203.9..205.4) | 205.3 (204.5..205.4) | 1.002 | 201.72 | 201.23 | 82 % |
| q35-gdn-blk0 | qkv 8192 q6k, gate 4096 q4k, alpha 32 q4k, beta 32 q4k | 18.6 | 0, 0 of 839936 | 180.9 (180.3..183.1) | 189.7 (188.4..191.2) | 1.049 | 102.61 | 97.85 | 76 % |
| q35-gdn-blk5 | qkv 8192 q4k, gate 4096 q4k, alpha 32 q4k, beta 32 q4k | 14.2 | 0, 0 of 839936 | 159.8 (158.6..159.9) | 186.0 (185.9..186.3) | 1.164 | 89.1 | 76.55 | 74 % |
| q35-attn-blk3 | q 8192 q4k, k 512 q4k, v 512 q6k | 10.9 | 0, 0 of 626688 | 148.4 (148.3..148.7) | 191.3 (190.9..191.7) | 1.289 | 73.4 | 56.94 | 77 % |
| q35-attn-blk11 | q 8192 q4k, k 512 q4k, v 512 q4k | 10.6 | 0, 0 of 626688 | 145.7 (145.3..146.0) | 191.0 (189.0..191.4) | 1.311 | 72.93 | 55.63 | 76 % |
| edge-gdn-odd | qkv 1002 q4k, gate 1001 q4k, alpha 3 q4k, beta 3 q4k | 4.6 | 0, 0 of 136612 | - | - | - | - | - | - |
| edge-attn-k256 | q 2048 q4k, k 256 q4k, v 256 q4k | 0.4 | 0, 0 of 174080 | - | - | - | - | - | - |
| edge-attn-k5120 | q 2048 q4k, k 256 q4k, v 256 q6k | 7.7 | 0, 0 of 174080 | - | - | - | - | - | - |

continue_rule (new >= old cold GB/s on every Ornith fused and per-shape case, 0 differing): **FAIL** ['ssm-alpha']

## [share] input projections' share of a block (old kernel; block = 18.5 ms / 15 = 1233 us)

| block type | old us | share | new us | share |
|---|---|---|---|---|
| attn-v4 | 232.28 | 18.8 % | 193.91 | 15.7 % |
| attn-v6 | 233.58 | 18.9 % | 195.5 | 15.9 % |
| gdn-q4 | 269.73 | 21.9 % | 229.44 | 18.6 % |
| gdn-q6 | 300.84 | 24.4 % | 289.08 | 23.4 % |

## [projection] computed, not measured

Saving: stage 0 0.370 ms a step (8 gdn-q6 + 4 gdn-q4 + 1 attn-v6 + 2 attn-v4), all 60 blocks 1.759 ms a token.
This kernel alone: stage 0 18.5 -> 18.13 ms; token 83.0 -> 81.24 ms = 12.05 -> 12.31 tok/s (projected).
With ORNITH-KERN3 router + outproj (0.088 ms a block): stage 0 18.5 -> 16.81 ms; token 83.0 -> 75.96 ms = 12.05 -> 13.16 tok/s (projected).

## [model] 35B, 4 gate prompts x 6 configs, --staged loopback

- run 0 `before` (sha256 98fee387fd98): exit 0, 171.6 s, load 4.93 -> 4.54; 24 results judged against P1 ids + 128 logits sha256; failures: 0 
- run 1 `after` (sha256 82c2219ea7a0): exit 0, 171.7 s, load 4.54 -> 8.94 (speed INVALID: load above 6); 24 results judged against P1 ids + 128 logits sha256; failures: 0 
- run 2 `before` (sha256 98fee387fd98): exit 0, 173.3 s, load 5.68 -> 4.28; 24 results judged against P1 ids + 128 logits sha256; failures: 0 
- run 3 `after` (sha256 82c2219ea7a0): exit 0, 174.9 s, load 4.28 -> 8.35 (speed INVALID: load above 6); 24 results judged against P1 ids + 128 logits sha256; failures: 0 
- run 4 `after` (sha256 82c2219ea7a0): exit 0, 172.4 s, load 5.43 -> 5.31; 24 results judged against P1 ids + 128 logits sha256; failures: 0 
- run 5 `before` (sha256 98fee387fd98): exit 0, 176.0 s, load 5.31 -> 4.72; 24 results judged against P1 ids + 128 logits sha256; failures: 0 

Decode tok/s world 1 graph on, all prompts, median (min..max): before 69.21 (67.39..71.17), after 69.50 (55.75..70.58), after/before 1.004.

