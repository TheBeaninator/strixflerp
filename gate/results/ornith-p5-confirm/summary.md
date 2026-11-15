# FLERP-P5 Ornith 4-node confirm, run 11132252

Written 2026-11-13T23:16:09Z. Reference: P5 run 11121855 (world 4, cuts 0/15/30/45/60, 12.05 tok/s, 82.96 ms a token). Criteria: gate/ornith15-397b.flerp-p5.toml, gate/ornith-o1o2.toml, gate/ornith-o3-mtp.toml [ornith] (unchanged). Binary sha256 9e0d3a8b466c3cb645ccb82e17649c49b676ef8745fb3c63129d95efe956d547. Hold 2026-11-13T22:53:13Z -> 2026-11-13T23:16:09Z.

**PASS** (faults: 0; criterion-4 copy bound misses, recorded: 96)

## Runs (rank 0: ids / sha256 vs 11121855; records vs launch 0's full:1)

| launch | rows | reply:S | prompt | tokens | ids eq | sha eq | TTFT s | decode ms/tok | copy ms/call r1/r2/r3 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | pinned | full:1 | bridge | 21 | True | True | 0.650 | 77.21 | 1.782/1.492/1.919 |
| 0 | pinned | full:1 | code | 20 | True | True | 0.525 | 77.55 | 1.779/1.744/1.883 |
| 0 | pinned | full:1 | rivers | 18 | True | True | 0.480 | 74.78 | 1.751/1.599/1.338 |
| 0 | pinned | full:1 | records | 1808 | True | True | 44.298 | 0.00 | 4.339/4.900/3.711 |
| 0 | pinned | full:4 | bridge | 21 | True | True | 0.315 | 76.92 | 1.612/1.643/1.655 |
| 0 | pinned | full:4 | code | 20 | True | True | 0.294 | 77.38 | 1.570/1.718/1.779 |
| 0 | pinned | full:4 | rivers | 18 | True | True | 0.290 | 77.66 | 1.704/1.730/1.760 |
| 0 | pinned | full:4 | records | 1808 | True | True | 20.213 | 0.00 | 2.312/2.145/2.309 |
| 0 | pinned | full:16 | bridge | 21 | True | True | 0.408 | 76.77 | 1.240/1.216/1.140 |
| 0 | pinned | full:16 | code | 20 | True | True | 0.399 | 75.38 | 1.159/1.231/0.942 |
| 0 | pinned | full:16 | rivers | 18 | True | True | 0.384 | 77.45 | 1.158/1.251/1.218 |
| 0 | pinned | full:16 | records | 1808 | True | True | 14.483 | 0.00 | 0.628/0.373/1.850 |
| 0 | pinned | token:1 | bridge | 21 | True | n/a | 0.539 | 77.21 | 1.863/1.633/1.771 |
| 0 | pinned | token:1 | code | 20 | True | n/a | 0.525 | 76.89 | 1.830/1.792/1.767 |
| 0 | pinned | token:1 | rivers | 18 | True | n/a | 0.480 | 77.76 | 1.892/1.751/1.866 |
| 0 | pinned | token:1 | records | 1808 | True | n/a | 44.377 | 0.00 | 4.465/4.664/3.438 |
| 1 | pageable | full:1 | bridge | 21 | True | True | 0.647 | 76.45 | 1.894/1.794/1.822 |
| 1 | pageable | full:1 | code | 20 | True | True | 0.527 | 76.48 | 1.797/1.802/1.976 |
| 1 | pageable | full:1 | rivers | 18 | True | True | 0.484 | 74.50 | 1.637/1.694/1.764 |
| 1 | pageable | full:1 | records | 1808 | True | True | 44.408 | 0.00 | 15.143/18.947/3.783 |
| 1 | pageable | token:1 | bridge | 21 | True | n/a | 0.539 | 75.81 | 1.783/1.888/2.024 |
| 1 | pageable | token:1 | code | 20 | True | n/a | 0.526 | 76.19 | 1.621/1.891/2.027 |
| 1 | pageable | token:1 | rivers | 18 | True | n/a | 0.483 | 77.08 | 1.946/1.904/2.030 |
| 1 | pageable | token:1 | records | 1808 | True | n/a | 44.375 | 0.00 | 15.565/18.478/3.682 |
| 2 | pinned | token:1 | bridge | 21 | True | n/a | 0.638 | 77.29 | 1.981/1.843/1.269 |
| 2 | pinned | token:1 | code | 20 | True | n/a | 0.523 | 76.68 | 1.938/1.814/1.732 |
| 2 | pinned | token:1 | rivers | 18 | True | n/a | 0.480 | 75.67 | 1.910/1.823/1.694 |
| 2 | pinned | token:1 | records | 1808 | True | n/a | 44.375 | 0.00 | 4.330/4.189/3.723 |
| 2 | pinned | full:16 | bridge | 21 | True | True | 0.411 | 76.18 | 1.098/1.245/1.287 |
| 2 | pinned | full:16 | code | 20 | True | True | 0.399 | 76.49 | 1.280/1.113/1.327 |
| 2 | pinned | full:16 | rivers | 18 | True | True | 0.389 | 74.30 | 1.177/1.171/1.071 |
| 2 | pinned | full:16 | records | 1808 | True | True | 14.426 | 0.00 | 0.683/0.393/1.783 |
| 2 | pinned | full:4 | bridge | 21 | True | True | 0.319 | 75.60 | 1.651/1.697/1.511 |
| 2 | pinned | full:4 | code | 20 | True | True | 0.295 | 76.33 | 1.588/1.663/1.713 |
| 2 | pinned | full:4 | rivers | 18 | True | True | 0.289 | 77.49 | 1.670/1.637/1.800 |
| 2 | pinned | full:4 | records | 1808 | True | True | 20.300 | 0.00 | 2.466/1.774/2.478 |
| 2 | pinned | full:1 | bridge | 21 | True | True | 0.537 | 76.59 | 1.977/1.827/1.882 |
| 2 | pinned | full:1 | code | 20 | True | True | 0.525 | 75.44 | 1.690/1.708/1.644 |
| 2 | pinned | full:1 | rivers | 18 | True | True | 0.483 | 71.78 | 1.470/1.331/1.141 |
| 2 | pinned | full:1 | records | 1808 | True | True | 44.385 | 0.00 | 4.188/4.071/4.259 |
| 3 | pageable | token:1 | bridge | 21 | True | n/a | 0.648 | 75.37 | 1.749/1.771/1.512 |
| 3 | pageable | token:1 | code | 20 | True | n/a | 0.526 | 74.49 | 1.769/1.679/1.668 |
| 3 | pageable | token:1 | rivers | 18 | True | n/a | 0.478 | 75.55 | 1.396/1.875/1.838 |
| 3 | pageable | token:1 | records | 1808 | True | n/a | 44.419 | 0.00 | 16.132/19.054/4.251 |
| 3 | pageable | full:1 | bridge | 21 | True | True | 0.538 | 75.33 | 1.690/1.671/1.894 |
| 3 | pageable | full:1 | code | 20 | True | True | 0.525 | 76.21 | 1.801/1.818/1.923 |
| 3 | pageable | full:1 | rivers | 18 | True | True | 0.482 | 75.14 | 1.907/1.665/1.605 |
| 3 | pageable | full:1 | records | 1808 | True | True | 44.374 | 0.00 | 16.275/19.477/3.033 |

## Medians

- decode pageable reply full S=1: median 75.24 ms a token = 13.29 tok/s (n 8; P5 82.96 / 12.05)
- decode pageable reply token S=1: median 75.46 ms a token = 13.25 tok/s (n 8; P5 82.96 / 12.05)
- copy pageable rank 1: median 1.799 ms a call (max 16.275)
- copy pageable rank 2: median 1.847 ms a call (max 19.477)
- copy pageable rank 3: median 1.949 ms a call (max 4.251)
- decode pinned reply full S=1: median 75.11 ms a token = 13.31 tok/s (n 8; P5 82.96 / 12.05)
- decode pinned reply token S=1: median 76.79 ms a token = 13.02 tok/s (n 8; P5 82.96 / 12.05)
- copy pinned rank 1: median 1.728 ms a call (max 4.465)
- copy pinned rank 2: median 1.702 ms a call (max 4.900)
- copy pinned rank 3: median 1.763 ms a call (max 4.259)
- TTFT pinned full S=1 bridge: 0.650, 0.537 s
- TTFT pinned full S=1 code: 0.525, 0.525 s
- TTFT pinned full S=1 records: 44.298, 44.385 s
- TTFT pinned full S=1 rivers: 0.480, 0.483 s
- TTFT pinned full S=4 bridge: 0.315, 0.319 s
- TTFT pinned full S=4 code: 0.294, 0.295 s
- TTFT pinned full S=4 records: 20.213, 20.300 s
- TTFT pinned full S=4 rivers: 0.290, 0.289 s
- TTFT pinned full S=16 bridge: 0.408, 0.411 s
- TTFT pinned full S=16 code: 0.399, 0.399 s
- TTFT pinned full S=16 records: 14.483, 14.426 s
- TTFT pinned full S=16 rivers: 0.384, 0.389 s

## ORNITH-O3 MTP acceptance (4-node, greedy)

head: {"event": "mtp_head", "rank": 3, "block": 60, "depth": 3, "weight_bytes": 4844701696, "buffer_bytes": 15439876, "load_s": 3.17864, "own_embedding": true, "data_file": "/cache/flerp-p5/Ornith-1.5-397B-Q4_K_M.gguf.staged"}
MTP ms a draft step (median): 4.8304

| class | k | n | marginal | chain | conditional | E tok/round |
|---|---|---|---|---|---|---|
| code | 1 | 1024 | 0.918 | 0.918 | 0.918 | 1.918 |
| code | 2 | 1020 | 0.782 | 0.770 | 0.838 | 2.688 |
| code | 3 | 1016 | 0.649 | 0.597 | 0.776 | 3.285 |
| prose | 1 | 1024 | 0.794 | 0.794 | 0.794 | 1.794 |
| prose | 2 | 1020 | 0.553 | 0.527 | 0.664 | 2.321 |
| prose | 3 | 1016 | 0.371 | 0.312 | 0.592 | 2.633 |
| all | 1 | 2048 | 0.856 | 0.856 | 0.856 | 1.856 |
| all | 2 | 2040 | 0.668 | 0.649 | 0.758 | 2.504 |
| all | 3 | 2032 | 0.510 | 0.455 | 0.701 | 2.959 |

PROJECTED speculative tok/s (verify of k+1 rows = one decode step at 76.89 ms, plus k draft steps): {"code": {"1": 23.47, "2": 31.05, "3": 35.95}, "prose": {"1": 21.95, "2": 26.82, "3": 28.82}, "all": {"1": 22.71, "2": 28.94, "3": 32.38}}

## Faults and bound misses (first 40)

- launch 0 (pinned) full:1 bridge: rank 1 copy 1.7822 ms a call (bound < 0.1)
- launch 0 (pinned) full:1 bridge: rank 2 copy 1.4921 ms a call (bound < 0.1)
- launch 0 (pinned) full:1 bridge: rank 3 copy 1.9190 ms a call (bound < 0.1)
- launch 0 (pinned) full:1 code: rank 1 copy 1.7789 ms a call (bound < 0.1)
- launch 0 (pinned) full:1 code: rank 2 copy 1.7441 ms a call (bound < 0.1)
- launch 0 (pinned) full:1 code: rank 3 copy 1.8835 ms a call (bound < 0.1)
- launch 0 (pinned) full:1 rivers: rank 1 copy 1.7514 ms a call (bound < 0.1)
- launch 0 (pinned) full:1 rivers: rank 2 copy 1.5994 ms a call (bound < 0.1)
- launch 0 (pinned) full:1 rivers: rank 3 copy 1.3382 ms a call (bound < 0.1)
- launch 0 (pinned) full:1 records: rank 1 copy 4.3389 ms a call (bound < 0.1)
- launch 0 (pinned) full:1 records: rank 2 copy 4.9005 ms a call (bound < 0.1)
- launch 0 (pinned) full:1 records: rank 3 copy 3.7112 ms a call (bound < 0.1)
- launch 0 (pinned) full:4 bridge: rank 1 copy 1.6118 ms a call (bound < 0.1)
- launch 0 (pinned) full:4 bridge: rank 2 copy 1.6426 ms a call (bound < 0.1)
- launch 0 (pinned) full:4 bridge: rank 3 copy 1.6553 ms a call (bound < 0.1)
- launch 0 (pinned) full:4 code: rank 1 copy 1.5702 ms a call (bound < 0.1)
- launch 0 (pinned) full:4 code: rank 2 copy 1.7177 ms a call (bound < 0.1)
- launch 0 (pinned) full:4 code: rank 3 copy 1.7787 ms a call (bound < 0.1)
- launch 0 (pinned) full:4 rivers: rank 1 copy 1.7040 ms a call (bound < 0.1)
- launch 0 (pinned) full:4 rivers: rank 2 copy 1.7295 ms a call (bound < 0.1)
- launch 0 (pinned) full:4 rivers: rank 3 copy 1.7600 ms a call (bound < 0.1)
- launch 0 (pinned) full:4 records: rank 1 copy 2.3123 ms a call (bound < 0.1)
- launch 0 (pinned) full:4 records: rank 2 copy 2.1451 ms a call (bound < 0.1)
- launch 0 (pinned) full:4 records: rank 3 copy 2.3088 ms a call (bound < 0.1)
- launch 0 (pinned) full:16 bridge: rank 1 copy 1.2397 ms a call (bound < 0.1)
- launch 0 (pinned) full:16 bridge: rank 2 copy 1.2156 ms a call (bound < 0.1)
- launch 0 (pinned) full:16 bridge: rank 3 copy 1.1399 ms a call (bound < 0.1)
- launch 0 (pinned) full:16 code: rank 1 copy 1.1589 ms a call (bound < 0.1)
- launch 0 (pinned) full:16 code: rank 2 copy 1.2315 ms a call (bound < 0.1)
- launch 0 (pinned) full:16 code: rank 3 copy 0.9420 ms a call (bound < 0.1)
- launch 0 (pinned) full:16 rivers: rank 1 copy 1.1577 ms a call (bound < 0.1)
- launch 0 (pinned) full:16 rivers: rank 2 copy 1.2510 ms a call (bound < 0.1)
- launch 0 (pinned) full:16 rivers: rank 3 copy 1.2183 ms a call (bound < 0.1)
- launch 0 (pinned) full:16 records: rank 1 copy 0.6279 ms a call (bound < 0.1)
- launch 0 (pinned) full:16 records: rank 2 copy 0.3729 ms a call (bound < 0.1)
- launch 0 (pinned) full:16 records: rank 3 copy 1.8496 ms a call (bound < 0.1)
- launch 0 (pinned) token:1 bridge: rank 1 copy 1.8634 ms a call (bound < 0.1)
- launch 0 (pinned) token:1 bridge: rank 2 copy 1.6328 ms a call (bound < 0.1)
- launch 0 (pinned) token:1 bridge: rank 3 copy 1.7712 ms a call (bound < 0.1)
- launch 0 (pinned) token:1 code: rank 1 copy 1.8301 ms a call (bound < 0.1)

## Reading (written after the run; the criteria are unchanged)

- Identity: every launch's ids and every logits sha256 equal run 11121855's (bridge, code, rivers) at pinned and pageable rows, S = 1, 4, 16. Token-reply ids are equal too. The HEAD kernels (ORNITH-KERN3 d0c3714, ATTN-PROJ e77a2bf) and O1/O2 (35fa289) are bit-equal on Ornith at world 4. The MTP launch also holds: the two runs' ids are equal and the probe saw rank 0's ids.
- Decode: 75.1-76.8 ms a token median (13.0-13.3 tok/s), against 82.96 / 12.05. That is -7.5 ms, close to the 75.9 ms the kernel work projected. Pinned and pageable rows decode at the same speed.
- O1 copy: pinned rows do NOT remove Ornith's copy cost. Median per call on ranks 1-3 is 1.70-1.76 ms pinned against 1.80-1.95 ms pageable, so criterion 4 (< 0.1 ms) misses on all 96 pinned rows. As on one node, the time is not pageable registration. Next suspect: the synchronous copy waits for the stage's queued kernels, so copy_ms includes GPU time.
- O2 TTFT: 21-token prompt 0.48-0.65 s at S=1, 0.29-0.32 s at S=4, 0.38-0.41 s at S=16. Records (1,808 tokens) 44.3 s at S=1, 20.2 s at S=4, 14.5 s at S=16 (3.1x). Records generates no decode tokens at context 2048 (prompt pass only), so it is a TTFT measurement only.
- O3: Ornith MTP chain acceptance at k = 1/2/3 is code 0.918/0.770/0.597 and prose 0.794/0.527/0.312. a1 on code is 0.918, above the survey's 0.6 bar. Draft step 4.83 ms. The speculative tok/s above is a projection: it takes the verify of k+1 rows as one decode step.
- Sources: ranks 0 and 1 streamed over the rail from PLACE placements on control-node's NVMe (about 1.15 GB/s), ranks 2 and 3 read PLACE placements on A's and C's /cache. No cold-storage bytes. The --model header (11 MB) is read from the ark path, as in P5.
- Run: hold 22:53:13Z to 23:16:09Z (23 min). Pair healthy again at 23:20:49Z.
