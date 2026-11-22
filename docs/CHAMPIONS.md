# Champions

Generated from `configs/champions.toml` by `flerp champion`; do not edit. Each area has one current champion, a
named configuration (`configs/NAME.toml`) with the best number that passed the area's fidelity floor. A candidate
is promoted with `flerp champion propose CONFIG AREA RESULTS_DIR` only when it beats the incumbent and passes the
floor; the incumbent moves to history. `ornith397.serve` (decode-best + prefill-best) is what the engine launches.

## decode-single

Decode speed of one stream. Metric: tok/s, higher is better. Fidelity floor: identity: the generated ids equal the drafts-off, per-token-path greedy ids byte for byte.

| status | case | config | value | scale | fidelity | evidence (run) | date |
|---|---|---|---|---|---|---|---|
| current |  | `ornith397.decode-best` | 24.4 tok/s | 397B | identical ids, 48/48 runs at 3 drafts equal to drafts-off; 4 of the 48 pairs generated nothing on either setting (the records prompt, silent at both drafts settings), so 44/48 comparisons are substantive | gate/results/ornith-confirm2 (card 3bed1267a2, branch card/3bed1267a2 @ 9e1bb283e569) (11140653) | 2026-11-14 |
| history |  | `ornith397.baseline-p5` | 13.3 tok/s | 397B | identical ids and sha256 of every logits row to P5 run 11121855 | gate/results/ornith-p5-confirm (11132252) | 2026-11-13 |
| history |  | `ornith397.decode-best` | 25.7 tok/s | 397B | identical ids, MTP 3 drafts vs drafts-off, every prompt that ran (window launch 0) | gate/results/ornith-o3-spec (spec window launch 0) (11140139) | 2026-11-14 |

- decode-single ornith397.decode-best: median of the per-run tok/s at 3 drafts over both launches (p10-p90 21.3-25.8; per-class means: code 24.8, P5 prompts 25.3, prose 21.6 - prose is the class that misses the 23.1 bar, so quote the prose row for chat; medians 24.96 / 25.27 / 21.86); drafts off median 13.3 (13.1-13.5) in the same window; verdict.json decode criterion (pooled median >= 23.1) true, and ids identical to drafts-off on every prompt that generated
- decode-single ornith397.baseline-p5: 75.1-76.8 ms a token (13.0-13.3 tok/s) vs 82.96 ms before pinned rows
- decode-single ornith397.decode-best: 25.7 was total tokens / total time over one launch (window launch 0 of 11140139), not a per-prompt median; the Ornith-measured median at 3 drafts is 24.4 (ornith-confirm2 11140653). peak prompt; the launch mean at 3 drafts is about 23 tok/s, 2 drafts about 22.5, drafts off 13.0. Evidence is on the pod (strixflerp-ornith tree), pull it to the laptop with tools/pod-pull.sh.

## throughput

Total decode speed over many concurrent streams. Metric: tok/s, higher is better. Fidelity floor: identity per stream against the same stream run alone.

| status | case | config | value | scale | fidelity | evidence (run) | date |
|---|---|---|---|---|---|---|---|
| current |  | `ornith397.baseline-p5` | 13.3 tok/s | 397B | identical ids and sha256 (single stream) | gate/results/ornith-p5-confirm (11132252) | 2026-11-13 |

- throughput ornith397.baseline-p5: baseline only: one stream, because many-stream throughput has not been measured on Ornith. The first throughput gate replaces this.

## prefill

Prompt pass speed. Metric: tok/s, higher is better. Fidelity floor: mean KL vs the FP32 reference <= 0.0176, today's per-token path's own distance (owner 2026-11-14; gate/prefill-ornith.toml amendment).

| status | case | config | value | scale | fidelity | evidence (run) | date |
|---|---|---|---|---|---|---|---|
| current |  | `ornith397.prefill-best` | 2725 tok/s | 35B proxy | KL vs FP32 0.00399 <= 0.0176 | gate/results/prefill-lrc/zgate (zgate (control-node 2026-11-14T18:29Z, card 5083008fe9)) | 2026-11-14 |
| history |  | `ornith397.prefill-best` | 1297 tok/s | 35B proxy | KL vs FP32 0.0193 | /data/projects/strixflerp-pfo/build/pfo-35b-speedE (card a74fe1bdc1) (speedE) | 2026-11-14 |
| history |  | `ornith397.baseline-p5` | 276 tok/s | 35B proxy | KL vs FP32 0.0176 | /data/projects/strixflerp-pfo/build/pfo-35b-speedE (card a74fe1bdc1) (speedE) | 2026-11-14 |
| history |  | `ornith397.prefill-best` | 1327 tok/s | 35B proxy | KL vs FP32 0.00653 <= 0.0176 | gate/results/prefill-ornith (pfo-35b-bulk short + pfo-35b-dnD) | 2026-11-14 |
| variant |  | `ornith397.prefill-exact` | 1026 tok/s | 35B proxy | KL vs FP32 0.00268 <= 0.0176 | gate/results/prefill-ornith (pfo-35b-bulk short + pfo-35b-dnD) | 2026-11-14 |
| history |  | `ornith397.prefill-best` | 1974 tok/s | 35B proxy | KL vs FP32 0.00399 <= 0.0176 | gate/results/prefill-attn (pfo-35b-attn + pfo-35b-attnD) | 2026-11-14 |
| measured |  | `ornith397.prefill-best` | 503 tok/s | 397B | no FP32 reference on the 397B (floor not checkable); vs the old per-token path top-1 11/12, greedy equal 3/12 over the prompts that generated output (the 1,808-token records prompt is silent on both paths, excluded); the card's own criterion (greedy equal on all 3 P5 prompts) is 2/3, so this row is measured, not adopted | gate/results/ornith-confirm2 (card 3bed1267a2, branch card/3bed1267a2 @ 9e1bb283e569) (11140653) | 2026-11-14 |
| current | qwen38-flash-next | `qwen38-fnext.prefill-best` | 1700.4 tok/s | Flash-Next (K, 8192-token prompt) | logits sha256 and greedy ids identical to qwen38-fnext.served (logits sha256 of the last 16 rows and 32 greedy ids, 8 prompts 128..32768) (its KL vs FP32 unchanged) | gate/results/strix-loop/20261116-sfx-p16 (20261116-sfx-p16) | 2026-11-17 |
| history | qwen38-flash-next | `qwen38-fnext.served` | 1618.4 tok/s | Flash-Next (K, 8192-token prompt) | logits sha256 and greedy ids identical to the served Flash-Next path itself (strixite modules, v4 weights) (its KL vs FP32 unchanged) | gate/results/strix-loop/20261116-sfx-p16/champion-base (20261116-sfx-p16) | 2026-11-17 |

- prefill ornith397.prefill-best: arm Z = G + live-tile items (moe_kgrouped 3dbd959) + tiled prompt router (moe_kquant 379fc3a) + 16-byte MoE combine and one ToF16 per shared input (moe_combine 6e95ab2): 2725 tok/s at 2,048 tokens and 2560 at 16,384 (one load, median of 3 interleaved; G in the same process 2012 / 1962, Z/G 1.355 / 1.305); logits sha256 of every recorded step and the ids equal G's on the 8 fidelity prompts and a 16K prompt (108 steps, 0 mismatches), so G's KL vs FP32 0.00399 holds unchanged; GPU time a 4,096-row chunk 2,049 -> 1,410 ms (zprof); gate/prefill-lrc.toml Z1-Z4 all pass; supersedes the withdrawn single-lever proposals of 9506f4f116, c6473d6063, 2dc55e67f0
- prefill ornith397.prefill-best: 35B proxy, 4.7x the per-token path; Ornith itself is not measured yet, the first Ornith prefill gate replaces this
- prefill ornith397.baseline-p5: derived: 1297 / 4.7, the per-token path in the same run
- prefill ornith397.prefill-best: arm F (experts + dense F16), 35B proxy: 1,327 tok/s at 2,048 tokens (median of 3, one load; A 295, B 1,313); KL vs FP32 0.0065 (top-1 98.8 %, greedy 5/8; A 0.0176); evidence on the pod: /data/projects/strixflerp-pfo/build/pfo-35b-bulk and pfo-35b-dnD; card de279ffbf6
- prefill ornith397.prefill-exact: arm X (experts + dense exact codes, hi/lo F16): the most faithful prompt pass (top-1 99.2 %, greedy 6/8), 23 % slower than prefill-best; a named variant, not the champion (owner 2026-11-14)
- prefill ornith397.prefill-best: arm G (experts + dense F16 + GQA WMMA prompt attention), 35B proxy: 1,974 tok/s at 2,048 tokens and 1,937 at 16,384 (one load, median of 3; F 1,308 / 175); KL vs FP32 0.0040 (max 0.20, top-1 98.8 %, greedy 5/8); evidence on the pod: /data/projects/strixflerp-pfo/build/pfo-35b-attn and pfo-35b-attnD; card 671acb4cef
- prefill ornith397.prefill-best: Ornith 397B measured, world 4 F D A C, 4,096-row chunks x 16 sub-chunks: 527 tok/s at 16K, 503 at 32K, 362-377 at 1.8-2K (old path about 125); 200-1,000-token prompts stay at about 125 at 16 sub-chunks (under the 64-row grouped floor). Misses the ~2,500 tok/s projection (PROJECTION MISSED, about 5x): 256-row sub-chunks put each stage on the weight-bandwidth roof plus a 16 % pipeline fill/drain per 4K call; the 35B proxy entry (1,974, one GPU, whole 4K chunks) stays the floor-checked champion
- qwen38-flash-next qwen38-fnext.prefill-best: strix-loop 20261116-sfx-p16 on K, A B A B: prefill ratio vs served 1.101 / 1.112 / 1.073 / 1.046 / 1.069 / 1.051 / 1.051 / 1.051 at 128..32768 (median 1.051 over >= 2048, every pair outside both sides' ranges); 1700.4 tok/s at 8192 vs 1618.4; decode ratio 1.003-1.005 (not promoted for decode); the isa scratch check is still owed (the run's isa scan did not run, so it vouches for no scratch)
- qwen38-flash-next qwen38-fnext.served: base side of strix-loop 20261116-sfx-p16 on K (job 20261117T000124Z-run-pmodel): median second-request prefill tok/s at 8192 (A1/A2 1617.5-1619.2); 128..32768: 716 / 1206 / 1372 / 1466 / 1531 / 1618 / 1605 / 1589; decode 33.1-33.9 tok/s T=1 MTP off

## ttft

Time to first token, per prompt length. Metric: s, lower is better. Fidelity floor: identity of the first token and the ids after it against the per-token path (grouped prefill falls under the prefill KL floor instead).

| status | case | config | value | scale | fidelity | evidence (run) | date |
|---|---|---|---|---|---|---|---|
| current | 1808 tok | `ornith397.baseline-p5` | 14.5 s | 397B | identical ids and sha256 | gate/results/ornith-p5-confirm (11132252) | 2026-11-13 |
| current | 21 tok | `ornith397.baseline-p5` | 0.32 s | 397B | identical ids and sha256 | gate/results/ornith-p5-confirm (11132252) | 2026-11-13 |

- 1808 tok ornith397.baseline-p5: 1,808-token prompt, S=16 sub-chunks (44.3 s at S=1)
- 21 tok ornith397.baseline-p5: 21-token prompt, S=4 sub-chunks
