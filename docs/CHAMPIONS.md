# Champions

Generated from `configs/champions.toml` by `flerp champion`; do not edit. Each area has one current champion, a
named configuration (`configs/NAME.toml`) with the best number that passed the area's fidelity floor. A candidate
is promoted with `flerp champion propose CONFIG AREA RESULTS_DIR` only when it beats the incumbent and passes the
floor; the incumbent moves to history. `ornith397.serve` (decode-best + prefill-best) is what the engine launches.

## decode-single

Decode speed of one stream. Metric: tok/s, higher is better. Fidelity floor: identity: the generated ids equal the drafts-off, per-token-path greedy ids byte for byte.

| status | case | config | value | scale | fidelity | evidence (run) | date |
|---|---|---|---|---|---|---|---|
| current |  | `ornith397.decode-best` | 24.4 tok/s | 397B | identical ids, 48/48 runs at 3 drafts equal to drafts-off | gate/results/ornith-confirm2 (card 3bed1267a2, branch card/3bed1267a2 @ 9e1bb283e569) (11140653) | 2026-11-14 |
| history |  | `ornith397.baseline-p5` | 13.3 tok/s | 397B | identical ids and sha256 of every logits row to P5 run 11121855 | gate/results/ornith-p5-confirm (11132252) | 2026-11-13 |
| history |  | `ornith397.decode-best` | 25.7 tok/s | 397B | identical ids, MTP 3 drafts vs drafts-off, every prompt that ran (window launch 0) | gate/results/ornith-o3-spec (spec window launch 0) (11140139) | 2026-11-14 |

- decode-single ornith397.decode-best: median of the per-run tok/s at 3 drafts over both launches (p10-p90 21.3-25.8; means: code 24.8, P5 prompts 25.3); drafts off median 13.3 (13.1-13.5) in the same window; verdict.json decode criterion (median >= 23.1) true
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
| current |  | `ornith397.prefill-best` | 1974 tok/s | 35B proxy | KL vs FP32 0.00399 <= 0.0176 | gate/results/prefill-attn (pfo-35b-attn + pfo-35b-attnD) | 2026-11-14 |
| history |  | `ornith397.prefill-best` | 1297 tok/s | 35B proxy | KL vs FP32 0.0193 | /data/projects/strixflerp-pfo/build/pfo-35b-speedE (card a74fe1bdc1) (speedE) | 2026-11-14 |
| history |  | `ornith397.baseline-p5` | 276 tok/s | 35B proxy | KL vs FP32 0.0176 | /data/projects/strixflerp-pfo/build/pfo-35b-speedE (card a74fe1bdc1) (speedE) | 2026-11-14 |
| history |  | `ornith397.prefill-best` | 1327 tok/s | 35B proxy | KL vs FP32 0.00653 <= 0.0176 | gate/results/prefill-ornith (pfo-35b-bulk short + pfo-35b-dnD) | 2026-11-14 |
| variant |  | `ornith397.prefill-exact` | 1026 tok/s | 35B proxy | KL vs FP32 0.00268 <= 0.0176 | gate/results/prefill-ornith (pfo-35b-bulk short + pfo-35b-dnD) | 2026-11-14 |
| measured |  | `ornith397.prefill-best` | 503 tok/s | 397B | no FP32 reference on the 397B (floor not checkable); vs the old per-token path top-1 12/13, greedy equal 4/13 | gate/results/ornith-confirm2 (card 3bed1267a2, branch card/3bed1267a2 @ 9e1bb283e569) (11140653) | 2026-11-14 |

- prefill ornith397.prefill-best: arm G (experts + dense F16 + GQA WMMA prompt attention), 35B proxy: 1,974 tok/s at 2,048 tokens and 1,937 at 16,384 (one load, median of 3; F 1,308 / 175); KL vs FP32 0.0040 (max 0.20, top-1 98.8 %, greedy 5/8); evidence on the pod: /data/projects/strixflerp-pfo/build/pfo-35b-attn and pfo-35b-attnD; card 671acb4cef; on Ornith 397B itself (ornith-confirm2 11140653) the same config measured 503-527 tok/s at 16-32K: the ~2,500 tok/s Ornith projection (docs/PREFILL-ORNITH.md section 5) was missed
- prefill ornith397.prefill-best: 35B proxy, 4.7x the per-token path; Ornith itself is not measured yet, the first Ornith prefill gate replaces this
- prefill ornith397.baseline-p5: derived: 1297 / 4.7, the per-token path in the same run
- prefill ornith397.prefill-best: arm F (experts + dense F16), 35B proxy: 1,327 tok/s at 2,048 tokens (median of 3, one load; A 295, B 1,313); KL vs FP32 0.0065 (top-1 98.8 %, greedy 5/8; A 0.0176); evidence on the pod: /data/projects/strixflerp-pfo/build/pfo-35b-bulk and pfo-35b-dnD; card de279ffbf6
- prefill ornith397.prefill-exact: arm X (experts + dense exact codes, hi/lo F16): the most faithful prompt pass (top-1 99.2 %, greedy 6/8), 23 % slower than prefill-best; a named variant, not the champion (owner 2026-11-14)
- prefill ornith397.prefill-best: Ornith 397B measured, world 4 F D A C, 4,096-row chunks x 16 sub-chunks: 527 tok/s at 16K, 503 at 32K, 362-377 at 1.8-2K (old path about 125); 200-1,000-token prompts stay at about 125 at 16 sub-chunks (under the 64-row grouped floor). Misses the ~2,500 tok/s projection (PROJECTION MISSED, about 5x): 256-row sub-chunks put each stage on the weight-bandwidth roof plus a 16 % pipeline fill/drain per 4K call; the 35B proxy entry (1,974, one GPU, whole 4K chunks) stays the floor-checked champion

## ttft

Time to first token, per prompt length. Metric: s, lower is better. Fidelity floor: identity of the first token and the ids after it against the per-token path (grouped prefill falls under the prefill KL floor instead).

| status | case | config | value | scale | fidelity | evidence (run) | date |
|---|---|---|---|---|---|---|---|
| current | 1808 tok | `ornith397.baseline-p5` | 14.5 s | 397B | identical ids and sha256 | gate/results/ornith-p5-confirm (11132252) | 2026-11-13 |
| current | 21 tok | `ornith397.baseline-p5` | 0.32 s | 397B | identical ids and sha256 | gate/results/ornith-p5-confirm (11132252) | 2026-11-13 |

- 1808 tok ornith397.baseline-p5: 1,808-token prompt, S=16 sub-chunks (44.3 s at S=1)
- 21 tok ornith397.baseline-p5: 21-token prompt, S=4 sub-chunks
