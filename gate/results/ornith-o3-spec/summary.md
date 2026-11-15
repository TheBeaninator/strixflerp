# ORNITH-O3 step two: MTP speculative decoding, 35B proxy (world 4, TCP 127.0.0.1, one GPU)

Written 2026-11-14T00:33:39Z. Criteria: `gate/ornith-o3-spec.toml` (committed before the first GPU run). Binary sha256 8d94fd6f5a5ab774a6fd06cbaba7941c8f3d32f6590b51eb74848dbb3b2253ef.

**PASS**: 0 failure(s).

## Identity: ids with drafts against drafts off (same launch), 256 tokens

| launch | spec | prompts equal |
|---|---|---|
| 0 | 0 | 24/24 |
| 0 | 2 | 24/24 |
| 0 | 1 | 12/12 |
| 0 | 3 | 12/12 |
| 0 | v3 | 12/12 |
| 1 | 2 | 24/24 |
| 1 | 0 | 24/24 |
| 1 | 3 | 12/12 |
| 1 | 1 | 12/12 |
| 1 | v3 | 12/12 |

Anchors (drafts off = P1's first 128 / O3's ids): 12/12

## Acceptance against the O3 probe's schedule (O3's 8 prompts)

| k | launch | accepted / proposed | probe schedule |
|---|---|---|---|
| 1 | 0 | 0.8701 (951/1093) | 0.8701 |
| 1 | 1 | 0.8701 (951/1093) | 0.8701 |
| 2 | 0 | 0.7691 (2478/3222) | 0.7691 |
| 2 | 1 | 0.7691 (2478/3222) | 0.7691 |
| 3 | 0 | 0.6586 (1356/2059) | 0.6586 |
| 3 | 1 | 0.6586 (1356/2059) | 0.6586 |

## Recorded: tok/s on ONE GPU (four ranks time-share it: drafts cannot overlap here)

| spec | p5 | code | prose | all |
|---|---|---|---|---|
| 0 | 61.62 | 61.2 | 63.62 | 62.15 |
| 1 | 63.85 | 62.59 | 61.07 | 62.5 |
| 2 | 63.48 | 67.06 | 57.26 | 62.6 |
| 3 | 60.12 | 63.54 | 49.6 | 57.75 |
| v3 | 63.29 | 50.52 | 48.71 | 54.17 |

## Recorded: acceptance and the last stage's drafting

| spec | class | rounds | kept/proposed | tokens a round |
|---|---|---|---|---|
| 1 | code | 1042 | 0.9635 | 1.9635 |
| 1 | prose | 1144 | 0.785 | 1.785 |
| 1 | p5 | 938 | 0.9041 | 1.9041 |
| 1 | all | 3124 | 0.8803 | 1.8803 |
| 2 | code | 1448 | 0.9155 | 2.826 |
| 2 | prose | 1780 | 0.6502 | 2.2989 |
| 2 | p5 | 1320 | 0.8571 | 2.7091 |
| 2 | all | 4548 | 0.7946 | 2.5858 |
| 3 | code | 576 | 0.8515 | 3.5486 |
| 3 | prose | 802 | 0.5196 | 2.5511 |
| 3 | p5 | 544 | 0.7598 | 3.2794 |
| 3 | all | 1922 | 0.6873 | 3.0562 |
| v3 | code | 626 | 0.9366 | 3.2652 |
| v3 | prose | 952 | 0.7594 | 2.1471 |
| v3 | p5 | 604 | 0.8847 | 2.957 |
| v3 | all | 2182 | 0.861 | 2.692 |

| spec | frames | skipped | MTP draft ms a step | refill ms a step | top-2 ms a round | depths |
|---|---|---|---|---|---|---|
| 1 | 6248 | 374 | 2.944 | 0.491 | 0.000 | {0: 0, 1: 3124} |
| 2 | 13624 | 1864 | 2.803 | 0.597 | 0.000 | {0: 0, 1: 20, 2: 4528} |
| 3 | 7672 | 1798 | 2.971 | 0.697 | 0.000 | {0: 0, 1: 6, 2: 4, 3: 1912} |
| v3 | 6470 | 596 | 3.653 | 0.720 | 0.552 | {0: 76, 1: 662, 2: 706, 3: 738} |

Rings (k = 2, launch 0): {"0": {"slots": 3, "steps": 290, "graphs_captured": 3, "bytes": 133693440}, "1": {"slots": 3, "steps": 290, "graphs_captured": 3, "bytes": 133693440}, "2": {"slots": 3, "steps": 290, "graphs_captured": 3, "bytes": 133693440}, "3": {"slots": 3, "steps": 256, "graphs_captured": 3, "bytes": 133693440}}

## Ornith projection (computed, not measured)

Inputs: {"stage_ms": [16.5, 15.35, 15.68, 19.68], "R_ms": 7.99, "hop_ms": 2.0, "mtp_ms": 4.83, "refill_ms": 1.028, "refill_over_draft_35b": 0.2129, "top2_ms_per_round_35b": 0.552}

Drafts off: 13.3 tok/s.

| class | k | tok/s (hop = latency) | tok/s (hop occupies the stage) | survey T_B |
|---|---|---|---|---|
| code | 1 | 19.37 | 18.64 | 19.24 |
| code | 2 | 22.43 | 21.47 | 21.64 |
| code | 3 | 23.98 | 22.88 | 22.09 |
| prose | 1 | 18.6 | 17.93 | 17.99 |
| prose | 2 | 20.68 | 19.86 | 18.68 |
| prose | 3 | 21.32 | 20.45 | 17.7 |
