# ORNITH-O3 step two: MTP speculative decoding, 35B proxy (world 4, TCP 127.0.0.1, one GPU)

Written 2026-11-14T00:13:03Z. Criteria: `gate/ornith-o3-spec.toml` (committed before the first GPU run). Binary sha256 8d94fd6f5a5ab774a6fd06cbaba7941c8f3d32f6590b51eb74848dbb3b2253ef.

**PASS**: 0 failure(s).

## Identity: ids with drafts against drafts off (same launch), 256 tokens

| launch | spec | prompts equal |
|---|---|---|
| 0 | 0 | 3/3 |
| 0 | 2 | 3/3 |
| 0 | 1 | 3/3 |
| 0 | 3 | 3/3 |
| 0 | v3 | 3/3 |

Anchors (drafts off = P1's first 128 / O3's ids): 0/0

## Acceptance against the O3 probe's schedule (O3's 8 prompts)

| k | launch | accepted / proposed | probe schedule |
|---|---|---|---|

## Recorded: tok/s on ONE GPU (four ranks time-share it: drafts cannot overlap here)

| spec | p5 | code | prose | all |
|---|---|---|---|---|
| 0 | 63.67 | 65.41 | 66.65 | 65.24 |
| 1 | 65.58 | 69.0 | 64.36 | 66.31 |
| 2 | 52.67 | 69.85 | 55.55 | 59.36 |
| 3 | 41.48 | 62.41 | 38.89 | 47.59 |
| v3 | 61.2 | 66.4 | 61.07 | 62.89 |

## Recorded: acceptance and the last stage's drafting

| spec | class | rounds | kept/proposed | tokens a round |
|---|---|---|---|---|
| 1 | code | 16 | 1.0 | 2.0 |
| 1 | prose | 18 | 0.7222 | 1.7222 |
| 1 | p5 | 17 | 0.8235 | 1.8235 |
| 1 | all | 51 | 0.8431 | 1.8431 |
| 2 | code | 11 | 0.9545 | 2.9091 |
| 2 | prose | 15 | 0.5667 | 2.1333 |
| 2 | p5 | 13 | 0.6923 | 2.3846 |
| 2 | all | 39 | 0.7179 | 2.4359 |
| 3 | code | 9 | 0.8519 | 3.5556 |
| 3 | prose | 14 | 0.439 | 2.2857 |
| 3 | p5 | 11 | 0.6452 | 2.8182 |
| 3 | all | 34 | 0.6162 | 2.7941 |
| v3 | code | 11 | 0.9545 | 2.9091 |
| v3 | prose | 15 | 0.7727 | 2.1333 |
| v3 | p5 | 16 | 0.75 | 1.9375 |
| v3 | all | 42 | 0.8281 | 2.2619 |

| spec | frames | skipped | MTP draft ms a step | refill ms a step | top-2 ms a round | depths |
|---|---|---|---|---|---|---|
| 1 | 102 | 8 | 2.609 | 0.396 | 0.000 | {0: 0, 1: 51} |
| 2 | 117 | 22 | 2.717 | 0.575 | 0.000 | {0: 0, 1: 0, 2: 39} |
| 3 | 133 | 38 | 3.196 | 0.745 | 0.000 | {0: 0, 1: 1, 2: 1, 3: 32} |
| v3 | 106 | 11 | 2.923 | 0.562 | 0.614 | {0: 0, 1: 24, 2: 14, 3: 4} |

Rings (k = 2, launch 0): {"0": {"slots": 3, "steps": 39, "graphs_captured": 3, "bytes": 133693440}, "1": {"slots": 3, "steps": 39, "graphs_captured": 3, "bytes": 133693440}, "2": {"slots": 3, "steps": 39, "graphs_captured": 3, "bytes": 133693440}, "3": {"slots": 3, "steps": 31, "graphs_captured": 3, "bytes": 133693440}}
