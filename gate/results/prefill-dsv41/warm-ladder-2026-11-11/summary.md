# Prefill from a cache on the live pair, 2026-11-11 20:13Z-20:22Z

Served build f64b5ff2e28b, serving conf, drafts on, greedy, 0 other streams, no hold. `gate/prefill_dsv41_warm_ladder.py`:
resume a kept prefix, append new tokens, read the server's `prefill_seconds`. The script died on the last request
(HTTP 500, the pair exited; pod card PAIR-STALL c18e4b0bd2) before writing its JSON; rows below are its printed lines.

From the standard learned state (67,584 tokens, the librarian prefix):

| new tokens | prefill s, pass 1 | prefill s, pass 2 | new tok/s, pass 2 |
|---|---|---|---|
| 1,318 | 2.03 | 1.62 | 811 |
| 2,068 | 2.49 | (came back cached 0: 69,652 tokens re-read in 71.0 s) | |
| 3,068 | 4.35 | 3.30 | 929 |
| 5,068 | 7.51 | 6.02 | 842 |
| 9,068 | 12.62 | 10.49 | 864 |
| 17,068 | 34.24 | 19.81 | 861 |
| 33,068 | 46.86 | 37.47 | 883 |

From a 4,096-token state of our own (kept in RAM only):

| new tokens | prefill s, pass 1 | prefill s, pass 2 | new tok/s, pass 1 |
|---|---|---|---|
| 491 | 0.76 | 0.82 | 648 |
| 1,241 | 1.19 | 1.38 | 1,044 |
| 2,241 | 1.90 | 1.91 | 1,177 |
| 4,241 | 3.29 | 13.57 (stall) | 1,289 |
| 8,241 | 6.06 | 36.46 (stall) | 1,359 |
| 16,241 | 11.50 | 64.24 (stall) | 1,413 |
| 32,241 | 22.82 | pair exited | 1,413 |

Reading: appended tokens cost about 1,410 tok/s on a 4k state and about 880 tok/s on the 67.6k state, about 38% slower
at depth. Short appends pay a fixed 0.4-0.6 s. Pass 1 on the standard state was slower and uneven (17,068 tokens in
34 s). The stalls in the last rows are rank 1 frames of 5-20 s, not a property of the prompt size.
