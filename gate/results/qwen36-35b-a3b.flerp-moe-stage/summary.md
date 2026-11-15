# Gate of the stages and the relay: qwen36-35b-a3b.flerp-moe

Host control-node, written 2026-11-12T00:16:45Z. Criteria: `[criteria]` of `gate/qwen36-35b-a3b.flerp-moe-stage.toml` (committed before the first GPU run). Stage P2 of `docs/PHASE5-FOURNODE-DESIGN.md`.

| stage | result | when (UTC) | failures |
|---|---|---|---|
| loopback | PASS | 2026-11-11T23:52:08Z | 0  |
| tcp | PASS | 2026-11-11T23:57:13Z | 0  |
| faults | PASS | 2026-11-12T00:16:45Z | 0  |

## Invariance matrix (each cell: prompts whose ids and 128 logits-row hashes equal P1's)

| transport | configuration | equal |
|---|---|---|
| loopback | 1:even:on | 4/4 |
| loopback | 1:even:off | 4/4 |
| loopback | 2:even:on | 4/4 |
| loopback | 2:even:off | 4/4 |
| loopback | 2:0,7,40:on | 4/4 |
| loopback | 2:0,7,40:off | 4/4 |
| loopback | 4:even:on | 4/4 |
| loopback | 4:even:off | 4/4 |
| loopback | 4:0,1,4,38,40:on | 4/4 |
| loopback | 4:0,1,4,38,40:off | 4/4 |
| tcp | 2:0,7,40:on | 4/4 |
| tcp | 2:0,7,40:off | 4/4 |
| tcp | 4:even:on | 4/4 |
| tcp | 4:even:off | 4/4 |

## TCP faults

| case | pass | exit codes | seconds from fault to each rank's failure |
|---|---|---|---|
| control | True | {0: 0, 1: 0, 2: 0} | None |
| kill | True | {0: 3, 1: -9, 2: 3} | {0: 0.019, 2: 0.019} |
| sigkill | True | {0: 3, 1: 3, 2: -9} | {0: 0.025, 1: 0.025} |
| drop | True | {0: 3, 1: 3, 2: 3} | {0: 10.019, 1: 10.019, 2: 10.02} |
| stale-epoch | True | {0: 3, 1: 3, 2: 3} | {0: 0.0, 1: 0.0, 2: 0.0} |
| replay | True | {0: 3, 1: 3, 2: 3} | {0: 0.006, 1: 0.01, 2: 0.003} |
| epoch | True | {0: 3, 1: 3, 2: 3} | {0: 0.292, 1: 0.291, 2: 0.291} |

## Loopback faults

- {'fault': 'drop', 'fired': True, 'rank0_call_failed': True, 'all_poisoned': True}
- {'fault': 'stale-epoch', 'fired': True, 'rank0_call_failed': True, 'all_poisoned': True}
- {'fault': 'replay', 'fired': True, 'rank0_call_failed': True, 'all_poisoned': True}
- {'fault': 'bad-bytes', 'fired': True, 'rank0_call_failed': True, 'all_poisoned': True}
- {'fault': 'wrong-epoch-at-handshake', 'fired': True}

## Disk reads

| run | GB | fill s | when |
|---|---|---|---|
| loopback | 21.16 | 101.304 | 2026-11-11T23:52:08Z |
| tcp.txt rank 0 | 3.98 | 18.5856 | 2026-11-11T23:54:49Z |
| tcp.txt rank 1 | 17.18 | 83.282 | 2026-11-11T23:54:49Z |
| tcp.txt rank 0 | 5.47 | 25.9503 | 2026-11-11T23:57:13Z |
| tcp.txt rank 1 | 5.04 | 24.6485 | 2026-11-11T23:57:13Z |
| tcp.txt rank 2 | 5.45 | 27.5355 | 2026-11-11T23:57:13Z |
| tcp.txt rank 3 | 5.19 | 23.3135 | 2026-11-11T23:57:13Z |
| faults.txt rank 0 | 0.57 | 6.45211 | 2026-11-11T23:59:54Z |
| faults.txt rank 1 | 0.55 | 23.6186 | 2026-11-11T23:59:54Z |
| faults.txt rank 2 | 0.57 | 6.53718 | 2026-11-11T23:59:54Z |
| faults.txt rank 0 | 0.57 | 6.52798 | 2026-11-12T00:00:33Z |
| faults.txt rank 1 | 0.55 | 23.4066 | 2026-11-12T00:00:33Z |
| faults.txt rank 2 | 0.57 | 6.51412 | 2026-11-12T00:00:33Z |
| faults.txt rank 0 | 0.57 | 6.46482 | 2026-11-12T00:01:12Z |
| faults.txt rank 1 | 0.55 | 23.3946 | 2026-11-12T00:01:12Z |
| faults.txt rank 2 | 0.57 | 6.48447 | 2026-11-12T00:01:12Z |
| faults.txt rank 0 | 0.57 | 6.51664 | 2026-11-12T00:02:01Z |
| faults.txt rank 1 | 0.55 | 23.4586 | 2026-11-12T00:02:01Z |
| faults.txt rank 2 | 0.57 | 6.51651 | 2026-11-12T00:02:01Z |
| faults.txt rank 0 | 0.57 | 6.57038 | 2026-11-12T00:02:39Z |
| faults.txt rank 1 | 0.55 | 23.4088 | 2026-11-12T00:02:39Z |
| faults.txt rank 2 | 0.57 | 6.46828 | 2026-11-12T00:02:39Z |
| faults.txt rank 0 | 0.57 | 6.86318 | 2026-11-12T00:10:06Z |
| faults.txt rank 1 | 0.55 | 23.3067 | 2026-11-12T00:10:06Z |
| faults.txt rank 2 | 0.57 | 6.5836 | 2026-11-12T00:10:06Z |
| faults.txt rank 0 | 0.57 | 6.79522 | 2026-11-12T00:16:45Z |
| faults.txt rank 1 | 0.55 | 23.5695 | 2026-11-12T00:16:45Z |
| faults.txt rank 2 | 0.57 | 6.51287 | 2026-11-12T00:16:45Z |
| total | 75.23 | | |
