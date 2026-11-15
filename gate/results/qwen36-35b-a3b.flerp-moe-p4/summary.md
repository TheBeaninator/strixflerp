# Phase 5 P4: stages over the rail, F + D (hold 03:41:04Z-04:24:55Z 2026-11-12, owner "agent FLERP-P4")

Criteria: `gate/qwen36-35b-a3b.flerp-moe-p4.toml`, committed 4e7ed7d before the first worker run. The only later edit is
`[host] bundle`, moved off /cache because /cache is mounted noexec (first q35 attempt exited 126 before loading anything).
Binary: build/flerp-p4 (recipe qwen36-35b-a3b.flerp-moe + `--host` taking one address per rank).

## 35B, world 2 over the rail (rank 0 F 198.51.100.50, rank 1 D 198.51.100.53): PASS

| configuration | prompts with ids + 128 logits-row sha256 equal to P1 world 1 |
|---|---|
| 2:even (0,20,40):on | 4/4 |
| 2:even:off | 4/4 |
| 2:0,7,40:on | 4/4 |
| 2:0,7,40:off | 4/4 |

Bit equality holds across two machines. Loads are read over NFS on the 2.5 GbE management network at 0.13-0.14 GB/s: 10.5 GB in 80 s per rank (even cuts); 4.0 GB in 28 s and 17.2 GB in 124 s (lopsided cuts).
Decode ms per block: even cuts 0.42-0.43 (blocks 0-20), 0.52-0.53 (blocks 20-40); lopsided cuts 0.52-0.56 (blocks 0-7), 0.44 (blocks 7-40).
Relay overhead over the rail: 1.9-2.5 ms per decode token, 2.1-5.5 ms per chunk on short prompts and 20.7-23.7 ms per chunk on the 1,808-id prompt (chunk forward ~165-178 ms per block).

## M1: 160 KiB two-way exchange over the rail (TCP, host memory, eager, 300 reps)

median 0.101 ms, p10 0.041, p90 0.236, max 0.282. Design rule: 0.57 x expert ms per block > 2 x gather (0.20 ms).
Ornith decode is 1.08 ms per whole block, so 0.57 x 1.08 = 0.62 ms. The rule passes as long as experts take at least about 1/3 of a block.
35B: 0.57 x 0.43 = 0.25 ms, a marginal pass. Caveat: the exchange does not include device-to-host copies.

## Ornith-1.5 397B stages 0 and 1 of the world-4 plan: PASS (gate/results/ornith15-397b.p4/)

The file is reached only via /mnt/nv1 (NFSv4 over 2.5 GbE from control-node's USB disk); it is not on the rail mount /data/models-rail.
Load path: the placed loader's single throttled O_DIRECT reader, straight into GPU memory, nothing on /cache.

| stage | node | blocks | GB | fill s | GB/s | decode ms (15 blocks, eager) | ms/block | 1,024-row chunk ms/block | block check |
|---|---|---|---|---|---|---|---|---|---|
| 0 | F | 0-15 | 61.24 | 862 | 0.071 | 16.2 | 1.08 | 472-474 | 30/30 same experts, out err/RMS 7.2e-5 |
| 1 | D | 15-30 | 58.37 | 755 | 0.077 | 15.9 | 1.06 | 460 | 15/15 same experts, out err/RMS 5.0e-5 |

Not done: no TCP hop at Ornith shapes. The --staged tcp driver requires cuts that cover all 60 blocks, with the head on the last rank, so two stages of the world-4 plan cannot form a ring without a code change.
