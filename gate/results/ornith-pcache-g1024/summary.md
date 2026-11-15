# ORNITH-PCACHE step 0: slot-resident prefix reuse on the 35B proxy (card a3656fc5e9, run card ef1ebae055)

2026-11-14, control-node GPU (tools/gpu-queue.sh), world 4 loopback, serve config ornith397.serve (context 65536, drafts 1, reuse_tail 32), one load a run.
Criteria gate/ornith-pcache.toml, committed before the first run. Sequence: T1 cold (19,533 tokens), T2 = T1 + reply + ~1.3K-token turn (warm), T3 warm,
short request, T2 cold, short request, T3 cold. Warm ids == cold ids is judged on 32 greedy tokens.

| run | reuse_grid | T1 cold | T2 warm / cold | speedup | T3 warm / cold | ids warm == cold | pass |
|---|---|---|---|---|---|---|---|
| v1 (ornith-pcache-v1, snapshot inside the final chunk) | 0 | 108 s (old build) | 6.3 s / 120.8 s | 19x | 0.8 / 109 s | NO (differ at token 23 and 19) | fail |
| A (ornith-pcache) | 0 | 27.22 s | 6.74 / 25.07 s | 3.72x | 0.74 / 25.03 s | yes | fail (speedup under 5x) |
| B (ornith-pcache-g1024) | 1024 | 23.84 s | 2.42 / 26.5 s | 10.93x | 0.72 / 26.31 s | yes | PASS |

v1 split the final chunk at prompt end minus 32, so the warm run's chunk boundaries differed from a cold run's and the greedy ids drifted at a near tie.
v2 snapshots only at the start of the last prompt forward and cuts all prompt forwards at the grid, so a warm run forwards the chunks a cold run does
(the GQA attention is chunk-invariant bit for bit) and its ids equal cold. Per-stage GTT growth [552, 551, 551, 552] MiB (KV + recurrent snapshot copy).
Hits 2 reusing 39936 tokens; the short requests miss and reset. The draft head (drafts 1) and the recurrent ring decode correctly after a restore.
