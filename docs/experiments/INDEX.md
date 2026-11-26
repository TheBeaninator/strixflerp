# Experiment index

Newest first. Verdict words: CONFIRMED, REFUTED, NEITHER (ran, fell between the confirm and refute lines), CLOSED
(bounded or built, not worth shipping), PROMOTED (in the champions ledger or the served build). Evidence paths
under `/data/pod/reports/2026-11/` are on control-node (the pod-reports archive). Seeded 2026-11-17 from those reports and the card
logs; Ornith-1.5 results before this date are in `docs/CHAMPIONS.md`.

## DeepSeek-V4.1-Flash 4-bit, StrixFold TP4 on F D C A

| date | experiment | card | verdict | number | evidence |
|---|---|---|---|---|---|
| 11-25 | SF-KEEP-PAIR: kept states and sf-disk on the TP4 serve, factory replay (13 streams, 68 turns) at 1, 2, 4, 8 in flight | 3ddb72f4b1 | REJECT (served values stay: PROMPT_DISK_GIB 2.5, POOL 22000000, PROMPT_CACHE_GIB 4); hypothesis CONFIRMED | cached share 0.7817 at every level = 99.4 % of the 0.7864 replay ceiling, 0 of 55 warm turns from 0; sf-disk off +0.6 % committed tok/s at c8 (20.189 vs 20.061, bar 8 %); ~20 tok/s flat from 1 to 8 in flight with prompt passes 0.75 of wall; warm rep 63.5 tok/s at 0.976; 13/13 identical | /data/pod/reports/2026-11/sf-keep-pair.md |
| 11-25 | TP4-SWITCHES L2_PREFETCH=1: MALL prefetch of the next kernels' weights on the served TP4 round | 8854252e67 | CONFIRMED (+2 to +3.5 % predicted); adopted into tp4-serve1-image-v9.conf | A B B A drafts off +3.93 % (A means 38.555, B means 40.07 tok/s), drafts on +2.23 %; replies 8/8 byte-equal to the served reference | /data/pod/reports/2026-11/tp4-switches.md |
| 11-25 | TP4-SWITCHES DRAFT_PAIR=1: drafter pairGather as one row-pair exchange | 8854252e67 | CLOSED (REJECT) | first A/B drafts off -0.05 %, drafts on +0.94 %; under the +1 % gate; replies 8/8 equal | /data/pod/reports/2026-11/tp4-switches.md |
| 11-25 | TP4-SWITCHES XINPLACE=1: one-row cat_rank straight into place | 8854252e67 | CLOSED (REJECT) | first A/B drafts off +0.66 % (predicted +0.3 to +0.6), drafts on +0.15 %; under the +1 % gate; replies 8/8 equal | /data/pod/reports/2026-11/tp4-switches.md |

## Qwen3.8-Flash-Next (strixite executor)

| date | experiment | card | verdict | number | evidence |
|---|---|---|---|---|---|
| 11-17 | FACTORY-REPLAY baseline: recorded FreeFitty traffic at 2 in flight on one worker (68 of the 207 exchanges of factory-v1, 13 of 27 streams, 3 reps) | 2587bc1599 | CONFIRMED (18 to 37 predicted); rep spread 3.1 % missed the 3 % bar | 23.75 committed tokens a second (reps 23.75, 23.34, 24.08; one stream alone 19.59); turn time p50 10.11 s, p95 57.30 s; 6484 prefilled and 39064 cached tokens a request; 2 of 13 streams identical to solo | /data/pod/reports/2026-11/factory-replay-baseline.md |
| 11-17 | FOLD-1: fold tiny launches into their producers | 75f1d18d2f | PROMOTED (in the p16fold build) | decode +1.5 % on identical requests | /data/pod/reports/2026-11/flashnext-vs-published.md |
| 11-17 | CHAIN-1: chained decode launches on the real forward | 40d8943aa2 | REFUTED | stalls at the fill that would pay; at the clean fill +0.16 ms a step (29.11 vs 29.02 ms), bytes identical | /data/pod/reports/2026-11/chain1.md |
| 11-17 | Decode launch gaps: what a kernel boundary costs | 09d26982ac | measured | 1.84 us idle a boundary, 926 boundaries, about 1.7 ms of a 28.8 ms forward (5.9 %); HIP graphs no gain | /data/pod/reports/2026-11/decode-launch-design.md |
| 11-17 | STACK-DECODE lever 2a: hc_inject folded into the next mix | 09d26982ac | CLOSED | fused pair 48.41 us vs 44.96 us unfused, bits identical | /data/pod/reports/2026-11/subthreshold-ledger.md |
| 11-17 | STACK-DECODE lever 1: hyper-connection kernel occupancy | 09d26982ac | CLOSED | no variant faster: 22.09 us (160 GB/s) down, 20.55 us (169 GB/s) up at the decode shape | /data/pod/reports/2026-11/subthreshold-ledger.md |
| 11-17 | HC-Q4: Q4 hyper-connection mixes, round to nearest | 89f1c108d8 | REFUTED (fidelity) | 28.55 to 27.09 ms a step, but mean KL 0.123 against a 0.02 bar and 0 of 8 prompts identical | /data/pod/reports/2026-11/hc-q4.md |
| 11-17 | HC-DECODE: byte floor of the hyper-connection mixes | 1c2d1592b3 | measured | 684 MB a token, floor 3.03 ms against 3.88 ms measured; gap under the 1 ms bar | /data/pod/reports/2026-11/hc-decode.md |
| 11-17 | V10-QSA cost: indexer share of decode | 8ed7cc2c42 | CONFIRMED | 0.80 % at 4K, 0.91 % at 16K, 1.70 % at 64K: not worth optimizing | /data/pod/reports/2026-11/v10-qsa-stability.md |
| 11-17 | V10-QSA stability: reuse the previous token's selection | 8ed7cc2c42 | REFUTED | token-to-token Jaccard 0.649 at 1,000 blocks, 0.427 at 16,000 (needed 0.9) | /data/pod/reports/2026-11/v10-qsa-stability.md |
| 11-17 | V10-BLOCKBOUND: per-block bound for the coarse draft head | 7abcf5bed1 | REFUTED | certified fraction 0 / 6,144 steps (q2 arm, per-block and per-row bound); q2 keeps the true top-1 on 6,123 of 6,144 | /data/pod/reports/2026-11/v10-draft-blockbound.md |
| 11-17 | FACTORY-CENSUS: where FreeFitty threads spend tokens and time (2.9 days, 30,836 turns) | bd1b7b515b | CONFIRMED (both) | queue wait is 71.5 % of thread wall time (85 % on 11-17); 84.8 % of prefilled tokens are recompute, in a tail (21 % of turns recompute over 8,192 tokens), half of it on threads that never changed node | /data/pod/reports/2026-11/factory-census.md |
| 11-16 | SFX-PKG panel p16 weights (strixw v5) | 1fd5fa6482 | PROMOTED (prefill) | prefill 1.046 to 1.112x (1700.4 vs 1618.4 tok/s at 8K), decode 1.003 to 1.005x, logits and ids identical on 8 prompts | /data/pod/reports/2026-11/2026-11-16-strixflerp-package.md |
| 11-16 | SFX-PKG fused residual / norm / gate kernel | 1fd5fa6482 | CLOSED (not built) | 192 of 288 launches removable = 0.27 ms of 30.2 ms (0.9 %) | /data/pod/reports/2026-11/2026-11-16-strixflerp-package.md |
| 11-16 | SFX-PKG IndexCache: cross-layer QSA reuse | 1fd5fa6482 | CLOSED (not built) | adjacent-layer recall 0.647 decode, 0.671 prefill at 32K (needed 0.75) | /data/pod/reports/2026-11/2026-11-16-strixflerp-package.md |
| 11-15 | PF-13: LDS-staged activation panel for few-row GEMVs | d24965cf3b, 4e9dc7391b | PROMOTED (rolled) | at 16 rows attn_qkv 97 to 143 GB/s, lm_head 115 to 167 GB/s, bit-identical; decode +0.2 % over 3 of 5 rolled workers, inconclusive (strix-pf13-roll.md; the +1.5 to 2 % in strix-pf13-verify-gbps.md was a prediction) | /data/pod/reports/2026-11/strix-pf13-verify-gbps.md, /data/pod/reports/2026-11/strix-pf13-roll.md |
| 11-15 | STRIX-ZPORT: arm Z prefill levers ported to strixite | 85e3d36cf5 | REFUTED | router + combine are 3.0 % of a chunk; LIVE-WAVE 1.0038 to 1.0044x against a 1.05 bar | /data/pod/reports/2026-11/strix-zport.md |
| 11-15 | PF-SLOTS-TTFT: bound the prefill chunk for a waiting slot | 85205e0033 | REFUTED | first token 7.7 to 5.2 s (predicted about 1 s); aggregate 0.73x; a chunk has a fixed cost of about 0.45 s | /data/pod/reports/2026-11/strixite-slots-2.md |
| 11-15 | PF-SLOTS-SAVE: cache save takes the GPU turn once | fb8cf88003 | mechanism CONFIRMED, one of six criteria failed | save block 0.80 to 0.18 s a request | /data/pod/reports/2026-11/strixite-slots-2.md |
| 11-14 | PF-SLOTS: where the second slot's time goes | 331a35995a | measured | two slots = 1.17x one slot (48.0 vs 41.7 tok/s); an extra verify row costs 6 ms; greedy output at two slots depends on the other slot | /data/pod/reports/2026-11/strixite-slots.md |

## Qwen3.6-35B-A3B proxy and method (gufo-derived executor)

| date | experiment | card | verdict | number | evidence |
|---|---|---|---|---|---|
| 11-17 | V10-SUBCHUNK: stage tok/s against sub-chunk rows | bcf5450397 | NEITHER | 1,024 vs 256 rows = 1.567x (confirm 2.0, refute 1.5); 2,048 rows adds nothing (7,081 vs 7,077 tok/s reconstructed) | /data/pod/reports/2026-11/v10-subchunk.md |
| 11-17 | V10-NOISE: decode spread of one binary on K | aceca5acc3 | NEITHER | CV 0.61 % at 32 tokens, 0.84 % at 64 (confirm 0.70 %); 3 reps a side judge a 3 % bar; a loaded host moves a step from 16.0 to 21.4 ms | /data/pod/reports/2026-11/v10-noise-floor.md |

## Not run

| experiment | card | why |
|---|---|---|
| V10-STREAMS: 1, 2, 4, 8 decode streams against a byte model | 19aa2f3c03 | waits for MULTI-DECODE defc70210f (the 35B tool decodes one sequence) |
| CHAIN-2: worker ABBA of chained decode | 1ea1060dfe | superseded by CHAIN-1's result |
