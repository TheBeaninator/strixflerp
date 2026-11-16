# breakdown of r6-fix-s2 (fix-s2, slots 2, 3 reps, 225.8 s of main-phase wall)

aggregate decode tok/s per rep: 54.38, 52.22, 51.13; generated 11857 tokens; decode_seconds 292.3 (slot-seconds), prefill_seconds 118.0, PLE wait 2.78, cache saves (engine export block) 31.39

| state | slot 0 s (% of wall) | slot 1 s (% of wall) | calls (slot 0 / 1) | ms per call |
|---|---|---|---|---|
| gpu_prefill | 37.53 (16.6%) | 35.47 (15.7%) | 31 / 29 | 1216.6 |
| gpu_forward | 2.38 (1.1%) | 2.47 (1.1%) | 41 / 43 | 57.8 |
| gpu_cache | 1.79 (0.8%) | 2.13 (0.9%) | 119 / 121 | 16.3 |
| gpu_call | 0.00 (0.0%) | 0.00 (0.0%) | 0 / 0 | 0.0 |
| gpu_draft | 11.21 (5.0%) | 10.27 (4.5%) | 5953 / 5596 | 1.9 |
| gpu_resolve | 0.47 (0.2%) | 0.44 (0.2%) | 962 / 925 | 0.5 |
| gpu_batch | 109.79 (48.6%) | 105.89 (46.9%) | 962 / 925 | 114.3 |
| wait_chain | 5.62 (2.5%) | 6.00 (2.7%) |  |  |
| wait_prefill | 14.66 (6.5%) | 14.26 (6.3%) |  |  |
| wait_forward | 0.73 (0.3%) | 0.85 (0.4%) |  |  |
| wait_cache | 0.32 (0.1%) | 0.42 (0.2%) |  |  |
| wait_call | 0.00 (0.0%) | 0.00 (0.0%) |  |  |
| wait_batch_gpu | 51.64 (22.9%) | 47.50 (21.0%) |  |  |
| wait_resolving | 0.28 (0.1%) | 0.29 (0.1%) |  |  |
| wait_fair | 0.13 (0.1%) | 0.20 (0.1%) |  |  |
| wait_lead | 0.00 (0.0%) | 0.00 (0.0%) |  |  |
| wait_partner | 0.20 (0.1%) | 0.23 (0.1%) |  |  |

prefill rows per chunk: slot 0 1079, slot 1 1134

| batched verify by runs | batches | ms per batch | rows per batch | ms per row |
|---|---|---|---|---|
| 1 | 333 | 80.4 | 6.8 | 11.75 |
| 2 | 777 | 121.6 | 13.6 | 8.93 |

singles by the partner's state: partner_host_side=4, partner_idle=266, partner_prefill_waiting=63
