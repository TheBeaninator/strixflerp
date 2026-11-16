# breakdown of r5-acct-s2-cs512 (acct-s2-cs512, slots 2, 3 reps, 220.1 s of main-phase wall)

aggregate decode tok/s per rep: 53.70, 52.66, 55.86; generated 11901 tokens; decode_seconds 282.0 (slot-seconds), prefill_seconds 114.5, PLE wait 6.23, cache saves (engine export block) 35.21

| state | slot 0 s (% of wall) | slot 1 s (% of wall) | calls (slot 0 / 1) | ms per call |
|---|---|---|---|---|
| gpu_prefill | 37.68 (17.1%) | 39.40 (17.9%) | 48 / 74 | 631.9 |
| gpu_forward | 2.32 (1.1%) | 2.81 (1.3%) | 43 / 42 | 60.4 |
| gpu_cache | 1.74 (0.8%) | 2.89 (1.3%) | 113 / 126 | 19.4 |
| gpu_call | 0.00 (0.0%) | 0.00 (0.0%) | 0 / 0 | 0.0 |
| gpu_draft | 10.22 (4.6%) | 9.29 (4.2%) | 6059 / 5570 | 1.7 |
| gpu_resolve | 0.48 (0.2%) | 0.37 (0.2%) | 996 / 920 | 0.4 |
| gpu_batch | 100.12 (45.5%) | 94.51 (42.9%) | 996 / 920 | 101.6 |
| wait_chain | 5.15 (2.3%) | 5.04 (2.3%) |  |  |
| wait_prefill | 7.01 (3.2%) | 11.64 (5.3%) |  |  |
| wait_forward | 0.83 (0.4%) | 0.45 (0.2%) |  |  |
| wait_cache | 0.77 (0.4%) | 0.34 (0.2%) |  |  |
| wait_call | 0.00 (0.0%) | 0.00 (0.0%) |  |  |
| wait_batch_gpu | 44.23 (20.1%) | 43.25 (19.6%) |  |  |
| wait_resolving | 0.24 (0.1%) | 0.34 (0.2%) |  |  |
| wait_fair | 0.25 (0.1%) | 0.12 (0.1%) |  |  |
| wait_lead | 0.00 (0.0%) | 0.01 (0.0%) |  |  |
| wait_partner | 0.20 (0.1%) | 0.22 (0.1%) |  |  |

prefill rows per chunk: slot 0 686, slot 1 448

| batched verify by runs | batches | ms per batch | rows per batch | ms per row |
|---|---|---|---|---|
| 1 | 402 | 73.7 | 6.7 | 10.97 |
| 2 | 757 | 109.0 | 13.5 | 8.07 |

singles by the partner's state: partner_idle=328, partner_prefill_waiting=74
