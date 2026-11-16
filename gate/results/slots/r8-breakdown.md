# breakdown of r8-fix-s2b (fix-s2b, slots 2, 3 reps, 241.2 s of main-phase wall)

aggregate decode tok/s per rep: 53.25, 47.70, 48.04; generated 11949 tokens; decode_seconds 323.3 (slot-seconds), prefill_seconds 109.6, PLE wait 9.08, cache saves (engine export block) 37.07

| state | slot 0 s (% of wall) | slot 1 s (% of wall) | calls (slot 0 / 1) | ms per call |
|---|---|---|---|---|
| gpu_prefill | 42.61 (17.7%) | 31.54 (13.1%) | 29 / 30 | 1256.7 |
| gpu_forward | 2.66 (1.1%) | 2.52 (1.0%) | 43 / 42 | 61.0 |
| gpu_cache | 2.07 (0.9%) | 2.01 (0.8%) | 117 / 118 | 17.4 |
| gpu_call | 0.00 (0.0%) | 0.00 (0.0%) | 0 / 0 | 0.0 |
| gpu_draft | 12.40 (5.1%) | 10.23 (4.2%) | 6024 / 5661 | 1.9 |
| gpu_resolve | 0.49 (0.2%) | 0.43 (0.2%) | 989 / 943 | 0.5 |
| gpu_batch | 117.16 (48.6%) | 112.28 (46.6%) | 989 / 943 | 118.8 |
| wait_chain | 5.38 (2.2%) | 5.54 (2.3%) |  |  |
| wait_prefill | 6.87 (2.9%) | 9.42 (3.9%) |  |  |
| wait_forward | 0.79 (0.3%) | 0.82 (0.3%) |  |  |
| wait_cache | 0.39 (0.2%) | 0.43 (0.2%) |  |  |
| wait_call | 0.00 (0.0%) | 0.00 (0.0%) |  |  |
| wait_batch_gpu | 54.02 (22.4%) | 49.30 (20.4%) |  |  |
| wait_resolving | 0.28 (0.1%) | 0.33 (0.1%) |  |  |
| wait_fair | 0.27 (0.1%) | 0.27 (0.1%) |  |  |
| wait_lead | 0.00 (0.0%) | 0.00 (0.0%) |  |  |
| wait_partner | 0.21 (0.1%) | 0.21 (0.1%) |  |  |

prefill rows per chunk: slot 0 1135, slot 1 1106

| batched verify by runs | batches | ms per batch | rows per batch | ms per row |
|---|---|---|---|---|
| 1 | 438 | 87.8 | 6.8 | 12.91 |
| 2 | 747 | 127.8 | 13.4 | 9.54 |

singles by the partner's state: partner_idle=347, partner_prefill_waiting=91
