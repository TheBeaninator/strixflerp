# breakdown of /home/pod/c331/r2-acct-s2 (acct-s2, slots 2, 3 reps, 221.6 s of main-phase wall)

aggregate decode tok/s per rep: 50.08, 53.62, 48.02; generated 11176 tokens; decode_seconds 281.1 (slot-seconds), prefill_seconds 124.2, PLE wait 4.80, cache saves (engine export block) 26.82

| state | slot 0 s (% of wall) | slot 1 s (% of wall) | calls (slot 0 / 1) | ms per call |
|---|---|---|---|---|
| gpu_prefill | 37.96 (17.1%) | 37.34 (16.8%) | 31 / 28 | 1276.3 |
| gpu_forward | 2.37 (1.1%) | 2.49 (1.1%) | 41 / 44 | 57.1 |
| gpu_cache | 2.14 (1.0%) | 1.99 (0.9%) | 127 / 113 | 17.2 |
| gpu_call | 0.00 (0.0%) | 0.00 (0.0%) | 0 / 0 | 0.0 |
| gpu_draft | 9.15 (4.1%) | 10.40 (4.7%) | 5190 / 5778 | 1.8 |
| gpu_resolve | 0.40 (0.2%) | 0.51 (0.2%) | 864 / 939 | 0.5 |
| gpu_batch | 97.11 (43.8%) | 102.84 (46.4%) | 864 / 939 | 110.9 |
| wait_chain | 5.53 (2.5%) | 4.55 (2.1%) |  |  |
| wait_prefill | 7.68 (3.5%) | 19.48 (8.8%) |  |  |
| wait_forward | 0.65 (0.3%) | 0.67 (0.3%) |  |  |
| wait_cache | 0.30 (0.1%) | 0.27 (0.1%) |  |  |
| wait_call | 0.01 (0.0%) | 0.00 (0.0%) |  |  |
| wait_batch_gpu | 51.09 (23.1%) | 40.42 (18.2%) |  |  |
| wait_resolving | 0.32 (0.1%) | 0.27 (0.1%) |  |  |
| wait_fair | 0.13 (0.1%) | 0.09 (0.0%) |  |  |
| wait_lead | 0.00 (0.0%) | 0.00 (0.0%) |  |  |
| wait_partner | 0.22 (0.1%) | 0.30 (0.1%) |  |  |

prefill rows per chunk: slot 0 1079, slot 1 1165

| batched verify by runs | batches | ms per batch | rows per batch | ms per row |
|---|---|---|---|---|
| 1 | 373 | 78.7 | 6.8 | 11.59 |
| 2 | 715 | 119.3 | 13.5 | 8.81 |

singles by the partner's state: partner_host_side=13, partner_idle=277, partner_prefill_waiting=83
