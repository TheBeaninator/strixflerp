# slots replay compare (2026-11-15T04:29:06Z)

baseline: r2-acct-s2  (binary /data/projects/strixite-331a35995a/build/strix/strix_server, slots 2)
candidate: r5-acct-s2-cs512  (binary /data/projects/strixite-331a35995a/build/strix/strix_server.acct, slots 2)

| metric | baseline median | min | max | candidate median | min | max | ratio |
|---|---|---|---|---|---|---|---|
| aggregate decode tok/s | 50.077 | 48.025 | 53.615 | 53.704 | 52.660 | 55.856 | 1.072 |
| per-thread decode tok/s (median) | 41.716 | 38.732 | 43.536 | 40.833 | 39.929 | 45.891 | 0.979 |
| prefill tok/s (median) | 667.500 | 602.800 | 769.300 | 587.700 | 526.600 | 616.000 | 0.880 |
| TTFT of a new thread during the other's prefill (s) | 7.508 | 6.933 | 9.044 | 7.069 | 5.832 | 7.592 | 0.942 |

token identity (captures matched by prompt ids): {"matched_prompts": 38, "diverged": 4, "first_divergence_index": 19, "unmatched_prompts": 16}

| criterion | result |
|---|---|
| aggregate >= baseline x 1.03 | PASS |
| aggregate never below baseline x 0.99 | PASS |
| token ids identical on every matched prompt | FAIL |
| TTFT median <= baseline median + 0.050 s | PASS |

VERDICT: FAIL

## baseline: arbiter accounting over the main phases (221.6 s wall)

| series | seconds or count | share of wall |
|---|---|---|
| strix_arbiter_calls_total{slot="0",state="gpu_batch"} | 864.000 |  |
| strix_arbiter_calls_total{slot="0",state="gpu_cache"} | 127.000 |  |
| strix_arbiter_calls_total{slot="0",state="gpu_draft"} | 5190.000 |  |
| strix_arbiter_calls_total{slot="0",state="gpu_forward"} | 41.000 |  |
| strix_arbiter_calls_total{slot="0",state="gpu_prefill"} | 31.000 |  |
| strix_arbiter_calls_total{slot="0",state="gpu_resolve"} | 864.000 |  |
| strix_arbiter_calls_total{slot="1",state="gpu_batch"} | 939.000 |  |
| strix_arbiter_calls_total{slot="1",state="gpu_cache"} | 113.000 |  |
| strix_arbiter_calls_total{slot="1",state="gpu_draft"} | 5778.000 |  |
| strix_arbiter_calls_total{slot="1",state="gpu_forward"} | 44.000 |  |
| strix_arbiter_calls_total{slot="1",state="gpu_prefill"} | 28.000 |  |
| strix_arbiter_calls_total{slot="1",state="gpu_resolve"} | 939.000 |  |
| strix_arbiter_prefill_rows_total{slot="0"} | 33452.000 |  |
| strix_arbiter_prefill_rows_total{slot="1"} | 32630.000 |  |
| strix_arbiter_seconds_total{slot="0",state="gpu_batch"} | 97.113 | 43.8% |
| strix_arbiter_seconds_total{slot="0",state="gpu_cache"} | 2.138 | 1.0% |
| strix_arbiter_seconds_total{slot="0",state="gpu_draft"} | 9.151 | 4.1% |
| strix_arbiter_seconds_total{slot="0",state="gpu_forward"} | 2.371 | 1.1% |
| strix_arbiter_seconds_total{slot="0",state="gpu_prefill"} | 37.963 | 17.1% |
| strix_arbiter_seconds_total{slot="0",state="gpu_resolve"} | 0.402 | 0.2% |
| strix_arbiter_seconds_total{slot="0",state="wait_batch_gpu"} | 51.086 | 23.1% |
| strix_arbiter_seconds_total{slot="0",state="wait_cache"} | 0.300 | 0.1% |
| strix_arbiter_seconds_total{slot="0",state="wait_call"} | 0.007 | 0.0% |
| strix_arbiter_seconds_total{slot="0",state="wait_chain"} | 5.527 | 2.5% |
| strix_arbiter_seconds_total{slot="0",state="wait_fair"} | 0.131 | 0.1% |
| strix_arbiter_seconds_total{slot="0",state="wait_forward"} | 0.648 | 0.3% |
| strix_arbiter_seconds_total{slot="0",state="wait_partner"} | 0.224 | 0.1% |
| strix_arbiter_seconds_total{slot="0",state="wait_prefill"} | 7.680 | 3.5% |
| strix_arbiter_seconds_total{slot="0",state="wait_resolving"} | 0.323 | 0.1% |
| strix_arbiter_seconds_total{slot="1",state="gpu_batch"} | 102.836 | 46.4% |
| strix_arbiter_seconds_total{slot="1",state="gpu_cache"} | 1.990 | 0.9% |
| strix_arbiter_seconds_total{slot="1",state="gpu_draft"} | 10.399 | 4.7% |
| strix_arbiter_seconds_total{slot="1",state="gpu_forward"} | 2.486 | 1.1% |
| strix_arbiter_seconds_total{slot="1",state="gpu_prefill"} | 37.336 | 16.8% |
| strix_arbiter_seconds_total{slot="1",state="gpu_resolve"} | 0.514 | 0.2% |
| strix_arbiter_seconds_total{slot="1",state="wait_batch_gpu"} | 40.415 | 18.2% |
| strix_arbiter_seconds_total{slot="1",state="wait_cache"} | 0.269 | 0.1% |
| strix_arbiter_seconds_total{slot="1",state="wait_call"} | 0.004 | 0.0% |
| strix_arbiter_seconds_total{slot="1",state="wait_chain"} | 4.554 | 2.1% |
| strix_arbiter_seconds_total{slot="1",state="wait_fair"} | 0.091 | 0.0% |
| strix_arbiter_seconds_total{slot="1",state="wait_forward"} | 0.675 | 0.3% |
| strix_arbiter_seconds_total{slot="1",state="wait_partner"} | 0.298 | 0.1% |
| strix_arbiter_seconds_total{slot="1",state="wait_prefill"} | 19.475 | 8.8% |
| strix_arbiter_seconds_total{slot="1",state="wait_resolving"} | 0.266 | 0.1% |
| strix_verify_batch_rows_by_runs_total{runs="1"} | 2531.000 |  |
| strix_verify_batch_rows_by_runs_total{runs="2"} | 9678.000 |  |
| strix_verify_batch_runs_total | 1803.000 |  |
| strix_verify_batch_seconds_by_runs_total{runs="1"} | 29.337 | 13.2% |
| strix_verify_batch_seconds_by_runs_total{runs="2"} | 85.306 | 38.5% |
| strix_verify_batches_by_runs_total{runs="1"} | 373.000 |  |
| strix_verify_batches_by_runs_total{runs="2"} | 715.000 |  |
| strix_verify_batches_total | 1088.000 |  |
| strix_verify_single_total{partner="partner_host_side"} | 13.000 |  |
| strix_verify_single_total{partner="partner_idle"} | 277.000 |  |
| strix_verify_single_total{partner="partner_prefill_waiting"} | 83.000 |  |

## candidate: arbiter accounting over the main phases (220.1 s wall)

| series | seconds or count | share of wall |
|---|---|---|
| strix_arbiter_calls_total{slot="0",state="gpu_batch"} | 996.000 |  |
| strix_arbiter_calls_total{slot="0",state="gpu_cache"} | 113.000 |  |
| strix_arbiter_calls_total{slot="0",state="gpu_draft"} | 6059.000 |  |
| strix_arbiter_calls_total{slot="0",state="gpu_forward"} | 43.000 |  |
| strix_arbiter_calls_total{slot="0",state="gpu_prefill"} | 48.000 |  |
| strix_arbiter_calls_total{slot="0",state="gpu_resolve"} | 996.000 |  |
| strix_arbiter_calls_total{slot="1",state="gpu_batch"} | 920.000 |  |
| strix_arbiter_calls_total{slot="1",state="gpu_cache"} | 126.000 |  |
| strix_arbiter_calls_total{slot="1",state="gpu_draft"} | 5570.000 |  |
| strix_arbiter_calls_total{slot="1",state="gpu_forward"} | 42.000 |  |
| strix_arbiter_calls_total{slot="1",state="gpu_prefill"} | 74.000 |  |
| strix_arbiter_calls_total{slot="1",state="gpu_resolve"} | 920.000 |  |
| strix_arbiter_prefill_rows_total{slot="0"} | 32911.000 |  |
| strix_arbiter_prefill_rows_total{slot="1"} | 33171.000 |  |
| strix_arbiter_seconds_total{slot="0",state="gpu_batch"} | 100.116 | 45.5% |
| strix_arbiter_seconds_total{slot="0",state="gpu_cache"} | 1.736 | 0.8% |
| strix_arbiter_seconds_total{slot="0",state="gpu_draft"} | 10.221 | 4.6% |
| strix_arbiter_seconds_total{slot="0",state="gpu_forward"} | 2.323 | 1.1% |
| strix_arbiter_seconds_total{slot="0",state="gpu_prefill"} | 37.684 | 17.1% |
| strix_arbiter_seconds_total{slot="0",state="gpu_resolve"} | 0.481 | 0.2% |
| strix_arbiter_seconds_total{slot="0",state="wait_batch_gpu"} | 44.228 | 20.1% |
| strix_arbiter_seconds_total{slot="0",state="wait_cache"} | 0.773 | 0.4% |
| strix_arbiter_seconds_total{slot="0",state="wait_call"} | 0.004 | 0.0% |
| strix_arbiter_seconds_total{slot="0",state="wait_chain"} | 5.152 | 2.3% |
| strix_arbiter_seconds_total{slot="0",state="wait_fair"} | 0.253 | 0.1% |
| strix_arbiter_seconds_total{slot="0",state="wait_forward"} | 0.827 | 0.4% |
| strix_arbiter_seconds_total{slot="0",state="wait_partner"} | 0.197 | 0.1% |
| strix_arbiter_seconds_total{slot="0",state="wait_prefill"} | 7.005 | 3.2% |
| strix_arbiter_seconds_total{slot="0",state="wait_resolving"} | 0.242 | 0.1% |
| strix_arbiter_seconds_total{slot="1",state="gpu_batch"} | 94.509 | 42.9% |
| strix_arbiter_seconds_total{slot="1",state="gpu_cache"} | 2.892 | 1.3% |
| strix_arbiter_seconds_total{slot="1",state="gpu_draft"} | 9.294 | 4.2% |
| strix_arbiter_seconds_total{slot="1",state="gpu_forward"} | 2.814 | 1.3% |
| strix_arbiter_seconds_total{slot="1",state="gpu_prefill"} | 39.404 | 17.9% |
| strix_arbiter_seconds_total{slot="1",state="gpu_resolve"} | 0.370 | 0.2% |
| strix_arbiter_seconds_total{slot="1",state="wait_batch_gpu"} | 43.252 | 19.6% |
| strix_arbiter_seconds_total{slot="1",state="wait_cache"} | 0.339 | 0.2% |
| strix_arbiter_seconds_total{slot="1",state="wait_call"} | 0.004 | 0.0% |
| strix_arbiter_seconds_total{slot="1",state="wait_chain"} | 5.035 | 2.3% |
| strix_arbiter_seconds_total{slot="1",state="wait_fair"} | 0.121 | 0.1% |
| strix_arbiter_seconds_total{slot="1",state="wait_forward"} | 0.446 | 0.2% |
| strix_arbiter_seconds_total{slot="1",state="wait_lead"} | 0.010 | 0.0% |
| strix_arbiter_seconds_total{slot="1",state="wait_partner"} | 0.224 | 0.1% |
| strix_arbiter_seconds_total{slot="1",state="wait_prefill"} | 11.645 | 5.3% |
| strix_arbiter_seconds_total{slot="1",state="wait_resolving"} | 0.342 | 0.2% |
| strix_verify_batch_rows_by_runs_total{runs="1"} | 2702.000 |  |
| strix_verify_batch_rows_by_runs_total{runs="2"} | 10224.000 |  |
| strix_verify_batch_runs_total | 1916.000 |  |
| strix_verify_batch_seconds_by_runs_total{runs="1"} | 29.633 | 13.5% |
| strix_verify_batch_seconds_by_runs_total{runs="2"} | 82.496 | 37.5% |
| strix_verify_batches_by_runs_total{runs="1"} | 402.000 |  |
| strix_verify_batches_by_runs_total{runs="2"} | 757.000 |  |
| strix_verify_batches_total | 1159.000 |  |
| strix_verify_single_total{partner="partner_idle"} | 328.000 |  |
| strix_verify_single_total{partner="partner_prefill_waiting"} | 74.000 |  |
