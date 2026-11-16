# slots replay compare (2026-11-15T04:29:06Z)

baseline: r2-acct-s2  (binary /data/projects/strixite-331a35995a/build/strix/strix_server, slots 2)
candidate: r6-fix-s2, r8-fix-s2b  (binary /data/projects/strixite-331a35995a/build/strix/strix_server.fix, slots 2)

| metric | baseline median | min | max | candidate median | min | max | ratio |
|---|---|---|---|---|---|---|---|
| aggregate decode tok/s | 50.077 | 48.025 | 53.615 | 51.676 | 47.699 | 54.375 | 1.032 |
| per-thread decode tok/s (median) | 41.716 | 38.732 | 43.536 | 42.350 | 40.304 | 44.795 | 1.015 |
| prefill tok/s (median) | 667.500 | 602.800 | 769.300 | 750.850 | 553.100 | 774.100 | 1.125 |
| TTFT of a new thread during the other's prefill (s) | 7.508 | 6.933 | 9.044 | 7.430 | 6.011 | 11.796 | 0.990 |

token identity (captures matched by prompt ids): {"matched_prompts": 64, "diverged": 8, "first_divergence_index": 10, "unmatched_prompts": 44}

| criterion | result |
|---|---|
| aggregate >= baseline x 1.03 | PASS |
| aggregate never below baseline x 0.99 | FAIL |
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

## candidate: arbiter accounting over the main phases (467.0 s wall)

| series | seconds or count | share of wall |
|---|---|---|
| strix_arbiter_calls_total{slot="0",state="gpu_batch"} | 1951.000 |  |
| strix_arbiter_calls_total{slot="0",state="gpu_cache"} | 236.000 |  |
| strix_arbiter_calls_total{slot="0",state="gpu_draft"} | 11977.000 |  |
| strix_arbiter_calls_total{slot="0",state="gpu_forward"} | 84.000 |  |
| strix_arbiter_calls_total{slot="0",state="gpu_prefill"} | 60.000 |  |
| strix_arbiter_calls_total{slot="0",state="gpu_resolve"} | 1951.000 |  |
| strix_arbiter_calls_total{slot="1",state="gpu_batch"} | 1868.000 |  |
| strix_arbiter_calls_total{slot="1",state="gpu_cache"} | 239.000 |  |
| strix_arbiter_calls_total{slot="1",state="gpu_draft"} | 11257.000 |  |
| strix_arbiter_calls_total{slot="1",state="gpu_forward"} | 85.000 |  |
| strix_arbiter_calls_total{slot="1",state="gpu_prefill"} | 59.000 |  |
| strix_arbiter_calls_total{slot="1",state="gpu_resolve"} | 1868.000 |  |
| strix_arbiter_prefill_rows_total{slot="0"} | 66358.000 |  |
| strix_arbiter_prefill_rows_total{slot="1"} | 66070.000 |  |
| strix_arbiter_seconds_total{slot="0",state="gpu_batch"} | 226.950 | 48.6% |
| strix_arbiter_seconds_total{slot="0",state="gpu_cache"} | 3.859 | 0.8% |
| strix_arbiter_seconds_total{slot="0",state="gpu_draft"} | 23.612 | 5.1% |
| strix_arbiter_seconds_total{slot="0",state="gpu_forward"} | 5.045 | 1.1% |
| strix_arbiter_seconds_total{slot="0",state="gpu_prefill"} | 80.133 | 17.2% |
| strix_arbiter_seconds_total{slot="0",state="gpu_resolve"} | 0.958 | 0.2% |
| strix_arbiter_seconds_total{slot="0",state="wait_batch_gpu"} | 105.657 | 22.6% |
| strix_arbiter_seconds_total{slot="0",state="wait_cache"} | 0.714 | 0.2% |
| strix_arbiter_seconds_total{slot="0",state="wait_call"} | 0.008 | 0.0% |
| strix_arbiter_seconds_total{slot="0",state="wait_chain"} | 10.997 | 2.4% |
| strix_arbiter_seconds_total{slot="0",state="wait_fair"} | 0.402 | 0.1% |
| strix_arbiter_seconds_total{slot="0",state="wait_forward"} | 1.524 | 0.3% |
| strix_arbiter_seconds_total{slot="0",state="wait_partner"} | 0.412 | 0.1% |
| strix_arbiter_seconds_total{slot="0",state="wait_prefill"} | 21.534 | 4.6% |
| strix_arbiter_seconds_total{slot="0",state="wait_resolving"} | 0.559 | 0.1% |
| strix_arbiter_seconds_total{slot="1",state="gpu_batch"} | 218.164 | 46.7% |
| strix_arbiter_seconds_total{slot="1",state="gpu_cache"} | 4.146 | 0.9% |
| strix_arbiter_seconds_total{slot="1",state="gpu_draft"} | 20.504 | 4.4% |
| strix_arbiter_seconds_total{slot="1",state="gpu_forward"} | 4.993 | 1.1% |
| strix_arbiter_seconds_total{slot="1",state="gpu_prefill"} | 67.006 | 14.3% |
| strix_arbiter_seconds_total{slot="1",state="gpu_resolve"} | 0.870 | 0.2% |
| strix_arbiter_seconds_total{slot="1",state="wait_batch_gpu"} | 96.805 | 20.7% |
| strix_arbiter_seconds_total{slot="1",state="wait_cache"} | 0.844 | 0.2% |
| strix_arbiter_seconds_total{slot="1",state="wait_call"} | 0.009 | 0.0% |
| strix_arbiter_seconds_total{slot="1",state="wait_chain"} | 11.533 | 2.5% |
| strix_arbiter_seconds_total{slot="1",state="wait_fair"} | 0.473 | 0.1% |
| strix_arbiter_seconds_total{slot="1",state="wait_forward"} | 1.670 | 0.4% |
| strix_arbiter_seconds_total{slot="1",state="wait_partner"} | 0.446 | 0.1% |
| strix_arbiter_seconds_total{slot="1",state="wait_prefill"} | 23.687 | 5.1% |
| strix_arbiter_seconds_total{slot="1",state="wait_resolving"} | 0.615 | 0.1% |
| strix_verify_batch_rows_by_runs_total{runs="1"} | 5258.000 |  |
| strix_verify_batch_rows_by_runs_total{runs="2"} | 20587.000 |  |
| strix_verify_batch_runs_total | 3819.000 |  |
| strix_verify_batch_seconds_by_runs_total{runs="1"} | 65.238 | 14.0% |
| strix_verify_batch_seconds_by_runs_total{runs="2"} | 189.938 | 40.7% |
| strix_verify_batches_by_runs_total{runs="1"} | 771.000 |  |
| strix_verify_batches_by_runs_total{runs="2"} | 1524.000 |  |
| strix_verify_batches_total | 2295.000 |  |
| strix_verify_single_total{partner="partner_host_side"} | 4.000 |  |
| strix_verify_single_total{partner="partner_idle"} | 613.000 |  |
| strix_verify_single_total{partner="partner_prefill_waiting"} | 154.000 |  |
