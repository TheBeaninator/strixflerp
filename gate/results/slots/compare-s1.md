# slots replay compare (2026-11-15T04:36:31Z)

baseline: r3-acct-s1  (binary /data/projects/strixite-331a35995a/build/strix/strix_server, slots 1)
candidate: r9-fix-s1  (binary /data/projects/strixite-331a35995a/build/strix/strix_server.fix, slots 1)

| metric | baseline median | min | max | candidate median | min | max | ratio |
|---|---|---|---|---|---|---|---|
| aggregate decode tok/s | 41.738 | 33.848 | 45.926 | 38.944 | 37.744 | 43.441 | 0.933 |
| per-thread decode tok/s (median) | 73.196 | 73.164 | 74.230 | 72.696 | 71.241 | 75.546 | 0.993 |
| prefill tok/s (median) | 897.900 | 865.700 | 899.000 | 902.800 | 898.400 | 990.500 | 1.005 |
| TTFT of a new thread during the other's prefill (s) | 7.294 | 6.294 | 10.944 | 7.724 | 6.555 | 13.908 | 1.059 |

token identity (captures matched by prompt ids): {"matched_prompts": 54, "diverged": 0, "first_divergence_index": null, "unmatched_prompts": 0}

| criterion | result |
|---|---|
| aggregate >= baseline x 1.03 | FAIL |
| aggregate never below baseline x 0.99 | FAIL |
| token ids identical on every matched prompt | PASS |
| TTFT median <= baseline median + 0.050 s | FAIL |

VERDICT: FAIL
