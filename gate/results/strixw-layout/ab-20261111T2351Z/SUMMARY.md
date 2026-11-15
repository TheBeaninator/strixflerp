# strixw layout A/B, worker A, 2026-11-11 (criteria: gate/strixw-layout-ab.toml, commit aada561)

Hold 23:51:27Z -> release 23:58:42Z; A healthy on :5300 (OLD weights) at 00:00Z.
All four headers fingerprint d1121c2771a0beb1: fleet ngram/vision/preprocessor reused, NEW ngram not copied.

| | NEW (exp_down q4g128) | OLD (fleet, q4g32) |
|---|---|---|
| load to /health (s) | 28.6 (fresh copy) | 30.7 (page-cache warm) |
| GTT used after load | 80.15 GB | 84.02 GB |
| decode ~2k (1903 tok) tok/s x3 | 44.6 / 44.5 / 45.3 | 43.8 / 44.5 / 42.1 |
| decode ~32k (29.6k tok) tok/s x2 | 40.3 / 42.3 | 44.9 / 43.8 |
| prefill 8k (7472 tok) tok/s x2 | 1167 / 1279 | 1176 / 1278 |
| prefill 32k (29583 tok) tok/s x2 | 1265 / 1293 | 1262 / 1291 |

Quality (OLD is not a reference): 6/10 greedy outputs identical, median first divergence 57,
shared-prefix top-1 0.888 mean (0.988 excluding prompt 4, which diverges at token 0), mean KL(top-20) 0.0064.

Verdict per criteria: NOT faster (decode 2k within noise, prefill equal, decode 32k slower); agreement
below the 0.98 bar. No roll. NEW left staged on A at /cache/models/qwen38-fnext-upstream-g128 (66 GB).
