# Prefill A/B/C on F+A, 2026-11-11

Deployment f64b5ff2e28b, scratch copy of the serving conf (LEARN_DIR empty, PROMPT_CACHE_GIB=0, drafts on), one
8,163-token prompt asked twice per boot, greedy, 2 output tokens. Hold 18:53:35Z to 19:05:22Z (owner agent).

| arm | setting | prefill s (1st, 2nd request) | tok/s on the 2nd | rank 1 fill_chunk frames, 2nd request (ms) | prompt-logits sha256 |
|---|---|---|---|---|---|
| A | as served (PREFILL_SPLIT=256, zip) | 8.18, 6.1 | ~1,340 | 1496 1439 1435 1448 | 5396f3ae... |
| B | PREFILL_SPLIT empty | 8.85, 7.6 | ~1,075 | 1846 1819 1821 1816 | 5396f3ae... (equal to A) |
| C | tf-dsv41-lanes --profile 2 | did not run: `error: DriverUnavailable` on both ranks after 5 s | | | |
| A2 | as served, repeat | 7.07, 6.1 | ~1,340 | (see A2-rank1.log) | 5396f3ae... |

Reading:
- The split is worth about 0.38 s per 2,048-row chunk on the pair (1.82 s without, 1.44 s with), with identical logits.
- A's 1.44 s per chunk equals the one-GPU kernel sum of docs/PREFILL-DSV41.md (1.46 s): with the split's zip the pair
  is kernel-bound and the gathers are hidden. B's kernels would sum to about 1.19 s, so its unhidden gathers and host
  time cost about 0.6 s per chunk. Turning the split off (candidate 2) loses; candidate 1 (zip two whole chunks, one
  experts pass per 2,048 rows) keeps the overlap and is the one to build. Its predicted gain (0.27 s of 1.44 s) is
  still a prediction.
- The first long prompt after a boot is 1 to 2 s slower than the second.
- No per-phase table: arm C failed to open the device (cause not established; exact command in C-lane-rank0.cmd).

Restore: both `strixfold-pair.sh up` attempts made while the hold was still in place did not come back within a keep
tick (2 min). After the release at 19:05:22Z the keeper started the pair itself at 19:05:42Z and /health reported
dsv41, warming false at 19:07:42Z. Serving was down 18:53:35Z to 19:07:42Z. Nothing left on F or A.
