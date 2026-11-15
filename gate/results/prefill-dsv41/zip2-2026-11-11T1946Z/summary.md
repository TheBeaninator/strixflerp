# SPEED-ZIP2 on F+A, 2026-11-11

Hold 19:46:32Z to 19:59:43Z (owner agent); keeper restored serving by 20:02:36Z. Deployed build
f64b5ff2e28b against recipe dsv41-flash.strixfold-tp2-zip2 (commit 66754d8), scratch conf, cache 0, drafts on,
greedy, one 8,163-token prompt asked twice per boot.

| arm | build, TF_DS_PREFILL_ZIP2 | prefill s (1st, 2nd) | rank 1 frames on the 2nd request (ms) | prompt-logits sha | GTT used per rank |
|---|---|---|---|---|---|
| A | deployed, unset | 8.05, 6.060 | 1473 1440 1438 1454 (2,048 rows each) | 5396f3ae... | 111.87 GB |
| Z0 | zip2 build, unset | 6.92, 6.061 | 1474 1442 1439 1452 | equal | 111.87 GB |
| Z1 | zip2 build, 1 | 39.7, 5.858 | 2707 (4,096 rows) 1445 1456 | equal | 114.37 GB |
| Z2 | zip2 build, 2 | 36.5, 5.668 | 2703 2715 (4,096 rows each) | equal | 114.37 GB |

State check (SA under the deployed build, SZ under zip2=1): a state kept at 2,048 inside a zipped unit equals the
reference bytes; a state exported by the deployed build imports under zip2 and the request resumes with
cached_tokens 2048 and the reference token ids; one prompt-logits sha per prompt, equal.

Reading:
- Correct: logits bit-identical in every arm, states interchangeable, the build with the setting off is the served one.
- Gain is small: a 4,096-row unit takes 2.70-2.71 s against 2 x 1.44 = 2.88 s, 0.085 s per 2,048 rows (6%), not the
  predicted 0.27 s. Whole prompt: -3.3% with 1, -6.5% with 2 (about 1,340 -> 1,440 tok/s).
- Cost: +2.5 GB of GTT per rank (MemAvailable 16.0 -> 13.4 GiB on rank 0, 14.4 -> 11.9 on rank 1), and the first long
  prompt after a boot takes 36-40 s instead of 7-8 s (rank 1 frames of 22, 9 and 8 s; cause not established).
- Why the prediction missed is not established. The one-GPU bench measured the experts pass alone; on the pair the
  saved experts time is evidently not all on the critical path.
