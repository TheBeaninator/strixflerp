# ORNITH-O1O2 on one node: 35B, world 4, TCP on 127.0.0.1, one GPU

Written 2026-11-13T18:49:52Z. Criteria: `gate/ornith-o1o2.toml` (committed before the first GPU run). Binary sha256 0b9f59243400770c223d0ef3c81a69cfd998f5c978111105ed9c0f471c1632b2.

**FAIL**: 46 failure(s).

## Identity (ids / every logits sha256 against P1)

| launch | rows | reply:S | prompts ids equal | prompts sha equal |
|---|---|---|---|---|
| 0 | pageable | full:1 | 4/4 | 4/4 |
| 0 | pageable | token:1 | 4/4 | n/a |
| 1 | pinned | full:1 | 4/4 | 4/4 |
| 1 | pinned | token:1 | 4/4 | n/a |
| 1 | pinned | full:4 | 4/4 | 4/4 |
| 1 | pinned | full:16 | 4/4 | 4/4 |
| 2 | pageable | token:1 | 4/4 | n/a |
| 2 | pageable | full:1 | 4/4 | 4/4 |
| 3 | pinned | full:16 | 4/4 | 4/4 |
| 3 | pinned | full:4 | 4/4 | 4/4 |
| 3 | pinned | token:1 | 4/4 | n/a |
| 3 | pinned | full:1 | 4/4 | 4/4 |

## O1: copy ms a call, ranks 1 / 2 / 3 (median over sessions; max)

| rows | rank 1 | rank 2 | rank 3 |
|---|---|---|---|
| pageable | 0.1053 (max 0.3279) | 0.0723 (max 0.2750) | 0.1737 (max 0.2978) |
| pinned | 0.1209 (max 0.1829) | 0.0364 (max 0.0589) | 0.1673 (max 0.1877) |

## Decode ms a token (rank 0 wall a call), by launch, S = 1, mean over the four prompts

| launch | rows | full | token |
|---|---|---|---|
| 0 | pageable | 20.438 | 17.159 |
| 1 | pinned | 15.554 | 15.479 |
| 2 | pageable | 15.543 | 14.996 |
| 3 | pinned | 15.397 | 15.152 |

## O2: TTFT s (rank 0 prefill_s) and summed stage chunk ms, pinned launches, reply full

| launch | prompt | tokens | S | TTFT s | chunk ms a stage (0/1/2/3) |
|---|---|---|---|---|---|
| 1 | bridge | 21 | 1 | 0.158 | 39.716/38.028/40.084/38.437 |
| 1 | code | 20 | 1 | 0.075 | 18.512/17.222/19.038/18.353 |
| 1 | rivers | 18 | 1 | 0.078 | 18.587/17.328/20.047/19.399 |
| 1 | records | 1808 | 1 | 6.028 | 1564.31/1481.04/1565.9/1380.53 |
| 1 | bridge | 21 | 4 | 0.103 | 56.042/60.76/71.022/54.923 |
| 1 | code | 20 | 4 | 0.100 | 51.912/57.627/69.072/52.987 |
| 1 | rivers | 18 | 4 | 0.094 | 47.905/54.392/64.287/49.934 |
| 1 | records | 1808 | 4 | 6.328 | 3557.064/3977.11/4805.475/3586.415 |
| 1 | bridge | 21 | 16 | 0.199 | 144.534/161.759/167.834/146.563 |
| 1 | code | 20 | 16 | 0.189 | 143.171/161.374/168.053/125.705 |
| 1 | rivers | 18 | 16 | 0.187 | 146.793/156.576/164.731/140.028 |
| 1 | records | 1808 | 16 | 6.300 | 4845.288/5677.956/5915.03/4573.66 |
| 3 | bridge | 21 | 16 | 0.229 | 118.282/146.985/165.648/129.185 |
| 3 | code | 20 | 16 | 0.189 | 144.678/162.851/168.281/137.689 |
| 3 | rivers | 18 | 16 | 0.183 | 150.371/145.315/159.505/130.284 |
| 3 | records | 1808 | 16 | 6.160 | 5378.316/3742.648/5772.352/3342.096 |
| 3 | bridge | 21 | 4 | 0.102 | 60.388/56.359/70.649/49.968 |
| 3 | code | 20 | 4 | 0.098 | 55.287/53.751/64.817/47.161 |
| 3 | rivers | 18 | 4 | 0.093 | 54.661/50.156/62.538/44.615 |
| 3 | records | 1808 | 4 | 6.227 | 3820.293/3822.413/4692.166/3310.599 |
| 3 | bridge | 21 | 1 | 0.085 | 21.439/19.648/22.125/20.752 |
| 3 | code | 20 | 1 | 0.081 | 19.598/18.844/21.112/20.085 |
| 3 | rivers | 18 | 1 | 0.077 | 18.402/17.427/19.857/19.204 |
| 3 | records | 1808 | 1 | 6.116 | 1583.12/1504.08/1595.26/1406.65 |

## Failures (first 40)

- launch 1 (pinned) full:1 bridge: rank 1 copy 0.1327 ms a call (bound < 0.1)
- launch 1 (pinned) full:1 bridge: rank 3 copy 0.1538 ms a call (bound < 0.1)
- launch 1 (pinned) full:1 code: rank 1 copy 0.1769 ms a call (bound < 0.1)
- launch 1 (pinned) full:1 code: rank 3 copy 0.1776 ms a call (bound < 0.1)
- launch 1 (pinned) full:1 rivers: rank 1 copy 0.1829 ms a call (bound < 0.1)
- launch 1 (pinned) full:1 rivers: rank 3 copy 0.1835 ms a call (bound < 0.1)
- launch 1 (pinned) full:1 records: rank 1 copy 0.1701 ms a call (bound < 0.1)
- launch 1 (pinned) full:1 records: rank 3 copy 0.1653 ms a call (bound < 0.1)
- launch 1 (pinned) token:1 bridge: rank 1 copy 0.1562 ms a call (bound < 0.1)
- launch 1 (pinned) token:1 bridge: rank 3 copy 0.1573 ms a call (bound < 0.1)
- launch 1 (pinned) token:1 code: rank 1 copy 0.1537 ms a call (bound < 0.1)
- launch 1 (pinned) token:1 code: rank 3 copy 0.1601 ms a call (bound < 0.1)
- launch 1 (pinned) token:1 rivers: rank 1 copy 0.1575 ms a call (bound < 0.1)
- launch 1 (pinned) token:1 rivers: rank 3 copy 0.1596 ms a call (bound < 0.1)
- launch 1 (pinned) token:1 records: rank 1 copy 0.1688 ms a call (bound < 0.1)
- launch 1 (pinned) token:1 records: rank 3 copy 0.1686 ms a call (bound < 0.1)
- launch 1 (pinned) full:4 bridge: rank 1 copy 0.1499 ms a call (bound < 0.1)
- launch 1 (pinned) full:4 bridge: rank 3 copy 0.1535 ms a call (bound < 0.1)
- launch 1 (pinned) full:4 code: rank 1 copy 0.1458 ms a call (bound < 0.1)
- launch 1 (pinned) full:4 code: rank 3 copy 0.1517 ms a call (bound < 0.1)
- launch 1 (pinned) full:4 rivers: rank 1 copy 0.1693 ms a call (bound < 0.1)
- launch 1 (pinned) full:4 rivers: rank 3 copy 0.1877 ms a call (bound < 0.1)
- launch 1 (pinned) full:4 records: rank 1 copy 0.1638 ms a call (bound < 0.1)
- launch 1 (pinned) full:4 records: rank 3 copy 0.1636 ms a call (bound < 0.1)
- launch 1 (pinned) full:16 code: rank 1 copy 0.1092 ms a call (bound < 0.1)
- launch 1 (pinned) full:16 rivers: rank 1 copy 0.1606 ms a call (bound < 0.1)
- launch 1 (pinned) full:16 rivers: rank 3 copy 0.1678 ms a call (bound < 0.1)
- launch 1 (pinned) full:16 records: rank 1 copy 0.1813 ms a call (bound < 0.1)
- launch 1 (pinned) full:16 records: rank 3 copy 0.1696 ms a call (bound < 0.1)
- launch 3 (pinned) full:16 bridge: rank 1 copy 0.1360 ms a call (bound < 0.1)
- launch 3 (pinned) full:16 bridge: rank 3 copy 0.1406 ms a call (bound < 0.1)
- launch 3 (pinned) full:16 code: rank 1 copy 0.1555 ms a call (bound < 0.1)
- launch 3 (pinned) full:16 code: rank 3 copy 0.1688 ms a call (bound < 0.1)
- launch 3 (pinned) full:16 records: rank 3 copy 0.1627 ms a call (bound < 0.1)
- launch 3 (pinned) full:4 bridge: rank 3 copy 0.1725 ms a call (bound < 0.1)
- launch 3 (pinned) full:4 code: rank 3 copy 0.1720 ms a call (bound < 0.1)
- launch 3 (pinned) full:4 rivers: rank 3 copy 0.1726 ms a call (bound < 0.1)
- launch 3 (pinned) full:4 records: rank 3 copy 0.1770 ms a call (bound < 0.1)
- launch 3 (pinned) token:1 bridge: rank 3 copy 0.1777 ms a call (bound < 0.1)
- launch 3 (pinned) token:1 code: rank 3 copy 0.1570 ms a call (bound < 0.1)

## Reading (written after the run; the criteria are unchanged)

- Criteria 1-3 PASS: every run's ids equal P1's; every logits sha256 equals P1's for pageable, pinned, and 4 and 16 sub-chunks (all four prompts, the 1,808-token one included). The token-only reply gives P1's ids, and rank 3 sends 88 B a reply where it sent 993 KB.
- Criterion 4 FAILS: with pinned rows, ranks 1 and 3 still take 0.12 and 0.17 ms a call (median). Pageable takes 0.11 and 0.17. On one node the copy is not the pageable-registration cost: both modes are equal within the noise, except rank 2 (0.072 -> 0.036). Nothing here shows whether pinned rows remove Ornith's 1.87 ms.
- Decode ms a token, launches 1-3 (launch 0 is the cold first launch): pageable 15.0-15.6, pinned 15.2-15.6, with no difference. Token reply against full: -0.23 to -0.58 ms a token.
- TTFT on one GPU: S=4 and S=16 add to S=1 (records 6.03-6.12 s -> 6.16-6.33 s; 20-token prompts 0.08 -> 0.10 -> 0.19 s). The four ranks time-share one GPU, so the overlap shows only as stage time summing past the wall time: at S=16 the four stages add to about 21 s inside a 6.3 s TTFT. This run cannot measure per-row cost at smaller sub-chunks.

## Projection for Ornith world 4 (computed, not measured)

- Decode: 82.4 ms (P5) - copy 3 x (1.87 - x) - reply (0.09 + 0.41). If pinned rows reach this node's 0.17 ms: 82.4 - 5.1 - 0.5 = 76.8 ms a token (13.0 tok/s, +7 %). If they do not help on Ornith: 82.4 - 0.5 = 81.9 ms (token reply only).
- TTFT, modelled as stage time = 17 ms fixed + rows x 6.9 ms (P4: 7.08 s / 1,024 rows; P5: decode 17 ms a step, a 21-row chunk ~125 ms), with TTFT(S) = (S + 3) x stage time of one sub-chunk:
  - 1K prompt: 28.4 s at S=1 -> 12.5 s at S=4 (2.3x) -> 8.7 s at S=16 (3.3x).
  - 21-token prompt: 0.65 s -> 0.37 s at S=4 (1.75x); S=16 gives 0.49 s, since the fixed part repeats per sub-chunk.
- A 4-node confirm reruns P5 with `--rows pinned --reply token` and `--run-specs on:full:1,on:full:4,on:full:16,on:token:1`, plus a `--rows pageable` launch A B A B. It needs ids and every sha256 equal to run 11121855, copy_ms a call on ranks 1-3, decode ms, and TTFT at 21 / 1K tokens. One four-worker window (holds on A B C D or the P5 set) of about 30 min.
