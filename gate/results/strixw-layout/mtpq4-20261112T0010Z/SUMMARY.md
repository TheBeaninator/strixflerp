# mtp_fc q8g64 -> q4g64 A/B, worker A, 2026-11-12 (criteria: gate/strixw-mtpq4-ab.toml, commit ca8fbfe)

File: strixw_patch on A, 4:48 wall (ionice idle, 4 threads), ~149 GB read / 74.6 GB written, max RSS 17 GB;
own verification OK (all component hashes, layout mtp_fc=q4g64, 74,573,103,104 bytes).
Hold 00:11:52Z -> release 00:19:10Z (first attempt 00:10Z refused by RES-7; REASON must be an env var, DRAIN=240
drained in 12 s). A /health ok on fleet weights at 00:20:20Z (keeper restore).

| | MTPQ4 | OLD |
|---|---|---|
| load to /health (s) | 32.2 | 30.6 |
| MTP accept decode2k | 404/660 = 0.612 | 403/686 = 0.587 |
| MTP accept decode30k | 408/647 = 0.631 | 426/667 = 0.639 |
| MTP accept quality prompts | 554/687 = 0.806 | 563/707 = 0.796 |
| decode ~2k tok/s x3 | 23.4 / 41.6 / 44.0 | 45.0 / 42.3 / 38.1 |
| decode ~30k tok/s x3 | 43.0 / 45.0 / 40.9 | 43.4 / 46.0 / 46.4 |
| prefill 7.5k tok/s x1 | 1140 | 773 (single run, outlier vs 1176/1278 last A/B) |

Output identity: 8/10 identical. FINDING: prompt 6 diverges at token 57 ("Arrival Time - ..." vs
"Duration = ..."), prompt 8 at token 14. Greedy output was expected to be identical because the main model checks
the drafts. It was not, so draft acceptance changes the outputs in this engine. A likely cause is that the batched
verify path is not bitwise equal to single-token decode. OLD vs OLD has not been run, so this is unconfirmed.
Verdict: fails identity -> MTPQ4 file deleted from A. Acceptance and decode are within noise.
