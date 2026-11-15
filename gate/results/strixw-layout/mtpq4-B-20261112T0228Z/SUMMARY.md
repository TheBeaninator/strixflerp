# mtp_fc q8g64 -> q4g64 A/B with control, worker B, 2026-11-12 (criteria: gate/strixw-mtpq4-ab2.toml, unchanged)

Hold 02:28:03Z -> release 02:42:39Z (DRAIN=240, free after 12 s). Driver on control-node (driver.sh): strix-solo down,
arms OLD-a, MTPQ4, OLD-b as user units ab2-<arm> on :5310 (arm.sh, B's live args), cleanup trap verified
units=0 and :5310 free, then released. Trap path dry-run before the hold: dry.log.

| | OLD-a | MTPQ4 | OLD-b |
|---|---|---|---|
| load to /health (s) | 72.1 | 65.9 | 68.0 |
| MTP accept decode2k | 410/623 = 0.658 | 404/694 = 0.582 | 409/715 = 0.572 |
| MTP accept decode30k | 414/646 = 0.641 | 397/679 = 0.585 | 408/665 = 0.614 |
| MTP accept quality (20 prompts) | 2085/2654 = 0.786 | 2106/2668 = 0.789 | 2085/2654 = 0.786 |
| rejections accepted=0/1/2/3+ (quality) | 63/56/40/73 | 66/51/43/80 | 63/56/40/73 |
| decode ~2k tok/s x3 | 44.5 / 44.1 / 43.3 | 41.9 / 41.1 / 43.1 | 43.0 / 39.9 / 42.6 |
| decode ~30k tok/s x3 | 42.5 / 46.5 / 42.5 | 41.4 / 41.2 / 40.8 | 41.7 / 43.2 / 42.5 |

Divergence (compare-*.json): OLD-a vs OLD-b identical 20/20 (the control is deterministic, quality counters equal
too). MTPQ4 vs each OLD: 13/20 identical; differing prompts 2,6,8,11,13,17,19, first divergence at
247,57,14,118,79,111,103 (median 103), the same against both OLD arms.

Verdict: FAIL. With OLD-a == OLD-b on all 20, the committed rule requires MTPQ4 identical too; 7 prompts differ, so
the q4 draft layer changes greedy output (a real difference, not run-to-run noise). Acceptance is within noise
(decode phases inside the OLD-a/OLD-b spread, quality +0.003); decode 2k within the OLD spread (39.9-44.5), decode 30k
40.8-41.4 sits just under the OLD minimum 41.7. /cache/models/qwen38-fnext-tpn-mtpq4 deleted from B.
