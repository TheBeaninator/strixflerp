# Interface gate: logits.topk/-/-

Host control-node, 2026-11-10T23:22:42Z. Config `gate/interface/logits-topk.toml` (sha256 2a71e648e119cbd4). Consumer API: `strix::kernels::logits_topk(logits, rows, ld, n_valid, out, nan, workspace, bytes, stream, mask, mask_words)`.

Overall: **PASS**

| provider | module that runs | built | parity vs CPU reference | adapter launches (foreign / origin) | gate |
|---|---|---|---|---|---|
| strixite | `kernel.strixite.logits_topk` | yes | bit-identical to the CPU reference | - | PASS |
| gufo | `kernel.gufo.qwen_sampling` | yes | bit-identical to the CPU reference | 5962 / 2 | PASS |

Time per call (median of the repeats, launches queued back to back then one synchronize). A call is one launch of the consumer API; adapters may need several kernel launches for it. GB/s is the size of the call's operands divided by that time: for operands of a few MB it is not a memory bandwidth (they stay in the GPU's caches between launches), for the large shapes it is.

| provider | case | shape | median us | min..max us | GB/s | kernel launches | host load (1 min) |
|---|---|---|---|---|---|---|---|
| strixite | bench-decode-1row | 1 row x 248,077 logits (a decode step) | 79.91 | 79.83..79.99 | 12.4 | 2 per call | 23.36 |
| strixite | bench-verify-4rows | 4 rows x 248,077 (an MTP verify window) | 132.69 | 132.42..132.89 | 29.9 | 2 per call | 23.36 |
| strixite | bench-16rows | 16 rows x 248,077 (the widest window; prefill needs one row) | 374.93 | 373.46..377.18 | 42.3 | 2 per call | 23.36 |
| gufo | bench-decode-1row | 1 row x 248,077 logits (a decode step) | 472.19 | 471.29..545.75 | 2.1 | 5 per row + 1 | 21.97 |
| gufo | bench-verify-4rows | 4 rows x 248,077 (an MTP verify window) | 389.58 | 371.88..1730.62 | 10.2 | 5 per row + 1 | 21.97 |
| gufo | bench-16rows | 16 rows x 248,077 (the widest window; prefill needs one row) | 1531.87 | 1481.61..3365.69 | 10.4 | 5 per row + 1 | 21.97 |

Providers against each other (raw outputs of the parity cases):

| a | b | cases identical | cases differing | largest gap |
|---|---|---|---|---|
| strixite | gufo | 13 | 2 | 0 |

Notes:

- gufo / nan-row-few-finite: differs from the reference (reported, not a pass criterion): mismatched=25, structure_ok=True, detail=row 0 slot 9: got (-inf, 3), reference (-inf, 4)
- gufo / pos-inf: differs from the reference (reported, not a pass criterion): mismatched=40, structure_ok=True, detail=row 0 slot 0: got (13.3182, 21998), reference (inf, 83519)
