# Prefill gate: qwen38-flash-next.strixite-pf2

Host control-node, 2026-11-12T03:07:36Z. Config `gate/qwen38-flash-next.strixite-pf2.toml`.

- **PASS** combine 8192: logits sha256 and greedy ids equal to base
- **PASS** combine 32768: logits sha256 and greedy ids equal to base
- **PASS** combine 131072: logits sha256 and greedy ids equal to base
- **FAIL** combine: prefill tok/s (second request) at least +0.5% over base at [32768, 131072], every repeat above the base's max

| variant | prompt | logits = base | greedy = base | top-1 agree | KL mean / max | prefill tok/s base (med, min-max) | prefill tok/s | gain | decode T=1 base / variant | device GiB base / variant |
|---|---|---|---|---|---|---|---|---|---|---|
| combine | 8192 | True | True (first diff None) | 1.0 | 0 / 0 | 1460.65 (769.44-1493.16) | 833.14 (768.67-846.95) | -42.96% | 30.32 / 31.13 | 77.75 / 77.75 |
| combine | 32768 | True | True (first diff None) | 1.0 | 0 / 0 | 897.51 (872.15-1415.12) | 899.47 (881.2-995.95) | 0.22% | 31.78 / 31.47 | 78.08 / 78.08 |
| combine | 131072 | True | True (first diff None) | 1.0 | 0 / 0 | 1049.98 (949.21-1139.58) | 1172.19 (1031.61-1218.57) | 11.64% | 29.6 / 30.13 | 79.32 / 79.32 |

GPU holds: 
Storage read by the bench processes: 90.15 GB
