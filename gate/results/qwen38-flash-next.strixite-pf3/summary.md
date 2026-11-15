# Prefill gate: qwen38-flash-next.strixite-pf3

Host control-node, 2026-11-12T00:54:47Z. Config `gate/qwen38-flash-next.strixite-pf3.toml`.

- **FAIL** c16k: prefill tok/s (second request) at least +2.0% over base at [32768, 131072], every repeat above the base's max
- **PASS** c16k: server decode (MTP on) tok/s within 5% of base or inside its range
- **REPORT** c16k: server generated ids equal to base (yes)

| variant | prompt | logits = base | greedy = base | top-1 agree | KL mean / max | prefill tok/s base (med, min-max) | prefill tok/s | gain | decode T=1 base / variant | device GiB base / variant |
|---|---|---|---|---|---|---|---|---|---|---|
| c16k | 8192 | True | True (first diff None) | 1.0 | 0 / 0 | 882.35 (865.87-944.0) | 839.69 (650.48-860.97) | -4.83% | 31.65 / 30.99 | 77.75 / 81.98 |
| c16k | 32768 | True | True (first diff None) | 1.0 | 0 / 0 | 1095.33 (938.06-1356.37) | 984.66 (902.09-991.74) | -10.1% | 30.33 / 30.85 | 78.08 / 82.3 |
| c16k | 131072 | True | True (first diff None) | 1.0 | 0 / 0 | 912.22 (904.74-980.18) | 945.16 (925.75-950.48) | 3.61% | 29.71 / 28.94 | 79.32 / 83.55 |

Timeline base (second request, prompt 32768): per chunk wall / GPU busy / gap ms

- chunk 0: 10771.3 / 8085.3 / 2686.0, host between calls 0.02 ms
- chunk 1: 9354.7 / 9077.5 / 277.2, host between calls 0.05 ms
- chunk 2: 10842.8 / 10462.4 / 380.3, host between calls 0.0 ms
- chunk 3: 9804.1 / 9458.2 / 345.8

Top kernels (ms over the request): void strix::kernels:: 35972.0, strix::kernels:: 1111.3, __amd_rocclr_fillBufferAligned 0.1, __amd_rocclr_copyBuffer 0.0

Timeline c16k (second request, prompt 32768): per chunk wall / GPU busy / gap ms

- chunk 0: 16398.0 / 14192.8 / 2205.1, host between calls 0.0 ms
- chunk 1: 11617.3 / 10873.7 / 743.6

Top kernels (ms over the request): void strix::kernels:: 24290.8, strix::kernels:: 775.6, __amd_rocclr_fillBufferAligned 0.1, __amd_rocclr_copyBuffer 0.0

Server c16k vs base: identical True; decode short-chat-thinking [52.39415249141732, 57.49964321054511, 56.14927463796201] -> [51.033303995153226, 57.03398824161455, 56.9012909794693]; long-chat [67.58535936848634, 67.85131260564224, 67.98560441958807] -> [65.43001881841218, 68.20707216026217, 68.54020444642936]; raw-completion [50.80008077276666, 50.83121513307773, 41.26129853149348] -> [44.89660093338264, 51.11256343000537, 51.39579153823133]

GPU holds: bench 1451.8 s; trace base 130.6 s; trace c16k 111.8 s; server base 105.0 s; server c16k 152.4 s
Storage read by the bench processes: 85.34 GB
