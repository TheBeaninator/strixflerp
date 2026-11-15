# ORNITH-KERN3: DECODE-KERN3 kernels at Ornith-1.5 397B dims (synthetic codes), control-node GPU

Criteria `gate/ornith-kern3.toml` (committed before the GPU run). Bench build/kern3/decode-kern3 (catalogue at 83d5366).

26 (case, x, tokens) results, 1300934 outputs compared; with any differing byte: 0 -> **PASS**. Cases off-load or failed: 1 [{'case': 'router-4096x512', 'rc': 0, 'load_start': 5.77, 'load_end': 6.03}]

| case | rows x k | differing | ref GB/s | new GB/s | new/ref | ref us | new us |
|---|---|---|---|---|---|---|---|
| router-4096x512 | 513 x 4096 | 0 of 1217862 | 84.9 (84.6..85.9) | 195.2 (192.5..196.4) | 2.298 | 99.14 | 43.13 |
| outproj-q4k-4096x8192 | 4096 x 8192 | 0 of 8192 | 129.4 (128.3..129.8) | 166.1 (164.3..166.4) | 1.284 | 146.17 | 113.81 |
| ssm-alpha-q4k-64x4096 | 64 x 4096 | 0 of 128 | - | - | - | - | - |
| attn-kv-q4k-512x4096 | 512 x 4096 | 0 of 1024 | - | - | - | - | - |
| attn-gate-q4k-8192x4096 | 8192 x 4096 | 0 of 16384 | - | - | - | - | - |
| attn-q-q4k-16384x4096 | 16384 x 4096 | 0 of 32768 | - | - | - | - | - |
| attn-qkv-q4k-12288x4096 | 12288 x 4096 | 0 of 24576 | - | - | - | - | - |

## Projection (computed, not measured; criteria [projection])

Saving a block 0.088 ms (router 0.056 + outproj 0.032).
Stage (15 blocks): 18.5 -> 17.17 ms a step.
World 4 token: 83.0 -> 77.70 ms = 12.05 -> 12.87 tok/s (projected).
