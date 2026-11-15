# Interface gate: residual.add/-/f32

Host control-node, 2026-11-11T00:16:16Z. Config `gate/interface/residual-add.toml` (sha256 524785ddba368772). Consumer API: `strix::kernels::residual_add(x, y, n, Act::F32, stream)`.

Overall: **PASS**

| provider | module that runs | built | parity vs CPU reference | adapter launches (foreign / origin) | gate |
|---|---|---|---|---|---|
| strixite | `kernel.strixite.residual` | yes | bit-identical to the CPU reference | - | PASS |
| gufo | `kernel.gufo.qwen_norm_rope` | yes | bit-identical to the CPU reference | 18375 / 0 | PASS |
| hipfire | `kernel.hipfire.tensor_ops` | yes | bit-identical to the CPU reference | 18375 / 0 | PASS |

Time per call (median of the repeats, launches queued back to back then one synchronize). A call is one launch of the consumer API; adapters may need several kernel launches for it. GB/s is the size of the call's operands divided by that time: for operands of a few MB it is not a memory bandwidth (they stay in the GPU's caches between launches), for the large shapes it is.

| provider | case | shape | median us | min..max us | GB/s | kernel launches | host load (1 min) |
|---|---|---|---|---|---|---|---|
| strixite | bench-decode-4x2560 | n=10240 (1 token x 4 streams x 2560) | 2.01 | 2.00..2.04 | 61.3 | 1 | 22.64 |
| strixite | bench-prefill-2048x4x2560 | n=20971520 (2048 tokens x 4 streams x 2560) | 1207.48 | 1201.44..7097.72 | 208.4 | 1 | 22.64 |
| gufo | bench-decode-4x2560 | n=10240 (1 token x 4 streams x 2560) | 2.19 | 2.02..2.21 | 56.1 | 1 | 22.64 |
| gufo | bench-prefill-2048x4x2560 | n=20971520 (2048 tokens x 4 streams x 2560) | 1207.96 | 1206.16..1331.37 | 208.3 | 1 | 22.64 |
| hipfire | bench-decode-4x2560 | n=10240 (1 token x 4 streams x 2560) | 1.98 | 1.97..1.99 | 62.0 | 1 | 11.1 |
| hipfire | bench-prefill-2048x4x2560 | n=20971520 (2048 tokens x 4 streams x 2560) | 1247.00 | 1235.10..1253.31 | 201.8 | 1 | 11.1 |

Providers against each other (raw outputs of the parity cases):

| a | b | cases identical | cases differing | largest gap |
|---|---|---|---|---|
| strixite | gufo | 9 | 0 | 0 |
| strixite | hipfire | 9 | 0 | 0 |
| gufo | hipfire | 9 | 0 | 0 |
