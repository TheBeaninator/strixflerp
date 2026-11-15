# Recipe gate: qwen35-decider-2b.gufo

Host control-node, written 2026-11-11T07:39:01Z. **PASSED**.

Model: `/data/ark/models/Mapika__decider-2b-GGUF/gguf/decider-2b-v12-Q4_K_M.gguf` (general.architecture qwen35). Reference: gufo at f070a17, built from a git export of `/data/ark-inbox/src/github.com__neuhaus__gufo`.

| stage | result | when (UTC) | what |
|---|---|---|---|
| check | PASS | 2026-11-11T05:57:25Z | flerp check (origin gufo): 38 modules, 12 recipe(s), 0 error(s), 54 warning(s); catalogue: flerp check (catalogue): 232 modules, 12 recipe(s), 0 error(s), 55 warning(s) |
| origin | PASS | 2026-11-11T05:57:25Z | 542/542 catalogue files byte-identical to `git archive f070a17`, 0 problem(s), 1493 origin files listed as unharvested |
| build | PASS | 2026-11-11T06:20:03Z | 39 modules, 545 files in the tree; 542/542 origin files equal the export's; 131 files compiled in 190.3 s |
| reference | PASS | 2026-11-11T06:04:28Z | 131 files compiled in 182.1 s; same source list as the assembled build: True |
| cpu | PASS | 2026-11-11T06:11:42Z | prompt tokens {'bridge': 21, 'code': 20, 'rivers': 18, 'records': 1808}; model {'architecture': 'qwen35', 'layers': 24, 'hidden': 2048, 'intermediate': 6144, 'heads': 8, 'kv_heads': 2, 'vocab': 248320, 'mtp_layers': 0} |
| llama | PASS | 2026-11-11T06:15:29Z | llama.cpp 86a24a182b, 4 prompts |
| gpu | PASS | 2026-11-11T07:39:01Z | token identity True; speed {'reference': 122.84, 'assembled': 121.58} |

## What was assembled

`./flerp assemble qwen35-decider-2b.gufo` wrote 545 files of 39 modules to `build/qwen35-decider-2b.gufo/tree`; 3 are written here (`flerp/gufo_qwen/CMakeLists.txt`, `flerp/gufo_qwen/generate.cpp`, `flerp/gufo_qwen/no_image.cpp`), the other 542 are the origin's. The origin's CMake files compiled 131 of them:

| module | files compiled |
|---|---|
| `executor.gufo.qwen` | 32 |
| `executor.gufo.qwen_vision` | 4 |
| `format.gufo.gguf_kquant` | 4 |
| `kernel.gufo.qwen_attention` | 8 |
| `kernel.gufo.qwen_dflash` | 1 |
| `kernel.gufo.qwen_gdn` | 6 |
| `kernel.gufo.qwen_linear_decode` | 8 |
| `kernel.gufo.qwen_linear_prefill` | 7 |
| `kernel.gufo.qwen_norm_rope` | 8 |
| `kernel.gufo.qwen_sampling` | 1 |
| `kernel.gufo.snapshot_copy` | 1 |
| `runtime.gufo.cache` | 14 |
| `runtime.gufo.dflash` | 5 |
| `runtime.gufo.ext_api` | 4 |
| `runtime.gufo.hip_host` | 3 |
| `runtime.gufo.mtp` | 4 |
| `runtime.gufo.platform` | 13 |
| `runtime.gufo.sampling` | 4 |
| `runtime.gufo.speculative` | 2 |
| `tool.flerp.gufo_qwen_generate` | 2 |

In the tree and not compiled (headers only, or code the Qwen 3.5 program does not build): `executor.gufo.deepseek_v4_flash`, `executor.gufo.qwen38_flash_next`, `kernel.gufo.core_headers`, `kernel.gufo.ds4_attention`, `kernel.gufo.ds4_backend`, `kernel.gufo.ds4_dspark`, `kernel.gufo.ds4_hc`, `kernel.gufo.ds4_indexer`, `kernel.gufo.ds4_linear`, `kernel.gufo.ds4_mmq`, `kernel.gufo.ds4_moe`, `kernel.gufo.ds4_norm_rope`, `kernel.gufo.qfn_kernels`, `kernel.gufo.qfn_mmq`, `kernel.gufo.qwen_ops`, `runtime.gufo.dspark`, `runtime.gufo.image`, `runtime.gufo.tp_verbs`, `serve.gufo.http`.

Assembled binary: sha256 `049125b3524f9b49...`, 64890824 bytes.
Reference binary: sha256 `9dc4ca1dbb270b89...`, 64895640 bytes (the build trees' paths are in the debug information, so the bytes are not expected to be equal).

## The executor and a qwen35moe file

`/data/ark/models/Qwen__Qwen3.6-35B-A3B-GGUF/gguf/Qwen3.6-35B-A3B-Q4_K_M.gguf`:

- cpu-reference: exit 1, opened: False. `error: Qwen CPU generator (general.architecture=qwen35moe): Missing required Qwen tensor: blk.0.ffn_gate.weight`

## Tokens on the GPU: reference build against assembled build

| prompt | repeat | prompt tokens | prompt ids equal | tokens ref / asm | generated ids | decode tok/s ref / asm | first token s ref / asm |
|---|---|---|---|---|---|---|---|
| bridge | 1 | 21 | True | 128 / 128 | equal | 124.0 / 124.59 | 0.031 / 0.027 |
| code | 1 | 20 | True | 128 / 128 | equal | 124.88 / 125.12 | 0.021 / 0.019 |
| rivers | 1 | 18 | True | 28 / 28 | equal | 122.02 / 121.36 | 0.019 / 0.019 |
| records | 1 | 1808 | True | 128 / 128 | equal | 123.33 / 121.34 | 0.376 / 0.383 |
| bridge | 2 | 21 | True | 128 / 128 | equal | 124.29 / 123.53 | 0.027 / 0.028 |
| code | 2 | 20 | True | 128 / 128 | equal | 122.34 / 121.8 | 0.021 / 0.021 |
| rivers | 2 | 18 | True | 28 / 28 | equal | 118.05 / 117.74 | 0.021 / 0.021 |
| records | 2 | 1808 | True | 128 / 128 | equal | 121.36 / 121.33 | 0.415 / 0.417 |

Each side equal to itself across repeats: {'reference': True, 'assembled': True}.
Decode speed, median over prompts and repeats: {'reference': 122.84, 'assembled': 121.58} tok/s, relative difference -0.0102, tolerance 0.05, reference range [118.05, 124.88]: pass. One process a run, 128 tokens a prompt: a record, not a benchmark.
Device memory the executor reports: {'reference': {'device_bytes': 969790420, 'weight_resident_bytes': 1274396192, 'weight_regions': 1, 'load_s': 3.95083}, 'assembled': {'device_bytes': 969790420, 'weight_resident_bytes': 1274396192, 'weight_regions': 1, 'load_s': 0.426132}}.
GPU holds: [{'for': 'gpu stage', 'from': '2026-11-11T07:38:39Z', 'to': '2026-11-11T07:39:01Z', 'seconds': 21.6, 'pod_waiters_at_release': 0}]; waits: [].

## Other engines on the same prompt ids (a record, not a criterion)

First position where the greedy tokens leave the assembled GPU run's (repeat 1).

| engine | prompt | tokens compared | first difference |
|---|---|---|---|
| llama.cpp (CPU) | bridge | 128 | 5 |
| llama.cpp (CPU) | code | 128 | 81 |
| llama.cpp (CPU) | rivers | 28 | none |
| llama.cpp (CPU) | records | 128 | 1 |
| gufo CPU reference | bridge | 8 | 5 |

## Without the GPU: gufo's CPU reference (from the assembled build)

8 tokens asked on 8 threads, exit 0, 167.1 s. bridge: 8 tokens at 0.17 tok/s, text ' **What is the main component that supports'.
Against llama.cpp on the same prompt ids: bridge: 8 tokens compared, first difference none.

llama.cpp's tokenizer on the prompt texts against gufo's: bridge: equal, code: equal, rivers: equal, records: equal. Its CPU decode speed: bridge 55.4 tok/s, code 54.8 tok/s, rivers 52.4 tok/s, records 52.2 tok/s.

## Commands

```
tools/pod-sync.sh
tools/pod-run.sh './flerp gate recipe qwen35-decider-2b.gufo --stages check,origin,build,reference,cpu,llama'
tools/pod-run.sh 'echo "gufo-gpu | ./flerp gate recipe qwen35-decider-2b.gufo --stages gpu" > build/jobs-gufo.txt; nohup setsid tools/gpu-queue.sh build/jobs-gufo.txt > /dev/null 2>&1 &'
tools/pod-pull.sh gate/results/qwen35-decider-2b.gufo
```

Every command a stage ran is the `$` line above its output in the stage's evidence file.
