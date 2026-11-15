# Recipe gate: qwen36-35b-a3b.flerp-moe

Host control-node, written 2026-11-11T18:37:45Z. **FAILED**.

Model: `/data/ark/models/Qwen__Qwen3.6-35B-A3B-GGUF/gguf/Qwen3.6-35B-A3B-Q4_K_M.gguf` (general.architecture qwen35moe). Reference: llama.cpp 86a24a182b on the CPU, the same prompt ids.

| stage | result | when (UTC) | what |
|---|---|---|---|
| check | PASS | 2026-11-11T16:38:46Z | flerp check (origin gufo): 38 modules, 13 recipe(s), 0 error(s), 84 warning(s); catalogue: flerp check (catalogue): 235 modules, 13 recipe(s), 0 error(s), 85 warning(s) |
| origin | PASS | 2026-11-11T16:38:47Z | 542/542 catalogue files byte-identical to `git archive f070a17`, 0 problem(s); 7 adapted files |
| build | PASS | 2026-11-11T17:04:03Z | 42 modules, 558 files in the tree; 542/542 origin files equal the export's; 135 files compiled in 3.2 s |
| cpu | PASS | 2026-11-11T16:40:56Z | prompt tokens {'bridge': 21, 'code': 20, 'rivers': 18, 'records': 1808}; model {'architecture': 'qwen35moe', 'layers': 40, 'hidden': 2048, 'intermediate': 16384, 'heads': 16, 'kv_heads': 2, 'vocab': 248320, 'mtp_layers': 1, 'expert_count': 256, 'expert_used_count': 8, 'expert_feed_forward_length': 512, 'expert_shared_feed_forward_length': 512} |
| llama | PASS | 2026-11-11T16:45:42Z | 4 prompts; dense baseline {'error': None, 'prompts': 4}; eval callback {'code': {'rc': 0, 'seconds': 11.1, 'tensors': 80, 'input_ids_equal': False, 'input_ids_are_a_prefix': True, 'input_tokens': 19}, 'rivers': {'rc': 0, 'seconds': 14.1, 'tensors': 80, 'input_ids_equal': True, 'input_ids_are_a_prefix': True, 'input_tokens': 18}} |
| gpu | FAIL | 2026-11-11T16:48:25Z | 2 failure(s) of [criteria] |

## The adapted files

Copies of the origin's files with the MoE path added, in their own module; the origin's stay as they are.

| file | origin lines | lines added | lines removed |
|---|---|---|---|
| `flerp/qwen35moe/src/models/qwen/forward.cpp` | 336 | 126 | 4 |
| `flerp/qwen35moe/src/models/qwen/hip/batched_decode.cpp` | 856 | 38 | 10 |
| `flerp/qwen35moe/src/models/qwen/hip/decode_step.cpp` | 271 | 13 | 0 |
| `flerp/qwen35moe/src/models/qwen/hip/model_loader.cpp` | 424 | 7 | 1 |
| `flerp/qwen35moe/src/models/qwen/hip/prefill_chunk.cpp` | 763 | 27 | 2 |
| `flerp/qwen35moe/src/models/qwen/state.hpp` | 233 | 32 | 0 |
| `flerp/qwen35moe/src/models/qwen/weights.cpp` | 344 | 185 | 6 |

## What was assembled

558 files of 42 modules; 16 are of origin `flerp`, the other 542 are the origin's. 135 files compiled:

| module | files compiled |
|---|---|
| `executor.flerp.qwen35moe` | 7 |
| `executor.gufo.qwen` | 26 |
| `executor.gufo.qwen_vision` | 4 |
| `format.gufo.gguf_kquant` | 4 |
| `kernel.flerp.moe_kquant` | 1 |
| `kernel.gufo.qfn_kernels` | 2 |
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
| `tool.flerp.gufo_qwen_generate` | 1 |
| `tool.flerp.qwen35moe_generate` | 1 |

Binary: sha256 `d353ccb428e4ebf5...`, 75103104 bytes.

## Verdict of the gpu stage

Criteria that did not hold:
- llama.cpp, rivers: first position: log-probabilities differ by up to 0.96672 nats (tolerance 0.5)
- llama.cpp, records: first position: log-probabilities differ by up to 0.8321 nats (tolerance 0.5)

## 1. The dense model through this binary (regression)

`/data/ark/models/Mapika__decider-2b-GGUF/gguf/decider-2b-v12-Q4_K_M.gguf`, the dense recipe's prompt ids, against its recorded GPU run (`gate/results/qwen35-decider-2b.gufo/gpu-runs.json`, assembled, repeat 1).

| prompt | tokens | generated ids |
|---|---|---|
| bridge | [128, 128] | equal |
| code | [128, 128] | equal |
| rivers | [28, 28] | equal |
| records | [128, 128] | equal |

## 2. The MoE model against itself

| prompt | tokens | repeats equal | traced run equal | argmax of the copied logits is the token |
|---|---|---|---|---|
| bridge | 128 | True | True | True |
| code | 128 | True | True | True |
| rivers | 128 | True | True | True |
| records | 128 | True | True | True |

## 3. Against llama.cpp on the same prompt ids

Log-probabilities are compared over the tokens of llama.cpp's top 20 that are among this side's 40 largest logits, in nats. Tolerance at the first generated position 0.5; a tie is a llama.cpp margin below 1.0.

| prompt | tokens this / llama | ids agree for | first position: max / mean abs difference (tokens) | at the first difference: llama margin, its two best, this side's token | max over compared positions (median) | result |
|---|---|---|---|---|---|---|
| bridge | 128 / 128 | 115 | 0.18439 / 0.06386 (20) | 0.04675, [513, 6564], 6564 | 1.89591 (0.35552) | pass |
| code | 128 / 128 | 128 (all) | 0.26306 / 0.125 (20) | none | 3.46647 (0.41823) | pass |
| rivers | 128 / 128 | 74 | 0.96672 / 0.23729 (20) | 0.23107, [725, 579], 579 | 2.83029 (0.53647) | FAIL |
| records | 128 / 128 | 88 | 0.8321 / 0.26863 (20) | 0.47091, [279, 25], 25 | 3.49669 (0.84786) | FAIL |

The same comparison on the dense model, through the origin's unmodified feed forward (the baseline; not a criterion):

| prompt | ids agree for | first position: max / mean abs difference | at the first difference: llama margin | max over compared positions (median) |
|---|---|---|---|---|
| bridge | 5 | 0.363 / 0.15944 | 0.00104 | 0.36737 (0.28143) |
| code | 81 | 0.16634 / 0.04857 | 0.06438 | 0.55879 (0.20321) |
| rivers | 28 (all) | 0.48584 / 0.11629 | none | 0.69422 (0.26375) |
| records | 1 | 0.31885 / 0.11266 | 0.02444 | 0.43933 (0.37909) |

Hidden states after a block (llama.cpp's `l_out-<block>` as its eval callback prints it: four decimals, the first and last three values of the first and last three tokens it evaluated; the tokens both sides show are compared, and the sums where it evaluated the whole prompt):

| prompt | block | values compared | max abs difference | mean abs difference | mean abs value | sum this / llama |
|---|---|---|---|---|---|---|
| code | 0 | 30 | 0.00119 | 0.00047 | 0.01393 | None |
| code | 1 | 30 | 0.00224 | 0.00065 | 0.02352 | None |
| code | 19 | 30 | 0.19855 | 0.01216 | 0.54374 | None |
| code | 38 | 30 | 0.03596 | 0.01352 | 0.19196 | None |
| code | 39 | 30 | 0.06591 | 0.01612 | 0.17759 | None |
| rivers | 0 | 36 | 0.00204 | 0.00057 | 0.03074 | [16.7611, 16.619678] |
| rivers | 1 | 36 | 0.00458 | 0.00092 | 0.04104 | [24.2417, 23.971758] |
| rivers | 19 | 36 | 0.21465 | 0.01177 | 0.55959 | [33.7811, 33.031296] |
| rivers | 38 | 36 | 0.09498 | 0.02007 | 0.19363 | [191.3272, 190.886459] |
| rivers | 39 | 36 | 0.09681 | 0.01933 | 0.20408 | [110.4895, 112.066185] |

Experts chosen (llama.cpp's `ffn_moe_topk-<block>`: the first and last three of a token's experts in selection order, for the first and last three tokens):

| prompt | block | tokens compared | same experts in the same order | same experts | slots equal |
|---|---|---|---|---|---|
| code | 0 | 5 | 5 | 5 | 30 / 30 |
| code | 1 | 5 | 4 | 5 | 28 / 30 |
| code | 19 | 5 | 2 | 3 | 24 / 30 |
| code | 38 | 5 | 1 | 1 | 23 / 30 |
| code | 39 | 5 | 4 | 4 | 28 / 30 |
| rivers | 0 | 6 | 2 | 2 | 28 / 36 |
| rivers | 1 | 6 | 5 | 5 | 35 / 36 |
| rivers | 19 | 6 | 2 | 3 | 29 / 36 |
| rivers | 38 | 6 | 3 | 4 | 32 / 36 |
| rivers | 39 | 6 | 1 | 2 | 24 / 36 |

Three ways on `rivers`, the first 3 generated positions, against the executor's CPU reference (FP32 activations; its top 20 tokens a position): where activations are quantized. The decode step computes in FP32 like the CPU reference; the origin's prompt chunks quantize activations for the attention and DeltaNet projections, and llama.cpp's CPU kernels quantize them for every K-quant projection.

| against the CPU reference | tokens | positions compared | max abs difference | mean abs difference | largest on the best token |
|---|---|---|---|---|---|
| gpu, prompt through the decode step | [271, 248068, 271] | 3 | 0.00274 | 0.00068 | 3e-05 |
| gpu, prompt pass | [271, 248068, 271] | 3 | 0.62585 | 0.17467 | 0.01361 |
| llama.cpp | [271, 248068, 271] | 3 | 0.59105 | 0.17948 | 0.01873 |

CPU reference tokens [271, 248068, 271]. This executor against llama.cpp over the same positions (llama.cpp's top 20): prompt pass {'positions': 3, 'max_abs': 0.96672, 'mean_abs': 0.17673, 'top1_max_abs': 0.01494, 'first_position': {'compared': 20, 'max_abs': 0.96672, 'mean_abs': 0.23729}}; prompt through the decode step {'positions': 3, 'max_abs': 0.6692, 'mean_abs': 0.18751, 'top1_max_abs': 0.01875, 'first_position': {'compared': 20, 'max_abs': 0.6692, 'mean_abs': 0.31153}}.

## Speed and memory (a record, nothing was tuned)

Decode: median 56.44 tok/s over prompts and repeats, range [53.29, 57.74]; llama.cpp on 8 CPU threads: {'bridge': 24.938365908664707, 'code': 13.677645256592625, 'rivers': 24.73743742353551, 'records': 13.467192275982004}.
Prompt pass (time to the first token, s): {'bridge': [0.179, 0.129], 'code': [0.101, 0.099], 'rivers': [0.099, 0.097], 'records': [7.779, 7.897]} for {'bridge': 21, 'code': 20, 'rivers': 18, 'records': 1808} prompt tokens, so {'bridge': [117.3, 162.8], 'code': [198.0, 202.0], 'rivers': [181.8, 185.6], 'records': [232.4, 228.9]} tok/s; llama.cpp: {'bridge': 74.26057682787105, 'code': 87.1721781276288, 'rivers': 78.1596019070943, 'records': 123.38737377277162} tok/s.
The dense model through this binary: [117.26, 120.93, 118.77, 120.74] tok/s.
Load time and device memory the executor reports: {'dense': {'device_bytes': 969790420, 'weight_resident_bytes': 1274396192, 'weight_regions': 1, 'load_s': 0.444595, 'intermediate': 6144}, 'dense-traced': {'device_bytes': 969790420, 'weight_resident_bytes': 1274396192, 'weight_regions': 1, 'load_s': 0.694285, 'intermediate': 6144}, 'moe-1': {'device_bytes': 1872518100, 'weight_resident_bytes': 21713462624, 'weight_regions': 1, 'load_s': 28.7627, 'intermediate': 16384}, 'moe-2': {'device_bytes': 1872518100, 'weight_resident_bytes': 21713462624, 'weight_regions': 1, 'load_s': 28.8638, 'intermediate': 16384}, 'moe-traced': {'device_bytes': 1872518100, 'weight_resident_bytes': 21713462624, 'weight_regions': 1, 'load_s': 17.9963, 'intermediate': 16384}, 'moe-decode-fed': {'device_bytes': 1872518100, 'weight_resident_bytes': 21713462624, 'weight_regions': 1, 'load_s': 2.1546, 'intermediate': 16384}}.
GPU holds: [{'for': 'gpu stage', 'from': '2026-11-11T16:46:02Z', 'to': '2026-11-11T16:48:25Z', 'seconds': 143.2, 'pod_waiters_at_release': 0}]; waits: [].

## Without the GPU: the executor's CPU reference

2 tokens asked on 8 threads, exit 0, 71.2 s. bridge: ids [561, 4528], text ' The weight'.
bridge: llama.cpp's first 2 ids [561, 4528]; the GPU run's [561, 4528].

## Fidelity stages (separate from the verdict above)

`--stages reference,fidelity`: each engine against a full-precision reference, criteria in `[fidelity]`. Failed on 2026-11-11T18:37:45Z; see `fidelity.md`.

## Commands

```
tools/pod-sync.sh
tools/pod-run.sh './flerp gate recipe qwen36-35b-a3b.flerp-moe --stages check,origin,build,cpu,llama'
tools/pod-run.sh 'echo "moe-gpu | ./flerp gate recipe qwen36-35b-a3b.flerp-moe --stages gpu" > build/jobs-moe.txt; nohup setsid tools/gpu-queue.sh build/jobs-moe.txt > /dev/null 2>&1 &'
tools/pod-pull.sh gate/results/qwen36-35b-a3b.flerp-moe
```

Every command a stage ran is the `$` line above its output in the stage's evidence file.
