# Fidelity: qwen36-35b-a3b.flerp-moe

Host control-node, written 2026-11-11T18:37:45Z. **FAILED** under `[fidelity]` of `gate/qwen36-35b-a3b.flerp-moe.toml` (the criteria were committed before these runs). The recipe gate's own verdict is in `summary.md` and is not changed by this.

Each engine against a full-precision evaluation of the weights as shipped (K-quant weights dequantized, FP32 activations), on the same context: the prompt ids and llama.cpp's recorded continuation. KL(reference || engine) in nats over the 1000 tokens each side kept; max: the largest absolute log-probability difference over tokens the reference gives at least 0.001; top-1: the best token is the reference's.

Criteria that did not hold:
- gpu prompt pass, bridge position 0: compared tokens hold 0.98074 of the reference's mass, 0 above the floor missing
- gpu prompt pass, bridge position 4: compared tokens hold 0.99408 of the reference's mass, 0 above the floor missing
- gpu prompt pass, bridge position 24: compared tokens hold 0.99517 of the reference's mass, 0 above the floor missing
- gpu prompt pass, records position 0: KL 0.00566 is over 3.0 times llama.cpp's 0.00127 (floor 0.002)
- gpu prompt pass, records position 1: KL 0.00381 is over 3.0 times llama.cpp's 0.00072 (floor 0.002)
- gpu, prompt through the decode step, bridge position 0: compared tokens hold 0.98095 of the reference's mass, 0 above the floor missing
- gpu, prompt through the decode step, bridge position 4: compared tokens hold 0.99411 of the reference's mass, 0 above the floor missing
- gpu, prompt through the decode step, bridge position 24: compared tokens hold 0.99517 of the reference's mass, 0 above the floor missing

## The MoE model (held)

`/data/ark/models/Qwen__Qwen3.6-35B-A3B-GGUF/gguf/Qwen3.6-35B-A3B-Q4_K_M.gguf`. Reference by prompt: {'bridge': 'cpu reference', 'code': 'cpu reference', 'rivers': 'cpu reference', 'records': 'cpu reference'}.

| prompt | position | reference margin | gpu prompt pass: KL / max / top-1 | gpu, prompt through the decode step: KL / max / top-1 | gpu prompt pass, FP32 activations: KL / max / top-1 | llama.cpp: KL / max / top-1 |
|---|---|---|---|---|---|---|
| bridge | 0 | 2.737 | 0.00098 / 0.2363 / yes | 0.00009 / 0.0939 / yes | 0.00009 / 0.0939 / yes | 0.00113 / 0.3046 / yes |
| bridge | 4 | 0.171 | 0.00070 / 0.1530 / yes | 0.00032 / 0.1072 / yes | 0.00032 / 0.1072 / yes | 0.00212 / 0.3779 / yes |
| bridge | 12 | 0.998 | 0.00035 / 0.1621 / yes | 0.00021 / 0.0617 / yes | 0.00021 / 0.0617 / yes | 0.00184 / 0.2172 / yes |
| bridge | 24 | 0.751 | 0.00127 / 0.1383 / yes | 0.00078 / 0.1293 / yes | 0.00078 / 0.1293 / yes | 0.00174 / 0.2350 / yes |
| code | 0 | 4.556 | 0.00000 / 0.0129 / yes | 0.00000 / 0.0028 / yes | 0.00000 / 0.0028 / yes | 0.00002 / 0.0668 / yes |
| code | 4 | 5.154 | 0.00000 / 0.0345 / yes | 0.00000 / 0.0242 / yes | 0.00000 / 0.0242 / yes | 0.00004 / 0.2182 / yes |
| code | 12 | 10.059 | 0.00000 / 0.0000 / yes | 0.00000 / 0.0000 / yes | 0.00000 / 0.0000 / yes | -0.00000 / 0.0000 / yes |
| code | 24 | 9.655 | 0.00001 / 0.0000 / yes | 0.00000 / 0.0000 / yes | 0.00000 / 0.0000 / yes | 0.00001 / 0.0001 / yes |
| rivers | 0 | 3.972 | 0.00256 / 0.5227 / yes | 0.00000 / 0.0014 / yes | 0.00000 / 0.0012 / yes | 0.00086 / 0.2810 / yes |
| rivers | 4 | 15.447 | 0.00000 / 0.0000 / yes | 0.00000 / 0.0000 / yes | 0.00000 / 0.0000 / yes | -0.00000 / 0.0000 / yes |
| rivers | 12 | 6.680 | 0.00009 / 0.3817 / yes | 0.00000 / 0.0002 / yes | 0.00000 / 0.0000 / yes | 0.00038 / 0.8787 / yes |
| rivers | 24 | 7.713 | 0.00009 / 0.0002 / yes | 0.00001 / 0.0001 / yes | 0.00001 / 0.0001 / yes | 0.00010 / 0.0003 / yes |
| records | 0 | 3.874 | 0.00566 / 0.6020 / yes | 0.00032 / 0.2299 / yes | 0.00038 / 0.2506 / yes | 0.00127 / 0.3843 / yes |
| records | 1 | 2.177 | 0.00381 / 0.2445 / yes | 0.00006 / 0.0347 / yes | 0.00005 / 0.0336 / yes | 0.00072 / 0.2316 / yes |
| records | 2 | 0.514 | 0.00026 / 0.0292 / yes | 0.00024 / 0.0281 / yes | 0.00024 / 0.0280 / yes | 0.00020 / 0.0265 / yes |
| records | 3 | 7.658 | 0.00002 / 0.0001 / yes | 0.00000 / 0.0000 / yes | 0.00000 / 0.0000 / yes | 0.00001 / 0.0001 / yes |

Mean KL over these positions: gpu prompt pass 0.00099, gpu, prompt through the decode step 0.00013, gpu prompt pass, FP32 activations 0.00013, llama.cpp 0.00065.

## The dense model through the same binary (a record)

`/data/ark/models/Mapika__decider-2b-GGUF/gguf/decider-2b-v12-Q4_K_M.gguf`. Reference by prompt: {'bridge': 'cpu reference', 'code': 'cpu reference', 'rivers': 'cpu reference', 'records': 'llama.cpp on the F32 conversion'}.

| prompt | position | reference margin | gpu prompt pass: KL / max / top-1 | gpu, prompt through the decode step: KL / max / top-1 | gpu prompt pass, FP32 activations: KL / max / top-1 | llama.cpp: KL / max / top-1 | llama.cpp on the F32 conversion: KL / max / top-1 |
|---|---|---|---|---|---|---|---|
| bridge | 0 | 0.974 | 0.00115 / 0.1091 / yes | -0.00000 / 0.0010 / yes | 0.00000 / 0.0009 / yes | 0.00634 / 0.3975 / yes | -0.00059 / 0.0024 / yes |
| bridge | 4 | 0.498 | 0.00049 / 0.1168 / yes | -0.00000 / 0.0005 / yes | -0.00000 / 0.0005 / yes | 0.00445 / 0.2925 / yes | -0.00076 / 0.0025 / yes |
| bridge | 12 | 5.811 | 0.00001 / 0.0119 / yes | 0.00000 / 0.0007 / yes | 0.00000 / 0.0006 / yes | 0.00003 / 0.3473 / yes | -0.00026 / 0.0013 / yes |
| bridge | 24 | 0.997 | 0.00007 / 0.0367 / yes | 0.00000 / 0.0004 / yes | 0.00000 / 0.0005 / yes | 0.00113 / 0.2560 / yes | -0.00025 / 0.0021 / yes |
| code | 0 | 4.141 | 0.00007 / 0.0263 / yes | -0.00000 / 0.0007 / yes | -0.00000 / 0.0007 / yes | -0.00178 / 0.0311 / yes | -0.00196 / 0.0038 / yes |
| code | 4 | 4.461 | -0.00005 / 0.0505 / yes | -0.00000 / 0.0006 / yes | -0.00000 / 0.0006 / yes | -0.00060 / 0.1481 / yes | -0.00117 / 0.0028 / yes |
| code | 12 | 5.028 | 0.00003 / 0.0712 / yes | -0.00000 / 0.0008 / yes | -0.00000 / 0.0010 / yes | -0.00114 / 0.1257 / yes | -0.00126 / 0.0023 / yes |
| code | 24 | 4.749 | -0.00002 / 0.0073 / yes | -0.00000 / 0.0001 / yes | -0.00000 / 0.0001 / yes | -0.00095 / 0.0972 / yes | -0.00104 / 0.0011 / yes |
| rivers | 0 | 0.689 | -0.00292 / 0.1851 / yes | -0.00004 / 0.0014 / yes | -0.00004 / 0.0015 / yes | -0.00446 / 0.3013 / yes | -0.00169 / 0.0031 / yes |
| rivers | 4 | 4.268 | 0.00003 / 0.0222 / yes | 0.00000 / 0.0003 / yes | 0.00000 / 0.0004 / yes | -0.00028 / 0.2179 / yes | -0.00113 / 0.0019 / yes |
| rivers | 12 | 2.292 | 0.00003 / 0.0602 / yes | 0.00000 / 0.0006 / yes | 0.00000 / 0.0006 / yes | 0.00025 / 0.3167 / yes | -0.00127 / 0.0025 / yes |
| rivers | 24 | 5.615 | 0.00000 / 0.0210 / yes | -0.00000 / 0.0006 / yes | -0.00000 / 0.0004 / yes | -0.00042 / 0.4929 / yes | -0.00064 / 0.0027 / yes |
| records | 0 | 2.039 | 0.00387 / 0.1140 / yes | -0.00038 / 0.0553 / yes | -0.00039 / 0.0552 / yes | -0.00987 / 0.2406 / yes |  |
| records | 1 | 0.179 | 0.00449 / 0.1587 / yes | 0.00281 / 0.0411 / yes | 0.00282 / 0.0412 / yes | 0.02328 / 0.4849 / no |  |
| records | 2 | 1.850 | 0.01374 / 0.1491 / yes | 0.00467 / 0.0587 / yes | 0.00469 / 0.0588 / yes | -0.00920 / 0.2386 / yes |  |
| records | 3 | 5.423 | 0.00297 / 0.0727 / yes | 0.00279 / 0.0313 / yes | 0.00279 / 0.0313 / yes | 0.00277 / 0.4647 / yes |  |

Mean KL over these positions: gpu prompt pass 0.00150, gpu, prompt through the decode step 0.00061, gpu prompt pass, FP32 activations 0.00062, llama.cpp 0.00060, llama.cpp on the F32 conversion -0.00100.
Held to the same criteria this model would fail: gpu prompt pass, bridge position 0: compared tokens hold 0.98663 of the reference's mass, 0 above the floor missing; gpu prompt pass, bridge position 4: compared tokens hold 0.97707 of the reference's mass, 0 above the floor missing; gpu prompt pass, bridge position 24: compared tokens hold 0.99896 of the reference's mass, 0 above the floor missing; gpu prompt pass, code position 0: compared tokens hold 0.99620 of the reference's mass, 0 above the floor missing; gpu prompt pass, code position 4: compared tokens hold 0.99662 of the reference's mass, 0 above the floor missing; gpu prompt pass, code position 12: compared tokens hold 0.99826 of the reference's mass, 0 above the floor missing.

## What gives the reference standing

- `gate/moe_block_check.py` (numpy, llama.cpp's gguf-py dequantization, float64; no code shared with the executor) on 126 real block inputs of `rivers` in blocks [0, 19, 39]: the same experts in the same order in 126, the same set in 126; weights within 1.2858101428037116e-07, the shared gate logit within 1.1628863338941642e-07, the block output within 5.053607283755923e-06 (4.893014161759504e-05 of its RMS) (`block-check.json`).
- On the dense model, llama.cpp run on an F32 conversion of the same file (`llama-quantize --allow-requantize ... F32`: dequantized weights, FP32 activations, none of the executor's code) is in the dense table as a side against gufo's CPU reference.

## Cost

CPU reference runs (8 threads): {'moe': {'bridge': {'seconds': 290.5, 'cached': False, 'tokens_evaluated': 45, 'error': None}, 'code': {'seconds': 128.5, 'cached': False, 'tokens_evaluated': 44, 'error': None}, 'rivers': {'seconds': 100.4, 'cached': False, 'tokens_evaluated': 42, 'error': None}, 'records': {'seconds': 4529.3, 'cached': False, 'tokens_evaluated': 1811, 'error': None}}, 'dense': {'bridge': {'seconds': 55.7, 'cached': False, 'tokens_evaluated': 45, 'error': None}, 'code': {'seconds': 56.5, 'cached': False, 'tokens_evaluated': 44, 'error': None}, 'rivers': {'seconds': 51.8, 'cached': False, 'tokens_evaluated': 42, 'error': None}}}. F32 conversion: {'rc': 0, 'seconds': 9.6, 'bytes': 7538261536, 'error': None}.
GPU runs: {'moe|gpu prompt pass': 80.9, 'moe|gpu, prompt through the decode step': 34.0, 'moe|gpu prompt pass, FP32 activations': 20.7, 'dense|gpu prompt pass': 5.6, 'dense|gpu, prompt through the decode step': 13.8, 'dense|gpu prompt pass, FP32 activations': 7.9}; holds [{'for': 'fidelity stage', 'from': '2026-11-11T18:35:02Z', 'to': '2026-11-11T18:37:45Z', 'seconds': 163.1, 'pod_waiters_at_release': 0}].
