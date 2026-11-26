# Maya-TP modules

Every module of catalog/maya_tp (uv-genai/maya-tp 6951bd7, tp-v1.1.45). Card ORIGIN-MAYA-TP 2982083a71. Written by
`docs/harvest/harvest_maya_tp.py` from its `FAMILIES` table. **TP2** marks the modules of the two-host tensor-parallel
delta over Project Maya; the rest is the engine Maya-TP carries (Project Maya v1.1.58 with the strata:: names).

| module | TP2 | files | lines | interface | compile |
|---|---|---:|---:|---|---|
| `kernel.maya_tp.common` |  | 12 | 2533 | `kernel-abi.maya-tp-hip-compat/-/-` | header-only |
| `kernel.maya_tp.s_gemv` |  | 10 | 1340 | `linear.decode/strata-s/f16+q8_k+q8_0-f32.maya-tp` | standalone |
| `kernel.maya_tp.gguf_gemv` |  | 5 | 3362 | `linear.decode/gguf-kquant+iq/q8_1-f32.maya-tp-mmvq` | standalone |
| `kernel.maya_tp.quantize_act` |  | 2 | 371 | `quant.activation/-/q8_0+q8_k.maya-tp` | standalone |
| `kernel.maya_tp.bf16_gemv` |  | 5 | 765 | `linear.dense/bf16/bf16-f32.maya-tp` | standalone |
| `kernel.maya_tp.router` |  | 6 | 517 | `moe.router/-/f32.maya-tp-sigmoid`, `moe.router/-/f32.maya-tp-top10` | standalone |
| `kernel.maya_tp.moe_experts` |  | 6 | 1458 | `moe.combine/-/f32.maya-tp`, `moe.experts-decode/strata-s2+bf16/q8_0-f32.maya-tp-grouped` | standalone |
| `kernel.maya_tp.gdn` |  | 8 | 926 | `mixer.gdn/-/f32.maya-tp` | standalone |
| `kernel.maya_tp.hyperconn` |  | 8 | 1261 | `hc.mix/bf16/f32.maya-tp` | standalone |
| `kernel.maya_tp.qsa` |  | 14 | 3520 | `attn.qsa-indexer/-/f32.maya-tp`, `attn.softmax-gqa/kv-f16+i8+q4_0/f32.maya-tp-qsa` | standalone |
| `kernel.maya_tp.kvcache` |  | 6 | 816 | `kvcache.store/q4_0-wht+i8/f32.maya-tp` | standalone |
| `kernel.maya_tp.rope` |  | 4 | 256 | `pos.rope/-/f32.maya-tp` | standalone |
| `kernel.maya_tp.ple` |  | 4 | 745 | `ple.inject/-/f32.maya-tp` | standalone |
| `kernel.maya_tp.elementwise` |  | 4 | 677 | `tensor.ops/-/f32.maya-tp-glue` | standalone |
| `kernel.maya_tp.sampler` |  | 2 | 677 | `sample.next-token/-/f32.maya-tp` | standalone |
| `kernel.maya_tp.verify` |  | 2 | 664 | `speculation.window-verify/-/f32.maya-tp` | standalone |
| `kernel.maya_tp.flash_attn` |  | 2 | 248 | `attn.softmax-gqa/kv-f16/f32.maya-tp-native-fa` | standalone |
| `kernel.maya_tp.exl3` |  | 3 | 4415 | `linear.decode/exl3/f32-f32.maya-tp` | standalone |
| `kernel.maya_tp.glm_fast` |  | 5 | 8784 | `moe.experts-decode/gguf-iq2+iq3/f32.maya-tp` | standalone |
| `kernel.maya_tp.glm_batch` |  | 2 | 2914 | `moe.experts-decode/gguf-iq2+iq3/f32.maya-tp-rows` | standalone |
| `kernel.maya_tp.glm_dsa` |  | 2 | 278 | `attn.sparse-dsa-mla/-/f32.maya-tp` | standalone |
| `kernel.maya_tp.glm_kda` |  | 2 | 285 | `mixer.kda/-/f32.maya-tp` | standalone |
| `kernel.maya_tp.glm_hc` |  | 2 | 294 | `hc.mix/bf16/f32.maya-tp-mhc` | standalone |
| `kernel.maya_tp.glm_kv_stream` |  | 3 | 363 | `kvcache.store/mla-latent-i8/f32.maya-tp` | standalone |
| `kernel.maya_tp.glm_ffn` |  | 2 | 59 | `act.swiglu-clamped/-/f32.maya-tp` | standalone |
| `kernel.maya_tp.glmd_decode` |  | 11 | 1816 | `attn.sparse-dsa-mla/-/f32.maya-tp-glmd` | standalone |
| `kernel.maya_tp.glmd_mixer` |  | 2 | 247 | `attn.dsa-mixer/-/f32.maya-tp` | standalone |
| `kernel.maya_tp.glmd_batch` |  | 2 | 2257 | `attn.prefill-causal/-/f32.maya-tp-glmd` | standalone |
| `kernel.maya_tp.glmd_volta` |  | 2 | 1045 | `attn.indexer-score/-/f32.maya-tp-volta` | standalone |
| `kernel.maya_tp.glmd_ref` |  | 3 | 749 | `verify.kernel-reference-maya-tp/-/-` | standalone |
| `kernel.maya_tp.tp_kernels` | TP2 | 2 | 183 | `collective.device-stage/-/f32.maya-tp` | standalone |
| `kernel.maya_tp.peer_probe` |  | 1 | 116 | `verify.peer-access/-/-` | standalone |
| `kernel.maya_tp.prefill` |  | 5 | 1503 | `linear.prefill/bf16+f16/f32.maya-tp-blaslt` | standalone |
| `kernel.maya_tp.prefill_moe_rdna4` |  | 2 | 865 | `moe.experts-prefill/gguf-iq+kquant/i8-f32.maya-tp-fused` | n/a |
| `kernel.maya_tp.prefill_moe_mmq` |  | 6 | 1933 | `moe.experts-prefill/gguf-kquant/q8_1-f32.maya-tp-mmq` | n/a |
| `kernel.maya_tp.ds4_kernels` |  | 5 | 5945 | `attn.sparse-sink/-/f32.maya-tp-ds4` | standalone |
| `kernel.maya_tp.ds4_iq` |  | 5 | 3127 | `moe.experts-decode/gguf-iq+mxfp4/f32.maya-tp-ds4` | standalone |
| `kernel.maya_tp.ds4_gguf` |  | 2 | 2022 | `linear.decode/bf16+mxfp4/f32.maya-tp-ds4` | standalone |
| `kernel.maya_tp.ds4_lt` |  | 2 | 261 | `linear.prefill/bf16/f32.maya-tp-ds4-lt` | standalone |
| `kernel.maya_tp.ds4_vision` |  | 2 | 669 | `vision.encoder/-/f32.maya-tp-ds4` | standalone |
| `kernel.maya_tp.ds4_pcie` |  | 2 | 342 | `moe.expert-stream-pcie/-/-` | header-only |
| `kernel.maya_tp.mm3_kernels` |  | 2 | 2770 | `attn.softmax-gqa/kv-f16/f32.maya-tp-mm3` | standalone |
| `kernel.maya_tp.mm3_exl3` |  | 2 | 3257 | `linear.decode/exl3/f32-f32.maya-tp-mm3` | standalone |
| `kernel.maya_tp.mm3_gguf` |  | 3 | 820 | `moe.experts-decode/gguf-kquant/f32.maya-tp-mm3` | standalone |
| `kernel.maya_tp.mm3_gv` |  | 2 | 528 | `linear.decode/gguf-kquant+q8_0/f32.maya-tp-mm3-gv` | standalone |
| `kernel.maya_tp.mm3_vision` |  | 2 | 733 | `vision.encoder/-/f32.maya-tp-mm3` | standalone |
| `kernel.maya_tp.cpu_iq` |  | 11 | 2792 | `moe.experts-decode-cpu/gguf-iq+kquant/q8_k-f32.maya-tp` | n/a |
| `kernel.maya_tp.cpu_exl3` |  | 6 | 2016 | `moe.experts-decode-cpu/exl3/f32.maya-tp` | n/a |
| `kernel.maya_tp.cpu_pool` |  | 7 | 1856 | `sched.cpu-expert-pool/-/-` | n/a |
| `kernel.maya_tp.cpu_glm_lane` |  | 4 | 962 | `moe.experts-decode-cpu/gguf-iq/f32.maya-tp-row4` | n/a |
| `runtime.maya_tp.tp_rdma` | TP2 | 5 | 1811 | `transport.verbs-maya-tp/-/-`, `collective.exchange-partial/-/f32.maya-tp` | standalone |
| `runtime.maya_tp.tp_comm_abi` | TP2 | 2 | 159 | `collective.interface-maya-tp/-/-` | header-only |
| `runtime.maya_tp.tp_context` | TP2 | 4 | 310 | `collective.allreduce-tp2/-/f32.maya-tp`, `partition.intra-expert-tp2/-/-` | standalone |
| `runtime.maya_tp.glm_tp_batch` | TP2 | 1 | 260 | `sched.tp2-batch-frames-glm/-/-` | header-only |
| `runtime.maya_tp.ds4_tp_control` | TP2 | 7 | 482 | `sched.tp2-control-ds4/-/-` | header-only |
| `runtime.maya_tp.ds4_tp_batch` | TP2 | 2 | 262 | `sched.tp2-batch-frames-ds4/-/-` | header-only |
| `runtime.maya_tp.ds4_tp_worker` | TP2 | 3 | 299 | `sched.tp2-worker-ds4/-/-` | header-only |
| `runtime.maya_tp.ds4_tp_loader` | TP2 | 6 | 422 | `format.loader-tp2-slices/-/-` | header-only |
| `runtime.maya_tp.mm3_tp` | TP2 | 2 | 2943 | `collective.host-sum-tp/-/f32.maya-tp-mm3` | n/a |
| `runtime.maya_tp.glm_model` |  | 11 | 7890 | `executor.glm5-next/-/-` | n/a |
| `runtime.maya_tp.glm_fast_path` |  | 6 | 10277 | `executor.glm5-next-fast-decode/-/-` | n/a |
| `runtime.maya_tp.glm_prefill` |  | 2 | 5397 | `executor.glm5-next-prefill/-/-` | n/a |
| `runtime.maya_tp.glm_dsa_decode` |  | 14 | 3461 | `executor.glm-dsa-decode/-/-` | n/a |
| `runtime.maya_tp.glm_dsa_prompt` |  | 7 | 1946 | `executor.glm-dsa-prefill/-/-` | n/a |
| `runtime.maya_tp.glm_dsa_reference` |  | 6 | 1807 | `verify.reference-forward-glm-dsa/-/-` | n/a |
| `runtime.maya_tp.glm_mtp` |  | 7 | 4978 | `speculation.draft-mtp/-/-` | n/a |
| `runtime.maya_tp.glm_dflash` |  | 3 | 2293 | `speculation.draft-dflash/-/-` | n/a |
| `runtime.maya_tp.glm_ngram` |  | 4 | 1019 | `speculation.draft-ngram/-/-` | n/a |
| `runtime.maya_tp.glm_tiers` |  | 13 | 2819 | `sched.expert-tiers/-/-` | n/a |
| `runtime.maya_tp.glm_prompt_cache` |  | 4 | 772 | `cache.prompt-maya-tp/-/-` | n/a |
| `runtime.maya_tp.glm_tools` |  | 6 | 715 | `steer.ablate-direction/-/f32.maya-tp` | n/a |
| `runtime.maya_tp.core_session` |  | 18 | 5596 | `executor.strata-core/-/-` | n/a |
| `runtime.maya_tp.expert_store` |  | 7 | 4324 | `sched.expert-cache/-/-` | n/a |
| `runtime.maya_tp.verify_mtp` |  | 4 | 2371 | `speculation.window-verify-driver/-/-` | n/a |
| `runtime.maya_tp.spec_controller` |  | 8 | 1242 | `speculation.policy/-/-` | n/a |
| `runtime.maya_tp.prefill_host` |  | 3 | 2193 | `executor.prefill-driver/-/-` | n/a |
| `runtime.maya_tp.generate` |  | 3 | 10409 | `serve.program-maya-tp/-/-` | n/a |
| `runtime.maya_tp.ds4_engine` |  | 13 | 23876 | `executor.dsv41-maya-tp/-/-` | n/a |
| `runtime.maya_tp.ds4_main` |  | 1 | 4122 | `serve.program-ds4-maya-tp/-/-` | n/a |
| `runtime.maya_tp.ds4_store` |  | 8 | 3058 | `cache.prompt-engram-maya-tp/-/-` | n/a |
| `runtime.maya_tp.ds4_cpu` |  | 6 | 2015 | `moe.experts-decode-cpu/gguf-iq+exl3/f32.maya-tp-ds4` | n/a |
| `runtime.maya_tp.mm3_engine` |  | 18 | 7743 | `executor.minimax-m31-maya-tp/-/-` | n/a |
| `runtime.maya_tp.mm3_dspark` |  | 2 | 602 | `speculation.draft-dspark/-/-` | n/a |
| `runtime.maya_tp.platform` |  | 15 | 1345 | `platform.memory-io/-/-` | n/a |
| `format.maya_tp.gguf_reader` |  | 4 | 1008 | `format.reader-maya-tp/-/-` | n/a |
| `format.maya_tp.ple_reader` |  | 2 | 540 | `format.reader-ple/-/-` | n/a |
| `runtime.maya_tp.core_headers` |  | 21 | 1584 | `kernel-abi.maya-tp-engine-headers/-/-` | header-only |
| `serve.maya_tp.server` |  | 11 | 13586 | `serve.http-maya-tp/-/-` | n/a |
| `serve.maya_tp.responses` |  | 5 | 1262 | `serve.responses-api/-/-` | n/a |
| `serve.maya_tp.templates` |  | 7 | 1984 | `serve.chat-template-glm5/-/-` | n/a |
| `serve.maya_tp.tools_mcp` |  | 4 | 2268 | `serve.tools-mcp/-/-` | n/a |
| `serve.maya_tp.vision_front` |  | 2 | 361 | `serve.vision-preprocess/-/-` | n/a |
| `tool.maya_tp.tp2_launch` | TP2 | 14 | 803 | `tool.tp2-launch/-/-` | n/a |
| `tool.maya_tp.tp2_validation` | TP2 | 11 | 1289 | `verify.tp2-parity-suite/-/-` | n/a |
| `tool.maya_tp.tp_tests` | TP2 | 58 | 6488 | `verify.tp2-unit-tests/-/-` | n/a |
| `tool.maya_tp.kernel_parity` |  | 37 | 12914 | `verify.kernel-parity-maya-tp/-/-` | n/a |
| `tool.maya_tp.hip_tests` |  | 16 | 3728 | `verify.hip-kernel-tests-maya-tp/-/-` | n/a |
| `tool.maya_tp.hipblaslt_tuning` |  | 16 | 1239 | `tool.hipblaslt-tables/-/-` | n/a |
| `tool.maya_tp.engine_probes` |  | 4 | 570 | `tool.engine-probes/-/-` | n/a |

99 modules, 647 files, 245184 lines; 13 TP2 modules.

## Not done here

No kernel or engine was run or timed on the pod, and no RDMA link was opened (the pod nodes have no Thunderbolt
link between them). The compile gate (`gate/compile/run.py --origin maya_tp`, control-node CPU, gfx1151) compiles the
standalone modules; the host engine modules are harvested only (their CMake targets link dozens of units together).
