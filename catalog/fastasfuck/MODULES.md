# fastasfuck modules

One row for each module of catalog/fastasfuck at 1.1.0 (`73bd1703cbe0`). Card FASTASFUCK-110 17eead1048 (first harvest
ORIGIN-FASTASFUCK 80ae224db4). Written by `docs/harvest/harvest_fastasfuck.py`; read from source, nothing was run or timed.

- **1.1.0**: files of the module added (A) or changed (M) since 35015b5 (`docs/harvest/data/fastasfuck/delta-35015b5.md`); - = unchanged.
- **strixite counterpart**: the catalog/strixite module holding the same files (fastasfuck is a fork of strixite).
- **same op elsewhere**: modules of other origins that provide the same `<family>.<op>`.

| module | lang | files | lines | compile | 1.1.0 | strixite counterpart | tag | same op elsewhere |
|---|---|---:|---:|---|---|---|---|---|
| `kernel.fastasfuck.common` | hip-c++ | 7 | 275 | header-only | - | `kernel.strixite.hip_common and others` | `device.hip-check/-/-`, `device.info/-/-`, `host.check/-/-` +1 | - |
| `format.fastasfuck.qn_device` | c++ | 9 | 575 | header-only | - | `format.strixite.q4 .. q8_device` | `device.buffer/-/-`, `format.device-view/strixw-q4/-`, `format.device-view/strixw-q5/-` +6 | `format.ciru_iu4.c1_g16`, `format.crucible.exl3`, `format.flerp.gguf_placed`, `format.flerp.panel_device`, `format.flerp.q4panel`, +7 more |
| `kernel.fastasfuck.attention_gqa` | hip-c++ | 2 | 1292 | standalone | - | `kernel.strixite.attention` | `attn.softmax-gqa/-/bf16`, `attn.softmax-gqa/-/f32` | `kernel.ciru_inkling.attn_decode`, `kernel.ciru_iu4.attn`, `kernel.ds4.glmtp_rocm_attention`, `kernel.ds4.k53flash_rocm_attention`, `kernel.ds4.k53perf_rocm_attention`, +28 more |
| `kernel.fastasfuck.embedding` | hip-c++ | 2 | 209 | standalone | - | `kernel.strixite.embedding` | `embed.lookup/bf16/bf16`, `embed.lookup/bf16/f32`, `embed.lookup/strixw-q8/bf16` +1 | `kernel.ds4.glmtp_rocm_common`, `kernel.ds4.kmultinode_rocm_common`, `kernel.ds4.rocm_common`, `kernel.gufo.ds4_linear`, `kernel.gufo.qwen_norm_rope`, +4 more |
| `kernel.fastasfuck.gdn` | hip-c++ | 2 | 1106 | standalone | - | `kernel.strixite.gdn` | `mixer.gdn/-/bf16`, `mixer.gdn/-/f32` | `kernel.crucible.gdn`, `kernel.ds4.k53perf_rocm_qwen4`, `kernel.ds4.qwen4_cuda`, `kernel.enginelab.cuda_recur`, `kernel.enginelab.gdn_decode`, +12 more |
| `kernel.fastasfuck.hc` | hip-c++ | 2 | 513 | standalone | - | `kernel.strixite.hc` | `hc.mix/strixw-q4/bf16`, `hc.mix/strixw-q4/f32`, `hc.mix/strixw-q8/bf16` +1 | `kernel.crucible.hc`, `kernel.ds4.rocm_hc`, `kernel.enginelab.core_hc`, `kernel.enginelab.cuda_mix`, `kernel.gufo.ds4_hc`, +10 more |
| `kernel.fastasfuck.hc_wmma` | hip-c++ | 2 | 690 | standalone | - | `kernel.strixite.hc_wmma` | `hc.mix-prefill/strixw-q4/bf16`, `hc.mix-prefill/strixw-q4/f32`, `hc.mix-prefill/strixw-q8/bf16` +1 | `kernel.ciru_iu4.mmb`, `kernel.crucible.hc`, `kernel.ds4.k53perf_rocm_halo_hc`, `kernel.ds4.k53perf_rocm_hc_sgemm`, `kernel.enginelab.hc_prefill`, +7 more |
| `kernel.fastasfuck.kv8` | hip-c++ | 3 | 513 | standalone | - | `kernel.strixite.kv8` | `kvcache.quant8/-/bf16`, `kvcache.quant8/-/f32` | `kernel.kyojin.kvcache_quant`, `kernel.kyojin.triton_qdeq`, `kernel.kyojin.triton_qwrite`, `kernel.strata.kvcache` |
| `kernel.fastasfuck.linear` | hip-c++ | 12 | 2294 | standalone | - | `kernel.strixite.linear, kernel.strixite.linear_q4, kernel.strixite.linear_q56, kernel.strixite.linear_q8, kernel.strixite.linear_wmma` | `linear.decode/strixw-q4/bf16`, `linear.decode/strixw-q4/f32`, `linear.decode/strixw-q5/bf16` +21 | `kernel.ciru_inkling.glue_decode`, `kernel.ciru_inkling.moe_decode`, `kernel.ciru_inkling.quant_decode`, `kernel.ciru_iu4.c1`, `kernel.ciru_iu4.q5`, +167 more |
| `kernel.fastasfuck.logits_bias` | hip-c++ | 2 | 70 | standalone | - | `kernel.strixite.logits_bias` | `logits.bias/-/-` | - |
| `kernel.fastasfuck.logits_topk` | hip-c++ | 2 | 232 | standalone | - | `kernel.strixite.logits_topk` | `logits.topk/-/-` | `kernel.flerp.logits_topk_gufo`, `kernel.gufo.qwen_sampling`, `kernel.hipfire.sampling`, `kernel.lucebox.sampling`, `kernel.radiance.halo_sample` |
| `kernel.fastasfuck.moe_grouped` | hip-c++ | 2 | 1534 | standalone | - | `kernel.strixite.moe_grouped` | `moe.experts-prefill/strixw-q4/bf16`, `moe.experts-prefill/strixw-q4/f32`, `moe.experts-prefill/strixw-q5/bf16` +1 | `kernel.ciru_inkling.moe_prefill`, `kernel.ciru_iu4.c1`, `kernel.ciru_iu4.mmb`, `kernel.crucible.moe_prefill`, `kernel.ds4.k53perf_rocm_halo_moe`, +29 more |
| `kernel.fastasfuck.moe_router` | hip-c++ | 2 | 357 | standalone | - | `kernel.strixite.moe_router` | `moe.router/-/bf16`, `moe.router/-/f32` | `kernel.ciru_inkling.glue_decode`, `kernel.crucible.ops`, `kernel.ds4.k53flash_rocm_router`, `kernel.ds4.k53perf_rocm_router`, `kernel.ds4.k53prefill_rocm_router`, +23 more |
| `kernel.fastasfuck.mtp_pick` | hip-c++ | 2 | 125 | standalone | - | `kernel.strixite.mtp_pick` | `draft.mtp-pick/-/-` | - |
| `kernel.fastasfuck.norm` | hip-c++ | 2 | 295 | standalone | - | `kernel.strixite.norm` | `norm.rmsnorm-gated/-/bf16`, `norm.rmsnorm-gated/-/f32`, `norm.rmsnorm-zc/-/bf16` +1 | `kernel.kyojin.fused_elt`, `kernel.kyojin.kda_fused`, `kernel.libstrix.norm`, `kernel.radiance.halo_gdn` |
| `kernel.fastasfuck.ple` | hip-c++ | 2 | 212 | standalone | - | `kernel.strixite.ple` | `ple.inject/-/bf16`, `ple.inject/-/f32` | `kernel.crucible.ops`, `kernel.enginelab.ple`, `kernel.gufo.qfn_kernels`, `kernel.kyojin.ple`, `kernel.kyojin.ple_hip`, +3 more |
| `kernel.fastasfuck.qsa_indexer` | hip-c++ | 2 | 1198 | standalone | - | `kernel.strixite.qsa` | `attn.qsa-indexer/-/bf16`, `attn.qsa-indexer/-/f32` | `kernel.ciru_iu4.attn`, `kernel.crucible.qsa`, `kernel.enginelab.indexed_attention`, `kernel.enginelab.qsa_prefill`, `kernel.gufo.qfn_kernels`, +10 more |
| `kernel.fastasfuck.residual` | hip-c++ | 2 | 72 | standalone | - | `kernel.strixite.residual` | `residual.add/-/bf16`, `residual.add/-/f32` | `kernel.flerp.residual_add_gufo`, `kernel.flerp.residual_add_hipfire`, `kernel.gufo.qwen_norm_rope`, `kernel.hipfire.tensor_ops`, `kernel.kyojin.elementwise` |
| `kernel.fastasfuck.rope` | hip-c++ | 3 | 234 | standalone | - | `kernel.strixite.rope, kernel.strixite.rope_math` | `pos.rope-math/-/-`, `pos.rope/-/bf16`, `pos.rope/-/f32` | `kernel.crucible.ops`, `kernel.enginelab.ops_rope`, `kernel.enginelab.qwen_rotary`, `kernel.gufo.qwen_norm_rope`, `kernel.hipfire.norm_rope`, +5 more |
| `kernel.fastasfuck.swiglu` | hip-c++ | 2 | 83 | standalone | - | `kernel.strixite.swiglu` | `act.swiglu/-/bf16`, `act.swiglu/-/f32` | `kernel.ciru_inkling.glue_decode`, `kernel.ds4.kmultinode_rocm_output`, `kernel.ds4.rocm_output`, `kernel.enginelab.ops_glu`, `kernel.gufo.qfn_kernels`, +5 more |
| `kernel.fastasfuck.vision` | hip-c++ | 2 | 734 | standalone | - | `kernel.strixite.vision` | `vision.tower-ops/bf16/bf16`, `vision.tower-ops/bf16/f32` | `kernel.crucible.vision`, `kernel.ds4.glm53_vision`, `kernel.ds4.k53flash_glm53_vision`, `kernel.ds4.k53perf_glm53_vision`, `kernel.ds4.k53prefill_glm53_vision`, +1 more |
| `kernel.fastasfuck.wmma` | hip-c++ | 3 | 684 | standalone | - | `kernel.strixite.wmma, kernel.strixite.wmma_gemm` | `gemm.wmma-blocks/-/bf16`, `gemm.wmma-tile/-/bf16` | `kernel.crucible.device_common`, `kernel.enginelab.core_tiled_wmma`, `kernel.enginelab.h3_gemm` |
| `runtime.fastasfuck.args` | c++ | 1 | 34 | header-only | - | `runtime.strixite.args` | `host.args/-/-` | - |
| `runtime.fastasfuck.bf16` | c++ | 1 | 28 | header-only | - | `runtime.strixite.bf16` | `host.bf16/-/-` | - |
| `runtime.fastasfuck.log` | c++ | 1 | 94 | header-only | - | `runtime.strixite.log` | `host.log/-/-` | - |
| `runtime.fastasfuck.trace` | c++ | 1 | 59 | header-only | - | `runtime.strixite.trace` | `host.trace/-/-` | - |
| `format.fastasfuck.q4` | c++ | 1 | 136 | n/a | - | `format.strixite.q4` | `format.host/strixw-q4/-` | `format.flerp.q4panel`, `format.flerp.q8panel` |
| `format.fastasfuck.q5` | c++ | 1 | 117 | n/a | - | `format.strixite.q5` | `format.host/strixw-q5/-` | `format.flerp.q4panel`, `format.flerp.q8panel` |
| `format.fastasfuck.q6` | c++ | 1 | 117 | n/a | - | `format.strixite.q6` | `format.host/strixw-q6/-` | `format.flerp.q4panel`, `format.flerp.q8panel` |
| `format.fastasfuck.q8` | c++ | 1 | 129 | n/a | - | `format.strixite.q8` | `format.host/strixw-q8/-` | `format.flerp.q4panel`, `format.flerp.q8panel` |
| `format.fastasfuck.q4_from_q8` | c++ | 2 | 185 | n/a | - | `format.strixite.q4_from_q8` | `format.requant/strixw-q4/-` | - |
| `format.fastasfuck.safetensors` | c++ | 2 | 526 | n/a | - | `format.strixite.safetensors` | `format.reader/safetensors/-` | - |
| `format.fastasfuck.merge_plan` | c++ | 2 | 189 | n/a | - | `format.strixite.merge_plan` | `format.merge-plan/strixw/-` | - |
| `format.fastasfuck.ngram_table` | c++ | 2 | 393 | n/a | - | `format.strixite.ngram_table` | `format.container/ngram-table/-` | `format.flerp.strixw_v5`, `format.flerp.strixw_v5p` |
| `format.fastasfuck.faf` | c++ | 2 | 700 | n/a | - | none | `format.container/faf/-` | `format.flerp.strixw_v5`, `format.flerp.strixw_v5p` |
| `format.fastasfuck.convert_qwen4exp` | c++ | 2 | 1530 | n/a | 0A 2M | `format.strixite.convert_qwen4exp` | `format.convert/faf/-`, `format.prebuilt-import/faf/-` | `format.flerp.convert_qwen4exp_p16` |
| `format.fastasfuck.safetensors_loader` | c++ | 2 | 226 | n/a | - | `format.strixite.safetensors_loader` | `format.device-loader/safetensors/-` | `format.flerp.strixw_loader_p` |
| `format.fastasfuck.faf_loader` | c++ | 2 | 385 | n/a | - | `format.strixite.strixw_loader` | `format.device-loader/faf/-` | `format.flerp.strixw_loader_p` |
| `runtime.fastasfuck.ngram_rows` | c++ | 2 | 169 | n/a | - | `runtime.strixite.ngram_rows` | `ple.ngram-rows/safetensors/-` | `format.crucible.ngram16` |
| `runtime.fastasfuck.ngram_table` | c++ | 2 | 502 | n/a | - | `runtime.strixite.ngram_table` | `ple.ngram-rows/ngram-table/-` | `format.crucible.ngram16` |
| `runtime.fastasfuck.paged_buffer` | c++ | 1 | 537 | header-only | - | `runtime.strixite.paged_buffer` | `device.paged-buffer/-/-` | - |
| `runtime.fastasfuck.paged_state` | c++ | 1 | 51 | header-only | - | `runtime.strixite.paged_state` | `kvcache.parked-state/-/-` | - |
| `runtime.fastasfuck.ple_hash` | c++ | 2 | 78 | n/a | - | `runtime.strixite.ple_hash` | `ple.ngram-ids/-/-` | - |
| `runtime.fastasfuck.qweight` | c++ | 1 | 175 | header-only | - | `runtime.strixite.qweight` | `linear.dispatch/strixw/-` | - |
| `executor.fastasfuck.qwen3_5` | c++ | 2 | 436 | n/a | - | `executor.strixite.qwen3_5` | `executor.forward/qwen3.5/bf16` | - |
| `executor.fastasfuck.qwen4exp` | c++ | 2 | 5284 | n/a | - | `executor.strixite.qwen4exp` | `executor.forward/qwen4exp/bf16` | - |
| `executor.fastasfuck.vision_tower` | c++ | 2 | 418 | n/a | - | `executor.strixite.vision_tower` | `executor.vision-tower/qwen4exp/bf16` | - |
| `executor.fastasfuck.qwen4exp_backend` | c++ | 2 | 621 | n/a | 0A 1M | `executor.strixite.qwen4exp_backend` | `executor.backend/qwen4exp/-` | - |
| `runtime.fastasfuck.capture` | c++ | 2 | 148 | n/a | - | `runtime.strixite.capture` | `serve.capture/-/-` | - |
| `runtime.fastasfuck.chat_template` | c++ | 2 | 360 | n/a | - | `runtime.strixite.chat_template` | `text.chat-template/-/-` | - |
| `runtime.fastasfuck.engine` | c++ | 4 | 3521 | n/a | - | `runtime.strixite.engine` | `serve.engine/-/-` | - |
| `runtime.fastasfuck.grammar` | c++ | 2 | 618 | n/a | - | `runtime.strixite.grammar` | `grammar.matcher/-/-` | - |
| `runtime.fastasfuck.grammar_cache` | c++ | 2 | 110 | n/a | - | `runtime.strixite.grammar_cache` | `grammar.cache/-/-` | - |
| `runtime.fastasfuck.host_buffer` | c++ | 2 | 116 | n/a | - | `runtime.strixite.host_buffer` | `host.buffer/-/-` | - |
| `runtime.fastasfuck.json` | c++ | 2 | 561 | n/a | - | `runtime.strixite.json` | `host.json/-/-` | - |
| `runtime.fastasfuck.json_schema` | c++ | 2 | 633 | n/a | - | `runtime.strixite.json_schema` | `grammar.json-schema/-/-` | - |
| `runtime.fastasfuck.serve_log` | c++ | 1 | 103 | header-only | - | `runtime.strixite.serve_log` | `host.log-rows/-/-` | - |
| `runtime.fastasfuck.memory_governor` | c++ | 1 | 93 | header-only | - | `runtime.strixite.memory_governor` | `host.memory-governor/-/-` | - |
| `runtime.fastasfuck.memory_plan` | c++ | 2 | 254 | n/a | - | `runtime.strixite.memory_plan` | `serve.memory-plan/-/-` | - |
| `runtime.fastasfuck.mtp_calib` | c++ | 2 | 481 | n/a | - | none | `draft.mtp-calibration/-/-` | - |
| `runtime.fastasfuck.output_parser` | c++ | 2 | 654 | n/a | - | `runtime.strixite.output_parser` | `text.output-parser/-/-` | - |
| `runtime.fastasfuck.prompt_cache` | c++ | 2 | 2128 | n/a | - | `runtime.strixite.prompt_cache` | `cache.prompt/-/-` | - |
| `runtime.fastasfuck.resume_loss` | c++ | 2 | 76 | n/a | - | `runtime.strixite.resume_loss` | `serve.resume-loss/-/-` | - |
| `runtime.fastasfuck.response_store` | c++ | 2 | 122 | n/a | - | none | `serve.response-store/-/-` | - |
| `runtime.fastasfuck.sampler` | c++ | 2 | 505 | n/a | - | `runtime.strixite.sampler` | `sample.next-token/-/-` | `kernel.enginelab.token_sampling`, `kernel.kyojin.softmax_sampling`, `kernel.maya_tp.sampler`, `kernel.radiance.halo_sample`, `kernel.strata.sampling`, +7 more |
| `runtime.fastasfuck.structured_output` | c++ | 2 | 139 | n/a | - | `runtime.strixite.structured_output` | `grammar.structured-output/-/-` | - |
| `runtime.fastasfuck.think_nudge` | c++ | 2 | 151 | n/a | - | `runtime.strixite.think_nudge` | `serve.think-nudge/-/-` | - |
| `runtime.fastasfuck.token_mask` | c++ | 2 | 126 | n/a | - | `runtime.strixite.token_mask` | `grammar.token-mask/-/-` | - |
| `runtime.fastasfuck.tokenizer` | c++ | 2 | 396 | n/a | - | `runtime.strixite.tokenizer` | `text.tokenizer/-/-` | - |
| `runtime.fastasfuck.unicode` | c++ | 3 | 5080 | n/a | - | `runtime.strixite.unicode` | `text.unicode/-/-` | - |
| `runtime.fastasfuck.vision_preprocess` | c++ | 2 | 654 | n/a | - | `runtime.strixite.vision_preprocess` | `vision.preprocess/-/-` | - |
| `runtime.fastasfuck.vision_request` | c++ | 2 | 500 | n/a | - | `runtime.strixite.vision_request` | `vision.request/-/-` | - |
| `runtime.fastasfuck.vocab_filter` | c++ | 2 | 295 | n/a | - | none | `text.vocab-filter/-/-` | - |
| `serve.fastasfuck.http` | c++ | 4 | 2329 | n/a | - | `serve.strixite.http` | `serve.http/openai/-` | `serve.flerp.http` |
| `serve.fastasfuck.openai` | c++ | 2 | 763 | n/a | - | `serve.strixite.openai` | `serve.api/openai/-` | - |
| `serve.fastasfuck.responses_api` | c++ | 2 | 691 | n/a | - | none | `serve.api/openai-responses/-` | - |
| `serve.fastasfuck.fastasfuck_server` | c++ | 1 | 749 | n/a | - | `serve.strixite.strix_server` | `serve.program/qwen4exp/-` | `serve.flerp.strixfold_host` |
| `tool.fastasfuck.check_stream_calls` | c++ | 1 | 389 | n/a | - | `tool.strixite.check_stream_calls` | `tool.check-stream-calls/-/-` | - |
| `tool.fastasfuck.convert_ngram_table` | c++ | 1 | 172 | n/a | - | `tool.strixite.convert_ngram_table` | `tool.convert-ngram-table/-/-` | `format.crucible.ngram16` |
| `tool.fastasfuck.convert_qwen4exp` | c++ | 1 | 113 | n/a | 0A 1M | `tool.strixite.convert_qwen4exp` | `tool.convert-qwen4exp/-/-` | - |
| `tool.fastasfuck.transcode_ngram_table` | c++ | 1 | 173 | n/a | - | `tool.strixite.transcode_ngram_table` | `tool.transcode-ngram-table/-/-` | - |
| `tool.fastasfuck.inspect_faf` | c++ | 1 | 62 | n/a | - | none | `tool.inspect-faf/-/-` | - |
| `tool.fastasfuck.eval_ppl_kld` | c++ | 1 | 456 | n/a | - | none | `tool.eval-ppl-kld/-/-` | - |
| `tool.fastasfuck.measure_verify_curve` | c++ | 1 | 308 | n/a | - | none | `tool.measure-verify-curve/-/-` | - |
| `tool.fastasfuck.test_linear_q6` | c++ | 1 | 258 | n/a | - | none | `tool.test-linear-q6/-/-` | - |
| `tool.fastasfuck.test_ngram_cache` | c++ | 1 | 123 | n/a | - | none | `tool.test-ngram-cache/-/-` | - |
| `tool.fastasfuck.test_split_invariance` | c++ | 1 | 249 | n/a | - | none | `tool.test-split-invariance/-/-` | - |
| `tool.fastasfuck.test_vision_batchable_edge_cases` | c++ | 1 | 337 | n/a | - | none | `tool.test-vision-batchable-edge-cases/-/-` | - |
| `tool.fastasfuck.breakdown_split` | c++ | 1 | 683 | n/a | - | none | `tool.breakdown-split/-/-` | - |
| `tool.fastasfuck.bench_threshold_gemv_vs_wmma` | c++ | 1 | 382 | n/a | - | none | `tool.bench-threshold-gemv-vs-wmma/-/-` | - |
| `tool.fastasfuck.audit_kernels` | python | 1 | 119 | n/a | - | none | `tool.audit-kernels/-/-` | - |
| `tool.fastasfuck.bench_adaptive_margin` | python | 1 | 155 | n/a | - | none | `tool.bench-adaptive-margin/-/-` | - |
| `tool.fastasfuck.profile_run` | python | 1 | 207 | n/a | - | none | `tool.profile-run/-/-` | - |
| `tool.fastasfuck.quantize_qwen_gptq` | python | 1 | 846 | n/a | 1A 0M | none | `tool.quantize-qwen-gptq/-/-` | - |
| `tool.fastasfuck.sim_qwen_quant` | python | 1 | 1608 | n/a | 1A 0M | none | `tool.sim-qwen-quant/-/-` | - |
| `tool.fastasfuck.sim_qwen_calib` | python | 1 | 892 | n/a | 1A 0M | none | `tool.sim-qwen-calib/-/-` | - |
| `tool.fastasfuck.sim_qwen_awq_gptq` | python | 1 | 1524 | n/a | 1A 0M | none | `tool.sim-qwen-awq-gptq/-/-` | - |
| `tool.fastasfuck.sim_qwen_gptq_r3` | python | 1 | 957 | n/a | 1A 0M | none | `tool.sim-qwen-gptq-r3/-/-` | - |
| `tool.fastasfuck.sim_qwen_wave2` | python | 1 | 1690 | n/a | 1A 0M | none | `tool.sim-qwen-wave2/-/-` | - |
| `tool.fastasfuck.sim_qwen_fp64_mtp` | python | 1 | 1291 | n/a | 1A 0M | none | `tool.sim-qwen-fp64-mtp/-/-` | - |
| `tool.fastasfuck.solve_qwen_quant` | python | 1 | 947 | n/a | 1A 0M | none | `tool.solve-qwen-quant/-/-` | - |
| `tool.fastasfuck.bench_common` | python | 1 | 288 | n/a | - | none | `tool.bench-common/-/-` | - |
| `tool.fastasfuck.bench_ab_quant` | python | 1 | 562 | n/a | 1A 0M | none | `tool.bench-ab-quant/-/-` | - |
| `tool.fastasfuck.bench_chunk_pareto` | python | 1 | 258 | n/a | - | none | `tool.bench-chunk-pareto/-/-` | - |
| `tool.fastasfuck.bench_mixed_concurrency` | python | 1 | 349 | n/a | - | none | `tool.bench-mixed-concurrency/-/-` | - |
| `tool.fastasfuck.bench_multislot_scaling` | python | 1 | 165 | n/a | - | none | `tool.bench-multislot-scaling/-/-` | - |
| `tool.fastasfuck.bench_prefill_scaling` | python | 1 | 186 | n/a | - | none | `tool.bench-prefill-scaling/-/-` | - |
| `tool.fastasfuck.bench_solo_sterile` | python | 1 | 140 | n/a | - | none | `tool.bench-solo-sterile/-/-` | - |
| `tool.fastasfuck.run_mixed_triad` | python | 1 | 135 | n/a | - | none | `tool.run-mixed-triad/-/-` | - |

109 modules: 4 executor, 13 format, 21 kernel, 35 runtime, 4 serve, 32 tool.
