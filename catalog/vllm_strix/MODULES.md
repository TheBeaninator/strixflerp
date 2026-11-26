# vllm-strix-halo modules by strixflerp interface

One row for each module of vllm-strix-halo at fe5ac10. Card ORIGIN-VLLM-STRIX d2a7b91dec. Written by
`docs/harvest/harvest_vllm_strix.py` from its `FAMILIES` table.

- **tag**: the strixflerp interface class the module matches (a class some catalogue module already provides), or
  none. The module's `[[provides]]` names what it would be called.
- The transport under the rig (OdinLink-Five: driver, library, RCCL / NCCL net plugins) is its own origin,
  `catalog/odinlink5`.

| module | lang | files | lines | provides | tag |
|---|---|---:|---:|---|---|
| `kernel.vllm_strix.moe_int4_gemv` | hip-c++ | 1 | 1206 | `moe.experts-decode/awq-int4/bf16`, `moe.experts-decode/mxfp4/bf16` | `moe.experts-decode` |
| `kernel.vllm_strix.w8a16_gemv` | hip-c++ | 1 | 538 | `linear.decode/w8a16-g128/bf16`, `linear.decode/fp8-block128/bf16`, `moe.gate-logits/bf16/bf16-f32`, `moe.router/-/f32.sigmoid-bias-topk` | `linear.decode` |
| `runtime.vllm_strix.odl_ar2` | hip-c++ | 4 | 1024 | `collective.allreduce-p2p/-/bf16.odl-stream` | `collective.allreduce-p2p` |
| `tool.vllm_strix.odl_ar2_bench` | hip-c++ | 1 | 180 | `tool.collective-bench/-/-` | none |
| `tool.vllm_strix.fp8_sweep` | hip-c++ | 3 | 155 | `tool.kernel-bench/fp8-block128/-` | `tool.kernel-bench` |
| `tool.vllm_strix.w8_sweep` | hip-c++ | 3 | 152 | `tool.kernel-bench/w8a16-g128/-` | `tool.kernel-bench` |
| `kernel.vllm_strix.kpool_paged_logits` | python | 3 | 256 | `attn.indexer-score/fp8-kpool-shuffle/f32` | `attn.indexer-score` |
| `kernel.vllm_strix.idx_logits` | python | 3 | 172 | `attn.indexer-score/fp8-ds4/f32.wmma` | `attn.indexer-score` |
| `kernel.vllm_strix.sparse_attn_split` | python | 2 | 148 | `attn.sparse-mla/-/bf16.split-kv` | none |
| `runtime.vllm_strix.moe_int4_dispatch` | python | 3 | 395 | `moe.experts-decode-dispatch/awq-int4/-` | none |
| `runtime.vllm_strix.moe_direct` | python | 3 | 152 | `moe.experts-decode-direct/awq-int4/bf16` | `moe.experts-decode` |
| `runtime.vllm_strix.mxfp4_direct` | python | 2 | 126 | `moe.experts-decode-direct/mxfp4/bf16` | `moe.experts-decode` |
| `tool.vllm_strix.mxfp4_sweeps` | python | 7 | 356 | `tool.kernel-bench-routing-replay/mxfp4/-` | `tool.kernel-bench` |
| `runtime.vllm_strix.w8a16_requant` | python | 3 | 381 | `linear.requant-load/bf16-to-w8a16-g128/-` | none |
| `runtime.vllm_strix.gate_lmhead` | python | 2 | 87 | `moe.gate-logits/f32/f32`, `logits.lm-head/w8a16-g128/bf16` | `moe.gate-logits` |
| `runtime.vllm_strix.gate_bf16` | python | 4 | 100 | `moe.gate-logits/bf16/f32` | `moe.gate-logits` |
| `runtime.vllm_strix.fused_router` | python | 3 | 73 | `moe.router/-/f32.sigmoid-bias-topk-fused` | `moe.router` |
| `runtime.vllm_strix.router_det_topk` | python | 3 | 772 | `moe.router/-/f32.stable-topk` | `moe.router` |
| `runtime.vllm_strix.router_packed_topk` | python | 2 | 60 | `moe.router/-/f32.packed-key-topk` | `moe.router` |
| `runtime.vllm_strix.fp8_gemv_hook` | python | 3 | 193 | `linear.decode/fp8-block128/bf16.hook` | `linear.decode` |
| `runtime.vllm_strix.fp8_prefill` | python | 2 | 225 | `linear.prefill/fp8-block128/bf16.dequant-hipblaslt` | `linear.prefill` |
| `runtime.vllm_strix.sparse_attn_split_hook` | python | 1 | 35 | `attn.sparse-mla-dispatch/-/-` | none |
| `runtime.vllm_strix.kpool_paged_logits_hook` | python | 1 | 75 | `attn.indexer-score-dispatch/fp8-kpool-shuffle/-` | `attn.indexer-score` |
| `runtime.vllm_strix.pr59412_indexer_blocks` | python | 2 | 140 | `kvcache.block-align-indexer/-/-` | none |
| `runtime.vllm_strix.idx_force_tail` | python | 2 | 917 | `attn.indexer-topk/-/f32.force-tail` | `attn.indexer-topk` |
| `runtime.vllm_strix.idx_fp8max_fnuz` | python | 1 | 50 | `attn.indexer-quant-fnuz-safe/fp8-e4m3fn/-` | none |
| `runtime.vllm_strix.idx_pool_subchunks` | python | 1 | 1830 | `attn.indexer-chunking-pool-aligned/-/-` | none |
| `runtime.vllm_strix.idx_bt_gather_v4` | python | 1 | 4384 | `attn.indexer-gather-bt-expand/-/-` | none |
| `runtime.vllm_strix.ragged_compact_ops` | python | 1 | 4335 | `attn.sparse-window-ops-ragged-compact/-/-` | none |
| `runtime.vllm_strix.pr55222_workspace` | python | 2 | 2434 | `attn.workspace-units/-/-` | none |
| `runtime.vllm_strix.pr57979_kda_stride` | python | 2 | 2059 | `mixer.kda/-/bf16.stride-fix` | none |
| `runtime.vllm_strix.apc_align` | python | 1 | 194 | `cache.prompt-align-spec/-/-` | `cache.prompt` |
| `runtime.vllm_strix.apc_retention` | python | 1 | 73 | `cache.prompt-state-interval/-/-` | `cache.prompt` |
| `runtime.vllm_strix.adaptive_k` | python | 2 | 395 | `speculation.draft-length-ema/-/-` | `speculation.draft` |
| `runtime.vllm_strix.dflash_drafter_kv_group` | patch | 1 | 170 | `speculation.draft/dflash/bf16.kv-group` | `speculation.draft` |
| `runtime.vllm_strix.eagle3_aux_states` | patch | 1 | 92 | `speculation.aux-hidden/-/bf16` | `speculation.draft` |
| `runtime.vllm_strix.triton_ptr_cache` | python | 1 | 162 | `runtime.launch-pointer-cache/-/-` | none |
| `runtime.vllm_strix.cg_eager_collectives` | python | 3 | 133 | `runtime.graph-capture-break-collectives/-/-` | none |
| `runtime.vllm_strix.shm_spin` | python | 1 | 98 | `runtime.mq-spin-window/-/-` | none |
| `runtime.vllm_strix.wna16_empty_cache` | patch | 1 | 21 | `format.load-release-staging/-/-` | none |
| `runtime.vllm_strix.aiter_gfx1151_gate` | patch | 1 | 17 | `runtime.platform-gate-aiter-gfx1151/-/-` | none |
| `runtime.vllm_strix.fp8_fnuz_mqa` | python | 2 | 685 | `attn.indexer-score/fp8-fnuz/f32.paged-mqa` | `attn.indexer-score` |
| `runtime.vllm_strix.mhc_no_tilelang` | patch | 1 | 15 | `runtime.platform-gate-mhc-tilelang/-/-` | none |
| `runtime.vllm_strix.ds4_native_port` | python | 6 | 7752 | `runtime.model-port-ds4-vllm031/-/-` | none |
| `runtime.vllm_strix.ds4_gfx1151_gates` | patch | 5 | 95 | `runtime.platform-gate-ds4-gfx1151/-/-` | none |
| `runtime.vllm_strix.ds4_decode_inv_rope` | python | 1 | 59 | `pos.inverse-rope-rows/-/bf16` | none |
| `runtime.vllm_strix.ds4_decode_rows` | python | 1 | 45 | `kvcache.workspace-cap-ds4-decode-logits/-/-` | none |
| `runtime.vllm_strix.ds4_idx_hadamard` | patch | 3 | 457 | `attn.indexer-quant-hadamard/fp4/-` | none |
| `runtime.vllm_strix.ds4_idx_logits_hook` | python | 1 | 87 | `attn.indexer-score-dispatch/fp8-ds4/-` | `attn.indexer-score` |
| `runtime.vllm_strix.ds4_mhc_cfg` | python | 5 | 231 | `mixer.hyper-connection/-/bf16.gfx1151-tiles` | none |
| `runtime.vllm_strix.ds4_mmf32` | python | 1 | 68 | `linear.dense/bf16/f32.ds4-compressor` | `linear.decode` |
| `runtime.vllm_strix.cgfusion_soft` | python | 1 | 23 | `runtime.platform-gate-ds4-cgfusion/-/-` | none |
| `runtime.vllm_strix.moe_tile_configs` | python | 2 | 190 | `moe.tile-config/awq-int4/-` | none |
| `runtime.vllm_strix.aiter_gemm_configs` | python | 45 | 4244 | `linear.tile-config/fp8-a8w8/-` | none |
| `runtime.vllm_strix.odl_ar2_hook` | patch | 1 | 35 | `collective.allreduce-dispatch-size-split/-/-` | `collective.allreduce-p2p` |
| `runtime.vllm_strix.odl_mq` | python | 1 | 250 | `transport.message-queue-odl/-/-` | none |
| `runtime.vllm_strix.odl_mq_hook` | patch | 1 | 138 | `transport.message-queue-dispatch/-/-` | none |
| `runtime.vllm_strix.odinlink_local_patch` | patch | 2 | 1008 | `transport.odl-hardening/-/-` | none |
| `runtime.vllm_strix.odl_state` | c | 1 | 17 | `tool.link-state/-/-` | none |
| `tool.vllm_strix.odl_pingpong` | c | 1 | 174 | `tool.link-bench-rtt/-/-` | none |
| `runtime.vllm_strix.nhi_throttle` | c | 2 | 75 | `transport.nhi-irq-throttle/-/-` | none |
| `runtime.vllm_strix.odinlink_host_kit` | shell | 7 | 446 | `tool.transport-install-odinlink/-/-` | none |
| `serve.vllm_strix.launcher` | shell | 5 | 462 | `serve.cluster-launcher/-/-` | none |
| `serve.vllm_strix.cluster_env` | shell | 3 | 172 | `serve.transport-profile/-/-` | none |
| `serve.vllm_strix.manual_serve` | shell | 6 | 786 | `serve.vllm-launch/-/-` | none |
| `serve.vllm_strix.warmup` | python | 1 | 74 | `serve.warmup/-/-` | none |
| `serve.vllm_strix.toolcall_failopen` | python | 1 | 154 | `serve.toolcall-parse-fail-open/-/-` | `serve.http` |
| `tool.vllm_strix.toolcall_drop_log` | python | 1 | 119 | `tool.parser-drop-log/-/-` | none |
| `tool.vllm_strix.sync_instr` | python | 1 | 86 | `tool.host-sync-probe/-/-` | none |
| `tool.vllm_strix.gpu_sclk` | shell | 1 | 12 | `tool.gpu-clock-cap/-/-` | none |
| `tool.vllm_strix.container_ops` | shell | 4 | 337 | `tool.image-build/-/-` | none |
| `tool.vllm_strix.decode_bench` | python | 8 | 643 | `tool.decode-bench-spec-receipts/-/-` | `tool.prefill-bench` |
| `tool.vllm_strix.context_probes` | python | 5 | 197 | `tool.context-probe/-/-` | none |
| `tool.vllm_strix.quality_probes` | python | 8 | 666 | `tool.quality-ab/-/-` | none |
| `tool.vllm_strix.toolcall_probes` | python | 5 | 359 | `tool.toolcall-replay/-/-` | none |
| `tool.vllm_strix.trace_analysis` | python | 10 | 531 | `tool.trace-analysis/-/-` | none |
| `tool.vllm_strix.ds4_probes` | python | 6 | 226 | `tool.model-probe-ds4/-/-` | none |
| `tool.vllm_strix.debug_hooks` | python | 15 | 14043 | `tool.debug-hooks/-/-` | none |

78 modules, 255 files, 59326 lines; 30 carry a tag, 48 are none.

Not harvested (ORIGIN.toml [harvest] unharvested lists every file):

- `container/pinned-vllm/, container/pinned-triton/`: vLLM and Triton files with the patches applied, kept by the origin for byte-exact rebuilds: upstream code, not this repo's (the vsh_* files inside are byte-identical to container/patches copies, harvested there)
- `container/patches/libvsh_*.so`: prebuilt gfx1151 libraries of vsh_moe_int4.hip and vsh_w8a16.hip; the sources are harvested and compiled by the gate
- `scripts/data/ds4_dspark_routing.pt`: binary routing dump of real DS4 decode (input to tool.vllm_strix.mxfp4_sweeps)
- `scripts/download-models.sh`: downloads model weights (the catalogue holds no weights)
- `host/__pycache__/, *.pre-ds4native, container/Dockerfile.pre-ds4fnuz`: bytecode and pre-migration backups
- `README.md, PATCHES.md, PENDINGWORK.md, AGENTS.md, FIXSPEC.md, FRESH-EYES-20tps.md, HANDOFF-*.md, HANDOVER-tool-calls.md, host/PATCHES.md`: documentation and history; read for the module records and ORIGIN.toml
- `LICENSE, THIRD_PARTY_NOTICES.md`: copied beside ORIGIN.toml
- `.gitignore`: repository plumbing
