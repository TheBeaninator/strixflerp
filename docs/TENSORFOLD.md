# Upstream TensorFold: what it has that StrixFold's fork does not, and which recipes pay on the pod

Written 2026-11-12 from upstream TensorFold `000b534` (main, 2026-11-12), harvested as origin `tensorfold`
(`catalog/tensorfold`, `docs/harvest/harvest_tensorfold.py`). Nothing of upstream was built or run; no GPU was used.
Labels as in `docs/CRUCIBLE-VS-STRIXITE.md`: **[claimed]** upstream's own CHANGELOG / docs, **[measured]** ours,
**[inferred]** our arithmetic. Upstream paths are origin-relative (`zig/...`, `docs/...`).

The principle (`docs/SHAPE-TRANSFERS.md`): reuse across engines means carrying a module's recipe (algorithm, layout,
work mapping, schedule) into another format, "the same recipe in a different size pan". Gate: bit identity where the
target's rounding points and summation order are kept, `docs/FIDELITY.md` where they are not.

## 1. Lineage, and what the harvest holds

StrixFold (`catalog/strixfold`, f64b5ff) descends from TensorFold **1.0.2** by way of bertholomus/TensorFold
`deepseek-v41-tp2`. Upstream is now 1.2.7 + Unreleased. Two facts reframe the task brief:

- **Upstream's HIP backend is not StrixFold's.** Upstream grew its own (`zig/src/hip`, `zig/kernels/hip`) from 1.0.5
  (#463) on; it is MLX-affine and BF16 only, with no EXL3 and no DeepSeek. StrixFold's HIP path
  (`gpu/hip_facade`, the dsv41 kit, the Triton AOT kit) is the fork's own and has no upstream counterpart.
- **Upstream's HIP registry** (`zig/src/native/hip.zig:22`) is: Qwen3.5/3.6/3.8 dense and MoE, Flash Next, Nemotron
  3.5 (Nemotron-H), Qwen3 (embedding / reranker), GLM-OCR, dots.mocr, Nemotron Parse, LightOnOCR. **GLM-5.3-Flash,
  Kimi K3 and DeepSeek-V4.1 have no HIP path upstream**: GLM is CUDA (TP2/TP3) and paired Metal, Kimi is Metal,
  DeepSeek-V4.1 is CUDA and Metal and "no release binary serves DeepSeek checkpoints until its whole-model gate (#827)
  passes" (`docs/known-issues.md`).
- **Upstream's DeepSeek family is a rewrite**, `zig/src/families/deepseek_v41` (358 files), not StrixFold's
  `families/dsv41` (77). Four basenames are shared (`model.zig`, `plan.zig`, `config.zig`, `engram_io.zig`), all
  changed (732 to 1,290 changed lines each). Its CUDA kernels say "Copy of src/tensorfold/families/deepseek_v41/cuda/..."
  (the Python 0.6.6 engine).

Harvest: 46 modules, 2,497 files, byte-identical to the checkout (`flerp diff-origin`), 8,843 origin files listed as
unharvested (docs, receipts, apps, tests, Metal/Vulkan/XPU, CUDA runtime, families with neither HIP nor a pod model).

| kind | modules |
|---|---|
| kernel, HIP (12) | `hip_common`, `hip_quant` (MLX affine / BF16 decoders), `hip_tiles` (dot2 GEMV, gfx11/gfx12 WMMA GEMM), `hip_ops`, `hip_attention`, `hip_decode`, `hip_recurrence` (GDN, Mamba-2), `hip_flashnext` (QSA indexer, sparse attention), `hip_draft`, `hip_vision_ocr`, `hip_joint_schema`, `hip_affine` |
| kernel, CUDA (8, read for recipes, `lang = "cuda"`, never built) | `cuda_common`, `cuda_dsv41`, `cuda_exl3` (incl. `exl3_block/x3gm*`), `cuda_glm` (KDA, DSA indexer), `cuda_nemotron`, `cuda_flashnext`, `cuda_experts`, `cuda_roce_gather` |
| runtime (8) | `hip`, `hip_comm` (RCCL, TCP link), `hip_serve`, `fabric`, `lanes`, `prompt_cache`, `core`, `engine_host` |
| format (3) | `safetensors`, `quant` (MLX affine, FP8), `exl3` (EXL3 + EXLR8 host) |
| executor (12) | `qwen3_5`, `flashnext`, `nemotron`, `qwen3`, `glm_ocr`, `dots_mocr`, `nemotron_parse`, `lighton_ocr` (HIP); `deepseek_v41` (no fixtures), `glm`, `glm_dsa`, `kimi_k3` (no HIP) |
| serve / tool | `serve.http`; `tool.checkpoint_cli`, `tool.zig_build` (the HIP build half) |

**Relation to StrixFold's modules** (same path, `HARVEST.json relation_to_strixfold`; changed = +/- lines of a
zero-context diff): core 32 paths (8 identical, 3,344 changed), lanes 27 (3, 5,593) + sampling 1 (133), fabric 25
(7, 470), prompt_cache 6 (0, 1,877), engine_host 4 (0, 1,604), safetensors 4 (0, 796), server 55 (2, 10,304),
checkpoint_cli 5 (0, 992), zig_build 3 (1, 701). Same files ten upstream releases later, not duplicates.

Licences: Apache-2.0; pre-0.6.0 code keeps MIT (`LICENSES/MIT.txt`); EXL3 sources after ExLlamaV3 (MIT, Turboderp)
are `Apache-2.0 AND MIT`; `glm_ocr/pointwise.hip` adapts PyTorch (BSD-3-Clause); `hip/attention_aotriton.zig` follows
AOTriton's ABI (MIT); Kimi fixtures and the server's Kimi prompt adapter are under the Kimi K3 / K3.1-Pro licences.
All texts are in `catalog/tensorfold/` (LICENSE, NOTICE, THIRD_PARTY_NOTICES.md, LICENSES/).

## 2. What upstream's numbers mean for us

- **Upstream's maintainers have no AMD GPU** ("The records are @AbrahamCete's own, as we have no AMD GPU", #1694).
  gfx1151 numbers are contributors' (AbrahamCete, gilby, ThinkOffApp) on a Ryzen AI MAX+ 395 / 8060S, the pod's chip;
  the qualification receipt (`docs/receipts/gfx1151-qwen-833bbfcb`, v1.0.12) used Ubuntu HIP 7.1.52801 and clang 21,
  ReleaseFast, kernel 7.0.0-38. We build with TheRock 10.2 (HIP 7.17, clang 24): same chip, different compiler.
- Many HIP numbers are on a **Radeon AI PRO R9700 (gfx1201)**: discrete, ~640 GB/s, WMMA12. Those (reranker,
  embedding, `--policy matrix=on`) do not transfer to gfx1151; the gfx12 WMMA tile does not even run here.
- Flash Next HIP tok/s (`docs/recipes/qwen3.8-flash-next.md` "Native HIP on a Strix Halo iGPU") are **with MTP, 3
  drafts**, `--parallel 1`, `--context 131072`, prompt cache off, decode medians of `bench_openai.py` over two starts:
  50.0 / 44.0 tok/s code / chat on the uniform 4-bit MLX pack [claimed]. Compare with strixite's and Crucible's
  drafted numbers, not with 32-34 tok/s plain decode.
- CUDA numbers (GLM, DeepSeek, EXL3 experts, RoCE gather) are on **GB10 (DGX Spark)** with ConnectX-7; ours is CX6 at
  100 GbE through the SN2700, host-staged on unified memory. Shapes transfer; absolute numbers do not.
- Speed claims are "same bits" claims almost everywhere: the kernel contract (`docs/kernel-contract.md`) holds a new
  tile byte-equal to a frozen copy of the old. That is the bit-identity gate we want, already done upstream.

## 3. Recipes, ranked by expected gain for the pod

Pod targets: StrixFold DeepSeek-V4.1 prefill (per `gate/results/prefill-dsv41/profile-2026-11-12T0342Z/phases.md`
[measured]: 904 ms GPU a 1,024-row chunk; moe 34.2 / 41.7 %, gather_m 20.5 / 13.0 %, gather_a 12.4 / 12.8 %, attn
17.6 / 17.2 %, engram 10.3 % on ranks 0 / 1), DSv41 decode (71 ms a round: moe 45.9 %, gather_a 17.7 %), Ornith four-node
decode (~13 tok/s), and models the pod cannot serve today.

### R1. Prefill exchange overlap in row pieces (`TF_DSV41_PF_OVERLAP`, `_SITE`, `_TBO`) -> StrixFold prefill

- **What / key idea.** Upstream's DeepSeek prefill (`deepseek_v41/block_prefill_attn.zig:247`, `block_prefill.zig:40`,
  `block_prefill_emit.zig:155`, tests `pfovl_test.zig`) cuts the row-independent output projection (wo_a / wo_b, and the
  MoE down) into k row pieces (k = 2..8, at least 256 rows a piece: "smaller GEMMs cost more than they hide") and sends
  piece p's partial all-gather on a side stream while piece p+1's GEMM runs. `_SITE` lets the next mHC boundary site
  start on the pieces already gathered (stream marks). `_TBO` runs two consecutive 512+-row encoder segments as one
  program on two streams, B half a layer behind A, so A's exchanges hide behind B's experts.
- **Layout / mapping.** No new kernel: the same GEMMs on row views; the gathered buffer is the same bytes in the
  same place ("piece rows all-gathered into L.xrows, copied into place: same bytes", `forward_glue.zig:120`).
- **Measured effect.** None published [claimed: tests only]. Ours: gathers are 25-33 % of the chunk and unhidden on
  both ranks; `docs/PREFILL-DSV41.md` section 4 showed the half-chunk zip hides only some of them.
- **StrixFold (EXL3, TP2).** Carries over whole: the attention output and the experts' down output in
  `families/dsv41/model.zig` `fillSplit` become k pieces, each handed to `tcp_proxy` / `rdma_proxy` (whose slot
  collective is already graph-capturable and stream-ordered) behind the next piece. New code: piece loop and a
  second HIP stream with events in the step order (~150 lines of `model.zig`, inferred). TBO is the larger form of the
  same idea and overlaps with section 5 of PREFILL-DSV41 (larger units); try pieces first.
- **strixite / strixflerp GGUF / Crucible.** Single node or pipeline stages: no TP gather in prefill; does not apply,
  except Ornith's stage hand-off (R6).
- **Gate.** Bit identity of prompt logits and kept states (row-independent GEMMs, same gathered bytes), then the phase
  timer: gather_a + gather_m must fall. **Expected gain [inferred]:** hide up to the gather share less the piece
  overhead: 15-25 % of chunk time at k = 2..4. **First experiment:** k = 2 on gather_m alone (the 186 ms on rank 0)
  at 2,048-row chunks, phase timer on, bits checked.

### R2. Routed-expert prefill: tile chosen by rows an expert, tiling-free sums -> StrixFold, Crucible

- **What / key idea.** Two upstream pieces. (a) `cuda_exl3`: `exl3_block/x3gm*.cu`, "each output is one mma chain over
  ascending k from +0, any tiling", grouped by (uids, ucount, members) built on the device, mixed bit widths a layer,
  no atomics (`docs/recipes/exl3-universal-experts.md`): because a sum's order does not depend on the tile, the tile
  can be chosen per shape without changing bits. (b) HIP on gfx1151 (`hip_tiles`, CHANGELOG #1658, #2087, #2576):
  routed prompt products pick the 64-row tile only while experts average at most 64 rows, else a skewed 128-row tile
  running three blocks a WGP up to 8,192 codes deep.
- **Measured effect [claimed].** Grouped experts 1.5-3.6x ExLlamaV3's mixed-K dispatch on GB10, equal on uniform
  layers; #2576 on gfx1151: Flash Next 8K prompts ~5 % sooner, Nemotron ~6 %; #1467 (EXL3 prefill on CUDA) 38-42 %.
- **StrixFold.** Our profile: experts are 34-42 % of a chunk with a ~15 ms fixed decode cost a pass
  (`docs/PREFILL-DSV41.md`). Carries over: the decision rule (tile by mean rows an expert, 385 experts at 1,024-2,048
  rows -> ~20-40 rows each, so the small tile is right today and the 128-row tile becomes right at 4,096-row units),
  and making the WMMA accumulation order tile-independent so the choice is free. Changes: EXL3 trellis decode to LDS
  replaces MLX dequant. Gate: bit identity against today's `exl3.pf_experts` (`bench.json` checks already show one pass
  over 2,048 equals two halves). **Gain [inferred]:** 5-15 % of the experts phase; larger once units grow.
- **Crucible (EXL3, one node).** Its `moe_prefill` is already 192x128 items with stable dispatch; the transfer is
  the per-shape tile switch. **strixite / GGUF.** Same rule for Q4/Q8 and K-quant experts; FIDELITY gate if the
  order changes.

### R3. Sparse-attention prefill: a lane holds a key and several heads -> StrixFold, strixite, Crucible

- **What / key idea.** `hip_flashnext` (`flashnext/sparse.hpp`, `sparse_lanes.hpp`, #1877): for QSA prefill a lane
  holds one key and several query heads, keys and values staged in the cache type, so the selected key is read once
  for all heads sharing it; same sums. **[claimed]** gfx1151: cold prompts of 2,048-63,728 tokens finish 29-41 % sooner.
- **StrixFold.** DSv41 sparse attention (attn 17 % of a chunk) is a different shape (top-512 compressed KV, FP8 KV,
  64 heads a rank, Triton and `sparse_attn.hip`); the head-sharing mapping carries over to the HIP kernel,
  `kernel.strixfold.sparse_attn`. Gate: bit identity if the per-head softmax order is kept. **Gain [inferred]:** a
  quarter to a third of attn -> 4-6 % of chunk.
- **strixite / Crucible (Flash Next).** Same model and the same QSA shape: the most direct transfer in this document.
  Compare first: `kernel.crucible.qsa` (`QsaUnionFlashKernel`, four query rows sharing KV) already shares KV across
  rows; upstream shares across heads. Gate: bit identity per kernel.

### R4. Mapped tables read at random: advise, release, gather in parallel -> StrixFold Engram, strixite PLE

- **What / key idea.** Flash Next on HIP advises random access on the n-gram table's mapped tensors so a scattered row
  read faults only its own page, gathers n-gram rows in parallel, and drops the checkpoint's pages after upload
  (#1611, #1615). **[claimed]** gfx1151: first request 24.4 -> 35.8 tok/s, cold 7K prompt 48.4 -> 28.9 s, page cache
  31.5 -> 2.8 GiB. Upstream DeepSeek also reads Engram through the page cache where direct I/O needs 4 KiB sectors
  (#2610).
- **StrixFold.** Engram is 10.3 % of a prefill chunk and 3.1 % of decode (`format.strixfold.engram`, O_DIRECT /
  io_uring rows). Carries over: the read pattern question (direct rows vs advised mmap) and dropping load-time pages.
  Gate: bit identity (bytes read are the bytes). **Gain [inferred]:** up to the engram share if the reads are on the
  critical path; first experiment is timing engram with the rows pre-faulted (upper bound) before any code.
- **strixite / Crucible.** PLE n-gram rows (`format.crucible.ngram16`, strixite PLE) are the same table: cheap check of
  cold first-request decode.

### R5. gfx11 WMMA prompt tile hygiene: no scratch spills, row-block bands -> strixite, StrixFold, Crucible

- **What / key idea.** (a) #1663: the matrix-core prompt tile used a run-time division and a select per accumulator;
  shift-and-mask indexing and a uniform zero-start branch fit it in 248 VGPRs with no scratch: **[claimed]** gfx1151
  Qwen3.8-27B MLX 4-bit, seven prompt products at 2,048 rows 105.8 -> 63.7 ms, cold prefill -32 % at 2k. (b) #1648:
  blocks run in bands of four row blocks so neighbours reuse a column block's weights from L2: -12 % at 2k, -9.5 % at
  64k. (c) #1685: round to the activation type in the epilogue instead of fp32 + cast launch, -2-3 %. All same bits.
- **strixite (strixw Q4/Q8, panel layout).** The panel layout (`docs/STRIXW-LAYOUT-NEXT.md`) changes what a block reads;
  banding is a launch-order change on top of it. First experiment: dump ISA (`--save-temps`) of strixite's prefill
  GEMM and StrixFold's `exl3_pf_experts` / dense pf kernels, grep `scratch_` and VGPR counts; a spilling WMMA tile is
  the cheapest win in this list. Gate: bit identity (order unchanged).
- **Crucible.** `kernel.crucible.exl3_gemm` decodes trellis tiles to LDS: same check. **GGUF executor.** Its prefill is
  not WMMA today; the recipe applies when it is.

### R6. One-shot small all-gather with device-side flags -> Ornith four-node, DSv41 decode

- **What / key idea.** `cuda_roce_gather` + `docs/roce-gather.md`: one kernel stages the shard in pinned memory and rings
  a doorbell; a proxy thread RDMA-writes it to every peer per HCA followed by a sequence flag on the same QP; one thread a
  (peer, HCA) spins on the flag; a device epoch makes it graph-replayable; a stalled stripe poisons the exchange and
  names the rank. **[claimed]** GB10/CX-7: 10.4 vs NCCL 12.3 us at 4 KiB, world 4 behind a switch 22.0 vs 39.4 us at
  32 KiB.
- **StrixFold.** `runtime.strixfold.coll_rdma` (pod SFR-2) is already this shape (slot collective, write-with-immediate).
  Decode's gather_a at 12.6 ms a round is three orders above this latency, so the cost is waiting on the other rank
  (imbalance or host sync), not the wire: measure per-rank arrival skew first.
- **strixflerp GGUF four-node (Ornith, ~13 tok/s).** Stage hand-off is point-to-point, not a gather; the transferable
  parts are the flag-after-stripe ordering on one QP and the device epoch for graph replay. **Gain [inferred]:** small
  unless hand-offs are a measured share of the ~77 ms token; first experiment is that share.

### R7. Decode rounds: weight reuse across lanes, wider key walks, early loads

**[claimed]** gfx1151: a round of more than four rows reuses cached weights across row blocks (27B at 16 streams
51 -> 75 tok/s, #1694); decode attention takes four keys a warp on long spans (7.5-18.9 % at 32-64K, #1932); the
residual-add + norm tail issues loads before branches (2.7 %, #2247). Carries over to StrixFold's DSv41 verify windows
(moe 46 % of a round), strixite's multi-stream decode and Crucible's 1..8-row GEMV; bit identity in each case (same folds).

## 4. Models the pod could serve with upstream's HIP path

- **Nemotron 3.5 Lightning 30B-A3B (MLX 4-bit, `mlx-q4g64`)**: HIP with MTP and TP since 1.1.4 (#1082-#1103). A new
  family for the pod (hybrid Mamba-2), small enough for one worker. No gfx1151 receipt upstream.
- **Qwen3.6-35B-A3B MTP, Qwen3.8-27B, Qwen3.5-9B (MLX 4-bit)**: qualified on gfx1151 (receipt v1.0.12).
- **Qwen3.8-Flash-Next (MLX affine, group 32)**: 50 / 44 tok/s with MTP [claimed]; the pod already serves it
  (strixite, Crucible): a third engine to compare, not a new model.
- **OCR / retrieval**: GLM-OCR, dots.mocr, LightOnOCR-3, Nemotron Parse, Qwen3-Embedding/Reranker-8B (mostly gfx1201
  checked).
- **Not servable through upstream HIP**: GLM-5.3-Flash (its KDA and DSA indexer exist only as CUDA,
  `kernel.tensorfold.cuda_glm`, and Metal), Kimi K3, DeepSeek-V4.1. A GLM-5.3-Flash HIP path would be new work: port
  `glm_kda.cu` (fp32 128x128 state a head, 34 KDA layers, 32 heads a rank at TP2 [claimed]) and the DSA indexer, on
  StrixFold's TP2 collectives. The pod's 2026-10-15 verdict parked GLM-5.3-Flash on size; nothing here changes that.

## 5. Should StrixFold pull from upstream's DeepSeek path?

Not files: the trees share no DeepSeek lineage past 1.0.2. Ideas and fixes worth carrying:

| upstream | what | for StrixFold |
|---|---|---|
| `TF_DSV41_PF_OVERLAP` / `_SITE` / `_TBO` (`block_prefill*.zig`, `pfovl_test.zig`) | R1 | prefill gathers |
| #2491 `tp.device` | N TP ranks as threads on one GPU with device copies; exchanges checked bit for bit at worlds 1-4 | a one-GPU gate for R1 without the pair (PREFILL-DSV41 notes none exists) |
| #2182 | NVMe session tier restore: a 319K-token conversation recalled in 1.2-1.4 s instead of 155-178 s, byte-identical | StrixFold's Imprint states |
| #2610 | Engram rows through the page cache on 4 KiB-sector disks | worker /cache NVMe sector size |
| #1681 | indexer scratch planned before the workspace is priced | ragged prompts |
| #2325 | heads split by whole output groups (24/24/16 at three ranks) | TP3 / uneven ranks |
| `cuda_dsv41/mhc_pf.cu`, `l2pf.cu`, `l2pace.cu` | mHC with prefetch; L2 prefetch / pacing | read for StrixFold's `mhc_mix` |

Shared host files (lanes, prompt cache, server, fabric) are ten releases apart (section 1 counts); rebasing them is a
merge, not a pull, and none is on the profile's critical path.
