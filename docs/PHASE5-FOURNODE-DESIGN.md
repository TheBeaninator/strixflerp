# Phase 5: Ornith-1.5 397B on four nodes with `executor.flerp.qwen35moe` (design)

Written 2026-11-11 from the tree; nothing here was built or run. Tags: **[R]** read in the tree at the cited file:line, **[I]** inferred or computed here, **[N]** not known or assumed.
Path roots, all under `/home/user/Code/strixflerp/` unless absolute:

- `Q:` = `catalog/flerp/executor/qwen35moe/src/flerp/qwen35moe/src/models/qwen/`
- `M:` = `catalog/flerp/executor/qwen35moe/src/flerp/qwen35moe/`
- `K:` = `catalog/flerp/kernel/moe_kquant/src/flerp/moe_kquant/`
- `G:` = `catalog/gufo/`
- `F:` = `catalog/gufo/executor/qwen38_flash_next/src/src/models/qwen38_flash_next/kernels/rocm/`
- `T:` = `catalog/gufo/runtime/tp_verbs/src/src/`
- `S:` = `catalog/strixfold/runtime/`
- `U:` = `/home/user/Code/strixflerp-upstream/strixite-tp-rdma-v2/`

## 0. Decision

1. **First tokens: split (a), layer stages.** Each node runs the existing single-node executor on a contiguous block range and relays the FP32 hidden rows to the next node.
   - Nothing crosses the wire inside a layer, so the decode graph and every kernel stay as they are.
   - The logits are the single-node executor's bit for bit at any number of stages, and that is testable on one GPU.
2. **Useful speed, same split:** experts grouped by expert inside a prompt chunk (a single-node kernel), then chunks and conversations pipelined through the stages.
3. **Expert-parallel decode (split d) is an optional later overlay**, decided by one measurement on two nodes (M1, section 6).
   - My byte count gives it at most about 1.35x on Ornith single-stream decode, before 60 collectives a token **[I]**.
   - It needs a four-rank graph-capturable collective that does not exist in C++ in the catalogue.
4. **Split (c) is rejected for Ornith** unless the single-node numeric contract is changed first: it cannot reproduce today's bits (section 2).

The one loader serves (a), (b) and (d): a placement is "which dense blocks and which experts of which blocks".

## 1. What exists

**The single-node executor [R]**

- The MoE block is seven launches, none decided on the host (`M:moe_ffn.hpp:13-17`): `GateLogits`, `RouterTopK`, four `IndexedExperts`, `MoeEpilogue` (`M:moe_ffn.cpp:52-90`).
- Decode calls it per block at `Q:hip/decode_step.cpp:228-238`; the prompt chunk at `Q:hip/prefill_chunk.cpp:572-585`.
- The decode step is one loop over all blocks (`Q:hip/decode_step.cpp:76`): embedding at `:67-69`, head and argmax at `:265-279`.
- The origin captures that step into a HIP graph (`G:executor/qwen/src/src/models/qwen/hip/decode.cpp:86-93`, `G:runtime/hip_host/src/src/core/hip/detail/hip_graph_decode_executor.hpp:66`).
- An expert row is one wave per output element with gufo's row reader, the same summation order at any token count (`K:moe_kquant.hpp:11-12`, `K:moe_kquant.hip:68`, `:98-126`).
- The routed sum runs slot by slot from zero, then adds the gated shared expert (`G:kernel/qfn_kernels/.../kernels.hip.cpp:3530-3558`).
- A prompt chunk reads every selected expert once per token, with no grouping (`catalog/flerp/kernel/moe_kquant/module.toml`, regime; `docs/PHASE5-ORNITH-GAPMAP.md:108-110`).

**The loader [R]**

- One device allocation per mapped GGUF shard, `hipMalloc(source.size)` (`Q:hip/model_loader.cpp:70-73`, `:190-271`).
- Tensors are then re-pointed by host address (`RemapTensor`, `:273-296`; the list at `:360-379`). An empty tensor remaps as true (`:276-278`).
- The file is mapped `PROT_READ / MAP_PRIVATE` (`G:runtime/platform/src/src/core/platform/mapped_file.cpp:166-172`, `:262-270`).
- A byte-range uploader already exists: `WeightUpload::Copy(shard, offset, size, device)` and `CopyMapped(source, size, device)`, which "resolves a pointer within the supplied mappings without reading its pages" (`G:kernel/core_headers/src/src/core/hip/weight_upload.hpp:25-29`).
  - It reads through an unbuffered reopen of the file and 16 MiB staging (`G:runtime/hip_host/src/src/core/hip/weight_upload.cpp:23`, `:127-140`).
  - The loader uses it only on the path where mapped chunks cannot be visited (`Q:hip/model_loader.cpp:160-175`).
  - gufo's Flash-Next loader uses the same uploader for per-rank slices (`F:device_model.cpp:179-221`).

**Gate state of the executor [R]**

- On Qwen3.6-35B-A3B the runs equal themselves and the dense regression holds; the llama.cpp tolerance failed on two prompts (`gate/results/qwen36-35b-a3b.flerp-moe/summary.md:186-190`, `:203-210`).
- Against the full-precision reference the decode step has mean KL 0.00013; the prompt pass has 0.00099, from activation quantization in the origin's chunk (`docs/FIDELITY.md:30-52`).
- Decode is 56.4 tok/s, the prompt pass about 230 tok/s at 1,808 tokens (`summary.md:274-275`).
- Never run: hidden 4096, width 1024, the batched decode path, a block without a shared expert (`docs/PHASE5-ORNITH-GAPMAP.md:111-114`).
- There is no `executor.h` adapter for this executor. `catalog/flerp/executor/` holds `strixite_abi`, `strixfold_abi`, `fake_abi`, `conformance`; the MoE recipe's binary is `tool.flerp.qwen35moe_generate`.

**Collectives [R]**

| code | what it is | ranks | graph capture | language, status |
|---|---|---|---|---|
| `runtime.strixfold.coll` + `coll_tcp` + `coll_rdma` + `kernel.strixfold.coll_proxy` | rank-ordered all-gather of byte blocks (`S:coll/.../coll.zig:1-15`, `:100-103`); slot protocol in a coherent slab | 2 | yes: "the sequence number lives in the slab and the stage kernel advances it, so graph replays run fresh gathers" (`S:coll_tcp/.../tcp_proxy.zig:9-10`) | Zig, closure `zig:families/dsv41`; status `builds`; served on the pair at 30.8 tok/s (`S:coll_rdma/module.toml`, regime) |
| `runtime.gufo.tp_verbs` | two-operand partial exchange (`T:.../communicator.hpp:12-15`) | 2 | no: "Host-driven communication is not graph-safe yet", so TP runs eager (`F:executor.cpp:3289-3293`) | C++, harvested, never compiled here; own notes say unqualified (`G:runtime/tp_verbs/src/RDMA_TEMP_STATE.md`) |
| `TpComm` (upstream, not in the catalogue) | all-reduce in rank order with one rounding, all-gather, reduce-scatter form (`U:runtime/tp_comm.hpp:6-18`, `:32-38`, `:97-102`) | 2 to 4 (`U:runtime/tp_comm.hip:476-478`) | no: exchange numbers are host state (`tp_comm.hpp:120-123`) | C++, model-agnostic |

Two-rank assumptions, for whoever generalises:

- **StrixFold transports**
  - `tcp_proxy.zig:62` (the collect arguments hold one `peer`), `:306` (`.world = 2`), `:423` (`peer = 1 - t.rank`).
  - `rdma_proxy.zig:382` (trailer check `got.rank != 1 - k.rank`), `:555`, `:649`, `:923`.
  - `tcp_coll.zig:125`, `:258`, `:313`, `:464` (`id.world != 2` is `UnsupportedWorld`), `:490`.
  - `tcp_wire.zig:136`, `:285`.
  - `S:coll/.../fan.zig:12-13` (world 4 is mapped to two pairs), `:29`.
  - The device kernels copy one peer block (`catalog/strixfold/kernel/coll_proxy/src/zig/kernels/hip/dsv41/coll.hip:46`, `:86-108`).
  - Not limited to two: `rdma_seq.zig:5` (`max_ranks = 8`), and `Coll.exchange` exists in the vtable but is null on HIP (`coll.zig:104-105`).
- **gufo**
  - `T:models/qwen38_flash_next/kernels/rocm/verbs.cpp:518`, `T:cli/serve/tp_control.cpp:744`, `F:device_model.cpp:901`.
  - The sum is "local plus peer, commutative" (`communicator.hpp:12-15`), which has no fixed order for four operands.
  - Its Flash-Next loader also refuses Q6_K expert stacks (`F:device_model.cpp:880-896`), and Ornith has 91 Q6_K tensors.

**Calibration from others' measurements [R]**

- Upstream strixite TP-4 on Strix Halo: 96 all-reduces per forward (`U:docs/tp-rdma.md:20`).
  - Plain decode 34.0 to 60.2 tok/s on four nodes, 1.77x (`:167`).
  - Prefill at 64k context 46 s to 37 s (`:138`).
  - Perplexity +0.15 % against one node (`:151`), so it is not world-invariant.
- StrixFold prefill is kernel-bound and its gathers are hidden by overlap (`docs/PREFILL-DSV41.md:126-136`).
  - An experts pass has a fixed cost of about 15 ms whatever the row count (`:138-143`).
  - Gathers above one 4,352 KiB slot leave RDMA for host-staged TCP (`:42-47`).

## 2. The splits compared (Ornith, four ranks)

**Sizes [I] from the header facts at `docs/PHASE5-ORNITH-GAPMAP.md:11-24`**

- One expert matrix is 1024 x 4096 weights: 2.25 MiB in Q4_K, 3.28 MiB in Q6_K.
- A 512-expert stack is 1.125 GiB (Q4_K) or 1.64 GiB (Q6_K).
- A block's three stacks are 3.4 to 3.9 GiB; sixty blocks are 202.5 GiB if all Q4_K.
- Dense weights are about 10.5 B of the 397 B, so about 7 to 10 GiB.
  - That leaves about 218 GiB of experts and implies roughly 30 Q6_K `ffn_down_exps` stacks.
  - **[N]** The tree records only type counts, not which stacks are Q6_K. Stage P0 reads it.
- A decode token reads, per block:
  - about 68 MiB dense: DeltaNet `attn_qkv` 4096 to 12288, `attn_gate`, `ssm_out`, the shared expert; or attention `attn_q` 4096 to 16384 and `attn_output`;
  - about 72 MiB of routed experts: 10 x 3 matrices.
- Per token that is 60 x 140 MiB plus a 0.8 GiB LM head: **about 9 GiB**.
- The 35B run moves its roughly 1.8 GB a token at 56 tok/s, on the order of 100 GB/s, half the 208 to 226 GB/s read peak (`SPEC.md:40-41`).
  - **[I]** So Ornith on one stream is about 10 tok/s (range 7 to 20) whenever all sixty blocks are on the critical path.
- In a prompt chunk dense and routed arithmetic are equal per token: about 125 M against 126 M weights per block.

| | (a) layer stages | (b) experts placed, dense on rank 0 | (c) TP inside each expert, dense replicated | (d) experts placed, dense replicated |
|---|---|---|---|---|
| weights per node | 55 to 58 GiB (file / 4, cut by bytes) | rank 0: 9 + 54.6 = 64 GiB; others 54.6 | 54.6 + 9 = 64 GiB | 64 GiB |
| with arena, KV, scratch (about 2 GiB MoE scratch at 4,096 rows, `GAPMAP:103-105`) | about 60 GiB | about 67 / 57 | about 67 | about 67 |
| fits three nodes / two | yes (76 GiB) / no (114) | yes / no | (c) needs width / world to be whole blocks: 3 does not divide | yes (82 GiB) / no |
| collectives per block | 0 | 2 (scatter x and ids, gather rows) | 1 | 1 |
| per decode token | 3 hops + 1 reply | 120 | 60 | 60 |
| wire per decode token | 3 x 16 KiB hidden + a token id or candidates | 60 x (48 KiB out + up to 160 KiB back) = 12 MiB | 60 x 16 KiB partial to 3 peers: 2.9 MiB of egress per rank | fixed blocks of k rows: 60 x 160 KiB x 3 peers = 28.8 MiB of egress per rank |
| wire per 1,024-row chunk | 3 x 16 MiB | 60 x 208 MiB = 12.2 GiB | 60 x 16 MiB x 3 = 2.9 GiB of egress per rank | about 7.2 GiB of egress per rank (40 MiB to each peer per block); about 2.9x less with a reduce-scatter by token slice |
| critical path, decode bytes per token | all 9 GiB, serially; hops under 1 % | dense 4 GiB + largest rank's experts (about 0.43 of 4.2 GiB) + head = 6.6 GiB, + 60 round trips | 4 + 1.05 + 0.8 = 5.9 GiB, + 60 gathers | 6.6 GiB, + 60 gathers |
| decode, one stream | 1x (about 10 tok/s) | at most 1.36x | at most 1.5x | at most 1.36x |
| prompt pass | 4N / (N + 3) for N chunks in flight: 1x at one chunk, 2.3x at four, 3.4x at sixteen | dense serial on rank 0: at most 1.6x | at most 1.6x | at most 1.6x |
| several conversations | up to 4 in flight, about 4x aggregate | no | no | no |
| bit identity with one node | **exact by construction**: bytes move, the same kernels run in the same order | exact **if** raw expert rows are gathered and the unchanged `MoeEpilogue` sums them in slot order | **no**, see below | as (b) |
| invariant to the number of ranks | yes, any cut, any world | yes | only after the single-node kernel is changed | yes |
| decode graph capture | unchanged, one graph per stage | broken: host round trip in every block | needs a capturable four-rank gather | needs a capturable four-rank gather |
| state | each stage holds KV and DeltaNet state of its own blocks | all on rank 0 | replicated on all ranks | replicated on all ranks |
| new distributed code | a framed relay and a mirrored call list | relay + slot ownership + a round trip in the block | loader slices + collective + a new row kernel | loader slices + collective + ownership mask |

The speed rows are **[I]**: they count bytes on the critical path and ignore collective latency. The upstream numbers suggest under 0.17 ms per exchange all-in (96 exchanges in a 16.6 ms token), so 60 exchanges would take up to about 10 ms off a 70 ms token.

**Why (c) is not bit-identical [R]**

- The Q4_K row dot gives each of 32 lanes every 32nd sixteen-value sub-block over the whole row, then reduces the lanes (`G:kernel/qwen_ops/src/src/families/qwen/hip_common/quant_ops.hpp:787-804`).
- For a 1,024-wide `ffn_down_exps` row, lane l adds sub-blocks l and l + 32, columns from two different 256-blocks. A 256-column share sums in another order.
- It can be made invariant: the reader's own `warp_id / num_warps` arguments cut a row by whole blocks (`quant_ops.hpp:723-726`).
  - Define the row as four block partials added in block order and compute it that way on one node too.
  - That is a new numeric contract for the single-node executor, and its gates run again.
- The down share is not contiguous in the file: one 144-byte block of every row. Each rank would read every down stack to keep a quarter (gufo's `CopyColumns`, `F:device_model.cpp:224-242`).
- 1024 / 4 = 256 is the only reason it fits at all (`T:.../tp_partition.hpp:45-53`).

**Why (a) for speed too**

- For Ornith half the decode bytes are dense, which (b) and (d) do not parallelise.
- Upstream's full TP-4, which splits everything, reached only 1.77x decode and 1.24x prefill.
- A stage pipeline gets its gain from overlap between chunks and between conversations, with a 16 MiB hop per chunk and no collective in a block.
- What (a) does not do is speed up one short chat. M1 measures whether (d) is worth building for that.

## 3. The chosen design

### 3.1 Placement rule

- A placement is a pure function of (GGUF header, world, mode), computed on every rank and compared by hash in the handshake.
- Mode `stage`: rank r holds blocks `[b_r, b_{r+1})` with all their tensors.
  - Cuts are chosen from the header's tensor byte sizes to minimise the largest stage. Blocks differ by up to 0.5 GiB with a Q6_K down stack.
  - `token_embd` goes on rank 0; `output_norm` and `output` on the last rank.
  - `blk.60.nextn.*` stays unloaded, as today (`catalog/flerp/executor/qwen35moe/module.toml`, notes).
- World 1 is the whole model.
- Mode `expert` (the overlay only): every rank holds all dense tensors and expert e of every stack where `e mod world == rank`, compacted to local index `e / world`.

### 3.2 Per-rank loader (`format.flerp.gguf_placed`)

1. **Parse.** Open the GGUF on the rail NFS with the origin's reader. Mapping a 227.5 GiB file reads only the pages touched, and the header parse touches the header **[I]**. `QwenModelWeights::LoadFromGguf` validates every tensor of the whole model from the header as now (`Q:weights.cpp:198-291`, `:419-431`), on every rank.
2. **Drop.** Clear the tensor refs the placement does not hold.
3. **Allocate.** In place of `CreateWeightRegions` (`Q:hip/model_loader.cpp:190-271`): one `hipMalloc` of the placement's byte total, and one `QwenGpuWeightRegion {host_data = the tensor's mapped address, device_data, size}` per held tensor.
4. **Fill.** `WeightUpload::CopyMapped(tensor.data, bytes, device)`. The mapping's data pages are never read; host residency is the staging buffers.
5. **Remap.** `RemapTensor` (`:273-296`) works unchanged: it searches regions by host address.
6. **Expert mode.** One read per owned expert, at `base + e * ExpertBytes` (`K:moe_kquant.hpp:30-33`) into `dst + (e / world) * ExpertBytes`. That is 23,040 reads of 2 to 3 MiB per rank. The loader writes the stack's device pointer itself.

Rules and unknowns:

- A rank reports the bytes it allocated, and the gate compares that with the plan.
- **[N]** Unbuffered reads over NFS, the read rate, and whether a 57 GiB `hipMalloc` succeeds on a worker are not measured. P1 and P4 measure them.
- Model identity in the handshake is the header's hash plus the file size, never a hash of the data region.

### 3.3 Wire protocol (`runtime.flerp.stage_relay`)

**Why no existing collective carries it**

- A hop is a point-to-point send of `[rows, 4096]` FP32.
- StrixFold's point-to-point `exchange` is null on HIP.
- Its transports are Zig files of the dsv41 closure. SPEC's `c_link` links C++ into a Zig build, not the reverse (`SPEC.md:392-395`).
- gufo's and upstream's communicators are all-to-all sums.

So this is a new C++ module, small, with the contract taken from `coll.zig:7-15`:

- every rank issues the same call sequence;
- every frame is checked;
- the first failure poisons;
- abort is idempotent.

**Links.** TCP on the rail addresses: a chain 0 to 1 to 2 to 3 for data, a reply link from the last rank to rank 0, and a control link from rank 0 to every rank. Rank r listens on `port + r`, so ranks can share a host for the gates (the upstream trick, `U:runtime/tp_comm.hpp:23-25`).

**Frame.** `{magic, epoch, ordinal, op, seq slot, start_pos, rows, bytes, plan hash}` plus payload. The receiver refuses a wrong epoch or ordinal, or a byte count that is not `rows x hidden x 4`.

**Ops.**

- Data: `PREFILL_CHUNK`, `DECODE`, `VERIFY`.
- Control, mirrored from rank 0 before rank 0 acts (the pattern of `U:runtime/tp_mirror.hpp:3-10`): `KEEP`, `KEEP_PREFIX`, `DROP`, `RESET`, `SNAPSHOT_SAVE / RESTORE / DROP`, `CLOSE`.
- The last rank replies with what the ABI call asked for: token ids (device argmax, `Q:hip/decode_step.cpp:277-279`), candidates, or full logits rows of 0.95 MiB each.

**Sizes.** 16 KiB per decode hop; 16 MiB per 1,024-row chunk hop. TCP on 100 GbE is enough: a chunk takes seconds of kernels per stage **[I]**. RDMA is not needed for this split.

### 3.4 Ordering rules for bit-reproducible sums

- **Stage mode:** there is no cross-rank sum. A stage computes `hidden_out` from `hidden_in` with the single-node kernels in the single-node order; the wire carries the FP32 bytes.
  - Each stage keeps `config.num_layers` and the batch sizes of the whole model. Kernel selection reads them (`Q:hip/prefill_chunk.cpp:17-31`) and must not see the cut.
  - Chunk boundaries are the single-node ones (`G:executor/qwen/src/src/models/qwen/hip/prefill.cpp:71-75`).
- **Expert overlay:**
  - The router runs on every rank on identical inputs.
  - A (token, slot) row is computed whole, on the rank that owns its expert, by the same row kernel.
  - Ranks gather raw rows, never partial sums.
  - Every rank runs the unchanged `MoeEpilogue` over slots 0..k-1 from zero (`kernels.hip.cpp:3551-3556`).
  - The result is the single-node bits at any world and any placement.
  - A partial sum per rank (smaller wire) is refused: it changes the order with the placement.

### 3.5 Decode graph capture

**Stage mode.**

- `ExecuteDecodeStep` takes a block range.
- Rank 0 runs the embedding (`decode_step.cpp:67-69`) and its blocks; the last rank runs the output norm, head and argmax (`:265-279`).
- A later stage skips the embedding. Its input row is copied into `decode_scratch.hidden` before the graph launch, like the token and position words the graph already reads through device pointers (`:62-63`).
- The graph's workload identity is folded per layer (`G:.../hip/decode.cpp:60-63`): fold only the owned blocks and add the range.
- Each stage then captures and replays its own graph through the origin's `HipGraphDecodeExecutor`.
- Per token there are four stream synchronisations and four 16 KiB host copies in place of one: well under 1 % of a roughly 100 ms token **[I]**.

**Overlay.**

- Only a gather whose sequence lives on the device survives capture (StrixFold's slab protocol).
- gufo's own answer was to run eagerly.
- So the overlay needs `runtime.flerp.coll_slots`: the rdma_proxy protocol in C++ with world - 1 peers per slot.
- Until it exists, an eager overlay is only good for M1.

### 3.6 Prompt chunks: experts grouped by expert (`kernel.flerp.moe_grouped_kquant`)

1. After `RouterTopK`, on the device: counts per expert (gufo's `ExpertCounts`, declared at `kernels.hpp:576-579` of `kernel.gufo.qfn_kernels`), prefix offsets, and a stable counting sort of (token, slot) pairs by expert.
2. Per expert, tiles of up to R rows: each weight sub-block is decoded once and applied to the R grouped activation rows, for gate/up with SwiGLU and for down.
3. Scatter to `expert_out[t, s]`, then the unchanged epilogue.

**Numeric rule.**

- Per (token, slot, output row), keep the lane and sub-block order of `QuantWarpBlockDot`.
- Then grouped equals ungrouped bit for bit, and the interface gate asks exactly that against `IndexedExperts`.
- If speed needs another order, it becomes a tagged interface and is measured against the full-precision reference (`docs/FIDELITY.md`).
- The rule holds for one node and four alike, since no row is split across ranks.

**Chunk size.**

- A pass touches up to every expert of a block once: about 3.4 to 3.9 GiB read per block per pass **[I]**. This is the fixed-cost-per-pass lesson of `docs/PREFILL-DSV41.md:138-143`.
- Chunks therefore stay large: 1,024 to 4,096 rows, the arena's cap (`G:executor/qwen/src/src/models/qwen/hip/arena.cpp:21`).
- They are not cut small to feed the pipeline. Measure the curve before fixing the size.

**Pipelining.** A stage sends chunk n and starts chunk n + 1 at once, so the one hop per chunk is hidden behind kernels. Each stage needs only its own blocks' state from chunk n.

**Overlay.** The sort takes only pairs with `e mod world == rank`; the row gather for large chunks goes by token slice (reduce-scatter, as `U:runtime/tp_comm.hpp:32-38`).

### 3.7 KV and DeltaNet state

- **Stage mode:** a stage's arena holds the attention K/V planes and the recurrent state of its own blocks (`decode_step.cpp:116-129`, `:197-208`).
  - First version: allocate the origin's whole-model arena on every stage and use the owned part. That is about 31 KiB per position for all 15 attention blocks, 1 GiB at 32K context **[I]**, and it needs no arena change.
  - Snapshots are per stage under the mirrored calls; an exported state is the ranks' states in rank order.
- **Overlay:** identical full state on every rank.

### 3.8 Failure

- **Detection:** a socket reset, a frame check, or no progress within a deadline. The rank poisons its relay and closes its links, so every rank's blocked call returns.
- **Effect:** rank 0's ABI call fails, `flerp_failed` is true, and every seq is invalid. A dead stage's KV and DeltaNet state is gone.
- **Recovery is a restart, not a failover.** The launcher starts the survivors with world 3 (76 GiB a stage fits **[I]**) and conversations run their prompts again. The epoch in every frame keeps a stale rank out.
- Nothing more is claimed for the first version.
- Note for later: in the overlay every rank holds the full state, so three survivors could load the lost quarter of experts (about +18 GiB each) and continue.

### 3.9 Through the executor ABI

- New `executor.flerp.qwen35moe_abi`: a C++ shim over `QwenGpuExecutor`, modelled on `executor.flerp.strixite_abi`.
- `flerp_open` keys: `weights`, `context`, `world`, `rank`, `peers`, `placement`.
  - Rank 0 opens the relay and waits for the stages, as StrixFold's adapter links rank 1 at open (`docs/EXECUTOR-ABI.md:114-115`).
- Ranks 1..3 run `tool.flerp.qwen35moe_stage`, a loop that executes rank 0's calls.
- The front end sees one executor. World 1 is the same code with no links.
- Caps, first version:
  - one seq;
  - `FLERP_OUT_TOKENS` (greedy) and `FLERP_OUT_FULL`;
  - `max_forward` is the arena's chunk;
  - no drafts and no snapshots;
  - `FLERP_CAP_FORWARD_ROWS` only once the batched path has run.
- Text: `runtime.flerp.text_gufo`, a shim over gufo's tokenizer and template check, which the loader already runs (`Q:hip/model_loader.cpp:337-350`).

## 4. Qwen3-Coder-480B-A35B

All architecture facts here are **[N]**. The only source in the tree is `docs/PHASE5-GLM52-GAPMAP.md:363-366`, itself marked unverified:

- plain Qwen3 MoE, 62 layers, hidden 6144;
- 160 experts, 8 used;
- no shared expert, no MTP head;
- softmax attention with RoPE in every block.

I further assume an expert width near 2560 (480 B / (62 x 160 x 3 x 6144) is about 2,600) and an architecture string other than `qwen35moe`.

- **The split serves it unchanged.** Stage placement needs only tensor sizes. 270 GiB / 4 = 67.5 GiB a stage; three stages are 90 GiB, which is tight; two do not fit.
- **The executor does not serve it.** `executor.flerp.qwen35moe` is gufo's Qwen 3.5 engine.
  - Its structure check requires `full_attention_interval == 4` and the DeltaNet sizes (`G:kernel/core_headers/src/src/core/model_config.hpp:68-79`).
  - Its loader requires DeltaNet tensors on three of four blocks (`Q:weights.cpp:419-431`).
  - Its attention reads a doubled `attn_q` with a gate (`Q:hip/decode_step.cpp:47`, `:104-107`).
  - Coder needs an `executor.flerp.qwen3moe`: plain attention in every block, with the same MoE feed forward, loader, relay and adapter.
  - `LaunchMoeFfn` has the no-shared-expert path (`M:moe_ffn.cpp:48-51`); it has never run.
  - The router limits (1024 experts, 32 used; `Q:weights.cpp:212-219`) hold.
  - An expert row must be whole K-quant blocks: 6144 is; a width of 2560 is.
- **What changes in the comparison.**
  - (c) fails at four ranks if the width is 2560: 640 is not a multiple of 256.
  - With 8 x 3 matrices of about 8.4 MiB against perhaps 55 MiB dense per block, experts would be about 80 % of the decode bytes. (d) would then be worth up to about 1.75x, more than on Ornith **[I]**.
  - KV exists for 62 blocks instead of 15.
- So the overlay decision is per model, and M1 should be repeated on Coder's shapes once the header is on hand.

## 5. Modules (SPEC terms)

| module | kind | provides | requires |
|---|---|---|---|
| `format.flerp.gguf_placed` | format | `format.device-view/gguf-kquant/-.placed`: placement plan, per-tensor and per-expert device regions, plan hash | `format.gufo.gguf_kquant`, `runtime.gufo.hip_host`, `kernel.gufo.core_headers`, `runtime.gufo.platform` |
| `executor.flerp.qwen35moe` (extended, same derived module) | executor | `executor.model/qwen35moe/gguf` with a block range. More adapted copies: `hip/decode.cpp`, `hip/prefill.cpp`, `hip/executor.hpp` for the stage input and output; `model_loader.cpp` calls the placed loader | as now + `format.device-view/gguf-kquant/-.placed` |
| `runtime.flerp.stage_relay` | runtime | `transport.stage-relay/-/f32` (framed hidden rows, bytes only), `control.mirror/flerp-v1/-` (mirrored calls, poison, epoch) | none of the catalogue |
| `executor.flerp.qwen35moe_abi` | executor | `executor.abi/flerp-v1/-` | `executor.flerp.qwen35moe`, `runtime.flerp.stage_relay` |
| `tool.flerp.qwen35moe_stage` | tool | the binary of ranks 1..N-1; `--loopback` runs N stages in one process | the adapter's closure |
| `runtime.flerp.text_gufo` | runtime | `text.model/flerp-v1/-` | `executor.gufo.qwen` (tokenizer, template) |
| `kernel.flerp.moe_grouped_kquant` | kernel | `moe.experts-prefill/gguf-kquant/f32.indexed-order` (bits of `IndexedExperts`) | `kernel.gufo.qwen_ops`, `kernel.gufo.core_headers` |
| later, overlay only: `runtime.flerp.coll_slots`, `kernel.flerp.coll_proxy_n` | runtime, kernel | `collective.allgather-rdma/-/f32.world-n`, `collective.device-proxy/-/-.world-n` | derived from `runtime.strixfold.coll_rdma`, `kernel.strixfold.coll_proxy` |

**Recipes**

- `qwen36-35b-a3b.flerp-moe-stage` (the gates);
- `ornith15-397b.flerp-stage4`;
- both bind `serve.flerp.http`, `runtime.flerp.decode_loop`, `runtime.flerp.sampler_none` or `sampler_strixite`.

**SPEC changes in the same commits**

- `[deploy] topology` gains `stage2 | stage3 | stage4` (`SPEC.md:350`).
- Section 10 gains the rule for a four-worker window: today only the F and A pair is covered (`SPEC.md:502-504`).
- Section 9 gains the stage gates.

**Gate code**

- Runner `gate/moe_stage.py`; criteria in `gate/qwen36-35b-a3b.flerp-moe-stage.toml` and `gate/ornith15-397b.flerp-stage4.toml`, committed before the first run (`SPEC.md:446-447`).

## 6. Staged plan

Agent-days exclude waiting for hardware. "Control" is control-node's GPU under the `control-gpu` lease rules (`SPEC.md:505-508`).

| # | stage | gate (criteria written first) | device memory | agent-days | pod time |
|---|---|---|---|---|---|
| P0 | Header census of the Ornith file: every tensor's type, bytes and offset; the placement tables for world 3 and 4 | tables committed; largest stage under budget; resolves the [N] items of section 2 | none | 0.5 | none (header read on control-node) |
| R | Reference for the 397B file: llama.cpp on the CPU with the file mapped, on one worker whose model is unloaded. Teacher-forced logits for three prompts under 32 tokens, and `l_out` / `ffn_moe_topk` for sampled blocks | two runs equal; files committed with hashes. If it cannot be produced, say so and hold P5 to gates (ii) and (iii) only | host RAM as page cache; about 9 GiB dense resident, about 9 GiB touched per token **[I]** | 1 to 2 | one worker, CPU only, hours |
| P1 | Placed loader, world 1 | 35B: generated ids and copied logits bit-equal to today's binary on the four gate prompts; device bytes equal the plan. Then a block range of Ornith (as many blocks as fit): loads from NFS, the decode step and a 1,024-row chunk run at hidden 4096 / width 1024, `gate/moe_block_check.py` on three blocks; read rate and ms per block recorded | 20 GiB; then up to control's free memory | 2 to 3 | control, holds of minutes |
| P2 | Stages and relay on one GPU: N stages in one process (loopback), then N processes over TCP on localhost | **world invariance on 35B**: world 1, 2, 4, even and lopsided cuts, graph on and off. sha256 of the prompt-pass logits and of 128 decode steps' logits equal to P1's, including the 1,808-token prompt | about 22 GiB + N arenas | 3 to 4 | control, holds of about 10 min |
| P3 | ABI adapter, stage tool, text shim, front end | `executor.flerp.conformance` (`plain`; `loop` without drafts) and `serve.flerp.http` greedy chat on 35B at world 1 and 2-on-one-GPU: tokens equal P1's | as P2 | 3 to 4 | control |
| P4 | Two workers | 35B at world 2 over the rail from NFS: the P2 hashes again (bit equality across two machines is an assumption until here). Stages 0 and 1 of the Ornith world-4 plan: load time, ms per block for decode and per chunk, block check. **M1**: one all-gather of 160 KiB between the pair, eager, against the measured expert time per block | 2 x about 12 GiB; then 2 x about 60 GiB | 1 to 2 | the pair under `pod-quad hold`, 1 to 2 h |
| P5 | **Four workers, first tokens** | (i) two runs equal; (ii) **world 4 equals world 3** on three of the nodes, logits hashes (the only whole-model invariance check that fits); (iii) block check on sampled blocks of every stage; (iv) against R, the `[fidelity]` metrics at the recorded positions; tok/s and TTFT recorded | 4 x about 60 GiB; 3 x about 78 GiB | 1 to 2 | four workers, 2 to 3 h (load time **[N]**) |
| P6 | Speed 1: grouped experts; chunk pipelining; several seqs in flight; the batched decode path run for the first time | interface gate: grouped bits equal `IndexedExperts`, GB/s. P2's hashes unchanged at world 1, 2, 4. Then numbers on the pair, then one four-node window for numbers only | as P2; then as P5 | 5 to 8 | control; pair 1 h; four workers 1 to 2 h |
| P7 | Optional overlay (d): expert placement, `coll_slots`, masks | go only if M1 shows 0.57 x expert time per block is over twice the gather time. Gate: world invariance on 35B with experts split artificially over 1, 2, 4 ranks on one GPU, hashes equal to P1; then the pair; then four | 4 x about 67 GiB | 8 to 12 | control; pair; four workers |

First tokens are P0 to P5 plus R: about 12 to 18 agent-days and one four-worker window. Every stage before P5 runs on one GPU or on the pair.

## 7. The single riskiest unknown

**Nothing on the pod can say what the right logits of the 397B file are.**

- Every gate before P5 proves that four nodes compute what one node would. None proves that one node would be right at hidden 4096, 32 heads, width 1024 and 512 experts.
- The 35B fidelity result does not transfer by itself. Its own record says the attention and DeltaNet path was never checked independently on a MoE file (`docs/FIDELITY.md:67`).
- A forward that is wrong in a plausible way passes world invariance.
- Stage R is the attempt: llama.cpp over the mapped file on one worker's CPU. Whether a 227.5 GiB mapping from NFS is usable at all on a 128 GB machine is untested **[N]**.
- If R fails, the honest claim after P5 is: tokens, invariance between three and four stages, and block-level agreement with an independent dequantization.

Second: the per-token cost at Ornith's shapes, about 10 tok/s by my byte count. It decides whether "useful" needs P7; P1 measures it on one GPU.

## 8. Not verified here

- Worker free GPU memory (taken as 110 GiB).
- control-node's free GPU memory.
- NFS read rate and unbuffered reads on the rail.
- Whether several processes can share control-node's GPU for P2. The in-process loopback avoids the question.
- Bit equality of the kernels across two physical nodes.
- Which Ornith tensors are Q6_K.
- What `blk.60` holds beyond the four `nextn` tensors.
- Every Qwen3-Coder fact.
- That `WeightUpload::CopyMapped` behaves over NFS as on a local disk.
- No code was compiled for this document.

### Critical Files for Implementation

- /home/user/Code/strixflerp/catalog/flerp/executor/qwen35moe/src/flerp/qwen35moe/src/models/qwen/hip/model_loader.cpp
- /home/user/Code/strixflerp/catalog/flerp/executor/qwen35moe/src/flerp/qwen35moe/src/models/qwen/hip/decode_step.cpp
- /home/user/Code/strixflerp/catalog/flerp/executor/qwen35moe/src/flerp/qwen35moe/src/models/qwen/hip/prefill_chunk.cpp
- /home/user/Code/strixflerp/catalog/gufo/kernel/core_headers/src/src/core/hip/weight_upload.hpp
- /home/user/Code/strixflerp/catalog/flerp/executor/qwen35moe/src/flerp/qwen35moe/moe_ffn.cpp

## Staging Ornith with loom-swarm

Added 2026-11-12. Details and byte checks: gate/results/ornith15-397b.swarm/README.md.

The origin on control-node (`loom-swarm-origin`, *:7150, rail 198.51.100.31) serves
`/data/models/ornith-1.5-397b/`: a symlink to the GGUF on the read-only USB NVMe nv1, the full manifest, and
one layout artifact per world-4 rank, `Ornith-1.5-397B-Q4_K_M.gguf.r<r>of4.swarm.json`. Each layout holds the
header plus that rank's gguf_placed tensors (cuts 0/15/30/45/60, 58.4-62.0 GB). Workers pull from the rail origin first, then from
peers. The ranks do not overlap, so peers only share the header, and every tensor byte is read once from nv1.
**nv1 is the bottleneck, not the rail.** nv1 read 126 MB/s with one O_DIRECT reader, and 68 MB/s while hashing under load. All four ranks therefore take **31-55 min** to seed together (233 GB), against 13-14 min a rank for the P4 NFS loads (`ornith15-397b.p4`). Loading from /srv/swarm or /cache afterwards is local.

Run the commands on control-node from /data/pod/tools, in the hold that covers A B C D. M=/data/models/ornith-1.5-397b/Ornith-1.5-397B-Q4_K_M.gguf.

    HOSTS=(<rank0 host> <rank1 host> <rank2 host> <rank3 host>)   # 192.0.2.x of the hold's four workers
    # 1. seed: all four at once (they share nv1's rate; starting them in sequence gains nothing)
    for r in 0 1 2 3; do h=${HOSTS[$r]}; SWARM_GB=70 ./swarm.sh seed $h $M@$r/4 & done; wait
    # 2. wait (timeout sized to the measured rate, with margin)
    for r in 0 1 2 3; do ./swarm.sh wait ${HOSTS[$r]} 4500; done
    # 3. load: each rank from its local path. That is /srv/swarm/Ornith-1.5-397B-Q4_K_M.gguf, a sparse full-size
    #    file with only the rank's bytes present; on a /cache node it is a symlink to /cache/models/ornith-1.5-397b/...
    #    The binary takes the P4 flags with --model "$(./swarm.sh path $M)" --placed --world 4 --rank r.
    #    Its loader reads only the plan's ranges, so the holes are never touched.
    # 4. purge after the run (RAM-disk nodes at once after `loaded` if memory is tight)
    for r in 0 1 2 3; do ./swarm.sh purge ${HOSTS[$r]}; done

Memory: on a RAM-disk node the tmpfs copy (up to 57.8 GiB) sits beside the GPU copy of the same bytes. With the
5 GiB arena that is about 121 GiB of 124. Prefer nodes with free /cache for ranks 0 and 3. F and D /cache hold
StrixFold's Engram tables, so do not let `seed` put Ornith there: give those nodes no /cache path, or put them on
ranks 1 and 2 and purge as soon as `loaded` is printed. swarm.sh picks /cache whenever it is mounted and writable.
That needs checking on F and D before the window. A /cache copy persists, so later windows load locally with no pull.

## P5 runner: loom-swarm straight into the GPU

Added 2026-11-12, after the section above. Code: `format.flerp.gguf_placed` (placed_loader.hpp `FillPolicy::swarm`,
`data_path`, `punch_behind`), loom-swarm's `stream` verb (~/Code/loom-swarm cd2cc13), `gate/ornith_p5.py`,
criteria `gate/ornith15-397b.flerp-p5.toml`.

- **Default source: no file on the worker.** The four world-4 ranks want disjoint bytes, so peers add nothing and a
  RAM-disk copy only costs memory from the same 128 GB pool. The loader starts `loom-swarm stream --manifest M
  --from 198.51.100.31:7150 --ranges -`, hands it the plan's (offset, length) ranges, and reads verified 16 MiB pieces
  from a pipe as frames {u64 offset, u64 length, bytes} into its pinned staging buffer. It copies every part that lies
  in a plan entry to the entry's device offset. loom-swarm's own client does the GETs, the BLAKE3 check against the
  full manifest, retries on the next source and `conns` requests in flight. Every entry must be covered exactly once;
  a failed helper is run again for the incomplete entries. Flags: `--swarm-helper/--swarm-manifest/--swarm-from/
  --swarm-conns` (generator), `swarm_*` keys (ABI, stage tool).
- **Fallback (`--source file`).** The same stream is written into a sparse file on /srv/swarm, never /cache on F or D.
  The rank then loads with `--data-file F --punch-behind` (fallocate PUNCH_HOLE behind each copied piece, so the tmpfs
  copy shrinks as the GPU copy grows), and the file is deleted after `loaded`. The runner checks MemAvailable before
  the fetch.
- **Runner.** `--plan` is read-only: memory, GTT, /cache, RAM disk, `ip -o route get 198.51.100.31`, serving processes,
  holds, the origin's HAVE for the manifest root, tokenizer identity, window estimate and stage R. `run` takes holds
  as "agent FLERP-P5" (DRAIN on retry, never FORCE). It refuses a node whose serving is up, loads one rank
  at a time (nv1 serves one reader best), and runs w4 (two runs plus dumps) → block check → w3 → serve chat. On any
  exit it stops every unit and checks none is left, and only then releases the holds.
