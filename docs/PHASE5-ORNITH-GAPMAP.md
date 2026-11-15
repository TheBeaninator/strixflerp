# Phase 5, step 1: what `executor.gufo.qwen` lacks for Ornith-1.5 397B

Read from the catalogue (gufo f070a17, branch rdma) and from the GGUF headers on control-node, 2026-11-11. Paths are
under `catalog/gufo/`; `Q:` is `executor/qwen/src/src/models/qwen/`, `F:` is
`executor/qwen38_flash_next/src/src/models/qwen38_flash_next/`, `T:` is `runtime/tp_verbs/src/src/`.
Nothing here was run against Ornith: its 227.5 GiB file was not loaded. The recipe that this step did assemble and
what its gate showed are in `gate/results/qwen35-decider-2b.gufo/summary.md`.

## The two files

| key | Ornith-1.5-397B-Q4_K_M | decider-2b v12 Q4_K_M (the gate's model) |
|---|---|---|
| `general.architecture` | `qwen35moe` | `qwen35` |
| blocks / hidden / heads / KV heads / head dim | 61 (60 + 1 MTP) / 4096 / 32 / 2 / 256 | 24 / 2048 / 8 / 2 / 256 |
| `full_attention_interval`, `ssm.conv_kernel`, `ssm.state_size`, `ssm.group_count` | 4, 4, 128, 16 | 4, 4, 128, 16 |
| `ssm.inner_size` / `ssm.time_step_rank` | 8192 / 64 | 2048 / 16 |
| feed forward | no `feed_forward_length`; `expert_count` 512, `expert_used_count` 10, `expert_feed_forward_length` 1024, `expert_shared_feed_forward_length` 1024 | `feed_forward_length` 6144 |
| FFN tensors of a block | `ffn_gate_inp` F32 [4096, 512], `ffn_gate_inp_shexp` F32 [4096], `ffn_{gate,up}_exps` Q4_K [4096, 1024, 512], `ffn_down_exps` [1024, 4096, 512], `ffn_{gate,up,down}_shexp` | `ffn_gate`, `ffn_up`, `ffn_down` |
| attention and Gated DeltaNet tensors | the same names and layout as the dense file (`attn_qkv`, `attn_gate`, `ssm_*`; `attn_q/k/v/output` with q and k norms every fourth block) | |
| MTP head | in the file: `blk.60.nextn.{eh_proj,enorm,hnorm,shared_head_norm}` | none |
| tensor types | Q4_K 567, Q6_K 91, F32 460 | Q4_K 162, Q6_K 25, F32 133 |
| LM head | `output.weight` Q6_K | tied to `token_embd` |

The routed experts are 60 x 3 tensors of 512 x 1024 x 4096 weights: 386.5 of the 397 billion parameters.

## What would have to change in `executor.gufo.qwen`

1. **There is no MoE in it: no expert count or used-count limit exists because no expert code exists.** The loader
   asks every block for `ffn_gate.weight`, `ffn_up.weight`, `ffn_down.weight` (`Q:weights.cpp:247-249`) and
   validates them as `intermediate x hidden` matrices (`:266-275`); a missing one is "Missing required Qwen tensor"
   (`:98`). The assembled binary says exactly that for a qwen35moe file
   (`gate/results/qwen35-decider-2b.gufo/moe-probe.txt`: `blk.0.ffn_gate.weight`). The config reader takes
   `feed_forward_length` and no `expert_*` key (`executor/qwen/src/src/families/qwen/config/qwen_config.cpp:28-30`);
   without the key `intermediate_size` keeps its default 9728 (`kernel/core_headers/src/src/core/model_config.hpp:17`)
   and the structural check still passes (`:68-79`), so the failure comes one step later, at the tensors. Every FFN
   call site is one gate/up SwiGLU and one down projection on whole matrices: decode `Q:hip/decode_step.cpp:221`,
   batched decode `Q:hip/batched_decode.cpp:55-98, 362-370, 749-757`, CPU reference `Q:modules/ffn.cpp:19-51`.
   To add: router logits on `ffn_gate_inp` (F32), top-10 of 512 with renormalised weights, the 10 selected experts'
   gate/up/down on 3-D K-quant tensors, the shared expert and its scalar gate (`ffn_gate_inp_shexp`), the routed sum.
2. **The MoE pieces the catalogue has are the Flash-Next engine's, and that engine is locked to its own geometry.**
   Router top-k for up to 512 or 1024 experts (`kernel/qfn_kernels/.../kernels.hip.cpp:9020-9030`; passed the
   `moe-router` interface gate), indexed expert GEMV on K-quants (`kernel.gufo.qfn_mmq`, `.mmid`; compile-gated only).
   `F:config.cpp:249-259` rejects anything but 48 layers, hidden 2560, 512 experts, 10 used, expert width 640 and
   hyper-connections: Ornith is 512 / 10 like it, with width 1024, hidden 4096 and no hyper-connections, so the
   executor cannot be reused, only its kernels, and their shape limits at width 1024 were not checked.
3. **Hard-coded dimensions.** The structural check fixes `full_attention_interval == 4`, `ssm_conv_kernel == 4`,
   `ssm_state_size == 128`, `inner / time_step_rank == 128` (`model_config.hpp:68-79`): Ornith satisfies all four
   (8192 / 64 = 128; 60 % 4 == 0 after the MTP block is taken off, `qwen_config.cpp:72-85`). The 27B-only fast
   paths are guarded by equality tests and fall back for other shapes: prefill chunk
   (`Q:hip/prefill_chunk.cpp:21-23`: hidden 5120, intermediate 17408, 64 layers), batched decode
   (`Q:hip/batched_decode.cpp:64, 114`). Prompt batch is capped at 4096 rows (`Q:hip/arena.cpp:21`), decode batch at 8
   (`Q:hip/batched_decode.cpp:20`). The MTP reference names `blk.64.nextn.*` literally
   (`runtime/mtp/src/src/models/qwen/mtp_reference.cpp:101-105`): Ornith's head is `blk.60.nextn.*`, inside the main
   file, where gufo's Qwen MTP backend loads a separate file.
4. **Memory: the whole file goes to one device.** `CreateWeightRegions` makes one device allocation per mapped
   region of the GGUF, `hipMalloc(source.size)` (`Q:hip/model_loader.cpp:68-73, 192-236`), and tensors are remapped
   into it (`:352-358`): there is no per-tensor or per-rank selection, so Ornith needs 227.5 GiB on one device.
   The arena sizes scratch from `intermediate_size` and the prompt batch (`Q:hip/arena.cpp:337-380`: three
   `batch x intermediate` buffers); for MoE that becomes rows x 10 experts x 1024, and KV plus recurrent state
   grow with hidden 4096 / 32 heads, none of it large beside the weights.
5. **Multi-node: the rdma branch gives none for this executor.** No file under `Q:` or the Qwen family names a
   rank, a partition or a communicator. The TP code in the catalogue (`runtime.gufo.tp_verbs`, harvested, never
   compiled here, never run) serves the Flash-Next engine only:
   - how it splits: **tensor-parallel, every layer on every rank.** The routed experts' intermediate width is cut
     evenly, each rank holding rows `[ff_begin, ff_begin + ff_count)` of every expert's gate and up and those
     columns of down, all ranks route alike and the MoE exchange sums the shares
     (`T:models/qwen38_flash_next/distributed/tp_partition.hpp:13-35`); attention is cut by KV head
     (`F:kernels/rocm/device_model.cpp:525-541`), DeltaNet by key head (`:479-490`), the shared expert by width
     where block boundaries allow (`:544-560`). Not by layers, not by whole experts.
   - how many ranks: **two.** `partition->world_size > 2` is refused (`F:kernels/rocm/device_model.cpp:901`), the
     verbs transport and the control channel refuse `world_size != 2`
     (`T:models/qwen38_flash_next/kernels/rocm/verbs.cpp:518`, `T:cli/serve/tp_control.cpp:744`), and the
     communicator's contract is a two-operand sum (`T:models/qwen38_flash_next/kernels/rocm/communicator.hpp:12-15`).
   - its own state: "not qualified for deployment", with an open HTTP 500 on a continued conversation
     (`RDMA_TEMP_STATE.md` in the origin; not harvested).
   For Ornith two ranks are not enough anyway (half the experts is about 110 GiB a rank before anything else), and
   the attention cut cannot go past two (2 KV heads). What carries over is the idea, and it fits the file: expert
   width 1024 over four ranks is 256 a rank, one K-quant block of `ffn_down_exps` rows, so the shares can be taken
   without re-encoding (the condition `PlanRoutedBytes` checks, `tp_partition.hpp:45-53`); attention, DeltaNet,
   the router and the LM head, 3% of the weights, would be whole on every rank. That needs a partition-aware loader,
   an all-rank sum (StrixFold's `runtime.strixfold.coll_rdma` is a two-node allgather; SGLang's RCCL path on the pod
   runs four) and a control channel for more than two ranks: none of it is in the harvested gufo modules.

## Step 2: the MoE feed forward on one node (2026-11-11)

Item 1 of the list above is written: `executor.flerp.qwen35moe` (seven adapted copies of the executor's files in a
module of their own; the origin's stay byte-identical) with `kernel.flerp.moe_kquant`, in the recipe
`qwen36-35b-a3b.flerp-moe`, run on control-node's GPU on `Qwen3.6-35B-A3B-Q4_K_M.gguf` (qwen35moe, 256 experts,
8 used, widths 512). What its gate showed and did not show is in `gate/results/qwen36-35b-a3b.flerp-moe/summary.md`.
What the path does, with every size read from the header (`weights.cpp`, `LoadMoeBlock`):

- router `ffn_gate_inp` (F32) and the shared gate `ffn_gate_inp_shexp` in one launch; softmax over all experts,
  top `expert_used_count`, weights renormalised over the selection, by gufo's `RouterTopK` (item 2: its limits are
  1024 experts and 32 selected, so Ornith's 512 / 10 fit);
- the selected experts' gate/up with SwiGLU and down, and the shared expert's, by indexed K-quant projections
  that read expert ids on the device (`flerp::moe::IndexedExperts`: gufo's K-quant row reader, any expert width
  that is whole blocks, so 1024 fits); the weighted sum and the gated shared expert by gufo's `MoeEpilogue`;
- the same seven launches a block for a prompt chunk and for a decode step, none decided on the host, so the
  decode step stays in the origin's captured HIP graph.

What that leaves for Ornith on one executor instance, besides the multi-node split (items 4 and 5):

- **Memory.** Unchanged: one device region for the whole file, 227.5 GiB. The MoE scratch is three buffers of
  `prompt batch x max(k * hidden, E + 1 + 2k + hidden, k * F + S)` floats: 40,960 floats a token for Ornith, about
  2 GiB at the arena's 4,096-row cap.
- **The MTP head.** Not loaded here (`nextn_predict_layers` only takes the block off the stack, as in the origin);
  item 3's `blk.64.nextn.*` literal is untouched.
- **Prompt speed.** A prompt chunk reads each token's selected experts once a token (no grouping by expert), so it
  is bound by memory bandwidth at about `tokens x blocks x k x 3` expert matrices: measured on the 35B file in the
  gate. Ornith's experts are four times larger and it has 60 blocks and k = 10.
- **Not exercised.** Expert stacks in a type other than Q4_K and Q6_K (the kernel takes the others by gufo's
  reader; none was run), a block without a shared expert, a merged `ffn_gate_up_exps` tensor (refused by name),
  per-block expert widths (the loader takes them per block; the test file has one width), widths 1024 and
  hidden 4096, and the batched decode and verification paths (`batched_decode.cpp`: adapted, compiled, not run).

## What is in hand

- The dense executor builds from the catalogue and runs a qwen35 file (the recipe gate).
- The GGUF reader, K-quant decode GEMV (Q4_K, Q6_K: Ornith's types) and the Gated DeltaNet, attention, norm and
  RoPE kernels are the ones Ornith's non-FFN path would use unchanged, by tensor name and layout; that is read from
  the headers, not run.
- llama.cpp at the ark's commit knows `qwen35moe` (`src/llama-arch.cpp:42`) and is the reference for a MoE gate on
  a smaller file: `Qwen3.6-35B-A3B-Q4_K_M.gguf` (20 GiB, 256 experts) and `Ornith-1.5-35B-Q8_0.gguf` are on the pod.
