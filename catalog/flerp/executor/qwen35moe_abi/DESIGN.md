# executor.flerp.qwen35moe_abi: design record

What the module is for, and the one mechanism in it that needs more than a paragraph: slot-resident prefix reuse
(ORNITH-PCACHE step 0, card a3656fc5e9, criteria gate/ornith-pcache.toml). [R] = read in the tree at the cited
place, [M] = measured (gate/results), [I] = inferred here.

## Place in the engine

Ornith-1.5 397B is served as four block-range stages (docs/PHASE5-FOURNODE-DESIGN.md section 0). Rank 0 is this
executor inside the OpenAI front end (serve.flerp.http); ranks 1..3 are tool.flerp.qwen35moe_stage processes. Each
stage runs executor.flerp.qwen35moe (gufo's Qwen executor with the MoE feed forward) over its block range, and
runtime.flerp.stage_relay carries FP32 hidden rows down the ring and the last stage's reply back. The front end and
runtime.flerp.decode_loop see one executor through include/flerp/executor.h; nothing above this module knows there
are stages. [R: stages.hpp header, abi.cpp header]

## The problem prefix reuse answers

A serving window has one slot. FreeFitty sends each turn of a thread as a whole new request: system prompt, tool
definitions and the history, 20-60K tokens, of which all but the newest turn equals the previous request's prompt.
Without reuse every turn pays the whole prefill again: 26.5 s for a 19.5K-token thread on the 35B proxy at world 4
[M: gate/results/ornith-pcache-g1024/summary.md]. The per-stage state that a prefix leaves behind is small enough
to keep: attention KV of about 7.7 KB a position a stage, plus a fixed recurrent (DeltaNet) state [I: card
a3656fc5e9 sizes], about 552 MiB a stage measured for the 19.5K-token thread [M].

## Mechanism

1. **One snapshot, on the device, in every stage.** RESET frames carry two new flags (stage_relay kSnapSave,
   kSnapRestore) instead of new ops, so the wire format and its checks are unchanged. On kSnapSave each stage
   synchronizes and takes the executor's own snapshot (gufo QwenGpuSnapshot via Executor::SaveSnapshot, a device to
   device copy of KV up to `positions` and the recurrent state) without resetting; on kSnapRestore it resets, puts
   the snapshot back and re-arms the recurrent ring for the session's draft depth. A save replaces the old one.
   [R: stages.cpp Stage::SnapSave / SnapRestore, ServeOne]
2. **Where the snapshot is taken: the start of the prompt's last forward, when at least `reuse_tail` ids follow.**
   The chat template appends a generation prompt (`<|im_start|>assistant` and the think opener) that the next
   request's rendering of the same thread does not repeat at that place, so the kept prefix must end before it.
   reuse_tail 32 keeps those ids out. [R: abi.cpp prompt_piece]
3. **Why the start of a forward, not "prompt end minus reuse_tail".** The first version (v1) cut the final chunk at
   prompt end minus 32. A warm request then forwarded different chunk boundaries than a cold one, and the greedy ids
   drifted at a near tie (tokens 19 and 23) [M: gate/results/ornith-pcache-v1]. A snapshot at a forward boundary
   that every cold prefill of that prompt also has means a warm run forwards exactly the chunks a cold run forwards
   after that point, and with chunk-invariant prompt kernels (the GQA WMMA attention is bit-identical across cuts,
   PREFILL-ORNITH-2 G4) its ids are the cold run's. [M: warm ids == cold ids, run B]
4. **Why a grid (`reuse_grid`).** With the caller's own 4,096-row chunks the last forward can start up to 4,096
   positions before the end, so the warm request re-prefills that much: 3.7x (run A). With reuse_grid 1024 every
   prompt forward is cut at multiples of 1,024 positions (cold ones too, so both still share boundaries), the kept
   prefix ends within 1,024 ids of the prompt's end, and the warm turn is 10.9x faster (2.4 s against 26.5 s) with
   cold prefill no slower. [M: runs A and B; R: abi.cpp prompt_with_reuse]
5. **Matching.** A new seq with a snapshot of the same session kind (same draft depth) does not reset the stages.
   Its prompt pieces are compared with the snapshot's ids as they arrive and skipped while they match. When the
   prompt runs past the kept prefix, the stages restore the snapshot and only the rest is forwarded. If the ids
   differ, or the prompt ends inside the kept prefix (its last row needs the stages at the prompt's end), the stages
   reset and the skipped pieces are replayed in the chunks they came in, so the request is then exactly a cold one.
   A prompt forwarded from position 0 drops the snapshot, because it overwrites the KV the snapshot stands on.
   [R: abi.cpp seq_new, replay_skipped, prompt_piece]
6. **The draft head.** The last stage's MTP head pairs each prompt row with the next chunk's first id, so a chunk's
   last row waits as the prompt's "edge". kSnapSave also keeps that edge (DraftRound::SaveEdge in
   executor.flerp.qwen35moe), and a restore puts it back (RestoreEdge), so the first draft after a warm restore is
   the one a cold run makes. Drafting after a restore decoded correctly at drafts 1 [M: summary.md].

## Costs and limits

- One snapshot per executor, so one thread at a time benefits; a different thread resets it. [R]
- Device memory: one extra copy of the kept KV and recurrent state per stage (552 MiB a stage at 19.5K tokens) [M].
- The ABI snapshot calls (flerp_snapshot_*) stay FLERP_E_UNSUPPORTED: the snapshot is internal to the executor and
  is not exported, written to a file or shared between slots. The frozen RAM-tier cache on control-node that card
  a3656fc5e9 describes as its later step is not built. [R: abi.cpp]
- Off unless rank 0 is given reuse_tail > 0. configs/ornith397.serve.toml sets reuse_tail 32 and reuse_grid 1024,
  and tools/ornith-engine/engine.py passes them to rank 0. [R]
