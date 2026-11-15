# What the header resolves of docs/PHASE5-FOURNODE-DESIGN.md (stage P0, 2026-11-11)

`summary.md` and `census.json` beside this file are generated (`gate/gguf_census.py`, header only: 11,013,904 bytes of
the 227.53 GiB file were read). This file is written by hand from them.

## [N] items the header resolves

| item (design section) | answer |
|---|---|
| Which expert stacks are Q6_K (2, 8) | 30 `ffn_down_exps` stacks: blocks 0-6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 53-59, and block 60 (not loaded). 29 of the 60 running blocks. Every `ffn_gate_exps` and `ffn_up_exps` is Q4_K. The design's "roughly 30" holds. |
| The other Q6_K tensors | 61: `ffn_down_shexp` of the same 30 blocks, `attn_qkv` of 22 DeltaNet blocks, `attn_v` of 8 attention blocks, `output.weight`. |
| What `blk.60` holds beyond the four `nextn` tensors (8) | A whole attention block with its own MoE feed forward: 20 tensors, 3.98 GiB (three expert stacks 3.89 GiB, `attn_q/k/v/output`, norms, router, shared expert, and the four `nextn` tensors). So "nextn unloaded" leaves out 3.98 GiB, not four small tensors, and an MTP head for this file is a 61st block's worth of device memory on whichever rank runs it. |
| Dense bytes (2: "about 7 to 10 GiB") | 4.78 GiB in the 60 blocks, + 0.53 `token_embd` + 0.78 head = 6.09 GiB. Experts 217.45 GiB (design: about 218). |
| A block's expert stacks (2: "3.4 to 3.9 GiB") | 3.375 GiB (all Q4_K) or 3.891 GiB (Q6_K down). A block's dense part is 0.069 to 0.090 GiB. |
| Bytes a decode token reads (2: "about 9 GiB") | Dense 4.78 GiB + head 0.78 GiB + routed experts 10 x (2 x 2.25 + 2.25 or 3.28 MiB) = 67.5 MiB (31 blocks) or 77.8 MiB (29 blocks) = 4.25 GiB: **9.8 GiB** a token, a count from sizes, not a measurement. |
| Stage sizes (2: "55 to 58 GiB" at four, 76 at three) | World 4: 57.03 / 54.36 / 54.38 / 57.77 GiB, cuts 0, 15, 30, 45, 60. World 3: 75.32 / 72.67 / 75.55 GiB, cuts 0, 20, 40, 60. |
| Largest stage under the budget (P0's gate) | Yes at both worlds against 110 GiB a node: 62.8 GiB at world 4 and 80.6 GiB at world 3 with the arena estimate (5.0 GiB a rank at 32K context and 4,096-row chunks; an estimate from header sizes). |
| Tied head | No: `output.weight` exists (Q6_K), so `token_embd` is needed on rank 0 only. |

## What the header cannot resolve

- Worker free GPU memory (the 110 GiB is taken as given), control-node's free GPU memory.
- Whether a 57 GiB (or 76 GiB) `hipMalloc` succeeds on a worker; the read rate over NFS; unbuffered reads over NFS.
- The arena's real size at Ornith's shapes: the 5.0 GiB is computed from header sizes, not read from an arena.
- Whether the kernels are right at hidden 4096 / width 1024 / 512 experts / Q6_K down stacks: nothing ran for P0.
- Bit equality across two physical nodes; every Qwen3-Coder fact; what the right logits of the file are (design section 7).
