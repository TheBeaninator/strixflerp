# GGUF header census: Ornith-1.5 397B Q4_K_M

`/data/ark/models/ornith-1.5-397b/Ornith-1.5-397B-Q4_K_M.gguf`: 244309803808 bytes (227.53 GiB), GGUF v3. Read: the header only, 11013904 bytes from the start of the file (the data region starts at 11013920); sha256 of the header's 11013904 bytes `bfcea545c4922125b2e00c2c21f43f765e147867488df2b3f455816dff8ce202`. Tool: `gate/gguf_census.py`. Every tensor's name, type, dims, bytes and offset is in `census.json`.

`qwen35moe`: 61 blocks in the header, 1 of them past the stack (`nextn_predict_layers`), so 60 run; hidden 4096, 32 heads, 2 KV heads of 256; 512 experts, 10 used, expert width 1024, shared expert width 1024; vocabulary 248320; context 262144.

## Totals by type

| type | tensors | bytes | GiB |
|---|---|---|---|
| Q4_K | 567 | 189067198464 | 176.08 |
| Q6_K | 91 | 54707896320 | 50.95 |
| F32 | 460 | 523695104 | 0.49 |
| all | 1118 | 244298789888 | 227.52 |

## Where the bytes are

| part | GiB |
|---|---|
| expert stacks of the 60 blocks | 217.45 |
| everything else in those blocks (dense) | 4.78 |
| `token_embd` | 0.53 |
| `output_norm`, `output` | 0.78 |
| blocks past the stack (not loaded) | 3.98 |
| other tensors | 0.00 |

## Q6_K

91 Q6_K tensors, 30 of them expert stacks: `ffn_down_exps` in blocks 0-6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 53-60.
The other Q6_K tensors: `attn_qkv.weight` in blocks 0-2, 4-6, 9, 12, 18, 21, 24, 30, 33, 36, 42, 45, 48, 53-54, 56-58; `attn_v.weight` in blocks 3, 15, 27, 39, 51, 55, 59-60; `ffn_down_shexp.weight` in blocks 0-6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 53-60; `output.weight`.

## Past the stack

| tensor | type | dims | bytes |
|---|---|---|---|
| `blk.60.attn_k.weight` | Q4_K | [4096, 512] | 1179648 |
| `blk.60.attn_k_norm.weight` | F32 | [256] | 1024 |
| `blk.60.attn_norm.weight` | F32 | [4096] | 16384 |
| `blk.60.attn_output.weight` | Q4_K | [8192, 4096] | 18874368 |
| `blk.60.attn_q.weight` | Q4_K | [4096, 16384] | 37748736 |
| `blk.60.attn_q_norm.weight` | F32 | [256] | 1024 |
| `blk.60.attn_v.weight` | Q6_K | [4096, 512] | 1720320 |
| `blk.60.ffn_down_exps.weight` | Q6_K | [1024, 4096, 512] | 1761607680 |
| `blk.60.ffn_down_shexp.weight` | Q6_K | [1024, 4096] | 3440640 |
| `blk.60.ffn_gate_exps.weight` | Q4_K | [4096, 1024, 512] | 1207959552 |
| `blk.60.ffn_gate_inp.weight` | F32 | [4096, 512] | 8388608 |
| `blk.60.ffn_gate_inp_shexp.weight` | F32 | [4096] | 16384 |
| `blk.60.ffn_gate_shexp.weight` | Q4_K | [4096, 1024] | 2359296 |
| `blk.60.ffn_up_exps.weight` | Q4_K | [4096, 1024, 512] | 1207959552 |
| `blk.60.ffn_up_shexp.weight` | Q4_K | [4096, 1024] | 2359296 |
| `blk.60.nextn.eh_proj.weight` | Q4_K | [8192, 4096] | 18874368 |
| `blk.60.nextn.enorm.weight` | F32 | [4096] | 16384 |
| `blk.60.nextn.hnorm.weight` | F32 | [4096] | 16384 |
| `blk.60.nextn.shared_head_norm.weight` | F32 | [4096] | 16384 |
| `blk.60.post_attention_norm.weight` | F32 | [4096] | 16384 |

## Dense and expert bytes per block

| block | kind | tensors | dense GiB | expert GiB | gate / up / down stacks |
|---|---|---|---|---|---|
| 0 | DeltaNet | 19 | 0.090 | 3.891 | Q4_K / Q4_K / Q6_K |
| 1 | DeltaNet | 19 | 0.090 | 3.891 | Q4_K / Q4_K / Q6_K |
| 2 | DeltaNet | 19 | 0.090 | 3.891 | Q4_K / Q4_K / Q6_K |
| 3 | attention | 16 | 0.071 | 3.891 | Q4_K / Q4_K / Q6_K |
| 4 | DeltaNet | 19 | 0.090 | 3.891 | Q4_K / Q4_K / Q6_K |
| 5 | DeltaNet | 19 | 0.090 | 3.891 | Q4_K / Q4_K / Q6_K |
| 6 | DeltaNet | 19 | 0.090 | 3.891 | Q4_K / Q4_K / Q6_K |
| 7 | attention | 16 | 0.069 | 3.375 | Q4_K / Q4_K / Q4_K |
| 8 | DeltaNet | 19 | 0.076 | 3.375 | Q4_K / Q4_K / Q4_K |
| 9 | DeltaNet | 19 | 0.090 | 3.891 | Q4_K / Q4_K / Q6_K |
| 10 | DeltaNet | 19 | 0.076 | 3.375 | Q4_K / Q4_K / Q4_K |
| 11 | attention | 16 | 0.069 | 3.375 | Q4_K / Q4_K / Q4_K |
| 12 | DeltaNet | 19 | 0.090 | 3.891 | Q4_K / Q4_K / Q6_K |
| 13 | DeltaNet | 19 | 0.076 | 3.375 | Q4_K / Q4_K / Q4_K |
| 14 | DeltaNet | 19 | 0.076 | 3.375 | Q4_K / Q4_K / Q4_K |
| 15 | attention | 16 | 0.071 | 3.891 | Q4_K / Q4_K / Q6_K |
| 16 | DeltaNet | 19 | 0.076 | 3.375 | Q4_K / Q4_K / Q4_K |
| 17 | DeltaNet | 19 | 0.076 | 3.375 | Q4_K / Q4_K / Q4_K |
| 18 | DeltaNet | 19 | 0.090 | 3.891 | Q4_K / Q4_K / Q6_K |
| 19 | attention | 16 | 0.069 | 3.375 | Q4_K / Q4_K / Q4_K |
| 20 | DeltaNet | 19 | 0.076 | 3.375 | Q4_K / Q4_K / Q4_K |
| 21 | DeltaNet | 19 | 0.090 | 3.891 | Q4_K / Q4_K / Q6_K |
| 22 | DeltaNet | 19 | 0.076 | 3.375 | Q4_K / Q4_K / Q4_K |
| 23 | attention | 16 | 0.069 | 3.375 | Q4_K / Q4_K / Q4_K |
| 24 | DeltaNet | 19 | 0.090 | 3.891 | Q4_K / Q4_K / Q6_K |
| 25 | DeltaNet | 19 | 0.076 | 3.375 | Q4_K / Q4_K / Q4_K |
| 26 | DeltaNet | 19 | 0.076 | 3.375 | Q4_K / Q4_K / Q4_K |
| 27 | attention | 16 | 0.071 | 3.891 | Q4_K / Q4_K / Q6_K |
| 28 | DeltaNet | 19 | 0.076 | 3.375 | Q4_K / Q4_K / Q4_K |
| 29 | DeltaNet | 19 | 0.076 | 3.375 | Q4_K / Q4_K / Q4_K |
| 30 | DeltaNet | 19 | 0.090 | 3.891 | Q4_K / Q4_K / Q6_K |
| 31 | attention | 16 | 0.069 | 3.375 | Q4_K / Q4_K / Q4_K |
| 32 | DeltaNet | 19 | 0.076 | 3.375 | Q4_K / Q4_K / Q4_K |
| 33 | DeltaNet | 19 | 0.090 | 3.891 | Q4_K / Q4_K / Q6_K |
| 34 | DeltaNet | 19 | 0.076 | 3.375 | Q4_K / Q4_K / Q4_K |
| 35 | attention | 16 | 0.069 | 3.375 | Q4_K / Q4_K / Q4_K |
| 36 | DeltaNet | 19 | 0.090 | 3.891 | Q4_K / Q4_K / Q6_K |
| 37 | DeltaNet | 19 | 0.076 | 3.375 | Q4_K / Q4_K / Q4_K |
| 38 | DeltaNet | 19 | 0.076 | 3.375 | Q4_K / Q4_K / Q4_K |
| 39 | attention | 16 | 0.071 | 3.891 | Q4_K / Q4_K / Q6_K |
| 40 | DeltaNet | 19 | 0.076 | 3.375 | Q4_K / Q4_K / Q4_K |
| 41 | DeltaNet | 19 | 0.076 | 3.375 | Q4_K / Q4_K / Q4_K |
| 42 | DeltaNet | 19 | 0.090 | 3.891 | Q4_K / Q4_K / Q6_K |
| 43 | attention | 16 | 0.069 | 3.375 | Q4_K / Q4_K / Q4_K |
| 44 | DeltaNet | 19 | 0.076 | 3.375 | Q4_K / Q4_K / Q4_K |
| 45 | DeltaNet | 19 | 0.090 | 3.891 | Q4_K / Q4_K / Q6_K |
| 46 | DeltaNet | 19 | 0.076 | 3.375 | Q4_K / Q4_K / Q4_K |
| 47 | attention | 16 | 0.069 | 3.375 | Q4_K / Q4_K / Q4_K |
| 48 | DeltaNet | 19 | 0.090 | 3.891 | Q4_K / Q4_K / Q6_K |
| 49 | DeltaNet | 19 | 0.076 | 3.375 | Q4_K / Q4_K / Q4_K |
| 50 | DeltaNet | 19 | 0.076 | 3.375 | Q4_K / Q4_K / Q4_K |
| 51 | attention | 16 | 0.071 | 3.891 | Q4_K / Q4_K / Q6_K |
| 52 | DeltaNet | 19 | 0.076 | 3.375 | Q4_K / Q4_K / Q4_K |
| 53 | DeltaNet | 19 | 0.090 | 3.891 | Q4_K / Q4_K / Q6_K |
| 54 | DeltaNet | 19 | 0.090 | 3.891 | Q4_K / Q4_K / Q6_K |
| 55 | attention | 16 | 0.071 | 3.891 | Q4_K / Q4_K / Q6_K |
| 56 | DeltaNet | 19 | 0.090 | 3.891 | Q4_K / Q4_K / Q6_K |
| 57 | DeltaNet | 19 | 0.090 | 3.891 | Q4_K / Q4_K / Q6_K |
| 58 | DeltaNet | 19 | 0.090 | 3.891 | Q4_K / Q4_K / Q6_K |
| 59 | attention | 16 | 0.071 | 3.891 | Q4_K / Q4_K / Q6_K |

## Placement, mode `stage`

Contiguous block ranges that minimise the largest stage by bytes (then the sum of squares, then the earliest cuts); `token_embd` on rank 0, `output_norm` and `output` on the last rank, blocks past the stack unloaded. Device bytes are the tensors each padded to 64 bytes plus 4096. Beside the weights, the first version allocates the origin's whole-model arena on every rank: estimated from the header at context 32768 and 4096-row chunks as K/V 0.94 GiB, MoE scratch 1.88 GiB (40960 floats a token in each of three buffers), DeltaNet state 0.21 GiB, chunk rows 2.00 GiB: **5.02 GiB** a rank. That is an estimate; the arena was not run at these sizes for it. Budget 110.0 GiB a node.

### World 3

Cuts [0, 20, 40, 60]; plan sha256 `b2f93d3d13f8b961b97a6e199f221c57ced778fc934bcc4e788099988ed50a67`.

| rank | blocks | tensors | weights GiB | + arena estimate GiB | of the budget | under it |
|---|---|---|---|---|---|---|
| 0 | 0..19 (20) | 366 | 75.32 | 80.34 | 73% | yes |
| 1 | 20..39 (20) | 365 | 72.67 | 77.69 | 71% | yes |
| 2 | 40..59 (20) | 367 | 75.55 | 80.57 | 73% | yes |

Largest stage 75.55 GiB of weights; not loaded: 20 tensors, 3.98 GiB.

### World 4

Cuts [0, 15, 30, 45, 60]; plan sha256 `cff5a9877427e74ea8298f05facb4cd82b5e4c377b7f73cee0bfeb2292c4327d`.

| rank | blocks | tensors | weights GiB | + arena estimate GiB | of the budget | under it |
|---|---|---|---|---|---|---|
| 0 | 0..14 (15) | 277 | 57.03 | 62.05 | 56% | yes |
| 1 | 15..29 (15) | 273 | 54.36 | 59.38 | 54% | yes |
| 2 | 30..44 (15) | 273 | 54.38 | 59.40 | 54% | yes |
| 3 | 45..59 (15) | 275 | 57.77 | 62.79 | 57% | yes |

Largest stage 57.77 GiB of weights; not loaded: 20 tensors, 3.98 GiB.
