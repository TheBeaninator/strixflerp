# GGUF header census: Qwen3.6-35B-A3B Q4_K_M

`/data/ark/models/Qwen__Qwen3.6-35B-A3B-GGUF/gguf/Qwen3.6-35B-A3B-Q4_K_M.gguf`: 21713462624 bytes (20.22 GiB), GGUF v3. Read: the header only, 10990426 bytes from the start of the file (the data region starts at 10990432); sha256 of the header's 10990426 bytes `fba528ebdddf96d097979ad19abc4b009f27790621e0db6a3fda7ae480d04ee5`. Tool: `gate/gguf_census.py`. Every tensor's name, type, dims, bytes and offset is in `census.json`.

`qwen35moe`: 41 blocks in the header, 1 of them past the stack (`nextn_predict_layers`), so 40 run; hidden 2048, 16 heads, 2 KV heads of 256; 256 experts, 8 used, expert width 512, shared expert width 512; vocabulary 248320; context 262144.

## Totals by type

| type | tensors | bytes | GiB |
|---|---|---|---|
| Q4_K | 379 | 16353312768 | 15.23 |
| Q6_K | 64 | 5258158080 | 4.90 |
| F32 | 310 | 91001344 | 0.08 |
| all | 753 | 21702472192 | 20.21 |

## Where the bytes are

| part | GiB |
|---|---|
| expert stacks of the 40 blocks | 18.16 |
| everything else in those blocks (dense) | 0.88 |
| `token_embd` | 0.27 |
| `output_norm`, `output` | 0.39 |
| blocks past the stack (not loaded) | 0.51 |
| other tensors | 0.00 |

## Q6_K

64 Q6_K tensors, 21 of them expert stacks: `ffn_down_exps` in blocks 0-4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34-40.
The other Q6_K tensors: `attn_qkv.weight` in blocks 0-2, 4, 10, 13, 16, 22, 25, 28, 34, 36-38; `attn_v.weight` in blocks 3, 7, 19, 31, 35, 39-40; `ffn_down_shexp.weight` in blocks 0-4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34-40; `output.weight`.

## Past the stack

| tensor | type | dims | bytes |
|---|---|---|---|
| `blk.40.attn_k.weight` | Q4_K | [2048, 512] | 589824 |
| `blk.40.attn_k_norm.weight` | F32 | [256] | 1024 |
| `blk.40.attn_norm.weight` | F32 | [2048] | 8192 |
| `blk.40.attn_output.weight` | Q4_K | [4096, 2048] | 4718592 |
| `blk.40.attn_q.weight` | Q4_K | [2048, 8192] | 9437184 |
| `blk.40.attn_q_norm.weight` | F32 | [256] | 1024 |
| `blk.40.attn_v.weight` | Q6_K | [2048, 512] | 860160 |
| `blk.40.ffn_down_exps.weight` | Q6_K | [512, 2048, 256] | 220200960 |
| `blk.40.ffn_down_shexp.weight` | Q6_K | [512, 2048] | 860160 |
| `blk.40.ffn_gate_exps.weight` | Q4_K | [2048, 512, 256] | 150994944 |
| `blk.40.ffn_gate_inp.weight` | F32 | [2048, 256] | 2097152 |
| `blk.40.ffn_gate_inp_shexp.weight` | F32 | [2048] | 8192 |
| `blk.40.ffn_gate_shexp.weight` | Q4_K | [2048, 512] | 589824 |
| `blk.40.ffn_up_exps.weight` | Q4_K | [2048, 512, 256] | 150994944 |
| `blk.40.ffn_up_shexp.weight` | Q4_K | [2048, 512] | 589824 |
| `blk.40.nextn.eh_proj.weight` | Q4_K | [4096, 2048] | 4718592 |
| `blk.40.nextn.enorm.weight` | F32 | [2048] | 8192 |
| `blk.40.nextn.hnorm.weight` | F32 | [2048] | 8192 |
| `blk.40.nextn.shared_head_norm.weight` | F32 | [2048] | 8192 |
| `blk.40.post_attention_norm.weight` | F32 | [2048] | 8192 |

## Dense and expert bytes per block

| block | kind | tensors | dense GiB | expert GiB | gate / up / down stacks |
|---|---|---|---|---|---|
| 0 | DeltaNet | 19 | 0.026 | 0.486 | Q4_K / Q4_K / Q6_K |
| 1 | DeltaNet | 19 | 0.026 | 0.486 | Q4_K / Q4_K / Q6_K |
| 2 | DeltaNet | 19 | 0.026 | 0.486 | Q4_K / Q4_K / Q6_K |
| 3 | attention | 16 | 0.018 | 0.486 | Q4_K / Q4_K / Q6_K |
| 4 | DeltaNet | 19 | 0.026 | 0.486 | Q4_K / Q4_K / Q6_K |
| 5 | DeltaNet | 19 | 0.021 | 0.422 | Q4_K / Q4_K / Q4_K |
| 6 | DeltaNet | 19 | 0.021 | 0.422 | Q4_K / Q4_K / Q4_K |
| 7 | attention | 16 | 0.018 | 0.486 | Q4_K / Q4_K / Q6_K |
| 8 | DeltaNet | 19 | 0.021 | 0.422 | Q4_K / Q4_K / Q4_K |
| 9 | DeltaNet | 19 | 0.021 | 0.422 | Q4_K / Q4_K / Q4_K |
| 10 | DeltaNet | 19 | 0.026 | 0.486 | Q4_K / Q4_K / Q6_K |
| 11 | attention | 16 | 0.018 | 0.422 | Q4_K / Q4_K / Q4_K |
| 12 | DeltaNet | 19 | 0.021 | 0.422 | Q4_K / Q4_K / Q4_K |
| 13 | DeltaNet | 19 | 0.026 | 0.486 | Q4_K / Q4_K / Q6_K |
| 14 | DeltaNet | 19 | 0.021 | 0.422 | Q4_K / Q4_K / Q4_K |
| 15 | attention | 16 | 0.018 | 0.422 | Q4_K / Q4_K / Q4_K |
| 16 | DeltaNet | 19 | 0.026 | 0.486 | Q4_K / Q4_K / Q6_K |
| 17 | DeltaNet | 19 | 0.021 | 0.422 | Q4_K / Q4_K / Q4_K |
| 18 | DeltaNet | 19 | 0.021 | 0.422 | Q4_K / Q4_K / Q4_K |
| 19 | attention | 16 | 0.018 | 0.486 | Q4_K / Q4_K / Q6_K |
| 20 | DeltaNet | 19 | 0.021 | 0.422 | Q4_K / Q4_K / Q4_K |
| 21 | DeltaNet | 19 | 0.021 | 0.422 | Q4_K / Q4_K / Q4_K |
| 22 | DeltaNet | 19 | 0.026 | 0.486 | Q4_K / Q4_K / Q6_K |
| 23 | attention | 16 | 0.018 | 0.422 | Q4_K / Q4_K / Q4_K |
| 24 | DeltaNet | 19 | 0.021 | 0.422 | Q4_K / Q4_K / Q4_K |
| 25 | DeltaNet | 19 | 0.026 | 0.486 | Q4_K / Q4_K / Q6_K |
| 26 | DeltaNet | 19 | 0.021 | 0.422 | Q4_K / Q4_K / Q4_K |
| 27 | attention | 16 | 0.018 | 0.422 | Q4_K / Q4_K / Q4_K |
| 28 | DeltaNet | 19 | 0.026 | 0.486 | Q4_K / Q4_K / Q6_K |
| 29 | DeltaNet | 19 | 0.021 | 0.422 | Q4_K / Q4_K / Q4_K |
| 30 | DeltaNet | 19 | 0.021 | 0.422 | Q4_K / Q4_K / Q4_K |
| 31 | attention | 16 | 0.018 | 0.486 | Q4_K / Q4_K / Q6_K |
| 32 | DeltaNet | 19 | 0.021 | 0.422 | Q4_K / Q4_K / Q4_K |
| 33 | DeltaNet | 19 | 0.021 | 0.422 | Q4_K / Q4_K / Q4_K |
| 34 | DeltaNet | 19 | 0.026 | 0.486 | Q4_K / Q4_K / Q6_K |
| 35 | attention | 16 | 0.018 | 0.486 | Q4_K / Q4_K / Q6_K |
| 36 | DeltaNet | 19 | 0.026 | 0.486 | Q4_K / Q4_K / Q6_K |
| 37 | DeltaNet | 19 | 0.026 | 0.486 | Q4_K / Q4_K / Q6_K |
| 38 | DeltaNet | 19 | 0.026 | 0.486 | Q4_K / Q4_K / Q6_K |
| 39 | attention | 16 | 0.018 | 0.486 | Q4_K / Q4_K / Q6_K |

## Placement, mode `stage`

Contiguous block ranges that minimise the largest stage by bytes (then the sum of squares, then the earliest cuts); `token_embd` on rank 0, `output_norm` and `output` on the last rank, blocks past the stack unloaded. Device bytes are the tensors each padded to 64 bytes plus 4096. Beside the weights, the first version allocates the origin's whole-model arena on every rank: estimated from the header at context 32768 and 4096-row chunks as K/V 0.62 GiB, MoE scratch 0.75 GiB (16384 floats a token in each of three buffers), DeltaNet state 0.07 GiB, chunk rows 1.00 GiB: **2.44 GiB** a rank. That is an estimate; the arena was not run at these sizes for it. Budget 110.0 GiB a node.

### World 1

Cuts [0, 40]; plan sha256 `9385002c01da29bd15b162a5df0012c7b509160347ede649d63139fda5e658f9`.

| rank | blocks | tensors | weights GiB | + arena estimate GiB | of the budget | under it |
|---|---|---|---|---|---|---|
| 0 | 0..39 (40) | 733 | 19.70 | 22.15 | 20% | yes |

Largest stage 19.70 GiB of weights; not loaded: 20 tensors, 0.51 GiB.

### World 2

Cuts [0, 20, 40]; plan sha256 `0c4e8f8b716a58ea1b4e9f70fb67ae2d192f23d7060a6d61fdd98efb9a43d20a`.

| rank | blocks | tensors | weights GiB | + arena estimate GiB | of the budget | under it |
|---|---|---|---|---|---|---|
| 0 | 0..19 (20) | 366 | 9.79 | 12.23 | 11% | yes |
| 1 | 20..39 (20) | 367 | 9.91 | 12.36 | 11% | yes |

Largest stage 9.91 GiB of weights; not loaded: 20 tensors, 0.51 GiB.

### World 4

Cuts [0, 10, 20, 31, 40]; plan sha256 `463f54f32ff5eafa57bfaae3a2255e4d529d5c0df7bebcdd9ce3b79bc963257c`.

| rank | blocks | tensors | weights GiB | + arena estimate GiB | of the budget | under it |
|---|---|---|---|---|---|---|
| 0 | 0..9 (10) | 185 | 5.10 | 7.54 | 7% | yes |
| 1 | 10..19 (10) | 181 | 4.69 | 7.14 | 6% | yes |
| 2 | 20..30 (11) | 203 | 5.08 | 7.52 | 7% | yes |
| 3 | 31..39 (9) | 164 | 4.84 | 7.28 | 7% | yes |

Largest stage 5.10 GiB of weights; not loaded: 20 tensors, 0.51 GiB.
