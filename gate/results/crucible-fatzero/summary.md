# Crucible fat-zero spill fix under TheRock 10.1, on bench box K

2026-11-13/14, bench box K (ltsp-k, gfx1151, balanced power mode), through `pod-kbench` (bench:K lease per job, card tag
`crucible-fatzero`). The kernel is `PrefillMoePcFatPairKernel<5|6>`. Under TheRock 10.1 it spills 772 B/lane. The patch
(`crucible-fat-zero.patch`, "fresh zero per item") removes the spill.

## What ran

| tree | source | toolchain | build sha256 |
|---|---|---|---|
| stock724 | Crucible d6af02c (git archive) | ROCm 7.2.4 (/opt/rocm-7.2.4) | b2187aa2...0d0f0 (byte-identical to the crucible gate's reference build on control-node) |
| stock101 | Crucible d6af02c (git archive) | TheRock 10.1 (/data/models/tools/therock-10.1) | 0efc3b98...60ecb2 |
| fatzero101 | build/upstream/crucible-fatzero (= d6af02c + the patch; only prefill_moe_tile.h differs) | TheRock 10.1 | 6030a28a...36c3f |

All three trees used the gate's command, `make -j1 SPLIT=0 STATE_EPOCH=0 GPU_ARCH=gfx1151 ROCM_PATH=<tc> HIPCC=<tc>/bin/hipcc
build/engine/crucible.mono`. They built in parallel on K in 904-986 s. Each binary has its toolchain's lib dir as RUNPATH, so
each runs on its own HIP runtime. Weights: /data/models/crucible-exl3/4.05bpw_h6_ng6, which K reads over NFS from
control-node's NVMe (no ark or nv* path).

K was booted with the image's default worker-llama (Qwen3.8-27B, ~29 GB GPU). That left 95 GB, and Crucible's first gen
failed with `hipMalloc` out of memory (job 20261113T222551Z-run-1build). The worker-llama unit was stopped at runtime only,
inside our lease (job 20261113T234237Z-run-pllama). It was not masked, and K brings it back at its next boot (card
RC-BOOT-EMPTY 1255617010).

## Tokens: `crucible gen --n 128` on the crucible gate's three prompts (job 20261113T234253Z-run-tokens)

The reference is stock 7.2.4's ids from the crucible gate on control-node (gate/results/qwen38-flash-next.crucible/tokens.json).
The prompt files hash to the gate's prompt hashes (c7514ffbb7bb3a90, af084ad0afe0b8f1, 33ae4c913bb4571c).

| prompt (tokens) | stock 7.2.4 spec 0 | stock 10.1 spec 0 | stock 10.1 spec 3 | patched 10.1 spec 0 | patched 10.1 spec 3 |
|---|---|---|---|---|---|
| short-chat-thinking (85) | = gate | = gate | = gate | = gate | = gate |
| long-chat (4971) | = gate | = gate | = gate | = gate | = gate |
| raw-completion (21) | = gate | = gate | = gate | = gate | = gate |

All 15 runs produced 128 ids identical to the gate's 7.2.4 ids. Patched 10.1 equals stock 7.2.4, and spec 3 equals spec 0.
**Stock 10.1 also matches**, so on these three prompts at n=128, 10.1 shows no token difference for the spill fix to
repair. That is measured only for these prompts. Crucible's own token gates are not reproduced here.

The rates came from the same runs (single runs, so not an A/B):

| prompt | stock 7.2.4 prefill / decode | stock 10.1 prefill / decode | patched 10.1 prefill / decode |
|---|---|---|---|
| short-chat-thinking | 398.5 / 32.60 | 399.3 / 32.74 | 399.2 / 32.72 |
| long-chat | 1733.4 / 32.19 | 1767.9 / 32.26 | 1768.9 / 32.28 |
| raw-completion | 163.6 / 32.62 | 160.9 / 32.74 | 163.2 / 32.70 |

At 4971 tokens, stock and patched 10.1 prefill the same. The fat-pair kernel's cost shows only at 32K (below).

## Speed: 32K-token prefill, stock 10.1 (A) vs patched 10.1 (B), k_ab job 20261114T015418Z-ab-ab1

Command: `crucible gen <ckpt> --ids-file kbin/prefill-32768.ids --n 1 --spec 0 --ctx 33792`. The ids are the crucible
gate's speed-stage prompt (bench_ids(1234, 32768)). Each run is a cold process started from Tctl <= 50 C, in ABBA order
after one warmup per side. The job was **stopped after 4 pairs**: every process reloaded the 108 GB checkpoint over NFS
(372-475 s a load, against ~19-31 s of prefill), which is the cold-load antipattern. So n=4, not the planned 7.
pod-kbench wrote no result.json. The numbers below come from the per-run outputs in the job dir.

| side | runs (tok/s) | median | warmup |
|---|---|---|---|
| A stock 10.1 (772 B/lane spill) | 1049.3, 1047.9, 1044.3, 1045.8 | **1046.9** | 1046.6 |
| B patched 10.1 (no spill) | 1759.6, 1768.7, 1759.1, 1762.4 | **1761.0** | 1759.3 |

B is 68.2% faster than A. The bootstrap 95% interval of B-A is +711..+722 tok/s, so the difference is far outside the
noise. The ranges don't overlap: the spread is 5 tok/s on A and 10 tok/s on B.

These were not measured: patched 10.1 against stock 7.2.4 on K at 32K (the second k_ab was cancelled before it started),
and the 7 planned reps. For context only, since it ran on another host and is not an A/B: the crucible gate measured stock
7.2.4 32K prefill on control-node at 1652.6 tok/s (speed-prefill-32768-r1).

## Lesson for the next Crucible bench on K

Load once per process for all reps (or have the bench loop prefill inside one process), from a copy local to K. Never
reload over NFS on every repeat. K has no NVMe and is not a PLACE node, and its RAM root is 63 GB, so a 108 GB local copy
does not fit today.

## Files

- `tokens.json`: every run's ids, rates and load time
- `raw/crucible-fat-zero.patch`: the patch
- `raw/<job>/`: pod-kbench job dirs (args, per-run stdout/stderr, fetched gen outputs and build logs)
- `raw/stage/`: the staged scripts (build.sh, gen.sh, tokens.sh), the prompt ids and every job's args file
