# strixw panel decode2: decode kernels at the origin's speed

Criteria: `gate/strixw-panel-decode2.toml` (committed e45401c before any GPU run). Runner: `gate/strixw_panel_decode2.py`,
sweep jobs `tools/decode2-bench.sh`, FETCH_SIZE `tools/decode2-fetch.sh`. All on control-node's GPU through
`tools/gpu-queue.sh`.

## Verdicts (final run, `full/`, `summary.json`)

| criterion | result |
|---|---|
| equal | PASS: 489 / 489 cases byte-equal (21 bad-id cases, 24 Wide256 cases included) |
| decode_dense | PASS: median panel % per class attn_o 98.3, attn_qkv 99.0, gdn_in 97.4, gdn_out 98.6, lm_head 96.8, mtp_fc 98.0, ple_kv 99.4, sh_down 84.1, sh_gu 96.4; no case slower outside noise by > 3% |
| decode_expert | PASS: combine 96.0, gather 99.0, gather + SwiGLU 99.1; none slower |
| prefill | PASS: the committed speed_prefill_dense and speed_prefill_expert still pass |
| read | PASS: 2.232 GB per run |

**Load caveat.** The pod ran at load 6-40 all session (other lanes). The final run started at load 5.75 but ended at
10.24, so under the >6 rule its speed numbers are marked invalid. Runs 1-4 (`full-run1..4`) are kept. All had
489 / 489 equal and the same per-class medians within about 1%. The only case that ever crossed the 3% bar is attn_o Q8
at M = 5 (Q8 K = 6144, KS = 4, map RW 16 / TPW 2): 103.3% in run 1 and 104.5% in run 4 (that run also started at load 7.9).
It was under the bar in runs 2, 3 and 5. Treat it as borderline, not settled.

## Wave maps tried (`sweep.md`, `sweep1/`, `sweep/`; dense = 8 classes × Q4/Q8 × M 1..8 × 2 outputs)

Every map reads the same bytes per load instruction (512 B in 4 full 128-byte lines). RW 8 reads 4 lines 256 B apart.
RW 16 and the origin read one 512-byte run. The old map's problem was wave count and registers, not bytes.

| map | waves / origin's | dense decode, median panel % per class (range) |
|---|---|---|
| committed RW 8 / TPW 8 (one wave per 8 rows) | 1/8 | 109-238 |
| RW 8 / TPW 1 | 1 | 100-105 (Q8 KS 4: 103-105) |
| RW 8 / TPW 2, 4 | 1/2, 1/4 | 101-151 |
| RW 16 / TPW 1 | 1 | Q4 87-98; Q8 K ≤ 2560 97-101; KS 4 not possible (64 waves) |
| RW 16 / TPW 2 (top-bit split) | 1/2 | Q8 99-100.5 except sh_gu (N 1280: 105-113 at M 4-8); Q8 KS 4 99-100 |
| RW 16 / TPW 4 | 1/4 | 101-131 |

Short-K Q4 (sh_down, K 640), 2 runs each: TPW 1 M=1 102-104, TPW 2 102-105, TPW 4 100-103, old map up to 137.
Experts: RW 16 / TPW 1 gave combine 95.8, gather 98.8, SwiGLU 98.8. RW 8 / TPW 1 gave 99.7 / 103.9 / 103.6. Old map: 101.7 / 103.4 / 107.0.

FETCH_SIZE (rocprofv3, gdn_in and sh_gu, M 1 and 8, KB per dispatch). Panel kernels equal the origin within 1% under
both the chosen table and the old map: gdn_in Q8 42.5-42.8 MB against 42.7-43.2 MB, sh_gu Q8 3.40-3.44 MB against
3.43-3.50 MB. The panel layout costs no extra bytes. The decode loss was occupancy and parallelism.

## Chosen map (`p_map` in kernel.flerp.linear_q4p kernels/panel_gemv.hpp)

All use RW 16 (whole panel). TPW 1 for Q4 with K ≥ 2048, for the expert kernels, and for Q8 with N ≤ 2048. TPW 2 for
Q8 with N > 2048. TPW 4 for Q4 with K < 2048. TPW is doubled wherever NT × KS would exceed 32 waves. Bit-equality comes
from replaying the origin's butterfly order: top t-bits in registers, then the rest in LDS, then h by shuffle, then KS in order.

## Add-on: moe_grouped_p Wide256 spills (gate/results/isa-spills)

The panel row's offsets are now formed once, as 32-bit values with shifts. There is no division in the K loop, and
the launcher checks the panel buffer is under 4 GiB. Result (`isa/`, from llvm-readelf notes and llvm-objdump):
- Before: 9 of 24 instantiations used 24-36 B of scratch, with 3 scratch loads per K step.
- After: 3 of 24 (F32 activations, G 128 only) use 8 B, with 1 scratch load per K step. Every BF16 and F16 variant
  (the served ones) and the other G now have 0 scratch.
- A version holding a single offset spilled more (6 instantiations) and was dropped.
- All 24 Wide256 cases are byte-equal, and expert prefill still passes.
