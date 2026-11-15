# strixw next layout: one stored copy for decode and prefill

Status: design only. Nothing was built or measured. Claims about existing code cite file:line under `/home/user/Code/strixflerp/catalog/strixite/` (for some I relied on the module DESIGN.md records that quote the source; the moe_grouped and wmma_gemm lines I read myself). Anything marked **(inf)** is my own inference or arithmetic and has not been measured. I could not reach the prefill report on control-node, so its numbers below come from the brief and from the linear_wmma and moe_grouped DESIGN records, which quote the PF-CF profile.

Owner's principle: "strixite-izing also means modifying the weights themselves to mesh with the kernel optimizations."

## 0. Summary

The proposal is a **panel-interleaved Q4/Q8 encoding (Q4P16 / Q8P16)**.
- Rows are grouped into panels of 16, which matches the WMMA N dimension.
- Inside a panel, codes are stored chunk-major: the 16-byte chunk *c* of rows 0..15 sits contiguously, then chunk *c+1*, and so on.
- Each panel's scales and mins sit right after its codes, as interleaved (scale, min) BF16 pairs ordered group-major.

What this buys:
- Each 16-byte chunk is still one row's 32 codes, in today's nibble order. It serves as a decode lane load and as a WMMA K-slice.
- One wave load is a contiguous 512 B run for both decode and prefill.
- It only reorders bytes: same size (74.6 GB), same group sizes, and dequantized values bit-identical to today's.
- It supports any row split that is a multiple of 16 and any K split that is a multiple of 32 codes and of G. That includes the pod's 4 x 160 expert split.

Recommendation: use it for routed experts (both projections) and for dense Q4/Q8. Keep HC mix_up chunk-major and keep BF16 as is. Expected prefill gain is about 9-13% of GPU time per 8,192-token chunk **(inf)**. Decode is expected to be about neutral to +2% **(inf)**.

## 1. What each hot path needs from the bytes

| path | kernel | how it reads weights today | what it needs |
|---|---|---|---|
| decode GEMV (dense, M=1..8) | `linear_q4.hip` | 1-8 waves per output row (split-K, ~2048 codes per wave, `:342-350`). Each lane loads 16 B = 32 codes of one row inside one group, then computes `s*dot + m*xsum` per chunk (`:97-121`, `:177-199`). Codes must be 16 B aligned (`:407-408`). Scale and min are separate 2-byte reads per chunk; the record calls them "scattered" (`kernel/linear_q4/DESIGN.md` Costs). | Coalesced 16 B code loads (row-major gives one 512 B run per wave along one row). Scale/min next to the codes, so they do not cost an extra cache line per group per row. |
| decode expert gather + SwiGLU | `linear_q4.hip:162-165`, `:244-276` | Experts stacked [E*N, K]; gate row j and up row j of one expert are computed together. | Gate and up rows of an expert addressable as matching units. For decode, a 16-row panel of gate next to the matching up panel is ideal. |
| prefill dense WMMA | `linear_wmma.hip:14` (BM=64 x-rows, BN=128 W-rows, 8 waves as 2x4); K step `kWmmaKC=64` (`wmma_gemm.hpp:23`) | Per step: 128 rows x 32 B of codes, dequantized to F16 into LDS once (`code*scale+min`, no min-fold, `linear_wmma.hip:64-68`). Scales/mins come from windows. | Row-major means each step touches **128 cache lines, 32 B used of each (25%)**. Later K steps rely on L2 hits for the rest of each line **(inf)**. Wanted: 128 rows x 64 K as a few full lines, plus scale/min in the same stream. |
| prefill grouped experts | `moe_grouped.hip:366+` `grouped_wmma_f16_kernel`; tiles of TR slots x 128 W-rows (`kWmmaBN=128`, `:281`); Wide = 128 slots, from M >= 4096 (`:1019`) | Thread t loads row t/2, 16-byte half t%2 (`:413-418`), so 4 KB is spread over 128 rows. The packed codes go to LDS `cs[128*3 uint4]` and each wave decodes its own 16 rows (`:372-376`). Scale/min windows of kWin=8 groups per row, consecutive threads reading consecutive groups (`:374`, `:421-434`). Comment at `:348-353`: loading rows straight from memory was 1.65x slower, because "what costs is cache lines touched, not bytes". | Same 25% line use as the dense path. Today the LDS staging exists only to make the loads coalesced. If the bytes were already in fragment order, a wave could load its 16 rows directly with full lines **(inf)**. G32 expert-down refreshes the window every 256 codes. |
| HC mix_up | `hc.hip:166-171`; chunk-major `q[(c*N+n)*16]` (`format/q4 q4.hpp:43-55`) | Each thread owns a row; adjacent threads read adjacent 16 B chunks, so a wave reads one 512 B run. | This is already the panel idea with panel height = N. It is a precedent, not a target: 0.35 GB, 16% of prefill is HC, but the time is in `hc_wmma` math and mix_down **(inf)**. |

Notes on cache lines **(inf, assuming 128 B gfx11 L2 lines)**:
- Dense K=2560 Q4 tile step: 128 lines, 4 KB useful, 16 KB pulled.
- Expert gate/up at G128: the scale+min window is 8 groups x 2 B x 2 planes per row, so 2 more lines per row every 8 groups (another 256 lines per 128-row tile per window).

## 2. Candidate layouts

All candidates keep the asymmetric BF16 scale+min and today's nibble order (low nibble = even k, `q4.hpp:4-5`), unless stated.

| id | specification | group size |
|---|---|---|
| **A. Q4P16 (panel-interleaved)** | Rows grouped into panels of 16 (N % 16 == 0; every class qualifies: 640, 1280, 2560, 13952, 16480, 248320 are all multiples of 16). Panel p: codes `[c = 0..K/32)[r = 0..15]` x 16 B, i.e. byte `(p*K/32*16 + c*16 + r)*16`, so each chunk is unchanged from today. Then the s/m block: `[g = 0..K/G)[r = 0..15]` x (bf16 scale, bf16 min) = 4 B. Panel bytes = 8K + 64K/G. K=2560 G128: 20480 + 1280 = 21760 = 170 x 128 exactly. Panels 128-aligned when 64K/G % 128 == 0 **(inf: true for every class at G32/64/128 with K >= 256)**. Experts: expert e's gate panels and up panels interleaved (gate p, up p, gate p+1, ...) so SwiGLU pairs sit side by side. | unchanged (G32/64/128) |
| B. Q4T (tile-major, 128 rows x 64 K) | The linear_wmma DESIGN suggestion: a whole 128x64 block tile contiguous (4 KB) plus its s/m. | must divide tile K |
| C. Q4P16-swz | A + a fixed nibble permutation inside each 16 B (e.g. k, k+8 pairs for the 0x6400 magic F16 convert). x is loaded in the matching permuted order on the GEMV side. | unchanged |
| D. regroup | A with exp_down G32 to G128 (the published layout, -3.8 GB, header-diff) or G64. | changed |
| E. second copy | A tile-major prefill copy beside row-major. | n/a |

Splits for A:
- Row (N) splits must be multiples of 16: gate/up 640/{2,4,5,8} = 320/160/128/80, all fine.
- K splits (down, K=640, 4 nodes x 160) must be multiples of 32 and of G. 160 = 5 chunks, so it works at G32 only, exactly as today. That K constraint is inherent to G, not to A.
- A node's slice of a panel is two contiguous runs (chunks [c0, c1) and groups [g0, g1)), so the converter can write per-node files or the loader can gather once.
- B cannot split K=640 at 160 if its tile K is 64 or 128 (160 % 64 != 0). Rejected for the pod.

## 3. Effect of each candidate

| | decode bytes/transactions | prefill | size | numerics | kernels |
|---|---|---|---|---|---|
| A | Same code bytes. Today one wave owns one row: 512 B runs. With A, a wave owns 16 rows: lanes 0-15 take chunk c and lanes 16-31 chunk c+1 of rows 0..15, still one 512 B run. s/m per group per panel is 64 B contiguous, 1 line per 2 groups for 16 rows, versus roughly 2 line touches per row per group today. Each lane accumulates its own row, so there is no cross-lane reduce (only the two half-waves combine). Split-K still works by giving waves panel K ranges. **(inf) neutral to +2%**; its own s/m traffic falls about 16x in transactions. | Per K64 step a 16-row panel is 512 B contiguous (4 full lines) instead of 16 partial lines, so a 128-row tile is 32 lines instead of 128. s/m come from the same panel stream. LDS staging is optional: lane l's 16 B is exactly row l's 32 K for two 16x16x16 B-fragments. Lanes 16-31 need the same row on RDNA3 wave32 **(inf; check the fragment layout in wmma_gemm.hpp)**, so either one load per half-wave plus a DPP swap, or keep LDS staging with full-line loads. | identical | bit-identical dequant, so token identity is expected | new GEMV (from linear_q4), new dense WMMA (from linear_wmma), new grouped (from moe_grouped `grouped_wmma_f16_kernel`), loader/qweight views |
| B | Bad for GEMV: one row's codes are spread over 128-row tiles, 32 B per tile. | Best tile locality. | +0 | identical | all kernels, and no 160 split |
| C | Saves a few ALU ops per decode in prefill; GEMV needs permuted x | ~+3-5% on dequant-bound kernels **(inf)** | identical | identical | as A, plus x permute |
| D | -6% routed bytes per token (header-diff: -78.6 MB) | fewer s/m loads | -3.8 GB | **changed**: FIDELITY.md KL against the full-precision reference; the pod's 4x160 split breaks at G128 | same as A |
| E | none | best | +68 GB, **does not fit** beside 69.5 GiB | identical | rejected |

## 4. Recommendation and arithmetic

**Adopt A (Q4P16 / Q8P16)** for:
- routed experts gate/up and down;
- shared expert;
- GDN in/out;
- attention qkv/o;
- PLE;
- LM head (Q8P16; draft Q4 copy from it via q4_from_q8 adapted);
- MTP fc.

Keep:
- HC mix_up chunk-major (already coalesced; it is a 1-panel layout);
- HC mix_down: either way; A optional, moving it to A is cheap;
- BF16 embed and router.

Keep G per class as the fleet has it, so the change is order-only. Evaluate D as a separate later step under the fidelity rule, and only for single-node layouts.

Expected prefill **(inf; time shares from the report, 5,523 ms per chunk)**:
- Experts (33%; grouped gemm 1,519 ms at ~43% of the assumed peak): if line efficiency lifts it to the dense kernel's ~57%, 1,519 x 43/57 = 1,146 ms, a saving of about 370 ms (6.7%).
- Dense WMMA (1,424 ms at ~57%; it covers the GDN projections 18% plus attention/shared-expert projections): to 62-65% saves about 110-175 ms (2-3%).
- Removing LDS code staging and the s/m windows frees LDS for 256-slot tiles more often, which cuts the expert re-reads (~136 GB per chunk, `moe_grouped DESIGN`): another 0-3%.
- **Total about 9-13%: 5,523 down to about 4,850-5,000 ms per chunk.** Decode is about 0 to +2%. The risk is waves-per-row tuning for small N (sh_down N=2560 gives 160 panels, fine).

Staged plan:

| stage | work | gate | where |
|---|---|---|---|
| 1 | `format.flerp.q4panel` / `q8panel`: types plus a reference relayout and its inverse (panel to row-major). New strixw encodings (e.g. Q4Panel16, Q8Panel16) and a header **v5**; the reader accepts 1..5 (`strixw.hpp:24`). The layout string gains a suffix `p16` (e.g. `exp_gu=q4g128p16`), which changes the layout hash and so invalidates prompt-cache states (`qwen4exp_backend.cpp:225-241`), as it should. | Round-trip byte equality per tensor; component hashes. | laptop/CI |
| 2 | Converter tool `convert_strixw_panel`: **reads the existing v4 strixw** (order-only, so no BF16 source is needed) and writes v5. The `source_fingerprint` is kept, so vision and ngram pairing still hold (`qwen4exp.cpp:169-171`). Also a `--layout` path in convert_qwen4exp for fresh builds from the ark BF16. | Every tensor's dequantized values equal the v4 file's (full compare); `inspect_strixw --verify`. | a worker's local NVMe (reads 74.6 GB, writes 74.6 GB); never on the pod control node while it serves |
| 3 | `kernel.flerp.linear_q4p` / `linear_q8p` (decode GEMV, gather+SwiGLU), starting from `linear_q4.hip` | Output bit-equal to linear_q4 on the same dequantized weights for M=1..8 and the expert-id paths (NaN on a bad id). Summation order changes per lane, so use a tight FP32 tolerance if not bit-equal; state which. | worker |
| 4 | `kernel.flerp.wmma_q4p` (dense) from `linear_wmma.hip` + `wmma_gemm.hpp`; `kernel.flerp.moe_grouped_p` from `grouped_wmma_f16_kernel` (Narrow/Wide/Wide256, SWI, XH, COMBINE kept) | Equal to the old kernels' outputs (same F16 dequant, same K order give bit-equality); rocprofv3 per-kernel ms on control-node's GPU, against the PF-CF baseline | worker, then control-node GPU in a maintenance window |
| 5 | `runtime.qweight`: a panel view type, dispatched by encoding, so a row-major kernel cannot be handed panels (as `Q4ChunkMajor` does today) | Whole-model **token identity** vs v4 on the fixed prompt set, decode and 8k prefill | worker |
| 6 | Multi-node: per-node slicing (rows by 16, down K by 160 at G32) | Partial sums equal to single node | 4 nodes |
| 7 (optional) | D: exp_down G128 / C swizzle | FIDELITY.md KL vs the full-precision reference; speed | worker plus ark source |

## 5. GGUF MoE path (Ornith / Qwen3-Coder, executor.flerp.qwen35moe)

**Yes, the same idea applies.** `flerp/kernel/moe_kquant/src/flerp/moe_kquant/moe_kquant.hpp:1-12` reads llama.cpp's layout as is: per expert a row-major [rows, row_elements] matrix of K-quant blocks (Q4_K is 144 B per 256 values, with its scales inside the block), one wave per row (gufo `QuantWarpBlockDot`).

A "strixw for GGUF" would interleave 16 rows' superblocks, block b of rows 0..15 contiguous. This is like llama.cpp's own CPU repack (Q4_0x8), with values untouched, so it is order-only and gives token identity.

Two options:
- **At load:** relayout per tensor through a staging buffer. This costs load time but no second copy in memory.
- **Offline:** a `.strixg` sidecar per model, on the worker's NVMe.

Constraints:
- Superblocks are 256 wide, so a K split must be a multiple of 256. PHASE5's row (N) splits by 16 are fine.
- The expert grouped prefill kernel (`moe_grouped_pf`) would need the same panel adaptation.
- Do this after the strixw panel kernels have proved the gain, and reuse their wave-to-panel mapping. Expected gain is similar in kind; it has not been measured **(inf)**.

### Critical Files for Implementation
- /home/user/Code/strixflerp/catalog/strixite/kernel/moe_grouped/src/kernels/moe_grouped.hip
- /home/user/Code/strixflerp/catalog/strixite/kernel/linear_wmma/src/kernels/linear_wmma.hip (+ /home/user/Code/strixflerp/catalog/strixite/kernel/wmma_gemm/src/kernels/wmma_gemm.hpp)
- /home/user/Code/strixflerp/catalog/strixite/kernel/linear_q4/src/kernels/linear_q4.hip
- /home/user/Code/strixflerp/catalog/strixite/format/strixw/src/formats/strixw.hpp (+ format/q4/src/formats/q4.hpp)
- /home/user/Code/strixflerp/catalog/strixite/runtime/qweight/src/runtime/qweight.hpp
