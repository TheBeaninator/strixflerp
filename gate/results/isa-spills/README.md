# Prefill GEMM register spills on gfx1151 (TENSORFOLD.md R5, first experiment)

2026-11-12, compile-side only: no GPU was used and no kernel was run. Per-kernel resources come from the AMDGPU
metadata notes (`llvm-readelf --notes`) of the gfx1151 code objects already in the control-node builds. Scratch
instructions were counted in `llvm-objdump -d --mcpu=gfx1151`. A scratch op counts as "in the hot loop" when it sits
inside a backward branch range that contains `v_wmma`. Occupancy is the waves per SIMD that the VGPR count allows
(1,536 VGPRs per lane, granule 24, wave32). LDS limits are not applied to it.

Sources (control-node `/data/projects/strixflerp/build/`):

- strixite and flerp: `qwen38-flash-next.strixite-p16/ninja/strixflerp-qwen38-flash-next.strixite-p16` and
  `qwen38-flash-next.strixite-pf/...` (therock-10.1, -O3; extracted with `roc-obj`).
- gufo: `qwen35-decider-2b.gufo/out/flerp-gufo-qwen-generate` (therock-10.1, -O2).
- StrixFold: `exl3-pf-bytes/exl3_pf.hsaco` and the zip2 recipe's `exl3.hsaco` (therock-10.2, -O3).

The scripts are in `build/isa-check/` (`extract.sh`, `notes.py`, `hotloop.py`).

## Table

The rows are kernel families. The ranges are over the template instantiations.

| kernel (instantiations) | engine | VGPRs | scratch B | VGPR spills | LDS B | occ (waves/SIMD) | scratch in WMMA loop |
|---|---|---|---|---|---|---|---|
| `linear_wmma_kernel` scaled path, Q5/Q6 (36) | strixite | 162-256 | **200-276** (6 inst.) | **90-144** | <=43,008 | 5-9 | **33-55 ld + 16-30 st per K step, against 32 WMMA** |
| `linear_wmma_f16_kernel` Q4/Q8 (36) | strixite | 126-216 | 0 | 0 | 41,984 | 7-10 | none |
| `linear_bf16w_wmma_kernel` (3) | strixite | 130-210 | 0 | 0 | 28,160 | 7-10 | none |
| `grouped_wmma_f16` Narrow (24) | strixite moe_grouped | 162-165 | 0 | 0 | 37,120 | 9 | none |
| `grouped_wmma_f16_wide` (24) | strixite moe_grouped | 218-223 | 0 | 0 | 43,008 | 6 | none |
| `grouped_wmma_f16_wide256` (24) | strixite moe_grouped | 233-253 | 0 | 0 | 32,768 | 5-6 | none |
| `grouped_wmma` bf16 (24) | strixite moe_grouped | 164-168 | 0 | 0 | 33,280 | 9 | none |
| same five families | flerp moe_grouped_pf | identical to strixite moe_grouped | 0 | 0 | same | same | none |
| `linear_wmma_f16_p_kernel` (36) | flerp wmma_q4p | 134-224 | 0 | 0 | 41,984 | 6-10 | none |
| `grouped_wmma_f16` Narrow (24) | flerp moe_grouped_p | 163-171 | 0 | 0 | 30,976 | 8-9 | none |
| `grouped_wmma_f16_wide` (24) | flerp moe_grouped_p | 229-230 | 0 | 0 | 43,008 | 6 | none |
| `grouped_wmma_f16_wide256` (24) | flerp moe_grouped_p | 206-256 | **24-36** (9 inst.) | **5-8** | 32,768 | 5-7 | **3 ld per K step (32 WMMA) in gate/up and plain; down (COMBINE) spills outside the loop** |
| `tf_ds_exl3_gemm_k{8..16}_m{128,256}` (8) | StrixFold exl3 | 221-251 | 0 | 0 | 38,976-61,504 | 5-6 | none |
| `tf_ds_exl3_pf_experts_{gu,down}{64,128}_k{4..12}` (20, incl. `gu_k6`/`down_k6`) | StrixFold exl3_pf_experts | 188-256 | 0 | 0 | 27,456-55,872 | 5-8 | none |
| prefill GEMM kernels (`HalfPrefillGemm`, quant GEMM, ...) (535 unique) | gufo qwen_linear_prefill | 25-256 | 0, except 4 with 8-28 | 1-6 | <=65,536 | 5-16 | `HalfPrefillGemm<256,256,8,4,Q4_K>`: 2 ld; the 3 `SmallBatchKQuantExactFp32GEMM` spill outside the WMMA loop |
| `affine_wmma_gemm/skew/item` (`wmma_gemm_tile`, the #1663 prompt tile) (22) | upstream TensorFold | 139-217 | 0 | 0 | 34,304-45,056 | 6-10 | none |
| `PrefillGemmOutKernel<RM=16, 2, ...>` (4, incl. GdnPrep epilogue) | Crucible exl3_gemm | **256** | **1,760-1,792** | **749-1,092** | 8,448 | 5 | **51-55 ld + 38-40 st per 32 WMMA (innermost loop)** |
| `PrefillGemmOutKernel` RM 2/4/8 (30) and `PrefillGemmOutWideKernel` | Crucible exl3_gemm | 83-238 | 0 | 0 | 8,448 | 6-16 | none |
| `Exl3GemmKernel<16,...,0,0,1,0>` (1 of 8) | Crucible exl3_gemm | 256 | 164 | 160 | 3,072 | 5 | spill outside the WMMA loop |
| `PrefillMoeTilePair/TileDown/Fx` (11) | Crucible moe_prefill | 192 (cap) / 256 | 28-172 | 6-51 | 8,704-50,180 | 8 / 5 | none inside the WMMA loop (0-1 ld) |

The per-instantiation rows are in `kernels.tsv`. The loop ranges with their scratch counts are in `hotloop-raw.txt`.

TensorFold's `dot2_tiles.hip` and Crucible's `engine.hip` (from the assembled `qwen38-flash-next.crucible/tree`) had
no built objects. They were compiled device-only, after control-node's load fell below 12, with
`-Rpass-analysis=kernel-resource-usage -save-temps`. TensorFold used therock-10.2 with the module's options; Crucible
used rocm-7.2.4 at -O3, as in its Makefile. The summaries are `tensorfold-tiles.tsv`, `crucible-engine.tsv` and
`crucible-hotloop-raw.txt`. The upstream prompt tile is confirmed spill-free here (139-217 VGPRs, no scratch).
TensorFold's own non-WMMA `affine_gemm_kp` dot2 variants do spill at 256 VGPRs, but those are not the prompt tile.

## Which of ours spill in the hot loop

1. **strixite `linear_wmma_kernel<*, *, 5|6, G=32, BMT=128>`**: heavy spilling. The kernel sits at the 256-VGPR cap
   with 90-144 spilled VGPRs and 200-276 B of scratch a lane. Each K step (64 codes) does 33-55 scratch loads and
   16-30 stores around its 32 WMMAs. The other 30 instantiations of this kernel do not spill. That covers Q5/Q6 at
   G64/G128, any BITS at BMT=64, and the Q4/Q8 f16 path.
2. **flerp `moe_grouped_p` `grouped_wmma_f16_wide256_kernel`**: light spilling. 9 of 24 instantiations use 24-36 B of
   scratch. 3 scratch loads per K step fall inside the WMMA loop in the gate/up (SWI) and plain variants. The strixite
   original of the same kernel does not spill (233-253 VGPRs). The panel port pushed it over the cap.
3. **Crucible `PrefillGemmOutKernel<RM=16>`**: the worst spiller found. It has 1.8 KB of scratch per lane, 749-1,092
   spilled VGPRs, and 51-55 scratch loads plus 38-40 stores per 32 WMMAs in the innermost loop. The RM 2/4/8
   instantiations of the same kernel are spill-free (83-238 VGPRs).
4. gufo `HalfPrefillGemm<256,256,8,4,Q4_K>`: 16 B, 2 loads in the loop. That is a single instantiation. It is noise
   unless that tile is the one being dispatched.

None of the prefill GEMMs below spill: StrixFold `exl3.gemm`, `exl3_pf_experts` (`gu_k6`/`down_k6` among them),
strixite `moe_grouped` (Narrow, Wide and Wide256), `moe_grouped_pf` and `wmma_q4p`. For these kernels R5(a) has nothing
to remove. Their occupancy (5-6 waves per SIMD at 220-256 VGPRs) is the same regime as upstream's 248-VGPR tile.

## Cause and the upstream recipe, per spilling kernel

**strixite `linear_wmma_kernel`, Q5/Q6 G32 BMT128.** The scaled path (`wmma_step_scaled`, `wmma_gemm.hpp:463`) keeps a
second accumulator set `p[TM][TN]` for each scale sub-group. G32 gives NSUB=2 per 64-code step. At BMT=128, TM=4, so
`acc[4][2]` and `p[4][2]` take 128 VGPRs together. On top of that come the A/B fragments (48) and the software-pipelined
prefetch registers for 128 x-rows and the wider Q5/Q6 code words (`WmmaXTile` and `WmmaWCodes` fetch during the step).
That total passes 256. The recipe is #1663's: get the live set under the cap without changing the order of the bits.
Re-expressed for strixw, either option below keeps the same bits:
(a) dispatch Q5/Q6 G32 at BMT=64. The BMT=64 instantiation is spill-free here, and the K order is unchanged. This is a
one-line change in `linear_wmma.hip:194`.
(b) give Q5/Q6 the `linear_wmma_f16_kernel` treatment (dequant at staging, which removes `p[][]`). That is a numerics
change, so not R5.

**Expected gain: about 0 on the current fleet.** The fleet's strixw dense tensors are Q8 and its experts are Q4
(`docs/STRIXW-LAYOUT-NEXT.md`, `docs/CRUCIBLE.md:21`). Both take the f16 path, which does not spill. The 18 % of
prefill in GDN projections therefore runs in spill-free code. The fix pays only for a Q5/Q6-G32 strixw build.
**[inferred]** In such a build it would be in upstream's range: 30-40 % of those GEMMs, at 33-55 scratch loads per 32
WMMAs.

**flerp `moe_grouped_p` Wide256.** The panel port adds run-time integer math to the kernel body. The divisions and
modulos are `i / rows_w`, `i % rows_w`, `slot / A`, `N / 2` and the panel address
`flerp::panel_chunk(q, 8*K + 64*K/G, wrow, k0/32)`, recomputed per chunk (`moe_grouped_p.hip:529-563`). The kernel
grows to 256 VGPRs, against 233-253 for the strixite original. That is #1663 exactly. The recipe:

- hoist the panel stride `8*K + 64*K/G` and `K/G` out of the K loop;
- use shift-and-mask for the panel row (`wrow >> 4`, `wrow & 15`) and the chunk (`k0 >> 5`), and `__builtin_ctz` for
  the power-of-two G, which is what upstream's `per_log2` does;
- precompute `slot / A` once per thread into the existing `xoff` table.

**Expected gain [inferred]:** 3 scratch loads per 32 WMMAs, in the Wide256 gate/up only. That is a few % of that kernel
(Wide256 is the large-M class). With experts at 33 % of strixite-pf prefill and the panel recipe's share inside it,
the estimate is <=1 % of prefill. The fix is worth doing as hygiene when the panel kernel is next touched. It is not a
stand-alone win.

**Crucible `PrefillGemmOutKernel<RM=16>`** (`prefill_linear.h:468-474`, the non-K6 body and front paths at
rm == 16). The kernel holds 16 rows x 2 N blocks of accumulators and also issues the A loads early. That live set
does not fit in 256 VGPRs, and the RM=8 version of the same body fits in 233-238. #1663's recipe is to shrink the live
set without changing the bit order. For this kernel that means one of two things. Either dispatch rm=16 jobs as RM=8
over two row halves, which keeps the K order of each row, so the bits are the same. Or drop the early A loads at
RM=16. Upstream's shift-and-mask and uniform-branch cleanup applies to the tile body as well. **Expected gain
[inferred]:** upstream got -32 % cold prefill at 2k from the same kind of spill. This kernel spills at that density
(about 1.6-1.7 scratch ops per WMMA, against upstream's "spill-free fits 248"). So a 20-40 % cut of the RM=16 dense
output GEMM time is plausible. The prefill-level gain is that cut multiplied by the RM=16 GEMMs' share of Crucible's
prefill, which nobody has measured. The first step is a rocprofv3 share for `PrefillGemmOutKernel<16` on a long prompt.

**R5(b) row-block banding** is independent of spills and is the transferable part for the spill-free kernels. That
includes StrixFold's MoE at 34-42 % of prefill and strixite's grouped experts at 33 %. It is a launch-order change with
the same bits, and the upstream result is -12 % at 2k. It needs a GPU measurement, which is outside this check.

## Bottom line

R5(a)'s cheapest win exists in **Crucible**, whose `PrefillGemmOutKernel<RM=16>` spills heavily in its WMMA loop
(fix: run it as RM=8 halves, same bits). It mostly does not exist elsewhere. The prefill GEMMs that serve real models do not spill: strixite
Q4/Q8 f16, the grouped experts, StrixFold exl3 and exl3_pf, and the panel dense kernel. The heavy spiller is a strixite
Q5/Q6-G32 instantiation the fleet does not use. The one live spill is minor: flerp's `moe_grouped_p` Wide256, under
1 % of prefill, with a #1663-style fix. Next steps: run R5(b) banding on StrixFold `exl3_pf_experts` and strixite
`grouped_wmma_f16*`, and measure Crucible's RM=16 share, then run the RM=8 split under a bit-identity gate.
