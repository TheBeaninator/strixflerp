# kernel.flerp.linear_q4p (and the other strixw panel modules): design record

docs/STRIXW-LAYOUT-NEXT.md stages 3-5: kernels that read the Q4Panel16 / Q8Panel16 layout of format.flerp.q4panel /
q8panel (panel p of 16 rows at p * P; codes [chunk c][row r] x 16 B; then (bf16 scale, bf16 min) [group g][row r];
Q4 P = 8K + 64K/G, Q8 P = 16K + 64K/G), and the plumbing that hands them panel tensors inside strixite's executor.

## Modules

| module | what |
|---|---|
| format.flerp.panel_device | the device view, device addressing, and the tag carrying a panel tensor in strixite's Q4DeviceView / Q8DeviceView (q = panels, scale = min = null, G = encoding id << 32 \| G) |
| kernel.flerp.linear_q4p | adapter of kernel.strixite.linear_q4: decode GEMV (kernels/panel_gemv.hpp, Q4 and Q8) and the expert gather / gather + SwiGLU / combine |
| kernel.flerp.linear_q8p | adapter of kernel.strixite.linear_q8: the same GEMV at 16 codes a chunk |
| kernel.flerp.wmma_q4p | adapter of kernel.strixite.linear_wmma: the dense F16 prefill GEMM with WmmaWCodesP (kernels/wmma_panel.hpp) |
| kernel.flerp.moe_grouped_p | adapter of kernel.strixite.moe_grouped: the F16 grouped expert kernels Narrow / Wide / Wide256 (SWI, XH, COMBINE) |
| format.flerp.strixw_loader_p | the loader: StrixwDevice::qw hands out panel tensors tagged |
| format.flerp.strixw_v5p | strixw v5 as an overlay of format.strixite.strixw (one StrixwFile per binary) |
| executor.flerp.qwen4exp_p16 | overlay of the executor: the MTP draft-head copy from a panel LM head |
| tool.flerp.strixw_panel_kernel_bench | the kernel gate's bench |

Recipe: recipes/qwen38-flash-next.strixite-p16.toml. Gate: gate/strixw-panel-kernels.toml.

## Dispatch: a row-major kernel never sees panels

A call site does not choose. The loader tags a panel tensor's view; the adapters (one per strixite kernel module, the
SPEC.md section 5 mechanism) define the names the executor calls and route a tagged view to the panel kernel and any
other view to strixite's kernel (compiled renamed `*_flerp_origin`). Every strixite row-major kernel refuses the tag
by itself (null scale pointer, group size outside 32/64/128), and every panel kernel checks the tag. The only place
the executor reads a projection's planes itself (make_draft_head_q4) is the executor overlay's one change.

## Decode: bit-equal, not merely close

The origin gives each output row 1-8 waves (split-K) and each lane every 32nd 16-byte chunk of that row. In a panel a
row's consecutive chunks are 256 B apart, so keeping the origin's lane mapping would touch 32 lines a load. A wave here
owns 8 rows (half a panel, so a chunk column is one 128-byte line) and the lane L = 8h + r takes, in step t of each
32-chunk block, chunk 4t + h of row r: origin lane o = 4t + h. One accumulator per t means each accumulator sees
exactly origin lane o's chunks in the same order through the same expressions; the butterfly is replayed on o (offsets
16, 8, 4 inside the lane, 2 and 1 as lane shuffles 16 and 8), and the split-K LDS sum is the origin's. Float addition
commutes, so every intermediate is formed from the same operands and the outputs are bit-equal. Cost: 8 accumulators
per x row per lane (64 at M = 8).

decode2 (gate/strixw-panel-decode2.toml, results gate/results/strixw-panel-decode2/): that map ran dense decode at
108-189% of the origin's time - 8x fewer waves than the origin, each 8x longer, 8 accumulators per x row. The map is
now a family `PMap<RW, TPW, BT>` (kernels/panel_gemv.hpp):
- RW = 16 rows a row group (a whole panel), lane L = 16 h + rr, H = 2 chunk columns a load: the two h's read one
  512-byte run, the origin's load pattern (RW = 8 reads 4 lines 256 B apart and measured 2-6% slower).
- The 16 steps t of a 32-chunk block are split over NT = 16 / TPW waves; wave tg's accumulator u is step tg + NT u, so a
  wave's accumulators differ in the top bits of t, the bits the butterfly sums first: each wave folds its TPW values in
  registers, the tg >= 1 waves leave one value per x row in LDS, the tg = 0 wave sums them in the butterfly's order,
  then the h-step is a shuffle (lane ^ 16), then the KS parts in ks order. Still bit-equal.
- TPW = 1 gives the origin's register count and wave count; blocks are max(8, NT x KS) waves, up to 1024 threads.
The table `p_map`: TPW 1 for Q4 with K >= 2048, the expert kernels, and Q8 with N <= 2048; TPW 2 for Q8 at N > 2048, TPW 4 for Q4 with K < 2048 (sh_down: 82-93% against 85-94% at
TPW 2; doubled wherever NT x KS would pass 32 waves). FLERP_PANEL_RW / FLERP_PANEL_TPW override it for
sweeps (tools/decode2-bench.sh).

## Prefill and the RDNA3 WMMA fragment layout

kernel.strixite.wmma `kernels/wmma.hpp:6-8`: an A or B fragment is lane l < 16 holding row l's 16 K values, lanes
16..31 holding the same. A panel's chunks c and c + 1 for its 16 rows are one 512-byte run, which is exactly a wave's
16 rows x 64 K of codes.
- Narrow grouped kernel (`grouped_wmma_f16_kernel`): each wave already decodes its own 16 weight rows in registers;
  the origin staged the packed codes through LDS only to make the loads coalesced (moe_grouped.hip:348-353). With
  panels the wave loads its run directly (lane l: chunk c + l/16 of row l%16) and one lane-pair swap (shfl_xor 16)
  gives each lane both chunks: straight to registers, no code tile in LDS.
- Dense (`linear_wmma_f16_kernel`), Wide and Wide256: two waves read each weight slab and the codes are dequantized
  once into the LDS tile; decoding in registers would do it four times (two waves, two half-waves) in kernels that are
  VALU-bound (linear_wmma.hip:16-18). The LDS staging stays; only the global loads change: WmmaWCodesP gives each
  thread the bytes WmmaWCodes gave it, from whole 128-byte lines (Q4: a wave = one panel's 512-byte run; Q8: four
  128-byte runs), and maps the scale-window entries panel-major (a wave reads one line, scale and min in one 4-byte
  load) while writing them to the same LDS slots.
In every case the F16 values in LDS or registers, the K order and the WMMA sequence are the origin's: bit-equal.

## Not covered

MoeMath::F32 and WmmaBf16 grouped kernels, Q5 / Q6 and chunk-major (HC mix_up) have no panel form; a panel view sent
there is refused. Multi-node slicing (stage 6) is not done.
