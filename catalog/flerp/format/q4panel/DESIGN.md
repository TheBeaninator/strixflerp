# format.flerp.q4panel: design record

## What it is for
One stored copy of a Q4 weight that both decode (GEMV) and prefill (WMMA, grouped experts) can read in full
128-byte lines: docs/STRIXW-LAYOUT-NEXT.md section 2, candidate A (Q4P16), stage 1 of its plan.

## Byte layout (as implemented)
W [N, K], group size G, row-major source = strixite's `Q4Weight` (`format/q4 q4.hpp`: q [N, K/2], low nibble = even k;
scale, min [N, K/G] BF16). Panel p = rows 16p .. 16p+15, at buffer byte p * P, P = 8K + 64K/G:
- codes: for chunk c = 0 .. K/32-1, for row r = 0 .. 15: the 16 bytes `q[16p+r][16c .. 16c+15]` at panel byte
  (c * 16 + r) * 16. A chunk is today's row-major 16 bytes unchanged (32 codes of one row, inside one group).
- scale/min: for group g = 0 .. K/G-1, for row r: `scale[16p+r][g]` (2 B) then `min[16p+r][g]` (2 B) at panel
  byte 8K + (g * 16 + r) * 4.
Total = N/16 * P = the row-major q + scale + min bytes exactly.

The design text's code formula `(p*K/32*16 + c*16 + r)*16` counts code bytes only; with each panel's scale/min block
after its codes ("Then the s/m block"), panel p starts at p * P, which is what is implemented and tested.

## Alignment (checked per class by tests/cpp/strixw_panel_selftest.cpp)
P and the codes/s-m boundary (8K) are multiples of 128 for every fleet class (P: exp_gu 21,760; exp_down 6,400;
gdn_in 21,760; Q8 classes in format.flerp.q8panel). The design's inference "64K/G % 128 == 0 for every class at
G32/64/128 with K >= 256" fails for K = 640 at G128 (the upstream exp_down): 64 * 640 / 128 = 320, P = 5,440,
64-byte aligned only. The fleet's exp_down is G32 (P = 6,400 = 50 * 128).

## Refusals
N % 16 != 0, K not a multiple of 32, K % G != 0, G outside 32/64/128 (`panel16_geom`).

## Not here
Expert gate/up panel interleaving (design: gate p, up p, gate p+1, ...) is a row permutation, not part of the
encoding: panels keep row order, so within an expert's 1,280 rows of the checkpoint's fused `experts.gate_up_proj`
(gate half then up half, as the linear_q4 gather pairs row j with row 640 + j) gate panel p and up panel p are whole
panels at expert rows 16p and 640 + 16p. Any interleave can be a later, separate step. Kernels reading panels are stages 3-5.
