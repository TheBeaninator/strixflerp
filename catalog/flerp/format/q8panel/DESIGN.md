# format.flerp.q8panel: design record

Q8P16 of docs/STRIXW-LAYOUT-NEXT.md: format.flerp.q4panel's layout with one code per byte, from strixite's
`Q8Weight` (`format/q8 q8.hpp`: q [N, K] bytes; scale, min [N, K/G] BF16).

## Byte layout
Panel p (rows 16p .. 16p+15) at byte p * P, P = 16K + 64K/G:
- codes: chunk c = 0 .. K/16-1, row r: `q[16p+r][16c .. 16c+15]` (16 codes) at panel byte (c * 16 + r) * 16;
- scale/min: group g, row r: bf16 scale then bf16 min at panel byte 16K + (g * 16 + r) * 4.

16-byte chunks of 16 codes, not the 32-byte chunks of strixite's `Q8ChunkMajor`, so a lane load is 16 B at both
widths and 16 rows of one chunk are 256 contiguous bytes in Q4 and Q8 alike. The relayout code is q4panel's
(`panel16_pack` / `panel16_unpack` at bits 8).

## Alignment
Every fleet Q8 class: sh_gu 43,520; sh_down (G32) 11,520; gdn_out 104,448; attn_qkv 43,520; attn_o 104,448;
ple_kv 43,520; lm_head 43,520; mtp_fc 43,520 bytes per panel, all multiples of 128, and so is 16K.

## Refusals
As q4panel. Order-only: dequantized values bit-identical.
