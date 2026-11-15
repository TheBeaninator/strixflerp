# strixw design: which module holds which choice

strixite stores weights in one container, `weights.strixw`, with every tensor already in its kernel's layout. Each
choice below is recorded in a `DESIGN.md` beside the module that embodies it (SPEC.md section 5, key `design`).
Evidence is file:line in the module's `src/` or in the origin's docs; inferred points are marked as such.

The fleet file (read 2026-11-11 from its header on worker C): layout
`exp_gu=q4g128,exp_down=q4g32,sh_gu=q8g64,sh_down=q8g32,gdn_in=q4g128,gdn_out=q8g64,attn_qkv=q8g64,attn_o=q8g64,hc_down=q8g64,hc_up=q8g64,ple_kv=q8g64,lm_head=q8g64,mtp_fc=q8g64,embed=bf16`,
74.6 GB, shared expert separate. The origin's published layout has expert down at G128 instead; per-class sizes are in
the strixw record.

| choice | what it buys | modules |
|---|---|---|
| asymmetric per-group Q4/Q8, BF16 scale + min; decode folds the min (`s*sum(xq) + m*sum(x)`) | symmetric 4-bit lost accuracy at every group size; one FMA per weight | [q4](../catalog/strixite/format/q4/DESIGN.md), [q8](../catalog/strixite/format/q8/DESIGN.md), [linear_q4](../catalog/strixite/kernel/linear_q4/DESIGN.md), [linear_q8](../catalog/strixite/kernel/linear_q8/DESIGN.md) |
| group size per tensor, chosen per class by a measured study; kernels template on G | G64 to G128 kept accuracy, -5.6% bytes, -1.9% per-token time | [convert_qwen4exp](../catalog/strixite/format/convert_qwen4exp/DESIGN.md), [strixw](../catalog/strixite/format/strixw/DESIGN.md), [moe_grouped](../catalog/strixite/kernel/moe_grouped/DESIGN.md) |
| separate code / scale / min planes, row-major | simple exact row append; decode loads 32 codes per 16 bytes | [q4](../catalog/strixite/format/q4/DESIGN.md), [q4_device](../catalog/strixite/format/q4_device/DESIGN.md), [q8_device](../catalog/strixite/format/q8_device/DESIGN.md) |
| chunk-major copy for one consumer | coalesced per-row loads in the HC up-mix | [q4](../catalog/strixite/format/q4/DESIGN.md), [q8](../catalog/strixite/format/q8/DESIGN.md), [hc](../catalog/strixite/kernel/hc/DESIGN.md), [hc_wmma](../catalog/strixite/kernel/hc_wmma/DESIGN.md) |
| Q5/Q6 = Q4 low plane + high plane | Q5 reuses the Q4 loop at no Q4 cost | [q5](../catalog/strixite/format/q5/DESIGN.md), [q6](../catalog/strixite/format/q6/DESIGN.md), [qsplit](../catalog/strixite/kernel/qsplit/DESIGN.md), [linear_q56](../catalog/strixite/kernel/linear_q56/DESIGN.md) |
| 4096 / 256 alignment, one GTT allocation, pointer = base + offset, gated pinned staging | no load-time conversion; vector-load alignment carried from the file | [strixw](../catalog/strixite/format/strixw/DESIGN.md), [strixw_loader](../catalog/strixite/format/strixw_loader/DESIGN.md) |
| load-time merges of same-input weights; experts stacked [E*N, K] | small projections are launch-bound; fused gather + SwiGLU | [merge_plan](../catalog/strixite/format/merge_plan/DESIGN.md), [linear_q4](../catalog/strixite/kernel/linear_q4/DESIGN.md), [qweight](../catalog/strixite/runtime/qweight/DESIGN.md) |
| shared expert outside the stack when its format differs (fleet: Q8) | 8 bits where 4 hurt | [convert_qwen4exp](../catalog/strixite/format/convert_qwen4exp/DESIGN.md), [qwen4exp](../catalog/strixite/executor/qwen4exp/DESIGN.md) |
| Q4 copy of the LM head's first 65,536 rows for the MTP draft | 178 MB to 94 MB a draft read | [q4_from_q8](../catalog/strixite/format/q4_from_q8/DESIGN.md), [qwen4exp](../catalog/strixite/executor/qwen4exp/DESIGN.md) |
| n-gram table as flat rows on SSD, compression off | 48 GiB kept out of GTT | [ngram_table (format)](../catalog/strixite/format/ngram_table/DESIGN.md), [ngram_table (runtime)](../catalog/strixite/runtime/ngram_table/DESIGN.md), [ple](../catalog/strixite/kernel/ple/DESIGN.md) |
| hashes: header and index at load, components only on `--verify`; prompt cache keyed on fingerprint + layout hash | fast load; a layout change invalidates saved states | [strixw](../catalog/strixite/format/strixw/DESIGN.md), [strixw_loader](../catalog/strixite/format/strixw_loader/DESIGN.md) |
| prefill: grouped experts and WMMA dense dequantize codes to F16 once in LDS, no min-fold; packed codes staged first (1.65x lesson) | matrix units: 141 to ~1,370 tok/s | [linear_wmma](../catalog/strixite/kernel/linear_wmma/DESIGN.md), [moe_grouped](../catalog/strixite/kernel/moe_grouped/DESIGN.md), [qwen4exp](../catalog/strixite/executor/qwen4exp/DESIGN.md) |

Prefill report built on these records: control-node `/data/pod/reports/2026-11/2026-11-11-strixw-prefill.md`
(repository pod-reports).
