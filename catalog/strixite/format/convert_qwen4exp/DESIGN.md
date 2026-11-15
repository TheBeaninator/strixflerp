# format.strixite.convert_qwen4exp: design record

## What it is for
Turns the HF checkpoint into `weights.strixw`: applies the merge plan, quantizes each tensor class to the format
the `--layout` string names, stacks the routed experts, writes the index metadata (`layout`, `source_fingerprint`,
`shared_expert`, ...).

## Choices (origin's code and docs)
- Per-class layout string, e.g. `default=q4g64,gdn_in=q8g64,...`; 14 classes (`exp_gu exp_down sh_gu sh_down gdn_in
  gdn_out attn_qkv attn_o hc_down hc_up ple_kv lm_head mtp_fc embed`); formats `q<4|5|6|8>g<32|64|128>`, `embed`
  BF16 or Q8 (`src/formats/convert_qwen4exp.cpp:286-355`). Only formats a kernel can run are accepted: experts Q4
  or Q5 (no Q6/Q8 gather/combine, `:346-349`); HC mixes not Q5/Q6 (`:350-353`).
- The classes and group sizes come from a measured quality study against the full-precision model (mean KL 0.028,
  226/242 top-1 for the published layout; G64 to G128 for the 4-bit classes kept accuracy for 5.6% fewer bytes and
  1.9% less per-token time) (`strixite docs/how-it-got-fast.md:84-93`; `weights.md:17-31,48-63`).
- Shared expert: stacked as expert #512 only when `sh_gu`/`sh_down` have the experts' formats; otherwise the
  routed experts are written alone and the shared expert as its own (merged gate/up) tensors (`:503-519`).

## What the fleet file says about its conversion
Layout `exp_gu=q4g128,exp_down=q4g32,sh_gu=q8g64,sh_down=q8g32,gdn_in=q4g128,gdn_out=q8g64,attn_qkv=q8g64,
attn_o=q8g64,hc_down=q8g64,hc_up=q8g64,ple_kv=q8g64,lm_head=q8g64,mtp_fc=q8g64,embed=bf16`, converter_git ed4c1f4,
2026-10-05 (header read 2026-11-11; table in `format.strixite.strixw/DESIGN.md`). It differs from the origin's
published layout (`weights.md:19-20`: experts down G128, MTP fc Q4 G64): the fleet pays +3.8 GB for G32 expert
down. Whether a measurement chose it is not recorded (open question).

## Costs and limits
Every layout change is a full re-conversion (~70 GB written) and a new `layout` hash, which invalidates the prompt
cache's saved states (`executor.strixite.qwen4exp_backend` `qwen4exp_backend.cpp:225-241`).

## Coupled to
`merge_plan`, `q4`/`q5`/`q6`/`q8` (quantizers), `strixw` (writer), `safetensors` (input), `tool.strixite.convert_qwen4exp`.

## Prefill remarks
Two prefill-relevant knobs live here: the expert group size (G128 vs G32 changes scale/min traffic and the scale
window cadence of the grouped WMMA kernel) and the shared expert's format (Q4 would let it ride in the stack; it
changes logits on a dense class where 4 bits hurt most, `weights.md:46-53`).
