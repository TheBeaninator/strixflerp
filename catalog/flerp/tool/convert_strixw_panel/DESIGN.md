# tool.flerp.convert_strixw_panel: design record

## What it is for
Stage 2 of docs/STRIXW-LAYOUT-NEXT.md: turn the fleet's strixw v4 into a v5 whose Q4/Q8 classes are panel-encoded,
without the BF16 checkpoint (the change is order-only). Structure from tool.flerp.strixw_patch (same RangeReader,
StrixwWriter plan, copy-with-hash loop, delete-on-failure).

## Classes
Tensor -> class by name, mirroring convert_qwen4exp `make_plan` (experts_gate_up / experts_down, shared_expert
gate_up / down_proj, linear_attn in_proj / out_proj, self_attn qkv / o_proj, hyper_connection mix_down / mix_up,
ple.kv_proj, lm_head, mtp.fc_*, embed_tokens). Default `--classes` = the design's recommendation: exp_gu exp_down
sh_gu sh_down gdn_in gdn_out attn_qkv attn_o ple_kv lm_head mtp_fc. hc_down may be added; hc_up (already
chunk-major), embed and every BF16/F32/I64 tensor are refused or copied. Every converted tensor's encoding must equal
its class's layout entry.

## Streaming and checks
- OLD must be v1..v4 (no panels); --out must not exist; shapes are checked (N % 16, K % 32, K % G, per-expert rows)
  before anything is written.
- One tensor at a time: scale and min read and copied, q read in place, packed into one buffer, written; every
  component read from OLD is hashed and compared with OLD's index first. Copied components are written from the
  read buffer, and the writer's hash must equal OLD's.
- Verification after finish(): StrixwFile (v5) reopens it; metadata must equal the plan (OLD's with `layout`
  rewritten); one pass reads every output component and checks its hash; converted tensors are inverted and compared
  byte for byte with OLD's re-read q/scale/min, copied components byte for byte with OLD's.
- `--plan`: header and index only (writer constructor on a sparse scratch file, deleted). `--check-tensors`: the
  named tensors' byte ranges of OLD only, relayout + offsets + inverse in memory (the real-file gate).

## Costs (fleet file)
I/O: OLD read twice for converted tensors (~72 GB each) plus once for copied ones, output written once and read once:
~300 GB. Memory: the largest tensor (exp_gu, 865 MB of planes) about 2x while writing, 4x while verifying.
Output size: see gate/results/strixw-panel/README.md (alignment differences only).
