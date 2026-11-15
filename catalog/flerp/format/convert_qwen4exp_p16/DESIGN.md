# format.flerp.convert_qwen4exp_p16 / tool.flerp.convert_qwen4exp_p16: design record

## What it is for
It builds a strixw v5 panel file from the original HF BF16 checkpoint, the `--layout` path of
docs/STRIXW-LAYOUT-NEXT.md stage 2. It does not start from an existing v4 file, which is what
tool.flerp.convert_strixw_panel does.

## What changed from format.strixite.convert_qwen4exp (20937dc)
These are adapted copies of `formats/convert_qwen4exp.{cpp,hpp}` and `tools/convert_qwen4exp.cpp`. Every change is
marked `flerp`:
- **Namespace.** The code is in namespace `flerp` (with `using namespace strix`), and the entry point is renamed
  `convert_qwen4exp_p16`. This lets both converters link into one test binary.
- **Layout string.** Any class except `hc_up` and `embed` may carry the suffix `p16`: `q4g<G>p16` or `q8g<G>p16`.
  Refused: p16 on Q5 or Q6, p16 on `default=`, and a group size followed by anything other than p16 (the origin's
  `stoll` accepted `q4g64x16`). `QFmt` equality includes p16, so the shared expert is stacked as expert #E only
  when its format, p16 included, equals the experts'.
- **Planner.** A p16 class gets encoding `Q4Panel16` / `Q8Panel16` (format.flerp.strixw_v5).
- **Execution.** The origin's per-step body moved into `produce_step(..., emit)`, with the quantization unchanged
  line for line:
  - a p16 tensor is quantized exactly as the origin does (same `quantize_rows`, chunking and threads, HC fold for
    `hc_down`);
  - then `to_panel16` (format.flerp.q4panel / q8panel) re-lays it out, and it is written as one Q component.
  The writer is the v5 `StrixwWriter`, so the output header is v5 even for a layout without p16.
- **`--plan --plan-scratch DIR`.** Plans the file with the writer on a sparse scratch file, deleted again. It logs
  bytes per encoding, the exact file size and data offset, and the source fingerprint.
- **`--check-against REF --tensors NAME,..|all`.** Quantizes only those tensors from the checkpoint and compares them
  with REF's same-named tensors, read through the map, so only their pages are read:
  - REF row-major vs our panels: (a) our panels inverted must equal REF's q/scale/min, and (b) REF's planes run
    through `panel16_pack` (convert_strixw_panel's relayout) must equal our panels.
  - Otherwise: byte equality per component.
- **Tool default `--layout`.** The fleet's layout (gate/results/strixw-layout/header-diff.md) with p16 on the
  classes docs/STRIXW-LAYOUT-NEXT.md section 4 recommends, which are every Q4/Q8 class except the HC mixes:
  `exp_gu=q4g128p16,exp_down=q4g32p16,sh_gu=q8g64p16,sh_down=q8g32p16,gdn_in=q4g128p16,gdn_out=q8g64p16,
  attn_qkv=q8g64p16,attn_o=q8g64p16,hc_down=q8g64,hc_up=q8g64,ple_kv=q8g64p16,lm_head=q8g64p16,mtp_fc=q8g64p16,embed=bf16`.

## Metadata
The metadata matches the origin's: `source_fingerprint` is the checkpoint's, so vision.strixw and the n-gram table
still pair. `layout` carries p16. It therefore differs from convert_strixw_panel's output only in `source_dir`,
`created`, `converter_git` and `complete`/`layers`-style keys. In the stacked case, convert_strixw_panel keeps OLD's
`sh_*` label (those classes have no tensors of their own), while this path writes `sh_*=...p16`.

## Evidence
- `tests/test_convert_qwen4exp_p16.py`: a synthetic two-layer checkpoint (GDN and full attention, PLE, MTP, two
  shards).
  - Without p16, the output equals the origin's tensor for tensor.
  - With p16, it equals convert_strixw_panel applied to the origin's output, component bytes and hashes, in both the
    split and the stacked shared-expert case.
  - `--check-against` passes against both, and fails on a real difference.
  - The `--plan` size equals the written file.
  - Refusals are tested.
- `gate/results/strixw-p16-raw/`: real tensors of the ark checkpoint against the fleet file, and `--plan` for the
  full model.

## Costs
A full conversion reads the whole checkpoint (~336 GiB) and writes ~74.6 GB. That is the same work as the origin
converter, plus one relayout copy per tensor in memory (the largest, exp_gu, is ~0.9 GB).
