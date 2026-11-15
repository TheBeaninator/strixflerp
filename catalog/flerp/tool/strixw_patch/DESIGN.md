# tool.flerp.strixw_patch: design record

## What it is for
Produce the fleet's `weights.strixw` with exactly one class changed, without a 4-hour reconversion: today
`mtp_fc` (the MTP head's two input projections, `mtp.fc_embedding.weight` and `mtp.fc_hidden.weight`) from the
fleet's Q8 G64 to the Q4 G64 the upstream file ships. Everything else stays byte-identical.

## How the bytes are made (all strixite's own code)
- Class to tensors: `convert_qwen4exp.cpp` `make_plan` plans `mtp_fc` as `quant_single` of those two checkpoint
  tensors (no parts, no merge). The tool hard-codes the pair and checks both are in OLD with the layout's format.
- Quantization: `quantize_q4` from `format.strixite.q4` over the checkpoint's BF16 widened exactly. The converter's
  `quantize_rows` calls the same function on row blocks across threads; Q4 groups run along K inside a row, so one
  call over all rows gives the same bytes (`append_rows_q4` documents the same exactness).
- Container: `StrixwWriter` plans offsets (4096 per tensor, 256 per component), hashes each component on write and
  writes index and header; OLD's metadata is passed through with only `layout` rewritten. OLD must be v4 (what the
  writer writes), so the header differs only in sizes and hashes.
- Source check: `checkpoint_fingerprint` (index JSON + every shard's name, size and header) must equal OLD's
  `source_fingerprint`; a file with `mtp_source` (a replaced draft head) is refused.

## Refusals
Any override other than `mtp_fc=q4g32|64|128`, more than one class, a class already in the target format, a
fingerprint mismatch, a non-v4 OLD, an existing `--out`. A failed run deletes its partial output.

## Verification before success
Each copied component's hash (computed by the writer from the bytes read) must equal OLD's index hash; after
`finish()` the output is reopened with `StrixwFile` (header, index, every record), every record compared with OLD's
(the two patched ones against the quantized planes), and `verify_hashes` re-reads the whole file.

## Costs
Reads OLD once and the output once (verify); one component in memory at a time (largest: an expert gate/up code plane,
~840 MB). Output size = OLD - 6,553,600 bytes for the fleet file.
