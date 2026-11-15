# tool.flerp.convert_qwen4exp_p16: design record

This is the CLI of format.flerp.convert_qwen4exp_p16. The design, evidence and costs are in
`catalog/flerp/format/convert_qwen4exp_p16/DESIGN.md`.

It is an adapted copy of strixite's `tools/convert_qwen4exp.cpp` (20937dc). The changes, marked `flerp`:
- it calls `flerp::convert_qwen4exp_p16`;
- the default `--layout` is the fleet's layout with p16 on the classes the format design names;
- new flags `--plan-scratch DIR`, `--check-against REF` and `--tensors NAME,..|all`.

Built by `src/flerp/convert_qwen4exp_p16/build.sh OUT` with the host g++, using the same flags as
tool.flerp.strixw_patch and convert_strixw_panel (`-O2 -ffp-contract=off`).
