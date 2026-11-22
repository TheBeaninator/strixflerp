# SFX-PKG: results of the 2026-11-16 strixflerp package on bench box K

Criteria: `gate/criteria/sfx-pkg.toml` (committed before any hardware run, fa04124).

- Panel Q4P16: `gate/results/strix-loop/20261116-sfx-p16/` (K job 20261117T000124Z-run-pmodel). Identity on all 8 prompts;
  prefill 1.046-1.112x; decode 1.003-1.005x. Promoted: prefill case qwen38-flash-next, `qwen38-fnext.prefill-best`.
  The loop's isa scan did not run (the toolchain was missing), so that run's overall verdict is FAIL on the isa
  criterion. The scratch check of the panel kernels is still owed; identity and speed are unaffected.
- Fused residual + norm + HC: `launch-tax/` (K job 20261116T233509Z-run-x). Dependent launch 2.21 us, grid barrier
  0.82 us. 96 boundaries x 2 x 1.39 us = 0.27 ms of a 30.2 ms decode step (0.9 %), under the 0.6 ms build bar. Not built.
- IndexCache: `qsa-overlap/` (K job 20261117T000158Z-run-x, real text, 12 QSA layers, k = 512 blocks). Adjacent
  attention-layer recall at 32K is 0.647 (decode) and 0.671 (prefill), against the 0.75 threshold. Not built.
