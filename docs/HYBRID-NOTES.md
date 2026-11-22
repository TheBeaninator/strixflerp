# Hybrid recipe notes: GDN majority, sparse minority, multi-branch residual, fine MoE

Status: design note. Nothing here is built.

Source: owner's research package of 2026-11-16, `experimental/recipes/qwen-hybrid.notes.toml`. It sketches a recipe
`qwen-hybrid-gdn-sparse` with these parts:
- a majority of fixed-state linear layers (GDN);
- a minority of sparse-attention layers behind an indexer;
- a multi-branch residual (HC / mHC);
- fine-grained MoE;
- the panel layout;
- a fused residual kernel.

## The shape already exists: `qwen38-flash-next.strixite`

Qwen3.8-Flash-Next is that hybrid. It runs as `executor.strixite.qwen4exp`
(`catalog/strixite/executor/qwen4exp/src/runtime/qwen4exp.hpp:4`, dims at `:61-73`):
- 48 decoder layers: 36 GDN and 12 full-attention layers, one attention layer in every four;
- a QSA indexer on each attention layer;
- H = 4 hyper-connection residual streams, low rank r = 320;
- 512 routed experts, top-10, plus a shared expert;
- an n-gram embedding (PLE) at one layer;
- an MTP draft head.

Each line of the notes maps to a binding the served recipes already make:

| notes line | what the catalogue binds today | where |
|---|---|---|
| `executor.strixite.qwen4exp` (or a derived hybrid executor) | the same executor; the derived form is an `[overlay]`, as `executor.flerp.qwen4exp_p16` | `recipes/qwen38-flash-next.strixite-p16.toml` |
| `kernel.strixite.gdn` (majority linear layers) | `mixer.gdn/-/bf16` = `kernel.strixite.gdn` (chunked delta rule on WMMA for prefill) | every qwen38 recipe |
| `kernel.strixite.qsa` (or IndexCache-wrapped) | `attn.qsa-indexer/-/bf16` = `kernel.strixite.qsa` | every qwen38 recipe |
| `kernel.experimental.fused_residual_hc` | HC is `kernel.strixite.hc` (decode) / `hc_wmma` (prefill); the injection is already fused with the next mix's RMS on prefill (`hc_inject_inv`) and with the shared-expert add on decode (`moe_shared_add_inject`) | `qwen4exp.cpp:2877-2884`, `:3217-3219` |
| `kernel.flerp.linear_q4p` (panel layout) | `linear.decode/strixw-q4` and `moe.experts-decode/strixw-q4` in the p16 recipe | `recipes/qwen38-flash-next.strixite-p16.toml` |
| `kernel.strixite.moe_grouped` | `moe.experts-prefill/strixw-q4`: `moe_grouped` (origin), `moe_grouped_pf` (served pf), `moe_grouped_p` (panels) | the pf / p16 recipes |

So the "hybrid recipe" is not a new model shape for strixflerp. It is a choice among variants of the existing Flash-Next
recipe. Its parts are measured one at a time by the SFX-PKG card (pod board, 2026-11-16):
- panel layout: whole-model strix-loop gate on K;
- fused residual: the launch-tax arithmetic;
- IndexCache: the measured cross-layer QSA overlap.

A winning part goes into the Flash-Next champion entry (`configs/champions.toml`) through `flerp champion propose`.
A hybrid recipe file is not the place for it.

## When a separate hybrid recipe would be worth building

Build one only when one of these happens:

1. **A second model with this shape arrives with different dims.** Examples: another GDN/QSA/HC checkpoint, or
   Flash-Next with a different layer ratio. The executor reads its dims from the checkpoint header (`Qwen4ExpDims`),
   so the first step is a recipe with the same bindings and a gate against that model's own reference. No new code
   should be needed unless the ratio breaks an assumption. The one assumption to check is the HC kernels' `kH = 4`
   (`kernel/hc/src/kernels/hc.hip:15`).
2. **Two winning variants do not compose.** Example: PF-13's LDS-panel GEMV (`kernel.flerp.linear_panel`, served
   from the PF-13 roll) and the p16 panel decode (`kernel.flerp.linear_q4p`) bind the same interface
   `linear.decode/strixw-q4/bf16`. If both won, the champion would need one module that does both. That would be a
   new kernel module, not a recipe.
3. **A sparse-attention change needs layer-to-layer state.** An IndexCache is the example: it carries the previous
   attention layer's selection into the next one. The executor's layer loop owns that state (`qwen4exp.cpp:3108-3121`),
   so it would be an executor overlay plus a QSA adapter, gated for token agreement under the FIDELITY.md rule,
   because a cached selection changes which keys are attended.

## The cheap measurement that decides

The note asks for a hybrid recipe that combines the package's parts. That recipe pays only if a part pays alone, and
each part has its own gate on the SFX-PKG card. None of them is free of its own measurement, so no extra measurement
is run for the hybrid as a whole. Result: no recipe is built until a part is promoted.
