# Recipe gate: qwen38-flash-next.three-engines

Host control-node, 2026-11-24T19:02:26Z. Stages run: check, origin, build, unit, gpu.

- **check**: PASS (2026-11-24T18:58:26Z) - flerp check (origin strixite): 107 modules, 19 recipe(s), 0 error(s), 8 warning(s), 3 dependency cycle warning(s) inside vendored origins (gufo 3)
- **origin**: PASS (2026-11-24T18:58:27Z) - diff-origin origin strixite vs /data/projects/strixite-d19945e: 107 module(s), 272/272 files byte-identical, 0 problem(s), 29 origin files listed as unharvested
- **build**: PASS (2026-11-24T19:02:26Z)
- **unit**: PASS (carried over from an earlier run of this gate)
- **gpu**: PASS (carried over from an earlier run of this gate)

Origin commit (ORIGIN.toml, reference checkout HEAD): d19945e.

| MTP | prompt | decode t/s reference (median) | decode t/s assembled (median) | diff | within noise | host load, 1 min (reference / assembled) |
|---|---|---|---|---|---|---|
| on | short-chat-thinking | 56.59 | 56.3 | -0.5% | yes | 10.15 / 14.06 |
| on | long-chat | 67.56 | 67.61 | +0.1% | yes | 10.15 / 14.06 |
| on | raw-completion | 50.75 | 43.26 | -14.8% | NO | 10.15 / 14.06 |
| off | short-chat-thinking | 30.5 | 32.57 | +6.8% | NO | 16.16 / 10.42 |
| off | long-chat | 30.42 | 32.12 | +5.6% | yes | 16.16 / 10.42 |
| off | raw-completion | 30.94 | 32.45 | +4.9% | yes | 16.16 / 10.42 |

Token identity over 18 requests: DIVERGED (prompt ids, generated ids, MTP steps).

Criterion: token-agreement. Decode speed is recorded, not a pass criterion.

| MTP | adapter | foreign provider | foreign launches | launches left to the origin's kernel | tally line |
|---|---|---|---|---|---|
| on | `kernel.flerp.dense_bf16_strixfold` | `kernel.strixfold.dense_bf16:tf_ds_dense_gemv` | 36732 | 0 | exit |
| on | `kernel.flerp.logits_topk_gufo` | `kernel.gufo.qwen_sampling:LaunchGPUSortLogits` | 720 | 0 | exit |
| on | `kernel.flerp.moe_router_gufo` | `kernel.gufo.qfn_kernels:RouterTopK` | 37164 | 0 | exit |
| off | `kernel.flerp.dense_bf16_strixfold` | `kernel.strixfold.dense_bf16:tf_ds_dense_gemv` | 95472 | 0 | exit |
| off | `kernel.flerp.logits_topk_gufo` | `kernel.gufo.qwen_sampling:LaunchGPUSortLogits` | 1992 | 0 | exit |
| off | `kernel.flerp.moe_router_gufo` | `kernel.gufo.qfn_kernels:RouterTopK` | 95904 | 0 | exit |

Module gate `gate/results/interface/logits-topk/result.json (strixite, gufo), gate/results/interface/dense-bf16/result.json (strixite, strixfold), gate/results/interface/moe-router/result.json (strixite, gufo)`: PASS.

12 of 18 requests leave the reference's tokens, the earliest at generated token 5. From that token on the two servers generate different text, so their decode speeds for those requests are not a like-for-like comparison (MTP acceptance depends on the text).

For scale: one server against itself, MTP drafting on against off (the same kernels at other batch shapes), first repeat of each prompt. A position here means that prompt's greedy text already turns on last-bit differences inside one binary.

| server | prompt | first position where MTP on and MTP off differ |
|---|---|---|
| reference strixite | short-chat-thinking | 95 |
| reference strixite | long-chat | none |
| reference strixite | raw-completion | 43 |
| this variant | short-chat-thinking | 5 |
| this variant | long-chat | none |
| this variant | raw-completion | 44 |

Token agreement with the reference, request by request (first position where the generated ids differ; `-` = none within the request):

| MTP | prompt | repeat | generated tokens (reference / variant) | first divergence | tokens agreeing |
|---|---|---|---|---|---|
| on | short-chat-thinking | 0 | 164 / 162 | 154 | 154 |
| on | long-chat | 0 | 255 / 255 | - | 255 |
| on | raw-completion | 0 | 256 / 255 | 44 | 44 |
| on | short-chat-thinking | 1 | 164 / 162 | 154 | 154 |
| on | long-chat | 1 | 255 / 255 | - | 255 |
| on | raw-completion | 1 | 256 / 255 | 44 | 44 |
| on | short-chat-thinking | 2 | 164 / 162 | 154 | 154 |
| on | long-chat | 2 | 255 / 255 | - | 255 |
| on | raw-completion | 2 | 256 / 255 | 44 | 44 |
| off | short-chat-thinking | 0 | 150 / 151 | 5 | 5 |
| off | long-chat | 0 | 255 / 255 | - | 255 |
| off | raw-completion | 0 | 255 / 255 | 43 | 43 |
| off | short-chat-thinking | 1 | 150 / 151 | 5 | 5 |
| off | long-chat | 1 | 255 / 255 | - | 255 |
| off | raw-completion | 1 | 255 / 255 | 43 | 43 |
| off | short-chat-thinking | 2 | 150 / 151 | 5 | 5 |
| off | long-chat | 2 | 255 / 255 | - | 255 |
| off | raw-completion | 2 | 255 / 255 | 43 | 43 |
