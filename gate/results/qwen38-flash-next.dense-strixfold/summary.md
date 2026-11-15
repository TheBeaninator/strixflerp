# Recipe gate: qwen38-flash-next.dense-strixfold

Host control-node, 2026-11-10T23:54:07Z. Stages run: check, origin, build, unit, gpu.

- **check**: PASS - flerp check (origin strixite): 105 modules, 8 recipe(s), 0 error(s), 4 warning(s)
- **origin**: PASS - diff-origin origin strixite vs /data/projects/strixite-v070: 105 module(s), 269/269 files byte-identical, 0 problem(s), 29 origin files listed as unharvested
- **build**: PASS
- **unit**: PASS
- **gpu**: PASS

| MTP | prompt | decode t/s reference (median) | decode t/s assembled (median) | diff | within noise | host load, 1 min (reference / assembled) |
|---|---|---|---|---|---|---|
| on | short-chat-thinking | 52.99 | 59.04 | +11.4% | NO | 15.2 / 27.1 |
| on | long-chat | 65.04 | 66.94 | +2.9% | yes | 15.2 / 27.1 |
| on | raw-completion | 48.93 | 47.34 | -3.3% | yes | 15.2 / 27.1 |
| off | short-chat-thinking | 33.42 | 31.79 | -4.9% | yes | 17.65 / 10.57 |
| off | long-chat | 32.93 | 31.76 | -3.5% | yes | 17.65 / 10.57 |
| off | raw-completion | 33.53 | 32.66 | -2.6% | yes | 17.65 / 10.57 |

Token identity over 18 requests: DIVERGED (prompt ids, generated ids, MTP steps).

Criterion: token-agreement. Decode speed is recorded, not a pass criterion.

| MTP | adapter | foreign provider | foreign launches | launches left to the origin's kernel | tally line |
|---|---|---|---|---|---|
| on | `kernel.flerp.dense_bf16_strixfold` | `kernel.strixfold.dense_bf16:tf_ds_dense_gemv` | 34587 | 0 | exit |
| off | `kernel.flerp.dense_bf16_strixfold` | `kernel.strixfold.dense_bf16:tf_ds_dense_gemv` | 97056 | 0 | exit |

Module gate `gate/results/interface/dense-bf16/result.json (strixite, strixfold)`: PASS.

15 of 18 requests leave the reference's tokens, the earliest at generated token 5. From that token on the two servers generate different text, so their decode speeds for those requests are not a like-for-like comparison (MTP acceptance depends on the text).

For scale: one server against itself, MTP drafting on against off (the same kernels at other batch shapes), first repeat of each prompt. A position here means that prompt's greedy text already turns on last-bit differences inside one binary.

| server | prompt | first position where MTP on and MTP off differ |
|---|---|---|
| reference strixite | short-chat-thinking | 95 |
| reference strixite | long-chat | none |
| reference strixite | raw-completion | 43 |
| this variant | short-chat-thinking | 5 |
| this variant | long-chat | 11 |
| this variant | raw-completion | 147 |

Token agreement with the reference, request by request (first position where the generated ids differ; `-` = none within the request):

| MTP | prompt | repeat | generated tokens (reference / variant) | first divergence | tokens agreeing |
|---|---|---|---|---|---|
| on | short-chat-thinking | 0 | 164 / 151 | 5 | 5 |
| on | long-chat | 0 | 255 / 255 | - | 255 |
| on | raw-completion | 0 | 256 / 256 | 46 | 46 |
| on | short-chat-thinking | 1 | 164 / 151 | 5 | 5 |
| on | long-chat | 1 | 255 / 255 | - | 255 |
| on | raw-completion | 1 | 256 / 256 | 46 | 46 |
| on | short-chat-thinking | 2 | 164 / 151 | 5 | 5 |
| on | long-chat | 2 | 255 / 255 | - | 255 |
| on | raw-completion | 2 | 256 / 256 | 46 | 46 |
| off | short-chat-thinking | 0 | 150 / 162 | 95 | 95 |
| off | long-chat | 0 | 255 / 255 | 11 | 11 |
| off | raw-completion | 0 | 255 / 255 | 43 | 43 |
| off | short-chat-thinking | 1 | 150 / 162 | 95 | 95 |
| off | long-chat | 1 | 255 / 255 | 11 | 11 |
| off | raw-completion | 1 | 255 / 255 | 43 | 43 |
| off | short-chat-thinking | 2 | 150 / 162 | 95 | 95 |
| off | long-chat | 2 | 255 / 255 | 11 | 11 |
| off | raw-completion | 2 | 255 / 255 | 43 | 43 |
