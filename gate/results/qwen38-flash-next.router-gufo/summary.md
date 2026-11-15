# Recipe gate: qwen38-flash-next.router-gufo

Host control-node, 2026-11-11T00:26:01Z. Stages run: check, origin, build, unit, gpu.

- **check**: PASS - flerp check (origin strixite): 105 modules, 8 recipe(s), 0 error(s), 4 warning(s)
- **origin**: PASS - diff-origin origin strixite vs /data/projects/strixite-v070: 105 module(s), 269/269 files byte-identical, 0 problem(s), 29 origin files listed as unharvested
- **build**: PASS
- **unit**: PASS
- **gpu**: PASS

| MTP | prompt | decode t/s reference (median) | decode t/s assembled (median) | diff | within noise | host load, 1 min (reference / assembled) |
|---|---|---|---|---|---|---|
| on | short-chat-thinking | 51.99 | 49.22 | -5.3% | yes | 10.5 / 12.18 |
| on | long-chat | 63.3 | 67.3 | +6.3% | NO | 10.5 / 12.18 |
| on | raw-completion | 47.24 | 49.26 | +4.3% | yes | 10.5 / 12.18 |
| off | short-chat-thinking | 32.69 | 32.26 | -1.3% | yes | 15.37 / 11.26 |
| off | long-chat | 32.29 | 32.03 | -0.8% | yes | 15.37 / 11.26 |
| off | raw-completion | 32.84 | 32.96 | +0.3% | yes | 15.37 / 11.26 |

Token identity over 18 requests: DIVERGED (prompt ids, generated ids, MTP steps).

Criterion: token-agreement. Decode speed is recorded, not a pass criterion.

| MTP | adapter | foreign provider | foreign launches | launches left to the origin's kernel | tally line |
|---|---|---|---|---|---|
| on | `kernel.flerp.moe_router_gufo` | `kernel.gufo.qfn_kernels:RouterTopK` | 31227 | 0 | exit |
| off | `kernel.flerp.moe_router_gufo` | `kernel.gufo.qfn_kernels:RouterTopK` | 95760 | 0 | exit |

Module gate `gate/results/interface/moe-router/result.json (strixite, gufo)`: PASS.

15 of 18 requests leave the reference's tokens, the earliest at generated token 5. From that token on the two servers generate different text, so their decode speeds for those requests are not a like-for-like comparison (MTP acceptance depends on the text).

For scale: one server against itself, MTP drafting on against off (the same kernels at other batch shapes), first repeat of each prompt. A position here means that prompt's greedy text already turns on last-bit differences inside one binary.

| server | prompt | first position where MTP on and MTP off differ |
|---|---|---|
| reference strixite | short-chat-thinking | 95 |
| reference strixite | long-chat | none |
| reference strixite | raw-completion | 43 |
| this variant | short-chat-thinking | 6 |
| this variant | long-chat | 11 |
| this variant | raw-completion | 58 |

Token agreement with the reference, request by request (first position where the generated ids differ; `-` = none within the request):

| MTP | prompt | repeat | generated tokens (reference / variant) | first divergence | tokens agreeing |
|---|---|---|---|---|---|
| on | short-chat-thinking | 0 | 164 / 96 | 5 | 5 |
| on | long-chat | 0 | 255 / 256 | 11 | 11 |
| on | raw-completion | 0 | 256 / 255 | 53 | 53 |
| on | short-chat-thinking | 1 | 164 / 96 | 5 | 5 |
| on | long-chat | 1 | 255 / 256 | 11 | 11 |
| on | raw-completion | 1 | 256 / 255 | 53 | 53 |
| on | short-chat-thinking | 2 | 164 / 96 | 5 | 5 |
| on | long-chat | 2 | 255 / 256 | 11 | 11 |
| on | raw-completion | 2 | 256 / 255 | 53 | 53 |
| off | short-chat-thinking | 0 | 150 / 150 | 5 | 5 |
| off | long-chat | 0 | 255 / 255 | - | 255 |
| off | raw-completion | 0 | 255 / 255 | 43 | 43 |
| off | short-chat-thinking | 1 | 150 / 150 | 5 | 5 |
| off | long-chat | 1 | 255 / 255 | - | 255 |
| off | raw-completion | 1 | 255 / 255 | 43 | 43 |
| off | short-chat-thinking | 2 | 150 / 150 | 5 | 5 |
| off | long-chat | 2 | 255 / 255 | - | 255 |
| off | raw-completion | 2 | 255 / 255 | 43 | 43 |
