# Recipe gate: qwen38-flash-next.topk-gufo

Host control-node, 2026-11-11T01:17:51Z. Stages run: check, origin, build, unit, gpu.

- **check**: PASS (2026-11-11T01:06:44Z) - flerp check (origin strixite): 105 modules, 8 recipe(s), 0 error(s), 4 warning(s)
- **origin**: PASS (2026-11-11T01:06:44Z) - diff-origin origin strixite vs /data/projects/strixite-v070: 105 module(s), 269/269 files byte-identical, 0 problem(s), 29 origin files listed as unharvested
- **build**: PASS (2026-11-11T01:06:45Z)
- **unit**: PASS (2026-11-11T01:06:54Z)
- **gpu**: PASS (2026-11-11T01:17:51Z)

| MTP | prompt | decode t/s reference (median) | decode t/s assembled (median) | diff | within noise | host load, 1 min (reference / assembled) |
|---|---|---|---|---|---|---|
| on | short-chat-thinking | 52.73 | 54.57 | +3.5% | yes | 15.75 / 34.31 |
| on | long-chat | 66.34 | 65.33 | -1.5% | yes | 15.75 / 34.31 |
| on | raw-completion | 48.03 | 49.15 | +2.3% | yes | 15.75 / 34.31 |
| off | short-chat-thinking | 33.59 | 33.49 | -0.3% | yes | 13.59 / 6.46 |
| off | long-chat | 33.01 | 32.75 | -0.8% | yes | 13.59 / 6.46 |
| off | raw-completion | 33.43 | 33.38 | -0.1% | yes | 13.59 / 6.46 |

Token identity over 18 requests: identical (prompt ids, generated ids, MTP steps).

Criterion: token-identity. Decode speed is recorded, not a pass criterion.

| MTP | adapter | foreign provider | foreign launches | launches left to the origin's kernel | tally line |
|---|---|---|---|---|---|
| on | `kernel.flerp.logits_topk_gufo` | `kernel.gufo.qwen_sampling:LaunchGPUSortLogits` | 666 | 0 | exit |
| off | `kernel.flerp.logits_topk_gufo` | `kernel.gufo.qwen_sampling:LaunchGPUSortLogits` | 1989 | 0 | exit |
