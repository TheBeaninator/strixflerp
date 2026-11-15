# Recipe gate: qwen38-flash-next.strixite

Host control-node, 2026-11-11T00:28:11Z. Stages run: check, origin, build, unit, gpu.

- **check**: PASS (2026-11-11T00:28:03Z) - flerp check (origin strixite): 105 modules, 8 recipe(s), 0 error(s), 4 warning(s)
- **origin**: PASS (2026-11-11T00:28:03Z) - diff-origin origin strixite vs /data/projects/strixite-v070: 105 module(s), 269/269 files byte-identical, 0 problem(s), 29 origin files listed as unharvested
- **build**: PASS (2026-11-11T00:28:05Z)
- **unit**: PASS (2026-11-11T00:28:11Z)
- **gpu**: PASS (carried over from an earlier run of this gate)

| MTP | prompt | decode t/s reference (median) | decode t/s assembled (median) | diff | within noise | host load, 1 min (reference / assembled) |
|---|---|---|---|---|---|---|
| on | short-chat-thinking | 55.39 | 55.37 | -0.0% | yes | ? / ? |
| on | long-chat | 66.88 | 67.01 | +0.2% | yes | ? / ? |
| on | raw-completion | 47.8 | 49.46 | +3.5% | yes | ? / ? |
| off | short-chat-thinking | 32.07 | 33.45 | +4.3% | yes | ? / ? |
| off | long-chat | 32.36 | 32.8 | +1.4% | yes | ? / ? |
| off | raw-completion | 32.75 | 33.36 | +1.9% | yes | ? / ? |

Token identity over 18 requests: identical (prompt ids, generated ids, MTP steps).
