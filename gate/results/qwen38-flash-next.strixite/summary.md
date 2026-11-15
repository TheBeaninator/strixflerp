# Recipe gate: qwen38-flash-next.strixite

Host control-node, 2026-11-13T00:17:52Z. Stages run: check, origin, build, unit, gpu.

- **check**: PASS (2026-11-12T22:25:00Z) - flerp check (origin strixite): 107 modules, 18 recipe(s), 0 error(s), 11 warning(s)
- **origin**: PASS (2026-11-12T22:25:00Z) - diff-origin origin strixite vs /data/projects/strixite-d19945e: 107 module(s), 272/272 files byte-identical, 0 problem(s), 29 origin files listed as unharvested
- **build**: PASS (2026-11-12T22:26:58Z)
- **unit**: PASS (2026-11-13T00:03:26Z)
- **gpu**: PASS (2026-11-13T00:17:52Z)

Origin commit (ORIGIN.toml, reference checkout HEAD): d19945e.
- reference server `/data/models/strix-infer/runtime/bin/strix_server.v070tg` sha256 `b3b8b3d8e928ddd54f3f506a44a41af6356410cc3ddc2c701f47a15328282878`
- assembled server `build/qwen38-flash-next.strixite/ninja/strixflerp-qwen38-flash-next.strixite` sha256 `b8759f3a879bc9d853fb15e1eb69e25ba2c70f1a6837498420c5d4dc50bbb269`

| MTP | prompt | decode t/s reference (median) | decode t/s assembled (median) | diff | within noise | host load, 1 min (reference / assembled) |
|---|---|---|---|---|---|---|
| on | short-chat-thinking | 53.69 | 53.96 | +0.5% | yes | 25.0 / 17.68 |
| on | long-chat | 66.4 | 68.49 | +3.1% | yes | 25.0 / 17.68 |
| on | raw-completion | 50.61 | 51.14 | +1.1% | yes | 25.0 / 17.68 |
| off | short-chat-thinking | 32.81 | 32.97 | +0.5% | yes | 18.51 / 16.67 |
| off | long-chat | 32.71 | 32.38 | -1.0% | yes | 18.51 / 16.67 |
| off | raw-completion | 32.56 | 33.01 | +1.4% | yes | 18.51 / 16.67 |

Token identity over 18 requests: identical (prompt ids, generated ids, MTP steps).
