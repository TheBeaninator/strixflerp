# ORNITH-CTX-RUN: the Ornith serve config's context 65536 on the 35B proxy (card 266647f0af, run card 0675339890)

2026-11-14, control-node GPU through tools/gpu-queue.sh, world 4 over loopback TCP (front end + 3 stage processes), one load (18.0 s),
config ornith397.serve (context 65536, drafts 1, grouped prefill + GQA WMMA attention), 35B Q4_K_M, build = the ORNITH-PCACHE build of the same
sources (reuse keys not set, so the stages run as before). Criteria: gate/ornith-ctx.toml. **PASS** (gate/results/ornith-ctx/result.json).

| criterion | measured |
|---|---|
| serve config context | 65536 |
| chat prompt tokens (FreeFitty-shaped: 101 policy sections + 40 tool definitions + 24 turns) | 19533 |
| HTTP status / completion tokens | 200 / 48 |
| prefill (timings.prompt_ms) | 22.94 s = 851.5 tok/s |
| ids, serve /v1/completions vs generator --staged loopback, both at context 65536 | 32/32 equal |
| GTT per stage process (MiB), idle after load | [8448, 8068, 8458, 9396] |
| GTT growth through the 19.5K request (MiB) | [134, 134, 134, 134] (limit 4,096) |
| a short request after the long one | 200 |
| exit codes | [0, 0, 0, 0] |

Earlier runs of the same harness: run 1 used a stale build (executor key drafts unknown, gate/results/ornith-ctx-run3 holds run 3 on the
rebased build: HTTP 200 and the same 22.7 s prefill, whose generator step used an option the staged driver lacks; criteria unchanged).
The 65536 context costs nothing visible: stage GTT at idle 8.1 to 9.4 GiB is the weights; the KV is allocated for the full context at open.
