# Front end gate: qwen38-flash-next.flerp-serve

Host control-node, 2026-11-16T03:11:52Z. Stages run: check, build, gpu.

- **check**: PASS
- **build**: PASS
- **gpu**: PASS

Built: `strixflerp-qwen38-flash-next.flerp-serve` (16839312 bytes, sha256 e9f166f8f729fb22), 85 modules in the assembly.

GPU: each request over HTTP against the native server's capture of the same request with the same MTP setting. Equal step records are equal forwards.

| front end load | request | prompt tokens | prompt ids equal | tokens native / front end | tokens equal | steps equal | finish | text equal | streamed: tokens, steps, text | delivery | rows a verify | drafts accepted / made | decode tok/s native / front end (streamed) | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mtp-on | raw-256 | 21 | yes | 256 / 256 | yes | yes | length | yes | yes, yes, yes | candidates (host) | 2, 3, 4, 5, 6, 7, 8 | 145 / 178 | 38.4 / 48.58 (50.82) | yes |
| mtp-on | chat-256 | 85 | yes | 164 / 164 | yes | yes | stop | not compared (the native reply is split) | yes, yes, yes | candidates (host) | 2, 3, 4, 5, 6, 7, 8 | 106 / 132 | 56.99 / 56.38 (57.19) | yes |
| mtp-on | raw-stop | 21 | yes | 167 / 167 | yes | yes | stop | yes | - | candidates (host) | 2, 3, 4, 5, 6, 7, 8 | 90 / 112 | 49.28 / 50.2 | yes |
| mtp-on | long-chat-256 | 4971 | yes | 255 / 255 | yes | yes | length | yes | yes, yes, yes | candidates (host) | 2, 3, 4, 5, 6, 8 | 205 / 252 | 67.87 / 68.55 (69.02) | yes |
| mtp-on | raw-bias | 21 | yes | 64 / 64 | yes | yes | length | yes | - | candidates (host) | 2, 3, 4, 5, 8 | 39 / 46 | 53.26 / 53.47 | yes |
| mtp-on | chat-sampled | 45 | yes | 128 / 128 | yes | yes | length | yes | yes, yes, yes | candidates (host) | 2, 3, 4, 5, 6, 7, 8 | 93 / 114 | 63.33 / 63.06 (64.42) | yes |
| mtp-on | raw-sampled | 21 | yes | 127 / 127 | yes | yes | length | yes | - | candidates (host) | 2, 3, 4, 5, 6, 7 | 44 / 66 | 38.88 / 40.04 | yes |
| mtp-off | raw-256 | 21 | yes | 255 / 255 | yes | yes | length | yes | yes, yes, yes | candidates (host) | - | 0 / 0 | 32.96 / 32.63 (32.73) | yes |
| mtp-off | chat-256 | 85 | yes | 150 / 150 | yes | yes | stop | not compared (the native reply is split) | yes, yes, yes | candidates (host) | - | 0 / 0 | 32.89 / 33.17 (33.83) | yes |
| mtp-off | raw-stop | 21 | yes | 172 / 172 | yes | yes | stop | yes | - | candidates (host) | - | 0 / 0 | 33.51 / 33.79 | yes |
| mtp-off | long-chat-256 | 4971 | yes | 255 / 255 | yes | yes | length | yes | yes, yes, yes | candidates (host) | - | 0 / 0 | 32.0 / 32.01 (31.29) | yes |
| mtp-off | raw-bias | 21 | yes | 63 / 63 | yes | yes | length | yes | - | candidates (host) | - | 0 / 0 | 33.68 / 33.77 | yes |
| mtp-off | chat-sampled | 45 | yes | 127 / 127 | yes | yes | length | yes | yes, yes, yes | candidates (host) | - | 0 / 0 | 33.49 / 32.65 (32.57) | yes |
| mtp-off | raw-sampled | 21 | yes | 127 / 127 | yes | yes | length | yes | - | candidates (host) | - | 0 / 0 | 30.1 / 32.26 | yes |
| tokens-mtp-on | raw-256 | 21 | yes | 256 / 256 | yes | yes | length | yes | yes, yes, yes | tokens (executor) | 2, 3, 4, 5, 6, 7, 8 | 145 / 178 | 38.4 / 48.56 (50.93) | yes |
| tokens-mtp-on | long-chat-256 | 4971 | yes | 255 / 255 | yes | yes | length | yes | yes, yes, yes | tokens (executor) | 2, 3, 4, 5, 6, 8 | 205 / 252 | 67.87 / 53.33 (67.69) | yes |

Also required: mtp-on logprobs:chat-256: pass (164 tokens, delivery full, mean logprob -0.1679); mtp-on serve:response_format: pass; mtp-off logprobs:chat-256: pass (150 tokens, delivery full, mean logprob -0.16); mtp-off serve:response_format: pass.

Native requests that resumed from an earlier request's state or were fed ids by the engine: none.

GPU: the control-gpu lease was held 499.1 s in 5 holds (native server, MTP on: 100.2 s, native server, MTP off: 101.7 s, front end, executor MTP on: 103.1 s, front end, executor MTP off: 124.0 s, front end taking the executor's draws, MTP on: 70.1 s); no pod job was waiting when a hold began.

The front end calls the executor only through `include/flerp/executor.h`; its token ids and step records are the `flerp` object of its HTTP responses (`"flerp_trace": true`). Native tokens and step records are read from the server's capture files (`--capture-dir`). Decode tok/s is each server's own `timings.predicted_per_second` for the one request: recorded, not a criterion.
