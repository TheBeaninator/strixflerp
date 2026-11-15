# Front end gate: qwen38-flash-next.flerp-serve

Host control-node, 2026-11-11T02:25:05Z. Stages run: check, build, gpu.

- **check**: PASS
- **build**: PASS
- **gpu**: PASS

Built: `strixflerp-qwen38-flash-next.flerp-serve` (16502640 bytes, sha256 1417347bd2d0be94), 82 modules in the assembly.

GPU: each request over HTTP against the native server's capture of the same request with the same MTP setting. Equal step records are equal forwards.

| front end load | request | prompt tokens | prompt ids equal | tokens native / front end | tokens equal | steps equal | finish | text equal | streamed: tokens, steps, text | delivery | rows a verify | drafts accepted / made | decode tok/s native / front end (streamed) | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mtp-on | raw-256 | 21 | yes | 256 / 256 | yes | yes | length | yes | yes, yes, yes | candidates (host) | 2, 3, 4, 5, 6, 7, 8 | 145 / 178 | 44.9 / 50.86 (50.41) | yes |
| mtp-on | chat-256 | 85 | yes | 164 / 164 | yes | yes | stop | not compared (the native reply is split) | yes, yes, yes | candidates (host) | 2, 3, 4, 5, 6, 7, 8 | 106 / 132 | 51.52 / 38.33 (57.51) | yes |
| mtp-on | raw-stop | 21 | yes | 167 / 167 | yes | yes | stop | yes | - | candidates (host) | 2, 3, 4, 5, 6, 7, 8 | 90 / 112 | 49.42 / 49.67 | yes |
| mtp-on | long-chat-256 | 4971 | yes | 255 / 255 | yes | yes | length | yes | yes, yes, yes | candidates (host) | 2, 3, 4, 5, 6, 8 | 205 / 252 | 67.8 / 64.72 (68.54) | yes |
| mtp-on | raw-bias | 21 | yes | 64 / 64 | yes | yes | length | yes | - | candidates (host) | 2, 3, 4, 5, 8 | 39 / 46 | 52.64 / 52.39 | yes |
| mtp-on | chat-sampled | 45 | yes | 128 / 128 | yes | yes | length | no (chat: recorded) | yes, yes, yes | candidates (host) | 2, 3, 4, 5, 6, 7, 8 | 93 / 114 | 64.03 / 63.39 (63.15) | yes |
| mtp-on | raw-sampled | 21 | yes | 127 / 127 | yes | yes | length | yes | - | candidates (host) | 2, 3, 4, 5, 6, 7 | 44 / 66 | 39.84 / 39.03 | yes |
| mtp-off | raw-256 | 21 | yes | 255 / 255 | yes | yes | length | yes | yes, yes, yes | candidates (host) | - | 0 / 0 | 27.46 / 19.21 (32.46) | yes |
| mtp-off | chat-256 | 85 | yes | 150 / 150 | yes | yes | stop | not compared (the native reply is split) | yes, yes, yes | candidates (host) | - | 0 / 0 | 32.39 / 32.03 (32.38) | yes |
| mtp-off | raw-stop | 21 | yes | 172 / 172 | yes | yes | stop | yes | - | candidates (host) | - | 0 / 0 | 23.85 / 32.33 | yes |
| mtp-off | long-chat-256 | 4971 | yes | 255 / 255 | yes | yes | length | yes | yes, yes, yes | candidates (host) | - | 0 / 0 | 32.36 / 30.93 (32.5) | yes |
| mtp-off | raw-bias | 21 | yes | 63 / 63 | yes | yes | length | yes | - | candidates (host) | - | 0 / 0 | 32.68 / 32.95 | yes |
| mtp-off | chat-sampled | 45 | yes | 127 / 127 | yes | yes | length | no (chat: recorded) | yes, yes, yes | candidates (host) | - | 0 / 0 | 32.88 / 33.11 (33.23) | yes |
| mtp-off | raw-sampled | 21 | yes | 127 / 127 | yes | yes | length | yes | - | candidates (host) | - | 0 / 0 | 32.7 / 33.32 | yes |
| tokens-mtp-on | raw-256 | 21 | yes | 256 / 256 | yes | yes | length | yes | yes, yes, yes | tokens (executor) | 2, 3, 4, 5, 6, 7, 8 | 145 / 178 | 44.9 / 36.35 (51.61) | yes |
| tokens-mtp-on | long-chat-256 | 4971 | yes | 255 / 255 | yes | yes | length | yes | yes, yes, yes | tokens (executor) | 2, 3, 4, 5, 6, 8 | 205 / 252 | 67.8 / 65.56 (68.17) | yes |

Also required: mtp-on logprobs:chat-256: pass (164 tokens, delivery full, mean logprob -0.1679); mtp-on refuse:response_format: pass; mtp-off logprobs:chat-256: pass (150 tokens, delivery full, mean logprob -0.16); mtp-off refuse:response_format: pass.

Native requests that resumed from an earlier request's state or were fed ids by the engine: none.

GPU: the control-gpu lease was held 468.0 s in 5 holds (native server, MTP on: 80.4 s, native server, MTP off: 103.1 s, front end, executor MTP on: 86.2 s, front end, executor MTP off: 122.0 s, front end taking the executor's draws, MTP on: 76.3 s); no pod job was waiting when a hold began.

The front end calls the executor only through `include/flerp/executor.h`; its token ids and step records are the `flerp` object of its HTTP responses (`"flerp_trace": true`). Native tokens and step records are read from the server's capture files (`--capture-dir`). Decode tok/s is each server's own `timings.predicted_per_second` for the one request: recorded, not a criterion.
