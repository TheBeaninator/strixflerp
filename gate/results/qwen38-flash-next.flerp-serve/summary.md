# Front end gate: qwen38-flash-next.flerp-serve

Host control-node, 2026-11-13T02:18:02Z. Stages run: check, build, gpu.

- **check**: PASS
- **build**: PASS
- **gpu**: FAIL

Built: `strixflerp-qwen38-flash-next.flerp-serve` (16830552 bytes, sha256 ffec2475fd5aed37), 85 modules in the assembly.

GPU: each request over HTTP against the native server's capture of the same request with the same MTP setting. Equal step records are equal forwards.

| front end load | request | prompt tokens | prompt ids equal | tokens native / front end | tokens equal | steps equal | finish | text equal | streamed: tokens, steps, text | delivery | rows a verify | drafts accepted / made | decode tok/s native / front end (streamed) | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mtp-on | raw-256 | 21 | yes | 256 / 256 | yes | yes | length | yes | yes, yes, yes | candidates (host) | 2, 3, 4, 5, 6, 7, 8 | 145 / 178 | 44.37 / 50.02 (50.54) | yes |
| mtp-on | chat-256 | 85 | yes | 164 / 164 | yes | yes | stop | not compared (the native reply is split) | yes, yes, yes | candidates (host) | 2, 3, 4, 5, 6, 7, 8 | 106 / 132 | 51.06 / 56.05 (56.28) | yes |
| mtp-on | raw-stop | 21 | yes | 167 / 167 | yes | yes | stop | yes | - | candidates (host) | 2, 3, 4, 5, 6, 7, 8 | 90 / 112 | 49.35 / 49.19 | yes |
| mtp-on | long-chat-256 | 4971 | yes | 255 / 255 | yes | yes | length | yes | yes, yes, yes | candidates (host) | 2, 3, 4, 5, 6, 8 | 205 / 252 | 59.42 / 67.21 (68.05) | yes |
| mtp-on | raw-bias | 21 | yes | 64 / 64 | yes | yes | length | yes | - | candidates (host) | 2, 3, 4, 5, 8 | 39 / 46 | 51.97 / 47.22 | yes |
| mtp-on | chat-sampled | 45 | yes | 128 / 128 | yes | yes | length | yes | yes, yes, yes | candidates (host) | 2, 3, 4, 5, 6, 7, 8 | 93 / 114 | 57.7 / 55.71 (63.9) | yes |
| mtp-on | raw-sampled | 21 | yes | 127 / 127 | yes | yes | length | yes | - | candidates (host) | 2, 3, 4, 5, 6, 7 | 44 / 66 | 33.34 / 34.06 | yes |
| mtp-off | raw-256 | 21 | yes | 255 / 255 | yes | yes | length | yes | yes, yes, yes | candidates (host) | - | 0 / 0 | 29.72 / 28.32 (32.75) | yes |
| mtp-off | chat-256 | 85 | yes | 150 / 150 | yes | yes | stop | not compared (the native reply is split) | yes, yes, yes | candidates (host) | - | 0 / 0 | 32.03 / 30.77 (33.0) | yes |
| mtp-off | raw-stop | 21 | yes | 172 / 172 | yes | yes | stop | yes | - | candidates (host) | - | 0 / 0 | 32.99 / 32.94 | yes |
| mtp-off | long-chat-256 | 4971 | yes | 255 / 255 | yes | yes | length | yes | yes, yes, yes | candidates (host) | - | 0 / 0 | 32.21 / 31.6 (32.0) | yes |
| mtp-off | raw-bias | 21 | yes | 63 / 63 | yes | yes | length | yes | - | candidates (host) | - | 0 / 0 | 32.95 / 31.25 | yes |
| mtp-off | chat-sampled | 45 | yes | 127 / 127 | yes | yes | length | yes | yes, yes, yes | candidates (host) | - | 0 / 0 | 32.76 / 31.38 (32.81) | yes |
| mtp-off | raw-sampled | 21 | yes | 127 / 127 | yes | yes | length | yes | - | candidates (host) | - | 0 / 0 | 31.97 / 31.6 | yes |
| tokens-mtp-on | raw-256 | 21 | yes | 256 / 256 | yes | yes | length | yes | yes, yes, yes | tokens (executor) | 2, 3, 4, 5, 6, 7, 8 | 145 / 178 | 44.37 / 25.4 (50.57) | yes |
| tokens-mtp-on | long-chat-256 | 4971 | yes | 255 / 255 | yes | yes | length | yes | yes, yes, yes | tokens (executor) | 2, 3, 4, 5, 6, 8 | 205 / 252 | 59.42 / 67.55 (67.71) | yes |

Also required: mtp-on logprobs:chat-256: pass (164 tokens, delivery full, mean logprob -0.1679); mtp-on refuse:response_format: FAIL; mtp-off logprobs:chat-256: pass (150 tokens, delivery full, mean logprob -0.16); mtp-off refuse:response_format: FAIL.

Native requests that resumed from an earlier request's state or were fed ids by the engine: none.

GPU: the control-gpu lease was held 575.8 s in 5 holds (native server, MTP on: 118.9 s, native server, MTP off: 90.4 s, front end, executor MTP on: 112.8 s, front end, executor MTP off: 139.3 s, front end taking the executor's draws, MTP on: 114.4 s); waited for the pod's own GPU jobs 1 times (30 s before front end, executor MTP off).

The front end calls the executor only through `include/flerp/executor.h`; its token ids and step records are the `flerp` object of its HTTP responses (`"flerp_trace": true`). Native tokens and step records are read from the server's capture files (`--capture-dir`). Decode tok/s is each server's own `timings.predicted_per_second` for the one request: recorded, not a criterion.
