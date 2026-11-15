# Executor ABI conformance: qwen38-flash-next.strixite-abi

Host control-node, 2026-11-13T02:05:39Z. Stages run: check, build, gpu.

- **check**: PASS
- **build**: PASS
- **gpu**: PASS

Required: the engine's decode loop, run through `include/flerp/executor.h`, against the native server's capture of the same request with the same MTP setting. Equal step records are equal forwards.

| request | MTP | prompt tokens | tokens native / ABI | tokens equal | steps native / ABI | steps equal | rows a verify | verifies | drafts accepted / made | verifies kept as a prefix | verifies dropped and run again | driver checks | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chat-48 | on | 85 | 48 / 48 | yes | 7 / 7 | yes | 3, 4, 7, 8 | 5 | 17 / 20 | 1 | 1 | 10 of 10 | yes |
| raw-256 | on | 21 | 256 / 256 | yes | 111 / 111 | yes | 2, 3, 4, 5, 6, 7, 8 | 66 | 145 / 178 | 14 | 16 | 8 of 8 | yes |
| chat-256 | on | 85 | 164 / 164 | yes | 58 / 58 | yes | 2, 3, 4, 5, 6, 7, 8 | 33 | 106 / 132 | 8 | 8 | 4 of 4 | yes |
| long-chat-256 | on | 4971 | 255 / 255 | yes | 50 / 50 | yes | 2, 3, 4, 5, 6, 8 | 47 | 205 / 252 | 17 | 11 | 10 of 10 | yes |
| chat-48 | off | 85 | 47 / 47 | yes | 23 / 23 | yes | - | 0 | 0 / 0 | 0 | - | 7 of 7 | yes |
| raw-256 | off | 21 | 255 / 255 | yes | 255 / 255 | yes | - | 0 | 0 / 0 | 0 | - | 5 of 5 | yes |
| chat-256 | off | 85 | 150 / 150 | yes | 150 / 150 | yes | - | 0 | 0 / 0 | 0 | - | 1 of 1 | yes |
| long-chat-256 | off | 4971 | 255 / 255 | yes | 255 / 255 | yes | - | 0 | 0 / 0 | 0 | - | 7 of 7 | yes |

Ids the engine fed instead of sampling (capture steps with drafted = 255; the driver feeds the same ids at the same place): chat-48: 25 ids after 0 generated tokens.

Not required - the forwards differ. The native server against itself, MTP on (several rows a verify) and off (one row a forward):

| request | tokens on / off | first token that differs | rows a verify with MTP on |
|---|---|---|---|
| chat-48 | 48 / 47 | none in the common length | 3, 4, 7, 8 |
| raw-256 | 256 / 255 | 43 | 2, 3, 4, 5, 6, 7, 8 |
| chat-256 | 164 / 150 | 95 | 2, 3, 4, 5, 6, 7, 8 |
| long-chat-256 | 255 / 255 | none in the common length | 2, 3, 4, 5, 6, 8 |

Not required - the forwards differ. The driver's own paths against each other: raw-256 (MTP on) speculative_equals_plain = False; raw-256 (MTP on) loop_equals_plain = False; raw-256 (MTP on) speculative_first_difference = 11; raw-256 (MTP on) loop_first_difference_from_plain = 43; raw-256 (MTP off) speculative_equals_plain = True; raw-256 (MTP off) speculative_first_difference = 48; raw-256 (MTP off) loop_first_difference_from_plain = 48.

GPU: the control-gpu lease was held 448.0 s in 4 holds (native server, MTP on: 99.3 s, native server, MTP off: 100.0 s, conformance driver, executor MTP on: 119.5 s, conformance driver, executor MTP off: 129.2 s); no pod job was waiting when a hold began.

The conformance driver calls the executor only through `include/flerp/executor.h`. Native tokens and step records are read from the server's capture files (`--capture-dir`).
