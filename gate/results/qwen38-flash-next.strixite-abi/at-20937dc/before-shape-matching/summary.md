# Executor ABI conformance: qwen38-flash-next.strixite-abi

Host control-node, 2026-11-11T00:32:40Z. Stages run: check, build, gpu.

- **check**: PASS
- **build**: PASS
- **gpu**: FAIL

| case | prompt | executor MTP | prompt forward cut at | drafts a verify | prompt tokens | tokens compared | plain = native server | speculative = plain | snapshot paths | rounds | drafts accepted / made | dropped rounds | required |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a-raw-mtp-on-draft4-drops | raw-completion | on | - | 4 | 21 | 48 | NO | NO | yes | 19 | 28 / 73 | 4 | yes |
| b-raw-mtp-on-draft4-no-drops | raw-completion | on | - | 4 | 21 | 48 | NO | NO | yes | 19 | 28 / 73 | 0 | yes |
| c-raw-mtp-on-draft0-drops | raw-completion | on | - | 0 | 21 | 48 | NO | yes | yes | 47 | 0 / 0 | 11 | yes |
| d-chat-mtp-off-cut | short-chat-thinking | off | 80 | 4 | 85 | 48 | NO | yes | NO | 47 | 0 / 0 | 11 | yes |
| e-chat-mtp-on-cut | short-chat-thinking | on | 80 | 4 | 85 | 48 | NO | yes | NO | 15 | 32 / 59 | 3 | yes |
| f-chat-mtp-off-uncut | short-chat-thinking | off | - | 4 | 85 | 48 | NO | yes | yes | 47 | 0 / 0 | 11 | no (observation) |
| g-raw-mtp-on-draft4-batch-verify | raw-completion | on | - | 4 | 21 | 48 | NO | NO | yes | 19 | 28 / 73 | 0 | no (observation) |

The conformance driver calls the executor only through `include/flerp/executor.h`. Native tokens are read from the server's capture files (`--capture-dir`).
