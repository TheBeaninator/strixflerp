# Front end gate (StrixFold executor): dsv41-flash.flerp-serve-tp2

Host control-node, 2026-11-24T19:04:37Z. Stages run: check, build, pair; not run: none.

- **check**: PASS
- **build**: PASS
- **pair**: FAIL - RemoteDisconnected: Remote end closed connection without response

Built `strixflerp-dsv41-flash.flerp-serve-tp2`: 31575160 bytes, sha256 0d8acdea5962851f; 45 modules, 337 tree files, 6 C and C++ units linked by the Zig build.

Without a GPU or a worker: usage: ok; text_and_executor_refusal: ok (`er.json and the checkpoint's chat template), 129280 tokens, /mnt/nv4/models/LibertAIDAI__DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw/raw`).

Drafts on:

| request | prompt ids equal | tokens origin / front end | tokens equal | streamed equal | drafts accepted / made | decode tok/s origin / front end | pass |
|---|---|---|---|---|---|---|---|
| short-chat | yes | 165 / 164 | NO, from token None | yes | 121 / 215 | 62.42001217987996 / 59.0 | NO |
| long-chat | yes | 146 / 145 | NO, from token None | yes | 117 / 140 | 60.722682623323514 / 70.0 | NO |
| code | yes | 192 / 192 | yes | yes | 147 / 222 | 62.543106669728566 / 59.03 | yes |

Also required: health: pass; refuse:logprobs: pass; refuse:logit_bias: pass; refuse:temperature-without-top_k: pass; refuse:top_k-above-the-bound: pass; refuse:response_format: pass; refuse:tools: pass; clean exit on SIGTERM: NO.
Recorded: the sampled request's tokens equal the origin's: False.
