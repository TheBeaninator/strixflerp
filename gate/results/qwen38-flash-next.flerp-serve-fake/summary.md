# Front end gate: qwen38-flash-next.flerp-serve-fake

Host control-node, 2026-11-11T02:01:11Z. Stages run: check, build, nogpu.

- **check**: PASS
- **build**: PASS
- **nogpu**: PASS

Built: `strixflerp-qwen38-flash-next.flerp-serve-fake` (1032056 bytes, sha256 6677ac874ff05ba2), 12 modules in the assembly.

No GPU: the front end over the stand-in executor (`executor.flerp.fake_abi`) in 9 shapes, 551 checks.

| stand-in executor | checks | passed | failed |
|---|---|---|---|
| rows: `executor.vocab=248077 executor.lo=1000 executor.hi=30000 executor.reply=40 executor.end=248046` | 60 | 60 | - |
| rows-step: `executor.vocab=248077 executor.lo=1000 executor.hi=30000 executor.reply=40 executor.end=248046 executor.draft=step` | 62 | 62 | - |
| rows-held: `executor.vocab=248077 executor.lo=1000 executor.hi=30000 executor.reply=40 executor.end=248046 executor.draft=held` | 62 | 62 | - |
| full-step: `executor.vocab=248077 executor.lo=1000 executor.hi=30000 executor.reply=40 executor.end=248046 executor.deliveries=full,tokens executor.draft=step` | 62 | 62 | - |
| candidates-only: `executor.vocab=248077 executor.lo=1000 executor.hi=30000 executor.reply=40 executor.end=248046 executor.deliveries=candidates` | 61 | 61 | - |
| tokens-held-sampled: `executor.vocab=248077 executor.lo=1000 executor.hi=30000 executor.reply=40 executor.end=248046 executor.deliveries=tokens executor.draft=held executor.sampled=on executor.needs_top_k=on executor.top_k_max=8 executor.masks=off` | 65 | 65 | - |
| tokens-greedy-v1: `executor.vocab=248077 executor.lo=1000 executor.hi=30000 executor.reply=40 executor.end=248046 executor.deliveries=tokens executor.abi=1 executor.masks=off` | 58 | 58 | - |
| tokens-sampled-v1: `executor.vocab=248077 executor.lo=1000 executor.hi=30000 executor.reply=40 executor.end=248046 executor.deliveries=tokens executor.abi=1 executor.masks=off executor.sampled=on executor.needs_top_k=on executor.draft=step` | 61 | 61 | - |
| rows-taken-as-tokens: `executor.vocab=248077 executor.lo=1000 executor.hi=30000 executor.reply=40 executor.end=248046 executor.draft=step` | 60 | 60 | - |
