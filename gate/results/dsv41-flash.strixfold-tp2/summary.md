# Recipe gate: dsv41-flash.strixfold-tp2

Host control-node, 2026-11-11T19:39:37Z. Stages run: check, origin, build, pair.

- **check**: PASS - flerp check (origin strixfold): 35 modules, 14 recipe(s), 0 error(s), 0 warning(s)
- **origin**: PASS - 314/314 files byte-identical to git archive f64b5ff of /data/pod/plans/strixfold/fork, 0 problem(s), 1267 origin files listed as unharvested
- **build**: PASS - 314 files assembled, built in 33.3 s; kit 169 objects, manifest sha256 7693647972ab1b22... equals the reference deployment's
- **pair**: FAIL - 

Drafts on: token ids identical over 6 requests; prompt logits sha256 equal; drawn tokens by position equal; window schedule equal.

| prompt | decode t/s reference (median) | decode t/s assembled (median) | diff | within noise |
|---|---|---|---|---|
| short-chat | 60.93 | 60.22 | -1.2% | yes |
| long-chat | 63.92 | 63.0 | -1.4% | yes |
| code | 63.57 | 62.88 | -1.1% | yes |

Drafts off: token ids identical over 6 requests; prompt logits sha256 equal; drawn tokens by position equal; window schedule equal.

| prompt | decode t/s reference (median) | decode t/s assembled (median) | diff | within noise |
|---|---|---|---|---|
| short-chat | 30.82 | 30.69 | -0.4% | yes |
| long-chat | 29.97 | 29.71 | -0.9% | yes |
| code | 30.55 | 30.08 | -1.5% | yes |

Executor ABI conformance (StrixFold adapter as rank 0):
- short-chat: plain = server NO, checks None
- long-chat: plain = server NO, checks None
