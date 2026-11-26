# Gate of the executor ABI over the stages: qwen36-35b-a3b.flerp-serve-stage

Host control-node, written 2026-11-24T19:06:01Z. Criteria: `[criteria]` of `gate/qwen36-35b-a3b.flerp-serve-stage.toml` (committed before the first GPU run). Stage P3 of `docs/PHASE5-FOURNODE-DESIGN.md`.

| stage | result | when (UTC) | failures |
|---|---|---|---|
| check | PASS | 2026-11-24T18:58:26Z | 0  |
| build | PASS | 2026-11-24T19:06:01Z | 0  |
| conformance | PASS | 2026-11-12T01:00:06Z | 0  |
| serve1 | PASS | 2026-11-12T01:02:26Z | 0  |
| reference | PASS | 2026-11-12T01:04:28Z | 0  |
| serve2 | PASS | 2026-11-12T01:06:51Z | 0  |
| tcp | PASS | 2026-11-12T01:09:32Z | 0  |
| shutdown | PASS | 2026-11-12T01:11:38Z | 0  |

`/health` of `serve1` while idle:

```
{"status": "ok", "model": "qwen36-35b-a3b", "front_end": "serve.flerp.http", "executor": "executor.flerp.qwen35moe_abi: Qwen3.6-35B-A3B-Q4_K_M.gguf, 40 blocks, world 1, cuts 0,40, context 4096, decode graph on, one seq, greedy tokens or full rows", "executor_abi": 2, "deliveries": ["full", "tokens"], "sampled_by": "host", "sampler": "greedy (runtime.flerp.sampler_greedy: argmax of each row, temperature 0 only)", "text": "gufo Qwen tokenizer (248320 tokens) and compiled Qwen ChatML formatter; the file's template qwen-chatml-compiled-v1 sha256 e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259; /data/ark/models/Qwen__Qwen3.6-35B-A3B-GGUF/gguf/Qwen3.6-35B-A3B-Q4_K_M.gguf", "draft_form": "none", "drafts": 0, "context": 4096, "conversations_at_once": 1, "busy": false, "requests_served": 0, "slots": 1, "busy_slots": 0, "queued": 0, "context_length": 4096}
```

`serve1` SIGTERM: exit status {'0': 0} in 0.2 s

`/health` of `serve2` while idle:

```
{"status": "ok", "model": "qwen36-35b-a3b", "front_end": "serve.flerp.http", "executor": "executor.flerp.qwen35moe_abi: Qwen3.6-35B-A3B-Q4_K_M.gguf, 40 blocks, world 2 (loopback, one process), cuts 0,20,40, context 4096, decode graph on, one seq, greedy tokens or full rows", "executor_abi": 2, "deliveries": ["full", "tokens"], "sampled_by": "host", "sampler": "greedy (runtime.flerp.sampler_greedy: argmax of each row, temperature 0 only)", "text": "gufo Qwen tokenizer (248320 tokens) and compiled Qwen ChatML formatter; the file's template qwen-chatml-compiled-v1 sha256 e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259; /data/ark/models/Qwen__Qwen3.6-35B-A3B-GGUF/gguf/Qwen3.6-35B-A3B-Q4_K_M.gguf", "draft_form": "none", "drafts": 0, "context": 4096, "conversations_at_once": 1, "busy": false, "requests_served": 0, "slots": 1, "busy_slots": 0, "queued": 0, "context_length": 4096}
```

`serve2` SIGTERM: exit status {'0': 0} in 0.3 s

`/health` of `tcp` while idle:

```
{"status": "ok", "model": "qwen36-35b-a3b", "front_end": "serve.flerp.http", "executor": "executor.flerp.qwen35moe_abi: Qwen3.6-35B-A3B-Q4_K_M.gguf, 40 blocks, world 2 (TCP 127.0.0.1:39760), cuts 0,20,40, context 4096, decode graph on, one seq, greedy tokens or full rows", "executor_abi": 2, "deliveries": ["full", "tokens"], "sampled_by": "host", "sampler": "greedy (runtime.flerp.sampler_greedy: argmax of each row, temperature 0 only)", "text": "gufo Qwen tokenizer (248320 tokens) and compiled Qwen ChatML formatter; the file's template qwen-chatml-compiled-v1 sha256 e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259; /data/ark/models/Qwen__Qwen3.6-35B-A3B-GGUF/gguf/Qwen3.6-35B-A3B-Q4_K_M.gguf", "draft_form": "none", "drafts": 0, "context": 4096, "conversations_at_once": 1, "busy": false, "requests_served": 0, "slots": 1, "busy_slots": 0, "queued": 0, "context_length": 4096}
```

`tcp` SIGTERM: exit status {'0': 0, '1': 0} in 0.2 s

## Shutdown cases (dense model, TCP world 2)

- a: exit status [0, 0], failures []
- b: exit status [0, 0], failures []
- c: exit status [0, 0], failures []

## Disk reads

| run | GB | fill s | when |
|---|---|---|---|
| conformance world1 rank 0 | 21.16 | 101.11 | 2026-11-12T00:57:45Z |
| conformance loopback2 rank 0 | 21.16 | 101.197 | 2026-11-12T01:00:06Z |
| serve world1 rank 0 | 21.16 | 101.387 | 2026-11-12T01:02:26Z |
| reference rank 0 | 21.16 | 101.607 | 2026-11-12T01:04:28Z |
| serve loopback2 rank 0 | 21.16 | 101.877 | 2026-11-12T01:06:51Z |
| serve tcp2 rank 0 | 10.51 | 50.6628 | 2026-11-12T01:09:32Z |
| serve tcp2 rank 1 | 10.64 | 50.898 | 2026-11-12T01:09:32Z |
| shutdown a rank 0 | 0.84 | 17.9165 | 2026-11-12T01:10:19Z |
| shutdown a rank 1 | 0.84 | 26.1421 | 2026-11-12T01:10:19Z |
| shutdown b rank 0 | 0.84 | 18.2748 | 2026-11-12T01:10:58Z |
| shutdown b rank 1 | 0.84 | 18.0311 | 2026-11-12T01:10:58Z |
| shutdown c rank 0 | 0.84 | 18.3022 | 2026-11-12T01:11:38Z |
| shutdown c rank 1 | 0.84 | 18.167 | 2026-11-12T01:11:38Z |
| total | 131.98 | | |

## GPU holds

| for | from | to | seconds |
|---|---|---|---|
| conformance world1 | 2026-11-12T00:55:27Z | 2026-11-12T01:00:06Z | 278.8 |
| serve world1 | 2026-11-12T01:00:06Z | 2026-11-12T01:02:26Z | 140.0 |
| reference | 2026-11-12T01:02:26Z | 2026-11-12T01:04:28Z | 122.1 |
| serve loopback2 | 2026-11-12T01:04:28Z | 2026-11-12T01:06:51Z | 143.0 |
| serve tcp2 | 2026-11-12T01:07:06Z | 2026-11-12T01:09:32Z | 146.1 |
| shutdown cases | 2026-11-12T01:09:32Z | 2026-11-12T01:11:38Z | 125.4 |
