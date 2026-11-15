# FLERP-SWAP: strixflerp's front end in place of the origin servers (plan)

Written 2026-11-11 by a planning agent from the laptop tree (HEAD 70c56c3) and FreeFitty (ff:, HEAD 1300a57); `[I]` marks inference. Cards on the pod board (2026-11-11): FLERP-SWAP 259695a55c, FLERP-GIT 30ad2cc982, FLERP-USAGE d2b2632afe, FLERP-ABI3 4f511cb8ad, FLERP-RECUT 3387cdf4c1, FLERP-REPLAYKIT ca92617154, FLERP-SF-PAIRPASS b390d523b9, FLERP-CACHE db829c8a0b, FLERP-CORPUS 74f780b3ac, FLERP-REPLY 8444862ff6, FLERP-SF-STATE 152c723a5e, FLERP-SF-STREAMS 2f0b2dfc19, FLERP-CONSTRAIN 2422606de0, FLERP-SELFID bab6510b88, FLERP-SF-CACHE e11fabbf9e, FLERP-SF-REPLY d1443f40e3, FLERP-VISION cb7f93a5f5, FLERP-CRITERIA 009c3634b8, FLERP-SLOTS e5b4edba91, FLERP-STEER 3efa0d5e66, FLERP-ROLL efff2c38c2, FLERP-GO 8cb3087697, FLERP-CANARY 2eaddc7d01.

## Findings while planning

- The task brief says StrixFold drafts-on "served and matched", but `gate/results/dsv41-flash.flerp-serve-tp2/summary.md` shows short-chat and long-chat as NO (165/164, 146/145) and "sampled request's tokens equal the origin's: False". `docs/EXECUTOR-ABI.md` (Results, last bullet) attributes the greedy miss to the stage counting the origin's trailing end-of-sequence id (`gate/flerp_serve_pair.py:345`). The stage has never passed.
- `gate/results/dsv41-flash.strixfold-tp2/summary.md` (19:39Z, newer than the 02:00Z conformance run) shows pair FAIL with conformance "plain = server NO, checks None".
- `steer_scope` appears nowhere in the strixflerp tree, and the catalogue's `/steer` reply has `vectors` only (`catalog/strixite/serve/steer_library/src/serve/steer_library.cpp:90`). FreeFitty sends `steer_scope` only if `GET /steer` says `"scope": true` (`ff:src/router.rs:488-510`).
- strixite gives every unseeded request a random seed (`catalog/strixite/serve/http/src/serve/http_server.cpp:466`) and FreeFitty sends no sampling fields, so recorded replies cannot be reproduced. The replay gate has to run each recorded request against both servers with an injected seed.
- strixite at 20937dc has no `/tokenize` route (StrixFold has: `catalog/strixfold/serve/http/src/zig/src/server/tokens.zig:17`).


# A. Plan

## A1. Target

A strixite worker (B, C or D), or the StrixFold pair (F+A), can be switched to a strixflerp recipe and the pod's clients cannot tell. "Cannot tell" is verified by gates: for the recorded real requests, replayed in recorded order from an empty cache with an injected seed, the strixflerp server returns the same token ids and the same response bytes (minus a declared list of volatile fields: id, created, timings) as the server it replaces, on the hardware it replaces it on.

Two standing rules:
1. A recipe's north star stays the full-precision reference (`docs/FIDELITY.md:5-10`); other engines are records. For REPLACEMENT the criterion is equivalence with the replaced server on real requests. The two do not conflict: the replacement recipes bind the origin's own executor, so their fidelity is the origin's.
2. Model data is on worker-local NVMe at serving time. Every deployment key of a replacement recipe (weights, n-gram table, tokenizer, steer-dir, prompt-cache dir, StrixFold model_dir, Engram tables) is checked to resolve under local NVMe before a roll. The existing pair gate already reads `/mnt/nv4/models/...` (`gate/results/dsv41-flash.flerp-serve-tp2/summary.md`). Commit 70c56c3 records the cost of NFS reads.

## A2. Where the front end is today

Serves `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/tokenize` (`catalog/flerp/serve/http/src/flerp/serve/main.cpp:655-690`). One conversation at a time (`main.cpp:21`, `:684`). ABI is v2 (`include/flerp/executor.h:47`).

Refusals in `catalog/flerp/serve/http/src/flerp/serve/request.cpp` (HTTP 400, code `not_served`):

| refused | line |
|---|---|
| `response_format` other than text | 57-60 |
| `grammar`, `json_schema`, `guided_*`, `structured_outputs` | 62-63 |
| `tools` (non-empty) | 64-65 |
| `tool_choice` other than none/auto | 66-67 |
| `functions`, `function_call` | 68-69 |
| `steer` | 70 |
| `cache_prompt` | 71 |
| `max_thinking_tokens`, `thinking_budget`, `reasoning.max_tokens`, `thinking.budget_tokens`, `chat_template_kwargs.thinking_budget` | 72-77 |
| `n` != 1 | 78 |
| `echo`, `suffix`, `best_of` (completions) | 80-82 |
| any non-text content part (vision) | 85-96 |
| prompt as token ids or a batch | 116 |
| logprobs on `/v1/completions`; on a streamed reply | 177, 185 |
| executor cannot honour (code `executor_cannot`): logit_bias, logprobs, temperature, top_k, top_p, min_p, seed | 189-205 |

Routes answered 501 in `main.cpp:712-721`: `/cache/export`, `/cache/import`, `/cache/probe`, `/cache`, `/steer`, `/steer/reload`, `/v1/messages`, `/v1/responses`, `/v1/embeddings`. No `/metrics` route is registered. The reply text is returned whole, with no reasoning or tool-call split (`catalog/flerp/serve/http/module.toml`, notes).

Evidence:
- strixite: 16 rows token- and step-identical, 14 requests over MTP on/off plus 2 with executor draws (`gate/results/qwen38-flash-next.flerp-serve/summary.md`).
- StrixFold: see the first two bullets at the top. After SIGTERM the process dies with SIGSEGV after `flerp_close` returned (`docs/EXECUTOR-ABI.md`, Results).
- Exported StrixFold states are refused because `flerp_state_import` carries no prefix ids (`catalog/flerp/executor/strixfold_abi/src/flerp/executor/strixfold_abi.zig:620`).

## A3. What the pod's client demonstrably uses

From FreeFitty source. This is what the client can send; what it does send on the fleet is the corpus's job.

| feature | evidence |
|---|---|
| `POST /v1/chat/completions`, always `"stream": false` | `ff:src/engine.rs:407,885,985`, `ff:src/experts.rs:269`, `ff:src/router.rs:613` |
| `tools` on every turn; `tool_choice: "none"` on refocus, removed on a 400 | `ff:src/engine.rs:407`, `:985-999` |
| tool-role messages and assistant `tool_calls` in history | `ff:src/engine.rs:152-163,290` |
| `image_url` content parts | `ff:src/window.rs:38` |
| `steer`, `steer_scope {"messages": k}`; `GET /steer` (`vectors`, `scope`) | `ff:src/experts.rs:121-126`, `ff:src/router.rs:488-527` |
| `shared_messages` on nodes flagged for it (the pair) | `ff:src/experts.rs:133`, `ff:src/config.rs:27-34`; served at `catalog/strixfold/serve/http/src/zig/src/server/chat.zig:179` |
| `/cache/export` with header `x-cache-tokens`, `/cache/import` (octet-stream), `/cache/probe` (`tokens`) | `ff:src/router.rs:395-406,449-455,477-479` |
| route detection: empty-body export answers 400 = present, 404 = absent | `ff:src/router.rs:278-281` |
| parked state files parsed as `STRXPCCH` headers (n at byte 16, size at 24, offset at 32) | `ff:src/park.rs:78-92` |
| `/health`: `status`, `slots` or `max_batch_size`, `busy_slots` or `busy`, `queued` or `live.waiting`, `context_length`, `model` | `ff:src/router.rs:698-708` |
| reply: `choices[0].message`, `finish_reason`, `usage.prompt_tokens`, `usage.prompt_tokens_details.cached_tokens` | `ff:src/router.rs:720-727` |
| 400 = rejected, anything else = transient | `ff:src/router.rs:641` |
| context overflow recognised by message substrings; steer refusal by "steer" in the message | `ff:src/engine.rs:115`, `:465` |
| no `temperature`, `top_p`, `seed` sent (grep of `ff:src`), so server defaults apply | - |

Consequence for today's front end: its `/health` has `busy` but no `slots`, `busy_slots`, `queued` or `context_length` (`main.cpp:669-686`). FreeFitty would assume 2 slots (`ff:src/config.rs:217`) and send two turns to a one-conversation server.

Offered by the origins, not called by FreeFitty: SSE streaming, `/v1/completions`, `response_format` / json_schema (strixite `catalog/strixite/serve/openai/src/serve/openai.cpp:264-282`; StrixFold `.../server/grammar.zig:56-91`), logprobs, logit_bias, reasoning-budget fields (`openai.cpp:316-320`), `/tokenize`, engine `/metrics` (`.../http_server.cpp:991`), `/steer/reload` (`:842`), StrixFold's Anthropic and responses routes, `/decisions`, `/stats`, `/dashboard`, auth (`.../server/routes.zig:42-67`).

Cannot be determined from the laptop (goes to FLERP-USAGE):
- real concurrency per strixite worker (registry SESSIONS 1, origin default `total-slots` 2 at `.../serve/server_config.cpp:247`, FreeFitty default 2);
- clients other than FreeFitty;
- live `node_meta`; whether `cache-xfer`, `steer-dir`, `logprobs` are on;
- whether the fleet build answers `scope`;
- the pair launcher's `--parallel`, drafts and learn settings;
- whether images occur in traffic.

## A4. Principle per feature

Rule: engine-independent work is written once in the front end. Model-specific text work is a bound module behind a C header (as `text.h` is today, `SPEC.md` "What sits on the ABI"). Engine state crosses the ABI as a capability. Where bytes must equal the origin's, the bound module is a shim over the origin's own code, not a rewrite. `[I]` throughout.

| feature | where | why |
|---|---|---|
| OpenAI surface: routes, JSON shapes, SSE, usage, error bodies, health keys | front end, with a per-recipe "surface profile" (key names, error wording) | Engine-independent; the two origins differ only in spelling |
| reply split (reasoning / content) and tool-call parsing | new `reply.model/flerp-v1` (`include/flerp/reply.h`); shims over `runtime.strixite.output_parser` and StrixFold `tool_parse.zig` / `reply_dsv41.zig` | Belongs to the model's template, not to the engine |
| tool-call grammar, JSON-schema constrained sampling | front end compiles (origin's `runtime.strixite.grammar`, `json_schema`, `token_mask`, `structured_output`) to per-row masks: existing `FLERP_CAP_MASKS` (`executor.h:74,142`). Executors that draw on the device get the constraint text through the ABI | strixite hands rows, so masks suffice. StrixFold compiles a grammar into the stream and shows no logits (`docs/EXECUTOR-ABI.md`, "Logits delivery") |
| reasoning budget, nudge, end guard, close-on-call | front end decode-loop policy (fed ids already exist: `decode_loop.h:92`) | It is `Engine` policy in strixite, not backend calls |
| request scheduling: slots, queue, verify packing | front end, over `caps.seqs` and `FLERP_CAP_VERIFY_BATCH` | The arbiter is above the backend (`docs/EXECUTOR-ABI.md`, "Cross-slot verify batching") |
| prompt cache bookkeeping, `/cache` routes, file format | new `cache.prompt/flerp-v1` (`include/flerp/prompt_cache.h`); shims over each origin's `runtime.*.prompt_cache` | Keys, tiers and the `STRXPCCH` layout the client parses are host code |
| kept / exported / learned / shared state | ABI capability | Engine state |
| drafts | ABI capability (exists: `DRAFT_STEP`, `DRAFT_HELD`) | Engine state |
| steering vectors | ABI capability; library, routes, scope arithmetic and cache-key salt in the front end | Vectors live on the device; names and scopes are host |
| image rows | ABI capability; picture preparation and span placement in the front end; tower as its own module where rows can be handed in (`docs/VISION-DSV41.md`, option 2) | Rows are engine input |

## A5. ABI v3 sketch

Append-only, each item behind a flag; v2 callers and executors keep working (rule at `docs/EXECUTOR-ABI.md`, "Version 2"). `[I]`

1. Slots. `FLERP_CAP_SEQ_BATCHABLE`: `flerp_seq_batchable(ex, seq)` (a seq with image rows verifies alone). `FLERP_CAP_PREFETCH`: `flerp_prefetch(ex, seq, ids, n)` (strixite `prefetch_ple`). `FLERP_CAP_RENEW`: `flerp_seq_renew` / `flerp_seq_can_renew`. `caps.seqs` and `max_verify_rows` exist.
2. State with prefix ids. `flerp_state_desc {size, tokens, from, prefix_ids, n_prefix, fingerprint}`. `FLERP_CAP_STATE_PREFIX`: `flerp_state_export2(ex, seq, snap, from, sink, ctx)` and `flerp_state_import2(ex, seq, desc, source, ctx)`, streamed because states are 0.5 to 1.3 GB. `caps.keep_step` (2,048 for dsv41, 0 = anywhere). `FLERP_CAP_SNAPSHOT_MARK`: `flerp_snapshot_mark(ex, seq, at, &snap)`, taken by the prompt pass at a chunk end. `FLERP_CAP_SNAPSHOT_SHARED`: `flerp_snapshot_share(ex, snap)`, restorable into any seq (today a snapshot belongs to its seq, `executor.h:241`); this carries `shared_messages` and learned states. `FLERP_CAP_STATE_PARK`: `flerp_seq_park` / `flerp_seq_resume` (strixite paged and small states).
3. Constraint for device-side draws. `FLERP_CAP_CONSTRAINT`, `caps.constraints` (json_schema, regex, choice, grammar, tool_call_required), `flerp_seq_args.constraint {kind, text}`. `FLERP_CAP_SEQ_POLICY`: `flerp_seq_args.{eos, think_budget, think_close, loop_guard}` for executors whose lane owns them.
4. Steering. `FLERP_CAP_STEER`: `flerp_steer_load(ex, name, layer, f32, dim, &h)`, `flerp_steer_drop`, `flerp_seq_steer(ex, seq, items, n)`. `FLERP_CAP_STEER_SPAN`: items carry `[from, to)` positions (`steer_scope`).
5. Images. `flerp_image_span {at, tokens, grid_t, grid_h, grid_w, rows, payload}` on a prompt forward: `flerp_forward_spans(...)`. `FLERP_CAP_IMAGE_ROWS` (finished rows handed in), `FLERP_CAP_IMAGE_PREPARE` (`flerp_image_prepare`, executor's own tower). Caps gain image token id, hidden width, row bounds.
6. Counters for `/health` and `/metrics` travel in `flerp_stats_json` (no new call).

## A6. The replay corpus (centre of the plan)

Record: a pass-through recorder between FreeFitty and one node at a time. It logs request body, status, response headers and body, arrival and finish times, and concurrent requests in flight. For `/cache/export` and `/cache/import` it logs sizes and sha256 only. Two zero-intrusion sources exist, both partial:
- strixite `--capture-dir` gives token ids and steps but no body or sampling (`catalog/strixite/runtime/capture/src/serve/capture.hpp:1-17`);
- StrixFold `TENSORFOLD_REQUEST_LOG` gives bodies with images redacted, so it is unusable for vision replay (`.../server/request_log.zig:1`).

Scrub: the corpus holds conversation contents. It stays on the pod (mode 700), never in git; the files themselves say so (`capture.hpp:6`).
- Strip auth headers and keys.
- Keep `data:` images; drop remote image URLs (the request then counts as "not replayable").
- Git gets a manifest only: request ids, sha256, field census, token counts, conversation order.

Replay rules `[I]`:
- conversation order preserved;
- both servers start from an empty prompt cache (a cache hit moves the prompt cut, and for strixite the forward shape is part of the numbers: `docs/EXECUTOR-ABI.md`, "Forward shape");
- seed injected from the request id where absent;
- origin-against-origin is run first. Any request class the origin does not reproduce against itself gets its criterion from owner (FLERP-CRITERIA), not from the implementer.

## A7. Order and gates

Phase 0 (parallel, no feature work): FLERP-GIT, FLERP-USAGE, FLERP-RECUT, FLERP-REPLAYKIT, FLERP-SF-PAIRPASS, FLERP-ABI3. Then FLERP-CORPUS, FLERP-SELFID, FLERP-CRITERIA.

Phase 1, strixite worker (parallel after RECUT): FLERP-REPLY, then FLERP-CONSTRAIN; FLERP-SLOTS; FLERP-CACHE, then FLERP-STEER; FLERP-VISION.

Phase 2, pair (parallel after SF-PAIRPASS and ABI3): FLERP-SF-STATE, then FLERP-SF-CACHE; FLERP-SF-STREAMS; FLERP-SF-REPLY.

Phase 3: FLERP-ROLL, FLERP-GO, FLERP-CANARY.

Gate per piece, always in two layers:
- no GPU: `tests/` over `executor.flerp.fake_abi`;
- on hardware: the corpus slice that exercises the feature, identical token ids and identical bytes against the origin server, via `./flerp gate recipe <recipe> --stages replay`.

Specific gates:
- Cache transfer: a state exported by the native server imports into the flerp server and back, by file, on one worker.
- Slots on strixite: criterion from FLERP-CRITERIA.
- Streams on StrixFold: identity, since drafts are output-invariant.
- Client-level: FreeFitty's own `cargo test --test real_nodes -- --ignored` (`ff:README.md:438`) against a flerp node.

## A8. Rollout

- Registry entry beside `strixite_v070pod` in `/data/pod/engines.conf`, naming the assembled binary, its sha256, the recipe and commit.
- Stage the build and check every data path is on local NVMe.
- One strixite worker first, in a low-traffic window; FreeFitty keeps the thread-to-node mapping.
- Because cache files are origin-format, threads move both ways between native and flerp nodes.
- Rollback = the registry's previous entry. It is rehearsed before the canary and timed.
- The keeper needs: `/health` with the origin's keys and `status` values, the same port, the same stop signal with exit status 0, and a `/metrics` it can scrape.
- The pair goes last and only in a window agreed with its owner.

## A9. Out of scope

- New kernels or executors; the Ornith four-node work (`docs/PHASE5-FOURNODE-DESIGN.md` 3.9 keeps its own caps: one seq, no drafts, no snapshots).
- Making the two executors' states interchangeable.
- Sampled-output equality across executors (`SPEC.md` section 6, Deliveries).
- StrixFold's Anthropic, responses, decisions, compaction, dashboard and auth routes, unless FLERP-USAGE finds a caller.
- DeepSeek vision on HIP (`docs/VISION-DSV41.md` obstacles).
- `/v1/embeddings`, `n > 1`.
- Changing FreeFitty.
- Performance beyond "within noise of the origin".

# C. Dependency sketch

```
FLERP-GIT ──► RECUT ──► REPLY ──► CONSTRAIN ─┐
   │            ├──────► SLOTS (also CRITERIA)┤
   │            ├──────► CACHE ──► STEER      ├─► ROLL ──► GO ──► CANARY
   │            └──────► VISION (ABI3, CORPUS)┘
   ├──► REPLAYKIT ──► CORPUS ──► SELFID ──► CRITERIA   (USAGE feeds CORPUS, CRITERIA, ROLL, SF-STREAMS)
   ├──► ABI3 ──► STEER, VISION, SF-STATE
   └──► SF-PAIRPASS ──► SF-STATE ──► SF-CACHE
                   ├──► SF-STREAMS
                   └──► SF-REPLY (after REPLY)
```

- Start in parallel on day one: FLERP-GIT, FLERP-USAGE (no dependency). Right after GIT: RECUT, REPLAYKIT, ABI3, SF-PAIRPASS. Of these four, only RECUT (control GPU) and SF-PAIRPASS (pair window) need hardware.
- Control-GPU cards are serialised by the GPU queue: RECUT, SELFID, REPLY, CONSTRAIN, SLOTS, CACHE, STEER, VISION.
- Pair-window cards, each agreed with the pair's owner: SF-PAIRPASS, SF-STATE, SF-CACHE, SF-STREAMS, SF-REPLY. They can share windows; conformance already rides the StrixFold recipe gate's window.
- CRITERIA blocks only SLOTS and STEER (and VISION's priority). REPLY, CONSTRAIN and CACHE run one conversation at a time, where identity is already the demonstrated criterion.
- Count: 1 epic, 20 work/research cards, 2 decide cards.

### Critical Files for Implementation
- /home/user/Code/strixflerp/include/flerp/executor.h
- /home/user/Code/strixflerp/catalog/flerp/serve/http/src/flerp/serve/request.cpp
- /home/user/Code/strixflerp/catalog/flerp/serve/http/src/flerp/serve/main.cpp
- /home/user/Code/strixflerp/docs/EXECUTOR-ABI.md
- /home/user/Code/strixflerp/catalog/flerp/executor/strixfold_abi/src/flerp/executor/strixfold_abi.zig
- (client evidence) /home/user/Code/freefitty/src/router.rs
- (origin behaviour to match) /home/user/Code/strixflerp/catalog/strixite/serve/http/src/serve/http_server.cpp