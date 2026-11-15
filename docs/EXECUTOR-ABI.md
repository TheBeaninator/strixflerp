# The executor ABI (`include/flerp/executor.h`, version 3)

SPEC section 6 asks for one C ABI both executors can sit behind. This file maps every call of the two real
interfaces onto it and names what does not map. Sources read:

- strixite v0.7.0-pod (20937dc): `LmBackend`, `catalog/strixite/runtime/engine/src/serve/engine.hpp` lines 101-417,
  and how `Engine` drives it (`serve/engine.cpp`, the decode loop around line 1533).
- StrixFold f44d736: the lane backend vtable, `catalog/strixfold/runtime/lanes/src/zig/src/core/lanes/backend.zig`;
  the engine a server drives, `.../core/engine_api.zig`; the dsv41 implementation,
  `catalog/strixfold/executor/dsv41/src/zig/src/families/dsv41/lanes.zig` and `native.zig`; the kept-state
  functions, `catalog/strixfold/runtime/prompt_cache/src/zig/src/core/prompt_cache.zig` (`Snapshots`).

The adapters are modules under origin `flerp`:

| module | what | built by |
|---|---|---|
| `executor.flerp.strixite_abi` | C++ shim over `LmBackend`, opens `Qwen4ExpBackend` | `recipes/qwen38-flash-next.strixite-abi.toml` (CMake) |
| `executor.flerp.strixfold_abi` | Zig `export fn`s over the lane backend and `native.Snaps`, plus `flerp.build.zig` | `recipes/dsv41-flash.strixfold-tp2-abi.toml` (the origin's build graph and the step `flerp-abi`) |
| `executor.flerp.conformance` | the C driver that uses only `executor.h` and the shared decode loop; linked with either adapter | both |
| `executor.flerp.fake_abi` | a stand-in executor with no model (the next token is a hash), whose deliveries, draft form, sampled draws and ABI version are chosen at `flerp_open`: for testing callers without a GPU | `recipes/qwen38-flash-next.flerp-serve-fake.toml`, `tests/test_flerp_serve.py` |
| `runtime.flerp.decode_loop` | `include/flerp/decode_loop.h`: the decode loop over the ABI that the conformance driver and the front end both run | every recipe with either |

## The one difference that shapes the ABI

strixite hands **logits** to the host (full rows, or the top 20 candidates of each row reduced on the GPU) and the
host samples. StrixFold **draws on the device**: a greedy row comes back as its argmax token, a sampled row is drawn
by the keyed sampler at the row's absolute position, and logits never cross the lane backend vtable. The ABI
therefore has three deliveries (`FLERP_OUT_FULL`, `FLERP_OUT_CANDIDATES`, `FLERP_OUT_TOKENS`) and each executor says
which it offers. A front end that wants to serve either must take `FLERP_OUT_TOKENS` for greedy requests (both
offer it) and, for sampled requests, either sample itself from strixite's rows or let StrixFold draw
(`FLERP_CAP_SAMPLED_TOKENS`): the two samplers are not the same function, so sampled output is per executor.

`serve.flerp.http` is that front end. It decides once, at startup, from `caps.deliveries`: where rows are offered
it samples on the host with the recipe's `sample.host/flerp-v1/-` module (candidates when the sampler needs only
the top 20, full rows otherwise and for logprobs), which for strixite is the origin's own `Sampler` and so the
origin's draws; where only tokens are offered it passes the request's sampling to `flerp_seq_new` and takes the
executor's draws. `--delivery tokens` makes it take the executor's draws even where rows are offered.

## Version 2

Version 2 appends to version 1; a version 1 caller and a version 1 executor keep working with a version 2 peer.

| added | why the front end needs it |
|---|---|
| `flerp_caps.sampling` (`FLERP_SAMPLE_TOP_K`, `_TOP_P`, `_MIN_P`, `_SEED`, `_NEEDS_TOP_K`) and `flerp_caps.top_k_max` | An executor that draws on the device has its own sampler with its own limits: StrixFold's dsv41 samples only with `top_k` set, at most 56 (`families/dsv41/native.zig`: `needs_top_k`, `top_k_max`; its server refuses the request, `server/server.zig` 191). Version 1 could only fail at `flerp_seq_new`. With the fields the front end answers HTTP 400 naming the parameter before anything runs; against a version 1 executor it passes the request on and returns the executor's refusal |
| the rule for a struct the executor fills | `flerp_get_caps` writes at most the caller's `size` bytes and returns the `size` it knows, so the appended fields are zero and marked absent for a version 2 caller of a version 1 executor, and a version 1 caller is never written past its struct (both adapters refused such a caller before) |

strixite's adapter reports `sampling = 0` (it draws greedy tokens only; the host samples). StrixFold's reports its
sampler's parameters, `NEEDS_TOP_K` and the bound; that code is built and has not run.

What the front end did not need, and version 2 therefore does not have:

- **Stop conditions.** The end-of-turn ids come from the model's text module and stop strings are matched on the
  host; the loop simply stops calling. StrixFold's `Spec.eos` reaches only its grammar and its telemetry.
- **Grammars** (StrixFold's `Spec.structure`) and **the prefix token ids of a returning StrixFold state.** The
  front end serves neither structured output nor a prompt cache, so nothing would have exercised the calls.
  Both remain what is listed under "Not mapped from StrixFold".
- **Sampling parameters for device-side draws.** Version 1 already carries them (`flerp_seq_args.sampling`); the
  front end opens a seq per request.

Not in the ABI and a possible cost: strixite's engine starts loading a draft's n-gram rows while the next draft
runs (`prefetch_ple` between `forward_mtp_top2` calls). `flerp_draft_step` has no such hint.

## Version 3

Version 3 appends to version 2 (FLERP-ABI3, for the FLERP-SWAP plan `docs/FLERP-SWAP-PLAN.md` A5): engine state the
front end must pass to an executor to replace the origin servers. Nothing in versions 1 and 2 changes. Rules:

- Every addition is behind a `FLERP_CAP_*` flag. An executor that reports `abi >= 3` exports every version 3 call and
  answers `FLERP_E_UNSUPPORTED` where its flag is absent; a caller calls none of them unless `caps.abi >= 3` and the
  flag is set.
- `flerp_caps` gains fields after `top_k_max`; the size rule is version 2's (written for at most the caller's `size`,
  returned `size` = what the executor knows; `FLERP_CAPS_V2_SIZE` is the version 2 struct).
- `flerp_seq_args` gains fields after `drafts`; an executor reads them only as far as the caller's `size`
  (`FLERP_SEQ_ARGS_V2_SIZE` is the version 2 struct). A version 2 executor never reads them, so a version 3 caller
  that sets them on a version 2 executor gets them ignored: it checks the flag first.
- The real adapters (`strixite_abi`, `strixfold_abi`, `qwen35moe_abi`) report version 2 and fill only the version 2
  struct until their own cards implement the calls (FLERP-CACHE, FLERP-STEER, FLERP-VISION, FLERP-SF-STATE).
  The conformance driver accepts version 2 or 3.

| addition | strixite (`catalog/strixite/runtime/engine/src/serve/engine.hpp`) | StrixFold | why the front end needs it |
|---|---|---|---|
| `FLERP_CAP_STATE_PREFIX`: `flerp_state_desc {tokens, from, prefix_ids, n_prefix, fingerprint, bytes}`, `flerp_state_export2` (to a `flerp_sink`), `flerp_state_import2` (from a `flerp_source`) | `export_snapshot(slot, out, from)`, `snapshot_bytes`, `import_state(state, n, from)`: the prefix ids are not needed and may be passed | `state_file.zig` / `cache_routes.zig` (`/cache/export`, `/cache/import`, fork d21ae2e, f64b5ff): a returning state needs its prefix ids for the resuming prompt pass, which `flerp_state_import` cannot carry (`strixfold_abi.zig:620` refuses) | the prompt cache and `/cache/*` routes FreeFitty calls (`ff:src/router.rs:395-479`); streamed because a state is 0.5 to 1.3 GB |
| `caps.keep_step` | 0 (`FLERP_CAP_SNAPSHOT_ANYWHERE`) | 2,048: a state is kept only at a prompt chunk end (`lanes.zig snapSave`) | where the front end may ask for a kept state |
| `FLERP_CAP_SNAPSHOT_MARK`: `flerp_snapshot_mark(seq, at)` | `save_snapshot` at the turn point, after the forward that reaches it | `Snapshots.save(owner, at)` recorded as a mark and taken by the prompt pass's hook (`Reuse.marks`, `MarkHook`) | StrixFold cannot save after the fact; a mark is the one form both take |
| `FLERP_CAP_SNAPSHOT_SHARED`: `flerp_snapshot_share`, restored into any seq, dropped with seq -1 | snapshots are per slot; a shared one is a host copy imported into another slot | `Stream.reuse` of a kept state by another stream; `shared_messages` (`server/chat.zig:179`) and learned states | `shared_messages` on the pair (`ff:src/experts.rs:133`) |
| `FLERP_CAP_STATE_PARK`: `flerp_seq_park`, `flerp_seq_resume` | `park_snapshot`, `resume_paged`, `release_parked`, `small_states`, `resume_small` (K/V parked as its pages) | none (dsv41 states are written to files instead) | strixite's cheap conversation switch between slots |
| `FLERP_CAP_STEER`: `flerp_steer_load(name, layer, v, dim)`, `flerp_steer_drop`, `flerp_seq_steer(items)`; `caps.steer_dim`, `.steer_layers` | `set_steer(seq, SteerSet)`; vectors from `serve/steer_library` | none | `steer` in FreeFitty's requests (`ff:src/experts.rs:121-126`) |
| `FLERP_CAP_STEER_SPAN`: `flerp_steer_item.from / .to` | `steer_scope` is not in the catalogue (`/steer` answers `vectors` only, `steer_library.cpp:90`); it was built in strixite's v070sp test server, not rolled to the fleet (2026-11-09, card STEER-PREFIX 07f2a43421) | none | `steer_scope {"messages": k}`, sent only when `GET /steer` says `"scope": true` (`ff:src/router.rs:488-510`) |
| `FLERP_CAP_IMAGE_ROWS`, `FLERP_CAP_IMAGE_PREPARE`: `flerp_image_span {at, tokens, grid_t/h/w, rows, payload}`, `flerp_forward_spans`, `flerp_image_prepare`; `caps.image_token`, `.image_hidden`, `.image_max_rows`, `.images` | `forward_image_prefill(ids, PrefillImage)`: one image a prefill forward, mRoPE triples from the grid | `Stream.images` spans with a payload `Vision.prepare` made (`core/lanes/stream.zig:112-121`), or finished rows (`Image.rows`, `lanes.zig:698-700`); options 1 and 2 of `docs/VISION-DSV41.md`, "Behind strixflerp's executor ABI" | `image_url` content parts (`ff:src/window.rs:38`) |
| `FLERP_CAP_SEQ_BATCHABLE`: `flerp_seq_batchable` | `batchable(seq)`: a seq with image rows verifies alone | every stream may share a round | the front end's verify packing (`GpuArbiter` work moves above the ABI) |
| `FLERP_CAP_CONSTRAINT`: `flerp_seq_args.constraint {kind, text}`; `caps.constraints` (`json_schema`, `regex`, `choice`, `grammar`, `tool_call_required`) | not offered: rows go to the host and the front end masks them (`FLERP_CAP_MASKS`) | `Spec.structure` compiled into the stream (`grammar.Constraint`, `server/grammar.zig:56-91`); draws on the device | `response_format`, tools with `tool_choice` on an executor that shows no logits |
| `FLERP_CAP_SEQ_POLICY`: `flerp_seq_args.eos`, `.think_budget`, `.think_close`, `.loop_guard` | not offered: `Engine` policy, the front end's (`decode_loop.h`) | `Spec.eos`, `think_budget`, `think_close`, `loop_guard` (the lane core forces the close text in its rounds) | the reasoning budget and stop where the lane owns them; a forward that feeds an end id after the prompt is refused |
| `FLERP_CAP_PREFETCH`: `flerp_prefetch(seq, ids)` | `prefetch_ple(seq, ids, first)` between draft steps | none | the cost noted under "Version 2" |
| `FLERP_CAP_RENEW`: `flerp_seq_renew`, `flerp_seq_can_renew` | `renew`, `can_renew` (soft K/V reset; `kv_renewals`) | none | long-running slots on strixite |
| counters for `/health` and `/metrics` | `backend_stats()` | drafter telemetry | no new call: `flerp_stats_json` |

Not in version 3: StrixFold's `think_end` / `think_open`, `stop_check`, `cancel_check`, `proposer`, `chunks`; draft
trees and probabilities; the prompt cache's own bookkeeping. The prefix ids make StrixFold states importable through
the ABI; they do not make the two executors' states interchangeable.

`executor.flerp.fake_abi` implements every version 3 call (`features=state,steer,image-rows,image-prepare,constraint,
seq-policy,slots` or `all`, at `flerp_open`; `abi=2` makes it a version 2 executor). Its state is the ids it covers,
so an import with the wrong prefix ids, fingerprint or position is refused as StrixFold would. `tests/test_flerp_serve.py`
(class `Version3`, program `tests/cpp/fake_abi_v3_selftest.c`) runs every call, a version 2 caller built against the
frozen version 2 header (`tests/cpp/abi_v2/`) on the version 3 executor, and a version 3 caller on the frozen version 2
executor.

## strixite: `LmBackend` onto the ABI

| `LmBackend` | ABI | notes |
|---|---|---|
| `capacity()`, `max_chunk()`, `logits_row()`, `max_verify_rows()`, `n_seqs()`, `has_mtp()`, `has_vision()` | `flerp_get_caps` | `positions`, `max_forward`, `logits_row`, `max_verify_rows`, `seqs`, `FLERP_CAP_DRAFT_STEP`, `FLERP_CAP_VISION`; `candidates` = `Sampler::kCandidates` (20) |
| `max_positions()` | not mapped | the model's position ceiling; `positions` is the session's capacity |
| constructor of `Qwen4ExpModel` + `Qwen4ExpBackend` (`tools/strix_server.cpp`) | `flerp_open` | keys `weights`, `ngram`, `capacity`, `chunk`, `mtp`, `mtp_vocab`, `kv_cache`, `gdn_precision`, `attention_precision`, `rope_yarn_factor`, `seqs`, `kv_generations`, `ngram_cache_rows`, `verify`; defaults are the served ones |
| `pos()`, `pos(seq)` | `flerp_seq_pos` | |
| `reset()`, `reset(seq)` | `flerp_seq_new`, `flerp_seq_reset`, `flerp_seq_free` | seqs are fixed slots; new = the first free one, reset |
| `forward(ids, want_logits)`, `forward(ids, n_logits)`, `forward_rows(ids, n_logits, cands, n_valid, masks)`, `forward_rows(seq, ...)` | `flerp_forward` | `want.rows` = `n_logits`; `FLERP_OUT_FULL` / `FLERP_OUT_CANDIDATES` = `cands`; `FLERP_OUT_TOKENS` = candidate 0 (greedy only) |
| `LogitMasks` (bits and `LogitBias`) | `flerp_masks` | |
| `set_lookahead(next_ids)` | `flerp_forward(..., lookahead, n_lookahead)` | a hint |
| `forward_mtp(token, step)` | not mapped | the head's full logits; nothing in the engine uses them besides `forward_mtp_top2`'s default |
| `forward_mtp_top2(token, step)`, `(seq, ...)` | `flerp_draft_step` | `Top2.best`, `best_v`, `second_v` → `flerp_draft` |
| `forward_verify(ids, n_logits)`, `forward_verify_rows(...)` | `flerp_verify` with one run | |
| `verify_batch(runs, cands, n_valid)` | `flerp_verify` with several runs | `FLERP_CAP_VERIFY_BATCH` when `seqs > 1` |
| `batchable(seq)` | not mapped | see below |
| `keep_verify()`, `keep_verify(seq, rows, carry)` with all rows | `flerp_keep` | |
| `keep_verify_prefix(ids, rows)`, `keep_verify(seq, rows, carry)` with fewer | `flerp_keep_prefix(seq, rows, carry)` | `FLERP_CAP_KEEP_PREFIX` |
| `drop_verify()`, `drop_verify(seq)` | `flerp_drop` | |
| `save_snapshot(slot)`, `snapshot_pos`, `can_restore_snapshot`, `restore_snapshot` (and the `seq` forms) | `flerp_snapshot_save`, `_pos`, `_restore`, `_drop` | slots `kTurnSlot`, `kSystemSlot`, `kUserSlot` become three handles; `kDraftSlot` stays the verify's; `can_restore` false = `FLERP_E_STATE` |
| `export_snapshot(slot, out, from)`, `snapshot_bytes`, `import_state(state, n, from)` | `flerp_snapshot_export`, `flerp_snapshot_bytes`, `flerp_state_import` | one copy more than strixite's own path (its `HostBuffer` to the caller's buffer) |
| `state_fingerprint()`, `describe()`, `failed()` | `flerp_state_fingerprint`, `flerp_describe`, `flerp_failed` | |
| `backend_stats()` | `flerp_stats_json` | six of its counters; `flerp_get_stats` counts the ABI's own calls |

Not mapped from strixite:

- **The draft chain is uninterruptible and its acceptance rule lives in the engine.** `forward_mtp_top2(token, step)`
  keeps the chain's state inside the session between steps: the steps of one chain are consecutive calls and nothing
  else may run on that seq until its verify. The ABI states the rule (`flerp_draft_step`) but cannot check it; the
  margin rule (`mtp-margin`), the stop at end-of-turn, the vision ban and the structured-output filter that decide
  how long a chain gets are `Engine` code (`engine.cpp` 1556-1576), not backend calls, and stay on the caller's side.
- **Cross-slot verify batching is only partly expressible.** `flerp_verify` takes several runs, which is
  `verify_batch`. What is missing is the arbiter around it (`GpuArbiter`, `serve/seq_backend.hpp`): it packs the
  runs of whichever slots are decoding into the 16-row budget, lowers each slot's draft depth to make them fit
  (`max_verify_rows / decoding slots - 1`) and splits a batch in two forwards. That is scheduling above the backend;
  a front end over the ABI has to do it again. `batchable(seq)` (a conversation with image rows verifies alone) has
  no ABI call; an executor that needs it must refuse the batched run.
- **Parked and small states**: `paged_states`, `park_snapshot`, `resume_paged`, `release_parked`,
  `park_budget_left` (a conversation's K/V parked as the physical pages it occupies) and `small_states`,
  `small_snapshot_bytes`, `export_small_snapshot`, `resume_small` (a resume point without K/V rows). These are what
  make strixite's conversation switch cheap; version 1 has only whole states and deltas as host bytes.
- **`renew` / `can_renew`** (the soft K/V reset that rebuilds the session), **`set_steer`** (activation steering),
  **`prefetch_ple`** (start loading n-gram rows for ids the next forward will run), **`forward_image_prefill`** and
  `PrefillImage` / `VisionPrefill` (one image a prefill forward, mRoPE triples). No call in version 1.

## StrixFold: the lane backend, `native.Loaded` and `Snapshots` onto the ABI

| StrixFold | ABI | notes |
|---|---|---|
| `native.open(gpa, io, ctx, dir, kernels, options)` | `flerp_open` | keys `model_dir`, `kit`, `context`, `drafts`; everything else from the environment as the server's rank 0 has it; links rank 1 |
| `Loaded.deinit` | `flerp_close` | rank 1 hears done |
| `Loaded.facts` (`lanes.Model`), `.rows`, `.context_window`, `.snaps`, `.vision` | `flerp_get_caps` | `seqs` = `max_streams`, `max_verify_rows` = a round's rows, `max_draft` = `facts.drafts`, `FLERP_CAP_DRAFT_HELD`, `FLERP_CAP_SNAPSHOT` |
| `Stream.init(spec)` | `flerp_seq_new` and the first forwards | `sampling`, `max_new`, `drafts` from `flerp_seq_args`; the prompt is gathered from the forwards |
| `Backend.prefill(stream)` + `first(stream, prompt_len)` + `read(handle)` | `flerp_forward` (prompt) with `want.rows = 1`, `FLERP_OUT_TOKENS` | forwards with `rows = 0` only gather ids: StrixFold runs a prompt as one pass and cuts its own chunks |
| `Backend.queue(stream, feed, position)` | not mapped | dsv41 returns `NotPipelined`; a one-token forward is a one-row window |
| `Backend.verify(windows, out)` | `flerp_verify`; `flerp_forward` after the prompt (a window whose rows are all kept) | `Window.pending` = `ids[0]`, `Window.tokens` = `ids[1..]`, `Window.held` = `run.held`, positions from the seq; `Verified.sampled` → `out.tokens`, `Verified.drafts` → `held_out` |
| `Backend.keep(windows, paths)` | `flerp_keep`, `flerp_keep_prefix`, `flerp_drop` | a path is a prefix count; dsv41's caches are positional, so the adapter moves `stream.cache_len` and a drop moves nothing |
| `Backend.draft(requests)` | `flerp_draft_hold(seq, pending, depth)` | the request's `follow`, `rows`, `start`, `position` come from the last kept window |
| `Backend.release(stream)` | `flerp_seq_reset`, `flerp_seq_free` | |
| `Snapshots.save(owner, at)` | `flerp_snapshot_save` | only inside a prompt pass, at a chunk end: the save is recorded as a mark and taken by the pass's hook (`Reuse.marks`, `MarkHook`) |
| `Snapshots.restore` (dsv41: `error.BackendRestores`), `Stream.reuse` | `flerp_snapshot_restore` on a reset seq | the next prompt pass starts from the kept state |
| `Snapshots.drop`, `Snapshots.bytes` | `flerp_snapshot_drop`; `bytes` not mapped | |
| `Snapshots.write`, `read`, `forget` | `flerp_snapshot_write`, `flerp_snapshot_read` | declared in the ABI (`FLERP_CAP_SNAPSHOT_FILES`); dsv41 has none, so its adapter refuses |
| `Engine.vtable.info` (`engine_api.Info`) | `flerp_get_caps`, `flerp_describe` | the serving facts (`efforts`, `prefill_step`, `top_k_max`, `needs_top_k`, call gates) are the front end's |

Not mapped from StrixFold:

- **Rounds.** The lane core (`lanes/engine.zig`) owns the schedule: which streams share a round, the depth rule that
  prices draft depths from measured costs (`depth.zig`, `plan_lanes.zig`), the draft bypass under load, copied
  windows from the suffix lookup, forced tokens for the thinking budget. Through the ABI a caller can run the same
  forwards (several runs a `flerp_verify`, `flerp_draft_hold` with a depth), but the policy is not behind the ABI,
  and `engine_api.Engine` (`submit`, `cancel`, `status`, events to a sink) is a serving loop, not an executor call.
  A front end over the ABI would replace `LaneHost`, not sit on it.
- **Handles and pipelining.** `first` and `queue` return a handle; `read` waits for the GPU. The ABI is synchronous:
  a token is returned by the call that draws it. dsv41 is not pipelined (`gpu_tokens = false`), so nothing is lost
  there; a pipelined family (Flash Next on Metal) would lose its one-step-ahead rounds.
- **Snapshot and park forms.** A dsv41 state is kept only while the prompt pass stands at a chunk end (a multiple of
  2,048 tokens, at least a window before the prompt's end, `lanes.zig snapSave`) and is restored only into a new
  prompt pass; it cannot be taken during decode. Since fork commits d21ae2e and f64b5ff the family can write kept
  states to disk (`state_file.zig`) and hand them out as bytes (`cache_routes.zig`: `/cache/probe`, `/cache/export`,
  `/cache/import`), but the ABI does not map that: a returning state needs the token ids of its prefix for
  the resuming prompt pass, and `flerp_state_import` / `flerp_snapshot_read` carry none. The adapter therefore still
  reports `FLERP_CAP_SNAPSHOT_ANYWHERE` and `FLERP_CAP_SNAPSHOT_BYTES` absent and refuses with `FLERP_E_STATE`; the
  ABI does not make the two executors' states interchangeable in a prompt cache. The prompt cache's own bookkeeping
  (`charged`, `spare`, `trim`, `reuses`, the budget) has no ABI call.
- **Logits delivery.** No full rows and no candidates: `FLERP_OUT_FULL` and `FLERP_OUT_CANDIDATES` are refused.
  Masks are also not expressible: StrixFold constrains draws with a grammar compiled into the stream
  (`Spec.structure`, `grammar.Constraint`), not with per-row allowed-token masks, so `flerp_want.masks` is refused
  (`FLERP_CAP_MASKS` absent) and structured output does not cross the ABI in version 1.
- **Trees, alternatives, probabilities, early speculation.** `Window.parents` and `DraftRequest.lanes` (a tree of
  drafts), `alternatives`, `tree`, `probabilities` (each held draft's chance of landing, which the depth rule
  reads) and `unspeculate` / `Window.early`. dsv41 builds chains only and has no early speculation; the ABI has
  chains only.
- **Vision.** `Stream.images` (`Image` spans whose payload the family's `Vision.prepare` made) and
  `Loaded.vision`. HIP builds have no tower (`gpu.features.vision = false`); the ABI has a capability flag and no
  call that carries an image, for either executor.
- **Stream policy fields**: `eos`, `stop_check`, `cancel_check`, `think_budget`, `think_close`, `loop_guard`,
  `proposer`, `chunks`. The adapter leaves them empty: stopping is the caller's.
- **Telemetry**: a stream's drafter counters are written at its release (`Stream.telemetry_buf`); the adapter's
  `flerp_stats_json` returns `{}` until a seq has been released, and `engine_api.Stats` (rounds, drafted, accepted)
  is counted by the adapter's own `flerp_stats` instead.

## Forward shape

The ABI lets the caller decide how ids are grouped into forwards: where a prompt is cut, how many drafts ride on
a verify, how many rows of it are kept. For strixite that grouping is part of the numeric contract.

- `Qwen4ExpSession` picks its kernels by the rows a forward carries: one row is the decode path, 6 rows and more
  take the grouped expert kernels (`kGroupedMinTokens`), 12 and more the WMMA prefill kernels (`kWmmaMinTokens`).
  Its header says what follows (`runtime/qwen4exp.hpp` 800-805): a run's results are bit-identical next to the
  same run alone "when the batch's total rows and the run's own fall on the same side of 1, kGroupedMinTokens and
  kWmmaMinTokens, otherwise they differ by rounding".
- A rounding difference flips a greedy token at a near tie, and the text is different from there on. The reference
  server does this to itself: the same request with MTP on (2 to 8 rows a verify) and off (one row a forward)
  parts at token 43 of `raw-completion` and token 95 of `short-chat-thinking`
  (`gate/results/qwen38-flash-next.strixite/identity.json`, `reference-mtp-on` against `reference-mtp-off`).
- So a serving loop over the ABI reproduces the native server's tokens exactly when it makes the native server's
  forwards: the prompt cut at the last `<|im_start|>` (the engine's turn point), the engine's draft rule (greedy
  from the head while the top-2 margin reaches `mtp-margin`, at most `mtp-draft`, never past `max_tokens`, no end
  of turn), one verify of the token and its drafts, the accepted prefix kept and the corrected token carried into
  the next verify. Any other grouping is a valid use of the ABI and gives valid logits; it is not the same
  arithmetic and is not expected to give the same tokens.
- The engine also puts ids into the sequence that the model did not sample: the thinking budget's stop text
  (`kThinkingStop`, 25 ids), fed when the budget is spent. With `enable_thinking` and `max_tokens` 48 the budget is
  zero and the stop text is the first thing after the prompt. The capture file marks such a step with
  `drafted = 255` (`serve/capture.hpp`); they are the caller's to feed, not the executor's to produce.

StrixFold promises the opposite for drafts (output-invariant: a draft changes when a token arrives, never which
token), so its conformance keeps the cross-shape comparison as a requirement.

## What the conformance driver proves, and what it does not

`executor.flerp.conformance` opens an executor once and runs cases on it, through `executor.h` only. Its paths:

| path | what it does | compared with |
|---|---|---|
| `loop` | strixite's `Engine::run` decode loop written over the ABI (`serve/engine.cpp` 1256-1691), which is `flerp_decode_loop` of `runtime.flerp.decode_loop`, the loop the front end runs: prompt cut at the turn point with the lookahead hint and the turn snapshot, sample, draft by the margin rule, verify, keep / keep_prefix and carry, fed ids where the engine fed them; writes tokens and `{drafted, accepted, emitted}` step records in the capture file's form | the native server's capture of the same request and MTP setting, token for token and record for record (the gate does this) |
| `loop-drop` | the loop with every fourth verify dropped (`flerp_drop`) and run again | the second run's rows equal the first's (candidate ids and logits bit for bit); tokens and steps equal the loop's |
| `snapshot` | save at a prompt position, decode, restore, decode; export, reset, import, decode | each equals the same forwards without a save |
| `plain`, `speculative` | one token a forward; a fixed draft depth a verify with dropped rounds | each other. Required for an executor with output-invariant drafts (`--cross-shape require`, the default); recorded only for strixite (`--cross-shape observe`) |

The strixite gate (`gate/executor_abi.py`, criteria in `gate/qwen38-flash-next.strixite-abi.toml`) loads the native
server with MTP on and with MTP off, takes each request's capture, then runs the driver with the executor's MTP
head on and off and requires: the loop's tokens and step records equal the capture's for every request; the
`loop-drop` and `snapshot` checks; no native request resumed from an earlier one's state. Equal step records mean
the two made the same forwards. It records, and does not require, where the native server parts from itself
between MTP on and off and where the driver's one-row and fixed-depth paths part.

What a pass does not cover: more than one seq and the batched verify path (`verify_batch`), masks and logit bias,
sampled draws (temperature above 0), state deltas (`from > 0`), the thinking nudge, a think block closed ahead of
a tool call, vision prompts, steering. The lookahead is passed and is a hint with no result of its own.

## Results (control-node GPU)

**strixite adapter: the conformance gate passes** (2026-11-11T02:16Z, `gate/results/qwen38-flash-next.strixite-abi/`:
`summary.md`, the native captures `native-mtp-{on,off}.json`, the driver's results and traces
`conformance-mtp-{on,off}.{json,txt}`). This run is the version 2 adapter and the driver on the shared decode loop
(`runtime.flerp.decode_loop`); the run of 01:26Z, before the loop was lifted out of the driver, had the same eight
rows. A run at 01:06Z reported the stage as failed because of an error in the gate's own verdict line (it read a
null `error` field as an error); its summary is kept in `run-2026-11-11T0106Z-verdict-bug/`.

| request | MTP | prompt tokens | generated tokens equal to the native capture | step records equal | rows a verify | verifies (kept as a prefix / dropped and run again) |
|---|---|---|---|---|---|---|
| chat-48 | on | 85 | 48 of 48 | 7 of 7 | 3, 4, 7, 8 | 5 (1 / 1) |
| raw-256 | on | 21 | 256 of 256 | 111 of 111 | 2 to 8 | 66 (14 / 16) |
| chat-256 | on | 85 | 164 of 164 | 58 of 58 | 2 to 8 | 33 (8 / 8) |
| long-chat-256 | on | 4,971 | 255 of 255 | 50 of 50 | 2 to 6, 8 | 47 (17 / 11) |
| chat-48 | off | 85 | 47 of 47 | 23 of 23 | - | - |
| raw-256 | off | 21 | 255 of 255 | 255 of 255 | - | - |
| chat-256 | off | 85 | 150 of 150 | 150 of 150 | - | - |
| long-chat-256 | off | 4,971 | 255 of 255 | 255 of 255 | - | - |

- With the engine's forwards the adapter gives the native server's tokens and its step records on every request,
  MTP on and off: 1,430 generated tokens, 151 verifies of 2 to 8 rows, 40 of them kept as a prefix with the
  corrected token carried. `keep_prefix` is exact.
- Every dropped verify, run again, returned the same candidate ids and logits bit for bit (36 of them), and the
  runs with drops ended with the same tokens and steps. Snapshot save, restore, export and import decode as the
  same forwards do without them (chat-48 and long-chat-256, both MTP settings).
- Recorded, not required: the native server with MTP on parts from itself with MTP off at token 43 of raw-256 and
  token 95 of chat-256 (none on the other two). The driver's one-row path parts from its engine loop at the same
  token 43, and its fixed four-drafts path parts from the one-row path at token 11.
- The chat-48 capture starts with 25 fed ids (the thinking budget's stop text); the driver fed the same ids.

Three earlier runs under the first criteria (2026-11-10 23:57Z to 2026-11-11 00:32Z) did not pass; their files
are in `gate/results/qwen38-flash-next.strixite-abi/before-shape-matching/`. The criteria asked for what the
origin does not do, and the two results those runs left open are explained:

| observation | cause | evidence |
|---|---|---|
| The chat request's first token differed from the native server's in every run (`\n\n` against `We`) | The native capture's first 25 ids are not sampled: at `max_tokens` 48 the thinking budget is zero and the engine feeds its stop text ("Considering the limited time by the user, ..." and `</think>`) before the first sample. The driver compared its first sampled token with fed text | `before-shape-matching/run3-native-captures.json`: step record `{255, 0, 25}`, the ids decoded |
| One row a forward (run 3) differed from the native server with MTP on at token 43 of `raw-completion` | The native server differs from itself there: the same 48 tokens equal the reference server's with MTP off, and the reference with MTP on parts from it at token 43 | `gate/results/qwen38-flash-next.strixite/identity.json` (`reference-mtp-off`, `reference-mtp-on`) against `before-shape-matching/conformance-c-raw-mtp-on-draft0-drops.json` |
| Four drafts a verify differed from one row a forward (token 11 or 44), `verify_batch` from `forward_verify_rows`, a prompt cut at 42 from one cut at 80 | Different rows a forward: different kernels and rounding, by the session's own account ("Forward shape" above). No native run makes four drafts at every step, so these runs had no native counterpart | `runtime/qwen4exp.hpp` 800-805 |
| Run 1's 48 of 48 on `raw-completion` | Obtained with RoPE YaRN factor 1 against the server's 2; it does not count | run 1's files |

Still established by those runs, all at one row a forward: plain decode, one-row verify + keep, one-row verify +
drop + verify again, snapshot save / restore and export / reset / import agree with each other, and the plain
path equals the reference server with MTP off for 48 tokens of `raw-completion`.

**StrixFold adapter: one run on the pair, checks passed, the process crashed at close; the stage failed.**
`flerp-executor-conformance.strixfold` ran as rank 0 on workers F and A on 2026-11-11T02:00Z, inside the StrixFold
recipe gate's pair window (`gate/results/dsv41-flash.strixfold-tp2/`: `pair.txt` holds both result files as the
driver wrote them, `summary.json` the stage's verdict). That binary was the version 1 adapter and the driver before
the decode loop was lifted out of it.

| prompt | prompt tokens | driver checks | one token a forward: 48 tokens against the native server's | four held drafts a verify against one token a forward | rounds (dropped and run again) | drafts accepted / made | state kept at 2,048, restored | exit |
|---|---|---|---|---|---|---|---|---|
| short-chat | 38 | 6 of 6 | equal | equal | 13 (3) | 34 / 52 | - (prompt too short) | SIGSEGV (139) |
| long-chat | 3,176 | 11 of 11 | equal | equal | 13 (3) | 34 / 49 | equal to no save, and to the plain run | SIGSEGV (139) |

- Every check the driver makes was true, and its one-token-a-forward tokens equal the first 48 the assembled
  StrixFold server generated for the same prompt with drafts on (compared after the run from the two committed
  files; the stage itself never made the comparison). Drafts were output-invariant through the ABI, as StrixFold
  promises: `speculative_equals_plain` held with rounds dropped and run again.
- Both processes then died of SIGSEGV after the result file was complete and closed: inside `flerp_close` or at
  process exit. Nothing else was captured (the worker scratch was removed), so the cause is not known. The stage
  reported failure because its reader took the shell's "Segmentation fault" line for part of the JSON.
- Since then, none of it run on the pair: the gate copies the driver's trace, the command's output and the end of
  rank 1's log into the results before teardown, reads the result as one JSON value, and requires exit status 0;
  `flerp_close` makes the device context current first (as the origin's host does), prints each step as it
  reaches it.
- 2026-11-11T02:48Z to 02:50Z, the front end over this adapter on the pair, drafts on
  (`gate/results/dsv41-flash.flerp-serve-tp2/`): the version 2 adapter under the shared decode loop, with held
  drafts and device-side draws, served three greedy chat requests with the origin server's prompt ids and reply
  tokens (164, 145 and 192; the stage's comparison then counted the origin's trailing end-of-sequence id as a
  difference) and a sampled request with the origin's tokens (134). On SIGTERM `flerp_close` ran every step to its
  end, the front end printed that the executor was closed, and the process then died of SIGSEGV (exit status 139):
  after the close, in a build that left the HIP runtime loaded. So the crash is not inside `flerp_close` and not
  the `dlclose`. The close is the origin's order again, and the host now prints a last line when its main
  returns, to tell the return path from the process's exit handlers. Drafts off has not run through the front
  end, and the stage as a whole has not passed.
