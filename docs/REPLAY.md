# Replay: recorded requests against two servers (FLERP-REPLAYKIT)

The FLERP-SWAP criterion for replacing an origin server (`docs/FLERP-SWAP-PLAN.md` A1, A6) is equivalence on
recorded real requests. Three tools do it, Python stdlib only:

| tool | does |
|---|---|
| `tools/replay/record_proxy.py --listen HOST:PORT --upstream http://NODE --log FILE.jsonl --target NAME` | sits between the client and one node, forwards every request unchanged, appends one JSON line per exchange |
| `tools/replay/scrub.py --out CORPUS [--manifest PATH] LOG.jsonl ...` | scrubs the logs into a corpus directory, one file per exchange, and writes the manifest |
| `tools/replay/scrub.py --check CORPUS [--manifest PATH]` | checks a corpus: modes, secrets, every file against the manifest's sha256, no message text in the manifest |
| `gate/replay.py --corpus CORPUS --a URL --b URL --out RESULTS ...` | replays the corpus against two servers and compares them |

Tests: `python3 -m pytest tests/test_replay.py` (stand-in HTTP servers for the mechanics; the front end over
`executor.flerp.fake_abi` as both servers, recorded, scrubbed and replayed, where its binary and the tokenizer are
at hand: control-node).

## The exchange schema

One JSON object per exchange, the same from the recorder, from FLERP-CORPUS's pcap extractor
(`/data/pod/flerp-replay/tools/extract.py`) and in a corpus file. That directory is a scratch area and its copies are
not maintained: the tools in this repo (`tools/replay/`, `gate/replay.py`) are the authoritative ones, including
`tools/replay/extract.py`, which the repo copy follows: `id`, `target` (`strixite`, `pair`), `server`,
`client`, `method`, `path`, `t_start`, `t_req_end`, `t_end` (epoch seconds), `req_headers`, `request` (the parsed JSON
body, `null` for none) or `request_raw` (a body that is not JSON), `status`, `resp_headers`, `response_raw` (the body
as text), `in_flight_at_start` (POSTs to `/v1/` in flight on that server when this one started), `req_complete`,
`resp_complete`. Bodies of `/cache/export` and `/cache/import` (0.5 to 1.3 GB) are kept as `req_body_bytes`,
`req_body_sha256`, `resp_body_bytes`, `resp_body_sha256` only. `scrub.py` adds `scrubbed` (placeholders written) and
`replayable` (with `not_replayable`, the reason, when false).

A corpus is `CORPUS/<target>/<id>.json`. `CORPUS/synthetic/` (requests made up for a feature the traffic lacked) is
never part of a replay or a manifest. The manifest (`CORPUS/manifest.json`, or committed as
`gate/replay/<corpus>.manifest.json`) lists each exchange: id, file, sha256, route, status, times, concurrency,
conversation (a hash), request keys, message count, token counts, replayable; and a census per target (routes,
`tools`, `tool_choice`, `steer`, `steer_scope`, `shared_messages`, images, cache routes, streams, seeds,
concurrency, every request key). It holds no message text, and `--check` fails if it does.

## Privacy

The corpus holds users' conversations. The recorder never logs `authorization`, `cookie`, `x-api-key`,
`proxy-authorization`, `set-cookie`, creates its log's directory mode 700 and the log mode 600. `scrub.py` replaces
private keys, GitHub/Slack/AWS keys, API keys, bearer and basic tokens, password/token assignments, e-mail addresses
and IPv4 addresses outside the pod's ranges with `<SCRUB:kind>` (the rules of FLERP-CORPUS's scrubber, kept equal),
writes directories mode 700 and files mode 600. Corpus files stay on the pod and are never committed; only the
manifest goes into git. `data:` images stay (they are the request); a remote image URL makes the exchange not
replayable.

## Replay rules

- **Conversation order.** Exchanges are replayed one at a time in recorded order (`t_start`), each to server A and
  then to server B, so every conversation's turns arrive in their order. Concurrency as recorded is not reproduced;
  slot behaviour is FLERP-SLOTS's, with the criterion from FLERP-CRITERIA.
- **Empty-cache start.** A cache hit moves the prompt cut, and for strixite the forward shape is part of the
  numbers (`docs/EXECUTOR-ABI.md`, "Forward shape"), so both servers must start with an empty prompt cache. With
  `--a-cmd` / `--b-cmd` the gate starts each server itself (a fresh process; give it an empty cache directory), waits
  for `/health` and stops it at the end. Servers started by someone else count as fresh only with `--assume-fresh`,
  and the summary says so. A run whose servers are not shown fresh is FAIL.
- **Seed injection.** strixite gives an unseeded request a random seed (`catalog/strixite/serve/http/src/serve/http_server.cpp:466`).
  A request body under `/v1/` recorded without `seed` gets `seed` = the first 31 bits of sha256(exchange id), the
  same on both servers and on every run. A recorded seed is kept.
- **A corpus you put exact numbers on must be the same traffic every run.** An unseeded recorded request replays
  under a fixed injected seed, but the reply *at recording time* was drawn under a random one, so a later turn whose
  prompt embeds that reply makes the corpus differ run to run (this cost `tests/test_replay.py` its stability: the same
  unchanged code produced 4 or 5 differing exchanges, 7 failures in 26 runs). Either seed the recording request, or
  build multi-turn corpora from fixed text.
- **Not replayed** (listed as skipped with the reason): `/health`, `/metrics`, `/stats`, `/dashboard` (the state of
  the moment); exchanges marked not replayable; incomplete exchanges; a request body that is not JSON; a
  `/cache/import` whose bytes no earlier `/cache/export` of the corpus answered. When one did, each server is sent
  its own export of that exchange back.

## What is compared

| | how |
|---|---|
| status | equal |
| token ids | where both servers say them: `--X-kind flerp` (`serve.flerp.http`) adds `"flerp_trace": true` to the body and reads the reply's `flerp.token_ids` (every id drawn; a stream: each chunk's) and `flerp.sequence_ids` (the ids fed, without the last drawn one), then removes the `flerp` object before the body comparison; `--X-capture-dir` (strixite's `--capture-dir`) reads the request's capture file, whose `generated` are the fed ids. A capture is one id array plus the header's `prompt_n` as the split (`serve/capture.cpp`): the first `prompt_n` ids are the prompt, the rest are `generated` -- that is all it covers, so the ids compared are this request's prompt and draw, not a running conversation | Two front ends compare `token_ids`, a front end and strixite `sequence_ids` (the last drawn id then shows in the body). StrixFold says none: "n/a", and the body decides |
| body | after dropping the volatile fields: JSON parsed and compared as values (key order included), SSE event by event, any other body byte for byte; `/cache/export` by size and sha256 |
| headers | the ones named by `--header` (default `content-type`) |

The volatile list starts as `id`, `created`, `timings`, `system_fingerprint` at the top level of a reply (and of
each SSE event). `--volatile` replaces it with comma-separated paths: dots go deeper, `*` is every element
(`choices.*.message.tool_calls.*.id` if an origin draws tool-call ids at random). FLERP-SELFID (origin against
origin) decides what else the origin does not reproduce against itself; such fields go into the list only with that
evidence.

Results: `RESULTS/results.json` (per exchange: equal or not, the first differing token index, the first differing
body path; never message text) and `RESULTS/summary.md` with the verdict. PASS = every replayed exchange equal on
all four and both servers shown fresh.

Example, a strixite worker's corpus against the front end on the same hardware (after FLERP-CORPUS):

```
python3 gate/replay.py --corpus /data/pod/flerp-replay/corpus-v1 --target strixite \
  --a http://127.0.0.1:5391 --a-kind origin --a-capture-dir /cache/replay-cap-a --a-cmd '<origin argv, fresh cache dir>' \
  --b http://127.0.0.1:5392 --b-kind flerp --b-cmd '<front end argv>' \
  --out gate/results/qwen38-flash-next.flerp-swap.replay
```

## Factory replay: the same traffic as load (FACTORY-REPLAY 2587bc1599)

`tools/replay/factory_replay.py` answers a different question from the gate above: not "do two servers agree" but
"how much does one engine commit under the factory's traffic, and does a thread get the same reply in company as
alone". It takes the engine URL and the model name as arguments, so the same set runs against strixite on a worker
(`qwen3.8-flash-next`) and against the DeepSeek pair (`dsv41`). Python stdlib only. It starts, stops and holds
nothing: the engine belongs to whoever holds the window (`gate/factory-replay.toml` is the baseline run).

```
# once: freeze a set from a corpus directory (numbers and hashes only, no message text; commit it)
python3 tools/replay/factory_replay.py freeze --corpus /data/pod/flerp-replay/corpus-v1 --targets strixite \
  --out gate/replay/factory-v1.set.json
# any time: is the corpus still the frozen one
python3 tools/replay/factory_replay.py verify --set gate/replay/factory-v1.set.json --corpus /data/pod/flerp-replay/corpus-v1
# the run: every stream alone first, then 3 reps with 2 in flight
python3 tools/replay/factory_replay.py run --set gate/replay/factory-v1.set.json --corpus /data/pod/flerp-replay/corpus-v1 \
  --url http://WORKER:5300 --model qwen3.8-flash-next --concurrency 2 --reps 3 --max-tokens 256 --solo all --out RESULTS
```

Tests: `python3 -m unittest tests.test_factory_replay` (a fake OpenAI server in the test file; no GPU, no pod).

**The set.** `freeze` keeps the complete `/v1/chat/completions` exchanges with status 200 and chains them into
streams: an exchange continues the stream whose last request's messages are a prefix of its own, so a stream is one
recorded thread, turn after turn. `gate/replay/factory-v1.set.json` is corpus-v1's strixite traffic: 207 exchanges in
27 streams (1 to 21 turns; recorded median 44,955 prompt tokens and 794 new tokens a turn), set_sha256
`176af79dc121...`. Each turn carries the sha256 of its exchange file, and `run` exits 2 before sending anything when a
file differs. `--targets` picks corpus subdirectories (all of corpus-v1: 358 exchanges in 50 streams, the pair's
bench drives included), `--max-turns T` splits longer threads into streams of at most T turns, `--max-streams S` keeps
the first S. These sha256 are of the file bytes; `scrub.py`'s manifest hashes are its own and are not compared.

**What a turn sends.** The recorded request with `model` replaced by `--model`, `temperature` 0, `stream` false, the
recorded `seed` (or the gate's rule: the first 31 bits of sha256(exchange id)), and `max_tokens` capped at
`--max-tokens` (0 = as recorded, 16,384 in corpus-v1). `--strip KEY` drops a request key an engine does not take,
`--set-json KEY=JSON` adds one. The history a turn resends is the recorded one, not the engine's replies of this run:
every engine and every rep gets the same prompts, and a reply that differs cannot change the load that follows it.

`run --first-streams N` plays only the first N streams in the order of their first turn's sha256, a fixed rule for
cutting the set to fit a hold; the summary names the streams kept. `gate/replay/baseline-run.sh` and
`gate/replay/serve.sh` are the baseline's win_run command and its engine start.

**Phases.** `solo-K`: the streams named by `--solo` (all, none, or the first K), one at a time. It is the identity
reference and it warms the engine's prompt cache, so the reps start from the same cache state. `rep-K`: all streams
with `--concurrency N` in flight; N workers each take the next stream and play its turns in order. `result.json` and
`summary.md` in `--out` are rewritten after every phase, so a job stopped by its time limit keeps what it finished.
`--wait-ready SECS` polls `/health` first; `--expect-slots N` refuses to run unless `/metrics` says `strix_slots N`.

**The numbers** (per phase, and the median of the reps):

| printed | is |
|---|---|
| committed tokens a second | the sum of `usage.completion_tokens` of the phase's answered turns over the phase's wall time: prefill, queueing and decode all count against it. This is the number a throughput gate compares |
| while saturated | the same up to the moment the first worker found no stream left; after that fewer than N are in flight |
| turn time p50, p95 | wall seconds from sending a turn to its full reply, nearest rank over the phase's answered turns |
| prefilled tokens a request | `usage.prompt_tokens` minus `usage.prompt_tokens_details.cached_tokens` (or `timings.prompt_n`), mean over the turns: what the engine had to run forward |
| cached tokens a request | `usage.prompt_tokens_details.cached_tokens` (or `timings.cache_n`), mean: what the prompt cache gave |
| errors | turns with no reply (HTTP error, timeout); they add no tokens and make the exit code 4 |
| identity | per rep, each stream against the same stream of the last solo pass, turn by turn: the reply's token ids when it carries them (`flerp.token_ids`; add `--set-json flerp_trace=true` for a flerp front end), else the sha256 of content, reasoning, tool calls (name and arguments; call ids are random) and finish reason. A stream is identical when every turn is; otherwise the first differing turn and its exchange id are listed, and the exit code is 3 |

Read identity with the cache in mind: strixite's forward shape depends on where the cache cut the prompt
(`docs/EXECUTOR-ABI.md`, "Forward shape"), and the solo pass runs colder than the reps. `--solo-passes 2` runs the
solo pass twice and reports solo-1 against solo-2 as well: a stream that differs there differs without any company.
With `--max-tokens` below the recorded reply lengths the decode share of a turn is smaller than in the factory
(corpus-v1 strixite: 375,385 recorded new tokens, 49,872 under a cap of 256), so compare runs at one cap only.
Nothing in `--out` holds message text.
