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
(`/data/pod/flerp-replay/tools/extract.py`) and in a corpus file: `id`, `target` (`strixite`, `pair`), `server`,
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
- **Not replayed** (listed as skipped with the reason): `/health`, `/metrics`, `/stats`, `/dashboard` (the state of
  the moment); exchanges marked not replayable; incomplete exchanges; a request body that is not JSON; a
  `/cache/import` whose bytes no earlier `/cache/export` of the corpus answered. When one did, each server is sent
  its own export of that exchange back.

## What is compared

| | how |
|---|---|
| status | equal |
| token ids | where both servers say them: `--X-kind flerp` (`serve.flerp.http`) adds `"flerp_trace": true` to the body and reads the reply's `flerp.token_ids` (every id drawn; a stream: each chunk's) and `flerp.sequence_ids` (the ids fed, without the last drawn one), then removes the `flerp` object before the body comparison; `--X-capture-dir` (strixite's `--capture-dir`) reads the request's capture file, whose `generated` are the fed ids. Two front ends compare `token_ids`, a front end and strixite `sequence_ids` (the last drawn id then shows in the body). StrixFold says none: "n/a", and the body decides |
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
