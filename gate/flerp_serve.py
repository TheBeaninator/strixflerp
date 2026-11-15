#!/usr/bin/env python3
"""The gate of the common front end (SPEC.md section 9, phase 3): `flerp gate recipe <recipe>` for a recipe whose
binary is serve.flerp.http in front of include/flerp/executor.h.

Stages (the recipe's config names which run, `[serve] stages`):
  check   flerp check passes for the catalogue
  build   the recipe assembles with no hand edits and its binary builds
  nogpu   (the stand-in executor's recipe) the front end against executor.flerp.fake_abi, started once in each shape
          of executor the config lists - rows or tokens only, a stepped or a held draft head, sampled draws or
          none, ABI version 1 or 2. No GPU, no model. Required, every one:
            - the routes answer (/health, /v1/models, /tokenize);
            - every feature outside the front end's scope is refused by name (HTTP 400 or 501, code "not_served"),
              a malformed request with 400, on both endpoints;
            - a request an executor cannot honour is refused from its capability flags (code "executor_cannot"):
              logprobs and logit_bias on a tokens-only executor, temperature above 0 on a greedy-only one, a
              top_k outside what its sampler takes;
            - the stand-in's tokens do not depend on how ids are grouped into forwards, so a greedy request gives
              the same token ids on every server: sampled on the host from candidates, from full rows, drawn by
              the executor, with a stepped draft head, a held one, and none;
            - a streamed response carries the same token ids and the same text as the one that is not streamed;
            - stop strings, max_tokens, the model's end of turn, a logit bias, logprobs, a seeded sample and a
              prompt past the context behave as the config's cases say.
  gpu     (the strixite recipe, control-node's GPU) the origin's own server, then the front end, one loaded at a
          time. The native server answers the configured requests with MTP drafting on, then off; its capture
          files (--capture-dir) give each request's prompt ids, the tokens that entered the sequence and one
          {drafted, accepted, emitted} record a decode step. The front end is then loaded with the executor's MTP
          head on and off and the settings the native server logged, and answers the same requests over HTTP with
          "flerp_trace": true, which adds the same three things to its response.
          strixite's logits depend on the rows a forward carries (docs/EXECUTOR-ABI.md, "Forward shape"), so the
          front end is run shape-matched - the prompt cut where the engine cuts it (--prompt-cut "<|im_start|>"),
          the engine's draft rule (--drafts, --draft-margin as logged) - and every required comparison is between
          runs that make the same forwards. Required, for every configured request, MTP on and off:
            - the prompt's token ids equal the capture's (the model's text module renders and tokenizes as the
              origin's server does);
            - the tokens that entered the sequence and the step records equal the capture's, token for token and
              record for record (equal step records are equal forwards);
            - for the requests marked `stream`, the same from the streamed response, whose chunks' token ids and
              text also join to the non-streamed response's;
            - on /v1/completions the text equals the native server's text (a chat reply goes through the origin's
              output parser, which this front end does not have: its text is compared and recorded, not required);
            - no native request resumed from an earlier one's state, and none was fed ids by the engine (the
              thinking budget's stop text: this front end serves no thinking budget, so such a request has no
              counterpart and fails the gate rather than being skipped).
          The sampled requests (temperature above 0, a fixed seed) are held to the same criteria: the recipe binds
          the origin's sampler module (runtime.flerp.sampler_strixite over strix::Sampler) and the loop samples
          rows in the engine's order. The logit-bias request too (the bias is the native request's).
          One more load takes the executor's own draws (--delivery tokens, MTP on): greedy requests must again
          equal the native capture. A logprobs request (full rows) must give the tokens of the same request
          without logprobs; its numbers are checked for consistency only (the native server was not asked).
          Recorded, not required: decode tokens per second of both servers for each request.

One model is loaded at a time. The stage starts only when the GPU is free and no job of the pod waits for it
(`pod-gpu --wait`); it holds the control-gpu lease while a model is loaded and gives it back between loads, waiting
when a pod job has started to wait. Lease and server helpers are the phase 1 gate's (gate/strixite_recipe.py).
"""
import argparse
import datetime
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gate"))
import strixite_recipe as p1                     # noqa: E402  (lease, GPU check, server, capture reader)
from flerplib.catalog import Catalog             # noqa: E402

STAGES = ("check", "build", "nogpu", "gpu")


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- HTTP -----------------------------------------------------------------------------------------------------
def http(port, path, body=None, timeout=600):
    """(status, parsed JSON or text). `body` None is a GET; a str is sent as it is; anything else as JSON."""
    data = None if body is None else body.encode() if isinstance(body, str) else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw.decode(errors="replace")


def sse(port, path, body, timeout=600):
    """A streamed request: (status, [parsed data payloads], saw [DONE], error body or None)."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, [], False, json.loads(raw)
        except ValueError:
            return e.code, [], False, raw.decode(errors="replace")
    chunks, done = [], False
    with r:
        for line in r:
            line = line.decode().rstrip("\n")
            if line.startswith("data: "):
                if line[6:] == "[DONE]":
                    done = True
                else:
                    chunks.append(json.loads(line[6:]))
    return r.status, chunks, done, None


def stream_join(chunks, chat):
    """What a stream's chunks add up to: text, token ids, finish reason, the final usage chunk's trace."""
    text, ids, finish, trace, usage, timings = "", [], None, None, None, None
    for c in chunks:
        for ch in c.get("choices", []):
            text += (ch.get("delta", {}).get("content") or "") if chat else (ch.get("text") or "")
            finish = ch.get("finish_reason") or finish
        if c.get("choices"):
            ids += c.get("flerp", {}).get("token_ids", [])
        else:
            trace, usage, timings = c.get("flerp"), c.get("usage"), c.get("timings")
    return {"text": text, "token_ids": ids, "finish_reason": finish, "flerp": trace, "usage": usage, "timings": timings}


def text_of(resp, chat):
    ch = resp["choices"][0]
    return ch["message"]["content"] if chat else ch["text"]


# --- stages: check, build -----------------------------------------------------------------------------------------
def stage_check(ctx):
    log = os.path.join(ctx["res"], "check.txt")
    open(log, "w").close()
    rc, _ = p1.sh([sys.executable, "flerp", "check"], log)
    return {"pass": rc == 0, "evidence": "check.txt", "last_line": p1.tail(log, 1)}


def stage_build(ctx):
    log = os.path.join(ctx["res"], "build.txt")
    open(log, "w").close()
    tree = os.path.join(ROOT, "build", ctx["recipe"].name)
    out = {"tree": os.path.relpath(tree, ROOT), "evidence": "build.txt", "started": now()}
    nice = ["nice", "-n", "19"]
    rc, _ = p1.sh([sys.executable, "flerp", "assemble", ctx["recipe"].name], log)
    if rc == 0:
        rc, out["configure_s"] = p1.sh(nice + ["cmake", "-G", "Ninja", "-S", tree, "-B", ctx["ninja"]], log, timeout=600)
    if rc == 0:
        rc, out["build_s"] = p1.sh(nice + ["ninja", "-C", ctx["ninja"], "-j", str(ctx["cfg"]["host"].get("build_jobs", 16))]
                                   + list(ctx["recipe"].binaries), log, timeout=3600)
    bins = {}
    for name in ctx["recipe"].binaries:
        p = os.path.join(ctx["ninja"], name)
        bins[name] = {"sha256": p1.sha256(p), "bytes": os.path.getsize(p)} if os.path.isfile(p) else None
    try:
        with open(os.path.join(tree, "assembly.json")) as f:
            asm = json.load(f)
        out["modules"] = len(asm.get("modules", {}))
    except (OSError, ValueError):
        pass
    out.update(binaries=bins, rc=rc, finished=now(), **{"pass": rc == 0 and all(bins.values())})
    return out


# --- stage: nogpu ---------------------------------------------------------------------------------------------------
CHAT = [{"role": "user", "content": "Name three rivers in Europe."}]
RAW = "The following is a list of rivers:\n\n1."


def refusals(port, tokens_only):
    """Requests this front end must refuse by name: (name, method/path, body, status, code)."""
    chat, comp = "/v1/chat/completions", "/v1/completions"
    c = lambda **kw: dict({"messages": CHAT, "max_tokens": 4, "temperature": 0}, **kw)
    r = lambda **kw: dict({"prompt": RAW, "max_tokens": 4, "temperature": 0}, **kw)
    image = [{"role": "user", "content": [{"type": "text", "text": "what is this"},
                                          {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
    tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}
    cases = [
        ("tools", chat, c(tools=[tool]), 400, "not_served"),
        ("tool_choice-named", chat, c(tool_choice={"type": "function", "function": {"name": "f"}}), 400, "not_served"),
        ("functions", chat, c(functions=[{"name": "f"}]), 400, "not_served"),
        ("response_format-json_object", chat, c(response_format={"type": "json_object"}), 400, "not_served"),
        ("response_format-json_schema", chat, c(response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}), 400, "not_served"),
        ("grammar", comp, r(grammar="root ::= \"a\""), 400, "not_served"),
        ("guided_json", chat, c(guided_json={"type": "object"}), 400, "not_served"),
        ("vision-image_url", chat, c(messages=image), 400, "not_served"),
        ("steer", chat, c(steer=[{"vector": "brevity", "scale": 1}]), 400, "not_served"),
        ("cache_prompt", comp, r(cache_prompt=True), 400, "not_served"),
        ("thinking-budget-max_thinking_tokens", chat, c(max_thinking_tokens=64), 400, "not_served"),
        ("thinking-budget-thinking.budget_tokens", chat, c(thinking={"type": "enabled", "budget_tokens": 64}), 400, "not_served"),
        ("thinking-budget-reasoning.max_tokens", chat, c(reasoning={"max_tokens": 64}), 400, "not_served"),
        ("thinking-budget-chat_template_kwargs", chat, c(chat_template_kwargs={"thinking_budget": 64}), 400, "not_served"),
        ("n-2", chat, c(n=2), 400, "not_served"),
        ("echo", comp, r(echo=True), 400, "not_served"),
        ("suffix", comp, r(suffix="."), 400, "not_served"),
        ("best_of-2", comp, r(best_of=2), 400, "not_served"),
        ("completions-logprobs", comp, r(logprobs=3), 400, "not_served"),
        ("route-cache", "GET /cache", None, 501, "not_served"),
        ("route-cache-export", "/cache/export", {}, 501, "not_served"),
        ("route-steer", "GET /steer", None, 501, "not_served"),
        ("route-anthropic-messages", "/v1/messages", {}, 501, "not_served"),
        ("malformed-json", chat, "{not json", 400, None),
        ("malformed-no-messages", chat, {"max_tokens": 4}, 400, None),
        ("malformed-prompt-array", comp, {"prompt": [1, 2, 3]}, 400, None),
        ("malformed-temperature", chat, c(temperature=-1), 400, None),
        ("malformed-top_p", comp, r(top_p=0), 400, None),
        ("malformed-max_tokens", comp, r(max_tokens=0), 400, None),
        ("malformed-stop", comp, r(stop=[""]), 400, None),
        ("malformed-logit_bias-key", comp, r(logit_bias={"not-an-id": 1}), 400, None),
        ("malformed-role", chat, c(messages=[{"role": "narrator", "content": "x"}]), 400, None),
    ]
    if not tokens_only:
        cases.append(("stream-logprobs", chat, c(stream=True, logprobs=True), 400, "not_served"))
    out = []
    for name, path, body, status, code in cases:
        get = path.startswith("GET ")
        st, resp = http(port, path[4:] if get else path, None if get else body, 30)
        err = resp.get("error", {}) if isinstance(resp, dict) else {}
        ok = st == status and bool(err.get("message")) and (code is None or err.get("code") == code)
        out.append({"check": "refuse:" + name, "pass": ok, "status": st, "code": err.get("code"), "message": err.get("message")})
    return out


def fake_run(ctx, spec, shared):
    """One load of the front end over the stand-in executor in one shape; returns its checks."""
    cfg, h, fk = ctx["cfg"], ctx["cfg"]["host"], ctx["cfg"]["fake"]
    port, name = h["port"], spec["name"]
    binary = os.path.join(ctx["ninja"], ctx["cfg"]["serve"]["binary"])
    args = (["--host", "127.0.0.1", "--port", str(port), "--model-id", "fake", f"text.tokenizer={h['tokenizer']}"]
            + [str(a) for a in fk["args"]] + [str(a) for a in spec["args"]])
    log = os.path.join(ctx["res"], f"fake-{name}.log")
    checks = []

    def check(what, ok, **detail):
        checks.append(dict({"check": what, "pass": bool(ok)}, **detail))
        return ok

    srv = p1.Server(binary, args, port, log)
    try:
        srv.wait_ready(60)
        st, health = http(port, "/health")
        check("health", st == 200 and health.get("status") == "ok" and health.get("sampled_by") == spec["sampled_by"]
              and health.get("draft_form") == spec["draft_form"] and health.get("executor_abi") == spec.get("abi", 2),
              health=health)
        st, models = http(port, "/v1/models")
        check("models", st == 200 and models["data"][0]["id"] == "fake")
        st, tok = http(port, "/tokenize", {"content": "Hello world<|im_end|>"})
        check("tokenize", st == 200 and tok.get("tokens") == shared["hello_ids"], got=tok)
        tokens_only = spec["sampled_by"] == "executor"
        checks.extend(refusals(port, tokens_only))

        # greedy generation: both endpoints, streamed and not
        for ep, chat, base in (("/v1/completions", False, {"prompt": RAW}), ("/v1/chat/completions", True, {"messages": CHAT, "enable_thinking": False})):
            key = "chat" if chat else "raw"
            body = dict(base, max_tokens=fk["max_tokens"], temperature=0, flerp_trace=True)
            st, r = http(port, ep, body)
            if not check(f"{key}:greedy", st == 200 and r["choices"][0]["finish_reason"] == "stop"
                         and r["usage"]["completion_tokens"] == fk["reply"] + 1 and len(r["flerp"]["token_ids"]) == fk["reply"],
                         status=st, got=r if st != 200 else {"usage": r["usage"], "finish": r["choices"][0]["finish_reason"]}):
                continue
            ids, text = r["flerp"]["token_ids"], text_of(r, chat)
            check(f"{key}:delivery", r["flerp"]["delivery"] == spec["delivery"] and r["flerp"]["sampled_by"] == spec["sampled_by"], trace={k: r["flerp"][k] for k in ("delivery", "sampled_by", "verifies", "drafted", "accepted", "forwards")})
            if spec["draft_form"] != "none":
                check(f"{key}:drafts-used", r["flerp"]["verifies"] > 0 and 0 < r["flerp"]["accepted"] < r["flerp"]["drafted"],
                      verifies=r["flerp"]["verifies"], accepted=r["flerp"]["accepted"], drafted=r["flerp"]["drafted"])
            # the same tokens on every server shape: the stand-in's tokens do not depend on the forwards
            ref = shared.setdefault(f"{key}:ids", {"server": name, "ids": ids, "text": text, "prompt": r["flerp"]["prompt_token_ids"]})
            check(f"{key}:same-tokens-as-{ref['server']}", ids == ref["ids"] and text == ref["text"] and r["flerp"]["prompt_token_ids"] == ref["prompt"])
            st, chunks, done, err = sse(port, ep, dict(body, stream=True))
            j = stream_join(chunks, chat)
            check(f"{key}:stream", st == 200 and done and j["token_ids"] == ids and j["text"] == text and j["finish_reason"] == "stop"
                  and (j["flerp"] or {}).get("token_ids") == ids and j["usage"] == r["usage"], status=st, error=err, chunks=len(chunks))
            st, chunks, done, err = sse(port, ep, dict(base, max_tokens=5, temperature=0, stream=True))
            check(f"{key}:stream-plain", st == 200 and done and all("flerp" not in c and c.get("choices") for c in chunks), chunks=len(chunks))
            # max_tokens
            st, r2 = http(port, ep, dict(body, max_tokens=7))
            check(f"{key}:max_tokens", st == 200 and r2["choices"][0]["finish_reason"] == "length" and r2["usage"]["completion_tokens"] == 7
                  and r2["flerp"]["token_ids"] == ids[:7])
            # a stop string taken from the reply: the text ends before it
            stop = text[len(text) // 2: len(text) // 2 + 6]
            want = text[:text.find(stop)]
            st, r3 = http(port, ep, dict(body, stop=[stop, "\x00never"]))
            check(f"{key}:stop", st == 200 and text_of(r3, chat) == want and r3["choices"][0]["finish_reason"] == "stop", stop=stop)
            st, chunks, done, err = sse(port, ep, dict(body, stop=stop, stream=True))
            check(f"{key}:stop-stream", st == 200 and done and stream_join(chunks, chat)["text"] == want)

        # what this executor shape can and cannot honour
        raw = {"prompt": RAW, "max_tokens": 12, "flerp_trace": True}
        first = shared["raw:ids"]["ids"][0]
        st, r = http(port, "/v1/completions", dict(raw, temperature=0, logit_bias={str(first): -100}))
        if spec.get("masks", True):
            check("logit_bias", st == 200 and r["flerp"]["token_ids"][0] != first, first=first, got=r["flerp"]["token_ids"][:2] if st == 200 else r)
        else:
            check("refuse:logit_bias", st == 400 and r["error"]["code"] == "executor_cannot", status=st, message=r.get("error", {}).get("message"))
        chatb = {"messages": CHAT, "enable_thinking": False, "max_tokens": 12, "temperature": 0, "flerp_trace": True}
        st, r = http(port, "/v1/chat/completions", dict(chatb, logprobs=True, top_logprobs=3))
        if spec.get("logprobs"):
            lp = r["choices"][0]["logprobs"]["content"] if st == 200 else []
            check("logprobs", st == 200 and r["flerp"]["delivery"] == "full" and [e["token_id"] for e in lp] == r["flerp"]["token_ids"]
                  and r["flerp"]["token_ids"] == shared["chat:ids"]["ids"][:12]
                  and all(e["logprob"] <= 0 and len(e["top_logprobs"]) == 3 and e["top_logprobs"][0]["token_id"] == e["token_id"]
                          and abs(e["top_logprobs"][0]["logprob"] - e["logprob"]) < 1e-9 for e in lp), status=st)
        else:
            check("refuse:logprobs", st == 400 and r["error"]["code"] == "executor_cannot", status=st, message=r.get("error", {}).get("message"))
        # a temperature at which the stand-in's top three candidates all carry weight (logits 10, 8 and below 4)
        sampled = dict(raw, temperature=4.0, top_k=3, seed=42)
        st, r = http(port, "/v1/completions", sampled)
        if spec.get("sampled", True):
            st2, r2 = http(port, "/v1/completions", sampled)
            st3, r3 = http(port, "/v1/completions", dict(sampled, seed=43))
            ok = st == 200 and st2 == 200 and st3 == 200 and r["flerp"]["token_ids"] == r2["flerp"]["token_ids"]
            check("sampled-seeded", ok and r["flerp"]["token_ids"] != r3["flerp"]["token_ids"], status=st,
                  got=r["flerp"]["token_ids"][:6] if st == 200 else r)
            if ok and spec["sampled_by"] == "host":   # the host sampler's draws do not depend on the draft form
                ref = shared.setdefault("sampled:ids", {"server": name, "ids": r["flerp"]["token_ids"]})
                check(f"sampled:same-tokens-as-{ref['server']}", r["flerp"]["token_ids"] == ref["ids"])
        else:
            check("refuse:temperature", st == 400 and r["error"]["code"] == "executor_cannot", status=st, message=r.get("error", {}).get("message"))
        for label, extra, status in spec.get("sampling_cases", []):
            st, r = http(port, "/v1/completions", dict(raw, **extra))
            check("sampling:" + label, st == status and (status == 200 or r["error"]["code"] == "executor_cannot"), status=st,
                  message=r.get("error", {}).get("message") if isinstance(r, dict) else None)
        # a prompt past the context
        st, r = http(port, "/v1/completions", {"prompt": "river " * 6000, "max_tokens": 4, "temperature": 0})
        check("context-exceeded", st == 400 and r["error"]["code"] == "context_length_exceeded", status=st)
        # two requests at once: both are served, one after the other
        got = []
        ts = [threading.Thread(target=lambda: got.append(http(port, "/v1/completions", dict(raw, temperature=0)))) for _ in range(2)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        check("two-at-once", len(got) == 2 and all(s == 200 and r["flerp"]["token_ids"] == shared["raw:ids"]["ids"][:12] for s, r in got))
        st, health = http(port, "/health")
        check("health-after", st == 200 and health["status"] == "ok" and not health["busy"], served=health.get("requests_served"))
    except Exception as e:                           # a check that could not run fails the server's record
        check("ran-to-the-end", False, error=f"{type(e).__name__}: {e}")
    finally:
        rc = srv.stop()
    check("clean-exit", rc == 0, rc=rc, log_tail=p1.tail(log, 2))
    return {"server": name, "args": args, "checks": checks, "pass": all(c["pass"] for c in checks),
            "failed": [c["check"] for c in checks if not c["pass"]]}


def stage_nogpu(ctx):
    h, fk = ctx["cfg"]["host"], ctx["cfg"]["fake"]
    with open(h["tokenizer"]) as f:
        added = {t.get("content"): t["id"] for t in json.load(f).get("added_tokens", [])}
    out = {"host": socket.gethostname(), "started": now(), "servers": []}
    shared = {"hello_ids": None}
    # "Hello world" then the end-of-turn token: the ids of the text, then the added token's own id
    shared["hello_ids"] = fk["hello_ids"] + [added["<|im_end|>"]]
    for spec in fk["server"]:
        spec = dict(spec, sampling_cases=[(c["name"], {k: v for k, v in c.items() if k not in ("name", "status")}, c["status"])
                                          for c in spec.get("sampling_case", [])])
        out["servers"].append(fake_run(ctx, spec, shared))
    out["finished"] = now()
    out["checks"] = sum(len(s["checks"]) for s in out["servers"])
    out["pass"] = len(out["servers"]) == len(fk["server"]) and all(s["pass"] for s in out["servers"])
    return out


# --- stage: gpu -----------------------------------------------------------------------------------------------------
FACTS = (("n_valid", r"startup: tokenizer \S+ \((\d+) tokens\)"), ("mtp_draft", r"startup: config mtp-draft = (\d+)"),
         ("mtp_margin", r"startup: config mtp-margin = ([0-9.]+)"), ("chunk", r"startup: config chunk = (\d+)"),
         ("capacity", r"startup: config capacity = (\d+)"), ("total_slots", r"startup: config total-slots = (\d+)"),
         ("rope_yarn_factor", r"startup: config rope-yarn-factor = ([0-9.]+)"), ("kv_cache", r"startup: config kv-cache = (\S+)"),
         ("gdn_precision", r"startup: config gdn-precision = (\S+)"), ("attention_precision", r"startup: config attention-precision = (\S+)"),
         ("mtp_vocab", r"startup: config mtp-vocab = (\d+)"), ("ngram_cache_rows", r"startup: config ngram-cache-rows = (\d+)"),
         ("kv_generations", r"startup: config kv-generations = (\d+)"))


def gate_requests(cfg):
    """The configured requests as (spec, path, body): a phase 1 prompt by name plus what the spec changes."""
    known = {name: (path, body) for name, path, body in p1.requests(cfg)}
    out = []
    for r in cfg["serve"]["request"]:
        path, body = known[r["prompt"]]
        body = dict(body, max_tokens=r["max_tokens"])
        body.update(r.get("body", {}))
        out.append((r, path, body))
    return out


def native_text(resp):
    ch = resp["choices"][0]
    if "text" in ch:
        return ch["text"], False
    m = ch.get("message", {})
    return m.get("content") or "", bool(m.get("reasoning_content")) or bool(m.get("tool_calls"))


def derive(spec, body, first):
    """A request that depends on the native server's answer to an earlier one (`from`): a stop string taken from
    its text, or a logit bias against its first token."""
    if spec.get("stop_from"):
        text = first[spec["stop_from"]]["text"]
        at = len(text) * 2 // 3
        body = dict(body, stop=[text[at:at + 12]])
    if spec.get("bias_from"):
        body = dict(body, logit_bias={str(first[spec["bias_from"]]["generated"][0]): -100})
    return body


def native_run(ctx, mtp):
    cfg, h, run, sv = ctx["cfg"], ctx["cfg"]["host"], ctx["cfg"]["run"], ctx["cfg"]["serve"]
    reqs = gate_requests(cfg)
    scratch = os.path.join(ctx["scratch"], f"native-mtp-{mtp}")
    cap = os.path.join(scratch, "capture")
    subprocess.run(["rm", "-rf", scratch])
    os.makedirs(cap)
    binary = h["reference_server"]
    args = (["--weights", h["weights"], "--ngram", h["ngram"], "--tokenizer", h["tokenizer"],
             "--generation-config", h["generation_config"], "--host", "127.0.0.1", "--port", str(h["port"]),
             "--mtp", mtp, "--capture-dir", cap] + [str(a) for a in cfg["server"]["args"]] + [str(a) for a in sv["native_args"]])
    out = {"mtp": mtp, "binary": binary, "binary_sha256": p1.sha256(binary), "args": args, "requests": []}
    answers, bodies = {}, {}
    srv = p1.Server(binary, args, h["port"], os.path.join(scratch, "server.log"))
    try:
        out["startup_s"] = srv.wait_ready(run["startup_timeout_s"])
        for spec, path, body in reqs:
            body = derive(spec, body, answers)
            bodies[spec["name"]] = body
            r = srv.post(path, body, run["request_timeout_s"])
            text, split = native_text(r)
            usage = r.get("usage", {})
            answers[spec["name"]] = {"text": text, "split": split, "finish_reason": r["choices"][0].get("finish_reason"),
                                     "usage": usage, "timings": r.get("timings", {}),
                                     "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens")}
            # the capture is read after the server stops; a derived request needs the first tokens now
            answers[spec["name"]]["generated"] = latest_capture(cap, len(answers))["generated"]
    finally:
        out["exit_code"] = srv.stop()
    with open(srv.log_path, errors="replace") as f:
        log = f.read()
    with open(os.path.join(ctx["res"], f"native-mtp-{mtp}-server.log"), "w") as f:
        f.write(log)
    facts = {}
    for key, rx in FACTS:
        m = re.search(rx, log)
        if not m:
            raise RuntimeError(f"native server log: no line for {key}")
        facts[key] = m.group(1)
    out["facts"] = facts
    caps = sorted((p1.read_capture(os.path.join(cap, f)) for f in os.listdir(cap) if f.endswith(".cap")), key=lambda c: c["id"])
    if len(caps) != len(reqs):
        raise RuntimeError(f"native server: {len(caps)} capture files for {len(reqs)} requests")
    for (spec, _, _), c in zip(reqs, caps):
        a = answers[spec["name"]]
        a.update(prompt=c["prompt"], generated=c["generated"], steps=c["steps"], capture_finish=c["finish"])
        out["requests"].append({"name": spec["name"], "finish_reason": a["finish_reason"], "usage": a["usage"],
                                "cached_tokens": a["cached_tokens"], "timings": a["timings"],
                                "fed_steps": sum(1 for d, _, _ in c["steps"] if d == 255)})
    with open(os.path.join(ctx["res"], f"native-mtp-{mtp}.json"), "w") as f:
        json.dump({"run": out, "bodies": bodies, "answers": answers}, f, indent=1)
        f.write("\n")
    return out, answers, bodies


def latest_capture(cap_dir, n):
    """The capture of the n-th request (written as the request finishes, a moment after its response)."""
    for _ in range(100):
        files = [f for f in os.listdir(cap_dir) if f.endswith(".cap")]
        if len(files) >= n:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError(f"native server: {len(files)} capture files after {n} requests")
    return max((p1.read_capture(os.path.join(cap_dir, f)) for f in files), key=lambda c: c["id"])


def first_diff(a, b):
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), None if len(a) == len(b) else n)


def compare(spec, native, resp, chat, streamed=None):
    """One front end response against the native capture of the same request."""
    t = resp["flerp"]
    rec = {"name": spec["name"], "prompt_tokens": len(native["prompt"]), "native_tokens": len(native["generated"]),
           "tokens": len(t["sequence_ids"]), "native_steps": len(native["steps"]), "steps": len(t["steps"]),
           "delivery": t["delivery"], "sampled_by": t["sampled_by"], "prompt_cut": t["prompt_cut"],
           "prompt_equal": t["prompt_token_ids"] == native["prompt"],
           "tokens_equal": t["sequence_ids"] == native["generated"],
           "first_token_difference": first_diff(t["sequence_ids"], native["generated"]),
           "steps_equal": t["steps"] == native["steps"],
           "first_step_difference": first_diff(t["steps"], native["steps"]),
           "finish_reason": resp["choices"][0]["finish_reason"], "native_finish_reason": native["finish_reason"],
           "finish_equal": resp["choices"][0]["finish_reason"] == native["finish_reason"],
           "usage_equal": {k: resp["usage"].get(k) for k in ("prompt_tokens", "completion_tokens")}
                          == {k: native["usage"].get(k) for k in ("prompt_tokens", "completion_tokens")},
           "native_fed_steps": sum(1 for d, _, _ in native["steps"] if d == 255),
           "native_cached_tokens": native["cached_tokens"],
           "verifies": t["verifies"], "drafted": t["drafted"], "accepted": t["accepted"],
           "rows_a_verify": sorted({d + 1 for d, _, _ in t["steps"] if 0 < d < 255}),
           "decode_tps": round(resp["timings"]["predicted_per_second"], 2),
           "native_decode_tps": round(native["timings"].get("predicted_per_second", 0.0), 2)}
    # the text: required equal on /v1/completions, where the native server returns the reply as it is. A chat reply
    # goes through the origin's output parser (reasoning and tool calls split off), which this front end does not
    # have: there the comparison is recorded when the native reply is one text, and never required
    rec["text_comparable"] = not native["split"]
    rec["text_equal"] = text_of(resp, chat) == native["text"] if rec["text_comparable"] else None
    rec["text_required"] = not chat
    need = [rec["prompt_equal"], rec["tokens_equal"], rec["steps_equal"], rec["finish_equal"], rec["usage_equal"],
            rec["native_fed_steps"] == 0, rec["native_cached_tokens"] == 0, chat or rec["text_equal"] is True]
    if streamed is not None:
        st, chunks, done, err = streamed
        j = stream_join(chunks, chat)
        tr = j["flerp"] or {}
        rec["stream"] = {"status": st, "done": done, "chunks": len(chunks), "error": err,
                         "tokens_equal": tr.get("sequence_ids") == native["generated"],
                         "steps_equal": tr.get("steps") == native["steps"],
                         "chunk_ids_join": j["token_ids"] == t["token_ids"],
                         "text_joins": j["text"] == text_of(resp, chat),
                         "finish_equal": j["finish_reason"] == native["finish_reason"],
                         "decode_tps": round((j["timings"] or {}).get("predicted_per_second", 0.0), 2)}
        s = rec["stream"]
        need += [st == 200, done, s["tokens_equal"], s["steps_equal"], s["chunk_ids_join"], s["text_joins"], s["finish_equal"]]
    rec["pass"] = all(need)
    return rec


def front_run(ctx, label, mtp, facts, native, bodies, delivery="auto", only=None):
    """One load of the front end; every configured request (or `only`) over HTTP."""
    cfg, h, run, sv = ctx["cfg"], ctx["cfg"]["host"], ctx["cfg"]["run"], ctx["cfg"]["serve"]
    binary = os.path.join(ctx["ninja"], sv["binary"])
    log = os.path.join(ctx["res"], f"front-{label}-server.log")
    args = ["--host", "127.0.0.1", "--port", str(h["port"]), "--model-id", "qwen3.8-flash-next",
            "--generation-config", h["generation_config"], "--prompt-cut", sv["prompt_cut"], "--delivery", delivery,
            "--drafts", facts["mtp_draft"], "--draft-margin", facts["mtp_margin"],
            f"text.tokenizer={h['tokenizer']}", f"executor.weights={h['weights']}", f"executor.ngram={h['ngram']}", f"executor.mtp={mtp}"]
    # the executor is opened with the settings the native server logged, key for key
    for key, fact in (("capacity", "capacity"), ("chunk", "chunk"), ("seqs", "total_slots"), ("rope_yarn_factor", "rope_yarn_factor"),
                      ("kv_cache", "kv_cache"), ("gdn_precision", "gdn_precision"), ("attention_precision", "attention_precision"),
                      ("mtp_vocab", "mtp_vocab"), ("ngram_cache_rows", "ngram_cache_rows"), ("kv_generations", "kv_generations")):
        args.append(f"executor.{key}={facts[fact]}")
    out = {"label": label, "mtp": mtp, "delivery": delivery, "binary": binary, "binary_sha256": p1.sha256(binary), "args": args,
           "cases": [], "extra": []}
    srv = p1.Server(binary, args, h["port"], log)
    try:
        out["startup_s"] = srv.wait_ready(run["startup_timeout_s"])
        st, out["health"] = http(h["port"], "/health")
        st, tok = http(h["port"], "/tokenize", {"content": "Name three rivers in Europe.<|im_end|>"})
        out["tokenize"] = tok
        for spec, path, _ in gate_requests(cfg):
            if only is not None and spec["name"] not in only:
                continue
            chat = path.endswith("/chat/completions")
            body = dict(bodies[spec["name"]], flerp_trace=True)
            st, resp = http(h["port"], path, body, run["request_timeout_s"])
            if st != 200:
                out["cases"].append({"name": spec["name"], "pass": False, "error": f"HTTP {st}: {json.dumps(resp)[:300]}"})
                continue
            streamed = sse(h["port"], path, dict(body, stream=True), run["request_timeout_s"]) if spec.get("stream") else None
            out["cases"].append(compare(spec, native[spec["name"]], resp, chat, streamed))
        if only is None and delivery == "auto":
            # logprobs (full rows): the tokens of the same request without them; the numbers for consistency
            for spec, path, _ in gate_requests(cfg):
                if not spec.get("logprobs"):
                    continue
                body = dict(bodies[spec["name"]], flerp_trace=True, logprobs=True, top_logprobs=5)
                st, resp = http(h["port"], path, body, run["request_timeout_s"])
                rec = {"check": "logprobs:" + spec["name"], "status": st}
                if st == 200:
                    lp, t = resp["choices"][0]["logprobs"]["content"], resp["flerp"]
                    rec.update(delivery=t["delivery"], tokens=len(lp),
                               tokens_equal_native=t["sequence_ids"] == native[spec["name"]]["generated"],
                               entries_match_tokens=[e["token_id"] for e in lp] == t["token_ids"],
                               greedy_is_top=all(e["top_logprobs"][0]["token_id"] == e["token_id"] for e in lp),
                               logprobs_in_range=all(-1e9 < e["logprob"] <= 0 for e in lp),
                               mean_logprob=round(sum(e["logprob"] for e in lp) / max(1, len(lp)), 4))
                    rec["pass"] = (t["delivery"] == "full" and rec["tokens_equal_native"] and rec["entries_match_tokens"]
                                   and rec["greedy_is_top"] and rec["logprobs_in_range"])
                else:
                    rec.update({"pass": False, "error": json.dumps(resp)[:300]})
                out["extra"].append(rec)
            # Amendment (FLERP-RECUT, after 9ef8d14): response_format is served now (FLERP-CONSTRAIN), so the check that
            # expected a refusal here is replaced by one that expects the constraint to hold against the real executor:
            # HTTP 200 and a reply that validates against the schema sent. Every other check is unchanged.
            schema = {"type": "object", "properties": {"rivers": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3}},
                      "required": ["rivers"], "additionalProperties": False}
            st, resp = http(h["port"], "/v1/chat/completions",
                            {"messages": CHAT, "max_tokens": 256, "temperature": 0, "enable_thinking": False,
                             "response_format": {"type": "json_schema", "json_schema": {"name": "rivers", "schema": schema, "strict": True}}},
                            run["request_timeout_s"])
            content, valid = None, False
            if st == 200:
                try:
                    content = resp["choices"][0]["message"]["content"]
                    valid = schema_valid(json.loads(content), schema)
                except (KeyError, IndexError, TypeError, ValueError):
                    pass
            out["extra"].append({"check": "serve:response_format", "status": st, "content": content,
                                 "pass": st == 200 and valid})
    finally:
        out["exit_code"] = srv.stop()
        out["server_log_tail"] = p1.tail(log, 3)
    out["pass"] = (out["exit_code"] == 0 and bool(out["cases"]) and all(c["pass"] for c in out["cases"])
                   and all(e["pass"] for e in out["extra"]))
    return out


def schema_valid(v, sc):
    """The subset of JSON Schema the gate's own schema uses: type, properties, required, additionalProperties, items, minItems, maxItems."""
    t = sc.get("type")
    types = {"object": dict, "array": list, "string": str}
    if t in types and not isinstance(v, types[t]):
        return False
    if t == "object":
        props = sc.get("properties", {})
        if any(k not in v for k in sc.get("required", [])):
            return False
        if sc.get("additionalProperties") is False and any(k not in props for k in v):
            return False
        return all(schema_valid(x, props[k]) for k, x in v.items() if k in props)
    if t == "array":
        if not sc.get("minItems", 0) <= len(v) <= sc.get("maxItems", len(v)):
            return False
        return all(schema_valid(x, sc["items"]) for x in v) if "items" in sc else True
    return True


def wait_gpu_free(cfg, seconds):
    t0 = time.time()
    while True:
        ok, state = p1.gpu_free(cfg)
        if ok or time.time() - t0 > seconds:
            return ok, state
        time.sleep(5)


def stage_gpu(ctx):
    cfg, sv = ctx["cfg"], ctx["cfg"]["serve"]
    out = {"host": socket.gethostname(), "started": now(), "native": [], "front": []}
    waiters = p1.pod_waiters()
    ok, state = p1.gpu_free(cfg)
    out["gpu_before"] = dict(state, pod_gpu_waiters=len(waiters))
    if not ok or waiters:
        out.update({"pass": False, "blocked": "the GPU is not free (" + (f"{len(waiters)} pod-gpu processes wait for it" if waiters
                    else "another engine is loaded here") + "): run this stage later"})
        return out
    hold = p1.Hold(cfg, os.path.join(ctx["res"], "lease.txt"), out, sv.get("yield_wait_s", 1800))
    label = f"strixflerp front end gate ({ctx['recipe'].name})"
    native, bodies, facts = {}, {}, {}

    def load(what, fn):
        hold.take(f"{label}: {what}", what)
        try:
            return fn()
        finally:
            ok, state = wait_gpu_free(cfg, 120)
            hold.give()
            if not ok:
                raise RuntimeError(f"the GPU did not return to idle after {what}: {state}")

    try:
        for mtp in sv["mtp"]:
            rec, native[mtp], bodies[mtp] = load(f"native server, MTP {mtp}", lambda: native_run(ctx, mtp))
            out["native"].append(rec)
            facts[mtp] = rec["facts"]
        for mtp in sv["mtp"]:
            out["front"].append(load(f"front end, executor MTP {mtp}",
                                     lambda: front_run(ctx, f"mtp-{mtp}", mtp, facts[mtp], native[mtp], bodies[mtp])))
        mtp = sv["tokens_delivery"]["mtp"]
        out["front"].append(load(f"front end taking the executor's draws, MTP {mtp}",
                                 lambda: front_run(ctx, f"tokens-mtp-{mtp}", mtp, facts[mtp], native[mtp], bodies[mtp], "tokens",
                                                   sv["tokens_delivery"]["requests"])))
    except Exception as e:                           # the stage fails with its reason; the lease is still released
        out["error"] = f"{type(e).__name__}: {e}"
    finally:
        hold.give()
    out["gpu_after"] = dict(p1.gpu_free(cfg)[1], pod_gpu_waiters=len(p1.pod_waiters()))
    out["finished"] = now()
    out["held_seconds"] = round(sum(x["seconds"] for x in out["holds"]), 1)
    want = len(sv["mtp"]) + 1
    out["pass"] = ("error" not in out and len(out["front"]) == want and all(f["pass"] for f in out["front"])
                   and all(len(f["cases"]) == (len(sv["tokens_delivery"]["requests"]) if f["delivery"] == "tokens" else len(sv["request"]))
                           for f in out["front"]))
    return out


# --- summary ----------------------------------------------------------------------------------------------------------
def write_summary(ctx, results):
    stages = ctx["cfg"]["serve"]["stages"]
    ok = all(results.get(s, {}).get("pass") for s in stages)
    with open(os.path.join(ctx["res"], "summary.json"), "w") as f:
        json.dump({"recipe": ctx["recipe"].name, "host": socket.gethostname(), "date": now(), "pass": ok, "stages": results}, f, indent=1)
        f.write("\n")
    lines = [f"# Front end gate: {ctx['recipe'].name}", "", f"Host {socket.gethostname()}, {now()}. Stages run: {', '.join(results)}.", ""]
    for s in stages:
        if s in results:
            r = results[s]
            lines.append(f"- **{s}**: {'PASS' if r.get('pass') else 'FAIL'}" + (f" - {r['blocked']}" if r.get("blocked") else "")
                         + (f" - {r['error']}" if r.get("error") else ""))
    b = results.get("build", {})
    if b.get("binaries"):
        lines += ["", "Built: " + "; ".join(f"`{n}` ({v['bytes']} bytes, sha256 {v['sha256'][:16]})" for n, v in b["binaries"].items() if v)
                  + (f", {b['modules']} modules in the assembly." if b.get("modules") else ".")]
    n = results.get("nogpu", {})
    if n.get("servers"):
        lines += ["", f"No GPU: the front end over the stand-in executor (`executor.flerp.fake_abi`) in {len(n['servers'])} shapes, "
                  f"{n['checks']} checks.", "", "| stand-in executor | checks | passed | failed |", "|---|---|---|---|"]
        for s in n["servers"]:
            lines.append(f"| {s['server']}: `{' '.join(a for a in s['args'] if a.startswith('executor.')) or 'defaults'}` | {len(s['checks'])} | "
                         f"{sum(1 for c in s['checks'] if c['pass'])} | {', '.join(s['failed']) or '-'} |")
    g = results.get("gpu", {})
    if g.get("front"):
        lines += ["", "GPU: each request over HTTP against the native server's capture of the same request with the same MTP setting. "
                  "Equal step records are equal forwards.", "",
                  "| front end load | request | prompt tokens | prompt ids equal | tokens native / front end | tokens equal | steps equal | "
                  "finish | text equal | streamed: tokens, steps, text | delivery | rows a verify | drafts accepted / made | "
                  "decode tok/s native / front end (streamed) | pass |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for f in g["front"]:
            for c in f["cases"]:
                if "error" in c:
                    lines.append(f"| {f['label']} | {c['name']} | - | - | - | - | - | - | - | - | - | - | - | - | NO: {c['error']} |")
                    continue
                s = c.get("stream")
                lines.append(
                    f"| {f['label']} | {c['name']} | {c['prompt_tokens']} | {'yes' if c['prompt_equal'] else 'NO'} | "
                    f"{c['native_tokens']} / {c['tokens']} | {'yes' if c['tokens_equal'] else 'NO, from token ' + str(c['first_token_difference'])} | "
                    f"{'yes' if c['steps_equal'] else 'NO, from step ' + str(c['first_step_difference'])} | "
                    f"{c['finish_reason']}{'' if c['finish_equal'] else ' (native ' + str(c['native_finish_reason']) + ')'} | "
                    f"{'yes' if c['text_equal'] else 'not compared (the native reply is split)' if c['text_equal'] is None else 'NO' if c['text_required'] else 'no (chat: recorded)'} | "
                    + ("-" if s is None else ", ".join("yes" if s[k] else "NO" for k in ("tokens_equal", "steps_equal", "text_joins")))
                    + f" | {c['delivery']} ({c['sampled_by']}) | {', '.join(str(x) for x in c['rows_a_verify']) or '-'} | "
                    f"{c['accepted']} / {c['drafted']} | {c['native_decode_tps']} / {c['decode_tps']}"
                    + (f" ({s['decode_tps']})" if s else "") + f" | {'yes' if c['pass'] else 'NO'} |")
        extra = [(f["label"], e) for f in g["front"] for e in f["extra"]]
        if extra:
            lines += ["", "Also required: " + "; ".join(
                f"{label} {e['check']}: {'pass' if e['pass'] else 'FAIL'}"
                + (f" ({e['tokens']} tokens, delivery {e['delivery']}, mean logprob {e['mean_logprob']})" if "mean_logprob" in e else "")
                for label, e in extra) + "."]
        bad = [f"{n['mtp']}/{r['name']}" for n in g.get("native", []) for r in n["requests"] if r["cached_tokens"] != 0 or r["fed_steps"]]
        lines += ["", "Native requests that resumed from an earlier request's state or were fed ids by the engine: " + (", ".join(bad) or "none") + "."]
        lines += ["", f"GPU: the control-gpu lease was held {g.get('held_seconds')} s in {len(g.get('holds', []))} holds ("
                  + ", ".join(f"{x['for']}: {x['seconds']} s" for x in g.get("holds", [])) + "); "
                  + (f"waited for the pod's own GPU jobs {len(g['yields'])} times ("
                     + ", ".join(f"{y['waited_s']} s before {y['before']}" for y in g["yields"]) + ")." if g.get("yields")
                     else "no pod job was waiting when a hold began."),
                  "", "The front end calls the executor only through `include/flerp/executor.h`; its token ids and step records are "
                  "the `flerp` object of its HTTP responses (`\"flerp_trace\": true`). Native tokens and step records are read from "
                  "the server's capture files (`--capture-dir`). Decode tok/s is each server's own `timings.predicted_per_second` "
                  "for the one request: recorded, not a criterion."]
    with open(os.path.join(ctx["res"], "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--stages", default=None)
    a = ap.parse_args()
    cat = Catalog(ROOT)
    recipe = cat.recipe(a.recipe)
    if recipe is None:
        sys.exit(f"no recipe {a.recipe}")
    cfg = p1.load_config(os.path.join(ROOT, recipe.gate["config"]))
    stages = cfg["serve"]["stages"]
    asked = a.stages.split(",") if a.stages else stages
    res = os.path.join(ROOT, "gate", "results", recipe.name)
    os.makedirs(res, exist_ok=True)
    ctx = {"cat": cat, "recipe": recipe, "cfg": cfg, "res": res,
           "scratch": os.path.join(ROOT, cfg["host"].get("scratch", "build/gate-scratch-serve")),
           "ninja": os.path.join(ROOT, "build", recipe.name, "ninja")}
    os.makedirs(ctx["scratch"], exist_ok=True)
    results = {}
    prior = os.path.join(res, "summary.json")
    if os.path.isfile(prior) and set(asked) != set(stages):
        with open(prior) as f:
            results = json.load(f).get("stages", {})
    fns = {"check": stage_check, "build": stage_build, "nogpu": stage_nogpu, "gpu": stage_gpu}
    for s in STAGES:
        if s not in stages or s not in asked:
            continue
        print(f"== {s}", flush=True)
        results[s] = fns[s](ctx)
        print(f"   {'PASS' if results[s].get('pass') else 'FAIL'} {results[s].get('blocked', '')}{results[s].get('error', '')}", flush=True)
        if not results[s].get("pass"):
            break
    return 0 if write_summary(ctx, {s: results[s] for s in STAGES if s in results}) else 1


if __name__ == "__main__":
    sys.exit(main())
