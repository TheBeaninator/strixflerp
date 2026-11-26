#!/usr/bin/env python3
"""Factory replay (card 2587bc1599): a frozen set of recorded conversations played at N concurrent threads against one
OpenAI-compatible engine, with the throughput numbers every later gate compares and a per-stream identity check.

  factory_replay.py freeze --corpus DIR --out SET.json [--targets strixite,pair] [--max-turns T] [--max-streams S]
  factory_replay.py verify --set SET.json --corpus DIR
  factory_replay.py run    --set SET.json --corpus DIR --url http://HOST:PORT --model NAME --concurrency N
                           [--reps 3] [--max-tokens 512] [--solo all|none|K] [--solo-passes 1] [--out DIR]
                           [--strip KEY ...] [--set-json KEY=JSON ...] [--timeout 1800] [--wait-ready SECS]
                           [--expect-slots N]

freeze reads a corpus directory (docs/REPLAY.md: CORPUS/<target>/<id>.json), keeps the complete, replayable
/v1/chat/completions exchanges with status 200, and chains them into streams: an exchange continues the stream whose
last request's messages are a prefix of its own. The set file lists every stream's turns with the sha256 of each
exchange file and a set_sha256 over all of them. It holds no message text. run refuses a corpus that does not match.

run plays every stream turn by turn, each turn resending the stream's history as recorded (the growing prompt of the
real thread; the engine's own reply is measured and hashed, never fed back, so every engine and every rep gets the
same prompts). Every request is greedy: temperature 0, a fixed seed, stream false, max_tokens capped. Phases:
  solo-K   the chosen streams one at a time (concurrency 1): the identity reference and the cache warm-up;
  rep-K    all streams with N in flight (N workers, each takes the next stream and plays its turns in order).
Printed per phase: committed tokens a second (completion tokens over the phase's wall), p50 and p95 turn time,
prefilled and cached tokens a request (from usage), and per rep how many streams gave the reply they gave alone.
Exit: 0 clean, 3 a stream differed from its solo run, 4 a request failed, 2 usage or a corpus that does not match.
Python stdlib only. It starts, stops and holds nothing: the engine URL is somebody's window (docs/REPLAY.md).
"""
import argparse
import hashlib
import json
import os
import queue
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

FORMAT = "factory-replay-set-v1"
CHAT = "/v1/chat/completions"


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def canon(o):
    return json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def read(path):
    with open(path, "rb") as fh:
        return fh.read()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------- freeze

def usable(e):
    """The reason an exchange is left out, or None."""
    if e.get("path") != CHAT or e.get("method", "POST") != "POST":
        return "not a chat completion"
    if e.get("status") != 200:
        return "status %s" % e.get("status")
    if e.get("replayable") is False:
        return "not replayable"
    if e.get("req_complete") is False or e.get("resp_complete") is False:
        return "incomplete"
    r = e.get("request")
    if not isinstance(r, dict) or not isinstance(r.get("messages"), list) or not r["messages"]:
        return "no messages"
    return None


def recorded(e):
    """Recorded sizes of an exchange (numbers only), for the set file's estimate."""
    out = {"wall_s": None, "prompt_tokens": None, "cached_tokens": None, "completion_tokens": None}
    try:
        out["wall_s"] = round(float(e["t_end"]) - float(e["t_start"]), 2)
    except (KeyError, TypeError, ValueError):
        pass
    try:
        u = json.loads(e.get("response_raw") or "").get("usage") or {}
    except (ValueError, AttributeError):
        u = {}
    out["prompt_tokens"] = u.get("prompt_tokens")
    out["completion_tokens"] = u.get("completion_tokens")
    out["cached_tokens"] = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    return out


def prefix_keys(messages):
    """keys[k-1] identifies the first k messages."""
    h, keys = hashlib.sha256(), []
    for m in messages:
        h.update(canon(m))
        h.update(b"\x00")
        keys.append(h.hexdigest())
    return keys


def freeze(corpus, targets=None, max_turns=0, max_streams=0):
    corpus = os.path.abspath(corpus)
    rows, skipped = [], {}
    for target in sorted(os.listdir(corpus)):
        d = os.path.join(corpus, target)
        if not os.path.isdir(d) or target == "synthetic" or (targets and target not in targets):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".json"):
                continue
            raw = read(os.path.join(d, name))
            try:
                e = json.loads(raw)
            except ValueError:
                skipped["not JSON"] = skipped.get("not JSON", 0) + 1
                continue
            why = usable(e)
            if why:
                skipped[why] = skipped.get(why, 0) + 1
                continue
            rows.append({"id": e.get("id") or name[:-5], "file": target + "/" + name, "sha256": sha_bytes(raw),
                         "target": target, "t": e.get("t_start") or 0, "keys": prefix_keys(e["request"]["messages"]),
                         "recorded": recorded(e)})
    rows.sort(key=lambda r: (r["t"], r["id"]))
    chains, tails = [], {}                      # tails: (target, n messages, prefix key) -> chain index
    for r in rows:
        n, found = len(r["keys"]), None
        for k in range(n - 1, 0, -1):           # the longest earlier request that is a strict prefix of this one
            found = tails.get((r["target"], k, r["keys"][k - 1]))
            if found is not None:
                del tails[(r["target"], k, r["keys"][k - 1])]
                break
        if found is None:
            found = len(chains)
            chains.append([])
        chains[found].append(r)
        tails[(r["target"], n, r["keys"][-1])] = found
    streams = []
    for c in chains:
        parts = [c] if max_turns <= 0 else [c[i:i + max_turns] for i in range(0, len(c), max_turns)]
        for p in parts:
            streams.append({"id": "s%04d" % (len(streams) + 1), "target": p[0]["target"],
                            "turns": [{"id": r["id"], "file": r["file"], "sha256": r["sha256"],
                                       "messages": len(r["keys"]), "recorded": r["recorded"]} for r in p]})
    if max_streams > 0:
        streams = streams[:max_streams]
    s = {"format": FORMAT, "corpus": os.path.basename(corpus), "frozen_at": now(),
         "options": {"targets": sorted(targets) if targets else None, "max_turns": max_turns, "max_streams": max_streams},
         "skipped": skipped, "streams": streams}
    s["exchanges"] = sum(len(x["turns"]) for x in streams)
    s["set_sha256"] = set_sha(s)
    return s


def set_sha(s):
    lines = ["%s %s %s" % (x["id"], t["id"], t["sha256"]) for x in s["streams"] for t in x["turns"]]
    return sha_bytes("\n".join(lines).encode())


def verify(s, corpus):
    """Problems of a set against a corpus directory: [] when every exchange file has the frozen sha256."""
    bad = []
    if s.get("format") != FORMAT:
        return ["set format is %r, not %s" % (s.get("format"), FORMAT)]
    if set_sha(s) != s.get("set_sha256"):
        bad.append("set_sha256 does not match the set's own turns")
    for x in s["streams"]:
        for t in x["turns"]:
            p = os.path.join(corpus, t["file"])
            try:
                got = sha_bytes(read(p))
            except OSError as err:
                bad.append("%s: %s" % (t["file"], err.strerror))
                continue
            if got != t["sha256"]:
                bad.append("%s: sha256 differs from the frozen set" % t["file"])
    return bad


def estimate(s):
    rec = [t["recorded"] for x in s["streams"] for t in x["turns"]]
    def tot(k):
        return sum(r[k] for r in rec if isinstance(r.get(k), (int, float)))
    return {"recorded_wall_s": round(tot("wall_s"), 1), "recorded_prompt_tokens": tot("prompt_tokens"),
            "recorded_cached_tokens": tot("cached_tokens"), "recorded_completion_tokens": tot("completion_tokens")}


def first_streams(s, n):
    """The first n streams in the order of their first turn's sha256: a fixed rule that cuts a set to fit a hold.
    The cut set keeps the full set's set_sha256 and names what it kept in "selection"."""
    order = sorted(s["streams"], key=lambda x: x["turns"][0]["sha256"])
    kept = order[:n]
    out = dict(s, streams=kept, exchanges=sum(len(x["turns"]) for x in kept))
    out["selection"] = {"rule": "first %d streams by the sha256 of their first turn, ascending" % n,
                        "streams": [x["id"] for x in kept], "of_streams": len(s["streams"]),
                        "of_exchanges": s["exchanges"]}
    return out


# ---------------------------------------------------------------- run

def build_body(e, model, max_tokens, strip=(), extra=None):
    body = dict(e["request"])
    for k in strip:
        body.pop(k, None)
    body["model"] = model
    body["temperature"] = 0
    body["stream"] = False
    body.pop("stream_options", None)
    if "seed" not in body:                      # the rule of gate/replay.py: the first 31 bits of sha256(exchange id)
        body["seed"] = int(hashlib.sha256(str(e.get("id")).encode()).hexdigest()[:8], 16) & 0x7FFFFFFF
    if max_tokens > 0:
        rec = body.get("max_tokens") or body.get("max_completion_tokens")
        body.pop("max_completion_tokens", None)
        body["max_tokens"] = min(int(rec), max_tokens) if rec else max_tokens
    body.update(extra or {})
    return body


def signature(reply):
    """(kind, sha256) of what the engine drew: its token ids when the reply carries them, else its text."""
    ch = (reply.get("choices") or [{}])[0]
    ids = (reply.get("flerp") or {}).get("token_ids") or ch.get("token_ids") or reply.get("token_ids")
    if ids:
        return "ids", sha_bytes(canon(ids))
    m = ch.get("message") or {}
    calls = [[(c.get("function") or {}).get("name"), (c.get("function") or {}).get("arguments")]
             for c in m.get("tool_calls") or []]    # call ids are drawn at random per reply
    return "text", sha_bytes(canon([m.get("content"), m.get("reasoning_content"), calls, ch.get("finish_reason")]))


def usage_numbers(reply):
    u = reply.get("usage") or {}
    t = reply.get("timings") or {}
    prompt, comp = u.get("prompt_tokens"), u.get("completion_tokens")
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is None:
        cached = t.get("cache_n")
    prefilled = None
    if prompt is not None and cached is not None:
        prefilled = prompt - cached
    elif t.get("prompt_n") is not None:
        prefilled = t.get("prompt_n")
    return {"prompt_tokens": prompt, "completion_tokens": comp, "cached_tokens": cached, "prefilled_tokens": prefilled}


def endpoint(url):
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return url + ("/chat/completions" if url.endswith("/v1") else CHAT)


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def base_url(url):
    url = endpoint(url)[:-len("/chat/completions")]
    return url[:-3] if url.endswith("/v1") else url


def wait_ready(url, secs, every=5):
    """Poll GET /health until it answers 200; False when secs run out."""
    end = time.monotonic() + secs
    while True:
        try:
            with urllib.request.urlopen(base_url(url) + "/health", timeout=10) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        if time.monotonic() >= end:
            return False
        time.sleep(every)


def engine_slots(url):
    """The slot count the engine reports on /metrics (strix_slots), or None."""
    try:
        with urllib.request.urlopen(base_url(url) + "/metrics", timeout=10) as r:
            for line in r.read().decode("utf-8", "replace").splitlines():
                if line.startswith("strix_slots"):
                    return int(float(line.split()[-1]))
    except (urllib.error.URLError, OSError, ValueError):
        pass
    return None


class Phase:
    """One pass over some streams with a fixed number in flight."""

    def __init__(self, name, streams, concurrency, cfg):
        self.name, self.streams, self.concurrency, self.cfg = name, streams, concurrency, cfg
        self.turns, self.lock = [], threading.Lock()
        self.in_flight = self.max_in_flight = 0
        self.t0 = self.wall = self.sat_end = None

    def play(self, stream):
        cfg = self.cfg
        for i, t in enumerate(stream["turns"]):
            e = json.loads(read(os.path.join(cfg["corpus"], t["file"])))
            body = build_body(e, cfg["model"], cfg["max_tokens"], cfg["strip"], cfg["extra"])
            del e
            row = {"stream": stream["id"], "turn": i, "exchange": t["id"]}
            with self.lock:
                self.in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self.in_flight)
            t0 = time.monotonic()
            try:
                reply = post(cfg["url"], body, cfg["timeout"])
                row["wall_s"] = time.monotonic() - t0
                row["sig_kind"], row["sig"] = signature(reply)
                row.update(usage_numbers(reply))
            except (urllib.error.URLError, OSError, ValueError) as err:
                row["wall_s"] = time.monotonic() - t0
                code = getattr(err, "code", None)
                row["error"] = ("HTTP %s" % code) if code else type(err).__name__ + ": " + str(err)[:120]
            row["t_end"] = time.monotonic() - self.t0
            with self.lock:
                self.in_flight -= 1
                self.turns.append(row)
            if cfg["progress"]:
                print("%s %s %s turn %d/%d: %s in %.1f s" % (
                    time.strftime("%H:%M:%S"), self.name, stream["id"], i + 1, len(stream["turns"]),
                    row.get("error") or "%s tokens" % row.get("completion_tokens"), row["wall_s"]),
                    file=sys.stderr, flush=True)

    def run(self):
        q = queue.Queue()
        for s in self.streams:
            q.put(s)

        def worker():
            while True:
                try:
                    s = q.get_nowait()
                except queue.Empty:
                    with self.lock:             # from here on fewer than N streams are in flight
                        if self.sat_end is None:
                            self.sat_end = time.monotonic() - self.t0
                    return
                self.play(s)

        self.t0 = time.monotonic()
        th = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, self.concurrency))]
        for t in th:
            t.start()
        for t in th:
            t.join()
        self.wall = time.monotonic() - self.t0
        return self

    def by_stream(self):
        out = {}
        for r in sorted(self.turns, key=lambda r: (r["stream"], r["turn"])):
            out.setdefault(r["stream"], []).append(r)
        return out

    def summary(self):
        ok = [r for r in self.turns if "error" not in r]
        walls = sorted(r["wall_s"] for r in ok)
        toks = sum(r.get("completion_tokens") or 0 for r in ok)
        sat = [r for r in ok if self.sat_end is not None and r["t_end"] <= self.sat_end]
        sat_toks = sum(r.get("completion_tokens") or 0 for r in sat)
        return {"phase": self.name, "concurrency": self.concurrency, "streams": len(self.streams),
                "turns": len(self.turns), "errors": len(self.turns) - len(ok), "wall_s": round(self.wall, 3),
                "max_in_flight": self.max_in_flight, "completion_tokens": toks,
                "committed_tokens_per_s": round(toks / self.wall, 3) if self.wall else 0.0,
                "saturated_s": round(self.sat_end, 3) if self.sat_end is not None else None,
                "committed_tokens_per_s_saturated": round(sat_toks / self.sat_end, 3) if self.sat_end else None,
                "turn_p50_s": rnd(percentile(walls, 50)), "turn_p95_s": rnd(percentile(walls, 95)),
                "prefilled_tokens_per_request": rnd(mean([r.get("prefilled_tokens") for r in ok]), 1),
                "cached_tokens_per_request": rnd(mean([r.get("cached_tokens") for r in ok]), 1),
                "prompt_tokens_per_request": rnd(mean([r.get("prompt_tokens") for r in ok]), 1)}


def percentile(sorted_vals, p):
    """Nearest rank: the smallest value with at least p % of the values at or below it."""
    if not sorted_vals:
        return None
    k = max(1, -(-len(sorted_vals) * p // 100))
    return sorted_vals[int(k) - 1]


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def rnd(v, n=3):
    return None if v is None else round(v, n)


def compare(ref, got):
    """One stream against its solo run: 'identical', or the first turn that differs or failed."""
    for i, a in enumerate(ref):
        b = got[i] if i < len(got) else None
        if b is None or "error" in a or "error" in b:
            return {"result": "error", "turn": i}
        if a["sig"] != b["sig"]:
            return {"result": "differs", "turn": i, "exchange": a["exchange"],
                    "completion_tokens": [a.get("completion_tokens"), b.get("completion_tokens")]}
    return {"result": "identical", "turns": len(ref)}


def identity(ref_phase, phase):
    ref, got, out = ref_phase.by_stream(), phase.by_stream(), {}
    for sid, turns in ref.items():
        out[sid] = compare(turns, got.get(sid, []))
    n = lambda k: sum(1 for v in out.values() if v["result"] == k)
    kinds = sorted({r.get("sig_kind") for r in ref_phase.turns if r.get("sig_kind")})
    return {"phase": phase.name, "against": ref_phase.name, "compared_by": "+".join(kinds) or None,
            "streams_checked": len(out), "identical": n("identical"), "differs": n("differs"), "errors": n("error"),
            "not_identical": {k: v for k, v in sorted(out.items()) if v["result"] != "identical"}}


def run(s, corpus, url, model, concurrency, reps=3, max_tokens=512, solo="all", solo_passes=1, strip=(), extra=None,
        timeout=1800, progress=False, checkpoint=None):
    cfg = {"corpus": corpus, "url": endpoint(url), "model": model, "max_tokens": max_tokens, "strip": tuple(strip),
           "extra": extra or {}, "timeout": timeout, "progress": progress}
    streams = s["streams"]
    solo_streams = [] if solo == "none" else streams if solo == "all" else streams[:int(solo)]
    res = {"tool": "factory_replay", "started": now(), "set_sha256": s["set_sha256"], "corpus": s.get("corpus"),
           "exchanges": s["exchanges"], "streams": len(streams), "selection": s.get("selection"),
           "url": cfg["url"], "model": model,
           "concurrency": concurrency, "reps": reps, "max_tokens": max_tokens, "solo_streams": len(solo_streams),
           "solo_passes": solo_passes if solo_streams else 0, "strip": list(strip), "extra": cfg["extra"],
           "phases": [], "identity": [], "turns": {}}
    solos, ref = [], None

    def note(p):
        res["phases"].append(p.summary())
        res["turns"][p.name] = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
                                for r in sorted(p.turns, key=lambda r: (r["stream"], r["turn"]))]
        rs = [x for x in res["phases"] if x["phase"].startswith("rep-")]

        def med(k):
            vals = [x[k] for x in rs if x[k] is not None]
            return rnd(statistics.median(vals)) if vals else None
        res["median_of_reps"] = {k: med(k) for k in ("committed_tokens_per_s", "committed_tokens_per_s_saturated",
                                                     "turn_p50_s", "turn_p95_s", "prefilled_tokens_per_request",
                                                     "cached_tokens_per_request")} if rs else {}
        res["reps_done"] = len(rs)
        res["errors"] = sum(x["errors"] for x in res["phases"])
        res["streams_not_identical"] = sum(x["differs"] + x["errors"] for x in res["identity"]
                                           if x["phase"].startswith("rep-"))
        if checkpoint:
            checkpoint(res)                      # a job stopped by its time limit keeps the phases it finished

    for k in range(solo_passes if solo_streams else 0):
        solos.append(Phase("solo-%d" % (k + 1), solo_streams, 1, cfg).run())
        if k + 1 == solo_passes:
            ref = solos[-1]                      # the last solo pass is the reference: the cache is as warm as in the reps
            for p in solos[:-1]:
                res["identity"].append(identity(ref, p))
        note(solos[-1])
    for k in range(reps):
        p = Phase("rep-%d" % (k + 1), streams, concurrency, cfg).run()
        if ref:
            res["identity"].append(identity(ref, p))
        note(p)
    if not res["phases"]:
        res.update({"median_of_reps": {}, "reps_done": 0, "errors": 0, "streams_not_identical": 0})
    res["finished"] = now()
    return res


def fmt(v, spec="%.2f"):
    return "n/a" if v is None else spec % v


def report(res):
    L = ["factory replay: set %s (%s), %d exchanges in %d streams" % (res["set_sha256"][:12], res["corpus"],
                                                                     res["exchanges"], res["streams"]),
         "engine %s model %s, concurrency %d, %d of %d reps done, max_tokens %s, greedy" % (
             res["url"], res["model"], res["concurrency"], res.get("reps_done", 0), res["reps"],
             res["max_tokens"] or "as recorded"),
         "selection: %s: %s (of %d streams, %d exchanges)" % (
             res["selection"]["rule"], " ".join(res["selection"]["streams"]), res["selection"]["of_streams"],
             res["selection"]["of_exchanges"]) if res.get("selection") else "selection: the whole set", "",
         "| phase | in flight | turns | errors | wall s | committed tokens a second | while saturated | turn time p50 s | p95 s "
         "| prefilled tokens a request | cached tokens a request |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in res["phases"]:
        L.append("| %s | %d | %d | %d | %.1f | %.2f | %s | %s | %s | %s | %s |" % (
            p["phase"], p["concurrency"], p["turns"], p["errors"], p["wall_s"], p["committed_tokens_per_s"],
            fmt(p["committed_tokens_per_s_saturated"]), fmt(p["turn_p50_s"]), fmt(p["turn_p95_s"]),
            fmt(p["prefilled_tokens_per_request"], "%.0f"), fmt(p["cached_tokens_per_request"], "%.0f")))
    m = res.get("median_of_reps") or {}
    if m:
        L += ["", "median of %d reps: committed tokens a second %s (while saturated %s), turn time p50 %s s, p95 %s s, "
                  "prefilled tokens a request %s, cached tokens a request %s" % (
                      res.get("reps_done", 0), fmt(m["committed_tokens_per_s"]), fmt(m["committed_tokens_per_s_saturated"]),
                      fmt(m["turn_p50_s"]), fmt(m["turn_p95_s"]), fmt(m["prefilled_tokens_per_request"], "%.0f"),
                      fmt(m["cached_tokens_per_request"], "%.0f"))]
    L.append("")
    if not res["identity"]:
        L.append("identity: not checked (no solo pass)")
    for i in res["identity"]:
        L.append("identity %s against %s (by %s): %d of %d streams identical, %d differ, %d with a failed request" % (
            i["phase"], i["against"], i["compared_by"], i["identical"], i["streams_checked"], i["differs"], i["errors"]))
        for sid, v in list(i["not_identical"].items())[:20]:
            L.append("  %s: %s at turn %d%s" % (sid, v["result"], v["turn"] + 1,
                                              " (%s)" % v["exchange"] if v.get("exchange") else ""))
    if res["solo_streams"] and res["solo_streams"] < res["streams"]:
        L.append("identity covers the first %d of %d streams (the ones run alone)" % (res["solo_streams"], res["streams"]))
    return "\n".join(L) + "\n"


def parse_extra(pairs):
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--corpus", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--targets", default="")
    f.add_argument("--max-turns", type=int, default=0, help="split longer conversations into streams of at most this many turns")
    f.add_argument("--max-streams", type=int, default=0)
    v = sub.add_parser("verify")
    v.add_argument("--set", required=True)
    v.add_argument("--corpus", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", required=True)
    r.add_argument("--corpus", required=True)
    r.add_argument("--url", required=True, help="engine base URL (http://HOST:PORT or .../v1)")
    r.add_argument("--model", required=True, help="model name sent in every request")
    r.add_argument("--concurrency", type=int, required=True)
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--max-tokens", type=int, default=512, help="cap on new tokens a turn; 0 = as recorded")
    r.add_argument("--first-streams", type=int, default=0,
                   help="play only the first N streams by the sha256 of their first turn (0 = the whole set)")
    r.add_argument("--solo", default="all", help="streams also run alone for the identity check: all, none or a count")
    r.add_argument("--solo-passes", type=int, default=1)
    r.add_argument("--strip", action="append", default=[], help="request key to drop (repeatable)")
    r.add_argument("--set-json", action="append", default=[], help="KEY=JSON added to every request (repeatable)")
    r.add_argument("--timeout", type=float, default=1800)
    r.add_argument("--wait-ready", type=float, default=0, help="first poll GET /health for up to this many seconds")
    r.add_argument("--expect-slots", type=int, default=0, help="refuse to run unless /metrics strix_slots says this")
    r.add_argument("--out", help="directory for result.json and summary.md")
    r.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "freeze":
        s = freeze(a.corpus, set(x for x in a.targets.split(",") if x) or None, a.max_turns, a.max_streams)
        with open(a.out, "w") as fh:
            json.dump(s, fh, indent=1, sort_keys=True)
        print("frozen %d exchanges in %d streams, set_sha256 %s -> %s" % (s["exchanges"], len(s["streams"]), s["set_sha256"], a.out))
        print("skipped: %s" % json.dumps(s["skipped"], sort_keys=True))
        print("as recorded: %s" % json.dumps(estimate(s), sort_keys=True))
        return 0
    s = json.loads(read(a.set))
    bad = verify(s, a.corpus)
    if bad:
        print("corpus does not match the frozen set (%d problems):" % len(bad), file=sys.stderr)
        for b in bad[:20]:
            print("  " + b, file=sys.stderr)
        return 2
    if a.cmd == "verify":
        print("ok: %d exchanges in %d streams match set_sha256 %s" % (s["exchanges"], len(s["streams"]), s["set_sha256"]))
        return 0
    if a.first_streams:
        s = first_streams(s, a.first_streams)
    if a.solo not in ("all", "none") and not a.solo.isdigit():
        ap.error("--solo is all, none or a count")
    if a.wait_ready and not wait_ready(a.url, a.wait_ready):
        print("engine at %s not healthy after %.0f s" % (base_url(a.url), a.wait_ready), file=sys.stderr)
        return 4
    slots = engine_slots(a.url)
    if a.expect_slots and slots != a.expect_slots:
        print("engine reports strix_slots %s, expected %d: not run" % (slots, a.expect_slots), file=sys.stderr)
        return 2
    def save(res):
        res["engine_slots"] = slots
        text = report(res) + ("engine reports strix_slots %d\n" % slots if slots is not None else "")
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            for name, data in (("result.json", json.dumps(res, indent=1, sort_keys=True)), ("summary.md", text)):
                with open(os.path.join(a.out, name + ".tmp"), "w") as fh:
                    fh.write(data)
                os.replace(os.path.join(a.out, name + ".tmp"), os.path.join(a.out, name))
        return text

    res = run(s, a.corpus, a.url, a.model, a.concurrency, a.reps, a.max_tokens, a.solo, a.solo_passes, a.strip,
              parse_extra(a.set_json), a.timeout, progress=not a.quiet, checkpoint=save)
    text = save(res)
    sys.stdout.write(text)
    return 4 if res["errors"] else 3 if res["streams_not_identical"] else 0


if __name__ == "__main__":
    sys.exit(main())
