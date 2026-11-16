#!/usr/bin/env python3
"""The replay gate (FLERP-SWAP, docs/REPLAY.md): two servers, one recorded corpus, compared exchange by exchange.

  gate/replay.py --corpus DIR --a URL --b URL --out RESULTS [options]

Each replayable exchange of the corpus (tools/replay/scrub.py's schema; DIR or DIR/<target>) is sent to server A and
then to server B, one at a time, in recorded order (t_start), so every conversation's turns arrive in their order.
A request body without "seed" gets one derived from the exchange id (strixite seeds unseeded requests at random:
catalog/strixite/serve/http/src/serve/http_server.cpp:466). For each exchange it compares:

  status    the HTTP status
  tokens    the reply's token ids, where a server can say them: kind flerp (serve.flerp.http: "flerp_trace": true is
            added to the body and the reply's "flerp" object is read, then removed before the body comparison), kind
            origin with --X-capture-dir (strixite --capture-dir files). Otherwise "n/a" for that exchange
  body      the body bytes after dropping the volatile fields (--volatile, default: id, created, timings,
            system_fingerprint at the top level); a JSON body is compared as parsed values, SSE event by event, any
            other body byte for byte. /cache/export answers are compared by size and sha256
  headers   the headers named by --header (default: content-type)

/cache/import has no recorded body (the corpus keeps size and sha256 only). It is replayed when an earlier
/cache/export of the corpus answered those bytes: each server then gets back its own export of that exchange.
Otherwise, and for remote images or incomplete exchanges, the exchange is listed as skipped with the reason.

Empty-cache start: with --a-cmd / --b-cmd the gate starts each server itself (a shell command; it must serve the
URL; a fresh process with an empty prompt cache), waits for /health and stops it at the end. Servers started by
someone else are taken as fresh only with --assume-fresh, which the summary records. The verdict is PASS when every
replayed exchange is equal on all four and both servers started fresh. Results: RESULTS/results.json (per exchange:
equal or not, first differing token index and body path, never message text) and RESULTS/summary.md.
"""
import argparse
import datetime
import glob
import hashlib
import http.client
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import summary as SM  # noqa: E402  (results.json / summary.md, config_sha256)
VOLATILE = ("id", "created", "timings", "system_fingerprint")
SKIP_ROUTES = ("/health", "/metrics", "/dashboard", "/stats")   # state of the moment, not a reply


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def seed_of(eid):
    """The seed injected into a request recorded without one: a function of the exchange id only."""
    return int(hashlib.sha256(eid.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


# --- the corpus --------------------------------------------------------------------------------------------------
def load_corpus(path, target=None):
    d = os.path.join(path, target) if target else path
    out = []
    for p in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)):
        rel = os.path.relpath(p, path)
        if os.path.basename(p) == "manifest.json" or rel.startswith("synthetic" + os.sep):
            continue
        with open(p, encoding="utf-8") as f:
            e = json.load(f)
        if "id" in e and "method" in e:
            out.append(e)
    out.sort(key=lambda e: (e.get("t_start") or 0, e["id"]))
    return out


def skip_reason(e):
    route = e.get("path", "").split("?")[0]
    if e.get("replayable") is False:
        return e.get("not_replayable") or "marked not replayable"
    if not (e.get("req_complete", True) and e.get("resp_complete", True)) or "status" not in e:
        return "incomplete exchange"
    if route in SKIP_ROUTES:
        return "state route (not a reply)"
    if "request_raw" in e:
        return "request body is not JSON"
    return None


# --- HTTP --------------------------------------------------------------------------------------------------------
class Target:
    def __init__(self, label, url, kind, capture_dir=None, cmd=None, assume_fresh=False, log_dir=None):
        u = urllib.parse.urlsplit(url)
        self.label, self.host, self.port, self.kind = label, u.hostname, u.port or 80, kind
        self.capture_dir, self.cmd, self.proc = capture_dir, cmd, None
        self.fresh = assume_fresh
        self.log_dir = log_dir
        self.exports = {}      # recorded export sha256 -> this server's own export bytes (a temp file)
        self.n_capture = 0

    def start(self, timeout):
        if not self.cmd:
            return
        self.log_f = open(os.path.join(self.log_dir, f"server-{self.label}.log"), "wb")
        self.proc = subprocess.Popen(self.cmd, shell=True, stdout=self.log_f, stderr=subprocess.STDOUT, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server {self.label} exited with {self.proc.returncode} during startup")
            try:
                if self.request("GET", "/health", None, {})[0] == 200:
                    self.fresh = True
                    return
            except OSError:
                pass
            time.sleep(0.2)
        raise RuntimeError(f"server {self.label} not ready after {timeout} s")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(30)
        if self.proc:
            self.log_f.close()

    def request(self, method, path, body, headers, sink=None):
        """(status, headers, body bytes or None, size, sha256). With `sink` the body is streamed to it."""
        c = http.client.HTTPConnection(self.host, self.port, timeout=3600)
        try:
            c.request(method, path, body=body, headers=headers)
            r = c.getresponse()
            sha, n, buf = hashlib.sha256(), 0, bytearray()
            while True:
                b = r.read(1 << 16)
                if not b:
                    break
                sha.update(b)
                n += len(b)
                if sink:
                    sink.write(b)
                else:
                    buf += b
            return r.status, {k.lower(): v for k, v in r.getheaders()}, None if sink else bytes(buf), n, sha.hexdigest()
        finally:
            c.close()


# --- comparisons -------------------------------------------------------------------------------------------------
def drop_paths(o, paths):
    for p in paths:
        parts = p.split(".")
        def rm(x, i):
            if i == len(parts) - 1:
                if isinstance(x, dict):
                    x.pop(parts[i], None)
                elif isinstance(x, list) and parts[i] == "*":
                    x.clear()
                return
            k = parts[i]
            kids = (x if isinstance(x, list) else list(x.values()) if isinstance(x, dict) else []) if k == "*" else \
                   [x[k]] if isinstance(x, dict) and k in x else []
            for y in kids:
                rm(y, i + 1)
        rm(o, 0)
    return o


def first_path(a, b, at=""):
    """The first path where two JSON values differ, or None."""
    if type(a) is not type(b):
        return at or "$"
    if isinstance(a, dict):
        for k in list(a) + [k for k in b if k not in a]:
            if k not in a or k not in b:
                return f"{at}.{k}" if at else k
            p = first_path(a[k], b[k], f"{at}.{k}" if at else k)
            if p:
                return p
        if list(a) != list(b):
            return (at or "$") + " (key order)"
        return None
    if isinstance(a, list):
        for i, (x, y) in enumerate(zip(a, b)):
            p = first_path(x, y, f"{at}[{i}]")
            if p:
                return p
        return None if len(a) == len(b) else f"{at}[{min(len(a), len(b))}]"
    return None if a == b else (at or "$")


def parse_body(raw, ctype):
    """('json', value) / ('sse', [events]) / ('bytes', raw)."""
    if raw is None:
        return "bytes", None
    if "text/event-stream" in (ctype or ""):
        ev = []
        for line in raw.decode("utf-8", "replace").splitlines():
            if line.startswith("data: "):
                d = line[6:]
                try:
                    ev.append(json.loads(d))
                except ValueError:
                    ev.append(d)
        return "sse", ev
    try:
        return "json", json.loads(raw)
    except ValueError:
        return "bytes", raw


def trace_of(kind, value):
    """A flerp_trace reply's token ids, removed from the value: {"token_ids": every id the reply drew,
    "sequence_ids": the ids fed into the sequence (the last drawn one is not)}."""
    if kind == "json" and isinstance(value, dict) and isinstance(value.get("flerp"), dict):
        t = value.pop("flerp")
        return {k: t[k] for k in ("token_ids", "sequence_ids") if k in t} or None
    if kind == "sse":
        out, ids, seen = {}, [], False
        for ev in value:
            if isinstance(ev, dict) and isinstance(ev.get("flerp"), dict):
                t = ev.pop("flerp")
                if ev.get("choices"):
                    ids += t.get("token_ids", [])
                    seen = True
                elif "sequence_ids" in t:
                    out["sequence_ids"] = t["sequence_ids"]
        if seen:
            out["token_ids"] = ids
        return out or None
    return None


def normalise(kind, value, volatile):
    if kind == "json":
        return drop_paths(value, volatile)
    if kind == "sse":
        return [drop_paths(v, volatile) if isinstance(v, dict) else v for v in value]
    return value


def first_diff(a, b):
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), None if len(a) == len(b) else n)


# --- one exchange ------------------------------------------------------------------------------------------------
def send(t, e, scratch):
    """Replays exchange e on target t: (answer dict, or a skip reason)."""
    route = e["path"].split("?")[0]
    headers = {k: v for k, v in (e.get("req_headers") or {}).items()
               if k not in ("host", "content-length", "transfer-encoding", "connection", "accept-encoding")}
    body, sink, tmp = None, None, None
    tokens = None
    if route == "/cache/import":
        src = t.exports.get(e.get("req_body_sha256"))
        if src is None:
            return None, "import body not recorded and no earlier export of these bytes"
        body = open(src, "rb")
        headers["content-length"] = str(os.path.getsize(src))
    elif e.get("request") is not None:
        req = json.loads(json.dumps(e["request"]))
        if isinstance(req, dict):
            if "seed" not in req and route.startswith("/v1/"):
                req["seed"] = seed_of(e["id"])
            if t.kind == "flerp" and route.startswith("/v1/"):
                req["flerp_trace"] = True
        body = json.dumps(req, ensure_ascii=False).encode()
        headers.setdefault("content-type", "application/json")
    if route == "/cache/export":
        fd, tmp = tempfile.mkstemp(dir=scratch, prefix=f"export-{t.label}-")
        sink = os.fdopen(fd, "wb")
    try:
        status, rh, raw, size, sha = t.request(e["method"], e["path"], body, headers, sink)
    finally:
        if sink:
            sink.close()
        if hasattr(body, "close"):
            body.close()
    if tmp and e.get("resp_body_sha256"):
        t.exports[e["resp_body_sha256"]] = tmp
    kind, value = parse_body(raw, rh.get("content-type"))
    if t.kind == "flerp":
        tokens = trace_of(kind, value)
    elif t.capture_dir and route in ("/v1/chat/completions", "/v1/completions") and status == 200:
        t.n_capture += 1
        sys.path.insert(0, os.path.join(ROOT, "gate"))
        import flerp_serve                                           # noqa: E402  (strixite capture reader)
        tokens = {"sequence_ids": flerp_serve.latest_capture(t.capture_dir, t.n_capture)["generated"]}
    return {"status": status, "headers": rh, "kind": kind, "value": value, "size": size, "sha256": sha, "tokens": tokens}, None


def compare(e, a, b, volatile, header_names):
    rec = {"id": e["id"], "route": f"{e['method']} {e['path'].split('?')[0]}"}
    rec["status"] = {"a": a["status"], "b": b["status"], "equal": a["status"] == b["status"]}
    key = next((k for k in ("token_ids", "sequence_ids") if k in (a["tokens"] or {}) and k in (b["tokens"] or {})), None)
    if key:
        ta, tb = a["tokens"][key], b["tokens"][key]
        rec["tokens"] = {"equal": ta == tb, "compared": key, "a": len(ta), "b": len(tb), "first_difference": first_diff(ta, tb)}
    else:
        rec["tokens"] = {"equal": None, "why": "a server does not report token ids"}
    if e["path"].split("?")[0] == "/cache/export":
        rec["body"] = {"equal": (a["size"], a["sha256"]) == (b["size"], b["sha256"]), "compared": "size and sha256"}
    else:
        va, vb = normalise(a["kind"], a["value"], volatile), normalise(b["kind"], b["value"], volatile)
        eq = a["kind"] == b["kind"] and va == vb
        rec["body"] = {"equal": eq, "compared": a["kind"] if a["kind"] == b["kind"] else f"{a['kind']} / {b['kind']}"}
        if not eq and a["kind"] == b["kind"] and a["kind"] in ("json", "sse"):
            rec["body"]["first_path"] = first_path(va, vb)
    hd = {h: (a["headers"].get(h), b["headers"].get(h)) for h in header_names}
    rec["headers"] = {"equal": all(x == y for x, y in hd.values()), "differ": sorted(h for h, (x, y) in hd.items() if x != y)}
    rec["equal"] = rec["status"]["equal"] and rec["tokens"]["equal"] is not False and rec["body"]["equal"] and rec["headers"]["equal"]
    return rec


# --- the run -----------------------------------------------------------------------------------------------------
def run(a):
    os.makedirs(a.out, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix="flerp-replay-")
    os.chmod(scratch, 0o700)
    volatile = [v for v in a.volatile.split(",") if v]
    corpus = load_corpus(a.corpus, a.target)
    if a.max:
        corpus = corpus[:a.max]
    ta = Target("a", a.a, a.a_kind, a.a_capture_dir, a.a_cmd, a.assume_fresh, a.out)
    tb = Target("b", a.b, a.b_kind, a.b_capture_dir, a.b_cmd, a.assume_fresh, a.out)
    out = {"started": now(), "corpus": os.path.abspath(a.corpus), "target": a.target, "exchanges": len(corpus),
           "volatile": volatile, "headers": a.header, "a": {"url": a.a, "kind": a.a_kind, "cmd": a.a_cmd},
           "b": {"url": a.b, "kind": a.b_kind, "cmd": a.b_cmd}, "results": [], "skipped": []}
    try:
        for t in (ta, tb):
            t.start(a.ready_timeout)
        out["a"]["fresh"], out["b"]["fresh"] = ta.fresh, tb.fresh
        for e in corpus:
            why = skip_reason(e)
            if why is None:
                ra, why = send(ta, e, scratch)
                if why is None:
                    rb, why = send(tb, e, scratch)
            if why is not None:
                out["skipped"].append({"id": e["id"], "route": f"{e['method']} {e['path'].split('?')[0]}", "why": why})
                continue
            out["results"].append(compare(e, ra, rb, volatile, a.header))
    finally:
        for t in (ta, tb):
            t.stop()
        for t in (ta, tb):
            for p in t.exports.values():
                if os.path.exists(p):
                    os.unlink(p)
        os.rmdir(scratch) if not os.listdir(scratch) else None
    res = out["results"]
    out["finished"] = now()
    out["counts"] = {"replayed": len(res), "equal": sum(r["equal"] for r in res), "skipped": len(out["skipped"]),
                     "tokens_compared": sum(r["tokens"]["equal"] is not None for r in res)}
    out["verdict"] = "PASS" if res and all(r["equal"] for r in res) and ta.fresh and tb.fresh else "FAIL"
    SM.write_json(a.out, out, name="results.json")       # no gate TOML: config_sha256 is null
    write_summary(out, os.path.join(a.out, "summary.md"))
    return out


def write_summary(out, path):
    c = out["counts"]
    L = [f"# Replay: {out['a']['url']} ({out['a']['kind']}) against {out['b']['url']} ({out['b']['kind']})", "",
         f"Verdict: **{out['verdict']}**", "",
         f"Corpus `{out['corpus']}`{' target ' + out['target'] if out['target'] else ''}: {out['exchanges']} exchanges, "
         f"{c['replayed']} replayed in recorded order, {c['equal']} equal, {c['skipped']} skipped; token ids compared on "
         f"{c['tokens_compared']}. Volatile fields: {', '.join(out['volatile']) or 'none'}; headers compared: "
         f"{', '.join(out['headers']) or 'none'}. Started {out['started']}, finished {out['finished']}.", "",
         f"Empty-cache start: A {'yes' if out['a'].get('fresh') else 'NOT SHOWN'}, B {'yes' if out['b'].get('fresh') else 'NOT SHOWN'} "
         f"({'started by the gate' if out['a']['cmd'] and out['b']['cmd'] else 'taken as fresh by --assume-fresh, or not'}).", ""]
    diff = [r for r in out["results"] if not r["equal"]]
    if diff:
        L += ["| exchange | route | status | tokens (first difference) | body (first path) | headers |", "|---|---|---|---|---|---|"]
        for r in diff:
            tk = r["tokens"]
            L.append(f"| {r['id']} | {r['route']} | {r['status']['a']} / {r['status']['b']} | "
                     f"{'n/a' if tk['equal'] is None else 'equal' if tk['equal'] else tk['first_difference']} | "
                     f"{'equal' if r['body']['equal'] else r['body'].get('first_path', 'differs')} | {', '.join(r['headers']['differ']) or 'equal'} |")
        L.append("")
    if out["skipped"]:
        why = {}
        for s in out["skipped"]:
            why[s["why"]] = why.get(s["why"], 0) + 1
        L += ["Skipped: " + "; ".join(f"{n} {w}" for w, n in sorted(why.items())), ""]
    SM.write_md(os.path.dirname(path), L, name=os.path.basename(path), newline=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--target", help="the corpus's subdirectory (strixite, pair)")
    ap.add_argument("--a", required=True, help="server A, http://host:port")
    ap.add_argument("--b", required=True, help="server B")
    for x in ("a", "b"):
        ap.add_argument(f"--{x}-kind", choices=("flerp", "origin"), default="origin")
        ap.add_argument(f"--{x}-capture-dir", help="the server's --capture-dir (strixite): token ids")
        ap.add_argument(f"--{x}-cmd", help="a shell command that starts the server fresh; the gate starts and stops it")
    ap.add_argument("--assume-fresh", action="store_true", help="the servers were started fresh by the caller")
    ap.add_argument("--volatile", default=",".join(VOLATILE), help="comma-separated JSON paths dropped before the body "
                    "comparison; dots for depth, * for every element (choices.*.message.tool_calls.*.id)")
    ap.add_argument("--header", action="append", default=None, help="a response header that must be equal (repeat)")
    ap.add_argument("--max", type=int, default=0, help="only the first N exchanges")
    ap.add_argument("--ready-timeout", type=float, default=900)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    a.header = [h.lower() for h in (a.header or ["content-type"])]
    out = run(a)
    print(json.dumps({"verdict": out["verdict"], **out["counts"], "summary": os.path.join(a.out, "summary.md")}))
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
