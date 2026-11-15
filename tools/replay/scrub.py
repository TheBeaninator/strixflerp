#!/usr/bin/env python3
"""Scrubber, corpus writer and checker for the FLERP-SWAP replay corpus (docs/REPLAY.md).

  scrub.py --out CORPUS [--manifest PATH] [--corpus NAME] LOG.jsonl ...
        each exchange of the recorder's logs (tools/replay/record_proxy.py), scrubbed, becomes
        CORPUS/<target>/<id>.json; the manifest (default CORPUS/manifest.json) lists every exchange with its sha256
        and the field census, and holds no message text
  scrub.py --manifest-only --out CORPUS [--manifest PATH] [--corpus NAME]
        (re)write the manifest of a corpus directory whose exchange files exist already (any writer of the schema)
  scrub.py --check CORPUS [--manifest PATH]
        exit 1 listing each problem: a directory readable by group or other, a secret pattern left in an exchange,
        a manifest entry whose file is missing or whose sha256 differs, an exchange file the manifest does not list,
        message text in the manifest. The manifest is CORPUS/manifest.json, else gate/replay/<CORPUS name>.manifest.json
        of this tree.

Scrubbing (the rules of the corpus extractor of FLERP-CORPUS, /data/pod/flerp-replay/tools/scrub.py, kept equal):
private keys, GitHub/Slack/AWS keys, API keys, bearer and basic tokens, password/token assignments, e-mail addresses
and IPv4 outside the pod's ranges become a placeholder `<SCRUB:kind>`, so a scrubbed request stays a valid request
and both servers of a replay get the same text. Secret headers never reach the log. A remote image URL (not a
`data:` URL) cannot be replayed: the exchange is kept and marked `replayable: false`. Exchange files hold user
conversations: directories mode 700, files mode 600, never committed; only the manifest goes into git.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATTERNS = [
    ("privkey", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)),
    ("ghtoken", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}")),
    ("slack", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("aws", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("apikey", re.compile(r"\b(?:sk|pk|rk)-(?:ant-|proj-|live-|test-)?[A-Za-z0-9_\-]{20,}")),
    ("bearer", re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=\-]{16,}")),
    ("assign", re.compile(r"(?i)\b(password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)(\\?[\"']?\s*[:=]\s*\\?[\"']?)[^\s\"'\\,;]{6,}")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@(?!users\.noreply\.github\.com)[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("ip", re.compile(r"\b(?!10\.4[24]\.0\.)(?!127\.)(?!0\.0\.0\.0)(?!10\.42\.)(?!192\.168\.)(?:\d{1,3}\.){3}\d{1,3}\b")),
]
ALLOW = ("noreply@anthropic.com",)
TEXT_FIELDS = ("request", "request_raw", "response_raw", "req_headers", "resp_headers")
SECRET_HEADERS = {"authorization", "cookie", "x-api-key", "proxy-authorization", "set-cookie"}


def scrub_text(s):
    hits = 0
    for kind, rx in PATTERNS:
        def rep(m):
            nonlocal hits
            if m.group(0) in ALLOW or "<SCRUB:" in m.group(0):
                return m.group(0)
            hits += 1
            if kind == "assign":
                return m.group(1) + m.group(2) + "<SCRUB:assign>"
            if kind == "bearer":
                return m.group(1) + " <SCRUB:bearer>"
            return f"<SCRUB:{kind}>"
        s = rx.sub(rep, s)
    return s, hits


def walk(o):
    """(o with every string scrubbed, hits). data: URLs are left alone: they are pictures, not text."""
    if isinstance(o, str):
        return (o, 0) if o.startswith("data:") else scrub_text(o)
    if isinstance(o, list):
        r, h = [], 0
        for x in o:
            y, k = walk(x)
            r.append(y)
            h += k
        return r, h
    if isinstance(o, dict):
        r, h = {}, 0
        for kk, v in o.items():
            y, k = walk(v)
            r[kk] = y
            h += k
        return r, h
    return o, 0


def image_urls(req):
    for m in (req or {}).get("messages", []) if isinstance(req, dict) else []:
        if isinstance(m, dict) and isinstance(m.get("content"), list):
            for p in m["content"]:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    u = p.get("image_url")
                    yield u.get("url", "") if isinstance(u, dict) else str(u or "")


def scrub_exchange(e):
    for k in ("req_headers", "resp_headers"):
        if isinstance(e.get(k), dict):
            e[k] = {h: v for h, v in e[k].items() if h.lower() not in SECRET_HEADERS}
    total = 0
    for k in TEXT_FIELDS:
        if k in e:
            e[k], h = walk(e[k])
            total += h
    e["scrubbed"] = total
    remote = [u for u in image_urls(e.get("request")) if not u.startswith("data:")]
    if remote:
        e["replayable"], e["not_replayable"] = False, "remote image URL"
    elif not (e.get("req_complete", True) and e.get("resp_complete", True) and "status" in e):
        e["replayable"], e["not_replayable"] = False, "incomplete exchange"
    else:
        e["replayable"] = True
    return total


# --- the manifest ------------------------------------------------------------------------------------------------
def conversation_of(req):
    """A conversation's name: the sha256 of its first system and first user message (FreeFitty resends the whole
    history each turn, so every turn of a thread starts the same). Text never leaves the hash."""
    msgs = req.get("messages") if isinstance(req, dict) else None
    if not isinstance(msgs, list):
        return None
    head = [m for m in msgs if isinstance(m, dict) and m.get("role") == "system"][:1] + \
           [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"][:1]
    return hashlib.sha256(json.dumps(head, sort_keys=True).encode()).hexdigest()[:16]


def features(e):
    """The feature labels of one exchange, for the census (no text)."""
    req, route = e.get("request"), e.get("path", "").split("?")[0]
    f = [f"route {e.get('method')} {route} -> {e.get('status')}"]
    if route in ("/cache/export", "/cache/import", "/cache/probe", "/cache"):
        f.append("cache route")
    if not isinstance(req, dict):
        return f
    if req.get("tools"):
        f.append("tools")
    if "tool_choice" in req:
        f.append(f"tool_choice={json.dumps(req['tool_choice'], sort_keys=True)[:40]}")
    for k in ("steer", "steer_scope", "shared_messages", "response_format", "stream", "seed", "temperature", "logprobs"):
        if req.get(k) not in (None, False):
            f.append(k)
    msgs = req.get("messages") if isinstance(req.get("messages"), list) else []
    if any(True for _ in image_urls(req)):
        f.append("images")
    if any(isinstance(m, dict) and m.get("role") == "tool" for m in msgs):
        f.append("tool messages")
    if any(isinstance(m, dict) and m.get("tool_calls") for m in msgs):
        f.append("assistant tool_calls")
    f.append(f"concurrency {e.get('in_flight_at_start', 0) + 1}")
    return f


def entry(e, rel, sha):
    req = e.get("request")
    try:
        resp = json.loads(e.get("response_raw") or "null")
    except ValueError:
        resp = None
    usage = resp.get("usage") if isinstance(resp, dict) and isinstance(resp.get("usage"), dict) else {}
    keys = sorted(req) if isinstance(req, dict) else []
    return {"id": e["id"], "file": rel, "sha256": sha, "target": e.get("target"), "route": f"{e.get('method')} {e.get('path', '').split('?')[0]}",
            "status": e.get("status"), "t_start": e.get("t_start"), "t_end": e.get("t_end"), "in_flight_at_start": e.get("in_flight_at_start", 0),
            "conversation": conversation_of(req), "request_keys": keys,
            "n_messages": len(req.get("messages", [])) if isinstance(req, dict) and isinstance(req.get("messages"), list) else None,
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") if isinstance(usage.get("prompt_tokens_details"), dict) else None,
            "req_body_sha256": e.get("req_body_sha256"), "resp_body_sha256": e.get("resp_body_sha256"),
            "replayable": e.get("replayable", True), "not_replayable": e.get("not_replayable"), "scrubbed": e.get("scrubbed", 0)}


def exchange_files(corpus):
    return sorted(p for p in glob.glob(os.path.join(corpus, "**", "*.json"), recursive=True)
                  if os.path.basename(p) != "manifest.json" and not os.path.relpath(p, corpus).startswith("synthetic" + os.sep))


def write_manifest(corpus, path, name):
    entries, census = [], collections.defaultdict(collections.Counter)
    for p in exchange_files(corpus):
        with open(p, "rb") as f:
            raw = f.read()
        e = json.loads(raw)
        if "id" not in e:
            continue
        rel = os.path.relpath(p, corpus)
        entries.append(entry(e, rel, hashlib.sha256(raw).hexdigest()))
        tgt = e.get("target") or rel.split(os.sep)[0]
        census[tgt]["exchanges"] += 1
        for k in features(e):
            census[tgt][k] += 1
        if isinstance(e.get("request"), dict):
            for k in e["request"]:
                census[tgt][f"key {k}"] += 1
    entries.sort(key=lambda x: (x["t_start"] or 0, x["id"]))
    for i, x in enumerate(entries):
        x["order"] = i
    man = {"corpus": name, "schema": "flerp-replay-exchange/1", "exchanges": entries,
           "census": {t: dict(sorted(c.items())) for t, c in sorted(census.items())},
           "conversations": len({x["conversation"] for x in entries if x["conversation"]})}
    with open(path, "w") as f:
        json.dump(man, f, indent=1, sort_keys=True)
        f.write("\n")
    return man


def write_corpus(logs, corpus):
    os.makedirs(corpus, mode=0o700, exist_ok=True)
    os.chmod(corpus, 0o700)
    n = collections.Counter()
    for log in logs:
        with open(log, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                scrub_exchange(e)
                d = os.path.join(corpus, e.get("target") or "node")
                os.makedirs(d, mode=0o700, exist_ok=True)
                os.chmod(d, 0o700)
                p = os.path.join(d, f"{e['id']}.json")
                fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as out:
                    json.dump(e, out, ensure_ascii=False, sort_keys=True)
                n[e.get("target") or "node"] += 1
    return dict(n)


# --- the check ---------------------------------------------------------------------------------------------------
def message_texts(e):
    """Strings of an exchange that are conversation text (long enough to be recognisable)."""
    out = []
    def coll(o):
        if isinstance(o, str):
            if len(o) >= 24 and not o.startswith("data:"):
                out.append(o[:200])
        elif isinstance(o, list):
            for x in o:
                coll(x)
        elif isinstance(o, dict):
            for v in o.values():
                coll(v)
    req = e.get("request")
    if isinstance(req, dict):
        coll(req.get("messages"))
        coll(req.get("tools"))
    try:
        resp = json.loads(e.get("response_raw") or "null")
    except ValueError:
        resp = None
    if isinstance(resp, dict):
        coll(resp.get("choices"))
    return out


def default_manifest(corpus):
    p = os.path.join(corpus, "manifest.json")
    return p if os.path.isfile(p) else os.path.join(ROOT, "gate", "replay", os.path.basename(os.path.normpath(corpus)) + ".manifest.json")


def check(corpus, manifest=None):
    bad = []
    for dp, _, _ in os.walk(corpus):
        m = os.stat(dp).st_mode
        if m & 0o077:
            bad.append(f"directory mode {oct(m & 0o777)}: {dp}")
    files = exchange_files(corpus)
    exchanges = {}
    for p in files:
        with open(p, "rb") as f:
            raw = f.read()
        try:
            e = json.loads(raw)
        except ValueError:
            bad.append(f"not JSON: {p}")
            continue
        exchanges[os.path.relpath(p, corpus)] = (e, hashlib.sha256(raw).hexdigest())
        for k in TEXT_FIELDS:
            if k in e:
                _, h = walk(e[k])
                if h:
                    bad.append(f"{h} secret-pattern hit(s) in {os.path.relpath(p, corpus)}:{k}")
        for k in ("req_headers", "resp_headers"):
            if isinstance(e.get(k), dict) and SECRET_HEADERS & {h.lower() for h in e[k]}:
                bad.append(f"secret header in {os.path.relpath(p, corpus)}:{k}")
    manifest = manifest or default_manifest(corpus)
    if not os.path.isfile(manifest):
        bad.append(f"no manifest: {manifest}")
    else:
        with open(manifest) as f:
            mtext = f.read()
        man = json.loads(mtext)
        listed = {x["file"]: x for x in man.get("exchanges", [])}
        for rel, x in listed.items():
            if rel not in exchanges:
                bad.append(f"manifest lists a missing file: {rel}")
            elif exchanges[rel][1] != x.get("sha256"):
                bad.append(f"sha256 differs from the manifest: {rel}")
        for rel in exchanges:
            if rel not in listed:
                bad.append(f"not in the manifest: {rel}")
        leaks = sum(1 for e, _ in exchanges.values() for t in message_texts(e) if json.dumps(t)[1:-1] in mtext or t in mtext)
        if leaks:
            bad.append(f"{leaks} message text(s) found in the manifest {manifest}")
    for b in bad:
        print("FAIL", b)
    print(f"scrub check: {'FAIL' if bad else 'PASS'} {corpus} ({len(exchanges)} exchanges, manifest {manifest})")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", metavar="CORPUS")
    ap.add_argument("--out", metavar="CORPUS")
    ap.add_argument("--manifest")
    ap.add_argument("--corpus", help="the corpus's name in the manifest (default: the directory's name)")
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("logs", nargs="*")
    a = ap.parse_args(argv)
    if a.check:
        return check(a.check, a.manifest)
    if not a.out or (not a.logs and not a.manifest_only):
        ap.error("--out CORPUS with LOG.jsonl files, --out CORPUS --manifest-only, or --check CORPUS")
    if a.logs:
        print(json.dumps({"written": write_corpus(a.logs, a.out)}))
    man = write_manifest(a.out, a.manifest or os.path.join(a.out, "manifest.json"), a.corpus or os.path.basename(os.path.normpath(a.out)))
    print(json.dumps({"manifest": a.manifest or os.path.join(a.out, "manifest.json"), "exchanges": len(man["exchanges"]),
                      "conversations": man["conversations"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
