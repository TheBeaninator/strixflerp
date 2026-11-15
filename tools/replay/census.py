#!/usr/bin/env python3
"""census.py CORPUS_DIR MANIFEST_OUT : field census + sha256 manifest for corpus-v1 (strixite/ and pair/)."""
import collections, glob, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay import classes, primary

root, mout = sys.argv[1], sys.argv[2]
man = {"corpus": "corpus-v1", "root": root, "exchanges": [], "census": {}}
for tgt in ("strixite", "pair"):
    c = collections.Counter(); keys = collections.Counter(); cls = collections.Counter(); prim = collections.Counter()
    ptoks, inflight = [], collections.Counter()
    for f in sorted(glob.glob(os.path.join(root, tgt, "*.json"))):
        raw = open(f, "rb").read(); e = json.loads(raw)
        man["exchanges"].append({"id": e["id"], "file": os.path.relpath(f, root), "sha256": hashlib.sha256(raw).hexdigest(),
                                 "route": f'{e["method"]} {e["path"]}', "status": e.get("status")})
        c[f'route {e["method"]} {e["path"].split("?")[0]} -> {e.get("status")}'] += 1
        if e["path"].startswith("/v1/chat/completions"):
            r = e.get("request") or {}
            for k in r:
                keys[k] += 1
            try:
                rec = json.loads(e.get("response_raw") or "{}")
            except Exception:
                rec = {}
            cl = classes(r, rec)
            for k in cl:
                cls[k] += 1
            prim[primary(cl)] += 1
            if r.get("tool_choice") is not None:
                c[f'tool_choice={json.dumps(r["tool_choice"])}'] += 1
            if r.get("stream"):
                c["stream=true"] += 1
            if r.get("shared_messages") is not None:
                c["shared_messages present"] += 1
            if r.get("response_format"):
                c["response_format"] += 1
            if any(k in r for k in ("logprobs", "top_logprobs")):
                c["logprobs"] += 1
            for m in r.get("messages", []):
                c[f'msg role {m.get("role")}'] += 1
                if isinstance(m.get("content"), list):
                    for p in m["content"]:
                        c[f'part {p.get("type")}'] += 1
            ptoks.append((rec.get("usage") or {}).get("prompt_tokens", 0))
            inflight[e.get("in_flight_at_start", 0) + 1] += 1
    ptoks.sort()
    q = lambda p: ptoks[min(len(ptoks) - 1, int(p * len(ptoks)))] if ptoks else None
    man["census"][tgt] = {"counts": dict(sorted(c.items())), "request_keys": dict(keys), "class_labels": dict(cls),
                          "primary_class": dict(prim), "concurrency_at_start": dict(sorted(inflight.items())),
                          "prompt_tokens": {"n": len(ptoks), "p50": q(.5), "p90": q(.9), "max": ptoks[-1] if ptoks else None}}
json.dump(man, open(mout, "w"), indent=1, sort_keys=True)
print(json.dumps(man["census"], indent=1, sort_keys=True))
