#!/usr/bin/env python3
"""repair.py CORPUS : re-pair requests and responses by time within each TCP connection.

extract.py paired the k-th request with the k-th parsed response on a connection. When a response on a keep-alive
connection failed to parse (a capped bulk body, a gap), every later pair on that connection shifted. HTTP/1.1
without pipelining puts each response after its own request and before the next request, so a response r belongs to
request i when t_req_end(i) <= t_end(r) and no other request on the connection started in between. Exchanges left
without such a response are removed. Then in_flight_at_start is recomputed.
"""
import collections, glob, json, os, sys

root = sys.argv[1]
files = sorted(glob.glob(os.path.join(root, "*", "*.json")))
ex = [(f, json.load(open(f))) for f in files]
by_conn = collections.defaultdict(list)
for f, e in ex:
    by_conn[(e["server"], e["client"])].append((f, e))
keep, drop, moved = [], [], 0
for conn, lst in by_conn.items():
    lst.sort(key=lambda x: x[1]["t_start"])
    resp = [(e["t_end"], {k: e[k] for k in e if k in ("status", "t_end", "resp_headers", "resp_complete", "response_raw",
                                                     "resp_body_bytes", "resp_body_sha256")}, f) for f, e in lst if "t_end" in e]
    resp.sort(key=lambda r: r[0])
    starts = [e["t_start"] for _, e in lst]
    for i, (f, e) in enumerate(lst):
        lo = e.get("t_req_end") or e["t_start"]
        hi = starts[i + 1] if i + 1 < len(starts) else float("inf")
        cand = [r for r in resp if lo <= r[0] <= hi]
        if len(cand) != 1:
            drop.append(f); continue
        _, r, src = cand[0]
        if src != f:
            moved += 1
        for k in ("status", "t_end", "resp_headers", "resp_complete", "response_raw", "resp_body_bytes", "resp_body_sha256"):
            e.pop(k, None)
        e.update(r)
        keep.append((f, e))
for f in drop:
    os.remove(f)
for f, e in keep:
    e["in_flight_at_start"] = sum(1 for _, o in keep if o is not e and o["server"] == e["server"]
                                  and o["t_start"] <= e["t_start"] < o.get("t_end", 1e18) and o["path"].startswith("/v1/"))
    e["repaired"] = "re-paired by time (tools/replay/repair.py)"
    json.dump(e, open(f, "w"), ensure_ascii=False, sort_keys=True)
print(json.dumps({"kept": len(keep), "dropped": len(drop), "response_moved": moved}))
