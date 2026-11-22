#!/usr/bin/env python3
"""Markdown verdict tables for docs/experiments/coll-model.md from the sweeps' JSON lines (COLL-MODEL).

    python3 tools/coll-model/report_tables.py <dir with s3.json mut.jsonl w2.jsonl w2deep.jsonl w4.jsonl w4t1.jsonl porval*.jsonl>
"""
import glob
import json
import sys


def load(p):
    out = []
    for f in sorted(glob.glob(p)):
        out += [json.loads(l) for l in open(f)]
    return out


def verdict(r, props):
    if r is None:
        return "not run"
    v = r.get("violation")
    if v and v["prop"] in props:
        return "**VIOLATED**"
    if v:
        return "holds until the %s stop" % v["prop"]
    return "holds" if r["exhausted"] else "not exhausted (%d states)" % r["states"]


def st(r):
    return "-" if r is None else ("%d%s" % (r["states"], "" if r["exhausted"] else "+"))


def main():
    d = sys.argv[1]
    w2 = load(d + "/w2.jsonl")
    deep = load(d + "/w2deep.jsonl")
    w4 = load(d + "/w4.jsonl")
    w4t1 = load(d + "/w4t1.jsonl")
    mut = load(d + "/mut.jsonl")
    pv = load(d + "/porval*.jsonl")
    s3 = json.load(open(d + "/s3.json"))
    s3v = "holds" if not s3["problems"] else "**VIOLATED**"
    idx = {(r["transport"], r["stripe"], r["wire"], r["nslot"], r["mode"]): r for r in w2}
    didx = {(r["transport"], r["stripe"], r["nslot"]): r for r in deep}
    print("### World 2 (tcp_proxy; rdma_proxy RdmaProxy)\n")
    print("S1 and S2 are checked on every step of both runs; S3 is the static check (%d write ranges, no problem); states: D1 run /" % s3["checked"])
    print("T1 run (3 calls) / T1 with NSLOT + 2 calls (wire threads 2 only; + = budget reached, not exhausted).\n")
    print("| transport | proxy threads | wire threads | NSLOT | S1 | S2 | S3 | D1 | T1 | states D1 / T1 / T1 deep |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for tp in ("tcp", "rdma"):
        for s in (0, 1):
            for wi in (1, 2):
                for n in range(2, 9):
                    a = idx.get((tp, s, wi, n, "D1"))
                    b = idx.get((tp, s, wi, n, "T1"))
                    dp = didx.get((tp, s, n)) if wi == 2 else None
                    s1 = verdict(a, ("S1", "S1-TX"))
                    for x in (b, dp):
                        if x and x.get("violation") and x["violation"]["prop"] in ("S1", "S1-TX"):
                            s1 = "**VIOLATED** (T1 run)"
                    thr = "2 (hypothetical)" if (tp == "rdma" and s) else str(1 + s)
                    s3c = s3v if tp == "rdma" else "n/a"
                    t1 = verdict(b, ("T1",))
                    if dp and dp.get("violation"):
                        t1 = "**VIOLATED** (deep run)"
                    print("| %s | %s | %d | %d | %s | %s | %s | %s | %s | %s / %s / %s |" % (tp, thr, wi, n, s1, verdict(a, ("S2",)), s3c, verdict(a, ("D1",)), t1, st(a), st(b), st(dp)))
    print("\n### World 4 (rdma_proxy RdmaProxyN over CoreN: the deployed TP4 transport)\n")
    print("Proxy threads 2: refused (rdma_proxy.zig:509-512). TCP at world 4: refused (comm.zig:24). Wire threads: RdmaProxyN never")
    print("opens tcp_host (openN, rdma_proxy.zig:491-579), so TF_COLL_WIRE_THREADS has no code path and every row holds for 1 and 2.")
    print("Shapes: decode = 2m+1 calls of {a, s, x, z} (k = 1, lane 0); prompt = 2m+1 calls of {c, y} (every lane, every lane's")
    print("slots reused); prompt1 = one gather of lanes + 1 chunks. m = slots a lane. States: decode / prompt / prompt1.\n")
    print("| NSLOT | lanes (m) | S1 | S2 | S3 | D1 decode | D1 prompt | D1 prompt1 | states |")
    print("|---|---|---|---|---|---|---|---|---|")
    idx4 = {(r["nslot"], r["lanes"], r.get("shape")): r for r in w4}
    for n in range(2, 9):
        for L in (1, 2, 3, 4):
            if n % L or n // L < 2:
                continue
            rs = [idx4.get((n, L, sh)) for sh in ("decode", "prompt", "prompt1")]
            def s(props):
                vs = [verdict(x, props) for x in rs]
                if any("VIOLATED" in x for x in vs):
                    return "**VIOLATED**"
                if any(("not exhausted" in x) or x == "not run" for x in vs):
                    return "holds where exhausted"
                return "holds"
            dep = " (deployed)" if (n, L) == (8, 4) else ""
            print("| %d%s | %d (%d) | %s | %s | %s | %s | %s | %s | %s / %s / %s |" % (n, dep, L, n // L, s(("S1", "S1-TX")), s(("S2",)), s3v,
                  verdict(rs[0], ("D1",)), verdict(rs[1], ("D1",)), verdict(rs[2], ("D1",)), st(rs[0]), st(rs[1]), st(rs[2])))
    print("\n### T1 at world 4\n")
    print("One slot gather (a) or one pair exchange (x) at 8 slots / 4 lanes; with one call the model is the same for every NSLOT and")
    print("lanes setting with 2+ slots a lane. Faults in every visited state; before the fault the reduced steps of the D1 search.\n")
    print("| call | T1 | S1 / S2 under faults | states | seconds |")
    print("|---|---|---|---|---|")
    for r in w4t1:
        v = r.get("violation")
        print("| %s | %s | %s | %s | %s |" % (r["kinds"], verdict(r, ("T1",)), "**VIOLATED**" if v and v["prop"] in ("S1", "S1-TX", "S2") else ("holds" if r["exhausted"] else "none seen"), st(r), r["seconds"]))
    print("\n### The reduction against weaker ones\n")
    print("| configuration | full | basic | none |")
    print("|---|---|---|---|")
    groups = {}
    for r in pv:
        key = json.dumps({k: r[k] for k in ("world", "transport", "nslot", "lanes", "stripe", "mode", "calls", "kinds") if k in r}, sort_keys=True)
        groups.setdefault(key, {})[r.get("level", "full")] = r
    for key, g in groups.items():
        c = json.loads(key)
        name = ("w2 %s stripe=%d NSLOT=%d %s %d calls" % (c["transport"], c["stripe"], c["nslot"], c["mode"], c["calls"])) if c["world"] == 2 else ("w4 NSLOT=%d lanes=%d %s %d calls of %s" % (c["nslot"], c["lanes"], c["mode"], c["calls"], c["kinds"]))
        cell = lambda r: "-" if r is None else ("%s, %s" % (st(r), "VIOLATION " + r["violation"]["prop"] if r.get("violation") else "clean"))
        print("| %s | %s | %s | %s |" % (name, cell(g.get("full")), cell(g.get("basic")), cell(g.get("none"))))
    print("\n### Controls (mutations and refused settings: the checker must find these)\n")
    print("| control | what it changes | expected | found | states |")
    print("|---|---|---|---|---|")
    for r in mut:
        v = r.get("violation")
        print("| %s | %s | %s | %s | %d |" % (r["name"], r["what"], r["expected"], ("%s: %s" % (v["prop"], v["msg"])) if v else ("none (exhausted)" if r["exhausted"] else "none (not exhausted)"), r["states"]))


if __name__ == "__main__":
    main()
