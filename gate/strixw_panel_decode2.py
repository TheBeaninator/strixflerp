#!/usr/bin/env python3
"""decode2 of the strixw panel kernels (criteria: gate/strixw-panel-decode2.toml, committed before any GPU run).

Stages (on control-node inside /data/projects/strixflerp; GPU jobs through tools/gpu-queue.sh):
  sweep    reads gate/results/strixw-panel-decode2/sweep/*.jsonl (tools/decode2-bench.sh, one wave map a file) and
           writes sweep.md: panel % of origin time per class, bits, M and map
  full     the committed kernel gate's bench (gate/strixw-panel-kernels.toml [run], every class, path and M) with the
           chosen map, raw lines to gate/results/strixw-panel-decode2/full/raw.jsonl
  report   the committed gate's report on full/ (its own criteria, README.md / summary.json there) and the decode2
           criteria -> gate/results/strixw-panel-decode2/summary.json
"""
import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import strixw_panel_kernels as base  # noqa: E402

RES = os.path.join(ROOT, "gate", "results", "strixw-panel-decode2")
FULL = os.path.join(RES, "full")


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def rows_of(path):
    out = []
    for line in open(path):
        d = json.loads(line)
        if not d.get("summary") and "skipped" not in d:
            out.append(d)
    return out


def slower(d):  # slower outside noise by more than 3%
    return d["panel_ms"]["q1"] > d["origin_ms"]["q3"] and d["panel_pct_of_origin_time"] > 103.0


def sweep():
    files = sorted(glob.glob(os.path.join(RES, "sweep", "*.jsonl")))
    L = ["# decode2 sweep: panel % of origin time (median of the reps), per wave map", ""]
    for f in files:
        name = os.path.basename(f)[:-6]
        load = open(f[:-6] + ".load").read().strip().replace("\n", "; ") if os.path.exists(f[:-6] + ".load") else "?"
        rows = [d for d in rows_of(f) if "decode" in d["path"] and not d.get("bad_id")]
        L += [f"## {name}", "", f"load: {load}; {sum(d['equal'] for d in rows)} / {len(rows)} byte-equal", "",
              "| class | bits | path | out | M=1..8 panel % | median |", "|---|---|---|---|---|---|"]
        grp = {}
        for d in rows:
            grp.setdefault((d["class"], d["bits"], d["path"], d.get("out", "bf16")), []).append(d)
        for k, v in sorted(grp.items()):
            v.sort(key=lambda d: d["M"])
            L.append(f"| {k[0]} | Q{k[1]} | {k[2]} | {k[3]} | " +
                     " ".join(f"{d['panel_pct_of_origin_time']:.0f}{'*' if slower(d) else ''}" for d in v) +
                     f" | {med([d['panel_pct_of_origin_time'] for d in v]):.1f} |")
        L.append("")
    L.append("`*`: slower outside noise by more than 3% (panel q1 > origin q3 and > 103%).")
    open(os.path.join(RES, "sweep.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def full():
    cfg = base.load()
    h, r = cfg["host"], cfg["run"]
    base.wait_load(h["max_load"])
    os.makedirs(FULL, exist_ok=True)
    cmd = base.NICE + [os.path.join(base.ninja_dir(cfg), h["bench"]), "--weights", h["weights"], "--out",
                       os.path.join(FULL, "raw.jsonl"), "--decode-m", ",".join(map(str, r["decode_m"])), "--prefill-m",
                       ",".join(map(str, r["prefill_m"])), "--active", str(r["active"]), "--reps", str(r["reps"]),
                       "--warm", str(r["warm"])]
    for c in r["classes"]:
        cmd += ["--class", c]
    env = dict(os.environ)
    env.pop("FLERP_PANEL_TPW", None)  # the dispatch table
    t0 = time.time()
    with open(os.path.join(FULL, "bench.log"), "w") as log:
        log.write(f"load_before {open('/proc/loadavg').read().strip()}\n+ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        log.flush()
        rc = subprocess.call(["/usr/bin/time", "-v"] + cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        log.write(f"load_after {open('/proc/loadavg').read().strip()}\n")
    print(f"bench rc={rc} in {time.time() - t0:.0f} s", flush=True)
    return rc


def report():
    base.RES = FULL
    base.report(base.load())  # the committed criteria on the rerun: full/README.md, full/summary.json
    rows = rows_of(os.path.join(FULL, "raw.jsonl"))
    summ = json.loads([line for line in open(os.path.join(FULL, "raw.jsonl")) if '"summary"' in line][-1])
    dec = [d for d in rows if "decode" in d["path"] and not d.get("bad_id")]
    v = []
    unequal = [d for d in rows if not d["equal"]]
    v.append({"criterion": "equal", "pass": not unequal and len(rows) > 0,
              "detail": f"{len(rows) - len(unequal)} of {len(rows)} cases byte-equal (bad ids included)"})
    dense, exp = {}, {}
    for d in dec:
        (dense if d["path"] == "dense-decode" else exp).setdefault(d["class"] if d["path"] == "dense-decode" else d["path"], []).append(d)

    def crit(name, groups):
        meds = {k: med([d["panel_pct_of_origin_time"] for d in g]) for k, g in groups.items()}
        bad = [f"{d['class']} Q{d['bits']} {d['path']} M{d['M']} {d.get('out', 'bf16')} {d['panel_pct_of_origin_time']:.1f}%"
               for g in groups.values() for d in g if slower(d)]
        v.append({"criterion": name, "pass": bool(meds) and all(m <= 100.0 for m in meds.values()) and not bad,
                  "detail": "median panel % per group: " + ", ".join(f"{k} {m:.1f}" for k, m in sorted(meds.items())) +
                            "; slower outside noise by > 3%: " + ("; ".join(bad) or "none")})
    crit("decode_dense", dense)
    crit("decode_expert", exp)
    committed = json.load(open(os.path.join(FULL, "summary.json")))["verdicts"]
    pre = [c for c in committed if c["criterion"] in ("speed_prefill_dense", "speed_prefill_expert")]
    v.append({"criterion": "prefill", "pass": len(pre) == 2 and all(c["pass"] for c in pre),
              "detail": "; ".join(f"{c['criterion']} {'PASS' if c['pass'] else 'FAIL'}" for c in pre)})
    gb = summ.get("bytes_read", 0) / 1e9
    v.append({"criterion": "read", "pass": gb <= 2.5, "detail": f"{gb:.3f} GB read"})
    load = [line for line in open(os.path.join(FULL, "bench.log")) if line.startswith("load_")]
    out = {"criteria_file": "gate/strixw-panel-decode2.toml", "cases": len(rows), "verdicts": v, "load": load}
    json.dump(out, open(os.path.join(RES, "summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sweep", "full", "report"], action="append", default=[])
    a = ap.parse_args()
    rc = 0
    for s in a.stage:
        if s == "sweep":
            sweep()
        elif s == "full":
            rc = full()
        else:
            report()
    sys.exit(rc)


if __name__ == "__main__":
    main()
