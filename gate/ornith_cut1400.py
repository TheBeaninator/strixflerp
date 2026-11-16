#!/usr/bin/env python3
"""ORNITH-CUT1400 (card 72e1417340): is arm Z wrong at the cut1400 prefill end, or is the model on a knife edge there?
Criteria: gate/ornith-cut1400.toml (committed before any run). Runs ON control-node as one tools/gpu-queue.sh job.

    FLERP_GEN_BIN=<flerp-qwen35moe-generate built from card/5083008fe9> python3 gate/ornith_cut1400.py run --out DIR
        one load of the 35B; prompts = the records prefixes ending right before a record number's second digit
        (1,000 <= cut < 1,808; cut1400 among them); arms D, X, Z, G, teacher-forced on records[c:c+4]; logits top
        1,000 + logsumexp at positions 0-3 (--trace --record-at 0,1,2,3)
    python3 gate/ornith_cut1400.py judge DIR          C1-C3 and the per-cut table into DIR/verdict.json
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import prefill_ornith_35b as po  # noqa: E402  (MODEL, RECORDS, ROCM, split_arms, kl)

ARMS = "D,X,Z,G"
RECORD, SPACE, COLON = "6269", "220", "25"
DIGITS = {str(i) for i in range(15, 25)}       # "0".."9"
LO, HI = 1000, 1808


def cuts(rec):
    """c such that records[:c] ends "...Record <d>" and records[c] is the number's second digit (then ":")."""
    out = []
    for c in range(LO, min(HI, len(rec) - 4)):
        if rec[c - 3] == RECORD and rec[c - 2] == SPACE and rec[c - 1] in DIGITS and rec[c] in DIGITS and rec[c + 1] == COLON:
            out.append(c)
    return out


def binary():
    b = os.environ.get("FLERP_GEN_BIN") or po.binary()
    if not os.path.exists(b):
        raise SystemExit("no generator at " + b)
    return b


def cmd_run(a):
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    rec = open(po.RECORDS).read().split()
    cs = [int(x) for x in a.cuts.split(",")] if a.cuts else cuts(rec)
    arms = a.arms or ARMS
    if 1400 not in cs:
        raise SystemExit("cut1400 is not a record-digit position in %s: %s" % (po.RECORDS, cs))
    args = []
    for c in cs:
        p = os.path.join(out, "cut%d.ids" % c)
        f = os.path.join(out, "force%d.ids" % c)
        open(p, "w").write(" ".join(rec[:c]) + "\n")
        open(f, "w").write(" ".join(rec[c:c + 4]) + "\n")
        args += ["--ids-file", p]
    for c in cs:
        args += ["--force-file", os.path.join(out, "force%d.ids" % c)]
    exe = binary()
    cmd = [exe, "--model", po.MODEL, "--context", "4096"] + args + ["--trace", "--trace-top", "1000", "--record-at", "0,1,2,3",
                                                                   "--max-tokens", "4", "--arms", arms]
    env = dict(os.environ, LD_LIBRARY_PATH=po.ROCM + "/lib:" + po.ROCM + "/lib/llvm/lib")
    t0 = time.time()
    with open(os.path.join(out, "run.err"), "w") as ef:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=ef, text=True, env=env, timeout=3000)
    open(os.path.join(out, "run.jsonl"), "w").write(p.stdout)
    fallbacks = sum("re-run on the per-token path" in ln for ln in open(os.path.join(out, "run.err"), errors="replace"))
    r = {"rc": p.returncode, "wall_s": round(time.time() - t0, 1), "arms": arms, "cuts": cs, "binary": exe,
         "binary_sha256": hashlib.sha256(open(exe, "rb").read()).hexdigest(), "fallbacks": fallbacks, "cmd": cmd}
    json.dump(r, open(os.path.join(out, "run.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in r.items() if k != "cmd"}))
    return 0 if p.returncode == 0 else 1


def top8(step):
    return [[t, round(v, 4)] for t, v in step["top"][:8]]


def gap(step):
    return step["top"][0][1] - step["top"][1][1]


def cmd_judge(a):
    d = a.dir
    run = json.load(open(os.path.join(d, "run.json")))
    ev = [json.loads(l) for l in open(os.path.join(d, "run.jsonl")) if l.startswith("{")]
    arms = {arm: {e["prompt"]: e for e in es if e.get("event") == "traced"} for arm, es in po.split_arms(ev)}
    cs = run["cuts"]
    want = set(ARMS.split(","))
    c1_bad = []
    if run["rc"] != 0:
        c1_bad.append("rc %d" % run["rc"])
    if run["fallbacks"]:
        c1_bad.append("fallbacks %d" % run["fallbacks"])
    for arm in want:
        for i in range(len(cs)):
            e = arms.get(arm, {}).get(i)
            if e is None or len(e["steps"]) < 4:
                c1_bad.append("%s missing prompt %d" % (arm, i))
    if "Z" in arms and "G" in arms:
        for i in range(len(cs)):
            for s, (z, g) in enumerate(zip(arms["Z"].get(i, {}).get("steps", []), arms["G"].get(i, {}).get("steps", []))):
                if not z.get("sha256") or z.get("sha256") != g.get("sha256"):
                    c1_bad.append("Z != G sha at cut%d step %d" % (cs[i], s))
    table = []
    for i, c in enumerate(cs):
        D = arms["D"][i]["steps"]
        row = {"cut": c, "true_next": int(open(os.path.join(d, "force%d.ids" % c)).read().split()[0]),
               "D_margin": round(gap(D[0]), 4), "arms": {}}
        for arm in ARMS.split(","):
            S = arms[arm][i]["steps"]
            row["arms"][arm] = {"top8": top8(S[0]), "gap": round(gap(S[0]), 4), "top1": S[0]["top"][0][0],
                                "agrees_D": S[0]["top"][0][0] == D[0]["top"][0][0],
                                "kl_D": [round(po.kl(sd, sx)[0], 6) for sd, sx in zip(D, S)],
                                "first_token_s": arms[arm][i].get("first_token_s")}
        table.append(row)
    r14 = next(r for r in table if r["cut"] == 1400)
    z, x = r14["arms"]["Z"], r14["arms"]["X"]
    defect_top1 = (not z["agrees_D"]) and x["agrees_D"] and r14["D_margin"] >= 0.3
    defect_kl = z["kl_D"][0] > 5 * x["kl_D"][0] and z["kl_D"][0] > 0.02
    c2 = {"defect": defect_top1 or defect_kl, "defect_top1": defect_top1, "defect_kl": defect_kl, "Z_top1": z["top1"], "X_top1": x["top1"],
          "D_top1": r14["arms"]["D"]["top1"], "true_next": r14["true_next"], "D_margin": r14["D_margin"],
          "kl_D_Z": z["kl_D"][0], "kl_D_X": x["kl_D"][0], "Z_gap": z["gap"], "X_gap": x["gap"]}
    if not c2["defect"]:
        c2["why"] = ("Z top-1 equals D's" if z["agrees_D"] else "D margin < 0.3, a tie for FP32 too" if r14["D_margin"] < 0.3
                     else "X misses D's top-1 as well (shared prefill-activation precision property)")
    pattern = [r["cut"] for r in table if not r["arms"]["Z"]["agrees_D"] and r["arms"]["X"]["agrees_D"] and r["D_margin"] >= 0.3]
    fam = {arm: {"top1_agree": sum(r["arms"][arm]["agrees_D"] for r in table), "cuts": len(table),
                 "mean_kl0": sum(r["arms"][arm]["kl_D"][0] for r in table) / len(table),
                 "max_kl": max(k for r in table for k in r["arms"][arm]["kl_D"])} for arm in ("X", "Z", "G")}
    c3 = {"defect_pattern": len(pattern) >= 2, "z_misses_x_hits_margin": pattern, "family": fam}
    res = {"C1": {"pass": not c1_bad, "problems": c1_bad}, "C2": c2, "C3": c3,
           "verdict": "DEFECT" if (c2["defect"] or c3["defect_pattern"]) else "NOT A DEFECT",
           "binary_sha256": run["binary_sha256"], "wall_s": run["wall_s"], "table": table}
    json.dump(res, open(os.path.join(d, "verdict.json"), "w"), indent=1)
    print("C1", res["C1"]["pass"], c1_bad)
    print("C2", json.dumps(c2))
    print("C3", json.dumps({k: v for k, v in c3.items() if k != "family"}), json.dumps(fam))
    for r in table:
        print("cut%d true %d D-margin %.3f" % (r["cut"], r["true_next"], r["D_margin"]),
              " ".join("%s:%d/%.2f/kl%.4f" % (arm, v["top1"], v["gap"], v["kl_D"][0]) for arm, v in r["arms"].items()))
    print("verdict", res["verdict"])


def cmd_diff(a):
    """Per-kernel localisation (the card's 'find it'): KL(D || arm) at position 0 per cut for every arm of a run made with
    --arms D,B,E,F,X,G,Y, and the pair deltas that isolate one kernel class each: B-E experts F16 vs exact codes, F-B dense
    F16 grouped vs per-token, X-E dense exact grouped vs per-token, G-F and Y-X GQA WMMA attention vs the origin chain."""
    d = a.dir
    run = json.load(open(os.path.join(d, "run.json")))
    ev = [json.loads(l) for l in open(os.path.join(d, "run.jsonl")) if l.startswith("{")]
    arms = {arm: {e["prompt"]: e for e in es if e.get("event") == "traced"} for arm, es in po.split_arms(ev)}
    cs = run["cuts"]
    pairs = [("B", "E", "experts F16 vs exact codes"), ("F", "B", "dense F16 grouped vs per-token"), ("X", "E", "dense exact grouped vs per-token"),
             ("G", "F", "GQA WMMA attention (F16 experts+dense)"), ("Y", "X", "GQA WMMA attention (exact experts+dense)")]
    out = {"cuts": {}, "pairs": {}}
    for i, c in enumerate(cs):
        D = arms["D"][i]["steps"][0]
        row = {arm: {"kl_D": round(po.kl(D, arms[arm][i]["steps"][0])[0], 6), "top1": arms[arm][i]["steps"][0]["top"][0][0],
                     "gap": round(gap(arms[arm][i]["steps"][0]), 4)} for arm in arms if arm != "D" and i in arms[arm]}
        row["D"] = {"top1": D["top"][0][0], "gap": round(gap(D), 4), "top8": top8(D)}
        out["cuts"][c] = row
        print("cut%d D:%d/%.2f " % (c, D["top"][0][0], gap(D)) + " ".join("%s:%d/%.2f/kl%.4f" % (k, v["top1"], v["gap"], v["kl_D"]) for k, v in row.items() if k != "D"))
    for x, y, what in pairs:
        if x in arms and y in arms:
            sh = {c: round((out["cuts"][c][x]["kl_D"] - out["cuts"][c][y]["kl_D"]) / max(out["cuts"][c].get("G", out["cuts"][c][x])["kl_D"], 1e-9), 3) for c in cs}
            out["pairs"]["%s-%s" % (x, y)] = {"what": what, "delta_kl": {c: round(out["cuts"][c][x]["kl_D"] - out["cuts"][c][y]["kl_D"], 6) for c in cs}, "share_of_G": sh}
            print("%s-%s (%s): delta KL per cut %s; share of KL(D,G) %s" % (x, y, what, out["pairs"]["%s-%s" % (x, y)]["delta_kl"], sh))
    json.dump(out, open(os.path.join(d, "diff.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--arms", default=None, help="default D,X,Z,G; the per-kernel diff uses D,B,E,F,X,G,Y")
    r.add_argument("--cuts", default=None, help="comma list; default every record-digit cut 1,000-1,808")
    j = sub.add_parser("judge")
    j.add_argument("dir")
    f = sub.add_parser("diff")
    f.add_argument("dir")
    a = ap.parse_args()
    sys.exit({"run": cmd_run, "judge": cmd_judge, "diff": cmd_diff}[a.cmd](a) or 0)


if __name__ == "__main__":
    main()
