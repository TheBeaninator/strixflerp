#!/usr/bin/env python3
"""Model gate for a prefill lever on the 35B proxy (ROUTER-GEMM c6473d6063, MOE-COMBINE 2dc55e67f0): arm G against a
candidate arm in one load, on control-node's GPU as a tools/gpu-queue.sh job, from the export's built binary.

    python3 gate/lever_gate.py run --arm R --out DIR       the 8 fidelity prompts of gate/prefill_ornith_35b.py and a
                                                           16,384-id prompt, arms G,X,G,X,G,X, --trace (logits sha256
                                                           of every recorded step), 2 tokens
    python3 gate/lever_gate.py judge DIR --arm R --min2k 1.08 --min16k 1.08
        identity: every recorded step's sha256 and the ids of every X rep equal G's first rep (any mismatch fails);
        speed: median first_token_s of 3 interleaved reps, X tok/s / G tok/s at 2,048 and 16,384 tokens
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import prefill_ornith_35b as po  # noqa: E402

TR = "/data/tools/therock-10.1.0a20260823"


def cmd_run(a):
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    ps = po.prompts(out)
    rec = open(po.RECORDS).read().split()
    big = os.path.join(out, "p16k.ids")
    open(big, "w").write(" ".join((rec * 10)[:16384]) + "\n")
    ids = [x for p in ps + [big] for x in ("--ids-file", p)]
    arms = ",".join(["G", a.arm] * 3)
    cmd = [po.binary(), "--model", po.MODEL, "--context", "16640"] + ids + ["--trace", "--trace-top", "2", "--max-tokens", "2",
                                                                          "--arms", arms]
    env = dict(os.environ, LD_LIBRARY_PATH=TR + "/lib:" + TR + "/lib/llvm/lib")
    t0 = time.time()
    with open(os.path.join(out, "gate.err"), "w") as ef:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=ef, text=True, env=env, timeout=3000)
    open(os.path.join(out, "gate.jsonl"), "w").write(p.stdout)
    r = {"rc": p.returncode, "wall_s": round(time.time() - t0, 1), "arms": arms}
    json.dump(r, open(os.path.join(out, "runs.json"), "w"), indent=1)
    print(json.dumps(r))
    return 0 if p.returncode == 0 else 1


def cmd_judge(a):
    ev = [json.loads(l) for l in open(os.path.join(a.dir, "gate.jsonl")) if l.startswith("{")]
    tr = [(arm, {e["prompt"]: e for e in es if e.get("event") == "traced"}) for arm, es in po.split_arms(ev)]
    ref = next(t for arm, t in tr if arm == "G")
    mism, checked = [], 0
    for arm, t in tr:
        for p, e in ref.items():
            x = t.get(p)
            if x is None:
                mism.append({"arm": arm, "prompt": p, "missing": True})
                continue
            for i, (s0, s1) in enumerate(zip(e["steps"], x["steps"])):
                checked += 1
                if not s0.get("sha256") or s0.get("sha256") != s1.get("sha256"):
                    mism.append({"arm": arm, "prompt": p, "step": i})
            if e["ids"] != x["ids"]:
                mism.append({"arm": arm, "prompt": p, "ids": True})
    sp = defaultdict(list)
    for arm, t in tr:
        for e in t.values():
            sp[(arm, e["prompt_tokens"])].append(e["first_token_s"])
    med = lambda xs: sorted(xs)[len(xs) // 2]
    speed = {"%s@%d" % k: {"runs": v, "median_s": med(v), "tok_s": k[1] / med(v)} for k, v in sp.items() if k[1] in (2048, 16384)}
    ratio = lambda n: (speed.get("%s@%d" % (a.arm, n), {}).get("tok_s") or 0) / (speed.get("G@%d" % n, {}).get("tok_s") or 1)
    res = {"arm": a.arm, "identity": {"checked_steps": checked, "mismatches": mism, "pass": checked > 0 and not mism},
           "speed": {"ratio_2048": ratio(2048), "ratio_16384": ratio(16384), "min2k": a.min2k, "min16k": a.min16k},
           "runs": speed}
    res["speed"]["pass"] = res["speed"]["ratio_2048"] >= a.min2k and res["speed"]["ratio_16384"] >= a.min16k
    json.dump(res, open(os.path.join(a.dir, "verdict-gate.json"), "w"), indent=1)
    print(json.dumps({"identity": {"checked_steps": checked, "mismatches": len(mism), "pass": res["identity"]["pass"]},
                      "speed": res["speed"], "tok_s": {k: round(v["tok_s"], 1) for k, v in speed.items()}}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True)
    r.add_argument("--out", required=True)
    j = sub.add_parser("judge")
    j.add_argument("dir")
    j.add_argument("--arm", required=True)
    j.add_argument("--min2k", type=float, required=True)
    j.add_argument("--min16k", type=float, required=True)
    a = ap.parse_args()
    sys.exit({"run": cmd_run, "judge": cmd_judge}[a.cmd](a) or 0)


if __name__ == "__main__":
    main()
