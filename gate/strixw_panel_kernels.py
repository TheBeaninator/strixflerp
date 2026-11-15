#!/usr/bin/env python3
"""Kernel gate of the strixw panel layout (criteria: gate/strixw-panel-kernels.toml, committed before any GPU run).

Stages (run on control-node inside /data/projects/strixflerp; tools/pod-run.sh):
  build    flerp assemble qwen38-flash-next.strixite-p16, cmake, ninja -j2 of the kernel bench and the server, under
           ionice -c3 nice -n 19, after the 1-minute load average is <= [host] max_load
  kernels  the bench on control-node's GPU (start it through tools/gpu-queue.sh, which waits for a free GPU): every class,
           path and M of [run]; raw lines to gate/results/strixw-panel-kernels/raw.jsonl
  report   reads raw.jsonl: equality table per class / path, speed table vs the origin, the [criteria] verdicts ->
           summary.json and README.md beside it
  --plan   prints the whole-model token-identity plan ([whole_model]); runs and writes nothing
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "gate", "strixw-panel-kernels.toml")
RES = os.path.join(ROOT, "gate", "results", "strixw-panel-kernels")
NICE = ["ionice", "-c3", "nice", "-n", "19"]


def load():
    with open(CFG, "rb") as f:
        return tomllib.load(f)


def wait_load(limit):
    while os.getloadavg()[0] > limit:
        print(f"load {os.getloadavg()[0]:.1f} > {limit}: waiting", flush=True)
        time.sleep(60)


def ninja_dir(cfg):
    return os.path.join(ROOT, "build", cfg["host"]["recipe"], "ninja")


def stage_build(cfg):
    h = cfg["host"]
    wait_load(h["max_load"])
    tree = os.path.join(ROOT, "build", h["recipe"])
    for cmd in ([sys.executable, "flerp", "assemble", h["recipe"]],
                NICE + ["cmake", "-G", "Ninja", "-S", tree, "-B", ninja_dir(cfg)],
                NICE + ["ninja", "-C", ninja_dir(cfg), f"-j{h['build_jobs']}", h["bench"],
                        "strixflerp-" + h["recipe"], "strixflerp-prefill-bench"]):
        print("+", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc:
            sys.exit(rc)


def stage_kernels(cfg):
    h, r = cfg["host"], cfg["run"]
    wait_load(h["max_load"])
    os.makedirs(RES, exist_ok=True)
    cmd = NICE + [os.path.join(ninja_dir(cfg), h["bench"]), "--weights", h["weights"], "--out",
                  os.path.join(RES, "raw.jsonl"), "--decode-m", ",".join(map(str, r["decode_m"])), "--prefill-m",
                  ",".join(map(str, r["prefill_m"])), "--active", str(r["active"]), "--reps", str(r["reps"]),
                  "--warm", str(r["warm"])]
    for c in r["classes"]:
        cmd += ["--class", c]
    print("+", " ".join(shlex.quote(c) for c in cmd), flush=True)
    t0 = time.time()
    with open(os.path.join(RES, "bench.log"), "w") as log:
        log.write("+ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        log.flush()
        rc = subprocess.call(["/usr/bin/time", "-v"] + cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    print(f"bench rc={rc} in {time.time() - t0:.0f} s", flush=True)
    return rc


def noise_word(o, p):
    if p["q3"] < o["q1"]:
        return "faster"
    if p["q1"] > o["q3"]:
        return "slower"
    return "noise"


def report(cfg):
    rows, summ, skipped = [], None, []
    for line in open(os.path.join(RES, "raw.jsonl")):
        d = json.loads(line)
        if d.get("summary"):
            summ = d
        elif "skipped" in d:
            skipped.append(d)
        else:
            rows.append(d)
    verdicts = []
    # equality table
    eq = {}
    for d in rows:
        k = (d["class"], d["bits"], d["codes"], d["path"] + (" " + d["math"] if "math" in d else ""))
        e = eq.setdefault(k, {"cases": 0, "equal": 0, "diff_bytes": 0})
        e["cases"] += 1
        e["equal"] += d["equal"]
        e["diff_bytes"] += d["diff_bytes"]
    unequal = [d for d in rows if not d["equal"]]
    verdicts.append({"criterion": "equal", "pass": not unequal and bool(rows),
                     "detail": f"{len(rows) - len(unequal)} of {len(rows)} cases byte-equal"})
    bad = [d for d in rows if d.get("bad_id")]
    verdicts.append({"criterion": "bad_id", "pass": bool(bad) and all(d["equal"] for d in bad),
                     "detail": f"{sum(d['equal'] for d in bad)} of {len(bad)} out-of-range-id cases byte-equal"})
    classes = [c.split("=")[0] for c in cfg["run"]["classes"]]
    dense_cls = [c for c in classes if c not in ("exp_gu", "exp_down")]
    miss = [f"{c} q{b} {p}" for c in dense_cls for b in (4, 8) for p in ("dense-decode", "dense-prefill-wmma")
            if not any(d["class"] == c and d["bits"] == b and d["path"] == p for d in rows)]
    verdicts.append({"criterion": "coverage", "pass": not miss and bool(bad), "detail": "missing: " + (", ".join(miss) or "none")})
    gb = (summ or {}).get("bytes_read", 0) / 1e9
    verdicts.append({"criterion": "read", "pass": summ is not None and gb <= 2.0, "detail": f"{gb:.3f} GB read"})
    # speed
    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else None
    dec = {}
    for d in rows:
        if d["path"] == "dense-decode":
            dec.setdefault(d["class"], []).append(d["panel_pct_of_origin_time"])
    worst = {c: med(v) for c, v in dec.items()}
    verdicts.append({"criterion": "speed_decode_dense", "pass": bool(worst) and all(v <= 105 for v in worst.values()),
                     "detail": "median panel % of origin time per class: " + ", ".join(f"{c} {v:.1f}" for c, v in sorted(worst.items()))})
    pre = [d for d in rows if d["path"] == "dense-prefill-wmma" and d["codes"] == "real" and d["M"] in (2048, 8192)]
    verdicts.append({"criterion": "speed_prefill_dense", "pass": bool(pre) and all(d["panel_ms"]["median"] <= d["origin_ms"]["median"] for d in pre),
                     "detail": "; ".join(f"{d['class']} M{d['M']} {d['out']} {d['panel_pct_of_origin_time']:.1f}%" for d in pre)})

    def served(d):
        wide = d["M"] >= 4096
        return ((d["path"] == "expert-prefill-swiglu-f16h" and d["math"] == ("wide256" if wide else "narrow")) or
                (d["path"] == "expert-prefill-combine-f16h-shared" and d["math"] == ("wide" if wide else "narrow")))
    ex = [d for d in rows if d["path"].startswith("expert-prefill") and not d.get("bad_id") and d["M"] in (2048, 8192) and served(d)]
    verdicts.append({"criterion": "speed_prefill_expert", "pass": bool(ex) and all(d["panel_ms"]["median"] < d["origin_ms"]["median"] for d in ex),
                     "detail": "; ".join(f"{d['path']} {d['math']} M{d['M']} {d['panel_pct_of_origin_time']:.1f}% ({noise_word(d['origin_ms'], d['panel_ms'])})" for d in ex)})
    out = {"criteria_file": "gate/strixw-panel-kernels.toml", "cases": len(rows), "skipped": len(skipped),
           "bytes_read": (summ or {}).get("bytes_read"), "verdicts": verdicts,
           "equality": [{"class": k[0], "bits": k[1], "codes": k[2], "path": k[3], **v} for k, v in sorted(eq.items())]}
    json.dump(out, open(os.path.join(RES, "summary.json"), "w"), indent=1)
    L = ["# strixw panel kernels: kernel gate results", "",
         f"Criteria: `gate/strixw-panel-kernels.toml` (committed before the run). Raw: `raw.jsonl` (one case a line), "
         f"bench log `bench.log`. {len(rows)} cases, {len(skipped)} skipped (output too large), "
         f"{gb:.3f} GB read from the fleet file.", "", "## Verdicts", "", "| criterion | pass | detail |", "|---|---|---|"]
    L += [f"| {v['criterion']} | {'PASS' if v['pass'] else 'FAIL'} | {v['detail']} |" for v in verdicts]
    L += ["", "## Equality (cases byte-equal / cases)", "", "| class | bits | codes | path | equal / cases | differing bytes |",
          "|---|---|---|---|---|---|"]
    L += [f"| {k[0]} | Q{k[1]} | {k[2]} | {k[3]} | {v['equal']} / {v['cases']} | {v['diff_bytes']} |" for k, v in sorted(eq.items())]
    L += ["", "## Speed (median ms over the reps; q1-q3 in brackets; GB/s on the weight bytes the case touches)", "",
          "| class | bits | path | M | out | origin ms | panel ms | panel % | GB/s origin / panel | |", "|---|---|---|---|---|---|---|---|---|---|"]
    for d in rows:
        if d.get("bad_id"):
            continue
        o, p = d["origin_ms"], d["panel_ms"]
        L.append(f"| {d['class']} | Q{d['bits']}{'' if d['codes'] == 'real' else ' syn'} | {d['path']}{' ' + d['math'] if 'math' in d else ''} | "
                 f"{d['M']} | {d.get('out', 'bf16')} | {o['median']:.4f} [{o['q1']:.4f}-{o['q3']:.4f}] | "
                 f"{p['median']:.4f} [{p['q1']:.4f}-{p['q3']:.4f}] | {d['panel_pct_of_origin_time']:.1f} | "
                 f"{d['origin_gbps']:.0f} / {d['panel_gbps']:.0f} | {noise_word(o, p)} |")
    open(os.path.join(RES, "README.md"), "w").write("\n".join(L) + "\n")
    print(json.dumps(verdicts, indent=1))


def plan(cfg):
    w = cfg["whole_model"]
    rec = cfg["host"]["recipe"]
    b = f"/data/projects/strixflerp/build/{rec}/ninja"
    base = "/data/projects/strixflerp/build/qwen38-flash-next.strixite/ninja"
    print(f"""Whole-model token-identity plan for recipes/{rec}.toml (NOT run; the owner arranges the hold)

Where: worker A - it holds the p16 file {w['p16_file']} and the fleet v4 file {w['v4_file']} (the p16 file's
source) on its local NVMe; {w['needs']}. Worker A is cordoned for StrixFold: this needs its GPU free for the hold.
Binaries come from control-node's builds over /data (NFS); no build on the worker.

0. On control-node (no GPU): build both recipes -j2 under ionice/nice:
     gate/strixw_panel_kernels.py --stage build                    # {rec}: server, prefill bench, kernel bench
     python3 flerp assemble qwen38-flash-next.strixite && cmake/ninja its server   # the unmodified recipe
   Check the p16 file first (another task finishes and verifies it): inspect_strixw --verify on A (reads 74.6 GB, ~2 min).

1. Prompt-pass logits and greedy ids, 8k and 32k prompts, prefill and decode speed (one process per file, one
   weights load each, ~75 GB device memory, sequential):
     {b}/strixflerp-prefill-bench --weights {w['v4_file']} --ngram {w['ngram']} --out <scratch>/v4 \\
         --prompts 8192,32768 --variants base:chunk=8192 --rounds 3 --gen 128 --save-logits 0
     {b}/strixflerp-prefill-bench --weights {w['p16_file']} --ngram {w['ngram']} --out <scratch>/p16 \\
         --prompts 8192,32768 --variants base:chunk=8192 --rounds 3 --gen 128 --save-logits 0
   The p16 build on the v4 file runs strixite's kernels (every view row-major: the adapters hand them back; the
   overlays change nothing on a v4 file), so (1a) is the baseline; as a cross-check on the build itself, also run the
   unmodified recipe's server in step 2. Compare per prompt: sha256 of the last logits rows, argmax, greedy ids (must be
   identical); prefill tok/s per 8k / 32k (median, min-max of 3 rounds) and decode tok/s (T = 1, MTP off).
   Time: load ~1-2 min a file (local NVMe), 2 prompts x 3 rounds ~2 min; ~10 min for both.

2. Server token identity on the phase-1 requests (gate/qwen38-flash-next.strixite.toml [server] args and [run]
   requests: 3 requests x 3 repeats, seed 1234, max_tokens 256, MTP on then off - four server loads):
     reference: {base}/strixflerp-qwen38-flash-next.strixite --weights {w['v4_file']} --ngram {w['ngram']} ...
     assembled: {b}/strixflerp-{rec} --weights {w['p16_file']} --ngram {w['ngram']} ...
   (gate/strixite_recipe.py's run_server / compare, pointed at worker A's paths and the two binaries.) Compare the
   greedy ids of every request (identical) and decode tok/s (median, min-max). Also check stderr: the four
   'flerp-adapter kernel.flerp.* foreign_launches=N' lines must show N > 0 for the p16 file and 0 for the v4 file.
   Time: 4 loads x ~2 min + 18 requests ~3 min: ~15 min.

Pass: {w['pass']}. Total hold: about 30 minutes of worker A's GPU, ~150 GB read from its NVMe (two files, two to three
loads each; the page cache keeps neither). Results would go to gate/results/strixw-panel-kernels/whole-model/.
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "kernels", "report"], action="append", default=[])
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()
    cfg = load()
    if a.plan:
        plan(cfg)
        return
    rc = 0
    for s in a.stage:
        if s == "build":
            stage_build(cfg)
        elif s == "kernels":
            rc = stage_kernels(cfg)
        else:
            report(cfg)
    sys.exit(rc)


if __name__ == "__main__":
    main()
