#!/usr/bin/env python3
"""SHAPE-TRANSFERS R1 fold (criteria gate/xfer-r1-fold.toml). Run on control-node by gate/xfer_r1_fold_run.sh, which
takes the GPU lease; this script only runs the binaries and judges.

    python3 gate/xfer_r1_fold.py remeasure|kernel|model|summary

remeasure  the R1 tool (build/xfer-r1/xfer-r1-splitk) on the erratic rows, three repeats each
kernel     build/xfer-r1-fold/xfer-r1-fold on every 35B and Ornith expert shape: byte equality and GB/s
model      the 35B gate prompts, --staged loopback, before / after / before / after binaries: P1 hashes, decode tok/s
summary    gate/results/xfer-r1-fold/summary.md from the three
"""
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
RES = os.path.join(ROOT, "gate", "results", "xfer-r1-fold")
M35 = "/data/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-Q4_K_M.gguf"
POLITE = ["timeout", "1500", "ionice", "-c3", "nice", "-n", "19", "taskset", "-c", "0-3", "env", "OMP_NUM_THREADS=4"]
CONFIGS = "1:even:on;1:even:on;1:even:on;2:even:on;2:even:off"


def load1():
    with open("/proc/loadavg") as f:
        return float(f.read().split()[0])


def run(cmd, out_path, err_path):
    l0 = load1()
    with open(out_path, "a") as o, open(err_path, "a") as e:
        rc = subprocess.call(POLITE + cmd, stdout=o, stderr=e, cwd=ROOT)
    return rc, l0, load1()


def offsets():
    c = json.load(open(os.path.join(ROOT, "gate/results/qwen36-35b-a3b.census/census.json")))
    t = {x["name"]: x for x in c["tensors"]}

    def off(name, want):
        assert t[name]["type"] == want, (name, t[name]["type"], want)
        return str(t[name]["offset"])
    return off


def kernel_cases():
    off = offsets()
    g = ["--gguf", M35]
    q35 = [
        ("q35-gateup", ["--type", "q4k", "--rows", "512", "--k", "2048", "--experts", "256", "--slots", "8", "--pair"] + g
         + ["--off-a", off("blk.10.ffn_gate_exps.weight", "Q4_K"), "--off-b", off("blk.10.ffn_up_exps.weight", "Q4_K")]),
        ("q35-down-q6", ["--type", "q6k", "--rows", "2048", "--k", "512", "--experts", "256", "--slots", "8", "--xshare", "1"] + g
         + ["--off-a", off("blk.10.ffn_down_exps.weight", "Q6_K")]),
        ("q35-down-q4", ["--type", "q4k", "--rows", "2048", "--k", "512", "--experts", "256", "--slots", "8", "--xshare", "1"] + g
         + ["--off-a", off("blk.11.ffn_down_exps.weight", "Q4_K")]),
        ("q35-sh-gateup", ["--type", "q4k", "--rows", "512", "--k", "2048", "--shared", "--pair"] + g
         + ["--off-a", off("blk.10.ffn_gate_shexp.weight", "Q4_K"), "--off-b", off("blk.10.ffn_up_shexp.weight", "Q4_K")]),
        ("q35-sh-down-q6", ["--type", "q6k", "--rows", "2048", "--k", "512", "--shared"] + g
         + ["--off-a", off("blk.10.ffn_down_shexp.weight", "Q6_K")]),
        ("q35-sh-down-q4", ["--type", "q4k", "--rows", "2048", "--k", "512", "--shared"] + g
         + ["--off-a", off("blk.11.ffn_down_shexp.weight", "Q4_K")]),
    ]
    orn = [
        ("ornith-gateup", ["--type", "q4k", "--rows", "1024", "--k", "4096", "--experts", "64", "--slots", "10", "--pair"]),
        ("ornith-down-q6", ["--type", "q6k", "--rows", "4096", "--k", "1024", "--experts", "64", "--slots", "10", "--xshare", "1"]),
        ("ornith-down-q4", ["--type", "q4k", "--rows", "4096", "--k", "1024", "--experts", "64", "--slots", "10", "--xshare", "1"]),
        ("ornith-sh-gateup", ["--type", "q4k", "--rows", "1024", "--k", "4096", "--shared", "--pair"]),
        ("ornith-sh-down-q6", ["--type", "q6k", "--rows", "4096", "--k", "1024", "--shared"]),
        ("ornith-sh-down-q4", ["--type", "q4k", "--rows", "4096", "--k", "1024", "--shared"]),
    ]
    return q35 + orn


def stage_kernel(tag="kernel", extra=()):
    out, err = os.path.join(RES, tag + ".jsonl"), os.path.join(RES, tag + ".err")
    open(out, "w").close()
    loads = []
    only = os.environ.get("XFER_CASES", "").split(",") if os.environ.get("XFER_CASES") else None
    for name, args in kernel_cases():
        if only and name not in only:
            continue
        rc, l0, l1 = run(["build/xfer-r1-fold/xfer-r1-fold", "--case", name] + args + list(extra), out, err)
        loads.append({"case": name, "rc": rc, "load_start": l0, "load_end": l1})
    json.dump(loads, open(os.path.join(RES, tag + "-load.json"), "w"), indent=1)


def stage_kernel_rounds():
    """The amendment of 2026-11-12 (gate/xfer-r1-fold.toml [kernel_rounds]): timing only, 7 alternating rounds.
    XFER_CASES=a,b reruns only those cases into kernel-rounds-rerun.jsonl (cases whose first run was off-load)."""
    stage_kernel("kernel-rounds-rerun" if os.environ.get("XFER_CASES") else "kernel-rounds", ["--tokens-equal", "1", "--rounds", "7"])


def stage_remeasure():
    out, err = os.path.join(RES, "remeasure.jsonl"), os.path.join(RES, "remeasure.err")
    open(out, "w").close()
    off = offsets()
    tool = "build/xfer-r1/xfer-r1-splitk"
    q35 = ["--case", "q35-gateup", "--rows", "512", "--k", "2048", "--experts", "256", "--slots", "8", "--pair", "--gguf", M35,
           "--off-a", off("blk.10.ffn_gate_exps.weight", "Q4_K"), "--off-b", off("blk.10.ffn_up_exps.weight", "Q4_K")]
    og = ["--case", "ornith-gateup", "--rows", "1024", "--k", "4096", "--experts", "64", "--slots", "10", "--pair"]
    od = ["--case", "ornith-down", "--rows", "4096", "--k", "1024", "--experts", "64", "--slots", "10", "--xshare", "1"]
    loads = []
    for rep in range(3):
        for args in ([*q35, "--tokens", "2"], [*q35, "--tokens", "4"], [*q35, "--tokens", "8"], [*og, "--tokens", "4"], [*od, "--tokens", "4"]):
            rc, l0, l1 = run([tool] + args, out, err)
            loads.append({"rep": rep, "case": args[1], "tokens": args[-1], "rc": rc, "load_start": l0, "load_end": l1})
    json.dump(loads, open(os.path.join(RES, "remeasure-load.json"), "w"), indent=1)


def stage_model(order=("before", "after", "before", "after"), append=False):
    import moe_stage as MS
    ids = json.load(open(os.path.join(ROOT, "gate/results/qwen36-35b-a3b.flerp-moe/prompt-ids.json")))
    names = list(ids)
    p1 = json.load(open(os.path.join(ROOT, "gate/results/qwen36-35b-a3b.flerp-moe-placed/equal.json")))["traced"]
    pdir = os.path.join(ROOT, "build", "xfer-r1-fold", "prompts")
    os.makedirs(pdir, exist_ok=True)
    files = []
    for p in names:
        f = os.path.join(pdir, p + ".ids")
        with open(f, "w") as fh:
            fh.write(" ".join(str(i) for i in ids[p]) + "\n")
        files.append(f)
    env = dict(os.environ, LD_LIBRARY_PATH="/data/tools/therock-10.1.0a20260823/lib:/data/tools/therock-10.1.0a20260823/lib/llvm/lib")
    mpath = os.path.join(RES, "model.json")
    runs = json.load(open(mpath)) if append and os.path.isfile(mpath) else []
    for which in order:
        i = len(runs)
        while load1() > 6:
            time.sleep(30)
        b = os.path.join(ROOT, "build", "xfer-r1-fold", which, "out", "flerp-qwen35moe-generate")
        args = ["--staged", "loopback", "--model", M35, "--context", "4096", "--max-tokens", "128", "--configs", CONFIGS,
                "--epoch", "7", "--deadline-ms", "60000", "--single-reader", "--pause-ms", "100"] + [x for f in files for x in ("--ids-file", f)]
        log = os.path.join(RES, f"model-{i}-{which}.txt")
        l0, t0 = load1(), time.time()
        with open(log, "w") as lf:
            p = subprocess.run(POLITE + [b] + args, stdout=subprocess.PIPE, stderr=lf, env=env, cwd=ROOT)
        l1 = load1()
        ev = []
        for line in p.stdout.decode().splitlines():
            try:
                ev.append(json.loads(line))
            except ValueError:
                pass
        staged = [e for e in ev if e.get("event") == "staged"]
        why = [] if p.returncode == 0 else [f"exit {p.returncode}"]
        for e in staged:
            w = MS.judge_staged(e, p1[names[e["prompt"]]])
            why += [f"{e['config']} {names[e['prompt']]}: {x}" for x in w]
        if len(staged) != 5 * len(names):
            why.append(f"{len(staged)} staged results, want {5 * len(names)}")
        speed = [{"config": e["config"], "prompt": names[e["prompt"]], "graph": e["graph"],
                  "tok_s": e["calls"]["decode_calls"] / e["calls"]["decode_wall_ms"] * 1e3,
                  "decode_ms_per_step": e["stages"][0]["decode_ms_per_step"] if e["world"] == 1 else None} for e in staged]
        runs.append({"run": i, "binary": which, "rc": p.returncode, "seconds": round(time.time() - t0, 1), "load_start": l0,
                     "load_end": l1, "valid_speed": l0 <= 6 and l1 <= 6, "failures": why, "speed": speed,
                     "binary_sha256": subprocess.check_output(["sha256sum", b]).decode().split()[0]})
        json.dump(runs, open(os.path.join(RES, "model.json"), "w"), indent=1)


def med(v):
    return statistics.median(v) if v else None


def stage_summary():
    L = ["# SHAPE-TRANSFERS R1 fold: arm a in kernel.flerp.moe_kquant (Q4_K, and Q6_K by the same recipe)", "",
         "Criteria `gate/xfer-r1-fold.toml` (committed before the GPU runs). Host control-node.", ""]
    p = os.path.join(RES, "remeasure.jsonl")
    if os.path.isfile(p):
        rows = [json.loads(x) for x in open(p) if x.strip()]
        ld = json.load(open(os.path.join(RES, "remeasure-load.json")))
        bad = [x for x in ld if x["load_start"] > 6 or x["load_end"] > 6]
        L += ["## Re-measure of the erratic R1 rows (xfer-r1-splitk, 3 repeats)", "",
              f"Load: {'all runs at or under 6' if not bad else f'{len(bad)} run(s) above 6: INVALID'}"
              f" (1-min load {min(x['load_start'] for x in ld):.2f}..{max(max(x['load_start'], x['load_end']) for x in ld):.2f}).", "",
              "| case | tokens | ref GB/s (median, min..max) | arm a GB/s | a / ref | a bit-equal |", "|---|---|---|---|---|---|"]
        keys = sorted({(r["case"], r["tokens"]) for r in rows})
        for c, t in keys:
            rs = [r for r in rows if r["case"] == c and r["tokens"] == t]
            ref = [r["timing"]["ref"]["GBps"] for r in rs]
            a = [r["timing"]["a"]["GBps"] for r in rs]
            eq = all(r["correctness"]["a"]["outputs_differing_from_ref"] == 0 for r in rs)
            L.append(f"| {c} | {t} | {med(ref):.1f} ({min(ref):.1f}..{max(ref):.1f}) | {med(a):.1f} ({min(a):.1f}..{max(a):.1f}) | {med(a) / med(ref):.3f} | {eq} |")
        L.append("")
    for tag in ("kernel", "kernel-rounds", "kernel-rounds-rerun"):
        L += kernel_section(tag)
    p = os.path.join(RES, "model.json")
    if os.path.isfile(p):
        L += model_section(p)
    with open(os.path.join(RES, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def kernel_section(tag):
    L = []
    p = os.path.join(RES, tag + ".jsonl")
    if os.path.isfile(p):
        rows = [json.loads(x) for x in open(p) if x.strip()]
        ld = json.load(open(os.path.join(RES, tag + "-load.json")))
        bad = [x for x in ld if x["load_start"] > 6 or x["load_end"] > 6 or x["rc"] != 0]
        diff = [r for r in rows if r["outputs_differing"] != 0]
        note = {"kernel": " - byte equality final; its GB/s are NOT usable (single 200-launch loops, clock states: see the toml's [kernel_rounds])",
                "kernel-rounds": " - 7 alternating rounds a case (the Ornith cases here ran off-load and were rerun below)",
                "kernel-rounds-rerun": " - the off-load Ornith cases again, 7 rounds"}.get(tag, "")
        L += [f"## Kernel ({tag}){note}", "",
              f"{len(rows)} (case, tokens) results, {sum(r['outputs'] for r in rows)} outputs; differing bytes in {len(diff)} of them"
              f" -> **{'PASS' if not diff and rows else 'FAIL'}** (byte equality). Runs off-load or failed: {len(bad)} {bad if bad else ''}", "",
              "| case | type | tokens | outputs differing | ref GB/s | new GB/s | new / ref |", "|---|---|---|---|---|---|---|"]
        for r in rows:
            t = r["timing"]
            if not t:
                continue
            sp = (lambda a: f"{a['GBps']} ({a['GBps_min']}..{a['GBps_max']})" if "GBps_min" in a else f"{a['GBps']}")
            L.append(f"| {r['case']} | {r['type']} | {r['tokens']} | {r['outputs_differing']} of {r['outputs']} | "
                     f"{sp(t['ref'])} | {sp(t['new'])} | {t['speedup']:.3f} |")
        L.append(f"Byte equality was checked at every tokens value of the run ({sorted({r['tokens'] for r in rows})}); rows without timing are not listed.")
        L.append("")
    return L


def model_section(p):
    L = []
    if True:
        runs = json.load(open(p))
        L += ["## Whole model: 35B, 4 gate prompts, --staged loopback", ""]
        for r in runs:
            L.append(f"- run {r['run']} `{r['binary']}` (sha256 {r['binary_sha256'][:12]}): exit {r['rc']}, {r['seconds']} s, load {r['load_start']:.2f} -> {r['load_end']:.2f}"
                     f"{'' if r['valid_speed'] else ' (speed INVALID: load above 6)'}; P1/P2 hash failures: {len(r['failures'])} {r['failures'][:3] if r['failures'] else ''}")
        L += ["", "Decode tok/s, world 1, graph on (median of the valid world-1 runs, min..max):", "",
              "| prompt | before | after | after / before |", "|---|---|---|---|"]
        prompts = []
        for r in runs:
            for s in r["speed"]:
                if s["prompt"] not in prompts:
                    prompts.append(s["prompt"])
        allb, alla = [], []
        for pr in prompts:
            v = {w: [s["tok_s"] for r in runs if r["binary"] == w and r["valid_speed"] for s in r["speed"]
                     if s["prompt"] == pr and s["config"] == "1:even:on"] for w in ("before", "after")}
            if v["before"] and v["after"]:
                allb += v["before"]
                alla += v["after"]
                L.append(f"| {pr} | {med(v['before']):.2f} ({min(v['before']):.2f}..{max(v['before']):.2f}) | "
                         f"{med(v['after']):.2f} ({min(v['after']):.2f}..{max(v['after']):.2f}) | {med(v['after']) / med(v['before']):.3f} |")
        if allb:
            L.append(f"| all | {med(allb):.2f} | {med(alla):.2f} | {med(alla) / med(allb):.3f} |")
        L.append("")
    return L


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    {"model-more": lambda: stage_model(tuple(os.environ.get("XFER_ORDER", "before,after").split(",")), append=True), "remeasure": stage_remeasure, "kernel": stage_kernel, "kernel-rounds": stage_kernel_rounds, "model": stage_model, "summary": stage_summary}[sys.argv[1]]()
