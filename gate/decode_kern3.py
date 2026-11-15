#!/usr/bin/env python3
"""DECODE-KERN3 (criteria gate/decode-kern3.toml). Run on control-node by gate/decode_kern3_run.sh, which takes the GPU
lease; this script only runs the binaries and judges.

    python3 gate/decode_kern3.py [--backend kbench|queue] kernel|model|summary

kernel   build/kern3/decode-kern3 on every case of the criteria: byte equality (two x, router at every token count)
         and GB/s, 7 alternating rounds, one process a case
model    the 35B gate prompts, --staged loopback, before / after binaries (build/kern3/{before,after}): P1 hashes and
         decode tok/s
summary  gate/results/decode-kern3/summary.md
"""
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import xfer_r1_fold as R1  # noqa: E402  (offsets, polite runner, the R1 model stage's shape)
import kbench_backend as KB  # noqa: E402

BACKEND, ARGV = KB.split_backend(sys.argv[1:])     # --backend kbench (default) | queue
if ARGV and ARGV[0] == "model":
    BACKEND = "queue"          # whole-model processes on the 35B weights: control-node's queue only
RES = KB.results_dir(os.path.join(ROOT, "gate", "results", "decode-kern3"), BACKEND)
M35 = R1.M35
CONFIGS = "1:even:on;1:even:on;1:even:on;1:even:off;2:even:on;2:even:off"
ORDER = ("before", "after", "before", "after", "after", "before")


def kernel_cases():
    off = R1.offsets()
    g = ["--gguf", M35]
    teq = ["--tokens-equal", "1,2,3,8,21,128,1024"]
    cases = []
    for b in (0, 10, 39):
        cases.append((f"router-blk{b}", ["--kind", "router", "--hidden", "2048", "--experts", "256"] + g + teq
                      + ["--off-a", off(f"blk.{b}.ffn_gate_inp.weight", "F32"), "--off-b", off(f"blk.{b}.ffn_gate_inp_shexp.weight", "F32")]))
    cases.append(("lm-head", ["--kind", "gemv", "--type", "q6k", "--rows", "248320", "--k", "2048"] + g + ["--off-a", off("output.weight", "Q6_K")]))
    for n, t, rows, k, ty in (("attn-out-blk3", "blk.3.attn_output.weight", 2048, 4096, "Q4_K"),
                              ("attn-out-blk39", "blk.39.attn_output.weight", 2048, 4096, "Q4_K"),
                              ("ssm-out-blk0", "blk.0.ssm_out.weight", 2048, 4096, "Q4_K"),
                              ("ssm-out-blk10", "blk.10.ssm_out.weight", 2048, 4096, "Q4_K"),
                              ("ssm-alpha-blk0", "blk.0.ssm_alpha.weight", 32, 2048, "Q4_K"),
                              ("attn-v-blk3", "blk.3.attn_v.weight", 512, 2048, "Q6_K"),
                              ("attn-qkv-blk0", "blk.0.attn_qkv.weight", 8192, 2048, "Q6_K"),
                              ("attn-gate-blk0", "blk.0.attn_gate.weight", 4096, 2048, "Q4_K")):
        cases.append((n, ["--kind", "gemv", "--type", ty.lower().replace("_", ""), "--rows", str(rows), "--k", str(k)] + g
                      + ["--off-a", off(t, ty)]))
    cases += [
        ("syn-router-4096x512", ["--kind", "router", "--hidden", "4096", "--experts", "512", "--tokens-equal", "1,2,21,128"]),
        ("syn-router-noshared", ["--kind", "router", "--hidden", "2048", "--experts", "128", "--no-shared", "--tokens-equal", "1,8"]),
        ("syn-router-hidden1536", ["--kind", "router", "--hidden", "1536", "--experts", "64", "--tokens-equal", "1,8"]),
        ("syn-q4k-oddM", ["--kind", "gemv", "--type", "q4k", "--rows", "1001", "--k", "2048"]),
        ("syn-q6k-oddM", ["--kind", "gemv", "--type", "q6k", "--rows", "1001", "--k", "2048"]),
        ("syn-q4k-k256", ["--kind", "gemv", "--type", "q4k", "--rows", "4096", "--k", "256"]),
        ("syn-q6k-k256", ["--kind", "gemv", "--type", "q6k", "--rows", "4096", "--k", "256"]),
        ("syn-q4k-k5120", ["--kind", "gemv", "--type", "q4k", "--rows", "6144", "--k", "5120"]),
        ("syn-q6k-k5120", ["--kind", "gemv", "--type", "q6k", "--rows", "6144", "--k", "5120"]),
    ]
    return cases


def stage_kernel(tag="kernel", extra=()):
    os.makedirs(RES, exist_ok=True)
    out, err = os.path.join(RES, tag + ".jsonl"), os.path.join(RES, tag + ".err")
    open(out, "w").close()
    only = os.environ.get("K3_CASES", "").split(",") if os.environ.get("K3_CASES") else None
    loads = []
    if BACKEND == "kbench":
        # K runs the synthetic-code cases only (the criteria's real-weight cases read the 35B GGUF, which stays on
        # control-node's queue backend and the NVMe rule); the skipped case names are written beside the results.
        sel = [(n, a) for n, a in kernel_cases() if n.startswith("syn-") and not (only and n not in only)]
        json.dump({"skipped_real_weight_cases": [n for n, a in kernel_cases() if not n.startswith("syn-")]},
                  open(os.path.join(RES, tag + "-skipped.json"), "w"), indent=1)
        KB.run_cases_k(ROOT, RES, tag, "build/kern3/decode-kern3", [(n, a + list(extra)) for n, a in sel], "strixflerp DECODE-KERN3 " + tag)
        return
    for name, args in kernel_cases():
        if only and name not in only:
            continue
        while R1.load1() > 6:
            time.sleep(30)
        rc, l0, l1 = R1.run(["build/kern3/decode-kern3", "--case", name] + args + list(extra), out, err)
        loads.append({"case": name, "rc": rc, "load_start": l0, "load_end": l1})
        json.dump(loads, open(os.path.join(RES, tag + "-load.json"), "w"), indent=1)


def stage_model():
    import moe_stage as MS
    os.makedirs(RES, exist_ok=True)
    ids = json.load(open(os.path.join(ROOT, "gate/results/qwen36-35b-a3b.flerp-moe/prompt-ids.json")))
    names = list(ids)
    p1 = json.load(open(os.path.join(ROOT, "gate/results/qwen36-35b-a3b.flerp-moe-placed/equal.json")))["traced"]
    pdir = os.path.join(ROOT, "build", "kern3", "prompts")
    os.makedirs(pdir, exist_ok=True)
    files = []
    for p in names:
        f = os.path.join(pdir, p + ".ids")
        with open(f, "w") as fh:
            fh.write(" ".join(str(i) for i in ids[p]) + "\n")
        files.append(f)
    nconf = len(CONFIGS.split(";"))
    env = dict(os.environ, LD_LIBRARY_PATH="/data/tools/therock-10.1.0a20260823/lib:/data/tools/therock-10.1.0a20260823/lib/llvm/lib")
    mpath = os.path.join(RES, "model.json")
    append = os.environ.get("K3_APPEND") == "1"
    runs = json.load(open(mpath)) if append and os.path.isfile(mpath) else []
    order = tuple(os.environ["K3_ORDER"].split(",")) if os.environ.get("K3_ORDER") else ORDER
    for which in order:
        i = len(runs)
        while R1.load1() > 6:
            time.sleep(30)
        b = os.path.join(ROOT, "build", "kern3", which, "out", "flerp-qwen35moe-generate")
        args = ["--staged", "loopback", "--model", M35, "--context", "4096", "--max-tokens", "128", "--configs", CONFIGS,
                "--epoch", "7", "--deadline-ms", "60000", "--single-reader", "--pause-ms", "100"] + [x for f in files for x in ("--ids-file", f)]
        log = os.path.join(RES, f"model-{i}-{which}.txt")
        l0, t0 = R1.load1(), time.time()
        with open(log, "w") as lf:
            p = subprocess.run(R1.POLITE + [b] + args, stdout=subprocess.PIPE, stderr=lf, env=env, cwd=ROOT)
        l1 = R1.load1()
        ev = []
        for line in p.stdout.decode().splitlines():
            try:
                ev.append(json.loads(line))
            except ValueError:
                pass
        staged = [e for e in ev if e.get("event") == "staged"]
        why = [] if p.returncode == 0 else [f"exit {p.returncode}"]
        checked = 0
        for e in staged:
            w = MS.judge_staged(e, p1[names[e["prompt"]]])
            checked += 1
            why += [f"{e['config']} g{e['graph']} {names[e['prompt']]}: {x}" for x in w]
        if len(staged) != nconf * len(names):
            why.append(f"{len(staged)} staged results, want {nconf * len(names)}")
        speed = [{"config": e["config"], "prompt": names[e["prompt"]], "graph": e["graph"], "world": e["world"],
                  "tok_s": e["calls"]["decode_calls"] / e["calls"]["decode_wall_ms"] * 1e3} for e in staged]
        runs.append({"run": i, "binary": which, "rc": p.returncode, "seconds": round(time.time() - t0, 1), "load_start": l0,
                     "load_end": l1, "valid_speed": l0 <= 6 and l1 <= 6, "staged_checked": checked, "failures": why,
                     "speed": speed, "binary_sha256": subprocess.check_output(["sha256sum", b]).decode().split()[0]})
        json.dump(runs, open(mpath, "w"), indent=1)


def med(v):
    return statistics.median(v) if v else None


def stage_summary():
    L = ["# DECODE-KERN3: router, lm head and Q4_K out-projection kernels, memory access only", "",
         f"Criteria `gate/decode-kern3.toml` (committed before the GPU runs). Backend {BACKEND}: " + KB.METHOD[BACKEND] + ".", ""]
    notes = {
        "kernel-cold": "FINAL source, cache-cold ([kernel_cold]): the run the continue rule is judged on",
        "kernel-rerun": "FINAL source, cache-hot (as [kernel]), for comparison",
        "kernel": "first source (router 2 waves x 16 float4, GEMV Q4_K+Q6_K 4 waves), cache-hot: byte equality final; hot GB/s, see [kernel_cold]",
        "kernel-cold-v1": "first source, cache-cold",
        "kernel-cold-v2": "second try (router loads before x staging; GEMV 16-byte headers, 4-sub groups ahead): slower, dropped",
        "kernel-cold-v3": "third (router 4 waves x 8 float4; GEMV codes-then-decode, 8 waves, Q4_K+Q6_K): lm head Q6_K no faster -> Q6_K route dropped",
        "kernel-rerun-v3": "third, cache-hot",
    }
    for tag in ("kernel-cold", "kernel-rerun", "kernel", "kernel-cold-v1", "kernel-cold-v2", "kernel-cold-v3", "kernel-rerun-v3"):
        p = os.path.join(RES, tag + ".jsonl")
        if not os.path.isfile(p):
            continue
        rows = [json.loads(x) for x in open(p) if x.strip()]
        ld = json.load(open(os.path.join(RES, tag + "-load.json")))
        bad = [x for x in ld if x["load_start"] > 6 or x["load_end"] > 6 or x["rc"] != 0]
        diff = [r for r in rows if r["outputs_differing"] != 0 or r["gap_touched"] != 0]
        L += [f"## Kernel ({tag}): {notes.get(tag, '')}", "", KB.method_line(RES, tag), "",
              f"{len(rows)} (case, x, tokens) results, {sum(r['outputs'] for r in rows)} outputs compared; results with any differing byte: "
              f"{len(diff)} -> **{'PASS' if not diff and rows else 'FAIL'}** (byte equality). Cases off-load or failed: {len(bad)} {bad if bad else ''}", "",
              "| case | type | rows x k | outputs differing (all x, tokens) | ref GB/s | new GB/s | new / ref |", "|---|---|---|---|---|---|---|"]
        cases = []
        for r in rows:
            if r["case"] not in cases:
                cases.append(r["case"])
        for c in cases:
            rs = [r for r in rows if r["case"] == c]
            nd, no = sum(r["outputs_differing"] for r in rs), sum(r["outputs"] for r in rs)
            t = next((r["timing"] for r in rs if r["timing"]), None)
            sp = (lambda a: f"{a['GBps']} ({a['GBps_min']}..{a['GBps_max']})")
            r0 = rs[0]
            L.append(f"| {c} | {r0['type']} | {r0['rows']} x {r0['k']} | {nd} of {no} | "
                     + (f"{sp(t['ref'])} | {sp(t['new'])} | {t['speedup']:.3f} |" if t else "- | - | - |"))
        L.append("")
    p = os.path.join(RES, "model.json")
    if os.path.isfile(p):
        runs = json.load(open(p))
        L += ["## Whole model: 35B, 4 gate prompts x 6 configs, --staged loopback", "", f"Configs `{CONFIGS}`.", ""]
        for r in runs:
            L.append(f"- run {r['run']} `{r['binary']}` (sha256 {r['binary_sha256'][:12]}): exit {r['rc']}, {r['seconds']} s, load {r['load_start']:.2f} -> {r['load_end']:.2f}"
                     f"{'' if r['valid_speed'] else ' (speed INVALID: load above 6)'}; {r['staged_checked']} (config, prompt) results judged against P1 ids + 128 logits sha256; "
                     f"failures: {len(r['failures'])} {r['failures'][:3] if r['failures'] else ''}")
        L += ["", "Decode tok/s, world 1, graph on (median of the load-valid runs, min..max):", "",
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
            L.append(f"Goal: after >= 68.3 tok/s (+20 % over R1's 56.95). Measured {med(alla):.2f} -> "
                     f"{'MET' if med(alla) >= 68.3 else 'NOT MET'}.")
        L.append("")
    with open(os.path.join(RES, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    {"kernel": stage_kernel, "kernel-rerun": lambda: stage_kernel("kernel-rerun"),
     "kernel-cold": lambda: stage_kernel("kernel-cold", ["--cold-mib", "512"]), "model": stage_model,
     "summary": stage_summary}[ARGV[0]]()
