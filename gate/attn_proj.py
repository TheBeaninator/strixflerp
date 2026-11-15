#!/usr/bin/env python3
"""ATTN-PROJ (criteria gate/attn-proj.toml). Run on control-node by gate/attn_proj_run.sh, which takes the GPU lease;
this script only runs the binaries and judges.

    python3 gate/attn_proj.py [--backend kbench|queue] tune|kernel|model|summary

tune     [tune]: every variant binary build/attnproj/tune/<v>/attn-proj on the four Ornith fused cases, cold, 1 token
kernel   [kernel]: build/attnproj/attn-proj (final catalogue) on every case, byte equality at every token count, cold GB/s
model    [model]: the 35B gate prompts, --staged loopback, before / after binaries (build/attnproj/{before,after})
summary  gate/results/attn-proj/summary.md, with [share] and [projection]
"""
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import xfer_r1_fold as R1  # noqa: E402
import kbench_backend as KB  # noqa: E402

BACKEND, ARGV = KB.split_backend(sys.argv[1:])     # --backend kbench (default) | queue
if ARGV and ARGV[0] == "model":
    BACKEND = "queue"          # the model stage is whole-model processes on the 35B weights: control-node's queue only
RES = KB.results_dir(os.path.join(ROOT, "gate", "results", "attn-proj"), BACKEND)
M35 = R1.M35
P5_STAGE_MS, P5_TOKEN_MS, BLOCK_US = 18.5, 83.0, 18.5e3 / 15
KERN3_BLOCK_MS = 0.088            # ORNITH-KERN3 router + outproj saving a block (try1-summary.md), measured
PEAK = 250.0
TOKENS = "1,2,3,4,8,16"
VARIANTS = "v1 2 8;v2 2 4;v3 4 4;v4 4 8"
# block-type counts: (stage 0, all 60) - gate/attn-proj.toml [facts] ornith
COUNTS = {"gdn-q6": (8, 22), "gdn-q4": (4, 23), "attn-v6": (1, 7), "attn-v4": (2, 8)}
FUSED = ("attn-v4", "attn-v6", "gdn-q4", "gdn-q6")
CONTROL = ("attn-qkv-q6",)


def ornith_cases():
    A = ["--kind", "attn", "--k", "4096"]
    G = ["--kind", "gdn", "--k", "4096"]
    return [
        ("attn-v4", A + ["--q", "16384", "--qt", "q4k", "--kv", "512", "--kt", "q4k", "--vt", "q4k"]),
        ("attn-v6", A + ["--q", "16384", "--qt", "q4k", "--kv", "512", "--kt", "q4k", "--vt", "q6k"]),
        ("gdn-q4", G + ["--qkv", "12288", "--qkvt", "q4k", "--gate", "8192", "--gt", "q4k", "--rank", "64", "--rt", "q4k"]),
        ("gdn-q6", G + ["--qkv", "12288", "--qkvt", "q6k", "--gate", "8192", "--gt", "q4k", "--rank", "64", "--rt", "q4k"]),
        ("attn-q", A + ["--q", "16384", "--qt", "q4k"]),
        ("attn-kv", A + ["--kv", "512", "--kt", "q4k", "--vt", "q4k"]),
        ("attn-gate", G + ["--gate", "8192", "--gt", "q4k"]),
        ("attn-qkv", G + ["--qkv", "12288", "--qkvt", "q4k"]),
        ("ssm-alpha", G + ["--rank", "64", "--rt", "q4k"]),
        ("attn-qkv-q6", G + ["--qkv", "12288", "--qkvt", "q6k"]),
    ]


def q35_cases():
    off = R1.offsets()
    g = ["--gguf", M35]
    out = []
    for b, qkvt in ((0, "Q6_K"), (5, "Q4_K")):
        p = f"blk.{b}."
        out.append((f"q35-gdn-blk{b}", ["--kind", "gdn", "--k", "2048", "--qkv", "8192", "--qkvt", qkvt.lower().replace("_", ""),
                                        "--gate", "4096", "--gt", "q4k", "--rank", "32", "--rt", "q4k"] + g
                    + ["--off", "qkv=" + off(p + "attn_qkv.weight", qkvt), "--off", "gate=" + off(p + "attn_gate.weight", "Q4_K"),
                       "--off", "alpha=" + off(p + "ssm_alpha.weight", "Q4_K"), "--off", "beta=" + off(p + "ssm_beta.weight", "Q4_K")]))
    for b, vt in ((3, "Q6_K"), (11, "Q4_K")):
        p = f"blk.{b}."
        out.append((f"q35-attn-blk{b}", ["--kind", "attn", "--k", "2048", "--q", "8192", "--qt", "q4k", "--kv", "512", "--kt", "q4k",
                                         "--vt", vt.lower().replace("_", "")] + g
                    + ["--off", "q=" + off(p + "attn_q.weight", "Q4_K"), "--off", "k=" + off(p + "attn_k.weight", "Q4_K"),
                       "--off", "v=" + off(p + "attn_v.weight", vt)]))
    return out


def edge_cases():
    return [
        ("edge-gdn-odd", ["--kind", "gdn", "--k", "4096", "--qkv", "1002", "--qkvt", "q4k", "--gate", "1001", "--gt", "q4k", "--rank", "3", "--rt", "q4k"]),
        ("edge-attn-k256", ["--kind", "attn", "--k", "256", "--q", "2048", "--qt", "q4k", "--kv", "256", "--kt", "q4k", "--vt", "q4k"]),
        ("edge-attn-k5120", ["--kind", "attn", "--k", "5120", "--q", "2048", "--qt", "q4k", "--kv", "256", "--kt", "q4k", "--vt", "q6k"]),
    ]


def run_cases(binary, cases, tag, extra=(), tokens=TOKENS):
    os.makedirs(RES, exist_ok=True)
    out, err = os.path.join(RES, tag + ".jsonl"), os.path.join(RES, tag + ".err")
    open(out, "w").close()
    if BACKEND == "kbench":
        KB.run_cases_k(ROOT, RES, tag, binary, [(n, ["--tokens", tokens] + a + list(extra)) for n, a in cases],
                       "strixflerp ATTN-PROJ " + tag)
        return
    loads = []
    for name, args in cases:
        while R1.load1() > 6:
            time.sleep(30)
        rc, l0, l1 = R1.run([binary, "--case", name, "--tokens", tokens] + args + list(extra), out, err)
        loads.append({"case": name, "rc": rc, "load_start": l0, "load_end": l1})
        json.dump(loads, open(os.path.join(RES, tag + "-load.json"), "w"), indent=1)


def stage_tune():
    for v in VARIANTS.split(";"):
        n = v.split()[0]
        run_cases(f"build/attnproj/tune/{n}/attn-proj", [c for c in ornith_cases() if c[0] in FUSED], f"tune-{n}",
                  ["--cold-mib", "512"], tokens="1")


def stage_kernel():
    cases = ornith_cases() + q35_cases() + edge_cases()
    timed = {c[0] for c in ornith_cases()} | {c[0] for c in q35_cases()}
    run_cases("build/attnproj/attn-proj", [(n, a + (["--cold-mib", "512"] if n in timed else ["--no-timing"])) for n, a in cases],
              "kernel-cold")


def stage_model():
    import moe_stage as MS
    import decode_kern3 as K3
    ids = json.load(open(os.path.join(ROOT, "gate/results/qwen36-35b-a3b.flerp-moe/prompt-ids.json")))
    names = list(ids)
    p1 = json.load(open(os.path.join(ROOT, "gate/results/qwen36-35b-a3b.flerp-moe-placed/equal.json")))["traced"]
    pdir = os.path.join(ROOT, "build", "attnproj", "prompts")
    os.makedirs(pdir, exist_ok=True)
    files = []
    for p in names:
        f = os.path.join(pdir, p + ".ids")
        with open(f, "w") as fh:
            fh.write(" ".join(str(i) for i in ids[p]) + "\n")
        files.append(f)
    nconf = len(K3.CONFIGS.split(";"))
    env = dict(os.environ, LD_LIBRARY_PATH="/data/tools/therock-10.1.0a20260823/lib:/data/tools/therock-10.1.0a20260823/lib/llvm/lib")
    mpath = os.path.join(RES, "model.json")
    runs = []
    for which in K3.ORDER:
        i = len(runs)
        while R1.load1() > 6:
            time.sleep(30)
        b = os.path.join(ROOT, "build", "attnproj", which, "out", "flerp-qwen35moe-generate")
        args = ["--staged", "loopback", "--model", M35, "--context", "4096", "--max-tokens", "128", "--configs", K3.CONFIGS,
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
        for e in staged:
            why += [f"{e['config']} g{e['graph']} {names[e['prompt']]}: {x}" for x in MS.judge_staged(e, p1[names[e["prompt"]]])]
        if len(staged) != nconf * len(names):
            why.append(f"{len(staged)} staged results, want {nconf * len(names)}")
        speed = [{"config": e["config"], "prompt": names[e["prompt"]], "graph": e["graph"], "world": e["world"],
                  "tok_s": e["calls"]["decode_calls"] / e["calls"]["decode_wall_ms"] * 1e3} for e in staged]
        runs.append({"run": i, "binary": which, "rc": p.returncode, "seconds": round(time.time() - t0, 1), "load_start": l0,
                     "load_end": l1, "valid_speed": l0 <= 6 and l1 <= 6, "staged_checked": len(staged), "failures": why,
                     "speed": speed, "binary_sha256": subprocess.check_output(["sha256sum", b]).decode().split()[0]})
        json.dump(runs, open(mpath, "w"), indent=1)


def med(v):
    return statistics.median(v) if v else None


def load_rows(tag):
    p = os.path.join(RES, tag + ".jsonl")
    if not os.path.isfile(p):
        return None, None
    rows = [json.loads(x) for x in open(p) if x.strip()]
    ld = json.load(open(os.path.join(RES, tag + "-load.json")))
    return rows, ld


def stage_summary():
    L = ["# ATTN-PROJ: Ornith-1.5 397B K=4096 input projections (fused q/k/v and GDN qkv/gate/alpha/beta), memory access only", "",
         f"Criteria `gate/attn-proj.toml` (committed before any GPU run). Backend {BACKEND}: " + KB.METHOD[BACKEND] + ". "
         "Old = the one-row fused launchers as at the criteria commit (gate/attn_proj_ref.hip); new = the catalogue's.", ""]
    # tune
    tl = []
    for v in VARIANTS.split(";"):
        n = v.split()[0]
        rows, ld = load_rows(f"tune-{n}")
        if not rows:
            continue
        t = {r["case"]: r["timing"] for r in rows if r["timing"]}
        nd = sum(r["outputs_differing"] + r["verify_differing"] + r["guard_touched"] for r in rows)
        tot = sum(t[c]["new"]["us"] for c in FUSED if c in t)
        tl.append((n, v, nd, tot, t, ld))
    if tl:
        L += ["## [tune] variants (rows a wave R, waves a block W), Ornith fused cases, cold, 1 token", "",
              "| variant | R W | differing | " + " | ".join(f"{c} new us (GB/s)" for c in FUSED) + " | sum new us |", "|---" * (4 + len(FUSED)) + "|"]
        for n, v, nd, tot, t, ld in tl:
            L.append(f"| {n} | {' '.join(v.split()[1:])} | {nd} | " + " | ".join(f"{t[c]['new']['us']} ({t[c]['new']['GBps']})" if c in t else "-" for c in FUSED) + f" | {tot:.1f} |")
        ok = [x for x in tl if x[2] == 0 and all(c in x[4] for c in FUSED)]
        if ok:
            best = min(ok, key=lambda x: x[3])
            L += ["", f"Pick ([tune] rule): **{best[0]}** (R W = {' '.join(best[1].split()[1:])}), sum {best[3]:.1f} us.", ""]
    rows, ld = load_rows("kernel-cold")
    save = {}
    if rows:
        L += [KB.method_line(RES, "kernel-cold"), ""]
        bad = [x for x in ld if x["load_start"] > 6 or x["load_end"] > 6 or x["rc"] != 0]
        diff = [r for r in rows if r["outputs_differing"] or r["verify_differing"] or r["guard_touched"]]
        L += ["## [kernel] final source, cache-cold", "",
              f"{len(rows)} (case, x, tokens) results, {sum(r['outputs'] for r in rows)} outputs compared three ways (new, old, verify path); "
              f"results with any differing byte or a touched guard: {len(diff)} -> **{'PASS' if not diff else 'FAIL'}**. "
              f"Cases off-load or failed: {len(bad)} {bad if bad else ''}", "",
              "| case | segments | MB | differing new/old, verify/old (all x, tokens) | old GB/s | new GB/s | new/old | old us | new us | new % of 250 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        cases = []
        for r in rows:
            if r["case"] not in cases:
                cases.append(r["case"])
        judged_fail = []
        for c in cases:
            rs = [r for r in rows if r["case"] == c]
            t = next((r["timing"] for r in rs if r["timing"]), None)
            r0 = rs[0]
            seg = ", ".join(f"{k} {v}" for k, v in r0["segments"].items() if not v.startswith("0 "))
            nd = (sum(r["outputs_differing"] for r in rs), sum(r["verify_differing"] for r in rs), sum(r["outputs"] for r in rs))
            if t:
                save[c] = t["ref"]["us"] - t["new"]["us"]
                sp = lambda a: f"{a['GBps']} ({a['GBps_min']}..{a['GBps_max']})"
                L.append(f"| {c} | {seg} | {r0['weight_bytes'] / 1e6:.1f} | {nd[0]}, {nd[1]} of {nd[2]} | {sp(t['ref'])} | {sp(t['new'])} | "
                         f"{t['speedup']:.3f} | {t['ref']['us']} | {t['new']['us']} | {100 * t['new']['GBps'] / PEAK:.0f} % |")
                if not c.startswith(("q35-", "edge-")) and c not in CONTROL and t["new"]["GBps"] < t["ref"]["GBps"]:
                    judged_fail.append(c)
            else:
                L.append(f"| {c} | {seg} | {r0['weight_bytes'] / 1e6:.1f} | {nd[0]}, {nd[1]} of {nd[2]} | - | - | - | - | - | - |")
        L += ["", f"continue_rule (new >= old cold GB/s on every Ornith fused and per-shape case, 0 differing): "
              f"**{'PASS' if not judged_fail and not diff and not bad else 'FAIL'}** {judged_fail if judged_fail else ''}", ""]
        tm = {r["case"]: r["timing"] for r in rows if r["timing"]}
        if all(c in tm for c in FUSED):
            L += ["## [share] input projections' share of a block (old kernel; block = 18.5 ms / 15 = 1233 us)", "",
                  "| block type | old us | share | new us | share |", "|---|---|---|---|---|"]
            for c in FUSED:
                L.append(f"| {c} | {tm[c]['ref']['us']} | {100 * tm[c]['ref']['us'] / BLOCK_US:.1f} % | {tm[c]['new']['us']} | {100 * tm[c]['new']['us'] / BLOCK_US:.1f} % |")
            s0 = sum(COUNTS[c][0] * save[c] for c in FUSED) / 1e3
            sa = sum(COUNTS[c][1] * save[c] for c in FUSED) / 1e3
            k3s, k3t = 15 * KERN3_BLOCK_MS, 60 * KERN3_BLOCK_MS
            L += ["", "## [projection] computed, not measured", "",
                  f"Saving: stage 0 {s0:.3f} ms a step (8 gdn-q6 + 4 gdn-q4 + 1 attn-v6 + 2 attn-v4), all 60 blocks {sa:.3f} ms a token.",
                  f"This kernel alone: stage 0 {P5_STAGE_MS} -> {P5_STAGE_MS - s0:.2f} ms; token {P5_TOKEN_MS} -> {P5_TOKEN_MS - sa:.2f} ms = "
                  f"{1000 / P5_TOKEN_MS:.2f} -> {1000 / (P5_TOKEN_MS - sa):.2f} tok/s (projected).",
                  f"With ORNITH-KERN3 router + outproj (0.088 ms a block): stage 0 {P5_STAGE_MS} -> {P5_STAGE_MS - k3s - s0:.2f} ms; token "
                  f"{P5_TOKEN_MS} -> {P5_TOKEN_MS - k3t - sa:.2f} ms = {1000 / P5_TOKEN_MS:.2f} -> {1000 / (P5_TOKEN_MS - k3t - sa):.2f} tok/s (projected).", ""]
    p = os.path.join(RES, "model.json")
    if os.path.isfile(p):
        runs = json.load(open(p))
        L += ["## [model] 35B, 4 gate prompts x 6 configs, --staged loopback", ""]
        for r in runs:
            L.append(f"- run {r['run']} `{r['binary']}` (sha256 {r['binary_sha256'][:12]}): exit {r['rc']}, {r['seconds']} s, load {r['load_start']:.2f} -> {r['load_end']:.2f}"
                     f"{'' if r['valid_speed'] else ' (speed INVALID: load above 6)'}; {r['staged_checked']} results judged against P1 ids + 128 logits sha256; "
                     f"failures: {len(r['failures'])} {r['failures'][:3] if r['failures'] else ''}")
        v = {w: [s["tok_s"] for r in runs if r["binary"] == w and r["valid_speed"] for s in r["speed"] if s["config"] == "1:even:on"]
             for w in ("before", "after")}
        if v["before"] and v["after"]:
            L += ["", f"Decode tok/s world 1 graph on, all prompts, median (min..max): before {med(v['before']):.2f} ({min(v['before']):.2f}..{max(v['before']):.2f}), "
                  f"after {med(v['after']):.2f} ({min(v['after']):.2f}..{max(v['after']):.2f}), after/before {med(v['after']) / med(v['before']):.3f}."]
        L.append("")
    with open(os.path.join(RES, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    {"tune": stage_tune, "kernel": stage_kernel, "model": stage_model, "summary": stage_summary}[ARGV[0]]()
