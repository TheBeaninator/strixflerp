#!/usr/bin/env python3
"""PREFILL-ORNITH fidelity and speed on the 35B proxy (gate/prefill-ornith.toml [fidelity], [fidelity_speed]).

Runs ON control-node, as one job of tools/gpu-queue.sh (one process at a time on the control GPU), in a clean export
(POD_SYNC_DEST=/data/projects/strixflerp-pfo) whose `./flerp gate recipe qwen36-35b-a3b.flerp-moe --stages build`
built the binary. The same binary three ways: A = FLERP_MOE_GROUPED=0 (the current prompt path), B = 1 (grouped
experts), D = --decode-prompt (prompt through the decode step, FP32 activations: the yardstick).

    python3 gate/prefill_ornith_35b.py run  [--out DIR]     the GPU part (about 15 minutes)
    python3 gate/prefill_ornith_35b.py judge DIR            the verdict from DIR's JSON lines (no GPU)
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "/data/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-Q4_K_M.gguf"
RECORDS = os.path.join(ROOT, "gate/results/ornith15-397b.p4/long.ids")     # 1,808 ids (the P5 long prompt)
POSITIONS = 32
GREEDY = 64
TOP = 1000
ROCM = "/data/tools/therock-10.1.0a20260823"        # toolchains.toml therock-10.1 (gufo_recipe.runtime_env)


def binary():
    c = glob.glob(os.path.join(ROOT, "build/qwen36-35b-a3b.flerp-moe/out/**/flerp-qwen35moe-generate"), recursive=True)
    if not c:
        raise SystemExit("no built flerp-qwen35moe-generate under build/qwen36-35b-a3b.flerp-moe/out")
    return c[0]


def prompts(out):
    """Eight prompts of 96 to 3,000 ids (all above the grouped path's 64-row floor); the last one spans two chunks."""
    rec = open(RECORDS).read().split()
    if not os.path.exists(RECORDS):
        raise SystemExit("missing " + RECORDS)
    cuts = [(0, 96), (200, 500), (100, 700), (900, 1800), (300, 1600), (0, 1808)]
    ps = [rec[a:b] for a, b in cuts]
    ps.append(rec + rec[:240])                          # 2,048: one chunk (the speed prompt)
    ps.append(rec + rec[::-1][:1192])                   # 3,000: two chunks
    paths = []
    for i, p in enumerate(ps):
        path = os.path.join(out, "p%d.ids" % i)
        open(path, "w").write(" ".join(p) + "\n")
        paths.append(path)
    return paths


def run(args, env, log):
    full = dict(os.environ, **env)
    full["LD_LIBRARY_PATH"] = ROCM + "/lib:" + ROCM + "/lib/llvm/lib" + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")
    t0 = time.time()
    with open(log, "w") as f:
        p = subprocess.run([binary()] + args, stdout=subprocess.PIPE, stderr=f, text=True, env=full, timeout=3600)
    ev = []
    for line in p.stdout.splitlines():
        try:
            ev.append(json.loads(line))
        except ValueError:
            pass
    fallbacks = sum("re-run on the per-token path" in ln for ln in open(log, errors="replace"))
    return {"rc": p.returncode, "events": ev, "wall_s": time.time() - t0, "fallbacks": fallbacks}


def split_arms(events):
    """[(arm, [events])] in run order, from the `arm` events --arms prints."""
    out = []
    for e in events:
        if e.get("event") == "arm":
            out.append((e["arm"], []))
        elif out:
            out[-1][1].append(e)
    return out


def cmd_run(a):
    # One model load a phase (owner 2026-11-14: weights loaded once per process from the node's own copy, every arm
    # and rep measured in that process): phase 1 = greedy, arms A B A B A B over all prompts (the 2,048-id prompt's
    # first_token_s of each arm is the speed sample); phase 2 = traced A B D with A's greedy ids forced.
    out = a.out or os.path.join(ROOT, "build/pfo-35b-" + time.strftime("%m%dT%H%MZ", time.gmtime()))
    os.makedirs(out, exist_ok=True)
    ps = prompts(out)
    ids_args = [x for p in ps for x in ("--ids-file", p)]
    base = ["--model", MODEL, "--context", "4096"]
    r = run(base + ids_args + ["--max-tokens", str(GREEDY), "--arms", "A,B,A,B,A,B"], {}, os.path.join(out, "greedy.err"))
    json.dump(r, open(os.path.join(out, "greedy.json"), "w"))
    arms = split_arms(r["events"])
    first = {}
    for arm, ev in arms:
        first.setdefault(arm, ev)
    forced = []
    for e in first.get("A", []):
        if e.get("event") == "generated":
            path = os.path.join(out, "force%d.ids" % e["prompt"])
            open(path, "w").write(" ".join(str(i) for i in e["ids"]) + "\n")
            forced.append(path)
    speed = []
    for rep, (arm, ev) in enumerate(arms):
        g = next((e for e in ev if e.get("event") == "generated" and e["prompt"] == 6), {})
        speed.append({"arm": arm, "rep": rep // 2, "first_token_s": g.get("first_token_s"), "prompt_tokens": len(g.get("prompt_ids", []))})
    json.dump(speed, open(os.path.join(out, "speed.json"), "w"), indent=1)
    force_args = [x for p in forced for x in ("--force-file", p)]
    trace = ["--trace", "--trace-top", str(TOP), "--record-at", ",".join(str(i) for i in range(POSITIONS))]
    r2 = run(base + ids_args + force_args + trace + ["--arms", "A,B,D"], {}, os.path.join(out, "traced.err"))
    json.dump(r2, open(os.path.join(out, "traced.json"), "w"))
    print(out)


def cmd_ref(a):
    """Against a reference arm (owner 2026-11-14: grouped vs full FP32). Phase 1, one load: free-running greedy of every
    arm through --trace (so D, the prompt through the decode step, runs too), 64 tokens a prompt. Phase 2, one load:
    every arm teacher-forced on the reference arm's greedy ids, logits at the prefill end and 31 decode positions."""
    out = a.out
    os.makedirs(out, exist_ok=True)
    ps = prompts(out)
    ids_args = [x for p in ps for x in ("--ids-file", p)]
    base = ["--model", MODEL, "--context", "4096"]
    r = run(base + ids_args + ["--trace", "--trace-top", "2", "--max-tokens", str(GREEDY), "--arms", a.arms], {},
            os.path.join(out, "greedy.err"))
    json.dump(r, open(os.path.join(out, "greedy.json"), "w"))
    ref = dict(split_arms(r["events"])).get(a.ref, [])
    forced = []
    for e in ref:
        if e.get("event") == "traced":
            path = os.path.join(out, "force%d.ids" % e["prompt"])
            open(path, "w").write(" ".join(str(i) for i in e["ids"]) + "\n")
            forced.append(path)
    if len(forced) != len(ps):
        raise SystemExit("reference arm %s gave %d of %d prompts" % (a.ref, len(forced), len(ps)))
    force_args = [x for p in forced for x in ("--force-file", p)]
    trace = ["--trace", "--trace-top", str(TOP), "--record-at", ",".join(str(i) for i in range(POSITIONS))]
    r2 = run(base + ids_args + force_args + trace + ["--arms", a.arms], {}, os.path.join(out, "traced.err"))
    json.dump(r2, open(os.path.join(out, "traced.json"), "w"))
    print(out)


def cmd_bulk(a):
    """Step 2 speed (gate/prefill-ornith-dense.toml Q1-Q3, G4). Process 1 (one load): the 2,048-id prompt, arms A B E F X
    three times interleaved. Processes 2 and 3 (one load each, the batch bound is per process): a 16,384-id prompt,
    arms B F X three times, at FLERP_MAX_PROMPT_BATCH 4096 and 16384. All through --trace (greedy, 2 tokens) so the
    prefill-end logits are recorded beside first_token_s."""
    out = a.out
    os.makedirs(out, exist_ok=True)
    ps = prompts(out)
    rec = open(RECORDS).read().split()
    big = (rec * 10)[:16384]
    bigp = os.path.join(out, "p16k.ids")
    open(bigp, "w").write(" ".join(big) + "\n")
    tr = ["--trace", "--trace-top", str(TOP), "--max-tokens", "2"]
    r = run(["--model", MODEL, "--context", "4096", "--ids-file", ps[6]] + tr + ["--arms", "A,B,E,F,X,A,B,E,F,X,A,B,E,F,X"], {},
            os.path.join(out, "short.err"))
    json.dump(r, open(os.path.join(out, "short.json"), "w"))
    for batch in (4096, 16384):
        r = run(["--model", MODEL, "--context", "16640", "--ids-file", bigp] + tr + ["--arms", "B,F,X,B,F,X,B,F,X"],
                {"FLERP_MAX_PROMPT_BATCH": str(batch)}, os.path.join(out, "bulk%d.err" % batch))
        json.dump(r, open(os.path.join(out, "bulk%d.json" % batch), "w"))
    print(out)


def cmd_judge_bulk(a):
    d = a.dir
    def times(name):
        r = json.load(open(os.path.join(d, name)))
        t, steps = {}, {}
        for arm, ev in split_arms(r["events"]):
            for e in ev:
                if e.get("event") == "traced":
                    t.setdefault(arm, []).append((e["prompt_tokens"], e["first_token_s"]))
                    steps.setdefault(arm, e["steps"])
        return t, steps, r["rc"], r["fallbacks"]
    res = {}
    for name in ("short.json", "bulk4096.json", "bulk16384.json"):
        if not os.path.exists(os.path.join(d, name)):
            continue
        t, steps, rc, fb = times(name)
        res[name] = {"rc": rc, "fallbacks": fb, "arms": {arm: {"prompt_tokens": v[0][0], "median_s": sorted(x[1] for x in v)[len(v) // 2],
                                                                "tok_s": v[0][0] / sorted(x[1] for x in v)[len(v) // 2], "runs": [x[1] for x in v]}
                                                          for arm, v in t.items()}}
        res[name]["_steps"] = steps
    g4 = None
    if "bulk4096.json" in res and "bulk16384.json" in res:
        s0, s1 = res["bulk4096.json"]["_steps"].get("X"), res["bulk16384.json"]["_steps"].get("X")
        if s0 and s1:
            g4 = [kl(x, y)[0] for x, y in zip(s0, s1)]
    for v in res.values():
        v.pop("_steps", None)
    sh = res.get("short.json", {}).get("arms", {})
    b4 = res.get("bulk4096.json", {}).get("arms", {})
    b16 = res.get("bulk16384.json", {}).get("arms", {})
    res["Q1"] = {"F_over_B": sh["F"]["tok_s"] / sh["B"]["tok_s"] if "F" in sh and "B" in sh else None}
    res["Q1"]["pass"] = bool(res["Q1"]["F_over_B"] and res["Q1"]["F_over_B"] >= 1.3)
    res["Q2"] = {"F_16384_over_4096": b16["F"]["tok_s"] / b4["F"]["tok_s"] if "F" in b16 and "F" in b4 else None}
    res["Q2"]["pass"] = bool(res["Q2"]["F_16384_over_4096"] and res["Q2"]["F_16384_over_4096"] >= 1.1)
    res["G4"] = {"kl_X_4096_vs_16384": g4, "pass": bool(g4) and max(g4) <= 1e-4}
    json.dump(res, open(os.path.join(d, "verdict-bulk.json"), "w"), indent=1)
    for k in ("short.json", "bulk4096.json", "bulk16384.json"):
        if k in res:
            print(k, {arm: (v["prompt_tokens"], round(v["median_s"], 3), round(v["tok_s"], 1)) for arm, v in res[k]["arms"].items()}, "fallbacks", res[k]["fallbacks"])
    print("Q1", res["Q1"], "Q2", res["Q2"], "G4", res["G4"])


def cmd_attn(a):
    """gate/prefill-attn.toml M1-M2: one load, the 2,048-id and a 16,384-id prompt, arms F and G three times
    interleaved, the origin's 4,096-row chunk bound (bulk off). Greedy through --trace, 2 tokens."""
    out = a.out
    os.makedirs(out, exist_ok=True)
    ps = prompts(out)
    rec = open(RECORDS).read().split()
    bigp = os.path.join(out, "p16k.ids")
    open(bigp, "w").write(" ".join((rec * 10)[:16384]) + "\n")
    r = run(["--model", MODEL, "--context", "16640", "--ids-file", ps[6], "--ids-file", bigp, "--trace", "--trace-top", "2",
             "--max-tokens", "2", "--arms", "F,G,F,G,F,G"], {}, os.path.join(out, "attn.err"))
    json.dump(r, open(os.path.join(out, "attn.json"), "w"))
    t = {}
    for arm, ev in split_arms(r["events"]):
        for e in ev:
            if e.get("event") == "traced":
                t.setdefault((arm, e["prompt_tokens"]), []).append(e["first_token_s"])
    res = {"%s@%d" % k: {"runs": v, "median_s": sorted(v)[len(v) // 2], "tok_s": k[1] / sorted(v)[len(v) // 2]} for k, v in t.items()}
    get = lambda arm, n: res.get("%s@%d" % (arm, n), {}).get("tok_s")
    res["M1"] = {"G_over_F_16384": (get("G", 16384) or 0) / (get("F", 16384) or 1)}
    res["M1"]["pass"] = res["M1"]["G_over_F_16384"] >= 2.0
    res["M2"] = {"G_over_F_2048": (get("G", 2048) or 0) / (get("F", 2048) or 1)}
    res["M2"]["pass"] = res["M2"]["G_over_F_2048"] >= 0.95
    res["rc"], res["fallbacks"] = r["rc"], r["fallbacks"]
    json.dump(res, open(os.path.join(out, "verdict-attn.json"), "w"), indent=1)
    print(json.dumps(res))


def cmd_g4(a):
    """G4 diagnosis (chunk-size drift): an 8,192-id prompt at FLERP_MAX_PROMPT_BATCH 4096 (two chunks) and 8192 (one),
    one load each, arms A F G X, traced: the hidden state after blocks 0, 1, 2 (DeltaNet), 3 (the first attention block)
    and 7 (sums over every row and corner rows), and the logits (sha256, top 1,000) at the prefill end and 3 forced
    decode positions. Where the two chunkings first differ names the cause."""
    out = a.out
    os.makedirs(out, exist_ok=True)
    rec = open(RECORDS).read().split()
    p = os.path.join(out, "p%dk.ids" % (a.tokens // 1024))
    open(p, "w").write(" ".join((rec * 10)[:a.tokens]) + "\n")
    f = os.path.join(out, "force.ids")
    open(f, "w").write(" ".join(rec[:8]) + "\n")
    b0, b1 = (int(x) for x in a.batches.split(","))
    for batch in (b0, b1):
        r = run(["--model", MODEL, "--context", str(a.tokens + 256), "--ids-file", p, "--force-file", f, "--trace", "--trace-top", str(TOP),
                 "--record-at", "0,1,2,3", "--trace-layers", "0,1,2,3,7", "--arms", a.arms],
                {"FLERP_MAX_PROMPT_BATCH": str(batch)}, os.path.join(out, "g4-%d.err" % batch))
        json.dump(r, open(os.path.join(out, "g4-%d.json" % batch), "w"))
    res = {}
    load = lambda b: {arm: next(e for e in ev if e.get("event") == "traced")
                      for arm, ev in split_arms(json.load(open(os.path.join(out, "g4-%d.json" % b)))["events"])}
    r4, r8 = load(b0), load(b1)
    for arm in r4:
        h4, h8 = r4[arm]["hidden"], r8[arm]["hidden"]
        layers = []
        for x, y in zip(h4, h8):
            layers.append({"layer": x["layer"], "sum_equal": x["sum_all_tokens"] == y["sum_all_tokens"],
                           "sum_4096": x["sum_all_tokens"], "sum_8192": y["sum_all_tokens"],
                           "last_l2_4096": x["last_token_l2"], "last_l2_8192": y["last_token_l2"],
                           "corners_equal": x["corners"] == y["corners"]})
        s4, s8 = r4[arm]["steps"], r8[arm]["steps"]
        res[arm] = {"layers": layers, "logits_sha_equal": [x.get("sha256") == y.get("sha256") for x, y in zip(s4, s8)],
                    "kl": [kl(x, y)[0] for x, y in zip(s4, s8)], "first_s": [r4[arm]["first_token_s"], r8[arm]["first_token_s"]]}
    json.dump(res, open(os.path.join(out, "verdict-g4.json"), "w"), indent=1)
    for arm, v in res.items():
        print(arm, "layers sum equal", [(l["layer"], l["sum_equal"], l["corners_equal"]) for l in v["layers"]],
              "logits sha equal", v["logits_sha_equal"], "kl", ["%.2g" % k for k in v["kl"]])


def cmd_judge_ref(a):
    """Every arm against the reference arm: KL(ref || X) mean and max over the forced positions, top-1 agreement
    (and the reference margin at each miss), greedy ids equal to the reference's (first divergence a prompt)."""
    d = a.dir
    tr = dict(split_arms(json.load(open(os.path.join(d, "traced.json")))["events"]))
    gr = dict(split_arms(json.load(open(os.path.join(d, "greedy.json")))["events"]))
    steps = {arm: {e["prompt"]: e["steps"] for e in ev if e.get("event") == "traced"} for arm, ev in tr.items()}
    greedy = {arm: {e["prompt"]: e["ids"] for e in ev if e.get("event") == "traced"} for arm, ev in gr.items()}
    R = steps[a.ref]
    res = {"reference": a.ref}
    for arm in steps:
        if arm == a.ref:
            continue
        kls, top, miss, n = [], 0, [], 0
        for p in sorted(R):
            for sr, sx in zip(R[p], steps[arm].get(p, [])):
                kls.append(kl(sr, sx)[0])
                n += 1
                if sr["top"][0][0] == sx["top"][0][0]:
                    top += 1
                else:
                    miss.append({"prompt": p, "position": sr["position"], "ref_margin": sr["top"][0][1] - sr["top"][1][1]})
        div = []
        for p in sorted(greedy[a.ref]):
            g0, g1 = greedy[a.ref][p], greedy[arm].get(p, [])
            i = next((j for j in range(min(len(g0), len(g1))) if g0[j] != g1[j]), None)
            if i is None and len(g0) != len(g1):
                i = min(len(g0), len(g1))
            div.append(i)
        res[arm] = {"positions": n, "mean_kl": sum(kls) / n, "max_kl": max(kls), "top1": top / n, "misses": miss,
                    "greedy_equal_prompts": sum(x is None for x in div), "first_divergence": div}
    json.dump(res, open(os.path.join(d, "verdict-ref-%s.json" % a.ref), "w"), indent=1)
    for arm, v in res.items():
        if isinstance(v, dict):
            print(arm, "vs", a.ref, "mean_kl %.5f max_kl %.4f top1 %.4f greedy_equal %d/%d first_div %s" % (
                v["mean_kl"], v["max_kl"], v["top1"], v["greedy_equal_prompts"], len(v["first_divergence"]), v["first_divergence"]))


SHORT_ARMS = "A,G,Z,X,D"
SHORT_TOKENS = 48


def short_prompts(out):
    """gate/ornith-bridge.toml [run]: the three P5 prompts the 397B ran, plus 8 records slices of 24 ids."""
    p5 = json.load(open(os.path.join(ROOT, "gate/results/qwen36-35b-a3b.flerp-moe/prompt-ids.json")))
    rec = open(RECORDS).read().split()
    named = [(n, [str(x) for x in p5[n]]) for n in ("bridge", "code", "rivers")]
    named += [("rec%d" % o, rec[o:o + 24]) for o in (0, 150, 300, 450, 600, 900, 1200, 1500)]
    paths = []
    for n, ids in named:
        path = os.path.join(out, "s-%s.ids" % n)
        open(path, "w").write(" ".join(ids) + "\n")
        paths.append(path)
    json.dump([n for n, _ in named], open(os.path.join(out, "names.json"), "w"))
    return [n for n, _ in named], paths


def cmd_short(a):
    """ORNITH-BRIDGE (gate/ornith-bridge.toml). Phase 1, one load: greedy SHORT_TOKENS a prompt, every arm, through
    --trace. Phase 2, one load: every arm forced on D's greedy ids and on A's greedy ids (each prompt listed twice),
    logits top 1,000 at every generated position."""
    out = a.out
    os.makedirs(out, exist_ok=True)
    names, ps = short_prompts(out)
    ids_args = [x for p in ps for x in ("--ids-file", p)]
    base = ["--model", MODEL, "--context", "4096"]
    r = run(base + ids_args + ["--trace", "--trace-top", "2", "--max-tokens", str(SHORT_TOKENS), "--arms", SHORT_ARMS], {},
            os.path.join(out, "greedy.err"))
    json.dump(r, open(os.path.join(out, "greedy.json"), "w"))
    arms = dict(split_arms(r["events"]))
    g = {arm: {e["prompt"]: e["ids"] for e in ev if e.get("event") == "traced"} for arm, ev in arms.items()}
    # A prompt whose D or A continuation stopped early (a stop token) cannot be forced to SHORT_TOKENS: leave it out
    # of phase 2 and say so.
    keep = [i for i in range(len(ps)) if len(g.get("D", {}).get(i, [])) >= SHORT_TOKENS and len(g.get("A", {}).get(i, [])) >= SHORT_TOKENS]
    ids2, force2, plan = [], [], []
    for traj in ("D", "A"):
        for i in keep:
            f = os.path.join(out, "force-%s-%s.ids" % (traj, names[i]))
            open(f, "w").write(" ".join(str(t) for t in g[traj][i]) + "\n")
            ids2 += ["--ids-file", ps[i]]
            force2 += ["--force-file", f]
            plan.append({"traj": traj, "prompt": i, "name": names[i]})
    json.dump({"names": names, "phase2": plan, "left_out": [names[i] for i in range(len(ps)) if i not in keep]},
              open(os.path.join(out, "plan.json"), "w"), indent=1)
    trace = ["--trace", "--trace-top", str(TOP), "--record-at", ",".join(str(i) for i in range(SHORT_TOKENS))]
    r2 = run(base + ids2 + force2 + trace + ["--arms", SHORT_ARMS], {}, os.path.join(out, "traced.err"))
    json.dump(r2, open(os.path.join(out, "traced.json"), "w"))
    print(out)


def cmd_judge_short(a):
    d = a.dir
    plan = json.load(open(os.path.join(d, "plan.json")))
    names = plan["names"]
    gr = dict(split_arms(json.load(open(os.path.join(d, "greedy.json")))["events"]))
    greedy = {arm: {e["prompt"]: e["ids"] for e in ev if e.get("event") == "traced"} for arm, ev in gr.items()}
    tr = json.load(open(os.path.join(d, "traced.json")))
    steps = {}   # steps[arm][traj][prompt] = [step]
    for arm, ev in split_arms(tr["events"]):
        for e in ev:
            if e.get("event") == "traced":
                p = plan["phase2"][e["prompt"]]
                steps.setdefault(arm, {}).setdefault(p["traj"], {})[p["prompt"]] = e["steps"]
    lp = lambda st, t: next((v - st["logsumexp"] for tt, v in st["top"] if tt == t), None)
    margin = lambda st: st["top"][0][1] - st["top"][1][1]
    res = {"names": names, "left_out": plan["left_out"], "arms": {}, "splits": [], "fallbacks": tr["fallbacks"]}
    D = steps.get("D", {})
    for arm in steps:
        if arm == "D":
            continue
        kls, top, n = [], 0, 0
        for p, sd in D.get("D", {}).items():
            for x, y in zip(sd, steps[arm]["D"].get(p, [])):
                kls.append(kl(x, y)[0])
                top += x["top"][0][0] == y["top"][0][0]
                n += 1
        div = {}
        for p in greedy["D"]:
            g0, g1 = greedy["D"][p], greedy[arm].get(p, [])
            i = next((j for j in range(min(len(g0), len(g1))) if g0[j] != g1[j]), None)
            div[names[p]] = i
        res["arms"][arm] = {"positions": n, "mean_kl_vs_D": sum(kls) / n if n else None, "max_kl_vs_D": max(kls) if kls else None,
                            "top1_vs_D": top / n if n else None, "greedy_first_divergence_vs_D": div,
                            "greedy_equal_D": sum(v is None for v in div.values())}
    # A-vs-G (and A-vs-Z, A-vs-X) splits, judged on the reference's logits forced on A's ids (common prefix)
    for q in ("G", "Z", "X"):
        for p in greedy["A"]:
            ga, gq = greedy["A"][p], greedy[q].get(p, [])
            i = next((j for j in range(min(len(ga), len(gq))) if ga[j] != gq[j]), None)
            if i is None:
                continue
            row = {"pair": "A-" + q, "prompt": names[p], "index": i, "token_A": ga[i], "token_" + q: gq[i]}
            sD = D.get("A", {}).get(p)
            if sD is not None and i < len(sD):
                s = sD[i]
                la, lq = lp(s, ga[i]), lp(s, gq[i])
                row.update({"D_top1": s["top"][0][0], "D_margin_top2": margin(s), "D_logp_A": la, "D_logp_" + q: lq,
                            "gap_D": abs(la - lq) if la is not None and lq is not None else None,
                            "A_is_D_top1": ga[i] == s["top"][0][0], q + "_is_D_top1": gq[i] == s["top"][0][0]})
                for arm in ("A", q):
                    sa = steps.get(arm, {}).get("A", {}).get(p)
                    if sa is not None and i < len(sa):
                        row[arm + "_margin_top2"] = margin(sa[i])
            else:
                row["note"] = "no forced logits for this prompt (left out of phase 2)"
            res["splits"].append(row)
    ag = [s for s in res["splits"] if s["pair"] == "A-G"]
    kA = res["arms"].get("A", {}).get("mean_kl_vs_D")
    kG = res["arms"].get("G", {}).get("mean_kl_vs_D")
    zg = all(greedy["Z"].get(p) == greedy["G"].get(p) for p in greedy["G"])
    zt = all(x["top"][0][0] == y["top"][0][0] for traj in ("D", "A") for p, sg in steps.get("G", {}).get(traj, {}).items()
             for x, y in zip(sg, steps["Z"][traj].get(p, [])))
    gaps = [s.get("gap_D") for s in ag]
    crit = {
        "N1": all(g is not None and g < 0.3 for g in gaps),
        "N2": sum(s.get("G_is_D_top1", False) for s in ag) >= sum(s.get("A_is_D_top1", False) for s in ag),
        "N3": kA is not None and kG is not None and kG <= kA,
        "N4": zg and zt,
    }
    defect = (any(s.get("gap_D") is not None and s["gap_D"] >= 1.0 and not s.get("G_is_D_top1", False) for s in ag)
              or (kA is not None and kG is not None and kG > 2 * kA) or not crit["N4"])
    res["criteria"] = crit
    res["A_G_splits"] = len(ag)
    res["verdict"] = "defect" if defect else ("near-tie" if all(crit.values()) else "inconclusive")
    if not ag and res["verdict"] == "near-tie":
        res["verdict_note"] = "no split on the proxy"
    json.dump(res, open(os.path.join(d, "verdict.json"), "w"), indent=1)
    for arm, v in res["arms"].items():
        print(arm, "vs D: mean_kl %.5f max_kl %.4f top1 %.4f greedy_equal %d/%d" % (
            v["mean_kl_vs_D"], v["max_kl_vs_D"], v["top1_vs_D"], v["greedy_equal_D"], len(v["greedy_first_divergence_vs_D"])))
    for s in res["splits"]:
        print(json.dumps(s))
    print("criteria", crit, "verdict", res["verdict"], res.get("verdict_note", ""))


def logp(step):
    return {t: v - step["logsumexp"] for t, v in step["top"]}


def kl(r, e):
    """KL(r || e) over r's kept tokens that e also kept, exact log-probabilities (both rows normalised over the whole
    vocabulary by their logsumexp), and the mass of r those tokens hold."""
    lr, le = logp(r), logp(e)
    common = [t for t in lr if t in le]
    k = sum(math.exp(lr[t]) * (lr[t] - le[t]) for t in common)
    return k, sum(math.exp(lr[t]) for t in common)


def traced(d, tag):
    r = json.load(open(os.path.join(d, "traced.json")))
    ev = dict(split_arms(r["events"])).get(tag, [])
    return {e["prompt"]: e["steps"] for e in ev if e.get("event") == "traced"}, r["fallbacks"]


def cmd_judge(a):
    d = a.dir
    A, _ = traced(d, "A")
    B, fb = traced(d, "B")
    D, _ = traced(d, "D")
    kls, klda, kldb, top1, misses, cover = [], [], [], 0, [], []
    n = 0
    for p in sorted(A):
        for sa, sb, sd in zip(A[p], B.get(p, []), D.get(p, [])):
            k, c = kl(sa, sb)
            kls.append(k)
            cover.append(c)
            klda.append(kl(sd, sa)[0])
            kldb.append(kl(sd, sb)[0])
            n += 1
            ta, tb = sa["top"][0][0], sb["top"][0][0]
            if ta == tb:
                top1 += 1
            else:
                misses.append({"prompt": p, "position": sa["position"], "margin_A": sa["top"][0][1] - sa["top"][1][1]})
    gBr = json.load(open(os.path.join(d, "greedy.json")))
    first = {}
    for arm, ev in split_arms(gBr["events"]):
        first.setdefault(arm, ev)
    gA = {e["prompt"]: e["ids"] for e in first.get("A", []) if e.get("event") == "generated"}
    gB = {e["prompt"]: e["ids"] for e in first.get("B", []) if e.get("event") == "generated"}
    div = []
    for p in sorted(gA):
        a_, b_ = gA[p], gB.get(p, [])
        i = next((j for j in range(min(len(a_), len(b_))) if a_[j] != b_[j]), None)
        if i is None and len(a_) != len(b_):
            i = min(len(a_), len(b_))
        div.append({"prompt": p, "first_divergence": i})
    speed = json.load(open(os.path.join(d, "speed.json")))
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
    sa = med([s["first_token_s"] for s in speed if s["arm"] == "A" and s["first_token_s"]])
    sb = med([s["first_token_s"] for s in speed if s["arm"] == "B" and s["first_token_s"]])
    ptok = next((s["prompt_tokens"] for s in speed if s["prompt_tokens"]), 0)
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    res = {
        "positions": n,
        "F1": {"mean_kl_A_B": mean(kls), "max_kl_A_B": max(kls) if kls else None, "min_cover": min(cover) if cover else None,
               "pass": n > 0 and mean(kls) <= 0.002 and max(kls) <= 0.02},
        "F2": {"top1_agree": top1 / n if n else None, "misses": misses,
               "pass": n > 0 and top1 / n >= 0.98 and all(m["margin_A"] < 0.3 for m in misses)},
        "F3": {"mean_kl_D_A": mean(klda), "mean_kl_D_B": mean(kldb),
               "pass": n > 0 and mean(kldb) <= 1.25 * mean(klda) + 0.0002},
        "F4": {"divergences": div, "equal_prompts": sum(x["first_divergence"] is None for x in div),
               "pass": sum(x["first_divergence"] is None for x in div) >= 7},
        "F5": {"fallbacks_traced_process": fb, "fallbacks_greedy_process": gBr["fallbacks"]},
        "T1": {"prompt_tokens": ptok, "A_s": sa, "B_s": sb, "A_tok_s": ptok / sa if sa else None,
               "B_tok_s": ptok / sb if sb else None, "ratio": sa / sb if sa and sb else None,
               "pass": bool(sa and sb and sa / sb >= 3.0), "runs": speed},
    }
    json.dump(res, open(os.path.join(d, "verdict.json"), "w"), indent=1)
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in res.items()}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out")
    j = sub.add_parser("judge")
    j.add_argument("dir")
    rr = sub.add_parser("ref")
    rr.add_argument("--out", required=True)
    rr.add_argument("--arms", default="A,B,D")
    rr.add_argument("--ref", default="D")
    jr = sub.add_parser("judge-ref")
    jr.add_argument("dir")
    jr.add_argument("--ref", default="D")
    g4 = sub.add_parser("g4")
    g4.add_argument("--out", required=True)
    g4.add_argument("--tokens", type=int, default=8192)
    g4.add_argument("--batches", default="4096,8192")
    g4.add_argument("--arms", default="A,F,G,X")
    at = sub.add_parser("attn")
    at.add_argument("--out", required=True)
    bb = sub.add_parser("bulk")
    bb.add_argument("--out", required=True)
    jb = sub.add_parser("judge-bulk")
    jb.add_argument("dir")
    sh = sub.add_parser("short")
    sh.add_argument("--out", required=True)
    js = sub.add_parser("judge-short")
    js.add_argument("dir")
    a = ap.parse_args()
    {"short": cmd_short, "judge-short": cmd_judge_short, "g4": cmd_g4, "attn": cmd_attn, "bulk": cmd_bulk, "judge-bulk": cmd_judge_bulk, "run": cmd_run, "judge": cmd_judge, "ref": cmd_ref, "judge-ref": cmd_judge_ref}[a.cmd](a)


if __name__ == "__main__":
    main()
