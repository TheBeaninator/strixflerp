#!/usr/bin/env python3
"""Prefill gate for recipes/qwen38-flash-next.strixite-pf.toml, on control-node's GPU.

    gate/strixite_prefill.py <config.toml> [--stages bench,trace,server,report]

Stages (each takes the GPU lease only while a model is loaded, and gives it up between loads when a pod job waits):
  bench   strixflerp-prefill-bench: every [[variant]] (a chunk size and module switches) on the same fixed token-id
          prompts, `rounds` rounds alternating the variants' order, one weights load. Round 0 is the first request
          (the n-gram row cache cold for the first variant); rounds 1.. are the second-request numbers.
  trace   the same tool, one round, the trace prompt only, under rocprofv3 --kernel-trace: per chunk, the GPU's busy
          time (union of kernel intervals inside the chunk's forward call) against the call's wall time.
  server  the recipe's server per variant (the variant's --chunk and environment), MTP on, the phase-1 recipe
          gate's three requests x repeats: decode tok/s and generated ids against the base variant.
  report  results.json + summary.md under gate/results/<config name>/ with every criterion's verdict.

Criteria live in the config's [criteria] and per-variant `expect`; they are committed before any GPU run.
"""
import argparse
import csv
import glob
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strixite_recipe as sr  # noqa: E402
import summary as SM  # noqa: E402  (results.json / summary.md, config_sha256)

ROOT = sr.ROOT


def bench_cmd(cfg, binary, out_dir, variants, prompts, rounds, save_logits):
    h, b = cfg["host"], cfg["bench"]
    vs = []
    for v in variants:
        parts = [v["name"], f"chunk={v['chunk']}"] + [f"{k}={val}" for k, val in v.get("env", {}).items()]
        vs.append(":".join(parts))
    return ["ionice", "-c3", "nice", "-n", "19", binary, "--weights", h["weights"], "--ngram", h["ngram"],
            "--out", out_dir, "--prompts", ",".join(str(p) for p in prompts), "--variants", ",".join(vs),
            "--rounds", str(rounds), "--gen", str(b["gen"]), "--seed", str(b["seed"]),
            "--capacity", str(b["capacity"]), "--logits-rows", str(b["logits_rows"]),
            "--save-logits", "1" if save_logits else "0"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def stage_bench(cfg, ctx, hold):
    out_dir = os.path.join(ctx["scratch"], "bench")
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "*")):
        os.remove(f)
    hold.take(f"strixflerp prefill bench {ctx['name']}", "bench")
    try:
        cmd = bench_cmd(cfg, ctx["bench_bin"], out_dir, cfg["variant"], cfg["bench"]["prompts"],
                        cfg["bench"]["rounds"], True)
        rc, secs = sr.sh(cmd, os.path.join(ctx["scratch"], "bench.log"), timeout=cfg["bench"]["timeout_s"])
    finally:
        hold.give()
    if rc != 0:
        raise RuntimeError(f"bench exited {rc}; see {ctx['scratch']}/bench.log")
    return {"rc": rc, "seconds": secs, "rows": read_jsonl(os.path.join(out_dir, "results.jsonl"))}


def busy_union(intervals, lo, hi):
    """Total time inside [lo, hi] covered by at least one interval."""
    segs = sorted((max(a, lo), min(b, hi)) for a, b in intervals if b > lo and a < hi)
    total, cur_a, cur_b = 0, None, None
    for a, b in segs:
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        total += cur_b - cur_a
    return total


def stage_trace(cfg, ctx, hold):
    t = cfg["trace"]
    out = {}
    for v in cfg["variant"]:
        if v["name"] not in t["variants"]:
            continue
        d = os.path.join(ctx["scratch"], "trace-" + v["name"])
        os.makedirs(d, exist_ok=True)
        for f in glob.glob(os.path.join(d, "**", "*"), recursive=True):
            if os.path.isfile(f):
                os.remove(f)
        hold.take(f"strixflerp prefill trace {ctx['name']} {v['name']}", "trace " + v["name"])
        try:
            cmd = bench_cmd(cfg, ctx["bench_bin"], d, [v], [t["prompt"]], 2, False)
            cmd = cmd[:5] + ["rocprofv3", "--kernel-trace", "--output-format", "csv", "-d", os.path.join(d, "prof"),
                             "-o", "run", "--"] + cmd[5:]
            rc, secs = sr.sh(cmd, os.path.join(d, "trace.log"), timeout=t["timeout_s"])
        finally:
            hold.give()
        if rc != 0:
            raise RuntimeError(f"trace {v['name']} exited {rc}; see {d}/trace.log")
        rows = read_jsonl(os.path.join(d, "results.jsonl"))
        csvs = glob.glob(os.path.join(d, "prof", "**", "*kernel_trace.csv"), recursive=True)
        if not csvs:
            raise RuntimeError(f"trace {v['name']}: no kernel_trace.csv under {d}/prof")
        kernels, by_name = [], {}
        with open(csvs[0]) as f:
            for k in csv.DictReader(f):
                a, b = int(k["Start_Timestamp"]), int(k["End_Timestamp"])
                kernels.append((a, b, k["Kernel_Name"]))
        iv = [(a, b) for a, b, _ in kernels]
        per = []
        for r in rows:  # the last round is the second request
            if r["round"] != 1:
                continue
            ns = r["chunk_ns"]
            for i in range(0, len(ns), 2):
                lo, hi = ns[i], ns[i + 1]
                busy = busy_union(iv, lo, hi)
                per.append({"chunk": i // 2, "wall_ms": round((hi - lo) / 1e6, 1), "gpu_busy_ms": round(busy / 1e6, 1),
                            "gap_ms": round((hi - lo - busy) / 1e6, 1)})
                if i + 2 < len(ns):  # between two forward calls (host only: the forward ended with a sync)
                    per[-1]["between_calls_ms"] = round((ns[i + 2] - hi) / 1e6, 2)
            lo, hi = ns[0], ns[-1]
            for a, b, name in kernels:
                if a >= lo and b <= hi:
                    short = name.split("(")[0].split("<")[0][:60]
                    by_name[short] = by_name.get(short, 0) + (b - a)
        top = sorted(by_name.items(), key=lambda kv: -kv[1])[:15]
        out[v["name"]] = {"chunks": per, "kernels_ms": [(n, round(t_ / 1e6, 1)) for n, t_ in top],
                          "kernel_count": len(kernels), "csv": os.path.relpath(csvs[0], ROOT)}
    return out


def stage_server(cfg, ctx, hold):
    reqs = sr.requests(cfg)
    out = {}
    for v in cfg["variant"]:
        if v["name"] not in cfg["server_stage"]["variants"]:
            continue
        c = json.loads(json.dumps(cfg))
        args = list(c["server"]["args"])
        args[args.index("--chunk") + 1] = str(v["chunk"])
        c["server"]["args"] = args
        saved = {k: os.environ.get(k) for k in v.get("env", {})}
        os.environ.update(v.get("env", {}))
        hold.take(f"strixflerp prefill server {ctx['name']} {v['name']}", "server " + v["name"])
        try:
            out[v["name"]] = sr.run_server("pf-" + v["name"], ctx["server_bin"], True, c, reqs, ctx["scratch"])
        finally:
            hold.give()
            for k, val in saved.items():
                if val is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = val
    return out


def kl_rows(base, other, vocab):
    p, q = base.reshape(-1, vocab).astype(np.float64), other.reshape(-1, vocab).astype(np.float64)
    lp = p - p.max(1, keepdims=True)
    lp -= np.log(np.exp(lp).sum(1, keepdims=True))
    lq = q - q.max(1, keepdims=True)
    lq -= np.log(np.exp(lq).sum(1, keepdims=True))
    return (np.exp(lp) * (lp - lq)).sum(1)


def med_range(xs):
    return {"median": round(statistics.median(xs), 2), "min": round(min(xs), 2), "max": round(max(xs), 2),
            "n": len(xs), "spread_pct": round(100 * (max(xs) - min(xs)) / statistics.median(xs), 2)}


def report(cfg, ctx, res):
    crit, base = cfg["criteria"], cfg["criteria"]["base"]
    rows = res.get("bench", {}).get("rows", [])
    verdicts, table = [], []
    bdir = os.path.join(ctx["scratch"], "bench")
    for v in cfg["variant"]:
        if v["name"] == base:
            continue
        expect = v.get("expect", "identical")
        for L in cfg["bench"]["prompts"]:
            br = [r for r in rows if r["variant"] == base and r["len"] == L]
            vr = [r for r in rows if r["variant"] == v["name"] and r["len"] == L]
            if not br or not vr:
                continue
            shas_b, shas_v = {r["logits_sha256"] for r in br}, {r["logits_sha256"] for r in vr}
            ids_b, ids_v = {tuple(r["gen_ids"]) for r in br}, {tuple(r["gen_ids"]) for r in vr}
            same_sha = shas_b == shas_v and len(shas_b) == 1
            same_ids = ids_b == ids_v and len(ids_b) == 1
            row = {"variant": v["name"], "len": L, "expect": expect, "base_sha": sorted(shas_b), "sha": sorted(shas_v),
                   "logits_identical": same_sha, "greedy_identical": same_ids}
            lb, lv = os.path.join(bdir, f"logits.{base}.{L}.f32"), os.path.join(bdir, f"logits.{v['name']}.{L}.f32")
            if os.path.exists(lb) and os.path.exists(lv):
                a, b = np.fromfile(lb, np.float32), np.fromfile(lv, np.float32)
                V = br[0]["vocab"]
                kl = kl_rows(a, b, V)
                row.update(kl_mean=float(kl.mean()), kl_max=float(kl.max()),
                           top1_agree=float((a.reshape(-1, V).argmax(1) == b.reshape(-1, V).argmax(1)).mean()),
                           max_abs_diff=float(np.abs(a - b).max()))
            first_diff = None
            gb, gv = list(br[0]["gen_ids"]), list(vr[0]["gen_ids"])
            for i, (x, y) in enumerate(zip(gb, gv)):
                if x != y:
                    first_diff = i
                    break
            row["greedy_first_diff"] = first_diff
            tb = [r["prefill_tps"] for r in br if r["round"] >= 1]
            tv = [r["prefill_tps"] for r in vr if r["round"] >= 1]
            if tb and tv:
                row["prefill_tps_base"], row["prefill_tps"] = med_range(tb), med_range(tv)
                row["gain_pct"] = round(100 * (statistics.median(tv) / statistics.median(tb) - 1), 2)
                row["faster_outside_noise"] = min(tv) > max(tb)
            row["decode_tps_base"] = med_range([r["decode_tps"] for r in br])
            row["decode_tps"] = med_range([r["decode_tps"] for r in vr])
            row["dev_used_session_gib_base"] = round(max(r["dev_used_after"] for r in br) / 2**30, 2)
            row["dev_used_session_gib"] = round(max(r["dev_used_after"] for r in vr) / 2**30, 2)
            row["ple_wait_s_base"] = [r["ple_wait_s"] for r in br]
            row["ple_wait_s"] = [r["ple_wait_s"] for r in vr]
            table.append(row)
            if expect == "identical":
                verdicts.append({"criterion": f"{v['name']} {L}: logits sha256 and greedy ids equal to {base}",
                                 "pass": same_sha and same_ids})
    # speed: per variant, over the prompts of criteria.speed_prompts
    for v in cfg["variant"]:
        if v["name"] == base or "min_gain_pct" not in v:
            continue
        rs = [r for r in table if r["variant"] == v["name"] and r["len"] in crit["speed_prompts"] and "gain_pct" in r]
        ok = bool(rs) and all(r["gain_pct"] >= v["min_gain_pct"] and r["faster_outside_noise"] for r in rs)
        verdicts.append({"criterion": f"{v['name']}: prefill tok/s (second request) at least +{v['min_gain_pct']}% "
                                      f"over {base} at {crit['speed_prompts']}, every repeat above the base's max",
                         "pass": ok})
    srv = res.get("server", {})
    if base in srv:
        for name, s in srv.items():
            if name == base:
                continue
            rows_, speed, identical, speed_ok = sr.compare(srv[base], s, crit["decode_tolerance"])
            verdicts.append({"criterion": f"{name}: server decode (MTP on) tok/s within {crit['decode_tolerance']:.0%} "
                                          f"of {base} or inside its range", "pass": speed_ok})
            verdicts.append({"criterion": f"{name}: server generated ids equal to {base}", "pass": identical,
                             "report_only": name not in [v["name"] for v in cfg["variant"] if v.get("expect", "identical") == "identical"]})
            res.setdefault("server_compare", {})[name] = {"speed": speed, "identical": identical}
    res["table"], res["verdicts"] = table, verdicts
    return res


def write_summary(cfg, ctx, res):
    os.makedirs(ctx["results"], exist_ok=True)
    slim = json.loads(json.dumps(res))
    for s in slim.get("server", {}).values():
        for r in s.get("runs", []):
            r.pop("generated", None)
            r.pop("steps", None)
    SM.write_json(ctx["results"], slim, name="results.json", ctx=ctx)
    L = [f"# Prefill gate: {ctx['name']}", "", f"Host control-node, {sr.utc()}. Config `{ctx['config_rel']}`.", ""]
    for v in res.get("verdicts", []):
        tag = "REPORT" if v.get("report_only") else ("PASS" if v["pass"] else "FAIL")
        L.append(f"- **{tag}** {v['criterion']}" + ("" if not v.get("report_only") else f" ({'yes' if v['pass'] else 'no'})"))
    L += ["", "| variant | prompt | logits = base | greedy = base | top-1 agree | KL mean / max | prefill tok/s base "
               "(med, min-max) | prefill tok/s | gain | decode T=1 base / variant | device GiB base / variant |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res.get("table", []):
        pb, pv = r.get("prefill_tps_base", {}), r.get("prefill_tps", {})
        L.append(f"| {r['variant']} | {r['len']} | {r['logits_identical']} | {r['greedy_identical']} "
                 f"(first diff {r['greedy_first_diff']}) | {r.get('top1_agree', '?')} | "
                 f"{r.get('kl_mean', 0):.3g} / {r.get('kl_max', 0):.3g} | {pb.get('median')} ({pb.get('min')}-{pb.get('max')}) | "
                 f"{pv.get('median')} ({pv.get('min')}-{pv.get('max')}) | {r.get('gain_pct')}% | "
                 f"{r['decode_tps_base']['median']} / {r['decode_tps']['median']} | "
                 f"{r['dev_used_session_gib_base']} / {r['dev_used_session_gib']} |")
    for name, t in res.get("trace", {}).items():
        L += ["", f"Timeline {name} (second request, prompt {cfg['trace']['prompt']}): per chunk wall / GPU busy / gap ms",
              ""] + [f"- chunk {c['chunk']}: {c['wall_ms']} / {c['gpu_busy_ms']} / {c['gap_ms']}"
                     + (f", host between calls {c['between_calls_ms']} ms" if "between_calls_ms" in c else "")
                     for c in t["chunks"]]
        L += ["", "Top kernels (ms over the request): " + ", ".join(f"{n} {ms}" for n, ms in t["kernels_ms"][:10])]
    for name, c in res.get("server_compare", {}).items():
        L += ["", f"Server {name} vs base: identical {c['identical']}; decode " +
              "; ".join(f"{s['prompt']} {s['decode_tps_reference']} -> {s['decode_tps_assembled']}" for s in c["speed"])]
    L += ["", "GPU holds: " + "; ".join(f"{h['for']} {h['seconds']} s" for h in res.get("holds", [])),
          f"Storage read by the bench processes: {res.get('io_read_gb', '?')} GB"]
    SM.write_md(ctx["results"], L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--stages", default="bench,trace,server,report")
    a = ap.parse_args()
    cfg = sr.load_config(a.config)
    name = os.path.splitext(os.path.basename(a.config))[0]
    ctx = {"name": name, "config_rel": os.path.relpath(os.path.abspath(a.config), ROOT),
           "scratch": os.path.join(ROOT, "build", "gate-scratch-" + name),
           "results": os.path.join(ROOT, "gate", "results", name)}
    bdir = os.path.join(ROOT, "build", cfg["build"]["recipe"], "ninja")
    ctx["bench_bin"], ctx["server_bin"] = os.path.join(bdir, "strixflerp-prefill-bench"), os.path.join(bdir, cfg["build"]["server"])
    os.makedirs(ctx["scratch"], exist_ok=True)
    stages = a.stages.split(",")
    prev = os.path.join(ctx["scratch"], "res.json")
    res = json.load(open(prev)) if os.path.exists(prev) and "bench" not in stages else {}
    hold = sr.Hold(cfg, os.path.join(ctx["scratch"], "lease.log"), res)
    if "bench" in stages:
        res["bench"] = stage_bench(cfg, ctx, hold)
        res["io_read_gb"] = round(max(r["io_read_bytes"] for r in res["bench"]["rows"]) / 1e9, 2)
    if "trace" in stages:
        res["trace"] = stage_trace(cfg, ctx, hold)
    if "server" in stages:
        res["server"] = stage_server(cfg, ctx, hold)
    json.dump(res, open(prev, "w"))
    if "report" in stages:
        report(cfg, ctx, res)
        write_summary(cfg, ctx, res)
    print(json.dumps(res.get("verdicts", []), indent=1))


if __name__ == "__main__":
    main()
