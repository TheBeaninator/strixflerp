#!/usr/bin/env python3
"""Executor ABI conformance gate (SPEC.md section 6, phase 3): `flerp gate recipe <recipe>-abi`.

Stages, for an ABI variant recipe whose executor loads on this host's GPU (strixite on control-node):
  check   flerp check passes for the catalogue
  build   the recipe assembles and its two binaries build: the conformance driver linked with the ABI adapter,
          and the origin's native server from the same assembly
  gpu     four loads, one at a time. The native server answers the configured greedy requests with MTP drafting
          on, then again with it off; its capture files (--capture-dir) give each request's prompt ids, the tokens
          that entered the sequence, and one {drafted, accepted, emitted} record a decode step. Then the
          conformance driver, opened once with the executor's MTP head on and once with it off, runs the engine's
          decode loop through include/flerp/executor.h only: the prompt cut where the engine cuts it, the engine's
          draft rule, the ids the engine fed (the thinking budget's stop text) fed where it fed them.

Pass criteria. strixite's logits depend on how many rows a forward carries (docs/EXECUTOR-ABI.md, "Forward shape";
SPEC.md section 1, fact 3), so every required comparison is between runs that make the same forwards:
  - the driver's loop reproduces the native capture of the same MTP setting, every token and every step record
    (equal step records mean equal forwards: the same rows a verify, the same rows kept);
  - the loop with every fourth verify dropped and run again returns the same rows the second time (candidate ids
    and logits bit for bit) and ends with the same tokens and steps;
  - a snapshot saved, restored, exported and imported decodes as the same forwards do without it;
  - no native request resumed from an earlier one's state (each is a full prefill, as the driver's is).
Recorded and not required, because the forwards differ: where the native server with MTP on parts from itself with
MTP off, and where the driver's one-row path and its fixed-depth draft path part from each other.

One model is loaded at a time. The stage starts only when the GPU is free and no job of the pod waits for it
(`pod-gpu --wait`); it holds the control-gpu lease while a model is loaded, and between loads it gives the lease up
when a pod job has started waiting and takes it again when none waits. The lease and server helpers are the phase 1
gate's (gate/strixite_recipe.py). The StrixFold adapter's conformance needs the worker pair and runs from
gate/strixfold_recipe.py's pair window, with the same driver.
"""
import argparse
import datetime
import json
import os
import re
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gate"))
import strixite_recipe as p1                     # noqa: E402  (lease, GPU check, server, capture reader)
import summary as SM                             # noqa: E402  (summary.json / summary.md, config_sha256)
from flerplib.catalog import Catalog             # noqa: E402

STAGES = ("check", "build", "gpu")


def stage_check(ctx):
    log = os.path.join(ctx["res"], "check.txt")
    open(log, "w").close()
    rc, _ = p1.sh([sys.executable, "flerp", "check"], log)
    return {"pass": rc == 0, "evidence": "check.txt", "last_line": p1.tail(log, 1)}


def stage_build(ctx):
    log = os.path.join(ctx["res"], "build.txt")
    open(log, "w").close()
    tree = os.path.join(ROOT, "build", ctx["recipe"].name)
    out = {"tree": os.path.relpath(tree, ROOT), "evidence": "build.txt", "started": now()}
    nice = ["nice", "-n", "19"]
    rc, _ = p1.sh([sys.executable, "flerp", "assemble", ctx["recipe"].name], log)
    if rc == 0:
        rc, out["configure_s"] = p1.sh(nice + ["cmake", "-G", "Ninja", "-S", tree, "-B", ctx["ninja"]], log, timeout=600)
    if rc == 0:
        rc, out["build_s"] = p1.sh(nice + ["ninja", "-C", ctx["ninja"], "-j", str(ctx["cfg"]["host"].get("build_jobs", 16))]
                                   + list(ctx["recipe"].binaries), log, timeout=3600)
    bins = {}
    for name in ctx["recipe"].binaries:
        p = os.path.join(ctx["ninja"], name)
        bins[name] = {"sha256": p1.sha256(p), "bytes": os.path.getsize(p)} if os.path.isfile(p) else None
    out.update(binaries=bins, rc=rc, finished=now(), **{"pass": rc == 0 and all(bins.values())})
    return out


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def wait_gpu_free(cfg, seconds):
    t0 = time.time()
    while True:
        ok, state = p1.gpu_free(cfg)
        if ok or time.time() - t0 > seconds:
            return ok, state
        time.sleep(5)


def abi_requests(cfg):
    """The configured requests: a phase 1 request by name and its max_tokens. A list of (name, path, body)."""
    known = {name: (path, body) for name, path, body in p1.requests(cfg)}
    out = []
    for r in cfg["abi"]["request"]:
        path, body = known[r["prompt"]]
        out.append((r["name"], path, dict(body, max_tokens=r["max_tokens"])))
    return out


def native_run(ctx, mtp):
    """One load of the native server: the record of the run and {request name: capture}."""
    cfg, h, run, abi = ctx["cfg"], ctx["cfg"]["host"], ctx["cfg"]["run"], ctx["cfg"]["abi"]
    reqs = abi_requests(cfg)
    scratch = os.path.join(ctx["scratch"], f"native-mtp-{mtp}")
    cap = os.path.join(scratch, "capture")
    subprocess.run(["rm", "-rf", scratch])
    os.makedirs(cap)
    binary = os.path.join(ctx["ninja"], abi["server"])
    args = (["--weights", h["weights"], "--ngram", h["ngram"], "--tokenizer", h["tokenizer"],
             "--generation-config", h["generation_config"], "--host", "127.0.0.1", "--port", str(h["port"]),
             "--mtp", mtp, "--capture-dir", cap] + [str(a) for a in cfg["server"]["args"]] + [str(a) for a in abi["native_args"]])
    out = {"mtp": mtp, "binary": binary, "binary_sha256": p1.sha256(binary), "args": args, "requests": []}
    srv = p1.Server(binary, args, h["port"], os.path.join(scratch, "server.log"))
    try:
        out["startup_s"] = srv.wait_ready(run["startup_timeout_s"])
        for name, path, body in reqs:
            r = srv.post(path, body, run["request_timeout_s"])
            usage = r.get("usage", {})
            out["requests"].append({"name": name, "max_tokens": body["max_tokens"], "finish_reason": r["choices"][0].get("finish_reason"),
                                    "usage": usage,
                                    "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens")})
    finally:
        out["exit_code"] = srv.stop()
        out["server_log_tail"] = p1.tail(srv.log_path, 4)
    with open(srv.log_path, errors="replace") as f:
        log = f.read()
    with open(os.path.join(ctx["res"], f"native-mtp-{mtp}-server.log"), "w") as f:
        f.write(log)
    # what the driver must match, as the server itself logged it
    facts = {}
    for key, rx in (("n_valid", r"startup: tokenizer \S+ \((\d+) tokens\)"), ("mtp_draft", r"startup: config mtp-draft = (\d+)"),
                    ("mtp_margin", r"startup: config mtp-margin = ([0-9.]+)"), ("chunk", r"startup: config chunk = (\d+)"),
                    ("capacity", r"startup: config capacity = (\d+)"), ("total_slots", r"startup: config total-slots = (\d+)"),
                    ("rope_yarn_factor", r"startup: config rope-yarn-factor = ([0-9.]+)"), ("kv_cache", r"startup: config kv-cache = (\S+)"),
                    ("gdn_precision", r"startup: config gdn-precision = (\S+)"), ("attention_precision", r"startup: config attention-precision = (\S+)"),
                    ("mtp_vocab", r"startup: config mtp-vocab = (\d+)"), ("ngram_cache_rows", r"startup: config ngram-cache-rows = (\d+)"),
                    ("kv_generations", r"startup: config kv-generations = (\d+)")):
        m = re.search(rx, log)
        if not m:
            raise RuntimeError(f"native server log: no line for {key}")
        facts[key] = m.group(1)
    out["facts"] = facts
    caps = sorted((p1.read_capture(os.path.join(cap, f)) for f in os.listdir(cap) if f.endswith(".cap")),
                  key=lambda c: c["id"])
    if len(caps) != len(reqs):
        raise RuntimeError(f"native server: {len(caps)} capture files for {len(reqs)} requests")
    native = {name: c for (name, _, _), c in zip(reqs, caps)}
    with open(os.path.join(ctx["res"], f"native-mtp-{mtp}.json"), "w") as f:
        json.dump({"run": out, "captures": native}, f, indent=1)
        f.write("\n")
    return out, native


def fed_steps(cap):
    """The ids the engine fed instead of sampling (step records with drafted = 255, serve/capture.hpp):
    [(tokens in the sequence before them, ids)]."""
    out, at = [], 0
    for drafted, _, emitted in cap["steps"]:
        if drafted == 255:
            out.append((at, cap["generated"][at:at + emitted]))
        at += emitted
    return out


def driver_run(ctx, mtp, native, facts, tokens):
    """One load of the executor behind the ABI: every request as a case of one driver process."""
    cfg, h, abi = ctx["cfg"], ctx["cfg"]["host"], ctx["cfg"]["abi"]
    res = os.path.join(ctx["res"], f"conformance-mtp-{mtp}.json")
    log = os.path.join(ctx["res"], f"conformance-mtp-{mtp}.txt")
    cmd = [os.path.join(ctx["ninja"], abi["conformance"]), "--out", res, "--delivery", "candidates",
           f"weights={h['weights']}", f"ngram={h['ngram']}", f"mtp={mtp}"]
    # the executor is opened with the settings the native server logged, key for key
    for key, fact in (("capacity", "capacity"), ("chunk", "chunk"), ("seqs", "total_slots"), ("rope_yarn_factor", "rope_yarn_factor"),
                      ("kv_cache", "kv_cache"), ("gdn_precision", "gdn_precision"), ("attention_precision", "attention_precision"),
                      ("mtp_vocab", "mtp_vocab"), ("ngram_cache_rows", "ngram_cache_rows"), ("kv_generations", "kv_generations")):
        cmd.append(f"{key}={facts[fact]}")
    meta = {}
    for i, r in enumerate(abi["request"]):
        cap = native[r["name"]]
        P = len(cap["prompt"])
        ids = os.path.join(ctx["scratch"], f"{r['name']}.ids")
        with open(ids, "w") as f:
            f.write(" ".join(str(t) for t in cap["prompt"]) + "\n")
        # Engine::run: a prompt forward ends at the prompt's last <|im_start|> (the turn point, where it saves its
        # turn snapshot); a prompt without one has its turn point at its end
        turn = next((k for k in range(P - 1, 0, -1) if cap["prompt"][k] == tokens["im_start"]), P)
        paths = ["loop-drop" if mtp == "on" else "loop"]
        if r.get("snapshot"):
            paths.append("snapshot")
        if r.get("observe_cross_shape"):
            paths += ["plain", "speculative"]
        cmd += ["--case", r["name"], "--ids", ids, "--paths", ",".join(paths), "--max-tokens", str(r["max_tokens"]),
                "--mtp-draft", facts["mtp_draft"], "--mtp-margin", facts["mtp_margin"], "--n-valid", facts["n_valid"],
                "--end-ids", f"{tokens['im_end']},{tokens['endoftext']}", "--turn", str(turn), "--cross-shape", "observe",
                "--new", str(abi["observe_new"]), "--draft", str(abi["observe_draft"]), "--snap-new", str(abi["snap_new"])]
        if 0 < turn < P:
            cmd += ["--cut", str(turn)]
        for at, fed in fed_steps(cap):
            cmd += ["--fed", f"{at}:{','.join(str(t) for t in fed)}"]
        meta[r["name"]] = {"turn": turn, "paths": paths, "fed": [[at, len(fed)] for at, fed in fed_steps(cap)]}
    open(log, "w").close()
    t0 = time.time()
    rc, _ = p1.sh(cmd, log, timeout=abi["timeout_s"])
    out = {"mtp": mtp, "rc": rc, "seconds": round(time.time() - t0, 1), "evidence": os.path.basename(res), "cases": []}
    try:
        with open(res) as f:
            j = json.load(f)
    except (OSError, ValueError) as e:
        out["error"] = f"no readable result: {e}"
        return out
    out.update({k: j.get(k) for k in ("describe", "caps", "delivery", "stats", "failures", "error")})
    out["executor_checks"] = {k[6:]: v for k, v in j.items() if k.startswith("check_")}
    for c in j.get("cases", []):
        cap = native[c["name"]]
        rec = {"name": c["name"], "mtp": mtp, "prompt_tokens": len(cap["prompt"]), "native_tokens": len(cap["generated"]),
               "native_steps": len(cap["steps"]), "native_finish": cap["finish"], **meta[c["name"]],
               "checks": {k[6:]: v for k, v in c.items() if k.startswith("check_")},
               "observations": {k[8:]: v for k, v in c.items() if k.startswith("observe_")},
               "error": c.get("error")}
        loop, steps = c.get("loop"), c.get("loop_steps")
        if loop is not None:
            rec["loop_tokens"], rec["loop_steps"] = len(loop), len(steps)
            rec["tokens_equal_native"] = loop == cap["generated"]
            rec["steps_equal_native"] = steps == cap["steps"]
            rec["first_token_difference"] = next((i for i, (x, y) in enumerate(zip(loop, cap["generated"])) if x != y),
                                                 None if len(loop) == len(cap["generated"]) else min(len(loop), len(cap["generated"])))
            rec["first_step_difference"] = next((i for i, (x, y) in enumerate(zip(steps, cap["steps"])) if x != y),
                                                None if len(steps) == len(cap["steps"]) else min(len(steps), len(cap["steps"])))
            rec["verifies"], rec["drafted"], rec["accepted"] = c.get("loop_verifies"), c.get("loop_drafted"), c.get("loop_accepted")
            rec["kept_prefix_verifies"] = sum(1 for d, a, _ in steps if 0 < d < 255 and a < d)
            rec["dropped_verifies"] = c.get("loop_drop_dropped_verifies")
            rec["rows_a_verify"] = sorted({d + 1 for d, _, _ in steps if 0 < d < 255})
        for k in ("speculative_first_difference", "loop_first_difference_from_plain"):
            if k in c:
                rec["observations"][k] = c[k]
        rec["pass"] = (not rec["error"] and rec.get("tokens_equal_native") is True and rec.get("steps_equal_native") is True
                       and bool(rec["checks"]) and all(rec["checks"].values()))
        out["cases"].append(rec)
    return out


def stage_gpu(ctx):
    cfg, h, abi = ctx["cfg"], ctx["cfg"]["host"], ctx["cfg"]["abi"]
    out = {"host": socket.gethostname(), "started": now(), "native": [], "executor": []}
    waiters = p1.pod_waiters()
    ok, state = p1.gpu_free(cfg)
    out["gpu_before"] = dict(state, pod_gpu_waiters=len(waiters))
    if not ok or waiters:
        out.update({"pass": False, "blocked": "the GPU is not free (" + (f"{len(waiters)} pod-gpu processes wait for it" if waiters
                    else "another engine is loaded here") + "): run this stage later"})
        return out
    with open(h["tokenizer"]) as f:
        added = {t.get("content"): t["id"] for t in json.load(f).get("added_tokens", [])}
    tokens = {"im_start": added["<|im_start|>"], "im_end": added["<|im_end|>"], "endoftext": added["<|endoftext|>"]}
    out["tokens"] = tokens
    hold = p1.Hold(cfg, os.path.join(ctx["res"], "lease.txt"), out, abi.get("yield_wait_s", 1800))
    label = f"strixflerp executor ABI conformance ({ctx['recipe'].name})"
    native, facts = {}, {}
    try:
        for mtp in abi["mtp"]:
            hold.take(f"{label}: native server, MTP {mtp}", f"native server, MTP {mtp}")
            rec, native[mtp] = native_run(ctx, mtp)
            out["native"].append(rec)
            facts[mtp] = rec["facts"]
            ok, state = wait_gpu_free(cfg, 120)
            hold.give()
            if not ok:
                raise RuntimeError(f"the GPU did not return to idle after the native server stopped: {state}")
        for mtp in abi["mtp"]:
            hold.take(f"{label}: conformance driver, executor MTP {mtp}", f"conformance driver, executor MTP {mtp}")
            out["executor"].append(driver_run(ctx, mtp, native[mtp], facts[mtp], tokens))
            ok, state = wait_gpu_free(cfg, 120)
            hold.give()
            if not ok:
                raise RuntimeError(f"the GPU did not return to idle after the conformance run: {state}")
    except Exception as e:                       # the stage fails with its reason; the lease is still released
        out["error"] = str(e)
    finally:
        hold.give()
    out["gpu_after"] = dict(p1.gpu_free(cfg)[1], pod_gpu_waiters=len(p1.pod_waiters()))
    out["finished"] = now()
    out["held_seconds"] = round(sum(x["seconds"] for x in out["holds"]), 1)
    # the origin against itself: the same requests with MTP on and off make different forwards
    if set(abi["mtp"]) == {"on", "off"} and len(native) == 2:
        out["native_mtp_on_vs_off"] = []
        for r in abi["request"]:
            a, b = native["on"][r["name"]]["generated"], native["off"][r["name"]]["generated"]
            n = min(len(a), len(b))
            first = next((i for i in range(n) if a[i] != b[i]), None)
            out["native_mtp_on_vs_off"].append({"name": r["name"], "tokens": [len(a), len(b)], "first_difference": first,
                                                "rows_a_verify_mtp_on": sorted({d + 1 for d, _, _ in native["on"][r["name"]]["steps"] if 0 < d < 255})})
    resumed = [f"{n['mtp']}/{r['name']}" for n in out["native"] for r in n["requests"] if r["cached_tokens"] != 0]
    out["native_requests_resumed"] = resumed
    cases = [c for e in out["executor"] for c in e["cases"]]
    out["pass"] = ("error" not in out and not resumed and len(out["executor"]) == len(abi["mtp"])
                   and all(e["rc"] == 0 and not e.get("error") and all(e.get("executor_checks", {"x": False}).values()) for e in out["executor"])
                   and len(cases) == len(abi["mtp"]) * len(abi["request"]) and all(c["pass"] for c in cases))
    return out


def write_summary(ctx, results):
    ok = SM.all_pass(STAGES, results)
    SM.write_json(ctx["res"], {"recipe": ctx["recipe"].name, "host": socket.gethostname(), "date": now(), "pass": ok,
                               "stages": results}, ctx=ctx)
    lines = [f"# Executor ABI conformance: {ctx['recipe'].name}", "",
             f"Host {socket.gethostname()}, {now()}. Stages run: {', '.join(results)}.", ""]
    lines += SM.stage_bullets(STAGES, results)
    g = results.get("gpu", {})
    cases = [c for e in g.get("executor", []) for c in e["cases"]]
    if cases:
        lines += ["", "Required: the engine's decode loop, run through `include/flerp/executor.h`, against the native server's capture "
                  "of the same request with the same MTP setting. Equal step records are equal forwards.", "",
                  "| request | MTP | prompt tokens | tokens native / ABI | tokens equal | steps native / ABI | steps equal | rows a verify | "
                  "verifies | drafts accepted / made | verifies kept as a prefix | verifies dropped and run again | driver checks | pass |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in cases:
            bad = [k for k, v in c["checks"].items() if not v]
            lines.append(f"| {c['name']} | {c['mtp']} | {c['prompt_tokens']} | {c['native_tokens']} / {c.get('loop_tokens')} | "
                         f"{'yes' if c.get('tokens_equal_native') else 'NO, from token ' + str(c.get('first_token_difference'))} | "
                         f"{c['native_steps']} / {c.get('loop_steps')} | "
                         f"{'yes' if c.get('steps_equal_native') else 'NO, from step ' + str(c.get('first_step_difference'))} | "
                         f"{', '.join(str(x) for x in c.get('rows_a_verify') or []) or '-'} | {c.get('verifies')} | "
                         f"{c.get('accepted')} / {c.get('drafted')} | {c.get('kept_prefix_verifies')} | "
                         f"{c.get('dropped_verifies') if c.get('dropped_verifies') is not None else '-'} | "
                         f"{len(c['checks']) - len(bad)} of {len(c['checks'])}{' (failed: ' + ', '.join(bad) + ')' if bad else ''} | "
                         f"{'yes' if c['pass'] else 'NO'} |")
        fed = sorted({(c["name"], at, n) for c in cases for at, n in c["fed"]})
        if fed:
            lines += ["", "Ids the engine fed instead of sampling (capture steps with drafted = 255; the driver feeds the same ids at the "
                      "same place): " + "; ".join(f"{name}: {n} ids after {at} generated tokens" for name, at, n in fed) + "."]
        if g.get("native_requests_resumed"):
            lines += ["", "Native requests that resumed from an earlier request's state (not a full prefill): "
                      + ", ".join(g["native_requests_resumed"]) + "."]
        if g.get("native_mtp_on_vs_off"):
            lines += ["", "Not required - the forwards differ. The native server against itself, MTP on (several rows a verify) and "
                      "off (one row a forward):", "", "| request | tokens on / off | first token that differs | rows a verify with MTP on |",
                      "|---|---|---|---|"]
            for r in g["native_mtp_on_vs_off"]:
                lines.append(f"| {r['name']} | {r['tokens'][0]} / {r['tokens'][1]} | "
                             f"{r['first_difference'] if r['first_difference'] is not None else 'none in the common length'} | "
                             f"{', '.join(str(x) for x in r['rows_a_verify_mtp_on']) or '-'} |")
        obs = [(c["name"], c["mtp"], k, v) for c in cases for k, v in c["observations"].items()]
        if obs:
            lines += ["", "Not required - the forwards differ. The driver's own paths against each other: "
                      + "; ".join(f"{name} (MTP {mtp}) {k} = {v}" for name, mtp, k, v in obs) + "."]
        lines += ["", f"GPU: the control-gpu lease was held {g.get('held_seconds')} s in {len(g.get('holds', []))} holds ("
                  + ", ".join(f"{x['for']}: {x['seconds']} s" for x in g.get("holds", [])) + "); "
                  + (f"waited for the pod's own GPU jobs {len(g['yields'])} times ("
                     + ", ".join(f"{y['waited_s']} s before {y['before']}" for y in g["yields"]) + ")." if g.get("yields")
                     else "no pod job was waiting when a hold began."),
                  "", "The conformance driver calls the executor only through `include/flerp/executor.h`. Native tokens and step "
                  "records are read from the server's capture files (`--capture-dir`)."]
    SM.write_md(ctx["res"], lines)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--stages", default=",".join(STAGES))
    a = ap.parse_args()
    cat = Catalog(ROOT)
    recipe = cat.recipe(a.recipe)
    if recipe is None:
        sys.exit(f"no recipe {a.recipe}")
    cfg = p1.load_config(os.path.join(ROOT, recipe.gate["config"]))
    res = os.path.join(ROOT, "gate", "results", recipe.name)
    os.makedirs(res, exist_ok=True)
    ctx = {"cat": cat, "recipe": recipe, "cfg": cfg, "res": res,
           "scratch": os.path.join(ROOT, cfg["host"].get("scratch", "build/gate-scratch-abi")),
           "ninja": os.path.join(ROOT, "build", recipe.name, "ninja")}
    os.makedirs(ctx["scratch"], exist_ok=True)
    results = {}
    prior = os.path.join(res, "summary.json")
    if os.path.isfile(prior) and set(a.stages.split(",")) != set(STAGES):
        with open(prior) as f:
            results = json.load(f).get("stages", {})
    fns = {"check": stage_check, "build": stage_build, "gpu": stage_gpu}
    for s in STAGES:
        if s not in a.stages.split(","):
            continue
        print(f"== {s}", flush=True)
        results[s] = fns[s](ctx)
        print(f"   {'PASS' if results[s].get('pass') else 'FAIL'} {results[s].get('blocked', '')}{results[s].get('error', '')}", flush=True)
        if not results[s].get("pass"):
            break
    return 0 if write_summary(ctx, {s: results[s] for s in STAGES if s in results}) else 1


if __name__ == "__main__":
    sys.exit(main())
