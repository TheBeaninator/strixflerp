#!/usr/bin/env python3
"""Gate of the placed loader (format.flerp.gguf_placed) and the block range of executor.flerp.qwen35moe: stage P1 of
docs/PHASE5-FOURNODE-DESIGN.md. Run on the gfx1151 build host, after the recipe is built:

    python3 gate/moe_placed.py [--config gate/qwen36-35b-a3b.flerp-moe-placed.toml] [--stages plan,equal,range35,ornith]

Stages, each writing evidence under gate/results/<recipe>-placed/:
  plan     no GPU, headers only: the C++ plan (binary --plan-only) against gate/gguf_census.py, both files.
  equal    gate (1): the 35B model, world 1, whole, through the placed loader, against the recorded run of the
           binary before this stage; the dense model through the placed loader against the dense recipe's record.
  range35  block ranges of the 35B model as stages (--stage-bench) with the numpy block check: the mechanism of
           gate (2) at shapes that have run before.
  ornith   gate (2): as many leading blocks of the Ornith file as the config's rule allows, as a stage.

Criteria: [criteria] of the config, committed before the first GPU run. Every load is the placed loader's single
direct reader ([host] of the config says why); the bytes each stage read from disk are recorded.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gate"))
import gguf_census as C  # noqa: E402

STAGES = ("plan", "equal", "range35", "ornith")
GIB = float(1 << 30)


# --- judgements (no I/O: tests/test_moe_placed.py) ---------------------------------------------------------------
def compare_traced(mine, recorded):
    """Two `traced` events of one prompt: ids, the number of rows, and every row's token, log-sum-exp and top list.
    Returns {"equal", "ids_equal", "rows": [mine, recorded], "first_row_difference", "why"}."""
    out = {"ids_equal": mine["ids"] == recorded["ids"], "rows": [len(mine["steps"]), len(recorded["steps"])],
           "first_row_difference": None, "why": []}
    if not out["ids_equal"]:
        out["why"].append("generated ids differ")
    if out["rows"][0] != out["rows"][1]:
        out["why"].append(f"{out['rows'][0]} rows against {out['rows'][1]}")
    for i, (a, b) in enumerate(zip(mine["steps"], recorded["steps"])):
        n = len(b["top"])
        same = (a["token"] == b["token"] and a["position"] == b["position"] and a["logsumexp"] == b["logsumexp"]
                and [[int(t), v] for t, v in a["top"][:n]] == [[int(t), v] for t, v in b["top"]])
        if not same:
            out["first_row_difference"] = i
            out["why"].append(f"row {i} differs (token {a['token']} / {b['token']}, log-sum-exp {a['logsumexp']} / {b['logsumexp']})")
            break
    out["equal"] = not out["why"]
    return out


def judge_load(loaded, plan_rank, plan_sha, crit, resident_is_tensor_bytes=True):
    """Criteria 4, 5 and 7 on a `loaded` event against the plan gguf_census.py computed. Returns [failures]."""
    why, p = [], (loaded or {}).get("placed")
    if not p:
        return ["the run did not load through the placed loader"]
    if p["allocated_bytes"] != plan_rank["device_bytes"]:
        why.append(f"allocated {p['allocated_bytes']} bytes, the plan has {plan_rank['device_bytes']}")
    if p["plan_sha256"] != plan_sha:
        why.append("the loader's plan hash is not gguf_census.py's")
    if p["tensors"] != plan_rank["tensors"] or p["tensor_bytes"] != plan_rank["tensor_bytes"]:
        why.append(f"{p['tensors']} tensors / {p['tensor_bytes']} bytes filled, the plan has {plan_rank['tensors']} / {plan_rank['tensor_bytes']}")
    if resident_is_tensor_bytes and loaded.get("weight_resident_bytes") != plan_rank["tensor_bytes"]:
        why.append(f"the executor reports {loaded.get('weight_resident_bytes')} resident weight bytes, the plan's tensors are {plan_rank['tensor_bytes']}")
    grown = p["rss_file_after_fill"] - p["rss_file_before_fill"]
    if grown > crit["rss_file_growth_max"]:
        why.append(f"RssFile grew by {grown} bytes across the fill (bound {crit['rss_file_growth_max']})")
    if p.get("single_reader") and not p.get("direct_io"):
        why.append("the single reader fell back to buffered reads")
    return why


def judge_block_check(check, crit, records_min=None):
    """Criterion 9 on gate/moe_block_check.py's output. Returns {"pass", "why", counts and maxima}."""
    detail = check.get("detail") or []
    need = crit["check_records_min"] if records_min is None else records_min
    why = [check["error"]] if check.get("error") else []
    if len(detail) < need:
        why.append(f"{len(detail)} records, at least {need} wanted")
    ties = [r for r in detail if not r["ids_same_set"] and r["margin_to_next_expert"] < crit["check_tie_margin"]]
    wrong = [r for r in detail if not r["ids_same_set"] and r["margin_to_next_expert"] >= crit["check_tie_margin"]]
    same = [r for r in detail if r["ids_same_set"]]
    for r in wrong[:4]:
        why.append(f"block {r['layer']} position {r['pos']}: another expert set (float64 margin {r['margin_to_next_expert']:.3g})")
    def mx(k):
        vals = [r[k] for r in same if r[k] is not None]
        return None if not vals else (float("nan") if any(v != v for v in vals) else max(vals))
    out = {"records": len(detail), "layers": sorted({r["layer"] for r in detail}), "same_set": len(same), "same_order": sum(r["ids_equal"] for r in detail),
           "ties_excused": len(ties), "other_set": len(wrong), "weights_max_abs": mx("weights_max_abs"), "gate_logit_max_abs": mx("gate_logit_abs"),
           "out_max_abs_over_rms": mx("out_max_abs_over_rms"), "smallest_margin": min((r["margin_to_next_expert"] for r in detail), default=None)}
    for key, bound, label in (("weights_max_abs", "check_weights_max", "router weights"), ("gate_logit_max_abs", "check_gate_max", "the shared gate logit"),
                              ("out_max_abs_over_rms", "check_out_max", "the block output over its RMS")):
        if out[key] is not None and not out[key] <= crit[bound]:      # NaN fails
            why.append(f"{label} differ by up to {out[key]:.3g} (bound {crit[bound]})")
    return dict(out, **{"pass": not why, "why": why})


def judge_bench(events, starts_at_zero):
    """Criterion 8 on the events of a --stage-bench run. Returns [failures]."""
    why, phases = [], {e["phase"]: e for e in events if e.get("event") == "stage_bench"}
    for name in (["tokens"] if starts_at_zero else []) + ["synthetic"]:
        e = phases.get(name)
        if e is None:
            why.append(f"phase {name} did not run")
            continue
        outs = {"chunk": e["chunk_out"], "chunk again": e["chunk_out_again"], "chunk third": e["chunk_out_third"],
                "decode after the chunk": e["decode_out_after_chunk"], "decode from empty": e["decode_fresh"]["out_after_with_logits"]}
        for label, o in outs.items():
            if o["non_finite"] or not o["rms"] > 0:
                why.append(f"phase {name}, {label}: {o['non_finite']} non-finite values, RMS {o['rms']}")
        if not e["chunk_repeats_equal"]:
            why.append(f"phase {name}: the chunk run three times did not give the same bytes")
    return why


def pick_blocks(plan_bytes, free_bytes, reserve_bytes, max_blocks, min_blocks):
    """The most leading blocks n <= max_blocks with plan_bytes[n] + reserve_bytes <= free_bytes (plan_bytes maps a
    block count to its plan's device bytes); None when fewer than min_blocks fit."""
    best = None
    for n in sorted(plan_bytes):
        if n <= max_blocks and plan_bytes[n] + reserve_bytes <= free_bytes:
            best = n
    return best if best is not None and best >= min_blocks else None


# --- host --------------------------------------------------------------------------------------------------------
def load1():
    with open("/proc/loadavg") as f:
        return float(f.read().split()[0])


def wait_for_load(cfg, what, log):
    """The I/O rule: a heavy step starts only when the 1-minute load is at or under [host] load_wait_above."""
    h, t0 = cfg["host"], time.time()
    while load1() > h["load_wait_above"]:
        if time.time() - t0 > h["load_wait_s"]:
            raise RuntimeError(f"the 1-minute load stayed above {h['load_wait_above']} for {h['load_wait_s']} s before {what}")
        time.sleep(20)
    with open(log, "a") as f:
        f.write(f"# {G.utc()} load {load1()} before {what}, waited {round(time.time() - t0)} s\n")
    return round(time.time() - t0)


def memory_now():
    def one(name):
        vals = []
        for card in sorted(os.listdir("/sys/class/drm")):
            p = f"/sys/class/drm/{card}/device/{name}"
            if os.path.isfile(p):
                with open(p) as f:
                    vals.append(int(f.read()))
        return max(vals) if vals else 0
    avail = 0
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) * 1024
    return {"gtt_total": one("mem_info_gtt_total"), "gtt_used": one("mem_info_gtt_used"), "mem_available": avail}


def run_binary(ctx, label, args, log, timeout):
    """One run of the recipe's binary, bound to [host] cpus, under ionice idle and nice."""
    h = ctx["cfg"]["host"]
    wait_for_load(ctx["cfg"], label, log)
    prefix = list(h["polite"]) + ["taskset", "-c", h["cpus"]]
    rc, events, secs = G.generate(G.assembled_binary(ctx), args, log, timeout, G.runtime_env(ctx), prefix)
    loaded = next((e for e in events if e.get("event") == "loaded"), None)
    read = ((loaded or {}).get("placed") or {}).get("tensor_bytes", 0)
    ctx["disk"].append({"run": label, "bytes": read, "seconds": ((loaded or {}).get("placed") or {}).get("fill_s"), "when": G.utc()})
    return {"rc": rc, "seconds": secs, "events": events, "loaded": loaded}


def py_plan(gguf, world=1, rng=None, cache={}):
    key = (gguf, world, tuple(rng) if rng else None)
    if key not in cache:
        if gguf not in cache:
            cache[gguf] = C.read_header(gguf)
        p = C.stage_plan(cache[gguf], world, tuple(rng) if rng else None)
        cache[key] = {"sha256": C.plan_hash(p), "cuts": p["cuts"],
                      "ranks": [{k: r[k] for k in ("rank", "blocks", "tensors", "tensor_bytes", "device_bytes")} for r in p["ranks"]]}
    return cache[key]


def recipe_gate_config(ctx):
    return ctx["recipe"].gate["config"]


def placed_flags(ctx, rng=None):
    return ["--placed"] + list(ctx["cfg"]["host"]["reader_flags"]) + (["--blocks", f"{rng[0]}:{rng[1]}"] if rng else [])


def write_ids(ctx, name, ids):
    d = os.path.join(ctx["scratch"], "prompts")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name + ".ids")
    with open(path, "w") as f:
        f.write(" ".join(str(i) for i in ids))
    return path


def load_json(path):
    with open(os.path.join(ROOT, path)) as f:
        return json.load(f)


def save(ctx, name, doc):
    with open(os.path.join(ctx["res"], name), "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")


def digest_traced(e):
    """A traced event without its megabytes: what the summary and the next comparison need."""
    return {"ids": e["ids"], "rows": len(e["steps"]), "argmax_is_token": e["argmax_is_token"], "first_token_s": e["first_token_s"],
            "total_s": e["total_s"], "logits_sha256": [s.get("sha256") for s in e["steps"]],
            "logsumexp": [s["logsumexp"] for s in e["steps"]]}


# --- stages ------------------------------------------------------------------------------------------------------
def stage_plan(ctx):
    cfg, log = ctx["cfg"], os.path.join(ctx["res"], "plan.txt")
    open(log, "w").close()
    rows, ok = [], True
    env = dict(G.NO_GPU, **G.runtime_env(ctx))
    asks = [(cfg["model"]["gguf"], w, None) for w in cfg["run"]["plan_worlds"]] + [(cfg["model"]["ornith_gguf"], w, None) for w in cfg["run"]["plan_worlds"]]
    asks += [(cfg["model"]["gguf"], 1, r) for r in cfg["run"]["range35_blocks"]]
    asks += [(cfg["model"]["ornith_gguf"], 1, [0, n]) for n in range(cfg["run"]["min_blocks"], cfg["run"]["max_blocks"] + 1)]
    asks += [(cfg["model"]["dense_gguf"], 1, None)]
    for gguf, world, rng in asks:
        args = ["--model", gguf, "--plan-only", "--world", str(world)] + (["--blocks", f"{rng[0]}:{rng[1]}"] if rng else [])
        rc, events, _ = G.generate(G.assembled_binary(ctx), args, log, 300, env, cfg["host"]["polite"])
        mine = next((e for e in events if e.get("event") == "plan"), None)
        ref = py_plan(gguf, world, rng)
        same = bool(mine) and mine["sha256"] == ref["sha256"] and mine["cuts"] == ref["cuts"] and \
            [(r["tensors"], r["tensor_bytes"], r["device_bytes"]) for r in mine["ranks"]] == [(r["tensors"], r["tensor_bytes"], r["device_bytes"]) for r in ref["ranks"]]
        ok = ok and same
        rows.append({"file": os.path.basename(gguf), "world": world, "range": rng, "equal": same, "sha256": ref["sha256"], "cuts": ref["cuts"],
                     "device_gib": [round(r["device_bytes"] / GIB, 2) for r in ref["ranks"]], "binary_sha256": mine and mine["sha256"]})
    save(ctx, "plan.json", rows)
    return {"evidence": "plan.txt, plan.json", "plans": len(rows), "equal": sum(r["equal"] for r in rows), "pass": ok}


def stage_equal(ctx):
    cfg, crit, run = ctx["cfg"], ctx["cfg"]["criteria"], ctx["cfg"]["run"]
    log = os.path.join(ctx["res"], "equal.txt")
    open(log, "w").close()
    out = {"evidence": "equal.txt, equal.json, equal-lease.txt", "binary": {"bytes": os.path.getsize(G.assembled_binary(ctx)), "mtime_utc": G.utc(os.path.getmtime(G.assembled_binary(ctx)))}}
    ids = load_json(cfg["model"]["prompt_ids"])
    names = list(ids)
    recorded = load_json(cfg["model"]["recorded_runs"])
    rec_plain = {names[e["prompt"]]: e for e in recorded["moe-1"]["events"] if e.get("event") == "generated"}
    rec_traced = {names[e["prompt"]]: e for e in recorded["moe-traced"]["events"] if e.get("event") == "traced"}
    rec_loaded = next(e for e in recorded["moe-1"]["events"] if e.get("event") == "loaded")
    dense_ids = load_json(os.path.join(cfg["model"]["dense_results"], "prompt-ids.json"))
    dense_rec = {names[i]: g["ids"] for i, g in enumerate(load_json(os.path.join(cfg["model"]["dense_results"], "gpu-runs.json"))["assembled"][0]["generated"])}
    base = ["--context", str(run["context"]), "--max-tokens", str(run["max_tokens"])]
    files = [x for p in names for x in ("--ids-file", write_ids(ctx, p, ids[p]))]
    dfiles = [x for p in names for x in ("--ids-file", write_ids(ctx, "dense-" + p, dense_ids[p]))]
    layers = ",".join(str(x) for x in S.load_config(os.path.join(ROOT, recipe_gate_config(ctx)))["run"]["trace_layers"])   # as the recorded traced run
    plan = [("placed-plain", ["--model", cfg["model"]["gguf"]] + placed_flags(ctx) + base + files),
            ("placed-traced", ["--model", cfg["model"]["gguf"]] + placed_flags(ctx) + base + ["--trace", "--trace-top", str(run["trace_top"]), "--trace-layers", layers] + files),
            ("dense-placed", ["--model", cfg["model"]["dense_gguf"]] + placed_flags(ctx) + base + dfiles)]
    hold = S.Hold(cfg, os.path.join(ctx["res"], "equal-lease.txt"), out, wait_s=cfg["host"]["gpu_wait_s"])
    runs = {}
    try:
        hold.take("strixflerp gate moe-placed", "equal")
        for label, args in plan:
            hold.between("strixflerp gate moe-placed", label)
            runs[label] = run_binary(ctx, label, args, log, run["run_timeout_s"])
    except RuntimeError as e:
        out["error"] = str(e)
    finally:
        hold.give()
    with open(os.path.join(ctx["scratch"], "equal-runs.json"), "w") as f:      # megabytes: kept on the host, digests are committed
        json.dump(runs, f)
    if "error" in out:
        return out
    why = [f"run {k} exited {v['rc']}" for k, v in runs.items() if v["rc"] != 0]
    by = lambda label, kind: {names[e["prompt"]]: e for e in runs[label]["events"] if e.get("event") == kind}  # noqa: E731
    plain, traced, dense = by("placed-plain", "generated"), by("placed-traced", "traced"), by("dense-placed", "generated")
    doc = {"plain": {}, "traced": {}, "dense": {}, "load": {}}
    for p in names:
        a = {"tokens": len(plain.get(p, {}).get("ids", [])), "equal_recorded": p in plain and plain[p]["ids"] == rec_plain[p]["ids"],
             "decode_tok_s": plain.get(p, {}).get("decode_tok_s"), "first_token_s": plain.get(p, {}).get("first_token_s"),
             "recorded_decode_tok_s": rec_plain[p].get("decode_tok_s"), "recorded_first_token_s": rec_plain[p].get("first_token_s")}
        doc["plain"][p] = a
        if not a["equal_recorded"]:
            why.append(f"plain, {p}: ids differ from the recorded run's ({G.first_difference(plain.get(p, {}).get('ids', []), rec_plain[p]['ids'])})")
        if p in traced:
            c = compare_traced(traced[p], rec_traced[p])
            c.update(digest_traced(traced[p]), traced_equals_plain=p in plain and traced[p]["ids"] == plain[p]["ids"])
            why += [f"traced, {p}: {w}" for w in c["why"]]
            if not c["traced_equals_plain"] or not c["argmax_is_token"]:
                why.append(f"traced, {p}: ids equal the plain run's {c['traced_equals_plain']}, argmax is the token {c['argmax_is_token']}")
            doc["traced"][p] = c
        else:
            why.append(f"traced, {p}: missing")
        d = {"equal_recorded": p in dense and dense[p]["ids"] == dense_rec[p], "tokens": len(dense.get(p, {}).get("ids", []))}
        doc["dense"][p] = d
        if not d["equal_recorded"]:
            why.append(f"dense through the placed loader, {p}: ids differ from the dense recipe's recorded run")
    for label, gguf in (("placed-plain", cfg["model"]["gguf"]), ("placed-traced", cfg["model"]["gguf"]), ("dense-placed", cfg["model"]["dense_gguf"])):
        ref = py_plan(gguf)
        ld = runs[label]["loaded"]
        why += [f"{label}: {w}" for w in judge_load(ld, ref["ranks"][0], ref["sha256"], crit)]
        doc["load"][label] = {"load_s": (ld or {}).get("load_s"), "placed": (ld or {}).get("placed"), "device_bytes": (ld or {}).get("device_bytes"),
                              "weight_resident_bytes": (ld or {}).get("weight_resident_bytes"), "weight_regions": (ld or {}).get("weight_regions")}
    doc["recorded_load"] = {k: rec_loaded.get(k) for k in ("load_s", "device_bytes", "weight_resident_bytes", "weight_regions")}
    doc["failures"] = why
    save(ctx, "equal.json", doc)
    out.update(failures=why, runs={k: {"rc": v["rc"], "seconds": v["seconds"]} for k, v in runs.items()},
               not_judged="the brief's criterion (sha256 of the logits against today's binary) was not run: today's binary has only the buffered loader, "
                          "which the host's I/O rule forbids; the comparison made is with the recorded run of the binary before this stage")
    out["pass"] = not why
    return out


def block_check(ctx, gguf, dump, log, tag):
    """gate/moe_block_check.py on a dump, niced, on [host] block_check_threads threads, killed when the load rises by
    [host] load_stop; the pages it read of the file are dropped from the page cache afterwards."""
    cfg, h = ctx["cfg"], ctx["cfg"]["host"]
    wait_for_load(cfg, f"block check {tag}", log)
    n = str(h["block_check_threads"])
    env = dict(os.environ, OMP_NUM_THREADS=n, OPENBLAS_NUM_THREADS=n, MKL_NUM_THREADS=n, **G.NO_GPU)
    cmd = list(h["polite"]) + ["taskset", "-c", h["cpus"], sys.executable, os.path.join(ROOT, "gate", "moe_block_check.py"), "--gguf", gguf,
                               "--gguf-py", cfg["model"]["gguf_py"], "--dump", dump]
    start, t0 = load1(), time.time()
    with open(log, "a") as f:
        f.write("$ " + " ".join(cmd) + "\n")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    killed = None
    while p.poll() is None:
        time.sleep(5)
        if load1() - start > h["load_stop"]:
            killed = f"stopped: the 1-minute load went from {start} to {load1()}"
        elif time.time() - t0 > cfg["run"]["check_timeout_s"]:
            killed = "stopped: timeout"
        if killed:
            p.kill()
            break
    so, se = p.communicate()
    try:                                     # the experts it read came through the page cache: give them back
        fd = os.open(gguf, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except OSError:
        pass
    with open(log, "a") as f:
        f.write(se[-1500:] + "\n")
    try:
        check = json.loads(so)
    except ValueError:
        check = {"error": killed or (se or so)[-300:]}
    check.update(seconds=round(time.time() - t0, 1), load_before=start, load_after=load1())
    return check


def bench_range(ctx, hold, gguf, rng, layers, context, log, tag, timeout):
    """One block range as a stage: the --stage-bench run under the hold, then the numpy check. Returns the record."""
    cfg, crit, run = ctx["cfg"], ctx["cfg"]["criteria"], ctx["cfg"]["run"]
    ids = load_json(cfg["model"]["prompt_ids"])
    long_ids = max(ids.values(), key=len)
    dump = os.path.join(ctx["scratch"], f"moe-dump-{tag}.bin")
    if os.path.isfile(dump):
        os.remove(dump)
    args = ["--model", gguf] + placed_flags(ctx, rng) + ["--context", str(context), "--stage-bench", "--chunk-rows", str(run["chunk_rows"]),
            "--decode-steps", str(run["decode_steps"]), "--dump", dump, "--dump-layers", ",".join(str(x) for x in layers),
            "--dump-rows", ",".join(str(x) for x in run["check_rows"]), "--ids-file", write_ids(ctx, "long", long_ids)]
    rec = {"range": rng, "check_layers": layers, "memory_before": memory_now(), "load_before": load1()}
    hold.between("strixflerp gate moe-placed", tag)
    r = run_binary(ctx, tag, args, log, timeout)
    rec["memory_after_exit"] = memory_now()
    rec.update(rc=r["rc"], seconds=r["seconds"], loaded=r["loaded"],
               events=[e for e in r["events"] if e.get("event") in ("stage_bench", "stage_bench_skipped", "synthetic_input", "moe_dump")])
    ref = py_plan(gguf, 1, rng)
    why = [f"exit {r['rc']}"] if r["rc"] != 0 else []
    why += judge_load(r["loaded"], ref["ranks"][0], ref["sha256"], crit)
    if r["rc"] == 0:
        why += judge_bench(r["events"], rng[0] == 0)
    rec["plan"] = ref
    rec["failures_before_check"] = list(why)
    rec["dump"] = dump
    return rec, why


def finish_range(ctx, rec, why, gguf, log, tag, records_min):
    crit = ctx["cfg"]["criteria"]
    if os.path.isfile(rec["dump"]) and os.path.getsize(rec["dump"]):
        check = block_check(ctx, gguf, rec["dump"], log, tag)
        save(ctx, f"block-check-{tag}.json", check)
        j = judge_block_check(check, crit, records_min)
        rec["block_check"] = dict(j, seconds=check.get("seconds"), load_before=check.get("load_before"), load_after=check.get("load_after"))
        why += [f"block check: {w}" for w in j["why"]]
        os.remove(rec["dump"])
    else:
        why.append("block check: no dump")
    del rec["dump"]
    rec["failures"], rec["pass"] = why, not why
    return rec


def stage_range35(ctx):
    cfg, run = ctx["cfg"], ctx["cfg"]["run"]
    log = os.path.join(ctx["res"], "range35.txt")
    open(log, "w").close()
    out = {"evidence": "range35.txt, range35.json, block-check-range35-*.json, range35-lease.txt"}
    hold = S.Hold(cfg, os.path.join(ctx["res"], "range35-lease.txt"), out, wait_s=cfg["host"]["gpu_wait_s"])
    pending = []
    try:
        hold.take("strixflerp gate moe-placed", "range35")
        for rng, layers in zip(run["range35_blocks"], run["range35_check"]):
            tag = f"range35-{rng[0]}-{rng[1]}"
            pending.append((tag, *bench_range(ctx, hold, cfg["model"]["gguf"], rng, layers, run["ornith_context"], log, tag, run["run_timeout_s"])))
    except RuntimeError as e:
        out["error"] = str(e)
    finally:
        hold.give()
    # the numpy checks run after the GPU is given back
    recs = [finish_range(ctx, rec, why, cfg["model"]["gguf"], log, tag,
                         cfg["criteria"]["check_records_min"] if rec["range"][0] == 0 else cfg["criteria"]["check_records_min"] // 2)
            for tag, rec, why in pending]
    save(ctx, "range35.json", recs)
    out["ranges"] = [{"range": r["range"], "pass": r["pass"], "failures": r["failures"], "block_check": r.get("block_check")} for r in recs]
    out["pass"] = "error" not in out and bool(recs) and all(r["pass"] for r in recs)
    return out


def stage_ornith(ctx):
    cfg, run = ctx["cfg"], ctx["cfg"]["run"]
    log = os.path.join(ctx["res"], "ornith.txt")
    open(log, "w").close()
    out = {"evidence": "ornith.txt, ornith.json, block-check-ornith.json, ornith-lease.txt"}
    gguf = cfg["model"]["ornith_gguf"]
    hold = S.Hold(cfg, os.path.join(ctx["res"], "ornith-lease.txt"), out, wait_s=cfg["host"]["gpu_wait_s"])
    pending = None
    try:
        hold.take("strixflerp gate moe-placed", "ornith")
        mem = memory_now()                       # measured with the GPU held and nothing of this gate loaded
        free = min(mem["gtt_total"] - mem["gtt_used"], mem["mem_available"])
        sizes = {n: py_plan(gguf, 1, [0, n])["ranks"][0]["device_bytes"] for n in range(1, run["max_blocks"] + 1)}
        n = pick_blocks(sizes, free, int((run["margin_gib"] + run["arena_reserve_gib"]) * GIB), run["max_blocks"], run["min_blocks"])
        out["memory"] = dict(mem, free_bytes=free, margin_gib=run["margin_gib"], arena_reserve_gib=run["arena_reserve_gib"],
                             plan_gib={k: round(v / GIB, 2) for k, v in sizes.items()}, blocks=n)
        if n is None:
            raise RuntimeError(f"fewer than {run['min_blocks']} blocks fit: {free} bytes free")
        layers = sorted({0, 7, n - 1})
        pending = bench_range(ctx, hold, gguf, [0, n], layers, run["ornith_context"], log, "ornith", run["ornith_timeout_s"])
    except RuntimeError as e:
        out["error"] = str(e)
    finally:
        hold.give()
    if pending is None:
        return out
    rec = finish_range(ctx, pending[0], pending[1], gguf, log, "ornith", cfg["criteria"]["check_records_min"])
    save(ctx, "ornith.json", rec)
    out.update(range=rec["range"], failures=rec["failures"], block_check=rec.get("block_check"))
    out["pass"] = "error" not in out and rec["pass"]
    return out


# --- summary -----------------------------------------------------------------------------------------------------
def bench_lines(rec):
    L, p = [], (rec.get("loaded") or {}).get("placed") or {}
    ld = rec.get("loaded") or {}
    if p:
        L.append(f"Loaded blocks {p['blocks'][0]}..{p['blocks'][1] - 1} of {p['model_blocks']}: {p['tensors']} tensors, {p['tensor_bytes'] / GIB:.2f} GiB read into one "
                 f"allocation of {p['allocated_bytes']} bytes (the plan's: {rec['plan']['ranks'][0]['device_bytes']}); fill {p['fill_s']:.1f} s of which {p['pause_s']:.1f} s "
                 f"were pauses, {p['read_gb_per_s']:.3f} GB/s ({p['read_calls']} direct reads by one thread: {p['direct_io']}); RssFile across the fill "
                 f"{p['rss_file_before_fill']} -> {p['rss_file_after_fill']} bytes; device free {p['hip_free_before'] / GIB:.1f} -> {p['hip_free_after'] / GIB:.1f} GiB of "
                 f"{p['hip_total'] / GIB:.1f}. Whole load {ld.get('load_s')} s. Arena and state as the executor reports them: {ld.get('device_bytes', 0) / GIB:.2f} GiB at "
                 f"context {ld.get('context')} (batch {ld.get('max_prompt_batch')}); the origin's estimate at context 32768: "
                 f"{(ld.get('arena_estimate_bytes') or {}).get('32768', 0) / GIB:.2f} GiB. Scratch floats a token (`intermediate`): {ld.get('intermediate')}.")
    for e in rec.get("events", []):
        if e["event"] == "stage_bench":
            n = e["blocks"][1] - e["blocks"][0]
            d, f = e["decode_at_context"], e["decode_fresh"]
            L += ["", f"Phase `{e['phase']}`, {n} blocks, hidden {e['hidden']}:",
                  f"- chunk of {e['chunk_rows']} rows: {[round(x, 3) for x in e['chunk_s']]} s (the first run is dumped), **{e['chunk_ms_per_block']:.1f} ms a block**, "
                  f"{e['chunk_rows_per_s']:.0f} rows/s through these blocks; three runs the same bytes: {e['chunk_repeats_equal']}; output RMS {e['chunk_out']['rms']:.4g}, "
                  f"non-finite {e['chunk_out']['non_finite']};",
                  f"- decode from an empty context, {f['with_logits_request']['steps']} steps asked for logits (the origin's graph path): first "
                  f"{f['with_logits_request']['first_ms']:.1f} ms, then {f['with_logits_request']['mean_ms_after_first']:.2f} ms a step; eager: "
                  f"{f['eager']['mean_ms_after_first']:.2f} ms a step; **ms a block: {e['decode_ms_per_block']['with_logits_request']} (graph path), "
                  f"{e['decode_ms_per_block']['eager']:.3f} (eager)**;",
                  f"- decode at context {d['context']}: {d['with_logits_request']['mean_ms_after_first']:.2f} ms a step asked for logits, {d['eager']['mean_ms_after_first']:.2f} eager."]
        elif e["event"] == "stage_bench_skipped":
            L += ["", f"Phase `{e['phase']}` not run: {e['why']}."]
    bc = rec.get("block_check")
    if bc:
        L += ["", f"Block check (`gate/moe_block_check.py`, numpy float64, {bc.get('seconds')} s, load {bc.get('load_before')} -> {bc.get('load_after')}): {bc['records']} records "
              f"of blocks {bc['layers']}; the same experts in {bc['same_set']} (in the same order in {bc['same_order']}; {bc['ties_excused']} excused as ties, "
              f"{bc['other_set']} not); weights within {bc['weights_max_abs']}, the shared gate logit within {bc['gate_logit_max_abs']}, the block output within "
              f"{bc['out_max_abs_over_rms']} of its RMS; smallest float64 margin to the next expert {bc['smallest_margin']}."]
    if rec.get("failures"):
        L += ["", "Criteria that did not hold:"] + [f"- {x}" for x in rec["failures"]]
    return L


def write_summary(ctx, results):
    cfg, res = ctx["cfg"], ctx["res"]
    ok = all(results.get(s, {}).get("pass") for s in STAGES)
    missing = [s for s in STAGES if s not in results]
    verdict = "PASSED" if ok else ("NOT COMPLETE: " + ", ".join(missing) + " not run" if missing and all(results[s].get("pass") for s in results) else "FAILED")
    save(ctx, "summary.json", {"gate": "moe-placed", "host": socket.gethostname(), "written_utc": G.utc(), "stages": results, "pass": ok, "disk_reads": ctx["disk_all"]})
    L = [f"# Gate of the placed loader and the block range: {cfg['recipe']}", "", f"Host {socket.gethostname()}, written {G.utc()}. **{verdict}** under `[criteria]` of "
         f"`{ctx['config']}` (committed before the first GPU run). Stage P1 of `docs/PHASE5-FOURNODE-DESIGN.md`.", "",
         "**Gate (1) was not judged as the brief set it.** Its criterion is bit equality (sha256 of the copied logits) with today's binary. Today's binary has only the "
         "origin's loader, which reads the whole 20 GiB file through its mapping with eight workers, and the host's I/O rule of 2026-11-11 forbids that here. Stage `equal` "
         "compares with the recorded run of the binary before this stage instead: ids, and per row the log-sum-exp over the whole row and the 40 largest logits as printed.", "",
         "| stage | result | when (UTC) | what |", "|---|---|---|---|"]
    for s in STAGES:
        r = results.get(s)
        if r is None:
            L.append(f"| {s} | not run | | |")
            continue
        what = r.get("error") or {"plan": lambda: f"{r.get('equal')} of {r.get('plans')} plans equal between the binary and gguf_census.py",
                                  "equal": lambda: f"{len(r.get('failures', []))} failure(s); runs {r.get('runs')}",
                                  "range35": lambda: "; ".join(f"blocks {x['range']}: {'pass' if x['pass'] else 'FAIL'}" for x in r.get("ranges", [])),
                                  "ornith": lambda: f"blocks {r.get('range')}: {len(r.get('failures', []))} failure(s)"}[s]()
        L.append(f"| {s} | {'PASS' if r.get('pass') else 'FAIL'} | {r.get('when_utc', '')} | {what} |")
    eq = os.path.join(res, "equal.json")
    if "equal" in results and os.path.isfile(eq):
        with open(eq) as f:
            d = json.load(f)
        L += ["", "## Stage `equal`: the 35B model, world 1, whole, through the placed loader", ""]
        L += ["Criteria that did not hold:"] + [f"- {x}" for x in d["failures"]] if d["failures"] else ["Every criterion judged held."]
        L += ["", "| prompt | plain ids equal the recorded run's | traced: ids, rows, log-sum-exp and top 40 of every row equal | rows | dense model through the placed loader equals its record |",
              "|---|---|---|---|---|"]
        for p in d["plain"]:
            t = d["traced"].get(p, {})
            L.append(f"| {p} | {d['plain'][p]['equal_recorded']} ({d['plain'][p]['tokens']} tokens) | {t.get('equal')} | {t.get('rows')} | {d['dense'][p]['equal_recorded']} |")
        for label, v in d["load"].items():
            p = v.get("placed") or {}
            if p:
                L.append("")
                L.append(f"`{label}`: load {v['load_s']} s (fill {p['fill_s']:.1f} s with {p['pause_s']:.1f} s of pauses, {p['read_gb_per_s']:.3f} GB/s, one direct reader: "
                         f"{p['direct_io']}); {p['tensors']} tensors, {p['tensor_bytes']} bytes into one allocation of {p['allocated_bytes']}; {v['weight_regions']} regions; "
                         f"RssFile {p['rss_file_before_fill']} -> {p['rss_file_after_fill']}; not loaded: {p['unloaded_tensors']} tensors, {p['unloaded_bytes']} bytes.")
        L += ["", f"The recorded run (the origin's loader, eight buffered workers): {d['recorded_load']}. Decode: "
              + ", ".join(f"{p} {v['decode_tok_s']:.1f} tok/s (recorded {v['recorded_decode_tok_s']:.1f})" for p, v in d["plain"].items() if v.get("decode_tok_s")) + "."]
    for name, title in (("range35.json", "## Stage `range35`: block ranges of the 35B model as stages"), ("ornith.json", "## Stage `ornith`: leading blocks of Ornith-1.5 397B as a stage")):
        path = os.path.join(res, name)
        if name.split(".")[0] in results and os.path.isfile(path):
            with open(path) as f:
                recs = json.load(f)
            L += ["", title]
            if name == "ornith.json":
                L += ["", f"Memory rule: {results['ornith'].get('memory')}."]
            for rec in recs if isinstance(recs, list) else [recs]:
                L += ["", f"### Blocks {rec['range'][0]}..{rec['range'][1] - 1}", ""] + bench_lines(rec)
    reads = ctx["disk_all"]
    if reads:
        tot = sum(r["bytes"] for r in reads)
        L += ["", "## Disk reads on this host", "", f"{tot / 1e9:.1f} GB by the loader (one direct reader, `{' '.join(cfg['host']['reader_flags'])}`), "
              f"{sum(r['seconds'] or 0 for r in reads):.0f} s of reading with its pauses, between {reads[0]['when']} and {reads[-1]['when']}:", "",
              "| run | GB | seconds | ended (UTC) |", "|---|---|---|---|"]
        L += [f"| {r['run']} | {r['bytes'] / 1e9:.2f} | {r['seconds']} | {r['when']} |" for r in reads]
    with open(os.path.join(res, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="gate/qwen36-35b-a3b.flerp-moe-placed.toml")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--summarize", action="store_true", help="rewrite summary.md from the recorded results; runs nothing")
    a = ap.parse_args()
    global G, S
    import gufo_recipe as G                   # noqa: E402  (needs the catalogue; the judgements above do not)
    import strixite_recipe as S               # noqa: E402
    from flerplib.catalog import Catalog
    cfg = S.load_config(os.path.join(ROOT, a.config))
    cat = Catalog(ROOT)
    recipe = cat.recipe(cfg["recipe"])
    res = os.path.join(ROOT, "gate", "results", recipe.name + "-placed")
    os.makedirs(res, exist_ok=True)
    ctx = {"cat": cat, "recipe": recipe, "cfg": cfg, "res": res, "scratch": os.path.join(ROOT, cfg["host"]["scratch"]), "config": a.config, "disk": []}
    os.makedirs(ctx["scratch"], exist_ok=True)
    results, prior, disk_all = {}, os.path.join(res, "summary.json"), []
    if os.path.isfile(prior):
        with open(prior) as f:
            old = json.load(f)
        results, disk_all = old.get("stages", {}), old.get("disk_reads", [])
    ctx["disk_all"] = disk_all
    if a.summarize:
        return 0 if write_summary(ctx, results) else 1
    fns = {"plan": stage_plan, "equal": stage_equal, "range35": stage_range35, "ornith": stage_ornith}
    want = a.stages.split(",")
    for s in STAGES:
        if s not in want:
            continue
        print(f"== {s}", flush=True)
        try:
            results[s] = fns[s](ctx)
        except RuntimeError as e:
            results[s] = {"error": str(e)}
        results[s]["when_utc"] = G.utc()
        ctx["disk_all"] += ctx["disk"]
        ctx["disk"] = []
        print(f"   {'PASS' if results[s].get('pass') else 'FAIL'} {results[s].get('error', '')}", flush=True)
        write_summary(ctx, results)
        if not results[s].get("pass") and not a.keep_going:
            break
    return 0 if all(results.get(s, {}).get("pass") for s in want) else 1


if __name__ == "__main__":
    sys.exit(main())
