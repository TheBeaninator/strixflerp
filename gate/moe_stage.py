#!/usr/bin/env python3
"""Gate of the stages and the relay (runtime.flerp.stage_relay, executor.flerp.qwen35moe as N stages): stage P2 of
docs/PHASE5-FOURNODE-DESIGN.md. Run on the gfx1151 build host, after the recipe is built:

    python3 gate/moe_stage.py [--config gate/qwen36-35b-a3b.flerp-moe-stage.toml] [--stages loopback,tcp,faults]

Stages, each writing evidence under gate/results/<recipe>-stage/:
  loopback  one process, one load of the 35B model: every configuration of [run] loopback_configs as stages over
            that load joined by the in-process relay, the four prompts traced; then the in-process fault cases.
  tcp       one process a rank on 127.0.0.1 for each of [run] tcp_configs, each rank loading only its blocks; every
            prompt with the decode graph on, then off.
  faults    TCP on the small dense model: a killed rank, a dropped frame, a stale-epoch frame, a replayed frame, a rank
            of another epoch; every rank must fail by itself within the deadline.

Criteria: [criteria] of the config, committed before the first GPU run. P1's world-1 values (ids and the sha256 of
every logits row) are the reference.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gate"))
import moe_placed as P  # noqa: E402
import summary as SM  # noqa: E402  (summary.json / summary.md, config_sha256)

STAGES = ("loopback", "tcp", "faults")
GIB = float(1 << 30)


# --- judgements (no I/O: tests/test_moe_stage.py) ----------------------------------------------------------------
def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def judge_staged(e, ref):
    """Criterion 1 for one `staged` event against P1's traced digest of the same prompt. Returns [failures]."""
    why = []
    if e["ids"] != ref["ids"]:
        why.append(f"ids differ from P1's at {first_diff(e['ids'], ref['ids'])}")
    want = ref["logits_sha256"]
    if e["rows"] != len(want) or len(e["logits_sha256"]) != len(want):
        why.append(f"{e['rows']} logits rows, P1 has {len(want)}")
    d = first_diff(e["logits_sha256"], want)
    if d is not None:
        why.append(f"logits row {d} differs from P1's (sha256)")
    if not e["argmax_is_token"]:
        why.append("a row's argmax is not the token")
    return why


def judge_matrix(events, configs, names, refs, transport):
    """Criterion 1 over one run's events: every (configuration, prompt) present and equal. Returns (rows, [failures])."""
    by = {(e["config"], e["prompt"]): e for e in events if e.get("event") == "staged"}
    rows, why = [], []
    for c in configs:
        equal = 0
        for i, p in enumerate(names):
            e = by.get((c, i))
            if e is None:
                why.append(f"{transport} {c}, {p}: no result")
                continue
            w = judge_staged(e, refs[p])
            why += [f"{transport} {c}, {p}: {x}" for x in w]
            equal += not w
        rows.append({"transport": transport, "config": c, "prompts_equal": equal, "prompts": len(names)})
    for e in events:
        if e.get("event") in ("staged_failed", "error"):
            why.append(f"{transport}: {e.get('config', '')} {e.get('error', '')}")
    return rows, why


def judge_loop_faults(events):
    """Criterion 3 on the `fault_case` events of the loopback run. Returns (cases, [failures])."""
    cases = [e for e in events if e.get("event") == "fault_case"]
    why, seen = [], {e["fault"] for e in cases}
    for need in ("drop", "stale-epoch", "replay", "bad-bytes", "wrong-epoch-at-handshake"):
        if need not in seen:
            why.append(f"loopback fault {need}: did not run")
    for e in cases:
        if e["fault"] == "wrong-epoch-at-handshake":
            if not e.get("refused"):
                why.append("loopback: a rank of another epoch was not refused at the handshake")
            continue
        if not (e["fired"] and e["rank0_call_failed"] and e["all_poisoned"]):
            why.append(f"loopback fault {e['fault']}: fired {e['fired']}, rank 0 failed {e['rank0_call_failed']}, all poisoned {e['all_poisoned']}")
        reasons = " ".join(e["reasons"])
        if e["fault"] == "stale-epoch" and "epoch" not in reasons:
            why.append("loopback stale-epoch: no rank names the epoch")
        if e["fault"] == "replay" and "ordinal" not in reasons:
            why.append("loopback replay: no rank names the ordinal")
    return cases, why


def judge_tcp_fault(case, ranks, fault_t, crit_s, killed=()):
    """Criterion 4 for one TCP fault case. ranks: {rank: {"rc", "events", "exited_at"}}; fault_t: unix time of the
    fault. Returns {"pass", "why", per rank seconds to failure}."""
    why, after = [], {}
    for r, info in sorted(ranks.items()):
        if info["rc"] is None:
            why.append(f"rank {r} did not exit (hang)")
            continue
        if r in killed:
            if info["rc"] not in (-signal.SIGKILL, 128 + signal.SIGKILL, 137):
                why.append(f"rank {r} was killed but exited {info['rc']}")
            continue
        failed = [e for e in info["events"] if e.get("event") == "relay_failed"]
        if info["rc"] != 3 or not failed:
            why.append(f"rank {r} exited {info['rc']} with {len(failed)} relay_failed event(s)")
            continue
        after[r] = round(failed[0]["t_unix"] - fault_t, 3) if fault_t else None
        if fault_t is None or after[r] > crit_s:
            why.append(f"rank {r} failed {after[r]} s after the fault (bound {crit_s} s)")
    reasons = " ".join(e.get("reason", "") for info in ranks.values() for e in info["events"] if e.get("event") == "relay_failed")
    if case in ("stale-epoch", "epoch") and "epoch" not in reasons:
        why.append(f"{case}: no rank names the epoch")
    if case == "replay" and "ordinal" not in reasons:
        why.append("replay: no rank names the ordinal")
    return {"pass": not why, "why": why, "seconds_to_fail": after}


def tcp_overheads(rank0_events, timing_events):
    """Relay overhead of TCP runs: rank 0's call wall time minus every rank's forwards (and the last rank's logits
    copy, for decode). One record a run (graph mode x prompt)."""
    out = []
    staged = [e for e in rank0_events if e.get("event") == "staged"]
    for k, e in enumerate(staged):
        others = [t["timing"] for t in timing_events if t.get("run") == k]
        st = [e["stages"][0]] + others
        dec = sum(t["decode_ms"] for t in st)
        chunk = sum(c["ms"] for t in st for c in t["chunks"])
        last = max(st, key=lambda t: t["blocks"][1])
        calls = e["calls"]
        out.append({"config": e["config"], "prompt": e["prompt"], "stages_reported": len(st),
                    "decode_ms_per_token": (calls["decode_wall_ms"] - dec - last["logits_copy_ms"]) / max(calls["decode_calls"], 1),
                    "chunk_ms_per_chunk": (calls["chunk_wall_ms"] - chunk) / max(calls["chunks"], 1),
                    "stages": [{"rank": t["rank"], "blocks": t["blocks"], "decode_ms_per_block": t["decode_ms_per_block"],
                                "chunk_ms_per_block": [round(c["ms_per_block"], 3) for c in t["chunks"]]} for t in st]})
    return out


# --- host ------------------------------------------------------------------------------------------------------
def memory_ok(ctx, plan_bytes, stages):
    h, m = ctx["cfg"]["host"], P.memory_now()
    free = min(m["gtt_total"] - m["gtt_used"], m["mem_available"])
    need = plan_bytes + stages * h["arena_gib"] * GIB + h["margin_gib"] * GIB
    return free >= need, dict(m, free_bytes=free, need_bytes=int(need))


class Rank:
    """One process of a TCP run: stdout read line by line on a thread, each JSON event stamped."""

    def __init__(self, ctx, rank, args, log_path):
        h = ctx["cfg"]["host"]
        cmd = list(h["polite"]) + ["taskset", "-c", h["cpus"], P.G.assembled_binary(ctx)] + args
        self.rank, self.events, self.t0 = rank, [], time.time()
        self.log = open(log_path, "ab")
        self.log.write(("$ " + " ".join(cmd) + "\n").encode())
        self.log.flush()
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self.log, preexec_fn=P.S.die_with_parent,
                                     env=dict(os.environ, **P.G.runtime_env(ctx)))
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.proc.stdout:
            self.log.write(line)
            self.log.flush()
            try:
                e = json.loads(line)
                e["_seen"] = time.time()
                self.events.append(e)
            except ValueError:
                pass

    def has(self, kind):
        return any(e.get("event") == kind for e in self.events)

    def wait(self, deadline):
        try:
            rc = self.proc.wait(timeout=max(1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
            rc = None
        self.thread.join(timeout=10)
        self.log.close()
        return rc


def launch(ctx, gguf, world, cuts, rank_extra, ids_files, port, log, timeout, deadline_ms, graph_modes="on,off", watch=None):
    """Ranks started one after another (each after the previous printed `loaded`: one reader at a time). Returns
    ({rank: {"rc", "events"}}, last loaded time)."""
    cfg = ctx["cfg"]
    end = time.time() + timeout
    ranks = []
    for r in range(world):
        P.wait_for_load(cfg, f"rank {r}", log)
        args = ["--staged", "tcp", "--model", gguf, "--context", str(cfg["run"]["context"]), "--max-tokens", str(cfg["run"]["max_tokens"]),
                "--world", str(world), "--rank", str(r), "--cuts", cuts, "--port", str(port), "--epoch", str(cfg["run"]["epoch"]),
                "--deadline-ms", str(deadline_ms), "--graph-modes", graph_modes] + list(cfg["host"]["reader_flags"]) + rank_extra.get(r, [])
        if r == 0:
            args += [x for f in ids_files for x in ("--ids-file", f)]
        ranks.append(Rank(ctx, r, args, log))
        while not ranks[-1].has("loaded") and ranks[-1].proc.poll() is None and time.time() < end:
            time.sleep(0.5)
    loaded_at = max((e["_seen"] for rk in ranks for e in rk.events if e.get("event") == "loaded"), default=None)
    if watch:
        watch(ranks, end)
    out = {}
    for rk in ranks:
        rc = rk.wait(end)
        out[rk.rank] = {"rc": rc, "events": rk.events}
    for rk in ranks:
        for e in rk.events:
            if e.get("event") == "loaded":
                b = (e.get("placed") or {}).get("tensor_bytes", 0)
                ctx["disk"].append({"run": f"{os.path.basename(log)} rank {rk.rank}", "bytes": b, "seconds": e["placed"].get("fill_s"), "when": P.S.utc()})
    return out, loaded_at


def refs_and_names(ctx):
    cfg = ctx["cfg"]
    ids = P.load_json(cfg["model"]["prompt_ids"])
    names = list(ids)
    p1 = P.load_json(cfg["model"]["p1_results"])["traced"]
    files = [P.write_ids(ctx, p, ids[p]) for p in names]
    return names, {p: p1[p] for p in names}, files


def clean(e):
    return {k: v for k, v in e.items() if k != "_seen"}


# --- stages ----------------------------------------------------------------------------------------------------
def stage_loopback(ctx):
    cfg, run = ctx["cfg"], ctx["cfg"]["run"]
    log = os.path.join(ctx["res"], "loopback.txt")
    open(log, "w").close()
    out = {"evidence": "loopback.txt, loopback.json, loopback-lease.txt"}
    names, refs, files = refs_and_names(ctx)
    plan = P.py_plan(cfg["model"]["gguf"])
    hold = P.S.Hold(cfg, os.path.join(ctx["res"], "loopback-lease.txt"), out, wait_s=cfg["host"]["gpu_wait_s"])
    res = None
    try:
        hold.take("strixflerp gate moe-stage", "loopback")
        ok, mem = memory_ok(ctx, plan["ranks"][0]["device_bytes"], 5)
        out["memory_before"] = mem
        if not ok:
            raise RuntimeError(f"not enough free memory: {mem}")
        args = ["--staged", "loopback", "--model", cfg["model"]["gguf"], "--context", str(run["context"]), "--max-tokens", str(run["max_tokens"]),
                "--configs", ";".join(run["loopback_configs"]), "--fault-config", run["loopback_fault"], "--epoch", str(run["epoch"]),
                "--deadline-ms", str(run["deadline_ms"])] + list(cfg["host"]["reader_flags"]) + [x for f in files for x in ("--ids-file", f)]
        res = P.run_binary(ctx, "loopback", args, log, run["run_timeout_s"])
    except RuntimeError as e:
        out["error"] = str(e)
    finally:
        hold.give()
    if res is None:
        return out
    ev = res["events"]
    why = [] if res["rc"] == 0 else [f"the loopback run exited {res['rc']}"]
    rows, w = judge_matrix(ev, run["loopback_configs"], names, refs, "loopback")
    why += w
    cases, w = judge_loop_faults(ev)
    why += w
    loaded = res["loaded"] or {}
    pl = loaded.get("placed") or {}
    if pl.get("allocated_bytes") != plan["ranks"][0]["device_bytes"]:
        why.append(f"loopback load: allocated {pl.get('allocated_bytes')} bytes, the plan has {plan['ranks'][0]['device_bytes']}")
    if not (pl.get("single_reader") and pl.get("direct_io")):
        why.append("loopback load: not the single direct reader")
    staged = [e for e in ev if e.get("event") == "staged"]
    P.save(ctx, "loopback.json", {"matrix": rows, "faults": cases, "loaded": loaded, "staged": [
        {k: v for k, v in e.items() if k != "logits_sha256"} for e in staged], "failures": why})
    out.update(failures=why, matrix=rows, faults=[{k: c[k] for k in ("fault", "fired", "rank0_call_failed", "all_poisoned") if k in c} for c in cases],
               load={"load_s": loaded.get("load_s"), "bytes": pl.get("tensor_bytes"), "fill_s": pl.get("fill_s"), "pause_s": pl.get("pause_s")},
               seconds=res["seconds"])
    out["pass"] = not why
    return out


def stage_tcp(ctx):
    cfg, run = ctx["cfg"], ctx["cfg"]["run"]
    log = os.path.join(ctx["res"], "tcp.txt")
    open(log, "w").close()
    out = {"evidence": "tcp.txt, tcp.json, tcp-lease.txt", "runs": []}
    names, refs, files = refs_and_names(ctx)
    gguf = cfg["model"]["gguf"]
    why, rows, doc = [], [], {"runs": []}
    hold = P.S.Hold(cfg, os.path.join(ctx["res"], "tcp-lease.txt"), out, wait_s=cfg["host"]["gpu_wait_s"])
    try:
        for k, (world, cuts_text) in enumerate(run["tcp_configs"]):
            cuts = P.py_plan(gguf, world)["cuts"] if cuts_text == "even" else [int(x) for x in cuts_text.split(",")]
            plans = [P.py_plan(gguf, 1, [cuts[r], cuts[r + 1]]) for r in range(world)]
            hold.between("strixflerp gate moe-stage", f"tcp {world}:{cuts_text}") if k else hold.take("strixflerp gate moe-stage", f"tcp {world}:{cuts_text}")
            ok, mem = memory_ok(ctx, sum(p["ranks"][0]["device_bytes"] for p in plans), world)
            if not ok:
                why.append(f"tcp {world}:{cuts_text}: not enough free memory: {mem}")
                continue
            t0 = time.time()
            ranks, _ = launch(ctx, gguf, world, cuts_text, {}, files, run["port"] + 10 * k, log, run["run_timeout_s"], run["deadline_ms"])
            r0 = ranks[0]["events"]
            configs = [f"{world}:{cuts_text}:{g}" for g in ("on", "off")]
            mrows, w = judge_matrix(r0, configs, names, refs, "tcp")
            why += w
            rows += mrows
            for r, info in ranks.items():
                if info["rc"] != 0:
                    why.append(f"tcp {world}:{cuts_text}: rank {r} exited {info['rc']}")
                ld = next((e for e in info["events"] if e.get("event") == "loaded"), {})
                pl = ld.get("placed") or {}
                if pl.get("allocated_bytes") != plans[r]["ranks"][0]["device_bytes"]:
                    why.append(f"tcp {world}:{cuts_text}: rank {r} allocated {pl.get('allocated_bytes')} bytes, its range plan has {plans[r]['ranks'][0]['device_bytes']}")
                if not (pl.get("single_reader") and pl.get("direct_io")):
                    why.append(f"tcp {world}:{cuts_text}: rank {r} load not by the single direct reader")
            timing = [e for r, info in ranks.items() if r != 0 for e in info["events"] if e.get("event") == "stage_timing"]
            over = tcp_overheads(r0, timing)
            doc["runs"].append({"world": world, "cuts": cuts, "seconds": round(time.time() - t0, 1), "rc": {r: i["rc"] for r, i in ranks.items()},
                                "loaded": {r: next((clean(e) for e in i["events"] if e.get("event") == "loaded"), None) for r, i in ranks.items()},
                                "overheads": over, "staged": [{k2: v for k2, v in clean(e).items() if k2 != "logits_sha256"} for e in r0 if e.get("event") == "staged"]})
            out["runs"].append({"world": world, "cuts": cuts, "seconds": round(time.time() - t0, 1)})
    except RuntimeError as e:
        out["error"] = str(e)
    finally:
        hold.give()
    doc.update(matrix=rows, failures=why)
    P.save(ctx, "tcp.json", doc)
    out.update(matrix=rows, failures=why)
    out["pass"] = "error" not in out and not why and len(rows) == 2 * len(run["tcp_configs"])
    return out


def stage_faults(ctx):
    cfg, run = ctx["cfg"], ctx["cfg"]["run"]
    log = os.path.join(ctx["res"], "faults.txt")
    open(log, "w").close()
    out = {"evidence": "faults.txt, faults.json, faults-lease.txt"}
    gguf, world, n = cfg["model"]["fault_gguf"], run["fault_world"], run["fault_ordinal"]
    ids = P.load_json(cfg["model"]["fault_prompt_ids"])
    files = [P.write_ids(ctx, "fault-" + p, v) for p, v in ids.items()]
    bound = run["fault_deadline_ms"] / 1000 + run["fault_slack_s"]
    cases = [("control", {}), ("kill", {1: ["--fault", f"kill:{n}"]}), ("sigkill", {}), ("drop", {1: ["--fault", f"drop:{n}"]}),
             ("stale-epoch", {1: ["--fault", f"stale-epoch:{n}"]}), ("replay", {1: ["--fault", f"replay:{n}"]}),
             ("epoch", {1: ["--epoch", str(run["epoch"] - 1)]})]
    plan = P.py_plan(gguf)
    why, doc = [], {}
    hold = P.S.Hold(cfg, os.path.join(ctx["res"], "faults-lease.txt"), out, wait_s=cfg["host"]["gpu_wait_s"])
    try:
        hold.take("strixflerp gate moe-stage", "faults")
        ok, mem = memory_ok(ctx, plan["ranks"][0]["device_bytes"], world)
        out["memory_before"] = mem
        if not ok:
            raise RuntimeError(f"not enough free memory: {mem}")
        for k, (case, extra) in enumerate(cases):
            killed, kill_t = set(), {}

            def watch(ranks, end, case=case):
                if case != "sigkill":
                    return
                while time.time() < end and not ranks[0].has("staged") and ranks[0].proc.poll() is None:
                    time.sleep(0.05)
                if ranks[2].proc.poll() is None:
                    os.kill(ranks[2].proc.pid, signal.SIGKILL)
                    kill_t["t"] = time.time()
                    killed.add(2)
            ranks, loaded_at = launch(ctx, gguf, world, "even", extra, files, run["port"] + 100 + 10 * k, log, run["fault_timeout_s"],
                                      run["fault_deadline_ms"], watch=watch)
            # the process under taskset/ionice is the binary itself (exec), so its pid is the rank's
            if case == "kill":
                killed.add(1)
            fault_t = None
            if case in ("kill", "drop", "stale-epoch", "replay"):
                fault_t = next((e["t_unix"] for e in ranks[1]["events"] if e.get("event") == "fault_injected"), None)
            elif case == "sigkill":
                fault_t = kill_t.get("t")
            elif case == "epoch":
                fault_t = loaded_at
            if case == "control":
                r = {"pass": all(i["rc"] == 0 for i in ranks.values()), "why": [f"rank {q} exited {i['rc']}" for q, i in ranks.items() if i["rc"] != 0]}
                staged = [e for e in ranks[0]["events"] if e.get("event") == "staged"]
                r["dense_staged_ids"] = [e["ids"] for e in staged]
            else:
                r = judge_tcp_fault(case, ranks, fault_t, bound, killed)
            r["rc"] = {q: i["rc"] for q, i in ranks.items()}
            r["reasons"] = {q: [e.get("reason") for e in i["events"] if e.get("event") == "relay_failed"] for q, i in ranks.items()}
            doc[case] = r
            why += [f"{case}: {w}" for w in r["why"]]
    except RuntimeError as e:
        out["error"] = str(e)
    finally:
        hold.give()
    P.save(ctx, "faults.json", doc)
    out.update(cases={c: {"pass": d["pass"], "rc": d["rc"], "seconds_to_fail": d.get("seconds_to_fail")} for c, d in doc.items()}, failures=why)
    out["pass"] = "error" not in out and not why and len(doc) == len(cases)
    return out


# --- summary ---------------------------------------------------------------------------------------------------
def write_summary(ctx, results):
    SM.write_json(ctx["res"], {"stages": results, "disk_reads": ctx["disk_all"]}, ctx=ctx)
    L = [f"# Gate of the stages and the relay: {ctx['recipe'].name}", "",
         f"Host {os.uname().nodename}, written {P.S.utc()}. Criteria: `[criteria]` of `{ctx['config']}` (committed before the first GPU run). "
         "Stage P2 of `docs/PHASE5-FOURNODE-DESIGN.md`.", "",
         "| stage | result | when (UTC) | failures |", "|---|---|---|---|"]
    for s in STAGES:
        r = results.get(s)
        if r:
            L.append(f"| {s} | {'PASS' if r.get('pass') else 'FAIL'} | {r.get('when_utc', '')} | {len(r.get('failures', []))} {r.get('error', '')} |")
    L += ["", "## Invariance matrix (each cell: prompts whose ids and 128 logits-row hashes equal P1's)", "", "| transport | configuration | equal |", "|---|---|---|"]
    for s in ("loopback", "tcp"):
        for row in (results.get(s) or {}).get("matrix", []):
            L.append(f"| {row['transport']} | {row['config']} | {row['prompts_equal']}/{row['prompts']} |")
    for s in STAGES:
        fl = (results.get(s) or {}).get("failures") or []
        if fl:
            L += ["", f"Failures of `{s}` (first 20):", ""] + [f"- {x}" for x in fl[:20]]
    if (results.get("faults") or {}).get("cases"):
        L += ["", "## TCP faults", "", "| case | pass | exit codes | seconds from fault to each rank's failure |", "|---|---|---|---|"]
        for c, d in results["faults"]["cases"].items():
            L.append(f"| {c} | {d['pass']} | {d['rc']} | {d.get('seconds_to_fail')} |")
    if (results.get("loopback") or {}).get("faults"):
        L += ["", "## Loopback faults", ""] + [f"- {c}" for c in results["loopback"]["faults"]]
    L += ["", "## Disk reads", "", "| run | GB | fill s | when |", "|---|---|---|---|"]
    tot = 0
    for d in ctx["disk_all"]:
        tot += d["bytes"] or 0
        L.append(f"| {d['run']} | {(d['bytes'] or 0) / 1e9:.2f} | {d['seconds']} | {d['when']} |")
    L.append(f"| total | {tot / 1e9:.2f} | | |")
    SM.write_md(ctx["res"], L)
    return SM.all_pass(STAGES, results)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="gate/qwen36-35b-a3b.flerp-moe-stage.toml")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--keep-going", action="store_true")
    a = ap.parse_args()
    import gufo_recipe as G                   # noqa: E402  (needs the catalogue; the judgements above do not)
    import strixite_recipe as S               # noqa: E402
    from flerplib.catalog import Catalog
    P.G, P.S = G, S
    cfg = S.load_config(os.path.join(ROOT, a.config))
    cat = Catalog(ROOT)
    recipe = cat.recipe(cfg["recipe"])
    res = os.path.join(ROOT, "gate", "results", recipe.name + "-stage")
    os.makedirs(res, exist_ok=True)
    ctx = {"cat": cat, "recipe": recipe, "cfg": cfg, "res": res, "scratch": os.path.join(ROOT, cfg["host"]["scratch"]), "config": a.config, "disk": []}
    os.makedirs(ctx["scratch"], exist_ok=True)
    results, disk_all, prior = {}, [], os.path.join(res, "summary.json")
    if os.path.isfile(prior):
        with open(prior) as f:
            old = json.load(f)
        results, disk_all = old.get("stages", {}), old.get("disk_reads", [])
    ctx["disk_all"] = disk_all
    fns = {"loopback": stage_loopback, "tcp": stage_tcp, "faults": stage_faults}
    want = a.stages.split(",")
    for s in STAGES:
        if s not in want:
            continue
        print(f"== {s}", flush=True)
        try:
            results[s] = fns[s](ctx)
        except RuntimeError as e:
            results[s] = {"error": str(e)}
        results[s]["when_utc"] = S.utc()
        ctx["disk_all"] += ctx["disk"]
        ctx["disk"] = []
        print(f"   {'PASS' if results[s].get('pass') else 'FAIL'} {results[s].get('error', '')}", flush=True)
        write_summary(ctx, results)
        if not results[s].get("pass") and not a.keep_going:
            break
    return 0 if all(results.get(s, {}).get("pass") for s in want) else 1


if __name__ == "__main__":
    sys.exit(main())
