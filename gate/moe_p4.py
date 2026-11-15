#!/usr/bin/env python3
"""Stage P4 of docs/PHASE5-FOURNODE-DESIGN.md on two workers, driven from control-node, inside a pod-quad hold of both
that someone else took (this script takes and releases none; it refuses to start without the hold):

    python3 gate/moe_p4.py m1        # M1: a 160 KiB two-way exchange over the rail, timed
    python3 gate/moe_p4.py q35       # the 35B at world 2 over the rail, both cut sets, graph on and off
    python3 gate/moe_p4.py probe     # read rate of the Ornith file from each worker (1 GiB, O_DIRECT)
    python3 gate/moe_p4.py ornith [--blocks-per-stage N]   # stages 0 and 1 of the world-4 plan, stage-bench + block check
    python3 gate/moe_p4.py cleanup   # stop every flerp-p4-* unit on both workers; verify none remain

Every process on a worker is a transient user unit named [host] unit_prefix + a tag. Criteria: [criteria] of
gate/qwen36-35b-a3b.flerp-moe-p4.toml, committed before the first run. Evidence: gate/results/qwen36-35b-a3b.flerp-moe-p4/
and gate/results/ornith15-397b.p4/.
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
sys.path.insert(0, os.path.join(ROOT, "gate"))
import moe_placed as MP  # noqa: E402
import moe_stage as MS  # noqa: E402

CONFIG = os.path.join(ROOT, "gate", "qwen36-35b-a3b.flerp-moe-p4.toml")
RES35 = os.path.join(ROOT, "gate", "results", "qwen36-35b-a3b.flerp-moe-p4")
RESORN = os.path.join(ROOT, "gate", "results", "ornith15-397b.p4")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
XDG = "export XDG_RUNTIME_DIR=/run/user/$(id -u); "


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def say(log, text):
    line = f"{utc()} {text}"
    print(line, flush=True)
    with open(log, "a") as f:
        f.write(line + "\n")


def on(ip, cmd, timeout=120):
    p = subprocess.run(SSH + [f"pod@{ip}", cmd], capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def start(cfg, ip, tag, argv, env=""):
    h = cfg["host"]
    unit = h["unit_prefix"] + tag
    inner = f"cd {h['work']} && {env} exec {' '.join(shlex.quote(x) for x in argv)} > {tag}.jsonl 2> {tag}.err"
    wrapped = f"bash -c {shlex.quote(inner)}; echo $? > {h['work']}/{tag}.rc"
    cmd = (XDG + f"rm -f {h['work']}/{tag}.jsonl {h['work']}/{tag}.err {h['work']}/{tag}.rc; "
           f"systemd-run --user --collect --unit={unit} bash -c {shlex.quote(wrapped)}")
    rc, text = on(ip, cmd)
    if rc != 0:
        raise RuntimeError(f"could not start {unit} on {ip}: {text[-300:]}")
    return unit


def events(cfg, ip, tag):
    rc, text = on(ip, f"cat {cfg['host']['work']}/{tag}.jsonl 2>/dev/null", 60)
    out = []
    for line in text.splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def exit_code(cfg, ip, tag):
    rc, text = on(ip, f"cat {cfg['host']['work']}/{tag}.rc 2>/dev/null", 30)
    return int(text.strip()) if text.strip().lstrip("-").isdigit() else None


def wait_event(cfg, ip, tag, kind, timeout):
    end = time.time() + timeout
    while time.time() < end:
        ev = events(cfg, ip, tag)
        if any(e.get("event") == kind for e in ev):
            return ev
        if exit_code(cfg, ip, tag) is not None:
            return ev
        time.sleep(5)
    return events(cfg, ip, tag)


def wait_exit(cfg, ip, tag, timeout):
    end = time.time() + timeout
    while time.time() < end:
        rc = exit_code(cfg, ip, tag)
        if rc is not None:
            return rc
        time.sleep(5)
    return None


def fetch(cfg, ip, tag, dest):
    os.makedirs(dest, exist_ok=True)
    for ext in ("jsonl", "err"):
        subprocess.run(["rsync", "-a", "-e", " ".join(SSH), f"pod@{ip}:{cfg['host']['work']}/{tag}.{ext}", os.path.join(dest, f"{tag}.{ext}")],
                       capture_output=True, timeout=600)


def binary(cfg):
    b = cfg["host"]["bundle"]
    return f"{b}/flerp-qwen35moe-generate", f"LD_LIBRARY_PATH={b}/lib"


def check_hold(cfg):
    p = subprocess.run(["pod-quad", "status"], capture_output=True, text=True, timeout=60)
    ips = [r["mgmt"] for r in cfg["host"]["ranks"]]
    if f"owner: {cfg['host']['hold_owner']}" not in p.stdout or not all(ip in p.stdout for ip in ips):
        sys.exit(f"no pod-quad hold of {ips} by {cfg['host']['hold_owner']!r}: nothing done")


def gpu_idle(cfg):
    busy = {}
    for r in cfg["host"]["ranks"]:
        rc, text = on(r["mgmt"], "cat /sys/class/drm/card*/device/mem_info_gtt_used; pgrep -af 'tensorfold-native|tf-dsv41-|strix_' | grep -v pgrep | head -3")
        nums = [int(x) for x in text.split() if x.isdigit()]
        if not nums or max(nums) > (2 << 30) or any(c.isalpha() for c in text.replace("\n", "")):
            busy[r["name"]] = text.strip()[:200]
    return busy


# --- M1 --------------------------------------------------------------------------------------------------------
def cmd_m1(cfg, a):
    os.makedirs(RES35, exist_ok=True)
    log = os.path.join(RES35, "m1.txt")
    r0, r1 = cfg["host"]["ranks"]
    run = cfg["run"]
    for r in (r0, r1):
        subprocess.run(["rsync", "-a", "-e", " ".join(SSH), os.path.join(ROOT, "gate", "m1_gather.py"), f"pod@{r['mgmt']}:{cfg['host']['work']}/m1_gather.py"], check=True)
    start(cfg, r1["mgmt"], "m1-r1", ["python3", "m1_gather.py", "--listen", r1["rail"], "--port", str(run["m1_port"]),
                                     "--bytes", str(run["m1_bytes"]), "--reps", str(run["m1_reps"]), "--warm", str(run["m1_warm"])])
    time.sleep(1)
    start(cfg, r0["mgmt"], "m1-r0", ["python3", "m1_gather.py", "--connect", r1["rail"], "--port", str(run["m1_port"]),
                                     "--bytes", str(run["m1_bytes"]), "--reps", str(run["m1_reps"]), "--warm", str(run["m1_warm"])])
    rc0, rc1 = wait_exit(cfg, r0["mgmt"], "m1-r0", 300), wait_exit(cfg, r1["mgmt"], "m1-r1", 60)
    fetch(cfg, r0["mgmt"], "m1-r0", RES35)
    fetch(cfg, r1["mgmt"], "m1-r1", RES35)
    ev = [e for e in events(cfg, r0["mgmt"], "m1-r0") if e.get("event") == "m1"]
    doc = {"rc": [rc0, rc1], "when": utc(), "transport": "TCP over the rail, host memory, TCP_NODELAY",
           "m1": ev[0] if ev else None}
    MP.save({"res": RES35}, "m1.json", doc)
    say(log, json.dumps({k: (doc["m1"] or {}).get(k) for k in ("median_ms", "p10_ms", "p90_ms", "min_ms", "max_ms")} | {"rc": doc["rc"]}))


# --- 35B over the rail -----------------------------------------------------------------------------------------
def cmd_q35(cfg, a):
    os.makedirs(RES35, exist_ok=True)
    log = os.path.join(RES35, "q35.txt")
    run, h = cfg["run"], cfg["host"]
    r0, r1 = h["ranks"]
    exe, env = binary(cfg)
    gguf = cfg["model"]["gguf"]
    ids = MP.load_json(cfg["model"]["prompt_ids"])
    names = list(ids)
    p1 = MP.load_json(cfg["model"]["p1_results"])["traced"]
    refs = {p: p1[p] for p in names}
    remote_ids = []
    for p in names:
        local = os.path.join(RES35, f"{p}.ids")
        with open(local, "w") as f:
            f.write(" ".join(str(i) for i in ids[p]))
        subprocess.run(["rsync", "-a", "-e", " ".join(SSH), local, f"pod@{r0['mgmt']}:{h['work']}/{p}.ids"], check=True)
        remote_ids.append(f"{h['work']}/{p}.ids")
    hosts = f"{r0['rail']},{r1['rail']}"
    doc, rows, why = {"runs": [], "hosts": hosts}, [], []
    for k, cuts_text in enumerate(run["tcp_configs"]):
        cuts = MP.py_plan(gguf, 2)["cuts"] if cuts_text == "even" else [int(x) for x in cuts_text.split(",")]
        plans = [MP.py_plan(gguf, 1, [cuts[r], cuts[r + 1]]) for r in range(2)]
        t0 = time.time()
        base = ["--staged", "tcp", "--model", gguf, "--context", str(run["context"]), "--max-tokens", str(run["max_tokens"]), "--world", "2",
                "--cuts", cuts_text, "--port", str(run["port"] + 10 * k), "--host", hosts, "--epoch", str(run["epoch"]),
                "--deadline-ms", str(run["deadline_ms"]), "--graph-modes", "on,off"] + list(h["reader_flags"])
        tags = [f"q35-{k}-r0", f"q35-{k}-r1"]
        say(log, f"config {cuts_text} cuts {cuts}: rank 0 on {r0['name']}, rank 1 on {r1['name']}, hosts {hosts}")
        start(cfg, r0["mgmt"], tags[0], [exe] + base + ["--rank", "0"] + [x for f in remote_ids for x in ("--ids-file", f)], env)
        ev0 = wait_event(cfg, r0["mgmt"], tags[0], "loaded", run["run_timeout_s"])
        say(log, f"rank 0 loaded: {next((e.get('placed', {}).get('read_gb_per_s') for e in ev0 if e.get('event') == 'loaded'), None)} GB/s")
        start(cfg, r1["mgmt"], tags[1], [exe] + base + ["--rank", "1"], env)
        rc0 = wait_exit(cfg, r0["mgmt"], tags[0], run["run_timeout_s"])
        rc1 = wait_exit(cfg, r1["mgmt"], tags[1], 120)
        for ip, t in ((r0["mgmt"], tags[0]), (r1["mgmt"], tags[1])):
            fetch(cfg, ip, t, RES35)
        e0, e1 = events(cfg, r0["mgmt"], tags[0]), events(cfg, r1["mgmt"], tags[1])
        configs = [f"2:{cuts_text}:{g}" for g in ("on", "off")]
        mrows, w = MS.judge_matrix(e0, configs, names, refs, "rail")
        rows += mrows
        why += w
        for r, (rc, ev) in enumerate(((rc0, e0), (rc1, e1))):
            if rc != 0:
                why.append(f"{cuts_text}: rank {r} exited {rc}")
            pl = next((e.get("placed") or {} for e in ev if e.get("event") == "loaded"), {})
            if pl.get("allocated_bytes") != plans[r]["ranks"][0]["device_bytes"]:
                why.append(f"{cuts_text}: rank {r} allocated {pl.get('allocated_bytes')}, plan {plans[r]['ranks'][0]['device_bytes']}")
            if not (pl.get("single_reader") and pl.get("direct_io")):
                why.append(f"{cuts_text}: rank {r} not the single direct reader")
        timing = [e for e in e1 if e.get("event") == "stage_timing"]
        over = MS.tcp_overheads(e0, timing)
        doc["runs"].append({"cuts": cuts, "cuts_text": cuts_text, "seconds": round(time.time() - t0, 1), "rc": [rc0, rc1],
                            "loaded": [next((e for e in ev if e.get("event") == "loaded"), None) for ev in (e0, e1)],
                            "overheads": over})
        say(log, f"config {cuts_text}: rc {rc0},{rc1}; " + ", ".join(f"{m['config']} {m['prompts_equal']}/{m['prompts']}" for m in mrows))
    doc.update(matrix=rows, failures=why, passed=not why and len(rows) == 2 * len(run["tcp_configs"]), when=utc())
    MP.save({"res": RES35}, "q35.json", doc)
    say(log, f"q35 {'PASS' if doc['passed'] else 'FAIL'}; failures {why[:6]}")


# --- Ornith ----------------------------------------------------------------------------------------------------
def cmd_probe(cfg, a):
    os.makedirs(RESORN, exist_ok=True)
    log = os.path.join(RESORN, "probe.txt")
    g = cfg["model"]["ornith_gguf"]
    out = {}
    for i, r in enumerate(cfg["host"]["ranks"]):
        rc, text = on(r["mgmt"], f"readlink -f {g}; findmnt -T $(readlink -f {g}) -o SOURCE,FSTYPE,OPTIONS -n | cut -c1-120; "
                                 f"dd if={g} of=/dev/null bs=16M count=64 skip={300 + 640 * i} iflag=direct 2>&1 | tail -1", 600)
        out[r["name"]] = text.strip()
        say(log, f"{r['name']}: {text.strip()}")
    MP.save({"res": RESORN}, "probe.json", out)


def cmd_ornith(cfg, a):
    os.makedirs(RESORN, exist_ok=True)
    log = os.path.join(RESORN, "ornith.txt")
    run, h, crit = cfg["run"], cfg["host"], cfg["criteria"]
    exe, env = binary(cfg)
    g = cfg["model"]["ornith_gguf"]
    ids = MP.load_json(cfg["model"]["prompt_ids"])
    long_ids = max(ids.values(), key=len)
    local = os.path.join(RESORN, "long.ids")
    with open(local, "w") as f:
        f.write(" ".join(str(i) for i in long_ids))
    stages = [list(s) for s in run["ornith_stages"]]
    checks = [list(c) for c in run["ornith_check"]]
    if a.blocks_per_stage:
        stages = [[s[0], s[0] + a.blocks_per_stage] for s in stages]
        checks = [[s[0], s[0] + a.blocks_per_stage // 2, s[1] - 1] for s in stages]
        say(log, f"REDUCED: {a.blocks_per_stage} blocks a stage: {stages}")
    recs, why = [], []
    started = []
    for i, (r, rng, layers) in enumerate(zip(h["ranks"], stages, checks)):
        subprocess.run(["rsync", "-a", "-e", " ".join(SSH), local, f"pod@{r['mgmt']}:{h['work']}/long.ids"], check=True)
        tag = f"orn-{i}"
        dump = f"{h['work']}/{tag}.dump"
        on(r["mgmt"], f"rm -f {dump}")
        args = [exe, "--model", g, "--placed"] + list(h["reader_flags"]) + ["--blocks", f"{rng[0]}:{rng[1]}", "--context", str(run["ornith_context"]),
                "--stage-bench", "--chunk-rows", str(run["chunk_rows"]), "--decode-steps", str(run["decode_steps"]), "--dump", dump,
                "--dump-layers", ",".join(str(x) for x in layers), "--dump-rows", ",".join(str(x) for x in run["check_rows"]),
                "--ids-file", f"{h['work']}/long.ids"]
        say(log, f"stage {i} blocks {rng} on {r['name']}: start")
        start(cfg, r["mgmt"], tag, args, env)
        started.append((r, tag, rng, layers, dump, time.time()))
        ev = wait_event(cfg, r["mgmt"], tag, "loaded", run["ornith_max_load_s"] + 600)
        ld = next((e for e in ev if e.get("event") == "loaded"), None)
        say(log, f"stage {i} loaded: {json.dumps((ld or {}).get('placed'))[:400] if ld else 'NOT LOADED'}")
    for i, (r, tag, rng, layers, dump, t0) in enumerate(started):
        rc = wait_exit(cfg, r["mgmt"], tag, 1800)
        fetch(cfg, r["mgmt"], tag, RESORN)
        ev = events(cfg, r["mgmt"], tag)
        rec = {"stage": i, "node": r["name"], "range": rng, "check_layers": layers, "rc": rc, "seconds": round(time.time() - t0, 1),
               "loaded": next((e for e in ev if e.get("event") == "loaded"), None),
               "events": [e for e in ev if e.get("event") in ("stage_bench", "stage_bench_skipped", "synthetic_input", "moe_dump")]}
        plan = MP.py_plan(g, 1, rng)
        pl = (rec["loaded"] or {}).get("placed") or {}
        f = []
        if rc != 0:
            f.append(f"exited {rc}")
        if pl.get("allocated_bytes") != plan["ranks"][0]["device_bytes"]:
            f.append(f"allocated {pl.get('allocated_bytes')}, plan {plan['ranks'][0]['device_bytes']}")
        f += MP.judge_bench(rec["events"], rng[0] == 0)
        ldump = os.path.join(RESORN, f"{tag}.dump")
        subprocess.run(["rsync", "-a", "-e", " ".join(SSH), f"pod@{r['mgmt']}:{dump}", ldump], capture_output=True, timeout=600)
        if os.path.isfile(ldump):
            p = subprocess.run(["ionice", "-c3", "nice", "-n", "19", "env", f"OMP_NUM_THREADS={h['block_check_threads']}",
                                f"OPENBLAS_NUM_THREADS={h['block_check_threads']}", sys.executable, os.path.join(ROOT, "gate", "moe_block_check.py"),
                                "--gguf", g, "--gguf-py", h["gguf_py"], "--dump", ldump], capture_output=True, text=True, timeout=3600)
            try:
                check = json.loads(p.stdout)
            except ValueError:
                check = {"error": (p.stderr or "no output")[-400:]}
            os.remove(ldump)
            MP.save({"res": RESORN}, f"block-check-{tag}.json", check)
            j = MP.judge_block_check(check, crit, crit["check_records_min"])
            rec["block_check"] = {k: v for k, v in j.items() if k != "detail"}
            f += [f"block check: {x}" for x in j["why"]]
        else:
            f.append("no dump")
        rec["failures"] = f
        why += [f"stage {i}: {x}" for x in f]
        recs.append(rec)
        say(log, f"stage {i} rc {rc}; failures {f[:4]}; block check {rec.get('block_check', {}).get('records')} records")
        on(r["mgmt"], f"rm -f {dump}")
    doc = {"stages": recs, "failures": why, "passed": not why, "reduced": bool(a.blocks_per_stage), "when": utc()}
    MP.save({"res": RESORN}, "ornith.json", doc)
    say(log, f"ornith {'PASS' if doc['passed'] else 'FAIL'}")


def cmd_cleanup(cfg, a):
    left = {}
    for r in cfg["host"]["ranks"]:
        on(r["mgmt"], XDG + f"systemctl --user stop '{cfg['host']['unit_prefix']}*' 2>/dev/null; true")
        time.sleep(2)
        rc, text = on(r["mgmt"], XDG + f"systemctl --user list-units --all --no-legend '{cfg['host']['unit_prefix']}*' | grep -c . ; "
                                       "pgrep -fc 'flerp-qwen35moe-generate|m1_gather' || true")
        left[r["name"]] = text.split()
    print(json.dumps({"left_units_and_procs": left}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["m1", "q35", "probe", "ornith", "cleanup"])
    ap.add_argument("--blocks-per-stage", type=int, default=0)
    a = ap.parse_args()
    with open(CONFIG, "rb") as f:
        cfg = tomllib.load(f)
    if a.cmd != "cleanup":
        check_hold(cfg)
        busy = gpu_idle(cfg) if a.cmd in ("q35", "ornith") else {}
        if busy:
            sys.exit(f"a worker's GPU is not idle: {busy}")
        for r in cfg["host"]["ranks"]:
            on(r["mgmt"], f"mkdir -p {cfg['host']['work']}")
    {"m1": cmd_m1, "q35": cmd_q35, "probe": cmd_probe, "ornith": cmd_ornith, "cleanup": cmd_cleanup}[a.cmd](cfg, a)


if __name__ == "__main__":
    main()
