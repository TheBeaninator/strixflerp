#!/usr/bin/env python3
"""One short pair window: tf-dsv41-lanes --profile 2 on both ranks of the serving pair, one 8,163-token prompt and a
~256-token greedy decode, for the per-phase prefill and decode table (gate/lane_profile.py) the next StrixFold work
starts from.

    python3 gate/prefill_dsv41_profile.py --plan        # read-only: the resolved pair and the plan
    python3 gate/prefill_dsv41_profile.py --minutes 15  # the window
    python3 gate/prefill_dsv41_profile.py --under-hold "agent FLERP-P4"   # inside a hold someone else owns

The pair is resolved from the live serving conf (gate/pair_nodes.py). One pod-quad hold as agent over both
ranks; the launcher's serving run taken down; the deployed build (DEPLOY) with a scratch copy of the serving conf
(RUN_ROOT and LEARN_DIR scratch, no prompt cache, drafts on, BENCH=0, GRAPHS=0 so the decode rounds run eager and are
timed: under GRAPHS=1 the phase timer sees no decode round), launched through a scratch lane_rank.sh that applies the
rank's WRAP as serving does. At the end: units stopped, scratch removed, THE HOLD RELEASED FIRST, then the pod's keeper
brings serving back (nothing started from here). Evidence: gate/results/prefill-dsv41/profile-<UTC>/.
"""
import argparse
import datetime
import json
import os
import shlex
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_profile  # noqa: E402
import pair_nodes  # noqa: E402
import prefill_dsv41_pair as P  # noqa: E402

ROOT = P.ROOT
F = A = URL = PAIR = None
S = "/home/pod/tfp/strixflerp-profile"
PROMPT = os.path.join(ROOT, "gate", "results", "prefill-dsv41", "pair-2026-11-11T1853Z", "prompt_ids.json")
DECODE_TOKENS = 256
OVER = {"RUN_ROOT": f"{S}/run", "LEARN_DIR": f"{S}/learn", "PROMPT_CACHE_GIB": "0", "DRAFTS": "1", "BENCH": "0",
        "LOGITS_SHA": "1", "GRAPHS": "0", "GRAPHS_EAGER": ""}
LANES_WAIT_S = 420   # load, the prompt, the decode
TAIL_S = 120         # stop, collect, remove the scratch, release
UNITS = ("strixflerp-profile-r0", "strixflerp-profile-r1")


def plan(minutes, ids):
    print(f"hold: OWNER=agent TTL={int(minutes * 60 + 600)} pod-quad hold {F} {A}; then {P.LAUNCHER} down; wait idle")
    print(f"deployed build {P.DEPLOY} (never written); scratch {S} on both workers")
    print(f"conf: {P.LIVE_CONF} + appended " + " ".join(f"{k}='{v}'" for k, v in OVER.items()))
    print(f"tools: {S}/lane-tools/dsv41-hip = symlinks to the deployment's, qual/lane_rank.sh a copy whose exec line is:")
    print("  " + lane_profile.WRAP_EXEC.replace("\n", "\n  "))
    print(f"rank 1 on {A}: lane_rank.sh conf 1 {S}/run/lane --profile 2   (unit {UNITS[1]})")
    print(f"rank 0 on {F}: lane_rank.sh conf 0 {S}/run/lane --requests {S}/conf/lane-requests.json --parallel 1 --logits-sha 1 --profile 2   (unit {UNITS[0]})")
    print(f"request: {len(ids) if ids else 'MISSING'} prompt ids from {os.path.relpath(PROMPT, ROOT)}, max_tokens {DECODE_TOKENS}, temperature 0")
    print(f"budget {minutes:g} min: lanes up to {LANES_WAIT_S} s, tail {TAIL_S} s; then stop, rm -rf {S}, pod-quad release FIRST, keeper restores (watched up to 8 min)")
    print("evidence: rank logs, rankR.cmd with its 'wrap (WRAPR, as serve_spawn): ...' line, phases.md (prefill per rank, decode on rank 0)")
    return 0 if ids else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--plan", action="store_true", help="print the resolved pair and the plan; change nothing")
    ap.add_argument("--under-hold", metavar="OWNER", help="run inside an existing pod-quad hold of both ranks by OWNER: "
                    "take none, release none, do not wait for serving (the hold's owner releases it)")
    pair_nodes.add_args(ap)
    a = ap.parse_args()
    pair = pair_nodes.from_args(a)
    pair_nodes.bind(pair, sys.modules[__name__], P)
    print(pair_nodes.describe(pair), flush=True)
    try:
        with open(PROMPT) as f:
            ids = json.load(f)
    except OSError:
        ids = None
    rc = plan(a.minutes, ids)
    if a.plan:
        sys.exit(rc)
    if rc != 0:
        sys.exit("no prompt ids: nothing done")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%MZ")
    out = os.path.join(ROOT, "gate", "results", "prefill-dsv41", "profile-" + stamp)
    os.makedirs(out, exist_ok=True)
    run = P.Run(out, time.time() + a.minutes * 60)
    doc = {"started": P.now(), "pair": pair, "health_before": run.health()}

    def flush():
        run.save("result.json", json.dumps(doc, indent=1, default=str))

    if a.under_hold:
        rc, text = run.sh("pod-quad status", 60)
        if f"owner: {a.under_hold}" not in text or not all(ip in text for ip in (F, A)):
            sys.exit(f"--under-hold: no hold of {F} {A} by {a.under_hold!r}: nothing done")
        doc["under_hold"] = a.under_hold
    else:
        for ip in (F, A):
            if run.sh(["pod-quad", "held", ip], 30)[0] == 0:
                sys.exit(f"{ip} is held by someone: nothing done")
        rc, text = run.sh(f"OWNER=agent REASON={shlex.quote('strixflerp lanes --profile 2 phase table (agreed with the pair owner)')} "
                          f"TTL={int(a.minutes * 60 + 600)} pod-quad hold {F} {A}", 900)
        if rc != 0:
            sys.exit(f"pod-quad hold refused: {text[-300:]}")
        doc["hold_taken"] = P.now()
    try:
        run.sh(f"KANBAN_ACTOR=agent {P.LAUNCHER} down", 400)
        if not run.idle(240):
            raise RuntimeError("the pair did not go idle under the hold")
        run.save("profile.conf", P.conf_text(OVER))
        local = os.path.join(out, "lane-requests.json")
        with open(local, "w") as f:
            json.dump({"requests": [{"name": "p8k", "prompt": ids, "max_tokens": DECODE_TOKENS, "temperature": 0}]}, f)
        for ip in (F, A):
            run.on(ip, f"mkdir -p {S}/conf {S}/run {S}/learn")
            run.sh(["rsync", "-a", "-e", " ".join(P.SSH), os.path.join(out, "profile.conf"), f"pod@{ip}:{S}/conf/profile.conf"])
        run.sh(["rsync", "-a", "-e", " ".join(P.SSH), local, f"pod@{F}:{S}/conf/lane-requests.json"])
        err = lane_profile.stage(run, (F, A), P.DEPLOY, f"{S}/lane-tools", P.SSH, out)
        if err:
            raise RuntimeError(err)
        got = lane_profile.run_lanes(run, F, A, f"{S}/lane-tools", f"{S}/run/lane", f"{S}/conf/profile.conf",
                                     f"{S}/conf/lane-requests.json", P.XDG, min(LANES_WAIT_S, max(60, run.left() - TAIL_S)), UNITS)
        doc["lanes_seconds"], doc["wrap_applied"] = got["seconds"], got["wrap"]
        for (r, ext), t in got["texts"].items():
            run.save(f"lane-rank{r}.{ext}", t)
        parsed = lane_profile.parse({r: got["texts"][(r, "jsonl")] for r in (0, 1)})
        doc["phases"] = parsed
        run.save("phases.md", lane_profile.table(parsed))
        flush()
    except Exception as e:
        doc["error"] = repr(e)[:400]
        run.say("error: " + doc["error"])
    finally:
        for ip in (F, A):
            run.on(ip, P.XDG + f"systemctl --user stop {' '.join(UNITS)} 2>/dev/null; true", 120)
        run.idle(120)
        for ip in (F, A):
            run.on(ip, f"rm -rf {S}")
        if a.under_hold:
            doc["units_left"] = {ip: run.on(ip, P.XDG + f"systemctl --user is-active {' '.join(UNITS)} 2>/dev/null | grep -c '^active' || true", 60)[1] for ip in (F, A)}
            flush()
            print(json.dumps({k: doc.get(k) for k in ("under_hold", "wrap_applied", "units_left", "error")}, indent=1))
            return
        rc, _ = run.sh("OWNER=agent pod-quad release", 300)
        doc["hold_released"], doc["release_rc"] = P.now(), rc
        time.sleep(2)
        doc["held_after"] = {ip: run.sh(["pod-quad", "held", ip], 30)[0] == 0 for ip in (F, A)}
        restored, t0 = None, time.time()
        run.say("hold released; waiting for the pod's keeper to bring serving back (no start from here)")
        while time.time() - t0 < 480:
            h = run.health()
            if h and h.get("model") == "dsv41" and h.get("warming") is False:
                restored = h
                break
            time.sleep(5)
        doc["health_after"], doc["restore_wait_s"] = restored, round(time.time() - t0)
        if not restored:
            run.say("SERVING DID NOT COME BACK in 8 minutes after the release: tell the pair's owner (nothing was started from here)")
        flush()
    print(json.dumps({k: doc.get(k) for k in ("hold_taken", "hold_released", "wrap_applied", "health_after", "error")}, indent=1))


if __name__ == "__main__":
    main()
