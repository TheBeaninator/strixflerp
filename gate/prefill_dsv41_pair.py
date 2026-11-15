#!/usr/bin/env python3
"""The pair window of docs/PREFILL-DSV41.md section 6, run from control-node against workers F and A: the deployed
reference build with a scratch copy of the serving conf (no learned states, no prompt cache, drafts on), one long
prompt, three arms: A as served, B with the prefill split off, C the lane tool with --profile 2 on both ranks.

    python3 gate/prefill_dsv41_pair.py --minutes 20

One pod-quad hold as agent over F and A; the pod launcher's serving run is taken down and brought back at the
end (strixfold-pair.sh up, started just after a keep tick: under a hold the keep tick takes the launcher's own run
down), health confirmed, then the hold released. Arms start as transient user units on the workers with run ids of
their own (not pair-*) and a RUN_ROOT of their own, by the deployment's first_light.sh / qual/lane_rank.sh.
Evidence: gate/results/prefill-dsv41/pair-<UTC>/.
"""
import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_profile  # noqa: E402
import pair_nodes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# rank 0's and rank 1's mgmt addresses and the server: bound at run time from the live serving conf (pair_nodes.bind)
F = A = URL = PAIR = None
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
LAUNCHER = "/data/pod/tools/strixfold-pair.sh"
LIVE_CONF = "/data/pod/plans/strixfold/tp2-128k.conf"
S = "/home/pod/tfp/strixflerp-prefill"
DEPLOY = "/home/pod/tfp/deploy/f64b5ff2e28b"
XDG = "export XDG_RUNTIME_DIR=/run/user/$(id -u); "
WORDS = ["amber", "harbour", "ledger", "copper", "signal", "orchard", "granite", "lantern", "meadow", "pilot", "quarry",
         "ribbon", "saddle", "timber", "valley", "walnut", "anchor", "basket", "candle", "depot", "engine", "ferry"]


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Run:
    def __init__(self, out, deadline):
        self.out, self.deadline, self.rows = out, deadline, []
        self.log = os.path.join(out, "commands.txt")

    def say(self, text):
        with open(self.log, "a") as f:
            f.write(f"{now()} {text}\n")
        print(f"{now()} {text}", flush=True)

    def sh(self, cmd, timeout=300, quiet=False):
        shell = isinstance(cmd, str)
        if not quiet:
            self.say("RUN " + (cmd if shell else " ".join(cmd)))
        try:
            p = subprocess.run(cmd, shell=shell, capture_output=True, text=True, timeout=timeout)
            rc, text = p.returncode, (p.stdout + p.stderr).strip()
        except subprocess.TimeoutExpired:
            rc, text = 124, "TIMEOUT"
        if not quiet:
            with open(self.log, "a") as f:
                f.write(text[-3000:] + f"\n(rc {rc})\n")
        return rc, text

    def on(self, ip, cmd, timeout=120, quiet=False):
        return self.sh(SSH + [f"pod@{ip}", cmd], timeout, quiet)

    def left(self):
        return self.deadline - time.time()

    def health(self):
        try:
            with urllib.request.urlopen(URL + "/health", timeout=5) as r:
                return json.loads(r.read())
        except Exception:
            return None

    def idle(self, seconds=200):
        t0 = time.time()
        while time.time() - t0 < seconds:
            busy = []
            for ip in (F, A):
                rc, text = self.on(ip, "cat /sys/class/drm/card*/device/mem_info_gtt_used; echo ==; pgrep -af 'tensorfold-native|tf-dsv41-' | grep -v pgrep | head -3", 30, True)
                gtt, _, procs = text.partition("==")
                nums = [int(x) for x in gtt.split() if x.isdigit()]
                if "==" not in text or not nums or max(nums) > (1 << 30) or procs.strip():
                    busy.append(ip)
            if not busy:
                return True
            time.sleep(4)
        return False

    def save(self, name, text):
        with open(os.path.join(self.out, name), "w") as f:
            f.write(text)


def conf_text(over):
    with open(LIVE_CONF) as f:
        text = f.read()
    base = {"BENCH": "0", "LOGITS_SHA": "1", "PROMPT_CACHE_GIB": "0", "LEARN_DIR": "", "DRAFTS": "1",
            "RUN_ROOT": "$HOME/tfp/strixflerp-prefill/run"}
    base.update(over)
    return text + "\n# --- strixflerp prefill window: appended, a later assignment wins ---\n" + "".join(f"{k}='{v}'\n" for k, v in base.items())


def body_of(n):
    x, rows = 12345, []
    for i in range(n):
        ws = []
        for _ in range(6):
            x = (1103515245 * x + 12345) % (1 << 31)
            ws.append(WORDS[(x >> 16) % len(WORDS)])
        rows.append(f"Record {i + 1}: the {ws[0]} {ws[1]} was moved from the {ws[2]} {ws[3]} to the {ws[4]} {ws[5]}.")
    return {"temperature": 0, "max_tokens": 4, "stream": False, "return_token_ids": True,
            "messages": [{"role": "system", "content": "You answer questions about the records you are given."},
                         {"role": "user", "content": "\n".join(rows) + "\n\nWhich record mentions a harbour first? Answer with its number."}]}


def post(path, body, timeout=300):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), round(time.time() - t0, 3)


def server_arm(run, tag, conf, state, reps=2):
    """One boot of the deployed server with `conf`, the prompt asked `reps` times; a row of results."""
    row = {"arm": tag, "conf": conf, "requests": [], "started": now()}
    for attempt in (1, 2):
        rid = "pf-" + tag + "-" + datetime.datetime.now(datetime.timezone.utc).strftime("%H%M%S")
        unit = "strixflerp-prefill-up"
        run.on(A, XDG + f"systemd-run --user --collect --unit={unit} --property=KillMode=process bash -c "
               + shlex.quote(f"cd {DEPLOY}/tools/dsv41-hip && RUN_ID={rid} TF_SERVE_CONF={S}/conf/{conf} ./first_light.sh up"))
        t0, ok = time.time(), False
        while time.time() - t0 < min(240, max(60, run.left() - 240)):
            h = run.health()
            active = run.on(A, XDG + f"systemctl --user is-active -q {unit}", 20, True)[0] == 0
            if h and not active:
                ok = True
                break
            if not h and not active and time.time() - t0 > 30:
                break
            time.sleep(4)
        row.update(run_id=rid, startup_s=round(time.time() - t0, 1), attempts=attempt)
        down = f"cd {DEPLOY}/tools/dsv41-hip && RUN_ID={rid} TF_SERVE_CONF={S}/conf/{conf} ./first_light.sh down"
        if ok:
            break
        run.say(f"arm {tag}: start {attempt} did not reach health")
        _, t1 = run.on(A, f"tail -n 30 {S}/run/{rid}/rank1.log {S}/run/{rid}/*.txt 2>/dev/null", 30, True)
        _, t0_ = run.on(F, f"tail -n 40 {S}/run/{rid}/rank0.log 2>/dev/null", 30, True)
        run.save(f"{tag}-failed-start-{attempt}.txt", t1 + "\n==== rank 0\n" + t0_ + "\n")
        run.on(A, XDG + f"systemctl --user stop {unit} 2>/dev/null; " + down, 300)
        run.idle()
    if not ok:
        row["error"] = "did not reach health in 2 starts"
        return row
    try:
        row["health"] = run.health()
        if "body" not in state:
            b = body_of(385)
            n = len(post("/tokenize", {"messages": b["messages"]}, 60)[0]["tokens"])
            b = body_of(int(385 * 8170 / n))
            state["body"], state["ids"] = b, post("/tokenize", {"messages": b["messages"]}, 60)[0]["tokens"]
            run.save("request.json", json.dumps(b))
            run.save("prompt_ids.json", json.dumps(state["ids"]))
        for rep in range(reps):
            reply, wall = post("/v1/chat/completions", state["body"])
            tf = reply.get("tensorfold", {}) if isinstance(reply.get("tensorfold"), dict) else {}
            row["requests"].append({"rep": rep, "wall_s": wall, "usage": reply.get("usage"), "tensorfold": tf,
                                    "text": reply["choices"][0].get("message", {}).get("content")})
            run.say(f"arm {tag} rep {rep}: wall {wall} s usage {reply.get('usage')}")
    except Exception as e:                                          # a dead server is a result, not the end of the window
        row["error"] = repr(e)[:300]
        run.say(f"arm {tag}: {row['error']}")
    for ip, r in ((F, 0), (A, 1)):
        _, text = run.on(ip, f"cat {S}/run/{rid}/rank{r}.log 2>/dev/null", 60, True)
        run.save(f"{tag}-rank{r}.log", text)
        if r == 0:
            row["logits_sha"] = [json.loads(l)["sha256"] for l in text.splitlines() if l.startswith('{"prompt_logits"') and f'"rows": {len(state.get("ids", []))}' in l]
            row["pass_lines"] = [l[l.find("prompt pass"):][:260] for l in text.splitlines() if "prompt pass:" in l][-reps:]
            row["rank1_frame_warnings"] = len([l for l in text.splitlines() if "frame" in l and "spent" in l])
    run.on(A, XDG + f"systemctl --user stop {unit} 2>/dev/null; " + down, 300)
    row["idle_after"] = run.idle()
    row["ended"] = now()
    return row


def lanes_arm(run, tag, conf, state):
    """tf-dsv41-lanes --profile 2 on both ranks through a scratch copy of the deployment's qual/lane_rank.sh that
    applies the rank's WRAP as serving does (gate/lane_profile.py: without it the runtime's dlopen of libamdhip64.so.7
    fails on the workers, DriverUnavailable); the phase lines and their table."""
    row = {"arm": tag, "conf": conf, "started": now()}
    d = f"{S}/run/lane-{tag}"
    err = lane_profile.stage(run, (F, A), DEPLOY, f"{S}/lane-tools", SSH, run.out)
    if err:
        row["error"] = err
        return row
    reqs = {"requests": [{"name": "p8k", "prompt": state["ids"], "max_tokens": 4, "temperature": 0}]}
    local = os.path.join(run.out, "lane-requests.json")
    with open(local, "w") as f:
        json.dump(reqs, f)
    run.sh(["rsync", "-a", "-e", " ".join(SSH), local, f"pod@{F}:{S}/conf/lane-requests.json"])
    got = lane_profile.run_lanes(run, F, A, f"{S}/lane-tools", d, f"{S}/conf/{conf}", f"{S}/conf/lane-requests.json", XDG,
                                 min(300, max(60, run.left() - 230)))
    row["seconds"], row["wrap_applied"] = got["seconds"], got["wrap"]
    for (r, ext), text in got["texts"].items():
        run.save(f"{tag}-lane-rank{r}.{ext}", text)
        if ext == "jsonl":
            row[f"rank{r}"] = [json.loads(l) for l in text.splitlines() if l.startswith("{") and ("prefill" in l or "error" in l or "loaded_s" in l or "token_sha" in l)]
        if ext == "err":
            row[f"rank{r}_err_tail"] = text.splitlines()[-6:]
    parsed = lane_profile.parse({r: got["texts"][(r, "jsonl")] for r in (0, 1)})
    run.save(f"{tag}-phases.md", lane_profile.table(parsed))
    row["phases"] = parsed
    row["idle_after"] = run.idle()
    row["ended"] = now()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--plan", action="store_true", help="print the resolved pair and change nothing")
    pair_nodes.add_args(ap)
    a = ap.parse_args()
    pair_nodes.bind(pair_nodes.from_args(a), sys.modules[__name__])
    print(pair_nodes.describe(PAIR), flush=True)
    if a.plan:
        return 0
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%MZ")
    out = os.path.join(ROOT, "gate", "results", "prefill-dsv41", "pair-" + stamp)
    os.makedirs(out, exist_ok=True)
    run = Run(out, time.time() + a.minutes * 60)
    doc = {"started": now(), "release_by": datetime.datetime.fromtimestamp(run.deadline, datetime.timezone.utc).strftime("%H:%M:%SZ"),
           "pair": PAIR, "health_before": run.health(), "arms": []}

    def flush():
        run.save("result.json", json.dumps(doc, indent=1))

    for ip in (F, A):
        if run.sh(["pod-quad", "held", ip], 30)[0] == 0:
            sys.exit(f"{ip} is held by someone: nothing done")
    rc, text = run.sh(f"OWNER=agent REASON={shlex.quote('strixflerp prefill A/B/C window (agreed with the pair owner)')} "
                      f"TTL={int(a.minutes * 60 + 420)} pod-quad hold {F} {A}", 900)
    if rc != 0:
        sys.exit(f"pod-quad hold refused: {text[-300:]}")
    doc["hold_taken"] = now()
    state = {}
    try:
        run.sh(f"KANBAN_ACTOR=agent {LAUNCHER} down", 400)
        if not run.idle(240):
            raise RuntimeError("the pair did not go idle under the hold")
        for name, over in (("A.conf", {}), ("B.conf", {"PREFILL_SPLIT": ""})):
            run.save(name, conf_text(over))
        for ip in (F, A):
            run.on(ip, f"mkdir -p {S}/conf {S}/run")
            for name in ("A.conf", "B.conf"):
                run.sh(["rsync", "-a", "-e", " ".join(SSH), os.path.join(out, name), f"pod@{ip}:{S}/conf/{name}"])
        for tag, kind, conf in (("A", "server", "A.conf"), ("B", "server", "B.conf"), ("C", "lanes", "A.conf"), ("A2", "server", "A.conf")):
            need = 330 if kind == "server" else 400
            if run.left() < need + 200:
                doc.setdefault("skipped", []).append(f"{tag}: {int(run.left())} s to the release time")
                run.say(doc["skipped"][-1])
                continue
            if kind == "lanes" and "ids" not in state:
                doc.setdefault("skipped", []).append("C: no prompt ids (no server arm answered)")
                continue
            doc["arms"].append(server_arm(run, tag, conf, state) if kind == "server" else lanes_arm(run, tag, conf, state))
            flush()
    except Exception as e:
        doc["error"] = repr(e)[:400]
        run.say("error: " + doc["error"])
    finally:
        # stop anything of this window, bring serving back, confirm, release
        for ip, units in ((A, "strixflerp-prefill-up strixflerp-prefill-r1"), (F, "strixflerp-prefill-r0")):
            run.on(ip, XDG + f"systemctl --user stop {units} 2>/dev/null; "
                   "for u in $(systemctl --user list-units 'tfp-*pf-*' --state=active --no-legend --plain | cut -d' ' -f1); do systemctl --user stop $u; echo stopped $u; done; "
                   "true", 120)
        run.idle(120)
        for ip in (F, A):
            run.on(ip, f"rm -rf {S}")
        restored = None
        for attempt in (1, 2):
            # a keep tick under the hold takes the launcher's own run down: start right after one (every 2 min at :26)
            wait = (86 - (time.time() % 120)) % 120
            if wait > 100:
                wait = 0
            run.say(f"restore attempt {attempt}: waiting {int(wait) + 3} s for the launcher's keep tick to pass")
            time.sleep(wait + 3)
            run.sh(f"KANBAN_ACTOR=agent {LAUNCHER} up", 300)
            t0 = time.time()
            while time.time() - t0 < 112:
                h = run.health()
                if h and h.get("model") == "dsv41" and h.get("warming") is False:
                    restored = h
                    break
                time.sleep(4)
            if restored:
                break
            run.say("serving did not come back before the next keep tick")
        doc["health_after"], doc["restored_at"] = restored, now()
        rc, text = run.sh(f"OWNER=agent pod-quad release", 300)
        doc["hold_released"], doc["release_rc"] = now(), rc
        time.sleep(2)
        doc["held_after"] = {ip: run.sh(["pod-quad", "held", ip], 30)[0] == 0 for ip in (F, A)}
        doc["left_on_workers"] = {ip: run.on(ip, XDG + f"ls -d {S} 2>&1 | head -2; systemctl --user list-units 'strixflerp-prefill*' --no-legend --plain | head; pgrep -af 'strixflerp-prefill' | grep -v pgrep | head -3", 60)[1] for ip in (F, A)}
        doc["health_final"] = run.health()
        flush()
    print(json.dumps({k: doc.get(k) for k in ("hold_taken", "hold_released", "health_after", "health_final", "skipped", "error")}))


if __name__ == "__main__":
    main()
