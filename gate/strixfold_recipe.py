#!/usr/bin/env python3
"""Recipe gate for a StrixFold recipe (SPEC.md section 9, phase 3): `flerp gate recipe dsv41-flash.strixfold-tp2`.

Stages (run on control-node; evidence under gate/results/<recipe>/):
  check   flerp check passes for the strixfold origin and the recipe
  origin  every vendored file is byte-identical to the pod's git tree at ORIGIN.toml's commit (exported with git
          archive into build/), and the tree is covered
  build   the recipe assembles from the catalogue alone and builds with the origin's own Zig build graph; the kit's
          manifest sha256 equals the reference deployment's
  pair    on workers F and A, under a hold: the reference deployment and the assembled one, one after the other,
          DSpark drafts on and off, answer the same greedy requests. Pass: the generated token ids are equal request
          for request, the receipts agree (each prompt's logits sha256 and every verify window's drawn tokens by
          position, from rank 0's log with TF_DS_LOGITS_SHA and TF_DS_TRACE_ROUNDS), and decode tokens per second
          agree within the tolerance. Then the executor ABI conformance driver (recipe [abi]) runs as rank 0 and
          must reproduce the assembled server's tokens.

The pair stage touches nothing unless the pair is free: no lease or hold on F or A, no engine process, GPU memory
idle, the pod's own launcher not mid-start, no other session driving the pair (a start or stop "by hand" in the
launcher's log within the quiet period, or a script on this host that names the launcher or the pair's runs).
With --take-from-launcher it also runs when the one thing on the pair is the launcher's own serving run. It holds both workers with `pod-quad hold` in one call (owner and
reason from the config; this takes the pod-lease worker leases and is what the pod's pair launcher yields to),
starts and stops the reference with the pod's launcher, stops only what it started, checks GPU memory is back to
idle, removes its scratch directory on each worker and releases the hold. A reference that does not reach health in
the configured attempts ends the stage with its logs; nothing is debugged or restarted beyond that.
`--plan` prints what the pair stage would run and changes nothing.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shlex
import socket
import statistics
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from flerplib import origin as origin_mod        # noqa: E402
from flerplib.catalog import Catalog             # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "gate"))
import pair_nodes                                # noqa: E402  (which workers are the pair: the live serving conf)
import summary as SM                             # noqa: E402  (summary.json / summary.md, config_sha256)

STAGES = ("check", "origin", "build", "pair")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
WORDS = ("amber", "basalt", "cedar", "delta", "ember", "fjord", "garnet", "harbour", "iris", "juniper", "kelp",
         "lantern", "meadow", "nickel", "orchard", "pewter", "quarry", "russet", "sable", "thistle")


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sh(cmd, log, timeout=None, env=None, cwd=ROOT):
    """Run a command, append its output to `log`, return (rc, seconds)."""
    t0 = time.time()
    with open(log, "ab") as f:
        f.write(("$ " + (cmd if isinstance(cmd, str) else " ".join(shlex.quote(c) for c in cmd)) + "\n").encode())
        f.flush()
        try:
            rc = subprocess.call(cmd, shell=isinstance(cmd, str), cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                                 timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            f.write(b"\nTIMEOUT\n")
            rc = 124
    return rc, round(time.time() - t0, 1)


def out_of(cmd, timeout=60):
    """(rc, stdout + stderr) of a short read-only command."""
    try:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tail(path, n=12):
    try:
        with open(path, errors="replace") as f:
            return [line.rstrip("\n")[:400] for line in f.readlines()[-n:]]
    except OSError:
        return []


# --- check, origin, build -------------------------------------------------------------------------------------
def stage_check(ctx):
    log = os.path.join(ctx["res"], "check.txt")
    open(log, "w").close()
    rc, _ = sh([sys.executable, "flerp", "check", "--origin", ctx["cfg"]["host"]["origin"]], log)
    return {"pass": rc == 0, "evidence": "check.txt", "last_line": (tail(log, 1) or [""])[0]}


def stage_origin(ctx):
    h = ctx["cfg"]["host"]
    data = ctx["cat"].origin_data(h["origin"])
    commit, checkout = data["commit"], h["reference_checkout"]
    export = os.path.join(ctx["scratch"], f"origin-{commit}")
    log = os.path.join(ctx["res"], "diff-origin.txt")
    open(log, "w").close()
    rc, head = out_of(["git", "-C", checkout, "rev-parse", "HEAD"])
    out = {"checkout": checkout, "commit": commit, "head": head, "evidence": "diff-origin.txt",
           "export": os.path.relpath(export, ROOT)}
    # the pod tree is a shared checkout with build products in it: compare against a clean export of the commit
    sh(f"rm -rf {shlex.quote(export)} && mkdir -p {shlex.quote(export)} && "
       f"git -C {shlex.quote(checkout)} archive {commit} | tar -x -C {shlex.quote(export)}", log)
    res = origin_mod.diff_origin(ctx["cat"], h["origin"], export)
    with open(os.path.join(ctx["res"], "diff-origin.json"), "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    with open(log, "a") as f:
        for mid, path, why in res["problems"]:
            f.write(f"DIFF {mid} {path}: {why}\n")
        f.write(f"diff-origin origin {h['origin']} vs git archive {commit} of {checkout}: {res['modules']} module(s), "
                f"{res['identical']}/{res['files']} files byte-identical, {len(res['problems'])} problem(s), "
                f"{res['unharvested']} origin files listed as unharvested\n")
    out.update(modules=res["modules"], files=res["files"], identical=res["identical"], problems=len(res["problems"]),
               unharvested=res["unharvested"])
    out["pass"] = rc == 0 and not res["problems"] and res["files"] > 0
    return out


def build_tree(ctx, recipe_name, log):
    """Assemble and build one recipe. Returns (rc, tree, seconds)."""
    tree = os.path.join(ROOT, "build", recipe_name)
    rc, _ = sh([sys.executable, "flerp", "assemble", recipe_name], log)
    secs = None
    if rc == 0:
        rc, secs = sh([os.path.join(tree, "build.sh"), f"-j{ctx['cfg']['host'].get('build_jobs', 8)}"], log, timeout=3600)
    return rc, tree, secs


def stage_build(ctx):
    h = ctx["cfg"]["host"]
    log = os.path.join(ctx["res"], "build.txt")
    open(log, "w").close()
    out = {"evidence": "build.txt", "started": now()}
    rc, tree, out["build_s"] = build_tree(ctx, ctx["recipe"].name, log)
    out["tree"] = os.path.relpath(tree, ROOT)
    if rc == 0:
        with open(os.path.join(tree, "assembly.json")) as f:
            asm = json.load(f)
        with open(os.path.join(tree, "built.json")) as f:
            built = json.load(f)
        with open(os.path.join(tree, "build.sh")) as f:
            script = f.read()
        for name in ("assembly.json", "built.json"):
            with open(os.path.join(tree, name), "rb") as src, open(os.path.join(ctx["res"], name), "wb") as dst:
                dst.write(src.read())
        kit = [k for k in built if k.endswith("/manifest.json")]
        out["files_in_tree"] = len(asm["files"])
        out["references_origin_checkout"] = h["reference_checkout"] in script or h["reference_checkout"] in json.dumps(asm)
        out["binaries"] = {k: v for k, v in built.items() if "/bin/" in k}
        out["kit_objects"] = sum(1 for k in built if k.endswith(".hsaco") and "/share/" in k)
        with open(os.path.join(tree, "tree/zig-out", kit[0])) as f:
            named = {o["file"] for o in json.load(f)["objects"]} if kit else set()
        on_disk = {k.split("/dsv41/", 1)[1] for k in built if k.endswith(".hsaco") and "/share/" in k}
        out["kit_directory_is_the_manifest"] = bool(named) and named == on_disk
        out["kit_manifest_sha256"] = built[kit[0]]["sha256"] if kit else None
        out["kit_equals_reference_deployment"] = out["kit_manifest_sha256"] == h["reference_kit_manifest_sha256"]
        out["hipcc_output_pins"] = len(asm.get("hipcc_output_pins", {}))
    out.update(rc=rc, finished=now())
    out["pass"] = (rc == 0 and bool(out.get("binaries")) and out.get("kit_equals_reference_deployment") is True
                   and out.get("kit_directory_is_the_manifest") is True and out.get("references_origin_checkout") is False)
    return out


def lease_lines_for(nodes, lease_text):
    """The pod-lease status lines that name either rank of the pair. Matched against the LIVE pair (gate/pair_nodes.py:
    each rank's letter, mgmt and rail address), never a hardcoded F/A pattern: since the pair moved to D (NODE1=D,
    2026-11-11), a worker:D lease was invisible to "is the pair free" -- the old pattern was
    worker:[FA]|192.0.2.5[04] (found by card b390d523b9, 2026-11-16)."""
    names = set()
    for r in nodes.values():
        if isinstance(r, dict):
            for key in ("letter", "mgmt", "rail"):
                if r.get(key):
                    names.add(str(r[key]))
    hits = []
    for line in (lease_text or "").splitlines():
        if any(re.search(r"(?<![\w.-])" + re.escape(n) + r"(?![\w-])", line) for n in names):
            hits.append(line)
    return hits


# --- the pair -------------------------------------------------------------------------------------------------
class Pair:
    def __init__(self, ctx, plan=False):
        self.ctx, self.p, self.run, self.plan = ctx, ctx["cfg"]["pair"], ctx["cfg"]["run"], plan
        # a plan writes its log beside the scratch, never into gate/results (a plan is not evidence)
        self.log = (os.path.join(ctx["scratch"], "plan-pair.txt") if plan else os.path.join(ctx["res"], "pair.txt"))
        # the live pair: the pod's serving conf (gate/pair_nodes.py), command-line overrides first; the [pair] rank0,
        # rank1, *_rail and serve_url of the gate's toml are the record of earlier runs, kept in self.recorded
        self.recorded = {k: self.p.get(k) for k in ("rank0", "rank1", "rank0_rail", "rank1_rail", "serve_url")}
        a = ctx.get("pair_args")
        self.nodes = pair_nodes.from_args(a) if a else pair_nodes.load()
        self.f, self.a = self.nodes["rank0"]["mgmt"], self.nodes["rank1"]["mgmt"]
        self.p = dict(self.p, rank0=self.f, rank1=self.a, rank0_rail=self.nodes["rank0"]["rail"],
                      rank1_rail=self.nodes["rank1"]["rail"], serve_url=self.nodes["serve_url"])
        self.started = []                    # what this stage started, for the teardown

    def say(self, text):
        with open(self.log, "a") as f:
            f.write(f"{now()} {text}\n")
        print("   " + text, flush=True)

    def on(self, ip, cmd, timeout=120):
        """A command on a worker: (rc, output). In --plan mode only read-only commands (ro=) run."""
        return out_of(SSH + [f"pod@{ip}", cmd], timeout)

    def do(self, what, cmd, timeout=600):
        """A command that changes something: logged; skipped in --plan mode."""
        self.say(("PLAN " if self.plan else "RUN  ") + what + ": " + (cmd if isinstance(cmd, str) else " ".join(cmd)))
        if self.plan:
            return 0, ""
        rc, out = out_of(cmd, timeout)
        with open(self.log, "a") as f:
            f.write(out + f"\n(rc {rc})\n")
        return rc, out

    # -- read-only state
    def worker_state(self, ip):
        rc, out = self.on(ip, "hostname; cat /sys/class/drm/card*/device/mem_info_gtt_used; "
                              f"pgrep -a -f {shlex.quote(self.p['busy_pattern'])} | grep -v pgrep | cut -c1-200")
        lines = out.splitlines()
        gtt = [int(x) for x in lines[1:] if x.isdigit()]
        procs = [x for x in lines[1:] if not x.isdigit()]
        return {"reachable": rc == 0, "host": lines[0] if lines else "", "gtt_used_bytes": sum(gtt),
                "engine_processes": procs}

    def state(self):
        st = {"time": now(), "workers": {self.f: self.worker_state(self.f), self.a: self.worker_state(self.a)}}
        _, st["leases"] = out_of(["pod-lease", "status"])
        st["holds"] = {ip: out_of(["pod-quad", "held", ip])[1] for ip in (self.f, self.a)}
        st["launcher"] = out_of([self.p["launcher"], "status"])[1]
        rc, _ = self.on(self.a, "XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user is-active -q strixfold-pair-up")
        st["launcher_starting"] = rc == 0
        try:
            with urllib.request.urlopen(self.p["serve_url"] + "/health", timeout=5) as r:
                st["serving"] = r.status == 200
        except (urllib.error.URLError, OSError):
            st["serving"] = False
        # other sessions at work on the pair: starts and stops "by hand" in the launcher's log, and processes on
        # this host that name the pair's launcher, its conf or its port
        st["by_hand"] = []
        try:
            with open(self.p["launcher_log"], errors="replace") as f:
                lines = f.readlines()[-200:]
        except OSError:
            lines = []
        t_now = datetime.datetime.now(datetime.timezone.utc)
        for line in lines:
            m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)Z (down: by hand.*)", line)
            if m:
                age = (t_now - datetime.datetime.fromisoformat(m.group(1) + "+00:00")).total_seconds()
                if age <= self.p["quiet_s"]:
                    st["by_hand"].append(f"{int(age)} s ago: {m.group(2).strip()[:160]}")
        _, procs = out_of("pgrep -af 'strixfold-pair.sh (up|down)|tfp/run/serve128k|/data/pod/plans/[0-9a-f]{10}/[a-z0-9_]+\\.sh' | grep -v pgrep")
        # a script being run (bash <script>), not a command line that merely names one (someone reading the file)
        st["sessions"] = [x[:200] for x in procs.splitlines() if "strixflerp" not in x and str(os.getpid()) not in x.split()[:1]
                          and re.match(r"\d+ (/usr)?(/bin/)?(bash|sh) (/data/pod/plans/[0-9a-f]{10}/|/data/pod/tools/strixfold-pair\.sh (up|down))", x)]
        return st

    def free(self, st):
        """Why the pair is not free, or [] when it is."""
        why = []
        for ip, w in st["workers"].items():
            if not w["reachable"]:
                why.append(f"{ip} does not answer ssh")
            if w["engine_processes"]:
                why.append(f"{ip}: engine processes running: {w['engine_processes'][:3]}")
            if w["gtt_used_bytes"] > self.p["gpu_free_gtt_bytes"]:
                why.append(f"{ip}: {w['gtt_used_bytes']} bytes of GPU memory in use")
            if st["holds"][ip]:
                why.append(f"{ip}: held ({st['holds'][ip]})")
        for line in lease_lines_for(self.nodes, st["leases"]):
            why.append(f"lease: {line.strip()[:160]}")
        if st["launcher_starting"]:
            why.append("the pod's pair launcher is starting the pair (strixfold-pair-up is active on A)")
        if st["serving"]:
            why.append(f"a server answers {self.p['serve_url']}/health: the pair is serving")
        for x in st["by_hand"]:
            why.append(f"another session drove the pair within the quiet period ({self.p['quiet_s']} s): {x}")
        for x in st["sessions"]:
            why.append(f"a script that drives the pair is running on this host: {x}")
        return why

    def only_launcher_serving(self, st):
        """True when the one thing on the pair is the pod launcher's own serving run and nobody else is at work:
        the state a card window takes the pair from (the launcher takes its run down when it sees the hold)."""
        why = self.free(st)
        mine = [w for w in why if "engine processes running" in w or "bytes of GPU memory" in w or "the pair is serving" in w]
        return st["serving"] and len(mine) == len(why)

    def idle(self, seconds=180):
        t0 = time.time()
        while True:
            ws = {ip: self.worker_state(ip) for ip in (self.f, self.a)}
            busy = [ip for ip, w in ws.items() if w["engine_processes"] or w["gtt_used_bytes"] > self.p["gpu_free_gtt_bytes"]]
            if not busy or time.time() - t0 > seconds or self.plan:
                return not busy, ws
            time.sleep(5)

    # -- confs and deployments
    def conf_text(self, deploy, drafts, run_root=None):
        with open(os.path.join(ROOT, self.p["base_conf"])) as f:
            text = f.read()
        over = dict(self.p["conf"], DEPLOY_DIR=deploy, DRAFTS=str(drafts), MASTER_IP=self.nodes["rank0"]["rail"],
                    SERVE_HOST=self.nodes["rank0"]["mgmt"])
        # ... and the live rank 1: strixfold-pair.sh reads NODE1/NODE1_IP/NODE1_RAIL out of the conf it is handed (cv,
        # :21), and without them it falls back to A. The gate's base conf is a recording of the pod's serving conf of
        # 2026-11-10, which predates the move to D (NODE1=D) - so a reference run started by the launcher from one of
        # these confs would try to bring up the OLD pair, refusing on a node nobody holds, while the gate's own hold
        # sits on the live one (card b390d523b9, 2026-11-16).
        if self.nodes["rank1"].get("letter"):
            over["NODE1"] = self.nodes["rank1"]["letter"]
        over["NODE1_IP"] = self.nodes["rank1"]["mgmt"]
        over["NODE1_RAIL"] = self.nodes["rank1"]["rail"]
        if run_root:
            over["RUN_ROOT"] = run_root
        text += "\n# --- strixflerp phase 3 gate: appended, a later assignment wins ---\n"
        return text + "".join(f"{k}='{v}'\n" for k, v in over.items())

    def stage_assembled(self, tree, abi_tree):
        """The assembled deployment on each worker: this build's native-hip tree, the reference deployment's launch
        tools (the pod's scripts, the same for both sides), a DEPLOY.json naming the source commit."""
        s, ref = self.p["worker_scratch"], self.p["reference_deploy"]
        meta = {"schema": "tf-port-deploy/1", "commit": self.p["reference_commit"], "clean": True,
                "host": socket.gethostname(), "created_utc": now(),
                "assembled_by": f"strixflerp recipe {self.ctx['recipe'].name} (catalogue sources, byte-identical to the commit)"}
        local = os.path.join(self.ctx["scratch"], "DEPLOY.json")
        with open(local, "w") as f:
            json.dump(meta, f, indent=1)
        for ip in (self.f, self.a):
            rc, _ = self.do(f"scratch on {ip}", SSH + [f"pod@{ip}", f"mkdir -p {s}/deploy {s}/conf {s}/run {s}/abi && "
                            f"cp -a {ref}/tools {s}/deploy/ && chmod -R u+w {s}/deploy"])
            if rc:
                return False
            for src, dst in ((os.path.join(tree, "tree/zig-out/native-hip"), f"{s}/deploy/"), (local, f"{s}/deploy/DEPLOY.json"),
                             (os.path.join(abi_tree, "tree/zig-out", self.ctx["cfg"]["abi"]["conformance"]), f"{s}/abi/")):
                rc, _ = self.do(f"copy to {ip}", ["rsync", "-a", "-e", " ".join(SSH), src, f"pod@{ip}:{dst}"], 900)
                if rc:
                    return False
        return True

    def put_conf(self, name, text):
        local = os.path.join(self.ctx["scratch"], name)
        with open(local, "w") as f:
            f.write(text)
        for ip in (self.f, self.a):
            self.do(f"conf to {ip}", ["rsync", "-a", "-e", " ".join(SSH), local, f"pod@{ip}:{self.p['worker_scratch']}/conf/{name}"])
        return local

    # -- runs
    def wait_health(self, unit, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout and not self.plan:
            try:
                with urllib.request.urlopen(self.p["serve_url"] + "/health", timeout=5) as r:
                    if r.status == 200:
                        # first_light's own probe request runs after /health: wait for its unit to finish
                        rc, _ = self.on(self.a, f"XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user is-active -q {unit}")
                        if rc != 0:
                            return True, round(time.time() - t0, 1)
            except (urllib.error.URLError, OSError):
                rc, _ = self.on(self.a, f"XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user is-active -q {unit}")
                if rc != 0 and time.time() - t0 > 30:
                    return False, round(time.time() - t0, 1)          # the start gave up (first_light took the run down)
            time.sleep(5)
        return self.plan, round(time.time() - t0, 1)

    def post(self, body):
        req = urllib.request.Request(self.p["serve_url"] + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.run["request_timeout_s"]) as r:
            reply = json.loads(r.read())
        return reply, round(time.time() - t0, 3)

    def requests(self):
        n = self.run["max_tokens"]
        x, rows = 12345, []
        for i in range(self.run["long_prompt_sentences"]):
            ws = []
            for _ in range(6):
                x = (1103515245 * x + 12345) % (1 << 31)
                ws.append(WORDS[(x >> 16) % len(WORDS)])
            rows.append(f"Record {i + 1}: the {ws[0]} {ws[1]} was moved from the {ws[2]} {ws[3]} to the {ws[4]} {ws[5]}.")
        common = {"temperature": 0, "max_tokens": n, "stream": False, "return_token_ids": True}
        return [("short-chat", dict(common, messages=[{"role": "user", "content": "A train leaves at 09:40 and arrives "
                 "at 13:05. How long is the trip? Then name three rivers in Europe and one fact about each."}])),
                ("long-chat", dict(common, messages=[{"role": "system", "content": "You answer questions about the records you are given."},
                 {"role": "user", "content": "\n".join(rows) + "\n\nList the first five record numbers whose sentence "
                  "mentions a harbour, and quote each of those sentences."}])),
                ("code", dict(common, messages=[{"role": "user", "content": "Write a Python function that merges two "
                 "sorted lists into one sorted list without using sort, then explain its running time."}]))]

    def serve_and_ask(self, label, run_dir_cmd):
        """The loaded server answers every request `repeats` times; rank 0's log is fetched for the receipts."""
        out = {"label": label, "runs": []}
        for rep in range(self.run["repeats"]):
            for name, body in self.requests():
                reply, wall = self.post(body)
                tf = reply.get("tensorfold", {}) if isinstance(reply.get("tensorfold"), dict) else {}
                msg = reply["choices"][0].get("message", {})
                text = (msg.get("reasoning_content") or "") + "\x1f" + (msg.get("content") or "")
                out["runs"].append({"prompt": name, "repeat": rep, "wall_s": wall, "usage": reply.get("usage", {}),
                                    "finish_reason": reply["choices"][0].get("finish_reason"),
                                    "token_ids": tf.get("token_ids"), "decode_tokens_per_second": tf.get("tokens_per_second"),
                                    "tensorfold": {k: v for k, v in tf.items() if k != "token_ids"},
                                    "text_sha256": hashlib.sha256(text.encode()).hexdigest()[:16], "text_head": text[:120]})
        rc, log = self.on(self.f, run_dir_cmd + " && cat rank0.log", 120)
        local = os.path.join(self.ctx["res"], f"{label}-rank0.log")
        with open(local, "w") as f:
            f.write(log)
        out["rank0_log"] = os.path.basename(local)
        out["receipts"] = receipts(log)
        return out

    def reference(self, drafts, conf_local):
        """Start the reference with the pod's launcher; (unit ok, seconds, run id)."""
        env = f"SP_CONF={shlex.quote(conf_local)} KANBAN_ACTOR={self.p['owner']}"
        self.do("reference up (the pod's launcher)", f"{env} {self.p['launcher']} up", 300)
        self.started.append(("launcher", None))
        ok, secs = self.wait_health("strixfold-pair-up", self.p["startup_timeout_s"])
        _, rid = self.on(self.a, "ls -t ~/tfp/run/serve128k | grep '^pair-' | head -1")
        return ok, secs, rid.strip()

    def reference_down(self):
        self.do("reference down (the pod's launcher)", f"KANBAN_ACTOR={self.p['owner']} {self.p['launcher']} down", 400)
        self.started = [s for s in self.started if s[0] != "launcher"]

    def assembled_up(self, conf_name, rid):
        s = self.p["worker_scratch"]
        cmd = (f"XDG_RUNTIME_DIR=/run/user/$(id -u) systemd-run --user --collect --unit=strixflerp-gate-up "
               f"--property=KillMode=process bash -c 'cd {s}/deploy/tools/dsv41-hip && RUN_ID={rid} "
               f"TF_SERVE_CONF={s}/conf/{conf_name} ./first_light.sh up'")
        self.do("assembled up (first_light.sh of the same tools, the assembled deployment)", SSH + [f"pod@{self.a}", cmd], 120)
        self.started.append(("assembled", (conf_name, rid)))
        return self.wait_health("strixflerp-gate-up", self.p["startup_timeout_s"])

    def assembled_down(self, conf_name, rid):
        s = self.p["worker_scratch"]
        self.do("assembled down", SSH + [f"pod@{self.a}", f"XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user stop strixflerp-gate-up 2>/dev/null; "
                f"cd {s}/deploy/tools/dsv41-hip && RUN_ID={rid} TF_SERVE_CONF={s}/conf/{conf_name} ./first_light.sh down"], 400)
        self.started = [x for x in self.started if x != ("assembled", (conf_name, rid))]

    def conformance(self, conf_name, native):
        """The ABI driver as rank 0 of the assembled deployment: rank 1 started by the pod's rank script, the driver
        in rank 0's resolved environment in place of the server."""
        s, abi = self.p["worker_scratch"], self.ctx["cfg"]["abi"]
        rid = "flerp-abi-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        tools, env = f"{s}/deploy/tools/dsv41-hip", f"RUN_ID={rid} TF_SERVE_CONF={s}/conf/{conf_name}"
        out = {"run": rid, "cases": []}
        rc, resolved = self.on(self.f, f"cd {tools} && {env} ./serve-rank0.sh check", 120)
        if self.plan:
            self.say("PLAN conformance: rank 1 by serve-rank1.sh, the driver in serve-rank0.sh's resolved environment")
            return out
        res = json.loads(resolved[resolved.index("{"):])
        ids_dir = f"{s}/abi"
        for name, cap in native.items():
            n = min(abi["new_tokens"], len(cap["generated"]))
            local = os.path.join(self.ctx["scratch"], f"{name}.ids")
            with open(local, "w") as f:
                f.write(" ".join(str(i) for i in cap["prompt"]) + "\n")
            out_of(["rsync", "-a", "-e", " ".join(SSH), local, f"pod@{self.f}:{ids_dir}/"])
            self.started.append(("abi", (conf_name, rid)))
            self.do("GPU locks and rank 1", SSH + [f"pod@{self.f}", f"cd {tools} && {env} ./serve-rank0.sh lock"], 120)
            self.do("rank 1", SSH + [f"pod@{self.a}", f"cd {tools} && {env} ./serve-rank1.sh lock 60 && {env} ./serve-rank1.sh start --locked"], 180)
            envs = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in res["env"].items())
            argv = res["argv"]                    # serve_lib.sh serve_resolved: the server's argv and environment
            model_dir = argv[argv.index("serve") + 1]
            ctxn = argv[argv.index("--context") + 1]
            drafts = "off" if "--no-drafts" in argv else "on"
            kit = res["env"].get("TENSORFOLD_HIP_KERNELS") or f"{s}/deploy/native-hip/share/tensorfold/hip/gfx1151/dsv41"
            # a dsv41 state is kept at a chunk end at least a window before the prompt's end: a shorter prompt has none
            snap = ["--snap-at", str(abi["snap_at"])] if len(cap["prompt"]) > abi["snap_at"] + 512 else ["--no-snapshot"]
            cmd = (f"cd {ids_dir} && env {envs} LD_LIBRARY_PATH=/data/models/tools/therock-10.2/lib timeout {abi['timeout_s']} "
                   f"./{os.path.basename(abi['conformance'])} --ids {name}.ids --new {n} --out {name}.json --draft {abi['draft']} "
                   f"{' '.join(snap)} --snap-new {abi['snap_new']} model_dir={model_dir} kit={kit} context={ctxn} drafts={drafts} "
                   f"> {name}.txt 2>&1; echo rc=$?; cat {name}.json")
            rc, text = self.do(f"conformance {name}", SSH + [f"pod@{self.f}", cmd], abi["timeout_s"] + 60)
            self.do("rank 1 stop, locks released", SSH + [f"pod@{self.a}", f"cd {tools} && {env} ./serve-rank1.sh stop; {env} ./serve-rank1.sh unlock"], 200)
            self.do("rank 0 lock released", SSH + [f"pod@{self.f}", f"cd {tools} && {env} ./serve-rank0.sh unlock"], 60)
            self.started = [x for x in self.started if x[0] != "abi"]
            # What the driver wrote goes into the results before anything is torn down, pass or fail: its trace
            # (stdout and stderr, {name}.txt on the worker), the command's own output (the exit status, the result
            # file, and the shell's word on how the process ended), and the end of rank 1's log for this run.
            _, trace = self.on(self.f, f"tail -c 400000 {ids_dir}/{name}.txt 2>/dev/null", 60)
            _, rank1 = self.on(self.a, f"tail -n 60 {s}/run/{rid}/rank1.log 2>/dev/null", 60)
            for suffix, body in (("driver.txt", trace), ("output.txt", text), ("rank1-tail.txt", rank1)):
                with open(os.path.join(self.ctx["res"], f"conformance-{name}-{suffix}"), "w") as f:
                    f.write(body + "\n")
            rec = {"prompt": name, "compared_tokens": n, "prompt_tokens": len(cap["prompt"]),
                   "evidence": [f"conformance-{name}-{x}" for x in ("driver.txt", "output.txt", "rank1-tail.txt")]}
            m = re.search(r"^rc=(\d+)$", text, re.M)
            rec["exit_status"] = int(m.group(1)) if m else None
            try:
                # the result file is followed by whatever the shell said (a signal's name): read the one JSON value
                j, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
                with open(os.path.join(self.ctx["res"], f"conformance-{name}.json"), "w") as f:
                    json.dump(j, f, indent=1)
                # the driver writes its executor-wide checks at the top and one object a case; this run has one case
                # with the default paths (plain, speculative, snapshot) and cross-shape comparisons required:
                # StrixFold's drafts are output-invariant (SPEC.md section 1, fact 3)
                case = j["cases"][0]
                rec["checks"] = {k[6:]: v for d in (j, case) for k, v in d.items() if k.startswith("check_")}
                rec["plain_equals_native"] = case.get("plain") == cap["generated"][:n]
                rec.update({k: case.get(k) for k in ("rounds", "drafted", "accepted", "dropped_rounds")})
                rec["caps"] = j.get("caps")
            except (ValueError, KeyError, IndexError):
                rec["error"] = text[-600:]
            # The driver's process must also end with status 0. On 2026-11-11T02:00Z both cases wrote a complete
            # result with every check true and then died of SIGSEGV (rc 139) while closing the executor; the
            # stage failed then because the reader took the shell's message for part of the JSON. It still fails,
            # for the stated reason: an executor that cannot be closed is not conformant.
            rec["pass"] = (rec.get("plain_equals_native") is True and all(rec.get("checks", {"x": False}).values())
                           and rec["exit_status"] == 0)
            out["cases"].append(rec)
            ok, _ = self.idle()
            if not ok:
                raise RuntimeError("GPU memory did not return to idle after the conformance run")
        out["pass"] = bool(out["cases"]) and all(c["pass"] for c in out["cases"])
        return out

    def teardown(self):
        """Stop whatever this stage still has running, remove the scratch directories, release the hold."""
        for kind, arg in list(self.started):
            if kind == "launcher":
                self.reference_down()
            elif kind == "assembled":
                self.assembled_down(*arg)
            elif kind == "abi":
                s = self.p["worker_scratch"]
                env = f"RUN_ID={arg[1]} TF_SERVE_CONF={s}/conf/{arg[0]}"
                self.do("abi leftovers", SSH + [f"pod@{self.a}", f"cd {s}/deploy/tools/dsv41-hip && {env} ./serve-rank1.sh stop; {env} ./serve-rank1.sh unlock"], 200)
                self.do("abi leftovers", SSH + [f"pod@{self.f}", f"pkill -f flerp-executor-conformance; cd {s}/deploy/tools/dsv41-hip && {env} ./serve-rank0.sh unlock"], 60)
        # a crashed run of this gate leaves its GPU-lock unit holding /var/tmp/halo-gpu.lock: stop this gate's own
        # (run ids flerp-*) whose rank unit is gone, as the pod's launcher does for its runs before a start
        for ip in (self.f, self.a):
            self.do(f"stale lock units of this gate on {ip}", SSH + [f"pod@{ip}", "export XDG_RUNTIME_DIR=/run/user/$(id -u); "
                    "for u in $(systemctl --user list-units 'tfp-lock*flerp-*' --state=active --no-legend --plain | cut -d' ' -f1); do "
                    "r=${u#tfp-lock}; systemctl --user is-active -q \"tfp-rank${r}\" && continue; systemctl --user stop \"$u\" && echo stopped $u; done; true"])
        ok, ws = self.idle()
        for ip in (self.f, self.a):
            self.do(f"remove scratch on {ip}", SSH + [f"pod@{ip}", f"rm -rf {self.p['worker_scratch']}"])
        return ok, ws


VERIFY = re.compile(r'^\{"verify": (\d+), "rows": \d+, "ids": (.*?), "drawn": (.*?)\}?\}$')


def nums(text):
    return [int(x) for x in re.findall(r"-?\d+", text)]


def receipts(log):
    """rank 0's receipts: each prompt's logits sha256 (TF_DS_LOGITS_SHA) and every verify window (TF_DS_TRACE_ROUNDS).
    Windows are grouped by prompt (a prompt_logits line starts a request); `drawn` maps a cache position to the token
    drawn there, which is the schedule-independent part."""
    reqs, cur = [], None
    for line in log.splitlines():
        line = line.strip()
        if line.startswith('{"prompt_logits"'):
            try:
                j = json.loads(line)
            except ValueError:
                continue
            cur = {"prompt_logits_sha256": j.get("sha256"), "rows": j.get("rows"), "windows": [], "drawn": {}}
            reqs.append(cur)
            continue
        m = VERIFY.match(line)                  # Zig prints the slices as "{ 1, 2, 3 }": read the numbers, not JSON
        if m and cur is not None:
            start, ids, drawn = int(m.group(1)), nums(m.group(2)), nums(m.group(3))
            cur["windows"].append([start, ids, drawn])
            for r, t in enumerate(drawn):
                cur["drawn"].setdefault(str(start + 1 + r), t)
    for r in reqs:
        r["windows_sha256"] = hashlib.sha256(json.dumps(r.pop("windows")).encode()).hexdigest()[:16]
        r["drawn_sha256"] = hashlib.sha256(json.dumps(sorted(r.pop("drawn").items(), key=lambda kv: int(kv[0]))).encode()).hexdigest()[:16]
    return reqs


def compare(ref, asm, tol):
    rows, ok = [], True
    for a, b in zip(ref["runs"], asm["runs"]):
        same = a["token_ids"] is not None and a["token_ids"] == b["token_ids"]
        rows.append({"prompt": a["prompt"], "repeat": a["repeat"], "tokens": len(a["token_ids"] or []), "token_ids_equal": same})
        ok = ok and same
    ra, rb = ref["receipts"], asm["receipts"]
    rec = {"requests_with_receipts": [len(ra), len(rb)],
           "prompt_logits_equal": bool(ra) and len(ra) == len(rb) and all(x["prompt_logits_sha256"] == y["prompt_logits_sha256"] for x, y in zip(ra, rb)),
           "drawn_by_position_equal": bool(ra) and len(ra) == len(rb) and all(x["drawn_sha256"] == y["drawn_sha256"] for x, y in zip(ra, rb)),
           "window_schedule_equal": bool(ra) and len(ra) == len(rb) and all(x["windows_sha256"] == y["windows_sha256"] for x, y in zip(ra, rb))}
    speed = []
    for name in dict.fromkeys(r["prompt"] for r in ref["runs"]):
        x = [r["decode_tokens_per_second"] for r in ref["runs"] if r["prompt"] == name and r["decode_tokens_per_second"]]
        y = [r["decode_tokens_per_second"] for r in asm["runs"] if r["prompt"] == name and r["decode_tokens_per_second"]]
        if not x or not y:
            speed.append({"prompt": name, "within_noise": False, "reference": x, "assembled": y})
            continue
        mx, my = statistics.median(x), statistics.median(y)
        within = abs(my - mx) <= tol * mx or min(x) <= my <= max(x)
        speed.append({"prompt": name, "reference_median": round(mx, 2), "assembled_median": round(my, 2),
                      "diff": round((my - mx) / mx, 4), "within_noise": within, "reference": x, "assembled": y})
    return {"token_identity": ok, "requests": rows, "receipts": rec, "speed": speed,
            "speed_within_noise": all(s["within_noise"] for s in speed),
            "pass": ok and rec["prompt_logits_equal"] and rec["drawn_by_position_equal"] and all(s["within_noise"] for s in speed)}


def stage_pair(ctx, plan=False):
    pair = Pair(ctx, plan)
    open(pair.log, "a").close()
    for line in pair_nodes.describe(pair.nodes).splitlines():
        pair.say(line)
    p, cfg = pair.p, ctx["cfg"]
    out = {"started": now(), "plan_only": plan, "evidence": "pair.txt", "pair": pair.nodes, "pair_recorded": pair.recorded,
           "before": pair.state()}
    why = pair.free(out["before"])
    out["free"] = not why
    take = ctx.get("take") and pair.only_launcher_serving(out["before"])
    out["taken_from_launcher"] = bool(take)
    if why and not plan and not take:
        out.update({"pass": False, "blocked": "the pair is not free: " + "; ".join(why)})
        pair.say("blocked: " + out["blocked"])
        return out
    tree = os.path.join(ROOT, "build", ctx["recipe"].name)
    abi_tree = os.path.join(ROOT, "build", cfg["abi"]["recipe"])
    for t in (tree, abi_tree):
        if not os.path.isfile(os.path.join(t, "built.json")) and not plan:
            out.update({"pass": False, "blocked": f"{os.path.relpath(t, ROOT)} is not built: run the build stage (and assemble + build {cfg['abi']['recipe']})"})
            return out
    hold = (f"OWNER={shlex.quote(p['owner'])} REASON={shlex.quote(p['why'])} TTL={p['hold_ttl_s']} pod-quad hold {pair.f} {pair.a}")
    rc, text = pair.do("hold both workers (one call; takes the pod-lease worker leases)", hold, 900)
    if rc != 0:
        out.update({"pass": False, "blocked": f"pod-quad hold refused (rc {rc}): {text[-300:]}"})
        return out
    out["comparisons"], out["servers"] = [], []
    out["hold_taken"] = now()
    left = lambda: ctx["release_by"] - time.time()          # seconds to the hard stop (a year when none was given)
    try:
        # under the hold the pod's launcher takes its own serving run down within a tick (--take-from-launcher);
        # nothing else may be on the pair
        ok, ws = pair.idle(360 if take else 0)
        if not ok:
            raise RuntimeError(f"the pair is not idle under the hold: {ws}")
        if not pair.stage_assembled(tree, abi_tree):
            raise RuntimeError("staging the assembled deployment on the workers failed")
        native = {}
        for n_done, drafts in enumerate(ctx["cfg"]["run"]["drafts"]):
            tag = f"drafts-{'on' if drafts else 'off'}"
            if left() < p["setting_margin_s"]:               # measured, in the config
                out.setdefault("skipped", []).append(f"{tag}: {int(left())} s to the hard stop, a reference and an assembled load do not fit")
                pair.say(out["skipped"][-1])
                continue
            pair.p["startup_timeout_s"] = int(min(ctx["cfg"]["pair"]["startup_timeout_s"], max(120, left() - 360)))
            ref_conf = pair.put_conf(f"reference-{tag}.conf", pair.conf_text(p["reference_deploy"], drafts))
            asm_name = f"assembled-{tag}.conf"
            pair.put_conf(asm_name, pair.conf_text(f"{p['worker_scratch']}/deploy", drafts, f"{p['worker_scratch']}/run"))
            # reference, with the pod's launcher
            ok = False
            for attempt in range(1, p["reference_attempts"] + 1):
                ok, secs, rid = pair.reference(drafts, ref_conf)
                if ok:
                    break
                pair.say(f"reference start {attempt} did not reach health after {secs} s (run {rid})")
                _, logs = pair.on(pair.a, f"cd ~/tfp/run/serve128k/{rid} 2>/dev/null && tail -n 30 rank1.log rank0-ready.txt 2>/dev/null")
                _, logs0 = pair.on(pair.f, f"tail -n 40 ~/tfp/run/serve128k/{rid}/rank0.log 2>/dev/null")
                with open(os.path.join(ctx["res"], f"reference-{tag}-failed-start-{attempt}.txt"), "w") as f:
                    f.write(logs + "\n==== rank 0\n" + logs0 + "\n")
                pair.reference_down()
                pair.idle()
            if not ok and not plan:
                raise RuntimeError(f"the reference deployment did not reach health in {p['reference_attempts']} attempts "
                                   f"(known problem, card STRIXFOLD-PAIR 7f019a8c41): logs in reference-{tag}-failed-start-*.txt")
            ref = {"label": f"reference-{tag}", "runs": [], "receipts": []} if plan else pair.serve_and_ask(
                f"reference-{tag}", f"cd ~/tfp/run/serve128k/{rid}")
            ref.update(startup_s=secs, run=rid, drafts=drafts)
            pair.reference_down()
            ok, ws = pair.idle()
            if not ok:
                raise RuntimeError(f"GPU memory did not return to idle after the reference: {ws}")
            # assembled, by the same first_light.sh
            rid = "flerp-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            ok, secs = pair.assembled_up(asm_name, rid)
            if not ok and not plan:
                _, logs0 = pair.on(pair.f, f"tail -n 40 {p['worker_scratch']}/run/{rid}/rank0.log 2>/dev/null")
                with open(os.path.join(ctx["res"], f"assembled-{tag}-failed-start.txt"), "w") as f:
                    f.write(logs0 + "\n")
                raise RuntimeError(f"the assembled deployment did not reach health (assembled-{tag}-failed-start.txt)")
            asm = {"label": f"assembled-{tag}", "runs": [], "receipts": []} if plan else pair.serve_and_ask(
                f"assembled-{tag}", f"cd {p['worker_scratch']}/run/{rid}")
            asm.update(startup_s=secs, run=rid, drafts=drafts)
            if not plan and drafts and not native:
                # the prompt ids of two requests, for the ABI run: the server's tokenize route, while it is loaded
                for name, body in pair.requests()[:2]:
                    gen = next(r["token_ids"] for r in asm["runs"] if r["prompt"] == name)
                    native[name] = {"generated": gen, "prompt": prompt_ids(pair, body)}
            pair.assembled_down(asm_name, rid)
            ok, ws = pair.idle()
            if not ok:
                raise RuntimeError(f"GPU memory did not return to idle after the assembled server: {ws}")
            out["servers"] += [ref, asm]
            if not plan:
                cmp_ = compare(ref, asm, ctx["cfg"]["run"]["tps_tolerance"])
                cmp_["drafts"] = drafts
                out["comparisons"].append(cmp_)
        if left() >= p["abi_margin_s"] and native:
            out["abi"] = pair.conformance(f"assembled-drafts-{'on' if ctx['cfg']['run']['drafts'][0] else 'off'}.conf", native)
        else:
            out["abi"] = {"pass": False, "skipped": f"{int(left())} s to the hard stop: the conformance run needs {p['abi_margin_s']} s"}
            pair.say("ABI conformance skipped: " + out["abi"]["skipped"])
    except Exception as e:                       # the stage fails with its reason; the teardown still runs
        out["error"] = str(e)
        pair.say("error: " + str(e))
    finally:
        idle, ws = pair.teardown()
        rc, text = pair.do("release the hold", f"OWNER={shlex.quote(p['owner'])} pod-quad release", 300)
        out["hold_released"] = now()
        out["after"] = pair.state()
        out["released"] = rc == 0 and not any(out["after"]["holds"].values())
        out["gpu_idle_after"] = idle
        # the pod's launcher brings its own serving run back: watch for it (up to 10 minutes), repair nothing
        back = {"healthy": False}
        t0 = time.time()
        while not plan and time.time() - t0 < 600:
            st = pair.state()
            procs = " ".join(st["workers"][pair.f]["engine_processes"])
            if st["serving"]:
                back = {"healthy": True, "seconds_after_release": int(time.time() - t0), "rank0_process": procs[:200]}
                break
            time.sleep(15)
        back["launcher_log_tail"] = tail(p["launcher_log"], 8)
        out["pod_serving_after_release"] = back
        pair.say(f"pod serving after release: {back}")
    out["finished"] = now()
    out["pass"] = (not plan and "error" not in out and len(out["comparisons"]) == len(ctx["cfg"]["run"]["drafts"])
                   and all(c["pass"] for c in out["comparisons"]) and out.get("abi", {}).get("pass") is True
                   and out["released"] and out["gpu_idle_after"])
    return out


def prompt_ids(pair, body):
    """A chat request's prompt token ids from the loaded server (POST /tokenize with the same messages)."""
    req = urllib.request.Request(pair.p["serve_url"] + "/tokenize", data=json.dumps({"messages": body["messages"]}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["tokens"]


# --- summary --------------------------------------------------------------------------------------------------
def write_summary(ctx, results):
    ok = SM.all_pass(STAGES, results)
    SM.write_json(ctx["res"], {"recipe": ctx["recipe"].name, "host": socket.gethostname(), "date": now(), "pass": ok, "stages": results}, ctx=ctx)
    lines = [f"# Recipe gate: {ctx['recipe'].name}", "", f"Host {socket.gethostname()}, {now()}. Stages run: {', '.join(results)}.", ""]
    notes = {"check": lambda r: r.get("last_line", ""),
             "origin": lambda r: f"{r.get('identical')}/{r.get('files')} files byte-identical to git archive {r.get('commit')} of "
                                 f"{r.get('checkout')}, {r.get('problems')} problem(s), {r.get('unharvested')} origin files listed as unharvested",
             "build": lambda r: f"{r.get('files_in_tree')} files assembled, built in {r.get('build_s')} s; kit {r.get('kit_objects')} objects, "
                                f"manifest sha256 {str(r.get('kit_manifest_sha256'))[:16]}... "
                                f"{'equals' if r.get('kit_equals_reference_deployment') else 'DIFFERS from'} the reference deployment's"
                                f"{'' if r.get('kit_directory_is_the_manifest') else '; the kit directory holds objects the manifest does not name'}",
             "pair": lambda r: r.get("blocked") or r.get("error") or ""}
    for s in STAGES:
        if s in results:
            r = results[s]
            state = "PASS" if r.get("pass") else ("NOT RUN" if r.get("blocked") else "FAIL")
            lines.append(f"- **{s}**: {state} - {notes[s](r)}")
    pr = results.get("pair", {})
    for c in pr.get("comparisons", []):
        rc = c["receipts"]
        lines += ["", f"Drafts {'on' if c['drafts'] else 'off'}: token ids {'identical' if c['token_identity'] else 'DIFFER'} over "
                  f"{len(c['requests'])} requests; prompt logits sha256 {'equal' if rc['prompt_logits_equal'] else 'DIFFER'}; drawn tokens by "
                  f"position {'equal' if rc['drawn_by_position_equal'] else 'DIFFER'}; window schedule {'equal' if rc['window_schedule_equal'] else 'differs'}.",
                  "", "| prompt | decode t/s reference (median) | decode t/s assembled (median) | diff | within noise |", "|---|---|---|---|---|"]
        for s in c["speed"]:
            lines.append(f"| {s['prompt']} | {s.get('reference_median')} | {s.get('assembled_median')} | "
                         f"{s.get('diff', 0) * 100:+.1f}% | {'yes' if s['within_noise'] else 'NO'} |")
    if pr.get("abi", {}).get("cases"):
        lines += ["", "Executor ABI conformance (StrixFold adapter as rank 0):"]
        for c in pr["abi"]["cases"]:
            lines.append(f"- {c['prompt']}: plain = server {'yes' if c.get('plain_equals_native') else 'NO'}, checks {c.get('checks')}")
    SM.write_md(ctx["res"], lines)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--plan", action="store_true", help="pair stage: read the pair's state and print what would run; change nothing")
    ap.add_argument("--take-from-launcher", action="store_true",
                    help="pair stage: also run when the only thing on the pair is the pod launcher's own serving run and "
                         "no other session drove the pair within the quiet period; the hold makes the launcher yield")
    ap.add_argument("--quiet-s", type=int, help="pair stage: override the config's quiet period (an agreed window)")
    ap.add_argument("--release-by", metavar="HH:MM", help="pair stage: hard stop (UTC today); loads that do not fit are skipped")
    ap.add_argument("--keep-going", action="store_true")
    pair_nodes.add_args(ap)
    a = ap.parse_args()
    cat = Catalog(ROOT)
    recipe = cat.recipe(a.recipe)
    if recipe is None:
        sys.exit(f"no recipe {a.recipe}")
    with open(os.path.join(ROOT, recipe.gate["config"]), "rb") as f:
        cfg = tomllib.load(f)
    res = os.path.join(ROOT, "gate", "results", recipe.name)
    os.makedirs(res, exist_ok=True)
    ctx = {"cat": cat, "recipe": recipe, "cfg": cfg, "res": res, "scratch": os.path.join(ROOT, cfg["host"]["scratch"]),
           "take": a.take_from_launcher, "release_by": time.time() + 365 * 86400, "pair_args": a}
    if a.quiet_s is not None:
        cfg["pair"]["quiet_s"] = a.quiet_s
    if a.release_by:
        hh, mm = (int(x) for x in a.release_by.split(":"))
        t = datetime.datetime.now(datetime.timezone.utc).replace(hour=hh, minute=mm, second=0, microsecond=0)
        ctx["release_by"] = t.timestamp()
    os.makedirs(ctx["scratch"], exist_ok=True)
    results = {}
    prior = os.path.join(res, "summary.json")
    if os.path.isfile(prior) and set(a.stages.split(",")) != set(STAGES):
        with open(prior) as f:
            results = json.load(f).get("stages", {})
    fns = {"check": stage_check, "origin": stage_origin, "build": stage_build, "pair": lambda c: stage_pair(c, a.plan)}
    for s in STAGES:
        if s not in a.stages.split(","):
            continue
        print(f"== {s}", flush=True)
        r = fns[s](ctx)
        if s == "pair" and a.plan:
            print(json.dumps({"pair": r.get("pair"), "free": r.get("free"), "why_not": Pair(ctx, True).free(r["before"])}, indent=1))
            continue                                        # a plan is not a result
        results[s] = r
        print(f"   {'PASS' if r.get('pass') else 'FAIL'} {r.get('blocked', '')}{r.get('error', '')}", flush=True)
        if not r.get("pass") and not a.keep_going:
            break
    if a.plan:
        return 0                                            # --plan writes nothing under gate/results
    return 0 if write_summary(ctx, {s: results[s] for s in STAGES if s in results}) else 1


if __name__ == "__main__":
    sys.exit(main())
