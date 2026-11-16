#!/usr/bin/env python3
"""The gate of the common front end over the StrixFold executor (SPEC.md section 9, phase 3):
`flerp gate recipe dsv41-flash.flerp-serve-tp2`.

Stages:
  check   flerp check passes for the catalogue
  build   the recipe assembles with no hand edits and builds on control-node: the origin's native-hip step (rank 1's
          process, the kernel kit, the origin's own server) and the step flerp-serve (serve.flerp.http's C++ linked
          by the Zig build with the Zig-exported executor adapter and text module). The front end binary must
          start, load the checkpoint's tokenizer and template through the text module (`[serve] text_probe`, a
          checkpoint directory on this host), and refuse to open an executor without its keys - no GPU, no worker.
  pair    on workers F and A, inside a hold of both (`pod-quad hold`), in an agreed window. Modelled on the
          StrixFold recipe gate's pair stage (gate/strixfold_recipe.py), whose helpers this file imports: the
          same hold, the same staging of an assembled deployment, the same reference runs with the pod's
          launcher, the same teardown. For DSpark drafting on, then off:
            1. the reference deployment (the origin's server, started by the pod's launcher) answers the
               configured greedy chat requests with return_token_ids, and tokenizes each request's prompt;
            2. the front end runs as rank 0 of this recipe's build - rank 1 started by the pod's rank script, the
               front end in rank 0's resolved environment in place of the server, as the ABI conformance run does
               it - and answers the same requests over HTTP with "flerp_trace": true, not streamed and streamed.
          Required, every request, drafts on and off:
            - the prompt's token ids equal the origin's (the text module renders and tokenizes as the server does);
            - the reply's token ids equal the origin's (whose return_token_ids list ends with the end-of-sequence id
              when the reply finished by stop: that id is taken off, and the two completion token counts, which
              both include it, must be equal). StrixFold's drafts are output-invariant (SPEC.md section 1,
              fact 3), so this is required although the front end holds a fixed draft depth and the origin's lane
              core prices its depths: a difference is a failure, not a forward-shape observation;
            - the streamed response carries the same token ids and the same text as the one that is not streamed;
            - /health reports tokens drawn by the executor and held drafts;
            - a request this executor cannot honour is refused from its capability flags (HTTP 400, code
              "executor_cannot"): logprobs, logit_bias, temperature above 0 without top_k, top_k above the
              family's bound; a feature outside the front end's scope (tools) with code "not_served";
              response_format json_schema is served: HTTP 200 and content valid against the schema (amended after 9ef8d14);
            - the front end stops on SIGTERM, closes the executor and exits with status 0, rank 1 ends, and GPU
              memory returns to idle.
          The front end runs as a transient user unit, as the pod's scripts start a rank (SPAWN_UNIT=1): F's logind
          kills what an ssh session leaves behind. Before anything is stopped or removed, for a failure as for a
          pass, the stage copies into the results from both workers: the front end's whole log with the wrapper's
          exit line, the launch script, rank 1's log, `journalctl --user` since the stage began, the units' status
          and the processes. The wait for health tells a process that ended from an ssh that failed.
          Recorded, not required: a sampled request (temperature above 0, top_k and a seed) on both servers - the
          keyed sampler is the executor's and the two servers reach it through different callers; decode tokens
          per second of both.

`--plan` prints what the pair stage would run and changes nothing. Unlike the StrixFold recipe gate's plan it does
not read the pair's state either: it opens no connection to a worker and calls no pod tool.
"""
import argparse
import datetime
import glob
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gate"))
import strixfold_recipe as sf                    # noqa: E402  (the pair: hold, staging, launcher runs, teardown)
import strixite_recipe as p1                     # noqa: E402  (config inheritance)
import summary as SM                             # noqa: E402  (summary.json / summary.md, config_sha256)
from flerplib.catalog import Catalog             # noqa: E402

STAGES = ("check", "build", "pair")
SSH = sf.SSH


def stage_build(ctx):
    log = os.path.join(ctx["res"], "build.txt")
    open(log, "w").close()
    sv = ctx["cfg"]["serve"]
    out = {"evidence": "build.txt", "started": sf.now()}
    rc, tree, out["build_s"] = sf.build_tree(ctx, ctx["recipe"].name, log)
    out["tree"] = os.path.relpath(tree, ROOT)
    binary = os.path.join(tree, sv["binary"])
    built = os.path.isfile(os.path.join(tree, "built.json")) and os.path.isfile(binary)
    out["binary"] = {"sha256": sf.sha256(binary), "bytes": os.path.getsize(binary)} if built else None
    try:
        with open(os.path.join(tree, "assembly.json")) as f:
            asm = json.load(f)
        out["modules"], out["files"] = len(asm["modules"]), len(asm["files"])
        with open(os.path.join(tree, "generated", os.path.basename(asm_link(ctx)))) as f:
            out["c_link_units"] = f.read().count(".file = ")
    except (OSError, ValueError, KeyError):
        pass
    # the binary starts without a GPU: usage, then the text module on a checkpoint of this host, then the
    # executor's refusal of an open without its keys (which comes before it touches a device)
    probes = {}
    if built:
        r = subprocess.run([binary, "--no-such-option"], capture_output=True, text=True, timeout=60)
        probes["usage"] = {"rc": r.returncode, "ok": r.returncode == 2 and "usage: strixflerp-<recipe>" in r.stderr}
        d = sv.get("text_probe")
        if d and os.path.isdir(d):
            r = subprocess.run([binary, "--port", str(sv["port"]), f"text.model_dir={d}"], capture_output=True, text=True, timeout=120)
            probes["text_and_executor_refusal"] = {
                "rc": r.returncode, "stdout": r.stdout.strip()[-300:], "stderr": r.stderr.strip()[-300:],
                "ok": r.returncode == 1 and "startup: text strixfold HfText" in r.stdout and "model_dir and kit are required" in r.stderr}
        else:
            probes["text_and_executor_refusal"] = {"ok": False, "error": f"[serve] text_probe {d} is not a directory on this host"}
    out.update(probes=probes, rc=rc, finished=sf.now(),
               **{"pass": rc == 0 and built and bool(probes) and all(p["ok"] for p in probes.values())})
    return out


def asm_link(ctx):
    return ctx["recipe"].data.get("build", {}).get("c_link", "flerp/link.zig")


def http(url, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"raw": raw.decode(errors="replace")[:300]}


def sse(url, body, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    text, ids, trace, done = "", [], None, False
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode().rstrip("\n")
            if line == "data: [DONE]":
                done = True
            elif line.startswith("data: "):
                c = json.loads(line[6:])
                for ch in c.get("choices", []):
                    text += ch.get("delta", {}).get("content") or ""
                if c.get("choices"):
                    ids += c.get("flerp", {}).get("token_ids", [])
                else:
                    trace = c.get("flerp")
    return {"text": text, "token_ids": ids, "flerp": trace, "done": done}


class Front:
    """The front end as rank 0 of the assembled deployment, by the steps of sf.Pair.conformance."""

    def __init__(self, pair, ctx, conf_name, drafts):
        self.pair, self.ctx, self.conf, self.drafts = pair, ctx, conf_name, drafts
        self.sv, self.s = ctx["cfg"]["serve"], pair.p["worker_scratch"]
        self.rid = "flerp-serve-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.tools = f"{self.s}/deploy/tools/dsv41-hip"
        self.env = f"RUN_ID={self.rid} TF_SERVE_CONF={self.s}/conf/{conf_name}"
        self.url = f"http://{pair.f}:{self.sv['port']}"
        self.up_ = False

    def conf_value(self, text, key, default=""):
        """A KEY=VALUE of the serving conf (a later assignment wins, as when serve_lib.sh sources it)."""
        value = default
        for line in text.splitlines():
            if line.startswith(key + "="):
                value = line[len(key) + 1:].split("#")[0].strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                    value = value[1:-1]
        return value

    def up(self):
        """Rank 0's lock, rank 1, then the front end as a transient user unit, the way serve_lib.sh starts a rank
        with SPAWN_UNIT=1 (serve_unit_start): F's logind has KillUserProcesses=yes, so a process left behind by an
        ssh session - nohup and setsid or not - is killed with the session. The unit's command is a script written
        beside the binary (a unit's command line may not hold `$`): rank 0's resolved environment and nothing else
        (env -i), the conf's CPU pin and wrapper words, the front end, then its exit status appended to its log."""
        pair, sv = self.pair, self.sv
        name = os.path.basename(sv["binary"])
        conf = pair.conf_text(f"{self.s}/deploy", self.drafts, f"{self.s}/run")
        cpus, wrap = self.conf_value(conf, "ENGINE_CPUS"), self.conf_value(conf, "WRAP0")
        if pair.plan:
            pair.say("PLAN front end: rank 0's resolved launch read with serve-rank0.sh check (its argv gives the model "
                     "directory, the context and the kit; its environment is the front end's)")
            argv_model, ctxn, kit, envs = "<model_dir>", "<context>", "<kit>", "<rank 0's environment>"
        else:
            rc, resolved = pair.on(pair.f, f"cd {self.tools} && {self.env} ./serve-rank0.sh check", 120)
            res = json.loads(resolved[resolved.index("{"):])
            argv = res["argv"]
            argv_model, ctxn = argv[argv.index("serve") + 1], argv[argv.index("--context") + 1]
            kit = res["env"].get("TENSORFOLD_HIP_KERNELS") or f"{self.s}/deploy/native-hip/share/tensorfold/hip/gfx1151/dsv41"
            envs = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in res["env"].items())
        pair.started.append(("abi", (self.conf, self.rid)))       # sf.Pair.teardown stops rank 1 and both locks of such a run
        pair.do("GPU lock, rank 0", SSH + [f"pod@{pair.f}", f"cd {self.tools} && {self.env} ./serve-rank0.sh lock"], 120)
        pair.do("rank 1", SSH + [f"pod@{pair.a}", f"cd {self.tools} && {self.env} ./serve-rank1.sh lock 60 && {self.env} ./serve-rank1.sh start --locked"], 180)
        args = (f"--host 0.0.0.0 --port {sv['port']} --model-id {shlex.quote(sv['model_id'])} "
                f"executor.model_dir={argv_model} executor.kit={kit} executor.context={ctxn} "
                f"executor.drafts={'on' if self.drafts else 'off'} text.model_dir={argv_model}")
        script = ("#!/bin/bash\n"
                  f"# Written by gate/flerp_serve_pair.py for run {self.rid}: the front end as rank 0, in rank 0's resolved environment.\n"
                  f"cd {self.s}/serve\n"
                  + (f"taskset -c {cpus} " if cpus else "") + f"env -i {envs} {wrap} ./{name} {args} >> serve.log 2>&1 &\n"
                  "echo $! > serve.pid\n"
                  "wait $!\n"
                  "echo \"front end exit status $?\" >> serve.log\n")
        local = os.path.join(self.ctx["scratch"], f"launch-{self.rid}.sh")
        with open(local, "w") as f:
            f.write(script)
        os.chmod(local, 0o755)
        pair.say(("PLAN " if pair.plan else "RUN  ") + "launch script: " + script.replace("\n", " ; ")[:700])
        pair.do("launch script to rank 0", ["rsync", "-a", "-e", " ".join(SSH), local, f"pod@{pair.f}:{self.s}/serve/launch.sh"])
        unit = f"export XDG_RUNTIME_DIR=/run/user/$(id -u); systemctl --user reset-failed {self.unit} 2>/dev/null; "
        pair.do("the front end as rank 0 (a transient user unit, as serve_lib.sh starts a rank)", SSH + [f"pod@{pair.f}",
                unit + f"rm -f {self.s}/serve/serve.pid; systemd-run --user --quiet --collect --unit {self.unit} "
                f"-p WorkingDirectory={self.s}/serve -- /bin/bash {self.s}/serve/launch.sh"], 60)
        self.up_ = True
        t0, unknown = time.time(), 0
        while not pair.plan and time.time() - t0 < pair.p["startup_timeout_s"]:
            try:
                st, health = http(self.url + "/health", timeout=5)
                if st == 200:
                    return True, round(time.time() - t0, 1), health
            except (urllib.error.URLError, OSError):
                pass
            state = self.alive()
            if state == "gone":
                pair.say(f"the front end's process is gone {round(time.time() - t0)} s after its start")
                break
            unknown = unknown + 1 if state == "unknown" else 0
            if unknown >= 6:                                       # half a minute without an answer from the worker
                pair.say("the front end's state could not be read 6 times in a row (ssh to rank 0's worker fails)")
                break
            time.sleep(5)
        return pair.plan, round(time.time() - t0, 1), {}

    @property
    def unit(self):
        return "strixflerp-serve-" + self.rid.replace("flerp-serve-", "")

    def alive(self):
        """"alive", "gone" or "unknown". The remote command prints one word of its own, so an ssh that failed (no
        word) is never taken for a process that ended. Gone: the unit is not active and the recorded pid is dead."""
        rc, out = self.pair.on(self.pair.f, f"export XDG_RUNTIME_DIR=/run/user/$(id -u); p=$(cat {self.s}/serve/serve.pid 2>/dev/null); "
                               f"if systemctl --user is-active -q {self.unit} || {{ [ -n \"$p\" ] && kill -0 $p 2>/dev/null; }}; "
                               "then echo FRONT-ALIVE; else echo FRONT-GONE; fi", 30)
        return "alive" if "FRONT-ALIVE" in out else "gone" if "FRONT-GONE" in out else "unknown"

    def evidence(self, since):
        """Everything this run left on the two workers, copied into the results before anything is removed, for a
        failure as for a pass: the front end's whole log (the wrapper's exit line is its last), the launch script,
        rank 1's log, the user journal of both workers since the stage began, the units and the processes."""
        pair, tag = self.pair, f"front-drafts-{'on' if self.drafts else 'off'}"
        if pair.plan:
            pair.say(f"PLAN evidence into gate/results before teardown ({tag}-*): serve.log, launch.sh, rank 1's log, "
                     "journalctl --user since the stage began, systemctl --user status and ps, from both workers")
            return
        xdg = "export XDG_RUNTIME_DIR=/run/user/$(id -u); "
        procs = "ps -eo pid,ppid,lstart,etime,stat,args | grep -E 'strixflerp-|tf-dsv41-|tensorfold-native|halo-gpu.lock' | grep -v grep"
        for ip, rank, files in ((pair.f, 0, [f"{self.s}/serve/serve.log", f"{self.s}/serve/launch.sh", f"{self.s}/run/{self.rid}/lock.log"]),
                                (pair.a, 1, [f"{self.s}/run/{self.rid}/rank1.log", f"{self.s}/run/{self.rid}/lock.log"])):
            cmd = (xdg + f"echo EVIDENCE-BEGIN $(hostname) $(date -u +%FT%TZ); "
                   + "".join(f"echo; echo '==== {p}'; tail -c 600000 {p} 2>&1; " for p in files)
                   + f"echo; echo '==== systemctl --user status (this run)'; systemctl --user --no-pager --full status {self.unit if rank == 0 else ''} "
                   f"'tfp-*{self.rid[-16:]}*' 2>&1 | tail -n 60; "
                   f"echo; echo '==== journalctl --user since {since}'; journalctl --user --no-pager -o short-iso --since {shlex.quote(since)} 2>&1 | tail -n 400; "
                   f"echo; echo '==== processes'; {procs}; echo EVIDENCE-END")
            rc, out = pair.on(ip, cmd, 120)
            with open(os.path.join(self.ctx["res"], f"{tag}-rank{rank}-evidence.txt"), "w") as f:
                f.write(out + ("" if "EVIDENCE-END" in out else f"\n(incomplete: the command ended with rc {rc})") + "\n")

    def down(self):
        """SIGTERM to the front end, the wrapper's record of its exit status, then rank 1 and the locks."""
        pair = self.pair
        rc, text = pair.do("stop the front end (SIGTERM; its exit status)", SSH + [f"pod@{pair.f}",
                           f"export XDG_RUNTIME_DIR=/run/user/$(id -u); p=$(cat {self.s}/serve/serve.pid); kill -TERM $p; "
                           f"for i in $(seq 90); do systemctl --user is-active -q {self.unit} || break; sleep 1; done; "
                           f"systemctl --user is-active -q {self.unit} && {{ systemctl --user kill -s KILL {self.unit}; echo killed; }}; "
                           f"sleep 1; tail -n 12 {self.s}/serve/serve.log"], 150)
        pair.do("rank 1 stop, locks released", SSH + [f"pod@{pair.a}", f"cd {self.tools} && {self.env} ./serve-rank1.sh stop; {self.env} ./serve-rank1.sh unlock"], 200)
        pair.do("rank 0 lock released", SSH + [f"pod@{pair.f}", f"cd {self.tools} && {self.env} ./serve-rank0.sh unlock"], 60)
        pair.started = [x for x in pair.started if x != ("abi", (self.conf, self.rid))]
        self.up_ = False
        m = re.search(r"front end exit status (\d+)", text)
        last = [line for line in text.splitlines() if line.startswith(("flerp_close", "shutdown:", "host:"))]
        self.exit = {"killed": "killed" in text, "executor_closed": "shutdown: the executor is closed" in text,
                     "main_returned": "host: main returned" in text, "exit_status": int(m.group(1)) if m else None,
                     "last_step": last[-1] if last else None}
        return (not self.exit["killed"] and self.exit["executor_closed"] and self.exit["exit_status"] == 0), text[-600:]


def write_exit_backtrace(ctx, tag, exit_info):
    """exit-backtrace.txt: the frame that faulted, named.

    The workers have no gdb, core_pattern is apport and `ulimit -c 0`, so the front end prints its own note (serve/http
    main.cpp, install_crash_note): the signal, the address, which phase of the run, the faulting thread's frames, and
    the /proc/self/maps lines they fall in. This reads that note out of the evidence the stage collected and resolves
    it here, where the tools and the unstripped binaries are: a maps line gives the module's file offset of the frame
    (seg_off + addr - seg_start), and addr2line turns that into a function and a source line. A module we have no local
    copy of is named as such, not guessed at."""
    res, tree = ctx["res"], os.path.join(ROOT, "build", ctx["recipe"].name)
    local = [os.path.join(tree, ctx["cfg"]["serve"]["binary"])]
    for pat in ("*/libamdhip64.so*", "*/libomp.so*", "*/libgcc_s.so*"):
        local += sorted(glob.glob(os.path.join("/data/models/tools", "therock*", "lib", os.path.basename(pat))))
    by_name = {}
    for f in local:
        if os.path.isfile(f):
            by_name.setdefault(os.path.basename(f.split(".so")[0]), f)
    lines = [f"# exit backtrace of the front end (dsv41 flerp-serve pair stage), run {tag}", "",
             f"recorded {sf.now()}; the front end's exit as the stage saw it: {json.dumps(exit_info)}", "",
             "The kernel's own line is not available on these workers (dmesg and journalctl -k are refused to pod),",
             "so this is the process's own note. Frames are resolved with addr2line against the local copies of the",
             "modules named in the maps lines; a module without a local copy is left as the note wrote it.", ""]
    src_files = [os.path.join(res, f"front-{tag}-rank0-evidence.txt")]
    note = None
    for f in src_files:
        try:
            text = open(f, errors="replace").read()
        except OSError:
            continue
        i = text.find("\ncrash: ")
        if i < 0:
            continue
        note = text[i + 1:]
        j = note.find("EVIDENCE-END")
        if j > 0:
            note = note[:j].rstrip()
        break
    if note is None:
        lines.append("NO CRASH NOTE in the collected evidence: the process did not fault under the note's signals, or")
        lines.append("the build that ran predates the note (check `host: main returned` in the same file). The stage's")
        lines.append("exit fields above are then the whole record.")
    else:
        lines += ["--- the note, as printed by the process (its stdout, serve.log) ---", "", note, "",
                  "--- resolved frames ---", ""]
        maps = {}
        for m in re.finditer(r"^([0-9a-f]+)-([0-9a-f]+) \S+ ([0-9a-f]+) \S+ \S+\s+(\S+)\s*$", note, re.M):
            lo, hi, off, path = (int(m.group(1), 16), int(m.group(2), 16), int(m.group(3), 16), m.group(4))
            maps.setdefault(os.path.basename(path), (lo, off, path))
        for m in re.finditer(r"^(.+?)\(([^()]+)\)\[0x([0-9a-f]+)\]$", note, re.M):
            path, printed, addr = m.group(1), m.group(2), int(m.group(3), 16)
            key, stem = os.path.basename(path), os.path.basename(path).split(".so")[0]
            if not printed.startswith("+0x"):   # backtrace_symbols_fd knew the symbol: it is already a name
                lines.append(f"{key}({printed}) [{m.group(3)}]")
                continue
            foff = int(printed[3:], 16)
            base = next((k for k in maps if k.split(".so")[0] == stem), None)
            if base is None:
                lines.append(f"{key}(+0x{foff:x}): no maps line for it in the note")
                continue
            lo, seg_off, abspath = maps[base]
            file_off = seg_off + (addr - lo)      # the file offset of the frame: seg_off places the segment in the file
            local_copy = by_name.get(stem) or (abspath if os.path.isfile(abspath) else None)
            if local_copy is None:
                lines.append(f"{key}(+0x{foff:x}) = {abspath} file offset {file_off:#x}: no local copy to resolve")
                continue
            try:
                r = subprocess.run(["addr2line", "-f", "-C", "-i", "-e", local_copy, hex(file_off)],
                                   capture_output=True, text=True, timeout=60)
                got = r.stdout.strip().replace("\n", " <- ") or f"unresolved (rc {r.returncode})"
            except OSError as e:
                got = f"addr2line could not run: {e}"
            lines.append(f"{key}(+0x{foff:x}) file offset {file_off:#x} in {local_copy}: {got}")
    path = os.path.join(res, "exit-backtrace.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def end_ids_of(ctx):
    """The model's end-of-sequence ids, from the config.json of the checkpoint the build stage probes."""
    try:
        with open(os.path.join(ctx["cfg"]["serve"]["text_probe"], "config.json")) as f:
            e = json.load(f).get("eos_token_id")
    except (OSError, ValueError):
        return []
    return [e] if isinstance(e, int) else [x for x in e or [] if isinstance(x, int)]


def origin_reply(ids, finish, end_ids):
    """The reply's tokens out of the origin server's return_token_ids: its list ends with the end-of-sequence id when
    the reply finished by stop (165 ids for a 164-token reply on 2026-11-11: `tokens=165 ... finish=stop` in its
    log), and the front end's token_ids are the reply without it. Returns (reply ids, the end id or None)."""
    if finish == "stop" and ids and ids[-1] in end_ids:
        return ids[:-1], ids[-1]
    return ids, None


def first_diff(a, b):
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), None if len(a) == len(b) else n)


def refusals(url, caps_top_k_max):
    chat = url + "/v1/chat/completions"
    base = {"messages": [{"role": "user", "content": "Name three rivers in Europe."}], "max_tokens": 4}
    cases = [("logprobs", dict(base, temperature=0, logprobs=True), "executor_cannot"),
             ("logit_bias", dict(base, temperature=0, logit_bias={"100": -100}), "executor_cannot"),
             ("temperature-without-top_k", dict(base, temperature=0.7, top_k=0), "executor_cannot"),
             ("top_k-above-the-bound", dict(base, temperature=0.7, top_k=caps_top_k_max + 1000), "executor_cannot"),
             ("tools", dict(base, temperature=0, tools=[{"type": "function", "function": {"name": "f"}}]), "not_served")]
    out = []
    for name, body, code in cases:
        st, r = http(chat, body, 60)
        out.append({"check": "refuse:" + name, "status": st, "code": r.get("error", {}).get("code"),
                    "pass": st == 400 and r.get("error", {}).get("code") == code})
    # Amendment (as f0457f6 for flerp_serve): FLERP-CONSTRAIN (9ef8d14) serves response_format on purpose, so the
    # check that expected "not_served" is replaced by one that expects the constraint to hold: HTTP 200 and a reply
    # that validates against the schema sent. Every other refusal is unchanged.
    schema = {"type": "object", "properties": {"rivers": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3}},
              "required": ["rivers"], "additionalProperties": False}
    body = dict(base, max_tokens=256, temperature=0, enable_thinking=False,
                response_format={"type": "json_schema", "json_schema": {"name": "rivers", "schema": schema, "strict": True}})
    st, r = http(chat, body, 120)
    content, valid = None, False
    if st == 200:
        try:
            content = r["choices"][0]["message"]["content"]
            valid = schema_valid(json.loads(content), schema)
        except (KeyError, IndexError, TypeError, ValueError):
            pass
    out.append({"check": "serve:response_format", "status": st, "content": content, "pass": st == 200 and valid})
    return out


def schema_valid(v, sc):
    """The subset of JSON Schema the gate's own schema uses: type, properties, required, additionalProperties, items, minItems, maxItems."""
    t = sc.get("type")
    types = {"object": dict, "array": list, "string": str}
    if t in types and not isinstance(v, types[t]):
        return False
    if t == "object":
        props = sc.get("properties", {})
        if any(k not in v for k in sc.get("required", [])):
            return False
        if sc.get("additionalProperties") is False and any(k not in props for k in v):
            return False
        return all(schema_valid(x, props[k]) for k, x in v.items() if k in props)
    if t == "array":
        if not sc.get("minItems", 0) <= len(v) <= sc.get("maxItems", len(v)):
            return False
        return all(schema_valid(x, sc["items"]) for x in v) if "items" in sc else True
    return True


def ask(pair, front, native, native_sampled):
    """Every configured request to the loaded front end, against the origin's ids."""
    sv, out = front.sv, {"cases": [], "extra": []}
    ends = end_ids_of(front.ctx)
    chat = front.url + "/v1/chat/completions"
    for name, body in pair.requests():
        body = dict({k: v for k, v in body.items() if k != "return_token_ids"}, flerp_trace=True)
        st, r = http(chat, body, pair.run["request_timeout_s"])
        if st != 200:
            out["cases"].append({"name": name, "pass": False, "error": f"HTTP {st}: {json.dumps(r)[:300]}"})
            continue
        t, ref = r["flerp"], native[name]
        s = sse(chat, dict(body, stream=True), pair.run["request_timeout_s"])
        want, end = origin_reply(ref["generated"], ref["finish_reason"], ends)
        rec = {"name": name, "prompt_tokens": len(ref["prompt"]), "native_tokens": len(want), "tokens": len(t["token_ids"]),
               "native_end_id": end, "native_token_ids": ref["generated"], "token_ids": t["token_ids"],
               "prompt_equal": t["prompt_token_ids"] == ref["prompt"], "tokens_equal": t["token_ids"] == want,
               "first_token_difference": first_diff(t["token_ids"], want),
               # both count the end-of-sequence token: the origin's list holds it, the front end's usage counts it
               "completion_tokens": r["usage"]["completion_tokens"], "completion_tokens_equal": r["usage"]["completion_tokens"] == len(ref["generated"]),
               "delivery": t["delivery"], "sampled_by": t["sampled_by"], "verifies": t["verifies"], "drafted": t["drafted"],
               "accepted": t["accepted"], "finish_reason": r["choices"][0]["finish_reason"], "native_finish_reason": ref["finish_reason"],
               "decode_tps": round(r["timings"]["predicted_per_second"], 2), "native_decode_tps": round(ref.get("decode_tps") or 0.0, 2),
               "stream_equal": s["done"] and s["token_ids"] == t["token_ids"] and s["text"] == r["choices"][0]["message"]["content"]
                               and (s["flerp"] or {}).get("token_ids") == t["token_ids"]}
        rec["pass"] = (rec["prompt_equal"] and rec["tokens_equal"] and rec["completion_tokens_equal"] and rec["stream_equal"]
                       and rec["delivery"] == "tokens" and rec["sampled_by"] == "executor" and rec["finish_reason"] == rec["native_finish_reason"])
        out["cases"].append(rec)
    st, health = http(front.url + "/health", timeout=30)
    out["health"] = health
    out["extra"].append({"check": "health", "pass": st == 200 and health.get("sampled_by") == "executor"
                         and health.get("draft_form") == ("held" if front.drafts else "none")})
    out["extra"] += refusals(front.url, 1000000)
    # recorded: a sampled request, the same body the origin's server answered
    name, body = pair.requests()[0]
    st, r = http(chat, dict({k: v for k, v in body.items() if k != "return_token_ids"}, flerp_trace=True, **sv["sampled"]), pair.run["request_timeout_s"])
    out["sampled"] = {"status": st, "token_ids": r.get("flerp", {}).get("token_ids") if st == 200 else None,
                      "finish_reason": r["choices"][0]["finish_reason"] if st == 200 else None,
                      "error": None if st == 200 else json.dumps(r)[:300]}
    if st == 200 and native_sampled.get("token_ids") is not None:
        want, _ = origin_reply(native_sampled["token_ids"], native_sampled.get("finish_reason"), ends)
        out["sampled"].update(equal_native=out["sampled"]["token_ids"] == want, first_difference=first_diff(out["sampled"]["token_ids"], want))
    out["pass"] = bool(out["cases"]) and all(c["pass"] for c in out["cases"]) and all(e["pass"] for e in out["extra"])
    return out


def stage_pair(ctx, plan=False):
    pair = sf.Pair(ctx, plan)
    open(pair.log, "a").close()
    for line in sf.pair_nodes.describe(pair.nodes).splitlines():
        pair.say(line)
    p, cfg, sv = pair.p, ctx["cfg"], ctx["cfg"]["serve"]
    # started_local: the stage's start as `journalctl --since` takes it (the workers keep UTC)
    out = {"started": sf.now(), "plan_only": plan, "evidence": "pair.txt", "pair": pair.nodes, "pair_recorded": pair.recorded,
           "started_local": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}
    tree = os.path.join(ROOT, "build", ctx["recipe"].name)
    binary = os.path.join(tree, sv["binary"])
    if plan:
        pair.say("PLAN the pair's state is not read: this plan opens no connection to a worker and calls no pod tool. "
                 "A run first requires the pair free (sf.Pair.state, sf.Pair.free), or with --take-from-launcher only the "
                 "pod launcher's own serving run on it")
    else:
        out["before"] = pair.state()
        why = pair.free(out["before"])
        take = ctx.get("take") and pair.only_launcher_serving(out["before"])
        out.update(free=not why, taken_from_launcher=bool(take))
        if why and not take:
            out.update({"pass": False, "blocked": "the pair is not free: " + "; ".join(why)})
            pair.say("blocked: " + out["blocked"])
            return out
        if not os.path.isfile(os.path.join(tree, "built.json")) or not os.path.isfile(binary):
            out.update({"pass": False, "blocked": f"{os.path.relpath(tree, ROOT)} is not built: run the build stage"})
            return out
    hold = f"OWNER={shlex.quote(p['owner'])} REASON={shlex.quote(p['why'])} TTL={p['hold_ttl_s']} pod-quad hold {pair.f} {pair.a}"
    rc, text = pair.do("hold both workers (one call; takes the pod-lease worker leases)", hold, 900)
    if rc != 0:
        out.update({"pass": False, "blocked": f"pod-quad hold refused (rc {rc}): {text[-300:]}"})
        return out
    out.update(runs=[], hold_taken=sf.now())
    left = lambda: ctx["release_by"] - time.time()
    front = None
    try:
        if not plan:
            ok, ws = pair.idle(360 if out["taken_from_launcher"] else 0)
            if not ok:
                raise RuntimeError(f"the pair is not idle under the hold: {ws}")
        # the assembled deployment (this recipe's native-hip tree, the reference's launch tools) and the front end
        s, ref_deploy = p["worker_scratch"], p["reference_deploy"]
        meta = os.path.join(ctx["scratch"], "DEPLOY.json")
        with open(meta, "w") as f:
            json.dump({"schema": "tf-port-deploy/1", "commit": p["reference_commit"], "clean": True, "host": socket.gethostname(),
                       "created_utc": sf.now(), "assembled_by": f"strixflerp recipe {ctx['recipe'].name}"}, f, indent=1)
        for ip in (pair.f, pair.a):
            rc, _ = pair.do(f"scratch on {ip}", SSH + [f"pod@{ip}", f"mkdir -p {s}/deploy {s}/conf {s}/run {s}/abi {s}/serve && "
                            f"cp -a {ref_deploy}/tools {s}/deploy/ && chmod -R u+w {s}/deploy"])
            copies = [(os.path.join(tree, "tree/zig-out/native-hip"), f"{s}/deploy/"), (meta, f"{s}/deploy/DEPLOY.json")]
            if ip == pair.f:
                copies.append((os.path.realpath(binary) if not plan else binary, f"{s}/serve/"))
            for src, dst in copies:
                rc2, _ = pair.do(f"copy to {ip}", ["rsync", "-a", "-e", " ".join(SSH), src, f"pod@{ip}:{dst}"], 900)
                rc = rc or rc2
            if rc:
                raise RuntimeError("staging the assembled deployment on the workers failed")
        for n_done, drafts in enumerate(cfg["run"]["drafts"]):
            tag = f"drafts-{'on' if drafts else 'off'}"
            if left() < p["setting_margin_s"]:               # measured, in gate/dsv41-flash.strixfold-tp2.toml
                out.setdefault("skipped", []).append(f"{tag}: {int(left())} s to the hard stop, a reference and a front end load do not fit")
                pair.say(out["skipped"][-1])
                continue
            ref_conf = pair.put_conf(f"reference-{tag}.conf", pair.conf_text(p["reference_deploy"], drafts))
            asm_name = f"assembled-{tag}.conf"
            pair.put_conf(asm_name, pair.conf_text(f"{s}/deploy", drafts, f"{s}/run"))
            run = {"drafts": drafts}
            # 1. the origin's server, by the pod's launcher
            ok = plan
            if plan:                                 # sf.Pair.reference reads the run id from a worker: not in a plan
                pair.do("reference up (the pod's launcher)", f"SP_CONF={shlex.quote(ref_conf)} KANBAN_ACTOR={p['owner']} {p['launcher']} up", 300)
            for attempt in range(1, 0 if plan else p["reference_attempts"] + 1):
                ok, secs, rid = pair.reference(drafts, ref_conf)
                if ok:
                    break
                pair.say(f"reference start {attempt} did not reach health after {secs} s (run {rid})")
                pair.reference_down()
                if not plan:
                    pair.idle()
            if not ok and not plan:
                raise RuntimeError(f"the reference deployment did not reach health in {p['reference_attempts']} attempts")
            native = {}
            if plan:
                pair.say(f"PLAN the origin's server answers {len(pair.requests())} greedy chat requests (return_token_ids) and one "
                         "sampled request, and tokenizes each prompt (POST /tokenize)")
            else:
                ref = pair.serve_and_ask(f"reference-{tag}", f"cd ~/tfp/run/serve128k/{rid}")
                for name, body in pair.requests():
                    r = next(x for x in ref["runs"] if x["prompt"] == name)
                    native[name] = {"generated": r["token_ids"], "prompt": sf.prompt_ids(pair, body), "finish_reason": r["finish_reason"],
                                    "decode_tps": r["decode_tokens_per_second"]}
                # recorded only, and last: on 2026-11-11T02:54:52Z this request killed the reference server with
                # drafts off (SIGSEGV at 0x1 after 148 one-row rounds; reference-drafts-off-crash-on-sampled-request.txt).
                # A failure here is the origin's and is written down; it does not end the stage.
                name, body = pair.requests()[0]
                try:
                    reply, _ = pair.post(dict(body, **sv["sampled"]))
                    run["native_sampled"] = {"token_ids": (reply.get("tensorfold") or {}).get("token_ids"),
                                             "finish_reason": reply["choices"][0].get("finish_reason")}
                except Exception as e:
                    run["native_sampled"] = {"token_ids": None, "error": f"{type(e).__name__}: {e}"}
                    pair.say(f"the origin's server did not answer the sampled request ({run['native_sampled']['error']}); recorded, the stage goes on")
                    _, log = pair.on(pair.f, f"tail -n 30 ~/tfp/run/serve128k/{rid}/rank0.log 2>/dev/null", 60)
                    with open(os.path.join(ctx["res"], f"reference-{tag}-sampled-request-failed.txt"), "w") as f:
                        f.write(log + "\n")
                run["reference"] = {"run": rid, "startup_s": secs, "requests": len(ref["runs"])}
            pair.reference_down()
            if not plan:
                ok, ws = pair.idle()
                if not ok:
                    raise RuntimeError(f"GPU memory did not return to idle after the reference: {ws}")
            # 2. the front end as rank 0 of this recipe's build
            front = Front(pair, ctx, asm_name, drafts)
            ok, secs, health = front.up()
            if not ok and not plan:
                raise RuntimeError(f"the front end did not reach health in {secs} s (front-{tag}-rank0-evidence.txt, front-{tag}-rank1-evidence.txt)")
            if plan:
                pair.say(f"PLAN the front end answers the same requests over HTTP (flerp_trace), not streamed and streamed; "
                         "6 refusals; one sampled request")
            else:
                run["front"] = dict(ask(pair, front, native, run["native_sampled"]), startup_s=secs, run=front.rid)
                run["sampled_equal_native"] = run["front"]["sampled"].get("equal_native")
            front.evidence(out["started_local"])
            front.collected = True
            clean, tail = front.down()
            run["front_clean_exit"] = clean or plan
            run["front_exit"] = getattr(front, "exit", None)
            if not plan:
                run["exit_backtrace"] = os.path.relpath(write_exit_backtrace(ctx, tag, run["front_exit"]), ROOT)
                if not clean:
                    pair.say("the front end did not exit clean; the faulting frame: " + run["exit_backtrace"])
            if not plan:
                ok, ws = pair.idle()
                if not ok:
                    raise RuntimeError(f"GPU memory did not return to idle after the front end: {ws}")
            out["runs"].append(run)
    except Exception as e:                       # the stage fails with its reason; the teardown still runs
        out["error"] = f"{type(e).__name__}: {e}"
        pair.say("error: " + out["error"])
    finally:
        if front is not None and front.up_:
            if not getattr(front, "collected", False):   # a failure: what the run left, before anything is stopped or removed
                try:
                    front.evidence(out["started_local"])
                except Exception as e:
                    pair.say(f"evidence could not be collected: {type(e).__name__}: {e}")
            pair.do("front end leftovers", SSH + [f"pod@{pair.f}", f"export XDG_RUNTIME_DIR=/run/user/$(id -u); "
                    f"systemctl --user kill -s KILL {front.unit} 2>/dev/null; systemctl --user stop {front.unit} 2>/dev/null; true"], 60)
        if plan:
            pair.say("PLAN teardown (sf.Pair.teardown): stop what this stage still has running, remove the scratch directories")
            idle = True
        else:
            idle, _ = pair.teardown()
        rc, _ = pair.do("release the hold", f"OWNER={shlex.quote(p['owner'])} pod-quad release", 300)
        out["hold_released"] = sf.now()
        if not plan:
            out["after"] = pair.state()
            out["released"] = rc == 0 and not any(out["after"]["holds"].values())
            out["gpu_idle_after"] = idle
    out["finished"] = sf.now()
    out["pass"] = (not plan and "error" not in out and len(out["runs"]) == len(cfg["run"]["drafts"])
                   and all(r.get("front", {}).get("pass") and r["front_clean_exit"] for r in out["runs"])
                   and out.get("released") and out.get("gpu_idle_after"))
    return out


def write_summary(ctx, results):
    ok = SM.all_pass(STAGES, results)
    SM.write_json(ctx["res"], {"recipe": ctx["recipe"].name, "host": socket.gethostname(), "date": sf.now(), "pass": ok, "stages": results}, ctx=ctx)
    lines = [f"# Front end gate (StrixFold executor): {ctx['recipe'].name}", "",
             f"Host {socket.gethostname()}, {sf.now()}. Stages run: {', '.join(results) or 'none'}; not run: "
             f"{', '.join(s for s in STAGES if s not in results) or 'none'}.", ""]
    lines += SM.stage_bullets(STAGES, results)
    b = results.get("build", {})
    if b.get("binary"):
        lines += ["", f"Built `{ctx['cfg']['serve']['binary']}`: {b['binary']['bytes']} bytes, sha256 {b['binary']['sha256'][:16]}; "
                  f"{b.get('modules')} modules, {b.get('files')} tree files, {b.get('c_link_units')} C and C++ units linked by the Zig build.",
                  "", "Without a GPU or a worker: " + "; ".join(f"{k}: {'ok' if v['ok'] else 'FAILED'}" + (f" (`{v['stdout'].splitlines()[0][:160]}`)" if v.get("stdout") else "")
                                                               for k, v in b.get("probes", {}).items()) + "."]
    g = results.get("pair", {})
    for run in g.get("runs", []):
        f = run.get("front", {})
        lines += ["", f"Drafts {'on' if run['drafts'] else 'off'}:", "", "| request | prompt ids equal | tokens origin / front end | tokens equal | "
                  "streamed equal | drafts accepted / made | decode tok/s origin / front end | pass |", "|---|---|---|---|---|---|---|---|"]
        for c in f.get("cases", []):
            if "error" in c:
                lines.append(f"| {c['name']} | - | - | - | - | - | - | NO: {c['error']} |")
            else:
                lines.append(f"| {c['name']} | {'yes' if c['prompt_equal'] else 'NO'} | {c['native_tokens']} / {c['tokens']} | "
                             f"{'yes' if c['tokens_equal'] else 'NO, from token ' + str(c['first_token_difference'])} | "
                             f"{'yes' if c['stream_equal'] else 'NO'} | {c['accepted']} / {c['drafted']} | "
                             f"{c['native_decode_tps']} / {c['decode_tps']} | {'yes' if c['pass'] else 'NO'} |")
        lines += ["", "Also required: " + "; ".join(f"{e['check']}: {'pass' if e['pass'] else 'FAIL'}" for e in f.get("extra", []))
                  + f"; clean exit on SIGTERM: {'yes' if run.get('front_clean_exit') else 'NO'}"
                  + (f" (executor closed: {x['executor_closed']}; main returned: {x['main_returned']}; exit status {x['exit_status']}; "
                     f"last line: `{x['last_step']}`)" if (x := run.get("front_exit")) else "") + ".",
                  f"Recorded: the sampled request's tokens equal the origin's: {run.get('sampled_equal_native')}"
                  + (f" (the origin's server: {run['native_sampled']['error']})" if isinstance(run.get("native_sampled"), dict) and run["native_sampled"].get("error") else "") + "."]
    SM.write_md(ctx["res"], lines)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--plan", action="store_true", help="pair stage: print what would run; change nothing, connect to nothing")
    ap.add_argument("--take-from-launcher", action="store_true", help="pair stage: take the pair from the pod launcher's own serving run")
    ap.add_argument("--quiet-s", type=int, help="pair stage: override the config's quiet period (an agreed window)")
    ap.add_argument("--release-by", metavar="HH:MM", help="pair stage: hard stop (UTC today); loads that do not fit are skipped")
    sf.pair_nodes.add_args(ap)
    a = ap.parse_args()
    cat = Catalog(ROOT)
    recipe = cat.recipe(a.recipe)
    if recipe is None:
        sys.exit(f"no recipe {a.recipe}")
    cfg = p1.load_config(os.path.join(ROOT, recipe.gate["config"]))
    res = os.path.join(ROOT, "gate", "results", recipe.name)
    os.makedirs(res, exist_ok=True)
    ctx = {"cat": cat, "recipe": recipe, "cfg": cfg, "res": res, "scratch": os.path.join(ROOT, cfg["host"]["scratch"]),
           "take": a.take_from_launcher, "release_by": time.time() + 365 * 86400, "pair_args": a}
    if a.quiet_s is not None:
        cfg["pair"]["quiet_s"] = a.quiet_s
    if a.release_by:
        hh, mm = (int(x) for x in a.release_by.split(":"))
        ctx["release_by"] = datetime.datetime.now(datetime.timezone.utc).replace(hour=hh, minute=mm, second=0, microsecond=0).timestamp()
    os.makedirs(ctx["scratch"], exist_ok=True)
    results = {}
    prior = os.path.join(res, "summary.json")
    if os.path.isfile(prior) and set(a.stages.split(",")) != set(STAGES):
        with open(prior) as f:
            results = json.load(f).get("stages", {})
    fns = {"check": sf.stage_check, "build": stage_build, "pair": lambda c: stage_pair(c, a.plan)}
    for s in STAGES:
        if s not in a.stages.split(","):
            continue
        print(f"== {s}", flush=True)
        r = fns[s](ctx)
        if s == "pair" and a.plan:
            return 0                                        # a plan is not a result: no summary is written
        results[s] = r
        print(f"   {'PASS' if r.get('pass') else 'FAIL'} {r.get('blocked', '')}{r.get('error', '')}", flush=True)
        if not r.get("pass"):
            break
    return 0 if write_summary(ctx, {s: results[s] for s in STAGES if s in results}) else 1


if __name__ == "__main__":
    sys.exit(main())
