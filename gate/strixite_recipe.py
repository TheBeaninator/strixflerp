#!/usr/bin/env python3
"""Recipe gate for a strixite-origin recipe (SPEC.md section 9, phase 1). Run on the gfx1151 build host:

    ./flerp gate recipe qwen38-flash-next.strixite [--stages check,origin,build,unit,gpu] [--keep-going]

Stages, each writing evidence under gate/results/<recipe>/:
  check   `flerp check --origin <the recipe's origin>` has no errors (the whole catalogue's result is recorded too).
  origin  every vendored file of the origin is byte-identical to the reference checkout, and the checkout is covered.
  build   `flerp assemble`, then cmake + ninja; every binary of the recipe links.
  unit    the origin's host-only test programs, built from the assembled tree, exit 0.
  gpu     the origin's GPU test, then token identity and tokens per second against the reference server:
          the same greedy requests (temperature 0, fixed seed) go to the reference binary and to the assembled one,
          one server loaded at a time, with MTP drafting on and off. Token ids come from each server's own capture
          files (--capture-dir): prompt ids, generated ids and the MTP step records must be equal request for
          request. Decode speed passes when the medians differ by at most the configured tolerance, or the
          assembled median lies inside the reference's own run-to-run range.

Host facts (weights, port, reference binary, lease) come from the recipe's [gate] config file. The script never
leaves a server running: servers die with this process, and the GPU lease is released on the way out.

Swapped variants (SPEC.md section 9, phase 4). A recipe that binds another engine's kernel through an adapter runs
the same stages against the same reference binary, with three differences, all read from the recipe and its
config and none of them a relaxation of the unmodified recipe's gate:
  - every adapter of the assembly must have launched its foreign provider in every assembled server (the
    adapter's own tally lines in the server log); a swap that never ran fails the gate;
  - [gate] criterion = "token-identity" (the default) asks for equal tokens as above. "token-agreement" is for a
    swap whose numeric contract differs (a different summation order): the module gate named by the config's
    [variant] module_gate must have passed, and the gate records, request for request, where the generated
    tokens first differ from the reference's. It reports that as measured - identical where they were identical,
    diverged at token N where they were not - and never calls a diverged run identical;
  - decode tokens per second are recorded next to the reference's and are a pass criterion only when the config
    says so ([variant] speed_is_criterion; a foreign kernel may be slower or faster, that is a finding).
"""
import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from flerplib.catalog import Catalog  # noqa: E402

STAGES = ("check", "origin", "build", "unit", "gpu")


def sh(cmd, log, cwd=ROOT, timeout=None, env=None):
    """Run a command, tee its output into `log`, return (rc, seconds)."""
    t0 = time.time()
    with open(log, "ab") as f:
        f.write(("$ " + (cmd if isinstance(cmd, str) else " ".join(cmd)) + "\n").encode())
        f.flush()
        try:
            rc = subprocess.call(cmd, shell=isinstance(cmd, str), cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                                 timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            f.write(b"\nTIMEOUT\n")
            rc = 124
    return rc, round(time.time() - t0, 1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tail(path, n=12):
    with open(path, errors="replace") as f:
        return f.read().splitlines()[-n:]


# --- prompts --------------------------------------------------------------------------------------------------
WORDS = ("river stone amber signal lantern harbour copper window orchard meadow engine silver thread market "
         "bridge winter garden letter candle mountain paper anchor shadow forest island mirror ribbon valley "
         "kettle marble pocket summer tunnel violin wagon yellow zephyr basket cedar dragon ember falcon").split()


def long_text(sentences):
    """Deterministic filler: numbered sentences from a fixed linear congruential sequence."""
    x, out = 12345, []
    for i in range(sentences):
        ws = []
        for _ in range(14):
            x = (1103515245 * x + 12345) % (1 << 31)
            ws.append(WORDS[(x >> 16) % len(WORDS)])
        out.append(f"Record {i + 1}: the {ws[0]} {ws[1]} was moved from the {ws[2]} {ws[3]} to the {ws[4]} {ws[5]} "
                   f"by the {ws[6]} {ws[7]}, then logged under {ws[8]}-{ws[9]}-{(x >> 8) % 1000:03d} beside the "
                   f"{ws[10]} {ws[11]} and the {ws[12]} {ws[13]}.")
    return "\n".join(out)


def requests(cfg):
    seed, n = cfg["run"]["seed"], cfg["run"]["max_tokens"]
    common = {"temperature": 0, "seed": seed, "max_tokens": n, "stream": False}
    filler = long_text(cfg["run"]["long_prompt_sentences"])
    return [
        ("short-chat-thinking", "/v1/chat/completions", dict(common, enable_thinking=True, messages=[
            {"role": "user", "content": "A train leaves at 09:40 and arrives at 13:05. How long is the trip? "
                                        "Then name three rivers in Europe."}])),
        ("long-chat", "/v1/chat/completions", dict(common, enable_thinking=False, messages=[
            {"role": "system", "content": "You answer questions about the records you are given."},
            {"role": "user", "content": filler + "\n\nList the first five record numbers whose sentence mentions "
                                                 "a harbour, and quote each of those sentences."}])),
        ("raw-completion", "/v1/completions", dict(common, prompt="The following is a careful step by step "
                                                   "explanation of how a suspension bridge carries its load.\n\n1.")),
    ]


# --- GPU and lease --------------------------------------------------------------------------------------------
def gtt_used():
    total = 0
    drm = "/sys/class/drm"
    for card in sorted(os.listdir(drm)) if os.path.isdir(drm) else []:
        p = os.path.join(drm, card, "device", "mem_info_gtt_used")
        if os.path.isfile(p):
            total += int(open(p).read().strip() or 0)
    return total


def gpu_users(pattern):
    """Processes on this host whose program name (argv[0]'s basename, not its arguments) matches the pattern."""
    rx, out = re.compile(pattern), []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = f.read().split(b"\0")
        except OSError:
            continue
        name = os.path.basename(argv[0].decode(errors="replace")) if argv and argv[0] else ""
        if name and rx.search(name):
            out.append(f"{pid} {name}")
    return out


def gpu_free(cfg):
    """(ok, description). Free = no engine process on this host and the GPU's GTT use under the configured limit."""
    users = gpu_users(cfg["host"]["gpu_busy_pattern"])
    used = gtt_used()
    ok = not users and used <= cfg["host"]["gpu_free_gtt_bytes"]
    return ok, {"gtt_used_bytes": used, "engine_processes": users}


class Lease:
    def __init__(self, cfg, log):
        self.res, self.owner = cfg["host"].get("lease_resource"), cfg["host"].get("lease_owner", "strixflerp-gate")
        self.ttl, self.log, self.held = cfg["host"].get("lease_ttl_s", 3600), log, False
        self.tool = shutil.which("pod-lease")

    def acquire(self, why):
        if not self.res or not self.tool:
            return "no lease tool or resource configured: running without a lease"
        rc, _ = sh([self.tool, "acquire", self.res, "--owner", self.owner, "--why", why, "--ttl", str(self.ttl),
                    "--probe", f"pid=control:{os.getpid()}"], self.log)
        if rc != 0:
            raise RuntimeError(f"pod-lease acquire {self.res} failed (rc {rc}): the GPU is leased to someone else; "
                               f"see {self.log}")
        self.held = True
        return f"holding {self.res} as {self.owner}"

    def release(self):
        if self.held:
            sh([self.tool, "release", self.res, "--owner", self.owner], self.log)
            self.held = False


def pod_waiters():
    """The pod's own GPU jobs that are waiting their turn: processes run as `pod-gpu ... --wait ...`."""
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = [a.decode(errors="replace") for a in f.read().split(b"\0")]
        except OSError:
            continue
        if any(os.path.basename(a) == "pod-gpu" for a in argv[:3]) and any(a == "--wait" or a.startswith("--wait=") for a in argv[:6]):
            out.append(int(pid))
    return out


def utc(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


class Hold:
    """The GPU lease, held only while a model is loaded, and given up between loads when a job of the pod has
    started waiting for the GPU (`pod-gpu --wait`). Every hold is recorded in out["holds"], every wait in
    out["yields"]."""

    def __init__(self, cfg, log, out, wait_s=1800):
        self.cfg, self.out, self.wait_s = cfg, out, wait_s
        open(log, "w").close()
        self.lease = Lease(cfg, log)
        out["holds"], out["yields"] = [], []
        self.t0 = self.why = None

    def take(self, why, short=None):
        """Waits (at most wait_s) until no pod job waits for the GPU and the GPU is idle, then takes the lease."""
        t0 = time.time()
        while True:
            waiters = pod_waiters()
            ok, state = gpu_free(self.cfg)
            if not waiters and ok:
                break
            if time.time() - t0 > self.wait_s:
                raise RuntimeError(f"the GPU is not free: {len(waiters)} pod-gpu processes waiting, {state}, "
                                   f"after {int(time.time() - t0)} s")
            time.sleep(15)
        if time.time() - t0 > 20:
            self.out["yields"].append({"before": short or why, "waited_s": round(time.time() - t0)})
        text = self.lease.acquire(why)
        self.t0, self.why = time.time(), short or why
        return text

    def give(self):
        if self.t0 is not None:
            self.lease.release()
            self.out["holds"].append({"for": self.why, "from": utc(self.t0), "to": utc(), "seconds": round(time.time() - self.t0, 1),
                                      "pod_waiters_at_release": len(pod_waiters())})
            self.t0 = None

    def between(self, why, short=None):
        """Between two loads: when a pod job waits, the lease goes back until none does."""
        if self.t0 is not None and not pod_waiters():
            return
        self.give()
        self.take(why, short)


# --- servers --------------------------------------------------------------------------------------------------
def die_with_parent():
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG


class Server:
    def __init__(self, binary, args, port, log_path):
        self.port, self.log_path = port, log_path
        self.log = open(log_path, "wb")
        self.t0 = time.time()
        self.proc = subprocess.Popen([binary] + args, stdout=self.log, stderr=subprocess.STDOUT,
                                     preexec_fn=die_with_parent)

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def wait_ready(self, timeout):
        while time.time() - self.t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with {self.proc.returncode} during startup; see {self.log_path}")
            try:
                with urllib.request.urlopen(self.url("/health"), timeout=2) as r:
                    if r.status == 200:
                        return round(time.time() - self.t0, 1)
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(1)
        raise RuntimeError(f"server not ready after {timeout} s; see {self.log_path}")

    def post(self, path, body, timeout):
        req = urllib.request.Request(self.url(path), data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{path}: HTTP {e.code}: {e.read()[:400].decode(errors='replace')}") from None

    def stop(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(30)
        self.log.close()
        return self.proc.returncode


TALLY = re.compile(r"^flerp-adapter (\S+) provider=(\S+) foreign_launches=(\d+) origin_launches=(\d+) \(([^)]*)\)", re.M)


def adapter_tallies(log_path):
    """The adapters' own account of what they launched, from a server log: adapter id -> the last line it wrote
    (the exit line when the server shut down cleanly, else the line of its first foreign launch)."""
    with open(log_path, errors="replace") as f:
        text = f.read()
    out = {}
    for adapter, provider, foreign, origin, when in TALLY.findall(text):
        out[adapter] = {"provider": provider, "foreign_launches": int(foreign), "origin_launches": int(origin),
                        "line": when}
    return out


def merge(base, over):
    """`over` on top of `base`, tables merged key by key (a variant's config names only what it changes)."""
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    if "inherit" in cfg:
        cfg = merge(load_config(os.path.join(ROOT, cfg["inherit"])), {k: v for k, v in cfg.items() if k != "inherit"})
    return cfg


def read_capture(path):
    """serve/capture.hpp: one JSON header line, tokens_n little-endian int32, steps_n records of 3 bytes."""
    with open(path, "rb") as f:
        data = f.read()
    nl = data.index(b"\n")
    head = json.loads(data[:nl])
    n, s = head["tokens_n"], head["steps_n"]
    body = data[nl + 1:]
    if len(body) != 4 * n + 3 * s:
        raise RuntimeError(f"{path}: {len(body)} body bytes, header says {4 * n} + {3 * s}")
    tokens = list(struct.unpack(f"<{n}i", body[:4 * n]))
    steps = body[4 * n:]
    return {"id": head["id"], "prompt_n": head["prompt_n"], "finish": head.get("finish"),
            "prompt": tokens[:head["prompt_n"]], "generated": tokens[head["prompt_n"]:],
            "steps": [list(steps[i:i + 3]) for i in range(0, len(steps), 3)]}


def ids_hash(ids):
    return hashlib.sha256(struct.pack(f"<{len(ids)}i", *ids)).hexdigest()[:16]


def run_server(label, binary, mtp, cfg, reqs, scratch):
    """Load one server, send every request `repeats` times, stop it, read back its captures."""
    h, run = cfg["host"], cfg["run"]
    cap = os.path.join(scratch, label, "capture")
    shutil.rmtree(os.path.join(scratch, label), ignore_errors=True)
    os.makedirs(cap)
    ok, state = gpu_free(cfg)
    if not ok:
        raise RuntimeError(f"{label}: the GPU is not free before loading: {state}")
    args = ["--weights", h["weights"], "--ngram", h["ngram"], "--tokenizer", h["tokenizer"],
            "--generation-config", h["generation_config"], "--host", "127.0.0.1", "--port", str(h["port"]),
            "--mtp", "on" if mtp else "off", "--capture-dir", cap] + [str(a) for a in cfg["server"]["args"]]
    out = {"label": label, "binary": binary, "binary_sha256": sha256(binary), "mtp": mtp, "args": args,
           "gpu_before": state, "runs": [], "host_loadavg_before": [round(x, 2) for x in os.getloadavg()]}
    srv = Server(binary, args, h["port"], os.path.join(scratch, label, "server.log"))
    try:
        out["startup_s"] = srv.wait_ready(run["startup_timeout_s"])
        out["gtt_loaded_bytes"] = gtt_used()
        for rep in range(run["repeats"]):
            for name, path, body in reqs:
                r = srv.post(path, body, run["request_timeout_s"])
                choice = r["choices"][0]
                text = choice.get("text")
                if text is None:
                    msg = choice.get("message", {})
                    text = (msg.get("reasoning_content") or "") + "\x1f" + (msg.get("content") or "")
                out["runs"].append({"prompt": name, "repeat": rep, "timings": r.get("timings", {}),
                                    "usage": r.get("usage", {}), "strix": r.get("strix", {}),
                                    "finish_reason": choice.get("finish_reason"),
                                    "text_sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
                                    "text_head": text[:160]})
    finally:
        out["exit_code"] = srv.stop()
        out["server_log_tail"] = tail(srv.log_path, 6)
        out["adapter_tallies"] = adapter_tallies(srv.log_path)
        out["host_loadavg_after"] = [round(x, 2) for x in os.getloadavg()]
    time.sleep(cfg["run"]["settle_s"])
    out["gpu_after"] = gpu_free(cfg)[1]
    caps = sorted((read_capture(os.path.join(cap, f)) for f in os.listdir(cap) if f.endswith(".cap")),
                  key=lambda c: c["id"])
    if len(caps) != len(out["runs"]):
        raise RuntimeError(f"{label}: {len(caps)} capture files for {len(out['runs'])} requests")
    for r, c in zip(out["runs"], caps):
        r.update(prompt_n=c["prompt_n"], prompt_hash=ids_hash(c["prompt"]), generated=c["generated"],
                 generated_n=len(c["generated"]), generated_hash=ids_hash(c["generated"]),
                 steps=c["steps"], steps_n=len(c["steps"]), capture_finish=c["finish"])
    return out


def compare(ref, asm, tolerance):
    """Request-for-request token identity and per-prompt decode speed. Returns (rows, speed, identical, speed_ok)."""
    rows, identical = [], True
    for a, b in zip(ref["runs"], asm["runs"]):
        same = (a["prompt_hash"] == b["prompt_hash"] and a["generated"] == b["generated"]
                and a["steps"] == b["steps"])
        first = next((i for i, (x, y) in enumerate(zip(a["generated"], b["generated"])) if x != y), None)
        if first is None and len(a["generated"]) != len(b["generated"]):
            first = min(len(a["generated"]), len(b["generated"]))
        identical = identical and same
        rows.append({"prompt": a["prompt"], "repeat": a["repeat"], "prompt_tokens": a["prompt_n"],
                     "generated_tokens": [a["generated_n"], b["generated_n"]],
                     "prompt_ids_equal": a["prompt_hash"] == b["prompt_hash"],
                     "generated_ids_equal": a["generated"] == b["generated"],
                     "mtp_steps_equal": a["steps"] == b["steps"], "first_divergence": first,
                     "tokens_agreeing": first if first is not None else min(a["generated_n"], b["generated_n"]),
                     "generated_hash": [a["generated_hash"], b["generated_hash"]]})
    speed, speed_ok = [], True
    for name in dict.fromkeys(r["prompt"] for r in ref["runs"]):
        def tps(side, key):
            return [r["timings"].get(key, 0.0) for r in side["runs"] if r["prompt"] == name]
        rd, ad = tps(ref, "predicted_per_second"), tps(asm, "predicted_per_second")
        rm, am = statistics.median(rd), statistics.median(ad)
        rel = (am - rm) / rm if rm else 0.0
        ok = abs(rel) <= tolerance or min(rd) <= am <= max(rd)
        speed_ok = speed_ok and ok
        rp, apf = tps(ref, "prompt_per_second"), tps(asm, "prompt_per_second")
        speed.append({"prompt": name, "decode_tps_reference": rd, "decode_tps_assembled": ad,
                      "decode_median": [round(rm, 2), round(am, 2)], "decode_rel_diff": round(rel, 4),
                      "decode_within_noise": ok,
                      "host_loadavg_1m": [ref.get("host_loadavg_before", [None])[0], asm.get("host_loadavg_before", [None])[0]],
                      "prefill_tps_first_run": [round(rp[0], 1), round(apf[0], 1)]})
    return rows, speed, identical, speed_ok


def self_consistency(side):
    """Do repeats of one request on one server give the same ids? Informational: a repeat resumes from the slot's
    live state, the first run prefills from scratch."""
    out = {}
    for r in side["runs"]:
        out.setdefault(r["prompt"], set()).add(r["generated_hash"])
    return {k: len(v) == 1 for k, v in out.items()}


# --- stages ---------------------------------------------------------------------------------------------------
def stage_check(ctx):
    """Passes when the recipe's origin and the recipe itself check clean. The whole-catalogue result is recorded
    next to it: other origins' modules are not part of this recipe and do not decide its gate."""
    log = os.path.join(ctx["res"], "check.txt")
    open(log, "w").close()
    origin = ctx["cfg"]["host"]["origin"]
    rc, _ = sh([sys.executable, "flerp", "check", "--origin", origin, "--warnings"], log)
    scoped = tail(log, 1)[0]
    full_log = os.path.join(ctx["res"], "check-catalogue.txt")
    open(full_log, "w").close()
    rc_all, _ = sh([sys.executable, "flerp", "check"], full_log)
    return {"pass": rc == 0, "rc": rc, "summary": scoped, "catalogue_rc": rc_all,
            "catalogue_summary": tail(full_log, 1)[0], "evidence": "check.txt, check-catalogue.txt"}


def stage_origin(ctx):
    log = os.path.join(ctx["res"], "diff-origin.txt")
    open(log, "w").close()
    js = os.path.join(ctx["res"], "diff-origin.json")
    checkout = ctx["cfg"]["host"]["reference_checkout"]
    origin = ctx["cfg"]["host"]["origin"]
    rc, _ = sh([sys.executable, "flerp", "diff-origin", "--origin", origin, checkout, "--json", js], log)
    head = subprocess.run(["git", "-C", checkout, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", checkout, "status", "--porcelain"], capture_output=True, text=True).stdout
    want = (ctx["cat"].origin_data(origin) or {}).get("commit", "")
    return {"pass": rc == 0 and head == want and not dirty.strip(), "rc": rc, "checkout_head": head,
            "origin_toml_commit": want, "checkout_dirty": bool(dirty.strip()), "summary": tail(log, 1)[0],
            "evidence": "diff-origin.txt"}


def stage_build(ctx):
    log = os.path.join(ctx["res"], "build.txt")
    open(log, "w").close()
    tree = os.path.join(ROOT, "build", ctx["recipe"].name)
    out = {"tree": os.path.relpath(tree, ROOT), "evidence": "build.txt"}
    rc, _ = sh([sys.executable, "flerp", "assemble", ctx["recipe"].name], log)
    if rc == 0:
        with open(os.path.join(tree, "CMakeLists.txt")) as f:
            text = f.read()
        ref = ctx["cfg"]["host"]["reference_checkout"]
        out["references_origin_checkout"] = ref in text or "strixflerp-upstream" in text
        rc, out["configure_s"] = sh(["cmake", "-G", "Ninja", "-S", tree, "-B", ctx["ninja"]], log, timeout=600)
    if rc == 0:
        rc, out["build_s"] = sh(["ninja", "-C", ctx["ninja"], "-j", str(ctx["cfg"]["host"].get("build_jobs", 16))],
                                log, timeout=3600)
    bins = {}
    for name in ctx["recipe"].binaries:
        p = os.path.join(ctx["ninja"], name)
        bins[name] = {"sha256": sha256(p), "bytes": os.path.getsize(p)} if os.path.isfile(p) else None
    out["binaries"] = bins
    shutil.copyfile(os.path.join(tree, "assembly.json"), os.path.join(ctx["res"], "assembly.json"))
    out["pass"] = rc == 0 and all(bins.values()) and not out.get("references_origin_checkout", True)
    out["rc"] = rc
    return out


def stage_unit(ctx):
    h = ctx["cfg"]["host"]
    scratch = os.path.join(ctx["scratch"], "unit")
    shutil.rmtree(scratch, ignore_errors=True)
    os.makedirs(scratch)
    subs = {"tokenizer": h["tokenizer"], "preprocessor_config": h["preprocessor_config"],
            "model_config": h["model_config"], "scratch": scratch, "weights": h["weights"], "ngram": h["ngram"]}
    tests, ok = [], True
    for t in ctx["cfg"]["unit_test"]:
        if t.get("gpu"):
            continue
        log = os.path.join(ctx["res"], f"unit-{t['binary']}.txt")
        open(log, "w").close()
        d = os.path.join(scratch, t["binary"])
        os.makedirs(d)
        cmd = [os.path.join(ctx["ninja"], t["binary"])] + [a.format(**dict(subs, scratch=d)) for a in t["args"]]
        rc, secs = sh(cmd, log, timeout=1800)
        ok = ok and rc == 0
        tests.append({"binary": t["binary"], "rc": rc, "seconds": secs, "last_line": tail(log, 1)[0],
                      "evidence": os.path.basename(log)})
    return {"pass": ok and bool(tests), "tests": tests}


def stage_gpu(ctx):
    cfg, h = ctx["cfg"], ctx["cfg"]["host"]
    scratch = os.path.join(ctx["scratch"], "gpu")
    os.makedirs(scratch, exist_ok=True)
    lease_log = os.path.join(ctx["res"], "lease.txt")
    open(lease_log, "w").close()
    out = {"evidence": "identity.json"}
    ok, state = gpu_free(cfg)
    waiters = pod_waiters()
    out["gpu_before"] = dict(state, pod_gpu_waiters=len(waiters))
    if not ok or waiters:
        out.update({"pass": False, "error": "the GPU is not free" + (f" ({len(waiters)} pod-gpu processes wait for it)"
                    if waiters else "") + "; nothing was loaded"})
        return out
    lease = Hold(cfg, lease_log, out, cfg["host"].get("yield_wait_s", 1800))
    why = f"strixflerp recipe gate {ctx['recipe'].name}"
    asm_bin = os.path.join(ctx["ninja"], next(iter(ctx["recipe"].binaries)))
    ref_bin = h["reference_server"]
    out["binaries_sha256"] = {"reference": {"path": ref_bin, "sha256": sha256(ref_bin)},
                              "assembled": {"path": os.path.relpath(asm_bin, ROOT), "sha256": sha256(asm_bin)}}
    reqs = requests(cfg)
    detail = {"requests": [{"name": n, "path": p, "body_sha256": hashlib.sha256(
        json.dumps(b, sort_keys=True).encode()).hexdigest()[:16], "max_tokens": b["max_tokens"],
        "temperature": b["temperature"], "seed": b["seed"]} for n, p, b in reqs], "servers": [], "comparisons": []}
    try:
        out["lease"] = lease.take(f"{why}: the origin's GPU test", "the origin's GPU test")
        # the origin's GPU test, from the assembled tree
        tests, tests_ok = [], True
        for t in cfg["unit_test"]:
            if not t.get("gpu"):
                continue
            ok, state = gpu_free(cfg)
            if not ok:
                raise RuntimeError(f"GPU not free before {t['binary']}: {state}")
            log = os.path.join(ctx["res"], f"unit-{t['binary']}.txt")
            open(log, "w").close()
            subs = {"tokenizer": h["tokenizer"], "weights": h["weights"], "ngram": h["ngram"]}
            rc, secs = sh([os.path.join(ctx["ninja"], t["binary"])] + [a.format(**subs) for a in t["args"]], log,
                          timeout=1800)
            tests_ok = tests_ok and rc == 0
            tests.append({"binary": t["binary"], "rc": rc, "seconds": secs, "last_lines": tail(log, 4),
                          "evidence": os.path.basename(log)})
            time.sleep(cfg["run"]["settle_s"])
        out["gpu_tests"] = tests
        variant = cfg.get("variant", {})
        criterion = ctx["recipe"].gate.get("criterion", "token-identity")
        with open(os.path.join(ctx["res"], "assembly.json")) as f:
            manifest = json.load(f)
        adapters = sorted(r["adapter"] for r in manifest.get("renames", {}).values())
        swaps_ran, tallies = True, []
        identical = speed_ok = True
        for mtp in cfg["run"]["mtp"]:
            tag = "mtp-on" if mtp else "mtp-off"
            # one load a hold when a pod job waits for the GPU; otherwise the lease stays through the next load
            lease.between(f"{why}: reference server, {tag}", f"reference server, {tag}")
            ref = run_server(f"reference-{tag}", ref_bin, mtp, cfg, reqs, scratch)
            lease.between(f"{why}: assembled server, {tag}", f"assembled server, {tag}")
            asm = run_server(f"assembled-{tag}", asm_bin, mtp, cfg, reqs, scratch)
            rows, speed, same, fast = compare(ref, asm, cfg["run"]["tps_tolerance"])
            identical, speed_ok = identical and same, speed_ok and fast
            for ad in adapters:      # a swap counts only if the foreign kernel really ran in this server
                t = asm["adapter_tallies"].get(ad)
                tallies.append({"mtp": mtp, "adapter": ad, **(t or {"foreign_launches": 0, "line": "no tally line"})})
                swaps_ran = swaps_ran and bool(t) and t["foreign_launches"] > 0
            detail["servers"] += [ref, asm]
            detail["comparisons"].append({
                "mtp": mtp, "token_identity": same, "speed_within_noise": fast, "requests": rows, "speed": speed,
                "reference_repeats_self_consistent": self_consistency(ref),
                "assembled_repeats_self_consistent": self_consistency(asm)})
        speed_counts = variant.get("speed_is_criterion", True)
        if criterion == "token-agreement":      # the module gate carries the numeric criterion; tokens are measured
            gates = variant.get("module_gates") or [{"evidence": variant.get("module_gate", ""),
                                                      "providers": variant.get("module_gate_providers")}]
            checked = []
            for g in gates:      # every module gate the variant rests on, for the providers it uses
                mg_path = os.path.join(ROOT, g.get("evidence", ""))
                mg_pass, wanted = False, g.get("providers")
                if g.get("evidence") and os.path.isfile(mg_path):
                    with open(mg_path) as f:
                        mg_doc = json.load(f)
                    wanted = wanted or list(mg_doc.get("providers", {}))
                    mg_pass = all(mg_doc.get("providers", {}).get(w, {}).get("pass") for w in wanted) and bool(wanted)
                checked.append({"evidence": g.get("evidence"), "providers": wanted, "pass": mg_pass})
            tokens_ok = bool(checked) and all(c["pass"] for c in checked)
            out["module_gate"] = {"evidence": ", ".join(f"{c['evidence']} ({', '.join(c['providers'] or [])})" for c in checked),
                                  "pass": tokens_ok, "gates": checked}
        else:
            tokens_ok = identical
        agreement = [{"mtp": c["mtp"], "prompt": r["prompt"], "repeat": r["repeat"],
                      "generated_tokens": r["generated_tokens"], "first_divergence": r["first_divergence"],
                      "tokens_agreeing": r["tokens_agreeing"]}
                     for c in detail["comparisons"] for r in c["requests"]]
        out.update({"token_identity": identical, "speed_within_noise": speed_ok, "criterion": criterion,
                    "speed_is_criterion": speed_counts, "adapters": adapters, "adapter_tallies": tallies,
                    "swaps_ran": swaps_ran, "token_agreement": agreement,
                    "pass": tokens_ok and (speed_ok or not speed_counts) and tests_ok and swaps_ran,
                    "comparisons": [{k: c[k] for k in ("mtp", "token_identity", "speed_within_noise", "speed")}
                                    for c in detail["comparisons"]],
                    "requests_compared": sum(len(c["requests"]) for c in detail["comparisons"])})
    except Exception as e:  # report what happened; the finally below still frees the GPU and the lease
        out.update({"pass": False, "error": f"{type(e).__name__}: {e}"})
    finally:
        lease.give()
        out["held_seconds"] = round(sum(x["seconds"] for x in out["holds"]), 1)
        time.sleep(cfg["run"]["settle_s"])
        out["gpu_after"] = gpu_free(cfg)[1]
        with open(os.path.join(ctx["res"], "identity.json"), "w") as f:
            json.dump(detail, f, indent=1)
            f.write("\n")
    return out


def first_difference(a, b):
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    return min(len(a), len(b)) if first is None and len(a) != len(b) else first


def mode_sensitivity(res):
    """Context for a token-agreement result, from the evidence in identity.json: does ONE server give the same
    greedy tokens with MTP drafting on and off? Drafting changes only the batch shapes its kernels run at. Where
    the reference binary itself differs between the two modes, the model's greedy text at that prompt turns on
    last-bit differences, and a kernel with another summation order will move it too. Returns
    [{server, prompt, first_difference}] for the first repeat of each prompt, or [] when a mode was not run."""
    path = os.path.join(res, "identity.json")
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        servers = {s["label"]: s for s in json.load(f).get("servers", [])}
    out = []
    for side in ("reference", "assembled"):
        on, off = servers.get(f"{side}-mtp-on"), servers.get(f"{side}-mtp-off")
        if not on or not off:
            continue
        for r in on["runs"]:
            if r["repeat"] != 0 or "generated" not in r:
                continue
            twin = next((q for q in off["runs"] if q["prompt"] == r["prompt"] and q["repeat"] == 0 and "generated" in q), None)
            if twin:
                out.append({"server": side, "prompt": r["prompt"], "generated_tokens": [r["generated_n"], twin["generated_n"]],
                            "first_difference": first_difference(r["generated"], twin["generated"])})
    return out


def write_summary(ctx, results, recorded=None):
    """Writes summary.json and summary.md. With `recorded` (a summary.json read back), only summary.md is
    rewritten from it: the evidence keeps its host and time."""
    res = ctx["res"]
    overall = all(results.get(s, {}).get("pass") for s in STAGES)
    doc = recorded or {"recipe": ctx["recipe"].name, "host": os.uname().nodename,
                       "when_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "stages_run": list(results), "pass": overall if set(results) == set(STAGES) else None,
                       "stages": results}
    if recorded is None:
        with open(os.path.join(res, "summary.json"), "w") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")
    lines = [f"# Recipe gate: {ctx['recipe'].name}", "",
             f"Host {doc['host']}, {doc['when_utc']}. Stages run: {', '.join(results)}.", ""]
    stamped = any(r.get("when_utc") for r in results.values())      # a partial run: the stages without a time are older
    for s in STAGES:
        if s in results:
            r = results[s]
            lines.append(f"- **{s}**: {'PASS' if r.get('pass') else 'FAIL'}"
                         + (f" ({r['when_utc']})" if r.get("when_utc") else
                            " (carried over from an earlier run of this gate)" if stamped else "")
                         + (f" - {r['summary']}" if r.get("summary") else "")
                         + (f" - {r['error']}" if r.get("error") else ""))
    if results.get("origin", {}).get("origin_toml_commit"):
        lines += ["", f"Origin commit (ORIGIN.toml, reference checkout HEAD): {results['origin']['origin_toml_commit'][:7]}."]
    g = results.get("gpu", {})
    for k, b in g.get("binaries_sha256", {}).items():
        lines.append(f"- {k} server `{b['path']}` sha256 `{b['sha256']}`")
    if g.get("comparisons"):
        lines += ["", "| MTP | prompt | decode t/s reference (median) | decode t/s assembled (median) | diff | within noise "
                      "| host load, 1 min (reference / assembled) |", "|---|---|---|---|---|---|---|"]
        for c in g["comparisons"]:
            for s in c["speed"]:
                lines.append(f"| {'on' if c['mtp'] else 'off'} | {s['prompt']} | {s['decode_median'][0]} | "
                             f"{s['decode_median'][1]} | {100 * s['decode_rel_diff']:+.1f}% | "
                             f"{'yes' if s['decode_within_noise'] else 'NO'} | "
                             f"{' / '.join(str(x) for x in s.get('host_loadavg_1m', ['?', '?']))} |")
        lines += ["", f"Token identity over {g.get('requests_compared')} requests: "
                      f"{'identical' if g.get('token_identity') else 'DIVERGED'} (prompt ids, generated ids, MTP steps)."]
        if g.get("adapters"):
            lines += ["", f"Criterion: {g.get('criterion')}. Decode speed is "
                          f"{'a pass criterion' if g.get('speed_is_criterion') else 'recorded, not a pass criterion'}.", "",
                      "| MTP | adapter | foreign provider | foreign launches | launches left to the origin's kernel | tally line |",
                      "|---|---|---|---|---|---|"]
            for t in g.get("adapter_tallies", []):
                lines.append(f"| {'on' if t['mtp'] else 'off'} | `{t['adapter']}` | `{t.get('provider', '-')}` | "
                             f"{t.get('foreign_launches', 0)} | {t.get('origin_launches', '-')} | {t.get('line')} |")
            if not g.get("swaps_ran"):
                lines += ["", "**An adapter never launched its foreign provider: the swap did not run.**"]
        if g.get("criterion") == "token-agreement" or not g.get("token_identity"):
            mgate = g.get("module_gate", {})
            if mgate:
                lines += ["", f"Module gate `{mgate.get('evidence')}`: {'PASS' if mgate.get('pass') else 'FAIL or missing'}."]
            ta = g.get("token_agreement", [])
            div = [r for r in ta if r["first_divergence"] is not None]
            if div:
                lines += ["", f"{len(div)} of {len(ta)} requests leave the reference's tokens, the earliest at generated "
                              f"token {min(r['first_divergence'] for r in div)}. From that token on the two servers "
                              "generate different text, so their decode speeds for those requests are not a like-for-like "
                              "comparison (MTP acceptance depends on the text)."]
            else:
                lines += ["", f"No request left the reference's tokens in this sample ({len(ta)} requests); that is a "
                              "measurement over these prompts, not a guarantee."]
            sens = mode_sensitivity(res)
            if sens:
                lines += ["", "For scale: one server against itself, MTP drafting on against off (the same kernels at other "
                              "batch shapes), first repeat of each prompt. A position here means that prompt's greedy text "
                              "already turns on last-bit differences inside one binary.", "",
                          "| server | prompt | first position where MTP on and MTP off differ |", "|---|---|---|"]
                for x in sens:
                    fd = "none" if x["first_difference"] is None else x["first_difference"]
                    lines.append(f"| {'reference strixite' if x['server'] == 'reference' else 'this variant'} | {x['prompt']} | {fd} |")
            lines += ["", "Token agreement with the reference, request by request (first position where the generated "
                          "ids differ; `-` = none within the request):", "",
                      "| MTP | prompt | repeat | generated tokens (reference / variant) | first divergence | tokens agreeing |",
                      "|---|---|---|---|---|---|"]
            for r in g.get("token_agreement", []):
                fd = "-" if r["first_divergence"] is None else r["first_divergence"]
                lines.append(f"| {'on' if r['mtp'] else 'off'} | {r['prompt']} | {r['repeat']} | "
                             f"{r['generated_tokens'][0]} / {r['generated_tokens'][1]} | {fd} | {r['tokens_agreeing']} |")
    with open(os.path.join(res, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return overall


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--keep-going", action="store_true", help="run later stages after a failed one")
    ap.add_argument("--summarize", action="store_true", help="rewrite summary.md from the recorded summary.json; runs nothing")
    a = ap.parse_args()
    cat = Catalog(ROOT)
    recipe = cat.recipe(a.recipe)
    if recipe is None:
        sys.exit(f"no recipe {a.recipe}")
    cfg = load_config(os.path.join(ROOT, recipe.gate["config"]))
    res = os.path.join(ROOT, "gate", "results", recipe.name)
    os.makedirs(res, exist_ok=True)
    ctx = {"cat": cat, "recipe": recipe, "cfg": cfg, "res": res,
           "scratch": os.path.join(ROOT, cfg["host"].get("scratch", "build/gate-scratch")),
           "ninja": os.path.join(ROOT, "build", recipe.name, "ninja")}
    os.makedirs(ctx["scratch"], exist_ok=True)
    results = {}
    prior = os.path.join(res, "summary.json")
    if a.summarize:
        with open(prior) as f:
            doc = json.load(f)
        return 0 if write_summary(ctx, doc["stages"], recorded=doc) else 1
    if os.path.isfile(prior) and set(a.stages.split(",")) != set(STAGES):
        with open(prior) as f:
            results = json.load(f).get("stages", {})
    fns = {"check": stage_check, "origin": stage_origin, "build": stage_build, "unit": stage_unit, "gpu": stage_gpu}
    for s in STAGES:
        if s not in a.stages.split(","):
            continue
        print(f"== {s}", flush=True)
        results[s] = fns[s](ctx)
        results[s]["when_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())      # a partial run keeps the other stages' results: each says when it ran
        print(f"   {'PASS' if results[s].get('pass') else 'FAIL'}", flush=True)
        if not results[s].get("pass") and not a.keep_going:
            break
    ok = write_summary(ctx, {s: results[s] for s in STAGES if s in results})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
