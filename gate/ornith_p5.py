#!/usr/bin/env python3
"""Stage P5 of docs/PHASE5-FOURNODE-DESIGN.md: first tokens from Ornith-1.5 397B as four layer stages on four workers,
the bytes streamed by loom-swarm from control-node's rail origin straight into each rank's GPU allocation. Run on
control-node:

    python3 gate/ornith_p5.py --plan                 # read-only: nodes, memory, rail routes, origin, serving, stage R
    python3 gate/ornith_p5.py regate35               # control-node's GPU (through tools/gpu-queue.sh): the 35B with
                                                     #   the new code, loopback world 1/2/4 on one load + localhost
                                                     #   TCP world 2/4 loading from loom-swarm: P1's hashes
    python3 gate/ornith_p5.py stage [--dry-run] [--nodes F,D]   (before the window, no hold: each node's ranks into /cache)
    python3 gate/ornith_p5.py run [--card ID] [--source swarm|file|cache] [--steps w4,w3,serve]
    python3 gate/ornith_p5.py cleanup [--run-id ID]  # stop every flerp-p5-* unit on the four nodes, verify none left

ORNITH-O3 step one (gate/ornith-o3-mtp.toml [ornith]), prepared, needs owner's go: `stage --with-mtp` also stages the
MTP block (blk.60) and token_embd into the node running world 4's last rank; `run --steps w4 --mtp-probe 3` gives that
rank --mtp-probe 3, the ids files and --mtp-data-file (the staged file), runs the 8 probe prompts of
gate/results/ornith-o3-mtp/prompts.json for [probe] max_tokens, and writes its evidence to gate/results/ornith-o3-mtp/ornith/
(never over P5's). The ids are produced exactly as without the flag.

`run` takes the holds itself (owner [host] hold_owner, drained as pod-quad asks, a refused hold drained and retried,
never FORCE), refuses to start a process on a node whose serving is still up, starts every process as a transient
user unit named [host] unit_prefix + run id + tag, and on any exit (normal, error, SIGTERM, SIGINT) stops all of
them and checks none is left before it releases the holds. With --card it comments on that pod card every
[host] card_comment_s seconds. Criteria: [criteria] of gate/ornith15-397b.flerp-p5.toml, committed before any
worker run. Evidence: gate/results/ornith15-397b.flerp-p5/.
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
sys.path.insert(0, ROOT)
import flerplib.configs as CF  # noqa: E402
import gguf_census as C  # noqa: E402
import moe_placed as MP  # noqa: E402
import moe_stage as MS  # noqa: E402

CONFIG = os.path.join(ROOT, "gate", "ornith15-397b.flerp-p5.toml")
RES = os.path.join(ROOT, "gate", "results", "ornith15-397b.flerp-p5")
GEN_BUILD = os.path.join(ROOT, "build", "flerp-p5")   # the generator's build (--gen-build)
# ORNITH-O3: set by `run --mtp-probe D` (cmd_run): {"depth", "names", "ids", "max_tokens"}; None runs P5 as written
PROBE = None
# ORNITH-O3 step two (gate/ornith-o3-spec.toml [ornith]): set by `run --spec-launch SPECS` (run_confirm) while those
# launches run: {"names", "ids", "max_tokens", "context", "costs"}; world 4's last rank drafts (--draft-head), rank 0
# runs SPECS (graph:reply:subchunks:drafts), evidence in gate/results/ornith-o3-spec/ornith/. None runs P5 as written.
SPEC = None
BEFORE_RELEASE = ""   # run --before-release CMD: run on control-node after the units stop, before the holds are released
SPEC_RES = os.path.join(ROOT, "gate", "results", "ornith-o3-spec", "ornith")
SPEC_CONTEXT = 4096
PROBE_PROMPTS = os.path.join(ROOT, "gate", "results", "ornith-o3-mtp", "prompts.json")
PROBE_RES = os.path.join(ROOT, "gate", "results", "ornith-o3-mtp", "ornith")
SCRATCH = os.path.join(ROOT, "build", "gate-scratch-p5")
# Several launches under one hold (`run --launch ROWS=SPECS ...`): LAUNCH is the launch being run ({"rows", "specs"}),
# None runs P5 as written. --rows/--reply pass straight through when given without --launch.
LAUNCH = None
PASS = {"rows": None, "reply": None, "run_specs": None}
# --swarm-ranks R,.. --swarm-dir DIR: those world-4 ranks stream their bytes from an origin this runner starts on
# control-node's rail address, serving DIR (control-node NVMe); every other rank reads --source. Cold storage (the USB
# disks /mnt/nv*, the ark view /data/ark) is never a byte source: cold_path() refuses it everywhere.
SWARM_RANKS = set()
# --rank-origin R=LISTEN@HOST:FILE:RANGES: rank R streams with the FULL manifest from LISTEN, an origin this runner
# starts on HOST (`control` = control-node itself, else a worker's mgmt address, reached from control-node directly)
# as `loom-swarm.place serve --ranges` of a PLACE-staged file. {rank: {"listen", "host", "file", "ranges"}}
RANK_ORIGINS = {}
SWARM_BIN = "/data/models/.swarm/bin/loom-swarm.place"
COLD_PREFIXES = ("/mnt/nv", "/data/ark")
CONFIRM_RES = os.path.join(ROOT, "gate", "results", "ornith-p5-confirm")
P5_REF_RUN = os.path.join(ROOT, "gate", "results", "ornith15-397b.flerp-p5", "run.json")   # run 11121855
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
XDG = "export XDG_RUNTIME_DIR=/run/user/$(id -u); "
GIB = float(1 << 30)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Log:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path

    def __call__(self, text):
        line = f"{utc()} {text}"
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def save(name, doc):
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, name), "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)


def on(ip, cmd, timeout=120):
    try:
        p = subprocess.run(SSH + [f"pod@{ip}", cmd], capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return 124, "ssh timed out"


# --- judgements (no I/O) ------------------------------------------------------------------------------------------
def cold_path(path, real=None):
    """True when path (or its resolved target `real`) lies on cold storage (/mnt/nv*, /data/ark)."""
    return any(str(p).startswith(COLD_PREFIXES) for p in (path, real if real is not None else path))


def parse_launch(text):
    """'pinned=on:full:1,on:token:1' -> {"rows": "pinned", "specs": "on:full:1,on:token:1"}."""
    rows, _, specs = text.partition("=")
    if rows not in ("pinned", "pageable") or not specs:
        raise ValueError(f"--launch wants ROWS=SPECS (ROWS pinned|pageable), got {text!r}")
    for item in specs.split(","):
        g, r, s = item.split(":")
        if g not in ("on", "off") or r not in ("full", "token") or int(s) < 1:
            raise ValueError(f"--launch: bad run spec {item!r}")
    return {"rows": rows, "specs": specs}


def parse_rank_origin(text):
    """'1=10.20.0.51:7171@10.10.0.51:/cache/x/f.staged:/data/models/.place/ranges/r.ranges' -> (1, {...})."""
    r, _, rest = text.partition("=")
    listen, _, where = rest.partition("@")
    parts = where.split(":")
    if not r.isdigit() or ":" not in listen or len(parts) != 3 or not all(parts):
        raise ValueError(f"--rank-origin wants R=LISTEN@HOST:FILE:RANGES, got {text!r}")
    host, f, rng = parts
    if cold_path(f) or cold_path(rng):
        raise ValueError(f"--rank-origin {text!r}: cold storage is never a byte source")
    return int(r), {"listen": listen, "host": host, "file": f, "ranges": rng}


def pieces_in_ranges(manifest, ranges):
    """Pieces of a loom-swarm manifest that overlap any (offset, length) range: what a --ranges serve must hold."""
    ps = manifest["piece_size"]
    out = set()
    for o, n in ranges:
        out.update(range(o // ps, (o + n - 1) // ps + 1))
    return len(out)


def p5_refs(run_doc, names):
    """{prompt name: {"ids", "logits_sha256"}} of run 11121855's world-4 run 1 for the prompts it ran."""
    w4 = run_doc["w4"]
    out = {}
    for e, sha in zip(w4["runs"][0], w4["sha"][0]):
        p = run_doc.get("prompts", ["bridge", "code", "rivers"])[e["prompt"]]
        if p in names:
            out[p] = {"ids": e["ids"], "logits_sha256": sha}
    return out


def proc_pattern(bundle):
    """pgrep -f regex for the bundle's processes that does not match its own command line ('.../bundl[e]/')."""
    b = re.sub(r"([.^$*+?(){}\[\]|\\])", r"\\\1", bundle.rstrip("/"))   # POSIX ERE escapes only (pgrep is not Python re)
    return b[:-1] + "[" + b[-1] + "]/"


def parse_left(rc, text):
    """'units N' and 'procs N' lines -> {"units": N, "procs": N}; None when the probe failed (unknown is not gone)."""
    got = dict(re.findall(r"^(units|procs) (\d+)$", text, re.M))
    if rc != 0 or set(got) != {"units", "procs"}:
        return None
    return {k: int(v) for k, v in got.items()}


def conf_cache_paths(text, home="/home/pod"):
    """/cache/... directories a serving conf names (KEY=VALUE lines, comments and quotes stripped, $HOME expanded)."""
    out = []
    for line in text.replace("$HOME", home).splitlines():
        line = line.split("#")[0]
        if "=" not in line:
            continue
        for tok in re.split(r"[\s:,]+", line.split("=", 1)[1].strip().strip("'\"")):
            tok = tok.strip("'\"").rstrip("/")
            if tok.startswith("/cache/") and tok not in out:
                out.append(tok)
    return out


def serving_shortfall(probe, need):
    """Bytes the node's /cache still lacks for its serving set: need minus what the conf-named dirs hold, minus free
    space (those dirs may be re-staged into the free space). 0 = the serving set fits."""
    have = sum(probe["dirs"].values())
    return max(0, need - have - probe["avail"])


def serving_lost(pre, post):
    """Conf-named dirs that were there before the window and are gone or smaller after it."""
    return sorted(d for d, b in (pre or {}).get("dirs", {}).items() if b and post["dirs"].get(d, 0) < b)


def slices_to_evict(pre, post, need):
    """True when the window took someone's serving data or left the node unable to hold its serving set."""
    return bool(serving_lost(pre, post)) or serving_shortfall(post, need) > 0


def units_gone(left):
    return bool(left) and all(v is not None and v["units"] == 0 for v in left.values())


def help_flags(text):
    """The options a binary's --help lists (one a line, or anywhere in usage text)."""
    return set(re.findall(r"(?<![\w-])(--[a-z0-9][a-z0-9-]*)", text))


def argv_flags(argv):
    return {x for x in argv if isinstance(x, str) and x.startswith("--")}


def split_runs(events, n_prompts):
    """rank 0's `staged` events in order -> [[run 1 per prompt], [run 2 per prompt], ...]."""
    st = [e for e in events if e.get("event") == "staged"]
    return [st[i:i + n_prompts] for i in range(0, len(st), n_prompts)]


def judge_equal_runs(a, b, label):
    """Per prompt, ids and every logits row's sha256 equal between two runs (lists of `staged` events)."""
    why = []
    if len(a) != len(b):
        return [f"{label}: {len(a)} prompts against {len(b)}"]
    for x, y in zip(a, b):
        p = x["prompt"]
        if x["ids"] != y["ids"]:
            why.append(f"{label}: prompt {p}: ids differ at {MS.first_diff(x['ids'], y['ids'])}")
        d = MS.first_diff(x["logits_sha256"], y["logits_sha256"])
        if d is not None:
            why.append(f"{label}: prompt {p}: logits row {d} differs (sha256)")
        for e in (x, y):
            if not e.get("argmax_is_token"):
                why.append(f"{label}: prompt {p}: a row's argmax is not the token")
    return why


def speed(events):
    """Decode tok/s and TTFT per prompt from rank 0's `staged` events (wall time at rank 0, relay included)."""
    out = []
    for e in events:
        if e.get("event") != "staged":
            continue
        c = e.get("calls") or {}
        n, ms = c.get("decode_calls") or 0, c.get("decode_wall_ms") or 0
        out.append({"prompt": e["prompt"], "prompt_tokens": e["prompt_tokens"], "ttft_s": e.get("prefill_s"),
                    "decode_tok_s": round(n / (ms / 1000.0), 3) if ms else None, "decode_steps": n})
    return out


def have_reply(data, n):
    """Parse a loom-swarm HAVE reply body (after the status byte): u64 n, then ceil(n/64) words. -> pieces held."""
    got = struct.unpack_from("<Q", data, 0)[0]
    if got != n:
        raise ValueError(f"origin reports {got} pieces, manifest {n}")
    words = struct.unpack_from(f"<{(n + 63) // 64}Q", data, 8)
    return sum(bin(w).count("1") for w in words)


def window_estimate(plan4, plan3, gbs):
    """Minutes per step of the window at origin rate gbs (GB/s) - loads dominate."""
    w4 = sum(r["tensor_bytes"] for r in plan4) / 1e9 / gbs / 60
    w3 = sum(r["tensor_bytes"] for r in plan3) / 1e9 / gbs / 60
    return {"holds_and_drain": "2-15 (DRAIN up to [host] hold_drain_s a try)", "bundle_copy": 2,
            "w4_load": round(w4, 1), "w4_two_runs_and_dumps": 6, "block_check_cpu_pod_control": 10,
            "w3_load": round(w3, 1), "w3_run": 4, "serve_load": round(w4, 1), "serve_chat_and_shutdown": 4,
            "cleanup_and_release": 3, "total": round(2 * w4 + w3 + 35, 0)}


# --- config, bundle, nodes ----------------------------------------------------------------------------------------
def load_cfg():
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


def node(cfg, name):
    return next(n for n in cfg["host"]["nodes"] if n["name"] == name)


def nodes_of(cfg, world):
    names = [n["name"] for n in cfg["host"]["nodes"]] if world == 4 else cfg["host"]["world3_nodes"]
    return [node(cfg, x) for x in names]


def local_bundle(cfg):
    """build/flerp-p5-bundle: the two recipes' binaries, the ROCm libraries P4's bundle carried, the stream helper,
    the frame writer and the manifest."""
    b = os.path.join(ROOT, "build", "flerp-p5-bundle")
    os.makedirs(b, exist_ok=True)
    gen = os.path.join(GEN_BUILD, "out", "flerp-qwen35moe-generate")
    sv = os.path.join(ROOT, "build", "flerp-p5-serve", "out")
    files = {gen: "flerp-qwen35moe-generate", os.path.join(sv, "flerp-qwen35moe-stage"): "flerp-qwen35moe-stage",
             os.path.join(sv, "strixflerp-qwen36-35b-a3b.flerp-serve-stage"): "flerp-serve",
             os.path.join(ROOT, cfg["swarm"]["helper"]): "loom-swarm-stream",
             os.path.join(ROOT, "gate", "swarm_frames_to_file.py"): "swarm_frames_to_file.py",
             cfg["swarm"]["manifest"]: "ornith.swarm.json"}
    mdir, base = os.path.dirname(cfg["swarm"]["manifest"]), os.path.basename(cfg["model"]["gguf"])
    for r in range(4):
        lay = os.path.join(mdir, f"{base}.r{r}of4.swarm.json")
        if os.path.isfile(lay):
            files[lay] = f"ornith.r{r}of4.swarm.json"
    missing = [s for s in files if not os.path.isfile(s)]
    if missing:
        return None, missing
    for s, d in files.items():
        subprocess.run(["cp", "-L", s, os.path.join(b, d)], check=True)
    if not os.path.isdir(os.path.join(b, "lib")):
        subprocess.run(["cp", "-a", os.path.join(ROOT, "build", "flerp-p4", "bundle", "lib"), os.path.join(b, "lib")], check=True)
    subprocess.run(f"cd {shlex.quote(b)} && find . -type f ! -name bundle.md5 -print0 | sort -z | xargs -0 md5sum > bundle.md5", shell=True, check=True)
    return b, []


def push_bundle(cfg, ip, local):
    rb = cfg["host"]["bundle"]
    subprocess.run(["rsync", "-a", "--delete", "-e", " ".join(SSH), local + "/", f"pod@{ip}:{rb}/"], check=True, timeout=1800)
    rc, text = on(ip, f"cd {rb} && md5sum -c --quiet bundle.md5 && mkdir -p {cfg['host']['work']} && echo BUNDLE-OK", 600)
    if "BUNDLE-OK" not in text:
        raise RuntimeError(f"bundle on {ip} does not match: {text[-300:]}")


# --- per-node facts (read-only) -----------------------------------------------------------------------------------
NODE_FACTS = r"""
echo "mem $(awk '/MemAvailable/ {print $2*1024}' /proc/meminfo) $(awk '/MemTotal/ {print $2*1024}' /proc/meminfo)"
for d in /sys/class/drm/card*/device; do [ -f $d/mem_info_gtt_total ] && echo "gtt $(cat $d/mem_info_gtt_total) $(cat $d/mem_info_gtt_used) $(cat $d/mem_info_vram_total) $(cat $d/mem_info_vram_used)"; done
echo "cache $(df -B1 --output=avail /cache 2>/dev/null | tail -1) $(mountpoint -q /cache && echo mounted || echo absent)"
echo "ramdisk $(mountpoint -q /srv/swarm && df -B1 --output=size,avail /srv/swarm | tail -1 || echo absent)"
echo "route $(ip -o route get 10.20.0.31 2>/dev/null | head -1)"
echo "home $(df -B1 --output=avail $HOME | tail -1)"
echo "serving $(pgrep -af "SERVING_PATTERN" | grep -v pgrep | cut -c1-120 | tr '\n' ';')"
echo "units $(XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user list-units --no-legend 'flerp-p5-*' 2>/dev/null | wc -l)"
echo "load $(cut -d' ' -f1 /proc/loadavg)"
"""


def node_facts(cfg, n):
    rc, text = on(n["mgmt"], NODE_FACTS.replace("SERVING_PATTERN", cfg["host"]["serving_pattern"]), 60)
    f = {"reachable": rc == 0, "raw_rc": rc}
    for line in text.splitlines():
        k, _, v = line.partition(" ")
        parts = v.split()
        try:
            if k == "mem":
                f["mem_available"], f["mem_total"] = int(parts[0]), int(parts[1])
            elif k == "gtt":
                f["gtt_total"], f["gtt_used"], f["vram_total"], f["vram_used"] = map(int, parts[:4])
            elif k == "cache":
                f["cache_avail"] = int(parts[0]) if parts and parts[0].isdigit() else None
                f["cache"] = parts[-1] if parts else "absent"
            elif k == "ramdisk":
                f["ramdisk"] = "absent" if parts == ["absent"] else {"size": int(parts[0]), "avail": int(parts[1])}
            elif k == "route":
                f["route"] = v.strip()
                f["route_src"] = parts[parts.index("src") + 1] if "src" in parts else None
                f["route_dev"] = parts[parts.index("dev") + 1] if "dev" in parts else None
            elif k == "home":
                f["home_avail"] = int(parts[0])
            elif k == "serving":
                f["serving"] = [x for x in v.strip().split(";") if x]
            elif k == "units":
                f["p5_units"] = int(parts[0])
            elif k == "load":
                f["load1"] = float(parts[0])
        except (ValueError, IndexError):
            f.setdefault("parse_errors", []).append(line)
    f["serving_up"] = bool(f.get("serving")) or f.get("gtt_used", 0) > cfg["host"]["gtt_idle_max"]
    return f


def held(ip):
    p = subprocess.run(["pod-quad", "held", ip], capture_output=True, text=True, timeout=60)
    return (p.stdout + p.stderr).strip()


def origin_have(cfg, path=None, addr=None):
    """HAVE for a manifest's root at the origin (addr, default [swarm] origin): (pieces held, pieces in the manifest)
    or an error text."""
    try:
        with open(path or cfg["swarm"]["manifest"]) as f:
            m = json.load(f)
    except OSError as e:
        return {"error": f"no manifest: {e}"}
    host, port = (addr or cfg["swarm"]["origin"]).rsplit(":", 1)
    n = len(m["pieces"])
    try:
        with socket.create_connection((host, int(port)), timeout=5) as s:
            s.sendall(b"LSWM" + bytes([2, 1]) + bytes.fromhex(m["root"]))
            st = s.recv(1)
            if st != b"\x00":
                return {"error": f"origin status {st!r} (1 = it does not serve this root: restart the origin after the manifest)",
                        "root": m["root"][:16]}
            need = 8 + 8 * ((n + 63) // 64)
            data = b""
            while len(data) < need:
                chunk = s.recv(need - len(data))
                if not chunk:
                    break
                data += chunk
            return {"held": have_reply(data, n), "pieces": n, "root": m["root"][:16], "piece_mib": m["piece_size"] >> 20,
                    "file_size": m["file_size"], "layout": bool(m.get("layout"))}
    except (OSError, ValueError) as e:
        return {"error": str(e)}


def tokenizer_same(cfg):
    a = C.read_header(cfg["model"]["gguf"], keep_arrays=True)["metadata"]
    b = C.read_header(cfg["model"]["gguf35"], keep_arrays=True)["metadata"]

    def tok(m):
        t = m.get("tokenizer.ggml.tokens")
        t = t.get("values", t) if isinstance(t, dict) else t
        return hashlib.sha256(json.dumps(t).encode()).hexdigest()[:16], len(t) if isinstance(t, list) else None
    return {"ornith": tok(a), "q35": tok(b), "same": tok(a) == tok(b)}


STAGE_R = """Stage R (reference), a plan only - not run by this script.
  Tool: llama.cpp CPU build /data/projects/strixflerp/build/gate-scratch-gufo/llama-build (commit 86a24a1),
  llama-eval-callback or a 30-line llama-cpp logits dump (teacher forcing = one prompt pass, logits of every row).
  Bytes: the file must be local and mapped: the 227.53 GiB GGUF does not fit one worker's 128 GB, and over NFS
  (2.5 GbE, 0.14-0.20 GB/s from control-node's USB disk) a mapped run re-reads what the page cache evicts.
  Per token the forward touches 6.09 GiB dense (embedding row, 4.78 GiB block dense, 0.78 GiB head) plus 10 experts
  x 60 blocks (4.25 GiB of rows scattered over 217 GiB of stacks).
  Memory: dense 6.1 GiB resident + llama.cpp compute/KV for 32 tokens (~1.5 GiB) + page cache for the experts; a
  32-token prompt routes to at most 32 x 10 = 320 distinct experts a block (of 512), i.e. up to 62 % of the stacks:
  ~135 GiB touched for one 32-token prompt, more than RAM, so experts are read and evicted per prompt.
  Plan: one worker (C, idle CPU, /cache free checked by --plan), the file streamed once by loom-swarm into /cache if
  /cache holds 228 GiB (it does not on any node today: C's --plan figure below), else onto the rail NFS mount at
  1 GB/s (vers=3 from 10.20.0.31; ark must export it there first) and run with --no-mmap off (mmap, page cache).
  Time: the CPU at ~60 GB/s memory bandwidth and ~70 GFLOP/s for Q4_K matmuls: a 32-row prompt pass is bound by
  reading the touched expert bytes once: from page cache ~3-5 min, from 1 GB/s NFS ~2.5 h a prompt (135 GiB);
  three prompts sharing a cache that cannot hold them: ~7 h over NFS, ~20 min from a local copy.
  Verdict: feasible only with a local full copy (228 GiB on one node's /cache or NVMe) or a RAM-resident subset;
  the plan writes R as optional, run on a node whose /cache is cleared for it, outside the P5 window."""


def stage_r_numbers(facts_by_name):
    c = facts_by_name.get("C") or {}
    return {"C_cache_avail_gib": round((c.get("cache_avail") or 0) / GIB, 1), "file_gib": 227.53,
            "fits_local": (c.get("cache_avail") or 0) > 245e9}


def cmd_plan(cfg, a):
    log = Log(os.path.join(RES, "plan.txt"))
    g = cfg["model"]["gguf"]
    plan4 = MP.py_plan(g, 4)
    plan3 = MP.py_plan(g, 3)
    ranges4 = [MP.py_plan(g, 1, [plan4["cuts"][r], plan4["cuts"][r + 1]])["ranks"][0] for r in range(4)]
    ranges3 = [MP.py_plan(g, 1, [plan3["cuts"][r], plan3["cuts"][r + 1]])["ranks"][0] for r in range(3)]
    doc = {"when": utc(), "cuts4": plan4["cuts"], "cuts3": plan3["cuts"], "nodes": {}, "problems": []}
    log(f"plan: world 4 cuts {plan4['cuts']}, world 3 cuts {plan3['cuts']}")
    origin = origin_have(cfg)
    doc["origin"] = origin
    log(f"origin {cfg['swarm']['origin']}: {origin}")
    if "error" in origin or origin.get("held") != origin.get("pieces"):
        doc["problems"].append(f"the origin does not offer the whole Ornith manifest: {origin}")
    mdir = os.path.dirname(cfg["swarm"]["manifest"])
    base = os.path.basename(g)
    doc["layouts"] = {f"r{r}of{w}": os.path.isfile(os.path.join(mdir, f"{base}.r{r}of{w}.swarm.json")) for w in (3, 4) for r in range(w)}
    doc["layout_origin"] = {}
    for r in range(4):
        lp = os.path.join(mdir, f"{base}.r{r}of4.swarm.json")
        if os.path.isfile(lp):
            o = origin_have(cfg, lp)
            with open(lp) as f:
                lm = json.load(f)
            want = MP.py_plan(g, 1, [plan4["cuts"][r], plan4["cuts"][r + 1]])["ranks"][0]["tensor_bytes"]
            o["covers_plan"] = lm["file_size"] - 11013920 == want
            doc["layout_origin"][r] = o
            if "error" in o or o.get("held") != o.get("pieces") or not o["covers_plan"]:
                doc["problems"].append(f"layout r{r}of4: {o}")
    log(f"rank layout artifacts (the stream source uses them for world 4): {doc['layouts']}; origin: {doc['layout_origin']}")
    try:
        doc["tokenizer"] = tokenizer_same(cfg)
    except (OSError, ValueError) as e:
        doc["tokenizer"] = {"error": str(e)}
    log(f"tokenizer (prompt ids are the 35B's): {doc['tokenizer']}")
    if not doc["tokenizer"].get("same"):
        doc["problems"].append("the 35B prompt ids are not valid for Ornith (tokenizers differ)")
    _, missing = (None, [])
    for s in ("build/flerp-p5/out/flerp-qwen35moe-generate", "build/flerp-p5-serve/out/flerp-qwen35moe-stage",
              "build/flerp-p5-serve/out/strixflerp-qwen36-35b-a3b.flerp-serve-stage", cfg["swarm"]["helper"]):
        if not os.path.isfile(os.path.join(ROOT, s)):
            missing.append(s)
    doc["bundle_missing"] = missing
    if missing:
        doc["problems"].append(f"bundle sources missing: {missing}")
    else:
        bad, err = gen_flag_check(cfg)
        doc["gen_flags"] = {"missing": bad, "error": err}
        log(f"generate flags the runner passes vs its --help: missing {bad}, error {err}")
        if err or bad:
            doc["problems"].append(f"flerp-qwen35moe-generate does not take the runner's flags: {err or bad}")
    facts = {}
    for i, n in enumerate(cfg["host"]["nodes"]):
        f = node_facts(cfg, n)
        f["hold"] = held(n["mgmt"])
        need4 = ranges4[i]["device_bytes"] + (cfg["host"]["arena_gib"] + cfg["host"]["mem_margin_gib"]) * GIB
        f["need_w4_bytes"] = int(need4)
        f["rank_w4"] = {"rank": i, "blocks": ranges4[i]["blocks"], "gib": round(ranges4[i]["device_bytes"] / GIB, 2)}
        if n["name"] in cfg["host"]["world3_nodes"]:
            j = cfg["host"]["world3_nodes"].index(n["name"])
            f["rank_w3"] = {"rank": j, "blocks": ranges3[j]["blocks"], "gib": round(ranges3[j]["device_bytes"] / GIB, 2)}
            f["need_w3_bytes"] = int(ranges3[j]["device_bytes"] + (cfg["host"]["arena_gib"] + cfg["host"]["mem_margin_gib"]) * GIB)
        p = []
        if not f["reachable"]:
            p.append("unreachable")
        if f.get("route_src") != n["rail"]:
            p.append(f"route to the origin leaves from {f.get('route_src')} ({f.get('route_dev')}), not the rail {n['rail']}")
        worst = max(f.get("need_w4_bytes", 0), f.get("need_w3_bytes", 0))
        # MemAvailable and GTT while serving are today's, not the window's: judged after the drain, recorded now
        f["fits_after_drain"] = f.get("mem_total", 0) - 8 * GIB > worst and f.get("gtt_total", 0) > worst
        if not f["fits_after_drain"]:
            p.append(f"needs {worst / GIB:.1f} GiB, MemTotal {f.get('mem_total', 0) / GIB:.1f}, GTT {f.get('gtt_total', 0) / GIB:.1f}")
        if f.get("home_avail", 0) < 2e9:
            p.append("home has under 2 GB free for the bundle and logs")
        if f.get("p5_units"):
            p.append(f"{f['p5_units']} flerp-p5 units already present")
        f["problems"] = p
        doc["problems"] += [f"{n['name']}: {x}" for x in p]
        facts[n["name"]] = f
        doc["nodes"][n["name"]] = f
        log(f"{n['name']} {n['mgmt']} rail {n['rail']}: reachable {f['reachable']}; route {f.get('route_dev')} src {f.get('route_src')}; "
            f"MemAvailable {f.get('mem_available', 0) / GIB:.1f}/{f.get('mem_total', 0) / GIB:.1f} GiB; GTT used {f.get('gtt_used', 0) / GIB:.1f}/"
            f"{f.get('gtt_total', 0) / GIB:.1f} GiB; /cache free {(f.get('cache_avail') or 0) / GIB:.1f} GiB ({f.get('cache')}); "
            f"RAM disk {f.get('ramdisk')}; serving up {f['serving_up']} {f.get('serving', [])[:2]}; hold {f['hold'][:80]!r}; "
            f"w4 rank {f['rank_w4']}; w3 {f.get('rank_w3')}; fits after drain {f['fits_after_drain']}; load {f.get('load1')}; problems {p}")
    doc["window_minutes"] = {f"{r} GB/s": window_estimate(ranges4, ranges3, r) for r in (0.126, 0.2)}
    log(f"window estimate (minutes; nv1 reads ~0.126 GB/s with one reader, shared by every rank): {doc['window_minutes']}")
    doc["stage_r"] = {"text": STAGE_R, **stage_r_numbers(facts)}
    for line in STAGE_R.splitlines():
        log(line)
    log(f"stage R numbers: {stage_r_numbers(facts)}")
    log(f"problems before a window ({len(doc['problems'])}): {doc['problems']}")
    save("plan.json", doc)
    return 0 if not doc["problems"] else 1


# --- units on workers ---------------------------------------------------------------------------------------------
class Window:
    """Holds, units and cleanup of one run. Every exit path goes through close()."""

    def __init__(self, cfg, log, card=None):
        self.cfg, self.log, self.card = cfg, log, card
        self.run_id = time.strftime("%m%d%H%M", time.gmtime())
        self.prefix = cfg["host"]["unit_prefix"] + self.run_id + "-"
        self.units = []      # (ip, unit)
        self.origins = []    # (host, unit): --rank-origin serves, on control-node or a storage-only worker (no hold)
        self.held_ips = []
        self.closed = False
        self.pre_serving = {}
        self.status = "starting"
        self._stop = threading.Event()
        self._th = None

    # holds
    def take_holds(self, ips):
        h = self.cfg["host"]
        env = dict(os.environ, OWNER=h["hold_owner"], REASON=h["hold_reason"], TTL=str(h["hold_ttl_s"]))
        env.pop("FORCE", None)
        for t in range(1, h["hold_tries"] + 1):
            if t > 1:
                env["DRAIN"] = str(h["hold_drain_s"])
            p = subprocess.run(["pod-quad", "hold"] + ips, capture_output=True, text=True, env=env, timeout=h["hold_drain_s"] + 3600)
            self.log(f"pod-quad hold {ips} (try {t}{', DRAIN=' + env['DRAIN'] if 'DRAIN' in env else ''}): rc {p.returncode}: "
                     f"{(p.stdout + p.stderr).strip()[-300:]}")
            if p.returncode == 0:
                self.held_ips = list(ips)
                self.snapshot_serving()
                return True
            time.sleep(60)
        return False

    def serving_probe(self, n):
        """{"dirs": {path: bytes}, "avail": bytes} of the node's serving confs' /cache dirs; None when it cannot tell."""
        cd = self.cfg["host"].get("serving_conf_dir", "/home/pod/tfp/run/conf")
        rc, text = on(n["mgmt"], f"cat {shlex.quote(cd)}/*.conf 2>/dev/null; true", 60)
        if rc != 0:
            return None
        dirs = {}
        for d in conf_cache_paths(text):
            rc, t = on(n["mgmt"], f"du -sxB1 {shlex.quote(d)} 2>/dev/null | cut -f1; true", 600)
            dirs[d] = int(t.split()[0]) if rc == 0 and t.split() and t.split()[0].isdigit() else 0
        rc, t = on(n["mgmt"], "df -B1 --output=avail /cache | tail -1", 30)
        if rc != 0 or not t.strip().isdigit():
            return None
        return {"dirs": dirs, "avail": int(t.strip())}

    def snapshot_serving(self):
        self.pre_serving = {n["name"]: self.serving_probe(n) for n in self.cfg["host"]["nodes"]}
        self.log(f"serving dirs before the window: {json.dumps(self.pre_serving)}")

    def restore_serving_room(self):
        """After the units are gone, before the hold is released: a node whose serving data shrank, or that cannot hold
        its serving set, gets this run's own slices evicted through the place interface (cancel the request, then
        `evict --apply` as the holder). Returns {node: shortfall bytes after} for nodes still short (None = unknown)."""
        need = int(self.cfg["host"].get("serving_need_gib", 196) * GIB)
        pc, owner, still = place_cmd(), self.cfg["host"]["hold_owner"], {}
        for n in self.cfg["host"]["nodes"]:
            name = n["name"]
            post = self.serving_probe(n)
            if post is not None and not post["dirs"] and not (getattr(self, "pre_serving", {}).get(name) or {}).get("dirs"):
                continue     # no serving conf names a /cache dir on this node (A, C serve strixite): nothing to crowd
            if post is None:
                self.log(f"{name}: serving dirs/free space unreadable after the run; evicting this run's slices to be safe")
            elif not slices_to_evict(getattr(self, "pre_serving", {}).get(name), post, need):
                continue
            else:
                self.log(f"{name}: window removed {serving_lost(getattr(self, 'pre_serving', {}).get(name), post)} / shortfall "
                         f"{serving_shortfall(post, need) / GIB:.1f} GiB: evicting this run's slices")
            rid = f"flerp-p5-{name}"
            for argv in (pc + ["cancel", "--id", rid], pc + ["evict", "--node", name, "--owner", owner, "--apply"]):
                p = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
                self.log(f"{' '.join(argv[len(pc):])}: rc {p.returncode}: {(p.stdout + p.stderr).strip()[-300:]}")
            post = self.serving_probe(n)
            short = None if post is None else serving_shortfall(post, need)
            if short is None or short > 0:
                still[name] = short
        return still

    def release_holds(self):
        if not self.held_ips:
            return
        env = dict(os.environ, OWNER=self.cfg["host"]["hold_owner"])
        p = subprocess.run(["pod-quad", "release"], capture_output=True, text=True, env=env, timeout=600)
        self.log(f"pod-quad release (owner {env['OWNER']}): rc {p.returncode}: {(p.stdout + p.stderr).strip()[-200:]}")
        self.held_ips = []

    def wait_quiet(self, nodes, timeout):
        """Serving down on every node (pod-quad's drain stops it); False after timeout."""
        end = time.time() + timeout
        while True:
            up = {n["name"]: node_facts(self.cfg, n) for n in nodes}
            busy = {k: (f.get("serving"), f.get("gtt_used")) for k, f in up.items() if f["serving_up"] or not f["reachable"]}
            if not busy:
                return True, up
            if time.time() > end:
                self.log(f"serving still up: {busy}")
                return False, up
            time.sleep(20)

    # units
    def start(self, n, tag, argv, env="", serving_ok=False):
        """argv as a transient user unit; SIGTERM to the unit's main process is forwarded to the program and its
        exit status still lands in <tag>.rc. serving_ok: disk-only jobs (stage) that run beside serving."""
        h = self.cfg["host"]
        f = node_facts(self.cfg, n)
        if f["serving_up"] and not serving_ok:
            raise RuntimeError(f"refusing to start {tag} on {n['name']}: serving is up ({f.get('serving')}, GTT used {f.get('gtt_used')})")
        unit = self.prefix + tag
        w = h["work"]
        inner = f"cd {w} && {env} exec {' '.join(shlex.quote(x) for x in argv)} > {tag}.jsonl 2> {tag}.err"
        wrapped = (f"trap 'kill -TERM $c 2>/dev/null' TERM INT; bash -c {shlex.quote(inner)} & c=$!; "
                   f"wait $c; r=$?; [ $r -gt 128 ] && kill -0 $c 2>/dev/null && {{ wait $c; r=$?; }}; echo $r > {w}/{tag}.rc")
        cmd = (XDG + f"rm -f {w}/{tag}.jsonl {w}/{tag}.err {w}/{tag}.rc; "
               f"systemd-run --user --collect --unit={unit} bash -c {shlex.quote(wrapped)}")
        rc, text = on(n["mgmt"], cmd)
        if rc != 0:
            raise RuntimeError(f"could not start {unit} on {n['name']}: {text[-300:]}")
        self.units.append((n["mgmt"], unit))
        return unit

    def start_origin(self, r, o):
        """--rank-origin: `loom-swarm.place serve --ranges` of a PLACE-staged file as a transient user unit on its host
        (control-node, or a worker reached from control-node directly; disk and rail only, beside its serving)."""
        man = self.cfg["swarm"]["manifest"]
        unit = f"{self.prefix}origin-r{r}"
        serve = [SWARM_BIN, "serve", "--listen", o["listen"], "--manifest", man, "--file", o["file"], "--ranges", o["ranges"]]
        check = f"f=$(readlink -f {shlex.quote(o['file'])}) && [ -s \"$f\" ] && echo REAL $f"
        cmd = (XDG + f"{check} && systemd-run --user --collect --unit={unit} -p Nice=5 "
               + " ".join(shlex.quote(x) for x in serve))
        if o["host"] == "control":
            p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=60)
            rc, text = p.returncode, p.stdout + p.stderr
        else:
            rc, text = on(o["host"], cmd, 60)
        real = (re.findall(r"^REAL (\S+)", text, re.M) or [""])[0]
        if rc != 0 or not real or cold_path(real):
            raise RuntimeError(f"origin for rank {r} on {o['host']} not started (rc {rc}, file {real or '?'}): {text[-300:]}")
        self.origins.append((o["host"], unit))
        with open(man) as f:
            m = json.load(f)
        with open(o["ranges"]) as f:
            rng = [tuple(int(x) for x in line.split()) for line in f if line.strip()]
        want = pieces_in_ranges(m, rng)
        for _ in range(30):
            time.sleep(2)
            have = origin_have(self.cfg, man, o["listen"])
            if have.get("held", 0) >= want:
                self.log(f"origin rank {r}: {o['listen']} on {o['host']} serves {have['held']} of {have['pieces']} pieces "
                         f"(the rank's ranges need {want}) from {real}")
                return have
        raise RuntimeError(f"origin for rank {r} at {o['listen']}: HAVE {have}, the ranges need {want} pieces")

    def stop_origins(self):
        for host, unit in self.origins:
            cmd = XDG + f"systemctl --user stop {unit} 2>/dev/null; systemctl --user is-active {unit} 2>/dev/null; true"
            if host == "control":
                out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=60).stdout
            else:
                out = on(host, cmd, 60)[1]
            self.log(f"origin {unit} on {host} stopped (is-active: {out.strip() or 'inactive'})")
        self.origins = []

    def term(self, n, tag):
        on(n["mgmt"], XDG + f"systemctl --user kill -s TERM --kill-whom=main {self.prefix}{tag} 2>/dev/null; true")

    def events(self, n, tag):
        rc, text = on(n["mgmt"], f"cat {self.cfg['host']['work']}/{tag}.jsonl 2>/dev/null", 120)
        out = []
        for line in text.splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
        return out

    def rc(self, n, tag):
        _, text = on(n["mgmt"], f"cat {self.cfg['host']['work']}/{tag}.rc 2>/dev/null", 30)
        t = text.strip()
        return int(t) if t.lstrip("-").isdigit() else None

    def wait_event(self, n, tag, kinds, timeout):
        end = time.time() + timeout
        while time.time() < end:
            ev = self.events(n, tag)
            if any(e.get("event") in kinds for e in ev) or self.rc(n, tag) is not None:
                return ev
            time.sleep(10)
        return self.events(n, tag)

    def wait_rc(self, n, tag, timeout):
        end = time.time() + timeout
        while time.time() < end:
            r = self.rc(n, tag)
            if r is not None:
                return r
            time.sleep(10)
        return None

    def fetch(self, n, tag, dest, exts=("jsonl", "err")):
        os.makedirs(dest, exist_ok=True)
        for ext in exts:
            subprocess.run(["rsync", "-a", "-e", " ".join(SSH), f"pod@{n['mgmt']}:{self.cfg['host']['work']}/{tag}.{ext}",
                            os.path.join(dest, f"{tag}.{ext}")], capture_output=True, timeout=900)

    # cleanup
    def stop_all(self):
        """Stops every unit of this run. Returns {node: {"units": n, "procs": n}} after the stop (None: could not tell).
        The process count matches the bundle path with a bracketed last letter, so the pgrep's own shell (whose
        command line holds the pattern) is not counted: run 11121559 counted exactly that shell, 1 a node, and kept
        the four-node hold 53 min after every unit was gone."""
        left = {}
        for n in self.cfg["host"]["nodes"]:
            on(n["mgmt"], XDG + f"systemctl --user stop '{self.prefix}*' 2>/dev/null; true", 180)
        time.sleep(3)
        pat = proc_pattern(self.cfg["host"]["bundle"])
        for n in self.cfg["host"]["nodes"]:
            ip = n["mgmt"]
            probe = XDG + f"echo units $(systemctl --user list-units --all --no-legend --plain '{self.prefix}*' | grep -c .); echo procs $(pgrep -fc {shlex.quote(pat)})"
            got = parse_left(*on(ip, probe, 60))
            if got and got["procs"]:
                on(ip, f"pkill -KILL -f {shlex.quote(pat)}; true")
                time.sleep(2)
                got = parse_left(*on(ip, probe, 60))
            left[n["name"]] = got
        return left

    def close(self, why):
        if self.closed:
            return
        self.closed = True
        self._stop.set()
        self.log(f"cleanup ({why}): stopping every {self.prefix}* unit")
        left = self.stop_all()
        self.stop_origins()
        self.log(f"left after stop (units, processes): {left}")
        if not units_gone(left):
            self.log("this run's units are not confirmed gone: the holds are KEPT (they expire with their TTL); stop them by hand first")
            self.comment(f"FLERP-P5 cleanup left units {left}; holds kept until they are gone")
            return
        if not self.held_ips:
            # no hold was taken (preflight, refused hold): /cache is never touched and there is nothing to release.
            # 2026-11-13: a --preflight-only pass fell through to restore_serving_room and evicted A's flerp-p5 slices.
            self.log("no hold was taken: /cache untouched, nothing to release")
            return
        stray = {k: v["procs"] for k, v in left.items() if v["procs"]}
        if stray:
            # every process of this run lives in one of its units' cgroups, so with the units gone these are not ours
            self.log(f"processes outside this run's units still match the bundle path: {stray} (not this run's; holds released)")
        try:
            short = self.restore_serving_room()
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            short = {"all": f"cleanup failed: {e}"}
        if short:
            self.log(f"SERVING SET STILL SHORT after evicting this run's slices: {short} (bytes; None = unreadable)")
            self.comment(f"FLERP-P5 window closed ({why}); WARNING nodes still short of the serving set on /cache: {short}")
        if BEFORE_RELEASE:
            try:
                p = subprocess.run(["bash", "-c", BEFORE_RELEASE], capture_output=True, text=True, timeout=900)
                self.log(f"before release rc {p.returncode}: {(p.stdout + p.stderr).strip()[-1500:]}")
            except (subprocess.SubprocessError, OSError) as e:
                self.log(f"before release failed: {e}")
        self.release_holds()
        self.comment(f"FLERP-P5 window closed ({why}); units stopped, holds released")

    # card comments
    def comment(self, body):
        if not self.card:
            return
        env = dict(os.environ, KANBAN_ACTOR="agent")
        subprocess.run(["akb", "comment", "--card", self.card, "--body", body], capture_output=True, text=True, env=env, timeout=60)

    def start_heartbeat(self):
        if not self.card:
            return

        def beat():
            while not self._stop.wait(self.cfg["host"]["card_comment_s"]):
                self.comment(f"FLERP-P5 run {self.run_id} {utc()}: {self.status}")
        self._th = threading.Thread(target=beat, daemon=True)
        self._th.start()


# --- the generator over TCP ---------------------------------------------------------------------------------------
def rank_manifest(cfg, world, rank):
    """The world-4 rank's layout artifact (exactly its bytes, staged by the swarm task) when present, else the full
    manifest (world 3 has no layouts: the stream asks only the pieces over the rank's ranges)."""
    b = cfg["host"]["bundle"]
    lay = os.path.join(os.path.dirname(cfg["swarm"]["manifest"]), f"{os.path.basename(cfg['model']['gguf'])}.r{rank}of4.swarm.json")
    return f"{b}/ornith.r{rank}of4.swarm.json" if world == 4 and os.path.isfile(lay) else f"{b}/ornith.swarm.json"


def source_flags(cfg, src, world=4, rank=0):
    b = cfg["host"]["bundle"]
    if src == "swarm":
        s = cfg["swarm"]
        return ["--swarm-helper", f"{b}/loom-swarm-stream", "--swarm-manifest", rank_manifest(cfg, world, rank), "--swarm-from", s["origin"],
                "--swarm-conns", str(s["conns"])]
    return []


# --- pre-staging onto worker /cache (outside any hold) ------------------------------------------------------------
def cache_path(cfg):
    """Each node's staged copy: a sparse file the size of the GGUF holding, at their own offsets, the bytes of every
    rank the node runs (world 4 and world 3). /cache is the worker's scratch NVMe (owner: model data sits there, never
    pulled from control-node on demand)."""
    d = cfg.get("cache", {}).get("dir", "/cache/flerp-p5")
    return f"{d}/{os.path.basename(cfg['model']['gguf'])}.staged"


def node_block_ranges(cfg, name):
    g = cfg["model"]["gguf"]
    out = []
    for world in (4, 3):
        cuts = MP.py_plan(g, world)["cuts"]
        for r, n in enumerate(nodes_of(cfg, world)):
            if n["name"] == name:
                out.append((cuts[r], cuts[r + 1]))
    return out


def mtp_spans(cfg):
    """ORNITH-O3: (offset, end) of the MTP block's tensors (blk.<blocks>.*, past the stack) and token_embd: what the
    probe's head (mtp_head.hpp) reads beside world 4's last rank."""
    h = C.read_header(cfg["model"]["gguf"])
    blocks = MP.py_plan(cfg["model"]["gguf"], 4)["cuts"][-1]
    out = set()
    for t in h["tensors"]:
        if t["name"].startswith(f"blk.{blocks}.") or t["name"] == "token_embd.weight":
            out.add((t["offset"], t["offset"] + t["bytes"]))   # read_header: offset is absolute in the file
    return out


def node_byte_ranges(cfg, name, with_mtp=False):
    """Merged [offset, end) byte ranges of the tensors of every rank the node runs (with_mtp: and, on world 4's last
    rank's node, the MTP head's)."""
    h = C.read_header(cfg["model"]["gguf"])
    spans = {(e[3], e[3] + e[2]) for rng in node_block_ranges(cfg, name) for e in C.stage_plan(h, 1, rng)["ranks"][0]["entries"]}
    if with_mtp and nodes_of(cfg, 4)[-1]["name"] == name:
        spans |= mtp_spans(cfg)
    spans = sorted(spans)
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


PLACE_BIN = "/data/models/.swarm/bin/loom-swarm.place"
PLACE_RANGES = "/data/models/.place/ranges"   # workers read this over the /data/models NFS view


def place_cmd():
    """argv prefix of the placement interface (PLACE 103f0f5a96): `pod-weights place` once installed, else the
    loom-swarm.place binary it forwards to. PLACE_BIN overrides."""
    if os.environ.get("PLACE_BIN"):
        return [os.environ["PLACE_BIN"], "place"]
    pw = shutil.which("pod-weights")
    if pw:
        r = subprocess.run([pw, "place"], capture_output=True, text=True)
        if "loom-swarm place" in r.stdout + r.stderr:
            return [pw, "place"]
    return [PLACE_BIN, "place"]


def cmd_stage(cfg, a):
    """Before the window, no hold: each node's ranks into /cache through the placement interface (`place request`,
    `run`, `verify`): the same sparse file cache_path(cfg), bytes pulled from the origin, every 16 MiB piece BLAKE3-checked
    against the manifest, resumable, so a re-stage of an intact copy copies nothing. Runs beside serving (no hold, no
    GPU); nodes held by someone else are deferred by place; nothing on /cache is deleted by hand (the request's window
    ends it). --dry-run prints the plan only."""
    log = Log(os.path.join(RES, "stage.txt"))
    out, s = cache_path(cfg), cfg["swarm"]
    pc = place_cmd()
    rc_all = 0
    for n in cfg["host"]["nodes"]:
        if a.nodes and n["name"] not in a.nodes.split(","):
            continue
        ranges = node_byte_ranges(cfg, n["name"], getattr(a, "with_mtp", False))
        need = sum(e - o for o, e in ranges)
        rid = f"flerp-p5-{n['name']}"
        rf = os.path.join(PLACE_RANGES, f"{rid}.ranges")
        log(f"{n['name']}: blocks {node_block_ranges(cfg, n['name'])}, {need / GIB:.1f} GiB into {out} by place request {rid}")
        if a.dry_run:
            continue
        os.makedirs(PLACE_RANGES, exist_ok=True)
        with open(rf, "w") as fo:
            for o, e in ranges:
                fo.write(f"{o} {e - o}\n")
        req = pc + ["request", "--id", rid, "--nodes", n["name"], "--dest", os.path.dirname(out), "--out", os.path.basename(out),
                    "--manifest", s["manifest"], "--from", s["origin"], "--ranges", f"{n['name']}={rf}",
                    "--until", a.until, "--owner", "FLERP-P5"]
        if a.card:
            req += ["--card", a.card]
        t0 = time.time()
        rc = subprocess.run(req).returncode
        rc = rc or subprocess.run(pc + ["run", "--id", rid]).returncode
        rc = rc or subprocess.run(pc + ["verify", "--id", rid]).returncode
        log(f"{n['name']}: place {rid} run+verify exit {rc} in {time.time() - t0:.0f} s")
        rc_all = rc_all or rc
    return rc_all


def file_fallback(win, n, tag, rng):
    """--source file: stream the rank's bytes into a sparse file on the RAM disk (never /cache on F or D), return its
    path. The caller loads with --data-file --punch-behind and deletes the file after the load."""
    cfg = win.cfg
    g = cfg["model"]["gguf"]
    h = C.read_header(g)
    plan = C.stage_plan(h, 1, tuple(rng))["ranks"][0]
    f = node_facts(cfg, n)
    need = plan["tensor_bytes"] + plan["device_bytes"] + cfg["host"]["mem_margin_gib"] * GIB
    if f.get("ramdisk") == "absent":
        raise RuntimeError(f"{n['name']}: no RAM disk at {cfg['swarm']['ram_disk']} (swarm.sh seed mounts it); the fallback will not use /cache")
    if f.get("mem_available", 0) < need:
        raise RuntimeError(f"{n['name']}: MemAvailable {f.get('mem_available', 0) / GIB:.1f} GiB < the copy + the GPU share + margin "
                           f"{need / GIB:.1f} GiB (one 128 GB pool)")
    ranges = os.path.join(SCRATCH, f"{tag}.ranges")
    os.makedirs(SCRATCH, exist_ok=True)
    with open(ranges, "w") as fo:
        for e in plan["entries"]:
            fo.write(f"{e[3]} {e[2]}\n")
    w, b = cfg["host"]["work"], cfg["host"]["bundle"]
    subprocess.run(["rsync", "-a", "-e", " ".join(SSH), ranges, f"pod@{n['mgmt']}:{w}/{tag}.ranges"], check=True)
    out = f"{cfg['swarm']['ram_disk']}/flerp-p5-{tag}.gguf"
    s = cfg["swarm"]
    win.start(n, tag + "-fetch", ["bash", "-c", f"set -o pipefail; {b}/loom-swarm-stream stream --manifest {b}/ornith.swarm.json --from {s['origin']} "
                                                f"--conns {s['conns']} --ranges {w}/{tag}.ranges | python3 {b}/swarm_frames_to_file.py {out} {h['file_bytes']}"])
    rc = win.wait_rc(n, tag + "-fetch", cfg["run"]["load_timeout_s"])
    if rc != 0:
        raise RuntimeError(f"{n['name']}: fetch into {out} exited {rc}")
    return out


def gen_argv(cfg, world, r, src, dumps, hosts, id_files, data_path=None):
    """One rank's flerp-qwen35moe-generate --staged tcp command line (gen_world runs it; --plan checks its flags)."""
    run, w, b = cfg["run"], cfg["host"]["work"], cfg["host"]["bundle"]
    max_tokens = SPEC["max_tokens"] if SPEC else PROBE["max_tokens"] if PROBE else run["max_tokens"]
    argv = [f"{b}/flerp-qwen35moe-generate", "--staged", "tcp", "--model", cfg["model"]["gguf"],
            "--context", str(SPEC["context"] if SPEC else run["context"]), "--max-tokens", str(max_tokens), "--world", str(world), "--cuts", "even", "--port", str(run["port"] + (0 if world == 4 else 10)),
            "--host", hosts, "--epoch", str(run["epoch"] + world), "--deadline-ms", str(run["deadline_ms"]),
            "--setup-deadline-ms", str(run["setup_deadline_ms"]), "--graph-modes", run["graph_modes"] if world == 4 else "on", "--rank", str(r)]
    if world == 4 and r in SWARM_RANKS:
        o = RANK_ORIGINS[r]
        argv += ["--swarm-helper", f"{b}/loom-swarm-stream", "--swarm-manifest", f"{b}/ornith.swarm.json", "--swarm-from", o["listen"],
                 "--swarm-conns", str(cfg["swarm"]["conns"])]
    elif src == "file":
        argv += ["--data-file", data_path, "--punch-behind"]
    elif src == "cache":
        argv += ["--data-file", data_path]
    else:
        argv += source_flags(cfg, src, world, r)
    rows = (LAUNCH or {}).get("rows") or PASS["rows"]
    if rows:
        argv += ["--rows", rows]
    if r == 0:
        specs = (LAUNCH or {}).get("specs") or PASS["run_specs"]
        if specs:
            argv += ["--run-specs", specs]
        elif PASS["reply"]:
            argv += ["--reply", PASS["reply"]]
    if dumps:
        tag = f"w{world}-r{r}"
        argv += ["--dump", f"{w}/{tag}.dump", "--dump-layers", ",".join(str(x) for x in run["check_blocks4"][r]),
                 "--dump-rows", ",".join(str(x) for x in run["dump_rows"])]
    if PROBE and r == world - 1:
        # ORNITH-O3: the last rank drafts with the MTP head; it needs the prompts' ids and the head's bytes (staged)
        argv += ["--mtp-probe", str(PROBE["depth"]), "--mtp-data-file", cache_path(cfg)]
    if SPEC and r == world - 1:
        # ORNITH-O3 step two: the last rank drafts with the MTP head, read from its staged file (stage --with-mtp)
        argv += ["--draft-head", "--mtp-data-file", cache_path(cfg), "--draft-costs", SPEC["costs"]]
    return argv + (id_files if r == 0 or (PROBE and r == world - 1) else [])


def runner_gen_flags(cfg):
    """Every option the runner can pass to flerp-qwen35moe-generate (both worlds, both sources, with and without dumps)."""
    out = set()
    for world in (4, 3):
        for r in range(world):
            for src in ("swarm", "file", "cache"):
                for dumps in (True, False):
                    if dumps and world != 4:
                        continue
                    out |= argv_flags(gen_argv(cfg, world, r, src, dumps, "h", ["--ids-file", "x.ids"], "/x"))
    return out | {"--rows", "--reply", "--run-specs", "--mtp-probe", "--mtp-data-file", "--draft-head", "--draft-costs"}


def gen_flag_check(cfg):
    """--plan and run, before any hold: the local build of flerp-qwen35moe-generate (the one the bundle copies) lists every
    option the runner passes in its `--staged tcp --help`. Run 11121559's rank 0 died on 'unknown option --dump': the
    stage driver documented --dump but its parser never took it. Returns (missing, error)."""
    exe = os.path.join(GEN_BUILD, "out", "flerp-qwen35moe-generate")
    lib = os.path.join(ROOT, "build", "flerp-p4", "bundle", "lib")
    env = dict(os.environ, LD_LIBRARY_PATH=lib + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""))
    try:
        p = subprocess.run([exe, "--staged", "tcp", "--help"], capture_output=True, text=True, timeout=60, env=env)
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"{exe} --staged tcp --help: {e}"
    have = help_flags(p.stdout)
    if p.returncode != 0 or not have:
        return None, f"{exe} --staged tcp --help: rc {p.returncode}, no option list (a build from before the --help list): {(p.stdout + p.stderr)[-200:]}"
    return sorted(runner_gen_flags(cfg) - have), None


def gen_world(win, world, src, steps_doc, dumps, key="", res=None):
    """One world's load and runs. key prefixes the unit tags and evidence names (one launch of several under a hold);
    res is where the evidence goes (default RES). With a non-file source every rank starts at once (each reads its own
    NVMe, or the control-node NVMe origin over the rail), then each is waited for."""
    cfg, run = win.cfg, win.cfg["run"]
    res = res or RES
    g = cfg["model"]["gguf"]
    nodes = nodes_of(cfg, world)
    plan = MP.py_plan(g, world)
    cuts = plan["cuts"]
    hosts = ",".join(n["rail"] for n in nodes)
    w, b = cfg["host"]["work"], cfg["host"]["bundle"]
    ids = SPEC["ids"] if SPEC else PROBE["ids"] if PROBE else MP.load_json(cfg["model"]["prompt_ids"])
    names = SPEC["names"] if SPEC else PROBE["names"] if PROBE else cfg["model"]["prompts"]
    id_files = []
    for p in names:
        local = os.path.join(SCRATCH, f"{p}.ids")
        os.makedirs(SCRATCH, exist_ok=True)
        with open(local, "w") as f:
            f.write(" ".join(str(i) for i in ids[p]))
        for dest in ([nodes[0], nodes[-1]] if PROBE else [nodes[0]]):
            subprocess.run(["rsync", "-a", "-e", " ".join(SSH), local, f"pod@{dest['mgmt']}:{w}/{p}.ids"], check=True)
        id_files += ["--ids-file", f"{w}/{p}.ids"]
    env = f"LD_LIBRARY_PATH={b}/lib"
    tags, files, why, doc = [], [], [], {"world": world, "cuts": cuts, "nodes": [n["name"] for n in nodes], "source": src, "ranks": [],
                                         "swarm_ranks": sorted(SWARM_RANKS) if world == 4 else [], "launch": LAUNCH, "names": names}
    t0 = time.time()
    together = src != "file"
    for r, n in enumerate(nodes):
        tag = f"{key}w{world}-r{r}"
        path = cache_path(cfg) if src == "cache" else None
        if src == "file":
            path = file_fallback(win, n, tag, [cuts[r], cuts[r + 1]])
            files.append((n, path))
        argv = gen_argv(cfg, world, r, src, dumps, hosts, id_files, path)
        how = "swarm from " + cfg["swarm"]["origin"] if world == 4 and r in SWARM_RANKS else src
        win.log(f"{key}world {world} rank {r} on {n['name']} blocks [{cuts[r]}, {cuts[r + 1]}): start ({how}): {' '.join(argv[1:])[:400]}")
        win.start(n, tag, argv, env)
        tags.append(tag)
        if together:
            continue
        ev = win.wait_event(n, tag, ("loaded", "error"), run["load_timeout_s"])
        ld = next((e for e in ev if e.get("event") == "loaded"), None)
        win.log(f"world {world} rank {r} loaded: {json.dumps((ld or {}).get('placed'))[:500] if ld else 'NOT LOADED: ' + str(ev[-2:])}")
        if files:
            on(files[-1][0]["mgmt"], f"rm -f {files[-1][1]}")      # the RAM-disk copy goes as soon as the weights are resident
        if not ld:
            why.append(f"world {world} rank {r} did not load")
            break
        win.status = f"world {world}: rank {r} of {world} loaded"
    if together:
        for r, (n, tag) in enumerate(zip(nodes, tags)):
            ev = win.wait_event(n, tag, ("loaded", "error"), run["load_timeout_s"])
            ld = next((e for e in ev if e.get("event") == "loaded"), None)
            win.log(f"{key}world {world} rank {r} loaded at +{time.time() - t0:.0f} s: "
                    f"{json.dumps((ld or {}).get('placed'))[:400] if ld else 'NOT LOADED: ' + str(ev[-2:])}")
            if not ld:
                why.append(f"{key}world {world} rank {r} did not load")
                break
            win.status = f"{key}world {world}: rank {r} of {world} loaded"
    rcs = []
    if not why:
        for r, (n, tag) in enumerate(zip(nodes, tags)):
            rcs.append(win.wait_rc(n, tag, 3600 if SPEC else run["run_timeout_s"]))   # a spec launch: 84 runs of 256 tokens
    evs = []
    for r, (n, tag) in enumerate(zip(nodes, tags)):
        win.fetch(n, tag, res)
        evs.append(win.events(n, tag))
    for r, ev in enumerate(evs):
        pl = next((e.get("placed") or {} for e in ev if e.get("event") == "loaded"), {})
        want = MP.py_plan(g, 1, [cuts[r], cuts[r + 1]])["ranks"][0]["device_bytes"]
        if pl.get("allocated_bytes") != want:
            why.append(f"{key}world {world} rank {r}: allocated {pl.get('allocated_bytes')}, plan {want}")
        swarm_rank = src == "swarm" or (world == 4 and r in SWARM_RANKS)
        if not swarm_rank and src == "cache" and pl.get("data_path") != cache_path(cfg):
            why.append(f"{key}world {world} rank {r}: the bytes did not come from {cache_path(cfg)}")
        if swarm_rank and not pl.get("swarm"):
            why.append(f"{key}world {world} rank {r}: the bytes did not come from loom-swarm")
        if r < len(rcs) and rcs[r] != 0:
            why.append(f"{key}world {world} rank {r} exited {rcs[r]}")
        doc["ranks"].append({"rank": r, "node": nodes[r]["name"], "rc": rcs[r] if r < len(rcs) else None, "loaded": pl,
                             "timing": [e for e in ev if e.get("event") == "stage_timing"][-1:],
                             "dump": next((e for e in ev if e.get("event") == "moe_dump"), None)})
    doc["rank_events"] = {r: {"rc": rcs[r] if r < len(rcs) else None, "events": ev} for r, ev in enumerate(evs)}
    runs = split_runs(evs[0] if evs else [], len(names))
    if PROBE and evs:
        # ORNITH-O3: the last rank's probe events (one a prompt a run) and its head's load
        doc["mtp_probe"] = [e for e in evs[-1] if e.get("event") == "mtp_probe"]
        doc["mtp_head"] = next((e for e in evs[-1] if e.get("event") == "mtp_head"), None)
        for e in doc["mtp_probe"]:
            k = e.get("session", 0) // max(1, len(names))
            i = e["prompt"]
            r0 = runs[k][i]["ids"] if k < len(runs) and i < len(runs[k]) else None
            if r0 is None or e["ids"][:len(r0)] != r0:
                why.append(f"world {world} mtp probe session {e.get('session')}: the ids the probe saw differ from rank 0's")
        if len(doc["mtp_probe"]) != len(runs) * len(names):
            why.append(f"world {world}: {len(doc['mtp_probe'])} mtp_probe events for {len(runs)} runs x {len(names)} prompts")
    doc["runs"] = [[{k: v for k, v in e.items() if k != "logits_sha256"} for e in r] for r in runs]
    doc["sha"] = [[e["logits_sha256"] for e in r] for r in runs]
    doc["speed"] = speed(runs[0]) if runs else []
    doc["seconds"] = round(time.time() - t0, 1)
    steps_doc[f"{key}w{world}"] = doc
    return runs, why, (nodes, tags)


def block_checks(win, nodes, tags):
    cfg, crit = win.cfg, win.cfg["criteria"]
    out, why = [], []
    for r, (n, tag) in enumerate(zip(nodes, tags)):
        ldump = os.path.join(SCRATCH, f"{tag}.dump")
        subprocess.run(["rsync", "-a", "-e", " ".join(SSH), f"pod@{n['mgmt']}:{cfg['host']['work']}/{tag}.dump", ldump],
                       capture_output=True, timeout=900)
        if not os.path.isfile(ldump):
            why.append(f"stage {r}: no dump")
            continue
        th = cfg["host"]["block_check_threads"]
        p = subprocess.run(["ionice", "-c3", "nice", "-n", "19", "env", f"OMP_NUM_THREADS={th}", f"OPENBLAS_NUM_THREADS={th}", sys.executable,
                            os.path.join(ROOT, "gate", "moe_block_check.py"), "--gguf", cfg["model"]["gguf"], "--gguf-py", cfg["host"]["gguf_py"],
                            "--dump", ldump], capture_output=True, text=True, timeout=7200)
        try:
            check = json.loads(p.stdout)
        except ValueError:
            check = {"error": (p.stderr or "no output")[-400:]}
        os.remove(ldump)
        save(f"block-check-{tag}.json", check)
        j = MP.judge_block_check(check, crit, crit["check_records_min"])
        out.append({"stage": r, "node": n["name"], **{k: v for k, v in j.items() if k != "detail"}})
        why += [f"stage {r} block check: {x}" for x in j["why"]]
        win.log(f"block check stage {r} ({n['name']}): {out[-1]}")
    return out, why


def serve_step(win, src, doc):
    cfg, run = win.cfg, win.cfg["run"]
    g, b, w = cfg["model"]["gguf"], cfg["host"]["bundle"], cfg["host"]["work"]
    nodes = nodes_of(cfg, 4)
    peers = ",".join(n["rail"] for n in nodes) + f":{run['port'] + 20}"
    kv = {"weights": g, "context": run["context"], "world": 4, "placement": "even", "epoch": run["epoch"] + 40, "peers": peers,
          "single_reader": "on", "pause_ms": 0, "setup_deadline_ms": run["setup_deadline_ms"], "deadline_ms": run["deadline_ms"]}
    nd = CONFIG_RT.get("drafts", 0)
    if nd:
        kv["drafts"] = nd   # ORNITH-O3: caps.max_draft on every stage; the last stage also gets draft_head (below)
    doc["config"] = RTCFG["name"]
    if src == "swarm":
        kv.update(swarm_helper=f"{b}/loom-swarm-stream", swarm_manifest=rank_manifest(cfg, 4, 0), swarm_from=cfg["swarm"]["origin"],
                  swarm_conns=cfg["swarm"]["conns"])
    if src == "cache":
        kv["data_file"] = cache_path(cfg)
    env = f"LD_LIBRARY_PATH={b}/lib {CF.env_prefix(RTCFG)}".strip()
    gen = os.path.join(SCRATCH, "generation_config.json")
    with open(gen, "w") as f:
        json.dump({"temperature": 0}, f)
    subprocess.run(["rsync", "-a", "-e", " ".join(SSH), gen, f"pod@{nodes[0]['mgmt']}:{w}/generation_config.json"], check=True)
    why = []
    front = [f"{b}/flerp-serve", "--host", "127.0.0.1", "--port", str(run["http_port"]), "--model-id", "ornith-1.5-397b",
             "--generation-config", f"{w}/generation_config.json"] + (["--delivery", "tokens"] if RTCFG["stage"].get("reply") == "token" else []) + (["--drafts", str(nd)] if nd else []) + \
            [f"executor.{k}={v}" for k, v in kv.items()] + \
            [f"text.weights={g}", f"text.context={run['context']}"]
    win.start(nodes[0], "serve-r0", front, env)
    ev = win.wait_event(nodes[0], "serve-r0", ("waiting", "error"), run["load_timeout_s"])
    win.log(f"front end rank 0: {[e for e in ev if e.get('event') in ('loaded', 'waiting', 'error')][:2]}")
    for r in range(1, 4):
        kr = dict(kv, swarm_manifest=rank_manifest(cfg, 4, r)) if src == "swarm" else dict(kv)
        kr.pop("drafts", None)   # drafts is rank 0's key; the last stage takes draft_head instead
        if nd and r == 3:
            kr.update(draft_head="on", mtp_data_file=cache_path(cfg))   # the PLACE-staged file (stage --with-mtp)
        win.start(nodes[r], f"serve-r{r}", [f"{b}/flerp-qwen35moe-stage"] + [f"{k}={v}" for k, v in kr.items()] + [f"rank={r}"], env)
        ev = win.wait_event(nodes[r], f"serve-r{r}", ("loaded", "error"), run["load_timeout_s"])
        win.log(f"stage rank {r}: {[e for e in ev if e.get('event') in ('loaded', 'error')][:1]}")
        win.status = f"serve: rank {r} loaded"
    end = time.time() + 900
    health = None
    while time.time() < end:
        rc, text = on(nodes[0]["mgmt"], f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{run['http_port']}/health", 30)
        health = text.strip()
        if health == "200" or win.rc(nodes[0], "serve-r0") is not None:
            break
        time.sleep(10)
    doc["health"] = health
    if health != "200":
        why.append(f"serve: /health {health}")
    chats = {}
    for c in run["chat"] if health == "200" else []:
        body = json.dumps({"model": "ornith-1.5-397b", "messages": [{"role": "user", "content": c["text"]}], "max_tokens": run["chat_max_tokens"]})
        t0 = time.time()
        rc, text = on(nodes[0]["mgmt"], f"curl -s -w '\\n%{{http_code}}' -X POST -H 'content-type: application/json' "
                                        f"--data {shlex.quote(body)} http://127.0.0.1:{run['http_port']}/v1/chat/completions", 1800)
        resp, _, code = text.rpartition("\n")
        try:
            r = json.loads(resp)
        except ValueError:
            r = {"raw": resp[-500:]}
        ch = (r.get("choices") or [{}])[0]
        chats[c["name"]] = {"status": code.strip(), "seconds": round(time.time() - t0, 1), "text": (ch.get("message") or {}).get("content"),
                            "finish_reason": ch.get("finish_reason"), "usage": r.get("usage")}
        win.log(f"chat {c['name']}: HTTP {code.strip()} in {chats[c['name']]['seconds']} s: {str(chats[c['name']]['text'])[:300]!r}")
        if code.strip() != "200" or not chats[c["name"]]["text"] or ch.get("finish_reason") not in ("stop", "length"):
            why.append(f"chat {c['name']}: HTTP {code.strip()}, finish {ch.get('finish_reason')}")
    doc["chats"] = chats
    for r in range(4):
        win.term(nodes[r], f"serve-r{r}")
    rcs = [win.wait_rc(nodes[r], f"serve-r{r}", 180) for r in range(4)]
    doc["exit"] = rcs
    for r in range(4):
        win.fetch(nodes[r], f"serve-r{r}", RES)
    if any(x != 0 for x in rcs):
        why.append(f"serve: exit statuses {rcs}")
    return why


def cmd_run(cfg, a):
    global RES, PROBE
    if a.launch or a.spec_launch:
        return run_confirm(cfg, a)
    if a.mtp_probe:
        # ORNITH-O3 (gate/ornith-o3-mtp.toml [ornith]): world 4 only, its own prompts and evidence directory
        if a.steps != "w4":
            sys.exit("--mtp-probe runs --steps w4 only")
        pj = MP.load_json(PROBE_PROMPTS)
        with open(os.path.join(ROOT, "gate", "ornith-o3-mtp.toml"), "rb") as f:
            o3 = tomllib.load(f)
        PROBE = {"depth": a.mtp_probe, "names": pj["names"], "ids": pj["ids"], "max_tokens": o3["proxy"]["max_tokens"]}
        RES = PROBE_RES
    log = Log(os.path.join(RES, "run.txt"))
    win = Window(cfg, log, a.card)
    steps = a.steps.split(",")
    doc = {"when": utc(), "source": a.source, "steps": steps, "failures": []}

    def on_signal(sig, frame):
        raise KeyboardInterrupt(f"signal {sig}")
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    why = doc["failures"]
    try:
        local, missing = local_bundle(cfg)
        if missing:
            sys.exit(f"bundle sources missing: {missing}")
        bad, err = gen_flag_check(cfg)
        if err or bad:
            sys.exit(f"flerp-qwen35moe-generate does not take the runner's flags (rebuild build/flerp-p5): {err or bad}")
        o = origin_have(cfg)
        if a.source == "swarm" and ("error" in o or o["held"] != o["pieces"]):
            sys.exit(f"the origin does not offer the manifest: {o}")
        if a.source == "cache":
            bad = {}
            for n in cfg["host"]["nodes"]:
                _, t = on(n["mgmt"], f"stat -c %s {shlex.quote(cache_path(cfg))} 2>/dev/null", 60)
                if t.strip() != str(C.read_header(cfg["model"]["gguf"])["file_bytes"]):
                    bad[n["name"]] = t.strip() or "absent"
            if bad:
                sys.exit(f"--source cache: {cache_path(cfg)} not staged on {bad}; run `ornith_p5.py stage` first")
        nodes = cfg["host"]["nodes"]
        ips = [n["mgmt"] for n in nodes]
        win.start_heartbeat()
        win.comment(f"FLERP-P5 run {win.run_id}: taking holds on {ips} as {cfg['host']['hold_owner']!r}")
        if not win.take_holds(ips):
            why.append("holds refused after every try")
            raise RuntimeError(why[-1])
        ok, facts = win.wait_quiet(nodes, cfg["host"]["hold_drain_s"])
        doc["facts_after_hold"] = facts
        if not ok:
            why.append("serving still up on a held node after the drain")
            raise RuntimeError(why[-1])
        g = cfg["model"]["gguf"]
        for w in (4, 3):
            cuts = MP.py_plan(g, w)["cuts"]
            for r, n in enumerate(nodes_of(cfg, w)):
                f = facts[n["name"]]
                need = MP.py_plan(g, 1, [cuts[r], cuts[r + 1]])["ranks"][0]["device_bytes"] + \
                    (cfg["host"]["arena_gib"] + cfg["host"]["mem_margin_gib"]) * GIB
                f[f"need_w{w}_bytes"] = int(need)
        for n in nodes:
            f = facts[n["name"]]
            need = max(f.get("need_w4_bytes", 0), f.get("need_w3_bytes", 0))
            if f.get("mem_available", 0) < need:
                why.append(f"{n['name']}: MemAvailable {f.get('mem_available', 0) / GIB:.1f} GiB under {need / GIB:.1f}")
        if why:
            raise RuntimeError("memory")
        for n in nodes:
            push_bundle(cfg, n["mgmt"], local)
        log("bundle on every node (md5 checked)")
        if "w4" in steps:
            win.status = "world 4 loading"
            runs, w, (n4, t4) = gen_world(win, 4, a.source, doc, dumps=True)
            why += w
            if len(runs) >= 2:
                why += judge_equal_runs(runs[0], runs[1], "world 4 run 1 vs run 2")
            else:
                why.append(f"world 4: {len(runs)} runs")
            save("run.json", doc)
            win.stop_all()     # free the GPUs before the next world
            bc, w = block_checks(win, n4, t4)
            doc["block_check"] = bc
            why += w
        if "w3" in steps:
            win.status = "world 3 loading"
            runs3, w, _ = gen_world(win, 3, a.source, doc, dumps=False)
            why += w
            if "w4" in doc and runs3 and doc["w4"].get("runs"):
                r4 = [dict(e, logits_sha256=s) for e, s in zip(doc["w4"]["runs"][0], doc["w4"]["sha"][0])]
                why += judge_equal_runs(runs3[0], r4, "world 3 vs world 4")
            save("run.json", doc)
            win.stop_all()
        if "serve" in steps:
            win.status = "serve loading"
            doc["serve"] = {}
            why += serve_step(win, a.source, doc["serve"])
    except (KeyboardInterrupt, RuntimeError, subprocess.SubprocessError, OSError) as e:
        why.append(f"stopped: {e}")
    finally:
        win.close("end of run" if not why else f"failures: {why[:3]}")
        doc["passed"] = not why
        doc["ended"] = utc()
        save("run.json", doc)
        log(f"P5 {'PASS' if doc['passed'] else 'FAIL'}; failures {why[:8]}")
    return 0 if doc["passed"] else 1


def judge_confirm(k, launch, rank_events, names, refs):
    """One launch of the 4-node confirm, judged by ORNITH-O1O2's judge_launch (gate/ornith-o1o2.toml criteria 1-4)
    against run 11121855's ids and sha256 (refs). Returns (record, faults, bound_misses): a bound miss is criterion 4's
    copy < 0.1 ms (a measured result, recorded; the window goes on); every other line is a fault (stop)."""
    import ornith_o1o2 as O
    rec, why = O.judge_launch(k, launch, rank_events, names, refs)
    misses = [w for w in why if "(bound < 0.1)" in w]
    return rec, [w for w in why if w not in misses], misses


def ref_from_launch(rank_events, names, name):
    """The first reply-full, 1-sub-chunk `staged` result of prompt `name` (the reference for a prompt 11121855 did
    not run: later runs and launches must equal it)."""
    i = names.index(name)
    for e in rank_events.get(0, {}).get("events", []):
        if e.get("event") == "staged" and e["prompt"] == i and e.get("reply") == "full" and e.get("subchunks") == 1:
            return {"ids": e["ids"], "logits_sha256": e["logits_sha256"]}
    return None


def mtp_acceptance(probe_events, names, classes, depth):
    """ORNITH-O3 recorded table from the 4-node probe events (gate/ornith_o3_mtp.py tally / schedule)."""
    import ornith_o3_mtp as O3
    acc = {"code": [], "prose": [], "all": []}
    for e in probe_events:
        n = names[e["prompt"]]
        x = {"tally": O3.tally(e["ids"], e["drafts"], depth), "s1": O3.schedule(e["ids"], e["drafts"], 1),
             "s3": O3.schedule(e["ids"], e["drafts"], 3)}
        for cls in (classes[n], "all"):
            acc[cls].append(x)
    table = {}
    for cls, lst in acc.items():
        rows = []
        for k in range(depth):
            n = sum(x["tally"][k]["n"] for x in lst)
            ch = sum(x["tally"][k]["chain"] for x in lst)
            rows.append({"k": k + 1, "n": n, "marginal": sum(x["tally"][k]["marginal"] for x in lst) / n if n else None,
                         "chain": ch / n if n else None})
        for i, r in enumerate(rows):
            r["conditional"] = r["chain"] / rows[i - 1]["chain"] if i and rows[i - 1]["chain"] else r["chain"]
        table[cls] = {"depth": rows, "expected_tokens_per_round": [1 + sum(r["chain"] or 0 for r in rows[:k]) for k in range(1, depth + 1)]}
    return table


def spec_projection(table, decode_ms, mtp_ms):
    """PROJECTED (not measured) speculative tok/s at depth k: E_k tokens a round over one verify forward (taken as one
    decode step, decode_ms) plus k MTP draft steps (mtp_ms each)."""
    out = {}
    for cls, t in table.items():
        out[cls] = {k + 1: round(e / ((decode_ms + (k + 1) * mtp_ms) / 1000.0), 2) for k, e in enumerate(t["expected_tokens_per_round"])}
    return out


def confirm_summary(doc):
    import statistics
    L = [f"# FLERP-P5 Ornith 4-node confirm, run {doc.get('run_id')}", "",
         f"Written {utc()}. Reference: P5 run 11121855 (world 4, cuts 0/15/30/45/60, 12.05 tok/s, 82.96 ms a token). "
         f"Criteria: gate/ornith15-397b.flerp-p5.toml, gate/ornith-o1o2.toml, gate/ornith-o3-mtp.toml [ornith] (unchanged). "
         f"Binary sha256 {doc.get('binary_sha256')}. Hold {doc.get('hold_taken')} -> {doc.get('ended')}.", "",
         f"**{'PASS' if doc.get('passed') else 'FAIL'}** (faults: {len(doc['failures'])}; criterion-4 copy bound misses, recorded: "
         f"{len(doc.get('bound_misses', []))})", ""]
    L += ["## Runs (rank 0: ids / sha256 vs 11121855; records vs launch 0's full:1)", "",
          "| launch | rows | reply:S | prompt | tokens | ids eq | sha eq | TTFT s | decode ms/tok | copy ms/call r1/r2/r3 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for Lr in doc.get("launches", []):
        for r in Lr["runs"]:
            c = r["copy_ms_per_call"]
            cc = "/".join(f"{c.get(k, c.get(str(k), 0)):.3f}" for k in (1, 2, 3))
            L.append(f"| {r['launch']} | {r['rows']} | {r['reply']}:{r['subchunks']} | {r['prompt']} | {r['prompt_tokens']} | {r['equal_ids']} | "
                     f"{r['equal_sha'] if r['reply'] == 'full' else 'n/a'} | {r['ttft_s']:.3f} | {r['decode_ms_per_token']:.2f} | {cc} |")
    runs = [r for Lr in doc.get("launches", []) for r in Lr["runs"]]
    if runs:
        L += ["", "## Medians", ""]
        for mode in ("pageable", "pinned"):
            for reply in ("full", "token"):
                v = [r["decode_ms_per_token"] for r in runs if r["rows"] == mode and r["reply"] == reply and r["subchunks"] == 1]
                if v:
                    L.append(f"- decode {mode} reply {reply} S=1: median {statistics.median(v):.2f} ms a token = "
                             f"{1000 / statistics.median(v):.2f} tok/s (n {len(v)}; P5 82.96 / 12.05)")
            for k in (1, 2, 3):
                v = [r["copy_ms_per_call"].get(k, r["copy_ms_per_call"].get(str(k))) for r in runs if r["rows"] == mode]
                v = [x for x in v if x is not None]
                if v:
                    L.append(f"- copy {mode} rank {k}: median {statistics.median(v):.3f} ms a call (max {max(v):.3f})")
        for S in (1, 4, 16):
            for p in sorted({r["prompt"] for r in runs}):
                v = [r["ttft_s"] for r in runs if r["rows"] == "pinned" and r["reply"] == "full" and r["subchunks"] == S and r["prompt"] == p]
                if v:
                    L.append(f"- TTFT pinned full S={S} {p}: {', '.join(f'{x:.3f}' for x in v)} s")
    if doc.get("mtp"):
        m = doc["mtp"]
        L += ["", "## ORNITH-O3 MTP acceptance (4-node, greedy)", "", f"head: {json.dumps(m.get('head'))[:300]}",
              f"MTP ms a draft step (median): {m.get('mtp_ms')}", "", "| class | k | n | marginal | chain | conditional | E tok/round |", "|---|---|---|---|---|---|---|"]
        for cls, t in m["table"].items():
            for i, r in enumerate(t["depth"]):
                L.append(f"| {cls} | {r['k']} | {r['n']} | {r['marginal']:.3f} | {r['chain']:.3f} | {r['conditional']:.3f} | {t['expected_tokens_per_round'][i]:.3f} |")
        L += ["", f"PROJECTED speculative tok/s (verify of k+1 rows = one decode step at {m.get('proj_decode_ms')} ms, plus k draft steps): "
              f"{json.dumps(m.get('projected_tok_s'))}"]
    if doc["failures"] or doc.get("bound_misses"):
        L += ["", "## Faults and bound misses (first 40)", ""] + [f"- {w}" for w in (doc["failures"] + doc.get("bound_misses", []))[:40]]
    with open(os.path.join(CONFIRM_RES, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def confirm_preflight(cfg, a, log):
    """No hold, no GPU: everything a confirm window needs that can fail before it. Returns [problems]."""
    bad = []
    if a.source != "cache":
        bad.append(f"--source {a.source}: the confirm reads cache (ranks without --rank-origin) only; swarm/file use the USB origin")
    for r in SWARM_RANKS:
        if r not in RANK_ORIGINS:
            bad.append(f"rank {r}: no --rank-origin")
    for r, o in RANK_ORIGINS.items():
        if not os.path.isfile(o["ranges"]):
            bad.append(f"rank {r}: no ranges file {o['ranges']}")
    local, missing = local_bundle(cfg)
    if missing:
        bad.append(f"bundle sources missing: {missing}")
    fb, err = gen_flag_check(cfg)
    if err or fb:
        bad.append(f"generator flags: {err or fb}")
    fb = os.path.realpath(cfg["model"]["gguf"])
    log(f"header file (--model, header bytes only; tensor bytes never come from it): {cfg['model']['gguf']} -> {fb}")
    size = str(C.read_header(cfg["model"]["gguf"])["file_bytes"])
    for r, n in enumerate(nodes_of(cfg, 4)):
        if r in SWARM_RANKS:
            continue
        _, t = on(n["mgmt"], f"f={shlex.quote(cache_path(cfg))}; echo $(readlink -f $f) $(stat -c %s $f 2>/dev/null)", 60)
        parts = t.split()
        if len(parts) != 2 or parts[1] != size or cold_path(parts[0]):
            bad.append(f"rank {r} on {n['name']}: {cache_path(cfg)} not staged ({t.strip()})")
    if a.mtp_probe or a.spec_launch:
        n = nodes_of(cfg, 4)[-1]
        rc, t = on(n["mgmt"], f"cat {os.path.dirname(cache_path(cfg))}/.pod-staging.tsv 2>/dev/null; true", 30)
        log(f"mtp: the head reads {cache_path(cfg)} on {n['name']} (stage --with-mtp --nodes {n['name']} first; place verify covers it)")
    for r, n in enumerate(nodes_of(cfg, 4)):
        if r in SWARM_RANKS and RANK_ORIGINS.get(r, {}).get("host") == n["mgmt"]:
            bad.append(f"rank {r}: its origin is on its own node")
    return bad, local


def run_confirm(cfg, a):
    """`run --launch ROWS=SPECS ... [--mtp-probe D]`: every launch (world 4, graph on, rank 0 --run-specs) and then
    the MTP probe launch, under ONE hold; between launches only the rank units stop (the origins and the hold stay)."""
    global LAUNCH, PROBE, SPEC
    launches = [parse_launch(x) for x in a.launch]
    os.makedirs(CONFIRM_RES, exist_ok=True)
    log = Log(os.path.join(CONFIRM_RES, "run.txt"))
    win = Window(cfg, log, a.card)
    names = cfg["model"]["prompts"]
    refs = p5_refs(MP.load_json(P5_REF_RUN), names)
    doc = {"when": utc(), "run_id": win.run_id, "launches": [], "failures": [], "bound_misses": [], "names": names,
           "swarm_ranks": sorted(SWARM_RANKS), "rank_origins": RANK_ORIGINS, "launch_specs": launches, "mtp_probe": a.mtp_probe,
           "hold": {k: cfg["host"][k] for k in ("hold_owner", "hold_reason", "hold_ttl_s")}}
    why = doc["failures"]
    exe = os.path.join(GEN_BUILD, "out", "flerp-qwen35moe-generate")
    doc["binary_sha256"] = hashlib.sha256(open(exe, "rb").read()).hexdigest()
    log(f"confirm {win.run_id}: launches {launches}, mtp {a.mtp_probe}, swarm ranks {RANK_ORIGINS}, prompts {names}, "
        f"refs from 11121855 for {sorted(refs)}; generator sha256 {doc['binary_sha256']}")

    def on_signal(sig, frame):
        raise KeyboardInterrupt(f"signal {sig}")
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        bad, local = confirm_preflight(cfg, a, log)
        if bad:
            why += bad
            raise RuntimeError(f"preflight: {bad}")
        if a.preflight_only:
            for r, o in sorted(RANK_ORIGINS.items()):
                win.start_origin(r, o)
            win.stop_origins()
            log("preflight only: OK (no hold taken; /cache untouched)")
            win.closed = True
            return 0
        for r, o in sorted(RANK_ORIGINS.items()):
            win.start_origin(r, o)
        nodes = cfg["host"]["nodes"]
        win.start_heartbeat()
        if not win.take_holds([n["mgmt"] for n in nodes]):
            why.append("holds refused after every try")
            raise RuntimeError(why[-1])
        doc["hold_taken"] = utc()
        log(f"MILESTONE hold taken {doc['hold_taken']}")
        if a.after_hold:
            # e.g. the serving pair's own `strixfold-pair.sh down` (it stops only for a holder) and its exit evidence
            p = subprocess.run(["bash", "-c", a.after_hold], capture_output=True, text=True, timeout=900)
            doc["after_hold"] = {"rc": p.returncode, "out": (p.stdout + p.stderr)[-2000:]}
            log(f"after-hold rc {p.returncode}: {(p.stdout + p.stderr).strip()[-600:]}")
        ok, facts = win.wait_quiet(nodes, cfg["host"]["hold_drain_s"])
        if not ok:
            why.append("serving still up on a held node after the drain")
            raise RuntimeError(why[-1])
        g = cfg["model"]["gguf"]
        cuts = MP.py_plan(g, 4)["cuts"]
        for r, n in enumerate(nodes_of(cfg, 4)):
            need = MP.py_plan(g, 1, [cuts[r], cuts[r + 1]])["ranks"][0]["device_bytes"] + (cfg["host"]["arena_gib"] + cfg["host"]["mem_margin_gib"]) * GIB
            if facts[n["name"]].get("mem_available", 0) < need:
                why.append(f"{n['name']}: MemAvailable {facts[n['name']].get('mem_available', 0) / GIB:.1f} GiB under {need / GIB:.1f}")
        if why:
            raise RuntimeError("memory")
        for n in nodes:
            push_bundle(cfg, n["mgmt"], local)
        log("bundle on every node (md5 checked)")
        for k, L in enumerate(launches):
            LAUNCH = L
            win.status = f"launch {k} {L['rows']} loading"
            runs, w, _ = gen_world(win, 4, a.source, doc, dumps=False, key=f"L{k}-", res=CONFIRM_RES)
            ev = doc[f"L{k}-w4"].pop("rank_events")
            if w:
                why += w
                raise RuntimeError(f"launch {k}: {w[:2]}")
            for p in names:
                if p not in refs:
                    refs[p] = ref_from_launch(ev, names, p)
                    log(f"launch {k}: reference for {p} = this launch's full:1 run ({len((refs[p] or {}).get('ids', []))} ids)")
            rec, faults, misses = judge_confirm(k, L, ev, names, refs)
            doc["launches"].append(rec)
            doc["bound_misses"] += misses
            save_confirm(doc)
            dm = [round(r["decode_ms_per_token"], 2) for r in rec["runs"]]
            log(f"MILESTONE launch {k} ({L['rows']} {L['specs']}) RESULT faults {len(faults)} bound misses {len(misses)}; decode ms {dm}")
            if faults:
                why += faults
                raise RuntimeError(f"launch {k}: {faults[:3]}")
            win.stop_all()
        if a.spec_launch:
            # ORNITH-O3 step two (gate/ornith-o3-spec.toml [ornith]): drafts off and on in each launch, judged by
            # gate/ornith_o3_spec.py's criteria functions
            import ornith_o3_spec as OS
            with open(os.path.join(ROOT, "gate", "ornith-o3-spec.toml"), "rb") as f:
                crit = tomllib.load(f)
            prompts = OS.prompt_set(None)
            SPEC = {"names": [p[0] for p in prompts], "ids": {p[0]: p[2] for p in prompts}, "max_tokens": crit["proxy"]["max_tokens"],
                    "context": SPEC_CONTEXT, "costs": crit["proxy"]["draft_costs"]}
            os.makedirs(SPEC_RES, exist_ok=True)
            sdoc = {"launches": [], "failures": [], "_res": SPEC_RES, "binary_sha256": doc["binary_sha256"], "run_id": win.run_id}
            for k, specs in enumerate(a.spec_launch):
                LAUNCH = {"rows": "pinned", "specs": specs}
                win.status = f"spec launch {k} loading"
                runs, w, _ = gen_world(win, 4, a.source, doc, dumps=False, key=f"S{k}-", res=SPEC_RES)
                ev = doc[f"S{k}-w4"].pop("rank_events")
                rec, sw = OS.judge_launch(k, specs, ev, prompts, SPEC["max_tokens"])
                sw += OS.judge_all({"launches": [rec]}, prompts, SPEC["max_tokens"], anchors=False)   # identity, this launch
                sdoc["launches"].append(rec)
                sdoc["failures"] += w + sw
                log(f"MILESTONE spec launch {k} ({specs}) RESULT faults {len(w + sw)}")
                win.stop_all()
                if w or sw:
                    break
            if not sdoc["failures"]:
                sdoc["failures"] += OS.judge_all(sdoc, prompts, SPEC["max_tokens"], anchors=False)   # across launches
            # [ornith] criterion 2: drafts-off bridge / code / rivers equal P5 run 11121855's ids on their length
            for L_ in sdoc["launches"]:
                for r_ in L_["runs"]:
                    ref = (refs.get(r_["prompt"]) or {}).get("ids")
                    if r_["drafts"] == 0 and ref and r_["ids"][:len(ref)] != ref:
                        sdoc["failures"].append(f"spec launch {L_['launch']} {r_['prompt']}: drafts-off ids differ from 11121855's")
            sdoc["recorded"] = OS.recorded(sdoc)
            sdoc["projection"] = OS.project(crit["projection"], sdoc["recorded"])
            with open(os.path.join(SPEC_RES, "results.json"), "w") as f:
                json.dump(sdoc, f, indent=1, default=str)
            OS.summarize(sdoc, title="ORNITH-O3 step two on Ornith-1.5 397B, world 4 on F D A C")
            why += sdoc["failures"][:20]
            SPEC, LAUNCH = None, None
        if a.mtp_probe:
            LAUNCH = None
            pj = MP.load_json(PROBE_PROMPTS)
            with open(os.path.join(ROOT, "gate", "ornith-o3-mtp.toml"), "rb") as f:
                o3 = tomllib.load(f)
            PROBE = {"depth": a.mtp_probe, "names": pj["names"], "ids": pj["ids"], "max_tokens": o3["proxy"]["max_tokens"]}
            win.status = "mtp probe loading"
            runs, w, _ = gen_world(win, 4, a.source, doc, dumps=False, key="mtp-", res=PROBE_RES)
            doc["mtp-w4"].pop("rank_events", None)
            why += w
            if len(runs) >= 2:
                why += judge_equal_runs(runs[0], runs[1], "mtp launch run 1 vs run 2")
            else:
                why.append(f"mtp launch: {len(runs)} runs")
            pe = doc["mtp-w4"].get("mtp_probe", [])
            table = mtp_acceptance([e for e in pe if e.get("session", 0) < len(pj["names"])], pj["names"], pj["classes"], a.mtp_probe)
            ms = sorted(e.get("mtp_ms_per_step", 0) for e in pe) or [0]
            tok = [r["decode_ms_per_token"] for Lr in doc["launches"] for r in Lr["runs"] if r["reply"] == "token" and r["rows"] == "pinned"]
            dec = sorted(tok)[len(tok) // 2] if tok else 82.96
            doc["mtp"] = {"table": table, "head": doc["mtp-w4"].get("mtp_head"), "mtp_ms": ms[len(ms) // 2], "proj_decode_ms": round(dec, 2),
                          "projected_tok_s": spec_projection(table, dec, ms[len(ms) // 2])}
            log(f"MILESTONE mtp RESULT faults {len(w)}; acceptance {json.dumps({c: [round(r['chain'] or 0, 3) for r in t['depth']] for c, t in table.items()})}")
            PROBE = None
    except (KeyboardInterrupt, RuntimeError, subprocess.SubprocessError, OSError, ValueError) as e:
        why.append(f"stopped: {e}")
    finally:
        win.close("end of run" if not why else f"failures: {why[:3]}")
        doc["passed"] = not why
        doc["ended"] = utc()
        save_confirm(doc)
        try:
            confirm_summary(doc)
        except Exception as e:      # the summary never hides the result
            log(f"summary failed: {e}")
        log(f"MILESTONE confirm {'PASS' if doc['passed'] else 'FAIL'}; faults {why[:8]}; bound misses {len(doc['bound_misses'])}")
    return 0 if doc["passed"] else 1


def save_confirm(doc):
    os.makedirs(CONFIRM_RES, exist_ok=True)
    with open(os.path.join(CONFIRM_RES, "run.json"), "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True, default=str)


def cmd_cleanup(cfg, a):
    log = Log(os.path.join(RES, "run.txt"))
    win = Window(cfg, log)
    win.prefix = cfg["host"]["unit_prefix"] + (a.run_id + "-" if a.run_id else "")
    print(json.dumps({"left_units_and_procs": win.stop_all()}))
    return 0


# --- the 35B on control-node's GPU ---------------------------------------------------------------------------------
def cmd_regate35(cfg, a):
    """The code change on the 35B: loopback configs over one load (world 1, 2, 4; even and lopsided cuts; graph on
    and off), then TCP on this host at world 2 and 4 with every rank's bytes from loom-swarm (a test origin on the
    rail address serving the 35B manifest): P1's ids and logits-row hashes in every case."""
    log = Log(os.path.join(RES, "regate35.txt"))
    os.makedirs(SCRATCH, exist_ok=True)
    g = cfg["model"]["gguf35"]
    gen = os.path.join(ROOT, "build", "flerp-p5", "out", "flerp-qwen35moe-generate")
    helper = os.path.join(ROOT, cfg["swarm"]["helper"])
    manifest = os.path.join(ROOT, "build", "q35.swarm.json")
    while MP.load1() > 10.0:
        log(f"load {MP.load1()} > 10: waiting")
        time.sleep(60)
    ids = MP.load_json("gate/results/qwen36-35b-a3b.flerp-moe/prompt-ids.json")
    names = list(ids)
    p1 = MP.load_json("gate/results/qwen36-35b-a3b.flerp-moe-placed/equal.json")["traced"]
    refs = {p: p1[p] for p in names}
    id_files = []
    for p in names:
        f = os.path.join(SCRATCH, f"{p}.ids")
        with open(f, "w") as fo:
            fo.write(" ".join(str(i) for i in ids[p]))
        id_files += ["--ids-file", f]
    polite = ["ionice", "-c3", "nice", "-n", "19"]
    rocm = os.environ.get("FLERP_ROCM_ROOT", "/data/tools/therock-10.1.0a20260823") + "/lib"
    os.environ["LD_LIBRARY_PATH"] = rocm + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")
    doc, why = {"when": utc()}, []

    def run(label, argv, timeout=3600):
        out = os.path.join(SCRATCH, f"{label}.jsonl")
        with open(out, "w") as fo, open(out + ".err", "w") as fe:
            p = subprocess.run(polite + argv, stdout=fo, stderr=fe, timeout=timeout)
        ev = [json.loads(x) for x in open(out) if x.startswith("{")]
        return p.returncode, ev

    # 1. loopback: one load (the O_DIRECT single reader), every configuration over it
    configs = ["1:even:on", "1:even:off", "2:even:on", "2:0,7,40:off", "4:even:on", "4:even:off"]
    rc, ev = run("loopback", [gen, "--staged", "loopback", "--model", g, "--context", "4096", "--max-tokens", "128", "--configs", ";".join(configs),
                              "--single-reader", "--pause-ms", "100"] + id_files)
    rows, w = MS.judge_matrix(ev, configs, names, refs, "loopback")
    doc["loopback"] = {"rc": rc, "matrix": rows, "loaded": next((e for e in ev if e.get("event") == "loaded"), None)}
    why += w + ([f"loopback exited {rc}"] if rc else [])
    log(f"loopback rc {rc}: " + ", ".join(f"{m['config']} {m['prompts_equal']}/{m['prompts']}" for m in rows))
    # 2. TCP on this host, bytes from loom-swarm (a test origin on the rail address, nice'd: it reads the ark disk)
    origin = "10.20.0.31:7159"
    srv = subprocess.Popen(polite + [helper, "serve", "--listen", origin, "--manifest", manifest, "--file", g],
                           stdout=subprocess.DEVNULL, stderr=open(os.path.join(SCRATCH, "origin.err"), "w"))
    try:
        time.sleep(2)
        for k, (world, cuts) in enumerate(((2, "even"), (4, "even"))):
            procs, outs = [], []
            port = 39910 + 10 * k
            for r in range(world):
                out = os.path.join(SCRATCH, f"tcp{world}-r{r}.jsonl")
                argv = polite + [gen, "--staged", "tcp", "--model", g, "--context", "4096", "--max-tokens", "128", "--world", str(world), "--rank", str(r),
                                 "--cuts", cuts, "--port", str(port), "--host", "127.0.0.1", "--epoch", str(31 + k), "--graph-modes", "on,off",
                                 "--swarm-helper", helper, "--swarm-manifest", manifest, "--swarm-from", origin, "--swarm-conns", "4"] + \
                    (id_files if r == 0 else [])
                procs.append(subprocess.Popen(argv, stdout=open(out, "w"), stderr=open(out + ".err", "w")))
                outs.append(out)
                end = time.time() + 1800
                while time.time() < end and procs[-1].poll() is None and '"event":"loaded"' not in open(out).read():
                    time.sleep(2)
            rcs = [p.wait(timeout=3600) for p in procs]
            evs = [[json.loads(x) for x in open(o) if x.startswith("{")] for o in outs]
            cfgs = [f"{world}:{cuts}:{gm}" for gm in ("on", "off")]
            rows, w = MS.judge_matrix(evs[0], cfgs, names, refs, "tcp")
            loads = [next((e.get("placed") for e in ev if e.get("event") == "loaded"), None) for ev in evs]
            doc[f"tcp{world}"] = {"rc": rcs, "matrix": rows, "loaded": loads}
            why += w + [f"tcp world {world} rank {r} exited {x}" for r, x in enumerate(rcs) if x]
            why += [f"tcp world {world} rank {r}: not loaded from loom-swarm" for r, pl in enumerate(loads) if not (pl or {}).get("swarm")]
            for r, pl in enumerate(loads):
                pl = pl or {}
                log(f"tcp world {world} rank {r}: {pl.get('tensor_bytes')} bytes in {pl.get('fill_s')} s = {pl.get('read_gb_per_s')} GB/s "
                    f"from loom-swarm into the GPU (frames {pl.get('swarm_bytes')}, helper {pl.get('swarm_stats')})")
            log(f"tcp world {world} rc {rcs}: " + ", ".join(f"{m['config']} {m['prompts_equal']}/{m['prompts']}" for m in rows))
    finally:
        srv.terminate()
        srv.wait(timeout=30)
    doc["failures"], doc["passed"] = why, not why
    save("regate35.json", doc)
    log(f"regate35 {'PASS' if not why else 'FAIL'}; failures {why[:6]}")
    return 0 if not why else 1


RTCFG = {"name": "none", "stage": {}, "env": {}}
CONFIG_RT = {}


def set_config(a):
    """The named configuration (flerplib/configs.py) for this run: serve and stage use it by default."""
    global RTCFG, CONFIG_RT
    RTCFG = CF.override(CF.load(a.config), a.set)
    CONFIG_RT = {"drafts": CF.drafts(RTCFG)}
    if RTCFG["stage"].get("with_mtp") and CONFIG_RT["drafts"]:
        a.with_mtp = True
    if "--config" in sys.argv[1:] and RTCFG["stage"].get("rows") and not a.rows:
        PASS["rows"] = RTCFG["stage"]["rows"]
        if not a.reply and RTCFG["stage"].get("reply"):
            PASS["reply"] = RTCFG["stage"]["reply"]
    print(f"config {RTCFG['name']}: stage {RTCFG['stage']} env {RTCFG['env']}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", nargs="?", choices=["run", "regate35", "cleanup", "stage"])
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--card")
    ap.add_argument("--source", choices=["swarm", "file", "cache"], default="swarm")
    ap.add_argument("--dry-run", action="store_true", help="stage: print the plan only")
    ap.add_argument("--nodes", default="", help="stage: only these nodes (F,D,A,C)")
    ap.add_argument("--until", default="+3d", help="stage: how long the placement window keeps the bytes")
    ap.add_argument("--steps", default="w4,w3,serve")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--with-mtp", action="store_true", help="stage: also the MTP block and token_embd on world 4's last node (ORNITH-O3)")
    ap.add_argument("--mtp-probe", type=int, default=0, help="run: world 4's last rank drafts with the MTP head, DEPTH deep (ORNITH-O3)")
    ap.add_argument("--spec-launch", action="append", default=[], help="run: rank 0's --run-specs with drafts (G:R:S:K,..); "
                    "the last rank drafts with the MTP head (ORNITH-O3 step two, gate/ornith-o3-spec.toml [ornith]); after any --launch")
    ap.add_argument("--gen-build", default="", help="the generator's build directory (default build/flerp-p5)")
    ap.add_argument("--launch", action="append", default=[], help="run: ROWS=SPECS (pinned|pageable = rank 0's --run-specs); "
                    "several launches, then --mtp-probe's, under one hold (evidence gate/results/ornith-p5-confirm/)")
    ap.add_argument("--rows", choices=["pinned", "pageable"], help="every rank's --rows (without --launch)")
    ap.add_argument("--reply", choices=["full", "token"], help="rank 0's --reply (without --run-specs)")
    ap.add_argument("--run-specs", default="", help="rank 0's --run-specs (without --launch)")
    ap.add_argument("--prompts", default="", help="prompt names from [model] prompt_ids (default [model] prompts)")
    ap.add_argument("--rank-origin", action="append", default=[], help="R=LISTEN@HOST:FILE:RANGES (world-4 rank R streams the FULL "
                    "manifest from a `loom-swarm.place serve --ranges` of a PLACE-staged file this runner starts on HOST)")
    ap.add_argument("--hold-owner", default="")
    ap.add_argument("--hold-reason", default="")
    ap.add_argument("--hold-ttl", type=int, default=0)
    ap.add_argument("--after-hold", default="", help="run --launch: a shell command run on control-node right after the hold is taken")
    ap.add_argument("--before-release", default="", help="run: a shell command run on control-node after the units stop, before the holds are released")
    ap.add_argument("--config", default="ornith397.serve", help="named runtime configuration (configs/NAME.toml; 'none' = the old "
                    "behaviour): applied to `serve` and `stage` by default, to the generator launches only when given explicitly")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config key (drafts, reply, rows, draft_head, with_mtp, env.NAME), e.g. --set drafts=0")
    ap.add_argument("--preflight-only", action="store_true", help="run --launch: every no-hold check, origins started and stopped")
    a = ap.parse_args()
    PASS.update(rows=a.rows, reply=a.reply, run_specs=a.run_specs or None)
    global BEFORE_RELEASE
    BEFORE_RELEASE = a.before_release
    for x in a.rank_origin:
        r, o = parse_rank_origin(x)
        RANK_ORIGINS[r] = o
        SWARM_RANKS.add(r)
    if a.gen_build:
        global GEN_BUILD
        GEN_BUILD = os.path.abspath(a.gen_build)
    cfg = load_cfg()
    set_config(a)
    for k, v in (("hold_owner", a.hold_owner), ("hold_reason", a.hold_reason), ("hold_ttl_s", a.hold_ttl)):
        if v:
            cfg["host"][k] = v
    if a.prompts:
        cfg["model"]["prompts"] = a.prompts.split(",")
    if a.cmd == "stage" and cold_path(cfg["swarm"]["origin"]):
        sys.exit("cold origin")
    if a.plan:
        return cmd_plan(cfg, a)
    if a.cmd is None:
        ap.error("--plan or a command")
    return {"run": cmd_run, "regate35": cmd_regate35, "cleanup": cmd_cleanup, "stage": cmd_stage}[a.cmd](cfg, a)


if __name__ == "__main__":
    sys.exit(main())
