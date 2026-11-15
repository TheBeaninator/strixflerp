#!/usr/bin/env python3
"""Ornith-1.5 397B as a window engine: the `cmd` of a WINJOB win_run (runs on control-node, inside the window).

    engine.py up   [--serve-config ornith397.serve] [--set KEY=VAL ...] [--context N] [--dry-run]
    engine.py down

`up` is what win_run's cmd runs. WINDOW_LETTERS / WINDOW_IPS / WINDOW_RAILS / KIT_DIR come from win_run; rank r runs on the
r-th window letter (F D A C: world 4, cuts 0/15/30/45/60). It
  1. refuses any letter outside F D A C (B serves and hung on 2026-11-13) and any node whose serving processes are still up;
  2. checks the PLACE placements (read only): ranks 0-1 on control-node (/data/models/placed/ornith-w4-rR, streamed over
     the rail by a loom-swarm origin this script starts), ranks 2-3 on the nodes' /cache/flerp-p5 (+ MTP block and
     token_embd on rank 3); it reads no weight byte from /data/ark, /mnt/ark*, /mnt/nv* (the 11 MB GGUF header is the one ark read, as in P5);
  3. starts the origins, rank 0 (the OpenAI front end, serve.flerp.http) and ranks 1-3 (flerp-qwen35moe-stage) with the
     settings of the named config (configs/ornith397.serve.toml: drafts per that config (1 today, card 3f6d20b703),
     token reply path, grouped F16 prefill);
  4. waits for /health, writes /data/pod/winjob/ornith-engine/endpoint.json (pod-engine serve ornith397_flerp reads the
     url through the registry's WORKERS entry), and keeps serving until the stop file, SIGTERM or --max-min;
  5. stops every rank (SIGTERM), the origins and the endpoint file on every exit. win_run then closes the window and
     the keepers bring the StrixFold pair and strixite back.
--dry-run prints every command and touches nothing."""
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import flerplib.configs as CF  # noqa: E402

STATE = "/data/pod/winjob/ornith-engine"
ENDPOINT = f"{STATE}/endpoint.json"
STOP = f"{STATE}/stop"
PLACE = "/data/models/.swarm/bin/loom-swarm.place"
GGUF = "Ornith-1.5-397B-Q4_K_M.gguf"
# The 11 MB GGUF header (tokenizer, tensor table) is read from the ark copy exactly as P5 and the O3 window plan do; every
# weight byte comes from PLACE (control placements over the rail, node /cache).
HEADER = f"/data/ark/models/ornith-1.5-397b/{GGUF}"
CTRL_RAIL = "10.20.0.31"
ALLOWED = ["F", "D", "A", "C"]
MANIFESTS = "/data/pod/plans/ornith-engine/manifests"   # ornith.swarm.json + ornith.rRof4.swarm.json (copies of the swarm task's)
NODE_STAGED = f"/cache/flerp-p5/{GGUF}.staged"
ORIGIN = {0: {"port": 7172, "file": f"/data/models/placed/ornith-w4-r0/{GGUF}.staged", "ranges": "/data/models/.place/ranges/ornith-w4-r0-control.ranges", "id": "ornith-w4-r0-control"},
          1: {"port": 7173, "file": f"/data/models/placed/ornith-w4-r1/{GGUF}.staged", "ranges": "/data/models/.place/ranges/ornith-w4-r1-control.ranges", "id": "ornith-w4-r1-control"}}
PORT, HTTP_PORT, EPOCH = 39810, 39880, 61
MODEL_ID = "ornith-1.5-397b"
SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]


class Run:
    def __init__(self, dry):
        self.dry, self.cmds = dry, []

    def sh(self, argv, check=True, timeout=300):
        text = " ".join(shlex.quote(x) for x in argv)
        self.cmds.append(text)
        print("+", text, flush=True)
        if self.dry:
            return 0, ""
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        if check and p.returncode:
            raise RuntimeError(f"{text}: rc {p.returncode}: {(p.stderr or p.stdout)[-300:]}")
        return p.returncode, p.stdout

    def node(self, ip, script, timeout=300):
        return self.sh(SSH + [f"pod@{ip}", script], timeout=timeout)


def plan(a, env):
    """-> dict of everything `up` will run, from the window grant and the named config. Pure: no I/O."""
    L, I, Rl = env["WINDOW_LETTERS"].split(), env["WINDOW_IPS"].split(), env["WINDOW_RAILS"].split()
    if sorted(L) != sorted(ALLOWED) or len(I) != 4 or len(Rl) != 4:
        raise SystemExit(f"window letters {L}: the engine runs on exactly F D A C (B serves and hung on 2026-11-13; the rank order is the PLACE plan)")
    letters = list(ALLOWED)   # rank r runs on letters[r], whatever order the grant lists them in
    ips = [dict(zip(L, I))[x] for x in letters]
    rails = [dict(zip(L, Rl))[x] for x in letters]
    cfg = CF.override(CF.load(a.serve_config), a.set)
    nd = CF.drafts(cfg)
    kit = env["KIT_DIR"]
    peers = ",".join(rails) + f":{PORT}"
    kv = {"context": a.context, "world": 4, "placement": "even", "epoch": EPOCH, "peers": peers, "single_reader": "on", "pause_ms": 0,
          "setup_deadline_ms": 3600000, "deadline_ms": 120000}
    ranks = []
    for r in range(4):
        k = dict(kv, weights=HEADER)
        if r < 2:   # streamed over the rail from a rank origin on control-node (header only from --model; bytes from the origin)
            k.update(swarm_helper=f"{kit}/bin/loom-swarm-stream", swarm_manifest=f"{kit}/manifests/ornith.r{r}of4.swarm.json",
                     swarm_from=f"{CTRL_RAIL}:{ORIGIN[r]['port']}", swarm_conns=4)
        else:
            k["data_file"] = NODE_STAGED
        if nd and r == 0:
            k["drafts"] = nd   # rank 0's key; the last stage takes draft_head instead
        if nd and r == 3:
            k.update(draft_head="on", mtp_data_file=NODE_STAGED)
        ranks.append({"rank": r, "letter": letters[r], "ip": ips[r], "keys": k})
    front = [f"bin/flerp-serve", "--host", ips[0], "--port", str(HTTP_PORT), "--model-id", MODEL_ID,
             "--generation-config", f"{kit}/generation_config.json"] + (["--delivery", "tokens"] if cfg["stage"].get("reply") == "token" else []) + (["--drafts", str(nd)] if nd else []) + \
            [f"executor.{k}={v}" for k, v in ranks[0]["keys"].items()] + [f"text.weights={ranks[0]['keys']['weights']}", f"text.context={a.context}"]
    ranks[0]["argv"] = front
    for r in (1, 2, 3):
        ranks[r]["argv"] = ["bin/flerp-qwen35moe-stage"] + [f"{k}={v}" for k, v in ranks[r]["keys"].items()] + [f"rank={r}"]
    envs = [f"{k}={v}" for k, v in sorted(cfg["env"].items())]
    return {"config": cfg["name"], "drafts": nd, "env": envs, "ranks": ranks, "url": f"http://{ips[0]}:{HTTP_PORT}", "kit": kit}


def up(a):
    env = {k: os.environ.get(k, d) for k, d in (("WINDOW_LETTERS", "F D A C"), ("WINDOW_IPS", "10.10.0.50 10.10.0.53 10.10.0.54 10.10.0.52"),
                                                ("WINDOW_RAILS", "10.20.0.50 10.20.0.53 10.20.0.54 10.20.0.52"), ("KIT_DIR", "/home/pod/kits/<kit>"))}
    if not a.dry_run and "WINDOW_LETTERS" not in os.environ:
        raise SystemExit("not inside a win_run window (no WINDOW_LETTERS); use --dry-run to see the plan")
    P = plan(a, env)
    R = Run(a.dry_run)
    print(f"engine up: config {P['config']} drafts {P['drafts']} env {P['env']} url {P['url']}", flush=True)
    stopped = []

    def down(*_):
        for r in reversed(P["ranks"]):
            R.node(r["ip"], f"bash {P['kit']}/bin/nodectl.sh down r{r['rank']}")
        for r in (0, 1):
            R.sh(["systemctl", "--user", "stop", f"ornith-origin-r{r}"], check=False)
        if not a.dry_run and os.path.exists(ENDPOINT):
            os.remove(ENDPOINT)
        stopped.append(1)

    # 1. nothing may be serving on the window nodes (the window drained them; a survivor is a stop)
    for r in P["ranks"]:
        rc, out = R.node(r["ip"], f"bash {P['kit']}/bin/nodectl.sh serving")
        if out.strip():
            raise SystemExit(f"{r['letter']} still has serving processes, refusing to start a rank there:\n{out}")
    # 2. the PLACE placements, read only (verify); the engine never copies weights itself
    for rid in (ORIGIN[0]["id"], ORIGIN[1]["id"], "ornith-eng-A", "ornith-eng-C"):
        R.sh([PLACE, "place", "verify", "--id", rid])
    for r in P["ranks"][2:]:
        R.node(r["ip"], f"test -s {NODE_STAGED} && echo staged-ok")
    # 3. origins, then ranks 0..3
    if not a.dry_run:
        os.makedirs(STATE, exist_ok=True)
        with open(os.path.join(STATE, "plan.json"), "w") as f:
            json.dump(P, f, indent=1)
    try:
        for r in (0, 1):
            o = ORIGIN[r]
            R.sh(["systemd-run", "--user", "--unit", f"ornith-origin-r{r}", "--collect", "--", PLACE, "serve", "--listen", f"{CTRL_RAIL}:{o['port']}",
                  "--manifest", f"{MANIFESTS}/ornith.swarm.json", "--file", o["file"], "--ranges", o["ranges"]])
        for r in P["ranks"]:
            R.node(r["ip"], f"bash {P['kit']}/bin/nodectl.sh up r{r['rank']} " + " ".join(shlex.quote(e) for e in P["env"]) + " -- " +
                   " ".join(shlex.quote(x) for x in r["argv"]))
        if a.dry_run:
            print("dry run: would wait for /health, write", ENDPOINT, "and serve until", STOP, "or --max-min", a.max_min)
            return 0
        signal.signal(signal.SIGTERM, down)
        end = time.time() + a.load_timeout_s
        while time.time() < end:
            rc, out = R.sh(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", f"{P['url']}/health"], check=False, timeout=20)
            if out.strip() == "200":
                break
            time.sleep(10)
        else:
            raise RuntimeError("no /health 200 before the load timeout")
        with open(ENDPOINT, "w") as f:
            json.dump({"url": P["url"], "v1": P["url"] + "/v1", "model": MODEL_ID, "config": P["config"], "since": time.time(),
                       "job": os.environ.get("WJ_JOB", ""), "letters": ALLOWED}, f)
        print(f"engine up: {P['url']} (stop file {STOP})", flush=True)
        t_end = time.time() + a.max_min * 60
        while not stopped and not os.path.exists(STOP) and time.time() < t_end:
            time.sleep(5)
    finally:
        if not stopped and not a.dry_run:
            down()
        elif a.dry_run:
            down()
    if os.path.exists(STOP) and not a.dry_run:
        os.remove(STOP)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["up", "down"])
    ap.add_argument("--serve-config", default="ornith397.serve")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--max-min", type=int, default=120, help="serve at most this long, then close (win_run's `for` is the hard bound)")
    ap.add_argument("--load-timeout-s", type=int, default=5400)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cmd == "down":
        open(STOP, "w").close()
        return 0
    return up(a)


if __name__ == "__main__":
    sys.exit(main())
