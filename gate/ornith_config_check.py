#!/usr/bin/env python3
"""Defaults identity check for the Ornith serving configuration (configs/ornith397.serve.toml), on the 35B proxy:
world 4 over loopback TCP, four processes on control-node's one GPU (front end rank 0 + flerp-qwen35moe-stage ranks 1-3),
one load an arm, greedy chats. Arms (same build, same model):
  default   the named config as the launcher applies it (flerplib.configs, no flags)
  explicit  the same settings spelled out by hand: --delivery tokens --drafts 1, executor.drafts=3, draft_head=on on rank 3, FLERP_MOE_GROUPED=1
  nodrafts  the config with --set drafts=0 (grouped prefill on, drafts off): the speculative arms must equal it (lossless)
  old       --config none: drafts off, FLERP_MOE_GROUPED unset (the per-token prompt path)
Pass: default == explicit, token for token, on every prompt; default == nodrafts (speculation is lossless); every process
exits 0. default vs old is recorded, not judged: grouped F16 prefill is not bit-identical to the per-token path (its floor
is the KL bound of the prefill area).
Run on control-node through tools/gpu-queue.sh (gate/ornith_config_check_run.sh); evidence in gate/results/ornith-config-check/."""
import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import flerplib.configs as CF  # noqa: E402
import flerplib.champion as CH  # noqa: E402

RES = os.path.join(ROOT, "gate", "results", "ornith-config-check")
MODEL = "/data/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-Q4_K_M.gguf"
PROMPTS = ["Name three rivers in Europe and one sentence about each.",
           "Write a Python function that returns the n-th Fibonacci number, then explain it briefly.",
           "What is the capital of Australia, and why is it not Sydney?"]
MAX_TOKENS = 96


def die_with_parent():
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)


def events(path):
    out = []
    try:
        for line in open(path, errors="replace"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    except OSError:
        pass
    return out


def wait_event(path, proc, names, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        ev = [e for e in events(path) if e.get("event") in names]
        if ev or proc.poll() is not None:
            return ev
        time.sleep(1)
    return []


def chat(port, text):
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": text}], "max_tokens": MAX_TOKENS, "flerp_trace": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        return {"text": None, "ids": None, "error": f"HTTP {e.code}: {e.read()[:300]!r}"}
    ch = d["choices"][0]
    return {"text": ch["message"].get("content"), "ids": (d.get("flerp") or {}).get("token_ids"), "finish": ch.get("finish_reason"), "usage": d.get("usage")}


def arm(name, a, kv, front_extra, env_extra, rank3_extra, port):
    out = os.path.join(RES, name)
    os.makedirs(out, exist_ok=True)
    rocm = a.rocm or tomllib.load(open(os.path.join(ROOT, "toolchains.toml"), "rb"))["therock-10.1"]["rocm_root"]
    base_env = {k: v for k, v in os.environ.items() if k != "FLERP_MOE_GROUPED"}   # an arm sets it only through its own env
    env = dict(base_env, LD_LIBRARY_PATH=f"{rocm}/lib:{rocm}/lib/llvm/lib", **env_extra)
    gen = os.path.join(out, "generation_config.json")
    json.dump({"temperature": 0}, open(gen, "w"))
    base = {"weights": MODEL, "context": 4096, "world": 4, "placement": "even", "epoch": 7, "peers": f"127.0.0.1:{port + 20}",
            "single_reader": "on", "pause_ms": 0, "setup_deadline_ms": 600000, "deadline_ms": 120000}
    base.update(kv)
    procs, logs = [], []

    def start(label, argv, extra=()):
        lp = os.path.join(out, f"{label}.jsonl")
        f = open(lp, "wb")
        p = subprocess.Popen([os.path.join(a.build, argv[0])] + argv[1:], stdout=f, stderr=open(os.path.join(out, f"{label}.err"), "wb"),
                             env=env, preexec_fn=die_with_parent)
        procs.append(p)
        logs.append(lp)
        return p, lp

    result = {"arm": name, "kv": base, "front": front_extra, "env": env_extra, "rank3": rank3_extra}
    front = ["strixflerp-qwen36-35b-a3b.flerp-serve-stage", "--host", "127.0.0.1", "--port", str(port), "--model-id", "m",
             "--generation-config", gen] + front_extra + [f"executor.{k}={v}" for k, v in base.items()] + \
            [f"text.weights={MODEL}", "text.context=4096"]
    t0 = time.time()
    p0, l0 = start("r0", front)
    wait_event(l0, p0, ("waiting", "error"), 900)
    for r in (1, 2, 3):
        kr = {k: v for k, v in base.items() if k != "drafts"}   # drafts is rank 0's key
        kr.update(rank3_extra if r == 3 else {})
        p, lp = start(f"r{r}", ["flerp-qwen35moe-stage"] + [f"{k}={v}" for k, v in kr.items()] + [f"rank={r}"])
        ev = wait_event(lp, p, ("loaded", "error"), 900)
        if not ev or ev[0].get("event") == "error":
            result["error"] = f"rank {r}: {ev}"
    up = False
    while time.time() - t0 < 1800 and not result.get("error"):
        try:
            up = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3).status == 200
        except Exception:
            up = False
        if up or p0.poll() is not None:
            break
        time.sleep(2)
    result["load_s"] = round(time.time() - t0, 1)
    result["chats"] = []
    if up:
        for t in PROMPTS:
            t1 = time.time()
            c = chat(port, t)
            c["seconds"] = round(time.time() - t1, 2)
            result["chats"].append(c)
    for p in procs:
        p.terminate()
    result["exit"] = []
    for p in procs:
        try:
            result["exit"].append(p.wait(120))
        except subprocess.TimeoutExpired:
            p.kill()
            result["exit"].append("killed")
    json.dump(result, open(os.path.join(out, "result.json"), "w"), indent=1)
    return result


def main():
    global a
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=os.path.join(ROOT, "build", "flerp-p5-serve", "out"))
    ap.add_argument("--rocm", default="")
    ap.add_argument("--config", default="ornith397.serve")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override for the default arm (a probe, not the gate)")
    ap.add_argument("--arms", default="default,explicit,nodrafts,old")
    ap.add_argument("--port", type=int, default=39610)
    ap.add_argument("--step", type=int, default=100, help="port distance between arms")
    a = ap.parse_args()
    if not os.path.isfile(MODEL) or not MODEL.startswith("/data/models/"):
        sys.exit(f"{MODEL} is not there")
    os.makedirs(RES, exist_ok=True)
    mtp = {"draft_head": "on", "mtp_data_file": MODEL}

    def from_cfg(cfg):
        n = CF.drafts(cfg)
        kv = {"drafts": n} if n else {}
        fx = (["--delivery", "tokens"] if cfg["stage"].get("reply") == "token" else []) + (["--drafts", str(n)] if n else [])
        return kv, fx, dict(cfg["env"]), (mtp if n else {})

    specs = {
        "default": lambda: from_cfg(CF.override(CF.load(a.config), a.set)),
        "explicit": lambda: ({"drafts": 1}, ["--delivery", "tokens", "--drafts", "1"], {"FLERP_MOE_GROUPED": "1"}, mtp),
        "nodrafts": lambda: from_cfg(CF.override(CF.load(a.config), ["drafts=0"])),
        "old": lambda: from_cfg(CF.load("none")),
    }
    res = {}
    for i, n in enumerate(a.arms.split(",")):
        kv, fx, env, r3 = specs[n]()
        print(f"arm {n}: executor {kv} front {fx} env {env}", flush=True)
        res[n] = arm(n, a, kv, fx, env, r3, a.port + a.step * i)
        print(f"arm {n}: load {res[n]['load_s']} s, exit {res[n]['exit']}, error {res[n].get('error')}", flush=True)
    why = []
    toks = lambda n: [json.dumps(c.get("ids")) + "|" + str(c["text"]) for c in res[n]["chats"]]
    for n, r in res.items():
        if r.get("error") or len(r["chats"]) != len(PROMPTS) or any(x != 0 for x in r["exit"]) or any(c.get("error") or not c.get("ids") for c in r["chats"]):
            why.append(f"{n}: error {r.get('error')}, chats {len(r['chats'])}, exits {r['exit']}")
    if not why:
        if "explicit" in res and toks("default") != toks("explicit"):
            why.append("default != explicit")
        if "nodrafts" in res and toks("default") != toks("nodrafts"):
            why.append("default != nodrafts (speculation is not lossless here)")
    rec = {n: [sum(1 for x, y in zip(toks("default"), toks(n)) if x == y), len(PROMPTS)] for n in res if "default" in res and res[n]["chats"]}
    doc = {"config": a.config, "model": MODEL, "pass": not why, "why": why, "default_equal_count_vs": rec,
           "seconds": {n: [c["seconds"] for c in r["chats"]] for n, r in res.items()},
           "usage": {n: [c.get("usage") for c in r["chats"]] for n, r in res.items()}}
    json.dump(doc, open(os.path.join(RES, "summary.json"), "w"), indent=1)
    with open(os.path.join(RES, "summary.md"), "w") as f:
        f.write(f"# Ornith serving defaults: identity on the 35B proxy (config {a.config})\n\n"
                f"{'PASS' if not why else 'FAIL'}: {why}\n\nPrompts equal to the default arm: {json.dumps(rec)}\n\n"
                f"Seconds a chat ({MAX_TOKENS} tokens): {json.dumps(doc['seconds'])}\n")
    print(json.dumps(doc, indent=1))
    return 0 if not why else 1


if __name__ == "__main__":
    sys.exit(main())
