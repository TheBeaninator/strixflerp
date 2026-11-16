#!/usr/bin/env python3
"""ORNITH-SUBCHUNK (pod card 6abf28258c; criteria gate/ornith-subchunk.toml, committed before any GPU run): the 1,024-row
sub-chunk floor, the adaptive count and the in-flight window, on the 35B at world 4 over TCP on control-node's one GPU.

Run ON control-node in a clean checkout of the card branch (tools/card-branch.sh <card-id> on the laptop, tools/pod-push.sh --card <card-id>,
then on the pod: git clone /data/projects/strixflerp.git -b card/<card-id> /data/projects/<card-id>; docs/POD-GIT.md):

    python3 gate/ornith_subchunk.py build    # CPU: flerp assemble qwen36-35b-a3b.flerp-serve-stage + build.sh (nice/ionice)
    python3 gate/ornith_subchunk.py gen      # GPU: the generator, four tcp ranks, every spec in one load
    python3 gate/ornith_subchunk.py serve    # GPU: the serve front end + three stage processes, one launch an arm
    python3 gate/ornith_subchunk.py judge    # no GPU: verdict.json from the saved events

GPU phases go through tools/gpu-queue.sh behind gate/loadwait.sh (gate/ornith_subchunk_run.sh). Evidence:
gate/results/ornith-subchunk/.
"""
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
RES = os.path.join(ROOT, "gate", "results", "ornith-subchunk")
RECIPE = "qwen36-35b-a3b.flerp-serve-stage"
BUILD = os.path.join(ROOT, "build", RECIPE)
MODEL = "/data/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-Q4_K_M.gguf"
RECORDS = os.path.join(ROOT, "gate", "results", "ornith15-397b.p4", "long.ids")
P5_IDS = os.path.join(ROOT, "gate", "results", "qwen36-35b-a3b.flerp-moe", "prompt-ids.json")
ARM_G = {"FLERP_MOE_GROUPED": "1", "FLERP_DENSE_GROUPED": "1", "FLERP_ATTN_GQA": "1", "FLERP_MAX_PROMPT_BATCH": "16384"}
GEN_CONTEXT, GEN_TOKENS, SERVE_CONTEXT, SERVE_TOKENS = 33024, 32, 32768, 96
SPECS = [("ref", "on:full:1:0:0"), ("auto", "on:full:auto:0:8"), ("s16", "on:full:16:0:8"), ("tight", "on:full:auto:0:2"),
         ("tok", "on:token:1:0:0"), ("d3", "on:token:auto:3:8")]
SHORT = ["Name three rivers in Europe and one sentence about each.",
         "Write a Python function that returns the n-th Fibonacci number, then explain it briefly.",
         "What is the capital of Australia, and why is it not Sydney?"]
SERVE_CHARS = 72000
PORT = 41510
POLITE = ["ionice", "-c3", "nice", "-n", "19"]


def die_with_parent():
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)


def rocm_env(extra=None):
    rocm = tomllib.load(open(os.path.join(ROOT, "toolchains.toml"), "rb"))["therock-10.1"]["rocm_root"]
    env = {k: v for k, v in os.environ.items() if not k.startswith("FLERP_")}
    env["LD_LIBRARY_PATH"] = f"{rocm}/lib:{rocm}/lib/llvm/lib"
    env.update(extra or {})
    return env


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


# --- build ---------------------------------------------------------------------------------------------------------
def cmd_build(_a):
    os.makedirs(RES, exist_ok=True)
    log = os.path.join(RES, "build.txt")
    with open(log, "w") as f:
        f.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} assemble + build {RECIPE} in {ROOT}\n")
    rc = subprocess.call([sys.executable, "flerp", "assemble", RECIPE], cwd=ROOT, stdout=open(log, "a"), stderr=subprocess.STDOUT)
    if rc != 0:
        return rc
    host = tomllib.load(open(os.path.join(ROOT, "gate", "qwen36-35b-a3b.flerp-moe.toml"), "rb"))["host"]
    shim = os.path.join(ROOT, "build", "gate-scratch-subchunk", "openssl-include")
    os.makedirs(shim, exist_ok=True)
    link = os.path.join(shim, "openssl")
    if os.path.islink(link):
        os.remove(link)
    os.symlink(host["openssl_headers"], link)
    t0 = time.time()
    rc = subprocess.call(POLITE + [os.path.join(BUILD, "build.sh"), "-j6", f"-DFLERP_OPENSSL_INCLUDE_DIR={shim}",
                                   f"-DFLERP_OPENSSL_CRYPTO_LIBRARY={host['openssl_crypto']}"],
                         cwd=ROOT, stdout=open(log, "a"), stderr=subprocess.STDOUT)
    with open(log, "a") as f:
        f.write(f"# build rc {rc} in {time.time() - t0:.0f} s\n")
    return rc


# --- generator -------------------------------------------------------------------------------------------------------
def gen_prompts():
    rec = open(RECORDS).read().split()
    p5 = json.load(open(P5_IDS))
    ps = {"bridge": [str(x) for x in p5["bridge"]], "cut1000": rec[:1000], "p2k": rec + rec[:240],
          "p3000": rec + rec[::-1][:1192], "p16k": (rec * 10)[:16384], "p20k": (rec * 12)[:20480], "p32k": (rec * 19)[:32768]}
    paths = []
    for n, ids in ps.items():
        p = os.path.join(RES, f"gen-{n}.ids")
        open(p, "w").write(" ".join(ids) + "\n")
        paths.append(p)
    json.dump(list(ps), open(os.path.join(RES, "gen-names.json"), "w"))
    return list(ps), paths


def cmd_gen(_a):
    os.makedirs(RES, exist_ok=True)
    names, files = gen_prompts()
    binary = os.path.join(BUILD, "flerp-qwen35moe-generate")
    env = rocm_env(ARM_G)
    procs, logs = [], []
    t0 = time.time()
    for r in range(4):
        args = ["--staged", "tcp", "--model", MODEL, "--context", str(GEN_CONTEXT), "--max-tokens", str(GEN_TOKENS), "--world", "4",
                "--rank", str(r), "--cuts", "even", "--port", str(PORT), "--host", "127.0.0.1", "--epoch", "41", "--deadline-ms", "300000",
                "--setup-deadline-ms", "900000", "--single-reader", "--pause-ms", "0"]
        if r == 0:
            args += ["--run-specs", ",".join(s for _, s in SPECS)] + [x for f in files for x in ("--ids-file", f)]
        if r == 3:
            args += ["--draft-head"]
        lp = os.path.join(RES, f"gen-r{r}.jsonl")
        p = subprocess.Popen(POLITE[2:] + [binary] + args, stdout=open(lp, "wb"), stderr=open(os.path.join(RES, f"gen-r{r}.err"), "wb"),
                             env=env, preexec_fn=die_with_parent)
        procs.append(p)
        logs.append(lp)
        ev = wait_event(lp, p, ("loaded",), 900)   # one reader at a time
        if not ev:
            break
    rcs = []
    for p in procs:
        try:
            rcs.append(p.wait(5400))
        except subprocess.TimeoutExpired:
            p.kill()
            rcs.append("killed")
    json.dump({"rc": rcs, "seconds": round(time.time() - t0, 1), "specs": SPECS, "names": names, "env": ARM_G},
              open(os.path.join(RES, "gen-run.json"), "w"), indent=1)
    print("gen rc", rcs, "in", round(time.time() - t0), "s")
    return 0 if all(x == 0 for x in rcs) else 1


# --- serve -----------------------------------------------------------------------------------------------------------
def long_text():
    parts, n = [], 0
    for name in sorted(os.listdir(os.path.join(ROOT, "docs"))):
        if name.endswith(".md"):
            t = open(os.path.join(ROOT, "docs", name), errors="replace").read()
            parts.append(t)
            n += len(t)
            if n >= SERVE_CHARS:
                break
    body = "\n\n".join(parts)[:SERVE_CHARS]
    return "Here are some engineering notes.\n\n" + body + "\n\nIn three sentences, what are these notes mainly about?"


def chat(port, text):
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": text}], "max_tokens": SERVE_TOKENS, "flerp_trace": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"content-type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        return {"text": None, "ids": None, "error": f"HTTP {e.code}: {e.read()[:300]!r}", "seconds": round(time.time() - t0, 2)}
    except Exception as e:   # noqa: BLE001 - recorded, judged as a failure
        return {"text": None, "ids": None, "error": repr(e), "seconds": round(time.time() - t0, 2)}
    ch = d["choices"][0]
    return {"text": ch["message"].get("content"), "ids": (d.get("flerp") or {}).get("token_ids"), "finish": ch.get("finish_reason"),
            "usage": d.get("usage"), "seconds": round(time.time() - t0, 2)}


def serve_arm(name, kv, front_extra, rank3_extra, port):
    out = os.path.join(RES, f"serve-{name}")
    os.makedirs(out, exist_ok=True)
    env = rocm_env(ARM_G)
    gen = os.path.join(out, "generation_config.json")
    json.dump({"temperature": 0}, open(gen, "w"))
    base = {"weights": MODEL, "context": SERVE_CONTEXT, "world": 4, "placement": "even", "epoch": 43, "peers": f"127.0.0.1:{port + 20}",
            "single_reader": "on", "pause_ms": 0, "setup_deadline_ms": 900000, "deadline_ms": 300000}
    base.update(kv)
    procs = []

    def start(label, argv):
        lp = os.path.join(out, f"{label}.jsonl")
        p = subprocess.Popen([os.path.join(BUILD, argv[0])] + argv[1:], stdout=open(lp, "wb"), stderr=open(os.path.join(out, f"{label}.err"), "wb"),
                             env=env, preexec_fn=die_with_parent)
        procs.append(p)
        return p, lp

    result = {"arm": name, "kv": base, "front": front_extra, "rank3": rank3_extra}
    front = [f"strixflerp-{RECIPE}", "--host", "127.0.0.1", "--port", str(port), "--model-id", "m", "--generation-config", gen] + \
        front_extra + [f"executor.{k}={v}" for k, v in base.items()] + [f"text.weights={MODEL}", f"text.context={SERVE_CONTEXT}"]
    t0 = time.time()
    p0, l0 = start("r0", front)
    wait_event(l0, p0, ("waiting", "error"), 900)
    rank0_keys = ("drafts", "draft_value", "subchunks", "subchunk_min_rows", "in_flight")
    for r in (1, 2, 3):
        kr = {k: v for k, v in base.items() if k not in rank0_keys}
        kr.update(rank3_extra if r == 3 else {})
        p, lp = start(f"r{r}", ["flerp-qwen35moe-stage"] + [f"{k}={v}" for k, v in kr.items()] + [f"rank={r}"])
        ev = wait_event(lp, p, ("loaded", "error"), 900)
        if not ev or ev[0].get("event") == "error":
            result["error"] = f"rank {r}: {ev}"
    up = False
    while time.time() - t0 < 1800 and not result.get("error"):
        try:
            up = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3).status == 200
        except Exception:   # noqa: BLE001
            up = False
        if up or p0.poll() is not None:
            break
        time.sleep(2)
    result["load_s"] = round(time.time() - t0, 1)
    result["chats"] = []
    if up:
        for t in [long_text()] + SHORT:
            result["chats"].append(chat(port, t))
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


def cmd_serve(_a):
    os.makedirs(RES, exist_ok=True)
    mtp = {"draft_head": "on", "mtp_data_file": MODEL}
    arms = [("ref", {"subchunks": 1, "in_flight": 0}, ["--delivery", "tokens"], {}),
            ("auto", {}, ["--delivery", "tokens"], {}),
            ("auto_d3", {"drafts": 3}, ["--delivery", "tokens", "--drafts", "3"], mtp)]
    rcs = {}
    for i, (n, kv, fx, r3) in enumerate(arms):
        r = serve_arm(n, kv, fx, r3, PORT + 100 + 100 * i)
        rcs[n] = r["exit"]
        print(f"serve {n}: load {r['load_s']} s, exit {r['exit']}, error {r.get('error')}, chat s {[c['seconds'] for c in r['chats']]}", flush=True)
    return 0


# --- judge -----------------------------------------------------------------------------------------------------------
def judge_gen():
    run = json.load(open(os.path.join(RES, "gen-run.json")))
    names = run["names"]
    labels = [n for n, _ in SPECS]
    ev = events(os.path.join(RES, "gen-r0.jsonl"))
    staged = [e for e in ev if e.get("event") == "staged"]
    failed = [e for r in range(4) for e in events(os.path.join(RES, f"gen-r{r}.jsonl")) if e.get("event") == "relay_failed"]
    why = []
    G1 = all(x == 0 for x in run["rc"]) and not failed and len(run["rc"]) == 4
    if not G1:
        why.append(f"G1: rc {run['rc']}, relay_failed {len(failed)}")
    # rank 0 runs every spec over every prompt, spec-major (stage_driver: for spec, for prompt)
    by = {}
    for i, e in enumerate(staged):
        by[(labels[i // len(names)] if i // len(names) < len(labels) else "?", names[e["prompt"]])] = e
    table, g2, g3, g4 = [], True, True, True
    for n in names:
        ref = by.get(("ref", n))
        if ref is None:
            why.append(f"no ref run for {n}")
            g2 = g3 = g4 = False
            continue
        for lab in labels:
            e = by.get((lab, n))
            if e is None:
                why.append(f"no {lab} run for {n}")
                g2 = False
                continue
            row = {"prompt": n, "spec": lab, "ids_equal": e["ids"] == ref["ids"], "parts": e.get("parts"), "max_in_flight": e.get("max_in_flight"),
                   "prefill_s": e.get("prefill_s"), "decode_s": e.get("decode_s")}
            if lab in ("auto", "s16", "tight"):
                row["sha_equal"] = e["logits_sha256"] == ref["logits_sha256"]
                if not (row["ids_equal"] and row["sha_equal"]):
                    g2 = False
                    why.append(f"G2 {n} {lab}: ids {row['ids_equal']} sha {row['sha_equal']}")
            if lab in ("tok", "d3") and not row["ids_equal"]:
                g3 = False
                why.append(f"G3 {n} {lab}: ids differ")
            table.append(row)
        # G4: the flow asked for
        pa = lambda lab: by.get((lab, n), {}).get("parts")
        mf = lambda lab: by.get((lab, n), {}).get("max_in_flight")
        if n in ("bridge", "cut1000") and any(pa(lab) != [1] for lab in labels if by.get((lab, n))):
            g4 = False
            why.append(f"G4 {n}: parts {[pa(lab) for lab in labels]}")
        if n == "p20k" and (pa("auto") != [16, 4] or not (1 < (mf("auto") or 0) <= 8)):
            g4 = False
            why.append(f"G4 p20k auto: parts {pa('auto')} max_in_flight {mf('auto')}")
        if n == "p32k" and (pa("auto") != [16, 16] or not (1 < (mf("auto") or 0) <= 8)):
            g4 = False
            why.append(f"G4 p32k auto: parts {pa('auto')} max_in_flight {mf('auto')}")
        if (mf("tight") or 0) > 2 or mf("ref") != 1 or any(x != 1 for x in (pa("ref") or [0])):
            g4 = False
            why.append(f"G4 {n}: tight max_in_flight {mf('tight')}, ref parts {pa('ref')} max_in_flight {mf('ref')}")
    return {"G1": G1, "G2": g2, "G3": g3, "G4": g4, "table": table}, why


def judge_serve():
    why = []
    res = {}
    for n in ("ref", "auto", "auto_d3"):
        p = os.path.join(RES, f"serve-{n}", "result.json")
        res[n] = json.load(open(p)) if os.path.exists(p) else None
    s1 = True
    for n, r in res.items():
        if r is None or r.get("error") or len(r["chats"]) != 1 + len(SHORT) or any(x != 0 for x in r["exit"]) or \
                any(c.get("error") or not c.get("ids") for c in r["chats"]):
            s1 = False
            why.append(f"S1 {n}: {None if r is None else (r.get('error'), r['exit'], [c.get('error') for c in r['chats']])}")
    s2 = s1
    if s1:
        key = lambda n: [json.dumps(c["ids"]) + "|" + str(c["text"]) for c in res[n]["chats"]]
        for n in ("auto", "auto_d3"):
            if key(n) != key("ref"):
                s2 = False
                why.append(f"S2 {n}: {[x == y for x, y in zip(key('ref'), key(n))]}")
    secs = {n: [c.get("seconds") for c in r["chats"]] for n, r in res.items() if r}
    usage = {n: [(c.get("usage") or {}).get("prompt_tokens") for c in r["chats"]] for n, r in res.items() if r}
    return {"S1": s1, "S2": s2, "seconds": secs, "prompt_tokens": usage}, why


def cmd_judge(_a):
    g, wg = judge_gen() if os.path.exists(os.path.join(RES, "gen-run.json")) else ({"G1": False}, ["no gen run"])
    s, ws = judge_serve()
    u1 = subprocess.call([sys.executable, "-m", "unittest", "-q", "tests.test_stage_relay"], cwd=ROOT,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    crit = {"U1": u1, "G1": g.get("G1"), "G2": g.get("G2"), "G3": g.get("G3"), "G4": g.get("G4"), "S1": s["S1"], "S2": s["S2"]}
    doc = {"criteria": crit, "pass": all(crit.values()), "why": wg + ws + ([] if u1 else ["U1: tests/test_stage_relay.py failed"]),
           "gen": g, "serve": s}
    json.dump(doc, open(os.path.join(RES, "verdict.json"), "w"), indent=1)
    for k, v in crit.items():
        print(k, "PASS" if v else "FAIL")
    print("verdict", "PASS" if doc["pass"] else "FAIL", doc["why"][:12])
    return 0 if doc["pass"] else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("build", "gen", "serve", "judge"):
        sub.add_parser(c)
    a = ap.parse_args()
    return {"build": cmd_build, "gen": cmd_gen, "serve": cmd_serve, "judge": cmd_judge}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
