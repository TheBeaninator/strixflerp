#!/usr/bin/env python3
"""ORNITH-CONFIRM2 (card 3bed1267a2, owner 2026-11-14 "schedule the confirmation"): the champion configurations on
Ornith-1.5 397B itself, world 4 on F D A C, under ONE pod-quad hold taken and released by this runner.

Launches (each its own load; every rep measured inside it):
  pold   the old prompt path (no FLERP_* env), prompts P5 three + records + 2K + 8 records prefixes, 16 sub-chunks
  pnew   ornith397.prefill-best env (FLERP_MOE_GROUPED=1 FLERP_DENSE_GROUPED=1 FLERP_ATTN_GQA=1, 4,096-row chunks,
         bulk chunks off), the same prompts + 16K + 32K, runs S16, S1, S16
  d0 d1  ornith397.decode-best (pinned rows, token reply, MTP head on rank 3), drafts 0 and 3 interleaved, the run
         order reversed in d1 (A/B noise bounds), the 12 prompts of ORNITH-O3 (P5 four, code four, prose four)
  serve  ornith397.serve end to end (flerp-serve front end on F, flerp-qwen35moe-stage ranks 1-3), one OpenAI chat
         request of about 20K tokens

Bytes: ranks 0-1 stream over the rail from control-node's PLACE placements (--rank-origin), ranks 2-3 read A's and
C's /cache/flerp-p5 (PLACE). --stage-a stages A inside the hold (A serves strixite until the hold drains it; no
staging beside serving). Built on gate/ornith_p5.py (Window, gen_world, cmd_stage); run it from an export whose
ornith_p5.py has --rank-origin (main f60ff13 or later).

    python3 gate/ornith_confirm2.py run --card 3bed1267a2 --rank-origin 0=... --rank-origin 1=... [--stage-a]
    python3 gate/ornith_confirm2.py judge        # the verdict from gate/results/ornith-confirm2/run.json (no GPU)
"""
import argparse
import hashlib
import json
import os
import shlex
import signal
import statistics
import subprocess
import sys
import time
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
sys.path.insert(0, ROOT)
import flerplib.configs as CF  # noqa: E402
import moe_stage as MS  # noqa: E402
import ornith_p5 as P  # noqa: E402

RES = os.path.join(ROOT, "gate", "results", "ornith-confirm2")
RECORDS = os.path.join(ROOT, "gate", "results", "ornith15-397b.p4", "long.ids")   # 1,808 ids, the P5 long prompt
CUTS = [200, 400, 600, 800, 1000, 1200, 1400, 1600]   # records prefixes: prefill-end top-1 at real text positions
PREFILL_CONTEXT = 33024        # 32,768 prompt + 32 generated + room
PREFILL_TOKENS = 32
SERVE_CONTEXT = 32768
SERVE_CHARS = 72000            # about 18-22K tokens of the repo's own docs
SERVE_MAX_TOKENS = 128
EXTRA_ENV = ""                 # a launch's FLERP_* environment, added to every rank unit it starts


class Win(P.Window):
    def start(self, n, tag, argv, env="", serving_ok=False):
        return super().start(n, tag, argv, (env + " " + EXTRA_ENV).strip(), serving_ok)


def prefill_prompts():
    p5 = P.MP.load_json(os.path.join(ROOT, "gate/results/qwen36-35b-a3b.flerp-moe/prompt-ids.json"))
    rec = [int(x) for x in open(RECORDS).read().split()]
    ids = {n: p5[n] for n in ("bridge", "code", "rivers")}
    ids["records"] = rec
    ids["p2k"] = rec + rec[:240]                       # the 35B proxy's 2,048-id speed prompt
    for c in CUTS:
        ids[f"cut{c}"] = rec[:c]
    ids["p16k"] = (rec * 10)[:16384]                   # as the 35B's 16,384-row arm
    ids["p32k"] = (rec * 19)[:32768]
    return ids, rec


def env_of(name):
    return CF.env_prefix(CF.load(name))


def prefill_launch(win, doc, key, names, ids, specs, env, log):
    """One prefill launch: rank 0 runs SPECS over NAMES (no draft head), env on every rank."""
    global EXTRA_ENV
    cfg = win.cfg
    path = os.path.join(RES, f"{key}-prompts.json")
    with open(path, "w") as f:
        json.dump({n: ids[n] for n in names}, f)
    cfg["model"]["prompt_ids"], cfg["model"]["prompts"] = path, names
    cfg["run"]["context"], cfg["run"]["max_tokens"], cfg["run"]["run_timeout_s"] = PREFILL_CONTEXT, PREFILL_TOKENS, 3600
    P.SPEC, P.PROBE = None, None
    P.LAUNCH = {"rows": "pinned", "specs": specs}
    EXTRA_ENV = env
    win.status = f"{key} loading"
    log(f"{key}: env [{env}] specs {specs} prompts {[(n, len(ids[n])) for n in names]}")
    try:
        _, why, _ = P.gen_world(win, 4, "cache", doc, dumps=False, key=f"{key}-", res=RES)
    finally:
        EXTRA_ENV = ""
        P.LAUNCH = None
    ev = doc[f"{key}-w4"].pop("rank_events")
    staged = [e for e in ev.get(0, {}).get("events", []) if e.get("event") == "staged"]
    sl = specs.split(",")
    runs = []
    for i, e in enumerate(staged):
        runs.append({"spec": sl[i // len(names)] if i // len(names) < len(sl) else "?", "prompt": names[e["prompt"]],
                     "prompt_tokens": e["prompt_tokens"], "prefill_s": e["prefill_s"],
                     "prefill_tok_s": round(e["prompt_tokens"] / e["prefill_s"], 1) if e["prefill_s"] else None,
                     "chunks": e["calls"]["chunks"], "subchunks": e["subchunks"], "ids": e["ids"],
                     "decode_tok_s": round(len(e["ids"]) / (e["calls"]["decode_wall_ms"] / 1000.0), 2) if e["calls"]["decode_wall_ms"] else None})
    if len(staged) != len(sl) * len(names):
        why.append(f"{key}: {len(staged)} staged results, want {len(sl) * len(names)}")
    for r, info in ev.items():
        for e in info["events"]:
            if e.get("event") in ("error", "relay_failed", "staged_failed"):
                why.append(f"{key}: rank {r} {e.get('event')}: {str(e.get('error', e.get('reason', '')))[:200]}")
    doc[key] = {"env": env, "specs": specs, "names": names, "runs": runs}
    return why


def decode_launch(win, doc, k, specs, log):
    """ORNITH-O3 machinery: decode-best (no env), drafts on the last rank, judged by gate/ornith_o3_spec.py."""
    import ornith_o3_spec as OS
    with open(os.path.join(ROOT, "gate", "ornith-o3-spec.toml"), "rb") as f:
        crit = tomllib.load(f)
    prompts = OS.prompt_set(None)
    P.SPEC = {"names": [p[0] for p in prompts], "ids": {p[0]: p[2] for p in prompts}, "max_tokens": crit["proxy"]["max_tokens"],
              "context": P.SPEC_CONTEXT, "costs": crit["proxy"]["draft_costs"]}
    P.LAUNCH = {"rows": "pinned", "specs": specs}
    win.status = f"decode launch {k} loading"
    log(f"d{k}: decode-best specs {specs}")
    try:
        _, w, _ = P.gen_world(win, 4, "cache", doc, dumps=False, key=f"d{k}-", res=RES)
    finally:
        P.LAUNCH = None
    ev = doc[f"d{k}-w4"].pop("rank_events")
    rec, sw = OS.judge_launch(k, specs, ev, prompts, P.SPEC["max_tokens"])
    sw += OS.judge_all({"launches": [rec]}, prompts, P.SPEC["max_tokens"], anchors=False)
    doc.setdefault("decode", {"launches": []})["launches"].append(rec)
    P.SPEC = None
    return w + sw


def serve_text():
    parts, n = [], 0
    for root, _, files in sorted(os.walk(os.path.join(ROOT, "docs"))):
        for f in sorted(files):
            if f.endswith(".md"):
                t = open(os.path.join(root, f), errors="replace").read()
                parts.append(f"# file docs/{os.path.relpath(os.path.join(root, f), os.path.join(ROOT, 'docs'))}\n\n{t}")
                n += len(t)
                if n > SERVE_CHARS:
                    break
        if n > SERVE_CHARS:
            break
    text = "\n\n".join(parts)[:SERVE_CHARS]
    return text + "\n\n---\nIn three sentences: what is this repository for, and what are its main parts?"


def serve(win, doc, log):
    """ornith397.serve end to end: rank 0 = flerp-serve (OpenAI front end) on F, ranks 1-3 flerp-qwen35moe-stage;
    ranks 0-1 stream from the control origins (--rank-origin), 2-3 read /cache."""
    cfg, run = win.cfg, win.cfg["run"]
    rt = CF.load("ornith397.serve")
    nd = CF.drafts(rt)
    g, b, w = cfg["model"]["gguf"], cfg["host"]["bundle"], cfg["host"]["work"]
    nodes = P.nodes_of(cfg, 4)
    peers = ",".join(n["rail"] for n in nodes) + f":{run['port'] + 20}"
    kv = {"weights": g, "context": SERVE_CONTEXT, "world": 4, "placement": "even", "epoch": run["epoch"] + 41, "peers": peers,
          "single_reader": "on", "pause_ms": 0, "setup_deadline_ms": run["setup_deadline_ms"], "deadline_ms": run["deadline_ms"]}
    env = f"LD_LIBRARY_PATH={b}/lib {CF.env_prefix(rt)}".strip()
    out = {"config": rt["name"], "drafts": nd, "env": CF.env_prefix(rt), "context": SERVE_CONTEXT}
    doc["serve"] = out

    def keys(r):
        k = dict(kv)
        if r in P.SWARM_RANKS:
            o = P.RANK_ORIGINS[r]
            k.update(swarm_helper=f"{b}/loom-swarm-stream", swarm_manifest=f"{b}/ornith.swarm.json", swarm_from=o["listen"],
                     swarm_conns=cfg["swarm"]["conns"])
        else:
            k["data_file"] = P.cache_path(cfg)
        if r == 0 and nd:
            k["drafts"] = nd
        if r == 3 and nd:
            k.update(draft_head="on", mtp_data_file=P.cache_path(cfg))
        return k
    gen = os.path.join(RES, "generation_config.json")
    with open(gen, "w") as f:
        json.dump({"temperature": 0}, f)
    subprocess.run(["rsync", "-a", "-e", " ".join(P.SSH), gen, f"pod@{nodes[0]['mgmt']}:{w}/generation_config.json"], check=True)
    front = [f"{b}/flerp-serve", "--host", "127.0.0.1", "--port", str(run["http_port"]), "--model-id", "ornith-1.5-397b",
             "--generation-config", f"{w}/generation_config.json"] + (["--delivery", "tokens"] if rt["stage"].get("reply") == "token" else []) + \
        (["--drafts", str(nd)] if nd else []) + [f"executor.{k}={v}" for k, v in keys(0).items()] + [f"text.weights={g}", f"text.context={SERVE_CONTEXT}"]
    why = []
    t0 = time.time()
    win.status = "serve loading"
    win.start(nodes[0], "serve-r0", front, env)
    ev = win.wait_event(nodes[0], "serve-r0", ("waiting", "loaded", "error"), run["load_timeout_s"])
    log(f"serve: front end rank 0 at +{time.time() - t0:.0f} s: {[e for e in ev if e.get('event') in ('loaded', 'waiting', 'error')][:2]}"[:600])
    for r in range(1, 4):
        win.start(nodes[r], f"serve-r{r}", [f"{b}/flerp-qwen35moe-stage"] + [f"{k}={v}" for k, v in keys(r).items()] + [f"rank={r}"], env)
    log(f"serve: config {rt['name']} drafts {nd} env [{CF.env_prefix(rt)}]: front end + 3 stages started")
    health, end = None, time.time() + 1500
    while time.time() < end:
        rc, text = P.on(nodes[0]["mgmt"], f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{run['http_port']}/health", 30)
        health = text.strip()
        if health == "200" or any(win.rc(nodes[r], f"serve-r{r}") is not None for r in range(4)):
            break
        time.sleep(10)
    out["health"], out["load_s"] = health, round(time.time() - t0, 1)
    log(f"MILESTONE serve /health {health} at +{out['load_s']} s")
    if health == "200":
        text = serve_text()
        body = json.dumps({"model": "ornith-1.5-397b", "messages": [{"role": "user", "content": text}], "max_tokens": SERVE_MAX_TOKENS})
        bf = os.path.join(RES, "serve-request.json")
        with open(bf, "w") as f:
            f.write(body)
        subprocess.run(["rsync", "-a", "-e", " ".join(P.SSH), bf, f"pod@{nodes[0]['mgmt']}:{w}/serve-request.json"], check=True)
        t1 = time.time()
        rc, resp = P.on(nodes[0]["mgmt"], f"curl -s -w '\\n%{{http_code}}' -X POST -H 'content-type: application/json' "
                                           f"--data @{w}/serve-request.json http://127.0.0.1:{run['http_port']}/v1/chat/completions", 1800)
        body_txt, _, code = resp.rpartition("\n")
        try:
            r = json.loads(body_txt)
        except ValueError:
            r = {"raw": body_txt[-800:]}
        ch = (r.get("choices") or [{}])[0]
        out["chat"] = {"status": code.strip(), "seconds": round(time.time() - t1, 1), "chars": len(text), "usage": r.get("usage"),
                       "finish_reason": ch.get("finish_reason"), "text": (ch.get("message") or {}).get("content"), "raw": r.get("raw")}
        log(f"MILESTONE serve chat HTTP {code.strip()} in {out['chat']['seconds']} s usage {r.get('usage')} finish {ch.get('finish_reason')}: "
            f"{str(out['chat']['text'])[:200]!r}")
        if code.strip() != "200" or not out["chat"]["text"] or ch.get("finish_reason") not in ("stop", "length"):
            why.append(f"serve chat: HTTP {code.strip()}, finish {ch.get('finish_reason')}")
    else:
        why.append(f"serve: /health {health}")
    for r in range(4):
        win.term(nodes[r], f"serve-r{r}")
    out["exit"] = [win.wait_rc(nodes[r], f"serve-r{r}", 180) for r in range(4)]
    for r in range(4):
        win.fetch(nodes[r], f"serve-r{r}", RES)
    return why


def save(doc):
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "run.json"), "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True, default=str)


def cmd_run(a):
    cfg = P.load_cfg()
    cfg["host"].update(hold_owner=a.hold_owner, hold_reason=a.hold_reason, hold_ttl_s=a.hold_ttl)
    for x in a.rank_origin:
        r, o = P.parse_rank_origin(x)
        P.RANK_ORIGINS[r] = o
        P.SWARM_RANKS.add(r)
    P.BEFORE_RELEASE = a.before_release
    os.makedirs(RES, exist_ok=True)
    log = P.Log(os.path.join(RES, "run.txt"))
    win = Win(cfg, log, a.card)
    phases = a.phases.split(",")
    ids, rec = prefill_prompts()
    base = ["bridge", "code", "rivers", "records", "p2k"] + [f"cut{c}" for c in CUTS]
    exe = os.path.join(P.GEN_BUILD, "out", "flerp-qwen35moe-generate")
    doc = {"when": P.utc(), "run_id": win.run_id, "card": a.card, "phases": phases, "failures": [], "rank_origins": P.RANK_ORIGINS,
           "binary_sha256": hashlib.sha256(open(exe, "rb").read()).hexdigest(), "export": open(os.path.join(ROOT, ".export-commit")).read().strip()
           if os.path.exists(os.path.join(ROOT, ".export-commit")) else None, "hold": {k: cfg["host"][k] for k in ("hold_owner", "hold_reason", "hold_ttl_s")}}
    why = doc["failures"]
    with open(os.path.join(RES, "runner.pid"), "w") as f:
        f.write(str(os.getpid()))
    log(f"confirm2 {win.run_id}: phases {phases}, origins {P.RANK_ORIGINS}, generator sha256 {doc['binary_sha256']}, export {doc['export']}")

    def on_signal(sig, frame):
        raise KeyboardInterrupt(f"signal {sig}")
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        local, missing = P.local_bundle(cfg)
        if missing:
            raise RuntimeError(f"bundle sources missing: {missing}")
        fb, err = P.gen_flag_check(cfg)
        if err or fb:
            raise RuntimeError(f"generator flags: {err or fb}")
        for r, o in P.RANK_ORIGINS.items():
            if not os.path.isfile(o["ranges"]):
                raise RuntimeError(f"rank {r}: no ranges file {o['ranges']}")
        size = str(P.C.read_header(cfg["model"]["gguf"])["file_bytes"])
        nodes4 = P.nodes_of(cfg, 4)

        def staged(n):
            _, t = P.on(n["mgmt"], f"f={shlex.quote(P.cache_path(cfg))}; echo $(readlink -f $f) $(stat -c %s $f 2>/dev/null)", 60)
            parts = t.split()
            return len(parts) == 2 and parts[1] == size and not P.cold_path(parts[0])
        for r, n in enumerate(nodes4):
            if r in P.SWARM_RANKS or (a.stage_a and n["name"] == "A"):
                continue
            if not staged(n):
                raise RuntimeError(f"rank {r} on {n['name']}: {P.cache_path(cfg)} not staged")
        if a.preflight_only:
            for r, o in sorted(P.RANK_ORIGINS.items()):
                win.start_origin(r, o)
            win.stop_origins()
            log("preflight only: OK (no hold taken)")
            win.closed = True
            return 0
        for r, o in sorted(P.RANK_ORIGINS.items()):
            win.start_origin(r, o)
        nodes = cfg["host"]["nodes"]
        win.start_heartbeat()
        if a.winjob:
            # WINJOB win_run took F D A C (pod-window open, leases on the job heartbeat) and closes the window after cmd
            got = os.environ.get("WINDOW_LETTERS", "").split()
            if got != [n["name"] for n in nodes]:
                raise RuntimeError(f"--winjob: the grant is {got}, want {[n['name'] for n in nodes]}")
            doc["window"] = {k: os.environ.get(k) for k in ("WINDOW_OWNER", "WINDOW_LETTERS", "WINDOW_IPS", "WINDOW_EXPIRES", "WJ_JOB", "KIT_DIR")}
        elif not win.take_holds([n["mgmt"] for n in nodes]):
            raise RuntimeError("holds refused after every try")
        doc["hold_taken"] = P.utc()
        log(f"MILESTONE hold taken {doc['hold_taken']}")
        if a.after_hold:
            p = subprocess.run(["bash", "-c", a.after_hold], capture_output=True, text=True, timeout=900)
            log(f"after-hold rc {p.returncode}: {(p.stdout + p.stderr).strip()[-600:]}")
        ok, facts = win.wait_quiet(nodes, cfg["host"]["hold_drain_s"])
        if not ok:
            raise RuntimeError("serving still up on a held node after the drain")
        log("MILESTONE drained")
        if a.stage_a:
            # world-4 slice only (place request ornith-eng-A, as tools/ornith-engine/stage.py), owned by the window's
            # holder so PLACE does not defer the node it holds
            t0 = time.time()
            sys.path.insert(0, os.path.join(ROOT, "tools", "ornith-engine"))
            import stage as ST
            _, ranges = ST.ranges_for(cfg, 2)
            rf = os.path.join(P.PLACE_RANGES, "ornith-eng-A.ranges")
            with open(rf, "w") as fo:
                for o, e in ranges:
                    fo.write(f"{o} {e - o}\n")
            pc, s_ = P.place_cmd(), cfg["swarm"]
            owner = os.environ.get("WINDOW_OWNER") or cfg["host"]["hold_owner"]
            rid = "ornith-eng-A"
            rc = 0
            for argv in (pc + ["request", "--id", rid, "--nodes", "A", "--dest", os.path.dirname(P.cache_path(cfg)), "--out",
                               os.path.basename(P.cache_path(cfg)), "--manifest", s_["manifest"], "--from", s_["origin"], "--ranges", f"A={rf}",
                               "--until", a.stage_until, "--card", a.card, "--owner", owner],
                         pc + ["run", "--id", rid], pc + ["verify", "--id", rid]):
                p = subprocess.run(argv, capture_output=True, text=True, timeout=2400)
                log(f"{' '.join(argv[len(pc):len(pc) + 3])}: rc {p.returncode}: {(p.stdout + p.stderr).strip()[-500:]}")
                rc = rc or p.returncode
                if rc:
                    break
            doc["stage_a"] = {"rc": rc, "seconds": round(time.time() - t0, 1)}
            log(f"MILESTONE A staged rc {rc} in {time.time() - t0:.0f} s")
            if rc != 0 or not staged(next(n for n in nodes4 if n["name"] == "A")):
                raise RuntimeError(f"A staging failed (rc {rc})")
        for n in nodes:
            P.push_bundle(cfg, n["mgmt"], local)
        log("bundle on every node (md5 checked)")
        if "pold" in phases:
            w = prefill_launch(win, doc, "pold", base, ids, "on:token:16:0", "", log)
            save(doc)
            log(f"MILESTONE pold RESULT faults {len(w)} {w[:3]}")
            win.stop_all()
            why += w
            if w:
                raise RuntimeError("pold")
        if "pnew" in phases:
            w = prefill_launch(win, doc, "pnew", base + ["p16k", "p32k"], ids, "on:token:16:0,on:token:1:0,on:token:16:0",
                               env_of("ornith397.prefill-best"), log)
            save(doc)
            log(f"MILESTONE pnew RESULT faults {len(w)} {w[:3]}")
            win.stop_all()
            why += w
            if w:
                raise RuntimeError("pnew")
        for k, specs in enumerate(("on:token:1:0,on:token:1:3,on:token:1:0,on:token:1:3", "on:token:1:3,on:token:1:0,on:token:1:3,on:token:1:0")):
            if f"d{k}" not in phases:
                continue
            w = decode_launch(win, doc, k, specs, log)
            save(doc)
            log(f"MILESTONE d{k} RESULT faults {len(w)} {w[:3]}")
            win.stop_all()
            why += w
            if w:
                raise RuntimeError(f"d{k}")
        if "serve" in phases:
            w = serve(win, doc, log)
            save(doc)
            log(f"MILESTONE serve RESULT faults {len(w)} {w[:3]}")
            why += w
    except (KeyboardInterrupt, RuntimeError, subprocess.SubprocessError, OSError, ValueError, KeyError) as e:
        why.append(f"stopped: {e}")
    finally:
        win.close("end of run" if not why else f"failures: {why[:3]}")
        if a.winjob and a.before_release:
            p = subprocess.run(["bash", "-c", a.before_release], capture_output=True, text=True, timeout=900)
            doc["before_release"] = (p.stdout + p.stderr)[-3000:]
            log(f"before release (window closes after cmd) rc {p.returncode}: {(p.stdout + p.stderr).strip()[-1500:]}")
        doc["ended"] = P.utc()
        save(doc)
        try:
            judge(doc)
        except Exception as e:      # the verdict never hides the run
            log(f"judge failed: {e}")
        log(f"MILESTONE confirm2 END faults {why[:6]}")
    return 0 if not why else 1


def first_diff(a, b):
    return MS.first_diff(a, b)


def judge(doc=None):
    """Numbers and the card's inline criteria from run.json; writes verdict.json and summary.md."""
    doc = doc or json.load(open(os.path.join(RES, "run.json")))
    v = {"prefill": {}, "fidelity": {}, "decode": {}, "serve": {}}
    rec = [int(x) for x in open(RECORDS).read().split()]
    for key in ("pold", "pnew"):
        for r in (doc.get(key) or {}).get("runs", []):
            v["prefill"].setdefault(key, {}).setdefault(r["spec"], {}).setdefault(r["prompt"], []).append(
                {"tok_s": r["prefill_tok_s"], "prefill_s": r["prefill_s"], "tokens": r["prompt_tokens"]})
    old = {r["prompt"]: r for r in (doc.get("pold") or {}).get("runs", []) if r["spec"] == "on:token:16:0"}
    new_runs = (doc.get("pnew") or {}).get("runs", [])
    new16 = {}
    for r in new_runs:
        if r["spec"] == "on:token:16:0":
            new16.setdefault(r["prompt"], r)
    new1 = {r["prompt"]: r for r in new_runs if r["spec"] == "on:token:1:0"}
    fid = v["fidelity"]
    for name, o in old.items():
        n = new16.get(name)
        if not n:
            continue
        d = first_diff(n["ids"], o["ids"])
        fid[name] = {"greedy_equal": n["ids"] == o["ids"], "first_divergence": d, "top1_prefill_end": n["ids"][:1] == o["ids"][:1],
                     "s1_equal_s16": (new1.get(name) or {}).get("ids") == n["ids"],
                     "old_tokens": len(o["ids"]), "new_tokens": len(n["ids"])}
        if name.startswith("cut"):
            c = int(name[3:])
            fid[name]["next_token_old"] = o["ids"][:1] == rec[c:c + 1]
            fid[name]["next_token_new"] = n["ids"][:1] == rec[c:c + 1]
    p5 = ["bridge", "code", "rivers"]
    cuts = [k for k in fid if k.startswith("cut")]
    # a run that generated NOTHING on both paths compares [] to [] and says nothing about fidelity, so it is
    # named separately and never counted as an agreement. A prompt that spoke on one path and not the other IS
    # a divergence and stays in the count (greedy_equal already fails it). See "the silent records prompt".
    silent = sorted(k for k, x in fid.items() if not k.startswith("_") and not x.get("new_tokens") and not x.get("old_tokens"))
    live = {k: x for k, x in fid.items() if not k.startswith("_") and k not in silent}
    fid["_summary"] = {"p5_greedy_equal": sum(fid.get(k, {}).get("greedy_equal", False) for k in p5),
                       "all_greedy_equal": sum(x.get("greedy_equal", False) for k, x in live.items()),
                       "prompts": len(live), "silent_prompts": silent,
                       "top1_agree": sum(x.get("top1_prefill_end", False) for k, x in live.items()),
                       "cut_next_token_old": sum(fid[k]["next_token_old"] for k in cuts),
                       "cut_next_token_new": sum(fid[k]["next_token_new"] for k in cuts), "cuts": len(cuts),
                       "note": "no FP32 reference exists for the 397B: fidelity is against the old per-token prompt "
                               "path; prompts counted are those that generated output on both paths"}
    # decode
    runs = [r for L in (doc.get("decode") or {}).get("launches", []) for r in L["runs"]]
    dec = v["decode"]
    for d in sorted({r["drafts"] for r in runs}):
        for cls in ("code", "prose", "p5", "all"):
            per_launch = []
            for L in sorted({r["launch"] for r in runs}):
                ts = [len(r["ids"]) / (r["decode_wall_ms"] / 1000.0) for r in runs
                      if r["drafts"] == d and r["launch"] == L and (cls == "all" or r["class"] == cls) and r["decode_wall_ms"]]
                if ts:
                    per_launch.append(statistics.mean(ts))
            allts = [len(r["ids"]) / (r["decode_wall_ms"] / 1000.0) for r in runs
                     if r["drafts"] == d and (cls == "all" or r["class"] == cls) and r["decode_wall_ms"]]
            if allts:
                dec.setdefault(str(d), {})[cls] = {"mean": round(statistics.mean(allts), 2), "median": round(statistics.median(allts), 2),
                                                    "min_run": round(min(allts), 2),
                                                    "max_run": round(max(allts), 2), "per_launch_mean": [round(x, 2) for x in per_launch],
                                                    "n": len(allts)}
    # identity: each drafts>0 run against the drafts-0 run of the SAME prompt and launch. Counting drafts-0
    # runs against themselves (the first version did) made the count partly self-satisfying: equal == runs
    # could be met by runs that were never compared to anything.
    ident, refs, vacuous = [], 0, 0
    for L in (doc.get("decode") or {}).get("launches", []):
        by = {}
        for r in L["runs"]:
            by.setdefault(r["prompt"], []).append(r)
        for p, rs in by.items():
            off = [r["ids"] for r in rs if r["drafts"] == 0]
            for r in rs:
                if r["drafts"] == 0:
                    continue
                refs += bool(off)
                if off and r["ids"] == off[0] and not r["ids"]:
                    vacuous += 1   # [] == []: the prompt generated nothing on either setting
                ident.append(bool(off) and r["ids"] == off[0])
    dec["_identity"] = {"equal": sum(ident), "runs": len(ident), "compared_to_drafts0": refs,
                        "of_which_both_empty": vacuous,
                        "note": "equal counts every drafts>0 run whose ids match its prompt's drafts-0 run; "
                               "of_which_both_empty are pairs that generated nothing (the records prompt) and "
                               "prove nothing - the substantive count is equal minus of_which_both_empty"}
    s = doc.get("serve") or {}
    v["serve"] = {"health": s.get("health"), "drafts": s.get("drafts"), "load_s": s.get("load_s"), **{k: (s.get("chat") or {}).get(k) for k in ("status", "seconds", "usage", "finish_reason")}}
    # inline criteria (card 3bed1267a2)
    p2k = [x["tok_s"] for x in v["prefill"].get("pnew", {}).get("on:token:16:0", {}).get("p2k", []) if x["tok_s"]]
    d3 = dec.get("3", {}).get("all", {})
    crit = {"prefill": bool(p2k) and statistics.median(p2k) >= 250 and fid["_summary"]["p5_greedy_equal"] == 3,
            "decode": dec["_identity"]["runs"] > 0 and dec["_identity"]["equal"] == dec["_identity"]["runs"] and bool(d3)
            and statistics.median([len(r["ids"]) / (r["decode_wall_ms"] / 1000.0) for r in runs if r["drafts"] == 3 and r["decode_wall_ms"]]) >= 23.1,
            "serve": v["serve"].get("status") == "200" and v["serve"].get("finish_reason") in ("stop", "length")}
    v["criteria"] = crit
    with open(os.path.join(RES, "verdict.json"), "w") as f:
        json.dump(v, f, indent=1, sort_keys=True)
    print(json.dumps(v["criteria"]), json.dumps(fid["_summary"]), json.dumps({k: x for k, x in dec.items()}))
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "judge"])
    ap.add_argument("--card", default="3bed1267a2")
    ap.add_argument("--rank-origin", action="append", default=[])
    ap.add_argument("--hold-owner", default="agent")
    ap.add_argument("--hold-reason", default="ORNITH-CONFIRM2 3bed1267a2: Ornith 397B champion confirmation (owner approved 2026-11-14)")
    ap.add_argument("--hold-ttl", type=int, default=6300)
    ap.add_argument("--after-hold", default="")
    ap.add_argument("--before-release", default="")
    ap.add_argument("--stage-a", action="store_true", help="stage A's ranks inside the hold, after the drain")
    ap.add_argument("--stage-until", default="+6h")
    ap.add_argument("--phases", default="pold,pnew,d0,d1,serve")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--winjob", action="store_true", help="run as a WINJOB win_run cmd: the window holds F D A C, no pod-quad hold here")
    a = ap.parse_args()
    if a.cmd == "judge":
        judge()
        return 0
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
