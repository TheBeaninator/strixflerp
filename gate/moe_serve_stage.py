#!/usr/bin/env python3
"""Gate of the executor ABI over the stages (executor.flerp.qwen35moe_abi, tool.flerp.qwen35moe_stage,
runtime.flerp.text_gufo, serve.flerp.http): stage P3 of docs/PHASE5-FOURNODE-DESIGN.md. Run on the gfx1151 build host:

    python3 gate/moe_serve_stage.py [--config gate/qwen36-35b-a3b.flerp-serve-stage.toml] [--stages check,build,...]

Stages, each writing evidence under gate/results/<recipe>/:
  check        ./flerp check
  build        flerp assemble + build.sh (the four programs)
  conformance  the conformance driver over the adapter, world 1 and world 2 (loopback), the four gate prompts
  serve1       the front end at world 1: /health, completions, chats, refusals, SIGTERM
  reference    stage P2's generator on the chat prompt ids serve1 rendered; judges serve1's chats
  serve2       the front end at world 2 in one process (loopback)
  tcp          the front end as rank 0 and flerp-qwen35moe-stage as rank 1 over TCP on 127.0.0.1
  shutdown     SIGTERM cases on the dense model over TCP

Criteria: [criteria] of the config, committed before the first GPU run. P1's recorded ids are the reference for the
gate prompts; the P2 generator built in the same tree is the reference for the chats.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gate"))
import summary as SM  # noqa: E402  (summary.json / summary.md, config_sha256)

STAGES = ("check", "build", "conformance", "serve1", "reference", "serve2", "tcp", "shutdown")
SERVE_STAGE = {"serve1": "world1", "serve2": "loopback2", "tcp": "tcp2"}
GIB = float(1 << 30)
CAPS_EXPECTED = {"deliveries": 5, "flags": 0, "seqs": 1, "max_draft": 0, "snapshots": 0, "candidates": 0}


# --- judgements (no I/O: tests/test_moe_serve_stage.py) ---------------------------------------------------------
def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def judge_conformance(doc, p1, names, context, max_tokens):
    """Criteria 2 and 3 on one driver result. p1: {prompt: [ids]}. Returns [failures]."""
    why = []
    if doc.get("error"):
        why.append(f"the driver stopped: {doc['error']}")
    if doc.get("failures", 1) != 0:
        why.append(f"{doc.get('failures')} driver check(s) failed")
    caps = doc.get("caps") or {}
    if doc.get("abi") != 2:
        why.append(f"abi {doc.get('abi')}")
    if caps.get("positions") != context:
        why.append(f"caps.positions {caps.get('positions')}, context {context}")
    for k, v in CAPS_EXPECTED.items():
        if caps.get(k) != v:
            why.append(f"caps.{k} {caps.get(k)}, expected {v}")
    cases = {c.get("name"): c for c in doc.get("cases", []) if isinstance(c, dict)}
    for p in names:
        c = cases.get(p)
        if c is None:
            why.append(f"{p}: no case")
            continue
        for k, v in c.items():
            if k.startswith("check_") and v is not True:
                why.append(f"{p}: {k} false")
        want = p1[p][:max_tokens]
        plain = c.get("plain", [])
        if plain != want:
            why.append(f"{p}: plain differs from P1's at {first_diff(plain, want)}")
        loop = c.get("loop", [])
        if len(loop) < max_tokens - 1 or loop != want[:len(loop)]:
            why.append(f"{p}: loop ids ({len(loop)}) are not a prefix of P1's (first difference {first_diff(loop, want[:len(loop)])})")
        if c.get("loop_completion_tokens") != max_tokens or c.get("loop_finish") != "length":
            why.append(f"{p}: loop completion {c.get('loop_completion_tokens')}, finish {c.get('loop_finish')}")
    return why


def judge_health_idle(st, h, context, model_id):
    """Criterion 5, idle half."""
    want = {"status": "ok", "slots": 1, "busy": False, "busy_slots": 0, "queued": 0, "context_length": context, "model": model_id}
    if st != 200 or not isinstance(h, dict):
        return [f"/health answered {st}"]
    return [f"/health {k} = {h.get(k)!r}, expected {v!r}" for k, v in want.items() if h.get(k) != v]


def judge_refusal(name, st, body):
    """Criterion 8 for one request."""
    e = body.get("error") if isinstance(body, dict) else None
    if st != 400:
        return [f"refuse {name}: HTTP {st}"]
    if not isinstance(e, dict) or not isinstance(e.get("message"), str) or not e["message"] or e.get("type") != "invalid_request_error" \
            or "param" not in e or "code" not in e:
        return [f"refuse {name}: body is not the front end's error shape: {str(body)[:200]}"]
    return []


def judge_completion(name, st, resp, prompt_ids, ids, max_tokens):
    """Criterion 6 for one request."""
    if st != 200 or not isinstance(resp, dict):
        return [f"completion {name}: HTTP {st}: {str(resp)[:200]}"]
    f = resp.get("flerp") or {}
    why = []
    if f.get("prompt_token_ids") != prompt_ids:
        why.append(f"completion {name}: prompt ids differ from P1's at {first_diff(f.get('prompt_token_ids') or [], prompt_ids)}")
    if f.get("token_ids") != ids[:max_tokens]:
        why.append(f"completion {name}: token ids differ from P1's at {first_diff(f.get('token_ids') or [], ids[:max_tokens])}")
    fin = (resp.get("choices") or [{}])[0].get("finish_reason")
    if fin != "length":
        why.append(f"completion {name}: finish_reason {fin}")
    return why


def judge_chats(run, ref, world1):
    """Criterion 7 for one run's chats. run/world1: {name: {"status", "prompt_ids", "token_ids"}}; ref: {name: ids}."""
    why = []
    for name, r in run.items():
        if r.get("status") != 200:
            why.append(f"chat {name}: HTTP {r.get('status')}")
            continue
        if world1 is not None and r.get("prompt_ids") != (world1.get(name) or {}).get("prompt_ids"):
            why.append(f"chat {name}: prompt ids differ from the world-1 run's")
        want = ref.get(name)
        if want is None:
            why.append(f"chat {name}: no reference")
        elif r.get("token_ids") != want:
            why.append(f"chat {name}: token ids differ from the reference at {first_diff(r.get('token_ids') or [], want)}")
    return why


def judge_exit(what, rc, log_text):
    """Criterion 9 / 10: exit status 0, no signal, no segfault."""
    why = []
    if rc is None:
        why.append(f"{what}: did not exit within the shutdown timeout")
    elif rc != 0:
        why.append(f"{what}: exit status {rc}" + (" (a signal)" if rc < 0 or rc > 128 else ""))
    if "Segmentation fault" in (log_text or ""):
        why.append(f"{what}: 'Segmentation fault' in its log")
    return why


# --- host ---------------------------------------------------------------------------------------------------------
def load_json(path):
    with open(os.path.join(ROOT, path)) as f:
        return json.load(f)


def save(ctx, name, doc):
    with open(os.path.join(ctx["res"], name), "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")


def http(port, path, body=None, timeout=30):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            st = r.status
    except urllib.error.HTTPError as e:
        raw, st = e.read(), e.code
    except (urllib.error.URLError, OSError) as e:
        return None, str(e)
    try:
        return st, json.loads(raw)
    except ValueError:
        return st, raw[:400].decode(errors="replace")


class Proc:
    """One process of a run under ionice / nice / taskset (exec'd: its pid is the program's), stdout and stderr
    into `log`; JSON lines are kept as events."""

    def __init__(self, ctx, program, args, log_path):
        h = ctx["cfg"]["host"]
        cmd = list(h["polite"]) + ["taskset", "-c", h["cpus"], binary(ctx, program)] + args
        self.events, self.log_path = [], log_path
        self.log = open(log_path, "ab")
        self.log.write(("$ " + " ".join(cmd) + "\n").encode())
        self.log.flush()
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=ctx["S"].die_with_parent,
                                     env=dict(os.environ, **ctx["G"].runtime_env(ctx)))
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.proc.stdout:
            self.log.write(line)
            self.log.flush()
            s = line.strip()
            if s.startswith(b"{"):
                try:
                    e = json.loads(s)
                    e["_seen"] = time.time()
                    self.events.append(e)
                except ValueError:
                    pass

    def has(self, kind):
        return any(e.get("event") == kind for e in self.events)

    def term(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)

    def wait(self, timeout):
        try:
            rc = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            rc = None
        if rc is None:
            self.proc.kill()
            self.proc.wait()
        self.thread.join(timeout=10)
        self.log.close()
        return rc

    def text(self):
        with open(self.log_path, errors="replace") as f:
            return f.read()


def binary(ctx, name):
    return os.path.join(ROOT, "build", ctx["recipe"].name, name)


def memory_ok(ctx, plan_bytes, stages):
    h, m = ctx["cfg"]["host"], ctx["P"].memory_now()
    free = min(m["gtt_total"] - m["gtt_used"], m["mem_available"])
    need = plan_bytes + stages * h["arena_gib"] * GIB + h["margin_gib"] * GIB
    return free >= need, dict(m, free_bytes=free, need_bytes=int(need))


def note_loads(ctx, proc, label):
    for e in proc.events:
        if e.get("event") == "loaded":
            pl = e.get("placed") or {}
            ctx["disk"].append({"run": f"{label} rank {e.get('rank', 0)}", "bytes": pl.get("tensor_bytes", 0), "seconds": pl.get("fill_s"),
                                "when": ctx["S"].utc()})


def prompts(ctx):
    """{name: text} of the four gate prompts (gufo_recipe.prompts: the long filler, then the text), their P1 ids
    and P1's generated ids."""
    pc = ctx["S"].load_config(os.path.join(ROOT, ctx["cfg"]["model"]["prompts_config"]))
    texts = {p["name"]: (ctx["S"].long_text(pc["run"]["long_prompt_sentences"]) if p.get("long") else "") + p["text"] for p in pc["prompt"]}
    ids = load_json(ctx["cfg"]["model"]["prompt_ids"])
    p1 = {k: v["ids"] for k, v in load_json(ctx["cfg"]["model"]["p1_results"])["traced"].items()}
    names = list(ids)
    return names, texts, ids, p1


def end_ids(gguf):
    """The model text's end ids: the tokenizer's stop tokens (eos and <|endoftext|>), from the GGUF header."""
    import gguf_census as C
    h = C.read_header(gguf, keep_arrays=True)
    m = h["metadata"]
    tokens = m.get("tokenizer.ggml.tokens")
    tokens = tokens.get("values", tokens) if isinstance(tokens, dict) else tokens
    out = [int(m["tokenizer.ggml.eos_token_id"])]
    if isinstance(tokens, list) and "<|endoftext|>" in tokens and tokens.index("<|endoftext|>") not in out:
        out.append(tokens.index("<|endoftext|>"))
    return out


def executor_keys(ctx, gguf, world, peers, prefix):
    cfg, run = ctx["cfg"], ctx["cfg"]["run"]
    kv = {"weights": gguf, "context": run["context"], "world": world, "placement": run["placement"], "epoch": run["epoch"],
          "single_reader": "on", "pause_ms": cfg["host"]["pause_ms"]}
    if peers:
        kv["peers"] = peers
    return [f"{prefix}{k}={v}" for k, v in kv.items()]


# --- stages ---------------------------------------------------------------------------------------------------------
def stage_check(ctx):
    log = os.path.join(ctx["res"], "check.txt")
    open(log, "w").close()
    rc, _ = ctx["S"].sh([sys.executable, "flerp", "check"], log)
    with open(log) as f:
        text = f.read()
    ok = rc == 0 and " 0 error(s)" in text
    return {"pass": ok, "rc": rc, "evidence": "check.txt", "failures": [] if ok else ["flerp check reported errors"]}


def stage_build(ctx):
    cfg, name = ctx["cfg"], ctx["recipe"].name
    log = os.path.join(ctx["res"], "build.txt")
    open(log, "w").close()
    rc, _ = ctx["S"].sh([sys.executable, "flerp", "assemble", name], log)
    out = {"evidence": "build.txt, assembly.json"}
    if rc != 0:
        return dict(out, rc=rc, failures=["flerp assemble failed"])
    tree = os.path.join(ROOT, "build", name)
    import shutil
    shutil.copyfile(os.path.join(tree, "assembly.json"), os.path.join(ctx["res"], "assembly.json"))
    cmd = ctx["G"].polite(cfg) + [os.path.join(tree, "build.sh"), f"-j{cfg['host'].get('build_jobs', 2)}"] + ctx["G"].openssl_options(ctx)
    rc, secs = ctx["S"].sh(cmd, log, timeout=10800)
    out.update(rc=rc, build_s=round(secs, 1), binaries={})
    for b in ctx["recipe"].binaries:
        p = binary(ctx, b)
        if os.path.isfile(p):
            out["binaries"][b] = {"sha256": ctx["S"].sha256(p), "bytes": os.path.getsize(p)}
    why = [] if rc == 0 else [f"build.sh exited {rc}"]
    why += [f"{b} was not built" for b in ctx["recipe"].binaries if b not in out["binaries"]]
    out.update(failures=why, **{"pass": not why})
    return out


def stage_conformance(ctx):
    cfg, run = ctx["cfg"], ctx["cfg"]["run"]
    gguf = cfg["model"]["gguf"]
    log = os.path.join(ctx["res"], "conformance.txt")
    open(log, "w").close()
    names, _, ids, p1 = prompts(ctx)
    files = {p: ctx["P"].write_ids(ctx, p, ids[p]) for p in names}
    ends = ",".join(str(i) for i in end_ids(gguf))
    plan = ctx["P"].py_plan(gguf)
    out = {"evidence": "conformance.txt, conformance-<run>.json, conformance-lease.txt", "runs": {}, "end_ids": ends}
    why = []
    hold = ctx["S"].Hold(cfg, os.path.join(ctx["res"], "conformance-lease.txt"), out, wait_s=cfg["host"]["gpu_wait_s"])
    try:
        for k, rname in enumerate(run["conformance_runs"]):
            world, peers = (1, None) if rname == "world1" else (2, "loopback")
            (hold.between if k else hold.take)("strixflerp gate serve-stage", f"conformance {rname}")
            ok, mem = memory_ok(ctx, plan["ranks"][0]["device_bytes"], world)
            if not ok:
                why.append(f"conformance {rname}: not enough free memory: {mem}")
                continue
            ctx["P"].wait_for_load(cfg, f"conformance {rname}", log)
            res = os.path.join(ctx["res"], f"conformance-{rname}.json")
            args = ["--out", res] + executor_keys(ctx, gguf, world, peers, "")
            for p in names:
                args += ["--case", p, "--ids", files[p], "--paths", "plain,loop", "--new", str(run["max_tokens"]), "--max-tokens", str(run["max_tokens"]),
                         "--end-ids", ends, "--no-snapshot"]
            t0 = time.time()
            proc = Proc(ctx, "flerp-executor-conformance", args, log)
            rc = proc.wait(run["run_timeout_s"])
            note_loads(ctx, proc, f"conformance {rname}")
            try:
                with open(res) as f:
                    doc = json.load(f)
            except (OSError, ValueError) as e:
                doc = {"error": f"no result: {e}"}
            w = judge_conformance(doc, p1, names, run["context"], run["max_tokens"])
            w += judge_exit(f"conformance {rname}", rc, proc.text())
            loaded = next((e for e in proc.events if e.get("event") == "loaded"), {})
            pl = loaded.get("placed") or {}
            if pl.get("allocated_bytes") != plan["ranks"][0]["device_bytes"]:
                w.append(f"load: allocated {pl.get('allocated_bytes')} bytes, the plan has {plan['ranks'][0]['device_bytes']}")
            if not (pl.get("single_reader") and pl.get("direct_io")):
                w.append("load: not the single direct reader")
            why += [f"conformance {rname}: {x}" for x in w]
            out["runs"][rname] = {"rc": rc, "seconds": round(time.time() - t0, 1), "memory_before": mem, "caps": doc.get("caps"),
                                  "cases": {c.get("name"): {"plain_equal_p1": c.get("plain") == p1[c.get("name")][:run["max_tokens"]],
                                                            "loop_ids": len(c.get("loop", []))} for c in doc.get("cases", []) if c.get("name") in p1},
                                  "load_s": loaded.get("load_s"), "failures": w}
    except RuntimeError as e:
        out["error"] = str(e)
    finally:
        hold.give()
    out["failures"] = why
    out["pass"] = "error" not in out and not why and len(out["runs"]) == len(run["conformance_runs"])
    return out


def wait_up(proc, port, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.proc.poll() is not None:
            return False, round(time.time() - t0, 1)
        st, h = http(port, "/health", timeout=3)
        if st in (200, 503):
            return st == 200, round(time.time() - t0, 1)
        time.sleep(2)
    return False, round(time.time() - t0, 1)


def start_server(ctx, gguf, world, peers, port, log, label):
    """The front end (rank 0) and, for TCP, flerp-qwen35moe-stage (rank 1) started after rank 0 has loaded.
    Returns ([procs], up, seconds)."""
    cfg, run = ctx["cfg"], ctx["cfg"]["run"]
    gen = os.path.join(ctx["scratch"], "generation_config.json")
    with open(gen, "w") as f:
        json.dump(run["generation_config"], f)
    ctx["P"].wait_for_load(cfg, f"{label} rank 0", log)
    args = ["--host", "127.0.0.1", "--port", str(port), "--model-id", cfg["model"]["model_id"], "--generation-config", gen]
    args += executor_keys(ctx, gguf, world, peers, "executor.") + [f"text.weights={gguf}", f"text.context={run['context']}"]
    procs = [Proc(ctx, ctx["front"], args, log)]
    t0 = time.time()
    if peers and peers != "loopback":
        while not (procs[0].has("waiting") or procs[0].has("loaded")) and procs[0].proc.poll() is None and time.time() - t0 < run["startup_timeout_s"]:
            time.sleep(0.5)
        while not procs[0].has("waiting") and procs[0].proc.poll() is None and time.time() - t0 < run["startup_timeout_s"]:
            time.sleep(0.5)
        ctx["P"].wait_for_load(cfg, f"{label} rank 1", log)
        procs.append(Proc(ctx, "flerp-qwen35moe-stage", executor_keys(ctx, gguf, world, peers, "") + ["rank=1"], log + ".rank1"))
    up, secs = wait_up(procs[0], port, run["startup_timeout_s"] - (time.time() - t0))
    return procs, up, round(time.time() - t0, 1)


def stop_all(ctx, procs, label):
    """SIGTERM to every process at once; criterion 9."""
    for p in procs:
        p.term()
    t0 = time.time()
    why, rcs = [], {}
    for i, p in enumerate(procs):
        rc = p.wait(max(1, ctx["cfg"]["run"]["shutdown_timeout_s"] - (time.time() - t0)))
        rcs[i] = rc
        why += judge_exit(f"{label} rank {i}", rc, p.text())
    if "shutdown: the executor is closed" not in procs[0].text():
        why.append(f"{label}: the front end did not print that the executor is closed")
    return why, rcs, round(time.time() - t0, 1)


def serve_run(ctx, rname):
    """One front end run ([run] serve_runs): criteria 4 to 9 except the chats' reference (judged by judge_chats)."""
    cfg, run = ctx["cfg"], ctx["cfg"]["run"]
    gguf, port = cfg["model"]["gguf"], run["http_port"]
    world, peers = {"world1": (1, None), "loopback2": (2, "loopback"), "tcp2": (2, f"127.0.0.1:{run['relay_port']}")}[rname]
    log = os.path.join(ctx["res"], f"serve-{rname}.txt")
    open(log, "w").close()
    names, texts, ids, p1 = prompts(ctx)
    out = {"evidence": f"serve-{rname}.txt, serve-{rname}.json, serve-{rname}-lease.txt"}
    doc = {"run": rname, "health": {}, "completions": {}, "chats": {}, "refusals": {}}
    why = []
    if rname == "tcp2":
        cuts = ctx["P"].py_plan(gguf, 2)["cuts"]
        plan_bytes = sum(ctx["P"].py_plan(gguf, 1, [cuts[r], cuts[r + 1]])["ranks"][0]["device_bytes"] for r in range(2))
    else:
        plan_bytes = ctx["P"].py_plan(gguf)["ranks"][0]["device_bytes"]
    hold = ctx["S"].Hold(cfg, os.path.join(ctx["res"], f"serve-{rname}-lease.txt"), out, wait_s=cfg["host"]["gpu_wait_s"])
    procs = []
    try:
        hold.take("strixflerp gate serve-stage", f"serve {rname}")
        ok, mem = memory_ok(ctx, plan_bytes, world)
        out["memory_before"] = mem
        if not ok:
            out["not_run"] = f"not enough free memory for {rname}: {mem}"
            return out
        procs, up, secs = start_server(ctx, gguf, world, peers, port, log, rname)
        out["startup_s"] = secs
        if not up:
            why.append(f"{rname}: the front end did not answer /health 200 within {run['startup_timeout_s']} s")
            raise RuntimeError(why[-1])
        st, h = http(port, "/health")
        doc["health"]["idle"] = h
        why += judge_health_idle(st, h, run["context"], cfg["model"]["model_id"])
        # completions: the gate prompts; /health polled while the long one runs
        for p in names:
            body = {"model": cfg["model"]["model_id"], "prompt": texts[p], "max_tokens": run["max_tokens"], "flerp_trace": True}
            seen, done = [], threading.Event()

            def poll():
                while not done.is_set():
                    s2, h2 = http(port, "/health", timeout=5)
                    if isinstance(h2, dict):
                        seen.append({k: h2.get(k) for k in ("busy", "busy_slots", "queued", "status")})
                    time.sleep(0.2)
            th = threading.Thread(target=poll, daemon=True) if p == "records" else None
            if th:
                th.start()
            st, resp = http(port, "/v1/completions", body, timeout=run["request_timeout_s"])
            done.set()
            if th:
                th.join(10)
                doc["health"]["during_records"] = seen
                if not any(s.get("busy") is True and s.get("busy_slots") == 1 for s in seen):
                    why.append(f"{rname}: no /health answer during the records completion had busy true and busy_slots 1")
            why += [f"{rname}: {x}" for x in judge_completion(p, st, resp, ids[p], p1[p], run["max_tokens"])]
            f = (resp.get("flerp") or {}) if isinstance(resp, dict) else {}
            doc["completions"][p] = {"status": st, "prompt_ids_equal_p1": f.get("prompt_token_ids") == ids[p],
                                     "token_ids": f.get("token_ids"), "timings": resp.get("timings") if isinstance(resp, dict) else None}
        # chats
        for c in cfg["chat"]:
            msgs = c.get("messages") or [{"role": "user", "content": texts[c["prompt"]]}]
            body = {"model": cfg["model"]["model_id"], "messages": msgs, "max_tokens": run["max_tokens"], "flerp_trace": True}
            if c.get("chat_template_kwargs"):
                body["chat_template_kwargs"] = c["chat_template_kwargs"]
            st, resp = http(port, "/v1/chat/completions", body, timeout=run["request_timeout_s"])
            f = (resp.get("flerp") or {}) if isinstance(resp, dict) else {}
            doc["chats"][c["name"]] = {"status": st, "prompt_ids": f.get("prompt_token_ids"), "token_ids": f.get("token_ids"),
                                       "finish": ((resp.get("choices") or [{}])[0].get("finish_reason") if isinstance(resp, dict) else None),
                                       "text": (((resp.get("choices") or [{}])[0].get("message") or {}).get("content") if isinstance(resp, dict) else None),
                                       "timings": resp.get("timings") if isinstance(resp, dict) else None,
                                       "error": None if st == 200 else str(resp)[:300]}
            if st != 200:
                why.append(f"{rname}: chat {c['name']}: HTTP {st}: {str(resp)[:200]}")
        # refusals
        for r in cfg["refuse"]:
            st, resp = http(port, r["path"], r["body"], timeout=60)
            doc["refusals"][r["name"]] = {"status": st, "body": resp}
            why += [f"{rname}: {x}" for x in judge_refusal(r["name"], st, resp)]
        st, h = http(port, "/health")
        doc["health"]["after"] = h
        why += [f"{rname} (after the requests): {x}" for x in judge_health_idle(st, h, run["context"], cfg["model"]["model_id"])]
    except RuntimeError as e:
        out["error"] = str(e)
    finally:
        if procs:
            w, rcs, secs = stop_all(ctx, procs, rname)
            why += w
            out["shutdown"] = {"rc": rcs, "seconds": secs}
            for p in procs:
                note_loads(ctx, p, f"serve {rname}")
            doc["loaded"] = [{k: v for k, v in e.items() if k != "_seen"} for p in procs for e in p.events if e.get("event") == "loaded"]
        hold.give()
    save(ctx, f"serve-{rname}.json", doc)
    out.update(failures=why, health=doc["health"].get("idle"))
    out["pass"] = "error" not in out and not why
    return out


def chat_reference(ctx):
    path = os.path.join(ctx["res"], "reference.json")
    return load_json(os.path.relpath(path, ROOT))["ids"] if os.path.isfile(path) else {}


def serve_stage(ctx, stage):
    rname = SERVE_STAGE[stage]
    if rname not in ctx["cfg"]["run"]["serve_runs"]:
        return {"not_run": "not in [run] serve_runs", "pass": False}
    out = serve_run(ctx, rname)
    if stage != "serve1" and "not_run" not in out:
        doc = load_json(os.path.relpath(os.path.join(ctx["res"], f"serve-{rname}.json"), ROOT))
        w1p = os.path.join(ctx["res"], "serve-world1.json")
        w1 = load_json(os.path.relpath(w1p, ROOT))["chats"] if os.path.isfile(w1p) else None
        w = judge_chats(doc["chats"], chat_reference(ctx), w1)
        out["failures"] = out.get("failures", []) + [f"{rname}: {x}" for x in w]
        out["pass"] = out["pass"] and not w
    return out


def stage_reference(ctx):
    """Stage P2's generator (world 1, graph on) on the chat prompt ids serve1 rendered; judges serve1's chats."""
    cfg, run = ctx["cfg"], ctx["cfg"]["run"]
    w1p = os.path.join(ctx["res"], "serve-world1.json")
    if not os.path.isfile(w1p):
        return {"error": "serve1 has not run", "pass": False}
    chats = load_json(os.path.relpath(w1p, ROOT))["chats"]
    names = [n for n in chats if chats[n].get("prompt_ids")]
    files = [ctx["P"].write_ids(ctx, "chat-" + n, chats[n]["prompt_ids"]) for n in names]
    log = os.path.join(ctx["res"], "reference.txt")
    open(log, "w").close()
    out = {"evidence": "reference.txt, reference.json, reference-lease.txt"}
    hold = ctx["S"].Hold(cfg, os.path.join(ctx["res"], "reference-lease.txt"), out, wait_s=cfg["host"]["gpu_wait_s"])
    ids = {}
    try:
        hold.take("strixflerp gate serve-stage", "reference")
        ok, mem = memory_ok(ctx, ctx["P"].py_plan(cfg["model"]["gguf"])["ranks"][0]["device_bytes"], 1)
        out["memory_before"] = mem
        if not ok:
            raise RuntimeError(f"not enough free memory: {mem}")
        ctx["P"].wait_for_load(cfg, "reference", log)
        args = ["--staged", "loopback", "--model", cfg["model"]["gguf"], "--context", str(run["context"]), "--max-tokens", str(run["max_tokens"]),
                "--configs", "1:even:on", "--single-reader", "--pause-ms", str(cfg["host"]["pause_ms"])] + [x for f in files for x in ("--ids-file", f)]
        proc = Proc(ctx, "flerp-qwen35moe-generate", args, log)
        rc = proc.wait(run["run_timeout_s"])
        note_loads(ctx, proc, "reference")
        for e in proc.events:
            if e.get("event") == "staged":
                ids[names[e["prompt"]]] = e["ids"]
        out["rc"] = rc
    except RuntimeError as e:
        out["error"] = str(e)
    finally:
        hold.give()
    save(ctx, "reference.json", {"ids": ids, "prompt_tokens": {n: len(chats[n]["prompt_ids"]) for n in names}})
    why = [] if out.get("rc") == 0 else [f"the reference run exited {out.get('rc')}"]
    why += [f"reference: no result for {n}" for n in names if n not in ids]
    why += [f"world1: {x}" for x in judge_chats(chats, ids, None)]
    out.update(failures=why, chats={n: {"equal": chats[n].get("token_ids") == ids.get(n), "tokens": len(ids.get(n) or [])} for n in names})
    out["pass"] = "error" not in out and not why
    return out


def stage_shutdown(ctx):
    """Criterion 10 on the dense model over TCP."""
    cfg, run = ctx["cfg"], ctx["cfg"]["run"]
    gguf, port = cfg["model"]["dense_gguf"], run["http_port"] + 1
    peers = f"127.0.0.1:{run['relay_port'] + 10}"
    out = {"evidence": "shutdown-<case>.txt, shutdown.json, shutdown-lease.txt", "cases": {}}
    why = []
    body = {"prompt": "The capital of France is", "max_tokens": 8, "flerp_trace": True}
    hold = ctx["S"].Hold(cfg, os.path.join(ctx["res"], "shutdown-lease.txt"), out, wait_s=cfg["host"]["gpu_wait_s"])
    try:
        hold.take("strixflerp gate serve-stage", "shutdown cases")
        for case in ("a", "b", "c"):
            log = os.path.join(ctx["res"], f"shutdown-{case}.txt")
            open(log, "w").close()
            procs, up, secs = start_server(ctx, gguf, 2, peers, port, log, f"shutdown {case}")
            rec = {"startup_s": secs}
            w = []
            if not up:
                w.append("the front end did not come up")
                for p in procs:
                    p.term()
                    p.wait(30)
            else:
                timeout = run["shutdown_timeout_s"]
                if case in ("a", "c"):
                    st, resp = http(port, "/v1/completions", body, timeout=300)
                    rec["completion"] = st
                    if st != 200:
                        w.append(f"the completion answered {st}: {str(resp)[:200]}")
                    targets = procs[:1] if case == "a" else procs
                    for p in targets:
                        p.term()
                    t0 = time.time()
                    rcs = [p.wait(max(1, timeout - (time.time() - t0))) for p in procs]
                    for i, (p, rc) in enumerate(zip(procs, rcs)):
                        w += judge_exit(f"rank {i}", rc, p.text())
                    if case == "a" and not procs[1].has("closed"):
                        w.append("rank 1 did not print closed")
                else:
                    procs[1].term()
                    rc1 = procs[1].wait(timeout)
                    w += judge_exit("rank 1", rc1, procs[1].text())
                    if not procs[1].has("terminated"):
                        w.append("rank 1 did not print terminated")
                    st, resp = http(port, "/v1/completions", body, timeout=120)
                    code = ((resp or {}).get("error") or {}).get("code") if isinstance(resp, dict) else None
                    rec["after_rank1"] = {"status": st, "code": code}
                    if st != 500 or code != "executor_failed":
                        w.append(f"the request after rank 1 ended answered {st} {code}, expected 500 executor_failed")
                    st, h = http(port, "/health")
                    rec["health_after"] = {"status": st, "body": h}
                    if st != 503 or not isinstance(h, dict) or h.get("status") != "failed":
                        w.append(f"/health after the failure answered {st} {h if not isinstance(h, dict) else h.get('status')}")
                    procs[0].term()
                    rc0 = procs[0].wait(timeout)
                    w += judge_exit("rank 0", rc0, procs[0].text())
                    rcs = [rc0, rc1]
                rec["rc"] = rcs
            for p in procs:
                note_loads(ctx, p, f"shutdown {case}")
            rec["failures"] = w
            out["cases"][case] = rec
            why += [f"case {case}: {x}" for x in w]
            hold.between("strixflerp gate serve-stage", f"shutdown after {case}")
    except RuntimeError as e:
        out["error"] = str(e)
    finally:
        hold.give()
    save(ctx, "shutdown.json", out["cases"])
    out["failures"] = why
    out["pass"] = "error" not in out and not why and len(out["cases"]) == 3
    return out


# --- summary ----------------------------------------------------------------------------------------------------------
def write_summary(ctx, results):
    SM.write_json(ctx["res"], {"stages": results, "disk_reads": ctx["disk_all"]}, ctx=ctx)
    utc = ctx["S"].utc
    L = [f"# Gate of the executor ABI over the stages: {ctx['recipe'].name}", "",
         f"Host {os.uname().nodename}, written {utc()}. Criteria: `[criteria]` of `{ctx['config']}` (committed before the first GPU run). "
         "Stage P3 of `docs/PHASE5-FOURNODE-DESIGN.md`.", "",
         "| stage | result | when (UTC) | failures |", "|---|---|---|---|"]
    for s in STAGES:
        r = results.get(s)
        if r:
            res = "NOT RUN" if r.get("not_run") else "PASS" if r.get("pass") else "FAIL"
            L.append(f"| {s} | {res} | {r.get('when_utc', '')} | {len(r.get('failures', []))} {r.get('error', '') or r.get('not_run', '')} |")
    for s in STAGES:
        fl = (results.get(s) or {}).get("failures") or []
        if fl:
            L += ["", f"Failures of `{s}` (first 20):", ""] + [f"- {x}" for x in fl[:20]]
    for s in ("serve1", "serve2", "tcp"):
        r = results.get(s) or {}
        if r.get("health"):
            L += ["", f"`/health` of `{s}` while idle:", "", "```", json.dumps(r["health"]), "```"]
        if r.get("shutdown"):
            L.append(f"\n`{s}` SIGTERM: exit status {r['shutdown']['rc']} in {r['shutdown']['seconds']} s")
    if (results.get("shutdown") or {}).get("cases"):
        L += ["", "## Shutdown cases (dense model, TCP world 2)", ""]
        for c, d in results["shutdown"]["cases"].items():
            L.append(f"- {c}: exit status {d.get('rc')}, failures {d.get('failures')}")
    L += ["", "## Disk reads", "", "| run | GB | fill s | when |", "|---|---|---|---|"]
    tot = 0
    for d in ctx["disk_all"]:
        tot += d["bytes"] or 0
        L.append(f"| {d['run']} | {(d['bytes'] or 0) / 1e9:.2f} | {d['seconds']} | {d['when']} |")
    L.append(f"| total | {tot / 1e9:.2f} | | |")
    holds = [h for s in STAGES for h in (results.get(s) or {}).get("holds", [])]
    if holds:
        L += ["", "## GPU holds", "", "| for | from | to | seconds |", "|---|---|---|---|"]
        L += [f"| {h['for']} | {h['from']} | {h['to']} | {h['seconds']} |" for h in holds]
    SM.write_md(ctx["res"], L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="gate/qwen36-35b-a3b.flerp-serve-stage.toml")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--keep-going", action="store_true")
    a = ap.parse_args()
    import gufo_recipe as G                   # noqa: E402  (needs the catalogue; the judgements above do not)
    import moe_placed as P                    # noqa: E402
    import strixite_recipe as S               # noqa: E402
    from flerplib.catalog import Catalog
    P.G, P.S = G, S
    cfg = S.load_config(os.path.join(ROOT, a.config))
    cat = Catalog(ROOT)
    recipe = cat.recipe(cfg["recipe"])
    res = os.path.join(ROOT, "gate", "results", recipe.name)
    os.makedirs(res, exist_ok=True)
    ctx = {"cat": cat, "recipe": recipe, "cfg": cfg, "res": res, "scratch": os.path.join(ROOT, cfg["host"]["scratch"]), "config": a.config,
           "disk": [], "G": G, "P": P, "S": S, "front": next(iter(recipe.binaries))}
    os.makedirs(ctx["scratch"], exist_ok=True)
    results, disk_all, prior = {}, [], os.path.join(res, "summary.json")
    if os.path.isfile(prior):
        with open(prior) as f:
            old = json.load(f)
        results, disk_all = old.get("stages", {}), old.get("disk_reads", [])
    ctx["disk_all"] = disk_all
    fns = {"check": stage_check, "build": stage_build, "conformance": stage_conformance, "reference": stage_reference,
           "shutdown": stage_shutdown}
    want = a.stages.split(",")
    for s in STAGES:
        if s not in want:
            continue
        print(f"== {s}", flush=True)
        try:
            results[s] = fns[s](ctx) if s in fns else serve_stage(ctx, s)
        except RuntimeError as e:
            results[s] = {"error": str(e)}
        results[s]["when_utc"] = S.utc()
        ctx["disk_all"] += ctx["disk"]
        ctx["disk"] = []
        print(f"   {'PASS' if results[s].get('pass') else 'FAIL'} {results[s].get('error', '') or results[s].get('not_run', '')}", flush=True)
        for x in (results[s].get("failures") or [])[:10]:
            print(f"     - {x}", flush=True)
        write_summary(ctx, results)
        if not results[s].get("pass") and not a.keep_going:
            break
    return 0 if all(results.get(s, {}).get("pass") for s in want) else 1


if __name__ == "__main__":
    sys.exit(main())
