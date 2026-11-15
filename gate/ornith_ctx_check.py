#!/usr/bin/env python3
"""ORNITH-CTX-RUN (card 266647f0af): the Ornith serve configuration's context (configs/ornith397.serve.toml, 65536) proved on the 35B proxy.
World 4 over loopback TCP on control-node's one GPU: the serve front end (flerp-serve rank 0 + flerp-qwen35moe-stage ranks 1-3), ONE load, a
FreeFitty-shaped ~20K-token thread through /v1/chat/completions, then the same text as a raw prompt through /v1/completions against
flerp-qwen35moe-generate --staged loopback at the same context (the generator path). Criteria: gate/ornith-ctx.toml (committed before the run).
Run on control-node through tools/gpu-queue.sh (gate/ornith_ctx_check_run.sh); evidence in gate/results/ornith-ctx/."""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gate"))
import flerplib.configs as CF  # noqa: E402
import ornith_config_check as OC  # noqa: E402

RES = os.path.join(ROOT, "gate", "results", "ornith-ctx")
TOPICS = ["invoice reconciliation", "the nightly backup window", "a customer refund dispute", "rotating the signing keys", "the quarterly capacity review",
          "an on-call handover", "migrating the build farm", "vendor contract renewal", "the staging database refresh", "a flaky integration test",
          "onboarding a contractor", "the incident postmortem template"]


def para(i):
    t = TOPICS[i % len(TOPICS)]
    return (f"Section {i}. When the request concerns {t}, first read the thread to the end and quote the line you are acting on. "
            f"State the owner ({i % 7 + 1} candidates exist, named in the roster), the deadline ({i % 28 + 1} days out), and the risk in one sentence. "
            f"Never paste credentials; refer to the vault entry number {1000 + i * 7}. If the tool result is empty, say so and try the next tool before asking. "
            f"Keep the reply under {80 + i % 5 * 20} words unless the user asks for the full report, and end with the single next action.\n")


def tool(i):
    return {"type": "function", "function": {"name": f"tool_{i}_{TOPICS[i % len(TOPICS)].split()[-1]}",
            "description": f"Look up and update records about {TOPICS[i % len(TOPICS)]}; returns at most 20 rows as JSON with id, owner, status and due date.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "free text filter"},
                           "owner": {"type": "string"}, "status": {"type": "string", "enum": ["open", "blocked", "done"]},
                           "limit": {"type": "integer", "minimum": 1, "maximum": 20}}, "required": ["query"]}}}


def thread(n_sys, n_tools, n_turns):
    system = "You are the operations assistant for a small infrastructure team. Follow these rules.\n" + "".join(para(i) for i in range(n_sys)) + \
             "\nTools available (JSON):\n" + "\n".join(json.dumps(tool(i)) for i in range(n_tools))
    msgs = [{"role": "system", "content": system}]
    for k in range(n_turns):
        msgs.append({"role": "user", "content": f"Turn {k}: what is the status of {TOPICS[k % len(TOPICS)]}, and who owns the next step? Context so far is in the thread."})
        msgs.append({"role": "assistant", "content": f"Status of {TOPICS[k % len(TOPICS)]}: open, owner number {k % 7 + 1}, due in {k % 28 + 1} days. Next action: confirm the vault entry {1000 + k}."})
    msgs.append({"role": "user", "content": "Summarise where everything stands in two sentences."})
    return msgs


def flat(msgs):
    return "\n".join(f"<{m['role']}>\n{m['content']}" for m in msgs) + "\n<assistant>\n"


def post(port, path, obj, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(obj).encode(), {"content-type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read()[:400].decode(errors="replace")}, time.time() - t0


def gtt_kib(pid):
    """Sum of the process's amdgpu GTT KiB over its distinct DRM clients (/proc/PID/fdinfo)."""
    seen, tot = set(), 0
    for f in glob.glob(f"/proc/{pid}/fdinfo/*"):
        try:
            txt = open(f).read()
        except OSError:
            continue
        m = re.search(r"drm-client-id:\s*(\d+)", txt)
        g = re.search(r"drm-memory-gtt:\s*(\d+)\s*KiB", txt)
        if m and g and m.group(1) not in seen:
            seen.add(m.group(1))
            tot += int(g.group(1))
    return tot


def total_gtt():
    try:
        return max(int(open(p).read()) for p in glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used"))
    except (OSError, ValueError):
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=os.path.join(ROOT, "build", "flerp-p5-serve", "out"))
    ap.add_argument("--rocm", default="")
    ap.add_argument("--serve-config", default="ornith397.serve")
    ap.add_argument("--port", type=int, default=39710)
    ap.add_argument("--sys-sections", type=int, default=60)
    ap.add_argument("--tools", type=int, default=40)
    ap.add_argument("--turns", type=int, default=24)
    ap.add_argument("--target-tokens", type=int, default=20000)
    a = ap.parse_args()
    crit = tomllib.load(open(os.path.join(ROOT, "gate", "ornith-ctx.toml"), "rb"))
    os.makedirs(RES, exist_ok=True)
    cfg = CF.load(a.serve_config)
    ctx = CF.context(cfg, 4096)
    why = []
    if ctx != crit["config"]["context"]:
        why.append(f"{a.serve_config} context is {ctx}, not {crit['config']['context']}")
    n = CF.drafts(cfg)
    rocm = a.rocm or tomllib.load(open(os.path.join(ROOT, "toolchains.toml"), "rb"))["therock-10.1"]["rocm_root"]
    env = dict(os.environ, LD_LIBRARY_PATH=f"{rocm}/lib:{rocm}/lib/llvm/lib", **cfg["env"])
    gen = os.path.join(RES, "generation_config.json")
    json.dump({"temperature": 0}, open(gen, "w"))
    mtp = {"draft_head": "on", "mtp_data_file": OC.MODEL} if n else {}
    base = {"weights": OC.MODEL, "context": ctx, "world": 4, "placement": "even", "epoch": 7, "peers": f"127.0.0.1:{a.port + 20}",
            "single_reader": "on", "pause_ms": 0, "setup_deadline_ms": 600000, "deadline_ms": 120000}
    kv0 = dict(base, **({"drafts": n} if n else {}))
    procs, doc = [], {"config": a.serve_config, "context": ctx, "drafts": n, "env": cfg["env"]}

    def start(label, argv):
        lp = os.path.join(RES, f"{label}.jsonl")
        p = subprocess.Popen([os.path.join(a.build, argv[0])] + argv[1:], stdout=open(lp, "wb"), stderr=open(os.path.join(RES, f"{label}.err"), "wb"),
                             env=env, preexec_fn=OC.die_with_parent)
        procs.append(p)
        return p, lp

    front = ["strixflerp-qwen36-35b-a3b.flerp-serve-stage", "--host", "127.0.0.1", "--port", str(a.port), "--model-id", "m", "--generation-config", gen] + \
            (["--delivery", "tokens"] if cfg["stage"].get("reply") == "token" else []) + (["--drafts", str(n)] if n else []) + \
            [f"executor.{k}={v}" for k, v in kv0.items()] + [f"text.weights={OC.MODEL}", f"text.context={ctx}"]
    t0 = time.time()
    p0, l0 = start("r0", front)
    OC.wait_event(l0, p0, ("waiting", "error"), 900)
    for r in (1, 2, 3):
        kr = {k: v for k, v in base.items()}
        kr.update(mtp if r == 3 else {})
        p, lp = start(f"r{r}", ["flerp-qwen35moe-stage"] + [f"{k}={v}" for k, v in kr.items()] + [f"rank={r}"])
        ev = OC.wait_event(lp, p, ("loaded", "error"), 900)
        if not ev or ev[0].get("event") == "error":
            why.append(f"rank {r}: {ev}")
    up = False
    while time.time() - t0 < 1800 and not why:
        try:
            up = urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=3).status == 200
        except Exception:
            up = False
        if up or p0.poll() is not None:
            break
        time.sleep(2)
    doc["load_s"] = round(time.time() - t0, 1)
    pids = [p.pid for p in procs]
    idle = [gtt_kib(x) for x in pids]
    doc["gtt_total_idle_mib"] = total_gtt() // 2**20
    doc["gtt_idle_mib"] = [x // 1024 for x in idle]
    if up:
        try:
            doc["health"] = json.load(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=5))
        except Exception as e:
            doc["health"] = str(e)
        sys_n = a.sys_sections
        for _ in range(6):   # size the thread to the target with the server's own tokenizer
            msgs = thread(sys_n, a.tools, a.turns)
            st, d, _ = post(a.port, "/tokenize", {"content": flat(msgs)})
            ntok = len(d.get("tokens") or d.get("ids") or []) if st == 200 else 0
            if not ntok:
                break
            if abs(ntok - a.target_tokens) < 800:
                break
            sys_n = max(1, int(sys_n * a.target_tokens / ntok))
        doc["thread"] = {"sys_sections": sys_n, "tools": a.tools, "turns": a.turns, "tokens_by_tokenize": ntok, "tokenize_status": st}
        peak, stop = list(idle), threading.Event()

        def sample():
            while not stop.is_set():
                for i, x in enumerate(pids):
                    peak[i] = max(peak[i], gtt_kib(x))
                time.sleep(1)
        th = threading.Thread(target=sample, daemon=True)
        th.start()
        st, d, secs = post(a.port, "/v1/chat/completions", {"model": "m", "messages": msgs, "max_tokens": 48})
        stop.set()
        th.join()
        ch = (d.get("choices") or [{}])[0]
        doc["chat"] = {"status": st, "wall_s": round(secs, 2), "usage": d.get("usage"), "timings": d.get("timings"), "finish": ch.get("finish_reason"),
                       "text": (ch.get("message") or {}).get("content"), "error": d.get("error")}
        doc["gtt_peak_mib"] = [x // 1024 for x in peak]
        doc["gtt_growth_mib"] = [(p - i) // 1024 for p, i in zip(peak, idle)]
        doc["gtt_total_after_mib"] = total_gtt() // 2**20
        # same thread as raw text on the serve path, greedy
        raw = flat(msgs)
        open(os.path.join(RES, "prompt.txt"), "w").write(raw)
        # the generator path takes token ids (--ids-file): the serve front end's own tokenizer on the same text
        stt, dt, _ = post(a.port, "/tokenize", {"content": raw})
        pids = dt.get("tokens") or []
        open(os.path.join(RES, "prompt.ids"), "w").write(" ".join(str(i) for i in pids))
        doc["raw_prompt_tokens"] = len(pids)
        mt = crit["identity"]["max_tokens"]
        st2, d2, secs2 = post(a.port, "/v1/completions", {"model": "m", "prompt": raw, "max_tokens": mt, "flerp_trace": True})
        doc["serve_raw"] = {"status": st2, "wall_s": round(secs2, 2), "timings": d2.get("timings"), "ids": (d2.get("flerp") or {}).get("token_ids"),
                            "text": ((d2.get("choices") or [{}])[0]).get("text"), "usage": d2.get("usage"), "error": d2.get("error")}
        # a short request after the long one: the slot must be reusable
        st3, d3, _ = post(a.port, "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 8})
        doc["short_after"] = {"status": st3, "error": d3.get("error")}
    for p in procs:
        p.terminate()
    doc["exit"] = []
    for p in procs:
        try:
            doc["exit"].append(p.wait(120))
        except subprocess.TimeoutExpired:
            p.kill()
            doc["exit"].append("killed")
    # generator path at the same context: flerp-qwen35moe-generate --staged loopback, one load, the same prompt as ids, no drafts
    if up and doc.get("serve_raw", {}).get("ids"):
        out = os.path.join(RES, "generator.jsonl")
        with open(out, "w") as fo, open(out + ".err", "w") as fe:
            gp = subprocess.run([os.path.join(a.build, "flerp-qwen35moe-generate"), "--staged", "loopback", "--model", OC.MODEL, "--context", str(ctx),
                                 "--max-tokens", str(crit["identity"]["max_tokens"]), "--configs", "4:even:on", "--single-reader", "--pause-ms", "100",
                                 "--ids-file", os.path.join(RES, "prompt.ids")], stdout=fo, stderr=fe, env=env, timeout=3600)
        ev = [json.loads(x) for x in open(out) if x.startswith("{")]
        sg = next((e for e in ev if e.get("event") == "staged"), None)
        doc["generator"] = {"rc": gp.returncode, "ids": sg and sg.get("ids"), "timings": sg and sg.get("timings"), "events": [e.get("event") for e in ev]}
    # judge against gate/ornith-ctx.toml
    c = doc.get("chat", {})
    if not up:
        why.append("serve front end did not come up")
    else:
        pt = (c.get("usage") or {}).get("prompt_tokens", 0)
        if not (crit["prompt"]["tokens_min"] <= pt <= crit["prompt"]["tokens_max"]):
            why.append(f"prompt tokens {pt} outside {crit['prompt']['tokens_min']}..{crit['prompt']['tokens_max']}")
        if c.get("status") != crit["http"]["status"]:
            why.append(f"chat HTTP {c.get('status')}: {c.get('error')}")
        if ((c.get("usage") or {}).get("completion_tokens") or 0) < crit["http"]["completion_min"]:
            why.append("chat returned no completion tokens")
        pm = ((c.get("timings") or {}).get("prompt_ms") or 0) / 1000
        doc["prefill_s"] = round(pm, 2)
        doc["prefill_tok_s"] = round(pt / pm, 1) if pm else None
        if not pm or pm > crit["prefill"]["seconds_max"]:
            why.append(f"prefill {pm} s over {crit['prefill']['seconds_max']} s")
        if max(doc.get("gtt_growth_mib") or [0]) > crit["memory"]["stage_growth_gib_max"] * 1024:
            why.append(f"a stage grew {max(doc['gtt_growth_mib'])} MiB over idle")
        if (doc.get("short_after") or {}).get("status") != 200:
            why.append(f"short request after the long one: {doc.get('short_after')}")
        sr, g = doc.get("serve_raw", {}), doc.get("generator", {})
        if (sr.get("usage") or {}).get("prompt_tokens") != doc.get("raw_prompt_tokens"):
            why.append(f"/tokenize gave {doc.get('raw_prompt_tokens')} ids, the completion read {(sr.get('usage') or {}).get('prompt_tokens')}")
        k = crit["identity"]["max_tokens"]
        if sr.get("status") != 200 or not sr.get("ids"):
            why.append(f"serve raw completion: {sr.get('status')} {sr.get('error')}")
        elif not g.get("ids"):
            why.append(f"generator path produced no ids (rc {g.get('rc')})")
        else:
            m = min(len(sr["ids"]), len(g["ids"]), k)
            doc["identity"] = {"compared": m, "equal": sr["ids"][:m] == g["ids"][:m], "serve": sr["ids"][:m], "generator": g["ids"][:m]}
            if m < 8 or not doc["identity"]["equal"]:
                why.append(f"ids differ or too few ({m} compared)")
    if any(x != crit["memory"]["exit_codes"] for x in doc["exit"]):
        why.append(f"exit codes {doc['exit']}")
    doc["pass"], doc["why"] = not why, why
    json.dump(doc, open(os.path.join(RES, "result.json"), "w"), indent=1)
    print(json.dumps({k: doc.get(k) for k in ("pass", "why", "context", "load_s", "thread", "prefill_s", "prefill_tok_s", "gtt_idle_mib", "gtt_peak_mib", "gtt_growth_mib",
                                              "identity", "exit")}, indent=1))
    return 0 if not why else 1


if __name__ == "__main__":
    sys.exit(main())
