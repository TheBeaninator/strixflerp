#!/usr/bin/env python3
"""ORNITH-PCACHE step 0 (card a3656fc5e9): slot-resident prefix reuse in the serve path, on the 35B proxy. World 4 over loopback TCP
on control-node's one GPU, the serve front end (flerp-serve + 3 stage processes), ONE load, serve config ornith397.serve (drafts 1,
context 65536, reuse_tail 32). The sequence and the pass criteria are in gate/ornith-pcache.toml (committed before the run):
T1 cold, T2 warm (T1 + reply + a ~1K turn), T3 warm, a short request, T2 cold, a short request, T3 cold; warm ids must equal cold ids.
Run on control-node through tools/gpu-queue.sh (gate/ornith_pcache_check_run.sh); evidence in gate/results/ornith-pcache/."""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import tomllib
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gate"))
import flerplib.configs as CF  # noqa: E402
import ornith_config_check as OC  # noqa: E402
import ornith_ctx_check as CC  # noqa: E402

RES = os.path.join(ROOT, "gate", "results", "ornith-pcache")


def new_turn(tokens_hint):
    """A user turn of roughly `tokens_hint` tokens, shaped like a tool result pasted into the thread."""
    rows = []
    i = 0
    while len("".join(rows)) < tokens_hint * 4:
        rows.append(f"ticket {2000 + i}: owner {i % 7 + 1}, status {['open', 'blocked', 'done'][i % 3]}, due in {i % 28 + 1} days, "
                    f"topic {CC.TOPICS[i % len(CC.TOPICS)]}; note: confirm vault entry {1000 + i * 3} before the next hand-off.\n")
        i += 1
    return "Here is the latest export, please update the summary and flag anything that changed:\n" + "".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=os.path.join(ROOT, "build", "pcache-serve", "out"))
    ap.add_argument("--rocm", default="")
    ap.add_argument("--serve-config", default="ornith397.serve")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--port", type=int, default=39810)
    ap.add_argument("--sys-sections", type=int, default=60)
    ap.add_argument("--tools", type=int, default=40)
    ap.add_argument("--turns", type=int, default=24)
    ap.add_argument("--target-tokens", type=int, default=20000)
    ap.add_argument("--res", default=RES)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    res = a.res
    crit = tomllib.load(open(os.path.join(ROOT, "gate", "ornith-pcache.toml"), "rb"))
    os.makedirs(res, exist_ok=True)
    cfg = CF.override(CF.load(a.serve_config), a.set)
    ctx, n, rt = CF.context(cfg, 4096), CF.drafts(cfg), CF.reuse_tail(cfg)
    rg = CF.reuse_grid(cfg)
    why = []
    rocm = a.rocm or tomllib.load(open(os.path.join(ROOT, "toolchains.toml"), "rb"))["therock-10.1"]["rocm_root"]
    env = dict(os.environ, LD_LIBRARY_PATH=f"{rocm}/lib:{rocm}/lib/llvm/lib", **cfg["env"])
    gen = os.path.join(res, "generation_config.json")
    json.dump({"temperature": 0}, open(gen, "w"))
    mtp = {"draft_head": "on", "mtp_data_file": OC.MODEL} if n else {}
    base = {"weights": OC.MODEL, "context": ctx, "world": 4, "placement": "even", "epoch": 7, "peers": f"127.0.0.1:{a.port + 20}",
            "single_reader": "on", "pause_ms": 0, "setup_deadline_ms": 600000, "deadline_ms": 120000}
    kv0 = dict(base, **({"drafts": n} if n else {}), **({"reuse_tail": rt} if rt else {}), **({"reuse_grid": rg} if rg else {}))
    procs, doc = [], {"config": cfg["name"], "context": ctx, "drafts": n, "reuse_tail": rt, "reuse_grid": rg, "build": a.build}

    def start(label, argv):
        lp = os.path.join(res, f"{label}.jsonl")
        p = subprocess.Popen([os.path.join(a.build, argv[0])] + argv[1:], stdout=open(lp, "wb"), stderr=open(os.path.join(res, f"{label}.err"), "wb"),
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
        kr = dict(base)
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
    doc["gtt_idle_mib"] = [CC.gtt_kib(x) // 1024 for x in pids]
    mt = crit["identity"]["max_tokens"]
    runs = {}

    def chat(label, msgs):
        peak, stop = [0] * len(pids), threading.Event()

        def sample():
            while not stop.is_set():
                for i, x in enumerate(pids):
                    peak[i] = max(peak[i], CC.gtt_kib(x))
                time.sleep(1)
        th = threading.Thread(target=sample, daemon=True)
        th.start()
        st, d, secs = CC.post(a.port, "/v1/chat/completions", {"model": "m", "messages": msgs, "max_tokens": mt, "flerp_trace": True})
        stop.set()
        th.join()
        ch = (d.get("choices") or [{}])[0]
        runs[label] = {"status": st, "wall_s": round(secs, 2), "usage": d.get("usage"), "prompt_ms": (d.get("timings") or {}).get("prompt_ms"),
                       "ids": (d.get("flerp") or {}).get("token_ids"), "text": (ch.get("message") or {}).get("content"), "error": d.get("error"),
                       "gtt_peak_mib": [x // 1024 for x in peak]}
        print(label, json.dumps({k: runs[label][k] for k in ("status", "wall_s", "usage", "prompt_ms")}), flush=True)
        return runs[label]

    if up:
        sys_n = a.sys_sections
        for _ in range(6):
            t1 = CC.thread(sys_n, a.tools, a.turns)
            st, d, _ = CC.post(a.port, "/tokenize", {"content": CC.flat(t1)})
            ntok = len(d.get("tokens") or []) if st == 200 else 0
            if not ntok or abs(ntok - a.target_tokens) < 800:
                break
            sys_n = max(1, int(sys_n * a.target_tokens / ntok))
        doc["thread"] = {"sys_sections": sys_n, "tools": a.tools, "turns": a.turns, "tokenize_tokens": ntok}
        short = [{"role": "user", "content": "Say hello in five words."}]
        r1 = chat("T1", t1)
        t2 = t1 + [{"role": "assistant", "content": r1["text"] or "ok"}, {"role": "user", "content": new_turn(1000)}]
        r2 = chat("T2_warm", t2)
        t3 = t2 + [{"role": "assistant", "content": r2["text"] or "ok"}, {"role": "user", "content": "Thanks. Which item is due first?"}]
        r3 = chat("T3_warm", t3)
        chat("short1", short)
        chat("T2_cold", t2)
        chat("short2", short)
        chat("T3_cold", t3)
        doc["gtt_total_after_mib"] = CC.total_gtt() // 2**20
    for p in procs:
        p.terminate()
    doc["exit"] = []
    for p in procs:
        try:
            doc["exit"].append(p.wait(120))
        except subprocess.TimeoutExpired:
            p.kill()
            doc["exit"].append("killed")
    doc["runs"] = runs
    ev0 = OC.events(l0)
    doc["events"] = [e for e in ev0 if str(e.get("event", "")).startswith("prefix_")]
    # judge against gate/ornith-pcache.toml
    if not up:
        why.append("serve front end did not come up")
    else:
        ok = all(runs.get(k, {}).get("status") == crit["state"]["http_status"] for k in runs)
        if not ok or len(runs) < 7:
            why.append(f"HTTP: {[(k, v.get('status'), v.get('error')) for k, v in runs.items() if v.get('status') != 200]} ({len(runs)} runs)")
        p1 = (runs["T1"].get("usage") or {}).get("prompt_tokens", 0)
        p2 = (runs["T2_warm"].get("usage") or {}).get("prompt_tokens", 0)
        if not (crit["reuse"]["t1_prompt_tokens_min"] <= p1 <= crit["reuse"]["t1_prompt_tokens_max"]):
            why.append(f"T1 prompt tokens {p1} outside {crit['reuse']['t1_prompt_tokens_min']}..{crit['reuse']['t1_prompt_tokens_max']}")
        if not (crit["reuse"]["t2_new_tokens_min"] <= p2 - p1 <= crit["reuse"]["t2_new_tokens_max"]):
            why.append(f"T2 added {p2 - p1} tokens, outside {crit['reuse']['t2_new_tokens_min']}..{crit['reuse']['t2_new_tokens_max']}")
        hits = [e for e in doc["events"] if e.get("event") == "prefix_reuse"]
        doc["hits"], doc["reused_tokens"] = len(hits), sum(e.get("kept", 0) for e in hits)
        if len(hits) < crit["reuse"]["hits_min"] or doc["reused_tokens"] < crit["reuse"]["reused_tokens_min"]:
            why.append(f"prefix hits {len(hits)} reusing {doc['reused_tokens']} tokens; need {crit['reuse']['hits_min']} and {crit['reuse']['reused_tokens_min']}")
        doc["identity"] = {}
        for w, c in (("T2_warm", "T2_cold"), ("T3_warm", "T3_cold")):
            iw, ic = runs[w].get("ids"), runs[c].get("ids")
            m = min(len(iw or []), len(ic or []))
            eq = bool(iw) and bool(ic) and iw[:m] == ic[:m] and m >= crit["identity"]["min_tokens_compared"] and len(iw) == len(ic)
            doc["identity"][w] = {"compared": m, "equal": eq, "warm": iw, "cold": ic, "warm_text": runs[w].get("text"), "cold_text": runs[c].get("text")}
            if not eq:
                why.append(f"{w} ids differ from {c} (compared {m}, warm {len(iw or [])} cold {len(ic or [])})")
        pw, pc = runs["T2_warm"].get("prompt_ms") or 0, runs["T2_cold"].get("prompt_ms") or 0
        doc["t2_prefill_s"] = {"warm": round(pw / 1000, 2), "cold": round(pc / 1000, 2), "speedup": round(pc / pw, 2) if pw else None}
        doc["t3_prefill_s"] = {"warm": round((runs["T3_warm"].get("prompt_ms") or 0) / 1000, 2), "cold": round((runs["T3_cold"].get("prompt_ms") or 0) / 1000, 2)}
        doc["t1_prefill_s"] = round((runs["T1"].get("prompt_ms") or 0) / 1000, 2)
        if not pw or pc / pw < crit["speed"]["t2_speedup_min"]:
            why.append(f"T2 speedup {doc['t2_prefill_s']['speedup']} under {crit['speed']['t2_speedup_min']}")
        grow = [(max(r["gtt_peak_mib"][i] for r in runs.values()) - doc["gtt_idle_mib"][i]) for i in range(len(pids))]
        doc["gtt_growth_mib"] = grow
        if max(grow) > crit["state"]["snapshot_gib_per_stage_max"] * 1024:
            why.append(f"a stage grew {max(grow)} MiB over idle")
    if any(x != crit["state"]["exit_codes"] for x in doc["exit"]):
        why.append(f"exit codes {doc['exit']}")
    doc["pass"], doc["why"] = not why, why
    json.dump(doc, open(os.path.join(res, "result.json"), "w"), indent=1)
    print(json.dumps({k: doc.get(k) for k in ("pass", "why", "load_s", "thread", "t1_prefill_s", "t2_prefill_s", "t3_prefill_s", "hits", "reused_tokens",
                                              "gtt_growth_mib", "exit", "events")}, indent=1))
    return 0 if not why else 1


if __name__ == "__main__":
    sys.exit(main())
