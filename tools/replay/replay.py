#!/usr/bin/env python3
"""Self-replay of corpus-v1 strixite chat exchanges on a NON-serving strix_server (control-node GPU).

  replay.py run   --list LIST --out DIR --run NAME [--max-tokens N]   start server, replay seq + concurrent pairs, stop
  replay.py classify FILE...                                          print class labels
Each request is sent as recorded (scrubbed) plus "seed": SEED; max_tokens is capped (recorded in the output).
Server: the fleet binary strix_server.v070tg (= 20937dc + d19945e) with the fleet argv, a fresh empty prompt cache
dir per run, port 5391 on 127.0.0.1.
"""
import argparse, json, os, shutil, subprocess, sys, threading, time, urllib.request, urllib.error

SEED = 20261112
BIN = "/data/models/strix-infer/runtime/bin/strix_server.v070tg"
PORT = 5391


def classes(req, rec):
    c = []
    msgs = req.get("messages", [])
    if any(isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"]) for m in msgs):
        c.append("images")
    if req.get("steer"):
        c.append("steered")
    if req.get("steer_scope"):
        c.append("steer_scope")
    if req.get("tool_choice") == "none":
        c.append("tool_choice_none")
    usage = (rec or {}).get("usage", {})
    if usage.get("prompt_tokens", 0) >= 32768:
        c.append("long_prompt")
    if (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) > 0:
        c.append("cache_hit")
    if req.get("tools"):
        c.append("tools")
    if not c:
        c.append("plain")
    return c


def primary(c):
    for k in ("images", "steer_scope", "steered", "tool_choice_none", "long_prompt", "cache_hit", "tools", "plain"):
        if k in c:
            return k


def http(path, body=None, timeout=1800):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as f:
            return f.status, f.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def norm(raw):
    try:
        o = json.loads(raw)
    except Exception:
        return None, raw
    for k in ("id", "created", "timings", "system_fingerprint"):
        o.pop(k, None)
    for ch in o.get("choices", []):                # tool-call ids are random per reply (call_<random>): volatile
        for tc in (ch.get("message") or {}).get("tool_calls") or []:
            tc.pop("id", None)
    msg = [(ch.get("message"), ch.get("finish_reason")) for ch in o.get("choices", [])]
    tok = json.dumps({"m": msg, "n": (o.get("usage") or {}).get("completion_tokens")}, sort_keys=True)
    return tok, json.dumps(o, sort_keys=True)


def send(e, cap, out, key):
    req = dict(e["request"]); req.setdefault("seed", SEED)
    if cap:
        req["max_tokens"] = min(int(req.get("max_tokens") or cap), cap)
    t0 = time.time()
    st, raw = http("/v1/chat/completions", req)
    out[key] = {"status": st, "raw": raw.decode("utf-8", "replace"), "secs": round(time.time() - t0, 2), "max_tokens": req.get("max_tokens")}


def run(a):
    os.makedirs(a.out, exist_ok=True)
    cache = f"/cache/flerp-selfid-{a.run}" if os.path.isdir("/cache") and os.access("/cache", os.W_OK) else os.path.join(a.out, "pcache-" + a.run)
    shutil.rmtree(cache, ignore_errors=True); os.makedirs(cache)
    argv = [BIN, "--weights", "/data/models/strix-infer/converted/qwen38-fnext-tpn/weights.strixw",
            "--ngram", "/data/models/strix-infer/converted/qwen38-fnext-ngram-int8/ngram.table",
            "--tokenizer", "/data/models/strix-infer/tok/tokenizer.json",
            "--generation-config", "/data/models/strix-infer/tok/generation_config.json",
            "--host", "127.0.0.1", "--port", str(PORT), "--capacity", "262144", "--chunk", "8192", "--mtp", "on",
            "--prompt-cache-dir", cache, "--prompt-cache-gib", "32", "--prompt-cache-disk-free-gib", "8",
            "--prompt-cache-idle-s", "150", "--model-id", "qwen3.8-flash-next", "--total-slots", "2",
            "--cache-xfer", "on", "--steer-dir", "/data/models/strix-infer/steer", "--prompt-cache-ram-margin-gib", "10"]
    log = open(os.path.join(a.out, f"server-{a.run}.log"), "w")
    srv = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
    res = {"run": a.run, "argv": argv, "seed": SEED, "seq": {}, "pairs": {}}
    try:
        t0 = time.time()
        while True:
            if srv.poll() is not None:
                raise SystemExit(f"server exited rc={srv.returncode}")
            try:
                st, b = http("/health", timeout=5)
                if st == 200 and json.loads(b).get("status") in ("ok", "ready", None):
                    break
            except Exception:
                pass
            if time.time() - t0 > 1200:
                raise SystemExit("server not healthy in 20 min")
            time.sleep(5)
        res["load_secs"] = round(time.time() - t0, 1)
        ex = [json.load(open(p.strip())) for p in open(a.list) if p.strip()]
        for e in ex:                                   # one conversation at a time, recorded order
            send(e, a.max_tokens, res["seq"], e["id"])
            print(time.strftime("%H:%M:%S"), "seq", e["id"], res["seq"][e["id"]]["secs"], flush=True)
        pe = ex[: a.pairs * 2]
        for i in range(0, len(pe) - 1, 2):             # two conversations in flight at once
            th = [threading.Thread(target=send, args=(x, a.max_tokens, res["pairs"], x["id"])) for x in pe[i:i + 2]]
            th[0].start(); time.sleep(0.2); th[1].start()
            for t in th:
                t.join()
            print(time.strftime("%H:%M:%S"), "pair", pe[i]["id"], pe[i + 1]["id"], flush=True)
    finally:
        srv.terminate()
        try:
            srv.wait(60)
        except Exception:
            srv.kill()
        shutil.rmtree(cache, ignore_errors=True)
        json.dump(res, open(os.path.join(a.out, f"run-{a.run}.json"), "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--list"); ap.add_argument("--out"); ap.add_argument("--run")
    ap.add_argument("--max-tokens", type=int, default=384); ap.add_argument("--pairs", type=int, default=6)
    a = ap.parse_args()
    run(a)
