#!/usr/bin/env python3
"""Two-stream replay for strixite slots (card 331a35995a): where a worker's time goes with two conversations on one GPU.

  tools/slots-replay.py run --binary PATH --slots N --out DIR [--reps 3] [--turns 8] [--max-tokens 256] [--label X]
  tools/slots-replay.py compare --baseline DIR [DIR ...] --candidate DIR [DIR ...] [--criteria gate/slots-criteria.toml]
  tools/slots-replay.py breakdown --run DIR         # one run's accounting: seconds per slot and state, cost per call kind

run (on control-node, under tools/gpu-queue.sh): takes the control-gpu lease, starts strix_server with the fleet's serving
flags (engines.conf strixite_v070pod: capacity 262144, chunk 8192, MTP on, the prompt cache on a scratch dir, --total-slots N)
on a private port, then for each rep:
  1. the TTFT probe: stream B opens a conversation with a long system prompt (a 10k-token prefill); 2.5 s later stream A
     sends a short new-thread request and the time to its first streamed token is recorded;
  2. the main phase: streams A and B each run `turns` turns of one conversation at the same time, the prompt growing by
     the reply and a ~1,100-token record set per turn (the fleet's pattern: most prompt tokens cached, the tail prefilled).
     Aggregate decode tok/s = both streams' completion tokens over the phase wall; per-thread decode tok/s from the
     server's timings; /metrics deltas (strix_arbiter_* when the server has them) per phase.
Every request is greedy (temperature 0, fixed seed); the server's --capture-dir keeps token ids for the compare. One load
per process; weights only from the pod-weights NVMe copy. The lease is released and the server stopped on any exit.
compare: ABAB table of aggregate tok/s, per-thread tok/s and TTFT, token identity of the captures by prompt, and the pass
or fail against the criteria file. Writes nothing but what --out names.
"""
import argparse
import ctypes
import glob
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
import threading
import time
import tomllib
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
M = "/data/models/strix-infer"
FLEET = {  # the fleet's serving line (engines.conf strixite_v070pod + strix-solo.sh defaults), paths on control-node
    "weights": f"{M}/converted/qwen38-fnext-tpn/weights.strixw",
    "ngram": f"{M}/converted/qwen38-fnext-ngram-int8/ngram.table",
    "tokenizer": f"{M}/tok/tokenizer.json",
    "generation_config": f"{M}/tok/generation_config.json",
    "steer_dir": f"{M}/steer",
    "lib": f"{M}/runtime/lib",
}
LEASE = ("control-gpu", "agent")
WORDS = ("river stone amber signal lantern harbour copper window orchard meadow engine silver thread market "
         "bridge winter garden letter candle mountain paper anchor shadow forest island mirror ribbon valley "
         "kettle marble pocket summer tunnel violin wagon yellow zephyr basket cedar dragon ember falcon").split()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(out, *a):
    line = f"{now()} " + " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(os.path.join(out, "driver.log"), "a") as f:
        f.write(line + "\n")


def records(seed, sentences, first=1):
    """Deterministic filler: numbered record sentences from a linear congruential sequence (as gate/strixite_recipe.py)."""
    x, out = seed & 0x7fffffff, []
    for i in range(sentences):
        ws = []
        for _ in range(14):
            x = (1103515245 * x + 12345) % (1 << 31)
            ws.append(WORDS[(x >> 16) % len(WORDS)])
        out.append(f"Record {first + i}: the {ws[0]} {ws[1]} was moved from the {ws[2]} {ws[3]} to the {ws[4]} {ws[5]} "
                   f"by the {ws[6]} {ws[7]}, then logged under {ws[8]}-{ws[9]}-{(x >> 8) % 1000:03d} beside the "
                   f"{ws[10]} {ws[11]} and the {ws[12]} {ws[13]}.")
    return "\n".join(out)


def die_with_parent():
    try:
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:
        pass


# ------------------------------------------------------------------------------------------------ server + lease
class Lease:
    def __init__(self, out, ttl):
        self.out, self.ttl, self.held = out, ttl, False
        self.tool = shutil.which("pod-lease")

    def acquire(self):
        if not self.tool:
            log(self.out, "lease: no pod-lease tool; running without a lease")
            return
        r = subprocess.run([self.tool, "acquire", LEASE[0], "--owner", LEASE[1], "--why", "card 331a35995a slots replay",
                            "--ttl", str(self.ttl), "--card", "331a35995a"], capture_output=True, text=True)
        log(self.out, "lease acquire rc", r.returncode, (r.stdout + r.stderr).strip()[:300])
        if r.returncode != 0:
            raise SystemExit("GPU is not free: leased to someone else (pod-lease acquire failed)")
        self.held = True

    def release(self):
        if self.held and self.tool:
            r = subprocess.run([self.tool, "release", LEASE[0], "--owner", LEASE[1]], capture_output=True, text=True)
            log(self.out, "lease release rc", r.returncode, (r.stdout + r.stderr).strip()[:200])
            self.held = False


def gpu_free():
    """No process holds /dev/kfd and GTT use is under 1 GiB (tools/gpu-queue.sh's rule)."""
    for p in glob.glob("/proc/[0-9]*/fd"):
        try:
            for fd in os.listdir(p):
                if os.readlink(os.path.join(p, fd)) == "/dev/kfd":
                    return False
        except OSError:
            continue
    used = max(int(open(f).read()) for f in glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used"))
    return used < (1 << 30)


class Server:
    def __init__(self, binary, slots, port, scratch, out, extra):
        self.port, self.out = port, out
        cap = os.path.join(scratch, "capture")
        pc = os.path.join(scratch, "prompt-cache")
        os.makedirs(cap, exist_ok=True), os.makedirs(pc, exist_ok=True)
        self.capture_dir = cap
        self.args = [binary, "--weights", FLEET["weights"], "--ngram", FLEET["ngram"], "--tokenizer", FLEET["tokenizer"],
                     "--generation-config", FLEET["generation_config"], "--host", "127.0.0.1", "--port", str(port),
                     "--capacity", "262144", "--chunk", "8192", "--mtp", "on", "--prompt-cache-dir", pc,
                     "--prompt-cache-gib", "32", "--prompt-cache-disk-free-gib", "8", "--prompt-cache-idle-s", "150",
                     "--model-id", "qwen3.8-flash-next", "--total-slots", str(slots), "--cache-xfer", "on",
                     "--steer-dir", FLEET["steer_dir"], "--prompt-cache-ram-margin-gib", "10", "--vision", "off",
                     "--capture-dir", cap] + extra
        env = dict(os.environ, LD_LIBRARY_PATH=FLEET["lib"] + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
        self.log = open(os.path.join(out, "server.log"), "wb")
        self.t0 = time.time()
        self.proc = subprocess.Popen(self.args, stdout=self.log, stderr=subprocess.STDOUT, env=env,
                                     preexec_fn=die_with_parent)

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def wait_ready(self, timeout=900):
        while time.time() - self.t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with {self.proc.returncode} during startup; see server.log")
            try:
                with urllib.request.urlopen(self.url("/health"), timeout=2) as r:
                    if r.status == 200:
                        return round(time.time() - self.t0, 1)
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(1)
        raise RuntimeError(f"server not ready after {timeout} s")

    def metrics(self):
        with urllib.request.urlopen(self.url("/metrics"), timeout=10) as r:
            text = r.read().decode()
        out = {}
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            k, _, v = line.rpartition(" ")
            try:
                out[k] = float(v)
            except ValueError:
                pass
        return out

    def stop(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(120)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(30)
        self.log.close()
        return self.proc.returncode


# ------------------------------------------------------------------------------------------------ requests
def chat(server, messages, max_tokens, seed, stream=False, timeout=1800):
    body = {"model": "qwen3.8-flash-next", "messages": messages, "temperature": 0, "seed": seed,
            "max_tokens": max_tokens, "stream": stream, "enable_thinking": False}
    req = urllib.request.Request(server.url("/v1/chat/completions"), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    if not stream:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            j = json.loads(r.read())
        wall = time.time() - t0
        text = j["choices"][0]["message"].get("content") or ""
        return {"wall_s": wall, "text": text, "timings": j.get("timings", {}), "usage": j.get("usage", {}),
                "finish": j["choices"][0].get("finish_reason")}
    ttft, first_event, text, n = None, None, [], 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line.startswith("data: [DONE]"):
                continue
            try:
                ev = json.loads(line[5:])
            except ValueError:
                continue
            if first_event is None:
                first_event = time.time() - t0
            ch = ev.get("choices") or []  # the usage event at the end has no choices
            if not ch:
                continue
            d = ch[0].get("delta", {}) or {}
            piece = d.get("content") or d.get("reasoning_content") or ""
            if piece:
                if ttft is None:  # the first generated token, whatever field carries it (the role-only event is not one)
                    ttft = time.time() - t0
                text.append(piece)
                n += 1
    return {"wall_s": time.time() - t0, "ttft_s": ttft, "first_event_s": first_event, "text": "".join(text), "chunks": n}


def conversation(stream_id, rep, turns, max_tokens, seed):
    """The prompts of one stream: a system prompt of records and, per turn, a new record set + a question."""
    base = 1000003 * (rep + 1) + 7919 * (stream_id + 1)
    sys_prompt = ("You answer questions about the records you are given, precisely and briefly.\n\n" +
                  records(base, 40))
    turn_prompts = []
    for t in range(turns):
        rs = records(base + 31 * (t + 1), 24, first=100 * (t + 1))
        turn_prompts.append(rs + "\n\nWhich of these records mention a harbour or a bridge? List their numbers, "
                                 "quote each sentence, then summarise the whole set in three sentences.")
    return sys_prompt, turn_prompts, seed + stream_id


def run_stream(server, stream_id, rep, turns, max_tokens, seed, results, out, t_phase):
    sys_prompt, prompts, s = conversation(stream_id, rep, turns, max_tokens, seed)
    messages = [{"role": "system", "content": sys_prompt}]
    for t, p in enumerate(prompts):
        messages.append({"role": "user", "content": p})
        try:
            r = chat(server, messages, max_tokens, s)
        except Exception as e:
            log(out, f"stream {stream_id} turn {t} failed: {e}")
            results.append({"stream": stream_id, "turn": t, "error": str(e)})
            return
        messages.append({"role": "assistant", "content": r["text"]})
        results.append({"stream": stream_id, "turn": t, "t_start": r["wall_s"] and round(time.time() - r["wall_s"] - t_phase, 3),
                        "wall_s": round(r["wall_s"], 3), "timings": r["timings"], "usage": r["usage"], "finish": r["finish"]})
        log(out, f"rep {rep} stream {stream_id} turn {t}: {r['usage'].get('completion_tokens')} tokens in {r['wall_s']:.1f} s, "
                 f"decode {r['timings'].get('predicted_per_second', 0):.1f} t/s, prefill {r['timings'].get('prompt_ms', 0):.0f} ms")


def delta(a, b):
    return {k: round(b[k] - a.get(k, 0), 6) for k in b if k.startswith("strix_") and b[k] != a.get(k, 0)}


def run(args):
    os.makedirs(args.out, exist_ok=True)
    scratch = args.scratch or os.path.join(args.out, "scratch")
    if os.path.exists(scratch):
        shutil.rmtree(scratch)
    os.makedirs(scratch)
    if not gpu_free():
        print("GPU is not free: a process holds /dev/kfd or GTT is in use", file=sys.stderr)
        return 3
    meta = {"binary": args.binary, "binary_sha256": hashlib.sha256(open(args.binary, "rb").read()).hexdigest()[:16],
            "slots": args.slots, "reps": args.reps, "turns": args.turns, "max_tokens": args.max_tokens, "label": args.label,
            "started": now(), "host": os.uname().nodename, "extra": args.extra}
    lease = Lease(args.out, args.ttl)
    server = None
    result = {"meta": meta, "reps": []}
    try:
        lease.acquire()
        server = Server(args.binary, args.slots, args.port, scratch, args.out, args.extra)
        meta["server_args"] = server.args
        meta["load_s"] = server.wait_ready()
        log(args.out, f"server ready in {meta['load_s']} s: {args.binary} slots {args.slots}")
        for rep in range(args.reps):
            rep_out = {"rep": rep}
            # 1. TTFT probe: B's long first prompt, A's short request 2.5 s into it.
            m0 = server.metrics()
            long_sys = "You answer questions about the records you are given.\n\n" + records(777 + rep, 230)
            b_res, a_res = {}, {}
            t_phase = time.time()

            def b_long():
                try:
                    b_res.update(chat(server, [{"role": "system", "content": long_sys},
                                               {"role": "user", "content": "How many records mention a harbour? Answer with the number only."}],
                                      32, args.seed + 100 + rep))
                except Exception as e:
                    b_res["error"] = str(e)
            tb = threading.Thread(target=b_long)
            tb.start()
            time.sleep(2.5)
            try:
                a_res.update(chat(server, [{"role": "user", "content": "Name the capital of France in one word."}], 16,
                                  args.seed + 200 + rep, stream=True))
            except Exception as e:
                a_res["error"] = str(e)
            tb.join()
            rep_out["ttft_probe"] = {"a_ttft_s": a_res.get("ttft_s"), "a_wall_s": a_res.get("wall_s"),
                                     "a_first_event_s": a_res.get("first_event_s"), "a_chunks": a_res.get("chunks"),
                                     "b_prompt_ms": b_res.get("timings", {}).get("prompt_ms"),
                                     "b_prompt_tokens": b_res.get("usage", {}).get("prompt_tokens"),
                                     "b_wall_s": b_res.get("wall_s"), "errors": [x.get("error") for x in (a_res, b_res) if x.get("error")],
                                     "metrics_delta": delta(m0, server.metrics())}
            log(args.out, f"rep {rep} ttft probe: A first token {a_res.get('ttft_s')} s while B prefilled "
                          f"{rep_out['ttft_probe']['b_prompt_tokens']} tokens in {rep_out['ttft_probe']['b_prompt_ms']} ms")
            # 2. Main phase: two streams at once.
            m1 = server.metrics()
            results = []
            t_phase = time.time()
            threads = [threading.Thread(target=run_stream, args=(server, k, rep, args.turns, args.max_tokens, args.seed,
                                                                results, args.out, t_phase)) for k in (0, 1)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            wall = time.time() - t_phase
            m2 = server.metrics()
            ok = [r for r in results if "error" not in r]
            toks = sum(r["usage"].get("completion_tokens", 0) for r in ok)
            rep_out["main"] = {"wall_s": round(wall, 3), "completion_tokens": toks, "aggregate_tps": round(toks / wall, 3) if wall else 0,
                               "per_thread_tps_median": round(statistics.median([r["timings"].get("predicted_per_second", 0) for r in ok]), 3) if ok else 0,
                               "prefill_tps_median": round(statistics.median([r["timings"].get("prompt_per_second", 0) for r in ok]), 1) if ok else 0,
                               "requests": results, "errors": len(results) - len(ok), "metrics_delta": delta(m1, m2)}
            log(args.out, f"rep {rep} main: {toks} tokens in {wall:.1f} s = {rep_out['main']['aggregate_tps']:.2f} tok/s aggregate, "
                          f"per-thread median {rep_out['main']['per_thread_tps_median']:.2f}")
            result["reps"].append(rep_out)
            json.dump(result, open(os.path.join(args.out, "run.json"), "w"), indent=1)
        result["metrics_final"] = server.metrics()
    finally:
        if server is not None:
            result["server_rc"] = server.stop()
            caps = []
            for f in sorted(glob.glob(os.path.join(server.capture_dir, "*.cap"))):
                try:
                    caps.append(read_capture(f))
                except Exception as e:
                    caps.append({"file": f, "error": str(e)})
            result["captures"] = [{"id": c.get("id"), "prompt_n": c.get("prompt_n"), "prompt_hash": ids_hash(c.get("prompt", [])),
                                   "generated": c.get("generated", []), "finish": c.get("finish"), "error": c.get("error")} for c in caps]
        # The server's own first-token times of the probe requests (max_tokens 16), in rep order: the fallback when the
        # client-side stream parse has none (and the cross-check when it has).
        try:
            probe, server_ttft = False, []
            for line in open(os.path.join(args.out, "server.log"), errors="replace"):
                if "max_tokens  16" in line:
                    probe = True
                elif probe and " s first token" in line:
                    mt = re.search(r"([0-9.]+) s first token", line)
                    if mt:
                        server_ttft.append(float(mt.group(1)))
                    probe = False
            result["ttft_server_s"] = server_ttft
        except OSError:
            pass
        lease.release()
        result["meta"]["finished"] = now()
        json.dump(result, open(os.path.join(args.out, "run.json"), "w"), indent=1)
        if not args.keep_scratch and os.path.exists(scratch):
            shutil.rmtree(scratch, ignore_errors=True)
        log(args.out, "done; GPU free:", gpu_free())
    return 0


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
    return {"id": head["id"], "prompt_n": head["prompt_n"], "finish": head.get("finish"),
            "prompt": tokens[:head["prompt_n"]], "generated": tokens[head["prompt_n"]:]}


def ids_hash(ids):
    return hashlib.sha256(struct.pack(f"<{len(ids)}i", *ids)).hexdigest()[:16] if ids else "-"


# ------------------------------------------------------------------------------------------------ compare
def load_runs(dirs):
    return [json.load(open(os.path.join(d, "run.json"))) for d in dirs]


def agg(runs, key):
    return [rep["main"][key] for r in runs for rep in r["reps"] if "main" in rep]


def ttfts(runs):
    """The probe's TTFT per rep: the client's streamed first token, else the server's own 'first token' time."""
    out = []
    for r in runs:
        srv = r.get("ttft_server_s", [])
        for i, rep in enumerate(r["reps"]):
            v = rep.get("ttft_probe", {}).get("a_ttft_s")
            if v is None and i < len(srv):
                v = srv[i]
            if v is not None:
                out.append(v)
    return out


def identity(base_runs, cand_runs):
    """Captures matched by prompt ids: equal generated ids, or the first differing index."""
    base = {}
    for r in base_runs:
        for c in r.get("captures", []):
            base.setdefault(c["prompt_hash"], c["generated"])
    matched = diverged = 0
    first = None
    unmatched = 0
    for r in cand_runs:
        for c in r.get("captures", []):
            g = base.get(c["prompt_hash"])
            if g is None:
                unmatched += 1
                continue
            matched += 1
            if g != c["generated"]:
                diverged += 1
                k = next((i for i, (x, y) in enumerate(zip(g, c["generated"])) if x != y), min(len(g), len(c["generated"])))
                first = k if first is None else min(first, k)
    return {"matched_prompts": matched, "diverged": diverged, "first_divergence_index": first, "unmatched_prompts": unmatched}


def arbiter_table(runs):
    """Per-state seconds summed over the main phases, as shares of the wall, from the strix_arbiter_* deltas."""
    tot, wall = {}, 0.0
    for r in runs:
        for rep in r["reps"]:
            if "main" not in rep:
                continue
            wall += rep["main"]["wall_s"]
            for k, v in rep["main"]["metrics_delta"].items():
                if k.startswith("strix_arbiter_seconds_total") or k.startswith("strix_verify_batch") or k.startswith("strix_verify_single") \
                        or k.startswith("strix_arbiter_calls_total") or k.startswith("strix_arbiter_prefill_rows"):
                    tot[k] = tot.get(k, 0) + v
    return wall, tot


def compare(args):
    base, cand = load_runs(args.baseline), load_runs(args.candidate)
    crit = tomllib.load(open(args.criteria, "rb"))["pass"] if args.criteria else {}
    rows = []
    for name, b, c in (("aggregate decode tok/s", agg(base, "aggregate_tps"), agg(cand, "aggregate_tps")),
                       ("per-thread decode tok/s (median)", agg(base, "per_thread_tps_median"), agg(cand, "per_thread_tps_median")),
                       ("prefill tok/s (median)", agg(base, "prefill_tps_median"), agg(cand, "prefill_tps_median")),
                       ("TTFT of a new thread during the other's prefill (s)", ttfts(base), ttfts(cand))):
        if b and c:
            rows.append((name, statistics.median(b), min(b), max(b), statistics.median(c), min(c), max(c)))
    ident = identity(base, cand)
    out = [f"# slots replay compare ({now()})", "",
           f"baseline: {', '.join(args.baseline)}  (binary {base[0]['meta']['binary']}, slots {base[0]['meta']['slots']})",
           f"candidate: {', '.join(args.candidate)}  (binary {cand[0]['meta']['binary']}, slots {cand[0]['meta']['slots']})", "",
           "| metric | baseline median | min | max | candidate median | min | max | ratio |", "|---|---|---|---|---|---|---|---|"]
    for name, bm, bl, bh, cm, cl, ch in rows:
        out.append(f"| {name} | {bm:.3f} | {bl:.3f} | {bh:.3f} | {cm:.3f} | {cl:.3f} | {ch:.3f} | {cm / bm if bm else 0:.3f} |")
    out += ["", f"token identity (captures matched by prompt ids): {json.dumps(ident)}", ""]
    verdict = []
    if crit:
        ba, ca = statistics.median(agg(base, "aggregate_tps")), statistics.median(agg(cand, "aggregate_tps"))
        verdict.append(("aggregate >= baseline x %.2f" % crit["aggregate_ratio_min"], ca >= ba * crit["aggregate_ratio_min"]))
        verdict.append(("aggregate never below baseline x %.2f" % crit["aggregate_ratio_floor"],
                        min(agg(cand, "aggregate_tps")) >= ba * crit["aggregate_ratio_floor"]))
        verdict.append(("token ids identical on every matched prompt", ident["diverged"] == 0 and ident["matched_prompts"] > 0))
        bt, ct = ttfts(base), ttfts(cand)
        if bt and ct:
            verdict.append(("TTFT median <= baseline median + %.3f s" % crit["ttft_slack_s"],
                            statistics.median(ct) <= statistics.median(bt) + crit["ttft_slack_s"]))
        out += ["| criterion | result |", "|---|---|"] + [f"| {n} | {'PASS' if ok else 'FAIL'} |" for n, ok in verdict]
        out += ["", "VERDICT: " + ("PASS" if all(ok for _, ok in verdict) else "FAIL")]
    for label, runs in (("baseline", base), ("candidate", cand)):
        wall, tot = arbiter_table(runs)
        if tot:
            out += ["", f"## {label}: arbiter accounting over the main phases ({wall:.1f} s wall)", "", "| series | seconds or count | share of wall |", "|---|---|---|"]
            for k in sorted(tot):
                share = f"{100 * tot[k] / wall:.1f}%" if "seconds" in k and wall else ""
                out.append(f"| {k} | {tot[k]:.3f} | {share} |")
    text = "\n".join(out) + "\n"
    print(text)
    if args.out:
        open(args.out, "w").write(text)
    return 0


def breakdown(args):
    """One run's accounting over its main phases: where each slot's seconds went, and what a call of each kind costs."""
    r = json.load(open(os.path.join(args.run, "run.json")))
    wall, tot, aggs = 0.0, {}, []
    for rep in r["reps"]:
        if "main" not in rep:
            continue
        wall += rep["main"]["wall_s"]
        aggs.append(rep["main"]["aggregate_tps"])
        for k, v in rep["main"]["metrics_delta"].items():
            tot[k] = tot.get(k, 0) + v
    g = lambda k: tot.get(k, 0.0)
    slots = r["meta"]["slots"]
    out = [f"# breakdown of {args.run} ({r['meta'].get('label', '')}, slots {slots}, {len(aggs)} reps, {wall:.1f} s of main-phase wall)", "",
           f"aggregate decode tok/s per rep: {', '.join(f'{a:.2f}' for a in aggs)}; generated {g('strix_generated_tokens_total'):.0f} tokens; "
           f"decode_seconds {g('strix_decode_seconds_total'):.1f} (slot-seconds), prefill_seconds {g('strix_prefill_seconds_total'):.1f}, "
           f"PLE wait {g('strix_ple_wait_seconds_total'):.2f}, cache saves (engine export block) {g('strix_prompt_cache_export_seconds_total'):.2f}", ""]
    states = ["gpu_prefill", "gpu_forward", "gpu_cache", "gpu_call", "gpu_draft", "gpu_resolve", "gpu_batch", "wait_chain", "wait_prefill",
              "wait_forward", "wait_cache", "wait_call", "wait_batch_gpu", "wait_resolving", "wait_fair", "wait_lead", "wait_partner"]
    if slots > 1:
        out += ["| state | " + " | ".join(f"slot {k} s (% of wall)" for k in range(slots)) + " | calls (slot 0 / 1) | ms per call |", "|---|" + "---|" * (slots + 2)]
        for st in states:
            secs = [g(f'strix_arbiter_seconds_total{{slot="{k}",state="{st}"}}') for k in range(slots)]
            calls = [g(f'strix_arbiter_calls_total{{slot="{k}",state="{st}"}}') for k in range(slots)]
            per = (sum(secs) / sum(calls) * 1e3) if sum(calls) else 0
            cells = " | ".join(f"{x:.2f} ({100 * x / wall:.1f}%)" for x in secs)
            if st.startswith("gpu_"):
                out.append(f"| {st} | {cells} | " + " / ".join(f"{c:.0f}" for c in calls) + f" | {per:.1f} |")
            else:
                out.append(f"| {st} | {cells} |  |  |")
        rows = [g(f'strix_arbiter_prefill_rows_total{{slot="{k}"}}') for k in range(slots)]
        chunks = [g(f'strix_arbiter_calls_total{{slot="{k}",state="gpu_prefill"}}') for k in range(slots)]
        out += ["", f"prefill rows per chunk: " + ", ".join(f"slot {k} {rows[k] / chunks[k]:.0f}" if chunks[k] else f"slot {k} -" for k in range(slots)), "",
                "| batched verify by runs | batches | ms per batch | rows per batch | ms per row |", "|---|---|---|---|---|"]
        for n in range(1, 9):
            b = g(f'strix_verify_batches_by_runs_total{{runs="{n}"}}')
            if b:
                sec, rw = g(f'strix_verify_batch_seconds_by_runs_total{{runs="{n}"}}'), g(f'strix_verify_batch_rows_by_runs_total{{runs="{n}"}}')
                out.append(f"| {n} | {b:.0f} | {1e3 * sec / b:.1f} | {rw / b:.1f} | {1e3 * sec / rw if rw else 0:.2f} |")
        out += ["", "singles by the partner's state: " + ", ".join(f"{k.split('partner=')[1].strip('\"}')}={v:.0f}" for k, v in sorted(tot.items()) if k.startswith("strix_verify_single_total"))]
    text = "\n".join(out) + "\n"
    print(text)
    if args.out:
        open(args.out, "w").write(text)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--binary", required=True)
    r.add_argument("--slots", type=int, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--turns", type=int, default=8)
    r.add_argument("--max-tokens", type=int, default=256)
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--port", type=int, default=5391)
    r.add_argument("--ttl", type=int, default=3600)
    r.add_argument("--label", default="")
    r.add_argument("--scratch", default="")
    r.add_argument("--keep-scratch", action="store_true")
    r.add_argument("--extra", default="", help="more strix_server flags as one string, e.g. \"--chunk-shared 512\"")
    b = sub.add_parser("breakdown")
    b.add_argument("--run", required=True)
    b.add_argument("--out", default="")
    c = sub.add_parser("compare")
    c.add_argument("--baseline", nargs="+", required=True)
    c.add_argument("--candidate", nargs="+", required=True)
    c.add_argument("--criteria", default=os.path.join(HERE, "..", "gate", "slots-criteria.toml"))
    c.add_argument("--out", default="")
    args = ap.parse_args()
    if args.cmd == "run":
        args.extra = args.extra.split()
    return {"run": run, "compare": compare, "breakdown": breakdown}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
