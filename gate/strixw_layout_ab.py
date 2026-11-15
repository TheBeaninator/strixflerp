#!/usr/bin/env python3
"""Client side of gate/strixw-layout-ab.toml, stdlib only. Runs on the worker against the arm's scratch port.

  strixw_layout_ab.py PORT ARM OUT.json [--skip32k]
Writes per-request timings and the quality generations (with top-20 logprobs) to OUT.json.
  strixw_layout_ab.py --compare NEW.json OLD.json   -> agreement / KL table (JSON)
"""
import json
import math
import random
import sys
import time
import urllib.request

WORDS = ("river stone market ledger copper winter signal harbor engine lantern orchard bridge cipher meadow anchor "
         "furnace quarry thread mirror compass glacier timber beacon spindle vessel archive canyon pigment saddle "
         "falcon prism kettle granite rudder tapestry valley ember quill harvest lattice monsoon parcel turbine").split()


def filler(n_words, seed):
    r = random.Random(seed)
    out = []
    for i in range(n_words):
        out.append(r.choice(WORDS))
        if i % 14 == 13:
            out[-1] += "."
    return " ".join(out)


def post(port, body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    d["_wall_s"] = time.time() - t0
    return d


def chat(port, text, max_tokens, nonce, logprobs=False):
    body = {"model": "qwen3.8-flash-next", "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": f"[run {nonce}]\n{text}"}]}
    if logprobs:
        body.update(logprobs=True, top_logprobs=20)
    return post(port, body)


QUALITY = [
    [{"role": "user", "content": "Explain in three sentences why the sky is blue."}],
    [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."}],
    [{"role": "user", "content": "Write a C function `int popcount64(unsigned long long x)` without builtins and explain it briefly."}],
    [{"role": "system", "content": "You can call tools. Tools: get_weather(city: string) -> current weather. "
                                   "To call a tool reply only with JSON {\"tool\": name, \"arguments\": {...}}."},
     {"role": "user", "content": "What's the weather in Lisbon right now?"}],
    [{"role": "user", "content": "List the first ten prime numbers, comma separated, then their sum."}],
    [{"role": "user", "content": "Translate to French: 'The engine was restarted after the cooling fan failed.'"}],
    [{"role": "user", "content": "A train leaves at 14:05 and arrives at 17:50. How long is the trip? Show the arithmetic."}],
    [{"role": "user", "content": "Write a bash one-liner that counts lines in all .py files under the current directory."}],
    [{"role": "user", "content": "Summarize the plot of Hamlet in one paragraph."}],
    [{"role": "user", "content": "Give a JSON object describing a book with fields title, author, year, tags (array)."}],
]


def run(port, arm, out, skip32k):
    res = {"arm": arm, "port": port, "t0": time.time(), "speed": [], "quality": []}
    nonce = f"{arm}-{int(time.time())}"
    task = "Read the notes below, then write a long, detailed story that uses as many of the words as possible.\n\n"
    plan = [("warmup", 1500, 32, 1), ("decode2k", 1500, 256, 3), ("prefill8k", 6000, 1, 2)]
    if not skip32k:
        plan += [("prefill32k", 24000, 1, 2), ("decode32k", 24000, 256, 2)]
    k = 0
    for name, words, mt, reps in plan:
        text = task + filler(words, seed=words)          # same text on both arms; nonce makes each request uncached
        for i in range(reps):
            k += 1
            d = chat(port, text, mt, f"{nonce}-{k}")
            t = d.get("timings", {})
            row = {"kind": name, "rep": i, "prompt_tokens": d["usage"]["prompt_tokens"], "timings": t,
                   "wall_s": d["_wall_s"]}
            res["speed"].append(row)
            print(json.dumps(row), flush=True)
            json.dump(res, open(out, "w"))
    for i, msgs in enumerate(QUALITY):
        d = post(port, {"model": "qwen3.8-flash-next", "temperature": 0, "max_tokens": 128, "messages": msgs,
                        "logprobs": True, "top_logprobs": 20})
        ch = d["choices"][0]
        res["quality"].append({"i": i, "content": ch["message"].get("content"),
                               "tool_calls": ch["message"].get("tool_calls"),
                               "reasoning": ch["message"].get("reasoning_content"),
                               "logprobs": (ch.get("logprobs") or {}).get("content", [])})
        print("quality", i, len(res["quality"][-1]["logprobs"]), flush=True)
        json.dump(res, open(out, "w"))
    res["t1"] = time.time()
    json.dump(res, open(out, "w"))


def kl_top(a, b):
    """KL(a||b) over the union of both top-20 lists, each renormalised; missing mass floored at the list's min."""
    pa = {e["token"]: math.exp(e["logprob"]) for e in a}
    pb = {e["token"]: math.exp(e["logprob"]) for e in b}
    fa, fb = min(pa.values()) / 10, min(pb.values()) / 10
    keys = set(pa) | set(pb)
    sa = sum(pa.get(x, fa) for x in keys)
    sb = sum(pb.get(x, fb) for x in keys)
    return sum((pa.get(x, fa) / sa) * math.log((pa.get(x, fa) / sa) / (pb.get(x, fb) / sb)) for x in keys)


def compare(new_p, old_p):
    n, o = json.load(open(new_p)), json.load(open(old_p))
    rows = []
    for qn, qo in zip(n["quality"], o["quality"]):
        tn = [e["token"] for e in qn["logprobs"]]
        to = [e["token"] for e in qo["logprobs"]]
        div = next((i for i in range(min(len(tn), len(to))) if tn[i] != to[i]), min(len(tn), len(to)))
        kls, top1 = [], []
        for i in range(min(div + 1, len(tn), len(to))):      # shared prefix + the divergence position
            a, b = qn["logprobs"][i]["top_logprobs"], qo["logprobs"][i]["top_logprobs"]
            if a and b:
                kls.append(kl_top(a, b))
                top1.append(a[0]["token"] == b[0]["token"])
        rows.append({"i": qn["i"], "len_new": len(tn), "len_old": len(to), "identical": tn == to,
                     "first_div": div, "top1_agree": sum(top1) / max(1, len(top1)),
                     "kl_mean": sum(kls) / max(1, len(kls)), "kl_max": max(kls or [0])})
    s = {"rows": rows, "identical": sum(r["identical"] for r in rows), "n": len(rows),
         "median_first_div": sorted(r["first_div"] for r in rows)[len(rows) // 2],
         "mean_top1": sum(r["top1_agree"] for r in rows) / len(rows),
         "mean_kl": sum(r["kl_mean"] for r in rows) / len(rows)}
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(int(sys.argv[1]), sys.argv[2], sys.argv[3], "--skip32k" in sys.argv)
