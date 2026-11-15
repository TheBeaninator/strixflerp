#!/usr/bin/env python3
"""Client side of gate/strixw-mtpq4-ab2.toml, stdlib only; filler/QUALITY/chat/post from strixw_layout_ab.py.

  strixw_mtpq4_ab2.py PORT ARM OUT.json
  strixw_mtpq4_ab2.py --compare A.json B.json      first-divergence per prompt (token strings from logprobs)
Per-prompt /metrics deltas give acceptance and the rollback histogram (strix_mtp_rejections_total{accepted=..}).
"""
import json
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from strixw_layout_ab import QUALITY, chat, filler, post  # noqa: E402
from strixw_mtpq4_ab import delta, metrics  # noqa: E402

EXTRA = [
    [{"role": "user", "content": "Hi! I'm planning a weekend in Kyoto in November. What should I not miss?"}],
    [{"role": "user", "content": "My sourdough comes out dense. What are the most likely causes and fixes?"}],
    [{"role": "user", "content": "Write a Rust function that parses a line like 'key=value; key2=value2' into a HashMap<String, String>, with tests."}],
    [{"role": "user", "content": "Here is a Python bug: `def avg(xs): return sum(xs)/len(xs)` crashes on an empty list. Fix it and explain the choice."}],
    [{"role": "user", "content": "Write a SQL query that returns, per customer, the total order value in 2025 and their rank by that total."}],
    [{"role": "system", "content": "You can call tools. Tools: search_files(pattern: string, path: string), "
                                   "read_file(path: string). To call a tool reply only with JSON {\"tool\": name, \"arguments\": {...}}."},
     {"role": "user", "content": "Find where the function parse_header is defined under src/ and show me its body."}],
    [{"role": "system", "content": "You can call tools. Tools: create_event(title: string, start: ISO-8601, minutes: int). "
                                   "To call a tool reply only with JSON {\"tool\": name, \"arguments\": {...}}."},
     {"role": "user", "content": "Book a 45 minute dentist appointment next Tuesday 2026-11-17 at 9:30."}],
    [{"role": "user", "content": "Explain how TCP congestion control works (slow start, congestion avoidance, fast retransmit, "
                                 "fast recovery), then compare Reno, CUBIC and BBR in a table, then say which suits a "
                                 "long fat network with random loss and why. Be thorough."}],
    [{"role": "user", "content": "Write a detailed design for a home backup system: three machines, one NAS, one offsite "
                                 "cloud bucket. Cover schedules, retention, encryption, restore testing and failure modes."}],
    [{"role": "user", "content": "Write a short story (about 300 words) about a lighthouse keeper who finds a radio that "
                                 "receives tomorrow's weather."}],
]
PROMPTS = QUALITY + EXTRA   # 20: the first run's 10, then chat x2, code x3, tool-call x2, long-ish x3


def run(port, arm, out):
    res = {"arm": arm, "t0": time.time(), "speed": [], "phases": {}, "quality": []}
    nonce = f"{arm}-{int(time.time())}"
    task = "Read the notes below, then write a long, detailed story that uses as many of the words as possible.\n\n"
    plan = [("warmup", 1500, 32, 1), ("decode2k", 1500, 256, 3), ("decode30k", 24000, 256, 3)]
    k = 0
    for name, words, mt, reps in plan:
        text = task + filler(words, seed=words)
        m0 = metrics(port)
        for i in range(reps):
            k += 1
            d = chat(port, text, mt, f"{nonce}-{k}")
            row = {"kind": name, "rep": i, "prompt_tokens": d["usage"]["prompt_tokens"],
                   "completion_tokens": d["usage"].get("completion_tokens"), "timings": d.get("timings", {}),
                   "wall_s": d["_wall_s"]}
            res["speed"].append(row)
            print(json.dumps(row), flush=True)
        res["phases"][name] = delta(m0, metrics(port))
        print(name, json.dumps(res["phases"][name]), flush=True)
        json.dump(res, open(out, "w"))
    q0 = metrics(port)
    for i, msgs in enumerate(PROMPTS):
        m0 = metrics(port)
        d = post(port, {"model": "qwen3.8-flash-next", "temperature": 0, "max_tokens": 256, "messages": msgs,
                        "logprobs": True, "top_logprobs": 1})
        ch = d["choices"][0]
        res["quality"].append({"i": i, "content": ch["message"].get("content"),
                               "tool_calls": ch["message"].get("tool_calls"),
                               "reasoning": ch["message"].get("reasoning_content"),
                               "tokens": [e["token"] for e in (ch.get("logprobs") or {}).get("content", [])],
                               "mtp": delta(m0, metrics(port))})
        print("quality", i, len(res["quality"][-1]["tokens"]), flush=True)
        json.dump(res, open(out, "w"))
    res["phases"]["quality"] = delta(q0, metrics(port))
    res["t1"] = time.time()
    json.dump(res, open(out, "w"))


def compare(a_p, b_p):
    a, b = json.load(open(a_p)), json.load(open(b_p))
    rows = []
    for qa, qb in zip(a["quality"], b["quality"]):
        ta, tb = qa["tokens"], qb["tokens"]
        div = next((i for i in range(min(len(ta), len(tb))) if ta[i] != tb[i]), None)
        if div is None and len(ta) != len(tb):
            div = min(len(ta), len(tb))
        same = div is None and qa["content"] == qb["content"] and qa["reasoning"] == qb["reasoning"] \
            and qa["tool_calls"] == qb["tool_calls"]
        rows.append({"i": qa["i"], "len": [len(ta), len(tb)], "identical": same, "first_div": div})
    divs = sorted(r["first_div"] for r in rows if r["first_div"] is not None)
    print(json.dumps({"a": a["arm"], "b": b["arm"], "identical": sum(r["identical"] for r in rows), "n": len(rows),
                      "differing": len(divs), "median_first_div_of_differing": divs[len(divs) // 2] if divs else None,
                      "rows": rows}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(int(sys.argv[1]), sys.argv[2], sys.argv[3])
