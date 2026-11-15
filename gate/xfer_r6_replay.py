#!/usr/bin/env python3
"""SHAPE-TRANSFERS R6 part 1 (criteria gate/xfer-r6-spec.toml): replay recorded greedy outputs through a prompt-lookup
drafter and count accepted draft tokens a verify step. No GPU.

    python3 gate/xfer_r6_replay.py [--out gate/results/xfer-r6/replay.json]
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P4 = os.path.join(ROOT, "gate/results/qwen36-35b-a3b.flerp-moe-p4")
NAMES = ["bridge", "code", "rivers", "records"]   # prompt order of the P4 q35 run (prompt_tokens 21 20 18 1808)


def load():
    out = []
    for line in open(os.path.join(P4, "q35-0-r0.jsonl")):
        d = json.loads(line)
        if d.get("event") == "staged" and d.get("config") == "2:even:on":
            name = NAMES[d["prompt"]]
            prompt = [int(t) for t in open(os.path.join(P4, f"{name}.ids")).read().split()]
            assert len(prompt) == d["prompt_tokens"], (name, len(prompt), d["prompt_tokens"])
            out.append((name, prompt, d["ids"]))
    return out


def draft(ctx, k, nmax, nmin):
    for n in range(nmax, nmin - 1, -1):
        if len(ctx) <= n:
            continue
        tail = ctx[-n:]
        for i in range(len(ctx) - n - 1, -1, -1):        # most recent earlier match
            if ctx[i:i + n] == tail:
                return ctx[i + n:i + n + k]
    return []


def replay(prompt, gen, k, nmax, nmin):
    pos, steps, drafted_steps, accepted, rows = 0, 0, 0, 0, 0
    while pos < len(gen):
        ctx = prompt + gen[:pos]
        d = draft(ctx, k, nmax, nmin)
        a = 0
        while a < len(d) and pos + a < len(gen) and d[a] == gen[pos + a]:
            a += 1
        steps += 1
        drafted_steps += bool(d)
        accepted += a
        rows += len(d) + 1
        pos += a + 1
    return {"tokens": len(gen), "verify_steps": steps, "steps_with_draft": drafted_steps,
            "accepted": accepted, "accepted_per_step": round(accepted / steps, 3),
            "tokens_per_step": round(len(gen) / steps, 3), "rows_per_step": round(rows / steps, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "gate/results/xfer-r6/replay.json"))
    a = ap.parse_args()
    data = load()
    res = {"source": "q35-0-r0.jsonl config 2:even:on", "prompts": {n: {"prompt": len(p), "generated": len(g)} for n, p, g in data},
           "prompt_lookup": [], "mtp": "not measured: the recorded runs carry no MTP drafts (the generator attaches no draft backend)"}
    for nmin in (1, 2):
        for k in (1, 2, 3, 4, 7):
            per = {n: replay(p, g, k, 4, nmin) for n, p, g in data}
            tot = {key: sum(v[key] for v in per.values()) for key in ("tokens", "verify_steps", "steps_with_draft", "accepted")}
            rows = sum(v["rows_per_step"] * v["verify_steps"] for v in per.values())
            pooled = {**tot, "accepted_per_step": round(tot["accepted"] / tot["verify_steps"], 3),
                      "tokens_per_step": round(tot["tokens"] / tot["verify_steps"], 3), "rows_per_step": round(rows / tot["verify_steps"], 3)}
            res["prompt_lookup"].append({"k": k, "nmax": 4, "nmin": nmin, "pooled": pooled, "per_prompt": per})
            print(f"nmin {nmin} k {k}: pooled accepted/step {pooled['accepted_per_step']} tokens/step {pooled['tokens_per_step']} rows/step {pooled['rows_per_step']} | "
                  + " ".join(f"{n} {v['tokens_per_step']}" for n, v in per.items()))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
