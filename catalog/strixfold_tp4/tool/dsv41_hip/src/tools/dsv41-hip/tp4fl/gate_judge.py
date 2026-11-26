#!/usr/bin/env python3
"""gate_judge.py GATE_JSON TP2_DIR TP4_DIR [--md OUT.md]: the TP4-FIRST-LIGHT verdict (716f35d40a), by the gate written
before any run. Each DIR holds shas.jsonl (rank 0's prompt_logits lines for the 8 frozen prompts, in prompt order),
replies.jsonl (gate_prompts.py), aux.txt (the TF_DS_AUX the ranks ran with, empty = unset) and bench-rep*.json
(bench_stream.py, recorded only). PASS = both sides have exactly 8 lines, AUX unset on both, and all 8 sha256 equal
(rows equal too). Anything else is FAIL, with the first differing prompt named. Speed is printed, never judged."""
import json, sys, glob, os, re

# rank 0's log interleaves other writers into its lines (NCCL_DEBUG=INFO, run-1920): take the JSON object itself
PL = re.compile(r'\{"prompt_logits": "[^"]*", "sha256": "[0-9a-f]{64}", "rows": \d+\}')

def shas_of(path):
    return [json.loads(m) for l in open(path) for m in PL.findall(l)]

def load(d):
    shas = shas_of(os.path.join(d, "shas.jsonl"))
    reps = [json.loads(l) for l in open(os.path.join(d, "replies.jsonl")) if l.strip()]
    aux = open(os.path.join(d, "aux.txt")).read().strip() if os.path.exists(os.path.join(d, "aux.txt")) else "MISSING"
    bench = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "bench-rep*.json")))]
    return shas, reps, aux, bench

def main():
    gate = json.load(open(sys.argv[1])); d2, d4 = sys.argv[2:4]
    md = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
    s2, r2, a2, b2 = load(d2); s4, r4, a4, b4 = load(d4)
    n = gate["prompts"]; why = []
    if len(s2) != n or len(s4) != n: why.append(f"sha lines: TP2 {len(s2)}, TP4 {len(s4)}, want {n} each")
    if a2 != "" or a4 != "": why.append(f"TF_DS_AUX must be unset on both: TP2 '{a2}', TP4 '{a4}'")
    eq = 0; rows = []
    for i in range(min(len(s2), len(s4))):
        e = s2[i]["sha256"] == s4[i]["sha256"] and s2[i].get("rows") == s4[i].get("rows"); eq += e
        pid = r2[i]["id"] if i < len(r2) else str(i)
        rows.append(f"| {pid} | {s2[i].get('rows')} | {s2[i]['sha256'][:16]} | {s4[i]['sha256'][:16]} | {'equal' if e else 'DIFFERENT'} |")
        if not e and not any(w.startswith("first differing") for w in why): why.append(f"first differing prompt: {pid}")
    verdict = "PASS" if (not why and eq == n) else "FAIL"
    def speed(b):
        out = {}
        for rep in b:
            for c in rep.get("cases", []):
                out.setdefault(c.get("case"), []).append((c.get("decode_tok_per_s"), c.get("prefill_tok_per_s")))
        return out
    res = {"verdict": verdict, "equal": eq, "of": n, "why": why, "aux": {"tp2": a2, "tp4": a4},
           "speed_recorded_not_gated": {"tp2": speed(b2), "tp4": speed(b4)}}
    print(json.dumps(res))
    if md:
        with open(md, "w") as f:
            f.write(f"verdict: **{verdict}**, {eq} of {n} prompt-logits SHA equal (gate {sys.argv[1]})\n\n")
            f.write("| prompt | rows | TP2 sha256 | TP4 sha256 | |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n")
            for w in why: f.write(f"- {w}\n")
            f.write("\nspeed (recorded, not gated): " + json.dumps(res["speed_recorded_not_gated"]) + "\n")
    sys.exit(0 if verdict == "PASS" else 1)

main()
