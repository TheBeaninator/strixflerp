#!/usr/bin/env python3
"""MOE-COMBINE (card 2dc55e67f0): build gate/results/moe-combine/verdict.json from the committed evidence.

Every value is read from a file already committed under gate/results/moe-combine; nothing is measured here.
  cause, C0   k-cause1.jsonl (K job 20261114T102419Z-run-cause1, gate/combine_bench.hip)
  C2, C3      cgate/verdict-gate.json (python3 gate/lever_gate.py judge gate/results/moe-combine/cgate --arm C)
  C4          isa/rocm-7.2.4.tsv, isa/therock-10.1.tsv (tools/isa-scan.py)
  hypothesis  cprof/trace/kt_kernel_trace.csv (rocprofv3, 5 prompt passes of 4,096 rows; the last 3 are measured,
              as gate/prefill_kprof.py does), against arm G's 279 ms combine + 69 ms ToF16 (PREFILL-KPROF p2, 9506f4f116)
Rules: gate/moe-combine.toml. Usage: python3 gate/moe_combine_verdict.py [--check]
"""
import csv
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "gate", "results", "moe-combine")
OUT = os.path.join(RES, "verdict.json")
KERNEL = "MoeCombineVec4Kernel"
G_COMBINE_MS, G_TOF16_MS = 279.0, 69.0   # arm G, PREFILL-KPROF run p2 (the card's PREDICTED comment)


def cause():
    rows = [json.loads(l) for l in open(os.path.join(RES, "k-cause1.jsonl")) if l.strip()]
    dims, cur = [], None
    for r in rows:
        if r["event"] == "combine":
            key = (r["tokens"], r["slots"], r["dim"])
            if cur is None or cur["dims"] != list(key):
                cur = {"dims": list(key), "gbps": {}}
                dims.append(cur)
            cur["gbps"][f'{r["kernel"]}.{r["mode"]}'] = r["gbps"]
        elif r["event"] == "identity":
            cur["differ"], cur["words"] = r["orig_vs_vec4o_differ"], r["words"]
    for d in dims:
        g = d["gbps"]
        orig, after = g["orig.isolated"], g["orig.after_writer"]
        best = max(g["vec4o.isolated"], g["probe.isolated"])
        if orig < 100 and best >= 1.5 * orig:
            d["rule"] = "kernel_shape"
        elif orig >= 100 and after < 0.6 * orig:
            d["rule"] = "in_context"
        elif g["probe.isolated"] < 100:
            d["rule"] = "memory"
        else:
            d["rule"] = "none"
    return dims


def isa():
    out = {}
    for tc in ("rocm-7.2.4", "therock-10.1"):
        with open(os.path.join(RES, "isa", f"{tc}.tsv")) as f:
            rows = [r for r in csv.DictReader(f, delimiter="\t") if KERNEL in r["kernel"]]
        out[tc] = {"kernels": len(rows), "scratch_bytes": [int(r["scratch"]) for r in rows],
                   "vgpr": [int(r["vgpr"]) for r in rows]}
    return out


def in_model():
    starts = {"combine": [], "tof16": []}
    with open(os.path.join(RES, "cprof", "trace", "kt_kernel_trace.csv")) as f:
        for r in csv.DictReader(f):
            n = r["Kernel_Name"]
            k = "combine" if KERNEL in n else "tof16" if "ToF16Kernel" in n else None
            if k:
                starts[k].append((int(r["Start_Timestamp"]), (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e6))
    per = {}
    for k, v in starts.items():
        v.sort()
        n_pass = 5
        size = len(v) // n_pass
        groups = [sum(ms for _, ms in v[i * size:(i + 1) * size]) for i in range(n_pass)]
        per[k] = {"launches_a_pass": size, "passes_ms": [round(x, 2) for x in groups],
                  "measured_median_ms": round(statistics.median(groups[2:]), 2)}
    return per


def build():
    c = cause()
    gate = json.load(open(os.path.join(RES, "cgate", "verdict-gate.json")))
    i = isa()
    m = in_model()
    both = m["combine"]["measured_median_ms"] + m["tof16"]["measured_median_ms"]
    v = {
        "card": "2dc55e67f0",
        "criteria": "gate/moe-combine.toml",
        "cause": {"rule": sorted({d["rule"] for d in c}), "by_dims": c,
                  "source": "k-cause1.jsonl (K job 20261114T102419Z-run-cause1)"},
        "C0": {"pass": all(d["differ"] == 0 for d in c),
               "differ_by_dims": {"x".join(map(str, d["dims"])): [d["differ"], d["words"]] for d in c}},
        "C2": {"pass": gate["identity"]["pass"], "checked_steps": gate["identity"]["checked_steps"],
               "mismatches": len(gate["identity"]["mismatches"]), "source": "cgate/verdict-gate.json"},
        "C3": {"pass": gate["speed"]["pass"], "ratio_2048": round(gate["speed"]["ratio_2048"], 4),
               "ratio_16384": round(gate["speed"]["ratio_16384"], 4), "min": gate["speed"]["min2k"],
               "tok_s": {k: round(r["tok_s"], 1) for k, r in gate["runs"].items()}, "source": "cgate/verdict-gate.json"},
        "C4": {"pass": all(t["kernels"] > 0 and not any(t["scratch_bytes"]) for t in i.values()), "isa": i},
        "hypothesis": {
            "combine_ms": m["combine"]["measured_median_ms"], "tof16_ms": m["tof16"]["measured_median_ms"],
            "combine_plus_tof16_ms": round(both, 2), "arm_G_ms": {"combine": G_COMBINE_MS, "tof16": G_TOF16_MS},
            "confirmed_at_ms": 150, "refuted_above_ms": 250,
            "result": "confirmed" if both <= 150 else "refuted" if both > 250 else "neither",
            "per_pass": m, "source": "cprof/trace/kt_kernel_trace.csv, passes 3-5 measured (1-2 warm-up)"},
        "champion": "proposal withdrawn 2026-11-14 (PM decision); the lever enters through PREFILL-LRC 5083008fe9",
    }
    v["pass"] = all(v[k]["pass"] for k in ("C0", "C2", "C3", "C4"))
    return v


def main():
    v = build()
    text = json.dumps(v, indent=1) + "\n"
    if "--check" in sys.argv:
        same = os.path.exists(OUT) and open(OUT).read() == text
        print("verdict.json up to date" if same else "verdict.json differs from the evidence")
        sys.exit(0 if same else 1)
    with open(OUT, "w") as f:
        f.write(text)
    print(f"pass {v['pass']}: C0 {v['C0']['pass']} C2 {v['C2']['pass']} C3 {v['C3']['pass']} C4 {v['C4']['pass']}; "
          f"cause {v['cause']['rule']}; hypothesis {v['hypothesis']['result']} ({v['hypothesis']['combine_plus_tof16_ms']} ms)")


if __name__ == "__main__":
    main()
