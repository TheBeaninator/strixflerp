#!/usr/bin/env python3
"""ROUTER-GEMM (card c6473d6063): build gate/results/router-gemm/verdict.json from the committed evidence.

Every value is read from a file already committed under gate/results/router-gemm; nothing is measured here.
  R1, R2      k-r1r2-run2.jsonl (K job 20261114T102853Z-run-r1r2b, after the software-pipeline fix); run 1
              (k-r1r2-run1.jsonl, R2 FAIL at 0.171 before the fix, criteria unchanged) is reported beside it
  R3, R4      rgate/verdict-gate.json (python3 gate/lever_gate.py judge gate/results/router-gemm/rgate --arm R)
  R5          isa/rocm-7.2.4.tsv, isa/therock-10.1.tsv (tools/isa-scan.py)
  hypothesis  rprof/summary.json class "router" (gate/prefill_kprof.py over rprof/trace, 3 measured 4,096-row passes)
              and the K ratio at 35B dims, against arm G's 305 ms router (PREFILL-KPROF p2, 9506f4f116)
Rules: gate/router-gemm.toml. Usage: python3 gate/router_gemm_verdict.py [--check]
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "gate", "results", "router-gemm")
OUT = os.path.join(RES, "verdict.json")
KERNEL = "GateLogitsTiledKernel"
G_ROUTER_MS = 305.0   # arm G, PREFILL-KPROF run p2 (the card's PREDICTED comment)


def k_run(name):
    rows = [json.loads(l) for l in open(os.path.join(RES, name)) if l.strip()]
    rows = [r for r in rows if r.get("event") == "router"]
    shapes = [{"hidden": r["hidden"], "experts": r["experts"], "tokens": r["tokens"], "differ": r["differ"],
               "words": r["words"], "wide_ms": r["wide_ms"], "tiled_ms": r["tiled_ms"], "ratio": r["ratio"]}
              for r in rows]
    r2 = next(s for s in shapes if (s["hidden"], s["experts"], s["tokens"]) == (2048, 256, 4096))
    return {"shapes": shapes, "R1": all(s["differ"] == 0 for s in shapes) and len(shapes) == 4,
            "R2_ratio": r2["ratio"], "R2": r2["ratio"] <= 0.15}


def isa():
    out = {}
    for tc in ("rocm-7.2.4", "therock-10.1"):
        with open(os.path.join(RES, "isa", f"{tc}.tsv")) as f:
            rows = [r for r in csv.DictReader(f, delimiter="\t") if KERNEL in r["kernel"]]
        out[tc] = {"kernels": len(rows), "scratch_bytes": [int(r["scratch"]) for r in rows],
                   "vgpr": [int(r["vgpr"]) for r in rows], "lds": [int(r["lds"]) for r in rows],
                   "occupancy": [int(r["occ"]) for r in rows]}
    return out


def build():
    run1, run2 = k_run("k-r1r2-run1.jsonl"), k_run("k-r1r2-run2.jsonl")
    gate = json.load(open(os.path.join(RES, "rgate", "verdict-gate.json")))
    i = isa()
    router = json.load(open(os.path.join(RES, "rprof", "summary.json")))["classes"]["router"]
    ms, ratio = round(router["ms"], 2), run2["R2_ratio"]
    v = {
        "card": "c6473d6063",
        "criteria": "gate/router-gemm.toml",
        "R1": {"pass": run2["R1"], "shapes": run2["shapes"], "source": "k-r1r2-run2.jsonl"},
        "R2": {"pass": run2["R2"], "ratio": ratio, "max": 0.15, "source": "k-r1r2-run2.jsonl",
               "run1": {"pass": run1["R2"], "ratio": run1["R2_ratio"], "R1": run1["R1"],
                        "note": "before the software-pipeline fix (989b5a3); criteria unchanged"}},
        "R3": {"pass": gate["identity"]["pass"], "checked_steps": gate["identity"]["checked_steps"],
               "mismatches": len(gate["identity"]["mismatches"]), "source": "rgate/verdict-gate.json"},
        "R4": {"pass": gate["speed"]["pass"], "ratio_2048": round(gate["speed"]["ratio_2048"], 4),
               "ratio_16384": round(gate["speed"]["ratio_16384"], 4), "min": gate["speed"]["min2k"],
               "tok_s": {k: round(r["tok_s"], 1) for k, r in gate["runs"].items()}, "source": "rgate/verdict-gate.json"},
        "R5": {"pass": all(t["kernels"] > 0 and not any(t["scratch_bytes"]) for t in i.values()), "isa": i},
        "hypothesis": {
            "router_ms": ms, "router_runs_ms": [round(x, 2) for x in router["runs_ms"]], "arm_G_ms": G_ROUTER_MS,
            "k_ratio_35b": ratio, "confirmed_at": "router <= 60 ms or K ratio <= 0.20",
            "refuted_above": "router > 120 ms or K ratio > 0.40",
            "result": "confirmed" if (ms <= 60 or ratio <= 0.20) else "refuted" if (ms > 120 or ratio > 0.40) else "neither",
            "spread_note": "one of the three measured passes ran its router at ~3x the other two (runs_ms); unexplained, the median is quoted",
            "source": "rprof/summary.json class router, k-r1r2-run2.jsonl"},
        "fidelity_note": "rgate/champion.json carries G's KL vs FP32 0.00399: R was not measured against FP32, its logits are bit-identical to G's (R1, R3)",
        "champion": "proposal withdrawn 2026-11-14 (PM decision); the lever enters through PREFILL-LRC 5083008fe9",
    }
    v["pass"] = all(v[k]["pass"] for k in ("R1", "R2", "R3", "R4", "R5"))
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
    print(f"pass {v['pass']}: " + " ".join(f"{k} {v[k]['pass']}" for k in ("R1", "R2", "R3", "R4", "R5"))
          + f"; hypothesis {v['hypothesis']['result']} ({v['hypothesis']['router_ms']} ms, K {v['hypothesis']['k_ratio_35b']})")


if __name__ == "__main__":
    main()
