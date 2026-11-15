#!/usr/bin/env python3
"""ORNITH-KERN3 (criteria gate/ornith-kern3.toml): the DECODE-KERN3 bench at Ornith-1.5 397B dims, synthetic codes.
Run on control-node by gate/ornith_kern3_run.sh (GPU lease); this script runs build/kern3/decode-kern3 and judges.

    python3 gate/ornith_kern3.py [--backend kbench|queue] kernel|summary
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import xfer_r1_fold as R1  # noqa: E402
import kbench_backend as KB  # noqa: E402

BACKEND, ARGV = KB.split_backend(sys.argv[1:])     # --backend kbench (default) | queue
RES = KB.results_dir(os.path.join(ROOT, "gate", "results", "ornith-kern3"), BACKEND)
P5_STAGE_MS, P5_TOKEN_MS = 18.5, 83.0
TIMED = ("router-4096x512", "outproj-q4k-4096x8192")


def cases():
    c = [("router-4096x512", ["--kind", "router", "--hidden", "4096", "--experts", "512", "--tokens-equal", "1,2,3,8,21,128,1024"]),
         ("outproj-q4k-4096x8192", ["--kind", "gemv", "--type", "q4k", "--rows", "4096", "--k", "8192"])]
    for n, rows in (("ssm-alpha-q4k-64x4096", 64), ("attn-kv-q4k-512x4096", 512), ("attn-gate-q4k-8192x4096", 8192),
                    ("attn-q-q4k-16384x4096", 16384), ("attn-qkv-q4k-12288x4096", 12288)):
        c.append((n, ["--kind", "gemv", "--type", "q4k", "--rows", str(rows), "--k", "4096", "--no-timing"]))
    return c


def kernel():
    os.makedirs(RES, exist_ok=True)
    if BACKEND == "kbench":
        cs = [(n, a + (["--cold-mib", "512"] if n in TIMED else [])) for n, a in cases()]
        KB.run_cases_k(ROOT, RES, "kernel-cold", "build/kern3/decode-kern3", cs, "strixflerp ORNITH-KERN3 kernel bench")
        return
    out, err = os.path.join(RES, "kernel-cold.jsonl"), os.path.join(RES, "kernel-cold.err")
    open(out, "w").close()
    loads = []
    for name, args in cases():
        while R1.load1() > 6:
            time.sleep(30)
        extra = ["--cold-mib", "512"] if name in TIMED else []
        rc, l0, l1 = R1.run(["build/kern3/decode-kern3", "--case", name] + args + extra, out, err)
        loads.append({"case": name, "rc": rc, "load_start": l0, "load_end": l1})
        json.dump(loads, open(os.path.join(RES, "kernel-cold-load.json"), "w"), indent=1)


def summary():
    rows = [json.loads(x) for x in open(os.path.join(RES, "kernel-cold.jsonl")) if x.strip()]
    ld = json.load(open(os.path.join(RES, "kernel-cold-load.json")))
    bad = [x for x in ld if x["load_start"] > 6 or x["load_end"] > 6 or x["rc"] != 0]
    diff = [r for r in rows if r["outputs_differing"] != 0 or r["gap_touched"] != 0]
    L = [f"# ORNITH-KERN3: DECODE-KERN3 kernels at Ornith-1.5 397B dims (synthetic codes), backend {BACKEND}", "",
         "Criteria `gate/ornith-kern3.toml` (committed before the GPU run). Bench build/kern3/decode-kern3 (catalogue at 83d5366).", "",
         "Run: " + KB.method_line(RES, "kernel-cold"), "",
         f"{len(rows)} (case, x, tokens) results, {sum(r['outputs'] for r in rows)} outputs compared; with any differing byte: {len(diff)} -> "
         f"**{'PASS' if not diff and rows else 'FAIL'}**. Cases off-load or failed: {len(bad)} {bad if bad else ''}", "",
         "| case | rows x k | differing | ref GB/s | new GB/s | new/ref | ref us | new us |", "|---|---|---|---|---|---|---|---|"]
    save = {}
    names = []
    for r in rows:
        if r["case"] not in names:
            names.append(r["case"])
    for c in names:
        rs = [r for r in rows if r["case"] == c]
        t = next((r["timing"] for r in rs if r["timing"]), None)
        nd, no = sum(r["outputs_differing"] for r in rs), sum(r["outputs"] for r in rs)
        if t:
            us = {a: t[a].get("us") or t[a].get("us_median") for a in ("ref", "new")}
            save[c] = (us["ref"] - us["new"]) if us["ref"] and us["new"] else None
            L.append(f"| {c} | {rs[0]['rows']} x {rs[0]['k']} | {nd} of {no} | {t['ref']['GBps']} ({t['ref']['GBps_min']}..{t['ref']['GBps_max']}) | "
                     f"{t['new']['GBps']} ({t['new']['GBps_min']}..{t['new']['GBps_max']}) | {t['speedup']:.3f} | {us['ref']} | {us['new']} |")
        else:
            L.append(f"| {c} | {rs[0]['rows']} x {rs[0]['k']} | {nd} of {no} | - | - | - | - | - |")
    if all(save.get(c) is not None for c in TIMED):
        s = sum(save[c] for c in TIMED) / 1e3
        L += ["", "## Projection (computed, not measured; criteria [projection])", "",
              f"Saving a block {s:.3f} ms (router {save[TIMED[0]]/1e3:.3f} + outproj {save[TIMED[1]]/1e3:.3f}).",
              f"Stage (15 blocks): {P5_STAGE_MS} -> {P5_STAGE_MS - 15 * s:.2f} ms a step.",
              f"World 4 token: {P5_TOKEN_MS} -> {P5_TOKEN_MS - 60 * s:.2f} ms = {1000 / P5_TOKEN_MS:.2f} -> {1000 / (P5_TOKEN_MS - 60 * s):.2f} tok/s (projected)."]
    open(os.path.join(RES, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"kernel": kernel, "summary": summary}[ARGV[0]]()
