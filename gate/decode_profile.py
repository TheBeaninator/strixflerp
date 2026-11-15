#!/usr/bin/env python3
"""Per-kernel time of one 35B (Qwen3.6-35B-A3B Q4_K_M) world-1 decode step on control-node's GPU, under rocprofv3.
Run on control-node by gate/decode_profile_run.sh, which takes the control-gpu lease; this script only runs and parses.

    python3 gate/decode_profile.py run|summary

run      the R1-fold binary (build/xfer-r1-fold/after) --staged loopback, world 1, one short prompt (code.ids, 20
         tokens), under rocprofv3 --kernel-trace: four processes, graph on / off x --max-tokens 64 / 32. The decode
         kernels are the 64-token trace minus the 32-token trace (same prompt, same prefill), so 32 decode steps a
         graph mode with nothing of prefill or load left in them. A fifth process, graph on, 64 tokens, no profiler,
         gives the unprofiled decode ms a step to check the profiler's overhead.
summary  gate/results/decode-profile/summary.md: kernels grouped by role, ms a token, share, launches a token.

The LD_LIBRARY_PATH lesson (gate/lane_profile.py): the binary finds the ROCm runtime only through LD_LIBRARY_PATH, so
it is set on rocprofv3's own environment (rocprofv3 passes its environment on to the traced process).
"""
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DECODE_PROFILE_RES / DECODE_PROFILE_BIN: another results dir and binary (DECODE-KERN3 profiles build/kern3/after).
RES = os.path.join(ROOT, os.environ.get("DECODE_PROFILE_RES", "gate/results/decode-profile"))
M35 = "/data/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-Q4_K_M.gguf"
TR = "/data/tools/therock-10.1.0a20260823"
BIN = os.path.join(ROOT, os.environ.get("DECODE_PROFILE_BIN", "build/xfer-r1-fold/after/out/flerp-qwen35moe-generate"))
PROMPT = os.path.join(ROOT, "build", "xfer-r1-fold", "prompts", "code.ids")
POLITE = ["timeout", "1500", "ionice", "-c3", "nice", "-n", "19", "taskset", "-c", "0-3", "env", "OMP_NUM_THREADS=4",
          f"LD_LIBRARY_PATH={TR}/lib:{TR}/lib/llvm/lib"]
CASES = [("on", 64), ("on", 32), ("off", 64), ("off", 32)]
LONG, SHORT = 64, 32


def load1():
    with open("/proc/loadavg") as f:
        return float(f.read().split()[0])


def gen_args(graph, tokens):
    return [BIN, "--staged", "loopback", "--model", M35, "--context", "4096", "--max-tokens", str(tokens),
            "--configs", f"1:even:{graph}", "--epoch", "7", "--deadline-ms", "60000", "--single-reader", "--pause-ms",
            "100", "--ids-file", PROMPT]


def staged_events(text):
    out = []
    for line in text.splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("event") == "staged":
            out.append(e)
    return out


def run():
    os.makedirs(RES, exist_ok=True)
    rows = []
    for case in [(g, t, True) for g, t in CASES] + [("on", LONG, False)]:
        graph, tokens, prof = case
        name = f"{graph}-{tokens}" + ("" if prof else "-noprof")
        d = os.path.join(RES, "trace", name)
        cmd = gen_args(graph, tokens)
        if prof:
            subprocess.call(["rm", "-rf", d])
            cmd = [f"{TR}/bin/rocprofv3", "--kernel-trace", "--output-format", "csv", "-d", d, "-o", "kt", "--"] + cmd
        l0, t0 = load1(), time.time()
        with open(os.path.join(RES, f"{name}.err"), "w") as ef:
            p = subprocess.run(POLITE + cmd, stdout=subprocess.PIPE, stderr=ef, cwd=ROOT)
        out = p.stdout.decode()
        with open(os.path.join(RES, f"{name}.jsonl"), "w") as f:
            f.write(out)
        ev = staged_events(out)
        rows.append({"case": name, "graph": graph, "tokens": tokens, "profiled": prof, "rc": p.returncode,
                     "seconds": round(time.time() - t0, 1), "load_start": l0, "load_end": load1(),
                     "decode_ms_per_step": [e["stages"][0].get("decode_ms_per_step") for e in ev],
                     "tok_s": [e["calls"]["decode_calls"] / e["calls"]["decode_wall_ms"] * 1e3 for e in ev],
                     "decode_calls": [e["calls"]["decode_calls"] for e in ev],
                     "traces": sorted(glob.glob(os.path.join(d, "**", "*kernel_trace.csv"), recursive=True)) if prof else []})
        json.dump(rows, open(os.path.join(RES, "runs.json"), "w"), indent=1)
    return 0 if all(r["rc"] == 0 for r in rows) else 1


# --- summary --------------------------------------------------------------------------------------------------------

ROLES = [  # (role, regex on the full kernel name), first match wins; names read from the 2026-11-13 trace
    ("experts: routed + shared (IndexedExpertsKernel)", r"IndexedExpertsKernel"),
    ("router: gate logits (F32 ffn_gate_inp GEMV)", r"GateLogits(Wide)?Kernel"),
    ("router: top-k + MoE epilogue", r"RouterTopK|MoeEpilogue"),
    ("lm head (output.weight Q6_K GEMV)", r"Q8KBlockGEMVKernel_2Rows<4u, \(gufo::core::GgmlType\)0>|Q8KBlockGEMVKernel_2RowsWide<8u, \(gufo::core::GgmlType\)14>"),
    ("GDN: fused input projections", r"FusedSSMInputProjections"),
    ("GDN: recurrence + conv + q/g unpack", r"DeltaNet|SSMConv|UnpackQG"),
    ("attention: fused QKV projections", r"FusedQKVProjections"),
    ("attention: core + qk-norm/rope/kv write", r"Attention|RoPE"),
    ("out projections (attn_output / ssm_out, Q4_K GEMV)", r"Q8KBlockGEMVKernel_2Rows"),
    ("norms", r"Norm"),
    ("argmax / embed / residual / copies", r"Argmax|Embedding|Residual|rocclr"),
]


def role(name):
    n = name
    for r, rx in ROLES:
        if re.search(rx, n):
            return r
    return "other"


def read_trace(paths):
    """{kernel name: [count, total ns]} over every row of the traces, and the rows in time order."""
    rows = []
    for p in paths:  # runs.json holds the pod's absolute paths: read them under this tree's RES
        p = os.path.join(RES, "trace", p.split("/trace/", 1)[1])
        with open(p) as f:
            for r in csv.DictReader(f):
                rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
    rows.sort()
    agg = defaultdict(lambda: [0, 0])
    for s, e, n in rows:
        agg[n][0] += 1
        agg[n][1] += e - s
    return agg, rows


def short(name):
    n = re.sub(r"^void ", "", name).replace("(anonymous namespace)::", "")
    n = n.split("(gufo::core::GgmlType)")[0] + ("T" + n.split("(gufo::core::GgmlType)")[1][:2] if "(gufo::core::GgmlType)" in n else "")
    n = re.sub(r"\(.*$", "", n)
    return n[:90]


def summary():
    runs = {r["case"]: r for r in json.load(open(os.path.join(RES, "runs.json")))}
    L = ["# 35B world-1 decode: per-kernel time of one step (rocprofv3 kernel trace)", "",
         "Runner `gate/decode_profile.py` (+ `gate/decode_profile_run.sh` for the lease). Binary: "
         f"(`{os.path.relpath(BIN, ROOT)}`), `--staged loopback`, world 1, prompt `code` (20 tokens), context 4096, control-node GPU.",
         "Decode kernels = trace(64 tokens) minus trace(32 tokens), same prompt: 32 decode steps a graph mode, no "
         "prefill or load in them. GPU ms are kernel execution time (End - Start), summed; the step's wall time is "
         "the unprofiled decode ms a step.", "", "## Runs", "",
         "| case | rc | s | load start -> end | decode ms/step (tool) | tok/s (tool) | decode calls |", "|---|---|---|---|---|---|---|"]
    for r in runs.values():
        L.append(f"| {r['case']} | {r['rc']} | {r['seconds']} | {r['load_start']:.2f} -> {r['load_end']:.2f} | "
                 f"{', '.join(f'{x:.2f}' for x in r['decode_ms_per_step'] if x is not None)} | "
                 f"{', '.join(f'{x:.2f}' for x in r['tok_s'])} | {r['decode_calls']} |")
    data = {}
    for graph in ("on", "off"):
        a, ra = read_trace(runs[f"{graph}-{LONG}"]["traces"])
        b, rb = read_trace(runs[f"{graph}-{SHORT}"]["traces"])
        calls = [x - y for x, y in zip(runs[f"{graph}-{LONG}"]["decode_calls"], runs[f"{graph}-{SHORT}"]["decode_calls"])]
        steps = calls[0] if calls and calls[0] > 0 else LONG - SHORT
        diff = {}
        for n in set(a) | set(b):
            c = a.get(n, [0, 0])[0] - b.get(n, [0, 0])[0]
            ns = a.get(n, [0, 0])[1] - b.get(n, [0, 0])[1]
            if c or ns:
                diff[n] = (c / steps, ns / steps / 1e6)
        data[graph] = {"steps": steps, "kernels": diff, "rows_long": len(ra), "rows_short": len(rb)}
    json.dump({g: {"steps": d["steps"], "kernels": {k: list(v) for k, v in d["kernels"].items()}} for g, d in data.items()},
              open(os.path.join(RES, "kernels.json"), "w"), indent=1)
    for graph in ("on", "off"):
        d = data[graph]
        tot_ms = sum(v[1] for v in d["kernels"].values())
        launches = sum(v[0] for v in d["kernels"].values())
        wall = [x for x in runs.get(f"{graph}-{LONG}-noprof", runs[f"{graph}-{LONG}"])["decode_ms_per_step"] if x]
        L += ["", f"## Graph {graph}: {d['steps']} decode steps; {launches:.1f} kernels a token, {tot_ms:.3f} GPU ms a token"
              + (f"; tool's decode ms/step {wall[0]:.2f}" if wall else ""), "",
              f"Trace rows: {d['rows_long']} (64 tokens), {d['rows_short']} (32 tokens).", "",
              "### By role", "", "| role | ms a token | share of GPU ms | kernels a token |", "|---|---:|---:|---:|"]
        by = defaultdict(lambda: [0.0, 0.0])
        for n, (c, ms) in d["kernels"].items():
            by[role(n)][0] += ms
            by[role(n)][1] += c
        for r, (ms, c) in sorted(by.items(), key=lambda kv: -kv[1][0]):
            L.append(f"| {r} | {ms:.3f} | {100 * ms / tot_ms:.1f}% | {c:.1f} |")
        L += ["", "### By kernel (decode kernels only)", "", "| kernel | role | ms a token | share | launches a token | us a launch |",
              "|---|---|---:|---:|---:|---:|"]
        for n, (c, ms) in sorted(d["kernels"].items(), key=lambda kv: -kv[1][1]):
            if c <= 0 and abs(ms) < 0.005:
                continue
            L.append(f"| `{short(n)}` | {role(n)} | {ms:.3f} | {100 * ms / tot_ms:.1f}% | {c:.1f} | "
                     f"{(1e3 * ms / c) if c else 0:.1f} |")
    with open(os.path.join(RES, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit({"run": run, "summary": summary}[sys.argv[1]]())
