#!/usr/bin/env python3
"""PREFILL-KPROF (card 9506f4f116, gate/prefill-kprof.toml): per-kernel time of one 4,096-row prompt chunk of the 35B
(Qwen3.6-35B-A3B Q4_K_M) on arm G, on control-node's GPU, under rocprofv3 --kernel-trace.

Runs ON control-node as one tools/gpu-queue.sh job, in a clean export (POD_SYNC_DEST=/data/projects/strixflerp-kprof) whose
`./flerp gate recipe qwen36-35b-a3b.flerp-moe --stages build` built the binary.

    python3 gate/prefill_kprof.py run [--out DIR] [--bin PATH]   two processes, one load each:
        prof   rocprofv3 kernel trace, arms G x 4 (the first is warm-up), a 4,096-id prompt, 1 token; FLERP_KG_COUNTS=1
               prints the rows per expert of every block (host copy after the block's existing sync; no GPU work)
        plain  the same arms, no profiler: first_token_s a rep, to measure the profiler's overhead
    python3 gate/prefill_kprof.py summary DIR       DIR/summary.json + summary.md: per class ms, share, TFLOPS
    python3 gate/prefill_kprof.py port --out DIR    the port gate (amendment 2): one load, the 8 fidelity prompts of
        gate/prefill_ornith_35b.py and a 16,384-id prompt, arms G,L x 3 interleaved, --trace (logits sha256 a step)
    python3 gate/prefill_kprof.py judge-port DIR    P1 identity and P2 speed from DIR/port.jsonl
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "/data/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-Q4_K_M.gguf"
RECORDS = os.path.join(ROOT, "gate/results/ornith15-397b.p4/long.ids")
TR = "/data/tools/therock-10.1.0a20260823"
TOKENS = 4096
ARMS = os.environ.get("KPROF_ARMS", "G,G,G,G,G")      # amendment 1: two warm-up reps, then 3 measured
WARM = int(os.environ.get("KPROF_WARM", "2"))
PEAK = 59.0

# FLOPs a row (2 x params), from the GGUF header (card comment, predictions): 40 blocks, 30 GDN + 10 attention.
H, NB, NG, NA = 2048, 40, 30, 10
F_ROUTED = 8 * 3 * 2 * H * 512
F_SHARED = 3 * 2 * H * 512
F_DENSE = NG * 2 * H * (8192 + 4096 + 32 + 32) + NG * 2 * 4096 * H + NA * 2 * H * (8192 + 512 + 512) + NA * 2 * 4096 * H
F_ROUTER = NB * 2 * H * 256
F_ATTN_CHUNK = NA * 4 * 16 * 256 * TOKENS * (TOKENS + 1) / 2      # causal, depth 0
F_GDN = NG * 32 * (16384 + 4 * 16384 + 2 * 16384)                  # recurrence a row (VALU)


def binary(b):
    if b:
        return b
    c = glob.glob(os.path.join(ROOT, "build/qwen36-35b-a3b.flerp-moe/out/**/flerp-qwen35moe-generate"), recursive=True)
    if not c:
        raise SystemExit("no built flerp-qwen35moe-generate")
    return c[0]


def run_one(cmd, env, out, name):
    full = dict(os.environ, **env)
    full["LD_LIBRARY_PATH"] = TR + "/lib:" + TR + "/lib/llvm/lib"
    t0 = time.time()
    with open(os.path.join(out, name + ".err"), "w") as ef:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=ef, text=True, env=full, timeout=3000, cwd=out)
    open(os.path.join(out, name + ".jsonl"), "w").write(p.stdout)
    return {"name": name, "rc": p.returncode, "wall_s": round(time.time() - t0, 1)}


def cmd_run(a):
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    rec = open(RECORDS).read().split()
    p = os.path.join(out, "p4k.ids")
    open(p, "w").write(" ".join((rec * 10)[:TOKENS]) + "\n")
    gen = [binary(a.bin), "--model", MODEL, "--context", str(TOKENS + 256), "--ids-file", p, "--max-tokens", "1", "--arms", ARMS]
    rows = []
    d = os.path.join(out, "trace")
    subprocess.call(["rm", "-rf", d])
    rows.append(run_one([TR + "/bin/rocprofv3", "--kernel-trace", "--output-format", "csv", "-d", d, "-o", "kt", "--"] + gen,
                        {"FLERP_KG_COUNTS": "1"}, out, "prof"))
    rows.append(run_one(gen, {}, out, "plain"))
    json.dump(rows, open(os.path.join(out, "runs.json"), "w"), indent=1)
    print(json.dumps(rows))
    return 0 if all(r["rc"] == 0 for r in rows) else 1


def cmd_port(a):
    sys.path.insert(0, os.path.join(ROOT, "gate"))
    import prefill_ornith_35b as po
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    ps = po.prompts(out)
    rec = open(RECORDS).read().split()
    big = os.path.join(out, "p16k.ids")
    open(big, "w").write(" ".join((rec * 10)[:16384]) + "\n")
    ids = [x for p in ps + [big] for x in ("--ids-file", p)]
    gen = [binary(a.bin), "--model", MODEL, "--context", "16640"] + ids + ["--trace", "--trace-top", "2", "--max-tokens", "2",
                                                                          "--arms", "G,L,G,L,G,L"]
    r = run_one(gen, {}, out, "port")
    json.dump([r], open(os.path.join(out, "runs.json"), "w"), indent=1)
    print(json.dumps(r))
    return 0 if r["rc"] == 0 else 1


def cmd_judge_port(a):
    sys.path.insert(0, os.path.join(ROOT, "gate"))
    import prefill_ornith_35b as po
    ev = [json.loads(l) for l in open(os.path.join(a.dir, "port.jsonl")) if l.startswith("{")]
    arms = po.split_arms(ev)
    tr = [(arm, {e["prompt"]: e for e in es if e.get("event") == "traced"}) for arm, es in arms]
    ref = next(t for arm, t in tr if arm == "G")
    mism, checked = [], 0
    for arm, t in tr:
        for p, e in ref.items():
            x = t.get(p)
            if x is None:
                mism.append({"arm": arm, "prompt": p, "missing": True})
                continue
            for i, (s0, s1) in enumerate(zip(e["steps"], x["steps"])):
                checked += 1
                if s0.get("sha256") != s1.get("sha256") or not s0.get("sha256"):
                    mism.append({"arm": arm, "prompt": p, "step": i})
            if e["ids"] != x["ids"]:
                mism.append({"arm": arm, "prompt": p, "ids": True})
    sp = defaultdict(list)
    for arm, t in tr:
        for p, e in t.items():
            sp[(arm, e["prompt_tokens"])].append(e["first_token_s"])
    med = lambda xs: sorted(xs)[len(xs) // 2]
    speed = {"%s@%d" % k: {"runs": v, "median_s": med(v), "tok_s": k[1] / med(v)} for k, v in sp.items() if k[1] in (2048, 16384)}
    g2, l2 = speed.get("G@2048", {}).get("tok_s"), speed.get("L@2048", {}).get("tok_s")
    g16, l16 = speed.get("G@16384", {}).get("tok_s"), speed.get("L@16384", {}).get("tok_s")
    res = {"P1": {"checked_steps": checked, "mismatches": mism, "pass": checked > 0 and not mism},
           "P2": {"L_over_G_2048": l2 / g2 if g2 and l2 else None, "L_over_G_16384": l16 / g16 if g16 and l16 else None},
           "speed": speed}
    res["P2"]["pass"] = bool(res["P2"]["L_over_G_2048"] and res["P2"]["L_over_G_2048"] >= 1.00 and
                             res["P2"]["L_over_G_16384"] and res["P2"]["L_over_G_16384"] >= 1.03)
    json.dump(res, open(os.path.join(a.dir, "verdict-port.json"), "w"), indent=1)
    print(json.dumps({"P1": {k: v for k, v in res["P1"].items() if k != "mismatches"}, "mismatches": len(mism), "P2": res["P2"],
                      "speed": {k: (round(v["median_s"], 4), round(v["tok_s"], 1)) for k, v in speed.items()}}, indent=1))


# --- summary --------------------------------------------------------------------------------------------------------
# The kgrouped kernel serves routed experts, the shared expert and the dense projections; which one a launch belongs to
# is read from the plan kernel before it: CountKernel = routed grouping; IdentityKernel right after a routed launch =
# shared expert (LaunchGrouped); any other IdentityKernel = a dense projection (LaunchDenseGrouped).
CLASSES = [
    ("attention", r"Attention|RoPE|QKNorm|KvWrite|UnpackQG|PerHeadRMSNorm"),
    ("gdn_core", r"DeltaNet|SSMConv|SSMPostNorm|Ssm"),
    ("router", r"GateLogits|RouterTopK"),
    ("moe_combine", r"MoeEpilogue"),
    ("norms_residual", r"RMSNorm|Residual|Norm"),
    ("convert_group", r"ToF16|FloatToFp16|FloatToBfloat16|Quantize|CountKernel|ScanKernel|PlaceKernel|IdentityKernel"),
    ("embed_head_misc", r"Embedding|GEMV|Argmax|rocclr|Fill|Copy|copy"),
]


def classify(rows):
    """rows: [(start, end, name)] of one rep in time order -> [(class, sub, ns, name)]."""
    out, state, prev = [], "dense", ""
    for s, e, n in rows:
        if "CountKernel" in n:
            state = "routed"
        elif "IdentityKernel" in n:
            state = "shared" if (state == "routed" and "GroupedKernel" in prev) else "dense"
        if "GroupedKernel" in n:
            gu = bool(re.search(r"GroupedKernel<\d+, \([^)]*\)\d+, true(, (true|false))?>", n))
            cls = {"routed": "experts_routed", "shared": "experts_shared", "dense": "dense_proj"}[state]
            out.append((cls, ("gate_up" if gu else "down") if state != "dense" else "linear", e - s, n))
        else:
            cls = next((c for c, rx in CLASSES if re.search(rx, n)), "other")
            if cls == "other" and re.search(r"GEMM|Gemm", n):
                cls = "dense_proj"
            out.append((cls, "", e - s, n))
        prev = n
    return out


def reps(trace_rows):
    """Split the trace into prefill reps at each BatchedEmbeddingLookup launch (one a chunk of 4,096 rows)."""
    # A rep = from a BatchedEmbeddingLookup (the prompt chunk) to the next embedding launch of any kind (the one-token
    # decode step that --max-tokens 1 still runs, EmbeddingLookupPtrKernel, is left out).
    emb = [i for i, r in enumerate(trace_rows) if "EmbeddingLookup" in r[2]] + [len(trace_rows)]
    starts = [i for i in emb[:-1] if "BatchedEmbeddingLookup" in trace_rows[i][2]]
    return [trace_rows[a:next(j for j in emb if j > a)] for a in starts], len(starts)


def cmd_summary(a):
    d = a.dir
    rows = []
    for p in glob.glob(os.path.join(d, "trace", "**", "*kernel_trace.csv"), recursive=True):
        with open(p) as f:
            for r in csv.DictReader(f):
                rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
    rows.sort()
    rs, n_emb = reps(rows)
    measured = rs[WARM:] if len(rs) > WARM else rs      # the first WARM reps are warm-up (p1: 1)
    per = []
    for r in measured:
        agg = defaultdict(float)
        sub = defaultdict(float)
        kern = defaultdict(lambda: [0, 0.0])
        for cls, sb, ns, n in classify(r):
            agg[cls] += ns / 1e6
            if sb:
                sub[cls + ":" + sb] += ns / 1e6
            kern[n][0] += 1
            kern[n][1] += ns / 1e6
        per.append({"classes": dict(agg), "sub": dict(sub), "kernels": {k: v for k, v in kern.items()},
                    "gpu_ms": sum(agg.values()), "span_ms": (r[-1][1] - r[0][0]) / 1e6, "launches": len(r)})
    med = lambda xs: statistics.median(xs) if xs else None
    classes = sorted({c for p in per for c in p["classes"]})
    tot = med([p["gpu_ms"] for p in per])
    fl = {"experts_routed": F_ROUTED * NB * TOKENS, "experts_shared": F_SHARED * NB * TOKENS, "dense_proj": F_DENSE * TOKENS,
          "attention": F_ATTN_CHUNK, "gdn_core": F_GDN * TOKENS, "router": F_ROUTER * TOKENS}
    res = {"tokens": TOKENS, "reps_measured": len(per), "embedding_launches": n_emb, "gpu_ms_median": tot,
           "span_ms_median": med([p["span_ms"] for p in per]), "classes": {}}
    for c in classes:
        ms = med([p["classes"].get(c, 0.0) for p in per])
        e = {"ms": ms, "share": ms / tot, "runs_ms": [p["classes"].get(c, 0.0) for p in per]}
        if c in fl and ms:
            e["tflops"] = fl[c] / (ms * 1e-3) / 1e12
            e["pct_peak"] = e["tflops"] / PEAK
        res["classes"][c] = e
    res["sub"] = {k: med([p["sub"].get(k, 0.0) for p in per]) for k in sorted({k for p in per for k in p["sub"]})}
    top = sorted(per[0]["kernels"].items(), key=lambda kv: -kv[1][1])[:40]
    res["top_kernels_rep1"] = [{"name": k[:160], "calls": v[0], "ms": round(v[1], 3)} for k, v in top]
    # rows per expert
    counts = []
    for line in open(os.path.join(d, "prof.err"), errors="replace"):
        if line.startswith('{"event":"kg_counts"'):
            e = json.loads(line)
            if e["tokens"] == TOKENS:
                counts.append(e["counts"])
    if counts:
        blk = counts[WARM * NB:(WARM + 1) * NB] if len(counts) >= (WARM + 1) * NB else counts[:NB]    # first measured rep
        flat = sorted(c for b in blk for c in b)
        q = lambda f: flat[min(len(flat) - 1, int(f * len(flat)))]
        res["rows_per_expert"] = {"blocks": len(blk), "experts": len(blk[0]), "mean": sum(flat) / len(flat), "zero": sum(1 for c in flat if c == 0),
                                  "p10": q(0.10), "p25": q(0.25), "p50": q(0.5), "p75": q(0.75), "p90": q(0.90), "p99": q(0.99), "max": flat[-1],
                                  "lt32": sum(1 for c in flat if c < 32) / len(flat), "lt64": sum(1 for c in flat if c < 64) / len(flat),
                                  "tile_fill_bm128": sum(flat) / sum(128 * -(-c // 128) for c in flat if c),
                                  "per_block_max": [max(b) for b in blk], "per_block_min": [min(b) for b in blk]}
    plain = [json.loads(l) for l in open(os.path.join(d, "plain.jsonl")) if l.startswith("{")]
    prof = [json.loads(l) for l in open(os.path.join(d, "prof.jsonl")) if l.startswith("{")]
    ft = lambda ev: [e.get("first_token_s") for e in ev if e.get("event") == "generated"]
    res["first_token_s_plain"] = ft(plain)
    res["first_token_s_prof"] = ft(prof)
    json.dump(res, open(os.path.join(d, "summary.json"), "w"), indent=1)
    L = ["| class | ms | share | TFLOPS | % of 59 |", "|---|---|---|---|---|"]
    for c, e in sorted(res["classes"].items(), key=lambda kv: -kv[1]["ms"]):
        L.append("| %s | %.1f | %.1f%% | %s | %s |" % (c, e["ms"], 100 * e["share"], "%.1f" % e["tflops"] if "tflops" in e else "-",
                                                    "%.0f%%" % (100 * e["pct_peak"]) if "pct_peak" in e else "-"))
    L.append("| total GPU | %.1f | | %.1f | |" % (tot, sum(fl.values()) / (tot * 1e-3) / 1e12))
    open(os.path.join(d, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(json.dumps({k: v for k, v in res.items() if k not in ("classes", "top_kernels_rep1")}, indent=0)[:3000])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", default=os.path.join(ROOT, "build/kprof"))
    r.add_argument("--bin")
    s = sub.add_parser("summary")
    s.add_argument("dir")
    pp = sub.add_parser("port")
    pp.add_argument("--out", required=True)
    pp.add_argument("--bin")
    jp = sub.add_parser("judge-port")
    jp.add_argument("dir")
    a = ap.parse_args()
    sys.exit({"run": cmd_run, "summary": cmd_summary, "port": cmd_port, "judge-port": cmd_judge_port}[a.cmd](a) or 0)


if __name__ == "__main__":
    main()
