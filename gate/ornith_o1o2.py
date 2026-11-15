#!/usr/bin/env python3
"""ORNITH-O1O2 on one node: pinned stage rows and a token-only reply (O1), prompt sub-chunks back to back (O2), on the
35B as four TCP rank processes on control-node's one GPU. Criteria: gate/ornith-o1o2.toml (committed before any GPU
run). Run ON control-node, in the clean tree of [build] tree:

    python3 gate/ornith_o1o2.py build      # CPU only: flerp assemble + build.sh under nice/ionice
    python3 gate/ornith_o1o2.py run        # the GPU job (start it through tools/gpu-queue.sh): the four launches
    python3 gate/ornith_o1o2.py judge      # re-judge saved events (no GPU)

No hold, no lease, no worker: tools/gpu-queue.sh gives the GPU. Evidence: gate/results/ornith-o1o2/.
"""
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gate"))
import moe_placed as P  # noqa: E402
import moe_stage as MS  # noqa: E402

CONFIG = "gate/ornith-o1o2.toml"
RES = os.path.join(ROOT, "gate", "results", "ornith-o1o2")


def context():
    import gufo_recipe as G               # noqa: E402
    import strixite_recipe as S           # noqa: E402
    from flerplib.catalog import Catalog
    P.G, P.S = G, S
    cfg = S.load_config(os.path.join(ROOT, CONFIG))
    cat = Catalog(ROOT)
    os.makedirs(RES, exist_ok=True)
    ctx = {"cat": cat, "recipe": cat.recipe(cfg["recipe"]), "cfg": cfg, "res": RES, "config": CONFIG,
           "scratch": os.path.join(ROOT, cfg["host"]["scratch"]), "disk": []}
    os.makedirs(ctx["scratch"], exist_ok=True)
    return ctx


# --- judgements (pure) -----------------------------------------------------------------------------------------
def spec_list(text):
    out = []
    for item in text.split(","):
        g, r, s = item.split(":")
        out.append({"graph": g, "reply": r, "subchunks": int(s)})
    return out


def judge_launch(k, launch, rank_out, names, refs):
    """One launch: the criteria 1-4 rows. rank_out: {rank: {"rc", "events"}}. Returns (record, [failures])."""
    why, runs = [], []
    for r, info in sorted(rank_out.items()):
        if info["rc"] != 0:
            why.append(f"launch {k}: rank {r} exited {info['rc']}")
    staged = [e for e in rank_out.get(0, {}).get("events", []) if e.get("event") == "staged"]
    for e in rank_out.get(0, {}).get("events", []):
        if e.get("event") in ("staged_failed", "error", "relay_failed"):
            why.append(f"launch {k}: rank 0 {e.get('event')}: {e.get('error', e.get('reason', ''))}")
    timing = {}
    for r, info in rank_out.items():
        for e in info["events"]:
            if e.get("event") == "stage_timing":
                timing.setdefault(e["run"], {})[r] = e["timing"]
                timing[e["run"]].setdefault("_relay", {})[r] = e["relay"]
    specs = spec_list(launch["specs"])
    want = len(specs) * len(names)
    if len(staged) != want:
        why.append(f"launch {k}: {len(staged)} staged results, want {want}")
    for i, e in enumerate(staged):
        spec = specs[i // len(names)] if i // len(names) < len(specs) else {}
        p = names[e["prompt"]]
        ref = refs[p]
        tag = f"launch {k} ({launch['rows']}) {e['reply']}:{e['subchunks']} {p}"
        if e["reply"] != spec.get("reply") or e["subchunks"] != spec.get("subchunks") or e["rows_mode"] != launch["rows"]:
            why.append(f"{tag}: run {i} is not the spec {spec} / rows {launch['rows']}")
        if e["reply"] == "full":
            why += [f"{tag}: {w}" for w in MS.judge_staged(e, ref)]
        elif e["ids"] != ref["ids"]:
            why.append(f"{tag}: ids differ from P1's at {MS.first_diff(e['ids'], ref['ids'])}")
        st = {0: e["stages"][0]}
        st.update({r: t for r, t in timing.get(i, {}).items() if r != "_relay"})
        copy = {}
        for r in (1, 2, 3):
            t = st.get(r)
            if t is None:
                why.append(f"{tag}: no stage_timing from rank {r}")
                continue
            calls = t["decode_steps"] + len(t["chunks"])
            copy[r] = t["copy_ms"] / max(calls, 1)
            if launch["rows"] == "pinned" and not copy[r] < 0.1:
                why.append(f"{tag}: rank {r} copy {copy[r]:.4f} ms a call (bound < 0.1)")
        calls = e["calls"]
        runs.append({
            "launch": k, "rows": launch["rows"], "reply": e["reply"], "subchunks": e["subchunks"], "prompt": p,
            "prompt_tokens": e["prompt_tokens"], "equal_ids": e["ids"] == ref["ids"],
            "equal_sha": e["reply"] == "full" and e["logits_sha256"] == ref["logits_sha256"],
            "ttft_s": e["prefill_s"], "decode_ms_per_token": calls["decode_wall_ms"] / max(calls["decode_calls"], 1),
            "copy_ms_per_call": copy,
            "stage_chunk_ms": {r: round(sum(c["ms"] for c in t["chunks"]), 3) for r, t in st.items()},
            "stage_chunks": {r: len(t["chunks"]) for r, t in st.items()},
            "stage_decode_ms_per_step": {r: t["decode_ms_per_step"] for r, t in st.items()},
            "last_logits_copy_ms_per_call": (st.get(3) or {}).get("logits_copy_ms", 0) / max(calls["decode_calls"] + 1, 1),
            "rank3_bytes_sent": ((timing.get(i, {}).get("_relay") or {}).get(3) or {}).get("bytes_sent"),
        })
    return {"launch": k, "rows": launch["rows"], "specs": launch["specs"], "runs": runs,
            "rc": {r: i["rc"] for r, i in rank_out.items()}}, why


def summarize(doc):
    runs = [r for L in doc["launches"] for r in L["runs"]]
    L = ["# ORNITH-O1O2 on one node: 35B, world 4, TCP on 127.0.0.1, one GPU", "",
         f"Written {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}. Criteria: `gate/ornith-o1o2.toml` (committed before the "
         f"first GPU run). Binary sha256 {doc.get('binary_sha256', '?')}.", "",
         f"**{'PASS' if not doc['failures'] else 'FAIL'}**: {len(doc['failures'])} failure(s).", ""]
    L += ["## Identity (ids / every logits sha256 against P1)", "", "| launch | rows | reply:S | prompts ids equal | prompts sha equal |", "|---|---|---|---|---|"]
    for Lr in doc["launches"]:
        by = {}
        for r in Lr["runs"]:
            by.setdefault(f"{r['reply']}:{r['subchunks']}", []).append(r)
        for key, rs in by.items():
            sha = sum(r["equal_sha"] for r in rs) if rs[0]["reply"] == "full" else "n/a"
            L.append(f"| {Lr['launch']} | {Lr['rows']} | {key} | {sum(r['equal_ids'] for r in rs)}/{len(rs)} | {sha}{'' if sha == 'n/a' else '/' + str(len(rs))} |")
    L += ["", "## O1: copy ms a call, ranks 1 / 2 / 3 (median over sessions; max)", "", "| rows | rank 1 | rank 2 | rank 3 |", "|---|---|---|---|"]
    for mode in ("pageable", "pinned"):
        cells = []
        for r in ("1", "2", "3"):
            v = [x["copy_ms_per_call"].get(r, x["copy_ms_per_call"].get(int(r))) for x in runs if x["rows"] == mode]
            v = [x for x in v if x is not None]
            cells.append(f"{statistics.median(v):.4f} (max {max(v):.4f})" if v else "-")
        L.append(f"| {mode} | " + " | ".join(cells) + " |")
    L += ["", "## Decode ms a token (rank 0 wall a call), by launch, S = 1, mean over the four prompts", "", "| launch | rows | full | token |", "|---|---|---|---|"]
    for Lr in doc["launches"]:
        f = [r["decode_ms_per_token"] for r in Lr["runs"] if r["subchunks"] == 1 and r["reply"] == "full"]
        t = [r["decode_ms_per_token"] for r in Lr["runs"] if r["subchunks"] == 1 and r["reply"] == "token"]
        L.append(f"| {Lr['launch']} | {Lr['rows']} | {statistics.mean(f):.3f} | {statistics.mean(t):.3f} |" if f and t else f"| {Lr['launch']} | {Lr['rows']} | - | - |")
    L += ["", "## O2: TTFT s (rank 0 prefill_s) and summed stage chunk ms, pinned launches, reply full", "",
          "| launch | prompt | tokens | S | TTFT s | chunk ms a stage (0/1/2/3) |", "|---|---|---|---|---|---|"]
    for r in runs:
        if r["rows"] == "pinned" and r["reply"] == "full":
            ch = "/".join(str(r["stage_chunk_ms"].get(k, r["stage_chunk_ms"].get(str(k), "-"))) for k in (0, 1, 2, 3))
            L.append(f"| {r['launch']} | {r['prompt']} | {r['prompt_tokens']} | {r['subchunks']} | {r['ttft_s']:.3f} | {ch} |")
    if doc["failures"]:
        L += ["", "## Failures (first 40)", ""] + [f"- {w}" for w in doc["failures"][:40]]
    with open(os.path.join(RES, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")


# --- commands --------------------------------------------------------------------------------------------------
def build():
    ctx = context()
    log = os.path.join(RES, "build.txt")
    name = ctx["recipe"].name
    with open(log, "w") as f:
        f.write(f"# {P.G.utc()} assemble + build {name} in {ROOT}\n")
    rc = subprocess.call([sys.executable, "flerp", "assemble", name], cwd=ROOT, stdout=open(log, "a"), stderr=subprocess.STDOUT)
    if rc != 0:
        return rc
    host = P.S.load_config(os.path.join(ROOT, "gate", "qwen36-35b-a3b.flerp-moe.toml"))["host"]
    ctx["cfg"]["host"].update({k: host[k] for k in ("openssl_headers", "openssl_crypto")})
    cmd = list(ctx["cfg"]["host"]["polite"]) + [os.path.join(ROOT, "build", name, "build.sh"), "-j6"] + P.G.openssl_options(ctx)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=ROOT, stdout=open(log, "a"), stderr=subprocess.STDOUT)
    with open(log, "a") as f:
        f.write(f"# {P.G.utc()} build rc {rc} in {time.time() - t0:.0f} s\n")
    return rc


def run():
    ctx = context()
    cfg, run_ = ctx["cfg"], ctx["cfg"]["run"]
    names, refs, files = MS.refs_and_names(ctx)
    gguf = cfg["model"]["gguf"]
    if os.path.realpath(gguf).startswith(("/data/ark", "/mnt/")):
        raise SystemExit(f"{gguf} is not on control-node's NVMe")
    cuts = P.py_plan(gguf, run_["world"])["cuts"]
    plans = [P.py_plan(gguf, 1, [cuts[r], cuts[r + 1]]) for r in range(run_["world"])]
    binary = P.G.assembled_binary(ctx)
    doc = {"binary": binary, "binary_sha256": P.G.sha256(binary), "cuts": cuts,
           "launches": [], "failures": [], "memory": []}
    raw = {}
    for k, launch in enumerate(run_["launches"]):
        ok, mem = MS.memory_ok(ctx, sum(p["ranks"][0]["device_bytes"] for p in plans), run_["world"])
        doc["memory"].append(mem)
        if not ok:
            doc["failures"].append(f"launch {k}: not enough free memory: {mem}")
            break
        log = os.path.join(RES, f"launch{k}.txt")
        open(log, "w").close()
        extra = {r: ["--rows", launch["rows"]] for r in range(run_["world"])}
        extra[0] += ["--run-specs", launch["specs"]]
        print(f"{P.G.utc()} launch {k}: {launch}", flush=True)
        ranks, _ = MS.launch(ctx, gguf, run_["world"], run_["cuts"], extra, files, run_["port"] + 10 * k, log,
                             run_["run_timeout_s"], run_["deadline_ms"], graph_modes="on")
        raw[k] = {r: {"rc": i["rc"], "events": [MS.clean(e) for e in i["events"]]} for r, i in ranks.items()}
        rec, why = judge_launch(k, launch, ranks, names, refs)
        doc["launches"].append(rec)
        doc["failures"] += why
        print(f"{P.G.utc()} launch {k} done: {len(why)} failure(s)", flush=True)
    with open(os.path.join(RES, "events.json"), "w") as f:
        json.dump({str(k): {str(r): v for r, v in d.items()} for k, d in raw.items()}, f)
    doc["disk"] = ctx["disk"]
    P.save(ctx, "results.json", doc)
    summarize(doc)
    return 0 if not doc["failures"] else 1


def judge():
    ctx = context()
    names, refs, _ = MS.refs_and_names(ctx)
    with open(os.path.join(RES, "events.json")) as f:
        raw = json.load(f)
    with open(os.path.join(RES, "results.json")) as f:
        doc = json.load(f)
    doc["launches"], doc["failures"] = [], []
    for k, launch in enumerate(ctx["cfg"]["run"]["launches"]):
        if str(k) not in raw:
            continue
        ranks = {int(r): v for r, v in raw[str(k)].items()}
        rec, why = judge_launch(k, launch, ranks, names, refs)
        doc["launches"].append(rec)
        doc["failures"] += why
    P.save(ctx, "results.json", doc)
    summarize(doc)
    return 0 if not doc["failures"] else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit({"build": build, "run": run, "judge": judge}.get(cmd, lambda: print(__doc__) or 2)())
