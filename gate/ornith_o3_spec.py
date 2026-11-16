#!/usr/bin/env python3
"""ORNITH-O3 step two (criteria: gate/ornith-o3-spec.toml, committed before any GPU run): speculative decoding with the
file's own MTP head in the pipeline-parallel qwen35moe executor (survey option B), proved on the 35B as four TCP rank
processes on control-node's one GPU. Run ON control-node, in a clean checkout of the card branch (tools/card-branch.sh <card-id> on the laptop,
tools/pod-push.sh --card <card-id>, then git clone /data/projects/strixflerp.git -b card/<card-id> /data/projects/<card-id>;
never the shared /data/projects/strixflerp; docs/POD-GIT.md):

    python3 gate/ornith_o3_spec.py build     # CPU only: flerp assemble + build.sh under nice/ionice
    python3 gate/ornith_o3_spec.py smoke     # GPU (through tools/gpu-queue.sh): one launch, 3 prompts, 32 tokens
    python3 gate/ornith_o3_spec.py run       # GPU (through tools/gpu-queue.sh): the two launches of [proxy] launches
    python3 gate/ornith_o3_spec.py judge     # CPU: criteria, recorded numbers, the Ornith projection -> summary.md

The GPU steps go through gate/ornith_o3_spec_run.sh. No worker, no hold. Evidence: gate/results/ornith-o3-spec/.
"""
import json
import os
import statistics
import subprocess
import sys
import time
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gate"))
import moe_placed as P  # noqa: E402
import moe_stage as MS  # noqa: E402

CONFIG = "gate/ornith-o3-spec.toml"
RES = os.path.join(ROOT, "gate", "results", "ornith-o3-spec")
RECIPE = "qwen36-35b-a3b.flerp-moe"
O3 = os.path.join(ROOT, "gate", "results", "ornith-o3-mtp")
P1 = "gate/results/qwen36-35b-a3b.flerp-moe-placed/equal.json"
# host settings (how the job runs, not what it is judged by)
HOST = {"origin": "gufo", "scratch": "build/gate-scratch-o3-spec", "cpus": "0-3", "polite": ["ionice", "-c3", "nice", "-n", "19"],
        "reader_flags": ["--single-reader", "--pause-ms", "100"], "load_wait_above": 10.0, "load_wait_s": 1800, "arena_gib": 2,
        "margin_gib": 24}
PORT, EPOCH, DEADLINE_MS, TIMEOUT_S = 39810, 21, 120000, 5400


def context(res=RES):
    import gufo_recipe as G               # noqa: E402
    import strixite_recipe as S           # noqa: E402
    from flerplib.catalog import Catalog
    P.G, P.S = G, S
    with open(os.path.join(ROOT, CONFIG), "rb") as f:
        crit = tomllib.load(f)
    px = crit["proxy"]
    cfg = {"recipe": RECIPE, "host": dict(HOST), "crit": crit,
           "model": {"gguf": px["model"]},
           "run": {"context": px["context"], "max_tokens": px["max_tokens"], "epoch": EPOCH, "deadline_ms": DEADLINE_MS}}
    cat = Catalog(ROOT)
    os.makedirs(res, exist_ok=True)
    ctx = {"cat": cat, "recipe": cat.recipe(RECIPE), "cfg": cfg, "res": res, "config": CONFIG,
           "scratch": os.path.join(ROOT, HOST["scratch"]), "disk": []}
    os.makedirs(ctx["scratch"], exist_ok=True)
    return ctx


def prompt_set(ctx):
    """[(name, class, ids)]: P5's four, then O3's eight."""
    p5 = P.load_json("gate/results/qwen36-35b-a3b.flerp-moe/prompt-ids.json")
    o3 = P.load_json("gate/results/ornith-o3-mtp/prompts.json")
    out = [(n, "p5", p5[n]) for n in p5]
    out += [(n, o3["classes"][n], o3["ids"][n]) for n in o3["names"]]
    return out


def spec_list(text):
    out = []
    for item in text.split(","):
        f = item.split(":")
        d = f[3] if len(f) > 3 else "0"
        out.append({"graph": f[0], "reply": f[1], "subchunks": int(f[2]), "drafts": int(d.lstrip("v")), "value": d.startswith("v"),
                    "label": d})
    return out


# --- the O3 references (pure) ------------------------------------------------------------------------------------
def o3_reference():
    """O3's own ids and drafts per prompt name (on.jsonl: `generated` and `mtp_probe` events, prompt order)."""
    names = P.load_json("gate/results/ornith-o3-mtp/prompts.json")["names"]
    gen, probe = [], []
    with open(os.path.join(O3, "on.jsonl")) as f:
        for line in f:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("event") == "generated":
                gen.append(e)
            elif e.get("event") == "mtp_probe":
                probe.append(e)
    return {n: {"ids": gen[i]["ids"], "probe_ids": probe[i]["ids"], "drafts": probe[i]["drafts"]} for i, n in enumerate(names)}


def schedule_rate(ref, k):
    import ornith_o3_mtp as O3M  # noqa: E402
    drafted = accepted = 0
    for r in ref.values():
        s = O3M.schedule(r["probe_ids"], r["drafts"], k)
        drafted += s["drafted"]
        accepted += s["accepted"]
    return accepted / drafted if drafted else None


# --- judgement (pure) ----------------------------------------------------------------------------------------------
def judge_launch(k, specs_text, rank_out, prompts, max_tokens):
    """One launch: criteria 1, 2, 4 rows. Returns (record, [failures])."""
    why = []
    for r, info in sorted(rank_out.items()):
        if info["rc"] != 0:
            why.append(f"launch {k}: rank {r} exited {info['rc']}")
    ev0 = rank_out.get(0, {}).get("events", [])
    for e in ev0:
        if e.get("event") in ("staged_failed", "error", "relay_failed"):
            why.append(f"launch {k}: rank 0 {e.get('event')}: {e.get('error', e.get('reason', ''))}")
    for r, info in rank_out.items():
        for e in info["events"]:
            if e.get("event") in ("error", "relay_failed"):
                why.append(f"launch {k}: rank {r} {e.get('event')}: {e.get('error', e.get('reason', ''))}")
    staged = [e for e in ev0 if e.get("event") == "staged"]
    specs = spec_list(specs_text)
    names = [p[0] for p in prompts]
    if len(staged) != len(specs) * len(names):
        why.append(f"launch {k}: {len(staged)} staged results, want {len(specs) * len(names)}")
    timing = {}
    for r, info in rank_out.items():
        for e in info["events"]:
            if e.get("event") == "stage_timing":
                timing.setdefault(e["run"], {})[r] = e
    runs = []
    for i, e in enumerate(staged):
        spec = specs[i // len(names)] if i // len(names) < len(specs) else {}
        name = names[e["prompt"]]
        if e.get("drafts") != spec.get("drafts") or e.get("value") != spec.get("value"):
            why.append(f"launch {k} run {i}: drafts {e.get('drafts')}/{e.get('value')} is not the spec {spec}")
        calls = e["calls"]
        st = {0: {"timing": e["stages"][0]}}
        st.update(timing.get(i, {}))
        run = {"launch": k, "spec": spec.get("label"), "drafts": e.get("drafts", 0), "value": e.get("value", False), "prompt": name,
               "class": prompts[e["prompt"]][1], "ids": e["ids"], "decode_wall_ms": calls["decode_wall_ms"],
               "decode_calls": calls["decode_calls"], "frames": calls.get("frames"), "prefill_s": e["prefill_s"],
               "stage_ms_per_step": {r: t["timing"]["decode_ms_per_step"] for r, t in st.items()},
               "ring": {r: t["timing"].get("ring") for r, t in st.items()}}
        if e.get("drafts", 0) > 0:
            run["rounds"] = e["rounds"]
            run["draft"] = (st.get(3) or {}).get("draft")
            # 4. round accounting
            ids, rid, K = e["ids"], e["round_ids"], e["drafts"]
            cat = [x for r_ in rid for x in r_]
            tag = f"launch {k} {spec.get('label')} {name}"
            if not e.get("accounting_ok", False):
                why.append(f"{tag}: rank 0 accounting: {e.get('accounting_error')}")
            if cat[:len(ids)] != ids:
                why.append(f"{tag}: the rounds' ids do not concatenate to the run's ids")
            have = len(rid[0]) if rid else 0
            for j, (kp, n) in enumerate(e["rounds"][1:], start=1):
                room = max_tokens - min(have, max_tokens)
                if not (0 <= n <= kp <= K) or len(rid[j]) != n + 1:
                    why.append(f"{tag}: round {j}: proposed {kp}, kept {n}, {len(rid[j])} ids (K {K})")
                    break
                if not e.get("value") and kp != min(K, room):
                    why.append(f"{tag}: round {j}: proposed {kp}, want min({K}, {room})")
                    break
                have += n + 1
        runs.append(run)
    return {"launch": k, "specs": specs_text, "runs": runs, "rc": {r: i["rc"] for r, i in rank_out.items()}}, why


def judge_all(doc, prompts, max_tokens, anchors=True):
    """Criteria 2, 3, 5 across launches; fills doc["criteria"], appends failures. anchors=False (Ornith, [ornith]):
    criterion 2 only (the 35B anchors and the 35B probe's schedule do not apply)."""
    why = []
    names = [p[0] for p in prompts]
    off_ref = {}
    for L in doc["launches"]:
        by = {}
        for r in L["runs"]:
            by.setdefault(r["prompt"], []).append(r)
        for name in names:
            rs = by.get(name, [])
            off = [r for r in rs if r["drafts"] == 0]
            if not off:
                why.append(f"launch {L['launch']} {name}: no drafts-off run")
                continue
            ref = off[0]["ids"]
            for r in rs:
                if r["ids"] != ref:
                    why.append(f"2: launch {L['launch']} {name} {r['spec']}: ids differ from drafts off at {MS.first_diff(r['ids'], ref)}")
            if name in off_ref and off_ref[name] != ref:
                why.append(f"2: {name}: drafts-off ids differ across launches")
            off_ref.setdefault(name, ref)
    if not anchors:
        doc["criteria"] = {}
        return why
    # 3. anchors
    p1 = P.load_json(P1)["traced"]
    o3 = o3_reference()
    anchors = {}
    for name in names:
        if name not in off_ref:
            continue
        if name in p1:
            want = p1[name]["ids"][:128]
            ok = off_ref[name][:128] == want
        else:
            want = o3[name]["ids"]
            ok = off_ref[name] == want
        anchors[name] = ok
        if not ok:
            why.append(f"3: {name}: drafts-off ids differ from the anchor at {MS.first_diff(off_ref[name], want)}")
    # 5. acceptance against the probe's schedule on O3's prompts
    acc = {}
    for kk in (1, 2, 3):
        sim = schedule_rate(o3, kk)
        for L in doc["launches"]:
            rs = [r for r in L["runs"] if r["drafts"] == kk and not r["value"] and r["prompt"] in o3]
            prop = sum(kp for r in rs for kp, n in r["rounds"][1:])
            kept = sum(n for r in rs for kp, n in r["rounds"][1:])
            rate = kept / prop if prop else None
            acc.setdefault(str(kk), []).append({"launch": L["launch"], "rate": rate, "proposed": prop, "kept": kept, "probe_schedule": sim})
            if rate is None or sim is None or abs(rate - sim) > 0.03:
                why.append(f"5: launch {L['launch']} k={kk}: accepted/proposed {rate} against the probe's schedule {sim} (bound 0.03)")
    doc["criteria"] = {"anchors": anchors, "acceptance": acc}
    return why


# --- recorded numbers and the projection (pure) ------------------------------------------------------------------
def recorded(doc):
    runs = [r for L in doc["launches"] for r in L["runs"]]
    out = {"tok_s": {}, "acceptance": {}, "draft": {}, "ring": {}}
    for label in sorted({r["spec"] for r in runs}, key=lambda x: (x.startswith("v"), x)):
        rs = [r for r in runs if r["spec"] == label]
        by_cls = {}
        for r in rs:
            t = len(r["ids"]) / (r["decode_wall_ms"] / 1000.0) if r["decode_wall_ms"] else 0
            by_cls.setdefault(r["class"], []).append(t)
            by_cls.setdefault("all", []).append(t)
        out["tok_s"][label] = {c: round(statistics.mean(v), 2) for c, v in by_cls.items()}
        if label != "0":
            for cls in ("code", "prose", "p5", "all"):
                sel = [r for r in rs if cls == "all" or r["class"] == cls]
                prop = sum(kp for r in sel for kp, n in r["rounds"][1:])
                kept = sum(n for r in sel for kp, n in r["rounds"][1:])
                rounds = sum(len(r["rounds"]) - 1 for r in sel)
                toks = sum(n + 1 for r in sel for kp, n in r["rounds"][1:])
                out["acceptance"].setdefault(label, {})[cls] = {"rounds": rounds, "proposed": prop, "kept": kept,
                                                               "rate": round(kept / prop, 4) if prop else None,
                                                               "tokens_per_round": round(toks / rounds, 4) if rounds else None}
            d = [r["draft"] for r in rs if r.get("draft")]
            if d:
                agg = {k: sum(x[k] for x in d) for k in ("rounds", "frames", "skipped", "draft_steps", "draft_ms", "refill_steps", "refill_ms",
                                                         "prompt_steps", "prompt_ms", "top2_ms")}
                agg["draft_ms_per_step"] = agg["draft_ms"] / max(agg["draft_steps"], 1)
                agg["refill_ms_per_step"] = agg["refill_ms"] / max(agg["refill_steps"], 1)
                agg["top2_ms_per_round"] = agg["top2_ms"] / max(agg["rounds"], 1)
                hist = {}
                for x in d:
                    for i, c in enumerate(x["depth_histogram"]):
                        hist[i] = hist.get(i, 0) + c
                agg["depth_histogram"] = hist
                out["draft"][label] = agg
            rings = [r["ring"] for r in rs]
            out["ring"][label] = {str(k): v for k, v in (rings[0] or {}).items()} if rings else {}
    return out


def project(proj, rec):
    """[projection] of the criteria: Ornith tok/s per class at k = 1..3, two bounds on how the hop cost is paid."""
    S = proj["ornith_stage_ms"]
    R = proj["ornith_wall_ms"] - sum(S)
    h = R / len(S)
    D = proj["ornith_mtp_ms"]
    d35 = rec["draft"].get("2") or next(iter(rec["draft"].values()), None)
    ratio = d35["refill_ms_per_step"] / d35["draft_ms_per_step"] if d35 and d35["draft_ms_per_step"] else 0.5
    Dr = D * ratio
    top2 = (rec["draft"].get("v3") or {}).get("top2_ms_per_round", 0.0)

    def round_ms(k, n, occupy):
        # frames 0..k leave rank 0 back to back; stage s takes frame j at max(arrival, free); the last stage runs frames
        # 0..n (refill after each kept one), skips the rest, then drafts k MTP steps; the reply is one hop
        free = [0.0] * len(S)
        last_done = 0.0
        arrive = [0.0] * (k + 1)
        for s in range(len(S)):
            for j in range(k + 1):
                if s == len(S) - 1 and j > n:
                    continue
                start = max(arrive[j], free[s])
                cost = S[s] + (h if occupy else 0.0)
                if s == len(S) - 1 and j < n:
                    cost += Dr
                free[s] = start + cost
                arrive[j] = free[s] + (0.0 if occupy else h)
                if s == len(S) - 1:
                    last_done = free[s]
        return last_done + k * D + (0.0 if occupy else 0.0) + h

    out = {}
    for cls, chain in proj["ornith_chain"].items():
        out[cls] = {}
        for k in (1, 2, 3):
            p_ge = [1.0] + chain[:k]                    # P(n >= i), i = 0..k
            pn = [p_ge[i] - (p_ge[i + 1] if i + 1 <= k else 0.0) for i in range(k + 1)]
            res = {}
            for occupy in (False, True):
                et = sum(pn[n] * (n + 1) for n in range(k + 1))
                ems = sum(pn[n] * round_ms(k, n, occupy) for n in range(k + 1))
                res["hop_occupies_stage" if occupy else "hop_latency_only"] = round(1000.0 * et / ems, 2)
            out[cls][k] = res
    base = {"off": round(1000.0 / proj["ornith_wall_ms"], 2)}
    survey = {}
    for cls, chain in proj["ornith_chain"].items():
        survey[cls] = {}
        for k in (1, 2, 3):
            e = 1 + sum(chain[:k])
            survey[cls][k] = round(1000.0 * e / (sum(S) + max(S) * k + R + k * D), 2)
    return {"inputs": {"stage_ms": S, "R_ms": round(R, 2), "hop_ms": round(h, 2), "mtp_ms": D, "refill_ms": round(Dr, 3),
                       "refill_over_draft_35b": round(ratio, 4), "top2_ms_per_round_35b": round(top2, 4)},
            "tok_s": out, "drafts_off_tok_s": base, "survey_T_B_tok_s": survey}


def summarize(doc, title="ORNITH-O3 step two: MTP speculative decoding, 35B proxy (world 4, TCP 127.0.0.1, one GPU)"):
    L = [f"# {title}", "",
         f"Written {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}. Criteria: `{CONFIG}` (committed before the first GPU run). "
         f"Binary sha256 {doc.get('binary_sha256', '?')}.", "",
         f"**{'PASS' if not doc['failures'] else 'FAIL'}**: {len(doc['failures'])} failure(s).", ""]
    L += ["## Identity: ids with drafts against drafts off (same launch), 256 tokens", "", "| launch | spec | prompts equal |", "|---|---|---|"]
    for Lr in doc["launches"]:
        by = {}
        for r in Lr["runs"]:
            by.setdefault(r["prompt"], []).append(r)
        for label in dict.fromkeys(r["spec"] for r in Lr["runs"]):
            n = eq = 0
            for name, rs in by.items():
                off = [x for x in rs if x["drafts"] == 0]
                for x in rs:
                    if x["spec"] == label:
                        n += 1
                        eq += bool(off) and x["ids"] == off[0]["ids"]
            L.append(f"| {Lr['launch']} | {label} | {eq}/{n} |")
    c = doc.get("criteria", {})
    L += ["", f"Anchors (drafts off = P1's first 128 / O3's ids): {sum(c.get('anchors', {}).values())}/{len(c.get('anchors', {}))}", ""]
    L += ["## Acceptance against the O3 probe's schedule (O3's 8 prompts)", "", "| k | launch | accepted / proposed | probe schedule |", "|---|---|---|---|"]
    for k, rows in (c.get("acceptance") or {}).items():
        for r in rows:
            L.append(f"| {k} | {r['launch']} | {r['rate'] if r['rate'] is None else round(r['rate'], 4)} ({r['kept']}/{r['proposed']}) | "
                     f"{r['probe_schedule'] if r['probe_schedule'] is None else round(r['probe_schedule'], 4)} |")
    rec = doc.get("recorded", {})
    one_gpu = "35B proxy" in title
    L += ["", "## Recorded: tok/s, ids / rank 0's decode wall" + (" (ONE GPU: the four ranks time-share it, so drafts cannot overlap here)" if one_gpu else ""), "",
          "| spec | p5 | code | prose | all |", "|---|---|---|---|---|"]
    for label, t in rec.get("tok_s", {}).items():
        L.append(f"| {label} | {t.get('p5', '-')} | {t.get('code', '-')} | {t.get('prose', '-')} | {t.get('all', '-')} |")
    L += ["", "## Recorded: acceptance and the last stage's drafting", "", "| spec | class | rounds | kept/proposed | tokens a round |", "|---|---|---|---|---|"]
    for label, by in rec.get("acceptance", {}).items():
        for cls, a in by.items():
            L.append(f"| {label} | {cls} | {a['rounds']} | {a['rate']} | {a['tokens_per_round']} |")
    L += ["", "| spec | frames | skipped | MTP draft ms a step | refill ms a step | top-2 ms a round | depths |", "|---|---|---|---|---|---|---|"]
    for label, d in rec.get("draft", {}).items():
        L.append(f"| {label} | {d['frames']} | {d['skipped']} | {d['draft_ms_per_step']:.3f} | {d['refill_ms_per_step']:.3f} | "
                 f"{d['top2_ms_per_round']:.3f} | {d['depth_histogram']} |")
    if rec.get("ring"):
        L += ["", f"Rings (k = 2, launch 0): {json.dumps(rec['ring'].get('2', {}))}"]
    pj = doc.get("projection")
    if pj:
        L += ["", "## Ornith projection (computed, not measured)", "", f"Inputs: {json.dumps(pj['inputs'])}", "",
              f"Drafts off: {pj['drafts_off_tok_s']['off']} tok/s.", "", "| class | k | tok/s (hop = latency) | tok/s (hop occupies the stage) | survey T_B |",
              "|---|---|---|---|---|"]
        for cls, by in pj["tok_s"].items():
            for k, v in by.items():
                L.append(f"| {cls} | {k} | {v['hop_latency_only']} | {v['hop_occupies_stage']} | {pj['survey_T_B_tok_s'][cls][k]} |")
    if doc["failures"]:
        L += ["", "## Failures (first 40)", ""] + [f"- {w}" for w in doc["failures"][:40]]
    with open(os.path.join(doc.get("_res", RES), "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")


# --- commands ------------------------------------------------------------------------------------------------------
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


def launches(ctx, launch_specs, prompts, res, max_tokens=None):
    cfg = ctx["cfg"]
    if max_tokens is not None:
        cfg["run"]["max_tokens"] = max_tokens
    gguf = cfg["model"]["gguf"]
    if os.path.realpath(gguf).startswith(("/data/ark", "/mnt/")) or not os.path.realpath(gguf).startswith("/data/models/"):
        raise SystemExit(f"{gguf} is not on control-node's NVMe (/data/models)")
    px = cfg["crit"]["proxy"]
    cuts = P.py_plan(gguf, px["world"])["cuts"]
    plans = [P.py_plan(gguf, 1, [cuts[r], cuts[r + 1]]) for r in range(px["world"])]
    files = [P.write_ids(ctx, n, ids) for n, _, ids in prompts]
    binary = P.G.assembled_binary(ctx)
    doc = {"binary": binary, "binary_sha256": P.G.sha256(binary), "cuts": cuts, "launches": [], "failures": [], "memory": [],
           "prompts": [[n, c, len(i)] for n, c, i in prompts], "max_tokens": cfg["run"]["max_tokens"], "_res": res}
    raw = {}
    for k, specs in enumerate(launch_specs):
        ok, mem = MS.memory_ok(ctx, sum(p["ranks"][0]["device_bytes"] for p in plans), px["world"])
        doc["memory"].append(mem)
        if not ok:
            doc["failures"].append(f"launch {k}: not enough free memory: {mem}")
            break
        log = os.path.join(res, f"launch{k}.txt")
        open(log, "w").close()
        extra = {r: ["--rows", "pinned"] for r in range(px["world"])}
        extra[0] += ["--reply", "token", "--run-specs", specs]
        extra[px["world"] - 1] += ["--draft-head", "--draft-costs", px["draft_costs"]]
        print(f"{P.G.utc()} launch {k}: {specs}", flush=True)
        ranks, _ = MS.launch(ctx, gguf, px["world"], px["cuts"], extra, files, PORT + 10 * k, log, TIMEOUT_S, DEADLINE_MS, graph_modes="on")
        raw[k] = {r: {"rc": i["rc"], "events": [MS.clean(e) for e in i["events"]]} for r, i in ranks.items()}
        rec, why = judge_launch(k, specs, raw[k], prompts, cfg["run"]["max_tokens"])
        doc["launches"].append(rec)
        doc["failures"] += why
        print(f"{P.G.utc()} launch {k} done: {len(why)} failure(s)", flush=True)
    with open(os.path.join(res, "events.json"), "w") as f:
        json.dump({str(k): {str(r): v for r, v in d.items()} for k, d in raw.items()}, f)
    return doc


def run():
    ctx = context()
    px = ctx["cfg"]["crit"]["proxy"]
    prompts = prompt_set(ctx)
    doc = launches(ctx, px["launches"], prompts, RES)
    doc["failures"] += judge_all(doc, prompts, px["max_tokens"])
    doc["recorded"] = recorded(doc)
    doc["projection"] = project(ctx["cfg"]["crit"]["projection"], doc["recorded"])
    doc["disk"] = ctx["disk"]
    P.save(ctx, "results.json", doc)
    summarize(doc)
    return 0 if not doc["failures"] else 1


def smoke():
    """Not judged: one launch on three prompts, 32 tokens, to find crashes before the real run (evidence smoke/)."""
    res = os.path.join(RES, "smoke")
    ctx = context(res)
    prompts = [p for p in prompt_set(ctx) if p[0] in ("bridge", "code0", "prose0")]
    doc = launches(ctx, ["on:token:1:0,on:token:1:2,on:token:1:1,on:token:1:3,on:token:1:v3"], prompts, res, max_tokens=32)
    for L in doc["launches"]:
        by = {}
        for r in L["runs"]:
            by.setdefault(r["prompt"], []).append(r)
        for name, rs in by.items():
            off = [x for x in rs if x["drafts"] == 0]
            for x in rs:
                if off and x["ids"] != off[0]["ids"]:
                    doc["failures"].append(f"smoke {name} {x['spec']}: ids differ at {MS.first_diff(x['ids'], off[0]['ids'])}")
    doc["recorded"] = recorded(doc)
    P.save(ctx, "results.json", doc)
    summarize(doc)
    print(json.dumps({"failures": doc["failures"][:20], "tok_s": doc["recorded"]["tok_s"], "acceptance": doc["recorded"]["acceptance"]}, indent=1))
    return 0 if not doc["failures"] else 1


def judge():
    ctx = context()
    px = ctx["cfg"]["crit"]["proxy"]
    prompts = prompt_set(ctx)
    with open(os.path.join(RES, "events.json")) as f:
        raw = json.load(f)
    with open(os.path.join(RES, "results.json")) as f:
        doc = json.load(f)
    doc["launches"], doc["failures"], doc["_res"] = [], [], RES
    for k, specs in enumerate(px["launches"]):
        if str(k) not in raw:
            continue
        rank_out = {int(r): v for r, v in raw[str(k)].items()}
        rec, why = judge_launch(k, specs, rank_out, prompts, px["max_tokens"])
        doc["launches"].append(rec)
        doc["failures"] += why
    doc["failures"] += judge_all(doc, prompts, px["max_tokens"])
    doc["recorded"] = recorded(doc)
    doc["projection"] = project(ctx["cfg"]["crit"]["projection"], doc["recorded"])
    P.save(ctx, "results.json", doc)
    summarize(doc)
    return 0 if not doc["failures"] else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit({"build": build, "run": run, "smoke": smoke, "judge": judge}.get(cmd, lambda: print(__doc__) or 2)())
