#!/usr/bin/env python3
"""ORNITH-CRUCIBLE runner (criteria: gate/ornith-crucible.toml, committed before any GPU run). Run on control-node, in the
export /data/projects/strixflerp-ocr (never the shared checkout):

    bash gate/ornith_crucible_build.sh                 # CPU: build/ocr/ornith-crucible-bench
    python3 gate/ornith_crucible.py experts            # K (pod-kbench k_run): item 1 identity + speed, 4 cases
    python3 gate/ornith_crucible.py head               # K (pod-kbench k_run): item 2 kernels, 2 cases
    python3 gate/ornith_crucible.py prompts            # CPU: ORNITH-O3's 8 prompts -> ids (this build's tokenizer)
    python3 gate/ornith_crucible.py probe              # control-node GPU (gate/ornith_crucible_run.sh via tools/gpu-queue.sh): arms A-E
    python3 gate/ornith_crucible.py spec               # control-node GPU (gate/ornith_crucible_run.sh spec): item 4, one world-2 launch
    python3 gate/ornith_crucible.py judge              # CPU: summary.json, summary.md

Evidence: gate/results/ornith-crucible/ (K results under kbench/, as the other kbench gates).
"""
import json
import os
import statistics
import subprocess
import sys
import time
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
CFG = os.path.join(ROOT, "gate", "ornith-crucible.toml")
RES = os.path.join(ROOT, "gate", "results", "ornith-crucible")
KRES = os.path.join(RES, "kbench")
BENCH = "build/ocr/ornith-crucible-bench"
GEN = os.path.join(ROOT, "build", "ocr-35b", "out", "flerp-qwen35moe-generate")
WORK = os.path.join(ROOT, "build", "ocr-probe")
CARD = os.environ.get("KB_CARD", "")
POLITE = ["ionice", "-c3", "nice", "-n", "19"]
ARMS = {  # arm: environment
    "A": {},
    "B": {"FLERP_MTP_COARSE": "q2", "FLERP_MTP_COARSE_CHECK": "1"},
    "C": {"FLERP_MTP_COARSE": "q2"},
    "D": {"FLERP_MTP_COARSE": "q4", "FLERP_MTP_COARSE_CHECK": "1"},
    "E": {"FLERP_MTP_COARSE": "q4"},
}


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def events(path):
    out = []
    if not os.path.isfile(path):
        return out
    for line in open(path, errors="replace"):
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def kbench(tag, cases, why):
    import kbench_backend as kb
    if not CARD:
        sys.exit("set KB_CARD to the experiment card id")
    kb.DEFAULT_CARD = CARD
    return kb.run_cases_k(ROOT, KRES, tag, BENCH, cases, why, timeout_case=1500)


def cmd_experts():
    orn = ["--experts", "--hidden", "4096", "--ff", "1024", "--used", "10", "--n-experts", "512"]
    p35 = ["--experts", "--hidden", "2048", "--ff", "512", "--used", "8", "--n-experts", "256"]
    cases = [("orn-q6k", orn + ["--down", "q6k"]), ("orn-q4k", orn + ["--down", "q4k"]),
             ("p35-q6k", p35 + ["--down", "q6k"]), ("p35-q4k", p35 + ["--down", "q4k"])]
    kbench("experts", cases, "ORNITH-CRUCIBLE item 1: moe_sg identity + speed, synthetic, Ornith and 35B dims")


def cmd_head():
    cases = [("head-orn", ["--head", "--vocab", "248320", "--hidden", "4096", "--bits", "2,4"]),
             ("head-p35", ["--head", "--vocab", "248320", "--hidden", "2048", "--bits", "2,4"])]
    kbench("head", cases, "ORNITH-CRUCIBLE item 2: draft_coarse kernels, synthetic Q6_K head, Ornith and 35B dims")


def o3():
    import ornith_o3_mtp as m
    m.GEN, m.WORK, m.RES = GEN, WORK, os.path.join(RES, "probe")
    os.makedirs(m.WORK, exist_ok=True)
    os.makedirs(m.RES, exist_ok=True)
    return m


def cmd_prompts():
    m = o3()
    return m.cmd_prompts(m.cfg(), None)


def cmd_probe(arms):
    m = o3()
    p = m.cfg()["proxy"]
    case = p["model"]
    if not case.startswith("/data/models/") or not os.path.isfile(case):
        sys.exit(f"model {case} is not a file on /data/models; refusing")
    rc = 0
    for arm in arms:
        env = m.env()
        for k in [k for k in env if k.startswith("FLERP_MTP_COARSE")]:
            del env[k]
        env.update(ARMS[arm])
        argv = [GEN, "--model", case, "--context", str(p["context"]), "--max-tokens", str(p["max_tokens"]),
                "--mtp-probe", str(p["depth"])] + m.id_args()
        out, err = os.path.join(RES, "probe", f"{arm}.jsonl"), os.path.join(RES, "probe", f"{arm}.err")
        t0 = time.time()
        with open(out, "w") as fo, open(err, "w") as fe:
            r = subprocess.run(POLITE + argv, stdout=fo, stderr=fe, timeout=3600, env=env)
        with open(os.path.join(RES, "probe", "arms.txt"), "a") as f:
            f.write(f"{utc()} arm {arm} env {ARMS[arm]} rc {r.returncode} {time.time() - t0:.0f} s load {open('/proc/loadavg').read().split()[0]}\n")
        rc = rc or r.returncode
    return rc


SPECS = "on:token:1:0,on:token:1:2,on:token:1:2p8,on:token:1:0,on:token:1:2,on:token:1:2p8"


def cmd_spec():
    """Item 4: prompt-lookup drafts mixed with MTP, --staged tcp world 2 on this GPU, one launch for every arm."""
    m = o3()
    c = tomllib.load(open(CFG, "rb"))
    px = m.cfg()["proxy"]
    model = px["model"]
    if not model.startswith("/data/models/") or not os.path.isfile(model):
        sys.exit(f"model {model} is not a file on /data/models; refusing")
    work = os.path.join(ROOT, "build", "ocr-spec")
    res = os.path.join(RES, "spec")
    os.makedirs(work, exist_ok=True)
    os.makedirs(res, exist_ok=True)
    prompts = [(f"edit{i}", "edit", t) for i, t in enumerate(c["item4"]["edit"])]
    prompts += [("code0", "ordinary", px["code"][0]), ("code1", "ordinary", px["code"][1]),
                ("prose0", "ordinary", px["prose"][0]), ("prose1", "ordinary", px["prose"][1])]
    argv = [GEN, "--model", model, "--tokenize"]
    for n, _, t in prompts:
        f = os.path.join(work, f"{n}.txt")
        open(f, "w").write(m.render(t))
        argv += ["--prompt-file", f]
    p = subprocess.run(argv, capture_output=True, text=True, env=m.env())
    toks = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{") and '"tokenized"' in l]   # as ornith_o3_mtp.py reads them
    if len(toks) != len(prompts):
        sys.exit(f"tokenize: {len(toks)} results for {len(prompts)} prompts: {p.stderr[-500:]}")
    files = []
    for (n, _, _), e in zip(prompts, toks):
        f = os.path.join(work, f"{n}.ids")
        open(f, "w").write(" ".join(str(i) for i in e["prompt_ids"]) + "\n")
        files += ["--ids-file", f]
    json.dump({"prompts": [[n, k, len(e["prompt_ids"])] for (n, k, _), e in zip(prompts, toks)], "specs": SPECS}, open(os.path.join(res, "prompts.json"), "w"), indent=1)
    base = [GEN, "--staged", "tcp", "--model", model, "--context", "4096", "--max-tokens", "256", "--world", "2", "--cuts", "even",
            "--port", "39870", "--epoch", "41", "--deadline-ms", "120000", "--graph-modes", "on", "--single-reader", "--pause-ms", "100",
            "--rows", "pinned"]
    ranks = {0: base + ["--rank", "0", "--reply", "token", "--run-specs", SPECS] + files,
             1: base + ["--rank", "1", "--draft-head", "--draft-costs", "75.2,19.68,4.83"]}
    procs = {}
    for r in (1, 0):
        procs[r] = subprocess.Popen(POLITE + ranks[r], stdout=open(os.path.join(res, f"r{r}.jsonl"), "w"),
                                    stderr=open(os.path.join(res, f"r{r}.err"), "w"), env=m.env())
        time.sleep(2)
    rc = 0
    for r in (0, 1):
        try:
            rc = rc or procs[r].wait(timeout=3000)
        except subprocess.TimeoutExpired:
            procs[r].kill()
            rc = 99
    with open(os.path.join(res, "launch.txt"), "a") as f:
        f.write(f"{utc()} rc {rc} load {open('/proc/loadavg').read().split()[0]}\n" + "\n".join("$ " + " ".join(a) for a in ranks.values()) + "\n")
    return rc


def judge_spec():
    res = os.path.join(RES, "spec")
    pj = os.path.join(res, "prompts.json")
    if not os.path.isfile(pj):
        return None
    meta = json.load(open(pj))
    st = [e for e in events(os.path.join(res, "r0.jsonl")) if e.get("event") == "staged"]
    dr = [e for e in events(os.path.join(res, "r1.jsonl")) if e.get("event") == "stage_timing" and "draft" in e]
    specs = meta["specs"].split(",")
    names = [p[0] for p in meta["prompts"]]
    kind = {p[0]: p[1] for p in meta["prompts"]}
    # rank 0 runs every prompt once per run spec, spec-major
    runs = {}
    for idx, e in enumerate(st):
        si, pi = idx // len(names), idx % len(names)
        runs.setdefault(names[pi], []).append({"spec": specs[si], "ids": e["ids"], "decode_s": e["decode_s"], "tok_s": len(e["ids"]) / e["decode_s"] if e["decode_s"] else None})
    dmap = {}
    dd = [e for e in dr if e["draft"].get("rounds", 0) > 0 or e["draft"].get("max_draft", 0) > 0]
    for k, e in enumerate(dd):   # drafting sessions only, in session order (specs with drafts, prompt-major inside)
        dmap[k] = e["draft"]
    l1 = len(st) == len(specs) * len(names)
    per = {}
    for n in names:
        r = runs.get(n, [])
        off = [x for x in r if x["spec"].endswith(":0")]
        l1 = l1 and len(off) == 2 and off[0]["ids"] == off[1]["ids"] and all(x["ids"] == off[0]["ids"] for x in r)
        per[n] = {s: median([x["tok_s"] for x in r if x["spec"] == s]) for s in dict.fromkeys(specs)}
    def cls(k):
        ns = [n for n in names if kind[n] == k]
        return {s: statistics.mean([per[n][s] for n in ns]) for s in dict.fromkeys(specs)} if ns else {}
    ed, od = cls("edit"), cls("ordinary")
    s2, s2p = "on:token:1:2", "on:token:1:2p8"
    # draft stats per (spec, prompt): drafting sessions in rank 0's order skipping the off runs
    sess = [(specs[idx // len(names)], names[idx % len(names)]) for idx in range(len(st)) if not specs[idx // len(names)].endswith(":0")]
    agg = {}
    for (sp, n), k in zip(sess, range(len(dd))):
        a = agg.setdefault((sp, kind[n]), {"frontiers": 0, "lookup_rounds": 0, "lookup_proposed": 0, "lookup_accepted": 0,
                                           "mtp_proposed": 0, "mtp_accepted": 0, "rounds": 0})
        for key in a:
            a[key] += dmap[k].get(key, 0)
    rates = {f"{sp}|{k}": {"lookup_hit": v["lookup_rounds"] / max(1, v["frontiers"]), "lookup_accept": v["lookup_accepted"] / max(1, v["lookup_proposed"]),
                           "mtp_accept": v["mtp_accepted"] / max(1, v["mtp_proposed"]), **v} for (sp, k), v in agg.items()}
    return {"L1": l1, "L3": bool(ed) and ed[s2p] >= 1.15 * ed[s2], "L4": bool(od) and od[s2p] >= 0.97 * od[s2],
            "tok_s_edit": ed, "tok_s_ordinary": od, "per_prompt": per, "rates": rates, "staged_events": len(st), "draft_sessions": len(dd)}


def median(v):
    return statistics.median(v) if v else None


def cmd_judge():
    c = tomllib.load(open(CFG, "rb"))
    doc = {"when": utc(), "criteria": "gate/ornith-crucible.toml", "item1": {}, "item2": {}}
    # ---- item 1
    ev = events(os.path.join(KRES, "experts.jsonl"))
    loads = json.load(open(os.path.join(KRES, "experts-load.json"))) if os.path.isfile(os.path.join(KRES, "experts-load.json")) else []
    tot = {e["case"]: e for e in ev if e.get("event") == "check_total"}
    spd = {e["case"]: e for e in ev if e.get("event") == "speed"}
    i1 = bool(tot) and len(tot) == 4 and all(sum(t["diff"].values()) == 0 and t["ctl_dirty"] == 0 and t["checks"] == 32 for t in tot.values())
    cases = {}
    for name, s in spd.items():
        best = min(s["new"], key=lambda n: n["us"])
        cases[name] = {"old_us": s["old"]["us"], "old_gbs": s["old"]["gbs"], "old_spread": [s["old"]["us_min"], s["old"]["us_max"]],
                       "best_variant": best["variant"], "new_us": best["us"], "new_gbs": best["gbs"], "new_spread": [best["us_min"], best["us_max"]],
                       "ratio": best["us"] / s["old"]["us"], "grid": best["grid"],
                       "variants": {str(n["variant"]): {"us": n["us"], "gbs": n["gbs"]} for n in s["new"]}}
    i2 = all(k in cases and cases[k]["ratio"] <= 0.90 for k in ("orn-q6k", "orn-q4k"))
    load_ok = all(l["load_start"] <= 6 and l["load_end"] <= 6 and l["rc"] == 0 for l in loads) and len(loads) == 4
    proj = None
    if "orn-q6k" in cases and "orn-q4k" in cases:
        s6 = cases["orn-q6k"]["old_us"] - cases["orn-q6k"]["new_us"]
        s4 = cases["orn-q4k"]["old_us"] - cases["orn-q4k"]["new_us"]
        tok = 83.0 - (30 * s6 + 30 * s4) / 1000.0
        tok_k3 = 77.5 - (30 * s6 + 30 * s4) / 1000.0
        proj = {"saved_us_q6k": s6, "saved_us_q4k": s4, "token_ms": tok, "tok_s": 1000.0 / tok, "gain_pct": 100.0 * (83.0 / tok - 1),
                "token_ms_on_kern3": tok_k3, "tok_s_on_kern3": 1000.0 / tok_k3, "label": "projected"}
    doc["item1"] = {"I1": i1, "I2": i2, "load_ok": load_ok, "cases": cases, "checks": tot, "projection": proj,
                    "confirmed": all(k in cases and cases[k]["ratio"] <= 0.80 for k in ("orn-q6k", "orn-q4k")),
                    "refuted": any(k in cases and cases[k]["ratio"] > 0.95 for k in ("orn-q6k", "orn-q4k"))}
    # ---- item 2 kernels
    hv = events(os.path.join(KRES, "head.jsonl"))
    hc = [e for e in hv if e.get("event") == "head_check"]
    hs = [e for e in hv if e.get("event") == "head_speed"]
    d1 = len(hc) == 4 and all(e["rescore_mismatch"] == 0 for e in hc)
    d2 = len(hc) == 4 and all(e["wrong_ids_without_fallback"] == 0 for e in hc)
    orn2 = next((e for e in hs if e["case"] == "head-orn" and e["bits"] == 2), None)
    d3 = orn2 is not None and orn2["new_us"] <= 0.5 * orn2["old_us"]
    # ---- item 2 proxy
    probe = {}
    for arm in ARMS:
        evs = events(os.path.join(RES, "probe", f"{arm}.jsonl"))
        pr = [e for e in evs if e.get("event") == "mtp_probe"]
        co = [e for e in events(os.path.join(RES, "probe", f"{arm}.err")) if e.get("event") == "mtp_coarse"]
        probe[arm] = {"prompts": len(pr), "mtp_ms_per_step": statistics.mean([e["mtp_ms_per_step"] for e in pr]) if pr else None,
                      "coarse": co[-1] if co else None, "_ev": pr}
    def same(a, b):
        pa, pb = probe[a]["_ev"], probe[b]["_ev"]
        return len(pa) == 8 and len(pb) == 8 and all(x.get("ids") == y.get("ids") and x.get("drafts") == y.get("drafts") for x, y in zip(pa, pb))
    e1 = all(probe[a]["coarse"] is not None and probe[a]["coarse"]["checked"] > 0 and probe[a]["coarse"]["wrong_ids"] == 0
             and probe[a]["coarse"]["rescore_mismatch"] == 0 for a in ("B", "D"))
    e2 = same("A", "C") and same("A", "E")
    e3 = all(probe[a]["coarse"] is not None and probe[a]["coarse"]["unsafe"] == 0 for a in ("B", "D"))
    ma, mc = probe["A"]["mtp_ms_per_step"], probe["C"]["mtp_ms_per_step"]
    e4 = ma is not None and mc is not None and mc <= 0.65 * ma
    rates = {}
    for a in ("B", "D"):
        co = probe[a]["coarse"]
        if co and co["checked"]:
            rates[a] = {"miss_top1": co["miss_top1"] / co["checked"], "miss_top2": co["miss_top2"] / co["checked"],
                        "fallback": co["fallbacks"] / max(1, co["steps"]), "max_z": co["max_z"], "err_rms": co["err_rms"],
                        "checked": co["checked"], "build_s": co["build_s"]}
    oproj = {}
    for e in hs:
        if e["case"] == "head-orn":
            fb = rates.get("B" if e["bits"] == 2 else "D", {}).get("fallback", 0.0)
            oproj[f"q{e['bits']}"] = {"old_head_us": e["old_us"], "new_head_us": e["new_us"], "fallback_rate_35b": fb,
                                      "ornith_step_ms": 4.83 - (e["old_us"] - e["new_us"]) / 1000.0 + fb * e["old_us"] / 1000.0,
                                      "label": "projected"}
    for a in probe:
        probe[a].pop("_ev")
    doc["item2"] = {"D1": d1, "D2": d2, "D3": d3, "E1": e1, "E2": e2, "E3": e3, "E4": e4, "head_checks": hc, "head_speed": hs,
                    "probe": probe, "rates": rates, "projection": oproj,
                    "draft_ms_35b": {"A": ma, "C": mc, "E": probe["E"]["mtp_ms_per_step"]}}
    doc["item4"] = judge_spec()
    json.dump(doc, open(os.path.join(RES, "summary.json"), "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if isinstance(vv, bool)} for k, v in doc.items() if isinstance(v, dict)}, indent=1))


def main():
    a = sys.argv[1:] or ["judge"]
    os.makedirs(RES, exist_ok=True)
    if a[0] == "experts":
        return cmd_experts()
    if a[0] == "head":
        return cmd_head()
    if a[0] == "prompts":
        return cmd_prompts()
    if a[0] == "probe":
        return cmd_probe(a[1:] or list(ARMS))
    if a[0] == "spec":
        return cmd_spec()
    if a[0] == "judge":
        return cmd_judge()
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
