#!/usr/bin/env python3
"""The four-node gate of recipe dsv41-flash.strixfold-tp4 (pod cards bb68604fa9, fe4d694019): the recipe's stage
against the fork's served TP4 build 72c0f083fca1, on F D C A, the eight frozen prompts, greedy. Written before any run; the
criterion is not loosened after one.

    python3 gate/dsv41_tp4.py --plan          # DRY: control-node only, reads files, writes the plan; no worker contact
    python3 gate/dsv41_tp4.py --run           # in a window only: the four held (pod-quad hold ff-bb68604fa9), boxes.lock

PASS = for each of the 8 prompts of tools/dsv41-hip/tp4fl/prompts8.json (sha256 d63bdf7d...f147), arm F (the recipe's
stage) gives the same as arm R (the served 72c0f083fca1):
  - rank 0's prompt-logits line {"prompt_logits", "sha256", "rows"} (LOGITS_SHA=1): sha256 and rows equal;
  - the generated token ids (return_token_ids: tensorfold.token_ids of the reply) equal;
  - the reply text byte-equal and HTTP 200.
And before either arm boots: the recipe's kit manifest sha256 equals the served kit's (446c43eb...08a0a8). Anything
else is FAIL, the first differing prompt named. Speed is recorded, never judged.

Arms (one boot each, R first, then F; the same conf bytes but DEPLOY_DIR):
  R  CONF=gate-R.conf: tp4-serve1-image-v7.conf with LOGITS_SHA=1 and DEPLOY_DIR ~/tfp/deploy/72c0f083fca1 (served)
  F  CONF=gate-F.conf: the same with DEPLOY_DIR ~/tfp/deploy/flerp-72c0f083fca1, the stage tools/flerp-stage.py makes
     from the recipe's build: native-hip/ (binaries and kit), tools/dsv41-hip/ (tool.strixfold_tp4.dsv41_hip and the
     parent modules' files) and pylib/, compared file by file with the served stage before it is used
A stored reply set of arm R (tp4-serve1-ref/tp4-serve1-image-v7.conf.72c0f083fca1.json) can stand in for booting R
(FLERP-TP4-GATE 7f6825bee0).
Both arms boot through /data/pod/plans/strixfold/tp4-serve1.sh (CONF=...; up / down), the TP4-SERVE1 launcher, so the
window's etiquette is its own: boxes.lock, a hold covering the four, pair keepers off, first_light.sh down then no
rank process on any of the four. No thermal push during a run (pod card 7037bfa9e7). Weights are the conf's rail
bundle and each rank's /cache rank dir, exactly as served: nothing is staged or copied by this gate.
Evidence: gate/results/dsv41-flash.strixfold-tp4/{plan-<UTC>.txt, run-<UTC>/}.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARD = "fe4d694019"
REF_SHORT = "72c0f083fca1"
REF_COMMIT = "72c0f083fca18f80fcf73b2358cf8235c3dd5363"
REF_KIT = "446c43ebb75f0386b6c5ba9ef8de14540c0fbb7f8f814c1faa6ac3888908a0a8"
FLERP_SHORT = "flerp-" + REF_SHORT
STAGE_ROOT = "/data/pod/plans/strixfold/deploy-stage"
SERVE = "/data/pod/plans/strixfold/tp4-serve1.sh"
BASE_CONF = "/data/pod/plans/strixfold/tp4-serve1-image-v7.conf"
PROMPTS = f"{STAGE_ROOT}/{REF_SHORT}/tools/dsv41-hip/tp4fl/prompts8.json"
PROMPTS_SHA = "d63bdf7d6d48016dadd00be043a6f26577e50b588aace67ef5ac21ddc5caf147"
KIT = "native-hip/share/tensorfold/hip/gfx1151/dsv41/manifest.json"
BUILD = os.path.join(ROOT, "build", "dsv41-flash.strixfold-tp4")
WORK = f"/data/pod/plans/{CARD}/gate"
NODES = [("F", "192.0.2.50"), ("D", "192.0.2.53"), ("C", "192.0.2.52"), ("A", "192.0.2.54")]
HOLD_OWNER = f"ff-{CARD}"
COLL_PROGRESS_FLOOR = 15000  # ms; conf_for refuses less (docs/DSV41-TP4.md lesson 1)
PL = re.compile(r'\{"prompt_logits": "[^"]*", "sha256": "[0-9a-f]{64}", "rows": \d+\}')


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def conf_for(arm, short):
    """The gate conf of an arm: the serving conf's bytes with LOGITS_SHA=1 and the arm's DEPLOY_DIR."""
    text = open(BASE_CONF).read()
    text, n1 = re.subn(r"(?m)^DEPLOY_DIR=.*$", f"DEPLOY_DIR='$HOME/tfp/deploy/{short}'   # gate {CARD} arm {arm}", text)
    text, n2 = re.subn(r"(?m)^LOGITS_SHA=.*$", f"LOGITS_SHA=1   # gate {CARD}: rank 0 logs each prompt's logits sha256", text)
    if n1 != 1 or n2 != 1:
        sys.exit(f"{BASE_CONF}: DEPLOY_DIR or LOGITS_SHA is not one line (found {n1}, {n2})")
    # The collective deadline: 2000 ms timed out every long prompt with kept states on the pair (SFF-PARK d74bd09137,
    # 2026-11-10); the TP4 confs carried 2000 until 2026-11-21 (agent raised all 14 to 15000, .bak-collprog-20261121).
    # Refuse a conf below the floor rather than gate a recipe on a setting known to crash long prompts.
    m = re.findall(r"(?m)^COLL_PROGRESS_MS=(\d+)", text)
    if len(m) != 1 or int(m[0]) < COLL_PROGRESS_FLOOR:
        sys.exit(f"{BASE_CONF}: COLL_PROGRESS_MS must be one line >= {COLL_PROGRESS_FLOOR} (found {m}); see LESSONS item 1")
    return text


def flerp_stage_plan(built):
    """The stage of arm F: name, and the command that makes and checks it (control-node, no worker contact)."""
    cmds = [f"python3 {ROOT}/tools/flerp-stage.py dsv41-flash.strixfold-tp4 {STAGE_ROOT}/{FLERP_SHORT} "
            f"--compare {STAGE_ROOT}/{REF_SHORT}      # STAGE-EQUAL or the differing files; exit 0 only when equal"]
    if os.path.isdir(f"{STAGE_ROOT}/{FLERP_SHORT}"):
        cmds = [f"python3 {ROOT}/tools/flerp-stage.py dsv41-flash.strixfold-tp4 {STAGE_ROOT}/{FLERP_SHORT} --compare-only "
                f"--compare {STAGE_ROOT}/{REF_SHORT}      # already made: compare again"]
    return FLERP_SHORT, cmds


def plan(args):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%MZ")
    out, bad = [], 0

    def say(s=""):
        out.append(s)

    say(f"# gate/dsv41_tp4.py --plan {now} (pod card {CARD}): DRY, control-node only, no worker contact")
    say(f"gate file sha256 {sha(os.path.abspath(__file__))}")
    built_json = os.path.join(BUILD, "built.json")
    km = os.path.join(BUILD, "tree", "zig-out", KIT)
    if os.path.isfile(km):
        k = sha(km)
        ok = k == REF_KIT
        bad |= not ok
        say(f"[{'ok' if ok else 'FAIL'}] recipe kit manifest {k} {'==' if ok else '!='} deployed {REF_KIT}")
    else:
        bad = 1
        say(f"[FAIL] no recipe build: {km} (build/dsv41-flash.strixfold-tp4/build.sh first)")
    say(f"[{'ok' if os.path.isfile(built_json) else 'FAIL'}] {built_json}")
    stage_km = f"{STAGE_ROOT}/{REF_SHORT}/{KIT}"
    if os.path.isfile(stage_km):
        dj = json.load(open(f"{STAGE_ROOT}/{REF_SHORT}/DEPLOY.json"))
        ok = sha(stage_km) == REF_KIT and dj.get("commit") == REF_COMMIT
        bad |= not ok
        say(f"[{'ok' if ok else 'FAIL'}] reference stage {STAGE_ROOT}/{REF_SHORT}: commit {dj.get('commit')}, kit {sha(stage_km)[:16]}...")
    else:
        bad = 1
        say(f"[FAIL] no reference stage {STAGE_ROOT}/{REF_SHORT}")
    if os.path.isfile(PROMPTS):
        ok = sha(PROMPTS) == PROMPTS_SHA
        bad |= not ok
        say(f"[{'ok' if ok else 'FAIL'}] prompts {PROMPTS} sha256 {sha(PROMPTS)[:16]}...")
    for p in (SERVE, BASE_CONF):
        say(f"[{'ok' if os.path.isfile(p) else 'FAIL'}] {p}" + (f" sha256 {sha(p)[:16]}..." if os.path.isfile(p) else ""))
        bad |= not os.path.isfile(p)
    short, cmds = flerp_stage_plan(built_json)
    say()
    say("## window prerequisites (checked at --run, not now)")
    say(f"- a hold over F D C A owned by {HOLD_OWNER}: OWNER={HOLD_OWNER} REASON='FLERP-DSV41-TP4 gate' pod-quad hold "
        + " ".join(ip for _, ip in NODES) + "; renew with `OWNER=" + HOLD_OWNER + " pod-quad renew <s>`, never pod-window open;"
        " the card in an active column (a hold whose card is in review is reaped by pod-quad sweep)")
    say("- /data/pod/plans/br-run/boxes.lock taken (flock -w 1500); no other TP4 instance answering http://192.0.2.50:18080/health")
    say("- no pod-thermal push while ranks run (7037bfa9e7); no NIC pause/PFC/MTU change under live RDMA")
    say("- the engram and rank dirs on each /cache are the served ones (PLACE placements tp4s1-engram*); nothing staged by this gate")
    say()
    say(f"## step 1: arm F's stage {STAGE_ROOT}/{short} (control-node)")
    for c in cmds:
        say("  " + c)
    for arm, s in (("R", REF_SHORT), ("F", short)):
        text = conf_for(arm, s)
        p = f"{WORK}/gate-{arm}.conf"
        say()
        say(f"## arm {arm}: CONF={p} ({os.path.basename(BASE_CONF)} with LOGITS_SHA=1, DEPLOY_DIR ~/tfp/deploy/{s}; sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]})")
        say(f"  CONF={p} {SERVE} check              # stage sums, deploy dirs, kit, hold, keepers off, first_light plan")
        say(f"  CONF={p} {SERVE} up                 # deploys ~/tfp/deploy/{s} from the stage where missing (sha256sum -c), first_light.sh up on D, /health, smoke")
        say(f"  (a failed up: {SERVE} down, then one retry; a second failure ends the window FAIL, nothing judged)")
        say(f"  python3 {ROOT}/gate/dsv41_tp4.py --ask http://192.0.2.50:18080 {WORK}/run-<UTC>/{arm}   # 8 prompts, return_token_ids")
        say(f"  rank 0's log on F after the mark: the 8 prompt_logits lines -> {WORK}/run-<UTC>/{arm}/shas.jsonl")
        say(f"  CONF={p} {SERVE} down               # first_light.sh down, then no /health and no rank process on F D C A")
    say()
    say(f"## judge: python3 {ROOT}/gate/dsv41_tp4.py --judge {WORK}/run-<UTC>/R {WORK}/run-<UTC>/F  -> verdict.json, PASS or FAIL")
    say(f"## after: release the hold; leave boxes down; the stage {STAGE_ROOT}/{short} and ~/tfp/deploy/{short} stay (the serving path under test)")
    say()
    say(f"plan {'READY' if not bad else 'NOT READY'} (rc {int(bool(bad))})")
    d = os.path.join(ROOT, "gate", "results", "dsv41-flash.strixfold-tp4")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"plan-{now}.txt")
    open(path, "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"-> {os.path.relpath(path, ROOT)}")
    return int(bool(bad))


def ask(url, outdir):
    """The 8 prompts, greedy, one after another, with their token ids (run by --run, or by hand in a window)."""
    import time
    import urllib.request
    os.makedirs(outdir, exist_ok=True)
    doc = json.load(open(PROMPTS))
    model = json.load(urllib.request.urlopen(url + "/v1/models", timeout=30))["data"][0]["id"]
    bad = 0
    with open(os.path.join(outdir, "replies.jsonl"), "w") as f:
        for p in doc["prompts"]:
            body = {"model": model, "messages": p["messages"], "temperature": doc["temperature"],
                    "max_tokens": doc["max_tokens"], "stream": False, "return_token_ids": True}
            req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            t0, rec = time.time(), {"id": p["id"]}
            try:
                r = json.load(urllib.request.urlopen(req, timeout=900))
                rec.update(status=200, usage=r.get("usage"), text=r["choices"][0]["message"].get("content"),
                           token_ids=(r.get("tensorfold") or r["choices"][0].get("tensorfold") or {}).get("token_ids"))
            except Exception as e:  # recorded; the rest still run
                rec.update(status=getattr(e, "code", 0), error=str(e)[:300])
                bad += 1
            rec["seconds"] = round(time.time() - t0, 3)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return 1 if bad else 0


def judge(rdir, fdir):
    def load(d):
        shas = [json.loads(m) for l in open(os.path.join(d, "shas.jsonl")) for m in PL.findall(l)]
        reps = [json.loads(l) for l in open(os.path.join(d, "replies.jsonl")) if l.strip()]
        return shas, reps
    (sr, rr), (sf, rf) = load(rdir), load(fdir)
    first, why = None, ""
    if len(sr) != 8 or len(sf) != 8 or len(rr) != 8 or len(rf) != 8:
        first, why = "-", f"counts: R {len(sr)} shas {len(rr)} replies, F {len(sf)} shas {len(rf)} replies (want 8 each)"
    else:
        for a, b, x, y in zip(sr, sf, rr, rf):
            if x.get("status") != 200 or y.get("status") != 200:
                first, why = x["id"], "not HTTP 200"
            elif (a["sha256"], a["rows"]) != (b["sha256"], b["rows"]):
                first, why = x["id"], "prompt logits sha256 or rows differ"
            elif not x.get("token_ids") or x.get("token_ids") != y.get("token_ids"):
                first, why = x["id"], "token ids differ (or missing)"
            elif x.get("text") != y.get("text"):
                first, why = x["id"], "reply text differs"
            if first:
                break
    v = {"verdict": "FAIL" if first else "PASS", "first_differing": first, "why": why, "R": rdir, "F": fdir,
         "criterion": "8 of 8: prompt-logits sha256 + rows, token ids, reply text equal; kit manifest equal before boot"}
    json.dump(v, open(os.path.join(os.path.dirname(rdir.rstrip("/")), "verdict.json"), "w"), indent=1)
    print(json.dumps(v, indent=1))
    return 0 if not first else 1


def run(args):
    sys.exit("--run is for a window: take the hold and boxes.lock, then follow the steps of the newest "
             "gate/results/dsv41-flash.strixfold-tp4/plan-*.txt (this card left the hardware gate for a window; "
             "nothing here has run on a worker)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--plan", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--ask", nargs=2, metavar=("URL", "OUTDIR"))
    g.add_argument("--judge", nargs=2, metavar=("R_DIR", "F_DIR"))
    a = ap.parse_args()
    if a.ask:
        return ask(*a.ask)
    if a.judge:
        return judge(*a.judge)
    if a.run:
        return run(a)
    return plan(a)


if __name__ == "__main__":
    sys.exit(main())
