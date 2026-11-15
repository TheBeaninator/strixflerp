#!/usr/bin/env python3
"""The pair measurement of executor.flerp.dsv41_zip2 (docs/PREFILL-DSV41.md section 9; pod card SPEED-ZIP2
18b8d5d277), run from control-node against workers F and A. It needs the owner's approval and a window from the
session that owns the pair. `--plan` reads only this host's files, prints what would run and changes nothing.

    python3 gate/prefill_dsv41_zip2.py --plan
    python3 gate/prefill_dsv41_zip2.py --minutes 20            # arms A,Z0,Z1,SA,SZ and Z2 if time is left

Arms, one boot each, drafts on, greedy requests only:
  A    the deployed build (f64b5ff2e28b) as served, prompt cache off: the 8,163-token prompt of the baseline run
       (gate/results/prefill-dsv41/pair-2026-11-11T1853Z/request.json) asked twice
  Z0   the build of recipe dsv41-flash.strixfold-tp2-zip2 with TF_DS_PREFILL_ZIP2 unset: must give A's sha and time
  Z1   the same build, TF_DS_PREFILL_ZIP2=1: the sha must be A's; rank 1's fill_chunk frame times, prefill seconds
  SA   the deployed build, prompt cache 4 GiB (no learned-state directory): a ~4,150-token prompt asked (a state is
       kept at 2,048), its state exported, asked again (resumes at 2,048); a second prompt asked and exported
  SZ   the new build, ZIP2=1, the same conf bytes: the first prompt asked (its 2,048 state now comes from the copy
       saved inside a zipped unit), exported and compared byte for byte with SA's; asked again (must resume at 2,048
       with SA's ids and sha); SA's export of the second prompt imported, then that prompt asked (must resume from
       the imported state with SA's ids and sha)
  Z2   the new build, TF_DS_PREFILL_ZIP2=2 (the last, partial chunk rides as the second of a unit): as Z1
Every arm records /health memory (rank 0), each rank's arena line, and GPU memory on both workers.

How the arms differ without the conf's bytes differing (a kept state's identity hashes the conf file and the
deployed commit, native.zig learnIdentity): every conf names DEPLOY_DIR=$HOME/tfp/strixflerp-zip2/deploy, a symlink
this runner points at the deployment (A, SA) or at the scratch copy of the new build (Z*, SZ); the setting is a
one-line file in the scratch copy that its serve_lib.sh (one line added to the copy) puts in the rank's environment.

What it does on the pod: one pod-quad hold as agent over F and A; the launcher's serving run taken down; a
scratch directory /home/pod/tfp/strixflerp-zip2 on each worker (never ~/tfp/deploy, never /cache/tfp-learned, the
serving conf is copied, not edited); runs as transient user units. At the end it stops what it started, removes the
scratch directories, RELEASES THE HOLD FIRST and then waits for the pod's keeper to bring serving back (on
2026-11-11 two `strixfold-pair.sh up` under the hold failed; after the release the keeper restored it in 20 s + load).
Evidence: gate/results/prefill-dsv41/zip2-<UTC>/.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shlex
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pair_nodes  # noqa: E402
import prefill_dsv41_pair as PP  # noqa: E402
from prefill_dsv41_pair import A, F, LAUNCHER, LIVE_CONF, ROOT, SSH, URL, WORDS, XDG, Run, now  # noqa: E402

S = "/home/pod/tfp/strixflerp-zip2"
DEPLOY = "/home/pod/tfp/deploy/f64b5ff2e28b"
TREE = os.path.join(ROOT, "build", "dsv41-flash.strixfold-tp2-zip2")
BASE_DIR = os.path.join(ROOT, "gate", "results", "prefill-dsv41", "pair-2026-11-11T1853Z")
BASE_SHA = "5396f3ae8bcb5189f15b4b1e0375530283621c05bcebff4ab30ab7b66e2f1bf9"
REF_KIT = "7693647972ab1b22dbff5d75f44ac080b573b9f0bf27d2bcc8c896b385f08455"
UNIT = "strixflerp-zip2-up"
CREATOR = "tfp/qual/bench/creator/prompts-v1.json"     # the origin's prompt set, if a worker has it (qual_bench.sh)
# serve_lib.sh of the scratch copy: the one line added after the line that passes TF_DS_PREFILL_SPLIT_ZIP
PATCH_AFTER = "TF_DS_PREFILL_SPLIT_ZIP="
PATCH_LINE = ('  z2=$(cat "$DEPLOY_DIR/PREFILL_ZIP2" 2>/dev/null || true); [ -n "$z2" ] && '
              'SERVE_ENV+=(TF_DS_PREFILL_ZIP2="$z2")  # strixflerp executor.flerp.dsv41_zip2')
ARMS = {  # tag: (kind, conf, deploy the symlink points at, TF_DS_PREFILL_ZIP2)
    "A": ("timing", "T.conf", DEPLOY, ""), "Z0": ("timing", "T.conf", S + "/zip2-deploy", ""),
    "Z1": ("timing", "T.conf", S + "/zip2-deploy", "1"), "SA": ("state", "C.conf", DEPLOY, ""),
    "SZ": ("state", "C.conf", S + "/zip2-deploy", "1"), "Z2": ("timing", "T.conf", S + "/zip2-deploy", "2"),
}
state_live = [None]  # the `first_light.sh down` command of a run that is up (the teardown runs it)
ARM_S = 200          # a boot (about 100 s of load), its requests, its stop and the idle check
TAIL_S = 90          # stopping, removing the scratch directories and releasing, inside the window


def conf_text(cache_gib):
    with open(LIVE_CONF) as f:
        text = f.read()
    over = {"BENCH": "0", "LOGITS_SHA": "1", "PROMPT_CACHE_GIB": str(cache_gib), "LEARN_DIR": "", "DRAFTS": "1",
            "DEPLOY_DIR": "$HOME/tfp/strixflerp-zip2/deploy", "RUN_ROOT": "$HOME/tfp/strixflerp-zip2/run"}
    return text + "\n# --- strixflerp zip2 window: appended, a later assignment wins ---\n" + "".join(f"{k}='{v}'\n" for k, v in over.items())


def records(n, seed):
    x, rows = seed, []
    for i in range(n):
        ws = []
        for _ in range(6):
            x = (1103515245 * x + 12345) % (1 << 31)
            ws.append(WORDS[(x >> 16) % len(WORDS)])
        rows.append(f"Record {i + 1}: the {ws[0]} {ws[1]} was moved from the {ws[2]} {ws[3]} to the {ws[4]} {ws[5]}.")
    return {"temperature": 0, "max_tokens": 4, "stream": False, "return_token_ids": True,
            "messages": [{"role": "system", "content": "You answer questions about the records you are given."},
                         {"role": "user", "content": "\n".join(rows) + "\n\nWhich record mentions a harbour first? Answer with its number."}]}


def http(path, data, timeout=300, raw=False):
    """POST; (status, body, headers). A JSON body is sent for a dict, bytes as they are."""
    body = data if isinstance(data, (bytes, bytearray)) else json.dumps(data).encode()
    ctype = "application/octet-stream" if isinstance(data, (bytes, bytearray)) else "application/json"
    req = urllib.request.Request(URL + path, data=body, headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = r.read()
            return r.status, (out if raw else json.loads(out)), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:300].decode(errors="replace"), {}


def ntok(body):
    return len(http("/tokenize", {"messages": body["messages"]}, 60)[1]["tokens"])


def sized(lo, hi, seed, run):
    """A records prompt whose token count is in [lo, hi] (a record is about 21 tokens)."""
    n = int((lo + hi) / 2 / 21.2)
    for _ in range(8):
        b = records(n, seed)
        t = ntok(b)
        if lo <= t <= hi:
            return b, t
        n += max(1, round(abs((lo + hi) / 2 - t) / 21.2)) * (1 if t < lo else -1)
    run.say(f"no prompt of {lo}..{hi} tokens found (last {t})")
    return None, t


def ask(run, row, name, body):
    t0 = time.time()
    st, reply, _ = http("/v1/chat/completions", body)
    wall = round(time.time() - t0, 3)
    if st != 200:
        raise RuntimeError(f"{name}: HTTP {st} {reply}")
    tf = reply.get("tensorfold", {}) if isinstance(reply.get("tensorfold"), dict) else {}
    u = reply.get("usage", {})
    r = {"name": name, "wall_s": wall, "prompt_tokens": u.get("prompt_tokens"),
         "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
         "prefill_seconds": tf.get("prefill_seconds"), "token_ids": tf.get("token_ids"), "token_sha": tf.get("token_sha"),
         "drafts": tf.get("drafts"), "sampling": tf.get("sampling")}
    row["requests"].append(r)
    run.say(f"arm {row['arm']} {name}: {r['prompt_tokens']} tokens, cached {r['cached_tokens']}, prefill {r['prefill_seconds']} s, ids {r['token_ids']}")
    return r


def memory(run):
    """GPU memory and MemAvailable on both workers (bytes), read from sysfs and /proc."""
    out = {}
    for ip, name in ((F, "rank0"), (A, "rank1")):
        _, text = run.on(ip, "cat /sys/class/drm/card*/device/mem_info_gtt_used | sort -n | tail -1; "
                             "cat /sys/class/drm/card*/device/mem_info_vram_used | sort -n | tail -1; "
                             "awk '/^MemAvailable:/ {print $2 * 1024}' /proc/meminfo", 30, True)
        nums = [int(x) for x in text.split() if x.isdigit()]
        out[name] = dict(zip(("gtt_used", "vram_used", "mem_available"), nums))
    return out


def boot(run, row, conf, target, zip2):
    """Point the symlink and the setting file, start first_light.sh as a transient user unit on A; True at health."""
    setting = f"printf %s {shlex.quote(zip2)} > {S}/zip2-deploy/PREFILL_ZIP2" if zip2 else f"rm -f {S}/zip2-deploy/PREFILL_ZIP2"
    for ip in (F, A):
        rc, _ = run.on(ip, f"ln -sfn {target} {S}/deploy && {setting} && readlink {S}/deploy")
        if rc != 0:
            row["error"] = f"could not set the arm up on {ip}"
            return False
    tools = f"{S}/deploy/tools/dsv41-hip"
    for attempt in (1, 2):
        rid = "zip2x-" + row["arm"] + "-" + datetime.datetime.now(datetime.timezone.utc).strftime("%H%M%S")
        row.update(run_id=rid, attempts=attempt, down=f"cd {tools} && RUN_ID={rid} TF_SERVE_CONF={S}/conf/{conf} ./first_light.sh down")
        state_live[0] = row["down"]
        run.on(A, XDG + f"systemd-run --user --collect --unit={UNIT} --property=KillMode=process bash -c "
               + shlex.quote(f"cd {tools} && RUN_ID={rid} TF_SERVE_CONF={S}/conf/{conf} ./first_light.sh up"))
        t0, ok = time.time(), False
        while time.time() - t0 < min(240, max(60, run.left() - TAIL_S)):
            h = run.health()
            active = run.on(A, XDG + f"systemctl --user is-active -q {UNIT}", 20, True)[0] == 0
            if h and not active:
                ok = True
                break
            if not h and not active and time.time() - t0 > 30:
                break
            time.sleep(4)
        row["startup_s"] = round(time.time() - t0, 1)
        if ok:
            return True
        run.say(f"arm {row['arm']}: start {attempt} did not reach health")
        _, t1 = run.on(A, f"tail -n 30 {S}/run/{rid}/rank1.log {S}/run/{rid}/*.txt 2>/dev/null", 30, True)
        _, t0_ = run.on(F, f"tail -n 40 {S}/run/{rid}/rank0.log 2>/dev/null", 30, True)
        run.save(f"{row['arm']}-failed-start-{attempt}.txt", t1 + "\n==== rank 0\n" + t0_ + "\n")
        run.on(A, XDG + f"systemctl --user stop {UNIT} 2>/dev/null; " + row["down"], 300)
        state_live[0] = None
        run.idle()
    row["error"] = "did not reach health in 2 starts"
    return False


def creator_prompt(run, state):
    """The origin's 8000-target prefill prompt (benchmark.py: 11,594 tokens in docs/RESULTS.md), if a worker has the
    creator's prompt set. Read-only."""
    if "creator" in state:
        return state["creator"]
    state["creator"] = None
    for ip in (F, A):
        rc, text = run.on(ip, f"cat ~/{CREATOR} 2>/dev/null", 60, True)
        if rc == 0 and text.startswith("{"):
            import random
            words = json.loads(text)["prefill_filler"]["words"]
            rnd = random.Random(8000)
            filler = " ".join(f"{rnd.choice(words)}{rnd.randint(0, 999)}" for _ in range(int(8000 * 0.55)))
            body = {"temperature": 0, "max_tokens": 1, "stream": False, "return_token_ids": True,
                    "messages": [{"role": "user", "content": "[bench v1 prefill 8000 r0] " + filler + "\n\nReply with the single word OK."}]}
            state["creator"] = body
            run.save("creator-request.json", json.dumps(body))
            run.say(f"the origin's 8000-target prompt: {ntok(body)} tokens (docs/RESULTS.md says 11,594)")
            break
    else:
        run.say(f"no ~/{CREATOR} on F or A: the origin's 11,594-token prompt is not run")
    return state["creator"]


def arm(run, tag, state):
    kind, conf, target, zip2 = ARMS[tag]
    row = {"arm": tag, "kind": kind, "conf": conf, "deploy": target, "zip2": zip2, "requests": [], "started": now()}
    if not boot(run, row, conf, target, zip2):
        return row
    rid = row["run_id"]
    try:
        row["health"] = run.health()
        row["memory_loaded"] = memory(run)
        if kind == "timing":
            p = state["P"]
            ask(run, row, "p8163-1", p)
            ask(run, row, "p8163-2", p)
            c = creator_prompt(run, state) if run.left() > 2 * ARM_S else None
            if c:
                ask(run, row, "creator8000-1", c)
                ask(run, row, "creator8000-2", c)
        else:
            if "Q1" not in state:                    # 4,096..4,223 tokens: two whole chunks before the replay cut's
                state["Q1"], n1 = sized(4100, 4150, 777, run)      # chunk end, and the last kept chunk end at 2,048
                state["Q2"], n2 = sized(4170, 4220, 4242, run)     # (two lengths: a sha line names its prompt by rows)
                run.save("state-requests.json", json.dumps({"Q1": state["Q1"], "Q2": state["Q2"], "tokens": [n1, n2]}))
            q1, q2 = state["Q1"], state["Q2"]
            if not q1 or not q2:
                raise RuntimeError("no state prompts")
            ask(run, row, "q1-full", q1)             # the first long prompt after the boot on both sides: same history
            st, blob, hd = http("/cache/export", q1, 120, raw=True)
            row["export_q1"] = {"status": st, "bytes": len(blob) if st == 200 else None, "at": hd.get("x-cache-tokens"),
                                "sha256": hashlib.sha256(blob).hexdigest() if st == 200 else None}
            if st == 200:
                state.setdefault("exports", {})[tag + "-q1"] = blob
            ask(run, row, "q1-resumed", q1)
            if tag == "SA":
                ask(run, row, "q2-full", q2)
                st, blob, hd = http("/cache/export", q2, 120, raw=True)
                row["export_q2"] = {"status": st, "bytes": len(blob) if st == 200 else None, "at": hd.get("x-cache-tokens"),
                                    "sha256": hashlib.sha256(blob).hexdigest() if st == 200 else None}
                if st == 200:
                    state["exports"]["SA-q2"] = blob
            elif "SA-q2" in state.get("exports", {}):
                st, reply, _ = http("/cache/import", state["exports"]["SA-q2"], 300)
                row["import_q2"] = {"status": st, "reply": reply}
                ask(run, row, "q2-imported", q2)
        row["health_after"] = run.health()
        row["memory_after"] = memory(run)
    except Exception as e:                           # a dead server is a result, not the end of the window
        row["error"] = repr(e)[:300]
        run.say(f"arm {tag}: {row['error']}")
    for ip, r in ((F, 0), (A, 1)):
        _, text = run.on(ip, f"cat {S}/run/{rid}/rank{r}.log 2>/dev/null", 60, True)
        run.save(f"{tag}-rank{r}.log", text)
        if r == 0:
            row["logits_sha"] = [[json.loads(x)["rows"], json.loads(x)["sha256"]] for x in text.splitlines() if x.startswith('{"prompt_logits"')]
            row["pass_lines"] = [x[x.find("prompt pass"):][:260] for x in text.splitlines() if "prompt pass:" in x]
        else:
            row["rank1_fill_chunk_ms"] = [int(m.group(1)) for m in re.finditer(r"spent (\d+) ms on its fill_chunk frame", text)]
            row["rank1_arena"] = next((json.loads(x) for x in text.splitlines() if x.startswith('{"rank": 1')), None)
    _, text = run.on(A, f"cat {S}/run/{rid}/first_light.json 2>/dev/null", 60, True)
    run.save(f"{tag}-first_light.json", text)
    try:
        fl = json.loads(text)
        row["resolved"] = [{"binary_sha256": x.get("binary_sha256"), "kit_manifest_sha256": x.get("kit_manifest_sha256"),
                            "deploy": x.get("deploy"), "config_sha256": x.get("config_sha256"),
                            "TF_DS_PREFILL_ZIP2": (x.get("env") or {}).get("TF_DS_PREFILL_ZIP2"),
                            "TF_DS_PREFILL_SPLIT": (x.get("env") or {}).get("TF_DS_PREFILL_SPLIT")}
                           for x in fl.values() if isinstance(x, dict) and "binary_sha256" in x]
    except ValueError:
        row["resolved"] = None
    run.on(A, XDG + f"systemctl --user stop {UNIT} 2>/dev/null; " + row["down"], 300)
    state_live[0] = None
    row["idle_after"] = run.idle()
    row["ended"] = now()
    return row


def stage(run, deployed_commit):
    """The scratch directory on each worker: the confs, and the new build as a deployment (this build's native-hip, a
    copy of the deployment's launch tools with one line added to serve_lib.sh, a DEPLOY.json naming the commit)."""
    meta = {"schema": "tf-port-deploy/1", "commit": deployed_commit, "clean": False, "created_utc": now(),
            "assembled_by": "strixflerp recipe dsv41-flash.strixfold-tp2-zip2: catalogue sources at the commit, with "
                            "model.zig and lanes.zig of executor.flerp.dsv41_zip2"}
    run.save("DEPLOY.json", json.dumps(meta, indent=1))
    for name, gib in (("T.conf", 0), ("C.conf", 4)):
        run.save(name, conf_text(gib))
    sed = f"sed -i {shlex.quote('/' + PATCH_AFTER + '/a\\' + chr(10) + PATCH_LINE)} {S}/zip2-deploy/tools/dsv41-hip/serve_lib.sh"
    for ip in (F, A):
        rc, _ = run.on(ip, f"mkdir -p {S}/conf {S}/run {S}/zip2-deploy && cp -a {DEPLOY}/tools {S}/zip2-deploy/ && "
                           f"chmod -R u+w {S}/zip2-deploy && {sed} && "
                           f"test \"$(grep -c TF_DS_PREFILL_ZIP2 {S}/zip2-deploy/tools/dsv41-hip/serve_lib.sh)\" = 1")
        if rc != 0:
            return f"could not stage the tools on {ip} (the serve_lib.sh line did not apply once)"
        for src, dst in ((os.path.join(TREE, "tree/zig-out/native-hip"), f"{S}/zip2-deploy/"),
                         (os.path.join(run.out, "DEPLOY.json"), f"{S}/zip2-deploy/DEPLOY.json"),
                         (os.path.join(run.out, "T.conf"), f"{S}/conf/T.conf"), (os.path.join(run.out, "C.conf"), f"{S}/conf/C.conf")):
            rc, _ = run.sh(["rsync", "-a", "-e", " ".join(SSH), src, f"pod@{ip}:{dst}"], 900)
            if rc != 0:
                return f"could not copy {os.path.basename(src)} to {ip}"
        rc, text = run.on(ip, f"sha256sum {S}/zip2-deploy/native-hip/share/tensorfold/hip/gfx1151/dsv41/manifest.json "
                              f"{DEPLOY}/native-hip/share/tensorfold/hip/gfx1151/dsv41/manifest.json | cut -c1-64 | sort -u | wc -l")
        if text.strip().splitlines()[-1:] != ["1"]:
            return f"the staged kit on {ip} is not the deployed kit"
    return None


def verdict(doc):
    """What the window showed, from the rows (nothing here is a pass unless the row says so)."""
    arms = {a["arm"]: a for a in doc["arms"]}
    v = {}

    def sha(tag, rows=8163):
        return sorted({s for r, s in arms.get(tag, {}).get("logits_sha", []) if r == rows})

    def second(tag):
        r = [x for x in arms.get(tag, {}).get("requests", []) if x["name"] == "p8163-2"]
        return r[0]["prefill_seconds"] if r else None

    v["sha_8163"] = {t: sha(t) for t in ("A", "Z0", "Z1", "Z2") if t in arms}
    v["A_is_the_baseline_sha"] = sha("A") == [BASE_SHA] if "A" in arms else None
    for t in ("Z0", "Z1", "Z2"):
        if t in arms:
            v[f"{t}_sha_equals_A"] = bool(sha(t)) and sha(t) == sha("A")
    v["prefill_s_second_request"] = {t: second(t) for t in ("A", "Z0", "Z1", "Z2") if t in arms}
    v["rank1_fill_chunk_ms"] = {t: arms[t].get("rank1_fill_chunk_ms") for t in ("A", "Z0", "Z1", "Z2") if t in arms}
    # the setting as each rank's resolved environment has it (first_light.json), and the binary each rank ran
    v["TF_DS_PREFILL_ZIP2_by_rank"] = {t: [x.get("TF_DS_PREFILL_ZIP2") for x in arms[t].get("resolved") or []] for t in arms}
    v["binary_sha256_by_rank"] = {t: [(x.get("binary_sha256") or "")[:16] for x in arms[t].get("resolved") or []] for t in arms}
    v["memory_active_rank0"] = {t: (arms[t].get("health_after") or arms[t].get("health") or {}).get("memory") for t in arms}
    v["rank1_arena_used"] = {t: (arms[t].get("rank1_arena") or {}).get("arena_used") for t in arms}
    if "SA" in arms and "SZ" in arms:
        ra = {x["name"]: x for x in arms["SA"]["requests"]}
        rz = {x["name"]: x for x in arms["SZ"]["requests"]}
        ea, ez = arms["SA"].get("export_q1") or {}, arms["SZ"].get("export_q1") or {}

        def by_rows(tag):
            out = {}
            for r, s in arms[tag].get("logits_sha", []):
                out.setdefault(r, set()).add(s)
            return out
        sa, sz = by_rows("SA"), by_rows("SZ")
        v["state"] = {
            "kept_at_2048_inside_a_zipped_unit_equals_the_reference_bytes": bool(ea.get("sha256")) and ea.get("sha256") == ez.get("sha256") and ea.get("at") == ez.get("at") == "2048",
            "resumed_under_zip2": {k: (rz.get("q1-resumed") or {}).get(k) for k in ("cached_tokens", "token_ids")},
            "resumed_ids_equal_reference": "q1-resumed" in rz and rz["q1-resumed"]["token_ids"] == (ra.get("q1-resumed") or {}).get("token_ids") and (rz["q1-resumed"]["cached_tokens"] or 0) > 0,
            "imported_reference_state_resumes": "q2-imported" in rz and (rz["q2-imported"]["cached_tokens"] or 0) > 0 and rz["q2-imported"]["token_ids"] == (ra.get("q2-full") or {}).get("token_ids"),
            "import": arms["SZ"].get("import_q2"),
            # every pass of a prompt (whole, resumed, resumed from an imported state) gives one sha, the reference's
            "prompt_logits_one_sha_per_prompt_and_equal": {str(r): len(sa[r]) == 1 and sa[r] == sz.get(r) for r in sa if r > 2048},
        }
    return v


def plan():
    built = os.path.join(TREE, "built.json")
    ok = os.path.isfile(built)
    kit = None
    if ok:
        with open(built) as f:
            b = json.load(f)
        kit = next((v["sha256"] for k, v in b.items() if k.endswith("/dsv41/manifest.json")), None)
    print(f"build: {os.path.relpath(TREE, ROOT)} {'built' if ok else 'NOT BUILT'}; kit manifest {kit} ({'the deployed kit' if kit == REF_KIT else 'NOT the deployed kit'})")
    print(f"baseline request: {os.path.relpath(BASE_DIR, ROOT)}/request.json {'present' if os.path.isfile(os.path.join(BASE_DIR, 'request.json')) else 'MISSING'}; expected sha256 {BASE_SHA}")
    print(f"hold: OWNER=agent pod-quad hold {F} {A}; then {LAUNCHER} down; scratch {S} on both workers")
    print("serve_lib.sh of the scratch copy gains, after the line with " + PATCH_AFTER + ":\n" + PATCH_LINE)
    for tag, (kind, conf, target, zip2) in ARMS.items():
        print(f"arm {tag:2s} {kind:6s} conf {conf} deploy -> {target}  TF_DS_PREFILL_ZIP2={zip2 or '(unset)'}")
    print(f"each arm about {ARM_S} s; at the end: stop, rm -rf {S}, pod-quad release FIRST, then wait for the keeper (up to 8 min)")
    return 0 if ok and kit == REF_KIT else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--arms", default="A,Z0,Z1,SA,SZ,Z2")
    ap.add_argument("--plan", action="store_true")
    pair_nodes.add_args(ap)
    a = ap.parse_args()
    pair = pair_nodes.from_args(a)
    pair_nodes.bind(pair, sys.modules[__name__], PP)
    print(pair_nodes.describe(pair), flush=True)
    if a.plan:
        sys.exit(plan())
    if plan() != 0:
        sys.exit("the zip2 recipe is not built with the deployed kit: nothing done")
    with open(os.path.join(BASE_DIR, "request.json")) as f:
        state = {"P": json.load(f)}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%MZ")
    out = os.path.join(ROOT, "gate", "results", "prefill-dsv41", "zip2-" + stamp)
    os.makedirs(out, exist_ok=True)
    run = Run(out, time.time() + a.minutes * 60)
    doc = {"started": now(), "release_by": datetime.datetime.fromtimestamp(run.deadline, datetime.timezone.utc).strftime("%H:%M:%SZ"),
           "pair": pair, "health_before": run.health(), "arms": [], "skipped": []}

    def flush():
        doc["verdict"] = verdict(doc)
        run.save("result.json", json.dumps(doc, indent=1))

    for ip in (F, A):
        if run.sh(["pod-quad", "held", ip], 30)[0] == 0:
            sys.exit(f"{ip} is held by someone: nothing done")
    rc, text = run.sh(f"OWNER=agent REASON={shlex.quote('strixflerp SPEED-ZIP2 18b8d5d277 window (agreed with the pair owner)')} "
                      f"TTL={int(a.minutes * 60 + 600)} pod-quad hold {F} {A}", 900)
    if rc != 0:
        sys.exit(f"pod-quad hold refused: {text[-300:]}")
    doc["hold_taken"] = now()
    try:
        run.sh(f"KANBAN_ACTOR=agent {LAUNCHER} down", 400)
        if not run.idle(240):
            raise RuntimeError("the pair did not go idle under the hold")
        rc, text = run.on(A, f"python3 -c 'import json; print(json.load(open(\"{DEPLOY}/DEPLOY.json\"))[\"commit\"])'", 30)
        commit = text.strip().splitlines()[-1] if rc == 0 and text.strip() else ""
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise RuntimeError("the deployment's commit could not be read")
        err = stage(run, commit)
        if err:
            raise RuntimeError(err)
        for tag in [t for t in a.arms.split(",") if t in ARMS]:
            if run.left() < ARM_S + TAIL_S:
                doc["skipped"].append(f"{tag}: {int(run.left())} s to the release time")
                run.say(doc["skipped"][-1])
                continue
            if tag == "SZ" and "SA" not in [x["arm"] for x in doc["arms"]]:
                doc["skipped"].append("SZ: no SA arm to compare with")
                continue
            doc["arms"].append(arm(run, tag, state))
            flush()
    except Exception as e:
        doc["error"] = repr(e)[:400]
        run.say("error: " + doc["error"])
    finally:
        # stop anything of this window, remove the scratch, RELEASE FIRST, then let the pod's keeper restore serving
        run.on(A, XDG + f"systemctl --user stop {UNIT} 2>/dev/null; " + (state_live[0] or "true"), 300)
        for ip in (F, A):
            run.on(ip, XDG + "for u in $(systemctl --user list-units 'tfp-*zip2x*' --state=active --no-legend --plain | cut -d' ' -f1); "
                   "do systemctl --user stop $u; echo stopped $u; done; true", 120)
        run.idle(120)
        for ip in (F, A):
            run.on(ip, f"rm -rf {S}")
        rc, text = run.sh("OWNER=agent pod-quad release", 300)
        doc["hold_released"], doc["release_rc"] = now(), rc
        time.sleep(2)
        doc["held_after"] = {ip: run.sh(["pod-quad", "held", ip], 30)[0] == 0 for ip in (F, A)}
        doc["left_on_workers"] = {ip: run.on(ip, XDG + f"ls -d {S} 2>&1 | head -2; systemctl --user list-units 'strixflerp-zip2*' --no-legend --plain | head", 60)[1] for ip in (F, A)}
        restored, t0 = None, time.time()
        run.say("hold released; waiting for the pod's keeper to bring serving back (no start from here)")
        while time.time() - t0 < 480:
            h = run.health()
            if h and h.get("model") == "dsv41" and h.get("warming") is False:
                restored = h
                break
            time.sleep(5)
        doc["health_after"], doc["restored_at"], doc["restore_wait_s"] = restored, now(), round(time.time() - t0)
        if not restored:
            run.say("SERVING DID NOT COME BACK in 8 minutes after the release: tell the pair's owner (nothing was started from here)")
        flush()
    print(json.dumps({k: doc.get(k) for k in ("hold_taken", "hold_released", "health_after", "skipped", "error", "verdict")}, indent=1))


if __name__ == "__main__":
    main()
