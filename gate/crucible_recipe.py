#!/usr/bin/env python3
"""Recipe gate for qwen38-flash-next.crucible: Crucible d6af02c reassembled from the catalogue and built by its own
Makefile. Run on control-node (no GPU for check, origin, build):

    ./flerp gate recipe qwen38-flash-next.crucible --stages check,origin,build [--keep-going]
    ./flerp gate recipe qwen38-flash-next.crucible --plan        # the GPU stages, printed, not run

Stages (criteria in gate/qwen38-flash-next.crucible.toml), evidence under gate/results/<recipe>/:
  check   `flerp check --origin crucible`, and the whole catalogue's result.
  origin  catalogue files against `git archive <commit>` of the pinned clone, which must be at that commit.
  build   assembled tree vs the export file by file; then both trees build `build/engine/crucible.mono` with the same
          make command (plain hipcc), in parallel, under `ionice -c3 nice -n 19`, after waiting for the load average
          to fall below the configured bound. PASS: the two binaries are byte-identical (or, failing that, their
          device code sections are, with the difference recorded).
  tokens, speed: GPU stages, refused unless the weights pack has landed, and never run without --stages naming
          them. --plan prints exactly what they do. They reuse the strixite gates' lease hold (yields to
          `pod-gpu --wait` jobs between loads), server runner, captures and prefill bench. Queue them with
          tools/gpu-queue.sh.
"""
import argparse
import json
import os
import re
import shlex
import statistics
import shutil
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gate"))
from flerplib import origin as origin_mod  # noqa: E402
from flerplib.catalog import Catalog  # noqa: E402
import strixite_recipe as sr  # noqa: E402
from strixite_recipe import ids_hash, load_config, sh, sha256, tail, utc  # noqa: E402

STAGES = ("check", "origin", "build", "tokens", "speed")
GPU_STAGES = ("tokens", "speed")


def polite(cfg):
    pre = ["ionice", "-c3", "nice", "-n", "19"]
    adj = cfg["host"].get("oom_score_adj")
    if adj is not None and shutil.which("choom"):
        pre = ["choom", "-n", str(adj), "--"] + pre
    return pre


def wait_for_load(cfg, log):
    h = cfg["host"]
    t0 = time.time()
    while True:
        with open("/proc/loadavg") as f:
            load = float(f.read().split()[0])
        with open(log, "a") as f:
            f.write(f"{utc()} loadavg {load} (bound {h['max_loadavg']})\n")
        if load <= h["max_loadavg"]:
            return load
        if time.time() - t0 > h["load_wait_s"]:
            raise RuntimeError(f"load average stayed above {h['max_loadavg']} for {h['load_wait_s']} s")
        time.sleep(h["load_poll_s"])


def export_origin(ctx, log):
    h = ctx["cfg"]["host"]
    commit = ctx["cat"].origin_data(h["origin"])["commit"]
    export = os.path.join(ctx["scratch"], f"origin-{commit[:12]}")
    if not os.path.isfile(os.path.join(export, ".flerp-export")):
        ref = os.path.join(ROOT, h["reference_checkout"])
        sh(f"rm -rf {shlex.quote(export)} && mkdir -p {shlex.quote(export)} && "
           f"git -C {shlex.quote(ref)} archive {commit} | tar -x -C {shlex.quote(export)} && "
           f"touch {shlex.quote(export)}/.flerp-export", log)
    return export, commit


def rocm(ctx):
    name = ctx["cfg"]["host"]["toolchain"]
    # Crucible's author (2026-11-12, via owner): under ROCm 10.x the main MoE prefill kernel spills 772 B/lane to scratch
    # (about 4x slower, 32K prefill about -42%); 7.2.4 spills none. Refuse TheRock 10.x until a scratch check passes.
    if ctx["cat"].toolchains[name].get("kind") == "therock" and not ctx["cfg"]["host"].get("rocm10_scratch_checked"):
        sys.exit(f"crucible gate: toolchain {name} is ROCm 10.x, where Crucible's MoE prefill kernel spills to scratch; "
                 "use rocm-7.2.4, or set [host] rocm10_scratch_checked = true after an ISA scratch check passes")
    return ctx["cat"].toolchains[name]["rocm_root"]


def make_cmd(ctx):
    h, r = ctx["cfg"]["host"], rocm(ctx)
    return (polite(ctx["cfg"]) + ["env", "-u", "HIPCC_COMPILE_FLAGS_APPEND", "-u", "HIPCC_LINK_FLAGS_APPEND", "-u", "MAKEFLAGS",
            "make", "-j1", f"ROCM_PATH={r}", f"HIPCC={r}/bin/hipcc"] + h["make_options"] + [h["target"]])


def section(binary, name, out):
    rc = subprocess.call(["objcopy", "-O", "binary", f"--only-section={name}", binary, out])
    return sha256(out) if rc == 0 and os.path.isfile(out) else None


def sections(binary):
    p = subprocess.run(["readelf", "-SW", binary], capture_output=True, text=True)
    out = {}
    for line in p.stdout.splitlines():
        parts = line.replace("[ ", "[").split()
        if len(parts) > 5 and parts[0].startswith("[") and parts[1].startswith("."):
            out[parts[1]] = int(parts[5], 16)
    return out


# --- stages ---------------------------------------------------------------------------------------------------
def stage_check(ctx):
    log = os.path.join(ctx["res"], "check.txt")
    open(log, "w").close()
    rc, _ = sh([sys.executable, "flerp", "check", "--origin", ctx["cfg"]["host"]["origin"], "--warnings"], log)
    full = os.path.join(ctx["res"], "check-catalogue.txt")
    open(full, "w").close()
    rc_all, _ = sh([sys.executable, "flerp", "check"], full)
    return {"pass": rc == 0, "summary": tail(log, 1)[0], "catalogue_rc": rc_all, "catalogue_summary": tail(full, 1)[0],
            "evidence": "check.txt, check-catalogue.txt"}


def stage_origin(ctx):
    h = ctx["cfg"]["host"]
    log = os.path.join(ctx["res"], "diff-origin.txt")
    open(log, "w").close()
    export, commit = export_origin(ctx, log)
    head = subprocess.run(["git", "-C", os.path.join(ROOT, h["reference_checkout"]), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    res = origin_mod.diff_origin(ctx["cat"], h["origin"], export)
    res["problems"] = [p for p in res["problems"] if p[1] != ".flerp-export"]
    with open(log, "a") as f:
        for mid, path, why in res["problems"]:
            f.write(f"DIFF {mid} {path}: {why}\n")
        f.write(f"diff-origin origin {h['origin']} vs git archive {commit}: {res['modules']} module(s), {res['identical']}/"
                f"{res['files']} files byte-identical, {len(res['problems'])} problem(s), {res['unharvested']} unharvested\n")
    return {"pass": not res["problems"] and res["files"] > 0 and head == commit, "commit": commit, "head": head,
            "modules": res["modules"], "files": res["files"], "identical": res["identical"], "problems": len(res["problems"]),
            "unharvested": res["unharvested"], "evidence": "diff-origin.txt"}


def stage_build(ctx):
    cfg, name, h = ctx["cfg"], ctx["recipe"].name, ctx["cfg"]["host"]
    log = os.path.join(ctx["res"], "build.txt")
    open(log, "w").close()
    tree_root = os.path.join(ROOT, "build", name)
    out = {"evidence": "build.txt, build-assembled.txt, build-reference.txt, assembly.json"}
    rc, _ = sh([sys.executable, "flerp", "assemble", name], log)
    if rc != 0:
        return dict(out, rc=rc, pass_=False)
    with open(os.path.join(tree_root, "assembly.json")) as f:
        asm = json.load(f)
    shutil.copyfile(os.path.join(tree_root, "assembly.json"), os.path.join(ctx["res"], "assembly.json"))
    export, commit = export_origin(ctx, log)
    differ = [p for p, r in asm["files"].items() if not os.path.isfile(os.path.join(export, p))
              or sha256(os.path.join(export, p)) != r["sha256"]]
    out.update(files_in_tree=len(asm["files"]), files_differ=differ[:20], modules=len(asm["modules"]), commit=commit)
    tree = os.path.join(tree_root, "tree")
    shutil.rmtree(os.path.join(tree, "build"), ignore_errors=True)
    shutil.rmtree(os.path.join(export, "build"), ignore_errors=True)
    rc, _ = sh([sys.executable, os.path.join(ROOT, "flerplib", "assemble_zig.py"), "verify-tree", tree_root], log)
    if rc != 0 or differ:
        out["pass"] = False
        return out
    out["loadavg_at_start"] = wait_for_load(cfg, log)
    cmd = make_cmd(ctx)
    runs = {}

    def build(label, cwd):
        blog = os.path.join(ctx["res"], f"build-{label}.txt")
        open(blog, "w").close()
        runs[label] = sh(cmd, blog, cwd=cwd, timeout=h["build_timeout_s"])

    # The same command line in both directories: hipcc's compile unit id comes from the command's strings
    # (docs/HARVEST.md, "The kit hashes"), so equal relative paths give equal objects.
    ts = [threading.Thread(target=build, args=("assembled", tree)), threading.Thread(target=build, args=("reference", export))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    out["command"] = " ".join(cmd)
    for label, cwd in (("assembled", tree), ("reference", export)):
        b = os.path.join(cwd, h["target"])
        rc, secs = runs[label]
        out[label] = {"rc": rc, "seconds": secs}
        if rc == 0 and os.path.isfile(b):
            out[label].update(sha256=sha256(b), bytes=os.path.getsize(b),
                              hip_fatbin_sha256=section(b, ".hip_fatbin", os.path.join(ctx["scratch"], f"{label}.fatbin")),
                              sections=sections(b))
    a, r = out["assembled"], out["reference"]
    out["identical_binary"] = bool(a.get("sha256")) and a.get("sha256") == r.get("sha256")
    out["identical_device_code"] = bool(a.get("hip_fatbin_sha256")) and a.get("hip_fatbin_sha256") == r.get("hip_fatbin_sha256")
    if not out["identical_binary"] and a.get("sections") and r.get("sections"):
        out["section_size_differences"] = {s: [a["sections"].get(s), r["sections"].get(s)]
                                           for s in set(a["sections"]) | set(r["sections"])
                                           if a["sections"].get(s) != r["sections"].get(s)}
    out["pass"] = out["identical_binary"] or out["identical_device_code"]
    return out


def weights_dir(cfg):
    w = cfg["weights"]
    # The ark lays a single-revision row out under raw/ (2026-11-12 landing); that copy carries no revision name, so
    # it counts only when its quantization_config.json bits match the revision's leading "<bits>bpw".
    want_bits = w["revision"].split("bpw")[0]
    # NVMe copy first (model data never loads from a spinning disk; staged by an atomic rename, so it exists only whole).
    staged = [os.path.join(w["staged_dir"], w["revision"])] if w.get("staged_dir") else []
    for d in (*staged, os.path.join(w["ark_dir"], w["revision"]), w["ark_dir"], os.path.join(w["ark_dir"], "raw")):
        if not all(os.path.isfile(os.path.join(d, f)) for f in w["must_have"]):
            continue
        if d.endswith("/raw"):
            try:
                with open(os.path.join(d, "quantization_config.json")) as f:
                    if str(json.load(f).get("bits")) != want_bits:
                        continue
            except (OSError, ValueError):
                continue
        return d
    return None


def plan(ctx):
    cfg, name = ctx["cfg"], ctx["recipe"].name
    w = cfg["weights"]
    t, s = cfg["tokens"], cfg["speed"]
    asm_bin = f"build/{name}/tree/{cfg['host']['target']}"
    ref_bin = f"{cfg['host']['scratch']}/origin-<commit12>/{cfg['host']['target']}"
    print(f"""GPU stages of {name} (prepared, not run)

0. Weights landed? A folder holding {', '.join(w['must_have'])} at
   {w['ark_dir']}/{w['revision']}  or  {w['ark_dir']}
   check:  for d in {w['ark_dir']}/{w['revision']} {w['ark_dir']}; do ls "$d"/{{{','.join(w['must_have'])}}} && du -sh "$d"; done
   and every shard named by model.safetensors.index.json present with its size (no *.part, no ark ledger 'pending').
   Found now: {weights_dir(cfg) or 'NO (the pack is landing on ark-b; it is not on the pod yet)'}
   Also vision_k6.safetensors when --vision is used (engine/model/flashnext/weight_validation.h:44).

1. tokens (control-node GPU, lease {t['lease_resource']} as {t['lease_owner']}, ttl {t['lease_ttl_s']} s, only when idle):
   for each prompt of the strixite recipe gate (same token ids, read from gate/results/qwen38-flash-next.strixite):
     {ref_bin} gen <ckpt> --ids <ids> --n {t['n']} --spec {t['spec']}   x {t['repeats']}
     {asm_bin} gen <ckpt> --ids <ids> --n {t['n']} --spec {t['spec']}   x {t['repeats']}
     {asm_bin} gen <ckpt> --ids <ids> --n {t['n']} --spec {t['spec_check']}
   PASS: generated ids equal across builds and repeats; spec {t['spec_check']} equals spec {t['spec']} (drafts are lossless).
   Record: first position where Crucible's greedy ids leave strixite's on the same prompt (different quantization:
   EXL3 4.05 vs strixw; a fidelity record, not a criterion; scored under docs/FIDELITY.md's full-precision rule).

2. speed (same host, same power mode, one engine loaded at a time; power mode and sclk read before and after):
   prefill: cold prompts of {', '.join(map(str, s['prefill_lengths']))} tokens, {s['repeats']} repeats, median tok/s:
     Crucible: {asm_bin} gen <ckpt> --ids-file <ids> --n 1 --spec 0   (prefill rate from its own timing line)
     strixite: the strixite prefill gate (gate/strixite_prefill.py) on the same ids
   decode: {s['decode_tokens']} tokens on a code prompt and a prose prompt, --spec 0 and each engine's MTP default.
   Record per row: Crucible tok/s, strixite tok/s, ratio. Compare with Crucible's own claims (README: 1,831 tok/s cold
   32K, 1,782 at 130K; 71/58 tok/s decode with drafts, 34-39 without) and the fleet's (~1,220 prefill, 42-46 decode).

Run (after step 0 says yes):
   tools/pod-run.sh './flerp gate recipe {name} --stages tokens,speed'   (refuses without the weights)
""")


# --- GPU stages -----------------------------------------------------------------------------------------------
# Reused from the strixite gates: the lease and pod-job yielding (strixite_recipe.Hold), the strixite server runner
# and its capture reader (strixite_recipe.run_server/read_capture), the prefill bench command
# (strixite_prefill.bench_cmd). Host paths of the strixite side come from the strixite recipe gate's config.
STRIXITE_CFG = "gate/qwen38-flash-next.strixite.toml"
GEN_TIMEOUT_S = 1800
# Model data never loads from the archive (USB/HDD) mounts: a 101 GB load from ark-b took ~25 min a run.
COLD_STORAGE = ("/data/ark", "/mnt/ark*", "/mnt/nv*")
TIMING = re.compile(r"^prefill (\d+) tok in ([\d.]+) s \(([\d.]+) tok/s\), decode (\d+) tok in ([\d.]+) s \(([\d.]+) tok/s\)(.*)$",
                    re.M)
DECODE_PROMPTS = [
    ("code", "```python\n# lru_cache.py: a least-recently-used cache built from a dict and a doubly linked list,\n"
             "# with get, put, delete and resize, followed by unit tests.\n\n"),
    ("prose", "The lighthouse keepers of the northern coast kept a log of every storm, every ship and every "
              "failure of the lamp. What follows is the story of one winter on the island, as the last keeper "
              "told it many years later.\n\n"),
]


def gpu_cfg(ctx):
    """The strixite gate's host settings (paths, busy pattern) with this recipe's lease (owner, resource, ttl)."""
    scfg = load_config(os.path.join(ROOT, STRIXITE_CFG))
    t = ctx["cfg"]["tokens"]
    scfg["host"].update(lease_resource=t["lease_resource"], lease_owner=t["lease_owner"], lease_ttl_s=t["lease_ttl_s"])
    scfg["host"]["gpu_busy_pattern"] = scfg["host"]["gpu_busy_pattern"].rstrip(")") + "|crucible)"
    return scfg


def host_state():
    def r(p):
        try:
            with open(p) as f:
                return f.read().strip()
        except OSError:
            return None
    import glob
    dev = sorted(glob.glob("/sys/class/drm/card*/device/pp_dpm_sclk"))
    sclk = r(dev[0]) if dev else None
    tctl = None
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        if r(os.path.join(h, "name")) == "k10temp":
            tctl = int(r(os.path.join(h, "temp1_input")) or 0) / 1000
    return {"utc": utc(), "ec_power_mode": r("/sys/class/ec_su_axb35/apu/power_mode"),
            "dpm_level": r(os.path.dirname(dev[0]) + "/power_dpm_force_performance_level") if dev else None,
            "sclk_current": next((x.strip() for x in (sclk or "").splitlines() if x.strip().endswith("*")), None),
            "sclk_levels": sclk, "tctl_c": tctl, "loadavg": [round(x, 2) for x in os.getloadavg()]}


def crucible_bins(ctx):
    h = ctx["cfg"]["host"]
    export, _ = export_origin(ctx, os.path.join(ctx["res"], "export.txt"))
    return {"reference": os.path.join(export, h["target"]),
            "assembled": os.path.join(ROOT, "build", ctx["recipe"].name, "tree", h["target"])}


def crucible_gen(ctx, hold, label, binary, args, why):
    """One `crucible gen` process under the lease. Returns ids and the timing line's numbers."""
    d = os.path.join(ctx["scratch"], "gpu")
    os.makedirs(d, exist_ok=True)
    out_path, err_path = os.path.join(d, f"{label}.out"), os.path.join(d, f"{label}.err")
    cmd = [binary, "gen", ctx["ckpt"]] + [str(a) for a in args]
    if label.startswith("tokens-") and os.path.isfile(out_path) and TIMING.search(open(out_path).read()):
        # Resume: a finished token run of the same label (same binary, ids and options) is kept; its ids do not
        # depend on where the weights were read from. Speed runs are never reused.
        r = parse_gen(open(out_path).read(), label)
        r["reused"] = True
        return r
    hold.take(why, label)
    t0 = time.time()
    try:
        with open(out_path, "w") as o, open(err_path, "w") as e:
            e.write("$ " + " ".join(shlex.quote(c) if len(c) < 200 else c[:60] + "..." for c in cmd) + "\n")
            e.flush()
            rc = subprocess.call(cmd, cwd=ROOT, stdout=o, stderr=e, timeout=GEN_TIMEOUT_S)
    finally:
        hold.give()
    wall = round(time.time() - t0, 1)
    text = open(out_path).read()
    if rc != 0:
        raise RuntimeError(f"{label}: crucible gen exited {rc}; see {err_path}: {tail(err_path, 3)}")
    return parse_gen(text, label, wall)


def parse_gen(text, label, wall=None):
    ids = next((ln for ln in text.splitlines() if ln.startswith("ids:")), None)
    m = TIMING.search(text)
    if ids is None or m is None:
        raise RuntimeError(f"{label}: no ids or timing line in the output")
    gen = [int(x) for x in ids[4:].split()]
    return {"label": label, "wall_s": wall, "ids": gen, "ids_hash": ids_hash(gen), "prompt_n": int(m.group(1)),
            "prefill_s": float(m.group(2)), "prefill_tps": float(m.group(3)), "decode_n": int(m.group(4)),
            "decode_s": float(m.group(5)), "decode_tps": float(m.group(6)), "extra": m.group(7).strip(", ")}


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def strixite_prompts(ctx):
    """The strixite recipe gate's prompts as token ids, with its greedy (MTP off) ids, from the reference server's
    captures of that run; checked against the prompt and generated hashes recorded in its identity.json."""
    with open(os.path.join(ROOT, "gate", "results", "qwen38-flash-next.strixite", "identity.json")) as f:
        ident = json.load(f)
    srv = next(s for s in ident["servers"] if s["label"] == "reference-mtp-off")
    cap = os.path.join(ROOT, "build", "gate-scratch", "gpu", "reference-mtp-off", "capture")
    caps = sorted((sr.read_capture(os.path.join(cap, f)) for f in os.listdir(cap) if f.endswith(".cap")),
                  key=lambda c: c["id"])
    out = []
    for run, c in zip(srv["runs"], caps):
        if run["repeat"] != 0:
            continue
        if ids_hash(c["prompt"]) != run["prompt_hash"] or ids_hash(c["generated"]) != run["generated_hash"]:
            raise RuntimeError(f"capture {c['id']} does not match identity.json's {run['prompt']}")
        out.append({"name": run["prompt"], "prompt": c["prompt"], "strixite_greedy": c["generated"]})
    return out


def stage_tokens(ctx):
    t = ctx["cfg"]["tokens"]
    scfg = gpu_cfg(ctx)
    out = {"evidence": "tokens.json, lease-tokens.txt", "ckpt": ctx["ckpt"]}
    hold = sr.Hold(scfg, os.path.join(ctx["res"], "lease-tokens.txt"), out, 1800)
    bins = crucible_bins(ctx)
    out["binaries"] = {k: {"path": os.path.relpath(v, ROOT), "sha256": sha256(v)} for k, v in bins.items()}
    rows, ok = [], True
    for p in strixite_prompts(ctx):
        ids = ",".join(map(str, p["prompt"]))
        runs = []
        for label in ("reference", "assembled"):
            for rep in range(t["repeats"]):
                runs.append(crucible_gen(ctx, hold, f"tokens-{p['name']}-{label}-spec{t['spec']}-r{rep}", bins[label],
                                         ["--ids", ids, "--n", t["n"], "--spec", t["spec"]],
                                         f"crucible recipe gate tokens: {p['name']} {label} spec {t['spec']} r{rep}"))
        spec = crucible_gen(ctx, hold, f"tokens-{p['name']}-assembled-spec{t['spec_check']}", bins["assembled"],
                            ["--ids", ids, "--n", t["n"], "--spec", t["spec_check"]],
                            f"crucible recipe gate tokens: {p['name']} assembled spec {t['spec_check']}")
        base = runs[0]["ids"]
        same = all(r["ids"] == base for r in runs)
        lossless = spec["ids"] == base
        sg = p["strixite_greedy"]
        div = first_diff(base, sg[:len(base)]) if len(sg) >= len(base) else first_diff(base[:len(sg)], sg)
        rows.append({"prompt": p["name"], "prompt_n": len(p["prompt"]), "prompt_hash": ids_hash(p["prompt"]),
                     "ids_equal_across_builds_and_repeats": same, "spec_check_equals_spec0": lossless,
                     "crucible_ids": base, "crucible_ids_hash": runs[0]["ids_hash"],
                     "runs": [{k: r[k] for k in ("label", "ids_hash", "wall_s", "prefill_tps", "decode_tps", "extra")}
                              for r in runs + [spec]],
                     "strixite_greedy_n": len(sg), "strixite_greedy_head": sg[:len(base)],
                     "first_divergence_vs_strixite": div})
        ok = ok and same and lossless
    with open(os.path.join(ctx["res"], "tokens.json"), "w") as f:
        json.dump({"rows": rows, "holds": out["holds"], "yields": out["yields"]}, f, indent=1)
    out.update(ids_equal=all(r["ids_equal_across_builds_and_repeats"] for r in rows),
               spec_lossless=all(r["spec_check_equals_spec0"] for r in rows),
               rows=[{k: r[k] for k in ("prompt", "prompt_n", "ids_equal_across_builds_and_repeats",
                                        "spec_check_equals_spec0", "first_divergence_vs_strixite", "strixite_greedy_n")}
                     for r in rows])
    out["pass"] = ok
    return out


def bench_ids(seed, length):
    """strixflerp-prefill-bench's prompt_ids(seed, len): the same token ids for both engines."""
    m = (1 << 64) - 1
    s = (seed ^ (length * 0x100000001b3)) & m
    ids = []
    for _ in range(length):
        s = (s + 0x9e3779b97f4a7c15) & m
        z = s
        z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & m
        z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & m
        ids.append(256 + (z ^ (z >> 31)) % (151000 - 256))
    return ids


def med(xs):
    return round(statistics.median(xs), 2) if xs else None


def stage_speed(ctx):
    import strixite_prefill as sp
    s = ctx["cfg"]["speed"]
    scfg = gpu_cfg(ctx)
    out = {"evidence": "speed.json, lease-speed.txt", "ckpt": ctx["ckpt"], "host_before": host_state()}
    hold = sr.Hold(scfg, os.path.join(ctx["res"], "lease-speed.txt"), out, 1800)
    asm = crucible_bins(ctx)["assembled"]
    d = os.path.join(ctx["scratch"], "gpu")
    os.makedirs(d, exist_ok=True)
    rep = s["repeats"]
    # prefill, Crucible: one cold process per repeat
    pf = {}
    for L in s["prefill_lengths"]:
        f = os.path.join(d, f"prefill-{L}.ids")
        with open(f, "w") as fh:
            fh.write(",".join(map(str, bench_ids(1234, L))))
        ctxlen = L + 1024
        pf[L] = {"crucible": [crucible_gen(ctx, hold, f"speed-prefill-{L}-r{r}", asm,
                                           ["--ids-file", f, "--n", 1, "--spec", 0, "--ctx", ctxlen],
                                           f"crucible recipe gate speed: Crucible prefill {L} r{r}")["prefill_tps"]
                              for r in range(rep)]}
    out["host_after_crucible_prefill"] = host_state()
    # prefill, strixite: the prefill gate's bench (fleet chunk 8192, unmodified modules), one load, `rep` rounds
    bdir = os.path.join(d, "strixite-bench")
    shutil.rmtree(bdir, ignore_errors=True)
    os.makedirs(bdir)
    bcfg = {"host": scfg["host"], "bench": {"gen": 32, "seed": 1234, "capacity": max(s["prefill_lengths"]) + 8192,
                                            "logits_rows": 16}}
    bench = os.path.join(ROOT, "build", "qwen38-flash-next.strixite-pf", "ninja", "strixflerp-prefill-bench")
    cmd = sp.bench_cmd(bcfg, bench, bdir, [{"name": "base", "chunk": 8192}], s["prefill_lengths"], rep, False)
    hold.take("crucible recipe gate speed: strixite prefill bench", "strixite prefill bench")
    try:
        rc, secs = sr.sh(cmd, os.path.join(d, "strixite-bench.log"), timeout=5400)
    finally:
        hold.give()
    if rc != 0:
        raise RuntimeError(f"strixite prefill bench exited {rc}; see {d}/strixite-bench.log")
    brows = sp.read_jsonl(os.path.join(bdir, "results.jsonl"))
    for L in s["prefill_lengths"]:
        pf[L]["strixite"] = [r["prefill_tps"] for r in sorted(brows, key=lambda r: r["round"]) if r["len"] == L]
    out["host_after_strixite_prefill"] = host_state()
    # decode, strixite: the fleet's reference server, raw completions, MTP on (its default) then off
    reqs = [(n, "/v1/completions", {"prompt": p, "temperature": 0, "seed": 1234, "max_tokens": s["decode_tokens"],
                                    "stream": False}) for n, p in DECODE_PROMPTS]
    dcfg = dict(scfg, run=dict(scfg["run"], repeats=rep))
    sx = {}
    for mtp in (True, False):
        hold.take(f"crucible recipe gate speed: strixite server mtp {'on' if mtp else 'off'}", f"strixite mtp {mtp}")
        try:
            sx[mtp] = sr.run_server(f"speed-strixite-mtp-{'on' if mtp else 'off'}", scfg["host"]["reference_server"],
                                    mtp, dcfg, reqs, d)
        finally:
            hold.give()
    out["host_after_strixite_decode"] = host_state()
    # decode, Crucible: the same prompt ids (from strixite's captures), --spec 0 and the CLI default (no --spec)
    dec = {}
    for name, _ in DECODE_PROMPTS:
        prompt = next(r for r in sx[False]["runs"] if r["prompt"] == name)
        cap = os.path.join(d, "speed-strixite-mtp-off", "capture")
        caps = sorted((sr.read_capture(os.path.join(cap, f)) for f in os.listdir(cap) if f.endswith(".cap")),
                      key=lambda c: c["id"])
        ids = next(c["prompt"] for c in caps if ids_hash(c["prompt"]) == prompt["prompt_hash"])
        row = {"prompt_n": len(ids)}
        for mode, extra in (("spec0", ["--spec", 0]), ("default", [])):
            runs = [crucible_gen(ctx, hold, f"speed-decode-{name}-{mode}-r{r}", asm,
                                 ["--ids", ",".join(map(str, ids)), "--n", s["decode_tokens"]] + extra,
                                 f"crucible recipe gate speed: Crucible decode {name} {mode} r{r}") for r in range(rep)]
            row[f"crucible_{mode}"] = [r["decode_tps"] for r in runs]
            row[f"crucible_{mode}_extra"] = runs[0]["extra"]
        for mtp, mode in ((False, "spec0"), (True, "default")):
            rr = [r for r in sx[mtp]["runs"] if r["prompt"] == name]
            row[f"strixite_{mode}"] = [r["timings"].get("predicted_per_second") for r in rr]
            row[f"strixite_{mode}_n"] = [r["generated_n"] for r in rr]
        dec[name] = row
    out["host_after"] = host_state()
    table = []
    for L in s["prefill_lengths"]:
        c, x = med(pf[L]["crucible"]), med(pf[L]["strixite"])
        table.append({"row": f"prefill cold {L}", "crucible_tps": c, "strixite_tps": x,
                      "ratio": round(c / x, 3) if c and x else None, "crucible_all": pf[L]["crucible"],
                      "strixite_all": pf[L]["strixite"]})
    for name, row in dec.items():
        for mode, what in (("spec0", "no drafts"), ("default", "MTP default")):
            c, x = med(row[f"crucible_{mode}"]), med(row[f"strixite_{mode}"])
            table.append({"row": f"decode {name} {what}", "crucible_tps": c, "strixite_tps": x,
                          "ratio": round(c / x, 3) if c and x else None, "crucible_all": row[f"crucible_{mode}"],
                          "strixite_all": row[f"strixite_{mode}"], "strixite_generated_n": row[f"strixite_{mode}_n"],
                          "crucible_draft_stats": row[f"crucible_{mode}_extra"], "prompt_n": row["prompt_n"]})
    out["table"] = table
    with open(os.path.join(ctx["res"], "speed.json"), "w") as f:
        json.dump(dict(out, strixite_servers={str(k): {kk: v[kk] for kk in ("startup_s", "exit_code", "runs")}
                                              for k, v in sx.items()}), f, indent=1, default=str)
    out["pass"] = True   # recorded, not pass/fail
    return out


def stage_gpu(ctx, which):
    ctx["ckpt"] = weights_dir(ctx["cfg"])
    if ctx["ckpt"] is None:
        return {"pass": False, "error": "weights pack not on the pod yet; nothing run (see --plan)"}
    real = os.path.realpath(ctx["ckpt"])
    if any(real == p or real.startswith(p.rstrip("/") + "/") or (p.endswith("*") and real.startswith(p[:-1]))
           for p in COLD_STORAGE):
        return {"pass": False, "error": f"refused: weights at {real} are on cold storage (ark/archive mounts); "
                                        f"stage the pack on NVMe ([weights] staged_dir) first"}
    return {"tokens": stage_tokens, "speed": stage_speed}[which](ctx)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--recipe", default="qwen38-flash-next.crucible")
    ap.add_argument("--stages", default="check,origin,build")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--plan", action="store_true", help="print the GPU stages and the weights check; run nothing")
    a = ap.parse_args()
    cat = Catalog(ROOT)
    recipe = cat.recipe(a.recipe)
    if recipe is None:
        sys.exit(f"no recipe {a.recipe}")
    cfg = load_config(os.path.join(ROOT, recipe.gate["config"]))
    res = os.path.join(ROOT, "gate", "results", recipe.name)
    ctx = {"cat": cat, "recipe": recipe, "cfg": cfg, "res": res, "scratch": os.path.join(ROOT, cfg["host"]["scratch"])}
    if a.plan:
        plan(ctx)
        return 0
    os.makedirs(res, exist_ok=True)
    os.makedirs(ctx["scratch"], exist_ok=True)
    results, prior = {}, os.path.join(res, "summary.json")
    if os.path.isfile(prior):
        with open(prior) as f:
            results = json.load(f).get("stages", {})
    fns = {"check": stage_check, "origin": stage_origin, "build": stage_build}
    ok = True
    for s in a.stages.split(","):
        if s not in STAGES:
            sys.exit(f"unknown stage {s}")
        t0 = time.time()
        try:
            r = fns[s](ctx) if s in fns else stage_gpu(ctx, s)
        except Exception as e:  # noqa: BLE001 - recorded, not raised
            r = {"pass": False, "error": repr(e)}
        r["when_utc"], r["seconds"] = utc(t0), round(time.time() - t0, 1)
        results[s] = r
        print(f"{s}: {'PASS' if r.get('pass') else 'FAIL'} {json.dumps({k: v for k, v in r.items() if k not in ('sections',)})[:600]}")
        with open(prior, "w") as f:
            json.dump({"recipe": recipe.name, "written_utc": utc(), "stages": results}, f, indent=1)
            f.write("\n")
        ok = ok and r.get("pass")
        if not r.get("pass") and not a.keep_going:
            break
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
