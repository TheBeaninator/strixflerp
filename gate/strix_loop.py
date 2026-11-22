#!/usr/bin/env python3
"""STRIX-LOOP: one strixite kernel change, gated end to end on bench box K.

    gate/strix_loop.py <target> [--run NAME] [--stages plan,build,micro,model,isa,report] [--order ABAB|AB]
                       [--base qwen38-flash-next.strixite-pf] [--prompts 128,512,...] [--rounds 2] [--card ID]
                       [--weights PATH --ngram PATH] [--variant-weights PATH --variant-args "..."]

Runs ON control-node (CPU: assembly, builds, isa-scan) and drives bench box K through pod-kbench k_run (the only way
onto K; it queues behind the bench:K lease, never kills a holder). Nothing here touches workers A B C D F or the
control-node GPU. tools/strix-loop.sh is the entry from a laptop card worktree.

<target> is one of
    null                 no change: the base recipe against itself (the pipeline's zero; kernel.strixite.moe_router is
                         the probe for the microbench and the isa scan). Must come out identical, ratio about 1.00.
    recipe:<name>        a recipe in recipes/ that differs from the base in [modules] bindings and/or [overlay] (the
                         "recipe edit" that swaps a kernel module)
    module:<module-id>   a catalogue module; every interface it provides that the base binds elsewhere is rebound
    branch:<ref>         a branch (or commit) of the pod strixite repo: its kernels/ files that differ from the origin
                         commit become overlay modules (kernel.flerp.loop_<slug>_<name> over kernel.strixite.<name>) and
                         a generated recipe binds them. Only kernel modules are overlaid; other changes are listed.

Stages (each writes under gate/results/strix-loop/<run>/):
    plan    resolve the target to swaps, write the variant recipe (recipes/qwen38-flash-next.strixite-loop.toml,
            gitignored) and plan.json with the prediction fields to fill
    build   assemble + cmake + ninja the base and the variant recipe (compile_commands exported for the isa scan)
    micro   for each swapped interface with a gate/interface/<name>.toml: the interface driver bound to the base
            provider and to the variant, run on K in process order A B A B; parity dumps compared bit for bit
            (0 differing 32-bit words required), bench cases' median us tabulated (variant / base)
    model   strixflerp-prefill-bench from both builds on K, process order A B A B (or A B), one weights load per
            process, every prompt x rounds inside it: logits sha256 (last 16 rows) and greedy ids vs base per prompt,
            prefill tok/s (second request), decode tok/s (T = 1, MTP off), device bytes, load time per process
    isa     tools/isa-scan.py over the changed kernel units of both builds: scratch / occupancy / VGPR diff
    report  verdict.json + summary.md, and the one-line canary pin for a human

Weights: the Qwen3.8-Flash-Next weights.strixw is 74.6 GB. K has 124 GiB of RAM shared with the GPU (GTT) and
strixite's loader copies host -> device, so a /dev/shm (PLACE tier ram) copy plus the device copy does not fit; the
bench reads the pod-weights staged NVMe copy on control-node over K's NFS mount, once per process.
--weights may name a K placement path for a model that fits (docs/STRIX-LOOP.md).
"""
import argparse
import glob
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gate"))
from flerplib import assemble  # noqa: E402
from flerplib.catalog import Catalog, Module, Recipe, read_toml  # noqa: E402
from flerplib import tomlw  # noqa: E402
from flerplib.resolve import Assembly  # noqa: E402

BASE_RECIPE = "qwen38-flash-next.strixite-pf"
LOOP_RECIPE = "qwen38-flash-next.strixite-loop"          # generated per run, gitignored
HOST_CONFIG = "gate/qwen38-flash-next.strixite.toml"      # weights, n-gram table (NFS paths K sees too)
STRIXITE_REPO = os.environ.get("STRIX_LOOP_STRIXITE_REPO", "/data/projects/strixite-d19945e")
# Module kinds a branch: target may overlay (recipe [overlay] compiles the branch's copy in the origin module's target).
# kernel since the first run; format, runtime and executor since the in-place loader work (a loader change and the model
# code that reads weights back, gated the same way).
OVERLAY_KINDS = {"kernel", "format", "runtime", "executor"}
PROBE_MODULE = "kernel.strixite.moe_router"               # the null run's microbench / isa probe
PROMPTS = [128, 512, 1024, 2048, 4096, 8192, 16384, 32768]
KLIB = "/opt/rocm/lib:/opt/rocm/lib/llvm/lib"             # K's own ROCm 7.2.4 runtime; the binaries only need libamdhip64.so.7
BUILD_JOBS = int(os.environ.get("STRIX_LOOP_JOBS", "12"))
COOL_C, LOAD_MAX = 50, 6                                   # as gate/kbench_backend.py
NULL_BAND = 0.005                                          # identical binaries on K: ratios 0.9984-1.0042 (run 20261115T031124Z-null)


def utc(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def sh(cmd, log=None, cwd=ROOT, timeout=None, check=False, env=None):
    t0 = time.time()
    if log:
        with open(log, "ab") as f:
            f.write(("$ " + (cmd if isinstance(cmd, str) else " ".join(cmd)) + "\n").encode())
            f.flush()
            rc = subprocess.call(cmd, shell=isinstance(cmd, str), cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                                 timeout=timeout, env=env)
    else:
        rc = subprocess.call(cmd, shell=isinstance(cmd, str), cwd=cwd, timeout=timeout, env=env)
    if check and rc != 0:
        raise RuntimeError(f"{cmd if isinstance(cmd, str) else ' '.join(cmd)} exited {rc}" + (f"; see {log}" if log else ""))
    return rc, round(time.time() - t0, 1)


def out(cmd, cwd=ROOT):
    return subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, capture_output=True, text=True)


def jdump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)


def med_range(xs):
    xs = [float(x) for x in xs]
    if not xs:
        return {}
    m = statistics.median(xs)
    return {"median": round(m, 3), "min": round(min(xs), 3), "max": round(max(xs), 3), "n": len(xs),
            "spread_pct": round(100 * (max(xs) - min(xs)) / m, 2) if m else None}


def toml_text(d):
    """A whole document from a dict: scalars first, then [table]s and [[arrays of tables]]; keys quoted when not bare."""
    def key(k):
        return k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else json.dumps(k)

    def table(t):
        w = max((len(key(k)) for k in t), default=0)
        return "".join(f"{key(k).ljust(w)} = {tomlw.value(v)}\n" for k, v in t.items())
    scalars = {k: v for k, v in d.items() if not isinstance(v, dict) and not (isinstance(v, list) and v and isinstance(v[0], dict))}
    parts = [table(scalars)] if scalars else []
    for k, v in d.items():
        if isinstance(v, dict):
            parts.append(f"[{key(k)}]\n" + table(v))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            parts += [f"[[{key(k)}]]\n" + table(x) for x in v]
    return "\n".join(parts)


# ------------------------------------------------------------------------------------------------------- plan
def strixite_origin_commit(cat):
    data = read_toml(os.path.join(ROOT, "catalog", "strixite", "ORIGIN.toml"))
    return data.get("commit") or "d19945e"


def module_by_path(cat, origin, path):
    for m in cat.modules:
        if m.origin == origin and path in m.paths:
            return m
    return None


def derive_overlay_modules(cat, ref, run):
    """Overlay modules for a strixite branch: every strixite kernel module with a file that differs from the origin
    commit gets kernel.flerp.loop_<slug>_<name> holding the branch's copy of ALL its files (so headers travel too).
    Returns (overlay map {derived: origin}, notes, sha)."""
    rp = out(["git", "-C", STRIXITE_REPO, "rev-parse", "--verify", ref + "^{commit}"])
    if rp.returncode != 0:
        raise SystemExit(f"branch:{ref}: not a commit of {STRIXITE_REPO}: {rp.stderr.strip()}")
    sha = rp.stdout.strip()
    base = strixite_origin_commit(cat)
    diff = out(["git", "-C", STRIXITE_REPO, "diff", "--name-only", base, sha])
    if diff.returncode != 0:
        raise SystemExit(f"git diff {base} {sha}: {diff.stderr.strip()}")
    changed = [p for p in diff.stdout.split() if p]
    slug = re.sub(r"[^a-z0-9]+", "_", ref.lower()).strip("_")[-24:] or sha[:8]
    overlays, notes, done = {}, [], set()
    for p in changed:
        m = module_by_path(cat, "strixite", p)
        if m is None:
            notes.append(f"{p}: changed on {ref} but belongs to no strixite module; not overlaid")
            continue
        if m.kind not in OVERLAY_KINDS:
            notes.append(f"{p}: changed on {ref} in {m.id} ({m.kind}); the loop overlays {sorted(OVERLAY_KINDS)} modules only, not overlaid")
            continue
        if m.id in done:
            continue
        done.add(m.id)
        did = f"{m.kind}.flerp.loop_{slug}_{m.dir_name}"
        ddir = os.path.join(ROOT, "catalog", "flerp", m.kind, f"loop_{slug}_{m.dir_name}")
        shutil.rmtree(ddir, ignore_errors=True)
        root = os.path.join("src", "flerp", f"loop_{slug}_{m.dir_name}")
        srcs = []
        for path in m.paths:
            blob = out(["git", "-C", STRIXITE_REPO, "show", f"{sha}:{path}"])
            if blob.returncode != 0:
                raise SystemExit(f"git show {sha}:{path}: {blob.stderr.strip()}")
            dst = os.path.join(ddir, root, path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w") as f:
                f.write(blob.stdout)
            if "src/" + path in m.sources:
                srcs.append(os.path.join(root, path))
        data = {"id": did, "kind": m.kind,
                "summary": f"{m.id} as on strixite {ref} @ {sha[:12]} (STRIX-LOOP overlay, generated by gate/strix_loop.py run {run})",
                "provenance": {"origin": "flerp", "repo": "in-tree", "commit": "in-tree",
                               "paths": [f"flerp/loop_{slug}_{m.dir_name}/{p}" for p in m.paths], "authors": ["pod"],
                               "modified": True, "upstream": f"strixite {STRIXITE_REPO} {sha} ({ref}): {', '.join(m.paths)}"},
                "licence": {"spdx": m.spdx},
                "build": dict(m.build, sources=srcs, includes=[root]),
                "provides": [{"interface": f"loop.overlay/{m.dir_name}/-", "api": f"the symbols of {m.id}",
                              "numeric": "as the branch", "regime": "recipe [overlay]"}],
                "requires": [{"module": m.id}],
                "gate": {"parity": "none", "bench": "none", "status": "harvested", "compile": "n/a",
                         "reason": "compiled as an overlay inside the STRIX-LOOP variant recipe"}}
        with open(os.path.join(ddir, "module.toml"), "w") as f:
            f.write(toml_text(data))
        overlays[did] = m.id
        notes.append(f"{m.id}: {', '.join(p for p in changed if p in m.paths)} differ on {ref}; overlay {did}")
    return overlays, notes, sha


def write_loop_recipe(base, bindings, overlays, name, summary):
    data = json.loads(json.dumps(base.data))
    data["name"] = name
    data["modules"] = bindings
    if overlays:
        data["overlay"] = overlays
    else:
        data.pop("overlay", None)
    data.pop("gate", None)
    path = os.path.join(ROOT, "recipes", name + ".toml")
    with open(path, "w") as f:
        f.write(f"# {summary}\n# Generated by gate/strix_loop.py; gitignored; the run's plan.json records what it holds.\n")
        f.write(toml_text(data))
    return path


def stage_plan(a, ctx):
    cat = Catalog(ROOT)
    base = cat.recipe(a.base)
    if base is None:
        raise SystemExit(f"no base recipe {a.base}")
    bindings = dict(base.bindings)
    overlays, notes, kind, detail = {}, [], None, {}
    t = a.target
    if t == "null":
        kind = "null"
        variant_name = LOOP_RECIPE
        probe = cat.by_id.get(PROBE_MODULE)
        swaps = [{"interface": p["interface"], "base": probe.id, "variant": probe.id, "overlay": None}
                 for p in probe.provides if bindings.get(p["interface"]) == probe.id]
        write_loop_recipe(base, bindings, {}, variant_name, f"null run of {base.name}: the same bindings")
    elif t.startswith("recipe:") or t.endswith(".toml"):
        kind = "recipe"
        name = t[len("recipe:"):] if t.startswith("recipe:") else os.path.basename(t)[:-5]
        r = cat.recipe(name)
        if r is None:
            raise SystemExit(f"no recipe {name} in recipes/")
        variant_name = r.name
        vb, vo = r.bindings, r._table("overlay")
        swaps = [{"interface": i, "base": bindings.get(i), "variant": vb[i], "overlay": None}
                 for i in sorted(set(bindings) | set(vb)) if bindings.get(i) != vb.get(i)]
        for did, oid in vo.items():
            om = cat.by_id.get(oid)
            for i, b in bindings.items():
                if b == oid:
                    swaps.append({"interface": i, "base": oid, "variant": oid, "overlay": did})
            if om is None:
                notes.append(f"[overlay] {did} -> {oid}: unknown origin module")
        overlays = dict(vo)
    elif t.startswith("module:"):
        kind = "module"
        mid = t[len("module:"):]
        m = cat.by_id.get(mid)
        if m is None:
            raise SystemExit(f"no module {mid}")
        swaps = []
        for p in m.provides:
            i = p["interface"]
            if i in bindings and bindings[i] != mid:
                swaps.append({"interface": i, "base": bindings[i], "variant": mid, "overlay": None})
                bindings[i] = mid
        if not swaps:
            raise SystemExit(f"{mid} provides no interface the base recipe binds elsewhere; for a module that replaces "
                             f"an origin module's files use branch:<ref> or a recipe with [overlay]")
        variant_name = LOOP_RECIPE
        write_loop_recipe(base, bindings, {}, variant_name, f"{base.name} with {mid} bound: " + ", ".join(s["interface"] for s in swaps))
    else:
        kind = "branch"
        ref = t[len("branch:"):] if t.startswith("branch:") else t
        overlays, notes, sha = derive_overlay_modules(cat, ref, ctx["run"])
        detail = {"ref": ref, "sha": sha, "repo": STRIXITE_REPO, "origin_commit": strixite_origin_commit(cat)}
        if not overlays:
            raise SystemExit(f"branch:{ref}: no strixite kernel module differs from the origin commit; notes: {notes}")
        cat = Catalog(ROOT)   # the derived modules are on disk now
        base = cat.recipe(a.base)
        swaps = []
        for did, oid in overlays.items():
            for i, b in bindings.items():
                if b == oid:
                    swaps.append({"interface": i, "base": oid, "variant": oid, "overlay": did})
        variant_name = LOOP_RECIPE
        write_loop_recipe(base, bindings, overlays, variant_name, f"{base.name} with strixite {ref} @ {sha[:12]} kernels overlaid")
    cat = Catalog(ROOT)
    var = cat.recipe(variant_name)
    errs = []
    for r in (base, var):
        asm = Assembly(cat, r)
        errs += [f"{r.name}: {e}" for e in asm.errors]
    # the units the isa scan and the microbench look at
    units = sorted({os.path.basename(s) for s in swap_sources(cat, swaps, overlays)})
    plan = {"run": ctx["run"], "target": t, "kind": kind, "detail": detail, "base_recipe": base.name,
            "variant_recipe": variant_name, "swaps": swaps, "overlays": overlays, "notes": notes, "units": units,
            "assembly_errors": errs, "when_utc": utc(), "card": a.card,
            "prediction": {"identity": "logits sha256 and greedy ids equal to base on every prompt" if kind == "null" or
                           a.expect == "identical" else "as the card's criteria file", "speed": a.predict or
                           ("prefill and decode tok/s ratio 1.00 within the base's own spread" if kind == "null" else "fill in before the run")}}
    jdump(os.path.join(ctx["res"], "plan.json"), plan)
    if errs:
        raise SystemExit("assembly errors:\n  " + "\n  ".join(errs))
    return plan


def swap_sources(cat, swaps, overlays):
    srcs = []
    for s in swaps:
        for mid in {s["base"], s["variant"], s.get("overlay")}:
            m = cat.by_id.get(mid) if mid else None
            if m:
                srcs += [os.path.join(m.dir, x) for x in m.sources]
    return srcs


# ------------------------------------------------------------------------------------------------------- build
def build_recipe(name, ctx, log):
    tree = os.path.join(ROOT, "build", name)
    rc, _ = sh([os.path.join(ROOT, "flerp"), "assemble", name], log, check=True)
    ninja = os.path.join(tree, "ninja")
    sh(["cmake", "-G", "Ninja", "-S", tree, "-B", ninja, "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"], log, check=True, timeout=900)
    t0 = time.time()
    sh(["nice", "-n", "19", "ninja", "-C", ninja, "-j", str(BUILD_JOBS)], log, check=True, timeout=4 * 3600)
    return {"tree": os.path.relpath(tree, ROOT), "ninja": ninja, "build_s": round(time.time() - t0, 1),
            "bench": os.path.join(ninja, "strixflerp-prefill-bench"),
            "server": next(iter(glob.glob(os.path.join(ninja, f"strixflerp-{name}"))), None)}


def stage_build(a, ctx, plan):
    log = os.path.join(ctx["res"], "build.log")
    open(log, "w").close()
    res = {}
    for side, name in (("base", plan["base_recipe"]), ("variant", plan["variant_recipe"])):
        res[side] = build_recipe(name, ctx, log)
        if not os.path.isfile(res[side]["bench"]):
            raise RuntimeError(f"{name}: no strixflerp-prefill-bench in {res[side]['ninja']}; the base recipe must carry the bench binary")
        res[side]["bench_sha256"] = out(["sha256sum", res[side]["bench"]]).stdout.split()[0]
    res["identical_binaries"] = res["base"]["bench_sha256"] == res["variant"]["bench_sha256"]
    return res


# ------------------------------------------------------------------------------------------------------- K jobs
def kbench_run(ctx, stage_dir, cmd, fetch, why, timeout_s, tag):
    """One pod-kbench k_run job: stage_dir rsynced to K, cmd run there, fetch globs copied back. Returns the job's
    result.json plus where its files are."""
    args = {"_card": ctx["card"], "_call": f"loop-{tag}", "src": stage_dir, "cmd": cmd, "fetch": fetch,
            "timeout_s": timeout_s, "queue_wait_s": 6 * 3600, "why": why}
    af = os.path.join(ctx["res"], f"kbench-{tag}.args.json")
    jdump(af, args)
    log = os.path.join(ctx["res"], f"kbench-{tag}.txt")
    with open(log, "wb") as f:
        p = subprocess.run(["pod-kbench", "k_run", "--args", af], stdout=f, stderr=subprocess.STDOUT, timeout=timeout_s + 7 * 3600)
    text = open(log, errors="replace").read()
    m = re.search(r"files: (\S+)", text)
    jd = m.group(1) if m else None
    r = json.load(open(os.path.join(jd, "result.json"))) if jd and os.path.isfile(os.path.join(jd, "result.json")) else {}
    r["job_dir"] = jd
    r["pod_kbench_rc"] = p.returncode
    if p.returncode != 0 or not jd:
        raise RuntimeError(f"pod-kbench k_run {tag} failed (rc {p.returncode}); see {log}")
    return r


def runsh_head():
    return ["#!/usr/bin/env bash",
            "# generated by gate/strix_loop.py; runs ON bench box K inside pod-kbench k_run",
            f"export LD_LIBRARY_PATH={KLIB}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}",
            "TCTL=${KB_TCTL:-/sys/class/hwmon/hwmon2/temp1_input}",
            "tctl() { cat $TCTL 2>/dev/null || echo 0; }",
            "cool() { for i in $(seq 60); do awk '{exit !($1 <= %d)}' /proc/loadavg && break; sleep 30; done" % LOAD_MAX,
            "        for i in $(seq 30); do [ \"$(tctl)\" -le %d ] && break; sleep 3; done; }" % (COOL_C * 1000),
            "free_gb() { awk '/MemAvailable/ {printf \"%.1f\", $2/1e6}' /proc/meminfo; }",
            ": > cases.tsv"]


def order_list(order):
    # BB: two variant processes only, for a variant whose weights placement leaves no room
    # for the base's copy on the same box; the base reference then comes from a run of its own and the judge records
    # the base rows as missing.
    return {"ABAB": ["A", "B", "A", "B"], "AB": ["A", "B"], "BB": ["B", "B"]}[order]


def stage_micro(a, ctx, plan, build):
    """Interface drivers (gate/interface) for each swapped interface, base vs variant, on K."""
    import interface as ig
    cat = Catalog(ROOT)
    configs = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "gate", "interface", "*.toml"))):
        if os.path.basename(f) == "host.toml":
            continue
        d = read_toml(f)
        if "interface" in d:
            configs[d["interface"]] = os.path.basename(f)[:-5]
    res = {"interfaces": {}, "no_driver": []}
    log = os.path.join(ctx["res"], "micro-build.log")
    open(log, "w").close()
    for s in plan["swaps"]:
        iface = s["interface"]
        if iface not in configs:
            res["no_driver"].append(iface)
            continue
        name = configs[iface]
        if name in res["interfaces"]:
            continue
        cfg = ig.load_config(name)
        cfg["host"]["build_jobs"] = BUILD_JOBS
        trees, built = {}, {}
        for side in ("base", "variant"):
            bind = {iface: s["base"] if side == "base" else s["variant"]}
            ovl = {s["overlay"]: s["base"]} if side == "variant" and s.get("overlay") else {}
            if side == "variant" and not s.get("overlay"):
                # a foreign provider needs the bindings its adapter asks for: take them from the variant recipe
                vr = cat.recipe(plan["variant_recipe"])
                for k, v in vr.bindings.items():
                    if k != iface and k not in bind and k.startswith("format."):
                        bind[k] = v
            tree = os.path.join(ROOT, "build", "strix-loop", ctx["run"], "iface-" + name, side)
            shutil.rmtree(tree, ignore_errors=True)
            os.makedirs(tree)
            rdir = os.path.join(ctx["res"], "micro", name, side)
            os.makedirs(rdir, exist_ok=True)
            ddir = os.path.join(ROOT, cfg["driver"])
            drv = Module(os.path.join(ddir, "module.toml"), read_toml(os.path.join(ddir, "module.toml")))
            if drv.id not in cat.by_id:
                cat.modules.append(drv)
                cat.by_id[drv.id] = drv
            data = {"name": f"loop-{name}.{side}", "model": "none",
                    "reference": {"engine": "cpu", "build": "gate/interface/reference"}, "toolchain": cfg["toolchain"],
                    "modules": bind, "binaries": {"driver": {"main": drv.id, "entry": drv.sources[0][len("src/"):]}}}
            if ovl:
                data["overlay"] = ovl
            recipe = Recipe(os.path.join(ROOT, "recipes", data["name"] + ".toml"), data)
            asm, errors = assemble.generate(cat, recipe, tree)
            if errors:
                raise RuntimeError(f"micro {name} {side}: assembly errors: {errors}")
            ninja = os.path.join(tree, "ninja")
            sh(["cmake", "-G", "Ninja", "-S", tree, "-B", ninja], log, check=True, timeout=600)
            sh(["nice", "-n", "19", "ninja", "-C", ninja, "-j", str(BUILD_JOBS)], log, check=True, timeout=3600)
            trees[side] = tree
            built[side] = {"bindings": bind, "overlay": ovl, "driver_sha256": out(["sha256sum", os.path.join(ninja, "driver")]).stdout.split()[0]}
        # stage both drivers and the cases on K
        stage = os.path.join(ROOT, "build", "strix-loop", ctx["run"], "kstage-iface-" + name)
        shutil.rmtree(stage, ignore_errors=True)
        os.makedirs(stage)
        for side in ("base", "variant"):
            shutil.copy2(os.path.join(trees[side], "ninja", "driver"), os.path.join(stage, "driver." + side))
        provider = {"name": "base", "bind": {}}
        with open(os.path.join(stage, "cases.txt"), "w") as f:
            f.write(ig.case_lines(cfg, provider))
        L = runsh_head() + ["rm -rf out dumps; mkdir -p out dumps"]
        n = {"A": 0, "B": 0}
        for p in order_list(a.order):
            n[p] += 1
            side = "base" if p == "A" else "variant"
            tag = f"{p}{n[p]}"
            L += ["cool", f"l0=$(cut -d' ' -f1 /proc/loadavg); t0=$(tctl); s=$(date +%s.%N)",
                  f"mkdir -p dumps/{tag}; timeout 1700 ./driver.{side} cases.txt out/{tag}.json dumps/{tag} > out/{tag}.log 2>&1; rc=$?",
                  f"e=$(date +%s.%N); printf '{tag}\\t{side}\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' $rc $l0 $(cut -d' ' -f1 /proc/loadavg) $t0 $(tctl) $(echo \"$e - $s\" | bc) >> cases.tsv"]
        L += ["cat cases.tsv"]
        with open(os.path.join(stage, "run.sh"), "w") as f:
            f.write("\n".join(L) + "\n")
        r = kbench_run(ctx, stage, "bash ./run.sh", ["cases.tsv", "out/*.json", "out/*.log", "dumps/*/*.bin"],
                       f"strixflerp STRIX-LOOP {ctx['run']} microbench {name}", 4 * 1800, "micro-" + name)
        fetched = os.path.join(r["job_dir"], "fetched")
        rdir = os.path.join(ctx["res"], "micro", name)
        for f in glob.glob(os.path.join(fetched, "out", "*")) + glob.glob(os.path.join(fetched, "cases.tsv")):
            shutil.copy2(f, rdir)
        runs = {}
        for f in sorted(glob.glob(os.path.join(fetched, "out", "*.json"))):
            tag = os.path.basename(f)[:-5]
            try:
                runs[tag] = {c["case"]: c for c in json.load(open(f))}
            except json.JSONDecodeError:
                runs[tag] = {}
        # identity: variant dumps vs base dumps, every parity case, every pairing, 32-bit words
        ident = []
        a_tags = [t for t in runs if t.startswith("A")]
        b_tags = [t for t in runs if t.startswith("B")]
        for c in cfg["case"]:
            if c["kind"] != "parity":
                continue
            row = {"case": c["name"], "differing_words": 0, "compared": 0, "missing": []}
            for ta in a_tags:
                for tb in b_tags:
                    pa, pb = (os.path.join(fetched, "dumps", t, c["name"] + ".bin") for t in (ta, tb))
                    if not (os.path.isfile(pa) and os.path.isfile(pb)):
                        row["missing"].append(f"{ta}/{tb}")
                        continue
                    da, db = open(pa, "rb").read(), open(pb, "rb").read()
                    row["compared"] += 1
                    if da != db:
                        words = sum(1 for k in range(0, max(len(da), len(db)), 4) if da[k:k + 4] != db[k:k + 4])
                        row["differing_words"] += words
            row["identical"] = row["compared"] > 0 and row["differing_words"] == 0 and not row["missing"]
            ident.append(row)
        speed = []
        for c in cfg["case"]:
            if c["kind"] != "bench":
                continue
            va = [runs[t][c["name"]]["median_us"] for t in a_tags if c["name"] in runs[t] and runs[t][c["name"]].get("ran")]
            vb = [runs[t][c["name"]]["median_us"] for t in b_tags if c["name"] in runs[t] and runs[t][c["name"]].get("ran")]
            row = {"case": c["name"], "shape": c.get("shape"), "base_us": med_range(va), "variant_us": med_range(vb)}
            if va and vb:
                row["ratio_variant_over_base"] = round(statistics.median(vb) / statistics.median(va), 4)
                row["outside_noise"] = max(vb) < min(va) or min(vb) > max(va)
            speed.append(row)
        res["interfaces"][name] = {"interface": iface, "swap": s, "built": built, "job": r["job_dir"], "runs": sorted(runs),
                                   "identity": ident, "speed": speed, "kbench_rc": r.get("rc"),
                                   "all_identical": bool(ident) and all(x["identical"] for x in ident)}
    return res


def host_paths(a):
    import strixite_recipe as sr
    cfg = sr.load_config(os.path.join(ROOT, HOST_CONFIG))
    w = a.weights or cfg["host"]["weights"]
    # --variant-weights / --variant-args: the variant processes read another copy of the same
    # weights (a K placement) and take extra bench arguments (e.g. --placement mapped); the base keeps the host paths.
    return {"weights": w, "ngram": a.ngram or cfg["host"]["ngram"], "variant_weights": a.variant_weights or w,
            "variant_args": a.variant_args or ""}


def stage_model(a, ctx, plan, build):
    hp = host_paths(a)
    stage = os.path.join(ROOT, "build", "strix-loop", ctx["run"], "kstage-model")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    for side in ("base", "variant"):
        shutil.copy2(build[side]["bench"], os.path.join(stage, "bench." + side))
    prompts = ",".join(str(p) for p in a.prompts)
    cap = max(a.prompts) + 8192
    L = runsh_head() + ["rm -rf out; mkdir -p out", "echo \"free_gb_start $(free_gb)\" > mem.txt",
                        f"test -r {hp['weights']} || {{ echo \"weights not readable on K: {hp['weights']}\" >&2; exit 3; }}",
                        f"test -r {hp['variant_weights']} || {{ echo \"variant weights not readable on K: {hp['variant_weights']}\" >&2; exit 3; }}",
                        # MemAvailable every 2 s while a process runs: its minimum gives the process's peak RAM use
                        "sampler() { while :; do echo \"$(date +%s) $(free_gb)\" >> out/$1.mem; sleep 2; done; }"]
    n = {"A": 0, "B": 0}
    for p in order_list(a.order):
        n[p] += 1
        side = "base" if p == "A" else "variant"
        tag = f"{p}{n[p]}"
        w = hp["weights"] if side == "base" else hp["variant_weights"]
        extra = (" " + hp["variant_args"]) if side == "variant" and hp["variant_args"] else ""
        envp = (a.variant_env + " ") if side == "variant" and a.variant_env else ""  # K=V words before the command
        L += ["cool", f"l0=$(cut -d' ' -f1 /proc/loadavg); t0=$(tctl); s=$(date +%s.%N)", f"mkdir -p out/{tag}",
              f"sampler {tag} & sp=$!",
              f"{envp}timeout {a.proc_timeout} ./bench.{side} --weights {w} --ngram {hp['ngram']} --out out/{tag} "
              f"--prompts {prompts} --variants {side}:chunk={a.chunk} --rounds {a.rounds} --gen {a.gen} --seed 1234 "
              f"--capacity {cap} --logits-rows 16 --save-logits {'1' if n[p] == 1 else '0'}{extra} > out/{tag}.log 2>&1; rc=$?",
              "kill $sp 2>/dev/null; wait $sp 2>/dev/null",
              f"e=$(date +%s.%N); printf '{tag}\\t{side}\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' $rc $l0 $(cut -d' ' -f1 /proc/loadavg) $t0 $(tctl) $(echo \"$e - $s\" | bc) >> cases.tsv",
              f"echo \"free_gb_after_{tag} $(free_gb)\" >> mem.txt"]
    # keep logits only where the sha differs (KL offline); drop the rest so the fetch stays small
    L += ["python3 - <<'EOF'",
          "import json, glob, os",
          "rows = {}",
          "for f in glob.glob('out/*/results.jsonl'):",
          "    for line in open(f):",
          "        if line.strip():",
          "            r = json.loads(line); rows.setdefault(r['len'], {}).setdefault(os.path.basename(os.path.dirname(f)), set()).add(r['logits_sha256'])",
          "for L, by in rows.items():",
          "    shas = set().union(*by.values())",
          "    if len(shas) == 1:",
          "        for f in glob.glob('out/*/logits.*.%d.f32' % L): os.remove(f)",
          "EOF",
          "cat cases.tsv; cat mem.txt"]
    with open(os.path.join(stage, "run.sh"), "w") as f:
        f.write("\n".join(L) + "\n")
    per_proc = a.proc_timeout
    r = kbench_run(ctx, stage, "bash ./run.sh", ["cases.tsv", "mem.txt", "out/*.log", "out/*.mem", "out/*/results.jsonl", "out/*/*.f32"],
                   f"strixflerp STRIX-LOOP {ctx['run']} full-model gate {plan['target']}", min(6 * 3600, per_proc * len(order_list(a.order)) + 900), "model")
    fetched = os.path.join(r["job_dir"], "fetched")
    mdir = os.path.join(ctx["res"], "model")
    os.makedirs(mdir, exist_ok=True)
    for f in glob.glob(os.path.join(fetched, "*")) + glob.glob(os.path.join(fetched, "out", "*")):
        if os.path.isfile(f):
            shutil.copy2(f, mdir)
    rows = []
    for f in sorted(glob.glob(os.path.join(fetched, "out", "*", "results.jsonl"))):
        tag = os.path.basename(os.path.dirname(f))
        for line in open(f):
            if line.strip():
                row = json.loads(line)
                row["proc"] = tag
                rows.append(row)
        shutil.copy2(f, os.path.join(mdir, f"results.{tag}.jsonl"))
    procs = {}
    ct = os.path.join(fetched, "cases.tsv")
    if os.path.isfile(ct):
        for line in open(ct):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 8:
                procs[p[0]] = {"side": p[1], "rc": int(p[2]), "load_start": float(p[3]), "load_end": float(p[4]),
                               "tctl_start_c": int(p[5]) / 1000, "tctl_end_c": int(p[6]) / 1000, "wall_s": round(float(p[7]), 1)}
    for tag, pr in procs.items():
        pt = [r for r in rows if r["proc"] == tag]
        if pt:
            # the weights load = the process wall minus the time its prompts and decodes took
            busy = sum(r.get("prefill_s", 0) + (len(r.get("gen_ids", [])) / r["decode_tps"] if r.get("decode_tps") else 0) for r in pt)
            pr["bench_busy_s_lower_bound"] = round(busy, 1)
            pr["load_s_upper_bound"] = round(pr["wall_s"] - busy, 1)
            pr["io_read_gb"] = round(max(r.get("io_read_bytes", 0) for r in pt) / 1e9, 2)
            pr["dev_used_after_gib"] = round(max(r.get("dev_used_after", 0) for r in pt) / 2**30, 2)
        # the bench's own load line ("weights loaded in 12.3 s (69.5 GiB, placement mapped), ...") and the RAM the
        # process took at its peak: MemAvailable at the start of the job minus its minimum while the process ran
        lg = os.path.join(fetched, "out", f"{tag}.log")
        if os.path.isfile(lg):
            m = re.search(r"weights loaded in ([0-9.]+) s \(([0-9.]+) GiB(?:, placement ([a-z-]+))?\)", open(lg, errors="replace").read())
            if m:
                pr["load_s"] = float(m.group(1))
                pr["weights_gib"] = float(m.group(2))
                pr["placement"] = m.group(3) or "device"
        ms = os.path.join(fetched, "out", f"{tag}.mem")
        if os.path.isfile(ms):
            vals = [float(l.split()[1]) for l in open(ms) if len(l.split()) == 2]
            if vals:
                pr["mem_avail_min_gb"] = min(vals)
                pr["mem_avail_start_gb"] = vals[0]
                pr["peak_ram_gb"] = round(vals[0] - min(vals), 1)
    return {"job": r["job_dir"], "kbench_rc": r.get("rc"), "procs": procs, "rows": rows, "weights": hp,
            "prompts": a.prompts, "rounds": a.rounds, "order": order_list(a.order), "mem": open(os.path.join(fetched, "mem.txt")).read()
            if os.path.isfile(os.path.join(fetched, "mem.txt")) else ""}


def judge_model(res):
    rows = res["rows"]
    table, all_ident = [], True
    for L in res["prompts"]:
        A = [r for r in rows if r["len"] == L and r["proc"].startswith("A")]
        B = [r for r in rows if r["len"] == L and r["proc"].startswith("B")]
        if not A or not B:
            table.append({"len": L, "missing": True})
            all_ident = False
            continue
        sa, sb = {r["logits_sha256"] for r in A}, {r["logits_sha256"] for r in B}
        ia, ib = {tuple(r["gen_ids"]) for r in A}, {tuple(r["gen_ids"]) for r in B}
        same = sa == sb and len(sa) == 1 and ia == ib and len(ia) == 1
        all_ident = all_ident and same
        first = None
        for x, y in zip(A[0]["gen_ids"], B[0]["gen_ids"]):
            if x != y:
                first = A[0]["gen_ids"].index(x)
                break
        pa = [r["prefill_tps"] for r in A if r["round"] >= 1]
        pb = [r["prefill_tps"] for r in B if r["round"] >= 1]
        da, db = [r["decode_tps"] for r in A if r["round"] >= 1], [r["decode_tps"] for r in B if r["round"] >= 1]   # round 0 is cold
        row = {"len": L, "logits_identical": sa == sb and len(sa) == 1, "greedy_identical": ia == ib and len(ia) == 1,
               "base_sha": sorted(sa), "variant_sha": sorted(sb), "greedy_first_diff": first,
               "prefill_tps_base": med_range(pa), "prefill_tps_variant": med_range(pb),
               "decode_tps_base": med_range(da), "decode_tps_variant": med_range(db),
               "dev_used_gib_base": round(max(r["dev_used_after"] for r in A) / 2**30, 2),
               "dev_used_gib_variant": round(max(r["dev_used_after"] for r in B) / 2**30, 2)}
        if pa and pb:
            row["prefill_ratio"] = round(statistics.median(pb) / statistics.median(pa), 4)
            row["prefill_outside_noise"] = min(pb) > max(pa) or max(pb) < min(pa)
        if da and db:
            row["decode_ratio"] = round(statistics.median(db) / statistics.median(da), 4)
            row["decode_outside_noise"] = min(db) > max(da) or max(db) < min(da)
        table.append(row)
    return table, all_ident


# ------------------------------------------------------------------------------------------------------- isa
def stage_isa(a, ctx, plan, build):
    units = plan["units"] or ["moe_router.hip"]
    rx = "|".join(re.escape(u) for u in units)
    isa = os.path.join(ctx["res"], "isa")
    os.makedirs(isa, exist_ok=True)
    log = os.path.join(isa, "isa-scan.log")
    open(log, "w").close()
    tsv = {}
    for side in ("base", "variant"):
        d = os.path.join(ROOT, "build", "strix-loop", ctx["run"], "isa-" + side)
        shutil.rmtree(d, ignore_errors=True)
        ccdb = os.path.join(build[side]["ninja"], "compile_commands.json")
        rc, _ = sh(["ionice", "-c3", "nice", "-n", "19", "python3", os.path.join(ROOT, "tools", "isa-scan.py"), "compile",
                    "--toolchain", "therock-10.1", "--out", d, "-j", str(BUILD_JOBS), f"ccdb:{ccdb}:({rx})"], log, timeout=3600)
        t = os.path.join(isa, side + ".tsv")
        with open(t, "w") as f:
            subprocess.call(["python3", os.path.join(ROOT, "tools", "isa-scan.py"), "table"] + sorted(glob.glob(os.path.join(d, "*.s"))),
                            stdout=f, stderr=open(log, "ab"))
        rows = tsv_rows(t)
        tsv[side] = {"tsv": os.path.relpath(t, ROOT), "rc": rc, "units": len(glob.glob(os.path.join(d, "*.s"))), "rows": rows}
    diff = out(["python3", os.path.join(ROOT, "tools", "isa-scan.py"), "diff", os.path.join(isa, "base.tsv"), os.path.join(isa, "variant.tsv")])
    with open(os.path.join(isa, "diff.txt"), "w") as f:
        f.write(diff.stdout + diff.stderr)
    scratch = []
    for side in ("base", "variant"):
        for line in open(os.path.join(isa, side + ".tsv")).read().splitlines()[1:]:
            p = line.split("\t")
            if len(p) > 5 and p[5] not in ("0", "scratch") and p[5].isdigit() and int(p[5]) > 0:
                scratch.append({"side": side, "unit": p[0], "kernel": p[1][:80], "scratch_bytes": int(p[5])})
    return {"sides": tsv, "diff_lines": [l for l in diff.stdout.splitlines() if l.strip()], "scratch": scratch, "units_regex": rx,
            "diff_rc": diff.returncode, "problems": isa_problems(tsv, diff.returncode)}


def tsv_rows(path):
    with open(path) as f:
        return len(f.read().splitlines()[1:])


def isa_problems(sides, diff_rc):
    """Why the isa scan cannot vouch for anything; [] when both sides compiled and tabled at least one kernel row.
    A missing toolchain, a failed compile, no .s output or an empty table is a scan that did not run, never a pass."""
    probs = []
    for side in ("base", "variant"):
        s = sides.get(side)
        if not s:
            probs.append(f"{side}: no scan")
            continue
        if s.get("rc") != 0:
            probs.append(f"{side}: isa-scan compile exited {s.get('rc')}")
        if not s.get("units"):
            probs.append(f"{side}: no assembly (.s) produced")
        rows = s.get("rows")
        if rows is None and s.get("tsv") and os.path.isfile(os.path.join(ROOT, s["tsv"])):
            rows = tsv_rows(os.path.join(ROOT, s["tsv"]))      # a state.json from before "rows"
        if not rows:
            probs.append(f"{side}: {side}.tsv has no kernel rows")
    if diff_rc != 0:
        probs.append(f"isa-scan diff exited {diff_rc}")
    return probs


def isa_criterion(isa, kind):
    """The isa verdict row: PASS only when the scan ran on both sides and no variant kernel uses scratch."""
    probs = isa.get("problems")
    if probs is None:                                   # a state.json from before the scan recorded its health
        probs = isa_problems(isa.get("sides", {}), isa.get("diff_rc", 0))
    v = {"criterion": "isa: no kernel of the changed units uses scratch in the variant build",
         "pass": not probs and not any(s["side"] == "variant" for s in isa["scratch"]), "report_only": kind == "null"}
    if probs:
        v["ran"] = False
        v["detail"] = "the isa scan did not run: " + "; ".join(probs)
    return v


# ------------------------------------------------------------------------------------------------------- report
def pub(p):
    """A path as the published result records it: relative to the tree when inside it, a bench job by its id."""
    if not isinstance(p, str) or not os.path.isabs(p):
        return p
    if p.startswith(ROOT + os.sep):
        return os.path.relpath(p, ROOT)
    return os.path.basename(p.rstrip(os.sep))


def report(a, ctx, plan, res):
    verdicts = []
    model = res.get("model")
    table, all_ident = ([], None)
    if model:
        table, all_ident = judge_model(model)
        verdicts.append({"criterion": f"full model on K: logits sha256 (last 16 rows) and greedy ids equal to base on every prompt {model['prompts']}",
                         "pass": bool(all_ident)})
        bad = [p for p, pr in model["procs"].items() if pr["rc"] != 0]
        verdicts.append({"criterion": "every bench process on K exited 0", "pass": not bad and len(model["procs"]) == len(model["order"]),
                         "detail": bad})
    micro = res.get("micro")
    if micro:
        for name, m in micro["interfaces"].items():
            verdicts.append({"criterion": f"microbench {name}: variant parity dumps equal to base's, 0 differing 32-bit words",
                             "pass": m["all_identical"]})
    isa = res.get("isa")
    if isa:
        verdicts.append(isa_criterion(isa, plan["kind"]))
    speed = {}
    if table:
        for key in ("prefill", "decode"):
            rs = [r.get(f"{key}_ratio") for r in table if r.get(f"{key}_ratio")]
            if rs:
                speed[key] = {"ratio_median": round(statistics.median(rs), 4), "ratio_min": min(rs), "ratio_max": max(rs),
                              "outside_noise_prompts": [r["len"] for r in table if r.get(f"{key}_outside_noise")]}
        if plan["kind"] == "null":
            # The first null run (20261115T031124Z-null, K, A B A B, 2 rounds) measured identical binaries at prefill ratios
            # 0.9984-1.0042 while each side's own 2-sample spread was 0.01-0.3%: "inside the other side's spread" is not a
            # criterion two samples can meet, so the zero is a band: every prompt's ratio within NULL_BAND of 1.00.
            band = all(abs(r["prefill_ratio"] - 1) <= NULL_BAND for r in table if "prefill_ratio" in r)
            verdicts.append({"criterion": f"null run: prefill tok/s ratio within 1.00 +- {NULL_BAND:.3f} on every prompt "
                                          f"(the noise floor the first null run measured on K)", "pass": band})
    ok = all(v["pass"] for v in verdicts if not v.get("report_only"))
    serving = f"strixite_loop_{ctx['run']}"
    pin = {"line": f"pod-engine pin <letter> {serving}",
           "before": [f"register the serving: a SERVE_{serving} block in engines.conf (pod-engine register) whose binary is "
                      f"{pub(res.get('build', {}).get('variant', {}).get('server')) or 'the variant server'} copied to "
                      f"/data/models/strix-infer/runtime/bin/ (NFS, replace in place, never mv over a path workers exec)",
                      "one worker, the canary letter a human picks; the keeper converges it at its next pass",
                      "pod-engine pin <letter> - removes the pin"],
           "human": True, "only_if": "every criterion above passed and the gain is wanted"}
    verdict = {"run": ctx["run"], "card": a.card, "when_utc": utc(), "host": "control-node (builds, isa) + bench box K via pod-kbench",
               "target": plan["target"], "kind": plan["kind"], "detail": plan["detail"], "swaps": plan["swaps"],
               "overlays": plan["overlays"], "notes": plan["notes"], "prediction": plan["prediction"],
               "verdict": "PASS" if ok else "FAIL", "criteria": verdicts, "speed": speed, "model_table": table,
               "model_procs": model["procs"] if model else {}, "model_job": pub(model["job"]) if model else None,
               "weights": model["weights"] if model else None, "order": model["order"] if model else None,
               "micro": {k: {"identity": v["identity"], "speed": v["speed"], "job": pub(v["job"]), "all_identical": v["all_identical"]}
                         for k, v in micro["interfaces"].items()} if micro else {},
               "micro_no_driver": micro["no_driver"] if micro else [],
               "isa": {"diff_lines": isa["diff_lines"], "scratch": isa["scratch"], "units": isa["units_regex"],
                       "problems": isa_criterion(isa, plan["kind"]).get("detail")} if isa else None,
               "build": {k: {kk: pub(vv) for kk, vv in v.items() if kk != "ninja"} for k, v in res.get("build", {}).items() if isinstance(v, dict)},
               "identical_binaries": res.get("build", {}).get("identical_binaries"),
               "canary_pin": pin}
    jdump(os.path.join(ctx["res"], "verdict.json"), verdict)
    L = [f"# STRIX-LOOP run {ctx['run']}: {plan['target']} -> **{verdict['verdict']}**", "",
         f"{utc()}. Builds and isa on control-node, kernels and the model on bench box K through pod-kbench "
         f"(jobs: {', '.join(x for x in [verdict['model_job']] + [v['job'] for v in verdict['micro'].values()] if x)}). "
         f"Base `{plan['base_recipe']}`, variant `{plan['variant_recipe']}`. Prediction: {plan['prediction']}.", ""]
    for v in verdicts:
        L.append(f"- **{'REPORT' if v.get('report_only') else ('PASS' if v['pass'] else 'FAIL')}** {v['criterion']}"
                 + (f" ({v['detail']})" if v.get("detail") else ""))
    if plan["swaps"]:
        L += ["", "Swaps: " + "; ".join(f"{s['interface']}: {s['base']} -> {s['variant']}" + (f" overlay {s['overlay']}" if s.get("overlay") else "")
                                      for s in plan["swaps"])]
    if plan["notes"]:
        L += ["", "Notes: " + " | ".join(plan["notes"])]
    if table:
        L += ["", f"Full model on K, process order {' '.join(model['order'])}, {model['rounds']} rounds a process, chunk {a.chunk}, "
                  f"weights `{model['weights']['weights']}`:", "",
              "| prompt | logits = base | greedy = base (first diff) | prefill tok/s base med (min-max) | variant | ratio | outside noise | decode tok/s base / variant | ratio | device GiB |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in table:
            if r.get("missing"):
                L.append(f"| {r['len']} | missing | | | | | | | | |")
                continue
            pb, pv = r["prefill_tps_base"], r["prefill_tps_variant"]
            L.append(f"| {r['len']} | {r['logits_identical']} | {r['greedy_identical']} ({r['greedy_first_diff']}) | "
                     f"{pb.get('median')} ({pb.get('min')}-{pb.get('max')}) | {pv.get('median')} ({pv.get('min')}-{pv.get('max')}) | "
                     f"{r.get('prefill_ratio')} | {r.get('prefill_outside_noise')} | {r['decode_tps_base'].get('median')} / "
                     f"{r['decode_tps_variant'].get('median')} | {r.get('decode_ratio')} | {r['dev_used_gib_base']} / {r['dev_used_gib_variant']} |")
        L += ["", "Processes (K): " + "; ".join(f"{t} {p['side']} rc {p['rc']} wall {p['wall_s']} s, load <= {p.get('load_s_upper_bound')} s, "
                                                f"read {p.get('io_read_gb')} GB, device {p.get('dev_used_after_gib')} GiB, Tctl {p['tctl_start_c']}->{p['tctl_end_c']} C"
                                                for t, p in model["procs"].items())]
        if model.get("mem"):
            L += ["", "K MemAvailable (GB): " + model["mem"].replace("\n", "; ").strip("; ")]
    if micro:
        for name, m in micro["interfaces"].items():
            L += ["", f"Microbench `{name}` ({m['interface']}) on K, order {' '.join(order_list(a.order))}: parity cases "
                      f"{sum(1 for x in m['identity'] if x['identical'])}/{len(m['identity'])} identical "
                      f"({sum(x['differing_words'] for x in m['identity'])} differing words)", "",
                  "| bench case | shape | base us med (min-max) | variant us | ratio variant/base | outside noise |", "|---|---|---|---|---|---|"]
            for s in m["speed"]:
                b, v = s["base_us"], s["variant_us"]
                L.append(f"| {s['case']} | {s.get('shape')} | {b.get('median')} ({b.get('min')}-{b.get('max')}) | "
                         f"{v.get('median')} ({v.get('min')}-{v.get('max')}) | {s.get('ratio_variant_over_base')} | {s.get('outside_noise')} |")
        if micro["no_driver"]:
            L += ["", "No interface driver (gate/interface) for: " + ", ".join(micro["no_driver"]) + "; the full-model gate decides."]
    if isa and verdict["isa"]["problems"]:
        L += ["", f"ISA scan (units /{isa['units_regex']}/): NOT RUN, {verdict['isa']['problems']}"]
    elif isa:
        L += ["", f"ISA scan (units /{isa['units_regex']}/): {len(isa['diff_lines'])} changed rows" +
              ("; " + "; ".join(isa["diff_lines"][:8]) if isa["diff_lines"] else "") +
              (f"; scratch users: {isa['scratch']}" if isa["scratch"] else "; no scratch in either build")]
    L += ["", f"Canary pin (a human runs it, only if every criterion passed and the gain is wanted): `{pin['line']}`", ""]
    L += [f"  before: {b}" for b in pin["before"]]
    with open(os.path.join(ctx["res"], "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    return verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("target")
    ap.add_argument("--run", default=None, help="run name (default UTC stamp + target slug)")
    ap.add_argument("--stages", default="plan,build,micro,model,isa,report")
    ap.add_argument("--order", default="ABAB", choices=["ABAB", "AB", "BB"])
    ap.add_argument("--base", default=BASE_RECIPE)
    ap.add_argument("--prompts", default=",".join(str(p) for p in PROMPTS))
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--proc-timeout", type=int, default=2700, help="seconds per bench process on K (load + all prompts)")
    ap.add_argument("--card", default=os.environ.get("STRIX_LOOP_CARD"), help="the card the bench jobs are filed under "
                    "(default: STRIX_LOOP_CARD, which tools/strix-loop.sh sets from the card branch)")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--variant-weights", default=None, help="weights path the variant processes read (default: --weights)")
    ap.add_argument("--variant-args", default=None, help="extra strixflerp-prefill-bench arguments for the variant processes, e.g. '--placement mapped'")
    ap.add_argument("--variant-env", default=None, help="environment words set for the variant processes before they start, e.g. 'STRIX_MAPPED_COARSE=1'")
    ap.add_argument("--expect", default="identical", choices=["identical", "report"])
    ap.add_argument("--predict", default=None, help="the speed prediction, written before the run")
    a = ap.parse_args()
    if not a.card:
        raise SystemExit("strix_loop: set --card or STRIX_LOOP_CARD (the card the bench jobs are filed under)")
    a.prompts = [int(x) for x in a.prompts.split(",")]
    slug = re.sub(r"[^a-z0-9]+", "-", a.target.lower()).strip("-")[:32]
    run = a.run or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + slug
    ctx = {"run": run, "res": os.path.join(ROOT, "gate", "results", "strix-loop", run), "card": a.card}
    os.makedirs(ctx["res"], exist_ok=True)
    state = os.path.join(ctx["res"], "state.json")
    res = json.load(open(state)) if os.path.exists(state) else {}
    stages = a.stages.split(",")
    print(f"strix-loop run {run}: {a.target} stages {stages} -> {os.path.relpath(ctx['res'], ROOT)}", flush=True)
    plan = stage_plan(a, ctx) if "plan" in stages or "plan" not in res else res["plan"]
    res["plan"] = plan
    jdump(state, res)
    if "build" in stages:
        res["build"] = stage_build(a, ctx, plan)
        jdump(state, res)
        print(f"built base {res['build']['base']['build_s']} s, variant {res['build']['variant']['build_s']} s, identical binaries {res['build']['identical_binaries']}", flush=True)
    build = res.get("build")
    if build is None:
        raise SystemExit("no build in this run's state; run the build stage first")
    if "micro" in stages:
        res["micro"] = stage_micro(a, ctx, plan, build)
        jdump(state, res)
        print("micro: " + json.dumps({k: v["all_identical"] for k, v in res["micro"]["interfaces"].items()}), flush=True)
    if "model" in stages:
        res["model"] = stage_model(a, ctx, plan, build)
        jdump(state, res)
        print(f"model: K job {res['model']['job']} procs {json.dumps({t: p['rc'] for t, p in res['model']['procs'].items()})}", flush=True)
    if "isa" in stages:
        res["isa"] = stage_isa(a, ctx, plan, build)
        jdump(state, res)
    if "report" in stages:
        v = report(a, ctx, plan, res)
        print(json.dumps({"verdict": v["verdict"], "criteria": v["criteria"], "speed": v["speed"], "pin": v["canary_pin"]["line"]}, indent=1))


if __name__ == "__main__":
    main()
