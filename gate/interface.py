#!/usr/bin/env python3
"""Interface gate (SPEC.md section 9, phase 4): every provider of one interface, on the real GPU, against a CPU
reference written in this repository. Run on the gfx1151 build host:

    ./flerp gate interface <name> [--only PROVIDER[,PROVIDER]] [--build-only]
    ./flerp gate module <module-id>          the same gate, for the providers that are or wrap that module

<name> is a config file gate/interface/<name>.toml. It names the interface, the driver program
(gate/interface/drivers/<driver>/, which calls the interface through the consumer's launch API and holds no
kernel of its own), the providers with the bindings that select each, a fixed list of cases (shapes, seeds) and
the pass criteria. The criteria are written before the first run and are not edited to make a run pass; the
config's sha256 is recorded with every result.

For each provider the gate
  1. assembles the driver with the interface bound to that provider (`flerplib.assemble`, the same code that
     assembles a recipe: foreign modules are built by their origin's compile recipe, adapters take over symbols),
     configures and builds it under build/gate-interface/<name>/<provider>/;
  2. checks the GPU is free and takes the lease, as the recipe gate does;
  3. runs the driver over the cases. Parity cases compare the provider's output with the CPU reference in the
     driver and dump the raw output; bench cases time back-to-back launches at decode-sized and prefill-sized
     shapes. The host's load average is recorded with every provider's run;
  4. compares the providers' dumps with each other, bit for bit.
Results: gate/results/interface/<name>/ (result.json, summary.md, the config as run, and per provider the
assembly manifest, build log, driver output and adapter tallies).
"""
import argparse
import array
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gate"))
from flerplib import assemble  # noqa: E402
from flerplib.catalog import Catalog, Module, Recipe, read_toml  # noqa: E402
import strixite_recipe as guard  # noqa: E402  (GPU-free check and lease: one implementation for every gate)

CONFIG_DIR = os.path.join(ROOT, "gate", "interface")
TALLY = re.compile(r"^flerp-adapter (\S+) provider=(\S+) foreign_launches=(\d+) origin_launches=(\d+) \(exit\)", re.M)
# bit-identical: the CPU reference's bits. order-bound: within the bound any summation order satisfies.
# structure: the discrete part (flags, selected ids) equal, values reported. report: recorded, never decides.
PARITY_LEVELS = ("bit-identical", "order-bound", "structure", "report")


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_config(name):
    path = os.path.join(CONFIG_DIR, name + ".toml")
    if not os.path.isfile(path):
        have = sorted(f[:-5] for f in os.listdir(CONFIG_DIR) if f.endswith(".toml") and f != "host.toml")
        sys.exit(f"no interface gate '{name}' (have: {', '.join(have)})")
    cfg = read_toml(path)
    cfg["_path"], cfg["_name"], cfg["_sha256"] = path, name, sha256_file(path)
    cfg["host"] = read_toml(os.path.join(CONFIG_DIR, "host.toml"))["host"]
    return cfg


def configs_for_module(mid):
    """Interface gates in which the module is a provider, or the foreign module behind one."""
    out = []
    for f in sorted(os.listdir(CONFIG_DIR)):
        if not f.endswith(".toml") or f == "host.toml":
            continue
        cfg = read_toml(os.path.join(CONFIG_DIR, f))
        names = [p["name"] for p in cfg.get("provider", []) if mid in (p.get("module"), p.get("wraps"))]
        if names:
            out.append((f[:-5], names))
    return out


def driver_recipe(cat, cfg, provider):
    """An in-memory recipe: one binary, the driver, with the provider's bindings."""
    ddir = os.path.join(ROOT, cfg["driver"])
    drv = Module(os.path.join(ddir, "module.toml"), read_toml(os.path.join(ddir, "module.toml")))
    if drv.id not in cat.by_id:
        cat.modules.append(drv)
        cat.by_id[drv.id] = drv
    data = {"name": f"gate-{cfg['_name']}.{provider['name']}", "model": "none",
            "reference": {"engine": "cpu", "build": "gate/interface/reference"}, "toolchain": cfg["toolchain"],
            "modules": dict(provider["bind"]),
            "binaries": {"driver": {"main": drv.id, "entry": drv.sources[0][len("src/"):]}}}
    return Recipe(os.path.join(ROOT, "recipes", data["name"] + ".toml"), data)


def provider_cases(cfg, provider):
    """The cases that apply to a provider: all of them, or those its `cases` pattern matches."""
    rx = re.compile(provider["cases"]) if provider.get("cases") else None
    return [c for c in cfg["case"] if rx is None or rx.search(c["name"])]


def case_lines(cfg, provider):
    lines = []
    for c in provider_cases(cfg, provider):
        kv = " ".join(f"{k}={v}" for k, v in c.items() if k not in ("name", "require", "note", "shape", "dump"))
        lines.append(f"{c['name']} {kv}")
    return "\n".join(lines) + "\n"


def run_logged(cmd, log, timeout, cwd=ROOT):
    with open(log, "ab") as f:
        f.write(("$ " + " ".join(cmd) + "\n").encode())
        f.flush()
        try:
            return subprocess.call(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            f.write(b"\nTIMEOUT\n")
            return 124


def build_provider(cat, cfg, provider, tree, res):
    out = {"tree": os.path.relpath(tree, ROOT)}
    log = os.path.join(res, "build.txt")
    open(log, "w").close()
    recipe = driver_recipe(cat, cfg, provider)
    asm, errors = assemble.generate(cat, recipe, tree)
    if errors:
        out.update({"built": False, "errors": errors})
        return out
    shutil.copyfile(os.path.join(tree, "assembly.json"), os.path.join(res, "assembly.json"))
    with open(os.path.join(tree, "assembly.json")) as f:
        man = json.load(f)
    out["foreign_modules"] = man.get("foreign", {})
    out["code_objects"] = man.get("code_objects", [])
    out["renames"] = man.get("renames", {})
    out["contract_changes"] = man.get("contract_changes", [])
    ninja = os.path.join(tree, "ninja")
    t0 = time.time()
    rc = run_logged(["cmake", "-G", "Ninja", "-S", tree, "-B", ninja], log, 600)
    if rc == 0:
        rc = run_logged(["ninja", "-C", ninja, "-j", str(cfg["host"].get("build_jobs", 8))], log, 3600)
    binary = os.path.join(ninja, "driver")
    out.update({"built": rc == 0 and os.path.isfile(binary), "build_s": round(time.time() - t0, 1), "rc": rc})
    if out["built"]:
        out["driver_sha256"] = sha256_file(binary)
    else:
        with open(log, errors="replace") as f:
            out["log_tail"] = [ln for ln in f.read().splitlines() if "error" in ln.lower()][:12]
    return out


def run_provider(cfg, provider, tree, res):
    """Runs the built driver over the cases. The caller holds the lease and has checked the GPU."""
    cases = os.path.join(tree, "cases.txt")
    with open(cases, "w") as f:
        f.write(case_lines(cfg, provider))
    dumps = os.path.join(tree, "dumps")
    shutil.rmtree(dumps, ignore_errors=True)
    os.makedirs(dumps)
    out_json, log = os.path.join(res, "driver.json"), os.path.join(res, "driver.txt")
    if os.path.isfile(out_json):
        os.remove(out_json)
    ok, state = guard.gpu_free(cfg)
    if not ok:
        return {"ran": False, "error": f"the GPU is not free: {state}"}
    load0 = os.getloadavg()
    t0 = time.time()
    with open(log, "wb") as f:
        try:
            rc = subprocess.call([os.path.join(tree, "ninja", "driver"), cases, out_json, dumps], stdout=f,
                                 stderr=subprocess.STDOUT, timeout=cfg["host"].get("driver_timeout_s", 1800),
                                 preexec_fn=guard.die_with_parent)
        except subprocess.TimeoutExpired:
            rc = 124
    out = {"ran": rc == 0 and os.path.isfile(out_json), "rc": rc, "seconds": round(time.time() - t0, 1),
           "when_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),      # --only reruns one provider: each has its own time
           "host_loadavg_before": [round(x, 2) for x in load0],
           "host_loadavg_after": [round(x, 2) for x in os.getloadavg()], "gpu_before": state}
    with open(log, errors="replace") as f:
        text = f.read()
    out["adapter_tallies"] = [{"adapter": a, "provider": p, "foreign_launches": int(fl), "origin_launches": int(ol)}
                              for a, p, fl, ol in TALLY.findall(text)]
    if out["ran"]:
        with open(out_json) as f:
            out["cases"] = json.load(f)
    else:
        out["log_tail"] = text.splitlines()[-8:]
    return out


def num(v):
    return float("nan") if v == "nan" else float("inf") if v == "inf" else float("-inf") if v == "-inf" else v


def judge(cfg, provider, run):
    """Applies the config's criteria to one provider's driver output. Returns (pass, rows, problems)."""
    crit = cfg["criteria"]
    wanted = {c["name"]: c for c in provider_cases(cfg, provider)}
    rows, problems = [], []
    got = {c["case"]: c for c in run.get("cases", [])}
    for name, c in wanted.items():
        r = got.get(name)
        if r is None or not r.get("ran"):
            problems.append(f"{name}: did not run" + (f" ({r.get('error')})" if r else ""))
            rows.append({"case": name, "kind": c["kind"], "pass": False, "error": (r or {}).get("error", "no result")})
            continue
        row = dict(r)
        if c["kind"] == "parity":
            level = c.get("require", crit["parity"])
            assert level in PARITY_LEVELS, level
            row["required"] = level
            if level == "bit-identical":
                ok = bool(r.get("bit_identical"))
            elif level == "order-bound":
                ok = bool(r.get("bit_identical")) or num(r.get("err_over_bound", float("inf"))) <= crit["max_err_over_bound"]
            else:
                ok = True                       # structure / report: the values never decide the gate
            if "structure_ok" in r:             # e.g. selection: the set of ids, tie rule, flags
                ok = ok and (bool(r["structure_ok"]) or level == "report")
        else:
            ok = num(r.get("median_us", 0)) > 0
            row["required"] = "ran"
        row["pass"] = ok
        if not ok:
            problems.append(f"{name}: required {row['required']}, got "
                            + ", ".join(f"{k}={r[k]}" for k in ("bit_identical", "mismatched", "err_over_bound",
                                                                 "structure_ok", "max_abs") if k in r))
        rows.append(row)
    if provider.get("adapter"):
        tallies = run.get("adapter_tallies", [])
        foreign = sum(t["foreign_launches"] for t in tallies)
        origin = sum(t["origin_launches"] for t in tallies)
        if crit.get("adapter_must_launch_foreign", True) and foreign == 0:
            problems.append("the adapter never launched the foreign provider")
        allowed = crit.get("adapter_origin_launches_allowed", 0)
        if origin > allowed:
            problems.append(f"the adapter handed {origin} launches back to the origin's kernel (allowed: {allowed})")
    return not problems, rows, problems


def classify(rows):
    """The provider's parity class over the parity cases that decide the gate, as measured."""
    par = [r for r in rows if r.get("kind") == "parity" and r.get("required") in ("bit-identical", "order-bound")
           and "bit_identical" in r]
    if not par:
        return {"class": "not measured"}
    if all(r["bit_identical"] for r in par):
        return {"class": "bit-identical to the CPU reference", "cases": len(par)}
    out = {"class": "differs from the CPU reference", "cases": len(par),
           "bit_identical_cases": sum(1 for r in par if r["bit_identical"]),
           "max_abs": max(num(r.get("max_abs", 0)) for r in par), "max_rel": max(num(r.get("max_rel", 0)) for r in par)}
    if any("err_over_bound" in r for r in par):
        out["max_err_over_bound"] = max(num(r.get("err_over_bound", 0)) for r in par)
    return out


def order_evidence(rows):
    """Which parity cases a CPU emulation of a kernel's documented summation order reproduces bit for bit."""
    par = [r for r in rows if r.get("kind") == "parity" and "order_emulation" in r]
    if not par:
        return None
    by = {}
    for r in par:
        by.setdefault(r["order_emulation"], []).append(r["case"])
    matched = sum(len(v) for k, v in by.items() if k != "none")
    return {"cases": len(par), "reproduced": matched, "by_emulation": by, "order_only": matched == len(par)}


def pairwise(cfg, trees):
    """Providers' dumps against each other, case by case: byte equality, and for fp32 dumps the largest gap."""
    names = [n for n in trees if os.path.isdir(os.path.join(trees[n], "dumps"))]
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            same, differ, worst = 0, [], 0.0
            for c in cfg["case"]:
                if c["kind"] != "parity":
                    continue
                pa, pb = (os.path.join(trees[n], "dumps", c["name"] + ".bin") for n in (a, b))
                if not (os.path.isfile(pa) and os.path.isfile(pb)):
                    continue
                with open(pa, "rb") as f:
                    da = f.read()
                with open(pb, "rb") as f:
                    db = f.read()
                if da == db:
                    same += 1
                    continue
                entry = {"case": c["name"]}
                if c.get("dump", "f32") == "f32" and len(da) == len(db) and len(da) % 4 == 0:
                    fa, fb = array.array("f"), array.array("f")
                    fa.frombytes(da), fb.frombytes(db)
                    gaps = [abs(x - y) for x, y in zip(fa, fb) if x == x and y == y and abs(x) != float("inf") and abs(y) != float("inf")]
                    entry["elements_differing"] = sum(1 for k in range(0, len(da), 4) if da[k:k + 4] != db[k:k + 4])
                    entry["elements"] = len(fa)
                    entry["max_abs_gap"] = max(gaps) if gaps else 0.0
                    worst = max(worst, entry["max_abs_gap"])
                else:
                    entry["bytes_differing"] = sum(1 for x, y in zip(da, db) if x != y) + abs(len(da) - len(db))
                differ.append(entry)
            out.append({"a": a, "b": b, "cases_identical": same, "cases_differing": len(differ),
                        "bit_identical": not differ and same > 0, "max_abs_gap": worst, "differing": differ})
    return out


def summary_md(doc):
    lines = [f"# Interface gate: {doc['interface']}", "",
             f"Host {doc['host']}, {doc['when_utc']}. Config `gate/interface/{doc['name']}.toml` "
             f"(sha256 {doc['config_sha256'][:16]}). Consumer API: `{doc['api']}`.", "",
             f"Overall: **{'PASS' if doc['pass'] else 'FAIL'}**", "",
             "| provider | module that runs | built | parity vs CPU reference | adapter launches (foreign / origin) | gate |",
             "|---|---|---|---|---|---|"]
    for name, p in doc["providers"].items():
        cl = p.get("parity", {})
        desc = cl.get("class", "-")
        if "max_abs" in cl:
            desc += f": max abs {cl['max_abs']:.3g}, max rel {cl['max_rel']:.3g}"
        if "max_err_over_bound" in cl:
            desc += f", worst error {cl['max_err_over_bound']:.3g} of the gate's bound"
        oe = p.get("order_evidence") or order_evidence(p.get("cases", []))
        if oe:
            names = ", ".join(k for k in oe["by_emulation"] if k != "none") or "none"
            desc += (f"; a CPU emulation of the kernel's documented order ({names}) reproduces its bits in "
                     f"{oe['reproduced']} of {oe['cases']} cases")
        t = p.get("run", {}).get("adapter_tallies", [])
        tally = " / ".join(str(sum(x[k] for x in t)) for k in ("foreign_launches", "origin_launches")) if t else "-"
        lines.append(f"| {name} | `{p.get('wraps') or p['module']}` | {'yes' if p['build'].get('built') else 'NO'} | {desc} | "
                     f"{tally} | {'PASS' if p.get('pass') else 'FAIL'} |")
    bench = [c["name"] for c in doc["cases"] if c["kind"] == "bench"]
    if bench:
        lines += ["", "Time per call (median of the repeats, launches queued back to back then one synchronize). A call "
                      "is one launch of the consumer API; adapters may need several kernel launches for it. GB/s is the "
                      "size of the call's operands divided by that time: for operands of a few MB it is not a memory "
                      "bandwidth (they stay in the GPU's caches between launches), for the large shapes it is.", "",
                  "| provider | case | shape | median us | min..max us | GB/s | kernel launches | host load (1 min) |",
                  "|---|---|---|---|---|---|---|---|"]
        shapes = {c["name"]: c.get("shape", "") for c in doc["cases"]}
        for name, p in doc["providers"].items():
            load = p.get("run", {}).get("host_loadavg_before", ["?"])[0]
            for r in p.get("cases", []):
                if r.get("kind") == "bench" and r.get("ran"):
                    lines.append(f"| {name} | {r['case']} | {shapes.get(r['case'], '')} | {num(r['median_us']):.2f} | "
                                 f"{num(r['min_us']):.2f}..{num(r['max_us']):.2f} | {num(r.get('gb_per_s', 0)):.1f} | "
                                 f"{p.get('launches') or r.get('kernel_launches_per_call', '?')} | {load} |")
    if doc.get("pairwise"):
        lines += ["", "Providers against each other (raw outputs of the parity cases):", "",
                  "| a | b | cases identical | cases differing | largest gap |", "|---|---|---|---|---|"]
        for pw in doc["pairwise"]:
            lines.append(f"| {pw['a']} | {pw['b']} | {pw['cases_identical']} | {pw['cases_differing']} | {pw['max_abs_gap']:.3g} |")
    notes = []
    for name, p in doc["providers"].items():
        oe = p.get("order_evidence") or order_evidence(p.get("cases", []))
        if oe and oe["by_emulation"].get("none"):
            notes.append(f"- {name}: no order emulation reproduces " + ", ".join(oe["by_emulation"]["none"])
                         + " (within the bound; the order of that path is not shown)")
        for r in p.get("cases", []):
            if r.get("kind") == "parity" and r.get("required") in ("report", "structure") and not r.get("bit_identical", True):
                notes.append(f"- {name} / {r['case']}: differs from the reference (reported, not a pass criterion): "
                             + ", ".join(f"{k}={r[k]}" for k in ("mismatched", "max_abs", "structure_ok", "detail") if k in r))
        for prob in p.get("problems", []):
            notes.append(f"- {name}: FAIL - {prob}")
    if notes:
        lines += ["", "Notes:", ""] + notes
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--module", help="run the interface gate(s) this module takes part in, for its providers only")
    ap.add_argument("--only", help="comma-separated provider names; the others keep their recorded results")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--summarize", action="store_true", help="rewrite summary.md from the recorded result.json; runs nothing")
    a = ap.parse_args()
    if a.module:
        found = configs_for_module(a.module)
        if not found:
            return None            # the caller falls back to the compile-only module gate
        rc = 0
        for name, providers in found:
            rc |= subprocess.call([sys.executable, __file__, name, "--only", ",".join(providers)])
        return rc
    if not a.name:
        ap.error("give an interface gate name")
    cfg = load_config(a.name)
    res_root = os.path.join(ROOT, "gate", "results", "interface", a.name)
    if a.summarize:
        with open(os.path.join(res_root, "result.json")) as f:
            doc = json.load(f)
        md = summary_md(doc)
        with open(os.path.join(res_root, "summary.md"), "w") as f:
            f.write(md)
        print(md)
        return 0 if doc.get("pass") else 1
    cat = Catalog(ROOT)
    os.makedirs(res_root, exist_ok=True)
    only = a.only.split(",") if a.only else None
    doc = {"name": a.name, "interface": cfg["interface"], "api": cfg["api"], "host": os.uname().nodename,
           "config_sha256": cfg["_sha256"], "criteria": cfg["criteria"], "cases": cfg["case"], "providers": {}}
    prior = os.path.join(res_root, "result.json")
    if only and os.path.isfile(prior):
        with open(prior) as f:
            old = json.load(f)
        if old.get("config_sha256") == cfg["_sha256"]:
            doc["providers"] = {k: v for k, v in old.get("providers", {}).items() if k not in only}
    shutil.copyfile(cfg["_path"], os.path.join(res_root, "config.toml"))
    trees, todo = {}, []
    for p in cfg["provider"]:
        tree = os.path.join(ROOT, "build", "gate-interface", a.name, p["name"])
        trees[p["name"]] = tree
        if only and p["name"] not in only:
            continue
        res = os.path.join(res_root, p["name"])
        os.makedirs(res, exist_ok=True)
        print(f"== build {p['name']}", flush=True)
        entry = {"module": p["module"], "wraps": p.get("wraps"), "adapter": bool(p.get("adapter")), "bind": p["bind"],
                 "cases_pattern": p.get("cases"),
                 "launches": p.get("launches"),
                 "build": build_provider(cat, cfg, p, tree, res)}
        print(f"   {'built' if entry['build'].get('built') else 'BUILD FAILED: ' + str(entry['build'].get('errors') or entry['build'].get('log_tail'))}", flush=True)
        doc["providers"][p["name"]] = entry
        todo.append((p, tree, res))
    if not a.build_only:
        lease_log = os.path.join(res_root, "lease.txt")
        open(lease_log, "w").close()
        ok, state = guard.gpu_free(cfg)
        if not ok:
            sys.exit(f"the GPU is not free; nothing was run: {state}")
        lease = guard.Lease(cfg, lease_log)
        try:
            doc["lease"] = lease.acquire(f"strixflerp interface gate {a.name}: kernel-level parity and bench, no model loaded")
            for p, tree, res in todo:
                entry = doc["providers"][p["name"]]
                if not entry["build"].get("built"):
                    entry.update({"pass": False, "problems": ["did not build"], "cases": []})
                    continue
                print(f"== run {p['name']}", flush=True)
                run = run_provider(cfg, p, tree, res)
                entry["run"] = {k: v for k, v in run.items() if k != "cases"}
                ok, rows, problems = judge(cfg, p, run)
                entry.update({"pass": ok and run.get("ran", False), "cases": rows, "problems": problems,
                              "parity": classify(rows), "order_evidence": order_evidence(rows)})
                print(f"   {'PASS' if entry['pass'] else 'FAIL: ' + '; '.join(problems[:4])}", flush=True)
        finally:
            lease.release()
            time.sleep(2)
            doc["gpu_after"] = guard.gpu_free(cfg)[1]
        doc["pairwise"] = pairwise(cfg, trees)
    doc["when_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    order = [p["name"] for p in cfg["provider"] if p["name"] in doc["providers"]]
    doc["providers"] = {n: doc["providers"][n] for n in order}
    complete = len(order) == len(cfg["provider"])
    doc["pass"] = complete and all(p.get("pass") for p in doc["providers"].values())
    doc["complete"] = complete
    with open(prior, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    if not a.build_only:
        md = summary_md(doc)
        with open(os.path.join(res_root, "summary.md"), "w") as f:
            f.write(md)
        print(md)
    return 0 if doc["pass"] or a.build_only else 1


if __name__ == "__main__":
    rc = main()
    sys.exit(3 if rc is None else rc)
