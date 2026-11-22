#!/usr/bin/env python3
"""Per-kernel register, scratch and occupancy table for gfx1151 device code, and a diff between two toolchains.
Standard library only, CPU only: it compiles to assembly and reads it; nothing runs on the GPU.

    tools/isa-scan.py compile --toolchain T --out DIR SOURCE...           # gate modules and recipe builds
    tools/isa-scan.py table DIR/*.s > T.tsv                              # one row per kernel
    tools/isa-scan.py diff OLD.tsv NEW.tsv                               # rows whose scratch or occupancy changed

SOURCE is one of
    module:<module-id>      a [gate].compile = "standalone" module, compiled the way gate/compile/run.py does
    origin:<origin>         every standalone module of that origin
    ccdb:<compile_commands.json>[:REGEX]   the .hip entries of a recipe build (REGEX filters the file path)
    cmd:<name>:<command...> one literal compiler command (the -o / -c / --genco parts are replaced)

`compile` rewrites each command for toolchain T (toolchains.toml rocm_root replaces the command's own root), drops
-c / --genco / -o / -flto, and adds `--offload-device-only --no-gpu-bundle-output -S -o DIR/<unit>.s`. Extra
compiler flags go after `--extra` (e.g. --extra -mllvm -amdgpu-schedule-metric-bias=100). Use `-j` for parallel
units and --max-load to wait for a quiet host; prefix the call with `ionice -c3 nice -n 19` on control-node.
control-node's /opt/rocm-7.2.4 has no hipblas/hipblaslt/hipcub/rocprim headers (gufo needs them): point --extra -I at
a directory of symlinks to just those four from a TheRock include/, never at the whole TheRock include/.
FLERP_ROOT names the strixflerp checkout to scan when this file is run from somewhere else.

The table reads the .amdgpu_metadata block (vgpr/agpr/sgpr counts, private segment = scratch bytes per lane,
spill counts, LDS, dynamic stack) and the compiler's `; Occupancy:` comment (waves per SIMD, which includes the
LDS and amdgpu_waves_per_eu limits). It also counts scratch_load/scratch_store instructions per kernel, and the ones
inside an innermost loop (a backward branch's range) that holds v_wmma: those are the spills that cost the most.
"""
import argparse, concurrent.futures as cf, json, os, re, shlex, subprocess, sys, time, tomllib

ROOT = os.environ.get("FLERP_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the checkout to scan
COLS = ["unit", "kernel", "vgpr", "agpr", "sgpr", "scratch", "vspill", "sspill", "lds", "occ", "dynstack",
        "scratch_ld", "scratch_st", "loop_scratch_ld", "loop_scratch_st", "wmma", "code_bytes"]


def load(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------- command collection

def gate_jobs(sel, tcs):
    """(unit, argv, cwd, gen_text) for standalone gate modules, as gate/compile/run.py builds them."""
    sys.path.insert(0, os.path.join(ROOT, "gate", "compile"))
    import run as gate  # noqa: E402
    mods = gate.modules()
    out = []
    for mid, m in sorted(mods.items()):
        b = m["build"]
        if b.get("lang") != "hip-c++" or m.get("gate", {}).get("compile") != "standalone":
            continue
        origin = m["provenance"]["origin"]
        if sel[0] == "module" and mid != sel[1] or sel[0] == "origin" and origin != sel[1]:
            continue
        oc = load(os.path.join(ROOT, "catalog", origin, "ORIGIN.toml"))["compile"]
        root = tcs[b["toolchain"]]["rocm_root"]
        driver = oc["driver"].replace("<rocm>", root)
        incs = gate.include_dirs(m, mods) + [os.path.join(root, i) for i in oc.get("system_includes", [])]
        rf = oc.get("rocm_flags", [oc["rocm_flag"]] if "rocm_flag" in oc else [])
        base = [driver, oc["arch_flag"]] + [x for f in rf for x in f.replace("<rocm>", root).split()]
        flags = [f.replace("<rocm>", root) for f in b.get("options", oc.get("flags", []))]
        inc = [x for i in incs for x in ("-I", i)]
        if b.get("units"):
            for u in json.load(open(os.path.join(m["_dir"], b["units"])))["units"]:
                text = "".join(p["text"] if "text" in p else
                               open(os.path.join(ROOT, "catalog", origin, p["path"]), errors="replace").read()
                               for p in u["parts"])
                out.append((f"{mid}/{u['name']}", base + flags + u.get("flags", []) + inc, None, text))
        else:
            sf = b.get("source_options", {})
            for s in b.get("sources", []):
                out.append((f"{mid}/{os.path.basename(s)}", base + flags + sf.get(s, []) + inc +
                            oc.get("source_prefix", []) + [os.path.join(m["_dir"], s)], None, None))
    return out


def ccdb_jobs(path, rx):
    out = []
    for e in json.load(open(path)):
        if not e["file"].endswith(".hip") or (rx and not re.search(rx, e["file"])):
            continue
        argv = e.get("arguments") or shlex.split(e["command"])
        rel = os.path.relpath(e["file"], ROOT) if e["file"].startswith(ROOT) else e["file"]
        out.append((rel.replace("catalog/", "").replace("/", "~"), argv, e["directory"], None))
    return out


def tool_root(argv0):
    for tail in ("/lib/llvm/bin/", "/llvm/bin/", "/bin/"):
        if tail in argv0:
            return argv0.split(tail)[0]
    return None


def rewrite(argv, root, out_s, extra):
    old = tool_root(argv[0])
    a = [x.replace(old, root) if old else x for x in argv]
    res, skip = [], False
    for x in a:
        if skip:
            skip = False
            continue
        if x == "-o":
            skip = True
            continue
        if x in ("-c", "--genco", "-flto", "--offload-device-only", "--cuda-device-only", "--no-gpu-bundle-output",
                 "-save-temps") or x.startswith("-o") and len(x) > 2 and not x.startswith("-ob"):
            continue
        if x.startswith("--rocm-device-lib-path=") and not os.path.isdir(x.split("=", 1)[1]):
            continue  # an origin's device-lib path that this toolchain lays out elsewhere: let the driver find it
        res.append(x)
    return res + ["--offload-device-only", "--no-gpu-bundle-output", "-S"] + extra + ["-o", out_s]


def run_job(job):
    unit, argv, cwd, out_s, gen, max_load = job
    while max_load and os.getloadavg()[0] > max_load:  # control-node is the pod's hub: wait for a quiet moment
        time.sleep(30)
    t0 = time.time()
    if gen is not None:
        src = out_s[:-2] + ".hip"
        with open(src, "w") as f:
            f.write(gen)
        argv = argv + [src]
    p = subprocess.run(argv, cwd=cwd or os.path.dirname(out_s), capture_output=True, text=True)
    if gen is not None:
        os.remove(src)
    ok = p.returncode == 0 and os.path.isfile(out_s)
    with open(out_s[:-2] + ".log", "w") as f:
        f.write(" ".join(shlex.quote(x) for x in argv) + f"\n# rc={p.returncode} {time.time() - t0:.1f}s\n" + p.stderr[-20000:])
    return unit, ok, round(time.time() - t0, 1)


def cmd_compile(a):
    tcs = load(os.path.join(ROOT, "toolchains.toml"))
    root = tcs[a.toolchain]["rocm_root"]
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for s in a.sources:
        kind, _, rest = s.partition(":")
        if kind in ("module", "origin"):
            jobs += gate_jobs((kind, rest), tcs)
        elif kind == "ccdb":
            path, _, rx = rest.partition(":")
            jobs += ccdb_jobs(path, rx)
        elif kind == "cmd":
            name, _, c = rest.partition(":")
            jobs.append((name, shlex.split(c), os.getcwd(), None))
        else:
            sys.exit(f"unknown source {s}")
    work = []
    for unit, argv, cwd, gen in jobs:
        out_s = os.path.abspath(os.path.join(a.out, unit.replace("/", "~") + ".s"))
        work.append((unit, rewrite(argv, root, out_s, a.extra), cwd, out_s, gen, a.max_load))
    failed = 0
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for unit, ok, sec in ex.map(run_job, work):
            print(f"{'ok  ' if ok else 'FAIL'} {sec:7.1f}s {unit}", flush=True)
            failed += 0 if ok else 1
    if failed or not work:
        print(f"isa-scan: {failed} of {len(work)} unit(s) failed" if work else "isa-scan: no units matched", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------- assembly parsing

def parse_s(path):
    """Per function: its instruction mnemonics, back-edge ranges (indices), and the compiler's occupancy comment."""
    fn, fns = None, {}
    for line in open(path, errors="replace"):
        m = re.match(r"^\s*\.type\s+([^,]+),@function", line)
        if m:
            fn = m.group(1)
            fns[fn] = {"ins": [], "occ": "", "code": "", "labels": {}, "branches": []}
            continue
        if fn is None:
            continue
        f = fns[fn]
        m = re.match(r"^; (Occupancy|codeLenInByte)\s*[:=]\s*(\d+)", line)
        if m:
            f["occ" if m.group(1) == "Occupancy" else "code"] = m.group(2)
            continue
        m = re.match(r"^(\.LBB\w+):", line)
        if m:
            f["labels"][m.group(1)] = len(f["ins"])
            continue
        s = line.strip()
        if not s or s[0] in ".;" or s.endswith(":"):
            continue
        op = s.split()[0]
        f["ins"].append(op)
        if op.startswith("s_cbranch") or op == "s_branch":
            t = s.split()[-1]
            if t in f["labels"]:  # target already seen: a backward branch closes a loop
                f["branches"].append((f["labels"][t], len(f["ins"]) - 1))
    return fns


def parse_meta(path):
    """amdhsa.kernels entries as dicts of their top-level scalar keys."""
    txt = open(path, errors="replace").read()
    a = txt.find("amdhsa.kernels:")
    if a < 0:
        return []
    b = txt.find(".end_amdgpu_metadata", a)
    body = txt[a:b].splitlines()[1:]
    out, cur, kind = [], None, None
    for line in body:
        if re.match(r"^amdhsa\.", line):
            break
        m = re.match(r"^(\s*)- \.(\w+):\s*(.*)$", line)
        if m and kind is None:
            kind = len(m.group(1))
        if m and len(m.group(1)) == kind:
            cur = {}
            out.append(cur)
            cur[m.group(2)] = m.group(3).strip().strip("'\"")
            continue
        m = re.match(r"^(\s*)\.(\w+):\s*(.*)$", line)
        if m and cur is not None and kind is not None and len(m.group(1)) == kind + 2:
            cur[m.group(2)] = m.group(3).strip().strip("'\"")
    return out


def rows_for(path):
    unit, fns = os.path.basename(path)[:-2], parse_s(path)
    rows = []
    for k in parse_meta(path):
        name = k.get("name", "")
        f = fns.get(name) or fns.get(name.removesuffix(".kd"), {"ins": [], "occ": "", "code": "", "branches": []})
        ins = f["ins"]
        ld = sum(1 for o in ins if o.startswith("scratch_load"))
        st = sum(1 for o in ins if o.startswith("scratch_store"))
        # Innermost WMMA loops: back-edge ranges holding a v_wmma that contain no smaller such range (the K loop,
        # not the persistent item loop around it).
        wl = [(t, br) for t, br in f.get("branches", []) if any(o.startswith("v_wmma") for o in ins[t:br + 1])]
        inner = [(t, br) for t, br in wl if not any((t2, b2) != (t, br) and t <= t2 and b2 <= br for t2, b2 in wl)]
        hot = set()
        for t, br in inner:
            hot.update(j for j in range(t, br + 1) if ins[j].startswith("scratch_"))
        rows.append({"unit": unit, "kernel": name, "vgpr": k.get("vgpr_count", ""), "agpr": k.get("agpr_count", ""),
                     "sgpr": k.get("sgpr_count", ""), "scratch": k.get("private_segment_fixed_size", ""),
                     "vspill": k.get("vgpr_spill_count", ""), "sspill": k.get("sgpr_spill_count", ""),
                     "lds": k.get("group_segment_fixed_size", ""), "occ": f["occ"],
                     "dynstack": k.get("uses_dynamic_stack", ""), "scratch_ld": ld, "scratch_st": st,
                     "loop_scratch_ld": sum(1 for j in hot if ins[j].startswith("scratch_load")),
                     "loop_scratch_st": sum(1 for j in hot if ins[j].startswith("scratch_store")),
                     "wmma": sum(1 for o in ins if o.startswith("v_wmma")), "code_bytes": f["code"]})
    return rows


def cmd_table(a):
    print("\t".join(COLS))
    for p in a.files:
        for r in rows_for(p):
            print("\t".join(str(r[c]) for c in COLS))


def read_tsv(p, by_kernel=False):
    lines = open(p).read().splitlines()
    h = lines[0].split("\t")
    rows = (dict(zip(h, l.split("\t"))) for l in lines[1:])
    return {("*" if by_kernel else r["unit"], r["kernel"]): r for r in rows}


def num(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def cmd_diff(a):
    old, new = read_tsv(a.old, a.by_kernel), read_tsv(a.new, a.by_kernel)
    keys = sorted(set(old) & set(new))
    print(f"# kernels: old {len(old)}, new {len(new)}, both {len(keys)}")
    print("\t".join(["unit", "kernel", "what", "vgpr", "scratch", "vspill", "occ", "loop_scratch"]))
    for k in keys:
        o, n = old[k], new[k]
        what = []
        if num(n["scratch"]) > num(o["scratch"]):
            what.append("scratch+")
        if num(n["scratch"]) < num(o["scratch"]):
            what.append("scratch-")
        if num(n["occ"]) < num(o["occ"]):
            what.append("occ-")
        if num(n["occ"]) > num(o["occ"]):
            what.append("occ+")
        if n.get("dynstack") == "true" and o.get("dynstack") != "true":
            what.append("dynstack+")
        if not what and not a.all:
            continue
        ls = lambda r: f"{r.get('loop_scratch_ld', '')}/{r.get('loop_scratch_st', '')}"
        print("\t".join([k[0], k[1], ",".join(what) or "=", f"{o['vgpr']}->{n['vgpr']}", f"{o['scratch']}->{n['scratch']}",
                         f"{o['vspill']}->{n['vspill']}", f"{o['occ']}->{n['occ']}", f"{ls(o)}->{ls(n)}"]))
    for k in sorted(set(old) ^ set(new)):
        print("\t".join([k[0], k[1], "only-old" if k in old else "only-new"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("compile")
    c.add_argument("--toolchain", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("-j", "--jobs", type=int, default=1)
    c.add_argument("--max-load", type=float, default=0, help="start a unit only while the 1-min load average is at most this")
    c.add_argument("sources", nargs="+")
    c.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    t = sp.add_parser("table")
    t.add_argument("files", nargs="+")
    d = sp.add_parser("diff")
    d.add_argument("old")
    d.add_argument("new")
    d.add_argument("--all", action="store_true", help="also print unchanged kernels")
    d.add_argument("--by-kernel", action="store_true", help="match on the kernel symbol only (units named differently)")
    a = ap.parse_args()
    {"compile": cmd_compile, "table": cmd_table, "diff": cmd_diff}[a.cmd](a)


if __name__ == "__main__":
    main()
