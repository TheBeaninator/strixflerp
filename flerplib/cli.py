"""The `flerp` command line (SPEC.md section 8)."""
import argparse
import fnmatch
import json
import os
import subprocess
import sys

from . import assemble, assemble_cmake, assemble_make, assemble_zig, check, origin
from .catalog import Catalog, licence_matches, overlap_class, split_tag
from .resolve import Assembly
from .champion import cmd_champion


def cmd_list(cat, a):
    rows = []
    for m in cat.modules:
        if a.kind and m.kind != a.kind:
            continue
        if a.origin and m.dir_origin != a.origin:
            continue
        if a.licence and not licence_matches(m.spdx, a.licence):
            continue
        if a.provides and not any(fnmatch.fnmatchcase(i, a.provides) for i in m.interfaces):
            continue
        rows.append((m.id, m.spdx, str(m.gate.get("status", "?")), m.summary))
    w = [max((len(r[i]) for r in rows), default=0) for i in range(3)]
    for r in rows:
        print(f"{r[0].ljust(w[0])}  {r[1].ljust(w[1])}  {r[2].ljust(w[2])}  {r[3]}")
    print(f"{len(rows)} module(s)", file=sys.stderr)
    return 0


def cmd_check(cat, a):
    rep = check.run(cat, a.origin)
    for w in rep.warnings if a.warnings else []:
        print("warning:", w)
    for e in rep.errors:
        print("error:", e)
    scope = f"origin {a.origin}" if a.origin else "catalogue"
    n = len([m for m in cat.modules if not a.origin or m.dir_origin == a.origin])
    print(f"flerp check ({scope}): {n} modules, {len(cat.recipes)} recipe(s), "
          f"{len(rep.errors)} error(s), {len(rep.warnings)} warning(s)")
    return 1 if rep.errors else 0


def cmd_show(cat, a):
    m = cat.by_id.get(a.module)
    if m is None:
        print(f"no module {a.module}", file=sys.stderr)
        return 1
    prov = m.data.get("provenance", {})
    print(f"{m.id}  [{m.kind}]  {m.spdx}")
    print(f"  {m.summary}")
    print(f"  dir       {cat.rel(m.dir)}")
    print(f"  origin    {prov.get('origin')} {prov.get('commit')} ({prov.get('branch', '-')}), "
          f"authors {', '.join(prov.get('authors', []))}, modified {str(prov.get('modified')).lower()}")
    print(f"  build     {m.lang}, toolchain {m.build.get('toolchain', '-')}, arch {m.build.get('arch', '-')}")
    print(f"  gate      status {m.gate.get('status')}, parity {m.gate.get('parity')}, bench {m.gate.get('bench')}"
          + (f", evidence {m.gate['evidence']}" if m.gate.get("evidence") else ""))
    print(f"  files     {len(m.files())} ({len(m.sources)} compiled)")
    for p in m.files():
        print(f"    {'*' if 'src/' + p in m.sources else ' '} {p}")
    print("  provides")
    for p in m.provides:
        print(f"    {p.get('interface')}")
        for k in ("api", "numeric", "regime"):
            print(f"      {k.ljust(8)}{p.get(k)}")
    print("  requires")
    for r in m.requires:
        if "module" in r:
            print(f"    module    {r['module']}")
        else:
            provs = ", ".join(p.id for p in cat.providers.get(r.get("interface"), [])) or "NO PROVIDER"
            print(f"    interface {r.get('interface')}  <- {provs}")
    return 0


def foreign_of(cat, m):
    """The modules an adapter launches: what it requires by id from an origin other than its own and other than
    the origin of the module whose API it re-provides."""
    target = cat.by_id.get(m.overrides[0])
    skip = {m.dir_origin, target.dir_origin if target else None}
    return [r["module"] for r in m.requires if m.adapter and r.get("module") in cat.by_id
            and cat.by_id[r["module"]].dir_origin not in skip]


def cmd_interfaces(cat, a):
    """Interfaces by overlap class (the name without its tag): providers of one class do the same job on the
    same operands; providers under one full name are interchangeable. An adapter is shown with the foreign
    module it launches."""
    cons = cat.consumers()
    classes = {}
    for i in set(cat.providers) | set(cons):
        classes.setdefault(overlap_class(i), set()).add(i)
    single = unused = shared = cross = shown = 0
    for cls in sorted(classes):
        if a.glob and not fnmatch.fnmatchcase(cls, a.glob):
            continue
        shown += 1
        names = sorted(classes[cls], key=lambda n: (split_tag(n)[1] is not None, n))
        provs = [(n, m) for n in names for m in cat.providers.get(n, [])]
        users = [(n, u) for n in names for u in cons.get(n, [])]
        origins = set()
        for _, m in provs:
            origins.update(cat.by_id[w].dir_origin for w in foreign_of(cat, m))
            if not m.adapter:
                origins.add(m.dir_origin)
        flags = []
        if len(provs) == 1:
            flags.append("single-provider")
            single += 1
        elif len(provs) > 1:
            shared += 1
            flags.append(f"{len(provs)} providers from {len(origins)} origin(s)")
            cross += len(origins) > 1
        if not provs:
            flags.append("NO-PROVIDER")
        if not users:
            flags.append("unused")
            unused += 1
        print(f"{cls}  [{', '.join(flags)}]")
        for n, m in provs:
            tag = split_tag(n)[1]
            note = f"  as .{tag}" if tag else ""
            if m.adapter:
                note += f"  (adapter for {m.overrides[0]} over {', '.join(foreign_of(cat, m)) or '-'})"
            st = m.gate.get("status", "?")
            print(f"    provided by  {m.id}{note}  [{st}]")
        if not provs:
            print("    provided by  -")
        req = ", ".join(sorted({u.id + (f" (as .{split_tag(n)[1]})" if split_tag(n)[1] else "") for n, u in users}))
        print(f"    required by  {req or '-'}")
    print(f"{shown} interface classes: {shared} with two or more providers ({cross} across origins), "
          f"{single} single-provider, {unused} unused", file=sys.stderr)
    return 0


def _recipe(cat, name):
    r = cat.recipe(name)
    if r is None:
        print(f"no recipe {name} (have: {', '.join(x.name for x in cat.recipes) or 'none'})", file=sys.stderr)
    return r


def cmd_licence_report(cat, a):
    r = _recipe(cat, a.recipe)
    if r is None:
        return 1
    asm = Assembly(cat, r)
    for e in asm.errors:
        print("error:", e)
    groups = {}
    for mid in asm.modules:
        groups.setdefault(cat.by_id[mid].spdx, []).append(mid)
    for spdx in sorted(groups):
        print(f"{spdx}  ({len(groups[spdx])} modules)")
        for mid in sorted(groups[spdx]):
            print(f"    {mid}")
    for name, (_, _, ids) in asm.binaries.items():
        print(f"{name}: " + ", ".join(sorted({cat.by_id[i].spdx for i in ids})))
    return 1 if asm.errors else 0


def cmd_assemble(cat, a):
    r = _recipe(cat, a.recipe)
    if r is None:
        return 1
    out = a.output or os.path.join(cat.root, "build", r.name)
    asm, errors = assemble.generate(cat, r, out, a.licence)
    for w in asm.warnings:
        print("warning:", w)
    for e in errors:
        print("error:", e)
    if errors:
        return 1
    compiled = sum(1 for i in asm.modules if cat.by_id[i].sources)
    print(f"assembled {r.name}: {len(asm.modules)} modules ({compiled} with sources), "
          f"{len(asm.binaries)} binaries -> {os.path.relpath(out)}")
    rel = os.path.relpath(out)
    if assemble_zig.is_zig(r):       # a source tree and the origin's own build graph, run by build.sh
        print(f"build: {rel}/build.sh [-jN]   (zig and hipcc from toolchains.toml; FLERP_ZIG and FLERP_ROCM_ROOT override)")
    elif assemble_make.is_make(r):     # a source tree and the origin's own Makefile, run by build.sh
        print(f"build: {rel}/build.sh [-jN]   (ROCm from toolchains.toml; FLERP_ROCM_ROOT overrides)")
    elif assemble_cmake.is_cmake(r):   # a source tree and the origin's own CMake files, run by build.sh
        print(f"build: {rel}/build.sh [-jN]   (ROCm from toolchains.toml; FLERP_ROCM_ROOT overrides)")
    else:
        print(f"build: cmake -G Ninja -S {rel} -B {rel}/ninja && ninja -C {rel}/ninja")
    return 0


def cmd_diff_origin(cat, a):
    checkout = os.path.abspath(a.checkout)
    if not os.path.isdir(checkout):
        print(f"{checkout} is not a directory", file=sys.stderr)
        return 1
    if a.origin:
        res = origin.diff_origin(cat, a.origin, checkout)
    else:
        m = cat.by_id.get(a.module or "")
        if m is None:
            print(f"no module {a.module}; give a module id or --origin O", file=sys.stderr)
            return 1
        problems = [(m.id, p, why) for p, why in origin.diff_module(m, checkout)]
        n = len(m.files())
        res = {"module": m.id, "checkout": checkout, "modules": 1, "files": n,
               "identical": n - len(problems), "problems": problems}
    for mid, path, why in res["problems"]:
        print(f"DIFF {mid} {path}: {why}")
    what = f"origin {a.origin}" if a.origin else res["module"]
    print(f"diff-origin {what} vs {checkout}: {res['modules']} module(s), {res['identical']}/{res['files']} files "
          f"byte-identical, {len(res['problems'])} problem(s)"
          + (f", {res['unharvested']} origin files listed as unharvested" if a.origin else ""))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
            f.write("\n")
    return 1 if res["problems"] else 0


def cmd_gate(cat, a):
    if a.what == "recipe":
        r = _recipe(cat, a.name)
        if r is None:
            return 1
        runner = r.gate.get("runner")
        if not runner:
            print(f"recipe {r.name} has no [gate] runner", file=sys.stderr)
            return 1
        return subprocess.call([sys.executable, os.path.join(cat.root, runner), "--recipe", r.name] + a.rest)
    if a.what == "interface":
        return subprocess.call([sys.executable, os.path.join(cat.root, "gate", "interface.py"), a.name] + a.rest)
    # a module: the interface gates it takes part in (parity and bench on the GPU); none -> the compile-only gate
    rc = subprocess.call([sys.executable, os.path.join(cat.root, "gate", "interface.py"), "--module", a.name] + a.rest)
    if rc != 3:
        return rc
    return subprocess.call([sys.executable, os.path.join(cat.root, "gate", "module_builds.py"), a.name] + a.rest)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="flerp", description="strixflerp catalogue tooling (see SPEC.md)")
    ap.add_argument("--root", help="repository root (default: where this script lives)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="modules, filtered")
    p.add_argument("--kind")
    p.add_argument("--origin")
    p.add_argument("--licence", metavar="EXPR", help="comma-separated SPDX id globs; '!' denies")
    p.add_argument("--provides", metavar="GLOB")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("check", help="validate every manifest, origin file and recipe")
    p.add_argument("--origin", help="only this origin's modules and the recipes that use them")
    p.add_argument("-w", "--warnings", action="store_true", help="print the warnings, not only their count")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("show", help="one module in full")
    p.add_argument("module")
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("interfaces", help="every interface class with its providers and consumers")
    p.add_argument("glob", nargs="?", help="only classes matching this pattern, e.g. 'residual.*'")
    p.set_defaults(fn=cmd_interfaces)

    p = sub.add_parser("licence-report", help="a recipe's modules by SPDX id")
    p.add_argument("recipe")
    p.set_defaults(fn=cmd_licence_report)

    p = sub.add_parser("assemble", help="generate a recipe's CMake + Ninja build tree")
    p.add_argument("recipe")
    p.add_argument("--licence", metavar="EXPR")
    p.add_argument("-o", "--output")
    p.set_defaults(fn=cmd_assemble)

    p = sub.add_parser("diff-origin", help="prove vendored sources are verbatim")
    p.add_argument("module", nargs="?")
    p.add_argument("checkout")
    p.add_argument("--origin", help="every module of this origin, plus coverage of the checkout")
    p.add_argument("--json", metavar="FILE")
    p.set_defaults(fn=cmd_diff_origin)

    p = sub.add_parser("gate", help="run a gate on gfx1151 and write gate/results/")
    p.add_argument("what", choices=("module", "interface", "recipe"))
    p.add_argument("name")
    p.add_argument("rest", nargs=argparse.REMAINDER, help="passed to the gate runner")
    p.set_defaults(fn=cmd_gate)

    p = sub.add_parser("champion", help="the champion ledger per area: show, propose CONFIG AREA RESULTS_DIR (see docs/CHAMPIONS.md)")
    p.add_argument("action", choices=("show", "propose"))
    p.add_argument("args", nargs="*")
    p.add_argument("--case", default="", help="propose: the area's case (ttft: '21 tok', '1808 tok')")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_champion)

    a = ap.parse_args(argv)
    if a.cmd == "diff-origin" and a.origin and a.module and not os.path.isdir(a.checkout):
        ap.error("with --origin give only the checkout")
    return a.fn(Catalog(a.root), a)
