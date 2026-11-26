#!/usr/bin/env python3
"""Make a recipe's deploy directory from its build, laid out the way the pod's restore-tfp.sh `stage` and
deploy_rank0.sh lay out a StrixFold deploy, and compare it with a served one.

    tools/flerp-stage.py <recipe> <out-dir> [--compare <reference stage>] [--report <file>]

Reads the recipe's [deploy.stage] table and its build under build/<recipe>/ (`flerp assemble <recipe>` then
build/<recipe>/build.sh; the tree must still be what assemble wrote: assemble_zig.py verify-tree):

    [deploy.stage]
    commit     = "<40 hex>"              # the fork commit the catalogue's tree is (DEPLOY.json commit / bin_commit)
    reference  = "/abs/stage"            # the served stage it is meant to equal (default of --compare)
    collation  = "en_US.UTF-8"           # the locale `sort` ran under when the reference's SHA256SUMS was written
    [deploy.stage.copy]                  # stage path -> tree:<path under build/<recipe>/tree> | input:<name>
    [deploy.stage.inputs.<name>]         # files the build does not make (a Python wheel, say)
    where      = "/abs/dir"              # on the build host
    sha256sums = "<repo path>"           # the record: every file of the directory, relative to it; checked first

Then, as the reference: SHA256SUMS over every regular file except files named SHA256SUMS or DEPLOY.json (the
`find . -type f ! -name SHA256SUMS ! -name DEPLOY.json | sort | xargs sha256sum` of restore-tfp.sh, under
`collation`), DEPLOY.json (schema tf-port-deploy/1, plus the recipe and the strixflerp commit), everything
read-only. <out-dir> must not exist: a stage is never rewritten in place. --compare lists every path whose bytes
differ or that only one side has, the SHA256SUMS-named files included, and names a binary whose only difference is
Zig's per-thread __anon_<n> symbol names (symbol_names_only). Exit 0: STAGE-EQUAL; 3: STAGE-EQUAL-BUT-SYMBOL-NAMES (every
differing file is such a binary); 1: STAGE-DIFFERS.
It copies no weights and touches nothing outside <out-dir> and its parent's temporary directory.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import socket
import stat
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from flerplib import assemble_zig  # noqa: E402
from flerplib.catalog import Catalog  # noqa: E402


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_sums(path):
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                digest, name = line.rstrip("\n").split("  ", 1)
                out[name] = digest
    return out


def files_of(d):
    """{relative path: sha256} of every regular file under d (symlinks are not files, as for find -type f)."""
    out = {}
    for base, dirs, names in os.walk(d):
        for n in names:
            p = os.path.join(base, n)
            if not os.path.islink(p) and os.path.isfile(p):
                out[os.path.relpath(p, d)] = sha256(p)
    return out


def links_of(d):
    out = {}
    for base, dirs, names in os.walk(d):
        for n in dirs + names:
            p = os.path.join(base, n)
            if os.path.islink(p):
                out[os.path.relpath(p, d)] = os.readlink(p)
    return out


def recipe_stage(cat, name):
    r = next((r for r in cat.recipes if r.name == name), None)
    if r is None:
        sys.exit(f"flerp-stage: no recipe {name}")
    st = r.data.get("deploy", {}).get("stage")
    if not isinstance(st, dict) or not isinstance(st.get("copy"), dict) or not st["copy"]:
        sys.exit(f"flerp-stage: recipe {name} has no [deploy.stage.copy]")
    return r, st


def make(cat, r, st, out):
    build = os.path.join(ROOT, "build", r.name)
    if assemble_zig.verify_tree(build) != 0:
        sys.exit(f"flerp-stage: {build}/tree is not the assembly; run flerp assemble {r.name} and its build.sh")
    if not os.path.isfile(os.path.join(build, "built.json")):
        sys.exit(f"flerp-stage: {build}/built.json missing: build.sh has not finished")
    built = json.load(open(os.path.join(build, "built.json")))
    zout = os.path.join(build, "tree", "zig-out")
    if files_of(zout) != {k: v["sha256"] for k, v in built.items()}:
        sys.exit(f"flerp-stage: {zout} is not what built.json recorded")
    out = os.path.abspath(out)
    if os.path.exists(out):
        sys.exit(f"flerp-stage: {out} exists; a stage is never rewritten in place")
    tmp = tempfile.mkdtemp(prefix=".flerp-stage.", dir=os.path.dirname(out))
    os.chmod(tmp, 0o755)
    try:
        for dest, src in sorted(st["copy"].items()):
            kind, _, rel = src.partition(":")
            if kind == "tree":
                s = os.path.join(build, "tree", rel)
            elif kind == "input":
                inp = st.get("inputs", {}).get(rel)
                if not isinstance(inp, dict):
                    sys.exit(f"flerp-stage: [deploy.stage.inputs.{rel}] missing")
                s = inp["where"]
                want = read_sums(os.path.join(ROOT, inp["sha256sums"]))
                have = files_of(s)
                if have != want:
                    bad = sorted(set(have) ^ set(want)) + sorted(k for k in set(have) & set(want) if have[k] != want[k])
                    sys.exit(f"flerp-stage: {s} differs from {inp['sha256sums']}: {bad[:10]}")
            else:
                sys.exit(f"flerp-stage: [deploy.stage.copy] {dest} = {src!r}: tree:<path> or input:<name>")
            if not os.path.isdir(s):
                sys.exit(f"flerp-stage: {s} is not a directory")
            shutil.copytree(s, os.path.join(tmp, dest), symlinks=True)
        names = sorted(p for p in files_of(tmp) if os.path.basename(p) not in ("SHA256SUMS", "DEPLOY.json"))
        coll = st.get("collation", "C")
        order = subprocess.run(["sort"], input="".join(n + "\n" for n in names), capture_output=True, text=True,
                               check=True, env=dict(os.environ, LC_ALL=coll)).stdout.split("\n")[:-1]
        if sorted(order) != names:
            sys.exit("flerp-stage: sort lost names (a newline in a path?)")
        with open(os.path.join(tmp, "SHA256SUMS"), "w") as f:
            f.writelines(f"{sha256(os.path.join(tmp, n))}  {n}\n" for n in order)
        files = {n: sha256(os.path.join(tmp, n)) for n in order}
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        deploy = {"schema": "tf-port-deploy/1", "commit": st["commit"], "tree": None, "clean": not dirty,
                  "host": socket.gethostname(), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "target": f"flerp-stage:~/tfp/deploy/{os.path.basename(out)}", "bin_commit": st["commit"],
                  "flerp": {"recipe": r.name, "strixflerp_commit": head, "clean": not dirty,
                            "assembly_sha256": sha256(os.path.join(build, "assembly.json"))},
                  "files": files}
        with open(os.path.join(tmp, "DEPLOY.json"), "w") as f:
            json.dump(deploy, f, indent=1)
        for base, dirs, names_ in os.walk(tmp, topdown=False):
            for n in names_ + dirs:
                p = os.path.join(base, n)
                if not os.path.islink(p):
                    os.chmod(p, stat.S_IMODE(os.lstat(p).st_mode) & ~0o222)
        os.rename(tmp, out)
        os.chmod(out, 0o555)
    except BaseException:
        subprocess.run(["chmod", "-R", "u+w", tmp])
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print(f"flerp-stage: {out}: {len(files)} files in SHA256SUMS, recipe {r.name}, strixflerp {head[:12]}"
          + (" (dirty)" if dirty else ""))
    return out


ANON = re.compile(rb"__anon_(\d+)")


def elf_sections(data):
    """{name: (offset, size)} of an ELF64 little-endian file's sections, or None for anything else."""
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        return None
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    heads = [struct.unpack_from("<IIQQQQ", data, shoff + i * shentsize) for i in range(shnum)]
    names_off, names_size = heads[shstrndx][4], heads[shstrndx][5]
    names = data[names_off:names_off + names_size]
    out = {}
    for name, _, _, _, off, size in heads:
        out[names[name:names.index(b"\0", name)].decode()] = (off, size)
    return out


def symbol_names_only(a, b):
    """Why two files differ, when the reason is Zig's symbol naming and nothing else: both ELF of one size, every
    differing byte inside .strtab, and .strtab equal once each __anon_<n> keeps only its low 24 bits (Zig 0.17's
    InternPool puts the id of the compiler thread that interned the value in the high bits of n, and which worker
    thread that is varies from run to run). None when the difference is anything else."""
    x, y = open(a, "rb").read(), open(b, "rb").read()
    if len(x) != len(y):
        return None
    sx, sy = elf_sections(x), elf_sections(y)
    if not sx or sx != sy or ".strtab" not in sx:
        return None
    off, size = sx[".strtab"]
    if x[:off] != y[:off] or x[off + size:] != y[off + size:]:
        return None
    low = lambda s: ANON.sub(lambda m: b"__anon_%d" % (int(m.group(1)) & 0xFFFFFF), s)
    tx, ty = x[off:off + size], y[off:off + size]
    if low(tx) != low(ty):
        return None
    names = sum(1 for p, q in zip(ANON.findall(tx), ANON.findall(ty)) if p != q)
    return f"symbol names only: {names} __anon_<n> names in .strtab differ in the compiler thread id bits"


def compare(ours, ref):
    """Lines describing every difference between two stages, plus the counts."""
    a, b = files_of(ours), files_of(ref)
    for d in (a, b):
        d.pop("DEPLOY.json", None)       # names its own time, host and maker
        d.pop("SHA256SUMS", None)        # the list itself: compared line by line below
    lines = []
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    diff = sorted(p for p in set(a) & set(b) if a[p] != b[p])
    la, lb = links_of(ours), links_of(ref)
    for p in only_a:
        lines.append(f"ONLY-FLERP {p}")
    for p in only_b:
        lines.append(f"ONLY-REFERENCE {p}")
    names_only = 0
    for p in diff:
        why = symbol_names_only(os.path.join(ours, p), os.path.join(ref, p))
        names_only += why is not None
        lines.append(f"DIFFERS {p} flerp {a[p][:12]} reference {b[p][:12]} bytes {os.path.getsize(os.path.join(ours, p))} "
                     f"{os.path.getsize(os.path.join(ref, p))}" + (f" ({why})" if why else ""))
    for p in sorted(set(la) | set(lb)):
        if la.get(p) != lb.get(p):
            lines.append(f"LINK {p} flerp {la.get(p)} reference {lb.get(p)}")
    sa, sb = read_sums(os.path.join(ours, "SHA256SUMS")), read_sums(os.path.join(ref, "SHA256SUMS"))
    same = len(set(a) & set(b)) - len(diff)
    sums_lines = sorted(n for n in set(sa) | set(sb) if sa.get(n) != sb.get(n))
    if list(sa) != list(sb) or any(n not in diff for n in sums_lines):
        lines.append(f"SHA256SUMS differs beyond the differing files: {[n for n in sums_lines if n not in diff][:10]}")
    summary = (f"files: flerp {len(a)}, reference {len(b)}, byte-equal {same}, differing {len(diff)}, only flerp "
               f"{len(only_a)}, only reference {len(only_b)}; SHA256SUMS lines flerp {len(sa)} reference {len(sb)}, "
               f"same names {sorted(sa) == sorted(sb)}, same order {list(sa) == list(sb)}, same file bytes {sha256(os.path.join(ours, 'SHA256SUMS')) == sha256(os.path.join(ref, 'SHA256SUMS'))}, "
               f"lines that differ {len(sums_lines)} (all for differing files: {all(n in diff for n in sums_lines)})")
    summary += f"; of the differing files, symbol names only {names_only}"
    return lines, summary, (not lines) or (names_only == len(diff) and not only_a and not only_b
                                            and all(l.startswith("DIFFERS ") for l in lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("recipe")
    ap.add_argument("out")
    ap.add_argument("--compare", help="a served stage (default: [deploy.stage] reference)")
    ap.add_argument("--report", help="write the comparison here as well")
    ap.add_argument("--compare-only", action="store_true", help="out exists: compare it, make nothing")
    a = ap.parse_args()
    cat = Catalog(ROOT)
    r, st = recipe_stage(cat, a.recipe)
    out = os.path.abspath(a.out) if a.compare_only else make(cat, r, st, a.out)
    ref = a.compare or st.get("reference")
    if not ref:
        return 0
    lines, summary, names_only = compare(out, ref)
    verdict = "STAGE-EQUAL" if not lines else "STAGE-EQUAL-BUT-SYMBOL-NAMES" if names_only else "STAGE-DIFFERS"
    text = (f"# tools/flerp-stage.py {a.recipe}: {out} against {ref}\n" + "".join(l + "\n" for l in lines)
            + summary + "\n" + verdict + "\n")
    sys.stdout.write(text)
    if a.report:
        with open(a.report, "w") as f:
            f.write(text)
    return 0 if not lines else 3 if names_only else 1


if __name__ == "__main__":
    sys.exit(main())
