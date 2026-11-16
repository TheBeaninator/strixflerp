#!/usr/bin/env python3
"""Write gate/PROVENANCE-MAP.tsv for a public export (docs/PROVENANCE.md).

    python3 tools/provenance-map.py --internal <internal tree> --public <public tree> [--out FILE]

For every gate TOML of the internal tree (gate/**/*.toml outside gate/results), one row
`toml_path<TAB>internal_sha256<TAB>public_sha256`: the sha256 of the TOML's bytes in each tree. The export
rewrites hostnames, addresses and home paths, so a config_sha256 a gate recorded on the pod is the internal sha;
tools/gate-provenance.py in the public tree matches it through this map. A TOML absent from the public tree gets
"-" as its public sha. The first line is the column header. Default output: <public>/gate/PROVENANCE-MAP.tsv.
Point --public only at an export tree: the map is a public-tree file (the internal tree has none), so run
against a working checkout it leaves an untracked gate/PROVENANCE-MAP.tsv behind.
"""
import argparse
import hashlib
import os
import sys


def gate_tomls(root):
    gate = os.path.join(root, "gate")
    out = []
    for d, dirs, files in os.walk(gate):
        rel = os.path.relpath(d, gate)
        if rel == "results" or rel.startswith("results" + os.sep):
            dirs[:] = []
            continue
        dirs.sort()
        out += [os.path.relpath(os.path.join(d, f), root) for f in sorted(files) if f.endswith(".toml")]
    return sorted(out)


def sha_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="write gate/PROVENANCE-MAP.tsv (internal vs public gate TOML sha256)")
    ap.add_argument("--internal", required=True, help="the internal tree (pod git checkout or export source)")
    ap.add_argument("--public", required=True, help="the scrubbed public tree")
    ap.add_argument("--out", help="output file (default <public>/gate/PROVENANCE-MAP.tsv)")
    a = ap.parse_args()
    for d in (a.internal, a.public):
        if not os.path.isdir(os.path.join(d, "gate")):
            sys.exit(f"provenance-map: {d} has no gate/ directory")
    out = a.out or os.path.join(a.public, "gate", "PROVENANCE-MAP.tsv")
    rows = ["toml_path\tinternal_sha256\tpublic_sha256"]
    missing = 0
    for rel in gate_tomls(a.internal):
        pub = os.path.join(a.public, rel)
        ps = sha_file(pub) if os.path.isfile(pub) else "-"
        missing += ps == "-"
        rows.append(f"{rel.replace(os.sep, '/')}\t{sha_file(os.path.join(a.internal, rel))}\t{ps}")
    with open(out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"provenance-map: {len(rows) - 1} gate TOMLs -> {out}" + (f" ({missing} absent from the public tree)" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
