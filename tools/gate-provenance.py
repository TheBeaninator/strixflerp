#!/usr/bin/env python3
"""Which gate TOML did each result under gate/results run under? (docs/PROVENANCE.md)

    python3 tools/gate-provenance.py [--strict] [--json] [--root DIR]

A result is a directory under gate/results/ (each gate/results/interface/<name>/ is its own result) or a file
directly under gate/results/. Its recorded config_sha256 values are the top-level "config_sha256" keys of the
JSON files in it. Its gate TOML is the one a JSON file names ("config", "config_path", "criteria_file"), else
gate/<name>.toml, gate/interface/<name>.toml for an interface result, else the gate TOML whose name is the
longest prefix of the result's (or the reverse). Each recorded sha is matched, in order, against:

  current           sha256 of the TOML's bytes in this tree
  current-internal  the internal_sha256 of that TOML in gate/PROVENANCE-MAP.tsv (public export only: the export
                    rewrites hostnames, addresses and home paths, so a sha recorded on the pod is of the internal
                    bytes; the map is absent in the internal tree)
  older <commit>    the TOML's bytes at an earlier commit of this repository's history
  unmatched         none of these

A result that records no config_sha256 is "unrecorded" (results written before the shared summary helper
recorded it); that is listed, not a failure. Exit 0; with --strict, exit 1 when any result is unrecorded or
unmatched or has no gate TOML.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys

TOML_KEYS = ("config", "config_path", "criteria_file", "toml")


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Tree:
    def __init__(self, root):
        self.root = root
        self.gate = os.path.join(root, "gate")
        self.results = os.path.join(self.gate, "results")
        self.tomls = sorted(os.path.relpath(os.path.join(d, f), root)
                            for d, dirs, files in os.walk(self.gate)
                            if not os.path.relpath(d, self.gate).startswith("results")
                            for f in files if f.endswith(".toml"))
        self.map = {}
        mp = os.path.join(self.gate, "PROVENANCE-MAP.tsv")
        if os.path.isfile(mp):
            with open(mp, newline="") as f:
                for row in csv.reader(f, delimiter="\t"):
                    if len(row) >= 3 and row[0] != "toml_path" and not row[0].startswith("#"):
                        self.map[row[0]] = (row[1], row[2])
        self._hist = {}
        self.git = subprocess.run(["git", "-C", root, "rev-parse", "--git-dir"],
                                  capture_output=True).returncode == 0

    def current(self, rel):
        with open(os.path.join(self.root, rel), "rb") as f:
            return sha(f.read())

    def history(self, rel):
        """{sha256: first (oldest) commit with those bytes} over the TOML's git history."""
        if rel in self._hist:
            return self._hist[rel]
        out = {}
        if self.git:
            revs = subprocess.run(["git", "-C", self.root, "rev-list", "HEAD", "--", rel],
                                  capture_output=True, text=True).stdout.split()
            for c in revs:                                      # newest first; keep overwriting to get the oldest
                p = subprocess.run(["git", "-C", self.root, "show", f"{c}:{rel}"], capture_output=True)
                if p.returncode == 0:
                    out[sha(p.stdout)] = c[:12]
        self._hist[rel] = out
        return out

    def units(self):
        out = []
        for n in sorted(os.listdir(self.results)) if os.path.isdir(self.results) else []:
            p = os.path.join(self.results, n)
            if n == "interface" and os.path.isdir(p):
                out += [(f"interface/{m}", os.path.join(p, m), True) for m in sorted(os.listdir(p))
                        if os.path.isdir(os.path.join(p, m))]
            elif n not in ("ARTIFACTS.md",):
                out.append((n, p, False))
        return out

    def find_toml(self, name, named, interface):
        for t in named:
            t = t if t.startswith("gate/") else f"gate/{t}"
            if t in self.tomls:
                return t
        base = name.split("/")[-1]
        if interface and f"gate/interface/{base}.toml" in self.tomls:
            return f"gate/interface/{base}.toml"
        if f"gate/{base}.toml" in self.tomls:
            return f"gate/{base}.toml"
        stems = [t for t in self.tomls if t.count("/") == 1]
        cands = [t for t in stems if base.startswith(t[5:-5]) or t[5:-5].startswith(base)]
        return max(cands, key=len) if cands else None


def scan(path):
    """(recorded shas, TOML paths named) from the JSON files of a result."""
    shas, named = [], []
    files = [path] if os.path.isfile(path) else [os.path.join(d, f) for d, _, fs in os.walk(path) for f in fs]
    for f in sorted(files):
        if not f.endswith(".json"):
            continue
        try:
            with open(f) as fh:
                doc = json.load(fh)
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        v = doc.get("config_sha256")
        if isinstance(v, str) and len(v) == 64 and v not in shas:
            shas.append(v)
        for k in TOML_KEYS:
            t = doc.get(k)
            if isinstance(t, str) and t.endswith(".toml") and t not in named:
                named.append(t)
    return shas, named


def judge(tree, toml, s):
    if toml is None:
        return "unmatched", "no gate TOML found"
    if s == tree.current(toml):
        return "current", ""
    m = tree.map.get(toml)
    if m and s == m[0]:
        return "current-internal", "PROVENANCE-MAP.tsv internal_sha256"
    old = tree.history(toml).get(s)
    if old:
        return "older", old
    return "unmatched", ""


def main():
    ap = argparse.ArgumentParser(description="report which gate TOML each gate result ran under")
    ap.add_argument("--strict", action="store_true", help="exit 1 on unrecorded, unmatched or TOML-less results")
    ap.add_argument("--json", action="store_true", help="print JSON rows instead of a table")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    a = ap.parse_args()
    tree = Tree(os.path.abspath(a.root))
    rows = []
    for name, path, interface in tree.units():
        shas, named = scan(path)
        toml = tree.find_toml(name, named, interface)
        if not shas:
            rows.append({"result": name, "toml": toml, "state": "unrecorded", "sha": None, "detail": ""})
        for s in shas:
            st, det = judge(tree, toml, s)
            rows.append({"result": name, "toml": toml, "state": st, "sha": s, "detail": det})
    counts = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    if a.json:
        print(json.dumps({"rows": rows, "counts": counts, "map": bool(tree.map)}, indent=1))
    else:
        w = max([len(r["result"]) for r in rows] + [6])
        print(f"{'result':<{w}}  {'state':<16}  {'gate TOML':<44}  sha256 / detail")
        for r in rows:
            print(f"{r['result']:<{w}}  {r['state']:<16}  {r['toml'] or '-':<44}  "
                  f"{(r['sha'] or '')[:16]} {r['detail']}".rstrip())
        print(f"\n{len(rows)} rows: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
              + ("; PROVENANCE-MAP.tsv present" if tree.map else "; no PROVENANCE-MAP.tsv (internal tree)"))
    bad = sum(1 for r in rows if r["state"] in ("unrecorded", "unmatched") or r["toml"] is None)
    return 1 if a.strict and bad else 0


if __name__ == "__main__":
    sys.exit(main())
