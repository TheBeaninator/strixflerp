#!/usr/bin/env python3
"""describe-audit: does every catalogue module say what it is, how it works and why it exists?

Checks each catalog/**/module.toml for a `summary` (what and how, one sentence) and a `why` (why the module exists),
both long enough to carry an idea and neither shared word for word with another module (SPEC.md section 5).
Prints one line per module that falls short and a count per origin; exit status 1 when any does.

    tools/describe-audit.py                 # the whole catalogue
    tools/describe-audit.py --origin gufo   # one origin
    tools/describe-audit.py --list          # also print every module's summary and why (for reading)
"""
import argparse, collections, pathlib, sys, tomllib

MIN_SUMMARY, MIN_WHY = 40, 60


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--origin")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    root = pathlib.Path(__file__).resolve().parent.parent
    mods = []
    for f in sorted((root / "catalog").glob("*/*/*/module.toml")):
        origin = f.relative_to(root / "catalog").parts[0]
        if a.origin and origin != a.origin:
            continue
        t = tomllib.loads(f.read_text())
        mods.append((origin, f.relative_to(root), t.get("summary", "").strip(), t.get("why", "").strip()))
    seen_s = collections.Counter(m[2] for m in mods if m[2])
    seen_w = collections.Counter(m[3] for m in mods if m[3])
    bad = collections.Counter()
    total = collections.Counter()
    for origin, path, s, w in mods:
        total[origin] += 1
        why_not = []
        if len(s) < MIN_SUMMARY:
            why_not.append("summary missing or too short")
        elif seen_s[s] > 1:
            why_not.append("summary shared with another module")
        if len(w) < MIN_WHY:
            why_not.append("why missing or too short")
        elif seen_w[w] > 1:
            why_not.append("why shared with another module")
        if why_not:
            bad[origin] += 1
            print(f"{path}: {'; '.join(why_not)}")
        if a.list:
            print(f"  {path.parent.name}\n    what/how: {s}\n    why: {w}")
    for origin in sorted(total):
        print(f"{origin:14} {total[origin] - bad[origin]:4} of {total[origin]:4} described")
    print(f"total          {sum(total.values()) - sum(bad.values()):4} of {sum(total.values()):4} described")
    return 1 if sum(bad.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
