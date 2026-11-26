"""kit_add.py BASE_KIT ADD_KIT OUT_KIT --label NAME [--card ID]: OUT_KIT = BASE_KIT's Triton variants plus the ones of
ADD_KIT (an aot_build.py output) that BASE_KIT does not hold, by aot_build's distinct key (function, parameters,
constexprs, options). Code objects are hard-linked where the file system allows, else copied. aot.json keeps BASE's
records and adds NAME: the added count and functions, ADD's triton build and whether its release is BASE's. Refuses an
ADD built by another triton release than BASE's (3.7.0 and 3.8.0 lower _rowmm_parts' dot to other bits)."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def key(k):
    return (k["fn"], json.dumps(k["params"], sort_keys=True), json.dumps(k["consts"], sort_keys=True),
            json.dumps(k["options"], sort_keys=True))


def release(v):
    return v.split("+")[0]


def place(src, dst):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("add")
    ap.add_argument("out")
    ap.add_argument("--label", required=True)
    ap.add_argument("--card", default="")
    a = ap.parse_args()
    base, add, out = Path(a.base), Path(a.add), Path(a.out)
    bm = json.loads((base / "aot.json").read_text())
    am = json.loads((add / "aot.json").read_text())
    if release(am["triton"]) != release(bm["triton"]):
        raise SystemExit(f"REFUSED: {add} was built by triton {am['triton']}, {base} by {bm['triton']} (another release)")
    if out.exists():
        raise SystemExit(f"REFUSED: {out} exists")
    (out / "hsaco").mkdir(parents=True)
    have = {key(k): k for k in bm["kernels"]}
    kernels = list(bm["kernels"])
    added, same = [], []
    for k in am["kernels"]:
        if key(k) in have:
            same.append({"fn": k["fn"], "base_hash": have[key(k)]["hash"][:16], "add_hash": k["hash"][:16]})
            continue
        data = (add / k["file"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != k["hash"]:
            raise SystemExit(f"REFUSED: {add / k['file']} is not its hash")
        kernels.append(k)
        added.append(k)
    for k in kernels:
        src = (base if k in bm["kernels"] else add) / k["file"]
        place(src, out / k["file"])
    rec = {kk: v for kk, v in bm.items() if kk != "kernels"}
    rec[a.label] = {"card": a.card, "count": len(added), "fns": sorted({k["fn"] for k in added}),
                    "triton": am["triton"], "triton_build": am.get("triton_build"),
                    "toolchain_match": "build" if am["triton"] == bm["triton"] else "release",
                    "already_in_base": same}
    rec["kernels"] = kernels
    (out / "aot.json").write_text(json.dumps(rec, indent=1))
    if (add / "build_log.json").exists():
        shutil.copy2(add / "build_log.json", out / f"build_log.{a.label}.json")
    if (base / "build_log.json").exists():
        place(base / "build_log.json", out / "build_log.json")
    print(f"kit {out}: {len(bm['kernels'])} + {len(added)} = {len(kernels)} kernels ({len(same)} of the added already in base)")


if __name__ == "__main__":
    main()

