"""kit_strict.py KIT [--fn F,...] [--json OUT]: the kit's strict-mode coverage on the host (no HIP runtime, no GPU):
every recorded and derived launch (fixtures/launches.json and derived_launches.json, the 2D split's tp4 cases
included) checked for its own variant in KIT/aot.json by the strict rule (hiprt.Kit._matches, zig/src/gpu/hip/aot.zig
with TF_GPU_STRICT=1). Prints the launches a strict run would refuse (the Triton half of a GPU_STRICT open's list) and
exits 1 when there is any. t_cover.py checks the same through hiprt, which initializes the HIP runtime."""
import argparse
import json
import struct
import sys
from pathlib import Path

from aot_build import TORCH_TO_TRITON

FIX = Path(__file__).resolve().parents[3] / "zig" / "src" / "families" / "dsv41" / "fixtures"


def f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def matches(k, args, consts):
    """hiprt.Kit._matches: the variant's constexprs exactly, every runtime parameter's type and specialization."""
    for name, v in consts.items():
        got = k["consts"].get(name)
        if got is None:
            return False
        if isinstance(v, float):
            if "f32" not in got or got["f32"] != f32_bits(v):
                return False
        elif "int" not in got or got["int"] != int(v):
            return False
    for n in set(k["consts"]) - set(consts):
        a = args.get(n)
        if not (isinstance(a, int) and not isinstance(a, bool) and a == 1 and k["consts"][n].get("int") == 1):
            return False
    runtime = 0
    params = {p["name"]: p for p in k["params"]}
    for name, a in args.items():
        p = params.get(name)
        if isinstance(a, tuple):
            ty, aligned = a
            if p is None or p["type"] != ty or p["div16"] != aligned:
                return False
        elif isinstance(a, float):
            if p is None or p["type"] != "fp32":
                return False
        else:
            if p is None:
                if a != 1 or k["consts"].get(name, {}).get("int") != 1:
                    return False
                continue
            if p["type"] != "i32" or (not p["nospec"] and a == 1):
                return False
            if p["div16"] != ((not p["nospec"]) and a % 16 == 0):
                return False
        runtime += 1
    return runtime == len(k["params"])


def launch_of(case):
    """t_cover.launch_of: pointers as (type, 16-byte aligned), the scalars, the constexprs."""
    args = {n: (TORCH_TO_TRITON[d], bool(al)) for n, (d, _s, _st, al) in case["tensors"].items()}
    for n, v in case["scalars"].items():
        args[n] = float(v["float"]) if isinstance(v, dict) else int(v)
    consts = {n: (float(v["float"]) if isinstance(v, dict) else int(v)) for n, v in case["consts"].items()}
    return args, consts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kit")
    ap.add_argument("--fn", default="")
    ap.add_argument("--json")
    a = ap.parse_args()
    kernels = json.loads((Path(a.kit) / "aot.json").read_text())["kernels"]
    launches = json.loads((FIX / "launches.json").read_text())
    for fn, cases in json.loads((FIX / "derived_launches.json").read_text()).items():
        launches.setdefault(fn, []).extend(cases)
    only = set(filter(None, a.fn.split(",")))
    report, total, missing = {}, 0, 0
    for fn, cases in sorted(launches.items()):
        if only and fn not in only:
            continue
        mine = [k for k in kernels if k["fn"] == fn]
        miss = []
        for c in cases:
            args, consts = launch_of(c)
            total += 1
            if not any(matches(k, args, consts) for k in mine):
                miss.append({"phase": c["phase"], "rule": c.get("derived", {}).get("rule", "recorded"), "grid": c["grid"],
                             "consts": c["consts"]})
        missing += len(miss)
        report[fn] = {"launches": len(cases), "variants": len(mine), "missing": miss}
        print(f"{fn}: {len(cases)} launches, {len(mine)} variants, {len(miss)} without their own")
        for m in miss:
            k = m["consts"]
            print("  MISSING " + m["phase"] + " " + " ".join(f"{x}={k[x]}" for x in sorted(k)))
    print(f"strict: {total} launches, {missing} without their own variant in {a.kit}")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1, sort_keys=True))
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()

