"""aot_fuse2.py SRC_KIT OUT_KIT [--check-ref]: OUT_KIT = SRC_KIT plus SPEED-FUSE-2's fused Triton kernels AOT-compiled
for gfx1151 (wave32), the entries in aot_build.py's schema, the pattern of hc/aot_chain.py:
  _rowmm_route (route_fuse.py): the main model's router at every decode / verify window of 1-16 rows (N 384, top 6 of
  7 slots, K 5120), one variant a row count as _rowmm_parts has (RB == rows; BN by tri_norm.rpTile).
The kit's other kernels are untouched; _rowmm_parts and _route_parts stay for TF_FUSE2=0.
Build with a triton 3.8.0 (TRITON_WHEEL names the wheel, recorded in fuse2_added; a triton whose release is not the
kit manifest's is REFUSED and nothing is written): _rowmm_parts' dot reduction comes
out of triton 3.7.0 with other bits than the kit's (bench box K: 2,277 of a row's 7,680 chunk sums differ), while
3.8.0+git4cff872c.rocm10.0.0 gives the kit's bits (128/128 cases, test_route_fuse.py).
--check-ref first recompiles the kit's own _route_parts (NE 384) with THIS triton and prints whether the code object
is the kit's byte for byte (the fused kernel inlines the same source, so equal objects mean equal lowering here)."""
import hashlib, json, os, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
import route_fuse as RF
import tri_kernels as K

TARGET = GPUTarget("hip", "gfx1151", 32)
PTR = {"X": "*bf16", "W": "*fp16", "S": "*fp32", "CNT": "*i32", "BIAS": "*fp32", "PICK": "*i32", "WTS": "*fp32"}
# the ints the engine passes are multiples of 16 (xs = 5120, shared_id = 384), as the kit's own variants record them
SCAL = {"xs": ("i32", True), "scale": ("fp32", False), "shared_id": ("i32", True)}


def compile_one(fn, consts, num_warps):
    sig, constexprs, attrs, params = {}, {}, {}, []
    for i, name in enumerate(fn.arg_names):
        key = "K" if name == "K_" else name
        if key in consts:
            sig[name] = "constexpr"
            constexprs[name] = consts[key]
        elif name in PTR:
            sig[name] = PTR[name]
            attrs[(i,)] = [["tt.divisibility", 16]]
            params.append({"name": name, "type": PTR[name], "div16": True, "nospec": False})
        else:
            ty, d16 = SCAL[name]
            sig[name] = ty
            if d16:
                attrs[(i,)] = [["tt.divisibility", 16]]
            params.append({"name": name, "type": ty, "div16": d16, "nospec": False})
    k = triton.compile(ASTSource(fn, sig, constexprs=constexprs, attrs=attrs), target=TARGET,
                       options={"num_warps": num_warps})
    return k, params


def entry(fname, k, params, consts, num_warps, grid, origin):
    hsaco = k.asm["hsaco"]
    sha = hashlib.sha256(hsaco).hexdigest()
    md = k.metadata
    stats = {}
    for line in k.asm.get("amdgcn", "").splitlines():
        line = line.strip()
        for key in (".vgpr_count:", ".sgpr_count:", ".vgpr_spill_count:", ".sgpr_spill_count:",
                    ".private_segment_fixed_size:", ".group_segment_fixed_size:"):
            if line.startswith(key):
                stats[key.strip(".:")] = int(line.split(":")[1].strip().split()[0])
    return hsaco, {"fn": fname, "hash": sha, "name": md.name, "num_warps": md.num_warps, "num_ctas": md.num_ctas,
                   "shared": md.shared, "global_scratch": getattr(md, "global_scratch_size", 0),
                   "global_align": getattr(md, "global_scratch_align", 1),
                   "profile_scratch": getattr(md, "profile_scratch_size", 0), "pdl": False, "params": params,
                   "consts": {kk: {"int": vv} for kk, vv in consts.items()},
                   "arch": "gfx1151", "warp_size": md.warp_size, "block": md.num_warps * md.warp_size,
                   "file": f"hsaco/{sha}.hsaco", "options": {"num_warps": num_warps}, "isa_stats": stats,
                   "recorded_grid": grid, "origin": origin}


def check_ref(kit):
    """the kit's _route_parts (NE 384) recompiled here: the same code object?"""
    m = json.load(open(kit / "aot.json"))
    ref = [e for e in m["kernels"] if e["fn"] == "_route_parts" and e["consts"]["NE"]["int"] == 384][0]
    consts = {kk: vv["int"] for kk, vv in ref["consts"].items()}
    fn = K._route_parts
    sig, constexprs, attrs = {}, {}, {}
    types = {p["name"]: p for p in ref["params"]}
    for i, name in enumerate(fn.arg_names):
        if name in consts:
            sig[name] = "constexpr"
            constexprs[name] = bool(consts[name]) if name == "PDL" else consts[name]
        else:
            sig[name] = types[name]["type"]
            if types[name]["div16"]:
                attrs[(i,)] = [["tt.divisibility", 16]]
    k = triton.compile(ASTSource(fn, sig, constexprs=constexprs, attrs=attrs), target=TARGET,
                       options={"num_warps": ref["num_warps"]})
    sha = hashlib.sha256(k.asm["hsaco"]).hexdigest()
    print(json.dumps({"check_ref": "_route_parts NE=384", "kit_triton": m.get("triton"), "this_triton": triton.__version__,
                      "kit_hash": ref["hash"][:16], "rebuilt_hash": sha[:16], "identical": sha == ref["hash"]}), flush=True)
    return k


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    if "--check-ref" in sys.argv:
        check_ref(src)
    # the toolchain gate (b7269694f8, as hc/aot_chain.py's): another triton release lowers _rowmm_parts' dot reduction
    # to other bits than the kit's, so a release (the version before '+') that is not the kit manifest's is refused
    kit_tri = json.load(open(src / "aot.json")).get("triton") or ""
    if not kit_tri or triton.__version__.split("+")[0] != kit_tri.split("+")[0]:
        sys.exit(f"REFUSED: this triton is {triton.__version__}, the kit's kernels were built with "
                 f"{kit_tri or 'an unrecorded triton'}: another release lowers the same source to other bits.")
    if out.exists():
        sys.exit(f"{out} exists")
    shutil.copytree(src, out)
    m = json.load(open(out / "aot.json"))
    added = []
    for rows in range(1, 17):
        bn = RF.rp_tile(rows)
        consts = {"K": 5120, "N": 384, "BN": bn, "BK": 256, "CPG": 1, "RB": rows, "NB": 512, "TOPK": 6, "SLOTS": 7,
                  "SP": 8}
        k, params = compile_one(RF._rowmm_route, consts, 2)
        hsaco, e = entry("_rowmm_route", k, params, consts, 2, [(384 + bn - 1) // bn, 20, 1],
                         "SPEED-FUSE-2 route_fuse.py (tools/dsv41-hip/exl3/kdec/fuse2)")
        (out / "hsaco" / f"{e['hash']}.hsaco").write_bytes(hsaco)
        m["kernels"].append(e)
        added.append(e["hash"])
        print("_rowmm_route RB", rows, "BN", bn, e["hash"][:12], e["isa_stats"], "shared", e["shared"], flush=True)
    if "--diag" in sys.argv:       # diagnostic only: the kit's own _rowmm_parts source recompiled by this triton
        PTR["rows"] = None
        for rows in (1, 2, 16):
            bn = RF.rp_tile(rows)
            consts = {"K": 5120, "N": 384, "BN": bn, "BK": 256, "CPG": 1, "RB": rows, "PDL": 1}
            fn = K._rowmm_parts
            sig, constexprs, attrs, params = {}, {}, {}, []
            for i, name in enumerate(fn.arg_names):
                if name in consts:
                    sig[name], constexprs[name] = "constexpr", (bool(consts[name]) if name == "PDL" else consts[name])
                elif name == "rows":
                    sig[name], constexprs[name] = "constexpr", rows
                else:
                    ty = dict(PTR, xs="i32")[name]
                    sig[name] = ty
                    attrs[(i,)] = [["tt.divisibility", 16]]
                    params.append({"name": name, "type": ty, "div16": True, "nospec": False})
            k = triton.compile(ASTSource(fn, sig, constexprs=constexprs, attrs=attrs), target=TARGET,
                               options={"num_warps": 2})
            hsaco, e = entry("_rowmm_parts_re", k, params, consts, 2, [(384 + bn - 1) // bn, 20, 1], "diagnostic")
            e["name"] = k.metadata.name
            (out / "hsaco" / f"{e['hash']}.hsaco").write_bytes(hsaco)
            m["kernels"].append(e)
            print("_rowmm_parts_re RB", rows, e["hash"][:12], flush=True)
    m["fuse2_added"] = {"fns": ["_rowmm_route"], "count": len(added),
                        "source_sha256": {"route_fuse.py": hashlib.sha256(open(RF.__file__, "rb").read()).hexdigest()},
                        "triton": triton.__version__, "triton_wheel": os.environ.get("TRITON_WHEEL", ""),
                        "python": sys.version.split()[0]}
    json.dump(m, open(out / "aot.json", "w"), indent=1)
    print("kit", out, "kernels", len(m["kernels"]))
