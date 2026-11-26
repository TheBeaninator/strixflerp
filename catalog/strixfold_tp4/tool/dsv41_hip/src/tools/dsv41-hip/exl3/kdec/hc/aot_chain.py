"""aot_chain.py SRC_KIT OUT_KIT [--accept-build TRITON_VERSION] [--check-ref]: OUT_KIT = SRC_KIT plus
hc_chain._hc_chain AOT-compiled for gfx1151 (wave32) for every decode / verify window of 1-16 rows and world 1-2: one
variant a (RB, rows' specialization, WORLD) as the engine's tri_hc.chainLaunch launches it (rows 1 folded to a
constexpr as Triton does; rows % 16 == 0 specialized; RS = rows * D taken as 16-divisible, as the kit's own
_hc_post_only records it), the entries in aot_build.py's schema, the pattern of aot_fused.py. The kit's other kernels
are untouched; the 4-launch chain it replaces (post_only / finish_s / mix_part / sinkhorn) stays in the kit for
TF_HC_CHAIN=0. _hc_chain entries SRC_KIT already has are replaced.

The toolchain gate (b7269694f8, 2026-11-17): the fused kernel must be lowered as the kit's four kernels were, or its
bits are not theirs (triton 3.7.0 against the 3.8.0 kit: other bits in a dot reduction, SPEED-FUSE-2's finding). So:
  - REFUSED when this triton's release (the version before '+') is not the kit manifest's, or when this tree's
    tri_kernels.py is not the kit's (triton_build.kernels_py_sha256): the fused kernel inlines that source.
  - the same triton version and python minor as the manifest: built, recorded toolchain_match "exact".
  - the same release from another build or python (the kit's own nightly wheel is not on the pod): REFUSED unless
    --accept-build names this triton's full version; recorded toolchain_match "release". Such a kit is proven only by
    the device test AOT against AOT (test_chain_aot.py on the gfx1151 bench box), never by a JIT-vs-JIT run.
--check-ref recompiles the kit's own four chain kernels with this triton and records whether each code object is the
kit's byte for byte. TRITON_WHEEL names the wheel for the record."""
import hashlib, json, os, shutil, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "tri"))  # this tree's tri_kernels.py, checked against the kit's below
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
import tri_kernels as K
import hc_chain as C
import hc_fuse as F

TARGET = GPUTarget("hip", "gfx1151", 32)


def triton_version():
    """the installed distribution's full version (3.8.0+git4cff872c.rocm10.0.0: the build is in the local part, which
    triton.__version__ drops), as the kit manifest records its own"""
    try:
        from importlib.metadata import version
        full = version("triton")
        if full.split("+")[0] == triton.__version__.split("+")[0]:
            return full
    except Exception:  # noqa: BLE001 - no distribution metadata (a source tree)
        pass
    return triton.__version__


TRITON = triton_version()
FLAGS = ("PDL", "POSTED", "XPF", "SINK")


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def release(v):
    return v.split("+")[0]


def toolchain_gate(m, accept):
    tb = m.get("triton_build") or {}
    kit_tri, kit_py, here_py = m.get("triton") or tb.get("version"), tb.get("python", ""), sys.version.split()[0]
    if not kit_tri:
        sys.exit("REFUSED: the kit manifest records no triton version")
    if release(TRITON) != release(kit_tri):
        sys.exit(f"REFUSED: this triton is {TRITON}, the kit's kernels were built with {kit_tri}: another "
                 f"release lowers the same source to other bits. Build with a triton {release(kit_tri)}.")
    if tb.get("kernels_py_sha256") and sha_file(K.__file__) != tb["kernels_py_sha256"]:
        sys.exit(f"REFUSED: {K.__file__} is not the kit's tri_kernels.py (sha256 {tb['kernels_py_sha256'][:16]}...)")
    if TRITON == kit_tri and here_py.split(".")[:2] == kit_py.split(".")[:2]:
        return "exact", kit_tri, kit_py
    if accept != TRITON:
        sys.exit(f"REFUSED: this toolchain is triton {TRITON} / python {here_py}, the kit manifest records "
                 f"{kit_tri} / python {kit_py}: the same release from another build. Pass --accept-build "
                 f"{TRITON} to build, then prove the bits AOT against AOT with test_chain_aot.py.")
    return "release", kit_tri, kit_py


def check_ref(m):
    """the kit's own four chain kernels recompiled here: the same code objects?"""
    want = {"_hc_post_only": {}, "_hc_finish_s": {"SINK": 0}, "_hc_sinkhorn": {},
            "_hc_mix_part": {"WORLD": 1, "POSTED": 0, "PDL": 0, "RB": 2}}
    res = {}
    for fname, sel in want.items():
        ref = [e for e in m["kernels"] if e["fn"] == fname and all(e["consts"][k]["int"] == v for k, v in sel.items())][0]
        consts = {k: v["int"] for k, v in ref["consts"].items()}
        types = {p["name"]: p for p in ref["params"]}
        fn = getattr(K, fname)
        sig, constexprs, attrs = {}, {}, {}
        for i, name in enumerate(fn.arg_names):
            if name in consts:
                sig[name] = "constexpr"
                constexprs[name] = bool(consts[name]) if name in FLAGS else consts[name]
            else:
                sig[name] = types[name]["type"]
                if types[name]["div16"]:
                    attrs[(i,)] = [["tt.divisibility", 16]]
        try:
            k = triton.compile(ASTSource(fn, sig, constexprs=constexprs, attrs=attrs), target=TARGET,
                               options={"num_warps": ref["num_warps"]})
            sha = hashlib.sha256(k.asm["hsaco"]).hexdigest()
            res[fname] = {"kit_hash": ref["hash"][:16], "rebuilt_hash": sha[:16], "identical": sha == ref["hash"]}
        except Exception as e:  # noqa: BLE001 - a diagnostic, never the build
            res[fname] = {"kit_hash": ref["hash"][:16], "error": str(e)[:200]}
        print(json.dumps({"check_ref": fname, "this_triton": TRITON, **res[fname]}), flush=True)
    return res


src, out = Path(sys.argv[1]), Path(sys.argv[2])
accept = sys.argv[sys.argv.index("--accept-build") + 1] if "--accept-build" in sys.argv else None
m = json.load(open(src / "aot.json"))
match, kit_tri, kit_py = toolchain_gate(m, accept)
print(json.dumps({"toolchain_match": match, "this_triton": TRITON, "this_python": sys.version.split()[0],
                  "kit_triton": kit_tri, "kit_python": kit_py}), flush=True)
ref = check_ref(m) if "--check-ref" in sys.argv else None
if out.exists():
    sys.exit(f"{out} exists")
shutil.copytree(src, out)
stale = [e for e in m["kernels"] if e["fn"] == "_hc_chain"]
m["kernels"] = [e for e in m["kernels"] if e["fn"] != "_hc_chain"]
kept = {e["file"] for e in m["kernels"]}
for e in stale:
    if e["file"] not in kept and (out / e["file"]).exists():
        (out / e["file"]).unlink()
print("replaced", len(stale), "_hc_chain entries of the source kit", flush=True)
P = [("X", "*bf16"), ("XO", "*bf16"), ("G", "*fp32"), ("RS", "i32"), ("POST", "*fp32"), ("COMB", "*fp32"),
     ("FN", "*fp32"), ("PART", "*fp32"), ("CNT", "*i32"), ("BASE", "*fp32"), ("SCALE", "*fp32"),
     ("PRE_IN", "*fp32"), ("NW", "*bf16"), ("OUT", "*bf16"), ("PRE_OUT", "*fp32"), ("eps", "fp32"), ("hc_eps", "fp32")]
base_consts = {"WIDE": 20480, "D": 5120, "NB": 40, "SUB": 128, "MB": 8, "ITERS": 20, "BLOCK": 1024}
added = []
for world in (1, 2):
    for rb, rows_kind in ((1, "one"), (2, "i32"), (3, "i32"), (4, "i32"), (8, "i32"), (8, "i32d16")):
        fn = C._hc_chain
        sig, constexprs, attrs, params = {}, {}, {}, []
        consts = dict(base_consts, RB=rb, WORLD=world)
        for i, name in enumerate(fn.arg_names):
            if name in consts:
                sig[name] = "constexpr"
                constexprs[name] = consts[name]
            elif name == "rows":
                if rows_kind == "one":
                    sig[name] = "constexpr"
                    constexprs[name] = 1
                    consts["rows"] = 1
                else:
                    sig[name] = "i32"
                    d16 = rows_kind == "i32d16"
                    if d16:
                        attrs[(i,)] = [["tt.divisibility", 16]]
                    params.append({"name": name, "type": "i32", "div16": d16, "nospec": False})
            elif name == "RS":  # rows * D: a multiple of 16 at every launch, as the kit's _hc_post_only takes it
                sig[name] = "i32"
                attrs[(i,)] = [["tt.divisibility", 16]]
                params.append({"name": name, "type": "i32", "div16": True, "nospec": False})
            else:
                ty = dict(P)[name]
                sig[name] = ty
                if ty.startswith("*"):
                    attrs[(i,)] = [["tt.divisibility", 16]]
                params.append({"name": name, "type": ty, "div16": ty.startswith("*"), "nospec": False})
        k = triton.compile(ASTSource(fn, sig, constexprs=constexprs, attrs=attrs), target=TARGET,
                           options={"num_warps": 8})
        hsaco = k.asm["hsaco"]
        sha = hashlib.sha256(hsaco).hexdigest()
        (out / "hsaco" / f"{sha}.hsaco").write_bytes(hsaco)
        md = k.metadata
        stats = {}
        for line in k.asm.get("amdgcn", "").splitlines():
            line = line.strip()
            for key in (".vgpr_count:", ".sgpr_count:", ".vgpr_spill_count:", ".sgpr_spill_count:",
                        ".private_segment_fixed_size:", ".group_segment_fixed_size:"):
                if line.startswith(key):
                    stats[key.strip(".:")] = int(line.split(":")[1].strip().split()[0])
        e = {"fn": "_hc_chain", "hash": sha, "name": md.name, "num_warps": md.num_warps, "num_ctas": md.num_ctas,
             "shared": md.shared, "global_scratch": getattr(md, "global_scratch_size", 0),
             "global_align": getattr(md, "global_scratch_align", 1),
             "profile_scratch": getattr(md, "profile_scratch_size", 0), "pdl": False, "params": params,
             "consts": {kk: {"int": vv} if isinstance(vv, int) else vv for kk, vv in consts.items()},
             "arch": "gfx1151", "warp_size": md.warp_size, "block": md.num_warps * md.warp_size,
             "file": f"hsaco/{sha}.hsaco", "options": {"num_warps": 8},
             "isa_stats": stats, "recorded_grid": [1, 41, 3], "origin": "SPEED-FUSE-1 hc_chain.py (tools/dsv41-hip/exl3)"}
        m["kernels"].append(e)
        added.append((world, rb, rows_kind, sha[:12], stats))
        print("_hc_chain WORLD", world, "RB", rb, rows_kind, sha[:12], stats, "shared", md.shared, flush=True)
m["kdec2_chain_added"] = {"fn": "_hc_chain", "count": len(added), "source_sha256": sha_file(C.__file__),
                          "sources": {"hc_chain.py": sha_file(C.__file__), "hc_fuse.py": sha_file(F.__file__),
                                      "tri_kernels.py": sha_file(K.__file__)},
                          "triton": TRITON, "triton_wheel": os.environ.get("TRITON_WHEEL", ""),
                          "python": sys.version.split()[0], "kit_triton": kit_tri, "kit_python": kit_py,
                          "toolchain_match": match, "check_ref": ref}
json.dump(m, open(out / "aot.json", "w"), indent=1)
print("kit", out, "kernels", len(m["kernels"]))
