"""`flerp assemble` for a recipe whose host language is Zig (`[build] system = "zig"`, docs/SPEC-PROPOSALS.md).

A Zig origin is not a list of compile units: its build graph is a `build.zig`, a module is an import closure, and
the kernel kit is made by the same graph (hipcc per `.hip` source, then the origin's kit tool). So the assembly is
a source tree, not a CMake file:

    build/<recipe>/tree/         every file of the recipe's module closure at its origin-relative path (copies,
                                 byte for byte; the catalogue is the only source)
    build/<recipe>/assembly.json modules, the owner and sha256 of every tree file, the build command, the checks
    build/<recipe>/build.sh      verifies prebuilt kits against the catalogue, then runs the origin's own build
                                 steps with the toolchain of toolchains.toml and links strixflerp-<recipe>

The paths handed to hipcc are relative to the tree, as in the origin, which is what makes the code objects
reproducible (see docs/HARVEST.md, "the kit hashes"). Before anything is written the tree is checked: no path owned
twice, every relative `@import` / `@embedFile` resolves inside the tree, and a file only reaches into modules its
own module requires.

Recipe keys read here:

    [build]
    system   = "zig"
    zig      = "zig-0.17"                       # a toolchains.toml entry of kind "zig" (or an absolute path to
                                                # the binary); FLERP_ZIG overrides at build time
    root     = "<module id>"                    # the module that holds build.zig; it must be in the assembly
    steps    = ["native-hip"]                   # zig build steps
    options  = ["-Dname=value", ...]            # {rocm} = the toolchain's rocm_root, {prebuilt:<module id>} = that
                                                # module's prebuilt kit directory (see prebuilt below)
    build_file = "flerp.build.zig"              # optional: a build file of an assembled module, used with --build-file
    headers  = ["include/flerp/executor.h"]     # optional: repository files copied into the tree at the same path
    c_link   = "flerp/link.zig"                 # optional: a tree path the assembler writes the C and C++ compile
                                                # units of every binary to (file, flags, include roots, from the
                                                # manifests of the binary's module closure), for the build file to
                                                # compile and link into the Zig executable: how a C or C++ module
                                                # (a front end) and a Zig-exported one (an executor) become one binary
    [build.prebuilt]
    "<module id>" = "/abs/dir"                  # where a `[[prebuilt]]` kit of that module lives on the build host;
                                                # FLERP_PREBUILT_<NAME> overrides (NAME = last id part, upper case)
    [build.installs]                            # binary name in [binaries] -> path under tree/zig-out
    [build.hipcc_output_pins]                   # optional: code object name -> the cache directory name the
    [build.overlay]                             # optional: derived module (SPEC.md section 5) -> its root in the
    "executor.flerp.x" = "flerp/x"              # tree. Each of its files under that root is laid over the file at
                                                # the origin-relative path (root removed), which must be a file of
                                                # a module the derived one requires; the catalogue is unchanged and
                                                # assembly.json records both hashes
    "glue.hsaco" = "0b20..."                    # reference build compiled it into. hipcc then runs through
                                                # build/<recipe>/hipcc-pinned ({hipcc} in options), which compiles
                                                # to ./.zig-cache/o/<pin>/<name> and copies the object to where Zig
                                                # asked. clang hashes its argument strings, the -o path among them,
                                                # into the HIP compile unit id (__hip_cuid_<hash>), and Zig derives
                                                # that path from the absolute location of the checkout, so without
                                                # the pin every checkout gets different object bytes.
"""
import hashlib
import json
import os
import re
import shutil
import sys

SYSTEM = "zig"
_REF = re.compile(r'@(import|embedFile)\(\s*"([^"\\]+)"\s*\)')
_LINE_COMMENT = re.compile(r"(?m)^\s*(//|\\\\).*$")      # comments and multiline string literals
BUILD_KEYS = {"system": str, "zig": str, "root": str, "steps": list, "options": list, "build_file": str,
              "headers": list, "prebuilt": dict, "installs": dict, "hipcc_output_pins": dict, "c_link": str,
              "overlay": dict}
C_LINK_STD = {"c": "-std=c11", "c++": "-std=c++20"}     # C++20 as the CMake assembly compiles it


def zig_binary(cat, recipe):
    """The zig binary of a recipe: [build] zig names a toolchains.toml entry of kind "zig" (root + bin), or is
    itself a path. None when it is neither."""
    name = recipe.data.get("build", {}).get("zig")
    tc = cat.toolchains.get(name) if isinstance(name, str) else None
    if isinstance(tc, dict) and tc.get("kind") == "zig" and isinstance(tc.get("root"), str):
        return os.path.join(tc["root"], tc.get("bin", "zig"))
    return name if isinstance(name, str) and os.path.isabs(name) else None


def is_zig(recipe):
    b = recipe.data.get("build")
    return isinstance(b, dict) and b.get("system") == SYSTEM


def handles(recipe, module):
    """True when this assembler compiles `module`'s sources for `recipe` (the hook `flerp check` asks)."""
    return is_zig(recipe) and module.lang in ("zig", "hip-c++", "c", "c++", "hsaco", "triton-aot")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def references(text):
    """Relative files a Zig source names: [(kind, path)] for @import("x.zig") and @embedFile("x")."""
    out = []
    for kind, target in _REF.findall(_LINE_COMMENT.sub("", text)):
        if (kind == "import" and not target.endswith((".zig", ".zon"))) or ("/" not in target and "." not in target):
            continue                       # a named module of the build graph (std, gpu, lanes, dsv41_registry, ...)
        out.append((kind, target))
    return out


def tree_dir(cat, m, d):
    """A module's include root as a tree path: the tree holds a module's src/ at its root, and the repository's
    include/ (the [build] headers) at include/. None when the root is neither."""
    d = os.path.normpath(os.path.join(m.dir, d))
    src = os.path.join(m.dir, "src")
    if d == src or d.startswith(src + os.sep):
        return os.path.relpath(d, src)
    if d == os.path.join(cat.root, "include"):
        return "include"
    return None


def c_link(cat, recipe, asm):
    """The text of the [build] c_link file: for each binary, the C and C++ compile units of its module closure
    with their flags and include roots (a unit sees the roots of its own module and of that module's requirement
    closure, as in the CMake assembly). Returns (text, errors)."""
    errors, out = [], ["// Generated by `flerp assemble %s` from the module manifests. Do not edit." % recipe.name,
                       "pub const Unit = struct { file: []const u8, flags: []const []const u8, include: []const []const u8, "
                       "system: []const []const u8 };",
                       "pub const Binary = struct { name: []const u8, units: []const Unit };", "pub const binaries = [_]Binary{"]
    zs = lambda items: "&.{" + ", ".join(json.dumps(i) for i in items) + "}"
    for name, (_, _, ids) in sorted(asm.binaries.items()):
        out.append("    .{ .name = %s, .units = &.{" % json.dumps(name))
        for mid in ids:
            m = cat.by_id[mid]
            if m.lang not in C_LINK_STD or not m.sources:
                continue
            normal, system = [], []
            for dep in [mid] + asm.deps(mid):
                dm = cat.by_id[dep]
                for d in dm.includes:
                    t = tree_dir(cat, dm, d)
                    if t is None:
                        errors.append(f"{dep}: include root {d} is outside the tree ([build] c_link)")
                    elif t not in normal + system:
                        (system if dm.build.get("system") else normal).append(t)
            flags = ([f"-std=c{m.build['c_std']}"] if m.lang == "c" and m.build.get("c_std") else [C_LINK_STD[m.lang]])
            flags += ["-D" + d for d in m.build.get("defines", [])] + list(m.build.get("options", []))
            for src in m.sources:
                out.append("        .{ .file = %s, .flags = %s, .include = %s, .system = %s }," % (
                    json.dumps(src[len("src/"):]), zs(flags), zs(normal), zs(system)))
        out.append("    } },")
    out.append("};")
    return "\n".join(out) + "\n", errors


def overlaid(cat, recipe, asm):
    """[build.overlay]: ({tree path: (derived module, its file, the module it replaces a file of, that file)},
    errors). A derived module's file flerp/<name>/<origin path> takes the tree path <origin path>."""
    out, errors = {}, []
    for mid, root in sorted(recipe.data.get("build", {}).get("overlay", {}).items()):
        if mid not in asm.modules or not isinstance(root, str):
            errors.append(f"[build.overlay] {mid} is not a module of the assembly")
            continue
        m, pre, allowed, n = cat.by_id[mid], root.strip("/") + "/", set(asm.deps(mid)), 0
        for rel in m.files():
            if not rel.startswith(pre):
                continue
            n += 1
            dest = rel[len(pre):]
            owner = next((o for o in allowed if dest in cat.by_id[o].files()), None)
            if owner is None:
                errors.append(f"[build.overlay] {rel} ({mid}) would replace {dest}, which is in no module {mid} requires")
            elif dest in out:
                errors.append(f"[build.overlay] {dest} is replaced by both {out[dest][0]} and {mid}")
            else:
                out[dest] = (mid, os.path.join(m.dir, "src", rel), owner, os.path.join(cat.by_id[owner].dir, "src", dest))
        if n == 0:
            errors.append(f"[build.overlay] {mid} has no file under {root}")
    return out, errors


def plan(cat, recipe, asm, generated=None):
    """The tree as {tree path: (module id, absolute source)} plus errors and warnings. Nothing is written.
    `generated`: {tree path: absolute file} of files the assembler itself wrote ([build] c_link)."""
    errors, warnings, tree = [], [], {}
    over, oerrors = overlaid(cat, recipe, asm)
    errors += oerrors
    moved = {(mid, src) for mid, src, _, _ in over.values()}
    for mid in asm.modules:
        m = cat.by_id[mid]
        for rel in m.files():
            if (mid, os.path.join(m.dir, "src", rel)) in moved:
                continue                   # a derived module's copy: it goes to the path it replaces, below
            if rel in tree:
                errors.append(f"{rel}: in both {tree[rel][0]} and {mid}")
            tree[rel] = (mid, os.path.join(m.dir, "src", rel))
    for dest, (mid, src, _, _) in over.items():
        tree[dest] = (mid, src)
    build = recipe.data.get("build", {})
    for rel in build.get("headers", []):
        src = os.path.join(cat.root, rel)
        if not os.path.isfile(src):
            errors.append(f"[build] headers: {rel} is not a file of the repository")
        elif rel in tree:
            errors.append(f"[build] headers: {rel} is also a file of {tree[rel][0]}")
        else:
            tree[rel] = ("-", src)
    for rel, src in (generated or {}).items():
        if rel in tree:
            errors.append(f"[build] c_link: {rel} is also a file of {tree[rel][0]}")
        else:
            tree[rel] = ("-", src)
    root = build.get("root")
    if root not in asm.modules:
        errors.append(f"[build] root {root} is not a module of the assembly (add it under [include])")
    build_file = build.get("build_file", "build.zig")
    if build_file not in tree:
        errors.append(f"build file {build_file} is in no module of the assembly")

    # A file may name files of its own module, of the modules its module requires, and of the modules that
    # declare the same `[build] closure`: one Zig import closure that the catalogue cuts apart for the record,
    # because its files import each other in both directions (`requires` may not form a cycle).
    group = {mid: cat.by_id[mid].build.get("closure") for mid in asm.modules}
    reach = {mid: set(asm.deps(mid)) | {mid} | {o for o in asm.modules if group[mid] and group[o] == group[mid]}
             for mid in asm.modules}
    for rel, (mid, src) in sorted(tree.items()):
        if not rel.endswith(".zig"):
            continue
        with open(src, errors="replace") as f:
            refs = references(f.read())
        for kind, target in refs:
            dest = os.path.normpath(os.path.join(os.path.dirname(rel), target))
            if dest not in tree:
                # no exception for test files: Zig 0.17 loads every file a compiled file imports, `test` blocks
                # included ("unable to load ...: FileNotFound"), although it never analyses them in a build
                errors.append(f"{rel} ({mid}): @{kind}(\"{target}\") is in no module of the assembly")
                continue
            owner = tree[dest][0]
            if mid != "-" and owner != "-" and owner not in reach[mid]:
                errors.append(f"{rel} ({mid}) reaches into {owner} ({dest}) without requiring it")
    return tree, errors, warnings


def prebuilt_dirs(cat, recipe, asm):
    """{module id: (directory on the build host, the module)} for [build.prebuilt]."""
    out = {}
    for mid, where in recipe.data.get("build", {}).get("prebuilt", {}).items():
        if mid in asm.modules:
            out[mid] = (where, cat.by_id[mid])
    return out


def command(cat, recipe, asm):
    """(argv with {zig} first, env additions, env removals) for the build, paths relative to the tree."""
    build = recipe.data.get("build", {})
    rocm = cat.toolchains.get(recipe.toolchain, {}).get("rocm_root", "")
    subs = {"rocm": "${FLERP_ROCM_ROOT}",
            "hipcc": "${here}/hipcc-pinned" if build.get("hipcc_output_pins") else "${FLERP_ROCM_ROOT}/bin/hipcc"}
    for mid in prebuilt_dirs(cat, recipe, asm):
        subs["prebuilt:" + mid] = "${" + prebuilt_env(mid) + "}"
    argv = ["build"]
    if build.get("build_file"):
        argv += ["--build-file", build["build_file"]]
    argv += list(build.get("steps", []))
    for opt in build.get("options", []):
        argv.append(re.sub(r"\{([^}]+)\}", lambda m: subs.get(m.group(1), m.group(0)), opt))
    return argv, rocm


def prebuilt_env(mid):
    return "FLERP_PREBUILT_" + re.sub(r"[^A-Z0-9]", "_", mid.split(".")[-1].upper())


HIPCC_PINNED = r'''#!/bin/bash
# Generated by `flerp assemble @RECIPE@`. hipcc with the output path string of the reference build.
# clang derives each code object's compile unit id (the __hip_cuid_<hash> symbol) from its argument strings, the
# -o path included; Zig names that path after a digest that covers the checkout's absolute location. Compiling to
# the reference build's path string (relative to the tree, where Zig runs this) and copying the object to the path
# Zig asked for gives the reference build's bytes. Every other argument passes through untouched.
set -euo pipefail
real="${FLERP_ROCM_ROOT:?}/bin/hipcc"
args=("$@"); oi=-1
for ((i = 0; i < ${#args[@]}; i++)); do [ "${args[i]}" = "-o" ] && oi=$((i + 1)); done
[ "$oi" -ge 0 ] || exec "$real" "$@"
out=${args[oi]}
case "$(basename "$out")" in
@CASES@
  *) exec "$real" "$@";;
esac
pinned="./.zig-cache/o/$pin/$(basename "$out")"
[ "$pinned" = "$out" ] && exec "$real" "$@"
mkdir -p "$(dirname "$pinned")"
args[oi]=$pinned
"$real" "${args[@]}"
cp "$pinned" "$out"
'''

BUILD_SH = r'''#!/bin/bash
# Generated by `flerp assemble @RECIPE@`. Do not edit: change the recipe or a module manifest and assemble again.
# Builds tree/ with the origin's own Zig build graph. Usage: build.sh [-jN] [extra zig build arguments]
set -euo pipefail
export here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/@ROOT_REL@" && pwd)
zig=${FLERP_ZIG:-@ZIG@}
export FLERP_ROCM_ROOT=${FLERP_ROCM_ROOT:-@ROCM@}
jobs=-j${FLERP_JOBS:-8}
case "${1:-}" in -j*) jobs=$1; shift;; esac
[ -x "$zig" ] || { echo "build.sh: no zig at $zig (set FLERP_ZIG)" >&2; exit 2; }
[ -x "$FLERP_ROCM_ROOT/bin/hipcc" ] || { echo "build.sh: no hipcc under $FLERP_ROCM_ROOT (set FLERP_ROCM_ROOT)" >&2; exit 2; }
# the tree must still be what `flerp assemble` wrote from the catalogue
python3 "$root/flerplib/assemble_zig.py" verify-tree "$here"
@PREBUILT@
cd "$here/tree"
# Zig installs over an existing prefix and never removes a file it no longer makes: an earlier build's code objects
# would stay in the kit directory next to the manifest that does not name them.
rm -rf "$here/tree/zig-out"
# hipcc would append these after the explicit device flags (the origin's zig_build.sh removes them too); nice 19:
# the build host serves other work. The caches are this assembly's own, as the origin's wrapper keeps them per tree.
env -u HIPCC_COMPILE_FLAGS_APPEND -u HIPCC_LINK_FLAGS_APPEND \
    ZIG_LOCAL_CACHE_DIR="$here/tree/.zig-cache" ZIG_GLOBAL_CACHE_DIR="$here/.zig-global" \
    nice -n 19 "$zig" @ARGV@ "$jobs" "$@"
@LINKS@
python3 "$root/flerplib/assemble_zig.py" record "$here"
'''


def generate(cat, recipe, out_dir, licence=None):
    """Writes out_dir/{tree/, assembly.json, build.sh}. Returns (assembly, errors) like assemble.generate."""
    from .catalog import licence_matches
    from .resolve import Assembly
    asm = Assembly(cat, recipe)
    errors = list(asm.errors)
    build = recipe.data.get("build", {})
    if not isinstance(cat.toolchains.get(recipe.toolchain), dict):
        errors.append(f"toolchain '{recipe.toolchain}' is not in toolchains.toml")
    for key, typ in (("zig", str), ("root", str), ("steps", list)):
        if not isinstance(build.get(key), typ) or not build.get(key):
            errors.append(f"[build] {key} is missing or not a non-empty {typ.__name__}")
    for mid in asm.modules:
        m = cat.by_id[mid]
        if m.sources and not handles(recipe, m):
            errors.append(f"{mid}: lang {m.lang} cannot be assembled by the zig assembler")
        if licence and not licence_matches(m.spdx, licence):
            errors.append(f"{mid}: licence '{m.spdx}' is outside --licence {licence}")
    zig = zig_binary(cat, recipe)
    if zig is None and isinstance(build.get("zig"), str):
        errors.append(f"[build] zig '{build['zig']}' is neither a toolchains.toml entry of kind \"zig\" nor an absolute path")
    if errors:
        return asm, errors
    out_dir = os.path.abspath(out_dir)
    generated = {}
    if build.get("c_link"):
        text, cerrors = c_link(cat, recipe, asm)
        if cerrors:
            return asm, cerrors
        os.makedirs(os.path.join(out_dir, "generated"), exist_ok=True)
        generated[build["c_link"]] = os.path.join(out_dir, "generated", os.path.basename(build["c_link"]))
        with open(generated[build["c_link"]], "w") as f:
            f.write(text)
    tree, perrors, warnings = plan(cat, recipe, asm, generated)
    asm.warnings += warnings
    installs = build.get("installs", {})
    for name in asm.binaries:
        if name not in installs:
            perrors.append(f"[build.installs] has no entry for binary {name}")
    if perrors:
        return asm, perrors

    out_dir = os.path.abspath(out_dir)
    tdir = os.path.join(out_dir, "tree")
    os.makedirs(tdir, exist_ok=True)
    keep = (".zig-cache", "zig-out")                 # build products of an earlier build of this tree
    for d, dirs, names in os.walk(tdir):
        if d == tdir:
            dirs[:] = [x for x in dirs if x not in keep]
        for n in names:
            rel = os.path.relpath(os.path.join(d, n), tdir)
            if rel not in tree:
                os.remove(os.path.join(d, n))        # a file of an earlier assembly that is no longer in the closure
    files = {}
    over = overlaid(cat, recipe, asm)[0]
    for rel, (mid, src) in sorted(tree.items()):
        dst = os.path.join(tdir, rel)
        digest = sha256(src)
        files[rel] = {"module": mid, "sha256": digest}
        if rel in over:                              # a derived module's copy in the place of an origin file
            files[rel]["over"] = {"module": over[rel][2], "sha256": sha256(over[rel][3])}
        if os.path.isfile(dst) and sha256(dst) == digest:
            continue                                 # unchanged: keep its mtime so Zig's cache stays warm
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        shutil.copymode(src, dst)

    argv, rocm = command(cat, recipe, asm)
    pre = prebuilt_dirs(cat, recipe, asm)
    # Zig's cache names a step's output after its arguments, not after what a wrapped compiler wrote there: a kit
    # directory made before the build command or the output pins changed keeps the earlier code objects beside the
    # new ones. A changed command starts from an empty cache.
    try:
        with open(os.path.join(out_dir, "assembly.json")) as f:
            old = json.load(f)
    except (OSError, ValueError):
        old = None
    if old is not None and (old.get("zig_argv") != argv or old.get("hipcc_output_pins") != build.get("hipcc_output_pins", {})):
        shutil.rmtree(os.path.join(tdir, ".zig-cache"), ignore_errors=True)
    pre_lines = []
    for mid, (where, m) in sorted(pre.items()):
        env = prebuilt_env(mid)
        pre_lines.append(f'export {env}=${{{env}:-{where}}}\n'
                         f'python3 "$root/flerplib/assemble_zig.py" verify-prebuilt "${env}" "$root/{cat.rel(m.dir)}"')
    link_lines = [f'ln -sfn "tree/zig-out/{installs[name]}" "$here/{name}"' for name in asm.binaries]
    sub = {"RECIPE": recipe.name, "ROOT_REL": os.path.relpath(cat.root, out_dir), "ZIG": zig, "ROCM": rocm,
           "PREBUILT": "\n".join(pre_lines), "ARGV": " ".join(f'"{a}"' for a in argv), "LINKS": "\n".join(link_lines)}
    script = re.sub(r"@([A-Z_]+)@", lambda m: sub[m.group(1)], BUILD_SH)
    with open(os.path.join(out_dir, "build.sh"), "w") as f:
        f.write(script)
    os.chmod(os.path.join(out_dir, "build.sh"), 0o755)

    pins = build.get("hipcc_output_pins", {})
    wrapper = os.path.join(out_dir, "hipcc-pinned")
    if pins:
        cases = "\n".join(f"  {name}) pin={pin};;" for name, pin in sorted(pins.items()))
        with open(wrapper, "w") as f:
            f.write(HIPCC_PINNED.replace("@RECIPE@", recipe.name).replace("@CASES@", cases))
        os.chmod(wrapper, 0o755)
    elif os.path.exists(wrapper):
        os.remove(wrapper)

    manifest = {"recipe": recipe.name, "system": SYSTEM, "toolchain": recipe.toolchain, "zig": zig,
                "zig_argv": argv, "steps": build.get("steps", []), "bindings": dict(recipe.bindings),
                "binaries": {n: {"main": main, "entry": entry, "modules": ids, "installed": installs[n]}
                             for n, (main, entry, ids) in asm.binaries.items()},
                "modules": {mid: {"dir": cat.rel(cat.by_id[mid].dir), "spdx": cat.by_id[mid].spdx,
                                  "files": len(cat.by_id[mid].files())} for mid in asm.modules},
                "prebuilt": {mid: {"where": where, "env": prebuilt_env(mid)} for mid, (where, _) in pre.items()},
                "hipcc_output_pins": pins, "files": files, "warnings": warnings}
    with open(os.path.join(out_dir, "assembly.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"zig assembly: {len(files)} files -> {os.path.relpath(tdir)}; build: {os.path.relpath(out_dir)}/build.sh")
    return asm, []


# --- build-time helpers, run by build.sh (this file as a script: no package imports) ---------------------------
def verify_tree(out_dir):
    with open(os.path.join(out_dir, "assembly.json")) as f:
        files = json.load(f)["files"]
    tdir, bad = os.path.join(out_dir, "tree"), []
    for rel, rec in files.items():
        p = os.path.join(tdir, rel)
        if not os.path.isfile(p) or sha256(p) != rec["sha256"]:
            bad.append(rel)
    extra = []
    for d, dirs, names in os.walk(tdir):
        if d == tdir:
            dirs[:] = [x for x in dirs if x not in (".zig-cache", "zig-out")]
        extra += [r for r in (os.path.relpath(os.path.join(d, n), tdir) for n in names) if r not in files]
    for rel in bad:
        print(f"verify-tree: {rel} is missing or differs from the catalogue", file=sys.stderr)
    for rel in extra:
        print(f"verify-tree: {rel} is not a file of the assembly", file=sys.stderr)
    if bad or extra:
        return 1
    print(f"verify-tree: {len(files)} files match assembly.json")
    return 0


def verify_prebuilt(kit_dir, module_dir):
    """A prebuilt kit directory against the module's record: prebuilt/SHA256SUMS names every object (a file is
    named by its sha256) and prebuilt/aot.json is the kit's manifest, byte for byte."""
    sums_path = os.path.join(module_dir, "prebuilt", "SHA256SUMS")
    want = {}
    with open(sums_path) as f:
        for line in f:
            if line.strip():
                digest, name = line.split()[:2]
                want[name.lstrip("*")] = digest
    bad = 0
    for name, digest in sorted(want.items()):
        p = os.path.join(kit_dir, name)
        if not os.path.isfile(p):
            p = os.path.join(kit_dir, "hsaco", os.path.basename(name))
        if not os.path.isfile(p) or sha256(p) != digest:
            print(f"verify-prebuilt: {name} is missing or differs in {kit_dir}", file=sys.stderr)
            bad += 1
    have = [n for n in os.listdir(os.path.join(kit_dir, "hsaco"))] if os.path.isdir(os.path.join(kit_dir, "hsaco")) else []
    extra = sorted(set(have) - {os.path.basename(n) for n in want})
    for n in extra:
        print(f"verify-prebuilt: {n} is in {kit_dir}/hsaco but not in the catalogue record", file=sys.stderr)
    for name in ("aot.json",):
        ours, theirs = os.path.join(module_dir, "prebuilt", name), os.path.join(kit_dir, name)
        if os.path.isfile(ours) and (not os.path.isfile(theirs) or sha256(ours) != sha256(theirs)):
            print(f"verify-prebuilt: {name} differs from the catalogue record", file=sys.stderr)
            bad += 1
    if bad or extra:
        return 1
    print(f"verify-prebuilt: {len(want)} objects of {kit_dir} match {os.path.relpath(sums_path)}")
    return 0


def record(out_dir):
    """After a build: sha256 of everything installed under tree/zig-out, written to built.json."""
    zout, out = os.path.join(out_dir, "tree", "zig-out"), {}
    for d, _, names in os.walk(zout):
        for n in names:
            p = os.path.join(d, n)
            out[os.path.relpath(p, zout)] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
    with open(os.path.join(out_dir, "built.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"record: {len(out)} installed files -> {os.path.relpath(os.path.join(out_dir, 'built.json'))}")
    return 0


if __name__ == "__main__":
    cmds = {"verify-tree": verify_tree, "verify-prebuilt": verify_prebuilt, "record": record}
    if len(sys.argv) < 3 or sys.argv[1] not in cmds:
        sys.exit("usage: assemble_zig.py verify-tree DIR | verify-prebuilt KIT_DIR MODULE_DIR | record DIR")
    sys.exit(cmds[sys.argv[1]](*sys.argv[2:]))
