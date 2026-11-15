"""`flerp check`: every manifest, origin file and recipe against SPEC.md. Errors fail the check; warnings do not."""
import os
import re

from . import assemble_cmake, assemble_make, assemble_zig, includes as inc
import fnmatch
import hashlib
import json

from .catalog import (COMPILED, GATE_COMPILE, INTERFACE, KINDS, LANGS, ORIGINS, PREBUILT_LANGS, STATUSES, USES,
                      overlap_class, spdx_ids)
from .resolve import Assembly

MODULE_KEYS = {
    "": {"id": str, "kind": str, "summary": str},
    "provenance": {"origin": str, "repo": str, "commit": str, "paths": list, "authors": list, "modified": bool},
    "licence": {"spdx": str},
    "build": {"lang": str, "sources": list, "includes": list},
    "gate": {"parity": str, "bench": str, "status": str},
}
MODULE_OPTIONAL = {
    "": {"provenance": dict, "licence": dict, "build": dict, "gate": dict, "provides": list, "requires": list,
         "notes": str, "why": str, "design": str, "prebuilt": list, "adapter": dict},
    "provenance": {"branch": str, "reason": str, "upstream": str},
    "licence": {},
    "build": {"arch": str, "toolchain": str, "system": bool, "uses": list, "defines": list, "options": list,
              "c_std": int, "source_options": dict, "units": str, "variants": int, "closure": str},
    "gate": {"evidence": str, "compile": str, "reason": str},
}
ADAPTER_KEYS = {"for": str, "symbols": list, "code_object": list}
C_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PROVIDES_KEYS = ("interface", "api", "numeric", "regime")
SPDX_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]*$")
GATE_REF = re.compile(r"^(none|origin-test:.+|flerp:gate/.+)$")
ROCM_KINDS = ("therock", "rocm")      # a ROCm install (the default when an entry names no kind)
NOT_ROCM_KINDS = ("triton", "zig")
RECIPE_KEYS = {"name": str, "model": str, "reference": dict, "toolchain": str, "modules": dict, "binaries": dict}
RECIPE_OPTIONAL = {"include": dict, "deploy": dict, "gate": dict, "build": dict, "overlay": dict}


class Report:
    def __init__(self):
        self.errors, self.warnings = [], []

    def error(self, where, msg):
        self.errors.append(f"{where}: {msg}")

    def warn(self, where, msg):
        self.warnings.append(f"{where}: {msg}")


def _typed(rep, where, table_name, table, required, optional):
    label = f"[{table_name}] " if table_name else ""
    for key, typ in required.items():
        if key not in table:
            rep.error(where, f"{label}{key} is missing")
        elif not isinstance(table[key], typ) or (typ is str and not table[key].strip()):
            rep.error(where, f"{label}{key} must be a non-empty {typ.__name__}")
    for key, val in table.items():
        if key in required:
            continue
        if key not in optional:
            rep.warn(where, f"{label}{key} is not a key SPEC.md defines")
        elif not isinstance(val, optional[key]):
            rep.error(where, f"{label}{key} must be a {optional[key].__name__}")


def _str_list(rep, where, label, v):
    if isinstance(v, list) and not all(isinstance(x, str) and x for x in v):
        rep.error(where, f"{label} must be a list of non-empty strings")


def check_spdx(rep, where, expr):
    if not expr.strip():
        rep.error(where, "[licence] spdx is empty")
        return
    ids = spdx_ids(expr)
    if not ids:
        rep.error(where, f"[licence] spdx '{expr}' names no licence")
    for i in ids:
        if not SPDX_ID.match(i):
            rep.error(where, f"[licence] spdx: '{i}' is not an SPDX licence id")


def check_module(cat, m, rep):
    where = cat.rel(m.path)
    d = m.data
    _typed(rep, where, "", d, MODULE_KEYS[""], MODULE_OPTIONAL[""])
    for t in ("provenance", "licence", "build", "gate"):
        if not isinstance(d.get(t), dict):
            rep.error(where, f"[{t}] table is missing")
        else:
            _typed(rep, where, t, d[t], MODULE_KEYS[t], MODULE_OPTIONAL[t])

    # identity
    want = f"{m.dir_kind}.{m.dir_origin}.{m.dir_name}"
    if d.get("id") != want:
        rep.error(where, f"id is '{d.get('id')}', its directory says '{want}'")
    if m.kind not in KINDS:
        rep.error(where, f"kind '{m.kind}' is not one of {', '.join(KINDS)}")
    elif m.kind != m.dir_kind:
        rep.error(where, f"kind '{m.kind}' does not match its directory '{m.dir_kind}'")
    if m.dir_origin not in ORIGINS:
        rep.error(where, f"origin directory '{m.dir_origin}' is not one of {', '.join(ORIGINS)}")
    if m.origin != m.dir_origin:
        rep.error(where, f"[provenance] origin '{m.origin}' does not match its directory '{m.dir_origin}'")

    # provenance and files
    prov = d.get("provenance") if isinstance(d.get("provenance"), dict) else {}
    for key in ("paths", "authors"):
        _str_list(rep, where, f"[provenance] {key}", prov.get(key))
    if isinstance(prov.get("paths"), list) and not prov["paths"]:
        rep.error(where, "[provenance] paths is empty")
    if isinstance(prov.get("authors"), list) and not prov["authors"]:
        rep.error(where, "[provenance] authors is empty")
    if prov.get("modified") is True and not str(prov.get("reason", "")).strip():
        rep.error(where, "[provenance] modified = true needs a one-line reason")
    if len(set(m.paths)) != len(m.paths):
        rep.error(where, "[provenance] paths lists a file twice")
    have = m.src_files()
    listed = set()
    for p in m.paths:
        if os.path.isabs(p) or ".." in p.split("/"):
            rep.error(where, f"[provenance] path '{p}' must be origin-relative")
            continue
        hits = m.expand(p, have)
        if not hits:
            rep.error(where, f"[provenance] path '{p}' names nothing under src/")
        listed.update(hits)
    for f in have:
        if f not in listed:
            rep.error(where, f"src/{f} is not listed in [provenance] paths")

    # design record: a file beside module.toml, never under src/ (src/ stays byte-identical to the origin)
    if isinstance(d.get("design"), str):
        rel = d["design"]
        parts = rel.split("/")
        if not rel.strip() or os.path.isabs(rel) or ".." in parts:
            rep.error(where, f"design '{rel}' must be a module-relative path")
        elif parts[0] == "src":
            rep.error(where, f"design '{rel}' must not be under src/ (src/ holds only origin files)")
        elif not os.path.isfile(os.path.join(m.dir, rel)):
            rep.error(where, f"design '{rel}' does not exist beside module.toml")

    # licence
    if isinstance(d.get("licence"), dict) and isinstance(d["licence"].get("spdx"), str):
        check_spdx(rep, where, m.spdx)

    # build
    b = m.build
    if m.lang is not None and m.lang not in LANGS:
        rep.error(where, f"[build] lang '{m.lang}' is not one of {', '.join(LANGS)}")
    for key in ("sources", "includes", "uses", "defines", "options"):
        _str_list(rep, where, f"[build] {key}", b.get(key))
    for s in m.sources:
        if not s.startswith("src/") or s[len("src/"):] not in listed:
            rep.error(where, f"[build] source '{s}' is not src/<a path listed in [provenance] paths>")
        elif not os.path.isfile(os.path.join(m.dir, s)):
            rep.error(where, f"[build] source '{s}' does not exist")
        elif m.lang in COMPILED and not s.endswith(COMPILED[m.lang]):
            rep.error(where, f"[build] source '{s}' is not a {m.lang} source")
    if len(set(m.sources)) != len(m.sources):
        rep.error(where, "[build] sources lists a file twice")
    for i in m.includes:
        if not os.path.isdir(os.path.join(m.dir, i)):
            rep.error(where, f"[build] include directory '{i}' does not exist")
    if m.lang in ("hip-c++", "hsaco", "triton-aot") and not b.get("arch"):
        rep.error(where, f"[build] lang {m.lang} needs arch")
    tc = b.get("toolchain")
    if isinstance(tc, str) and tc not in cat.toolchains:
        rep.error(where, f"[build] toolchain '{tc}' is not in toolchains.toml")
    elif isinstance(tc, str) and isinstance(cat.toolchains[tc], dict) and cat.toolchains[tc].get("kind") == "zig":
        rep.error(where, f"[build] toolchain '{tc}' is a zig entry: a module names the ROCm install or Triton kit "
                         "its device code is built with; the zig binary is the recipe's ([build] zig)")
    for u in b.get("uses", []) if isinstance(b.get("uses"), list) else []:
        if u not in USES:
            rep.error(where, f"[build] uses '{u}' is not one of {', '.join(USES)}")

    # provides / requires
    if not m.provides:
        rep.error(where, "provides nothing: a module needs at least one [[provides]]")
    seen = set()
    for p in d.get("provides", []) if isinstance(d.get("provides"), list) else []:
        if not isinstance(p, dict):
            rep.error(where, "[[provides]] must be a table")
            continue
        iface = p.get("interface")
        for key in PROVIDES_KEYS:
            if not isinstance(p.get(key), str) or not p[key].strip():
                rep.error(where, f"[[provides]] {iface or '?'}: {key} is missing or empty")
        for key in p:
            if key not in PROVIDES_KEYS:
                rep.warn(where, f"[[provides]] {key} is not a key SPEC.md defines")
        if isinstance(iface, str):
            if not INTERFACE.match(iface):
                rep.error(where, f"[[provides]] '{iface}' is not <family>.<op>/<weight-format-or-dash>/<dtype-or-dash>")
            if iface in seen:
                rep.error(where, f"[[provides]] '{iface}' is listed twice")
            seen.add(iface)
    seen = set()
    for r in d.get("requires", []) if isinstance(d.get("requires"), list) else []:
        if not isinstance(r, dict) or len(r) != 1 or next(iter(r)) not in ("interface", "module"):
            rep.error(where, f"[[requires]] must have exactly one of interface or module, got {r}")
            continue
        key, val = next(iter(r.items()))
        if not isinstance(val, str) or not val:
            rep.error(where, f"[[requires]] {key} must be a non-empty string")
            continue
        if (key, val) in seen:
            rep.error(where, f"[[requires]] {key} '{val}' is listed twice")
        seen.add((key, val))
        if key == "module":
            if val == m.id:
                rep.error(where, "[[requires]] names the module itself")
            elif val not in cat.by_id:
                rep.error(where, f"[[requires]] module '{val}' is not in the catalogue")
        else:
            if not INTERFACE.match(val):
                rep.error(where, f"[[requires]] '{val}' is not a well-formed interface name")
            elif not cat.providers.get(val) and not any(overlap_class(p) == val for p in cat.providers):
                rep.error(where, f"[[requires]] interface '{val}' has no provider in the catalogue")

    # gate
    g = m.gate
    if isinstance(g.get("status"), str) and g["status"] not in STATUSES:
        rep.error(where, f"[gate] status '{g['status']}' is not one of {', '.join(STATUSES)}")
    prebuilt_lang = m.lang in PREBUILT_LANGS
    if g.get("status") == "verified" and not prebuilt_lang:
        rep.error(where, f"[gate] status 'verified' is for prebuilt kits ({', '.join(PREBUILT_LANGS)}); "
                         f"a {m.lang} module goes harvested -> builds")
    if g.get("status") == "builds" and prebuilt_lang:
        rep.error(where, "[gate] status 'builds' does not apply to a prebuilt kit: harvested -> verified -> parity")
    comp = g.get("compile")
    if isinstance(comp, str):
        if comp.startswith("via:"):
            unit = cat.by_id.get(comp[len("via:"):])
            if unit is None:
                rep.error(where, f"[gate] compile '{comp}' names a module that is not in the catalogue")
            elif not unit.sources and not unit.build.get("units"):
                rep.error(where, f"[gate] compile '{comp}': {unit.id} has no translation unit to compile it through")
            if m.sources:
                rep.error(where, "[gate] compile = via:<id> is for fragments: a module with [build] sources "
                                 "compiles on its own")
        elif comp not in GATE_COMPILE:
            rep.error(where, f"[gate] compile '{comp}' is not one of {', '.join(GATE_COMPILE)} or via:<module id>")
        elif comp == "prebuilt" and not d.get("prebuilt"):
            rep.error(where, "[gate] compile = prebuilt needs at least one [[prebuilt]] entry")
        elif comp == "header-only" and (m.sources or m.build.get("units")):
            rep.error(where, "[gate] compile = header-only, but the module has sources or units")
    check_build_extras(cat, m, rep, where)
    check_prebuilt(cat, m, rep, where)
    check_adapter(cat, m, rep, where)
    for key in ("parity", "bench"):
        if isinstance(g.get(key), str) and not GATE_REF.match(g[key]):
            rep.error(where, f"[gate] {key} '{g[key]}' is not none, origin-test:<path> or flerp:gate/<path>")
    if g.get("status") in STATUSES[1:]:
        ev = g.get("evidence")
        if not isinstance(ev, str) or not ev.startswith("gate/results/"):
            rep.error(where, f"[gate] status '{g['status']}' needs evidence = \"gate/results/<file>\"")
        elif not os.path.isfile(os.path.join(cat.root, ev)):
            rep.error(where, f"[gate] evidence '{ev}' does not exist")

    check_includes(cat, m, rep, where)


def check_build_extras(cat, m, rep, where):
    """[build] source_options, units and variants (SPEC.md section 5)."""
    b = m.build
    so = b.get("source_options")
    if isinstance(so, dict):
        for src, opts in so.items():
            if src not in m.sources:
                rep.error(where, f"[build] source_options names '{src}', which is not in [build] sources")
            if not isinstance(opts, list) or not all(isinstance(o, str) and o for o in opts):
                rep.error(where, f"[build] source_options for '{src}' must be a list of non-empty strings")
    if isinstance(b.get("variants"), int) and b["variants"] < 1:
        rep.error(where, "[build] variants must be at least 1")
    rel = b.get("units")
    if not isinstance(rel, str):
        return
    if m.sources:
        rep.error(where, "[build] has both sources and units: a module compiles files or units, not both")
    path = os.path.join(m.dir, rel)
    try:
        with open(path) as f:
            doc = json.load(f)
        units = doc["units"]
        assert isinstance(units, list) and units
    except (OSError, ValueError, KeyError, AssertionError, TypeError):
        rep.error(where, f"[build] units '{rel}' is not a JSON file with a non-empty \"units\" list")
        return
    origin_dir = os.path.join(cat.root, "catalog", m.dir_origin)
    names = set()
    for u in units:
        name = u.get("name") if isinstance(u, dict) else None
        if not isinstance(name, str) or not name or name in names:
            rep.error(where, f"[build] units: a unit has no name or repeats one ({name!r})")
            continue
        names.add(name)
        parts = u.get("parts")
        if not isinstance(parts, list) or not parts:
            rep.error(where, f"[build] units: unit '{name}' has no parts")
            continue
        for part in parts:
            if not isinstance(part, dict) or ("text" in part) == ("path" in part):
                rep.error(where, f"[build] units: unit '{name}': a part is either text or a path")
            elif "path" in part and not os.path.isfile(os.path.join(origin_dir, str(part["path"]))):
                rep.error(where, f"[build] units: unit '{name}' names {part['path']}, which is not under "
                                 f"catalog/{m.dir_origin}")
        if not isinstance(u.get("flags", []), list):
            rep.error(where, f"[build] units: unit '{name}': flags must be a list")


def check_prebuilt(cat, m, rep, where):
    """[[prebuilt]]: an object in the tree is named with its sha256 and size and must match; a kit that lives
    on the build host is named with where it is and a manifest in the tree."""
    pre = m.data.get("prebuilt")
    if not isinstance(pre, list):
        return
    for i, e in enumerate(pre):
        if not isinstance(e, dict):
            rep.error(where, f"[[prebuilt]] entry {i + 1} must be a table")
            continue
        if isinstance(e.get("file"), str):
            path = os.path.join(m.dir, e["file"])
            if not isinstance(e.get("sha256"), str) or not SHA256.match(e["sha256"]):
                rep.error(where, f"[[prebuilt]] {e['file']}: sha256 is missing or not 64 hex digits")
            if not os.path.isfile(path):
                rep.error(where, f"[[prebuilt]] {e['file']} does not exist")
                continue
            if isinstance(e.get("bytes"), int) and os.path.getsize(path) != e["bytes"]:
                rep.error(where, f"[[prebuilt]] {e['file']}: {os.path.getsize(path)} bytes, the manifest says {e['bytes']}")
            if isinstance(e.get("sha256"), str) and SHA256.match(e["sha256"]):
                with open(path, "rb") as f:
                    if hashlib.sha256(f.read()).hexdigest() != e["sha256"]:
                        rep.error(where, f"[[prebuilt]] {e['file']}: sha256 differs from the manifest's")
        elif isinstance(e.get("kit"), str):
            if not isinstance(e.get("where"), str) or not e["where"]:
                rep.error(where, f"[[prebuilt]] kit '{e['kit']}': where is missing")
            man, msha = e.get("manifest"), e.get("manifest_sha256")
            if man is None:      # a kit whose manifest stays on the build host is pinned by the manifest's hash
                if not isinstance(msha, str) or not SHA256.match(msha):
                    rep.error(where, f"[[prebuilt]] kit '{e['kit']}': needs manifest (a file of the module) or "
                                     "manifest_sha256")
            elif not isinstance(man, str) or not os.path.isfile(os.path.join(m.dir, man)):
                rep.error(where, f"[[prebuilt]] kit '{e['kit']}': manifest '{man}' is not a file of the module")
            elif isinstance(msha, str):
                with open(os.path.join(m.dir, man), "rb") as f:
                    if hashlib.sha256(f.read()).hexdigest() != e["manifest_sha256"]:
                        rep.error(where, f"[[prebuilt]] kit '{e['kit']}': manifest_sha256 differs from {man}")
        else:
            rep.error(where, f"[[prebuilt]] entry {i + 1} needs file (an object in the tree) or kit (one on the build host)")


def check_adapter(cat, m, rep, where):
    """[adapter]: a module that re-provides another module's launch API on top of a foreign provider
    (SPEC.md section 5, "Adapters")."""
    a = m.data.get("adapter")
    if not isinstance(a, dict):
        return
    for key, val in a.items():
        if key not in ADAPTER_KEYS:
            rep.warn(where, f"[adapter] {key} is not a key SPEC.md defines")
        elif not isinstance(val, ADAPTER_KEYS[key]):
            rep.error(where, f"[adapter] {key} must be a {ADAPTER_KEYS[key].__name__}")
    required = {r["module"] for r in m.requires if isinstance(r.get("module"), str)}
    target, symbols = m.overrides
    if "for" in a or "symbols" in a:
        if target not in cat.by_id:
            rep.error(where, f"[adapter] for = '{target}' is not a module of the catalogue")
        elif target not in required:
            rep.error(where, f"[adapter] for = '{target}' must also be in [[requires]] (its headers declare the API)")
        elif not cat.by_id[target].sources:
            rep.error(where, f"[adapter] for = '{target}' has no sources whose symbols could be taken over")
        if not symbols or not all(C_IDENT.match(x) for x in symbols) or len(set(symbols)) != len(symbols):
            rep.error(where, "[adapter] symbols must be a non-empty list of distinct C identifiers")
    seen = set()
    for c in m.code_objects:
        mod, unit, sym = c.get("module"), c.get("unit"), c.get("symbol")
        if not isinstance(sym, str) or not C_IDENT.match(sym) or sym in seen:
            rep.error(where, f"[[adapter.code_object]] symbol '{sym}' is not a C identifier or is used twice")
        seen.add(sym)
        src = cat.by_id.get(mod)
        if src is None:
            rep.error(where, f"[[adapter.code_object]] module '{mod}' is not in the catalogue")
            continue
        if mod not in required:
            rep.error(where, f"[[adapter.code_object]] module '{mod}' must also be in [[requires]]")
        cr = cat.compile_recipe(src)
        if cr is None or cr["product"] != "code-object":
            rep.error(where, f"[[adapter.code_object]] {mod}: its origin's compile recipe does not produce "
                             "device code objects")
        try:
            known = src.units()
        except (OSError, ValueError, KeyError, TypeError):
            known = {}
        if unit not in known:
            rep.error(where, f"[[adapter.code_object]] {mod} has no compile unit '{unit}'")


def unharvested(cat, origin, rel):
    """Whether ORIGIN.toml lists the origin-relative file as left out."""
    h = (cat.origin_data(origin) or {}).get("harvest")
    pats = h.get("unharvested", []) if isinstance(h, dict) else []
    return any(fnmatch.fnmatchcase(rel, p) or rel.startswith(p.rstrip("*").rstrip("/") + "/") and p.endswith("**")
               for p in pats if isinstance(p, str))


def check_includes(cat, m, rep, where):
    """Every header a C-family file includes from another module of the catalogue must come from a module this
    one requires (by id, or through an interface that module provides)."""
    allowed = {m.id}
    for r in m.requires:
        if isinstance(r.get("module"), str):
            allowed.add(r["module"])
        elif isinstance(r.get("interface"), str):
            allowed.update(p.id for p in cat.providers.get(r["interface"], []))
    same_origin = [o for o in cat.modules if o.dir_origin == m.dir_origin]
    # A module may also include headers of another origin's modules (an adapter does): those are found through
    # what it requires, transitively, and through nothing else.
    reach, todo = set(), [m]
    while todo:
        for r in todo.pop().requires:
            deps = ([cat.by_id[r["module"]]] if r.get("module") in cat.by_id else
                    cat.providers.get(r.get("interface"), []))
            for dep in deps:
                if dep.id not in reach:
                    reach.add(dep.id)
                    todo.append(dep)
    # Its own roots first, then those of the modules it reaches of another origin (what its build searches: an
    # adapter's header naming its target's, e.g. kernels/moe_grouped.hpp, resolves there even when a derived module of
    # this origin holds a file at the same path), then the rest of its origin (a hit there that it does not require
    # is the error below).
    roots = (m.include_dirs + [d for i in sorted(reach) if cat.by_id[i].dir_origin != m.dir_origin
                               for d in cat.by_id[i].include_dirs]
             + [d for o in same_origin if o is not m for d in o.include_dirs])
    owner = {}
    for o in same_origin:
        for d in o.include_dirs:
            # an include root outside the module's own directory (the repository's include/, where the ABIs
            # written here live) belongs to no module: any module may include from it
            if os.path.normpath(d).startswith(os.path.normpath(o.dir) + os.sep):
                owner[d] = o
    src = os.path.join(m.dir, "src")
    tree_srcs = [os.path.join(o.dir, "src") for o in same_origin if o is not m]
    for p in m.files():
        path = os.path.join(src, p)
        if not inc.is_c_family(p) or not os.path.isfile(path):
            continue
        for name, quoted in inc.scan(path):
            hit = inc.resolve(os.path.dirname(path), name, quoted, roots)
            if hit is None and quoted:
                hit = inc.resolve_tree(os.path.dirname(p), name, tree_srcs)
            if hit is None:
                # Not an error: it may sit under a conditional the build never takes. A module whose build is
                # proven (status past harvested) has shown exactly that, so only unproven modules are flagged.
                if quoted and m.gate.get("status") == "harvested":
                    left = " (ORIGIN.toml lists it as unharvested)" if unharvested(cat, m.dir_origin, name) else ""
                    rep.warn(where, f"src/{p} includes \"{name}\", which no module of origin {m.dir_origin} "
                                    f"holds{left}: the module cannot build as harvested")
                continue
            if hit.startswith(m.dir + os.sep):
                continue
            holder = next((o for d, o in owner.items() if hit.startswith(d + os.sep)), None)
            if holder is not None and holder.id not in allowed:
                rep.error(where, f"src/{p} includes {name} from {holder.id}, which is not in [[requires]]")


def check_catalogue(cat, rep, origin=None):
    for path, msg in cat.load_errors:
        if origin is None or path.startswith(os.path.join(cat.root, "catalog", origin) + os.sep):
            rep.error(cat.rel(path), msg)
    mods = [m for m in cat.modules if origin is None or m.dir_origin == origin]
    ids = {}
    for m in cat.modules:
        ids.setdefault(m.id, []).append(m)
    for mid, ms in ids.items():
        if len(ms) > 1 and any(m in mods for m in ms):
            rep.error(mid, "id is used by " + ", ".join(cat.rel(m.path) for m in ms))
    for m in mods:
        check_module(cat, m, rep)

    # a file of an origin lives in exactly one module
    for o in cat.origins():
        if origin is not None and o != origin:
            continue
        seen = {}
        for m in cat.modules:
            if m.dir_origin != o:
                continue
            for p in m.files():
                if p in seen:
                    rep.error(cat.rel(m.path), f"origin file {p} is also in {seen[p]}")
                seen[p] = m.id
        check_origin(cat, o, rep)


def check_origin(cat, origin, rep):
    where = cat.rel(cat.origin_file(origin))
    data = cat.origin_data(origin)
    if data is None:
        rep.error(where, "missing: every origin needs ORIGIN.toml (repo, commit, licence, toolchain)")
        return
    for key in ("repo", "commit"):
        if not isinstance(data.get(key), str) or not data[key]:
            rep.error(where, f"{key} is missing")
    named = data.get("origin", data.get("name"))
    if named not in (None, origin):
        rep.error(where, f"names itself '{named}', its directory says '{origin}'")
    lic = data.get("licence")
    if not isinstance(lic, dict) or not isinstance(lic.get("spdx"), str):
        rep.error(where, "[licence] spdx is missing")
    else:
        check_spdx(rep, where, lic["spdx"])
        for f in lic.get("files", []):
            if not os.path.isfile(os.path.join(cat.root, "catalog", origin, f)):
                rep.error(where, f"[licence] file '{f}' does not exist next to ORIGIN.toml")
    tc = data.get("toolchain")
    if not isinstance(tc, dict) or not tc:
        rep.error(where, "[toolchain] is missing: say what the origin was built with")
    # harvest: a table (mode = "modules" | "none", tool, unharvested), or the string "none" with harvest_reason.
    h = data.get("harvest")
    mode = h.get("mode", "modules") if isinstance(h, dict) else h
    has_modules = any(m.dir_origin == origin for m in cat.modules)
    if h is not None and mode not in ("modules", "none"):
        rep.error(where, f"harvest mode '{mode}' is not modules or none")
    if mode == "none":
        reason = h.get("reason") if isinstance(h, dict) else data.get("harvest_reason")
        if not isinstance(reason, str) or not reason.strip():
            rep.error(where, "harvest = none needs a reason (harvest_reason, or [harvest] reason)")
        if has_modules:
            rep.error(where, "harvest = none, but the catalogue holds modules of this origin")
    comp = data.get("compile")
    if comp is not None:
        if not isinstance(comp, dict) or not isinstance(comp.get("driver"), str) or not comp.get("arch_flag"):
            rep.error(where, "[compile] needs driver and arch_flag (SPEC.md section 5, origin compile recipes)")
    commit = data.get("commit")
    for m in cat.modules:
        if m.dir_origin == origin and isinstance(commit, str):
            mc = m.data.get("provenance", {}).get("commit") if isinstance(m.data.get("provenance"), dict) else None
            if isinstance(mc, str) and not (commit.startswith(mc) or mc.startswith(commit)):
                rep.warn(cat.rel(m.path), f"[provenance] commit {mc} differs from ORIGIN.toml's {commit}")


def check_recipe(cat, r, rep):
    where = cat.rel(r.path)
    for key, typ in RECIPE_KEYS.items():
        if not isinstance(r.data.get(key), typ) or r.data.get(key) in ("", {}):
            rep.error(where, f"{key} is missing or not a non-empty {typ.__name__}")
    for key, val in r.data.items():
        if key not in RECIPE_KEYS:
            if key not in RECIPE_OPTIONAL:
                rep.warn(where, f"{key} is not a key SPEC.md defines")
            elif not isinstance(val, RECIPE_OPTIONAL[key]):
                rep.error(where, f"{key} must be a {RECIPE_OPTIONAL[key].__name__}")
    if r.name != r.file_name:
        rep.error(where, f"name is '{r.name}', its file says '{r.file_name}'")
    if isinstance(r.toolchain, str) and r.toolchain not in cat.toolchains:
        rep.error(where, f"toolchain '{r.toolchain}' is not in toolchains.toml")
    elif isinstance(r.toolchain, str) and isinstance(cat.toolchains[r.toolchain], dict) and \
            cat.toolchains[r.toolchain].get("kind") in NOT_ROCM_KINDS:
        rep.error(where, f"toolchain '{r.toolchain}' is not a ROCm install (its kind is "
                         f"{cat.toolchains[r.toolchain]['kind']})")
    build = r.data.get("build")
    if isinstance(build, dict):          # SPEC.md section 7: a recipe built by its origin's build graph
        systems = {assemble_zig.SYSTEM: (assemble_zig.BUILD_KEYS, ("zig", "root", "steps")),
                   assemble_cmake.SYSTEM: (assemble_cmake.BUILD_KEYS, assemble_cmake.REQUIRED),
                   assemble_make.SYSTEM: (assemble_make.BUILD_KEYS, assemble_make.REQUIRED)}
        if build.get("system") not in systems:
            rep.error(where, "[build] system must be " + " or ".join(f'"{s}"' for s in systems) +
                             " (the build graphs flerp assemble drives)")
        keys, required = systems.get(build.get("system"), (assemble_zig.BUILD_KEYS, ()))
        for key, val in build.items():
            if key not in keys:
                rep.warn(where, f"[build] {key} is not a key SPEC.md defines")
            elif not isinstance(val, keys[key]):
                rep.error(where, f"[build] {key} must be a {keys[key].__name__}")
        for key in required:
            if not build.get(key):
                rep.error(where, f"[build] {key} is missing")
        if isinstance(build.get("zig"), str) and assemble_zig.zig_binary(cat, r) is None:
            rep.error(where, f"[build] zig '{build['zig']}' is neither a toolchains.toml entry of kind \"zig\" nor an "
                             "absolute path")
        if isinstance(build.get("root"), str) and build["root"] not in cat.by_id:
            rep.error(where, f"[build] root '{build['root']}' is not a module of the catalogue")
        for key in ("prebuilt",):
            for mid in build.get(key, {}) if isinstance(build.get(key), dict) else ():
                if mid not in cat.by_id:
                    rep.error(where, f"[build.{key}] '{mid}' is not a module of the catalogue")
        for name in build.get("installs", {}) if isinstance(build.get("installs"), dict) else ():
            if name not in r.binaries:
                rep.error(where, f"[build.installs] '{name}' is not a binary of the recipe")
    ref = r.data.get("reference")
    if isinstance(ref, dict) and not (isinstance(ref.get("engine"), str) and isinstance(ref.get("build"), str)):
        rep.error(where, "reference needs engine and build")
    for iface, mid in r.bindings.items():
        if not INTERFACE.match(iface):
            rep.error(where, f"[modules] '{iface}' is not a well-formed interface name")
        if not isinstance(mid, str):
            rep.error(where, f"[modules] {iface} must name a module id")
    if not isinstance(r.data.get("include", {}).get("modules", []), list):
        rep.error(where, "[include] modules must be a list of module ids")
    runner = r.gate.get("runner")
    if runner is not None and not (isinstance(runner, str) and runner.startswith("gate/")
                                   and os.path.isfile(os.path.join(cat.root, runner))):
        rep.error(where, f"[gate] runner '{runner}' is not a file under gate/")
    asm = Assembly(cat, r)
    for e in asm.errors:
        rep.error(where, e)
    if asm.overlays and (assemble_zig.is_zig(r) or assemble_cmake.is_cmake(r) or assemble_make.is_make(r)):
        rep.error(where, "[overlay] is for recipes flerp assemble generates the CMake of; a recipe built by its "
                         "origin's graph names derived modules under [build.overlay]")
    for oid, did in asm.overlays.items():
        from .assemble import overlay_sources
        for e in overlay_sources(cat, cat.by_id[oid], cat.by_id[did])[1]:
            rep.error(where, e)
    for w in asm.warnings:
        rep.warn(where, w)
    for mid in asm.modules:
        m = cat.by_id[mid]
        if m.sources and m.lang not in COMPILED and not assemble_zig.handles(r, m) and not assemble_cmake.handles(r, m) \
                and not assemble_make.handles(r, m):
            rep.error(where, f"{mid}: lang {m.lang} cannot be assembled yet")
        tc = m.build.get("toolchain")
        # A module built by its origin's compile recipe keeps its own toolchain: that is the point of the recipe.
        # So does a host C or C++ module in a Zig-hosted recipe: Zig's own clang compiles it ([build] c_link),
        # and the recipe's toolchain there only names where hipcc comes from.
        zig_host = assemble_zig.is_zig(r) and m.lang in ("c", "c++")
        if isinstance(tc, str) and tc != r.toolchain and m.sources and cat.compile_recipe(m) is None and not zig_host \
                and not assemble_cmake.is_cmake(r) and not assemble_make.is_make(r):
            rep.warn(where, f"{mid} names toolchain {tc}, the recipe builds with {r.toolchain}")
    crit = r.gate.get("criterion")
    if crit is not None and crit not in ("token-identity", "token-agreement"):
        rep.error(where, f"[gate] criterion '{crit}' is not token-identity or token-agreement")
    if asm.contract_changes and not asm.errors and crit is None:
        names = ", ".join(f"{i} -> {p}" for i, p, _ in asm.contract_changes)
        rep.error(where, f"binds a provider whose numeric contract is not the required one ({names}): [gate] "
                         "criterion must say what the recipe gate holds it to, \"token-identity\" or "
                         "\"token-agreement\" (SPEC.md section 7)")


def run(cat, origin=None):
    rep = Report()
    check_catalogue(cat, rep, origin)
    if not isinstance(cat.toolchains, dict) or not cat.toolchains:
        rep.error("toolchains.toml", "missing or empty")
    for name, tc in cat.toolchains.items():
        if not isinstance(tc, dict):
            rep.error("toolchains.toml", f"[{name}] must be a table")
        elif tc.get("kind") == "triton":
            for key in ("version", "python", "target"):
                if not isinstance(tc.get(key), str):
                    rep.error("toolchains.toml", f"[{name}] (triton) needs {key}")
        elif tc.get("kind") == "zig":
            if not isinstance(tc.get("root"), str):
                rep.error("toolchains.toml", f"[{name}] (zig) needs root")
        elif tc.get("kind") not in ROCM_KINDS + (None,):
            rep.error("toolchains.toml", f"[{name}] kind '{tc.get('kind')}' is not one of {', '.join(ROCM_KINDS + NOT_ROCM_KINDS)}")
        elif not isinstance(tc.get("rocm_root"), str):
            rep.error("toolchains.toml", f"[{name}] needs rocm_root")
    for r in cat.recipes:
        asm_origins = None
        if origin is not None:
            asm_origins = {cat.by_id[i].dir_origin for i in r.bindings.values() if isinstance(i, str) and i in cat.by_id}
            asm_origins |= {cat.by_id[s["main"]].dir_origin for s in r.binaries.values()
                            if isinstance(s, dict) and s.get("main") in cat.by_id}
        if origin is None or origin in asm_origins:
            check_recipe(cat, r, rep)
    return rep
