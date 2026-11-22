"""The catalogue on disk: modules, origins, toolchains and recipes, loaded without judging them (check.py judges)."""
import fnmatch
import os
import re
import tomllib

KINDS = ("format", "kernel", "executor", "runtime", "serve", "tool")
ORIGINS = ("strixite", "strixfold", "strixfold_pr6", "gufo", "hipfire", "crucible", "tensorfold", "sglang", "halogen", "ds4", "radiance", "flerp")
LANGS = ("hip-c++", "c++", "c", "zig", "hsaco", "triton-aot", "rust", "python", "cuda")
# `verified` is the first step of a prebuilt kit (hashes checked on the build host); it stands where `builds` does.
STATUSES = ("harvested", "builds", "verified", "parity", "benched")
PREBUILT_LANGS = ("hsaco", "triton-aot")
USES = ("hip-runtime",)
# <family>.<op>/<weight-format-or-dash>/<dtype-or-dash>[.<tag>]; the dtype may be in-out (bf16-f32).
INTERFACE = re.compile(r"^[a-z0-9][a-z0-9_-]*(\.[a-z0-9][a-z0-9_-]*)+/([a-z0-9][a-z0-9._+-]*|-)/"
                       r"([a-z0-9][a-z0-9_+-]*|-)(\.[a-z0-9][a-z0-9_+-]*)?$")
GATE_COMPILE = ("standalone", "header-only", "prebuilt", "n/a")     # or via:<module id>
RENAME_SUFFIX = "_flerp_origin"       # what an adapter's overridden symbols are renamed to in the origin module


def split_tag(iface):
    """('linear.dense/bf16/bf16-f32', 'row-invariant') for 'linear.dense/bf16/bf16-f32.row-invariant'; the tag
    is None when the name has none. The tag sits after the first dot of the last field."""
    head, _, last = iface.rpartition("/")
    dtype, dot, tag = last.partition(".")
    return (f"{head}/{dtype}", tag) if dot else (iface, None)


def overlap_class(iface):
    """The interface name without its tag: providers of one class do the same job on the same operands."""
    return split_tag(iface)[0]
COMPILED = {"hip-c++": (".hip", ".cpp", ".cc"), "c++": (".cpp", ".cc"), "c": (".c",)}


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_toml(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


class Module:
    def __init__(self, path, data):
        self.path = path                      # .../catalog/<origin>/<kind>/<name>/module.toml
        self.dir = os.path.dirname(path)
        self.data = data
        parts = self.dir.split(os.sep)
        self.dir_origin, self.dir_kind, self.dir_name = parts[-3], parts[-2], parts[-1]
        self.id = data.get("id") if isinstance(data.get("id"), str) else f"?{self.dir_kind}.{self.dir_origin}.{self.dir_name}"

    def _table(self, name):
        t = self.data.get(name)
        return t if isinstance(t, dict) else {}

    def _strs(self, table, key):
        v = self._table(table).get(key, [])
        return [x for x in v if isinstance(x, str)] if isinstance(v, list) else []

    @property
    def kind(self):
        return self.data.get("kind")

    @property
    def summary(self):
        return self.data.get("summary", "")

    @property
    def origin(self):
        return self._table("provenance").get("origin")

    @property
    def paths(self):
        return self._strs("provenance", "paths")

    @property
    def spdx(self):
        v = self._table("licence").get("spdx")
        return v if isinstance(v, str) else ""

    @property
    def build(self):
        return self._table("build")

    @property
    def lang(self):
        return self.build.get("lang")

    @property
    def sources(self):
        return self._strs("build", "sources")

    @property
    def includes(self):
        return self._strs("build", "includes")

    @property
    def include_dirs(self):
        return [os.path.normpath(os.path.join(self.dir, d)) for d in self.includes]

    @property
    def provides(self):
        v = self.data.get("provides", [])
        return [p for p in v if isinstance(p, dict)] if isinstance(v, list) else []

    @property
    def requires(self):
        v = self.data.get("requires", [])
        return [r for r in v if isinstance(r, dict)] if isinstance(v, list) else []

    @property
    def interfaces(self):
        return [p["interface"] for p in self.provides if isinstance(p.get("interface"), str)]

    @property
    def gate(self):
        return self._table("gate")

    @property
    def adapter(self):
        return self._table("adapter")

    @property
    def overrides(self):
        """(module id, [symbols]) an adapter takes over, or (None, [])."""
        a = self.adapter
        syms = a.get("symbols", [])
        return (a.get("for") if isinstance(a.get("for"), str) else None,
                [x for x in syms if isinstance(x, str)] if isinstance(syms, list) else [])

    @property
    def code_objects(self):
        """[{module, unit, symbol}]: device code objects of other modules this adapter loads at run time."""
        v = self.adapter.get("code_object", [])
        return [c for c in v if isinstance(c, dict)] if isinstance(v, list) else []

    def units(self):
        """Compile units of a module whose origin builds from concatenated text ([build] units = "units.json"):
        name -> {parts, flags}. A module with plain sources has one unit per source, named by its path."""
        rel = self.build.get("units")
        if isinstance(rel, str):
            import json
            with open(os.path.join(self.dir, rel)) as f:
                return {u["name"]: u for u in json.load(f).get("units", [])}
        return {s: {"name": s, "source": s, "flags": self.build.get("source_options", {}).get(s, [])}
                for s in self.sources}

    def expand(self, entry, candidates):
        """The files a [provenance] paths entry names among `candidates` (origin-relative paths): the file itself,
        every file under a directory, or the matches of a glob pattern."""
        if entry in candidates:
            return [entry]
        under = [c for c in candidates if c.startswith(entry.rstrip("/") + "/")]
        if under:
            return under
        return [c for c in candidates if fnmatch.fnmatchcase(c, entry)] if any(ch in entry for ch in "*?[") else []

    def files(self):
        """Origin-relative paths of the module's files: [provenance] paths expanded against src/."""
        have, out = self.src_files(), []
        for entry in self.paths:
            out += [f for f in self.expand(entry, have) if f not in out]
        return out

    def src_files(self):
        """Every file under src/, relative to src/."""
        src = os.path.join(self.dir, "src")
        out = []
        for d, _, names in os.walk(src):
            out += [os.path.relpath(os.path.join(d, n), src) for n in names]
        return sorted(out)


class Recipe:
    def __init__(self, path, data):
        self.path = path
        self.data = data
        self.file_name = os.path.basename(path)[:-len(".toml")]
        self.name = data.get("name") if isinstance(data.get("name"), str) else self.file_name

    def _table(self, name):
        t = self.data.get(name)
        return t if isinstance(t, dict) else {}

    @property
    def bindings(self):
        return self._table("modules")

    @property
    def include(self):
        v = self._table("include").get("modules", [])
        return v if isinstance(v, list) else []

    @property
    def binaries(self):
        return self._table("binaries")

    @property
    def toolchain(self):
        return self.data.get("toolchain")

    @property
    def gate(self):
        return self._table("gate")


class Catalog:
    def __init__(self, root=None):
        self.root = os.path.abspath(root or repo_root())
        self.modules = []
        self.load_errors = []                # (path, message) for files that are not valid TOML
        cat = os.path.join(self.root, "catalog")
        for origin in sorted(os.listdir(cat)) if os.path.isdir(cat) else []:
            for kind in sorted(self._dirs(os.path.join(cat, origin))):
                for name in sorted(self._dirs(os.path.join(cat, origin, kind))):
                    path = os.path.join(cat, origin, kind, name, "module.toml")
                    if not os.path.isfile(path):
                        self.load_errors.append((os.path.dirname(path), "directory has no module.toml"))
                        continue
                    data = self._read(path)
                    if data is not None:
                        self.modules.append(Module(path, data))
        self.by_id = {}
        for m in self.modules:
            self.by_id.setdefault(m.id, m)
        self.providers = {}
        for m in self.modules:
            for i in m.interfaces:
                self.providers.setdefault(i, []).append(m)
        self.recipes = []
        rdir = os.path.join(self.root, "recipes")
        for f in sorted(os.listdir(rdir)) if os.path.isdir(rdir) else []:
            if f.endswith(".toml"):
                data = self._read(os.path.join(rdir, f))
                if data is not None:
                    self.recipes.append(Recipe(os.path.join(rdir, f), data))
        tpath = os.path.join(self.root, "toolchains.toml")
        self.toolchains = (self._read(tpath) or {}) if os.path.isfile(tpath) else {}

    @staticmethod
    def _dirs(path):
        return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))] if os.path.isdir(path) else []

    def _read(self, path):
        try:
            return read_toml(path)
        except (tomllib.TOMLDecodeError, OSError, UnicodeDecodeError) as e:
            self.load_errors.append((path, f"cannot be read as TOML: {e}"))
            return None

    def origins(self):
        return sorted(self._dirs(os.path.join(self.root, "catalog")))

    def origin_file(self, origin):
        return os.path.join(self.root, "catalog", origin, "ORIGIN.toml")

    def origin_data(self, origin):
        path = self.origin_file(origin)
        return (self._read(path) or {}) if os.path.isfile(path) else None

    def compile_recipe(self, module):
        """How a module's device sources are compiled when its origin does not build like the recipe does:
        None for a module built by the assembly's own rules (strixite's, which the generated CMake file carries),
        else {driver, base flags, flags, includes, product, toolchain root}. product is "host-object" (an object
        with the device code embedded, linked statically) or "code-object" (a device-only object or offload bundle
        that the host loads at run time). See SPEC.md section 5, "Origin compile recipes"."""
        if module.lang != "hip-c++":
            return None
        oc = (self.origin_data(module.dir_origin) or {}).get("compile")
        if not isinstance(oc, dict):
            return None
        tc = self.toolchains.get(module.build.get("toolchain"), {})
        root = tc.get("rocm_root", "")
        flags = module.build.get("options") or oc.get("flags", [])
        rocm_flags = oc.get("rocm_flags", [oc["rocm_flag"]] if "rocm_flag" in oc else [])
        device_only = any(f in ("--genco", "--offload-device-only", "--cuda-device-only") for f in flags)
        return {"driver": oc.get("driver", "").replace("<rocm>", root),
                "base": [oc.get("arch_flag", "")] + [f.replace("<rocm>", root) for f in rocm_flags],
                "flags": list(flags), "product": "code-object" if device_only else "host-object",
                "system_includes": [os.path.join(root, i) for i in oc.get("system_includes", [])],
                "source_prefix": oc.get("source_prefix", []), "toolchain": module.build.get("toolchain")}

    def recipe(self, name):
        for r in self.recipes:
            if name in (r.name, r.file_name):
                return r
        return None

    def rel(self, path):
        return os.path.relpath(path, self.root)

    def consumers(self):
        """interface -> modules that require it."""
        out = {}
        for m in self.modules:
            for r in m.requires:
                if isinstance(r.get("interface"), str):
                    out.setdefault(r["interface"], []).append(m)
        return out


# --- licence expressions -------------------------------------------------------------------------------------
def spdx_ids(expr):
    """The licence ids of an SPDX expression (operators and parentheses dropped; 'X WITH exception' keeps X)."""
    ids, skip = [], False
    for tok in expr.replace("(", " ").replace(")", " ").split():
        if skip:
            skip = False
        elif tok == "WITH":
            skip = True
        elif tok not in ("AND", "OR"):
            ids.append(tok)
    return ids


def _satisfied(expr, ok):
    """Evaluate an SPDX expression where each id is replaced by ok(id); AND binds tighter than OR."""
    toks = expr.replace("(", " ( ").replace(")", " ) ").split()
    pos = 0

    def atom():
        nonlocal pos
        if pos >= len(toks):
            return False
        t = toks[pos]
        pos += 1
        if t == "(":
            v = any_of()
            pos += 1                           # the closing parenthesis
            return v
        v = ok(t)
        if pos < len(toks) and toks[pos] == "WITH":
            pos += 2
        return v

    def all_of():
        nonlocal pos
        v = atom()
        while pos < len(toks) and toks[pos] == "AND":
            pos += 1
            v = atom() and v
        return v

    def any_of():
        nonlocal pos
        v = all_of()
        while pos < len(toks) and toks[pos] == "OR":
            pos += 1
            v = all_of() or v
        return v

    return any_of()


def licence_matches(spdx, expr):
    """expr: comma-separated glob patterns over SPDX ids; a leading '!' denies. With only denials, the rest is
    allowed. 'A AND B' needs both allowed, 'A OR B' either. Example: 'AGPL-3.0-only,MIT,BSD-*' or
    '!LicenseRef-*'."""
    pats = [p.strip() for p in expr.split(",") if p.strip()]
    deny = [p[1:] for p in pats if p.startswith("!")]
    allow = [p for p in pats if not p.startswith("!")]

    def ok(i):
        if any(fnmatch.fnmatchcase(i, p) for p in deny):
            return False
        return not allow or any(fnmatch.fnmatchcase(i, p) for p in allow)

    return bool(spdx) and _satisfied(spdx, ok)
