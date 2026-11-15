"""Unit tests for flerplib: python3 -m unittest (from the repository root). They build small catalogues in a
temporary directory and also check the real catalogue's strixite recipe resolves."""
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from flerplib import assemble, check, cli, includes, origin, tomlw
from flerplib.catalog import INTERFACE, Catalog, licence_matches, repo_root, spdx_ids
from flerplib.resolve import Assembly, closure

TOOLCHAINS = '["tc"]\nrocm_root = "/opt/rocm"\n'


def manifest(origin_name, kind, name, paths, sources=(), provides=(), requires=(), spdx="MIT", lang="c++", **build):
    b = dict(lang=lang, sources=["src/" + s for s in sources], includes=["src"], toolchain="tc")
    if lang == "hip-c++":
        b["arch"] = "gfx1151"
    b.update(build)
    return tomlw.dump([
        ("", dict(id=f"{kind}.{origin_name}.{name}", kind=kind, summary="a test module")),
        ("provenance", dict(origin=origin_name, repo="https://example.invalid/r", commit="abc1234", paths=list(paths),
                            authors=["someone"], modified=False)),
        ("licence", dict(spdx=spdx)),
        ("build", b),
        ("[provides]", [dict(interface=i, api="f", numeric="n/a", regime="any") for i in provides]),
        ("[requires]", [({"module": r[1:]} if r.startswith("@") else {"interface": r}) for r in requires]),
        ("gate", dict(parity="none", bench="none", status="harvested")),
    ])


class Tree:
    """A throwaway repository root with a catalogue."""

    def __init__(self):
        self.root = tempfile.mkdtemp(prefix="flerp-test-")
        self.write("toolchains.toml", TOOLCHAINS)

    def write(self, rel, text):
        path = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
        return path

    def module(self, origin_name, kind, name, files, **kw):
        base = f"catalog/{origin_name}/{kind}/{name}"
        for rel, text in files.items():
            self.write(f"{base}/src/{rel}", text)
        self.write(f"{base}/module.toml", manifest(origin_name, kind, name, kw.pop("paths", list(files)), **kw))

    def origin(self, name, extra=""):
        self.write(f"catalog/{name}/ORIGIN.toml", f'origin = "{name}"\nrepo = "https://example.invalid/r"\n'
                   f'commit = "abc1234"\n[licence]\nspdx = "MIT"\n[toolchain]\nname = "tc"\n{extra}')

    def cat(self):
        return Catalog(self.root)

    def close(self):
        shutil.rmtree(self.root)


def small_tree():
    """kernel a (header + source) <- kernel b (requires a by interface) <- tool t (requires b by module)."""
    t = Tree()
    t.origin("flerp")
    t.module("flerp", "kernel", "a", {"k/a.hpp": "int a();\n", "k/a.cpp": '#include "k/a.hpp"\nint a() { return 1; }\n'},
             sources=["k/a.cpp"], provides=["linear.decode/q4/bf16"])
    t.module("flerp", "kernel", "b", {"k/b.hpp": '#include "k/a.hpp"\nint b();\n',
                                      "k/b.cpp": '#include "k/b.hpp"\nint b() { return a(); }\n'},
             sources=["k/b.cpp"], provides=["norm.rms/-/bf16"], requires=["linear.decode/q4/bf16"])
    t.module("flerp", "tool", "t", {"t/main.cpp": '#include "k/b.hpp"\nint main() { return b(); }\n'},
             sources=["t/main.cpp"], provides=["tool.t/-/-"], requires=["@kernel.flerp.b"])
    t.write("recipes/r.toml", 'name = "r"\nmodel = "m"\ntoolchain = "tc"\n'
            'reference = { engine = "x", build = "/nowhere" }\n'
            '[modules]\n"linear.decode/q4/bf16" = "kernel.flerp.a"\n'
            '[binaries]\nprog = { main = "tool.flerp.t", entry = "t/main.cpp" }\n')
    return t


def slurp(path):
    with open(path) as f:
        return f.read()


def spit(path, text, mode="w"):
    with open(path, mode) as f:
        f.write(text)


def edit(path, old, new):
    with open(path) as f:
        text = f.read()
    assert old in text, old
    with open(path, "w") as f:
        f.write(text.replace(old, new, 1))


class Base(unittest.TestCase):
    def setUp(self):
        self.t = small_tree()
        self.addCleanup(self.t.close)

    def errors(self, origin_name=None):
        return check.run(self.t.cat(), origin_name).errors

    def assertError(self, fragment, origin_name=None):
        errs = self.errors(origin_name)
        self.assertTrue(any(fragment in e for e in errs), f"no error containing {fragment!r} in {errs}")


class CheckTests(Base):
    def test_clean_tree_has_no_errors(self):
        self.assertEqual(self.errors(), [])

    def test_id_must_match_directory(self):
        p = os.path.join(self.t.root, "catalog/flerp/kernel/a/module.toml")
        edit(p, '"kernel.flerp.a"', '"kernel.flerp.other"')
        self.assertError("its directory says 'kernel.flerp.a'")

    def test_listed_path_must_exist(self):
        self.t.module("flerp", "kernel", "c", {"k/c.hpp": "\n"}, paths=["k/c.hpp", "k/gone.hpp"],
                      provides=["x.y/-/-"])
        self.assertError("path 'k/gone.hpp' names nothing under src/")

    def test_unlisted_file_is_an_error(self):
        self.t.write("catalog/flerp/kernel/a/src/k/stowaway.hpp", "\n")
        self.assertError("src/k/stowaway.hpp is not listed")

    def test_directory_and_pattern_paths_cover_files(self):
        self.t.module("flerp", "runtime", "d", {"d/one.hpp": "\n", "d/two.hpp": "\n", "e/x.zig": "\n"},
                      paths=["d", "e/*.zig"], provides=["x.d/-/-"])
        self.assertEqual(self.errors(), [])
        self.assertEqual(sorted(self.t.cat().by_id["runtime.flerp.d"].files()), ["d/one.hpp", "d/two.hpp", "e/x.zig"])

    def test_design_record_beside_manifest(self):
        p = os.path.join(self.t.root, "catalog/flerp/kernel/a/module.toml")
        spit(p, 'design = "DESIGN.md"\n' + slurp(p))
        self.assertError("design 'DESIGN.md' does not exist")
        self.t.write("catalog/flerp/kernel/a/DESIGN.md", "# a\n")
        self.assertEqual(self.errors(), [])
        edit(p, 'design = "DESIGN.md"', 'design = "src/k/a.hpp"')
        self.assertError("must not be under src/")

    def test_requires_must_resolve(self):
        self.t.module("flerp", "kernel", "c", {"k/c.hpp": "\n"}, provides=["x.y/-/-"],
                      requires=["no.such/-/-", "@kernel.flerp.nobody"])
        self.assertError("interface 'no.such/-/-' has no provider")
        self.assertError("module 'kernel.flerp.nobody' is not in the catalogue")

    def test_spdx_must_be_present(self):
        self.t.module("flerp", "kernel", "c", {"k/c.hpp": "\n"}, provides=["x.y/-/-"], spdx="")
        self.assertError("spdx")

    def test_interface_name_shape(self):
        self.t.module("flerp", "kernel", "c", {"k/c.hpp": "\n"}, provides=["notaninterface"])
        self.assertError("is not <family>.<op>/")
        self.assertTrue(INTERFACE.match("linear.decode/strixw-q4/bf16"))
        self.assertTrue(INTERFACE.match("logits.topk/-/-"))
        self.assertFalse(INTERFACE.match("linear/strixw-q4/bf16"))
        self.assertFalse(INTERFACE.match("linear.decode/strixw-q4"))

    def test_include_from_unrequired_module_is_an_error(self):
        self.t.module("flerp", "kernel", "c", {"k/c.cpp": '#include "k/a.hpp"\n'}, sources=["k/c.cpp"],
                      provides=["x.y/-/-"])
        self.assertError("includes k/a.hpp from kernel.flerp.a, which is not in [[requires]]")

    def test_source_must_be_a_listed_file_of_the_language(self):
        self.t.module("flerp", "kernel", "c", {"k/c.hpp": "\n"}, sources=["k/c.hpp"], provides=["x.y/-/-"])
        self.assertError("is not a c++ source")

    def test_status_past_harvested_needs_evidence(self):
        p = os.path.join(self.t.root, "catalog/flerp/kernel/a/module.toml")
        edit(p, '"harvested"', '"builds"')
        self.assertError("needs evidence")

    def test_file_in_two_modules(self):
        self.t.module("flerp", "kernel", "c", {"k/a.hpp": "int a();\n"}, provides=["x.y/-/-"])
        self.assertError("origin file k/a.hpp is also in")

    def test_missing_origin_file(self):
        os.remove(os.path.join(self.t.root, "catalog/flerp/ORIGIN.toml"))
        self.assertError("every origin needs ORIGIN.toml")

    def test_recipe_unfilled_interface(self):
        p = os.path.join(self.t.root, "recipes/r.toml")
        edit(p, '"linear.decode/q4/bf16" = "kernel.flerp.a"', '"norm.rms/-/bf16" = "kernel.flerp.b"')
        self.assertError("interface linear.decode/q4/bf16 is unfilled")

    def test_recipe_binding_must_provide(self):
        p = os.path.join(self.t.root, "recipes/r.toml")
        edit(p, "kernel.flerp.a", "kernel.flerp.b")
        self.assertError("does not provide it")

    def test_origin_filter_ignores_other_origins(self):
        self.t.origin("gufo")
        self.t.module("gufo", "kernel", "bad", {"k/z.hpp": "\n"}, provides=["notaninterface"])
        self.assertTrue(self.errors())
        self.assertEqual(self.errors("flerp"), [])


class ResolveTests(Base):
    def test_closure_orders_dependencies_first(self):
        cat = self.t.cat()
        errs = []
        ids = closure(cat, ["tool.flerp.t"], cat.recipe("r").bindings, errs)
        self.assertEqual(errs, [])
        self.assertEqual(ids, ["kernel.flerp.a", "kernel.flerp.b", "tool.flerp.t"])

    def test_cycle_is_reported(self):
        self.t.module("flerp", "kernel", "a", {"k/a.hpp": "int a();\n"}, provides=["linear.decode/q4/bf16"],
                      requires=["@kernel.flerp.b"])
        asm = Assembly(self.t.cat(), self.t.cat().recipe("r"))
        # reported, not refused: the closure is the same set and objects link in any order
        self.assertTrue(any("dependency cycle" in w for w in asm.warnings))
        self.assertFalse(any("dependency cycle" in e for e in asm.errors))

    def test_two_providers_in_one_assembly(self):
        self.t.module("flerp", "kernel", "a2", {"k/a2.hpp": "\n"}, provides=["linear.decode/q4/bf16"])
        self.t.module("flerp", "tool", "t", {"t/main.cpp": "int main() { return 0; }\n"}, sources=["t/main.cpp"],
                      provides=["tool.t/-/-"], requires=["@kernel.flerp.b", "@kernel.flerp.a2"])
        asm = Assembly(self.t.cat(), self.t.cat().recipe("r"))
        self.assertTrue(any("provided twice" in e for e in asm.errors))

    def test_without_a_recipe_one_provider_resolves_and_two_do_not(self):
        cat = self.t.cat()
        errs = []
        closure(cat, ["kernel.flerp.b"], None, errs)
        self.assertEqual(errs, [])
        self.t.module("flerp", "kernel", "a2", {"k/a2.hpp": "\n"}, provides=["linear.decode/q4/bf16"])
        closure(self.t.cat(), ["kernel.flerp.b"], None, errs)
        self.assertTrue(any("a recipe must choose" in e for e in errs))


class AssembleTests(Base):
    def generate(self, **kw):
        cat = self.t.cat()
        out = os.path.join(self.t.root, "build", "r")
        asm, errs = assemble.generate(cat, cat.recipe("r"), out, **kw)
        return out, errs

    def test_generates_only_catalogue_paths(self):
        out, errs = self.generate()
        self.assertEqual(errs, [])
        text = slurp(os.path.join(out, "CMakeLists.txt"))
        self.assertIn("add_library(m_kernel_flerp_a OBJECT", text)
        self.assertIn("${FLERP_ROOT}/catalog/flerp/kernel/a/src/k/a.cpp", text)
        self.assertIn("add_executable(prog", text)
        self.assertIn("$<TARGET_OBJECTS:m_tool_flerp_t>", text)
        self.assertNotIn(self.t.root, text)               # relocatable: paths hang off FLERP_ROOT
        man = json.loads(slurp(os.path.join(out, "assembly.json")))
        self.assertEqual(man["binaries"]["prog"]["modules"], ["kernel.flerp.a", "kernel.flerp.b", "tool.flerp.t"])

    def test_include_dirs_are_the_requirement_closure(self):
        self.t.module("flerp", "kernel", "lonely", {"k/l.hpp": "\n", "k/l.cpp": "\n"}, sources=["k/l.cpp"],
                      provides=["x.l/-/-"])
        p = os.path.join(self.t.root, "recipes/r.toml")
        spit(p, '[include]\nmodules = ["kernel.flerp.lonely"]\n', "a")
        out, errs = self.generate()
        self.assertEqual(errs, [])
        text = slurp(os.path.join(out, "CMakeLists.txt"))
        block = text[text.index("add_library(m_kernel_flerp_lonely"):text.index("# --- binaries")]
        self.assertNotIn("kernel/a/src", block)
        self.assertIn("flerp_include_only", text)

    def test_hip_sources_get_device_options_and_runtime(self):
        self.t.module("flerp", "kernel", "a", {"k/a.hpp": "int a();\n", "k/a.hip": "int a() { return 1; }\n"},
                      sources=["k/a.hip"], provides=["linear.decode/q4/bf16"], lang="hip-c++", uses=["hip-runtime"])
        out, errs = self.generate()
        self.assertEqual(errs, [])
        text = slurp(os.path.join(out, "CMakeLists.txt"))
        self.assertIn('PROPERTIES LANGUAGE CXX COMPILE_OPTIONS "${FLERP_HIP_OPTIONS}"', text)
        self.assertIn("target_compile_definitions(m_kernel_flerp_b PRIVATE __HIP_PLATFORM_AMD__)", text)
        self.assertIn("${FLERP_HIP_LIB}", text[text.index("add_executable(prog"):])

    def test_licence_filter_refuses(self):
        _, errs = self.generate(licence="Apache-2.0")
        self.assertTrue(any("outside --licence" in e for e in errs))
        _, errs = self.generate(licence="MIT")
        self.assertEqual(errs, [])

    def test_module_gate_tree_for_header_only_module(self):
        self.t.module("flerp", "kernel", "h", {"k/h.hpp": "inline int h() { return 2; }\n"}, provides=["x.h/-/-"])
        cat = self.t.cat()
        out = os.path.join(self.t.root, "build", "gate-module", "h")
        self.assertEqual(assemble.generate_module(cat, cat.by_id["kernel.flerp.h"], out), [])
        self.assertIn("#include <k/h.hpp>", slurp(os.path.join(out, "headers_probe.cpp")))
        self.assertIn("headers_probe.cpp", slurp(os.path.join(out, "CMakeLists.txt")))


    def test_overlay_swaps_sources_and_puts_its_root_first(self):
        # kernel a overlaid by a derived module whose k/a.cpp and k/a.hpp sit under src/flerp/a2
        self.t.module("flerp", "kernel", "a2", {"flerp/a2/k/a.hpp": "int a(); // v2\n",
                                                "flerp/a2/k/a.cpp": '#include "k/a.hpp"\nint a() { return 2; }\n'},
                      sources=["flerp/a2/k/a.cpp"], provides=["x.a2/-/-"], requires=["@kernel.flerp.a", "@kernel.flerp.z"],
                      includes=["src/flerp/a2"])
        self.t.module("flerp", "kernel", "z", {"k/z.hpp": "\n"}, provides=["x.z/-/-"])
        p = os.path.join(self.t.root, "recipes/r.toml")
        spit(p, '[overlay]\n"kernel.flerp.a2" = "kernel.flerp.a"\n', "a")
        out, errs = self.generate()
        self.assertEqual(errs, [])
        text = slurp(os.path.join(out, "CMakeLists.txt"))
        a = text[text.index("add_library(m_kernel_flerp_a OBJECT"):text.index("add_library(m_kernel_flerp_b")]
        self.assertIn("kernel/a2/src/flerp/a2/k/a.cpp", a)
        self.assertNotIn("kernel/a/src/k/a.cpp", a)
        self.assertNotIn("add_library(m_kernel_flerp_a2 ", text)          # compiled in a's target only
        b = text[text.index("add_library(m_kernel_flerp_b OBJECT"):]
        self.assertLess(b.index("kernel/a2/src/flerp/a2"), b.index("kernel/a/src\""))   # derived root first
        self.assertIn("kernel/z/src", a)                                   # a's target gets a2's requirements
        man = json.loads(slurp(os.path.join(out, "assembly.json")))
        self.assertEqual(man["overlays"]["kernel.flerp.a2"]["origin"], "kernel.flerp.a")
        self.assertEqual(self.errors(), [])

    def test_overlay_source_without_counterpart_is_refused(self):
        self.t.module("flerp", "kernel", "a2", {"flerp/a2/k/other.cpp": "int o() { return 0; }\n"},
                      sources=["flerp/a2/k/other.cpp"], provides=["x.a2/-/-"], requires=["@kernel.flerp.a"],
                      includes=["src/flerp/a2"])
        spit(os.path.join(self.t.root, "recipes/r.toml"), '[overlay]\n"kernel.flerp.a2" = "kernel.flerp.a"\n', "a")
        _, errs = self.generate()
        self.assertTrue(any("replaces no source" in e for e in errs), errs)
        self.assertError("replaces no source")


class OriginTests(Base):
    def checkout(self):
        co = os.path.join(self.t.root, "upstream")
        for m in self.t.cat().modules:
            for f in m.files():
                dst = os.path.join(co, f)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(os.path.join(m.dir, "src", f), dst)
        return co

    def test_verbatim_tree_is_clean(self):
        co = self.checkout()
        spit(os.path.join(co, ".git"), "gitdir: elsewhere\n")     # a worktree's .git file is not an origin file
        res = origin.diff_origin(self.t.cat(), "flerp", co)
        self.assertEqual(res["problems"], [])
        self.assertEqual(res["identical"], res["files"])

    def test_changed_byte_is_found(self):
        co = self.checkout()
        with open(os.path.join(co, "k/a.cpp"), "a") as f:
            f.write(" ")
        res = origin.diff_origin(self.t.cat(), "flerp", co)
        self.assertEqual([(m, p) for m, p, _ in res["problems"]], [("kernel.flerp.a", "k/a.cpp")])

    def test_unharvested_origin_file_must_be_listed(self):
        co = self.checkout()
        os.makedirs(os.path.join(co, "docs"))
        spit(os.path.join(co, "docs/x.md"), "x\n")
        spit(os.path.join(co, "README.md"), "x\n")
        res = origin.diff_origin(self.t.cat(), "flerp", co)
        self.assertEqual(sorted(p for _, p, _ in res["problems"]), ["README.md", "docs/x.md"])
        self.t.origin("flerp", '[harvest]\nunharvested = ["README.md", "docs/**"]\n')
        res = origin.diff_origin(self.t.cat(), "flerp", co)
        self.assertEqual(res["problems"], [])
        self.assertEqual(res["unharvested"], 2)


class SmallPieces(unittest.TestCase):
    def test_licence_expressions(self):
        self.assertEqual(spdx_ids("AGPL-3.0-only AND Unicode-3.0"), ["AGPL-3.0-only", "Unicode-3.0"])
        self.assertEqual(spdx_ids("GPL-2.0-only WITH Classpath-exception-2.0"), ["GPL-2.0-only"])
        self.assertTrue(licence_matches("MIT", "MIT,BSD-*"))
        self.assertTrue(licence_matches("BSD-3-Clause", "MIT,BSD-*"))
        self.assertFalse(licence_matches("AGPL-3.0-only AND Unicode-3.0", "AGPL-*"))
        self.assertTrue(licence_matches("AGPL-3.0-only AND Unicode-3.0", "AGPL-*,Unicode-3.0"))
        self.assertTrue(licence_matches("MIT OR Unlicense", "MIT"))
        self.assertTrue(licence_matches("MIT", "!LicenseRef-*"))
        self.assertFalse(licence_matches("LicenseRef-proprietary", "!LicenseRef-*"))
        self.assertFalse(licence_matches("", "MIT"))

    def test_include_scan(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        p = os.path.join(d, "x.cpp")
        spit(p, '#include "a/b.hpp"\n  #  include <c.h>\n// #include "no.h"\nint x;\n')
        self.assertEqual(includes.scan(p), [("a/b.hpp", True), ("c.h", False)])

    def test_toml_writer_round_trips(self):
        import tomllib
        doc = [("", {"id": 'a "quoted" id', "n": 3, "flag": True}), ("t", {"list": ["x", "y"]}),
               ("[p]", [{"k": "v"}, {"k": "w"}])]
        back = tomllib.loads(tomlw.dump(doc))
        self.assertEqual(back, {"id": 'a "quoted" id', "n": 3, "flag": True, "t": {"list": ["x", "y"]},
                                "p": [{"k": "v"}, {"k": "w"}]})


class CliTests(Base):
    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = cli.main(["--root", self.t.root] + list(argv))
        return rc, out.getvalue(), err.getvalue()

    def test_commands(self):
        rc, out, _ = self.run_cli("check")
        self.assertEqual(rc, 0, out)
        rc, out, _ = self.run_cli("list", "--kind", "kernel", "--provides", "linear.*")
        self.assertEqual(out.split()[0], "kernel.flerp.a")
        rc, out, _ = self.run_cli("show", "kernel.flerp.b")
        self.assertIn("linear.decode/q4/bf16  <- kernel.flerp.a", out)
        rc, out, _ = self.run_cli("interfaces")
        self.assertIn("tool.t/-/-  [single-provider, unused]", out)
        rc, out, _ = self.run_cli("licence-report", "r")
        self.assertIn("MIT  (3 modules)", out)
        rc, out, _ = self.run_cli("assemble", "r", "-o", os.path.join(self.t.root, "build/r"))
        self.assertEqual(rc, 0, out)
        self.assertIn("build: cmake -G Ninja", out)      # the next step of a CMake assembly
        rc, out, _ = self.run_cli("show", "kernel.flerp.nope")
        self.assertEqual(rc, 1)


class ZigRecipeTests(Base):
    """SPEC.md section 7: a recipe built by its origin's build graph ([build] system = "zig")."""

    BUILD = '[build]\nsystem = "zig"\nzig = "z"\nroot = "tool.flerp.t"\nsteps = ["all"]\n'

    def setUp(self):
        super().setUp()
        spit(os.path.join(self.t.root, "toolchains.toml"), '["z"]\nkind = "zig"\nroot = "/opt/zig"\n', "a")
        edit(os.path.join(self.t.root, "recipes/r.toml"), "[modules]", self.BUILD + "[modules]")

    def recipe_errors(self):
        return [e for e in self.errors() if e.startswith("recipes/r.toml") and "[build]" in e]

    def test_zig_toolchain_entry_names_the_binary(self):
        from flerplib import assemble_zig
        cat = self.t.cat()
        self.assertEqual(assemble_zig.zig_binary(cat, cat.recipe("r")), "/opt/zig/zig")
        self.assertEqual(self.recipe_errors(), [])
        edit(os.path.join(self.t.root, "recipes/r.toml"), 'zig = "z"', 'zig = "/abs/zig"')
        self.assertEqual(assemble_zig.zig_binary(self.t.cat(), self.t.cat().recipe("r")), "/abs/zig")

    def test_build_table_is_checked(self):
        r = os.path.join(self.t.root, "recipes/r.toml")
        edit(r, 'zig = "z"', 'zig = "tc"')               # a ROCm install is not a zig
        self.assertError("[build] zig 'tc' is neither")
        edit(r, 'zig = "tc"', 'zig = "z"')
        edit(r, 'root = "tool.flerp.t"', 'root = "tool.flerp.nope"')
        self.assertError("[build] root 'tool.flerp.nope' is not a module")
        edit(r, 'system = "zig"', 'system = "bazel"')
        self.assertError('[build] system must be "zig"')

    def test_a_zig_entry_is_not_a_recipe_or_module_toolchain(self):
        edit(os.path.join(self.t.root, "recipes/r.toml"), 'toolchain = "tc"', 'toolchain = "z"')
        self.assertError("toolchain 'z' is not a ROCm install")
        edit(os.path.join(self.t.root, "toolchains.toml"), 'kind = "zig"', 'kind = "llvm"')
        self.assertError("kind 'llvm' is not one of")

    def test_real_zig_recipe_prints_its_build_script(self):
        out = io.StringIO()
        tmp = tempfile.mkdtemp(prefix="flerp-test-")
        self.addCleanup(shutil.rmtree, tmp)
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            rc = cli.main(["assemble", "dsv41-flash.strixfold-tp2", "-o", tmp])
        self.assertEqual(rc, 0, out.getvalue())
        self.assertIn("/build.sh [-jN]", out.getvalue())
        self.assertNotIn("cmake", out.getvalue())
        self.assertIn("zig=${FLERP_ZIG:-/home/pod/opt/zig-x86_64-linux-0.17.0/zig}", slurp(os.path.join(tmp, "build.sh")))


class CmakeRecipeTests(Base):
    """SPEC.md section 7: a recipe built by its origin's CMake files ([build] system = "cmake")."""

    BUILD = ('[build]\nsystem = "cmake"\nroot = "tool.flerp.t"\nbuild_file = "t/CMakeLists.txt"\n'
             'targets = ["all"]\n')

    def setUp(self):
        super().setUp()
        edit(os.path.join(self.t.root, "recipes/r.toml"), "[modules]", self.BUILD + "[modules]")

    def test_build_table_is_checked(self):
        r = os.path.join(self.t.root, "recipes/r.toml")
        self.assertEqual([e for e in self.errors() if e.startswith("recipes/r.toml") and "[build]" in e], [])
        edit(r, 'targets = ["all"]\n', "")
        self.assertError("[build] targets is missing")
        edit(r, 'root = "tool.flerp.t"', 'root = "tool.flerp.nope"')
        self.assertError("[build] root 'tool.flerp.nope' is not a module")

    def test_real_gufo_recipe_is_a_tree_of_its_closure(self):
        out = io.StringIO()
        tmp = tempfile.mkdtemp(prefix="flerp-test-")
        self.addCleanup(shutil.rmtree, tmp)
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            rc = cli.main(["assemble", "qwen35-decider-2b.gufo", "-o", tmp])
        self.assertEqual(rc, 0, out.getvalue())
        self.assertIn("/build.sh [-jN]", out.getvalue())
        script = slurp(os.path.join(tmp, "build.sh"))
        self.assertIn('-S "$here/tree/flerp/gufo_qwen" -B "$here/out"', script)
        self.assertIn("ninja -C \"$here/out\" \"$jobs\" flerp-gufo-qwen-generate", script)
        with open(os.path.join(tmp, "assembly.json")) as f:
            asm = json.load(f)
        on_disk = sorted(os.path.relpath(os.path.join(d, n), os.path.join(tmp, "tree"))
                         for d, _, names in os.walk(os.path.join(tmp, "tree")) for n in names)
        self.assertEqual(on_disk, sorted(asm["files"]))
        # the origin's own CMake files and the stand-in for its root are in the tree; the origin's root is not
        for rel in ("src/models/qwen/CMakeLists.txt", "src/families/qwen/kernels/CMakeLists.txt",
                    "src/families/qwen/vision/device_input.hpp", "flerp/gufo_qwen/CMakeLists.txt"):
            self.assertIn(rel, asm["files"])
        self.assertNotIn("CMakeLists.txt", asm["files"])
        self.assertEqual(asm["files"]["flerp/gufo_qwen/generate.cpp"]["module"], "tool.flerp.gufo_qwen_generate")


class GufoGateTests(unittest.TestCase):
    def test_agreement_reports_the_first_difference(self):
        import sys
        sys.path.insert(0, os.path.join(repo_root(), "gate"))
        try:
            import gufo_recipe
        finally:
            sys.path.remove(os.path.join(repo_root(), "gate"))
        self.assertIsNone(gufo_recipe.first_difference([1, 2, 3], [1, 2, 3, 4]))
        self.assertEqual(gufo_recipe.first_difference([1, 2, 3], [1, 9, 3]), 1)
        self.assertEqual(gufo_recipe.agreement([1, 2, 3], [1, 2, 3]),
                         {"first_difference": None, "compared": 3, "lengths": [3, 3], "equal": True})
        self.assertFalse(gufo_recipe.agreement([1, 2, 3], [1, 2])["equal"])        # a prefix is not equal
        self.assertEqual(gufo_recipe.agreement([1, 2, 3], [1, 5, 3])["compared"], 1)


class RealCatalogue(unittest.TestCase):
    """The committed strixite harvest and its recipe."""

    def test_strixite_origin_checks_clean(self):
        cat = Catalog(repo_root())
        if "strixite" not in cat.origins():
            self.skipTest("no strixite origin in this tree")
        self.assertEqual(check.run(cat, "strixite").errors, [])

    def test_recipe_host_tests_link_no_device_code(self):
        cat = Catalog(repo_root())
        r = cat.recipe("qwen38-flash-next.strixite")
        if r is None:
            self.skipTest("no strixite recipe in this tree")
        asm = Assembly(cat, r)
        self.assertEqual(asm.errors, [])
        for name in ("check_stream_calls", "check_logprobs", "cache_xfer_check", "sv9_cache_check"):
            ids = asm.binaries[name][2]
            self.assertFalse([i for i in ids if cat.by_id[i].lang == "hip-c++"], name)
        server = asm.binaries["strixflerp-qwen38-flash-next.strixite"][2]
        self.assertIn("kernel.strixite.linear_q4", server)
        self.assertNotIn("executor.strixite.qwen3_5", server)


if __name__ == "__main__":
    unittest.main()
