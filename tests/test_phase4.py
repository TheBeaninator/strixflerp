"""Unit tests for what phase 4 added: interface tags, adapters, origin compile recipes, the manifest keys adopted
from docs/SPEC-PROPOSALS.md, and the pure parts of the interface and recipe gates. python3 -m unittest"""
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout

from flerplib import assemble, check, cli
from flerplib.catalog import INTERFACE, Catalog, overlap_class, repo_root, split_tag
from flerplib.resolve import Assembly, satisfies
from tests.test_flerp import Tree, edit, slurp

ADAPTER = '''
[adapter]
for       = "kernel.strixite.add"
symbols   = ["eng_add"]
'''
CODE_OBJECT = '''
[[adapter.code_object]]
module    = "kernel.hipfire.addco"
unit      = "src/k/add.hip"
symbol    = "flerp_co_dev_add"
'''


def load_gate(name):
    """A gate script as a module (gate/ is not a package)."""
    path = os.path.join(repo_root(), "gate", name + ".py")
    sys.path.insert(0, os.path.dirname(path))
    try:
        spec = importlib.util.spec_from_file_location("gate_" + name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(os.path.dirname(path))


def swap_tree():
    """Three origins cut down to one kernel each: `strixite` (its executor calls eng_add), `gufo` (host objects by
    its own compile recipe) and `hipfire` (device code objects); adapters under `flerp` that put them behind eng_add."""
    t = Tree()
    t.write("toolchains.toml", '["tc"]\nrocm_root = "/opt/rocm"\n["tc2"]\nrocm_root = "/opt/rocm2"\n')
    t.origin("strixite")
    t.origin("gufo", '[compile]\ndriver = "<rocm>/bin/clang++"\narch_flag = "--offload-arch=gfx1151"\n'
                    'rocm_flag = "--rocm-path=<rocm>"\nflags = ["-x", "hip", "-O2", "-c"]\nsystem_includes = ["include"]\n')
    t.origin("hipfire", '[compile]\ndriver = "<rocm>/bin/hipcc"\narch_flag = "--offload-arch=gfx1151"\n'
                    'rocm_flag = "--rocm-path=<rocm>"\nflags = ["--genco", "-O3"]\n')
    t.origin("flerp")
    t.module("strixite", "kernel", "add", {"k/add.hpp": "void eng_add(float *x, const float *y, int n);\n",
                                      "k/add.hip": '#include "k/add.hpp"\nvoid eng_add(float *x, const float *y, int n) {}\n'},
             sources=["k/add.hip"], provides=["residual.add/-/f32"], lang="hip-c++")
    t.module("strixite", "executor", "model", {"m/model.cpp": '#include "k/add.hpp"\nint main() { eng_add(0, 0, 0); }\n'},
             sources=["m/model.cpp"], provides=["executor.forward/m/f32"], requires=["residual.add/-/f32"])
    t.module("gufo", "kernel", "addobj", {"o/add.hpp": "void ObjAdd(const float *a, const float *b, float *out, int n);\n",
                                         "o/add.hip": '#include "o/add.hpp"\nvoid ObjAdd(const float *a, const float *b, float *o, int n) {}\n'},
             sources=["o/add.hip"], provides=["residual.add/-/f32.obj"], lang="hip-c++",
             source_options={"src/o/add.hip": ["-mwavefrontsize64"]})
    t.module("hipfire", "kernel", "addco", {"k/add.hip": "extern \"C\" void dev_add(float *a, const float *b, int n) {}\n"},
             sources=["k/add.hip"], provides=["residual.add/-/f32.dev"], lang="hip-c++", toolchain="tc2")
    t.module("flerp", "kernel", "add_obj", {"a/add_obj.cpp": '#include "k/add.hpp"\n#include "o/add.hpp"\n'
                                            "void eng_add(float *x, const float *y, int n) { ObjAdd(x, y, x, n); }\n"},
             sources=["a/add_obj.cpp"], provides=["residual.add/-/f32"],
             requires=["@kernel.strixite.add", "@kernel.gufo.addobj"])
    t.module("flerp", "kernel", "add_dev", {"a/add_dev.cpp": '#include "k/add.hpp"\nvoid eng_add(float *x, const float *y, int n) {}\n'},
             sources=["a/add_dev.cpp"], provides=["residual.add/-/f32.dev-order"],
             requires=["@kernel.strixite.add", "@kernel.hipfire.addco"])
    for name, extra in (("add_obj", ADAPTER), ("add_dev", ADAPTER + CODE_OBJECT)):
        edit(os.path.join(t.root, f"catalog/flerp/kernel/{name}/module.toml"), "[[provides]]", extra.lstrip("\n") + "\n[[provides]]")
    for name, provider, extra in (("base", "kernel.strixite.add", ""), ("swap-obj", "kernel.flerp.add_obj", ""),
                                  ("swap-dev", "kernel.flerp.add_dev", '[gate]\ncriterion = "token-agreement"\n')):
        t.write(f"recipes/{name}.toml", f'name = "{name}"\nmodel = "m"\ntoolchain = "tc"\n'
                'reference = { engine = "strixite", build = "/nowhere" }\n'
                f'[modules]\n"residual.add/-/f32" = "{provider}"\n'
                '[binaries]\nprog = { main = "executor.strixite.model", entry = "m/model.cpp" }\n' + extra)
    return t


class TagTests(unittest.TestCase):
    def test_split_and_class(self):
        self.assertEqual(split_tag("linear.dense/bf16/bf16-f32.row-invariant"), ("linear.dense/bf16/bf16-f32", "row-invariant"))
        self.assertEqual(split_tag("logits.topk/-/-.gufo-select"), ("logits.topk/-/-", "gufo-select"))
        self.assertEqual(split_tag("executor.model/qwen3.5/gguf"), ("executor.model/qwen3.5/gguf", None))
        self.assertEqual(overlap_class("residual.add/-/f32.hipfire"), "residual.add/-/f32")

    def test_pattern(self):
        for good in ("residual.add/-/f32", "linear.dense/bf16/bf16-f32.row-invariant", "logits.topk/-/-.gufo-select",
                     "attn.softmax-gqa/kv-f16/bf16", "executor.model/qwen3.5/gguf"):
            self.assertTrue(INTERFACE.match(good), good)
        for bad in ("residual/-/f32", "residual.add/-/f32.a.b", "residual.add/-/.tag", "Residual.add/-/f32"):
            self.assertFalse(INTERFACE.match(bad), bad)

    def test_real_catalogue_names_parse(self):
        cat = Catalog(repo_root())
        for name in set(cat.providers) | set(cat.consumers()):
            self.assertTrue(INTERFACE.match(name), name)


class SwapTests(unittest.TestCase):
    def setUp(self):
        self.t = swap_tree()
        self.addCleanup(self.t.close)
        self.cat = self.t.cat()

    def asm(self, name):
        return Assembly(self.cat, self.cat.recipe(name))

    def test_tree_checks_clean(self):
        self.assertEqual(check.run(self.cat).errors, [])

    def test_untagged_requirement_takes_a_tagged_provider(self):
        m = self.cat.by_id["kernel.flerp.add_dev"]
        self.assertEqual(satisfies(m, "residual.add/-/f32"), "residual.add/-/f32.dev-order")
        self.assertIsNone(satisfies(m, "residual.add/-/f32.obj"))      # a tagged requirement wants its full name
        self.assertEqual(self.asm("swap-dev").contract_changes,
                         [("residual.add/-/f32", "residual.add/-/f32.dev-order", "kernel.flerp.add_dev")])
        self.assertEqual(self.asm("swap-obj").contract_changes, [])

    def test_contract_change_needs_a_declared_criterion(self):
        edit(os.path.join(self.t.root, "recipes/swap-dev.toml"), '[gate]\ncriterion = "token-agreement"\n', "")
        errs = check.run(self.t.cat()).errors
        self.assertTrue(any("criterion" in e and "swap-dev" in e for e in errs), errs)

    def test_adapter_takes_over_symbols_without_a_double_provider(self):
        a = self.asm("swap-obj")
        self.assertEqual(a.errors, [])
        self.assertEqual(a.renames, {"kernel.strixite.add": ["eng_add"]})
        self.assertEqual(a.adapters, {"kernel.strixite.add": "kernel.flerp.add_obj"})
        self.assertEqual(self.asm("base").renames, {})

    def test_generated_tree(self):
        out = os.path.join(self.t.root, "build", "swap-obj")
        _, errors = assemble.generate(self.cat, self.cat.recipe("swap-obj"), out)
        self.assertEqual(errors, [])
        cm = slurp(os.path.join(out, "CMakeLists.txt"))
        self.assertIn("eng_add=eng_add_flerp_origin", cm)                       # the rename, on the origin module only
        self.assertEqual(cm.count("eng_add=eng_add_flerp_origin"), 1)
        self.assertIn('"/opt/rocm/bin/clang++" "--offload-arch=gfx1151" "--rocm-path=/opt/rocm" "-x" "hip" "-O2" "-c" '
                      '"-mwavefrontsize64"', cm)                                # the origin's command, source option included
        self.assertIn("add_library(m_kernel_gufo_addobj STATIC", cm)
        self.assertIn("target_link_libraries(prog PRIVATE m_kernel_gufo_addobj", cm)
        self.assertNotIn("$<TARGET_OBJECTS:m_kernel_gufo_addobj>", cm)
        man = json.loads(slurp(os.path.join(out, "assembly.json")))
        self.assertEqual(man["foreign"]["kernel.gufo.addobj"]["product"], "host-object")
        self.assertEqual(man["renames"]["kernel.strixite.add"]["adapter"], "kernel.flerp.add_obj")

    def test_code_object_is_built_by_its_origin_and_embedded(self):
        out = os.path.join(self.t.root, "build", "swap-dev")
        _, errors = assemble.generate(self.cat, self.cat.recipe("swap-dev"), out)
        self.assertEqual(errors, [])
        cm = slurp(os.path.join(out, "CMakeLists.txt"))
        self.assertIn('"/opt/rocm2/bin/hipcc" "--offload-arch=gfx1151" "--rocm-path=/opt/rocm2" "--genco" "-O3"', cm)
        self.assertIn(".incbin", cm)
        self.assertIn("flerp_co_dev_add_end", cm)
        self.assertIn("${CMAKE_BINARY_DIR}/co/flerp_co_dev_add.o", cm.split("add_executable(prog")[1])
        self.assertNotIn("m_kernel_hipfire_addco", cm)                              # nothing of a code-object module is linked
        man = json.loads(slurp(os.path.join(out, "assembly.json")))
        self.assertEqual(man["code_objects"][0]["symbol"], "flerp_co_dev_add")
        self.assertEqual(man["contract_changes"][0]["provided"], "residual.add/-/f32.dev-order")

    def test_the_unmodified_recipe_is_untouched_by_adapters_in_the_catalogue(self):
        out = os.path.join(self.t.root, "build", "base")
        assemble.generate(self.cat, self.cat.recipe("base"), out)
        cm = slurp(os.path.join(out, "CMakeLists.txt"))
        self.assertNotIn("_flerp_origin", cm)
        self.assertNotIn("add_custom_command", cm)

    def test_adapter_manifest_errors(self):
        path = os.path.join(self.t.root, "catalog/flerp/kernel/add_dev/module.toml")
        edit(path, 'unit      = "src/k/add.hip"', 'unit      = "src/k/nope.hip"')
        self.assertTrue(any("no compile unit" in e for e in check.run(self.t.cat()).errors))
        edit(path, 'for       = "kernel.strixite.add"', 'for       = "kernel.strixite.gone"')
        self.assertTrue(any("is not a module of the catalogue" in e for e in check.run(self.t.cat()).errors))

    def test_code_object_must_come_from_a_code_object_origin(self):
        path = os.path.join(self.t.root, "catalog/flerp/kernel/add_dev/module.toml")
        edit(path, 'module    = "kernel.hipfire.addco"\nunit      = "src/k/add.hip"',
             'module    = "kernel.strixite.add"\nunit      = "src/k/add.hip"')
        errs = check.run(self.t.cat()).errors
        self.assertTrue(any("does not produce" in e for e in errs), errs)

    def test_interfaces_groups_by_class_and_names_what_an_adapter_launches(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = cli.main(["--root", self.t.root, "interfaces", "residual.*"])
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn("residual.add/-/f32  [5 providers from 3 origin(s)]", text)
        self.assertIn("kernel.flerp.add_dev  as .dev-order  (adapter for kernel.strixite.add over kernel.hipfire.addco)", text)
        self.assertIn("kernel.gufo.addobj  as .obj", text)


class ManifestKeyTests(unittest.TestCase):
    def setUp(self):
        self.t = swap_tree()
        self.addCleanup(self.t.close)
        self.path = os.path.join(self.t.root, "catalog/hipfire/kernel/addco/module.toml")

    def errors(self):
        return check.run(self.t.cat()).errors

    def test_adopted_keys_do_not_warn(self):
        edit(self.path, 'summary = "a test module"', 'summary = "a test module"\nnotes = "free text"')
        edit(self.path, 'modified = false', 'modified = false\nupstream = "somewhere else first"')
        edit(self.path, 'status = "harvested"', 'status = "harvested"\ncompile = "standalone"\nreason = "not run yet"')
        rep = check.run(self.t.cat())
        self.assertEqual(rep.errors, [])
        self.assertEqual([w for w in rep.warnings if "not a key" in w], [])

    def test_gate_compile_values(self):
        edit(self.path, 'status = "harvested"', 'status = "harvested"\ncompile = "sometimes"')
        self.assertTrue(any("compile 'sometimes'" in e for e in self.errors()))
        edit(self.path, 'compile = "sometimes"', 'compile = "via:kernel.hipfire.nothing"')
        self.assertTrue(any("not in the catalogue" in e for e in self.errors()))

    def test_prebuilt_object_is_hashed(self):
        blob = os.path.join(self.t.root, "catalog/hipfire/kernel/addco/src/k/add.hip")
        good = hashlib.sha256(open(blob, "rb").read()).hexdigest()
        with open(self.path, "a") as f:
            f.write(f'\n[[prebuilt]]\nfile = "src/k/add.hip"\nsha256 = "{good}"\nbytes = {os.path.getsize(blob)}\n')
        self.assertEqual(self.errors(), [])
        edit(self.path, good, "0" * 64)
        self.assertTrue(any("sha256 differs" in e for e in self.errors()))

    def test_units_file(self):
        line = next(ln for ln in slurp(self.path).splitlines() if ln.startswith("sources"))
        edit(self.path, line, 'sources = []\nunits = "units.json"')
        self.t.write("catalog/hipfire/kernel/addco/units.json", json.dumps({"units": [
            {"name": "add", "parts": [{"text": "#define N 1\n"}, {"path": "kernel/addco/src/k/add.hip"}], "flags": ["-DX"]}]}))
        cat = self.t.cat()
        self.assertEqual(sorted(cat.by_id["kernel.hipfire.addco"].units()), ["add"])
        self.assertFalse(any("units" in e for e in check.run(cat).errors))
        self.t.write("catalog/hipfire/kernel/addco/units.json", json.dumps({"units": [
            {"name": "add", "parts": [{"path": "kernel/addco/src/k/missing.hip"}]}]}))
        self.assertTrue(any("missing.hip" in e for e in self.errors()))

    def test_verified_is_for_prebuilt_kits(self):
        edit(self.path, 'status = "harvested"', 'status = "verified"\nevidence = "gate/results/x.json"')
        self.t.write("gate/results/x.json", "{}")
        self.assertTrue(any("'verified' is for prebuilt kits" in e for e in self.errors()))

    def test_harvest_none_holds_no_modules(self):
        edit(os.path.join(self.t.root, "catalog/hipfire/ORIGIN.toml"), '[licence]', 'harvest = "none"\nharvest_reason = "closed"\n[licence]')
        self.assertTrue(any("harvest = none, but the catalogue holds modules" in e for e in self.errors()))

    def test_triton_toolchain_needs_no_rocm_root(self):
        with open(os.path.join(self.t.root, "toolchains.toml"), "a") as f:
            f.write('["tri"]\nkind = "triton"\nversion = "3.8"\npython = "cp313"\ntarget = "hip:gfx1151:32"\n')
        self.assertEqual(self.errors(), [])


class RealCatalogueTests(unittest.TestCase):
    def setUp(self):
        self.cat = Catalog(repo_root())

    def test_check_has_no_errors(self):
        self.assertEqual(check.run(self.cat).errors, [])

    def test_check_warns_only_about_real_problems(self):
        """A module that includes a header the harvest left out cannot build as harvested; a cycle between
        modules of an assembly is worth knowing; a recipe binding nothing requires is a mistake in the recipe.
        Nothing else may warn: an unknown manifest key is a schema question to settle in SPEC.md, not noise to
        live with."""
        real = ("cannot build as harvested", "dependency cycle", "is bound but nothing in the assembly requires it")
        other = [w for w in check.run(self.cat).warnings if not any(r in w for r in real)]
        self.assertEqual(other, [])

    def test_variants_differ_from_the_unmodified_recipe_only_in_the_swapped_bindings(self):
        base = self.cat.recipe("qwen38-flash-next.strixite")
        expect = {"qwen38-flash-next.topk-gufo": {"logits.topk/-/-": "kernel.flerp.logits_topk_gufo"},
                  "qwen38-flash-next.dense-strixfold": {"linear.dense/bf16/bf16": "kernel.flerp.dense_bf16_strixfold"},
                  "qwen38-flash-next.router-gufo": {"moe.router/-/bf16": "kernel.flerp.moe_router_gufo"}}
        for name, swapped in expect.items():
            r = self.cat.recipe(name)
            self.assertIsNotNone(r, name)
            changed = {k: v for k, v in r.bindings.items() if base.bindings.get(k) != v and k in base.bindings}
            self.assertEqual(changed, swapped, name)
            asm = Assembly(self.cat, r)
            self.assertEqual(asm.errors, [], name)
            self.assertEqual(sorted(asm.renames), sorted(self.cat.by_id[a].overrides[0] for a in swapped.values()))
            self.assertEqual(r.gate.get("runner"), base.gate.get("runner"))

    def test_every_adapter_names_a_gate(self):
        for m in self.cat.modules:
            if m.adapter:
                ref = m.gate.get("parity", "")
                # an interface gate (cross-engine adapters), or a gate config of its own (the strixw panel adapters'
                # gate/strixw-panel-kernels.toml compares them with the kernels they adapt)
                self.assertTrue(ref.startswith("flerp:gate/"), m.id)
                self.assertTrue(os.path.isfile(os.path.join(repo_root(), ref[len("flerp:"):])), m.id)


class GateLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iface = load_gate("interface")
        cls.recipe = load_gate("strixite_recipe")

    def cfg(self, **crit):
        base = {"parity": "bit-identical", "max_err_over_bound": 1.0}
        base.update(crit)
        return {"criteria": base, "case": [
            {"name": "a", "kind": "parity"}, {"name": "tol", "kind": "parity", "require": "order-bound"},
            {"name": "edge", "kind": "parity", "require": "report"}, {"name": "k8-b", "kind": "bench"}]}

    def run_rows(self, **over):
        rows = {"a": {"bit_identical": True}, "tol": {"bit_identical": False, "err_over_bound": 0.5},
                "edge": {"bit_identical": False}, "k8-b": {"median_us": 3.0}}
        for k, v in over.items():
            rows[k.replace("_", "-") if k.startswith("k8") else k].update(v)
        return {"cases": [dict(case=n, ran=True, **r) for n, r in rows.items()]}

    def test_criteria(self):
        ok, rows, problems = self.iface.judge(self.cfg(), {"name": "p"}, self.run_rows())
        self.assertTrue(ok, problems)
        ok, _, problems = self.iface.judge(self.cfg(), {"name": "p"}, self.run_rows(a={"bit_identical": False}))
        self.assertFalse(ok)
        ok, _, problems = self.iface.judge(self.cfg(), {"name": "p"}, self.run_rows(tol={"err_over_bound": 1.5}))
        self.assertFalse(ok)
        ok, _, _ = self.iface.judge(self.cfg(), {"name": "p"}, self.run_rows(edge={"structure_ok": False}))
        self.assertTrue(ok)                       # a reported case never decides

    def test_a_case_that_did_not_run_fails(self):
        run = self.run_rows()
        run["cases"] = run["cases"][1:]
        ok, _, problems = self.iface.judge(self.cfg(), {"name": "p"}, run)
        self.assertFalse(ok)
        self.assertIn("a: did not run", problems[0])

    def test_an_adapter_must_have_launched_the_foreign_kernel(self):
        run = self.run_rows()
        ok, _, problems = self.iface.judge(self.cfg(), {"name": "p", "adapter": True}, run)
        self.assertFalse(ok)
        self.assertIn("never launched", problems[0])
        run["adapter_tallies"] = [{"foreign_launches": 10, "origin_launches": 3}]
        ok, _, problems = self.iface.judge(self.cfg(), {"name": "p", "adapter": True}, run)
        self.assertFalse(ok)                      # more fall-backs than the config allows
        ok, _, _ = self.iface.judge(self.cfg(adapter_origin_launches_allowed=3), {"name": "p", "adapter": True}, run)
        self.assertTrue(ok)

    def test_provider_case_filter(self):
        cfg = self.cfg()
        self.assertEqual([c["name"] for c in self.iface.provider_cases(cfg, {"cases": "^k8-"})], ["k8-b"])
        self.assertEqual(len(self.iface.provider_cases(cfg, {})), 4)

    def test_gate_module_finds_the_interface_gates_of_a_module(self):
        found = dict(self.iface.configs_for_module("kernel.gufo.qwen_sampling"))
        self.assertEqual(found, {"logits-topk": ["gufo"]})             # the foreign module behind an adapter
        found = dict(self.iface.configs_for_module("kernel.flerp.residual_add_hipfire"))
        self.assertEqual(found, {"residual-add": ["hipfire"]})         # the adapter itself
        self.assertEqual(self.iface.configs_for_module("kernel.strixite.gdn"), [])     # none: compile-only gate

    def test_every_gate_config_states_its_criteria_and_real_modules(self):
        cat = Catalog(repo_root())
        names = [f[:-5] for f in os.listdir(self.iface.CONFIG_DIR) if f.endswith(".toml") and f != "host.toml"]
        self.assertEqual(sorted(names), ["dense-bf16", "gguf-gemv", "logits-topk", "moe-router", "residual-add",
                                        "strixw-decode"])
        for name in names:
            cfg = self.iface.load_config(name)
            self.assertIn(cfg["criteria"]["parity"], self.iface.PARITY_LEVELS, name)
            self.assertTrue(os.path.isfile(os.path.join(repo_root(), cfg["driver"], "module.toml")), name)
            kinds = {c["kind"] for c in cfg["case"]}
            self.assertEqual(kinds, {"parity", "bench"}, name)
            for p in cfg["provider"]:
                self.assertIn(p["module"], cat.by_id, name)
                self.assertEqual(bool(p.get("adapter")), bool(cat.by_id[p["module"]].adapter), p["module"])
                recipe = self.iface.driver_recipe(cat, cfg, p)
                self.assertEqual(Assembly(cat, recipe).errors, [], f"{name}/{p['name']}")

    def test_order_evidence(self):
        rows = [{"kind": "parity", "case": "x", "order_emulation": "lanes"}, {"kind": "parity", "case": "y", "order_emulation": "none"}]
        ev = self.iface.order_evidence(rows)
        self.assertEqual((ev["cases"], ev["reproduced"], ev["order_only"]), (2, 1, False))

    def test_tally_lines(self):
        text = ("flerp-adapter kernel.flerp.a provider=kernel.x:K foreign_launches=1 origin_launches=0 (first launch through the provider)\n"
                "noise\nflerp-adapter kernel.flerp.a provider=kernel.x:K foreign_launches=42 origin_launches=2 (exit)\n")
        self.assertEqual(self.iface.TALLY.findall(text), [("kernel.flerp.a", "kernel.x:K", "42", "2")])
        path = os.path.join(repo_root(), "build", "test-tally.log")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
        self.addCleanup(os.remove, path)
        self.assertEqual(self.recipe.adapter_tallies(path)["kernel.flerp.a"]["foreign_launches"], 42)

    # linear_q4p's label holds spaces and parentheses; the strixw-decode GPU run of 2026-11-24 printed these lines
    # and the old provider=(\S+) read none of them, so both panel adapters failed "never launched" (card a9ac7b1c0f).
    PANEL = ("flerp-adapter kernel.flerp.linear_q4p provider=panel kernels (Q4Panel16) foreign_launches=1 origin_launches=0 "
             "(first launch through the provider)\nstrixw_decode {}\n"
             "flerp-adapter kernel.flerp.linear_q4p provider=panel kernels (Q4Panel16) foreign_launches=9013 "
             "origin_launches=0 (exit)\n")

    def test_tally_label_with_spaces(self):
        self.assertEqual(self.iface.adapter_tallies(self.PANEL),
                         [{"adapter": "kernel.flerp.linear_q4p", "provider": "panel kernels (Q4Panel16)",
                           "foreign_launches": 9013, "origin_launches": 0}])
        path = os.path.join(repo_root(), "build", "test-tally-panel.log")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(self.PANEL)
        self.addCleanup(os.remove, path)
        t = self.recipe.adapter_tallies(path)["kernel.flerp.linear_q4p"]
        self.assertEqual((t["provider"], t["foreign_launches"], t["line"]), ("panel kernels (Q4Panel16)", 9013, "exit"))

    def test_rejudge_reads_the_recorded_driver_output(self):
        root = os.path.join(repo_root(), "build", "test-rejudge")
        shutil.rmtree(root, ignore_errors=True)
        self.addCleanup(shutil.rmtree, root, True)
        os.makedirs(os.path.join(root, "p"))
        with open(os.path.join(root, "p", "driver.txt"), "w") as f:
            f.write(self.PANEL)
        with open(os.path.join(root, "p", "driver.json"), "w") as f:
            json.dump(self.run_rows()["cases"], f)
        cfg = dict(self.cfg(), _sha256="x", provider=[{"name": "p", "adapter": True}])
        doc = {"config_sha256": "x", "complete": True, "pass": False,
               "providers": {"p": {"pass": False, "problems": ["the adapter never launched the foreign provider"],
                                   "run": {"ran": True, "rc": 0, "adapter_tallies": []}}}}
        doc = self.iface.rejudge(cfg, doc, root)
        self.assertTrue(doc["pass"], doc["providers"]["p"]["problems"])
        self.assertEqual(doc["providers"]["p"]["run"]["adapter_tallies"][0]["foreign_launches"], 9013)
        with open(os.path.join(root, "p", "driver.txt"), "w") as f:
            f.write(self.PANEL.replace("origin_launches=0 (exit)", "origin_launches=5 (exit)"))
        self.assertFalse(self.iface.rejudge(cfg, doc, root)["pass"])          # a fall-back still fails
        with self.assertRaises(SystemExit):
            self.iface.rejudge(dict(cfg, _sha256="y"), doc, root)            # other criteria: re-run, not re-judge

    def test_variant_config_inherits(self):
        cfg = self.recipe.load_config(os.path.join(repo_root(), "gate", "qwen38-flash-next.dense-strixfold.toml"))
        base = self.recipe.load_config(os.path.join(repo_root(), "gate", "qwen38-flash-next.strixite.toml"))
        self.assertEqual(cfg["run"], base["run"])                  # the same requests, repeats and tolerance
        self.assertEqual(cfg["server"], base["server"])
        self.assertEqual(cfg["unit_test"], base["unit_test"])
        self.assertNotEqual(cfg["host"]["scratch"], base["host"]["scratch"])
        own = ("scratch", "lease_owner")              # its own scratch directory and its own name on the lease
        self.assertEqual({k: v for k, v in cfg["host"].items() if k not in own},
                         {k: v for k, v in base["host"].items() if k not in own})

    def test_compare_reports_first_divergence(self):
        def side(gen):
            return {"runs": [{"prompt": "p", "repeat": 0, "prompt_n": 3, "prompt_hash": "h", "generated": gen,
                              "generated_n": len(gen), "generated_hash": str(gen), "steps": [],
                              "timings": {"predicted_per_second": 10.0, "prompt_per_second": 100.0}}]}
        rows, _, identical, _ = self.recipe.compare(side([1, 2, 3, 4]), side([1, 2, 9, 4]), 0.05)
        self.assertFalse(identical)
        self.assertEqual((rows[0]["first_divergence"], rows[0]["tokens_agreeing"]), (2, 2))
        rows, _, identical, _ = self.recipe.compare(side([1, 2, 3]), side([1, 2, 3]), 0.05)
        self.assertTrue(identical)
        self.assertEqual((rows[0]["first_divergence"], rows[0]["tokens_agreeing"]), (None, 3))


if __name__ == "__main__":
    unittest.main()
