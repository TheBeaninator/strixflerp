"""The Crucible origin: the make assembler and tree-relative include resolution. python3 -m unittest tests.test_crucible"""
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from flerplib import assemble, check, includes
from flerplib.catalog import Catalog, repo_root


class ResolveTree(unittest.TestCase):
    def test_parent_relative_include_found_in_another_module(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "b", "src", "engine", "gfx1151"))
        target = os.path.join(d, "b", "src", "engine", "gfx1151", "common.h")
        open(target, "w").close()
        # included from engine/exl3/exl3.h of module a, which has no engine/gfx1151 directory
        self.assertEqual(includes.resolve_tree("engine/exl3", "../gfx1151/common.h", [os.path.join(d, "b", "src")]), target)
        self.assertIsNone(includes.resolve_tree("engine", "../../outside.h", [os.path.join(d, "b", "src")]))


class CrucibleRecipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = Catalog(repo_root())

    def test_origin_checks_clean(self):
        rep = check.run(self.cat, "crucible")
        self.assertEqual(rep.errors, [])
        self.assertEqual([w for w in rep.warnings if "catalog/crucible/" in w], [])

    def test_make_assembly(self):
        r = self.cat.recipe("qwen38-flash-next.crucible")
        out = tempfile.mkdtemp()
        with redirect_stdout(io.StringIO()):
            asm, errors = assemble.generate(self.cat, r, out)
        self.assertEqual(errors, [])
        man = json.load(open(os.path.join(out, "assembly.json")))
        self.assertEqual(man["system"], "make")
        for f in ("Makefile", "engine/engine.hip", "tools/hipcc_checked.sh", "engine/exl3/kernels/exl3_gemm.h"):
            self.assertIn(f, man["files"])
            self.assertTrue(os.path.isfile(os.path.join(out, "tree", f)))
        self.assertNotIn("server/api.py", man["files"])          # the Python server is not in the engine build
        sh = open(os.path.join(out, "build.sh")).read()
        self.assertIn("build/engine/crucible.mono", sh)
        self.assertIn("ionice -c3 nice -n 19 make", sh)


if __name__ == "__main__":
    unittest.main()
