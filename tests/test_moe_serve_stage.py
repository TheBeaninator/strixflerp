"""Stage P3 of docs/PHASE5-FOURNODE-DESIGN.md without a GPU: the stage recipe resolves and assembles (with the ABI
headers in its tree), the adapter and the text module define every call of their headers, the greedy sampler, and
the gate's judgements (gate/moe_serve_stage.py)."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gate"))
import moe_serve_stage as M                           # noqa: E402
from flerplib import assemble_cmake                   # noqa: E402
from flerplib.catalog import Catalog                  # noqa: E402
from flerplib.resolve import Assembly                 # noqa: E402

RECIPE = "qwen36-35b-a3b.flerp-serve-stage"
FLERP = os.path.join(ROOT, "catalog", "flerp")


def declared(header, include=os.path.join(ROOT, "include")):
    with open(os.path.join(include, "flerp", header)) as f:
        text = re.sub(r"/\*.*?\*/", "", f.read(), flags=re.S)
    return set(re.findall(r"\b(flerp_\w+)\s*\(", text)) - {"flerp_caps", "flerp_stats"}


def defined(path):
    with open(path) as f:
        text = f.read()
    return set(re.findall(r"^\S[^\n;]*\b(flerp_\w+)\s*\([^;]*?\)\s*\{", text, flags=re.M))


class Recipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = Catalog(ROOT)

    def test_four_programs_one_adapter(self):
        asm = Assembly(self.cat, self.cat.recipe(RECIPE))
        self.assertEqual(asm.errors, [])
        self.assertEqual(set(asm.binaries), {"strixflerp-qwen36-35b-a3b.flerp-serve-stage", "flerp-qwen35moe-stage",
                                             "flerp-executor-conformance", "flerp-qwen35moe-generate"})
        for name in ("strixflerp-qwen36-35b-a3b.flerp-serve-stage", "flerp-qwen35moe-stage", "flerp-executor-conformance"):
            self.assertIn("executor.flerp.qwen35moe_abi", asm.binaries[name][2], name)
            self.assertIn("runtime.flerp.stage_relay", asm.binaries[name][2], name)
        front = asm.binaries["strixflerp-qwen36-35b-a3b.flerp-serve-stage"][2]
        for m in ("serve.flerp.http", "runtime.flerp.text_gufo", "runtime.flerp.sampler_greedy", "runtime.flerp.decode_loop"):
            self.assertIn(m, front)

    def test_tree_holds_the_abi_headers(self):
        with tempfile.TemporaryDirectory() as d:
            asm, errors = assemble_cmake.generate(self.cat, self.cat.recipe(RECIPE), d)
            self.assertEqual(errors, [])
            with open(os.path.join(d, "assembly.json")) as f:
                files = json.load(f)["files"]
            for h in ("executor.h", "decode_loop.h", "text.h", "sampler.h"):
                self.assertEqual(files[f"include/flerp/{h}"]["module"], "-")
                self.assertTrue(os.path.isfile(os.path.join(d, "tree", "include", "flerp", h)))
            for p in ("flerp/qwen35moe_stage/CMakeLists.txt", "flerp/qwen35moe_abi/abi.cpp", "flerp/text/text_gufo.cpp",
                      "flerp/serve/main.cpp", "serve/json.cpp", "third_party/cpp-httplib/httplib.h", "flerp/qwen35moe_tool/stage_driver.cpp"):
                self.assertIn(p, files)

    def test_build_file_is_the_p2_one_up_to_its_program(self):
        """The stage recipe's CMakeLists.txt repeats stage P2's statement for statement before its own programs."""
        with open(os.path.join(FLERP, "tool/qwen35moe_generate/src/flerp/qwen35moe_tool/CMakeLists.txt")) as f:
            p2 = f.read()
        with open(os.path.join(FLERP, "tool/qwen35moe_stage/src/flerp/qwen35moe_stage/CMakeLists.txt")) as f:
            mine = f.read()
        body = p2[p2.index("\ncmake_minimum_required(") + 1:p2.index("# The one program.")]
        body = body.replace("project(flerp_qwen35moe ", "project(flerp_qwen35moe_stage ").replace("flerp_qwen35moe_SOURCE_DIR", "flerp_qwen35moe_stage_SOURCE_DIR")
        self.assertIn(body, mine)

    def test_a_header_that_is_not_in_the_repository_is_refused(self):
        r = self.cat.recipe(RECIPE)
        saved = r.data["build"]["headers"]
        try:
            r.data["build"]["headers"] = saved + ["include/flerp/no_such.h"]
            _, errors = assemble_cmake.plan(self.cat, r, Assembly(self.cat, r))
            self.assertTrue(any("not a file of the repository" in e for e in errors), errors)
        finally:
            r.data["build"]["headers"] = saved


class Sources(unittest.TestCase):
    def test_adapter_defines_every_executor_call(self):
        # the adapter reports ABI version 2 (like strixite_abi and strixfold_abi): every call of that version, from the
        # frozen version 2 header; the version 3 calls come with the card that implements them
        src = os.path.join(FLERP, "executor/qwen35moe_abi/src/flerp/qwen35moe_abi/abi.cpp")
        with open(src) as f:
            self.assertIn("c.abi = 2;", f.read())
        self.assertEqual(declared("executor.h", os.path.join(ROOT, "tests", "cpp", "abi_v2")) - defined(src), set())

    def test_text_module_defines_every_text_call(self):
        self.assertEqual(declared("text.h") - defined(os.path.join(FLERP, "runtime/text_gufo/src/flerp/text/text_gufo.cpp")), set())

    def test_open_keys_are_documented(self):
        with open(os.path.join(FLERP, "executor/qwen35moe_abi/src/flerp/qwen35moe_abi/stages.cpp")) as f:
            src = f.read()
        keys = re.findall(r'key == "(\w+)"', src)
        with open(os.path.join(FLERP, "executor/qwen35moe_abi/module.toml")) as f:
            notes = f.read()
        self.assertGreaterEqual(len(keys), 12)
        for k in keys:
            self.assertIn(k, notes)


@unittest.skipUnless(shutil.which("cc"), "no C compiler on this host")
class GreedySampler(unittest.TestCase):
    def test_argmax_lower_id_on_a_tie_and_temperature_refused(self):
        prog = r'''
#include "flerp/sampler.h"
#include <stdio.h>
int main(void) {
    flerp_sampler *s = 0;
    flerp_sampling p = {sizeof(p), 0.7, 1, 0, 0, 0};
    if (flerp_sampler_new(&p, 6, &s) != FLERP_E_ARGUMENT) return 1;
    p.temperature = 0;
    if (flerp_sampler_new(&p, 5, &s) != FLERP_OK) return 2;
    float full[12] = {0, 3, 1, 3, 2, 9,   4, 4, 1, 0, 0, 0};
    flerp_rows r = {0};
    r.size = sizeof(r), r.rows = 2, r.full = full, r.full_cap = 12;
    flerp_token t = -1;
    if (flerp_sampler_pick(s, FLERP_OUT_FULL, &r, 0, 0, 6, &t) != FLERP_OK || t != 1) return 3;   /* id 5 is past n_valid */
    if (flerp_sampler_pick(s, FLERP_OUT_FULL, &r, 1, 0, 6, &t) != FLERP_OK || t != 0) return 4;
    flerp_sampler_free(s);
    printf("ok\n");
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as d:
            src, exe = os.path.join(d, "t.c"), os.path.join(d, "t")
            with open(src, "w") as f:
                f.write(prog)
            subprocess.run(["cc", "-std=c11", "-Wall", "-Werror", "-I", os.path.join(ROOT, "include"), src,
                            os.path.join(FLERP, "runtime/sampler_greedy/src/flerp/sampler/sampler_greedy.c"), "-o", exe], check=True)
            p = subprocess.run([exe], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)


class Judgements(unittest.TestCase):
    P1 = {"a": list(range(128)), "b": list(range(5, 133))}

    def doc(self):
        cases = [{"name": n, "check_reset_position": True, "check_plain_position": True, "plain": ids[:128], "loop": ids[:127],
                  "loop_completion_tokens": 128, "loop_finish": "length"} for n, ids in self.P1.items()]
        return {"abi": 2, "failures": 0, "caps": {"positions": 4096, "deliveries": 5, "flags": 0, "seqs": 1, "max_draft": 0, "snapshots": 0,
                                                  "candidates": 0}, "cases": cases}

    def test_conformance_pass_and_each_failure(self):
        self.assertEqual(M.judge_conformance(self.doc(), self.P1, ["a", "b"], 4096, 128), [])
        d = self.doc()
        d["cases"][0]["plain"][7] = -1
        self.assertTrue(any("plain differs" in w and "at 7" in w for w in M.judge_conformance(d, self.P1, ["a", "b"], 4096, 128)))
        d = self.doc()
        d["cases"][1]["loop"] = d["cases"][1]["loop"][:100]
        self.assertTrue(any("loop ids" in w for w in M.judge_conformance(d, self.P1, ["a", "b"], 4096, 128)))
        d = self.doc()
        d["caps"]["flags"] = 64
        self.assertTrue(any("caps.flags" in w for w in M.judge_conformance(d, self.P1, ["a", "b"], 4096, 128)))
        d = self.doc()
        d["cases"][0]["check_plain_position"] = False
        self.assertTrue(M.judge_conformance(d, self.P1, ["a", "b"], 4096, 128))
        self.assertTrue(M.judge_conformance(self.doc(), self.P1, ["a", "b", "c"], 4096, 128))

    def test_health(self):
        h = {"status": "ok", "slots": 1, "busy": False, "busy_slots": 0, "queued": 0, "context_length": 4096, "model": "m"}
        self.assertEqual(M.judge_health_idle(200, h, 4096, "m"), [])
        self.assertTrue(M.judge_health_idle(200, dict(h, busy=True), 4096, "m"))
        self.assertTrue(M.judge_health_idle(200, dict(h, context_length=2048), 4096, "m"))
        self.assertTrue(M.judge_health_idle(503, h, 4096, "m"))

    def test_refusal_shape(self):
        ok = {"error": {"message": "no", "type": "invalid_request_error", "param": "n", "code": "not_served"}}
        self.assertEqual(M.judge_refusal("x", 400, ok), [])
        self.assertTrue(M.judge_refusal("x", 500, ok))
        self.assertTrue(M.judge_refusal("x", 400, {"error": "no"}))
        self.assertTrue(M.judge_refusal("x", 400, {"error": {"message": "", "type": "invalid_request_error", "param": None, "code": None}}))

    def test_completion_and_chats(self):
        resp = {"flerp": {"prompt_token_ids": [1, 2], "token_ids": list(range(128))}, "choices": [{"finish_reason": "length"}]}
        self.assertEqual(M.judge_completion("a", 200, resp, [1, 2], list(range(128)), 128), [])
        self.assertTrue(M.judge_completion("a", 200, resp, [1, 3], list(range(128)), 128))
        self.assertTrue(M.judge_completion("a", 200, resp, [1, 2], list(range(1, 129)), 128))
        run = {"c": {"status": 200, "prompt_ids": [1], "token_ids": [4, 5]}}
        self.assertEqual(M.judge_chats(run, {"c": [4, 5]}, run), [])
        self.assertTrue(M.judge_chats(run, {"c": [4, 6]}, run))
        self.assertTrue(M.judge_chats(run, {}, None))
        self.assertTrue(M.judge_chats(run, {"c": [4, 5]}, {"c": {"prompt_ids": [2]}}))

    def test_exit(self):
        self.assertEqual(M.judge_exit("r", 0, "shutdown"), [])
        self.assertTrue(M.judge_exit("r", -11, ""))
        self.assertTrue(M.judge_exit("r", 139, ""))
        self.assertTrue(M.judge_exit("r", None, ""))
        self.assertTrue(M.judge_exit("r", 0, "Segmentation fault (core dumped)"))


if __name__ == "__main__":
    unittest.main()
