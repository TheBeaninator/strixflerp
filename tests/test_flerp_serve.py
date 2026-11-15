"""Phase 3, the common front end: the recipes that bind it, the Zig assembler's C/C++ link table, the shared decode
loop against the stand-in executor (when a C compiler is at hand), and the gate's own comparisons."""
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
from flerplib import assemble_zig                     # noqa: E402
from flerplib.catalog import Catalog                  # noqa: E402
from flerplib.resolve import Assembly                 # noqa: E402

SERVE = ("qwen38-flash-next.flerp-serve", "qwen38-flash-next.flerp-serve-fake", "dsv41-flash.flerp-serve-tp2")
FLERP = os.path.join(ROOT, "catalog", "flerp")


class Recipes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = Catalog(ROOT)

    def test_three_recipes_bind_one_front_end(self):
        """Each recipe resolves, and each binary's closure holds serve.flerp.http with a different executor."""
        executors = {}
        for name in SERVE:
            asm = Assembly(self.cat, self.cat.recipe(name))
            self.assertEqual(asm.errors, [], name)
            (_, _, ids), = asm.binaries.values()
            self.assertIn("serve.flerp.http", ids)
            self.assertIn("runtime.flerp.decode_loop", ids)
            executors[name] = self.cat.recipe(name).bindings["executor.abi/flerp-v1/-"]
            for iface in ("text.model/flerp-v1/-", "sample.host/flerp-v1/-"):
                self.assertIn(self.cat.recipe(name).bindings[iface], ids)
        self.assertEqual(len(set(executors.values())), 3, executors)

    def test_front_end_names_no_engine(self):
        """One source tree: the front end's sources include nothing of an executor and branch on no engine."""
        src = os.path.join(FLERP, "serve", "http", "src", "flerp", "serve")
        for f in os.listdir(src):
            with open(os.path.join(src, f)) as fh:
                text = fh.read()
            code = re.sub(r"//[^\n]*", "", text)
            self.assertEqual(re.findall(r"#\s*if\w*\s+(\w+)", code), ["FLERP_HOSTED"] if f == "main.cpp" else [], f)
            for inc in re.findall(r'#include [<"]([^">]+)', code):
                self.assertFalse(re.search(r"qwen|dsv41|engine|backend|strixfold|lanes", inc), f"{f} includes {inc}")
            self.assertNotRegex(code, r"(?i)strixfold|qwen4exp|dsv41")

    def test_conformance_driver_and_front_end_share_the_loop(self):
        """The driver holds no draft loop of its own: both callers reach flerp_decode_loop."""
        with open(os.path.join(FLERP, "executor/conformance/src/flerp/executor/conformance.c")) as f:
            driver = f.read()
        with open(os.path.join(FLERP, "serve/http/src/flerp/serve/main.cpp")) as f:
            front = f.read()
        for text in (driver, front):
            self.assertIn("flerp_decode_loop(", text)
            self.assertIn("flerp_feed_ids(", text)
        self.assertNotIn("draft_chain", driver)             # the engine's draft rule lives in the shared module only
        for call in ("flerp_draft_step(", "flerp_draft_hold(", "flerp_verify(", "flerp_keep(", "flerp_keep_prefix(", "flerp_drop("):
            self.assertNotIn(call, front)

    def test_zig_link_table(self):
        """[build] c_link: the C and C++ units of the binary's closure, each with its module's include roots."""
        r = self.cat.recipe("dsv41-flash.flerp-serve-tp2")
        asm = Assembly(self.cat, r)
        text, errors = assemble_zig.c_link(self.cat, r, asm)
        self.assertEqual(errors, [])
        files = re.findall(r'\.file = "([^"]+)"', text)
        self.assertEqual(sorted(files), ["flerp/decode_loop.c", "flerp/reply/reply_none.c", "flerp/sampler/sampler_none.c", "flerp/serve/main.cpp",
                                         "flerp/serve/request.cpp", "serve/json.cpp"])
        main = next(line for line in text.splitlines() if "flerp/serve/main.cpp" in line)
        self.assertIn('"-std=c++20"', main)
        self.assertIn('.system = &.{"third_party/cpp-httplib"}', main)
        self.assertIn('"include"', main)
        with tempfile.TemporaryDirectory() as d:
            asm2, errors = assemble_zig.generate(self.cat, r, d)
            self.assertEqual(errors, [])
            self.assertTrue(os.path.isfile(os.path.join(d, "tree", "flerp", "link.zig")))
            self.assertTrue(os.path.isfile(os.path.join(d, "tree", "include", "flerp", "text.h")))
            with open(os.path.join(d, "assembly.json")) as f:
                self.assertEqual(json.load(f)["files"]["flerp/link.zig"]["module"], "-")
            self.assertEqual(assemble_zig.verify_tree(d), 0)

    def test_recipes_without_c_link_get_no_generated_file(self):
        r = self.cat.recipe("dsv41-flash.strixfold-tp2")
        tree, errors, _ = assemble_zig.plan(self.cat, r, Assembly(self.cat, r))
        self.assertEqual(errors, [])
        self.assertNotIn("flerp/link.zig", tree)


@unittest.skipUnless(shutil.which("cc"), "no C compiler on this host")
class LoopOverTheStandIn(unittest.TestCase):
    """runtime.flerp.decode_loop driven by the conformance driver over executor.flerp.fake_abi, whose tokens do not
    depend on how ids are grouped into forwards."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="flerp-loop-")
        srcs = [os.path.join(FLERP, p) for p in ("executor/conformance/src/flerp/executor/conformance.c",
                                                 "runtime/decode_loop/src/flerp/decode_loop.c",
                                                 "executor/fake_abi/src/flerp/executor/fake_abi.c")]
        cls.exe = os.path.join(cls.dir, "conformance")
        subprocess.run(["cc", "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-Wall", "-Werror", "-I", os.path.join(ROOT, "include")]
                       + srcs + ["-lm", "-o", cls.exe], check=True)
        cls.ids = os.path.join(cls.dir, "p.ids")
        with open(cls.ids, "w") as f:
            f.write(" ".join(str(i) for i in range(1, 31)))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def run_case(self, draft, delivery, extra=()):
        out = os.path.join(self.dir, f"{draft}-{delivery}.json")
        r = subprocess.run([self.exe, "--out", out, "--delivery", delivery, "vocab=5000", f"draft={draft}", *(extra or ("reply=200",)),
                            "--case", "c", "--ids", self.ids, "--paths", "plain,speculative,loop-drop", "--new", "60",
                            "--max-tokens", "61", "--mtp-draft", "7", "--mtp-margin", "1.5", "--end-ids", "4999", "--cut", "11"],
                           capture_output=True, text=True)
        with open(out) as f:
            return r.returncode, json.load(f)

    def test_every_draft_form_and_delivery_gives_the_plain_tokens(self):
        want = None
        for draft in ("none", "step", "held"):
            for delivery in ("candidates", "tokens", "full"):
                rc, j = self.run_case(draft, delivery)
                c = j["cases"][0]
                self.assertEqual((rc, j["failures"]), (0, 0), (draft, delivery))
                self.assertEqual(c["loop"], c["plain"][:len(c["loop"])], (draft, delivery))
                want = want or c["loop"]
                self.assertEqual(c["loop"], want, (draft, delivery))
                if draft != "none":
                    self.assertGreater(c["loop_verifies"], 0)
                    self.assertTrue(0 < c["loop_accepted"] < c["loop_drafted"])
                    self.assertGreater(c["loop_drop_dropped_verifies"], 0)
                self.assertEqual(sum(s[2] for s in c["loop_steps"]), len(c["loop"]))

    def test_end_of_turn_stops_the_loop(self):
        rc, j = self.run_case("held", "tokens", ("reply=20", "end=4999"))
        c = j["cases"][0]
        self.assertEqual(c["loop_finish"], "stop")
        self.assertEqual(len(c["loop"]), 20)            # the end id is counted, never forwarded
        self.assertEqual(c["loop_completion_tokens"], 21)

    def test_a_version_1_caller_is_not_written_past_its_struct(self):
        """flerp_get_caps fills a caller's shorter struct as far as it goes (include/flerp/executor.h, Versions)."""
        prog = os.path.join(self.dir, "v1.c")
        with open(prog, "w") as f:
            f.write(r'''
#include "flerp/executor.h"
#include <stdio.h>
#include <string.h>
int main(void) {
    flerp_kv kv[] = {{"vocab", "100"}, {"sampled", "on"}, {"top_k_max", "8"}};
    flerp_open_args oa = {sizeof(oa), "t", kv, 3};
    flerp_executor *ex;
    if (flerp_open(&oa, &ex) != FLERP_OK) return 1;
    struct { flerp_caps c; unsigned char guard[16]; } v2;
    union { flerp_caps c; unsigned char raw[sizeof(flerp_caps)]; } v1;
    const size_t v1_size = FLERP_HAS(flerp_caps, flags);
    memset(&v1, 0xAB, sizeof(v1));
    v1.c.size = (uint32_t)v1_size;
    if (flerp_get_caps(ex, &v1.c) != FLERP_OK) return 2;
    for (size_t i = v1_size; i < sizeof(v1); ++i) if (v1.raw[i] != 0xAB) return 3;   /* written past a v1 struct */
    if (v1.c.size != v1_size || v1.c.abi != FLERP_EXECUTOR_ABI || v1.c.deliveries != 7) return 4;
    memset(&v2, 0, sizeof(v2));
    memset(v2.guard, 0xCD, sizeof(v2.guard));
    v2.c.size = sizeof(v2.c) + 8;                                                    /* a caller newer than the executor */
    if (flerp_get_caps(ex, &v2.c) != FLERP_OK) return 5;
    for (size_t i = 0; i < sizeof(v2.guard); ++i) if (v2.guard[i] != 0xCD) return 6;
    if (v2.c.size != sizeof(flerp_caps) || v2.c.top_k_max != 8 || !(v2.c.sampling & FLERP_SAMPLE_TOP_K)) return 7;
    v1.c.size = 8;
    if (flerp_get_caps(ex, &v1.c) != FLERP_E_ARGUMENT) return 8;                       /* shorter than version 1 */
    flerp_close(ex);
    return 0;
}
''')
        exe = os.path.join(self.dir, "v1")
        subprocess.run(["cc", "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-I", os.path.join(ROOT, "include"), prog,
                        os.path.join(FLERP, "executor/fake_abi/src/flerp/executor/fake_abi.c"), "-lm", "-o", exe], check=True)
        self.assertEqual(subprocess.run([exe]).returncode, 0)


@unittest.skipUnless(shutil.which("cc"), "no C compiler on this host")
class Version3(unittest.TestCase):
    """include/flerp/executor.h version 3 over executor.flerp.fake_abi (tests/cpp/fake_abi_v3_selftest.c): every
    version 3 call, a version 2 caller (built against the frozen version 2 header in tests/cpp/abi_v2) against the
    version 3 executor, and a version 3 caller against the frozen version 2 executor."""
    CPP = os.path.join(ROOT, "tests", "cpp")
    FAKE = os.path.join(FLERP, "executor/fake_abi/src/flerp/executor/fake_abi.c")
    CC = ["cc", "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-Wall", "-Wextra", "-Werror"]

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="flerp-abi3-")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def build_and_run(self, name, caller_inc, exec_src, exec_inc, define=None):
        """Caller and executor compiled each against its own header, then linked: as two separately built parts."""
        objs = []
        for src, inc, extra in ((os.path.join(self.CPP, "fake_abi_v3_selftest.c"), caller_inc, [f"-D{define}"] if define else []),
                                (exec_src, exec_inc, [])):
            o = os.path.join(self.dir, f"{name}-{len(objs)}.o")
            subprocess.run(self.CC + ["-I", inc] + extra + ["-c", src, "-o", o], check=True)
            objs.append(o)
        exe = os.path.join(self.dir, name)
        subprocess.run(["cc"] + objs + ["-lm", "-o", exe], check=True)
        r = subprocess.run([exe], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("0 failures", r.stdout)

    def test_header_says_version_3(self):
        with open(os.path.join(ROOT, "include", "flerp", "executor.h")) as f:
            self.assertIn("#define FLERP_EXECUTOR_ABI 3", f.read())
        with open(os.path.join(self.CPP, "abi_v2", "flerp", "executor.h")) as f:
            self.assertIn("#define FLERP_EXECUTOR_ABI 2", f.read())

    def test_every_version_3_call(self):
        inc = os.path.join(ROOT, "include")
        self.build_and_run("v3", inc, self.FAKE, inc)

    def test_version_2_caller_against_version_3_executor(self):
        self.build_and_run("v2-caller", os.path.join(self.CPP, "abi_v2"), self.FAKE, os.path.join(ROOT, "include"), "V2_CALLER")

    def test_version_3_caller_against_version_2_executor(self):
        v2 = os.path.join(self.CPP, "abi_v2")
        self.build_and_run("v3-on-v2", os.path.join(ROOT, "include"), os.path.join(v2, "fake_abi_v2.c"), v2, "V3_ON_V2")

    def test_the_real_adapters_report_version_2(self):
        """Until their own cards implement the version 3 calls, the adapters say version 2 and fill only its struct."""
        for path in ("executor/strixite_abi/src/flerp/executor/strixite_abi.cpp", "executor/qwen35moe_abi/src/flerp/qwen35moe_abi/abi.cpp",
                     "executor/strixfold_abi/src/flerp/executor/strixfold_abi.zig"):
            with open(os.path.join(FLERP, path)) as f:
                text = f.read()
            self.assertRegex(text, r"abi = 2;", path)
            self.assertNotIn("FLERP_EXECUTOR_ABI;", text, path)


class GateComparisons(unittest.TestCase):
    def setUp(self):
        import flerp_serve
        self.g = flerp_serve
        self.native = {"prompt": [1, 2, 3], "generated": [7, 8, 9], "steps": [[2, 2, 3]], "finish_reason": "length", "split": False,
                       "text": "abc", "usage": {"prompt_tokens": 3, "completion_tokens": 4}, "cached_tokens": 0,
                       "timings": {"predicted_per_second": 50.0}}
        self.resp = {"choices": [{"text": "abc", "finish_reason": "length"}], "usage": {"prompt_tokens": 3, "completion_tokens": 4},
                     "timings": {"predicted_per_second": 49.0},
                     "flerp": {"prompt_token_ids": [1, 2, 3], "token_ids": [7, 8, 9, 10], "sequence_ids": [7, 8, 9], "steps": [[2, 2, 3]],
                               "delivery": "candidates", "sampled_by": "host", "prompt_cut": -1, "verifies": 1, "drafted": 2, "accepted": 2}}

    def test_equal_runs_pass_and_each_difference_fails(self):
        spec = {"name": "x"}
        self.assertTrue(self.g.compare(spec, self.native, self.resp, False)["pass"])
        for change in ({"sequence_ids": [7, 8, 5]}, {"steps": [[1, 1, 2], [0, 0, 1]]}, {"prompt_token_ids": [1, 2]}):
            resp = dict(self.resp, flerp=dict(self.resp["flerp"], **change))
            rec = self.g.compare(spec, self.native, resp, False)
            self.assertFalse(rec["pass"], change)
        self.assertEqual(self.g.compare(spec, self.native, dict(self.resp, flerp=dict(self.resp["flerp"], sequence_ids=[7, 8, 5])), False)
                         ["first_token_difference"], 2)
        self.assertFalse(self.g.compare(spec, dict(self.native, text="abd"), self.resp, False)["pass"])        # completions: text required
        self.assertFalse(self.g.compare(spec, dict(self.native, cached_tokens=3), self.resp, False)["pass"])  # the native run resumed
        self.assertFalse(self.g.compare(spec, dict(self.native, steps=[[255, 0, 3]]), self.resp, False)["pass"])  # the engine fed ids

    def test_a_stream_must_join_to_the_response(self):
        chunk = lambda text, ids, fin=None: {"choices": [{"text": text, "finish_reason": fin}], "flerp": {"token_ids": ids}}
        last = {"choices": [], "usage": self.resp["usage"], "timings": {"predicted_per_second": 48.0}, "flerp": self.resp["flerp"]}
        good = (200, [chunk("ab", [7, 8]), chunk("c", [9, 10]), chunk("", [], "length"), last], True, None)
        self.assertTrue(self.g.compare({"name": "x"}, self.native, self.resp, False, good)["pass"])
        short = (200, [chunk("ab", [7, 8]), chunk("", [], "length"), last], True, None)
        rec = self.g.compare({"name": "x"}, self.native, self.resp, False, short)
        self.assertFalse(rec["pass"])
        self.assertFalse(rec["stream"]["text_joins"])
        self.assertFalse(self.g.compare({"name": "x"}, self.native, self.resp, False, (200, good[1], False, None))["pass"])   # no [DONE]


if __name__ == "__main__":
    unittest.main()


class DerivedZigOverlay(unittest.TestCase):
    """[build.overlay] of a Zig-hosted recipe (SPEC.md section 5, derived modules): recipe
    dsv41-flash.strixfold-tp2-zip2 lays the two adapted files of executor.flerp.dsv41_zip2 over the origin's."""

    def setUp(self):
        self.cat = Catalog(ROOT)

    def test_overlay_replaces_two_files_and_nothing_else(self):
        base, zip2 = self.cat.recipe("dsv41-flash.strixfold-tp2"), self.cat.recipe("dsv41-flash.strixfold-tp2-zip2")
        t0, e0, _ = assemble_zig.plan(self.cat, base, Assembly(self.cat, base))
        t1, e1, _ = assemble_zig.plan(self.cat, zip2, Assembly(self.cat, zip2))
        self.assertEqual((e0, e1), ([], []))
        self.assertEqual(sorted(t0), sorted(t1))                      # the same tree paths
        changed = sorted(p for p in t0 if t0[p] != t1[p])
        self.assertEqual(changed, ["zig/src/families/dsv41/lanes.zig", "zig/src/families/dsv41/model.zig"])
        for p in changed:
            self.assertEqual(t1[p][0], "executor.flerp.dsv41_zip2")
            self.assertEqual(t0[p][0], "executor.strixfold.dsv41")
            with open(t1[p][1]) as f:
                self.assertIn("flerp (executor.flerp.dsv41_zip2)", f.read())

    def test_overlay_refuses_a_path_outside_the_modules_it_requires(self):
        zip2 = self.cat.recipe("dsv41-flash.strixfold-tp2-zip2")
        asm = Assembly(self.cat, zip2)
        zip2.data["build"]["overlay"] = {"executor.flerp.dsv41_zip2": "flerp"}     # a root that maps to no origin file
        over, errors = assemble_zig.overlaid(self.cat, zip2, asm)
        self.assertEqual(over, {})
        self.assertEqual(len(errors), 2)
        zip2.data["build"]["overlay"] = {"executor.strixfold.dsv41": "nowhere"}
        self.assertTrue(any("has no file under" in e for e in assemble_zig.overlaid(self.cat, zip2, asm)[1]))
