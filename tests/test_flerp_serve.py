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

    def test_a_hold_shorter_than_asked_survives_a_redo(self):
        """Held form, an executor with a depth rule of its own (hold_cap=3 under --mtp-draft 7, as qwen35moe_abi
        draft_value=on): the loop takes each round's k from the verify's rows, and a dropped round drafted and verified
        again compares equal. Before the fix the redo asked the full depth, got a shorter k2 and failed with
        "the draft chain changed after a drop"."""
        rc, j = self.run_case("held", "candidates", ("reply=200", "hold_cap=3"))
        c = j["cases"][0]
        self.assertEqual((rc, j["failures"]), (0, 0), j)
        self.assertGreater(c["loop_drop_dropped_verifies"], 0)
        self.assertEqual(c["loop_drop"], c["loop"])
        self.assertEqual(c["loop"], c["plain"][:len(c["loop"])])
        self.assertEqual(max(s[0] for s in c["loop_steps"] if s[0] != 255), 3)    # never more than the executor held

    def test_hooks_carry_their_size(self):
        """flerp_loop_hooks starts with size like every other struct; the loop refuses a size short of pick or a null
        pick (FLERP_E_ARGUMENT, result.error set) and reads hooks only as far as size."""
        with open(os.path.join(ROOT, "include", "flerp", "decode_loop.h")) as f:
            self.assertRegex(f.read(), r"typedef struct flerp_loop_hooks \{\s*uint32_t size;")
        for path in ("executor/conformance/src/flerp/executor/conformance.c", "serve/http/src/flerp/serve/main.cpp"):
            with open(os.path.join(FLERP, path)) as f:
                self.assertIn("H.size = sizeof(H)", f.read(), path)
        prog = os.path.join(self.dir, "hooks.c")
        with open(prog, "w") as f:
            f.write(r'''
#include "flerp/decode_loop.h"
#include <stdio.h>
#include <string.h>
static flerp_token pick(void *ctx, const flerp_rows *rows, int64_t row) { (void)ctx; return rows->tokens[row]; }
static int cancelled(void *ctx) { (void)ctx; return 1; }   /* past a short size: must not be called */
int main(void) {
    flerp_kv kv[] = {{"vocab", "100"}, {"reply", "3"}};
    flerp_open_args oa = {sizeof(oa), "t", kv, 2};
    flerp_executor *ex;
    flerp_caps caps; flerp_token tok[16]; uint8_t nan[16]; flerp_rows rows; flerp_seq_args sa; flerp_want w; int32_t seq;
    flerp_loop L; flerp_loop_hooks H; flerp_loop_result R;
    const flerp_token prompt[2] = {1, 2}, end = 99;
    if (flerp_open(&oa, &ex) != FLERP_OK) return 1;
    memset(&caps, 0, sizeof(caps)); caps.size = sizeof(caps);
    if (flerp_get_caps(ex, &caps) != FLERP_OK) return 2;
    memset(&sa, 0, sizeof(sa)); sa.size = sizeof(sa); sa.max_new = 8;
    if (flerp_seq_new(ex, &sa, &seq) != FLERP_OK) return 3;
    memset(&rows, 0, sizeof(rows)); rows.size = sizeof(rows); rows.tokens = tok; rows.tokens_cap = 16; rows.nan = nan; rows.nan_cap = 16;
    memset(&w, 0, sizeof(w)); w.size = sizeof(w); w.delivery = FLERP_OUT_TOKENS; w.rows = 1; w.n_valid = 100;
    if (flerp_forward(ex, seq, prompt, 2, &w, &rows, NULL, 0) != FLERP_OK) return 4;
    memset(&L, 0, sizeof(L)); L.size = sizeof(L); L.ex = ex; L.seq = seq; L.caps = &caps; L.delivery = FLERP_OUT_TOKENS; L.n_valid = 100;
    L.rows = &rows; L.max_tokens = 8; L.end_ids = &end; L.n_end = 1;
    memset(&H, 0, sizeof(H)); H.pick = pick; H.cancelled = cancelled;
    memset(&R, 0, sizeof(R)); R.size = sizeof(R);
    H.size = (uint32_t)offsetof(flerp_loop_hooks, pick);                    /* short of pick */
    if (flerp_decode_loop(&L, &H, &R) != FLERP_E_ARGUMENT || R.error == NULL) return 5;
    H.size = sizeof(H); H.pick = NULL;
    if (flerp_decode_loop(&L, &H, &R) != FLERP_E_ARGUMENT || R.error == NULL) return 6;
    H.pick = pick; H.size = (uint32_t)FLERP_HAS(flerp_loop_hooks, pick);      /* cancelled lies past size: absent */
    if (flerp_decode_loop(&L, &H, &R) != FLERP_OK || R.finish == NULL || strcmp(R.finish, "stop") != 0 || R.completion < 1) return 7;
    H.size = sizeof(H);                                                      /* the whole struct: cancelled runs first */
    if (flerp_seq_reset(ex, seq) != FLERP_OK || flerp_forward(ex, seq, prompt, 2, &w, &rows, NULL, 0) != FLERP_OK) return 8;
    if (flerp_decode_loop(&L, &H, &R) != FLERP_OK || strcmp(R.finish, "cancelled") != 0) return 9;
    flerp_close(ex);
    return 0;
}
''')
        exe = os.path.join(self.dir, "hooks")
        subprocess.run(["cc", "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-Wall", "-Werror", "-I", os.path.join(ROOT, "include"), prog,
                        os.path.join(FLERP, "runtime/decode_loop/src/flerp/decode_loop.c"),
                        os.path.join(FLERP, "executor/fake_abi/src/flerp/executor/fake_abi.c"), "-lm", "-o", exe], check=True)
        self.assertEqual(subprocess.run([exe]).returncode, 0)

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


def _reply_harness():
    """The scripted stand-in front end's build and tokenizer (tests/test_flerp_reply.py), or None where they are not here."""
    from tests import test_flerp_reply as r
    if not (shutil.which("g++") and shutil.which("gcc") and os.path.isfile(r.TOKENIZER)):
        return None
    return r


@unittest.skipUnless(_reply_harness(), "needs g++ and the model's tokenizer.json (FLERP_TEST_TOKENIZER)")
class FrontEndUnderARouter(unittest.TestCase):
    """serve.flerp.http under a router that times out and retries (FLERP-FRONT-ROUTER, card c2567775e3): the scripted
    stand-in with delay_ms=100 holds the generation lane for 0.1 s a token, so the queue, the refusal, /health under
    load and a client that leaves can be seen without a model."""

    @classmethod
    def setUpClass(cls):
        cls.r = _reply_harness()
        cls.dir = tempfile.mkdtemp(prefix="flerp-front-")
        cls.exe = cls.r.build(cls.dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def start(self, *args, delay_ms=100):
        import time
        import urllib.request
        port = self.r.free_port()
        log = open(os.path.join(self.dir, f"server-{port}.log"), "w")
        proc = subprocess.Popen([self.exe, "--port", str(port), "--model-id", "fake", f"text.tokenizer={self.r.TOKENIZER}", "executor.vocab=248077",
                                 "executor.lo=1000", "executor.hi=30000", "executor.reply=4000", f"executor.end={self.r.END}",
                                 f"executor.delay_ms={delay_ms}", *args], stdout=log, stderr=subprocess.STDOUT)
        self.addCleanup(lambda: (proc.kill() if proc.poll() is None else None, log.close()))
        for _ in range(300):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).read()
                return proc, port
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("the server did not start: " + open(log.name).read())

    @staticmethod
    def health(port):
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
            return json.loads(r.read())

    @staticmethod
    def post(port, max_tokens):
        import urllib.error
        import urllib.request
        body = json.dumps({"prompt": "hello", "max_tokens": max_tokens, "temperature": 0}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status, dict(r.headers), json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), json.loads(e.read())

    @staticmethod
    def leave(port, max_tokens, after_s):
        """A client that sends a request and goes away: a raw socket closed after `after_s` without reading."""
        import socket
        import time
        body = json.dumps({"prompt": "hello", "max_tokens": max_tokens, "temperature": 0}).encode()
        s = socket.create_connection(("127.0.0.1", port))
        s.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
        time.sleep(after_s)
        s.close()

    @staticmethod
    def until(pred, seconds):
        import time
        t0 = time.time()
        while not pred():
            if time.time() - t0 > seconds:
                return False
            time.sleep(0.05)
        return True

    def test_a_client_that_leaves_cancels_its_request(self):
        """A non-streaming request whose client has gone stops between decode steps (is_connection_closed polled by the
        waiting HTTP thread), not at max_tokens: 400 tokens at 0.1 s each would run 40 s; it is over within 3 s."""
        proc, port = self.start()
        self.leave(port, 400, 0.5)
        self.assertTrue(self.until(lambda: not self.health(port)["busy"], 3))
        h = self.health(port)
        self.assertEqual((h["requests_cancelled"], h["requests_served"], h["queued"]), (1, 1, 0))

    def test_the_queue_is_bounded_and_health_never_waits(self):
        """--queue 1: A runs, B waits, C is refused with 503 + Retry-After (the router's own wait) at once; /health
        answers under load with queued 1; B's client leaves while it waits and B is dropped before it starts."""
        import threading
        import time
        proc, port = self.start("--queue", "1", "--retry-after", "7")
        out = {}
        a = threading.Thread(target=lambda: out.__setitem__("A", self.post(port, 30)))
        a.start()
        self.assertTrue(self.until(lambda: self.health(port)["busy"], 3))
        b = threading.Thread(target=lambda: self.leave(port, 30, 1.5))
        b.start()
        self.assertTrue(self.until(lambda: self.health(port)["queued"] == 1, 3))
        t0 = time.time()
        h = self.health(port)
        self.assertLess(time.time() - t0, 1.0)
        self.assertEqual((h["busy"], h["queued"], h["queue_max"]), (True, 1, 1))
        t0 = time.time()
        status, headers, body = self.post(port, 30)
        self.assertLess(time.time() - t0, 1.0)
        self.assertEqual((status, headers.get("Retry-After"), body["error"]["code"]), (503, "7", "queue_full"))
        b.join()
        a.join(30)
        self.assertEqual(out["A"][0], 200)
        self.assertTrue(self.until(lambda: self.health(port)["requests_dropped"] == 1, 3))
        h = self.health(port)
        self.assertEqual((h["requests_served"], h["requests_refused"], h["requests_cancelled"], h["busy"]), (1, 1, 0, False))

    def test_health_says_whether_the_think_end_guard_runs(self):
        """The guard bans end ids out of a logits row; with --delivery tokens the executor draws and the host sees no
        row, so /health says so instead of the guard being skipped in silence."""
        _, port = self.start(delay_ms=0)
        self.assertEqual(self.health(port)["think_end_guard"], "on")
        _, port = self.start("--delivery", "tokens", delay_ms=0)
        self.assertTrue(self.health(port)["think_end_guard"].startswith("unavailable: the executor draws its tokens"))
        _, port = self.start("--think-end-guard", "off", delay_ms=0)
        self.assertEqual(self.health(port)["think_end_guard"], "off")

    def test_sigterm_stops_the_server_from_the_main_loop(self):
        """The handler only sets a flag; the 200 ms main loop stops the server and the process exits 0, also mid-request."""
        import signal
        import threading
        proc, port = self.start()
        threading.Thread(target=lambda: self.post(port, 100), daemon=True).start()
        self.assertTrue(self.until(lambda: self.health(port)["busy"], 3))
        proc.send_signal(signal.SIGTERM)
        self.assertEqual(proc.wait(timeout=10), 0)


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
