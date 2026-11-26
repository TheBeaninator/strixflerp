"""The StrixFold adapter hosts' way out (card 91a9cda544): executor.flerp.strixfold_abi's conformance host and
serve.flerp.strixfold_host leave through strixfold_abi.zig's `leave` (last line, stdio flushed, _exit), not by a
return from main.

Why: the engine dlopens libhipblas.so.3 and closes it at flerp_close; therock's libomp.so came with it and is
DF_1_NODELETE, and at libc's exit() its __do_fini calls through NULL (SIGSEGV at pc 0, exit status 139, after every
step of the close). HostBinary reproduces that with no device and no weights: tests/cpp/hipblas_exit_shim.c, preloaded,
opens and closes hipBLAS before main, and the host is run with no arguments (the driver's usage, status 2, long
before flerp_open). A build from before the fix dies of SIGSEGV after its last line; a fixed build exits 2.

  FLERP_SF_HOST        the host binary (default: build/dsv41-flash.strixfold-tp2-abi/flerp-executor-conformance.strixfold)
  FLERP_SF_HOST_CRASH  a host binary built before the fix, to see the fault as well as its absence
  FLERP_ROCM_ROOT      the ROCm tree whose lib/libhipblas.so.3 the engine opens (default: /data/models/tools/therock-10.2)
"""
import os
import re
import shutil
import signal
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABI = os.path.join(ROOT, "catalog/flerp/executor/strixfold_abi/src/flerp/executor/strixfold_abi.zig")
SERVE = os.path.join(ROOT, "catalog/flerp/serve/strixfold_host/src/flerp_serve_strixfold.zig")
SHIM = os.path.join(ROOT, "tests/cpp/hipblas_exit_shim.c")
ROCM = os.environ.get("FLERP_ROCM_ROOT", "/data/models/tools/therock-10.2")
HOST = os.environ.get("FLERP_SF_HOST") or os.path.join(ROOT, "build/dsv41-flash.strixfold-tp2-abi/flerp-executor-conformance.strixfold")
CRASH = os.environ.get("FLERP_SF_HOST_CRASH", "")
FIXED_LINE = b"the process leaves by _exit"   # leave's last line; a binary without it was built before the fix


def read(path):
    with open(path) as f:
        return f.read()


def body(src, head):
    """The text of the Zig function that starts with `head`, to its closing brace in column 0."""
    i = src.index(head)
    return src[i:src.index("\n}\n", i)]


class Sources(unittest.TestCase):
    def test_leave_flushes_then_leaves_by__exit(self):
        b = body(read(ABI), "pub fn leave(rc: c_int) noreturn {")
        said, flushed, left = b.index("host: main returned"), b.index("fflush(null)"), b.index("std.c._exit(")
        self.assertLess(said, flushed)
        self.assertLess(flushed, left)
        self.assertNotIn("std.process.exit", b)   # that is libc's exit(): the fini list again

    def test_both_hosts_leave_through_it(self):
        for path, call in ((ABI, "leave(flerp_host_main("), (SERVE, "abi.leave(flerp_host_main(")):
            b = body(read(path), "pub fn main(init: std.process.Init) !u8 {")
            self.assertIn(call, b, path)
            # nothing after the host's main but the way out: no return that would reach exit()
            self.assertEqual(b[b.index(call):].count("return"), 0, path)

    def test_the_stage_still_finds_the_last_line(self):
        gate = read(os.path.join(ROOT, "gate/flerp_serve_pair.py"))
        self.assertIn('"host: main returned" in text', gate)
        self.assertIn("host: main returned {d};", body(read(ABI), "pub fn leave(rc: c_int) noreturn {"))


@unittest.skipUnless(shutil.which("cc") and os.path.isfile(os.path.join(ROCM, "lib/libhipblas.so.3")) and os.path.isfile(HOST),
                     "needs cc, therock's libhipblas.so.3 and a built host (FLERP_SF_HOST)")
class HostBinary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="sf-host-exit-")
        cls.shim = os.path.join(cls.tmp, "shim.so")
        subprocess.run(["cc", "-shared", "-fPIC", "-Wall", "-Werror", SHIM, "-ldl", "-o", cls.shim], check=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_host(self, exe, shim=True):
        env = dict(os.environ, LD_LIBRARY_PATH=f"{ROCM}/lib:{ROCM}/lib/llvm/lib")
        if shim:
            env["LD_PRELOAD"] = self.shim
        r = subprocess.run(["nice", "-n", "19", exe], env=env, capture_output=True, timeout=300, cwd=self.tmp)
        return r.returncode, r.stderr.decode(errors="replace")

    def check(self, exe):
        with open(exe, "rb") as f:
            fixed = FIXED_LINE in f.read()
        rc, err = self.run_host(exe)
        self.assertIn("shim: dlclose: 0", err)                      # hipBLAS was opened and closed in this process
        last = err.strip().splitlines()[-1]
        self.assertRegex(last, r"^host: main returned 2; ")           # the driver's usage; the fault is after it
        if fixed:
            self.assertEqual(rc, 2, err[-400:])
        else:
            self.assertEqual(rc, -signal.SIGSEGV, err[-400:])       # the crash this card is about
            self.assertEqual(self.run_host(exe, shim=False)[0], 2)  # and only with hipBLAS opened and closed
        return fixed

    def test_the_built_host(self):
        self.check(HOST)

    @unittest.skipUnless(CRASH and os.path.isfile(CRASH), "FLERP_SF_HOST_CRASH names no binary")
    def test_fault_before_the_fix_and_none_after(self):
        self.assertFalse(self.check(CRASH), "FLERP_SF_HOST_CRASH is a fixed build")
        self.assertTrue(self.check(HOST), "FLERP_SF_HOST is not a fixed build")


if __name__ == "__main__":
    unittest.main()
