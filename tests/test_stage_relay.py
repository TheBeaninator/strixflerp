"""runtime.flerp.stage_relay without a GPU: builds tests/cpp/stage_relay_selftest.cpp with the host's C++ compiler
and runs it (loopback ring, frame checks, injected faults, a TCP ring of forked processes on 127.0.0.1)."""
import os
import random
import shutil
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "catalog", "flerp", "runtime", "stage_relay", "src")


@unittest.skipUnless(shutil.which("g++"), "no g++ on this host")
class StageRelaySelfTest(unittest.TestCase):
    def test_selftest(self):
        with tempfile.TemporaryDirectory() as d:
            exe = os.path.join(d, "selftest")
            subprocess.run(["g++", "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror", "-I", SRC,
                            os.path.join(ROOT, "tests", "cpp", "stage_relay_selftest.cpp"),
                            os.path.join(SRC, "flerp", "stage_relay", "relay.cpp"), "-o", exe], check=True)
            p = subprocess.run([exe, str(random.randint(30000, 50000))], capture_output=True, text=True, timeout=120)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("0 failure(s)", p.stdout)


if __name__ == "__main__":
    unittest.main()
