"""format.flerp.q4panel / q8panel / strixw_v5 and tool.flerp.convert_strixw_panel on synthetic data, no GPU.

The C++ selftest covers every fleet class shape at Q4 and Q8 (offsets, alignment, exact round trip, refusals, a v5
file). Here: a small v4 OLD written by strixite's own writer (tests/cpp/strixw_panel_mkold.cpp) in the fleet's
names and encodings, converted by the tool; the output is then checked from Python, independently of the C++:
header/index via gate/strixw_header.py, copied components byte-identical to OLD's, each converted tensor's panel
bytes equal to the layout of docs/STRIXW-LAYOUT-NEXT.md candidate A built here from OLD's planes. Refusals too."""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import strixw_header as S  # noqa: E402

TOOL_BUILD = os.path.join(ROOT, "catalog", "flerp", "tool", "convert_strixw_panel", "src", "flerp",
                          "convert_strixw_panel", "build.sh")
V4_BUILD = os.path.join(ROOT, "catalog", "flerp", "tool", "strixw_patch", "src", "flerp", "strixw_patch", "build.sh")
DEFAULT = ["exp_gu", "exp_down", "sh_gu", "sh_down", "gdn_in", "gdn_out", "attn_qkv", "attn_o", "ple_kv", "lm_head",
           "mtp_fc"]


def read(path, off, n):
    with open(path, "rb") as f:
        f.seek(off)
        return f.read(n)


def panels_from(q, s, m, n, k, g, bits):
    """Candidate A from the row-major planes, written from the design text."""
    rcb = k // 2 if bits == 4 else k
    groups = k // g
    out = bytearray()
    for p in range(n // 16):
        for c in range(rcb // 16):
            for r in range(16):
                row = p * 16 + r
                out += q[row * rcb + c * 16: row * rcb + c * 16 + 16]
        for gr in range(groups):
            for r in range(16):
                i = ((p * 16 + r) * groups + gr) * 2
                out += s[i:i + 2] + m[i:i + 2]
    return bytes(out)


@unittest.skipUnless(shutil.which("g++"), "no g++ on this host")
class ConvertStrixwPanelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        cls.tool = os.path.join(cls.d, "convert_strixw_panel")
        cls.selftest = os.path.join(cls.d, "selftest")
        mk = os.path.join(cls.d, "mkold")
        subprocess.run([TOOL_BUILD, cls.tool], check=True)
        subprocess.run([TOOL_BUILD, cls.selftest, os.path.join(ROOT, "tests", "cpp", "strixw_panel_selftest.cpp")],
                       check=True)
        subprocess.run([V4_BUILD, mk, os.path.join(ROOT, "tests", "cpp", "strixw_panel_mkold.cpp")], check=True)
        cls.old = os.path.join(cls.d, "old.strixw")
        cls.bad = os.path.join(cls.d, "bad.strixw")
        subprocess.run([mk, cls.old, cls.bad], check=True)
        cls.new = os.path.join(cls.d, "new.strixw")
        p = subprocess.run([cls.tool, "--old", cls.old, "--out", cls.new], capture_output=True, text=True, timeout=120)
        cls.convert_out = p

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d)

    def run_tool(self, *args):
        return subprocess.run([self.tool, *args], capture_output=True, text=True, timeout=120)

    def test_selftest_every_fleet_shape(self):
        p = subprocess.run([self.selftest, self.d], capture_output=True, text=True, timeout=300)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("ALL PASSED (0 failures)", p.stdout)
        self.assertEqual(p.stdout.count("offsets as designed, round trip exact"), 26)

    def test_convert_output(self):
        p = self.convert_out
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("OK ", p.stdout)
        old, new = S.read_strixw(self.old), S.read_strixw(self.new)
        self.assertEqual((old["version"], new["version"]), (4, 5))
        self.assertEqual(new["file_bytes"], os.path.getsize(self.new))
        mo, mn = dict(old["meta"]), dict(new["meta"])
        lo, ln = mo.pop("layout"), mn.pop("layout")
        self.assertEqual(mo, mn)  # source_fingerprint and everything else kept
        want = ",".join(x + "p16" if x.split("=")[0] in DEFAULT else x for x in lo.split(","))
        self.assertEqual(ln, want)
        self.assertIn("hc_up=q8g64,", ln)
        self.assertIn("hc_down=q8g64,", ln)
        converted = 0
        for a, b in zip(old["tensors"], new["tensors"]):
            self.assertEqual((a["name"], a["shape"], a["group_size"], a["experts"]),
                             (b["name"], b["shape"], b["group_size"], b["experts"]))
            if b["encoding"].endswith("panel16"):
                converted += 1
                bits = 4 if a["encoding"] == "q4" else 8
                self.assertEqual(b["encoding"], "q%d-panel16" % bits)
                planes = {c["role"]: read(self.old, c["offset"], c["bytes"]) for c in a["components"]}
                n, k = a["shape"]
                expect = panels_from(planes["q"], planes["scale"], planes["min"], n, k, a["group_size"], bits)
                (c,) = b["components"]
                got = read(self.new, c["offset"], c["bytes"])
                self.assertEqual(got, expect, a["name"])
                self.assertEqual(S.strix_hash64(got), c["hash"])
                self.assertEqual(c["offset"] % 4096, 0)
            else:
                self.assertEqual(a["encoding"], b["encoding"])
                for ca, cb in zip(a["components"], b["components"]):
                    self.assertEqual((ca["bytes"], ca["hash"]), (cb["bytes"], cb["hash"]))
                    self.assertEqual(read(self.old, ca["offset"], ca["bytes"]), read(self.new, cb["offset"], cb["bytes"]))
        self.assertEqual(converted, 14)

    def test_plan_matches_written_file(self):
        p = self.run_tool("--old", self.old, "--plan", "--scratch", self.d)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("new_bytes %d " % os.path.getsize(self.new), p.stdout)
        self.assertEqual(p.stdout.count("PANEL "), 14)
        self.assertNotIn("NOT-128-aligned", p.stdout)
        self.assertEqual([f for f in os.listdir(self.d) if f.startswith("convert_strixw_panel_plan")], [])
        new = S.read_strixw(self.new)
        for t in new["tensors"]:
            c = t["components"][0]
            self.assertIn(" %s@%d+%d" % (c["role"], c["offset"], c["bytes"]), p.stdout)

    def test_classes_subset_and_hc_down(self):
        out = os.path.join(self.d, "sub.strixw")
        p = self.run_tool("--old", self.old, "--out", out, "--classes", "exp_gu,hc_down")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        t = S.read_strixw(out)
        self.assertIn("exp_gu=q4g128p16,exp_down=q4g32,", t["meta"]["layout"])
        self.assertIn("hc_down=q8g64p16,hc_up=q8g64,", t["meta"]["layout"])
        self.assertEqual(sum(x["encoding"].endswith("panel16") for x in t["tensors"]), 4)
        os.remove(out)

    def test_check_tensors(self):
        p = self.run_tool("--old", self.old, "--check-tensors",
                          "model.language_model.layers.0.mlp.experts_gate_up,lm_head.weight")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("RESULT: every tensor round-trips byte-identical", p.stdout)
        p = self.run_tool("--old", self.old, "--check-tensors", "model.language_model.embed_tokens.weight")
        self.assertNotEqual(p.returncode, 0)

    def test_refusals(self):
        out = os.path.join(self.d, "never.strixw")
        for cls in ["hc_up", "embed", "router", "exp_gu,exp_gu", ""]:
            p = self.run_tool("--old", self.old, "--out", out, "--classes", cls)
            self.assertNotEqual(p.returncode, 0, cls)
            self.assertFalse(os.path.exists(out), cls)
        # a v5 input (already panels), N % 16 != 0, an existing --out
        for old in [self.new, self.bad]:
            p = self.run_tool("--old", old, "--out", out)
            self.assertNotEqual(p.returncode, 0, old)
            self.assertFalse(os.path.exists(out))
        p = self.run_tool("--old", self.old, "--out", self.new)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("exists", p.stderr)

    def test_corrupt_old_is_refused_and_output_removed(self):
        bad = os.path.join(self.d, "corrupt.strixw")
        shutil.copy(self.old, bad)
        t = S.read_strixw(bad)["tensors"][1]  # a copied tensor (hc mix_down is not in the default classes)
        off = t["components"][0]["offset"]
        with open(bad, "r+b") as f:
            f.seek(off)
            b = f.read(1)
            f.seek(off)
            f.write(bytes([b[0] ^ 1]))
        out = os.path.join(self.d, "fromcorrupt.strixw")
        p = self.run_tool("--old", bad, "--out", out)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("corrupt", p.stderr)
        self.assertFalse(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
