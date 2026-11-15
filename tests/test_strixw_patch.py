"""tool.flerp.strixw_patch on small synthetic files, no GPU: a BF16 checkpoint written here, OLD and EXPECTED written by
tests/cpp/strixw_patch_selftest.cpp with strixite's own writer and quantizers; the tool's output must equal EXPECTED
byte for byte, every refusal must refuse, and --plan / --check-against must agree."""
import json
import os
import random
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "catalog", "flerp", "tool", "strixw_patch", "src", "flerp", "strixw_patch", "build.sh")


def bf16(x):
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    u += 0x7FFF + ((u >> 16) & 1)
    return u >> 16


def write_safetensors(path, tensors):
    """tensors: name -> (dtype, shape, list of values)."""
    hdr, blobs, at = {}, [], 0
    for name, (dt, shape, vals) in tensors.items():
        b = struct.pack("<%dH" % len(vals), *[bf16(v) for v in vals]) if dt == "BF16" else struct.pack(
            "<%df" % len(vals), *vals)
        hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [at, at + len(b)]}
        blobs.append(b)
        at += len(b)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


@unittest.skipUnless(shutil.which("g++"), "no g++ on this host")
class StrixwPatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        cls.tool = os.path.join(cls.d, "strixw_patch")
        cls.mk = os.path.join(cls.d, "mk")
        subprocess.run([BUILD, cls.tool], check=True)
        subprocess.run([BUILD, cls.mk, os.path.join(ROOT, "tests", "cpp", "strixw_patch_selftest.cpp")], check=True)
        rnd = random.Random(7)
        g = lambda n, s=0.05: [rnd.gauss(0, s) for _ in range(n)]  # noqa: E731
        cls.ckpt = os.path.join(cls.d, "ckpt")
        os.mkdir(cls.ckpt)
        write_safetensors(os.path.join(cls.ckpt, "model.safetensors"), {
            "model.embed.weight": ("BF16", [96, 256], g(96 * 256)),
            "lm_head.weight": ("BF16", [96, 256], g(96 * 256)),
            "model.layers.0.experts.weight": ("BF16", [64, 256], g(64 * 256)),
            "mtp.fc_embedding.weight": ("BF16", [130, 256], g(130 * 256, 0.2)),
            "mtp.fc_hidden.weight": ("BF16", [130, 256], g(130 * 256, 0.01)),
            "mtp.norm.weight": ("F32", [256], g(256)),
        })
        cls.old = os.path.join(cls.d, "old.strixw")
        cls.exp = os.path.join(cls.d, "expected.strixw")
        subprocess.run([cls.mk, cls.ckpt, cls.old, cls.exp, "64"], check=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d)

    def run_tool(self, *args):
        return subprocess.run([self.tool, *args], capture_output=True, text=True, timeout=120)

    def test_patch_is_byte_identical_to_converter_output(self):
        out = os.path.join(self.d, "out.strixw")
        p = self.run_tool("--old", self.old, "--source", self.ckpt, "--override", "mtp_fc=q4g64", "--out", out)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("OK ", p.stdout)
        with open(out, "rb") as a, open(self.exp, "rb") as b:
            self.assertEqual(a.read(), b.read())
        # refuses to overwrite
        p = self.run_tool("--old", self.old, "--source", self.ckpt, "--override", "mtp_fc=q4g64", "--out", out)
        self.assertNotEqual(p.returncode, 0)
        os.remove(out)

    def test_plan_matches_written_file(self):
        p = self.run_tool("--old", self.old, "--source", self.ckpt, "--override", "mtp_fc=q4g64", "--plan",
                          "--scratch", self.d)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("layout exp_gu=q4g128,lm_head=q8g64,mtp_fc=q4g64,embed=bf16", p.stdout)
        self.assertIn("new_bytes %d " % os.path.getsize(self.exp), p.stdout)
        self.assertEqual(p.stdout.count("PATCH "), 2)
        self.assertEqual([f for f in os.listdir(self.d) if f.startswith("strixw_patch_plan")], [])

    def test_check_against(self):
        p = self.run_tool("--source", self.ckpt, "--override", "mtp_fc=q4g64", "--check-against", self.exp)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("RESULT: byte-identical", p.stdout)
        p = self.run_tool("--source", self.ckpt, "--override", "mtp_fc=q4g32", "--check-against", self.exp)
        self.assertNotEqual(p.returncode, 0)

    def test_refusals(self):
        out = os.path.join(self.d, "never.strixw")
        for ovr in ["lm_head=q4g64", "mtp_fc=q8g64", "mtp_fc=q4g64,lm_head=q4g64", "default=q4g64", "mtp_fc=q5g64"]:
            p = self.run_tool("--old", self.old, "--source", self.ckpt, "--override", ovr, "--out", out)
            self.assertNotEqual(p.returncode, 0, ovr)
            self.assertIn("refused", p.stderr, ovr)
        # already q4g64
        p = self.run_tool("--old", self.exp, "--source", self.ckpt, "--override", "mtp_fc=q4g64", "--out", out)
        self.assertNotEqual(p.returncode, 0)
        # wrong checkpoint (fingerprint differs)
        other = os.path.join(self.d, "other")
        os.mkdir(other)
        write_safetensors(os.path.join(other, "model.safetensors"),
                          {"mtp.fc_embedding.weight": ("BF16", [130, 256], [0.1] * (130 * 256))})
        p = self.run_tool("--old", self.old, "--source", other, "--override", "mtp_fc=q4g64", "--out", out)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("fingerprint", p.stderr)
        self.assertFalse(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
