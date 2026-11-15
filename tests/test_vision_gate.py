"""What gate/vision_dsv41_cpu.py and runtime.flerp.vision_sidecar decide without torch or a host: the span and pipe
framing, the verdict from numbers, the worker's environment setting and the admission's records."""
import importlib.util
import json
import os
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import vision_dsv41_cpu as G  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    keep, sys.dont_write_bytecode = sys.dont_write_bytecode, True      # no __pycache__ in a module's src/
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = keep
    return mod


WORKER = load(G.WORKER, "flerp_vision_cpu_t")
ADMISSION = load(G.ADMISSION, "flerp_verify_vision_t")


class Framing(unittest.TestCase):
    def test_span_tokens(self):
        self.assertEqual(G.span_tokens(1, 1), (1, 1, 4))
        self.assertEqual(G.span_tokens(2, 5), (1, 2, 5))
        self.assertEqual(G.span_tokens(39, 39), (13, 13, 184))
        self.assertEqual(G.span_tokens(42, 216), (14, 72, 1024))

    def test_header_is_what_the_worker_accepts(self):
        for nh, nw in ((1, 1), (2, 5), (42, 216)):
            self.assertEqual(WORKER.decode_grid(G.grid_header(nh, nw)), (nh, nw, nh * nw, G.span_tokens(nh, nw)[2]))
        with self.assertRaises(ValueError):
            WORKER.decode_grid(G.grid_header(1, 1, lh=2))
        with self.assertRaises(ValueError):
            WORKER.decode_grid(G.grid_header(96, 97))          # 9,312 patches

    def test_row_kinds(self):
        start, rows, newlines, end = G.row_kinds(2, 3)
        self.assertEqual((start, rows, newlines, end), (0, [1, 2, 3, 5, 6, 7], [4, 8], 9))
        self.assertEqual(end + 1, G.span_tokens(6, 9)[2])

    def test_parse_reply(self):
        body = bytes(4 * G.ROW_BYTES)
        good = struct.pack("<4sIQ", b"VCPU", 4, len(body)) + body
        self.assertEqual(G.parse_reply(good, 4), body)
        for bad in (good + b"\0", good[:-1], b"VCPX" + good[4:], good[:10], struct.pack("<4sIQ", b"VCPU", 5, len(body)) + body):
            with self.assertRaises(ValueError):
                G.parse_reply(bad, 4)

    def test_git_blob(self):
        self.assertEqual(G.git_blob_sha1(b""), "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391")


class Verdict(unittest.TestCase):
    def setUp(self):
        self.cfg = G.load_config(os.path.join(ROOT, "gate", "vision-dsv41-cpu312.toml"))
        self.images = [{"name": "a", "expected_tokens": 184}]
        m = lambda rel, cos: {"rel_rms": rel, "row_cosine_min": cos}       # noqa: E731
        self.rows = {"a": {"tokens": 184, "lh": 13, "lw": 13,
                           "compare": {"S32_vs_R32": m(2e-6, 0.99999999), "R16_vs_R32": m(0.02, 0.999)}}}
        self.protocol = {"images": {"a": {"reply_ok": True, "finite": True, "delimiters_equal": True, "twice_equal": True,
                                          "equals_S16": True, "W_vs_R32": m(0.021, 0.999),
                                          "W_vs_R16_bits": {"values_differ": 0, "rows_differ": 0, "rows": 169}}},
                         "refusals": [{"case": "x", "refused": True, "rc": 1, "stdout_bytes": 0}], "admission": {"ok": True}}

    def judge(self):
        return G.judge(self.cfg, self.images, self.rows, self.protocol)

    def test_pass(self):
        ok, fails, held = self.judge()
        self.assertTrue(ok, fails)
        self.assertTrue(all(held.values()))

    def test_each_criterion_can_fail(self):
        def broken(change, key):
            self.setUp()
            change()
            ok, fails, held = self.judge()
            self.assertFalse(ok)
            self.assertEqual([k for k, v in held.items() if not v], [key], fails)
        broken(lambda: self.rows["a"].update(tokens=185), "1 shape")
        broken(lambda: self.protocol["images"]["a"].update(delimiters_equal=False), "1 shape")
        broken(lambda: self.rows["a"]["compare"]["S32_vs_R32"].update(rel_rms=2e-4), "2 arithmetic")
        broken(lambda: self.rows["a"]["compare"]["S32_vs_R32"].update(row_cosine_min=0.999999), "2 arithmetic")
        broken(lambda: self.protocol["images"]["a"]["W_vs_R16_bits"].update(values_differ=1, rows_differ=1), "3a bf16 rows = R16")
        broken(lambda: self.protocol["images"]["a"]["W_vs_R32"].update(rel_rms=0.045), "3b bf16 vs R16")
        broken(lambda: self.protocol["images"]["a"].update(twice_equal=False), "4 determinism")
        broken(lambda: self.protocol["images"]["a"].update(equals_S16=False), "4 determinism")
        broken(lambda: self.protocol["refusals"][0].update(refused=False), "5 protocol")
        broken(lambda: self.protocol.update(admission={"ok": False, "why": "x"}), "5 protocol")

    def test_distance_from_fp32_alone_is_a_record_since_the_restatement(self):
        self.protocol["images"]["a"]["W_vs_R32"].update(rel_rms=0.02, row_cosine_min=0.86)   # R16's own outlier row
        self.assertTrue(self.judge()[0])
        self.assertEqual(self.cfg["restated"]["bf16_row_cosine_min"], 0.9855)

    def test_mean_ratio(self):
        self.protocol["images"]["a"]["W_vs_R32"]["rel_rms"] = 0.03       # 1.5 times R16: under 2, over the mean's 1.25
        ok, fails, held = self.judge()
        self.assertFalse(held["3b bf16 vs R16"])

    def test_missing_image_fails_everything(self):
        self.rows["a"] = {"image": "a", "error": ["boom"]}
        ok, fails, held = self.judge()
        self.assertFalse(ok)
        self.assertFalse(any(held.values()))

    def test_shrunk_image_needs_no_count_but_a_smaller_grid(self):
        self.images[0]["expected_tokens"] = 0
        self.rows["a"]["unshrunk_tokens"] = 1192
        self.assertTrue(self.judge()[0])
        self.rows["a"]["unshrunk_tokens"] = 184
        self.assertFalse(self.judge()[0])


class Setting(unittest.TestCase):
    def setting(self, value):
        old = os.environ.pop("FLERP_VISION_ENV", None)
        try:
            if value is not None:
                os.environ["FLERP_VISION_ENV"] = value
            return WORKER.environment_setting()
        finally:
            os.environ.pop("FLERP_VISION_ENV", None)
            if old is not None:
                os.environ["FLERP_VISION_ENV"] = old

    def test_unset_is_the_origins_pin(self):
        self.assertEqual(self.setting(None), {"torch": "2.9.0+cpu"})

    def test_parse(self):
        self.assertEqual(self.setting("python=3.12,torch=2.11.0+cpu,numpy=2.3.5"),
                         {"python": "3.12", "torch": "2.11.0+cpu", "numpy": "2.3.5"})
        for bad in ("", "torch", "python=3.12", "torch=2.11.0+cpu,torch=2.9.0+cpu", "torch=2.11.0+cpu,pillow=12.3.0", "torch="):
            with self.assertRaises(ValueError, msg=bad):
                self.setting(bad)

    def test_check_compares_every_named_key(self):
        class Torch:
            __version__ = "2.11.0+cpu"
        try:
            import numpy
        except ImportError:
            self.skipTest("numpy is not installed here")
        WORKER.torch = Torch
        here = "%d.%d" % sys.version_info[:2]
        WORKER.check_environment({"torch": "2.11.0+cpu", "python": here, "numpy": numpy.__version__})
        for wrong in ({"torch": "2.9.0+cpu"}, {"torch": "2.11.0+cpu", "python": "3.99"}, {"torch": "2.11.0+cpu", "numpy": "0"}):
            with self.assertRaises(RuntimeError):
                WORKER.check_environment(wrong)

    def test_the_tower_is_the_origins(self):
        """Outside the marked changes the adapted worker is the origin's file line for line."""
        with open(os.path.join(ROOT, G.ORIGIN_WORKER)) as f:
            origin = f.read().splitlines()
        with open(os.path.join(ROOT, G.WORKER)) as f:
            mine = f.read().splitlines()
        cut = lambda lines, a, b: lines[lines.index(a):lines.index(b)]     # noqa: E731
        self.assertEqual(cut(origin, "import hashlib", "def main():"),
                         [x for x in cut(mine, "import hashlib", "# flerp (runtime.flerp.vision_sidecar): the origin refuses any torch but 2.9.0+cpu (its line 158). Here the pin is")])
        self.assertEqual(cut(origin, "def main():", "if __name__ == '__main__':"), cut(mine, "def main():", "if __name__ == '__main__':"))


class Records(unittest.TestCase):
    def test_shipped_file(self):
        self.assertEqual(ADMISSION.environment("origin")["torch"], "2.9.0+cpu")
        self.assertEqual(ADMISSION.setting(ADMISSION.environment("origin")), "torch=2.9.0+cpu,numpy=2.3.5")
        rec = ADMISSION.environment("py312-torch2110")          # admitted without a flag: status pass, with evidence
        self.assertEqual(rec["status"], "pass")
        self.assertEqual(ADMISSION.setting(rec), "python=3.12,torch=2.11.0+cpu,numpy=2.3.5")
        self.assertTrue(ADMISSION.setting(ADMISSION.environment("origin")).startswith(WORKER.ORIGIN_PIN))

    def test_status_rules(self):
        base = {"python": "3.12", "torch": "2.11.0+cpu", "numpy": "2.3.5", "pillow": "12.3.0"}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "env.json")

            def admitted(record, **kw):
                with open(path, "w") as f:
                    json.dump({"environments": {"x": record}}, f)
                try:
                    ADMISSION.environment("x", path, **kw)
                    return True
                except ValueError:
                    return False
            self.assertFalse(admitted(dict(base, status="candidate")))
            self.assertTrue(admitted(dict(base, status="candidate"), allow_candidate=True))
            self.assertFalse(admitted(dict(base, status="pass")))                       # pass needs evidence
            self.assertTrue(admitted(dict(base, status="pass", evidence="gate/results/x/summary.md")))
            self.assertFalse(admitted(dict(base, status="fail", evidence="x"), allow_candidate=True))
            self.assertFalse(admitted(dict(base, status="pass", evidence="x", torch="2.11.0+cpu; import os")))
            self.assertFalse(admitted(dict(base, status="pass", evidence="x", python="3")))
            with self.assertRaises(ValueError):
                ADMISSION.environment("y", path)


if __name__ == "__main__":
    unittest.main()
