"""gate/placed_layout.py on a tiny synthetic GGUF: per-rank loom-swarm read logs of the gguf_placed stage plan."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import gguf_census as C  # noqa: E402
import placed_layout as L  # noqa: E402
from test_gguf_census import synthetic  # noqa: E402


def read_log(path):
    with open(path) as f:
        return [tuple(int(x) for x in line.split()) for line in f if line.strip()]


class PlacedLayout(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "t.gguf")
        synthetic(self.path, blocks=4)
        self.out = os.path.join(self.dir.name, "logs")

    def tearDown(self):
        self.dir.cleanup()

    def test_logs_match_plan(self):
        doc = L.write_logs(self.path, 2, self.out)
        h = C.read_header(self.path)
        plan = C.stage_plan(h, 2)
        self.assertEqual(doc["plan_sha256"], C.plan_hash(plan))
        size = os.path.getsize(self.path)
        held = []
        for rk, pr in zip(doc["ranks"], plan["ranks"]):
            rs = read_log(os.path.join(self.out, f"t.gguf.r{rk['rank']}of2.ranges"))
            self.assertEqual(rs[0], (0, h["data_start"]))                      # the header first
            self.assertEqual(rk["layout_bytes"], pr["tensor_bytes"] + h["data_start"])
            self.assertEqual(sum(n for _, n in rs), rk["layout_bytes"])
            self.assertEqual([o for o, _ in rs], sorted(o for o, _ in rs))     # file order
            for (o, n), (o2, _) in zip(rs, rs[1:]):
                self.assertLessEqual(o + n, o2)                                # no overlap
            self.assertTrue(all(o + n <= size for o, n in rs))
            self.assertEqual({(e[3], e[2]) for e in pr["entries"]}, set(rs[1:]))
            held.append(set(rs[1:]))
        self.assertFalse(held[0] & held[1])                                    # ranks disjoint past the header
        unloaded = sum(u["bytes"] for u in plan["unloaded"])
        tensors = sum(t["bytes"] for t in h["tensors"])
        self.assertEqual(sum(n for s in held for _, n in s), tensors - unloaded)

    def test_cli_and_loom_swarm(self):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "gate", "placed_layout.py"), self.path, "--world", "2",
                            "--out", self.out], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("plan sha256", r.stdout)
        bin_ = os.environ.get("LOOM_SWARM_BIN") or shutil.which("loom-swarm")
        if not bin_:
            self.skipTest("no loom-swarm binary (set LOOM_SWARM_BIN)")
        log = os.path.join(self.out, "t.gguf.r1of2.ranges")
        m = os.path.join(self.out, "r1.swarm.json")
        r = subprocess.run([bin_, "manifest", self.path, "--piece-mib", "1", "--layout", log, "-o", m],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        import json
        with open(m) as f:
            man = json.load(f)
        self.assertEqual(man["file_size"], sum(n for _, n in read_log(log)))
        self.assertEqual(man["disk_size"], os.path.getsize(self.path))


if __name__ == "__main__":
    unittest.main()
