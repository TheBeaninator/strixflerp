"""strix-loop results are fit to publish: no pod paths, bench host names, disk dumps or card ids in the committed
results, and gate/strix_loop.py records paths relative to the tree or as bench job ids."""
import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import strix_loop as S  # noqa: E402

PATHS = ["gate/results/strix-loop", "gate/results/sfx-pkg", "gate/criteria/sfx-pkg.toml", "gate/strix_loop.py",
         "tools/strix-loop.sh", "docs/STRIX-LOOP.md"]
BAD = re.compile(r"/data/pod|/data/projects/[0-9a-f]{10}|ltsp-k|lsblk|MOUNTPOINT|Use% Mounted")
CARD_ID = re.compile(r"\b(?=[0-9a-f]{10}\b)[0-9]*[a-f][0-9a-f]*\b")       # 10 hex with a letter (not a timestamp)


def tracked(paths):
    out = subprocess.run(["git", "ls-files", "--"] + paths, cwd=ROOT, capture_output=True, text=True)
    return [f for f in out.stdout.split() if os.path.isfile(os.path.join(ROOT, f))]


class PublishablePaths(unittest.TestCase):
    def test_pub(self):
        self.assertEqual(S.pub(os.path.join(S.ROOT, "build", "r", "ninja", "bench")), os.path.join("build", "r", "ninja", "bench"))
        self.assertEqual(S.pub("/srv/bench/runs/x/20261117T000124Z-run-pmodel"), "20261117T000124Z-run-pmodel")
        self.assertEqual(S.pub("gate/results/x"), "gate/results/x")
        self.assertIsNone(S.pub(None))


class ScrubRun(unittest.TestCase):
    """A fresh run's output, written with real paths and the card id, is publishable after scrub_run."""
    def test_fresh_run_output_is_clean(self):
        card = "0123456789"
        with tempfile.TemporaryDirectory() as d:
            res = os.path.join(d, "gate", "results", "strix-loop", "r1")
            os.makedirs(os.path.join(res, "model"))
            job = "/data/pod/bench/runs/20261117T000124Z-run-pmodel"
            files = {
                "plan.json": '{"card": "%s"}' % card,
                "kbench-model.args.json": '{"_card": "%s", "src": "%s/build/r1/kstage"}' % (card, d),
                "kbench-model.txt": "files: %s\nran on ltsp-k\n" % job,
                "state.json": '{"job": "%s", "tool": "/data/models/tools/therock-10.1/bin/clang", "x": "%s/tools/isa-scan.py"}' % (job, d),
                "loop.log": "$ /data/pod/x ran for card %s\n" % card,
            }
            for n, t in files.items():
                with open(os.path.join(res, n), "w") as f:
                    f.write(t)
            changed = S.scrub_run(res, d, card)
            self.assertEqual(len(changed), len(files))
            for n in files:
                with open(os.path.join(res, n)) as f:
                    t = f.read()
                self.assertIsNone(BAD.search(t), n)
                self.assertIsNone(CARD_ID.search(t), n)
                self.assertNotIn(d, t)
            with open(os.path.join(res, "kbench-model.args.json")) as f:
                self.assertIn("<checkout>/build/r1/kstage", f.read())
            self.assertEqual(S.scrub_run(res, d, card), [])    # idempotent


class NoInternalTraces(unittest.TestCase):
    def test_no_pod_paths_hosts_or_dumps(self):
        files = tracked(PATHS)
        if not files:
            self.skipTest("not a git checkout")
        hits = []
        for f in files:
            with open(os.path.join(ROOT, f), errors="replace") as fh:
                for n, line in enumerate(fh, 1):
                    if BAD.search(line):
                        hits.append(f"{f}:{n}")
        self.assertEqual(hits, [])

    def test_no_card_ids(self):
        files = [f for f in tracked(PATHS) if not f.endswith((".mem", ".jsonl", ".tsv"))]
        if not files:
            self.skipTest("not a git checkout")
        hits = []
        for f in files:
            with open(os.path.join(ROOT, f), errors="replace") as fh:
                for n, line in enumerate(fh, 1):
                    for m in CARD_ID.finditer(line):
                        hits.append(f"{f}:{n}:{m.group(0)}")
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
