"""The judgements and helpers of gate/ornith_p5.py and gate/swarm_frames_to_file.py (no GPU, no worker)."""
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import ornith_p5 as P  # noqa: E402


def staged(prompt, ids=(1, 2), sha=("a", "b"), argmax=True, calls=None):
    return {"event": "staged", "prompt": prompt, "ids": list(ids), "logits_sha256": list(sha), "argmax_is_token": argmax,
            "prompt_tokens": 20, "prefill_s": 1.5, "calls": calls or {"decode_calls": 10, "decode_wall_ms": 1000.0}}


class Cleanup(unittest.TestCase):
    def test_pgrep_pattern_skips_its_own_shell(self):
        # run 11121559: `pgrep -fc '<bundle>/'` counted the ssh shell that ran it, 1 a node, and kept the holds
        with tempfile.TemporaryDirectory() as d:
            bundle = os.path.join(d, "flerp-p5-bundle")
            pat = P.proc_pattern(bundle)
            p = subprocess.run(["bash", "-c", f"pgrep -fc '{pat}'; true"], capture_output=True, text=True)
            self.assertEqual(p.stdout.strip(), "0")
            os.makedirs(bundle)
            exe = os.path.join(bundle, "sleeper")
            os.symlink("/bin/sleep", exe)
            child = subprocess.Popen([exe, "30"])
            try:
                p = subprocess.run(["bash", "-c", f"pgrep -fc '{pat}'; true"], capture_output=True, text=True)
                self.assertEqual(p.stdout.strip(), "1")
            finally:
                child.kill()
                child.wait()

    def test_left_and_release(self):
        self.assertEqual(P.parse_left(0, "units 0\nprocs 2\n"), {"units": 0, "procs": 2})
        self.assertIsNone(P.parse_left(255, "ssh: connect timed out"))
        self.assertIsNone(P.parse_left(0, "units 0\n"))
        self.assertTrue(P.units_gone({"F": {"units": 0, "procs": 1}, "D": {"units": 0, "procs": 0}}))
        self.assertFalse(P.units_gone({"F": {"units": 1, "procs": 0}}))
        self.assertFalse(P.units_gone({"F": None}))
        self.assertFalse(P.units_gone({}))


GIB = 1 << 30


class ServingRoom(unittest.TestCase):
    """FLERP-P5 2026-11-12: the run left 76-91 GB of its own slices on F and D after taking StrixFold's engram dir;
    the pair then failed three starts with 'No space left on device'. The cleanup evicts the run's slices."""
    CONF = "ENGRAM_DIR='$HOME/../../cache/x'\nENGRAM_DIR=/cache/engram-1017   # the engram\nOTHER=/data/x\nW=\"/cache/w/\"\n# Z=/cache/z\n"

    def test_conf_paths(self):
        self.assertEqual(P.conf_cache_paths(self.CONF), ["/cache/engram-1017", "/cache/w"])
        self.assertEqual(P.conf_cache_paths("A=$HOME/x:/cache/a,/cache/b", home="/h"), ["/cache/a", "/cache/b"])

    def test_shortfall_and_lost(self):
        need = 196 * GIB
        pre = {"dirs": {"/cache/e": 196 * GIB}, "avail": 10 * GIB}
        gone = {"dirs": {}, "avail": 100 * GIB}          # run 11121559: engram removed, 76-91 GB slices left behind
        self.assertEqual(P.serving_lost(pre, gone), ["/cache/e"])
        self.assertEqual(P.serving_shortfall(gone, need), 96 * GIB)
        self.assertTrue(P.slices_to_evict(pre, gone, need))
        intact = {"dirs": {"/cache/e": 196 * GIB}, "avail": 1}
        self.assertFalse(P.slices_to_evict(pre, intact, need))
        # nothing of the engram ever there, but the disk cannot hold the set now: still evict
        self.assertTrue(P.slices_to_evict({"dirs": {}, "avail": 300 * GIB}, {"dirs": {}, "avail": 150 * GIB}, need))
        self.assertFalse(P.slices_to_evict({"dirs": {}, "avail": 300 * GIB}, {"dirs": {}, "avail": 250 * GIB}, need))

    def _window(self, probes, place):
        cfg = {"host": {"unit_prefix": "u-", "hold_owner": "me", "serving_need_gib": 196, "nodes": [{"name": "F", "mgmt": "ipF"}, {"name": "A", "mgmt": "ipA"}]}}
        w = P.Window(cfg, lambda t: None)
        seq = {k: list(v) for k, v in probes.items()}
        w.serving_probe = lambda n: seq[n["name"]].pop(0)
        w.pre_serving = {"F": {"dirs": {"/cache/e": 196 * GIB}, "avail": 0}, "A": {"dirs": {}, "avail": 250 * GIB}}
        self.addCleanup(setattr, P, "place_cmd", P.place_cmd)
        P.place_cmd = lambda: place
        return w

    def test_evicts_own_slices_where_window_took_data(self):
        with tempfile.TemporaryDirectory() as d:
            fake = os.path.join(d, "place")
            with open(fake, "w") as f:
                f.write("#!/bin/sh\necho \"$@\" >> " + os.path.join(d, "calls") + "\n")
            os.chmod(fake, 0o755)
            # F: engram gone and 100 GiB free (short), after eviction 200 GiB free (fits); A: untouched
            w = self._window({"F": [{"dirs": {}, "avail": 100 * GIB}, {"dirs": {}, "avail": 290 * GIB}],
                              "A": [{"dirs": {}, "avail": 250 * GIB}]}, [fake, "place"])
            self.assertEqual(w.restore_serving_room(), {})
            calls = open(os.path.join(d, "calls")).read().splitlines()
            self.assertEqual(calls, ["place cancel --id flerp-p5-F", "place evict --node F --owner me --apply"])

    def test_reports_node_still_short(self):
        with tempfile.TemporaryDirectory() as d:
            fake = os.path.join(d, "place")
            with open(fake, "w") as f:
                f.write("#!/bin/sh\nexit 0\n")
            os.chmod(fake, 0o755)
            w = self._window({"F": [{"dirs": {}, "avail": 100 * GIB}, {"dirs": {}, "avail": 120 * GIB}],
                              "A": [{"dirs": {}, "avail": 250 * GIB}]}, [fake, "place"])
            self.assertEqual(w.restore_serving_room(), {"F": 76 * GIB})

    def test_unreadable_probe_still_evicts_and_reports(self):
        with tempfile.TemporaryDirectory() as d:
            fake = os.path.join(d, "place")
            with open(fake, "w") as f:
                f.write("#!/bin/sh\nexit 0\n")
            os.chmod(fake, 0o755)
            w = self._window({"F": [None, None], "A": [{"dirs": {}, "avail": 250 * GIB}]}, [fake, "place"])
            self.assertEqual(w.restore_serving_room(), {"F": None})


class Flags(unittest.TestCase):
    def test_runner_flags_and_help(self):
        cfg = P.load_cfg()
        want = P.runner_gen_flags(cfg)
        self.assertTrue({"--dump", "--dump-layers", "--dump-rows", "--swarm-manifest", "--data-file"} <= want)
        src = open(os.path.join(ROOT, "catalog", "flerp", "tool", "qwen35moe_generate", "src", "flerp", "qwen35moe_tool",
                                "stage_driver.cpp")).read()
        i = src.index("for (const char* k", src.index('a == "--help"'))
        listed = P.help_flags(src[i:src.index("return 0;", i)])
        self.assertEqual(want - listed, set(), "the stage driver's --help list lacks options the runner passes")
        parsed = set(__import__("re").findall(r'a == "(--[a-z-]+)"', src))
        self.assertEqual(listed - parsed, set(), "--help lists an option the parser does not take")


class Judgements(unittest.TestCase):
    def test_split_and_equal(self):
        ev = [staged(0), staged(1), {"event": "x"}, staged(0), staged(1)]
        runs = P.split_runs(ev, 2)
        self.assertEqual(len(runs), 2)
        self.assertEqual(P.judge_equal_runs(runs[0], runs[1], "w4"), [])

    def test_differences(self):
        a, b = [staged(0)], [staged(0, sha=("a", "z"))]
        self.assertTrue(any("row 1" in w for w in P.judge_equal_runs(a, b, "x")))
        self.assertTrue(any("ids differ" in w for w in P.judge_equal_runs([staged(0)], [staged(0, ids=(1, 3))], "x")))
        self.assertTrue(any("argmax" in w for w in P.judge_equal_runs([staged(0, argmax=False)], [staged(0)], "x")))
        self.assertTrue(P.judge_equal_runs([staged(0)], [], "x"))

    def test_speed(self):
        s = P.speed([staged(0)])
        self.assertEqual(s[0]["decode_tok_s"], 10.0)
        self.assertEqual(s[0]["ttft_s"], 1.5)

    def test_have_reply(self):
        n = 130
        words = [2 ** 64 - 1, 2 ** 64 - 1, 0b11]
        data = struct.pack("<Q", n) + struct.pack("<3Q", *words)
        self.assertEqual(P.have_reply(data, n), 130)
        with self.assertRaises(ValueError):
            P.have_reply(data, 129)

    def test_window(self):
        r = [{"tensor_bytes": 50e9}] * 4
        w = P.window_estimate(r, r[:3], 0.2)
        self.assertAlmostEqual(w["w4_load"], 200 / 0.2 / 60, places=1)

    def test_config_parses(self):
        cfg = P.load_cfg()
        self.assertEqual([n["name"] for n in P.nodes_of(cfg, 4)], ["F", "D", "A", "C"])
        self.assertEqual([n["name"] for n in P.nodes_of(cfg, 3)], ["F", "D", "A"])
        self.assertEqual(len(cfg["run"]["check_blocks4"]), 4)


class Frames(unittest.TestCase):
    def test_frames_to_file(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "f.bin")
            stream = struct.pack("<QQ", 10, 3) + b"abc" + struct.pack("<QQ", 0, 2) + b"xy" + struct.pack("<QQ", 2 ** 64 - 1, 0)
            p = subprocess.run([sys.executable, os.path.join(ROOT, "gate", "swarm_frames_to_file.py"), out, "16"], input=stream, capture_output=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(out, "rb") as f:
                self.assertEqual(f.read(), b"xy" + b"\0" * 8 + b"abc" + b"\0" * 3)
            p = subprocess.run([sys.executable, os.path.join(ROOT, "gate", "swarm_frames_to_file.py"), out, "16"],
                               input=struct.pack("<QQ", 0, 2) + b"xy", capture_output=True)
            self.assertNotEqual(p.returncode, 0)


if __name__ == "__main__":
    unittest.main()
