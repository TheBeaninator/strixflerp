"""The judgements of gate/moe_stage.py (no GPU, no catalogue)."""
import os
import signal
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import moe_stage as M  # noqa: E402

REF = {"ids": [1, 2, 3], "logits_sha256": ["a", "b", "c"]}


def staged(config="2:even:on", prompt=0, ids=(1, 2, 3), sha=("a", "b", "c"), argmax=True):
    return {"event": "staged", "config": config, "prompt": prompt, "ids": list(ids), "rows": len(sha),
            "logits_sha256": list(sha), "argmax_is_token": argmax}


class Judgements(unittest.TestCase):
    def test_equal(self):
        self.assertEqual(M.judge_staged(staged(), REF), [])

    def test_row_and_ids(self):
        self.assertTrue(any("row 1" in w for w in M.judge_staged(staged(sha=("a", "x", "c")), REF)))
        self.assertTrue(any("ids differ" in w for w in M.judge_staged(staged(ids=(1, 2, 4)), REF)))
        self.assertTrue(any("rows" in w for w in M.judge_staged(staged(sha=("a", "b")), REF)))
        self.assertTrue(M.judge_staged(staged(argmax=False), REF))

    def test_matrix_missing(self):
        rows, why = M.judge_matrix([staged()], ["2:even:on"], ["p", "q"], {"p": REF, "q": REF}, "loopback")
        self.assertEqual(rows[0]["prompts_equal"], 1)
        self.assertTrue(any("no result" in w for w in why))

    def test_loop_faults(self):
        ev = [{"event": "fault_case", "fault": f, "fired": True, "rank0_call_failed": True, "all_poisoned": True,
               "reasons": ["epoch 6 ordinal 7"]} for f in ("drop", "stale-epoch", "replay", "bad-bytes")]
        ev.append({"event": "fault_case", "fault": "wrong-epoch-at-handshake", "refused": True})
        self.assertEqual(M.judge_loop_faults(ev)[1], [])
        ev[0]["all_poisoned"] = False
        self.assertTrue(M.judge_loop_faults(ev)[1])

    def test_tcp_fault(self):
        ok = {"rc": 3, "events": [{"event": "relay_failed", "t_unix": 101.0, "reason": "ordinal 4, expected 5"}]}
        ranks = {0: ok, 1: {"rc": -signal.SIGKILL, "events": []}, 2: ok}
        self.assertTrue(M.judge_tcp_fault("kill", ranks, 100.0, 15, {1})["pass"])
        self.assertFalse(M.judge_tcp_fault("kill", ranks, 80.0, 15, {1})["pass"])          # too late
        self.assertFalse(M.judge_tcp_fault("kill", {**ranks, 2: {"rc": None, "events": []}}, 100.0, 15, {1})["pass"])
        self.assertTrue(M.judge_tcp_fault("replay", {0: ok, 2: ok}, 100.0, 15)["pass"])
        self.assertFalse(M.judge_tcp_fault("stale-epoch", {0: ok, 2: ok}, 100.0, 15)["pass"])

    def test_overheads(self):
        r0 = [{"event": "staged", "config": "2:even:on", "prompt": 0, "calls": {"decode_wall_ms": 100.0, "decode_calls": 10, "chunk_wall_ms": 50.0, "chunks": 1},
               "stages": [{"rank": 0, "blocks": [0, 20], "decode_ms": 40.0, "chunks": [{"ms": 20.0, "ms_per_block": 1.0}], "logits_copy_ms": 0.0, "decode_ms_per_block": 0.2}]}]
        t = [{"event": "stage_timing", "run": 0, "timing": {"rank": 1, "blocks": [20, 40], "decode_ms": 40.0, "chunks": [{"ms": 20.0, "ms_per_block": 1.0}],
                                                         "logits_copy_ms": 10.0, "decode_ms_per_block": 0.2}}]
        o = M.tcp_overheads(r0, t)[0]
        self.assertAlmostEqual(o["decode_ms_per_token"], 1.0)
        self.assertAlmostEqual(o["chunk_ms_per_chunk"], 10.0)


if __name__ == "__main__":
    unittest.main()
