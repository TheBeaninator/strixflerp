"""The judgements of gate/moe_placed.py (no I/O)."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import moe_placed as P  # noqa: E402

CRIT = {"rss_file_growth_max": 1 << 26, "check_records_min": 2, "check_tie_margin": 1e-6, "check_weights_max": 1e-4,
        "check_gate_max": 1e-3, "check_out_max": 1e-3}


def traced(ids, lse=1.5):
    return {"ids": ids, "steps": [{"position": i, "token": t, "logsumexp": lse, "top": [[t, 2.0], [t + 1, 1.0], [t + 2, 0.5]]} for i, t in enumerate(ids)]}


def loaded(**over):
    p = {"allocated_bytes": 1000, "plan_sha256": "abc", "tensors": 3, "tensor_bytes": 900, "rss_file_before_fill": 10, "rss_file_after_fill": 20,
         "single_reader": True, "direct_io": True}
    p.update(over)
    return {"placed": p, "weight_resident_bytes": 900}


def rec(**over):
    r = {"layer": 0, "pos": 0, "ids_equal": True, "ids_same_set": True, "weights_max_abs": 1e-7, "gate_logit_abs": 1e-6, "out_max_abs": 1e-6,
         "out_rms": 1.0, "out_max_abs_over_rms": 1e-5, "margin_to_next_expert": 1e-3}
    r.update(over)
    return r


class Judgements(unittest.TestCase):
    def test_compare_traced(self):
        a, b = traced([5, 6, 7]), traced([5, 6, 7])
        b["steps"] = [dict(s, top=s["top"][:2]) for s in b["steps"]]          # the record keeps fewer logits a row
        self.assertTrue(P.compare_traced(a, b)["equal"])
        c = traced([5, 6, 7])
        c["steps"][1]["top"][1][1] = 1.0000001
        r = P.compare_traced(c, traced([5, 6, 7]))
        self.assertFalse(r["equal"])
        self.assertEqual(r["first_row_difference"], 1)
        self.assertFalse(P.compare_traced(traced([5, 6, 7], lse=1.5000001), traced([5, 6, 7]))["equal"])
        self.assertFalse(P.compare_traced(traced([5, 6]), traced([5, 6, 7]))["equal"])
        self.assertFalse(P.compare_traced(traced([5, 6, 8]), traced([5, 6, 7]))["ids_equal"])

    def test_judge_load(self):
        plan = {"device_bytes": 1000, "tensors": 3, "tensor_bytes": 900}
        self.assertEqual(P.judge_load(loaded(), plan, "abc", CRIT), [])
        self.assertTrue(P.judge_load(loaded(allocated_bytes=1001), plan, "abc", CRIT))
        self.assertTrue(P.judge_load(loaded(), plan, "abd", CRIT))
        self.assertTrue(P.judge_load(loaded(rss_file_after_fill=(1 << 26) + 11), plan, "abc", CRIT))
        self.assertTrue(P.judge_load(loaded(direct_io=False), plan, "abc", CRIT))
        self.assertTrue(P.judge_load({"weight_resident_bytes": 900}, plan, "abc", CRIT))
        self.assertTrue(P.judge_load(dict(loaded(), weight_resident_bytes=5), plan, "abc", CRIT))

    def test_judge_block_check(self):
        self.assertTrue(P.judge_block_check({"detail": [rec(), rec(layer=3)]}, CRIT)["pass"])
        self.assertFalse(P.judge_block_check({"detail": [rec()]}, CRIT)["pass"])                                   # too few
        tie = rec(ids_same_set=False, ids_equal=False, margin_to_next_expert=1e-8, out_max_abs_over_rms=0.3)
        j = P.judge_block_check({"detail": [rec(), rec(), tie]}, CRIT)
        self.assertTrue(j["pass"])
        self.assertEqual(j["ties_excused"], 1)
        wrong = rec(ids_same_set=False, ids_equal=False, margin_to_next_expert=1e-2)
        self.assertFalse(P.judge_block_check({"detail": [rec(), rec(), wrong]}, CRIT)["pass"])
        self.assertFalse(P.judge_block_check({"detail": [rec(), rec(out_max_abs_over_rms=0.01)]}, CRIT)["pass"])
        self.assertFalse(P.judge_block_check({"detail": [rec(), rec(out_max_abs_over_rms=float("nan"))]}, CRIT)["pass"])
        self.assertFalse(P.judge_block_check({"detail": [rec(), rec(weights_max_abs=1e-3)]}, CRIT)["pass"])
        self.assertFalse(P.judge_block_check({"error": "boom"}, CRIT)["pass"])
        self.assertTrue(P.judge_block_check({"detail": [rec(), rec(ids_equal=False)]}, CRIT)["pass"])               # order is recorded only

    def test_judge_bench(self):
        o = {"non_finite": 0, "rms": 0.5}
        ev = {"event": "stage_bench", "phase": "synthetic", "chunk_out": o, "chunk_out_again": o, "chunk_out_third": o, "decode_out_after_chunk": o,
              "decode_fresh": {"out_after_with_logits": o}, "chunk_repeats_equal": True}
        self.assertEqual(P.judge_bench([ev], False), [])
        self.assertEqual(len(P.judge_bench([ev], True)), 1)                                                          # the tokens phase is missing
        self.assertEqual(P.judge_bench([ev, dict(ev, phase="tokens")], True), [])
        self.assertTrue(P.judge_bench([dict(ev, chunk_repeats_equal=False)], False))
        self.assertTrue(P.judge_bench([dict(ev, chunk_out={"non_finite": 3, "rms": 0.5})], False))
        self.assertTrue(P.judge_bench([dict(ev, decode_out_after_chunk={"non_finite": 0, "rms": 0.0})], False))

    def test_pick_blocks(self):
        sizes = {n: n * 4 for n in range(1, 13)}
        self.assertEqual(P.pick_blocks(sizes, 100, 30, 12, 9), 12)
        self.assertEqual(P.pick_blocks(sizes, 70, 30, 12, 9), 10)
        self.assertIsNone(P.pick_blocks(sizes, 60, 30, 12, 9))
        self.assertEqual(P.pick_blocks(sizes, 1000, 0, 10, 9), 10)


if __name__ == "__main__":
    unittest.main()
