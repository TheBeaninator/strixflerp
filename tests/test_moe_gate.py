"""The comparisons of gate/moe_recipe.py that need no host: llama.cpp's eval callback output, log-probabilities,
the [criteria] of one prompt, hidden states and router choices."""
import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import moe_recipe as M  # noqa: E402

CALLBACK = """common_debug_cb_eval:                  l_out-0 = (f32)        ADD(ffn_out-0{4, 7, 1, 1}, attn_residual-0{4, 7, 1, 1}}) = {4, 7, 1, 1}
    [
        [
            [      1.0000,       2.0000,      -3.5000,       4.0000  ],
            [      0.5000,       0.2500,       0.1250,       0.0000  ],
            [     -1.0000,      -2.0000,      -3.0000,      -4.0000  ],
            ..., 
            [      9.0000,       8.0000,       7.0000,       6.0000  ],
            [      5.0000,       4.0000,       3.0000,       2.0000  ],
            [      1.5000,       2.5000,       3.5000,       4.5000  ],
        ],
    ]
    sum = 51.625000
common_debug_cb_eval:           ffn_moe_topk-0 = (i32)       VIEW(ffn_moe_argsort-0{8, 2, 1, 1}, }) = {8, 2, 1, 1}
    [
        [
            [      5.0000,       1.0000,       7.0000,    ...,       2.0000,       0.0000,       3.0000  ],
            [      4.0000,       6.0000,       1.0000,    ...,       0.0000,       2.0000,       7.0000  ],
        ],
    ]
    sum = 48.000000
common_debug_cb_eval:                  other-0 = (f32)        ADD(a{1, 1, 1, 1}, }) = {1, 1, 1, 1}
    [
        [
            [      1.0000  ],
        ],
    ]
    sum = 1.000000
"""


def step(token, top, lse=0.0):
    return {"token": token, "top": top, "logsumexp": lse}


def prob(token, top):
    return {"id": token, "logprob": top[0][1], "top_logprobs": [{"id": i, "logprob": v} for i, v in top]}


class Callback(unittest.TestCase):
    def test_parse(self):
        t = M.parse_eval_callback(CALLBACK, r"(l_out|ffn_moe_topk)-\d+")
        self.assertEqual(sorted(t), ["ffn_moe_topk-0", "l_out-0"])
        self.assertEqual(t["l_out-0"]["dims"], [4, 7, 1, 1])
        self.assertEqual(len(t["l_out-0"]["rows"]), 6)
        self.assertEqual(t["l_out-0"]["rows"][0], [1.0, 2.0, -3.5, 4.0])
        self.assertEqual(t["l_out-0"]["sum"], 51.625)
        self.assertEqual(t["ffn_moe_topk-0"]["rows"][1], [4.0, 6.0, 1.0, 0.0, 2.0, 7.0])

    def test_input_ids(self):
        text = "0.00.840.191 I number of input tokens = 3\n0.00.840.192 I   14162\n0.00.840.192 I   25\n0.00.840.192 I   3855\ncommon_debug_cb_eval: x\n"
        self.assertEqual(M.callback_input_ids(text), [14162, 25, 3855])
        self.assertEqual(M.callback_input_ids("number of input tokens = 2\n  7\n  9\n"), [7, 9])
        self.assertEqual(M.callback_input_ids("nothing"), [])

    def test_corners(self):
        self.assertEqual(M.corner_rows(7), [0, 1, 2, 4, 5, 6])
        self.assertEqual(M.corner_rows(4), [0, 1, 2, 3])
        self.assertEqual(M.ends([1, 2, 3, 4, 5, 6, 7, 8]), [1, 2, 3, 6, 7, 8])
        self.assertEqual(M.ends([1, 2, 3]), [1, 2, 3])

    def test_router(self):
        t = M.parse_eval_callback(CALLBACK, r"ffn_moe_topk-\d+")
        mine = [{"layer": 0, "tokens": [{"token": 0, "ids": [5, 1, 7, 9, 9, 2, 0, 3]}, {"token": 1, "ids": [4, 6, 1, 9, 9, 7, 2, 0]}]}]
        r = M.compare_router(mine, t, 2)[0]
        self.assertEqual((r["tokens_compared"], r["same_order"], r["same_set"], r["slots"], r["slots_equal"]), (2, 1, 2, 12, 10))

    def test_router_when_llama_evaluated_fewer_tokens(self):
        t = M.parse_eval_callback(CALLBACK, r"ffn_moe_topk-\d+")
        mine = [{"layer": 0, "tokens": [{"token": 0, "ids": [5, 1, 7, 9, 9, 2, 0, 3]}, {"token": 1, "ids": [4, 6, 1, 9, 9, 0, 2, 7]},
                                        {"token": 2, "ids": [1, 1, 1, 1, 1, 1, 1, 1]}]}]
        r = M.compare_router(mine, t, 3, 2)[0]     # this side has three tokens, llama.cpp's callback two
        self.assertEqual((r["tokens_compared"], r["same_order"]), (2, 2))
        self.assertIn("error", M.compare_router(mine, t, 3)[0])

    def test_hidden(self):
        t = M.parse_eval_callback(CALLBACK, r"l_out-\d+")
        # a tensor of 4 values a row shows all four; this side reports three and three, so nothing lines up
        mine = [{"layer": 0, "sum_all_tokens_f32": 51.5, "last_token_sum_f32": 12.0, "last_token_l2": 2.0, "hidden": 4,
                 "corners": [{"token": k, "first3": [0, 0, 0], "last3": [0, 0, 0]} for k in (0, 1, 2, 4, 5, 6)]}]
        r = M.compare_hidden(mine, t, 7)[0]
        self.assertEqual(r["values_compared"], 0)
        self.assertEqual(r["sum"], [51.5, 51.625])


class Criteria(unittest.TestCase):
    crit = {"logprob_tolerance": 0.5, "tie_margin": 1.0}

    def rows(self, mine, llama, d):
        return M.compare_logprobs(mine, llama, d)

    def test_pair_difference(self):
        a = [step(7, [[7, 2.0], [8, 0.0]], 2.0), step(9, [[9, 1.0]], 1.0), step(1, [[1, 0.0]], 0.0)]
        b = M.as_steps([prob(7, [(7, -0.1), (8, -2.3)]), prob(4, [(9, -0.5), (4, -0.4)]), prob(1, [(1, 0.0)])])
        d = M.pair_difference(a, b)
        self.assertEqual(d["positions"], 2)            # stops after the position where the tokens part
        self.assertAlmostEqual(d["max_abs"], 0.5, places=4)
        self.assertAlmostEqual(d["first_position"]["max_abs"], 0.3, places=4)
        self.assertEqual(M.pair_difference([], b), {"positions": 0})

    def test_equal_ids_within_tolerance(self):
        mine = [step(7, [[7, 2.0], [8, 0.0]], 2.2), step(9, [[9, 1.0], [3, 0.5]], 1.5)]
        llama = [prob(7, [(7, -0.1), (8, -2.3)]), prob(9, [(9, -0.6), (3, -0.9)])]
        rows = self.rows(mine, llama, None)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]["max_abs"], 0.1, places=4)
        j = M.judge_prompt([7, 9], [7, 9], rows, self.crit)
        self.assertTrue(j["pass"], j)
        self.assertIsNone(j["first_difference"])
        self.assertEqual(j["agree"], 2)

    def test_few_tokens_is_not_a_failure_when_llama_gave_few(self):
        mine = [step(7, [[7, 2.0], [8, 0.0]], 2.2)]
        llama = [prob(7, [(7, -0.1), (8, -2.3)])]
        j = M.judge_prompt([7], [7], self.rows(mine, llama, None), self.crit)
        self.assertTrue(j["pass"], j)

    def test_out_of_tolerance(self):
        mine = [step(7, [[7, 2.0], [8, 0.0]], 2.2)]
        llama = [prob(7, [(7, -0.1), (8, -3.3)])]
        j = M.judge_prompt([7], [7], self.rows(mine, llama, None), self.crit)
        self.assertFalse(j["pass"])
        self.assertIn("tolerance", j["why"][0])

    def test_difference_at_a_tie_passes(self):
        mine = [step(7, [[7, 2.0], [8, 0.0]], 2.2), step(3, [[3, 1.0], [9, 0.9]], 1.7)]
        llama = [prob(7, [(7, -0.1), (8, -2.3)]), prob(9, [(9, -0.7), (3, -0.8)])]
        rows = self.rows(mine, llama, 1)
        j = M.judge_prompt([7, 3], [7, 9], rows, self.crit)
        self.assertTrue(j["pass"], j)
        self.assertEqual(j["first_difference"], 1)

    def test_difference_away_from_a_tie_fails(self):
        mine = [step(7, [[7, 2.0], [8, 0.0]], 2.2), step(3, [[3, 1.0], [9, 0.9]], 1.7)]
        llama = [prob(7, [(7, -0.1), (8, -2.3)]), prob(9, [(9, -0.1), (3, -2.8)])]
        j = M.judge_prompt([7, 3], [7, 9], self.rows(mine, llama, 1), self.crit)
        self.assertFalse(j["pass"])

    def test_difference_with_a_third_token_fails(self):
        mine = [step(5, [[5, 2.0], [7, 1.9], [8, 0.0]], 2.7)]
        llama = [prob(7, [(7, -0.7), (8, -0.8), (5, -3.0)])]
        j = M.judge_prompt([5], [7], self.rows(mine, llama, 0), self.crit)
        self.assertFalse(j["pass"])
        self.assertTrue(any("not one of" in w for w in j["why"]))


class Fidelity(unittest.TestCase):
    crit = {"kl_bound": 0.02, "kl_factor": 3.0, "kl_floor": 0.002, "maxdiff_factor": 3.0, "maxdiff_floor": 0.1, "mean_kl_factor": 1.5,
            "mean_kl_floor": 0.001, "top1_margin": 0.5, "coverage": 0.999}

    @staticmethod
    def dist(*probs):
        return {i: math.log(p) for i, p in enumerate(probs)}

    def test_metrics(self):
        ref = self.dist(0.7, 0.2, 0.0995, 0.0005)
        same = M.fidelity_metrics(ref, dict(ref), 1e-3)
        self.assertAlmostEqual(same["kl"], 0.0)
        self.assertAlmostEqual(same["covered"], 1.0)
        self.assertEqual((same["tokens_above_floor"], same["missing_above_floor"], same["top1_agrees"]), (3, 0, True))
        self.assertAlmostEqual(same["reference_margin"], math.log(0.7 / 0.2))
        eng = self.dist(0.2, 0.7, 0.0995, 0.0005)
        m = M.fidelity_metrics(ref, eng, 1e-3)
        self.assertAlmostEqual(m["kl"], 0.5 * math.log(0.7 / 0.2), places=6)
        self.assertAlmostEqual(m["max_abs_above_floor"], math.log(0.7 / 0.2))
        self.assertFalse(m["top1_agrees"])
        short = M.fidelity_metrics(ref, {0: math.log(0.7), 1: math.log(0.2)}, 1e-3)
        self.assertEqual(short["missing_above_floor"], 1)
        self.assertAlmostEqual(short["covered"], 0.9)

    def test_step_dist(self):
        self.assertEqual(M.step_dist({"logsumexp": 1.0, "top": [[5, 1.0], [6, 0.0]]}), {5: 0.0, 6: -1.0})

    def rows(self, held_kl, llama_kl, **over):
        base = {"covered": 1.0, "max_abs_above_floor": 0.05, "tokens_above_floor": 3, "missing_above_floor": 0, "top1_agrees": True,
                "reference_margin": 2.0}
        return [{"prompt": "p", "position": 0, "sides": {"a": dict(base, kl=held_kl, **over), "llama.cpp": dict(base, kl=llama_kl)}}]

    def test_judgement(self):
        j = M.judge_fidelity(self.rows(0.001, 0.004), ("a",), "llama.cpp", self.crit)
        self.assertTrue(j["pass"], j)
        self.assertFalse(M.judge_fidelity(self.rows(0.03, 0.02), ("a",), "llama.cpp", self.crit)["pass"])            # the bound
        self.assertFalse(M.judge_fidelity(self.rows(0.015, 0.004), ("a",), "llama.cpp", self.crit)["pass"])          # 3 x llama
        self.assertTrue(M.judge_fidelity(self.rows(0.0019, 0.0001), ("a",), "llama.cpp", self.crit)["pass"] is False)  # mean: 1.5 x, floor 0.001
        self.assertTrue(M.judge_fidelity(self.rows(0.0009, 0.0001), ("a",), "llama.cpp", self.crit)["pass"])         # under both floors
        self.assertFalse(M.judge_fidelity(self.rows(0.001, 0.004, top1_agrees=False), ("a",), "llama.cpp", self.crit)["pass"])
        self.assertTrue(M.judge_fidelity(self.rows(0.001, 0.004, top1_agrees=False, reference_margin=0.1), ("a",), "llama.cpp", self.crit)["pass"])
        self.assertFalse(M.judge_fidelity(self.rows(0.001, 0.004, max_abs_above_floor=0.4), ("a",), "llama.cpp", self.crit)["pass"])
        self.assertFalse(M.judge_fidelity(self.rows(0.001, 0.004, covered=0.9), ("a",), "llama.cpp", self.crit)["pass"])
        self.assertFalse(M.judge_fidelity(self.rows(0.001, 0.004), ("b",), "llama.cpp", self.crit)["pass"])          # not measured


if __name__ == "__main__":
    unittest.main()
