"""catalog/flerp/tool/edge_reap_keepset/mask_keepset.py: Edge-REAP snapshots into a MASK_EXPERTS file (card 97f0f92e7a).

Proofs on synthetic snapshots (no GPU): the budget is the keep fraction with the floor of 6; each layer keeps its
highest-saliency experts (not the most frequent); a hub of any domain is kept though it ranks low in this one; hubs
beyond the budget are cut by share and said so; ties go to the lower id; the mask file parses with tools/ablate.py's
parse_mask (the engine's mask.zig refusals, its canonical id) at the real shape 40 x 384 and leaves 192 / 144 a layer,
never names an MTP layer or the shared expert, and is deterministic byte for byte; the CLI works on snapshot files."""
import json
import math
import os
import random
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "catalog", "flerp", "tool", "edge_reap_keepset", "src", "flerp",
                                "edge_reap_keepset"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import edge_reap_report as R  # noqa: E402
import mask_keepset as M  # noqa: E402
import ablate  # noqa: E402


def report(L, E, seed, hot=(), sal=None):
    """A valid report: random traffic, the 'hot' experts get most of the gate mass; sal overrides saliency."""
    rng = random.Random(seed)
    cnt = [[rng.randint(1, 50) + (5000 if j in hot else 0) for j in range(E)] for _ in range(L)]
    gm = [[c * rng.uniform(0.5, 1.5) / 100.0 for c in row] for row in cnt]
    s = [[rng.uniform(0.1, 2.0) for _ in range(E)] for _ in range(L)] if sal is None else sal
    return {"model_id": "fake", "num_layers": L, "num_experts": E, "reap_saliency": s,
            "reap_weighted_norm_sum": [[s[l][j] * cnt[l][j] for j in range(E)] for l in range(L)],
            "gate_mass_sum": gm, "token_count": cnt, "max_l2_norm": [[1.0] * E for _ in range(L)],
            "early_pos_count": [[0] * E for _ in range(L)]}


class Rules(unittest.TestCase):
    def test_budget_and_floor(self):
        self.assertEqual(M.budget(0.5, 384), 192)
        self.assertEqual(M.budget(0.375, 384), 144)
        self.assertEqual(M.budget(0.01, 384), 6)  # floor
        self.assertEqual(M.budget(1.0, 384), 384)
        for bad in (0, -0.1, 1.5, float("nan")):
            with self.assertRaises(M.KeepsetError):
                M.budget(bad, 384)

    def test_ranks_by_saliency_not_frequency(self):
        L, E = 2, 32
        rep = report(L, E, 1)
        # expert 0: huge traffic, tiny saliency; expert 31: tiny traffic, top saliency
        for l in range(L):
            rep["reap_saliency"][l][0] = 0.0
            rep["reap_saliency"][l][31] = 99.0
            rep["token_count"][l][31] = 1
            rep["gate_mass_sum"][l][31] = 0.0001
            rep["gate_mass_sum"][l][0] = 0.0001
        sol = M.solve(rep, 0.5, hub_share=2.0)  # no hubs
        for l in range(L):
            self.assertIn(31, sol["kept"][l])
            self.assertNotIn(0, sol["kept"][l])
            order = sorted(range(E), key=lambda j: (-rep["reap_saliency"][l][j], j))[:16]
            self.assertEqual(sol["kept"][l], sorted(order))

    def test_hub_of_another_domain_is_kept(self):
        L, E = 2, 64
        code = report(L, E, 2)
        for l in range(L):
            code["reap_saliency"][l][7] = 0.0  # worst saliency in this domain
        other = report(L, E, 3, hot=(7,))      # but a hub in the other
        without = M.solve(code, 0.25, hub_share=0.02)
        self.assertEqual([7 in without["kept"][l] for l in range(L)], [False] * L)
        with_hub = M.solve(code, 0.25, [other], hub_share=0.02)
        for l in range(L):
            self.assertIn(7, with_hub["kept"][l])
            self.assertIn(7, with_hub["hubs"][l])
            self.assertEqual(len(with_hub["kept"][l]), 16)

    def test_hubs_beyond_budget_cut_by_share(self):
        L, E = 1, 64
        rep = report(L, E, 4, hot=tuple(range(20)))
        sol = M.solve(rep, 0.125, hub_share=0.001)  # k = 8, many hubs
        self.assertEqual(len(sol["kept"][0]), 8)
        self.assertEqual(sol["hubs"][0], sol["kept"][0])
        self.assertGreater(sol["hubs_dropped"][0], 0)

    def test_ties_go_to_lower_id_and_floor_holds(self):
        L, E = 1, 32
        rep = report(L, E, 5, sal=[[1.0] * E])
        sol = M.solve(rep, 0.01, hub_share=2.0)
        self.assertEqual(sol["kept"][0], [0, 1, 2, 3, 4, 5])

    def test_deterministic(self):
        rep = report(3, 48, 6)
        self.assertEqual(M.solve(rep, 0.5), M.solve(rep, 0.5))


class MaskFile(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def write(self, name, rep):
        p = os.path.join(self.dir, name)
        with open(p, "w") as f:
            json.dump(rep, f)
        return p

    def test_real_shape_masks_parse_with_the_engine_rules(self):
        code = self.write("code.text.json", report(40, 384, 7))
        gen = self.write("general.text.json", report(40, 384, 8, hot=(5, 6)))
        for keep, k in ((0.5, 192), (0.375, 144)):
            res = M.build([code], keep, [gen], label="code")
            mask = ablate.parse_mask(res["text"])  # the engine's refusals: layers, experts, fewer than 6 left
            self.assertEqual(sorted(mask), list(range(40)))
            for l in range(40):
                self.assertEqual(384 - len(mask[l]), k)
            self.assertTrue(all(l < 40 for l in mask))      # no MTP block
            self.assertTrue(all(e < 384 for s in mask.values() for e in s))  # no shared expert
            self.assertEqual(ablate.mask_id(mask), res["info"]["mask_id"])
            self.assertEqual(M.canonical({l: sorted(s) for l, s in mask.items()}), ablate.canonical(mask))
            c = res["info"]["coverage"]["overall"]
            self.assertTrue(0.0 < c["saliency"] <= 1.0 and 0.0 < c["traffic"] <= 1.0)
            for l in range(40):  # the hubs of the other domain are kept
                self.assertNotIn(5, mask[l])
                self.assertNotIn(6, mask[l])

    def test_byte_deterministic_and_keep_one_masks_nothing(self):
        code = self.write("code.text.json", report(4, 32, 9))
        a, b = M.build([code], 0.5), M.build([code], 0.5)
        self.assertEqual(a["text"], b["text"])
        full = M.build([code], 1.0)
        self.assertEqual(ablate.parse_mask(full["text"]), {})
        self.assertEqual(full["info"]["mask_id"], "")

    def test_modalities_merge(self):
        t = self.write("code.text.json", report(2, 32, 10))
        v = self.write("code.vision.json", report(2, 32, 11))
        res = M.build([t, v], 0.5)
        self.assertEqual(res["info"]["tokens_routed"],
                         sum(sum(r) for r in R.load(t)["token_count"]) + sum(sum(r) for r in R.load(v)["token_count"]))

    def test_snap_dir_cli(self):
        self.write("code.text.json", report(4, 32, 12))
        self.write("code.vision.json", report(4, 32, 13))
        self.write("chat.text.json", report(4, 32, 14, hot=(3,)))
        out, rep = os.path.join(self.dir, "m.mask"), os.path.join(self.dir, "m.json")
        rc = M.main(["--snap-dir", self.dir, "--domain", "code", "--modality", "text", "--keep", "0.5", "--out", out,
                     "--report", rep, "--hub-min-tokens", "0"])
        self.assertEqual(rc, 0)
        info = json.load(open(rep))
        self.assertEqual(info["domain_files"], ["code.text.json"])
        self.assertIn("chat.text.json", info["hub_files"])
        self.assertTrue(os.path.exists(os.path.join(self.dir, "m.md")))
        mask = ablate.parse_mask(open(out).read(), layers=4, experts=32)
        for l in range(4):
            self.assertEqual(32 - len(mask[l]), 16)
            self.assertNotIn(3, mask[l])

    def test_thin_hub_report_is_skipped(self):
        code = self.write("code.text.json", report(2, 64, 20))
        thin = report(2, 64, 21, hot=(9,))
        thin["tokens"] = 13
        t = self.write("smoke.text.json", thin)
        res = M.build([code], 0.25, [t])
        self.assertEqual(res["info"]["hub_files_skipped_thin"], ["smoke.text.json"])
        self.assertEqual(res["info"]["hub_files"], [])
        thin["tokens"] = 50000
        t = self.write("smoke.text.json", thin)
        self.assertEqual(M.build([code], 0.25, [t])["info"]["hub_files"], ["smoke.text.json"])

    def test_overlap_jaccard(self):
        a, b = os.path.join(self.dir, "a.mask"), os.path.join(self.dir, "b.mask")
        with open(a, "w") as f:
            f.write("0:" + ",".join(str(i) for i in range(16, 32)) + "\n")  # keeps 0..15
        with open(b, "w") as f:
            f.write("0:" + ",".join(str(i) for i in range(8, 24)) + "\n")   # keeps 0..7, 24..31
        res = M.overlap(a, b, layers=1, experts=32)
        self.assertAlmostEqual(res["jaccard_median"], 8 / 24)
        self.assertEqual(M.overlap(a, a, layers=1, experts=32)["jaccard_median"], 1.0)

    def test_refusals(self):
        self.assertEqual(M.main(["--snap-dir", self.dir, "--domain", "kernel", "--keep", "0.5", "--out",
                                 os.path.join(self.dir, "x")]), 2)
        bad = self.write("bad.json", {"model_id": "x"})
        self.assertEqual(M.main(["--report-in", bad, "--keep", "0.5", "--out", os.path.join(self.dir, "y")]), 2)
        ok = self.write("ok.json", report(2, 32, 1))
        self.assertEqual(M.main(["--report-in", ok, "--keep", "0", "--out", os.path.join(self.dir, "z")]), 2)
        self.assertFalse(os.path.exists(os.path.join(self.dir, "z")))


if __name__ == "__main__":
    unittest.main()
