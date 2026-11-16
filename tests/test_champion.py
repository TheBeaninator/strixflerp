import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flerplib import champion as C, configs as CF  # noqa: E402

# A fixture ledger, never the live configs/champions.toml: the live floors and champions move with the pod's
# measurements, and a test that copies them breaks on every promotion (card 7d465f340b).
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "champions.toml")


class T(unittest.TestCase):
    def test_serve_config_is_both_champions(self):
        c = CF.load("ornith397.serve")
        self.assertEqual(c["chain"], ["ornith397.decode-best", "ornith397.prefill-best", "ornith397.serve"])
        self.assertEqual((c["stage"]["reply"], CF.drafts(c), c["env"]["FLERP_MOE_GROUPED"]), ("token", 1, "1"))

    def test_old_behaviour_is_one_override(self):
        c = CF.override(CF.load("ornith397.serve"), ["drafts=0", "env.FLERP_MOE_GROUPED=0"])
        self.assertEqual((CF.drafts(c), c["env"]["FLERP_MOE_GROUPED"]), (0, "0"))
        self.assertEqual(CF.load("none")["stage"], {})

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.led, self.doc = os.path.join(self.dir, "l.toml"), os.path.join(self.dir, "d.md")
        shutil.copy(FIXTURE, self.led)
        self.res = os.path.join(self.dir, "r")
        self.assertNotEqual(os.path.realpath(FIXTURE), os.path.realpath(C.LEDGER))

    def tearDown(self):
        shutil.rmtree(self.dir)

    def propose(self, config, area, case=""):
        return C.propose(config, area, self.res, case=case, ledger=self.led, doc=self.doc)

    def test_promotion_rules_higher_is_better(self):
        # The fixture incumbent is 1200 tok/s under a KL floor of 0.02. Each refusal is checked by its reason, so a
        # broken incumbent rule cannot hide behind the floor and a broken floor cannot hide behind the incumbent.
        C.write_result(self.res, "ornith397.prefill-best", "prefill", 1100, "t", {"kl_vs_fp32": 0.01})
        ok, why = self.propose("ornith397.prefill-best", "prefill")
        self.assertFalse(ok)
        self.assertIn("does not beat the incumbent ornith397.prefill-best 1200", why)     # floor passed, incumbent refused
        self.assertNotIn("floor", why)
        C.write_result(self.res, "ornith397.prefill-best", "prefill", 1500, "t", {"kl_vs_fp32": 0.03})
        ok, why = self.propose("ornith397.prefill-best", "prefill")
        self.assertFalse(ok)
        self.assertTrue(why.startswith("fidelity floor: KL vs FP32 0.03 > 0.02"), why)    # would beat 1200, floor refused
        self.assertNotIn("incumbent", why)
        C.write_result(self.res, "ornith397.prefill-best", "prefill", 1500, "t", {"kl_vs_fp32": 0.01})
        ok, why = self.propose("ornith397.prefill-best", "prefill")
        self.assertTrue(ok, why)
        self.assertIn("replaces ornith397.prefill-best 1200", why)
        areas, champs = C.load_ledger(self.led)
        self.assertEqual(C.current(champs, "prefill")["value"], 1500)
        hist = [c for c in champs if c["area"] == "prefill" and c["status"] == "history"]
        self.assertEqual([c["value"] for c in hist], [1200])
        self.assertTrue(hist[0]["superseded_by"].startswith("ornith397.prefill-best t"))
        # The live ledger is not what was written.
        self.assertTrue(os.path.isfile(self.doc))
        self.assertNotEqual(os.path.realpath(self.led), os.path.realpath(C.LEDGER))

    def test_promotion_rules_lower_is_better_and_identity_floor(self):
        # No ttft incumbent in the fixture: the first result that passes the floor is promoted whatever its value.
        C.write_result(self.res, "ornith397.baseline-p5", "ttft", 0.5, "t", {"identical": True}, case="21 tok")
        ok, why = self.propose("ornith397.baseline-p5", "ttft", case="21 tok")
        self.assertTrue(ok, why)
        self.assertNotIn("replaces", why)
        # Lower is better: a slower time with identical ids fails only the incumbent rule.
        C.write_result(self.res, "ornith397.baseline-p5", "ttft", 0.6, "t", {"identical": True}, case="21 tok")
        ok, why = self.propose("ornith397.baseline-p5", "ttft", case="21 tok")
        self.assertFalse(ok)
        self.assertIn("does not beat the incumbent ornith397.baseline-p5 0.5 (lower is better)", why)
        # A faster time without identity fails only the floor.
        C.write_result(self.res, "ornith397.baseline-p5", "ttft", 0.4, "t", {"identical": False}, case="21 tok")
        ok, why = self.propose("ornith397.baseline-p5", "ttft", case="21 tok")
        self.assertFalse(ok)
        self.assertTrue(why.startswith("fidelity floor: identity not shown"), why)
        self.assertNotIn("incumbent", why)
        # A different case is its own ledger row: no incumbent, promoted.
        C.write_result(self.res, "ornith397.baseline-p5", "ttft", 9.0, "t", {"identical": True}, case="1808 tok")
        self.assertTrue(self.propose("ornith397.baseline-p5", "ttft", case="1808 tok")[0])
        _, champs = C.load_ledger(self.led)
        self.assertEqual(C.current(champs, "ttft", "21 tok")["value"], 0.5)
        self.assertEqual(C.current(champs, "ttft", "1808 tok")["value"], 9.0)

    def test_refusals_before_the_rules(self):
        self.assertFalse(self.propose("ornith397.prefill-best", "nosuch")[0])
        ok, why = self.propose("ornith397.prefill-best", "prefill")
        self.assertFalse(ok)
        self.assertIn("champion.json missing", why)
        C.write_result(self.res, "ornith397.prefill-best", "prefill", 1500, "t", {"kl_vs_fp32": 0.01})
        ok, why = self.propose("ornith397.prefill-exact", "prefill")        # the results name another config
        self.assertFalse(ok)
        self.assertIn("results say config 'ornith397.prefill-best'", why)
        ok, why = C.propose("ornith397.prefill-best", "prefill", self.res, dry=True, ledger=self.led, doc=self.doc)
        self.assertTrue(ok)
        self.assertTrue(why.startswith("would promote"))
        self.assertEqual(C.current(C.load_ledger(self.led)[1], "prefill")["value"], 1200)   # a dry run writes nothing


if __name__ == "__main__":
    unittest.main()
