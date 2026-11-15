import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flerplib import champion as C, configs as CF  # noqa: E402


class T(unittest.TestCase):
    def test_serve_config_is_both_champions(self):
        c = CF.load("ornith397.serve")
        self.assertEqual(c["chain"], ["ornith397.decode-best", "ornith397.prefill-best", "ornith397.serve"])
        self.assertEqual((c["stage"]["reply"], CF.drafts(c), c["env"]["FLERP_MOE_GROUPED"]), ("token", 1, "1"))

    def test_old_behaviour_is_one_override(self):
        c = CF.override(CF.load("ornith397.serve"), ["drafts=0", "env.FLERP_MOE_GROUPED=0"])
        self.assertEqual((CF.drafts(c), c["env"]["FLERP_MOE_GROUPED"]), (0, "0"))
        self.assertEqual(CF.load("none")["stage"], {})

    def test_promotion_rules(self):
        d = tempfile.mkdtemp()
        led, doc = os.path.join(d, "l.toml"), os.path.join(d, "d.md")
        shutil.copy(C.LEDGER, led)
        r = os.path.join(d, "r")
        P = lambda: C.propose("ornith397.prefill-best", "prefill", r, ledger=led, doc=doc)
        C.write_result(r, "ornith397.prefill-best", "prefill", 1200, "t", {"kl_vs_fp32": 0.019})
        self.assertFalse(P()[0])      # does not beat the incumbent
        C.write_result(r, "ornith397.prefill-best", "prefill", 1500, "t", {"kl_vs_fp32": 0.03})
        self.assertFalse(P()[0])      # fails the floor
        C.write_result(r, "ornith397.prefill-best", "prefill", 1500, "t", {"kl_vs_fp32": 0.019})
        self.assertTrue(P()[0])
        areas, champs = C.load_ledger(led)
        self.assertEqual(C.current(champs, "prefill")["value"], 1500)
        self.assertEqual(sum(1 for c in champs if c["area"] == "prefill" and c["status"] == "history"), 2)


if __name__ == "__main__":
    unittest.main()
