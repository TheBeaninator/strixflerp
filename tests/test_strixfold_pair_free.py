"""gate/strixfold_recipe.py's "is the pair free" lease check, without a pod: which pod-lease status lines name one of
the pair's workers. It must follow the LIVE pair (the serving conf can move rank 1 off A -- NODE1=D since 2026-11-11),
not a hardcoded F/A pattern: a worker:D lease was invisible to the check while D was rank 1 (card b390d523b9)."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import strixfold_recipe as R  # noqa: E402

PAIR_D = {"rank0": {"mgmt": "192.0.2.50", "rail": "198.51.100.50", "letter": "F"},
          "rank1": {"mgmt": "192.0.2.53", "rail": "198.51.100.53", "letter": "D"},
          "serve_url": "http://192.0.2.50:18080", "conf": "/x/tp2-128k.conf",
          "source": {}, "fallback": [], "launcher_rank1_letter": "D", "warnings": []}

LEASES = """RESOURCE     OWNER           LEFT   PRIO  WHY                                                   PROBE
bench:K      img-k           0h28m  1     IMG-K 52ee31ee19 ComfyUI Z-Image on K :8188 [yields]  unit=control:usr:img-k-keeper
control-gpu  pod-gpu:252883  3h59m  0     ISEKAID 63713f6bb7: wi.json scene seed 123          pid=control:252883:11833502
worker:D     sf-window       1h00m  1     a card window on the pair's rank 1
worker:F     other-card      0h30m  1     a run of something else
worker:A     old-pair-card   2h00m  1     the former rank 1, no longer the pair
worker:C     quad            1h00m  1     the quad's C
lane:4       lane4           0h10m  1     a lane slot
"""


class PairFreeLeases(unittest.TestCase):
    def test_names_either_rank_by_letter_or_address(self):
        hits = R.lease_lines_for(PAIR_D, LEASES)
        self.assertEqual([h.split()[0] for h in hits], ["worker:D", "worker:F"])

    def test_a_foreign_lease_that_names_the_pair_in_its_why_is_the_pair_too(self):
        text = LEASES + "\nnonpool:1     x               1h     1     kernel leg on 192.0.2.53\n"
        hits = [h.split(":")[0] for h in R.lease_lines_for(PAIR_D, text)]
        self.assertIn("nonpool", hits)

    def test_the_rail_address_counts_and_the_old_a_lease_does_not(self):
        text = ("worker:X y 1h 1 rail 198.51.100.53 is mine\n"
                "worker:A z 1h 1 at 192.0.2.54 and 198.51.100.54\n")
        self.assertEqual([h.split()[0] for h in R.lease_lines_for(PAIR_D, text)], ["worker:X"])

    def test_no_substring_matches(self):
        # a letter is a word: "D" must not match "DISK" / "card10D4", and "F" not "F00"
        text = "worker:DISK a 1h 1 nothing to do with rank\nreserve:10D44 b 1h 1 no\nworker:F c 1h 1 real\n"
        self.assertEqual([h.split()[0] for h in R.lease_lines_for(PAIR_D, text)], ["worker:F"])

    def test_empty_and_header(self):
        self.assertEqual(R.lease_lines_for(PAIR_D, ""), [])
        self.assertEqual(R.lease_lines_for(PAIR_D, "RESOURCE OWNER LEFT PRIO WHY PROBE"), [])


if __name__ == "__main__":
    unittest.main()
