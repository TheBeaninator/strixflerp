"""gate/pair_nodes.py without a pod: the pair from synthetic serving conf text, with and without NODE1* keys, and
with command-line overrides."""
import argparse
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import pair_nodes as N  # noqa: E402

BASE = """# the ranks
WORLD=2
MASTER_IP=198.51.100.50
SERVE_HOST=192.0.2.50
SERVE_PORT=18080
"""
WITH_D = BASE + "NODE1=D\nNODE1_IP='192.0.2.53'   # rank 1\nNODE1_RAIL=198.51.100.53\n"
MAP = """dhcp-host=84:47:09:71:e7:60,192.0.2.50,ltsp-f
dhcp-host=84:47:09:71:e7:5d,192.0.2.53,ltsp-d
dhcp-host=84:47:09:85:03:67,192.0.2.54,ltsp-a
dhcp-host=84:47:09:88:78:fd,192.0.2.52,ltsp-c
"""


class PairNodes(unittest.TestCase):
    def test_conf_values_like_the_launcher(self):
        v = N.conf_values("A=1\nB='x y'  # c\nA=2\n  C=3\n")
        self.assertEqual(v, {"A": "2", "B": "x"})

    def test_node1_keys(self):
        p = N.resolve(WITH_D, MAP)
        self.assertEqual(p["rank0"], {"mgmt": "192.0.2.50", "rail": "198.51.100.50", "letter": "F", "rail_nic": None})
        self.assertEqual(p["rank1"], {"mgmt": "192.0.2.53", "rail": "198.51.100.53", "letter": "D", "rail_nic": None})
        self.assertEqual(p["serve_url"], "http://192.0.2.50:18080")
        self.assertEqual(p["fallback"], [])
        self.assertEqual(p["warnings"], [])
        self.assertEqual(p["source"]["rank1.mgmt"], "conf NODE1_IP")

    def test_no_node1_keys_falls_back_like_the_launcher_and_says_so(self):
        p = N.resolve(BASE, MAP)
        self.assertEqual((p["rank1"]["mgmt"], p["rank1"]["rail"], p["rank1"]["letter"]), ("192.0.2.54", "198.51.100.54", "A"))
        self.assertIn("rank1.mgmt", p["fallback"])
        self.assertIn("rank1.rail", p["fallback"])
        self.assertIn("FALLBACK", N.describe(p))
        self.assertIn("strixfold-pair.sh", p["source"]["rank1.mgmt"])

    def test_partial_keys_each_fall_back_on_its_own(self):
        p = N.resolve(BASE + "NODE1=D\nNODE1_IP=192.0.2.53\n", MAP)
        self.assertEqual(p["rank1"]["mgmt"], "192.0.2.53")
        self.assertEqual(p["fallback"], ["rank1.rail"])
        with self.assertRaises(ValueError):            # rank 1 on rank 0's rail is no pair
            N.resolve(BASE + "NODE1_IP=192.0.2.53\nNODE1_RAIL=198.51.100.50\n", MAP)

    def test_launcher_letter_mismatch_warns(self):
        p = N.resolve(BASE + "NODE1_IP=192.0.2.53\nNODE1_RAIL=198.51.100.53\n", MAP)
        self.assertEqual(p["rank1"]["letter"], "D")
        self.assertTrue(any("NODE1" in w for w in p["warnings"]))

    def test_override_wins(self):
        p = N.resolve(WITH_D, MAP, rank1="192.0.2.52", rank1_rail="198.51.100.52", serve_url="http://x:1")
        self.assertEqual((p["rank1"]["mgmt"], p["rank1"]["rail"], p["rank1"]["letter"]), ("192.0.2.52", "198.51.100.52", "C"))
        self.assertEqual(p["source"]["rank1.mgmt"], "override")
        self.assertEqual(p["serve_url"], "http://x:1")

    def test_override_mgmt_needs_its_rail(self):
        with self.assertRaises(ValueError):
            N.resolve(WITH_D, MAP, rank1="192.0.2.52")
        p = N.resolve(WITH_D, MAP, rank1="192.0.2.53")         # the conf's own node: its NODE1_RAIL holds
        self.assertEqual(p["rank1"]["rail"], "198.51.100.53")

    def test_no_rank0_is_an_error(self):
        with self.assertRaises(ValueError):
            N.resolve("NODE1_IP=192.0.2.53\n")

    def test_args_and_bind(self):
        with tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False) as f:
            f.write(WITH_D)
        try:
            a = N.add_args(argparse.ArgumentParser()).parse_args(["--serving-conf", f.name, "--rank1", "192.0.2.54",
                                                                  "--rank1-rail", "198.51.100.54"])
            p = N.from_args(a)
        finally:
            os.unlink(f.name)
        m = type(sys)("m")
        m.F = m.A = m.URL = None
        N.bind(p, m)
        self.assertEqual((m.F, m.A, m.URL), ("192.0.2.50", "192.0.2.54", "http://192.0.2.50:18080"))

    def test_rail_nic_cmd(self):
        self.assertIn("ip -o route get 198.51.100.53", N.rail_nic_cmd("198.51.100.53"))


if __name__ == "__main__":
    unittest.main()
