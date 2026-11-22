"""gate/strix_loop.py isa stage: a scan that did not run is never a PASS."""
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import strix_loop as S  # noqa: E402

HEAD = "unit\tkernel\tvgpr\tsgpr\tspill\tscratch\n"


def healthy(rows=3):
    return {"base": {"rc": 0, "units": 2, "rows": rows}, "variant": {"rc": 0, "units": 2, "rows": rows}}


class IsaCriterion(unittest.TestCase):
    def test_empty_scan_is_not_pass(self):
        isa = {"sides": {"base": {"rc": 1, "units": 0, "rows": 0}, "variant": {"rc": 1, "units": 0, "rows": 0}},
               "scratch": [], "diff_lines": [], "diff_rc": 1}
        isa["problems"] = S.isa_problems(isa["sides"], isa["diff_rc"])
        v = S.isa_criterion(isa, "recipe")
        self.assertFalse(v["pass"])
        self.assertFalse(v["ran"])
        self.assertIn("did not run", v["detail"])

    def test_compiled_but_empty_table_is_not_pass(self):
        sides = healthy()
        sides["variant"]["rows"] = 0
        v = S.isa_criterion({"sides": sides, "scratch": [], "problems": S.isa_problems(sides, 0)}, "recipe")
        self.assertFalse(v["pass"])
        self.assertIn("variant.tsv has no kernel rows", v["detail"])

    def test_healthy_scan_without_scratch_passes(self):
        v = S.isa_criterion({"sides": healthy(), "scratch": [], "problems": S.isa_problems(healthy(), 0)}, "recipe")
        self.assertTrue(v["pass"])
        self.assertNotIn("ran", v)

    def test_variant_scratch_fails(self):
        isa = {"sides": healthy(), "problems": [], "scratch": [{"side": "variant", "unit": "u", "kernel": "k", "scratch_bytes": 16}]}
        self.assertFalse(S.isa_criterion(isa, "recipe")["pass"])

    def test_legacy_state_without_rows_reads_the_table(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(S, "ROOT", d):
            for side in ("base", "variant"):
                open(os.path.join(d, side + ".tsv"), "w").close()
            sides = {s: {"rc": 0, "units": 1, "tsv": s + ".tsv"} for s in ("base", "variant")}
            v = S.isa_criterion({"sides": sides, "scratch": [], "diff_lines": []}, "recipe")
            self.assertFalse(v["pass"])
            with open(os.path.join(d, "variant.tsv"), "w") as f:
                f.write(HEAD + "u\tk\t32\t8\t0\t0\n")
            with open(os.path.join(d, "base.tsv"), "w") as f:
                f.write(HEAD + "u\tk\t32\t8\t0\t0\n")
            self.assertTrue(S.isa_criterion({"sides": sides, "scratch": [], "diff_lines": []}, "recipe")["pass"])

    def test_null_run_stays_report_only(self):
        sides = {"base": {"rc": 1, "units": 0, "rows": 0}, "variant": {"rc": 1, "units": 0, "rows": 0}}
        v = S.isa_criterion({"sides": sides, "scratch": [], "problems": S.isa_problems(sides, 1)}, "null")
        self.assertTrue(v["report_only"])
        self.assertFalse(v["pass"])


class StageIsaMissingToolchain(unittest.TestCase):
    """stage_isa end to end with the scan tool failing the way a missing clang++ does."""

    def test_crashed_scan_records_problems(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(S, "ROOT", d), \
                mock.patch.object(S, "sh", return_value=(1, 0.1)), \
                mock.patch.object(S.subprocess, "call", return_value=2), \
                mock.patch.object(S, "out", return_value=subprocess.CompletedProcess([], 1, "", "IndexError")):
            ctx = {"run": "t", "res": os.path.join(d, "res")}
            build = {"base": {"ninja": d}, "variant": {"ninja": d}}
            isa = S.stage_isa(SimpleNamespace(), ctx, {"units": ["x.hip"]}, build)
            self.assertTrue(isa["problems"])
            self.assertIn("isa-scan diff exited 1", isa["problems"])
            self.assertFalse(S.isa_criterion(isa, "recipe")["pass"])


class SfxP16Corrected(unittest.TestCase):
    def test_published_verdict_says_isa_did_not_run(self):
        import json
        v = json.load(open(os.path.join(ROOT, "gate/results/strix-loop/20261116-sfx-p16/verdict.json")))
        isa = [c for c in v["criteria"] if c["criterion"].startswith("isa")]
        self.assertEqual(len(isa), 1)
        self.assertFalse(isa[0]["pass"])
        self.assertEqual(v["verdict"], "FAIL")


if __name__ == "__main__":
    unittest.main()
