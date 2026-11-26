"""The strixw-decode interface gate's CPU reference (gate/interface/reference/strixw_decode.hpp) against strixite's
own format code, on the host (no GPU), and the provider case_args of gate/interface.py. python3 -m unittest"""
import importlib.util
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FMT = os.path.join(ROOT, "catalog", "strixite", "format")


def load_interface():
    spec = importlib.util.spec_from_file_location("gate_interface", os.path.join(ROOT, "gate", "interface.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(shutil.which("g++"), "no g++ on this host")
class ReferenceSelfTest(unittest.TestCase):
    def test_reference_reads_strixite_formats(self):
        with tempfile.TemporaryDirectory() as d:
            exe = os.path.join(d, "selftest")
            srcs = [os.path.join(FMT, f"q{b}", "src", "formats", f"q{b}.cpp") for b in (4, 5, 6, 8)]
            incs = [f"-I{os.path.join(FMT, f'q{b}', 'src')}" for b in (4, 5, 6, 8)]
            incs += [f"-I{os.path.join(ROOT, 'catalog', 'strixite', 'runtime', 'check', 'src')}",
                     f"-I{os.path.join(ROOT, 'gate', 'interface', 'reference')}"]
            subprocess.run(["g++", "-std=c++20", "-O1", "-ffp-contract=off", *incs, "-o", exe,
                            os.path.join(ROOT, "tests", "cpp", "strixw_decode_ref_selftest.cpp"), *srcs], check=True)
            p = subprocess.run([exe], capture_output=True, text=True, timeout=300)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertEqual(p.stdout.count(": ok"), 12, p.stdout)


class CaseArgsTest(unittest.TestCase):
    def test_provider_case_args_reach_its_cases_only(self):
        iface = load_interface()
        cfg = iface.load_config("strixw-decode")
        by = {p["name"]: p for p in cfg["provider"]}
        panel = iface.case_lines(cfg, by["flerp-q4p"]).splitlines()
        row = iface.case_lines(cfg, by["strixite-q4"]).splitlines()
        self.assertTrue(panel and all("layout=panel" in ln for ln in panel))
        self.assertTrue(row and all("layout=row" in ln for ln in row))
        self.assertEqual([ln.split()[0] for ln in panel], [ln.split()[0] for ln in row])   # same cases, compared bit for bit
        self.assertTrue(all(ln.split()[0].startswith("q4-") for ln in panel))
        cfg2 = {"case": [{"name": "a", "kind": "parity", "layout": "row"}]}
        self.assertIn("layout=row", iface.case_lines(cfg2, {"case_args": {"layout": "panel"}}))   # a case's own key wins


if __name__ == "__main__":
    unittest.main()
