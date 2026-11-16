"""gate/summary.py, the one summary writer of the gate scripts (card 6249b8db6f), against the format the thirteen
scripts wrote before it: json.dump indent 1 plus a newline, lines joined with a trailing newline, and the verdicts."""
import hashlib
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import summary as SM  # noqa: E402

GATE_SCRIPTS = ["executor_abi", "flerp_serve", "flerp_serve_pair", "gufo_recipe", "moe_placed", "moe_recipe", "moe_serve_stage",
                "moe_stage", "strixfold_recipe", "strixite_prefill", "strixite_recipe", "gguf_census", "replay"]


class R:
    def __init__(self, config):
        self.name, self.gate = "r", {"config": config}


class T(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.d = self.dir.name

    def tearDown(self):
        self.dir.cleanup()

    def test_json_is_the_old_format_plus_config_sha256_last(self):
        doc = {"recipe": "r", "pass": True, "stages": {"check": {"pass": True}}}
        cfg = os.path.join(self.d, "g.toml")
        with open(cfg, "w") as f:
            f.write("[criteria]\nx = 1\n")
        out = SM.write_json(self.d, doc, config=cfg)
        with open(os.path.join(self.d, "summary.json")) as f:
            text = f.read()
        old = json.dumps(doc, indent=1) + "\n"                   # what every script wrote before
        self.assertEqual(text, old[:-3] + ',\n "config_sha256": "' + hashlib.sha256(b"[criteria]\nx = 1\n").hexdigest() + '"\n}\n')
        self.assertEqual(list(json.loads(text))[-1], "config_sha256")
        self.assertEqual(out["config_sha256"], hashlib.sha256(b"[criteria]\nx = 1\n").hexdigest())
        self.assertEqual(SM.FIELD, "config_sha256")             # read by agent's gate-provenance check: the name is fixed
        self.assertEqual(doc.keys(), {"recipe", "pass", "stages"})   # the caller's document is not changed

    def test_no_config_writes_null_and_other_names(self):
        SM.write_json(self.d, {"a": 1}, name="results.json")
        with open(os.path.join(self.d, "results.json")) as f:
            self.assertEqual(json.load(f), {"a": 1, "config_sha256": None})
        self.assertIsNone(SM.config_sha256(None))
        self.assertIsNone(SM.config_sha256(os.path.join(self.d, "missing.toml")))

    def test_config_is_discovered_from_the_scripts_ctx(self):
        rel = "gate/qwen38-flash-next.strixite.toml"
        want = os.path.join(ROOT, rel)
        self.assertEqual(SM.config_path({"recipe": R(rel)}), want)             # recipe gates: recipe.gate["config"]
        self.assertEqual(SM.config_path({"config": rel}), want)                # moe_* scripts: --config, relative to the tree
        self.assertEqual(SM.config_path({"config_rel": rel}), want)            # strixite_prefill
        self.assertEqual(SM.config_path({"config": want}), want)
        self.assertIsNone(SM.config_path({}))
        self.assertIsNone(SM.config_path(None))
        self.assertIsNone(SM.config_path({"recipe": object()}))
        with open(want, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(SM.write_json(self.d, {}, ctx={"recipe": R(rel)})["config_sha256"], digest)

    def test_md_is_the_lines_joined(self):
        self.assertEqual(SM.write_md(self.d, ["# T", "", "x"]), "# T\n\nx\n")
        with open(os.path.join(self.d, "summary.md")) as f:
            self.assertEqual(f.read(), "# T\n\nx\n")
        SM.write_md(self.d, ["a", "b"], name="other.md", newline=False)     # replay's recorded format has no trailing newline
        with open(os.path.join(self.d, "other.md")) as f:
            self.assertEqual(f.read(), "a\nb")

    def test_verdicts_and_bullets(self):
        S = ["check", "build", "gpu"]
        self.assertTrue(SM.all_pass(S, {s: {"pass": True} for s in S}))
        self.assertFalse(SM.all_pass(S, {"check": {"pass": True}}))
        self.assertFalse(SM.all_pass(S, {"check": {"pass": True}, "build": None, "gpu": {"pass": True}}))
        self.assertEqual(SM.verdict(S, {s: {"pass": True} for s in S}), "PASSED")
        self.assertEqual(SM.verdict(S, {"check": {"pass": True}}), "NOT COMPLETE: build, gpu not run")
        self.assertEqual(SM.verdict(S, {"check": {"pass": True}, "build": {"pass": False}}), "FAILED")
        self.assertEqual(SM.verdict(S, {s: {"pass": False} for s in S}), "FAILED")
        self.assertEqual(SM.stage_bullets(S, {"check": {"pass": True}, "gpu": {"pass": False, "blocked": "no GPU", "error": "e"}}),
                         ["- **check**: PASS", "- **gpu**: FAIL - no GPU - e"])
        self.assertEqual(SM.stage_bullets(S, {}), [])

    def test_every_gate_script_imports_the_writer_and_writes_no_summary_itself(self):
        for name in GATE_SCRIPTS:
            with open(os.path.join(ROOT, "gate", name + ".py")) as f:
                src = f.read()
            self.assertIn("import summary as SM", src, name)
            self.assertNotIn('"summary.json"), "w")', src, name)
            self.assertNotIn('"summary.md"), "w")', src, name)
        others = [f for f in os.listdir(os.path.join(ROOT, "gate")) if f.endswith(".py") and f[:-3] not in GATE_SCRIPTS]
        for f in others:
            with open(os.path.join(ROOT, "gate", f)) as fh:
                self.assertNotIn("def write_summary", fh.read(), f)


if __name__ == "__main__":
    unittest.main()
