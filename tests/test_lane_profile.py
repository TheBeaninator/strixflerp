"""gate/lane_profile.py without a pod: the lane_rank.sh patch and the --profile 2 parser on synthetic lines."""
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import lane_profile as L  # noqa: E402

LANE_RANK = """#!/bin/bash
set -euo pipefail
cd "$DIR"
echo $$ > "rank$R.pid"
""" + L.EXEC_LINE + "\n"

R0 = """{"rank": 0, "loaded_s": 40.1, "pool": 4}
{"rank": 0, "request": 0, "name": "p8k", "prompt_tokens": 8163, "prefill_s": 9.50}
{"rank": 0, "request": 0, "name": "p8k", "tokens": 256, "reason": "length", "rounds": 120, "drafted": 300, "accepted": 136, "seconds": 18.2, "reply": [1, 2]}
{"rank": 0, "profile": {"rounds": 120, "rows_a_round": 3.1, "step_ms": 70.0}}
{"rank": 0, "prefill_phases_ms": {"start": 0, "embed": 1.0, "mix_attn": 3.0, "attn": 10.0, "gather_a": 2.0, "moe": 30.0, "gather_m": 4.0, "a_q": 0.0}, "chunks": 4}
{"rank": 0, "phases_ms": {"start": 0, "attn": 6.0, "moe": 14.0, "gather_m": 0.0}}
{"rank": 0, "requests": 1, "tokens": 256, "steps": 120, "seconds": 28.0, "all_equal": true}
"""
R1 = """not json
{"rank": 1, "loaded_s": 39.0}
{"rank": 1, "follow_profile": {"rounds": 120, "enqueue_ms": 1.0, "forward_ms": 60.0}}
{"rank": 1, "prefill_phases_ms": {"attn": 12.0, "moe": 28.0}, "chunks": 4}
"""


class LaneProfile(unittest.TestCase):
    def test_patch_applies_the_wrap_once(self):
        out = L.patch_lane_rank(LANE_RANK)
        self.assertNotIn(L.EXEC_LINE, out)
        self.assertEqual(out.count('"${wrap[@]}" "${SERVE_CMD[@]}"'), 1)
        self.assertEqual(out.count(L.WRAP_MARK), 1)  # the stage check counts these
        with self.assertRaises(ValueError):
            L.patch_lane_rank(LANE_RANK + L.EXEC_LINE)
        with self.assertRaises(ValueError):
            L.patch_lane_rank("exec foo\n")

    def test_patched_exec_puts_the_wrap_before_the_command(self):
        # the patched lines, run with exec replaced by echo: env -i ENV... WRAP... CMD...
        body = L.WRAP_EXEC.split("\n", 1)[1].replace('exec "${pin[@]}"', 'echo "${pin[@]}"')
        with tempfile.TemporaryDirectory() as d:
            script = ("set -euo pipefail\nR=1; RANK=1; pin=()\nSERVE_ENV=(A=1)\nSERVE_CMD=(/bin/lanes x)\n"
                      "WRAP1='/usr/bin/env LD_LIBRARY_PATH=/opt/lib'\n" + body + "\n")
            subprocess.run(["bash", "-c", script], cwd=d, check=True)
            self.assertEqual(open(os.path.join(d, "rank1.jsonl")).read().split(),
                             ["env", "-i", "A=1", "/usr/bin/env", "LD_LIBRARY_PATH=/opt/lib", "/bin/lanes", "x"])
            self.assertIn("wrap (WRAP1, as serve_spawn): /usr/bin/env LD_LIBRARY_PATH=/opt/lib",
                          open(os.path.join(d, "rank1.cmd")).read())
            # no WRAP for the rank: the command as before, and the record says so
            subprocess.run(["bash", "-c", script.replace("WRAP1=", "WRAP0=")], cwd=d, check=True)
            self.assertEqual(open(os.path.join(d, "rank1.jsonl")).read().split(), ["env", "-i", "A=1", "/bin/lanes", "x"])
            self.assertIn("(none)", open(os.path.join(d, "rank1.cmd")).read())

    def test_parse_and_table(self):
        doc = L.parse({0: R0, 1: R1})
        self.assertEqual(doc["prefill"][0]["chunks"], 4)
        self.assertEqual(doc["prefill"][1]["ms"], {"attn": 12.0, "moe": 28.0})
        self.assertEqual(doc["decode"], {"rank": 0, "ms": {"start": 0, "attn": 6.0, "moe": 14.0, "gather_m": 0.0}, "rounds": 120})
        self.assertEqual(doc["summary"]["tokens"], 256)
        self.assertEqual(len(doc["requests"]), 2)
        t = L.table(doc)
        self.assertIn("prefill, rank 0: 4 chunks, 50.00 ms GPU a chunk, 0.20 s over the prompt", t)
        self.assertIn("| moe | 30.000 | 120.0 | 60.0% |", t)
        self.assertIn("| attn | 12.000 | 48.0 | 30.0% |", t)
        self.assertIn("decode, rank 0: 20.00 ms GPU a timed round (120 rounds)", t)
        self.assertIn("| moe | 14.000 | 70.0% |", t)
        self.assertNotIn("| a_q |", t)  # zero phases left out
        self.assertIn("8163 prompt tokens, prefill 9.5 s", t)

    def test_no_decode_line(self):
        t = L.table(L.parse({1: R1}))
        self.assertIn("decode: no phases_ms line", t)

    def test_untimed_decode(self):
        t = L.table(L.parse({0: '{"rank": 0, "phases_ms": {"attn": 0, "moe": 0}}\n'}))
        self.assertIn("no timed rounds", t)


if __name__ == "__main__":
    unittest.main()
