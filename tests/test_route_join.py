"""tools/route_join: join eval rows to route records and rank experts whose gate mass goes with wrong answers
(suspects) or right answers (keepers). Synthetic records only; no server.

Proofs: a planted suspect and a planted keeper have lift intervals excluding 1 and a null expert does not; a
task-level bootstrap does not let one hard task with many draws dominate the way a row-level bootstrap does;
a multi-turn attempt sums its records; a truncated last line, a missing record and a duplicate id are counted
and are not fatal; a cell under --min-attempts gets no verdict; the same seed writes the same bytes; --mask-out
is layer:expert lines and a directory only reads route-*.jsonl.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "tools"))
import route_join as J  # noqa: E402

LAYER = 5
SUSPECT, KEEPER, NULL = 11, 12, 13


def dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            if isinstance(r, str):
                f.write(r if r.endswith("\n") else r + "\n")
            else:
                f.write(json.dumps(r) + "\n")


def record(rid, layers):
    """layers: [(layer, mass, [(expert, mass, count), ...])]."""
    return {"v": 1, "id": rid, "request_id": "req-" + rid, "route_tag": None, "model": "deepseek-v4.1-flash",
            "build": "test", "world": "w", "t_start": 0, "t_end": 1, "finish": "stop",
            "rows": {"prompt": 4, "prompt_cached": 0, "decode": 2, "vision": 0}, "topk": 16,
            "layers": [{"l": l, "mass": mass, "count": sum(c for _, _, c in top), "top": [[e, m, c] for e, m, c in top]}
                       for l, mass, top in layers]}


def attempt_row(task, correct, ids, suite="math", arm="base", draw=1, status="ok", single=False):
    row = {"task": task, "suite": suite, "arm": arm, "draw": draw, "status": status, "score": {"correct": correct}}
    if single:
        row["route_record"] = ids[0] if ids else None
    else:
        row["route_records"] = ids
    return row


def planted_pair(task, draw_correct, draw_wrong, suite="math", arm="base"):
    """One task, one correct and one incorrect attempt. Expert 11 is the suspect, 12 the keeper, 13 null."""
    recs, rows = [], []
    for correct, draw, top in ((True, draw_correct, [(SUSPECT, 0.5, 1), (KEEPER, 4.0, 4), (NULL, 1.5, 2)]),
                               (False, draw_wrong, [(SUSPECT, 4.0, 4), (KEEPER, 0.5, 1), (NULL, 1.5, 2)])):
        rid = "rr-%s-%d" % (task, draw)
        recs.append(record(rid, [(LAYER, 10.0, top)]))
        rows.append(attempt_row(task, correct, [rid], suite=suite, arm=arm, draw=draw))
    return recs, rows


class Join(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="route-join-")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def go(self, rows, recs, **kw):
        rp, cp = os.path.join(self.d, "rows.jsonl"), os.path.join(self.d, "recs.jsonl")
        dump(rp, rows)
        dump(cp, recs)
        kw.setdefault("boot", 40)
        kw.setdefault("seed", 0)
        return J.join([rp], [cp], **kw)

    def test_planted_suspect_keeper_and_null(self):
        recs, rows = [], []
        for t in range(30):
            r, w = planted_pair("task-%02d" % t, 1, 2)
            recs += r
            rows += w
        rep = self.go(rows, recs)
        dom = rep["domains"]["math"]
        by = {(c["layer"], c["expert"]): c for c in dom["cells"]}
        suspect, keeper, null = by[(LAYER, SUSPECT)], by[(LAYER, KEEPER)], by[(LAYER, NULL)]
        self.assertGreater(suspect["ci"][0], 1)
        self.assertLess(keeper["ci"][1], 1)
        self.assertLessEqual(null["ci"][0], 1)
        self.assertGreaterEqual(null["ci"][1], 1)
        self.assertEqual(suspect["verdict"], "suspect")
        self.assertEqual(keeper["verdict"], "keeper")
        self.assertIsNone(null["verdict"])
        self.assertEqual([s["expert"] for s in dom["suspects"]], [SUSPECT])
        self.assertEqual([s["expert"] for s in dom["keepers"]], [KEEPER])
        self.assertEqual(set(dom["suspects"][0]), {"layer", "expert", "lift", "ci", "within_task", "support"})
        self.assertGreater(suspect["within_task"], 0)
        self.assertLess(keeper["within_task"], 0)
        self.assertAlmostEqual(null["within_task"], 0.0, places=9)
        self.assertGreater(suspect["within_task_ci"][0], 0)
        self.assertEqual(suspect["mass_incorrect"], 120.0)  # 30 tasks * 4.0
        self.assertEqual(suspect["count_incorrect"], 120)
        self.assertAlmostEqual(suspect["mean_share_incorrect"], 0.4)
        self.assertAlmostEqual(rep["coverage"]["domains"]["math"]["mean_topk_mass_fraction"], 0.6)
        self.assertEqual(rep["method"]["eps"], J.EPS)
        self.assertEqual(rep["method"]["cluster"], "task")

    def test_task_bootstrap_does_not_let_one_hard_task_dominate(self):
        """40 incorrect draws of one task, all on the expert, must not outvote 10 balanced tasks.
        Resampling rows flags the expert; resampling tasks does not (the lower bound stays at 1)."""
        recs, rows = [], []
        for i in range(40):
            rid = "rr-hard-%d" % i
            recs.append(record(rid, [(2, 10.0, [(21, 9.0, 9)])]))
            rows.append(attempt_row("hard", False, [rid], draw=i + 1))
        for t in range(10):
            for correct, draw in ((True, 1), (False, 2)):
                for k in range(4):
                    rid = "rr-n%02d-%d-%d" % (t, draw, k)
                    recs.append(record(rid, [(2, 10.0, [(21, 2.0, 2)])]))
                    rows.append(attempt_row("n%02d" % t, correct, [rid], draw=draw * 10 + k))
        kw = dict(boot=100, seed=0, min_attempts=10)
        task = self.go(rows, recs, **kw)
        row = self.go(rows, recs, cluster="row", **kw)
        ct = task["domains"]["math"]["cells"][0]
        cr = row["domains"]["math"]["cells"][0]
        self.assertEqual(ct["expert"], 21)
        self.assertAlmostEqual(ct["lift"], cr["lift"])
        self.assertLessEqual(ct["ci"][0], 1.0)
        self.assertGreater(cr["ci"][0], 1.0)
        self.assertIsNone(ct["verdict"])
        self.assertEqual(cr["verdict"], "suspect")
        self.assertEqual(task["domains"]["math"]["suspects"], [])
        self.assertEqual(row["domains"]["math"]["suspects"][0]["expert"], 21)
        self.assertAlmostEqual(ct["within_task"], 0.0, places=9)  # the hard task has no correct draw, so it is excluded

    def test_multi_turn_sums_records(self):
        recs = [record("rr-a", [(1, 10.0, [(3, 2.0, 2)])]),
                record("rr-b", [(1, 10.0, [(3, 3.0, 3), (4, 1.0, 1)])])]
        # enough other attempts that the cell is real; the summed one is the incorrect draw of task t0
        rows = [attempt_row("t0", False, ["rr-a", "rr-b"], draw=1)]
        rep = self.go(rows, recs, boot=5, min_attempts=10)
        by = {(c["layer"], c["expert"]): c for c in rep["domains"]["math"]["cells"]}
        self.assertEqual(by[(1, 3)]["mass_incorrect"], 5.0)
        self.assertEqual(by[(1, 3)]["count_incorrect"], 5)
        self.assertAlmostEqual(by[(1, 3)]["mean_share_incorrect"], 0.25)
        self.assertEqual(by[(1, 4)]["mass_incorrect"], 1.0)
        self.assertAlmostEqual(by[(1, 4)]["mean_share_incorrect"], 0.05)
        self.assertEqual(by[(1, 3)]["support"], 1)

    def test_truncated_missing_duplicate_and_skips(self):
        recs = [record("rr-keep", [(0, 10.0, [(1, 4.0, 4)])]),
                record("rr-keep", [(0, 10.0, [(1, 100.0, 100)])]),  # duplicate, ignored
                record("rr-orphan", [(0, 10.0, [(2, 1.0, 1)])]),
                '{"id": "rr-cut", "layers": [']  # truncated last line
        rows = [attempt_row("u", False, ["rr-keep", "rr-missing"], draw=1),
                attempt_row("sing", True, ["rr-keep"], draw=1, single=True),
                {"task": "ns", "suite": "math", "arm": "base", "draw": 1, "status": "http-error", "score": None,
                 "route_records": ["rr-orphan"]},
                {"task": "nu", "suite": "math", "arm": "base", "draw": 1, "status": "ok", "score": {"correct": None},
                 "route_records": ["rr-keep"]},
                attempt_row("empty", True, [], draw=1),
                attempt_row("gone", True, ["rr-nope"], draw=1),
                "{not json"]
        rep = self.go(rows, recs, boot=5)
        cov = rep["coverage"]
        self.assertEqual(cov["truncated_lines"], 1)
        self.assertEqual(cov["duplicate_ids"], 1)
        self.assertEqual(cov["records_missing"], 2)  # rr-missing on the used row, rr-nope on the skipped row
        self.assertEqual(cov["records_unmatched"], 0)  # rr-orphan is named by the unscored row
        self.assertEqual(cov["skipped_rows"], {"bad_row": 1, "no_score": 2, "no_ids": 1, "no_record": 1})
        self.assertEqual(cov["rows_used"], 2)
        cell = rep["domains"]["math"]["cells"][0]
        self.assertEqual((cell["expert"], cell["mass_incorrect"], cell["mass_correct"]), (1, 4.0, 4.0))
        # an unmatched record is counted when nothing names it
        recs2 = recs[:1] + [record("rr-free", [(0, 10.0, [(9, 1.0, 1)])])]
        rows2 = [attempt_row("u", False, ["rr-keep"], draw=1)]
        rep2 = self.go(rows2, recs2, boot=5)
        self.assertEqual(rep2["coverage"]["records_unmatched"], 1)
        self.assertEqual(rep2["coverage"]["rows_used"], 1)

    def test_min_attempts_gates_the_verdict(self):
        recs, rows = [], []
        for i in range(3):
            rid = "rr-s-%d" % i
            recs.append(record(rid, [(4, 10.0, [(7, 9.0, 9)])]))
            rows.append(attempt_row("t%d" % i, False, [rid], draw=1))
        for i in range(3):
            rid = "rr-c-%d" % i
            recs.append(record(rid, [(4, 10.0, [(8, 1.0, 1)])]))  # the suspect is absent here: share 0
            rows.append(attempt_row("c%d" % i, True, [rid], draw=1))
        rep = self.go(rows, recs, min_attempts=10, boot=20)
        cell = [c for c in rep["domains"]["math"]["cells"] if c["expert"] == 7][0]
        self.assertEqual(cell["support"], 3)
        self.assertGreater(cell["lift"], 2)
        self.assertIsNone(cell["ci"])
        self.assertIsNone(cell["verdict"])
        self.assertEqual(rep["domains"]["math"]["suspects"], [])

    def test_domain_key(self):
        recs, rows = [], []
        for t in range(6):
            r, w = planted_pair("b%d" % t, 1, 2, arm="bad")
            recs += r
            rows += w
            # the same experts, balanced, on the other arm
            for correct, draw, mass in ((True, 1, 2.0), (False, 2, 2.0)):
                rid = "rr-g%d-%d" % (t, draw)
                recs.append(record(rid, [(LAYER, 10.0, [(SUSPECT, mass, 2)])]))
                rows.append(attempt_row("g%d" % t, correct, [rid], arm="good", draw=draw))
        rep = self.go(rows, recs, domain_key="arm", boot=20, min_attempts=10)
        self.assertEqual([s["expert"] for s in rep["domains"]["bad"]["suspects"]], [SUSPECT])
        self.assertEqual(rep["domains"]["good"]["suspects"], [])
        self.assertNotIn("math", rep["domains"])

    def test_same_seed_same_bytes_and_mask(self):
        recs, rows = [], []
        for t in range(12):
            r, w = planted_pair("task-%02d" % t, 1, 2)
            recs += r
            rows += w
        d = os.path.join(self.d, "recs")
        os.makedirs(d)
        dump(os.path.join(d, "route-20261122T000000Z-0001.jsonl"), recs)
        dump(os.path.join(d, "ignore-me.jsonl"), [record("rr-task-00-1", [(LAYER, 10.0, [(SUSPECT, 100.0, 100)])])])
        rows_path = os.path.join(self.d, "rows.jsonl")
        dump(rows_path, rows)
        out1, out2 = os.path.join(self.d, "a.json"), os.path.join(self.d, "b.json")
        mask = os.path.join(self.d, "mask.txt")
        argv = ["--rows", rows_path, "--records", d, "--boot", "30", "--seed", "0", "--min-attempts", "10"]
        self.assertEqual(J.main(argv + ["--out", out1, "--mask-out", mask]), 0)
        self.assertEqual(J.main(argv + ["--out", out2, "--mask-out", mask]), 0)
        self.assertEqual(open(out1, "rb").read(), open(out2, "rb").read())
        lines = open(mask, encoding="utf-8").read().splitlines()
        self.assertTrue(lines and all(part.split(":")[0].isdigit() and part.split(":")[1].isdigit() and part.count(":") == 1 for part in lines))
        self.assertIn("%d:%d" % (LAYER, SUSPECT), lines)
        self.assertNotIn("%d:%d" % (LAYER, KEEPER), lines)
        self.assertNotIn("%d:%d" % (LAYER, NULL), lines)
        # the decoy file is not a route-*.jsonl, so the first record's mass stands
        rep = json.loads(open(out1, encoding="utf-8").read())
        cell = [c for c in rep["domains"]["math"]["cells"] if c["expert"] == SUSPECT][0]
        self.assertEqual(cell["mass_correct"], 12 * 0.5)
        self.assertEqual(rep["coverage"]["duplicate_ids"], 0)


if __name__ == "__main__":
    unittest.main()
