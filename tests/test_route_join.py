"""tools/route_join: join eval rows to route records and rank experts whose gate mass goes with wrong answers
(suspects) or right answers (keepers). Synthetic records only; no server.

Proofs: a planted suspect and a planted keeper have lift intervals excluding 1 and a null expert does not; a
task-level bootstrap does not let one hard task with many draws dominate the way a row-level bootstrap does;
a multi-turn attempt sums its records; a truncated last line, a missing record and a duplicate id are counted
and are not fatal; a cell under --min-attempts gets no verdict; the same seed writes the same bytes; --mask-out
is layer:expert lines and a directory only reads route-*.jsonl.

Multiple comparisons (card 010ab353c3): on pure-noise records (labels independent of routing) no cell reaches
q < 0.05 in at least 19 of 20 seeds, under both the permutation FDR and Benjamini-Hochberg, while the uncorrected
CI verdict flags noise cells; a planted effect is recovered, is the only mask line, and the mask header records
method, threshold, permutations and seed; the same seed gives the same bytes; the numpy and pure-python paths
agree; a hard task cannot move the permutation null; BH q-values match a hand calculation.
"""
import json
import os
import random
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
        self.assertEqual(set(dom["suspects"][0]), {"layer", "expert", "lift", "ci", "within_task", "support", "p", "q", "q_perm", "q_bh"})
        self.assertLess(dom["suspects"][0]["q"], 0.05)
        self.assertEqual(dom["suspects"][0]["q"], dom["suspects"][0]["q_perm"])
        self.assertGreater(null["q_perm"], 0.05)
        self.assertGreater(null["q_bh"], 0.05)
        self.assertEqual(suspect["fdr_verdict"], "suspect")
        self.assertEqual(keeper["fdr_verdict"], "keeper")
        self.assertIsNone(null["fdr_verdict"])
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
        with open(out1, "rb") as f1, open(out2, "rb") as f2:
            self.assertEqual(f1.read(), f2.read())
        with open(mask, encoding="utf-8") as f:
            raw = f.read().splitlines()
        self.assertTrue(raw[0].startswith("#"))
        for word in ("q < 0.05", "permutation", "permutations=1000", "seed=0"):
            self.assertIn(word, raw[0].replace("perm-fdr", "permutation"))
        lines = [x for x in raw if not x.startswith("#")]
        self.assertTrue(lines and all(part.split(":")[0].isdigit() and part.split(":")[1].isdigit() and part.count(":") == 1 for part in lines))
        self.assertIn("%d:%d" % (LAYER, SUSPECT), lines)
        self.assertNotIn("%d:%d" % (LAYER, KEEPER), lines)
        self.assertNotIn("%d:%d" % (LAYER, NULL), lines)
        # the decoy file is not a route-*.jsonl, so the first record's mass stands
        with open(out1, encoding="utf-8") as f:
            rep = json.load(f)
        cell = [c for c in rep["domains"]["math"]["cells"] if c["expert"] == SUSPECT][0]
        self.assertEqual(cell["mass_correct"], 12 * 0.5)
        self.assertEqual(rep["coverage"]["duplicate_ids"], 0)


def noise_data(seed, tasks=40, draws=3, arms=("a", "b"), layers=2, experts=12, plant=None):
    """Attempts whose correct/incorrect label is independent of routing. plant=(layer, expert): that expert is
    always in an incorrect attempt's top list with a big share and only sometimes, and small, in a correct one."""
    rng = random.Random(seed)
    recs, rows = [], []
    for arm in arms:
        for t in range(tasks):
            for d in range(draws):
                correct = rng.random() < 0.5
                rid = "rr-%s-%d-%d" % (arm, t, d)
                lay = []
                for l in range(layers):
                    top = {e: rng.uniform(0.5, 3.0) for e in rng.sample(range(experts), 4)}
                    if plant and plant[0] == l:
                        if not correct:
                            top[plant[1]] = 8.0 + rng.random()
                        elif plant[1] in top:
                            top[plant[1]] = 0.5
                    lay.append((l, 20.0, [(e, m, 1) for e, m in sorted(top.items())]))
                recs.append(record(rid, lay))
                rows.append(attempt_row("t%02d" % t, correct, [rid], arm=arm, draw=d + 1))
    return recs, rows


class Fdr(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="route-join-fdr-")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def go(self, rows, recs, **kw):
        rp, cp = os.path.join(self.d, "rows.jsonl"), os.path.join(self.d, "recs.jsonl")
        dump(rp, rows)
        dump(cp, recs)
        kw.setdefault("boot", 50)
        return J.join([rp], [cp], **kw)

    def test_pure_noise_has_no_cell_under_q_05_in_19_of_20_seeds(self):
        clean_perm = clean_bh = tested = ci_flagged = 0
        for seed in range(20):
            recs, rows = noise_data(seed)
            rep = self.go(rows, recs, seed=seed, perms=1000)
            cells = rep["domains"]["math"]["cells"]
            tested += sum(c["q_perm"] is not None for c in cells)
            ci_flagged += sum(c["verdict"] is not None for c in cells)
            clean_perm += all(c["q_perm"] is None or c["q_perm"] >= 0.05 for c in cells)
            clean_bh += all(c["q_bh"] is None or c["q_bh"] >= 0.05 for c in cells)
            if all(c["q_perm"] is None or c["q_perm"] >= 0.05 for c in cells):
                self.assertEqual(rep["domains"]["math"]["mask"], [])
        self.assertGreater(tested, 20 * 20)  # most of the 24 cells per run were really tested
        self.assertGreaterEqual(clean_perm, 19)
        self.assertGreaterEqual(clean_bh, 19)
        self.assertGreater(ci_flagged, 0)  # the uncorrected CI verdict does flag noise, which is why it no longer gates

    def test_planted_effect_is_recovered_and_alone_in_the_mask(self):
        recs, rows = noise_data(7, plant=(1, 5))
        mask = os.path.join(self.d, "mask.txt")
        rp, cp, out = (os.path.join(self.d, n) for n in ("rows.jsonl", "recs.jsonl", "out.json"))
        dump(rp, rows)
        dump(cp, recs)
        self.assertEqual(J.main(["--rows", rp, "--records", cp, "--out", out, "--mask-out", mask, "--boot", "20", "--seed", "3", "--perms", "1000"]), 0)
        with open(out, encoding="utf-8") as f:
            rep = json.load(f)
        dom = rep["domains"]["math"]
        self.assertEqual([(s["layer"], s["expert"]) for s in dom["suspects"]], [(1, 5)])
        top = dom["suspects"][0]
        self.assertLess(top["q_perm"], 0.05)
        self.assertLess(top["q_bh"], 0.05)
        self.assertGreater(top["lift"], 3)
        self.assertEqual(dom["keepers"], [])
        with open(mask, encoding="utf-8") as f:
            raw = f.read().splitlines()
        self.assertEqual([x for x in raw if not x.startswith("#")], ["1:5"])
        self.assertEqual(rep["method"]["fdr"]["alpha"], 0.05)
        self.assertEqual(rep["method"]["fdr"]["perms"], 1000)
        self.assertEqual(rep["method"]["fdr"]["seed"], 3)
        self.assertIn("seed=3", raw[0])
        self.assertIn("permutations=1000", raw[0])

    def test_seeded_and_numpy_matches_pure_python(self):
        recs, rows = noise_data(11, tasks=12, draws=2, plant=(0, 2))
        a = self.go(rows, recs, seed=4, perms=200)
        b = self.go(rows, recs, seed=4, perms=200)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        c = self.go(rows, recs, seed=5, perms=200)
        self.assertNotEqual([x["p_perm"] for x in a["domains"]["math"]["cells"]], [x["p_perm"] for x in c["domains"]["math"]["cells"]])
        if J.np is not None:
            saved, J.np = J.np, None
            try:
                py = self.go(rows, recs, seed=4, perms=200)
            finally:
                J.np = saved
            for x, y in zip(a["domains"]["math"]["cells"], py["domains"]["math"]["cells"]):
                for k in ("p_perm", "q_perm", "q_bh", "fdr_verdict"):
                    if isinstance(x[k], float):
                        self.assertAlmostEqual(x[k], y[k], places=9)
                    else:
                        self.assertEqual(x[k], y[k])

    def test_permutation_keeps_task_structure(self):
        """A task whose draws are all incorrect stays all incorrect under the null, so it cannot create a hit."""
        recs, rows = [], []
        for i in range(30):
            rid = "rr-hard-%d" % i
            recs.append(record(rid, [(2, 10.0, [(21, 9.0, 9)])]))
            rows.append(attempt_row("hard", False, [rid], draw=i + 1))
        for t in range(10):
            for k in range(4):
                for correct in (True, False):
                    rid = "rr-n%02d-%d-%d" % (t, k, correct)
                    recs.append(record(rid, [(2, 10.0, [(21, 2.0, 2)])]))
                    rows.append(attempt_row("n%02d" % t, correct, [rid], draw=k * 2 + correct))
        rep = self.go(rows, recs, perms=200)
        cell = rep["domains"]["math"]["cells"][0]
        self.assertGreater(cell["lift"], 1.5)  # the raw lift is large
        self.assertEqual(cell["p_perm"], 1.0)  # but no permutation can tell: the structure fixes it
        self.assertEqual(rep["domains"]["math"]["suspects"], [])
        # row clustering gives up the task structure and the same data is flagged
        row = self.go(rows, recs, perms=200, cluster="row")
        self.assertEqual([s["expert"] for s in row["domains"]["math"]["suspects"]], [21])

    def test_perms_zero_and_bad_fdr(self):
        recs, rows = noise_data(1, tasks=6, draws=2, plant=(0, 1))
        rep = self.go(rows, recs, perms=0)
        self.assertTrue(all(c["q_perm"] is None and c["fdr_verdict"] is None for c in rep["domains"]["math"]["cells"]))
        self.assertEqual(rep["domains"]["math"]["mask"], [])
        with self.assertRaises(ValueError):
            self.go(rows, recs, fdr="nope")

    def test_bh_matches_hand_calculation(self):
        # p sorted: .01 .02 .03 .04 .5 with m = 5 -> raw .05 .05 .05 .05 .5, so every q is its running minimum
        q = J.bh_qvalues([0.5, 0.01, 0.04, 0.02, 0.03])
        for got, want in zip(q, [0.5, 0.05, 0.05, 0.05, 0.05]):
            self.assertAlmostEqual(got, want)
        self.assertEqual(J.bh_qvalues([]), [])
        self.assertEqual(J.bh_qvalues([0.9]), [0.9])

    def test_fdr_bh_selects_the_gating_q(self):
        recs, rows = noise_data(7, plant=(1, 5))
        rep = self.go(rows, recs, perms=1000, fdr="bh")
        s = rep["domains"]["math"]["suspects"][0]
        self.assertEqual(s["q"], s["q_bh"])
        self.assertEqual(rep["method"]["fdr"]["primary"], "bh")


if __name__ == "__main__":
    unittest.main()
