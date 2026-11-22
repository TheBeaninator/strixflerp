"""tools/flerp-research: research cards, module PRs, LOG and amendments, merge plans, batches, scrub (docs/research/README.md)."""
import contextlib, glob, hashlib, importlib.machinery, importlib.util, io, json, os, re, shutil, subprocess, tempfile, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_loader = importlib.machinery.SourceFileLoader("flerp_research", os.path.join(ROOT, "tools", "flerp-research"))
_spec = importlib.util.spec_from_loader("flerp_research", _loader)
fr = importlib.util.module_from_spec(_spec)
_loader.exec_module(fr)
EXAMPLE = glob.glob(os.path.join(ROOT, "research", "module", "*-moe-combine-vec4"))[0]
GOOD = fr.slurp(os.path.join(EXAMPLE, "card.md"))
MOD = "catalog/otherproj/kernel/fastk"


def put(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def rules(text):
    return {r for r, _, _ in fr.lint_card(fr.parse(text))}


def swap(text, key, value):
    sub = key in fr.METHOD_KEYS or key in fr.MODULE_KEYS
    return re.sub(r"^%s%s: .*$" % ("  " if sub else "", re.escape(key)), "%s%s: %s" % ("  " if sub else "", key, value), text, count=1, flags=re.M)


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            rc = fr.main(list(argv))
        except SystemExit as e:
            rc = e.code
    return rc, out.getvalue()


def module_card(series="otherproj-kernels"):
    """A card from another project's author who brings their kernel as a strixflerp module."""
    c = swap(swap(GOOD, "AREA", "kernel"), "code", "this PR")
    c = swap(c, "subject", MOD + " against the catalogue's combine")
    c = swap(swap(c, "path", MOD), "tests", "tests/test_fastk.py")
    c = swap(c, "APPLIES TO", "otherproj: which combine kernel otherproj ships on gfx1151")
    return swap(c, "SERIES", series)


class Fixture(unittest.TestCase):
    """A scratch repository root with toolchains.toml and a ported module from another project."""
    def setUp(self):
        self.root = tempfile.mkdtemp()
        shutil.copy(os.path.join(ROOT, "toolchains.toml"), self.root)
        put(os.path.join(self.root, "catalog/otherproj/ORIGIN.toml"), 'repo = "https://example.org/otherproj"\ncommit = "abc1234"\n')
        put(os.path.join(self.root, MOD, "module.toml"), '\n'.join([
            'id = "kernel.otherproj.fastk"', 'kind = "kernel"',
            '[provenance]', 'origin = "otherproj"', 'repo = "https://example.org/otherproj"', 'commit = "abc1234"', 'authors = ["someone"]',
            '[licence]', 'spdx = "Apache-2.0"', '[build]', 'lang = "hip-c++"', 'sources = ["src/fastk.hip"]', 'toolchain = "therock-10.1"', '']))
        put(os.path.join(self.root, MOD, "src/fastk.hip"), "// a kernel\n")
        put(os.path.join(self.root, "tests/test_fastk.py"), "import unittest\n")
        self.draft = os.path.join(self.root, "research", "kernel", "fastk-combine")
        put(os.path.join(self.draft, "card.md"), module_card())

    def tearDown(self):
        shutil.rmtree(self.root)

    def fails(self, d):
        return {r for r, _, _ in fr.lint_folder(d)}

    def gate(self, d):
        shutil.copy(os.path.join(EXAMPLE, "gate.toml"), d)


class Card(unittest.TestCase):
    def test_example_card_is_clean(self):
        self.assertEqual(fr.lint_card(fr.parse(GOOD)), [])

    def test_example_id_is_its_hash(self):
        self.assertEqual(os.path.basename(EXAMPLE)[:10], hashlib.sha256(GOOD.encode()).hexdigest()[:10])

    def test_template_is_refused_until_filled(self):
        self.assertIn("Q0", rules(fr.TEMPLATE % "module"))

    def test_prediction_needs_numbers(self):
        self.assertIn("Q5", rules(swap(GOOD, "PREDICTION", "the combine gets faster")))
        self.assertIn("Q5", rules(swap(GOOD, "PREDICTION", "about 115 ms; confirmed at 150 ms or less")))  # no refute number
        self.assertIn("Q4", rules(swap(GOOD, "MECHANISM", "it is probably slow because it looks old")))

    def test_commits_not_branches(self):
        self.assertIn("Q6", rules(swap(GOOD, "code", "strixflerp@main")))
        self.assertIn("Q6", rules(swap(GOOD, "code", "otherrepo@ad7270763b753f09a811e7ae75781d9963c3e412")))
        self.assertIn("Q6", rules(swap(GOOD, "boxes", "3")))

    def test_network_secrets_and_resource_plan(self):
        self.assertIn("Q14", rules(swap(GOOD, "GIVES", "a script that does pip install of our tokenizer")))
        self.assertIn("Q14", rules(swap(GOOD, "model", "https://huggingface.co/someone/model")))
        self.assertIn("Q14", rules(swap(GOOD, "GIVES", "the patch and our api key")))
        self.assertIn("Q15", rules(swap(GOOD, "NOTE", "the run plan\nRESOURCES: four boxes")))
        self.assertIn("Q15", rules(swap(GOOD, "NOTE", "the run plan\n  hold: 2 h on four boxes")))

    def test_ordinary_words_are_not_a_resource_plan(self):
        note = "the TTL of a cache line, a staging buffer written in place, a place to stand and a hold on the lock"
        self.assertEqual(rules(swap(GOOD, "NOTE", note)), set())

    def test_unbounded_and_narrative(self):
        self.assertIn("Q13", rules(swap(GOOD, "TITLE", "Prefill speed for every setting of the MoE executor")))
        self.assertIn("Q10", rules(swap(GOOD, "REPS", "50, because more is better")))
        self.assertIn("Q10", rules(swap(GOOD, "REPS", "3")))
        self.assertIn("Q12", rules(swap(GOOD, "GIVES", "the commit and then a second patch")))

    def test_area_and_exactness_rules(self):
        self.assertIn("Q7", rules(swap(swap(GOOD, "AREA", "transport"), "boxes", "1")))
        self.assertIn("Q9", rules(swap(swap(GOOD, "AREA", "fidelity"), "EXACTNESS", "n/a")))
        self.assertIn("Q9", rules(swap(GOOD, "EXACTNESS", "close enough")))
        self.assertIn("Q2", rules(swap(GOOD, "SERIES", "Not A Tag!")))

    def test_module_pr_needs_module_section(self):
        c = re.sub(r"^MODULE:\n(  .*\n)+", "", module_card(), flags=re.M)
        self.assertIn("Q7", rules(c))


def limit(text, name, line):
    """Replace the LIMITS line that starts with name (None: drop it)."""
    return re.sub(r"^  %s = .*\n" % re.escape(name), "" if line is None else "  %s\n" % line, text, count=1, flags=re.M)


class Limits(unittest.TestCase):
    """LIMITS (owner 2026-11-20): every limit on the card with source, why and changes-outcome; outcome-changing limits real or
    justified; no derived caps; a hang guard; the verdict gives each limit's hit count (card-intake R26-R29)."""
    def test_limits_are_required(self):
        self.assertIn("Q0", rules(re.sub(r"^LIMITS:\n(  .*\n)+", "", GOOD, flags=re.M)))

    def test_line_needs_source_why_and_changes_outcome(self):
        F = fr.lint_limits("context = 16,640 tokens")
        self.assertEqual([r for r, _, _ in F], ["Q16"])
        for part in ("source:", "why:", "changes-outcome"):
            self.assertIn(part, F[0][1])
        self.assertIn("Q16", rules(limit(GOOD, "context", "context = 16,640 tokens; source: guessed; why: holds the prompt; changes-outcome: no")))
        self.assertIn("Q16", rules(limit(GOOD, "context", "context = 16,640 tokens; source: chosen; why: holds the prompt; changes-outcome: maybe")))

    def test_outcome_changing_chosen_limit_needs_justification(self):
        line = "speed bar = 1.05x baseline; source: chosen; why: smaller gains sit inside the spread; changes-outcome: yes"
        self.assertIn("Q17", rules(limit(GOOD, "speed bar", line)))
        self.assertNotIn("Q17", rules(limit(GOOD, "speed bar", line + "; justified: the mechanism predicts 1.11x, twice the bar")))
        # a real limit needs no justification: the model's own context window
        real = "context = 262,144 tokens; source: model-limit; why: the model's own window; changes-outcome: yes"
        self.assertEqual(rules(limit(GOOD, "context", real)), set())

    def test_derived_caps_and_hang_guard(self):
        derived = "max_tokens per reply = 2x p99 of a pilot, 3,680; source: measured; why: replies end well before it; changes-outcome: no"
        self.assertIn("Q18", rules(limit(GOOD, "context", derived)))
        self.assertNotIn("Q18", rules(limit(GOOD, "context", derived + "; justified: the pilot used the same prompts and the cap was never reached")))
        self.assertIn("Q18", rules(limit(GOOD, "hang guard", None)))  # generated tokens is a cap: something must catch a hang
        unguarded = "hang guard = process timeout 3,000 s; source: chosen; why: a hang ends the process; changes-outcome: no"
        self.assertIn("Q18", rules(limit(GOOD, "hang guard", unguarded)))  # it must say a hang is a harness fault

    def test_template_carries_limits(self):
        self.assertIn("LIMITS:", fr.TEMPLATE)
        self.assertIn("Q0", rules(fr.TEMPLATE % "module"))

    def test_render_copies_the_limits(self):
        out = fr.render(fr.parse(GOOD), EXAMPLE)
        lim = re.search(r"^LIMITS:\n((?:  .*\n)+)", out, re.M).group(1).splitlines()
        self.assertEqual(len(lim), len(fr.limits_parse(fr.parse(GOOD)["LIMITS"])))
        self.assertTrue(all("; shown: " in l for l in lim))
        self.assertIn("limits in force and how often each was hit", out)

    def test_experiments_template_limits_table(self):
        text = fr.slurp(os.path.join(ROOT, "docs", "EXPERIMENTS.md"))
        t = fr.tomllib.loads(re.search(r"```toml\n(.*?)```", text, re.S).group(1))
        self.assertTrue(t.get("limits"))
        for name, v in t["limits"].items():
            self.assertTrue({"value", "source", "why", "changes_outcome"} <= set(v), name)
            self.assertIn(v["source"], fr.LIM_SOURCES, name)
            if v["changes_outcome"] and v["source"] not in fr.LIM_REAL:
                self.assertTrue(v.get("justified") and v.get("shown"), name)
        self.assertTrue(any("harness fault" in v["why"] for v in t["limits"].values()))


class ModulePR(Fixture):
    def test_ported_module_card_lints(self):
        self.assertEqual(fr.lint_folder(self.draft), [])

    def test_module_checks(self):
        mt = os.path.join(self.root, MOD, "module.toml")
        good = fr.slurp(mt)
        for bad, rule in ((good.replace('"Apache-2.0"', '"LicenseRef-proprietary"'), "M2"),
                          (good.replace('authors = ["someone"]\n', ""), "M2"),
                          (good.replace('"therock-10.1"', '"my-own-clang"'), "M3")):
            put(mt, bad)
            self.assertIn(rule, self.fails(self.draft))
        put(mt, good)
        put(os.path.join(self.root, MOD, "src/CMakeLists.txt"), "FetchContent_Declare(x URL https://example.org/x.tgz)\n")
        self.assertIn("M5", self.fails(self.draft))
        os.remove(os.path.join(self.root, "tests/test_fastk.py"))
        self.assertIn("M4", self.fails(self.draft))

    def test_submit_amend_and_wont_do(self):
        rc, out = run("submit", self.draft)
        self.assertEqual(rc, 0, out)
        sd = out.strip().splitlines()[-1]
        cid = os.path.basename(sd)[:10]
        self.assertEqual(cid, fr.h10(os.path.join(sd, "card.md")))
        self.assertNotEqual(run("log", sd, "run", "--by", "owner")[0], 0)  # not accepted yet
        self.assertNotEqual(run("log", sd, "accepted", "--by", "owner")[0], 0)  # no gate.toml yet
        self.gate(sd)
        self.assertEqual(run("log", sd, "accepted", "--by", "owner")[0], 0)
        # the owner finds a wrong premise: a recorded amendment, not tampering
        put(os.path.join(sd, "card.md"), fr.slurp(os.path.join(sd, "card.md")).replace("boxes: 1", "boxes: 2"))
        self.assertIn("F2", self.fails(sd))
        self.assertNotEqual(run("log", sd, "amended", "--by", "owner")[0], 0)  # needs a reason
        self.assertEqual(run("log", sd, "amended", "--by", "owner", "--reason", "needs the pair; boxes 1 -> 2")[0], 0)
        self.assertEqual(self.fails(sd), set())
        self.assertEqual(os.path.basename(sd)[:10], cid)
        put(os.path.join(sd, "gate.toml"), fr.slurp(os.path.join(sd, "gate.toml")).replace("value = 1.05", "value = 1.0"))
        self.assertIn("F3", self.fails(sd))  # a loosened gate without an amendment
        self.assertEqual(run("log", sd, "amended", "--by", "owner", "--reason", "speed bar 1.05 -> 1.0: the baseline was re-measured")[0], 0)
        self.assertEqual(run("log", sd, "wont-do", "--by", "owner", "--reason", "the module needs a toolchain we do not carry")[0], 0)
        self.assertEqual(self.fails(sd), set())
        merge, drop, why = fr.merge_plan(sd, self.root)
        self.assertEqual(merge, [])
        self.assertIn(MOD, drop)


class Verdicts(Fixture):
    def place_example(self, verdict):
        d = os.path.join(self.root, "research", "module", os.path.basename(EXAMPLE))
        shutil.copytree(EXAMPLE, d)
        if verdict != "CONFIRMED":
            put(os.path.join(d, "verdict.md"), fr.slurp(os.path.join(d, "verdict.md")).replace("VERDICT: CONFIRMED", "VERDICT: " + verdict))
            put(os.path.join(d, "LOG"), fr.slurp(os.path.join(d, "LOG")).replace("verdict=CONFIRMED", "verdict=" + verdict))
        return d

    def test_example_folder_is_clean(self):
        self.assertEqual(fr.lint_folder(EXAMPLE), [])
        ok, rows = fr.judge(EXAMPLE)
        self.assertTrue(ok)
        self.assertEqual(len(rows), 6)

    def test_confirmed_merges_folder_and_module(self):
        d = self.place_example("CONFIRMED")
        merge, drop, _ = fr.merge_plan(d, self.root)
        self.assertEqual(merge, [os.path.relpath(d, self.root), "catalog/flerp/executor/qwen35moe"])
        self.assertEqual(drop, [])

    def test_refuted_merges_folder_only(self):
        for v in ("REFUTED", "INCONCLUSIVE"):
            d = self.place_example(v)
            merge, drop, _ = fr.merge_plan(d, self.root)
            self.assertEqual(merge, [os.path.relpath(d, self.root)])
            self.assertEqual(drop, ["catalog/flerp/executor/qwen35moe"])
            shutil.rmtree(d)

    def test_open_pr_merges_nothing(self):
        d = self.place_example("CONFIRMED")
        put(os.path.join(d, "LOG"), "\n".join(fr.slurp(os.path.join(d, "LOG")).splitlines()[:3]) + "\n")
        self.assertEqual(fr.merge_plan(d, self.root)[:2], ([], []))

    def test_confirmed_needs_a_passing_gate(self):
        d = self.place_example("CONFIRMED")
        put(os.path.join(d, "verdict.md"), fr.slurp(os.path.join(d, "verdict.md")).replace("GATE: PASS", "GATE: FAIL"))
        self.assertIn("F4", self.fails(d))

    def test_verdict_gives_each_limit_and_its_hits(self):
        d = self.place_example("CONFIRMED")
        vp = os.path.join(d, "verdict.md"); good = fr.slurp(vp)
        put(vp, re.sub(r"^## Limits\n.*?(?=^## )", "", good, flags=re.M | re.S))
        self.assertIn("F6", self.fails(d))
        put(vp, re.sub(r"^\| hang guard \|.*\n", "", good, flags=re.M))
        self.assertIn("F6", self.fails(d))  # a limit with no row
        put(vp, re.sub(r"^(\| hang guard \|[^|]*\|) [^|]*\|", r"\1 never |", good, flags=re.M))
        self.assertIn("F6", self.fails(d))  # the hit count must be a number
        put(vp, good)
        self.assertEqual({r for r in self.fails(d) if not r.startswith("M")}, set())

    def test_verdict_step_refuses_a_verdict_without_limits(self):
        d = self.place_example("CONFIRMED")
        put(os.path.join(d, "LOG"), "\n".join(l for l in fr.slurp(os.path.join(d, "LOG")).splitlines() if " verdict " not in l) + "\n")
        vp = os.path.join(d, "verdict.md")
        put(vp, re.sub(r"^## Limits\n.*?(?=^## )", "", fr.slurp(vp), flags=re.M | re.S))
        rc, out = run("log", d, "verdict", "--by", "owner")
        self.assertIn("Limits", str(rc))

    def test_results_and_scrub(self):
        d = self.place_example("CONFIRMED")
        put(os.path.join(d, "results", "cause.md"), "edited\n")
        self.assertIn("F4", self.fails(d))
        put(os.path.join(d, "results", "extra.txt"), "served at %s\n" % ".".join(["10", "1", "2", "3"]))
        self.assertIn("F5", self.fails(d))
        # a private host-name family comes from a deny-list regex line, not from the tool
        put(os.path.join(d, "results", "extra.txt"), "served by rackhost-q\n")
        self.assertNotIn("F5", self.fails(d))
        self.assertIn("F5", {r for r, _, _ in fr.lint_folder(d, ["re:(?i)\\brackhost-[a-z]\\w*\\b"])})
        self.assertEqual(fr.scrub_text("served by RackHost-q", {}, [], ["re:(?i)\\brackhost-[a-z]\\w*\\b"]), "served by <private>")

    def test_private_names_come_from_the_deny_list(self):
        d = self.place_example("CONFIRMED")
        deny = os.path.join(self.root, "deny.txt"); put(deny, "# private names\nexamplehost\n")
        put(os.path.join(d, "results", "extra.txt"), "served by ExampleHost\n")
        self.assertNotIn("F5", {r for r, _, _ in fr.lint_folder(d, ())})
        old = os.environ.get("FLERP_RESEARCH_DENY"); os.environ["FLERP_RESEARCH_DENY"] = deny
        try:
            self.assertEqual(fr.load_deny(), ["examplehost"])
            self.assertIn("F5", {r for r, _, _ in fr.lint_folder(d, fr.load_deny())})
        finally:
            if old is None: os.environ.pop("FLERP_RESEARCH_DENY")
            else: os.environ["FLERP_RESEARCH_DENY"] = old
        run_dir = os.path.join(self.root, "run"); put(os.path.join(run_dir, "r.log"), "examplehost: 10 tok/s\n")
        self.assertEqual(run("ingest", run_dir, d, "--files", "r.log", "--deny", deny)[0], 0)
        self.assertEqual(fr.slurp(os.path.join(d, "results", "r.log")), "<private>: 10 tok/s\n")

    def test_deny_list_regex_lines(self):
        deny = ["re:\\bTOOLX\\b"]
        self.assertEqual(fr.scrub_findings("a run in place", deny), [])
        self.assertEqual(fr.scrub_findings("staged by TOOLX", deny), [("deny list", "TOOLX")])
        self.assertEqual(fr.scrub_text("staged by TOOLX in place", {}, [], deny), "staged by <private> in place")
        card = os.path.join(self.root, "c.md"); put(card, swap(GOOD, "NOTE", "staged by TOOLX"))
        dl = os.path.join(self.root, "deny.txt"); put(dl, "re:\\bTOOLX\\b\n")
        self.assertEqual(run("lint", card, "--deny", dl)[0], 1)
        put(card, GOOD)
        self.assertEqual(run("lint", card, "--deny", dl)[0], 0)

    def test_merged_records_the_commit(self):
        d = self.place_example("CONFIRMED")
        self.assertNotEqual(run("log", d, "merged", "--by", "owner")[0], 0)
        self.assertEqual(run("log", d, "merged", "--by", "owner", "--commit", "0123456789abcdef")[0], 0)
        self.assertEqual({r for r in self.fails(d) if not r.startswith("M")}, set())  # the scratch root has no copy of the module
        self.assertIn("0123456789ab", fr.build_index(os.path.join(self.root, "research")))

    def test_index_links_verdict_only_when_written(self):
        a = run("submit", self.draft)[1].strip().splitlines()[-1]
        d = self.place_example("CONFIRMED")
        idx = fr.build_index(os.path.join(self.root, "research"))
        self.assertIn("(%s/card.md)" % os.path.relpath(a, os.path.join(self.root, "research")), idx)
        self.assertIn("(%s/verdict.md)" % os.path.relpath(d, os.path.join(self.root, "research")), idx)
        self.assertEqual(idx.count("/verdict.md)"), 1)

    def test_batch_by_series(self):
        a = run("submit", self.draft)[1].strip().splitlines()[-1]
        second = os.path.join(self.root, "research", "kernel", "fastk-decode")
        put(os.path.join(second, "card.md"), module_card().replace("prefill tok/s", "decode tok/s"))
        b = run("submit", second)[1].strip().splitlines()[-1]
        self.assertNotEqual(os.path.basename(a)[:10], os.path.basename(b)[:10])
        self.assertEqual(fr.series_dirs(os.path.join(self.root, "research"), "otherproj-kernels"), sorted([a, b]))
        rc, out = run("merge-plan", "--series", "otherproj-kernels", "--root", os.path.join(self.root, "research"))
        self.assertEqual(rc, 0)
        self.assertEqual(out.count("no verdict yet"), 2)


class Repo(unittest.TestCase):
    def test_outward_docs_are_clean(self):
        for p in ("docs/research", "research"):
            self.assertEqual(fr.scrub_check([os.path.join(ROOT, p)], fr.load_deny()), [], p)

    def test_index_is_current(self):
        self.assertEqual(fr.slurp(os.path.join(ROOT, "research", "INDEX.md")), fr.build_index(os.path.join(ROOT, "research")))

    @unittest.skipUnless(os.path.isdir(os.path.join(ROOT, ".git")) or os.path.isfile(os.path.join(ROOT, ".git")), "needs a git checkout")
    def test_example_module_passes_in_a_clean_checkout(self):
        self.assertEqual(fr.clean_check(ROOT, fr.parse(GOOD)), [])


if __name__ == "__main__":
    unittest.main()
