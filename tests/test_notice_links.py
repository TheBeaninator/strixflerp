"""Every licence text a catalog notices file points at ships in the tree (card 412a84208f).

catalog/*/THIRD_PARTY_NOTICES.md and catalog/*/THIRD_PARTY.md are verbatim copies of an origin's notices, so their
relative paths are origin-relative and resolve against the origin's catalog directory, where the harvest copies the
texts as [licence] files. Checked: every relative Markdown link, and every backtick-quoted path that names a licence
text (a LICENSE / COPYING / PATENTS / NOTICE file, or a file or directory under LICENSES/ or licenses/)."""
import glob
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTICES = ("THIRD_PARTY_NOTICES.md", "THIRD_PARTY.md")
LINK = re.compile(r"\]\(([^()\s]+)\)")
TICK = re.compile(r"`([^`\s]+)`")
LICENCE_NAME = re.compile(r"(^|/)(LICEN[CS]E|COPYING|PATENTS|NOTICE)([._-][^/]*)?$")
LICENCE_DIR = re.compile(r"^(LICENSES|licenses)/")              # origin-relative only: share/licenses/... is an install path


def notices_files(root=ROOT):
    return sorted(p for n in NOTICES for p in glob.glob(os.path.join(root, "catalog", "*", n)))


def targets(text):
    """Origin-relative paths a notices text points at: relative links, and backtick-quoted licence paths."""
    out = []
    for m in LINK.finditer(text):
        t = m.group(1).split("#", 1)[0]
        if t and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", t) and not t.startswith("/"):
            out.append(t)
    for m in TICK.finditer(text):
        t = m.group(1)
        if "/" in t and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", t) and not t.startswith("/") \
                and (LICENCE_NAME.search(t.rstrip("/")) or LICENCE_DIR.search(t)):
            out.append(t)
    return out


def missing(root=ROOT):
    """[(notices file relative to root, target)] for every target with no file or directory in the tree."""
    out = []
    for f in notices_files(root):
        d = os.path.dirname(f)
        with open(f, encoding="utf-8") as fh:
            text = fh.read()
        for t in targets(text):
            if not os.path.exists(os.path.normpath(os.path.join(d, t))):
                out.append((os.path.relpath(f, root), t))
    return out


class NoticeLinks(unittest.TestCase):
    def test_catalog_notices_resolve(self):
        self.assertTrue(notices_files(), "no catalog notices files found")
        bad = missing()
        self.assertEqual(bad, [], "notices point at files the tree does not carry:\n" +
                         "\n".join(f"  {f}: {t}" for f, t in bad))

    def test_a_missing_target_fails(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, "catalog", "demo")
            os.makedirs(os.path.join(d, "LICENSES"))
            open(os.path.join(d, "LICENSES", "MIT.txt"), "w").close()
            with open(os.path.join(d, "THIRD_PARTY_NOTICES.md"), "w") as fh:
                fh.write("[MIT](LICENSES/MIT.txt), [gone](LICENSES/Apache-2.0.txt), "
                         "`third_party/x/LICENSE`, `src/x.h`, [web](https://example.org/LICENSE), [top](#top)\n")
            self.assertEqual(missing(tmp), [("catalog/demo/THIRD_PARTY_NOTICES.md", "LICENSES/Apache-2.0.txt"),
                                            ("catalog/demo/THIRD_PARTY_NOTICES.md", "third_party/x/LICENSE")])


if __name__ == "__main__":
    unittest.main()
