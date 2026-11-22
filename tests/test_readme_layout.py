"""README.md ## Layout: every path the table names exists in the tree."""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCOPED = ("docs/", "tools/", "gate/")  # rows whose second cell names files inside that directory


def layout_paths(text):
    lay = text.split("## Layout", 1)[1].split("\n## ", 1)[0]
    paths = []
    for line in lay.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        first = re.findall(r"`([^`]+)`", cells[0])
        paths += [p for p in first if "<" not in p]
        if len(first) == 1 and first[0] in SCOPED:
            for p in re.findall(r"`([^`]+)`", cells[1]):
                paths.append(p if p.startswith(("docs/", "tools/", "gate/")) else first[0] + p)
    return paths


class ReadmeLayout(unittest.TestCase):
    def test_every_layout_path_exists(self):
        with open(os.path.join(ROOT, "README.md")) as f:
            paths = layout_paths(f.read())
        self.assertGreater(len(paths), 20)
        missing = [p for p in paths if not os.path.exists(os.path.join(ROOT, p.rstrip("/")))]
        self.assertEqual(missing, [])

    def test_names_the_newer_tools(self):
        with open(os.path.join(ROOT, "README.md")) as f:
            paths = set(layout_paths(f.read()))
        for p in ("docs/STRIX-LOOP.md", "docs/research/", "research/", "tools/flerp-research", "tools/strix-loop.sh",
                  "gate/strix_loop.py"):
            self.assertIn(p, paths)


if __name__ == "__main__":
    unittest.main()
