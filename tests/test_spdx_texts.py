"""Every licence an origin's sources name by SPDX identifier has its text in that origin's catalog tree (card 35c55badef).

For each catalog/<origin>, the SPDX-License-Identifier lines of its harvested sources (catalog/<origin>/<kind>/<name>/src/**)
name licences; the texts the origin carries are the licence files anywhere under catalog/<origin> (LICENSE*, COPYING*,
NOTICE*, THIRD_PARTY*, *_LICENSE / LICENSE.* variants, files named after a licence such as GPL-2.0.txt, files under a
LICENSES/ or licenses/ directory; not source files), recognised by
their wording. An id is covered when a carried text has that licence's wording; version suffixes (-only, -or-later, +)
fold onto the base licence, an OR expression needs one of its ids, exceptions after WITH (Linux-syscall-note, ...)
and LicenseRef-* ids need no text."""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPDX_LINE = re.compile(rb"SPDX-License-Identifier:\s*([A-Za-z0-9.+()\- ]+)")
TEXT_NAME = re.compile(r"(^|[._-])(LICEN[CS]E|COPYING|NOTICE|UNLICENSE)([._-].*)?$|^THIRD[_-]PARTY"
                       r"|^(A|L)?GPL-[0-9]|^(MIT|ISC|Zlib|Unlicense)(\.txt)?$|^(Apache|BSD|MPL|BSL)-", re.I)
CODE_EXT = (".py", ".c", ".h", ".cc", ".cpp", ".hpp", ".cu", ".cuh", ".hip", ".zig", ".rs", ".js", ".ts", ".go", ".sh")
TEXT_DIR = {"LICENSES", "licenses", "LICENCES", "licences"}
MAX_SOURCE = 4 << 20            # bytes read per source file

# licence base id -> phrases (all must appear, case-insensitive, whitespace folded) that identify its text
WORDING = {
    "Apache-2.0": ["apache license", "version 2.0", "grant of patent license"],
    "MIT": ["permission is hereby granted, free of charge", "the above copyright notice and this permission notice shall be included"],
    "GPL-2.0": ["gnu general public license", "version 2, june 1991"],
    "GPL-3.0": ["gnu general public license", "version 3, 29 june 2007"],
    "LGPL-2.1": ["gnu lesser general public license", "version 2.1"],
    "LGPL-3.0": ["gnu lesser general public license", "version 3"],
    "AGPL-3.0": ["gnu affero general public license", "version 3"],
    "BSD-3-Clause": ["redistribution and use in source and binary forms", "neither the name"],
    "BSD-2-Clause": ["redistribution and use in source and binary forms", "redistributions in binary form must reproduce"],
    "ISC": ["permission to use, copy, modify, and/or distribute this software for any purpose"],
    "Zlib": ["this software is provided 'as-is'", "altered source versions must be plainly marked"],
    "BSL-1.0": ["boost software license - version 1.0"],
    "MPL-2.0": ["mozilla public license version 2.0"],
    "Unlicense": ["this is free and unencumbered software released into the public domain"],
    "NCSA": ["university of illinois/ncsa open source license"],
    "Unicode-3.0": ["unicode license v3"],
}
NO_TEXT = re.compile(r"^LicenseRef-")


def base(lic):
    lic = lic.strip().rstrip("+")
    for suf in ("-only", "-or-later"):
        if lic.endswith(suf):
            lic = lic[: -len(suf)]
    return lic


def expression_terms(expr):
    """[[id, ...], ...]: AND-joined terms, each a list of OR alternatives; exceptions after WITH dropped."""
    expr = re.sub(r"\bWITH\s+[A-Za-z0-9.\-]+", " ", expr).replace("(", " ").replace(")", " ")
    terms = []
    for part in re.split(r"\s+AND\s+", expr.strip()):
        alts = [a for a in re.split(r"\s+OR\s+", part.strip()) if a]
        if alts:
            terms.append([a.split()[0] for a in alts])
    return terms


def is_text_file(rel):
    parts = rel.split("/")
    if parts[-1].lower().endswith(CODE_EXT):
        return False
    return bool(TEXT_NAME.search(parts[-1])) or any(p in TEXT_DIR for p in parts[:-1])


def fold(s):
    return re.sub(r"\s+", " ", s.lower())


def carried(odir):
    """The licence ids whose wording appears in some licence file under catalog/<origin>."""
    out = set()
    for d, _, names in os.walk(odir):
        for n in names:
            p = os.path.join(d, n)
            if not is_text_file(os.path.relpath(p, odir)):
                continue
            with open(p, "rb") as f:
                t = fold(f.read(1 << 21).decode("utf-8", "replace"))
            out |= {lic for lic, ws in WORDING.items() if all(w in t for w in ws)}
    return out


def named(odir):
    """{SPDX expression: [origin-relative source path, ...]} over the harvested module sources."""
    out = {}
    for kind in sorted(os.listdir(odir)):
        kd = os.path.join(odir, kind)
        if not os.path.isdir(kd):
            continue
        for name in sorted(os.listdir(kd)):
            src = os.path.join(kd, name, "src")
            if not os.path.isfile(os.path.join(kd, name, "module.toml")) or not os.path.isdir(src):
                continue
            for d, _, files in os.walk(src):
                for n in files:
                    p = os.path.join(d, n)
                    with open(p, "rb") as f:
                        head = f.read(MAX_SOURCE)
                    for m in SPDX_LINE.finditer(head):
                        expr = m.group(1).decode().strip()
                        if expr:
                            out.setdefault(expr, []).append(os.path.relpath(p, src))
    return out


def gaps(root=ROOT):
    """[(origin, licence id, an example source path)] for every named licence with no carried text."""
    out = []
    cat = os.path.join(root, "catalog")
    for origin in sorted(os.listdir(cat)):
        odir = os.path.join(cat, origin)
        if not os.path.isfile(os.path.join(odir, "ORIGIN.toml")):
            continue
        have, seen = carried(odir), set()
        for expr, paths in sorted(named(odir).items()):
            for alts in expression_terms(expr):
                ids = [a for a in alts if not NO_TEXT.match(a)]
                if not ids or any(base(a) in have for a in ids):
                    continue
                key = "/".join(base(a) for a in ids)
                if key not in seen:
                    seen.add(key)
                    out.append((origin, key, sorted(paths)[0]))
    return out


class SpdxTexts(unittest.TestCase):
    def test_every_named_licence_has_a_text(self):
        bad = gaps()
        self.assertEqual(bad, [], "origins name licences whose text they do not carry:\n" +
                         "\n".join(f"  {o}: {lic} (e.g. {p})" for o, lic, p in bad))

    def test_a_missing_text_fails(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            o = os.path.join(tmp, "catalog", "demo")
            src = os.path.join(o, "kernel", "k", "src")
            os.makedirs(src)
            open(os.path.join(o, "ORIGIN.toml"), "w").close()
            open(os.path.join(o, "kernel", "k", "module.toml"), "w").close()
            with open(os.path.join(src, "a.c"), "w") as f:
                f.write("// SPDX-License-Identifier: GPL-2.0-only WITH Linux-syscall-note\n")
            with open(os.path.join(src, "b.c"), "w") as f:
                f.write("// SPDX-License-Identifier: MIT OR Apache-2.0\n// SPDX-License-Identifier: LicenseRef-X\n")
            with open(os.path.join(o, "LICENSE"), "w") as f:
                f.write("Permission is hereby granted, free of charge, to any person ... The above copyright notice "
                        "and this permission notice shall be included in all copies\n")
            self.assertEqual(gaps(tmp), [("demo", "GPL-2.0", "a.c")])


if __name__ == "__main__":
    unittest.main()
