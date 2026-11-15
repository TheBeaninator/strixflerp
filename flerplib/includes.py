"""#include scanning for C-family sources, used to check that `requires` covers what a module really includes."""
import os
import re

C_FAMILY = (".c", ".cc", ".cpp", ".h", ".hpp", ".hip", ".inc")
_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*(["<])([^">]+)[">]', re.M)


def is_c_family(path):
    return path.endswith(C_FAMILY)


def scan(path):
    """[(name, quoted)] for every #include in the file; quoted is False for <angle> includes."""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return [(m.group(2), m.group(1) == '"') for m in _INCLUDE.finditer(text)]


def resolve(src_dir, name, quoted, roots):
    """The first existing file for an include: next to the including file (quoted only), then each root in order."""
    cands = [os.path.join(src_dir, name)] if quoted else []
    cands += [os.path.join(r, name) for r in roots]
    for c in cands:
        if os.path.isfile(c):
            return os.path.normpath(c)
    return None


def resolve_tree(rel_dir, name, srcs):
    """A quoted include as the compiler finds it in an assembled tree: next to the including file, whichever module
    holds the target. rel_dir is the including file's origin-relative directory, srcs the src/ directories of the
    other modules of its origin. The path is normalised first ("../x.h" need not have its parent in that module)."""
    rel = os.path.normpath(os.path.join(rel_dir, name))
    if rel.startswith(".."):
        return None
    for s in srcs:
        c = os.path.join(s, rel)
        if os.path.isfile(c):
            return c
    return None
