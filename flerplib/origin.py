"""`flerp diff-origin`: prove vendored sources are byte-identical to an origin checkout, and that the harvest
covers the checkout (every origin file is in exactly one module or named as unharvested in ORIGIN.toml)."""
import fnmatch
import hashlib
import os

SKIP = (".git", "build")     # top-level entries of a checkout that are not origin files


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def diff_module(module, checkout, theirs_all=None):
    """[(origin path, problem)] - empty when every file of the module matches the checkout. A directory or
    pattern entry in [provenance] paths is expanded on both sides, so a file the origin has and the module lacks
    is a problem too."""
    out, ours = [], module.files()
    if any(module.expand(e, ours) != [e] for e in module.paths):
        theirs_all = theirs_all if theirs_all is not None else origin_files(checkout)
        for e in module.paths:
            for f in module.expand(e, theirs_all):
                if f not in ours:
                    out.append((f, "missing in the module"))
    for p in ours:
        mine, theirs = os.path.join(module.dir, "src", p), os.path.join(checkout, p)
        if not os.path.isfile(theirs):
            out.append((p, "missing in the origin checkout"))
        elif sha256(mine) != sha256(theirs):
            out.append((p, "differs from the origin"))
    return out


def origin_files(checkout):
    out = []
    for d, dirs, names in os.walk(checkout):
        if d == checkout:              # .git is a file in a git worktree
            dirs[:] = [x for x in dirs if x not in SKIP]
            names = [n for n in names if n not in SKIP]
        out += [os.path.relpath(os.path.join(d, n), checkout) for n in names]
    return sorted(out)


def glob_match(path, pattern):
    if pattern.endswith("/**"):
        return path.startswith(pattern[:-2])
    return fnmatch.fnmatchcase(path, pattern)


def diff_origin(cat, origin, checkout):
    """A dict: modules, files, identical, problems [(module id or '-', path, problem)], unharvested count."""
    data = cat.origin_data(origin) or {}
    harvest = data.get("harvest", {}) if isinstance(data.get("harvest"), dict) else {}
    globs = harvest.get("unharvested", [])
    mods = [m for m in cat.modules if m.dir_origin == origin]
    problems, owner, files = [], {}, 0
    theirs_all = origin_files(checkout)
    for m in mods:
        mine = m.files()
        files += len(mine)
        problems += [(m.id, p, why) for p, why in diff_module(m, checkout, theirs_all)]
        for p in mine:
            if p in owner:
                problems.append((m.id, p, f"also in {owner[p]}"))
            owner[p] = m.id
    skipped = 0
    for f in theirs_all:
        if f in owner:
            continue
        if any(glob_match(f, g) for g in globs):
            skipped += 1
        else:
            problems.append(("-", f, "in the origin checkout but in no module and not listed as unharvested"))
    lic = data.get("licence", {}) if isinstance(data.get("licence"), dict) else {}
    for f in lic.get("files", []):
        ours, theirs = os.path.join(cat.root, "catalog", origin, f), os.path.join(checkout, f)
        if not (os.path.isfile(ours) and os.path.isfile(theirs)) or sha256(ours) != sha256(theirs):
            problems.append(("-", f, "licence file differs from the origin or is missing"))
    return {"origin": origin, "checkout": checkout, "modules": len(mods), "files": files,
            "identical": files - sum(1 for mid, _, _ in problems if mid != "-"),
            "unharvested": skipped, "problems": problems}
