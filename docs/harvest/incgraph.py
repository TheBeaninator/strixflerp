"""Include-closure helper for the harvest scripts: which origin files does a translation unit pull in?
Standard library only. Usage: incgraph.py <origin-root> <extra-include-root,...> <tu> [<tu> ...]"""
import os, re, sys
INC = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', re.M)

def closure(root, roots, tus):
    """Return {tu: sorted origin-relative files reachable through #include}, resolving against the including
    file's directory (quote form only) and then each include root. Unresolved names (system headers) are dropped."""
    cache = {}
    def direct(rel):
        if rel in cache: return cache[rel]
        out = []
        try: text = open(os.path.join(root, rel), errors='replace').read()
        except OSError: text = ''
        for kind, name in INC.findall(text):
            cands = []
            if kind == '"': cands.append(os.path.normpath(os.path.join(os.path.dirname(rel), name)))
            cands += [os.path.normpath(os.path.join(r, name)) for r in roots]
            for c in cands:
                if os.path.isfile(os.path.join(root, c)):
                    out.append(c); break
        cache[rel] = out
        return out
    res = {}
    for tu in tus:
        seen, todo = [], [tu]
        while todo:
            f = todo.pop()
            if f in seen: continue
            seen.append(f); todo += direct(f)
        res[tu] = sorted(seen)
    return res

if __name__ == '__main__':
    root, roots, tus = sys.argv[1], sys.argv[2].split(','), sys.argv[3:]
    for tu, files in closure(root, roots, tus).items():
        print(tu, len(files)); [print('  ', f) for f in files if f != tu]
