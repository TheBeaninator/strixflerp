"""Shared machinery of the phase-2 harvest scripts (docs/harvest/harvest_<origin>.py). Standard library only.

A harvest script describes an origin as a list of M(...) module specs and calls run(). run() copies each module's
origin files verbatim to catalog/<origin>/<kind>/<name>/src/<origin-relative path>, writes module.toml and, when
gate/results/compile/<module-id>.json exists and passed, records gate.status = "builds" with that evidence path.
Re-running is idempotent: a module directory is rebuilt from the origin checkout every time.
"""
import glob, hashlib, json, os, re, shutil, sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
UPSTREAM = os.environ.get('STRIXFLERP_UPSTREAM', os.path.expanduser('~/Code/strixflerp-upstream'))
SUPPLEMENT = os.environ.get('STRIXFLERP_SUPPLEMENT', os.path.expanduser('~/Code/strixflerp-upstream-supplement'))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


class Origin:
    def __init__(self, name, checkout, repo, commit, branch, spdx, authors, toolchain, supplement=None):
        self.name, self.repo, self.commit, self.branch = name, repo, commit, branch
        self.spdx, self.authors, self.toolchain = spdx, authors, toolchain
        self.roots = [os.path.join(UPSTREAM, checkout)]
        if supplement:
            self.roots.append(os.path.join(SUPPLEMENT, supplement))
        self._files = None

    def files(self):
        """Every origin-relative file path (union of the checkout and its supplement)."""
        if self._files is None:
            out = set()
            for r in self.roots:
                for d, _, fs in os.walk(r):
                    for f in fs:
                        out.add(os.path.relpath(os.path.join(d, f), r))
            self._files = sorted(out)
        return self._files

    def find(self, rel):
        for r in self.roots:
            p = os.path.join(r, rel)
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(rel)

    def expand(self, patterns, exclude=()):
        """Origin-relative files matching any pattern. A pattern ending in '/' takes the whole subtree; '*' does not
        cross '/', '**' does. A leading '!' in `patterns` removes matches."""
        def rx(pat):
            if pat.endswith('/'):
                return re.compile(re.escape(pat) + '.*')
            r = ''.join('.*' if t == '**' else '[^/]*' if t == '*' else '[^/]' if t == '?' else re.escape(t)
                        for t in re.split(r'(\*\*|\*|\?)', pat))
            return re.compile(r)
        inc = [rx(p) for p in patterns if not p.startswith('!')]
        exc = [rx(p[1:]) for p in patterns if p.startswith('!')] + [rx(p) for p in exclude]
        out = []
        for f in self.files():
            if any(r.fullmatch(f) for r in inc) and not any(r.fullmatch(f) for r in exc):
                out.append(f)
        return out


# The harvest scripts were written with their own op words; catalog/strixite (harvested in parallel) settled the
# vocabulary, so names are mapped onto it here: an overlap then shows up as a shared <family>.<op> prefix.
VOCAB = [('attention.full/', 'attn.softmax-gqa/'), ('attention.sparse-sink/', 'attn.sparse-sink/'),
         ('attention.qk-norm-rope-store/', 'attn.qk-norm-rope-store/'), ('gdn.recurrence/', 'mixer.gdn/'),
         ('gdn.front/', 'mixer.gdn-front/'), ('kv.store/', 'kvcache.store/'), ('kv.compress2/', 'kvcache.compress2/'),
         ('kv.compress/', 'kvcache.compress/'), ('kv.tiering/', 'kvcache.tiering/'), ('moe.route/', 'moe.router/'),
         ('norm.rms/', 'norm.rmsnorm/'), ('rope.apply/', 'pos.rope/'), ('rope.heads/', 'pos.rope-heads/'),
         ('indexer.qsa/', 'attn.qsa-indexer/'), ('indexer.score/', 'attn.indexer-score/'),
         ('indexer.topk/', 'attn.indexer-topk/'), ('sample.topk/', 'logits.topk/'), ('sample.argmax/', 'logits.argmax/'),
         ('sample.keyed/', 'sample.next-token/'), ('sample.host/', 'sample.next-token/'),
         ('cache.prompt-state/', 'cache.prompt/'), ('grammar.constraint/', 'grammar.matcher/'),
         ('linear.decode/bf16/', 'linear.dense/bf16/')]


def iface(name):
    """SPEC names are <family>.<op>/<weight-format-or-dash>/<dtype-or-dash>. A variant tag that tells two numerically
    different providers apart rides on the dtype field (`.../f32.tag`); when the dtype is a dash the tag moves to the
    op (`family.op-tag/x/-`), because `flerp check` wants a bare dash there."""
    for old, new in VOCAB:
        if name.startswith(old):
            name = new + name[len(old):]
            break
    m = re.fullmatch(r'([^/]+)/([^/]+)/-\.([a-z0-9_+-]+)', name)
    return f'{m.group(1)}-{m.group(3)}/{m.group(2)}/-' if m else name


def P(interface, api, numeric, regime):
    return dict(interface=iface(interface), api=api, numeric=numeric, regime=regime)


class M:
    """One module. paths: origin-relative patterns (see Origin.expand). sources: origin-relative compile units (subset of
    paths). requires: interface names, or 'module:<id>' for a direct module dependency."""
    def __init__(self, id, summary, paths, provides=(), requires=(), lang='hip-c++', sources=(), includes=None,
                 flags=(), toolchain=None, authors=None, spdx=None, upstream=None, notes=None, compile='standalone',
                 reason=None, prebuilt=None, variants=None, parity='none', bench='none', extra_files=None,
                 units=None, exclude=(), display_paths=None, source_flags=None, closure=None):
        self.__dict__.update(locals())
        del self.__dict__['self']
        self.kind, self.origin_name, self.name = id.split('.', 2)


def tq(s):
    return '"' + str(s).replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ') + '"'


def tlist(xs, indent='  ', width=118):
    xs = [tq(x) for x in xs]
    one = '[' + ', '.join(xs) + ']'
    if len(one) <= width - 12:
        return one
    return '[\n' + ''.join(f'{indent}{x},\n' for x in xs) + ']'


def write_module(origin, m, compile_results, extra_requires=()):
    mdir = os.path.join(REPO, 'catalog', origin.name, m.kind, m.name)
    if os.path.isdir(mdir):
        shutil.rmtree(mdir)
    os.makedirs(mdir)
    m._dir, m._includes = mdir, []
    files = origin.expand(m.paths, m.exclude)
    if not files and not m.prebuilt and not m.extra_files:
        raise SystemExit(f'{m.id}: no origin files match {m.paths}')
    for f in files:
        dst = os.path.join(mdir, 'src', f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(origin.find(f), dst)
    for name, content in (m.extra_files or {}).items():
        dst = os.path.join(mdir, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if isinstance(content, bytes):
            open(dst, 'wb').write(content)
        else:
            open(dst, 'w').write(content)
    for s in m.sources:
        if s not in files:
            raise SystemExit(f'{m.id}: source {s} is not among its paths')
    includes = m.includes if m.includes is not None else (['src'] if m.lang in ('hip-c++', 'c++') else [])
    includes = [i for i in includes if os.path.isdir(os.path.join(mdir, i))]
    m._includes = [os.path.normpath(os.path.join(mdir, i)) for i in includes]

    # gate status from the compile-gate result, if any
    status, evidence, reason = 'harvested', None, m.reason
    res_path = os.path.join('gate', 'results', 'compile', m.id + '.json')
    res = compile_results.get(m.id)
    if res is not None:
        if res.get('status') == 'pass':
            status, evidence = 'builds', res_path
        else:
            reason = (reason + '; ' if reason else '') + f"compile gate: {res.get('status')}: {res.get('reason', '')}"[:400]
            evidence = res_path

    o = []
    o.append(f'id        = {tq(m.id)}')
    o.append(f'kind      = {tq(m.kind)}')
    o.append(f'summary   = {tq(m.summary)}')
    if m.notes:
        o.append(f'notes     = {tq(m.notes)}')
    o.append('')
    o.append('[provenance]')
    o.append(f'origin    = {tq(origin.name)}')
    o.append(f'repo      = {tq(origin.repo)}')
    o.append(f'commit    = {tq(origin.commit)}')
    o.append(f'branch    = {tq(origin.branch)}')
    o.append(f'paths     = {tlist(files)}')
    o.append(f'authors   = {tlist(m.authors or origin.authors)}')
    o.append('modified  = false')
    if m.upstream:
        o.append(f'upstream  = {tq(m.upstream)}')
    o.append('')
    o.append('[licence]')
    o.append(f'spdx      = {tq(m.spdx or origin.spdx)}')
    o.append('')
    o.append('[build]')
    o.append(f'lang      = {tq(m.lang)}')
    srcs = ['src/' + s for s in m.sources]
    o.append(f'sources   = {tlist(srcs)}')
    o.append(f'includes  = {tlist(includes)}')
    o.append('arch      = "gfx1151"')
    if m.toolchain != '':
        o.append(f'toolchain = {tq(m.toolchain or origin.toolchain)}')
    if m.flags:
        o.append(f'options   = {tlist(m.flags)}')
    if m.units:
        o.append(f'units     = {tq(m.units)}')
    if m.source_flags:
        o.append('source_options = { ' + ', '.join(f'{tq("src/" + k)} = {tlist(v, width=10000)}' for k, v in m.source_flags.items()) + ' }')
    if m.variants is not None:
        o.append(f'variants  = {m.variants}')
    if m.closure:
        o.append(f'closure   = {tq(m.closure)}')
    for p in m.provides:
        o.append('')
        o.append('[[provides]]')
        for k in ('interface', 'api', 'numeric', 'regime'):
            o.append(f'{k:<9} = {tq(p[k])}')
    reqs = []
    for r in list(m.requires) + sorted(extra_requires):
        r = r if r.startswith('module:') else iface(r)
        if r not in reqs and r != 'module:' + m.id:
            reqs.append(r)
    for r in reqs:
        o.append('')
        o.append('[[requires]]')
        if r.startswith('module:'):
            o.append(f'module    = {tq(r[7:])}')
        else:
            o.append(f'interface = {tq(r)}')
    for pb in (m.prebuilt or []):
        o.append('')
        o.append('[[prebuilt]]')
        for k, v in pb.items():
            o.append(f'{k:<9} = {v if isinstance(v, int) else tq(v)}')
    o.append('')
    o.append('[gate]')
    o.append(f'parity    = {tq(m.parity)}')
    o.append(f'bench     = {tq(m.bench)}')
    o.append(f'status    = {tq(status)}')
    o.append(f'compile   = {tq(m.compile)}')
    if evidence:
        o.append(f'evidence  = {tq(evidence)}')
    if reason:
        o.append(f'reason    = {tq(reason)}')
    open(os.path.join(mdir, 'module.toml'), 'w').write('\n'.join(o) + '\n')
    return files


def load_compile_results():
    out = {}
    for p in glob.glob(os.path.join(REPO, 'gate', 'results', 'compile', '*.json')):
        try:
            j = json.load(open(p))
        except ValueError:
            continue
        if isinstance(j, dict) and 'module' in j:
            out[j['module']] = j
    return out


C_FAMILY = ('.c', '.cc', '.cpp', '.h', '.hpp', '.hip', '.inc')
_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*(["<])([^">]+)[">]', re.M)


def include_requires(modules, provided_by):
    """{module id: {'module:<id>', ...}}: modules whose headers a module includes without requiring them (the rule
    `flerp check` enforces: next to the including file for quoted includes, then every include root of the origin)."""
    out = {}
    for m in modules:
        allowed = {m.id}
        for r in m.requires:
            allowed.update([r[7:]] if r.startswith('module:') else provided_by.get(iface(r), []))
        roots = m._includes + [d for o in modules if o is not m for d in o._includes]
        owner = {d: o for o in modules for d in o._includes}
        need = set()
        src = os.path.join(m._dir, 'src')
        for d, _, fs in os.walk(src):
            for f in fs:
                if not f.endswith(C_FAMILY):
                    continue
                text = open(os.path.join(d, f), errors='replace').read()
                for q, name in _INCLUDE.findall(text):
                    cands = ([os.path.join(d, name)] if q == '"' else []) + [os.path.join(r, name) for r in roots]
                    hit = next((os.path.normpath(c) for c in cands if os.path.isfile(c)), None)
                    if hit is None and q == '"':      # next to the file in the assembled tree (flerplib.includes.resolve_tree)
                        rel = os.path.normpath(os.path.join(os.path.relpath(d, src), name))
                        hit = next((os.path.join(o._dir, 'src', rel) for o in modules if o is not m and
                                    not rel.startswith('..') and os.path.isfile(os.path.join(o._dir, 'src', rel))), None)
                    if hit is None or hit.startswith(m._dir + os.sep):
                        continue
                    holder = next((o for dd, o in owner.items() if hit.startswith(dd + os.sep)), None)
                    if holder is not None and holder.id not in allowed:
                        need.add('module:' + holder.id)
        out[m.id] = need
    return out


def unharvested(origin, owned, licence_files):
    """Patterns (flerp diff-origin's [harvest] unharvested) naming every origin file that is in no module: whole
    directories where nothing under them was taken, single files elsewhere."""
    files = [f for f in origin.files() if f not in owned]
    owned_dirs = set()
    for f in owned:
        parts = f.split('/')
        for i in range(1, len(parts)):
            owned_dirs.add('/'.join(parts[:i]))
    out = []
    for f in files:
        parts = f.split('/')
        for i in range(1, len(parts)):
            d = '/'.join(parts[:i])
            if d not in owned_dirs:
                pat = d + '/**'
                break
        else:
            pat = f
        if pat not in out:
            out.append(pat)
    return out


def run(origin, modules, origin_toml, left_out=None, licence_files=()):
    odir = os.path.join(REPO, 'catalog', origin.name)
    os.makedirs(odir, exist_ok=True)
    # drop module directories that are no longer in the spec
    want = {(m.kind, m.name) for m in modules}
    for kind in os.listdir(odir):
        kd = os.path.join(odir, kind)
        if os.path.isdir(kd):
            for name in os.listdir(kd):
                if (kind, name) not in want:
                    shutil.rmtree(os.path.join(kd, name))
    results = load_compile_results()
    owner, total = {}, 0
    for m in modules:
        for f in write_module(origin, m, results):
            if f in owner:
                raise SystemExit(f'{f}: in both {owner[f]} and {m.id}')
            owner[f] = m.id
            total += 1
    # second pass: add the module requires that the sources' #include lines imply
    provided_by = {}
    for m in modules:
        for p in m.provides:
            provided_by.setdefault(p['interface'], []).append(m.id)
    extra = include_requires(modules, provided_by)
    for m in modules:
        if extra[m.id]:
            write_module(origin, m, results, extra[m.id])
    for f in licence_files:
        shutil.copyfile(origin.find(f), os.path.join(odir, f))
    un = unharvested(origin, set(owner), licence_files)
    toml = origin_toml.strip() + '\n\n[harvest]\nmode      = "modules"\ntool      = ' + tq(f'docs/harvest/harvest_{origin.name}.py') + \
        '\n# Origin files that are in no module (what and why: docs/HARVEST.md). Whole directories where nothing was taken.\nunharvested = ' + tlist(un) + '\n'
    open(os.path.join(odir, 'ORIGIN.toml'), 'w').write(toml)
    idx = {'origin': origin.name, 'commit': origin.commit, 'modules': len(modules), 'files': total,
           'by_kind': {}, 'owner': owner}
    for m in modules:
        idx['by_kind'][m.kind] = idx['by_kind'].get(m.kind, 0) + 1
    if left_out is not None:
        idx['left_out'] = left_out
    json.dump(idx, open(os.path.join(odir, 'HARVEST.json'), 'w'), indent=1, sort_keys=True)
    print(f'{origin.name}: {len(modules)} modules {idx["by_kind"]}, {total} files, {sum(len(v) for v in extra.values())} include-derived requires')
    return owner
