#!/usr/bin/env python3
"""Compile gate: every hip-c++ kernel module compiles standalone for gfx1151 on control-node, the way its origin
compiles it. Standard library only. Run on control-node from the repo root:

    python3 gate/compile/run.py [--origin NAME] [--module ID] [--jobs N] [--keep]

For each catalog/<origin>/<kind>/<name>/module.toml with [build].lang = "hip-c++" and [gate].compile = "standalone":
  - the compiler command comes from catalog/<origin>/ORIGIN.toml [compile] (driver, arch_flag, rocm_flag) with the
    module's [build].options (or the origin's default flags), plus [build].source_options per source;
    the toolchain root is [build].toolchain looked up in toolchains.toml (rocm_root);
  - include directories are the module's own [build].includes followed by those of every module it requires by id,
    transitively, then the origin's [compile].system_includes under the toolchain root;
  - a module compiles either its [build].sources one by one, or the compile units listed in [build].units (a JSON
    file: units of concatenated text and files, as hipfire builds its sources from kernels.rs);
  - the result goes to gate/results/compile/<module-id>.json: per unit the command, exit code, seconds, object size
    and sha256, and the tail of the compiler's stderr on failure. Objects are deleted unless --keep.
A module passes when every unit compiled. Nothing here reads the GPU."""
import argparse, concurrent.futures as cf, hashlib, json, os, shlex, shutil, subprocess, sys, time, tomllib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'gate', 'results', 'compile')
TMP = os.path.join(ROOT, 'build', 'compile-gate')


def load(path):
    with open(path, 'rb') as f:
        return tomllib.load(f)


def modules():
    out = {}
    cat = os.path.join(ROOT, 'catalog')
    for origin in sorted(os.listdir(cat)):
        for d, _, fs in os.walk(os.path.join(cat, origin)):
            if 'module.toml' in fs:
                m = load(os.path.join(d, 'module.toml'))
                m['_dir'] = d
                out[m['id']] = m
    return out


def include_dirs(m, mods, seen=None):
    seen = seen if seen is not None else set()
    if m['id'] in seen:
        return []
    seen.add(m['id'])
    dirs = [os.path.join(m['_dir'], i) for i in m['build'].get('includes', [])]
    for r in m.get('requires', []):
        dep = r.get('module')
        if dep:
            if dep not in mods:
                raise KeyError(f"{m['id']} requires unknown module {dep}")
            dirs += include_dirs(mods[dep], mods, seen)
    return dirs


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def compile_unit(job):
    t0 = time.time()
    try:
        p = subprocess.run(job['cmd'], capture_output=True, text=True, timeout=job['timeout'], cwd=job['cwd'])
        rc, err = p.returncode, p.stderr
    except subprocess.TimeoutExpired:
        rc, err = -9, f"timeout after {job['timeout']} s"
    r = {'unit': job['unit'], 'cmd': ' '.join(shlex.quote(c) for c in job['cmd']).replace(ROOT + '/', ''),
         'rc': rc, 'seconds': round(time.time() - t0, 2)}
    obj = job['obj']
    if rc == 0 and os.path.isfile(obj) and os.path.getsize(obj) > 0:
        r['object_bytes'] = os.path.getsize(obj)
        r['object_sha256'] = sha256(obj)
    elif rc == 0:
        r['rc'], err = -1, 'compiler exited 0 but wrote no object\n' + err
    if r['rc'] != 0:
        lines = [l for l in err.splitlines() if 'error' in l.lower()] or err.splitlines()
        r['stderr'] = '\n'.join(lines[:12])[:3000]
    if not job['keep']:
        for f in (obj, job.get('gen')):
            if f and os.path.isfile(f):
                os.remove(f)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--origin'); ap.add_argument('--module'); ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--keep', action='store_true'); ap.add_argument('--timeout', type=int, default=900)
    a = ap.parse_args()
    tcs = load(os.path.join(ROOT, 'toolchains.toml'))
    mods = modules()
    os.makedirs(OUT, exist_ok=True)
    summary = []
    versions = {}
    for mid, m in sorted(mods.items()):
        b = m['build']
        if b.get('lang') != 'hip-c++' or m.get('gate', {}).get('compile') != 'standalone':
            continue
        origin = m['provenance']['origin']
        if (a.origin and origin != a.origin) or (a.module and mid != a.module):
            continue
        oc = load(os.path.join(ROOT, 'catalog', origin, 'ORIGIN.toml'))['compile']
        tc = b['toolchain']
        root = tcs[tc]['rocm_root']
        driver = oc['driver'].replace('<rocm>', root)
        if tc not in versions:
            v = subprocess.run([driver, '--version'], capture_output=True, text=True)
            versions[tc] = ' | '.join((v.stdout + v.stderr).splitlines()[:2])
        incs = include_dirs(m, mods) + [os.path.join(root, i) for i in oc.get('system_includes', [])]
        base = [driver, oc['arch_flag']] + [f.replace('<rocm>', root) for f in oc.get('rocm_flags', [oc['rocm_flag']] if 'rocm_flag' in oc else [])]
        flags = b.get('options', oc.get('flags', []))
        tmp = os.path.join(TMP, mid)
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        jobs = []
        ext = oc.get('object_ext', '.o')
        if b.get('units'):
            units = json.load(open(os.path.join(m['_dir'], b['units'])))['units']
            for u in units:
                gen = os.path.join(tmp, u['name'] + '.hip')
                with open(gen, 'w') as g:
                    for part in u['parts']:
                        g.write(part['text'] if 'text' in part else open(os.path.join(ROOT, 'catalog', origin, part['path']), errors='replace').read())
                obj = os.path.join(tmp, u['name'] + ext)
                cmd = base + flags + u.get('flags', []) + [x for i in incs for x in ('-I', i)] + [gen, '-o', obj]
                jobs.append({'unit': u['name'], 'cmd': cmd, 'obj': obj, 'gen': gen, 'keep': a.keep, 'timeout': a.timeout, 'cwd': tmp})
        else:
            sf = b.get('source_options', {})
            for i, s in enumerate(b.get('sources', [])):
                obj = os.path.join(tmp, f'{i}-' + os.path.basename(s) + ext)
                cmd = base + flags + sf.get(s, []) + [x for i2 in incs for x in ('-I', i2)] + \
                    oc.get('source_prefix', []) + [os.path.join(m['_dir'], s), '-o', obj]
                jobs.append({'unit': s, 'cmd': cmd, 'obj': obj, 'keep': a.keep, 'timeout': a.timeout, 'cwd': tmp})
        if not jobs:
            continue
        t0 = time.time()
        with cf.ThreadPoolExecutor(a.jobs) as ex:
            res = list(ex.map(compile_unit, jobs))
        failed = [r for r in res if r['rc'] != 0]
        out = {'module': mid, 'gate': 'compile', 'arch': b.get('arch', 'gfx1151'), 'host': os.uname().nodename,
               'date': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'toolchain': tc, 'toolchain_root': root,
               'compiler': versions[tc], 'units': len(res), 'passed': len(res) - len(failed), 'failed': len(failed),
               'status': 'pass' if not failed else 'fail', 'seconds': round(time.time() - t0, 1)}
        if failed:
            out['reason'] = f"{len(failed)} of {len(res)} units failed; first: {failed[0]['unit']}: " + \
                (failed[0].get('stderr', '').splitlines() or ['?'])[0][:300]
        out['results'] = res
        json.dump(out, open(os.path.join(OUT, mid + '.json'), 'w'), indent=1)
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)
        summary.append((mid, out['status'], out['passed'], out['units'], out['seconds']))
        print(f"{out['status']:4} {mid}  {out['passed']}/{out['units']} units  {out['seconds']} s" +
              (f"  -- {out['reason'][:200]}" if failed else ''), flush=True)
    print(f"modules: {len(summary)}  pass: {sum(1 for s in summary if s[1] == 'pass')}  fail: {sum(1 for s in summary if s[1] != 'pass')}")


if __name__ == '__main__':
    main()
