#!/usr/bin/env python3
"""Harvest the pod's served StrixFold TP4 build (fork commit 72c0f083, branch serve-next-20261123c) into catalog/strixfold_tp4.

    tools/pod-run.sh 'E=build/strixfold-72c0f083; rm -rf $E && mkdir -p $E && \\
        git -C /data/pod/plans/strixfold/fork archive 72c0f083fca18f80fcf73b2358cf8235c3dd5363 | tar -x -C $E'
    tools/pod-pull.sh build/strixfold-72c0f083
    python3 docs/harvest/harvest_strixfold_tp4.py          # STRIXFOLD_TP4_EXPORT moves the export

The data files it reads (docs/harvest/data/strixfold_tp4): commits.json = `git log --format='%h %s' --name-only
88ea779..<HEAD>` of the fork as {path: [[hash8, subject], ...]}; dsv41-kit.manifest.json = the served stage's kit
manifest (deploy-stage/72c0f083fca1/native-hip/share/tensorfold/hip/gfx1151/dsv41/manifest.json); aot.json and
tri-SHA256SUMS = the TP4 Triton kit 4d4c5173d2 (unchanged since 003f679c); pylib-SHA256SUMS = the Pillow the stage
carries (recipe [stage.inputs]).

Why a sub-origin and not a re-cut of catalog/strixfold: the TP2 recipes (dsv41-flash.strixfold-tp2*, their zip2 and
ABI variants and the vision work on top of them) are pinned to the TP2 pair's deployment f64b5ff2e28b, and the TP4
serving commit is not a descendant of f64b5ff (both sit on upstream 88ea779; 190 pod commits on the TP4 side). A
re-cut would move every one of those recipes off its reference. An origin has one commit (SPEC.md section 3), so the
TP4 tree is its own origin directory, `strixfold_tp4`, parent `strixfold`, holding only what it changes or adds
against the catalogue (docs/SPEC-PROPOSALS.md section 12):

  - for every module of catalog/strixfold with a file whose bytes differ at HEAD, an overlay module of the same
    kind and name (`<kind>.strixfold_tp4.<name>`) holding the TP4 version of each such file verbatim at its
    origin-relative path, plus overlay.json (per file: the parent module, the sha256 of the parent's and of the TP4
    bytes, the commits between the base and HEAD that touched it);
  - files the TP4 tree adds that the build needs (the import closure of the assembled tree, the build graph's
    b.path() roots and the kit registry's HIP sources), assigned to the overlay module of the file that imports them;
    the three new HIP sources get kernel modules of their own;
  - the TP4 Triton kit (pod card 4d4c5173d2, gfx1151-v6f2c38-tp4, 204 objects) as the [[prebuilt]] record of
    kernel.strixfold_tp4.triton_kit, and the deployed HIP+Triton kit manifest as the record of runtime.strixfold_tp4.kit;
  - tool.strixfold_tp4.dsv41_hip: every other file of tools/dsv41-hip (launcher, serve.conf, checks), verbatim. The
    build never reads them; the deploy stage carries the whole directory (git archive of tools/dsv41-hip), so a stage
    made from the catalogue (tools/flerp-stage.py) needs them byte for byte.

A recipe lays these modules over the strixfold ones with `[build.overlay] "<id>" = "."` (flerplib/assemble_zig.py:
root "." = the sub-origin's files sit at their own origin paths; a file that replaces nothing is an addition). A file
of the parent that is gone at HEAD (zig/src/families/dsv41/state_file.zig) stays in the tree, imported by nothing, as
HARVEST.json's gone_from_parent records. Every overlay module declares `closure = "zig:strixfold_tp4"`: the TP4 change set is one
import closure that crosses the parent's module lines.
It reads the export only; it runs nothing from it.
"""
import fnmatch
import hashlib
import json
import os
import shutil
import sys
import tomllib

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from flerplib.assemble_zig import references  # noqa: E402

HEAD = '72c0f083fca18f80fcf73b2358cf8235c3dd5363'
SHORT, DEPLOY = HEAD[:8], HEAD[:12]
EXPORT = os.environ.get('STRIXFOLD_TP4_EXPORT', os.path.join(REPO, 'build', 'strixfold-' + SHORT))
DATA = os.path.join(REPO, 'docs', 'harvest', 'data', 'strixfold_tp4')
ORIGIN, PARENT = 'strixfold_tp4', 'strixfold'
LICENCE_FILES = ('LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md', 'LICENSES/Apache-2.0.txt', 'LICENSES/CUTLASS-BSD-3-Clause.txt',
                 'LICENSES/MIT.txt', 'LICENSES/MiaAI-Lab-MIT.txt', 'LICENSES/Pillow-MIT-CMU.txt',
                 'LICENSES/PyTorch-BSD-3-Clause.txt')
BASE = '88ea77961851b60e8a1747fddd2410d29443283c'       # merge-base of f64b5ff and HEAD (upstream Shoestring-dev)
BRANCH = ('serve-next-20261123c (TP4-SERVE1 003f679c + VGATHER, Edge-REAP, route records, per-request acts, logprobs, '
          'loop policy, expert mask), local to the pod fork')
BUILT_IN = '/data/pod/plans/serve-next-20261123c/wt (run.sh)'   # the worktree the served stage's native-hip came from
LAUNCHER = 'tools/dsv41-hip/'                                   # tool.strixfold_tp4.dsv41_hip: the rest of this directory
LAUNCHER_ID = 'tool.strixfold_tp4.dsv41_hip'
REPO_URL = 'https://github.com/Shoestring-dev/Strixfold'
ODIR = os.path.join(REPO, 'catalog', ORIGIN)
CLOSURE = 'zig:strixfold_tp4'
K = 'zig/kernels/hip/dsv41/'
KITFLAGS = ['--offload-device-only', '--no-gpu-bundle-output', '-O3', '-std=c++20', '-fno-gpu-rdc',
            '-mno-wavefrontsize64', '-ffp-contract=off', '-fno-fast-math', '-fdenormal-fp-math=ieee',
            '-D__HIP_PLATFORM_AMD__']
TRI_DIR = 'tools/dsv41-hip/tri/'
EVIDENCE = 'gate/results/dsv41-flash.strixfold-tp4/build.txt'   # 34/34 steps, kit manifest = the deployed one
TRI_NEW = ['kit_add.py', 'kit_strict.py']   # added since f64b5ff: the TP4 variant additions and the strict kit check (4d4c5173d2)

# the HIP sources the TP4 tree adds: kernel modules of their own (interfaces from the registry and file headers)
NEW_KERNELS = {
    'aux_markov.hip': dict(
        name='aux_markov',
        summary='The drafter\'s Markov step for a sampled stream on gfx1151 in place of the Triton _score and _finish: '
                'score = logit + bias, then the best of score / draft temperature + Gumbel noise of the verify row\'s '
                'own key, so a sampled request drafts the token its verify row is likely to draw',
        why='A greedy draft lands only with the target\'s probability of its single best token; drawing the draft with '
            'the same noise the verify row uses lifts acceptance for sampled requests (draft_temperature) while the '
            'verify rows still decide every token, so replies do not depend on this kernel',
        provides=[('draft.markov-sampled/f16/bf16.hip', 'tf_ds_aux_markov_race',
                   'one fp32 add as _score, fp32 Gumbel g within a few ulps of the host double; a greedy row picks '
                   '_score\'s token bit for bit; drafts stay proposals', 'one row a block of 1024 threads, draft steps')]),
    'exl3_mlp.hip': dict(
        name='exl3_mlp',
        summary='exl3.hip\'s decode glinear with a compile-time prefetch depth (2..4 trellis tiles in flight, register '
                'or LDS ring), same arguments, grid and accumulate order as exl3.hip so a row\'s sum is the same bits',
        why='Decode glinear on gfx1151 stalls on trellis loads; a deeper ring hides them without changing a single '
            'accumulate (pod card e5cb84a995 picks a no-spill schedule per shape through the plan, TF_DS_MLP_SCAN)',
        provides=[('linear.decode/exl3/f16-f32.row-invariant-prefetch', 'tf_ds_exl3_glinear_mlp_k<k>_w<warps> (24 variants)',
                   'the dot2 chain, half-wave, wave and split adds of exl3.hip in its order: bit-equal to exl3.hip\'s glinear',
                   'decode / verify windows, memory-bound')]),
    'reap.hip': dict(
        name='reap',
        summary='Edge-REAP routing statistics on gfx1151: pack a sampled row\'s routed expert outputs, their sum of squares, '
                'the per-(expert, bucket) count and gate-mass adds and the norm join, all into f64 device accumulators',
        why='Expert pruning needs per-expert statistics from served traffic; under tensor parallelism no rank holds a whole '
            'expert output, so the engine joins the partial norms itself; off (REAP_STATS=0) nothing launches '
            '(pod card 023c5d2ad9, docs of catalog/strixfold_reap)',
        provides=[('moe.route-stats/-/f64.hip', 'tf_ds_reap_pack, tf_ds_reap_ssq, tf_ds_reap_route, tf_ds_reap_norm',
                   'exact: f64 adds of the router\'s own bf16/f32 outputs; replies do not depend on it',
                   'REAP_STATS=1 only, every REAP_EVERY-th committed token')]),
    'route_rec.hip': dict(
        name='route_rec',
        summary='Per-request routing records on gfx1151: one thread per (layer, expert) adds a request\'s gate mass into '
                'its stream slot, read by rank 0 at stream release',
        why='Routing records of a single request (ROUTE_RECORDS, pod card 28153e18f3) need the gate mass summed per stream '
            'on the device; off nothing launches',
        provides=[('moe.route-record/-/f32.hip', 'tf_ds_route_rec', 'exact: fp32 adds in a fixed order; replies do not depend on it',
                   'ROUTE_RECORDS=1 only, rank 0')]),
    'vision.hip': dict(
        name='vision', spdx='Apache-2.0 AND SunPro',
        licence_note='erff (vision.hip:42) is fdlibm\'s as musl carries it (FreeBSD s_erff.c), Copyright (C) 1993 Sun '
                     'Microsystems, under the SunPro notice kept in the file. ',
        summary='The DeepSeek-V4.1 vision tower and aligner on gfx1151: patch unfold, RMSNorm, linear, 2D RoPE, '
                'attention, GELU / SiLU-mul and the image-span copy, every fp32 operation one IEEE operation in a '
                'declared order',
        why='Image prompts at TP4 encode on rank 0 on the GPU (vit_hip.zig) instead of a CPU torch child; rows are '
            'then sent hidden-wide to every rank (pod card b9fafca926); compiled into the serving kit whether or not '
            'a request carries an image',
        provides=[('vision.tower/deepseek-v4.1/bf16.hip', 'tf_ds_vision_unfold, _rms_norm, _linear, _rope, _attention, '
                   '_gelu, _silu_mul, _add, _span', 'no contraction, no fast math; bf16 rounding RNE with NaN quieting',
                   'rank 0, one picture at a time')]),
}


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def tq(s):
    return json.dumps(s, ensure_ascii=False)


def parent():
    idx = json.load(open(os.path.join(REPO, 'catalog', PARENT, 'HARVEST.json')))
    org = tomllib.load(open(os.path.join(REPO, 'catalog', PARENT, 'ORIGIN.toml'), 'rb'))
    mods = {}
    for d, _, names in os.walk(os.path.join(REPO, 'catalog', PARENT)):
        if 'module.toml' in names:
            m = tomllib.load(open(os.path.join(d, 'module.toml'), 'rb'))
            mods[m['id']] = m
    return idx['owner'], idx['commit'], org['harvest']['unharvested'], mods


def pfile(owner, p):
    kind, _, name = owner[p].split('.')
    return os.path.join(REPO, 'catalog', PARENT, kind, name, 'src', p)


def unharvested_by_parent(pats, p):
    return any(fnmatch.fnmatchcase(p, pat) or (pat.endswith('/**') and p.startswith(pat[:-2])) for pat in pats)


def build_refs():
    """b.path("...") roots of the build graph files and the registry's HIP sources, as export paths."""
    import re
    out = set()
    for f in ('build.zig', 'zig/build/hip_dsv41.zig'):
        text = open(os.path.join(EXPORT, f)).read()
        out |= set(re.findall(r'b\.path\("([^"{}]+)"\)', text))
    reg = json.load(open(os.path.join(EXPORT, K + 'registry.json')))
    out |= {K + s['source'] for s in reg['sources'] if s.get('kit') == 'dsv41'}
    return out


def main():
    if not os.path.isfile(os.path.join(EXPORT, 'build.zig')):
        sys.exit(f'{EXPORT}: no export of {HEAD[:8]} (see the docstring)')
    owner, pcommit, pats, pmods = parent()
    commits = json.load(open(os.path.join(DATA, 'commits.json')))
    changed = {p: owner[p] for p in owner if os.path.isfile(os.path.join(EXPORT, p)) and sha(os.path.join(EXPORT, p)) != sha(pfile(owner, p))}
    gone = sorted(p for p in owner if not os.path.isfile(os.path.join(EXPORT, p)))

    # additions: the import closure of the tree as the TP4 build sees it, the build graph's roots, the kit's sources
    tree = set(owner)
    added = {}                                              # path -> the tree file that needs it ('' = the build graph)
    for r in sorted(build_refs()):
        if r not in tree and os.path.isfile(os.path.join(EXPORT, r)) and not unharvested_by_parent(pats, r):
            added[r] = ''
    todo = sorted(tree | set(added))
    while todo:
        nxt = []
        for p in todo:
            src = os.path.join(EXPORT, p) if os.path.isfile(os.path.join(EXPORT, p)) else pfile(owner, p)
            if not p.endswith('.zig'):
                continue
            for _, target in references(open(src, errors='replace').read()):
                dest = os.path.normpath(os.path.join(os.path.dirname(p), target))
                if dest in tree or dest in added or not os.path.isfile(os.path.join(EXPORT, dest)):
                    continue
                added[dest] = p
                nxt.append(dest)
        todo = nxt

    # owners of the additions: a new HIP source has its own module; anything else the overlay of its importer's module
    assign = {}
    for p in sorted(added):
        if p.startswith(K) and os.path.basename(p) in NEW_KERNELS:
            assign[p] = 'kernel.%s.%s' % (ORIGIN, NEW_KERNELS[os.path.basename(p)]['name'])
    for _ in range(20):
        for p, by in sorted(added.items()):
            if p in assign:
                continue
            if by in owner:
                assign[p] = owner[by].replace('.%s.' % PARENT, '.%s.' % ORIGIN)
            elif by in assign:
                assign[p] = assign[by]
            elif by == '':                                  # a build graph root: the module that owns its directory
                d = os.path.dirname(p)
                near = sorted((q for q in owner if os.path.dirname(q) == d), key=lambda q: q)
                if near:
                    assign[p] = owner[near[0]].replace('.%s.' % PARENT, '.%s.' % ORIGIN)
    missing = sorted(set(added) - set(assign))
    if missing:
        sys.exit(f'no owner for {missing}')

    # modules
    mods = {}
    for p, mid in changed.items():
        mods.setdefault(mid.replace('.%s.' % PARENT, '.%s.' % ORIGIN), {'parent': mid, 'over': [], 'new': []})['over'].append(p)
    for p, mid in assign.items():
        parent_id = mid.replace('.%s.' % ORIGIN, '.%s.' % PARENT)
        mods.setdefault(mid, {'parent': parent_id if parent_id in pmods else None, 'over': [], 'new': []})['new'].append(p)
    tri = json.load(open(os.path.join(DATA, 'aot.json')))
    tp4_tri = 'kernel.%s.triton_kit' % ORIGIN
    mods.setdefault(tp4_tri, {'parent': 'kernel.%s.triton_kit' % PARENT, 'over': [], 'new': []})
    for p in TRI_NEW:                       # the TP4 kit's generator inputs (the parent keeps tests and probes out)
        rel = TRI_DIR + p
        if os.path.isfile(os.path.join(EXPORT, rel)) and rel not in owner and rel not in assign:
            mods[tp4_tri]['new'].append(rel)
            assign[rel] = tp4_tri
    # the rest of tools/dsv41-hip: launcher, serve.conf and checks, which the deploy stage carries whole (never built)
    for d, _, names in os.walk(os.path.join(EXPORT, LAUNCHER)):
        for n in sorted(names):
            rel = os.path.relpath(os.path.join(d, n), EXPORT)
            if rel not in owner and rel not in assign:
                mods.setdefault(LAUNCHER_ID, {'parent': None, 'over': [], 'new': []})['new'].append(rel)
                assign[rel] = LAUNCHER_ID

    if os.path.isdir(ODIR):
        shutil.rmtree(ODIR)
    os.makedirs(ODIR)
    files_owner, overlays = {}, []
    for mid in sorted(mods):
        rec = mods[mid]
        write_module(mid, rec, pmods, owner, commits, tri)
        for p in rec['over'] + rec['new']:
            files_owner[p] = mid
        overlays += [{'path': p, 'module': mid, 'parent_module': rec['parent']} for p in rec['over']]
    for f in LICENCE_FILES:
        if os.path.isfile(os.path.join(EXPORT, f)):
            os.makedirs(os.path.dirname(os.path.join(ODIR, f)), exist_ok=True)
            shutil.copyfile(os.path.join(EXPORT, f), os.path.join(ODIR, f))
    write_origin(files_owner, pats, pcommit)
    idx = {'origin': ORIGIN, 'parent': PARENT, 'parent_commit': pcommit, 'commit': HEAD, 'base': BASE,
           'modules': len(mods), 'files': len(files_owner), 'owner': dict(sorted(files_owner.items())),
           'overlays': overlays, 'gone_from_parent': gone,
           'added_because': {p: (by or 'build graph root / kit registry') for p, by in sorted(added.items())}}
    json.dump(idx, open(os.path.join(ODIR, 'HARVEST.json'), 'w'), indent=1, sort_keys=True)
    print(f'{ORIGIN}: {len(mods)} modules, {len(files_owner)} files ({len(overlays)} overlay, '
          f'{len(files_owner) - len(overlays)} added); gone from the parent: {gone}')


def tagged(iface):
    head, _, last = iface.rpartition('/')
    return f'{head}/{last}-tp4' if '.' in last else f'{head}/{last}.tp4'


def cards(paths, commits):
    import re
    seen = []
    for p in paths:
        for _, subj in commits.get(p, []):
            for c in re.findall(r'card ([0-9a-f]{10})', subj):
                if c not in seen:
                    seen.append(c)
    return seen


def write_module(mid, rec, pmods, owner, commits, tri):
    kind, _, name = mid.split('.')
    mdir = os.path.join(ODIR, kind, name)
    paths = sorted(rec['over'] + rec['new'])
    for p in paths:
        dst = os.path.join(mdir, 'src', p)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(EXPORT, p), dst)
        shutil.copymode(os.path.join(EXPORT, p), dst)
    pm = pmods.get(rec['parent']) if rec['parent'] else None
    kfile = os.path.basename(rec['new'][0]) if kind == 'kernel' and rec['new'] and os.path.basename(rec['new'][0]) in NEW_KERNELS else None
    cs = cards(paths, commits)
    if rec['over']:
        ov = {'what': f"StrixFold {SHORT}'s versions of files that a module of origin strixfold also holds; laid over "
                      "them by a recipe's [build.overlay] \"<id>\" = \".\" (docs/harvest/harvest_strixfold_tp4.py)",
              'parent_origin': PARENT, 'parent_commit': 'f64b5ff', 'head_commit': HEAD, 'files': []}
        for p in sorted(rec['over']):
            ov['files'].append({'path': p, 'parent_module': owner[p], 'parent_sha256': sha(pfile(owner, p)),
                                'tp4_sha256': sha(os.path.join(EXPORT, p)),
                                'commits': [f'{h} {s}' for h, s in commits.get(p, [])]})
        for p in sorted(rec['new']):
            ov['files'].append({'path': p, 'added': True, 'tp4_sha256': sha(os.path.join(EXPORT, p)),
                                'commits': [f'{h} {s}' for h, s in commits.get(p, [])]})
        with open(os.path.join(mdir, 'overlay.json'), 'w') as f:
            json.dump(ov, f, indent=1)
            f.write('\n')
    if mid == LAUNCHER_ID:
        summary = (f'The rest of StrixFold {SHORT}\'s tools/dsv41-hip ({len(paths)} files): the TP2 / TP4 launcher '
                   '(serve_lib.sh, serve.conf, the rank and keeper scripts), the RDMA and inputs checks, the qualification, '
                   'bench and kit-preparation scripts and their assets, verbatim')
        why = ('The served deploy directory is the git archive of tools/dsv41-hip next to the native-hip build '
               '(restore-tfp.sh stage, deploy_rank0.sh), and the ranks start from it (DEPLOY_DIR/tools/dsv41-hip); a stage '
               'made from the catalogue (tools/flerp-stage.py) needs these files byte for byte to be the served one')
        notes = ('Nothing here is compiled or imported by the build; the recipe dsv41-flash.strixfold-tp4 includes the '
                 'module so the assembled tree holds the directory and [stage] copies it. Parent-owned files of '
                 'tools/dsv41-hip (registry, numerics contract, tri generator inputs, zig_build.sh, rdma_check.py) stay '
                 f'in their modules or overlays. Cards behind the changes: {", ".join(cs[:40]) or "-"}.')
        provides = [('serve.launcher/dsv41-tp4/-', 'tools/dsv41-hip/serve_lib.sh with a serve conf (WORLD 2 or 4)',
                     'n/a', 'scripts: bash and python3, run on each rank box from the deploy directory')]
        lang, sources, includes, flags = 'python', [], [], []
        requires = []
        gate_reason = ('scripts the deploy stage carries; tools/flerp-stage.py compares every one with the served stage')
    elif kfile:
        k = NEW_KERNELS[kfile]
        summary, why = k['summary'], k['why']
        notes = k.get('licence_note', '') + (f'New in the TP4 serving tree ({K}{kfile}, not in f64b5ff); compiled into the serving kit by the '
                 f'origin\'s build graph (zig/build/hip_dsv41.zig, flags as catalog/strixfold ORIGIN.toml [compile]). '
                 f'Cards: {", ".join(cs) or "-"}.')
        provides = k['provides']
        lang, sources, includes, flags = 'hip-c++', ['src/' + K + kfile], ['src/' + K.rstrip('/')], KITFLAGS
        requires = [('module', 'kernel.strixfold_tp4.dsv41_abi')]
        gate_reason = 'compiled only inside the recipe dsv41-flash.strixfold-tp4 by the origin\'s build graph (its object is pinned there and came out as the deployed one); not in gate/compile'
    else:
        pn = rec['parent']
        n_over, n_new = len(rec['over']), len(rec['new'])
        what = f'{n_over} file(s) of {pn}' + (f' and {n_new} file(s) the TP4 tree adds beside them' if n_new else '')
        top = ', '.join(sorted({os.path.basename(p) for p in paths})[:6]) + (' ...' if len(paths) > 6 else '')
        summary = f'StrixFold {SHORT} (the pod\'s TP4 serving build) for {pn}: {what} ({top}), laid over the f64b5ff files'
        why = (f'{pn} is cut at f64b5ff, the TP2 pair\'s deployment; the pod serves DeepSeek-V4.1 at TP4 (four gfx1151 nodes) '
               f'from {SHORT}, so the TP4 recipe lays these files over {pn} without touching it or the TP2 recipes')
        subj = []
        for p in paths:
            for h, s in commits.get(p, []):
                if f'{h} {s}' not in subj:
                    subj.append(f'{h} {s}')
        notes = (f'Overlay of {pn} (overlay.json has every file with both sha256 and its commits). Cards behind the '
                 f'changes: {", ".join(cs) or "-"}. Latest commits: ' + '; '.join(subj[:4]))
        provides = [(tagged(pv['interface']), pv['api'], pv['numeric'],
                     f'TP4 serving tree {SHORT} (world 2 to 4); ' + pv['regime']) for pv in pm.get('provides', [])]
        if mid == 'kernel.strixfold_tp4.triton_kit':
            summary = ('The TP4 Triton kit gfx1151-v6f2c38-tp4 (pod card 4d4c5173d2): the TP2 kit\'s variants plus the '
                       'shapes a quarter of the heads and columns takes at world 4, AOT-compiled into 204 gfx1151 '
                       f'code objects for {len(tri["kernels"])} launch variants, and the generator files the TP4 tree changes or adds')
            why = ('A TP4 rank launches Triton kernels at shapes the TP2 kit was never compiled for: built against the TP2 '
                   'kit af9276f296 the first request fails with MissingTritonVariant (pod card 96431609cf); the serving '
                   f'build {SHORT} was built with -Dhip-triton-kit naming this kit')
        bm = pm.get('build', {})
        lang = bm.get('lang', 'zig')
        sources, includes, flags = [], [], []
        requires = [('module', pn)]
        gate_reason = ('builds inside the recipe dsv41-flash.strixfold-tp4 (control-node, zig 0.17.0, TheRock 10.2) over '
                       'its parent module; the kit came out as the deployed one; not run on a GPU from that build')
        if mid == 'kernel.strixfold_tp4.triton_kit':
            gate_reason = ('prebuilt code objects (Triton AOT, pod card 4d4c5173d2); build.sh verifies every object and aot.json '
                           'against this record before the recipe builds')
    o = [f'id        = {tq(mid)}', f'kind      = {tq(kind)}', f'summary   = {tq(summary)}', f'why       = {tq(why)}',
         f'notes     = {tq(notes)}', '', '[provenance]', f'origin    = {tq(ORIGIN)}', f'repo      = {tq(REPO_URL)}',
         f'commit    = {tq(HEAD[:8])}', f'branch    = {tq(BRANCH)}', 'paths     = [']
    o += [f'  {tq(p)},' for p in paths]
    spdx = NEW_KERNELS[kfile].get('spdx', 'Apache-2.0') if kfile else 'Apache-2.0'
    o += [']', 'authors   = ["StrixFold contributors", "TensorFold contributors", "pod"]', 'modified  = false']
    if rec['over']:
        o.append('upstream  = ' + tq(f'overlay on {rec["parent"]} (origin strixfold, f64b5ff): {len(rec["over"])} of '
                                     f'these paths are its files too, replaced whole; overlay.json'))
    o += ['', '[licence]', f'spdx      = {tq(spdx)}', '', '[build]', f'lang      = {tq(lang)}',
          'sources   = [' + ', '.join(tq(s) for s in sources) + ']',
          'includes  = [' + ', '.join(tq(s) for s in includes) + ']', 'arch      = "gfx1151"', 'toolchain = "therock-10.2"']
    if flags:
        o.append('options   = [' + ', '.join(tq(s) for s in flags) + ']')   # SPEC [build] options, as catalog/strixfold
    if lang == 'zig' and paths and any(p.endswith('.zig') for p in paths):
        o.append(f'closure   = {tq(CLOSURE)}')
    if mid == 'kernel.strixfold_tp4.triton_kit':
        o.append(f'variants  = {len(tri["kernels"])}')
    for iface, api, numeric, regime in provides:
        o += ['', '[[provides]]', f'interface = {tq(iface)}', f'api       = {tq(api)}', f'numeric   = {tq(numeric)}', f'regime    = {tq(regime)}']
    for key, val in requires:
        o += ['', '[[requires]]', f'{key:<9} = {tq(val)}']
    pre = os.path.join(mdir, 'prebuilt')
    if mid == 'kernel.strixfold_tp4.triton_kit':
        os.makedirs(pre, exist_ok=True)
        shutil.copyfile(os.path.join(DATA, 'aot.json'), os.path.join(pre, 'aot.json'))
        sums = [l.replace('  hsaco/', '  ') for l in open(os.path.join(DATA, 'tri-SHA256SUMS')) if l.strip()]
        open(os.path.join(pre, 'SHA256SUMS'), 'w').write(''.join(sorted(sums)))
        digest = hashlib.sha256(''.join(sorted(l.split()[0] + '\n' for l in sums)).encode()).hexdigest()
        o += ['', '[[prebuilt]]', 'kit       = "dsv41 Triton AOT kit for TP4 (gfx1151-v6f2c38-tp4)"', f'objects   = {len(sums)}',
              'where     = "control-node:/data/pod/plans/4d4c5173d2/kit/gfx1151-v6f2c38-tp4/hsaco/<sha256>.hsaco"',
              'manifest  = "prebuilt/aot.json"', f'manifest_sha256 = {tq(sha(os.path.join(pre, "aot.json")))}',
              'hashes    = "prebuilt/SHA256SUMS"', f'hashes_digest = "sha256 of the sorted object hashes = {digest}"']
    if mid == 'runtime.strixfold_tp4.kit':
        os.makedirs(pre, exist_ok=True)
        shutil.copyfile(os.path.join(DATA, 'dsv41-kit.manifest.json'), os.path.join(pre, 'dsv41-kit.manifest.json'))
        km = json.load(open(os.path.join(pre, 'dsv41-kit.manifest.json')))
        o += ['', '[[prebuilt]]', f'kit       = "dsv41 serving kit (HIP + Triton) of the TP4 serving build {DEPLOY}"',
              f'objects   = {len(km["objects"])}', 'manifest  = "prebuilt/dsv41-kit.manifest.json"',
              f'manifest_sha256 = {tq(sha(os.path.join(DATA, "dsv41-kit.manifest.json")))}',
              f'where     = "control-node:/data/pod/plans/strixfold/deploy-stage/{DEPLOY}/native-hip/share/tensorfold/hip/gfx1151/dsv41 (built in {BUILT_IN})"',
              'toolchain = "HIP version: 7.17.26392-0000000"']
    compile_ = 'prebuilt' if lang in ('triton-aot', 'hsaco') else 'n/a'
    status = 'verified' if compile_ == 'prebuilt' else 'builds'    # the recipe's build on control-node (EVIDENCE)
    if mid == LAUNCHER_ID:
        status = 'harvested'                                       # not built; the stage compare is its check
    o += ['', '[gate]', 'parity    = "none"', 'bench     = "none"', f'status    = {tq(status)}', f'evidence  = {tq(EVIDENCE)}',
          f'compile   = {tq(compile_)}', f'reason    = {tq(gate_reason)}', '']
    open(os.path.join(mdir, 'module.toml'), 'w').write('\n'.join(o))


def write_origin(files_owner, pats, pcommit):
    allf = []
    for d, dirs, names in os.walk(EXPORT):
        allf += [os.path.relpath(os.path.join(d, n), EXPORT) for n in names]
    owned_dirs = set()
    for f in files_owner:
        parts = f.split('/')
        owned_dirs.update('/'.join(parts[:i]) for i in range(1, len(parts)))
    un = []
    for f in sorted(allf):
        if f in files_owner:
            continue
        parts = f.split('/')
        for i in range(1, len(parts)):
            d = '/'.join(parts[:i])
            if d not in owned_dirs:
                f = d + '/**'
                break
        if f not in un:
            un.append(f)
    lic = ', '.join(tq(f) for f in LICENCE_FILES)
    text = f'''# The pod's served StrixFold TP4 build: DeepSeek-V4.1-Flash on four gfx1151 nodes (TP4-SERVE1 and the serve-next work).
# A sub-origin of `strixfold`: only what {SHORT} changes or adds against catalog/strixfold (f64b5ff) and the build
# needs is harvested; every other file of this tree is catalog/strixfold's or listed as unharvested here.
# docs/DSV41-TP4.md describes the recipe built from it.
name      = "{ORIGIN}"
parent    = "{PARENT}"
repo      = "{REPO_URL}"
commit    = "{HEAD[:8]}"
branch    = "{BRANCH}"
base      = "{BASE[:7]}"            # merge-base with the parent's commit {pcommit}: upstream Shoestring-dev
state     = "serving: deployed to F D C A as ~/tfp/deploy/{DEPLOY} (tp4-serve1-image-v7.conf), stage /data/pod/plans/strixfold/deploy-stage/{DEPLOY}"
pod_tree  = "/data/pod/plans/strixfold/fork"
reference_checkout = "build/strixfold-{SHORT} (git archive of the pod fork at {HEAD})"

[licence]
spdx      = "Apache-2.0"
# copied verbatim next to this file at their origin paths: the root texts and LICENSES/, which THIRD_PARTY_NOTICES.md cites
files     = [{lic}]
source    = "control-node:/data/pod/plans/strixfold/fork at {HEAD[:8]} (the export above, git archive of that commit)"
third_party = "as catalog/strixfold (THIRD_PARTY_NOTICES.md)"

[toolchain]
host      = "zig 0.17.0 (/home/pod/opt on control-node), as catalog/strixfold"
device    = "therock-10.2"
triton    = "the TP4 Triton kit gfx1151-v6f2c38-tp4 (pod card 4d4c5173d2), prebuilt"

[harvest]
mode      = "modules"
tool      = "docs/harvest/harvest_strixfold_tp4.py"
unharvested = [
''' + ''.join(f'  {tq(u)},\n' for u in un) + ']\n'
    open(os.path.join(ODIR, 'ORIGIN.toml'), 'w').write(text)


if __name__ == '__main__':
    main()
