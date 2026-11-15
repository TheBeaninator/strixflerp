#!/usr/bin/env python3
"""Harvest Shoestring-dev/Strixfold pull request #6 ("image input for DeepSeek-V4.1 on HIP") into catalog/strixfold_pr6.

    python3 docs/harvest/harvest_strixfold_pr6.py        # reads ~/Code/strixflerp-upstream/strixfold-pr6 (a git clone
                                                         # of the PR branch; STRIXFLERP_UPSTREAM moves the parent)

Why a sub-origin and not more modules of `strixfold`: an origin has one commit (ORIGIN.toml), `flerp diff-origin
--origin strixfold` proves every module of the origin against one checkout (the recipe gate's origin stage does it
against the deployed f64b5ff), and docs/harvest/harvest_strixfold.py deletes module directories it did not write. A
pull request is another tree at another commit, so it is its own origin directory with its own ORIGIN.toml.

What it writes:
  - the PR's new files, verbatim, in six modules;
  - the PR's versions of the files it MODIFIES, verbatim too, in the module they belong to. Those paths are also in
    modules of origin `strixfold` (at another commit, with the pod's patches). Nothing there is touched: the new
    module is an *overlay* (docs/SPEC-PROPOSALS.md section 12) and records, next to module.toml,
        overlay.json                 per file: the strixfold module that owns the path, the sha256 of the PR base, the
                                     PR head and the catalogue's version, and whether the PR's change applies to the
                                     catalogue's version (`patch`, first with no fuzz, then with 2), with the sha256 of the result
        overlay/<path>.patch         `git diff <base> <head> -- <path>`
    Both are derived data, like units.json.
It only reads the checkout (git diff, file copies); it runs nothing from it.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UP = os.environ.get('STRIXFLERP_UPSTREAM', os.path.expanduser('~/Code/strixflerp-upstream'))
CHECKOUT = os.path.join(UP, 'strixfold-pr6')
ORIGIN = 'strixfold_pr6'
PARENT = 'strixfold'
HEAD = '766df7554fd12c492f926f679011b2892df76a8c'
BASE = '70cfb8c6cfa0d0b9c393c8d3f2087e9c8c9f88d7'
REPO_URL = 'https://github.com/Shoestring-dev/Strixfold'
BRANCH = 'feat/hip-cpu-image-input (pull request #6, aercanozcan)'
ODIR = os.path.join(REPO, 'catalog', ORIGIN)
D = 'zig/src/families/dsv41/'
T = 'tools/dsv41-hip/'
ZIG_REASON = ('never compiled or run here: no recipe assembles it. It needs the overlay rule of docs/SPEC-PROPOSALS.md '
              'section 12 before `flerp assemble` can place its files over the strixfold modules')

MODULES = [
    dict(kind='runtime', name='vision_sidecar', lang='zig', closure='zig:families/dsv41', spdx='Apache-2.0 AND MIT',
         summary='DeepSeek-V4.1 image encoder as a per-image Python child: the checkpoint ViT and aligner on CPU torch, '
                 'the pipe protocol to it with one deadline and cancellation, an LRU of encoded rows',
         notes='vision_cpu.zig embeds vision_cpu.py and vision_hip.py (@embedFile) and passes them to `python -I -c`. '
               'vision_cpu.py is written after DeepSeek V4.1 Flash 1017 inference/vision.py (revision 1bae0e10, git blob '
               '77af0bde; MIT, Copyright (c) 2023 DeepSeek; text in src/tools/dsv41-hip/assets/DEEPSEEK_VISION_LICENSE): '
               'it is not a copy of that file. It refuses any torch but 2.9.0+cpu (vision_cpu.py:158). picture.zig and '
               'pil.zig (executor.strixfold.dsv41_vision) and model.zig (executor.strixfold.dsv41) are imported and '
               'not changed. docs/VISION-DSV41.md has the path end to end.',
         new=[D + 'vision_cpu.zig', D + 'vision_pipe.zig', D + 'vision_cache.zig', D + 'vision_cpu.py',
              D + 'vision_hip.py', T + 'VISION.md', T + 'requirements-vision.lock.txt',
              T + 'assets/DEEPSEEK_VISION_LICENSE'],
         authors=['aercanozcan', 'DeepSeek (the reference inference/vision.py that vision_cpu.py follows)'],
         provides=[('vision.tower/deepseek-v4.1/bf16.cpu-torch', 'vision_cpu.zig Vision (open, hook, rowsBatch, close)',
                    'ViT and aligner in bf16 tensors with fp32 inside rms / rotary, torch MATH attention over query '
                    'pieces of 128 rows; said by the PR to equal the reference; not measured here',
                    'one child process an uncached image, on the lane thread; at most 9,216 patches and 1,024 rows an image'),
                   ('vision.worker-pipe/vcpu-v1/bf16', 'vision_cpu.zig start / finish, vision_pipe.zig Transfer; '
                    'vision_cpu.py main', 'bytes only: 32-byte grid + bf16 patches in, "VCPU" + rows + byte count + bf16 rows out',
                    'one image a process; one deadline (TF_DS_VISION_TIMEOUT, 240 s) over spawn, write and read')],
         requires=[('module', 'executor.strixfold.dsv41_vision'), ('module', 'executor.strixfold.dsv41'),
                   ('module', 'runtime.strixfold.lanes')]),
    dict(kind='kernel', name='vision_softmax', lang='hip-c++', spdx='Apache-2.0',
         summary='Row softmax over one tile of fp32 attention scores (kernel normalize_scores), for the optional HIP '
                 'attention of the vision worker; and the script that compiles it',
         notes='Launched from Python by ctypes (runtime.strixfold_pr6.vision_sidecar, vision_hip.py), between two '
               'hipblasSgemmStridedBatched calls a tile. build_vision_worker.sh is the origin\'s compile command '
               '(amdclang++ -x hip --offload-device-only -O3); it was not run here. The PR says this path changes the '
               'reduction order and is not qualified (VISION.md:84-86).',
         new=[D + 'vision_softmax.hip', T + 'build_vision_worker.sh'], sources=['src/' + D + 'vision_softmax.hip'],
         authors=['aercanozcan'],
         provides=[('vision.attn-softmax/-/f32', 'normalize_scores(float *scores, int keys); grid = rows of the tile, block 256',
                    'max and sum by a 256-lane tree reduction per row, expf, one multiply by 1/sum', 'keys up to 9,216; tile of 128 query rows x 16 heads')],
         requires=[]),
    dict(kind='tool', name='prepare_vision', lang='zig', spdx='Apache-2.0',
         summary='Cut the vision tower (266 tensors) and the 43 image routing biases out of the pinned LibertAI EXL3 '
                 'checkpoint, with every source and output pinned by sha256; and the admission check the launcher runs',
         notes='Python 3 (hashlib.file_digest: 3.11 or newer), standard library only for prepare_vision.py; '
               'verify_vision.py starts the worker\'s interpreter and asserts torch 2.9.0+cpu, numpy 2.3.5, Pillow 12.3.0. '
               '[build] lang is "zig" only because SPEC.md has no value for a script; nothing is compiled.',
         new=[T + 'prepare_vision.py', T + 'verify_vision.py', T + 'vision-checkpoint.json'], authors=['aercanozcan'],
         provides=[('tool.vision-extract/safetensors/bf16', 'prepare_vision.py --model --out --bias-out; verify_vision.py --bias ...',
                    'byte copies of tensor ranges; outputs equal the pinned sha256 or are not published', 'host; reads the 42 shards the index names for these tensors')],
         requires=[]),
    dict(kind='runtime', name='image_key', lang='zig', closure='zig:core', spdx='Apache-2.0',
         summary='Image-aware prompt-cache keys: a shadow prompt whose first eight tokens of each image span are the '
                 'image\'s sha256, and the rule that a kept state is cut only between image spans',
         notes='image_key.zig is new; lane_host.zig and lanes/stream.zig are the PR\'s versions of files of '
               'runtime.strixfold.lanes (overlay.json).',
         new=['zig/src/core/lanes/image_key.zig'], overlay=['zig/src/core/lane_host.zig', 'zig/src/core/lanes/stream.zig'],
         authors=['aercanozcan', 'StrixFold contributors', 'TensorFold contributors'],
         provides=[('cache.prefix-key-image/-/-', 'core/lanes/image_key.zig make, validCut; lane_host.zig Job.cachePrompt',
                    'exact: sha256 over (rows flag, span length, payload bytes), 31 bits a word kept', 'on when Vision.cache_images (TF_DS_VISION_CACHE_MIB > 0) and the prompt cache is on')],
         requires=[('module', 'runtime.strixfold.lanes'), ('interface', 'cache.prompt-strixfold/-/-')]),
    dict(kind='executor', name='dsv41_image_rows', lang='zig', closure='zig:families/dsv41', spdx='Apache-2.0',
         summary='The PR\'s changes to the dsv41 family and the HIP host that let image rows into the prompt pass on HIP: '
                 'spans checked against a restored prefix, rows from the CPU tower, the vision feature switched on',
         notes='Overlay only: every file is the PR\'s version of a file of executor.strixfold.dsv41, '
               'runtime.strixfold.gpu_facade or runtime.strixfold.engine_host (overlay.json). The catalogue\'s versions '
               'are at f64b5ff with the pod\'s patches; the PR\'s are on upstream 70cfb8c.',
         overlay=[D + 'lanes.zig', D + 'model.zig', D + 'native.zig', 'zig/src/gpu/hip_facade.zig', 'zig/src/native/hip.zig'],
         authors=['aercanozcan', 'StrixFold contributors', 'TensorFold contributors'],
         provides=[('vision.image-rows/deepseek-v4.1/bf16', 'lanes.zig Lanes fill: Stream.images -> prompt.Span + rows -> Model.setImages',
                    'rows are copied, not computed: bf16 [tokens][5120] replace the embedding rows of the span', 'rank 0 encodes; followers get the rows in wire frames')],
         requires=[('interface', 'vision.tower/deepseek-v4.1/bf16.cpu-torch'), ('module', 'executor.strixfold.dsv41'),
                   ('module', 'runtime.strixfold.gpu_facade'), ('module', 'runtime.strixfold.engine_host')]),
    dict(kind='serve', name='image_prompt', lang='zig', spdx='Apache-2.0',
         summary='The PR\'s changes to the server\'s prompt path: each image keeps its identity through the chat template '
                 '(tool results may be reordered), spans are placed by identity, /tokenize counts the expanded rows',
         notes='Overlay only: the PR\'s versions of two files of serve.strixfold.http (overlay.json).',
         overlay=['zig/src/server/prompt.zig', 'zig/src/server/tokens.zig'],
         authors=['aercanozcan', 'StrixFold contributors', 'TensorFold contributors'],
         provides=[('serve.image-spans/-/-', 'server/prompt.zig renderImages, imageIdentities, spans; tokens.zig tokenize',
                    'n/a', 'PNG / JPEG base64 data URLs in image_url parts, tool results included')],
         requires=[('module', 'serve.strixfold.http')]),
]

ORIGIN_TOML = f'''# Shoestring-dev/Strixfold pull request #6: optional image input for DeepSeek-V4.1 on the HIP backend.
# A sub-origin of `strixfold`: only what the PR adds or modifies is harvested; every other file of this tree belongs
# to catalog/strixfold (at its own commit) and is listed as unharvested here. docs/VISION-DSV41.md describes it.
name      = "{ORIGIN}"
parent    = "{PARENT}"
repo      = "{REPO_URL}"
commit    = "{HEAD[:7]}"
branch    = "{BRANCH}"
base      = "{BASE[:7]}"            # merge-base with upstream main; the PR is two commits (6335ff9, 766df75) on it
state     = "open, no maintainer review when harvested (2026-11-11)"
reference_checkout = "~/Code/strixflerp-upstream/strixfold-pr6 (a clone of a git bundle made on relay-host; branch pr6)"

[licence]
spdx      = "Apache-2.0"
files     = ["LICENSE", "NOTICE"]        # NOTICE is the PR's: it adds the DeepSeek vision.py attribution
third_party = "vision_cpu.py follows DeepSeek V4.1 Flash 1017 inference/vision.py (MIT, Copyright (c) 2023 DeepSeek); licence text in runtime/vision_sidecar/src/tools/dsv41-hip/assets/DEEPSEEK_VISION_LICENSE. Run time, not vendored: torch 2.9.0+cpu (BSD-3-Clause), numpy 2.3.5 (BSD-3-Clause), Pillow 12.3.0 (MIT-CMU)"

[toolchain]
host      = "zig 0.17.0, as catalog/strixfold"
python    = "the PR's lock is for CPython 3.13 (tools/dsv41-hip/requirements-vision.lock.txt): torch 2.9.0+cpu, numpy 2.3.5, Pillow 12.3.0"
device    = "optional: amdclang++ -x hip --offload-device-only for vision_softmax.hip (tools/dsv41-hip/build_vision_worker.sh); hipBLAS and libamdhip64 by ctypes"
built_here = "nothing: harvested, not compiled, not run"
'''


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def git(*a):
    return subprocess.run(['git', '-C', CHECKOUT, *a], check=True, capture_output=True).stdout


def tq(s):
    return json.dumps(s, ensure_ascii=False)


def parent_owner():
    idx = json.load(open(os.path.join(REPO, 'catalog', PARENT, 'HARVEST.json')))
    return idx['owner'], idx['commit']


def overlay(mdir, paths):
    owner, pcommit = parent_owner()
    out = {'what': 'the PR\'s versions of files that modules of origin strixfold also hold (docs/SPEC-PROPOSALS.md section 12)',
           'tool': 'docs/harvest/harvest_strixfold_pr6.py', 'base_commit': BASE, 'head_commit': HEAD,
           'parent_origin': PARENT, 'parent_commit': pcommit, 'files': []}
    for p in paths:
        mid = owner[p]
        kind, _, name = mid.split('.')
        theirs = os.path.join(REPO, 'catalog', PARENT, kind, name, 'src', p)
        patch = git('diff', BASE, HEAD, '--', p)
        pp = os.path.join(mdir, 'overlay', p + '.patch')
        os.makedirs(os.path.dirname(pp), exist_ok=True)
        open(pp, 'wb').write(patch)
        base_sha = hashlib.sha256(git('show', f'{BASE}:{p}')).hexdigest()
        applies, fuzz, result = False, None, None
        for f in (0, 2):                  # context lines `patch` may ignore: 0 = every context line matches the parent
            with tempfile.TemporaryDirectory() as tmp:
                work = os.path.join(tmp, 'f')
                shutil.copyfile(theirs, work)
                r = subprocess.run(['patch', '-s', '--no-backup-if-mismatch', f'-F{f}', work, pp], capture_output=True, text=True)
                if r.returncode == 0:
                    applies, fuzz, result = True, f, sha(work)
                    break
        text = patch.decode()
        out['files'].append({
            'path': p, 'parent_module': mid, 'patch': 'overlay/' + p + '.patch',
            'lines_added': sum(1 for l in text.splitlines() if l.startswith('+') and not l.startswith('+++')),
            'lines_removed': sum(1 for l in text.splitlines() if l.startswith('-') and not l.startswith('---')),
            'pr_base_sha256': base_sha, 'pr_head_sha256': sha(os.path.join(CHECKOUT, p)), 'parent_sha256': sha(theirs),
            'parent_equals_pr_base': sha(theirs) == base_sha,
            'patch_applies_to_parent': applies, 'patch_fuzz': fuzz, 'parent_patched_sha256': result})
    json.dump(out, open(os.path.join(mdir, 'overlay.json'), 'w'), indent=1)
    open(os.path.join(mdir, 'overlay.json'), 'a').write('\n')
    return out


def write_module(m):
    mdir = os.path.join(ODIR, m['kind'], m['name'])
    if os.path.isdir(mdir):
        shutil.rmtree(mdir)
    paths = sorted(m.get('new', []) + m.get('overlay', []))
    for p in paths:
        dst = os.path.join(mdir, 'src', p)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(CHECKOUT, p), dst)
        shutil.copymode(os.path.join(CHECKOUT, p), dst)
    ov = overlay(mdir, m['overlay']) if m.get('overlay') else None
    mid = f"{m['kind']}.{ORIGIN}.{m['name']}"
    o = [f'id        = {tq(mid)}', f'kind      = {tq(m["kind"])}', f'summary   = {tq(m["summary"])}', f'notes     = {tq(m["notes"])}', '',
         '[provenance]', f'origin    = {tq(ORIGIN)}', f'repo      = {tq(REPO_URL)}', f'commit    = {tq(HEAD[:7])}',
         f'branch    = {tq(BRANCH)}', 'paths     = [']
    o += [f'  {tq(p)},' for p in paths]
    o += [']', 'authors   = [' + ', '.join(tq(a) for a in m['authors']) + ']', 'modified  = false']
    if ov:
        o.append('upstream  = ' + tq('overlay on ' + ', '.join(sorted({f['parent_module'] for f in ov['files']})) +
                                     f' (origin strixfold): {len(ov["files"])} of these paths are theirs too; overlay.json'))
    o += ['', '[licence]', f'spdx      = {tq(m["spdx"])}', '', '[build]', f'lang      = {tq(m["lang"])}',
          'sources   = [' + ', '.join(tq(s) for s in m.get('sources', [])) + ']', 'includes  = []', 'arch      = "gfx1151"',
          'toolchain = "therock-10.2"']
    if m.get('closure'):
        o.append(f'closure   = {tq(m["closure"])}')
    for iface, api, numeric, regime in m['provides']:
        o += ['', '[[provides]]', f'interface = {tq(iface)}', f'api       = {tq(api)}', f'numeric   = {tq(numeric)}', f'regime    = {tq(regime)}']
    for key, val in m['requires']:
        o += ['', '[[requires]]', f'{key:<9} = {tq(val)}']
    reason = ZIG_REASON if m['lang'] == 'zig' and m['kind'] != 'tool' else \
        'never run here' if m['kind'] == 'tool' else 'never compiled here; the origin compiles it with build_vision_worker.sh, not with catalog/strixfold\'s [compile] recipe'
    o += ['', '[gate]', 'parity    = "none"', 'bench     = "none"', 'status    = "harvested"', 'compile   = "n/a"', f'reason    = {tq(reason)}', '']
    open(os.path.join(mdir, 'module.toml'), 'w').write('\n'.join(o))
    return mid, paths, ov


def origin_files():
    out = []
    for d, dirs, names in os.walk(CHECKOUT):
        if d == CHECKOUT:
            dirs[:] = [x for x in dirs if x not in ('.git', 'build')]
        out += [os.path.relpath(os.path.join(d, n), CHECKOUT) for n in names]
    return sorted(out)


def unharvested(owned):
    """Globs for every file of the checkout that is in no module: a directory with nothing harvested under it as dir/**."""
    files = origin_files()
    owned_dirs = set()
    for f in owned:
        parts = f.split('/')
        owned_dirs.update('/'.join(parts[:i]) for i in range(1, len(parts)))
    out = []
    for f in files:
        if f in owned:
            continue
        parts = f.split('/')
        for i in range(1, len(parts)):
            d = '/'.join(parts[:i])
            if d not in owned_dirs:
                f = d + '/**'
                break
        if f not in out:
            out.append(f)
    return out


def main():
    if git('rev-parse', 'HEAD').decode().strip() != HEAD:
        sys.exit(f'{CHECKOUT} is not at {HEAD}')
    os.makedirs(ODIR, exist_ok=True)
    owner, overlays = {}, []
    for m in MODULES:
        mid, paths, ov = write_module(m)
        for p in paths:
            if p in owner:
                sys.exit(f'{p}: in both {owner[p]} and {mid}')
            owner[p] = mid
        overlays += [dict(f, module=mid) for f in (ov['files'] if ov else [])]
    for f in ('LICENSE', 'NOTICE'):
        shutil.copyfile(os.path.join(CHECKOUT, f), os.path.join(ODIR, f))
    un = unharvested(set(owner))
    toml = ORIGIN_TOML + '\n[harvest]\nmode      = "modules"\ntool      = "docs/harvest/harvest_strixfold_pr6.py"\n' \
        '# Everything the PR does not add or modify (it is catalog/strixfold\'s, at that origin\'s commit), and of the PR: its\n' \
        '# tests (tools/dsv41-hip/test_*vision*.py, zig/tests/server/fake_serve.zig, hip_test.zig), the launcher scripts it\n' \
        '# edits (serve.conf, serve_lib.sh, deploy_rank0.sh) and the documents (CHANGES.md, README.md, tools/dsv41-hip/README.md).\n' \
        'unharvested = [\n' + ''.join(f'  {tq(u)},\n' for u in un) + ']\n'
    open(os.path.join(ODIR, 'ORIGIN.toml'), 'w').write(toml)
    changed = git('diff', '--name-status', BASE, HEAD).decode().split('\n')
    changed = dict(reversed(l.split('\t')) for l in changed if l)
    idx = {'origin': ORIGIN, 'parent': PARENT, 'commit': HEAD, 'base': BASE, 'modules': len(MODULES), 'files': len(owner),
           'owner': owner, 'overlays': overlays,
           'pr_files': {p: {'status': s, 'module': owner.get(p)} for p, s in sorted(changed.items())}}
    json.dump(idx, open(os.path.join(ODIR, 'HARVEST.json'), 'w'), indent=1, sort_keys=True)
    print(f'{ORIGIN}: {len(MODULES)} modules, {len(owner)} files ({len(overlays)} overlay), {len(changed)} files in the PR, '
          f'{len(un)} unharvested patterns')
    for f in overlays:
        print(f"  overlay {f['path']} on {f['parent_module']}: parent == PR base: {f['parent_equals_pr_base']}; "
              f"patch applies to the parent: {f['patch_applies_to_parent']} (fuzz {f['patch_fuzz']})")


if __name__ == '__main__':
    main()
