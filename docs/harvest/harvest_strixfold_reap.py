#!/usr/bin/env python3
"""Harvest the passive Edge-REAP collector (a pod-local branch of the StrixFold fork) into catalog/strixfold_reap.

    python3 docs/harvest/harvest_strixfold_reap.py --git <a clone of the pod fork> --head <branch sha>

The collector is a change set over the pod's TP4 serving build 003f679c (catalog/strixfold_tp4): REAP_STATS, default
off, adds per-(layer, expert) routing statistics in the format of owner's strix_moe_surgery.py export_report (see
catalog/flerp/tool/edge_reap_keepset and docs/EDGE-REAP.md). Like strixfold_tp4 it is a sub-origin: an origin has one
commit, the change set's commit is a descendant of 003f679c, and the TP4 recipe must keep its reference. The harvest
takes every file the change set touches between BASE and HEAD that the build reads (the launcher under tools/ and the
docs are not built: the recipe builds the engine, not the launcher) into one overlay module whose files sit at their
origin paths (assemble_zig [build.overlay] root "."), plus overlay.json (per file: the sha256 at BASE, or null for an
added file, the sha256 at HEAD, the commits that touched it). The origin's experiment doc becomes the module's
DESIGN.md. It reads the git objects only; it runs nothing from the tree.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORIGIN, PARENT = 'strixfold_reap', 'strixfold_tp4'
BASE = '03d67fc1'                 # the base branch: 003f679c28b1 (TP4-SERVE1) + two server commits, the change set's base
TP4 = '003f679c'                  # catalog/strixfold_tp4's commit
REPO_URL = 'https://github.com/Shoestring-dev/Strixfold'
MODULE = 'runtime.strixfold_reap.edge_reap'
DOC = 'docs/experiments/tp4-reap-stats.md'
SKIP = ('docs/', 'tools/')
# Licence texts copied beside ORIGIN.toml from the head commit (ORIGIN.toml [licence] files): the root texts and
# LICENSES/, which THIRD_PARTY_NOTICES.md cites.
LICENCE_FILES = ('LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md', 'LICENSES/Apache-2.0.txt', 'LICENSES/CUTLASS-BSD-3-Clause.txt',
                 'LICENSES/MIT.txt', 'LICENSES/MiaAI-Lab-MIT.txt', 'LICENSES/Pillow-MIT-CMU.txt',
                 'LICENSES/PyTorch-BSD-3-Clause.txt')
CARD_TAG = re.compile(r'\s*\((?:pod )?card [0-9a-f]{10}\)')   # internal work-item tags in fork commit subjects


def git(g, *args):
    return subprocess.run(['git', '-C', g, *args], check=True, capture_output=True).stdout


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--git', required=True, help='a clone of the pod fork holding --head')
    ap.add_argument('--head', required=True)
    ap.add_argument('--branch', default='TP4-REAP-STATS over 03d67fc1, a branch local to the pod fork')
    a = ap.parse_args(argv)
    head = git(a.git, 'rev-parse', a.head).decode().strip()
    base = git(a.git, 'rev-parse', BASE).decode().strip()
    tp4 = git(a.git, 'rev-parse', TP4).decode().strip()
    names = [n for n in git(a.git, 'diff', '--name-only', f'{tp4}..{head}').decode().split('\n') if n]
    files = sorted(n for n in names if not n.startswith(SKIP))
    mdir = os.path.join(REPO, 'catalog', ORIGIN, 'runtime', 'edge_reap')
    if os.path.isdir(os.path.join(mdir, 'src')):
        shutil.rmtree(os.path.join(mdir, 'src'))
    overlay = {}
    for p in files:
        new = git(a.git, 'show', f'{head}:{p}')
        try:
            old = git(a.git, 'show', f'{tp4}:{p}')
        except subprocess.CalledProcessError:
            old = None
        log = git(a.git, 'log', '--format=%h %s', f'{tp4}..{head}', '--', p).decode().strip().split('\n')
        overlay[p] = {'tp4_sha256': sha(old) if old is not None else None, 'sha256': sha(new),
                      'commits': [CARD_TAG.sub('', c) for c in log if c]}
        dst = os.path.join(mdir, 'src', p)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, 'wb') as f:
            f.write(new)
    with open(os.path.join(mdir, 'overlay.json'), 'w', encoding='utf-8') as f:
        json.dump({'origin': ORIGIN, 'parent': PARENT, 'tp4_commit': tp4, 'base': base, 'head': head,
                   'files': overlay}, f, indent=1, sort_keys=True)
        f.write('\n')
    for p in LICENCE_FILES:
        dst = os.path.join(REPO, 'catalog', ORIGIN, p)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, 'wb') as f:
            f.write(git(a.git, 'show', f'{head}:{p}'))
    doc = git(a.git, 'show', f'{head}:{DOC}').decode()
    with open(os.path.join(mdir, 'DESIGN.md'), 'w', encoding='utf-8') as f:
        f.write(f'<!-- {DOC} at {head[:12]}, copied by docs/harvest/harvest_strixfold_reap.py -->\n' + doc)
    print(json.dumps({'head': head, 'files': len(files), 'added': sum(v['tp4_sha256'] is None for v in overlay.values())}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
