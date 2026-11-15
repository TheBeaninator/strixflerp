#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# flerp (runtime.flerp.vision_sidecar): an adapted copy of Strixfold pull request #6 (766df75, aercanozcan),
# tools/dsv41-hip/verify_vision.py. The origin's file is catalog/strixfold_pr6/tool/prepare_vision, unchanged. Every
# change here is marked with this module's id: the package versions come from a named record of vision-env.json
# in place of three literals, and the admission prints the FLERP_VISION_ENV value the worker is to be started with.
"""Fail-closed admission of the optional CPU tower before image-enabled startup."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def checked_file(path, expected):
    if not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('vision input requires a SHA256 pin')
    p = Path(path)
    if not p.is_file() or p.is_symlink():
        raise ValueError('vision input must be a regular file')
    with p.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
            raise ValueError('vision input SHA256 mismatch')


# flerp (runtime.flerp.vision_sidecar): the recorded environments. A record is python (major.minor, or null for
# "not checked", as the origin), torch, numpy, pillow, status and evidence. Admitted: status "pass" (a row gate of
# this tree passed with it; evidence names the result) and "origin-pin" (the pull request's own pin). A "candidate"
# is admitted only when the caller says so, which is how its gate runs it.
ENV_FILE = Path(__file__).resolve().parents[2] / 'vision-env.json'
VERSION = re.compile(r'[0-9A-Za-z.+_-]{1,64}')


def environment(name, env_file=ENV_FILE, allow_candidate=False):
    records = json.loads(Path(env_file).read_text())['environments']
    if name not in records:
        raise ValueError('no vision environment named ' + repr(name) + ' in ' + str(env_file))
    record = records[name]
    for key in ('torch', 'numpy', 'pillow'):
        if not isinstance(record.get(key), str) or not VERSION.fullmatch(record[key]):
            raise ValueError('vision environment ' + name + ': bad ' + key)
    if record.get('python') is not None and not re.fullmatch(r'3\.[0-9]{1,2}', str(record['python'])):
        raise ValueError('vision environment ' + name + ': bad python')
    status = record.get('status')
    if status == 'pass' and not record.get('evidence'):
        raise ValueError('vision environment ' + name + ' says pass without evidence')
    if status not in ('pass', 'origin-pin') and not (status == 'candidate' and allow_candidate):
        raise ValueError('vision environment ' + name + ' has status ' + repr(status) + ': not admitted')
    return record


def setting(record):
    """The worker's FLERP_VISION_ENV for a record."""
    keys = ('python', 'torch', 'numpy')
    return ','.join(key + '=' + record[key] for key in keys if record.get(key) is not None)


CHECK = ('import sys,torch,numpy,PIL\n'
         'actual=("%d.%d"%sys.version_info[:2],torch.__version__,numpy.__version__,PIL.__version__)\n'
         'want=sys.argv[1:5]\n'
         'bad=[(a,w) for a,w in zip(actual,want) if w!="-" and a!=w]\n'
         'assert not bad, bad\n')


def verify(bias, bias_sha, python='', weights='', weights_sha='', record=None):
    record = record or environment('origin')  # flerp (runtime.flerp.vision_sidecar)
    checked_file(bias, bias_sha)
    if python or weights or weights_sha:
        if not all((python, weights, weights_sha)):
            raise ValueError('vision encoder needs Python, weights and their SHA256 together')
        checked_file(weights, weights_sha)
        if not Path(python).is_absolute() or not Path(python).is_file():
            raise ValueError('vision Python must be an absolute executable path')
        # flerp (runtime.flerp.vision_sidecar): the origin asserted torch 2.9.0+cpu, numpy 2.3.5, PIL 12.3.0 here.
        subprocess.run([python, '-I', '-c', CHECK, record.get('python') or '-', record['torch'], record['numpy'],
                        record['pillow']], check=True, timeout=30, stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--bias', required=True)
        parser.add_argument('--bias-sha256', required=True)
        parser.add_argument('--python', default='')
        parser.add_argument('--weights', default='')
        parser.add_argument('--weights-sha256', default='')
        # flerp (runtime.flerp.vision_sidecar): three options and the printed setting.
        parser.add_argument('--env', default='origin', help='a record of vision-env.json')
        parser.add_argument('--env-file', default=str(ENV_FILE))
        parser.add_argument('--allow-candidate', action='store_true', help='admit a record no gate has passed yet')
        args = parser.parse_args()
        record = environment(args.env, args.env_file, args.allow_candidate)
        verify(args.bias, args.bias_sha256, args.python, args.weights, args.weights_sha256, record)
        print('FLERP_VISION_ENV=' + setting(record))
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
        sys.exit('vision preflight: ' + str(exc))
