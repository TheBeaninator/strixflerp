#!/usr/bin/env python3
"""Turn the hipfire compile gate's failures into docs/harvest/data/hipfire/not_gfx1151.json: the units that are not
gfx1151 compile units, each with its class and the compiler's own message. Only failures whose message matches one of
the rules below are excluded; anything else stays in units.json and keeps its module at 'harvested'.
Run after gate/compile/run.py --origin hipfire and tools/pod-pull.sh, then re-run harvest_hipfire.py and the gate."""
import glob, json, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
RULES = [
    (r'requires --offload-arch=gfx1201|gfx1201 only', 'arch', 'the source refuses any arch but gfx1201 (#error)'),
    (r"needs target feature dot1-insts", 'arch', 'uses __builtin_amdgcn_sdot4 (dot1-insts), which gfx1151 does not have'),
    (r"cuda_fp16\.h|ggml-cuda/dequantize\.cuh", 'arch', 'ggml shim shipped only as a gfx1201 prebuilt object (kernels/ggml_mmvq_shim_gfx1201.co); needs CUDA shim headers that are not in the tree'),
    (r"undeclared identifier 'fwht_shfl_forward_512'", 'origin-broken', 'the origin unit calls fwht_shfl_forward_512, which no file in the tree at efa8fbb defines (turbo_common.h has the 128 and 256 forms); its only launcher is gemma4_ext.rs, a left-out family'),
]
path = os.path.join(HERE, 'data', 'hipfire', 'not_gfx1151.json')
out = json.load(open(path)) if os.path.exists(path) else {}
for p in sorted(glob.glob(os.path.join(REPO, 'gate/results/compile/kernel.hipfire.*.json'))):
    j = json.load(open(p))
    for r in j['results']:
        if r['rc'] == 0:
            continue
        msg = (r.get('stderr') or '').splitlines()[0] if r.get('stderr') else ''
        msg = re.sub(r'^\S+?:\d+:\d+: ', '', msg)
        for rx, cls, why in RULES:
            if re.search(rx, r.get('stderr') or ''):
                out[r['unit']] = {'class': cls, 'reason': why, 'compiler': msg[:240], 'toolchain': j['toolchain'], 'module': j['module']}
                break
        else:
            print('unclassified failure kept in units.json:', j['module'], r['unit'], msg[:160])
json.dump(out, open(path, 'w'), indent=1, sort_keys=True)
import collections
print(collections.Counter(v['class'] for v in out.values()))
