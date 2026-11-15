#!/bin/bash
# GPU-queue job: snapshot the tap, pick the replay list, run the origin twice from an empty cache.
set -u; umask 077
R=/data/pod/flerp-replay; W=$R/selfid; mkdir -p $W
ln -sfn $R/corpus-v1 $W/snap   # selection reads the frozen corpus-v1
python3 - "$W" <<'P'
import glob, json, sys, os
sys.path.insert(0, "/data/pod/flerp-replay/tools")
from replay import classes, primary
w = sys.argv[1]; ex = []
for f in sorted(glob.glob(w + "/snap/strixite/*.json")):
    e = json.load(open(f))
    if e["path"].startswith("/v1/chat/completions") and e.get("status") == 200:
        rec = json.loads(e.get("response_raw") or "{}")
        ex.append((f, primary(classes(e["request"], rec)), (rec.get("usage") or {}).get("prompt_tokens", 0)))
# stratified: up to N per class, recorded order kept
N = int(os.environ.get("PER_CLASS", "4")); seen = {}; pick = []
for f, c, n in ex:
    if seen.get(c, 0) < N and n < 140000:
        seen[c] = seen.get(c, 0) + 1; pick.append(f)
open(w + "/list.txt", "w").write("\n".join(pick[:int(os.environ.get("MAXN", "14"))]) + "\n")
print(seen)
P
for r in A B; do python3 $R/tools/replay.py run --list $W/list.txt --out $W --run $r --max-tokens 256 --pairs 3 || exit 1; done
mkdir -p /data/projects/strixflerp/gate/results/qwen38-flash-next.strixite.selfreplay && python3 $R/tools/compare.py $W /data/projects/strixflerp/gate/results/qwen38-flash-next.strixite.selfreplay/summary.md
