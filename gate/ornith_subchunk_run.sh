#!/usr/bin/env bash
# One GPU job (tools/gpu-queue.sh): gate/ornith_subchunk.py gen then serve on control-node's GPU (35B from /data/models),
# behind gate/loadwait.sh, holding the control-gpu lease across both phases so no other job starts between launches.
set -u
cd "$(dirname "$0")/.."
pod-lease acquire control-gpu --owner agent --why "strixflerp ORNITH-SUBCHUNK 6abf28258c (35B world 4 tcp: generator + 3 serve arms), ~25 min" --ttl 2100 --probe pid=control:$$ || { echo "leased to someone else"; exit 1; }
trap 'pod-lease release control-gpu --owner agent' EXIT
bash gate/loadwait.sh python3 gate/ornith_subchunk.py gen
rc1=$?
python3 gate/ornith_subchunk.py serve
rc2=$?
python3 gate/ornith_subchunk.py judge
echo "gen rc $rc1 serve rc $rc2"
