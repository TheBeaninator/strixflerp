#!/usr/bin/env bash
# One GPU job (tools/gpu-queue.sh): gate/ornith_pcache_check.py on control-node's GPU, 35B from /data/models, as agent.
set -u
cd "$(dirname "$0")/.."
pod-lease acquire control-gpu --owner agent --why "strixflerp ORNITH-PCACHE step 0 (35B world 4 loopback, prefix reuse, 20K thread), minutes" --ttl 3600 --probe pid=control:$$ || { echo "leased to someone else"; exit 1; }
trap 'pod-lease release control-gpu --owner agent' EXIT
python3 gate/ornith_pcache_check.py "$@"
