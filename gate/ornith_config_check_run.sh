#!/usr/bin/env bash
# One GPU job (tools/gpu-queue.sh): gate/ornith_config_check.py on control-node's GPU, 35B from /data/models, as agent.
set -u
cd "$(dirname "$0")/.."
# identity only (no timing is judged), so no loadavg wait
pod-lease acquire control-gpu --owner agent --why "strixflerp Ornith config identity check (35B world 4 loopback, 4 arms), minutes" --ttl 7200 --probe pid=control:$$ || { echo "leased to someone else"; exit 1; }
trap 'pod-lease release control-gpu --owner agent' EXIT
python3 gate/ornith_config_check.py "$@"
