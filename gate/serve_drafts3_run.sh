#!/usr/bin/env bash
# One GPU job (tools/gpu-queue.sh): gate/serve_drafts3.py on control-node's GPU, 35B from /data/models, as agent.
set -u
cd "$(dirname "$0")/.."
pod-lease acquire control-gpu --owner agent --why "strixflerp SERVE-DRAFTS3 3f6d20b703 (35B world 4 loopback, 4 arms), minutes" --ttl 4500 --probe pid=control:$$ || { echo "leased to someone else"; exit 1; }
trap 'pod-lease release control-gpu --owner agent' EXIT
python3 gate/serve_drafts3.py "$@"
