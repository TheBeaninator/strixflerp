#!/usr/bin/env bash
# One GPU job (tools/gpu-queue.sh): gate/ornith_ctx_check.py on control-node's GPU, 35B from /data/models, as agent.
set -u
cd "$(dirname "$0")/.."
pod-lease acquire control-gpu --owner agent --why "strixflerp ORNITH-CTX-RUN (35B world 4 loopback, 64K context, 20K prompt), minutes" --ttl 2220 --probe pid=control:$$ || { echo "leased to someone else"; exit 1; }
trap 'pod-lease release control-gpu --owner agent' EXIT
python3 gate/ornith_ctx_check.py "$@"
