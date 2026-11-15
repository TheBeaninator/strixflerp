#!/usr/bin/env bash
# One GPU job of ORNITH-CRUCIBLE item 2 (gate/ornith-crucible.toml [item2_proxy]), for tools/gpu-queue.sh, in the export
# /data/projects/strixflerp-ocr: bash gate/ornith_crucible_run.sh probe [ARMS...] | spec (item 4)
# control-node GPU only (no worker, no hold). Waits (up to 10 min) for loadavg <= 6; holds control-gpu as agent
# for the job (released on every exit, TTL 75 min); records loadavg at start and end. The load wait is 10 min, not
# 90: on 2026-11-14 control-node sat at load 14-15 from NFS I/O (D state), not CPU, and no criterion of this job reads it.
set -u
cd "$(dirname "$0")/.."
out=gate/results/ornith-crucible/probe; mkdir -p $out
for i in $(seq 20); do awk '{exit !($1 <= 6)}' /proc/loadavg && break; sleep 30; done
echo "probe load_start $(date -u +%FT%TZ) $(cat /proc/loadavg)" >> $out/load.txt
pod-lease acquire control-gpu --owner agent --why "strixflerp ORNITH-CRUCIBLE $1 (35B, one load per arm), minutes" --ttl 4500 --probe pid=control:$$ || { echo "leased to someone else"; exit 1; }
trap 'pod-lease release control-gpu --owner agent' EXIT
what=$1; shift
python3 gate/ornith_crucible.py $what "$@"; rc=$?
echo "probe load_end $(date -u +%FT%TZ) $(cat /proc/loadavg) rc $rc" >> $out/load.txt
exit $rc
