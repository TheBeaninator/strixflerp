#!/usr/bin/env bash
# One GPU job of the SHAPE-TRANSFERS R1 fold (criteria gate/xfer-r1-fold.toml), for tools/gpu-queue.sh:
#   bash gate/xfer_r1_fold_run.sh remeasure|kernel|model
# Waits (up to 90 min) for loadavg <= 6, holds control-gpu for the job, records loadavg at start and end.
set -u
cd /data/projects/strixflerp
what=$1; out=gate/results/xfer-r1-fold; mkdir -p $out
for i in $(seq 180); do awk '{exit !($1 <= 6)}' /proc/loadavg && break; sleep 30; done
echo "$what load_start $(date -u +%FT%TZ) $(cat /proc/loadavg)" >> $out/load.txt
pod-lease acquire control-gpu --owner agent --why "SHAPE-TRANSFERS R1 fold $what, minutes" --ttl 2400 --probe pid=control:$$ || { echo "leased to someone else"; exit 1; }
python3 gate/xfer_r1_fold.py $what
rc=$?
echo "$what load_end $(date -u +%FT%TZ) $(cat /proc/loadavg)" >> $out/load.txt
pod-lease release control-gpu --owner agent
exit $rc
