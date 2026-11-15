#!/usr/bin/env bash
# The 35B decode kernel profile (gate/decode_profile.py) as one GPU job for tools/gpu-queue.sh:
#   bash gate/decode_profile_run.sh
# Waits (up to 60 min) for loadavg <= 6, holds control-gpu for the job (released on every exit), records loadavg.
set -u
cd /data/projects/strixflerp
out=${DECODE_PROFILE_RES:-gate/results/decode-profile}; mkdir -p $out
# The model must come from the NVMe copy, never cold storage (/data/ark): check the path the runner uses.
m=$(python3 -c "import sys; sys.path.insert(0, 'gate'); import decode_profile as d; print(d.M35)")
case $m in /data/models/*) ;; *) echo "model path $m is not on /data/models; refusing"; exit 1;; esac
[ -f "$m" ] || { echo "model $m is not there yet; refusing"; exit 1; }
echo "model $m" >> $out/load.txt
for i in $(seq 120); do awk '{exit !($1 <= 6)}' /proc/loadavg && break; sleep 30; done
echo "load_start $(date -u +%FT%TZ) $(cat /proc/loadavg)" >> $out/load.txt
pod-lease acquire control-gpu --owner agent --why "strixflerp 35B decode kernel profile, minutes" --ttl 1800 --probe pid=control:$$ || { echo "leased to someone else"; exit 1; }
trap 'pod-lease release control-gpu --owner agent' EXIT
python3 gate/decode_profile.py run; rc=$?
python3 gate/decode_profile.py summary || rc=1
echo "load_end $(date -u +%FT%TZ) $(cat /proc/loadavg)" >> $out/load.txt
exit $rc
