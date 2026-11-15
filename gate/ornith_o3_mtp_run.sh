#!/usr/bin/env bash
# One GPU job of ORNITH-O3 step one (criteria gate/ornith-o3-mtp.toml), for tools/gpu-queue.sh:
#   bash gate/ornith_o3_mtp_run.sh off|on|staged|llama
# control-node GPU only (no worker, no hold). Refuses a model path off /data/models; waits (up to 90 min) for loadavg
# <= 6; holds control-gpu as agent for the job (released on every exit); records loadavg at start and end.
set -u
cd /data/projects/strixflerp
what=$1; out=gate/results/ornith-o3-mtp; mkdir -p $out
m=$(python3 -c "import tomllib; print(tomllib.load(open('gate/ornith-o3-mtp.toml','rb'))['proxy']['model'])")
case $m in /data/models/*) ;; *) echo "model path $m is not on /data/models; refusing"; exit 1;; esac
[ -f "$m" ] || { echo "model $m is not there; refusing"; exit 1; }
for i in $(seq 180); do awk '{exit !($1 <= 6)}' /proc/loadavg && break; sleep 30; done
echo "$what load_start $(date -u +%FT%TZ) $(cat /proc/loadavg)" >> $out/load.txt
pod-lease acquire control-gpu --owner agent --why "strixflerp ORNITH-O3 $what (35B MTP probe), minutes" --ttl 5400 --probe pid=control:$$ || { echo "leased to someone else"; exit 1; }
trap 'pod-lease release control-gpu --owner agent' EXIT
python3 gate/ornith_o3_mtp.py $what; rc=$?
echo "$what load_end $(date -u +%FT%TZ) $(cat /proc/loadavg) rc $rc" >> $out/load.txt
exit $rc
