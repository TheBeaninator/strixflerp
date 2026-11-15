#!/usr/bin/env bash
# One GPU job of ORNITH-KERN3 (gate/ornith-kern3.toml): bash gate/ornith_kern3_run.sh [--backend kbench|queue]
#   kbench (default)  bench box K through pod-kbench (no control-node GPU, no control-gpu lease, no gpu-queue); results in
#                     gate/results/ornith-kern3/kbench/. Run it directly on control-node: pod-kbench takes the bench:K lease itself.
#   queue             the old path, for tools/gpu-queue.sh: control-node GPU only; waits (up to 90 min) for loadavg <= 6;
#                     holds control-gpu as agent for the job; results in gate/results/ornith-kern3/.
set -u
cd /data/projects/strixflerp
backend=kbench; [ "${1:-}" = "--backend" ] && { backend=$2; shift 2; }
case $backend in kbench|queue) ;; *) echo "backend must be kbench or queue"; exit 2;; esac
out=gate/results/ornith-kern3; [ $backend = kbench ] && out=$out/kbench; mkdir -p $out
if [ $backend = kbench ]; then
  python3 gate/ornith_kern3.py --backend kbench kernel; rc=$?
  python3 gate/ornith_kern3.py --backend kbench summary
  exit $rc
fi
for i in $(seq 180); do awk '{exit !($1 <= 6)}' /proc/loadavg && break; sleep 30; done
echo "kernel load_start $(date -u +%FT%TZ) $(cat /proc/loadavg)" >> $out/load.txt
pod-lease acquire control-gpu --owner agent --why "strixflerp ORNITH-KERN3 kernel bench, minutes" --ttl 3600 --probe pid=control:$$ || { echo "leased to someone else"; exit 1; }
trap 'pod-lease release control-gpu --owner agent' EXIT
python3 gate/ornith_kern3.py --backend queue kernel; rc=$?
python3 gate/ornith_kern3.py --backend queue summary
echo "kernel load_end $(date -u +%FT%TZ) $(cat /proc/loadavg)" >> $out/load.txt
exit $rc
