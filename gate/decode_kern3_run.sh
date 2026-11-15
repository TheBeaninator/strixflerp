#!/usr/bin/env bash
# One GPU job of DECODE-KERN3 (criteria gate/decode-kern3.toml):
#   bash gate/decode_kern3_run.sh [--backend kbench|queue] kernel|kernel-cold|kernel-rerun|model|sweep
#   kbench (default)  kernel, kernel-cold, kernel-rerun on bench box K through pod-kbench (no control-node GPU, no lease
#                     here, no gpu-queue), the synthetic-code cases only (the real-weight cases read the 35B GGUF);
#                     results in gate/results/decode-kern3/kbench/. model and sweep are refused there.
#   queue             the old path, for tools/gpu-queue.sh. Refuses a model path off /data/models; waits (up to 90 min) for loadavg <= 6; holds control-gpu as agent
# for the job (released on every exit); records loadavg at start and end.
set -u
cd /data/projects/strixflerp
backend=kbench; [ "${1:-}" = "--backend" ] && { backend=$2; shift 2; }
case $backend in kbench|queue) ;; *) echo "backend must be kbench or queue"; exit 2;; esac
what=$1; out=gate/results/decode-kern3; [ $backend = kbench ] && out=$out/kbench; mkdir -p $out
if [ $backend = kbench ]; then
  case $what in kernel|kernel-cold|kernel-rerun) ;; *) echo "$what needs control-node's GPU and the 35B weights: --backend queue"; exit 2;; esac
  python3 gate/decode_kern3.py --backend kbench $what; rc=$?
  python3 gate/decode_kern3.py --backend kbench summary
  exit $rc
fi
m=$(python3 -c "import sys; sys.path.insert(0, 'gate'); import decode_kern3 as d; print(d.M35)")
case $m in /data/models/*) ;; *) echo "model path $m is not on /data/models; refusing"; exit 1;; esac
[ -f "$m" ] || { echo "model $m is not there; refusing"; exit 1; }
for i in $(seq 180); do awk '{exit !($1 <= 6)}' /proc/loadavg && break; sleep 30; done
echo "$what load_start $(date -u +%FT%TZ) $(cat /proc/loadavg) model $m" >> $out/load.txt
pod-lease acquire control-gpu --owner agent --why "strixflerp DECODE-KERN3 $what, minutes" --ttl 3600 --probe pid=control:$$ || { echo "leased to someone else"; exit 1; }
trap 'pod-lease release control-gpu --owner agent' EXIT
if [ "$what" = sweep ]; then bash gate/decode_kern3_sweep.sh run; else python3 gate/decode_kern3.py --backend queue $what; fi
rc=$?
echo "$what load_end $(date -u +%FT%TZ) $(cat /proc/loadavg)" >> $out/load.txt
exit $rc
