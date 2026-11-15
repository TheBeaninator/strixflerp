#!/usr/bin/env bash
# One GPU job of ATTN-PROJ (criteria gate/attn-proj.toml):
#   bash gate/attn_proj_run.sh [--backend kbench|queue] tune|kernel|model|kernel+model
#   kbench (default)  tune and kernel on bench box K through pod-kbench (no control-node GPU, no lease here, no gpu-queue);
#                     results in gate/results/attn-proj/kbench/. `model` (whole-model processes on the 35B weights) is
#                     refused there: it stays on the queue backend.
#   queue             the old path, for tools/gpu-queue.sh: control-node GPU only. Refuses a model path off /data/models;
#                     waits (up to 90 min) for loadavg <= 6; holds control-gpu as agent for the job (released on
#                     every exit); records loadavg at start and end.
set -u
cd /data/projects/strixflerp
backend=kbench; [ "${1:-}" = "--backend" ] && { backend=$2; shift 2; }
case $backend in kbench|queue) ;; *) echo "backend must be kbench or queue"; exit 2;; esac
what=$1; out=gate/results/attn-proj; [ $backend = kbench ] && out=$out/kbench; mkdir -p $out
if [ $backend = kbench ]; then
  case $what in *model*) echo "the model stage needs control-node's GPU and the 35B weights: --backend queue"; exit 2;; esac
  rc=0
  for s in ${what//+/ }; do python3 gate/attn_proj.py --backend kbench $s || rc=$?; done
  python3 gate/attn_proj.py --backend kbench summary
  exit $rc
fi
m=$(python3 -c "import sys; sys.path.insert(0, 'gate'); import attn_proj as a; print(a.M35)")
case $m in /data/models/*) ;; *) echo "model path $m is not on /data/models; refusing"; exit 1;; esac
[ -f "$m" ] || { echo "model $m is not there; refusing"; exit 1; }
for i in $(seq 180); do awk '{exit !($1 <= 6)}' /proc/loadavg && break; sleep 30; done
echo "$what load_start $(date -u +%FT%TZ) $(cat /proc/loadavg)" >> $out/load.txt
pod-lease acquire control-gpu --owner agent --why "strixflerp ATTN-PROJ $what, minutes" --ttl 5400 --probe pid=control:$$ || { echo "leased to someone else"; exit 1; }
trap 'pod-lease release control-gpu --owner agent' EXIT
rc=0
for s in ${what//+/ }; do python3 gate/attn_proj.py --backend queue $s || rc=$?; done
python3 gate/attn_proj.py --backend queue summary
echo "$what load_end $(date -u +%FT%TZ) $(cat /proc/loadavg) rc $rc" >> $out/load.txt
exit $rc
