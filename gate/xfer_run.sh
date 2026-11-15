#!/usr/bin/env bash
# One GPU job of the SHAPE-TRANSFERS first experiments (criteria gate/xfer-r*.toml), for tools/gpu-queue.sh:
#   bash gate/xfer_run.sh r6|r1|r2a|r2b
# Waits (up to 90 min) for loadavg <= 6, holds control-gpu for the job, records loadavg at start and end.
set -u
cd /data/projects/strixflerp
what=$1; out=gate/results/xfer-$what; mkdir -p $out
for i in $(seq 180); do awk '{exit !($1 <= 6)}' /proc/loadavg && break; sleep 30; done
echo "load_start $(cat /proc/loadavg)" > $out/load.txt
pod-lease acquire control-gpu --owner agent --why "SHAPE-TRANSFERS $what first experiment, minutes" --ttl 1800 --probe pid=control:$$ || { echo "leased to someone else"; exit 1; }
M=/data/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-Q4_K_M.gguf
KIT=build/dsv41-flash.strixfold-tp2/tree/zig-out/native-hip/share/tensorfold/hip/gfx1151/dsv41
P=/data/models/tools/therock-10.2
run() { timeout 1500 ionice -c3 nice -n 19 env OMP_NUM_THREADS=4 "$@"; }
case $what in
  r6) run build/xfer-r6/xfer-verify-bench --model $M --ids-file gate/results/qwen36-35b-a3b.flerp-moe-p4/records.ids --context 4096 --repeats 30 > $out/verify.jsonl 2> $out/verify.err ;;
  r1) for t in 1 4 8; do
        run build/xfer-r1/xfer-r1-splitk --case q35-gateup --rows 512 --k 2048 --experts 256 --slots 8 --tokens $t --pair --gguf $M --off-a 6140566368 --off-b 6294256480
        run build/xfer-r1/xfer-r1-splitk --case ornith-gateup --rows 1024 --k 4096 --experts 64 --slots 10 --tokens $t --pair
        run build/xfer-r1/xfer-r1-splitk --case ornith-down --rows 4096 --k 1024 --experts 64 --slots 10 --tokens $t --xshare 1
      done > $out/splitk.jsonl 2> $out/splitk.err ;;
  r2a|r2b) arm=${what#r2}; kit=$KIT; [ $arm = b ] && kit=build/exl3-pf-bytes/kit-nodecode
      run env LD_LIBRARY_PATH=$P/lib $P/bin/rocprofv3 --kernel-trace --kernel-include-regex "pf_experts" --output-format csv -d $out/trace -o t -- \
        python3 gate/exl3_pf_bytes.py --kit $kit --arm $arm --rows 128,256,512,1024,2048 --k2 6,4 --repeats 3 --no-lease --out $out/bench.json > $out/run.txt 2>&1 ;;
esac
rc=$?
echo "load_end $(cat /proc/loadavg)" >> $out/load.txt
pod-lease release control-gpu --owner agent
exit $rc
