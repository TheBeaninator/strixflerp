#!/bin/bash
# arm.sh ARM WEIGHTS [--skip32k]: start the arm unit on :5310, time load, record memory, run client, stop unit.
set -u; A=$1; W=$2; S=${3:-}; O=/cache/ab-strixw/$A; mkdir -p $O /cache/ab-pc-$A
export XDG_RUNTIME_DIR=/run/user/1000
BIN=/data/models/strix-infer/runtime/bin/strix_server.v070tg
ARGS="--ngram /data/models/strix-infer/converted/qwen38-fnext-ngram-int8/ngram.table --tokenizer /data/models/strix-infer/tok/tokenizer.json --generation-config /data/models/strix-infer/tok/generation_config.json --host 0.0.0.0 --capacity 262144 --chunk 8192 --mtp on --prompt-cache-gib 32 --prompt-cache-disk-free-gib 8 --prompt-cache-idle-s 150 --model-id qwen3.8-flash-next --total-slots 2 --cache-xfer on --steer-dir /data/models/strix-infer/steer --prompt-cache-ram-margin-gib 10"
cat /sys/class/drm/card*/device/mem_info_gtt_used > $O/mem_before
t0=$(date +%s.%N)
systemd-run --user --collect --unit=ab-strixw-$A -p LimitMEMLOCK=infinity --setenv=LD_LIBRARY_PATH=/data/models/strix-infer/runtime/lib $BIN --weights $W --port 5310 --prompt-cache-dir /cache/ab-pc-$A --logprobs on $ARGS
for i in $(seq 1 600); do curl -sf -m 2 http://127.0.0.1:5310/health >/dev/null && break; systemctl --user is-active -q ab-strixw-$A || { echo DIED; journalctl --user -u ab-strixw-$A --no-pager | tail -30; exit 3; }; sleep 0.5; done
t1=$(date +%s.%N); echo "load_s $(echo "$t1 - $t0" | bc)" | tee $O/load
curl -s http://127.0.0.1:5310/health > $O/health
sleep 3; cat /sys/class/drm/card*/device/mem_info_gtt_used /sys/class/drm/card*/device/mem_info_vram_used > $O/mem_after; grep -E "MemAvailable" /proc/meminfo >> $O/mem_after
rocm-smi --showmeminfo vram gtt > $O/rocm-smi 2>&1
timeout 1200 python3 /cache/ab-strixw/strixw_layout_ab.py 5310 $A $O/client.json $S > $O/client.log 2>&1; echo "client rc $?" >> $O/client.log
journalctl --user -u ab-strixw-$A --no-pager > $O/journal.txt
systemctl --user stop ab-strixw-$A; echo stopped
