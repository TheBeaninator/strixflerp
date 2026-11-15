#!/bin/bash
# arm.sh ARM WEIGHTS (runs on B): transient user unit ab2-ARM on :5310 with B's live args, client, stop unit.
set -u; A=$1; W=$2; R=/cache/ab2-mtpq4B; O=$R/$A; mkdir -p $O $R/pc-$A
export XDG_RUNTIME_DIR=/run/user/$(id -u)
BIN=/data/models/strix-infer/runtime/bin/strix_server.v070tg
ARGS="--ngram /data/models/strix-infer/converted/qwen38-fnext-ngram-int8/ngram.table --tokenizer /data/models/strix-infer/tok/tokenizer.json --generation-config /data/models/strix-infer/tok/generation_config.json --host 0.0.0.0 --capacity 262144 --chunk 8192 --mtp on --prompt-cache-gib 32 --prompt-cache-disk-free-gib 8 --prompt-cache-idle-s 150 --model-id qwen3.8-flash-next --total-slots 2 --cache-xfer on --steer-dir /data/models/strix-infer/steer --prompt-cache-ram-margin-gib 10"
U=ab2-$A; t0=$(date +%s.%N)
systemd-run --user --collect --unit=$U -p LimitMEMLOCK=infinity --setenv=LD_LIBRARY_PATH=/data/models/strix-infer/runtime/lib $BIN --weights $W --port 5310 --prompt-cache-dir $R/pc-$A --logprobs on --log-level debug $ARGS || exit 4
ok=0; for i in $(seq 1 360); do curl -sf -m 2 http://127.0.0.1:5310/health >/dev/null && { ok=1; break; }; systemctl --user is-active -q $U || break; sleep 0.5; done
if [ $ok != 1 ]; then echo "NOT HEALTHY $A"; journalctl --user -u $U --no-pager | tail -30; systemctl --user stop $U; exit 3; fi
t1=$(date +%s.%N); echo "load_s $(echo "$t1 - $t0" | bc)" | tee $O/load
curl -s http://127.0.0.1:5310/health > $O/health
timeout 480 python3 $R/strixw_mtpq4_ab2.py 5310 $A $O/client.json > $O/client.log 2>&1; rc=$?; echo "client rc $rc" | tee -a $O/client.log
journalctl --user -u $U --no-pager | grep -v '^\s*$' > $O/journal.txt
systemctl --user stop $U; echo "stopped $U"; exit $rc
