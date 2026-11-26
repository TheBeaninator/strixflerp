#!/bin/bash
# FACTORY-REPLAY 2587bc1599 baseline: the command of the win_run job (runs on control-node with the grant in its
# environment: WINDOW_IPS, KIT_DIR). Starts gate/replay/serve.sh on the held worker, plays the cut set against it,
# keeps the engine's log and counters, stops the engine. gate/factory-replay.toml [amendment_2026_11_17] has the why.
set -u
P=/data/pod/plans/2587bc1599; WT=$P/wt; OUT=$P/baseline
IP=${WINDOW_IPS:?no window}; KIT=${KIT_DIR:?no kit}; PORT=5311; N=${FIRST_STREAMS:-13}; LIMIT=${CLIENT_LIMIT_S:-5400}
case $IP in *" "*) echo "one worker only, got: $IP"; exit 2;; esac
R() { ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$IP" "$@"; }
say() { echo "$(date -u +%FT%TZ) $*"; }
mkdir -p $OUT
# the pin must still be the line serve.sh spells out
want=$(pod-engine want $IP 2>/dev/null)
for k in 'SERVING=strixite_p16fold' 'strix_server.p16fold' 'CHUNK=16384' 'WDIR=qwen38-fnext-p16' 'SESSIONS=2'; do
  case $want in *"$k"*) ;; *) say "REFUSED: pod-engine want $IP has no '$k': the pin moved, serve.sh must follow"; exit 3;; esac
done
# the window stops the lane engine; two engines do not fit one GPU
t0=$(date +%s)
while R "pgrep -f 'strix_server.*--port 5300'" >/dev/null 2>&1; do
  [ $(( $(date +%s) - t0 )) -gt 180 ] && { say "REFUSED: the lane engine is still up on $IP 180 s into the window"; exit 5; }
  sleep 5
done
stop() { R "pkill -f '/serve.sh $PORT ' ; sleep 1; pkill -f 'strix_server.p16fold.*--port $PORT'; sleep 2; rm -rf /cache/factory-replay-2587bc1599; pgrep -f 'strix_server.*--port $PORT' >/dev/null && echo STILL-UP || echo engine stopped"; }
trap 'say "stop: $(stop 2>&1 | tail -1)"' EXIT
# The ssh stays attached for the life of the engine: a process left behind by an ssh that returned is killed on the
# worker (job ayrun1, 2026-11-17: serve.sh was gone within seconds and the window closed after 70 s with nothing loaded).
R "rm -rf ~/2587-replay; mkdir -p ~/2587-replay"
R "bash $KIT/serve.sh $PORT /home/pod/2587-replay" > $OUT/serve.out 2>&1 &
say "engine starting on $IP:$PORT"
timeout $LIMIT python3 $WT/tools/replay/factory_replay.py run --set $WT/gate/replay/factory-v1.set.json \
  --corpus /data/pod/flerp-replay/corpus-v1 --url http://$IP:$PORT --model qwen3.8-flash-next --concurrency 2 --reps 3 \
  --max-tokens 256 --solo all --first-streams $N --wait-ready 600 --expect-slots 2 --out $OUT
rc=$?
say "replay rc $rc"
curl -s -m 10 http://$IP:$PORT/health > $OUT/health.json
curl -s -m 10 http://$IP:$PORT/metrics > $OUT/metrics-end.txt
{ R "pgrep -af 'strix_server.*--port $PORT'; sha256sum /data/models/strix-infer/runtime/bin/strix_server.p16fold; "; cat $OUT/serve.out; } > $OUT/engine.txt 2>&1
R "tail -c 300000 ~/2587-replay/server.log" > $OUT/server.log.tail 2>&1
exit $rc
