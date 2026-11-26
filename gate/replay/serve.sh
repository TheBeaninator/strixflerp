#!/bin/bash
# FACTORY-REPLAY 2587bc1599: the pinned serving line of 2026-11-17 (strixite_p16fold, the flags strix-solo.sh up composes
# for it) as a user process on the held worker, the way the win_run benches start an engine (75f1d18d2f lenbench,
# 635876b74c stagebench). Runs on the worker from the kit dir, so win_run's cleanup finds it by its /kits/ path.
# Two flags differ from the live line: --port (the fleet port 5300 stays shut, so FreeFitty cannot reach this engine)
# and the prompt cache directory (new and empty, 12 GiB instead of the live 32 GiB one, so the run neither reads nor
# evicts the lanes' saved prefixes and starts from the same state every time). Weights and ngram table from /cache only.
# usage: serve.sh PORT OUTDIR
set -u
PORT=${1:?port}; OUT=${2:?outdir}
M=/data/models/strix-infer
W=/cache/models/qwen38-fnext-p16/weights.strixw
N=/cache/ngram/qwen38-fnext-ngram-int8/ngram.table
SCR=/cache/factory-replay-2587bc1599
for f in $W $N; do test -f $f || { echo "REFUSED: $f is not on this worker's /cache (nothing is loaded from anywhere else)"; exit 3; }; done
rm -rf $SCR; mkdir -p $SCR/pc $OUT || exit 3
ulimit -l unlimited 2>/dev/null || echo "WARN: memlock limit stays $(ulimit -l) kB (the strix-solo unit runs with infinity)"
export LD_LIBRARY_PATH=$M/runtime/lib
$M/runtime/bin/strix_server.p16fold --weights $W --ngram $N --tokenizer $M/tok/tokenizer.json \
  --generation-config $M/tok/generation_config.json --host 0.0.0.0 --port $PORT --capacity 262144 --chunk 16384 --mtp on \
  --prompt-cache-dir $SCR/pc --prompt-cache-gib 12 --prompt-cache-disk-free-gib 8 --prompt-cache-idle-s 150 \
  --model-id qwen3.8-flash-next --total-slots 2 --cache-xfer on --steer-dir $M/steer --prompt-cache-ram-margin-gib 10 \
  > $OUT/server.log 2>&1 < /dev/null &
pid=$!; echo $pid > $OUT/server.pid
trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null; rm -rf $SCR; exit 0' TERM INT HUP
wait $pid; rc=$?
rm -rf $SCR
exit $rc
