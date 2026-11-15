#!/usr/bin/env bash
# PREFILL-ORNITH attention kernel gate (gate/prefill-attn.toml) on bench box K through pod-kbench k_run.
#   bash gate/prefill_attn_run.sh [TAG]      -> gate/results/prefill-attn/k-TAG.{jsonl,err,result.txt}
set -euo pipefail
TAG=${1:-$(date -u +%m%dT%H%MZ)}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CARD=8ef0a645e7
STAGE=/data/pod/kbench/stage/$CARD-attn
OUT=$ROOT/gate/results/prefill-attn
mkdir -p "$OUT"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/src/flerp/attn_wmma_gqa"
cp "$ROOT/gate/prefill_attn_bench.hip" "$TMP/"
cp "$ROOT"/catalog/flerp/kernel/attn_wmma_gqa/src/flerp/attn_wmma_gqa/attn_wmma_gqa.{hip,hpp} "$TMP/src/flerp/attn_wmma_gqa/"
B=./prefill-attn-bench
CMDS=()
for h in 32 16; do
  CMDS+=("$B --check --heads $h --kv 2 --depth 4096 --rows 64" "$B --check --heads $h --kv 2 --depth 0 --rows 1024"
         "$B --check --heads $h --kv 2 --depth 2048 --rows 1024 --gate")
done
if [ "${PHASE:-}" = chunk ]; then
  CMDS=("$B --chunkcheck --heads 16 --kv 2 --depth 0 --rows 16384 --chunk 4096 --gate"
        "$B --chunkcheck --heads 32 --kv 2 --depth 0 --rows 16384 --chunk 4096 --gate"
        "$B --chunkcheck --heads 32 --kv 2 --depth 0 --rows 16384 --chunk 1000 --gate")
fi
for d in $([ "${PHASE:-}" = chunk ] || echo 4096 32768 131072); do CMDS+=("$B --speed --heads 32 --kv 2 --depth $d --rows 4096 --reps 3"); done
[ "${PHASE:-}" = chunk ] || CMDS+=("$B --speed --heads 16 --kv 2 --depth 12288 --rows 4096 --reps 3")
{
  echo '#!/usr/bin/env bash'
  echo 'echo "{\"event\":\"host\",\"host\":\"$(hostname)\",\"loadavg\":\"$(cut -d" " -f1-3 /proc/loadavg)\"}"'
  for c in "${CMDS[@]}"; do
    echo "echo '{\"event\":\"case\",\"cmd\":\"$c\"}'"
    echo "$c || echo '{\"event\":\"case_failed\"}'"
  done
} > "$TMP/run.sh"
BUILD="hipcc -O3 --offload-arch=gfx1151 -std=c++17 -I src prefill_attn_bench.hip src/flerp/attn_wmma_gqa/attn_wmma_gqa.hip -o prefill-attn-bench"
python3 -c 'import json,sys; print(json.dumps({"_card":sys.argv[1],"src":sys.argv[2],"build":sys.argv[3],"cmd":"bash ./run.sh > out.jsonl 2> err.txt; cat out.jsonl","fetch":["out.jsonl","err.txt"],"timeout_s":3000}))' "$CARD" "$STAGE" "$BUILD" > "$TMP/args.json"
tar -C "$TMP" -cf - . | ssh -A relay-host "ssh pod@control-node 'rm -rf $STAGE && mkdir -p $STAGE && tar -C $STAGE -xf - && mv $STAGE/args.json /tmp/pfo-attn.json'"
ssh -A relay-host "ssh pod@control-node 'pod-kbench k_run --args /tmp/pfo-attn.json'" | tee "$OUT/k-$TAG.result.txt"
JOBDIR=$(grep -o '/data/pod/kbench/runs/[^ ]*' "$OUT/k-$TAG.result.txt" | head -1 || true)
[ -n "$JOBDIR" ] && ssh -A relay-host "ssh pod@control-node 'cat $JOBDIR/fetched/out.jsonl'" > "$OUT/k-$TAG.jsonl" || true
[ -n "$JOBDIR" ] && ssh -A relay-host "ssh pod@control-node 'cat $JOBDIR/fetched/err.txt'" > "$OUT/k-$TAG.err" || true
echo "results: $OUT/k-$TAG.*"
