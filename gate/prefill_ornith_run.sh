#!/usr/bin/env bash
# PREFILL-ORNITH kernel gate (gate/prefill-ornith.toml) on bench box K through pod-kbench k_run.
#   bash gate/prefill_ornith_run.sh check|speed|dense|profile [TAG]
# Run from the laptop: stages the bench and the kernel on control-node (/data/pod/kbench/stage/8ef0a645e7-pfo), builds
# with hipcc on K (the job's build step), runs the phase, and fetches the JSON lines into
# gate/results/prefill-ornith/<phase>-<TAG>.jsonl (with the job's result.txt).
set -euo pipefail
PHASE=${1:?check|speed|speed2|exact|dense|dense2|profile}
TAG=${2:-$(date -u +%m%dT%H%MZ)}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CARD=8ef0a645e7
STAGE=/data/pod/kbench/stage/$CARD-pfo
OUT=$ROOT/gate/results/prefill-ornith
mkdir -p "$OUT"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/src/flerp/moe_kgrouped"
cp "$ROOT/gate/prefill_ornith_bench.hip" "$TMP/"
cp "$ROOT"/catalog/flerp/kernel/moe_kgrouped/src/flerp/moe_kgrouped/moe_kgrouped.{hip,hpp} "$TMP/src/flerp/moe_kgrouped/"
B=./prefill-ornith-bench
case "$PHASE" in
  check)
    CMDS=("$B --check --tokens 192 --down q6k" "$B --check --tokens 192 --down q4k"
          "$B --check-dense --rows 192 --n 12288 --k 4096 --type q6k" "$B --check-dense --rows 192 --n 8192 --k 4096 --type q4k"
          "$B --check-dense --rows 192 --n 4096 --k 8192 --type q4k") ;;
  speed)
    CMDS=("$B --speed --rows 1024 --down q6k" "$B --speed --rows 4096 --down q6k" "$B --speed --rows 4096 --down q6k --skew --old 0"
          "$B --speed --rows 4096 --down q4k --old 0" "$B --speed --rows 16384 --down q6k --reps 3") ;;
  dense)
    CMDS=()
    for r in 1024 4096 16384; do
      for s in "12288 4096 q6k" "8192 4096 q4k" "4096 8192 q4k" "16384 4096 q4k" "1024 4096 q4k"; do
        set -- $s; CMDS+=("$B --dense --rows $r --n $1 --k $2 --type $3")
      done
    done ;;
  speed2)
    CMDS=()
    for o in 1 0; do
      CMDS+=("FLERP_KG_ORDER=$o $B --speed --rows 1024 --down q6k --old 0" "FLERP_KG_ORDER=$o $B --speed --rows 4096 --down q6k --old 0"
             "FLERP_KG_ORDER=$o $B --speed --rows 16384 --down q6k --old 0 --reps 3")
    done
    for r in 1024 4096 16384; do
      for s in "12288 4096 q6k" "8192 4096 q4k" "4096 8192 q4k" "16384 4096 q4k"; do
        set -- $s; CMDS+=("$B --dense --rows $r --n $1 --k $2 --type $3")
      done
    done ;;
  exact)
    CMDS=("$B --exact --check --tokens 192 --down q6k" "$B --exact --check --tokens 192 --down q4k"
          "$B --exact --check-dense --rows 192 --n 12288 --k 4096 --type q6k" "$B --exact --check-dense --rows 192 --n 8192 --k 4096 --type q4k"
          "$B --exact --check-dense --rows 192 --n 4096 --k 8192 --type q4k"
          "$B --exact --speed --rows 1024 --down q6k --old 0" "$B --exact --speed --rows 4096 --down q6k --old 0"
          "$B --exact --speed --rows 16384 --down q6k --old 0 --reps 3")
    for r in 4096 16384; do
      for s in "12288 4096 q6k" "4096 8192 q4k"; do
        set -- $s; CMDS+=("$B --exact --dense --rows $r --n $1 --k $2 --type $3")
      done
    done ;;
  dense2)
    CMDS=()
    for x in "" "--exact "; do
      for sh in "4096 2048 q4k" "2048 4096 q4k" "4096 2048 q6k" "512 4096 q6k" "64 4096 q4k"; do
        set -- $sh; CMDS+=("$B $x--check-dense --rows 192 --n $1 --k $2 --type $3")
        for r in 4096 16384; do CMDS+=("$B $x--dense --rows $r --n $1 --k $2 --type $3"); done
      done
    done ;;
  profile)
    CMDS=("$B --speed --rows 4096 --down q6k --old 0 --reps 2") ;;
  *) echo "unknown phase $PHASE" >&2; exit 2 ;;
esac
{
  echo '#!/usr/bin/env bash'
  echo 'set -u'
  echo 'echo "{\"event\":\"host\",\"host\":\"$(hostname)\",\"loadavg\":\"$(cut -d" " -f1-3 /proc/loadavg)\"}"'
  for c in "${CMDS[@]}"; do
    echo "echo '{\"event\":\"case\",\"cmd\":\"$c\"}'"
    echo "echo \"{\\\"event\\\":\\\"load\\\",\\\"at\\\":\\\"start\\\",\\\"loadavg\\\":\\\"\$(cut -d' ' -f1-3 /proc/loadavg)\\\"}\""
    echo "$c || echo '{\"event\":\"case_failed\"}'"
    echo "echo \"{\\\"event\\\":\\\"load\\\",\\\"at\\\":\\\"end\\\",\\\"loadavg\\\":\\\"\$(cut -d' ' -f1-3 /proc/loadavg)\\\"}\""
  done
} > "$TMP/run.sh"
BUILD="hipcc -O3 --offload-arch=gfx1151 -std=c++17 -I src prefill_ornith_bench.hip src/flerp/moe_kgrouped/moe_kgrouped.hip -o prefill-ornith-bench -lpthread"
if [ "$PHASE" = profile ]; then
  ARGS=$(python3 -c 'import json,sys; print(json.dumps({"_card":sys.argv[1],"src":sys.argv[2],"build":sys.argv[3],"cmd":"bash ./run.sh","top":20}))' "$CARD" "$STAGE" "$BUILD")
  VERB=k_profile
else
  ARGS=$(python3 -c 'import json,sys; print(json.dumps({"_card":sys.argv[1],"src":sys.argv[2],"build":sys.argv[3],"cmd":"bash ./run.sh > out.jsonl 2> err.txt; cat out.jsonl","fetch":["out.jsonl","err.txt"],"timeout_s":3000}))' "$CARD" "$STAGE" "$BUILD")
  VERB=k_run
fi
echo "$ARGS" > "$TMP/args.json"
tar -C "$TMP" -cf - . | ssh -A relay-host "ssh pod@control-node 'rm -rf $STAGE && mkdir -p $STAGE && tar -C $STAGE -xf - && mv $STAGE/args.json /tmp/pfo-$PHASE.json'"
ssh -A relay-host "ssh pod@control-node 'pod-kbench $VERB --args /tmp/pfo-$PHASE.json'" | tee "$OUT/$PHASE-$TAG.result.txt"
JOBDIR=$(grep -o '/data/pod/kbench/runs/[^ ]*' "$OUT/$PHASE-$TAG.result.txt" | head -1 || true)
if [ -n "$JOBDIR" ] && [ "$PHASE" != profile ]; then
  ssh -A relay-host "ssh pod@control-node 'cat $JOBDIR/fetched/out.jsonl 2>/dev/null || find $JOBDIR -name out.jsonl -exec cat {} +'" > "$OUT/$PHASE-$TAG.jsonl" || true
  ssh -A relay-host "ssh pod@control-node 'cat $JOBDIR/fetched/err.txt 2>/dev/null || find $JOBDIR -name err.txt -exec cat {} +'" > "$OUT/$PHASE-$TAG.err" || true
fi
echo "results: $OUT/$PHASE-$TAG.*"
