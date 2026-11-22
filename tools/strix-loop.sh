#!/usr/bin/env bash
# STRIX-LOOP: one command gates a strixite kernel change on bench box K.
#
#   tools/strix-loop.sh <target> [gate/strix_loop.py options]
#
#   <target>  null | recipe:<name> | module:<module-id> | branch:<strixite ref>     (gate/strix_loop.py --help)
#   options   --run NAME --stages plan,build,micro,model,isa,report --order ABAB|AB --prompts ... --rounds N
#             --predict "the speed prediction, written before the run"
#
# From a laptop card worktree (branch card/<id>, tools/card-branch.sh): pushes HEAD to Buzz as card/<id>
# (tools/pod-push.sh), checks that branch out in the pod clone /data/projects/<id>/strixflerp (cloned from the pod bare
# repo /data/projects/strixflerp.git; never the shared checkout /data/projects/strixflerp), starts
# gate/strix_loop.py there detached (the run outlives this ssh), polls its log until verdict.json exists, then copies
# gate/results/strix-loop/<run>/ back into this worktree for the author to read and commit.
# On control-node itself it runs gate/strix_loop.py in the foreground in the current tree.
#
# Everything on the GPU side happens on K through pod-kbench k_run (queue-only; waits behind the bench:K holder, never
# kills anyone). Builds and the isa scan are CPU work on control-node. No worker, no control-node GPU.
# STRIX_LOOP_DETACH_ONLY=1 returns right after the start (poll later with: tools/strix-loop.sh --attach <run>).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
[ $# -ge 1 ] || { sed -n 2,20p "$0"; exit 2; }

HOP='ssh -A relay-host ssh pod@control-node'
on_pod() { [ "$(hostname)" = control-node ] && [ -d /data/projects ]; }

if on_pod; then
  exec python3 gate/strix_loop.py "$@"
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
CARD=${STRIX_LOOP_CARD:-${BRANCH#card/}}
[[ "$CARD" =~ ^[0-9a-f]{10}$ ]] || { echo "strix-loop: run from a card worktree (branch card/<10-hex id>) or set STRIX_LOOP_CARD" >&2; exit 2; }
CLONE=/data/projects/$CARD/strixflerp
RUN=""
ATTACH=0
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[$i]}" in
    --run) RUN="${args[$((i + 1))]}" ;;
    --attach) ATTACH=1; RUN="${args[$((i + 1))]}" ;;
  esac
done
if [ -z "$RUN" ]; then
  slug=$(echo "$1" | tr 'A-Z' 'a-z' | sed 's/[^a-z0-9]\+/-/g; s/^-//; s/-$//' | cut -c1-32)
  RUN="$(date -u +%Y%m%dT%H%M%SZ)-$slug"
  args+=(--run "$RUN")
fi
RES="gate/results/strix-loop/$RUN"

if [ "$ATTACH" = 0 ]; then
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "strix-loop: commit your changes first (the pod runs what Buzz has)" >&2; exit 2; }
  tools/pod-push.sh --card "$CARD"
  HEAD=$(git rev-parse HEAD)
  # two ssh hops re-split the command line: the argument vector travels base64-encoded
  $HOP bash -s -- "$CARD" "$CLONE" "$HEAD" "$RUN" "$(printf '%q ' "${args[@]}" | base64 -w0)" <<'EOF'
set -euo pipefail
CARD=$1; CLONE=$2; HEAD=$3; RUN=$4; ARGS=$(echo "$5" | base64 -d)
mkdir -p "$(dirname "$CLONE")"
if [ ! -d "$CLONE/.git" ]; then
  git clone -q /data/projects/strixflerp.git "$CLONE"
fi
cd "$CLONE"
git fetch -q /data/projects/strixflerp.git "card/$CARD"
git checkout -q -B "card/$CARD" FETCH_HEAD
[ "$(git rev-parse HEAD)" = "$HEAD" ] || { echo "pod clone is at $(git rev-parse HEAD), laptop pushed $HEAD" >&2; exit 3; }
mkdir -p "gate/results/strix-loop/$RUN"
LOG="gate/results/strix-loop/$RUN/loop.log"
eval "set -- $ARGS"
STRIX_LOOP_CARD=$CARD setsid nohup python3 gate/strix_loop.py "$@" > "$LOG" 2>&1 < /dev/null &
echo "started strix_loop.py pid $! in $CLONE @ $(git rev-parse --short HEAD); log $CLONE/$LOG"
EOF
  [ "${STRIX_LOOP_DETACH_ONLY:-0}" = 1 ] && { echo "detached; attach with: tools/strix-loop.sh --attach $RUN"; exit 0; }
fi

echo "strix-loop: waiting for $CLONE/$RES/verdict.json (poll every 120 s)"
while :; do
  # one script to the pod over both hops: a quoted command line re-splits on relay-host and only its first command reaches the pod
  state=$(printf 'cd %q && tail -n 2 %q/loop.log 2>/dev/null; test -f %q/verdict.json && echo VERDICT_READY; pgrep -f %q > /dev/null || echo PROCESS_GONE\n' \
            "$CLONE" "$RES" "$RES" "strix_loop.py.*$RUN" | $HOP bash -s 2>/dev/null || true)
  echo "$(date -u +%H:%M:%SZ) $(echo "$state" | grep -v 'VERDICT_READY\|PROCESS_GONE' | tail -n 1)"
  echo "$state" | grep -q VERDICT_READY && break
  if echo "$state" | grep -q PROCESS_GONE; then echo "strix-loop: the run ended without a verdict; see $CLONE/$RES/loop.log" >&2; break; fi
  sleep 120
done
mkdir -p "$RES"
rsync -a -e "ssh -A relay-host ssh" "pod@control-node:$CLONE/$RES/" "$RES/"
echo "strix-loop: results in $RES"
[ -f "$RES/verdict.json" ] && python3 -c "import json,sys; v=json.load(open('$RES/verdict.json')); print(v['verdict']); [print(('PASS' if c['pass'] else 'FAIL') if not c.get('report_only') else 'REPORT', c['criterion']) for c in v['criteria']]; print('canary:', v['canary_pin']['line'])"
