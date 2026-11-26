#!/bin/bash
# serve-rank1.sh start [--locked] | stop | status | wait-linked | mark: a follower, tf-dsv41-lanes following rank 0, from
# serve.conf (serve_lib.sh resolves it; RUN_ID names the run, default the UTC time): rank 1 of the TP2 server (node1), or
# with SERVE_RANK=g (1-3, WORLD=4: tp4.conf) rank g of the four-node split on its box. Files and messages carry the rank.
#   start        resolve, check the inputs, write RUN/resolved-rank<g>.json, start the rank in the background
#                (RUN/rank<g>.log, RUN/rank<g>.pid) under this box's GPU lock (--locked: the caller holds it already)
#   wait-linked  until the rank has loaded and linked to rank 0 (its "loaded_s" line), or it exits, or READY_TIMEOUT_S
#   stop         SIGTERM (a follower also leaves by itself when rank 0 stops), SIGKILL after STOP_TIMEOUT_S
#   mark         RUN/stopping written on this box: the rank leaves with 0, not 1, when rank 0 goes (first_light.sh down
#                marks every follower's box before it stops rank 0)
#   check   resolve and check only: RUN/resolved-rank<g>.json printed (identities: deploy commit, kit, config)
#   lock [WAIT_S] / unlock   hold (waiting at most WAIT_S, default LOCK_WAIT_S) / release this box's GPU lock for
#           the run (first_light.sh takes node0's, then node1's)
#   status       running or not, the run's hold on this box's GPU lock, and the log's last line
set -euo pipefail
source "$(dirname "$0")/serve_lib.sh"
R=${SERVE_RANK:-1}
[[ $R =~ ^[1-7]$ ]] || die "SERVE_RANK $R: a follower is rank 1 .. WORLD-1"
serve_load "$R"
RUN_ID=${RUN_ID:-$(serve_latest)}
RUN=$RUN_ROOT/${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
case ${1:-} in
  start)
    mkdir -p "$RUN"; ln -sfn "$(basename "$RUN")" "$RUN_ROOT/latest"
    rm -f "$RUN/stopping"   # (a reused run dir, as RUN_ID's default latest: the marker was an earlier down's)
    [ -z "$(serve_pid "$RUN/rank$R.pid")" ] || die "rank $R already runs ($(cat "$RUN/rank$R.pid"))"
    serve_check; serve_idle; serve_env; serve_cmd; serve_resolved "$RUN/resolved-rank$R.json"
    serve_spawn "$RUN/rank$R.pid" "$RUN/rank$R.log" "$([ "${2:-}" = --locked ] && echo 1 || echo 0)"
    echo "rank $R started ($(cat "$RUN/rank$R.pid")): $RUN";;
  wait-linked)
    for _ in $(seq 1 "$READY_TIMEOUT_S"); do
      grep -q '"loaded_s"' "$RUN/rank$R.log" 2>/dev/null && { echo "rank $R linked: $(grep -m1 '"loaded_s"' "$RUN/rank$R.log")"; exit 0; }
      [ -n "$(serve_pid "$RUN/rank$R.pid")" ] || { echo "rank $R exited:"; tail -5 "$RUN/rank$R.log"; exit 1; }
      sleep 1
    done
    die "rank $R not linked within ${READY_TIMEOUT_S}s";;
  mark) mkdir -p "$RUN"; : > "$RUN/stopping"; echo "rank $R: stop marker in $RUN";;
  check) mkdir -p "$RUN"; serve_check; serve_env; serve_cmd; serve_resolved "$RUN/resolved-rank$R.json"; cat "$RUN/resolved-rank$R.json";;
  lock) mkdir -p "$RUN"; serve_lock_hold "$RUN" "${2:-}"; echo "GPU lock held on $(hostname) ($(cat "$RUN/lock.pid"))";;
  unlock) serve_lock_release "$RUN"; echo "GPU lock released on $(hostname)";;
  stop) serve_stop "$RUN/rank$R.pid" "rank $R";;
  status)
    pid=$(serve_pid "$RUN/rank$R.pid")  # (ended: it started in this run and is gone)
    echo "rank $R ${pid:+running ($pid)}${pid:-not running$([ -e "$RUN/rank$R.pid" ] && echo " (ended)")}: $RUN"
    pid=$(serve_pid "$RUN/lock.pid"); echo "GPU lock on $(hostname): ${pid:+held by this run ($pid)}${pid:-not held by this run}"
    tail -1 "$RUN/rank$R.log" 2>/dev/null || true;;
  *) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 2;;
esac
