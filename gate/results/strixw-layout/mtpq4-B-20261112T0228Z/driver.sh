#!/bin/bash
# driver.sh [dry] (runs on control-node, hold on B already taken): strix-solo down, arms OLD-a MTPQ4 OLD-b on B, then
# on EVERY exit: stop ab2-* units on B, verify none and :5310 free, release the hold. dry = no strix-solo/servers/release.
set -u; DRY=${1:-}; D=/tmp/ab2-mtpq4B; LOG=$D/driver.log; B=10.10.0.51; R=/cache/ab2-mtpq4B
OLD=/cache/models/qwen38-fnext-tpn/weights.strixw; Q4=/cache/models/qwen38-fnext-tpn-mtpq4/weights.strixw
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
cleanup(){ rc=$?; trap - EXIT INT TERM; log "cleanup start (rc=$rc)"
  for t in 1 2 3; do
    st=$(ssh -o ConnectTimeout=10 $B 'export XDG_RUNTIME_DIR=/run/user/$(id -u); for u in $(systemctl --user list-units --all --plain --no-legend "ab2-*" | awk "{print \$1}"); do systemctl --user stop "$u"; systemctl --user reset-failed "$u" 2>/dev/null; done; sleep 1; echo "units=$(systemctl --user list-units --all --plain --no-legend "ab2-*" | wc -l) listen5310=$(ss -ltn | grep -c ":5310 ")"' 2>&1)
    log "cleanup check $t: $st"; [ "$st" = "units=0 listen5310=0" ] && break; sleep 3; done
  if [ -n "$DRY" ]; then log "DRY: release skipped"; else log "release: $(OWNER=agent pod-quad release 2>&1 | tr '\n' ' ')"; fi
  log "driver end rc=$rc"; exit $rc; }
trap cleanup EXIT; trap 'log "signal"; exit 130' INT TERM
log "driver start dry=${DRY:-no}"
if [ -n "$DRY" ]; then
  ssh $B 'export XDG_RUNTIME_DIR=/run/user/$(id -u); systemd-run --user --collect --unit=ab2-dry sleep 600' 2>&1 | tee -a $LOG
  log "dry unit started; exiting 9 to exercise cleanup"; exit 9; fi
log "strix-solo down: $(/data/pod/tools/strix-solo.sh down $B 2>&1 | tr '\n' ' ')"
for i in $(seq 1 30); do ssh $B 'pgrep -f strix_server >/dev/null' || break; sleep 2; done
ssh $B 'pgrep -af strix_server' && { log "strix_server still running on B; abort"; exit 5; }
for a in OLD-a:$OLD MTPQ4:$Q4 OLD-b:$OLD; do
  log "arm ${a%%:*} start"; ssh $B "bash $R/arm.sh ${a%%:*} ${a#*:}" 2>&1 | tee -a $LOG; rc=${PIPESTATUS[0]}
  log "arm ${a%%:*} rc=$rc"; [ $rc = 0 ] || exit 6; done
log "all arms done"; exit 0
