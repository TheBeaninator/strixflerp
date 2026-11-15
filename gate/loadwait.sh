#!/usr/bin/env bash
# Start rule for control-node GPU gates (PREFILL-KPROF amendment 3, PREFILL-LRC): wait until the 1-min loadavg is <= MAX
# (default 6), at most WAIT seconds (default 5400), then run the command. The loadavg sample that passed (or the last one
# on timeout) and the loadavg at the end go to stderr, so the job log shows what the run started under.
MAX=${LOADWAIT_MAX:-6}; WAIT=${LOADWAIT_S:-5400}; t0=$(date +%s); state=timeout
while [ $(( $(date +%s) - t0 )) -lt "$WAIT" ]; do
  read -r l1 l5 l15 _ < /proc/loadavg
  if awk -v l="$l1" -v m="$MAX" 'BEGIN{exit !(l + 0 <= m + 0)}'; then state=ok; break; fi
  sleep 30
done
echo "{\"event\":\"loadwait\",\"state\":\"$state\",\"load_start\":\"$l1 $l5 $l15\",\"waited_s\":$(( $(date +%s) - t0 ))}" >&2
"$@"; rc=$?
echo "{\"event\":\"loadwait_end\",\"load_end\":\"$(cut -d' ' -f1-3 /proc/loadavg)\"}" >&2
exit $rc
