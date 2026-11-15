#!/usr/bin/env bash
# Runs ON control-node: GPU jobs one after another, each started only when the control GPU is free - no control-gpu
# lease, no gate or engine process, no pod job waiting in `pod-gpu --wait`, GTT use under 1 GiB, seen twice 20 s apart. Two lanes share that GPU; a gate
# refuses to load when it is taken, and this queue is how a lane waits its turn instead of failing.
#   tools/gpu-queue.sh <jobs-file> [log]       jobs-file: one job per line, "name | command" (# comments allowed)
# A job that still finds the GPU taken (its output says so and it exits non-zero) is retried, up to 6 times.
# Start it detached:  nohup setsid tools/gpu-queue.sh build/jobs.txt > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
jobs=${1:?usage: tools/gpu-queue.sh <jobs-file> [log]}
log=${2:-build/gpu-queue.log}
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
free() {
  pod-lease status 2>/dev/null | grep -q control-gpu && return 1
  [ "$(cat /sys/class/drm/card*/device/mem_info_gtt_used | sort -n | tail -1)" -lt 1073741824 ] || return 1
  # Busy = some process actually holds the GPU (/dev/kfd open), not a command line that merely names a strixflerp
  # path. 2026-11-13/14: name matching ("flerp gate|...|strixflerp-") made parked launcher shells and even polling ssh
  # commands block every queue for hours, and an agent tried to kill another job's shell to get past it.
  for p in /proc/[0-9]*; do
    ls -l "$p/fd" 2>/dev/null | grep -q /dev/kfd && return 1
  done
  # The pod's own GPU jobs queue with `pod-gpu --wait`: they were here first and their wait has a deadline.
  pgrep -f "pod-gpu --wait" > /dev/null && return 1
  return 0
}
wait_free() { while true; do if free; then sleep 20; free && return 0; fi; sleep 30; done; }
echo "$(stamp) queue started: $jobs" >> "$log"
while IFS='|' read -r name cmd; do
  name=$(echo "$name" | xargs); [ -z "$name" ] && continue; case "$name" in \#*) continue;; esac
  out="build/q-$name.log"; done_=0
  for try in 1 2 3 4 5 6; do
    # One start at a time across every queue (2026-11-12: two queues both saw a free GPU at 00:01Z and the second
    # job died out of memory). The start lock is held through the free check and until this job shows up as busy
    # (its process matches free()'s pattern, or it takes a lease or GTT), at most 300 s.
    exec 8> /tmp/gpu-queue.start.lock
    flock 8
    wait_free
    echo "$(stamp) START $name (try $try): $cmd" >> "$log"
    bash -c "$cmd" 8>&- > "$out" 2>&1 &
    pid=$!
    for _ in $(seq 60); do free || break; kill -0 $pid 2>/dev/null || break; sleep 5; done
    flock -u 8; exec 8>&-
    wait $pid; rc=$?
    if [ $rc -ne 0 ] && grep -q "GPU is not free\|leased to someone else" "$out"; then
      echo "$(stamp) BUSY $name rc=$rc: the GPU was taken; will retry" >> "$log"; sleep 60; continue
    fi
    echo "$(stamp) DONE $name rc=$rc" >> "$log"; done_=1; break
  done
  [ $done_ = 1 ] || echo "$(stamp) GAVE-UP $name after 6 tries" >> "$log"
done < "$jobs"
echo "$(stamp) queue finished" >> "$log"
