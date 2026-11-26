#!/bin/bash
# reslice_tp4.sh TP2_BUNDLE OUT_BUNDLE NODE...: the TP4 (2D split) rank files cut from a TP2 inputs bundle's rank files
# (tf-dsv41-rank-reslice; no checkpoint), one node at a time in the order given. For each node:
#   - the file's exact size from --index-only, refused (exit 3) unless the free space under OUT_BUNDLE covers it plus
#     FLOOR_GB (default 60, the pod-weights guard floor);
#   - OUT_BUNDLE/rank{G}/rank-cache/rank{G}of4-<index hash>.bin (log OUT_BUNDLE/rank{G}-build.log);
#   - its --check against the TP2 file: keys, shapes, offsets and every byte (OUT_BUNDLE/rank{G}-check.log);
# stops at the first failure. Nothing is deleted: moving a finished file off this disk (PLACE to its node) before the
# next node keeps the peak at one file. Register OUT_BUNDLE first (pod-stage alloc ... --card <card>).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
BIN=${TFP_BIN:-$root/zig-out/dsv41-cpu/bin}
FLOOR_GB=${FLOOR_GB:-60}
[ $# -ge 3 ] || { echo "usage: [FLOOR_GB=60] reslice_tp4.sh TP2_BUNDLE OUT_BUNDLE NODE..." >&2; exit 2; }
tp2=$1 out=$2
shift 2
for g in "$@"; do
  case $g in 0|1|2|3) ;; *) echo "NODE is 0..3: $g" >&2; exit 2 ;; esac
done
mkdir -p "$out"
for g in "$@"; do
  need=$(nice -n 19 "$BIN/tf-dsv41-rank-reslice" "$tp2" "$out/rank$g/rank-cache" "$g" --index-only |
    python3 -c 'import json, sys; b = sys.stdin.buffer.read(); r = json.loads(b); print(r[-1][3] + r[-1][4] + len(b) + 16)')
  free=$(df -B1 --output=avail "$out" | tail -1 | tr -d ' ')
  if [ $((need + FLOOR_GB * 1000000000)) -gt "$free" ]; then
    echo "node $g: needs $need bytes + floor ${FLOOR_GB} GB, $free free under $out: refused" >&2
    exit 3
  fi
  echo "$(date -u +%FT%TZ) node $g: building $need bytes ($free free)"
  mkdir -p "$out/rank$g/rank-cache"
  nice -n 19 ionice -c3 "$BIN/tf-dsv41-rank-reslice" "$tp2" "$out/rank$g/rank-cache" "$g" > "$out/rank$g-build.log" 2>&1 ||
    { echo "node $g: build FAILED, see $out/rank$g-build.log" >&2; exit 1; }
  nice -n 19 ionice -c3 "$BIN/tf-dsv41-rank-reslice" "$tp2" "$out/rank$g/rank-cache" "$g" --check > "$out/rank$g-check.log" 2>&1 ||
    { echo "node $g: check FAILED, see $out/rank$g-check.log" >&2; exit 1; }
  f=$(ls "$out/rank$g/rank-cache"/rank${g}of4-*.bin)
  echo "$(date -u +%FT%TZ) node $g: $f $(stat -c %s "$f") bytes, full-byte check ok"
done
