#!/usr/bin/env bash
# Push this tree to control-node:/data/projects/strixflerp. build/ and gate/results/ are never pushed or deleted:
# results are written on the pod and come back with tools/pod-pull.sh.
# POD_SYNC_DEST=/data/projects/<name> pushes to another pod tree (a clean export built apart from the shared checkout).
set -euo pipefail
cd "$(dirname "$0")/.."
dest=${POD_SYNC_DEST:-/data/projects/strixflerp}
case "$dest" in /data/projects/strixflerp*) ;; *) echo "pod-sync: POD_SYNC_DEST must be /data/projects/strixflerp*" >&2; exit 2;; esac
rsync -a --delete --exclude=.git --exclude=/build --exclude=__pycache__ --exclude=/gate/results \
  -e "ssh -A relay-host ssh" ./ "pod@control-node:${dest%/}/"
