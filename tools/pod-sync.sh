#!/usr/bin/env bash
# RETIRED: rsync --delete of this tree into control-node:/data/projects/strixflerp*. Laptop-side copies into the
# shared pod tree destroyed pod work five times (docs/POD-GIT.md); that tree is a git working tree and nothing but
# git may write into it. Code reaches the pod as a card branch:
#
#   tools/card-branch.sh <card-id>                 # worktree ../strixflerp-card-<card-id> on card/<card-id>
#   (cd ../strixflerp-card-<card-id> && tools/pod-push.sh --card <card-id>)
#
# A build that needs its own checkout clones the pod bare repo into /data/projects/<card>/ (docs/POD-GIT.md).
# This script refuses to run unless given --i-know-this-deletes; it then rsyncs with --delete (build/, .git and
# gate/results/ excluded) into POD_SYNC_DEST (default /data/projects/strixflerp; must be /data/projects/strixflerp*).
set -euo pipefail
if [ "${1:-}" != "--i-know-this-deletes" ]; then
  cat >&2 <<'EOF'
pod-sync: REFUSED. This rsyncs with --delete into a pod tree, and the shared tree /data/projects/strixflerp is a
git working tree that nothing but git may write into (docs/POD-GIT.md: copies like this destroyed pod work five
times). Use the git path instead:
  tools/card-branch.sh <card-id>
  (cd ../strixflerp-card-<card-id> && tools/pod-push.sh --card <card-id>)
To run the rsync anyway (it deletes files at the destination): tools/pod-sync.sh --i-know-this-deletes
EOF
  exit 2
fi
cd "$(dirname "$0")/.."
dest=${POD_SYNC_DEST:-/data/projects/strixflerp}
case "$dest" in /data/projects/strixflerp*) ;; *) echo "pod-sync: POD_SYNC_DEST must be /data/projects/strixflerp*" >&2; exit 2;; esac
rsync -a --delete --exclude=.git --exclude=/build --exclude=__pycache__ --exclude=/gate/results \
  -e "ssh -A relay-host ssh" ./ "pod@control-node:${dest%/}/"
