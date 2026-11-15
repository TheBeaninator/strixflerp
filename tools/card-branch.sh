#!/usr/bin/env bash
# Create a per-card worktree so only this card's commits get pushed.
#
#   tools/card-branch.sh <10-hex card id>
#
# Makes ../<repo>-card-<id> on a new branch card/<id>, cut from the last-pushed
# Buzz main (local ref refs/remotes/buzz/main; refreshed through pod-push.sh
# --refresh when missing). Work and commit there (name the card id in each commit
# message), then run tools/pod-push.sh --card <id> from that worktree.
set -euo pipefail
ID="${1:-}"
[[ "$ID" =~ ^[0-9a-f]{10}$ ]] || { echo "usage: tools/card-branch.sh <10-hex card id>" >&2; exit 1; }
cd "$(git rev-parse --show-toplevel)"
REPO=$(basename "$PWD")
DIR="$(cd .. && pwd)/$REPO-card-$ID"
[ ! -e "$DIR" ] || { echo "card-branch: $DIR already exists" >&2; exit 2; }
git rev-parse -q --verify refs/remotes/buzz/main >/dev/null || tools/pod-push.sh --refresh
git worktree add -b "card/$ID" "$DIR" refs/remotes/buzz/main
echo "card-branch: worktree $DIR on card/$ID (from buzz/main $(git rev-parse --short refs/remotes/buzz/main))"
echo "card-branch: push with: (cd $DIR && tools/pod-push.sh --card $ID)"
