#!/usr/bin/env bash
# Push the laptop's current HEAD to the pod's Buzz git host as a card branch.
#
#   tools/pod-push.sh --card <10-hex card id> [--dry-run] [--allow-mixed "<reason>"]
#   tools/pod-push.sh --refresh        # only update the local buzz/main ref
#
# Result: Buzz refs/heads/card/<card-id> = HEAD. Nothing here ever touches main:
# main on Buzz is owner-only and moves only when a pod reviewer merges the card
# branch on REVIEW PASS (see /data/pod/plans/buzz-code/GIT.md, "Review and merge").
#
# Repo name = basename of the main checkout (also when run from a card worktree) (strixflerp, card-intake, loom-swarm).
# Path: HEAD -> git bundle (incremental over what the pod bare repo already has)
# -> ssh relay-host -> ssh pod@control-node -> fetch into
# /data/projects/<repo>.git -> push to the Buzz relay.
# Refuses: no/invalid --card, any target outside card/*, a card branch on Buzz
# that is not an ancestor of HEAD (non-fast-forward), a Buzz card branch the
# laptop lacks, and mixed work: every commit on HEAD that is not on Buzz main must
# name this card's 10-hex id in its message (a commit naming another card, or none,
# is listed and the push refused). --allow-mixed "<reason>" overrides and the reason
# is appended to /data/pod/plans/buzz-code/mixed-pushes.log on the pod. Use
# tools/card-branch.sh <card-id> to work in a per-card worktree instead.
# The local ref refs/remotes/buzz/main tracks Buzz main (updated here). Never forces; the fetch and push on the pod are fast-forward
# only. Never touches GitHub.
set -euo pipefail

CARD=""; DRY=0; MIXED=""; REFRESH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --card) CARD="${2:-}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --refresh) REFRESH=1; shift ;;
    --allow-mixed) MIXED="${2:-}"; [ -n "$MIXED" ] || { echo "pod-push: --allow-mixed needs a reason" >&2; exit 1; }; shift 2 ;;
    *) echo "usage: tools/pod-push.sh --card <10-hex card id> [--dry-run] [--allow-mixed \"<reason>\"] | --refresh" >&2; exit 1 ;;
  esac
done
[ $REFRESH = 1 ] || [[ "$CARD" =~ ^[0-9a-f]{10}$ ]] || { echo "pod-push: --card <10-hex card id> is required (got '${CARD}')" >&2; exit 1; }

cd "$(git rev-parse --show-toplevel)"
REPO=$(basename "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")")   # main checkout name, also from a card worktree
BRANCH="card/$CARD"
case "$BRANCH" in card/*) ;; *) echo "pod-push: REFUSED, only card/* branches" >&2; exit 2 ;; esac
BUZZ_URL=http://127.0.0.1:3000/git/pod/$REPO   # pod-local alias; /usr/local/bin/git on the pod signs NIP-98
BARE=/data/projects/$REPO.git
LOCAL=$(git rev-parse --verify HEAD)

pod() { ssh -A relay-host "ssh pod@control-node '$1'"; }
MAINREF=refs/remotes/buzz/main

# Make the local buzz/main ref equal Buzz main; fetch the objects through a bundle if the laptop lacks them.
refresh_main() {
  BUZZMAIN=$(pod "git ls-remote $BUZZ_URL refs/heads/main" | cut -f1)
  [ -n "$BUZZMAIN" ] || { echo "pod-push: Buzz main not found" >&2; return 1; }
  if ! git cat-file -e "$BUZZMAIN^{commit}" 2>/dev/null; then
    local T; T=$(mktemp)
    pod "git -C $BARE fetch -q $BUZZ_URL refs/heads/main:refs/pod-push/main && git -C $BARE bundle create - refs/pod-push/main" > "$T" \
      && git fetch -q "$T" "refs/pod-push/main:$MAINREF" || { rm -f "$T"; echo "pod-push: could not fetch Buzz main" >&2; return 1; }
    rm -f "$T"
  else
    git update-ref "$MAINREF" "$BUZZMAIN"
  fi
}

if [ $REFRESH = 1 ]; then refresh_main; echo "pod-push: $MAINREF = $(git rev-parse --short $MAINREF)"; exit 0; fi
BUZZ=$(pod "git ls-remote $BUZZ_URL refs/heads/$BRANCH" | cut -f1)
refresh_main || { [ -n "$MIXED" ] || { echo "pod-push: REFUSED, cannot verify the commit range without Buzz main" >&2; exit 2; }; }
echo "$REPO: HEAD ${LOCAL:0:9}  buzz $BRANCH ${BUZZ:0:9}  buzz main ${BUZZMAIN:0:9}"

if [ -n "$BUZZ" ]; then
  if [ "$BUZZ" = "$LOCAL" ]; then echo "pod-push: $BRANCH already at $LOCAL, nothing to do"; exit 0; fi
  if ! git cat-file -e "$BUZZ^{commit}" 2>/dev/null; then
    echo "pod-push: REFUSED, Buzz $BRANCH has commit $BUZZ the laptop lacks (no force push)" >&2; exit 2
  fi
  if ! git merge-base --is-ancestor "$BUZZ" "$LOCAL"; then
    echo "pod-push: REFUSED, HEAD is not a descendant of Buzz $BRANCH ($BUZZ); non-fast-forward, no force push" >&2; exit 2
  fi
fi
if git rev-parse -q --verify "$MAINREF" >/dev/null && ! git merge-base --is-ancestor "$MAINREF" "$LOCAL"; then
  echo "pod-push: note: Buzz main has commits HEAD lacks; the reviewer will merge with a merge commit" >&2
fi

# One card per branch: commits on HEAD not on Buzz main must all name this card.
BAD=""
if git rev-parse -q --verify "$MAINREF" >/dev/null; then
  while IFS= read -r c; do
    git log -1 --format=%B "$c" | grep -qE "(^|[^0-9a-zA-Z])$CARD([^0-9a-zA-Z]|$)" || BAD+="$(git log -1 --format='  %h %s' "$c" | cut -c1-150)"$'\n'
  done < <(git rev-list "$MAINREF..$LOCAL")
fi
if [ -n "$BAD" ]; then
  if [ -z "$MIXED" ]; then
    echo "pod-push: REFUSED, commits on HEAD (not on Buzz main) that do not name card $CARD:" >&2
    printf '%s' "$BAD" >&2
    echo "pod-push: work in a per-card worktree (tools/card-branch.sh $CARD), or --allow-mixed \"<reason>\"" >&2
    exit 3
  fi
  echo "pod-push: --allow-mixed: $MIXED" >&2; printf '%s' "$BAD" >&2
fi

# Exclude whatever the pod bare repo already has and the laptop also has.
EXCL=()
for ref in "refs/heads/$BRANCH" refs/heads/main; do
  B=$(pod "git -C $BARE rev-parse --verify -q $ref || true")
  if [ -n "$B" ] && git cat-file -e "$B^{commit}" 2>/dev/null; then EXCL+=("^$B"); fi
done
echo "pod-push: bundling ${#EXCL[@]} base(s) .. ${LOCAL:0:9}"
if [ $DRY = 1 ]; then echo "pod-push: dry run, not sending"; exit 0; fi

TMP=$(mktemp); trap 'rm -f "$TMP"' EXIT
git bundle create "$TMP" "${EXCL[@]}" HEAD
REMOTE_B=/tmp/pod-push-$$-$LOCAL.bundle
ssh -A relay-host "ssh pod@control-node 'cat > $REMOTE_B'" < "$TMP"
if [ -n "$MIXED" ] && [ -n "$BAD" ]; then
  LOGLINE=$(printf '%s %s %s %s mixed-allowed: %s' "$(date -u +%FT%TZ)" "$REPO" "$BRANCH" "${LOCAL:0:12}" "$MIXED" | tr -cd 'A-Za-z0-9 _.,:;/@=()+-')
  pod "echo '$LOGLINE' >> /data/pod/plans/buzz-code/mixed-pushes.log"
fi
pod "git -C $BARE fetch $REMOTE_B HEAD:refs/heads/$BRANCH && git -C $BARE push $BUZZ_URL refs/heads/$BRANCH:refs/heads/$BRANCH; rc=\$?; rm -f $REMOTE_B; exit \$rc"
echo "pod-push: Buzz $BRANCH now $(pod "git ls-remote $BUZZ_URL refs/heads/$BRANCH" | cut -f1)"
echo "pod-push: cite it on the card as branch $BRANCH @ ${LOCAL:0:12}"
