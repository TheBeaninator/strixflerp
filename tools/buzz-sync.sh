#!/usr/bin/env bash
# buzz-sync: keep main equal between a working checkout and the pod's Buzz git host (card 727be05aef).
#
#   buzz-sync run [<repo>...]        sync every configured repo (or the named ones) once; the timers call this
#   buzz-sync status                 per repo: local and Buzz main, last action, failures, open conflict; timer state
#   buzz-sync log [N]                last N log lines (default 40)
#   buzz-sync merge-card <repo> <card-id> [--dry-run]
#                                    pod side: merge Buzz card/<id> into Buzz main (fast-forward, else a merge
#                                    commit naming the card), then sync the pod checkout. The reviewer runs this on
#                                    REVIEW PASS; akb refuses to close the card to done while the branch is unmerged.
#
# One run per repo:
#   fetch Buzz main; local behind -> fast-forward; local ahead -> push main fast-forward;
#   diverged -> merge commit built in the object store (git merge-tree), local fast-forwards to it, push;
#   conflict -> nothing is changed; one board card + one Buzz notice per new conflict (repeats only log).
#   Then card/* and pod/* branches the local side has and Buzz lacks (or that are strictly ahead of Buzz) are pushed.
# Never forces a push or a ref, never deletes, never commits a working-tree file, never stashes or resets.
# The only refs it moves are the local main branch (fast-forward only, compare-and-swap) and Buzz main / card branches
# (non-force pushes the Buzz hook also enforces). A checked-out main moves with `git merge --ff-only`, which refuses
# to overwrite local edits; then nothing moves locally and the run retries next period. Untracked files in the way that
# are byte-identical to the incoming blobs are set aside, the fast-forward runs, and they come back as tracked files.
# A repo with a rebase/merge/cherry-pick/bisect in progress is skipped for that run.
# Repeated failures (BUZZ_SYNC_FAILCAP runs in a row, default 6 = 30 min) open one diagnosis card.
#
# Sides (auto: hostname control-node* = pod, else laptop; override BUZZ_SYNC_SIDE):
#   laptop  Buzz through ssh relay-host -> ssh pod@control-node and the bare pass-through /data/projects/<repo>.git
#           (git bundles both ways). Config ~/.config/buzz-sync/repos: lines "<repo> <checkout path> <local branch> [<Buzz branch>]"
#           (Buzz branch default main).
#           State + log: ~/.local/state/buzz-sync/. Merge commits: TheBeaninator, Co-Authored-By trailer.
#   pod     /usr/local/bin/git against http://127.0.0.1:3000/git/pod/<repo> (the wrapper signs NIP-98).
#           Repos: the config file if present, else /data/projects/{strixflerp,card-intake,loom-swarm} on main and the reports
#           repo /data/pod/reports on master (Buzz pod/pod-reports master), each only if it exists.
#           Tracking refs: refs/remotes/origin/* when the checkout's origin is the Buzz alias URL, else refs/remotes/buzz/*.
#           State /data/pod/buzz-sync/, log /data/pod/logs/buzz-sync.log. Merge commits: pod <pod@pod>.
set -uo pipefail
BUZZ_SYNC_VERSION=2
REPO=-

SIDE=${BUZZ_SYNC_SIDE:-auto}
if [ "$SIDE" = auto ]; then case "$(hostname)" in control-node*) SIDE=pod ;; *) SIDE=laptop ;; esac; fi
FAILCAP=${BUZZ_SYNC_FAILCAP:-6}
DEFAULT_REPOS="strixflerp card-intake loom-swarm"
REPORTS_DIR=${BUZZ_SYNC_REPORTS_DIR:-/data/pod/reports}
CARD_PARENT=${BUZZ_SYNC_CARD_PARENT:-727be05aef}
if [ "$SIDE" = pod ]; then
  STATE=${BUZZ_SYNC_STATE:-/data/pod/buzz-sync}
  LOG=${BUZZ_SYNC_LOG:-/data/pod/logs/buzz-sync.log}
  GIT_BUZZ=${BUZZ_SYNC_GIT:-/usr/local/bin/git}
  export GIT_AUTHOR_NAME=pod GIT_AUTHOR_EMAIL=pod@pod GIT_COMMITTER_NAME=pod GIT_COMMITTER_EMAIL=pod@pod
  TRAILER=""
  ACTOR=${KANBAN_ACTOR:-pod}
else
  STATE=${BUZZ_SYNC_STATE:-${XDG_STATE_HOME:-$HOME/.local/state}/buzz-sync}
  LOG=${BUZZ_SYNC_LOG:-$STATE/buzz-sync.log}
  GIT_BUZZ=git
  export GIT_AUTHOR_NAME=TheBeaninator GIT_AUTHOR_EMAIL=9450984+TheBeaninator@users.noreply.github.com
  export GIT_COMMITTER_NAME=TheBeaninator GIT_COMMITTER_EMAIL=9450984+TheBeaninator@users.noreply.github.com
  TRAILER=$'\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'
  ACTOR=${KANBAN_ACTOR:-agent}
fi
CONF=${BUZZ_SYNC_CONF:-$HOME/.config/buzz-sync/repos}
HOP=${BUZZ_SYNC_HOP:-relay-host}
POD=${BUZZ_SYNC_POD:-pod@control-node}
SSHO=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
mkdir -p "$STATE" "$(dirname "$LOG")"

now() { date -u +%FT%TZ; }
log() { printf '%s %s %s %s\n' "$(now)" "$SIDE" "$REPO" "$*" >> "$LOG"; [ -t 1 ] && printf '%s %s\n' "$REPO" "$*"; return 0; }
URL_BASE=${BUZZ_SYNC_URL_BASE:-http://127.0.0.1:3000/git/pod}
BARE_BASE=${BUZZ_SYNC_BARE_BASE:-/data/projects}
url() { echo "$URL_BASE/$1"; }
short() { echo "${1:0:9}"; }

# ---- per-repo state: key=value lines in $STATE/<repo>.state ----
st_get() { sed -n "s/^$2=//p" "$STATE/$1.state" 2>/dev/null | tail -1; }
st_set() {  # repo key value
  local f="$STATE/$1.state" t; t=$(mktemp "$STATE/.st.XXXXXX")
  { grep -v "^$2=" "$f" 2>/dev/null; printf '%s=%s\n' "$2" "$3"; } > "$t"; mv -f "$t" "$f"
}

# ---- transport ----
pod_sh() { ssh "${SSHO[@]}" "$HOP" "ssh ${SSHO[*]} $POD bash -s"; }   # script on stdin, runs as pod on control-node

buzz_refs() {  # prints "<sha> <ref>" for main, card/*, pod/*
  local u; u=$(url "$REPO")
  if [ "$SIDE" = pod ]; then
    $GIT_BUZZ ls-remote "$u" "refs/heads/$BB" 'refs/heads/card/*' 'refs/heads/pod/*'
  else
    printf 'git ls-remote %q %q "refs/heads/card/*" "refs/heads/pod/*"\n' "$u" "refs/heads/$BB" | pod_sh
  fi | awk 'NF==2{print $1" "$2}'
}

# Make sure the local repo has commit $1 (Buzz main), and point the tracking ref at it.
buzz_fetch_main() {
  local want=$1 u; u=$(url "$REPO")
  if [ "$SIDE" = pod ]; then
    local ns=buzz
    [ "$(git remote get-url origin 2>/dev/null)" = "$u" ] && ns=origin
    $GIT_BUZZ fetch -q "$u" "+refs/heads/*:refs/remotes/$ns/*" || return 1
    git cat-file -e "$want^{commit}" 2>/dev/null
    return
  fi
  if ! git cat-file -e "$want^{commit}" 2>/dev/null; then
    local base="" t; t=$(mktemp)
    base=$(git rev-parse -q --verify "refs/remotes/buzz/$BB" 2>/dev/null || true)
    { printf 'set -e; B=/data/projects/%q.git; U=%q; BASE=%q; BB=%q\n' "$REPO" "$u" "$base" "$BB"
      cat <<'EOS'
git -C "$B" fetch -q "$U" "+refs/heads/$BB:refs/buzz-sync/main"
if [ -n "$BASE" ] && git -C "$B" cat-file -e "$BASE^{commit}" 2>/dev/null && [ "$(git -C "$B" rev-parse refs/buzz-sync/main)" != "$BASE" ]; then
  git -C "$B" bundle create -q - refs/buzz-sync/main "^$BASE"
else
  git -C "$B" bundle create -q - refs/buzz-sync/main
fi
EOS
    } | pod_sh > "$t" || { rm -f "$t"; return 1; }
    git fetch -q "$t" '+refs/buzz-sync/main:refs/buzz-sync/fetched' || { rm -f "$t"; return 1; }
    rm -f "$t"
    git cat-file -e "$want^{commit}" 2>/dev/null || return 1
  fi
  git update-ref "refs/remotes/buzz/$BB" "$want"
}

# Non-force push of local commit $1 to Buzz ref $2 (full refname). $3 = a commit Buzz is known to have (or empty).
buzz_push() {
  local sha=$1 ref=$2 have=${3:-} u out rc; u=$(url "$REPO")
  if [ "$SIDE" = pod ]; then
    out=$($GIT_BUZZ push -q "$u" "$sha:$ref" 2>&1); rc=$?
    [ $rc = 0 ] || log "push-error $ref $(short "$sha"): $(echo "$out" | tr '\n' ' ' | cut -c1-300)"
    return $rc
  fi
  local t="" rb="/tmp/buzz-sync-$REPO-$$-$RANDOM.bundle"
  if [ -z "$have" ] || ! git merge-base --is-ancestor "$sha" "$have" 2>/dev/null; then
    t=$(mktemp); git update-ref refs/buzz-sync/out "$sha"
    if [ -n "$have" ] && git cat-file -e "$have^{commit}" 2>/dev/null; then
      git bundle create -q "$t" refs/buzz-sync/out "^$have" || { rm -f "$t"; return 1; }
    else
      git bundle create -q "$t" refs/buzz-sync/out || { rm -f "$t"; return 1; }
    fi
    ssh "${SSHO[@]}" "$HOP" "ssh ${SSHO[*]} $POD 'cat > $rb'" < "$t" || { rm -f "$t"; return 1; }
    rm -f "$t"
  else
    rb=""
  fi
  out=$({ printf 'B=/data/projects/%q.git; U=%q; RB=%q; SHA=%q; REF=%q; BB=%q\n' "$REPO" "$u" "$rb" "$sha" "$ref" "$BB"
    cat <<'EOS'
rc=0
git -C "$B" fetch -q "$U" "+refs/heads/$BB:refs/buzz-sync/main" || rc=1
[ "$REF" = "refs/heads/$BB" ] || git -C "$B" fetch -q "$U" "+$REF:refs/buzz-sync/cur" 2>/dev/null || true
if [ $rc = 0 ] && [ -n "$RB" ]; then git -C "$B" fetch -q "$RB" +refs/buzz-sync/out:refs/buzz-sync/out || rc=1; fi
if [ $rc = 0 ]; then git -C "$B" push -q "$U" "$SHA:$REF" 2>&1 || rc=1; fi
[ -n "$RB" ] && rm -f "$RB"
exit $rc
EOS
  } | pod_sh 2>&1); rc=$?
  [ $rc = 0 ] || log "push-error $ref $(short "$sha"): $(echo "$out" | tr '\n' ' ' | cut -c1-300)"
  return $rc
}

# ---- board + notices (always run on the pod; laptop through the hop) ----
on_pod() { if [ "$SIDE" = pod ]; then bash -s; else pod_sh; fi; }

notify() {  # one line to the pod Buzz room; laptop side also to #owner-room on relay-host
  [ -z "${BUZZ_SYNC_NO_BOARD:-}" ] || { log "notice (board off): $1"; return 0; }
  local msg="buzz-sync $SIDE $REPO: $1"
  printf 'KANBAN_ACTOR=%q buzz-say %q >/dev/null 2>&1 || true\n' "$ACTOR" "$msg" | on_pod >/dev/null 2>&1 || true
  if [ "$SIDE" = laptop ]; then printf '%s\n' "$msg" | ssh "${SSHO[@]}" "$HOP" 'owner-room-say' >/dev/null 2>&1 || true; fi
}

# Open (or comment on, and pull back to todo) one board card. $1 = title, $2 = WHY, $3 = NEXT, $4 = body for the comment / NOTE.
card_upsert() {
  local title=$1 why=$2 next=$3 body=$4
  [ -z "${BUZZ_SYNC_NO_BOARD:-}" ] || { echo "card (board off): $title"; return 0; }
  { printf 'export KANBAN_DB=/data/pod/kanban.db KANBAN_ACTOR=%q\n' "$ACTOR"
    printf 'TITLE=%q; WHY=%q; NEXT=%q; BODY=%q; PARENT=%q; REPO=%q; SIDE=%q\n' "$title" "$why" "$next" "$body" "$CARD_PARENT" "$REPO" "$SIDE"
    cat <<'EOS'
ID=$(printf %s "$TITLE" | sha256sum | cut -c1-10)
if akb read "$ID" >/dev/null 2>&1; then
  akb comment --card "$ID" --body "$BODY" >/dev/null && echo "commented $ID"
  col=$(akb read "$ID" | sed -n 's/^column: //p')
  case "$col" in done|review|backlog) akb move "$ID" --to todo >/dev/null 2>&1 && echo "moved $ID to todo";; esac
  exit 0
fi
D=$(mktemp)
NOTE=$(printf %s "$BODY" | tr '\n' ' ' | cut -c1-1500)
cat > "$D" <<EOF
TITLE: $TITLE
KIND: work
WHY: $WHY
DONE: buzz-sync status on the $SIDE side shows $REPO in sync with Buzz main
DONE: buzz-sync log shows an ok run for $REPO after the fix, with fails=0 in buzz-sync status
VERIFY: buzz-sync status | grep "^$REPO " (on the $SIDE side; the log is buzz-sync log)
NEXT: $NEXT
SIZE: est-s
PRIORITY: p1
PARENT: $PARENT
LABELS: buzz-git buzz-sync
NOTE: $NOTE
EOF
if out=$(card-intake lint "$D" 2>&1) && card-intake create --column todo "$D" >/dev/null 2>&1 && akb read "$ID" >/dev/null 2>&1; then
  echo "created $ID"
else
  echo "CARD NOT CREATED ($ID): $(printf %s "$out" | grep -E 'FAIL|rror' | head -3 | tr '\n' ' ')"
fi
rm -f "$D"
EOS
  } | on_pod 2>&1 | tail -2
}

fail() {  # count a failure; escalate once at the cap
  local why=$1 n
  n=$(( $(st_get "$REPO" fails) + 1 ))
  st_set "$REPO" fails "$n"; st_set "$REPO" last_result "fail: $why"; st_set "$REPO" last_run "$(now)"
  log "fail ($n in a row): $why"
  if [ "$n" -ge "$FAILCAP" ] && [ "$(st_get "$REPO" escalated)" != 1 ]; then
    local r; r=$(card_upsert "BUZZ-SYNC-STUCK: $REPO on the $SIDE side failed $FAILCAP syncs in a row" \
      "buzz-sync keeps $REPO main equal on the laptop, the pod checkout and Buzz; $n runs in a row failed, so the sides drift apart." \
      "read the buzz-sync log for $REPO and fix the failing step" \
      "buzz-sync $SIDE $REPO: $n consecutive failures; last: $why. Log: $LOG on the $SIDE side. Status: buzz-sync status.")
    log "escalated: $r"; notify "stuck, $n failures in a row ($why); card: $r"
    st_set "$REPO" escalated 1
  fi
  return 1
}

ok() { st_set "$REPO" fails 0; st_set "$REPO" escalated 0; st_set "$REPO" last_result "ok: $1"; st_set "$REPO" last_run "$(now)"; }

# ---- local fast-forward of the configured branch to $2 (a descendant of $1) ----
worktree_of() {  # path of the worktree that has refs/heads/$1 checked out, or empty
  git worktree list --porcelain | awk -v b="branch refs/heads/$1" '/^worktree /{w=substr($0,10)} $0==b{print w}'
}
busy_op() {  # any worktree mid rebase/merge/cherry-pick/revert/bisect?
  local w g
  while IFS= read -r w; do
    g=$(git -C "$w" rev-parse --absolute-git-dir 2>/dev/null) || continue
    for f in rebase-merge rebase-apply MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD BISECT_LOG; do
      [ -e "$g/$f" ] && { echo "$w ($f)"; return 0; }
    done
  done < <(git worktree list --porcelain | sed -n 's/^worktree //p')
  return 1
}

advance_local() {  # old new -> 0 moved, 1 blocked (logged)
  local old=$1 new=$2 w out rc
  w=$(worktree_of "$BRANCH")
  if [ -z "$w" ]; then
    git update-ref -m "buzz-sync fast-forward" "refs/heads/$BRANCH" "$new" "$old" && { log "ff $BRANCH $(short "$old")..$(short "$new") (not checked out)"; return 0; }
    log "blocked: update-ref $BRANCH lost a race"; return 1
  fi
  [ "$(git -C "$w" rev-parse HEAD)" = "$old" ] || { log "blocked: $BRANCH moved during the run"; return 1; }
  out=$(git -C "$w" merge --ff-only -q "$new" 2>&1); rc=$?
  if [ $rc = 0 ]; then log "ff $BRANCH $(short "$old")..$(short "$new") in $w"; return 0; fi
  # Untracked files in the way: adopt them only if every one is byte-identical to the incoming blob.
  local paths=() p ident=1 bak
  while IFS= read -r p; do [ -n "$p" ] && paths+=("$p"); done < <(
    comm -12 <(git -C "$w" diff --name-only "$old" "$new" | sort) <(git -C "$w" ls-files -o --exclude-standard | sort))
  if [ ${#paths[@]} -gt 0 ] && [ "$(git -C "$w" rev-parse HEAD)" = "$old" ]; then
    for p in "${paths[@]}"; do
      [ -f "$w/$p" ] && [ ! -L "$w/$p" ] && [ "$(git -C "$w" hash-object -- "$p")" = "$(git -C "$w" rev-parse -q --verify "$new:$p" 2>/dev/null)" ] || { ident=0; break; }
    done
    if [ $ident = 1 ]; then
      bak="$STATE/adopt-$REPO-$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$bak"
      for p in "${paths[@]}"; do mkdir -p "$bak/$(dirname "$p")"; mv -- "$w/$p" "$bak/$p"; done
      out=$(git -C "$w" merge --ff-only -q "$new" 2>&1); rc=$?
      if [ $rc = 0 ]; then
        local bad=0
        for p in "${paths[@]}"; do cmp -s "$w/$p" "$bak/$p" || bad=1; done
        if [ $bad = 0 ]; then rm -rf "$bak"; else log "WARN: adopted file differs, backup kept in $bak"; fi
        log "ff $BRANCH $(short "$old")..$(short "$new") in $w; adopted ${#paths[@]} identical untracked file(s): $(printf '%s ' "${paths[@]}" | cut -c1-300)"
        return 0
      fi
      for p in "${paths[@]}"; do [ -e "$w/$p" ] || mv -- "$bak/$p" "$w/$p"; done
      rmdir -p "$bak" 2>/dev/null || true
    fi
  fi
  log "blocked: ff of checked-out $BRANCH refused, local tree left as is: $(echo "$out" | tr '\n' ' ' | cut -c1-300)"
  return 1
}

# ---- one repo ----
sync_one() {
  REPO=$1; local path=$2; BRANCH=$3; BB=${4:-main}
  [ -d "$path" ] && git -C "$path" rev-parse --git-dir >/dev/null 2>&1 || { fail "no git checkout at $path"; return 1; }
  cd "$path" || return 1
  exec 9>"$STATE/$REPO.lock"; flock -n 9 || { log "skip: another run holds the lock"; return 0; }
  local w; w=$(busy_op) && { log "skip: operation in progress in $w"; st_set "$REPO" last_result "skip: $w busy"; st_set "$REPO" last_run "$(now)"; return 0; }
  local refs bm l
  refs=$(buzz_refs) || { fail "ls-remote Buzz failed"; return 1; }
  bm=$(awk -v r="refs/heads/$BB" '$2==r{print $1}' <<<"$refs")
  [ -n "$bm" ] || { fail "Buzz has no $BB"; return 1; }
  l=$(git rev-parse -q --verify "refs/heads/$BRANCH") || { fail "no local branch $BRANCH"; return 1; }
  buzz_fetch_main "$bm" || { fail "could not fetch Buzz main $(short "$bm")"; return 1; }
  st_set "$REPO" buzz "$bm"; st_set "$REPO" local "$l"
  local result="" rc=0
  if [ "$l" = "$bm" ]; then
    result="in sync at $(short "$l")"
  elif git merge-base --is-ancestor "$l" "$bm"; then
    if advance_local "$l" "$bm"; then result="pulled $(short "$l")..$(short "$bm")"; else result="behind, local ff blocked"; rc=1; fi
  elif git merge-base --is-ancestor "$bm" "$l"; then
    if buzz_push "$l" "refs/heads/$BB" "$bm"; then log "pushed main $(short "$bm")..$(short "$l")"; result="pushed $(short "$bm")..$(short "$l")"; st_set "$REPO" buzz "$l"; else result="push of main failed"; rc=1; fi
  else
    local tree mrc
    tree=$(git merge-tree --write-tree "$l" "$bm" 2>/dev/null); mrc=$?
    if [ $mrc = 1 ]; then
      conflict "$l" "$bm" "$(git merge-tree --write-tree --name-only "$l" "$bm" 2>/dev/null | sed -n '2,/^$/p' | grep -v '^$' | head -20 | tr '\n' ' ')"
      return 1
    elif [ $mrc != 0 ]; then
      fail "git merge-tree error ($mrc)"; return 1
    fi
    tree=$(head -1 <<<"$tree")
    local nl nb m
    nl=$(git rev-list --count "$bm..$l"); nb=$(git rev-list --count "$l..$bm")
    m=$(printf 'Merge Buzz main into %s %s (buzz-sync)\n\nbuzz-sync %s: local %s (%s commits not on Buzz) and Buzz main %s (%s commits not local)\ndiverged and merged without conflicts.%s\n' \
        "$SIDE" "$BRANCH" "$SIDE" "$(short "$l")" "$nl" "$(short "$bm")" "$nb" "$TRAILER" | git commit-tree "$tree" -p "$l" -p "$bm") \
      || { fail "commit-tree failed"; return 1; }
    log "merge $(short "$l") + Buzz $(short "$bm") -> $(short "$m")"
    advance_local "$l" "$m" || rc=1
    if buzz_push "$m" "refs/heads/$BB" "$bm"; then log "pushed main $(short "$bm")..$(short "$m")"; result="merged $(short "$m") and pushed"; st_set "$REPO" buzz "$m"; else result="merged $(short "$m"), push failed"; rc=1; fi
    [ $rc = 0 ] || result="$result (local ff blocked or push failed)"
  fi
  st_set "$REPO" conflict ""
  push_branches "$refs" "$bm"
  st_set "$REPO" local "$(git rev-parse "refs/heads/$BRANCH")"
  if [ $rc = 0 ]; then ok "$result"; else fail "$result"; fi
}

conflict() {  # local buzz files
  local l=$1 bm=$2 files=$3 key="$1 $2"
  st_set "$REPO" last_run "$(now)"; st_set "$REPO" last_result "conflict: local $(short "$l") vs Buzz $(short "$bm")"
  if [ "$(st_get "$REPO" conflict)" = "$key" ]; then log "conflict (unchanged, already reported): $files"; return; fi
  st_set "$REPO" conflict "$key"
  log "CONFLICT local $(short "$l") vs Buzz main $(short "$bm"); nothing changed; files: $files"
  local r; r=$(card_upsert "BUZZ-SYNC-CONFLICT: $REPO main on the $SIDE side conflicts with Buzz main" \
    "buzz-sync cannot merge $REPO main automatically, so the laptop, the pod checkout and Buzz drift apart until someone merges it by hand." \
    "merge Buzz main into $REPO main on the $SIDE side by hand, resolving the conflicting files, then run buzz-sync run $REPO" \
    "buzz-sync $SIDE $REPO $(now): local $BRANCH $l and Buzz main $bm both moved and conflict in: $files. Nothing was changed on either side; the timer keeps retrying and syncs again once local main contains Buzz main.")
  log "card: $r"; notify "CONFLICT local $(short "$l") vs Buzz main $(short "$bm") in: $files -- nothing changed; card: $r"
}

push_branches() {  # card/* and pod/* the local side has that Buzz lacks or is strictly behind on
  local refs=$1 bm=$2 ref sha bz
  while read -r sha ref; do
    bz=$(awk -v r="$ref" '$2==r{print $1}' <<<"$refs")
    if [ "$bz" = "$sha" ]; then continue; fi
    if [ -n "$bz" ]; then
      git cat-file -e "$bz^{commit}" 2>/dev/null && git merge-base --is-ancestor "$bz" "$sha" || continue
    fi
    if buzz_push "$sha" "$ref" "${bz:-$bm}"; then log "pushed ${ref#refs/heads/} ${bz:+$(short "$bz")..}$(short "$sha")"; fi
  done < <(git for-each-ref --format='%(objectname) %(refname)' refs/heads/card refs/heads/pod)
}

repos() {  # "<repo> <path> <branch>" lines
  if [ -f "$CONF" ]; then grep -vE '^\s*(#|$)' "$CONF"; return; fi
  if [ "$SIDE" = pod ]; then
    local r; for r in $DEFAULT_REPOS; do [ -d "$BARE_BASE/$r/.git" ] && echo "$r $BARE_BASE/$r main main"; done
    [ -d "$REPORTS_DIR/.git" ] && echo "pod-reports $REPORTS_DIR master master"
  else
    echo "buzz-sync: no config at $CONF (lines: <repo> <checkout path> <local branch>)" >&2
  fi
}

cmd_run() {
  local want=" $* " r p b bb any=0
  while read -r r p b bb; do
    [ -n "$r" ] || continue
    [ $# -eq 0 ] || [[ "$want" == *" $r "* ]] || continue
    any=1
    ( sync_one "$r" "${p/#\~/$HOME}" "${b:-main}" "${bb:-main}" ) || true
  done < <(repos)
  [ $any = 1 ] || { echo "buzz-sync: no repos matched" >&2; return 1; }
}

cmd_status() {
  echo "buzz-sync v$BUZZ_SYNC_VERSION side=$SIDE script=$(sha256sum "$0" | cut -c1-12) log=$LOG"
  local r p b bb
  while read -r r p b bb; do
    [ -n "$r" ] || continue
    REPO=$r
    local l bz c
    l=$(git -C "${p/#\~/$HOME}" rev-parse -q --verify "refs/heads/${b:-main}" 2>/dev/null)
    bz=$(st_get "$r" buzz); c=$(st_get "$r" conflict)
    printf '%-12s local %s=%s  buzz %s(last seen)=%s  last %s: %s  fails=%s%s\n' "$r" "${b:-main}" "$(short "$l")" "${bb:-main}" "$(short "$bz")" \
      "$(st_get "$r" last_run)" "$(st_get "$r" last_result)" "$(st_get "$r" fails)" "${c:+  CONFLICT open ($c)}"
  done < <(repos)
  if command -v systemctl >/dev/null; then systemctl --user list-timers buzz-sync.timer --no-pager 2>/dev/null | sed -n '1,2p'; fi
}

cmd_merge_card() {  # pod side: repo card [--dry-run]
  [ "$SIDE" = pod ] || { echo "merge-card runs on control-node (as pod); from the laptop: ssh relay-host 'ssh pod@control-node buzz-sync merge-card $*'" >&2; return 2; }
  REPO=${1:-}; local id=${2:-} dry=${3:-} u B c m main tree
  [[ "$id" =~ ^[0-9a-f]{10}$ ]] && [ -n "$REPO" ] || { echo "usage: buzz-sync merge-card <repo> <10-hex card id> [--dry-run]" >&2; return 2; }
  u=$(url "$REPO"); B=$BARE_BASE/$REPO.git
  [ -d "$B" ] || { echo "merge-card: no bare repo $B" >&2; return 2; }
  exec 9>"$STATE/$REPO.merge.lock"; flock -w 120 9 || { echo "merge-card: lock busy" >&2; return 1; }
  $GIT_BUZZ -C "$B" fetch -q "$u" +refs/heads/main:refs/buzz-sync/main "+refs/heads/card/$id:refs/buzz-sync/card" \
    || { echo "merge-card: fetch of main or card/$id from Buzz failed (does card/$id exist?)" >&2; return 1; }
  main=$(git -C "$B" rev-parse refs/buzz-sync/main); c=$(git -C "$B" rev-parse refs/buzz-sync/card)
  if git -C "$B" merge-base --is-ancestor "$c" "$main"; then echo "merge-card: card/$id @ $(short "$c") is already in Buzz main $(short "$main")"; return 0; fi
  if git -C "$B" merge-base --is-ancestor "$main" "$c"; then
    m=$c; echo "merge-card: fast-forward Buzz main $(short "$main")..$(short "$c")"
  else
    tree=$(git -C "$B" merge-tree --write-tree "$main" "$c"); local mrc=$?
    if [ $mrc != 0 ]; then
      echo "merge-card: CONFLICT merging card/$id @ $(short "$c") into Buzz main $(short "$main"); nothing pushed. Files:" >&2
      git -C "$B" merge-tree --write-tree --name-only "$main" "$c" 2>/dev/null | sed -n '2,/^$/p' | grep -v '^$' | head -30 >&2
      echo "merge-card: resolve on card/$id (merge Buzz main into the card branch, push it), then rerun" >&2
      return 3
    fi
    tree=$(head -1 <<<"$tree")
    m=$(printf 'Merge card/%s (card %s)\n\nbuzz-sync merge-card on REVIEW PASS: card/%s @ %s into Buzz main %s.\n' "$id" "$id" "$id" "$(short "$c")" "$(short "$main")" \
        | git -C "$B" commit-tree "$tree" -p "$main" -p "$c") || return 1
    echo "merge-card: merge commit $(short "$m") (main $(short "$main") + card/$id $(short "$c"))"
  fi
  if [ "$dry" = --dry-run ]; then echo "merge-card: dry run, not pushed"; return 0; fi
  $GIT_BUZZ -C "$B" push -q "$u" "$m:refs/heads/main" || { echo "merge-card: push refused (Buzz main moved? rerun)" >&2; return 1; }
  printf '%s %s %s merge-card card/%s %s -> main %s\n' "$(now)" "$SIDE" "$REPO" "$id" "$(short "$c")" "$(short "$m")" >> "$LOG"
  echo "merge-card: Buzz main = $(short "$m"); cite it on the card: merged card/$id @ $(short "$c") into main $(short "$m")"
  exec 9>&-
  if repos | awk -v r="$REPO" '$1==r{f=1} END{exit !f}'; then cmd_run "$REPO"; fi
}

case "${1:-}" in
  run) shift; cmd_run "$@" ;;
  status) cmd_status ;;
  log) tail -n "${2:-40}" "$LOG" ;;
  merge-card) shift; cmd_merge_card "$@" ;;
  version) echo "buzz-sync $BUZZ_SYNC_VERSION ($SIDE)" ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
