#!/usr/bin/env bash
# test-buzz-sync: offline checks of buzz-sync's sync decisions against a local bare repo standing in for Buzz.
# Runs the pod side (local git, no ssh, no board): BUZZ_SYNC_SIDE=pod BUZZ_SYNC_GIT=git BUZZ_SYNC_NO_BOARD=1.
# Usage: test-buzz-sync.sh [path to buzz-sync]   (default: buzz-sync next to this file, else on PATH)
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
BS=${1:-}
[ -n "$BS" ] || for c in "$here/buzz-sync.sh" "$here/../home-bin/buzz-sync" "$(command -v buzz-sync 2>/dev/null)"; do [ -x "$c" ] && { BS=$c; break; }; done
[ -x "$BS" ] || { echo "test-buzz-sync: buzz-sync not found" >&2; exit 2; }
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
export BUZZ_SYNC_SIDE=pod BUZZ_SYNC_GIT=git BUZZ_SYNC_NO_BOARD=1 BUZZ_SYNC_URL_BASE=$T/buzz BUZZ_SYNC_BARE_BASE=$T/proj
export BUZZ_SYNC_STATE=$T/state BUZZ_SYNC_LOG=$T/log BUZZ_SYNC_CONF=$T/repos
export GIT_CONFIG_GLOBAL=$T/gitconfig GIT_CONFIG_NOSYSTEM=1
git config --global init.defaultBranch main; git config --global user.name t; git config --global user.email t@t
pass=0; failn=0
ok() { if eval "$2"; then pass=$((pass+1)); else failn=$((failn+1)); echo "FAIL: $1   [$2]"; fi; }
g() { git -C "$1" "${@:2}"; }
commit() { echo "$3" > "$1/$2"; g "$1" add "$2"; g "$1" commit -qm "$4"; }

mkdir -p "$T/buzz" "$T/proj"
git init -q --bare "$T/buzz/r"; git init -q --bare "$T/proj/r.git"
git init -q "$T/seed"; commit "$T/seed" a.txt 1 init; g "$T/seed" push -q "$T/buzz/r" main
git clone -q "$T/buzz/r" "$T/proj/r"; git clone -q "$T/buzz/r" "$T/other"
echo "r $T/proj/r main" > "$T/repos"
P=$T/proj/r; O=$T/other
run() { "$BS" run r >/dev/null 2>&1; }
bz() { git --git-dir="$T/buzz/r" rev-parse main; }

run; ok "in sync" '[ "$(g $P rev-parse main)" = "$(bz)" ] && grep -q "in sync" $T/state/r.state'

commit "$O" b.txt 1 "buzz moves"; g "$O" push -q origin main
run; ok "behind -> ff" '[ "$(g $P rev-parse main)" = "$(bz)" ] && [ -f $P/b.txt ]'

commit "$P" c.txt 1 "local moves"
run; ok "ahead -> push" '[ "$(g $P rev-parse main)" = "$(bz)" ] && grep -q "pushed main" $T/log'

g "$O" pull -q; commit "$O" d.txt 1 "buzz again"; g "$O" push -q origin main; commit "$P" e.txt 1 "local again"
L=$(g $P rev-parse main)
run; ok "diverged -> merge commit pushed" '[ "$(g $P rev-parse main)" = "$(bz)" ] && [ "$(g $P rev-list --parents -1 main | wc -w)" = 3 ] && g $P merge-base --is-ancestor $L main && [ -f $P/d.txt ]'
ok "merge author is the pod identity" '[ "$(g $P log -1 --format=%an main)" = pod ]'

g "$O" pull -q; commit "$O" a.txt buzzside "conflict buzz"; g "$O" push -q origin main; commit "$P" a.txt localside "conflict local"
L=$(g $P rev-parse main); B=$(bz)
run; ok "conflict -> nothing changed" '[ "$(g $P rev-parse main)" = "$L" ] && [ "$(bz)" = "$B" ] && grep -q "^conflict=" $T/state/r.state && [ "$(cat $P/a.txt)" = localside ]'
run; ok "conflict repeat is not re-reported" '[ "$(grep -c "already reported" $T/log)" = 1 ]'
# resolve by hand, sync recovers
g "$P" fetch -q origin; g "$P" merge -q -X ours origin/main -m resolve >/dev/null 2>&1
run; ok "after manual resolve -> pushed" '[ "$(g $P rev-parse main)" = "$(bz)" ] && grep -q "^conflict=$" $T/state/r.state'

# dirty tracked file that the incoming change touches: refused, tree untouched
g "$O" pull -q; commit "$O" b.txt buzz2 "touch b"; g "$O" push -q origin main
echo "my edit" > "$P/b.txt"; L=$(g $P rev-parse main)
run; ok "dirty overlap -> blocked, edit kept" '[ "$(g $P rev-parse main)" = "$L" ] && [ "$(cat $P/b.txt)" = "my edit" ] && grep -q "blocked" $T/log'
g "$P" checkout -q -- b.txt
run; ok "clean again -> ff" '[ "$(g $P rev-parse main)" = "$(bz)" ]'

# dirty unrelated file: ff proceeds and keeps it
echo "unrelated" > "$P/c.txt"; g "$O" pull -q; commit "$O" f.txt 1 "f"; g "$O" push -q origin main
run; ok "unrelated dirty file kept, ff done" '[ "$(g $P rev-parse main)" = "$(bz)" ] && [ "$(cat $P/c.txt)" = unrelated ]'
g "$P" checkout -q -- c.txt

# untracked file identical to the incoming blob: adopted
g "$O" pull -q; commit "$O" g.txt same "g"; g "$O" push -q origin main; echo same > "$P/g.txt"
run; ok "identical untracked adopted" '[ "$(g $P rev-parse main)" = "$(bz)" ] && [ -z "$(g $P status --porcelain)" ]'
# untracked differing: blocked, file kept
g "$O" pull -q; commit "$O" h.txt theirs "h"; g "$O" push -q origin main; echo mine > "$P/h.txt"; L=$(g $P rev-parse main)
run; ok "differing untracked -> blocked, kept" '[ "$(g $P rev-parse main)" = "$L" ] && [ "$(cat $P/h.txt)" = mine ]'
rm "$P/h.txt"; run

# main not checked out: ref moves without touching the tree
g "$P" checkout -q -b pod/feature; echo wip > "$P/wip.txt"
g "$O" pull -q; commit "$O" i.txt 1 "i"; g "$O" push -q origin main
run; ok "main not checked out -> update-ref" '[ "$(g $P rev-parse main)" = "$(bz)" ] && [ "$(g $P rev-parse --abbrev-ref HEAD)" = pod/feature ] && [ ! -f $P/i.txt ]'
rm "$P/wip.txt"; g "$P" checkout -q main

# card branch the local side has and Buzz lacks
g "$P" branch card/0123456789 main~1
run; ok "card branch pushed" 'git --git-dir=$T/buzz/r rev-parse -q --verify card/0123456789 >/dev/null'

# operation in progress: skip
g "$P" rev-parse HEAD > "$(g $P rev-parse --absolute-git-dir)/MERGE_HEAD"
g "$O" pull -q; commit "$O" j.txt 1 "j"; g "$O" push -q origin main; L=$(g $P rev-parse main)
run; ok "merge in progress -> skipped" '[ "$(g $P rev-parse main)" = "$L" ] && grep -q "operation in progress" $T/log'
rm "$(g $P rev-parse --absolute-git-dir)/MERGE_HEAD"; run

# merge-card: ff, merge commit, already merged, conflict
g "$O" pull -q; g "$O" checkout -q -b card/aaaaaaaaaa; commit "$O" k.txt 1 "k aaaaaaaaaa"; g "$O" push -q origin card/aaaaaaaaaa
"$BS" merge-card r aaaaaaaaaa >/dev/null 2>&1
ok "merge-card ff" '[ "$(bz)" = "$(g $O rev-parse card/aaaaaaaaaa)" ] && [ "$(g $P rev-parse main)" = "$(bz)" ]'
"$BS" merge-card r aaaaaaaaaa 2>&1 | grep -q "already in Buzz main"; ok "merge-card already merged" '[ ${PIPESTATUS[0]} = 0 ]'
g "$O" checkout -q main; g "$O" pull -q; g "$O" checkout -q -b card/bbbbbbbbbb main~1; commit "$O" l.txt 1 "l bbbbbbbbbb"; g "$O" push -q origin card/bbbbbbbbbb
"$BS" merge-card r bbbbbbbbbb >/dev/null 2>&1
ok "merge-card merge commit" '[ "$(git --git-dir=$T/buzz/r log -1 --format=%s main)" = "Merge card/bbbbbbbbbb (card bbbbbbbbbb)" ]'
g "$O" checkout -q main; g "$O" pull -q; g "$O" checkout -q -b card/cccccccccc main; commit "$O" a.txt cardside "c cccccccccc"; g "$O" push -q origin card/cccccccccc
g "$O" checkout -q main; commit "$O" a.txt mainside "main a"; g "$O" push -q origin main
B=$(bz); "$BS" merge-card r cccccccccc > $T/mc.out 2>&1; rc=$?; [ $rc = 3 ] || cat $T/mc.out
ok "merge-card conflict -> rc 3, main unchanged" '[ $rc = 3 ] && [ "$(bz)" = "$B" ]'

# a repo tracked on master (Buzz branch master), checkout without an origin remote: tracking refs go to refs/remotes/buzz/*
git init -q --bare "$T/buzz/rep"; git init -q -b master "$T/rep"; commit "$T/rep" r.txt 1 "rep init"
git -C "$T/rep" push -q "$T/buzz/rep" master; git clone -q -b master "$T/buzz/rep" "$T/rep-other"
echo "rep $T/rep master master" >> "$T/repos"
commit "$T/rep-other" s.txt 1 "rep buzz moves"; g "$T/rep-other" push -q origin master
"$BS" run rep >/dev/null 2>&1
ok "master/master repo pulled" '[ "$(g $T/rep rev-parse master)" = "$(git --git-dir=$T/buzz/rep rev-parse master)" ] && g $T/rep rev-parse -q --verify refs/remotes/buzz/master >/dev/null'
commit "$T/rep" t.txt 1 "rep local moves"
"$BS" run rep >/dev/null 2>&1
ok "master/master repo pushed" '[ "$(g $T/rep rev-parse master)" = "$(git --git-dir=$T/buzz/rep rev-parse master)" ] && ! git --git-dir=$T/buzz/rep rev-parse -q --verify main >/dev/null'

echo "test-buzz-sync: $pass ok, $failn failed"
[ $failn = 0 ]
