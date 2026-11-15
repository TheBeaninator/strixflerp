# Git rules for this tree (pod side) — since 2026-11-14, card 30ad2cc982

**rsync/scp/tar into this tree is retired.** `/data/projects/strixflerp` is a git working tree now,
not a copy. Nothing may write into it except git operations run on the pod. The pre-conversion
snapshot of the old rsync copy is at `/data/projects/strixflerp.rsync-2026-11`, which since 2026-11-13
is a **symlink** to the read-only ark at `/mnt/ark/pod-overflow-2026-11-13/strixflerp.rsync-2026-11`
(70,517 files, 3,622 excluding `build/`). It exists purely so nothing from the conversion is lost;
it must never be synced over the tree. Before this, laptop-side copies into the directory destroyed
pod work five times (`docs/POD-GIT.md` alone, 2026-11-12 → 2026-11-14) — that is the failure mode
this rule ends.

The safety property that matters about the snapshot is not "identical to the tree" (the tree has since
moved on to the laptop's real history plus pod gate evidence, so a `diff -r` can never print nothing),
it is **no file in the snapshot is absent from the tree**. Check it with:

```
diff -rq -x .git -x build -x __pycache__ /data/projects/strixflerp.rsync-2026-11/ /data/projects/strixflerp/ \
  | grep '^Only in /mnt/ark'    # want: no output
```

(2026-11-14 08:40Z run: zero such lines. The 475 other diff lines are files that differ or are new
in the tree, which is expected.)

## Where the repository lives

| role | path / URL |
|---|---|
| canonical remote (both sides) | Buzz git host, pod-local alias `http://127.0.0.1:3000/git/pod/strixflerp` (the relay's owner-pubkey path is `/git/362160f79347ad5530704cbc32cc72c3407a2695665918e9952a215514b68cdc/strixflerp`) |
| pod working tree | `/data/projects/strixflerp` — `origin` stores the **alias** URL; use `git buzz-fetch` / `git buzz-push` (repo aliases), not a bare `git fetch origin` |
| pod bare repo (what the laptop's `tools/pod-push.sh` fetches from) | `/data/projects/strixflerp.git` — `buzz` remote = the alias URL |
| pod mirror of the relay | `/data/git/strixflerp.git` — `git clone --mirror` of the relay; a local, credential-free `git pull` target for anything on the pod that wants the history without the relay |
| laptop | `~/Code/strixflerp`; card branches go out with `tools/pod-push.sh` (in-repo): bundle → ssh → fast-forward fetch into the pod bare repo → push to the relay. Never forces. Never touches GitHub |
| main on all three | `buzz-sync` (`tools/buzz-sync.sh`), a systemd --user timer every 5 min on the laptop and on control-node (pod): fast-forward, push, or a conflict-free merge commit; a conflict changes nothing and opens a `BUZZ-SYNC-CONFLICT` card. `buzz-sync status` / `buzz-sync log`. Full description: /data/pod/plans/buzz-code/GIT.md "Automatic sync of main" |

### How to talk to the relay from the pod (this is F4; getting it wrong looks like a successful push)

`/usr/local/bin/git` is a wrapper that mints an NIP-98 token **from a relay URL that appears in the
command's own arguments** (and rewrites the `pod` alias to the pubkey there). It cannot see through
a remote *name*: `git fetch origin` / `git push origin` reach `/usr/bin/git` with no URL argument, get
no token, and fail with `could not read Username for 'http://127.0.0.1:3000/...'`. Use the aliases,
which carry the literal URL:

- fetch: `git buzz-fetch` → `!/usr/local/bin/git fetch http://127.0.0.1:3000/git/pod/strixflerp +refs/heads/*:refs/remotes/origin/*`
- push: `git buzz-push <src>:<dst>` → `!/usr/local/bin/git push http://127.0.0.1:3000/git/pod/strixflerp <src>:<dst>`
- always confirm a push landed: `git ls-remote http://127.0.0.1:3000/git/pod/strixflerp | grep <branch>`

If the wrapper should sign by remote name instead, that is a pod-infra change (asked on the card);
until then, plain `git push origin` is *not* a way to reach the relay.

## The branch rule

- `main` is kept equal on the laptop, in this tree and on the relay by buzz-sync (pod side: `buzz-sync.timer`, user pod,
  log /data/pod/logs/buzz-sync.log). A commit on `main` here reaches the relay and the laptop within about 5 minutes, and
  laptop commits arrive here the same way. buzz-sync only fast-forwards this checkout (`git merge --ff-only`), so it never
  overwrites an uncommitted edit: a dirty file that an incoming commit touches holds `main` back until it is committed or reverted.
- Work for a card goes on a branch **`card/<card-id>`**, one commit per step whose check passes, card id first in the message:
  `git commit -m "30ad2cc982 what"`. buzz-sync pushes a new `card/*` (or `pod/*`) branch to the relay on its next run; `git buzz-push`
  does it at once. Never `git add -A` — this tree is shared with other pod threads.
- The branch reaches `main` through review: on REVIEW PASS the reviewer runs `buzz-sync merge-card strixflerp <card-id>`
  (fast-forward, else a merge commit naming the card). akb refuses to close the card while `card/<id>` is unmerged; a
  `MERGE-SKIP: <reason>` comment waives it for a branch folded into another card's.
- A pod thread that must not touch the shared checkout uses `git worktree add /tmp/wt-<card> -b card/<card> origin/main`.
- A build or gate that needs its own clean checkout clones the bare repo into `/data/projects/<card>/`,
  not a second worktree of this checkout.

## What must not be committed

- `build/` — assembled engine trees, ~5.8 GB; gitignored (`/build/`), keep it that way.
- The replay corpus and any model data: weights, GGUF/safetensors, recorded corpora. They live under
  `/data/models`, `/mnt/*` and the workers' NVMe, never in this repository.
- Raw profiler kernel traces, `gate/results/**/trace/*_kernel_trace.csv` (6–20 MB each). The committed
  evidence for a run is its `summary.md` / `bench.json` / `run.txt` / `.out` files; the raw traces are
  archived outside git (see `gate/results/ARTIFACTS.md`) and are what the `git status --porcelain`
  exclusions in `.gitignore` cover.
- Secrets and keys.
