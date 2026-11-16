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
S=$(readlink -f /data/projects/strixflerp.rsync-2026-11)
comm -23 <(cd "$S" && find . -type f -not -path './build/*' -not -path './.git/*' -not -path '*/__pycache__/*' -not -name '*.pyc' | LC_ALL=C sort) \
         <(cd /data/projects/strixflerp && find . -type f -not -path './build/*' -not -path './.git/*' -not -path '*/__pycache__/*' -not -name '*.pyc' | LC_ALL=C sort)
# want: no output. Every line it prints is one file the snapshot has and the tree lost.
```

`readlink -f` is not decoration: the snapshot path is a symlink, and `diff -rq` prints the path **as
invoked**, so an `grep '^Only in /mnt/ark'` on a `diff` invoked through the symlink path matches
nothing and the check can never fail (this card published exactly that broken check on 2026-11-14;
review card 134c5596c2 caught it). Do not "simplify" the command back to a `diff`+`grep` one-liner
unless the grep pattern is built from `$S` after `readlink`.

The check was proved to be able to fail before it was written here: with a planted pair of files
missing from a toy tree it prints both (`./lost.txt`, `./sub/deep.txt`), while the old `grep '^Only in
/mnt/ark'` form prints 0 on the same planted loss. On the real paths it prints **0 lines** (measured
2026-11-15 15:20Z, list re-counted 16:00Z): nothing from the pre-conversion copy is missing from the
tree. Both sides of that comparison are non-empty — snapshot 3,601 files, tree 8,108 (re-measured with the final `.git` exclusion), with `build/`,
`.git/`, `__pycache__/` and `*.pyc` excluded — because an empty left-hand list would make any
"nothing missing" check vacuous, which is the mistake this section used to contain. Separately, a
plain `diff -rq` of snapshot vs tree reports 679 lines (measured 15:23Z) of files that differ or are
new in the tree — the tree having correctly advanced, which is not what this property is about.

## Where the repository lives

| role | path / URL |
|---|---|
| canonical remote (both sides) | Buzz git host, pod-local alias `http://127.0.0.1:3000/git/pod/strixflerp` (the relay's owner-pubkey path is `/git/<relay-owner-pubkey>/strixflerp`) |
| pod working tree | `/data/projects/strixflerp` — `origin` stores the **alias** URL; use `git buzz-fetch` / `git buzz-push` (repo aliases), not a bare `git fetch origin` |
| pod bare repo (what the laptop's `tools/pod-push.sh` fetches from) | `/data/projects/strixflerp.git` — `buzz` remote = the alias URL |
| pod mirror of the relay | `/data/git/strixflerp.git` — `git clone --mirror` of the relay; a local, credential-free `git pull` target for anything on the pod that wants the history without the relay. **Refreshed for you**: `strixflerp-mirror-refresh.timer` (user pod, ~every 15 min) runs `~/.local/bin/strixflerp-mirror-refresh`, which fetches and then compares the mirror to the relay ref-for-ref and sha-for-sha and exits non-zero on a difference; by hand: `git -C /data/git/strixflerp.git buzz-refresh` (alias, carries the literal relay URL) or the script itself. Its `--selftest` plants a deleted ref in a copy of the mirror and requires the comparison to report `DIFFERS` — the check is proved to be able to fail, not assumed (review 134c5596c2 F2: the mirror had no refresh path and lagged 3 refs; a `clone --mirror` is only as good as its last fetch) |
| pod bare repo, relay-side refs | `/data/projects/strixflerp.git` is the laptop's write target, **not a mirror** — the same timer records the relay's refs there under `refs/remotes/buzz/*` so "what is on the relay" is answerable locally without touching its `refs/heads` (which are the pod side of the `tools/pod-push.sh` handshake) |
| laptop | `~/Code/strixflerp`; card branches go out with `tools/pod-push.sh` (in-repo): bundle → ssh → fast-forward fetch into the pod bare repo → push to the relay. Never forces. Never touches GitHub |
| main on all three | `buzz-sync` (`tools/buzz-sync.sh`), a systemd --user timer every 5 min on the laptop and on control-node (pod): fast-forward, push, or a conflict-free merge commit; a conflict changes nothing and opens a `BUZZ-SYNC-CONFLICT` card. `buzz-sync status` / `buzz-sync log`. Full description: /data/pod/plans/buzz-code/GIT.md "Automatic sync of main" |

### How to talk to the relay from the pod (getting it wrong looks like a successful push)

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
- A branch that adds or removes modules, recipes, origins or champion rows, changes a manifest's `[gate] status`,
  or bumps `FLERP_EXECUTOR_ABI` also runs `python3 tools/readme-facts.py` and commits the README block it rewrites;
  `python3 tools/readme-facts.py --check` fails while the block is stale.
- The merge can make the block stale too, when main moved on after the branch last regenerated it. Before
  `buzz-sync merge-card`, the reviewer merges main into the card branch in a scratch clone and runs
  `python3 tools/readme-facts.py --check`; if it fails, the reviewer fails the round with the merged sha, and the
  author merges main, regenerates the block and pushes again. Main never takes a merge whose `--check` fails.
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

## Status of the pod side (2026-11-15, card 30ad2cc982 round 14)

- `docs/POD-GIT.md` is on `main` (the laptop merged it) and on relay refs `card/30ad2cc982`,
  `pod/30ad2cc982` (285f7de), `pod/30ad2cc982-mainline` (52406f7), `pod/8ef0a645e7` (75db9ed: card
  8ef0a645e7's three pod gate-evidence commits, which are also ancestors of `main`).
- `/data/git/strixflerp.git` mirrors the relay and is refreshed by `strixflerp-mirror-refresh.timer`;
  `~/.local/bin/strixflerp-mirror-refresh --selftest` proves that refresh's own check can fail.
- The NIP-98-by-remote-name change is tracked as board card **7151ce857c** (pod-infra), not as an
  untracked ask in a card log. Until it lands, the `buzz-*` aliases above are the supported path.
