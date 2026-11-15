# ORNITH-O3 step two: the 4-node Ornith window (PREPARED, NOT TAKEN)

Written 2026-11-14 by agent. Criteria: `gate/ornith-o3-spec.toml` [ornith] (committed before any GPU run).
The window needs owner's go and agent's agreement. Nothing below has been run.

## What it runs

World 4 on F, D, A, C (cuts 0/15/30/45/60, graph on, pinned rows, token reply, context 4096, 256 tokens), the 12
prompts of the 35B proxy (P5's bridge, code, rivers, records and O3's code0-3, prose0-3), two launches:

- launch 0: `on:token:1:0,on:token:1:2,on:token:1:0,on:token:1:2,on:token:1:1,on:token:1:3,on:token:1:v3`
- launch 1: `on:token:1:2,on:token:1:0,on:token:1:2,on:token:1:0,on:token:1:3,on:token:1:1,on:token:1:v3`

Rank 3 (C) runs `--draft-head --mtp-data-file /cache/flerp-p5/Ornith-1.5-397B-Q4_K_M.gguf.staged --draft-costs 75.2,19.68,4.83`:
blk.60 and a token_embd copy come from C's PLACE-staged file. Judged by `gate/ornith_o3_spec.py` (judge_launch,
judge_all) inside `gate/ornith_p5.py run --spec-launch`; evidence in `gate/results/ornith-o3-spec/ornith/`.

## Where the bytes come from (PLACE only; no cp, dd or rsync of weights; no fallback)

| rank | node | blocks | bytes | source |
|---|---|---|---|---|
| 0 | F | 0-15 | 61.3 GB | streamed over the rail from control-node's PLACE placement `/data/models/placed/ornith-w4-r0` (`--rank-origin`) |
| 1 | D | 15-30 | 58.4 GB | streamed from control-node's PLACE placement `/data/models/placed/ornith-w4-r1` |
| 2 | A | 30-45 | 58.4 GB | A's `/cache/flerp-p5` (PLACE request `flerp-p5-A`; evicted after the confirm, must be staged again) |
| 3 | C | 45-60 + blk.60 + token_embd | 66.9 GB | C's `/cache/flerp-p5`, PLACE request `flerp-p5-C`, placed with `--with-mtp`, window until 2026-11-14T21:58:29Z |

Nothing is staged on F or D (their /cache holds engram-1017, untouched), and nothing on B (B hung on 2026-11-13
while it served and stored). `/cache/tfp-learned`, `/data/pod/plans/strixfold/tp2-128k.conf` and `~/tfp` are
not touched. No byte is loaded from `/data/ark`, `/mnt/ark*` or `/mnt/nv*`; the swarm origin on control-node
(198.51.100.31:7150) reads the ark copy as PLACE's staging source only. `--model` reads the 11 MB header from the ark
path, as P5 did.

control-node `/data`: 299 GB free on 2026-11-14T00:15Z; ranks 0 + 1 take 119.7 GB, leaving about 179 GB, above
pod-weights' 60 GB floor and PLACE's hard 30 GB, so no floor override is needed.

## Steps (on control-node, as pod)

0. Code. Merge branch `ornith-spec` (laptop worktree `~/Code/strixflerp-spec`) and sync it to the Ornith tree:
   `POD_SYNC_DEST=/data/projects/strixflerp-ornith tools/pod-sync.sh`. First run
   `rsync -n --delete` with the same arguments and agree any deletion with agent: that tree is shared.
   Build the generator and the bundle as before: `bash build/ornith-build.sh` (assembles
   `qwen36-35b-a3b.flerp-moe` into `build/flerp-p5` and `qwen36-35b-a3b.flerp-serve-stage` into `build/flerp-p5-serve`).
1. Stage, no hold, beside serving (each command is request, run, verify inside):

   ```
   cd /data/projects/strixflerp-ornith
   P=/data/models/.swarm/bin/loom-swarm.place
   M=/data/models/ornith-1.5-397b/Ornith-1.5-397B-Q4_K_M.gguf.swarm.json
   $P place verify --id flerp-p5-C                       # C: still placed (window to 2026-11-14T21:58:29Z)
   # if C's window has ended or verify fails:  python3 gate/ornith_p5.py stage --with-mtp --nodes C --until +12h --card CARD
   python3 gate/ornith_p5.py stage --nodes A --until +12h --card CARD
   for r in 0 1; do
     $P place request --id ornith-w4-r$r-control --nodes control --dest /data/models/placed/ornith-w4-r$r \
       --out Ornith-1.5-397B-Q4_K_M.gguf.staged --manifest $M --from 198.51.100.31:7150 \
       --ranges control=/data/models/.place/ranges/ornith-w4-r$r-control.ranges --until +12h --card CARD --owner agent
     $P place run --id ornith-w4-r$r-control && $P place verify --id ornith-w4-r$r-control
   done
   ```

   Staging time at the 2026-11-13 rates (215-280 MB/s): about 4 min each for ranks 0 and 1, about 4 min for A.
2. Preflight, no hold: the same command as step 3 with `--preflight-only` added (origins started and stopped,
   every flag checked against the generator's `--help`, /cache untouched).
3. The window (takes the holds on F D A C itself, drains serving through `/home/pod/ornith-window/after-hold.sh`,
   releases on every exit):

   ```
   python3 gate/ornith_p5.py run --card CARD --source cache \
     --rank-origin 0=198.51.100.31:7172@control:/data/models/placed/ornith-w4-r0/Ornith-1.5-397B-Q4_K_M.gguf.staged:/data/models/.place/ranges/ornith-w4-r0-control.ranges \
     --rank-origin 1=198.51.100.31:7173@control:/data/models/placed/ornith-w4-r1/Ornith-1.5-397B-Q4_K_M.gguf.staged:/data/models/.place/ranges/ornith-w4-r1-control.ranges \
     --hold-owner agent --hold-reason "ORNITH-O3 step two: MTP speculative decoding on Ornith (owner approved DATE)" --hold-ttl 5400 \
     --after-hold "bash /home/pod/ornith-window/after-hold.sh" \
     --spec-launch on:token:1:0,on:token:1:2,on:token:1:0,on:token:1:2,on:token:1:1,on:token:1:3,on:token:1:v3 \
     --spec-launch on:token:1:2,on:token:1:0,on:token:1:2,on:token:1:0,on:token:1:3,on:token:1:1,on:token:1:v3
   ```

4. After: `$P place cancel --id ornith-w4-r0-control`, `... ornith-w4-r1-control`, `... flerp-p5-A`, then
   `$P place evict --owner agent --apply` (as the confirm did); C's `flerp-p5-C` ends with its own window.
   Pull the evidence with `tools/pod-pull.sh`.

## Duration

- Hold to release: about 65 min (budget the 5,400 s hold TTL, 90 min).
  - Holds, drain, bundle push, load: about 5 min (the confirm: load +45 s a launch, drain about 2 min).
  - Each launch: 7 run specs x 12 prompts. Decode: 256 tokens at 75 ms drafts off (19 s), about 13-14 s
    with drafts, so about 23 min a launch. Prompt passes: records (1,808 tokens) 44 s a run x 7 = 5 min,
    the other 11 prompts under 1 s each. About 28 min a launch, two launches 56 min.
  - Cleanup and release: about 3 min.
- Staging before the window: about 12 min of PLACE runs, no hold.
