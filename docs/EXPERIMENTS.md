# Experiments: the gate shape with a resource plan

An experiment is anything that runs hardware to measure something. Its gate toml (`gate/<name>.toml`) is committed before any
hardware run and is never edited to turn a failure into a pass; an amendment is its own commit with date and reason. Since 2026-11-14
(owner: "formalize what an experiment is and have 'resource efficiency' be a section in the requirements") every gate also carries a
`[resources]` table with the same fields as the card's `RESOURCES:` block. The card that runs the gate is KIND `experiment`
(`card-intake template --kind experiment`, rules R15-R25 in ~/Code/card-intake) and its CRITERIA line names this file.

Why: the waste of 2026-11-14 all came from experiments with no resource plan: a K bench re-pulled a 100 GB checkpoint over NFS for
each of 16 processes (half its 4 h), PLACE staging ran two fetches against one USB disk and hung worker B, agents staged weights with
raw dd/cp, and holds sat idle for hours.

Future gate tomls copy the template below. Existing gates are not rewritten; when one is next amended, add the table in that commit.

```toml
[hypothesis]
# Read the mechanism first (code and numbers, not a guess). One prediction, and the number that would refute it.
prediction = "decode rises from ~9 to ~20 tok/s because MTP acceptance reads 2.5 tokens per step"
confirmed_at = "accepted tokens per step >= 2.0"
refuted_below = "accepted tokens per step < 1.5"

[criteria]
# The pass rule, fixed before the run. Numbers and operators only; cannot be loosened after the run.

[resources]
# One line each. Every key is required. card-intake lints the card's copy of these lines.
nodes   = "F D A C via one pod-quad window, opened when staging is done; no idle holds"
          # which nodes/GPUs and the tool that takes them: pod-kbench for K (queue-only); WINJOB/win_run or
          # pod-window/pod-quad for workers; tools/gpu-queue.sh or the control-gpu lease for control-node
hold    = "~65 min expected, TTL 95 min"            # expected duration and a TTL of at most ~1.5x it
weights = "120 GB staged to control-node /data/models/placed via PLACE (source /data/ark/...), plus the A and C slices to /cache via PLACE, all staged before the hold"
          # local NVMe or /cache copy staged through PLACE (or pod-weights); never /mnt/nv*, /mnt/ark*, /data/ark or NFS
          # on demand, except as the explicit source of a PLACE staging. Give the bytes; say whether staging is before the hold.
loads   = "1 per launch, every rep measured inside the same process"   # >1 needs 'because ...' (a load per rep must be justified)
reps    = "3 because the gate is a 2x gap and run-to-run spread is under 10%"   # the smallest count that answers the hypothesis
shares  = "FreeFitty on B alone for ~35 min; StrixFold pair cordoned on F and D; no serving engine on A or C"
cleanup = "place evict for the placements, pod-quad release at the end or the first fault; control-node doctor green afterwards"

[monitor]
# Milestones with deadlines (T = window open). The watcher wakes on milestones and failures and is re-armed until release.
milestones = "staging done by T-10 min; launch by T+5 min; first rep by T+25 min; released by T+65 min"
stop_rule  = "stop and release at the first fault"
```

Checking a card against this: put the same lines in the card (`RESOURCES:` indented under the key, `HYPOTHESIS:`, `CRITERIA: gate/<name>.toml`,
`MONITOR:`) and run `card-intake lint`. A gate on control-node's GPU alone through tools/gpu-queue.sh holds no worker: say so in `nodes`,
write `hold = "none: queued GPU job, ~N min"` style text with a TTL, and keep the rest.
