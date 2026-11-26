# Experiment records

One place in the repository for what each experiment found. Owner, 2026-11-17: "we should start documenting our
experimental results into the strixflerp repo". Card EXP-DOCS d7834d0271.

- `INDEX.md`: one row per experiment, newest first: verdict, the number, card, evidence.
- `ROADMAP-dsv41-strixfold.md`: the ordered experiments for DeepSeek-V4.1-Flash on StrixFold, and why in that order.
- `ROADMAP.md`: the same for Qwen3.8-Flash-Next on strixite.
- `<date>-<slug>.md`: one record per experiment that needs more than its index row.

`docs/EXPERIMENTS.md` says how an experiment is set up (gate TOML before the run, resources, monitor). This directory
says how it ended. `docs/CHAMPIONS.md` holds only the winners.

## What a record holds

A record is short. Raw logs, traces and run directories stay where the run wrote them (`gate/results/`, the kbench
run directory, the pod report); the record names them.

```
# NAME: the question in one line
card: <id>   gate: gate/<name>.toml @ <commit>   date: <UTC date>   model: <model>   where: <K | worker | control-node>
Prediction: what the mechanism predicted, with the confirm and refute numbers, as written before the run.
Result: the measured numbers, with spread and rep count.
Verdict: CONFIRMED | REFUTED | NEITHER | NOT RUN, and the one reason.
Fidelity: identity, or KL against the named reference.
Champion change: the promotion made, or "none".
Closes / opens: what this result rules out, and the next card if any.
Evidence: paths.
Not measured: what the run did not cover.
```

## Rules

- A finished experiment adds its row to `INDEX.md`, and its record if it has one, on the card's own branch. The
  reviewer checks the row against the evidence.
- A refuted or closed result is recorded with the same care as a win. The index is how the next agent learns not to
  run it again.
- Numbers are copied from the evidence, never from memory. A row without an evidence path is not a result.
- An amended verdict keeps the old line and adds the new one below it with a date.
