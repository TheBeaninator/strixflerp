# Bulk artifacts under gate/results (card 30ad2cc982)

The durable evidence for a run is its text/JSON output (`summary.md`, `bench.json`, `run.txt`,
`load.txt`, `*.out`, `*_agent_info.csv`); those are committed.

Raw `rocprofv3` kernel traces (`trace/*_kernel_trace.csv`, 6–20 MB each) are excluded by `.gitignore`:
they are re-generatable and too large for the repo the laptop pulls. A tar of the ones generated on
the pod and not present in the laptop's history is archived at
`/data/pod/plans/30ad2cc982/artifacts-20261114/large-traces.tar` (76 MB: `decode-profile/trace/**`,
`xfer-r2a/**`, `xfer-r2b/**`, plus the `.out` files), taken 2026-11-14 08:37Z from this tree.
