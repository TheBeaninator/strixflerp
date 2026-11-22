# STRIX-LOOP: one command gates a strixite kernel change on bench box K

The goal (2026-11-14): iterate on strixite performance quickly. Bench box K tests the kernel modules, and a different
strixflerp configuration is a recipe edit. This is the loop that makes that one command.

```
tools/strix-loop.sh <target> [options]          # from a laptop card worktree (branch card/<id>), or on control-node
tools/strix-loop.sh --attach <run>              # re-join a run started earlier
```

`<target>`:

| form | what the variant is |
|---|---|
| `null` | the base recipe against itself: the pipeline's zero (must come out identical, ratio about 1.00) |
| `recipe:<name>` | a recipe in `recipes/` whose `[modules]` bindings or `[overlay]` differ from the base: the recipe edit that swaps a kernel module |
| `module:<module-id>` | a catalogue module; every interface it provides that the base binds elsewhere is rebound to it |
| `branch:<ref>` | a branch or commit of a local strixite checkout at origin commit d19945e (`STRIX_LOOP_STRIXITE_REPO`): its `kernels/` files that differ from the origin commit become overlay modules `kernel.flerp.loop_<slug>_<name>` over `kernel.strixite.<name>` (recipe `[overlay]`, the mechanism `recipes/qwen38-flash-next.strixite-p16.toml` uses), and a generated recipe binds them. Changes outside kernel modules are listed in the verdict, not built |

The base is `recipes/qwen38-flash-next.strixite-pf.toml` (the served strixite modules plus `strixflerp-prefill-bench`);
`--base` names another recipe that carries the bench binary. The generated variant recipe is
`recipes/qwen38-flash-next.strixite-loop.toml` and the branch overlay modules are `catalog/flerp/kernel/loop_*/`, both
gitignored: the run's `plan.json` records what they held.

## What runs where

Everything GPU-side runs on bench box K (one gfx1151 node) through `pod-kbench k_run`, the only way onto K:
each job queues behind the `bench:K` holder and never kills anyone. Builds, assembly and the ISA scan are CPU work on
control-node (therock-10.1 clang, `nice -n 19`, 12 jobs). No worker (A B C D F) and no control-node GPU are touched, so
a kernel gate no longer waits on the single control-node GPU lane.

From the laptop, `tools/strix-loop.sh` pushes the card branch (`tools/pod-push.sh --card`), checks it out in a per-card
clone on control-node (cloned from the pod bare repo; never the shared checkout), starts
`gate/strix_loop.py` detached there, polls its log every two minutes until `verdict.json` exists, and copies
`gate/results/strix-loop/<run>/` back for the author to commit.

Stages (`--stages plan,build,micro,model,isa,report`; `state.json` lets a stage rerun on its own):

1. **plan**: resolve the target to swaps (interface: base module -> variant module, or overlay), write the variant
   recipe, check both assemblies resolve, write `plan.json` with the prediction fields (`--predict "..."` before the run).
2. **build**: `flerp assemble` + cmake + ninja for base and variant, `compile_commands.json` exported. The two trees
   build in about 130-145 s each on control-node; the bench binaries' sha256 are recorded (a null run's are equal).
3. **micro** (kernel microbench): for each swapped interface that has a `gate/interface/<name>.toml`, the interface
   driver is assembled twice, bound to the base provider and to the variant, both staged on K and run in process
   order A B A B. Parity dumps are compared bit for bit across every A/B pairing (0 differing 32-bit words required);
   the bench cases' `median_us` give a speed table (variant / base, and whether the two sides' ranges overlap).
   An interface without a driver is listed; the full-model gate decides for it.
4. **model** (full-model gate): `strixflerp-prefill-bench` from both builds on K, process order A B A B (`--order AB`
   halves it), one weights load per process, inside it every prompt (default 128, 512, 1024, 2048, 4096, 8192, 16384,
   32768 tokens) for `--rounds 2` (round 0 = first request, round 1 = second request), 32 greedy tokens after each
   prompt. Per prompt: logits sha256 of the last 16 rows and greedy ids against base (identity), prefill tok/s
   (second request) and decode tok/s (T = 1, MTP off) as median/min/max per side with the ratio and whether the ranges
   overlap, device bytes, K's Tctl and loadavg per process, the process wall and an upper bound on its load time.
   Logits are kept only for prompts whose sha differs (KL offline).
5. **isa**: `tools/isa-scan.py` over the changed kernel units of both builds (scratch, spills, VGPR, occupancy) and
   the diff; a kernel using scratch in the variant build fails the gate (reported only on a null run).
6. **report**: `verdict.json` and `summary.md` under `gate/results/strix-loop/<run>/`, with the criteria, the tables
   and the canary pin line.

Criteria (fixed in the code, not per run): every bench process exits 0; logits sha256 and greedy ids equal to base on
every prompt (`--expect report` records instead of judging, for a change whose numeric contract differs); every
microbench parity case identical; no scratch in the variant's changed kernels; for a null run, the second binary's
prefill tok/s inside the first's own spread on every prompt. Speed gains are reported with their noise, never judged
here: the card's criteria file says what gain it wants.

## The canary pin

`verdict.json` ends with `canary_pin`: `pod-engine pin <letter> strixite_loop_<run>` plus the steps before it. It is
a line for a human to run, only when every criterion passed and the gain is wanted: register the serving (a
`SERVE_strixite_loop_<run>` block in engines.conf, `pod-engine register`) whose binary is the variant server from the
run's build copied to `/data/models/strix-infer/runtime/bin/` (replace in place; never `mv` over a path the workers
exec), pick one worker, pin it; `pod-engine pin <letter> -` removes the pin. The loop never pins anything.

## Weights on K

The Qwen3.8-Flash-Next `weights.strixw` is 74.6 GB. K has 124 GiB (133 GB) of RAM, a 1 GiB VRAM carve-out and GTT
over the rest, a 63 GiB `/dev/shm`, no NVMe and no rail NIC. strixite's strixw loader copies host to device, so a
model placed in `/dev/shm` through PLACE (`place request --nodes K --tier ram`) is held twice
while it runs: 2 x 74.6 = 149 GB does not fit in 133 GB, and PLACE refuses it (the bench RAM-tier cap is 46 GB:
2 x M + 40 GB of engine headroom <= 133 GB). So the bench reads the pod-weights staged NVMe copy
on control-node (`/data/models/strix-infer/converted/qwen38-fnext-tpn/weights.strixw`) over K's 2.5 GbE NFS mount,
once per process, and every rep is measured inside that process; the n-gram table is read lazily (pread) the same way.
`--weights PATH --ngram PATH` point the bench at a placement that does fit (a `pod-kbench` job may also name a PLACE
id as `place` and read `KB_WEIGHTS`); nothing is ever read from `/mnt/*` or `/data/ark`.

## The null run (2026-11-15)

Run `20261115T031124Z-null` (`gate/results/strix-loop/20261115T031124Z-null/`, K jobs
`20261115T031610Z-run-router` and `20261115T031615Z-run-pmodel`):

- builds on control-node 130 s (base) and 144 s (variant), byte-identical bench binaries;
- microbench `moe-router` on K, A B A B: 16/16 parity cases identical (0 differing words); bench cases
  6.6 us (M=1), 6.6-6.7 us (M=16), 20.9 us (M=2048, k=10) with ratios 0.98-1.01;
- full model on K, A B A B, 8 prompts x 2 rounds a process: logits sha256 and greedy ids identical on every prompt;
  prefill tok/s (second request) 711 (128) ... 1613 (8192), 1595 (16384), 1579 (32768); ratios 0.9984-1.0042,
  median 1.0017; decode 32.6-33.5 tok/s at T = 1, MTP off; device 76.6-78.3 GiB after the prompt;
- the weights load from the NFS copy took 281 s (A1), 553 s (B1), 365 s (A2), 296 s (B2) as upper bounds (process
  wall minus prompt and decode time, 78.9 GB read a process); K kept 91 GB available between processes;
- isa: 12 kernels in `moe_router.hip` in both builds, no scratch, no row changed.

The identity criteria passed; the speed criterion as first written ("the second binary's prefill tok/s inside the
first's own min-max on every prompt") failed at 8192 and 32768, where two identical binaries differed by 0.09 % and
0.4 % while each side's two samples spread 0.01-0.3 %. That criterion cannot be met with two samples a side and was
the wrong zero; the run's own `verdict-as-run.json` / `summary-as-run.md` keep that first verdict, and the code now
judges a null run by a band, every prompt's ratio within 1.00 +- 0.005 (`NULL_BAND`, the floor this run measured),
and takes decode tok/s from second-request rows only (round 0 is cold). A real change is judged by its card's
criteria file, with the 0.5 % floor in mind: a gain under it needs more rounds or processes before it is a gain.

## Not there

No real swapped module has gone through yet: an earlier analysis predicted (before profiling) that strixite already
has Z's router and combine shapes, so its lever branch may stay empty; the first real `branch:` or `recipe:` run is
whichever lever comes first. The microbench covers the five interfaces with drivers under `gate/interface/`
(dense-bf16, gguf-gemv, logits-topk, moe-router, residual-add); other kernels get the full-model gate only. The loop
runs one target per invocation (one K job per stage); a sweep is a shell loop over recipes.
