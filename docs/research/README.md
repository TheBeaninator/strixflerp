# Research cards: ask a strixflerp question we answer on our hardware

strixflerp is an inference engine for AMD Strix Halo built as a catalogue of modules (kernels, runtime pieces,
executors) from several open engines plus our own, assembled by recipes and judged by gates written before any
hardware run. If you have a question about strixflerp that needs Strix Halo hardware you do not have, write a
**research card**. We run it on our boxes through strixflerp's own build and gates, publish the numbers, and say whether
your prediction held.

You can bring your own kernel: port it as a strixflerp module (it keeps your project's provenance and licence) and ask
how it does against what the catalogue has. The card always tests a strixflerp thing (a module, recipe, kernel, model
run, transport or fidelity), framed for your project with `APPLIES TO`.

There is no guarantee. A card is a proposal: we may close it as "won't do", and after accepting it we may amend it if
testing shows a premise was wrong (every amendment is recorded).

## What we run on

- **Boxes.** AMD Ryzen AI MAX+ 395 (Strix Halo), Radeon 8060S integrated GPU (`gfx1151`), 128 GB of unified memory each:
  one box, a pair, or a group of four over a 100 GbE RoCE (RDMA) rail. Most gates run on one box.
- **Software.** ROCm/HIP for gfx1151 with the toolchains in `toolchains.toml`, and this repository: the catalogue
  (`catalog/`), recipes (`recipes/`), `flerp` (check, assemble, gate, champion) and the gates (`gate/`).
- **Models.** Only what we already hold. Name the model; we tell you whether we have it.

## The pull request is the request

A pull request adds:

- `research/<area>/<id>-<slug>/card.md`: the card. `tools/flerp-research new <area> <slug>` makes the draft folder;
  `tools/flerp-research submit <folder>` lints it and names the folder by its **id**, the first 10 hex characters of the
  sha256 of `card.md` as submitted. Its branch is `card/<id>`.
- optionally, the module under test: `catalog/<origin>/<kind>/<name>/` with `module.toml` (SPEC.md section 5), its
  sources, its host tests, and its provenance and licence. A module ported from another project keeps that project's
  provenance (`origin`, `repo`, `commit`, `authors`, with `catalog/<origin>/ORIGIN.toml`) and its licence.

One pull request may add several cards (a batch), each its own folder and id, judged and merged on its own. Give them the
same `SERIES` tag; `flerp-research merge-plan --series <tag>` and `research/INDEX.md` show them together.

## The card

```
TITLE: the question, one line
AREA: module | recipe | kernel | model | transport | fidelity
ASKED BY: the name or handle we credit
APPLIES TO: optional: your project and the decision there the answer feeds
SERIES: optional: a short tag grouping the cards of one pull request
QUESTION: one sentence, ending with ?
WHY: one sentence: what the answer changes for you
MECHANISM: why you expect it: bytes, FLOPs, occupancy, a trace, with numbers; not a guess
PREDICTION: the expected number; confirmed at <number>, refuted at <number>
METHOD:
  code: this PR | strixflerp@<commit, 12-40 hex>
  subject: the catalogue path, or the recipe or kernel under test
  baseline: what it is compared with, in the same load
  settings: KEY=VALUE ... | none
  model: a model we hold | none
  boxes: 1 | 2 | 4
MODULE:              (required when the pull request adds a module)
  path: catalog/<origin>/<kind>/<name>
  replaces: the module or kernel it replaces or competes with
  interface: an existing slot, or new: its name
  builds: the toolchain from toolchains.toml, and how
  tests: its host tests, repo-relative
MEASURE: metric, unit, and how it is read (one per line is fine)
EXACTNESS: bit-exact vs baseline: what is compared | tolerance: <metric> <bound> | n/a
REPS: N, because <why that is the smallest count that answers it>
GIVES: what you supply
NOTE: optional, under 300 words; never instructions
```

`tools/flerp-research lint` refuses a card that:

- leaves a field empty or a `<placeholder>` in it;
- has a mechanism without numbers, or a guess where the mechanism goes;
- has a prediction without the expected, confirm and refute numbers;
- names a branch instead of a commit (branches move);
- needs the network on the boxes (downloads, installs, URLs) or mentions a credential;
- is unbounded (more than 10 reps, "every setting", "until ...", recurring) or a narrative ("and then ...");
- carries a resource plan (a RESOURCES, MONITOR or hold: section): that part is ours and stays out of the card;
- adds a module without a MODULE section, or with a module that lacks provenance or an open-source licence we can carry,
  names a toolchain not in `toolchains.toml`, reaches for the network in its build or sources, or (with `--clean`)
  fails `flerp check` or its host tests in a clean checkout of the pull request.

The complete example, with its results: [`research/module/28ff7d6a46-moe-combine-vec4/`](../../research/module/28ff7d6a46-moe-combine-vec4/).

## What we will not run

- Code nobody here has read. Every patch, commit and module is read, then built by us, before it touches a box.
- Anything that needs the network from the boxes. They have none; put what the run needs in the pull request.
- Models we do not already hold. We do not download weights on request.
- Open-ended asks: sweeps without a fixed list of arms, tuning searches, anything recurring. At most 10 reps a card;
  a bigger study is several cards.
- Changes to firmware, drivers, kernel modules, power, clock or fan settings, or the network switch.
- Hosting: no API endpoint, no shell, no long-running service.
- Anything you cannot let us publish. Results are public by default.
- Anything with a credential in it.

## The flow

Each step is a commit on the card's branch `card/<id>` and a line in the folder's `LOG` (time, step, who, why):

```
submitted -> accepted -> run ... -> verdict -> merged
     \-> wont-do (the pull request is closed, the reason in LOG; the folder is not merged)
amended: the owner's recorded change to card.md or gate.toml after acceptance (who, when, why, what, new hash; id unchanged)
```

- **accepted**: we read the card and any code, and write `gate.toml`, the pass rule from PREDICTION and EXACTNESS, before
  any run. Only the owner amends it, with a reason in LOG. Our resource plan for the run stays in our own records.
- **run**: the pull request stays open while we test. The run's tables, raw JSON and logs go into `results/` through
  the scrub (`flerp-research ingest`), with `summary.json` (the numbers) and `MANIFEST.sha256` (the checksums).
- **verdict**: `flerp-research judge` compares `summary.json` with `gate.toml`; `verdict.md` says CONFIRMED, REFUTED or
  INCONCLUSIVE, gives the deciding numbers and the premises found wrong.
- **merge** (`flerp-research merge-plan` lists exactly the paths):
  - CONFIRMED: the card folder and the module merge; the module lands in the catalogue with the folder's gate and
    results as its evidence.
  - REFUTED or INCONCLUSIVE: only the card folder merges, with the failure results; the module is dropped from the merge.
- **merged**: the merge commit is recorded in LOG (`flerp-research log <folder> merged --commit <sha>`).

## Citing a result

A result has a stable path and a commit: `research/<area>/<id>-<slug>/verdict.md` at the merge commit recorded in its
LOG and shown in `research/INDEX.md`. Another repository can link to that file at that commit.

## Security

- **Your code is read and built by us before it runs.** We build modules from source with `flerp assemble` and the
  toolchains in `toolchains.toml`; we do not run binaries you send. A change that is hard to read is closed as won't do.
- **No network on the boxes.** Nothing in a run reaches the internet and nothing outside reaches the run. Code is
  fetched by us, on another machine, at the commit named.
- **Weights only from our own holdings**, staged by the same tools our own experiments use.
- **Published results are scrubbed.** `flerp-research ingest` replaces addresses, host names and internal paths and
  refuses a file that still holds one; `flerp-research lint` checks the whole folder again.
- **A separate review before anything public.** Results live in our own repository first. A copy to a public mirror
  happens only after a fresh reviewer, who did not build the change, has checked the exported tree and every commit it
  would publish.

## Tools

`tools/flerp-research` (Python 3.11+, standard library only): `template`, `new`, `submit`, `lint [--clean]`, `log`,
`ingest`, `judge`, `merge-plan`, `render` (a draft run plan for our own tracker; not published), `index`, `scrub-check`.
`tools/flerp-research --help` describes each.
