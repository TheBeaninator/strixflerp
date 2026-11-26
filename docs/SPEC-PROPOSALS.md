# Proposed changes to SPEC.md, and what became of them

Found while harvesting StrixFold, gufo and hipfire (phase 2); the catalogue under
`catalog/{strixfold,gufo,hipfire,sglang,halogen}` was written to them. Phase 4 settled each one: the **Decision**
line under a proposal says what went into `SPEC.md` and `flerplib`, and what did not and why. After it,
`flerp check` over the whole catalogue gives 0 errors; the warnings left are modules that include a header the
harvest left out (they cannot build as harvested) and module cycles inside an assembly.

## 1. Interface names need a place for "same job, not the same contract"

SPEC section 2 says two modules that provide the same interface are interchangeable. Almost no two engines give the
same bits for the same job (different summation order, rounding points or operand layout), so nearly every
cross-engine pair needs different names, and then the overlap is invisible.

Proposal: `<family>.<op>/<weight-format>/<dtype>[.<tag>]`.
- The dtype field may be `in-out` when a kernel changes dtype (`bf16-f32`).
- A `.tag` names a numeric contract or layout variant (`f32.gufo`, `bf16.triton-bits`, `bf16-f32.row-invariant`).
  Providers are interchangeable only on the full name; the name without the tag is the overlap class that
  `flerp interfaces` could group by.
- When the dtype is a dash the tag rides on the op (`cache.prompt-gufo/-/-`), because the checker wants a bare dash.
  A cleaner rule would let the tag follow the dash.
- For attention and KV kernels the weight-format field carries the KV storage format (`kv-f16`, `kv-fp4-e2m1`).

`flerp check` today: accepts all of these (its pattern allows `.`, `+`, `-` in the last two fields).

**Decision: adopted, with the tag made a field of its own** (SPEC sections 2 and 7; `flerplib.catalog.split_tag`,
`overlap_class`; `flerplib.resolve.satisfies`). The pattern is now
`<family>.<op>/<weight-format>/<dtype>[.<tag>]` with at most one tag, and the tag may follow a dash
(`logits.topk/-/-.gufo-select`). `flerp interfaces` groups by overlap class. One rule was added that the
proposal did not have: a requirement without a tag may be bound, in a recipe, to a tagged provider of its class,
and the recipe must then declare `[gate] criterion`; a requirement with a tag is filled only by its full name.
**Rejected:** renaming the names where the tag rides on the op (`cache.prompt-gufo/-/-`). They stay valid names;
rewriting other lanes' manifests for a cosmetic rule is not worth a harvest round.

## 2. Manifest keys the harvest needed

`flerp check` warns that these are "not a key SPEC.md defines" (about 270 warnings, no errors; the other 30 or so are includes of left-out headers, see docs/HARVEST.md):

| key | meaning | used by |
|---|---|---|
| `notes` (top level) | free text the summary has no room for | all three origins |
| `[provenance].upstream` | provenance behind the origin: gufo's DS4 import (DS4 84cc8823), the llama.cpp-derived mmq kernels, StrixFold's Triton kernels from bertholomus/TensorFold 222a9c0b | gufo, strixfold, hipfire |
| `[build].source_options` | per-source compile options (`-mwavefrontsize64`, `-fno-slp-vectorize`) from the origin's build files; `[build].options` alone cannot express them | gufo |
| `[build].units` | a JSON file of compile units for origins that build kernels from concatenated text, not from files | hipfire |
| `[build].variants` | how many kernels / compile units / recorded variants the module covers | all three |
| `[gate].compile` | `standalone`, `header-only`, `via:<module id>`, `prebuilt` or `n/a`: how the compile gate treats the module | all three |
| `[gate].reason` | why a module is still `harvested` | all three |
| `[[prebuilt]]` | a prebuilt code object or kit: where it is, sha256, bytes, manifest | strixfold, hipfire |

Proposal: add them to section 5. `[gate].compile = "via:<id>"` matters most: a header fragment of a single
translation unit (gufo's DS4 backend is 26 headers included by one file, sharing static state) can be shown to
build only through that unit, and the manifest should say so instead of claiming `builds` itself.

**Decision: all eight adopted** (SPEC section 5; `flerplib/check.py`). Each is now validated, not just allowed:
`source_options` must name sources of the module; `units` must be a JSON file whose parts exist; `compile` is one
of `standalone`, `header-only`, `prebuilt`, `n/a` or `via:<an existing module with a translation unit>`; a
`[[prebuilt]]` object in the tree is checked against its `bytes` and `sha256`; a kit on the build host needs
`where` and a manifest in the tree or a `manifest_sha256`. Also adopted, from the phase 3 lane: `[build] closure`
(the Zig import closure a module is cut from) and a recipe `[build]` table for Zig-hosted recipes.

## 3. A module kind or rule for header fragments and prebuilt kits

SPEC section 5 has "header-only helpers are their own modules (kind `kernel`, no `sources`, only `includes`)". Two
more cases exist:
- **Fragments**: headers that hold kernels but only compile inside one translation unit of another module (gufo
  DS4). Proposal: allow them as `kernel` modules with `[gate].compile = "via:<id>"`; `status` follows the unit.
- **Prebuilt kits**: `lang = "hsaco"` / `"triton-aot"` modules have no sources to compile. Proposal: their status
  path is `harvested -> verified` (hashes checked against the manifest on the build host) `-> parity`, not `builds`.

**Decision: both adopted** as rules of the `kernel` kind, not as new kinds (SPEC section 5). `verified` is a
status only a prebuilt-language module may carry, `builds` one it may not; a `via:` module may not have sources.
Phase 4 added the third case the proposal did not foresee: a module of device code objects that an adapter
loads (SPEC section 5, "Origin compile recipes").

## 4. Verbatim vendoring when the origin does not compile files

hipfire compiles strings: a constant in `crates/rdna-compute/src/kernels.rs` may be one file, or defines plus two
files concatenated, and some sources are assembled by Rust functions. Verbatim files under `src/` are therefore not
the compile units. Proposal for section 4: a module may carry a generated `units.json` next to `module.toml` that
lists the units as ordered parts (text or a path under `catalog/<origin>`); the files stay verbatim, the unit list
is derived data with the script that made it named in `ORIGIN.toml [harvest].tool`.

**Decision: adopted** (SPEC section 4; `Module.units()`). `flerp assemble` builds a unit the same way the compile
gate does: parts concatenated into the build tree, compiled with the origin's command and the unit's flags.
Three hipfire units run inside strixite-API drivers this way (`gate/results/interface/`).

## 5. Rust and Zig modules cannot be cut finer than a crate or an import closure

A Rust crate does not build with files removed, and a Zig family imports its siblings unconditionally. The harvest
takes crates whole (hipfire-runtime holds caches, speculation, sampling, mesh and serve in one module) and gives
StrixFold's Zig modules documentary boundaries. Proposal: say in section 3 that for `rust` the module is the crate,
and for `zig` module boundaries are checked only at recipe assembly.

**Decision: adopted as written** (SPEC section 3).

## 6. Origins and toolchains

- `ORIGIN.toml` for an engine the pod uses but the catalogue does not harvest: `harvest = "none"` with
  `harvest_reason`. Add `halogen` to the origin list in section 3 (`flerplib.catalog.ORIGINS` has `sglang`, not
  `halogen`; an origin with no modules passes the check either way).
- Harvested origins use a `[harvest]` table (`mode`, `tool`, `unharvested`), as catalog/strixite does. The task
  text for phase 2 asked for a `harvest` string field; a table with `mode = "modules" | "none"` would be one shape
  for both.
- `toolchains.toml` has no entry for the system ROCm 7.2.4 (`/opt/rocm-7.2.4`) or for the Triton AOT toolchain.
  All 1,046 hipfire units compile with TheRock 10.2 (the recorded run); 971 of them were also compiled with
  `/opt/rocm-7.2.4` in an earlier run. The catalogue names TheRock 10.2 so that it stays inside the table. Proposal: add `rocm-7.2.4` and a `kind = "triton"`
  entry (wheel sha256, python version).
- `therock-10.1` in `toolchains.toml` is `/data/tools/therock-10.1.0a20260823`, whose clang reports 23.0.0git;
  gufo's flake says ROCm 10.1 with HIP clang 24. The gufo gate ran with the table's entry.

**Decision:**
- `halogen` added to the origin list; an origin with `harvest = none` must give a reason and may hold no modules
  (checked). **Adopted.**
- One shape for `harvest`: **adopted in the checker, not forced on the files.** `[harvest] mode = "modules" |
  "none"` is the form; the string `harvest = "none"` with `harvest_reason` (sglang, halogen) is accepted as the
  same thing, because rewriting two files to change a spelling proves nothing.
- `rocm-7.2.4` and a `kind = "triton"` entry (version, python tag, wheel sha256, target) **added** to
  `toolchains.toml`; a triton entry needs no `rocm_root`. The catalogue keeps naming TheRock 10.2 for hipfire.
- The clang 23 against clang 24 difference for gufo is **recorded in `toolchains.toml`, not resolved**: there is
  no TheRock with clang 24 for ROCm 10.1 on the pod. gufo's kernels built with the table's entry pass the phase 4
  module gates they take part in, which is evidence for those kernels and for nothing else.
- New in phase 4: `ORIGIN.toml [compile]` is now load-bearing (SPEC section 5, "Origin compile recipes"), and
  `catalog/flerp/ORIGIN.toml` exists for code written here.

## 7. Section 1, fact 5: the launch cost figure

"Kernel launch on gfx1151 is cheap (about 0.024 ms)" has no single source (docs/SOURCES.md, section 5): strixite's
own document says about 2.8 us per launch, and the pod's per-class measurements give about 18 to 23 us per launch at
one row, where they call StrixFold's shared-expert class "launch-bound at 1 row". Proposal: state both numbers with
their sources, and weaken "so static assembly with no runtime dispatch layer is affordable" to what the sources
support: launch count matters at one row (strixite merged five launches into two per hyper-connection mix for this
reason, `kernels/hc.hpp:5-6`), and a dispatch layer adds nothing useful when the recipe already fixed the kernels.

**Decision: adopted, with a third number measured here.** SPEC section 1 fact 5 now gives 2.8 us (strixite's
document, enqueue), 18 to 23 us (the pod's per-class figure, per launch as a one-row layer sees it; the old
"0.024 ms" is of this size and no source gives it as a launch cost), and 2.0 to 2.2 us per back-to-back launch
measured by the phase 4 interface gate (`gate/results/interface/residual-add/result.json`, case
`bench-decode-4x2560`), and draws the narrower conclusion.

## 8. Section 9, phase 2 gate wording

"each kernel module compiles standalone for gfx1151" cannot hold for header-only modules, fragments and prebuilt
kits. Proposal: "each `hip-c++` kernel module with sources compiles standalone for gfx1151 with its origin's compile
recipe; header-only modules and fragments are covered by the unit that includes them; prebuilt kits are recorded
with hashes." This is what `gate/compile/run.py` implements: 46 of 46 such modules pass; 13 kernel modules are
header-only, fragments or prebuilt.

**Decision: adopted** (SPEC section 9, phase 2).

## 9. Not proposed in phase 2, decided in phase 4

- **Module cycles are warnings.** gufo and hipfire have modules whose headers include each other
  (`runtime.gufo.platform` and `runtime.gufo.sampling`; `kernel.hipfire.tensor_ops`, `gemm_mq`, `norm_rope`).
  `flerp assemble` used to refuse such a closure; it now reports the cycle and goes on, since the set of modules
  and their include roots are the same whichever edge is cut.
- **Includes of left-out headers are flagged only on unproven modules.** A module whose status is past
  `harvested` has shown that the include sits under a conditional its build does not take (libwebp's
  `config.h`); a `harvested` module that includes a file the harvest left out cannot build as it stands, and the
  warning says so (30 of them, all in gufo's executor and serve modules).
- **Adapters and origin compile recipes** (SPEC section 5): the manifest table `[adapter]`, the compile-time
  symbol rename, and code objects embedded per unit.

## 10. Phase 3: Zig-hosted recipes and the executor ABI

Found while assembling StrixFold (`recipes/dsv41-flash.strixfold-tp2.toml`) and writing `include/flerp/executor.h`.
Each was decided on 2026-11-11; SPEC.md and `flerp check` hold what was adopted.

Adopted:

- **A recipe may be built by its origin's build graph** (SPEC section 7). The `[build]` table of a Zig-hosted
  recipe is specified key by key (`system`, `zig`, `root`, `steps`, `options`, `build_file`, `headers`,
  `[build.prebuilt]`, `[build.installs]`, `[build.hipcc_output_pins]`); `flerp check` types them, requires
  `system`, `zig`, `root` and `steps`, resolves `root`, the prebuilt module ids and the installed binary names, and
  warns on a key it does not know. Section 8 no longer says every assembly is CMake + Ninja, and `flerp assemble`
  prints `build.sh` for such a recipe.
- **Zig module boundaries are checked at assembly, and one import closure may span modules** (section 3):
  `[build] closure` and the rule `assemble_zig` enforces are in the text.
- **Test files are part of a Zig module** (section 4).
- **`.gitignore` is anchored** (`/build/`): `catalog/strixfold/tool/zig_build/src/zig/build/` is no longer
  ignored and needs no `git add -f`.
- **A Zig toolchain entry** (section 10, `toolchains.toml`): `kind = "zig"` with `root` (and `bin`, `version`).
  A recipe's `[build] zig` names it (`zig-0.17`); an absolute path still works and `FLERP_ZIG` still overrides.
  A zig entry is refused as a recipe's or a module's `toolchain`, which name ROCm installs. The two StrixFold
  recipes were switched to the name and assemble to byte-identical trees, `build.sh` and `assembly.json`.
- **Section 6, the executor ABI**: three deliveries, two draft forms, capability flags for snapshots and the rest,
  as the header has them, plus what the conformance runs added: the shape of a forward is the executor's numeric
  contract, not the ABI's (section 1, fact 3).
- **Section 10**: the worker rule has its phase 3 exception (the pair, under a hold, through the pod's launcher),
  and the rule for control-node's shared GPU is written down.

Rejected:

- **A rule for alternatives chosen at run time, so that they share one interface name.** StrixFold's three
  collective transports and its two RoPE kernels keep distinct names (`collective.allgather-tcp/-/f32`, `-rdma`,
  `-nccl`; `pos.rope-heads/-/bf16.hip`, `.triton-bits`). A recipe binds one module to an interface and "the same
  full name" means "may replace each other in a recipe"; modules that are all compiled in and all required are
  not that. One name for them would need a second kind of binding (a set) for a case with one user.
- **Recording compile command strings with every prebuilt kit.** The only string of the command that a checkout
  cannot reproduce is the `-o` path, and the recipe that rebuilds a kit records it (`[build.hipcc_output_pins]`,
  now in section 7). A kit that is only verified by hash (`[[prebuilt]]`) is never rebuilt, so it needs no
  command; a manifest key for it would be data nothing reads.
- **Moving the conformance driver to kind `tool`.** Its id (`executor.flerp.conformance`) is in two recipes, a
  Zig build file and the gate evidence, and it is test code for the executor ABI, not a converter or inspector
  that ships with a recipe. Section 3 now says the `executor` kind covers the ABI's adapters and their driver
  under origin `flerp`.

## 11. Phase 5: a full-precision reference as a recipe's north star

Proposed after the qwen35moe recipe's gate (`gate/results/qwen36-35b-a3b.flerp-moe/`), not yet in SPEC.md.

- **Section 9, new paragraph: what a recipe for a model no origin serves is held to.** Sections 1 and 7 define a
  recipe gate as token identity (or agreement) with the origin engine. Where no origin runs the model, the gate
  has so far compared with another engine (llama.cpp), and that made another engine's rounding the target: the
  recipe failed a log-probability tolerance against llama.cpp that the origin's own dense path also sits at the
  edge of. Proposed wording: *"Where a recipe has no origin binary to equal, its reference is a full-precision
  evaluation of the weights as shipped: the stored weights dequantized, every activation in FP32, no activation
  quantization. A recipe gate measures the executor against that reference (KL divergence from the reference,
  the largest log-probability difference over tokens the reference gives real mass, the best token) and holds it
  to bounds written before the run. Other engines on the same inputs are measured against the same reference and
  recorded beside it; they are yardsticks, not criteria."* `docs/FIDELITY.md` has the metrics and what was found.
- **Section 5, `[gate]`: say what a reference is made of.** A module or recipe whose evidence rests on a
  reference written in this tree (the MoE CPU reference is) names what gives that reference standing:
  `reference_checked_by = "gate/moe_block_check.py"` or the like. A reference that only its own author's code
  agrees with is not evidence.
- **Section 8: gate stages that are asked for by name.** `flerp gate recipe <name> --stages reference,fidelity`
  runs stages that are not part of the recipe's verdict. The runner keeps the two apart (`STAGES`,
  `FIDELITY_STAGES` in `gate/moe_recipe.py`) so a later measurement cannot rewrite an earlier result. If the
  proposal above is taken, the fidelity stages become the recipe's verdict for recipes without an origin binary
  and the engine-against-engine comparison becomes the record.
- **Not proposed:** changing the failed criterion of the gpu stage. It stays as set and as failed.

## 12. Sub-origins and overlay modules: harvesting a pull request

Proposed while harvesting Shoestring-dev/Strixfold pull request #6 into `catalog/strixfold_pr6`
(`docs/VISION-DSV41.md`). The first point is in SPEC.md section 3 and `flerplib/catalog.py` (`ORIGINS`); the rest is
not in SPEC.md, and `flerp` reads none of it yet.

- **Sub-origin (taken).** A pull request is a tree at its own commit. More modules of `strixfold` with another
  `[provenance] commit` would have broken three things that assume one commit an origin: `flerp diff-origin --origin
  strixfold` against the deployed commit (the dsv41 recipe gate's `origin` stage would report the PR's files as
  "missing in the origin checkout"), `docs/harvest/harvest_strixfold.py` (it deletes module directories it did
  not write, so `gate/strixfold_recut.py` would remove them) and the origin's file and module counts. So the PR
  is `catalog/strixfold_pr6` with `parent = "strixfold"`, `base` (the merge-base) and `state` in its
  `ORIGIN.toml`; `flerp diff-origin --origin strixfold_pr6 <clone of the PR>` proves it (23 of 23 files).
  Proposed for `flerp check`: read `parent`, and require a sub-origin's `[licence] spdx` to be the parent's.
- **Overlay modules (proposed).** A PR modifies files that modules of the parent already hold. Those modules are
  not touched. The sub-origin's module holds the PR's version of each such file, verbatim at the same
  origin-relative path (so the verbatim rule and `diff-origin` hold for it), and next to `module.toml`:
  `overlay.json` (per file: the parent module, sha256 of the PR base, PR head and the parent's file, whether the
  PR's change applies to the parent's file and with how much `patch` fuzz, sha256 of the result) and
  `overlay/<path>.patch` (`git diff <base> <head> -- <path>`). Both are derived data, as `units.json` is, written
  by `ORIGIN.toml [harvest] tool`. The manifest says so in `[provenance] upstream` and `requires` each parent
  module by id. Proposed manifest form in place of the prose: `[overlay] on = ["<module id>", ...]`,
  `data = "overlay.json"`.
- **What `flerp check` should then verify:** every overlay path is in the named parent module; `parent_sha256`
  equals the parent's file today (a re-cut of the parent makes the record stale: re-run the harvest tool);
  `pr_head_sha256` equals the file under `src/`.
- **What `flerp assemble` should then do** (today two modules that own one tree path cannot be assembled
  together; section 5, "Derived modules", has the same rule for origin `flerp`): for a recipe that includes an
  overlay module, write the parent's file with the patch applied at that tree path, refuse when the patch does
  not apply with fuzz 0 unless the recipe names the file under `[build.overlay_fuzz]`, and record the resulting
  sha256 in `assembly.json`. The patch, not the PR's whole file, is what an assembly needs: 6 of the 9 files
  differ between the PR's base (upstream 70cfb8c) and the catalogue (f64b5ff, the pod's patches), and the PR's
  whole file would drop the pod's changes. Two of the nine (`families/dsv41/lanes.zig`, `native/hip.zig`) apply
  only with fuzz 2: a person has to read those two merges before a recipe is built on them.
- **A `[build] lang` for scripts.** `tool.strixfold_pr6.prepare_vision` is three Python and JSON files and says
  `lang = "zig"` because section 5 has no value for "nothing is compiled". Proposed: `lang = "script"`.
- **Whole-file overlays (taken for Zig recipes, pod card bb68604fa9).** When the sub-origin is the reference itself,
  not a patch on the parent (`catalog/strixfold_tp4`, the TP4 serving tree 003f679c over f64b5ff), the assembly needs
  the sub-origin's whole file, not a patch applied to the parent's. `[build.overlay] "<id>" = "."` in a Zig recipe
  (`flerplib/assemble_zig.py`): the module's files sit at their own origin paths, a file a required module holds is
  replaced, any other is an addition; `assembly.json` records both hashes. `docs/DSV41-TP4.md`.
- **Not done:** `docs/HARVEST.md` is generated and does not list the sub-origins (`make_harvest_md.py` `ORIGINS`).
