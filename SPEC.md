# strixflerp: specification

strixflerp is one inference engine for AMD Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S, gfx1151, ROCm/HIP),
built as a catalogue of modules. An engine for a given model is a **recipe**: a list of modules, assembled at build
time into one binary. "strixite" and "strixfold" are two recipes whose module lists overlap.

This file is the contract every contributor (human or agent) builds against. If the code and this file disagree,
fix one of them in the same commit.

## 1. Facts the design rests on

Measured on the pod or read from the engines' source (see `docs/SOURCES.md` for where each came from):

1. The cheap seams between engines are HIP kernel source, compiled code objects, and the C ABI. Host languages
   differ (C++ strixite and gufo, Zig StrixFold, Rust hipfire, Python SGLang).
2. The weight format is baked into every compiled weight-reading kernel: strixw Q4/Q8 groups, EXL3, ggml K-quants,
   hipfire MQ/MFP4 and FP8 each fix a different inner loop, and a kernel only reads the format it was written for.
   Its recipe is not baked in: the algorithm, tiling, work mapping and scheduling are reusable and get re-cooked for
   another format, a different size pan. That re-cooking is what cross-engine use means in strixflerp, gated by bit
   identity where the math is unchanged and by the fidelity rule (`docs/FIDELITY.md`) where it is not
   (`docs/SHAPE-TRANSFERS.md`).
3. Activation dtype and rounding points differ per engine and carry guarantees (strixite TP bit identity, StrixFold
   output-invariant drafts). Mixing kernels silently breaks them unless the numeric contract is declared and tested.
   **The shape of a forward is part of strixite's numeric contract**: its session picks kernels by the rows a
   forward carries (1, 2 to 5, 6 to 11, 12 and more: `runtime/qwen4exp.hpp` `kGroupedMinTokens`, `kWmmaMinTokens`,
   and lines 800-805), the results differ by rounding across those classes, and greedy tokens then part at near
   ties. The reference server does it to itself: MTP on (several rows a verify) against MTP off (one row a forward)
   differs from token 43 of one gate prompt and token 95 of another
   (`gate/results/qwen38-flash-next.strixite/identity.json`). So token identity with strixite is only ever asked
   between runs that make the same forwards - the same prompt cuts, the same rows a verify, the same rows kept -
   and a gate that compares across shapes reports what it measured and requires nothing
   (`docs/EXECUTOR-ABI.md`, "Forward shape").
4. Executors (the per-model fused pipeline) are not unifiable today. They are swappable behind one ABI.
5. A kernel launch on gfx1151 costs microseconds, and what it costs depends on how it is counted:
   - about 2.8 us to enqueue one, in strixite's own account (`strixite-v070pod/docs/how-it-got-fast.md:243-244`;
     about 1,130 launches and 3.18 ms of host turnaround and gaps per token at `:64-70`);
   - 2.0 to 2.2 us per launch measured here, queued back to back with one synchronize at the end, for a kernel
     that touches 123 kB (`gate/results/interface/residual-add/`, case `bench-decode-4x2560`: 2.01 us strixite's
     kernel, 2.19 us gufo's, 1.98 us hipfire's);
   - about 18 to 23 us per launch as a one-row layer sees it, in the pod's per-class measurements: three launches
     cost 55 to 70 us a layer against the weight read, and that source calls the class "launch-bound at 1 row"
     (`meta/meta-o2-perclass.md:238-239`, `meta/meta-s1-widget-catalog.md:185`). The "about 0.024 ms" this fact
     used to state is of that size; no source gives it as the cost of one launch.
   Memory bandwidth is the ceiling for everything that reads weights (read peak about 226 GB/s,
   `how-it-got-fast.md:53-58`; 208 GB/s on a 252 MB residual add here). So launches are not free at one row:
   strixite merged five launches into two per hyper-connection mix for this reason (`kernels/hc.hpp:5-6`), and an
   adapter that needs five launches where the origin needed two is measurably slower
   (`gate/results/interface/logits-topk/`). What follows for the design is narrower than "launches are cheap":
   a recipe fixes its kernels at build time, so a run-time dispatch layer would add launches and indirection
   and decide nothing. Static assembly is what the fast engines already do (`docs/SOURCES.md` section 5).

## 2. Vocabulary

- **module**: the unit of the catalogue. A directory with a `module.toml` manifest and its sources.
- **interface**: a named contract a module provides or requires, e.g. `linear.decode/strixw-q4/bf16`. Two modules
  that provide the same interface are interchangeable in a recipe. Interface names are
  `<family>.<op>/<weight-format-or-dash>/<activation-dtype-or-dash>[.<tag>]`:
  - the dtype field may be `in-out` when a kernel changes dtype (`bf16-f32`);
  - for attention and KV kernels the weight-format field carries the KV storage format (`kv-f16`, `kv-fp4-e2m1`);
  - a `.tag` names a numeric contract or operand layout that is not the untagged one (`f32.gufo`,
    `bf16.triton-bits`, `bf16-f32.row-invariant`). It may follow a dash (`-.gufo-select`).
  Providers are interchangeable only on the **full name**. The name without its tag is the **overlap class**:
  the same job on the same operands, with a gap (summation order, a rounding point, an edge rule) that an
  adapter or a tolerance has to account for. `flerp interfaces` groups by overlap class.
- **adapter**: a module of origin `flerp` that lets a provider from one engine satisfy the launch API another
  engine's executor calls (section 5, "Adapters").
- **recipe**: `recipes/<name>.toml`. Picks one module per required interface, plus toolchain and deployment.
- **assembly**: `flerp assemble <recipe>` resolves the module closure, checks interfaces, and generates a build
  directory that compiles exactly those modules into `strixflerp-<recipe>`.
- **gate**: mechanical checks. Module gates (compiles for gfx1151, parity against a reference, bandwidth bench) and
  recipe gates (token identity against the origin engine, tokens per second). Gate code lives in `gate/` and is not
  edited to make a result pass.

## 3. Module kinds

| kind | what it is | examples |
|---|---|---|
| `format` | a weight or state storage format: host loader, converter, device view | `format.strixw`, `format.exl3`, `format.gguf-kquant` |
| `kernel` | one or more gfx1151 kernels with a host-callable launch API | `kernel.strixite.linear_q4`, `kernel.strixfold.exl3` |
| `executor` | a model-family pipeline behind the executor ABI (section 6); under origin `flerp`, the ABI's adapters, the conformance driver that tests them, and a stand-in executor with no model for testing callers of the ABI | `executor.strixite.qwen4exp`, `executor.strixfold.dsv41`, `executor.flerp.strixite_abi`, `executor.flerp.fake_abi` |
| `runtime` | engine-agnostic services: device memory, KV and prompt cache tiers, slots, sampling, grammar, speculation, transport, verification; under origin `flerp`, the decode loop over the executor ABI and the modules that put a model's tokenizer and chat template, and a host sampler, behind the C interfaces of section 6 | `runtime.strixite.prompt_cache`, `runtime.strixfold.coll_rdma`, `runtime.flerp.decode_loop`, `runtime.flerp.text_strixite` |
| `serve` | the API front end: an origin's own server, or the one written here that serves any executor behind the ABI | `serve.strixite.http`, `serve.flerp.http` |
| `tool` | converters and inspectors that ship with a recipe | `tool.strixite.convert_qwen4exp` |

Module ids are `<kind>.<origin>.<name>`. `origin` is the engine the code came from (`strixite`, `strixfold`, `gufo`,
`hipfire`, `crucible`, `tensorfold`, `sglang`, `halogen`, `ds4`) or `flerp` for code written here. An origin the pod uses but the catalogue
does not harvest (`sglang`, `halogen`) has an `ORIGIN.toml` and no modules. A tree of an origin at another commit
that is harvested beside it (an open pull request) is a sub-origin with its own directory and `ORIGIN.toml`, named
`<origin>_<what>` (`strixfold_pr6`, `parent = "strixfold"`): an origin has one commit, and `flerp diff-origin` proves
each origin against one checkout. It holds only what that tree adds or changes (`docs/SPEC-PROPOSALS.md` section 12).

How fine a module can be cut depends on the language. C and C++ files and HIP sources are cut by file. A Rust
module is a crate: a crate does not build with files removed. A Zig module is a cut of an import closure, and
its boundaries are checked when a recipe is assembled, not module by module: every relative `@import` and
`@embedFile` of the assembled tree must resolve inside the tree, to a file of the importing file's own module, of
a module it requires, or of a module that declares the same `[build] closure` (`zig:core`, `zig:families/dsv41`).
The closure key is there because Zig files import each other in both directions and `requires` may not form a
cycle: it names one closure cut into several modules instead of merging them.

## 4. Layout

```
SPEC.md  README.md  toolchains.toml
flerp                    CLI entry (python3, standard library only; the pod is air-gapped)
flerplib/                CLI implementation
catalog/<origin>/<kind>/<name>/module.toml
catalog/<origin>/<kind>/<name>/src/...        sources, original relative paths kept under src/
catalog/<origin>/ORIGIN.toml                  repo, commit, licence, toolchain the origin was built with
include/flerp/           ABIs written here (executor.h, text.h, sampler.h, decode_loop.h); a module of origin `flerp`
                         names it as an include root, and it belongs to no module
recipes/<name>.toml
gate/                    gate runners and references
build/                   generated, never committed
docs/
```

Vendored sources are copied **verbatim** with their relative paths preserved under `src/`, so includes keep
working and a diff against the origin is empty. A change to vendored code is a separate commit that says why, and
the manifest's `modified = true` is set with a one-line reason.

A Zig module's test files (`*_test.zig`, `tests.zig`, `fake.zig` next to its sources) are part of the module:
Zig loads every file a compiled file imports, `test` blocks included. An origin's separate test roots and test
programs are not harvested with it.

When an origin does not compile files (hipfire compiles Rust string constants: one source, or defines plus
several sources concatenated), the files under `src/` stay verbatim and the module carries a generated
`units.json` next to `module.toml` (`[build] units`): the compile units as ordered parts, each a text or a path
under `catalog/<origin>`. It is derived data; `ORIGIN.toml [harvest] tool` names the script that made it.

## 5. module.toml

```toml
id        = "kernel.strixite.linear_q4"
kind      = "kernel"
summary   = "Q4 group-affine GEMV for decode, split-K, BF16 or F32 activations, FP32 accumulate"
why       = "Decode is memory-bound on gfx1151, so a split-K GEMV that streams each Q4 weight once at full bandwidth sets the token rate of every dense layer; it is strixite's reference decode path and the baseline the other origins' GEMVs are measured against"

[provenance]
origin    = "strixite"
repo      = "https://github.com/shawnshekari/strixite"
commit    = "20937dc"            # commit of the tree it was copied from
branch    = "v0.7.0-pod"
paths     = ["kernels/linear_q4.hip", "kernels/linear_q4.hpp"]   # origin-relative: files, directories or globs
authors   = ["shawnshekari"]     # plus "pod" when the pod's patches touch these files
modified  = false

[licence]
spdx      = "AGPL-3.0-only"      # SPDX id; "LicenseRef-proprietary" when closed

[build]
lang      = "hip-c++"            # hip-c++ | c++ | c | zig | hsaco | triton-aot | rust | python | cuda (CUDA sources kept to read their recipe; never built here)
sources   = ["src/kernels/linear_q4.hip"]
includes  = ["src"]              # include roots this module exports to itself and to the modules that require it
arch      = "gfx1151"
toolchain = "therock-10.1"       # named in toolchains.toml
# optional: system = true (third-party headers: included as system headers), uses = ["hip-runtime"] (this module
# and everything that requires it compiles and links against the toolchain's HIP runtime), defines = [...],
# options = [...] (compile options for its own sources), c_std = 11,
# source_options = { "src/a.hip" = ["-mwavefrontsize64"] } (per-source options from the origin's build files),
# units = "units.json" (compile units instead of sources, section 4), variants = N (kernels or units covered),
# closure = "zig:families/dsv41" (the import closure a Zig module is cut from)

[[provides]]
interface = "linear.decode/strixw-q4/bf16"
api       = "strix::kernels::linear_q4"          # the host-callable symbol(s)
numeric   = "fp32-accumulate, one rounding to bf16 on store"
regime    = "M 1..16 rows, memory-bound"

[[requires]]
interface = "format.device-view/strixw-q4/-"     # satisfied by format.strixite.q4_device
# requires may also name a module id directly:  module = "kernel.strixite.qsplit"

[gate]
parity    = "none"               # none | origin-test:<path> | flerp:<gate/..>
bench     = "none"
status    = "harvested"          # harvested -> builds -> parity -> benched (a prebuilt kit: harvested -> verified -> ..)
# optional: compile = "standalone" | "header-only" | "via:<module id>" | "prebuilt" | "n/a" (how the compile gate
# treats the module), reason = "..." (why it is still harvested), evidence = "gate/results/<file>"
```

The three sentences a reader needs come first, each in the module's own words: `summary` says what the module is and how it works (one sentence: the algorithm, data layout, precision, what it equals); `why` says why it exists (the problem it solves or the bottleneck it targets, the role it plays in recipes, what it replaces or was measured against); `notes` holds the rest (card and gate, origin it was re-expressed from, compiled shapes, callers). Reading only `summary` and `why` across the catalogue should teach the catalogue's ideas, so no two modules share wording and a `why` is never invented: it comes from the module's source, its origin's docs or a recorded measurement. `tools/describe-audit.py` lists modules missing either.

Other optional keys: `notes` (top level; free text the summary has no room for), `[provenance] upstream`
(provenance behind the origin: gufo's DS4 import, the llama.cpp-derived kernels, StrixFold's Triton kernels), and
`[[prebuilt]]` (a prebuilt code object in the tree: `file`, `sha256`, `bytes`, which `flerp check` verifies; or a
kit on the build host: `kit`, `where`, and `manifest` (a file of the module) or `manifest_sha256`).

`design` (top level, optional) names a design record of the module: a Markdown file beside `module.toml` (by
convention `DESIGN.md`), never under `src/`, which holds only origin files. It says what the module is for in its
engine, the format choices it relies on, what they buy and cost (with the origin's measurements), which modules it
is coupled to and why, and file:line evidence, keeping what the origin's code or docs say apart from what was
inferred. `flerp check` errors when the file is missing, under `src/`, or outside the module directory.

Rules:
- Every module has provenance and an SPDX licence. `flerp list --licence <expr>` and
  `flerp assemble --licence <expr>` filter on it. Nothing else in the tree treats licences specially.
- `provides`/`requires` are the only way modules refer to each other. Header-only helpers that several kernels
  include are their own modules (kind `kernel`, no `sources`, only `includes`). `flerp check` enforces it for
  C-family sources: a header a module includes from another module of its origin must come from a module it
  requires, and `flerp assemble` gives each module the include roots of its requirement closure only.
- Every file under `src/` is named by `[provenance] paths`, and an origin file is in exactly one module.
- `status` only moves forward when the gate that proves it has run on real gfx1151 hardware, and the manifest
  records where the evidence is (`gate.evidence = "gate/results/<file>"`).
- **Fragments**: headers that hold kernels but compile only inside one translation unit of another module (gufo's
  DS4 backend is 26 headers included by one file, sharing static state) are `kernel` modules with
  `[gate] compile = "via:<id>"`; their status follows that unit and they claim no build of their own.
- **Prebuilt kits** (`lang = "hsaco"` / `"triton-aot"`) have nothing to compile: their path is
  `harvested -> verified` (hashes checked against the manifest on the build host) `-> parity`.
- A cycle between modules (headers of two modules that include each other) is reported by `flerp check` and
  `flerp assemble` and does not stop an assembly: the closure is the same set of modules either way.

### Origin compile recipes

An origin whose kernels are not built the way the assembly builds strixite's says how in `ORIGIN.toml [compile]`:
`driver`, `arch_flag`, `rocm_flag` or `rocm_flags`, `flags`, `system_includes`. A `hip-c++` module of such an
origin is compiled by exactly that command (with the module's `options` in place of `flags` when it has them, its
`source_options`, and its own `toolchain`), by the compile gate and by `flerp assemble` alike. Two products:
- **host objects** (gufo: `clang++ -x hip`, device code embedded): the module's sources become a static library
  and a binary links the objects it needs;
- **device code objects** (StrixFold: `hipcc --offload-device-only`; hipfire: `hipcc --genco`): nothing is
  linked. An adapter names the units it loads (`[[adapter.code_object]]`); `flerp assemble` compiles those units
  and embeds each in the binary as bytes between the symbols `<symbol>` and `<symbol>_end`.
So one binary can hold kernels built by three compile recipes and two toolchains; `assembly.json` records each
command.

### Adapters

An adapter is a `kernel` module under origin `flerp` (`catalog/flerp/kernel/<name>/`, new code, its own SPDX id)
that re-provides a launch function of module X on top of a provider from another engine:

```toml
[adapter]
for       = "kernel.strixite.logits_topk"                  # the module whose launch API it re-provides
symbols   = ["logits_topk", "logits_topk_workspace_bytes"] # the identifiers it takes over

[[adapter.code_object]]                                    # only for providers that are device code objects
module    = "kernel.hipfire.tensor_ops"
unit      = "add_inplace"                                  # a units.json unit, or a source path
symbol    = "flerp_co_hipfire_add_inplace"
```

Rules:
- The adapter `requires` module X (its header declares the API the executor includes) and the foreign module.
- In an assembly that holds the adapter, X's translation units are compiled with
  `-D<symbol>=<symbol>_flerp_origin`: the adapter defines the name every caller uses, and X's own kernel stays
  linked under the renamed symbol, so the adapter can hand back the calls the foreign provider does not cover
  (another dtype, a shape outside its regime, an argument error X reports itself). No vendored file changes and
  `flerp diff-origin` stays empty. The rename covers calls inside X's own translation units too: those keep
  X's kernel.
- An adapter is thin: argument order and layout, the launch, and at most small kernels of its own for what the
  consumer's contract has and the provider's lacks (a NaN flag, an error word). It owns no device memory beyond
  what the consumer's API hands it. Its header comment lists every point where the two contracts differ.
- It counts launches that went to the foreign provider and launches handed back (`flerp/adapter_rt.hpp`), and
  prints both; the gates read those lines. A swap that never launched the foreign kernel has not run.
- Naming: an adapter whose results are the consumer's bits on every input provides the consumer's interface name
  (shown by its module gate). Any known gap makes it a tagged name of the same overlap class.

### Derived modules

A module of origin `flerp` may hold **adapted copies** of an origin's files when a change cannot be made beside
them (a struct that gains members, a loader that reads more tensors, a call site that branches). The origin's
module stays byte-identical and its origin gate keeps passing; the copies are new files of the derived module:

- they sit under a root of their own in the tree (`flerp/<name>/<origin-relative path>`), so no two modules own
  one tree path, and every change in them is marked with the derived module's id;
- the manifest's `[provenance] upstream` names the origin, its commit and the files, `notes` says what was added,
  and the SPDX id is the origin's;
- the derived module `requires` the origin's module, and the recipe's build file puts the derived root ahead of
  the origin's as an include root and swaps the adapted sources into the origin's target. A source the target
  does not name stops the configure. Nothing in the origin's build files changes.

In a Zig-hosted recipe there are no include roots to order: a Zig file imports its siblings by relative path. There
the recipe names the derived module and its root under `[build.overlay]`, and `flerp assemble` puts each file under
that root at the origin-relative path in the tree, in place of the origin's file, which must belong to a module the
derived one requires; `assembly.json` records both hashes, and the derived module declares the origin's
`[build] closure`. `executor.flerp.dsv41_zip2` is one (two files of `executor.strixfold.dsv41`).

In a recipe whose CMake `flerp assemble` generates (no `[build] system`), the recipe lists derived modules under a
top-level `[overlay]` table, derived module id = the module it overlays. The derived module requires it; each of its
sources at `src/<its include root>/<path>` compiles in the origin's target in place of the origin's `src/<path>` (a
source with no counterpart is an error), its include root goes ahead of the origin's on every include path that holds
the origin's, and the origin's target also gets the derived module's requirements. One definition of every symbol per
binary, and the modules that require the origin by id link the derived files unchanged. `assembly.json` records the
sources under `overlays`. `recipes/qwen38-flash-next.strixite-p16.toml` overlays `format.strixite.strixw` (strixw v5)
and `executor.strixite.qwen4exp`.

`executor.flerp.qwen35moe` is one: seven files of `executor.gufo.qwen`, with a mixture-of-experts feed forward.
The recipe gate records, file by file, how many lines differ from the origin's (`gate/moe_recipe.py`, stage
`origin`).

## 6. Executor ABI (`include/flerp/executor.h`)

C ABI, so C++, Zig and Rust executors all fit. It is written from strixite's `LmBackend`
(`serve/engine.hpp`) and StrixFold's lane backend vtable (`zig/src/core/lanes/backend.zig`), reduced to what a
serving loop needs; `docs/EXECUTOR-ABI.md` maps every call of both onto it and names what does not map.

```
caps            positions, ids a forward, rows a verify, logits row width, seqs, snapshots, draft depth,
                deliveries offered, capability flags
open/close      load weights for a recipe + deployment (string keys, the executor's own); one executor a process
seq_new/free    a conversation slot; reset, position
forward         run ids at the seq's position; results for the last rows asked for, in one delivery
draft           the executor's own draft head, in one of two forms
verify          a forward whose effect is pending; then exactly one of keep, keep_prefix(rows), drop
snapshot        save / restore / drop a seq's state; export / import as host bytes; write / read as files
stats           the ABI's own counters, and the origin's as JSON
```

- **Deliveries.** What a forward hands back is one of full logits rows, each row's top candidates, or the token
  the executor draws for each row. strixite hands logits to the host and the host samples; StrixFold draws on the
  device and never shows logits. Tokens are the one delivery both offer (strixite's adapter: greedy only), so a
  front end that serves either takes tokens for greedy requests, and sampled output is per executor.
- **Drafts.** A chained step returned to the caller (strixite's MTP head: `flerp_draft_step`), or drafts held on
  the device and named by count in the next verify (StrixFold: `flerp_draft_hold`). The policy that decides how
  many drafts a verify gets (strixite's margin rule, StrixFold's depth rule) is the caller's, not the executor's.
- **Capability flags.** An executor says what it has instead of failing late: masks, keep_prefix without a second
  run, several seqs a verify, snapshots, snapshots at any position (StrixFold keeps a state only at a prompt
  chunk end and restores it only into a prompt pass), states as host bytes, states as files, sampled tokens.
- **Forward shape.** The ABI lets a caller choose how ids are grouped into forwards. Whether that changes the
  numbers is the executor's contract, not the ABI's: for strixite it does (section 1, fact 3), for StrixFold
  drafts are output-invariant. A caller that needs one origin's exact tokens makes that origin's forwards.
- **Versions.** Version 2 appends to version 1 and changes nothing in it: `flerp_caps.sampling` and `.top_k_max`
  say what an executor that draws on the device accepts as sampling parameters (StrixFold's keyed sampler needs
  `top_k` and bounds it), so a front end refuses such a request before it runs. A struct an executor fills is
  written for at most the caller's `size` and returns the size the executor knows, in both directions. The
  interface name stays `executor.abi/flerp-v1/-`: a version that only appends is the same interface.

Phase 1 does not force strixite through this ABI: its executor and server are assembled as they are. Phase 3 has
the adapters (`executor.flerp.strixite_abi`, `executor.flerp.strixfold_abi`), a conformance driver that uses only
the header, and one front end.

### What sits on the ABI

Three more C interfaces in `include/flerp/` make a front end independent of the engine. A recipe binds a module to
each, as it binds the executor:

| interface | header | what | providers |
|---|---|---|---|
| `executor.abi/flerp-v1/-` | `executor.h` | the model | `executor.flerp.strixite_abi`, `executor.flerp.strixfold_abi`, `executor.flerp.fake_abi` (no model) |
| `text.model/flerp-v1/-` | `text.h` | the model's tokenizer and chat template: encode, a token's bytes, the ids that end a reply, the template on a chat request | `runtime.flerp.text_strixite` (strixite's `Tokenizer` and `render_chat`), `runtime.flerp.text_strixfold` (StrixFold's `HfText`) |
| `sample.host/flerp-v1/-` | `sampler.h` | a sampler for rows the executor hands to the host | `runtime.flerp.sampler_strixite` (strixite's `Sampler`), `runtime.flerp.sampler_none` (for an executor that draws on the device) |

Tokenizer and template belong to the model, not to the engine, and none is written here: each text module is a
shim over an origin's own code. `runtime.flerp.decode_loop` (`decode_loop.h`, required by module id) is the decode
loop every caller of an executor shares - strixite's `Engine::run` written over the ABI, with the origin's
acceptance rule for both draft forms - and the conformance driver and the front end run that one loop.

`serve.flerp.http` is the front end: an OpenAI-compatible server that reaches the model only through these
headers and names no engine. What a request may ask follows from `flerp_caps`: it samples on the host where the
executor offers rows and lets the executor draw where it offers tokens only, and a request the executor cannot
honour (logprobs or a logit bias on a tokens-only executor, a `top_k` its sampler does not take) is refused with
HTTP 400 before anything runs. It serves one conversation at a time, on the thread that opened the executor.

## 7. recipes/<name>.toml

```toml
name      = "qwen38-flash-next.strixite"
model     = "qwen3.8-flash-next"
reference = { engine = "strixite", build = "/data/projects/strixite-v070" }   # what the recipe gate compares to
toolchain = "therock-10.1"

[modules]                 # interface -> module id; `flerp assemble` fails on an unfilled or ambiguous interface
"linear.decode/strixw-q4/bf16" = "kernel.strixite.linear_q4"
...
[include]
modules   = ["executor.strixite.qwen3_5"]     # compiled with the recipe although no binary links them

[binaries]                # binary name -> the module holding its main(); it links that module's requirement closure
"strixflerp-qwen38-flash-next.strixite" = { main = "serve.strixite.strix_server", entry = "tools/strix_server.cpp" }

[deploy]
topology  = "solo"        # solo | tp2 | tp4 ; facts come from the pod's deployir.yaml, never restated here

[gate]
runner    = "gate/strixite_recipe.py"                 # what `flerp gate recipe <name>` runs
config    = "gate/qwen38-flash-next.strixite.toml"    # host facts and pass criteria for that runner
# criterion = "token-identity" (default) | "token-agreement"
```

A recipe binds every interface its modules require under `[modules]`. Modules required by id (`module = ...`)
come along without a binding, and `[include]` adds module ids outright.

A requirement without a tag may be bound to a provider of a tagged name in the same overlap class. That is a
statement that the recipe runs a different numeric contract at that point, so `flerp check` then requires
`[gate] criterion` to say what the recipe gate holds the recipe to: `token-identity` (the swap is exact on
everything the gate exercises: same tokens as the reference, as for the unmodified recipe) or `token-agreement`
(the contract differs by summation order or rounding: the module gate's tolerance is the criterion, and the
recipe gate measures and reports where the tokens first differ, request by request, never as "identical" when
they are not). A requirement that carries a tag is only ever filled by that full name.

A recipe whose origin has its own build graph is built by it: `[build] system` is `"zig"` or `"cmake"` (below).
`[build] system = "zig"`
hands the recipe to `flerplib/assemble_zig.py`: the assembly is a source tree (`build/<recipe>/tree/`, every file
of the module closure at its origin-relative path), `assembly.json` and `build.sh`, not a CMake file.

```toml
[build]
system     = "zig"
zig        = "zig-0.17"                   # a toolchains.toml entry of kind "zig", or an absolute path to the binary
root       = "tool.strixfold.zig_build"   # the module that holds build.zig
steps      = ["native-hip"]               # the build graph's steps
options    = ["-Dhipcc={hipcc}", ...]     # {rocm}, {hipcc}, {prebuilt:<module id>} are filled in
build_file = "flerp.build.zig"            # optional: a build file of an assembled module, in place of build.zig
headers    = ["include/flerp/executor.h"] # optional: repository files copied into the tree
c_link     = "flerp/link.zig"             # optional: a tree path `flerp assemble` writes the C and C++ compile units
                                          # of each binary's module closure to (file, flags, include roots)
[build.prebuilt]                          # module id -> where its [[prebuilt]] kit lives on the build host;
                                          # verified against the module's record before the build
[build.installs]                          # binary name of [binaries] -> the path the build installs it at
[build.hipcc_output_pins]                 # code object name -> the output directory name of the reference build
[build.overlay]                           # derived module id -> its root in the tree (section 5, derived modules)
```

`c_link` is how a C or C++ module and a Zig-exported one become one binary: the origin's build graph compiles the
executor, so it compiles and links the front end too (`flerp.serve.build.zig`), from a list the assembler derives
from the module manifests. The build file names no source of the front end, and a recipe is still assembled with
no hand edits.

`[build] system = "cmake"` (`flerplib/assemble_cmake.py`) is the same idea for an origin built by CMake, where an
executor is taken whole (gufo): the assembly is `build/<recipe>/tree/` (every file of the module closure at its
origin-relative path, the origin's own `CMakeLists.txt` files among them), `assembly.json` and `build.sh`, which
configures and builds `build/<recipe>/out/`.

```toml
[build]
system     = "cmake"
root       = "tool.flerp.gufo_qwen_generate"      # the module that holds the build file
build_file = "flerp/gufo_qwen/CMakeLists.txt"     # its tree path; cmake -S is its directory
targets    = ["flerp-gufo-qwen-generate"]         # ninja targets
options    = ["-DCMAKE_PREFIX_PATH={rocm}", ...]  # configure options; {rocm} is filled in
[build.installs]                                  # binary name of [binaries] -> the path under out/
```

The build file is the top of the graph and is written here when the origin's root builds more than the recipe
(gufo's root builds every model family into one binary and needs their libraries): it declares what the root
declares for the targets the recipe needs and adds the origin's directories unchanged, so targets, per-source
options and link order stay the origin's. A tree that lacks a file those CMake files name does not configure:
that is the check that the closure is whole. The same build file builds a checkout of the origin (the recipe's own
files copied in), which is how the recipe gate makes its reference when no build of the origin exists on the host
(`gate/gufo_recipe.py`).

`hipcc_output_pins` is how such a recipe reproduces a reference build's HIP code objects byte for byte: clang
hashes its command line, the `-o` path included, into each object (`docs/HARVEST.md`, "The kit hashes"), so the
recipe records the one string of the reference build's command that a checkout cannot know.

## 8. `flerp` commands

```
flerp list [--kind K] [--origin O] [--licence EXPR] [--provides GLOB]
flerp check [--origin O] [-w]     validate every manifest and recipe: schema, paths exist, requires resolvable
flerp show <module-id>
flerp interfaces [GLOB]           interfaces by overlap class with providers (tags, adapters and what they launch)
                                  and consumers; flags single-provider and unused
flerp licence-report <recipe>     modules by SPDX id
flerp assemble <recipe> [--licence EXPR] [-o build/<recipe>]     generate the build tree and print how to build
                                  it: CMake + Ninja, or for a [build] system = "zig" or "cmake" recipe a source
                                  tree and build.sh
flerp diff-origin <module-id> | --origin O  <origin-checkout>    prove the vendored sources are verbatim; with
                                  --origin also that every file of the checkout is in one module or listed as
                                  unharvested in ORIGIN.toml
flerp gate interface <name>       every provider of one interface on the GPU against a CPU reference: parity and
                                  time per launch (gate/interface/<name>.toml); writes gate/results/interface/<name>/
flerp gate module <id>            the interface gates the module is a provider in; a module in none of them gets
                                  the compile-only gate (gate/results/modules/)
flerp gate recipe <name>          the recipe gate; writes gate/results/<name>/
```

Gates run on gfx1151. Their pass criteria live in a config file that is written before the first run and is not
edited to make a run pass; an interface gate records the config's sha256 with each result.

## 9. Phases and their gates

A phase is done when its gate passes on gfx1151 hardware and the evidence is committed. Nothing is reported as
working on the strength of a build alone.

1. **Core + strixite recipe.** `flerp` tooling; strixite v0.7.0-pod harvested into modules; the recipe assembles
   and builds on control-node. Gate: strixite's own unit tests pass from the assembled tree, and
   `strixflerp-qwen38-flash-next.strixite` produces the same tokens as the reference strixite binary for the same
   greedy prompts on the same GPU (seed fixed, drafts on and off), at the same tokens per second within noise.
2. **Catalogue harvest.** StrixFold, gufo and hipfire kernels, formats and runtime pieces harvested with manifests;
   each `hip-c++` kernel module with sources or units compiles standalone for gfx1151 with its origin's compile
   recipe (`gate/compile/run.py`); header-only modules and fragments are covered by the unit that includes them;
   prebuilt kits are recorded with hashes. `flerp interfaces` shows where engines overlap.
3. **Second recipe + common front end.** StrixFold DSV4.1 TP2 recipe (Zig executor, kernel kit) with its own token
   identity gate on the worker pair; both executors behind `executor.h`; one `serve` module serves either. The
   front end's gates: without a GPU, against a stand-in executor in every shape the ABI allows (the request
   paths, the refusals); on control-node's GPU, the strixite recipe against the origin's own server, token ids and
   step records over HTTP for forwards of the same shape (`gate/flerp_serve.py`); on the worker pair, the
   StrixFold recipe against the origin's server (`gate/flerp_serve_pair.py`).
4. **Cross-engine swaps.** For each interface with two or more providers, a parity and bandwidth gate
   (`flerp gate interface`), and - where a recipe consumes the interface - a recipe variant
   (`recipes/<model>.<variant>.toml`) that binds the foreign provider through an adapter, is assembled with no
   hand edits, and passes the recipe gate against the same reference binary: token identity for an exact swap,
   the module gate's tolerance plus a measured token agreement for one that changes summation order. The gate
   also requires the adapter's own count of foreign launches to be above zero in every server it loads. This is
   the proof that the catalogue is modular and not a monorepo of forks. A swap that does not hold up is
   written down in `docs/INTERFACES.md` with the exact reason.
5. **New model from parts.** A recipe for a model none of the origins serve on this pod, assembled from catalogue
   modules, with the missing modules written under origin `flerp`. Step 1 is a rehearsal on a model an origin does
   serve: gufo's Qwen 3.5 executor assembled whole (`recipes/qwen35-decider-2b.gufo.toml`, `[build] system =
   "cmake"`) on a small dense qwen35 GGUF. Gate (`gate/gufo_recipe.py`): the vendored files are the origin's, the
   assembled tree and an export of the origin compile the same sources, and on the GPU the two binaries give the
   same greedy token ids; llama.cpp and gufo's CPU reference on the same prompt ids are recorded beside it.
   Step 2 is the first module written for the target's architecture: `executor.flerp.qwen35moe` (a derived
   module, section 5) and `kernel.flerp.moe_kquant`, in the recipe `qwen36-35b-a3b.flerp-moe` on one node, on a
   qwen35moe GGUF that fits one device. No origin runs that file with this executor, so the gate
   (`gate/moe_recipe.py`) holds it to another engine, llama.cpp on the CPU, under criteria written before the
   first run (`gate/qwen36-35b-a3b.flerp-moe.toml [criteria]`): the dense recipe's recorded tokens on the dense
   model; the MoE runs equal to themselves; against llama.cpp on the same prompt ids, log-probabilities at the
   first generated position within a tolerance and the generated ids equal up to a position where llama.cpp's
   own two best tokens are nearly tied. Hidden states and router choices against llama.cpp's eval callback, and
   the same log-probability comparison on the dense model through the origin's unmodified path, are recorded
   beside it. The multi-node split of the 397B file is not part of this step.

## 10. Build and test hosts

- Source of truth: this git repo on the laptop (`~/Code/strixflerp`). Reference checkouts of the origins are in
  `~/Code/strixflerp-upstream/` (read-only, never committed here).
- Build and GPU gates: control-node (`/data/projects/strixflerp`), which has gfx1151, TheRock toolchains under
  `/data/tools` and `/data/models/tools`, the system ROCm 7.2.4 under `/opt`, cmake, ninja, zig 0.17
  (`/home/pod/opt`), python 3.12. `toolchains.toml` names each (kind `therock` or `rocm` for a ROCm install,
  `triton` for the ahead-of-time Triton kit's wheel, `zig` for the Zig compiler of a Zig-hosted recipe). Sync with
  `tools/pod-sync.sh`; run with `tools/pod-run.sh`. The pod has no internet: no pip, no cargo fetch, no git clone.
- Workers serve the pod's roles. Never stop, restart or load an engine on a worker for phase 1 or 2. Phase 3's
  pair gate is the one exception and it is narrow: the pair F and A, only while it holds both with `pod-quad hold`,
  only through the pod's own launcher, and it stops only what it started (`gate/strixfold_recipe.py`).
- control-node's GPU is shared with the pod's own jobs, and theirs come first. A gate loads a model only when the
  GPU is idle and no `pod-gpu --wait` job is waiting, holds the `control-gpu` lease only while a model is loaded,
  and gives it back between loads when a pod job has started to wait (`gate/strixite_recipe.py`, `Hold`).
  `tools/gpu-queue.sh` runs gates one after another under the same rule.
