# Image input for DeepSeek-V4.1 on HIP: Strixfold pull request #6 in the catalogue

**Status, 2026-11-11.** Harvested, and one part has run: the CPU tower (the Python worker, its weights extraction
and its pipe protocol) on control-node's CPU, Python 3.12 and torch 2.11.0+cpu, against DeepSeek's own `vision.py`
(section "What ran"). The worker's rows are bit-identical to the reference's in BF16. Its gate first **failed** one
of six criteria (a bound against FP32); the owner restated that criterion the same day and under it the gate
**passes**, judged from the same run's evidence. Everything else below is still from reading: no Zig file of the PR was compiled, the HIP attention did not
run, no image request went through an engine. "The PR says" marks the PR's own claims; outside "What ran" they are
unverified.

- Source: Shoestring-dev/Strixfold pull request #6, author aercanozcan, branch `feat/hip-cpu-image-input`, head
  `766df75` (two commits, `6335ff9` and `766df75`, 2026-11-11), base `70cfb8c` (merge-base with upstream main, which
  has since moved two documentation commits to `773e68c`). Open, no maintainer review. 37 files, +2,081 / -36.
- The catalogue's StrixFold is the pod's fork at `f64b5ff` (upstream `88ea779` plus the pod's commits). The PR's base
  is a later upstream commit, and the fork is not an ancestor of it: the two share `88ea779`.
- Reference checkout: `~/Code/strixflerp-upstream/strixfold-pr6` (clone of a git bundle from relay-host).
  `./flerp diff-origin --origin strixfold_pr6 <it>`: 23 of 23 files byte-identical.

## What was harvested

Citations below are `<tag> <origin path>:<lines>`; the file is at `catalog/strixfold_pr6/<module dir>/src/<origin path>`.

| tag | module | files | licence |
|---|---|---|---|
| SC | `runtime.strixfold_pr6.vision_sidecar` (`runtime/vision_sidecar`) | `vision_cpu.zig`, `vision_pipe.zig`, `vision_cache.zig`, `vision_cpu.py`, `vision_hip.py`, `tools/dsv41-hip/VISION.md`, the lock file, DeepSeek's licence text | Apache-2.0 AND MIT |
| SK | `kernel.strixfold_pr6.vision_softmax` (`kernel/vision_softmax`) | `vision_softmax.hip`, `build_vision_worker.sh` | Apache-2.0 |
| PV | `tool.strixfold_pr6.prepare_vision` (`tool/prepare_vision`) | `prepare_vision.py`, `verify_vision.py`, `vision-checkpoint.json` | Apache-2.0 |
| IK | `runtime.strixfold_pr6.image_key` (`runtime/image_key`) | `core/lanes/image_key.zig`; overlay: `core/lane_host.zig`, `core/lanes/stream.zig` | Apache-2.0 |
| IR | `executor.strixfold_pr6.dsv41_image_rows` (`executor/dsv41_image_rows`) | overlay: `families/dsv41/{lanes,model,native}.zig`, `gpu/hip_facade.zig`, `native/hip.zig` | Apache-2.0 |
| SP | `serve.strixfold_pr6.image_prompt` (`serve/image_prompt`) | overlay: `server/prompt.zig`, `server/tokens.zig` | Apache-2.0 |

"SF" cites a file the PR does not change, in `catalog/strixfold` (at f64b5ff).

Not harvested (checkout only): the PR's tests (five Python files, edits to `hip_test.zig`, `fake_serve.zig`), its
launcher edits (`serve.conf`, `serve_lib.sh`, `deploy_rank0.sh`; not in `catalog/strixfold` either) and documents.

**Overlays.** Nine of the PR's files are files of four `strixfold` modules. Those modules are unchanged (the dsv41
recipe's gate against deployment f64b5ff2e28b reads only `catalog/strixfold`). Each overlay module holds the PR's
version verbatim plus `overlay.json` and `overlay/<path>.patch` (`docs/SPEC-PROPOSALS.md` section 12):

| file | strixfold module | catalogue file = PR base | PR's patch on the catalogue file |
|---|---|---|---|
| `zig/src/core/lane_host.zig` | `runtime.strixfold.lanes` | no | applies, no fuzz |
| `zig/src/core/lanes/stream.zig` | `runtime.strixfold.lanes` | yes | applies |
| `zig/src/families/dsv41/lanes.zig` | `executor.strixfold.dsv41` | no | applies with fuzz 2 only |
| `zig/src/families/dsv41/model.zig` | `executor.strixfold.dsv41` | no | applies, no fuzz |
| `zig/src/families/dsv41/native.zig` | `executor.strixfold.dsv41` | no | applies, no fuzz |
| `zig/src/gpu/hip_facade.zig` | `runtime.strixfold.gpu_facade` | yes | applies |
| `zig/src/native/hip.zig` | `runtime.strixfold.engine_host` | no | applies with fuzz 2 only |
| `zig/src/server/prompt.zig` | `serve.strixfold.http` | no | applies, no fuzz |
| `zig/src/server/tokens.zig` | `serve.strixfold.http` | yes | applies |

"Applies" is `patch` succeeding on text. Whether the merged `lanes.zig` and `lane_host.zig` are right is not known:
the pod's commits changed kept states there (`--learn`, `shared_messages`, the `/cache` routes) and the PR changes
when a state is kept and restored. Nobody has read the merge.

**Licences.** The repository is Apache-2.0 (`LICENSE`); no PR file carries an SPDX line. `vision_cpu.py` "follows"
DeepSeek V4.1 Flash 1017 `inference/vision.py` at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, git blob
`77af0bde...`, MIT, Copyright (c) 2023 DeepSeek (SC `tools/dsv41-hip/VISION.md:118-121`, `NOTICE`, licence text in
SC `tools/dsv41-hip/assets/DEEPSEEK_VISION_LICENSE`). It is a 160-line rewrite, not that file: DeepSeek's `vision.py`
itself is not in the PR, the checkout or the catalogue. torch, numpy and Pillow are run-time dependencies, not vendored.

## How it works, end to end

1. **Switches.** `gpu.features.vision` becomes true for every HIP build (IR `zig/src/gpu/hip_facade.zig:10`); the
   default stays text (IR `zig/src/families/dsv41/model.zig:41`). With `TF_DS_MODALITIES=text_image` and a
   checkpoint whose config declares a tower, `native.open` opens `vision_cpu.zig`'s `Vision` (IR `native.zig:42,199`).
   `open` reads the environment and starts nothing: it needs `TF_DS_VISION_PYTHON`, `_WEIGHTS` (absolute paths) and
   `_SHA256`, and the routing bias already loaded (SC `vision_cpu.zig:28-52`; the bias: IR `model.zig:476,717-731`).
2. **Request to spans.** An `image_url` part (PNG or JPEG data URL) becomes the placeholder
   `<｜deepseek_image｜>` in the message text. Before the template runs, each placeholder is rewritten to
   `placeholder + index + placeholder` (SP `zig/src/server/prompt.zig:78-101`); after rendering, the ids between two
   image tokens are decoded back to the index, so a template that reorders tool results cannot swap images
   (SP `prompt.zig:105-137`). For each image `Vision.prepare` runs on the request's thread: PNG by the native decoder,
   other formats by the existing Pillow child (`python3` on `PATH`), then `picture.fromRgb` (resize, pad, patches)
   (SC `vision_cpu.zig:72-117`). The result is a payload (32-byte grid + BF16 patches) and the span length; the
   prompt gets that many image tokens (SP `prompt.zig:133-134`). `/tokenize` counts the same (SP `tokens.zig:32-43`).
3. **Spans to rows.** In the prompt pass the family checks each span, skips images that lie wholly inside a restored
   prefix, and calls `rowsBatch` for the rest (IR `lanes.zig:674-697`). `rowsBatch` (SC `vision_cpu.zig:169-199`),
   per image: validate the payload (`:119-130`), look up `sha256(payload)` in the row cache
   (SC `vision_cache.zig:21-29`), else spawn the worker, write the payload, read the rows (`:132-166`), store them.
4. **The worker** (SC `vision_cpu.py`). Sets its alarm, imports torch, refuses any version but `2.9.0+cpu`
   (`:155-159`); reads the grid and patches (`:131-138`); hashes and loads the 266 tensors (`:11-37`); runs
   `encode` (`:50-93`): patch embed (588 to 1024), 32 blocks of RMS norm, QKV, 2D rotary, 16-head attention,
   output projection, RMS norm, SwiGLU MLP; final norm; 3 x 3 unfold; aligner (9216 to 5120, GELU, 5120 to 5120).
   `span` (`:96-101`) adds `image_start`, one `image_newline` a line, `image_end`. Rows go out as BF16 behind a
   16-byte header (`:147-151`). Attention is torch's MATH backend over query pieces of 128 rows (`:74-79`).
5. **Rows into the model.** `Model.setImages` copies the rows to the device and records the spans
   (IR `model.zig:733-749`); followers get them in wire frames (IR `lanes.zig:715-733`). The prompt pass reads the
   rows in place of embeddings and routes span positions with `gate_bias_vl`
   (SF `zig/src/families/dsv41/prompt.zig:1216-1230`). From there the request is an ordinary one: drafts, lanes, sampling.
6. **Caches.** Both off by default (`TF_DS_VISION_CACHE_MIB=0`). Above zero: an LRU of rows by payload hash
   (SC `vision_cache.zig`), and image-aware prompt-cache keys (IK `zig/src/core/lanes/image_key.zig`,
   IK `lane_host.zig:78-99,333-338`). Contract: `docs/INTERFACES.md` section 13.
7. **Optional HIP attention.** With `TF_DS_VISION_HIP_CODE` and `_HIP_SHA256`, `vision_hip.py` is prepended to the
   worker (SC `vision_cpu.zig:134-136`): per tile of 128 query rows, `hipblasSgemmStridedBatched` for the scores,
   the kernel `normalize_scores` (SK `vision_softmax.hip`), a second GEMM for the output, all fp32, by ctypes on
   `libamdhip64.so` and `libhipblas.so` (SC `vision_hip.py:47-98`). Everything else stays on the CPU.

## Per-request costs (read from the code; nothing was timed)

- One process **per uncached image**, not per request and not kept alive: interpreter start, `import torch`, a
  sha256 over the 970,631,680-byte weights file and a second read of it into tensors (SC `vision_cpu.py:14-34`),
  then the forward. With the cache off, the same image is encoded again on every turn that is not served from a
  restored prefix.
- Attention is quadratic in patches: an image at the 9,216-patch bound is 72 query pieces x 9,216 keys x 16 heads
  x 32 blocks on CPU threads (`TF_DS_VISION_THREADS`, else torch's choice). The timeout default is 240 s.
- It runs on the lane thread (SC `vision_cache.zig:1`; the call is inside the prompt pass, IR `lanes.zig:696`):
  as read, every other stream of the server waits while an image is encoded. Not measured.
- Memory on rank 0 beside the loaded model: about 1 GB of weights plus activations in the child; the HIP form
  adds device buffers of `16 * n * (8 * 64 + 128) * 4` bytes (SC `vision_hip.py:28-32`), on the GPU that holds
  the language model.

## What is exact, what is not

| claim | status |
|---|---|
| CPU rows equal DeepSeek's reference `vision.py` | **measured, on torch 2.11.0+cpu / Python 3.12 and 3.13 / control-node's CPU:** bit-identical to the reference run in BF16 on 11 of 11 images, at 1 and 8 threads ("What ran"). Not measured on the PR's own pin (torch 2.9.0+cpu) or another CPU. The PR itself has no receipt: its tests are synthetic |
| picture preparation equals the reference's | SF `picture.zig:3-6` says every step is in the reference's order; from the existing CUDA path, not from this PR |
| HIP attention | not qualified, by the PR's own words (SC `VISION.md:84-86`: it "changes the reduction order and requires independent numerical qualification") |
| text requests unchanged | the PR's claim; plausible from the diff (the image branches are behind `images.len > 0`), but `gpu.features.vision = true` changes what every HIP build compiles |
| image-keyed prompt cache returns the same tokens as an uncached run | not claimed with evidence; not measured |

## What it needs at run time

**Weights** (PV `tools/dsv41-hip/vision-checkpoint.json`): repository `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw`,
revision `c178a01284fc5c0e859d8668cffaac81ca8f4042`; the index and 42 shards pinned by size and sha256.
`prepare_vision.py` copies 266 tensors (`vision.*`, `aligner.*`, `image_*`, all BF16) into `vision.safetensors`
(970,631,680 bytes, sha256 `fc4d25f5...`; as run, the file holds 309 tensors: the 43 biases are in it too, and the
worker's loader skips them) and 43 `layers.N.ffn.gate.bias_vl` (F32) into `gate_bias_vl.safetensors`
(66,736 bytes, `c02efb1a...`), and publishes neither unless it matches the pin (PV `prepare_vision.py:66-122`).
Rank 0 needs both; a follower needs the bias in its kit's `vision/` and `text_image` too.

**Python.** What pins the interpreter and torch, and what does not:

| where | what |
|---|---|
| SC `vision_cpu.py:158-159` | `torch.__version__ != '2.9.0+cpu'` raises. The one hard pin in the worker |
| PV `verify_vision.py:31-34` | the launcher's admission asserts torch `2.9.0+cpu`, numpy `2.3.5`, PIL `12.3.0` |
| SC `requirements-vision.lock.txt:1-3` | "Linux x86_64 Python 3.13" wheels by hash: Pillow here; torch (`6c9b2175...`) and numpy (`11e06aa0...`) through `requirements-inputs.lock.txt`, which is in the checkout and not in the catalogue |
| SC `vision_cpu.py:15`, PV `prepare_vision.py:28,93`, `verify_vision.py:19` | `hashlib.file_digest`: Python 3.11 or newer. Nothing else in the Python files is tied to an interpreter version; nothing needs 3.13 |
| SC `vision_cpu.py:34,74-79,144,149` | torch API used: `frombuffer` of bfloat16, `torch.nn.attention.sdpa_kernel(SDPBackend.MATH)`, `scaled_dot_product_attention`, `F.linear` / `unfold` / `silu` / `gelu` on bf16 CPU tensors, `view(torch.uint16)`. As far as reading goes none is new in 2.9; not tried on another version |
| SC `vision_cpu.py:149` | numpy only for `.numpy().tobytes()`; Pillow is not used by the worker at all (only by the existing decoder child, SF `pil.zig`) |

So the 2.9.0 pin is not an API need. What it protects is the bytes: bf16 `F.linear` and the MATH attention go
through torch's CPU math libraries, whose kernels and summation order can change between torch builds, between
CPUs and possibly with the thread count. "Reference-exact" is therefore a property of (torch build, CPU, threads),
to be proved by the gate below on whichever torch is used; it does not carry over from 2.9.0.

**On the pod** (control-node, read-only, 2026-11-11):

| item | found |
|---|---|
| checkpoint | `/data/ark-inbox/models/LibertAIDAI__DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw/raw/`, 53 files. The index's sha256 equals the pin; shard 43 equals the pin in size and sha256 (`c5b94d51...`). The index puts all 266 vision tensors in shard 43 (BF16, 970,536,960 bytes of data) and the 43 biases across 42 shards. The other 41 shards were not hashed |
| extracted `vision.safetensors`, `gate_bias_vl.safetensors` | made 2026-11-11 by the PR's `prepare_vision.py`, unmodified: `/data/projects/strixflerp/build/vision/`, both equal to the pins (it hashed all 42 shards, 197 GB, in about 4 minutes). Mode 0600, owner pod. Not deployed anywhere |
| the reference, DeepSeek `inference/vision.py` | `/data/ark-inbox/models/deepseek-ai__DeepSeek-V4.1-Flash-1017/raw/inference/` (`vision.py`, `image_processor.py`, `config.json`, ...); `vision.py` is git blob `77af0bde...`, the one the PR names. Of that repository's weights only shards 47 and 48 are on the pod; its vision tensors are in shards 1 and 2, so LibertAI's copies of them could not be compared with DeepSeek's |
| interpreters | system Python 3.12.3. Also `/home/pod/opt/python313` (CPython 3.13.15, with pip; no torch, numpy or Pillow in its site-packages). No 3.14 seen |
| wheels, `/data/.ark-views/stable/wheels` (4,773 files) | numpy 2.3.5 and Pillow 12.3.0 for cp312, cp313 and cp314; the cp313 files have the lock's hashes. torch `2.9.0` (PyPI build, not `+cpu`) and `2.9.0+xpu` for cp312 only. torch `+cpu`: 2.11.0 for cp312 and cp313, 2.14.0 for cp313 and cp314. **No `torch-2.9.0+cpu` wheel for any interpreter** |
| workers | not looked at (their Python is said to be 3.12; whether they see `/home/pod/opt/python313` is unknown) |

Routes, none taken (each needs a venv, which is an install):

- **The PR as written:** Python 3.13 + `torch-2.9.0+cpu-cp313` (sha256 `6c9b2175...8682`). Only that one wheel is
  missing; the interpreter, numpy and Pillow are there.
- **(a) Python 3.12 + torch 2.11.0+cpu, from the ark as it is:** numpy 2.3.5 and Pillow 12.3.0 cp312 are present.
  The worker refuses at SC `vision_cpu.py:158` and the admission at PV `verify_vision.py:32`: two string
  comparisons, in harvested files that are not to be changed. **Taken 2026-11-11** (the owner's decision), with
  the derived module below: "What ran".
- **(b) Python 3.14 + torch 2.14.0+cpu:** the three cp314 wheels are on the ark; a CPython 3.14 standalone build
  has to be arked in. Same two comparisons. Nothing in the Python sources is known to break on 3.14; not tried.
- **The `flerp` module for (a) and (b), now in the catalogue:** `runtime.flerp.vision_sidecar`, a derived module
  (SPEC.md section 5) with adapted copies of `vision_cpu.py` and `verify_vision.py`. As built it differs from the
  proposal: `vision_cpu.zig` is not adapted (it embeds the origin's script; a Zig build that embeds the copy is a
  recipe's work); the worker checks a setting, `FLERP_VISION_ENV=python=..,torch=..,numpy=..`, against what it
  runs; the records with their status and evidence are `vision-env.json`, read by the admission, which admits
  `pass` and `origin-pin` and a `candidate` only on request. CPU model and threads are in a record's scope text,
  not checked. numpy was kept.

**HIP form, additionally:** a code object built by SK `build_vision_worker.sh` with `amdclang++`, its sha256, and
`libamdhip64.so` / `libhipblas.so` loadable by the child.

## Behind strixflerp's executor ABI and `serve.flerp.http`

Today: `FLERP_CAP_VISION` exists and no call carries an image (`include/flerp/executor.h:73`,
`docs/EXECUTOR-ABI.md`, "Vision"); `executor.flerp.strixfold_abi` would set the flag once `Loaded.vision` is not
null (`strixfold_abi.zig:262`); `serve.flerp.http` refuses any content part that is not text (`request.cpp:85-96`).

Two places the tower can sit, both using seams the PR already has:

1. **Inside the executor, as the PR has it.** A recipe with the overlay modules gives the StrixFold executor its
   own tower. The ABI then has to carry what the lane backend carries: a version that appends
   `flerp_image_prepare(ex, bytes, n, &tokens, &payload)` (the `Vision.prepare` hook, SF `core/lanes/stream.zig:112-121`)
   and a list of `{at, tokens, payload}` spans on the first forward of a prompt, since StrixFold wants every span
   of the prompt before its first chunk (IR `model.zig:733-735`). Caps gain the placeholder text, the image token
   id and the bounds.
2. **In front of the executor, rows across the ABI.** `Image.rows = true` already means "these bytes are finished
   rows" and the family copies them without a tower (IR `lanes.zig:698-700`, SC `vision_cpu.zig:175-179`). An ABI
   span `{at, tokens, rows}` in BF16 `[tokens][hidden]` is then the whole contract
   (`vision.image-rows/deepseek-v4.1/bf16`), and the tower is a module of its own, bound in the recipe: the sidecar
   now, native kernels later, with the executor unchanged. This is the modular form. It needs one thing the PR does
   not have: `native.open` opens the tower whenever `text_image` is set (IR `native.zig:199`), so a rows-only mode
   needs a switch there, and the routing bias loaded without a tower.

Either way the front end needs its own version of the prompt work (image parts, identity through the template,
span placement: SP `prompt.zig:70-137`), written over `text.h`, and the picture preparation (SF `picture.zig`,
`pil.zig`) reachable from it. Prompt-cache keys do not cross the ABI today at all (the front end refuses
`cache_prompt`).

Main obstacles, in order: (1) no assembly rule for overlays, and two merges nobody has read; (2) no
`torch-2.9.0+cpu` wheel, or else a version-neutral worker and a gate for it; (3) the extraction has not been run
(it hashes 42 shards) and its outputs have to reach both ranks; (4) ABI and front end have no image call;
(5) the encode blocks the lane thread and shares rank 0's memory with the model; (6) every claim of exactness is
still the PR's.

Since 2026-11-11: (3) the extraction ran on control-node and matched its pins (it is in a build directory, not
deployed to either rank); (2) a version-neutral worker exists and its gate ran ("What ran"); of (6), the CPU rows
are measured, on torch 2.11.0+cpu.

## The gate that would prove it

Stages 1 and 2b ran in the form described under "What ran" (stage 2b against the reference in BF16 came out as byte
equality on every image; the gate that ran also holds the rows to an FP32 reference, per `docs/FIDELITY.md`, and
that is where it failed). Stages 2a, 3 and 4 have not run.

Criteria are written to a config before the first run (SPEC.md section 8). Stages, cheapest first:

1. **inputs** (CPU, minutes): `prepare_vision.py` on the ark's checkpoint; pass = both outputs equal the pins.
2. **rows** (CPU, no model loaded): a fixed image set - the smallest grid, grids not divisible by 3 (the padding
   path), one at the 9,216-patch bound, one that hits the 1,024-row cap, a JPEG (Pillow path) and a PNG (native
   path), an image below `min_pixels`. For each: (a) patches from `picture.zig` against the reference's own
   preprocessing, byte equality; (b) rows from the worker, run as the server runs it (`python -I -c`, same
   weights), against rows from DeepSeek's `inference/vision.py` at `1bae0e10` (to be arked: it is not on the
   laptop or in the PR) on the same torch, same threads: **byte equality of every row**, by sha256. On a mismatch
   `encode`'s `probe` hook (SC `vision_cpu.py:50,62,84,91`) gives the first block that differs. Repeated per
   candidate environment (2.9.0+cpu, 2.11.0+cpu, 2.14.0+cpu; 1 thread and the default), with the cross-environment
   comparison recorded and not required.
3. **rows, HIP attention** (control-node GPU): against stage 2's rows, a bound in BF16 units written beforehand; this
   is a tolerance, never "identical".
4. **tokens** (the worker pair, in a `pod-quad hold`): the overlay build against the deployed reference on the
   existing text prompts, token identity (the PR must change nothing for text); then image prompts: same tokens
   with drafts on and off, with the row cache and the image prefix cache on and off, on a second turn that restores
   a prefix holding an image, and with rows handed in as `Image.rows` from the reference tower. There is no other
   HIP engine serving this model with images, so stage 4 proves invariance and plumbing; stage 2 carries the
   correctness of the rows.

## What ran (2026-11-11): the CPU tower on Python 3.12

Gate `gate/vision_dsv41_cpu.py`, criteria `gate/vision-dsv41-cpu312.toml` committed (2c20dd2) before any tower
forward; results `gate/results/vision-dsv41-cpu312/` (`summary.md`). control-node, CPU only, 8 threads, niced, while
other jobs ran (load average 6 to 36). Environment: a venv under `build/vision-venv312`, Python 3.12.3, torch
2.11.0+cpu, numpy 2.3.5, Pillow 12.3.0, all from `/data/ark/wheels`.

**The reference** is DeepSeek's own `inference/vision.py` (blob `77af0bde`), imported unmodified from the ark, its
`ViT` and `Aligner` holding the checkpoint's 266 tensors read from shard 43 by a reader written in the gate (not the
PR's extraction, not its loader). Patches and grids come from DeepSeek's `image_processor.py` for every arm. So the
reference shares no code with the PR. What it does share with the worker: torch itself (the same build runs both),
and the weights file's bytes.

**Verdict: PASSED, under criterion 3a as restated; FAILED as first committed.** The table is the first verdict
(2c20dd2's criteria, the run of 1cdfa3e), on one of six. Restated 3a (c7a727e, the owner's decision of 2026-11-11):
the worker's BF16 rows equal the reference's BF16 rows bitwise on every image. It held, 11 of 11, read from the bit
comparisons that run had stored; no forward was run again. `summary.md` carries both verdicts.

| criterion | result |
|---|---|
| 1 shape: tokens = `lh * (lw + 1) + 2`, the hand-computed counts, delimiter rows = the checkpoint's | held, 11 of 11 |
| 2 arithmetic before BF16: the sidecar's `encode()` on FP32 tensors against the reference in FP32 | held: identical values on all 11 (difference exactly 0) |
| 3a an absolute bound on the BF16 rows' distance from the FP32 reference (image: relative RMS <= 0.085; any row: cosine >= 0.9855) | **failed**: the image bound held everywhere (0.002 to 0.039); the row bound broke on 9 rows of 4 images (`tiny` 0.979, `text` 0.867, `cap` 0.899, `over` 0.935) |
| 3b no further from FP32 than twice the reference's own BF16 evaluation | held: ratio 1.000 on every image |
| 4 determinism: the worker twice, and against the in-process arm | held: same bytes |
| 5 the pipe protocol and nine refusals; the admission | held |

What the failure is. On all 11 images the worker's rows are **bit for bit** the rows DeepSeek's code gives when run
in BF16 (the dtype its `generate.py` sets and the tensors ship in), with and without the PR's 128-row query
pieces, at 1 and 8 threads, and on CPython 3.13 with the same torch. The rows that break 3a are therefore the
reference's own BF16 rows. The bound was derived as a random walk of BF16 roundings with a factor of 4 for
amplification, and a row was allowed twice the image's error; the tower does not behave that way. A typical row is
at cosine 0.9999 from FP32 (the median, every image), and one to four rows an image sit 10 to 40 times further
than the median row, not because they are short (`posthoc-bf16-rows.json`). The owner restated 3a as bit
equality with the reference in BF16; the record `py312-torch2110` is now `pass` and the admission takes it
without a flag. CPython 3.13 has no record: its rows were compared, its refusals and admission were not run.

**Fidelity note: the tower itself in BF16 against FP32** (DeepSeek's code both sides; aligner rows, cosine a row):

| image | rows | median | lowest | rows under 0.9855 | worst row's error / median row's |
|---|---|---|---|---|---|
| tiny (39 x 39 patches) | 169 | 0.999914 | 0.97946 | 2 | 9.1 |
| photo | 192 | 0.999917 | 0.98874 | 0 | 12.0 |
| photo-jpeg | 192 | 0.999924 | 0.99636 | 0 | 4.8 |
| text | 300 | 0.999934 | 0.86716 | 1 | 42.3 |
| rgba | 204 | 0.999926 | 0.99221 | 0 | 8.8 |
| palette | 184 | 0.999818 | 0.99640 | 0 | 5.8 |
| wide | 174 | 0.999885 | 0.99468 | 0 | 7.1 |
| cap (9,072 patches) | 1008 | 0.999914 | 0.89888 | 2 | 20.7 |
| over (8,649 patches) | 961 | 0.999901 | 0.93489 | 4 | 17.4 |

9 rows of 4 images. This is the model as DeepSeek runs it (its `generate.py` sets BF16), so any engine that is
faithful to the reference reproduces these rows; a native tower with FP32 accumulation would land elsewhere on
exactly them. What the outlier rows do to generated text was not measured (`docs/FIDELITY.md`, last section).

Recorded beside the verdict:
- **Thread count and interpreter do not change a bit** here: 1 against 8 threads, and 3.12 against 3.13
  (torch 2.11.0+cpu cp313, `build/vision-venv313`), 11 of 11 images. The PR's pin was not needed for byte
  stability across these; across torch builds or CPUs nothing was measured.
- **BF16 against FP32, the tower itself:** on the nine pictures only 8 to 12% of an image's values equal the FP32 result rounded to
  BF16. That is DeepSeek's evaluation as shipped, not the worker's doing.
- **Time, one process an image** (interpreter, `import torch` 0.8 s, sha256 of the weights 0.4 s with the file
  cached, load, forward): 1.5 s for the smallest grid, 3.4 to 10 s for images of 179 to 317 tokens (1,500 to 2,500
  patches), **106 to 131 s** for the two at the cap (8,649 and 9,072 patches). The forward alone at the cap: 101 to
  127 s on 8 threads, 308 to 357 s on 1. All inside the worker's 240 s default, the largest with a factor of
  under 2 to spare, on a loaded machine. The protocol has no warm form.
- **Memory:** peak RSS of the worker 1.17 GiB (smallest) to 1.79 GiB (cap).
- The origin's unmodified worker in this environment refuses, as written: `requires the pinned CPU-only torch 2.9.0+cpu`.

Not covered by this run: `picture.zig` (the worker's real input; here patches came from the reference's Pillow
preprocessing), `vision_cpu.zig` / `vision_pipe.zig` / `vision_cache.zig` (a Python stand-in drove the pipe), the
HIP attention, any image through an engine, torch 2.9.0+cpu, Python 3.14, another CPU, the workers.

## Long term: the tower as native HIP kernels

The tower is small and regular: 32 blocks at width 1024 (16 heads of 64, SwiGLU MLP, RMS norm, 2D rotary), an
aligner of two linears, BF16 weights of about 0.97 GB, at most 9,216 tokens with no mask. What the catalogue has:

| candidate | what it gives | caveat |
|---|---|---|
| `executor.strixfold.dsv41_vision` (`vit.zig`, `fmha.zig`, `vision.zig`; Apache-2.0) | this same tower already in Zig behind the same hook, for CUDA | its attention is a CUDA code object and its GEMMs cuBLAS; a HIP port needs both replaced. `hipblas.zig` (fp32 GEMMs by dlopen) is already in `executor.strixfold.dsv41` |
| `kernel.strixfold_pr6.vision_softmax` + hipBLAS | the PR's own tiled attention: two batched SGEMMs and a row softmax | fp32, unqualified; three launches a tile from Python today |
| `kernel.strixite.vision` (`vision.tower-ops/bf16/bf16`, builds) with `kernel.strixite.norm`, `kernel.strixite.linear_wmma` (`linear.prefill/bf16/bf16`) | linear with bias, GELU, attention, 2D RoPE, patch merge, RMS norm and BF16 GEMM on the matrix units, running in strixite's Qwen tower | AGPL-3.0-only, so the assembly is; `kMaxVisionSeq` is 8,192 (`kernels/vision.hpp:50`), below 9,216; LayerNorm and Qwen's merge, not this aligner |
| `executor.gufo.qwen_vision` (`encoder.hip`; MIT, harvested) | a ViT encoder on HIP with its device input and image cache | fp32 on a GGUF mmproj; another model's layout |
| `kernel.gufo.qwen_linear_prefill` (`linear.prefill/bf16/bf16-f32.hipblaslt`, MIT, builds) | BF16-weight GEMM by hipBLASLt | fp32 output |
| hipfire `vit_attention*.hip`, `apply_rope_2d_vision.hip` (Apache-2.0 OR MIT) | ViT attention variants for RDNA | left out of the harvest (`docs/HARVEST.md`); in the reference checkout only |

A native tower will not reproduce torch's CPU bytes. Its criterion is the tolerance form (rows within a bound of
stage 2's, then measured token agreement), as for the order-changing swaps of phase 4. With the rows-across-the-ABI
form above, sidecar and native tower are two providers of one overlap class and the recipe picks.

## Not verified

- Outside "What ran", anything about behaviour: no token, and no row, timing or memory figure of the Zig side,
  the HIP attention or an engine. Also not looked at: the workers' Python.
- That the PR builds (Zig 0.17) or that its tests pass; that the overlays build on the catalogue's files.
- The PR's pin itself: torch 2.9.0+cpu was never run (no wheel), so "the PR as written equals the reference" is
  still the PR's claim; what is measured is the same source on torch 2.11.0+cpu.
- That LibertAI's vision tensors are DeepSeek's: the deepseek-ai shards that hold them are not on the pod.
- `picture.zig` against the reference's preprocessing (stage 2a of the gate above); the gate that ran used the
  reference's preprocessing on both sides.
