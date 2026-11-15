# Sources for SPEC section 1

Where each of the five facts the design rests on can be checked. Paths are relative to the origin checkouts under
`~/Code/strixflerp-upstream/` (`strixite-v070pod`, `strixite-tp-rdma-v2`, `strixfold`, `gufo-neuhaus`, `hipfire`,
`meta`), with line numbers at the commits in `catalog/<origin>/ORIGIN.toml`. Files marked (catalogue) are also under
`catalog/<origin>/.../src/`. "Read" means read in the source; "measured" means a number someone measured on the pod
and wrote down. Nothing here was re-measured in phase 2.

## 1. The cheap seams are HIP kernel source, compiled code objects and the C ABI; host languages differ

| claim | where | kind |
|---|---|---|
| strixite is C++ with `.hip` sources compiled as device code | `strixite-v070pod/CMakeLists.txt:120,181-183`; `strixite-v070pod/kernels/*.hip` | read |
| gufo is C++20 with CMake's HIP language, gfx1151 only | `gufo-neuhaus/CMakeLists.txt:24,100-105,115`; `src/families/qwen/kernels/CMakeLists.txt:5` (catalogue: build options copied into manifests) | read |
| StrixFold's host is Zig; kernels are a kit of raw gfx1151 code objects loaded from a manifest | `strixfold/build.zig`; `tools/dsv41-hip/ADAPTER.md:29` (decision D6); `zig/src/gpu/hip/kit.zig:1-6` (catalogue) | read |
| one C header is the kernel ABI for both hipcc and Zig | `strixfold/zig/kernels/hip/dsv41/abi.h:1-5` (catalogue); `tools/dsv41-hip/ADAPTER.md:31` (D8) | read |
| Triton kernels are compiled ahead of time into code objects and launched from Zig | `strixfold/tools/dsv41-hip/tri/aot_build.py`, `tri_kernels.py:1-15` (catalogue); `zig/src/families/dsv41/tri.zig:1` | read |
| hipfire's host is Rust; every kernel is HIP source text in a Rust constant, compiled by `hipcc --genco`, or an embedded prebuilt bundle | `hipfire/crates/rdna-compute/src/kernels.rs:9-21`; `crates/rdna-compute/src/compiler.rs:430-438` (catalogue) | read |
| Crucible is one C++20/HIP translation unit (642 kernels) built by a Makefile; the server is Python over a JSON-line pipe | `crucible-strix-halo/docs/ARCHITECTURE.md:39-40`, `Makefile:49` (in `~/Code/strixflerp-upstream-supplement/`; catalogue, `docs/CRUCIBLE.md`) | read |
| gufo's quantized kernels cross a raw-pointer C ABI | `gufo-neuhaus/src/models/deepseek_v4_flash/kernels/rocm/mmq/VENDOR.md` ("`ds4_mmq.h` exposes the raw-pointer C ABI") (catalogue) | read |
| SGLang is Python on PyTorch | `catalog/sglang/ORIGIN.toml`; pod `/data/projects/sglang-0.5.20/python` | read (listing only) |
| the seams work in practice: all three origins' kernels compile standalone for gfx1151 outside their engines | `gate/results/compile/*.json` (46 modules, 1,037 units) | measured in phase 2 (compile only) |

## 2. Each compiled weight-reading kernel reads one format (shapes transfer)

| format | where the inner loop is tied to it | kind |
|---|---|---|
| strixw Q4 / Q5 / Q6 / Q8 | `strixite-v070pod/kernels/linear_q4.hpp:1-12` (32 codes per 16-byte load, always within one group), `linear_q5.hpp`, `linear_q6.hpp`, `linear_q8.hpp`; `kernels/qsplit.hpp` | read |
| EXL3 | `strixfold/zig/kernels/hip/dsv41/exl3.hip:8-12,17-20` (16 x 16 tile decoded once per wave; tile bit layout) (catalogue) | read |
| GGUF K-quants, Q8_0, IQ2 | `gufo-neuhaus/src/families/qwen/kernels/ops/gemm.hpp:99-123,148-156,178-181`; `src/core/quant/ggml_dequant.hpp`; `src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_iq2_gate.hip.hpp` (catalogue) | read |
| hipfire HFQ, MQ (V2), HFP4 / MFP4, Q8_0 | `hipfire/crates/rdna-compute/src/kernels.rs:2086-2089,2515-2554,7808-7816`; `docs/quant-formats/mq-v2-family.md`, `hfp4.md` (catalogue) | read |
| FP8 (E4M3) KV | `gufo-neuhaus/src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_fp8_kv.hip.hpp`; `hipfire/docs/quant-formats/fp8-e4m3.md` (hipfire's FP8 kernels are gfx1201-only: `docs/harvest/data/hipfire/not_gfx1151.json`) | read |
| FP4 KV | `strixfold/zig/kernels/hip/dsv41/aux_attn.hip:8-10`, `aux_index.hip:8-9` (catalogue) | read |
| "a GEMV per format": the catalogue has one decode GEMV module per format and none that reads two | `docs/INTERFACES.md` section 1; `flerp interfaces` under `linear.decode/` | read |

## 3. Activation dtype and rounding points differ per engine and carry guarantees

| claim | where | kind |
|---|---|---|
| strixite: bf16 or f32 activations, fp32 math, casts at the reference's points | `strixite-v070pod/kernels/norm.hpp:3-6`, `kernels/attention.hpp` (Precision paragraph), `kernels/linear_q4.hpp` | read |
| strixite TP: every rank sums the partials in fp32 in rank order and rounds once, so all ranks hold bit-identical activations | `strixite-tp-rdma-v2/docs/tp-rdma.md:25-27,47,133` | read (the document says it was checked hash for hash) |
| StrixFold: decode rows are row-invariant (a row's bits do not depend on what shares the window) | `strixfold/zig/kernels/hip/dsv41/dense.hip:5-8`, `exl3.hip:8-15` (catalogue) | read |
| StrixFold: drafts are output-invariant because the sampler is keyed by (seed, position, token id) | `strixfold/zig/src/families/dsv41/sampling.zig:1-9`, `zig/src/core/lanes/sampling.zig:1` (catalogue) | read |
| StrixFold: non-bit-exact kernels are judged under a written tolerance contract | `strixfold/tools/dsv41-hip/numerics/numerics.json` (version 1.7.0) (catalogue) | read |
| gufo: fp32 activations; batch forms bit-exact with the one-row form | `gufo-neuhaus/src/models/qwen38_flash_next/kernels/rocm/kernels.hpp:14-17`; `src/families/qwen/kernels/ops/ssm.hpp:26-29,76-79` (catalogue) | read |
| gufo: speculation is lossless in distribution, not in draws | `gufo-neuhaus/src/families/qwen/kernels/ops/token.hpp:88-93`; `src/models/deepseek_v4_flash/dspark_sampler.hpp:17-24` (catalogue) | read |
| hipfire: fp32 activations | `hipfire/crates/rdna-compute/src/kernels.rs:33-36` ("x is F32, y is F32") (catalogue) | read |
| mixing kernels silently breaks these: no source states it; it follows from the rows above (different summation orders and rounding points) | `docs/INTERFACES.md`, verdict columns | inference |

## 4. Executors are not unifiable today; they are swappable behind one ABI

| claim | where | kind |
|---|---|---|
| strixite's model sits behind `LmBackend` | `strixite-v070pod/serve/engine.hpp:14` | read |
| StrixFold's round loop drives a backend vtable | `strixfold/zig/src/core/lanes/backend.zig:1,45-49` (catalogue) | read |
| gufo has a provider API and a text generation backend | `gufo-neuhaus/include/gufo/ext/adapter.hpp`; `src/cli/serve/text_generation_backend.hpp:142` (catalogue) | read |
| hipfire has an `Architecture` trait and a `Carrier` registry | `hipfire/crates/hipfire-runtime/src/arch.rs:98`; `crates/hipfire-loader/src/lib.rs:72` (catalogue) | read |
| executors keep model-private kernels and layouts on purpose | `gufo-neuhaus/src/models/deepseek_v4_flash/UPSTREAM.md:12-13,29-31` ("intentionally duplicates numerical code instead of calling Qwen kernels") (catalogue) | read |
| the same model family is served by three executors with three weight formats and three splits of the mixer | `docs/INTERFACES.md` sections 6 and 7 | read |
| "swappable behind one ABI" is the plan of SPEC section 6, not yet shown | - | not verified |

## 5. Kernel launch is cheap; memory bandwidth is the ceiling; static assembly is affordable

| claim | where | kind |
|---|---|---|
| read peak about 226 GB/s; tokens/s is about bandwidth divided by bytes per token | `strixite-v070pod/docs/how-it-got-fast.md:53-58` | measured (origin) |
| a launch costs about 2.8 us on this ROCm; HIP graphs did not pay | `strixite-v070pod/docs/how-it-got-fast.md:243-244`; the time-per-token pie at `:64-70` (about 1,130 launches, 3.18 ms of host turnaround and gaps) | measured (origin) |
| at 1 row, three launches a layer cost about 55 to 70 us a layer against the weight read (so about 18 to 23 us a launch as seen from the stream): the SPEC's "about 0.024 ms" is of this size | `meta/meta-o2-perclass.md:238-239`; `meta/meta-s1-widget-catalog.md:185` | measured (pod) |
| the same source calls that class "launch-bound at 1 row" | `meta/meta-o2-perclass.md:238-239` | measured (pod) |
| the fast engines link or bind kernels statically, with no runtime dispatch layer | strixite: `CMakeLists.txt:181-183`; gufo: `src/families/qwen/kernels/CMakeLists.txt:5`; StrixFold: kernels bound by id at load, `zig/src/gpu/hip/kit.zig:1-6`; halogen's README on the pod (`/data/ark-inbox/src/github.com__peonist-ai__halogen-flash-server/README.md`): "No general-purpose runtime, no portability layer, no fallback path" | read |
| the counter-example: hipfire chooses kernels at run time from tables by format, arch and shape | `hipfire/crates/hipfire-dispatch/src/tables/`, `src/families/` (catalogue) | read |

The two launch-cost figures are not the same measurement (enqueue cost in strixite's document, per-layer launch
overhead against a bandwidth figure in the pod's). Which one the SPEC's 0.024 ms is was not found written down; see
`docs/SPEC-PROPOSALS.md`.
