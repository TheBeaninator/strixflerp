# Gate: vision-dsv41-cpu312 (`runtime.flerp.vision_sidecar`, environment `py312-torch2110`)

Host control-node, AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, written 2026-11-11T20:54:08Z. **PASSED** under the criteria of `gate/vision-dsv41-cpu312.toml` (sha256 `d9088dda6be4b34f`, criterion 3a as restated: see below).

Python 3.12.3, torch 2.11.0+cpu, numpy 2.3.5, Pillow 12.3.0; 8 threads. Reference: DeepSeek `inference/vision.py`, git blob `77af0bdef9f4649edebac2876cd1df50a147ce98` (the pinned blob), with the checkpoint's 266 tower tensors read from the shard by the gate's own reader.

| criterion | held |
|---|---|
| 1 shape | yes |
| 2 arithmetic | yes |
| 3a bf16 rows = R16 | yes |
| 3b bf16 vs R16 | yes |
| 4 determinism | yes |
| 5 protocol | yes |

## Criteria as first committed, and why 3a was restated

The criteria were committed in 2c20dd2 before any run. The run (1cdfa3e, config sha256 `9dd87b9c8cebe560`, which `env.json` still carries: it is that run's) was judged under them and **FAILED**, on 3a alone:

- 3a bf16 bound: tiny: rel rms 0.0235, lowest row cosine 0.979462
- 3a bf16 bound: text: rel rms 0.0388, lowest row cosine 0.867161
- 3a bf16 bound: cap: rel rms 0.0268, lowest row cosine 0.898884
- 3a bf16 bound: over: rel rms 0.0291, lowest row cosine 0.934885

3a then was an absolute bound of the worker's BF16 rows against the FP32 reference: relative RMS <= 0.085 an image (held, 11 of 11) and cosine >= 0.9855 for every row (broken).

Restated 2026-11-11 by the owner's decision: The old bound measured the tower's own BF16 behaviour: the reference (DeepSeek inference/vision.py in BF16) produces the same outlier rows, bit for bit. Restated as bit equality of the worker's BF16 rows with the reference's.

The verdict at the top is under the restated 3a. It was judged from the bit comparisons stored by that same run (`protocol.json`, `W_vs_R16_bits`): no tower forward was run again. The other five criteria and every number below are unchanged.

On all 11 images the worker's aligner rows are, bit for bit, the rows DeepSeek's own modules give in BF16 (R16) on this torch. Whatever a criterion of group 3 says about the worker it says about the reference's own BF16 evaluation. The rows far from FP32 are in the post hoc table below.

## Rows against the reference

`rel rms` is ||a - R32|| / ||R32|| over an image's aligner rows; `cos` the lowest row cosine. R32 = DeepSeek's modules in FP32; R16 = the same in BF16; S32 = the sidecar's `encode()` on FP32 tensors; W = the worker's reply.

| image | patches | tokens | S32 vs R32 rel rms | S32 cos | W vs R32 rel rms | W cos | R16 vs R32 rel rms | W / R16 | W max abs / rms | R32 vs R64 rel rms |
|---|---|---|---|---|---|---|---|---|---|---|
| tiny | 39 x 39 = 1521 | 184 | 0 | 1.000000000 | 0.0235 | 0.979462 | 0.0235 | 1.000 | 0.627 | 1.5e-06 |
| photo | 35 x 46 = 1610 | 206 | 0 | 1.000000000 | 0.0272 | 0.988743 | 0.0272 | 1.000 | 0.562 | 5.86e-06 |
| photo-jpeg | 35 x 46 = 1610 | 206 | 0 | 1.000000000 | 0.0222 | 0.996356 | 0.0222 | 1.000 | 0.503 | 2.85e-06 |
| text | 43 x 58 = 2494 | 317 | 0 | 1.000000000 | 0.0388 | 0.867161 | 0.0388 | 1.000 | 2.08 | 1.48e-06 |
| rgba | 50 x 36 = 1800 | 223 | 0 | 1.000000000 | 0.02 | 0.992211 | 0.02 | 1.000 | 0.438 | 3.24e-05 |
| palette | 68 x 23 = 1564 | 209 | 0 | 1.000000000 | 0.0287 | 0.996405 | 0.0287 | 1.000 | 0.393 | 2.66e-06 |
| wide | 9 x 174 = 1566 | 179 | 0 | 1.000000000 | 0.0236 | 0.994681 | 0.0236 | 1.000 | 0.421 | 1.72e-06 |
| cap | 42 x 216 = 9072 | 1024 | 0 | 1.000000000 | 0.0268 | 0.898884 | 0.0268 | 1.000 | 1.19 | 3.28e-06 |
| over | 93 x 93 = 8649 | 994 | 0 | 1.000000000 | 0.0291 | 0.934885 | 0.0291 | 1.000 | 1.36 | 3.28e-06 |
| grid-1x1 | 1 x 1 = 1 | 4 | 0 | 1.000000000 | 0.00204 | 0.999998 | 0.00204 | 1.000 | 0.0174 | 2.12e-07 |
| grid-2x5 | 2 x 5 = 10 | 5 | 0 | 1.000000000 | 0.0107 | 0.999944 | 0.0107 | 1.000 | 0.0484 | 8.23e-07 |

## Bits (BF16 rows; recorded, not criteria except the last two columns)

Rows that differ in any bit / rows compared. `round(R32)` is R32 rounded once to BF16: the closest BF16 rows to the full-precision result.

| image | W vs R16 | S16m (no query pieces) vs R16 | S16 vs S16m | 1 thread vs 8 | W vs round(R32): values equal, within one BF16 step | R16 vs round(R32): values equal | worker twice | worker = in-process |
|---|---|---|---|---|---|---|---|---|
| tiny | 0 / 169 | 0 / 169 | 0 / 169 | 0 / 184 | 10.5%, 29.3% | 10.5% | same bytes | yes |
| photo | 0 / 192 | 0 / 192 | 0 / 192 | 0 / 206 | 10.7%, 29.6% | 10.7% | same bytes | yes |
| photo-jpeg | 0 / 192 | 0 / 192 | 0 / 192 | 0 / 206 | 10.6%, 29.6% | 10.6% | same bytes | yes |
| text | 0 / 300 | 0 / 300 | 0 / 300 | 0 / 317 | 12.4%, 33.4% | 12.4% | same bytes | yes |
| rgba | 0 / 204 | 0 / 204 | 0 / 204 | 0 / 223 | 11.2%, 30.9% | 11.2% | same bytes | yes |
| palette | 0 / 184 | 0 / 184 | 0 / 184 | 0 / 209 | 8.4%, 23.8% | 8.4% | same bytes | yes |
| wide | 0 / 174 | 0 / 174 | 0 / 174 | 0 / 179 | 10.1%, 28.1% | 10.1% | same bytes | yes |
| cap | 0 / 1008 | 0 / 1008 | 0 / 1008 | 0 / 1024 | 10.6%, 29.3% | 10.6% | same bytes | yes |
| over | 0 / 961 | 0 / 961 | 0 / 961 | 0 / 994 | 10.0%, 27.9% | 10.0% | same bytes | yes |
| grid-1x1 | 0 / 1 | 0 / 1 | 0 / 1 | 0 / 4 | 56.9%, 85.4% | 56.9% | same bytes | yes |
| grid-2x5 | 0 / 2 | 0 / 2 | 0 / 2 | 0 / 5 | 14.7%, 40.4% | 14.7% | same bytes | yes |

## The worker over its pipe: time and memory an image

One process an image, as `vision_cpu.zig` starts it (`python -I -c <source> <weights> <sha256>`), 8 threads, niced. The protocol has no warm form: every image pays the interpreter, `import torch` (0.755 s alone) and a sha256 of the weights (0.434 s alone, file cache warm). `encode only` is the same forward inside an already loaded process (arm S16).

| image | patches | tokens | worker wall s (run 1, run 2) | CPU s (user + sys) | peak RSS GiB | encode only s (8 threads) | encode only s (1 thread) | reference R16 s | within the origin's 240 s |
|---|---|---|---|---|---|---|---|---|---|
| tiny | 1521 | 184 | 3.6, 5.9 | 17 | 1.27 | 1.7 | 8.0 | 5.1 | yes |
| photo | 1610 | 206 | 3.9, 3.5 | 20 | 1.27 | 2.0 | 12.5 | 1.7 | yes |
| photo-jpeg | 1610 | 206 | 8.0, 3.7 | 48 | 1.26 | 1.9 | 8.9 | 1.7 | yes |
| text | 2494 | 317 | 9.9, 6.0 | 51 | 1.34 | 5.5 | 19.8 | 8.3 | yes |
| rgba | 1800 | 223 | 7.1, 4.1 | 30 | 1.28 | 2.3 | 11.2 | 2.7 | yes |
| palette | 1564 | 209 | 3.8, 6.0 | 17 | 1.26 | 2.7 | 11.1 | 2.8 | yes |
| wide | 1566 | 179 | 3.4, 6.6 | 16 | 1.26 | 4.3 | 13.6 | 1.6 | yes |
| cap | 9072 | 1024 | 128.3, 130.8 | 851 | 1.79 | 127.0 | 356.7 | 102.0 | yes |
| over | 8649 | 994 | 105.7, 110.6 | 759 | 1.77 | 100.8 | 307.5 | 87.5 | yes |
| grid-1x1 | 1 | 4 | 1.5, 1.5 | 3 | 1.17 | 0.1 | 0.1 | 0.0 | yes |
| grid-2x5 | 10 | 5 | 1.5, 1.4 | 2 | 1.17 | 0.0 | 0.1 | 0.0 | yes |

## Refusals and admission

| case | exit | bytes on stdout | the worker says |
|---|---|---|---|
| a wrong weights sha256 | 1 | 0 | `ValueError: vision weights failed SHA256` |
| FLERP_VISION_ENV unset (the origin's pin) | 1 | 0 | `RuntimeError: vision environment: torch is 2.11.0+cpu, the setting requires 2.9.0+cpu` |
| a setting naming another torch | 1 | 0 | `RuntimeError: vision environment: torch is 2.11.0+cpu, the setting requires 2.9.0+cpu` |
| a setting naming another python | 1 | 0 | `RuntimeError: vision environment: python is 3.12, the setting requires 3.99` |
| a malformed setting | 1 | 0 | `ValueError: FLERP_VISION_ENV: expected python=,torch=,numpy= entries, got 'torch'` |
| a setting without torch | 1 | 0 | `ValueError: FLERP_VISION_ENV must name torch` |
| a grid whose token count does not follow from its patches | 1 | 0 | `ValueError: invalid or oversized picture grid` |
| a payload one value short | 1 | 0 | `ValueError: invalid patch payload length` |
| a payload one value long | 1 | 0 | `ValueError: invalid patch payload length` |

- The origin's unmodified `vision_cpu.py` in this environment: exit 1, 0 bytes, `RuntimeError: requires the pinned CPU-only torch 2.9.0+cpu` (recorded; it is why the derived module exists).
- Admission (`verify_vision.py --env py312-torch2110`), record status `candidate`: with `--allow-candidate` exit 0, printing `FLERP_VISION_ENV=python=3.12,torch=2.11.0+cpu,numpy=2.3.5`; without, exit 1 (`vision preflight: vision environment py312-torch2110 has status 'candidate': not admitted`).

## Post hoc: where the BF16 rows are far from FP32 (no criterion; looked at after the verdict)

R16 (= the worker's rows) against R32, row by row. `error ratio` is the worst row's error over the median row's; `norm ratio` its length over the median row's.

| image | rows | median cosine | 1st percentile | lowest | rows under 0.9855 | worst row: error ratio | worst row: norm ratio |
|---|---|---|---|---|---|---|---|
| tiny | 169 | 0.999914 | 0.99381 | 0.97946 | 2 | 9.1 | 0.65 |
| photo | 192 | 0.999917 | 0.98933 | 0.98874 | 0 | 12.0 | 0.83 |
| photo-jpeg | 192 | 0.999924 | 0.99772 | 0.99636 | 0 | 4.8 | 0.89 |
| text | 300 | 0.999934 | 0.99290 | 0.86716 | 1 | 42.3 | 1.03 |
| rgba | 204 | 0.999926 | 0.99835 | 0.99221 | 0 | 8.8 | 0.95 |
| palette | 184 | 0.999818 | 0.99788 | 0.99640 | 0 | 5.8 | 1.34 |
| wide | 174 | 0.999885 | 0.99832 | 0.99468 | 0 | 7.1 | 1.09 |
| cap | 1008 | 0.999914 | 0.99595 | 0.89888 | 2 | 20.7 | 0.62 |
| over | 961 | 0.999901 | 0.99715 | 0.93489 | 4 | 17.4 | 0.77 |
| grid-1x1 | 1 | 0.999998 | 1.00000 | 1.00000 | 0 | 1.0 | 1.00 |
| grid-2x5 | 2 | 0.999950 | 0.99994 | 0.99994 | 0 | 1.1 | 1.09 |

## What the executed code does at system level (from reading every file before running it; not traced)

- `vision_cpu.py` (the worker; the origin's and the adapted copy): one process, no children. Reads stdin, opens the weights file read-only (once to hash it, once to load 266 tensors), writes stdout, sets `signal.alarm`. Writes no file. No socket. Imports torch and numpy.
- `verify_vision.py` (adapted): reads the bias and weights files (sha256) and `vision-env.json`; starts one child, `<python> -I -c` with an import-and-compare of four versions, 30 s limit. Writes only stdout.
- `prepare_vision.py` (the origin's, unmodified): reads the index and all 42 pinned shards whole to hash them (197 GB here), then writes the two outputs through `mkstemp` + `os.link` in the output directory (mode 0600). No children, no socket.
- DeepSeek `vision.py`: module definitions, torch only. `image_processor.py`: imports `urllib.request.urlopen`, which it calls only for an `http(s)` URL record; the gate hands it bytes. Both were imported from the ark with bytecode writing off.

## CPython 3.13, the same torch (record only)

`/data/projects/strixflerp/build/vision-venv313/bin/python`: worker bytes equal to CPython 3.12's on 11 of 11 images.
