# format.strixite.strixw_loader: design record

## What it is for
Puts a whole strixw file into GPU-visible memory once at startup and hands out typed views (`Q4DeviceView`,
`Q8DeviceView`, `Q5/Q6DeviceView`, `QWeightView`, chunk-major views) by tensor name.

## Choices (origin's code)
- One `hipMalloc` for the data region; on gfx1151 that is GTT, i.e. system RAM ("124 GiB of a 128 GB machine")
  (`src/runtime/strixw_loader.hpp:3-8`).
- A component's device pointer is base + (file offset - data offset); runs keep their offset mod 4096
  (`src/runtime/strixw_loader.cpp:121-127`), so the file's 4096/256 alignment carries into device alignment.
- Read path: 8 parallel `pread`s (NVMe queue depth, `strixw_loader.cpp:33-55`, default `read_threads = 8`,
  `.hpp:42-43`) into two 256 MiB pinned staging buffers (`.cpp:25`), each reused only after an event says its last
  async copy finished; "reusing a pinned buffer for the next async copy without a gate corrupts data" on gfx1151
  (`.hpp:4-6`, gate at `.cpp:170-171`). Page cache dropped afterwards with `POSIX_FADV_DONTNEED` (`.cpp:191`).
- Every view comes from one routine shape: tensor lookup by name + expected encoding + shape, then pointers to the
  Q / Scale / Min components and the tensor's own group size (`.cpp:226-248`).
- Header and index hashes are checked (via `StrixwFile`); component hashes are not (`.hpp:42-43`).
- `WeightPlacement::HostExperts` (tools only): expert stacks in pinned host memory read over PCIe (`.hpp:9-15`).

## What it buys / costs
- Buys: zero conversion at load, weights at full bandwidth, no second copy in page cache.
- Costs (inferred): the device copy is the whole file (fleet: 69.5 GiB of GTT); any load-time derived copy
  (a prefill repack) adds to that budget. The loader has no hook for a derived layout today; the only derived copy,
  the Q4 draft head, is made by the executor after load (`executor.strixite.qwen4exp` `qwen4exp.cpp:229-281`).

## Coupled to
`format.strixite.strixw` (reader), `q4_device` / `q5_device` / `q6_device` / `q8_device` (view types),
`runtime.strixite.qweight` (`qw()` builds the format-dispatching view), `runtime.strixite.device_buffer`.

## Prefill remarks
A load-time repack for prefill would naturally live here (or beside it): after the device copy, a kernel pass
writes a second layout into another allocation, then views for prefill kernels point at it. Memory is the
constraint: the expert stacks are 68 GB in the fleet file, so a full second copy does not fit beside the first;
a repack of scales/mins only (7.5 GB in the fleet file) or of the dense projections only (~3.5 GB) would.
