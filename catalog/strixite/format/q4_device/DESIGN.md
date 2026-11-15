# format.strixite.q4_device: design record

## What it is for
`Q4DeviceView` (non-owning pointers q / scale / min plus N, K, G) and `Q4Device` (owning upload of a host
`Q4Weight`) (`src/runtime/q4_device.hpp:13-30`).

## Choices
- The view is three raw pointers and the group size: the device mirror of the three-plane format, no tiling or
  padding. Views from strixw point into the loader's single allocation; `Q4Device::upload` makes separate buffers
  (used for derived weights, e.g. the MTP draft head copy, `executor.strixite.qwen4exp` `qwen4exp.cpp:271`).
- G travels with the view and is the switch value of every Q4 kernel's template dispatch
  (`kernel.strixite.linear_q4` `linear_q4.hip:334-338`).

## Coupled to
`format.strixite.q4`, `format.strixite.strixw_loader`, all Q4 kernels, `runtime.strixite.qweight`.

## Prefill remarks (inferred)
A prefill-specific layout would need its own view type (as Q4ChunkMajor has), so kernels cannot be handed the
decode layout by mistake; `upload` is the existing path for a load-time derived copy.
