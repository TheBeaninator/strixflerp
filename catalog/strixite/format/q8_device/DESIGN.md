# format.strixite.q8_device: design record

## What it is for
`Q8DeviceView` / `Q8Device`: the device mirror of Q8's three planes (`src/runtime/q8_device.hpp:12-20`), same shape
as `Q4DeviceView`. Views come from `StrixwDevice::q8` (`format.strixite.strixw_loader` `strixw_loader.cpp:232-236`).

## Coupled to
`format.strixite.q8`, `strixw_loader`, `kernel.strixite.linear_q8`, `linear_wmma`, `hc`, `hc_wmma`,
`executor.strixite.qwen4exp` (the Q8 LM head is read back through this view to build the Q4 draft head,
`qwen4exp.cpp:250-266`).

## Prefill remarks
None specific; the view carries G so the WMMA kernels can pick their template.
