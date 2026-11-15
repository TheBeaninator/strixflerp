# format.flerp.strixw_v5: design record

## What changed from format.strixite.strixw (20937dc)
Adapted copies of `formats/strixw.hpp` / `formats/strixw.cpp`, changes marked `flerp (format.flerp.strixw_v5)`:
- `StrixwEncoding::Q4Panel16 = 10`, `Q8Panel16 = 11`, names `q4-panel16`, `q8-panel16`;
- `kStrixwVersion` 4 -> 5 (`strixw.hpp:24` of the origin): the writer writes v5, the reader accepts v1..v5;
- `strixw_components`: a panel tensor is ONE component, role Q, of N/16 * panel_bytes bytes (format.flerp.q4panel's
  `panel16_geom`), because a panel interleaves codes with their scale/min; it refuses what panel16_geom refuses and
  experts whose per-expert rows are not whole panels.
Unchanged: header (128 B, hashes), index encoding, 4096-byte tensor / 256-byte component alignment, StrixHasher,
writer streaming, reader validation. A v1..v4 file reads exactly as before.

## Layout string and pairing
The converter appends `p16` to each converted class in the `layout` metadata (`exp_gu=q4g128p16`), so the
prompt-cache key (`qwen4exp_backend.cpp:225-241` hashes `layout` with the fingerprint) changes and saved states are
dropped, not misread. `source_fingerprint` is copied, so vision.strixw and the n-gram table still pair
(`qwen4exp.cpp:169-171`). strixite's own `parse_layout` would read `q4g128p16` as G128 via stoll; no strixite
binary can load a v5 file anyway (its reader stops at v4), which is the intended guard until stage 5.

## Build
`flerp check` sees the include root `src`; a build puts `src/flerp/strixw_v5` ahead of the origin's root
(tool.flerp.convert_strixw_panel's build.sh).
