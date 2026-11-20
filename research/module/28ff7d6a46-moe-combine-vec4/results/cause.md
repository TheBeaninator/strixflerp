# MOE-COMBINE cause (bench job 20261114T102419Z-run-cause1, gate/combine_bench.hip, criteria gate/moe-combine.toml [criteria.cause])

| dims (tokens x slots x dim) | orig isolated | orig after writer | vec4o isolated | probe isolated | orig vs vec4o words differing |
|---|---|---|---|---|---|
| 4,096 x 8 x 2,048 (35B) | 6.9 ms, 48 GB/s | 6.94 ms, 48.3 GB/s | 1.78 ms, 189 GB/s | 1.71 ms, 196 GB/s | 0 of 8,388,608 |
| 4,096 x 10 x 4,096 (Ornith) | 15.66 ms, 51.4 GB/s | 15.66 ms, 51.4 GB/s | 3.97 ms, 203 GB/s | 3.87 ms, 208 GB/s | 0 of 16,777,216 |

**Rule that holds: kernel_shape.**

- The original runs under 100 GB/s isolated (48-51 GB/s).
- The same expression with 16-byte loads (vec4o) runs at 3.9x the original, and a plain read of the same bytes (probe) at 4.0x.
- `in_context` does not hold: after the writer, the original runs at 1.00x its isolated rate.
- `memory` does not hold: the probe reads at 196-208 GB/s.

The 35B in-model figure (7.0 ms a block, PREFILL-KPROF p2) is the isolated kernel's own 6.9 ms. One 4-byte load per thread over tokens x dim/256 blocks is load-issue bound. The fix keeps the expression, so C0 passes at both dims.
