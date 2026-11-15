#!/usr/bin/env python3
"""Post hoc to gate/vision_dsv41_cpu.py (no criterion): where the BF16 tower's rows are far from the FP32 reference.

    <venv python> -I -B gate/vision_dsv41_posthoc.py build/vision/gate IMAGE... > gate/results/vision-dsv41-cpu312/posthoc-bf16-rows.json

Reads the rows the gate kept (R16 = DeepSeek's modules in BF16, R32 = in FP32) and prints, an image, the row
cosines' median, first percentile and minimum, how many rows are under the gate's 0.9855, and the worst row's
error and length against the median row's."""
import json, sys, numpy as np
work, out = sys.argv[1], {}
for name in sys.argv[2:]:
    r32 = np.load(f"{work}/rows/{name}/R32.npy").astype(np.float64)
    r16 = np.load(f"{work}/rows/{name}/R16.npy")
    a = (r16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    na, nr = np.sqrt((a * a).sum(1)), np.sqrt((r32 * r32).sum(1))
    cos = (a * r32).sum(1) / (na * nr)
    err = np.sqrt(((a - r32) ** 2).sum(1))
    i = int(cos.argmin())
    out[name] = {"rows": len(cos), "rows_below_0.9855": int((cos < 0.9855).sum()), "rows_below_0.99": int((cos < 0.99).sum()),
                 "cosine_median": float(np.median(cos)), "cosine_p01": float(np.quantile(cos, 0.01)), "cosine_min": float(cos.min()),
                 "worst_row": i, "worst_row_norm_over_median_norm": float(nr[i] / np.median(nr)),
                 "worst_row_abs_error_over_median_abs_error": float(err[i] / np.median(err)),
                 "row_norm_min_over_median": float(nr.min() / np.median(nr)), "row_norm_max_over_median": float(nr.max() / np.median(nr)),
                 "corr_cos_vs_norm": float(np.corrcoef(cos, nr)[0, 1]) if len(cos) > 2 else None}
json.dump(out, sys.stdout, indent=1)
