#!/usr/bin/env python3
"""SHAPE-TRANSFERS R2 analysis (criteria gate/xfer-r2-exl3-intercept.toml): from gate/results/xfer-r2{a,b}/
(bench.json + rocprofv3 kernel trace) fit per phase kernel us = c0 + c_job * jobs + c_row * rows."""
import csv
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res = {}
for arm in "ab":
    d = os.path.join(ROOT, f"gate/results/xfer-r2{arm}")
    if not os.path.exists(os.path.join(d, "bench.json")):
        continue
    b = json.load(open(os.path.join(d, "bench.json")))
    tr = glob.glob(os.path.join(d, "trace", "**", "*kernel_trace.csv"), recursive=True)
    names = {}
    for f in tr:
        for r in csv.DictReader(open(f)):
            n = r["Kernel_Name"]
            names.setdefault(n, []).append((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3)
    out = {"loadavg": [b.get("loadavg_start"), b.get("loadavg_end")], "trace_kernels": {n: {"dispatches": len(v), "median_us": float(np.median(v))} for n, v in names.items()}, "fits": {}}
    for k2 in sorted({r["k2"] for r in b["results"]}):
        rs = [r for r in b["results"] if r["k2"] == k2]
        for ph in (f"gu_k{k2}", f"down_k{k2}"):
            jobs = np.array([r["jobs"] for r in rs], float)
            rows = np.array([r["rows"] for r in rs], float)
            us = np.array([r["phases"][ph]["median"] * 1e3 for r in rs])
            A = np.stack([np.ones_like(jobs), jobs, rows], 1)
            c, *_ = np.linalg.lstsq(A, us, rcond=None)
            A2 = np.stack([np.ones_like(rows), rows], 1)
            c2, *_ = np.linalg.lstsq(A2, us, rcond=None)
            out["fits"][ph] = {"rows": rows.tolist(), "jobs": jobs.tolist(), "us": us.round(1).tolist(),
                               "us_per_job": (us / jobs).round(2).tolist(),
                               "fit3": {"c0_us": round(c[0], 1), "c_job_us": round(c[1], 3), "c_row_us": round(c[2], 4)},
                               "fit_rows_only": {"intercept_us": round(c2[0], 1), "us_per_row": round(c2[1], 4),
                                                 "intercept_per_job_us_at_512_1024": [round(c2[0] / j, 2) for r_, j in zip(rows, jobs) if r_ in (512, 1024)]}}
    res[arm] = out
json.dump(res, sys.stdout, indent=1)
