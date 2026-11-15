#!/usr/bin/env python3
"""exl3_pf experts: as deployed / no-decode / cache-resident timings and a profiling pass (criteria gate/exl3-pf-bytes.toml).

    python3 gate/exl3_pf_bytes.py --kit DIR --arm a|b|c [--rows 512,1024] [--k2 6,4] [--repeats 5] [--profile] --out FILE
--arm c aliases the routed experts' pointer tables to experts e % 2 (weights then fit MALL / L2). --profile: no timing,
each phase launched 3 times (run under rocprofv3 --pmc). Reuses gate/prefill_dsv41_bench.py (Kit, Experts, routing).
"""
import argparse
import json
import os
import statistics
import sys
import tomllib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import prefill_dsv41_bench as pb  # noqa: E402
import strixite_recipe as guard  # noqa: E402


def alias(ex):
    for tab in ex.ptrs:
        t = tab.get(np.int64, pb.E)
        t[:pb.E - 1] = t[np.arange(pb.E - 1) % 2]
        tab.put(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True)
    ap.add_argument("--arm", required=True, choices=["a", "b", "c"])
    ap.add_argument("--rows", default="512,1024")
    ap.add_argument("--k2", default="6,4")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--no-lease", action="store_true", help="the caller holds control-gpu (rocprofv3 hangs injecting into pod-lease)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hip-lib", default="/data/models/tools/therock-10.2/lib/libamdhip64.so")
    a = ap.parse_args()
    rows_list = [int(r) for r in a.rows.split(",")]
    with open(os.path.join(ROOT, "gate", "interface", "host.toml"), "rb") as f:
        cfg = {"host": dict(tomllib.load(f)["host"])}
    cfg["host"]["lease_owner"] = "agent"
    cfg["host"]["lease_ttl_s"] = 900
    ok, state = guard.gpu_free(cfg)
    if not ok:
        sys.exit(f"the GPU is not free; nothing was run: {state}")
    if guard.pod_waiters():
        sys.exit("the GPU is not free; a pod job waits in pod-gpu --wait")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    lease = guard.Lease(cfg, a.out + ".lease.txt")
    doc = {"host": os.uname().nodename, "utc": guard.utc(), "kit": a.kit, "arm": a.arm, "profile": a.profile,
           "loadavg_start": open("/proc/loadavg").read().split()[:3], "gpu_before": state, "results": []}
    try:
        doc["lease"] = "held by caller" if a.no_lease else lease.acquire(f"exl3_pf bytes arm {a.arm}: kernels alone, minutes")
        pb.lib = __import__("ctypes").CDLL(a.hip_lib)
        pb.check(pb.lib.hipInit(0), "hipInit")
        pb.check(pb.lib.hipSetDevice(0), "hipSetDevice")
        kit = pb.Kit(a.kit)
        doc["kit_manifest_sha256"] = kit.sha256
        for k2 in [int(k) for k in a.k2.split(",")]:
            ex = pb.Experts(k2, max(rows_list))
            if a.arm == "c":
                alias(ex)
            for rows in rows_list:
                if a.profile:
                    ids_h = pb.routing(rows, "uniform")
                    x = pb.Dev(rows * pb.D * 2, pb.bf16((rows, pb.D)))
                    ids = pb.Dev(rows * pb.SLOTS * 4, ids_h)
                    wts = pb.Dev(rows * pb.SLOTS * 4, pb.rng.random((rows, pb.SLOTS), dtype=np.float32))
                    out = pb.Dev(rows * pb.D * 4)
                    args, nm, mj = ex.plan(x.ptr, ids.ptr, wts.ptr, out.ptr, rows)
                    for _ in range(3):
                        for _, c in ex.phases(kit, args, nm, mj):
                            c()
                        pb.sync()
                    counts = np.bincount(ids_h.ravel(), minlength=pb.E)
                    r = {"rows": rows, "k2": k2, "nm": nm, "jobs": int(((counts + nm - 1) // nm).sum()), "profiled": 3}
                    for b in (x, ids, wts, out):
                        b.free()
                else:
                    runs = [pb.bench_experts(kit, ex, rows, "uniform") for _ in range(a.repeats)]
                    r = {"rows": rows, "k2": k2, "jobs": runs[0]["jobs"], "repeats": a.repeats,
                         "ms": [x["ms"] for x in runs], "phases": {}}
                    for ph in runs[0]["phases_ms"]:
                        v = [x["phases_ms"][ph] for x in runs]
                        r["phases"][ph] = {"median": round(statistics.median(v), 4), "min": min(v), "max": max(v)}
                doc["results"].append(r)
                print(json.dumps(r), flush=True)
            pb.free_all()
        doc["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
        doc["finished"] = guard.utc()
    finally:
        pb.free_all()
        if not a.no_lease:
            lease.release()
        with open(a.out, "w") as f:
            json.dump(doc, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
