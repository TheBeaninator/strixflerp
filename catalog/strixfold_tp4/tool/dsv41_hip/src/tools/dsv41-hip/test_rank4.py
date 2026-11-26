#!/usr/bin/env python3
"""TP4 rank files on CPU fixtures at the real DeepSeek-V4.1 shapes (rank_fixture.py), no GPU and no weights:

1. sparse (real headers, data holes): for TP2 ranks 0-1 and TP4 nodes 0-3, the Zig tool's --index-only index is
   byte-identical to the Python schema's (inventory_checkpoint.rank_plan: same sha256 and length), and the sizes are
   the real rank files' sizes;
2. toy (random bytes, one word a trellis tile): prepare_inputs.build_rank_caches builds rank0of4..rank3of4 and checks
   every entry's bytes (--bytes 1) against the checkpoint, the file name carries the Python index hash, and each
   node's slices of every split tensor tile its TP2 rank's slice (pairs 0 and 1 side by side);
3. the tools refuse node 4 of 4, world 3 and --parity.

usage: test_rank4.py INVENTORY_JSON CONFIG_JSON WORKDIR BIN_DIR   (WORKDIR/sparse and WORKDIR/toy are made if absent)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inventory_checkpoint import rank_plan, scan  # noqa: E402
from prepare_inputs import build_rank_caches  # noqa: E402


def run(args, **kw):
    return subprocess.run([str(a) for a in args], capture_output=True, **kw)


def main() -> int:
    inventory, config, work, bindir = (Path(x) for x in sys.argv[1:5])
    rank_cache, check = bindir / "tf-dsv41-rank-cache", bindir / "tf-dsv41-check"
    for mode in ("sparse", "toy"):
        if not (work / mode).exists():
            subprocess.run([sys.executable, HERE / "rank_fixture.py", inventory, config, work / mode, "--" + mode], check=True)
    failures, summary = [], {}

    # 1. the real schema: Zig index == Python index, TP2 and TP4
    model = work / "sparse"
    raw = json.loads((model / "config.json").read_text())
    tensors, _ = scan(model)
    for world in (2, 4):
        for g in range(world):
            plan = rank_plan(raw, tensors, g, world)
            p = run([rank_cache, model, work / "unused", g, world, "--index-only"])
            got = hashlib.sha256(p.stdout).hexdigest()
            ok = p.returncode == 0 and got == plan["index_sha256"] and len(p.stdout) == plan["index_bytes"]
            summary[f"real rank{g}of{world}"] = {"rank_file_bytes": plan["rank_file_bytes"], "entries": plan["entry_count"],
                                                 "index_sha20": got[:20], "zig_equals_python": ok}
            if not ok:
                failures.append(f"real rank{g}of{world}: zig index {got[:20]} rc {p.returncode}, python {plan['index_sha256'][:20]}")

    # 2. the toy build: four files, full byte checks, schema and hash
    model = work / "toy"
    raw = json.loads((model / "config.json").read_text())
    tensors, _ = scan(model)
    plans = [rank_plan(raw, tensors, g, 4) for g in range(4)]
    out = work / "toy-tp4"
    if out.exists():
        raise SystemExit(f"{out} exists: remove it first")
    out.mkdir()
    reports = build_rank_caches(model, out, {"ranks": plans}, rank_cache, check)
    for rep, plan in zip(reports, plans):
        name = Path(rep["file"]).name
        r = rep["byte_check"]["report"]
        ok = name == f"rank{plan['rank']}of4-{plan['index_sha256'][:20]}.bin"
        summary[f"toy rank{plan['rank']}of4"] = {"file": name, "bytes": rep["bytes"], "compared": r["compared"],
                                                 "compared_bytes": r["compared_bytes"], "byte_mismatches": r["byte_mismatches"],
                                                 "name_hash_is_python_index": ok}
        if not ok:
            failures.append(f"toy {name}: not the Python index hash")
    # each split tensor: node (r, pair 0) and (r, pair 1) tile TP2 rank r's ranges (keys say the ranges)
    tp2 = [{e[0].split("|")[0] + "|" + e[0].split("|")[-1]: e for e in rank_plan(raw, tensors, r, 2)["entries"] if "|" in e[0] and e[0].count("|") == 3}
           for r in range(2)]
    tiled = 0
    for r in range(2):
        a = {e[0].split("|")[0] + "|" + e[0].split("|")[-1]: e for e in plans[r]["entries"] if e[0].count("|") == 3}
        b = {e[0].split("|")[0] + "|" + e[0].split("|")[-1]: e for e in plans[r + 2]["entries"] if e[0].count("|") == 3}
        for key, e in tp2[r].items():
            if not key.endswith("|tr") or key not in a or key not in b or a[key][2] == e[2]:
                continue
            ea, eb = a[key][2], b[key][2]
            # trellis [K/16, N/16, words]: the pairs split N (columns) and keep K (rows)
            if not (ea[0] == eb[0] == e[2][0] and ea[1] + eb[1] == e[2][1] and ea[2] == eb[2] == e[2][2]):
                failures.append(f"{key}: pair shapes {ea} + {eb} do not tile TP2 rank {r}'s {e[2]}")
            tiled += 1
    summary["split tensors tiled by the pairs"] = tiled
    if tiled == 0:
        failures.append("no split tensor compared")

    # 3. refusals
    for args in ([4, 4], [0, 3], [0, 4, "--parity"]):
        p = run([rank_cache, model, work / "unused", *args, "--index-only"])
        q = run([check, model, out / "rank0" / "rank-cache", *args])
        summary[f"refuse {' '.join(map(str, args))}"] = [p.returncode, q.returncode]
        if p.returncode == 0 or q.returncode == 0:
            failures.append(f"accepted {args}")
    print(json.dumps(summary, indent=1))
    print("FAIL" if failures else "PASS", *failures, sep="\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
