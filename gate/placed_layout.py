#!/usr/bin/env python3
"""Per-rank read logs of a format.flerp.gguf_placed stage plan, in the format `loom-swarm manifest <file> --layout`
expects ("offset length" lines, one byte range of the file a line). Standard library only; reads the GGUF header only.

A rank's log is the file header [0, data_start) (the placed loader parses it from the file) plus every tensor the
plan gives that rank, at its file offset and GGUF byte size: exactly what LoadRank reads. loom-swarm coalesces
adjacent ranges itself.

    python3 gate/placed_layout.py FILE.gguf --world 4 --out DIR       DIR/<file>.r<r>of<W>.ranges + DIR/plan.json

plan.json records the plan hash (compare with census.json's placement sha256) and each rank's bytes.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_census as C  # noqa: E402


def rank_ranges(h, plan, rank):
    """[[offset, length], ...] in file order: the header, then the rank's tensors."""
    rs = [[0, h["data_start"]]]
    rs += [[foff, size] for _name, _tid, size, foff, _doff in sorted(plan["ranks"][rank]["entries"], key=lambda e: e[3])]
    return rs


def write_logs(gguf, world, out, block_range=None):
    h = C.read_header(gguf)
    plan = C.stage_plan(h, world, block_range)
    os.makedirs(out, exist_ok=True)
    base = os.path.basename(gguf)
    doc = {"file": gguf, "file_bytes": h["file_bytes"], "data_start": h["data_start"], "world": world,
           "cuts": plan["cuts"], "plan_sha256": C.plan_hash(plan), "ranks": []}
    for rk in plan["ranks"]:
        r = rk["rank"]
        rs = rank_ranges(h, plan, r)
        path = os.path.join(out, f"{base}.r{r}of{world}.ranges")
        with open(path, "w") as f:
            f.writelines(f"{o} {n}\n" for o, n in rs)
        doc["ranks"].append({"rank": r, "blocks": rk["blocks"], "tensor_bytes": rk["tensor_bytes"],
                             "layout_bytes": sum(n for _, n in rs), "ranges": len(rs), "log": path})
    with open(os.path.join(out, "plan.json"), "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("gguf")
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--range", help="B0:B1, blocks [B0, B1) at world 1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = tuple(int(x) for x in a.range.split(":")) if a.range else None
    doc = write_logs(a.gguf, a.world, a.out, rng)
    for r in doc["ranks"]:
        print(f"rank {r['rank']} blocks {r['blocks']}: {r['layout_bytes']} bytes "
              f"(tensors {r['tensor_bytes']} + header {doc['data_start']}) -> {r['log']}")
    print("plan sha256", doc["plan_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
