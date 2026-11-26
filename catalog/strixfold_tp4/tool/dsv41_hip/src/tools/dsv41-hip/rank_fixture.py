#!/usr/bin/env python3
"""Checkpoint fixtures for the rank-cache tools at the real DeepSeek-V4.1 shapes, from an inventory.json
(inventory_checkpoint.py: every tensor's name, dtype, shape and shard; no weight bytes needed).

--sparse: every shard with its real header and its data left as a hole but the MUL1 markers (no disk used): the CPU tools' --index-only
  and the Python rank schema see the real layouts, so rank-file sizes come out exact before the source is staged.
--toy: a small checkpoint with random bytes for a full build and byte check: each EXL3 trellis keeps its K / 16 and
  N / 16 tiles but holds one 16-bit word a tile (the split tools slice tiles, never read the words), suh / svh and
  every tensor the rank files hold keep their real shapes, embed.weight shrinks to 16 rows, and the tensors no rank
  file holds (vision, aligner, image markers) are left out. MUL1 markers keep their word.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import struct
from collections import defaultdict
from pathlib import Path

MUL1 = 0x83DCD12D
BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I16": 2, "I32": 4, "U8": 1, "I8": 1}


def shape_of(t: dict, toy: bool) -> list:
    s = list(t["shape"])
    if toy and t["name"].endswith(".trellis"):
        s[2] = 1
    if toy and t["name"] == "embed.weight":
        s[0] = 16
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inventory", type=Path)
    ap.add_argument("config", type=Path)
    ap.add_argument("out", type=Path)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sparse", action="store_true")
    mode.add_argument("--toy", action="store_true")
    ap.add_argument("--seed", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True)
    shutil.copyfile(a.config, a.out / "config.json")
    inv = json.loads(a.inventory.read_text())
    shards = defaultdict(list)
    for t in inv["tensors"]:
        if a.toy and t["name"].split(".")[0] in ("vision", "aligner", "image_start", "image_newline", "image_end"):
            continue
        shards[t["shard"]].append(t)
    rng = random.Random(a.seed)
    total = 0
    for name, ts in sorted(shards.items()):
        ts.sort(key=lambda t: t["data_offset"])
        head, at, layout = {}, 0, []
        for t in ts:
            s = shape_of(t, a.toy)
            n = BYTES[t["dtype"]]
            for d in s:
                n *= d
            if not a.toy:
                if n != t["bytes"] or t["data_offset"] != at:
                    raise SystemExit(f"{t['name']}: inventory layout is not contiguous")
            head[t["name"]] = {"dtype": t["dtype"], "shape": s, "data_offsets": [at, at + n]}
            layout.append((t["name"], n))
            at += n
        body = json.dumps(head, separators=(",", ":")).encode()
        body += b" " * (-len(body) % 8)
        with open(a.out / name, "wb") as f:
            f.write(struct.pack("<Q", len(body)))
            f.write(body)
            if a.sparse:
                f.truncate(8 + len(body) + at)
                for tname, n in layout:
                    if tname.endswith(".mul1"):
                        f.seek(8 + len(body) + head[tname]["data_offsets"][0])
                        f.write(struct.pack("<I", MUL1))
            else:
                for tname, n in layout:
                    if tname.endswith(".mul1"):
                        f.write(struct.pack("<I", MUL1))
                    else:
                        f.write(rng.randbytes(n))
        total += at
    print(json.dumps({"out": str(a.out), "mode": "sparse" if a.sparse else "toy", "shards": len(shards), "data_bytes": total}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
