#!/usr/bin/env python3
"""Read a strixw file's 128-byte header and index (never the data region), stdlib only.

Port of catalog/strixite/format/strixw/src/formats/strixw.cpp (header :339-356, index :288-321, reader :400-509,
component sizes :135-193, strix_hash64 :195-248). Also reads an n-gram table's 512-byte header
(format/ngram_table/src/formats/ngram_table.cpp:22-27) for its source_fingerprint.

  strixw_header.py FILE.strixw            -> JSON summary (meta, per-tensor records, per-class totals)
  strixw_header.py --ngram ngram.table    -> JSON of the n-gram header fields
Runs anywhere python3 exists (pipe it over ssh: `python3 - FILE < strixw_header.py`).
"""
import json
import struct
import sys

MAGIC = b"STRIXWT\0"
NGRAM_MAGIC = b"STRXNGRM"
HEADER_BYTES = 128
K1, K2 = 0x9E3779B97F4A7C15, 0xD6E8FEB86659FD93
M64 = (1 << 64) - 1
ENC = {1: "q4", 2: "q4-chunk-major", 3: "bf16", 4: "f32", 5: "i64", 6: "q8", 7: "q8-chunk-major", 8: "q6", 9: "q5",
       10: "q4-panel16", 11: "q8-panel16"}  # 10, 11: strixw v5 (catalog/flerp/format/strixw_v5)
ROLE = {0: "data", 1: "q", 2: "scale", 3: "min"}


def _mix(h, w):
    h ^= (w * K1) & M64
    return ((h ^ (h >> 31)) * K2) & M64


def strix_hash64(b):
    """strixw.cpp:195-248 (4 lanes, 32-byte blocks, tail by 8-byte words then a length-tagged remainder)."""
    n = len(b)
    h = [(n ^ K1) & M64, (n ^ K2) & M64, ((~n) & M64) ^ K1, ((~n) & M64) ^ K2]
    full = n - n % 32
    for at in range(0, full, 32):
        w = struct.unpack_from("<4Q", b, at)
        h = [_mix(h[i], w[i]) for i in range(4)]
    tail = b[full:]
    i, lane = 0, 0
    while i + 8 <= len(tail):
        h[lane] = _mix(h[lane], struct.unpack_from("<Q", tail, i)[0])
        i += 8
        lane += 1
    if i < len(tail):
        w = int.from_bytes(tail[i:] + b"\0" * (8 - (len(tail) - i)), "little")
        h[0] = _mix(h[0], w ^ ((len(tail) - i) << 56))
    r = _mix(_mix(_mix(h[0], h[1]), h[2]), h[3])
    r ^= r >> 33
    r = (r * K1) & M64
    return r ^ (r >> 29)


def component_bytes(enc, shape, g):
    """strixw.cpp:135-193: [(role, bytes)] for an encoding."""
    if enc in (1, 2, 6, 7, 8, 9):
        n, k = shape
        planes = {1: k // 2, 2: k // 2, 6: k, 7: k, 8: k // 2 + k // 4, 9: k // 2 + k // 8}[enc]
        return [("q", n * planes), ("scale", n * (k // g) * 2), ("min", n * (k // g) * 2)]
    if enc in (10, 11):  # flerp/q4panel/q4panel.hpp: one component, panels of 16 rows x (codes + scale/min pairs)
        n, k = shape
        return [("q", n // 16 * (16 * (k // 2 if enc == 10 else k) + 16 * (k // g) * 4))]
    numel = 1
    for d in shape:
        numel *= d
    return [("data", numel * {3: 2, 4: 4, 5: 8}[enc])]


class _Cur:
    def __init__(self, b):
        self.b, self.at = b, 0

    def get(self, fmt):
        v = struct.unpack_from("<" + fmt, self.b, self.at)[0]
        self.at += struct.calcsize(fmt)
        return v

    def raw(self, n):
        if self.at + n > len(self.b):
            raise ValueError("index ends inside a string")
        v = self.b[self.at:self.at + n]
        self.at += n
        return v


def read_strixw(path):
    with open(path, "rb") as f:
        h = f.read(HEADER_BYTES)
        if len(h) < HEADER_BYTES or h[:8] != MAGIC:
            raise ValueError(f"{path}: not a strixw file")
        version, hsize = struct.unpack_from("<II", h, 8)
        index_off, index_bytes, data_off, file_bytes, index_hash = struct.unpack_from("<5Q", h, 16)
        if struct.unpack_from("<Q", h, 120)[0] != strix_hash64(h[:120]):
            raise ValueError(f"{path}: header hash mismatch")
        f.seek(index_off)
        idx = f.read(index_bytes)
        f.seek(0, 2)
        size = f.tell()
    if strix_hash64(idx) != index_hash:
        raise ValueError(f"{path}: index hash mismatch")
    if size != file_bytes:
        raise ValueError(f"{path}: header says {file_bytes} bytes, file has {size}")
    c = _Cur(idx)
    meta = {}
    for _ in range(c.get("I")):
        k = c.raw(c.get("H")).decode()
        meta[k] = c.raw(c.get("I")).decode(errors="replace")
    tensors = []
    for _ in range(c.get("I")):
        name = c.raw(c.get("H")).decode()
        enc = c.get("B")
        shape = [c.get("q") for _ in range(c.get("B"))]
        g, experts = c.get("q"), c.get("q")
        parts = []
        for _ in range(c.get("H")):
            src = c.raw(c.get("H")).decode()
            parts.append((src, c.get("q"), c.get("q")))
        comps = []
        for _ in range(c.get("B")):
            role = c.get("B")
            comps.append({"role": ROLE.get(role, role), "offset": c.get("Q"), "bytes": c.get("Q"), "hash": c.get("Q")})
        want = component_bytes(enc, shape, g)
        if [(x["role"], x["bytes"]) for x in comps] != want:
            raise ValueError(f"{path}: tensor {name} components {comps} != expected {want}")
        tensors.append({"name": name, "encoding": ENC.get(enc, str(enc)), "shape": shape, "group_size": g,
                        "experts": experts, "parts": len(parts), "bytes": sum(x["bytes"] for x in comps),
                        "components": comps})
    if c.at != len(idx):
        raise ValueError(f"{path}: {len(idx) - c.at} unread index bytes")
    return {"path": path, "version": version, "file_bytes": file_bytes, "index_bytes": index_bytes,
            "data_offset": data_off, "meta": meta, "tensors": tensors}


def read_ngram_header(path):
    with open(path, "rb") as f:
        h = f.read(512)
    if h[:8] != NGRAM_MAGIC:
        raise ValueError(f"{path}: not an n-gram table")
    s = lambda at: h[at:at + 32].split(b"\0")[0].decode()
    return {"path": path, "version": struct.unpack_from("<I", h, 8)[0], "rows": struct.unpack_from("<Q", h, 16)[0],
            "row_dim": struct.unpack_from("<I", h, 24)[0], "dtype": struct.unpack_from("<I", h, 28)[0],
            "file_bytes": struct.unpack_from("<Q", h, 40)[0], "source_fingerprint": s(336), "converter_git": s(368),
            "header_hash_ok": struct.unpack_from("<Q", h, 504)[0] == strix_hash64(h[:504])}


def main(argv):
    if len(argv) == 3 and argv[1] == "--ngram":
        print(json.dumps(read_ngram_header(argv[2]), indent=1))
        return 0
    r = read_strixw(argv[1])
    for t in r["tensors"]:
        t.pop("components")
    print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
