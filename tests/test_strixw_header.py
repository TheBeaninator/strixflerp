"""gate/strixw_header.py on a tiny synthetic strixw file laid out like StrixwWriter (strixw.cpp:252-356)."""
import os
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import strixw_header as S  # noqa: E402


def up(v, a):
    return (v + a - 1) // a * a


def write_strixw(path, meta, tensors):
    """tensors: [(name, enc_id, shape, g, experts)]; writes zero data, real hashes for header + index."""
    plans = [(n, e, sh, g, x, S.component_bytes(e, sh, g)) for n, e, sh, g, x in tensors]

    def index(offsets):
        b = struct.pack("<I", len(meta))
        for k, v in sorted(meta.items()):
            b += struct.pack("<H", len(k)) + k.encode() + struct.pack("<I", len(v)) + v.encode()
        b += struct.pack("<I", len(plans))
        for (n, e, sh, g, x, comps), offs in zip(plans, offsets):
            b += struct.pack("<H", len(n)) + n.encode() + struct.pack("<BB", e, len(sh))
            b += b"".join(struct.pack("<q", d) for d in sh) + struct.pack("<qqH", g, x, 0)
            b += struct.pack("<B", len(comps))
            role_id = {v: k for k, v in S.ROLE.items()}
            for (role, nb), off in zip(comps, offs):
                b += struct.pack("<BQQQ", role_id[role], off, nb, 0)
        return b

    ib = len(index([[0] * len(p[5]) for p in plans]))
    data_off = up(128 + ib, 4096)
    cur, offsets = data_off, []
    for p in plans:
        cur = up(cur, 4096)
        o = []
        for _, nb in p[5]:
            cur = up(cur, 256)
            o.append(cur)
            cur += nb
        offsets.append(o)
    idx = index(offsets)
    h = bytearray(128)
    h[:8] = S.MAGIC
    struct.pack_into("<II5Q", h, 8, 4, 128, 128, len(idx), data_off, cur, S.strix_hash64(idx))
    struct.pack_into("<Q", h, 120, S.strix_hash64(bytes(h[:120])))
    with open(path, "wb") as f:
        f.write(bytes(h) + idx)
        f.truncate(cur)
    return cur


class StrixwHeaderTest(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "w.strixw")
            size = write_strixw(p, {"layout": "exp_down=q4g32", "source_fingerprint": "abc123"},
                                [("experts.down", 1, [8, 256], 32, 2), ("lm_head", 6, [4, 128], 64, 0),
                                 ("embed", 3, [10, 16], 0, 0)])
            r = S.read_strixw(p)
            self.assertEqual(r["file_bytes"], size)
            self.assertEqual(r["meta"]["source_fingerprint"], "abc123")
            t = {x["name"]: x for x in r["tensors"]}
            self.assertEqual(t["experts.down"]["encoding"], "q4")
            self.assertEqual(t["experts.down"]["group_size"], 32)
            # Q4 G32: 8*256/2 codes + 2 * 8*(256/32)*2 scale/min = 1024 + 256
            self.assertEqual(t["experts.down"]["bytes"], 1024 + 256)
            self.assertEqual(t["lm_head"]["bytes"], 4 * 128 + 2 * 4 * 2 * 2)
            self.assertEqual(t["embed"]["bytes"], 320)

    def test_corrupt_header_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "w.strixw")
            write_strixw(p, {"a": "b"}, [("x", 4, [4], 0, 0)])
            with open(p, "r+b") as f:
                f.seek(10)
                f.write(b"\x07")
            with self.assertRaises(ValueError):
                S.read_strixw(p)

    def test_hash_tail(self):
        # different lengths and tails must hash differently (length-seeded lanes + tagged remainder)
        self.assertNotEqual(S.strix_hash64(b"\0" * 33), S.strix_hash64(b"\0" * 34))
        self.assertNotEqual(S.strix_hash64(b"a" * 40), S.strix_hash64(b"a" * 41))


if __name__ == "__main__":
    unittest.main()
