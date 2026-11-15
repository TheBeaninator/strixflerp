"""gate/gguf_census.py on a tiny synthetic GGUF: the header parse, the census and the placement plan."""
import hashlib
import os
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import gguf_census as C  # noqa: E402


def s(text):
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def kv(key, vtype, payload):
    return s(key) + struct.pack("<I", vtype) + payload


def synthetic(path, blocks=4, nextn=1, q6_down=(1,), tied=False, data=True):
    """A qwen35moe-shaped file: hidden 256, 2 experts of width 256, `blocks` blocks and `nextn` past the stack.
    Returns {name: (type id, dims, bytes)} in file order."""
    tensors = [("token_embd.weight", 12, [256, 8])]
    for b in range(blocks):
        p = f"blk.{b}."
        tensors += [(p + "attn_norm.weight", 0, [256]), (p + "ffn_gate_inp.weight", 0, [256, 2]),
                    (p + "ffn_gate_exps.weight", 12, [256, 256, 2]), (p + "ffn_up_exps.weight", 12, [256, 256, 2]),
                    (p + "ffn_down_exps.weight", 14 if b in q6_down else 12, [256, 256, 2]),
                    (p + "ffn_up_shexp.weight", 12, [256, 256])]
    for b in range(blocks, blocks + nextn):
        tensors += [(f"blk.{b}.nextn.eh_proj.weight", 12, [512, 256]), (f"blk.{b}.nextn.enorm.weight", 0, [256])]
    tensors += [("output_norm.weight", 0, [256])]
    if not tied:
        tensors += [("output.weight", 14, [256, 8])]
    meta = [kv("general.architecture", 8, s("qwen35moe")), kv("general.alignment", 4, struct.pack("<I", 32)),
            kv("qwen35moe.block_count", 4, struct.pack("<I", blocks + nextn)),
            kv("qwen35moe.nextn_predict_layers", 4, struct.pack("<I", nextn)),
            kv("qwen35moe.embedding_length", 4, struct.pack("<I", 256)),
            kv("qwen35moe.expert_count", 4, struct.pack("<I", 2)), kv("qwen35moe.expert_used_count", 4, struct.pack("<I", 1)),
            kv("qwen35moe.expert_feed_forward_length", 4, struct.pack("<I", 256)),
            kv("qwen35moe.full_attention_interval", 4, struct.pack("<I", 4)),
            kv("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 3) + s("a") + s("bc") + s("def")),
            kv("tokenizer.ggml.token_type", 9, struct.pack("<IQ", 5, 3) + struct.pack("<3i", 1, 1, 1))]
    infos, off, sizes = b"", 0, {}
    for name, tid, dims in tensors:
        size = C.tensor_bytes(tid, dims)
        infos += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", tid, off)
        sizes[name] = (tid, dims, size, off)
        off = (off + size + 31) // 32 * 32
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta)) + b"".join(meta) + infos
    pad = (-len(head)) % 32
    with open(path, "wb") as f:
        f.write(head + b"\0" * pad)
        if data:
            f.write(b"\x5a" * off)
    return sizes, len(head) + pad, off


class Census(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "t.gguf")

    def tearDown(self):
        self.dir.cleanup()

    def test_header(self):
        sizes, start, data = synthetic(self.path)
        h = C.read_header(self.path)
        self.assertEqual(h["data_start"], start)
        self.assertEqual(h["file_bytes"], start + data)
        self.assertEqual(len(h["tensors"]), len(sizes))
        for t in h["tensors"]:
            tid, dims, size, off = sizes[t["name"]]
            self.assertEqual((t["type_id"], t["dims"], t["bytes"], t["offset"]), (tid, dims, size, start + off))
        self.assertEqual(h["metadata"]["tokenizer.ggml.tokens"], {"array": 3, "of": 8})
        self.assertEqual(C.check_layout(h), [])
        with open(self.path, "rb") as f:
            self.assertEqual(h["header_sha256"], hashlib.sha256(f.read(h["header_end"])).hexdigest())

    def test_reads_only_the_header(self):
        # a file cut off right after its header parses the same: nothing of the data region is read
        _, start, data = synthetic(self.path)
        whole = C.read_header(self.path)
        cut = os.path.join(self.dir.name, "cut.gguf")
        synthetic(cut, data=False)
        h = C.read_header(cut)
        self.assertEqual(h["header_sha256"], whole["header_sha256"])
        self.assertEqual([t["offset"] for t in h["tensors"]], [t["offset"] for t in whole["tensors"]])
        self.assertTrue(any("the file has" in p for p in C.check_layout(h)))
        self.assertEqual(whole["header_bytes_read"], whole["header_end"])
        self.assertLessEqual(whole["header_end"], start)

    def test_sizes(self):
        self.assertEqual(C.tensor_bytes(12, [4096, 1024, 512]), 4096 * 1024 * 512 // 256 * 144)
        self.assertEqual(C.tensor_bytes(14, [1024, 4096, 512]), 4096 * 1024 * 512 // 256 * 210)
        self.assertEqual(C.tensor_bytes(0, [4096, 512]), 4096 * 512 * 4)
        self.assertIsNone(C.tensor_bytes(12, [100]))
        self.assertIsNone(C.tensor_bytes(99, [256]))

    def test_census(self):
        synthetic(self.path)
        c = C.census(C.read_header(self.path))
        self.assertEqual(c["facts"]["blocks"], 4)
        self.assertEqual(c["q6_expert_stacks"], ["blk.1.ffn_down_exps.weight"])
        self.assertEqual(sorted(c["q6_tensors"]), ["blk.1.ffn_down_exps.weight", "output.weight"])
        self.assertEqual([t["name"] for t in c["past_the_stack"]], ["blk.4.nextn.eh_proj.weight", "blk.4.nextn.enorm.weight"])
        q4, q6 = 256 * 256 * 2 // 256 * 144, 256 * 256 * 2 // 256 * 210
        self.assertEqual(c["per_block"][0]["expert_bytes"], 3 * q4)
        self.assertEqual(c["per_block"][1]["expert_bytes"], 2 * q4 + q6)
        self.assertEqual(c["per_block"][1]["expert_types"]["ffn_down_exps"], "Q6_K")
        self.assertEqual(c["per_block"][0]["dense_bytes"], 256 * 4 + 512 * 4 + 256 * 144)
        self.assertEqual(sum(v["bytes"] for v in c["by_type"].values()), c["tensor_bytes"])
        self.assertEqual(sum(c["groups"].values()), c["tensor_bytes"])

    def test_best_cuts(self):
        self.assertEqual(C.best_cuts([1, 1, 1, 1], 0, 0, 2), [0, 2, 4])
        self.assertEqual(C.best_cuts([1, 1, 1, 1], 2, 0, 2), [0, 1, 4])           # the embedding weighs on rank 0
        self.assertEqual(C.best_cuts([1, 1, 1, 1], 0, 2, 2), [0, 3, 4])
        self.assertEqual(C.best_cuts([5, 1, 1, 1, 1, 5], 0, 0, 3), [0, 1, 5, 6])
        self.assertEqual(C.best_cuts([3, 3, 3], 0, 0, 3), [0, 1, 2, 3])
        self.assertEqual(C.best_cuts([7], 1, 1, 1), [0, 1])
        # the same largest stage both ways (4): the more even split wins, then the earliest cut
        self.assertEqual(C.best_cuts([4, 1, 1, 2], 0, 0, 3), [0, 1, 3, 4])
        with self.assertRaises(ValueError):
            C.best_cuts([1, 1], 0, 0, 3)
        # brute force on a few shapes
        import itertools
        for costs, fe, le, w in (([3, 9, 2, 7, 7, 1, 4, 8], 5, 2, 3), ([10, 1, 1, 1, 1, 1, 10, 3, 3], 0, 6, 4), ([2] * 9, 1, 0, 4)):
            n = len(costs)

            def stages(cuts):
                return [sum(costs[cuts[r]:cuts[r + 1]]) + (fe if r == 0 else 0) + (le if r == w - 1 else 0) for r in range(w)]
            best = min(((max(stages(c)), sum(x * x for x in stages(c)), c) for c in
                        ([0] + list(mid) + [n] for mid in itertools.combinations(range(1, n), w - 1))))
            self.assertEqual(C.best_cuts(costs, fe, le, w), best[2])

    def test_stage_plan(self):
        synthetic(self.path)
        h = C.read_header(self.path)
        p1 = C.stage_plan(h, 1)
        self.assertEqual(p1["cuts"], [0, 4])
        self.assertEqual({u["name"] for u in p1["unloaded"]}, {"blk.4.nextn.eh_proj.weight", "blk.4.nextn.enorm.weight"})
        r = p1["ranks"][0]
        self.assertEqual(r["tensors"], len(h["tensors"]) - 2)
        offs = [e[4] for e in r["entries"]]
        self.assertEqual(offs, sorted(offs))
        self.assertTrue(all(o % C.PLAN_ALIGN == 0 for o in offs))
        self.assertEqual(r["device_bytes"], sum(C.aligned(e[2]) for e in r["entries"]) + C.PLAN_TAIL)
        self.assertEqual([e[3] for e in r["entries"]], sorted(e[3] for e in r["entries"]))      # file order
        p2 = C.stage_plan(h, 2)
        names = [{e[0] for e in rk["entries"]} for rk in p2["ranks"]]
        self.assertIn("token_embd.weight", names[0])
        self.assertNotIn("token_embd.weight", names[1])
        self.assertTrue({"output.weight", "output_norm.weight"} <= names[1])
        self.assertFalse(names[0] & names[1])
        self.assertEqual(len(names[0] | names[1]), r["tensors"])
        self.assertEqual(sum(rk["tensor_bytes"] for rk in p2["ranks"]), r["tensor_bytes"])
        # hash: the same for the same inputs, different for another world
        self.assertEqual(C.plan_hash(p2), C.plan_hash(C.stage_plan(C.read_header(self.path), 2)))
        self.assertNotEqual(C.plan_hash(p1), C.plan_hash(p2))
        self.assertTrue(C.plan_text(p1).startswith("flerp-placement v1\nmode stage\nworld 1\n"))

    def test_block_range(self):
        synthetic(self.path)
        h = C.read_header(self.path)
        p = C.stage_plan(h, 1, (1, 3))
        names = {e[0] for e in p["ranks"][0]["entries"]}
        self.assertEqual({n.split(".")[1] for n in names}, {"1", "2"})
        head = {e[0] for e in C.stage_plan(h, 1, (0, 2))["ranks"][0]["entries"]}
        self.assertIn("token_embd.weight", head)
        self.assertNotIn("output.weight", head)
        tail = {e[0] for e in C.stage_plan(h, 1, (2, 4))["ranks"][0]["entries"]}
        self.assertTrue({"output.weight", "output_norm.weight"} <= tail)
        self.assertNotIn("token_embd.weight", tail)
        with self.assertRaises(ValueError):
            C.stage_plan(h, 1, (2, 5))
        with self.assertRaises(ValueError):
            C.stage_plan(h, 2, (0, 2))

    def test_tied_head(self):
        synthetic(self.path, tied=True)
        h = C.read_header(self.path)
        p = C.stage_plan(h, 2)
        self.assertTrue(p["tied_head"])
        self.assertTrue(all("token_embd.weight" in {e[0] for e in rk["entries"]} for rk in p["ranks"]))
        one = C.stage_plan(h, 1)["ranks"][0]
        self.assertEqual(sum(e[0] == "token_embd.weight" for e in one["entries"]), 1)

    def test_summary_and_cli(self):
        synthetic(self.path)
        out = os.path.join(self.dir.name, "out")
        argv, sys.argv = sys.argv, ["gguf_census.py", self.path, "--out", out, "--worlds", "2,3", "--budget-gib", "1"]
        try:
            self.assertEqual(C.main(), 0)
        finally:
            sys.argv = argv
        with open(os.path.join(out, "summary.md")) as f:
            text = f.read()
        self.assertIn("### World 3", text)
        self.assertIn("`ffn_down_exps` in blocks 1", text)
        self.assertIn("blk.4.nextn.eh_proj.weight", text)
        self.assertEqual(C.compact([0, 1, 2, 5, 7, 8]), "0-2, 5, 7-8")


if __name__ == "__main__":
    unittest.main()
