"""gen_manifest.py PACK_DIR OUT_TSV: manifest_full.tsv over an EXL3 raw safetensors pack, for the kdec benches
(budget.py, cnt_test.py, test_hc_fuse.py, test_hc_chain.py): one line a tensor, name, path, byte offset, byte
length, dtype, shape, strides. Points INTO the pack (no extraction); rows are only the plain F32 / BF16 tensors
the benches load; offsets absolute (8 + header + data_offset). SPEED-FUSE-1 b7269694f8: the benches' tensors
(hc_*, *_norm.weight) live in the pack; the workers' tmpfs homes lost the old extracted copy.
"""
import json, struct, sys
from pathlib import Path

pack, out = Path(sys.argv[1]), Path(sys.argv[2])
ix = json.load(open(pack / "model.safetensors.index.json"))["weight_map"]
byshard = {}
for name, shard in ix.items():
    byshard.setdefault(shard, []).append(name)
n = 0
with open(out, "w") as w:
    for shard in sorted(byshard):
        path = (pack / shard).resolve()
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hlen))
        base = 8 + hlen
        for name in sorted(byshard[shard]):
            m = hdr.get(name)
            if m is None or m["dtype"] not in ("F32", "BF16"):
                continue
            o0, o1 = m["data_offsets"]
            w.write("\t".join([name, str(path), str(base + o0), str(o1 - o0), m["dtype"],
                              json.dumps(m.get("shape", [])), json.dumps(m.get("data_strides", []) if "data_strides" in m else [])]) + "\n")
            n += 1
print(f"manifest {out}: {n} tensors")
