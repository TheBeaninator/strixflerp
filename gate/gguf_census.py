#!/usr/bin/env python3
"""Header census of a GGUF file and the placement plan of format.flerp.gguf_placed (docs/PHASE5-FOURNODE-DESIGN.md
sections 3.1 and 3.2, plan row P0). Standard library only. Reads the header and nothing after it: the file is read
forward, unbuffered, by exactly the bytes each field needs, up to the last tensor record; never the data region.

    python3 gate/gguf_census.py FILE.gguf [--out DIR] [--worlds 3,4] [--budget-gib 110] [--context 32768]
    python3 gate/gguf_census.py FILE.gguf --plan --world W [--range B0:B1]      the plan text and its sha256

The placement is a pure function of (header, world, mode): `stage_plan`. Its text form (`plan_text`) is what the
loader hashes; the C++ loader (flerp/gguf_placed/placement.cpp) writes the same text and the gates compare the hashes.
"""
import argparse
import hashlib
import json
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summary as SM  # noqa: E402  (census.json / summary.md, config_sha256)

GIB = float(1 << 30)
# ggml type id -> (name, values a block, bytes a block): ggml.c type traits at llama.cpp 86a24a1.
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
}
SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
PLAN_ALIGN = 64          # a tensor's offset in the device allocation is a multiple of this
PLAN_TAIL = 4096         # bytes after the last tensor of an allocation
EXPERT_STACKS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")


class Forward:
    """A file read forward only, by exactly the bytes asked for (unbuffered: nothing past the header is requested);
    the header hash is over exactly those bytes."""

    def __init__(self, f):
        self.f, self.n, self.sha = f, 0, hashlib.sha256()

    def take(self, n):
        out = b""
        while len(out) < n:
            more = self.f.read(n - len(out))
            if not more:
                raise ValueError("the file ends inside its header")
            out += more
        self.n += n
        self.sha.update(out)
        return out

    def tell(self):
        return self.n

    def bytes_read(self):
        return self.n

    def unpack(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self):
        return self.take(self.unpack("<Q")).decode("utf-8", "replace")


def read_value(r, vtype, keep_arrays):
    if vtype in SCALARS:
        return r.unpack(SCALARS[vtype])
    if vtype == 8:
        return r.string()
    if vtype == 9:
        etype, n = r.unpack("<I"), r.unpack("<Q")
        if etype in SCALARS and not keep_arrays:
            r.take(n * struct.calcsize(SCALARS[etype]))
            return {"array": n, "of": etype}
        items = [read_value(r, etype, keep_arrays) for _ in range(n)]
        return items if keep_arrays else {"array": n, "of": etype}
    raise ValueError(f"unknown GGUF value type {vtype}")


def tensor_bytes(type_id, dims):
    """Encoded bytes of a tensor, or None for a type this table lacks or a count that is not whole blocks."""
    if type_id not in GGML_TYPES:
        return None
    _, per, size = GGML_TYPES[type_id]
    n = 1
    for d in dims:
        n *= d
    return None if n % per else n // per * size


def read_header(path, keep_arrays=False):
    """{"version", "metadata": {key: value}, "tensors": [{name, type, type_id, dims, bytes, offset}], "data_start",
    "alignment", "file_bytes", "header_sha256", "header_bytes_read"}. `offset` is absolute in the file."""
    with open(path, "rb", buffering=0) as f:
        size = os.fstat(f.fileno()).st_size
        r = Forward(f)
        if r.take(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version, n_tensors, n_kv = r.unpack("<I"), r.unpack("<Q"), r.unpack("<Q")
        meta = {}
        for _ in range(n_kv):
            key = r.string()
            meta[key] = read_value(r, r.unpack("<I"), keep_arrays)
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.unpack("<Q") for _ in range(r.unpack("<I"))]
            type_id, rel = r.unpack("<I"), r.unpack("<Q")
            tensors.append({"name": name, "type_id": type_id, "type": GGML_TYPES.get(type_id, (f"type{type_id}",))[0],
                            "dims": dims, "bytes": tensor_bytes(type_id, dims), "offset": rel})
        end = r.tell()
        align = meta.get("general.alignment", 32)
        start = (end + align - 1) // align * align
        for t in tensors:
            t["offset"] += start
        return {"version": version, "metadata": meta, "tensors": tensors, "data_start": start, "alignment": align,
                "file_bytes": size, "header_sha256": r.sha.hexdigest(), "header_end": end,
                "header_bytes_read": r.bytes_read()}


def check_layout(h):
    """Problems with the tensor table: an unknown type, a tensor past the end of the file, overlapping tensors."""
    bad, end = [], h["data_start"]
    for t in sorted(h["tensors"], key=lambda t: t["offset"]):
        if t["bytes"] is None:
            bad.append(f"{t['name']}: type {t['type']} with dims {t['dims']} has no known size")
            continue
        if t["offset"] < end:
            bad.append(f"{t['name']}: starts at {t['offset']}, before the end of the tensor ahead of it ({end})")
        end = t["offset"] + t["bytes"]
    if end > h["file_bytes"]:
        bad.append(f"the last tensor ends at {end}, the file has {h['file_bytes']} bytes")
    return bad


def model_facts(h):
    m = h["metadata"]
    arch = m.get("general.architecture", "")
    g = lambda k, d=None: m.get(f"{arch}.{k}", d)  # noqa: E731
    blocks, nextn = g("block_count", 0), g("nextn_predict_layers", 0)
    return {"architecture": arch, "block_count": blocks, "nextn_predict_layers": nextn, "blocks": blocks - nextn,
            "hidden": g("embedding_length"), "heads": g("attention.head_count"), "kv_heads": g("attention.head_count_kv"),
            "head_dim": g("attention.key_length"), "context_length": g("context_length"),
            "expert_count": g("expert_count"), "expert_used_count": g("expert_used_count"),
            "expert_feed_forward_length": g("expert_feed_forward_length"),
            "expert_shared_feed_forward_length": g("expert_shared_feed_forward_length"),
            "feed_forward_length": g("feed_forward_length"), "full_attention_interval": g("full_attention_interval"),
            "ssm_inner_size": g("ssm.inner_size"), "ssm_state_size": g("ssm.state_size"),
            "ssm_conv_kernel": g("ssm.conv_kernel"), "ssm_time_step_rank": g("ssm.time_step_rank"),
            "ssm_group_count": g("ssm.group_count"),
            "vocab": (m.get("tokenizer.ggml.tokens") or {}).get("array") if isinstance(m.get("tokenizer.ggml.tokens"), dict)
            else (len(m["tokenizer.ggml.tokens"]) if "tokenizer.ggml.tokens" in m else None)}


BLK = re.compile(r"blk\.(\d+)\.(.+)$")


def classify(name, blocks):
    """Where a tensor belongs: ("block", i) for a block the executor runs, ("embd",), ("head",), ("nextn", i) for a
    block past the stack (the MTP head), ("other",)."""
    m = BLK.match(name)
    if m:
        i = int(m.group(1))
        return ("block", i) if i < blocks else ("nextn", i)
    if name == "token_embd.weight":
        return ("embd",)
    if name in ("output_norm.weight", "output.weight"):
        return ("head",)
    return ("other",)


def is_expert(name):
    m = BLK.match(name)
    return bool(m) and m.group(2).split(".")[0] in EXPERT_STACKS


def aligned(n):
    return (n + PLAN_ALIGN - 1) // PLAN_ALIGN * PLAN_ALIGN


def best_cuts(block_cost, first_extra, last_extra, world):
    """Cuts [b_0 = 0, b_1, .., b_world = n] of n blocks into `world` contiguous non-empty stages. Stage r costs the
    sum of its blocks, plus first_extra on rank 0 and last_extra on the last rank. Minimises the largest stage;
    among those the sum of squared stage costs; among those the cut list that is smallest in order (all integers,
    so the C++ loader finds the same cuts)."""
    n = len(block_cost)
    if world < 1 or world > n:
        raise ValueError(f"world {world} does not fit {n} blocks")
    pre = [0]
    for c in block_cost:
        pre.append(pre[-1] + c)

    def cost(r, a, b):
        return pre[b] - pre[a] + (first_extra if r == 0 else 0) + (last_extra if r == world - 1 else 0)

    inf = float("inf")
    # 1. the smallest possible largest stage
    best = [[inf] * (n + 1) for _ in range(world + 1)]
    best[0][0] = 0
    for r in range(world):
        for a in range(n):
            if best[r][a] == inf:
                continue
            for b in range(a + 1, n + 1):
                v = max(best[r][a], cost(r, a, b))
                if v < best[r + 1][b]:
                    best[r + 1][b] = v
    limit = best[world][n]
    # 2. under that limit, the smallest sum of squares; ties go to the earliest cut (strict < keeps the first a)
    sq = [[None] * (n + 1) for _ in range(world + 1)]
    back = [[None] * (n + 1) for _ in range(world + 1)]
    sq[world][n] = 0
    for r in range(world - 1, -1, -1):          # from the last stage back, so ties resolve on the earliest cuts first
        for a in range(n - 1, -1, -1):
            for b in range(a + 1, n + 1):
                if sq[r + 1][b] is None:
                    continue
                c = cost(r, a, b)
                if c > limit:
                    continue
                v = sq[r + 1][b] + c * c
                if sq[r][a] is None or v < sq[r][a]:
                    sq[r][a], back[r][a] = v, b
    cuts, a = [0], 0
    for r in range(world):
        a = back[r][a]
        cuts.append(a)
    return cuts


def stage_plan(h, world, block_range=None):
    """Mode `stage`. Returns {"mode", "world", "blocks", "cuts", "ranks": [{rank, blocks: [b0, b1), tensors,
    tensor_bytes, device_bytes, entries: [(name, type_id, bytes, file offset, device offset)]}], "unloaded": [...]}.

    With `block_range` (b0, b1) and world 1 the one rank holds only those blocks, token_embd if b0 == 0 and the
    head tensors if b1 is the last block: the single-GPU gates of a range. A tied head (no output.weight) puts
    token_embd on the last rank as well."""
    facts = model_facts(h)
    n = facts["blocks"]
    by = {"embd": [], "head": [], "nextn": [], "other": [], "block": [[] for _ in range(n)]}
    for t in h["tensors"]:                      # file order
        c = classify(t["name"], n)
        (by["block"][c[1]] if c[0] == "block" else by[c[0]]).append(t)
    tied = not any(t["name"] == "output.weight" for t in by["head"])
    cost = lambda ts: sum(aligned(t["bytes"]) for t in ts)  # noqa: E731
    if block_range is not None:
        if world != 1:
            raise ValueError("a block range is a world-1 plan")
        b0, b1 = block_range
        if not 0 <= b0 < b1 <= n:
            raise ValueError(f"block range {b0}:{b1} is outside 0:{n}")
        cuts = [b0, b1]
    else:
        cuts = best_cuts([cost(b) for b in by["block"]], cost(by["embd"]),
                         cost(by["head"]) + (cost(by["embd"]) if tied and world > 1 else 0), world)
    ranks = []
    for r in range(world):
        b0, b1 = cuts[r], cuts[r + 1]
        held = []
        if b0 == 0:
            held += by["embd"]
        for b in range(b0, b1):
            held += by["block"][b]
        if b1 == n:
            held += by["head"]
            if tied and b0 != 0:
                held += by["embd"]
        held.sort(key=lambda t: t["offset"])
        entries, off = [], 0
        for t in held:
            entries.append((t["name"], t["type_id"], t["bytes"], t["offset"], off))
            off += aligned(t["bytes"])
        ranks.append({"rank": r, "blocks": [b0, b1], "tensors": len(held), "tensor_bytes": sum(t["bytes"] for t in held),
                      "device_bytes": off + PLAN_TAIL, "entries": entries})
    loaded = {e[0] for rk in ranks for e in rk["entries"]}
    return {"mode": "stage", "world": world, "blocks": n, "cuts": cuts, "ranks": ranks, "tied_head": tied,
            "file_bytes": h["file_bytes"], "range": list(block_range) if block_range else None,
            "unloaded": [{"name": t["name"], "bytes": t["bytes"]} for t in h["tensors"] if t["name"] not in loaded]}


def plan_text(plan):
    """The canonical text of a plan: what every rank hashes and compares in the handshake."""
    L = ["flerp-placement v1", f"mode {plan['mode']}", f"world {plan['world']}", f"file_bytes {plan['file_bytes']}",
         f"blocks {plan['blocks']}", "cuts " + " ".join(str(c) for c in plan["cuts"])]
    for rk in plan["ranks"]:
        L.append(f"rank {rk['rank']} blocks {rk['blocks'][0]} {rk['blocks'][1]} tensors {rk['tensors']} device_bytes {rk['device_bytes']}")
        L += [f"{name}\t{tid}\t{size}\t{foff}\t{doff}" for name, tid, size, foff, doff in rk["entries"]]
    return "\n".join(L) + "\n"


def plan_hash(plan):
    return hashlib.sha256(plan_text(plan).encode()).hexdigest()


def runtime_estimate(facts, context, chunk_rows=4096):
    """Bytes a rank needs beside its weights, first version of the design (3.7): every stage allocates the origin's
    whole-model arena. An estimate from the header, not a measurement: attention K/V in F16 for every attention
    block, the DeltaNet state of every other block, three MoE scratch buffers and the hidden rows of a chunk."""
    hidden, n = facts["hidden"] or 0, facts["blocks"]
    interval = facts["full_attention_interval"] or 4
    attn = n // interval
    kv = attn * 2 * (facts["kv_heads"] or 0) * (facts["head_dim"] or 0) * 2 * context
    E, k = facts["expert_count"] or 0, facts["expert_used_count"] or 0
    F, S = facts["expert_feed_forward_length"] or 0, facts["expert_shared_feed_forward_length"] or 0
    scratch_floats = max(k * hidden, E + 1 + 2 * k + hidden, k * F + S) if E else (facts["feed_forward_length"] or 0)
    scratch = 3 * chunk_rows * scratch_floats * 4
    # DeltaNet: a [heads, state, state]-like recurrent state and a conv window a block; small beside the rest
    inner, state = facts["ssm_inner_size"] or 0, facts["ssm_state_size"] or 0
    ssm = (n - attn) * (inner * state * 4 + 3 * (inner + 2 * hidden) * 4 * (facts["ssm_conv_kernel"] or 4))
    rows = 8 * chunk_rows * max(hidden, inner + 2 * hidden) * 4       # hidden, residual, norm and projection rows of a chunk
    return {"context": context, "chunk_rows": chunk_rows, "kv_bytes": kv, "moe_scratch_bytes": scratch,
            "moe_scratch_floats_per_token": scratch_floats, "deltanet_state_bytes": ssm, "chunk_rows_bytes": rows,
            "total_bytes": kv + scratch + ssm + rows}


def census(h):
    facts = model_facts(h)
    n = facts["blocks"]
    by_type, per_block, q6_stacks, other = {}, [dict(block=i, dense_bytes=0, expert_bytes=0, tensors=0, expert_types={}) for i in range(n)], [], []
    nextn, groups = [], {"embd": 0, "head": 0, "nextn": 0, "other": 0, "block_dense": 0, "block_expert": 0}
    for t in h["tensors"]:
        e = by_type.setdefault(t["type"], {"tensors": 0, "bytes": 0})
        e["tensors"] += 1
        e["bytes"] += t["bytes"] or 0
        c = classify(t["name"], n)
        if c[0] == "block":
            b = per_block[c[1]]
            b["tensors"] += 1
            if is_expert(t["name"]):
                b["expert_bytes"] += t["bytes"]
                b["expert_types"][BLK.match(t["name"]).group(2).split(".")[0]] = t["type"]
                groups["block_expert"] += t["bytes"]
            else:
                b["dense_bytes"] += t["bytes"]
                groups["block_dense"] += t["bytes"]
        else:
            groups[c[0]] += t["bytes"] or 0
            if c[0] == "nextn":
                nextn.append({"name": t["name"], "type": t["type"], "dims": t["dims"], "bytes": t["bytes"]})
            elif c[0] == "other":
                other.append(t["name"])
    for t in h["tensors"]:
        if t["type"] == "Q6_K" and is_expert(t["name"]):
            q6_stacks.append(t["name"])
    q6_all = [t["name"] for t in h["tensors"] if t["type"] == "Q6_K"]
    return {"facts": facts, "tensors": len(h["tensors"]), "by_type": by_type, "groups": groups, "per_block": per_block,
            "q6_expert_stacks": q6_stacks, "q6_tensors": q6_all, "past_the_stack": nextn, "other_tensors": other,
            "tensor_bytes": sum(t["bytes"] or 0 for t in h["tensors"]), "layout_problems": check_layout(h)}


def gib(n):
    return f"{n / GIB:.2f}"


def write_summary(path, name, h, c, plans, est, budget):
    f, L = c["facts"], []
    L += [f"# GGUF header census: {name}", "",
          f"`{h['path']}`: {h['file_bytes']} bytes ({gib(h['file_bytes'])} GiB), GGUF v{h['version']}. Read: the header only, "
          f"{h['header_bytes_read']} bytes from the start of the file (the data region starts at {h['data_start']}); sha256 of the "
          f"header's {h['header_end']} bytes `{h['header_sha256']}`. Tool: `gate/gguf_census.py`. Every tensor's name, type, dims, "
          "bytes and offset is in `census.json`.", "",
          f"`{f['architecture']}`: {f['block_count']} blocks in the header, {f['nextn_predict_layers']} of them past the stack "
          f"(`nextn_predict_layers`), so {f['blocks']} run; hidden {f['hidden']}, {f['heads']} heads, {f['kv_heads']} KV heads of "
          f"{f['head_dim']}; {f['expert_count']} experts, {f['expert_used_count']} used, expert width {f['expert_feed_forward_length']}, "
          f"shared expert width {f['expert_shared_feed_forward_length']}; vocabulary {f['vocab']}; context {f['context_length']}.",
          "", "## Totals by type", "", "| type | tensors | bytes | GiB |", "|---|---|---|---|"]
    for k, v in sorted(c["by_type"].items(), key=lambda kv: -kv[1]["bytes"]):
        L.append(f"| {k} | {v['tensors']} | {v['bytes']} | {gib(v['bytes'])} |")
    L.append(f"| all | {c['tensors']} | {c['tensor_bytes']} | {gib(c['tensor_bytes'])} |")
    g = c["groups"]
    L += ["", "## Where the bytes are", "", "| part | GiB |", "|---|---|",
          f"| expert stacks of the {f['blocks']} blocks | {gib(g['block_expert'])} |",
          f"| everything else in those blocks (dense) | {gib(g['block_dense'])} |",
          f"| `token_embd` | {gib(g['embd'])} |", f"| `output_norm`, `output` | {gib(g['head'])} |",
          f"| blocks past the stack (not loaded) | {gib(g['nextn'])} |", f"| other tensors | {gib(g['other'])} |"]
    q6 = c["q6_expert_stacks"]
    kinds = {}
    for nme in q6:
        kinds.setdefault(BLK.match(nme).group(2).split(".")[0], []).append(int(BLK.match(nme).group(1)))
    L += ["", "## Q6_K", "", f"{len(c['q6_tensors'])} Q6_K tensors, {len(q6)} of them expert stacks: "
          + ("; ".join(f"`{k}` in blocks {compact(v)}" for k, v in sorted(kinds.items())) or "none") + ".",
          "The other Q6_K tensors: " + summarise_names([x for x in c["q6_tensors"] if x not in q6]) + "."]
    L += ["", "## Past the stack", ""]
    if c["past_the_stack"]:
        L += ["| tensor | type | dims | bytes |", "|---|---|---|---|"]
        L += [f"| `{t['name']}` | {t['type']} | {t['dims']} | {t['bytes']} |" for t in c["past_the_stack"]]
    else:
        L.append("Nothing.")
    if c["other_tensors"]:
        L += ["", "Tensors that belong to no block, embedding or head (not loaded): " + ", ".join(f"`{x}`" for x in c["other_tensors"]) + "."]
    L += ["", "## Dense and expert bytes per block", "", "| block | kind | tensors | dense GiB | expert GiB | gate / up / down stacks |", "|---|---|---|---|---|---|"]
    interval = f["full_attention_interval"] or 4
    for b in c["per_block"]:
        et = b["expert_types"]
        L.append(f"| {b['block']} | {'attention' if (b['block'] + 1) % interval == 0 else 'DeltaNet'} | {b['tensors']} | {b['dense_bytes'] / GIB:.3f} | "
                 f"{b['expert_bytes'] / GIB:.3f} | {et.get('ffn_gate_exps', '-')} / {et.get('ffn_up_exps', '-')} / {et.get('ffn_down_exps', '-')} |")
    L += ["", "## Placement, mode `stage`", "",
          "Contiguous block ranges that minimise the largest stage by bytes (then the sum of squares, then the earliest cuts); "
          "`token_embd` on rank 0, `output_norm` and `output` on the last rank, blocks past the stack unloaded. Device bytes are the "
          f"tensors each padded to {PLAN_ALIGN} bytes plus {PLAN_TAIL}. Beside the weights, the first version allocates the origin's whole-model "
          f"arena on every rank: estimated from the header at context {est['context']} and {est['chunk_rows']}-row chunks as K/V "
          f"{gib(est['kv_bytes'])} GiB, MoE scratch {gib(est['moe_scratch_bytes'])} GiB ({est['moe_scratch_floats_per_token']} floats a token "
          f"in each of three buffers), DeltaNet state {gib(est['deltanet_state_bytes'])} GiB, chunk rows {gib(est['chunk_rows_bytes'])} GiB: "
          f"**{gib(est['total_bytes'])} GiB** a rank. That is an estimate; the arena was not run at these sizes for it. Budget {budget} GiB a node."]
    for p in plans:
        worst = max(r["device_bytes"] for r in p["ranks"])
        L += ["", f"### World {p['world']}", "", f"Cuts {p['cuts']}; plan sha256 `{plan_hash(p)}`.", "",
              "| rank | blocks | tensors | weights GiB | + arena estimate GiB | of the budget | under it |", "|---|---|---|---|---|---|---|"]
        for r in p["ranks"]:
            tot = r["device_bytes"] + est["total_bytes"]
            L.append(f"| {r['rank']} | {r['blocks'][0]}..{r['blocks'][1] - 1} ({r['blocks'][1] - r['blocks'][0]}) | {r['tensors']} | {gib(r['device_bytes'])} | "
                     f"{gib(tot)} | {100 * tot / (budget * GIB):.0f}% | {'yes' if tot <= budget * GIB else 'NO'} |")
        L.append("")
        L.append(f"Largest stage {gib(worst)} GiB of weights; not loaded: {len(p['unloaded'])} tensors, {gib(sum(u['bytes'] or 0 for u in p['unloaded']))} GiB.")
    if c["layout_problems"]:
        L += ["", "## Problems in the tensor table", ""] + [f"- {x}" for x in c["layout_problems"]]
    SM.write_md(os.path.dirname(path), L, name=os.path.basename(path))


def compact(nums):
    """[0, 1, 2, 5, 7, 8] -> "0-2, 5, 7-8"."""
    nums, out, i = sorted(nums), [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ", ".join(out)


def summarise_names(names):
    kinds = {}
    for nme in names:
        m = BLK.match(nme)
        kinds.setdefault(m.group(2) if m else nme, []).append(int(m.group(1)) if m else None)
    return "; ".join(f"`{k}`" + (f" in blocks {compact(v)}" if v[0] is not None else "") for k, v in sorted(kinds.items())) or "none"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("gguf")
    ap.add_argument("--out", help="directory for census.json and summary.md")
    ap.add_argument("--name", help="title of the summary; default the file name")
    ap.add_argument("--worlds", default="3,4")
    ap.add_argument("--budget-gib", type=float, default=110.0)
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--plan", action="store_true", help="print one plan's text and hash (with --world, --range)")
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--range", help="B0:B1, blocks [B0, B1) at world 1")
    ap.add_argument("--json", action="store_true", help="with --plan: a JSON object (hash, device bytes) instead of the text")
    a = ap.parse_args()
    h = read_header(a.gguf)
    h["path"] = a.gguf
    if a.plan:
        rng = tuple(int(x) for x in a.range.split(":")) if a.range else None
        p = stage_plan(h, a.world, rng)
        if a.json:
            print(json.dumps({"sha256": plan_hash(p), "world": p["world"], "cuts": p["cuts"], "header_sha256": h["header_sha256"],
                              "ranks": [{k: r[k] for k in ("rank", "blocks", "tensors", "tensor_bytes", "device_bytes")} for r in p["ranks"]],
                              "unloaded": len(p["unloaded"])}))
        else:
            sys.stdout.write(plan_text(p))
            print("sha256", plan_hash(p), file=sys.stderr)
        return 0
    c = census(h)
    plans = [stage_plan(h, int(w)) for w in a.worlds.split(",") if w]
    est = runtime_estimate(c["facts"], a.context)
    doc = {"file": a.gguf, "file_bytes": h["file_bytes"], "gguf_version": h["version"], "alignment": h["alignment"],
           "data_start": h["data_start"], "header_end": h["header_end"], "header_sha256": h["header_sha256"],
           "header_bytes_read": h["header_bytes_read"],
           "metadata": {k: v for k, v in h["metadata"].items() if not isinstance(v, (dict, list)) and not (isinstance(v, str) and len(v) > 400)},
           "metadata_arrays": {k: v for k, v in h["metadata"].items() if isinstance(v, dict)},
           "census": c, "runtime_estimate": est, "budget_gib": a.budget_gib,
           "placement": [{"world": p["world"], "cuts": p["cuts"], "sha256": plan_hash(p), "unloaded": p["unloaded"],
                          "ranks": [{k: r[k] for k in ("rank", "blocks", "tensors", "tensor_bytes", "device_bytes")} for r in p["ranks"]]} for p in plans],
           "tensors": h["tensors"]}
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        SM.write_json(a.out, doc, name="census.json")       # no gate TOML: config_sha256 is null
        write_summary(os.path.join(a.out, "summary.md"), a.name or os.path.basename(a.gguf), h, c, plans, est, a.budget_gib)
        print(f"wrote {a.out}/census.json and summary.md; header bytes read {h['header_bytes_read']}")
    else:
        json.dump({k: doc[k] for k in ("file_bytes", "header_sha256", "census", "placement")}, sys.stdout, indent=1)
        print()
    return 1 if c["layout_problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
