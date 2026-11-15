#!/usr/bin/env python3
"""An independent evaluation of single qwen35moe feed-forward blocks, to give the executor's CPU reference standing.

    python3 gate/moe_block_check.py --gguf FILE.gguf --gguf-py DIR --dump FILE [--max N]

The dump is what the CPU reference appends with FLERP_MOE_DUMP (forward.cpp, ForwardMoeFFN): for a block and a
position, the block's real input x, the experts and weights its router chose, the shared gate logit and the block's
output. This script shares no code with it: the GGUF is read and the K-quant blocks are dequantized by llama.cpp's
gguf-py (gguf/quants.py, numpy), and the block is computed here in float64 from the dequantized weights as
llama.cpp's graph states it (src/models/qwen35moe.cpp:494-548, src/llama-graph.cpp:2002-2310): softmax over the
router logits, the expert_used largest, weights renormalised over them, SwiGLU experts, plus sigmoid(gate) times
the shared expert. It prints one JSON object: per record what agreed, and the largest differences.
"""
import argparse
import json
import struct
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--gguf-py", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, a.gguf_py)
    import numpy as np
    from gguf import GGUFReader
    from gguf.quants import dequantize

    reader = GGUFReader(a.gguf)
    tensors = {t.name: t for t in reader.tensors}

    def weights(name, expert=None):
        t = tensors[name]
        data = t.data if expert is None else t.data[expert]
        return dequantize(data, t.tensor_type).astype(np.float64)

    silu = lambda v: v / (1.0 + np.exp(-v))  # noqa: E731
    records, raw = [], open(a.dump, "rb").read()
    off = 0
    while off < len(raw) and (not a.max or len(records) < a.max):
        layer, pos, hidden, used, experts = struct.unpack_from("<5i", raw, off)
        off += 20
        x = np.frombuffer(raw, "<f4", hidden, off).astype(np.float64); off += 4 * hidden
        ids = np.frombuffer(raw, "<i4", used, off); off += 4 * used
        w = np.frombuffer(raw, "<f4", used, off).astype(np.float64); off += 4 * used
        gate = struct.unpack_from("<f", raw, off)[0]; off += 4
        out = np.frombuffer(raw, "<f4", hidden, off).astype(np.float64); off += 4 * hidden
        p = f"blk.{layer}."
        logits = weights(p + "ffn_gate_inp.weight").reshape(experts, hidden) @ x
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        mine = np.argsort(-probs, kind="stable")[:used]
        mw = probs[mine] / max(probs[mine].sum(), 6.103515625e-5)
        y = np.zeros(hidden)
        for e, we in zip(mine, mw):
            act = silu(weights(p + "ffn_gate_exps.weight", e) @ x) * (weights(p + "ffn_up_exps.weight", e) @ x)
            y += we * (weights(p + "ffn_down_exps.weight", e) @ act)
        g = float(weights(p + "ffn_gate_inp_shexp.weight").reshape(hidden) @ x)
        act = silu(weights(p + "ffn_gate_shexp.weight") @ x) * (weights(p + "ffn_up_shexp.weight") @ x)
        y += (weights(p + "ffn_down_shexp.weight") @ act) / (1.0 + np.exp(-g))
        rms = float(np.sqrt(np.mean(y * y)))
        records.append({"layer": layer, "pos": pos, "ids_equal": [int(i) for i in mine] == [int(i) for i in ids],
                        "ids_same_set": sorted(int(i) for i in mine) == sorted(int(i) for i in ids),
                        "weights_max_abs": float(np.abs(mw - w).max()), "gate_logit_abs": abs(g - gate),
                        "out_max_abs": float(np.abs(y - out).max()), "out_rms": rms,
                        "out_max_abs_over_rms": float(np.abs(y - out).max() / rms) if rms else None,
                        "margin_to_next_expert": float(np.sort(probs)[-used] - np.sort(probs)[-used - 1])})
    n = len(records)
    print(json.dumps({"records": n, "layers": sorted({r["layer"] for r in records}),
                      "ids_equal": sum(r["ids_equal"] for r in records), "ids_same_set": sum(r["ids_same_set"] for r in records),
                      "weights_max_abs": max((r["weights_max_abs"] for r in records), default=None),
                      "gate_logit_max_abs": max((r["gate_logit_abs"] for r in records), default=None),
                      "out_max_abs": max((r["out_max_abs"] for r in records), default=None),
                      "out_max_abs_over_rms": max((r["out_max_abs_over_rms"] for r in records), default=None),
                      "detail": records}))


if __name__ == "__main__":
    main()
