"""SPEED-FUSE-1 (b7269694f8): the fused chain launch (_hc_chain, hc_chain.py) against the four launches it replaces as
the engine issues them on hc_dots' path (tri_hc.hcPre2: the kit's own _hc_post_only, _hc_finish_s role 1, _hc_mix_part
of the stored streams, _hc_sinkhorn), bit for bit, AOT against AOT on the kit through hiprt (no torch, no compiler, no
model weights): synthetic tensors of the served shapes (D 5120, 4 streams, 24 mixes, world 2), rows 1-16, 3 synthetic
sites x 3 input seeds, each case launched REPS times (the arrival order differs from launch to launch). Every word of
h_out, out, pre_out, post, comb and the partial sums' 25 written slots must be equal and the arrival counter back at
zero. The kit's variants are taken by exact selection (as TF_GPU_STRICT=1); a kernel with no exact variant falls back
to aot.zig's fits and the case records it. Then the device time a site of the two arms, host-launched back to back
(bench box numbers, not the pair's). A JIT-vs-JIT run (test_hc_chain.py) does not show what this shows: there both
arms come out of one compiler, here the four are the kit's objects.
Usage: TRI_KIT=<kit dir> python test_chain_aot.py [out.jsonl]   (ROWS=1,6,16 REPS=8 by default all of 1-16)"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "tri"))
import numpy as np

import hiprt as H
from hiprt import Buf, P

D, NB, HC, WORLD, MIX = 5120, 40, 4, 2, 24
WIDE = HC * D
EPS, HC_EPS, ITERS = 1e-20, 1e-6, 20
REPS = int(os.environ.get("REPS", "8"))
ROWS = [int(r) for r in os.environ.get("ROWS", ",".join(str(r) for r in range(1, 17))).split(",")]
GATE_ROWS = (1, 6, 16)
KIT = H.Kit(os.environ["TRI_KIT"])
out = open(sys.argv[1], "w") if len(sys.argv) > 1 else None
NAMES = ("h_out", "out", "pre_out", "post", "comb", "part")


def emit(d):
    line = json.dumps(d, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(line, flush=True)
    if out:
        out.write(line + "\n")
        out.flush()


def hc_rb(rows):  # tri_hc.hcRb
    t = [1, 2, 3, 2, 3, 2, 4, 4]
    return t[rows - 1] if 1 <= rows <= 8 else (4 if rows <= 12 else 8)


def cdiv(a, b):
    return (a + b - 1) // b


def bf16(x):
    return H.f32_to_bf16_bits(np.asarray(x, dtype=np.float32))


def site(rng):
    """a mix site's weights in the served shapes: hc fn fp32 [24, 4 D], base [24], scale [3], the norm weight bf16 [D]"""
    return {"fn": Buf.of((rng.standard_normal((MIX, WIDE)) * 0.02).astype(np.float32)),
            "base": Buf.of((rng.standard_normal(MIX) * 0.5).astype(np.float32)),
            "scale": Buf.of((rng.random(3) * 0.9 + 0.1).astype(np.float32)),
            "nw": Buf.of(bf16(1.0 + 0.1 * rng.standard_normal(D)))}


def inputs(rng, R):
    """the streams h bf16 [R, 4, D], the ranks' gathered partials fp32 [world, R, D], and the previous site's pre /
    post / comb (row-wise), as test_hc_chain.bufs draws them"""
    h = bf16(rng.standard_normal((R, HC, D)) * 0.5)
    g = (rng.standard_normal((WORLD, R, D)) * 0.25).astype(np.float32)
    pre_in = (rng.random((R, HC)) * 0.8 + 0.1).astype(np.float32)
    post = (rng.random((R, HC)) * 2.0).astype(np.float32)
    c = np.exp(rng.random((R, HC, HC)) * 4 - 2)
    comb = (c / c.sum(2, keepdims=True)).astype(np.float32)
    return h, g, pre_in, post, comb


class Arm:
    def __init__(self, R, s, h, g, pre_in, post0, comb0, cnt):
        self.R, self.post0, self.comb0 = R, post0, comb0
        self.h_out, self.out = Buf(R * WIDE * 2), Buf(R * D * 2)
        self.pre_out, self.post, self.comb, self.part = Buf(R * HC * 4), Buf(R * HC * 4), Buf(R * 16 * 4), Buf(R * NB * 32 * 4)
        rb = hc_rb(R)
        ho, o = P(self.h_out, "*bf16"), P(self.out, "*bf16")
        post, comb, part = P(self.post, "*fp32"), P(self.comb, "*fp32"), P(self.part, "*fp32")
        fin = dict(PART=part, BASE=P(s["base"], "*fp32"), SCALE=P(s["scale"], "*fp32"), PRE_IN=P(pre_in, "*fp32"),
                   NW=P(s["nw"], "*bf16"), OUT=o, PRE_OUT=P(self.pre_out, "*fp32"), POST=post, COMB=comb, eps=EPS,
                   hc_eps=HC_EPS)
        # tri_hc.hcPre2's hc_dots path: the post alone and the finish on the round's stream (PDL), the mixes of the
        # stored streams and the Sinkhorn on the side stream (no PDL)
        four = [
            ("_hc_post_only", (R, NB, 1), dict(X=P(h, "*bf16"), XO=ho, G=P(g, "*fp32"), RS=R * D, POST=post, COMB=comb),
             dict(WIDE=WIDE, D=D, KB=WIDE // NB, WORLD=WORLD, PDL=1)),
            ("_hc_finish_s", (R, 1, 1), dict(fin, X=ho, S0=o, H0=o, S1=o, H1=o, S2=o, H2=o, S3=o, H3=o),
             dict(D=D, NB=NB, ITERS=ITERS, BLOCK=1024, NROT=0, RSPLIT=1, SINK=0, PDL=1)),
            ("_hc_mix_part", (cdiv(R, rb), NB, 3), dict(X=ho, XO=ho, G=ho, RS=0, POST=post, COMB=comb,
                                                        FN=P(s["fn"], "*fp32"), PART=part, rows=R),
             dict(WIDE=WIDE, D=D, NB=NB, SUB=128, MB=8, RB=rb, PDL=0, WORLD=1, POSTED=0, XPF=0)),
            ("_hc_sinkhorn", (R, 1, 1), dict(PART=part, BASE=fin["BASE"], SCALE=fin["SCALE"], PRE_OUT=fin["PRE_OUT"],
                                             POST=post, COMB=comb, eps=EPS, hc_eps=HC_EPS),
             dict(D=D, NB=NB, ITERS=ITERS)),
        ]
        one = [("_hc_chain", (cdiv(R, rb), NB + 1, 3),
                dict(X=P(h, "*bf16"), XO=ho, G=P(g, "*fp32"), RS=R * D, POST=post, COMB=comb, FN=P(s["fn"], "*fp32"),
                     PART=part, CNT=P(cnt, "*i32"), BASE=fin["BASE"], SCALE=fin["SCALE"], PRE_IN=fin["PRE_IN"],
                     NW=fin["NW"], OUT=o, PRE_OUT=fin["PRE_OUT"], eps=EPS, hc_eps=HC_EPS, rows=R),
                dict(WIDE=WIDE, D=D, NB=NB, SUB=128, MB=8, RB=rb, ITERS=ITERS, BLOCK=1024, WORLD=WORLD))]
        self.four, self.exact4 = self.plan(four)
        self.one, self.exact1 = self.plan(one)

    @staticmethod
    def plan(launches):
        res, exact = [], True
        for fn, grid, args, consts in launches:
            try:
                v = KIT.find(fn, args, consts, exact_only=True)
            except KeyError:
                v, exact = KIT.find(fn, args, consts), False
            res.append((fn, grid, args, v))
        return res, exact

    def reset(self):
        self.post.upload(self.post0)
        self.comb.upload(self.comb0)
        for b in (self.h_out, self.out, self.pre_out):
            b.fill(0x5A)
        self.part.fill(0)

    def run(self, fused):
        for fn, grid, args, v in (self.one if fused else self.four):
            KIT.run(fn, grid, args, {}, variant=v)

    def get(self):
        R = self.R
        return (self.h_out.download((R, HC, D), np.uint16), self.out.download((R, D), np.uint16),
                self.pre_out.download((R, HC), np.uint32), self.post.download((R, HC), np.uint32),
                self.comb.download((R, 16), np.uint32),
                np.ascontiguousarray(self.part.download((R, NB, 32), np.uint32)[:, :, :25]))


def finite(vals):
    """the fused arm's results are numbers (a NaN would compare equal bit for bit and prove nothing)"""
    ho, o, pre, post, comb, part = vals
    bad = int(((ho & 0x7F80) == 0x7F80).sum() + ((o & 0x7F80) == 0x7F80).sum())
    for v in (pre, post, comb, part):
        bad += int((~np.isfinite(v.view(np.float32))).sum())
    return bad


def bits():
    cnt = Buf(64 * 4)
    cnt.fill(0)
    cases = n_ok = 0
    words = {n: 0 for n in NAMES}
    by_rows = {}
    hashes = set()
    for si in range(3):
        rng = np.random.default_rng(72696 + si)
        s = site(rng)
        for R in ROWS:
            for seed in range(3):
                h, g, pre_in, post0, comb0 = inputs(rng, R)
                hb, gb, pb = Buf.of(h), Buf.of(g), Buf.of(pre_in)
                a = Arm(R, s, hb, gb, pb, post0, comb0, cnt)
                f = Arm(R, s, hb, gb, pb, post0, comb0, cnt)
                hashes.add(f.one[0][3]["hash"][:12])
                diffs = {n: 0 for n in NAMES}
                cnt_left = nonfinite = 0
                for _ in range(REPS):
                    a.reset()
                    f.reset()
                    a.run(False)
                    f.run(True)
                    H.sync()
                    fv = f.get()
                    for nm, u, v in zip(NAMES, a.get(), fv):
                        diffs[nm] += int((u != v).sum())
                    cnt_left += int(np.abs(cnt.download((64,), np.int32)).sum())
                    nonfinite += finite(fv)
                    if cnt_left:
                        cnt.fill(0)
                ok = all(d == 0 for d in diffs.values()) and cnt_left == 0 and nonfinite == 0
                cases += 1
                n_ok += ok
                for nm in NAMES:
                    words[nm] += diffs[nm]
                br = by_rows.setdefault(R, {"cases": 0, "bit_identical": 0, "differing_words": 0})
                br["cases"] += 1
                br["bit_identical"] += ok
                br["differing_words"] += sum(diffs.values())
                emit({"site": si, "rows": R, "seed": seed, "reps": REPS, "diffs": diffs, "cnt_left": cnt_left,
                      "nonfinite": nonfinite, "exact_variants": {"four": a.exact4, "fused": f.exact1},
                      "fused_hash": f.one[0][3]["hash"][:12], "ok": ok})
    gate = {str(r): by_rows.get(r) for r in GATE_ROWS}
    emit({"summary": "bits", "cases": cases, "bit_identical": n_ok, "reps": REPS, "differing_words": words,
          "differing_words_total": sum(words.values()), "rows_1_6_16": gate, "by_rows": by_rows,
          "fused_variants_used": sorted(hashes), "ok": n_ok == cases})
    return n_ok == cases


def timing():
    cnt = Buf(64 * 4)
    cnt.fill(0)
    rng = np.random.default_rng(1151)
    s = site(rng)
    for R in GATE_ROWS:
        h, g, pre_in, post0, comb0 = inputs(rng, R)
        arm = Arm(R, s, Buf.of(h), Buf.of(g), Buf.of(pre_in), post0, comb0, cnt)
        t = {"four": [], "fused": []}
        for _ in range(5):  # interleaved batches
            arm.reset()
            t["four"].append(H.time_ms(lambda: arm.run(False), iters=400, warmup=20) * 1e3)
            arm.reset()
            t["fused"].append(H.time_ms(lambda: arm.run(True), iters=400, warmup=20) * 1e3)
        emit({"timing_us_per_site": R, "launches": {"four": 4, "fused": 1},
              "four_us": [round(v, 2) for v in t["four"]], "fused_us": [round(v, 2) for v in t["fused"]],
              "four_median": round(float(np.median(t["four"])), 2), "fused_median": round(float(np.median(t["fused"])), 2),
              "note": "host-launched back to back on one stream on the bench box (python launch overhead in both "
                      "arms), not a graph and not the pair"})


if __name__ == "__main__":
    emit({"device": H.device_name(), "kit": str(KIT.dir), "kit_triton": KIT.spec.get("triton"),
          "kdec2_chain_added": KIT.spec.get("kdec2_chain_added")})
    good = bits()
    if os.environ.get("TIMING", "1") == "1":
        timing()
    H.sync()
    sys.exit(0 if good else 1)
