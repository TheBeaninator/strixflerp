"""aux.sparse_attn's 16-head build (tf_ds_aux_sparse_attn_h16_d512, a TP4 rank's heads) on every recorded
_sparse_attn_part case, against its 32-head build (tf_ds_aux_sparse_attn_h32_d512, a TP2 rank's):
  - bits: on the same inputs, the 16-head build on heads 0-15 and on heads 16-31 (q and sink sliced) gives exactly the
    32-head build's bits for those heads: one-pass outputs (bf16), or split partials m, l, acc (fp32, compared before
    any merge) in the 16-head partial layout;
  - the 16-head output (split mode merged by the 16-head Triton _sparse_attn_merge) against the float64 reference with
    t_attn's contract (bound + normalised L2, a key-dropped negative control), as t_hipattn.py does at 32 heads;
  - the Triton path at 16 heads (_sparse_attn_part H=16, the kit's exact variant, else a test-only one): against the
    same reference, and its distance to the HIP output (reported);
  - row invariance at 16 heads (a row alone vs inside the window);
  - the decode path's Triton merges (_sparse_attn_merge, _sparse_attn_merge_rot2) at 16 heads on the 16-head
    partials vs at 32 heads on the 32-head partials, bits a head (reported: they are not this kernel).
TRI_KIT: a kit with the H=16 Triton variants (TP4-KIT 4d4c5173d2's gfx1151-v6f2c38-tp4)."""

import copy
import sys

import numpy as np

import common as C
import hiprt as H
import t_attn as TA
import t_hipattn as T
from hiprt import Buf, P

L = T.L
FN32 = T.FN
FN16 = H.Module(open(T.OBJ, "rb").read()).function("tf_ds_aux_sparse_attn_h16_d512")
HD = 512


def run_hip(fn, d, splits, keep):
    """t_hipattn.hip_attn with the kernel fn."""
    old = T.FN
    T.FN = fn
    try:
        return T.hip_attn(d, splits, keep)
    finally:
        T.FN = old


def half(d, g):
    """d's heads 16 g .. 16 g + 15 alone (the TP4 rank that holds them)."""
    e = dict(d)
    e["Hh"] = 16
    e["q"] = np.ascontiguousarray(d["q"][:, 16 * g:16 * g + 16])
    e["sink"] = np.ascontiguousarray(d["sink"][16 * g:16 * g + 16])
    e["k"] = dict(d["k"], H=16)
    return e


def merge16(d, parts, splits, out):
    mk = TA.consts_of(L["_sparse_attn_merge"][0])
    consts = dict(mk, H=16, SPLITS=splits)
    args = dict(PM=P(parts[0], "*fp32"), PL=P(parts[1], "*fp32"), PO=P(parts[2], "*fp32"),
                SINK=P(Buf.of(d["sink"]), "*fp32"), OUT=P(out, "*bf16"))
    src = run_any("_sparse_attn_merge", L["_sparse_attn_merge"][0], consts, (d["rows"], 1, 1), args)
    H.sync()
    return src


def triton16(d, c, splits):
    """_sparse_attn_part at 16 heads (as a TP4 rank with TF_DS_AUX=triton launches it), merged when split."""
    rows = d["rows"]
    parts = (Buf(rows * 16 * splits * 4), Buf(rows * 16 * splits * 4), Buf(rows * 16 * splits * HD * 4))
    src = {}
    real = C.KIT.run

    def run(fn, grid, args, consts, variant=None):
        C.KIT.run = real  # run_any launches through the kit itself
        try:
            src[fn] = run_any(fn, c, consts, grid, args)
        finally:
            C.KIT.run = run
    C.KIT.run = run
    try:
        out = TA.run_part(d, c, parts)
    finally:
        C.KIT.run = real
    if not d["final"]:
        src["_sparse_attn_merge"] = merge16(d, parts, splits, out)
    H.sync()
    return out, src


def run_any(fn, case, consts, grid, args):
    """The kit's exact variant of fn for consts, else a test-only one built from case; returns which."""
    try:
        var = C.KIT.find(fn, args, consts, exact_only=True)
    except KeyError:
        var = None
    if var is not None:
        C.KIT.run(fn, grid, args, consts, variant=var)
        return "kit"
    base = copy.deepcopy(case)
    if "rows" in base["consts"] and args.get("rows", 1) != 1:
        del base["consts"]["rows"]  # Triton specializes an int argument of 1; the recorded case had one row
    C.run_variant(C.extra_variant(fn, base, **{k: v for k, v in consts.items() if k in base["consts"]}), grid, args)
    return "test-only"


def merges(d, p32, p16s, splits, tag):
    """The decode path's Triton merges (they stay Triton at both head counts): _sparse_attn_merge and
    _sparse_attn_merge_rot2 at 16 heads on the 16-head build's partials against the same kernels at 32 heads on the
    32-head build's partials, bits a head: whether FIRST-LIGHT's decode logits can match (the 16-head variants are
    other compiled objects, from TP4-KIT's triton build)."""
    rows = d["rows"]
    mk = TA.consts_of(L["_sparse_attn_merge"][0])
    rk = {k: v for k, v in TA.consts_of(L["_sparse_attn_merge_rot2"][0]).items() if k != "rows"}
    gh = rk["GH"]
    suh = np.sign(np.random.default_rng(5).standard_normal(32 * HD)).astype(np.float16)
    pos_b = Buf.of(d["pos"])

    def run(hh, parts, sink, sh):
        out, out2, xh = Buf(rows * hh * HD * 2), Buf(rows * hh * HD * 2), Buf((hh // gh) * rows * gh * HD * 2)
        a = dict(PM=P(parts[0], "*fp32"), PL=P(parts[1], "*fp32"), PO=P(parts[2], "*fp32"),
                 SINK=P(Buf.of(sink), "*fp32"), OUT=P(out, "*bf16"))
        s1 = run_any("_sparse_attn_merge", L["_sparse_attn_merge"][0], dict(mk, H=hh, SPLITS=splits), (rows, hh // 16, 1), a)
        a2 = dict(a, OUT=P(out2, "*bf16"), COS=P(TA.COS_B, "*fp32"), SIN=P(TA.SIN_B, "*fp32"), POS=P(pos_b, "*i64"),
                  SUH=P(Buf.of(sh), "*fp16"), XH=P(xh, "*fp16"), rows=rows)
        s2 = run_any("_sparse_attn_merge_rot2", L["_sparse_attn_merge_rot2"][0], dict(rk, H=hh, SPLITS=splits), (rows, hh, 1), a2)
        H.sync()
        return (out.download((rows, hh, HD), np.uint16), out2.download((rows, hh, HD), np.uint16),
                xh.download((hh // gh, rows, gh * HD), np.uint16), f"merge {s1}, rot2 {s2}")

    m32, r32, x32, src32 = run(32, p32, d["sink"], suh)
    for g in (0, 1):
        m16, r16, x16, src16 = run(16, p16s[g], d["sink"][16 * g:16 * g + 16], suh[16 * g * HD:(16 * g + 16) * HD])
        hs = slice(16 * g, 16 * g + 16)
        ng = 16 // gh
        C.record(kernel="merges h16 vs h32", case=f"{tag} heads {16 * g}-{16 * g + 15} (h32: {src32}; h16: {src16})",
                 ok=None, merge_bits_equal=bits(m16, m32[:, hs]), rot2_out_bits_equal=bits(r16, r32[:, hs]),
                 rot2_xh_bits_equal=bits(x16, x32[ng * g:ng * (g + 1)]))


def bits(a, b):
    return bool(np.array_equal(np.ascontiguousarray(a).view(np.uint32 if a.dtype == np.float32 else np.uint16),
                               np.ascontiguousarray(b).view(np.uint32 if b.dtype == np.float32 else np.uint16)))


def main():
    keep = []
    for ci, c in enumerate(L["_sparse_attn_part"]):
        k = TA.consts_of(c)
        rows = min(c["tensors"]["Q"][1][0], 40)
        d = TA.build(c, rows)
        final = bool(k["FINAL"])
        splits = 1 if final else k["SPLITS"]
        tag = (f"rec{ci} rows={rows} ring={d['ring']} comp={d['has_comp']} packed={d['packed']} base={d['has_base']} "
               f"final={final} n_idx={d['n_idx']} splits={splits}")
        out32, p32 = run_hip(FN32, d, splits, keep)
        o32 = out32.download((rows, 32, HD), np.uint16)
        if p32 is not None:
            m32 = p32[0].download((rows, 2, splits, 16), np.float32)
            l32 = p32[1].download((rows, 2, splits, 16), np.float32)
            a32 = p32[2].download((rows, 2, splits, 16, HD), np.float32)
        p16s = []
        for g in (0, 1):
            e = half(d, g)
            out16, p16 = run_hip(FN16, e, splits, keep)
            p16s.append(p16)
            if p16 is None:
                same = bits(out16.download((rows, 16, HD), np.uint16), o32[:, 16 * g:16 * g + 16])
                what = "one-pass output bf16"
            else:
                same = (bits(p16[0].download((rows, splits, 16), np.float32), m32[:, g]) and
                        bits(p16[1].download((rows, splits, 16), np.float32), l32[:, g]) and
                        bits(p16[2].download((rows, splits, 16, HD), np.float32), a32[:, g]))
                what = "split partials m, l, acc fp32"
            C.record(kernel="aux.sparse_attn h16", case=f"{tag} heads {16 * g}-{16 * g + 15}: {what} == h32's", ok=same)
            if g:
                if p16 is not None:
                    merges(d, p32, p16s, splits, tag)
                continue
            # the 16-head output itself: merged at 16 heads, against the float64 reference
            if p16 is not None:
                src = merge16(e, p16, splits, out16)
            got = C.from_bf16_bits(out16.download((rows, 16, HD), np.uint16))
            T.check(tag + " h16", got, e)
            # Triton at 16 heads on the same inputs
            out_t, srcs = triton16(e, c, k["SPLITS"])
            got_t = C.from_bf16_bits(out_t.download((rows, 16, HD), np.uint16))
            T.check(tag + " h16 Triton (" + ",".join(f"{f}:{s}" for f, s in srcs.items()) + ")", got_t, e)
            C.record(kernel="aux.sparse_attn h16", case=tag + " vs Triton h16", ok=None,
                     nl2_vs_triton=float(np.linalg.norm(got - got_t) / max(np.linalg.norm(got_t), 1e-30)),
                     bits_equal=float(np.mean(got == got_t)))
            if rows > 1 and ci in (2, 4, 9):
                same = True
                for r in (0, rows // 2, rows - 1):
                    s1 = T.sub(e, r)
                    o1, q1 = run_hip(FN16, s1, splits, keep)
                    if q1 is not None:
                        merge16(s1, q1, splits, o1)
                    same &= bool(np.array_equal(o1.download((1, 16, HD), np.uint16)[0],
                                                out16.download((rows, 16, HD), np.uint16)[r]))
                C.record(kernel="aux.sparse_attn h16", case=f"rec{ci} row invariance (rows alone vs in {rows})", ok=same)
        keep.clear()
    H.sync()
    sys.exit(C.done())


if __name__ == "__main__":
    C.log_open("t_hipattn16")
    main()
