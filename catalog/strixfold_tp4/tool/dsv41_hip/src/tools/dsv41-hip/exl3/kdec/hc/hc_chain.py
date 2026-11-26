"""k-dec2 SPEED-FUSE-1 (b7269694f8): hc_pre2's whole decode chain, _hc_post_only included, as ONE launch,
TF_HC_CHAIN=1, from tri_kernels.py's own code so every part keeps its arithmetic and bits.

Programs (g, b < NB, mg) run _hc_mix_part's POSTED code for their row group (the same fma spell as _hc_post_only,
per tile: the post recomputed inline, and mg 0 stores the new stream to XO exactly as _hc_post_only would have);
its sum of squares is reduced in the 2-warp tree spelled out (_ss_tree, as in hc_fuse._hc_fused, else part slot 24
differs at 8 warps). Programs (g, NB, 0) run _hc_finish_s's role 1 (the collapse + RMSNorm) for their rows, with the
post of the whole row recomputed inline the same way (the same bits; nothing here reads XO, the kernel boundary is
what makes the stored h_out visible to the round's next kernels). The last of a group's 3 * NB mixes AND its finish
program (an acq_rel counter, reset after; so 3 * NB + 1 arrivals) runs _hc_sinkhorn's code for the group's rows: the
finish program reads the previous site's POST / COMB, which sinkhorn overwrites. Compiled with 8 warps, as _hc_fused.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "tfp/tri/src"))
import triton
import triton.language as tl
import tri_kernels as K
from hc_fuse import _sink_row, _ss_tree, hc_rb  # _hc_fused's row helpers, unchanged


@triton.jit
def _mix_rows8p(X, XO, G, RS, POST, COMB, PART, ws, r, b, mg, m, k, WIDE: tl.constexpr, D: tl.constexpr,
                NB: tl.constexpr, KB: tl.constexpr, SUB: tl.constexpr, MB: tl.constexpr, WORLD: tl.constexpr):
    """_hc_mix_rows (POSTED True) with its sum of squares reduced in _hc_mix_part's tree (_ss_tree): the post's
    prologue (comb's column s, post s, the gathered partials) is _hc_post_only's, and mg 0 stores the new stream."""
    s = (b * KB) // D
    col0 = b * KB - s * D
    acc = tl.zeros((MB,), dtype=tl.float32)
    ss = tl.zeros((SUB,), dtype=tl.float32)
    c0 = tl.load(COMB + r * 16 + 0 * 4 + s)
    c1 = tl.load(COMB + r * 16 + 1 * 4 + s)
    c2 = tl.load(COMB + r * 16 + 2 * 4 + s)
    c3 = tl.load(COMB + r * 16 + 3 * 4 + s)
    ps = tl.load(POST + r * 4 + s)
    for t in tl.static_range(KB // SUB):
        d = col0 + t * SUB + k
        ya = tl.load(G + r * D + d)
        for q in tl.static_range(1, WORLD):
            ya = ya + tl.load(G + q * RS + r * D + d)
        y = ya.to(tl.bfloat16).to(tl.float32)
        x0 = tl.load(X + r * WIDE + d).to(tl.float32)
        x1 = tl.load(X + r * WIDE + D + d).to(tl.float32)
        x2 = tl.load(X + r * WIDE + 2 * D + d).to(tl.float32)
        x3 = tl.load(X + r * WIDE + 3 * D + d).to(tl.float32)
        # _hc_post's contraction, spelled out: fma(ps, y, fma(c3, x3, fma(c2, x2, fma(c0, x0, c1 x1))))
        v = tl.fma(ps, y, tl.fma(c3, x3, tl.fma(c2, x2, tl.fma(c0, x0, c1 * x1))))
        xb = v.to(tl.bfloat16)
        if mg == 0:
            tl.store(XO + r * WIDE + s * D + d, xb)
        x = xb.to(tl.float32)
        acc += tl.sum(ws[t] * x[None, :], axis=1)
        ss = tl.fma(x, x, ss)                                   # _hc_partial's fma(x, x, ss)
    tl.store(PART + (r * NB + b) * 32 + m, acc)
    if mg == 0:
        tl.store(PART + (r * NB + b) * 32 + 24, _ss_tree(ss))


@triton.jit
def _fin_row_p(X, XO, G, RS, POST, COMB, PRE_IN, NW, OUT, eps, r, D: tl.constexpr, WIDE: tl.constexpr,
               BLOCK: tl.constexpr, WORLD: tl.constexpr):
    """_hc_finish_s's role 1 (NROT 0, the code of hc_fuse._fin_row) for row r, of the NEW streams recomputed inline:
    _post_only's v for the row's BLOCK columns and stream s, in the order _fin_row loads X (= h_out) back: the same
    bf16 bits, so the collapse, the RMSNorm and the stored row are its bits. Stores no XO (the mixes' mg 0 do)."""
    pl = tl.arange(0, BLOCK // 8)
    K._prefetch_l2(NW + tl.minimum(pl * 64, D - 1))
    p0 = tl.load(PRE_IN + r * 4 + 0)
    p1 = tl.load(PRE_IN + r * 4 + 1)
    p2 = tl.load(PRE_IN + r * 4 + 2)
    p3 = tl.load(PRE_IN + r * 4 + 3)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    cs = ()
    for c in tl.static_range(D // BLOCK):
        d = c * BLOCK + tl.arange(0, BLOCK)
        x0 = tl.load(X + r * WIDE + d).to(tl.float32)
        x1 = tl.load(X + r * WIDE + D + d).to(tl.float32)
        x2 = tl.load(X + r * WIDE + 2 * D + d).to(tl.float32)
        x3 = tl.load(X + r * WIDE + 3 * D + d).to(tl.float32)
        vs = ()
        for s in tl.static_range(4):
            ya = tl.load(G + r * D + d)
            for q in tl.static_range(1, WORLD):
                ya = ya + tl.load(G + q * RS + r * D + d)
            y = ya.to(tl.bfloat16).to(tl.float32)
            c0 = tl.load(COMB + r * 16 + 0 * 4 + s)
            c1 = tl.load(COMB + r * 16 + 1 * 4 + s)
            c2 = tl.load(COMB + r * 16 + 2 * 4 + s)
            c3 = tl.load(COMB + r * 16 + 3 * 4 + s)
            ps = tl.load(POST + r * 4 + s)
            v = tl.fma(ps, y, tl.fma(c3, x3, tl.fma(c2, x2, tl.fma(c0, x0, c1 * x1))))
            vs = vs + (v.to(tl.bfloat16).to(tl.float32),)
        cc = ((((p0 * vs[0] + p1 * vs[1]) + p2 * vs[2]) + p3 * vs[3]).to(tl.bfloat16)).to(tl.float32)
        acc += cc * cc
        cs = cs + (cc,)
    rinv = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / D + eps)
    K._rot_out(OUT, r, cs, rinv, NW, OUT, OUT, 0, D, BLOCK, False)


@triton.jit
def _hc_chain(X, XO, G, RS, POST, COMB, FN, PART, CNT, BASE, SCALE, PRE_IN, NW, OUT, PRE_OUT, eps, hc_eps, rows,
              WIDE: tl.constexpr, D: tl.constexpr, NB: tl.constexpr, SUB: tl.constexpr, MB: tl.constexpr,
              RB: tl.constexpr, ITERS: tl.constexpr, BLOCK: tl.constexpr, WORLD: tl.constexpr):
    g = tl.program_id(0)
    b = tl.program_id(1)
    mg = tl.program_id(2)
    r0 = g * RB
    if b < NB:
        KB: tl.constexpr = WIDE // NB
        m = mg * MB + tl.arange(0, MB)
        k = tl.arange(0, SUB)
        ws = ()
        for t in tl.static_range(KB // SUB):
            ws = ws + (tl.load(FN + m[:, None] * WIDE + b * KB + t * SUB + k[None, :]),)
        for i in tl.static_range(RB):
            if r0 + i < rows:
                _mix_rows8p(X, XO, G, RS, POST, COMB, PART, ws, r0 + i, b, mg, m, k, WIDE, D, NB, KB, SUB, MB, WORLD)
        tl.debug_barrier()
        old = tl.atomic_add(CNT + g, 1, sem="acq_rel")
        if old == 3 * NB:  # the last of the 3 * NB mixes and of the group's finish program
            for i in tl.static_range(RB):
                if r0 + i < rows:
                    _sink_row(PART, BASE, SCALE, PRE_OUT, POST, COMB, eps, hc_eps, r0 + i, D, NB, ITERS)
            tl.atomic_xchg(CNT + g, 0, sem="relaxed")
    else:
        if mg == 0:
            for i in tl.static_range(RB):
                if r0 + i < rows:
                    _fin_row_p(X, XO, G, RS, POST, COMB, PRE_IN, NW, OUT, eps, r0 + i, D, WIDE, BLOCK, WORLD)
            tl.debug_barrier()
            old = tl.atomic_add(CNT + g, 1, sem="acq_rel")
            if old == 3 * NB:
                for i in tl.static_range(RB):
                    if r0 + i < rows:
                        _sink_row(PART, BASE, SCALE, PRE_OUT, POST, COMB, eps, hc_eps, r0 + i, D, NB, ITERS)
                tl.atomic_xchg(CNT + g, 0, sem="relaxed")
