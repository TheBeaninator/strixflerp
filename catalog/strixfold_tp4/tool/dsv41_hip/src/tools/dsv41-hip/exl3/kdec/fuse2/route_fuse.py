"""SPEED-FUSE-2 (31e3ab5061): the router's two decode launches, _rowmm_parts > _route_parts, as ONE launch, TF_FUSE2=1,
from tri_kernels.py's own code so every value keeps its arithmetic and bits.

Programs (nb, g) run _rowmm_parts' code unchanged (the gate's chunk sums S [R, KC, N], chunk g of BN outputs). The
last program of the launch to arrive (an acq_rel counter CNT[0], reset after: cdiv(N, BN) * KC / CPG arrivals) runs
_route_parts' code for each of the window's rows in row order: the logits summed from zero, chunk 0 first, then
_route_row's selection (lowest index among ties) and the weights normalised in pick order. Nothing in the routing
reduces floats across lanes (max / min are exact, the picked score is a one-hot sum), so the 2 warps of
_rowmm_parts give the same bits as _route_parts' one.

Why here and not in decode_prep (the card's first plan, KDEC.md section 7 row 2): decode_prep's rotation blocks read
the picks while its block 0 groups them, so a route folded into decode_prep has every one of its ~70 blocks recompute
the row's N scores from the KC * N chunk sums, and has the score (Triton's _sqrt_softplus: libdevice.log1p under
Triton's fp fusion) re-implemented in hipcc C++ built -ffp-contract=off. This fold removes the same launch a site with
the kit's own code.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tri"))
import triton
import triton.language as tl
import tri_kernels as K


@triton.jit
def _route_row_parts(S, BIAS, PICK, WTS, scale, shared_id, r, NE: tl.constexpr, NB: tl.constexpr, TOPK: tl.constexpr,
                     SLOTS: tl.constexpr, SP: tl.constexpr, KC: tl.constexpr):
    """_route_parts' code for row r (its program_id(0)), line for line."""
    e = tl.arange(0, NB)
    ok = e < NE
    lg = tl.zeros((NB,), dtype=tl.float32)
    for t in tl.static_range(KC):
        lg += tl.load(S + (r * KC + t) * NE + e, mask=ok, other=0.0)
    PICK = PICK + r * SLOTS
    WTS = WTS + r * SLOTS
    sc = K._sqrt_softplus(lg)
    b = tl.load(BIAS + e, mask=ok, other=0.0)
    sel = tl.where(ok, sc + b, float("-inf"))
    tot = 0.0
    s = tl.arange(0, SP)
    picks = tl.full((SP,), shared_id, dtype=tl.int32)
    ws = tl.zeros((SP,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        best = tl.max(sel, axis=0)
        idx = tl.min(tl.where(sel == best, e, NB), axis=0)          # lowest index among ties
        v = tl.sum(tl.where(e == idx, sc, 0.0), axis=0)
        picks = tl.where(s == k, idx, picks)
        ws = tl.where(s == k, v, ws)
        tot += v
        sel = tl.where(e == idx, float("-inf"), sel)
    ws = tl.where(s < TOPK, ws / (tot + 1e-20) * scale, 1.0)
    tl.store(PICK + s, picks, mask=s < SLOTS)
    tl.store(WTS + s, ws, mask=s < SLOTS)


@triton.jit
def _rowmm_route(X, xs, W, S, CNT, BIAS, PICK, WTS, scale, shared_id, K_: tl.constexpr, N: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr, CPG: tl.constexpr, RB: tl.constexpr, NB: tl.constexpr,
                 TOPK: tl.constexpr, SLOTS: tl.constexpr, SP: tl.constexpr):
    nb = tl.program_id(0)
    g = tl.program_id(1)
    nn = nb * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    KC: tl.constexpr = K_ // BK
    ws = ()
    for c in tl.static_range(CPG):
        ws = ws + (tl.load(W + nn[:, None] * K_ + ((g * CPG + c) * BK + kk)[None, :], mask=(nn < N)[:, None],
                           other=0.0),)
    out = ()
    for i in tl.static_range(RB):                      # RB == rows (a variant a row count)
        for c in tl.static_range(CPG):
            # x's chunk as a [BN, BK] tile (every warp the same row: the tile's own layout, no conversion)
            x = tl.load(X + i * xs + ((g * CPG + c) * BK + kk)[None, :] + 0 * nn[:, None]).to(tl.float32)
            out = out + (tl.sum(ws[c].to(tl.float32) * x, axis=1),)
    for i in tl.static_range(RB):
        for c in tl.static_range(CPG):
            tl.store(S + (i * KC + g * CPG + c) * N + nn, out[i * CPG + c], mask=nn < N)
    tl.debug_barrier()
    NPROG: tl.constexpr = ((N + BN - 1) // BN) * (KC // CPG)
    old = tl.atomic_add(CNT, 1, sem="acq_rel")
    if old == NPROG - 1:                               # the launch's last program: every chunk sum is stored
        for r in range(RB):
            _route_row_parts(S, BIAS, PICK, WTS, scale, shared_id, r, N, NB, TOPK, SLOTS, SP, KC)
        tl.atomic_xchg(CNT, 0, sem="relaxed")


def rp_tile(rows):
    """tri_norm.rpTile: _rowmm_parts' outputs a program by rows (CPG 1, 2 warps)."""
    return 16 if rows <= 6 else 8
