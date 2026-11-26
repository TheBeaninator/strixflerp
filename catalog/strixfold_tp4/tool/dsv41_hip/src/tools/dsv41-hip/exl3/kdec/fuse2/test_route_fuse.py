"""SPEED-FUSE-2 (31e3ab5061): the fused router launch (_rowmm_route, route_fuse.py) against the two launches it
replaces as the engine issues them (the kit's own _rowmm_parts > _route_parts), bit for bit, on the AOT kit through
hiprt (no torch, no compiler, no model weights): rows 1-16, the four logit regimes of t_route.py (the chunk sums of
rows whose logits fall in the regime, built from x and the gate weight), 2 seeds, each case launched REPS times (the
arrival order differs from launch to launch); every byte of S, PICK and WTS equal and the arrival counter back at
zero. Each fused case is also graded by t_route.route_check (numerics.json router.noaux_tc), the contract the 44
router cases use. Then a back-to-back device timing of the two arms at rows 1, 6, 16 (bench box numbers, not the
pair's).  Usage: TRI_WORK=<dir with kit/gfx1151 and upstream/launches.json> python test_route_fuse.py [out.jsonl]"""
import json
import os
import sys
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "tri"))
try:
    import triton  # noqa: F401
except ImportError:  # t_route imports it for its JIT score grid, which this file does not run
    for _m in ("triton", "triton.language", "triton.language.extra", "triton.language.extra.libdevice"):
        sys.modules[_m] = mock.MagicMock()
import numpy as np

import common as C
import hiprt as H
import t_route as TR
from hiprt import Buf, P

K, N, KC, TOPK, SLOTS, NB, SP = 5120, 384, 20, 6, 7, 512, 8
REPS = int(os.environ.get("REPS", "8"))
DIAG = os.environ.get("DIAG") == "1"
case384 = [c for c in C.launches()["_route_parts"] if c["consts"]["NE"] == N][0]
SCALE, SHARED = case384["scalars"]["scale"]["float"], case384["scalars"]["shared_id"]
rng = np.random.default_rng(3162)
out = open(sys.argv[1], "w") if len(sys.argv) > 1 else None


def emit(d):
    line = json.dumps(d, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(line, flush=True)
    if out:
        out.write(line + "\n")
        out.flush()


def inputs(rows, draw):
    """x bf16 [rows, K] and the gate weight fp16 [N, K] whose chunk sums are draw()'s logits / KC plus N(0, 0.35):
    column c * 256 + i of chunk c carries row i's logits (x 1 there for row i, 0 for the other rows)."""
    x = rng.standard_normal((rows, K)).astype(np.float32)
    w = (rng.standard_normal((N, K)) * 0.022).astype(np.float16)
    lg = draw((rows, N))
    for c in range(KC):
        x[:, c * 256:c * 256 + rows] = np.eye(rows, dtype=np.float32)
        w[:, c * 256:c * 256 + rows] = (lg / KC).T.astype(np.float16)
    return C.bf16_bits(x), w


def tile(rows):
    return 16 if rows <= 6 else 8


class Arm:
    def __init__(self, rows, xb, wb, bias):
        self.rows, self.x, self.w, self.bias = rows, xb, wb, bias
        self.s, self.pick, self.wts = Buf(rows * KC * N * 4), Buf(rows * SLOTS * 4), Buf(rows * SLOTS * 4)
        self.grid = ((N + tile(rows) - 1) // tile(rows), KC, 1)

    def clear(self):
        for b in (self.s, self.pick, self.wts):
            b.fill(0x5A)

    def unfused(self):
        r = self.rows
        C.KIT.run("_rowmm_parts", self.grid, dict(X=P(self.x, "*bf16"), xs=K, W=P(self.w, "*fp16"), S=P(self.s, "*fp32"),
                  rows=r), dict(K=K, N=N, BN=tile(r), BK=256, CPG=1, RB=r, PDL=1))
        C.KIT.run("_route_parts", (r, 1, 1), dict(S=P(self.s, "*fp32"), BIAS=P(self.bias, "*fp32"),
                  PICK=P(self.pick, "*i32"), WTS=P(self.wts, "*fp32"), scale=SCALE, shared_id=SHARED),
                  dict(NE=N, NB=NB, TOPK=TOPK, SLOTS=SLOTS, SP=SP, KC=KC, PDL=1))

    def fused(self, cnt):
        r = self.rows
        C.KIT.run("_rowmm_route", self.grid, dict(X=P(self.x, "*bf16"), xs=K, W=P(self.w, "*fp16"),
                  S=P(self.s, "*fp32"), CNT=P(cnt, "*i32"), BIAS=P(self.bias, "*fp32"), PICK=P(self.pick, "*i32"),
                  WTS=P(self.wts, "*fp32"), scale=SCALE, shared_id=SHARED),
                  dict(K=K, N=N, BN=tile(r), BK=256, CPG=1, RB=r, NB=NB, TOPK=TOPK, SLOTS=SLOTS, SP=SP))

    def re_parts(self):
        """diagnostic kits only: the kit's _rowmm_parts source recompiled by the fused kernels' compiler"""
        r = self.rows
        C.KIT.run("_rowmm_parts_re", self.grid, dict(X=P(self.x, "*bf16"), xs=K, W=P(self.w, "*fp16"),
                  S=P(self.s, "*fp32")), dict(K=K, N=N, BN=tile(r), BK=256, CPG=1, RB=r, PDL=1))

    def route_only(self, s):
        """the kit's _route_parts on another arm's chunk sums"""
        r = self.rows
        C.KIT.run("_route_parts", (r, 1, 1), dict(S=P(s, "*fp32"), BIAS=P(self.bias, "*fp32"),
                  PICK=P(self.pick, "*i32"), WTS=P(self.wts, "*fp32"), scale=SCALE, shared_id=SHARED),
                  dict(NE=N, NB=NB, TOPK=TOPK, SLOTS=SLOTS, SP=SP, KC=KC, PDL=1))

    def get(self):
        r = self.rows
        return (self.s.download((r, KC, N), np.uint32), self.pick.download((r, SLOTS), np.uint32),
                self.wts.download((r, SLOTS), np.uint32))


def bits():
    cnt = Buf(64 * 4)
    cnt.fill(0)
    ok_all, n_ok, n_contract = True, 0, 0
    cases = 0
    for rows in range(1, 17):
        for regime, draw in TR.REGIMES.items():
            for seed in range(2):
                xb, wb = inputs(rows, draw)
                bias = (rng.standard_normal(N) * 0.1).astype(np.float32)
                x, w, b = Buf.of(xb), Buf.of(wb), Buf.of(bias)
                a, f = Arm(rows, x, w, b), Arm(rows, x, w, b)
                diffs = {"S": 0, "PICK": 0, "WTS": 0}
                cnt_left = 0
                diag = {}
                if DIAG:
                    a.clear()
                    f.clear()
                    a.unfused()
                    f.fused(cnt)
                    g = Arm(rows, x, w, b)
                    g.clear()
                    g.route_only(f.s)          # the kit's route on the fused arm's chunk sums
                    H.sync()
                    diag["route_given_fused_S"] = [int((u != v).sum()) for u, v in zip(g.get()[1:], f.get()[1:])]
                    if rows in (1, 2, 16):
                        g.clear()
                        g.re_parts()
                        H.sync()
                        diag["S_recompiled_vs_kit"] = int((g.get()[0] != a.get()[0]).sum())
                        diag["S_recompiled_vs_fused"] = int((g.get()[0] != f.get()[0]).sum())
                for _ in range(REPS):
                    a.clear()
                    f.clear()
                    a.unfused()
                    f.fused(cnt)
                    H.sync()
                    for nm, u, v in zip(diffs, a.get(), f.get()):
                        diffs[nm] += int((u != v).sum())
                    cnt_left += int(np.abs(cnt.download((64,), np.int32)).sum())
                s, pick, wts = f.get()
                s64 = s.view(np.float32).astype(np.float64)
                fails = len(C.FAILS)
                TR.route_check("_rowmm_route", f"rows={rows} logits {regime} seed {seed}", pick.view(np.int32),
                               wts.view(np.float32).astype(np.float64), s64.sum(1), bias, TOPK, SLOTS, SHARED, SCALE,
                               C.gamma(KC) * np.abs(s64).sum(1))
                contract = len(C.FAILS) == fails
                ok = all(d == 0 for d in diffs.values()) and cnt_left == 0
                cases += 1
                n_ok += ok
                n_contract += contract
                ok_all &= ok and contract
                emit({"rows": rows, "regime": regime, "seed": seed, "reps": REPS, "diffs": diffs, "cnt_left": cnt_left,
                      "bits_ok": ok, "contract_ok": contract, **diag, "min_logit": float(s64.sum(1).min()),
                      "max_logit": float(s64.sum(1).max())})
    emit({"summary": "bits", "cases": cases, "bit_identical": n_ok, "contract_pass": n_contract, "reps": REPS,
          "ok": bool(ok_all)})
    return ok_all


def timing():
    cnt = Buf(64 * 4)
    cnt.fill(0)
    for rows in (1, 2, 3, 4, 6, 16):
        xb, wb = inputs(rows, TR.REGIMES["normal(0,1.5)"])
        bias = (rng.standard_normal(N) * 0.1).astype(np.float32)
        arm = Arm(rows, Buf.of(xb), Buf.of(wb), Buf.of(bias))
        t = {"unfused": [], "fused": []}
        for _ in range(5):                       # interleaved batches
            t["unfused"].append(H.time_ms(arm.unfused, iters=400, warmup=20) * 1e3)
            t["fused"].append(H.time_ms(lambda: arm.fused(cnt), iters=400, warmup=20) * 1e3)
        emit({"timing_us_per_site": rows, "launches": {"unfused": 2, "fused": 1},
              "unfused_us": [round(v, 2) for v in t["unfused"]], "fused_us": [round(v, 2) for v in t["fused"]],
              "unfused_median": round(float(np.median(t["unfused"])), 2),
              "fused_median": round(float(np.median(t["fused"])), 2),
              "note": "host-launched back to back on the bench box (python launch overhead in both arms), not a graph"})


if __name__ == "__main__":
    emit({"device": H.device_name(), "kit": str(C.KIT.dir), "kit_triton": C.KIT.spec.get("triton"),
          "fuse2_added": C.KIT.spec.get("fuse2_added")})
    good = bits()
    timing()
    H.sync()
    sys.exit(0 if good else 1)
