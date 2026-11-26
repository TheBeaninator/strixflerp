"""Device time of aux.sparse_attn's 16-head build (a TP4 rank) against its 32-head build (a TP2 rank) and the Triton
_sparse_attn_part at 16 heads (what a TP4 rank runs without it) and at 32: decode / verify windows (1-16 rows, 8
splits, ring window + 512 FP4 picks) and prompt chunks (512 / 2048 rows, one pass, linear window + 512 FP4 picks),
x_hipattn_perf.py's inputs. The 16-head runs take heads 0-15 of the 32-head inputs (the same keys and picks), and each
case first checks the 16-head build's bits against the 32-head build's on both head halves. Run under rocprofv3
--kernel-trace; x_runs.py prints the runs in launch order (h32, h16, Triton H=16, Triton H=32 a case)."""

import os

import numpy as np

import common as C
import hiprt as H
import t_hipattn as T
import t_hipattn16 as T16
import x_hipattn_perf as XP
from hiprt import Buf

L = XP.L
HD = 512
REPS_DEC = int(os.environ.get("REPS_DEC", "20"))
REPS_PF = int(os.environ.get("REPS_PF", "5"))


def main():
    C.log_open("x_hipattn16_perf")
    keep = []
    kdec = T.TA.consts_of(next(c for c in L["_sparse_attn_part"] if c["consts"]["NBLK"] == 20 and not c["consts"]["FINAL"]))
    kfin = T.TA.consts_of(next(c for c in L["_sparse_attn_part"] if c["consts"]["FINAL"] and c["consts"]["NBLK"] == 20))
    shapes = [(r, False) for r in (1, 2, 4, 6, 8, 12, 16)] + [(512, True), (2048, True)]
    cases = []
    for rows, final in shapes:
        d = XP.inputs(rows, final)
        d["k"] = {}
        cases.append((rows, final, d, T16.half(d, 0)))
    C.record(kernel="window", case="load", ok=None, load=open("/proc/loadavg").read().split()[:3])
    for rows, final, d, e in cases:
        sp = 1 if final else 8
        # bits: h16 on each head half == h32's heads
        o32, p32 = T16.run_hip(T16.FN32, d, sp, keep)
        H.sync()
        same = True
        for g in (0, 1):
            eg = T16.half(d, g)
            o16, p16 = T16.run_hip(T16.FN16, eg, sp, keep)
            H.sync()
            if final:
                same &= T16.bits(o16.download((rows, 16, HD), np.uint16),
                                 o32.download((rows, 32, HD), np.uint16)[:, 16 * g:16 * g + 16])
            else:
                for i, shp in ((0, (rows, 2, sp, 16)), (1, (rows, 2, sp, 16)), (2, (rows, 2, sp, 16, HD))):
                    same &= T16.bits(p16[i].download((rows, sp) + shp[3:], np.float32), p32[i].download(shp, np.float32)[:, g])
        C.record(kernel="aux.sparse_attn h16", case=f"perf inputs rows={rows} final={final}: bits == h32's on both halves", ok=same)
        keep.clear()
        reps = REPS_PF if final else REPS_DEC
        C.record(kernel="case", case=f"rows={rows} final={final} order: h32, h16, triton16, triton32")
        for fn, x in ((T16.FN32, d), (T16.FN16, e)):
            for _ in range(reps):
                T16.run_hip(fn, x, sp, keep)
                keep.clear()
        for x, hh in ((e, 16), (d, 32)):
            out = Buf(rows * hh * HD * 2)
            parts = (Buf(rows * hh * 8 * 4), Buf(rows * hh * 8 * 4), Buf(rows * hh * 8 * HD * 4))
            a = XP.triton_args(x, out, parts)
            kk = dict(kfin if final else kdec, H=hh)
            try:
                var = C.KIT.find("_sparse_attn_part", a, kk, exact_only=True)
            except KeyError as err:
                C.record(kernel="case", case=f"rows={rows} final={final} triton{hh}: no exact kit variant", ok=False, error=str(err)[:300])
                continue
            for _ in range(reps):
                C.KIT.run("_sparse_attn_part", (rows, hh // 16, sp), a, kk, variant=var)
            H.sync()
    H.sync()
    C.done()


if __name__ == "__main__":
    main()
