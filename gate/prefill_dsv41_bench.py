#!/usr/bin/env python3
"""Prefill kernel bench for the recipe dsv41-flash.strixfold-tp2 (docs/PREFILL-DSV41.md): the HIP kernels one TP rank
launches for a prompt chunk of DeepSeek-V4.1-Flash, each alone on one GPU, at the shapes a rank's chunk has
(catalog/strixfold/executor/dsv41 prompt.zig; config.zig checkShapes: hidden 5120, 32 heads of 512 a rank, experts
385 x (5120 -> 1152 -> 5120), 7 slots a row, window 128, 512 picks, 32 index heads of 128).

It loads the assembled recipe's kit (the code objects the served deployment runs, byte for byte: the recipe gate's
build stage compares the kit manifest) and launches kernels through libamdhip64 with the argument structs of
kernel.strixfold.dsv41_abi (abi.h). No model is loaded: weights are random trellis words and vectors of the right
sizes, activations Gaussian. Times are device times of back-to-back launches with one synchronize, so they hold the
kernel's arithmetic and its memory traffic and nothing of a chunk's host steps or collectives.

    python3 gate/prefill_dsv41_bench.py --kit DIR [--rows 512,1024,2048,4096] [--only experts,linear,...] [--out DIR]

Also checked, because candidate 1 of the document rests on it: the routed experts' output of a row does not depend on
which rows share its launch (rows [0, n) run as one launch against the same rows run as two halves, compared bit for
bit).

Standard library and numpy only (the pod is air-gapped). GPU guard and lease as every other gate (strixite_recipe).
"""
import argparse
import ctypes
import hashlib
import json
import os
import statistics
import sys
import time
import tomllib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import strixite_recipe as guard  # noqa: E402

c_void_p, c_u64, c_i64, c_i32, c_u32, c_f32 = (ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int64, ctypes.c_int32,
                                               ctypes.c_uint32, ctypes.c_float)
D, HL, HD, WIN, TOPK, IH, ID, HC = 5120, 32, 512, 128, 512, 32, 128, 4
E, EI, SLOTS = 385, 1152, 7
lib = None
rng = np.random.default_rng(41)


def check(err, what):
    if err != 0:
        raise RuntimeError(f"HIP error {err} in {what}")


class Dev:
    """A device allocation; `fill` leaves random bytes, zeros, or a given array."""
    live = []

    def __init__(self, nbytes, fill="zero"):
        self.n = int(nbytes)
        p = c_void_p()
        check(lib.hipMalloc(ctypes.byref(p), ctypes.c_size_t(max(self.n, 256))), f"hipMalloc({self.n})")
        self.ptr = p.value
        Dev.live.append(self)
        if isinstance(fill, np.ndarray):
            self.put(fill)
        elif fill == "random":
            blk = rng.integers(0, 2**32, min(self.n // 4 + 1, 16 << 20), dtype=np.uint32)
            off = 0
            while off < self.n:
                n = min(self.n - off, blk.nbytes)
                check(lib.hipMemcpy(c_void_p(self.ptr + off), blk.ctypes.data_as(c_void_p), ctypes.c_size_t(n), 1), "H2D")
                off += n
        elif fill == "zero":
            check(lib.hipMemset(c_void_p(self.ptr), 0, ctypes.c_size_t(self.n)), "hipMemset")

    def put(self, arr, off=0):
        arr = np.ascontiguousarray(arr)
        assert off + arr.nbytes <= max(self.n, 256), (off, arr.nbytes, self.n)
        check(lib.hipMemcpy(c_void_p(self.ptr + off), arr.ctypes.data_as(c_void_p), ctypes.c_size_t(arr.nbytes), 1), "H2D")

    def get(self, dtype, count, off=0):
        out = np.empty(count, dtype)
        check(lib.hipMemcpy(out.ctypes.data_as(c_void_p), c_void_p(self.ptr + off), ctypes.c_size_t(out.nbytes), 2), "D2H")
        return out

    def free(self):
        if self.ptr:
            lib.hipFree(c_void_p(self.ptr))
            self.ptr = 0
            Dev.live.remove(self)


def free_all():
    for d in list(Dev.live):
        d.free()


def bf16(shape, scale=1.0):
    return (rng.standard_normal(shape, dtype=np.float32) * scale).view(np.uint32).__rshift__(16).astype(np.uint16)


def f16(shape, scale=1.0):
    return (rng.standard_normal(shape, dtype=np.float32) * scale).astype(np.float16)


def sync():
    check(lib.hipDeviceSynchronize(), "hipDeviceSynchronize")


class Kit:
    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, "manifest.json"), "rb") as f:
            raw = f.read()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.man = json.loads(raw)
        self.mods, self.fns, self.keep = {}, {}, []

    def fn(self, symbol):
        if symbol in self.fns:
            return self.fns[symbol]
        k = next(k for k in self.man["kernels"] if k["symbol"] == symbol)
        if k["object"] not in self.mods:
            with open(os.path.join(self.path, k["object"]), "rb") as f:
                img = f.read()
            buf = ctypes.create_string_buffer(img, len(img))
            m = c_void_p()
            check(lib.hipModuleLoadData(ctypes.byref(m), buf), "hipModuleLoadData " + k["object"])
            self.keep.append(buf)
            self.mods[k["object"]] = m
        f = c_void_p()
        check(lib.hipModuleGetFunction(ctypes.byref(f), self.mods[k["object"]], symbol.encode()), symbol)
        self.fns[symbol] = f
        return f


def launch(f, grid, block, params):
    holders = [ctypes.pointer(v) for v in params]
    arr = (c_void_p * len(holders))(*[ctypes.cast(h, c_void_p) for h in holders])
    g = list(grid) + [1] * (3 - len(grid))
    check(lib.hipModuleLaunchKernel(f, c_u32(g[0]), c_u32(g[1]), c_u32(g[2]), c_u32(block), c_u32(1), c_u32(1),
                                    c_u32(0), c_void_p(None), arr, c_void_p(None)), "hipModuleLaunchKernel")


def time_ms(call, min_s=0.25, repeats=5):
    """Milliseconds a call: the median over `repeats` batches of back-to-back calls, one synchronize a batch."""
    call()
    sync()
    t0 = time.perf_counter()
    call()
    sync()
    one = max(time.perf_counter() - t0, 1e-6)
    iters = int(max(1, min(200, min_s / one)))
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            call()
        sync()
        out.append((time.perf_counter() - t0) * 1e3 / iters)
    return statistics.median(out), min(out), iters


# --- the EXL3 prompt linear: exl3.rot_in + exl3.gemm (exl3_prefill.zig matmul) --------------------------------------
def gemm_tile(n, m):
    if (n // 128) * ((m + 255) // 256) >= 120:
        return 256
    return 256 if (m >= 512 and (m + 255) // 256 * 2 == (m + 127) // 128) else 128


def bench_linear(kit, name, k, n, k2, out_f32, m, x_k=None):
    x = Dev(m * k * 2, bf16((m, k)))
    suh = Dev(k * 2, f16(k, 0.1))
    svh = Dev(n * 2, f16(n, 0.1))
    xh = Dev(m * k * 2)
    stride_k, stride_nb = 32 * k2, (k // 16) * 32 * k2
    words = Dev((n // 128) * stride_nb * 4 + 4096, "random")
    out = Dev(m * n * (4 if out_f32 else 2))
    rot = kit.fn("tf_ds_exl3_rot_in")
    bm = gemm_tile(n, m)
    gm = kit.fn(f"tf_ds_exl3_gemm_k{k2}_m{bm}")

    def do_rot():
        launch(rot, ((k // 128 + 3) // 4, m), 128, [c_u64(x.ptr), c_i32(1), c_i64(k), c_u64(suh.ptr), c_u64(xh.ptr), c_i32(k)])

    def do_gemm():
        launch(gm, ((n // 128) * ((m + bm - 1) // bm),), 256,
               [c_u64(xh.ptr), c_u64(words.ptr), c_i64(stride_k), c_i64(stride_nb), c_u64(svh.ptr), c_u64(out.ptr),
                c_i64(n), c_i32(m), c_i32(k), c_i32(n), c_i32(1 if out_f32 else 0)])

    r_ms = time_ms(do_rot)[0]
    g_ms = time_ms(do_gemm)[0]
    for b in (x, suh, svh, xh, words, out):
        b.free()
    return {"kernel": "exl3.rot_in+exl3.gemm", "case": name, "rows": m, "K": k, "N": n, "k2": k2, "tile": bm,
            "rot_ms": round(r_ms, 4), "gemm_ms": round(g_ms, 4), "ms": round(r_ms + g_ms, 4),
            "tflops": round(2.0 * m * k * n / (g_ms * 1e-3) / 1e12, 2),
            "weight_GBps": round((n // 128) * stride_nb * 4 * ((m + bm - 1) // bm) / (g_ms * 1e-3) / 1e9, 1)}


# --- routed experts of a prompt chunk: exl3.pf_experts.* (exl3_experts.zig pfPlan / promptPf) -----------------------
class PfArgs(ctypes.Structure):
    _fields_ = [(n, c_u64) for n in ("gate_ptr", "up_ptr", "down_ptr", "gate_k2", "up_k2", "down_k2", "suh_g", "suh_u",
                                     "svh_g", "svh_u", "suh_d", "svh_d", "x", "ids", "wts", "xg", "xu", "xd", "z",
                                     "order", "jobs", "meta", "out")] + \
               [(n, c_u32) for n in ("r", "slots", "e", "d", "i", "ldx", "ldo", "maxjobs", "nmax", "accumulate")] + \
               [("limit", c_f32), ("x_fmt", c_u32)]


assert ctypes.sizeof(PfArgs) == 232


class Experts:
    """A rank's expert table of one layer with random trellises: 384 routed experts at k2_r, the shared one (the
    table's last) gate / up k2 8 and down k2 10 (exl3_experts.zig's test "layer 29")."""

    def __init__(self, k2_r, max_rows):
        self.k2_r = k2_r
        gk2 = np.full(E, k2_r, np.int32)
        dk2 = np.full(E, k2_r, np.int32)
        gk2[-1], dk2[-1] = 8, 10
        tiles = (D // 16) * (EI // 16)
        sz_g, sz_d = tiles * 4 * gk2.astype(np.int64) * 4, tiles * 4 * dk2.astype(np.int64) * 4
        self.weight_bytes = int(2 * sz_g.sum() + sz_d.sum())
        self.arena = Dev(self.weight_bytes + (1 << 20), "random")
        off, tabs = 0, []
        for sz in (sz_g, sz_g, sz_d):
            t = np.empty(E, np.int64)
            for e in range(E):
                t[e] = self.arena.ptr + off
                off += int(sz[e])
            tabs.append(Dev(E * 8, t))
        self.ptrs = tabs
        self.gk2, self.dk2 = Dev(E * 4, gk2), Dev(E * 4, dk2)
        self.masks = (sorted(set(gk2.tolist())), sorted(set(dk2.tolist())))
        self.suh_g, self.suh_u = Dev(E * D * 2, f16((E, D), 0.05)), Dev(E * D * 2, f16((E, D), 0.05))
        self.svh_g, self.svh_u = Dev(E * EI * 2, f16((E, EI), 0.05)), Dev(E * EI * 2, f16((E, EI), 0.05))
        self.suh_d, self.svh_d = Dev(E * EI * 2, f16((E, EI), 0.05)), Dev(E * D * 2, f16((E, D), 0.05))
        p = max_rows * SLOTS
        self.xg, self.xu, self.xd = Dev(2 * p * D), Dev(2 * p * D), Dev(2 * p * EI)
        self.z = Dev(4 * max(2 * EI + D // 2, D) * p)
        self.order, self.jobs, self.meta = Dev(4 * p), Dev(12 * (p // 64 + E)), Dev(8 * 1025)
        self.max_rows = max_rows

    def plan(self, x, ids, wts, out, r):
        p = r * SLOTS
        nm = 128 if p >= 48 * E else 64
        mj = p // nm + E
        a = PfArgs(self.ptrs[0].ptr, self.ptrs[1].ptr, self.ptrs[2].ptr, self.gk2.ptr, self.gk2.ptr, self.dk2.ptr,
                   self.suh_g.ptr, self.suh_u.ptr, self.svh_g.ptr, self.svh_u.ptr, self.suh_d.ptr, self.svh_d.ptr,
                   x, ids, wts, self.xg.ptr, self.xu.ptr, self.xd.ptr, self.z.ptr, self.order.ptr, self.jobs.ptr,
                   self.meta.ptr, out, r, SLOTS, E, D, EI, D, D, mj, nm, 0, 10.0, 1)
        return a, nm, mj

    def phases(self, kit, a, nm, mj):
        r = a.r
        ph = [("count", lambda: launch(kit.fn("tf_ds_exl3_pf_experts_count"), (1,), 1024, [a])),
              ("place", lambda: launch(kit.fn("tf_ds_exl3_pf_experts_place"), ((E + 7) // 8,), 256, [a])),
              ("rot", lambda: launch(kit.fn("tf_ds_exl3_pf_experts_rot"), (r * SLOTS,), 256, [a]))]
        for k2 in self.masks[0]:
            f = kit.fn(f"tf_ds_exl3_pf_experts_gu{nm}_k{k2}")
            ph.append((f"gu_k{k2}", lambda f=f: launch(f, (mj * (EI // 128),), 256, [a])))
        for k2 in self.masks[1]:
            f = kit.fn(f"tf_ds_exl3_pf_experts_down{nm}_k{k2}")
            ph.append((f"down_k{k2}", lambda f=f: launch(f, (mj * (D // 256),), 256, [a])))
        ph.append(("combine", lambda: launch(kit.fn("tf_ds_exl3_pf_experts_combine"), (min((r * (D // 4) + 255) // 256, 8192),), 256, [a])))
        return ph

    def run(self, kit, x, ids, wts, out, r):
        a, nm, mj = self.plan(x, ids, wts, out, r)
        for _, call in self.phases(kit, a, nm, mj):
            call()


def routing(rows, kind):
    """ids int32 [rows, 7]: six distinct routed experts a row and the shared one last. uniform: every expert alike;
    skew: expert e drawn with weight 1 / (1 + e / 24) (about a third of the experts take two thirds of the picks)."""
    w = np.ones(E - 1) if kind == "uniform" else 1.0 / (1.0 + np.arange(E - 1) / 24.0)
    keys = rng.random((rows, E - 1)) ** (1.0 / w)          # Efraimidis-Spirakis: weighted sampling without replacement
    ids = np.argpartition(-keys, 6, axis=1)[:, :6].astype(np.int32)
    return np.concatenate([ids, np.full((rows, 1), E - 1, np.int32)], axis=1)


def bench_experts(kit, ex, rows, kind):
    ids_h = routing(rows, kind)
    x = Dev(rows * D * 2, bf16((rows, D)))
    ids = Dev(rows * SLOTS * 4, ids_h)
    wts = Dev(rows * SLOTS * 4, rng.random((rows, SLOTS), dtype=np.float32))
    out = Dev(rows * D * 4)
    a, nm, mj = ex.plan(x.ptr, ids.ptr, wts.ptr, out.ptr, rows)
    ph = ex.phases(kit, a, nm, mj)
    for _, call in ph:
        call()
    sync()
    per = {name: round(time_ms(call, min_s=0.15, repeats=3)[0], 4) for name, call in ph}
    total, total_min, _ = time_ms(lambda: [c() for _, c in ph], min_s=0.3)
    distinct = int(len(np.unique(ids_h)))
    counts = np.bincount(ids_h.ravel(), minlength=E)
    jobs = int(((counts + nm - 1) // nm).sum())
    for b in (x, ids, wts, out):
        b.free()
    flops = 2.0 * rows * SLOTS * 3 * D * EI
    return {"kernel": "exl3.pf_experts", "case": f"{kind} k2={ex.k2_r}", "rows": rows, "routing": kind, "k2": ex.k2_r,
            "nm": nm, "jobs": jobs, "distinct_experts": distinct, "rows_per_routed_expert": round(rows * 6 / (E - 1), 1),
            "ms": round(total, 3), "ms_min": round(total_min, 3), "phases_ms": per,
            "ms_per_row": round(total / rows, 5), "tflops": round(flops / (total * 1e-3) / 1e12, 2),
            "weight_bytes_table": ex.weight_bytes, "table_GBps_if_read_once": round(ex.weight_bytes / (total * 1e-3) / 1e9, 1)}


def experts_row_invariance(kit, ex, rows, kind):
    """Rows [0, rows) as one launch against the same rows as two halves: the output's bits."""
    ids_h = routing(rows, kind)
    x = Dev(rows * D * 2, bf16((rows, D)))
    ids = Dev(rows * SLOTS * 4, ids_h)
    wts = Dev(rows * SLOTS * 4, rng.random((rows, SLOTS), dtype=np.float32))
    out1, out2 = Dev(rows * D * 4), Dev(rows * D * 4)
    ex.run(kit, x.ptr, ids.ptr, wts.ptr, out1.ptr, rows)
    sync()
    h = rows // 2
    for r0, n in ((0, h), (h, rows - h)):
        ex.run(kit, x.ptr + r0 * D * 2, ids.ptr + r0 * SLOTS * 4, wts.ptr + r0 * SLOTS * 4, out2.ptr + r0 * D * 4, n)
        sync()
    a, b = out1.get(np.uint32, rows * D), out2.get(np.uint32, rows * D)
    fl = a.view(np.float32)
    res = {"check": "experts-row-invariance", "rows": rows, "halves": [h, rows - h], "routing": kind, "k2": ex.k2_r,
           "elements": int(a.size), "differing": int((a != b).sum()), "finite": int(np.isfinite(fl).sum()),
           "nonzero": int((a != 0).sum()), "sha256_joint": hashlib.sha256(a.tobytes()).hexdigest(),
           "sha256_halves": hashlib.sha256(b.tobytes()).hexdigest()}
    res["equal"] = res["differing"] == 0 and res["nonzero"] > a.size // 2
    for d in (x, ids, wts, out1, out2):
        d.free()
    return res


# --- aux kernels ----------------------------------------------------------------------------------------------------
class AttnArgs(ctypes.Structure):
    _fields_ = [(n, c_u64) for n in ("q", "wsrc", "wlo", "comp", "csc", "idx", "pos", "pm", "pl", "po", "sink", "out",
                                     "rbase", "cbase")] + \
               [(n, c_i32) for n in ("ring_size", "n_idx", "rows", "heads", "win", "splits", "flags", "pad0")] + \
               [("scale", c_f32), ("pad1", c_i32)]


class IndexArgs(ctypes.Structure):
    _fields_ = [(n, c_u64) for n in ("q", "k", "ks", "w", "vis", "out", "tmax")] + [(n, c_i32) for n in ("rows", "n", "nt", "flags")]


class HcArgs(ctypes.Structure):
    _fields_ = [(n, c_u64) for n in ("x", "xo", "g", "post", "comb", "fn", "part")] + [("rs", c_i64)] + \
               [(n, c_i32) for n in ("rows", "world", "flags", "pad0")]


class GateArgs(ctypes.Structure):
    _fields_ = [(n, c_u64) for n in ("x", "w", "out")] + [(n, c_i32) for n in ("rows", "ne", "flags", "pad0")]


class RopeArgs(ctypes.Structure):
    _fields_ = [(n, c_u64) for n in ("x", "cos", "sin", "pos")] + [(n, c_i32) for n in ("rows", "h", "hd", "rd", "flags", "pad0")]


assert (ctypes.sizeof(AttnArgs), ctypes.sizeof(IndexArgs), ctypes.sizeof(HcArgs), ctypes.sizeof(GateArgs),
        ctypes.sizeof(RopeArgs)) == (152, 72, 80, 40, 56)


def bench_attn(kit, rows, n_idx, start=8192):
    """One pass (prompt chunks): the 128-key window in linear rows from wlo, then n_idx FP4 picks a row."""
    n_comp = max(start // 2, 1024)
    q = Dev(rows * HL * HD * 2, bf16((rows, HL, HD), 0.3))
    wsrc = Dev((WIN - 1 + rows) * HD * 2, bf16((WIN - 1 + rows, HD), 0.3))
    wlo = Dev(8, np.array([start - (WIN - 1)], np.int64))
    pos = Dev(rows * 8, np.arange(start, start + rows, dtype=np.int64))
    codes = Dev(n_comp * 256, rng.integers(0, 256, n_comp * 256, dtype=np.uint8))
    csc = Dev(n_comp * 32, np.full(n_comp * 32, 0x38, np.uint8))        # E4M3 1.0
    kp = (n_idx + 15) // 16 * 16
    idx = Dev(rows * max(kp, 16) * 8, rng.integers(0, n_comp, (rows, max(kp, 16)), dtype=np.int64))
    sink = Dev(HL * 4, np.zeros(HL, np.float32))
    out = Dev(rows * HL * HD * 2)
    has = n_idx > 0
    a = AttnArgs(q.ptr, wsrc.ptr, wlo.ptr, codes.ptr if has else wsrc.ptr, csc.ptr if has else wsrc.ptr,
                 idx.ptr if has else pos.ptr, pos.ptr, out.ptr, out.ptr, out.ptr, sink.ptr, out.ptr, pos.ptr, pos.ptr,
                 16, kp if has else 0, rows, HL, WIN, 1, (2 | 4 if has else 0) | 8, 0, HD ** -0.5, 0)
    f = kit.fn("tf_ds_aux_sparse_attn_h32_d512")
    ms = time_ms(lambda: launch(f, (rows, 1, 1), 256, [a]))[0]
    keys = WIN + (kp if has else 0)
    res = {"kernel": "aux.sparse_attn", "case": f"one pass, window 128 + {n_idx} FP4 picks", "rows": rows, "n_idx": n_idx,
           "ms": round(ms, 4), "tflops": round(2.0 * 2 * rows * HL * keys * HD / (ms * 1e-3) / 1e12, 2)}
    for b in (q, wsrc, wlo, pos, codes, csc, idx, sink, out):
        b.free()
    return res


def bench_index(kit, rows, n_comp):
    """The keyed indexer scores of a chunk (flags 3: int64 keys and tile maxima), in the executor's row blocks."""
    rb = max(16, min(rows, (1 << 26) // (4 * max(n_comp, 1))))
    nt = (n_comp + 63) // 64
    q = Dev(rows * IH * ID * 2, bf16((rows, IH, ID), 0.3))
    codes = Dev(n_comp * 64 + 4096, rng.integers(0, 256, n_comp * 64 + 4096, dtype=np.uint8))
    ks = Dev(n_comp * 4 + 256, np.full(n_comp * 4 + 256, 127, np.uint8))
    w = Dev(rows * IH * 2, bf16((rows, IH), 0.2))
    vis = Dev(rows * 8, np.full(rows, n_comp, np.int64))
    out = Dev(rb * n_comp * 8)
    tmax = Dev((rb * nt + rows) * 8)
    f = kit.fn("tf_ds_aux_index_rows_h32_d128")

    def call():
        r0 = 0
        while r0 < rows:
            m = min(rows, r0 + rb) - r0
            groups = (m + 7) // 8
            splits = max(1, min(nt, 240 // groups))
            a = IndexArgs(q.ptr + r0 * IH * ID * 2, codes.ptr, ks.ptr, w.ptr + r0 * IH * 2, vis.ptr + r0 * 8, out.ptr,
                          tmax.ptr, m, n_comp, nt, 3)
            launch(f, (groups, splits, 1), 256, [a])
            r0 += rb

    ms = time_ms(call)[0]
    res = {"kernel": "aux.index_rows", "case": f"keys + tile maxima, n_comp {n_comp}", "rows": rows, "n_comp": n_comp,
           "row_block": rb, "ms": round(ms, 4), "tflops": round(2.0 * rows * n_comp * IH * ID / (ms * 1e-3) / 1e12, 2),
           "keys_GBps": round(rows * n_comp * 8 / (ms * 1e-3) / 1e9, 1)}
    for b in (q, codes, ks, w, vis, out, tmax):
        b.free()
    return res


def bench_hc(kit, rows):
    x, xo = Dev(rows * HC * D * 2, bf16((rows, HC, D))), Dev(rows * HC * D * 2)
    g = Dev(2 * rows * D * 4, rng.standard_normal((2, rows, D), dtype=np.float32))
    post = Dev(rows * 4 * 4, np.ones((rows, 4), np.float32))
    comb = Dev(rows * 16 * 4, np.full((rows, 16), 0.25, np.float32))
    fn = Dev(24 * HC * D * 4, rng.standard_normal((24, HC * D), dtype=np.float32) * 0.01)
    part = Dev(rows * 40 * 32 * 4)
    a = HcArgs(x.ptr, xo.ptr, g.ptr, post.ptr, comb.ptr, fn.ptr, part.ptr, rows * D, rows, 2, 1, 0)
    f = kit.fn("tf_ds_aux_hc_mix_pf_d5120")
    ms = time_ms(lambda: launch(f, (10 * ((rows + 31) // 32),), 256, [a]))[0]
    nbytes = rows * HC * D * 2 * 2 + 2 * rows * D * 4
    res = {"kernel": "aux.hc_mix_pf", "case": "posted streams + 24 mixes, world 2", "rows": rows, "ms": round(ms, 4),
           "GBps": round(nbytes / (ms * 1e-3) / 1e9, 1)}
    for b in (x, xo, g, post, comb, fn, part):
        b.free()
    return res


def bench_gate(kit, rows, ne):
    x = Dev(rows * D * 2, bf16((rows, D)))
    w = Dev(ne * D * 2, f16((ne, D), 0.02))
    out = Dev(rows * ne * 4)
    a = GateArgs(x.ptr, w.ptr, out.ptr, rows, ne, 0, 0)
    if ne == 32:
        f, grid, block = kit.fn("tf_ds_aux_gate_logits_n32_d5120"), ((rows + 31) // 32,), 128
    else:
        f, grid, block = kit.fn("tf_ds_aux_gate_logits_d5120"), ((rows + 127) // 128, ne // 64), 256
    ms = time_ms(lambda: launch(f, grid, block, [a]))[0]
    res = {"kernel": "aux.gate_logits", "case": f"ne {ne}", "rows": rows, "ms": round(ms, 4),
           "tflops": round(2.0 * rows * D * ne / (ms * 1e-3) / 1e12, 2)}
    for b in (x, w, out):
        b.free()
    return res


def bench_rope(kit, rows, h, hd):
    x = Dev(rows * h * hd * 2, bf16((rows, h, hd)))
    cs = Dev(70000 * 32 * 4, rng.random((70000, 32), dtype=np.float32))
    pos = Dev(rows * 8, np.arange(8192, 8192 + rows, dtype=np.int64))
    a = RopeArgs(x.ptr, cs.ptr, cs.ptr, pos.ptr, rows, h, hd, 64, 0, 0)
    f = kit.fn("tf_ds_aux_rope_heads")
    ms = time_ms(lambda: launch(f, ((rows * h * 8 + 255) // 256,), 256, [a]))[0]
    res = {"kernel": "aux.rope_heads", "case": f"{h} heads of {hd}, rd 64", "rows": rows, "ms": round(ms, 4)}
    for b in (x, cs, pos):
        b.free()
    return res


# (name, K, N, k2, fp32 out, rows as a fraction of the chunk's, launches a layer): a TP2 rank's prompt linears
# (exl3_linear.zig's recorded glinear shapes; prompt.zig attention / kvSourceUpdate / indexer / engramRows)
LINEARS = [("wq_a", 5120, 1280, 8, False, 1, 1), ("wkv", 5120, 512, 8, False, 1, 1), ("wq_b", 1280, 16384, 8, False, 1, 1),
           ("wo_a (one of 4 groups)", 4096, 1024, 8, False, 1, 4), ("wo_b", 4096, 5120, 8, True, 1, 1),
           ("comp_wkv / comp_wgate", 5120, 512, 8, True, 1, 2), ("idx_wq_b", 1280, 4096, 10, False, 1, 1),
           ("engram_wkv (column half)", 3072, 12800, 8, True, 1, 1)]


def main():
    global lib
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True)
    ap.add_argument("--rows", default="512,1024,2048,4096")
    ap.add_argument("--only", default="linear,experts,invariance,attn,index,hc,gate,rope")
    ap.add_argument("--out", default=os.path.join(ROOT, "gate", "results", "prefill-dsv41"))
    ap.add_argument("--hip-lib", default=os.environ.get("TF_HIP_LIB", "/data/models/tools/therock-10.2/lib/libamdhip64.so"))
    ap.add_argument("--no-lease", action="store_true")
    ap.add_argument("--cpu-only-process", default="", help="a regex: engine processes of this name are known to run "
                    "on the CPU only and do not count as GPU users (the GTT limit still applies)")
    a = ap.parse_args()
    rows_list = [int(r) for r in a.rows.split(",")]
    only = set(a.only.split(","))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(ROOT, "gate", "interface", "host.toml"), "rb") as f:
        cfg = {"host": dict(tomllib.load(f)["host"])}
    cfg["host"]["lease_owner"] = "agent"
    cfg["host"]["lease_ttl_s"] = 1200
    ok, state = guard.gpu_free(cfg)
    if not ok and a.cpu_only_process and state["gtt_used_bytes"] <= cfg["host"]["gpu_free_gtt_bytes"]:
        import re
        ok = all(re.search(a.cpu_only_process, u.split(" ", 1)[1]) for u in state["engine_processes"])
        state["cpu_only_exempt"] = a.cpu_only_process
    if not ok:
        sys.exit(f"the GPU is not free; nothing was run: {state}")
    if guard.pod_waiters():
        sys.exit("the GPU is not free; a pod job waits in pod-gpu --wait")
    lease_log = os.path.join(a.out, "lease.txt")
    open(lease_log, "w").close()
    lease = guard.Lease(cfg, lease_log)
    doc = {"host": os.uname().nodename, "utc": guard.utc(), "kit": a.kit, "hip_lib": a.hip_lib, "rows": rows_list,
           "loadavg_start": open("/proc/loadavg").read().split()[:3], "gpu_before": state, "results": [], "checks": []}
    try:
        doc["lease"] = "none (--no-lease)" if a.no_lease else lease.acquire("strixflerp prefill kernel bench: kernels alone, no model loaded")
        lib = ctypes.CDLL(a.hip_lib)
        check(lib.hipInit(0), "hipInit")
        check(lib.hipSetDevice(0), "hipSetDevice")
        kit = Kit(a.kit)
        doc["kit_manifest_sha256"] = kit.sha256

        def rec(r, key="results"):
            doc[key].append(r)
            print(json.dumps(r), flush=True)
            with open(os.path.join(a.out, "bench.json"), "w") as f:
                json.dump(doc, f, indent=1)

        for rows in rows_list:
            if "hc" in only:
                rec(bench_hc(kit, rows))
            if "gate" in only:
                rec(bench_gate(kit, rows, 384))
                rec(bench_gate(kit, rows, 32))
            if "rope" in only:
                rec(bench_rope(kit, rows, HL, HD))
                rec(bench_rope(kit, rows, IH, ID))
            if "attn" in only:
                rec(bench_attn(kit, rows, 0))
                rec(bench_attn(kit, rows, TOPK))
            if "index" in only:
                for n_comp in (4096, 16384, 32768, 65536):
                    rec(bench_index(kit, rows, n_comp))
            if "linear" in only:
                for name, k, n, k2, f32o, _, _ in LINEARS:
                    rec(bench_linear(kit, name, k, n, k2, f32o, rows))
        if "linear" in only:      # the bits a weight: the same shape at the kit's other k2
            for k2 in (10, 12, 16):
                rec(bench_linear(kit, "wq_a", 5120, 1280, k2, False, 1024))
        free_all()
        if only & {"experts", "invariance"}:
            for k2 in (6, 4):
                ex = Experts(k2, max(rows_list + [2048]))
                if "invariance" in only:
                    for kind in ("uniform", "skew"):
                        rec(experts_row_invariance(kit, ex, 2048, kind), "checks")
                    rec(experts_row_invariance(kit, ex, 256, "uniform"), "checks")
                if "experts" in only:
                    for rows in rows_list:
                        for kind in ("uniform", "skew"):
                            rec(bench_experts(kit, ex, rows, kind))
                free_all()
        doc["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
        doc["finished"] = guard.utc()
        with open(os.path.join(a.out, "bench.json"), "w") as f:
            json.dump(doc, f, indent=1)
    finally:
        free_all()
        if not a.no_lease:
            lease.release()
    bad = [c for c in doc["checks"] if not c.get("equal")]
    print(f"done: {len(doc['results'])} timings, {len(doc['checks'])} checks, {len(bad)} failed", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
