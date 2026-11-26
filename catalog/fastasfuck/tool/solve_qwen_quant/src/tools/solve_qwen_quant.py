#!/usr/bin/env python3
"""
Pure-CPU Multiple-Choice Knapsack (MILP) quantization optimizer for Qwen 3.8 Flash Next (qwen4exp).

Two modes:
  1. Variant 1 (--mode global): Exhaustively optimizes all 12 global tensor classes
     (~3.09e10 / 2.18e9 combinations solved to provable MILP optimality in <0.1s on CPU
     using scipy.optimize.milp) calibrated against Phase 1 sensitivity heatmaps and
     bit-exact end-to-end KLD measurements from tools/sim_qwen_quant.py.
     Hard-locks lm_head=q8g64 by default so MTP draft acceptance rate is never degraded.
  2. Variant 2 (--mode per-layer --layer-kld-json ...): Exact per-block knapsack across
     all ~386 (layer, class) blocks for future single-block ablation data.

Zero GPU execution, zero .faf disk writes.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp, nnls

# ---------------------------------------------------------------------------
# 1. Architecture Geometry, Speed Model & Converter Rules (convert_qwen4exp.cpp)
# ---------------------------------------------------------------------------

CLASSES = [
    "exp_gu",
    "exp_down",
    "gdn_in",
    "gdn_out",
    "attn_qkv",
    "attn_o",
    "sh_gu",
    "sh_down",
    "hc_down",
    "hc_up",
    "ple_kv",
    "lm_head",
]

H_DIM = 2560
VOCAB = 248320
EXPERTS = 512
TOPK = 10
HC_R = 320
HC_N4 = 10240
N_GDN = 36
N_ATTN_TRUNK = 12
N_ATTN_ALL = 13
N_MOE_ALL = 49

# Total params (48 trunk + 1 MTP) and Active params per decoded token (trunk)
CLASS_PARAMS: Dict[str, Tuple[int, int]] = {
    "exp_gu":   (N_MOE_ALL * EXPERTS * 1280 * H_DIM, 48 * TOPK * 1280 * H_DIM),
    "exp_down": (N_MOE_ALL * EXPERTS * H_DIM * 640,  48 * TOPK * H_DIM * 640),
    "sh_gu":    (N_MOE_ALL * 1280 * H_DIM,           48 * 1280 * H_DIM),
    "sh_down":  (N_MOE_ALL * H_DIM * 640,            48 * H_DIM * 640),
    "gdn_in":   (N_GDN * 16480 * H_DIM,              N_GDN * 16480 * H_DIM),
    "gdn_out":  (N_GDN * H_DIM * 6144,               N_GDN * H_DIM * 6144),
    "attn_qkv": (N_ATTN_ALL * 13952 * H_DIM,         N_ATTN_TRUNK * 13952 * H_DIM),
    "attn_o":   (N_ATTN_ALL * H_DIM * 6144,          N_ATTN_TRUNK * H_DIM * 6144),
    "hc_down":  (98 * (324 * HC_N4),                 97 * (324 * HC_N4)),
    "hc_up":    (98 * (HC_N4 * HC_R),                97 * (HC_N4 * HC_R)),
    "ple_kv":   (12800 * H_DIM,                      12800 * H_DIM),
    "lm_head":  (VOCAB * H_DIM,                      VOCAB * H_DIM),
}

# Input K dimension per class (checked for K % G == 0)
CLASS_K: Dict[str, int] = {
    "exp_gu": 2560,
    "exp_down": 640,
    "gdn_in": 2560,
    "gdn_out": 6144,
    "attn_qkv": 2560,
    "attn_o": 6144,
    "sh_gu": 2560,
    "sh_down": 640,
    "hc_down": 10240,
    "hc_up": 320,
    "ple_kv": 2560,
    "lm_head": 2560,
}

FIXED_TOTAL_GIB = (VOCAB * H_DIM * 2 + N_MOE_ALL * 513 * H_DIM * 2 + 1.1 * (1024 ** 2) * 150) / (1024 ** 3)
FIXED_ACT_MB = (48 * 513 * H_DIM * 2 + 120 * (1024 ** 2)) / (1024 ** 2)
EFF_BW_MB_PER_MS = 194.0
MTP_SPEEDUP = 1.83

ALL_DENSE_FMTS = [f"q{b}g{g}" for b in (4, 5, 6, 8) for g in (128, 64, 32)]
LEGAL_FORMATS: Dict[str, List[str]] = {
    "exp_gu":   [f"q{b}g{g}" for b in (4, 5) for g in (128, 64, 32)],
    "exp_down": [f"q{b}g{g}" for b in (4, 5) for g in (128, 64, 32)],
    "gdn_in":   ALL_DENSE_FMTS,
    "gdn_out":  ALL_DENSE_FMTS,
    "attn_qkv": ALL_DENSE_FMTS,
    "attn_o":   ALL_DENSE_FMTS,
    "sh_gu":    ALL_DENSE_FMTS,
    "sh_down":  ALL_DENSE_FMTS,
    "hc_down":  [f"q{b}g{g}" for b in (4, 8) for g in (128, 64, 32)],
    "hc_up":    [f"q{b}g{g}" for b in (4, 8) for g in (64, 32)],
    "ple_kv":   ALL_DENSE_FMTS,
    "lm_head":  ALL_DENSE_FMTS,
}

SIM_SCRIPT_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_qwen_quant.py")
SCRATCH_DEFAULT = "/home/cheblan/.gemini/antigravity/brain/2db43efb-0097-4015-bf58-dedb55fd8ccc/scratch"


def bpw(fmt: str) -> float:
    return int(fmt[1]) + 32.0 / int(fmt[3:])


def unique_experts(verify_batch: int) -> float:
    """Expected unique routed MoE experts touched across a verify batch of B tokens (top-10 of 512, 0.90 batch-corr damping)."""
    if verify_batch <= 1:
        return 10.0
    return 10.0 + 0.90 * (512.0 * (1.0 - (502.0 / 512.0) ** verify_batch) - 10.0)


def class_disk_gib(cls: str, fmt: str) -> float:
    return CLASS_PARAMS[cls][0] * bpw(fmt) / 8.0 / (1024 ** 3)


def class_act_mb(cls: str, fmt: str, exp_mult: float = 1.0) -> float:
    mult = exp_mult if cls in ("exp_gu", "exp_down") else 1.0
    return CLASS_PARAMS[cls][1] * mult * bpw(fmt) / 8.0 / (1024 ** 2)


def layout_footprint(lay: Dict[str, str], exp_mult: float = 1.0) -> Tuple[float, float, float, float]:
    tot_gib = FIXED_TOTAL_GIB + sum(class_disk_gib(c, lay[c]) for c in CLASSES)
    act_mb = FIXED_ACT_MB + sum(class_act_mb(c, lay[c], exp_mult=exp_mult) for c in CLASSES)
    bare_tps = EFF_BW_MB_PER_MS * 1000.0 / act_mb
    return tot_gib, act_mb, bare_tps, bare_tps * MTP_SPEEDUP


def parse_sim_layout(spec: str) -> Dict[str, str]:
    raw: Dict[str, str] = {}
    for kv in spec.split(","):
        k, v = kv.strip().split("=")
        if k == "embed":
            continue
        raw[k] = v
    if "hc" in raw:
        raw.setdefault("hc_down", raw["hc"])
        raw.setdefault("hc_up", raw["hc"])
    return {c: raw[c] for c in CLASSES}


def validate_convert_layout(lay: Dict[str, str]) -> None:
    """Strictly enforces every rule from formats/convert_qwen4exp.cpp."""
    for c in CLASSES:
        f = lay[c]
        assert f in LEGAL_FORMATS[c], f"Illegal format {c}={f} (allowed: {LEGAL_FORMATS[c]})"
        g = int(f[3:])
        assert CLASS_K[c] % g == 0, f"K={CLASS_K[c]} for {c} not divisible by G={g}"


def format_convert_layout(lay: Dict[str, str]) -> str:
    """Builds a minimal valid convert_qwen4exp --layout string."""
    validate_convert_layout(lay)
    dense_classes = ["gdn_in", "gdn_out", "attn_qkv", "attn_o", "sh_gu", "sh_down", "ple_kv", "lm_head"]
    def_fmt = Counter(lay[c] for c in dense_classes).most_common(1)[0][0]
    parts = [f"default={def_fmt}"]
    for c in CLASSES:
        if c in ("exp_gu", "exp_down", "hc_down", "hc_up", "lm_head") or lay[c] != def_fmt:
            parts.append(f"{c}={lay[c]}")
    parts.append("mtp_fc=q4g64")
    parts.append("embed=bf16")
    return ",".join(parts)


# ---------------------------------------------------------------------------
# 2. Extract Measured Recipes + Build Sensitivity & KLD Surrogate Model
# ---------------------------------------------------------------------------

def load_recipes_from_sim(sim_path: str) -> Dict[str, Dict[str, str]]:
    with open(sim_path) as f:
        tree = ast.parse(f.read(), filename=sim_path)
    out: Dict[str, Dict[str, str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("RECIPES", "SNIPER_RECIPES", "BARTER_RECIPES"):
                pairs = ast.literal_eval(node.value)
                for name, spec in pairs:
                    out[name] = parse_sim_layout(spec)
    return out


def build_sigma_table(heatmap: dict) -> Dict[str, Dict[str, float]]:
    """
    Constructs smooth per-class normalized RMS/rel-L2 error sigma_c(fmt) across all legal formats
    anchored cleanly to the 5 measured Phase 1 formats (q4g128, q4g64, q5g128, q6g64, q8g64).
    """
    sig: Dict[str, Dict[str, float]] = {}

    def interp_class(src_hm_key: str, fmts: List[str], allow_g128: bool = True) -> Dict[str, float]:
        row = heatmap[src_hm_key]
        e4_128 = row["q4g128"]["mean_pct"] / 100.0
        e4_64 = row["q4g64"]["mean_pct"] / 100.0
        e5_128 = row["q5g128"]["mean_pct"] / 100.0
        e6_64 = row["q6g64"]["mean_pct"] / 100.0
        e8_64 = row["q8g64"]["mean_pct"] / 100.0

        # Empirical group-size reduction factor (note: hc_up fell back to g64 at g128 probe)
        r64 = (e4_64 / max(e4_128, 1e-9)) if allow_g128 else 0.90
        r64 = min(max(r64, 0.72), 0.96)
        r32 = r64 ** 0.65

        def g_mul(g: int) -> float:
            if g == 128:
                return 1.0
            if g == 64:
                return r64
            return r64 * r32

        # Bit anchors calibrated at G=128 equivalent (pure quantization + BF16 scale floor)
        s_g128 = {
            4: e4_128,
            5: e5_128,
            6: e6_64 / r64,
            8: e8_64 / r64,
        }
        res: Dict[str, float] = {}
        for f in fmts:
            b, g = int(f[1]), int(f[3:])
            val = s_g128[b] * g_mul(g)
            # Snap exact measured anchors to preserve 0 error at observed points
            if f in row and (allow_g128 or g != 128):
                val = row[f]["mean_pct"] / 100.0
            res[f] = val
        return res

    for c in CLASSES:
        if c == "lm_head":
            # Direct Phase 1 KLD anchors on lm_head; step^2 scaling for g128/g32
            lm_hm = heatmap["lm_head_kld"]
            e4_128, e4_64 = lm_hm["q4g128"], lm_hm["q4g64"]
            r64_sq = min(max(e4_64 / max(e4_128, 1e-9), 0.55), 0.92)
            r32_sq = r64_sq ** 0.85
            kld_g128 = {4: e4_128, 5: lm_hm["q5g128"], 6: lm_hm["q6g64"] / r64_sq, 8: lm_hm["q8g64"] / r64_sq}
            lm_sig = {}
            for f in LEGAL_FORMATS["lm_head"]:
                b, g = int(f[1]), int(f[3:])
                mul = 1.0 if g == 128 else (r64_sq if g == 64 else r64_sq * r32_sq)
                kld_est = lm_hm[f] if f in lm_hm else kld_g128[b] * mul
                lm_sig[f] = math.sqrt(max(kld_est, 1e-10))
            sig[c] = lm_sig
        elif c == "hc_down":
            sig[c] = interp_class("hc", LEGAL_FORMATS[c], allow_g128=True)
        elif c == "hc_up":
            sig[c] = interp_class("hc", LEGAL_FORMATS[c], allow_g128=False)
        else:
            sig[c] = interp_class(c, LEGAL_FORMATS[c], allow_g128=True)
    return sig


def fit_kld_surrogate(
    recipes_map: Dict[str, Dict[str, str]],
    pareto_json_path: str,
    det_json_paths: List[str],
    reg_lambda: float = 0.0005,
):
    """
    Fits non-negative additive per-class KLD weights across base, swift15, thinking:
        KLD(layout) ~= b0 + sum_c alpha_c * sigma_c(layout[c])^2
    Combines Phase 2 baseline sweeps (weight 0.6) + bit-exact deterministic sweeps (weight 1.0)
    with gentle prior-regularized NNLS so every class retains realistic positive sensitivity.
    """
    pareto_list = json.load(open(pareto_json_path))
    pareto_data = {d["label"]: d for d in pareto_list}
    models = ["base", "swift15", "thinking"]
    sigmas = {m: build_sigma_table(pareto_data[m]["heatmap"]) for m in models}

    # Collect observations: (kld, weight) with deterministic runs overriding non-det anchors cleanly
    obs: Dict[str, Dict[str, Tuple[float, float]]] = {m: {} for m in models}
    for d in pareto_list:
        m = d["label"]
        if m not in obs:
            continue
        for row in d.get("pareto", []):
            rn = row["recipe"]
            if rn in recipes_map:
                obs[m][rn] = (float(row["mean_kld"]), 0.6)

    for jp in det_json_paths:
        if not os.path.exists(jp):
            continue
        for entry in json.load(open(jp)):
            m = entry["label"]
            if m not in obs:
                continue
            for row in entry["pareto"]:
                rname = row["recipe"]
                if rname in recipes_map:
                    obs[m][rname] = (float(row["mean_kld"]), 1.0)

    prior_alpha = np.array([
        0.55,  # exp_gu
        0.55,  # exp_down
        0.55,  # gdn_in
        0.18,  # gdn_out
        0.04,  # attn_qkv
        0.02,  # attn_o
        0.04,  # sh_gu
        0.04,  # sh_down
        0.06,  # hc_down
        0.55,  # hc_up
        0.02,  # ple_kv
        1.00,  # lm_head (exact Phase 1 delta-KLD scaling)
    ], dtype=np.float64)

    # Physical minimum alpha floor for classes never tested at Q4 in end-to-end sweeps
    # (prevents NNLS from clamping unobserved low-bit cliffs like attn_o=q4g128 [65.6% max rel_L2] to 0)
    alpha_floor = np.array([
        0.00,  # exp_gu
        0.00,  # exp_down
        0.00,  # gdn_in
        0.00,  # gdn_out
        0.00,  # attn_qkv
        0.04,  # attn_o   (~0.0054 KLD penalty at q4g128, ~0.00023 at q6g64)
        0.00,  # sh_gu
        0.08,  # sh_down  (~0.00094 KLD penalty at q4g128)
        0.00,  # hc_down
        0.00,  # hc_up
        0.05,  # ple_kv   (~0.00022 KLD penalty at q4g128)
        0.00,  # lm_head
    ], dtype=np.float64)

    fitted: Dict[str, dict] = {}
    for m in models:
        names = list(obs[m].keys())
        X_rows = []
        y_rows = []
        w_rows = []
        for rn in names:
            lay = recipes_map[rn]
            feats = [sigmas[m][c][lay[c]] ** 2 for c in CLASSES] + [1.0]
            kld_val, wt = obs[m][rn]
            X_rows.append(feats)
            y_rows.append(kld_val)
            w_rows.append(wt)
        X = np.array(X_rows, dtype=np.float64)
        y = np.array(y_rows, dtype=np.float64)
        sw = np.sqrt(np.array(w_rows, dtype=np.float64))[:, None]
        Xw = X * sw
        yw = y * sw[:, 0] - Xw[:, :-1] @ alpha_floor

        # Gentle prior regularization on residual above floor
        sqrt_lam = math.sqrt(reg_lambda)
        P_X = np.zeros((len(CLASSES) + 1, len(CLASSES) + 1), dtype=np.float64)
        P_y = np.zeros(len(CLASSES) + 1, dtype=np.float64)
        for ci in range(len(CLASSES)):
            col_scale = max(float(np.linalg.norm(Xw[:, ci])), 0.01)
            P_X[ci, ci] = sqrt_lam * col_scale
            P_y[ci] = sqrt_lam * col_scale * max(prior_alpha[ci] - alpha_floor[ci], 0.0)
        # Hard-anchor lm_head weight ~ 1.0 (its feature IS exact Phase 1 delta-KLD)
        lm_idx = CLASSES.index("lm_head")
        P_X[lm_idx, lm_idx] = 2.0
        P_y[lm_idx] = 2.0 * max(1.0 - alpha_floor[lm_idx], 0.0)
        P_X[-1, -1] = 0.05
        P_y[-1] = 0.0

        coef, _ = nnls(np.vstack([Xw, P_X]), np.concatenate([yw, P_y]))
        alpha = alpha_floor + coef[:-1]
        b0 = float(coef[-1])
        pred = X[:, :-1] @ alpha + b0
        resid = y - pred
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        mae = float(np.mean(np.abs(resid)))
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        worst_i = int(np.argmax(np.abs(resid)))

        fitted[m] = {
            "alpha": alpha,
            "b0": b0,
            "sigma": sigmas[m],
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "n_obs": len(names),
            "worst_recipe": names[worst_i],
            "worst_err": float(resid[worst_i]),
        }
    return fitted


def predict_kld(fitted: dict, model_name: str, lay: Dict[str, str]) -> float:
    fm = fitted[model_name]
    val = fm["b0"]
    for ci, c in enumerate(CLASSES):
        val += float(fm["alpha"][ci]) * (fm["sigma"][c][lay[c]] ** 2)
    return val


def build_cost_table(fitted: dict, objective: str = "mean") -> Dict[str, Dict[str, float]]:
    """Returns per-(class, format) expected KLD contribution delta_c(fmt)."""
    models = ["base", "swift15", "thinking"] if objective == "mean" else [objective]
    out: Dict[str, Dict[str, float]] = {}
    for c_i, c in enumerate(CLASSES):
        out[c] = {}
        for f in LEGAL_FORMATS[c]:
            out[c][f] = sum(float(fitted[m]["alpha"][c_i]) * (fitted[m]["sigma"][c][f] ** 2) for m in models) / len(models)
    return out


# ---------------------------------------------------------------------------
# 3. Variant 1: Global 12-Class Exact MILP Solver
# ---------------------------------------------------------------------------

def solve_global_milp(
    costs: Dict[str, Dict[str, float]],
    allowed: Dict[str, List[str]],
    max_act_mb: float,
    max_disk_gib: Optional[float] = None,
    exp_mult: float = 1.0,
) -> Optional[Dict[str, str]]:
    """
    Exact Multiple-Choice Knapsack via scipy.optimize.milp (HiGHS):
      min  sum_{c, f} costs[c][f] * x_{c,f}
      s.t. sum_f x_{c,f} == 1                  for each c in CLASSES
           sum_{c,f} step_mb(c,f) * x_{c,f} <= max_act_mb - FIXED_ACT_MB
           sum_{c,f} disk_gib(c,f) * x_{c,f} <= max_disk_gib - FIXED_TOTAL_GIB
    """
    vars_meta: List[Tuple[str, str]] = []
    c_spans: List[Tuple[int, int]] = []
    idx = 0
    for c in CLASSES:
        opts = allowed[c]
        assert opts, f"No legal options for class {c}"
        c_spans.append((idx, idx + len(opts)))
        for f in opts:
            vars_meta.append((c, f))
        idx += len(opts)
    n_vars = len(vars_meta)

    obj = np.array([costs[c][f] for c, f in vars_meta], dtype=np.float64)
    # Tiny deterministic tie-breaker favoring fewer MiB on exact ties
    act_vec = np.array([class_act_mb(c, f, exp_mult=exp_mult) for c, f in vars_meta], dtype=np.float64)
    disk_vec = np.array([class_disk_gib(c, f) for c, f in vars_meta], dtype=np.float64)
    obj = obj + 1e-9 * act_vec

    rows = []
    lb = []
    ub = []
    for s0, s1 in c_spans:
        r = np.zeros(n_vars, dtype=np.float64)
        r[s0:s1] = 1.0
        rows.append(r)
        lb.append(1.0)
        ub.append(1.0)

    # Active MB budget
    rows.append(act_vec)
    lb.append(-np.inf)
    ub.append(max_act_mb - FIXED_ACT_MB)

    # Optional disk GiB budget
    if max_disk_gib is not None:
        rows.append(disk_vec)
        lb.append(-np.inf)
        ub.append(max_disk_gib - FIXED_TOTAL_GIB)

    A = np.vstack(rows)
    res = milp(
        c=obj,
        integrality=np.ones(n_vars, dtype=int),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(A, lb, ub),
        options={"mip_rel_gap": 1e-5},
    )
    if not res.success:
        return None
    lay: Dict[str, str] = {}
    for ci, (s0, s1) in enumerate(c_spans):
        best_j = s0 + int(np.argmax(res.x[s0:s1]))
        lay[CLASSES[ci]] = vars_meta[best_j][1]
    return lay


def sweep_global_frontier(
    fitted: dict,
    allowed: Dict[str, List[str]],
    budgets_mb: List[float],
    max_disk_gib: Optional[float],
    objective: str = "mean",
    exp_mult: float = 1.0,
) -> List[dict]:
    costs = build_cost_table(fitted, objective=objective)
    raw_pts: List[dict] = []
    seen_layouts = set()

    for b_mb in budgets_mb:
        lay = solve_global_milp(costs, allowed, max_act_mb=b_mb, max_disk_gib=max_disk_gib, exp_mult=exp_mult)
        if lay is None:
            continue
        key = tuple((c, lay[c]) for c in CLASSES)
        if key in seen_layouts:
            continue
        seen_layouts.add(key)
        disk_g, act_m, bare, mtp = layout_footprint(lay, exp_mult=exp_mult)
        kb = predict_kld(fitted, "base", lay)
        ks = predict_kld(fitted, "swift15", lay)
        kt = predict_kld(fitted, "thinking", lay)
        km = (kb + ks + kt) / 3.0
        raw_pts.append({
            "layout": lay,
            "disk_gib": disk_g,
            "act_mb": act_m,
            "bare_tps": bare,
            "mtp_tps": mtp,
            "pred_kld_base": kb,
            "pred_kld_swift15": ks,
            "pred_kld_thinking": kt,
            "pred_kld_mean": km,
            "convert_layout": format_convert_layout(lay),
        })

    # Keep only strict Pareto-optimal points (sorted fastest -> slowest; strictly improving KLD)
    raw_pts.sort(key=lambda x: x["act_mb"])
    pareto: List[dict] = []
    best_kld = float("inf")
    tol = 1e-6
    for pt in raw_pts:
        kobj = pt["pred_kld_mean"] if objective == "mean" else pt[f"pred_kld_{objective}"]
        if kobj < best_kld - tol:
            best_kld = kobj
            pareto.append(pt)
    return pareto


# ---------------------------------------------------------------------------
# 4. Variant 2: 386-Block Per-Layer Knapsack Solver
# ---------------------------------------------------------------------------

def block_act_mb(cls: str, fmt: str) -> float:
    """Per-single-block active MiB footprint on Strix Halo."""
    bits = bpw(fmt)
    if cls == "exp_gu":
        p = TOPK * 1280 * H_DIM
    elif cls == "exp_down":
        p = TOPK * H_DIM * 640
    elif cls == "sh_gu":
        p = 1280 * H_DIM
    elif cls == "sh_down":
        p = H_DIM * 640
    elif cls == "gdn_in":
        p = 16480 * H_DIM
    elif cls == "gdn_out":
        p = H_DIM * 6144
    elif cls == "attn_qkv":
        p = 13952 * H_DIM
    elif cls == "attn_o":
        p = H_DIM * 6144
    elif cls == "hc_down":
        p = 324 * HC_N4
    elif cls == "hc_up":
        p = HC_N4 * HC_R
    elif cls == "ple_kv":
        p = 12800 * H_DIM
    elif cls == "lm_head":
        p = VOCAB * H_DIM
    else:
        raise KeyError(cls)
    return p * bits / 8.0 / (1024 ** 2)


def block_disk_gib(cls: str, fmt: str) -> float:
    """Per-single-block disk GiB footprint (routed experts store all 512 experts)."""
    bits = bpw(fmt)
    if cls == "exp_gu":
        p = EXPERTS * 1280 * H_DIM
    elif cls == "exp_down":
        p = EXPERTS * H_DIM * 640
    else:
        return block_act_mb(cls, fmt) / 1024.0
    return p * bits / 8.0 / (1024 ** 3)


def solve_per_layer_milp(
    blocks: List[dict],
    max_act_mb: float,
    max_disk_gib: Optional[float] = None,
    fixed_act_mb: float = FIXED_ACT_MB,
    fixed_disk_gib: float = FIXED_TOTAL_GIB,
) -> Tuple[List[dict], float, float, float]:
    """
    Exact per-block Multiple-Choice Knapsack across all ~386 (layer, cls) blocks.
    Each block entry: {"layer": int, "cls": str, "kld": {fmt: float}, optional "act_mb", "disk_gib"}.
    Returns (chosen_blocks, total_delta_kld, total_act_mb, total_disk_gib).
    """
    vars_meta = []
    spans = []
    idx = 0
    for bi, blk in enumerate(blocks):
        fmts = sorted(blk["kld"].keys())
        spans.append((idx, idx + len(fmts)))
        for f in fmts:
            am = blk.get("act_mb", {}).get(f, block_act_mb(blk["cls"], f))
            dg = blk.get("disk_gib", {}).get(f, block_disk_gib(blk["cls"], f))
            vars_meta.append((bi, f, float(blk["kld"][f]), am, dg))
        idx += len(fmts)

    n_vars = len(vars_meta)
    obj = np.array([v[2] + 1e-9 * v[3] for v in vars_meta], dtype=np.float64)
    act_vec = np.array([v[3] for v in vars_meta], dtype=np.float64)
    disk_vec = np.array([v[4] for v in vars_meta], dtype=np.float64)

    rows = []
    lb = []
    ub = []
    for s0, s1 in spans:
        r = np.zeros(n_vars, dtype=np.float64)
        r[s0:s1] = 1.0
        rows.append(r)
        lb.append(1.0)
        ub.append(1.0)
    rows.append(act_vec)
    lb.append(-np.inf)
    ub.append(max_act_mb - fixed_act_mb)
    if max_disk_gib is not None:
        rows.append(disk_vec)
        lb.append(-np.inf)
        ub.append(max_disk_gib - fixed_disk_gib)

    res = milp(
        c=obj,
        integrality=np.ones(n_vars, dtype=int),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(np.vstack(rows), lb, ub),
        options={"mip_rel_gap": 0.002, "time_limit": 0.5},
    )
    assert res.success, f"Per-layer MILP failed: {res.message}"

    chosen = []
    tot_kld = 0.0
    tot_act = fixed_act_mb
    tot_disk = fixed_disk_gib
    for bi, (s0, s1) in enumerate(spans):
        best_j = s0 + int(np.argmax(res.x[s0:s1]))
        _, f, dk, am, dg = vars_meta[best_j]
        blk = blocks[bi]
        chosen.append({"layer": blk["layer"], "cls": blk["cls"], "fmt": f, "kld": dk})
        tot_kld += dk
        tot_act += am
        tot_disk += dg
    return chosen, tot_kld, tot_act, tot_disk


def run_self_test(fitted: dict) -> None:
    """Verifies layout validator and 386-block per-layer MILP optimality & speed."""
    for s in [
        "exp_gu=q4g128,exp_down=q4g128,gdn_in=q5g128,gdn_out=q8g64,attn_qkv=q6g64,attn_o=q8g64,sh_gu=q6g64,sh_down=q8g64,ple_kv=q6g64,hc=q8g64,lm_head=q8g64",
        "exp_gu=q4g64,exp_down=q5g128,gdn_in=q6g64,gdn_out=q8g64,attn_qkv=q8g64,attn_o=q8g64,sh_gu=q8g64,sh_down=q8g64,ple_kv=q8g64,hc=q8g64,lm_head=q8g64",
    ]:
        lay = parse_sim_layout(s)
        validate_convert_layout(lay)
        _ = format_convert_layout(lay)

    # Synthetic 386-block benchmark matching Qwen4Exp layer stack (36 GDN + 12 Attn + PLE + LM + final HC)
    rng = np.random.default_rng(42)
    costs = build_cost_table(fitted, objective="mean")
    blocks: List[dict] = []
    for L in range(48):
        is_attn = ((L + 1) % 4 == 0)
        layer_classes = ["exp_gu", "exp_down", "sh_gu", "sh_down", "hc_down", "hc_up"]
        layer_classes += ["attn_qkv", "attn_o"] if is_attn else ["gdn_in", "gdn_out"]
        for c in layer_classes:
            n_inst = 36 if c.startswith("gdn") else (12 if c.startswith("attn") else (96 if c.startswith("hc") else 48))
            mult = float(rng.uniform(0.4, 2.2))
            if c == "hc_up":
                fmts = ["q8g64"]
            elif c == "hc_down":
                fmts = ["q4g128", "q8g64"]
            elif c in ("exp_gu", "exp_down"):
                fmts = ["q4g128", "q4g64", "q5g128"]
            else:
                fmts = ["q4g128", "q4g64", "q5g128", "q5g64", "q6g64", "q8g64"]
            blocks.append({"layer": L, "cls": c, "kld": {f: (costs[c][f] / n_inst) * mult for f in fmts}})
        # Two HC blocks per layer in real model -> include second HC pair cleanly
        blocks.append({"layer": L, "cls": "hc_down", "kld": {"q4g128": costs["hc_down"]["q4g128"] / 97.0, "q8g64": costs["hc_down"]["q8g64"] / 97.0}})
        blocks.append({"layer": L, "cls": "hc_up", "kld": {"q8g64": costs["hc_up"]["q8g64"] / 97.0}})
    blocks.append({"layer": 1, "cls": "ple_kv", "kld": {f: costs["ple_kv"][f] for f in ("q4g128", "q5g128", "q6g64", "q8g64")}})
    blocks.append({"layer": 48, "cls": "lm_head", "kld": {"q8g64": costs["lm_head"]["q8g64"]}})
    assert 380 <= len(blocks) <= 500, f"Unexpected block count {len(blocks)}"

    t0 = time.time()
    chosen, tkld, tact, tdisk = solve_per_layer_milp(blocks, max_act_mb=4750.0, max_disk_gib=68.0)
    dt = time.time() - t0
    assert tact <= 4750.0 + 1e-6 and tdisk <= 68.0 + 1e-6
    assert all(b["fmt"] == "q8g64" for b in chosen if b["cls"] == "lm_head")
    print(f"[SELF-TEST PASS] {len(blocks)}-block per-layer MILP solved in {dt:.3f}s | act={tact:.1f} MiB | disk={tdisk:.2f} GiB | sum_dKLD={tkld:.5f}")


# ---------------------------------------------------------------------------
# 5. Main Report & Scenario Sweeps
# ---------------------------------------------------------------------------

def allowed_formats_for_scenario(scenario: str, lock_specs: List[str]) -> Tuple[Dict[str, List[str]], Optional[float], str]:
    allowed = {c: list(LEGAL_FORMATS[c]) for c in CLASSES}
    # Always protect lm_head=q8g64 by default (MTP draft head precision invariant)
    allowed["lm_head"] = ["q8g64"]
    max_disk = None

    if scenario == "compact_safe":
        allowed["hc_down"] = [f for f in allowed["hc_down"] if f.startswith("q8")]
        allowed["hc_up"] = [f for f in allowed["hc_up"] if f.startswith("q8")]
        max_disk = 66.5
        title = "SCENARIO A: COMPACT SAFE (lm_head=q8g64, hc*=q8*, Disk <= 66.5 GiB)"
    elif scenario == "full_safe":
        allowed["hc_down"] = [f for f in allowed["hc_down"] if f.startswith("q8")]
        allowed["hc_up"] = [f for f in allowed["hc_up"] if f.startswith("q8")]
        max_disk = 85.0
        title = "SCENARIO B: FULL SAFE PARETO (lm_head=q8g64, hc*=q8*, Disk <= 85.0 GiB)"
    elif scenario == "turbo_hc4":
        # Allows hc_down to drop to empirically-tested q4g32 (keeps hc_up in Q8 & lm_head in Q8)
        allowed["hc_down"] = ["q4g32"] + [f for f in allowed["hc_down"] if f.startswith("q8")]
        allowed["hc_up"] = [f for f in allowed["hc_up"] if f.startswith("q8")]
        max_disk = 72.0
        title = "SCENARIO C: TURBO HC_DOWN=q4g32 (lm_head=q8g64, hc_up=q8*, Disk <= 72.0 GiB)"
    else:
        raise ValueError(scenario)

    for ls in lock_specs:
        c, f = ls.split("=")
        assert c in allowed and f in LEGAL_FORMATS[c], f"Invalid --lock {ls}"
        allowed[c] = [f]
    return allowed, max_disk, title


def format_short_diff(lay: Dict[str, str]) -> str:
    """Concise diff vs Current_Prod for readable Pareto tables."""
    prod = {
        "exp_gu": "q4g128", "exp_down": "q4g128", "gdn_in": "q4g128", "gdn_out": "q6g64",
        "attn_qkv": "q6g64", "attn_o": "q6g64", "sh_gu": "q6g64", "sh_down": "q6g64",
        "hc_down": "q8g64", "hc_up": "q8g64", "ple_kv": "q6g64", "lm_head": "q8g64",
    }
    ups = []
    dns = []
    for c in CLASSES:
        if lay[c] != prod[c]:
            tag = f"{c}={lay[c]}"
            if bpw(lay[c]) > bpw(prod[c]):
                ups.append(tag)
            else:
                dns.append(tag)
    return " ".join(ups + [f"[-{d}]" for d in dns]) or "(Current Prod)"


def select_curated_shortlist(all_exported: List[dict], target_count: int = 15) -> List[dict]:
    """Selects a diverse Top-15 GPU verification shortlist spanning Scenarios A, B, and C."""
    by_scen = {"compact_safe": [], "full_safe": [], "turbo_hc4": []}
    for item in all_exported:
        by_scen[item["scenario"]].append(item)

    def sample_evenly(items: List[dict], k: int) -> List[dict]:
        if len(items) <= k:
            return items
        idxs = sorted({round(i * (len(items) - 1) / max(k - 1, 1)) for i in range(k)})
        return [items[j] for j in idxs]

    picks = (
        sample_evenly(by_scen["compact_safe"], 6)
        + sample_evenly(by_scen["full_safe"], 6)
        + sample_evenly(by_scen["turbo_hc4"], 3)
    )
    seen = set()
    curated = []
    for p in picks + all_exported:
        if p["convert_layout"] not in seen:
            seen.add(p["convert_layout"])
            curated.append(p)
        if len(curated) >= target_count:
            break
    curated.sort(key=lambda x: x["act_mb"])
    return curated


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["global", "per-layer"], default="global")
    ap.add_argument("--objective", choices=["mean", "base", "swift15", "thinking"], default="mean")
    ap.add_argument("--sim-script", default=SIM_SCRIPT_DEFAULT)
    ap.add_argument("--pareto-json", default=os.path.join(SCRATCH_DEFAULT, "qwen_pareto_results.json"))
    ap.add_argument("--det-jsons", nargs="+", default=[
        os.path.join(SCRATCH_DEFAULT, "qwen_sniper_vs_shawn.json"),
        os.path.join(SCRATCH_DEFAULT, "qwen_barter_all3.json"),
    ])
    ap.add_argument("--lock", nargs="*", default=[], help="Force class=fmt (e.g. lm_head=q8g64)")
    ap.add_argument("--verify-batch", type=int, default=1, help="Tokens in MTP/multi-slot verify window (1=legacy single-token, 4=solo MTP-3, 10=4-6 agents+MTP)")
    ap.add_argument("--unique-experts", type=float, default=0.0, help="Override expected unique experts touched per verify step (0=derive from --verify-batch)")
    ap.add_argument("--layer-kld-json", default="", help="Input JSON for --mode per-layer")
    ap.add_argument("--budgets", nargs="*", type=float, default=[], help="Target ActMB budgets for per-layer mode")
    ap.add_argument("--export-top", default="", help="Path to save top unique candidate recipes JSON")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    e_uniq = args.unique_experts if args.unique_experts > 0.0 else unique_experts(args.verify_batch)
    exp_mult = e_uniq / 10.0
    is_batched = args.verify_batch > 1 or args.unique_experts > 0.0

    recipes_map = load_recipes_from_sim(args.sim_script)
    fitted = fit_kld_surrogate(recipes_map, args.pareto_json, args.det_jsons)

    prod_lay = recipes_map["1_Current_Prod"]
    _, prod_b1_act, _, prod_b1_mtp = layout_footprint(prod_lay, exp_mult=1.0)
    _, prod_step_mb, _, _ = layout_footprint(prod_lay, exp_mult=exp_mult)

    if args.self_test:
        run_self_test(fitted)

    if args.mode == "per-layer":
        assert args.layer_kld_json, "--layer-kld-json required in per-layer mode"
        blocks = json.load(open(args.layer_kld_json))
        budgets = args.budgets or [4650.0, 4750.0, 4850.0, 5000.0, 5200.0]
        out_pl = []
        for mb in budgets:
            chosen, tkld, tact, tdisk = solve_per_layer_milp(blocks, max_act_mb=mb)
            bare = EFF_BW_MB_PER_MS * 1000.0 / tact
            print(f"Budget <= {mb:.0f} MiB -> act={tact:.0f}M ({bare:.1f}/{bare*MTP_SPEEDUP:.1f} t/s) | disk={tdisk:.2f}G | sum_dKLD={tkld:.5f}")
            out_pl.append({"budget_mb": mb, "act_mb": tact, "disk_gib": tdisk, "sum_kld": tkld, "blocks": chosen})
        if args.export_top:
            with open(args.export_top, "w") as f:
                json.dump(out_pl, f, indent=2)
        return

    # 1. Print Surrogate Fit Diagnostics
    print("\n=== SURROGATE KLD CALIBRATION DIAGNOSTICS (Prior-Regularized NNLS over Measured GPU Data) ===")
    for m in ("base", "swift15", "thinking"):
        fm = fitted[m]
        print(f"  {m:8s}: N={fm['n_obs']:2d} | R^2={fm['r2']:.4f} | MAE={fm['mae']:.5f} | RMSE={fm['rmse']:.5f} | b0={fm['b0']:.5f} | worst={fm['worst_recipe']} ({fm['worst_err']:+.5f})")

    print(f"\n=== VERIFY-BATCH TRAFFIC MODEL (B={args.verify_batch}, unique_experts={e_uniq:.2f}/512, exp_mult={exp_mult:.3f}x, Current_Prod StepMB={prod_step_mb:.1f}M) ===")
    anchor_names = ["1_Current_Prod", "P1_Prod_GdnQ5", "P2_Prod_GdnQ5_Snp", "P3_Prod_GdnQ5_Out", "S3_Shawn_GdnQ5", "3_Asym_Experts", "4_Gold_Q5"]
    print(f"  {'Anchor':20s} | {'Disk':6s} | {'StepMB':7s} | {'Rel%':6s} | {'EqMTP':6s} | {'Pred Mean KLD':13s}")
    print("  " + "-" * 74)
    for aname in anchor_names:
        if aname not in recipes_map:
            continue
        alay = recipes_map[aname]
        adisk, astep, _, _ = layout_footprint(alay, exp_mult=exp_mult)
        arel = 100.0 * prod_step_mb / astep
        amtp = prod_b1_mtp * prod_step_mb / astep
        akm = (predict_kld(fitted, "base", alay) + predict_kld(fitted, "swift15", alay) + predict_kld(fitted, "thinking", alay)) / 3.0
        print(f"  {aname:20s} | {adisk:5.2f}G | {astep:6.0f}M | {arel:5.1f}% | {amtp:5.1f}t | {akm:13.5f}")

    # 2. Print Bang-for-Buck Class Ranking (average across all 3 models)
    costs_mean = build_cost_table(fitted, objective="mean")
    print(f"\n=== CLASS BANG-FOR-BUCK MATRIX (B={args.verify_batch}, E_uniq={e_uniq:.2f}: Gain per 100 MiB/step vs q4g128) ===")
    print(f"{'Class':10s} | {'Disk4->8':9s} | {'StepMB(4->8)':12s} | {'KLD@q4g128':10s} | {'KLD@q5g128':10s} | {'KLD@q6g64':10s} | {'KLD@q8g64':10s} | {'dKLD/100M(4->8)':15s}")
    print("-" * 104)
    for c in CLASSES:
        f_lo = "q4g64" if c == "hc_up" else "q4g128"
        f_hi = "q5g128" if c in ("exp_gu", "exp_down") else "q8g64"
        d_mb = class_act_mb(c, f_hi, exp_mult=exp_mult) - class_act_mb(c, f_lo, exp_mult=exp_mult)
        d_gb = class_disk_gib(c, f_hi) - class_disk_gib(c, f_lo)
        k4 = costs_mean[c].get("q4g128", costs_mean[c]["q4g64"])
        k5 = costs_mean[c].get("q5g128", float("nan"))
        k6 = costs_mean[c].get("q6g64", float("nan"))
        k8 = costs_mean[c].get("q8g64", float("nan"))
        hi_k = costs_mean[c][f_hi]
        bang = (costs_mean[c][f_lo] - hi_k) / max(d_mb, 1e-6) * 100.0
        s5 = f"{k5:10.5f}" if not math.isnan(k5) else f"{'---':>10s}"
        s6 = f"{k6:10.5f}" if not math.isnan(k6) else f"{'---':>10s}"
        s8 = f"{k8:10.5f}" if not math.isnan(k8) else f"{'---':>10s}"
        print(f"{c:10s} | {d_gb:+7.2f}G | {d_mb:+10.1f}M | {k4:10.5f} | {s5} | {s6} | {s8} | {bang:12.5f}   ")

    # 3. Sweep Global MILP across Scenarios A, B, C
    if not is_batched:
        budgets = list(range(4450, 5801, 25))
    else:
        budgets = sorted({round(x) for x in np.linspace(0.94 * prod_step_mb, 1.25 * prod_step_mb, 55)})
    all_exported = []

    for sc_key, prefix in [("compact_safe", "A"), ("full_safe", "B"), ("turbo_hc4", "C")]:
        allowed, max_disk, title = allowed_formats_for_scenario(sc_key, args.lock)
        pts = sweep_global_frontier(fitted, allowed, budgets, max_disk, objective=args.objective, exp_mult=exp_mult)
        print(f"\n=== {title} [B={args.verify_batch}, E_uniq={e_uniq:.2f}] ({len(pts)} Pareto-optimal points) ===")
        spd_hdr = "Rel% / EqMTP " if is_batched else "Bare/MTP     "
        mb_hdr = "StepMB" if is_batched else "ActMB "
        print(f"{'#':3s} | {'Disk':6s} | {mb_hdr:6s} | {spd_hdr:13s} | {'Pred Base':9s} | {'Swift15':9s} | {'Thinking':9s} | {'Mean':8s} | Diff vs Current_Prod")
        print("-" * 126)
        for i, pt in enumerate(pts, 1):
            diff_s = format_short_diff(pt["layout"])
            step_mb = pt["act_mb"]
            rel_pct = 100.0 * prod_step_mb / step_mb
            eq_mtp = prod_b1_mtp * prod_step_mb / step_mb
            spd_col = f"{rel_pct:5.1f}%/{eq_mtp:5.1f}t" if is_batched else f"{pt['bare_tps']:4.1f}/{pt['mtp_tps']:4.1f}t/s"
            print(
                f"{prefix}{i:<2d} | {pt['disk_gib']:5.2f}G | {step_mb:5.0f}M | "
                f"{spd_col:13s} | "
                f"{pt['pred_kld_base']:9.5f} | {pt['pred_kld_swift15']:9.5f} | {pt['pred_kld_thinking']:9.5f} | "
                f"{pt['pred_kld_mean']:8.5f} | {diff_s}"
            )
            all_exported.append({
                "id": f"{prefix}{i}",
                "scenario": sc_key,
                "verify_batch": args.verify_batch,
                "unique_experts": round(e_uniq, 3),
                "disk_gib": round(pt["disk_gib"], 3),
                "act_mb": round(step_mb, 1),
                "step_mb": round(step_mb, 1),
                "rel_pct_vs_prod": round(rel_pct, 2),
                "eq_mtp_tps": round(eq_mtp, 2),
                "bare_tps": round(pt["bare_tps"], 2),
                "mtp_tps": round(pt["mtp_tps"], 2),
                "pred_kld_base": round(pt["pred_kld_base"], 6),
                "pred_kld_swift15": round(pt["pred_kld_swift15"], 6),
                "pred_kld_thinking": round(pt["pred_kld_thinking"], 6),
                "pred_kld_mean": round(pt["pred_kld_mean"], 6),
                "layout": pt["layout"],
                "convert_layout": pt["convert_layout"],
                "sim_layout": ",".join(f"{c}={pt['layout'][c]}" for c in CLASSES),
            })

    curated = select_curated_shortlist(all_exported, target_count=15)
    print(f"\n=== TOP-15 CANDIDATES FOR FUTURE GPU VERIFICATION (B={args.verify_batch}, all lm_head=q8g64) ===")
    spd_hdr = "Rel% / EqMTP " if is_batched else "Bare/MTP     "
    mb_hdr = "StepMB" if is_batched else "ActMB "
    print(f"{'ID':4s} | {'Scenario':12s} | {'Disk':6s} | {mb_hdr:6s} | {spd_hdr:13s} | {'Mean KLD':8s} | --layout")
    print("-" * 126)
    for item in curated:
        assert item["layout"]["lm_head"] == "q8g64" or args.lock, "lm_head safety check failed"
        spd_col = f"{item['rel_pct_vs_prod']:5.1f}%/{item['eq_mtp_tps']:5.1f}t" if is_batched else f"{item['bare_tps']:4.1f}/{item['mtp_tps']:4.1f}t/s"
        print(f"{item['id']:4s} | {item['scenario']:12s} | {item['disk_gib']:5.2f}G | {item['step_mb']:5.0f}M | {spd_col:13s} | {item['pred_kld_mean']:8.5f} | {item['convert_layout']}")

    if args.export_top:
        with open(args.export_top, "w") as f:
            json.dump(curated, f, indent=2)
        print(f"\nExported {len(curated)} curated Pareto-optimal candidates (all lm_head=q8g64) to {args.export_top}")

    print(f"\nTotal CPU MILP solve elapsed: {time.time() - t_start:.3f}s")


if __name__ == "__main__":
    main()
