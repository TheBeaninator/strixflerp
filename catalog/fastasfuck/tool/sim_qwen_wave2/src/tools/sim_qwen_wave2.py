#!/usr/bin/env python3
"""
Round 4 (Wave-2) Zero-Speed-Penalty .faf-Native Calibration Simulation for Qwen 3.8-FN.

Evaluates 6 strictly 0-kernel-edit / 0-speed-penalty variants on the exact disjoint
61,440-token calibration corpus (40 x 1536, sha=ddc381dad2eaae6a) and held-out CALIB_DOC:
  0_RTN_Prod            : Pure RTN control (q4g128 + q6g64)
  1_R3_ActOrder_Q4Only  : Round 3 Champion (ActOrder static_groups, damp=0.05, tau=384)
  2_R3_plus_Q6_Hotspots : Var 1 + ActOrder static_groups (damp=0.05, no clip) on q6g64 output hotspots
                          (gdn_out, attn_o, sh_down)
  3_R3_plus_CD_Refit    : Var 1 + 2 rounds [Block-128 Exact CD -> Exact Analytical BF16 (s,m) Refit]
  4_R3_plus_Q6_CD_Refit : Var 3 on q4g128 + Var 2 on q6g64 output hotspots
  5_GPTAQ_ActOrder_CD   : Asymmetric GPTAQ (alpha_aq=0.25, damp_aq=0.10) + ActOrder + CD + BF16 Refit
                          on q4g128 (gdn_in, exp_gu, exp_down Shrink512) + Var 2 on q6g64 hotspots

Untouched vs 1_Current_Prod / R3: hc_down/hc_up (q8g64), lm_head (q8g64), ple_kv/attn_qkv/sh_gu (RTN q6g64),
router weights (BF16), mtp.layers.0 (untouched).
"""

import os
os.environ.setdefault("ROCP_HSA_INTERCEPT", "0")
os.environ.setdefault("ROCP_TOOLS_LIB", "")
os.environ.setdefault("HSA_TOOLS_LIB", "")

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_qwen_quant as sq
import sim_qwen_calib as sc
import sim_qwen_awq_gptq as sag
import sim_qwen_gptq_r3 as r3


VARIANT_NAMES = [
    "0_RTN_Prod",
    "1_R3_ActOrder_Q4Only",
    "2_R3_plus_Q6_Hotspots",
    "3_R3_plus_CD_Refit",
    "4_R3_plus_Q6_CD_Refit",
    "5_GPTAQ_ActOrder_CD",
]


# ---------------------------------------------------------------------------
# 1. Exact .faf Q4 / Q6 Pack & Dequant Roundtrip Helpers
# ---------------------------------------------------------------------------

def pack_q4_components(codes_u8: torch.Tensor, s_bf16: torch.Tensor, m_bf16: torch.Tensor):
    c = codes_u8.to(torch.uint8)
    packed = (c[:, 0::2] | (c[:, 1::2] << 4)).contiguous()
    return packed, s_bf16.to(torch.bfloat16).contiguous(), m_bf16.to(torch.bfloat16).contiguous()


def dequant_q4_components(q_u8: torch.Tensor, s_bf16: torch.Tensor, m_bf16: torch.Tensor, G: int = 128) -> torch.Tensor:
    R, K2 = q_u8.shape
    K = K2 * 2
    lo = (q_u8 & 0x0F).to(torch.float32)
    hi = ((q_u8 >> 4) & 0x0F).to(torch.float32)
    codes = torch.stack([lo, hi], dim=-1).reshape(R, K)
    sf = s_bf16.float().repeat_interleave(G, dim=1)
    mf = m_bf16.float().repeat_interleave(G, dim=1)
    return (codes * sf + mf).to(torch.bfloat16)


def pack_q6_components(codes_u8: torch.Tensor, s_bf16: torch.Tensor, m_bf16: torch.Tensor):
    """Packs 4 x 6-bit unsigned codes (0..63) per group of 4 columns into 3 bytes."""
    R, K = codes_u8.shape
    assert K % 64 == 0
    c = codes_u8.to(torch.uint8).view(R, K // 4, 4)
    c0, c1, c2, c3 = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    b0 = (c0 | ((c1 & 0x03) << 6)).contiguous()
    b1 = (((c1 >> 2) & 0x0F) | ((c2 & 0x0F) << 4)).contiguous()
    b2 = (((c2 >> 4) & 0x03) | (c3 << 2)).contiguous()
    packed = torch.stack([b0, b1, b2], dim=-1).reshape(R, (K // 4) * 3).contiguous()
    return packed, s_bf16.to(torch.bfloat16).contiguous(), m_bf16.to(torch.bfloat16).contiguous()


def dequant_q6_components(q_u8: torch.Tensor, s_bf16: torch.Tensor, m_bf16: torch.Tensor, G: int = 64) -> torch.Tensor:
    R, B3 = q_u8.shape
    n4 = B3 // 3
    K = n4 * 4
    b = q_u8.view(R, n4, 3)
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    c0 = b0 & 0x3F
    c1 = ((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2)
    c2 = ((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4)
    c3 = (b2 >> 2) & 0x3F
    codes = torch.stack([c0, c1, c2, c3], dim=-1).reshape(R, K).to(torch.float32)
    sf = s_bf16.float().repeat_interleave(G, dim=1)
    mf = m_bf16.float().repeat_interleave(G, dim=1)
    return (codes * sf + mf).to(torch.bfloat16)


def dequant_from_params(codes_u8: torch.Tensor, s_bf16: torch.Tensor, m_bf16: torch.Tensor, G: int) -> torch.Tensor:
    sf = s_bf16.float().repeat_interleave(G, dim=-1)
    mf = m_bf16.float().repeat_interleave(G, dim=-1)
    return (codes_u8.float() * sf + mf).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# 2. Batched Full Attention Forward (with optional o_gated capture)
# ---------------------------------------------------------------------------

def full_attn_forward_batched(u_b, w_qkv, w_o, q_norm, k_norm, cos, sin, return_o_gated: bool = False):
    B, L, _ = u_b.shape
    qkv = F.linear(u_b, w_qkv)
    qg = qkv[..., :12288].view(B, L, 24, 512)
    q, gate = qg[..., :256], qg[..., 256:]
    k = qkv[..., 12288:12800].view(B, L, 2, 256)
    v = qkv[..., 12800:13312].view(B, L, 2, 256)
    q = sq.rope_rotate_half(sq.rmsnorm_zc(q, q_norm, 1e-6).transpose(1, 2), cos, sin)
    k = sq.rope_rotate_half(sq.rmsnorm_zc(k, k_norm, 1e-6).transpose(1, 2), cos, sin).repeat_interleave(12, dim=1)
    v = v.transpose(1, 2).repeat_interleave(12, dim=1)
    attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=256.0 ** -0.5).transpose(1, 2)
    out = (attn.float() * torch.sigmoid(gate.float())).to(u_b.dtype).reshape(B, L, 6144)
    ym = F.linear(out, w_o)
    if return_o_gated:
        return ym, out
    return ym


# ---------------------------------------------------------------------------
# 3. Static-Group ActOrder GPTQ Returning Exact (.q, .scale, .minv) Params
# ---------------------------------------------------------------------------

def _fmt_spec(fmt: str):
    if fmt == "q4g128":
        return 128, 15.0
    if fmt == "q4g64":
        return 64, 15.0
    if fmt == "q6g64":
        return 64, 63.0
    raise ValueError(f"Unsupported format {fmt}")


def quant_gptq_actorder_static_with_params(
    w_in: torch.Tensor,
    H_cpu_f64: torch.Tensor = None,
    precomp=None,
    fmt: str = "q4g128",
    damp: float = 0.05,
    row_subbatch: int = 40960,
):
    """
    Runs ActOrder GPTQ with strictly static_groups=True (zero g_idx), returning
    (Q_bf16, codes_u8 [..., K], s_bf16 [..., K//G], m_bf16 [..., K//G]).
    Matches r3.quant_gptq_actorder_static bit-for-bit on both Q4 and Q6.
    """
    G, levels = _fmt_spec(fmt)
    dev = w_in.device
    orig_shape = w_in.shape
    W_all = w_in.reshape(-1, orig_shape[-1]).float()
    R, K = W_all.shape
    ng = K // G

    if precomp is None:
        U_p_cpu, perm_cpu, invperm_cpu = r3.prepare_actorder_chol_cpu(H_cpu_f64, damp=damp)
    else:
        U_p_cpu, perm_cpu, invperm_cpu = precomp

    U_p = U_p_cpu.to(dev)
    perm = perm_cpu.to(dev)
    invperm = invperm_cpu.to(dev)

    Q_out = torch.empty((R, K), device=dev, dtype=torch.bfloat16)
    codes_out = torch.empty((R, K), device=dev, dtype=torch.uint8)
    s_out = torch.empty((R, ng), device=dev, dtype=torch.bfloat16)
    m_out = torch.empty((R, ng), device=dev, dtype=torch.bfloat16)

    for r0 in range(0, R, row_subbatch):
        r1 = min(r0 + row_subbatch, R)
        W_sb = W_all[r0:r1]
        s_vec, m_vec, safe_s = sag._rtn_minmax_params(W_sb.view(-1, ng, G), levels)
        s_out[r0:r1] = s_vec.to(torch.bfloat16)
        m_out[r0:r1] = m_vec.to(torch.bfloat16)

        s_p = s_vec.repeat_interleave(G, dim=1)[:, perm]
        m_p = m_vec.repeat_interleave(G, dim=1)[:, perm]
        safe_s_p = safe_s.repeat_interleave(G, dim=1)[:, perm]

        Wp = W_sb[:, perm].clone()
        codes_p = torch.empty((r1 - r0, K), device=dev, dtype=torch.uint8)
        Qp = torch.empty_like(Wp)

        for i1 in range(0, K, 128):
            i2 = min(i1 + 128, K)
            bs = i2 - i1
            W_blk_T = Wp[:, i1:i2].T.contiguous()
            s_blk_T = s_p[:, i1:i2].T.contiguous()
            m_blk_T = m_p[:, i1:i2].T.contiguous()
            safe_s_blk_T = safe_s_p[:, i1:i2].T.contiguous()
            Err_blk_T = torch.zeros_like(W_blk_T)
            codes_blk_T = torch.empty((bs, r1 - r0), device=dev, dtype=torch.uint8)
            Q_blk_T = torch.empty_like(W_blk_T)
            U_blk = U_p[i1:i2, i1:i2].contiguous()
            for j in range(bs):
                wj = W_blk_T[j]
                sj = s_blk_T[j]
                mj = m_blk_T[j]
                code = torch.clamp(torch.round((wj - mj) / safe_s_blk_T[j]), 0.0, levels)
                code = torch.where(sj > 0.0, code, torch.zeros_like(code))
                codes_blk_T[j] = code.to(torch.uint8)
                qj = mj + code * sj
                Q_blk_T[j] = qj
                err = (wj - qj) / U_blk[j, j]
                Err_blk_T[j] = err
                if j + 1 < bs:
                    W_blk_T[j + 1 :].addcmul_(-U_blk[j, j + 1 :].unsqueeze(1), err.unsqueeze(0))
            codes_p[:, i1:i2] = codes_blk_T.T
            Qp[:, i1:i2] = Q_blk_T.T
            if i2 < K:
                Wp[:, i2:].addmm_(Err_blk_T.T, U_p[i1:i2, i2:], alpha=-1.0)

        codes_orig = codes_p[:, invperm]
        codes_out[r0:r1] = codes_orig
        Q_out[r0:r1] = Qp[:, invperm].to(torch.bfloat16)
        del W_blk_T, s_blk_T, m_blk_T, safe_s_blk_T, Err_blk_T, codes_blk_T, Q_blk_T, Wp, codes_p, Qp, s_p, m_p, safe_s_p

    del U_p, perm, invperm
    return (
        Q_out.reshape(orig_shape),
        codes_out.reshape(orig_shape),
        s_out.reshape(orig_shape[:-1] + (ng,)),
        m_out.reshape(orig_shape[:-1] + (ng,)),
    )


# ---------------------------------------------------------------------------
# 4. Block-128 Coordinate Descent + Exact Analytical BF16 (scale, minv) Refit
# ---------------------------------------------------------------------------

def cd_and_bf16_refit_subbatch(
    W_sb: torch.Tensor,          # [r_sb, K] float32 target weights
    codes_sb: torch.Tensor,      # [r_sb, K] uint8 (modified in place)
    s_sb: torch.Tensor,          # [r_sb, ng] bfloat16 (modified in place)
    m_sb: torch.Tensor,          # [r_sb, ng] bfloat16 (modified in place)
    H_gpu: torch.Tensor,         # [K, K] float32 symmetric PSD
    G: int = 128,
    levels: float = 15.0,
    n_rounds: int = 2,
):
    """
    Runs n_rounds of [Phase A: Block-128 Exact CD] -> [Phase B: Exact Analytical BF16 (s,m) Refit]
    with clean Grad = (Q - W) @ H synchronization between every phase and round.
    """
    r_sb, K = W_sb.shape
    ng = K // G
    blocksize = 128
    H_sym = 0.5 * (H_gpu + H_gpu.T)
    Q_f = (m_sb.float().repeat_interleave(G, dim=1) + codes_sb.float() * s_sb.float().repeat_interleave(G, dim=1))
    inv_diag_ok = H_sym.diagonal() > 1e-12
    H_diag_safe = torch.where(inv_diag_ok, H_sym.diagonal(), torch.ones_like(H_sym.diagonal()))
    h_diag_list = H_diag_safe.cpu().tolist()
    ok_list = inv_diag_ok.cpu().tolist()
    A00_list = [float(H_sym[g * G : (g + 1) * G, g * G : (g + 1) * G].double().sum().item()) for g in range(ng)]

    for _ in range(n_rounds):
        # ---- Phase A: Block-128 Exact Coordinate Descent (coalesced [G, r_sb] blocks) ----
        G_mat = (Q_f - W_sb) @ H_sym

        for g in range(ng):
            b0, b1 = g * G, (g + 1) * G
            G_blk_T = G_mat[:, b0:b1].T.contiguous()
            c_blk_T = codes_sb[:, b0:b1].T.contiguous()
            d_blk_T = torch.zeros((G, r_sb), device=W_sb.device, dtype=torch.float32)
            H_blk = H_sym[b0:b1, b0:b1].contiguous()
            s_g = s_sb[:, g].float().contiguous()
            s_pos = s_g > 0.0
            inv_s_g = torch.where(s_pos, 1.0 / torch.where(s_pos, s_g, torch.ones_like(s_g)), torch.zeros_like(s_g))
            for j in range(G):
                k = b0 + j
                if not ok_list[k]:
                    continue
                h_kk = h_diag_list[k]
                g_k = G_blk_T[j]
                c_old = c_blk_T[j].float()
                c_new = torch.clamp(torch.round(c_old - g_k * (inv_s_g / h_kk)), 0.0, levels)
                dq = torch.where(s_pos, (c_new - c_old) * s_g, torch.zeros_like(s_g))
                acc = (dq * (2.0 * g_k + h_kk * dq)) < -1e-12
                dq = torch.where(acc, dq, torch.zeros_like(dq))
                c_blk_T[j] = torch.where(acc, c_new.to(torch.uint8), c_blk_T[j])
                d_blk_T[j] = dq
                if j + 1 < G:
                    G_blk_T[j + 1 :].addcmul_(H_blk[j, j + 1 :].unsqueeze(1), dq.unsqueeze(0))
            d_blk = d_blk_T.T
            codes_sb[:, b0:b1] = c_blk_T.T
            Q_f[:, b0:b1] += d_blk
            if b1 < K:
                G_mat[:, b1:].addmm_(d_blk, H_sym[b0:b1, b1:])

        del G_mat

        # ---- Phase B: Exact Analytical BF16 (scale, minv) Refit (FP32 matmul + FP64 2x2) ----
        G_mat = (Q_f - W_sb) @ H_sym
        for g in range(ng):
            A00 = A00_list[g]
            if A00 <= 1e-12:
                continue
            k0, k1 = g * G, (g + 1) * G
            H_gg = H_sym[k0:k1, k0:k1]
            c_g = codes_sb[:, k0:k1].float()
            Hc = c_g @ H_gg
            A01 = Hc.sum(dim=1).double()
            A11 = (Hc * c_g).sum(dim=1).double()
            g_g = G_mat[:, k0:k1]
            b0_v = -g_g.sum(dim=1).double()
            b1_v = -(g_g * c_g).sum(dim=1).double()

            det = A00 * A11 - A01 * A01
            well = (det > 1e-12 * A00 * A11.clamp(min=1e-12)) & (A11 > 1e-12)
            det_safe = torch.where(well, det, torch.ones_like(det))
            dm_star = torch.where(well, (b0_v * A11 - b1_v * A01) / det_safe, b0_v / A00)
            ds_star = torch.where(well, (A00 * b1_v - A01 * b0_v) / det_safe, torch.zeros_like(det))

            m_old_f = m_sb[:, g].double()
            s_old_f = s_sb[:, g].double()
            m_cand = (m_old_f.float() + dm_star.float()).to(torch.bfloat16)
            s_cand = torch.clamp(s_old_f.float() + ds_star.float(), min=0.0).to(torch.bfloat16)

            dm = m_cand.double() - m_old_f
            ds = s_cand.double() - s_old_f
            dL = -2.0 * (b0_v * dm + b1_v * ds) + A00 * dm * dm + 2.0 * A01 * dm * ds + A11 * ds * ds
            acc = dL < -1e-12
            m_sb[:, g] = torch.where(acc, m_cand, m_sb[:, g])
            s_sb[:, g] = torch.where(acc, s_cand, s_sb[:, g])
            dm_f = torch.where(acc, dm, torch.zeros_like(dm)).float()
            ds_f = torch.where(acc, ds, torch.zeros_like(ds)).float()
            dq_g = dm_f.unsqueeze(1) + ds_f.unsqueeze(1) * c_g
            Q_f[:, k0:k1] += dq_g
            if k1 < K:
                G_mat[:, k1:].addmm_(dq_g, H_sym[k0:k1, k1:])
        del G_mat

    return dequant_from_params(codes_sb, s_sb, m_sb, G)


def cd_refit_from_params(
    w_in: torch.Tensor,
    codes_in: torch.Tensor,
    s_in: torch.Tensor,
    m_in: torch.Tensor,
    H_cd_cpu_f64: torch.Tensor,
    fmt: str = "q4g128",
    row_subbatch: int = 40960,
    n_rounds: int = 2,
) -> torch.Tensor:
    """Runs CD + BF16 Refit starting from existing (codes, s, m) without re-running ActOrder."""
    G, levels = _fmt_spec(fmt)
    dev = w_in.device
    orig_shape = w_in.shape
    W_all = w_in.reshape(-1, orig_shape[-1]).float()
    R, K = W_all.shape
    ng = K // G

    codes_all = codes_in.reshape(R, K).clone()
    s_all = s_in.reshape(R, ng).clone()
    m_all = m_in.reshape(R, ng).clone()

    H_gpu = H_cd_cpu_f64.float().to(dev)
    md = max(float(H_gpu.diagonal().mean().item()), 1e-8)
    H_gpu = 0.5 * (H_gpu + H_gpu.T)
    if float(H_gpu.diagonal().min().item()) <= 1e-12:
        H_gpu.diagonal().add_(1e-6 * md)

    Q_out = torch.empty((R, K), device=dev, dtype=torch.bfloat16)
    cd_sb = min(row_subbatch, 40960)
    for r0 in range(0, R, cd_sb):
        r1 = min(r0 + cd_sb, R)
        Q_out[r0:r1] = cd_and_bf16_refit_subbatch(
            W_all[r0:r1], codes_all[r0:r1], s_all[r0:r1], m_all[r0:r1],
            H_gpu, G=G, levels=levels, n_rounds=n_rounds,
        )
    del H_gpu, codes_all, s_all, m_all
    return Q_out.reshape(orig_shape)


def quant_gptq_actorder_cd_refit(
    w_in: torch.Tensor,
    H_cpu_f64: torch.Tensor = None,
    precomp=None,
    H_cd_cpu_f64: torch.Tensor = None,
    fmt: str = "q4g128",
    damp: float = 0.05,
    row_subbatch: int = 40960,
    n_rounds: int = 2,
) -> torch.Tensor:
    """ActOrder static_groups GPTQ followed by n_rounds of [Block-128 CD -> BF16 Refit]."""
    _, codes_all, s_all, m_all = quant_gptq_actorder_static_with_params(
        w_in, H_cpu_f64=H_cpu_f64, precomp=precomp, fmt=fmt, damp=damp, row_subbatch=row_subbatch
    )
    H_src = H_cd_cpu_f64 if H_cd_cpu_f64 is not None else H_cpu_f64
    return cd_refit_from_params(w_in, codes_all, s_all, m_all, H_src, fmt=fmt, row_subbatch=row_subbatch, n_rounds=n_rounds)


def cd_and_bf16_refit_batched(
    W_b: torch.Tensor,        # [C, R, K] float32
    codes_b: torch.Tensor,    # [C, R, K] uint8 (modified in place)
    s_b: torch.Tensor,        # [C, R, ng] bfloat16 (modified in place)
    m_b: torch.Tensor,        # [C, R, ng] bfloat16 (modified in place)
    H_b: torch.Tensor,        # [C, K, K] float32 symmetric PSD
    G: int = 128,
    levels: float = 15.0,
    n_rounds: int = 2,
) -> torch.Tensor:
    C, R, K = W_b.shape
    ng = K // G
    blocksize = 128
    H_sym = 0.5 * (H_b + H_b.transpose(1, 2))
    diag_H = H_sym.diagonal(dim1=1, dim2=2)  # [C, K]
    diag_ok = diag_H > 1e-12
    diag_safe = torch.where(diag_ok, diag_H, torch.ones_like(diag_H))

    Q_f = m_b.float().repeat_interleave(G, dim=2) + codes_b.float() * s_b.float().repeat_interleave(G, dim=2)

    for _ in range(n_rounds):
        G_mat = torch.bmm(Q_f - W_b, H_sym)

        for g in range(ng):
            b0, b1 = g * G, (g + 1) * G
            G_blk_T = G_mat[:, :, b0:b1].transpose(1, 2).contiguous()       # [C, G, R]
            c_blk_T = codes_b[:, :, b0:b1].transpose(1, 2).contiguous()    # [C, G, R]
            d_blk_T = torch.zeros((C, G, R), device=W_b.device, dtype=torch.float32)
            H_blk = H_sym[:, b0:b1, b0:b1].contiguous()                    # [C, G, G]
            s_g = s_b[:, :, g].float().contiguous()                        # [C, R]
            s_pos = s_g > 0.0
            inv_s_g = torch.where(s_pos, 1.0 / torch.where(s_pos, s_g, torch.ones_like(s_g)), torch.zeros_like(s_g))
            for j in range(G):
                k = b0 + j
                ok_k = diag_ok[:, k].unsqueeze(1)                          # [C, 1]
                h_kk = diag_safe[:, k].unsqueeze(1)                        # [C, 1]
                g_k = G_blk_T[:, j]                                        # [C, R]
                c_old = c_blk_T[:, j].float()
                c_new = torch.clamp(torch.round(c_old - g_k * (inv_s_g / h_kk)), 0.0, levels)
                dq = torch.where(s_pos & ok_k, (c_new - c_old) * s_g, torch.zeros_like(s_g))
                acc = (dq * (2.0 * g_k + h_kk * dq)) < -1e-12
                dq = torch.where(acc, dq, torch.zeros_like(dq))
                c_blk_T[:, j] = torch.where(acc, c_new.to(torch.uint8), c_blk_T[:, j])
                d_blk_T[:, j] = dq
                if j + 1 < G:
                    G_blk_T[:, j + 1 :].addcmul_(H_blk[:, j, j + 1 :].unsqueeze(2), dq.unsqueeze(1))
            d_blk = d_blk_T.transpose(1, 2)
            codes_b[:, :, b0:b1] = c_blk_T.transpose(1, 2)
            Q_f[:, :, b0:b1] += d_blk
            if b1 < K:
                G_mat[:, :, b1:].baddbmm_(d_blk, H_sym[:, b0:b1, b1:])
        del G_mat

        # Phase B: Exact Analytical BF16 (s, m) Refit (FP32 bmm + FP64 2x2 scalars)
        G_mat = torch.bmm(Q_f - W_b, H_sym)
        for g in range(ng):
            k0, k1 = g * G, (g + 1) * G
            H_gg = H_sym[:, k0:k1, k0:k1]
            A00 = H_gg.double().sum(dim=(1, 2)).unsqueeze(1)  # [C, 1]
            c_g = codes_b[:, :, k0:k1].float()
            Hc = torch.bmm(c_g, H_gg)
            A01 = Hc.sum(dim=-1).double()
            A11 = (Hc * c_g).sum(dim=-1).double()
            g_g = G_mat[:, :, k0:k1]
            b0_v = -g_g.sum(dim=-1).double()
            b1_v = -(g_g * c_g).sum(dim=-1).double()

            det = A00 * A11 - A01 * A01
            well = (det > 1e-12 * A00 * A11.clamp(min=1e-12)) & (A11 > 1e-12) & (A00 > 1e-12)
            det_safe = torch.where(well, det, torch.ones_like(det))
            dm_star = torch.where(well, (b0_v * A11 - b1_v * A01) / det_safe, torch.where(A00 > 1e-12, b0_v / A00.clamp(min=1e-12), torch.zeros_like(det)))
            ds_star = torch.where(well, (A00 * b1_v - A01 * b0_v) / det_safe, torch.zeros_like(det))

            m_old_f = m_b[:, :, g].double()
            s_old_f = s_b[:, :, g].double()
            m_cand = (m_old_f.float() + dm_star.float()).to(torch.bfloat16)
            s_cand = torch.clamp(s_old_f.float() + ds_star.float(), min=0.0).to(torch.bfloat16)

            dm = m_cand.double() - m_old_f
            ds = s_cand.double() - s_old_f
            dL = -2.0 * (b0_v * dm + b1_v * ds) + A00 * dm * dm + 2.0 * A01 * dm * ds + A11 * ds * ds
            acc = (dL < -1e-12) & (A00 > 1e-12)
            m_b[:, :, g] = torch.where(acc, m_cand, m_b[:, :, g])
            s_b[:, :, g] = torch.where(acc, s_cand, s_b[:, :, g])
            dm_f = torch.where(acc, dm, torch.zeros_like(dm)).float()
            ds_f = torch.where(acc, ds, torch.zeros_like(ds)).float()
            dq_g = dm_f.unsqueeze(-1) + ds_f.unsqueeze(-1) * c_g
            Q_f[:, :, k0:k1] += dq_g
            if k1 < K:
                G_mat[:, :, k1:].baddbmm_(dq_g, H_sym[:, k0:k1, k1:])
        del G_mat

    return dequant_from_params(codes_b, s_b, m_b, G)


def quant_gptq_batched_exp_down_shrink_wave2(
    W_dn: torch.Tensor,          # [cnt, N, K] target weights
    H_e_chunk: torch.Tensor,     # [cnt, K, K]
    hits_chunk: torch.Tensor,    # [cnt]
    H_mean: torch.Tensor,        # [K, K]
    tau: float = 384.0,
    base_damp: float = 0.05,
    act_order: bool = True,
    cd_refit: bool = False,
    cd_n_rounds: int = 2,
    fmt: str = "q4g128",
) -> torch.Tensor:
    G, levels = _fmt_spec(fmt)
    cnt, N, K = W_dn.shape
    dev = W_dn.device
    ng = K // G

    H_e_cpu = H_e_chunk.double().cpu()
    H_m_cpu = H_mean.double().cpu()
    hits_list = hits_chunk.tolist()

    H_reg_list, U_list, perm_list, invperm_list = [], [], [], []
    for k_i, h_cnt in enumerate(hits_list):
        h_f = float(h_cnt)
        H_reg = (h_f * H_e_cpu[k_i] + tau * H_m_cpu) / (h_f + tau) if h_f > 0.0 else H_m_cpu.clone()
        H_reg = 0.5 * (H_reg + H_reg.T)
        H_reg_list.append(H_reg.float())
        if act_order:
            U_p, perm_k, invperm_k = r3.prepare_actorder_chol_cpu(H_reg, damp=base_damp)
            U_list.append(U_p)
            perm_list.append(perm_k)
            invperm_list.append(invperm_k)
        else:
            U_list.append(sc.compute_hinv_chol_cpu(H_reg, damp=base_damp))

    U_b = torch.stack(U_list, dim=0).to(dev)
    W_orig = W_dn.float()

    if not act_order:
        W_b = W_orig.clone()
        codes = torch.empty((cnt, N, K), device=dev, dtype=torch.uint8)
        s_grp = torch.empty((cnt, N, ng), device=dev, dtype=torch.bfloat16)
        m_grp = torch.empty((cnt, N, ng), device=dev, dtype=torch.bfloat16)
        for g_idx, b0 in enumerate(range(0, K, G)):
            b1 = b0 + G
            W_blk = W_b[:, :, b0:b1].clone()
            s_vec, m_vec, safe_s = sag._rtn_minmax_params(W_blk, levels)
            s_grp[:, :, g_idx] = s_vec.to(torch.bfloat16)
            m_grp[:, :, g_idx] = m_vec.to(torch.bfloat16)
            Err_blk = torch.zeros_like(W_blk)
            U_blk = U_b[:, b0:b1, b0:b1]
            for j in range(G):
                wj = W_blk[:, :, j]
                code = torch.clamp(torch.round((wj - m_vec) / safe_s), 0.0, levels)
                code = torch.where(s_vec > 0.0, code, torch.zeros_like(code))
                codes[:, :, b0 + j] = code.to(torch.uint8)
                qj = code * s_vec + m_vec
                err = (wj - qj) / U_blk[:, j, j].unsqueeze(1)
                Err_blk[:, :, j] = err
                if j + 1 < G:
                    W_blk[:, :, j + 1 :] -= err.unsqueeze(2) * U_blk[:, j, j + 1 :].unsqueeze(1)
            if b1 < K:
                W_b[:, :, b1:] -= torch.bmm(Err_blk, U_b[:, b0:b1, b1:])
        codes_orig = codes
    else:
        perm_b = torch.stack(perm_list, dim=0).to(dev).unsqueeze(1).expand(cnt, N, K)
        invperm_b = torch.stack(invperm_list, dim=0).to(dev).unsqueeze(1).expand(cnt, N, K)
        s_vec_g, m_vec_g, safe_s_g = sag._rtn_minmax_params(W_orig.view(cnt, N, ng, G), levels)
        s_grp = s_vec_g.to(torch.bfloat16)
        m_grp = m_vec_g.to(torch.bfloat16)
        s_p = torch.gather(s_vec_g.repeat_interleave(G, dim=2), 2, perm_b)
        m_p = torch.gather(m_vec_g.repeat_interleave(G, dim=2), 2, perm_b)
        safe_sp = torch.gather(safe_s_g.repeat_interleave(G, dim=2), 2, perm_b)

        W_p = torch.gather(W_orig, 2, perm_b).clone()
        codes_p = torch.empty((cnt, N, K), device=dev, dtype=torch.uint8)
        for b0 in range(0, K, 128):
            b1 = min(b0 + 128, K)
            bs = b1 - b0
            W_blk_T = W_p[:, :, b0:b1].transpose(1, 2).contiguous()
            s_blk_T = s_p[:, :, b0:b1].transpose(1, 2).contiguous()
            m_blk_T = m_p[:, :, b0:b1].transpose(1, 2).contiguous()
            safe_s_blk_T = safe_sp[:, :, b0:b1].transpose(1, 2).contiguous()
            Err_blk_T = torch.zeros_like(W_blk_T)
            codes_blk_T = torch.empty((cnt, bs, N), device=dev, dtype=torch.uint8)
            U_blk = U_b[:, b0:b1, b0:b1].contiguous()
            for j in range(bs):
                wj = W_blk_T[:, j]
                sj = s_blk_T[:, j]
                mj = m_blk_T[:, j]
                code = torch.clamp(torch.round((wj - mj) / safe_s_blk_T[:, j]), 0.0, levels)
                code = torch.where(sj > 0.0, code, torch.zeros_like(code))
                codes_blk_T[:, j] = code.to(torch.uint8)
                qj = code * sj + mj
                err = (wj - qj) / U_blk[:, j, j].unsqueeze(1)
                Err_blk_T[:, j] = err
                if j + 1 < bs:
                    W_blk_T[:, j + 1 :].addcmul_(-U_blk[:, j, j + 1 :].unsqueeze(2), err.unsqueeze(1))
            codes_p[:, :, b0:b1] = codes_blk_T.transpose(1, 2)
            if b1 < K:
                W_p[:, :, b1:].baddbmm_(Err_blk_T.transpose(1, 2), U_b[:, b0:b1, b1:], beta=1.0, alpha=-1.0)
        codes_orig = torch.gather(codes_p, 2, invperm_b)
        del perm_b, invperm_b, W_p, codes_p, s_p, m_p, safe_sp, W_blk_T, s_blk_T, m_blk_T, safe_s_blk_T, Err_blk_T, codes_blk_T

    del U_b
    Q_act = dequant_from_params(codes_orig, s_grp, m_grp, G)
    if not cd_refit:
        return Q_act

    H_reg_gpu = torch.stack(H_reg_list, dim=0).to(dev)
    Q_cd = cd_and_bf16_refit_batched(W_orig, codes_orig, s_grp, m_grp, H_reg_gpu, G=G, levels=levels, n_rounds=cd_n_rounds)
    del H_reg_gpu, codes_orig, s_grp, m_grp
    return Q_cd


def quant_gptq_batched_exp_down_shrink_both(
    W_dn: torch.Tensor,
    H_e_chunk: torch.Tensor,
    hits_chunk: torch.Tensor,
    H_mean: torch.Tensor,
    tau: float = 384.0,
    base_damp: float = 0.05,
    cd_n_rounds: int = 2,
    fmt: str = "q4g128",
):
    """Single ActOrder pass returning both (Q_actorder, Q_cd_refit) for Shrink512 exp_down."""
    G, levels = _fmt_spec(fmt)
    cnt, N, K = W_dn.shape
    dev = W_dn.device
    ng = K // G

    H_e_cpu = H_e_chunk.double().cpu()
    H_m_cpu = H_mean.double().cpu()
    hits_list = hits_chunk.tolist()

    H_reg_list, U_list, perm_list, invperm_list = [], [], [], []
    for k_i, h_cnt in enumerate(hits_list):
        h_f = float(h_cnt)
        H_reg = (h_f * H_e_cpu[k_i] + tau * H_m_cpu) / (h_f + tau) if h_f > 0.0 else H_m_cpu.clone()
        H_reg = 0.5 * (H_reg + H_reg.T)
        H_reg_list.append(H_reg.float())
        U_p, perm_k, invperm_k = r3.prepare_actorder_chol_cpu(H_reg, damp=base_damp)
        U_list.append(U_p)
        perm_list.append(perm_k)
        invperm_list.append(invperm_k)

    U_b = torch.stack(U_list, dim=0).to(dev)
    W_orig = W_dn.float()
    perm_b = torch.stack(perm_list, dim=0).to(dev).unsqueeze(1).expand(cnt, N, K)
    invperm_b = torch.stack(invperm_list, dim=0).to(dev).unsqueeze(1).expand(cnt, N, K)
    s_vec_g, m_vec_g, safe_s_g = sag._rtn_minmax_params(W_orig.view(cnt, N, ng, G), levels)
    s_grp = s_vec_g.to(torch.bfloat16)
    m_grp = m_vec_g.to(torch.bfloat16)
    s_p = torch.gather(s_vec_g.repeat_interleave(G, dim=2), 2, perm_b)
    m_p = torch.gather(m_vec_g.repeat_interleave(G, dim=2), 2, perm_b)
    safe_sp = torch.gather(safe_s_g.repeat_interleave(G, dim=2), 2, perm_b)

    W_p = torch.gather(W_orig, 2, perm_b).clone()
    codes_p = torch.empty((cnt, N, K), device=dev, dtype=torch.uint8)
    for b0 in range(0, K, 128):
        b1 = min(b0 + 128, K)
        bs = b1 - b0
        W_blk_T = W_p[:, :, b0:b1].transpose(1, 2).contiguous()
        s_blk_T = s_p[:, :, b0:b1].transpose(1, 2).contiguous()
        m_blk_T = m_p[:, :, b0:b1].transpose(1, 2).contiguous()
        safe_s_blk_T = safe_sp[:, :, b0:b1].transpose(1, 2).contiguous()
        Err_blk_T = torch.zeros_like(W_blk_T)
        codes_blk_T = torch.empty((cnt, bs, N), device=dev, dtype=torch.uint8)
        U_blk = U_b[:, b0:b1, b0:b1].contiguous()
        for j in range(bs):
            wj = W_blk_T[:, j]
            sj = s_blk_T[:, j]
            mj = m_blk_T[:, j]
            code = torch.clamp(torch.round((wj - mj) / safe_s_blk_T[:, j]), 0.0, levels)
            code = torch.where(sj > 0.0, code, torch.zeros_like(code))
            codes_blk_T[:, j] = code.to(torch.uint8)
            qj = code * sj + mj
            err = (wj - qj) / U_blk[:, j, j].unsqueeze(1)
            Err_blk_T[:, j] = err
            if j + 1 < bs:
                W_blk_T[:, j + 1 :].addcmul_(-U_blk[:, j, j + 1 :].unsqueeze(2), err.unsqueeze(1))
        codes_p[:, :, b0:b1] = codes_blk_T.transpose(1, 2)
        if b1 < K:
            W_p[:, :, b1:].baddbmm_(Err_blk_T.transpose(1, 2), U_b[:, b0:b1, b1:], beta=1.0, alpha=-1.0)
    codes_orig = torch.gather(codes_p, 2, invperm_b)
    del perm_b, invperm_b, W_p, codes_p, s_p, m_p, safe_sp, U_b, W_blk_T, s_blk_T, m_blk_T, safe_s_blk_T, Err_blk_T, codes_blk_T

    Q_act = dequant_from_params(codes_orig, s_grp, m_grp, G)
    H_reg_gpu = torch.stack(H_reg_list, dim=0).to(dev)
    Q_cd = cd_and_bf16_refit_batched(W_orig, codes_orig, s_grp, m_grp, H_reg_gpu, G=G, levels=levels, n_rounds=cd_n_rounds)
    del H_reg_gpu, codes_orig, s_grp, m_grp
    return Q_act, Q_cd


# ---------------------------------------------------------------------------
# 5. Asymmetric GPTAQ Target Correction
# ---------------------------------------------------------------------------

def compute_gptaq_target(
    W_bf16_or_f32: torch.Tensor,   # [..., K]
    H_q_f64: torch.Tensor,         # [K, K] = (X_q^T @ X_q) / T
    DeltaG_f64: torch.Tensor,      # [K, K] = ((X - X_q)^T @ X_q) / T
    alpha_aq: float = 0.25,
    damp_aq: float = 0.10,
) -> torch.Tensor:
    """
    Computes shifted GPTAQ target weights W_star = W + alpha_aq * (W @ M_aq)
    where DeltaG = ((X - X_q)^T @ X_q) / T satisfies W^* H_q = W (H_q + DeltaG)
    so M_aq = DeltaG @ (H_q + lambda I)^{-1}.
    Returns float32 tensor on W's device.
    """
    dev = W_bf16_or_f32.device
    K = W_bf16_or_f32.shape[-1]
    Hq = 0.5 * (H_q_f64.double() + H_q_f64.double().T)
    md = max(float(Hq.diagonal().mean().item()), 1e-12)
    Hq_reg = Hq + (damp_aq * md) * torch.eye(K, dtype=torch.float64, device=Hq.device)
    # Solve (H_q + lambda I) @ M_aq.T = DeltaG.T -> M_aq = (solve)^T = DeltaG @ inv(Hq_reg)
    M_aq = torch.linalg.solve(Hq_reg, DeltaG_f64.double().T).T.float().to(dev)
    Wf = W_bf16_or_f32.float()
    return Wf + alpha_aq * (Wf @ M_aq)


# ---------------------------------------------------------------------------
# 6. Pre-Run Mathematical & Serialization Self-Tests
# ---------------------------------------------------------------------------

def _naive_elementwise_cd_ref(W_sb, codes_in, s_bf16, m_bf16, H_gpu, G=128, levels=15.0):
    R, K = W_sb.shape
    ng = K // G
    codes = codes_in.clone()
    Q = (m_bf16.float().repeat_interleave(G, dim=1) + codes.float() * s_bf16.float().repeat_interleave(G, dim=1)).clone()
    sf = s_bf16.float().repeat_interleave(G, dim=1)
    safe_s = torch.where(sf > 0.0, sf, torch.ones_like(sf))
    H_sym = 0.5 * (H_gpu + H_gpu.T)
    for k in range(K):
        h_kk = float(H_sym[k, k].item())
        if h_kk <= 1e-12:
            continue
        g_k = ((Q - W_sb) @ H_sym)[:, k]
        c_old = codes[:, k].float()
        c_new = torch.clamp(torch.round(c_old - g_k / (safe_s[:, k] * h_kk)), 0.0, levels)
        dq = torch.where(sf[:, k] > 0.0, (c_new - c_old) * sf[:, k], torch.zeros_like(sf[:, k]))
        dL = dq * (2.0 * g_k + h_kk * dq)
        acc = dL < -1e-12
        codes[:, k] = torch.where(acc, c_new.to(torch.uint8), codes[:, k])
        Q[:, k] += torch.where(acc, dq, torch.zeros_like(dq))
    return codes, Q


def run_wave2_self_tests(dev: torch.device):
    torch.manual_seed(2026)

    # ---- Test 1: Block-128 CD == Naive CD, strict CD+Refit loss decrease, .faf Q4 roundtrip 0.0 ----
    X = torch.randn(512, 256, dtype=torch.float64) @ (torch.randn(256, 256, dtype=torch.float64) * 0.4 + torch.eye(256, dtype=torch.float64))
    H = (X.T @ X) / 512.0
    W = torch.randn(32, 256, device=dev, dtype=torch.float32) * 0.05

    Q_gptq, codes_g, s_g, m_g = quant_gptq_actorder_static_with_params(W, H_cpu_f64=H, fmt="q4g128", damp=0.05, row_subbatch=16)
    Q_r3_ref = r3.quant_gptq_actorder_static(W, H_cpu_f64=H, fmt="q4g128", damp=0.05, row_subbatch=16)
    diff_r3 = float((Q_gptq.float() - Q_r3_ref.float()).abs().max().item())
    assert diff_r3 == 0.0, f"Test1 ActOrder parity failed: {diff_r3}"

    # Verify Block-128 CD phase matches Naive elementwise CD on same start
    H_gpu = H.float().to(dev)
    codes_naive, Q_naive = _naive_elementwise_cd_ref(W, codes_g, s_g, m_g, H_gpu, G=128, levels=15.0)
    # Run just Phase A (Block-128 CD) using exact same FP32 Q = m + codes * s representation
    codes_b128 = codes_g.clone()
    sf = s_g.float().repeat_interleave(128, dim=1)
    mf = m_g.float().repeat_interleave(128, dim=1)
    Q_b128 = mf + codes_b128.float() * sf
    safe_s = torch.where(sf > 0.0, sf, torch.ones_like(sf))
    H_sym = 0.5 * (H_gpu + H_gpu.T)
    G_mat = (Q_b128 - W) @ H_sym
    for b0 in range(0, 256, 128):
        b1 = b0 + 128
        dblk = torch.zeros((32, 128), device=dev, dtype=torch.float32)
        for j in range(128):
            k = b0 + j
            h_kk = float(H_sym[k, k].item())
            g_k = G_mat[:, k]
            c_old = codes_b128[:, k].float()
            c_new = torch.clamp(torch.round(c_old - g_k / (safe_s[:, k] * h_kk)), 0.0, 15.0)
            dq = (c_new - c_old) * sf[:, k]
            acc = (dq * (2.0 * g_k + h_kk * dq)) < -1e-12
            dq = torch.where(acc, dq, torch.zeros_like(dq))
            codes_b128[:, k] = torch.where(acc, c_new.to(torch.uint8), codes_b128[:, k])
            Q_b128[:, k] += dq
            dblk[:, j] = dq
            if j + 1 < 128:
                G_mat[:, k + 1 : b1] += dq.unsqueeze(1) * H_sym[k, k + 1 : b1].unsqueeze(0)
        if b1 < 256:
            G_mat[:, b1:] += dblk @ H_sym[b0:b1, b1:]
    cd_diff = float((Q_b128 - Q_naive).abs().max().item())
    assert cd_diff < 1e-4, f"Test1 Block-128 CD vs Naive CD diff={cd_diff}"

    # Verify full 2-round CD + Refit (2D and Batched 3D) strictly decreases Hessian MSE and maintains 0.0 .faf roundtrip
    loss_before = float((((Q_gptq.float() - W) @ H_gpu) * (Q_gptq.float() - W)).sum().item())
    codes_cd, s_cd, m_cd = codes_g.clone(), s_g.clone(), m_g.clone()
    Q_cd = cd_and_bf16_refit_subbatch(W, codes_cd, s_cd, m_cd, H_gpu, G=128, levels=15.0, n_rounds=2)
    loss_after = float((((Q_cd.float() - W) @ H_gpu) * (Q_cd.float() - W)).sum().item())
    assert loss_after < loss_before, f"Test1 CD+Refit did not decrease loss: {loss_after} vs {loss_before}"

    codes_b, s_b, m_b = codes_g.unsqueeze(0).clone(), s_g.unsqueeze(0).clone(), m_g.unsqueeze(0).clone()
    Q_cd_b = cd_and_bf16_refit_batched(W.unsqueeze(0), codes_b, s_b, m_b, H_gpu.unsqueeze(0), G=128, levels=15.0, n_rounds=2).squeeze(0)
    batched_diff = float((Q_cd_b.float() - Q_cd.float()).abs().max().item())
    assert batched_diff < 1e-3, f"Test1 batched vs 2D CD+Refit mismatch: {batched_diff}"

    pq4, ps4, pm4 = pack_q4_components(codes_cd, s_cd, m_cd)
    Q_rt4 = dequant_q4_components(pq4, ps4, pm4, G=128)
    rt4_err = float((Q_rt4.float() - Q_cd.float()).abs().max().item())
    assert rt4_err == 0.0, f"Test1 Q4 .faf roundtrip mismatch: {rt4_err}"
    print(f"[PASS] Test 1 (Q4 ActOrder+CD+Refit): R3_diff={diff_r3:.1e}, B128_vs_Naive={cd_diff:.2e}, H_MSE {loss_before:.4f}->{loss_after:.4f} ({100*(loss_after/loss_before-1):+.1f}%), .faf_rt={rt4_err:.1e}")

    # ---- Test 2: q6g64 ActOrder static_groups .faf roundtrip 0.0 + >=40% outlier MSE cut ----
    # Create skewed outlier activation spectrum (like gdn_out / attn_o channels)
    scale_ch = torch.ones(256, dtype=torch.float64)
    scale_ch[:16] = 25.0
    scale_ch[16:48] = 5.0
    X_out = (torch.randn(512, 256, dtype=torch.float64) @ (torch.randn(256, 256, dtype=torch.float64) * 0.2 + torch.eye(256, dtype=torch.float64))) * scale_ch.unsqueeze(0)
    H_out = (X_out.T @ X_out) / 512.0
    W_q6 = torch.randn(32, 256, device=dev, dtype=torch.float32) * 0.06
    Q6_ao, codes6, s6, m6 = quant_gptq_actorder_static_with_params(W_q6, H_cpu_f64=H_out, fmt="q6g64", damp=0.05, row_subbatch=16)
    Q6_r3_ref = r3.quant_gptq_actorder_static(W_q6, H_cpu_f64=H_out, fmt="q6g64", damp=0.05, row_subbatch=16)
    diff6_r3 = float((Q6_ao.float() - Q6_r3_ref.float()).abs().max().item())
    assert diff6_r3 == 0.0, f"Test2 Q6 ActOrder parity failed: {diff6_r3}"

    pq6, ps6, pm6 = pack_q6_components(codes6, s6, m6)
    Q_rt6 = dequant_q6_components(pq6, ps6, pm6, G=64)
    rt6_err = float((Q_rt6.float() - Q6_ao.float()).abs().max().item())
    assert rt6_err == 0.0, f"Test2 Q6 .faf roundtrip mismatch: {rt6_err}"

    H_out_gpu = H_out.float().to(dev)
    Q6_rtn = sq.fake_quant(W_q6, "q6g64").float()
    mse6_rtn = float((((Q6_rtn - W_q6) @ H_out_gpu) * (Q6_rtn - W_q6)).sum().item())
    mse6_ao = float((((Q6_ao.float() - W_q6) @ H_out_gpu) * (Q6_ao.float() - W_q6)).sum().item())
    cut6 = 1.0 - mse6_ao / mse6_rtn
    assert cut6 >= 0.40, f"Test2 Q6 hotspot MSE cut only {100*cut6:.1f}% (< 40%)"
    print(f"[PASS] Test 2 (Q6 ActOrder static_groups): R3_diff={diff6_r3:.1e}, .faf_rt={rt6_err:.1e}, Outlier_H_MSE cut={100*cut6:.1f}%")

    # ---- Test 3: Asymmetric GPTAQ reduces error on asymmetric (X, X_q) and reduces to GPTQ when X_q==X ----
    T_t, K_t, R_t = 512, 128, 16
    X_t = torch.randn(T_t, K_t, device=dev, dtype=torch.float64)
    # Deliberately correlated + skewed asymmetric drift (so DeltaG != DeltaG.T)
    mix = torch.tril(torch.randn(K_t, K_t, device=dev, dtype=torch.float64) * 0.15)
    X_q_t = X_t @ (torch.eye(K_t, device=dev, dtype=torch.float64) + mix) + 0.08 * torch.randn(T_t, K_t, device=dev, dtype=torch.float64)
    W_t = torch.randn(R_t, K_t, device=dev, dtype=torch.float32) * 0.05

    H_q_eq = (X_t.T @ X_t) / float(T_t)
    dG_eq = torch.zeros_like(H_q_eq)
    W_star_eq = compute_gptaq_target(W_t, H_q_eq.cpu(), dG_eq.cpu(), alpha_aq=0.25, damp_aq=0.10)
    eq_diff = float((W_star_eq - W_t).abs().max().item())
    assert eq_diff == 0.0, f"Test3 zero-drift identity failed: {eq_diff}"

    H_q_t = (X_q_t.T @ X_q_t) / float(T_t)
    DeltaG_t = ((X_t - X_q_t).T @ X_q_t) / float(T_t)
    assert float((DeltaG_t - DeltaG_t.T).abs().max().item()) > 1e-3, "Test3 DeltaG unexpectedly symmetric"
    W_star_t = compute_gptaq_target(W_t, H_q_t.cpu(), DeltaG_t.cpu(), alpha_aq=0.25, damp_aq=0.05)

    Y_ref = X_t.float() @ W_t.T
    err_before = float(((X_q_t.float() @ W_t.T) - Y_ref).pow(2).mean().item())
    err_after = float(((X_q_t.float() @ W_star_t.T) - Y_ref).pow(2).mean().item())
    assert err_after < err_before, f"Test3 GPTAQ increased error: {err_after} vs {err_before}"
    print(f"[PASS] Test 3 (GPTAQ asymmetric target): zero_drift={eq_diff:.1e}, asym_MSE {err_before:.6f}->{err_after:.6f} ({100*(err_after/err_before-1):+.1f}%)")

    del X, H, W, H_gpu, Q_gptq, Q_r3_ref, Q_cd, Q_rt4, pq4, ps4, pm4
    del X_out, H_out, H_out_gpu, W_q6, Q6_ao, Q6_r3_ref, Q_rt6, pq6, ps6, pm6
    del X_t, X_q_t, W_t, H_q_eq, dG_eq, W_star_eq, H_q_t, DeltaG_t, W_star_t
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 7. Main 48-Layer Round 4 Wave-2 Calibration & Held-Out Evaluation
# ---------------------------------------------------------------------------

def run_wave2_sim(
    model_dir: str,
    out_json: str,
    n_seqs: int = 40,
    seq_len: int = 1536,
    cal_mb: int = 4,
    expert_chunk: int = 32,
    tau: float = 384.0,
    alpha_aq: float = 0.25,
    damp_aq: float = 0.10,
    max_layers: int = None,
):
    dev = torch.device("cuda:0")
    cpu = torch.device("cpu")
    _ = torch.zeros(1, device=dev)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(dev)

    run_wave2_self_tests(dev)

    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    tok = sq.QwenTokenizer(os.path.join(model_dir, "tokenizer.json"))
    ids_cal_2d, dom_summary, cal_sha = sag.build_diverse_calib_sequences(tok, workspace_root, n_seqs=n_seqs, seq_len=seq_len)
    T_cal = n_seqs * seq_len

    eval_ids = tok.encode(sq.CALIB_DOC)
    T_eval = len(eval_ids)
    eval_ids_cpu = torch.tensor(eval_ids, dtype=torch.long)
    target_ids = torch.tensor(eval_ids[1:], device=dev, dtype=torch.long)

    reader = sq.SafetensorsReader(model_dir)
    lm = "model.language_model."
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg_root = json.load(f)
    tc = cfg_root.get("text_config", cfg_root)
    n_layers = min(tc["num_hidden_layers"], max_layers) if max_layers is not None else tc["num_hidden_layers"]
    layer_types = tc.get("layer_types")

    cos_c, sin_c = sq.build_qwen_rope(seq_len, dev, torch.bfloat16)
    cos_e, sin_e = sq.build_qwen_rope(T_eval, dev, torch.bfloat16)

    print(f"\n=== Round 4 Wave-2 Zero-Speed Simulation ({n_layers} layers) ===", flush=True)
    print(f"Calibration: {n_seqs}x{seq_len}={T_cal} toks (sha={cal_sha}) | domains={dom_summary} | tau={tau}", flush=True)
    print(f"GPTAQ: alpha_aq={alpha_aq}, damp_aq={damp_aq} | Held-out T_eval={T_eval}", flush=True)

    t0 = time.time()
    # Aggregate Hessian MSE trackers across layers
    hmse_totals = {
        "gdn_in": {"rtn": 0.0, "r3": 0.0, "cd": 0.0, "gptaq": 0.0},
        "gdn_out": {"rtn": 0.0, "ao": 0.0},
        "attn_o": {"rtn": 0.0, "ao": 0.0},
        "sh_down": {"rtn": 0.0, "ao": 0.0},
        "exp_gu": {"rtn": 0.0, "r3": 0.0, "cd": 0.0, "gptaq": 0.0},
        "exp_down": {"rtn": 0.0, "r3": 0.0, "cd": 0.0, "gptaq": 0.0},
    }
    layer_metrics = []

    def _hmse(q_w, w_ref, H_f32):
        e = q_w.float() - w_ref.float()
        return float(((e @ H_f32) * e).sum().item())

    with torch.no_grad():
        emb_cpu = reader.read_tensor(lm + "embed_tokens.weight", cpu, torch.bfloat16)
        cal_states_cpu = F.embedding(ids_cal_2d, emb_cpu).unsqueeze(2).expand(n_seqs, seq_len, 4, 2560).contiguous()
        cal_states_q_cpu = cal_states_cpu.clone()

        eval_x0 = F.embedding(eval_ids_cpu, emb_cpu).to(dev)
        eval_bf16 = eval_x0.unsqueeze(1).expand(T_eval, 4, 2560).contiguous()
        eval_vars = {vn: eval_bf16.clone() for vn in VARIANT_NAMES}
        del emb_cpu, eval_x0
        torch.cuda.empty_cache()

        for L in range(n_layers):
            tl = time.time()
            P = f"{lm}layers.{L}."
            is_attn = (layer_types[L] == "full_attention") if layer_types is not None else ((L + 1) % 4 == 0)
            lstat = {"layer": L, "type": "attn" if is_attn else "gdn"}

            # ---- 1. Optional PLE Injection (Layer 1) ----
            if (P + "ple.key_proj.weight") in reader.wm:
                ple_p = P + "ple."
                mult = reader.read_tensor(ple_p + "ple_embedding.layer_multipliers", cpu).tolist()
                vsz = reader.read_tensor(ple_p + "ple_embedding.ngram_heads_vocab_sizes", cpu).tolist()
                offs = reader.read_tensor(ple_p + "ple_embedding.ngram_heads_offsets", cpu).tolist()

                w_kp = reader.read_tensor(ple_p + "key_proj.weight", dev, torch.bfloat16)
                w_vp = reader.read_tensor(ple_p + "value_proj.weight", dev, torch.bfloat16)
                w_ple_bf16 = torch.cat([w_kp, w_vp], dim=0)
                w_ple_q6 = sq.fake_quant(w_ple_bf16, "q6g64")
                del w_kp, w_vp
                ple_cw = reader.read_tensor(ple_p + "conv1d.weight", dev, torch.float32)
                ple_nk = reader.read_tensor(ple_p + "norm_key.weight", dev, torch.float32)
                ple_nq = reader.read_tensor(ple_p + "norm_query.weight", dev, torch.float32)
                ple_nc = reader.read_tensor(ple_p + "norm_conv.weight", dev, torch.float32)

                for b_idx in range(n_seqs):
                    ur_c, mi_c = sq.ple_ngram_ids_py(mult, vsz, offs, ids_cal_2d[b_idx].tolist(), eos=248044)
                    ple_c = reader.gather_ple_rows(ple_p, ur_c, dev)[torch.tensor(mi_c, device=dev, dtype=torch.long)].reshape(seq_len, 2560)
                    xc_b = cal_states_cpu[b_idx].to(dev)
                    cal_states_cpu[b_idx] = sq.ple_forward(xc_b, ple_c, w_ple_bf16, ple_cw, ple_nk, ple_nq, ple_nc).cpu()
                    xq_b = cal_states_q_cpu[b_idx].to(dev)
                    cal_states_q_cpu[b_idx] = sq.ple_forward(xq_b, ple_c, w_ple_q6, ple_cw, ple_nk, ple_nq, ple_nc).cpu()
                    del ple_c, xc_b, xq_b

                ur_e, mi_e = sq.ple_ngram_ids_py(mult, vsz, offs, eval_ids, eos=248044)
                ple_e = reader.gather_ple_rows(ple_p, ur_e, dev)[torch.tensor(mi_e, device=dev, dtype=torch.long)].reshape(T_eval, 2560)
                eval_bf16 = sq.ple_forward(eval_bf16, ple_e, w_ple_bf16, ple_cw, ple_nk, ple_nq, ple_nc)
                for vn in VARIANT_NAMES:
                    eval_vars[vn] = sq.ple_forward(eval_vars[vn], ple_e, w_ple_q6, ple_cw, ple_nk, ple_nq, ple_nc)
                del ple_e, w_ple_bf16, w_ple_q6, ple_cw, ple_nk, ple_nq, ple_nc

            # ---- 2. Attn / GDN Hyper-Connection Mix ----
            ahc = P + "attn_hyper_connection."
            a_wd = reader.read_tensor(ahc + "input_mix_weight_down.weight", dev, torch.float32)
            a_wi = reader.read_tensor(ahc + "block_inject_weight.weight", dev, torch.float32)
            a_wu_bf16 = reader.read_tensor(ahc + "input_mix_weight_up.weight", dev, torch.bfloat16)
            a_hn = reader.read_tensor(ahc + "hc_norm.weight", dev, torch.float32)
            a_wdf = a_wd * (1.0 + a_hn.unsqueeze(0))
            a_wif = a_wi * (1.0 + a_hn.unsqueeze(0))
            del a_wd, a_wi
            ahcd_bf16 = (a_wdf.to(torch.bfloat16), a_wif.to(torch.bfloat16))
            ahcd_q8 = (sq.fake_quant(a_wdf, "q8g64"), sq.fake_quant(a_wif, "q8g64"))
            awu_q8 = sq.fake_quant(a_wu_bf16, "q8g64")
            del a_wdf, a_wif

            t_ag0 = time.time()
            if not is_attn:
                gp = P + "linear_attn."
                w_qkv = reader.read_tensor(gp + "in_proj_qkv.weight", dev, torch.bfloat16)
                w_z = reader.read_tensor(gp + "in_proj_z.weight", dev, torch.bfloat16)
                w_b = reader.read_tensor(gp + "in_proj_b.weight", dev, torch.bfloat16)
                w_a = reader.read_tensor(gp + "in_proj_a.weight", dev, torch.bfloat16)
                gin_bf16 = torch.cat([w_qkv, w_z, w_b, w_a], dim=0)
                del w_qkv, w_z, w_b, w_a
                gout_bf16 = reader.read_tensor(gp + "out_proj.weight", dev, torch.bfloat16)
                cw = reader.read_tensor(gp + "conv1d.weight", dev, torch.float32)
                A_log = reader.read_tensor(gp + "A_log", dev, torch.float32)
                dt_bias = reader.read_tensor(gp + "dt_bias", dev, torch.float32)
                gnorm = reader.read_tensor(gp + "norm.weight", dev, torch.float32)

                H_gdn_cpu = torch.zeros((2560, 2560), dtype=torch.float64)
                H_gdn_q_cpu = torch.zeros((2560, 2560), dtype=torch.float64)
                dG_gdn_cpu = torch.zeros((2560, 2560), dtype=torch.float64)
                H_gout_cpu = torch.zeros((6144, 6144), dtype=torch.float64)

                for b0 in range(0, n_seqs, cal_mb):
                    b1 = min(b0 + cal_mb, n_seqs)
                    xb = cal_states_cpu[b0:b1].to(dev)
                    ub, win_b = sag.hc_mix_down_up_batched(xb, *ahcd_bf16, a_wu_bf16, a_hn, inject=True)
                    ym, og = sag.gdn_forward_batched(ub, gin_bf16, gout_bf16, cw, A_log, dt_bias, gnorm, return_o_gated=True)
                    cal_states_cpu[b0:b1] = (xb.float() + win_b.unsqueeze(-1) * ym.float().unsqueeze(2)).to(torch.bfloat16).cpu()

                    xq_b = cal_states_q_cpu[b0:b1].to(dev)
                    uq_b, _ = sag.hc_mix_down_up_batched(xq_b, *ahcd_q8, awu_q8, a_hn, inject=True)

                    u_f32 = ub.reshape(-1, 2560).float()
                    uq_f32 = uq_b.reshape(-1, 2560).float()
                    og_f32 = og.reshape(-1, 6144).float()
                    H_gdn_cpu.add_((u_f32.T @ u_f32).double().cpu())
                    H_gdn_q_cpu.add_((uq_f32.T @ uq_f32).double().cpu())
                    dG_gdn_cpu.add_(((u_f32 - uq_f32).T @ uq_f32).double().cpu())
                    H_gout_cpu.add_((og_f32.T @ og_f32).double().cpu())
                    del xb, ub, win_b, ym, og, xq_b, uq_b, u_f32, uq_f32, og_f32

                H_gdn_cpu.div_(float(T_cal))
                H_gdn_q_cpu.div_(float(T_cal))
                dG_gdn_cpu.div_(float(T_cal))
                H_gout_cpu.div_(float(T_cal))

                gin_rtn = sq.fake_quant(gin_bf16, "q4g128")
                gin_r3, codes_g, s_g, m_g = quant_gptq_actorder_static_with_params(gin_bf16, H_cpu_f64=H_gdn_cpu, fmt="q4g128", damp=0.05, row_subbatch=40960)
                gin_cd = cd_refit_from_params(gin_bf16, codes_g, s_g, m_g, H_gdn_cpu, fmt="q4g128", row_subbatch=40960, n_rounds=2)
                del codes_g, s_g, m_g

                W_star_gdn = compute_gptaq_target(gin_bf16, H_gdn_q_cpu, dG_gdn_cpu, alpha_aq=alpha_aq, damp_aq=damp_aq)
                gin_aq = quant_gptq_actorder_cd_refit(W_star_gdn, H_cpu_f64=H_gdn_q_cpu, fmt="q4g128", damp=0.05, row_subbatch=40960, n_rounds=2)
                del W_star_gdn

                gout_rtn = sq.fake_quant(gout_bf16, "q6g64")
                gout_ao, _, _, _ = quant_gptq_actorder_static_with_params(gout_bf16, H_cpu_f64=H_gout_cpu, fmt="q6g64", damp=0.05, row_subbatch=2560)

                H_gdn_gpu = H_gdn_cpu.float().to(dev)
                H_gout_gpu = H_gout_cpu.float().to(dev)
                hm_gi_r = _hmse(gin_rtn, gin_bf16, H_gdn_gpu)
                hm_gi_3 = _hmse(gin_r3, gin_bf16, H_gdn_gpu)
                hm_gi_c = _hmse(gin_cd, gin_bf16, H_gdn_gpu)
                hm_gi_a = _hmse(gin_aq, gin_bf16, H_gdn_gpu)
                hm_go_r = _hmse(gout_rtn, gout_bf16, H_gout_gpu)
                hm_go_a = _hmse(gout_ao, gout_bf16, H_gout_gpu)
                hmse_totals["gdn_in"]["rtn"] += hm_gi_r
                hmse_totals["gdn_in"]["r3"] += hm_gi_3
                hmse_totals["gdn_in"]["cd"] += hm_gi_c
                hmse_totals["gdn_in"]["gptaq"] += hm_gi_a
                hmse_totals["gdn_out"]["rtn"] += hm_go_r
                hmse_totals["gdn_out"]["ao"] += hm_go_a
                lstat["gdn_in_ratio_cd_vs_r3"] = round(hm_gi_c / max(hm_gi_3, 1e-12), 4)
                lstat["gdn_out_ratio_ao_vs_rtn"] = round(hm_go_a / max(hm_go_r, 1e-12), 4)
                del H_gdn_gpu, H_gout_gpu, H_gdn_cpu, H_gdn_q_cpu, dG_gdn_cpu, H_gout_cpu

                # Advance Var 5 calibration trajectory with Var 5 weights (gin_aq, gout_ao)
                for b0 in range(0, n_seqs, cal_mb):
                    b1 = min(b0 + cal_mb, n_seqs)
                    xq_b = cal_states_q_cpu[b0:b1].to(dev)
                    uq_b, winq_b = sag.hc_mix_down_up_batched(xq_b, *ahcd_q8, awu_q8, a_hn, inject=True)
                    ym_q = sag.gdn_forward_batched(uq_b, gin_aq, gout_ao, cw, A_log, dt_bias, gnorm)
                    cal_states_q_cpu[b0:b1] = (xq_b.float() + winq_b.unsqueeze(-1) * ym_q.float().unsqueeze(2)).to(torch.bfloat16).cpu()
                    del xq_b, uq_b, winq_b, ym_q

                # Advance held-out eval trajectories
                u_eb, win_eb = sq.hc_mix_down_up(eval_bf16, *ahcd_bf16, a_wu_bf16, a_hn, inject=True)
                y_eb = sq.gdn_forward(u_eb, gin_bf16, gout_bf16, cw, A_log, dt_bias, gnorm)
                eval_bf16 = (eval_bf16.float() + win_eb.unsqueeze(-1) * y_eb.float().unsqueeze(1)).to(torch.bfloat16)
                del u_eb, win_eb, y_eb

                g_map = {
                    "0_RTN_Prod": (gin_rtn, gout_rtn),
                    "1_R3_ActOrder_Q4Only": (gin_r3, gout_rtn),
                    "2_R3_plus_Q6_Hotspots": (gin_r3, gout_ao),
                    "3_R3_plus_CD_Refit": (gin_cd, gout_rtn),
                    "4_R3_plus_Q6_CD_Refit": (gin_cd, gout_ao),
                    "5_GPTAQ_ActOrder_CD": (gin_aq, gout_ao),
                }
                for vn in VARIANT_NAMES:
                    wi_v, wo_v = g_map[vn]
                    uv, wv = sq.hc_mix_down_up(eval_vars[vn], *ahcd_q8, awu_q8, a_hn, inject=True)
                    yv = sq.gdn_forward(uv, wi_v, wo_v, cw, A_log, dt_bias, gnorm)
                    eval_vars[vn] = (eval_vars[vn].float() + wv.unsqueeze(-1) * yv.float().unsqueeze(1)).to(torch.bfloat16)
                    del uv, wv, yv

                del gin_bf16, gout_bf16, gin_rtn, gin_r3, gin_cd, gin_aq, gout_rtn, gout_ao, cw, A_log, dt_bias, gnorm
            else:
                ap = P + "self_attn."
                wq = reader.read_tensor(ap + "q_proj.weight", dev, torch.bfloat16)
                wk = reader.read_tensor(ap + "k_proj.weight", dev, torch.bfloat16)
                wv = reader.read_tensor(ap + "v_proj.weight", dev, torch.bfloat16)
                qkv_bf16 = torch.cat([wq, wk, wv], dim=0)
                del wq, wk, wv
                ao_bf16 = reader.read_tensor(ap + "o_proj.weight", dev, torch.bfloat16)
                qn = reader.read_tensor(ap + "q_norm.weight", dev, torch.float32)
                kn = reader.read_tensor(ap + "k_norm.weight", dev, torch.float32)

                H_ao_cpu = torch.zeros((6144, 6144), dtype=torch.float64)
                for b0 in range(0, n_seqs, cal_mb):
                    b1 = min(b0 + cal_mb, n_seqs)
                    xb = cal_states_cpu[b0:b1].to(dev)
                    ub, win_b = sag.hc_mix_down_up_batched(xb, *ahcd_bf16, a_wu_bf16, a_hn, inject=True)
                    ym, og = full_attn_forward_batched(ub, qkv_bf16, ao_bf16, qn, kn, cos_c, sin_c, return_o_gated=True)
                    cal_states_cpu[b0:b1] = (xb.float() + win_b.unsqueeze(-1) * ym.float().unsqueeze(2)).to(torch.bfloat16).cpu()
                    og_f32 = og.reshape(-1, 6144).float()
                    H_ao_cpu.add_((og_f32.T @ og_f32).double().cpu())
                    del xb, ub, win_b, ym, og, og_f32
                H_ao_cpu.div_(float(T_cal))

                qkv_rtn = sq.fake_quant(qkv_bf16, "q6g64")
                ao_rtn = sq.fake_quant(ao_bf16, "q6g64")
                ao_ao, _, _, _ = quant_gptq_actorder_static_with_params(ao_bf16, H_cpu_f64=H_ao_cpu, fmt="q6g64", damp=0.05, row_subbatch=2560)

                H_ao_gpu = H_ao_cpu.float().to(dev)
                hm_ao_r = _hmse(ao_rtn, ao_bf16, H_ao_gpu)
                hm_ao_a = _hmse(ao_ao, ao_bf16, H_ao_gpu)
                hmse_totals["attn_o"]["rtn"] += hm_ao_r
                hmse_totals["attn_o"]["ao"] += hm_ao_a
                lstat["attn_o_ratio_ao_vs_rtn"] = round(hm_ao_a / max(hm_ao_r, 1e-12), 4)
                del H_ao_gpu, H_ao_cpu

                for b0 in range(0, n_seqs, cal_mb):
                    b1 = min(b0 + cal_mb, n_seqs)
                    xq_b = cal_states_q_cpu[b0:b1].to(dev)
                    uq_b, winq_b = sag.hc_mix_down_up_batched(xq_b, *ahcd_q8, awu_q8, a_hn, inject=True)
                    ym_q = full_attn_forward_batched(uq_b, qkv_rtn, ao_ao, qn, kn, cos_c, sin_c)
                    cal_states_q_cpu[b0:b1] = (xq_b.float() + winq_b.unsqueeze(-1) * ym_q.float().unsqueeze(2)).to(torch.bfloat16).cpu()
                    del xq_b, uq_b, winq_b, ym_q

                u_eb, win_eb = sq.hc_mix_down_up(eval_bf16, *ahcd_bf16, a_wu_bf16, a_hn, inject=True)
                y_eb = sq.full_attn_forward(u_eb, qkv_bf16, ao_bf16, qn, kn, cos_e, sin_e)
                eval_bf16 = (eval_bf16.float() + win_eb.unsqueeze(-1) * y_eb.float().unsqueeze(1)).to(torch.bfloat16)
                del u_eb, win_eb, y_eb

                use_ao_o = {"2_R3_plus_Q6_Hotspots", "4_R3_plus_Q6_CD_Refit", "5_GPTAQ_ActOrder_CD"}
                for vn in VARIANT_NAMES:
                    ao_v = ao_ao if vn in use_ao_o else ao_rtn
                    uv, wv = sq.hc_mix_down_up(eval_vars[vn], *ahcd_q8, awu_q8, a_hn, inject=True)
                    yv = sq.full_attn_forward(uv, qkv_rtn, ao_v, qn, kn, cos_e, sin_e)
                    eval_vars[vn] = (eval_vars[vn].float() + wv.unsqueeze(-1) * yv.float().unsqueeze(1)).to(torch.bfloat16)
                    del uv, wv, yv

                del qkv_bf16, ao_bf16, qkv_rtn, ao_rtn, ao_ao, qn, kn

            del ahcd_bf16, ahcd_q8, a_wu_bf16, awu_q8, a_hn
            t_ag = time.time() - t_ag0

            # ---- 3. MLP Hyper-Connection Mix ----
            mhc = P + "mlp_hyper_connection."
            m_wd = reader.read_tensor(mhc + "input_mix_weight_down.weight", dev, torch.float32)
            m_wi = reader.read_tensor(mhc + "block_inject_weight.weight", dev, torch.float32)
            m_wu_bf16 = reader.read_tensor(mhc + "input_mix_weight_up.weight", dev, torch.bfloat16)
            m_hn = reader.read_tensor(mhc + "hc_norm.weight", dev, torch.float32)
            m_wdf = m_wd * (1.0 + m_hn.unsqueeze(0))
            m_wif = m_wi * (1.0 + m_hn.unsqueeze(0))
            del m_wd, m_wi
            mhcd_bf16 = (m_wdf.to(torch.bfloat16), m_wif.to(torch.bfloat16))
            mhcd_q8 = (sq.fake_quant(m_wdf, "q8g64"), sq.fake_quant(m_wif, "q8g64"))
            mwu_q8 = sq.fake_quant(m_wu_bf16, "q8g64")
            del m_wdf, m_wif

            u_mlp_cal = torch.empty((T_cal, 2560), device=dev, dtype=torch.bfloat16)
            u_mlp_q_cal = torch.empty((T_cal, 2560), device=dev, dtype=torch.bfloat16)
            win_mlp_cal = torch.empty((T_cal, 4), device=dev, dtype=torch.float32)
            win_mlp_q_cal = torch.empty((T_cal, 4), device=dev, dtype=torch.float32)

            H_mlp_cpu = torch.zeros((2560, 2560), dtype=torch.float64)
            H_mlp_q_cpu = torch.zeros((2560, 2560), dtype=torch.float64)
            dG_mlp_cpu = torch.zeros((2560, 2560), dtype=torch.float64)

            for b0 in range(0, n_seqs, cal_mb):
                b1 = min(b0 + cal_mb, n_seqs)
                s0, s1 = b0 * seq_len, b1 * seq_len
                xb = cal_states_cpu[b0:b1].to(dev)
                ub, win_b = sag.hc_mix_down_up_batched(xb, *mhcd_bf16, m_wu_bf16, m_hn, inject=True)
                uf = ub.reshape(-1, 2560)
                u_mlp_cal[s0:s1] = uf
                win_mlp_cal[s0:s1] = win_b.reshape(-1, 4)

                xq_b = cal_states_q_cpu[b0:b1].to(dev)
                uq_b, winq_b = sag.hc_mix_down_up_batched(xq_b, *mhcd_q8, mwu_q8, m_hn, inject=True)
                uqf = uq_b.reshape(-1, 2560)
                u_mlp_q_cal[s0:s1] = uqf
                win_mlp_q_cal[s0:s1] = winq_b.reshape(-1, 4)

                u_f32 = uf.float()
                uq_f32 = uqf.float()
                H_mlp_cpu.add_((u_f32.T @ u_f32).double().cpu())
                H_mlp_q_cpu.add_((uq_f32.T @ uq_f32).double().cpu())
                dG_mlp_cpu.add_(((u_f32 - uq_f32).T @ uq_f32).double().cpu())
                del xb, ub, win_b, uf, xq_b, uq_b, winq_b, uqf, u_f32, uq_f32

            H_mlp_cpu.div_(float(T_cal))
            H_mlp_q_cpu.div_(float(T_cal))
            dG_mlp_cpu.div_(float(T_cal))
            act_precomp_mlp = r3.prepare_actorder_chol_cpu(H_mlp_cpu, damp=0.05)
            act_precomp_mlp_q = r3.prepare_actorder_chol_cpu(H_mlp_q_cpu, damp=0.05)

            # Eval MLP inputs
            u_eb_mlp, win_eb_mlp = sq.hc_mix_down_up(eval_bf16, *mhcd_bf16, m_wu_bf16, m_hn, inject=True)
            u_ev_mlp = {}
            win_ev_mlp = {}
            for vn in VARIANT_NAMES:
                u_ev_mlp[vn], win_ev_mlp[vn] = sq.hc_mix_down_up(eval_vars[vn], *mhcd_q8, mwu_q8, m_hn, inject=True)
            del mhcd_bf16, mhcd_q8, m_wu_bf16, mwu_q8, m_hn

            # ---- 4. Router + Shared Expert ----
            mp = P + "mlp."
            wg_r = reader.read_tensor(mp + "gate.weight", dev, torch.bfloat16)
            wsg_r = reader.read_tensor(mp + "shared_expert_gate.weight", dev, torch.bfloat16)
            w_router_all = torch.cat([wg_r, wsg_r], dim=0)
            del wg_r, wsg_r

            cal_route = sq.moe_route_qwen(u_mlp_cal, w_router_all)
            tidx_flat = cal_route[0].reshape(-1)
            ord_all = torch.argsort(tidx_flat, stable=True)
            sorted_tok_all = ord_all // 10
            sorted_w_all = cal_route[1].reshape(-1)[ord_all].unsqueeze(-1)
            counts_all = torch.bincount(tidx_flat, minlength=512)
            offsets_all = torch.zeros(513, device=dev, dtype=torch.long)
            torch.cumsum(counts_all, dim=0, out=offsets_all[1:])
            del tidx_flat, ord_all

            # Var 5 true calibration routing on quantized state
            cal_route_q = sq.moe_route_qwen(u_mlp_q_cal, w_router_all)
            tidx_q_flat = cal_route_q[0].reshape(-1)
            ord_q_all = torch.argsort(tidx_q_flat, stable=True)
            sorted_tok_q = ord_q_all // 10
            sorted_w_q = cal_route_q[1].reshape(-1)[ord_q_all].unsqueeze(-1)
            counts_q_all = torch.bincount(tidx_q_flat, minlength=512)
            offsets_q_all = torch.zeros(513, device=dev, dtype=torch.long)
            torch.cumsum(counts_q_all, dim=0, out=offsets_q_all[1:])
            del tidx_q_flat, ord_q_all

            eval_route_bf16 = sq.moe_route_qwen(u_eb_mlp, w_router_all)
            eval_routes = {vn: sq.moe_route_qwen(u_ev_mlp[vn], w_router_all) for vn in VARIANT_NAMES}
            del w_router_all

            w_sh_g = reader.read_tensor(mp + "shared_expert.gate_proj.weight", dev, torch.bfloat16)
            w_sh_u = reader.read_tensor(mp + "shared_expert.up_proj.weight", dev, torch.bfloat16)
            w_sh_gu = torch.cat([w_sh_g, w_sh_u], dim=0)
            del w_sh_g, w_sh_u
            w_sh_d = reader.read_tensor(mp + "shared_expert.down_proj.weight", dev, torch.bfloat16)

            sh_gu_c = F.linear(u_mlp_cal, w_sh_gu)
            sh_h_c = F.silu(sh_gu_c[:, :640]) * sh_gu_c[:, 640:]
            sh_out_cal = F.linear(sh_h_c, w_sh_d).float() * cal_route[2].unsqueeze(-1)
            H_shd_cpu = ((sh_h_c.float().T @ sh_h_c.float()) / float(T_cal)).double().cpu()
            del sh_gu_c, sh_h_c, cal_route

            w_sh_gu_rtn = sq.fake_quant(w_sh_gu, "q6g64")
            w_sh_d_rtn = sq.fake_quant(w_sh_d, "q6g64")
            w_sh_d_ao, _, _, _ = quant_gptq_actorder_static_with_params(w_sh_d, H_cpu_f64=H_shd_cpu, fmt="q6g64", damp=0.05, row_subbatch=2560)

            H_shd_gpu = H_shd_cpu.float().to(dev)
            hm_sh_r = _hmse(w_sh_d_rtn, w_sh_d, H_shd_gpu)
            hm_sh_a = _hmse(w_sh_d_ao, w_sh_d, H_shd_gpu)
            hmse_totals["sh_down"]["rtn"] += hm_sh_r
            hmse_totals["sh_down"]["ao"] += hm_sh_a
            lstat["sh_down_ratio_ao_vs_rtn"] = round(hm_sh_a / max(hm_sh_r, 1e-12), 4)
            del H_shd_gpu, H_shd_cpu

            # Var 5 cal shared output
            sh_gu_q_c = F.linear(u_mlp_q_cal, w_sh_gu_rtn)
            sh_h_q_c = F.silu(sh_gu_q_c[:, :640]) * sh_gu_q_c[:, 640:]
            sh_out_q_cal = F.linear(sh_h_q_c, w_sh_d_ao).float() * cal_route_q[2].unsqueeze(-1)
            del sh_gu_q_c, sh_h_q_c, cal_route_q

            def _run_sh(u_in, wgu, wdn, sh_c):
                g = F.linear(u_in, wgu)
                h = F.silu(g[:, :640]) * g[:, 640:]
                return F.linear(h, wdn).float() * sh_c.unsqueeze(-1)

            eval_sh_bf16 = _run_sh(u_eb_mlp, w_sh_gu, w_sh_d, eval_route_bf16[2])
            use_ao_sh = {"2_R3_plus_Q6_Hotspots", "4_R3_plus_Q6_CD_Refit", "5_GPTAQ_ActOrder_CD"}
            eval_sh_vars = {
                vn: _run_sh(u_ev_mlp[vn], w_sh_gu_rtn, w_sh_d_ao if vn in use_ao_sh else w_sh_d_rtn, eval_routes[vn][2])
                for vn in VARIANT_NAMES
            }
            del w_sh_gu, w_sh_d, w_sh_gu_rtn, w_sh_d_rtn, w_sh_d_ao

            # ---- 5. Routed Experts Pass A: Collect Global Hessians + Advance BF16 Cal State ----
            t_pA0 = time.time()
            gu_name = mp + "experts.gate_up_proj"
            dn_name = mp + "experts.down_proj"
            reader.info(gu_name)
            reader.info(dn_name)

            H_e_all_cpu = torch.zeros((512, 640, 640), dtype=torch.float32)
            H_sum_all = torch.zeros((640, 640), device=dev, dtype=torch.float64)
            cal_routed = torch.zeros((T_cal, 2560), device=dev, dtype=torch.float32)
            offsets_all_list = offsets_all.cpu().tolist()
            offsets_q_list = offsets_q_all.cpu().tolist()
            counts_all_list = counts_all.cpu().tolist()

            for c0 in range(0, 512, 64):
                cnt = min(64, 512 - c0)
                c1 = c0 + cnt
                gu_c = reader.read_slice_dim0(gu_name, c0, cnt, dev, torch.bfloat16)
                dn_c = reader.read_slice_dim0(dn_name, c0, cnt, dev, torch.bfloat16)
                H_ec = torch.zeros((cnt, 640, 640), device=dev, dtype=torch.float32)
                for le_i in range(cnt):
                    ge = c0 + le_i
                    p0, p1 = offsets_all_list[ge], offsets_all_list[ge + 1]
                    c_n = p1 - p0
                    if c_n <= 0:
                        continue
                    xs = u_mlp_cal[sorted_tok_all[p0:p1]]
                    gue = F.linear(xs, gu_c[le_i])
                    hs = F.silu(gue[:, :640]) * gue[:, 640:]
                    hs_f = hs.float()
                    h_sum = hs_f.T @ hs_f
                    H_sum_all.add_(h_sum.double())
                    H_ec[le_i] = h_sum / float(c_n)
                    ye = F.linear(hs, dn_c[le_i]).float() * sorted_w_all[p0:p1]
                    cal_routed.index_add_(0, sorted_tok_all[p0:p1], ye)
                    del xs, gue, hs, hs_f, h_sum, ye
                H_e_all_cpu[c0:c1] = H_ec.cpu()
                del gu_c, dn_c, H_ec

            tot_hits_all = max(int(counts_all.sum().item()), 1)
            H_mean_global = (H_sum_all / float(tot_hits_all)).float()
            del H_sum_all
            t_pA = time.time() - t_pA0

            # ---- 6. Routed Experts Pass B: Quantize All Variants + Evaluate Held-Out + Advance Var 5 Cal ----
            t_pB0 = time.time()
            cal_routed_q = torch.zeros((T_cal, 2560), device=dev, dtype=torch.float32)
            eval_routed_bf16 = torch.zeros((T_eval, 2560), device=dev, dtype=torch.float32)
            eval_routed_vars = {vn: torch.zeros((T_eval, 2560), device=dev, dtype=torch.float32) for vn in VARIANT_NAMES}
            slot_buf = torch.zeros((T_eval, 10, 2560), device=dev, dtype=torch.float32)

            H_mlp_gpu = H_mlp_cpu.float().to(dev)
            gu_mse_acc = {"rtn": 0.0, "r3": 0.0, "cd": 0.0, "gptaq": 0.0}
            dn_mse_acc = {"rtn": 0.0, "r3": 0.0, "cd": 0.0, "gptaq": 0.0}

            for c0 in range(0, 512, expert_chunk):
                cnt = min(expert_chunk, 512 - c0)
                c1 = c0 + cnt
                gu_bf16 = reader.read_slice_dim0(gu_name, c0, cnt, dev, torch.bfloat16)
                dn_bf16 = reader.read_slice_dim0(dn_name, c0, cnt, dev, torch.bfloat16)
                H_e_c = H_e_all_cpu[c0:c1].to(dev)
                hits_c = counts_all[c0:c1]

                # 6a. Gate/Up quantization for all strategies (single ActOrder pass for r3 & cd)
                gu_rtn = sq.fake_quant(gu_bf16, "q4g128")
                gu_r3, c_gu, s_gu, m_gu = quant_gptq_actorder_static_with_params(gu_bf16, precomp=act_precomp_mlp, fmt="q4g128", damp=0.05, row_subbatch=40960)
                gu_cd = cd_refit_from_params(gu_bf16, c_gu, s_gu, m_gu, H_mlp_cpu, fmt="q4g128", row_subbatch=40960, n_rounds=2)
                del c_gu, s_gu, m_gu

                W_star_gu = compute_gptaq_target(gu_bf16, H_mlp_q_cpu, dG_mlp_cpu, alpha_aq=alpha_aq, damp_aq=damp_aq)
                gu_aq = quant_gptq_actorder_cd_refit(W_star_gu, precomp=act_precomp_mlp_q, H_cd_cpu_f64=H_mlp_q_cpu, fmt="q4g128", damp=0.05, row_subbatch=40960, n_rounds=2)
                del W_star_gu

                gu_mse_acc["rtn"] += _hmse(gu_rtn, gu_bf16, H_mlp_gpu)
                gu_mse_acc["r3"] += _hmse(gu_r3, gu_bf16, H_mlp_gpu)
                gu_mse_acc["cd"] += _hmse(gu_cd, gu_bf16, H_mlp_gpu)
                gu_mse_acc["gptaq"] += _hmse(gu_aq, gu_bf16, H_mlp_gpu)

                # 6b. Compute Var 5 asymmetric expert-down Hessians & DeltaG using gu_aq on u_mlp_q_cal
                # Note: evaluate paired BF16 vs Var5 hidden states on same BF16 routed tokens for exact DeltaG_e
                H_eq_c = torch.zeros((cnt, 640, 640), device=dev, dtype=torch.float32)
                dG_e_c = torch.zeros((cnt, 640, 640), device=dev, dtype=torch.float32)
                hits_q_pair = counts_all[c0:c1]
                for le_i in range(cnt):
                    ge = c0 + le_i
                    p0, p1 = offsets_all_list[ge], offsets_all_list[ge + 1]
                    c_n = p1 - p0
                    if c_n <= 0:
                        continue
                    toks = sorted_tok_all[p0:p1]
                    gb = F.linear(u_mlp_cal[toks], gu_bf16[le_i])
                    hb = (F.silu(gb[:, :640]) * gb[:, 640:]).float()
                    gq = F.linear(u_mlp_q_cal[toks], gu_aq[le_i])
                    hq = (F.silu(gq[:, :640]) * gq[:, 640:]).float()
                    H_eq_c[le_i] = (hq.T @ hq) / float(c_n)
                    dG_e_c[le_i] = ((hb - hq).T @ hq) / float(c_n)
                    del toks, gb, hb, gq, hq

                # Compute chunk-paired shrunk H_e_q & DeltaG_e, then GPTAQ target W_star_dn
                H_m_q_c = (H_eq_c * hits_q_pair.float().view(-1, 1, 1)).sum(dim=0) / max(float(counts_all_list[c0:c1] and sum(counts_all_list[c0:c1])), 1.0)
                W_star_dn = torch.empty_like(dn_bf16, dtype=torch.float32)
                for le_i in range(cnt):
                    hf = float(counts_all_list[c0 + le_i])
                    if hf > 0.0:
                        dG_reg = (hf / (hf + tau)) * dG_e_c[le_i]
                        Hq_reg = (hf * H_eq_c[le_i] + tau * H_m_q_c) / (hf + tau)
                        W_star_dn[le_i] = compute_gptaq_target(dn_bf16[le_i], Hq_reg.cpu(), dG_reg.cpu(), alpha_aq=alpha_aq, damp_aq=damp_aq)
                    else:
                        W_star_dn[le_i] = dn_bf16[le_i].float()

                # 6c. Down_proj quantization for all strategies (single ActOrder pass for r3 & cd)
                dn_rtn = sq.fake_quant(dn_bf16, "q4g128")
                dn_r3, dn_cd = quant_gptq_batched_exp_down_shrink_both(dn_bf16, H_e_c, hits_c, H_mean_global, tau=tau, base_damp=0.05, cd_n_rounds=2, fmt="q4g128")
                dn_aq = quant_gptq_batched_exp_down_shrink_wave2(W_star_dn, H_eq_c, hits_q_pair, H_m_q_c, tau=tau, base_damp=0.05, act_order=True, cd_refit=True, cd_n_rounds=2, fmt="q4g128")
                del W_star_dn, H_eq_c, dG_e_c, H_m_q_c

                for le_i in range(cnt):
                    hf = float(counts_all_list[c0 + le_i])
                    if hf <= 0.0:
                        continue
                    Hr_i = ((hf * H_e_c[le_i] + tau * H_mean_global) / (hf + tau))
                    dn_mse_acc["rtn"] += hf * _hmse(dn_rtn[le_i], dn_bf16[le_i], Hr_i)
                    dn_mse_acc["r3"] += hf * _hmse(dn_r3[le_i], dn_bf16[le_i], Hr_i)
                    dn_mse_acc["cd"] += hf * _hmse(dn_cd[le_i], dn_bf16[le_i], Hr_i)
                    dn_mse_acc["gptaq"] += hf * _hmse(dn_aq[le_i], dn_bf16[le_i], Hr_i)

                # 6d. Advance Var 5 calibration routed state with Var 5 true routing
                for le_i in range(cnt):
                    ge = c0 + le_i
                    p0, p1 = offsets_q_list[ge], offsets_q_list[ge + 1]
                    if p1 <= p0:
                        continue
                    toks_q = sorted_tok_q[p0:p1]
                    xs_q = u_mlp_q_cal[toks_q]
                    gq_e = F.linear(xs_q, gu_aq[le_i])
                    hq_e = F.silu(gq_e[:, :640]) * gq_e[:, 640:]
                    ye_q = F.linear(hq_e, dn_aq[le_i]).float() * sorted_w_q[p0:p1]
                    cal_routed_q.index_add_(0, toks_q, ye_q)
                    del toks_q, xs_q, gq_e, hq_e, ye_q

                # 6e. Evaluate Held-Out CALIB_DOC for BF16 + all 6 variants
                def _eval_chunk_route(u_in, route_t, wgu_c, wdn_c, dst_acc):
                    tidx, tw, _ = route_t
                    in_c = (tidx >= c0) & (tidx < c1)
                    if not in_c.any():
                        return
                    tok_pos, slot_pos = torch.nonzero(in_c, as_tuple=True)
                    loc_e = tidx[tok_pos, slot_pos] - c0
                    order = torch.argsort(loc_e, stable=True)
                    loc_s = loc_e[order]
                    tok_s = tok_pos[order]
                    slot_s = slot_pos[order]
                    w_s = tw[tok_pos, slot_pos][order].unsqueeze(-1)
                    uniq_e, cnts = torch.unique_consecutive(loc_s, return_counts=True)
                    x_hits = u_in[tok_s]
                    outs = torch.empty_like(x_hits)
                    pos = 0
                    for le_i, cnt_i in zip(uniq_e.tolist(), cnts.tolist()):
                        xs = x_hits[pos : pos + cnt_i]
                        gue = F.linear(xs, wgu_c[le_i])
                        hs = F.silu(gue[:, :640]) * gue[:, 640:]
                        outs[pos : pos + cnt_i] = F.linear(hs, wdn_c[le_i])
                        pos += cnt_i
                    slot_buf.zero_()
                    slot_buf[tok_s, slot_s] = outs.float() * w_s
                    dst_acc.add_(slot_buf.sum(dim=1))

                _eval_chunk_route(u_eb_mlp, eval_route_bf16, gu_bf16, dn_bf16, eval_routed_bf16)
                v_w_map = {
                    "0_RTN_Prod": (gu_rtn, dn_rtn),
                    "1_R3_ActOrder_Q4Only": (gu_r3, dn_r3),
                    "2_R3_plus_Q6_Hotspots": (gu_r3, dn_r3),
                    "3_R3_plus_CD_Refit": (gu_cd, dn_cd),
                    "4_R3_plus_Q6_CD_Refit": (gu_cd, dn_cd),
                    "5_GPTAQ_ActOrder_CD": (gu_aq, dn_aq),
                }
                for vn in VARIANT_NAMES:
                    wgu_v, wdn_v = v_w_map[vn]
                    _eval_chunk_route(u_ev_mlp[vn], eval_routes[vn], wgu_v, wdn_v, eval_routed_vars[vn])

                del gu_bf16, dn_bf16, H_e_c, hits_c, hits_q_pair
                del gu_rtn, gu_r3, gu_cd, gu_aq, dn_rtn, dn_r3, dn_cd, dn_aq

            del H_e_all_cpu, H_mean_global, H_mlp_gpu, slot_buf
            t_pB = time.time() - t_pB0

            for k_m in ("rtn", "r3", "cd", "gptaq"):
                hmse_totals["exp_gu"][k_m] += gu_mse_acc[k_m]
                hmse_totals["exp_down"][k_m] += dn_mse_acc[k_m] / max(float(tot_hits_all), 1.0)
            lstat["exp_gu_ratio_cd_vs_r3"] = round(gu_mse_acc["cd"] / max(gu_mse_acc["r3"], 1e-12), 4)
            lstat["exp_dn_ratio_cd_vs_r3"] = round(dn_mse_acc["cd"] / max(dn_mse_acc["r3"], 1e-12), 4)
            layer_metrics.append(lstat)

            # Advance calibration states + eval states cleanly
            cal_moe = (cal_routed + sh_out_cal).to(torch.bfloat16)
            cal_moe_q = (cal_routed_q + sh_out_q_cal).to(torch.bfloat16)
            if L + 1 < n_layers:
                for b0 in range(0, n_seqs, cal_mb):
                    b1 = min(b0 + cal_mb, n_seqs)
                    s0, s1 = b0 * seq_len, b1 * seq_len
                    xb = cal_states_cpu[b0:b1].to(dev)
                    ym = cal_moe[s0:s1].view(b1 - b0, seq_len, 1, 2560).float()
                    wb = win_mlp_cal[s0:s1].view(b1 - b0, seq_len, 4, 1)
                    cal_states_cpu[b0:b1] = (xb.float() + wb * ym).to(torch.bfloat16).cpu()

                    xq = cal_states_q_cpu[b0:b1].to(dev)
                    ymq = cal_moe_q[s0:s1].view(b1 - b0, seq_len, 1, 2560).float()
                    wbq = win_mlp_q_cal[s0:s1].view(b1 - b0, seq_len, 4, 1)
                    cal_states_q_cpu[b0:b1] = (xq.float() + wbq * ymq).to(torch.bfloat16).cpu()
                    del xb, ym, wb, xq, ymq, wbq

            ym_eb = (eval_routed_bf16 + eval_sh_bf16).to(torch.bfloat16)
            eval_bf16 = (eval_bf16.float() + win_eb_mlp.unsqueeze(-1) * ym_eb.float().unsqueeze(1)).to(torch.bfloat16)
            for vn in VARIANT_NAMES:
                ym_v = (eval_routed_vars[vn] + eval_sh_vars[vn]).to(torch.bfloat16)
                eval_vars[vn] = (eval_vars[vn].float() + win_ev_mlp[vn].unsqueeze(-1) * ym_v.float().unsqueeze(1)).to(torch.bfloat16)

            del u_mlp_cal, u_mlp_q_cal, win_mlp_cal, win_mlp_q_cal
            del H_mlp_cpu, H_mlp_q_cpu, dG_mlp_cpu, act_precomp_mlp, act_precomp_mlp_q
            del counts_all, offsets_all, sorted_tok_all, sorted_w_all
            del counts_q_all, offsets_q_all, sorted_tok_q, sorted_w_q
            del sh_out_cal, sh_out_q_cal, cal_routed, cal_routed_q, cal_moe, cal_moe_q
            del u_eb_mlp, win_eb_mlp, eval_route_bf16, eval_sh_bf16, eval_routed_bf16, ym_eb
            del u_ev_mlp, win_ev_mlp, eval_routes, eval_sh_vars, eval_routed_vars
            torch.cuda.empty_cache()

            peak_gb = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
            extra = (
                f"gdn_cd/r3={lstat.get('gdn_in_ratio_cd_vs_r3', 0):.3f} gout_ao/rtn={lstat.get('gdn_out_ratio_ao_vs_rtn', 0):.3f}"
                if not is_attn
                else f"ao_ao/rtn={lstat.get('attn_o_ratio_ao_vs_rtn', 0):.3f}"
            )
            print(
                f"  [Layer {L:2d}/{n_layers-1}] {'Attn' if is_attn else 'GDN '} t={time.time()-tl:4.1f}s (ag={t_ag:4.1f} pA={t_pA:4.1f} pB={t_pB:4.1f} | {time.time()-t0:6.1f}s) | "
                f"{extra} sh_ao/rtn={lstat['sh_down_ratio_ao_vs_rtn']:.3f} gu_cd/r3={lstat['exp_gu_ratio_cd_vs_r3']:.3f} dn_cd/r3={lstat['exp_dn_ratio_cd_vs_r3']:.3f} | VRAM={peak_gb:.2f}G",
                flush=True,
            )

        del cal_states_cpu, cal_states_q_cpu, cos_c, sin_c
        torch.cuda.empty_cache()

        # ---- 7. Final Hyper-Connection Mixer + Chunked LM Head KLD / PPL Evaluation ----
        fhc = lm + "hyper_connection_mixer."
        f_wd = reader.read_tensor(fhc + "input_mix_weight_down.weight", dev, torch.float32)
        f_wu = reader.read_tensor(fhc + "input_mix_weight_up.weight", dev, torch.float32)
        f_hn = reader.read_tensor(fhc + "hc_norm.weight", dev, torch.float32)
        f_wdf = f_wd * (1.0 + f_hn.unsqueeze(0))
        del f_wd
        dummy_inj = torch.zeros((4, 10240), device=dev, dtype=torch.float32)
        lm_w = reader.read_tensor("lm_head.weight", dev, torch.bfloat16)
        qlm_q8 = sq.fake_quant_rows(lm_w, "q8g64")

        f_bf16_u, _ = sq.hc_mix_down_up(eval_bf16, f_wdf.to(torch.bfloat16), dummy_inj, f_wu.to(torch.bfloat16), f_hn, inject=False)
        f_wd_q8 = sq.fake_quant(f_wdf, "q8g64")
        f_wu_q8 = sq.fake_quant(f_wu, "q8g64")

        TOK_CHUNK = 256
        n_eval = T_eval - 1
        bf16_nll_sum = 0.0
        for s0 in range(0, n_eval, TOK_CHUNK):
            s1 = min(s0 + TOK_CHUNK, n_eval)
            lgt_c = F.linear(f_bf16_u[s0:s1], lm_w).float()
            lp_c = F.log_softmax(lgt_c, dim=-1)
            bf16_nll_sum += F.nll_loss(lp_c, target_ids[s0:s1], reduction="sum").item()
            del lgt_c, lp_c
        bf16_ppl = math.exp(bf16_nll_sum / float(n_eval))

        results_table = []
        print(f"\n=== HELD-OUT RESULTS (T_eval={T_eval}, BF16 PPL={bf16_ppl:.3f}) ===", flush=True)
        print(f"{'Variant':<24} | {'Mean KLD':>10} | {'Med KLD':>10} | {'p95 KLD':>10} | {'PPL':>7} | {'dPPL':>7} | {'Top1%':>6} | {'Top5%':>6}", flush=True)
        print("-" * 102, flush=True)

        for vn in VARIANT_NAMES:
            u_cand, _ = sq.hc_mix_down_up(eval_vars[vn], f_wd_q8, dummy_inj, f_wu_q8, f_hn, inject=False)
            kld_parts = []
            nll_sum = 0.0
            t1_hits = 0
            t5_hits = 0
            for s0 in range(0, n_eval, TOK_CHUNK):
                s1 = min(s0 + TOK_CHUNK, n_eval)
                ref_lgt = F.linear(f_bf16_u[s0:s1], lm_w).float()
                ref_lp = F.log_softmax(ref_lgt, dim=-1)
                ref_p = ref_lp.exp()
                q_lgt = F.linear(u_cand[s0:s1], qlm_q8).float()
                q_lp = F.log_softmax(q_lgt, dim=-1)
                kld_parts.append((ref_p * (ref_lp - q_lp)).sum(dim=-1))
                nll_sum += F.nll_loss(q_lp, target_ids[s0:s1], reduction="sum").item()
                ref_t1 = ref_lgt.argmax(dim=-1)
                t1_hits += (q_lgt.argmax(dim=-1) == ref_t1).sum().item()
                t5_hits += (q_lgt.topk(5, dim=-1).indices == ref_t1.unsqueeze(-1)).any(dim=-1).sum().item()
                del ref_lgt, ref_lp, ref_p, q_lgt, q_lp
            kld_v = torch.cat(kld_parts, dim=0)
            mean_kld = float(kld_v.mean().item())
            med_kld = float(torch.median(kld_v).item())
            p95_kld = float(torch.quantile(kld_v, 0.95).item())
            ppl = math.exp(nll_sum / float(n_eval))
            dppl = ppl - bf16_ppl
            t1 = 100.0 * t1_hits / float(n_eval)
            t5 = 100.0 * t5_hits / float(n_eval)
            print(f"{vn:<24} | {mean_kld:10.6f} | {med_kld:10.6f} | {p95_kld:10.6f} | {ppl:7.3f} | {dppl:+7.3f} | {t1:5.1f}% | {t5:5.1f}%", flush=True)
            results_table.append({
                "variant": vn,
                "mean_kld": round(mean_kld, 6),
                "med_kld": round(med_kld, 6),
                "p95_kld": round(p95_kld, 6),
                "ppl": round(ppl, 4),
                "dppl": round(dppl, 4),
                "top1_pct": round(t1, 2),
                "top5_pct": round(t5, 2),
            })
            del u_cand

    reader.close()
    peak_vram_gb = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
    elapsed_s = time.time() - t0

    # Compute aggregate Hessian MSE summaries
    hmse_summary = {
        "gdn_in_cd_vs_r3": round(hmse_totals["gdn_in"]["cd"] / max(hmse_totals["gdn_in"]["r3"], 1e-12), 4),
        "gdn_in_r3_vs_rtn": round(hmse_totals["gdn_in"]["r3"] / max(hmse_totals["gdn_in"]["rtn"], 1e-12), 4),
        "gdn_out_ao_vs_rtn": round(hmse_totals["gdn_out"]["ao"] / max(hmse_totals["gdn_out"]["rtn"], 1e-12), 4),
        "attn_o_ao_vs_rtn": round(hmse_totals["attn_o"]["ao"] / max(hmse_totals["attn_o"]["rtn"], 1e-12), 4),
        "sh_down_ao_vs_rtn": round(hmse_totals["sh_down"]["ao"] / max(hmse_totals["sh_down"]["rtn"], 1e-12), 4),
        "exp_gu_cd_vs_r3": round(hmse_totals["exp_gu"]["cd"] / max(hmse_totals["exp_gu"]["r3"], 1e-12), 4),
        "exp_dn_cd_vs_r3": round(hmse_totals["exp_down"]["cd"] / max(hmse_totals["exp_down"]["r3"], 1e-12), 4),
    }
    print(f"\nAggregate Hessian MSE Ratios: {hmse_summary}", flush=True)
    print(f"Elapsed: {elapsed_s:.1f}s | Peak VRAM: {peak_vram_gb:.2f} GiB", flush=True)

    payload = {
        "model": model_dir,
        "n_layers": n_layers,
        "t_cal": T_cal,
        "cal_sha": cal_sha,
        "domains": dom_summary,
        "t_eval": T_eval,
        "bf16_ppl": round(bf16_ppl, 4),
        "tau": tau,
        "alpha_aq": alpha_aq,
        "damp_aq": damp_aq,
        "elapsed_s": round(elapsed_s, 1),
        "peak_vram_gib": round(peak_vram_gb, 2),
        "results": results_table,
        "hmse_summary": hmse_summary,
        "hmse_raw": hmse_totals,
        "layer_metrics": layer_metrics,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[SAVED] {out_json}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/nvme2/Thinking-Cap-Qwen-3.8-FN")
    ap.add_argument("--n-seqs", type=int, default=40)
    ap.add_argument("--seq-len", type=int, default=1536)
    ap.add_argument("--cal-mb", type=int, default=4)
    ap.add_argument("--expert-chunk", type=int, default=64)
    ap.add_argument("--tau", type=float, default=384.0)
    ap.add_argument("--alpha-aq", type=float, default=0.25)
    ap.add_argument("--damp-aq", type=float, default=0.10)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--self-test-only", action="store_true")
    ap.add_argument(
        "--out",
        default="/home/cheblan/.gemini/antigravity/brain/2db43efb-0097-4015-bf58-dedb55fd8ccc/scratch/qwen_wave2_61k_results.json",
    )
    args = ap.parse_args()

    if args.self_test_only:
        dev = torch.device("cuda:0")
        _ = torch.zeros(1, device=dev)
        torch.cuda.synchronize()
        run_wave2_self_tests(dev)
        return

    run_wave2_sim(
        model_dir=args.model,
        out_json=args.out,
        n_seqs=args.n_seqs,
        seq_len=args.seq_len,
        cal_mb=args.cal_mb,
        expert_chunk=args.expert_chunk,
        tau=args.tau,
        alpha_aq=args.alpha_aq,
        damp_aq=args.damp_aq,
        max_layers=args.layers,
    )


if __name__ == "__main__":
    main()
