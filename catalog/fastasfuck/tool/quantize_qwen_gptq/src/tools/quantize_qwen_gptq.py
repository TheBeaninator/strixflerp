#!/usr/bin/env python3
"""
Production GPU GPTQ Q4 Quantizer for Qwen 3.8 Flash Next (Thinking-Cap / Swift-1.5 / Base).

Produces pre-quantized Q4_G128 components (<faf_tensor>.{q, scale, minv}) matching
strixite/formats/q4.cpp bit-for-bit, along with compact reproducible calibration
metadata (calib_meta.json, calib_manifest.json, layer_stats.json).

Zero runtime speed penalty vs standard RTN .faf:
  - Keeps the exact 1_Current_Prod binary layout (exp_gu=q4g128, exp_down=q4g128,
    gdn_in=q4g128; all dense q6g64 / hc q8g64 / lm_head q8g64 / mtp.layers.0 untouched).
  - Full-rank ~61k-token Hessians:
      * gdn_in   [16480, 2560]: 61,440 >> 2,560
      * exp_gu   [512, 1280, 2560]: shared layer H_mlp (61,440 >> 2,560)
      * exp_down [512, 2560, 640]: normalized Empirical Bayes shrinkage across all 512 experts
        H_{e,reg} = (hits_e * H_e + tau * H_mean) / (hits_e + tau)
  - --mode actorder (default, -30.7% held-out KLD): descending diag(H) column order with
    strictly static_groups=True (zero g_idx, 100% compatible with linear_q4.hip).
  - --mode d02_imp (-23.5% held-out KLD, +0.000 dPPL): damp=0.02 + weighted group params
    on gdn_in; dynamic RTN-grid GPTQ (damp=0.05) on exp_gu & exp_down.

Usage:
  LD_PRELOAD=/usr/lib/libjemalloc.so ROCP_TOOLS_LIB="" HSA_TOOLS_LIB="" ROCP_HSA_INTERCEPT=0 \\
    python3 -u tools/quantize_qwen_gptq.py \\
      --src /mnt/nvme2/Thinking-Cap-Qwen-3.8-FN \\
      --out /mnt/nvme2/tmp-gptq-q4 --mode actorder
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
import struct
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_qwen_quant as sq
import sim_qwen_calib as sc
import sim_qwen_awq_gptq as sag
import sim_qwen_gptq_r3 as r3


# ---------------------------------------------------------------------------
# 1. Exact formats/q4.cpp Bit-Packing & Safetensors Serialization
# ---------------------------------------------------------------------------

def pack_q4_components(codes_u8: torch.Tensor, s_bf16: torch.Tensor, m_bf16: torch.Tensor):
    """
    Packs [R, K] uint8 codes (0..15), [R, K/G] bfloat16 scales, [R, K/G] bfloat16 mins
    into exact formats/q4.hpp byte layout:
      q[r, k/2] = low nibble even k | (high nibble odd k << 4).
    """
    R, K = codes_u8.shape
    assert K % 2 == 0 and s_bf16.shape == m_bf16.shape and s_bf16.shape[0] == R
    c = codes_u8.to(torch.uint8)
    packed = (c[:, 0::2] | (c[:, 1::2] << 4)).contiguous()
    return packed, s_bf16.to(torch.bfloat16).contiguous(), m_bf16.to(torch.bfloat16).contiguous()


def dequant_packed_q4_torch(q_u8: torch.Tensor, s_bf16: torch.Tensor, m_bf16: torch.Tensor, G: int = 128) -> torch.Tensor:
    """Exact inverse matching formats/q4.cpp dequantize_q4 (q*s then +m, no FMA)."""
    R, K2 = q_u8.shape
    K = K2 * 2
    lo = (q_u8 & 0x0F).to(torch.float32)
    hi = ((q_u8 >> 4) & 0x0F).to(torch.float32)
    codes = torch.stack([lo, hi], dim=-1).reshape(R, K)
    sf = s_bf16.float().repeat_interleave(G, dim=1)
    mf = m_bf16.float().repeat_interleave(G, dim=1)
    return codes * sf + mf


def write_safetensors(path: str, tensors: dict) -> int:
    """Writes a dict of {name: CPU torch.Tensor (U8 or BF16 or F32)} atomically."""
    header = {}
    chunks = []
    offset = 0
    for name in sorted(tensors.keys()):
        t = tensors[name].detach().cpu().contiguous()
        if t.dtype == torch.uint8:
            st_dt = "U8"
            raw = t.numpy().tobytes()
        elif t.dtype == torch.bfloat16:
            st_dt = "BF16"
            raw = t.view(torch.uint16).numpy().tobytes()
        elif t.dtype == torch.float32:
            st_dt = "F32"
            raw = t.numpy().tobytes()
        else:
            raise ValueError(f"Unsupported tensor dtype {t.dtype} for {name}")
        # 64-byte align tensor data
        pad = (-offset) & 63
        if pad:
            chunks.append(b"\x00" * pad)
            offset += pad
        header[name] = {
            "dtype": st_dt,
            "shape": list(t.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        chunks.append(raw)
        offset += len(raw)

    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hpad = (-len(hdr_bytes)) & 7
    if hpad:
        hdr_bytes += b" " * hpad

    tmp = path + ".partial"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        for c in chunks:
            f.write(c)
    os.replace(tmp, path)
    return 8 + len(hdr_bytes) + offset


# ---------------------------------------------------------------------------
# 2. Direct (q, scale, minv) GPTQ Kernels (ActOrder StaticG + d02_imp)
# ---------------------------------------------------------------------------

def gptq_actorder_static_pack(
    w_2d_or_3d: torch.Tensor,
    H_cpu_f64: torch.Tensor = None,
    precomp=None,
    G: int = 128,
    damp: float = 0.05,
    row_subbatch: int = 40960,
):
    """Returns packed CPU (q_u8, s_bf16, m_bf16) shaped [R, K/2], [R, K/G], [R, K/G]."""
    orig_shape = w_2d_or_3d.shape
    K = orig_shape[-1]
    assert K % G == 0
    levels = 15.0
    dev = w_2d_or_3d.device

    if precomp is None:
        assert H_cpu_f64 is not None
        U_p_cpu, perm_cpu, invperm_cpu = r3.prepare_actorder_chol_cpu(H_cpu_f64, damp=damp)
    else:
        U_p_cpu, perm_cpu, invperm_cpu = precomp

    U_p = U_p_cpu.to(dev)
    perm = perm_cpu.to(dev)
    invperm = invperm_cpu.to(dev)

    W_flat = w_2d_or_3d.reshape(-1, K)
    R = W_flat.shape[0]
    n_g = K // G

    q_cpu = torch.empty((R, K // 2), dtype=torch.uint8)
    s_cpu = torch.empty((R, n_g), dtype=torch.bfloat16)
    m_cpu = torch.empty((R, n_g), dtype=torch.bfloat16)

    for r0 in range(0, R, row_subbatch):
        r1 = min(r0 + row_subbatch, R)
        W_sub = W_flat[r0:r1].float()
        s_grp, m_grp, safe_s_grp = sag._rtn_minmax_params(W_sub.view(r1 - r0, n_g, G), levels)
        s_p = s_grp.repeat_interleave(G, dim=1)[:, perm]
        m_p = m_grp.repeat_interleave(G, dim=1)[:, perm]
        safe_sp = safe_s_grp.repeat_interleave(G, dim=1)[:, perm]

        W_p = W_sub[:, perm].clone()
        codes_p = torch.empty((r1 - r0, K), device=dev, dtype=torch.uint8)

        for b0 in range(0, K, G):
            b1 = b0 + G
            W_blk = W_p[:, b0:b1].clone()
            s_blk = s_p[:, b0:b1]
            m_blk = m_p[:, b0:b1]
            safe_s_blk = safe_sp[:, b0:b1]
            Err_blk = torch.zeros_like(W_blk)
            U_blk = U_p[b0:b1, b0:b1]

            for j in range(G):
                wj = W_blk[:, j]
                sj = s_blk[:, j]
                mj = m_blk[:, j]
                code = ((wj - mj) / safe_s_blk[:, j]).round().clamp(0.0, levels)
                code = torch.where(sj > 0.0, code, torch.zeros_like(code))
                codes_p[:, b0 + j] = code.to(torch.uint8)
                qj = code * sj + mj
                err = (wj - qj) / U_blk[j, j]
                Err_blk[:, j] = err
                if j + 1 < G:
                    W_blk[:, j + 1 :] -= err.unsqueeze(1) * U_blk[j, j + 1 :].unsqueeze(0)

            if b1 < K:
                W_p[:, b1:] -= Err_blk @ U_p[b0:b1, b1:]

        codes_orig = codes_p[:, invperm]
        q_blk, sb, mb = pack_q4_components(codes_orig, s_grp.to(torch.bfloat16), m_grp.to(torch.bfloat16))
        q_cpu[r0:r1] = q_blk.cpu()
        s_cpu[r0:r1] = sb.cpu()
        m_cpu[r0:r1] = mb.cpu()
        del W_sub, s_grp, m_grp, safe_s_grp, s_p, m_p, safe_sp, W_p, codes_p, codes_orig, q_blk, sb, mb

    return q_cpu, s_cpu, m_cpu


def gptq_rtn_pack(
    w_2d_or_3d: torch.Tensor,
    H_cpu_f64: torch.Tensor = None,
    U_gpu: torch.Tensor = None,
    G: int = 128,
    damp: float = 0.02,
    imp_clip: bool = False,
    row_subbatch: int = 40960,
):
    """Sequential column-order GPTQ (RTN min-max or imp_clip) returning packed CPU (q, scale, minv)."""
    orig_shape = w_2d_or_3d.shape
    K = orig_shape[-1]
    assert K % G == 0
    levels = 15.0
    dev = w_2d_or_3d.device

    if U_gpu is None:
        assert H_cpu_f64 is not None
        U_gpu = sc.compute_hinv_chol_cpu(H_cpu_f64, damp=damp).to(dev)

    diag_h = None
    if imp_clip:
        assert H_cpu_f64 is not None
        dh = torch.diag(H_cpu_f64).to(device=dev, dtype=torch.float32).clamp(min=1e-12)
        dh = dh.view(1, K // G, G)
        diag_h = dh / (dh.mean(dim=-1, keepdim=True) + 1e-12)

    W_flat = w_2d_or_3d.reshape(-1, K)
    R = W_flat.shape[0]
    n_g = K // G

    q_cpu = torch.empty((R, K // 2), dtype=torch.uint8)
    s_cpu = torch.empty((R, n_g), dtype=torch.bfloat16)
    m_cpu = torch.empty((R, n_g), dtype=torch.bfloat16)

    for r0 in range(0, R, row_subbatch):
        r1 = min(r0 + row_subbatch, R)
        W = W_flat[r0:r1].float().clone()
        codes = torch.empty((r1 - r0, K), device=dev, dtype=torch.uint8)
        s_grp = torch.empty((r1 - r0, n_g), device=dev, dtype=torch.bfloat16)
        m_grp = torch.empty((r1 - r0, n_g), device=dev, dtype=torch.bfloat16)

        for g_idx, b0 in enumerate(range(0, K, G)):
            b1 = b0 + G
            W_blk = W[:, b0:b1].clone()
            if imp_clip:
                uf_blk = diag_h[:, g_idx, :].expand(r1 - r0, G)
                _, s_b, m_b = sc.quant_group_params(W_blk, f"q4g{G}", uf=uf_blk, ls_iters=2)
                s_vec = s_b.squeeze(-1)
                m_vec = m_b.squeeze(-1)
                safe_s = torch.where(s_vec > 0.0, s_vec, torch.ones_like(s_vec))
            else:
                s_vec, m_vec, safe_s = sag._rtn_minmax_params(W_blk, levels)

            s_grp[:, g_idx] = s_vec.to(torch.bfloat16)
            m_grp[:, g_idx] = m_vec.to(torch.bfloat16)
            Err_blk = torch.zeros_like(W_blk)
            U_blk = U_gpu[b0:b1, b0:b1]

            for j in range(G):
                wj = W_blk[:, j]
                code = ((wj - m_vec) / safe_s).round().clamp(0.0, levels)
                code = torch.where(s_vec > 0.0, code, torch.zeros_like(code))
                codes[:, b0 + j] = code.to(torch.uint8)
                qj = code * s_vec + m_vec
                err = (wj - qj) / U_blk[j, j]
                Err_blk[:, j] = err
                if j + 1 < G:
                    W_blk[:, j + 1 :] -= err.unsqueeze(1) * U_blk[j, j + 1 :].unsqueeze(0)

            if b1 < K:
                W[:, b1:] -= Err_blk @ U_gpu[b0:b1, b1:]

        q_blk, sb, mb = pack_q4_components(codes, s_grp, m_grp)
        q_cpu[r0:r1] = q_blk.cpu()
        s_cpu[r0:r1] = sb.cpu()
        m_cpu[r0:r1] = mb.cpu()
        del W, codes, s_grp, m_grp, q_blk, sb, mb

    return q_cpu, s_cpu, m_cpu


def gptq_exp_down_shrink_pack(
    W_dn: torch.Tensor,         # [cnt, 2560, 640] BF16
    H_e_chunk: torch.Tensor,    # [cnt, 640, 640] FP32
    hits_chunk: torch.Tensor,   # [cnt] int64
    H_mean: torch.Tensor,       # [640, 640] FP32
    tau: float = 384.0,
    base_damp: float = 0.05,
    act_order: bool = True,
    G: int = 128,
):
    """Returns packed CPU (q_u8 [cnt*2560, 320], s_bf16 [cnt*2560, 5], m_bf16 [cnt*2560, 5])."""
    cnt, N, K = W_dn.shape
    levels = 15.0
    dev = W_dn.device
    n_g = K // G

    H_e_cpu = H_e_chunk.double().cpu()
    H_m_cpu = H_mean.double().cpu()
    hits_list = hits_chunk.tolist()

    U_list, perm_list, invperm_list = [], [], []
    for k_i, h_cnt in enumerate(hits_list):
        h_f = float(h_cnt)
        H_reg = (h_f * H_e_cpu[k_i] + tau * H_m_cpu) / (h_f + tau) if h_f > 0.0 else H_m_cpu.clone()
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
        s_grp = torch.empty((cnt, N, n_g), device=dev, dtype=torch.bfloat16)
        m_grp = torch.empty((cnt, N, n_g), device=dev, dtype=torch.bfloat16)

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
                code = ((wj - m_vec) / safe_s).round().clamp(0.0, levels)
                code = torch.where(s_vec > 0.0, code, torch.zeros_like(code))
                codes[:, :, b0 + j] = code.to(torch.uint8)
                qj = code * s_vec + m_vec
                err = (wj - qj) / U_blk[:, j, j].unsqueeze(1)
                Err_blk[:, :, j] = err
                if j + 1 < G:
                    W_blk[:, :, j + 1 :] -= err.unsqueeze(2) * U_blk[:, j, j + 1 :].unsqueeze(1)

            if b1 < K:
                W_b[:, :, b1:] -= torch.bmm(Err_blk, U_b[:, b0:b1, b1:])

        q_blk, sb, mb = pack_q4_components(codes.reshape(-1, K), s_grp.reshape(-1, n_g), m_grp.reshape(-1, n_g))
        return q_blk.cpu(), sb.cpu(), mb.cpu()

    # ActOrder + static_groups=True
    perm_b = torch.stack(perm_list, dim=0).to(dev).unsqueeze(1).expand(cnt, N, K)
    invperm_b = torch.stack(invperm_list, dim=0).to(dev).unsqueeze(1).expand(cnt, N, K)

    s_grp, m_grp, safe_s_grp = sag._rtn_minmax_params(W_orig.view(cnt, N, n_g, G), levels)
    s_p = torch.gather(s_grp.repeat_interleave(G, dim=2), 2, perm_b)
    m_p = torch.gather(m_grp.repeat_interleave(G, dim=2), 2, perm_b)
    safe_sp = torch.gather(safe_s_grp.repeat_interleave(G, dim=2), 2, perm_b)

    W_p = torch.gather(W_orig, 2, perm_b).clone()
    codes_p = torch.empty((cnt, N, K), device=dev, dtype=torch.uint8)

    for b0 in range(0, K, G):
        b1 = b0 + G
        W_blk = W_p[:, :, b0:b1].clone()
        s_blk = s_p[:, :, b0:b1]
        m_blk = m_p[:, :, b0:b1]
        safe_s_blk = safe_sp[:, :, b0:b1]
        Err_blk = torch.zeros_like(W_blk)
        U_blk = U_b[:, b0:b1, b0:b1]

        for j in range(G):
            wj = W_blk[:, :, j]
            sj = s_blk[:, :, j]
            mj = m_blk[:, :, j]
            code = ((wj - mj) / safe_s_blk[:, :, j]).round().clamp(0.0, levels)
            code = torch.where(sj > 0.0, code, torch.zeros_like(code))
            codes_p[:, :, b0 + j] = code.to(torch.uint8)
            qj = code * sj + mj
            err = (wj - qj) / U_blk[:, j, j].unsqueeze(1)
            Err_blk[:, :, j] = err
            if j + 1 < G:
                W_blk[:, :, j + 1 :] -= err.unsqueeze(2) * U_blk[:, j, j + 1 :].unsqueeze(1)

        if b1 < K:
            W_p[:, :, b1:] -= torch.bmm(Err_blk, U_b[:, b0:b1, b1:])

    codes_orig = torch.gather(codes_p, 2, invperm_b)
    q_blk, sb, mb = pack_q4_components(codes_orig.reshape(-1, K), s_grp.reshape(-1, n_g).to(torch.bfloat16), m_grp.reshape(-1, n_g).to(torch.bfloat16))
    return q_blk.cpu(), sb.cpu(), mb.cpu()


# ---------------------------------------------------------------------------
# 3. Pre-Run Bit-Identity Self-Tests
# ---------------------------------------------------------------------------

def run_packer_self_tests(dev: torch.device, tmp_dir: str) -> None:
    torch.manual_seed(99)
    X = (torch.randn(512, 256, dtype=torch.float64) @ (torch.randn(256, 256, dtype=torch.float64) * 0.35 + torch.eye(256, dtype=torch.float64)))
    H = (X.T @ X) / 512.0
    W = torch.randn(16, 256, device=dev, dtype=torch.float32) * 0.04

    # 1. ActOrder StaticG packer bit-exact vs r3.quant_gptq_actorder_static
    q1, s1, m1 = gptq_actorder_static_pack(W, H_cpu_f64=H, G=128, damp=0.05, row_subbatch=8)
    Q_deq1 = dequant_packed_q4_torch(q1, s1, m1, G=128).to(torch.bfloat16)
    Q_ref1 = r3.quant_gptq_actorder_static(W, H_cpu_f64=H, fmt="q4g128", damp=0.05, row_subbatch=8).cpu()
    d1 = float((Q_deq1.float() - Q_ref1.float()).abs().max().item())
    assert d1 == 0.0, f"ActOrder pack mismatch: {d1}"

    # 2. d02_imp packer bit-exact vs sc.quant_gptq
    q2, s2, m2 = gptq_rtn_pack(W, H_cpu_f64=H, G=128, damp=0.02, imp_clip=True, row_subbatch=8)
    Q_deq2 = dequant_packed_q4_torch(q2, s2, m2, G=128).to(torch.bfloat16)
    Q_ref2 = sc.quant_gptq(W, H, "q4g128", damp=0.02, imp_clip=True).cpu()
    d2 = float((Q_deq2.float() - Q_ref2.float()).abs().max().item())
    assert d2 == 0.0, f"imp_clip pack mismatch: {d2}"

    # 3. ExpDown Shrink512 packer bit-exact vs r3.quant_gptq_batched_exp_down_shrink
    W3 = torch.randn(3, 16, 256, device=dev, dtype=torch.float32) * 0.04
    hits3 = torch.tensor([1000, 150, 0], device=dev, dtype=torch.long)
    He3 = torch.stack([H.float(), (H * 1.2).float(), torch.zeros_like(H.float())], dim=0).to(dev)
    Hm3 = H.float().to(dev)
    for ao in (False, True):
        qp, sp, mp = gptq_exp_down_shrink_pack(W3, He3, hits3, Hm3, tau=384.0, base_damp=0.05, act_order=ao, G=128)
        Q_deq3 = dequant_packed_q4_torch(qp, sp, mp, G=128).reshape(3, 16, 256).to(torch.bfloat16)
        Q_ref3 = r3.quant_gptq_batched_exp_down_shrink(W3, He3, hits3, Hm3, tau=384.0, base_damp=0.05, act_order=ao, fmt="q4g128").cpu()
        d3 = float((Q_deq3.float() - Q_ref3.float()).abs().max().item())
        assert d3 == 0.0, f"ExpDown pack mismatch (act_order={ao}): {d3}"

    # 4. Safetensors roundtrip check
    st_path = os.path.join(tmp_dir, "_selftest.safetensors")
    write_safetensors(st_path, {"test.q": q1, "test.scale": s1, "test.minv": m1})
    with open(st_path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    assert hdr["test.q"]["dtype"] == "U8" and hdr["test.scale"]["dtype"] == "BF16" and hdr["test.minv"]["dtype"] == "BF16"
    os.remove(st_path)
    print(f"[PASS] Packer self-tests: ActOrder diff={d1:.1e}, d02_imp diff={d2:.1e}, ExpDown shrink diff=0.0e+00, safetensors OK")


# ---------------------------------------------------------------------------
# 4. Production Layer-by-Layer Quantizer
# ---------------------------------------------------------------------------

def build_gptq_q4(
    src_dir: str,
    out_dir: str,
    mode: str = "actorder",
    n_seqs: int = 40,
    seq_len: int = 1536,
    cal_mb: int = 4,
    expert_chunk: int = 64,
    tau: float = 384.0,
    max_layers: int = None,
):
    assert mode in ("actorder", "d02_imp"), f"Unknown mode {mode}"
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device("cuda:0")
    _ = torch.zeros(1, device=dev)
    torch.cuda.synchronize()
    cpu = torch.device("cpu")
    torch.cuda.reset_peak_memory_stats(dev)

    run_packer_self_tests(dev, out_dir)

    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    tok = sq.QwenTokenizer(os.path.join(src_dir, "tokenizer.json"))
    ids_cal_2d, dom_summary, cal_sha = sag.build_diverse_calib_sequences(tok, workspace_root, n_seqs=n_seqs, seq_len=seq_len)
    T_cal = n_seqs * seq_len

    calib_manifest = {
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "t_cal": T_cal,
        "calib_sha256_16": cal_sha,
        "domains": dom_summary,
        "disjoint_checked_against": ["sim_qwen_quant.CALIB_DOC", "AGENTS_P1_to_P6_sentinels"],
    }
    calib_meta = {
        "method": f"gptq_{mode}",
        "mode": mode,
        "targets": ["gdn_in:q4g128", "exp_gu:q4g128", "exp_down:q4g128"],
        "untouched": ["dense:q6g64", "hc:q8g64", "lm_head:q8g64", "mtp.layers.0"],
        "t_cal": T_cal,
        "calib_sha256_16": cal_sha,
        "exp_down_shrink_tau": tau,
        "gdn_damp": 0.05 if mode == "actorder" else 0.02,
        "moe_damp": 0.05,
        "act_order": (mode == "actorder"),
        "static_groups": True,
    }
    with open(os.path.join(out_dir, "calib_meta.json"), "w") as f:
        json.dump(calib_meta, f, separators=(",", ":"))
    with open(os.path.join(out_dir, "calib_manifest.json"), "w") as f:
        json.dump(calib_manifest, f, indent=2)

    reader = sq.SafetensorsReader(src_dir)
    lm = "model.language_model."
    with open(os.path.join(src_dir, "config.json")) as f:
        cfg_root = json.load(f)
    tc = cfg_root.get("text_config", cfg_root)
    n_layers = min(tc["num_hidden_layers"], max_layers) if max_layers is not None else tc["num_hidden_layers"]
    layer_types = tc.get("layer_types")
    cos_c, sin_c = sq.build_qwen_rope(seq_len, dev, torch.bfloat16)

    print(f"Building {mode} GPTQ Q4 tensors for {n_layers} layers -> {out_dir}")
    print(f"Corpus: {n_seqs}x{seq_len}={T_cal} tokens (sha={cal_sha}) | domains={dom_summary} | tau={tau}")
    t0 = time.time()
    layer_stats = []

    with torch.no_grad():
        emb_cpu = reader.read_tensor(lm + "embed_tokens.weight", cpu, torch.bfloat16)
        cal_states_cpu = F.embedding(ids_cal_2d, emb_cpu).unsqueeze(2).expand(n_seqs, seq_len, 4, 2560).contiguous()
        del emb_cpu
        torch.cuda.empty_cache()

        for L in range(n_layers):
            tl = time.time()
            P = f"{lm}layers.{L}."
            is_attn = (layer_types[L] == "full_attention") if layer_types is not None else ((L + 1) % 4 == 0)
            out_tensors = {}
            st = {"layer": L, "type": "attn" if is_attn else "gdn"}

            # Optional PLE Injection (Layer 1)
            if (P + "ple.key_proj.weight") in reader.wm:
                ple_p = P + "ple."
                mult = reader.read_tensor(ple_p + "ple_embedding.layer_multipliers", cpu).tolist()
                vsz = reader.read_tensor(ple_p + "ple_embedding.ngram_heads_vocab_sizes", cpu).tolist()
                offs = reader.read_tensor(ple_p + "ple_embedding.ngram_heads_offsets", cpu).tolist()
                w_kp = reader.read_tensor(ple_p + "key_proj.weight", dev, torch.bfloat16)
                w_vp = reader.read_tensor(ple_p + "value_proj.weight", dev, torch.bfloat16)
                w_ple_bf16 = torch.cat([w_kp, w_vp], dim=0)
                del w_kp, w_vp
                ple_cw = reader.read_tensor(ple_p + "conv1d.weight", dev, torch.float32)
                ple_nk = reader.read_tensor(ple_p + "norm_key.weight", dev, torch.float32)
                ple_nq = reader.read_tensor(ple_p + "norm_query.weight", dev, torch.float32)
                ple_nc = reader.read_tensor(ple_p + "norm_conv.weight", dev, torch.float32)

                for b_idx in range(n_seqs):
                    seq_b = ids_cal_2d[b_idx].tolist()
                    ur_c, mi_c = sq.ple_ngram_ids_py(mult, vsz, offs, seq_b, eos=248044)
                    ple_c = reader.gather_ple_rows(ple_p, ur_c, dev)[torch.tensor(mi_c, device=dev, dtype=torch.long)].reshape(seq_len, 2560)
                    xc_b = cal_states_cpu[b_idx].to(dev)
                    cal_states_cpu[b_idx] = sq.ple_forward(xc_b, ple_c, w_ple_bf16, ple_cw, ple_nk, ple_nq, ple_nc).cpu()
                    del ple_c, xc_b
                del w_ple_bf16, ple_cw, ple_nk, ple_nq, ple_nc

            # Attn / GDN Hyper-Connection Mix (BF16 reference path)
            ahc = P + "attn_hyper_connection."
            a_wd = reader.read_tensor(ahc + "input_mix_weight_down.weight", dev, torch.float32)
            a_wi = reader.read_tensor(ahc + "block_inject_weight.weight", dev, torch.float32)
            a_wu = reader.read_tensor(ahc + "input_mix_weight_up.weight", dev, torch.bfloat16)
            a_hn = reader.read_tensor(ahc + "hc_norm.weight", dev, torch.float32)
            ahcd_bf16 = ((a_wd * (1.0 + a_hn.unsqueeze(0))).to(torch.bfloat16), (a_wi * (1.0 + a_hn.unsqueeze(0))).to(torch.bfloat16))
            del a_wd, a_wi

            if not is_attn:
                gp = P + "linear_attn."
                w_qkv = reader.read_tensor(gp + "in_proj_qkv.weight", dev, torch.float32)
                w_z = reader.read_tensor(gp + "in_proj_z.weight", dev, torch.float32)
                w_b = reader.read_tensor(gp + "in_proj_b.weight", dev, torch.float32)
                w_a = reader.read_tensor(gp + "in_proj_a.weight", dev, torch.float32)
                w_gdn_in = torch.cat([w_qkv, w_z, w_b, w_a], dim=0)
                del w_qkv, w_z, w_b, w_a
                gin_bf16 = w_gdn_in.to(torch.bfloat16)
                gout_bf16 = reader.read_tensor(gp + "out_proj.weight", dev, torch.bfloat16)
                cw = reader.read_tensor(gp + "conv1d.weight", dev, torch.float32)
                A_log = reader.read_tensor(gp + "A_log", dev, torch.float32)
                dt_bias = reader.read_tensor(gp + "dt_bias", dev, torch.float32)
                gnorm = reader.read_tensor(gp + "norm.weight", dev, torch.float32)

                H_gdn_cpu = torch.zeros((2560, 2560), dtype=torch.float64)
                for b0 in range(0, n_seqs, cal_mb):
                    b1 = min(b0 + cal_mb, n_seqs)
                    xb = cal_states_cpu[b0:b1].to(dev)
                    ub, win_b = sag.hc_mix_down_up_batched(xb, *ahcd_bf16, a_wu, a_hn, inject=True)
                    u_flat = ub.reshape(-1, 2560).double()
                    H_gdn_cpu.add_((u_flat.T @ u_flat).cpu())
                    ym = sag.gdn_forward_batched(ub, gin_bf16, gout_bf16, cw, A_log, dt_bias, gnorm)
                    cal_states_cpu[b0:b1] = (xb.float() + win_b.unsqueeze(-1) * ym.float().unsqueeze(2)).to(torch.bfloat16).cpu()
                    del xb, ub, win_b, u_flat, ym
                H_gdn_cpu.div_(float(T_cal))

                if mode == "actorder":
                    qg, sg, mg = gptq_actorder_static_pack(w_gdn_in, H_cpu_f64=H_gdn_cpu, G=128, damp=0.05, row_subbatch=40960)
                else:
                    qg, sg, mg = gptq_rtn_pack(w_gdn_in, H_cpu_f64=H_gdn_cpu, G=128, damp=0.02, imp_clip=True, row_subbatch=40960)

                base_g = f"{P}linear_attn.in_proj"
                out_tensors[base_g + ".q"] = qg
                out_tensors[base_g + ".scale"] = sg
                out_tensors[base_g + ".minv"] = mg

                # Quick calibration Hessian MSE ratio + sample RMS error
                dq_g = dequant_packed_q4_torch(qg.to(dev), sg.to(dev), mg.to(dev), G=128)
                dq_rtn = sq.fake_quant(w_gdn_in, "q4g128").float()
                H_gpu = H_gdn_cpu.float().to(dev)
                eg = dq_g - w_gdn_in
                er = dq_rtn - w_gdn_in
                st["gdn_hessian_mse_ratio"] = round(float(((eg @ H_gpu) * eg).sum().item()) / max(float(((er @ H_gpu) * er).sum().item()), 1e-12), 4)
                st["gdn_rms_err_pct"] = round(100.0 * float((eg.norm() / w_gdn_in.norm()).item()), 3)

                del w_gdn_in, gin_bf16, gout_bf16, cw, A_log, dt_bias, gnorm, H_gdn_cpu, H_gpu, dq_g, dq_rtn, eg, er
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

                for b0 in range(0, n_seqs, cal_mb):
                    b1 = min(b0 + cal_mb, n_seqs)
                    xb = cal_states_cpu[b0:b1].to(dev)
                    ub, win_b = sag.hc_mix_down_up_batched(xb, *ahcd_bf16, a_wu, a_hn, inject=True)
                    ym = sag.full_attn_forward_batched(ub, qkv_bf16, ao_bf16, qn, kn, cos_c, sin_c)
                    cal_states_cpu[b0:b1] = (xb.float() + win_b.unsqueeze(-1) * ym.float().unsqueeze(2)).to(torch.bfloat16).cpu()
                    del xb, ub, win_b, ym
                del qkv_bf16, ao_bf16, qn, kn

            del ahcd_bf16, a_wu, a_hn

            # MLP Hyper-Connection Mix (BF16 reference path)
            mhc = P + "mlp_hyper_connection."
            m_wd = reader.read_tensor(mhc + "input_mix_weight_down.weight", dev, torch.float32)
            m_wi = reader.read_tensor(mhc + "block_inject_weight.weight", dev, torch.float32)
            m_wu = reader.read_tensor(mhc + "input_mix_weight_up.weight", dev, torch.bfloat16)
            m_hn = reader.read_tensor(mhc + "hc_norm.weight", dev, torch.float32)
            mhcd_bf16 = ((m_wd * (1.0 + m_hn.unsqueeze(0))).to(torch.bfloat16), (m_wi * (1.0 + m_hn.unsqueeze(0))).to(torch.bfloat16))
            del m_wd, m_wi

            u_mlp_cal = torch.empty((T_cal, 2560), device=dev, dtype=torch.bfloat16)
            win_mlp_cal = torch.empty((T_cal, 4), device=dev, dtype=torch.float32)
            H_mlp_cpu = torch.zeros((2560, 2560), dtype=torch.float64)
            for b0 in range(0, n_seqs, cal_mb):
                b1 = min(b0 + cal_mb, n_seqs)
                xb = cal_states_cpu[b0:b1].to(dev)
                ub, win_b = sag.hc_mix_down_up_batched(xb, *mhcd_bf16, m_wu, m_hn, inject=True)
                u_flat = ub.reshape(-1, 2560)
                s0, s1 = b0 * seq_len, b1 * seq_len
                u_mlp_cal[s0:s1] = u_flat
                win_mlp_cal[s0:s1] = win_b.reshape(-1, 4)
                uf64 = u_flat.double()
                H_mlp_cpu.add_((uf64.T @ uf64).cpu())
                del xb, ub, win_b, u_flat, uf64

            H_mlp_cpu.div_(float(T_cal))
            if mode == "actorder":
                act_precomp_mlp = r3.prepare_actorder_chol_cpu(H_mlp_cpu, damp=0.05)
                U_mlp_seq = None
            else:
                act_precomp_mlp = None
                U_mlp_seq = sc.compute_hinv_chol_cpu(H_mlp_cpu, damp=0.05).to(dev)
            del H_mlp_cpu, mhcd_bf16, m_wu, m_hn

            # Router + Shared Expert
            mp = P + "mlp."
            wg_r = reader.read_tensor(mp + "gate.weight", dev, torch.bfloat16)
            wsg_r = reader.read_tensor(mp + "shared_expert_gate.weight", dev, torch.bfloat16)
            w_router_all = torch.cat([wg_r, wsg_r], dim=0)
            del wg_r, wsg_r

            cal_route = sq.moe_route_qwen(u_mlp_cal, w_router_all)
            tidx_flat = cal_route[0].reshape(-1)
            tw_flat = cal_route[1].reshape(-1)
            ord_all = torch.argsort(tidx_flat, stable=True)
            sorted_tok_all = ord_all // 10
            sorted_w_all = tw_flat[ord_all].unsqueeze(-1)
            counts_all = torch.bincount(tidx_flat, minlength=512)
            offsets_all = torch.zeros(513, device=dev, dtype=torch.long)
            torch.cumsum(counts_all, dim=0, out=offsets_all[1:])
            del w_router_all, tidx_flat, tw_flat, ord_all

            w_sh_g = reader.read_tensor(mp + "shared_expert.gate_proj.weight", dev, torch.bfloat16)
            w_sh_u = reader.read_tensor(mp + "shared_expert.up_proj.weight", dev, torch.bfloat16)
            w_sh_gu = torch.cat([w_sh_g, w_sh_u], dim=0)
            del w_sh_g, w_sh_u
            w_sh_d = reader.read_tensor(mp + "shared_expert.down_proj.weight", dev, torch.bfloat16)
            sh_gu_c = F.linear(u_mlp_cal, w_sh_gu)
            sh_h_c = F.silu(sh_gu_c[:, :640]) * sh_gu_c[:, 640:]
            sh_out_cal = F.linear(sh_h_c, w_sh_d).float() * cal_route[2].unsqueeze(-1)
            del w_sh_gu, w_sh_d, sh_gu_c, sh_h_c, cal_route

            # Preallocate CPU layer arrays for experts_gate_up [655360, ...] and experts_down [1310720, ...]
            gu_q_cpu = torch.empty((512 * 1280, 1280), dtype=torch.uint8)
            gu_s_cpu = torch.empty((512 * 1280, 20), dtype=torch.bfloat16)
            gu_m_cpu = torch.empty((512 * 1280, 20), dtype=torch.bfloat16)
            dn_q_cpu = torch.empty((512 * 2560, 320), dtype=torch.uint8)
            dn_s_cpu = torch.empty((512 * 2560, 5), dtype=torch.bfloat16)
            dn_m_cpu = torch.empty((512 * 2560, 5), dtype=torch.bfloat16)

            cal_routed = torch.zeros((T_cal, 2560), device=dev, dtype=torch.float32)
            gu_name = mp + "experts.gate_up_proj"
            dn_name = mp + "experts.down_proj"
            CHUNK = expert_chunk

            for c0 in range(0, 512, CHUNK):
                cnt = min(CHUNK, 512 - c0)
                c1 = c0 + cnt
                gu_bf16 = reader.read_slice_dim0(gu_name, c0, cnt, dev, torch.bfloat16)
                dn_bf16 = reader.read_slice_dim0(dn_name, c0, cnt, dev, torch.bfloat16)

                H_sum_chunk = torch.zeros((cnt, 640, 640), device=dev, dtype=torch.float32)
                H_e_chunk = torch.zeros((cnt, 640, 640), device=dev, dtype=torch.float32)
                hits_chunk = counts_all[c0:c1]

                for le_i in range(cnt):
                    ge = c0 + le_i
                    p0 = int(offsets_all[ge].item())
                    p1 = int(offsets_all[ge + 1].item())
                    c_n = p1 - p0
                    if c_n <= 0:
                        continue
                    xs = u_mlp_cal[sorted_tok_all[p0:p1]]
                    gue = F.linear(xs, gu_bf16[le_i])
                    hs = F.silu(gue[:, :640]) * gue[:, 640:]
                    hs_f = hs.float()
                    h_sum = hs_f.T @ hs_f
                    H_sum_chunk[le_i] = h_sum
                    H_e_chunk[le_i] = h_sum / float(c_n)

                    ye = F.linear(hs, dn_bf16[le_i]).float() * sorted_w_all[p0:p1]
                    cal_routed.index_add_(0, sorted_tok_all[p0:p1], ye)
                    del xs, gue, hs, hs_f, h_sum, ye

                tot_hits_chunk = max(int(hits_chunk.sum().item()), 1)
                H_mean_chunk = H_sum_chunk.sum(dim=0) / float(tot_hits_chunk)
                del H_sum_chunk

                dq, ds, dm = gptq_exp_down_shrink_pack(
                    dn_bf16, H_e_chunk, hits_chunk, H_mean_chunk,
                    tau=tau, base_damp=0.05, act_order=(mode == "actorder"), G=128,
                )
                r0_d, r1_d = c0 * 2560, c1 * 2560
                dn_q_cpu[r0_d:r1_d] = dq
                dn_s_cpu[r0_d:r1_d] = ds
                dn_m_cpu[r0_d:r1_d] = dm
                del dn_bf16, H_e_chunk, H_mean_chunk, dq, ds, dm

                if mode == "actorder":
                    gq, gs, gm = gptq_actorder_static_pack(gu_bf16, precomp=act_precomp_mlp, G=128, damp=0.05, row_subbatch=40960)
                else:
                    gq, gs, gm = gptq_rtn_pack(gu_bf16, U_gpu=U_mlp_seq, G=128, damp=0.05, imp_clip=False, row_subbatch=40960)
                r0_g, r1_g = c0 * 1280, c1 * 1280
                gu_q_cpu[r0_g:r1_g] = gq
                gu_s_cpu[r0_g:r1_g] = gs
                gu_m_cpu[r0_g:r1_g] = gm
                del gu_bf16, gq, gs, gm

            base_gu = f"{P}mlp.experts_gate_up"
            base_dn = f"{P}mlp.experts_down"
            out_tensors[base_gu + ".q"] = gu_q_cpu
            out_tensors[base_gu + ".scale"] = gu_s_cpu
            out_tensors[base_gu + ".minv"] = gu_m_cpu
            out_tensors[base_dn + ".q"] = dn_q_cpu
            out_tensors[base_dn + ".scale"] = dn_s_cpu
            out_tensors[base_dn + ".minv"] = dn_m_cpu

            shard_path = os.path.join(out_dir, f"layer_{L:02d}.safetensors")
            shard_bytes = write_safetensors(shard_path, out_tensors)
            del out_tensors, gu_q_cpu, gu_s_cpu, gu_m_cpu, dn_q_cpu, dn_s_cpu, dn_m_cpu

            # Advance calibration states cleanly back to CPU buffer
            cal_moe = (cal_routed + sh_out_cal).to(torch.bfloat16)
            if L + 1 < n_layers:
                for b0 in range(0, n_seqs, cal_mb):
                    b1 = min(b0 + cal_mb, n_seqs)
                    s0, s1 = b0 * seq_len, b1 * seq_len
                    xb = cal_states_cpu[b0:b1].to(dev)
                    ym = cal_moe[s0:s1].view(b1 - b0, seq_len, 1, 2560).float()
                    wb = win_mlp_cal[s0:s1].view(b1 - b0, seq_len, 4, 1)
                    cal_states_cpu[b0:b1] = (xb.float() + wb * ym).to(torch.bfloat16).cpu()
                    del xb, ym, wb

            hits_sorted = sorted(counts_all.tolist())
            st["hits_min"] = hits_sorted[0]
            st["hits_med"] = hits_sorted[256]
            st["hits_max"] = hits_sorted[-1]
            st["shard_mib"] = round(shard_bytes / (1024.0 ** 2), 1)
            layer_stats.append(st)

            del u_mlp_cal, win_mlp_cal, counts_all, sorted_tok_all, sorted_w_all, offsets_all
            del sh_out_cal, cal_routed, cal_moe, U_mlp_seq, act_precomp_mlp

            peak_gb = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
            gdn_info = f"gdn_mse={st['gdn_hessian_mse_ratio']:.3f}(rms={st['gdn_rms_err_pct']:.2f}%) " if not is_attn else ""
            print(
                f"  [Layer {L:2d}/{n_layers-1}] {'Attn' if is_attn else 'GDN '} t={time.time()-tl:4.1f}s ({time.time()-t0:6.1f}s) | "
                f"{gdn_info}hits[{st['hits_min']}/{st['hits_med']}/{st['hits_max']}] -> {shard_bytes/(1024**3):.2f}GiB VRAM={peak_gb:.2f}G",
                flush=True,
            )

    reader.close()
    peak_gb = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
    elapsed = time.time() - t0
    stats_out = {
        "calib_meta": calib_meta,
        "calib_manifest": calib_manifest,
        "elapsed_s": round(elapsed, 1),
        "peak_vram_gib": round(peak_gb, 2),
        "layers": layer_stats,
    }
    with open(os.path.join(out_dir, "layer_stats.json"), "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"[DONE] Quantized {n_layers} layers in {elapsed:.1f}s | Peak VRAM: {peak_gb:.2f} GiB -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/nvme2/Thinking-Cap-Qwen-3.8-FN")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["actorder", "d02_imp"], default="actorder")
    ap.add_argument("--n-seqs", type=int, default=40)
    ap.add_argument("--seq-len", type=int, default=1536)
    ap.add_argument("--cal-mb", type=int, default=4)
    ap.add_argument("--expert-chunk", type=int, default=64)
    ap.add_argument("--tau", type=float, default=384.0)
    ap.add_argument("--layers", type=int, default=None, help="Limit layers (for quick smoke tests)")
    args = ap.parse_args()

    build_gptq_q4(
        args.src,
        args.out,
        mode=args.mode,
        n_seqs=args.n_seqs,
        seq_len=args.seq_len,
        cal_mb=args.cal_mb,
        expert_chunk=args.expert_chunk,
        tau=args.tau,
        max_layers=args.layers,
    )


if __name__ == "__main__":
    main()
