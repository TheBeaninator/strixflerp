#!/usr/bin/env python3
"""
Round 3 Held-Out RX 7900 XTX Simulator for Qwen 3.8 Flash Next (Thinking-Cap)
at the EXACT 1_Current_Prod binary .faf layout (0.0% speed penalty, zero AWQ,
all q6g64/q8g64 untouched RTN).

Tests 6 pure-GPTQ variants on the same 61,440-token disjoint corpus (40x1536)
and held-out CALIB_DOC (T_eval=1807):
  0_RTN_Prod               : pure RTN control baseline
  1_MoE_Only_Shrink512     : GdnIn=RTN + ExpGU(H_mlp) + ExpDn(shrinkage all 512)
  2_GdnR1_d02_imp_MoE      : GdnIn(R1: damp=0.02, imp_clip=True) + MoE GPTQ
  3_Gdn_d05_rtn_MoE        : GdnIn(damp=0.05, dynamic RTN-grid) + MoE GPTQ
  4_Gdn_d05_imp_MoE        : GdnIn(damp=0.05, imp_clip=True) + MoE GPTQ
  5_All3_ActOrder_StaticG  : ActOrder strictly with static_groups=True (no g_idx) on all 3
"""

import os
os.environ.setdefault("ROCP_HSA_INTERCEPT", "0")
os.environ.setdefault("ROCP_TOOLS_LIB", "")
os.environ.setdefault("HSA_TOOLS_LIB", "")

import argparse
import concurrent.futures
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


# ---------------------------------------------------------------------------
# 1. ActOrder Strictly With static_groups=True (Zero g_idx, 100% .faf Match)
# ---------------------------------------------------------------------------

def prepare_actorder_chol_cpu(H_cpu_f64: torch.Tensor, damp: float = 0.05):
    """Computes descending diagonal permutation + permuted Cholesky inverse U_p."""
    perm = torch.argsort(torch.diag(H_cpu_f64), descending=True)
    invperm = torch.argsort(perm)
    H_p = H_cpu_f64.index_select(0, perm).index_select(1, perm)
    U_p = sc.compute_hinv_chol_cpu(H_p, damp=damp)
    return U_p, perm, invperm


def quant_gptq_actorder_static(
    w_2d_or_3d: torch.Tensor,
    H_cpu_f64: torch.Tensor = None,
    precomp=None,
    fmt: str = "q4g128",
    damp: float = 0.05,
    row_subbatch: int = 40960,
) -> torch.Tensor:
    """
    GPTQ with descending Hessian diagonal column ordering (ActOrder) and STRICTLY
    static_groups=True: per-group BF16 (scale, minv) are computed ONCE on the original
    contiguous [g*G : (g+1)*G] groups before permuting, so output Q encodes natively
    into formats/q4.cpp & linear_q4.hip with zero g_idx permutation tables.
    """
    orig_shape = w_2d_or_3d.shape
    K = orig_shape[-1]
    G = int(fmt[3:])
    if K % G != 0:
        G = math.gcd(K, G)
    bits = int(fmt[1])
    levels = float((1 << bits) - 1)
    dev = w_2d_or_3d.device

    if precomp is None:
        assert H_cpu_f64 is not None
        U_p_cpu, perm_cpu, invperm_cpu = prepare_actorder_chol_cpu(H_cpu_f64, damp=damp)
    else:
        U_p_cpu, perm_cpu, invperm_cpu = precomp

    U_p = U_p_cpu.to(dev)
    perm = perm_cpu.to(dev)
    invperm = invperm_cpu.to(dev)

    W_flat = w_2d_or_3d.reshape(-1, K)
    R = W_flat.shape[0]
    Q_out = torch.empty((R, K), device=dev, dtype=torch.bfloat16)
    n_g = K // G

    for r0 in range(0, R, row_subbatch):
        r1 = min(r0 + row_subbatch, R)
        W_sub = W_flat[r0:r1].float()  # [r, K]
        # Static group (scale, minv) on ORIGINAL contiguous blocks before permuting
        s_grp, m_grp, safe_s_grp = sag._rtn_minmax_params(W_sub.view(r1 - r0, n_g, G), levels)
        s_p = s_grp.repeat_interleave(G, dim=1)[:, perm]
        m_p = m_grp.repeat_interleave(G, dim=1)[:, perm]
        safe_sp = safe_s_grp.repeat_interleave(G, dim=1)[:, perm]

        W_p = W_sub[:, perm].clone()
        Q_p = torch.empty_like(W_p)

        for b0 in range(0, K, G):
            b1 = b0 + G
            W_blk = W_p[:, b0:b1].clone()
            s_blk = s_p[:, b0:b1]
            m_blk = m_p[:, b0:b1]
            safe_s_blk = safe_sp[:, b0:b1]
            Err_blk = torch.zeros_like(W_blk)
            Q_blk = torch.empty_like(W_blk)
            U_blk = U_p[b0:b1, b0:b1]

            for j in range(G):
                wj = W_blk[:, j]
                sj = s_blk[:, j]
                mj = m_blk[:, j]
                code = ((wj - mj) / safe_s_blk[:, j]).round().clamp(0.0, levels)
                code = torch.where(sj > 0.0, code, torch.zeros_like(code))
                qj = code * sj + mj
                Q_blk[:, j] = qj
                err = (wj - qj) / U_blk[j, j]
                Err_blk[:, j] = err
                if j + 1 < G:
                    W_blk[:, j + 1 :] -= err.unsqueeze(1) * U_blk[j, j + 1 :].unsqueeze(0)

            Q_p[:, b0:b1] = Q_blk
            if b1 < K:
                W_p[:, b1:] -= Err_blk @ U_p[b0:b1, b1:]

        Q_out[r0:r1] = Q_p[:, invperm].to(torch.bfloat16)
        del W_sub, s_grp, m_grp, safe_s_grp, s_p, m_p, safe_sp, W_p, Q_p

    return Q_out.reshape(orig_shape)


# ---------------------------------------------------------------------------
# 2. All-512 Normalized Empirical Bayes Shrinkage GPTQ for exp_down (K=640)
# ---------------------------------------------------------------------------

def quant_gptq_batched_exp_down_shrink(
    W_dn: torch.Tensor,         # [cnt, 2560, 640] BF16/FP32
    H_e_chunk: torch.Tensor,    # [cnt, 640, 640] FP32 token-averaged (1/hits_e) X_e^T X_e
    hits_chunk: torch.Tensor,   # [cnt] int64
    H_mean: torch.Tensor,       # [640, 640] FP32 token-averaged layer/chunk mean (1/N_tot) X^T X
    tau: float = 384.0,
    base_damp: float = 0.05,
    act_order: bool = False,
    fmt: str = "q4g128",
) -> torch.Tensor:
    """
    Runs GPTQ across ALL cnt experts (including low-hit & zero-hit experts) using the
    properly normalized Empirical Bayes shrinkage Hessian:
      H_{e,reg} = (hits_e * H_e + tau * H_mean) / (hits_e + tau),
    where BOTH H_e and H_mean are token-averaged covariances (1/n * X^T X).
    Supports both dynamic RTN-grid sequential order (act_order=False) and
    ActOrder + static_groups=True (act_order=True, zero g_idx).
    """
    cnt, N, K = W_dn.shape
    G = int(fmt[3:])
    bits = int(fmt[1])
    levels = float((1 << bits) - 1)
    dev = W_dn.device
    n_g = K // G

    H_e_cpu = H_e_chunk.double().cpu()
    H_m_cpu = H_mean.double().cpu()
    hits_list = hits_chunk.tolist()

    U_list = []
    perm_list = []
    invperm_list = []
    for k_i, h_cnt in enumerate(hits_list):
        h_f = float(h_cnt)
        if h_f > 0.0:
            H_reg = (h_f * H_e_cpu[k_i] + tau * H_m_cpu) / (h_f + tau)
        else:
            H_reg = H_m_cpu.clone()
        if act_order:
            U_p, perm_k, invperm_k = prepare_actorder_chol_cpu(H_reg, damp=base_damp)
            U_list.append(U_p)
            perm_list.append(perm_k)
            invperm_list.append(invperm_k)
        else:
            U_list.append(sc.compute_hinv_chol_cpu(H_reg, damp=base_damp))

    U_b = torch.stack(U_list, dim=0).to(dev)  # [cnt, K, K]
    W_orig = W_dn.float()                     # [cnt, N, K]

    if not act_order:
        W_b = W_orig.clone()
        Q_b = torch.empty_like(W_b)
        for b0 in range(0, K, G):
            b1 = b0 + G
            W_blk = W_b[:, :, b0:b1].clone()
            s_vec, m_vec, safe_s = sag._rtn_minmax_params(W_blk, levels)
            Err_blk = torch.zeros_like(W_blk)
            Q_blk = torch.empty_like(W_blk)
            U_blk = U_b[:, b0:b1, b0:b1]

            for j in range(G):
                wj = W_blk[:, :, j]
                code = ((wj - m_vec) / safe_s).round().clamp(0.0, levels)
                code = torch.where(s_vec > 0.0, code, torch.zeros_like(code))
                qj = code * s_vec + m_vec
                Q_blk[:, :, j] = qj
                err = (wj - qj) / U_blk[:, j, j].unsqueeze(1)
                Err_blk[:, :, j] = err
                if j + 1 < G:
                    W_blk[:, :, j + 1 :] -= err.unsqueeze(2) * U_blk[:, j, j + 1 :].unsqueeze(1)

            Q_b[:, :, b0:b1] = Q_blk
            if b1 < K:
                W_b[:, :, b1:] -= torch.bmm(Err_blk, U_b[:, b0:b1, b1:])
        return Q_b.to(torch.bfloat16)

    # ActOrder + static_groups=True batched across cnt experts
    perm_b = torch.stack(perm_list, dim=0).to(dev).unsqueeze(1).expand(cnt, N, K)
    invperm_b = torch.stack(invperm_list, dim=0).to(dev).unsqueeze(1).expand(cnt, N, K)

    s_grp, m_grp, safe_s_grp = sag._rtn_minmax_params(W_orig.view(cnt, N, n_g, G), levels)
    s_p = torch.gather(s_grp.repeat_interleave(G, dim=2), 2, perm_b)
    m_p = torch.gather(m_grp.repeat_interleave(G, dim=2), 2, perm_b)
    safe_sp = torch.gather(safe_s_grp.repeat_interleave(G, dim=2), 2, perm_b)

    W_p = torch.gather(W_orig, 2, perm_b).clone()
    Q_p = torch.empty_like(W_p)

    for b0 in range(0, K, G):
        b1 = b0 + G
        W_blk = W_p[:, :, b0:b1].clone()
        s_blk = s_p[:, :, b0:b1]
        m_blk = m_p[:, :, b0:b1]
        safe_s_blk = safe_sp[:, :, b0:b1]
        Err_blk = torch.zeros_like(W_blk)
        Q_blk = torch.empty_like(W_blk)
        U_blk = U_b[:, b0:b1, b0:b1]

        for j in range(G):
            wj = W_blk[:, :, j]
            sj = s_blk[:, :, j]
            mj = m_blk[:, :, j]
            code = ((wj - mj) / safe_s_blk[:, :, j]).round().clamp(0.0, levels)
            code = torch.where(sj > 0.0, code, torch.zeros_like(code))
            qj = code * sj + mj
            Q_blk[:, :, j] = qj
            err = (wj - qj) / U_blk[:, j, j].unsqueeze(1)
            Err_blk[:, :, j] = err
            if j + 1 < G:
                W_blk[:, :, j + 1 :] -= err.unsqueeze(2) * U_blk[:, j, j + 1 :].unsqueeze(1)

        Q_p[:, :, b0:b1] = Q_blk
        if b1 < K:
            W_p[:, :, b1:] -= torch.bmm(Err_blk, U_b[:, b0:b1, b1:])

    Q_orig = torch.gather(Q_p, 2, invperm_b)
    return Q_orig.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# 3. Round 3 Self-Tests (Static Groups .faf Invariant + Normalized Shrinkage)
# ---------------------------------------------------------------------------

def run_r3_self_tests(dev: torch.device) -> None:
    torch.manual_seed(123)

    # Test A: ActOrder + static_groups=True exact .faf integer-code compatibility + accuracy
    R_a, K_a, G_a = 32, 256, 128
    X_a = torch.randn(512, K_a, dtype=torch.float64)
    corr_a = torch.randn(K_a, K_a, dtype=torch.float64) * 0.4 + torch.eye(K_a, dtype=torch.float64)
    X_a = X_a @ corr_a
    # Inject strong outlier columns across groups
    X_a[:, 9] *= 8.0
    X_a[:, 201] *= 6.0
    H_a = (X_a.T @ X_a) / 512.0
    W_a = torch.randn(R_a, K_a, device=dev, dtype=torch.float32) * 0.04

    Q_act = quant_gptq_actorder_static(W_a, H_cpu_f64=H_a, fmt="q4g128", damp=0.05, row_subbatch=16)

    # Verify exact .faf static-group encoding: every contiguous 128-col group uses the SAME BF16 (scale, minv)
    # derived BEFORE permutation and yields exact integer codes 0..15 with 0.000 BF16 reconstruction diff!
    W_grp = W_a.view(R_a, K_a // G_a, G_a)
    Q_grp = Q_act.float().view(R_a, K_a // G_a, G_a)
    s_g, m_g, safe_sg = sag._rtn_minmax_params(W_grp, 15.0)
    codes_g = ((Q_grp - m_g.unsqueeze(-1)) / safe_sg.unsqueeze(-1)).round().clamp(0.0, 15.0)
    recon_g = (codes_g * s_g.unsqueeze(-1) + m_g.unsqueeze(-1)).reshape(R_a, K_a).to(torch.bfloat16)
    faf_roundtrip_diff = float((recon_g.float() - Q_act.float()).abs().max().item())

    Xg_a = X_a.float().to(dev)
    Q_rtn_a = sq.fake_quant(W_a, "q4g128")
    mse_rtn_a = float((((W_a - Q_rtn_a.float()) @ Xg_a.T).pow(2).sum()).item())
    mse_act_a = float((((W_a - Q_act.float()) @ Xg_a.T).pow(2).sum()).item())
    assert faf_roundtrip_diff == 0.0, f"Test A failed: .faf static_groups roundtrip diff={faf_roundtrip_diff}"
    assert mse_act_a <= 0.55 * mse_rtn_a, f"Test A failed: mse_act={mse_act_a} vs mse_rtn={mse_rtn_a}"
    print(f"[PASS] Test A (ActOrder static_groups=True): .faf_roundtrip_diff={faf_roundtrip_diff:.1e}, MSE_vs_RTN={(mse_act_a/mse_rtn_a - 1)*100:+.1f}%")

    # Test B: Batched exp_down normalized shrinkage + zero-hit stability + act_order equivalence
    E_b, N_b, K_b = 3, 16, 256
    W_b = torch.randn(E_b, N_b, K_b, device=dev, dtype=torch.float32) * 0.05
    hits_b = torch.tensor([1200, 120, 0], device=dev, dtype=torch.long)
    hs0 = (torch.randn(1200, K_b, dtype=torch.float64) @ corr_a)
    hs1 = (torch.randn(120, K_b, dtype=torch.float64) @ corr_a)
    H_sum_b = torch.zeros(E_b, K_b, K_b, dtype=torch.float64)
    H_sum_b[0] = hs0.T @ hs0
    H_sum_b[1] = hs1.T @ hs1
    H_e_b = torch.zeros(E_b, K_b, K_b, dtype=torch.float64)
    H_e_b[0] = H_sum_b[0] / 1200.0
    H_e_b[1] = H_sum_b[1] / 120.0
    H_mean_b = H_sum_b.sum(dim=0) / 1320.0
    # Verify exact normalization scale invariance: Tr(H_mean) == weighted average of Tr(H_e)
    tr_mean = float(torch.trace(H_mean_b).item())
    tr_weighted = float(((1200.0 * torch.trace(H_e_b[0]) + 120.0 * torch.trace(H_e_b[1])) / 1320.0).item())
    assert abs(tr_mean - tr_weighted) < 1e-6

    H_e_gpu = H_e_b.float().to(dev)
    H_m_gpu = H_mean_b.float().to(dev)
    tau_b = 384.0
    Q_seq_b = quant_gptq_batched_exp_down_shrink(W_b, H_e_gpu, hits_b, H_m_gpu, tau=tau_b, base_damp=0.05, act_order=False, fmt="q4g128")
    Q_act_b = quant_gptq_batched_exp_down_shrink(W_b, H_e_gpu, hits_b, H_m_gpu, tau=tau_b, base_damp=0.05, act_order=True, fmt="q4g128")
    assert torch.isfinite(Q_seq_b).all() and torch.isfinite(Q_act_b).all()

    H_reg0 = (1200.0 * H_e_b[0] + tau_b * H_mean_b) / (1200.0 + tau_b)
    H_reg2 = H_mean_b
    Q_ref_seq0 = sag.quant_gptq_rtn_grid(W_b[0], H_cpu_f64=H_reg0.float().double(), fmt="q4g128", damp=0.05, row_subbatch=16)
    Q_ref_act2 = quant_gptq_actorder_static(W_b[2], H_cpu_f64=H_reg2.float().double(), fmt="q4g128", damp=0.05, row_subbatch=16)
    d_seq = float((Q_seq_b[0].float() - Q_ref_seq0.float()).abs().max().item())
    d_act = float((Q_act_b[2].float() - Q_ref_act2.float()).abs().max().item())
    assert d_seq < 1e-5 and d_act < 1e-5, f"Test B failed: d_seq={d_seq}, d_act={d_act}"
    print(f"[PASS] Test B (Normalized Shrinkage & ActOrder Batched): zero-hit stable, d_seq={d_seq:.1e}, d_act={d_act:.1e}")

    # Test C: sc.quant_gptq(imp_clip=True) and quant_gptq_rtn_grid sanity on correlated X
    Q_imp_c = sc.quant_gptq(W_a, H_a, "q4g128", damp=0.05, imp_clip=True)
    Q_dyn_c = sag.quant_gptq_rtn_grid(W_a, H_cpu_f64=H_a, fmt="q4g128", damp=0.05, row_subbatch=16)
    mse_imp_c = float((((W_a - Q_imp_c.float()) @ Xg_a.T).pow(2).sum()).item())
    mse_dyn_c = float((((W_a - Q_dyn_c.float()) @ Xg_a.T).pow(2).sum()).item())
    assert mse_imp_c < 0.70 * mse_rtn_a and mse_dyn_c < 0.70 * mse_rtn_a
    print(f"[PASS] Test C (gdn_in variants): imp_d05={(mse_imp_c/mse_rtn_a - 1)*100:+.1f}%, dyn_rtn_d05={(mse_dyn_c/mse_rtn_a - 1)*100:+.1f}% vs RTN")


# ---------------------------------------------------------------------------
# 4. Full 48-Layer Round 3 Held-Out Simulator
# ---------------------------------------------------------------------------

VARIANTS = [
    "0_RTN_Prod",
    "1_MoE_Only_Shrink512",
    "2_GdnR1_d02_imp_MoE",
    "3_Gdn_d05_rtn_MoE",
    "4_Gdn_d05_imp_MoE",
    "5_All3_ActOrder_StaticG",
]


def run_r3_sim(
    model_dir: str,
    n_seqs: int = 40,
    seq_len: int = 1536,
    cal_mb: int = 4,
    expert_chunk: int = 64,
    tau: float = 384.0,
    moe_damp: float = 0.05,
):
    dev = torch.device("cuda:0")
    _ = torch.zeros(1, device=dev)
    torch.cuda.synchronize()
    cpu = torch.device("cpu")
    torch.cuda.reset_peak_memory_stats(dev)

    run_r3_self_tests(dev)

    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    tok = sq.QwenTokenizer(os.path.join(model_dir, "tokenizer.json"))
    ids_eval = tok.encode(sq.CALIB_DOC)
    T_eval = len(ids_eval)

    ids_cal_2d, dom_summary, cal_sha = sag.build_diverse_calib_sequences(tok, workspace_root, n_seqs=n_seqs, seq_len=seq_len)
    T_cal = n_seqs * seq_len
    print(f"Held-out Eval: T_eval={T_eval} | Disjoint Calib: {n_seqs}x{seq_len} = {T_cal} tokens (sha={cal_sha}) | domains={dom_summary}")

    reader = sq.SafetensorsReader(model_dir)
    lm = "model.language_model."
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg_root = json.load(f)
    tc = cfg_root.get("text_config", cfg_root)
    n_layers = tc["num_hidden_layers"]
    layer_types = tc.get("layer_types")

    target_eval = torch.tensor(ids_eval[1:], device=dev, dtype=torch.long)
    cos_e, sin_e = sq.build_qwen_rope(T_eval, dev, torch.bfloat16)
    cos_c, sin_c = sq.build_qwen_rope(seq_len, dev, torch.bfloat16)

    state_names = ["BF16"] + VARIANTS
    prefetch_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    t0 = time.time()
    layer_telemetry = []

    with torch.no_grad():
        emb_cpu = reader.read_tensor(lm + "embed_tokens.weight", cpu, torch.bfloat16)
        x_eval0 = F.embedding(torch.tensor(ids_eval, dtype=torch.long), emb_cpu).to(dev)
        cal_states_cpu = F.embedding(ids_cal_2d, emb_cpu).unsqueeze(2).expand(n_seqs, seq_len, 4, 2560).contiguous()
        del emb_cpu
        torch.cuda.empty_cache()

        states = {n: x_eval0.unsqueeze(1).expand(T_eval, 4, 2560).contiguous().clone() for n in state_names}
        del x_eval0

        for L in range(n_layers):
            tl = time.time()
            P = f"{lm}layers.{L}."
            is_attn = (layer_types[L] == "full_attention") if layer_types is not None else ((L + 1) % 4 == 0)
            lt = {"layer": L, "type": "attn" if is_attn else "gdn"}

            # ---- Optional PLE Injection (Layer 1, pure RTN q6g64) ----
            if (P + "ple.key_proj.weight") in reader.wm:
                ple_p = P + "ple."
                mult = reader.read_tensor(ple_p + "ple_embedding.layer_multipliers", cpu).tolist()
                vsz = reader.read_tensor(ple_p + "ple_embedding.ngram_heads_vocab_sizes", cpu).tolist()
                offs = reader.read_tensor(ple_p + "ple_embedding.ngram_heads_offsets", cpu).tolist()

                ur_e, mi_e = sq.ple_ngram_ids_py(mult, vsz, offs, ids_eval, eos=248044)
                ple_e = reader.gather_ple_rows(ple_p, ur_e, dev)[torch.tensor(mi_e, device=dev, dtype=torch.long)].reshape(T_eval, 2560)

                w_kp = reader.read_tensor(ple_p + "key_proj.weight", dev, torch.float32)
                w_vp = reader.read_tensor(ple_p + "value_proj.weight", dev, torch.float32)
                w_ple = torch.cat([w_kp, w_vp], dim=0)
                del w_kp, w_vp
                ple_cw = reader.read_tensor(ple_p + "conv1d.weight", dev, torch.float32)
                ple_nk = reader.read_tensor(ple_p + "norm_key.weight", dev, torch.float32)
                ple_nq = reader.read_tensor(ple_p + "norm_query.weight", dev, torch.float32)
                ple_nc = reader.read_tensor(ple_p + "norm_conv.weight", dev, torch.float32)
                w_ple_bf16 = w_ple.to(torch.bfloat16)
                w_ple_rtn = sq.fake_quant(w_ple, "q6g64")
                del w_ple

                for b_idx in range(n_seqs):
                    seq_b = ids_cal_2d[b_idx].tolist()
                    ur_c, mi_c = sq.ple_ngram_ids_py(mult, vsz, offs, seq_b, eos=248044)
                    ple_c = reader.gather_ple_rows(ple_p, ur_c, dev)[torch.tensor(mi_c, device=dev, dtype=torch.long)].reshape(seq_len, 2560)
                    xc_b = cal_states_cpu[b_idx].to(dev)
                    cal_states_cpu[b_idx] = sq.ple_forward(xc_b, ple_c, w_ple_bf16, ple_cw, ple_nk, ple_nq, ple_nc).cpu()
                    del ple_c, xc_b

                for name in state_names:
                    wp = w_ple_bf16 if name == "BF16" else w_ple_rtn
                    states[name] = sq.ple_forward(states[name], ple_e, wp, ple_cw, ple_nk, ple_nq, ple_nc)
                del ple_e, w_ple_bf16, w_ple_rtn, ple_cw, ple_nk, ple_nq, ple_nc

            # ---- Attention / GDN Hyper-Connection Mix ----
            ahc = P + "attn_hyper_connection."
            a_wd = reader.read_tensor(ahc + "input_mix_weight_down.weight", dev, torch.float32)
            a_wi = reader.read_tensor(ahc + "block_inject_weight.weight", dev, torch.float32)
            a_wu = reader.read_tensor(ahc + "input_mix_weight_up.weight", dev, torch.float32)
            a_hn = reader.read_tensor(ahc + "hc_norm.weight", dev, torch.float32)
            a_wdf = a_wd * (1.0 + a_hn.unsqueeze(0))
            a_wif = a_wi * (1.0 + a_hn.unsqueeze(0))
            del a_wd, a_wi

            ahcd_bf16 = (a_wdf.to(torch.bfloat16), a_wif.to(torch.bfloat16))
            ahcu_bf16 = a_wu.to(torch.bfloat16)
            ahcd_q8 = (sq.fake_quant(a_wdf, "q8g64"), sq.fake_quant(a_wif, "q8g64"))
            ahcu_q8 = sq.fake_quant(a_wu, "q8g64")

            if not is_attn:
                gp = P + "linear_attn."
                w_qkv = reader.read_tensor(gp + "in_proj_qkv.weight", dev, torch.float32)
                w_z = reader.read_tensor(gp + "in_proj_z.weight", dev, torch.float32)
                w_b = reader.read_tensor(gp + "in_proj_b.weight", dev, torch.float32)
                w_a = reader.read_tensor(gp + "in_proj_a.weight", dev, torch.float32)
                w_gdn_in = torch.cat([w_qkv, w_z, w_b, w_a], dim=0)
                del w_qkv, w_z, w_b, w_a
                w_gdn_out = reader.read_tensor(gp + "out_proj.weight", dev, torch.float32)
                cw = reader.read_tensor(gp + "conv1d.weight", dev, torch.float32)
                A_log = reader.read_tensor(gp + "A_log", dev, torch.float32)
                dt_bias = reader.read_tensor(gp + "dt_bias", dev, torch.float32)
                gnorm = reader.read_tensor(gp + "norm.weight", dev, torch.float32)
                gin_bf16 = w_gdn_in.to(torch.bfloat16)
                gout_bf16 = w_gdn_out.to(torch.bfloat16)
                gout_rtn = sq.fake_quant(w_gdn_out, "q6g64")
                del w_gdn_out

                H_gdn_cpu = torch.zeros((2560, 2560), dtype=torch.float64)
                for b0 in range(0, n_seqs, cal_mb):
                    b1 = min(b0 + cal_mb, n_seqs)
                    xb = cal_states_cpu[b0:b1].to(dev)
                    ub, win_b = sag.hc_mix_down_up_batched(xb, *ahcd_bf16, ahcu_bf16, a_hn, inject=True)
                    u_flat = ub.reshape(-1, 2560).double()
                    H_gdn_cpu.add_((u_flat.T @ u_flat).cpu())
                    ym = sag.gdn_forward_batched(ub, gin_bf16, gout_bf16, cw, A_log, dt_bias, gnorm)
                    cal_states_cpu[b0:b1] = (xb.float() + win_b.unsqueeze(-1) * ym.float().unsqueeze(2)).to(torch.bfloat16).cpu()
                    del xb, ub, win_b, u_flat, ym

                H_gdn_cpu.div_(float(T_cal))

                gin_rtn = sq.fake_quant(w_gdn_in, "q4g128")
                gin_d02_imp = sc.quant_gptq(w_gdn_in, H_gdn_cpu, "q4g128", damp=0.02, imp_clip=True)
                gin_d05_rtn = sag.quant_gptq_rtn_grid(w_gdn_in, H_cpu_f64=H_gdn_cpu, fmt="q4g128", damp=0.05, row_subbatch=40960)
                gin_d05_imp = sc.quant_gptq(w_gdn_in, H_gdn_cpu, "q4g128", damp=0.05, imp_clip=True)
                gin_d05_act = quant_gptq_actorder_static(w_gdn_in, H_cpu_f64=H_gdn_cpu, fmt="q4g128", damp=0.05, row_subbatch=40960)

                # Diagnostic calibration Hessian MSE ratios vs RTN
                H_gdn_gpu = H_gdn_cpu.float().to(dev)
                def _gdn_mse(q):
                    e = q.float() - w_gdn_in
                    return float(((e @ H_gdn_gpu) * e).sum().item())
                mse_base = max(_gdn_mse(gin_rtn), 1e-12)
                lt["gdn_mse_ratio_d02_imp"] = round(_gdn_mse(gin_d02_imp) / mse_base, 4)
                lt["gdn_mse_ratio_d05_rtn"] = round(_gdn_mse(gin_d05_rtn) / mse_base, 4)
                lt["gdn_mse_ratio_d05_imp"] = round(_gdn_mse(gin_d05_imp) / mse_base, 4)
                lt["gdn_mse_ratio_d05_act"] = round(_gdn_mse(gin_d05_act) / mse_base, 4)
                del H_gdn_gpu

                gin_variants = {
                    "BF16": gin_bf16,
                    "0_RTN_Prod": gin_rtn,
                    "1_MoE_Only_Shrink512": gin_rtn,
                    "2_GdnR1_d02_imp_MoE": gin_d02_imp,
                    "3_Gdn_d05_rtn_MoE": gin_d05_rtn,
                    "4_Gdn_d05_imp_MoE": gin_d05_imp,
                    "5_All3_ActOrder_StaticG": gin_d05_act,
                }

                for name in state_names:
                    hd = ahcd_bf16 if name == "BF16" else ahcd_q8
                    hu = ahcu_bf16 if name == "BF16" else ahcu_q8
                    u_e, win_e = sq.hc_mix_down_up(states[name], *hd, hu, a_hn, inject=True)
                    go = gout_bf16 if name == "BF16" else gout_rtn
                    ym = sq.gdn_forward(u_e, gin_variants[name], go, cw, A_log, dt_bias, gnorm)
                    states[name] = (states[name].float() + win_e.unsqueeze(-1) * ym.float().unsqueeze(1)).to(torch.bfloat16)
                    del u_e, win_e, ym

                del w_gdn_in, gin_bf16, gin_rtn, gin_d02_imp, gin_d05_rtn, gin_d05_imp, gin_d05_act
                del gout_bf16, gout_rtn, cw, A_log, dt_bias, gnorm, H_gdn_cpu, gin_variants
            else:
                ap = P + "self_attn."
                wq = reader.read_tensor(ap + "q_proj.weight", dev, torch.float32)
                wk = reader.read_tensor(ap + "k_proj.weight", dev, torch.float32)
                wv = reader.read_tensor(ap + "v_proj.weight", dev, torch.float32)
                w_qkv = torch.cat([wq, wk, wv], dim=0)
                del wq, wk, wv
                w_ao = reader.read_tensor(ap + "o_proj.weight", dev, torch.float32)
                qn = reader.read_tensor(ap + "q_norm.weight", dev, torch.float32)
                kn = reader.read_tensor(ap + "k_norm.weight", dev, torch.float32)

                qkv_bf16, ao_bf16 = w_qkv.to(torch.bfloat16), w_ao.to(torch.bfloat16)
                qkv_rtn, ao_rtn = sq.fake_quant(w_qkv, "q6g64"), sq.fake_quant(w_ao, "q6g64")
                del w_qkv, w_ao

                for b0 in range(0, n_seqs, cal_mb):
                    b1 = min(b0 + cal_mb, n_seqs)
                    xb = cal_states_cpu[b0:b1].to(dev)
                    ub, win_b = sag.hc_mix_down_up_batched(xb, *ahcd_bf16, ahcu_bf16, a_hn, inject=True)
                    ym = sag.full_attn_forward_batched(ub, qkv_bf16, ao_bf16, qn, kn, cos_c, sin_c)
                    cal_states_cpu[b0:b1] = (xb.float() + win_b.unsqueeze(-1) * ym.float().unsqueeze(2)).to(torch.bfloat16).cpu()
                    del xb, ub, win_b, ym

                for name in state_names:
                    hd = ahcd_bf16 if name == "BF16" else ahcd_q8
                    hu = ahcu_bf16 if name == "BF16" else ahcu_q8
                    u_e, win_e = sq.hc_mix_down_up(states[name], *hd, hu, a_hn, inject=True)
                    ym = sq.full_attn_forward(
                        u_e,
                        qkv_bf16 if name == "BF16" else qkv_rtn,
                        ao_bf16 if name == "BF16" else ao_rtn,
                        qn, kn, cos_e, sin_e,
                    )
                    states[name] = (states[name].float() + win_e.unsqueeze(-1) * ym.float().unsqueeze(1)).to(torch.bfloat16)
                    del u_e, win_e, ym
                del qkv_bf16, ao_bf16, qkv_rtn, ao_rtn, qn, kn

            del a_wdf, a_wif, a_wu, a_hn, ahcd_bf16, ahcu_bf16, ahcd_q8, ahcu_q8

            # ---- MLP Hyper-Connection Mix ----
            mhc = P + "mlp_hyper_connection."
            m_wd = reader.read_tensor(mhc + "input_mix_weight_down.weight", dev, torch.float32)
            m_wi = reader.read_tensor(mhc + "block_inject_weight.weight", dev, torch.float32)
            m_wu = reader.read_tensor(mhc + "input_mix_weight_up.weight", dev, torch.float32)
            m_hn = reader.read_tensor(mhc + "hc_norm.weight", dev, torch.float32)
            m_wdf = m_wd * (1.0 + m_hn.unsqueeze(0))
            m_wif = m_wi * (1.0 + m_hn.unsqueeze(0))
            del m_wd, m_wi

            mhcd_bf16 = (m_wdf.to(torch.bfloat16), m_wif.to(torch.bfloat16))
            mhcu_bf16 = m_wu.to(torch.bfloat16)
            mhcd_q8 = (sq.fake_quant(m_wdf, "q8g64"), sq.fake_quant(m_wif, "q8g64"))
            mhcu_q8 = sq.fake_quant(m_wu, "q8g64")

            u_mlp_cal = torch.empty((T_cal, 2560), device=dev, dtype=torch.bfloat16)
            win_mlp_cal = torch.empty((T_cal, 4), device=dev, dtype=torch.float32)
            H_mlp_cpu = torch.zeros((2560, 2560), dtype=torch.float64)
            for b0 in range(0, n_seqs, cal_mb):
                b1 = min(b0 + cal_mb, n_seqs)
                xb = cal_states_cpu[b0:b1].to(dev)
                ub, win_b = sag.hc_mix_down_up_batched(xb, *mhcd_bf16, mhcu_bf16, m_hn, inject=True)
                u_flat = ub.reshape(-1, 2560)
                s0, s1 = b0 * seq_len, b1 * seq_len
                u_mlp_cal[s0:s1] = u_flat
                win_mlp_cal[s0:s1] = win_b.reshape(-1, 4)
                uf64 = u_flat.double()
                H_mlp_cpu.add_((uf64.T @ uf64).cpu())
                del xb, ub, win_b, u_flat, uf64

            H_mlp_cpu.div_(float(T_cal))
            U_mlp_seq = sc.compute_hinv_chol_cpu(H_mlp_cpu, damp=moe_damp).to(dev)
            act_precomp_mlp = prepare_actorder_chol_cpu(H_mlp_cpu, damp=moe_damp)
            del H_mlp_cpu

            # ---- Router & Shared Expert (pure RTN q6g64) ----
            mp = P + "mlp."
            wg_r = reader.read_tensor(mp + "gate.weight", dev, torch.float32)
            wsg_r = reader.read_tensor(mp + "shared_expert_gate.weight", dev, torch.float32)
            w_router_all = torch.cat([wg_r, wsg_r], dim=0)
            del wg_r, wsg_r

            cal_route = sq.moe_route_qwen(u_mlp_cal, w_router_all.to(torch.bfloat16))
            tidx_flat = cal_route[0].reshape(-1)
            tw_flat = cal_route[1].reshape(-1)
            ord_all = torch.argsort(tidx_flat, stable=True)
            sorted_tok_all = ord_all // 10
            sorted_w_all = tw_flat[ord_all].unsqueeze(-1)
            counts_all = torch.bincount(tidx_flat, minlength=512)
            offsets_all = torch.zeros(513, device=dev, dtype=torch.long)
            torch.cumsum(counts_all, dim=0, out=offsets_all[1:])
            del tidx_flat, tw_flat, ord_all

            w_sh_g = reader.read_tensor(mp + "shared_expert.gate_proj.weight", dev, torch.float32)
            w_sh_u = reader.read_tensor(mp + "shared_expert.up_proj.weight", dev, torch.float32)
            w_sh_gu = torch.cat([w_sh_g, w_sh_u], dim=0)
            del w_sh_g, w_sh_u
            w_sh_d = reader.read_tensor(mp + "shared_expert.down_proj.weight", dev, torch.float32)

            sh_gu_c = F.linear(u_mlp_cal, w_sh_gu.to(torch.bfloat16))
            sh_h_c = F.silu(sh_gu_c[:, :640]) * sh_gu_c[:, 640:]
            sh_out_cal = F.linear(sh_h_c, w_sh_d.to(torch.bfloat16)).float() * cal_route[2].unsqueeze(-1)
            del sh_gu_c, sh_h_c

            gu_name = mp + "experts.gate_up_proj"
            dn_name = mp + "experts.down_proj"
            reader.info(gu_name)
            reader.info(dn_name)

            w_router_bf16 = w_router_all.to(torch.bfloat16)
            shgu_bf16, shdn_bf16 = w_sh_gu.to(torch.bfloat16), w_sh_d.to(torch.bfloat16)
            shgu_rtn, shdn_rtn = sq.fake_quant(w_sh_gu, "q6g64"), sq.fake_quant(w_sh_d, "q6g64")
            del w_router_all, w_sh_gu, w_sh_d

            u_mlp, win_mlp, routes, sh_outs = {}, {}, {}, {}
            for name in state_names:
                hd = mhcd_bf16 if name == "BF16" else mhcd_q8
                hu = mhcu_bf16 if name == "BF16" else mhcu_q8
                u_m, wm = sq.hc_mix_down_up(states[name], *hd, hu, m_hn, inject=True)
                u_mlp[name] = u_m
                win_mlp[name] = wm
                routes[name] = sq.moe_route_qwen(u_m, w_router_bf16)
                wgu_s = shgu_bf16 if name == "BF16" else shgu_rtn
                wdn_s = shdn_bf16 if name == "BF16" else shdn_rtn
                g_s = F.linear(u_m, wgu_s)
                h_s = F.silu(g_s[:, :640]) * g_s[:, 640:]
                sh_outs[name] = F.linear(h_s, wdn_s).float() * routes[name][2].unsqueeze(-1)
                del g_s, h_s

            del m_wdf, m_wif, m_wu, m_hn, mhcd_bf16, mhcu_bf16, mhcd_q8, mhcu_q8
            del w_router_bf16, shgu_bf16, shdn_bf16, shgu_rtn, shdn_rtn

            # ---- Stream Routed Experts in 64-Expert Chunks (Zero AWQ, Shrinkage All 512) ----
            cal_routed = torch.zeros((T_cal, 2560), device=dev, dtype=torch.float32)
            routed_acc = {name: torch.zeros((T_eval, 2560), device=dev, dtype=torch.float32) for name in state_names}
            slot_buf = torch.zeros((T_eval, 10, 2560), device=dev, dtype=torch.float32)

            def _load_chunk_cpu(c_start, c_cnt):
                return (
                    reader.read_slice_dim0(gu_name, c_start, c_cnt, cpu, torch.bfloat16),
                    reader.read_slice_dim0(dn_name, c_start, c_cnt, cpu, torch.bfloat16),
                )

            CHUNK = expert_chunk
            next_fut = prefetch_pool.submit(_load_chunk_cpu, 0, min(CHUNK, 512))

            def moe_key(name):
                if name == "BF16": return "bf16"
                if name == "0_RTN_Prod": return "rtn"
                if name == "5_All3_ActOrder_StaticG": return "gptq_act"
                return "gptq_seq"

            for c0 in range(0, 512, CHUNK):
                cnt = min(CHUNK, 512 - c0)
                c1 = c0 + cnt
                gu_cpu, dn_cpu = next_fut.result()
                if c1 < 512:
                    next_fut = prefetch_pool.submit(_load_chunk_cpu, c1, min(CHUNK, 512 - c1))

                gu_bf16 = gu_cpu.to(dev)
                dn_bf16 = dn_cpu.to(dev)
                del gu_cpu, dn_cpu

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

                q_dn = {
                    "bf16": dn_bf16,
                    "rtn": sq.fake_quant(dn_bf16, "q4g128"),
                    "gptq_seq": quant_gptq_batched_exp_down_shrink(
                        dn_bf16, H_e_chunk, hits_chunk, H_mean_chunk, tau=tau, base_damp=moe_damp, act_order=False, fmt="q4g128"
                    ),
                    "gptq_act": quant_gptq_batched_exp_down_shrink(
                        dn_bf16, H_e_chunk, hits_chunk, H_mean_chunk, tau=tau, base_damp=moe_damp, act_order=True, fmt="q4g128"
                    ),
                }
                del dn_bf16, H_e_chunk, H_mean_chunk

                q_gu = {
                    "bf16": gu_bf16,
                    "rtn": sq.fake_quant(gu_bf16, "q4g128"),
                    "gptq_seq": sag.quant_gptq_rtn_grid(gu_bf16, U_gpu=U_mlp_seq, fmt="q4g128", row_subbatch=40960),
                    "gptq_act": quant_gptq_actorder_static(gu_bf16, precomp=act_precomp_mlp, fmt="q4g128", row_subbatch=40960),
                }
                del gu_bf16

                for name in state_names:
                    tidx, tw, _ = routes[name]
                    in_chunk = (tidx >= c0) & (tidx < c1)
                    if not in_chunk.any():
                        continue
                    tok_pos, slot_pos = torch.nonzero(in_chunk, as_tuple=True)
                    loc_e = tidx[tok_pos, slot_pos] - c0
                    order = torch.argsort(loc_e, stable=True)
                    tok_sorted = tok_pos[order]
                    slot_sorted = slot_pos[order]
                    w_sorted = tw[tok_pos, slot_pos][order].unsqueeze(-1)
                    uniq_e, counts_e = torch.unique_consecutive(loc_e[order], return_counts=True)

                    mk = moe_key(name)
                    wgu_c = q_gu[mk]
                    wdn_c = q_dn[mk]
                    x_hits = u_mlp[name][tok_sorted]
                    outs = torch.empty_like(x_hits)
                    pos = 0
                    for le_i, cnt_i in zip(uniq_e.tolist(), counts_e.tolist()):
                        xs = x_hits[pos : pos + cnt_i]
                        gue = F.linear(xs, wgu_c[le_i])
                        hs = F.silu(gue[:, :640]) * gue[:, 640:]
                        outs[pos : pos + cnt_i] = F.linear(hs, wdn_c[le_i])
                        pos += cnt_i

                    slot_buf.zero_()
                    slot_buf[tok_sorted, slot_sorted] = outs.float() * w_sorted
                    routed_acc[name].add_(slot_buf.sum(dim=1))

                del q_gu, q_dn

            del slot_buf, sorted_tok_all, sorted_w_all, offsets_all

            # Advance calibration states cleanly back into CPU buffer
            cal_moe = (cal_routed + sh_out_cal).to(torch.bfloat16)
            for b0 in range(0, n_seqs, cal_mb):
                b1 = min(b0 + cal_mb, n_seqs)
                s0, s1 = b0 * seq_len, b1 * seq_len
                xb = cal_states_cpu[b0:b1].to(dev)
                ym = cal_moe[s0:s1].view(b1 - b0, seq_len, 1, 2560).float()
                wb = win_mlp_cal[s0:s1].view(b1 - b0, seq_len, 4, 1)
                cal_states_cpu[b0:b1] = (xb.float() + wb * ym).to(torch.bfloat16).cpu()
                del xb, ym, wb

            for name in state_names:
                y_moe = (routed_acc[name] + sh_outs[name]).to(torch.bfloat16)
                states[name] = (states[name].float() + win_mlp[name].unsqueeze(-1) * y_moe.float().unsqueeze(1)).to(torch.bfloat16)

            hits_sorted = sorted(counts_all.tolist())
            lt["hits_min"] = hits_sorted[0]
            lt["hits_med"] = hits_sorted[256]
            lt["hits_max"] = hits_sorted[-1]
            layer_telemetry.append(lt)

            del u_mlp_cal, win_mlp_cal, cal_route, counts_all, sh_out_cal, cal_routed, cal_moe, U_mlp_seq, act_precomp_mlp
            del u_mlp, win_mlp, routes, routed_acc, sh_outs

            if L in (0, 1, 2, 3, 15, 31, 47):
                peak_gb = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
                gdn_s = (
                    f"gdn_mse[d02i={lt['gdn_mse_ratio_d02_imp']:.3f}, d05r={lt['gdn_mse_ratio_d05_rtn']:.3f}, "
                    f"d05i={lt['gdn_mse_ratio_d05_imp']:.3f}, d05a={lt['gdn_mse_ratio_d05_act']:.3f}] "
                    if not is_attn else ""
                )
                print(
                    f"  [Layer {L:2d}/47] {'Attn' if is_attn else 'GDN '} t={time.time()-tl:4.1f}s ({time.time()-t0:6.1f}s) | "
                    f"{gdn_s}hits[{lt['hits_min']}/{lt['hits_med']}/{lt['hits_max']}] VRAM={peak_gb:.2f}G",
                    flush=True,
                )

        # ---- Final HC Mixer (q8g64) + LM Head (q8g64) Evaluation on Held-Out CALIB_DOC ----
        del cal_states_cpu
        fhc = lm + "hyper_connection_mixer."
        f_wd = reader.read_tensor(fhc + "input_mix_weight_down.weight", dev, torch.float32)
        f_wu = reader.read_tensor(fhc + "input_mix_weight_up.weight", dev, torch.float32)
        f_hn = reader.read_tensor(fhc + "hc_norm.weight", dev, torch.float32)
        f_wdf = f_wd * (1.0 + f_hn.unsqueeze(0))
        del f_wd
        dummy_inj = torch.zeros((4, 10240), device=dev, dtype=torch.float32)

        lm_w = reader.read_tensor("lm_head.weight", dev, torch.bfloat16)
        f_bf16_u, _ = sq.hc_mix_down_up(states["BF16"], f_wdf.to(torch.bfloat16), dummy_inj, f_wu.to(torch.bfloat16), f_hn, inject=False)

        n_eval = T_eval - 1
        TOK_CHUNK = 256
        bf16_nll_sum = 0.0
        for s0 in range(0, n_eval, TOK_CHUNK):
            s1 = min(s0 + TOK_CHUNK, n_eval)
            lgt = F.linear(f_bf16_u[s0:s1], lm_w).float()
            bf16_nll_sum += F.nll_loss(F.log_softmax(lgt, dim=-1), target_eval[s0:s1], reduction="sum").item()
            del lgt
        bf16_ppl = math.exp(bf16_nll_sum / float(n_eval))

        qlm_q8 = sq.fake_quant_rows(lm_w, "q8g64")
        fhcd_q8 = sq.fake_quant(f_wdf, "q8g64")
        fhcu_q8 = sq.fake_quant(f_wu, "q8g64")

        def eval_u(u_cand):
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
                nll_sum += F.nll_loss(q_lp, target_eval[s0:s1], reduction="sum").item()
                ref_t1 = ref_lgt.argmax(dim=-1)
                t1_hits += (q_lgt.argmax(dim=-1) == ref_t1).sum().item()
                t5_hits += (q_lgt.topk(5, dim=-1).indices == ref_t1.unsqueeze(-1)).any(dim=-1).sum().item()
                del ref_lgt, ref_lp, ref_p, q_lgt, q_lp
            kld_v = torch.cat(kld_parts, dim=0)
            return (
                kld_v.mean().item(),
                torch.quantile(kld_v, 0.50).item(),
                torch.quantile(kld_v, 0.95).item(),
                kld_v.max().item(),
                math.exp(nll_sum / float(n_eval)),
                100.0 * t1_hits / float(n_eval),
                100.0 * t5_hits / float(n_eval),
            )

        print(f"\n=== ROUND 3 HELD-OUT EVALUATION (Thinking-Cap, T_eval={T_eval}, T_cal={T_cal}, tau={tau}, BF16 PPL={bf16_ppl:.3f}) ===")
        print(f"{'Variant':26s} | {'Mean KLD':9s} | {'vs RTN':8s} | {'Med KLD':9s} | {'p95 KLD':9s} | {'Max KLD':9s} | {'PPL':7s} | {'dPPL':7s} | {'Top1%':6s} | {'Top5%':6s}")
        print("-" * 114)
        print(f"{'BF16 Baseline':26s} | {'0.000000':9s} | {'---':8s} | {'0.000000':9s} | {'0.000000':9s} | {'0.000000':9s} | {bf16_ppl:7.3f} | {'+0.000':7s} | {'100.0%':6s} | {'100.0%':6s}")

        out_rows = []
        rtn_kld = None
        for name in VARIANTS:
            u_fin, _ = sq.hc_mix_down_up(states[name], fhcd_q8, dummy_inj, fhcu_q8, f_hn, inject=False)
            mk, medk, p95k, maxk, ppl, t1, t5 = eval_u(u_fin)
            del u_fin
            if name == "0_RTN_Prod":
                rtn_kld = mk
            rel_pct = (mk / rtn_kld - 1.0) * 100.0
            rel_s = f"{rel_pct:+6.1f}%"
            dp = ppl - bf16_ppl
            print(f"{name:26s} | {mk:9.6f} | {rel_s:8s} | {medk:9.6f} | {p95k:9.6f} | {maxk:9.6f} | {ppl:7.3f} | {dp:+7.3f} | {t1:5.1f}% | {t5:5.1f}%")
            out_rows.append({
                "variant": name,
                "mean_kld": mk,
                "delta_vs_rtn_pct": rel_pct,
                "median_kld": medk,
                "p95_kld": p95k,
                "max_kld": maxk,
                "ppl": ppl,
                "dppl": dp,
                "top1_pct": t1,
                "top5_pct": t5,
            })

        peak_gb = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
        elapsed = time.time() - t0
        gdns = [x for x in layer_telemetry if x["type"] == "gdn"]
        print(
            f"\nGDN Mean Calib Hessian MSE ratios vs RTN: "
            f"d02_imp={statistics.mean(x['gdn_mse_ratio_d02_imp'] for x in gdns):.4f}, "
            f"d05_rtn={statistics.mean(x['gdn_mse_ratio_d05_rtn'] for x in gdns):.4f}, "
            f"d05_imp={statistics.mean(x['gdn_mse_ratio_d05_imp'] for x in gdns):.4f}, "
            f"d05_act={statistics.mean(x['gdn_mse_ratio_d05_act'] for x in gdns):.4f}"
        )
        print(f"Elapsed: {elapsed:.1f}s | Peak VRAM: {peak_gb:.2f} GiB")

    prefetch_pool.shutdown()
    reader.close()
    return {
        "model": model_dir,
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "t_cal": T_cal,
        "calib_sha256_16": cal_sha,
        "domains": dom_summary,
        "t_eval": T_eval,
        "tau": tau,
        "moe_damp": moe_damp,
        "bf16_ppl": bf16_ppl,
        "peak_vram_gib": peak_gb,
        "elapsed_s": elapsed,
        "variants": out_rows,
        "layer_telemetry": layer_telemetry,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/nvme2/Thinking-Cap-Qwen-3.8-FN")
    ap.add_argument("--n-seqs", type=int, default=40)
    ap.add_argument("--seq-len", type=int, default=1536)
    ap.add_argument("--cal-mb", type=int, default=4)
    ap.add_argument("--expert-chunk", type=int, default=64)
    ap.add_argument("--tau", type=float, default=384.0)
    ap.add_argument("--moe-damp", type=float, default=0.05)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    res = run_r3_sim(
        args.model,
        n_seqs=args.n_seqs,
        seq_len=args.seq_len,
        cal_mb=args.cal_mb,
        expert_chunk=args.expert_chunk,
        tau=args.tau,
        moe_damp=args.moe_damp,
    )
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
