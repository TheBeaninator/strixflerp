#!/usr/bin/env python3
"""
Strict FP64 Trunk (R3 + Q6_Hotspots) + mtp.layers.0 GPTQ Calibration & Held-Out Simulator
for Thinking-Cap-Qwen-3.8-FN (zero-speed-penalty .faf layout).

Evaluates:
  Trunk (48 layers, 61,440 disjoint tokens, all Hessians accumulated in FP64 .double()):
    0_RTN_Prod            : pure RTN control (0.031249 KLD)
    1_R3_FP64_Exact       : exact Round 3 reproduction (0.021643 KLD)
    2_R3_FP64_Q6_Hotspots : R3 + FP64 exp_down + ActOrder static_groups (no clip) on q6g64
                            output hotspots (gdn_out, attn_o, sh_down)

  MTP Head (mtp.layers.0, driven by L47 output of 2_R3_FP64_Q6_Hotspots, FP64 Hessians):
    M_BF16                : BF16 MTP ceiling on top of quantized trunk
    M0_RTN_MTP            : current prod RTN MTP (fc q4g64, qkv/o/sh q6g64, experts q4g128)
    M1_GPTQ_Q4_MTP        : ActOrder GPTQ on 4-bit MTP weights (fc_emb/fc_hid q4g64, experts q4g128 Shrink512)
    M2_GPTQ_Q4_Q6_MTP     : M1 + ActOrder GPTQ on MTP q6g64 hotspots (self_attn.o_proj, shared_expert.down_proj)
"""

import os
os.environ.setdefault("ROCP_HSA_INTERCEPT", "0")
os.environ.setdefault("ROCP_TOOLS_LIB", "")
os.environ.setdefault("HSA_TOOLS_LIB", "")

import argparse
import concurrent.futures
import json
import math
import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_qwen_quant as sq
import sim_qwen_calib as sc
import sim_qwen_awq_gptq as sag
import sim_qwen_gptq_r3 as r3
import sim_qwen_wave2 as w2


TRUNK_VARIANTS = [
    "0_RTN_Prod",
    "1_R3_FP64_Exact",
    "2_R3_FP64_Q6_Hotspots",
]

MTP_VARIANTS = [
    "M_BF16",
    "M0_RTN_MTP",
    "M1_GPTQ_Q4_MTP",
    "M2_GPTQ_Q4_Q6_MTP",
]


# ---------------------------------------------------------------------------
# 1. Exact MTP Input Helper (matching runtime/qwen4exp.cpp:3651-3672)
# ---------------------------------------------------------------------------

def mtp_input_streams(
    emb_rows: torch.Tensor,          # [..., D] BF16/F32 embeddings for current tokens
    prev_streams: torch.Tensor,      # [..., 4, D] BF16/F32 previous L47 streams (zeros at t=0)
    w_norm_emb: torch.Tensor,        # [D] float32 pre_fc_norm_embedding.weight
    w_norm_hid: torch.Tensor,        # [10240] float32 pre_fc_norm_hidden.weight -> [4, D]
    w_fc_emb: torch.Tensor,          # [D, D] BF16/F32
    w_fc_hid: torch.Tensor,          # [D, D] BF16/F32
    eps: float = 1e-6,
    return_normed: bool = False,
):
    """
    Applies independent zero-centered RMSNorm on embedding [..., D] and per-stream
    group_rmsnorm_zc on prev_streams [..., 4, D], then projects:
      x_s = fc_embedding(norm_emb) + fc_hidden(norm_hid_s)
    matching Qwen4ExpSession::mtp_input / mtp_input_seq bit-for-bit in spirit.
    """
    D = emb_rows.shape[-1]
    wn_h = w_norm_hid.view(4, D)
    norm_emb = sq.rmsnorm_zc(emb_rows, w_norm_emb, eps=eps)           # [..., D] BF16
    norm_hid = sq.rmsnorm_zc(prev_streams, wn_h, eps=eps)             # [..., 4, D] BF16
    proj_emb = F.linear(norm_emb, w_fc_emb)                           # [..., D] BF16
    proj_hid = F.linear(norm_hid, w_fc_hid)                           # [..., 4, D] BF16
    out = (proj_hid.float() + proj_emb.float().unsqueeze(-2)).to(torch.bfloat16)
    if return_normed:
        return out, norm_emb, norm_hid
    return out


# ---------------------------------------------------------------------------
# 2. Pre-Run Self-Tests (Q4_G64 / Q6_G64 Static Groups + MTP Input Shift)
# ---------------------------------------------------------------------------

def run_fp64_mtp_self_tests(dev: torch.device) -> None:
    torch.manual_seed(777)

    # Test 1: q4g64 & q6g64 ActOrder static_groups .faf roundtrip = 0.0 + >=40% MSE cut
    K, R = 256, 32
    scale_ch = torch.ones(K, dtype=torch.float64)
    scale_ch[5] = 20.0
    scale_ch[140] = 16.0
    X = (torch.randn(512, K, dtype=torch.float64) @ (torch.randn(K, K, dtype=torch.float64) * 0.25 + torch.eye(K, dtype=torch.float64))) * scale_ch.unsqueeze(0)
    H = (X.T @ X) / 512.0
    H_gpu = H.float().to(dev)
    W = torch.randn(R, K, device=dev, dtype=torch.float32) * 0.05

    for fmt, G, levels in (("q4g64", 64, 15.0), ("q6g64", 64, 63.0)):
        Q_act = w2.quant_gptq_actorder_static_with_params(W, H_cpu_f64=H, fmt=fmt, damp=0.05, row_subbatch=16)[0]
        Q_r3 = r3.quant_gptq_actorder_static(W, H_cpu_f64=H, fmt=fmt, damp=0.05, row_subbatch=16)
        r3_diff = float((Q_act.float() - Q_r3.float()).abs().max().item())
        assert r3_diff == 0.0, f"Test1 {fmt} fast vs r3 mismatch: {r3_diff}"
        s_g, m_g, safe_sg = sag._rtn_minmax_params(W.view(R, K // G, G), levels)
        codes_g = ((Q_act.float().view(R, K // G, G) - m_g.unsqueeze(-1)) / safe_sg.unsqueeze(-1)).round().clamp(0.0, levels)
        codes_g = torch.where(s_g.unsqueeze(-1) > 0.0, codes_g, torch.zeros_like(codes_g))
        recon = (codes_g * s_g.unsqueeze(-1) + m_g.unsqueeze(-1)).reshape(R, K).to(torch.bfloat16)
        rt_diff = float((recon.float() - Q_act.float()).abs().max().item())
        assert rt_diff == 0.0, f"Test1 {fmt} static_groups roundtrip failed: {rt_diff}"

        Q_rtn = sq.fake_quant(W, fmt).float()
        mse_rtn = float((((Q_rtn - W) @ H_gpu) * (Q_rtn - W)).sum().item())
        mse_act = float((((Q_act.float() - W) @ H_gpu) * (Q_act.float() - W)).sum().item())
        cut = 1.0 - mse_act / mse_rtn
        assert cut >= 0.40, f"Test1 {fmt} outlier MSE cut {100*cut:.1f}% < 40%"
        print(f"[PASS] Test 1 ({fmt} ActOrder static_groups): r3_diff={r3_diff:.1e}, .faf_rt={rt_diff:.1e}, Outlier_H_MSE cut={100*cut:.1f}%")

    # Test 2: Exact MTP shift & group RMSNorm verification
    D, T_s = 64, 6
    trunk_s = torch.randn(T_s, 4, D, device=dev, dtype=torch.bfloat16)
    prev_s = torch.zeros_like(trunk_s)
    prev_s[1:] = trunk_s[:-1]
    emb_s = torch.randn(T_s, D, device=dev, dtype=torch.bfloat16)
    wn_e = torch.randn(D, device=dev, dtype=torch.float32) * 0.1
    wn_h = torch.randn(4 * D, device=dev, dtype=torch.float32) * 0.1
    wf_e = torch.randn(D, D, device=dev, dtype=torch.bfloat16) * 0.1
    wf_h = torch.randn(D, D, device=dev, dtype=torch.bfloat16) * 0.1

    out_mtp, ne, nh = mtp_input_streams(emb_s, prev_s, wn_e, wn_h, wf_e, wf_h, return_normed=True)
    # t=0 must depend ONLY on emb_s[0] since prev_s[0] == 0
    assert float(nh[0].abs().max().item()) == 0.0, "Test2 t=0 prev streams not zero!"
    # Manual reference at t=3, stream s=2
    x_e_ref = sq.rmsnorm_zc(emb_s[3:4], wn_e, 1e-6)
    x_h_ref = sq.rmsnorm_zc(trunk_s[2:3, 2, :], wn_h.view(4, D)[2], 1e-6)
    ref_t3_s2 = (F.linear(x_e_ref, wf_e).float() + F.linear(x_h_ref, wf_h).float()).to(torch.bfloat16)
    diff_t3_s2 = float((out_mtp[3, 2].float() - ref_t3_s2.squeeze(0).float()).abs().max().item())
    assert diff_t3_s2 == 0.0, f"Test2 mtp_input per-stream group_rmsnorm mismatch: {diff_t3_s2}"
    print(f"[PASS] Test 2 (mtp_input shift & group_rmsnorm): t=0 prev=0.0, t=3 s=2 diff={diff_t3_s2:.1e}")

    del X, H, H_gpu, W, trunk_s, prev_s, emb_s, wn_e, wn_h, wf_e, wf_h, out_mtp, ne, nh
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------

def _hmse(q_w: torch.Tensor, w_ref: torch.Tensor, H_f32: torch.Tensor) -> float:
    e = q_w.float() - w_ref.float()
    return float(((e @ H_f32) * e).sum().item())


def _hmse_ratio(q_w: torch.Tensor, rtn_w: torch.Tensor, w_ref: torch.Tensor, H_f32: torch.Tensor) -> float:
    return round(_hmse(q_w, w_ref, H_f32) / max(_hmse(rtn_w, w_ref, H_f32), 1e-12), 4)


def _eval_logits_pair(lgt_ref_f32: torch.Tensor, lgt_cand_f32: torch.Tensor, target_ids: torch.Tensor = None):
    lp_ref = F.log_softmax(lgt_ref_f32, dim=-1)
    p_ref = lp_ref.exp()
    lp_cand = F.log_softmax(lgt_cand_f32, dim=-1)
    kld_v = (p_ref * (lp_ref - lp_cand)).sum(dim=-1)
    t1_ref = lgt_ref_f32.argmax(dim=-1)
    t1_cand = lgt_cand_f32.argmax(dim=-1)
    top3_cand = lgt_cand_f32.topk(3, dim=-1).indices
    t1_match = float((t1_cand == t1_ref).sum().item())
    t3_match = float((top3_cand == t1_ref.unsqueeze(-1)).any(dim=-1).sum().item())
    nll_sum = float(F.nll_loss(lp_cand, target_ids, reduction="sum").item()) if target_ids is not None else 0.0
    return kld_v, t1_match, t3_match, nll_sum


# ---------------------------------------------------------------------------
# 4. Main Simulation: 48-Layer FP64 Trunk + mtp.layers.0
# ---------------------------------------------------------------------------

def run_fp64_mtp_sim(
    model_dir: str,
    out_json: str,
    n_seqs: int = 40,
    seq_len: int = 1536,
    cal_mb: int = 4,
    expert_chunk: int = 64,
    tau: float = 384.0,
    max_layers: int = None,
    ckpt_dir: str = "/mnt/nvme2/tmp-fp64-mtp-ckpt",
):
    dev = torch.device("cuda:0")
    cpu = torch.device("cpu")
    _ = torch.zeros(1, device=dev)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(dev)

    run_fp64_mtp_self_tests(dev)

    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    tok = sq.QwenTokenizer(os.path.join(model_dir, "tokenizer.json"))
    ids_cal_2d, dom_summary, cal_sha = sag.build_diverse_calib_sequences(tok, workspace_root, n_seqs=n_seqs, seq_len=seq_len)
    T_cal = n_seqs * seq_len

    eval_ids = tok.encode(sq.CALIB_DOC)
    T_eval = len(eval_ids)
    eval_ids_cpu = torch.tensor(eval_ids, dtype=torch.long)
    target_eval = torch.tensor(eval_ids[1:], device=dev, dtype=torch.long)

    reader = sq.SafetensorsReader(model_dir)
    lm = "model.language_model."
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg_root = json.load(f)
    tc = cfg_root.get("text_config", cfg_root)
    n_layers = min(tc["num_hidden_layers"], max_layers) if max_layers is not None else tc["num_hidden_layers"]
    layer_types = tc.get("layer_types")

    cos_c, sin_c = sq.build_qwen_rope(seq_len, dev, torch.bfloat16)
    cos_e, sin_e = sq.build_qwen_rope(T_eval, dev, torch.bfloat16)

    print(f"\n=== Strict FP64 Trunk ({n_layers} layers) + mtp.layers.0 Simulation ===", flush=True)
    print(f"Calib: {n_seqs}x{seq_len}={T_cal} (sha={cal_sha}) | domains={dom_summary} | tau={tau}", flush=True)

    t0 = time.time()
    prefetch_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    hmse_totals = {
        "gdn_in": {"rtn": 0.0, "act": 0.0},
        "gdn_out": {"rtn": 0.0, "act": 0.0},
        "attn_o": {"rtn": 0.0, "act": 0.0},
        "sh_down": {"rtn": 0.0, "act": 0.0},
        "exp_gu": {"rtn": 0.0, "act": 0.0},
        "exp_down": {"rtn": 0.0, "act": 0.0},
    }
    trunk_layers_log = []

    with torch.no_grad():
        emb_cpu = reader.read_tensor(lm + "embed_tokens.weight", cpu, torch.bfloat16)
        ckpt_file = os.path.join(ckpt_dir, "ckpt_latest.pt") if ckpt_dir else ""
        if ckpt_file and os.path.exists(ckpt_file):
            ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            assert ckpt["cal_sha"] == cal_sha and ckpt["n_layers"] == n_layers, "Checkpoint mismatch!"
            start_layer = int(ckpt["next_layer"])
            cal_states_cpu = ckpt["cal_states_cpu"]
            cal_states_q2_cpu = ckpt["cal_states_q2_cpu"]
            eval_bf16 = ckpt["eval_bf16"].to(dev)
            eval_vars = {vn: ckpt["eval_vars"][vn].to(dev) for vn in TRUNK_VARIANTS}
            hmse_totals = ckpt["hmse_totals"]
            trunk_layers_log = ckpt["trunk_layers_log"]
            del ckpt
            print(f"[CKPT] Resumed from {ckpt_file} at Layer {start_layer}/{n_layers}", flush=True)
        else:
            start_layer = 0
            cal_states_cpu = F.embedding(ids_cal_2d, emb_cpu).unsqueeze(2).expand(n_seqs, seq_len, 4, 2560).contiguous()
            cal_states_q2_cpu = cal_states_cpu.clone()
            eval_x0 = F.embedding(eval_ids_cpu, emb_cpu).to(dev)
            eval_bf16 = eval_x0.unsqueeze(1).expand(T_eval, 4, 2560).contiguous()
            eval_vars = {vn: eval_bf16.clone() for vn in TRUNK_VARIANTS}
            del eval_x0
        torch.cuda.empty_cache()

        for L in range(start_layer, n_layers):
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

                w_kp = reader.read_tensor(ple_p + "key_proj.weight", dev, torch.float32)
                w_vp = reader.read_tensor(ple_p + "value_proj.weight", dev, torch.float32)
                w_ple = torch.cat([w_kp, w_vp], dim=0)
                del w_kp, w_vp
                w_ple_bf16 = w_ple.to(torch.bfloat16)
                w_ple_rtn = sq.fake_quant(w_ple, "q6g64")
                del w_ple

                ple_cw = reader.read_tensor(ple_p + "conv1d.weight", dev, torch.float32)
                ple_nk = reader.read_tensor(ple_p + "norm_key.weight", dev, torch.float32)
                ple_nq = reader.read_tensor(ple_p + "norm_query.weight", dev, torch.float32)
                ple_nc = reader.read_tensor(ple_p + "norm_conv.weight", dev, torch.float32)

                for b_idx in range(n_seqs):
                    ur_c, mi_c = sq.ple_ngram_ids_py(mult, vsz, offs, ids_cal_2d[b_idx].tolist(), eos=248044)
                    ple_c = reader.gather_ple_rows(ple_p, ur_c, dev)[torch.tensor(mi_c, device=dev, dtype=torch.long)].reshape(seq_len, 2560)
                    xc_b = cal_states_cpu[b_idx].to(dev)
                    cal_states_cpu[b_idx] = sq.ple_forward(xc_b, ple_c, w_ple_bf16, ple_cw, ple_nk, ple_nq, ple_nc).cpu()
                    xq_b = cal_states_q2_cpu[b_idx].to(dev)
                    cal_states_q2_cpu[b_idx] = sq.ple_forward(xq_b, ple_c, w_ple_rtn, ple_cw, ple_nk, ple_nq, ple_nc).cpu()
                    del ple_c, xc_b, xq_b

                ur_e, mi_e = sq.ple_ngram_ids_py(mult, vsz, offs, eval_ids, eos=248044)
                ple_e = reader.gather_ple_rows(ple_p, ur_e, dev)[torch.tensor(mi_e, device=dev, dtype=torch.long)].reshape(T_eval, 2560)
                eval_bf16 = sq.ple_forward(eval_bf16, ple_e, w_ple_bf16, ple_cw, ple_nk, ple_nq, ple_nc)
                for vn in TRUNK_VARIANTS:
                    eval_vars[vn] = sq.ple_forward(eval_vars[vn], ple_e, w_ple_rtn, ple_cw, ple_nk, ple_nq, ple_nc)
                del ple_e, w_ple_bf16, w_ple_rtn, ple_cw, ple_nk, ple_nq, ple_nc

            # ---- 2. Attn / GDN Hyper-Connection Mix ----
            ahc = P + "attn_hyper_connection."
            a_wd = reader.read_tensor(ahc + "input_mix_weight_down.weight", dev, torch.float32)
            a_wi = reader.read_tensor(ahc + "block_inject_weight.weight", dev, torch.float32)
            a_wu = reader.read_tensor(ahc + "input_mix_weight_up.weight", dev, torch.float32)
            a_hn = reader.read_tensor(ahc + "hc_norm.weight", dev, torch.float32)
            a_wdf = a_wd * (1.0 + a_hn.unsqueeze(0))
            a_wif = a_wi * (1.0 + a_hn.unsqueeze(0))
            del a_wd, a_wi
            ahcd_bf16 = (a_wdf.to(torch.bfloat16), a_wif.to(torch.bfloat16))
            awu_bf16 = a_wu.to(torch.bfloat16)
            ahcd_q8 = (sq.fake_quant(a_wdf, "q8g64"), sq.fake_quant(a_wif, "q8g64"))
            awu_q8 = sq.fake_quant(a_wu, "q8g64")
            del a_wdf, a_wif, a_wu

            if not is_attn:
                gp = P + "linear_attn."
                w_qkv = reader.read_tensor(gp + "in_proj_qkv.weight", dev, torch.float32)
                w_z = reader.read_tensor(gp + "in_proj_z.weight", dev, torch.float32)
                w_b = reader.read_tensor(gp + "in_proj_b.weight", dev, torch.float32)
                w_a = reader.read_tensor(gp + "in_proj_a.weight", dev, torch.float32)
                w_gdn_in = torch.cat([w_qkv, w_z, w_b, w_a], dim=0)
                del w_qkv, w_z, w_b, w_a
                gin_bf16 = w_gdn_in.to(torch.bfloat16)
                w_gdn_out = reader.read_tensor(gp + "out_proj.weight", dev, torch.float32)
                gout_bf16 = w_gdn_out.to(torch.bfloat16)
                cw = reader.read_tensor(gp + "conv1d.weight", dev, torch.float32)
                A_log = reader.read_tensor(gp + "A_log", dev, torch.float32)
                dt_bias = reader.read_tensor(gp + "dt_bias", dev, torch.float32)
                gnorm = reader.read_tensor(gp + "norm.weight", dev, torch.float32)

                H_gdn_cpu = torch.zeros((2560, 2560), dtype=torch.float64)
                H_gout_cpu = torch.zeros((6144, 6144), dtype=torch.float64)
                for b0 in range(0, n_seqs, cal_mb):
                    b1 = min(b0 + cal_mb, n_seqs)
                    xb = cal_states_cpu[b0:b1].to(dev)
                    ub, win_b = sag.hc_mix_down_up_batched(xb, *ahcd_bf16, awu_bf16, a_hn, inject=True)
                    ym, og = sag.gdn_forward_batched(ub, gin_bf16, gout_bf16, cw, A_log, dt_bias, gnorm, return_o_gated=True)
                    cal_states_cpu[b0:b1] = (xb.float() + win_b.unsqueeze(-1) * ym.float().unsqueeze(2)).to(torch.bfloat16).cpu()

                    u_f64 = ub.reshape(-1, 2560).double()
                    og_f64 = og.reshape(-1, 6144).double()
                    H_gdn_cpu.add_((u_f64.T @ u_f64).cpu())
                    H_gout_cpu.add_((og_f64.T @ og_f64).cpu())
                    del xb, ub, win_b, ym, og, u_f64, og_f64

                H_gdn_cpu.div_(float(T_cal))
                H_gout_cpu.div_(float(T_cal))

                gin_rtn = sq.fake_quant(w_gdn_in, "q4g128")
                gin_act = w2.quant_gptq_actorder_static_with_params(w_gdn_in, H_cpu_f64=H_gdn_cpu, fmt="q4g128", damp=0.05, row_subbatch=40960)[0]
                gout_rtn = sq.fake_quant(w_gdn_out, "q6g64")
                gout_act = w2.quant_gptq_actorder_static_with_params(w_gdn_out, H_cpu_f64=H_gout_cpu, fmt="q6g64", damp=0.05, row_subbatch=2560)[0]

                H_gdn_g = H_gdn_cpu.float().to(dev)
                H_gout_g = H_gout_cpu.float().to(dev)
                hm_gi_r, hm_gi_a = _hmse(gin_rtn, w_gdn_in, H_gdn_g), _hmse(gin_act, w_gdn_in, H_gdn_g)
                hm_go_r, hm_go_a = _hmse(gout_rtn, w_gdn_out, H_gout_g), _hmse(gout_act, w_gdn_out, H_gout_g)
                hmse_totals["gdn_in"]["rtn"] += hm_gi_r
                hmse_totals["gdn_in"]["act"] += hm_gi_a
                hmse_totals["gdn_out"]["rtn"] += hm_go_r
                hmse_totals["gdn_out"]["act"] += hm_go_a
                lstat["gdn_in_ratio"] = round(hm_gi_a / max(hm_gi_r, 1e-12), 4)
                lstat["gdn_out_ratio"] = round(hm_go_a / max(hm_go_r, 1e-12), 4)
                del H_gdn_g, H_gout_g, H_gdn_cpu, H_gout_cpu, w_gdn_in, w_gdn_out

                # Advance Var 2 calibration trajectory (cal_states_q2_cpu)
                for b0 in range(0, n_seqs, cal_mb):
                    b1 = min(b0 + cal_mb, n_seqs)
                    xq_b = cal_states_q2_cpu[b0:b1].to(dev)
                    uq_b, winq_b = sag.hc_mix_down_up_batched(xq_b, *ahcd_q8, awu_q8, a_hn, inject=True)
                    ym_q = sag.gdn_forward_batched(uq_b, gin_act, gout_act, cw, A_log, dt_bias, gnorm)
                    cal_states_q2_cpu[b0:b1] = (xq_b.float() + winq_b.unsqueeze(-1) * ym_q.float().unsqueeze(2)).to(torch.bfloat16).cpu()
                    del xq_b, uq_b, winq_b, ym_q

                # Held-out eval states
                u_eb, win_eb = sq.hc_mix_down_up(eval_bf16, *ahcd_bf16, awu_bf16, a_hn, inject=True)
                y_eb = sq.gdn_forward(u_eb, gin_bf16, gout_bf16, cw, A_log, dt_bias, gnorm)
                eval_bf16 = (eval_bf16.float() + win_eb.unsqueeze(-1) * y_eb.float().unsqueeze(1)).to(torch.bfloat16)
                del u_eb, win_eb, y_eb

                g_map = {
                    "0_RTN_Prod": (gin_rtn, gout_rtn),
                    "1_R3_FP64_Exact": (gin_act, gout_rtn),
                    "2_R3_FP64_Q6_Hotspots": (gin_act, gout_act),
                }
                for vn in TRUNK_VARIANTS:
                    wi_v, wo_v = g_map[vn]
                    uv, wv = sq.hc_mix_down_up(eval_vars[vn], *ahcd_q8, awu_q8, a_hn, inject=True)
                    yv = sq.gdn_forward(uv, wi_v, wo_v, cw, A_log, dt_bias, gnorm)
                    eval_vars[vn] = (eval_vars[vn].float() + wv.unsqueeze(-1) * yv.float().unsqueeze(1)).to(torch.bfloat16)
                    del uv, wv, yv

                del gin_bf16, gout_bf16, gin_rtn, gin_act, gout_rtn, gout_act, cw, A_log, dt_bias, gnorm
            else:
                ap = P + "self_attn."
                wq = reader.read_tensor(ap + "q_proj.weight", dev, torch.float32)
                wk = reader.read_tensor(ap + "k_proj.weight", dev, torch.float32)
                wv = reader.read_tensor(ap + "v_proj.weight", dev, torch.float32)
                w_qkv = torch.cat([wq, wk, wv], dim=0)
                del wq, wk, wv
                w_ao = reader.read_tensor(ap + "o_proj.weight", dev, torch.float32)
                qkv_bf16, ao_bf16 = w_qkv.to(torch.bfloat16), w_ao.to(torch.bfloat16)
                qkv_rtn, ao_rtn = sq.fake_quant(w_qkv, "q6g64"), sq.fake_quant(w_ao, "q6g64")
                del w_qkv

                qn = reader.read_tensor(ap + "q_norm.weight", dev, torch.float32)
                kn = reader.read_tensor(ap + "k_norm.weight", dev, torch.float32)

                H_ao_cpu = torch.zeros((6144, 6144), dtype=torch.float64)
                for b0 in range(0, n_seqs, cal_mb):
                    b1 = min(b0 + cal_mb, n_seqs)
                    xb = cal_states_cpu[b0:b1].to(dev)
                    ub, win_b = sag.hc_mix_down_up_batched(xb, *ahcd_bf16, awu_bf16, a_hn, inject=True)
                    ym, og = w2.full_attn_forward_batched(ub, qkv_bf16, ao_bf16, qn, kn, cos_c, sin_c, return_o_gated=True)
                    cal_states_cpu[b0:b1] = (xb.float() + win_b.unsqueeze(-1) * ym.float().unsqueeze(2)).to(torch.bfloat16).cpu()
                    og_f64 = og.reshape(-1, 6144).double()
                    H_ao_cpu.add_((og_f64.T @ og_f64).cpu())
                    del xb, ub, win_b, ym, og, og_f64
                H_ao_cpu.div_(float(T_cal))

                ao_act = w2.quant_gptq_actorder_static_with_params(w_ao, H_cpu_f64=H_ao_cpu, fmt="q6g64", damp=0.05, row_subbatch=2560)[0]
                H_ao_g = H_ao_cpu.float().to(dev)
                hm_ao_r, hm_ao_a = _hmse(ao_rtn, w_ao, H_ao_g), _hmse(ao_act, w_ao, H_ao_g)
                hmse_totals["attn_o"]["rtn"] += hm_ao_r
                hmse_totals["attn_o"]["act"] += hm_ao_a
                lstat["attn_o_ratio"] = round(hm_ao_a / max(hm_ao_r, 1e-12), 4)
                del H_ao_g, H_ao_cpu, w_ao

                for b0 in range(0, n_seqs, cal_mb):
                    b1 = min(b0 + cal_mb, n_seqs)
                    xq_b = cal_states_q2_cpu[b0:b1].to(dev)
                    uq_b, winq_b = sag.hc_mix_down_up_batched(xq_b, *ahcd_q8, awu_q8, a_hn, inject=True)
                    ym_q = w2.full_attn_forward_batched(uq_b, qkv_rtn, ao_act, qn, kn, cos_c, sin_c)
                    cal_states_q2_cpu[b0:b1] = (xq_b.float() + winq_b.unsqueeze(-1) * ym_q.float().unsqueeze(2)).to(torch.bfloat16).cpu()
                    del xq_b, uq_b, winq_b, ym_q

                u_eb, win_eb = sq.hc_mix_down_up(eval_bf16, *ahcd_bf16, awu_bf16, a_hn, inject=True)
                y_eb = sq.full_attn_forward(u_eb, qkv_bf16, ao_bf16, qn, kn, cos_e, sin_e)
                eval_bf16 = (eval_bf16.float() + win_eb.unsqueeze(-1) * y_eb.float().unsqueeze(1)).to(torch.bfloat16)
                del u_eb, win_eb, y_eb

                for vn in TRUNK_VARIANTS:
                    ao_v = ao_act if vn == "2_R3_FP64_Q6_Hotspots" else ao_rtn
                    uv, wv = sq.hc_mix_down_up(eval_vars[vn], *ahcd_q8, awu_q8, a_hn, inject=True)
                    yv = sq.full_attn_forward(uv, qkv_rtn, ao_v, qn, kn, cos_e, sin_e)
                    eval_vars[vn] = (eval_vars[vn].float() + wv.unsqueeze(-1) * yv.float().unsqueeze(1)).to(torch.bfloat16)
                    del uv, wv, yv

                del qkv_bf16, ao_bf16, qkv_rtn, ao_rtn, ao_act, qn, kn

            del ahcd_bf16, awu_bf16, ahcd_q8, awu_q8, a_hn

            # ---- 3. MLP Hyper-Connection Mix ----
            mhc = P + "mlp_hyper_connection."
            m_wd = reader.read_tensor(mhc + "input_mix_weight_down.weight", dev, torch.float32)
            m_wi = reader.read_tensor(mhc + "block_inject_weight.weight", dev, torch.float32)
            m_wu = reader.read_tensor(mhc + "input_mix_weight_up.weight", dev, torch.float32)
            m_hn = reader.read_tensor(mhc + "hc_norm.weight", dev, torch.float32)
            m_wdf = m_wd * (1.0 + m_hn.unsqueeze(0))
            m_wif = m_wi * (1.0 + m_hn.unsqueeze(0))
            del m_wd, m_wi
            mhcd_bf16 = (m_wdf.to(torch.bfloat16), m_wif.to(torch.bfloat16))
            mwu_bf16 = m_wu.to(torch.bfloat16)
            mhcd_q8 = (sq.fake_quant(m_wdf, "q8g64"), sq.fake_quant(m_wif, "q8g64"))
            mwu_q8 = sq.fake_quant(m_wu, "q8g64")
            del m_wdf, m_wif, m_wu

            u_mlp_cal = torch.empty((T_cal, 2560), device=dev, dtype=torch.bfloat16)
            win_mlp_cal = torch.empty((T_cal, 4), device=dev, dtype=torch.float32)
            u_mlp_q2_cal = torch.empty((T_cal, 2560), device=dev, dtype=torch.bfloat16)
            win_mlp_q2_cal = torch.empty((T_cal, 4), device=dev, dtype=torch.float32)
            H_mlp_cpu = torch.zeros((2560, 2560), dtype=torch.float64)

            for b0 in range(0, n_seqs, cal_mb):
                b1 = min(b0 + cal_mb, n_seqs)
                s0, s1 = b0 * seq_len, b1 * seq_len
                xb = cal_states_cpu[b0:b1].to(dev)
                ub, win_b = sag.hc_mix_down_up_batched(xb, *mhcd_bf16, mwu_bf16, m_hn, inject=True)
                uf = ub.reshape(-1, 2560)
                u_mlp_cal[s0:s1] = uf
                win_mlp_cal[s0:s1] = win_b.reshape(-1, 4)
                uf64 = uf.double()
                H_mlp_cpu.add_((uf64.T @ uf64).cpu())

                xq_b = cal_states_q2_cpu[b0:b1].to(dev)
                uq_b, winq_b = sag.hc_mix_down_up_batched(xq_b, *mhcd_q8, mwu_q8, m_hn, inject=True)
                u_mlp_q2_cal[s0:s1] = uq_b.reshape(-1, 2560)
                win_mlp_q2_cal[s0:s1] = winq_b.reshape(-1, 4)
                del xb, ub, win_b, uf, uf64, xq_b, uq_b, winq_b

            H_mlp_cpu.div_(float(T_cal))
            act_precomp_mlp = r3.prepare_actorder_chol_cpu(H_mlp_cpu, damp=0.05)

            u_eb_mlp, win_eb_mlp = sq.hc_mix_down_up(eval_bf16, *mhcd_bf16, mwu_bf16, m_hn, inject=True)
            u_ev_mlp, win_ev_mlp = {}, {}
            for vn in TRUNK_VARIANTS:
                u_ev_mlp[vn], win_ev_mlp[vn] = sq.hc_mix_down_up(eval_vars[vn], *mhcd_q8, mwu_q8, m_hn, inject=True)
            del mhcd_bf16, mwu_bf16, mhcd_q8, mwu_q8, m_hn

            # ---- 4. Router & Shared Expert ----
            mp = P + "mlp."
            wg_r = reader.read_tensor(mp + "gate.weight", dev, torch.float32)
            wsg_r = reader.read_tensor(mp + "shared_expert_gate.weight", dev, torch.float32)
            w_router_bf16 = torch.cat([wg_r, wsg_r], dim=0).to(torch.bfloat16)
            del wg_r, wsg_r

            cal_route = sq.moe_route_qwen(u_mlp_cal, w_router_bf16)
            tidx_flat = cal_route[0].reshape(-1)
            ord_all = torch.argsort(tidx_flat, stable=True)
            sorted_tok_all = ord_all // 10
            sorted_w_all = cal_route[1].reshape(-1)[ord_all].unsqueeze(-1)
            counts_all = torch.bincount(tidx_flat, minlength=512)
            offsets_all = torch.zeros(513, device=dev, dtype=torch.long)
            torch.cumsum(counts_all, dim=0, out=offsets_all[1:])
            offsets_list = offsets_all.cpu().tolist()
            del tidx_flat, ord_all

            cal_route_q2 = sq.moe_route_qwen(u_mlp_q2_cal, w_router_bf16)
            tidx_q2_flat = cal_route_q2[0].reshape(-1)
            ord_q2 = torch.argsort(tidx_q2_flat, stable=True)
            sorted_tok_q2 = ord_q2 // 10
            sorted_w_q2 = cal_route_q2[1].reshape(-1)[ord_q2].unsqueeze(-1)
            counts_q2 = torch.bincount(tidx_q2_flat, minlength=512)
            offsets_q2 = torch.zeros(513, device=dev, dtype=torch.long)
            torch.cumsum(counts_q2, dim=0, out=offsets_q2[1:])
            offsets_q2_list = offsets_q2.cpu().tolist()
            del tidx_q2_flat, ord_q2

            eval_route_bf16 = sq.moe_route_qwen(u_eb_mlp, w_router_bf16)
            eval_routes = {vn: sq.moe_route_qwen(u_ev_mlp[vn], w_router_bf16) for vn in TRUNK_VARIANTS}
            del w_router_bf16

            w_sh_g = reader.read_tensor(mp + "shared_expert.gate_proj.weight", dev, torch.float32)
            w_sh_u = reader.read_tensor(mp + "shared_expert.up_proj.weight", dev, torch.float32)
            w_sh_gu = torch.cat([w_sh_g, w_sh_u], dim=0)
            del w_sh_g, w_sh_u
            w_sh_d = reader.read_tensor(mp + "shared_expert.down_proj.weight", dev, torch.float32)

            sh_gu_c = F.linear(u_mlp_cal, w_sh_gu.to(torch.bfloat16))
            sh_h_c = F.silu(sh_gu_c[:, :640]) * sh_gu_c[:, 640:]
            sh_out_cal = F.linear(sh_h_c, w_sh_d.to(torch.bfloat16)).float() * cal_route[2].unsqueeze(-1)
            H_shd_cpu = ((sh_h_c.double().T @ sh_h_c.double()) / float(T_cal)).cpu()
            del sh_gu_c, sh_h_c, cal_route

            w_sh_gu_rtn = sq.fake_quant(w_sh_gu, "q6g64")
            w_sh_d_rtn = sq.fake_quant(w_sh_d, "q6g64")
            w_sh_d_act = w2.quant_gptq_actorder_static_with_params(w_sh_d, H_cpu_f64=H_shd_cpu, fmt="q6g64", damp=0.05, row_subbatch=2560)[0]

            H_shd_g = H_shd_cpu.float().to(dev)
            hm_sh_r, hm_sh_a = _hmse(w_sh_d_rtn, w_sh_d, H_shd_g), _hmse(w_sh_d_act, w_sh_d, H_shd_g)
            hmse_totals["sh_down"]["rtn"] += hm_sh_r
            hmse_totals["sh_down"]["act"] += hm_sh_a
            lstat["sh_down_ratio"] = round(hm_sh_a / max(hm_sh_r, 1e-12), 4)
            del H_shd_g, H_shd_cpu

            def _run_sh(u_in, wgu, wdn, sh_c):
                g = F.linear(u_in, wgu)
                h = F.silu(g[:, :640]) * g[:, 640:]
                return F.linear(h, wdn).float() * sh_c.unsqueeze(-1)

            sh_out_q2_cal = _run_sh(u_mlp_q2_cal, w_sh_gu_rtn, w_sh_d_act, cal_route_q2[2])
            eval_sh_bf16 = _run_sh(u_eb_mlp, w_sh_gu.to(torch.bfloat16), w_sh_d.to(torch.bfloat16), eval_route_bf16[2])
            eval_sh_vars = {
                vn: _run_sh(u_ev_mlp[vn], w_sh_gu_rtn, w_sh_d_act if vn == "2_R3_FP64_Q6_Hotspots" else w_sh_d_rtn, eval_routes[vn][2])
                for vn in TRUNK_VARIANTS
            }
            del w_sh_gu, w_sh_d, w_sh_gu_rtn, w_sh_d_rtn, w_sh_d_act, cal_route_q2

            # ---- 5. Routed Experts Single-Pass Chunks (64 experts/chunk) ----
            gu_name = mp + "experts.gate_up_proj"
            dn_name = mp + "experts.down_proj"
            reader.info(gu_name)
            reader.info(dn_name)

            cal_routed = torch.zeros((T_cal, 2560), device=dev, dtype=torch.float32)
            cal_routed_q2 = torch.zeros((T_cal, 2560), device=dev, dtype=torch.float32)
            eval_routed_bf16 = torch.zeros((T_eval, 2560), device=dev, dtype=torch.float32)
            eval_routed_vars = {vn: torch.zeros((T_eval, 2560), device=dev, dtype=torch.float32) for vn in TRUNK_VARIANTS}
            slot_buf = torch.zeros((T_eval, 10, 2560), device=dev, dtype=torch.float32)

            def _load_chunk_cpu(c_start, c_cnt):
                return (
                    reader.read_slice_dim0(gu_name, c_start, c_cnt, cpu, torch.bfloat16),
                    reader.read_slice_dim0(dn_name, c_start, c_cnt, cpu, torch.bfloat16),
                )

            CHUNK = expert_chunk
            next_fut = prefetch_pool.submit(_load_chunk_cpu, 0, min(CHUNK, 512))
            H_mlp_gpu = H_mlp_cpu.float().to(dev)
            gu_r_tot, gu_a_tot, dn_r_tot, dn_a_tot = 0.0, 0.0, 0.0, 0.0

            for c0 in range(0, 512, CHUNK):
                cnt = min(CHUNK, 512 - c0)
                c1 = c0 + cnt
                gu_cpu, dn_cpu = next_fut.result()
                if c1 < 512:
                    next_fut = prefetch_pool.submit(_load_chunk_cpu, c1, min(CHUNK, 512 - c1))

                gu_bf16 = gu_cpu.to(dev)
                dn_bf16 = dn_cpu.to(dev)
                del gu_cpu, dn_cpu

                H_sum_f64 = torch.zeros((640, 640), device=dev, dtype=torch.float64)
                H_e_f32 = torch.zeros((cnt, 640, 640), device=dev, dtype=torch.float32)
                hits_chunk = counts_all[c0:c1]

                for le_i in range(cnt):
                    ge = c0 + le_i
                    p0, p1 = offsets_list[ge], offsets_list[ge + 1]
                    c_n = p1 - p0
                    if c_n <= 0:
                        continue
                    toks = sorted_tok_all[p0:p1]
                    xs = u_mlp_cal[toks]
                    gue = F.linear(xs, gu_bf16[le_i])
                    hs = F.silu(gue[:, :640]) * gue[:, 640:]
                    hs_f = hs.float()
                    h_sum_32 = hs_f.T @ hs_f
                    H_sum_f64.add_(h_sum_32.double())
                    H_e_f32[le_i] = h_sum_32 / float(c_n)

                    ye = F.linear(hs, dn_bf16[le_i]).float() * sorted_w_all[p0:p1]
                    cal_routed.index_add_(0, toks, ye)
                    del toks, xs, gue, hs, hs_f, h_sum_32, ye

                tot_hits_c = max(int(hits_chunk.sum().item()), 1)
                H_mean_c_f32 = (H_sum_f64 / float(tot_hits_c)).float()
                del H_sum_f64

                gu_rtn = sq.fake_quant(gu_bf16, "q4g128")
                gu_act = w2.quant_gptq_actorder_static_with_params(gu_bf16, precomp=act_precomp_mlp, fmt="q4g128", damp=0.05, row_subbatch=40960)[0]

                dn_rtn = sq.fake_quant(dn_bf16, "q4g128")
                dn_act = w2.quant_gptq_batched_exp_down_shrink_wave2(dn_bf16, H_e_f32, hits_chunk, H_mean_c_f32, tau=tau, base_damp=0.05, act_order=True, cd_refit=False, fmt="q4g128")

                gu_r_tot += _hmse(gu_rtn, gu_bf16, H_mlp_gpu)
                gu_a_tot += _hmse(gu_act, gu_bf16, H_mlp_gpu)
                for le_i in range(cnt):
                    hf = float(hits_chunk[le_i].item())
                    if hf <= 0.0:
                        continue
                    Hr_i = (hf * H_e_f32[le_i] + tau * H_mean_c_f32) / (hf + tau)
                    dn_r_tot += hf * _hmse(dn_rtn[le_i], dn_bf16[le_i], Hr_i)
                    dn_a_tot += hf * _hmse(dn_act[le_i], dn_bf16[le_i], Hr_i)

                del H_e_f32, H_mean_c_f32

                # Advance Var 2 calibration trajectory (true routed tokens on u_mlp_q2_cal)
                for le_i in range(cnt):
                    ge = c0 + le_i
                    p0, p1 = offsets_q2_list[ge], offsets_q2_list[ge + 1]
                    if p1 <= p0:
                        continue
                    toks_q = sorted_tok_q2[p0:p1]
                    xs_q = u_mlp_q2_cal[toks_q]
                    gq_e = F.linear(xs_q, gu_act[le_i])
                    hq_e = F.silu(gq_e[:, :640]) * gq_e[:, 640:]
                    ye_q = F.linear(hq_e, dn_act[le_i]).float() * sorted_w_q2[p0:p1]
                    cal_routed_q2.index_add_(0, toks_q, ye_q)
                    del toks_q, xs_q, gq_e, hq_e, ye_q

                # Held-out routing accumulation
                def _eval_route(u_in, rt, wgu_c, wdn_c, dst_acc):
                    tidx, tw, _ = rt
                    in_c = (tidx >= c0) & (tidx < c1)
                    if not in_c.any():
                        return
                    tok_pos, slot_pos = torch.nonzero(in_c, as_tuple=True)
                    loc_e = tidx[tok_pos, slot_pos] - c0
                    order = torch.argsort(loc_e, stable=True)
                    tok_s = tok_pos[order]
                    slot_s = slot_pos[order]
                    w_s = tw[tok_pos, slot_pos][order].unsqueeze(-1)
                    uniq_e, cnts = torch.unique_consecutive(loc_e[order], return_counts=True)
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

                _eval_route(u_eb_mlp, eval_route_bf16, gu_bf16, dn_bf16, eval_routed_bf16)
                v_w_map = {
                    "0_RTN_Prod": (gu_rtn, dn_rtn),
                    "1_R3_FP64_Exact": (gu_act, dn_act),
                    "2_R3_FP64_Q6_Hotspots": (gu_act, dn_act),
                }
                for vn in TRUNK_VARIANTS:
                    wgu_v, wdn_v = v_w_map[vn]
                    _eval_route(u_ev_mlp[vn], eval_routes[vn], wgu_v, wdn_v, eval_routed_vars[vn])

                del gu_bf16, dn_bf16, gu_rtn, gu_act, dn_rtn, dn_act

            tot_hits_layer = max(int(counts_all.sum().item()), 1)
            hmse_totals["exp_gu"]["rtn"] += gu_r_tot
            hmse_totals["exp_gu"]["act"] += gu_a_tot
            hmse_totals["exp_down"]["rtn"] += dn_r_tot / float(tot_hits_layer)
            hmse_totals["exp_down"]["act"] += dn_a_tot / float(tot_hits_layer)
            lstat["exp_gu_ratio"] = round(gu_a_tot / max(gu_r_tot, 1e-12), 4)
            lstat["exp_dn_ratio"] = round(dn_a_tot / max(dn_r_tot, 1e-12), 4)
            trunk_layers_log.append(lstat)

            cal_moe = (cal_routed + sh_out_cal).to(torch.bfloat16)
            cal_moe_q2 = (cal_routed_q2 + sh_out_q2_cal).to(torch.bfloat16)
            for b0 in range(0, n_seqs, cal_mb):
                b1 = min(b0 + cal_mb, n_seqs)
                s0, s1 = b0 * seq_len, b1 * seq_len
                if L + 1 < n_layers:
                    xb = cal_states_cpu[b0:b1].to(dev)
                    ym = cal_moe[s0:s1].view(b1 - b0, seq_len, 1, 2560).float()
                    wb = win_mlp_cal[s0:s1].view(b1 - b0, seq_len, 4, 1)
                    cal_states_cpu[b0:b1] = (xb.float() + wb * ym).to(torch.bfloat16).cpu()
                    del xb, ym, wb

                # Keep L47 output in cal_states_q2_cpu as input to MTP!
                xq = cal_states_q2_cpu[b0:b1].to(dev)
                ymq = cal_moe_q2[s0:s1].view(b1 - b0, seq_len, 1, 2560).float()
                wbq = win_mlp_q2_cal[s0:s1].view(b1 - b0, seq_len, 4, 1)
                cal_states_q2_cpu[b0:b1] = (xq.float() + wbq * ymq).to(torch.bfloat16).cpu()
                del xq, ymq, wbq

            ym_eb = (eval_routed_bf16 + eval_sh_bf16).to(torch.bfloat16)
            eval_bf16 = (eval_bf16.float() + win_eb_mlp.unsqueeze(-1) * ym_eb.float().unsqueeze(1)).to(torch.bfloat16)
            for vn in TRUNK_VARIANTS:
                ym_v = (eval_routed_vars[vn] + eval_sh_vars[vn]).to(torch.bfloat16)
                eval_vars[vn] = (eval_vars[vn].float() + win_ev_mlp[vn].unsqueeze(-1) * ym_v.float().unsqueeze(1)).to(torch.bfloat16)

            del u_mlp_cal, win_mlp_cal, u_mlp_q2_cal, win_mlp_q2_cal, H_mlp_cpu, H_mlp_gpu, act_precomp_mlp
            del counts_all, offsets_all, sorted_tok_all, sorted_w_all, counts_q2, offsets_q2, sorted_tok_q2, sorted_w_q2
            del sh_out_cal, sh_out_q2_cal, cal_routed, cal_routed_q2, cal_moe, cal_moe_q2, slot_buf
            del u_eb_mlp, win_eb_mlp, eval_route_bf16, eval_sh_bf16, eval_routed_bf16, ym_eb
            del u_ev_mlp, win_ev_mlp, eval_routes, eval_sh_vars, eval_routed_vars
            torch.cuda.empty_cache()

            peak_gb = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
            hs_info = f"gdn_in={lstat['gdn_in_ratio']:.3f} gout={lstat['gdn_out_ratio']:.3f}" if not is_attn else f"attn_o={lstat['attn_o_ratio']:.3f}"
            print(
                f"  [Layer {L:2d}/{n_layers-1}] {'Attn' if is_attn else 'GDN '} t={time.time()-tl:4.1f}s ({time.time()-t0:6.1f}s) | "
                f"{hs_info} sh={lstat['sh_down_ratio']:.3f} gu={lstat['exp_gu_ratio']:.3f} dn={lstat['exp_dn_ratio']:.3f} | VRAM={peak_gb:.2f}G",
                flush=True,
            )

            if ckpt_file and L in (7, 15, 23, 31, 39, 47):
                os.makedirs(ckpt_dir, exist_ok=True)
                tmp_ck = os.path.join(ckpt_dir, "ckpt_tmp.pt")
                torch.save({
                    "cal_sha": cal_sha,
                    "n_layers": n_layers,
                    "next_layer": L + 1,
                    "cal_states_cpu": cal_states_cpu,
                    "cal_states_q2_cpu": cal_states_q2_cpu,
                    "eval_bf16": eval_bf16.cpu(),
                    "eval_vars": {vn: eval_vars[vn].cpu() for vn in TRUNK_VARIANTS},
                    "hmse_totals": hmse_totals,
                    "trunk_layers_log": trunk_layers_log,
                }, tmp_ck)
                os.replace(tmp_ck, ckpt_file)
                print(f"  [CKPT] Saved layer {L} checkpoint -> {ckpt_file}", flush=True)

        del cal_states_cpu
        torch.cuda.empty_cache()

        # ---- 6. Trunk Final HC Mixer + Logit Evaluation ----
        fhc = lm + "hyper_connection_mixer."
        f_wd = reader.read_tensor(fhc + "input_mix_weight_down.weight", dev, torch.float32)
        f_wu = reader.read_tensor(fhc + "input_mix_weight_up.weight", dev, torch.float32)
        f_hn = reader.read_tensor(fhc + "hc_norm.weight", dev, torch.float32)
        f_wdf = f_wd * (1.0 + f_hn.unsqueeze(0))
        del f_wd
        dummy_inj = torch.zeros((4, 10240), device=dev, dtype=torch.float32)

        lm_w = reader.read_tensor("lm_head.weight", dev, torch.bfloat16)
        qlm_q8 = sq.fake_quant_rows(lm_w, "q8g64")
        f_wd_q8, f_wu_q8 = sq.fake_quant(f_wdf, "q8g64"), sq.fake_quant(f_wu, "q8g64")

        f_bf16_u, _ = sq.hc_mix_down_up(eval_bf16, f_wdf.to(torch.bfloat16), dummy_inj, f_wu.to(torch.bfloat16), f_hn, inject=False)
        del f_wdf, f_wu, f_hn, eval_bf16

        TOK_CHUNK = 256
        n_eval = T_eval - 1
        bf16_nll_sum = 0.0
        for s0 in range(0, n_eval, TOK_CHUNK):
            s1 = min(s0 + TOK_CHUNK, n_eval)
            lgt = F.linear(f_bf16_u[s0:s1], lm_w).float()
            bf16_nll_sum += F.nll_loss(F.log_softmax(lgt, dim=-1), target_eval[s0:s1], reduction="sum").item()
            del lgt
        bf16_ppl = math.exp(bf16_nll_sum / float(n_eval))

        trunk_results = []
        trunk_u_q2 = None
        print(f"\n=== TRUNK HELD-OUT RESULTS (FP64 Hessians, T_eval={T_eval}, BF16 PPL={bf16_ppl:.3f}) ===", flush=True)
        print(f"{'Variant':<24} | {'Mean KLD':>10} | {'Med KLD':>10} | {'p95 KLD':>10} | {'PPL':>7} | {'dPPL':>7} | {'Top1%':>6} | {'Top5%':>6}", flush=True)
        print("-" * 102, flush=True)

        for vn in TRUNK_VARIANTS:
            u_cand, _ = sq.hc_mix_down_up(eval_vars[vn], f_wd_q8, dummy_inj, f_wu_q8, reader.read_tensor(fhc + "hc_norm.weight", dev, torch.float32), inject=False)
            if vn == "2_R3_FP64_Q6_Hotspots":
                trunk_u_q2 = u_cand.clone()
            kld_parts = []
            nll_sum, t1_hits, t5_hits = 0.0, 0, 0
            for s0 in range(0, n_eval, TOK_CHUNK):
                s1 = min(s0 + TOK_CHUNK, n_eval)
                ref_lgt = F.linear(f_bf16_u[s0:s1], lm_w).float()
                ref_lp = F.log_softmax(ref_lgt, dim=-1)
                q_lgt = F.linear(u_cand[s0:s1], qlm_q8).float()
                q_lp = F.log_softmax(q_lgt, dim=-1)
                kld_parts.append((ref_lp.exp() * (ref_lp - q_lp)).sum(dim=-1))
                nll_sum += F.nll_loss(q_lp, target_eval[s0:s1], reduction="sum").item()
                ref_t1 = ref_lgt.argmax(dim=-1)
                t1_hits += (q_lgt.argmax(dim=-1) == ref_t1).sum().item()
                t5_hits += (q_lgt.topk(5, dim=-1).indices == ref_t1.unsqueeze(-1)).any(dim=-1).sum().item()
                del ref_lgt, ref_lp, q_lgt, q_lp
            kld_v = torch.cat(kld_parts, dim=0)
            mk = float(kld_v.mean().item())
            medk = float(torch.quantile(kld_v, 0.50).item())
            p95k = float(torch.quantile(kld_v, 0.95).item())
            ppl = math.exp(nll_sum / float(n_eval))
            dppl = ppl - bf16_ppl
            t1, t5 = 100.0 * t1_hits / float(n_eval), 100.0 * t5_hits / float(n_eval)
            print(f"{vn:<24} | {mk:10.6f} | {medk:10.6f} | {p95k:10.6f} | {ppl:7.3f} | {dppl:+7.3f} | {t1:5.1f}% | {t5:5.1f}%", flush=True)
            trunk_results.append({
                "variant": vn, "mean_kld": round(mk, 6), "med_kld": round(medk, 6), "p95_kld": round(p95k, 6),
                "ppl": round(ppl, 4), "dppl": round(dppl, 4), "top1_pct": round(t1, 2), "top5_pct": round(t5, 2),
            })
            del u_cand

        # Save L47 winning quantized trunk state for MTP evaluation, drop other eval states
        eval_trunk_q2_l47 = eval_vars["2_R3_FP64_Q6_Hotspots"]
        del eval_vars, f_wd_q8, f_wu_q8
        torch.cuda.empty_cache()

        trunk_hmse_summary = {
            cls: round(hmse_totals[cls]["act"] / max(hmse_totals[cls]["rtn"], 1e-12), 4)
            for cls in hmse_totals
        }
        print(f"Trunk Aggregate Hessian MSE (act/rtn): {trunk_hmse_summary}", flush=True)

        # ===================================================================
        # 7. MTP.LAYERS.0 Calibration & Held-Out Verification (FP64 Hessians)
        # ===================================================================
        print(f"\n=== MTP.LAYERS.0 FP64 Calibration & Evaluation (on Quantized Trunk L47 Output) ===", flush=True)
        t_mtp0 = time.time()
        mtp_hmse = {}

        wn_emb = reader.read_tensor("mtp.pre_fc_norm_embedding.weight", dev, torch.float32)
        wn_hid = reader.read_tensor("mtp.pre_fc_norm_hidden.weight", dev, torch.float32)
        w_fc_emb = reader.read_tensor("mtp.fc_embedding.weight", dev, torch.float32)
        w_fc_hid = reader.read_tensor("mtp.fc_hidden.weight", dev, torch.float32)
        w_fc_emb_bf16, w_fc_hid_bf16 = w_fc_emb.to(torch.bfloat16), w_fc_hid.to(torch.bfloat16)

        # Calibration shifted prev_streams from 2_R3_FP64_Q6_Hotspots L47 output
        H_fc_emb_cpu = torch.zeros((2560, 2560), dtype=torch.float64)
        H_fc_hid_cpu = torch.zeros((2560, 2560), dtype=torch.float64)
        mtp_cal_x_cpu = torch.empty((n_seqs, seq_len, 4, 2560), dtype=torch.bfloat16)

        for b0 in range(0, n_seqs, cal_mb):
            b1 = min(b0 + cal_mb, n_seqs)
            emb_b = F.embedding(ids_cal_2d[b0:b1], emb_cpu).to(dev)
            prev_b = torch.zeros((b1 - b0, seq_len, 4, 2560), device=dev, dtype=torch.bfloat16)
            prev_b[:, 1:] = cal_states_q2_cpu[b0:b1, :-1].to(dev)
            mx_b, ne_b, nh_b = mtp_input_streams(emb_b, prev_b, wn_emb, wn_hid, w_fc_emb_bf16, w_fc_hid_bf16, return_normed=True)
            mtp_cal_x_cpu[b0:b1] = mx_b.cpu()

            ne_f64 = ne_b.reshape(-1, 2560).double()
            nh_f64 = nh_b.reshape(-1, 2560).double()
            H_fc_emb_cpu.add_((ne_f64.T @ ne_f64).cpu())
            H_fc_hid_cpu.add_((nh_f64.T @ nh_f64).cpu())
            del emb_b, prev_b, mx_b, ne_b, nh_b, ne_f64, nh_f64

        H_fc_emb_cpu.div_(float(T_cal))
        H_fc_hid_cpu.div_(float(T_cal * 4))
        del cal_states_q2_cpu

        fce_rtn = sq.fake_quant(w_fc_emb, "q4g64")
        fce_act = w2.quant_gptq_actorder_static_with_params(w_fc_emb, H_cpu_f64=H_fc_emb_cpu, fmt="q4g64", damp=0.05, row_subbatch=2560)[0]
        fch_rtn = sq.fake_quant(w_fc_hid, "q4g64")
        fch_act = w2.quant_gptq_actorder_static_with_params(w_fc_hid, H_cpu_f64=H_fc_hid_cpu, fmt="q4g64", damp=0.05, row_subbatch=2560)[0]

        mtp_hmse["fc_embedding_q4g64"] = _hmse_ratio(fce_act, fce_rtn, w_fc_emb, H_fc_emb_cpu.float().to(dev))
        mtp_hmse["fc_hidden_q4g64"] = _hmse_ratio(fch_act, fch_rtn, w_fc_hid, H_fc_hid_cpu.float().to(dev))
        del w_fc_emb, w_fc_hid, H_fc_emb_cpu, H_fc_hid_cpu

        # Held-out MTP initial streams from eval_trunk_q2_l47
        eval_emb_all = F.embedding(eval_ids_cpu, emb_cpu).to(dev)
        eval_prev_h = torch.zeros_like(eval_trunk_q2_l47)
        eval_prev_h[1:] = eval_trunk_q2_l47[:-1]
        del emb_cpu, eval_trunk_q2_l47

        mtp_eval_states = {
            "M_BF16": mtp_input_streams(eval_emb_all, eval_prev_h, wn_emb, wn_hid, w_fc_emb_bf16, w_fc_hid_bf16),
            "M0_RTN_MTP": mtp_input_streams(eval_emb_all, eval_prev_h, wn_emb, wn_hid, fce_rtn, fch_rtn),
            "M1_GPTQ_Q4_MTP": mtp_input_streams(eval_emb_all, eval_prev_h, wn_emb, wn_hid, fce_act, fch_act),
            "M2_GPTQ_Q4_Q6_MTP": mtp_input_streams(eval_emb_all, eval_prev_h, wn_emb, wn_hid, fce_act, fch_act),
        }
        del eval_emb_all, eval_prev_h, wn_emb, wn_hid, w_fc_emb_bf16, w_fc_hid_bf16, fce_rtn, fce_act, fch_rtn, fch_act

        # ---- 7a. MTP Attn Hyper-Connection + Full Attention ----
        MP = "mtp.layers.0."
        mahc = MP + "attn_hyper_connection."
        ma_wd = reader.read_tensor(mahc + "input_mix_weight_down.weight", dev, torch.float32)
        ma_wi = reader.read_tensor(mahc + "block_inject_weight.weight", dev, torch.float32)
        ma_wu = reader.read_tensor(mahc + "input_mix_weight_up.weight", dev, torch.float32)
        ma_hn = reader.read_tensor(mahc + "hc_norm.weight", dev, torch.float32)
        ma_wdf, ma_wif = ma_wd * (1.0 + ma_hn.unsqueeze(0)), ma_wi * (1.0 + ma_hn.unsqueeze(0))
        del ma_wd, ma_wi
        mahcd_bf16 = (ma_wdf.to(torch.bfloat16), ma_wif.to(torch.bfloat16))
        mawu_bf16 = ma_wu.to(torch.bfloat16)
        mahcd_q8 = (sq.fake_quant(ma_wdf, "q8g64"), sq.fake_quant(ma_wif, "q8g64"))
        mawu_q8 = sq.fake_quant(ma_wu, "q8g64")
        del ma_wdf, ma_wif, ma_wu

        map_p = MP + "self_attn."
        mwq = reader.read_tensor(map_p + "q_proj.weight", dev, torch.float32)
        mwk = reader.read_tensor(map_p + "k_proj.weight", dev, torch.float32)
        mwv = reader.read_tensor(map_p + "v_proj.weight", dev, torch.float32)
        mw_qkv = torch.cat([mwq, mwk, mwv], dim=0)
        del mwq, mwk, mwv
        mw_ao = reader.read_tensor(map_p + "o_proj.weight", dev, torch.float32)
        mqkv_bf16, mao_bf16 = mw_qkv.to(torch.bfloat16), mw_ao.to(torch.bfloat16)
        mqkv_rtn, mao_rtn = sq.fake_quant(mw_qkv, "q6g64"), sq.fake_quant(mw_ao, "q6g64")
        del mw_qkv
        mqn = reader.read_tensor(map_p + "q_norm.weight", dev, torch.float32)
        mkn = reader.read_tensor(map_p + "k_norm.weight", dev, torch.float32)

        H_mtp_ao_cpu = torch.zeros((6144, 6144), dtype=torch.float64)
        for b0 in range(0, n_seqs, cal_mb):
            b1 = min(b0 + cal_mb, n_seqs)
            xb = mtp_cal_x_cpu[b0:b1].to(dev)
            ub, win_b = sag.hc_mix_down_up_batched(xb, *mahcd_bf16, mawu_bf16, ma_hn, inject=True)
            ym, og = w2.full_attn_forward_batched(ub, mqkv_bf16, mao_bf16, mqn, mkn, cos_c, sin_c, return_o_gated=True)
            mtp_cal_x_cpu[b0:b1] = (xb.float() + win_b.unsqueeze(-1) * ym.float().unsqueeze(2)).to(torch.bfloat16).cpu()
            og_f64 = og.reshape(-1, 6144).double()
            H_mtp_ao_cpu.add_((og_f64.T @ og_f64).cpu())
            del xb, ub, win_b, ym, og, og_f64
        H_mtp_ao_cpu.div_(float(T_cal))

        mao_act = w2.quant_gptq_actorder_static_with_params(mw_ao, H_cpu_f64=H_mtp_ao_cpu, fmt="q6g64", damp=0.05, row_subbatch=2560)[0]
        mtp_hmse["mtp_attn_o_q6g64"] = _hmse_ratio(mao_act, mao_rtn, mw_ao, H_mtp_ao_cpu.float().to(dev))
        del H_mtp_ao_cpu, mw_ao

        for mn in MTP_VARIANTS:
            hd = mahcd_bf16 if mn == "M_BF16" else mahcd_q8
            hu = mawu_bf16 if mn == "M_BF16" else mawu_q8
            qkv_m = mqkv_bf16 if mn == "M_BF16" else mqkv_rtn
            ao_m = mao_bf16 if mn == "M_BF16" else (mao_act if mn == "M2_GPTQ_Q4_Q6_MTP" else mao_rtn)
            uv, wv = sq.hc_mix_down_up(mtp_eval_states[mn], *hd, hu, ma_hn, inject=True)
            yv = sq.full_attn_forward(uv, qkv_m, ao_m, mqn, mkn, cos_e, sin_e)
            mtp_eval_states[mn] = (mtp_eval_states[mn].float() + wv.unsqueeze(-1) * yv.float().unsqueeze(1)).to(torch.bfloat16)
            del uv, wv, yv
        del mahcd_bf16, mawu_bf16, mahcd_q8, mawu_q8, ma_hn, mqkv_bf16, mao_bf16, mqkv_rtn, mao_rtn, mao_act, mqn, mkn

        # ---- 7b. MTP MLP HC + Shared Expert + 512 Routed Experts ----
        mmhc = MP + "mlp_hyper_connection."
        mm_wd = reader.read_tensor(mmhc + "input_mix_weight_down.weight", dev, torch.float32)
        mm_wi = reader.read_tensor(mmhc + "block_inject_weight.weight", dev, torch.float32)
        mm_wu = reader.read_tensor(mmhc + "input_mix_weight_up.weight", dev, torch.float32)
        mm_hn = reader.read_tensor(mmhc + "hc_norm.weight", dev, torch.float32)
        mm_wdf, mm_wif = mm_wd * (1.0 + mm_hn.unsqueeze(0)), mm_wi * (1.0 + mm_hn.unsqueeze(0))
        del mm_wd, mm_wi
        mmhcd_bf16 = (mm_wdf.to(torch.bfloat16), mm_wif.to(torch.bfloat16))
        mmwu_bf16 = mm_wu.to(torch.bfloat16)
        mmhcd_q8 = (sq.fake_quant(mm_wdf, "q8g64"), sq.fake_quant(mm_wif, "q8g64"))
        mmwu_q8 = sq.fake_quant(mm_wu, "q8g64")
        del mm_wdf, mm_wif, mm_wu

        u_mtp_cal = torch.empty((T_cal, 2560), device=dev, dtype=torch.bfloat16)
        H_mtp_mlp_cpu = torch.zeros((2560, 2560), dtype=torch.float64)
        for b0 in range(0, n_seqs, cal_mb):
            b1 = min(b0 + cal_mb, n_seqs)
            xb = mtp_cal_x_cpu[b0:b1].to(dev)
            ub, _ = sag.hc_mix_down_up_batched(xb, *mmhcd_bf16, mmwu_bf16, mm_hn, inject=True)
            uf = ub.reshape(-1, 2560)
            u_mtp_cal[b0 * seq_len : b1 * seq_len] = uf
            uf64 = uf.double()
            H_mtp_mlp_cpu.add_((uf64.T @ uf64).cpu())
            del xb, ub, uf, uf64
        H_mtp_mlp_cpu.div_(float(T_cal))
        del mtp_cal_x_cpu
        act_precomp_mtp_mlp = r3.prepare_actorder_chol_cpu(H_mtp_mlp_cpu, damp=0.05)

        u_ev_mtp, win_ev_mtp = {}, {}
        for mn in MTP_VARIANTS:
            hd = mmhcd_bf16 if mn == "M_BF16" else mmhcd_q8
            hu = mmwu_bf16 if mn == "M_BF16" else mmwu_q8
            u_ev_mtp[mn], win_ev_mtp[mn] = sq.hc_mix_down_up(mtp_eval_states[mn], *hd, hu, mm_hn, inject=True)
        del mmhcd_bf16, mmwu_bf16, mmhcd_q8, mmwu_q8, mm_hn

        mmp = MP + "mlp."
        mwg_r = reader.read_tensor(mmp + "gate.weight", dev, torch.float32)
        mwsg_r = reader.read_tensor(mmp + "shared_expert_gate.weight", dev, torch.float32)
        mw_router = torch.cat([mwg_r, mwsg_r], dim=0).to(torch.bfloat16)
        del mwg_r, mwsg_r

        m_cal_route = sq.moe_route_qwen(u_mtp_cal, mw_router)
        m_tidx_flat = m_cal_route[0].reshape(-1)
        m_ord = torch.argsort(m_tidx_flat, stable=True)
        m_sorted_tok = m_ord // 10
        m_counts = torch.bincount(m_tidx_flat, minlength=512)
        m_offsets = torch.zeros(513, device=dev, dtype=torch.long)
        torch.cumsum(m_counts, dim=0, out=m_offsets[1:])
        m_offsets_list = m_offsets.cpu().tolist()
        del m_tidx_flat, m_ord

        m_eval_routes = {mn: sq.moe_route_qwen(u_ev_mtp[mn], mw_router) for mn in MTP_VARIANTS}
        del mw_router

        mw_sh_g = reader.read_tensor(mmp + "shared_expert.gate_proj.weight", dev, torch.float32)
        mw_sh_u = reader.read_tensor(mmp + "shared_expert.up_proj.weight", dev, torch.float32)
        mw_sh_gu = torch.cat([mw_sh_g, mw_sh_u], dim=0)
        del mw_sh_g, mw_sh_u
        mw_sh_d = reader.read_tensor(mmp + "shared_expert.down_proj.weight", dev, torch.float32)

        msh_gu_c = F.linear(u_mtp_cal, mw_sh_gu.to(torch.bfloat16))
        msh_h_c = F.silu(msh_gu_c[:, :640]) * msh_gu_c[:, 640:]
        H_mtp_shd_cpu = ((msh_h_c.double().T @ msh_h_c.double()) / float(T_cal)).cpu()
        del msh_gu_c, msh_h_c, m_cal_route

        msh_gu_bf16, msh_d_bf16 = mw_sh_gu.to(torch.bfloat16), mw_sh_d.to(torch.bfloat16)
        msh_gu_rtn, msh_d_rtn = sq.fake_quant(mw_sh_gu, "q6g64"), sq.fake_quant(mw_sh_d, "q6g64")
        msh_d_act = w2.quant_gptq_actorder_static_with_params(mw_sh_d, H_cpu_f64=H_mtp_shd_cpu, fmt="q6g64", damp=0.05, row_subbatch=2560)[0]
        mtp_hmse["mtp_sh_down_q6g64"] = _hmse_ratio(msh_d_act, msh_d_rtn, mw_sh_d, H_mtp_shd_cpu.float().to(dev))
        del mw_sh_gu, mw_sh_d, H_mtp_shd_cpu

        mtp_eval_sh = {}
        for mn in MTP_VARIANTS:
            wgu_s = msh_gu_bf16 if mn == "M_BF16" else msh_gu_rtn
            wdn_s = msh_d_bf16 if mn == "M_BF16" else (msh_d_act if mn == "M2_GPTQ_Q4_Q6_MTP" else msh_d_rtn)
            g = F.linear(u_ev_mtp[mn], wgu_s)
            h = F.silu(g[:, :640]) * g[:, 640:]
            mtp_eval_sh[mn] = F.linear(h, wdn_s).float() * m_eval_routes[mn][2].unsqueeze(-1)
            del g, h
        del msh_gu_bf16, msh_d_bf16, msh_gu_rtn, msh_d_rtn, msh_d_act

        m_gu_name = mmp + "experts.gate_up_proj"
        m_dn_name = mmp + "experts.down_proj"
        reader.info(m_gu_name)
        reader.info(m_dn_name)
        mtp_routed = {mn: torch.zeros((T_eval, 2560), device=dev, dtype=torch.float32) for mn in MTP_VARIANTS}
        m_slot_buf = torch.zeros((T_eval, 10, 2560), device=dev, dtype=torch.float32)

        def _load_m_chunk(c_start, c_cnt):
            return (
                reader.read_slice_dim0(m_gu_name, c_start, c_cnt, cpu, torch.bfloat16),
                reader.read_slice_dim0(m_dn_name, c_start, c_cnt, cpu, torch.bfloat16),
            )

        CHUNK = expert_chunk
        next_fut = prefetch_pool.submit(_load_m_chunk, 0, min(CHUNK, 512))
        H_mtp_mlp_g = H_mtp_mlp_cpu.float().to(dev)
        m_gu_r, m_gu_a, m_dn_r, m_dn_a = 0.0, 0.0, 0.0, 0.0

        for c0 in range(0, 512, CHUNK):
            cnt = min(CHUNK, 512 - c0)
            c1 = c0 + cnt
            gu_cpu, dn_cpu = next_fut.result()
            if c1 < 512:
                next_fut = prefetch_pool.submit(_load_m_chunk, c1, min(CHUNK, 512 - c1))
            gu_bf16, dn_bf16 = gu_cpu.to(dev), dn_cpu.to(dev)
            del gu_cpu, dn_cpu

            H_sum_64 = torch.zeros((cnt, 640, 640), device=dev, dtype=torch.float64)
            H_e_64 = torch.zeros((cnt, 640, 640), device=dev, dtype=torch.float64)
            hits_c = m_counts[c0:c1]

            for le_i in range(cnt):
                ge = c0 + le_i
                p0, p1 = m_offsets_list[ge], m_offsets_list[ge + 1]
                c_n = p1 - p0
                if c_n <= 0:
                    continue
                xs = u_mtp_cal[m_sorted_tok[p0:p1]]
                gue = F.linear(xs, gu_bf16[le_i])
                hs_d = (F.silu(gue[:, :640]) * gue[:, 640:]).double()
                hs64 = hs_d.T @ hs_d
                H_sum_64[le_i] = hs64
                H_e_64[le_i] = hs64 / float(c_n)
                del xs, gue, hs_d, hs64

            tot_mc = max(int(hits_c.sum().item()), 1)
            H_mean_64 = H_sum_64.sum(dim=0) / float(tot_mc)
            del H_sum_64

            gu_rtn = sq.fake_quant(gu_bf16, "q4g128")
            gu_act = w2.quant_gptq_actorder_static_with_params(gu_bf16, precomp=act_precomp_mtp_mlp, fmt="q4g128", damp=0.05, row_subbatch=40960)[0]
            dn_rtn = sq.fake_quant(dn_bf16, "q4g128")
            dn_act = w2.quant_gptq_batched_exp_down_shrink_wave2(dn_bf16, H_e_64, hits_c, H_mean_64, tau=tau, base_damp=0.05, act_order=True, cd_refit=False, fmt="q4g128")

            m_gu_r += _hmse(gu_rtn, gu_bf16, H_mlp_gpu if False else H_mtp_mlp_g)
            m_gu_a += _hmse(gu_act, gu_bf16, H_mtp_mlp_g)
            for le_i in range(cnt):
                hf = float(hits_c[le_i].item())
                if hf <= 0.0:
                    continue
                Hr_i = ((hf * H_e_64[le_i] + tau * H_mean_64) / (hf + tau)).float()
                m_dn_r += hf * _hmse(dn_rtn[le_i], dn_bf16[le_i], Hr_i)
                m_dn_a += hf * _hmse(dn_act[le_i], dn_bf16[le_i], Hr_i)
            del H_e_64, H_mean_64

            mw_map = {
                "M_BF16": (gu_bf16, dn_bf16),
                "M0_RTN_MTP": (gu_rtn, dn_rtn),
                "M1_GPTQ_Q4_MTP": (gu_act, dn_act),
                "M2_GPTQ_Q4_Q6_MTP": (gu_act, dn_act),
            }
            for mn in MTP_VARIANTS:
                tidx, tw, _ = m_eval_routes[mn]
                in_c = (tidx >= c0) & (tidx < c1)
                if not in_c.any():
                    continue
                tok_pos, slot_pos = torch.nonzero(in_c, as_tuple=True)
                loc_e = tidx[tok_pos, slot_pos] - c0
                order = torch.argsort(loc_e, stable=True)
                tok_s = tok_pos[order]
                slot_s = slot_pos[order]
                w_s = tw[tok_pos, slot_pos][order].unsqueeze(-1)
                uniq_e, cnts = torch.unique_consecutive(loc_e[order], return_counts=True)
                wgu_c, wdn_c = mw_map[mn]
                x_hits = u_ev_mtp[mn][tok_s]
                outs = torch.empty_like(x_hits)
                pos = 0
                for le_i, cnt_i in zip(uniq_e.tolist(), cnts.tolist()):
                    xs = x_hits[pos : pos + cnt_i]
                    gue = F.linear(xs, wgu_c[le_i])
                    hs = F.silu(gue[:, :640]) * gue[:, 640:]
                    outs[pos : pos + cnt_i] = F.linear(hs, wdn_c[le_i])
                    pos += cnt_i
                m_slot_buf.zero_()
                m_slot_buf[tok_s, slot_s] = outs.float() * w_s
                mtp_routed[mn].add_(m_slot_buf.sum(dim=1))

            del gu_bf16, dn_bf16, gu_rtn, gu_act, dn_rtn, dn_act

        mtp_hmse["mtp_exp_gu_q4g128"] = round(m_gu_a / max(m_gu_r, 1e-12), 4)
        mtp_hmse["mtp_exp_down_q4g128"] = round(m_dn_a / max(m_dn_r, 1e-12), 4)
        del u_mtp_cal, H_mtp_mlp_cpu, H_mtp_mlp_g, act_precomp_mtp_mlp, m_counts, m_offsets, m_sorted_tok, m_slot_buf

        for mn in MTP_VARIANTS:
            ym_m = (mtp_routed[mn] + mtp_eval_sh[mn]).to(torch.bfloat16)
            mtp_eval_states[mn] = (mtp_eval_states[mn].float() + win_ev_mtp[mn].unsqueeze(-1) * ym_m.float().unsqueeze(1)).to(torch.bfloat16)
        del u_ev_mtp, win_ev_mtp, m_eval_routes, mtp_eval_sh, mtp_routed

        # ---- 7c. MTP Hyper-Connection Mixer + Evaluate vs QuantTrunk & BF16 ----
        mfhc = "mtp.hyper_connection_mixer."
        mf_wd = reader.read_tensor(mfhc + "input_mix_weight_down.weight", dev, torch.float32)
        mf_wu = reader.read_tensor(mfhc + "input_mix_weight_up.weight", dev, torch.float32)
        mf_hn = reader.read_tensor(mfhc + "hc_norm.weight", dev, torch.float32)
        mf_wdf = mf_wd * (1.0 + mf_hn.unsqueeze(0))
        del mf_wd
        mf_bf16_d, mf_bf16_u = mf_wdf.to(torch.bfloat16), mf_wu.to(torch.bfloat16)
        mf_q8_d, mf_q8_u = sq.fake_quant(mf_wdf, "q8g64"), sq.fake_quant(mf_wu, "q8g64")
        del mf_wdf, mf_wu

        # Evaluate positions p = 1 .. T_eval - 2 (1805 tokens, where p>=1 pairs trunk[p-1] with ids[p] to predict ids[p+1])
        p_start, p_end = 1, T_eval - 1
        n_mtp_eval = p_end - p_start
        mtp_results = []

        print(f"\n=== MTP.LAYERS.0 HELD-OUT RESULTS (p={p_start}..{p_end-1}, N={n_mtp_eval} vs QuantTrunk 2_R3_FP64_Q6_Hotspots) ===", flush=True)
        print(f"{'MTP Variant':<20} | {'Top1%Trunk':>10} | {'Top3%Trunk':>10} | {'KLD(Tr||M)':>10} | {'p95_Tr':>8} | {'Top1%BF16':>10} | {'KLD(BF||M)':>10} | {'MTP_PPL':>7}", flush=True)
        print("-" * 108, flush=True)

        for mn in MTP_VARIANTS:
            hd = mf_bf16_d if mn == "M_BF16" else mf_q8_d
            hu = mf_bf16_u if mn == "M_BF16" else mf_q8_u
            u_m, _ = sq.hc_mix_down_up(mtp_eval_states[mn], hd, dummy_inj, hu, mf_hn, inject=False)

            kld_tr_parts, kld_bf_parts = [], []
            t1_tr, t3_tr, t1_bf, nll_m = 0.0, 0.0, 0.0, 0.0
            for s0 in range(p_start, p_end, TOK_CHUNK):
                s1 = min(s0 + TOK_CHUNK, p_end)
                lgt_tr = F.linear(trunk_u_q2[s0:s1], qlm_q8).float()
                lgt_bf = F.linear(f_bf16_u[s0:s1], lm_w).float()
                lgt_m = F.linear(u_m[s0:s1], lm_w if mn == "M_BF16" else qlm_q8).float()

                k_tr, a1, a3, nll_s = _eval_logits_pair(lgt_tr, lgt_m, target_ids=target_eval[s0:s1])
                k_bf, b1, _, _ = _eval_logits_pair(lgt_bf, lgt_m)
                kld_tr_parts.append(k_tr)
                kld_bf_parts.append(k_bf)
                t1_tr += a1
                t3_tr += a3
                t1_bf += b1
                nll_m += nll_s
                del lgt_tr, lgt_bf, lgt_m

            ktr_v = torch.cat(kld_tr_parts, dim=0)
            kbf_v = torch.cat(kld_bf_parts, dim=0)
            mk_tr, p95_tr = float(ktr_v.mean().item()), float(torch.quantile(ktr_v, 0.95).item())
            mk_bf = float(kbf_v.mean().item())
            top1_tr_pct = 100.0 * t1_tr / float(n_mtp_eval)
            top3_tr_pct = 100.0 * t3_tr / float(n_mtp_eval)
            top1_bf_pct = 100.0 * t1_bf / float(n_mtp_eval)
            ppl_m = math.exp(nll_m / float(n_mtp_eval))
            print(f"{mn:<20} | {top1_tr_pct:9.2f}% | {top3_tr_pct:9.2f}% | {mk_tr:10.6f} | {p95_tr:8.5f} | {top1_bf_pct:9.2f}% | {mk_bf:10.6f} | {ppl_m:7.3f}", flush=True)
            mtp_results.append({
                "variant": mn,
                "top1_vs_trunk_pct": round(top1_tr_pct, 2),
                "top3_vs_trunk_pct": round(top3_tr_pct, 2),
                "mean_kld_vs_trunk": round(mk_tr, 6),
                "p95_kld_vs_trunk": round(p95_tr, 6),
                "top1_vs_bf16_pct": round(top1_bf_pct, 2),
                "mean_kld_vs_bf16": round(mk_bf, 6),
                "mtp_ppl": round(ppl_m, 4),
            })
            del u_m

        print(f"MTP Hessian MSE Ratios (act/rtn): {mtp_hmse} | mtp_time={time.time()-t_mtp0:.1f}s", flush=True)
        del mtp_eval_states, trunk_u_q2, f_bf16_u, lm_w, qlm_q8, mf_bf16_d, mf_bf16_u, mf_q8_d, mf_q8_u, mf_hn, dummy_inj

    prefetch_pool.shutdown()
    reader.close()
    peak_vram_gb = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
    elapsed_s = time.time() - t0
    print(f"\nTotal Elapsed: {elapsed_s:.1f}s | Peak VRAM: {peak_vram_gb:.2f} GiB", flush=True)

    payload = {
        "model": model_dir,
        "n_layers": n_layers,
        "t_cal": T_cal,
        "cal_sha": cal_sha,
        "domains": dom_summary,
        "t_eval": T_eval,
        "bf16_ppl": round(bf16_ppl, 4),
        "tau": tau,
        "elapsed_s": round(elapsed_s, 1),
        "peak_vram_gib": round(peak_vram_gb, 2),
        "trunk_results": trunk_results,
        "trunk_hmse_summary": trunk_hmse_summary,
        "trunk_hmse_raw": hmse_totals,
        "mtp_eval_positions": [p_start, p_end],
        "mtp_results": mtp_results,
        "mtp_hmse": mtp_hmse,
        "trunk_layers_log": trunk_layers_log,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[SAVED] {out_json}", flush=True)
    if ckpt_dir:
        ckpt_file = os.path.join(ckpt_dir, "ckpt_latest.pt")
        if os.path.exists(ckpt_file):
            os.remove(ckpt_file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/nvme2/Thinking-Cap-Qwen-3.8-FN")
    ap.add_argument("--n-seqs", type=int, default=40)
    ap.add_argument("--seq-len", type=int, default=1536)
    ap.add_argument("--cal-mb", type=int, default=4)
    ap.add_argument("--expert-chunk", type=int, default=64)
    ap.add_argument("--tau", type=float, default=384.0)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--ckpt-dir", default="/mnt/nvme2/tmp-fp64-mtp-ckpt")
    ap.add_argument("--self-test-only", action="store_true")
    ap.add_argument(
        "--out",
        default="/home/cheblan/.gemini/antigravity/brain/2db43efb-0097-4015-bf58-dedb55fd8ccc/scratch/qwen_fp64_mtp_61k_results.json",
    )
    args = ap.parse_args()

    if args.self_test_only:
        dev = torch.device("cuda:0")
        _ = torch.zeros(1, device=dev)
        torch.cuda.synchronize()
        run_fp64_mtp_self_tests(dev)
        return

    run_fp64_mtp_sim(
        model_dir=args.model,
        out_json=args.out,
        n_seqs=args.n_seqs,
        seq_len=args.seq_len,
        cal_mb=args.cal_mb,
        expert_chunk=args.expert_chunk,
        tau=args.tau,
        max_layers=args.layers,
        ckpt_dir=args.ckpt_dir,
    )


if __name__ == "__main__":
    main()
