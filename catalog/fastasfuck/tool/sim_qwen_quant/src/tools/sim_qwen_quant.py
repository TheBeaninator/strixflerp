#!/usr/bin/env python3
"""
Layer-by-layer BF16-to-{Q4,Q5,Q6,Q8} Pareto simulator for Qwen 3.8 Flash Next (qwen4exp)
and fine-tunes (Swift 1.5, Thinking-Cap) on AMD gfx1100 (RX 7900 XTX, 24 GB VRAM).

- 0 bytes of .faf written to disk; OS page cache dropped via POSIX_FADV_DONTNEED.
- Pure PyTorch forward (NO Triton / fla / transformers):
    * 4-stream mHC with exact pre-quant (1 + hc_norm) folding into W_down / W_inj
    * PLE n-gram hashing + dilated causal conv1d + gated RMSNorm injection (layer 1)
    * Chunked WY Gated DeltaNet (36 layers, verified < 1e-5 vs sequential recurrence)
    * Full GQA attention with zero-centered q/k norm, partial RoPE (64/256), sigmoid gate (12 layers)
    * 512 routed + 1 shared MoE experts streamed in 64-expert chunks (peak VRAM < 10 GB)
- Reuses bit-identical asymmetric per-group fake quantizer matching formats/q{4,5,6,8}.cpp.

Usage:
  LD_PRELOAD=/usr/lib/libjemalloc.so ROCP_TOOLS_LIB="" HSA_TOOLS_LIB="" ROCP_HSA_INTERCEPT=0 \
    python3 -u tools/sim_qwen_quant.py --mode pareto \
      --models base=/mnt/nvme2/Qwen-3.8-FN swift15=/mnt/nvme2/Swift-1.5-Qwen-3.8-FN thinking=/mnt/nvme2/Thinking-Cap-Qwen-3.8-FN
"""

import os
os.environ.setdefault("ROCP_HSA_INTERCEPT", "0")
os.environ.setdefault("ROCP_TOOLS_LIB", "")
os.environ.setdefault("HSA_TOOLS_LIB", "")

import argparse
import concurrent.futures
import heapq
import json
import math
import struct
import subprocess
import time
import unicodedata
from collections import defaultdict
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Exact Qwen3.8 ByteLevel BPE Tokenizer (handles str & list merges)
# ---------------------------------------------------------------------------

def _bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(x) for x in cs]))


def _is_l(ch):
    return unicodedata.category(ch).startswith("L")


def _is_lm(ch):
    c = unicodedata.category(ch)
    return c.startswith("L") or c.startswith("M")


def _is_n(ch):
    return unicodedata.category(ch).startswith("N")


def _pretokenize_qwen(text):
    r"""Matches Qwen Split regex:
    (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
    """
    n = len(text)
    i = 0
    out = []
    c2 = ("'ll", "'ve", "'re")
    c1 = ("'s", "'d", "'m", "'t")

    while i < n:
        if text[i] == "'":
            low3 = text[i : i + 3].lower()
            low2 = text[i : i + 2].lower()
            if low3 in c2:
                out.append(text[i : i + 3])
                i += 3
                continue
            if low2 in c1:
                out.append(text[i : i + 2])
                i += 2
                continue

        ch = text[i]
        if _is_lm(ch) and not _is_n(ch) and (_is_l(ch) or unicodedata.category(ch).startswith("M")):
            j = i + 1
            while j < n and _is_lm(text[j]):
                j += 1
            out.append(text[i:j])
            i = j
            continue
        if ch not in ("\r", "\n") and not _is_l(ch) and not _is_n(ch):
            if i + 1 < n and _is_lm(text[i + 1]):
                j = i + 2
                while j < n and _is_lm(text[j]):
                    j += 1
                out.append(text[i:j])
                i = j
                continue

        if _is_n(ch):
            out.append(ch)
            i += 1
            continue

        is_sym = not ch.isspace() and not _is_lm(ch) and not _is_n(ch)
        if is_sym or (
            ch == " "
            and i + 1 < n
            and not text[i + 1].isspace()
            and not _is_lm(text[i + 1])
            and not _is_n(text[i + 1])
        ):
            j = i + (1 if ch == " " else 0)
            while j < n and not text[j].isspace() and not _is_lm(text[j]) and not _is_n(text[j]):
                j += 1
            while j < n and text[j] in ("\r", "\n"):
                j += 1
            out.append(text[i:j])
            i = j
            continue

        j = i
        found_nl = False
        while j < n and text[j].isspace():
            if text[j] in ("\r", "\n"):
                found_nl = True
                j += 1
                while j < n and text[j] in ("\r", "\n"):
                    j += 1
                break
            j += 1
        if found_nl:
            out.append(text[i:j])
            i = j
        elif j < n and j - i > 1:
            out.append(text[i : j - 1])
            i = j - 1
        else:
            out.append(text[i:j])
            i = j
    return out


class QwenTokenizer:
    def __init__(self, tok_json_path):
        with open(tok_json_path) as f:
            tj = json.load(f)
        self.vocab = tj["model"]["vocab"]
        self.id_to_tok = {v: k for k, v in self.vocab.items()}
        self.added = {}
        for at in tj.get("added_tokens", []):
            self.added[at["content"]] = at["id"]
            self.id_to_tok[at["id"]] = at["content"]
        self.added_sorted = sorted(self.added.keys(), key=len, reverse=True)

        self.bpe_ranks = {}
        for rank, m in enumerate(tj["model"]["merges"]):
            if isinstance(m, (list, tuple)):
                a, b = m[0], m[1]
            else:
                a, b = m.split(" ")
            self.bpe_ranks[(a, b)] = rank

        self.b2u = _bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}
        self.cache = {}

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = list(token)
        if len(word) <= 1:
            return word
        while True:
            best_pair = None
            best_rank = 1 << 60
            for idx in range(len(word) - 1):
                pair = (word[idx], word[idx + 1])
                r = self.bpe_ranks.get(pair)
                if r is not None and r < best_rank:
                    best_rank = r
                    best_pair = pair
            if best_pair is None:
                break
            first, second = best_pair
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break
        self.cache[token] = word
        return word

    def encode(self, text):
        text = unicodedata.normalize("NFC", text)
        ids = []
        i = 0
        n = len(text)
        plain_start = 0
        while i < n:
            matched = None
            for at in self.added_sorted:
                if text.startswith(at, i):
                    matched = at
                    break
            if matched is not None:
                if i > plain_start:
                    self._encode_plain(text[plain_start:i], ids)
                ids.append(self.added[matched])
                i += len(matched)
                plain_start = i
            else:
                i += 1
        if n > plain_start:
            self._encode_plain(text[plain_start:n], ids)
        return ids

    def _encode_plain(self, chunk, out_ids):
        for piece in _pretokenize_qwen(chunk):
            bl = "".join(self.b2u[b] for b in piece.encode("utf-8"))
            for sym in self._bpe(bl):
                out_ids.append(self.vocab[sym])

    def decode(self, ids):
        parts = []
        byte_buf = bytearray()
        for tid in ids:
            t = self.id_to_tok[tid]
            if t in self.added:
                if byte_buf:
                    parts.append(byte_buf.decode("utf-8", errors="replace"))
                    byte_buf.clear()
                parts.append(t)
            else:
                for ch in t:
                    byte_buf.append(self.u2b[ch])
        if byte_buf:
            parts.append(byte_buf.decode("utf-8", errors="replace"))
        return "".join(parts)


# ---------------------------------------------------------------------------
# 2. Streaming Safetensors Reader (supports slice pread + POSIX_FADV_DONTNEED)
# ---------------------------------------------------------------------------

class SafetensorsReader:
    def __init__(self, model_dir):
        if ":" in model_dir and not os.path.exists(model_dir):
            raise RuntimeError(f"Remote ssh streaming not needed here; missing local dir '{model_dir}'")
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            self.wm = json.load(f)["weight_map"]
        self.fds = {}
        self.headers = {}
        self.data_starts = {}

    def _open_shard(self, shard_name):
        if shard_name not in self.fds:
            p = os.path.join(self.model_dir, shard_name)
            fd = os.open(p, os.O_RDONLY)
            hl = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
            hdr = json.loads(os.pread(fd, hl, 8))
            self.fds[shard_name] = fd
            self.headers[shard_name] = hdr
            self.data_starts[shard_name] = 8 + hl

    def info(self, name):
        shard = self.wm[name]
        self._open_shard(shard)
        return self.headers[shard][name], shard

    def read_tensor(self, name, device, dtype=torch.bfloat16):
        meta, shard = self.info(name)
        s, e = meta["data_offsets"]
        off = self.data_starts[shard] + s
        length = e - s
        fd = self.fds[shard]
        raw = bytearray(os.pread(fd, length, off))
        os.posix_fadvise(fd, off, length, os.POSIX_FADV_DONTNEED)
        dt = meta["dtype"]
        sh = meta["shape"]
        if dt == "BF16":
            t = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(sh)
            return t.to(device=device, dtype=dtype)
        if dt == "F32":
            t = torch.frombuffer(raw, dtype=torch.float32).reshape(sh)
            return t.to(device=device, dtype=dtype)
        if dt == "I64":
            return torch.frombuffer(raw, dtype=torch.int64).reshape(sh).clone()
        raise RuntimeError(f"Unsupported dtype {dt} for {name}")

    def read_slice_dim0(self, name, start_idx, count, device, dtype=torch.bfloat16):
        """Reads [start_idx : start_idx + count, ...] of a tensor via direct pread without reading the whole tensor."""
        meta, shard = self.info(name)
        sh = meta["shape"]
        dt = meta["dtype"]
        elem_bytes = 2 if dt == "BF16" else 4
        row_elems = 1
        for d in sh[1:]:
            row_elems *= d
        byte_off = self.data_starts[shard] + meta["data_offsets"][0] + start_idx * row_elems * elem_bytes
        byte_len = count * row_elems * elem_bytes
        fd = self.fds[shard]
        raw = bytearray(os.pread(fd, byte_len, byte_off))
        os.posix_fadvise(fd, byte_off, byte_len, os.POSIX_FADV_DONTNEED)
        tdt = torch.bfloat16 if dt == "BF16" else torch.float32
        t = torch.frombuffer(raw, dtype=tdt).reshape([count] + sh[1:])
        return t.to(device=device, dtype=dtype)

    def gather_ple_rows(self, ple_prefix, unique_rows, device):
        """Parallel pread of unique rows across the 128 ple_embedding.ngram_embedding.shard_* BF16 tensors."""
        sh0_meta, _ = self.info(ple_prefix + "ple_embedding.ngram_embedding.shard_0.weight")
        rows_per_shard, row_dim = sh0_meta["shape"]
        # Ensure all 128 shards are opened before worker threads pread concurrently
        for s in range(128):
            self._open_shard(self.wm[f"{ple_prefix}ple_embedding.ngram_embedding.shard_{s}.weight"])

        out = torch.empty((len(unique_rows), row_dim), dtype=torch.bfloat16)
        tasks = []
        for out_i, r in enumerate(unique_rows):
            s = r // rows_per_shard
            local_r = r % rows_per_shard
            tname = f"{ple_prefix}ple_embedding.ngram_embedding.shard_{s}.weight"
            shard = self.wm[tname]
            meta = self.headers[shard][tname]
            off = self.data_starts[shard] + meta["data_offsets"][0] + local_r * row_dim * 2
            tasks.append((out_i, self.fds[shard], off))

        rb = row_dim * 2
        def _fetch(item):
            out_i, fd, off = item
            buf = os.pread(fd, rb, off)
            out[out_i] = torch.frombuffer(bytearray(buf), dtype=torch.bfloat16)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(_fetch, tasks))
        return out.to(device=device)

    def close(self):
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()


# ---------------------------------------------------------------------------
# 3. Bit-Identical Fake Quantizer (reused from sim_ling_quant.py / formats/q{4..8})
# ---------------------------------------------------------------------------

def fake_quant(w, fmt):
    """Works on 2D [N, K] or 3D [E, N, K] weight tensors; returns BF16 in-place-efficiently."""
    if fmt is None or fmt == "bf16":
        return w.to(torch.bfloat16)
    bits = int(fmt[1])
    G = int(fmt[3:])
    orig_shape = w.shape
    K = orig_shape[-1]
    if K % G != 0:
        G_eff = math.gcd(K, G)
        assert G_eff >= 32 and K % G_eff == 0, f"K={K} not divisible by G={G}"
        G = G_eff
    wf = w.float().clone() if w.dtype == torch.float32 else w.float()
    wf = wf.reshape(-1, K // G, G)
    lo = wf.amin(dim=-1, keepdim=True)
    mx = wf.amax(dim=-1, keepdim=True)
    levels = float((1 << bits) - 1)
    sc = ((mx - lo) / levels).to(torch.bfloat16).float()
    mn = lo.to(torch.bfloat16).float()
    del lo, mx
    safe_sc = torch.where(sc > 0.0, sc, torch.ones_like(sc))
    wf.sub_(mn).div_(safe_sc).round_().clamp_(0.0, levels)
    wf.masked_fill_(sc <= 0.0, 0.0)
    wf.mul_(sc).add_(mn)
    return wf.reshape(orig_shape).to(torch.bfloat16)


def fake_quant_rows(w_bf16, fmt, row_chunk=32768):
    """Row-chunked fake_quant for huge 2D weights like lm_head [248320, 2560]."""
    if fmt is None or fmt == "bf16":
        return w_bf16
    N = w_bf16.shape[0]
    out = torch.empty_like(w_bf16, dtype=torch.bfloat16)
    for r0 in range(0, N, row_chunk):
        r1 = min(r0 + row_chunk, N)
        out[r0:r1] = fake_quant(w_bf16[r0:r1], fmt)
    return out


# ---------------------------------------------------------------------------
# 4. Exact Qwen4Exp Architecture Primitives (Pure PyTorch, NO Triton!)
# ---------------------------------------------------------------------------

def rmsnorm_zc(x, w, eps=1e-6):
    """Zero-centered RMSNorm: y = cast(norm(x) * (1 + w))."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * (1.0 + w.float())).to(x.dtype)


def rmsnorm_raw(x, w, eps=1e-6):
    """Raw RMSNorm (GDN o_norm): y = cast(norm(x)) * w."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * w.float()).to(x.dtype)


def hc_mix_down_up(X, w_down_f, w_inj_f, w_up, hc_norm, inject=True, eps=1e-6):
    """Exact kernels/hc.hpp semantics on X [T, 4, 2560]:
       inv = rsqrt(mean(X^2) + eps), n0 = X * inv, n = n0 * (1 + hc_norm),
       h = silu((n0 @ W'_down^T) / 4), w_in = 2*sigmoid((n0 @ W'_inj^T) / 4),
       u = sum_s sigmoid(h @ W_up^T)[s] * n[s] / 4.
    """
    T, H, d = X.shape
    xf = X.float()
    inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)  # [T, 4, 1]
    n0 = (xf * inv).reshape(T, H * d)
    n = n0 * (1.0 + hc_norm.float())

    a_down = F.linear(n0, w_down_f.float()) / float(H)
    h = F.silu(a_down)  # [T, 320]
    w_in = (2.0 * torch.sigmoid(F.linear(n0, w_inj_f.float()) / float(H))) if inject else None

    alpha = torch.sigmoid(F.linear(h, w_up.float())).view(T, H, d)
    u = ((alpha * n.view(T, H, d)).sum(dim=1) / float(H)).to(X.dtype)
    return u, w_in


def ple_forward(X, ple_e, w_kv, conv_w, nk, nq, nc, eps=1e-6):
    """Exact kernels/ple.hip semantics on X [T, 4, 2560], ple_e [T, 2560]."""
    T, H, d = X.shape
    n4 = H * d
    kv = F.linear(ple_e, w_kv)  # [T, 5*d]
    key = kv[:, :n4].view(T, H, d).float()
    val = kv[:, n4:].float()  # [T, d]

    xf = X.float()
    inv_k = torch.rsqrt(key.pow(2).mean(dim=-1) + eps)
    inv_q = torch.rsqrt(xf.pow(2).mean(dim=-1) + eps)
    w_kq = (1.0 + nk.float()).view(H, d) * (1.0 + nq.float()).view(H, d)
    kx = (key * xf * w_kq.unsqueeze(0)).sum(dim=-1)  # [T, H]
    g = inv_k * inv_q * kx / math.sqrt(float(d))
    g = torch.sign(g) * torch.sqrt(torch.clamp(g.abs(), min=1e-6))
    sg = torch.sigmoid(g)  # [T, H]

    inv_v = torch.rsqrt((sg * sg).unsqueeze(-1) * val.pow(2).mean(dim=-1, keepdim=True).unsqueeze(1) + eps)
    sv = sg.unsqueeze(-1) * val.unsqueeze(1)  # [T, H, d]
    vn = (sv * inv_v * (1.0 + nc.float()).view(1, H, d)).to(torch.bfloat16)

    # Dilated causal depthwise conv1d: K=4, dilation=3, left zero pad = 9
    vn_t = vn.reshape(T, n4).transpose(0, 1).unsqueeze(0).float()  # [1, n4, T]
    vn_pad = F.pad(vn_t, (9, 0))
    cv = F.conv1d(vn_pad, conv_w.reshape(n4, 1, -1).float(), dilation=3, groups=n4).squeeze(0).transpose(0, 1).view(T, H, d)
    cv = F.silu(cv)
    return (xf + sv + cv).to(X.dtype)


def chunked_delta_rule_gdn(q, k, v, beta, g, chunk_size=64):
    """Pure-PyTorch WY chunked Gated Delta Rule (NO Triton).
    Shapes: q, k, v: [T, 48, 128] FP32 (q already scaled by 128^-0.5 & L2-normed; k L2-normed),
            beta, g: [T, 48] FP32.
    """
    T, Hv, D = q.shape
    pad = (chunk_size - (T % chunk_size)) % chunk_size
    if pad > 0:
        q = F.pad(q, (0, 0, 0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, 0, 0, pad))
        beta = F.pad(beta, (0, 0, 0, pad))
        g = F.pad(g, (0, 0, 0, pad))
    T_pad = q.shape[0]
    nc = T_pad // chunk_size
    C = chunk_size

    # Reshape into chunks: [nc, Hv, C, D]
    qc = q.view(nc, C, Hv, D).permute(0, 2, 1, 3)
    kc = k.view(nc, C, Hv, D).permute(0, 2, 1, 3)
    vc = v.view(nc, C, Hv, D).permute(0, 2, 1, 3)
    bc = beta.view(nc, C, Hv).permute(0, 2, 1)  # [nc, Hv, C]
    gc = g.view(nc, C, Hv).permute(0, 2, 1)     # [nc, Hv, C]

    G = gc.cumsum(dim=-1)  # [nc, Hv, C]
    tril_strict = torch.tril(torch.ones((C, C), device=q.device, dtype=torch.bool), diagonal=-1)
    tril_incl = torch.tril(torch.ones((C, C), device=q.device, dtype=torch.bool), diagonal=0)

    # Pairwise exp(G_t - G_s) masked to lower triangle (prevents exp overflow for upper triangle)
    diff_strict = (G.unsqueeze(-1) - G.unsqueeze(-2)).masked_fill(~tril_strict, float("-inf"))
    decay_strict = torch.exp(diff_strict)
    diff_incl = (G.unsqueeze(-1) - G.unsqueeze(-2)).masked_fill(~tril_incl, float("-inf"))
    decay_incl = torch.exp(diff_incl)

    kk = torch.matmul(kc, kc.transpose(-1, -2)) * decay_strict  # [nc, Hv, C, C]
    A = bc.unsqueeze(-1) * kk
    eye = torch.eye(C, device=q.device, dtype=torch.float32)
    M_inv_mat = eye + A  # unit lower triangular

    k_beta = kc * bc.unsqueeze(-1)  # [nc, Hv, C, D]
    v_beta = vc * bc.unsqueeze(-1)  # [nc, Hv, C, D]

    # T_mat = (I + A)^(-1) -> then per chunk we only need M_inv_mat solve against (v_beta - K_decay @ S)!
    # Even faster: invert M_inv_mat once across all chunks batched!
    T_inv = torch.linalg.solve_triangular(M_inv_mat, eye.expand(nc, Hv, C, C), upper=False, unitriangular=True)

    # Inter-chunk recurrence across nc (e.g. 16..24 steps for T=1024..1536 — super fast!)
    qk = torch.matmul(qc, kc.transpose(-1, -2)) * decay_incl
    S = torch.zeros((Hv, D, D), device=q.device, dtype=torch.float32)
    outs = []
    for c_i in range(nc):
        Gc = G[c_i]                  # [Hv, C]
        exp_G = torch.exp(Gc)        # [Hv, C]
        exp_last = torch.exp(Gc[:, -1:])  # [Hv, 1]
        exp_rem = torch.exp(Gc[:, -1:] - Gc)  # [Hv, C]

        kS = torch.matmul(k_beta[c_i], S)  # [Hv, C, D]
        rhs = v_beta[c_i] - exp_G.unsqueeze(-1) * kS
        U = torch.matmul(T_inv[c_i], rhs)  # [Hv, C, D]

        o_inter = exp_G.unsqueeze(-1) * torch.matmul(qc[c_i], S)
        o_intra = torch.matmul(qk[c_i], U)
        outs.append((o_inter + o_intra).permute(1, 0, 2))  # [C, Hv, D]

        k_dec = kc[c_i] * exp_rem.unsqueeze(-1)  # [Hv, C, D]
        S = exp_last.unsqueeze(-1) * S + torch.matmul(k_dec.transpose(-1, -2), U)

    out = torch.cat(outs, dim=0)[:T]
    return out


def recurrent_delta_rule_gdn(q, k, v, beta, g):
    """Sequential reference GDN recurrence for self-test verification."""
    T, Hv, D = q.shape
    S = torch.zeros((Hv, D, D), device=q.device, dtype=torch.float32)
    out = torch.empty((T, Hv, D), device=q.device, dtype=torch.float32)
    for t in range(T):
        S = S * torch.exp(g[t]).unsqueeze(-1).unsqueeze(-1)
        kt = k[t]
        v_hat = torch.matmul(S.transpose(-1, -2), kt.unsqueeze(-1)).squeeze(-1)  # S^T k
        delta = beta[t].unsqueeze(-1) * (v[t] - v_hat)
        S = S + kt.unsqueeze(-1) * delta.unsqueeze(-2)
        out[t] = torch.matmul(S.transpose(-1, -2), q[t].unsqueeze(-1)).squeeze(-1)
    return out


def gdn_forward(x_u, w_in, w_out, conv_w, A_log, dt_bias, gdn_norm):
    """Exact GDN mixer on x_u [T, 2560]."""
    T = x_u.shape[0]
    # w_in is [16480, 2560]: [qkv(10240) | z(6144) | b(48) | a(48)]
    proj = F.linear(x_u, w_in)
    raw_qkv = proj[:, :10240]
    z = proj[:, 10240:16384].view(T, 48, 128)
    b = proj[:, 16384:16432].float()
    a = proj[:, 16432:16480].float()

    # Causal depthwise conv1d(K=4) + silu
    zt = raw_qkv.transpose(0, 1).unsqueeze(0).float()
    cv = F.conv1d(F.pad(zt, (3, 0)), conv_w.squeeze(1).unsqueeze(1).float(), groups=10240)
    qkv = F.silu(cv.squeeze(0).transpose(0, 1))

    q = qkv[:, :2048].view(T, 16, 128).float()
    k = qkv[:, 2048:4096].view(T, 16, 128).float()
    v = qkv[:, 4096:10240].view(T, 48, 128).float()

    q = F.normalize(q, p=2, dim=-1, eps=1e-6) * (128.0 ** -0.5)
    k = F.normalize(k, p=2, dim=-1, eps=1e-6)
    q = q.repeat_interleave(3, dim=1)
    k = k.repeat_interleave(3, dim=1)

    beta = torch.sigmoid(b)
    g = -torch.exp(A_log.float()) * F.softplus(a + dt_bias.float())

    o = chunked_delta_rule_gdn(q, k, v, beta, g, chunk_size=64)
    o_norm = rmsnorm_raw(o.to(x_u.dtype), gdn_norm, 1e-6)
    o_gated = (o_norm.float() * torch.sigmoid(z.float())).to(x_u.dtype).reshape(T, 6144)
    return F.linear(o_gated, w_out)


def rope_rotate_half(x, cos, sin):
    """Partial rotate_half RoPE on first rot=64 dims of x [1, H, T, 256]; cos/sin [1, 1, T, 32]."""
    rot = cos.shape[-1] * 2
    xr, xp = x[..., :rot], x[..., rot:]
    x1, x2 = xr[..., : rot // 2], xr[..., rot // 2 :]
    c = torch.cat([cos, cos], dim=-1)
    s = torch.cat([sin, sin], dim=-1)
    rot_half = torch.cat([-x2, x1], dim=-1)
    xr_out = xr * c + rot_half * s
    return torch.cat([xr_out, xp], dim=-1)


def full_attn_forward(x_u, w_qkv, w_o, q_norm, k_norm, cos, sin):
    """Exact GQA full attention (T <= 2051 -> dense SDPA == QSA) on x_u [T, 2560]."""
    T = x_u.shape[0]
    # w_qkv rows: [q_proj(12288) | k_proj(512) | v_proj(512)]
    qkv = F.linear(x_u, w_qkv)
    q_gate = qkv[:, :12288].view(T, 24, 512)
    q, gate = q_gate[:, :, :256], q_gate[:, :, 256:]
    k = qkv[:, 12288:12800].view(T, 2, 256)
    v = qkv[:, 12800:13312].view(T, 2, 256)

    q = rmsnorm_zc(q, q_norm, 1e-6).transpose(0, 1).unsqueeze(0)  # [1, 24, T, 256]
    k = rmsnorm_zc(k, k_norm, 1e-6).transpose(0, 1).unsqueeze(0)  # [1, 2, T, 256]
    v = v.transpose(0, 1).unsqueeze(0)                            # [1, 2, T, 256]

    q = rope_rotate_half(q, cos, sin)
    k = rope_rotate_half(k, cos, sin)
    k = k.repeat_interleave(12, dim=1)
    v = v.repeat_interleave(12, dim=1)

    attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=256.0 ** -0.5)
    attn = attn.squeeze(0).transpose(0, 1)  # [T, 24, 256]
    out = (attn.float() * torch.sigmoid(gate.float())).to(x_u.dtype).reshape(T, 6144)
    return F.linear(out, w_o)


def moe_route_qwen(x_u, w_router_all):
    """w_router_all [513, 2560]: 512 routed + 1 shared gate."""
    logits = F.linear(x_u.float(), w_router_all.float())
    r_logits = logits[:, :512]
    sh_logit = logits[:, 512]
    _, top_idx = torch.topk(r_logits, k=10, dim=-1)
    top_l = torch.gather(r_logits, 1, top_idx)
    top_w = F.softmax(top_l, dim=-1)
    sh_coef = torch.sigmoid(sh_logit)
    return top_idx, top_w, sh_coef


# ---------------------------------------------------------------------------
# 5. Recipes & Size/Speed Model (Qwen 3.8 Flash Next, 48 layers)
# ---------------------------------------------------------------------------

CLASSES = ["exp_gu", "exp_down", "gdn_in", "gdn_out", "attn_qkv", "attn_o", "sh_gu", "sh_down", "hc", "ple_kv", "lm_head"]

RECIPES = [
    ("1_Current_Prod",   "exp_gu=q4g128,exp_down=q4g128,gdn_in=q4g128,gdn_out=q6g64,attn_qkv=q6g64,attn_o=q6g64,sh_gu=q6g64,sh_down=q6g64,ple_kv=q6g64,hc=q8g64,lm_head=q8g64"),
    ("2_Fix_GDN_Shared", "exp_gu=q4g128,exp_down=q4g128,gdn_in=q6g64,gdn_out=q6g64,attn_qkv=q6g64,attn_o=q6g64,sh_gu=q6g64,sh_down=q8g64,ple_kv=q6g64,hc=q8g64,lm_head=q8g64"),
    ("3_Asym_Experts",   "exp_gu=q4g64,exp_down=q5g128,gdn_in=q6g64,gdn_out=q6g64,attn_qkv=q8g64,attn_o=q8g64,sh_gu=q8g64,sh_down=q8g64,ple_kv=q8g64,hc=q8g64,lm_head=q8g64"),
    ("4_Gold_Q5",        "exp_gu=q5g128,exp_down=q5g128,gdn_in=q6g64,gdn_out=q6g64,attn_qkv=q8g64,attn_o=q8g64,sh_gu=q8g64,sh_down=q8g64,ple_kv=q8g64,hc=q8g64,lm_head=q8g64"),
    ("5_Max_Q5g64",      "exp_gu=q5g64,exp_down=q5g64,gdn_in=q8g64,gdn_out=q8g64,attn_qkv=q8g64,attn_o=q8g64,sh_gu=q8g64,sh_down=q8g64,ple_kv=q8g64,hc=q8g64,lm_head=q8g64"),
]

SNIPER_RECIPES = [
    ("1_Current_Prod",    "exp_gu=q4g128,exp_down=q4g128,gdn_in=q4g128,gdn_out=q6g64,attn_qkv=q6g64,attn_o=q6g64,sh_gu=q6g64,sh_down=q6g64,ple_kv=q6g64,hc=q8g64,lm_head=q8g64"),
    ("P1_Prod_GdnQ5",     "exp_gu=q4g128,exp_down=q4g128,gdn_in=q5g128,gdn_out=q6g64,attn_qkv=q6g64,attn_o=q6g64,sh_gu=q6g64,sh_down=q6g64,ple_kv=q6g64,hc=q8g64,lm_head=q8g64"),
    ("P2_Prod_GdnQ5_Snp", "exp_gu=q4g128,exp_down=q4g128,gdn_in=q5g128,gdn_out=q6g64,attn_qkv=q6g64,attn_o=q8g64,sh_gu=q6g64,sh_down=q8g64,ple_kv=q6g64,hc=q8g64,lm_head=q8g64"),
    ("P3_Prod_GdnQ5_Out", "exp_gu=q4g128,exp_down=q4g128,gdn_in=q5g128,gdn_out=q8g64,attn_qkv=q6g64,attn_o=q8g64,sh_gu=q6g64,sh_down=q8g64,ple_kv=q6g64,hc=q8g64,lm_head=q8g64"),
    ("0_Shawn_U_gdn128",  "exp_gu=q4g128,exp_down=q4g128,gdn_in=q4g128,gdn_out=q8g64,attn_qkv=q8g64,attn_o=q8g64,sh_gu=q8g64,sh_down=q8g64,ple_kv=q8g64,hc_down=q8g64,hc_up=q8g64,lm_head=q8g64"),
    ("S1_Sniper_Turbo",   "exp_gu=q4g128,exp_down=q4g128,gdn_in=q4g128,gdn_out=q8g64,attn_qkv=q6g64,attn_o=q8g64,sh_gu=q6g64,sh_down=q8g64,ple_kv=q8g64,hc=q8g64,lm_head=q8g64"),
    ("S2_Sniper_Bal",     "exp_gu=q4g128,exp_down=q4g128,gdn_in=q4g64,gdn_out=q8g64,attn_qkv=q8g64,attn_o=q8g64,sh_gu=q6g64,sh_down=q8g64,ple_kv=q8g64,hc=q8g64,lm_head=q8g64"),
    ("S3_Shawn_GdnQ5",    "exp_gu=q4g128,exp_down=q4g128,gdn_in=q5g128,gdn_out=q8g64,attn_qkv=q8g64,attn_o=q8g64,sh_gu=q8g64,sh_down=q8g64,ple_kv=q8g64,hc=q8g64,lm_head=q8g64"),
    ("S3b_Shawn_GdnQ6",   "exp_gu=q4g128,exp_down=q4g128,gdn_in=q6g64,gdn_out=q8g64,attn_qkv=q8g64,attn_o=q8g64,sh_gu=q8g64,sh_down=q8g64,ple_kv=q8g64,hc=q8g64,lm_head=q8g64"),
    ("S4a_Sniper_HcUpQ4", "exp_gu=q4g128,exp_down=q4g128,gdn_in=q4g64,gdn_out=q8g64,attn_qkv=q8g64,attn_o=q8g64,sh_gu=q6g64,sh_down=q8g64,ple_kv=q8g64,hc_down=q8g64,hc_up=q4g64,lm_head=q8g64"),
    ("S5_Shawn_ExpDnQ5",  "exp_gu=q4g128,exp_down=q5g128,gdn_in=q5g128,gdn_out=q8g64,attn_qkv=q8g64,attn_o=q8g64,sh_gu=q8g64,sh_down=q8g64,ple_kv=q8g64,hc=q8g64,lm_head=q8g64"),
    ("S6_Asym_GdnOutQ8",  "exp_gu=q4g64,exp_down=q5g128,gdn_in=q6g64,gdn_out=q8g64,attn_qkv=q8g64,attn_o=q8g64,sh_gu=q8g64,sh_down=q8g64,ple_kv=q8g64,hc=q8g64,lm_head=q8g64"),
    ("S7_Gold_GdnOutQ8",  "exp_gu=q5g128,exp_down=q5g128,gdn_in=q6g64,gdn_out=q8g64,attn_qkv=q8g64,attn_o=q8g64,sh_gu=q8g64,sh_down=q8g64,ple_kv=q8g64,hc=q8g64,lm_head=q8g64"),
]

BARTER_RECIPES = [
    # Reference anchors
    ("1_Current_Prod",      "exp_gu=q4g128,exp_down=q4g128,gdn_in=q4g128,gdn_out=q6g64,attn_qkv=q6g64,attn_o=q6g64,sh_gu=q6g64,sh_down=q6g64,ple_kv=q6g64,hc=q8g64,lm_head=q8g64"),
    ("P3_Prod_GdnQ5_Out",   "exp_gu=q4g128,exp_down=q4g128,gdn_in=q5g128,gdn_out=q8g64,attn_qkv=q6g64,attn_o=q8g64,sh_gu=q6g64,sh_down=q8g64,ple_kv=q6g64,hc=q8g64,lm_head=q8g64"),
    # Group B: Pure lm_head Q6 barter (73.2 - 79.8 t/s MTP)
    ("B0_Prod_LM6g128",     "exp_gu=q4g128,exp_down=q4g128,gdn_in=q4g128,gdn_out=q6g64,attn_qkv=q6g64,attn_o=q6g64,sh_gu=q6g64,sh_down=q6g64,ple_kv=q6g64,hc=q8g64,lm_head=q6g128"),
    ("B1_Swap_LM6g64_P1",   "exp_gu=q4g128,exp_down=q4g128,gdn_in=q5g128,gdn_out=q6g64,attn_qkv=q6g64,attn_o=q6g64,sh_gu=q6g64,sh_down=q6g64,ple_kv=q6g64,hc=q8g64,lm_head=q6g64"),
    ("B1b_Swap_LM6g128_P1", "exp_gu=q4g128,exp_down=q4g128,gdn_in=q5g128,gdn_out=q6g64,attn_qkv=q6g64,attn_o=q6g64,sh_gu=q6g64,sh_down=q6g64,ple_kv=q6g64,hc=q8g64,lm_head=q6g128"),
    ("B2_Swap_LM6g128_P2",  "exp_gu=q4g128,exp_down=q4g128,gdn_in=q5g128,gdn_out=q6g64,attn_qkv=q6g64,attn_o=q8g64,sh_gu=q6g64,sh_down=q8g64,ple_kv=q6g64,hc=q8g64,lm_head=q6g128"),
    ("B3_Swap_LM6g128_P3",  "exp_gu=q4g128,exp_down=q4g128,gdn_in=q5g128,gdn_out=q8g64,attn_qkv=q6g64,attn_o=q8g64,sh_gu=q6g64,sh_down=q8g64,ple_kv=q6g64,hc=q8g64,lm_head=q6g128"),
    ("B3b_Swap_LM6g64_P3",  "exp_gu=q4g128,exp_down=q4g128,gdn_in=q5g128,gdn_out=q8g64,attn_qkv=q6g64,attn_o=q8g64,sh_gu=q6g64,sh_down=q8g64,ple_kv=q6g64,hc=q8g64,lm_head=q6g64"),
    # Group C: Dense diet + GDN group-size / bit sweep + hc_down probe
    ("C1_Diet_P3_LM6",      "exp_gu=q4g128,exp_down=q4g128,gdn_in=q5g128,gdn_out=q8g64,attn_qkv=q5g128,attn_o=q8g64,sh_gu=q5g128,sh_down=q8g64,ple_kv=q5g128,hc_down=q8g128,hc_up=q8g64,lm_head=q6g128"),
    ("C2_GdnQ4g32_P3_LM6",  "exp_gu=q4g128,exp_down=q4g128,gdn_in=q4g32,gdn_out=q8g64,attn_qkv=q6g64,attn_o=q8g64,sh_gu=q6g64,sh_down=q8g64,ple_kv=q6g64,hc=q8g64,lm_head=q6g128"),
    ("C3_GdnQ5g64_P3_LM6",  "exp_gu=q4g128,exp_down=q4g128,gdn_in=q5g64,gdn_out=q8g64,attn_qkv=q6g64,attn_o=q8g64,sh_gu=q6g64,sh_down=q8g64,ple_kv=q6g64,hc=q8g64,lm_head=q6g128"),
    ("C4_GdnQ6g128_P3_LM6", "exp_gu=q4g128,exp_down=q4g128,gdn_in=q6g128,gdn_out=q8g64,attn_qkv=q6g64,attn_o=q8g64,sh_gu=q6g64,sh_down=q8g64,ple_kv=q6g64,hc=q8g64,lm_head=q6g128"),
    ("C5_HcDownQ4g32_P3",   "exp_gu=q4g128,exp_down=q4g128,gdn_in=q5g128,gdn_out=q8g64,attn_qkv=q6g64,attn_o=q8g64,sh_gu=q6g64,sh_down=q8g64,ple_kv=q6g64,hc_down=q4g32,hc_up=q8g64,lm_head=q6g128"),
    # Group D: Expert upgrades funded by lm_head=q6g128 barter (69.4 - 74.5 t/s MTP)
    ("D1_Swap_ExpG64_P2",   "exp_gu=q4g64,exp_down=q4g64,gdn_in=q5g128,gdn_out=q6g64,attn_qkv=q6g64,attn_o=q8g64,sh_gu=q6g64,sh_down=q8g64,ple_kv=q6g64,hc=q8g64,lm_head=q6g128"),
    ("D2_Swap_ExpDnQ4g32",  "exp_gu=q4g128,exp_down=q4g32,gdn_in=q5g128,gdn_out=q8g64,attn_qkv=q6g64,attn_o=q8g64,sh_gu=q6g64,sh_down=q8g64,ple_kv=q6g64,hc=q8g64,lm_head=q6g128"),
    ("D3_Swap_ExpDnQ5_P2",  "exp_gu=q4g128,exp_down=q5g128,gdn_in=q5g128,gdn_out=q6g64,attn_qkv=q6g64,attn_o=q8g64,sh_gu=q6g64,sh_down=q8g64,ple_kv=q6g64,hc=q8g64,lm_head=q6g128"),
    ("D4_Swap_ExpDnQ5_P3",  "exp_gu=q4g128,exp_down=q5g128,gdn_in=q5g128,gdn_out=q8g64,attn_qkv=q6g64,attn_o=q8g64,sh_gu=q6g64,sh_down=q8g64,ple_kv=q6g64,hc=q8g64,lm_head=q6g128"),
    ("D5_Diet_ExpDnQ5_P3",  "exp_gu=q4g128,exp_down=q5g128,gdn_in=q5g128,gdn_out=q8g64,attn_qkv=q5g128,attn_o=q8g64,sh_gu=q5g128,sh_down=q8g64,ple_kv=q5g128,hc_down=q8g128,hc_up=q8g64,lm_head=q6g128"),
    ("D6_Swap_Asym_P3",     "exp_gu=q4g64,exp_down=q5g128,gdn_in=q5g128,gdn_out=q8g64,attn_qkv=q6g64,attn_o=q8g64,sh_gu=q6g64,sh_down=q8g64,ple_kv=q6g64,hc=q8g64,lm_head=q6g128"),
    ("D7_Diet_Asym_GdnQ6",  "exp_gu=q4g64,exp_down=q5g128,gdn_in=q6g128,gdn_out=q8g64,attn_qkv=q5g128,attn_o=q8g64,sh_gu=q5g128,sh_down=q8g64,ple_kv=q5g128,hc_down=q8g128,hc_up=q8g64,lm_head=q6g128"),
    ("D8_Swap_Gold_Fast",   "exp_gu=q5g128,exp_down=q5g128,gdn_in=q5g128,gdn_out=q8g64,attn_qkv=q6g64,attn_o=q8g64,sh_gu=q6g64,sh_down=q8g64,ple_kv=q6g64,hc=q8g64,lm_head=q6g128"),
    # Group E: Extreme lm_head=q4g32 probe
    ("E1_Extreme_LM4g32",   "exp_gu=q4g128,exp_down=q4g128,gdn_in=q5g128,gdn_out=q8g64,attn_qkv=q6g64,attn_o=q8g64,sh_gu=q6g64,sh_down=q8g64,ple_kv=q6g64,hc=q8g64,lm_head=q4g32"),
]

SOLVER_VERIFY_RECIPES = [
    ("1_Current_Prod",  "exp_gu=q4g128,exp_down=q4g128,gdn_in=q4g128,gdn_out=q6g64,attn_qkv=q6g64,attn_o=q6g64,sh_gu=q6g64,sh_down=q6g64,ple_kv=q6g64,hc=q8g64,lm_head=q8g64"),
    ("S3b_Shawn_GdnQ6", "exp_gu=q4g128,exp_down=q4g128,gdn_in=q6g64,gdn_out=q8g64,attn_qkv=q8g64,attn_o=q8g64,sh_gu=q8g64,sh_down=q8g64,ple_kv=q8g64,hc=q8g64,lm_head=q8g64"),
    ("B10_A11",         "exp_gu=q4g128,exp_down=q4g128,gdn_in=q6g128,gdn_out=q8g128,attn_qkv=q5g64,attn_o=q6g64,sh_gu=q6g64,sh_down=q6g64,hc_down=q8g128,hc_up=q8g64,ple_kv=q8g128,lm_head=q8g64"),
    ("B10_A14",         "exp_gu=q4g128,exp_down=q4g128,gdn_in=q6g64,gdn_out=q8g64,attn_qkv=q6g32,attn_o=q8g128,sh_gu=q8g128,sh_down=q8g64,hc_down=q8g128,hc_up=q8g64,ple_kv=q8g64,lm_head=q8g64"),
    ("B10_A7",          "exp_gu=q4g128,exp_down=q4g128,gdn_in=q5g64,gdn_out=q6g64,attn_qkv=q5g128,attn_o=q6g128,sh_gu=q5g64,sh_down=q5g64,hc_down=q8g128,hc_up=q8g64,ple_kv=q6g64,lm_head=q8g64"),
    ("B4_A18",          "exp_gu=q4g128,exp_down=q4g128,gdn_in=q6g128,gdn_out=q8g128,attn_qkv=q5g128,attn_o=q6g128,sh_gu=q5g64,sh_down=q6g128,hc_down=q8g128,hc_up=q8g64,ple_kv=q6g128,lm_head=q8g64"),
]


def parse_layout(spec):
    out = {}
    for kv in spec.split(","):
        k, v = kv.split("=")
        if k == "embed":
            continue
        out[k] = v
    if "hc" not in out:
        out["hc"] = out.get("hc_down", out.get("hc_up", "q8g64"))
    return out


def qwen_recipe_footprint(lay):
    def bpw(fmt):
        return int(fmt[1]) + 32.0 / int(fmt[3:])
    H = 2560; V = 248320; E = 512; topk = 10; r = 320; n4 = 10240
    # 48 trunk + 1 MTP (approx MTP sizes matched to attn/moe)
    n_gdn = 36; n_attn_tr = 12; n_attn_all = 13; n_moe_all = 49
    cls_params = {
        "exp_gu":   (n_moe_all * E * 1280 * H, 48 * topk * 1280 * H),
        "exp_down": (n_moe_all * E * H * 640,  48 * topk * H * 640),
        "sh_gu":    (n_moe_all * 1280 * H,     48 * 1280 * H),
        "sh_down":  (n_moe_all * H * 640,      48 * H * 640),
        "gdn_in":   (n_gdn * 16480 * H,        n_gdn * 16480 * H),
        "gdn_out":  (n_gdn * H * 6144,         n_gdn * H * 6144),
        "attn_qkv": (n_attn_all * 13952 * H,   n_attn_tr * 13952 * H),
        "attn_o":   (n_attn_all * H * 6144,    n_attn_tr * H * 6144),
        "hc_down":  (98 * (324 * n4),          97 * (324 * n4)),
        "hc_up":    (98 * (n4 * r),            97 * (n4 * r)),
        "ple_kv":   (12800 * H,                12800 * H),
        "lm_head":  (V * H,                    V * H),
    }
    fixed_tot = V * H * 2 + n_moe_all * 513 * H * 2 + 1.1 * (1024 ** 2) * 150
    fixed_act = 48 * 513 * H * 2 + 120 * (1024 ** 2)  # router + GDN state/KV short
    fmt_for = lambda c: lay.get(c, lay["hc"]) if c in ("hc_down", "hc_up") else lay[c]
    tot_B = fixed_tot + sum(cls_params[c][0] * bpw(fmt_for(c)) / 8.0 for c in cls_params)
    act_B = fixed_act + sum(cls_params[c][1] * bpw(fmt_for(c)) / 8.0 for c in cls_params)
    act_MB = act_B / (1024 ** 2)
    # Calibrated to Swift-1.5 measured 194.0 MB/ms effective wall (4956 MB / 25.55 ms = 39.1 tok/s bare, 71.7 MTP)
    tps_bare = 194.0 * 1000.0 / act_MB
    return tot_B / (1024 ** 3), act_MB, tps_bare, tps_bare * 1.83


# ---------------------------------------------------------------------------
# 6. Multi-Domain Calibration Document (T in 1024..2048)
# ---------------------------------------------------------------------------

CALIB_DOC = (
    "<|im_start|>system\nYou are a principal systems engineer and mathematician.<|im_end|>\n"
    "<|im_start|>user\n"
    "Provide a complete technical response covering four domains:\n"
    "1. Production Python 3.12 asyncio DAG executor with semaphore throttling and cycle detection.\n"
    "2. Formal step-by-step proof of the Riemann-Lebesgue Lemma for f in L1(R).\n"
    "3. Mechanical 4-level x86_64 page table walk, TLB shootdowns via IPIs, and Linux COW page faults.\n"
    "4. Rigorous Russian engineering translation on zero-copy Unified Memory Architecture contention.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n"
    "Let's cover all four sections rigorously and cleanly without omitting details.\n"
    "</think>\n"
    "### 1. Asynchronous Task Execution DAG (Python 3.12 `asyncio`)\n"
    "```python\n"
    "from __future__ import annotations\n"
    "import asyncio\n"
    "from collections import deque\n"
    "from dataclasses import dataclass, field\n"
    "from typing import Awaitable, Callable, Dict, List, Set\n\n"
    "class CycleDetectedError(ValueError):\n"
    "    pass\n\n"
    "@dataclass(slots=True)\n"
    "class TaskNode:\n"
    "    name: str\n"
    "    fn: Callable[[Dict[str, object]], Awaitable[object]]\n"
    "    deps: tuple[str, ...] = ()\n"
    "    timeout: float = 5.0\n\n"
    "async def execute_dag(nodes: List[TaskNode], max_concurrent_tasks: int = 4) -> Dict[str, object]:\n"
    "    by_name = {n.name: n for n in nodes}\n"
    "    indeg = {n.name: len(n.deps) for n in nodes}\n"
    "    children: Dict[str, List[str]] = {n.name: [] for n in nodes}\n"
    "    for n in nodes:\n"
    "        for d in n.deps:\n"
    "            if d not in by_name:\n"
    "                raise KeyError(f'Missing dependency {d} for {n.name}')\n"
    "            children[d].append(n.name)\n\n"
    "    # Kahn topological cycle validation before launching any coroutine\n"
    "    q = deque([k for k, v in indeg.items() if v == 0])\n"
    "    visited = 0\n"
    "    indeg_chk = dict(indeg)\n"
    "    while q:\n"
    "        u = q.popleft()\n"
    "        visited += 1\n"
    "        for v in children[u]:\n"
    "            indeg_chk[v] -= 1\n"
    "            if indeg_chk[v] == 0:\n"
    "                q.append(v)\n"
    "    if visited != len(nodes):\n"
    "        raise CycleDetectedError('DAG contains cyclic dependencies')\n\n"
    "    sem = asyncio.Semaphore(max_concurrent_tasks)\n"
    "    done_events = {n.name: asyncio.Event() for n in nodes}\n"
    "    failed_nodes: Set[str] = set()\n"
    "    results: Dict[str, object] = {}\n"
    "    errors: Dict[str, BaseException] = {}\n\n"
    "    async def runner(node: TaskNode) -> None:\n"
    "        await asyncio.gather(*(done_events[d].wait() for d in node.deps))\n"
    "        if any(d in failed_nodes for d in node.deps):\n"
    "            failed_nodes.add(node.name)\n"
    "            done_events[node.name].set()\n"
    "            return\n"
    "        try:\n"
    "            async with sem:\n"
    "                results[node.name] = await asyncio.wait_for(node.fn(results), timeout=node.timeout)\n"
    "        except BaseException as exc:\n"
    "            failed_nodes.add(node.name)\n"
    "            errors[node.name] = exc\n"
    "        finally:\n"
    "            done_events[node.name].set()\n\n"
    "    await asyncio.gather(*(runner(n) for n in nodes))\n"
    "    return results\n"
    "```\n\n"
    "### 2. Riemann-Lebesgue Lemma for $L^1(\\mathbb{R})$\n"
    "**Theorem:** Let $f \\in L^1(\\mathbb{R}, dx)$ where $dx$ denotes Lebesgue measure. Then the Fourier transform\n"
    "$\\hat{f}(\\xi) = \\int_{-\\infty}^{\\infty} f(x) e^{-i \\xi x} dx$ satisfies $\\lim_{|\\xi| \\to \\infty} \\hat{f}(\\xi) = 0$.\n\n"
    "**Proof:** Step 1 (Indicator of a bounded interval). For $\\chi_{[a,b]}(x)$ and $\\xi \\neq 0$, explicit integration gives:\n"
    "$\\int_a^b e^{-i \\xi x} dx = \\frac{e^{-i \\xi a} - e^{-i \\xi b}}{i \\xi}$, whose modulus is bounded by $\\frac{2}{|\\xi|} \\to 0$ as $|\\xi| \\to \\infty$.\n"
    "Step 2 (Step functions). By linearity of the integral and the triangle inequality, for any finite linear combination "
    "$s(x) = \\sum_{k=1}^M c_k \\chi_{[a_k, b_k]}(x)$, we have $|\\hat{s}(\\xi)| \\le \\frac{2}{|\\xi|} \\sum_{k=1}^M |c_k| \\to 0$.\n"
    "Step 3 (Density in $L^1$). Given $\\varepsilon > 0$, since compactly supported step functions are norm-dense in $L^1(\\mathbb{R})$, "
    "choose a step function $s$ with $\\|f - s\\|_1 < \\varepsilon / 2$. Applying the triangle inequality:\n"
    "$|\\hat{f}(\\xi)| \\le |\\widehat{f - s}(\\xi)| + |\\hat{s}(\\xi)| \\le \\|f - s\\|_1 + \\frac{C_s}{|\\xi|} < \\frac{\\varepsilon}{2} + \\frac{\\varepsilon}{2} = \\varepsilon$\n"
    "for all $|\\xi| > 2 C_s / \\varepsilon$. In physical frequency space, rapid oscillation cancels slowly varying integrals, "
    "whereas outside $L^1(\\mathbb{R})$ (for example $e^{i \\omega_0 x}$ or the Dirac delta measure) high-frequency resonance preserves energy.\n\n"
    "### 3. x86_64 4-Level Page Table Walk & Linux Fault Handling\n"
    "A 48-bit canonical virtual address decomposes into PGD [47:39], PUD [38:30], PMD [29:21], PTE [20:12] (each 9 bits = 512 entries "
    "of 8 bytes inside a 4 KiB page), and a 12-bit byte offset [11:0]. Register `CR3` holds the physical frame address of the top-level "
    "PGD plus the 12-bit PCID tag when `CR4.PCIDE=1`. On a TLB miss, the hardware page table walker issues physical reads sequentially: "
    "`PA(PML4E) = (CR3 & ~0xFFF) + PGD*8`, checking Present, R/W, U/S, and NX bits at each level. When memory permissions change or pages "
    "unmap across SMP cores, the kernel shoots down stale remote TLB entries via synchronous Inter-Processor Interrupts (`smp_call_function_many`), "
    "which stalls pipeline execution by hundreds of nanoseconds. By contrast, when a PTE is non-present or read-only for COW, the CPU raises `#PF` "
    "with the faulting VA in `CR2`, entering `do_page_fault -> handle_mm_fault`: anonymous zero pages allocate fresh frames via `alloc_zeroed_user_highpage_movable`, "
    "while Copy-on-Write clones the shared page frame (`wp_page_copy`), decrements `_refcount`, updates the PTE writable, and flushes local TLB.\n\n"
    "### 4. Профессиональный перевод на русский научно-технический язык\n"
    "«Архитектура унифицированной памяти с нулевым копированием (zero-copy UMA) выявляет конкурентную борьбу за блокировки "
    "в кольцевых буферах пользовательских очередей HSA, работающих в обход ядра ОС, в моменты, когда асинхронные вычислительные ядра префилла "
    "вытесняют проходы спекулятивной верификации. Вследствие этого регулятор памяти запускает каскадные циклы принудительного высвобождения "
    "страниц (direct reclaim), вызывая дефицит пропускной способности контроллеров памяти и остановку диспетчеризации волн (wavefronts) "
    "на всех вычислительных блоках.»\n"
    "<|im_end|>\n"
)


# ---------------------------------------------------------------------------
# 7. PLE Hash Helper & RoPE Builder
# ---------------------------------------------------------------------------

def ple_ngram_ids_py(multipliers, vocab_sizes, offsets, ids, eos=248044):
    """Exact 64-bit wrapping implementation of runtime/ple_hash.cpp."""
    MASK = (1 << 64) - 1
    def to_i64(x):
        x &= MASK
        return x - (1 << 64) if (x & (1 << 63)) else x
    def mul64(a, b):
        return to_i64((a & MASK) * (b & MASK))
    def c_rem_pos(h, v):
        # C++: m = h % v; if (m < 0) m += v  == Python's (h % v) since v > 0
        return h % v

    m0, m1, m2 = multipliers
    p0, p1 = eos, eos
    out = []
    uniq = set()
    for s0 in ids:
        s1 = p1
        s2 = eos if p1 == eos else p0
        h2 = to_i64(mul64(s0, m0) ^ mul64(s1, m1))
        h3 = to_i64(h2 ^ mul64(s2, m2))
        row_ids = []
        for j in range(16):
            h = h2 if j < 8 else h3
            r = c_rem_pos(h, vocab_sizes[j]) + offsets[j]
            row_ids.append(r)
            uniq.add(r)
        out.append(row_ids)
        p0, p1 = p1, s0
    uniq_sorted = sorted(uniq)
    pos_map = {r: idx for idx, r in enumerate(uniq_sorted)}
    mapped_idx = [[pos_map[r] for r in row] for row in out]
    return uniq_sorted, mapped_idx


def build_qwen_rope(T, device, dtype):
    inv_freq = 1.0 / (1e7 ** (torch.arange(0, 64, 2, device=device, dtype=torch.float32) / 64.0))
    freqs = torch.outer(torch.arange(T, device=device, dtype=torch.float32), inv_freq)
    return freqs.cos().to(dtype)[None, None, :, :], freqs.sin().to(dtype)[None, None, :, :]


# ---------------------------------------------------------------------------
# 8. Self-Test Mode
# ---------------------------------------------------------------------------

def run_self_test(model_dir):
    dev = torch.device("cuda:0")
    _ = torch.zeros(1, device=dev)
    torch.cuda.synchronize()

    tok = QwenTokenizer(os.path.join(model_dir, "tokenizer.json"))
    ids = tok.encode(CALIB_DOC)
    dec = tok.decode(ids)
    assert dec == unicodedata.normalize("NFC", CALIB_DOC), "Tokenizer round-trip mismatch!"
    assert 1024 <= len(ids) <= 2048, f"Expected T in [1024, 2048], got {len(ids)}"
    print(f"[PASS] Tokenizer round-trip exact! Calibration T = {len(ids)} tokens (<= dense_key_limit 2051).")

    torch.manual_seed(42)
    T, Hv, D = 192, 8, 128
    q = F.normalize(torch.randn(T, Hv, D, device=dev), dim=-1) * (D ** -0.5)
    k = F.normalize(torch.randn(T, Hv, D, device=dev), dim=-1)
    v = torch.randn(T, Hv, D, device=dev) * 0.5
    beta = torch.sigmoid(torch.randn(T, Hv, device=dev))
    g = -torch.exp(torch.randn(Hv, device=dev) * 0.5) * F.softplus(torch.randn(T, Hv, device=dev) - 1.0)

    ref = recurrent_delta_rule_gdn(q, k, v, beta, g)
    chk = chunked_delta_rule_gdn(q, k, v, beta, g, chunk_size=64)
    max_err = (ref - chk).abs().max().item()
    rel_err = ((ref - chk).norm() / ref.norm()).item()
    print(f"[PASS] Chunked WY GDN vs Sequential Recurrence (T={T}): max_abs={max_err:.2e}, rel_l2={rel_err:.2e}")
    assert max_err < 1e-4 and rel_err < 1e-5, f"GDN mismatch: {max_err} / {rel_err}"


# ---------------------------------------------------------------------------
# 9. Main Per-Model Single-Pass Simulation
# ---------------------------------------------------------------------------

def simulate_model(label, model_dir, ids, mode, recipes=None, expert_chunk=64):
    dev = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(dev)
    reader = SafetensorsReader(model_dir)
    lm = "model.language_model."
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg_root = json.load(f)
    tc = cfg_root.get("text_config", cfg_root)
    n_layers = tc["num_hidden_layers"]
    layer_types = tc.get("layer_types")

    T = len(ids)
    input_ids_cpu = torch.tensor(ids, dtype=torch.long)
    target_ids = torch.tensor(ids[1:], device=dev, dtype=torch.long)
    cos, sin = build_qwen_rope(T, dev, torch.bfloat16)

    recipe_list = recipes if recipes is not None else RECIPES
    if mode == "baseline":
        active_specs = [("BF16", None)]
    else:
        active_specs = [("BF16", None)] + [(n, parse_layout(s)) for n, s in recipe_list]
    state_names = [n for n, _ in active_specs]

    t0 = time.time()
    print(f"\n==========================================================================================")
    print(f"MODEL: {label} ({model_dir}) | layers={n_layers} | T={T} | mode={mode} | expert_chunk={expert_chunk}")
    print(f"==========================================================================================")

    sens_all = []  # list of (rel_err, layer_idx, cls, fmt)
    sens_cls = defaultdict(list)
    prefetch_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    with torch.no_grad():
        # Embedding: read full table on CPU (1.27 GB BF16) and index to 4 streams on GPU
        emb_cpu = reader.read_tensor(lm + "embed_tokens.weight", torch.device("cpu"), torch.bfloat16)
        x0_1 = F.embedding(input_ids_cpu, emb_cpu).to(dev)  # [T, 2560]
        del emb_cpu
        X0 = x0_1.unsqueeze(1).expand(T, 4, 2560).contiguous()  # [T, 4, 2560]
        del x0_1
        torch.cuda.empty_cache()

        states = {n: X0.clone() for n in state_names}
        del X0

        for L in range(n_layers):
            tl = time.time()
            P = f"{lm}layers.{L}."
            if layer_types is not None:
                is_attn = (layer_types[L] == "full_attention")
            else:
                is_attn = ((L + 1) % 4 == 0)

            # ---- Optional PLE Injection (check actual weight presence, matching convert_qwen4exp.cpp) ----
            if (P + "ple.key_proj.weight") in reader.wm:
                ple_p = P + "ple."
                mult = reader.read_tensor(ple_p + "ple_embedding.layer_multipliers", torch.device("cpu")).tolist()
                vsz = reader.read_tensor(ple_p + "ple_embedding.ngram_heads_vocab_sizes", torch.device("cpu")).tolist()
                offs = reader.read_tensor(ple_p + "ple_embedding.ngram_heads_offsets", torch.device("cpu")).tolist()
                uniq_rows, mapped_idx = ple_ngram_ids_py(mult, vsz, offs, ids, eos=248044)
                gathered = reader.gather_ple_rows(ple_p, uniq_rows, dev)  # [U, 160]
                idx_t = torch.tensor(mapped_idx, device=dev, dtype=torch.long)  # [T, 16]
                ple_e = gathered[idx_t].reshape(T, 2560)
                del gathered, idx_t

                w_kp = reader.read_tensor(ple_p + "key_proj.weight", dev, torch.float32)
                w_vp = reader.read_tensor(ple_p + "value_proj.weight", dev, torch.float32)
                w_ple_f32 = torch.cat([w_kp, w_vp], dim=0)
                del w_kp, w_vp
                ple_cw = reader.read_tensor(ple_p + "conv1d.weight", dev, torch.float32)
                ple_nk = reader.read_tensor(ple_p + "norm_key.weight", dev, torch.float32)
                ple_nq = reader.read_tensor(ple_p + "norm_query.weight", dev, torch.float32)
                ple_nc = reader.read_tensor(ple_p + "norm_conv.weight", dev, torch.float32)

                fmts_ple = set((lay["ple_kv"] if lay else "bf16") for _, lay in active_specs)
                if mode == "pareto":
                    fmts_ple.update(["q4g128", "q4g64", "q5g128", "q6g64", "q8g64"])
                b_ple = {f: fake_quant(w_ple_f32, f) for f in fmts_ple}

                if mode == "pareto":
                    ref_px = ple_forward(states["BF16"], ple_e, b_ple["bf16"], ple_cw, ple_nk, ple_nq, ple_nc)
                    d_px = (ref_px - states["BF16"]).float().norm().item() + 1e-12
                    for f in ("q4g128", "q4g64", "q5g128", "q6g64", "q8g64"):
                        yx = ple_forward(states["BF16"], ple_e, b_ple[f], ple_cw, ple_nk, ple_nq, ple_nc)
                        re = (yx - ref_px).float().norm().item() / d_px
                        sens_cls[f"ple_kv:{f}"].append(re)
                        sens_all.append((re, L, "ple_kv", f))

                for name, lay in active_specs:
                    f = lay["ple_kv"] if lay else "bf16"
                    states[name] = ple_forward(states[name], ple_e, b_ple[f], ple_cw, ple_nk, ple_nq, ple_nc)

                del ple_e, w_ple_f32, b_ple, ple_cw, ple_nk, ple_nq, ple_nc

            # ---- Attention / GDN Hyper-Connection Mix Down+Up ----
            ahc = P + "attn_hyper_connection."
            a_wd = reader.read_tensor(ahc + "input_mix_weight_down.weight", dev, torch.float32)
            a_wi = reader.read_tensor(ahc + "block_inject_weight.weight", dev, torch.float32)
            a_wu = reader.read_tensor(ahc + "input_mix_weight_up.weight", dev, torch.float32)
            a_hn = reader.read_tensor(ahc + "hc_norm.weight", dev, torch.float32)
            a_wdf = a_wd * (1.0 + a_hn.unsqueeze(0))
            a_wif = a_wi * (1.0 + a_hn.unsqueeze(0))
            del a_wd, a_wi

            fmts_hcd = set((lay.get("hc_down", lay["hc"]) if lay else "bf16") for _, lay in active_specs)
            fmts_hcu = set((lay.get("hc_up", lay["hc"]) if lay else "bf16") for _, lay in active_specs)
            if mode == "pareto":
                fmts_hcd.update(["q4g128", "q4g64", "q5g128", "q6g64", "q8g64"])
                fmts_hcu.update(["q4g128", "q4g64", "q5g128", "q6g64", "q8g64"])
            b_ahcd = {f: (fake_quant(a_wdf, f), fake_quant(a_wif, f)) for f in fmts_hcd}
            b_ahcu = {f: fake_quant(a_wu, f) for f in fmts_hcu}

            u_attn = {}
            w_in_attn = {}
            for name, lay in active_specs:
                fd = lay.get("hc_down", lay["hc"]) if lay else "bf16"
                fu = lay.get("hc_up", lay["hc"]) if lay else "bf16"
                u_attn[name], w_in_attn[name] = hc_mix_down_up(states[name], *b_ahcd[fd], b_ahcu[fu], a_hn, inject=True)

            if mode == "pareto":
                du = u_attn["BF16"].float().norm().item() + 1e-12
                for f in ("q4g128", "q4g64", "q5g128", "q6g64", "q8g64"):
                    uq, _ = hc_mix_down_up(states["BF16"], *b_ahcd[f], b_ahcu[f], a_hn, inject=True)
                    re = (uq - u_attn["BF16"]).float().norm().item() / du
                    sens_cls[f"hc:{f}"].append(re)
                    sens_all.append((re, L, "hc", f))

            del a_wdf, a_wif, a_wu, a_hn, b_ahcd, b_ahcu

            # ---- Token Mixer: GDN or Full Attention ----
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

                fg_in = set((lay["gdn_in"] if lay else "bf16") for _, lay in active_specs)
                fg_out = set((lay["gdn_out"] if lay else "bf16") for _, lay in active_specs)
                if mode == "pareto":
                    fg_in.update(["q4g128", "q4g64", "q5g128", "q6g64", "q8g64"])
                    fg_out.update(["q4g128", "q4g64", "q5g128", "q6g64", "q8g64"])
                b_gin = {f: fake_quant(w_gdn_in, f) for f in fg_in}
                b_gout = {f: fake_quant(w_gdn_out, f) for f in fg_out}

                y_bf16 = gdn_forward(u_attn["BF16"], b_gin["bf16"], b_gout["bf16"], cw, A_log, dt_bias, gnorm)
                if mode == "pareto":
                    dm = y_bf16.float().norm().item() + 1e-12
                    for f in ("q4g128", "q4g64", "q5g128", "q6g64", "q8g64"):
                        yq = gdn_forward(u_attn["BF16"], b_gin[f], b_gout["bf16"], cw, A_log, dt_bias, gnorm)
                        re = (yq - y_bf16).float().norm().item() / dm
                        sens_cls[f"gdn_in:{f}"].append(re)
                        sens_all.append((re, L, "gdn_in", f))

                        yq2 = gdn_forward(u_attn["BF16"], b_gin["bf16"], b_gout[f], cw, A_log, dt_bias, gnorm)
                        re2 = (yq2 - y_bf16).float().norm().item() / dm
                        sens_cls[f"gdn_out:{f}"].append(re2)
                        sens_all.append((re2, L, "gdn_out", f))

                states["BF16"] = (states["BF16"].float() + w_in_attn["BF16"].unsqueeze(-1) * y_bf16.float().unsqueeze(1)).to(torch.bfloat16)
                for name, lay in active_specs[1:]:
                    ym = gdn_forward(u_attn[name], b_gin[lay["gdn_in"]], b_gout[lay["gdn_out"]], cw, A_log, dt_bias, gnorm)
                    states[name] = (states[name].float() + w_in_attn[name].unsqueeze(-1) * ym.float().unsqueeze(1)).to(torch.bfloat16)

                del w_gdn_in, w_gdn_out, cw, A_log, dt_bias, gnorm, b_gin, b_gout, y_bf16
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

                fqkv = set((lay["attn_qkv"] if lay else "bf16") for _, lay in active_specs)
                fao = set((lay["attn_o"] if lay else "bf16") for _, lay in active_specs)
                if mode == "pareto":
                    fqkv.update(["q4g128", "q4g64", "q5g128", "q6g64", "q8g64"])
                    fao.update(["q4g128", "q4g64", "q5g128", "q6g64", "q8g64"])
                b_qkv = {f: fake_quant(w_qkv, f) for f in fqkv}
                b_ao = {f: fake_quant(w_ao, f) for f in fao}

                y_bf16 = full_attn_forward(u_attn["BF16"], b_qkv["bf16"], b_ao["bf16"], qn, kn, cos, sin)
                if mode == "pareto":
                    dm = y_bf16.float().norm().item() + 1e-12
                    for f in ("q4g128", "q4g64", "q5g128", "q6g64", "q8g64"):
                        yq = full_attn_forward(u_attn["BF16"], b_qkv[f], b_ao["bf16"], qn, kn, cos, sin)
                        re = (yq - y_bf16).float().norm().item() / dm
                        sens_cls[f"attn_qkv:{f}"].append(re)
                        sens_all.append((re, L, "attn_qkv", f))

                        yq2 = full_attn_forward(u_attn["BF16"], b_qkv["bf16"], b_ao[f], qn, kn, cos, sin)
                        re2 = (yq2 - y_bf16).float().norm().item() / dm
                        sens_cls[f"attn_o:{f}"].append(re2)
                        sens_all.append((re2, L, "attn_o", f))

                states["BF16"] = (states["BF16"].float() + w_in_attn["BF16"].unsqueeze(-1) * y_bf16.float().unsqueeze(1)).to(torch.bfloat16)
                for name, lay in active_specs[1:]:
                    ym = full_attn_forward(u_attn[name], b_qkv[lay["attn_qkv"]], b_ao[lay["attn_o"]], qn, kn, cos, sin)
                    states[name] = (states[name].float() + w_in_attn[name].unsqueeze(-1) * ym.float().unsqueeze(1)).to(torch.bfloat16)

                del w_qkv, w_ao, qn, kn, b_qkv, b_ao, y_bf16

            del u_attn, w_in_attn

            # ---- MLP Hyper-Connection Mix ----
            mhc = P + "mlp_hyper_connection."
            m_wd = reader.read_tensor(mhc + "input_mix_weight_down.weight", dev, torch.float32)
            m_wi = reader.read_tensor(mhc + "block_inject_weight.weight", dev, torch.float32)
            m_wu = reader.read_tensor(mhc + "input_mix_weight_up.weight", dev, torch.float32)
            m_hn = reader.read_tensor(mhc + "hc_norm.weight", dev, torch.float32)
            m_wdf = m_wd * (1.0 + m_hn.unsqueeze(0))
            m_wif = m_wi * (1.0 + m_hn.unsqueeze(0))
            del m_wd, m_wi

            fmts_hcd = set((lay.get("hc_down", lay["hc"]) if lay else "bf16") for _, lay in active_specs)
            fmts_hcu = set((lay.get("hc_up", lay["hc"]) if lay else "bf16") for _, lay in active_specs)
            if mode == "pareto":
                fmts_hcd.update(["q4g128", "q4g64", "q5g128", "q6g64", "q8g64"])
                fmts_hcu.update(["q4g128", "q4g64", "q5g128", "q6g64", "q8g64"])
            b_mhcd = {f: (fake_quant(m_wdf, f), fake_quant(m_wif, f)) for f in fmts_hcd}
            b_mhcu = {f: fake_quant(m_wu, f) for f in fmts_hcu}

            u_mlp = {}
            w_in_mlp = {}
            for name, lay in active_specs:
                fd = lay.get("hc_down", lay["hc"]) if lay else "bf16"
                fu = lay.get("hc_up", lay["hc"]) if lay else "bf16"
                u_mlp[name], w_in_mlp[name] = hc_mix_down_up(states[name], *b_mhcd[fd], b_mhcu[fu], m_hn, inject=True)

            if mode == "pareto":
                du = u_mlp["BF16"].float().norm().item() + 1e-12
                for f in ("q4g128", "q4g64", "q5g128", "q6g64", "q8g64"):
                    uq, _ = hc_mix_down_up(states["BF16"], *b_mhcd[f], b_mhcu[f], m_hn, inject=True)
                    re = (uq - u_mlp["BF16"]).float().norm().item() / du
                    sens_cls[f"hc:{f}"].append(re)
                    sens_all.append((re, L, "hc", f))

            del m_wdf, m_wif, m_wu, m_hn, b_mhcd, b_mhcu

            # ---- MoE Router + Shared Expert ----
            mp = P + "mlp."
            wg_r = reader.read_tensor(mp + "gate.weight", dev, torch.float32)
            wsg_r = reader.read_tensor(mp + "shared_expert_gate.weight", dev, torch.float32)
            w_router_all = torch.cat([wg_r, wsg_r], dim=0)
            del wg_r, wsg_r

            routes = {}
            for name in state_names:
                routes[name] = moe_route_qwen(u_mlp[name], w_router_all)
            del w_router_all

            w_sh_g = reader.read_tensor(mp + "shared_expert.gate_proj.weight", dev, torch.float32)
            w_sh_u = reader.read_tensor(mp + "shared_expert.up_proj.weight", dev, torch.float32)
            w_sh_gu = torch.cat([w_sh_g, w_sh_u], dim=0)
            del w_sh_g, w_sh_u
            w_sh_d = reader.read_tensor(mp + "shared_expert.down_proj.weight", dev, torch.float32)

            f_shgu = set((lay["sh_gu"] if lay else "bf16") for _, lay in active_specs)
            f_shdn = set((lay["sh_down"] if lay else "bf16") for _, lay in active_specs)
            if mode == "pareto":
                f_shgu.update(["q4g128", "q4g64", "q5g128", "q6g64", "q8g64"])
                f_shdn.update(["q4g128", "q4g64", "q5g128", "q6g64", "q8g64"])
            b_shgu = {f: fake_quant(w_sh_gu, f) for f in f_shgu}
            b_shdn = {f: fake_quant(w_sh_d, f) for f in f_shdn}

            def run_sh(x, wgu, wdn, sh_c):
                gu = F.linear(x, wgu)
                h = F.silu(gu[:, :640]) * gu[:, 640:]
                return F.linear(h, wdn).float() * sh_c.unsqueeze(-1)

            sh_bf16 = run_sh(u_mlp["BF16"], b_shgu["bf16"], b_shdn["bf16"], routes["BF16"][2])
            if mode == "pareto":
                d_sh = sh_bf16.norm().item() + 1e-12
                for f in ("q4g128", "q4g64", "q5g128", "q6g64", "q8g64"):
                    ys1 = run_sh(u_mlp["BF16"], b_shgu[f], b_shdn["bf16"], routes["BF16"][2])
                    re1 = (ys1 - sh_bf16).norm().item() / d_sh
                    sens_cls[f"sh_gu:{f}"].append(re1)
                    sens_all.append((re1, L, "sh_gu", f))

                    ys2 = run_sh(u_mlp["BF16"], b_shgu["bf16"], b_shdn[f], routes["BF16"][2])
                    re2 = (ys2 - sh_bf16).norm().item() / d_sh
                    sens_cls[f"sh_down:{f}"].append(re2)
                    sens_all.append((re2, L, "sh_down", f))

            sh_outs = {"BF16": sh_bf16}
            for name, lay in active_specs[1:]:
                sh_outs[name] = run_sh(u_mlp[name], b_shgu[lay["sh_gu"]], b_shdn[lay["sh_down"]], routes[name][2])
            del w_sh_gu, w_sh_d, b_shgu, b_shdn, sh_bf16

            # ---- Routed Experts: Stream in 64-Expert Chunks (Peak VRAM < 10 GB) ----
            routed_acc = {name: torch.zeros((T, 2560), device=dev, dtype=torch.float32) for name in state_names}
            sens_exp_acc = {}
            sens_exp_list = []
            if mode == "pareto":
                for f in ("q4g128", "q4g64", "q5g128", "q6g64", "q8g64"):
                    sens_exp_list.append((f"exp_gu:{f}", "exp_gu", f, "bf16"))
                    sens_exp_list.append((f"exp_down:{f}", "exp_down", "bf16", f))
                for tag, _, _, _ in sens_exp_list:
                    sens_exp_acc[tag] = torch.zeros((T, 2560), device=dev, dtype=torch.float32)

            exp_gu_fmts = set((lay["exp_gu"] if lay else "bf16") for _, lay in active_specs)
            exp_dn_fmts = set((lay["exp_down"] if lay else "bf16") for _, lay in active_specs)
            if mode == "pareto":
                exp_gu_fmts.update(["q4g128", "q4g64", "q5g128", "q6g64", "q8g64"])
                exp_dn_fmts.update(["q4g128", "q4g64", "q5g128", "q6g64", "q8g64"])

            CHUNK = expert_chunk
            slot_buf = torch.zeros((T, 10, 2560), device=dev, dtype=torch.float32)
            gu_name = mp + "experts.gate_up_proj"
            dn_name = mp + "experts.down_proj"
            reader.info(gu_name)
            reader.info(dn_name)
            cpu_dev = torch.device("cpu")
            def _load_chunk_cpu(c_start, c_cnt):
                return (
                    reader.read_slice_dim0(gu_name, c_start, c_cnt, cpu_dev, torch.bfloat16),
                    reader.read_slice_dim0(dn_name, c_start, c_cnt, cpu_dev, torch.bfloat16),
                )
            next_fut = prefetch_pool.submit(_load_chunk_cpu, 0, min(CHUNK, 512))
            for c0 in range(0, 512, CHUNK):
                cnt = min(CHUNK, 512 - c0)
                c1 = c0 + cnt
                gu_cpu, dn_cpu = next_fut.result()
                if c1 < 512:
                    next_fut = prefetch_pool.submit(_load_chunk_cpu, c1, min(CHUNK, 512 - c1))

                gu_raw = gu_cpu.to(dev)
                del gu_cpu
                q_gu = {f: fake_quant(gu_raw, f) for f in exp_gu_fmts}
                del gu_raw

                dn_raw = dn_cpu.to(dev)
                del dn_cpu
                q_dn = {f: fake_quant(dn_raw, f) for f in exp_dn_fmts}
                del dn_raw

                # Accumulate routed output for each state's hits inside [c0, c1)
                for name, lay in active_specs:
                    tidx, tw, _ = routes[name]
                    in_chunk = (tidx >= c0) & (tidx < c1)
                    if not in_chunk.any():
                        continue
                    tok_pos, slot_pos = torch.nonzero(in_chunk, as_tuple=True)
                    loc_e = tidx[tok_pos, slot_pos] - c0  # [K_hits]
                    # Group hits by local expert ID (stable sort avoids GPU atomic jitter)
                    order = torch.argsort(loc_e, stable=True)
                    loc_sorted = loc_e[order]
                    tok_sorted = tok_pos[order]
                    slot_sorted = slot_pos[order]
                    w_sorted = tw[tok_pos, slot_pos][order].unsqueeze(-1)
                    uniq_e, counts = torch.unique_consecutive(loc_sorted, return_counts=True)

                    fgu = lay["exp_gu"] if lay else "bf16"
                    fdn = lay["exp_down"] if lay else "bf16"
                    wgu_c = q_gu[fgu]
                    wdn_c = q_dn[fdn]
                    x_hits = u_mlp[name][tok_sorted]

                    outs = torch.empty_like(x_hits)
                    pos = 0
                    for le_i, cnt_i in zip(uniq_e.tolist(), counts.tolist()):
                        xs = x_hits[pos : pos + cnt_i]
                        gu_e = F.linear(xs, wgu_c[le_i])
                        hs = F.silu(gu_e[:, :640]) * gu_e[:, 640:]
                        outs[pos : pos + cnt_i] = F.linear(hs, wdn_c[le_i])
                        pos += cnt_i

                    slot_buf.zero_()
                    slot_buf[tok_sorted, slot_sorted] = outs.float() * w_sorted
                    routed_acc[name].add_(slot_buf.sum(dim=1))

                    if name == "BF16" and sens_exp_list:
                        for tag, _, fgu_s, fdn_s in sens_exp_list:
                            wgu_s = q_gu[fgu_s]
                            wdn_s = q_dn[fdn_s]
                            outs_s = torch.empty_like(x_hits)
                            pos = 0
                            for le_i, cnt_i in zip(uniq_e.tolist(), counts.tolist()):
                                xs = x_hits[pos : pos + cnt_i]
                                gu_e = F.linear(xs, wgu_s[le_i])
                                hs = F.silu(gu_e[:, :640]) * gu_e[:, 640:]
                                outs_s[pos : pos + cnt_i] = F.linear(hs, wdn_s[le_i])
                                pos += cnt_i
                            slot_buf.zero_()
                            slot_buf[tok_sorted, slot_sorted] = outs_s.float() * w_sorted
                            sens_exp_acc[tag].add_(slot_buf.sum(dim=1))

                del q_gu, q_dn
                torch.cuda.empty_cache()

            del slot_buf

            if sens_exp_list:
                r_den = routed_acc["BF16"].norm().item() + 1e-12
                for tag, cls_n, fgu_s, fdn_s in sens_exp_list:
                    fmt = fgu_s if cls_n == "exp_gu" else fdn_s
                    re = (sens_exp_acc[tag] - routed_acc["BF16"]).norm().item() / r_den
                    sens_cls[tag].append(re)
                    sens_all.append((re, L, cls_n, fmt))

            for name in state_names:
                y_moe = (routed_acc[name] + sh_outs[name]).to(torch.bfloat16)
                states[name] = (states[name].float() + w_in_mlp[name].unsqueeze(-1) * y_moe.float().unsqueeze(1)).to(torch.bfloat16)

            del u_mlp, w_in_mlp, routes, routed_acc, sens_exp_acc, sh_outs
            torch.cuda.empty_cache()

            if L in (0, 1, 3, 15, 31, 47):
                peak_gb = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
                print(f"  [Layer {L:2d}/47] {'Attn' if is_attn else 'GDN '} elapsed={time.time()-t0:5.1f}s ({time.time()-tl:.2f}s) | peakVRAM={peak_gb:.2f} GiB")

        # ---- Final Hyper-Connection Mixer + Chunked LM Head Evaluation ----
        fhc = lm + "hyper_connection_mixer."
        f_wd = reader.read_tensor(fhc + "input_mix_weight_down.weight", dev, torch.float32)
        f_wu = reader.read_tensor(fhc + "input_mix_weight_up.weight", dev, torch.float32)
        f_hn = reader.read_tensor(fhc + "hc_norm.weight", dev, torch.float32)
        f_wdf = f_wd * (1.0 + f_hn.unsqueeze(0))
        del f_wd
        dummy_inj = torch.zeros((4, 10240), device=dev, dtype=torch.float32)

        lm_w = reader.read_tensor("lm_head.weight", dev, torch.bfloat16)

        f_bf16_u, _ = hc_mix_down_up(states["BF16"], f_wdf.to(torch.bfloat16), dummy_inj, f_wu.to(torch.bfloat16), f_hn, inject=False)
        if mode == "pareto":
            du = f_bf16_u.float().norm().item() + 1e-12
            for f in ("q4g128", "q4g64", "q5g128", "q6g64", "q8g64"):
                uq, _ = hc_mix_down_up(states["BF16"], fake_quant(f_wdf, f), dummy_inj, fake_quant(f_wu, f), f_hn, inject=False)
                re = (uq - f_bf16_u).float().norm().item() / du
                sens_cls[f"hc:{f}"].append(re)
                sens_all.append((re, 48, "hc", f))

        TOK_CHUNK = 256
        n_eval = T - 1
        bf16_nll_sum = 0.0
        for s0 in range(0, n_eval, TOK_CHUNK):
            s1 = min(s0 + TOK_CHUNK, n_eval)
            lgt_c = F.linear(f_bf16_u[s0:s1], lm_w).float()
            lp_c = F.log_softmax(lgt_c, dim=-1)
            bf16_nll_sum += F.nll_loss(lp_c, target_ids[s0:s1], reduction="sum").item()
            del lgt_c, lp_c
        bf16_nll = bf16_nll_sum / float(n_eval)
        bf16_ppl = math.exp(bf16_nll)

        print(f"\n=== BF16 BASELINE ({label}, T={T}) ===")
        print(f"  BF16 NLL = {bf16_nll:.4f}, BF16 PPL = {bf16_ppl:.3f}, Peak VRAM = {torch.cuda.max_memory_allocated(dev)/(1024**3):.2f} GiB")

        if mode == "baseline":
            del lm_w, states, f_wdf, f_wu, f_hn, f_bf16_u, dummy_inj
            torch.cuda.empty_cache()
            reader.close()
            return {"label": label, "bf16_ppl": bf16_ppl, "peak_vram_gib": torch.cuda.max_memory_allocated(dev) / (1024 ** 3)}

        def eval_cand(u_cand, qlm_w, need_full=True):
            kld_parts = []
            nll_sum = 0.0
            t1_hits = 0
            t5_hits = 0
            for s0 in range(0, n_eval, TOK_CHUNK):
                s1 = min(s0 + TOK_CHUNK, n_eval)
                ref_lgt = F.linear(f_bf16_u[s0:s1], lm_w).float()
                ref_lp = F.log_softmax(ref_lgt, dim=-1)
                ref_p = ref_lp.exp()
                q_lgt = F.linear(u_cand[s0:s1], qlm_w).float()
                q_lp = F.log_softmax(q_lgt, dim=-1)
                kld_parts.append((ref_p * (ref_lp - q_lp)).sum(dim=-1))
                if need_full:
                    nll_sum += F.nll_loss(q_lp, target_ids[s0:s1], reduction="sum").item()
                    ref_t1 = ref_lgt.argmax(dim=-1)
                    t1_hits += (q_lgt.argmax(dim=-1) == ref_t1).sum().item()
                    t5_hits += (q_lgt.topk(5, dim=-1).indices == ref_t1.unsqueeze(-1)).any(dim=-1).sum().item()
                del ref_lgt, ref_lp, ref_p, q_lgt, q_lp
            kld_v = torch.cat(kld_parts, dim=0)
            mean_kld = kld_v.mean().item()
            if not need_full:
                return mean_kld
            p95_kld = torch.quantile(kld_v, 0.95).item()
            ppl = math.exp(nll_sum / float(n_eval))
            return mean_kld, p95_kld, ppl, 100.0 * t1_hits / float(n_eval), 100.0 * t5_hits / float(n_eval)

        # lm_head KLD sensitivity + cache needed Phase 2 lm_head formats
        needed_lm = set(lay["lm_head"] for _, lay in active_specs[1:])
        lm_kld = {}
        qlm_cache = {}
        heatmap = {}
        top15 = []
        if mode == "pareto":
            for f in ("q4g128", "q4g64", "q5g128", "q6g64", "q8g64"):
                qlm = fake_quant_rows(lm_w, f)
                lm_kld[f] = eval_cand(f_bf16_u, qlm, need_full=False)
                if f in needed_lm:
                    qlm_cache[f] = qlm
                else:
                    del qlm
                    torch.cuda.empty_cache()

            # Print Phase 1 heatmap
            fmts_hdr = ["q4g128", "q4g64", "q5g128", "q6g64", "q8g64"]
            print(f"\n=== PHASE 1 [{label}]: CLASS SENSITIVITY HEATMAP (Mean / Max Rel-L2 % vs BF16) ===")
            print(f"{'Class':10s} | " + " | ".join(f"{f:14s}" for f in fmts_hdr))
            print("-" * 96)
            for cls in CLASSES[:-1]:
                row_vals = {}
                cells = []
                for f in fmts_hdr:
                    vals = sens_cls.get(f"{cls}:{f}", [])
                    if vals:
                        mn_v = sum(vals) / len(vals) * 100.0
                        mx_v = max(vals) * 100.0
                        row_vals[f] = {"mean_pct": mn_v, "max_pct": mx_v}
                        cells.append(f"{mn_v:5.2f}%/{mx_v:6.2f}%")
                    else:
                        cells.append(f"{'---':14s}")
                heatmap[cls] = row_vals
                print(f"{cls:10s} | " + " | ".join(f"{c:14s}" for c in cells))
            heatmap["lm_head_kld"] = lm_kld
            print(f"{'lm_head':10s} | " + " | ".join(f"KLD={lm_kld[f]:.6f}" for f in fmts_hdr))

            # TOP-15 most sensitive (layer, class) pairs at q4g128
            q4_pairs = sorted([x for x in sens_all if x[3] == "q4g128"], reverse=True)[:15]
            print(f"\n=== TOP-15 MOST SENSITIVE (layer, class) AT q4g128 [{label}] ===")
            for rank, (err, lay_i, cls_n, fmt_n) in enumerate(q4_pairs, 1):
                print(f"  #{rank:2d}  L{lay_i:2d}.{cls_n:10s}  rel_L2 = {err*100:6.2f}%")
                top15.append({"layer": lay_i, "class": cls_n, "rel_l2_pct": err * 100.0})
        else:
            pass  # Stream one lm_head format at a time in Phase 2 to keep VRAM low

        # Phase 2: End-to-End Pareto Table (grouped by lm_head format to reuse 1 quantized head at a time)
        print(f"\n=== PHASE 2 [{label}]: END-TO-END PARETO FRONTIER (vs BF16) ===")
        print(f"{'Recipe':20s} | {'Disk':6s} | {'ActMB':6s} | {'Bare':8s} | {'MTP~':8s} | {'Mean KLD':9s} | {'p95 KLD':9s} | {'PPL':7s} | {'dPPL':7s} | {'Top1%':6s} | {'Top5%':6s}")
        print("-" * 122)
        print(f"{'BF16 Baseline':20s} | {'~240G':6s} | {'---':6s} | {'---':8s} | {'---':8s} | {'0.000000':9s} | {'0.000000':9s} | {bf16_ppl:7.3f} | {'+0.000':7s} | {'100.0%':6s} | {'100.0%':6s}")

        fhcd_cache = {f: fake_quant(f_wdf, f) for f in set(lay.get("hc_down", lay["hc"]) for _, lay in active_specs[1:])}
        fhcu_cache = {f: fake_quant(f_wu, f) for f in set(lay.get("hc_up", lay["hc"]) for _, lay in active_specs[1:])}
        by_lm = defaultdict(list)
        for name, lay in active_specs[1:]:
            by_lm[lay["lm_head"]].append((name, lay))

        results_by_name = {}
        for f_lm, items in by_lm.items():
            qlm_w = qlm_cache.get(f_lm)
            if qlm_w is None:
                qlm_w = fake_quant_rows(lm_w, f_lm)
            for name, lay in items:
                fd = lay.get("hc_down", lay["hc"])
                fu = lay.get("hc_up", lay["hc"])
                u_fin, _ = hc_mix_down_up(
                    states[name],
                    fhcd_cache[fd],
                    dummy_inj,
                    fhcu_cache[fu],
                    f_hn,
                    inject=False,
                )
                mean_kld, p95_kld, ppl, t1, t5 = eval_cand(u_fin, qlm_w, need_full=True)
                del u_fin
                disk_gb, act_mb, bare_tps, mtp_tps = qwen_recipe_footprint(lay)
                results_by_name[name] = {
                    "recipe": name, "disk_gib": disk_gb, "act_mb": act_mb, "bare_tps": bare_tps, "mtp_tps": mtp_tps,
                    "mean_kld": mean_kld, "p95_kld": p95_kld, "ppl": ppl, "dppl": ppl - bf16_ppl,
                    "top1_pct": t1, "top5_pct": t5,
                }
            if f_lm not in qlm_cache:
                del qlm_w
                torch.cuda.empty_cache()

        pareto_rows = []
        for name, _ in active_specs[1:]:
            row = results_by_name[name]
            pareto_rows.append(row)
            print(f"{name:20s} | {row['disk_gib']:5.1f}G | {row['act_mb']:5.0f}M | {row['bare_tps']:5.1f}t/s | {row['mtp_tps']:5.1f}t/s | {row['mean_kld']:9.6f} | {row['p95_kld']:9.6f} | {row['ppl']:7.3f} | {row['dppl']:+7.3f} | {row['top1_pct']:5.1f}% | {row['top5_pct']:5.1f}%")

        del lm_w, qlm_cache, fhcd_cache, fhcu_cache, states, f_wdf, f_wu, f_hn, f_bf16_u, dummy_inj
        torch.cuda.empty_cache()

    prefetch_pool.shutdown()
    reader.close()
    return {
        "label": label,
        "path": model_dir,
        "bf16_ppl": bf16_ppl,
        "peak_vram_gib": torch.cuda.max_memory_allocated(dev) / (1024 ** 3),
        "heatmap": heatmap,
        "top15": top15,
        "pareto": pareto_rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        default=[
            "base=/mnt/nvme2/Qwen-3.8-FN",
            "swift15=/mnt/nvme2/Swift-1.5-Qwen-3.8-FN",
            "thinking=/mnt/nvme2/Thinking-Cap-Qwen-3.8-FN",
        ],
    )
    ap.add_argument("--mode", choices=["self-test", "baseline", "recipes", "pareto"], default="pareto")
    ap.add_argument("--preset", choices=["default", "sniper", "barter", "solver_b10"], default="default")
    ap.add_argument("--expert-chunk", type=int, default=0, help="Experts per chunk (0 -> 64 for pareto, 128 for recipes/baseline)")
    ap.add_argument("--save-json", default="")
    args = ap.parse_args()

    models = []
    for item in args.models:
        assert "=" in item, f"Expected name=/path, got {item}"
        k, v = item.split("=", 1)
        models.append((k, v))

    if args.mode == "self-test":
        run_self_test(models[0][1])
        return

    recipes = {"default": RECIPES, "sniper": SNIPER_RECIPES, "barter": BARTER_RECIPES, "solver_b10": SOLVER_VERIFY_RECIPES}[args.preset]
    expert_chunk = args.expert_chunk if args.expert_chunk > 0 else (64 if args.mode == "pareto" else 128)

    dev = torch.device("cuda:0")
    _ = torch.zeros(1, device=dev)
    torch.cuda.synchronize()

    tok = QwenTokenizer(os.path.join(models[0][1], "tokenizer.json"))
    ids = tok.encode(CALIB_DOC)
    print(f"Shared calibration sequence: T = {len(ids)} tokens (head={tok.decode(ids[:12])!r}) | preset={args.preset}")

    all_res = []
    for label, path in models:
        all_res.append(simulate_model(label, path, ids, args.mode, recipes=recipes, expert_chunk=expert_chunk))

    if len(all_res) > 1 and args.mode in ("pareto", "recipes"):
        print("\n=== CROSS-CHECKPOINT COMPARISON SUMMARY (Mean KLD / dPPL / Top-1%) ===")
        hdr = f"{'Recipe':18s} | " + " | ".join(f"{r['label']:26s}" for r in all_res)
        print(hdr)
        print("-" * len(hdr))
        for ri in range(len(recipes)):
            rname = recipes[ri][0]
            cells = []
            for res in all_res:
                row = res["pareto"][ri]
                cells.append(f"{row['mean_kld']:.4f} / {row['dppl']:+5.2f} / {row['top1_pct']:5.1f}%")
            print(f"{rname:18s} | " + " | ".join(f"{c:26s}" for c in cells))

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(all_res, f, indent=2)
        print(f"\nSaved full JSON results to {args.save_json}")


if __name__ == "__main__":
    main()
