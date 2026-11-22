#!/usr/bin/env python3
"""
strix_moe_surgery.py — Zero-Retraining Edge-REAP Checkpoint & Router Surgery
for DeepSeek-V4.1-Flash (-1017) and Qwen3-Coder-480B-A35B, with Wave32 Q4P16
Panel Export and llama.cpp (b11870+) GGUF Conversion Drivers.

Designed for 4-Node AMD Strix Halo (gfx1151) Calibration -> 1- or 2-Node Deployment:
  1. PyTorch EP=4 / DDP Forward-Hook Observers:
     - DeepSeek-V4.1-Flash (`deepseek_v41`): exact `sqrtsoftplus` + `noaux_tc`
       modality-aware routing (`gate.bias` for text, `gate.bias_vl` for vision),
       `norm_topk_prob=True`, `routed_scaling_factor=1.5`.
     - Qwen3-Coder-480B-A35B (`qwen3_moe`): exact `softmax` -> `topk(8)` ->
       renormalize (`norm_topk_prob=True`), zero-shared-expert Super-Expert guard.
  2. Keep-Set Solver (`compute_edge_reap_keepset`):
     - Combines `toastydoc/Edge-MoE` Core + Domain pack union with Cerebras REAP
       gate-weighted output L2 norm saliency:
         S_EdgeREAP(j) = (sum_{t in T_j} g_j(t) * ||f_j(t)||_2 / |T_j|) * (1 - exp(-|T_j| / tau))
     - Pins Super-Experts (outlier L2 norm guard against pos [0, 64) drift),
       general-core experts (mandatory when `shared_experts == 0` on Qwen3-Coder),
       and optional top `gate.bias_vl` experts (preserving DeepSeek-ViT routing).
  3. Zero-Copy Raw-Byte Safetensors Shard Slicer (`slice_checkpoint`):
     - Operates directly on `.safetensors` headers + mmap byte slices without
       deserializing exotic dtypes (`F8_E4M3`, `U8` UE8M0, `I8` MXFP4, `BF16`).
     - Copies surviving routed experts byte-for-byte (renumbered 0 .. E_kept-1),
       slices dim-0 rows of `gate.weight`, `gate.bias`, and `gate.bias_vl` (plus
       3D fused expert tensors if present), leaves DeepSeek-V4.1 DSpark/MTP
       layers (40..42, 128 experts) and Engram shards (47..48) untouched, and
       verifies byte-exactness on readback.
  4. Post-Surgery Format Exporters (`gfx1151` RDNA 3.5 Target):
     - Vectorized OCP MXFP4 (`E2M1` + `UE8M0` block-32) and FP8 block (`128x128`
       `weight_scale_inv` / `32x32` `UE8M0`) dequantizers.
     - Streaming `Q4P16` Wave32 panel packer (16-row panels interleaved across
       32 Wave32 lanes with FP16 group-32 scales + bit-exact reference matvec).
     - `llama.cpp` (`>= b11870` / `b12549`) GGUF conversion + `--imatrix` driver,
       including DeepSeek-V4.1 `bias_vl` preservation, router `F32` overrides,
       and shard-6 NVMe lazy-mmap Engram linking.

References:
  - Edge-MoE: https://github.com/toastydoc/Edge-MoE
  - REAP (ICLR 2026): https://arxiv.org/abs/2510.13999 | https://github.com/CerebrasResearch/reap
  - DeepSeek-V4.1-Flash REAP192 layout: https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4
  - DeepSeek-V4.1-Flash MXFP4 spec: https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4
  - DeepSeek-V4.1-Flash GGUF b11870+: https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF
  - Qwen3-Coder-480B-A35B config: https://huggingface.co/cerebras/Qwen3-Coder-REAP-363B-A35B
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import mmap
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# ==============================================================================
# 1. SAFETENSORS ZERO-COPY BINARY READER / WRITER (DTYPE-AGNOSTIC)
# ==============================================================================

DTYPE_BYTE_SIZE: Dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass
class TensorEntry:
    dtype: str
    shape: List[int]
    data_offsets: Tuple[int, int]

    @property
    def nbytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]


class RawSafetensorsReader:
    """Memory-mapped safetensors reader supporting BF16, F8_E4M3, MXFP4 I8/U8 without PyTorch."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fh = open(self.path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        (header_len,) = struct.unpack("<Q", self._mm[:8])
        raw_header = json.loads(self._mm[8 : 8 + header_len].decode("utf-8"))
        self.data_base = 8 + header_len
        self.metadata: Dict[str, str] = raw_header.pop("__metadata__", {})
        self.tensors: Dict[str, TensorEntry] = {
            name: TensorEntry(
                dtype=info["dtype"],
                shape=list(info["shape"]),
                data_offsets=(int(info["data_offsets"][0]), int(info["data_offsets"][1])),
            )
            for name, info in raw_header.items()
        }

    def close(self) -> None:
        self._mm.close()
        self._fh.close()

    def __enter__(self) -> "RawSafetensorsReader":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def raw_bytes(self, name: str) -> memoryview:
        entry = self.tensors[name]
        start = self.data_base + entry.data_offsets[0]
        end = self.data_base + entry.data_offsets[1]
        return memoryview(self._mm)[start:end]

    def slice_dim0_bytes(self, name: str, kept_ids: Sequence[int]) -> Tuple[bytes, List[int]]:
        """Slices axis 0 of any tensor (BF16, F32, F8_E4M3, I8, U8) bit-exactly."""
        entry = self.tensors[name]
        if not entry.shape:
            raise ValueError(f"Cannot slice scalar tensor {name}")
        dim0 = entry.shape[0]
        if entry.nbytes % dim0 != 0:
            raise ValueError(f"Tensor {name} byte size {entry.nbytes} not divisible by dim0={dim0}")
        row_bytes = entry.nbytes // dim0
        src = self.raw_bytes(name)
        out = bytearray(len(kept_ids) * row_bytes)
        for out_idx, src_idx in enumerate(kept_ids):
            if src_idx < 0 or src_idx >= dim0:
                raise IndexError(f"Expert index {src_idx} out of bounds for {name} (dim0={dim0})")
            s0 = src_idx * row_bytes
            d0 = out_idx * row_bytes
            out[d0 : d0 + row_bytes] = src[s0 : s0 + row_bytes]
        src.release()
        new_shape = [len(kept_ids)] + entry.shape[1:]
        return bytes(out), new_shape

    def to_float32(self, name: str) -> np.ndarray:
        """Converts 1D/2D/3D BF16, F16, F32, or F8_E4M3 tensors to FP32 numpy array."""
        entry = self.tensors[name]
        buf = self.raw_bytes(name)
        if entry.dtype == "F32":
            res = np.frombuffer(buf, dtype=np.float32).reshape(entry.shape).copy()
            buf.release()
            return res
        if entry.dtype == "F16":
            res = np.frombuffer(buf, dtype=np.float16).astype(np.float32).reshape(entry.shape)
            buf.release()
            return res
        if entry.dtype == "BF16":
            u16 = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
            buf.release()
            return u16.view(np.float32).reshape(entry.shape)
        if entry.dtype == "F8_E4M3":
            res = decode_fp8_e4m3fn(np.frombuffer(buf, dtype=np.uint8)).reshape(entry.shape)
            buf.release()
            return res
        raise TypeError(f"Unsupported float conversion dtype {entry.dtype} on {name}")


def write_safetensors_file(
    out_path: Path,
    tensors: Sequence[Tuple[str, str, List[int], UnionBytes]],
    metadata: Optional[Dict[str, str]] = None,
) -> int:
    """
    Streams an ordered list of (name, dtype, shape, raw_bytes_or_memoryview) into a
    64-byte-aligned header safetensors shard and returns total tensor payload bytes.
    """
    header_obj: Dict[str, Any] = {}
    if metadata:
        header_obj["__metadata__"] = dict(metadata)

    offset = 0
    for name, dtype, shape, data in tensors:
        nbytes = len(data)
        expected = int(np.prod(shape, dtype=np.int64)) * DTYPE_BYTE_SIZE[dtype] if shape else DTYPE_BYTE_SIZE[dtype]
        if nbytes != expected:
            raise ValueError(
                f"Byte length mismatch writing {name}: got {nbytes}, expected {expected} for {dtype} {shape}"
            )
        header_obj[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes

    header_bytes = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")
    # Pad JSON header with spaces to 8-byte alignment per safetensors convention
    pad = (8 - (len(header_bytes) % 8)) % 8
    if pad:
        header_bytes += b" " * pad

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        for _, _, _, data in tensors:
            fh.write(data)
    return offset


UnionBytes = Any  # bytes | bytearray | memoryview


def float32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    """Round-to-nearest-even FP32 -> BF16 raw bytes."""
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32)
    rounding_bias = ((u32 >> 16) & 1) + np.uint32(0x7FFF)
    bf16_u16 = ((u32 + rounding_bias) >> 16).astype(np.uint16)
    return bf16_u16.tobytes()


# ==============================================================================
# 2. MODEL ARCHITECTURE SPECS & REGEX PATTERNS (DS-V4.1-FLASH & QWEN3-CODER-480B)
# ==============================================================================

# DeepSeek-V4.1-Flash (-1017):
#   - 40 CED MoE layers (0..39), 384 routed + 1 shared expert, top-6
#   - 3 DSpark/MTP speculative layers (40..42), 128 routed experts -> DO NOT PRUNE!
#   - 2 Engram shards (layers.1.engram.embed, layers.14.engram.embed) -> KEEP / SYMLINK!
#   - Routed experts: `layers.{l}.ffn.experts.{e}.{w1|w2|w3}` (+ `.scale`, `.weight`, etc.)
#   - Router tensors: `layers.{l}.ffn.gate.weight` [384, 5120], `gate.bias` [384], `gate.bias_vl` [384]
DSV41_EXPERT_RE = re.compile(
    r"^((?:language_model\.)?(?:model\.)?layers\.(\d+)\.(?:ffn|mlp)\.experts\.)(\d+)(\.(?:w1|w2|w3|gate_proj|up_proj|down_proj)(?:\.[A-Za-z0-9_]+)?)$"
)
DSV41_FUSED_EXPERT_RE = re.compile(
    r"^((?:language_model\.)?(?:model\.)?layers\.(\d+)\.(?:ffn|mlp)\.experts\.(?:gate_up_proj|down_proj|w13_weight|w2_weight)(?:\.[A-Za-z0-9_]+)?)$"
)
DSV41_ROUTER_RE = re.compile(
    r"^((?:language_model\.)?(?:model\.)?layers\.(\d+)\.(?:ffn|mlp)\.gate\.(?:weight|bias|bias_vl|e_score_correction_bias))$"
)

# Qwen3-Coder-480B-A35B (-Instruct & -FP8):
#   - 62 MoE layers (0..61), 160 routed + 0 shared experts, top-8
#   - Unfused experts: `model.layers.{l}.mlp.experts.{e}.{gate_proj|up_proj|down_proj}.{weight|weight_scale_inv|weight_scale|...}`
#   - Fused 3D fallback: `model.layers.{l}.mlp.experts.{gate_up_proj|down_proj|w13_weight|w2_weight}...`
#   - Router tensor: `model.layers.{l}.mlp.gate.weight` [160, 6144]
QWEN3_EXPERT_RE = re.compile(
    r"^((?:model\.)?layers\.(\d+)\.mlp\.experts\.)(\d+)(\.(?:gate_proj|up_proj|down_proj)\.[A-Za-z0-9_]+)$"
)
QWEN3_FUSED_EXPERT_RE = re.compile(
    r"^((?:model\.)?layers\.(\d+)\.mlp\.experts\.(?:gate_up_proj|down_proj|w13_weight|w2_weight)[A-Za-z0-9_.]*)$"
)
QWEN3_ROUTER_RE = re.compile(
    r"^((?:model\.)?layers\.(\d+)\.mlp\.gate\.(?:weight|bias))$"
)


@dataclass
class ModelSurgeryProfile:
    arch: str  # "deepseek_v41" | "qwen3_moe"
    num_moe_layers: int
    num_routed_experts: int
    num_shared_experts: int
    top_k: int
    expert_re: re.Pattern[str]
    fused_expert_re: re.Pattern[str]
    router_re: re.Pattern[str]
    engram_shards: Tuple[str, ...] = ()

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ModelSurgeryProfile":
        model_type = config.get("model_type", "")
        archs = config.get("architectures", [])
        if model_type == "deepseek_v41" or "DeepseekV41ForCausalLM" in archs:
            tc = config.get("text_config", config)
            return cls(
                arch="deepseek_v41",
                num_moe_layers=int(tc["num_hidden_layers"]),  # 40 (excludes 3 DSpark layers 40..42!)
                num_routed_experts=int(tc["n_routed_experts"]),  # 384
                num_shared_experts=int(tc.get("n_shared_experts", 1)),
                top_k=int(tc["num_experts_per_tok"]),  # 6
                expert_re=DSV41_EXPERT_RE,
                fused_expert_re=DSV41_FUSED_EXPERT_RE,
                router_re=DSV41_ROUTER_RE,
                engram_shards=(
                    "model-00047-of-00048.safetensors",
                    "model-00048-of-00048.safetensors",
                ),
            )
        if model_type == "qwen3_moe" or "Qwen3MoeForCausalLM" in archs:
            return cls(
                arch="qwen3_moe",
                num_moe_layers=int(config["num_hidden_layers"]),  # 62
                num_routed_experts=int(config["num_experts"]),  # 160
                num_shared_experts=1 if int(config.get("shared_expert_intermediate_size", 0)) > 0 else 0,
                top_k=int(config["num_experts_per_tok"]),  # 8
                expert_re=QWEN3_EXPERT_RE,
                fused_expert_re=QWEN3_FUSED_EXPERT_RE,
                router_re=QWEN3_ROUTER_RE,
            )
        raise ValueError(f"Unsupported model_type={model_type!r}, architectures={archs!r}")


# ==============================================================================
# 3. EXACT ROUTER MATH & 4-NODE EP=4 / DDP PYTORCH CALIBRATION HOOKS
# ==============================================================================

def dsv41_route_numpy(
    hidden_states: np.ndarray,
    gate_weight: np.ndarray,
    gate_bias: np.ndarray,
    gate_bias_vl: Optional[np.ndarray] = None,
    is_vision_token: Optional[np.ndarray] = None,
    top_k: int = 6,
    routed_scaling_factor: float = 1.5,
    norm_topk_prob: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact DeepSeek-V4.1-Flash (-1017) router (`scoring_func="sqrtsoftplus"`, `topk_method="noaux_tc"`).
    - Selection adds modality-specific bias (`gate.bias` for text, `gate.bias_vl` for image tokens).
    - Gate weights gather the UNBIASED `sqrtsoftplus` scores on top-k, normalize to sum=1, and scale by 1.5.
    """
    logits = hidden_states.astype(np.float32) @ gate_weight.astype(np.float32).T  # [T, E]
    # Numerically stable softplus(z) = log(1 + exp(-|z|)) + max(z, 0)
    softplus = np.log1p(np.exp(-np.abs(logits))) + np.maximum(logits, 0.0)
    raw_scores = np.sqrt(np.maximum(softplus, 0.0))  # sqrtsoftplus

    bias = gate_bias.astype(np.float32)[None, :]  # [1, E]
    if gate_bias_vl is not None and is_vision_token is not None:
        vl_mask = is_vision_token.astype(bool)[:, None]
        bias = np.where(vl_mask, gate_bias_vl.astype(np.float32)[None, :], bias)

    sel_scores = raw_scores + bias
    # Select top-k expert indices per token
    topk_idx = np.argpartition(-sel_scores, kth=top_k - 1, axis=-1)[:, :top_k]
    row_idx = np.arange(hidden_states.shape[0])[:, None]
    # Sort top-k deterministically by selection score descending
    order = np.argsort(-sel_scores[row_idx, topk_idx], axis=-1)
    topk_idx = topk_idx[row_idx, order]

    topk_weights = raw_scores[row_idx, topk_idx]
    if norm_topk_prob:
        denom = np.maximum(topk_weights.sum(axis=-1, keepdims=True), 1e-20)
        topk_weights = topk_weights / denom
    topk_weights = topk_weights * float(routed_scaling_factor)
    return topk_idx, topk_weights


def qwen3_moe_route_numpy(
    hidden_states: np.ndarray,
    gate_weight: np.ndarray,
    top_k: int = 8,
    norm_topk_prob: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact Qwen3-Coder-480B-A35B router (`softmax` over all E -> `topk(8)` -> renormalize).
    Note: global softmax denominator cancels out upon top-k renormalization!
    """
    logits = hidden_states.astype(np.float32) @ gate_weight.astype(np.float32).T
    logits = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=-1, keepdims=True)

    topk_idx = np.argpartition(-probs, kth=top_k - 1, axis=-1)[:, :top_k]
    row_idx = np.arange(hidden_states.shape[0])[:, None]
    order = np.argsort(-probs[row_idx, topk_idx], axis=-1)
    topk_idx = topk_idx[row_idx, order]

    topk_weights = probs[row_idx, topk_idx]
    if norm_topk_prob:
        topk_weights = topk_weights / np.maximum(topk_weights.sum(axis=-1, keepdims=True), 1e-20)
    return topk_idx, topk_weights


class DistributedEdgeREAPCollector:
    """
    PyTorch forward-hook accumulator for 4-Node Strix Halo RCCL RoCEv2 calibration
    (under EP=4 expert parallelism or 4-rank DDP sequential onloading).

    IMPORTANT TP=4 NOTE:
    Do NOT attach this hook to row-parallel TP=4 `down_proj` shards without first
    all-reducing per-expert outputs `[tokens, top_k, hidden_size]`, because
    `sum_r ||f_j^(r)(t)||_2 != ||sum_r f_j^(r)(t)||_2`. Under EP=4 or DDP sequential
    onloading, each rank computes the full unsharded expert output `f_j(t)` directly.
    """

    def __init__(self, num_layers: int, num_experts: int):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.reap_weighted_norm_sum = np.zeros((num_layers, num_experts), dtype=np.float64)
        self.gate_mass_sum = np.zeros((num_layers, num_experts), dtype=np.float64)
        self.token_count = np.zeros((num_layers, num_experts), dtype=np.int64)
        self.max_l2_norm = np.zeros((num_layers, num_experts), dtype=np.float64)
        self.early_pos_count = np.zeros((num_layers, num_experts), dtype=np.int64)

    def record_batch(
        self,
        layer_id: int,
        topk_indices: np.ndarray,     # [T, K] int64
        topk_gate_weights: np.ndarray,  # [T, K] float32 (post-renorm, post-scaling)
        expert_l2_norms: np.ndarray,  # [T, K] float32 (exact ||f_j(t)||_2 of unsharded expert)
        token_positions: Optional[np.ndarray] = None,  # [T] int64 sequence positions
    ) -> None:
        flat_e = topk_indices.reshape(-1)
        flat_g = topk_gate_weights.reshape(-1).astype(np.float64)
        flat_n = expert_l2_norms.reshape(-1).astype(np.float64)

        np.add.at(self.reap_weighted_norm_sum[layer_id], flat_e, flat_g * flat_n)
        np.add.at(self.gate_mass_sum[layer_id], flat_e, flat_g)
        np.add.at(self.token_count[layer_id], flat_e, 1)
        np.maximum.at(self.max_l2_norm[layer_id], flat_e, flat_n)

        if token_positions is not None:
            early_mask = (token_positions < 64)[:, None]
            early_e = topk_indices[np.broadcast_to(early_mask, topk_indices.shape)]
            np.add.at(self.early_pos_count[layer_id], early_e, 1)

    def all_reduce_torch_distributed(self) -> None:
        """Synchronizes calibration statistics across 4 Strix Halo nodes over RCCL RoCEv2."""
        import torch
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for attr, op in (
            ("reap_weighted_norm_sum", dist.ReduceOp.SUM),
            ("gate_mass_sum", dist.ReduceOp.SUM),
            ("token_count", dist.ReduceOp.SUM),
            ("early_pos_count", dist.ReduceOp.SUM),
            ("max_l2_norm", dist.ReduceOp.MAX),
        ):
            arr = getattr(self, attr)
            t = torch.from_numpy(arr).to(device)
            dist.all_reduce(t, op=op)
            setattr(self, attr, t.cpu().numpy())

    def export_report(self, path: Path, model_id: str) -> None:
        saliency = np.divide(
            self.reap_weighted_norm_sum,
            np.maximum(self.token_count, 1),
            dtype=np.float64,
        )
        payload = {
            "model_id": model_id,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "reap_saliency": saliency.tolist(),
            "reap_weighted_norm_sum": self.reap_weighted_norm_sum.tolist(),
            "gate_mass_sum": self.gate_mass_sum.tolist(),
            "token_count": self.token_count.tolist(),
            "max_l2_norm": self.max_l2_norm.tolist(),
            "early_pos_count": self.early_pos_count.tolist(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ==============================================================================
# 4. HYBRID EDGE-REAP KEEP-SET SOLVER (CORE + SUPER-EXPERT + VL + DOMAIN PACK)
# ==============================================================================

def compute_edge_reap_keepset(
    domain_report: Mapping[str, Any],
    keep_experts: int,
    core_report: Optional[Mapping[str, Any]] = None,
    pin_core_experts: int = 16,
    pin_super_experts: int = 4,
    pin_early_pos_experts: int = 4,
    vl_biases_by_layer: Optional[Mapping[int, np.ndarray]] = None,
    pin_vl_experts: int = 0,
    tau_domain: float = 64.0,
) -> Dict[int, List[int]]:
    """
    Computes deterministic sorted keep-sets `0 .. E_kept - 1` for every MoE layer:
      1. Pins `pin_super_experts` with highest peak output L2 norm (prevents outlier collapse).
      2. Pins `pin_early_pos_experts` with highest routing count on token positions [0, 64)
         (prevents prompt-prefix KL drift documented in DeepSeek-V4.1 REAP192).
      3. Pins `pin_core_experts` from `core_report` (or general gate_mass_sum), critical
         when `shared_experts == 0` on Qwen3-Coder-480B-A35B.
      4. Optionally pins `pin_vl_experts` with highest `gate.bias_vl` on DeepSeek-V4.1
         if calibration lacked image tokens.
      5. Fills remaining budget up to `keep_experts` via Edge-REAP score:
           S_EdgeREAP(j) = S_REAP(j) * (1 - exp(-count(j) / tau_domain))
    """
    num_layers = int(domain_report["num_layers"])
    num_experts = int(domain_report["num_experts"])
    if keep_experts <= 0 or keep_experts > num_experts:
        raise ValueError(f"keep_experts={keep_experts} must be in 1..{num_experts}")

    reap_sal = np.asarray(domain_report["reap_saliency"], dtype=np.float64)
    counts = np.asarray(domain_report["token_count"], dtype=np.float64)
    max_l2 = np.asarray(
        domain_report.get("max_l2_norm", np.zeros_like(reap_sal)), dtype=np.float64
    )
    early_counts = np.asarray(
        domain_report.get("early_pos_count", np.zeros_like(reap_sal)), dtype=np.float64
    )

    if core_report is not None:
        core_mass = np.asarray(
            core_report.get("gate_mass_sum", core_report["token_count"]), dtype=np.float64
        )
    else:
        core_mass = np.asarray(
            domain_report.get("gate_mass_sum", counts), dtype=np.float64
        )

    keepset: Dict[int, List[int]] = {}
    ids = np.arange(num_experts)

    for layer in range(num_layers):
        pinned: set[int] = set()

        def _add_top(metric: np.ndarray, k: int) -> None:
            if k <= 0:
                return
            # Deterministic tie-break: higher metric first, lower expert ID on exact ties
            order = np.lexsort((ids, -metric))
            added = 0
            for idx in order:
                if len(pinned) >= keep_experts or added >= k:
                    break
                if int(idx) not in pinned:
                    pinned.add(int(idx))
                    added += 1

        _add_top(max_l2[layer], pin_super_experts)
        _add_top(early_counts[layer], pin_early_pos_experts)
        _add_top(core_mass[layer], pin_core_experts)

        if vl_biases_by_layer is not None and pin_vl_experts > 0 and layer in vl_biases_by_layer:
            _add_top(np.asarray(vl_biases_by_layer[layer], dtype=np.float64), pin_vl_experts)

        # Edge-REAP domain saliency with smooth zero-hit suppression
        domain_factor = 1.0 - np.exp(-counts[layer] / max(tau_domain, 1.0))
        edge_reap_score = reap_sal[layer] * domain_factor
        order = np.lexsort((ids, -edge_reap_score))
        for idx in order:
            if len(pinned) >= keep_experts:
                break
            pinned.add(int(idx))

        # Renumber in ascending original expert ID order (preserves relative tie-break order)
        keepset[layer] = sorted(pinned)

    return keepset


# ==============================================================================
# 5. STREAMING SHARD-BY-SHARD CHECKPOINT & ROUTER SURGERY + READBACK VERIFY
# ==============================================================================

@dataclass
class SurgerySummary:
    arch: str
    orig_experts: int
    kept_experts: int
    shards_written: int
    engram_shards_linked: int
    total_output_bytes: int
    keepset_sha256: str


def slice_checkpoint(
    src_dir: Path,
    out_dir: Path,
    keepset: Mapping[int, Sequence[int]],
    symlink_engram: bool = True,
    verify_readback: bool = True,
) -> SurgerySummary:
    """
    Slices a multi-shard (or single-file) HuggingFace safetensors checkpoint for
    `DeepSeek-V4.1-Flash` or `Qwen3-Coder-480B-A35B` in streaming shard-by-shard mode.
    """
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = src_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    profile = ModelSurgeryProfile.from_config(config)

    # Validate uniform expert count across all MoE layers (required by standard config.json / GGUF)
    kept_counts = {len(v) for v in keepset.values()}
    if len(kept_counts) != 1:
        raise ValueError(f"Uniform pruning expects identical E_kept per layer, got {kept_counts}")
    kept_e = next(iter(kept_counts))
    if kept_e < profile.top_k:
        raise ValueError(f"kept_experts ({kept_e}) cannot be smaller than num_experts_per_tok ({profile.top_k})")

    # Build O(1) old_expert_id -> new_expert_id maps for layers 0 .. num_moe_layers - 1
    layer_remap: Dict[int, Dict[int, int]] = {}
    for layer_id in range(profile.num_moe_layers):
        if layer_id not in keepset:
            raise KeyError(f"Missing layer {layer_id} in keepset (expected 0..{profile.num_moe_layers - 1})")
        ordered = list(keepset[layer_id])
        if ordered != sorted(ordered) or len(set(ordered)) != kept_e:
            raise ValueError(f"Layer {layer_id} keepset must be strictly ascending unique IDs")
        layer_remap[layer_id] = {orig_e: new_e for new_e, orig_e in enumerate(ordered)}

    # Discover shards (supports sharded `model.safetensors.index.json` or single `model.safetensors`)
    index_path = src_dir / "model.safetensors.index.json"
    if index_path.exists():
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        orig_weight_map: Dict[str, str] = index_data["weight_map"]
        shard_Grouping: Dict[str, List[str]] = {}
        for t_name, shard_name in orig_weight_map.items():
            shard_Grouping.setdefault(shard_name, []).append(t_name)
        is_sharded = True
    else:
        single = "model.safetensors"
        with RawSafetensorsReader(src_dir / single) as rdr:
            shard_Grouping = {single: list(rdr.tensors.keys())}
        is_sharded = False

    new_weight_map: Dict[str, str] = {}
    total_output_bytes = 0
    shards_written = 0
    engram_shards_linked = 0

    for shard_name in sorted(shard_Grouping.keys()):
        src_shard = src_dir / shard_name
        dst_shard = out_dir / shard_name

        # DeepSeek-V4.1-Flash Engram shards (shards 47 & 48, ~94.6 GiB each): never load into RAM!
        if shard_name in profile.engram_shards or all(".engram." in k for k in shard_Grouping[shard_name]):
            for t_name in shard_Grouping[shard_name]:
                new_weight_map[t_name] = shard_name
            if src_shard.exists():
                total_output_bytes += src_shard.stat().st_size
                if symlink_engram:
                    if dst_shard.exists() or dst_shard.is_symlink():
                        dst_shard.unlink()
                    dst_shard.symlink_to(src_shard.resolve())
                    engram_shards_linked += 1
            continue

        with RawSafetensorsReader(src_shard) as reader:
            out_tensors: List[Tuple[str, str, List[int], UnionBytes]] = []
            verify_plan: List[Tuple[str, str, Optional[List[int]]]] = []  # (new_name, old_name, slice_ids)

            for name in reader.tensors:
                entry = reader.tensors[name]

                # Case A: Per-expert tensor (unfused MXFP4 / FP8 / BF16 + companion .scale / .weight_scale_inv)
                m_exp = profile.expert_re.match(name)
                if m_exp:
                    prefix, layer_str, exp_str, suffix = m_exp.groups()
                    layer_id = int(layer_str)
                    orig_e = int(exp_str)
                    # Guard: Do NOT prune DeepSeek-V4.1 DSpark/MTP speculative layers (layer_id >= 40)!
                    if layer_id >= profile.num_moe_layers:
                        out_tensors.append((name, entry.dtype, entry.shape, reader.raw_bytes(name)))
                        new_weight_map[name] = shard_name
                        verify_plan.append((name, name, None))
                        continue

                    remap = layer_remap[layer_id]
                    if orig_e not in remap:
                        continue  # Pruned expert -> drop tensor and its scale!
                    new_e = remap[orig_e]
                    new_name = f"{prefix}{new_e}{suffix}"
                    out_tensors.append((new_name, entry.dtype, entry.shape, reader.raw_bytes(name)))
                    new_weight_map[new_name] = shard_name
                    verify_plan.append((new_name, name, None))
                    continue

                # Case B: Fused 3D expert tensor `[num_experts, ...]`
                m_fused = profile.fused_expert_re.match(name)
                if m_fused:
                    _, layer_str = m_fused.groups()
                    layer_id = int(layer_str)
                    if layer_id < profile.num_moe_layers and entry.shape and entry.shape[0] == profile.num_routed_experts:
                        ids = list(keepset[layer_id])
                        sliced_bytes, new_shape = reader.slice_dim0_bytes(name, ids)
                        out_tensors.append((name, entry.dtype, new_shape, sliced_bytes))
                        new_weight_map[name] = shard_name
                        verify_plan.append((name, name, ids))
                        continue

                # Case C: Router gate weights and biases (`gate.weight`, `gate.bias`, `gate.bias_vl`)
                m_router = profile.router_re.match(name)
                if m_router:
                    _, layer_str = m_router.groups()
                    layer_id = int(layer_str)
                    if layer_id < profile.num_moe_layers and entry.shape and entry.shape[0] == profile.num_routed_experts:
                        ids = list(keepset[layer_id])
                        sliced_bytes, new_shape = reader.slice_dim0_bytes(name, ids)
                        out_tensors.append((name, entry.dtype, new_shape, sliced_bytes))
                        new_weight_map[name] = shard_name
                        verify_plan.append((name, name, ids))
                        continue

                # Case D: Shared experts, CSA2/GQA attention, mHC, ViT vision tower, embeddings, norms -> untouched
                out_tensors.append((name, entry.dtype, entry.shape, reader.raw_bytes(name)))
                new_weight_map[name] = shard_name
                verify_plan.append((name, name, None))

            if out_tensors:
                # Sort renumbered tensors cleanly so `experts.0 .. experts.{E_kept-1}` stay contiguous
                out_tensors.sort(key=lambda item: _natural_tensor_sort_key(item[0]))
                shard_bytes = write_safetensors_file(dst_shard, out_tensors, metadata=reader.metadata)
                total_output_bytes += shard_bytes
                shards_written += 1

                if verify_readback:
                    with RawSafetensorsReader(dst_shard) as check_rdr:
                        for new_name, old_name, slice_ids in verify_plan:
                            got = bytes(check_rdr.raw_bytes(new_name))
                            if slice_ids is None:
                                exp = bytes(reader.raw_bytes(old_name))
                            else:
                                exp, _ = reader.slice_dim0_bytes(old_name, slice_ids)
                            if got != exp:
                                raise RuntimeError(f"Bit-exact readback verification FAILED on {new_name} in {shard_name}")
                        del got, exp
            out_tensors.clear()
            del out_tensors

    # Update config.json (and DeepSeek `inference/config*.json` if bundled)
    new_config = json.loads(json.dumps(config))
    if profile.arch == "deepseek_v41":
        if "text_config" in new_config:
            new_config["text_config"]["n_routed_experts"] = kept_e
        else:
            new_config["n_routed_experts"] = kept_e
    elif profile.arch == "qwen3_moe":
        new_config["num_experts"] = kept_e

    (out_dir / "config.json").write_text(json.dumps(new_config, indent=2) + "\n", encoding="utf-8")

    if is_sharded:
        new_index = {
            "metadata": {"total_size": total_output_bytes},
            "weight_map": dict(sorted(new_weight_map.items(), key=lambda kv: _natural_tensor_sort_key(kv[0]))),
        }
        (out_dir / "model.safetensors.index.json").write_text(
            json.dumps(new_index, indent=2) + "\n", encoding="utf-8"
        )

    # Copy tokenizer, generation_config, chat_template, and DeepSeek `encoding/` folder
    _copy_auxiliary_repo_files(src_dir, out_dir, kept_e)

    # Write deterministic `reap_keepset.json` provenance manifest
    keepset_json = json.dumps(
        {str(k): list(v) for k, v in sorted(keepset.items())}, separators=(",", ":")
    ).encode("utf-8")
    keepset_sha = hashlib.sha256(keepset_json).hexdigest()
    (out_dir / "reap_keepset.json").write_text(
        json.dumps(
            {
                "arch": profile.arch,
                "orig_routed_experts": profile.num_routed_experts,
                "kept_routed_experts": kept_e,
                "sha256": keepset_sha,
                "layers": {str(k): list(v) for k, v in sorted(keepset.items())},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return SurgerySummary(
        arch=profile.arch,
        orig_experts=profile.num_routed_experts,
        kept_experts=kept_e,
        shards_written=shards_written,
        engram_shards_linked=engram_shards_linked,
        total_output_bytes=total_output_bytes,
        keepset_sha256=keepset_sha,
    )


def _natural_tensor_sort_key(name: str) -> List[Any]:
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", name)]


def _copy_auxiliary_repo_files(src_dir: Path, out_dir: Path, kept_e: int) -> None:
    if src_dir.resolve() == out_dir.resolve():
        return
    skip_suffixes = {".safetensors", ".bin", ".pt", ".gguf"}
    skip_names = {"config.json", "model.safetensors.index.json", "reap_keepset.json"}
    for item in src_dir.iterdir():
        if item.name in skip_names or item.name.startswith("."):
            continue
        if item.is_file() and item.suffix not in skip_suffixes:
            shutil.copy2(item, out_dir / item.name)
        elif item.is_dir() and item.name in {"encoding", "inference"}:
            dst_sub = out_dir / item.name
            if dst_sub.exists():
                shutil.rmtree(dst_sub)
            shutil.copytree(item, dst_sub)
            # Patch n_routed_experts inside DeepSeek reference inference configs if present
            for cfg_file in dst_sub.rglob("config*.json"):
                try:
                    sub_cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
                    if "n_routed_experts" in sub_cfg:
                        sub_cfg["n_routed_experts"] = kept_e
                        cfg_file.write_text(json.dumps(sub_cfg, indent=2) + "\n", encoding="utf-8")
                except Exception:
                    pass


# ==============================================================================
# 6. OCP MXFP4 (E2M1 + UE8M0) & BLOCK-FP8 DEQUANTIZERS + WAVE32 Q4P16 EXPORTER
# ==============================================================================

# Exact OCP / DeepSeek-V4.1 E2M1 4-bit nibble lookup table (sign=bit3, exp=bits2..1, mant=bit0)
OCP_E2M1_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def _build_fp8_e4m3fn_lut() -> np.ndarray:
    """Exact IEEE/OCP float8_e4m3fn (bias=7, 0x7F/0xFF = NaN) 256-entry lookup table."""
    lut = np.zeros(256, dtype=np.float32)
    for b in range(256):
        sign = -1.0 if (b & 0x80) else 1.0
        exp = (b >> 3) & 0x0F
        mant = b & 0x07
        if exp == 0x0F and mant == 0x07:
            lut[b] = np.nan
        elif exp == 0:
            lut[b] = sign * (mant / 8.0) * (2.0 ** -6)
        else:
            lut[b] = sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
    return lut


FP8_E4M3FN_LUT = _build_fp8_e4m3fn_lut()


def decode_fp8_e4m3fn(raw_u8: np.ndarray) -> np.ndarray:
    return FP8_E4M3FN_LUT[raw_u8.view(np.uint8)]


def dequant_mxfp4_block32(
    packed_bytes: UnionBytes,
    ue8m0_scale_bytes: UnionBytes,
    out_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Reconstructs FP32 weights from DeepSeek-V4.1-Flash's native MXFP4 expert tensors:
      - `packed_bytes`: int8/uint8 array of shape [rows, cols // 2], 2 E2M1 nibbles per byte
        (low nibble = even element 2*i, high nibble = odd element 2*i + 1).
      - `ue8m0_scale_bytes`: uint8 UE8M0 power-of-two exponent per 32 elements (`2^(u - 127)`).
    """
    rows, cols = out_shape
    if cols % 32 != 0:
        raise ValueError(f"MXFP4 cols={cols} must be a multiple of 32")
    raw = np.frombuffer(packed_bytes, dtype=np.uint8).reshape(rows, cols // 2)
    lo = OCP_E2M1_LUT[raw & 0x0F]
    hi = OCP_E2M1_LUT[(raw >> 4) & 0x0F]
    vals = np.empty((rows, cols), dtype=np.float32)
    vals[:, 0::2] = lo
    vals[:, 1::2] = hi

    scales_u8 = np.frombuffer(ue8m0_scale_bytes, dtype=np.uint8).reshape(rows, cols // 32)
    scales_f32 = np.ldexp(np.ones_like(scales_u8, dtype=np.float32), scales_u8.astype(np.int32) - 127)
    vals = (vals.reshape(rows, cols // 32, 32) * scales_f32[:, :, None]).reshape(rows, cols)
    return vals


def dequant_fp8_block2d(
    fp8_bytes: UnionBytes,
    scale_arr: np.ndarray,
    weight_shape: Tuple[int, int],
    block_size: Tuple[int, int] = (128, 128),
    scale_is_ue8m0: bool = False,
) -> np.ndarray:
    """
    Dequantizes 2D block-scaled FP8 (`Qwen3-Coder-480B-A35B-Instruct-FP8` [128, 128]
    `weight_scale_inv` or `DeepSeek-V4.1-Flash` [32, 32] `UE8M0` dense weights) to FP32.
    """
    rows, cols = weight_shape
    br, bc = block_size
    w = decode_fp8_e4m3fn(np.frombuffer(fp8_bytes, dtype=np.uint8)).reshape(rows, cols)
    if scale_is_ue8m0:
        s = np.ldexp(np.ones_like(scale_arr, dtype=np.float32), scale_arr.astype(np.int32) - 127)
    else:
        s = scale_arr.astype(np.float32)

    # Per-channel (`[rows, 1]`) vs 2D block (`[ceil(rows/br), ceil(cols/bc)]`)
    if s.size == rows:
        return w * s.reshape(rows, 1)
    s_expanded = np.repeat(np.repeat(s, br, axis=0)[:rows, :], bc, axis=1)[:, :cols]
    return w * s_expanded


@dataclass(frozen=True)
class Q4P16Layout:
    """
    Wave32-coalesced 16-row Q4 panel layout targeting AMD RDNA 3.5 (`gfx1151`, WaveSize=32).
    NOTE: Adjust `MAGIC` or `VALUES_PER_LANE` if adapting header bytes to a custom
    `strixflerp` / `strixite` `format.flerp.q4panel` version.
    """
    MAGIC: bytes = b"Q4P16\x00\x01\x00"
    PANEL_ROWS: int = 16       # 16 output rows per panel
    WAVE_LANES: int = 32       # RDNA 3.5 native Wave32 wavefront
    GROUP_SIZE: int = 32       # Scale group along K


def pack_q4p16_panel_matrix(
    weight_f32: np.ndarray,
    layout: Q4P16Layout = Q4P16Layout(),
) -> Tuple[bytes, np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Packs a `[N, K]` matrix into Wave32 `Q4P16` panels:
      - Pads `N` to a multiple of 16 (`PANEL_ROWS`) and `K` to a multiple of 32 (`GROUP_SIZE`).
      - Symmetric 4-bit quantization per `(row, k_group)` block of 32 weights:
          scale = max(|x|) / 7.0  (stored in FP16)
          q = round(clip(x / scale, -8, 7)) + 8  in [0..15]
      - Interleaves each `[16 rows, 32 k-cols]` tile across the 32 Wave32 lanes so each lane
        reads 16 nibbles (8 contiguous bytes = `uint64_t`) covering `(row_pair, k_lane)`
        before a 5-step Wave32 `permlane` / `__shfl_xor` butterfly reduction along `K`.
    Returns `(binary_blob_with_header, packed_u8_panels, scales_f16, padded_shape)`.
    """
    if weight_f32.ndim != 2:
        raise ValueError(f"Q4P16 expects 2D [N, K] matrix, got {weight_f32.shape}")
    orig_n, orig_k = weight_f32.shape
    pad_n = (layout.PANEL_ROWS - (orig_n % layout.PANEL_ROWS)) % layout.PANEL_ROWS
    pad_k = (layout.GROUP_SIZE - (orig_k % layout.GROUP_SIZE)) % layout.GROUP_SIZE

    if pad_n or pad_k:
        w = np.pad(weight_f32.astype(np.float32), ((0, pad_n), (0, pad_k)))
    else:
        w = np.ascontiguousarray(weight_f32, dtype=np.float32)

    n_padded, k_padded = w.shape
    n_panels = n_padded // layout.PANEL_ROWS
    n_groups = k_padded // layout.GROUP_SIZE

    # Reshape into [n_panels, 16, n_groups, 32]
    tiles = w.reshape(n_panels, layout.PANEL_ROWS, n_groups, layout.GROUP_SIZE)
    amax = np.max(np.abs(tiles), axis=-1)  # [n_panels, 16, n_groups]
    scales_f16 = (amax / 7.0).astype(np.float16)
    safe_scales = np.where(scales_f16.astype(np.float32) > 0.0, scales_f16.astype(np.float32), 1.0)

    q_signed = np.clip(np.rint(tiles / safe_scales[..., None]), -8, 7).astype(np.int8)
    q_u4 = (q_signed + 8).astype(np.uint8)  # [n_panels, 16, n_groups, 32] in 0..15

    # Transpose to Wave32 lane-major order: [n_panels, n_groups, 32_lanes (K), 16_rows]
    lane_major = np.transpose(q_u4, (0, 2, 3, 1))
    # Pack pairs of rows (even row in low nibble, odd row in high nibble) -> [n_panels, n_groups, 32, 8]
    packed_u8 = (lane_major[..., 0::2] & 0x0F) | ((lane_major[..., 1::2] & 0x0F) << 4)
    scales_panel = np.transpose(scales_f16, (0, 2, 1))  # [n_panels, n_groups, 16]

    header = struct.pack(
        "<8sIIII",
        layout.MAGIC,
        orig_n,
        orig_k,
        n_padded,
        k_padded,
    )
    blob = header + scales_panel.tobytes() + packed_u8.tobytes()
    return blob, packed_u8, scales_panel, (n_padded, k_padded)


def linear_q4p16_wave32_ref(
    blob: bytes,
    x_f32: np.ndarray,
    layout: Q4P16Layout = Q4P16Layout(),
) -> np.ndarray:
    """
    Reference Wave32 decoder + matmul (`y = x @ W^T`) simulating the RDNA 3.5 `gfx1151`
    lane reduction (`32 lanes = 32 K-elements per group`, butterfly sum across lanes).
    """
    magic, orig_n, orig_k, n_padded, k_padded = struct.unpack("<8sIIII", blob[:24])
    if magic != layout.MAGIC:
        raise ValueError(f"Invalid Q4P16 magic {magic!r}")
    n_panels = n_padded // layout.PANEL_ROWS
    n_groups = k_padded // layout.GROUP_SIZE

    scales_bytes = n_panels * n_groups * layout.PANEL_ROWS * 2
    scales_off = 24
    packed_off = scales_off + scales_bytes

    scales = np.frombuffer(blob, dtype=np.float16, count=n_panels * n_groups * layout.PANEL_ROWS, offset=scales_off)
    scales = scales.reshape(n_panels, n_groups, layout.PANEL_ROWS).astype(np.float32)

    packed = np.frombuffer(
        blob,
        dtype=np.uint8,
        count=n_panels * n_groups * layout.WAVE_LANES * (layout.PANEL_ROWS // 2),
        offset=packed_off,
    ).reshape(n_panels, n_groups, layout.WAVE_LANES, layout.PANEL_ROWS // 2)

    q_u4 = np.empty((n_panels, n_groups, layout.WAVE_LANES, layout.PANEL_ROWS), dtype=np.float32)
    q_u4[..., 0::2] = (packed & 0x0F).astype(np.float32) - 8.0
    q_u4[..., 1::2] = ((packed >> 4) & 0x0F).astype(np.float32) - 8.0

    # Dequantize per-group tiles and contract over the 32 Wave32 lanes (simulating butterfly sum)
    w_reconstructed = (
        np.transpose(q_u4 * scales[:, :, None, :], (0, 3, 1, 2))
        .reshape(n_padded, k_padded)[:orig_n, :orig_k]
    )
    return x_f32.astype(np.float32) @ w_reconstructed.T


def convert_pruned_checkpoint_to_q4p16(
    pruned_dir: Path,
    out_q4p_dir: Path,
) -> Dict[str, Any]:
    """
    Streams a pruned `DeepSeek-V4.1-Flash` (MXFP4) or `Qwen3-Coder-480B` (FP8/BF16)
    checkpoint into Wave32 `Q4P16` `.q4p` panel shards + BF16 dense/router safetensors,
    never materializing a full BF16 copy of the model on disk.
    """
    pruned_dir = Path(pruned_dir)
    out_q4p_dir = Path(out_q4p_dir)
    out_q4p_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((pruned_dir / "config.json").read_text(encoding="utf-8"))
    profile = ModelSurgeryProfile.from_config(config)

    index_path = pruned_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        shards = sorted({s for s in weight_map.values() if s not in profile.engram_shards})
    else:
        shards = ["model.safetensors"]

    manifest: Dict[str, Any] = {"arch": profile.arch, "panels": {}, "passthrough_Note": "routers/norms kept in BF16/F32"}
    panels_converted = 0

    for shard_name in shards:
        shard_path = pruned_dir / shard_name
        if not shard_path.exists():
            continue
        with RawSafetensorsReader(shard_path) as rdr:
            for name, entry in rdr.tensors.items():
                if name.endswith((".scale", ".weight_scale_inv", ".weight_scale")):
                    continue

                # 1. DeepSeek-V4.1 MXFP4 expert tensor (`int8`/`uint8` + `<name>.scale` UE8M0)
                scale_key = f"{name}.scale"
                if profile.arch == "deepseek_v41" and scale_key in rdr.tensors and entry.dtype in {"I8", "U8"}:
                    rows, half_cols = entry.shape
                    w_f32 = dequant_mxfp4_block32(
                        rdr.raw_bytes(name),
                        rdr.raw_bytes(scale_key),
                        (rows, half_cols * 2),
                    )
                # 2. Qwen3-Coder FP8 block/channel expert tensor (`F8_E4M3` + `weight_scale_inv` / `weight_scale`)
                elif entry.dtype == "F8_E4M3":
                    base_prefix = name[:-7] if name.endswith(".weight") else name
                    sinv_key = f"{base_prefix}.weight_scale_inv"
                    s_key = f"{base_prefix}.weight_scale"
                    if sinv_key in rdr.tensors:
                        s_arr = rdr.to_float32(sinv_key)
                        w_f32 = dequant_fp8_block2d(rdr.raw_bytes(name), s_arr, tuple(entry.shape), (128, 128), False)
                    elif s_key in rdr.tensors:
                        s_arr = rdr.to_float32(s_key)
                        w_f32 = dequant_fp8_block2d(rdr.raw_bytes(name), s_arr, tuple(entry.shape), (128, 128), False)
                    else:
                        continue
                # 3. Standard BF16 / F16 2D expert projection
                elif ".experts." in name and len(entry.shape) == 2 and entry.dtype in {"BF16", "F16", "F32"}:
                    w_f32 = rdr.to_float32(name)
                else:
                    continue

                blob, _, _, padded_shape = pack_q4p16_panel_matrix(w_f32)
                rel_file = f"{name.replace('/', '_')}.q4p"
                (out_q4p_dir / rel_file).write_bytes(blob)
                manifest["panels"][name] = {
                    "file": rel_file,
                    "orig_shape": entry.shape,
                    "padded_shape": list(padded_shape),
                    "nbytes": len(blob),
                }
                panels_converted += 1

    manifest["panels_converted"] = panels_converted
    (out_q4p_dir / "q4p16_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


# ==============================================================================
# 7. LLAMA.CPP (b11870+ / b12549+) BF16 BRIDGE & GGUF CONVERSION DRIVER
# ==============================================================================

def materialize_bf16_bridge_for_gguf(pruned_dir: Path, bf16_out_dir: Path) -> None:
    """
    Converts MXFP4 (`DeepSeek-V4.1-Flash`) or FP8-block (`Qwen3-Coder-480B-FP8`) expert
    and linear weights shard-by-shard into standard BF16 safetensors when running a
    `llama.cpp` `convert_hf_to_gguf.py` revision that requires BF16 inputs.
    """
    pruned_dir = Path(pruned_dir)
    bf16_out_dir = Path(bf16_out_dir)
    bf16_out_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((pruned_dir / "config.json").read_text(encoding="utf-8"))
    profile = ModelSurgeryProfile.from_config(config)

    index_path = pruned_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        shards = sorted(set(weight_map.values()))
        is_sharded = True
    else:
        shards = ["model.safetensors"]
        is_sharded = False

    new_weight_map: Dict[str, str] = {}
    total_bytes = 0

    for shard_name in shards:
        src_shard = pruned_dir / shard_name
        dst_shard = bf16_out_dir / shard_name
        if shard_name in profile.engram_shards:
            if src_shard.exists():
                if dst_shard.exists() or dst_shard.is_symlink():
                    dst_shard.unlink()
                dst_shard.symlink_to(src_shard.resolve())
                total_bytes += src_shard.stat().st_size
            continue

        with RawSafetensorsReader(src_shard) as rdr:
            out_tensors: List[Tuple[str, str, List[int], UnionBytes]] = []
            for name, entry in rdr.tensors.items():
                if name.endswith((".scale", ".weight_scale_inv", ".weight_scale")):
                    continue
                scale_key = f"{name}.scale"
                base_prefix = name[:-7] if name.endswith(".weight") else name
                sinv_key = f"{base_prefix}.weight_scale_inv"
                s_key = f"{base_prefix}.weight_scale"

                if profile.arch == "deepseek_v41" and scale_key in rdr.tensors and entry.dtype in {"I8", "U8"}:
                    rows, half_cols = entry.shape
                    full_shape = [rows, half_cols * 2]
                    w_f32 = dequant_mxfp4_block32(rdr.raw_bytes(name), rdr.raw_bytes(scale_key), (rows, half_cols * 2))
                    new_name = name if name.endswith(".weight") else f"{name}.weight"
                    out_tensors.append((new_name, "BF16", full_shape, float32_to_bf16_bytes(w_f32)))
                    new_weight_map[new_name] = shard_name
                elif entry.dtype == "F8_E4M3" and (sinv_key in rdr.tensors or s_key in rdr.tensors):
                    key_used = sinv_key if sinv_key in rdr.tensors else s_key
                    s_arr = rdr.to_float32(key_used)
                    w_f32 = dequant_fp8_block2d(rdr.raw_bytes(name), s_arr, tuple(entry.shape), (128, 128), False)
                    out_tensors.append((name, "BF16", entry.shape, float32_to_bf16_bytes(w_f32)))
                    new_weight_map[name] = shard_name
                elif entry.dtype == "F8_E4M3" and scale_key in rdr.tensors:
                    s_u8 = np.frombuffer(rdr.raw_bytes(scale_key), dtype=np.uint8).reshape(rdr.tensors[scale_key].shape)
                    w_f32 = dequant_fp8_block2d(rdr.raw_bytes(name), s_u8, tuple(entry.shape), (32, 32), True)
                    out_tensors.append((name, "BF16", entry.shape, float32_to_bf16_bytes(w_f32)))
                    new_weight_map[name] = shard_name
                else:
                    out_tensors.append((name, entry.dtype, entry.shape, rdr.raw_bytes(name)))
                    new_weight_map[name] = shard_name

            total_bytes += write_safetensors_file(dst_shard, out_tensors, metadata=rdr.metadata)
            out_tensors.clear()
            del out_tensors

    clean_cfg = json.loads(json.dumps(config))
    clean_cfg.pop("quantization_config", None)
    clean_cfg["torch_dtype"] = "bfloat16"
    (bf16_out_dir / "config.json").write_text(json.dumps(clean_cfg, indent=2) + "\n", encoding="utf-8")
    if is_sharded:
        (bf16_out_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": new_weight_map}, indent=2) + "\n",
            encoding="utf-8",
        )
    _copy_auxiliary_repo_files(pruned_dir, bf16_out_dir, profile.num_routed_experts)


def build_gguf_commands(
    pruned_dir: Path,
    llama_cpp_dir: Path,
    out_gguf_prefix: Path,
    quant_type: str = "Q4_K_M",
    imatrix_path: Optional[Path] = None,
    engram_gguf_shard: Optional[Path] = None,
) -> List[List[str]]:
    """
    Generates exact `llama.cpp` (`>= b11870` for DeepSeek-V4.1-Flash, `>= b6810` for
    Qwen3-Coder) conversion + mixed-precision quantization commands:
      - Keeps router gates (`ffn_gate_inp.weight`, `.bias`, `.bias_vl`) in `F32`
      - Keeps shared experts / embeddings / output head in `Q6_K`
      - Quantizes routed experts (`ffn_(up|down|gate)_exps`) to `quant_type` (`Q4_K_M` or `IQ3_M`)
      - On Strix Halo unified memory, run resulting GGUF with `llama-server --no-mmap`
        (except DeepSeek-V4.1 shard 6 Engram, which stays NVMe-mmap'd via `--lazy-mode auto`).
    """
    config = json.loads((Path(pruned_dir) / "config.json").read_text(encoding="utf-8"))
    profile = ModelSurgeryProfile.from_config(config)

    converter = str(Path(llama_cpp_dir) / "convert_hf_to_gguf.py")
    quantizer = str(Path(llama_cpp_dir) / "build" / "bin" / "llama-quantize")
    bf16_gguf = f"{out_gguf_prefix}-bf16.gguf"
    final_gguf = f"{out_gguf_prefix}-{quant_type}.gguf"

    cmds: List[List[str]] = [
        [sys.executable, converter, str(pruned_dir), "--outfile", bf16_gguf, "--outtype", "bf16"]
    ]

    q_cmd = [quantizer]
    if imatrix_path is not None:
        q_cmd.extend(["--imatrix", str(imatrix_path)])
    q_cmd.extend(
        [
            "--token-embedding-type", "Q6_K",
            "--output-tensor-type", "Q6_K",
            "--override-tensor", r"blk\.[0-9]+\.ffn_gate_inp\.(weight|bias|bias_vl)=F32",
        ]
    )
    if profile.arch == "deepseek_v41":
        # Keep shared expert at Q6_K while routed experts use target quant_type
        q_cmd.extend(["--override-tensor", r"blk\.[0-9]+\.ffn_(up|down|gate)_shexp\.weight=Q6_K"])

    q_cmd.extend([bf16_gguf, final_gguf, quant_type])
    cmds.append(q_cmd)

    # DeepSeek-V4.1-Flash GGUF (llama.cpp b11870+): split into 5 backbone shards + symlink Engram as shard 6
    if profile.arch == "deepseek_v41" and engram_gguf_shard is not None:
        gguf_split = str(Path(llama_cpp_dir) / "build" / "bin" / "llama-gguf-split")
        cmds.append([gguf_split, "--split-max-size", "40G", final_gguf, str(out_gguf_prefix)])
        cmds.append(
            [
                "ln",
                "-sf",
                str(engram_gguf_shard),
                f"{out_gguf_prefix}-00006-of-00006.gguf",
            ]
        )

    return cmds


# ==============================================================================
# 8. BUILT-IN END-TO-END SELF-TEST (SYNTHETIC DS-V4.1-FLASH & QWEN3-CODER-480B)
# ==============================================================================

def run_self_test() -> None:
    rng = np.random.default_rng(42)
    print("[self-test] 1/3 Testing DeepSeek-V4.1-Flash MXFP4 + dual-bias router + DSpark + Engram surgery...")

    with tempfile.TemporaryDirectory(prefix="dsv41_test_") as tmp:
        root = Path(tmp)
        src = root / "dsv41_src"
        out = root / "dsv41_pruned"
        q4p = root / "dsv41_q4p16"
        src.mkdir()

        # Synthetic 2-layer CED MoE (16 routed experts -> prune to 8) + 1 DSpark layer (layer 2, 8 experts untouched)
        dsv_cfg = {
            "architectures": ["DeepseekV41ForCausalLM"],
            "model_type": "deepseek_v41",
            "quantization_config": {
                "quant_method": "fp8",
                "weight_block_size": [32, 32],
                "scale_fmt": "ue8m0",
                "expert_dtype": "fp4",
            },
            "text_config": {
                "num_hidden_layers": 2,
                "hidden_size": 64,
                "moe_intermediate_size": 32,
                "n_routed_experts": 16,
                "n_shared_experts": 1,
                "num_experts_per_tok": 4,
                "scoring_func": "sqrtsoftplus",
                "topk_method": "noaux_tc",
                "norm_topk_prob": True,
                "routed_scaling_factor": 1.5,
                "dspark_n_routed_experts": 8,
            },
        }
        (src / "config.json").write_text(json.dumps(dsv_cfg), encoding="utf-8")

        shard1_tensors: List[Tuple[str, str, List[int], UnionBytes]] = []
        gate_w0 = rng.standard_normal((16, 64)).astype(np.float32) * 0.2
        gate_b0 = rng.uniform(9.0, 12.0, size=(16,)).astype(np.float32)
        gate_bvl0 = rng.uniform(9.0, 15.0, size=(16,)).astype(np.float32)

        for l_id in range(2):
            gw = gate_w0 if l_id == 0 else rng.standard_normal((16, 64)).astype(np.float32) * 0.2
            gb = gate_b0 if l_id == 0 else rng.uniform(9.0, 12.0, size=(16,)).astype(np.float32)
            gbvl = gate_bvl0 if l_id == 0 else rng.uniform(9.0, 15.0, size=(16,)).astype(np.float32)
            shard1_tensors.append((f"layers.{l_id}.ffn.gate.weight", "BF16", [16, 64], float32_to_bf16_bytes(gw)))
            shard1_tensors.append((f"layers.{l_id}.ffn.gate.bias", "F32", [16], gb.tobytes()))
            shard1_tensors.append((f"layers.{l_id}.ffn.gate.bias_vl", "F32", [16], gbvl.tobytes()))
            for e_id in range(16):
                for proj, shape in (("w1", [32, 32]), ("w2", [64, 16]), ("w3", [32, 32])):
                    packed = rng.integers(0, 256, size=shape, dtype=np.uint8).view(np.int8)
                    scales = rng.integers(124, 130, size=(shape[0], (shape[1] * 2) // 32), dtype=np.uint8)
                    shard1_tensors.append((f"layers.{l_id}.ffn.experts.{e_id}.{proj}", "I8", shape, packed.tobytes()))
                    shard1_tensors.append(
                        (f"layers.{l_id}.ffn.experts.{e_id}.{proj}.scale", "U8", list(scales.shape), scales.tobytes())
                    )

        # DSpark / MTP speculative layer (layer_id == 2 >= num_hidden_layers=2, 8 experts -> must stay 8!)
        for e_id in range(8):
            packed = rng.integers(0, 256, size=(32, 32), dtype=np.uint8).view(np.int8)
            scales = rng.integers(124, 130, size=(32, 2), dtype=np.uint8)
            shard1_tensors.append((f"layers.2.ffn.experts.{e_id}.w1", "I8", [32, 32], packed.tobytes()))
            shard1_tensors.append((f"layers.2.ffn.experts.{e_id}.w1.scale", "U8", [32, 2], scales.tobytes()))

        write_safetensors_file(src / "model-00001-of-00048.safetensors", shard1_tensors)
        engram_raw = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        write_safetensors_file(
            src / "model-00047-of-00048.safetensors",
            [("layers.1.engram.embed", "F8_E4M3", [64, 64], engram_raw.tobytes())],
        )

        wmap = {t[0]: "model-00001-of-00048.safetensors" for t in shard1_tensors}
        wmap["layers.1.engram.embed"] = "model-00047-of-00048.safetensors"
        (src / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": wmap}), encoding="utf-8"
        )

        # Simulate calibration collector + keep-set solver (16 -> 8 experts)
        collector = DistributedEdgeREAPCollector(num_layers=2, num_experts=16)
        x_cal = rng.standard_normal((128, 64)).astype(np.float32)
        with RawSafetensorsReader(src / "model-00001-of-00048.safetensors") as rdr:
            gw0_bf16 = rdr.to_float32("layers.0.ffn.gate.weight")
        idx0, w0 = dsv41_route_numpy(x_cal, gw0_bf16, gate_b0, gate_bvl0, top_k=4)
        fake_norms = rng.uniform(1.0, 10.0, size=idx0.shape).astype(np.float32)
        collector.record_batch(0, idx0, w0, fake_norms, np.arange(128))
        collector.record_batch(1, idx0, w0, fake_norms, np.arange(128))
        rep_path = root / "dsv41_report.json"
        collector.export_report(rep_path, "deepseek-v4.1-flash-test")

        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        keepset = compute_edge_reap_keepset(
            rep,
            keep_experts=8,
            pin_core_experts=2,
            pin_super_experts=1,
            pin_early_pos_experts=1,
            vl_biases_by_layer={0: gate_bvl0, 1: gate_bvl0},
            pin_vl_experts=1,
            tau_domain=4.0,
        )

        summary = slice_checkpoint(src, out, keepset, symlink_engram=True, verify_readback=True)
        assert summary.kept_experts == 8 and summary.engram_shards_linked == 1
        assert (out / "model-00047-of-00048.safetensors").is_symlink()

        # Verify DSpark layer 2 kept all 8 experts & layer 0 router math is exact when top-4 subset is in keepset
        with RawSafetensorsReader(out / "model-00001-of-00048.safetensors") as rdr_out:
            assert "layers.2.ffn.experts.7.w1" in rdr_out.tensors, "DSpark layer must remain unpruned!"
            assert "layers.0.ffn.experts.8.w1" not in rdr_out.tensors, "Renumbered experts must be 0..7!"
            gw_sliced = rdr_out.to_float32("layers.0.ffn.gate.weight")
            gb_sliced = rdr_out.to_float32("layers.0.ffn.gate.bias")
            gbvl_sliced = rdr_out.to_float32("layers.0.ffn.gate.bias_vl")

        kept0 = keepset[0]
        idx_sliced, w_sliced = dsv41_route_numpy(x_cal, gw_sliced, gb_sliced, gbvl_sliced, top_k=4)
        mapped_back = np.asarray(kept0)[idx_sliced]
        in_pack_mask = np.all(np.isin(idx0, kept0), axis=-1)
        if np.any(in_pack_mask):
            assert np.array_equal(mapped_back[in_pack_mask], idx0[in_pack_mask])
            assert np.max(np.abs(w_sliced[in_pack_mask] - w0[in_pack_mask])) < 1e-6

        # Test MXFP4 -> Wave32 Q4P16 streaming conversion & reference matmul
        q4p_manifest = convert_pruned_checkpoint_to_q4p16(out, q4p)
        assert q4p_manifest["panels_converted"] == 2 * 8 * 3 + 8  # 2 layers * 8e * 3 projs + 8 DSpark

    print("[self-test] 2/3 Testing Qwen3-Coder-480B-FP8 block-scale surgery & softmax renorm invariance...")
    with tempfile.TemporaryDirectory(prefix="qwen3_test_") as tmp:
        root = Path(tmp)
        src = root / "qwen_src"
        out = root / "qwen_pruned"
        bf16_dir = root / "qwen_bf16"
        src.mkdir()

        qwen_cfg = {
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
            "num_hidden_layers": 1,
            "hidden_size": 128,
            "moe_intermediate_size": 128,
            "num_experts": 12,
            "num_experts_per_tok": 4,
            "shared_expert_intermediate_size": 0,
            "norm_topk_prob": True,
            "quantization_config": {"quant_method": "fp8"},
        }
        (src / "config.json").write_text(json.dumps(qwen_cfg), encoding="utf-8")

        gw_qwen = rng.standard_normal((12, 128)).astype(np.float32) * 0.2
        q_tensors: List[Tuple[str, str, List[int], UnionBytes]] = [
            ("model.layers.0.mlp.gate.weight", "BF16", [12, 128], float32_to_bf16_bytes(gw_qwen))
        ]
        for e_id in range(12):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                # Valid non-NaN FP8 E4M3 bytes (avoid 0x7F and 0xFF)
                fp8_raw = rng.integers(0, 126, size=(128, 128), dtype=np.uint8)
                sinv = np.array([[0.02]], dtype=np.float32)
                q_tensors.append((f"model.layers.0.mlp.experts.{e_id}.{proj}.weight", "F8_E4M3", [128, 128], fp8_raw.tobytes()))
                q_tensors.append((f"model.layers.0.mlp.experts.{e_id}.{proj}.weight_scale_inv", "F32", [1, 1], sinv.tobytes()))

        write_safetensors_file(src / "model.safetensors", q_tensors)
        keepset_q = {0: [0, 2, 4, 7, 9, 11]}
        summary_q = slice_checkpoint(src, out, keepset_q, verify_readback=True)
        assert summary_q.kept_experts == 6

        with RawSafetensorsReader(src / "model.safetensors") as r0, RawSafetensorsReader(out / "model.safetensors") as r1:
            gw_orig = r0.to_float32("model.layers.0.mlp.gate.weight")
            gw_cut = r1.to_float32("model.layers.0.mlp.gate.weight")
            assert "model.layers.0.mlp.experts.5.down_proj.weight_scale_inv" in r1.tensors
            assert "model.layers.0.mlp.experts.6.down_proj.weight" not in r1.tensors

        x_q = rng.standard_normal((256, 128)).astype(np.float32)
        idx_full, w_full = qwen3_moe_route_numpy(x_q, gw_orig, top_k=4, norm_topk_prob=True)
        idx_cut, w_cut = qwen3_moe_route_numpy(x_q, gw_cut, top_k=4, norm_topk_prob=True)
        mapped_q = np.asarray(keepset_q[0])[idx_cut]
        mask_q = np.all(np.isin(idx_full, keepset_q[0]), axis=-1)
        if np.any(mask_q):
            assert np.array_equal(mapped_q[mask_q], idx_full[mask_q])
            assert np.max(np.abs(w_cut[mask_q] - w_full[mask_q])) < 1e-6

        materialize_bf16_bridge_for_gguf(out, bf16_dir)
        with RawSafetensorsReader(bf16_dir / "model.safetensors") as r_bf16:
            assert r_bf16.tensors["model.layers.0.mlp.experts.0.gate_proj.weight"].dtype == "BF16"

    print("[self-test] 3/3 Testing Wave32 Q4P16 panel pack/unpack numerical fidelity...")
    w_mat = rng.standard_normal((48, 96)).astype(np.float32) * 0.5
    x_vec = rng.standard_normal((8, 96)).astype(np.float32)
    blob, _, _, _ = pack_q4p16_panel_matrix(w_mat)
    y_ref = x_vec @ w_mat.T
    y_q4p = linear_q4p16_wave32_ref(blob, x_vec)
    cos_sim = float(np.sum(y_ref * y_q4p) / (np.linalg.norm(y_ref) * np.linalg.norm(y_q4p)))
    assert cos_sim > 0.992, f"Expected cosine similarity > 0.992, got {cos_sim:.6f}"
    print(f"[self-test] Q4P16 Wave32 matmul cosine similarity = {cos_sim:.6f} — ALL TESTS PASSED.")


# ==============================================================================
# 9. CLI ENTRYPOINT
# ==============================================================================

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Edge-REAP Safetensors & Router Slicer + Q4P16 / GGUF Converter for Strix Halo (gfx1151)"
    )
    p.add_argument("--self-test", action="store_true", help="Run built-in verification suite and exit")
    sub = p.add_subparsers(dest="cmd")

    # 1. Solve keep-set + slice checkpoint
    s_slice = sub.add_parser("slice", help="Compute Edge-REAP keep-set and slice safetensors shards")
    s_slice.add_argument("--src-dir", type=Path, required=True, help="Unpruned HF safetensors directory")
    s_slice.add_argument("--out-dir", type=Path, required=True, help="Output pruned checkpoint directory")
    s_slice.add_argument("--domain-report", type=Path, required=True, help="Calibration JSON from DistributedEdgeREAPCollector")
    s_slice.add_argument("--core-report", type=Path, default=None, help="Optional general-code/math calibration JSON")
    s_slice.add_argument("--keep-experts", type=int, required=True, help="Routed experts to keep per layer (e.g. 192 or 80)")
    s_slice.add_argument("--pin-core", type=int, default=16, help="General-core experts pinned per layer (use >=16 for Qwen3-Coder)")
    s_slice.add_argument("--pin-super", type=int, default=4, help="Peak-L2 Super-Experts pinned per layer")
    s_slice.add_argument("--pin-early-pos", type=int, default=4, help="Pos [0,64) experts pinned per layer")
    s_slice.add_argument("--pin-vl", type=int, default=0, help="DeepSeek-V4.1 gate.bias_vl experts to pin if text-only calibrated")
    s_slice.add_argument("--tau-domain", type=float, default=64.0, help="Domain token count saturation scale")
    s_slice.add_argument("--no-symlink-engram", action="store_true", help="Skip symlinking DeepSeek-V4.1 shards 47 & 48")

    # 2. Export Wave32 Q4P16 panels
    s_q4p = sub.add_parser("export-q4p16", help="Stream pruned MXFP4/FP8/BF16 experts into Wave32 Q4P16 panels")
    s_q4p.add_argument("--pruned-dir", type=Path, required=True)
    s_q4p.add_argument("--out-dir", type=Path, required=True)

    # 3. Export GGUF via llama.cpp (b11870+)
    s_gguf = sub.add_parser("export-gguf", help="Materialize BF16 bridge (if needed) and run/print llama.cpp GGUF commands")
    s_gguf.add_argument("--pruned-dir", type=Path, required=True)
    s_gguf.add_argument("--llama-cpp-dir", type=Path, required=True)
    s_gguf.add_argument("--out-prefix", type=Path, required=True)
    s_gguf.add_argument("--quant-type", type=str, default="Q4_K_M", help="Target routed-expert GGUF type (e.g. Q4_K_M, IQ3_M)")
    s_gguf.add_argument("--imatrix", type=Path, default=None)
    s_gguf.add_argument("--engram-gguf", type=Path, default=None, help="Path to DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf")
    s_gguf.add_argument("--materialize-bf16-dir", type=Path, default=None, help="Optional scratch dir to dequantize MXFP4/FP8 -> BF16 first")
    s_gguf.add_argument("--execute", action="store_true", help="Execute llama.cpp commands instead of dry-run printing")

    return p


def _extract_dsv41_vl_biases(src_dir: Path, num_layers: int) -> Dict[int, np.ndarray]:
    index_path = src_dir / "model.safetensors.index.json"
    vl_map: Dict[int, np.ndarray] = {}
    if not index_path.exists():
        return vl_map
    wmap = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    by_shard: Dict[str, List[Tuple[int, str]]] = {}
    for l_id in range(num_layers):
        for cand in (f"layers.{l_id}.ffn.gate.bias_vl", f"model.layers.{l_id}.ffn.gate.bias_vl"):
            if cand in wmap:
                by_shard.setdefault(wmap[cand], []).append((l_id, cand))
                break
    for shard_name, items in by_shard.items():
        with RawSafetensorsReader(src_dir / shard_name) as rdr:
            for l_id, t_name in items:
                vl_map[l_id] = rdr.to_float32(t_name)
    return vl_map


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_cli()
    args = parser.parse_args(argv)

    if args.self_test:
        run_self_test()
        return 0

    if args.cmd == "slice":
        domain_rep = json.loads(args.domain_report.read_text(encoding="utf-8"))
        core_rep = json.loads(args.core_report.read_text(encoding="utf-8")) if args.core_report else None
        vl_biases = (
            _extract_dsv41_vl_biases(args.src_dir, int(domain_rep["num_layers"]))
            if args.pin_vl > 0
            else None
        )
        keepset = compute_edge_reap_keepset(
            domain_report=domain_rep,
            keep_experts=args.keep_experts,
            core_report=core_rep,
            pin_core_experts=args.pin_core,
            pin_super_experts=args.pin_super,
            pin_early_pos_experts=args.pin_early_pos,
            vl_biases_by_layer=vl_biases,
            pin_vl_experts=args.pin_vl,
            tau_domain=args.tau_domain,
        )
        summary = slice_checkpoint(
            src_dir=args.src_dir,
            out_dir=args.out_dir,
            keepset=keepset,
            symlink_engram=not args.no_symlink_engram,
            verify_readback=True,
        )
        print(json.dumps(dataclasses.asdict(summary), indent=2))
        return 0

    if args.cmd == "export-q4p16":
        manifest = convert_pruned_checkpoint_to_q4p16(args.pruned_dir, args.out_dir)
        print(f"Converted {manifest['panels_converted']} expert matrices into Wave32 Q4P16 panels at {args.out_dir}")
        return 0

    if args.cmd == "export-gguf":
        target_hf_dir = args.pruned_dir
        if args.materialize_bf16_dir is not None:
            materialize_bf16_bridge_for_gguf(args.pruned_dir, args.materialize_bf16_dir)
            target_hf_dir = args.materialize_bf16_dir
        cmds = build_gguf_commands(
            pruned_dir=target_hf_dir,
            llama_cpp_dir=args.llama_cpp_dir,
            out_gguf_prefix=args.out_prefix,
            quant_type=args.quant_type,
            imatrix_path=args.imatrix,
            engram_gguf_shard=args.engram_gguf,
        )
        for cmd in cmds:
            print(" ".join(cmd))
            if args.execute:
                subprocess.run(cmd, check=True)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
