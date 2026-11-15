#!/usr/bin/env python3
"""Harvest a strixite checkout into catalog/strixite.

    tools/harvest-strixite.py <strixite-checkout> [--commit d19945e] [--branch v0.7.0-pod]

Copies every listed origin file verbatim to catalog/strixite/<kind>/<name>/src/<origin path> and writes the
module.toml manifests from the table below. `requires` is derived from the #include graph: a module requires
exactly the modules whose headers its files include. Kernels (with sources) and formats are required by interface
from other kinds; helpers, runtime and serve pieces by module id, because their C++ APIs are strixite's own.

The manifests are the source of truth once written; this script exists to redo the harvest at a new origin commit.
Per-file authors come from catalog/strixite/origin-authors.tsv (see ORIGIN.toml for how it is produced).
"""
import argparse
import os
import shutil
import sys
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from flerplib import includes as inc  # noqa: E402
from flerplib import tomlw  # noqa: E402

ORIGIN = "strixite"
REPO = "https://github.com/shawnshekari/strixite"
AGPL = "AGPL-3.0-only"
BOTH = ("bf16", "f32")

MODULES = []


def M(kind, name, summary, paths, provides, acts=None, **build):
    """provides: list of (interface, api, numeric, regime). An interface ending in '/*' is expanded once per
    activation dtype in `acts`."""
    MODULES.append(dict(kind=kind, name=name, summary=summary, paths=paths, provides=provides, acts=acts,
                        build=build))


def pair(stem, exts):
    return [f"{stem}.{e}" for e in exts]


FP32_ONE = "fp32 math, one rounding to the activation dtype where the output is stored"

# --- host helpers (common/) ----------------------------------------------------------------------------------
M("runtime", "check", "STRIX_CHECK precondition checks and the strix::Error types (header only, no dependencies)",
  ["common/check.hpp"], [("host.check/-/-", "STRIX_CHECK, strix::Error, strix::GpuFatalError", "n/a", "host")])
M("runtime", "args", "Checked command-line number parsing for tools",
  ["common/args.hpp"], [("host.args/-/-", "strix::parse_int, strix::parse_double", "n/a", "host")])
M("runtime", "log", "Levelled one-line logging with journald priority prefixes",
  ["common/log.hpp"], [("host.log/-/-", "strix::slog, strix::set_log_level", "n/a", "host")])
M("runtime", "bf16", "Host BF16 bit helpers, the memcpy twins of the device functions in kernels/rope_math.hpp",
  ["common/bf16.hpp"], [("host.bf16/-/-", "strix::f32_to_bf16_bits, strix::bf16_bits_to_f32",
                         "round to nearest even, non-finite values preserved", "host")])
M("runtime", "trace", "ROCTX range macros for profiler timelines; expand to nothing unless STRIX_ROCTX is set",
  ["common/trace.hpp"], [("host.trace/-/-", "STRIX_TRACE_RANGE, STRIX_TRACE_STAGE", "n/a", "host, profiling builds")])

# --- device helpers (header only) ----------------------------------------------------------------------------
M("kernel", "hip_common", "The repo's HIP runtime include and STRIX_HIP_CHECK return-code checks",
  ["common/hip_runtime.hpp", "common/hip_check.hpp"],
  [("device.hip-check/-/-", "STRIX_HIP_CHECK, strix::hip_error_is_sticky", "n/a", "host side of every launch")],
  uses=["hip-runtime"])
M("kernel", "device_info", "Wavefront size of the current device, queried once, for sizing launch grids",
  ["kernels/device_info.hpp"], [("device.info/-/-", "strix::kernels::device_warp_size", "n/a", "host, cached")])
M("kernel", "qsplit", "Row layout of Q4 and Q5 expert weights as the expert kernels read it (device helper)",
  ["kernels/qsplit.hpp"], [("linear.row-layout/strixw-q4/-", "strix::kernels::qsplit_row_bytes and loaders",
                            "raw codes, no arithmetic", "included by .hip files only; Q5 rows need K % 128 == 0")])
M("kernel", "rope_math", "Device-side RoPE and BF16 conversion arithmetic shared by rope, attention, qsa and kv8",
  ["kernels/rope_math.hpp"],
  [("pos.rope-math/-/-", "strix::kernels::rope_math::{cos_sin, rotate, f32_to_bf16_bits, bf16_bits_to_f32}",
    "transformers' arithmetic: each product and the sum rounded separately, no FMA", "device inline")])
M("kernel", "wmma_gemm", "Device building blocks of the tiled WMMA GEMMs: LDS staging, software pipelining, "
  "scaled-code accumulation",
  ["kernels/wmma_gemm.hpp"],
  [("gemm.wmma-blocks/-/bf16", "strix::kernels::{wmma_step, wmma_step_scaled, wmma_step_f16, wmma_stage_codes}",
    "x rounded to bf16 (or f16), Q4/Q8 weights enter exactly as scaled codes, fp32 accumulate",
    "included by .hip files only; K step kWmmaKC, lower-precision path")])

# --- kernels -------------------------------------------------------------------------------------------------
K = "strix::kernels::"


def kernel(name, summary, provides, extra=(), acts=BOTH, hpp=True):
    paths = [f"kernels/{name}.hip"] + ([f"kernels/{name}.hpp"] if hpp else []) + list(extra)
    M("kernel", name, summary, paths, provides, acts=acts)


kernel("norm", "RMSNorm kernels (zero-centred, grouped, gated) and the Act activation-dtype enum every kernel takes",
       [("norm.rmsnorm-zc/-/*", K + "rmsnorm_zc, " + K + "group_rmsnorm_zc",
         "fp32 math with fp32 weights; casts to the activation dtype at the reference's points, so bf16 output is "
         "within one bf16 rounding step of transformers", "rows x d contiguous, in place allowed"),
        ("norm.rmsnorm-gated/-/*", K + "rmsnorm_gated",
         "out = cast(w * cast(x * rsqrt(mean(x^2) + eps)) * gate(z)), fp32 math",
         "GDN output norm: rows = tokens x value heads, d = 128; gate SiLU or Sigmoid chosen per call")])
kernel("embedding", "Token embedding lookup from a BF16 or Q8 table, with a device-side bad-id error slot",
       [("embed.lookup/bf16/*", K + "embedding_lookup (uint16_t table), " + K + "check_embedding_error",
         "table rows copied (bf16) or widened exactly (f32); an out-of-range id zeroes its row and is reported",
         "one launch per forward; copies > 1 broadcasts into the hyper-connection streams"),
        ("embed.lookup/strixw-q8/*", K + "embedding_lookup (Q8DeviceView table)",
         "Q8 row dequantized (code * scale + min) and stored in the activation dtype",
         "one launch per forward")])
kernel("linear", "BF16-weight linear layer, deliberately simple: the correctness baseline and the path for tensors "
       "that stay BF16",
       [("linear.dense/bf16/*", K + "linear_bf16w",
         "fp32 accumulate, one rounding at the end; a bf16 -> f32 call stores the accumulator unrounded",
         "untuned; any M")])
kernel("linear_q4", "Q4 group-affine GEMV for decode (split-K, 32 codes per 16-byte load) and the per-token MoE "
       "expert gather/combine for Q4 and Q5 experts",
       [("linear.decode/strixw-q4/*", K + "linear_q4",
         "fp32 accumulate (s * sum(x*q) + m * sum(x) per chunk, one FMA per weight), one rounding to the activation "
         "dtype on store; a bf16 -> f32 call stores the accumulator unrounded",
         "decode: small M, memory-bound; M == 1 has its own instantiation, larger M runs 8 rows per pass; K % 32 == 0"),
        ("moe.experts-decode/strixw-q4/*",
         K + "linear_q4_experts_gather, _gather_swiglu, _combine, " + K + "check_expert_error",
         "fp32 accumulate, one rounding on store; an out-of-range expert id gives NaN and sets the error words",
         "one launch covers every (token, active expert) slot; reads an expert's rows once per slot"),
        ("moe.experts-decode/strixw-q5/*", K + "linear_q5_experts_gather, _gather_swiglu, _combine",
         "as the Q4 experts, one kernel template", "K % 128 == 0")])
kernel("linear_q56", "Q5 and Q6 GEMV for decode: a 4-bit low plane plus a high-bit plane per row, one kernel template",
       [("linear.decode/strixw-q5/*", K + "linear_q5",
         "fp32 accumulate (s * sum(x*q) + m * sum(x) per chunk), one rounding on store; bf16 -> f32 stores unrounded",
         "decode: 1-8 waves per output row (split-K, ~2048 codes a wave); K % 128 == 0"),
        ("linear.decode/strixw-q6/*", K + "linear_q6",
         "fp32 accumulate (s * sum(x*q) + m * sum(x) per chunk), one rounding on store; bf16 -> f32 stores unrounded",
         "decode: 1-8 waves per output row (split-K); K % 64 == 0")],
       extra=["kernels/linear_q5.hpp", "kernels/linear_q6.hpp"], hpp=False)
kernel("linear_q8", "Q8 GEMV for decode: linear_q4's structure at one code per byte",
       [("linear.decode/strixw-q8/*", K + "linear_q8",
         "fp32 accumulate (s * sum(x*q) + m * sum(x) per chunk), one rounding on store; bf16 -> f32 stores unrounded",
         "decode: 16 codes per 16-byte load, 1-8 waves per output row; M == 1 has its own instantiation")])
kernel("swiglu", "SwiGLU on the merged [gate | up] projection output",
       [("act.swiglu/-/*", K + "swiglu",
         "fp32 math; bf16 rounds where the reference does: h = bf16(bf16(silu(gate)) * up)",
         "rows of [gate (I) | up (I)]: dense MLP and the expert gather's output")])
kernel("rope", "Partial rotary position embedding (rotate-half on the first rot_dim dims), with YaRN and "
       "interleaved-mRoPE positions",
       [("pos.rope/-/*", K + "rope, " + K + "rope_inv_freq, " + K + "rope_yarn",
         "transformers' arithmetic: angle = fp32(pos * inv_freq), products and sum rounded separately (no FMA); "
         "bf16 also rounds cos/sin and each step through bf16",
         "theta 1e7, rot_dim 64 of head_dim 256 on the target models")])
kernel("attention", "Causal softmax attention with GQA, the sigmoid output gate, fused q/k norm + RoPE jobs, the "
       "QSA-gathered variant, WMMA prefill variants and 8-bit K/V readers",
       [("attn.softmax-gqa/-/*",
         K + "qk_norm_rope, qk_norm_rope_jobs, attention, attention_gathered, attention_wmma, attention_gathered_wmma",
         "inputs in the activation dtype, scores, online softmax and P*V in fp32, one rounding at the output; the "
         "_wmma variants run the score and P*V products on the matrix units (lower precision, behind a switch)",
         "decode-oriented; K/V cache [capacity, Hk, D] in the activation dtype or the kv8 8-bit format")])
kernel("gdn", "Gated DeltaNet mixer: causal depthwise conv front end and the delta-rule recurrence",
       [("mixer.gdn/-/*", K + "gdn_front, " + K + "delta_rule",
         "conv, silu and gates in fp32 on activation-dtype inputs; qkv_out rounded once; beta and g always fp32",
         "conv width 4 with a 3-row carried state; decode (T = 1) and chunked prompts give the same result")])
kernel("residual", "Residual add in place",
       [("residual.add/-/*", K + "residual_add", "fp32 add, one rounding to the activation dtype", "element-wise")])
kernel("moe_router", "MoE router: top-k experts per token with softmax-renormalized weights, plus the shared "
       "expert's sigmoid gate",
       [("moe.router/-/*", K + "moe_router, moe_shared_add, moe_shared_add_inject, check_router_error",
         "all fp32, coefficients stored fp32 (never rounded to bf16); ties go to the lower expert index; a non-finite "
         "logit fails that token and is reported",
         "k in 1..min(E, 32), E in 1..1024")])
kernel("moe_grouped", "MoE experts for multi-token forwards: slots grouped by expert so each expert's weights are "
       "read once per tile",
       [("moe.experts-prefill/strixw-q4/*",
         K + "moe_group_routes, linear_q4_experts_gather_grouped, _gather_grouped_swiglu, _combine_grouped",
         "MoeMath::F32: fp32 on the vector units, same semantics as the decode kernels, only the summation order "
         "differs; MoeMath::WmmaBf16 / WmmaF16*: matrix units with x rounded to bf16 or f16 and fp32 accumulation "
         "(lower precision, behind a switch); the three WmmaF16 kernels are bit-identical to each other",
         "prefill and MTP verify; tiles of 16 (fp32) or 64/128 (WMMA) slots; E <= 1023; WMMA needs K % 64 == 0"),
        ("moe.experts-prefill/strixw-q5/*", K + "linear_q5_experts_gather_grouped, _combine_grouped",
         "as the Q4 grouped kernels, one kernel template", "K % 128 == 0")])
M("kernel", "wmma", "gfx11 matrix-core 16x16 tile helpers (wave32) and the end-to-end lane-layout check",
  pair("kernels/wmma", ("hip", "hpp")),
  [("gemm.wmma-tile/-/bf16", K + "wmma_tile_product and the device helpers wmma_bf16, wmma_f16, wmma_load_rows",
    "bf16 (or f16) inputs, fp32 accumulate; not bit-exact IEEE: up to 2^-23 of sum|a*b| per 16x16x16 step",
    "v_wmma_f32_16x16x16_bf16 on RDNA3 / RDNA3.5")])
kernel("linear_wmma", "Multi-token (prefill) Q4/Q5/Q6/Q8 and BF16-weight linear on the matrix units",
       [(f"linear.prefill/strixw-{q}/*", K + f"linear_{q}_wmma",
         "x rounded to bf16, weights exact (raw codes on the matrix units, group scale and min applied in fp32), "
         "fp32 accumulate, y rounded once", "prefill: tiles of 64 x-rows by 128 weight rows, K % 64 == 0")
        for q in ("q4", "q5", "q6", "q8")] +
       [("linear.prefill/bf16/*", K + "linear_bf16w_wmma",
         "bf16(x) * W^T with exact products, fp32 accumulate; only the summation order differs from linear_bf16w",
         "prefill: K % 64 == 0 (the router)")])
kernel("hc_wmma", "Hyper-connection mixes for multi-token forwards on the matrix units",
       [(f"hc.mix-prefill/strixw-{q}/*", K + "hc_mix_down_wmma, hc_mix_up_wmma, hc_inject_inv",
         "x and h rounded once to f16, weights dequantized to f16 (one rounding), fp32 accumulation over each "
         "GEMM's whole K; norms, gates and the stream combine in fp32; hc_inject_inv is bit-identical to hc_inject "
         "followed by the mix's own inverse RMS",
         "prefill: 64-token tiles; H == 4, d % 64 == 0, r % 64 == 0") for q in ("q4", "q8")])
kernel("hc", "Hyper-connection input mix over the 4 residual streams and the residual injection, per token",
       [(f"hc.mix/strixw-{q}/*", K + "hc_mix_down, hc_mix_up, hc_inject",
         "everything inside in fp32 (normed streams never stored); u rounded once to the activation dtype; the "
         "injection rounds once per element",
         "decode: two launches per mix; H == 4, d % 32 == 0, r % 32 == 0, r <= 1024") for q in ("q4", "q8")])
kernel("ple", "PLE injection: the gate over key, value and streams, then the dilated depthwise conv",
       [("ple.inject/-/*", K + "ple_gate, " + K + "ple_conv",
         "fp32 math; vn and the conv state stored in the activation dtype, sigma fp32, X rounded once per element",
         "two launches; K = 4 taps, dilation 3, 9 carried state rows")])
kernel("qsa", "QSA indexer: block keys, block scores and per-query top-k block selection",
       [("attn.qsa-indexer/-/*", K + "qsa_block_keys, qsa_scores, qsa_topk",
         "block-key mean in fp32 then cast to the activation dtype as the reference; scores fp32; top-k is a set "
         "with ties to the lower block index; the Wmma score path runs on the matrix units",
         "4-token blocks; k = budget / 4 (512 on the target); H <= 8 heads, D % 8 == 0, D <= 256")])
kernel("mtp_pick", "The MTP draft's pick on the GPU: best logit, runner-up value and NaN flag in one 64-byte struct",
       [("draft.mtp-pick/-/-", K + "mtp_pick",
         "exact selection on fp32 logits: lowest index on a tie, runner-up is the multiset second",
         "one block per draft step")], acts=None)
kernel("logits_topk", "Each logits row's top 20 candidates on the GPU by rank counting, with optional allow-mask",
       [("logits.topk/-/-", K + "logits_topk",
         "exact selection on fp32 logits: value descending, lower id first on a tie, so sampling from the candidates "
         "is bit-identical to sampling from the row; NaN flagged per row",
         "kLogitCands = 20; 8,192 inputs per workgroup, 2 launches for 248,077 logits")], acts=None)
kernel("logits_bias", "A request's logit_bias added to its logits rows in place, ahead of logits_topk",
       [("logits.bias/-/-", K + "logits_bias", "one fp32 add per entry", "one thread per entry; entries must not "
         "collide")], acts=None)
kernel("vision", "Vision tower kernels around the BF16 GEMMs: linear with bias, layernorm, GELU, attention, "
       "2D RoPE, patch merge, position add, embedding scatter",
       [("vision.tower-ops/bf16/*",
         K + "vision_linear_bias, vision_layernorm, vision_gelu, vision_attn, vision_rope_qk, vision_merge_regroup, "
         "vision_pos_add, vision_embedding_scatter, vision_bias_add",
         "bf16 weights, fp32 math with the bias added on the fp32 accumulator, one rounding where the output is "
         "stored", "one image per forward; sequence capped at kMaxVisionSeq")])
kernel("kv8", "8-bit K/V cache for the full-attention layers: quantize on group completion, shared device "
       "dequantization arithmetic",
       [("kvcache.quant8/-/*", K + "kv8_commit, kv8_dequant, kv8_bytes, kv8_used_ranges and kv8_math::*",
         "K: asymmetric 8 bits per channel over groups of 32 rows; V: 8 bits per block of 32 channels; scale and "
         "zero stored as fp16; value = fma(code, scale, zero) in fp32, rounded to the activation dtype where read; "
         "the newest incomplete group stays exact in a tail",
         "about 56% of bf16 storage; what a query sees depends only on its own position")],
       extra=["kernels/kv8_math.hpp"])
kernel("steer", "Activation steering: add a scaled vector to rows of the residual streams",
       [("steer.add-rows/-/*", K + "steer_add_rows", "fp32 add, one rounding to the activation dtype",
         "x[r, k] += scale * v[k]")])

# --- formats -------------------------------------------------------------------------------------------------
def fmt(name, summary, provides, extra=()):
    M("format", name, summary, pair(f"formats/{name}", ("cpp", "hpp")) + list(extra), provides)


fmt("safetensors", "Hand-rolled read-only mmap reader for HF safetensors files",
    [("format.reader/safetensors/-", "strix::SafetensorsFile", "bytes as stored", "host; header and metadata "
      "inspection, converter input")])
fmt("q4", "Q4: 4-bit asymmetric per-group weight format (two codes per byte, BF16 scale and min per group)",
    [("format.host/strixw-q4/-", "strix::Q4Weight, quantize_q4, dequantize_q4, append_rows_q4",
      "w = q * scale + min per group; quantization bit-identical to the reference quantizer",
      "G in 32/64/128, K % G == 0")])
fmt("q5", "Q5: 5-bit two-plane weight format (Q4 low plane plus a 1-bit high plane)",
    [("format.host/strixw-q5/-", "strix::Q5Weight, quantize_q5, dequantize_q5, append_rows_q5",
      "s = bf16((max - min) / 31), q = clamp(round_half_even((w - m) / s), 0, 31), all fp32",
      "5.25 bits per weight at G = 128; K % 128 == 0")])
fmt("q6", "Q6: 6-bit two-plane weight format (Q4 low plane plus a 2-bit high plane)",
    [("format.host/strixw-q6/-", "strix::Q6Weight, quantize_q6, dequantize_q6, append_rows_q6",
      "s = bf16((max - min) / 63), q = clamp(round_half_even((w - m) / s), 0, 63), all fp32",
      "6.25 bits per weight at G = 128; K % 64 == 0")])
fmt("q8", "Q8: 8-bit asymmetric per-group weight format (one code per byte)",
    [("format.host/strixw-q8/-", "strix::Q8Weight, quantize_q8, dequantize_q8, append_rows_q8, to_chunk_major",
      "s = bf16((max - min) / 255), q = clamp(round_half_even((w - m) / s), 0, 255), all fp32",
      "8.5 bits per weight at G = 64")])
fmt("q4_from_q8", "Q8 to Q4 requantization for the MTP draft's cheaper LM-head copy",
    [("format.requant/strixw-q4/-", "strix::quantize_q4_from_q8",
      "dequantize the Q8 rows exactly, then quantize as quantize_q4; byte-identical regardless of thread split",
      "host, once at load")])
fmt("merge_plan", "Load-time merges of linear weights that read the same input into one row-concatenated weight",
    [("format.merge-plan/strixw/-", "strix::merge_plan, strix::resolve_merge, strix::quantize_merged",
      "row concatenation at quantization time is exact", "host, converter and loaders")])
fmt("strixw", "strixw container: header, index and data with every tensor already in its kernel's layout; "
    "streaming writer and validating mmap reader",
    [("format.container/strixw/-", "strix::StrixwFile, strix::StrixwWriter, strix::StrixHasher",
      "bytes as stored, hashed", "host; 4096-byte tensor alignment, 256 per component")])
fmt("ngram_table", "ngram.table: the PLE n-gram embedding table as a flat file of fixed-size rows",
    [("format.container/ngram-table/-", "strix::NgramTableFile, strix::NgramTableWriter",
      "rows stored as bf16 or int8 (e4m3 scale) per the header's dtype", "host; one positioned read per row")])
fmt("convert_qwen4exp", "Offline converter library: Qwen3.8-Flash-Next HF safetensors to weights.strixw",
    [("format.convert/strixw/-", "strix::convert_qwen4exp, strix::checkpoint_fingerprint",
      "per-class Q4/Q5/Q6/Q8 quantization by layout; HC norm weight folded before quantizing; router kept bf16",
      "host, streaming, all cores")])
for q, extra_api in (("q4", ", Q4ChunkMajorDevice/View"), ("q5", ""), ("q6", ""), ("q8", ", Q8ChunkMajorDevice/View")):
    Q = q.upper()
    M("format", f"{q}_device", f"{Q} weight resident on the device and the non-owning view the kernels take",
      [f"runtime/{q}_device.hpp"],
      [(f"format.device-view/strixw-{q}/-", f"strix::{Q}Device, strix::{Q}DeviceView{extra_api}",
        "bytes as stored in the host format", "header only; codes, scale and min as device pointers")])
M("format", "strixw_loader", "strixw weights onto the device: one allocation, concurrent pread into two pinned "
  "staging buffers with event-gated async copies",
  pair("runtime/strixw_loader", ("cpp", "hpp")),
  [("format.device-loader/strixw/-", "strix::StrixwDevice, strix::WeightPlacement", "bytes as stored",
    "tensors keep file offsets and alignment; HostExperts placement is for test and bench tools only")])
M("format", "safetensors_loader", "Bulk safetensors loader into pinned host memory with concurrent pread workers",
  pair("runtime/weight_loader", ("cpp", "hpp")),
  [("format.device-loader/safetensors/-", "strix::WeightLoader, strix::read_file_parallel", "bytes as stored",
    "buffered or O_DIRECT; drops the file from the page cache after the read")])

# --- runtime: device memory ----------------------------------------------------------------------------------
M("runtime", "device_buffer", "Owning, named, move-only device and pinned-host allocations with checked copies",
  ["runtime/device_buffer.hpp"],
  [("device.buffer/-/-", "strix::DeviceBuffer, strix::PinnedHostBuffer, strix::GpuMemoryTally", "n/a",
    "header only; hipMalloc comes out of GTT on gfx1151")])
M("runtime", "paged_buffer", "Device array that takes physical pages as it is used, inside one reserved address range",
  ["runtime/paged_buffer.hpp"],
  [("device.paged-buffer/-/-", "strix::PagedDeviceBuffer, strix::VmmReservation, strix::PageHandle", "n/a",
    "hipMemAddressReserve + hipMemCreate; pages are never unmapped at runtime and only map back into their own "
    "reservation (gfx1151 hazards)")])
M("runtime", "paged_state", "Host-only description of a parked conversation's K/V pages (no HIP handles)",
  ["runtime/paged_state.hpp"],
  [("kvcache.parked-state/-/-", "strix::PagedState, strix::PagedChunk", "n/a", "header only, host")])
M("runtime", "qweight", "A dense projection's weight in whichever Q format the layout gave it, dispatching to the "
  "matching linear kernel",
  ["runtime/qweight.hpp"],
  [("linear.dispatch/strixw/-", "strix::QWeightView, strix::QChunkMajorView, strix::kernels::linear_qw",
    "that of the kernel it dispatches to", "header only; format read from the strixw file, never guessed")])
M("runtime", "steer", "Activation steering sets: per-conversation vectors added to the residual streams after "
  "chosen layers",
  ["runtime/steer.hpp"],
  [("steer.set/-/-", "strix::SteerSet, strix::SteerLayer", "n/a",
    "header only; the request's cache key carries the steer salt as its first token")])
M("runtime", "ple_hash", "PLE n-gram ids on the host: 16 integer hashes per token selecting n-gram table rows",
  pair("runtime/ple_hash", ("cpp", "hpp")),
  [("ple.ngram-ids/-/-", "strix::ple_ngram_ids, strix::PleHistory, strix::PleHashParams",
    "integer hashing, validated against the HF reference ids", "host")])
M("runtime", "ngram_table", "PLE n-gram rows read on demand from the converted table file, deduplicated, with an "
  "LRU row cache",
  pair("runtime/ngram_table", ("cpp", "hpp")),
  [("ple.ngram-rows/ngram-table/-", "strix::NgramRowSource, strix::NgramTableRows", "rows as stored",
    "up to 64 concurrent preads; the table stays on SSD")])
M("runtime", "ngram_rows", "PLE n-gram rows straight from the HF checkpoint's shards (interim source kept for tests)",
  pair("runtime/ngram_rows", ("cpp", "hpp")),
  [("ple.ngram-rows/safetensors/-", "strix::NgramRowsFromShards", "the checkpoint's own bf16 rows",
    "one 4 KiB page read per row, no caching")])

# --- runtime: host services (serve/) -------------------------------------------------------------------------
def svc(kind, name, summary, files, provides, **build):
    M(kind, name, summary, [f"serve/{f}" for f in files], provides, **build)


svc("runtime", "json", "JSON value, parser and writer", ["json.cpp", "json.hpp"],
    [("host.json/-/-", "strix::json::Value", "n/a", "host, self-contained")])
M("runtime", "unicode", "UTF-8 coding, character classes and NFC with tables derived from the Unicode Character "
  "Database", ["serve/unicode.cpp", "serve/unicode.hpp", "serve/unicode_tables.inc"],
  [("text.unicode/-/-", "strix::unicode::{decode_utf8, encode_utf8, char_class, nfc}",
    "Unicode 16.0.0 categories, 15.0.0 normalization data", "host")],
  spdx="AGPL-3.0-only AND Unicode-3.0")
svc("runtime", "tokenizer", "BPE tokenizer reading the checkpoint's tokenizer.json",
    ["tokenizer.cpp", "tokenizer.hpp"],
    [("text.tokenizer/-/-", "strix::Tokenizer", "splits and normalizes as the model's reference tokenizer", "host")])
svc("runtime", "chat_template", "The model's chat template rendered in C++ (messages, tools, thinking switches)",
    ["chat_template.cpp", "chat_template.hpp"],
    [("text.chat-template/-/-", "strix::render_chat, strix::ChatTemplateOptions", "n/a", "host")])
svc("runtime", "sampler", "Next-token sampling on the host: temperature, top-k, top-p, logit bias and masks, from a "
    "full row or the GPU's candidates",
    ["sampler.cpp", "sampler.hpp"],
    [("sample.next-token/-/-", "strix::Sampler, strix::SamplingParams, strix::LogitRows, strix::LogitMasks",
      "transformers' warper order; temperature 0 is argmax with the lowest id on a tie; candidates path "
      "bit-identical to the full row", "host; only ids below n_valid can come out")])
svc("runtime", "grammar", "Byte-level grammar, automaton and matcher for structured output",
    ["grammar.cpp", "grammar.hpp"],
    [("grammar.matcher/-/-", "strix::grammar::{Grammar, GrammarAutomaton, GrammarMatcher}", "n/a", "host")])
svc("runtime", "json_schema", "JSON Schema to grammar compiler", ["json_schema.cpp", "json_schema.hpp"],
    [("grammar.json-schema/-/-", "strix::json_schema::compile, compile_any_object", "n/a", "host")])
svc("runtime", "token_mask", "Per-row allowed-token bit masks from a grammar state",
    ["token_mask.cpp", "token_mask.hpp"],
    [("grammar.token-mask/-/-", "strix::TokenMasker", "n/a", "host; masks are what logits_topk consumes")])
svc("runtime", "structured_output", "Structured output state for a generation: grammar cursor and row masks",
    ["structured_output.cpp", "structured_output.hpp"],
    [("grammar.structured-output/-/-", "strix::StructuredOutput", "n/a", "host")])
svc("runtime", "grammar_cache", "Cache of compiled grammars keyed by schema",
    ["grammar_cache.cpp", "grammar_cache.hpp"],
    [("grammar.cache/-/-", "strix::GrammarCache", "n/a", "host")])
svc("runtime", "tool_grammar", "Tool-call grammar: the grammar a request's tools allow, with the call tags as their "
    "added tokens and tool_choice required / named enforced",
    ["tool_grammar.cpp", "tool_grammar.hpp"],
    [("grammar.tool-call/-/-", "strix::tool_grammar::compile", "n/a", "host; call tags are the tag tokens, never "
      "their text spelling")])
svc("runtime", "output_parser", "Incremental parser of the model's output into content, reasoning and tool calls",
    ["output_parser.cpp", "output_parser.hpp"],
    [("text.output-parser/-/-", "strix::OutputParser, strix::OutputEvent, strix::ToolCallOut", "n/a",
      "host; streaming and buffered give the same events")])
svc("runtime", "logprobs", "Per-token logprobs from full logits rows", ["logprobs.cpp", "logprobs.hpp"],
    [("sample.logprobs/-/-", "strix::LogprobRecorder, strix::row_logprobs", "log-softmax in double over the valid "
      "ids", "host; needs full rows, not candidates")])
svc("runtime", "think_nudge", "Detector for circling thinking and the nudge text written into the think block",
    ["think_nudge.cpp", "think_nudge.hpp"],
    [("serve.think-nudge/-/-", "strix::ThinkWatch", "n/a", "host; 8-token repeats over a 1024-token window")])
svc("runtime", "resume_loss", "Describes what a resume point loses when an edited message forces a re-prefill",
    ["resume_loss.cpp", "resume_loss.hpp"],
    [("serve.resume-loss/-/-", "strix::resume_loss_text, strix::describe_prompt_position", "n/a", "host")])
svc("runtime", "capture", "Capture files of a generation's steps for replay and debugging",
    ["capture.cpp", "capture.hpp"],
    [("serve.capture/-/-", "strix::write_capture, strix::read_capture, strix::CaptureRecord", "n/a", "host")])
svc("runtime", "serve_log", "Request-block log helpers on top of common/log.hpp", ["log.hpp"],
    [("host.log-rows/-/-", "strix::slog_row, strix::fmt_n, strix::fmt_rate", "n/a", "header only")])
svc("runtime", "host_buffer", "Growable host buffer backed by mmap that returns its memory on release",
    ["host_buffer.cpp", "host_buffer.hpp"],
    [("host.buffer/-/-", "strix::HostBuffer", "n/a", "host")])
svc("runtime", "memory_governor", "RAM floor check: how much room MemAvailable leaves above a margin",
    ["memory_governor.hpp"],
    [("host.memory-governor/-/-", "strix::MemoryGovernor", "n/a", "header only")])
svc("runtime", "memory_plan", "Startup memory plan: weights, K/V budget, prompt cache and margins against the "
    "machine's RAM", ["memory_plan.cpp", "memory_plan.hpp"],
    [("serve.memory-plan/-/-", "strix::plan_memory, strix::MemoryPlan", "n/a", "host, once at startup")])
svc("runtime", "prompt_cache", "Prompt cache: token-prefix keyed RAM tier with disk write-back, parked "
    "conversations, export and import",
    ["prompt_cache.cpp", "prompt_cache.hpp"],
    [("cache.prompt/-/-", "strix::PromptCache, strix::PromptCacheStats",
      "restored state is the saved bytes; keys carry the weights, K/V dtype, MTP and YaRN fingerprint",
      "host only (no HIP on its compile line); disk tier gated by a free-space margin")])
svc("runtime", "vision_preprocess", "Image decode (PNG, JPEG, GIF, BMP, TGA, WebP) and the model's resize and "
    "patch preprocessing", ["vision_preprocess.cpp", "vision_preprocess.hpp"],
    [("vision.preprocess/-/-", "strix::vision::{decode_image, preprocess_image, resize_bicubic_aa_u8}",
      "antialiased bicubic resize on u8 as the reference processor", "host")])
svc("runtime", "vision_request", "Images of a chat request: data-URL parsing, decode memo, prompt planning and "
    "image-pad expansion", ["vision_request.cpp", "vision_request.hpp"],
    [("vision.request/-/-", "strix::vision::{prepare_images, plan_vision_prompt, load_vision_config, ServedVision}",
      "n/a", "host")])
svc("runtime", "engine", "The generation engine: LmBackend ABI, request loop, MTP draft and verify, slots and "
    "batched sequences (SeqBackend), thinking budget, structured output hooks",
    ["engine.cpp", "engine.hpp", "seq_backend.cpp", "seq_backend.hpp"],
    [("serve.engine/-/-", "strix::Engine, strix::LmBackend, strix::SeqBackend, strix::GpuArbiter, "
      "strix::GenerationRequest", "greedy and sampled token choice delegated to the sampler; an MTP draft only "
      "proposes, the verify forward decides every token", "host; one GPU, conversations batched per forward")])

# --- executors -----------------------------------------------------------------------------------------------
M("executor", "qwen4exp", "Qwen3.8-Flash-Next (qwen4exp) text model: 4 hyper-connected streams, 36 GDN + 12 "
  "QSA full-attention layers, PLE, MoE (top-10 of 512 + shared), MTP draft head, sessions with paged K/V",
  pair("runtime/qwen4exp", ("cpp", "hpp")),
  [("executor.forward/qwen4exp/bf16", "strix::Qwen4ExpModel, strix::Qwen4ExpSession",
    "bf16 activations in the server; fp32 math inside every kernel; prefill may use the WMMA lower-precision "
    "paths per PrecisionOptions; decode and verify use the exact kernels",
    "weights.strixw resident on the device; decode, chunked prefill, MTP verify, batched runs")])
M("executor", "qwen4exp_backend", "Adapter from the qwen4exp session to the engine's LmBackend ABI",
  pair("serve/qwen4exp_backend", ("cpp", "hpp")),
  [("executor.backend/qwen4exp/-", "strix::Qwen4ExpBackend", "that of executor.strixite.qwen4exp",
    "what the server binds the engine to")])
M("executor", "vision_tower", "The 0.45B ViT and merger resident on the device from vision.strixw",
  pair("runtime/vision_tower", ("cpp", "hpp")),
  [("executor.vision-tower/qwen4exp/bf16", "strix::VisionTower",
    "bf16 tower weights bit-exact from the checkpoint; fp32 math, one rounding where an op stores",
    "one image per forward (temporal axis 1)")])
M("executor", "qwen3_5", "Qwen3.5 text model: 24 layers (GDN or full attention + SwiGLU MLP), BF16 weights, the "
  "stage-1 correctness baseline",
  pair("runtime/qwen3_5", ("cpp", "hpp")),
  [("executor.forward/qwen3.5/bf16", "strix::Qwen35Model, strix::Qwen35Session",
    "bf16 linears through the simple linear_bf16w kernel", "not served; kept as the baseline forward")])

# --- serve ---------------------------------------------------------------------------------------------------
svc("serve", "openai", "OpenAI request parsing and response bodies (chat completions, completions, chunks, errors)",
    ["openai.cpp", "openai.hpp"],
    [("serve.api/openai/-", "strix::parse_chat_request, strix::parse_completion_request, strix::completion_body",
      "n/a", "host")])
svc("serve", "http", "HTTP server (cpp-httplib) with the OpenAI routes, health, cache and metrics endpoints, and "
    "the server's option table and config file",
    ["http_server.cpp", "http_server.hpp", "server_config.cpp", "server_config.hpp"],
    [("serve.http/openai/-", "strix::HttpServer, strix::ServerConfig, strix::ServerSettings", "n/a",
      "POST /v1/chat/completions, /v1/completions; GET /v1/models, /health, /cache, /metrics")])
svc("serve", "steer_library", "Steering vector library loaded from a directory and resolved per request",
    ["steer_library.cpp", "steer_library.hpp"],
    [("serve.steer-library/-/-", "strix::SteerLibrary", "vectors uploaded as stored", "needs the device")])
M("serve", "strix_server", "The server program: settings, memory plan, model and vision load, engine and HTTP "
  "wiring, signal handling",
  ["tools/strix_server.cpp"],
  [("serve.program/qwen4exp/-", "main (strix_server)", "n/a", "one process, one GPU")])

# --- third party vendored in the origin tree -----------------------------------------------------------------
M("serve", "cpp_httplib", "cpp-httplib 0.56.0, vendored unchanged by the origin",
  ["third_party/cpp-httplib/httplib.h", "third_party/cpp-httplib/LICENSE", "third_party/cpp-httplib/README.md"],
  [("serve.http-lib/-/-", "httplib::Server", "n/a", "header only")],
  spdx="MIT", authors=["Yuji Hirose"], includes=["src/third_party/cpp-httplib"], system=True)
M("runtime", "stb", "stb_image 2.30 and stb_image_write 1.16, vendored unchanged by the origin, with the one "
  "implementation translation unit",
  ["third_party/stb/stb_image.h", "third_party/stb/stb_image_write.h", "third_party/stb/README.md",
   "third_party/stb_impl/stb_impl.cpp"],
  [("vision.image-decode/stb/-", "stbi_load_from_memory, stbi_write_png", "n/a",
    "PNG, JPEG, GIF, BMP, TGA; stb_image_write is compiled in for the tests' encode side")],
  spdx="MIT OR Unlicense", authors=["Sean Barrett and contributors", "shawnshekari"],
  includes=["src/third_party/stb"], system=True, defines=["STRIX_STB_IMPL_TEST_ONLY"],
  options=["-Wno-missing-field-initializers"])

# --- tools ---------------------------------------------------------------------------------------------------
def tool(name, summary, regime, extra=()):
    M("tool", name, summary, [f"tools/{name}.cpp"] + list(extra),
      [(f"tool.{name.replace('_', '-')}/-/-", f"main ({name})", "n/a", regime)])


tool("convert_qwen4exp", "Converter CLI: HF checkpoint to weights.strixw (and vision.strixw with --vision)",
     "host only")
tool("convert_ngram_table", "Converter CLI: the checkpoint's n-gram shards to ngram.table", "host only")
tool("transcode_ngram_table", "Makes the int8 n-gram table from the BF16 one", "host only")
tool("inspect_strixw", "Prints a strixw file's header, components and tensors", "host only")
tool("check_stream_calls", "Test: streamed tool calls against the buffered parser, on the model's tokenizer.json",
     "host only test; check_stream_calls <tokenizer.json> [random cases] [seed]")
tool("check_logprobs", "Test: logprob math, the engine's capture sites and the response shape against a scripted "
     "fake backend", "host only test; check_logprobs <tokenizer.json>")
tool("sv9_cache_check", "Test: the prompt cache keys on image content",
     "host only test; sv9_cache_check <tokenizer.json> <preprocessor_config.json> <config.json> <scratch_dir>")
tool("cache_xfer_check", "Test: the prompt cache's export and import transfer API on synthetic states",
     "host only test; cache_xfer_check <scratch_dir>")
tool("tool_grammar_check", "Test: the tool-call grammar against the model's tokenizer (tags, free text, "
     "tool_choice)", "host only test; tool_grammar_check <tokenizer.json>")
tool("strix_vision", "One image through preprocess, the vision tower and the merger; HF-golden per-stage gate and "
     "encode bench", "needs the GPU and vision.strixw",
     extra=["tools/vision_batch_test.sh", "tools/vision_check.py"])
tool("omf_seg", "Batch export, import and probe of session states (K/V segment experiments)", "needs the GPU")
tool("strix_acts", "Mean activations of prompt sets at layer probe points, for steering directions", "needs the GPU")
tool("steer_check", "Test: a run's steering never touches another run of its batch, bit for bit",
     "GPU test; loads the model")

UNHARVESTED = [
    ".gitignore", "CMakeLists.txt", "CMakePresets.json", "CONTRIBUTING.md", "README.md", "LICENSE",
    "THIRD_PARTY.md", "assets/**", "deploy/**", "docs/**", "tutorial/**",
]
LICENCE_FILES = ["LICENSE", "THIRD_PARTY.md"]
SOURCE_EXT = (".cpp", ".hip", ".c")
UPSTREAM_AUTHORS = {"Shawn Reed": "shawnshekari"}


def webp_module(checkout):
    files = []
    for d, _, names in os.walk(os.path.join(checkout, "third_party/libwebp")):
        for n in names:
            files.append(os.path.relpath(os.path.join(d, n), checkout))
    M("runtime", "libwebp", "libwebp 1.6.0 decoder subset (x86), vendored unchanged by the origin, built as C11",
      sorted(files),
      [("vision.image-decode/webp/-", "WebPDecodeRGBA, WebPGetInfo", "n/a", "decoder only")],
      spdx="BSD-3-Clause", authors=["Google"], includes=["src/third_party/libwebp"], system=True,
      c_std=11, options=["-O2", "-w", "-fPIC"])


def load_authors(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            f, _, names = line.rstrip("\n").partition("\t")
            out[f] = [n for n in names.split(",") if n]
    return out


def module_authors(m, authors):
    if "authors" in m["build"]:
        return m["build"]["authors"]
    seen = set()
    for p in m["paths"]:
        for a in authors.get(p, ["Shawn Reed"]):
            seen.add(UPSTREAM_AUTHORS.get(a, "pod"))
    return sorted(seen, key=lambda a: (a != "shawnshekari", a))


def expand_provides(m):
    out = []
    for iface, api, numeric, regime in m["provides"]:
        if iface.endswith("/*"):
            for a in m["acts"]:
                out.append(dict(interface=iface[:-1] + a, api=api, numeric=numeric, regime=regime))
        else:
            out.append(dict(interface=iface, api=api, numeric=numeric, regime=regime))
    return out


def by_interface(consumer, provider):
    """Kernels with sources and formats are named by interface from other kinds; device views always."""
    if provider["kind"] == "format" and provider["name"].endswith("_device"):
        return True
    if consumer["kind"] in ("kernel", "format"):
        return False
    if provider["kind"] == "format":
        return True
    return provider["kind"] == "kernel" and any(p.endswith(SOURCE_EXT) for p in provider["paths"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkout")
    ap.add_argument("--commit", default="d19945e")
    ap.add_argument("--branch", default="v0.7.0-pod")
    args = ap.parse_args()
    checkout = os.path.abspath(args.checkout)
    out_root = os.path.join(ROOT, "catalog", ORIGIN)
    webp_module(checkout)
    authors = load_authors(os.path.join(out_root, "origin-authors.tsv"))

    owner = {}
    for m in MODULES:
        m["id"] = f"{m['kind']}.{ORIGIN}.{m['name']}"
        m["incdirs"] = [d[len("src/"):] if d != "src" else "" for d in m["build"].get("includes", ["src"])]
        for p in m["paths"]:
            if p in owner:
                sys.exit(f"{p} is in both {owner[p]['id']} and {m['id']}")
            if not os.path.isfile(os.path.join(checkout, p)):
                sys.exit(f"{m['id']}: {p} is not in the checkout")
            owner[p] = m

    # Every include root any module exports, as origin-relative directories.
    roots = sorted({d for m in MODULES for d in m["incdirs"]})

    def resolve(src, name, quoted):
        cands = [os.path.normpath(os.path.join(os.path.dirname(src), name))] if quoted else []
        cands += [os.path.normpath(os.path.join(r, name)) for r in roots]
        for c in cands:
            if c in owner:
                return c
        return None

    for m in MODULES:
        deps = {}
        for p in m["paths"]:
            if not inc.is_c_family(p):
                continue
            for name, quoted in inc.scan(os.path.join(checkout, p)):
                hit = resolve(p, name, quoted)
                if hit is None:
                    if quoted:
                        print(f"note: {p}: #include \"{name}\" resolves to no harvested file")
                    continue
                if owner[hit] is not m:
                    deps[owner[hit]["id"]] = owner[hit]
        m["deps"] = deps

    for m in MODULES:
        mdir = os.path.join(out_root, m["kind"], m["name"])
        gate = dict(parity="none", bench="none", status="harvested")
        design = None   # a module's DESIGN.md is written here, not harvested: kept across a re-harvest
        if os.path.isdir(mdir):
            if os.path.isfile(os.path.join(mdir, "DESIGN.md")):
                with open(os.path.join(mdir, "DESIGN.md"), "rb") as f:
                    design = f.read()
            # Gate results stay with a module only while it still holds the commit they were measured on.
            try:
                with open(os.path.join(mdir, "module.toml"), "rb") as f:
                    old = tomllib.load(f)
                if old.get("provenance", {}).get("commit") == args.commit:
                    gate = old.get("gate", gate)
            except (OSError, tomllib.TOMLDecodeError):
                pass
            shutil.rmtree(mdir)
        if design is not None:
            os.makedirs(mdir, exist_ok=True)
            with open(os.path.join(mdir, "DESIGN.md"), "wb") as f:
                f.write(design)
        for p in m["paths"]:
            dst = os.path.join(mdir, "src", p)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.join(checkout, p), dst)
            shutil.copymode(os.path.join(checkout, p), dst)
        b = m["build"]
        sources = ["src/" + p for p in m["paths"] if p.endswith(SOURCE_EXT)]
        lang = "hip-c++" if any(s.endswith(".hip") for s in sources) else \
            "c" if any(s.endswith(".c") for s in sources) else "c++"
        build = dict(lang=lang, sources=sources, includes=b.get("includes", ["src"]))
        if lang == "hip-c++":
            build["arch"] = "gfx1151"
        build["toolchain"] = "therock-10.1"
        for k in ("system", "uses", "defines", "options", "c_std"):
            if k in b:
                build[k] = b[k]
        requires = []
        for dep_id in sorted(m["deps"]):
            dep = m["deps"][dep_id]
            if by_interface(m, dep):
                for p in expand_provides(dep):
                    if p["interface"].rsplit("/", 1)[1] in ("bf16", "-"):
                        requires.append({"interface": p["interface"]})
            else:
                requires.append({"module": dep_id})
        doc = [
            ("", dict(id=m["id"], kind=m["kind"], summary=m["summary"], **({"design": "DESIGN.md"} if design else {}))),
            ("provenance", dict(origin=ORIGIN, repo=REPO, commit=args.commit, branch=args.branch, paths=m["paths"],
                                authors=module_authors(m, authors), modified=False)),
            ("licence", dict(spdx=b.get("spdx", AGPL))),
            ("build", build),
            ("[provides]", expand_provides(m)),
            ("[requires]", requires),
            ("gate", gate),
        ]
        with open(os.path.join(mdir, "module.toml"), "w") as f:
            f.write(tomlw.dump(doc))

    for lf in LICENCE_FILES:
        shutil.copyfile(os.path.join(checkout, lf), os.path.join(out_root, lf))
    kinds = {}
    for m in MODULES:
        kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
    print(f"{len(MODULES)} modules, {len(owner)} files:", ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    print("unharvested globs for ORIGIN.toml:", UNHARVESTED)


if __name__ == "__main__":
    main()
