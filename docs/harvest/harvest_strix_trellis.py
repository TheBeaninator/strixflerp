#!/usr/bin/env python3
"""Harvest strix-trellis (hyukudan/strix-trellis, a34516b) into catalog/strix_trellis.
Run: python3 docs/harvest/harvest_strix_trellis.py

Card ORIGIN-TRELLIS 41e8802da1; verdict of KSRC-SURVEY f7de2c34b1 (/data/pod/reports/2026-11/ksrc-survey.md, row 2).
Scope: the HIP runtime (src/, less its benches and probes), the C header (include/strix_trellis.h), the origin's
CMake build file and the NumPy CPU reference with its safetensors loader (python/). bench/, tests/, tools/,
experiments/ and docs/ are listed as unharvested.

strix-trellis builds one static library, libstrix_trellis.a (CMake target strix_trellis::runtime): host C++ and
device kernels in the same .hip translation units, launched by the library itself on a borrowed hipStream_t. That
is another shape than StrixFold's kit of device-only code objects bound by id; catalog/strix_trellis/ABI-DIFF.md
lists what an adapter has to bridge. Nothing is compiled or run by this harvest, so every module is
[gate] status = "harvested".

The reference checkout is the ark copy on control-node (/data/ark-inbox/src/github.com__hyukudan__strix-trellis, a
depth-1 clone); the laptop holds the same commit, moved by git bundle, in
~/Code/strixflerp-upstream-supplement/strix-trellis. STRIX_TRELLIS_CHECKOUT names another copy."""
import os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvestlib
from harvestlib import Origin, M, P, run

CHECKOUT = os.environ.get('STRIX_TRELLIS_CHECKOUT', os.path.join(harvestlib.SUPPLEMENT, 'strix-trellis'))
COMMIT = 'a34516bf91b9c66166bf6d3678d10fc68d6c122c'


class GitOrigin(Origin):
    """The origin's file set is git's (git ls-files of the checkout), and the checkout must sit on COMMIT."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.roots = [CHECKOUT]
        head = subprocess.run(['git', '-C', CHECKOUT, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        if head != COMMIT:
            raise SystemExit(f'{CHECKOUT} is at {head}, the harvest pins {COMMIT}')
        dirty = subprocess.run(['git', '-C', CHECKOUT, 'status', '--porcelain'], capture_output=True, text=True, check=True).stdout
        if dirty.strip():
            raise SystemExit(f'{CHECKOUT} has local changes; the harvest copies files verbatim from the commit')

    def files(self):
        if self._files is None:
            out = subprocess.run(['git', '-C', CHECKOUT, 'ls-files'], capture_output=True, text=True, check=True).stdout
            self._files = sorted(f for f in out.splitlines() if f)
        return self._files


O = GitOrigin('strix_trellis', 'strix-trellis', 'https://github.com/hyukudan/strix-trellis', COMMIT, 'master',
              'Apache-2.0', ['hyukudan'], 'therock-10.1')
S = 'src/'
LIB = 'a translation unit of libstrix_trellis.a: built by the origin CMake (STRIX_TRELLIS_ENABLE_HIP=ON, -O3 -ffp-contract=off, HIP standard 17, gfx1151), not compiled by this harvest'
TP = LIB + '; in the library only with STRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP=ON'
DEC = 'decode, 1..8 rows a call, memory-bound (the origin measures against a 246.7 GB/s cold read ceiling, README.md:231)'
PF = 'prefill, rank-2 input of any row count (register-blocked core from 128 rows), compute-bound on wave32 WMMA'
EXL3 = 'EXL3 format after exllamav3 (MIT, turboderp) and QTIP (Tseng et al., NeurIPS 2024); read for the format and its arithmetic, no upstream code included (NOTICE, docs/SOURCES.md)'
mods, claimed = [], set()


def take(names):
    out = []
    for n in names:
        hits = [f for f in O.expand([n if '/' in n or n == 'CMakeLists.txt' else S + n]) if f not in claimed]
        if not hits:
            raise SystemExit(f'nothing left for {n}')
        out += hits
    claimed.update(out)
    return out


def mod(id, tag, summary, names, provides, notes, lang='hip-c++', compile='n/a', reason=LIB, requires=(), sources=None):
    """tag: the strixflerp interface class the module matches (linear.decode/exl3, linear.prefill/exl3) or 'none'."""
    files = take(names)
    if sources is None:
        sources = [f for f in files if f.endswith('.hip')] if compile != 'header-only' else []
    incs = sorted({'src/' + os.path.dirname(f) for f in files if f.endswith(('.h', '.hpp', '.hip', '.inc'))})
    mods.append(M(id, summary, files, provides=provides, requires=requires, lang=lang, sources=sources, includes=incs,
                  compile=compile, reason=reason, notes='Interface tag: ' + tag + '. ' + notes, why=WHY[id], upstream=EXL3,
                  toolchain='' if lang == 'python' else None))


# Why each module exists (SPEC.md section 5; tools/describe-audit.py wants one per module, none shared).
WHY = {
 'kernel.strix_trellis.c_api': 'the survey gave strix-trellis a harvest verdict because it has a C header as its ABI where StrixFold has by-value kernel argument structs; the adapter question in ABI-DIFF.md is answered from this module, and nothing else in the origin can be called without it',
 'kernel.strix_trellis.trellis_common': 'every strix-trellis kernel decodes trellis states and applies H128 through these headers, so no unit can be read or compiled without them; the mul1 decode by one udot4 and the three-word lane window are the parts to compare with StrixFold\'s once-a-wave tile decode',
 'kernel.strix_trellis.exl3_gemv': 'decode on Strix Halo is weight bytes over memory bandwidth, and this is a second, independently written EXL3 decode product for the DeepSeek-V4.1-Flash shapes StrixFold serves: the origin reports 199.7 GB/s on q_b at one row, 80.9 % of its read ceiling (README.md:285), a number StrixFold\'s exl3 module has no like-for-like counterpart to yet',
 'kernel.strix_trellis.exl3_wmma_gemm': 'docs/SHAPE-TRANSFERS.md records that StrixFold\'s prompt experts pay about 12 ms a pass that is neither trellis decode nor DRAM bytes; a second EXL3 prefill product with its own tile schedule (register-blocked, activation rows reused while hot) is the comparison that surface lacks',
 'kernel.strix_trellis.exl3_tp_dense': 'StrixFold serves DeepSeek-V4.1-Flash on two ranks, and these kernels are written for exactly the per-rank halves of its projections with the whole operator in one launch, where the generic path spends up to three (input transform, core, output transform); the launch count per projection is what a decode step of many small linears pays',
 'kernel.strix_trellis.exl3_dense_r1': 'small attention projections sit far below the read ceiling because one launch\'s fixed cost dominates them; the R1 plans add output tiling without cutting an accumulation chain, which bought 1.43x on q_a at one row (README.md:475) while keeping each row equal to its own single-row call, the row invariance StrixFold needs for drafts',
 'kernel.strix_trellis.exl3_tp_moe': 'kernel.strixfold.exl3 serves the decode experts with an epoch / ready protocol over bf16 rows; this is the origin\'s K2 / K3 answer for the same top-6 block in two launches (one grouped gate / up, one down with the ordered combine), kept so the two launch plans can be compared when the pair is up; the origin itself has not run it in a full model',
 'kernel.strix_trellis.exl3_variants': 'the origin keeps the decode and WMMA kernels its shipped ones replaced, with a bench each; they record which schedules were tried beside the shipped ones on gfx1151 (the blocked-K search is written up as a negative result), which tells the pod what not to re-run first',
 'tool.strix_trellis.reference': 'a kernel swap between engines is only safe against a reference that fixes every rounding stage; this NumPy model decodes all 65,536 states of each codebook exactly and names the stages of the linear, so it can judge StrixFold\'s exl3 output and strix-trellis\' on the same tensors without a GPU',
}

COMMON = 'module:kernel.strix_trellis.trellis_common'
GEMV = 'module:kernel.strix_trellis.exl3_gemv'
TPD = 'module:kernel.strix_trellis.exl3_tp_dense'

mod('kernel.strix_trellis.trellis_common', 'none',
    'Header-only device layer of every strix-trellis kernel: the DeviceWeights view and the launch_linear entry (backend.hpp), trellis state decode for the 3inst, mcg and mul1 codebooks (mul1 as state * 0x83DCD12D summed by one udot4), per-lane windows over the packed u32 tile stream (WaveTile, DirectLane, AlignedLane, SaddrLane), the wave-local normalised Hadamard H128 with the input and output transform kernels, the split-count and launch policies, and two host models (the codebook, and gfx11\'s fp16 dot2 accumulate bit for bit, dot2acc_reference.hpp)',
    ['backend.hpp', 'trellis_device.hpp', 'trellis_optimized.hpp', 'packed_decoder.hpp', 'hadamard.hpp', 'transforms.hpp',
     'launch_policy.hpp', 'v14_core_policy.hpp', 'gemv_policy.hpp', 'tp_fused_layout.hpp', 'host_reference.hpp', 'dot2acc_reference.hpp'],
    [P('kernel-abi.strix-trellis-device/-/-', 'strix_trellis::DeviceWeights, launch_linear, linear_split_count (backend.hpp); codebook_value, mul1_half2, hadamard128, input_transform, output_transform', 'n/a', 'n/a')],
    'gemv_policy.hpp holds the measured choice of kernel and K split count per shape, bit width and row count (exact_gemv_mode, measured_vector_splits): the split count fixes the order of the partial sums, so it is part of the numerics.',
    compile='header-only', reason='headers only; compiled inside every strix_trellis kernel unit')

mod('kernel.strix_trellis.exl3_gemv', 'linear.decode/exl3',
    'EXL3 decode matvec for 1 to 8 rows behind strix_gemv: the input is scaled by suh, rounded to fp16, H128-rotated and rounded to fp16 again; each lane decodes its trellis states from the packed stream and accumulates fp16 dot pairs in fp32 over K16 tiles in ascending order, in a fixed number of K splits; the split partials are summed, H128-rotated, scaled by svh, and bias, alpha and beta applied. Eight units cover the cases: the aligned baseline, an LDS-staged body, K2 single-row and multi-row cores, and fused forms that fold into the core the input transform, the output transform (the direct-output kernel with dual-issue dots and next-row prefetch, and its grouped form) or both',
    ['gemv.hip', 'gemv.hpp', 'gemv_lds.hip', 'gemv_lds.hpp', 'gemv_k2.hip', 'gemv_k2.hpp', 'gemv_k2_multirow.hip', 'gemv_k2_multirow.hpp',
     'gemv_fused.hip', 'gemv_fused.hpp', 'gemv_fused_both.hip', 'gemv_fused_both.hpp', 'gemv_output_direct.hip', 'gemv_output_direct.hpp',
     'gemv_grouped_direct.hip', 'fused_both_policy.hpp', 'fused_output_policy.hpp', 'fused_output_resources.hpp'],
    [P('linear.decode/exl3/f16-f32.strix-trellis', 'strix_gemv (include/strix_trellis.h) -> launch_linear (backend.hpp) -> the kernels of gemv.hip, gemv_lds.hip, gemv_k2.hip, gemv_k2_multirow.hip, gemv_fused.hip, gemv_fused_both.hip, gemv_output_direct.hip, gemv_grouped_direct.hip', 'Y = alpha * Linear(X) + beta * Y; fp32 X * fp16 suh -> fp16 -> H128 in fp32 -> fp16; exact codebook decode; fp16 dot pairs with fp32 accumulation, K16 ascending, S partial sums added in split order; H128, svh and bias in fp32 (docs/API.md "Operation and orientation", docs/NUMERICS.md)', DEC)],
    'Beside kernel.strixfold.exl3 (glinear) and kernel.crucible.exl3_gemv. Origin numbers, EAQ layer-0 tensors, RC5 / RC6: q_b 5-bit 131.302 us at 1 row = 199.7 GB/s, 142.458 us at 8 rows = 184.0 GB/s (README.md:285-286); output head 6-bit 1 row 215.9 GB/s (README.md:295). Not row-invariant across row counts as a whole: the K split count is chosen per shape and row count (trellis_common gemv_policy.hpp), and the plans of exl3_dense_r1 are what pin every row to its single-row bits. Shipped options: STRIX_TRELLIS_DIRECT_DUAL_ISSUE, STRIX_TRELLIS_DIRECT_PREFETCH (CMakeLists.txt:8-9). No timing was taken on the pod.',
    requires=[COMMON])

mod('kernel.strix_trellis.exl3_wmma_gemm', 'linear.prefill/exl3',
    'EXL3 prefill matmul behind strix_gemm on wave32 WMMA: the trellis tile is decoded to fp16 into LDS and multiplied 16x16x16 with fp32 accumulation over K16 tiles in ascending order per split (wmma_core, row tiles of 16 or 64; wmma_packed\'s decoder_core is that core with the packed-stream lane decoder, selected for measured mul1 5- and 6-bit shapes at 64, 128 and 256 rows); the register-blocked core (wmma_tiled tiled_core, mul1 at 4 to 6 bits from 128 rows) tiles 128x256, 256x128 or 128x128 by row count, visits two row blocks a column so activation rows are reused while hot, and with one split can apply the output transform in its own epilogue',
    ['gemm.hip', 'gemm.hpp', 'wmma_tiled.hip', 'wmma_tiled.hpp', 'wmma_packed.hip', 'wmma_packed.hpp'],
    [P('linear.prefill/exl3/f16-f32.strix-trellis', 'strix_gemm (include/strix_trellis.h) -> launch_linear(prefer_wmma) -> wmma_core, tiled_core, decoder_core', 'the fp16 operands of the decode path; v_wmma_f32_16x16x16_f16 with fp32 accumulation, split bounds and K16 order kept, so the register-blocked core is bit-identical to the previous core (README.md "Prefill (WMMA GEMM)"); fp32 reductions may differ from the GEMV path (docs/API.md)', PF)],
    'Beside kernel.strixfold.exl3 (exl3.gemm, exl3.unpack), kernel.strixfold.exl3_pf_experts and kernel.crucible.exl3_gemm. Origin numbers: q_b (1280, 32768) 5-bit, the register-blocked core against its previous revision, 2.22x at 128 rows and 1.81x at 256 rows (README.md:361-362). Run-time switches read from the environment: STRIX_TRELLIS_WMMA_TILED, STRIX_TRELLIS_WMMA_TILED_FUSED (auto fuses the epilogue from 21 blocks on gfx1151). No timing was taken on the pod.',
    requires=[COMMON])

mod('kernel.strix_trellis.exl3_tp_dense', 'linear.decode/exl3',
    'Per-rank dense projections for two-way tensor parallelism in one launch each: strix_tp_dense_plan freezes a matmul descriptor and its kernel does scales, input H128, trellis decode, product, output H128, bias and alpha / beta for 1 to 8 rows of a K4 or K5 mul1 weight (tp_dense projection); the split-K form partitions the input dimension across CTAs with partial planes and integer arrival counters in a caller workspace, as one kernel or as a split kernel and a reduce kernel (tp_splitk)',
    ['tp_dense.hip', 'tp_dense.hpp', 'tp_dense_policy.hpp', 'tp_splitk.hip', 'tp_splitk.hpp', 'tp_split_layout.hpp', 'tp_activation.hpp', 'tp_fused_device.hpp'],
    [P('linear.decode/exl3/f16-f32.strix-trellis-tp', 'strix_tp_dense_plan_create, _create_tiled, _create_splitk, strix_tp_dense_workspace_bytes, strix_tp_dense_plan_submit / _destroy, strix_tp_dense_pair_plan_*, strix_tp_swiglu_plan_*', 'the S4 identity: four K16 accumulator chains per output, each row the arithmetic of its own single-row call; a local rank contribution, the inter-rank reduction is the caller\'s (docs/TP_RANK.md section 1)', DEC + '; local shapes q_b (1280, 16384), o_b (4096, 5120), shared gate / up (5120, 1152) and down (1152, 5120), q_a (5120, 1280), kv (5120, 512) (README.md "Per-rank shapes")')],
    'Beside kernel.strixfold.exl3 (glinear on a rank of the F+D pair). The origin reports these plans in use at every attention projection of its two-node DeepSeek-V4.1-Flash engine (README.md "Status"); docs/TP_RANK.md and docs/TP_DECODE_PLAN.md in the origin describe the plans by rank and were not copied.',
    reason=TP, requires=[COMMON])

mod('kernel.strix_trellis.exl3_dense_r1', 'linear.decode/exl3',
    'Dense R1 plans: the split-K projection with physical output tiling added, so more CTAs share one projection without cutting any declared K accumulation chain. A CTA covers 8 to 128 outputs of one H128 group; partitions fold from fp32 +0 in ascending order before output H128, svh, bias, alpha and beta, and the completing CTA resets the counters. One unit per input width and split count (i1152 / i1280 / i4096 / i5120 at S4, i5120 at S16 and S32), a pair kernel for q_a and kv over one input, grouped kernels for up to 8 linears of one geometry (4096 to 1024) and an owner-fold kernel where the CTA owning an output H128 folds its partials',
    ['dense_r1.hip', 'dense_r1.hpp', 'dense_r1_device.hpp', 'dense_r1_layout.hpp', 'dense_r1_policy.hpp', 'dense_r1_i1152_s4.hip',
     'dense_r1_i1280_s4.hip', 'dense_r1_i4096_s4.hip', 'dense_r1_i5120_s4.hip', 'dense_r1_i5120_s16.hip', 'dense_r1_i5120_s32.hip',
     'dense_r1_group_i4096_s16.hip', 'dense_r1_group_i4096_s32.hip', 'dense_r1_group_owner.hip', 'dense_r1_pair.hip'],
    [P('linear.decode/exl3/f16-f32.strix-trellis-r1', 'strix_tp_dense_plan_create_r1, strix_tp_dense_r1_workspace_bytes, strix_tp_dense_pair_plan_create_r1, strix_grouped_gemv_plan_create / _submit / _destroy (STRIX_GEMV_ROW_INVARIANT_R1)', 'two declared identities: STRIX_DENSE_R1_S4 (four K16 chains, two fp32 accumulators a lane, four-lane reduction by shuffle offsets 2 then 1) and STRIX_DENSE_R1_LEGACY (S16 for 5120 to 1280 / 1152, S32 for 5120 to 512, S4 elsewhere); row count, tile width, arrival order and prefetch depth never change the association, so every row equals its single-row call (docs/DENSE_R1.md "Numerical identities")', DEC + '; the shapes of exl3_tp_dense plus the head 5120 to 64640 / 129280, K4 / K5 mul1')],
    'Beside kernel.strixfold.exl3 (glinear, row-invariant). Origin numbers, RC5, synthetic weights, LEGACY identity: q_a (5120, 1280) K4 at 1 row 21.09 us against 30.19 us through strix_gemv, 1.43x, bits equal (README.md:475); grouped o_a K5 at 1 row 51.27 us, 82.9 % of the read ceiling (README.md:545). Switching between S4 and LEGACY is a model identity decision in the origin\'s words, not a tolerance check. dense_r1_api.inc and dense_r1_group_api.inc, the host side of these plans, are fragments of api.hip in kernel.strix_trellis.c_api. No timing was taken on the pod.',
    reason=TP, requires=[COMMON, TPD])

mod('kernel.strix_trellis.exl3_tp_moe', 'none',
    'Routed experts of one rank for K2 or K3 mul1 weights, six routes a row, 1 to 8 rows: the first plan runs gate and up with SiLU as one grouped launch and down with the ordered top-6 combine as a second (tp_moe gate_up_kernel, down_kernel); the split3 plan stages A, B and C over a workspace holding the hidden rows and the down partials, optionally as a pretransformed S4 pipeline with separate prepare, core and finish nodes, one- and two-row cores and a grouped finish of all six down routes; routes come from the host at plan creation or from device tensors a persistent expert bank reads each submit. An A8R variant with integer group accumulation is a different arithmetic identity and is compiled out by default',
    ['tp_moe.hip', 'tp_moe.hpp', 'tp_moe_split.hip', 'tp_moe_split_layout.hpp', 'tp_moe_a8r.hpp', 'tp_moe_a8r_math.hpp', 'tp_moe_a8r_reference.hpp',
     'tp_moe_activation.hpp', 'tp_moe_finish.hpp', 'tp_moe_fold.hpp', 'tp_moe_gate_finish.hpp', 'tp_moe_grouped_finish.hpp', 'tp_moe_hash.hpp',
     'tp_moe_persistent.hpp', 'tp_moe_pipeline.hpp', 'tp_moe_policy.hpp', 'tp_moe_reader.hpp', 'tp_moe_routes.hpp', 'tp_moe_s4_reference.hpp',
     'tp_moe_schedule.hpp', 'tp_moe_weight_tile.hpp', 'k2_local_decode.hpp'],
    [P('moe.experts-decode/exl3/f16-f32.strix-trellis', 'strix_tp_moe_plan_create, _create_device, _create_split, _create_device_split, _create_a8r, strix_tp_moe_plan_submit, _gate_up, _down, _down_core, _down_finish (include/strix_trellis.h)', 'output = alpha * sum over slots of score * down(SiLU(gate(X)) * up(X)) + beta * output, slots combined in slot order, zero scores skipped; S4 / DOT2ACC arithmetic, bitwise equal to the 0.3 operator on GPU fixtures per the origin; the scores are not normalised and the clamp of DeepSeek\'s SwiGLU is a build option (STRIX_TRELLIS_TP_MOE_DECLARED_SWIGLU, off)', 'decode, 1..8 rows, top-6 of a bank of 6..256 experts; gate / up (5120, 1152), down (1152, 5120); the origin lists it as measured on GPU, not used in a full model')],
    'The card\'s tag set has no word for it: this module matches moe.experts-decode/exl3, the class of kernel.strixfold.exl3 (grouped_cp) and kernel.crucible.moe_decode, not a linear interface. Routing, scatter and inter-rank reduction are the caller\'s. The STRIX_TRELLIS_TP_MOE_* options of CMakeLists.txt:19-65 pick the pipeline at build time.',
    reason=TP, requires=[COMMON, TPD])

mod('kernel.strix_trellis.exl3_variants', 'none',
    'Kernels the origin keeps beside its library and does not link into it: combined and static-combined decode GEMVs with their decoder (gemv_combined, gemv_static_combined, combined_decoder.hpp), fused-groups and fused-output GEMVs, a K2 rows kernel, a low-bit K3 / K4 reader pipeline, the K2 single-row kernel with an LDS split reduction (gemv_k2_local, its own optional library strix_trellis_k2_local), the blocked-K WMMA core with its submission control and a WMMA core with a replaced lane decoder',
    ['gemv_combined.hip', 'gemv_combined.hpp', 'combined_decoder.hpp', 'gemv_static_combined.hip', 'gemv_static_combined.hpp',
     'gemv_fused_groups.hip', 'gemv_fused_groups.hpp', 'gemv_fused_output.hip', 'gemv_fused_output.hpp', 'gemv_k2_rows.hip', 'gemv_k2_rows.hpp',
     'gemv_lowbit.hip', 'gemv_lowbit.hpp', 'gemv_k2_local.hip', 'gemv_k2_local.hpp', 'k2_local_policy.hpp',
     'wmma_blocked_submit.hip', 'wmma_blocked_submit.hpp', 'wmma_decoder.hip', 'wmma_decoder.hpp'],
    [P('kernel-research.exl3-variants/exl3/-', 'the entry declared in each unit\'s header; driven by the origin\'s bench/*.hip programs, which are not harvested', 'each keeps the fp16 operands of the shipped path; orders differ per experiment (docs/K2_LOCAL.md, WMMA_DECODER.md, BLOCKED_K.md, LOWBIT_PIPELINE.md in the origin)', 'operator experiments; none is selected by strix_gemv or strix_gemm')],
    'CMakeLists.txt names none of these sources in libstrix_trellis.a; gemv_k2_local.hip builds only with STRIX_TRELLIS_BUILD_EXPERIMENTAL_K2_LOCAL=ON. The origin\'s design notes record the blocked-K search and its submission control as negative results (docs/WMMA_DECODER.md).',
    reason='not a unit of libstrix_trellis.a: CMakeLists.txt names none of these sources except gemv_k2_local.hip (the optional library strix_trellis_k2_local); the origin\'s bench programs drive them',
    requires=[COMMON, GEMV, 'module:kernel.strix_trellis.exl3_tp_moe'])

mod('kernel.strix_trellis.c_api', 'none',
    'strix-trellis\' C ABI v1 (include/strix_trellis.h) and the host code that answers it: descriptor validation, weight preparation from an EXL3 transport blob or from four separate host regions into one device allocation, workspace queries, the strix_gemv / strix_gemm dispatch (backend.hip: kernel and split selection, the staged or fused transforms), prepared plans, the pre-routed grouped MoE call, and the tensor-parallel plan creators that the .inc fragments add to api.hip; with the origin\'s CMake file, which defines the library and its options',
    ['include/strix_trellis.h', 'api.hip', 'backend.hip', 'dense_r1_api.inc', 'dense_r1_group_api.inc', 'tp_api.inc', 'tp_split_api.inc', 'CMakeLists.txt'],
    [P('kernel-abi.strix-trellis-c/-/-', 'strix_context_create / _destroy, strix_weights_prepare / _prepare_parts / _device_span / _destroy, strix_matmul_workspace_bytes, strix_gemv, strix_gemm, strix_linear_plan_*, strix_grouped_moe, strix_tp_dense_*, strix_grouped_gemv_plan_*, strix_tp_moe_* (STRIX_TRELLIS_ABI_VERSION 1, library 0.3.0)', 'n/a here: the operator Y = alpha * Linear(X) + beta * Y is stated in the origin\'s docs/API.md and each kernel module states its accumulation order', 'host calls that validate and enqueue on a borrowed hipStream_t; no synchronisation is promised and device errors surface through HIP')],
    'catalog/strix_trellis/ABI-DIFF.md compares this header with StrixFold\'s zig/kernels/hip/dsv41/abi.h (kernel.strixfold.dsv41_abi). api.hip is one translation unit; with STRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP off the TP creators return STRIX_ERROR_NOT_IMPLEMENTED (docs/TP_RANK.md). The origin README says its sibling strix-halo-engine-lab links this library for attention, shared-expert projections and the head on two nodes.',
    requires=[COMMON, GEMV, 'module:kernel.strix_trellis.exl3_wmma_gemm', TPD, 'module:kernel.strix_trellis.exl3_dense_r1',
              'module:kernel.strix_trellis.exl3_tp_moe'])

mod('tool.strix_trellis.reference', 'none',
    'NumPy CPU reference and loader: exact trellis decode of the 3inst, mcg and mul1 codebooks at integer rates of 1 to 8 bits (and 1.5 / 2.5 / 3.5 bits for mul1), a reference matmul with every rounding stage explicit (input scale to fp16, H128, core dot in fp32, output H128 and svh), a safetensors reader that maps an EXL3 group\'s trellis, suh, svh, codebook marker and optional bias without expanding the weights and exports them as the regions strix_weights_prepare_parts takes, and the per-rank partition of a tensor for two-way tensor parallelism',
    ['python/'],
    [P('verify.numerics-contract/exl3/-.strix-trellis', 'strix_trellis.matmul_fp32, matmul_fp32_stages, input_stages_fp32, core_dot_fp32, output_stages_fp32, decode_states, dequantize_rotated (python/strix_trellis/linear.py, codec.py)', 'fp32 stages with declared roundings; per the origin README it matches the EXL3 reference CUDA kernels bit for bit on 2,621,440 real weights and on all 65,536 states of each codebook', 'offline, CPU'),
     P('format.loader/exl3-safetensors/-.strix-trellis', 'strix_trellis.load_exl3, SafetensorsFile, Exl3Tensor (transport_regions, transport_buffer, transport_blob), rank_partition (python/strix_trellis/container.py, tp.py)', 'exact (bytes mapped; tiles kept in EXL3 encoding)', 'load time')],
    'A reference and a loader, not a kernel. Beside tool.crucible.reference (NumPy EXL3 codec and matvec) and StrixFold\'s numerics contract (numerics.json exl3.linear). Rejects external out_scales and older formats without an explicit scale contract (README.md "Status"). Not run by this harvest.',
    lang='python', compile='n/a', reason='python: not compiled', sources=[])

LEFT_SRC = ['design_probe.hip', 'gemm_bench.hip', 'gemm_reuse_bench.hip', 'gemv_bench.hip', 'gemv_lds_bench.hip', 'microbench.hip',
            'optimized_microbench.hip', 'wmma_probe.hip']
left = sorted(f[len(S):] for f in O.expand([S]) if f not in claimed)
if left != LEFT_SRC:
    raise SystemExit('src files in no module differ from the declared benches and probes: ' + ' '.join(left))

ORIGIN_TOML = '''
# strix-trellis: an independent inference library for EXL3 / QTIP trellis-quantised weights on AMD Strix Halo
# (gfx1151, wave32): a HIP runtime with a C ABI (trellis decode, GEMV for 1 to 8 rows, WMMA GEMM for prefill, H128
# transforms, tensor-parallel rank paths), a NumPy CPU reference and a safetensors loader. Developed and measured
# against DeepSeek V4.1 Flash, the model StrixFold serves (card ORIGIN-TRELLIS 41e8802da1).
name      = "strix_trellis"
repo      = "https://github.com/hyukudan/strix-trellis"
commit    = "a34516bf91b9c66166bf6d3678d10fc68d6c122c"
branch    = "master"
version   = "library 0.3.0, C ABI 1 (CMakeLists.txt:2, include/strix_trellis.h); head of 2026-10-30 (Merge pull request #2 from hyukudan/readme-rc7); the README names RC7 9230a97 as the release checkpoint"
reference_checkout = "control-node:/data/ark-inbox/src/github.com__hyukudan__strix-trellis (ark copy, a depth-1 clone at a34516b); laptop: ~/Code/strixflerp-upstream-supplement/strix-trellis (the same commit, moved by git bundle)"

[licence]
spdx      = "Apache-2.0"
files     = ["LICENSE", "NOTICE"]                          # copied verbatim next to this file
third_party = "NOTICE: the EXL3 format is defined by exllamav3 (MIT, turboderp) and QTIP is Tseng et al., NeurIPS 2024 (its code is GPL-3.0); no code from either project is included (docs/SOURCES.md lists the revisions read). Test fixtures come from the public EAQ v1 EXL3 checkpoint of DeepSeek V4.1 Flash (MIT) and are not harvested. Weights are not part of the tree."

[toolchain]
host      = "C11 / C++17, CMake 3.24 or newer; python3 with NumPy for the CPU reference (python/pyproject.toml)"
device    = "therock-10.1"                               # docs/BUILD.md:55: tools/build_remote.sh configures ROCm /opt/rocm/core-10.1 for gfx1151; the pod has not built it
build     = "cmake -DSTRIX_TRELLIS_ENABLE_HIP=ON [-DSTRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP=ON] -> libstrix_trellis.a (target strix_trellis::runtime, links hip::host); strix_trellis::api is the header alone. CMake's HIP language compiles every unit with -O3 -ffp-contract=off, HIP standard 17, CMAKE_HIP_ARCHITECTURES gfx1151 (CMakeLists.txt:197-210): 13 units by default, 28 with STRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP. There is no hipcc line per kernel, so this file has no [compile] recipe"

[kernels]
loading   = "linked into the caller: kernels are C++ templates launched with <<< >>> by the library's own host code on a borrowed hipStream_t; there is no code-object kit, manifest or kernel id (ABI-DIFF.md)"
selection = "launch_linear picks the kernel and the K split count from shape, bit width, codebook and row count (src/backend.hip, src/gemv_policy.hpp); environment switches for the WMMA core (STRIX_TRELLIS_WMMA_TILED, STRIX_TRELLIS_WMMA_TILED_FUSED; docs/API.md)"
models    = "DeepSeek V4.1 Flash: dense mul1 K4 with a K5 head (the origin's mix-B) in use in its engine, K2 / K3 routed experts at operator level only (README.md Status)"
'''

LEFT_OUT = {
    'src/*_bench.hip, src/microbench.hip, src/optimized_microbench.hip, src/design_probe.hip, src/wmma_probe.hip, bench/': 'benchmark and probe programs with their recorded results; they need the GPU and the origin\'s fixtures',
    'tests/': 'CPU contract tests (C++ and pytest), GPU probes and their fixture manifests',
    'tools/, experiments/': 'the origin\'s qualification campaigns: remote GPU turn scripts, audits, proofs and frozen experiments tied to its own two machines',
    'docs/, README.md, CHANGELOG.md, AGENTS.md': 'documentation; API.md, DENSE_R1.md, TP_RANK.md, TP_DECODE_PLAN.md, FORMAT.md and NUMERICS.md were read for the module records and ABI-DIFF.md',
    '.github/, .gitattributes, .gitignore, python/.gitignore': 'repository plumbing (python/.gitignore rides along in tool.strix_trellis.reference)',
}

if __name__ == '__main__':
    run(O, mods, ORIGIN_TOML, LEFT_OUT, ['LICENSE', 'NOTICE'])
