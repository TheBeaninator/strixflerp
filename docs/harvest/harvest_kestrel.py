#!/usr/bin/env python3
"""Harvest Kestrel (itechmedic/kestrel, 717c769 = release kestrel-v0.1.6) into catalog/kestrel.
Run: python3 docs/harvest/harvest_kestrel.py

Card ORIGIN-KESTREL-HARVEST 1fac8c61fa; verdict of ORIGIN-KESTREL 230d679bc9 (/data/pod/reports/2026-11/origin-kestrel.md).
Kestrel is a fork of Project Maya (mw00/project-maya): git merge-base with Maya main is v1.1.78 (1016876a). Its HIP
kernel families are Maya's; Kestrel adds RDNA4 defaults, the helper-GPU light-expert prompt path and host plumbing
(exl3.cu +21 / -4, ds4_kernels.cu +57 / -5, ds4_iq.cu +90 / -50, ds4_model.cu +1,119 / -20 over v1.1.78). Scope: the
AMD (HIP) kernel families as one module each, with the headers they include from the origin, their docs and their
tests: EXL3 (decode GEMV, verify windows, routed experts, prompt sets), the DeepSeek-V4.1 device kernels, the HIP prompt
MoE kernels ds4_iq.cu links (MMQ, RDNA4 fused), the AMD GGUF IQ expert kernels (glm_fast.cu), the hipBLASLt tuning
tables and their GEMM callers, the native QSA scorer, the hip_compat layer, and the remaining HIP tests. The engine
hosts (src/ds4/ds4_model.cu with the peer prefill all-reduce, src/core, serve/), CPU paths, Metal, the other model
families and third_party/ are listed as unharvested.

Maya builds every .cu as HIP through include/maya/hip_compat (CMake option MAYA_ENABLE_HIP, cmake/hip_backend.cmake);
no unit compiles on its own without the engine's other headers and ggml from third_party, and nothing is compiled or
run by this harvest, so every module is [gate] status = "harvested".

The reference copy is the pod ark's tree export
/data/ark/software/src/github.com__itechmedic__kestrel#717c76975cd20d37b2e4162072d7745d6e1576ff (it resolves into
/data/ark-inbox/src; its .git holds 25 pack files but no HEAD, so git cannot read it). It was copied to the laptop with
tools/pod-pull.sh's rsync path into ~/Code/strixflerp-upstream-supplement/kestrel. KESTREL_CHECKOUT names another copy.
The pin is checked by KESTREL_VERSION (0.1.6) and the tracked file count (1,841 files outside .git, git ls-files of
717c769); flerp diff-origin --origin kestrel against the ark path proves the bytes."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvestlib
from harvestlib import Origin, M, P, run

CHECKOUT = os.environ.get('KESTREL_CHECKOUT', os.path.join(harvestlib.SUPPLEMENT, 'kestrel'))
COMMIT = '717c76975cd20d37b2e4162072d7745d6e1576ff'
N_FILES = 1841


class TreeOrigin(Origin):
    """The origin's file set is the tree export without its .git, checked against the pinned release."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.roots = [CHECKOUT]
        v = open(os.path.join(CHECKOUT, 'KESTREL_VERSION')).read().strip()
        if v != '0.1.6':
            raise SystemExit(f'{CHECKOUT}: KESTREL_VERSION {v}, the harvest pins 0.1.6 ({COMMIT})')
        if len(self.files()) != N_FILES:
            raise SystemExit(f'{CHECKOUT}: {len(self.files())} files outside .git, {COMMIT} tracks {N_FILES}')

    def files(self):
        if self._files is None:
            out = []
            for d, dirs, fs in os.walk(CHECKOUT):
                if os.path.relpath(d, CHECKOUT) == '.':
                    dirs[:] = [x for x in dirs if x != '.git']
                for f in fs:
                    out.append(os.path.relpath(os.path.join(d, f), CHECKOUT))
            self._files = sorted(out)
        return self._files


O = TreeOrigin('kestrel', 'kestrel', 'https://github.com/itechmedic/kestrel', COMMIT, 'kestrel', 'MIT',
               ['itechmedic', 'Project Maya contributors (mw00)'], 'rocm-7.2.4')
UNIT = ('a translation unit of Maya\'s CMake build with MAYA_ENABLE_HIP (cmake/hip_backend.cmake): the .cu file is compiled as HIP '
        'through include/maya/hip_compat, for gfx1100 / gfx1201 / gfx1151 (validated) with the engine\'s include roots and ggml from '
        'third_party; it needs headers this harvest leaves in the origin, so it is not compiled here')
NOT_TIMED = 'Nothing was compiled, run or timed on the pod.'
REPORT = '/data/pod/reports/2026-11/origin-kestrel.md'

WHY = {
 'kernel.kestrel.exl3': 'the only other EXL3 HIP implementation for DeepSeek-V4.1 beside kernel.strixfold.exl3, validated by its origin on gfx1151 (docs/EXL3_HIP.md); ' + REPORT + ' compares the two at the Wrench TP2 shapes and draws three ideas for few-row decode from it (row capacity by template, rotation inside the GEMV, an occupancy-driven split rule)',
 'kernel.kestrel.ds4': 'the DeepSeek-V4.1 device kernels of a second engine for the same model (attention with gfx11 / gfx12 WMMA modes, MoE combine, sampling and logprobs, the GGUF IQ routed experts), and the file Kestrel changed most for AMD (helper hand-off with 8 host-mapped loads in flight, light-expert gather / add)',
 'kernel.kestrel.prefill_moe': 'ds4_iq.cu links these prompt-expert kernels on HIP (Kestrel\'s MMQ link fix is about exactly this), and moe_fused_rdna4.cu is an MIT adaptation of Strata\'s IQ prompt experts for AMD, a second route beside gufo\'s ds4_moe prefill',
 'kernel.kestrel.amd_experts': 'packed sign expansion for the IQ2_XXS / IQ2_S / IQ3_XXS decode experts, measured by its origin on gfx1151 (gate_up 1.124x, IQ2_S down 1.038x, docs/AMD_EXPERT_KERNELS.md) with byte-identical parity over 620 cases; the technique transfers to any GGUF IQ expert path in the catalogue (gufo ds4_moe)',
 'kernel.kestrel.hipblaslt': 'a per-architecture, per-hipBLASLt-version solution table for the prompt\'s dense GEMMs with gfx1151 tables and the tool that makes them; INTERFACES.md section 3 names the bf16-weight hipBLASLt route as the one cross-engine GEMM gate worth having',
 'kernel.kestrel.qsa_score': 'a scalar HIP QSA indexer scorer for the Qwen3.x-FN layout (24 heads, indexer 4 x 128, block 4, top-k 2048), a fourth provider beside kernel.strixite.qsa, the libstrix and hipfire scorers, with its own device parity test',
 'kernel.kestrel.hip_compat': 'every Kestrel / Maya .cu file reaches HIP through these headers and this CMake file, so no other kestrel module can be read or built without them; they also record which RDNA / CDNA targets the origin validated',
 'tool.kestrel.hip_tests': 'the origin\'s remaining HIP device tests and benches (prefill parity, expert staging, handoffs, GLM benches); they define what correct means for the harvested kernels in the origin\'s own terms, which a timing or parity card needs first',
}

mods, claimed = [], set()


def take(names):
    out = []
    for n in names:
        hits = [f for f in O.expand([n]) if f not in claimed]
        if not hits:
            raise SystemExit(f'nothing left for {n}')
        out += hits
    claimed.update(out)
    return out


def mod(id, summary, names, provides, notes, requires=(), lang=None, reason=UNIT, toolchain=None, sources=None):
    files = take(names)
    if lang is None:   # CUDA-dialect .cu sources that Maya compiles as HIP (flerp's hip-c++ takes .hip / .cpp / .cc only)
        lang = 'cuda' if any(f.endswith('.cu') for f in files if not f.startswith(('tests/', 'tools/'))) else 'hip-c++'
    if sources is None:
        sources = [f for f in files if f.endswith(('.cu', '.cpp')) and not f.startswith(('tests/', 'tools/'))]
    incs = sorted({'src/' + os.path.dirname(f) for f in files if f.endswith(('.h', '.hpp', '.cuh', '.cu', '.cpp', '.inc'))})
    if any(f.startswith('include/') for f in files):
        incs = sorted(set(incs) | {'src/include'})
    mods.append(M(id, summary, files, provides=provides, requires=requires, lang=lang, sources=sources, includes=incs,
                  compile='n/a', reason=reason, notes=notes + ' ' + NOT_TIMED, why=WHY[id], toolchain=toolchain))


COMPAT = 'module:kernel.kestrel.hip_compat'

mod('kernel.kestrel.hip_compat', 'Maya\'s CUDA-to-HIP layer as Kestrel ships it: include/maya/hip_compat (cuda_runtime.h, cuda_fp16.h, cuda_bf16.h, cublas_v2.h, cuda_profiler_api.h, cuda_runtime_api.h mapped onto HIP / hipBLAS, intrinsics.hpp with the warp shuffles, dot4 and the wave64 logical-32-lane mapping), two platform headers the kernel modules include (env_alias.hpp: every MAYA_* setting also read as KESTREL_*; commit.hpp), cmake/hip_backend.cmake (MAYA_ENABLE_HIP: validated gfx1100 / gfx1201 / gfx1151, community gfx1101 / gfx1200, wave64 CDNA through the compat path), the intrinsics device test, the compat checkers and the AMD setup docs',
    ['include/maya/hip_compat/', 'include/maya/platform/env_alias.hpp', 'include/maya/platform/commit.hpp', 'cmake/hip_backend.cmake', 'tests/hip/intrinsics.cpp', 'tools/check_hip_compat.py', 'tools/test_hip_compat.py',
     'tools/test_hip_archs.py', 'tools/test_maya_hip.py', 'docs/AMD_MAYA.md', 'docs/AMD_RDNA4.md', 'docs/AMD_UPSTREAM.md', 'docs/KESTREL_LINUX_AMD.md'],
    [P('kernel-abi.kestrel-hip-compat/-/-', 'the CUDA runtime, fp16 / bf16 and cuBLAS names over HIP (include/maya/hip_compat/*.h); maya::hip intrinsics (intrinsics.hpp)', 'n/a', 'n/a')],
    'Interface tag: none. The origin\'s Strata lineage shows here: the CMake file says Strata\'s CUDA-shaped kernels target wave32 RDNA2 / 3 / 4 with 64 KiB LDS and a signed dot4. Kestrel\'s own setup guide for an R9700 plus 8060S helper is docs/KESTREL_LINUX_AMD.md.',
    sources=['tests/hip/intrinsics.cpp'])

mod('kernel.kestrel.exl3', 'Maya\'s EXL3 device kernels as Kestrel 0.1.6 ships them (src/kernels/cuda/exl3.cu, 3,836 lines; trellis codebooks and window arithmetic after exllamav3): the decode GEMV (block_gemv: 256 threads own 128 outputs over a k slice of at most 4096, up to 8 activations a pass, v_dot2_f32_f16 on RDNA3 / 4, the input rotated inside each block), verify-window and token-path routed experts (moe_gate_up / moe_down, win_gu / win_down with 2 / 4 / 8-row instances), the prompt set kernels (scalar hip_set and hip_scalar_set with NR rows, gfx11 v_wmma_f32_16x16x16_f16, gfx12 WMMA, CDNA MFMA) and the GEMM, with the NVIDIA-only legacy decode file, the format header, the policy and device tests and the EXL3 docs',
    ['src/kernels/cuda/exl3.cu', 'src/kernels/cuda/exl3_hip_scalar.hpp', 'src/kernels/cuda/exl3_legacy.cu', 'include/maya/kernels/exl3.hpp',
     'include/maya/kernels/dp4a.hpp', 'tests/exl3_test.cu', 'tests/exl3_hip_scalar_policy_test.cpp', 'tests/hip/ds4_exl3_helper.cpp',
     'docs/EXL3_HIP.md', 'docs/EXL3_HIP_SCALAR_ROWS.md', 'docs/EXL3_RDNA4.md', 'docs/EXL3-HELPER.md'],
    [P('linear.decode/exl3/f32-f32.kestrel', 'maya::kernels::exl3::mv(const MvJob*, int, cudaStream_t) (include/maya/kernels/exl3.hpp; src/kernels/cuda/exl3.cu)',
       'per block 8 waves, a lane holds an MMA-A fragment (2 outputs x 4 inputs of a 16 x 16 slice), one v_dot2_f32_f16 per weight pair per activation, fp32 sums reduced over 4 lanes by DPP xor, split-K added in split order by the last block; an activation\'s sums do not depend on the other columns (row-invariant within this kernel, not bit-equal to kernel.strixfold.exl3)',
       'decode and verify windows, 1-8 activations a pass (DS4_EXL3_MV1: a 1-row kernel by default on HIP); f32 or q8_1 input, rotated in-kernel'),
     P('moe.experts-decode/exl3/f32.kestrel', 'maya::kernels::exl3::moe_gate_up, moe_down, win_experts (exl3.hpp)',
       'the token path\'s arithmetic per entry: a window\'s token gets moe_gate_up + moe_down\'s bits for it alone', 'decode and verify windows; routed expert widths up to 8192 (DeepSeek-V4.1 n_embd 5120)'),
     P('linear.prefill/exl3/f16-f32.kestrel', 'maya::kernels::exl3::gemm_rows, set_gate_up, set_down (exl3.hpp)',
       'scalar FP32 / DOT2 tiles on HIP with NR rows (MAYA_EXL3_HIP_SCALAR_ROWS, gfx12 only), v_wmma_f32_16x16x16_f16 on gfx11 (EXL3_WMMA_ROWS 16, 32 on gfx12 since Kestrel), fp32 accumulate', 'prompt chunks')],
    'Interface tags: linear.decode/exl3/f32-f32, moe.experts-decode/exl3/f32, linear.prefill/exl3/f16-f32 (the same <family>.<op> as kernel.strixfold.exl3; not a drop-in: different input contract and sum order, ' + REPORT + '). Kestrel\'s delta over Maya v1.1.78 in exl3.cu: gfx12 defaults to 32 WMMA rows (an R9700: -12 % expert time on a 20K prompt) and a profiling hook in win_experts. exl3_legacy.cu is v1.1.8\'s NVIDIA-only decode (before sm_80) and is not built for HIP.',
    requires=[COMPAT])

mod('kernel.kestrel.ds4', 'Maya\'s DeepSeek-V4.1 device kernels as Kestrel 0.1.6 ships them: src/ds4/ds4_kernels.cu (5,404 lines: sparse attention with gfx11 / gfx12 WMMA modes (DS4_ATTN_SHARE), rotary, MoE combine, the helper hand-off helper_take, logprob rows, F16 / Q6_K GEMVs, the light-expert gather and add Kestrel added) and src/ds4/ds4_iq.cu (3,052 lines: a GGUF\'s low-bit routed experts on the device, ggml\'s block layouts and llama.cpp\'s sign and grid tricks), with the headers they include',
    ['src/ds4/ds4_kernels.cu', 'src/ds4/ds4_kernels.cuh', 'src/ds4/ds4_peer_cuda_compat.cuh', 'src/ds4/ds4_pdl.cuh', 'src/ds4/ds4_numerics.cuh',
     'src/ds4/ds4_q6k.cuh', 'src/ds4/ds4_pool.hpp', 'src/ds4/ds4_logprobs.hpp', 'src/ds4/ds4_iq.cu', 'src/ds4/ds4_iq.cuh', 'src/ds4/ds4_iq1_codes.hpp',
     'src/ds4/ds4_par_exp.h', 'src/ds4/ds4_os.hpp', 'include/maya/kernels/sampler.hpp', 'include/maya/kernels/lane_xor.hpp'],
    [P('moe.experts-decode/gguf-iq2+q2_k/f32.kestrel', 'maya::ds4 IQ expert launches (src/ds4/ds4_iq.cuh)',
       'ggml block layouts, llama.cpp vecdotq arithmetic, fp32 accumulate', 'decode and prompt sets of DeepSeek-V4.1 (n_embd 5120) and, since Kestrel, MiniMax-M3.1 routed experts'),
     P('kernel-abi.kestrel-ds4/-/-', 'maya::ds4 device kernels (src/ds4/ds4_kernels.cuh)', 'no fast math: rotary angles, softmaxes and roundings follow the reference\'s F32 arithmetic (file header)', 'decode windows (a window\'s token is blockIdx.y) and prompt chunks')],
    'Interface tags: moe.experts-decode/gguf-iq2+q2_k/f32 (the class of kernel.gufo.ds4_moe), the rest none (engine-internal launches). Kestrel\'s delta over Maya v1.1.78: ds4_kernels.cu +57 / -5 (helper_take keeps 8 host-mapped loads in flight, light_gather / light_add for DS4_PROMPT_LIGHT, gfx12 attention share mode 1), ds4_iq.cu +90 / -50 (MMQ link for maya-mm3 on HIP, MiniMax-M3.1 expert geometry). The host that drives them (ds4_model.cu, with the peer prefill all-reduce of docs/HIP_PEER_PREFILL.md) is not harvested.',
    requires=[COMPAT])

mod('kernel.kestrel.prefill_moe', 'the HIP prompt-expert kernels ds4_iq.cu links: src/prefill/moe_fused_rdna4.cu (IQ prompt experts with int8 activation codes and FP32 scales per 32 values, adapted from Niko1221/Strata fb58e0d and PR #1570, MIT), the llama.cpp MMQ view moe_mmq.cu and glm-dsa\'s moe_mmq_strata.cu (compiled from the pinned llama.cpp checkout), ggml_cuda_host.cu (the host symbols of ggml-cuda they reference), their headers and the MMQ parity test',
    ['src/prefill/moe_fused_rdna4.cu', 'src/prefill/moe_mmq.cu', 'src/prefill/moe_mmq_strata.cu', 'src/prefill/ggml_cuda_host.cu',
     'include/maya/prefill/moe_fused_rdna4.hpp', 'include/maya/prefill/moe_mmq.hpp', 'include/maya/prefill/moe_mmq_strata.hpp', 'include/maya/prefill/failure.hpp',
     'tests/hip/prefill_mmq_parity.cpp'],
    [P('moe.experts-prefill/gguf-iq2+q2_k/f32.kestrel', 'maya::prefill moe_mmq / moe_fused_rdna4 entry points (include/maya/prefill/moe_mmq.hpp, moe_fused_rdna4.hpp)',
       'int8 activation codes with FP32 scales per 32 values (MMQ / Q8_1 style); IQ blocks decoded with ggml\'s codebooks', 'prompt chunks; MMQ from a minimum of rows per expert (mmq_min_rows)')],
    'Interface tag: moe.experts-prefill/gguf-iq2+q2_k/f32 (the class of kernel.gufo.ds4_moe\'s prefill). moe_mmq.cu and moe_mmq_strata.cu include ggml-cuda\'s mmq.cuh / common.cuh from third_party/llama.cpp, which is not harvested.',
    requires=[COMPAT])

mod('kernel.kestrel.amd_experts', 'Maya\'s GLM fast-path kernels with the AMD expert kernels (src/kernels/cuda/glm_fast.cu, 7,911 lines; MAYA_HIP_EXPERTS_LEGACY): IQ2_XXS gate_up<16>, IQ2_S down<22> and IQ3_XXS down<18> decode experts with packed sign expansion (a multiply spreads four sign bits into four bytes, masks and shifts give the XOR masks and two\'s-complement corrections), 2 KiB / 1 KiB codebooks in LDS, chosen per architecture, with the headers glm_fast.cu includes, the expert kernel bench and docs/AMD_EXPERT_KERNELS.md',
    ['src/kernels/cuda/glm_fast.cu', 'src/kernels/cuda/dsa_topk.cuh', 'src/kernels/cuda/glmd_mv_ext.cuh', 'src/kernels/cuda/glmd_mv_dots.cuh',
     'include/maya/kernels/glm_fast.hpp', 'include/maya/kernels/glm_expert_bench.hpp', 'include/maya/kernels/glm_batch.hpp',
     'include/maya/kernels/glmd_random_route.hpp', 'include/maya/core/glm_req_knobs.hpp', 'include/maya/core/glm_env.hpp', 'tests/hip/expert_kernel_bench.cpp', 'docs/AMD_EXPERT_KERNELS.md'],
    [P('moe.experts-decode/gguf-iq2+iq3/f32.kestrel', 'maya::kernels::glm fast-path expert launches (include/maya/kernels/glm_fast.hpp)',
       'llama.cpp vecdotq expressions; the original integer scale divisions, FP32 accumulation order, clamping, q8_1 quantization and combine order kept, so byte-identical q8_1 and bit-identical FP32 outputs against the legacy kernels (docs/AMD_EXPERT_KERNELS.md: 620 comparisons)',
       'GLM decode experts n_embd 4096, n_ff 2048; gfx1100 all three optimized, gfx1151 gate_up<16> and down<22> optimized (down<18> legacy: 0.972x), gfx1201 legacy')],
    'Interface tag: moe.experts-decode/gguf-iq2+iq3/f32, beside kernel.gufo.ds4_moe (IQ2_XXS + Q2_K, DeepSeek shapes). Origin numbers on a Radeon 8060S (gfx1151), docs/AMD_EXPERT_KERNELS.md: gate_up<16> 203.9 -> 181.4 us (1.124x), down<22> 131.4 -> 126.6 us (1.038x), down<18> 126.5 -> 130.2 us (0.972x), full-model decode 17.8 -> 18.0 tok/s. The signed-codebook technique is adapted from Niko1221/Strata fb58e0db moe_fused_iq.cu (MIT). glm_fast.cu also includes ggml-common.h from third_party (not harvested).',
    requires=[COMPAT, 'module:kernel.kestrel.exl3'])

mod('kernel.kestrel.hipblaslt', 'hipBLASLt tuning for the prompt\'s dense GEMMs: src/prefill/hipblaslt_tuning.hpp (MAYA_HIPBLASLT_TUNING_V1 tables: input type, n, k, ldy, row bucket -> solution id, scoped to one architecture and hipBLASLt version), the callers src/prefill/gemm.cu and src/ds4/ds4_lt.cu, the table maker tools/hip/tune_hipblaslt.cpp, the shipped tables for gfx1100 / gfx1151 / gfx1201 / gfx942 (gfx1151: GLM and GLM-5.3 shapes for hipBLASLt 100202, 100401, 100500), and the table and GEMM tests',
    ['src/prefill/hipblaslt_tuning.hpp', 'src/prefill/gemm.cu', 'include/maya/prefill/gemm.hpp', 'src/ds4/ds4_lt.cu', 'src/ds4/ds4_lt.hpp',
     'tools/hip/tune_hipblaslt.cpp', 'tools/hip/*.txt', 'tests/hipblaslt_table_list_test.cpp', 'tests/hip/prefill_hipblaslt_gemm.cpp'],
    [P('linear.prefill/bf16/bf16-f32.kestrel-hipblaslt', 'maya::prefill GEMM (include/maya/prefill/gemm.hpp) and ds4_lt (src/ds4/ds4_lt.hpp) over hipBLASLt with TuningTable::load(path, arch, version)',
       'hipBLASLt\'s own solution; the table only picks the solution id, so the result is the library plan\'s order', 'prompt chunks; bf16 / f16 weights; a row bucket a table row')],
    'Interface tag: linear.prefill/bf16/bf16-f32 (the class of gufo\'s hipBLASLt route, INTERFACES.md section 3). The tables are GLM shapes; tune_hipblaslt.cpp would make DeepSeek-V4.1 or Qwen ones. gemm.cu includes include/maya/kernels/dequant_bf16.hpp, which is not harvested.',
    requires=[COMPAT, 'module:kernel.kestrel.prefill_moe'])

mod('kernel.kestrel.qsa_score', 'the optional native QSA scorer with its scalar HIP implementation (src/kernels/cuda/native_qsa_score.cu, 213 lines), its header, the QSA shapes header (qsa_real_shapes: 24 heads, 2 KV heads, head_dim 256, indexer 4 heads x 128, block 4, top-k 2048) and the device parity test',
    ['src/kernels/cuda/native_qsa_score.cu', 'include/maya/kernels/native_qsa_score.hpp', 'include/maya/kernels/qsa.hpp', 'tests/hip/native_qsa_score.cpp'],
    [P('attn.qsa-indexer/-/f32.kestrel', 'maya::kernels native QSA score (include/maya/kernels/native_qsa_score.hpp)', 'fp32 pooled keys and queries with a per-block bias; scalar HIP, checked against a host reference by tests/hip/native_qsa_score.cpp', 'decode and prompt scoring over 4-cell indexer blocks')],
    'Interface tag: attn.qsa-indexer/-/f32, the class of kernel.strixite.qsa\'s f32 variant (Qwen3.x-FN layout).',
    requires=[COMPAT])

mod('tool.kestrel.hip_tests', 'the origin\'s other HIP device tests and benches: tests/hip (expert cache staging, GLM attention / MV benches, GLM expert helper and handoffs, MiniMax attention bench, PLE IQ4, prefill fused bench and parity, native batch prefill) and tools/hip (the prefill bench driver and the Windows HIP build script)',
    ['tests/hip/', 'tools/hip/'],
    [P('verify.kernel-reference/-/-.kestrel', 'the CMake test targets under MAYA_BUILD_TESTS with MAYA_ENABLE_HIP; tools/hip/bench_prefill.py', 'each test states its own reference and tolerance', 'offline, on an AMD GPU with the origin built; not run by this harvest')],
    'Interface tag: none.', requires=[COMPAT], sources=[], reason=UNIT + '; tests and benches, not run')

ORIGIN_TOML = '''
# Kestrel: a fork of Project Maya (mw00/project-maya, a C++ / CUDA engine with a HIP build) that runs DeepSeek-V4.1-Flash
# and other MoE models with EXL3 or GGUF weights on AMD: a Radeon AI PRO R9700 (gfx1201) with an 8060S helper GPU in the
# author's PC (250K prompt read at 1,217 tok/s, 51 tok/s decode, README). The kernel families are Maya v1.1.78's; Kestrel adds
# RDNA4 defaults, the helper-GPU light-expert prompt path and host plumbing (card ORIGIN-KESTREL-HARVEST 1fac8c61fa,
# report /data/pod/reports/2026-11/origin-kestrel.md).
name      = "kestrel"
repo      = "https://github.com/itechmedic/kestrel"
commit    = "717c76975cd20d37b2e4162072d7745d6e1576ff"
branch    = "kestrel"
version   = "kestrel-v0.1.6 (KESTREL_VERSION 0.1.6), on Project Maya v1.1.78 (merge-base 1016876a387e332fc5bf8bb82030fca7dc2821b2)"
reference_checkout = "control-node:/data/ark/software/src/github.com__itechmedic__kestrel#717c76975cd20d37b2e4162072d7745d6e1576ff (ark tree export, 1,841 files; resolves into /data/ark-inbox/src); laptop: ~/Code/strixflerp-upstream-supplement/kestrel (copied from it)"

[licence]
spdx      = "MIT"
files     = ["LICENSE"]                                   # copied verbatim next to this file
third_party = "LICENSE: MIT, Copyright (c) 2026 Peasant Smith and the Project Maya contributors. Code inside follows MIT sources and says so in its headers: exllamav3 (turboderp; the EXL3 codebooks and trellis arithmetic in exl3.cu), ggml / llama.cpp (block layouts, vecdotq, MMQ; third_party/ggml/LICENSE, not harvested), Niko1221/Strata (moe_fused_rdna4.cu, the signed-codebook technique in glm_fast.cu). LICENSE.deepseek-v41, LICENSE.hy4-flash, LICENSE.minimax-m31 and LICENSE.qwen-community-1.0 are model licences and are not harvested. Weights are not part of the tree."

[toolchain]
host      = "C++20, CMake with MAYA_ENABLE_HIP=ON (setup.sh / setup.py pick the ROCm install)"
device    = "rocm-7.2.4"                                   # the origin documents ROCm 7.2.1 (docs/AMD_EXPERT_KERNELS.md) and 7.14.1 (docs/EXL3_HIP.md); the nearest catalogue toolchain is named, nothing was built
build     = "cmake -DMAYA_ENABLE_HIP=ON -DMAYA_ENABLE_CUDA=OFF -DCMAKE_HIP_ARCHITECTURES='gfx1100;gfx1201;gfx1151' (docs/AMD_EXPERT_KERNELS.md); every .cu compiled as HIP through include/maya/hip_compat; ggml / llama.cpp from third_party. There is no per-kernel compile line, so this file has no [compile] recipe"

[kernels]
loading   = "linked into the engine binaries (maya-ds4, maya-mm3, ...); kernels are launched by host functions in the same .cu files"
selection = "per device at run time by gcnArchName (gfx11 / gfx12 / gfx1151 branches in exl3.cu, ds4_kernels.cu, glm_fast.cu) and by environment settings, every MAYA_* also as KESTREL_* (docs/SETTINGS.md)"
models    = "DeepSeek-V4.1-Flash (EXL3 3.5 bpw and GGUF), GLM-5.x, MiniMax-M3.1, Inkling-2-Mini, Hy4-Flash, Qwen (README, docs/MODELS.md)"
'''

LEFT_OUT = {
    'src/ds4/ (except the kernel files above), src/core/, src/hy4/, src/mm3/, src/ik2/, src/spec/, src/plan/, src/program/, src/platform/, src/ngram/, src/artifact/': 'the engine hosts and model families: placement, SSD tiering, helper GPU (DS4_PROMPT_LIGHT, DS4_FILE_*), the peer prefill all-reduce in ds4_model.cu (docs/HIP_PEER_PREFILL.md: two GPUs in one host, not our TP pairs), speculation and serving',
    'src/kernels/cuda/ (other files), src/prefill/ (other files), src/metal/': 'CUDA, CPU and Metal kernels of the other model families; read where they bear on the harvested ones',
    'third_party/': 'vendored ggml / llama.cpp and other third-party code under their own licences',
    'serve/, chat.py, maya.py, kestrel.sh, setup.*, START-*.bat, docker/, examples/, bench/, assets/, data/, ref/': 'front ends, setup, packaging, benchmarks and reference data',
    'docs/ (other files), README.md, CHANGELOG.md': 'documentation; the AMD and EXL3 docs ride in their modules',
    'LICENSE.*': 'model licences, not code licences',
}

if __name__ == '__main__':
    run(O, mods, ORIGIN_TOML, LEFT_OUT, ['LICENSE'])
