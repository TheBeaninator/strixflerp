# SPDX-License-Identifier: MIT
# The top of the build graph of the stage recipe (recipes/qwen36-35b-a3b.flerp-serve-stage.toml, stage P3 of
# docs/PHASE5-FOURNODE-DESIGN.md): gufo's Qwen 3.5 text executor with the mixture-of-experts feed forward of
# executor.flerp.qwen35moe, behind the executor ABI (executor.flerp.qwen35moe_abi), and the programs over it.
#
# From `cmake_minimum_required` to the relay library this file is flerp/qwen35moe_tool/CMakeLists.txt
# (tool.flerp.qwen35moe_generate, stage P2), statement for statement, with its project name; its header lists what
# that file adds to gufo's and what it leaves out. After it, marked "flerp-serve-stage":
#   - flerp_qwen35moe_abi: the ABI adapter and the stage loop (executor.flerp.qwen35moe_abi);
#   - the model's text (runtime.flerp.text_gufo, with strixite's JSON reader runtime.strixite.json), the greedy host
#     sampler (runtime.flerp.sampler_greedy) and the shared decode loop (runtime.flerp.decode_loop);
#   - four programs: the front end (serve.flerp.http), the stage process (tool.flerp.qwen35moe_stage), the
#     conformance driver (executor.flerp.conformance) and stage P2's generator (tool.flerp.qwen35moe_generate) as
#     the gate's reference, each with gufo_qwen/no_image.cpp as that tool links it.
# include/flerp/*.h reach the tree through the recipe's [build] headers, at include/flerp (an include root here).
#   cmake -G Ninja -S <tree>/flerp/qwen35moe_stage -B <out> -DCMAKE_PREFIX_PATH=<rocm> -DCMAKE_HIP_COMPILER=<rocm>/lib/llvm/bin/clang++
cmake_minimum_required(VERSION 3.21)
project(flerp_qwen35moe_stage LANGUAGES C CXX)
get_filename_component(GUFO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
if (NOT EXISTS "${GUFO_ROOT}/src/models/qwen/CMakeLists.txt")
  message(FATAL_ERROR "${GUFO_ROOT} is not a gufo tree (no src/models/qwen/CMakeLists.txt)")
endif()
# The origin's CMake files name the root of the tree by these two.
set(PROJECT_SOURCE_DIR "${GUFO_ROOT}")
set(flerp_qwen35moe_stage_SOURCE_DIR "${GUFO_ROOT}")

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
option(BUILD_TESTING "Build correctness tests" OFF)
option(GUFO_BUILD_TOOLS "Build development benchmarks and tuning tools" OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# flerp-moe: the adapted files first.
set(FLERP_MOE_OVERLAY "${GUFO_ROOT}/flerp/qwen35moe")
if (NOT EXISTS "${FLERP_MOE_OVERLAY}/src/models/qwen/state.hpp")
  message(FATAL_ERROR "${FLERP_MOE_OVERLAY} does not hold executor.flerp.qwen35moe")
endif()
include_directories(${FLERP_MOE_OVERLAY} ${GUFO_ROOT} ${GUFO_ROOT}/include)

# Linux libstdc++ must be parsed before HIP defines its __noinline__ macro (the origin's root, same option).
add_compile_options(
  "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include \"${GUFO_ROOT}/src/core/hip/libstdcxx_noinline.hpp\">")
find_package(ICU REQUIRED COMPONENTS uc i18n)
find_package(Threads REQUIRED)

set(FLERP_OPENSSL_INCLUDE_DIR "" CACHE PATH "Directory holding openssl/evp.h, where no development package is installed")
set(FLERP_OPENSSL_CRYPTO_LIBRARY "" CACHE FILEPATH "libcrypto to link with FLERP_OPENSSL_INCLUDE_DIR")
if (FLERP_OPENSSL_INCLUDE_DIR AND FLERP_OPENSSL_CRYPTO_LIBRARY)
  if (NOT EXISTS "${FLERP_OPENSSL_INCLUDE_DIR}/openssl/evp.h" OR NOT EXISTS "${FLERP_OPENSSL_CRYPTO_LIBRARY}")
    message(FATAL_ERROR "FLERP_OPENSSL_INCLUDE_DIR has no openssl/evp.h or FLERP_OPENSSL_CRYPTO_LIBRARY does not exist")
  endif()
  add_library(OpenSSL::Crypto UNKNOWN IMPORTED)
  set_target_properties(OpenSSL::Crypto PROPERTIES
    IMPORTED_LOCATION "${FLERP_OPENSSL_CRYPTO_LIBRARY}"
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${FLERP_OPENSSL_INCLUDE_DIR}"
    INTERFACE_INCLUDE_DIRECTORIES "${FLERP_OPENSSL_INCLUDE_DIR}")
  message(STATUS "flerp: libcrypto ${FLERP_OPENSSL_CRYPTO_LIBRARY}, headers ${FLERP_OPENSSL_INCLUDE_DIR}")
else()
  find_package(OpenSSL REQUIRED COMPONENTS Crypto)
endif()

option(FLERP_GUFO_IMAGE "Compile src/core/image.cpp into gufo_core (needs libpng, libjpeg, libwebp, libcurl)" OFF)
if (FLERP_GUFO_IMAGE)
  find_package(CURL REQUIRED)
  find_package(PNG REQUIRED)
  find_package(JPEG REQUIRED)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(WEBP REQUIRED IMPORTED_TARGET libwebp)
endif()

# The HIP backend, as in the origin's root.
set(ENGINE_ENABLE_HIP ON)
set(CMAKE_HIP_ARCHITECTURES gfx1151 CACHE STRING "HIP architecture")
if (NOT CMAKE_HIP_ARCHITECTURES STREQUAL "gfx1151")
  message(FATAL_ERROR "Gufo kernels require CMAKE_HIP_ARCHITECTURES=gfx1151")
endif()
enable_language(HIP)
find_package(hip REQUIRED)
find_package(hipblas REQUIRED)
find_package(hipblaslt REQUIRED)
find_package(rocblas REQUIRED)
find_path(HIPCUB_INCLUDE_DIR hipcub/hipcub.hpp REQUIRED)
find_path(ROCPRIM_INCLUDE_DIR rocprim/rocprim.hpp REQUIRED)
find_path(ROCWMMA_INCLUDE_DIR rocwmma/rocwmma.hpp REQUIRED)
message(STATUS "flerp: gufo tree ${GUFO_ROOT}, HIP target ${CMAKE_HIP_ARCHITECTURES}")

# Compiler warnings, where the origin's root turns them on (after the model directories this file leaves out).
option(GUFO_ENABLE_WARNINGS "Enable compiler warnings" ON)
if (GUFO_ENABLE_WARNINGS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_subdirectory(${GUFO_ROOT}/src/cache gufo/src/cache)

add_library(gufo_hip_host STATIC ${GUFO_ROOT}/src/core/hip/host_allocations.cpp)
target_link_libraries(gufo_hip_host PUBLIC hip::host gufo_platform)
target_include_directories(gufo_hip_host PUBLIC ${GUFO_ROOT})
add_library(gufo_cache_hip STATIC
  ${GUFO_ROOT}/src/core/hip/committed_backing.cpp
  ${GUFO_ROOT}/src/core/hip/component_cache_resources.cpp
  ${GUFO_ROOT}/src/core/hip/staging.cpp
  ${GUFO_ROOT}/src/core/hip/transfer_pool.cpp)
target_link_libraries(gufo_cache_hip PUBLIC gufo_cache gufo_hip_host hip::host)

add_subdirectory(${GUFO_ROOT}/src/core/platform gufo/src/core/platform)

add_library(gufo_core STATIC
  ${GUFO_ROOT}/src/core/crypto/sha256.cpp
  ${GUFO_ROOT}/src/core/gguf_reader.cpp
  ${GUFO_ROOT}/src/core/gguf_identity.cpp
  ${GUFO_ROOT}/src/core/json_constraint.cpp
  ${GUFO_ROOT}/src/core/json_schema_lexeme.cpp
  ${GUFO_ROOT}/src/core/json_schema_regex.cpp
  ${GUFO_ROOT}/src/core/sampling.cpp
  ${GUFO_ROOT}/src/core/hip/hipblaslt_plan_database.cpp
  ${GUFO_ROOT}/src/core/quant/ggml_dequant.cpp
  ${GUFO_ROOT}/src/core/quant/ggml_gemm.cpp
)
target_include_directories(gufo_core PUBLIC ${GUFO_ROOT})
target_link_libraries(gufo_core PUBLIC gufo_platform ICU::uc ICU::i18n
  Threads::Threads OpenSSL::Crypto)
if (FLERP_GUFO_IMAGE)
  target_sources(gufo_core PRIVATE ${GUFO_ROOT}/src/core/image.cpp)
  target_link_libraries(gufo_core PUBLIC PNG::PNG JPEG::JPEG PkgConfig::WEBP CURL::libcurl)
endif()

# Family libraries come before the model that needs them (the origin's order).
add_subdirectory(${GUFO_ROOT}/src/families/qwen gufo/src/families/qwen)
add_subdirectory(${GUFO_ROOT}/src/models/qwen gufo/src/models/qwen)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  target_link_libraries(gufo_core PUBLIC OpenMP::OpenMP_CXX)
endif()

target_sources(gufo_core PRIVATE
  ${GUFO_ROOT}/src/core/hip/snapshot_copy.hip
  ${GUFO_ROOT}/src/core/hip/weight_upload.cpp
  ${GUFO_ROOT}/src/core/speculative/speculative_verifier.cpp
)
target_compile_definitions(gufo_core PUBLIC ENGINE_ENABLE_HIP=1)
target_include_directories(gufo_core SYSTEM PRIVATE
  ${HIPCUB_INCLUDE_DIR}
  ${ROCPRIM_INCLUDE_DIR})
target_link_libraries(gufo_core PUBLIC gufo_hip_host hip::host roc::hipblas
  roc::hipblaslt roc::rocblas)

add_subdirectory(${GUFO_ROOT}/src/ext gufo/src/ext)

# flerp-moe: the adapted sources replace the origin's in gufo_qwen27b. SOURCES holds what add_library was given
# (relative to the target's directory) and what target_sources was given (absolute).
set(FLERP_MOE_ADAPTED
  weights.cpp
  forward.cpp
  hip/model_loader.cpp
  hip/decode.cpp
  hip/decode_step.cpp
  hip/batched_decode.cpp
  hip/prefill_chunk.cpp
  hip/arena.cpp)
get_target_property(_moe_sources gufo_qwen27b SOURCES)
foreach (_rel IN LISTS FLERP_MOE_ADAPTED)
  set(_origin "${GUFO_ROOT}/src/models/qwen/${_rel}")
  set(_adapted "${FLERP_MOE_OVERLAY}/src/models/qwen/${_rel}")
  if (NOT EXISTS "${_adapted}")
    message(FATAL_ERROR "flerp-moe: no adapted file ${_adapted}")
  endif()
  list(LENGTH _moe_sources _before)
  list(REMOVE_ITEM _moe_sources "${_rel}" "${_origin}")
  list(LENGTH _moe_sources _after)
  math(EXPR _removed "${_before} - ${_after}")
  if (NOT _removed EQUAL 1)
    message(FATAL_ERROR "flerp-moe: gufo_qwen27b names ${_rel} ${_removed} times, expected once")
  endif()
  list(APPEND _moe_sources "${_adapted}")
endforeach()
list(APPEND _moe_sources "${FLERP_MOE_OVERLAY}/moe_ffn.cpp" "${FLERP_MOE_OVERLAY}/stage.cpp"
  "${FLERP_MOE_OVERLAY}/mtp_head.cpp" "${FLERP_MOE_OVERLAY}/draft_round.cpp")
set_property(TARGET gufo_qwen27b PROPERTY SOURCES ${_moe_sources})

# flerp-moe: the kernels. kernels.hip.cpp is gufo's Flash-Next unit with the options its own CMake file gives it
# (src/models/qwen38_flash_next/CMakeLists.txt: language HIP, QFN_ROCM_BUILD=1, -O3 -ffast-math
# -fno-finite-math-only); moe_kquant.hip is compiled like gufo's Qwen kernels (no fast-math).
set(FLERP_QFN_UNIT "${GUFO_ROOT}/src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp")
# w8a8_wave64.hip is the other source of kernel.gufo.qfn_kernels: the unit above calls it (W8A8GemmWave64).
set(FLERP_QFN_WAVE64 "${GUFO_ROOT}/src/models/qwen38_flash_next/kernels/rocm/w8a8_wave64.hip")
add_library(flerp_qfn_kernels STATIC ${FLERP_QFN_UNIT} ${FLERP_QFN_WAVE64})
set_source_files_properties(${FLERP_QFN_UNIT} PROPERTIES LANGUAGE HIP)
set_source_files_properties(${FLERP_QFN_WAVE64} PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64")
target_compile_definitions(flerp_qfn_kernels PRIVATE QFN_ROCM_BUILD=1 ENGINE_ENABLE_HIP=1)
target_compile_options(flerp_qfn_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
target_include_directories(flerp_qfn_kernels SYSTEM PUBLIC
  ${HIPCUB_INCLUDE_DIR}
  ${ROCPRIM_INCLUDE_DIR}
  ${ROCWMMA_INCLUDE_DIR})
target_link_libraries(flerp_qfn_kernels PUBLIC gufo_core hip::host)

add_library(flerp_moe_kernels STATIC ${GUFO_ROOT}/flerp/moe_kquant/moe_kquant.hip
  ${GUFO_ROOT}/flerp/moe_kgrouped/moe_kgrouped.hip
  ${GUFO_ROOT}/flerp/attn_wmma_gqa/attn_wmma_gqa.hip
  ${GUFO_ROOT}/flerp/draft_coarse/draft_coarse.hip)
target_include_directories(flerp_moe_kernels SYSTEM PUBLIC
  ${HIPCUB_INCLUDE_DIR}
  ${ROCPRIM_INCLUDE_DIR})
target_link_libraries(flerp_moe_kernels PUBLIC gufo_core gufo_qwen_hip_common flerp_qfn_kernels)
target_link_libraries(gufo_qwen27b PUBLIC flerp_moe_kernels)

# flerp-moe: format.flerp.gguf_placed, the placement plan and the placed loader the adapted model_loader.cpp calls
# (gufo's WeightUpload and SHA-256 are in gufo_core).
add_library(flerp_gguf_placed STATIC
  ${GUFO_ROOT}/flerp/gguf_placed/placement.cpp
  ${GUFO_ROOT}/flerp/gguf_placed/placed_loader.cpp)
target_link_libraries(flerp_gguf_placed PUBLIC gufo_core hip::host)
target_link_libraries(gufo_qwen27b PUBLIC flerp_gguf_placed)

# flerp-stage: runtime.flerp.stage_relay, the framed relay between stages (C++ and POSIX sockets only).
add_library(flerp_stage_relay STATIC ${GUFO_ROOT}/flerp/stage_relay/relay.cpp)
target_include_directories(flerp_stage_relay PUBLIC ${GUFO_ROOT})

# flerp-serve-stage: the ABI adapter over the stages (C++), linked into every program below that opens the executor.
add_library(flerp_qwen35moe_abi STATIC
  ${GUFO_ROOT}/flerp/qwen35moe_abi/stages.cpp
  ${GUFO_ROOT}/flerp/qwen35moe_abi/abi.cpp)
target_link_libraries(flerp_qwen35moe_abi PUBLIC ${GUFO_QWEN_LIBRARIES} gufo_core gufo_logging flerp_stage_relay)

# flerp-serve-stage: the model's text, the host sampler, the decode loop every caller of the ABI shares.
add_library(flerp_strixite_json STATIC ${GUFO_ROOT}/serve/json.cpp)
add_library(flerp_text_gufo STATIC ${GUFO_ROOT}/flerp/text/text_gufo.cpp)
target_link_libraries(flerp_text_gufo PUBLIC ${GUFO_QWEN_LIBRARIES} gufo_core flerp_strixite_json)
add_library(flerp_sampler_greedy STATIC ${GUFO_ROOT}/flerp/sampler/sampler_greedy.c)
add_library(flerp_decode_loop STATIC ${GUFO_ROOT}/flerp/decode_loop.c)
add_library(flerp_reply_none STATIC ${GUFO_ROOT}/flerp/reply/reply_none.c)

set(FLERP_NO_IMAGE "")
if (NOT FLERP_GUFO_IMAGE)
  set(FLERP_NO_IMAGE ${GUFO_ROOT}/flerp/gufo_qwen/no_image.cpp)
endif()

# flerp-serve-stage: the front end.
add_executable(strixflerp-qwen36-35b-a3b.flerp-serve-stage
  ${GUFO_ROOT}/flerp/serve/main.cpp ${GUFO_ROOT}/flerp/serve/request.cpp ${FLERP_NO_IMAGE})
target_include_directories(strixflerp-qwen36-35b-a3b.flerp-serve-stage PRIVATE ${GUFO_ROOT}/third_party/cpp-httplib)
target_link_libraries(strixflerp-qwen36-35b-a3b.flerp-serve-stage PRIVATE
  flerp_qwen35moe_abi flerp_text_gufo flerp_sampler_greedy flerp_reply_none flerp_decode_loop flerp_strixite_json Threads::Threads)

# flerp-serve-stage: the process of a stage 1..N-1.
add_executable(flerp-qwen35moe-stage main.cpp ${FLERP_NO_IMAGE})
target_link_libraries(flerp-qwen35moe-stage PRIVATE flerp_qwen35moe_abi)

# flerp-serve-stage: the conformance driver (C) over the adapter (C++): linked by the C++ driver.
add_executable(flerp-executor-conformance ${GUFO_ROOT}/flerp/executor/conformance.c ${FLERP_NO_IMAGE})
target_link_libraries(flerp-executor-conformance PRIVATE flerp_qwen35moe_abi flerp_decode_loop)
set_target_properties(flerp-executor-conformance PROPERTIES LINKER_LANGUAGE CXX)

# flerp-serve-stage: stage P2's generator (tool.flerp.qwen35moe_generate), built from the same tree: the reference.
add_executable(flerp-qwen35moe-generate
  ${GUFO_ROOT}/flerp/qwen35moe_tool/generate.cpp ${GUFO_ROOT}/flerp/qwen35moe_tool/stage_driver.cpp ${GUFO_ROOT}/flerp/qwen35moe_tool/mtp_probe.cpp ${FLERP_NO_IMAGE})
target_link_libraries(flerp-qwen35moe-generate PRIVATE ${GUFO_QWEN_LIBRARIES} gufo_core gufo_logging flerp_stage_relay)
