# SPDX-License-Identifier: MIT
# The top of gufo's build graph for one program: the Qwen 3.5 text executor and a token generator over it.
#
# gufo's own root CMakeLists.txt builds every model family into one `gufo` binary and needs their libraries
# (libpng, libjpeg, libwebp, libcurl, sndfile, soxr, FFmpeg). This file stands in for it and nothing else: every
# target below the top is declared by the origin's own CMake files (src/cache, src/core/platform, src/ext,
# src/families/qwen, src/models/qwen), which this file adds unchanged, and the targets the origin's root declares
# (gufo_hip_host, gufo_cache_hip, gufo_core) are declared here as the root declares them at f070a17, statement for
# statement, with the differences listed:
#   - gufo_core has no src/core/image.cpp and no PNG, JPEG, WebP or curl: image input is not served, and the one
#     symbol of it the vision library names (DecodeImage) is no_image.cpp, which throws
#     (FLERP_GUFO_IMAGE=ON compiles image.cpp as the origin does, where those libraries exist);
#   - OpenSSL's libcrypto may be named by FLERP_OPENSSL_INCLUDE_DIR and FLERP_OPENSSL_CRYPTO_LIBRARY on a host with
#     the library but no development package;
#   - no extension discovery, no diagnostics, CLI, serving, DeepSeek, Flash-Next, audio or image targets.
#
# It works on two trees: an assembled one (`flerp assemble`, the files of a recipe's module closure) and a checkout
# of the origin with this directory copied in, which is how a recipe gate builds its reference.
#   cmake -G Ninja -S <tree>/flerp/gufo_qwen -B <out> -DCMAKE_PREFIX_PATH=<rocm> -DCMAKE_HIP_COMPILER=<rocm>/lib/llvm/bin/clang++
cmake_minimum_required(VERSION 3.21)
project(flerp_gufo_qwen LANGUAGES C CXX)
get_filename_component(GUFO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
if (NOT EXISTS "${GUFO_ROOT}/src/models/qwen/CMakeLists.txt")
  message(FATAL_ERROR "${GUFO_ROOT} is not a gufo tree (no src/models/qwen/CMakeLists.txt)")
endif()
# The origin's CMake files name the root of the tree by these two.
set(PROJECT_SOURCE_DIR "${GUFO_ROOT}")
set(flerp_gufo_qwen_SOURCE_DIR "${GUFO_ROOT}")

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
include_directories(${GUFO_ROOT} ${GUFO_ROOT}/include)

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

# The one program. GUFO_QWEN_LIBRARIES is the origin's own list ("Qwen libraries a test or tool links").
add_executable(flerp-gufo-qwen-generate generate.cpp)
if (NOT FLERP_GUFO_IMAGE)
  target_sources(flerp-gufo-qwen-generate PRIVATE no_image.cpp)
endif()
target_link_libraries(flerp-gufo-qwen-generate PRIVATE ${GUFO_QWEN_LIBRARIES} gufo_core gufo_logging)
