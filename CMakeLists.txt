cmake_minimum_required(VERSION 3.16)
project(RBAC_HNSW VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# ── Compiler flags ─────────────────────────────────────────────────────────────
# AVX-512 for the bitwise RBAC mask comparisons (§5 of the paper)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -mavx512f -mavx512bw -ffast-math)
elseif(MSVC)
    add_compile_options(/O2 /arch:AVX512)
endif()

# ── hnswlib (header-only) ──────────────────────────────────────────────────────
# Clone with:  git submodule add https://github.com/nmslib/hnswlib.git third_party/hnswlib
include_directories(third_party/hnswlib)

# ── Main library target ────────────────────────────────────────────────────────
add_library(rbac_hnsw INTERFACE)
target_include_directories(rbac_hnsw INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/hnswlib
)

# ── Benchmark executable ───────────────────────────────────────────────────────
add_executable(run_benchmark
    benchmarks/benchmark_main.cpp
)
target_link_libraries(run_benchmark PRIVATE rbac_hnsw)

# ── Unit tests ────────────────────────────────────────────────────────────────
enable_testing()
add_executable(test_rbac
    tests/test_rbac_hnsw.cpp
)
target_link_libraries(test_rbac PRIVATE rbac_hnsw)
add_test(NAME rbac_unit_tests COMMAND test_rbac)

# ── Install ────────────────────────────────────────────────────────────────────
install(TARGETS rbac_hnsw EXPORT RBAC_HNSWTargets)
install(DIRECTORY include/ DESTINATION include)
