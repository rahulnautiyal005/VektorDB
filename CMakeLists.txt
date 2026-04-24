cmake_minimum_required(VERSION 3.20)
project(VektorDB
    VERSION 0.1.0
    DESCRIPTION "Hardware-Accelerated Vector Search Engine"
    LANGUAGES CXX
)

# ============================================================================
# C++20 Standard
# ============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Build Options
# ============================================================================
option(VEKTORDB_BUILD_TESTS      "Build unit tests"        ON)
option(VEKTORDB_BUILD_BENCHMARKS "Build benchmarks"        ON)
option(VEKTORDB_ENABLE_AVX2      "Enable AVX2 intrinsics"  ON)

# ============================================================================
# Compiler Flags
# ============================================================================
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O2)
    if(VEKTORDB_ENABLE_AVX2)
        add_compile_options(-mavx2 -mfma)
        add_compile_definitions(VEKTORDB_AVX2_ENABLED)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/W4 /O2)
    if(VEKTORDB_ENABLE_AVX2)
        add_compile_options(/arch:AVX2)
        add_compile_definitions(VEKTORDB_AVX2_ENABLED)
    endif()
endif()

# ============================================================================
# Core Library
# ============================================================================
add_library(vektordb_core STATIC
    src/math/scalar_ops.cpp
    src/math/simd_ops.cpp
    src/math/distance.cpp
)

target_include_directories(vektordb_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(vektordb_core PUBLIC cxx_std_20)

# ============================================================================
# Dependencies via FetchContent
# ============================================================================
include(FetchContent)

if(VEKTORDB_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.15.2
    )
    # Prevent overriding parent project's compiler/linker settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_subdirectory(tests)
endif()

if(VEKTORDB_BUILD_BENCHMARKS)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.9.1
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)

    add_subdirectory(benchmarks)
endif()

# ============================================================================
# Install
# ============================================================================
install(TARGETS vektordb_core ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
