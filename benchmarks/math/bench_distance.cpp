/// @file bench_distance.cpp
/// @brief Performance benchmarks comparing scalar vs AVX2 distance functions.
///
/// Benchmarks are run across real-world embedding dimensions:
///   128  — Lightweight embedding models
///   256  — MiniLM, small transformers
///   768  — BERT, sentence-transformers
///   1536 — OpenAI text-embedding-3-small
///
/// Usage:
///   ./vektordb_bench --benchmark_format=console
///   ./vektordb_bench --benchmark_out=results.json --benchmark_out_format=json

#include <benchmark/benchmark.h>

#include "vektordb/math/scalar_ops.h"
#include "vektordb/math/simd_ops.h"
#include "vektordb/core/platform.h"
#include "vektordb/core/types.h"

#include <vector>
#include <random>

using namespace vektordb;
using namespace vektordb::math;

// ============================================================================
// Benchmark Data Generator
// ============================================================================

/// Pre-generates random vectors for benchmarking (avoids RNG in hot loop).
class BenchData {
public:
    BenchData(dim_t dim, uint32_t seed = 42) : dim_(dim) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        a_.resize(dim);
        b_.resize(dim);
        for (dim_t i = 0; i < dim; ++i) {
            a_[i] = dist(rng);
            b_[i] = dist(rng);
        }
    }

    const float* a() const { return a_.data(); }
    const float* b() const { return b_.data(); }
    dim_t dim() const { return dim_; }

private:
    dim_t dim_;
    std::vector<float> a_, b_;
};

// ============================================================================
// Scalar Benchmarks
// ============================================================================

static void BM_ScalarL2(benchmark::State& state) {
    BenchData data(static_cast<dim_t>(state.range(0)));
    for (auto _ : state) {
        float result = scalar::l2_distance(data.a(), data.b(), data.dim());
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
    state.SetBytesProcessed(state.iterations() * state.range(0) * 2 * sizeof(float));
}

static void BM_ScalarCosine(benchmark::State& state) {
    BenchData data(static_cast<dim_t>(state.range(0)));
    for (auto _ : state) {
        float result = scalar::cosine_similarity(data.a(), data.b(), data.dim());
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
    state.SetBytesProcessed(state.iterations() * state.range(0) * 2 * sizeof(float));
}

static void BM_ScalarDot(benchmark::State& state) {
    BenchData data(static_cast<dim_t>(state.range(0)));
    for (auto _ : state) {
        float result = scalar::dot_product(data.a(), data.b(), data.dim());
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
    state.SetBytesProcessed(state.iterations() * state.range(0) * 2 * sizeof(float));
}

// ============================================================================
// AVX2 Benchmarks
// ============================================================================

#ifdef VEKTORDB_AVX2_ENABLED

static void BM_AVX2_L2(benchmark::State& state) {
    if (!platform::supports_avx2()) {
        state.SkipWithError("AVX2 not supported");
        return;
    }

    BenchData data(static_cast<dim_t>(state.range(0)));
    for (auto _ : state) {
        float result = avx2::l2_distance(data.a(), data.b(), data.dim());
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
    state.SetBytesProcessed(state.iterations() * state.range(0) * 2 * sizeof(float));
}

static void BM_AVX2_Cosine(benchmark::State& state) {
    if (!platform::supports_avx2()) {
        state.SkipWithError("AVX2 not supported");
        return;
    }

    BenchData data(static_cast<dim_t>(state.range(0)));
    for (auto _ : state) {
        float result = avx2::cosine_similarity(data.a(), data.b(), data.dim());
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
    state.SetBytesProcessed(state.iterations() * state.range(0) * 2 * sizeof(float));
}

static void BM_AVX2_Dot(benchmark::State& state) {
    if (!platform::supports_avx2()) {
        state.SkipWithError("AVX2 not supported");
        return;
    }

    BenchData data(static_cast<dim_t>(state.range(0)));
    for (auto _ : state) {
        float result = avx2::dot_product(data.a(), data.b(), data.dim());
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
    state.SetBytesProcessed(state.iterations() * state.range(0) * 2 * sizeof(float));
}

#endif // VEKTORDB_AVX2_ENABLED

// ============================================================================
// Register Benchmarks
// ============================================================================

// Dimensions to benchmark: real-world embedding model sizes
#define BENCH_DIMS ->Arg(128)->Arg(256)->Arg(768)->Arg(1536)->Arg(4096)

// Scalar
BENCHMARK(BM_ScalarL2)     BENCH_DIMS ->Unit(benchmark::kNanosecond);
BENCHMARK(BM_ScalarCosine) BENCH_DIMS ->Unit(benchmark::kNanosecond);
BENCHMARK(BM_ScalarDot)    BENCH_DIMS ->Unit(benchmark::kNanosecond);

// AVX2
#ifdef VEKTORDB_AVX2_ENABLED
BENCHMARK(BM_AVX2_L2)     BENCH_DIMS ->Unit(benchmark::kNanosecond);
BENCHMARK(BM_AVX2_Cosine) BENCH_DIMS ->Unit(benchmark::kNanosecond);
BENCHMARK(BM_AVX2_Dot)    BENCH_DIMS ->Unit(benchmark::kNanosecond);
#endif
