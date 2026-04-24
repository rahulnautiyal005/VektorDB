/// @file test_distance.cpp
/// @brief Comprehensive unit tests for all distance/similarity functions.
///
/// Tests cover:
///   - Scalar baseline correctness
///   - AVX2 correctness (cross-validated against scalar)
///   - Edge cases: zero vectors, unit vectors, orthogonal, opposite
///   - Various dimensions including non-aligned (not multiple of 8)
///   - Auto-dispatched API

#include <gtest/gtest.h>

#include "vektordb/math/distance.h"
#include "vektordb/math/scalar_ops.h"
#include "vektordb/math/simd_ops.h"
#include "vektordb/core/platform.h"
#include "vektordb/core/types.h"

#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace vektordb;
using namespace vektordb::math;

// ============================================================================
// Test Fixture — shared utilities
// ============================================================================

class DistanceTest : public ::testing::Test {
protected:
    static constexpr float EPSILON = 1e-4f;

    /// Generate a random float vector of given dimension.
    static std::vector<float> random_vector(dim_t dim, float min = -1.0f, float max = 1.0f,
                                            uint32_t seed = 42) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(min, max);
        std::vector<float> v(dim);
        for (auto& x : v) x = dist(rng);
        return v;
    }

    /// Generate a zero vector.
    static std::vector<float> zero_vector(dim_t dim) {
        return std::vector<float>(dim, 0.0f);
    }

    /// Generate a unit vector along axis `axis`.
    static std::vector<float> unit_vector(dim_t dim, dim_t axis) {
        std::vector<float> v(dim, 0.0f);
        if (axis < dim) v[axis] = 1.0f;
        return v;
    }

    /// Normalize a vector to unit length.
    static void normalize(std::vector<float>& v) {
        float norm = 0.0f;
        for (float x : v) norm += x * x;
        norm = std::sqrt(norm);
        if (norm > 1e-10f) {
            for (float& x : v) x /= norm;
        }
    }
};

// ============================================================================
// Scalar L2 Distance Tests
// ============================================================================

TEST_F(DistanceTest, ScalarL2_ZeroVectors) {
    auto a = zero_vector(128);
    auto b = zero_vector(128);
    float d = scalar::l2_distance(a.data(), b.data(), 128);
    EXPECT_NEAR(d, 0.0f, EPSILON);
}

TEST_F(DistanceTest, ScalarL2_IdenticalVectors) {
    auto a = random_vector(256, -10.0f, 10.0f, 123);
    float d = scalar::l2_distance(a.data(), a.data(), 256);
    EXPECT_NEAR(d, 0.0f, EPSILON);
}

TEST_F(DistanceTest, ScalarL2_KnownValue_3D) {
    // a = [1, 2, 3], b = [4, 5, 6]
    // L2² = (1-4)² + (2-5)² + (3-6)² = 9 + 9 + 9 = 27
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    float d = scalar::l2_distance(a, b, 3);
    EXPECT_NEAR(d, 27.0f, EPSILON);
}

TEST_F(DistanceTest, ScalarL2_Symmetry) {
    auto a = random_vector(128, -5.0f, 5.0f, 1);
    auto b = random_vector(128, -5.0f, 5.0f, 2);
    float d_ab = scalar::l2_distance(a.data(), b.data(), 128);
    float d_ba = scalar::l2_distance(b.data(), a.data(), 128);
    EXPECT_NEAR(d_ab, d_ba, EPSILON);
}

TEST_F(DistanceTest, ScalarL2_SingleDimension) {
    float a[] = {3.0f};
    float b[] = {7.0f};
    float d = scalar::l2_distance(a, b, 1);
    EXPECT_NEAR(d, 16.0f, EPSILON);  // (3-7)² = 16
}

// ============================================================================
// Scalar Cosine Tests
// ============================================================================

TEST_F(DistanceTest, ScalarCosine_IdenticalVectors) {
    auto a = random_vector(128, 0.1f, 1.0f, 42);
    float sim = scalar::cosine_similarity(a.data(), a.data(), 128);
    EXPECT_NEAR(sim, 1.0f, EPSILON);
}

TEST_F(DistanceTest, ScalarCosine_OppositeVectors) {
    auto a = random_vector(128, 0.1f, 1.0f, 42);
    std::vector<float> b(128);
    for (dim_t i = 0; i < 128; ++i) b[i] = -a[i];
    float sim = scalar::cosine_similarity(a.data(), b.data(), 128);
    EXPECT_NEAR(sim, -1.0f, EPSILON);
}

TEST_F(DistanceTest, ScalarCosine_OrthogonalVectors) {
    // e1 and e2 are orthogonal
    auto a = unit_vector(128, 0);
    auto b = unit_vector(128, 1);
    float sim = scalar::cosine_similarity(a.data(), b.data(), 128);
    EXPECT_NEAR(sim, 0.0f, EPSILON);
}

TEST_F(DistanceTest, ScalarCosine_KnownValue) {
    // a = [1, 0], b = [1, 1]
    // dot = 1, ||a||=1, ||b||=sqrt(2)
    // cos = 1/sqrt(2) ≈ 0.7071
    float a[] = {1.0f, 0.0f};
    float b[] = {1.0f, 1.0f};
    float sim = scalar::cosine_similarity(a, b, 2);
    EXPECT_NEAR(sim, 1.0f / std::sqrt(2.0f), EPSILON);
}

TEST_F(DistanceTest, ScalarCosineDistance_Range) {
    // Cosine distance should be in [0, 2]
    auto a = random_vector(256, -1.0f, 1.0f, 10);
    auto b = random_vector(256, -1.0f, 1.0f, 20);
    float d = scalar::cosine_distance(a.data(), b.data(), 256);
    EXPECT_GE(d, -EPSILON);
    EXPECT_LE(d, 2.0f + EPSILON);
}

// ============================================================================
// Scalar Dot Product Tests
// ============================================================================

TEST_F(DistanceTest, ScalarDot_KnownValue) {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    float dot = scalar::dot_product(a, b, 3);
    EXPECT_NEAR(dot, 32.0f, EPSILON);  // 1*4 + 2*5 + 3*6 = 32
}

TEST_F(DistanceTest, ScalarDot_Orthogonal) {
    auto a = unit_vector(64, 0);
    auto b = unit_vector(64, 1);
    float dot = scalar::dot_product(a.data(), b.data(), 64);
    EXPECT_NEAR(dot, 0.0f, EPSILON);
}

// ============================================================================
// AVX2 Tests — Cross-validated against scalar
// ============================================================================

#ifdef VEKTORDB_AVX2_ENABLED

class AVX2DistanceTest : public DistanceTest {
protected:
    void SetUp() override {
        if (!platform::supports_avx2() || !platform::supports_fma()) {
            GTEST_SKIP() << "AVX2+FMA not supported on this CPU";
        }
    }
};

TEST_F(AVX2DistanceTest, L2_MatchesScalar_Dim128) {
    auto a = random_vector(128, -10.0f, 10.0f, 100);
    auto b = random_vector(128, -10.0f, 10.0f, 200);

    float scalar_result = scalar::l2_distance(a.data(), b.data(), 128);
    float avx2_result   = avx2::l2_distance(a.data(), b.data(), 128);

    EXPECT_NEAR(avx2_result, scalar_result, EPSILON * scalar_result + EPSILON);
}

TEST_F(AVX2DistanceTest, L2_MatchesScalar_Dim768) {
    auto a = random_vector(768, -5.0f, 5.0f, 300);
    auto b = random_vector(768, -5.0f, 5.0f, 400);

    float scalar_result = scalar::l2_distance(a.data(), b.data(), 768);
    float avx2_result   = avx2::l2_distance(a.data(), b.data(), 768);

    EXPECT_NEAR(avx2_result, scalar_result, EPSILON * scalar_result + EPSILON);
}

TEST_F(AVX2DistanceTest, L2_MatchesScalar_Dim1536) {
    auto a = random_vector(1536, -1.0f, 1.0f, 500);
    auto b = random_vector(1536, -1.0f, 1.0f, 600);

    float scalar_result = scalar::l2_distance(a.data(), b.data(), 1536);
    float avx2_result   = avx2::l2_distance(a.data(), b.data(), 1536);

    EXPECT_NEAR(avx2_result, scalar_result, EPSILON * scalar_result + EPSILON);
}

TEST_F(AVX2DistanceTest, L2_NonAligned_Dim13) {
    // 13 is not a multiple of 8 — tests the scalar tail loop
    auto a = random_vector(13, -5.0f, 5.0f, 700);
    auto b = random_vector(13, -5.0f, 5.0f, 800);

    float scalar_result = scalar::l2_distance(a.data(), b.data(), 13);
    float avx2_result   = avx2::l2_distance(a.data(), b.data(), 13);

    EXPECT_NEAR(avx2_result, scalar_result, EPSILON * scalar_result + EPSILON);
}

TEST_F(AVX2DistanceTest, L2_NonAligned_Dim100) {
    auto a = random_vector(100, -5.0f, 5.0f, 900);
    auto b = random_vector(100, -5.0f, 5.0f, 1000);

    float scalar_result = scalar::l2_distance(a.data(), b.data(), 100);
    float avx2_result   = avx2::l2_distance(a.data(), b.data(), 100);

    EXPECT_NEAR(avx2_result, scalar_result, EPSILON * scalar_result + EPSILON);
}

TEST_F(AVX2DistanceTest, L2_ZeroVectors) {
    auto a = zero_vector(256);
    auto b = zero_vector(256);
    float d = avx2::l2_distance(a.data(), b.data(), 256);
    EXPECT_NEAR(d, 0.0f, EPSILON);
}

TEST_F(AVX2DistanceTest, L2_IdenticalVectors) {
    auto a = random_vector(512, -1.0f, 1.0f, 42);
    float d = avx2::l2_distance(a.data(), a.data(), 512);
    EXPECT_NEAR(d, 0.0f, EPSILON);
}

TEST_F(AVX2DistanceTest, Cosine_MatchesScalar_Dim128) {
    auto a = random_vector(128, -1.0f, 1.0f, 1100);
    auto b = random_vector(128, -1.0f, 1.0f, 1200);

    float scalar_result = scalar::cosine_similarity(a.data(), b.data(), 128);
    float avx2_result   = avx2::cosine_similarity(a.data(), b.data(), 128);

    EXPECT_NEAR(avx2_result, scalar_result, EPSILON);
}

TEST_F(AVX2DistanceTest, Cosine_MatchesScalar_Dim768) {
    auto a = random_vector(768, -1.0f, 1.0f, 1300);
    auto b = random_vector(768, -1.0f, 1.0f, 1400);

    float scalar_result = scalar::cosine_similarity(a.data(), b.data(), 768);
    float avx2_result   = avx2::cosine_similarity(a.data(), b.data(), 768);

    EXPECT_NEAR(avx2_result, scalar_result, EPSILON);
}

TEST_F(AVX2DistanceTest, Cosine_MatchesScalar_Dim1536) {
    auto a = random_vector(1536, -1.0f, 1.0f, 1500);
    auto b = random_vector(1536, -1.0f, 1.0f, 1600);

    float scalar_result = scalar::cosine_similarity(a.data(), b.data(), 1536);
    float avx2_result   = avx2::cosine_similarity(a.data(), b.data(), 1536);

    EXPECT_NEAR(avx2_result, scalar_result, EPSILON);
}

TEST_F(AVX2DistanceTest, Cosine_NonAligned_Dim13) {
    auto a = random_vector(13, 0.1f, 1.0f, 1700);
    auto b = random_vector(13, 0.1f, 1.0f, 1800);

    float scalar_result = scalar::cosine_similarity(a.data(), b.data(), 13);
    float avx2_result   = avx2::cosine_similarity(a.data(), b.data(), 13);

    EXPECT_NEAR(avx2_result, scalar_result, EPSILON);
}

TEST_F(AVX2DistanceTest, Cosine_Identical) {
    auto a = random_vector(256, 0.1f, 1.0f, 1900);
    float sim = avx2::cosine_similarity(a.data(), a.data(), 256);
    EXPECT_NEAR(sim, 1.0f, EPSILON);
}

TEST_F(AVX2DistanceTest, Cosine_Orthogonal) {
    auto a = unit_vector(256, 0);
    auto b = unit_vector(256, 1);
    float sim = avx2::cosine_similarity(a.data(), b.data(), 256);
    EXPECT_NEAR(sim, 0.0f, EPSILON);
}

TEST_F(AVX2DistanceTest, Dot_MatchesScalar_Dim128) {
    auto a = random_vector(128, -5.0f, 5.0f, 2000);
    auto b = random_vector(128, -5.0f, 5.0f, 2100);

    float scalar_result = scalar::dot_product(a.data(), b.data(), 128);
    float avx2_result   = avx2::dot_product(a.data(), b.data(), 128);

    EXPECT_NEAR(avx2_result, scalar_result, EPSILON * std::abs(scalar_result) + EPSILON);
}

TEST_F(AVX2DistanceTest, Dot_NonAligned_Dim7) {
    auto a = random_vector(7, -5.0f, 5.0f, 2200);
    auto b = random_vector(7, -5.0f, 5.0f, 2300);

    float scalar_result = scalar::dot_product(a.data(), b.data(), 7);
    float avx2_result   = avx2::dot_product(a.data(), b.data(), 7);

    EXPECT_NEAR(avx2_result, scalar_result, EPSILON * std::abs(scalar_result) + EPSILON);
}

// Stress test: many dimensions
TEST_F(AVX2DistanceTest, L2_LargeDimension_4096) {
    auto a = random_vector(4096, -1.0f, 1.0f, 3000);
    auto b = random_vector(4096, -1.0f, 1.0f, 3100);

    float scalar_result = scalar::l2_distance(a.data(), b.data(), 4096);
    float avx2_result   = avx2::l2_distance(a.data(), b.data(), 4096);

    // Larger dim means more FP accumulation error — use relative tolerance
    EXPECT_NEAR(avx2_result, scalar_result, 0.01f * scalar_result);
}

#endif // VEKTORDB_AVX2_ENABLED

// ============================================================================
// Auto-dispatched API Tests
// ============================================================================

TEST_F(DistanceTest, Dispatch_L2_Works) {
    auto a = random_vector(128, -1.0f, 1.0f, 5000);
    auto b = random_vector(128, -1.0f, 1.0f, 5001);

    float d = l2_distance(a.data(), b.data(), 128);
    float ref = scalar::l2_distance(a.data(), b.data(), 128);

    EXPECT_NEAR(d, ref, EPSILON * ref + EPSILON);
}

TEST_F(DistanceTest, Dispatch_Cosine_Works) {
    auto a = random_vector(128, 0.1f, 1.0f, 6000);
    auto b = random_vector(128, 0.1f, 1.0f, 6001);

    float d = cosine_distance(a.data(), b.data(), 128);
    float ref = scalar::cosine_distance(a.data(), b.data(), 128);

    EXPECT_NEAR(d, ref, EPSILON);
}

// ============================================================================
// Platform Detection Test
// ============================================================================

TEST(PlatformTest, DetectsCPUFeatures) {
    // Just verify these don't crash
    bool sse2 = platform::supports_sse2();
    bool avx  = platform::supports_avx();
    bool avx2 = platform::supports_avx2();
    bool fma  = platform::supports_fma();
    bool avx512 = platform::supports_avx512f();

    // SSE2 should always be true on x86-64
    EXPECT_TRUE(sse2);

    // Print for diagnostic visibility in test output
    printf("\n  CPU Features: SSE2=%d AVX=%d AVX2=%d FMA=%d AVX-512F=%d\n",
           sse2, avx, avx2, fma, avx512);
}
