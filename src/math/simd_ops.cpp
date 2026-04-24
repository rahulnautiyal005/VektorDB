/// @file simd_ops.cpp
/// @brief AVX2 SIMD-accelerated distance function implementations.
///
/// Each function processes 8 floats per cycle using 256-bit YMM registers.
/// Inner loops are further unrolled to process 32 floats (4x __m256) to
/// maximize instruction-level parallelism and hide latency.

#include "vektordb/math/simd_ops.h"

#ifdef VEKTORDB_AVX2_ENABLED

#include <immintrin.h>  // AVX2 + FMA intrinsics
#include <cmath>

namespace vektordb::math::avx2 {

// ============================================================================
// Helper: Horizontal sum of __m256 (8 floats → 1 float)
// ============================================================================
static inline float hsum_avx(__m256 v) {
    // Step 1: Add high 128 bits to low 128 bits
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum128 = _mm_add_ps(lo, hi);

    // Step 2: Horizontal add within 128-bit lane
    __m128 shuf = _mm_movehdup_ps(sum128);       // [1,1,3,3]
    __m128 sums = _mm_add_ps(sum128, shuf);      // [0+1, _, 2+3, _]
    shuf = _mm_movehl_ps(shuf, sums);             // [2+3, ...]
    sums = _mm_add_ss(sums, shuf);                // [0+1+2+3]

    return _mm_cvtss_f32(sums);
}

// ============================================================================
// L2 Squared Distance — AVX2
// ============================================================================
distance_t l2_distance(const float* a, const float* b, dim_t dim) {
    //
    // Strategy:
    //   - 4x unrolled: process 32 floats per outer iteration
    //   - Each iteration: load, subtract, FMA accumulate
    //   - Scalar tail for remainder
    //

    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    dim_t i = 0;

    // Main loop: 32 floats per iteration (4 × 8)
    const dim_t block_end = dim & ~dim_t(31);
    for (; i < block_end; i += 32) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 diff0 = _mm256_sub_ps(va0, vb0);
        sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);

        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        __m256 diff1 = _mm256_sub_ps(va1, vb1);
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);

        __m256 va2 = _mm256_loadu_ps(a + i + 16);
        __m256 vb2 = _mm256_loadu_ps(b + i + 16);
        __m256 diff2 = _mm256_sub_ps(va2, vb2);
        sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);

        __m256 va3 = _mm256_loadu_ps(a + i + 24);
        __m256 vb3 = _mm256_loadu_ps(b + i + 24);
        __m256 diff3 = _mm256_sub_ps(va3, vb3);
        sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);
    }

    // Process remaining 8-float chunks
    const dim_t simd_end = dim & ~dim_t(7);
    for (; i < simd_end; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum0 = _mm256_fmadd_ps(diff, diff, sum0);
    }

    // Merge accumulators
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);

    float result = hsum_avx(sum0);

    // Scalar tail for remaining elements
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        result += d * d;
    }

    return result;
}

// ============================================================================
// Cosine Distance — AVX2
// ============================================================================
distance_t cosine_distance(const float* a, const float* b, dim_t dim) {
    return 1.0f - cosine_similarity(a, b, dim);
}

distance_t cosine_similarity(const float* a, const float* b, dim_t dim) {
    //
    // Strategy:
    //   - 3 accumulators per unroll level: dot, norm_a, norm_b
    //   - 4x unrolled: 32 floats per outer iteration
    //   - FMA for multiply-accumulate
    //

    // Accumulators for 4x unrolled loop
    __m256 dot0 = _mm256_setzero_ps();
    __m256 dot1 = _mm256_setzero_ps();
    __m256 na0  = _mm256_setzero_ps();
    __m256 na1  = _mm256_setzero_ps();
    __m256 nb0  = _mm256_setzero_ps();
    __m256 nb1  = _mm256_setzero_ps();

    dim_t i = 0;

    // Main loop: 16 floats per iteration (2 × 8)
    // Using 2x unroll for cosine (6 accumulators = 6 of 16 YMM registers)
    const dim_t block_end = dim & ~dim_t(15);
    for (; i < block_end; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        dot0 = _mm256_fmadd_ps(va0, vb0, dot0);
        na0  = _mm256_fmadd_ps(va0, va0, na0);
        nb0  = _mm256_fmadd_ps(vb0, vb0, nb0);

        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        dot1 = _mm256_fmadd_ps(va1, vb1, dot1);
        na1  = _mm256_fmadd_ps(va1, va1, na1);
        nb1  = _mm256_fmadd_ps(vb1, vb1, nb1);
    }

    // Process remaining 8-float chunks
    const dim_t simd_end = dim & ~dim_t(7);
    for (; i < simd_end; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        dot0 = _mm256_fmadd_ps(va, vb, dot0);
        na0  = _mm256_fmadd_ps(va, va, na0);
        nb0  = _mm256_fmadd_ps(vb, vb, nb0);
    }

    // Merge accumulators
    dot0 = _mm256_add_ps(dot0, dot1);
    na0  = _mm256_add_ps(na0, na1);
    nb0  = _mm256_add_ps(nb0, nb1);

    float dot_sum  = hsum_avx(dot0);
    float na_sum   = hsum_avx(na0);
    float nb_sum   = hsum_avx(nb0);

    // Scalar tail
    for (; i < dim; ++i) {
        dot_sum += a[i] * b[i];
        na_sum  += a[i] * a[i];
        nb_sum  += b[i] * b[i];
    }

    // Final computation
    float denom = std::sqrt(na_sum) * std::sqrt(nb_sum);
    if (denom < 1e-10f) return 0.0f;

    return dot_sum / denom;
}

// ============================================================================
// Dot Product — AVX2
// ============================================================================
distance_t dot_product(const float* a, const float* b, dim_t dim) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    dim_t i = 0;

    // 32 floats per iteration
    const dim_t block_end = dim & ~dim_t(31);
    for (; i < block_end; i += 32) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  sum1);
        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), sum2);
        sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), sum3);
    }

    // Remaining 8-float chunks
    const dim_t simd_end = dim & ~dim_t(7);
    for (; i < simd_end; i += 8) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum0);
    }

    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);

    float result = hsum_avx(sum0);

    // Scalar tail
    for (; i < dim; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

} // namespace vektordb::math::avx2

#endif // VEKTORDB_AVX2_ENABLED
