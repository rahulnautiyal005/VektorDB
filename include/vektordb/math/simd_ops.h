#pragma once

/// @file simd_ops.h
/// @brief AVX2 SIMD-accelerated distance function implementations.
///
/// These implementations process 8 floats per cycle using 256-bit
/// AVX2 registers, achieving 8x throughput over scalar code.
/// FMA (Fused Multiply-Add) is used when available for additional speedup.
///
/// Requires: CPU with AVX2 + FMA support.
/// Compile with: -mavx2 -mfma (GCC/Clang) or /arch:AVX2 (MSVC)

#include "vektordb/core/types.h"

#ifdef VEKTORDB_AVX2_ENABLED

namespace vektordb::math::avx2 {

/// AVX2-accelerated squared L2 distance.
///
/// Algorithm:
///   - Process 8 floats per iteration via __m256
///   - _mm256_sub_ps for element-wise subtraction
///   - _mm256_mul_ps (or _mm256_fmadd_ps) for squaring
///   - Horizontal sum at the end
///   - Scalar tail loop for remaining elements (dim % 8)
///
/// @param a    Pointer to first vector (dim floats)
/// @param b    Pointer to second vector (dim floats)
/// @param dim  Number of dimensions
/// @return     Squared L2 distance
distance_t l2_distance(const float* a, const float* b, dim_t dim);

/// AVX2-accelerated cosine distance.
///
/// Algorithm:
///   - Simultaneously accumulate dot(a,b), norm(a), norm(b)
///   - Uses 3 __m256 accumulators per 8-float chunk
///   - Horizontal sum + final division
///   - Returns 1.0 - similarity
///
/// @param a    Pointer to first vector (dim floats)
/// @param b    Pointer to second vector (dim floats)
/// @param dim  Number of dimensions
/// @return     Cosine distance in range [0, 2]
distance_t cosine_distance(const float* a, const float* b, dim_t dim);

/// AVX2-accelerated cosine similarity.
///
/// @param a    Pointer to first vector (dim floats)
/// @param b    Pointer to second vector (dim floats)
/// @param dim  Number of dimensions
/// @return     Cosine similarity in range [-1, 1]
distance_t cosine_similarity(const float* a, const float* b, dim_t dim);

/// AVX2-accelerated dot product.
///
/// @param a    Pointer to first vector (dim floats)
/// @param b    Pointer to second vector (dim floats)
/// @param dim  Number of dimensions
/// @return     Dot product
distance_t dot_product(const float* a, const float* b, dim_t dim);

} // namespace vektordb::math::avx2

#endif // VEKTORDB_AVX2_ENABLED
