#pragma once

/// @file scalar_ops.h
/// @brief Scalar (non-SIMD) baseline implementations of distance functions.
///
/// These serve as:
/// 1. Reference implementations for correctness testing
/// 2. Fallback when SIMD is not available
/// 3. Performance baseline for benchmarking

#include "vektordb/core/types.h"

namespace vektordb::math::scalar {

/// Compute squared L2 (Euclidean) distance between two vectors.
///
/// Formula: d(a,b) = Σ (a_i - b_i)²
///
/// Note: Returns SQUARED distance (omits sqrt) since we only need
/// relative ordering for nearest-neighbor search. sqrt is monotonic
/// so it doesn't change the ranking.
///
/// @param a    Pointer to first vector (dim floats)
/// @param b    Pointer to second vector (dim floats)
/// @param dim  Number of dimensions
/// @return     Squared Euclidean distance
distance_t l2_distance(const float* a, const float* b, dim_t dim);

/// Compute cosine distance between two vectors.
///
/// Formula: d(a,b) = 1.0 - (a·b) / (||a|| · ||b||)
///
/// Returns cosine DISTANCE (1 - similarity) so that lower values
/// indicate more similar vectors, consistent with L2.
///
/// @param a    Pointer to first vector (dim floats)
/// @param b    Pointer to second vector (dim floats)
/// @param dim  Number of dimensions
/// @return     Cosine distance in range [0, 2]
distance_t cosine_distance(const float* a, const float* b, dim_t dim);

/// Compute raw cosine similarity (not distance).
///
/// Formula: sim(a,b) = (a·b) / (||a|| · ||b||)
///
/// @param a    Pointer to first vector (dim floats)
/// @param b    Pointer to second vector (dim floats)
/// @param dim  Number of dimensions
/// @return     Cosine similarity in range [-1, 1]
distance_t cosine_similarity(const float* a, const float* b, dim_t dim);

/// Compute dot product between two vectors.
///
/// Formula: dot(a,b) = Σ a_i · b_i
///
/// @param a    Pointer to first vector (dim floats)
/// @param b    Pointer to second vector (dim floats)
/// @param dim  Number of dimensions
/// @return     Dot product
distance_t dot_product(const float* a, const float* b, dim_t dim);

} // namespace vektordb::math::scalar
