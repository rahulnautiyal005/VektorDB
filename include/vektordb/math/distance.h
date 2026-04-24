#pragma once

/// @file distance.h
/// @brief Public API for distance/similarity computations.
///
/// This is the main entry point for all distance functions in VektorDB.
/// It provides auto-dispatching functions that select the best available
/// implementation (AVX2 → Scalar) based on runtime CPU detection.

#include "vektordb/core/types.h"

namespace vektordb::math {

// ============================================================================
// Distance Metric Enum
// ============================================================================

/// Supported distance metrics.
enum class DistanceMetric {
    L2,      ///< Squared Euclidean distance: Σ(a_i - b_i)²
    Cosine   ///< Cosine distance: 1 - cos(θ)
};

// ============================================================================
// Function Pointer Type
// ============================================================================

/// Signature for all distance computation functions.
/// @param a    Pointer to first vector
/// @param b    Pointer to second vector
/// @param dim  Number of dimensions
/// @return     Distance value (lower = more similar)
using DistanceFn = distance_t(*)(const float* a, const float* b, dim_t dim);

// ============================================================================
// Auto-Dispatching API (recommended)
// ============================================================================

/// Get the fastest available L2 distance function for this CPU.
/// Uses AVX2 if available, otherwise falls back to scalar.
/// The result is cached — zero overhead after first call.
DistanceFn get_l2_distance_fn();

/// Get the fastest available cosine distance function for this CPU.
/// Returns 1 - cosine_similarity (so lower = more similar).
DistanceFn get_cosine_distance_fn();

// ============================================================================
// Convenience wrappers
// ============================================================================

/// Compute L2 squared distance between vectors a and b.
inline distance_t l2_distance(const float* a, const float* b, dim_t dim) {
    return get_l2_distance_fn()(a, b, dim);
}

/// Compute cosine distance (1 - similarity) between vectors a and b.
inline distance_t cosine_distance(const float* a, const float* b, dim_t dim) {
    return get_cosine_distance_fn()(a, b, dim);
}

} // namespace vektordb::math
