/// @file scalar_ops.cpp
/// @brief Scalar baseline implementations of distance functions.

#include "vektordb/math/scalar_ops.h"
#include <cmath>
#include <algorithm>

namespace vektordb::math::scalar {

distance_t l2_distance(const float* a, const float* b, dim_t dim) {
    // Squared Euclidean distance: Σ (a_i - b_i)²
    //
    // We use a 4-way unrolled loop to help the compiler pipeline.
    // Even in "scalar" mode, the compiler can auto-vectorize this
    // with SSE2, but it won't match hand-written AVX2.

    distance_t sum = 0.0f;

    // Unrolled main loop (4 elements per iteration)
    dim_t i = 0;
    const dim_t unrolled_end = dim & ~dim_t(3);  // Round down to multiple of 4

    for (; i < unrolled_end; i += 4) {
        float d0 = a[i]     - b[i];
        float d1 = a[i + 1] - b[i + 1];
        float d2 = a[i + 2] - b[i + 2];
        float d3 = a[i + 3] - b[i + 3];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }

    // Tail loop for remaining elements
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }

    return sum;
}

distance_t cosine_distance(const float* a, const float* b, dim_t dim) {
    return 1.0f - cosine_similarity(a, b, dim);
}

distance_t cosine_similarity(const float* a, const float* b, dim_t dim) {
    // Cosine similarity: (a·b) / (||a|| · ||b||)
    //
    // Single pass: accumulate dot product, norm_a², norm_b² simultaneously.
    // This is cache-friendly since we only traverse each array once.

    float dot  = 0.0f;
    float na   = 0.0f;  // ||a||²
    float nb   = 0.0f;  // ||b||²

    // Unrolled main loop
    dim_t i = 0;
    const dim_t unrolled_end = dim & ~dim_t(3);

    for (; i < unrolled_end; i += 4) {
        dot += a[i] * b[i] + a[i+1] * b[i+1] + a[i+2] * b[i+2] + a[i+3] * b[i+3];
        na  += a[i] * a[i] + a[i+1] * a[i+1] + a[i+2] * a[i+2] + a[i+3] * a[i+3];
        nb  += b[i] * b[i] + b[i+1] * b[i+1] + b[i+2] * b[i+2] + b[i+3] * b[i+3];
    }

    // Tail
    for (; i < dim; ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }

    // Guard against zero-norm vectors
    float denom = std::sqrt(na) * std::sqrt(nb);
    if (denom < 1e-10f) return 0.0f;

    return dot / denom;
}

distance_t dot_product(const float* a, const float* b, dim_t dim) {
    float sum = 0.0f;

    dim_t i = 0;
    const dim_t unrolled_end = dim & ~dim_t(3);

    for (; i < unrolled_end; i += 4) {
        sum += a[i] * b[i] + a[i+1] * b[i+1] + a[i+2] * b[i+2] + a[i+3] * b[i+3];
    }

    for (; i < dim; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

} // namespace vektordb::math::scalar
