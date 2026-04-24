/// @file distance.cpp
/// @brief Runtime dispatch — selects the best distance implementation for this CPU.

#include "vektordb/math/distance.h"
#include "vektordb/math/scalar_ops.h"
#include "vektordb/math/simd_ops.h"
#include "vektordb/core/platform.h"

#include <cstdio>

namespace vektordb::math {

// ============================================================================
// Dispatch Logic
// ============================================================================

/// Selects the fastest L2 distance function available on this CPU.
/// Called once (lazily), then the function pointer is cached.
static DistanceFn resolve_l2_distance() {
#ifdef VEKTORDB_AVX2_ENABLED
    if (platform::supports_avx2() && platform::supports_fma()) {
        return avx2::l2_distance;
    }
#endif
    return scalar::l2_distance;
}

/// Selects the fastest cosine distance function available on this CPU.
static DistanceFn resolve_cosine_distance() {
#ifdef VEKTORDB_AVX2_ENABLED
    if (platform::supports_avx2() && platform::supports_fma()) {
        return avx2::cosine_distance;
    }
#endif
    return scalar::cosine_distance;
}

// ============================================================================
// Public API — Cached Function Pointers
// ============================================================================

DistanceFn get_l2_distance_fn() {
    // Static local → initialized exactly once, thread-safe in C++11+
    static const DistanceFn fn = resolve_l2_distance();
    return fn;
}

DistanceFn get_cosine_distance_fn() {
    static const DistanceFn fn = resolve_cosine_distance();
    return fn;
}

} // namespace vektordb::math
