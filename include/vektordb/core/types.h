#pragma once

/// @file types.h
/// @brief Core type definitions and aligned memory utilities for VektorDB.

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>
#include <limits>
#include <cstdlib>

#ifdef _WIN32
    #include <malloc.h>  // _aligned_malloc / _aligned_free (MSVC & MinGW)
#endif

namespace vektordb {

// ============================================================================
// Fundamental Types
// ============================================================================

/// Dimension of a vector (number of components).
using dim_t = uint32_t;

/// Unique identifier for a stored vector.
using vec_id_t = uint64_t;

/// Result type for distance computations.
using distance_t = float;

/// Invalid vector ID sentinel.
inline constexpr vec_id_t INVALID_VEC_ID = std::numeric_limits<vec_id_t>::max();

/// Maximum distance sentinel.
inline constexpr distance_t MAX_DISTANCE = std::numeric_limits<distance_t>::max();

// ============================================================================
// Aligned Allocator (for SIMD-friendly memory)
// ============================================================================

/// Custom allocator that guarantees memory alignment.
/// Required for aligned SIMD loads (_mm256_load_ps needs 32-byte alignment).
/// @tparam T         Element type
/// @tparam Alignment Byte alignment (32 for AVX2, 64 for AVX-512)
template <typename T, std::size_t Alignment = 32>
class AlignedAllocator {
public:
    using value_type = T;

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n == 0) return nullptr;

        // Ensure allocation size is at least aligned
        std::size_t bytes = n * sizeof(T);

        #ifdef _WIN32
            void* ptr = _aligned_malloc(bytes, Alignment);
        #else
            void* ptr = std::aligned_alloc(Alignment,
                // aligned_alloc requires size to be multiple of alignment
                (bytes + Alignment - 1) & ~(Alignment - 1));
        #endif

        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, [[maybe_unused]] std::size_t n) noexcept {
        #ifdef _WIN32
            _aligned_free(ptr);
        #else
            std::free(ptr);
        #endif
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    bool operator==(const AlignedAllocator&) const noexcept { return true; }
    bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

// ============================================================================
// Aligned Vector Types
// ============================================================================

/// 32-byte aligned float vector (optimal for AVX2).
using AlignedFloatVector = std::vector<float, AlignedAllocator<float, 32>>;

} // namespace vektordb
