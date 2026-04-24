#pragma once

/// @file platform.h
/// @brief Runtime CPU feature detection for SIMD dispatch.

#include <cstdint>

#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <cpuid.h>
#endif

namespace vektordb::platform {

namespace detail {

/// Execute CPUID instruction.
/// @param leaf     EAX input (function selector)
/// @param subleaf  ECX input (sub-function selector)
/// @param eax, ebx, ecx, edx  Output registers
inline void cpuid(uint32_t leaf, uint32_t subleaf,
                  uint32_t& eax, uint32_t& ebx,
                  uint32_t& ecx, uint32_t& edx) {
#ifdef _MSC_VER
    int regs[4];
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    eax = static_cast<uint32_t>(regs[0]);
    ebx = static_cast<uint32_t>(regs[1]);
    ecx = static_cast<uint32_t>(regs[2]);
    edx = static_cast<uint32_t>(regs[3]);
#else
    __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
#endif
}

/// Cached CPU features — computed once on first access.
struct CpuFeatures {
    bool has_sse2    = false;
    bool has_avx     = false;
    bool has_avx2    = false;
    bool has_fma     = false;
    bool has_avx512f = false;

    CpuFeatures() {
        uint32_t eax, ebx, ecx, edx;

        // Basic features (leaf 1)
        cpuid(1, 0, eax, ebx, ecx, edx);
        has_sse2 = (edx >> 26) & 1;   // SSE2: EDX bit 26
        has_avx  = (ecx >> 28) & 1;   // AVX:  ECX bit 28
        has_fma  = (ecx >> 12) & 1;   // FMA:  ECX bit 12

        // Extended features (leaf 7, subleaf 0)
        cpuid(7, 0, eax, ebx, ecx, edx);
        has_avx2    = (ebx >> 5)  & 1;  // AVX2:    EBX bit 5
        has_avx512f = (ebx >> 16) & 1;  // AVX-512F: EBX bit 16
    }
};

/// Singleton accessor for CPU features.
inline const CpuFeatures& get_features() {
    static const CpuFeatures features;
    return features;
}

} // namespace detail

// ============================================================================
// Public API
// ============================================================================

/// @return true if CPU supports SSE2.
inline bool supports_sse2() { return detail::get_features().has_sse2; }

/// @return true if CPU supports AVX.
inline bool supports_avx() { return detail::get_features().has_avx; }

/// @return true if CPU supports AVX2.
inline bool supports_avx2() { return detail::get_features().has_avx2; }

/// @return true if CPU supports FMA (Fused Multiply-Add).
inline bool supports_fma() { return detail::get_features().has_fma; }

/// @return true if CPU supports AVX-512 Foundation.
inline bool supports_avx512f() { return detail::get_features().has_avx512f; }

/// Print detected CPU features to stdout (for diagnostics).
inline void print_features() {
    const auto& f = detail::get_features();
    printf("=== VektorDB CPU Feature Detection ===\n");
    printf("  SSE2:     %s\n", f.has_sse2    ? "YES" : "NO");
    printf("  AVX:      %s\n", f.has_avx     ? "YES" : "NO");
    printf("  AVX2:     %s\n", f.has_avx2    ? "YES" : "NO");
    printf("  FMA:      %s\n", f.has_fma     ? "YES" : "NO");
    printf("  AVX-512F: %s\n", f.has_avx512f ? "YES" : "NO");
    printf("======================================\n");
}

} // namespace vektordb::platform
