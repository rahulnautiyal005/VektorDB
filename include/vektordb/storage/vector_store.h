#pragma once

/// @file vector_store.h
/// @brief Vector storage engine with .vkdb binary format and mmap support.
///
/// The .vkdb binary format:
///   Offset  Size     Field
///   0       4        Magic ("VKDB")
///   4       4        Version (uint32_t)
///   8       4        Dimension (uint32_t)
///   12      4        Metric type (uint32_t)
///   16      8        Vector count (uint64_t)
///   24      8        Reserved (padding to 32 bytes)
///   32      N*D*4    Contiguous float vectors
///
/// Total header size: 32 bytes (aligned for SIMD loads)

#include "vektordb/core/types.h"
#include "vektordb/math/distance.h"
#include "vektordb/storage/mmap_file.h"

#include <string>
#include <vector>
#include <memory>
#include <mutex>

namespace vektordb::storage {

/// Header of the .vkdb binary file format.
struct VkdbHeader {
    char     magic[4]  = {'V', 'K', 'D', 'B'};
    uint32_t version   = 1;
    uint32_t dimension = 0;
    uint32_t metric    = 0;   // 0 = L2, 1 = Cosine
    uint64_t num_vectors = 0;
    uint64_t reserved  = 0;
};
static_assert(sizeof(VkdbHeader) == 32, "VkdbHeader must be 32 bytes");

/// In-memory vector storage with optional mmap backing.
///
/// Supports two modes:
/// 1. In-memory: vectors stored in a contiguous std::vector<float>
/// 2. Memory-mapped: vectors read directly from a .vkdb file
class VectorStore {
public:
    /// Create an empty in-memory store.
    /// @param dimension  Number of float components per vector
    /// @param metric     Distance metric to use
    explicit VectorStore(dim_t dimension,
                         math::DistanceMetric metric = math::DistanceMetric::L2);

    /// Load from a .vkdb file via memory mapping.
    /// @param path  Path to .vkdb file
    /// @throws std::runtime_error on invalid format
    static VectorStore load(const std::string& path);

    /// Save current vectors to a .vkdb file.
    /// @param path  Output file path
    void save(const std::string& path) const;

    /// Add a single vector. Returns its assigned ID.
    /// @param vec  Pointer to dim floats
    /// @return     Assigned vector ID
    vec_id_t add(const float* vec);

    /// Add multiple vectors in batch.
    /// @param vecs      Pointer to n * dim floats (contiguous)
    /// @param count     Number of vectors
    /// @param out_ids   Output: assigned IDs (optional, can be nullptr)
    void add_batch(const float* vecs, uint64_t count, vec_id_t* out_ids = nullptr);

    /// Get pointer to a vector by ID.
    /// @param id  Vector ID (0-indexed)
    /// @return    Pointer to dim floats (valid as long as store exists)
    const float* get(vec_id_t id) const;

    /// @return Number of stored vectors.
    uint64_t size() const noexcept { return num_vectors_; }

    /// @return Dimension of each vector.
    dim_t dimension() const noexcept { return dimension_; }

    /// @return The distance metric in use.
    math::DistanceMetric metric() const noexcept { return metric_; }

    /// @return Pointer to raw contiguous vector data.
    const float* raw_data() const noexcept { return data_ptr_; }

private:
    dim_t dimension_;
    math::DistanceMetric metric_;
    uint64_t num_vectors_ = 0;

    // In-memory storage (owns the data)
    std::vector<float> data_;

    // Pointer to vector data (into data_ or mmap region)
    const float* data_ptr_ = nullptr;

    // Memory-mapped file (if loaded from disk)
    std::unique_ptr<MmapFile> mmap_file_;
};

} // namespace vektordb::storage
