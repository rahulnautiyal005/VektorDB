#pragma once

/// @file vektor_db.h
/// @brief Top-level API for VektorDB.

#include "vektordb/core/types.h"
#include "vektordb/math/distance.h"
#include "vektordb/storage/vector_store.h"
#include "vektordb/index/hnsw.h"

#include <string>
#include <vector>
#include <memory>

namespace vektordb {

/// Main interface to the VektorDB engine.
///
/// Combines the VectorStore (storage) and HnswIndex (similarity search)
/// into a unified, thread-safe database engine.
class VektorDB {
public:
    /// Create a new in-memory database.
    /// @param dimension Vector dimension
    /// @param metric    Distance metric
    /// @param config    HNSW configuration
    VektorDB(dim_t dimension, math::DistanceMetric metric,
             const index::HnswConfig& config = index::HnswConfig{});

    ~VektorDB() = default;

    /// Insert a vector into the database.
    /// @param vec Vector data (dim floats)
    /// @param id  External ID
    void insert(const float* vec, vec_id_t id);

    /// Search the database for the top-k nearest neighbors using HNSW.
    /// @param query     Query vector (dim floats)
    /// @param k         Number of results to return
    /// @param ef_search Exploration factor during search
    /// @return          List of (id, distance) pairs
    std::vector<search::SearchResult> search(
        const float* query, uint32_t k, uint32_t ef_search = 50) const;

    /// Exact (brute-force) search, bypassing the HNSW index.
    /// Useful for small datasets or validation.
    /// @param query Query vector
    /// @param k     Number of results to return
    /// @return      List of (id, distance) pairs
    std::vector<search::SearchResult> search_exact(
        const float* query, uint32_t k) const;

    /// @return Number of vectors in the database.
    uint64_t size() const noexcept { return store_.size(); }

    /// @return True if SIMD (AVX2) acceleration is available and active.
    static bool is_simd_enabled() noexcept;

private:
    dim_t dimension_;
    storage::VectorStore store_;
    index::HnswIndex index_;
};

} // namespace vektordb
