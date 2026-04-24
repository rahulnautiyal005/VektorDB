#pragma once

/// @file flat_search.h
/// @brief Brute-force O(N) nearest neighbor search using SIMD distance functions.
///
/// This is the baseline search engine that scans every vector in the store.
/// While O(N), it's hardware-accelerated via AVX2 and serves as:
/// 1. Ground truth for validating HNSW recall
/// 2. Optimal for small datasets (< 10K vectors)
/// 3. Fallback when no index is available

#include "vektordb/core/types.h"
#include "vektordb/math/distance.h"
#include "vektordb/storage/vector_store.h"

#include <vector>
#include <utility>

namespace vektordb::search {

/// Result of a nearest neighbor search.
struct SearchResult {
    vec_id_t   id;        ///< Vector ID
    distance_t distance;  ///< Distance from query

    bool operator<(const SearchResult& o) const { return distance < o.distance; }
    bool operator>(const SearchResult& o) const { return distance > o.distance; }
};

/// Brute-force nearest neighbor search.
///
/// Scans every vector in the store and returns the top-k closest.
/// Uses a max-heap to maintain the top-k candidates efficiently.
///
/// @param store   Vector store to search
/// @param query   Query vector (must match store dimension)
/// @param k       Number of nearest neighbors to return
/// @return        Vector of (id, distance) pairs, sorted by distance (ascending)
std::vector<SearchResult> flat_search(
    const storage::VectorStore& store,
    const float* query,
    uint32_t k
);

/// Batch search: run flat_search for multiple queries.
///
/// @param store    Vector store to search
/// @param queries  Pointer to num_queries * dim floats
/// @param num_queries  Number of query vectors
/// @param k        Number of nearest neighbors per query
/// @return         Vector of result vectors (one per query)
std::vector<std::vector<SearchResult>> flat_search_batch(
    const storage::VectorStore& store,
    const float* queries,
    uint32_t num_queries,
    uint32_t k
);

} // namespace vektordb::search
