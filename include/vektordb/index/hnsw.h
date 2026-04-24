#pragma once

/// @file hnsw.h
/// @brief Hierarchical Navigable Small World (HNSW) graph index.
///
/// Implementation based on the paper:
///   "Efficient and robust approximate nearest neighbor search using
///    Hierarchical Navigable Small World graphs"
///   — Malkov, Yashunin (2018)
///
/// Key concepts:
///   - Multi-layer navigable small-world graph
///   - Top layers are sparse (long-range connections for coarse navigation)
///   - Bottom layer is dense (short-range connections for precise search)
///   - Probabilistic level assignment via exponential distribution
///   - Greedy search from entry point through all layers
///
/// Thread safety:
///   - Concurrent reads (search) are safe via shared_mutex
///   - Inserts acquire exclusive locks at the node level

#include "vektordb/core/types.h"
#include "vektordb/math/distance.h"
#include "vektordb/search/flat_search.h"

#include <vector>
#include <mutex>
#include <shared_mutex>
#include <random>
#include <atomic>
#include <memory>
#include <queue>
#include <unordered_set>
#include <functional>

namespace vektordb::index {

/// Configuration parameters for HNSW index construction.
struct HnswConfig {
    /// Max number of connections per node per layer (except layer 0).
    /// Higher M = better recall, more memory, slower insert.
    /// Recommended: 12-48. Default: 16.
    uint32_t M = 16;

    /// Max connections at layer 0 (usually 2*M for better recall).
    uint32_t M0 = 32;

    /// Size of the dynamic candidate list during construction.
    /// Higher ef = better index quality, slower build.
    /// Recommended: 100-500. Default: 200.
    uint32_t ef_construction = 200;

    /// Multiplier for level generation: 1/ln(M).
    /// Controls the probability of a node appearing at higher layers.
    double level_mult = 0.0;  // 0 = auto-calculate as 1/ln(M)

    /// Random seed for reproducible index construction.
    uint64_t seed = 42;
};

/// HNSW Index for approximate nearest neighbor search.
///
/// Usage:
/// ```cpp
/// HnswConfig cfg;
/// cfg.M = 16;
/// cfg.ef_construction = 200;
///
/// HnswIndex index(128, DistanceMetric::L2, cfg);
///
/// // Insert vectors
/// for (auto& vec : vectors) {
///     index.insert(vec.data(), vec.id);
/// }
///
/// // Search
/// auto results = index.search(query.data(), 10, 50);
/// ```
class HnswIndex {
public:
    /// Create a new HNSW index.
    /// @param dimension  Vector dimension
    /// @param metric     Distance metric
    /// @param config     HNSW parameters
    HnswIndex(dim_t dimension, math::DistanceMetric metric,
              const HnswConfig& config = HnswConfig{});

    ~HnswIndex() = default;

    // Non-copyable, movable
    HnswIndex(const HnswIndex&) = delete;
    HnswIndex& operator=(const HnswIndex&) = delete;
    HnswIndex(HnswIndex&&) = default;
    HnswIndex& operator=(HnswIndex&&) = default;

    /// Insert a vector into the index.
    /// @param vec  Pointer to dim floats
    /// @param id   External vector ID
    void insert(const float* vec, vec_id_t id);

    /// Search for k approximate nearest neighbors.
    /// @param query      Query vector (dim floats)
    /// @param k          Number of neighbors to return
    /// @param ef_search  Size of the dynamic candidate list during search.
    ///                   Higher ef = better recall, slower search.
    ///                   Must be >= k. Recommended: 50-200.
    /// @return           Vector of (id, distance) pairs, sorted ascending
    std::vector<search::SearchResult> search(
        const float* query, uint32_t k, uint32_t ef_search = 50) const;

    /// @return Number of vectors in the index.
    uint64_t size() const noexcept { return num_elements_.load(); }

    /// @return Vector dimension.
    dim_t dimension() const noexcept { return dimension_; }

    /// @return Current max layer in the graph.
    int max_level() const noexcept { return max_level_.load(); }

    /// @return The HNSW configuration.
    const HnswConfig& config() const noexcept { return config_; }

private:
    // ========================================================================
    // Internal Data Structures
    // ========================================================================

    /// Internal node ID (0-indexed, sequential).
    using internal_id_t = uint32_t;

    /// A node in the HNSW graph.
    struct Node {
        vec_id_t external_id;              ///< User-facing vector ID
        int level;                          ///< Highest layer this node appears in
        std::vector<float> vector;          ///< Copy of the vector data
        std::vector<std::vector<internal_id_t>> neighbors;  ///< Adjacency list per layer
        mutable std::mutex mtx;             ///< Per-node lock for concurrent inserts
    };

    /// Priority queue element for HNSW search.
    struct Candidate {
        distance_t distance;
        internal_id_t id;

        bool operator<(const Candidate& o) const { return distance < o.distance; }
        bool operator>(const Candidate& o) const { return distance > o.distance; }
    };

    // Min-heap (closest first)
    using MinHeap = std::priority_queue<Candidate, std::vector<Candidate>,
                                        std::greater<Candidate>>;
    // Max-heap (farthest first) — for maintaining top-k
    using MaxHeap = std::priority_queue<Candidate, std::vector<Candidate>,
                                        std::less<Candidate>>;

    // ========================================================================
    // Core Algorithms
    // ========================================================================

    /// Greedy search on a single layer.
    /// Finds ef closest neighbors to the query starting from entry points.
    /// @param query       Query vector
    /// @param entry_points  Starting nodes
    /// @param ef          Number of candidates to track
    /// @param layer       Layer to search on
    /// @return            Max-heap of ef closest candidates
    MaxHeap search_layer(const float* query,
                         const std::vector<internal_id_t>& entry_points,
                         uint32_t ef, int layer) const;

    /// Select neighbors using the simple heuristic.
    /// Keeps the M closest candidates.
    /// @param candidates  Candidate list (passed by value to prevent destruction)
    /// @param M           Max neighbors to select
    /// @return            Selected neighbor IDs
    std::vector<internal_id_t> select_neighbors_simple(
        MaxHeap candidates, uint32_t M) const;

    /// Select neighbors using the heuristic from the paper.
    /// Considers connectivity and diversity.
    std::vector<internal_id_t> select_neighbors_heuristic(
        const float* query, MaxHeap candidates,
        uint32_t M, int layer) const;

    /// Assign a random level to a new node.
    /// Uses exponential distribution: floor(-ln(uniform) * level_mult)
    int random_level();

    /// Compute distance between query and internal node.
    distance_t compute_distance(const float* query, internal_id_t id) const;

    // ========================================================================
    // Member Variables
    // ========================================================================

    dim_t dimension_;
    math::DistanceMetric metric_;
    math::DistanceFn dist_fn_;
    HnswConfig config_;

    /// All nodes in the graph (indexed by internal ID).
    std::vector<std::unique_ptr<Node>> nodes_;

    /// Mapping: external ID → internal ID.
    std::vector<vec_id_t> id_to_external_;

    /// Entry point of the graph (node at the highest level).
    std::atomic<internal_id_t> entry_point_{0};

    /// Current maximum level in the graph.
    std::atomic<int> max_level_{-1};

    /// Number of elements in the index.
    std::atomic<uint64_t> num_elements_{0};

    /// Global lock for structural changes (entry point, max level).
    mutable std::shared_mutex global_mutex_;

    /// Random number generator for level assignment.
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uniform_dist_{0.0, 1.0};
    std::mutex rng_mutex_;  ///< Protects rng_ for thread safety
};

} // namespace vektordb::index
