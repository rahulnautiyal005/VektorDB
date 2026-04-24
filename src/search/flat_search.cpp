/// @file flat_search.cpp
/// @brief Brute-force nearest neighbor search implementation.

#include "vektordb/search/flat_search.h"
#include "vektordb/math/distance.h"

#include <queue>
#include <algorithm>
#include <functional>

namespace vektordb::search {

std::vector<SearchResult> flat_search(
    const storage::VectorStore& store,
    const float* query,
    uint32_t k)
{
    if (store.size() == 0 || k == 0) {
        return {};
    }

    // Get the appropriate distance function (auto-dispatches to AVX2 if available)
    math::DistanceFn dist_fn;
    if (store.metric() == math::DistanceMetric::L2) {
        dist_fn = math::get_l2_distance_fn();
    } else {
        dist_fn = math::get_cosine_distance_fn();
    }

    const dim_t dim = store.dimension();
    const uint64_t n = store.size();
    const uint32_t actual_k = static_cast<uint32_t>(std::min(static_cast<uint64_t>(k), n));

    // Max-heap: keeps the k closest vectors seen so far.
    // The top of the heap is the farthest of the k closest.
    // When we find a closer vector, we pop the farthest and push the new one.
    std::priority_queue<SearchResult, std::vector<SearchResult>> max_heap;

    const float* data = store.raw_data();

    for (uint64_t i = 0; i < n; ++i) {
        const float* vec = data + i * dim;
        distance_t d = dist_fn(query, vec, dim);

        if (max_heap.size() < actual_k) {
            max_heap.push({static_cast<vec_id_t>(i), d});
        } else if (d < max_heap.top().distance) {
            max_heap.pop();
            max_heap.push({static_cast<vec_id_t>(i), d});
        }
    }

    // Extract results in ascending distance order
    std::vector<SearchResult> results;
    results.reserve(max_heap.size());
    while (!max_heap.empty()) {
        results.push_back(max_heap.top());
        max_heap.pop();
    }

    // Reverse: heap gave us descending, we want ascending
    std::reverse(results.begin(), results.end());

    return results;
}

std::vector<std::vector<SearchResult>> flat_search_batch(
    const storage::VectorStore& store,
    const float* queries,
    uint32_t num_queries,
    uint32_t k)
{
    const dim_t dim = store.dimension();
    std::vector<std::vector<SearchResult>> all_results(num_queries);

    for (uint32_t q = 0; q < num_queries; ++q) {
        all_results[q] = flat_search(store, queries + static_cast<std::size_t>(q) * dim, k);
    }

    return all_results;
}

} // namespace vektordb::search
