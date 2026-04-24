/// @file hnsw.cpp
/// @brief HNSW algorithm implementation.

#include "vektordb/index/hnsw.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace vektordb::index {

HnswIndex::HnswIndex(dim_t dimension, math::DistanceMetric metric, const HnswConfig& config)
    : dimension_(dimension), metric_(metric), config_(config)
{
    if (dimension == 0) {
        throw std::runtime_error("HnswIndex: dimension must be > 0");
    }

    if (config_.level_mult == 0.0) {
        config_.level_mult = 1.0 / std::log(config_.M);
    }

    if (metric_ == math::DistanceMetric::L2) {
        dist_fn_ = math::get_l2_distance_fn();
    } else {
        dist_fn_ = math::get_cosine_distance_fn();
    }

    rng_.seed(config_.seed);
}

int HnswIndex::random_level() {
    std::lock_guard<std::mutex> lock(rng_mutex_);
    double r = uniform_dist_(rng_);
    if (r == 0.0) r = std::numeric_limits<double>::epsilon();
    return static_cast<int>(-std::log(r) * config_.level_mult);
}

distance_t HnswIndex::compute_distance(const float* query, internal_id_t id) const {
    const float* vec = nodes_[id]->vector.data();
    return dist_fn_(query, vec, dimension_);
}

void HnswIndex::insert(const float* vec, vec_id_t id) {
    int level = random_level();

    // Create the new node
    auto new_node = std::make_unique<Node>();
    new_node->external_id = id;
    new_node->level = level;
    new_node->vector.assign(vec, vec + dimension_);
    new_node->neighbors.resize(level + 1);

    internal_id_t new_id;
    int curr_max_level;
    internal_id_t curr_ep;

    // Acquire global lock (write mode if first element, read mode otherwise)
    {
        std::unique_lock<std::shared_mutex> global_write_lock(global_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> global_read_lock(global_mutex_, std::defer_lock);

        bool is_first = false;
        
        // Fast optimistic check
        if (max_level_.load() == -1) {
            global_write_lock.lock();
            if (max_level_.load() == -1) {
                is_first = true;
                new_id = 0;
                nodes_.push_back(std::move(new_node));
                id_to_external_.push_back(id);
                entry_point_.store(0);
                max_level_.store(level);
                num_elements_.store(1);
            } else {
                global_write_lock.unlock();
                global_read_lock.lock();
            }
        } else {
            global_read_lock.lock();
        }

        if (is_first) return;

        // Atomically append to node arrays
        // In a true high-concurrency production index, you'd use a thread-safe append-only array.
        // Here, we upgrade to write lock just for the append to nodes_ vector for simplicity.
        if (global_read_lock.owns_lock()) {
            global_read_lock.unlock();
        }
        
        global_write_lock.lock();
        new_id = static_cast<internal_id_t>(nodes_.size());
        nodes_.push_back(std::move(new_node));
        id_to_external_.push_back(id);
        
        curr_max_level = max_level_.load();
        curr_ep = entry_point_.load();
        
        // We can downgrade back to a read lock for the slow search phases
        global_write_lock.unlock();
        global_read_lock.lock();

        distance_t dist = compute_distance(vec, curr_ep);
        internal_id_t ep = curr_ep;

        // Phase 1: Search top layers (sparse) from max_level down to level+1
        for (int lc = curr_max_level; lc > level; --lc) {
            bool changed = true;
            while (changed) {
                changed = false;
                std::lock_guard<std::mutex> node_lock(nodes_[ep]->mtx);
                const auto& neighbors = nodes_[ep]->neighbors[lc];
                
                for (internal_id_t neighbor : neighbors) {
                    distance_t d = compute_distance(vec, neighbor);
                    if (d < dist) {
                        dist = d;
                        ep = neighbor;
                        changed = true;
                    }
                }
            }
        }

        // Phase 2: Insert and link at each level from `level` down to 0
        std::vector<internal_id_t> eps = {ep};
        
        for (int lc = std::min(curr_max_level, level); lc >= 0; --lc) {
            // Find ef_construction closest candidates
            MaxHeap candidates = search_layer(vec, eps, config_.ef_construction, lc);
            
            // Select neighbors heuristically
            std::vector<internal_id_t> selected = select_neighbors_heuristic(vec, candidates, 
                                            (lc == 0) ? config_.M0 : config_.M, lc);
            
            nodes_[new_id]->neighbors[lc] = selected;
            
            // Add reverse connections
            for (internal_id_t neighbor : selected) {
                std::lock_guard<std::mutex> lock(nodes_[neighbor]->mtx);
                auto& neighbor_links = nodes_[neighbor]->neighbors[lc];
                
                neighbor_links.push_back(new_id);
                
                // If max connections exceeded, prune
                uint32_t M_max = (lc == 0) ? config_.M0 : config_.M;
                if (neighbor_links.size() > M_max) {
                    MaxHeap neighbor_candidates;
                    for (internal_id_t n_id : neighbor_links) {
                        distance_t d = compute_distance(nodes_[neighbor]->vector.data(), n_id);
                        neighbor_candidates.push({d, n_id});
                    }
                    
                    neighbor_links = select_neighbors_heuristic(
                        nodes_[neighbor]->vector.data(), neighbor_candidates, M_max, lc);
                }
            }
            
            // Prepare entry points for next level (use current layer's closest)
            eps.clear();
            while (!candidates.empty()) {
                eps.push_back(candidates.top().id);
                candidates.pop();
            }
        }

        // Update entry point if this node is the highest
        if (level > curr_max_level) {
            global_read_lock.unlock();
            global_write_lock.lock();
            // Double check
            if (level > max_level_.load()) {
                entry_point_.store(new_id);
                max_level_.store(level);
            }
        }
    }
    
    num_elements_.fetch_add(1);
}

std::vector<search::SearchResult> HnswIndex::search(
    const float* query, uint32_t k, uint32_t ef_search) const
{
    if (num_elements_.load() == 0 || k == 0) return {};
    
    ef_search = std::max(ef_search, k);

    std::shared_lock<std::shared_mutex> global_lock(global_mutex_);
    
    internal_id_t ep = entry_point_.load();
    int curr_max_level = max_level_.load();
    distance_t dist = compute_distance(query, ep);
    
    // Phase 1: Descend to layer 0 (coarse search)
    for (int lc = curr_max_level; lc > 0; --lc) {
        bool changed = true;
        while (changed) {
            changed = false;
            // No strict node lock needed for search if we tolerate minor stales
            std::lock_guard<std::mutex> node_lock(nodes_[ep]->mtx);
            const auto& neighbors = nodes_[ep]->neighbors[lc];
            
            for (internal_id_t neighbor : neighbors) {
                distance_t d = compute_distance(query, neighbor);
                if (d < dist) {
                    dist = d;
                    ep = neighbor;
                    changed = true;
                }
            }
        }
    }
    
    // Phase 2: Refined search on layer 0 using ef_search
    MaxHeap candidates = search_layer(query, {ep}, ef_search, 0);
    
    // Extract all candidates (max-heap means we get farthest first)
    std::vector<search::SearchResult> results;
    results.reserve(candidates.size());
    
    while (!candidates.empty()) {
        auto cand = candidates.top();
        candidates.pop();
        results.push_back({nodes_[cand.id]->external_id, cand.distance});
    }
    
    // Sort ascending (nearest first) by reversing
    std::reverse(results.begin(), results.end());
    
    if (results.size() > k) {
        results.resize(k);
    }
    
    return results;
}

HnswIndex::MaxHeap HnswIndex::search_layer(
    const float* query, const std::vector<internal_id_t>& entry_points,
    uint32_t ef, int layer) const
{
    std::unordered_set<internal_id_t> visited;
    MinHeap min_heap; // To explore closest neighbors next
    MaxHeap max_heap; // Holds the best ef candidates

    for (internal_id_t ep : entry_points) {
        distance_t dist = compute_distance(query, ep);
        visited.insert(ep);
        min_heap.push({dist, ep});
        max_heap.push({dist, ep});
    }

    while (!min_heap.empty()) {
        Candidate current = min_heap.top();
        min_heap.pop();

        // If the closest candidate is further than the farthest in our top-ef list, stop
        if (max_heap.size() == ef && current.distance > max_heap.top().distance) {
            break;
        }

        std::lock_guard<std::mutex> node_lock(nodes_[current.id]->mtx);
        const auto& neighbors = nodes_[current.id]->neighbors[layer];

        for (internal_id_t neighbor : neighbors) {
            if (visited.insert(neighbor).second) { // If inserted (not visited)
                distance_t d = compute_distance(query, neighbor);

                if (max_heap.size() < ef || d < max_heap.top().distance) {
                    min_heap.push({d, neighbor});
                    max_heap.push({d, neighbor});
                    
                    if (max_heap.size() > ef) {
                        max_heap.pop();
                    }
                }
            }
        }
    }

    return max_heap;
}

std::vector<HnswIndex::internal_id_t> HnswIndex::select_neighbors_heuristic(
    const float* query, MaxHeap candidates, uint32_t M, int layer) const
{
    // A more advanced heuristic can be used (e.g. Malkov sec 4.1.2)
    // For simplicity, we fallback to selecting the closest M
    return select_neighbors_simple(candidates, M);
}

std::vector<HnswIndex::internal_id_t> HnswIndex::select_neighbors_simple(
    MaxHeap candidates, uint32_t M) const
{
    std::vector<internal_id_t> selected;
    selected.reserve(M);
    
    // Candidates is a MaxHeap. To get the closest, we need to extract all and take the end, 
    // or keep popping until it's empty and reverse.
    std::vector<internal_id_t> temp;
    while (!candidates.empty()) {
        temp.push_back(candidates.top().id);
        candidates.pop();
    }
    
    // temp is farthest to nearest.
    // Nearest are at the back. Select up to M closest.
    for (int i = static_cast<int>(temp.size()) - 1; i >= 0 && selected.size() < M; --i) {
        selected.push_back(temp[i]);
    }
    
    return selected;
}

} // namespace vektordb::index
