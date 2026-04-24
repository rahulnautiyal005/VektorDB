/// @file main.cpp
/// @brief CLI Demo program for VektorDB.

#include "vektordb/core/vektor_db.h"
#include "vektordb/core/platform.h"

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

using namespace vektordb;

int main() {
    std::cout << "🌌 VektorDB Demo 🌌\\n";
    std::cout << "==================\\n";
    
    vektordb::platform::print_features();
    
    if (VektorDB::is_simd_enabled()) {
        std::cout << "⚡ AVX2 Engine Active ⚡\\n\\n";
    } else {
        std::cout << "⚠️ Running in Scalar Mode ⚠️\\n\\n";
    }

    const dim_t dim = 128;
    const uint32_t num_vectors = 10000;
    
    std::cout << "Initializing database...\\n";
    std::cout << "  Dimension: " << dim << "\\n";
    std::cout << "  Metric:    L2 Distance\\n";
    
    index::HnswConfig config;
    config.M = 16;
    config.ef_construction = 100;
    
    VektorDB db(dim, math::DistanceMetric::L2, config);

    std::cout << "\\nGenerating " << num_vectors << " random vectors...\\n";
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<std::vector<float>> dataset(num_vectors, std::vector<float>(dim));
    for (uint32_t i = 0; i < num_vectors; ++i) {
        for (dim_t d = 0; d < dim; ++d) {
            dataset[i][d] = dist(rng);
        }
    }

    std::cout << "Building HNSW index...\\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    for (uint32_t i = 0; i < num_vectors; ++i) {
        db.insert(dataset[i].data(), i);
        if ((i + 1) % 2500 == 0) {
            std::cout << "  Inserted " << (i + 1) << " vectors...\\n";
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Index built in " << elapsed << " ms.\\n\\n";

    // Query
    std::vector<float> query(dim);
    for (dim_t d = 0; d < dim; ++d) query[d] = dist(rng);
    
    const uint32_t k = 5;
    
    std::cout << "Searching for top-" << k << " nearest neighbors...\\n";
    
    start = std::chrono::high_resolution_clock::now();
    // Use ef_search = 50 for high recall
    auto results_hnsw = db.search(query.data(), k, 50);
    end = std::chrono::high_resolution_clock::now();
    
    auto search_time_hnsw = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    auto results_exact = db.search_exact(query.data(), k);
    end = std::chrono::high_resolution_clock::now();
    
    auto search_time_exact = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "\\n🔍 HNSW Search (" << search_time_hnsw << " us)\\n";
    for (const auto& res : results_hnsw) {
        std::cout << "  ID: " << std::setw(5) << res.id 
                  << " | Distance: " << std::fixed << std::setprecision(4) << res.distance << "\\n";
    }

    std::cout << "\\n🎯 Exact Search (" << search_time_exact << " us)\\n";
    for (const auto& res : results_exact) {
        std::cout << "  ID: " << std::setw(5) << res.id 
                  << " | Distance: " << std::fixed << std::setprecision(4) << res.distance << "\\n";
    }
    
    std::cout << "\\nRecall Check: ";
    uint32_t matches = 0;
    for (size_t i = 0; i < k; ++i) {
        for (size_t j = 0; j < k; ++j) {
            if (results_hnsw[i].id == results_exact[j].id) {
                matches++;
                break;
            }
        }
    }
    
    std::cout << matches << "/" << k << " (" << (matches * 100.0 / k) << "%)\\n";
    std::cout << "\\nDone!\\n";

    return 0;
}
