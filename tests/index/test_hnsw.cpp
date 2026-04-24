/// @file test_hnsw.cpp
/// @brief Unit tests for the HNSW index.

#include <gtest/gtest.h>
#include "vektordb/index/hnsw.h"
#include <random>

using namespace vektordb;
using namespace vektordb::index;

TEST(HnswTest, InsertionAndSearchL2) {
    HnswConfig config;
    config.M = 16;
    config.ef_construction = 50;
    config.seed = 42;

    HnswIndex index(3, math::DistanceMetric::L2, config);

    std::vector<std::vector<float>> data = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {1.1f, 2.1f, 3.1f}, // Close to 0
        {9.0f, 9.0f, 9.0f}
    };

    for (size_t i = 0; i < data.size(); ++i) {
        index.insert(data[i].data(), static_cast<vec_id_t>(i));
    }

    EXPECT_EQ(index.size(), 4);

    // Query close to {1,2,3}
    float query[] = {1.05f, 2.05f, 3.05f};
    auto results = index.search(query, 2, 10);

    ASSERT_EQ(results.size(), 2);
    // Closest should be 0 or 2
    EXPECT_TRUE(results[0].id == 0 || results[0].id == 2);
    EXPECT_LT(results[0].distance, 1.0f);
}

TEST(HnswTest, HighDimensionalStress) {
    HnswConfig config;
    config.M = 8;
    config.ef_construction = 40;
    
    const dim_t dim = 128;
    const uint32_t num_vectors = 500;
    HnswIndex index(dim, math::DistanceMetric::Cosine, config);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::vector<float>> dataset(num_vectors, std::vector<float>(dim));
    for (uint32_t i = 0; i < num_vectors; ++i) {
        for (dim_t d = 0; d < dim; ++d) {
            dataset[i][d] = dist(rng);
        }
        index.insert(dataset[i].data(), i);
    }

    EXPECT_EQ(index.size(), num_vectors);

    // Search query
    std::vector<float> query(dim);
    for (dim_t d = 0; d < dim; ++d) query[d] = dist(rng);

    auto results = index.search(query.data(), 10, 50);
    EXPECT_EQ(results.size(), 10);
    
    // Distances should be sorted
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_GE(results[i].distance, results[i-1].distance);
    }
}
