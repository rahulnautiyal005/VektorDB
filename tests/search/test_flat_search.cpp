/// @file test_flat_search.cpp
/// @brief Tests for brute-force flat search.

#include <gtest/gtest.h>
#include "vektordb/search/flat_search.h"

using namespace vektordb;
using namespace vektordb::storage;
using namespace vektordb::search;

TEST(FlatSearchTest, SearchL2) {
    VectorStore store(2, math::DistanceMetric::L2);
    
    // Dataset
    std::vector<std::vector<float>> data = {
        {0.0f, 0.0f}, // id 0
        {1.0f, 1.0f}, // id 1
        {2.0f, 2.0f}, // id 2
        {0.1f, 0.1f}  // id 3
    };

    for (const auto& v : data) {
        store.add(v.data());
    }

    // Query close to {0, 0}
    float query[] = {0.05f, 0.05f};
    auto results = flat_search(store, query, 2);

    ASSERT_EQ(results.size(), 2);
    
    // Closest should be id 3 ({0.1, 0.1}) or id 0 ({0, 0})
    EXPECT_EQ(results[0].id, 3);
    EXPECT_EQ(results[1].id, 0);
}
