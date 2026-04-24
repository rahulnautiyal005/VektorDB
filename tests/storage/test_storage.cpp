/// @file test_storage.cpp
/// @brief Tests for VectorStore and MmapFile.

#include <gtest/gtest.h>
#include "vektordb/storage/vector_store.h"
#include <cstdio>
#include <vector>

using namespace vektordb;
using namespace vektordb::storage;

TEST(StorageTest, InMemoryStore) {
    VectorStore store(3, math::DistanceMetric::L2);
    EXPECT_EQ(store.dimension(), 3);
    EXPECT_EQ(store.size(), 0);

    float v1[] = {1.0f, 2.0f, 3.0f};
    vec_id_t id1 = store.add(v1);
    EXPECT_EQ(id1, 0);
    EXPECT_EQ(store.size(), 1);

    const float* stored_v1 = store.get(0);
    EXPECT_EQ(stored_v1[0], 1.0f);
    EXPECT_EQ(stored_v1[1], 2.0f);
    EXPECT_EQ(stored_v1[2], 3.0f);
}

TEST(StorageTest, SaveAndLoad) {
    const std::string file_path = "test_vectors.vkdb";

    {
        VectorStore store(2, math::DistanceMetric::Cosine);
        float v1[] = {0.5f, 0.5f};
        float v2[] = {-1.0f, 1.0f};
        store.add(v1);
        store.add(v2);
        store.save(file_path);
    }

    {
        VectorStore loaded = VectorStore::load(file_path);
        EXPECT_EQ(loaded.dimension(), 2);
        EXPECT_EQ(loaded.size(), 2);
        EXPECT_EQ(loaded.metric(), math::DistanceMetric::Cosine);

        const float* v1 = loaded.get(0);
        EXPECT_EQ(v1[0], 0.5f);
        EXPECT_EQ(v1[1], 0.5f);

        const float* v2 = loaded.get(1);
        EXPECT_EQ(v2[0], -1.0f);
        EXPECT_EQ(v2[1], 1.0f);
    }

    // Cleanup
    std::remove(file_path.c_str());
}
