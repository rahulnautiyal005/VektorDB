/// @file vektor_db.cpp
/// @brief Implementation of the main VektorDB API.

#include "vektordb/core/vektor_db.h"
#include "vektordb/core/platform.h"

namespace vektordb {

VektorDB::VektorDB(dim_t dimension, math::DistanceMetric metric, const index::HnswConfig& config)
    : dimension_(dimension),
      store_(dimension, metric),
      index_(dimension, metric, config)
{
}

void VektorDB::insert(const float* vec, vec_id_t id) {
    // 1. Add to flat storage
    store_.add(vec); // In a highly concurrent system, this needs its own lock
    
    // 2. Add to HNSW index
    index_.insert(vec, id);
}

std::vector<search::SearchResult> VektorDB::search(
    const float* query, uint32_t k, uint32_t ef_search) const
{
    return index_.search(query, k, ef_search);
}

std::vector<search::SearchResult> VektorDB::search_exact(
    const float* query, uint32_t k) const
{
    return search::flat_search(store_, query, k);
}

bool VektorDB::is_simd_enabled() noexcept {
#ifdef VEKTORDB_AVX2_ENABLED
    return platform::supports_avx2() && platform::supports_fma();
#else
    return false;
#endif
}

} // namespace vektordb
