/// @file vector_store.cpp
/// @brief Vector storage implementation with .vkdb binary format.

#include "vektordb/storage/vector_store.h"
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace vektordb::storage {

VectorStore::VectorStore(dim_t dimension, math::DistanceMetric metric)
    : dimension_(dimension), metric_(metric)
{
    if (dimension == 0) {
        throw std::runtime_error("VectorStore: dimension must be > 0");
    }
}

VectorStore VectorStore::load(const std::string& path) {
    auto mmap = std::make_unique<MmapFile>(path, MmapMode::ReadOnly);

    if (mmap->size() < sizeof(VkdbHeader)) {
        throw std::runtime_error("VectorStore: file too small for header");
    }

    // Read and validate header
    const auto* header = reinterpret_cast<const VkdbHeader*>(mmap->data());

    if (std::memcmp(header->magic, "VKDB", 4) != 0) {
        throw std::runtime_error("VectorStore: invalid magic bytes (not a .vkdb file)");
    }
    if (header->version != 1) {
        throw std::runtime_error("VectorStore: unsupported version: " +
                                 std::to_string(header->version));
    }
    if (header->dimension == 0) {
        throw std::runtime_error("VectorStore: dimension is 0");
    }

    // Validate file size
    std::size_t expected_size = sizeof(VkdbHeader) +
        static_cast<std::size_t>(header->num_vectors) * header->dimension * sizeof(float);

    if (mmap->size() < expected_size) {
        throw std::runtime_error("VectorStore: file truncated");
    }

    // Create store
    auto metric = static_cast<math::DistanceMetric>(header->metric);
    VectorStore store(header->dimension, metric);
    store.num_vectors_ = header->num_vectors;
    store.data_ptr_ = reinterpret_cast<const float*>(mmap->data() + sizeof(VkdbHeader));
    store.mmap_file_ = std::move(mmap);

    return store;
}

void VectorStore::save(const std::string& path) const {
    std::size_t data_size = static_cast<std::size_t>(num_vectors_) * dimension_ * sizeof(float);
    std::size_t total_size = sizeof(VkdbHeader) + data_size;

    // Write via mmap for efficiency with large files
    MmapFile out(path, total_size, MmapMode::ReadWrite);

    // Write header
    auto* header = reinterpret_cast<VkdbHeader*>(out.data());
    std::memcpy(header->magic, "VKDB", 4);
    header->version     = 1;
    header->dimension   = dimension_;
    header->metric      = static_cast<uint32_t>(metric_);
    header->num_vectors = num_vectors_;
    header->reserved    = 0;

    // Write vector data
    if (data_size > 0 && data_ptr_) {
        std::memcpy(out.data() + sizeof(VkdbHeader), data_ptr_, data_size);
    }

    out.flush();
}

vec_id_t VectorStore::add(const float* vec) {
    if (mmap_file_) {
        throw std::runtime_error("VectorStore: cannot add to memory-mapped store (read-only)");
    }

    vec_id_t id = num_vectors_;

    // Append vector data
    data_.insert(data_.end(), vec, vec + dimension_);
    num_vectors_++;
    data_ptr_ = data_.data();

    return id;
}

void VectorStore::add_batch(const float* vecs, uint64_t count, vec_id_t* out_ids) {
    if (mmap_file_) {
        throw std::runtime_error("VectorStore: cannot add to memory-mapped store (read-only)");
    }

    vec_id_t start_id = num_vectors_;

    // Reserve space for efficiency
    std::size_t new_elements = static_cast<std::size_t>(count) * dimension_;
    data_.reserve(data_.size() + new_elements);
    data_.insert(data_.end(), vecs, vecs + new_elements);
    num_vectors_ += count;
    data_ptr_ = data_.data();

    // Fill output IDs
    if (out_ids) {
        for (uint64_t i = 0; i < count; ++i) {
            out_ids[i] = start_id + i;
        }
    }
}

const float* VectorStore::get(vec_id_t id) const {
    if (id >= num_vectors_) {
        throw std::out_of_range("VectorStore: vector ID out of range: " +
                                std::to_string(id));
    }
    return data_ptr_ + static_cast<std::size_t>(id) * dimension_;
}

} // namespace vektordb::storage
