# 🌌 VektorDB

**Hardware-Accelerated, Out-of-Core Vector Search Engine for AI Workloads**

*C++20 | SIMD (AVX2/FMA) | HNSW Graph | Memory-Mapped I/O*

---

**VektorDB** is a native C++ vector database engineered from scratch to be the retrieval backend for Large Language Models (RAG pipelines). It easily handles large-scale semantic embeddings and computes semantic similarity at blazing speed.

## 🧠 Core Engineering Achievements

1. **⚡ SIMD-Accelerated Math Engine**
   - Hand-written **AVX2 / FMA intrinsics** for Cosine Similarity, L2 Euclidean Distance, and Dot Product.
   - Run-time CPU dispatching: gracefully scales back to Scalar logic if AVX is missing.
   - Up to **10x faster** than standard C++ calculations!

2. **💾 Zero-Copy Out-of-Core Storage**
   - Bypasses RAM constraints using OS-level **memory-mapped files** (`mmap` on POSIX, `MapViewOfFile` on Windows).
   - Custom `.vkdb` binary format explicitly tuned for 32-byte SIMD alignments.

3. **🕸️ Concurrent HNSW Graph Index**
   - Fully implemented the benchmark Hierarchical Navigable Small World (HNSW) algorithm from scratch based on Malkov's paper.
   - Incredible search speed: >10,000 embedded nodes traverse in **~344 microseconds**.
   - Thread-safe inserts utilizing node-level locking and atomic graph entry point shifting.

---

## 🛠️ Building the Project

This project uses modern CMake (FetchContent) and requires a C++20 compatible compiler (e.g., GCC 11+, MSVC 2022+).

### 1. Build
```sh
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j 8
```

### 2. Verify Run Tests
(Powered by Google Test)
```sh
./build/tests/vektordb_tests
# All 36 Tests pass!
```

### 3. Run the High-Performance Benchmarks
(Powered by Google Benchmark)
```sh
./build/benchmarks/vektordb_bench
```

### 4. Run the HNSW Graph Demo
```sh
./build/vektordb_demo
```

---

## 📊 Example Performance (1536-Dimensional Embeddings)
**(Hardware: AMD Ryzen 5, 2.3 GHz)**

| Operation | Scalar Backup | AVX2 SIMD | Speedup |
|-----------|---------------|-----------|---------|
| L2 Distance | 751 ns | 87 ns | **8.5x** |
| Cosine Sim | 997 ns | 135 ns | **7.4x** |
| Dot Product | 755 ns | 82 ns | **9.1x** |

---

## 🚀 Architecture Diagram

```text
VektorDB API
 ├── HNSW Index (Similarity Search - multi-layer graphing)
 └── Vector Store (Data Engine)
      ├── Runtime SIMD Dispatcher (AVX2/Scalar)
      └── Memory-Mapped OS Files (.vkdb format)
```

---

## 💻 Quick Start (C++ API)

Integrating `VektorDB` into your own application is straightforward. Here is a minimal example:

```cpp
#include "vektordb/core/vektor_db.h"
#include <iostream>

using namespace vektordb;

int main() {
    // 1. Configure the HNSW Index parameters
    index::HnswConfig config;
    config.M = 16;
    config.ef_construction = 200;

    // 2. Initialize the Database (Dimension: 128, Metric: L2 Distance)
    VektorDB db(128, math::DistanceMetric::L2, config);

    // 3. Insert vectors
    float vec1[128] = {0.5f, 0.1f, /* ... */};
    db.insert(vec1, 1001); // 1001 is the user-defined external ID

    // 4. Query the database
    float query[128] = {0.4f, 0.2f, /* ... */};
    uint32_t k = 5; // Get top 5 results
    
    auto results = db.search(query, k, 50 /* ef_search */);

    // 5. Process results
    for (const auto& res : results) {
        std::cout << "Matched ID: " << res.id << " (Distance: " << res.distance << ")\n";
    }

    return 0;
}
```

---

## 🗺️ Roadmap & Future Work

While the core math and HNSW engine are implemented, Phase 2 of VektorDB development will focus on distribution and accessibility:

- [ ] **gRPC / Protocol Buffers Server Layer**: Turn VektorDB into a standalone microservice.
- [ ] **Python Bindings (`pybind11`)**: Allow data scientists to use the database via a `pip install`.
- [ ] **IVF-FLAT / IVF-PQ Indexing**: Support inverted file indexes and product quantization for extreme memory compression.
- [ ] **AVX-512 Support**: Further optimize routines for the latest server hardware.

---

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for full details. 

Contributions, issues, and feature requests are highly welcome!
