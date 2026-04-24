# 🌌 VektorDB

**Hardware-Accelerated, Out-of-Core Vector Search Engine for AI Workloads**

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue?style=flat-square&logo=cplusplus)
![SIMD](https://img.shields.io/badge/SIMD-AVX2-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

> 💡 **Engineered for sub-millisecond similarity search across billion-scale vector datasets, bypassing RAM limits via memory-mapped I/O.**

---

## 📌 Architecture Overview

**VektorDB** is a native C++ vector database designed to act as the retrieval backend for Large Language Models (RAG).

It implements the **Hierarchical Navigable Small World (HNSW)** algorithm for Approximate Nearest Neighbor (ANN) search, optimized with CPU-specific SIMD instructions for lightning-fast distance calculations.

## ⚙️ Tech Stack

| Layer             | Technology                                  |
|-------------------|---------------------------------------------|
| Core Language     | C++20 (Templates, Concepts, Smart Pointers) |
| Math Acceleration | Intel Intrinsics (AVX2)                     |
| Storage I/O       | Memory-mapped I/O                           |
| Concurrency       | std::shared_mutex, atomic operations        |
| RPC / Networking  | gRPC, Protocol Buffers                      |
| Build / Testing   | CMake, Google Test, Google Benchmark        |

## 🚀 Quick Start

### Prerequisites

- **GCC 13+** or **MSVC 2022** with C++20 support
- **CMake 3.20+**
- **Git** (for FetchContent dependencies)
- CPU with **AVX2** support (Intel Haswell+ / AMD Zen+)

### Build

```bash
# Configure
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -G "MinGW Makefiles"

# Build
cmake --build build --config Release

# Run tests
cd build && ctest --output-on-failure

# Run benchmarks
./build/vektordb_bench.exe
```

## 📊 Phase 1: Math Engine

The foundation — SIMD-accelerated distance computations:

- **L2 (Euclidean) Distance**: Scalar baseline + AVX2 optimized
- **Cosine Similarity**: Scalar baseline + AVX2 optimized
- **Runtime dispatch**: Auto-detects CPU features, selects fastest path

## 👨‍💻 Author

**Rahul Nautiyal**

- GitHub: [rahulnautiyal005](https://github.com/rahulnautiyal005)
- LinkedIn: [Rahul Nautiyal](https://www.linkedin.com/in/rahul-nautiyal-b749b62a5)

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
