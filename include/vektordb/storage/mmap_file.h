#pragma once

/// @file mmap_file.h
/// @brief Cross-platform memory-mapped file abstraction.
///
/// On Windows: Uses CreateFileMapping + MapViewOfFile
/// On Linux:   Uses mmap + madvise (POSIX)
///
/// Provides zero-copy read access to files on disk, enabling
/// vector search on datasets larger than available RAM.

#include <cstddef>
#include <cstdint>
#include <string>
#include <stdexcept>

namespace vektordb::storage {

/// Access mode for memory-mapped files.
enum class MmapMode {
    ReadOnly,   ///< Read-only mapping (most common for search)
    ReadWrite   ///< Read-write mapping (for building index)
};

/// RAII wrapper around a memory-mapped file.
///
/// Usage:
/// ```cpp
/// MmapFile file("vectors.vkdb", MmapMode::ReadOnly);
/// const float* data = reinterpret_cast<const float*>(file.data() + header_size);
/// ```
class MmapFile {
public:
    /// Map an existing file into memory.
    /// @param path  Path to the file
    /// @param mode  Access mode (ReadOnly or ReadWrite)
    /// @throws std::runtime_error if file cannot be opened or mapped
    explicit MmapFile(const std::string& path, MmapMode mode = MmapMode::ReadOnly);

    /// Create and map a new file of given size.
    /// @param path  Path to create
    /// @param size  File size in bytes
    /// @param mode  Must be ReadWrite
    /// @throws std::runtime_error if file cannot be created
    MmapFile(const std::string& path, std::size_t size, MmapMode mode);

    /// Unmaps file and closes handles.
    ~MmapFile();

    // Non-copyable, movable
    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(const MmapFile&) = delete;
    MmapFile(MmapFile&& other) noexcept;
    MmapFile& operator=(MmapFile&& other) noexcept;

    /// @return Pointer to the mapped memory region.
    uint8_t* data() noexcept { return data_; }
    const uint8_t* data() const noexcept { return data_; }

    /// @return Size of the mapped region in bytes.
    std::size_t size() const noexcept { return size_; }

    /// @return true if a file is currently mapped.
    bool is_open() const noexcept { return data_ != nullptr; }

    /// Flush changes to disk (for ReadWrite mode).
    void flush();

    /// Unmap and close the file.
    void close();

private:
    uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
    MmapMode mode_ = MmapMode::ReadOnly;

#ifdef _WIN32
    void* file_handle_ = nullptr;      // HANDLE
    void* mapping_handle_ = nullptr;   // HANDLE
#else
    int fd_ = -1;
#endif

    void unmap();
};

} // namespace vektordb::storage
