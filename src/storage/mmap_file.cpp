/// @file mmap_file.cpp
/// @brief Memory-mapped file implementation.

#include "vektordb/storage/mmap_file.h"

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
#endif

#include <cstring>
#include <stdexcept>

namespace vektordb::storage {

// ============================================================================
// Windows Implementation
// ============================================================================
#ifdef _WIN32

MmapFile::MmapFile(const std::string& path, MmapMode mode)
    : mode_(mode)
{
    DWORD access = (mode == MmapMode::ReadOnly) ? GENERIC_READ : (GENERIC_READ | GENERIC_WRITE);
    DWORD share  = FILE_SHARE_READ;
    DWORD create = OPEN_EXISTING;

    file_handle_ = CreateFileA(
        path.c_str(), access, share, nullptr, create,
        FILE_ATTRIBUTE_NORMAL, nullptr
    );

    if (file_handle_ == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("MmapFile: Cannot open file: " + path);
    }

    // Get file size
    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(file_handle_, &file_size)) {
        CloseHandle(file_handle_);
        file_handle_ = nullptr;
        throw std::runtime_error("MmapFile: Cannot get file size: " + path);
    }
    size_ = static_cast<std::size_t>(file_size.QuadPart);

    if (size_ == 0) {
        CloseHandle(file_handle_);
        file_handle_ = nullptr;
        throw std::runtime_error("MmapFile: File is empty: " + path);
    }

    // Create file mapping
    DWORD protect = (mode == MmapMode::ReadOnly) ? PAGE_READONLY : PAGE_READWRITE;
    mapping_handle_ = CreateFileMappingA(file_handle_, nullptr, protect, 0, 0, nullptr);

    if (!mapping_handle_) {
        CloseHandle(file_handle_);
        file_handle_ = nullptr;
        throw std::runtime_error("MmapFile: Cannot create file mapping: " + path);
    }

    // Map view
    DWORD map_access = (mode == MmapMode::ReadOnly) ? FILE_MAP_READ : FILE_MAP_ALL_ACCESS;
    data_ = static_cast<uint8_t*>(MapViewOfFile(mapping_handle_, map_access, 0, 0, 0));

    if (!data_) {
        CloseHandle(mapping_handle_);
        CloseHandle(file_handle_);
        mapping_handle_ = nullptr;
        file_handle_ = nullptr;
        throw std::runtime_error("MmapFile: Cannot map file: " + path);
    }
}

MmapFile::MmapFile(const std::string& path, std::size_t size, MmapMode mode)
    : size_(size), mode_(mode)
{
    if (mode != MmapMode::ReadWrite) {
        throw std::runtime_error("MmapFile: Creating new file requires ReadWrite mode");
    }

    DWORD access = GENERIC_READ | GENERIC_WRITE;
    file_handle_ = CreateFileA(
        path.c_str(), access, 0, nullptr, CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL, nullptr
    );

    if (file_handle_ == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("MmapFile: Cannot create file: " + path);
    }

    // Create mapping with specified size
    DWORD size_high = static_cast<DWORD>(size >> 32);
    DWORD size_low  = static_cast<DWORD>(size & 0xFFFFFFFF);
    mapping_handle_ = CreateFileMappingA(
        file_handle_, nullptr, PAGE_READWRITE, size_high, size_low, nullptr
    );

    if (!mapping_handle_) {
        CloseHandle(file_handle_);
        file_handle_ = nullptr;
        throw std::runtime_error("MmapFile: Cannot create file mapping");
    }

    data_ = static_cast<uint8_t*>(
        MapViewOfFile(mapping_handle_, FILE_MAP_ALL_ACCESS, 0, 0, 0)
    );

    if (!data_) {
        CloseHandle(mapping_handle_);
        CloseHandle(file_handle_);
        mapping_handle_ = nullptr;
        file_handle_ = nullptr;
        throw std::runtime_error("MmapFile: Cannot map new file");
    }

    // Zero-fill
    std::memset(data_, 0, size_);
}

void MmapFile::unmap() {
    if (data_) {
        UnmapViewOfFile(data_);
        data_ = nullptr;
    }
    if (mapping_handle_) {
        CloseHandle(mapping_handle_);
        mapping_handle_ = nullptr;
    }
    if (file_handle_) {
        CloseHandle(file_handle_);
        file_handle_ = nullptr;
    }
    size_ = 0;
}

void MmapFile::flush() {
    if (data_ && mode_ == MmapMode::ReadWrite) {
        FlushViewOfFile(data_, size_);
    }
}

#else
// ============================================================================
// POSIX Implementation
// ============================================================================

MmapFile::MmapFile(const std::string& path, MmapMode mode)
    : mode_(mode)
{
    int flags = (mode == MmapMode::ReadOnly) ? O_RDONLY : O_RDWR;
    fd_ = open(path.c_str(), flags);
    if (fd_ < 0) {
        throw std::runtime_error("MmapFile: Cannot open file: " + path);
    }

    struct stat st;
    if (fstat(fd_, &st) < 0) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("MmapFile: Cannot stat file: " + path);
    }
    size_ = static_cast<std::size_t>(st.st_size);

    int prot = (mode == MmapMode::ReadOnly) ? PROT_READ : (PROT_READ | PROT_WRITE);
    data_ = static_cast<uint8_t*>(mmap(nullptr, size_, prot, MAP_SHARED, fd_, 0));

    if (data_ == MAP_FAILED) {
        data_ = nullptr;
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("MmapFile: Cannot mmap file: " + path);
    }

    // Advise kernel about access pattern
    madvise(data_, size_, MADV_RANDOM);
}

MmapFile::MmapFile(const std::string& path, std::size_t size, MmapMode mode)
    : size_(size), mode_(mode)
{
    fd_ = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd_ < 0) {
        throw std::runtime_error("MmapFile: Cannot create file: " + path);
    }

    if (ftruncate(fd_, size) < 0) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("MmapFile: Cannot resize file");
    }

    data_ = static_cast<uint8_t*>(
        mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0)
    );

    if (data_ == MAP_FAILED) {
        data_ = nullptr;
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("MmapFile: Cannot mmap new file");
    }
}

void MmapFile::unmap() {
    if (data_) {
        munmap(data_, size_);
        data_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    size_ = 0;
}

void MmapFile::flush() {
    if (data_ && mode_ == MmapMode::ReadWrite) {
        msync(data_, size_, MS_SYNC);
    }
}

#endif

// ============================================================================
// Common Implementation
// ============================================================================

MmapFile::~MmapFile() {
    unmap();
}

MmapFile::MmapFile(MmapFile&& other) noexcept
    : data_(other.data_), size_(other.size_), mode_(other.mode_)
{
#ifdef _WIN32
    file_handle_ = other.file_handle_;
    mapping_handle_ = other.mapping_handle_;
    other.file_handle_ = nullptr;
    other.mapping_handle_ = nullptr;
#else
    fd_ = other.fd_;
    other.fd_ = -1;
#endif
    other.data_ = nullptr;
    other.size_ = 0;
}

MmapFile& MmapFile::operator=(MmapFile&& other) noexcept {
    if (this != &other) {
        unmap();
        data_ = other.data_;
        size_ = other.size_;
        mode_ = other.mode_;
#ifdef _WIN32
        file_handle_ = other.file_handle_;
        mapping_handle_ = other.mapping_handle_;
        other.file_handle_ = nullptr;
        other.mapping_handle_ = nullptr;
#else
        fd_ = other.fd_;
        other.fd_ = -1;
#endif
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void MmapFile::close() {
    unmap();
}

} // namespace vektordb::storage
