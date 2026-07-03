/**
 * @file CSRBuilder.h
 * @brief CSR sparse matrix types, builder, binary I/O, memory-mapped access
 * 
 * Complete migration from original_source_backup/dataLoader.h
 */

#pragma once
#include "MatFileReader.h"
#include "io/CSRTypes.h"
#include <vector>
#include <string>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <cstdlib>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace spect {
namespace io {

// ================= CSR 验证 =================

/**
 * @brief CSR 验证级别
 */
enum class CSRValidationLevel {
    Basic,   ///< 基本校验: row_ptr[0]==0, row_ptr[rows]==nnz, 单调不减
    Strict   ///< 严格校验: 基本校验 + col_idx范围检查
};

// Default CSR validation level controlled by build option.
#if defined(SPECT_CSR_STRICT_VALIDATE) && SPECT_CSR_STRICT_VALIDATE
inline constexpr CSRValidationLevel kDefaultCSRValidationLevel = CSRValidationLevel::Strict;
#else
inline constexpr CSRValidationLevel kDefaultCSRValidationLevel = CSRValidationLevel::Basic;
#endif

inline uint64_t availablePhysicalMemoryBytes() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status))
        return static_cast<uint64_t>(status.ullAvailPhys);
    return 0;
#else
#if defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long pageSize = sysconf(_SC_PAGESIZE);
    if (pages > 0 && pageSize > 0)
        return static_cast<uint64_t>(pages) * static_cast<uint64_t>(pageSize);
#endif
    return 0;
#endif
}

inline bool shouldRefuseFullCopyCSRLoad(uint64_t fileSizeBytes,
                                        uint64_t availableMemoryBytes) {
    return availableMemoryBytes > 0 && fileSizeBytes >= availableMemoryBytes / 2;
}

inline uint64_t parseCSRByteThreshold(const char *text, uint64_t fallback) {
    if (text == nullptr || *text == '\0')
        return fallback;
    char *end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text)
        return fallback;
    uint64_t multiplier = 1;
    if (*end == 'k' || *end == 'K')
        multiplier = 1024ull;
    else if (*end == 'm' || *end == 'M')
        multiplier = 1024ull * 1024ull;
    else if (*end == 'g' || *end == 'G')
        multiplier = 1024ull * 1024ull * 1024ull;
    return static_cast<uint64_t>(value) * multiplier;
}

inline uint64_t csrMmapMinBytes() {
    return parseCSRByteThreshold(std::getenv("SPECT_CSR_MMAP_MIN_BYTES"),
                                 1ull * 1024ull * 1024ull * 1024ull);
}

inline bool shouldUseImplicitCSRMap(uint64_t fileSizeBytes,
                                    uint64_t minBytes = csrMmapMinBytes()) {
    return fileSizeBytes >= minBytes;
}

/**
 * @brief 校验 CSR 结构体数据完整性
 * 
 * @param A 要校验的 CSR 结构
 * @param level 校验级别
 * @return true 如果校验通过
 * @throws std::runtime_error 如果校验失败
 */
inline bool validateCSR(const CSR& A, CSRValidationLevel level = CSRValidationLevel::Basic) {
    if (A.rows < 0 || A.cols < 0) {
        throw std::runtime_error("CSR validation failed: negative dimensions");
    }
    if (A.row_ptr.size() != static_cast<size_t>(A.rows) + 1) {
        throw std::runtime_error("CSR validation failed: row_ptr size mismatch");
    }
    if (A.row_ptr[0] != 0) {
        throw std::runtime_error("CSR validation failed: row_ptr[0] != 0 (got " + 
                                 std::to_string(A.row_ptr[0]) + ")");
    }
    if (A.row_ptr[A.rows] != static_cast<uint32_t>(A.col_idx.size())) {
        throw std::runtime_error("CSR validation failed: row_ptr[rows] != nnz");
    }
    for (int i = 0; i < A.rows; ++i) {
        if (A.row_ptr[i] > A.row_ptr[i + 1]) {
            throw std::runtime_error("CSR validation failed: row_ptr not monotonic at row " + 
                                     std::to_string(i));
        }
    }
    if (level == CSRValidationLevel::Strict) {
        const uint32_t ncols = static_cast<uint32_t>(A.cols);
        for (size_t k = 0; k < A.col_idx.size(); ++k) {
            if (A.col_idx[k] >= ncols) {
                throw std::runtime_error("CSR validation failed: col_idx[" + 
                                         std::to_string(k) + "] = " + 
                                         std::to_string(A.col_idx[k]) + 
                                         " >= cols (" + std::to_string(ncols) + ")");
            }
        }
    }
    return true;
}

/**
 * @brief 校验 CSRView 数据完整性
 */
inline bool validateCSRView(const CSRView& view, CSRValidationLevel level = CSRValidationLevel::Basic) {
    if (!view.valid()) {
        throw std::runtime_error("CSRView is invalid (null pointers or negative dimensions)");
    }
    if (view.row_ptr[0] != 0) {
        throw std::runtime_error("CSR validation failed: row_ptr[0] != 0 (got " + 
                                 std::to_string(view.row_ptr[0]) + ")");
    }
    if (view.row_ptr[view.rows] != static_cast<uint32_t>(view.nnz)) {
        throw std::runtime_error("CSR validation failed: row_ptr[rows] != nnz");
    }
    for (int i = 0; i < view.rows; ++i) {
        if (view.row_ptr[i] > view.row_ptr[i + 1]) {
            throw std::runtime_error("CSR validation failed: row_ptr not monotonic at row " + 
                                     std::to_string(i));
        }
    }
    if (level == CSRValidationLevel::Strict) {
        const uint32_t ncols = static_cast<uint32_t>(view.cols);
        for (uint64_t k = 0; k < view.nnz; ++k) {
            if (view.col_idx[k] >= ncols) {
                throw std::runtime_error("CSR validation failed: col_idx[" + 
                                         std::to_string(k) + "] >= cols");
            }
        }
    }
    return true;
}

// ================= CSR 二进制 I/O =================

/**
 * @brief 保存 CSR 到二进制文件
 */
inline void saveCSRBinary(const CSR &A, const std::string &path) {
    CSRHeader h;
    h.rows = static_cast<uint32_t>(A.rows);
    h.cols = static_cast<uint32_t>(A.cols);
    h.nnz = A.val.size();
    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open for write: " + path);
    f.write(reinterpret_cast<const char*>(&h), sizeof(h));
    f.write(reinterpret_cast<const char*>(A.row_ptr.data()),
            A.row_ptr.size() * sizeof(uint32_t));
    f.write(reinterpret_cast<const char*>(A.col_idx.data()),
            A.col_idx.size() * sizeof(uint32_t));
    f.write(reinterpret_cast<const char*>(A.val.data()),
            A.val.size() * sizeof(float));
}

/**
 * @brief 从二进制文件加载 CSR
 */
inline CSR loadCSRBinary(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open CSR file: " + path);
    
    f.seekg(0, std::ios::end);
    const uint64_t fileSize = static_cast<uint64_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    const uint64_t availableMemory = availablePhysicalMemoryBytes();
    if (shouldRefuseFullCopyCSRLoad(fileSize, availableMemory)) {
        throw std::runtime_error(
            "CSR file is too large for full-copy loading: " + path +
            ". Use -mapCSR on for legacy single-file CSRB inputs.");
    }
    
    CSRHeader h{};
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (h.magic != 0x43535242)
        throw std::runtime_error("Invalid CSR file magic: " + path);
    if (h.version != 1)
        throw std::runtime_error("Unsupported CSR version " + std::to_string(h.version) +
                                 " in " + path);
    
    const uint64_t expectedSize = sizeof(CSRHeader) + 
                                   (static_cast<uint64_t>(h.rows) + 1) * sizeof(uint32_t) +
                                   static_cast<uint64_t>(h.nnz) * sizeof(uint32_t) +
                                   static_cast<uint64_t>(h.nnz) * sizeof(float);
    if (fileSize != expectedSize)
        throw std::runtime_error("CSR file size mismatch: " + path);
    
    CSR A;
    A.rows = static_cast<int>(h.rows);
    A.cols = static_cast<int>(h.cols);
    A.row_ptr.resize(static_cast<size_t>(h.rows) + 1);
    A.col_idx.resize(static_cast<size_t>(h.nnz));
    A.val.resize(static_cast<size_t>(h.nnz));
    f.read(reinterpret_cast<char*>(A.row_ptr.data()),
           A.row_ptr.size() * sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(A.col_idx.data()),
           A.col_idx.size() * sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(A.val.data()), A.val.size() * sizeof(float));
    
    validateCSR(A, kDefaultCSRValidationLevel);
    return A;
}

// ================= CSRMap 内存映射 =================

/**
 * @brief 内存映射的 CSR 文件视图（只读，零拷贝）
 * 
 * Windows 使用 CreateFileMapping/MapViewOfFile
 * POSIX 使用 mmap
 */
class CSRMap {
public:
    CSRMap() = default;
    ~CSRMap() { close(); }
    CSRMap(const CSRMap&) = delete;
    CSRMap& operator=(const CSRMap&) = delete;
    CSRMap(CSRMap&& o) noexcept { moveFrom(std::move(o)); }
    CSRMap& operator=(CSRMap&& o) noexcept {
        if (this != &o) {
            close();
            moveFrom(std::move(o));
        }
        return *this;
    }

    void open(const std::string& path) {
        close();
#ifdef _WIN32
        std::wstring wpath = std::filesystem::path(path).wstring();
        HANDLE hFile = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, 
                                   NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE)
            throw std::runtime_error("CreateFile failed: " + path);
        LARGE_INTEGER fsz;
        if (!GetFileSizeEx(hFile, &fsz)) {
            CloseHandle(hFile);
            throw std::runtime_error("GetFileSizeEx failed: " + path);
        }
        HANDLE hMap = CreateFileMappingW(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!hMap) {
            CloseHandle(hFile);
            throw std::runtime_error("CreateFileMapping failed: " + path);
        }
        void* base = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!base) {
            CloseHandle(hMap);
            CloseHandle(hFile);
            throw std::runtime_error("MapViewOfFile failed: " + path);
        }
        fileHandle_ = hFile;
        mapHandle_ = hMap;
        base_ = base;
        length_ = static_cast<size_t>(fsz.QuadPart);
#else
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0)
            throw std::runtime_error("open failed: " + path);
        struct stat st{};
        if (fstat(fd, &st) != 0) {
            ::close(fd);
            throw std::runtime_error("fstat failed: " + path);
        }
        void* base = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ,
                          MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) {
            ::close(fd);
            throw std::runtime_error("mmap failed: " + path);
        }
        fd_ = fd;
        base_ = base;
        length_ = static_cast<size_t>(st.st_size);
#endif
        // Parse header and set pointers
        if (length_ < sizeof(CSRHeader))
            throw std::runtime_error("CSR file too small");
        auto* h = reinterpret_cast<const CSRHeader*>(base_);
        if (h->magic != 0x43535242)
            throw std::runtime_error("Invalid CSR magic");
        if (h->version != 1)
            throw std::runtime_error("Unsupported CSR version " + std::to_string(h->version) +
                                     " in " + path);
        view_.rows = static_cast<int>(h->rows);
        view_.cols = static_cast<int>(h->cols);
        view_.nnz = h->nnz;
        size_t off = sizeof(CSRHeader);
        size_t szRow = (static_cast<size_t>(h->rows) + 1) * sizeof(uint32_t);
        size_t szCol = static_cast<size_t>(h->nnz) * sizeof(uint32_t);
        size_t szVal = static_cast<size_t>(h->nnz) * sizeof(float);
        if (off + szRow + szCol + szVal > length_)
            throw std::runtime_error("CSR file truncated");
        view_.row_ptr = reinterpret_cast<const uint32_t*>(
            static_cast<const uint8_t*>(base_) + off);
        off += szRow;
        view_.col_idx = reinterpret_cast<const uint32_t*>(
            static_cast<const uint8_t*>(base_) + off);
        off += szCol;
        view_.val = reinterpret_cast<const float*>(
            static_cast<const uint8_t*>(base_) + off);
        
        validateCSRView(view_, kDefaultCSRValidationLevel);
    }

    void close() noexcept {
#ifdef _WIN32
        if (base_) UnmapViewOfFile(base_);
        if (mapHandle_) CloseHandle(mapHandle_);
        if (fileHandle_) CloseHandle(fileHandle_);
        base_ = nullptr;
        mapHandle_ = nullptr;
        fileHandle_ = nullptr;
        length_ = 0;
#else
        if (base_ && length_) munmap(base_, length_);
        if (fd_ >= 0) ::close(fd_);
        base_ = nullptr;
        length_ = 0;
        fd_ = -1;
#endif
        view_ = CSRView{};
    }

    const CSRView& view() const { return view_; }

private:
    void moveFrom(CSRMap&& o) noexcept {
        view_ = o.view_;
        base_ = o.base_;
        o.base_ = nullptr;
        length_ = o.length_;
        o.length_ = 0;
#ifdef _WIN32
        fileHandle_ = o.fileHandle_;
        o.fileHandle_ = nullptr;
        mapHandle_ = o.mapHandle_;
        o.mapHandle_ = nullptr;
#else
        fd_ = o.fd_;
        o.fd_ = -1;
#endif
    }

    CSRView view_{};
    void* base_ = nullptr;
    size_t length_ = 0;
#ifdef _WIN32
    HANDLE fileHandle_ = nullptr;
    HANDLE mapHandle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

// ================= CSR 构建函数 =================

/**
 * @brief Build CSR matrix from MAT file (memory loading mode)
 */
CSR buildCSR_from_mat(
    const MatArray &W_ind,
    const MatArray &W_dat,
    size_t Nvox,
    double prune_eps = 1e-12,
    const std::vector<uint8_t> *mask = nullptr
);

/**
 * @brief 从 MAT 文件流式构建 CSR 矩阵（大文件模式）
 */
CSR buildCSR_from_mat_stream(
    const std::string &idxPath, const std::string &idxVar,
    const std::string &wgtPath, const std::string &wgtVar,
    size_t Nvox, double prune_eps = 1e-12,
    const std::vector<uint8_t> *mask = nullptr
);

} // namespace io
} // namespace spect
