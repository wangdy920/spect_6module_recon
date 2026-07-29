#include "io/CompactChunkCSR.h"

#include "cuda/CudaLaunchConfig.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace spect::io {
namespace {

constexpr uint32_t kChunkCSRMagic = 0x42525343u;
constexpr uint32_t kChunkCSRVersion1 = 1u;
constexpr uint32_t kChunkCSRVersion2 = 2u;
constexpr uint64_t kChunkHeaderBytes = 28ull;

enum class ChunkColIdxValidationMode { Off, Sample, Strict };

template <typename T>
void readExact(std::istream &in, T *dst, size_t count,
               const std::string &label) {
  const size_t bytes = sizeof(T) * count;
  if (bytes == 0)
    return;
  in.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(bytes));
  if (!in) {
    throw std::runtime_error("Failed to read " + label);
  }
}

template <typename T>
void readScalar(std::istream &in, T &dst, const std::string &label) {
  readExact(in, &dst, 1, label);
}

std::string lowerString(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return text;
}

bool startsWith(const std::string &text, const std::string &prefix) {
  return text.rfind(prefix, 0) == 0;
}

bool endsWith(const std::string &text, const std::string &suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool isChunkCSRFile(const std::filesystem::path &path) {
  const std::string name = lowerString(path.filename().string());
  return startsWith(name, "chunk_") && endsWith(name, ".csrb");
}

uint64_t fileSize(const std::filesystem::path &path) {
  const auto size = std::filesystem::file_size(path);
  return static_cast<uint64_t>(size);
}

uint64_t parseByteBudget(const char *text, uint64_t fallback);

uint64_t compactChunkMappedPayloadThresholdBytes() {
  constexpr uint64_t kDefaultThreshold =
      std::numeric_limits<uint64_t>::max();
  return parseByteBudget(
      std::getenv("SPECT_COMPACT_CHUNK_MMAP_MIN_BYTES"), kDefaultThreshold);
}

bool shouldMapChunkPayload(const CompactChunkCSRInfo &info) {
  return info.fileSizeBytes >= compactChunkMappedPayloadThresholdBytes();
}

ChunkColIdxValidationMode chunkColIdxValidationMode() {
  const char *raw = std::getenv("SPECT_CHUNK_VALIDATE");
  if (raw == nullptr || raw[0] == '\0')
    return ChunkColIdxValidationMode::Sample;
  const std::string value = lowerString(raw);
  if (value == "off" || value == "0" || value == "false")
    return ChunkColIdxValidationMode::Off;
  if (value == "strict" || value == "1" || value == "true" ||
      value == "full")
    return ChunkColIdxValidationMode::Strict;
  return ChunkColIdxValidationMode::Sample;
}

void warnChunkValidationOffOnce() {
  static bool warned = false;
  if (!warned) {
    std::cerr << "[WARN] SPECT_CHUNK_VALIDATE=off disables compact chunk "
                 "col_idx range validation; user assumes matrix integrity.\n";
    warned = true;
  }
}

class ReadOnlyChunkPayloadMap {
public:
  ReadOnlyChunkPayloadMap() = default;
  ~ReadOnlyChunkPayloadMap() { close(); }
  ReadOnlyChunkPayloadMap(const ReadOnlyChunkPayloadMap &) = delete;
  ReadOnlyChunkPayloadMap &operator=(const ReadOnlyChunkPayloadMap &) = delete;
  ReadOnlyChunkPayloadMap(ReadOnlyChunkPayloadMap &&other) noexcept {
    moveFrom(std::move(other));
  }
  ReadOnlyChunkPayloadMap &operator=(ReadOnlyChunkPayloadMap &&other) noexcept {
    if (this != &other) {
      close();
      moveFrom(std::move(other));
    }
    return *this;
  }

  void open(const std::filesystem::path &path) {
    close();
#ifdef _WIN32
    const std::wstring wpath = path.wstring();
    HANDLE file =
        CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
      throw std::runtime_error("CreateFile failed: " + path.string());
    }
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(file, &size)) {
      CloseHandle(file);
      throw std::runtime_error("GetFileSizeEx failed: " + path.string());
    }
    HANDLE mapping =
        CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping) {
      CloseHandle(file);
      throw std::runtime_error("CreateFileMapping failed: " + path.string());
    }
    void *base = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (!base) {
      CloseHandle(mapping);
      CloseHandle(file);
      throw std::runtime_error("MapViewOfFile failed: " + path.string());
    }
    fileHandle_ = file;
    mappingHandle_ = mapping;
    base_ = base;
    length_ = static_cast<size_t>(size.QuadPart);
#else
    int fd = ::open(path.string().c_str(), O_RDONLY);
    if (fd < 0) {
      throw std::runtime_error("open failed: " + path.string());
    }
    struct stat st {};
    if (fstat(fd, &st) != 0) {
      ::close(fd);
      throw std::runtime_error("fstat failed: " + path.string());
    }
    void *base =
        mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_SHARED,
             fd, 0);
    if (base == MAP_FAILED) {
      ::close(fd);
      throw std::runtime_error("mmap failed: " + path.string());
    }
    fd_ = fd;
    base_ = base;
    length_ = static_cast<size_t>(st.st_size);
#endif
  }

  const std::byte *bytes() const {
    return reinterpret_cast<const std::byte *>(base_);
  }

  size_t length() const { return length_; }

private:
  void close() {
#ifdef _WIN32
    if (base_) {
      UnmapViewOfFile(base_);
      base_ = nullptr;
    }
    if (mappingHandle_) {
      CloseHandle(mappingHandle_);
      mappingHandle_ = nullptr;
    }
    if (fileHandle_ != INVALID_HANDLE_VALUE) {
      CloseHandle(fileHandle_);
      fileHandle_ = INVALID_HANDLE_VALUE;
    }
#else
    if (base_) {
      munmap(base_, length_);
      base_ = nullptr;
    }
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
#endif
    length_ = 0;
  }

  void moveFrom(ReadOnlyChunkPayloadMap &&other) {
    base_ = other.base_;
    length_ = other.length_;
    other.base_ = nullptr;
    other.length_ = 0;
#ifdef _WIN32
    fileHandle_ = other.fileHandle_;
    mappingHandle_ = other.mappingHandle_;
    other.fileHandle_ = INVALID_HANDLE_VALUE;
    other.mappingHandle_ = nullptr;
#else
    fd_ = other.fd_;
    other.fd_ = -1;
#endif
  }

  void *base_ = nullptr;
  size_t length_ = 0;
#ifdef _WIN32
  HANDLE fileHandle_ = INVALID_HANDLE_VALUE;
  HANDLE mappingHandle_ = nullptr;
#else
  int fd_ = -1;
#endif
};

uint64_t parseByteBudget(const char *text, uint64_t fallback) {
  if (text == nullptr || *text == '\0')
    return fallback;
  std::string s(text);
  s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char ch) {
            return std::isspace(ch) != 0;
          }),
          s.end());
  if (s.empty())
    return fallback;

  char suffix = '\0';
  if (!std::isdigit(static_cast<unsigned char>(s.back()))) {
    suffix = static_cast<char>(std::tolower(static_cast<unsigned char>(s.back())));
    s.pop_back();
  }
  if (s.empty())
    return fallback;

  char *end = nullptr;
  const double value = std::strtod(s.c_str(), &end);
  if (end == s.c_str() || value < 0.0)
    return fallback;

  double scale = 1.0;
  switch (suffix) {
  case 'k':
    scale = 1024.0;
    break;
  case 'm':
    scale = 1024.0 * 1024.0;
    break;
  case 'g':
    scale = 1024.0 * 1024.0 * 1024.0;
    break;
  case 't':
    scale = 1024.0 * 1024.0 * 1024.0 * 1024.0;
    break;
  default:
    break;
  }
  const double bytes = value * scale;
  if (bytes >= static_cast<double>(std::numeric_limits<uint64_t>::max()))
    return std::numeric_limits<uint64_t>::max();
  return static_cast<uint64_t>(bytes);
}

CompactChunkCSRInfo readChunkHeader(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("Cannot open compact chunk CSR: " + path.string());

  uint32_t magic = 0;
  uint32_t version = 0;
  int32_t rows = 0;
  int32_t angleOffset = 0;
  int32_t angleCount = 0;
  uint64_t nnz = 0;
  readScalar(in, magic, "chunk magic");
  readScalar(in, version, "chunk version");
  readScalar(in, rows, "chunk rows");
  readScalar(in, angleOffset, "chunk angle_offset");
  readScalar(in, angleCount, "chunk angle_count");
  readScalar(in, nnz, "chunk nnz");

  if (magic != kChunkCSRMagic) {
    throw std::runtime_error("Invalid chunk CSR magic in " + path.string());
  }
  if (version != kChunkCSRVersion1 && version != kChunkCSRVersion2) {
    throw std::runtime_error("Unsupported chunk CSR version in " + path.string());
  }
  if (rows <= 0 || angleOffset < 0 || angleCount <= 0) {
    throw std::runtime_error("Invalid chunk CSR dimensions in " + path.string());
  }
  if (nnz > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    std::ostringstream oss;
    oss << "Chunk " << path.string() << " has nnz=" << nnz
        << ", which exceeds the current per-chunk uint32 row_ptr limit.";
    throw std::runtime_error(oss.str());
  }

  const uint64_t rows64 = static_cast<uint64_t>(rows);
  const uint64_t expectedSize =
      (version == kChunkCSRVersion2)
          ? kChunkHeaderBytes + 8ull * (rows64 + 1ull) + 4ull * nnz +
                4ull * rows64 + 2ull * nnz
          : kChunkHeaderBytes + 8ull * (rows64 + 1ull) + 4ull * nnz +
                4ull * nnz;
  const uint64_t actualSize = fileSize(path);
  if (actualSize != expectedSize) {
    std::ostringstream oss;
    oss << "Chunk CSR size mismatch for " << path.string() << ": file="
        << actualSize << ", expected=" << expectedSize;
    throw std::runtime_error(oss.str());
  }

  CompactChunkCSRInfo info;
  info.path = path;
  info.version = version;
  info.rows = rows;
  info.angleOffset = angleOffset;
  info.angleCount = angleCount;
  info.nnz = nnz;
  info.fileSizeBytes = actualSize;
  return info;
}

std::vector<CompactChunkCSRInfo>
discoverChunkInfos(const std::filesystem::path &directory) {
  if (!std::filesystem::is_directory(directory)) {
    throw std::runtime_error("Compact chunk CSR directory not found: " +
                             directory.string());
  }

  std::vector<CompactChunkCSRInfo> infos;
  for (const auto &entry : std::filesystem::directory_iterator(directory)) {
    if (entry.is_regular_file() && isChunkCSRFile(entry.path())) {
      infos.push_back(readChunkHeader(entry.path()));
    }
  }
  std::sort(infos.begin(), infos.end(), [](const auto &a, const auto &b) {
    if (a.angleOffset != b.angleOffset)
      return a.angleOffset < b.angleOffset;
    return a.path.filename().string() < b.path.filename().string();
  });
  if (infos.empty()) {
    throw std::runtime_error("No chunk_*.csrb files found in " +
                             directory.string());
  }
  return infos;
}

void loadChunkRowPtr32(const CompactChunkCSRInfo &info, std::istream &in,
                       std::vector<uint32_t> &rowPtr) {
  rowPtr.resize(static_cast<size_t>(info.rows) + 1u);
  std::vector<int64_t> row64(static_cast<size_t>(info.rows) + 1u);
  readExact(in, row64.data(), row64.size(), "chunk row_ptr");
  if (row64.front() != 0 || row64.back() != static_cast<int64_t>(info.nnz)) {
    throw std::runtime_error("Invalid chunk row_ptr endpoints in " +
                             info.path.string());
  }
  for (size_t i = 0; i < row64.size(); ++i) {
    if (row64[i] < 0 ||
        static_cast<uint64_t>(row64[i]) >
            static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      throw std::runtime_error("Chunk row_ptr exceeds uint32 range in " +
                               info.path.string());
    }
    if (i > 0 && row64[i] < row64[i - 1]) {
      throw std::runtime_error("Chunk row_ptr is not monotonic in " +
                               info.path.string());
    }
    rowPtr[i] = static_cast<uint32_t>(row64[i]);
  }
}

void validateChunkColIdxRange(const CompactChunkCSRInfo &info,
                              const uint32_t *colIdx, uint64_t nnz, int cols) {
  const auto mode = chunkColIdxValidationMode();
  if (mode == ChunkColIdxValidationMode::Off) {
    warnChunkValidationOffOnce();
    return;
  }

  auto validateAt = [&](uint64_t i) {
    if (colIdx[i] >= static_cast<uint32_t>(cols)) {
      std::ostringstream oss;
      oss << "Chunk col_idx out of range in " << info.path.string()
          << " at nnz index " << i << ": " << colIdx[i] << " >= " << cols;
      throw std::runtime_error(oss.str());
    }
  };

  if (mode == ChunkColIdxValidationMode::Strict || nnz <= 4096) {
    for (uint64_t i = 0; i < nnz; ++i)
      validateAt(i);
    return;
  }

  constexpr uint64_t kSampleHeadCount = 1024;
  constexpr uint64_t kSampleTargetCount = 100000;
  for (uint64_t i = 0; i < kSampleHeadCount; ++i)
    validateAt(i);
  const uint64_t remaining = nnz - kSampleHeadCount;
  const uint64_t stride =
      std::max<uint64_t>(1, remaining / kSampleTargetCount);
  for (uint64_t i = kSampleHeadCount; i < nnz; i += stride)
    validateAt(i);
  validateAt(nnz - 1);
}

CompactChunkCSRData loadChunkDataCopied(const CompactChunkCSRInfo &info,
                                        int cols) {
  std::ifstream in(info.path, std::ios::binary);
  if (!in)
    throw std::runtime_error("Cannot open compact chunk CSR: " +
                             info.path.string());

  in.seekg(28, std::ios::beg);
  if (!in)
    throw std::runtime_error("Failed to seek chunk payload: " +
                             info.path.string());

  CompactChunkCSRData data;
  data.info = info;
  data.cols = cols;
  loadChunkRowPtr32(info, in, data.rowPtr);

  data.colIdx.resize(static_cast<size_t>(info.nnz));
  readExact(in, data.colIdx.data(), data.colIdx.size(), "chunk col_idx");

  if (info.version == kChunkCSRVersion2) {
    data.rowScales.resize(static_cast<size_t>(info.rows));
    data.quantizedValues.resize(static_cast<size_t>(info.nnz));
    readExact(in, data.rowScales.data(), data.rowScales.size(),
              "chunk row_scales");
    readExact(in, data.quantizedValues.data(), data.quantizedValues.size(),
              "chunk quantized values");
  } else {
    data.values.resize(static_cast<size_t>(info.nnz));
    readExact(in, data.values.data(), data.values.size(), "chunk values");
  }

  validateChunkColIdxRange(info, data.colIdx.data(), info.nnz, cols);

  return data;
}

CompactChunkCSRData loadChunkDataMapped(const CompactChunkCSRInfo &info,
                                        int cols) {
  auto mapping = std::make_shared<ReadOnlyChunkPayloadMap>();
  mapping->open(info.path);

  if (mapping->length() != static_cast<size_t>(info.fileSizeBytes)) {
    throw std::runtime_error("Mapped chunk size mismatch for " +
                             info.path.string());
  }

  const size_t row64Bytes =
      (static_cast<size_t>(info.rows) + 1u) * sizeof(int64_t);
  const size_t colIdxBytes = static_cast<size_t>(info.nnz) * sizeof(uint32_t);
  const std::byte *base = mapping->bytes();
  const std::byte *row64Base = base + kChunkHeaderBytes;
  const std::byte *colIdxBase = row64Base + row64Bytes;

  CompactChunkCSRData data;
  data.info = info;
  data.cols = cols;
  {
    std::string rowBytesText(reinterpret_cast<const char *>(row64Base), row64Bytes);
    std::istringstream rowIn(rowBytesText, std::ios::in | std::ios::binary);
    loadChunkRowPtr32(info, rowIn, data.rowPtr);
  }
  data.mappedPayloadOwner = mapping;
  data.mappedColIdx = reinterpret_cast<const uint32_t *>(colIdxBase);

  if (info.version == kChunkCSRVersion2) {
    const size_t rowScalesBytes = static_cast<size_t>(info.rows) * sizeof(float);
    const size_t valuesBytes = static_cast<size_t>(info.nnz) * sizeof(uint16_t);
    const std::byte *rowScalesBase = colIdxBase + colIdxBytes;
    const std::byte *valuesBase = rowScalesBase + rowScalesBytes;
    data.mappedRowScales = reinterpret_cast<const float *>(rowScalesBase);
    data.mappedQuantizedValues =
        reinterpret_cast<const uint16_t *>(valuesBase);
    data.mappedPayloadBytes = static_cast<uint64_t>(
        colIdxBytes + rowScalesBytes + valuesBytes);
  } else {
    const size_t valuesBytes = static_cast<size_t>(info.nnz) * sizeof(float);
    const std::byte *valuesBase = colIdxBase + colIdxBytes;
    data.mappedValues = reinterpret_cast<const float *>(valuesBase);
    data.mappedPayloadBytes =
        static_cast<uint64_t>(colIdxBytes + valuesBytes);
  }

  validateChunkColIdxRange(info, data.mappedColIdx, info.nnz, cols);
  return data;
}

CompactChunkCSRData loadChunkData(const CompactChunkCSRInfo &info, int cols) {
  if (!shouldMapChunkPayload(info)) {
    return loadChunkDataCopied(info, cols);
  }
  try {
    return loadChunkDataMapped(info, cols);
  } catch (const std::exception &ex) {
    std::cerr << "[WARN] compact-chunk mmap failed for " << info.path
              << ": " << ex.what()
              << "; falling back to full copy load\n";
    return loadChunkDataCopied(info, cols);
  }
}

} // namespace

CSRView CompactChunkCSRData::view() const {
  CSRView out;
  out.rows = info.rows;
  out.cols = cols;
  out.row_ptr = rowPtr.data();
  out.col_idx = colIdxData();
  out.val = valuesData();
  out.nnz = static_cast<uint64_t>(valuesCount());
  return out;
}

CSRQuantizedView CompactChunkCSRData::quantizedView() const {
  CSRQuantizedView out;
  out.rows = info.rows;
  out.cols = cols;
  out.row_ptr = rowPtr.data();
  out.col_idx = colIdxData();
  out.quantized_values = quantizedValuesData();
  out.row_scales = rowScalesData();
  out.nnz = static_cast<uint64_t>(quantizedValuesCount());
  return out;
}

bool CompactChunkCSRData::isQuantized() const {
  return info.version == kChunkCSRVersion2;
}

uint64_t CompactChunkCSRData::residentBytes() const {
  const uint64_t rowPtrBytes =
      static_cast<uint64_t>(rowPtr.size()) * sizeof(uint32_t);
  if (mappedPayloadOwner) {
    return rowPtrBytes + mappedPayloadBytes;
  }
  if (isQuantized()) {
    return rowPtrBytes +
           static_cast<uint64_t>(colIdx.size()) * sizeof(uint32_t) +
           static_cast<uint64_t>(rowScales.size()) * sizeof(float) +
           static_cast<uint64_t>(quantizedValues.size()) * sizeof(uint16_t);
  }
  return rowPtrBytes +
         static_cast<uint64_t>(colIdx.size()) * sizeof(uint32_t) +
         static_cast<uint64_t>(values.size()) * sizeof(float);
}

const uint32_t *CompactChunkCSRData::colIdxData() const {
  return mappedColIdx != nullptr ? mappedColIdx : colIdx.data();
}

const float *CompactChunkCSRData::valuesData() const {
  return mappedValues != nullptr ? mappedValues : values.data();
}

const uint16_t *CompactChunkCSRData::quantizedValuesData() const {
  return mappedQuantizedValues != nullptr ? mappedQuantizedValues
                                          : quantizedValues.data();
}

const float *CompactChunkCSRData::rowScalesData() const {
  return mappedRowScales != nullptr ? mappedRowScales : rowScales.data();
}

size_t CompactChunkCSRData::colIdxCount() const {
  return mappedColIdx != nullptr ? static_cast<size_t>(info.nnz) : colIdx.size();
}

size_t CompactChunkCSRData::valuesCount() const {
  return mappedValues != nullptr ? static_cast<size_t>(info.nnz) : values.size();
}

size_t CompactChunkCSRData::quantizedValuesCount() const {
  return mappedQuantizedValues != nullptr ? static_cast<size_t>(info.nnz)
                                          : quantizedValues.size();
}

size_t CompactChunkCSRData::rowScalesCount() const {
  return mappedRowScales != nullptr ? static_cast<size_t>(info.rows)
                                    : rowScales.size();
}

CompactChunkCSRDirectory::CompactChunkCSRDirectory(
    std::filesystem::path directory, std::vector<CompactChunkCSRInfo> chunks,
    int cols, uint64_t cacheBudgetBytes)
    : directory_(std::move(directory)), chunks_(std::move(chunks)), cols_(cols),
      cacheBudgetBytes_(cacheBudgetBytes), cache_(chunks_.size()) {
  for (const auto &chunk : chunks_) {
    totalRows_ += static_cast<uint64_t>(chunk.rows);
    totalNnz_ += chunk.nnz;
  }
}

std::shared_ptr<const CompactChunkCSRData>
CompactChunkCSRDirectory::loadChunk(size_t index) const {
  if (index >= chunks_.size()) {
    throw std::runtime_error("Compact chunk index out of range.");
  }
  if (index < cache_.size() && cache_[index]) {
    return cache_[index];
  }

  auto loaded =
      std::make_shared<CompactChunkCSRData>(loadChunkData(chunks_[index], cols_));
  const uint64_t bytes = loaded->residentBytes();
  if (index < cache_.size() &&
      (cacheBudgetBytes_ == std::numeric_limits<uint64_t>::max() ||
       cachedBytes_ + bytes <= cacheBudgetBytes_)) {
    cache_[index] = loaded;
    cachedBytes_ += bytes;
  }
  return loaded;
}

uint64_t defaultCompactChunkHostCacheBudgetBytes() {
  return parseByteBudget(std::getenv("SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES"),
                         spect::cuda::kCompactChunkHostCacheDefaultBytes);
}

CompactChunkCSRDirectory openCompactChunkCSRDirectory(
    const std::string &directory, const CompactSymmetryMetadata &meta,
    uint64_t cacheBudgetBytes) {
  const std::filesystem::path dir(directory);
  std::vector<CompactChunkCSRInfo> infos = discoverChunkInfos(dir);

  int expectedOffset = 0;
  uint64_t rows = 0;
  for (const auto &info : infos) {
    if (info.angleOffset != expectedOffset) {
      std::ostringstream oss;
      oss << "Compact chunk angle range has a gap or overlap at "
          << info.path.filename().string() << ": angle_offset="
          << info.angleOffset << ", expected=" << expectedOffset;
      throw std::runtime_error(oss.str());
    }
    const int64_t expectedRows =
        static_cast<int64_t>(info.angleCount) *
        static_cast<int64_t>(meta.compactRaysPerAngle);
    if (info.rows != expectedRows) {
      std::ostringstream oss;
      oss << "Chunk " << info.path.filename().string() << " rows="
          << info.rows << ", expected angle_count*compactRaysPerAngle="
          << expectedRows;
      throw std::runtime_error(oss.str());
    }
    expectedOffset += info.angleCount;
    rows += static_cast<uint64_t>(info.rows);
  }

  if (expectedOffset != meta.compactNumAngles) {
    std::ostringstream oss;
    oss << "Compact chunks cover " << expectedOffset
        << " compact angles, metadata expects " << meta.compactNumAngles;
    throw std::runtime_error(oss.str());
  }
  if (rows != static_cast<uint64_t>(meta.compactRows())) {
    std::ostringstream oss;
    oss << "Compact chunk row sum=" << rows
        << ", metadata compact rows=" << meta.compactRows();
    throw std::runtime_error(oss.str());
  }

  return CompactChunkCSRDirectory(dir, std::move(infos), meta.totalVoxels,
                                  cacheBudgetBytes);
}

} // namespace spect::io
