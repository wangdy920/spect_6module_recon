#include "io/CompactChunkCSR.h"

#include "cuda/CudaLaunchConfig.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
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
#include <unistd.h>
#endif

namespace spect::io {

namespace {

constexpr uint32_t kChunkCSRMagic = 0x42525343u;
constexpr uint32_t kChunkCSRVersion1 = 1u;
constexpr uint64_t kChunkHeaderBytes = 28ull;

// CSRQ header layout is frozen in tools/convert_csrb_to_csrq.py.
constexpr uint32_t kChunkCSRQMagic = 0x51525343u;
constexpr uint32_t kChunkCSRQVersion1 = 1u;
constexpr uint64_t kChunkCSRQHeaderBytes = 48ull;
constexpr uint32_t kChunkCSRQValueEncodingU16Sqrt = 1u;

uint64_t csrqValuesPadBytes(uint64_t nnz) { return (4ull - (2ull * nnz) % 4ull) % 4ull; }

uint64_t csrqExpectedFileSize(uint64_t rows, uint64_t nnz, uint64_t runCount) {
  return kChunkCSRQHeaderBytes + 12ull * rows + 8ull + 2ull * nnz +
         csrqValuesPadBytes(nnz) + 4ull * runCount;
}

enum class ChunkColIdxValidationMode { Off, Sample, Strict };

struct SystemMemoryInfo {
  uint64_t totalBytes = 0;
  uint64_t availableBytes = 0;
};

SystemMemoryInfo querySystemMemoryInfo() {
#ifdef _WIN32
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  if (!GlobalMemoryStatusEx(&status))
    return {};
  return {static_cast<uint64_t>(status.ullTotalPhys),
          static_cast<uint64_t>(status.ullAvailPhys)};
#else
#if defined(_SC_PHYS_PAGES) && defined(_SC_AVPHYS_PAGES) && \
    defined(_SC_PAGESIZE)
  const long totalPages = sysconf(_SC_PHYS_PAGES);
  const long availablePages = sysconf(_SC_AVPHYS_PAGES);
  const long pageBytes = sysconf(_SC_PAGESIZE);
  if (totalPages <= 0 || availablePages <= 0 || pageBytes <= 0)
    return {};
  return {static_cast<uint64_t>(totalPages) *
              static_cast<uint64_t>(pageBytes),
          static_cast<uint64_t>(availablePages) *
              static_cast<uint64_t>(pageBytes)};
#else
  return {};
#endif
#endif
}

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

bool isChunkCSRQFile(const std::filesystem::path &path) {
  const std::string name = lowerString(path.filename().string());
  return startsWith(name, "chunk_") && endsWith(name, ".csrq");
}

uint64_t fileSize(const std::filesystem::path &path) {
  const auto size = std::filesystem::file_size(path);
  return static_cast<uint64_t>(size);
}

uint64_t parseByteBudget(const char *text, uint64_t fallback);


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
    std::cerr << "[NOTE] SPECT_CHUNK_VALIDATE=off disables compact chunk "
                 "col_idx range validation; user assumes matrix integrity.\n";
    warned = true;
  }
}


uint64_t parseByteBudget(const char *text, uint64_t fallback) {
  if (text == nullptr || *text == '\0')
    return fallback;
  std::string s(text);
  s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char ch) {
            return std::isspace(ch) != 0;
          }),
          s.end());
  if (s.empty() || lowerString(s) == "auto")
    return fallback;

  char suffix = '\0';
  if (std::isalpha(static_cast<unsigned char>(s.back()))) {
    suffix = static_cast<char>(std::tolower(static_cast<unsigned char>(s.back())));
    s.pop_back();
  }
  char *end = nullptr;
  const double value = std::strtod(s.c_str(), &end);
  if (end == s.c_str() || *end != '\0' || !std::isfinite(value) || value < 0.0 ||
      (suffix != '\0' && suffix != 'k' && suffix != 'm' && suffix != 'g' && suffix != 't'))
    throw std::runtime_error("Invalid SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES: " +
                             std::string(text) + "; expected auto or nonnegative bytes with optional K/M/G/T.");

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
  if (version != kChunkCSRVersion1) {
    throw std::runtime_error(
        "Unsupported chunk CSR version in " + path.string() +
        "; canonical v4 matrices require CSRB version 1 float values.");
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
      kChunkHeaderBytes + 8ull * (rows64 + 1ull) + 4ull * nnz + 4ull * nnz;
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

CompactChunkCSRInfo readChunkCSRQHeader(const std::filesystem::path &path,
                                       int expectedCols) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("Cannot open compact chunk CSRQ: " + path.string());

  uint32_t magic = 0;
  uint32_t version = 0;
  int32_t rows = 0;
  int32_t angleOffset = 0;
  int32_t angleCount = 0;
  int32_t cols = 0;
  uint64_t nnz = 0;
  uint64_t runCount = 0;
  uint32_t runLenBits = 0;
  uint32_t valueEncoding = 0;
  readScalar(in, magic, "csrq magic");
  readScalar(in, version, "csrq version");
  readScalar(in, rows, "csrq rows");
  readScalar(in, angleOffset, "csrq angle_offset");
  readScalar(in, angleCount, "csrq angle_count");
  readScalar(in, cols, "csrq cols");
  readScalar(in, nnz, "csrq nnz");
  readScalar(in, runCount, "csrq run_count");
  readScalar(in, runLenBits, "csrq run_len_bits");
  readScalar(in, valueEncoding, "csrq value_encoding");

  if (magic != kChunkCSRQMagic)
    throw std::runtime_error("Invalid chunk CSRQ magic in " + path.string());
  if (version != kChunkCSRQVersion1)
    throw std::runtime_error("Unsupported chunk CSRQ version in " + path.string());
  if (runLenBits != static_cast<uint32_t>(kCompactRunLenBits) ||
      valueEncoding != kChunkCSRQValueEncodingU16Sqrt) {
    std::ostringstream oss;
    oss << "Unsupported CSRQ encoding in " << path.string()
        << ": run_len_bits=" << runLenBits << " value_encoding=" << valueEncoding
        << " (expected " << kCompactRunLenBits << " / "
        << kChunkCSRQValueEncodingU16Sqrt << ")";
    throw std::runtime_error(oss.str());
  }
  if (rows <= 0 || angleOffset < 0 || angleCount <= 0 || cols <= 0)
    throw std::runtime_error("Invalid chunk CSRQ dimensions in " + path.string());
  if (cols != expectedCols)
    throw std::runtime_error("Chunk CSRQ cols=" + std::to_string(cols) +
                             " differ from metadata totalVoxels=" +
                             std::to_string(expectedCols) + " in " + path.string());
  if (nnz > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
      runCount > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::runtime_error("Chunk CSRQ nnz/run_count exceed the u32 row "
                             "pointer range in " + path.string());
  }
  const uint64_t expectedSize =
      csrqExpectedFileSize(static_cast<uint64_t>(rows), nnz, runCount);
  const uint64_t actualSize = fileSize(path);
  if (actualSize != expectedSize) {
    std::ostringstream oss;
    oss << "Chunk CSRQ size mismatch for " << path.string() << ": file="
        << actualSize << ", expected=" << expectedSize;
    throw std::runtime_error(oss.str());
  }

  CompactChunkCSRInfo info;
  info.path = path;
  info.format = CompactChunkFormat::CsrqRunU16;
  info.version = version;
  info.rows = rows;
  info.angleOffset = angleOffset;
  info.angleCount = angleCount;
  info.nnz = nnz;
  info.runCount = runCount;
  info.fileSizeBytes = actualSize;
  return info;
}

std::vector<CompactChunkCSRInfo>
discoverChunkInfos(const std::filesystem::path &directory, int cols) {
  if (!std::filesystem::is_directory(directory)) {
    throw std::runtime_error("Compact chunk CSR directory not found: " +
                             directory.string());
  }

  std::vector<CompactChunkCSRInfo> infos;
  size_t csrbCount = 0;
  size_t csrqCount = 0;
  for (const auto &entry : std::filesystem::directory_iterator(directory)) {
    if (!entry.is_regular_file())
      continue;
    if (isChunkCSRFile(entry.path())) {
      infos.push_back(readChunkHeader(entry.path()));
      ++csrbCount;
    } else if (isChunkCSRQFile(entry.path())) {
      infos.push_back(readChunkCSRQHeader(entry.path(), cols));
      ++csrqCount;
    }
  }
  if (csrbCount > 0 && csrqCount > 0) {
    throw std::runtime_error(
        "Compact chunk directory mixes chunk_*.csrb and chunk_*.csrq files: " +
        directory.string());
  }
  std::sort(infos.begin(), infos.end(), [](const auto &a, const auto &b) {
    if (a.angleOffset != b.angleOffset)
      return a.angleOffset < b.angleOffset;
    return a.path.filename().string() < b.path.filename().string();
  });
  if (infos.empty()) {
    throw std::runtime_error("No chunk_*.csrb or chunk_*.csrq files found in " +
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

void validateChunkValues(const CompactChunkCSRInfo &info, const float *values,
                         uint64_t count, uint64_t sourceOffset = 0) {
  // Keep the valid-data pass vectorizable; scan again only to locate an error.
  int valid = 1;
  for (uint64_t i = 0; i < count; ++i)
    valid &= (values[i] >= 0.0f) &
             (values[i] <= std::numeric_limits<float>::max());
  if (valid)
    return;
  for (uint64_t i = 0; i < count; ++i) {
    if (!std::isfinite(values[i]) || values[i] < 0.0f) {
      throw std::runtime_error("Invalid nonnegative finite chunk value in " +
                               info.path.string() + " at nnz index " +
                               std::to_string(sourceOffset + i));
    }
  }
}

CompactChunkCSRData loadChunkDataCopied(const CompactChunkCSRInfo &info,
                                        int cols) {
  std::ifstream in(info.path, std::ios::binary);
  if (!in)
    throw std::runtime_error("Cannot open compact chunk CSR: " +
                             info.path.string());

  in.seekg(kChunkHeaderBytes, std::ios::beg);

  CompactChunkCSRData data;
  data.info = info;
  data.cols = cols;
  loadChunkRowPtr32(info, in, data.rowPtr);

  data.colIdx.resize(static_cast<size_t>(info.nnz));
  readExact(in, data.colIdx.data(), data.colIdx.size(), "chunk col_idx");

  data.values.resize(static_cast<size_t>(info.nnz));
  readExact(in, data.values.data(), data.values.size(), "chunk values");

  validateChunkColIdxRange(info, data.colIdx.data(), info.nnz, cols);
  validateChunkValues(info, data.values.data(), info.nnz);

  return data;
}

CompactChunkCSRData loadChunkDataSubset(const CompactChunkCSRInfo &info,
                                       int cols, const std::vector<int> &angles,
                                       int raysPerAngle) {
  std::ifstream in(info.path, std::ios::binary);
  if (!in)
    throw std::runtime_error("Cannot open compact chunk CSR: " + info.path.string());
  in.seekg(kChunkHeaderBytes, std::ios::beg);
  std::vector<uint32_t> sourceRows;
  loadChunkRowPtr32(info, in, sourceRows);

  CompactChunkCSRData data;
  data.info = info;
  data.cols = cols;
  data.rowPtr.resize(sourceRows.size());
  std::vector<std::pair<uint32_t, uint32_t>> ranges;
  uint32_t packedNnz = 0;
  for (int angle = 0; angle < info.angleCount; ++angle) {
    const size_t firstRow = static_cast<size_t>(angle) * raysPerAngle;
    const size_t lastRow = firstRow + raysPerAngle;
    const bool selected = std::binary_search(angles.begin(), angles.end(),
                                             info.angleOffset + angle);
    for (size_t row = firstRow; row < lastRow; ++row) {
      data.rowPtr[row] = packedNnz;
      if (selected) packedNnz += sourceRows[row + 1] - sourceRows[row];
    }
    if (selected && sourceRows[lastRow] != sourceRows[firstRow]) {
      if (!ranges.empty() && ranges.back().second == sourceRows[firstRow])
        ranges.back().second = sourceRows[lastRow];
      else
        ranges.emplace_back(sourceRows[firstRow], sourceRows[lastRow]);
    }
  }
  data.rowPtr.back() = packedNnz;
  data.info.nnz = packedNnz;
  data.colIdx.resize(packedNnz);
  data.values.resize(packedNnz);
  const uint64_t columnsOffset = kChunkHeaderBytes + 8ull * sourceRows.size();
  const uint64_t valuesOffset = columnsOffset + 4ull * info.nnz;
  size_t destination = 0;
  for (const auto &range : ranges) {
    const size_t count = range.second - range.first;
    in.seekg(static_cast<std::streamoff>(columnsOffset + 4ull * range.first));
    readExact(in, data.colIdx.data() + destination, count, "subset col_idx");
    in.seekg(static_cast<std::streamoff>(valuesOffset + 4ull * range.first));
    readExact(in, data.values.data() + destination, count, "subset values");
    validateChunkValues(info, data.values.data() + destination, count, range.first);
    destination += count;
  }
  validateChunkColIdxRange(info, data.colIdx.data(), packedNnz, cols);
  // Preserve the original angle positions with empty unselected rows. This
  // private backing storage is only exposed through the existing subsetView.
  return data;
}


void loadChunkPtr32(const CompactChunkCSRInfo &info, std::istream &in,
                    uint64_t expectedEnd, const char *label,
                    std::vector<uint32_t> &ptr) {
  ptr.resize(static_cast<size_t>(info.rows) + 1u);
  readExact(in, ptr.data(), ptr.size(), label);
  if (ptr.front() != 0 || ptr.back() != expectedEnd) {
    throw std::runtime_error(std::string("Invalid ") + label +
                             " endpoints in " + info.path.string());
  }
  for (size_t i = 1; i < ptr.size(); ++i) {
    if (ptr[i] < ptr[i - 1]) {
      throw std::runtime_error(std::string("Non-monotonic ") + label + " in " +
                               info.path.string());
    }
  }
}

// Full structural validation: every run stays inside [0, cols), run lengths
// sum to the row nnz, and runs never overlap or go backwards within a row.
// This walks ~nnz/15 words once per load; the cost is negligible next to
// reading the chunk.
void validateChunkRuns(const CompactChunkCSRInfo &info,
                       const CompactChunkCSRData &data) {
  const uint32_t cols = static_cast<uint32_t>(data.cols);
  for (int row = 0; row < info.rows; ++row) {
    const size_t r = static_cast<size_t>(row);
    uint64_t total = 0;
    uint64_t prevEnd = 0;
    for (uint32_t j = data.rowRunPtr[r]; j < data.rowRunPtr[r + 1]; ++j) {
      const uint32_t word = data.runs[j];
      const uint32_t start = word >> kCompactRunLenBits;
      const uint32_t len = (word & (kCompactRunMaxLen - 1u)) + 1u;
      if (start < prevEnd || static_cast<uint64_t>(start) + len > cols) {
        std::ostringstream oss;
        oss << "Chunk CSRQ run out of range/order in " << info.path.string()
            << " row " << row << " run " << j << ": start=" << start
            << " len=" << len << " cols=" << cols;
        throw std::runtime_error(oss.str());
      }
      prevEnd = static_cast<uint64_t>(start) + len;
      total += len;
    }
    if (total != static_cast<uint64_t>(data.rowPtr[r + 1] - data.rowPtr[r])) {
      std::ostringstream oss;
      oss << "Chunk CSRQ run lengths (" << total << ") do not match row nnz ("
          << (data.rowPtr[r + 1] - data.rowPtr[r]) << ") in "
          << info.path.string() << " row " << row;
      throw std::runtime_error(oss.str());
    }
  }
}

CompactChunkCSRData loadChunkDataCSRQ(const CompactChunkCSRInfo &info,
                                      int cols) {
  std::ifstream in(info.path, std::ios::binary);
  if (!in)
    throw std::runtime_error("Cannot open compact chunk CSRQ: " +
                             info.path.string());
  in.seekg(kChunkCSRQHeaderBytes, std::ios::beg);

  CompactChunkCSRData data;
  data.info = info;
  data.cols = cols;
  loadChunkPtr32(info, in, info.nnz, "csrq row_nnz_ptr", data.rowPtr);
  loadChunkPtr32(info, in, info.runCount, "csrq row_run_ptr", data.rowRunPtr);
  data.rowMax.resize(static_cast<size_t>(info.rows));
  readExact(in, data.rowMax.data(), data.rowMax.size(), "csrq row_max");
  data.valuesQ.resize(static_cast<size_t>(info.nnz));
  readExact(in, data.valuesQ.data(), data.valuesQ.size(), "csrq values");
  in.seekg(static_cast<std::streamoff>(csrqValuesPadBytes(info.nnz)), std::ios::cur);
  data.runs.resize(static_cast<size_t>(info.runCount));
  readExact(in, data.runs.data(), data.runs.size(), "csrq runs");
  for (size_t r = 0; r < data.rowMax.size(); ++r) {
    if (!(data.rowMax[r] >= 0.0f) || !std::isfinite(data.rowMax[r])) {
      throw std::runtime_error("Chunk CSRQ row_max is negative or non-finite in " +
                               info.path.string());
    }
  }
  validateChunkRuns(info, data);
  return data;
}

CompactChunkCSRData loadChunkData(const CompactChunkCSRInfo &info, int cols) {
  const auto timeBefore = std::filesystem::last_write_time(info.path);
  auto data = info.format == CompactChunkFormat::CsrqRunU16
      ? loadChunkDataCSRQ(info, cols)
      : loadChunkDataCopied(info, cols);
  if (std::filesystem::file_size(info.path) != info.fileSizeBytes ||
      std::filesystem::last_write_time(info.path) != timeBefore)
    throw std::runtime_error("Matrix file changed while loading: " + info.path.string());
  return data;
}

} // namespace

CSRView CompactChunkCSRData::view() const {
  if (info.format != CompactChunkFormat::CsrbFloat32) {
    throw std::runtime_error(
        "CSRQ compact chunk has no float CSR view: " + info.path.string());
  }
  CSRView out;
  out.sourceOwner = weak_from_this().lock();
  out.rows = info.rows;
  out.cols = cols;
  out.row_ptr = rowPtr.data();
  out.col_idx = colIdx.data();
  out.val = values.data();
  out.nnz = static_cast<uint64_t>(values.size());
  return out;
}

CSRSubsetView CompactChunkCSRData::subsetView(
    const std::vector<int> &compactAngles, int raysPerAngle,
    uint64_t tag) const {
  const CSRView full = view();
  if (raysPerAngle <= 0 || info.rows % raysPerAngle != 0) {
    throw std::runtime_error(
        "compact chunk rows are not a whole number of angles: " +
        info.path.string());
  }
  CSRSubsetView out;
  out.sourceOwner = full.sourceOwner;
  out.cols = cols;
  out.rowsPerBlock = raysPerAngle;
  out.sourceRowPtr = full.row_ptr;
  out.tag = tag;
  const int firstAngle = info.angleOffset;
  const int lastAngle = info.angleOffset + info.angleCount;
  out.row_ptr.push_back(0u);
  int previousAngle = -1;
  for (const int angle : compactAngles) {
    if (angle < firstAngle || angle >= lastAngle) {
      continue;
    }
    if (angle <= previousAngle) {
      throw std::runtime_error(
          "compact subset angles must be strictly ascending: " +
          info.path.string());
    }
    const size_t rowBase =
        static_cast<size_t>(angle - firstAngle) * static_cast<size_t>(raysPerAngle);
    const uint32_t nnzStart = full.row_ptr[rowBase];
    const uint32_t nnzEnd = full.row_ptr[rowBase + static_cast<size_t>(raysPerAngle)];
    const uint32_t base = out.row_ptr.back();
    for (int r = 1; r <= raysPerAngle; ++r) {
      out.row_ptr.push_back(base + (full.row_ptr[rowBase + static_cast<size_t>(r)] -
                                    nnzStart));
    }
    // Adjacent angles are adjacent in the chunk arrays: extend the segment
    // so a whole chunk (or a run of angles) is one copy.
    if (previousAngle >= 0 && angle == previousAngle + 1) {
      out.segments.back().nnz += nnzEnd - nnzStart;
    } else {
      CSRSegment segment;
      segment.col_idx = full.nnz == 0 ? full.col_idx : full.col_idx + nnzStart;
      segment.val = full.nnz == 0 ? full.val : full.val + nnzStart;
      segment.nnz = nnzEnd - nnzStart;
      out.segments.push_back(segment);
    }
    out.blockAngles.push_back(angle);
    out.nnz += nnzEnd - nnzStart;
    out.rows += raysPerAngle;
    previousAngle = angle;
  }
  return out;
}

CompactRunU16View CompactChunkCSRData::runView() const {
  if (info.format != CompactChunkFormat::CsrqRunU16) {
    throw std::runtime_error(
        "CSRB compact chunk has no run-encoded view: " + info.path.string());
  }
  CompactRunU16View out;
  out.rows = info.rows;
  out.cols = cols;
  out.nnz = info.nnz;
  out.runCount = info.runCount;
  out.rowNnzPtr = rowPtr.data();
  out.rowRunPtr = rowRunPtr.data();
  out.rowMax = rowMax.data();
  out.runs = runs.data();
  out.valuesQ = valuesQ.data();
  return out;
}


uint64_t CompactChunkCSRData::residentBytes() const {
  const uint64_t rowPtrBytes =
      static_cast<uint64_t>(rowPtr.size()) * sizeof(uint32_t);
  return rowPtrBytes +
         static_cast<uint64_t>(colIdx.size()) * sizeof(uint32_t) +
         static_cast<uint64_t>(values.size()) * sizeof(float) +
         static_cast<uint64_t>(rowRunPtr.size()) * sizeof(uint32_t) +
         static_cast<uint64_t>(rowMax.size()) * sizeof(float) +
         static_cast<uint64_t>(runs.size()) * sizeof(uint32_t) +
         static_cast<uint64_t>(valuesQ.size()) * sizeof(uint16_t);
}

CompactChunkCSRDirectory::CompactChunkCSRDirectory(
    std::filesystem::path directory, std::vector<CompactChunkCSRInfo> chunks,
    int cols, uint64_t cacheBudgetBytes, int branchCount)
    : directory_(std::move(directory)), chunks_(std::move(chunks)), cols_(cols),
      branchCount_(branchCount), cacheBudgetBytes_(cacheBudgetBytes), cache_(chunks_.size()),
      liveChunks_(chunks_.size()), liveSubsets_(chunks_.size()) {
  cacheMutexes_.reserve(chunks_.size());
  for (size_t i = 0; i < chunks_.size(); ++i)
    cacheMutexes_.push_back(std::make_shared<std::mutex>());
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
  std::lock_guard<std::mutex> chunkLock(*cacheMutexes_[index]);
  if (auto live = liveChunks_[index].lock()) {
    std::lock_guard<std::mutex> budgetLock(*cacheBudgetMutex_);
    readStats_.hostReuseBytes += live->residentBytes();
    return live;
  }

  const auto readStart = std::chrono::steady_clock::now();
  auto loaded =
      std::make_shared<CompactChunkCSRData>(loadChunkData(chunks_[index], cols_));
  const uint64_t bytes = loaded->residentBytes();
  if (index < cache_.size()) {
    std::lock_guard<std::mutex> budgetLock(*cacheBudgetMutex_);
    ++readStats_.payloadLoads;
    readStats_.loaderWorkSeconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - readStart).count();
    readStats_.payloadReadBytes += chunks_[index].fileSizeBytes -
        (chunks_[index].format == CompactChunkFormat::CsrbFloat32 ? 28u : 48u);
    if (cacheBudgetBytes_ == std::numeric_limits<uint64_t>::max() ||
        cachedBytes_ + bytes <= cacheBudgetBytes_) {
      cache_[index] = loaded;
      cachedBytes_ += bytes;
    }
  }
  liveChunks_[index] = loaded;
  return loaded;
}

void CompactChunkCSRDirectory::appendCacheInputs(
    std::ostream &out, const std::string &prefix) const {
  out << prefix << "chunk_count=" << chunks_.size() << '\n';
  out << prefix << "chunk_total_rows=" << totalRows_ << '\n';
  out << prefix << "chunk_total_nnz=" << totalNnz_ << '\n';
  for (size_t i = 0; i < chunks_.size(); ++i) {
    const auto &chunk = chunks_[i];
    const auto path = std::filesystem::canonical(chunk.path);
    const auto size = std::filesystem::file_size(path);
    if (size != chunk.fileSizeBytes)
      throw std::runtime_error("Matrix file size changed after opening: " + path.string());
    const auto key = prefix + "chunk_" + std::to_string(i);
    out << key << "_path=" << path.generic_string() << '\n';
    out << key << "_mtime="
        << std::filesystem::last_write_time(path).time_since_epoch().count() << '\n';
    out << key << "_format=" << (chunk.format == CompactChunkFormat::CsrbFloat32 ? "csrb" : "csrq") << '\n';
    out << key << "_version=" << chunk.version << '\n';
    out << key << "_rows=" << chunk.rows << '\n';
    out << key << "_angle_offset=" << chunk.angleOffset << '\n';
    out << key << "_angle_count=" << chunk.angleCount << '\n';
    out << key << "_nnz=" << chunk.nnz << '\n';
    out << key << "_run_count=" << chunk.runCount << '\n';
    out << key << "_file_size_bytes=" << size << '\n';
  }
}

CSRSubsetView CompactChunkCSRDirectory::loadSubset(
    size_t index, const std::vector<int> &compactAngles,
    int raysPerAngle, uint64_t tag) const {
  const auto &info = chunks_.at(index);
  if (info.format != CompactChunkFormat::CsrbFloat32 || raysPerAngle <= 0 ||
      static_cast<int64_t>(info.angleCount) * raysPerAngle != info.rows)
    throw std::runtime_error("Invalid CSRB subset layout: " + info.path.string());
  if (std::adjacent_find(compactAngles.begin(), compactAngles.end(),
                        [](int a, int b) { return a >= b; }) != compactAngles.end())
    throw std::runtime_error("compact subset angles must be strictly ascending.");
  const auto first = std::lower_bound(compactAngles.begin(), compactAngles.end(),
                                      info.angleOffset);
  if (first == compactAngles.end() ||
      *first >= static_cast<int64_t>(info.angleOffset) + info.angleCount) {
    std::lock_guard<std::mutex> budgetLock(*cacheBudgetMutex_);
    ++readStats_.emptySlicesSkipped;
    return {};
  }
  std::unique_lock<std::mutex> chunkLock(*cacheMutexes_[index]);
  auto owner = liveChunks_[index].lock();
  if (!owner && liveSubsets_[index].first == compactAngles)
    owner = liveSubsets_[index].second.lock();
  if (owner) {
    std::lock_guard<std::mutex> budgetLock(*cacheBudgetMutex_);
    readStats_.hostReuseBytes += owner->residentBytes();
  } else {
    const auto last = std::lower_bound(first, compactAngles.end(),
                                      info.angleOffset + info.angleCount);
    bool loadWhole = std::distance(first, last) == info.angleCount;
    {
      std::lock_guard<std::mutex> budgetLock(*cacheBudgetMutex_);
      const uint64_t bytes = 4ull * (static_cast<uint64_t>(info.rows) + 1) + 8ull * info.nnz;
      loadWhole = loadWhole || bytes <= cacheBudgetBytes_ - cachedBytes_;
    }
    if (loadWhole) {
      chunkLock.unlock();
      owner = loadChunk(index);
    } else {
      const auto readStart = std::chrono::steady_clock::now();
      owner = std::make_shared<CompactChunkCSRData>(
          loadChunkDataSubset(info, cols_, compactAngles, raysPerAngle));
      liveSubsets_[index] = {compactAngles, owner};
      std::lock_guard<std::mutex> budgetLock(*cacheBudgetMutex_);
      ++readStats_.payloadLoads;
      readStats_.loaderWorkSeconds += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - readStart).count();
      readStats_.payloadReadBytes += 8ull * (static_cast<uint64_t>(info.rows) + 1) +
                                    8ull * owner->info.nnz;
    }
  }
  auto slice = owner->subsetView(compactAngles, raysPerAngle, tag);
  slice.sourceOwner = std::move(owner);
  return slice;
}

CompactChunkCSRDirectory::ReadStats CompactChunkCSRDirectory::readStats() const {
  std::lock_guard<std::mutex> budgetLock(*cacheBudgetMutex_);
  return readStats_;
}

uint64_t recommendCompactChunkHostCacheBudgetBytes(
    uint64_t matrixBytes, uint64_t totalMemoryBytes,
    uint64_t availableMemoryBytes) {
  if (matrixBytes == 0)
    return 0;
  if (totalMemoryBytes == 0 || availableMemoryBytes == 0) {
    return std::min(matrixBytes,
                    spect::cuda::kCompactChunkHostCacheFallbackBytes);
  }
  const uint64_t reserve = std::max(
      spect::cuda::kCompactChunkHostCacheMinReserveBytes,
      totalMemoryBytes / 4);
  const uint64_t totalCap =
      totalMemoryBytes > reserve ? totalMemoryBytes - reserve : 0;
  const uint64_t availableCap =
      availableMemoryBytes > spect::cuda::kCompactChunkHostCacheMinReserveBytes
          ? availableMemoryBytes -
                spect::cuda::kCompactChunkHostCacheMinReserveBytes
          : 0;
  return std::min(matrixBytes, std::min(totalCap, availableCap));
}

uint64_t defaultCompactChunkHostCacheBudgetBytes(uint64_t matrixBytes) {
  const SystemMemoryInfo memory = querySystemMemoryInfo();
  return parseByteBudget(std::getenv("SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES"),
      recommendCompactChunkHostCacheBudgetBytes(
          matrixBytes, memory.totalBytes, memory.availableBytes));
}

namespace {

CompactChunkCSRDirectory makeCompactChunkCSRDirectory(
    const std::filesystem::path &dir, const CompactSymmetryMetadata &meta,
    std::vector<CompactChunkCSRInfo> infos,
    uint64_t cacheBudgetBytes) {
  if (meta.profile != CompactSymmetryProfile::Rot90 &&
      meta.profile != CompactSymmetryProfile::Rot90Z)
    throw std::runtime_error("Unsupported compact matrix symmetry profile.");

  int64_t expectedOffset = 0;
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
                                  cacheBudgetBytes, meta.xySymmetryCount() * meta.zBranchCount());
}

} // namespace

CompactChunkCSRDirectory openCompactChunkCSRDirectory(
    const std::string &directory, const CompactSymmetryMetadata &meta) {
  const std::filesystem::path dir(directory);
  auto infos = discoverChunkInfos(dir, meta.totalVoxels);
  uint64_t matrixBytes = 0;
  for (const auto &info : infos) {
    matrixBytes = info.fileSizeBytes >
                          std::numeric_limits<uint64_t>::max() - matrixBytes
                      ? std::numeric_limits<uint64_t>::max()
                      : matrixBytes + info.fileSizeBytes;
  }
  // Keep the CSRQ host cache opt-in; residency and streaming own its uploads.
  const bool csrq = !infos.empty() &&
                    infos.front().format == CompactChunkFormat::CsrqRunU16;
  const uint64_t cacheBudget =
      csrq ? parseByteBudget(std::getenv("SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES"), 0)
           : defaultCompactChunkHostCacheBudgetBytes(matrixBytes);
  return makeCompactChunkCSRDirectory(dir, meta, std::move(infos), cacheBudget);
}

CompactChunkCSRDirectory openCompactChunkCSRDirectory(
    const std::string &directory, const CompactSymmetryMetadata &meta,
    uint64_t cacheBudgetBytes) {
  const std::filesystem::path dir(directory);
  return makeCompactChunkCSRDirectory(dir, meta,
      discoverChunkInfos(dir, meta.totalVoxels), cacheBudgetBytes);
}

} // namespace spect::io
